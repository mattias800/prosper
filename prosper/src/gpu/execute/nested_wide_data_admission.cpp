#include "gpu/execute/gpu_execute.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"
#include "hle/memory/guest_memory_topology.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <set>
#include <vector>

namespace prosper::gpu {
namespace {
struct Range { uint64_t address, bytes; };

bool exact_storage_write_range(const ShaderResource& r, Range& out) {
    // One uncompressed, single-level 2D image. Its declared linear size can be smaller than
    // tiled backing; use the same physical extent as the backend's image staging/writeback.
    if (r.cls != ResourceClass::StorageImage || !r.gpu_addr || !r.width || !r.height ||
        r.width > 16384 || r.height > 16384 || r.depth != 1 || r.img_dim != 1 ||
        r.sample_count != 1 || r.declared_mip_levels != 1 || r.in_mip_tail ||
        r.metadata_addr || r.compression_enabled || r.write_compress_enabled ||
        !tile_mode_is_tiled(r.tile_mode)) return false;
    const uint64_t bpe = static_cast<uint64_t>(data_format_bytes(r.format)) *
                         r.num_components;
    const uint64_t logical = static_cast<uint64_t>(r.width) * r.height * bpe;
    if (!bpe || bpe > 16 || logical != r.size) return false;
    const size_t physical = tiled_surface_bytes(r.width, r.height, r.tile_mode, 0,
                                                static_cast<uint32_t>(bpe));
    if (!physical || physical > UINT32_MAX || r.gpu_addr > UINT64_MAX - physical) return false;
    out = {r.gpu_addr, physical};
    return true;
}

const ShaderResource* exact_buffer(const ShaderResourceTable& table, uint32_t pc) {
    const ShaderResource* r = table.by_fetch_pc(pc);
    return r && r->cls == ResourceClass::ConstantBuffer && r->fetch_pc == pc &&
                   r->gpu_addr && r->size && r->size <= (1u << 20u) && !r->host_data
        ? r : nullptr;
}
} // namespace

bool admit_compute_nested_wide_data(const GuestMappingLease* lease,
                                    const std::vector<Rdna2Inst>& decoded,
                                    const std::vector<SrtUse>& uses,
                                    ShaderResourceTable& table) {
    const auto nested = rdna2_proven_raw_nested_wide_data_loads(decoded);
    if (nested.empty()) return true;
    if (!lease) return false;
    const auto parents = rdna2_proven_raw_immediate_wide_data_loads(decoded);
    std::vector<std::pair<size_t, size_t>> chains;
    std::set<uint32_t> source_pcs;
    for (uint32_t child_pc : nested) {
        const auto child = std::find_if(decoded.begin(), decoded.end(),
                                        [child_pc](const Rdna2Inst& in) {
                                            return in.pc == child_pc;
                                        });
        if (child == decoded.end() || !exact_buffer(table, child_pc)) return false;
        size_t parent_index = decoded.size();
        const size_t child_index = static_cast<size_t>(child - decoded.begin());
        for (size_t i = 0; i < child_index; ++i)
            if (decoded[i].fmt == Rdna2Format::SMEM &&
                (decoded[i].opcode == 0x2u || decoded[i].opcode == 0x3u) &&
                decoded[i].dst.kind == OperandKind::SGPR &&
                decoded[i].dst.value == child->src[0].value &&
                std::binary_search(parents.begin(), parents.end(), decoded[i].pc))
                parent_index = i;
        if (parent_index == decoded.size() ||
            !exact_buffer(table, decoded[parent_index].pc)) return false;
        chains.emplace_back(parent_index, child_index);
        source_pcs.insert(decoded[parent_index].pc);
        source_pcs.insert(child_pc);
    }

    std::vector<Range> writes;
    for (const Rdna2Inst& in : decoded) {
        if (in.fmt == Rdna2Format::MIMG && in.opcode == 0x08u) {
            const ShaderResource* r = table.by_fetch_pc(in.pc);
            Range range{};
            if (!r || r->fetch_pc != in.pc || !exact_storage_write_range(*r, range))
                return false;
            writes.push_back(range);
            continue;
        }
        // Known read-only FLAT and comparison sample opcodes are conservatively classified as
        // possible writes by the older raw-load inventory. They cannot change a source here.
        if ((in.fmt == Rdna2Format::FLAT && in.opcode >= 0x08u &&
             in.opcode <= 0x0fu) ||
            (in.fmt == Rdna2Format::MIMG && in.opcode == 0x2fu)) continue;
        if (rdna2_may_write_guest_memory(in)) return false;
    }
    if (writes.empty()) return false;

    std::vector<Range> sources;
    for (uint32_t pc : source_pcs) {
        const ShaderResource* r = exact_buffer(table, pc);
        if (!r) return false;
        sources.push_back({r->gpu_addr, r->size});
    }
    // Every image descriptor that this shader consumes must have an exact byte source. A folded
    // but untraced descriptor could itself alias the store and change between waves.
    for (const Rdna2Inst& in : decoded) {
        if (in.fmt != Rdna2Format::MIMG) continue;
        const auto use = std::find_if(uses.begin(), uses.end(), [&](const SrtUse& candidate) {
            return candidate.kind == 0 && candidate.use_pc == in.pc;
        });
        if (use == uses.end() || !use->descriptor_source_addr ||
            use->descriptor_source_addr > UINT64_MAX - 32u ||
            !guest_memory_direct_range_fault_safe(*lease, use->descriptor_source_addr, 32u) ||
            !guest_readable(use->descriptor_source_addr, 32u) ||
            std::memcmp(reinterpret_cast<const void*>(
                            static_cast<uintptr_t>(use->descriptor_source_addr)),
                        use->t8.data(), 32u) != 0) return false;
        sources.push_back({use->descriptor_source_addr, 32u});
    }
    for (const Range& write : writes)
        if (!guest_memory_direct_range_fault_safe(*lease, write.address, write.bytes))
            return false;
    for (const Range& source : sources) {
        if (!guest_memory_direct_range_fault_safe(*lease, source.address, source.bytes))
            return false;
        for (const Range& write : writes)
            if (guest_memory_topology_relation(source.address, source.bytes,
                                               write.address, write.bytes) !=
                GuestMemoryTopologyRelation::Disjoint)
                return false;
    }

    // Stage every member of the pointer chain before publishing any host_data pointer. The
    // original fold computed child addresses from current parent bytes. Verify those exact bytes
    // again: if the parent changed between fold and staging, refuse the mixed snapshot.
    std::vector<std::pair<uint32_t, std::shared_ptr<std::vector<uint8_t>>>> snapshots;
    for (uint32_t pc : source_pcs) {
        const ShaderResource* r = exact_buffer(table, pc);
        if (!r || !guest_readable(r->gpu_addr, r->size)) return false;
        const auto* src = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(r->gpu_addr));
        snapshots.emplace_back(pc, std::make_shared<std::vector<uint8_t>>(
            src, src + r->size));
    }
    const auto snapshot_for = [&](uint32_t pc) -> const std::vector<uint8_t>* {
        const auto it = std::find_if(snapshots.begin(), snapshots.end(),
                                     [pc](const auto& entry) { return entry.first == pc; });
        return it == snapshots.end() ? nullptr : it->second.get();
    };
    for (const auto [parent_index, child_index] : chains) {
        const Rdna2Inst& parent = decoded[parent_index];
        const Rdna2Inst& child = decoded[child_index];
        const auto* bytes = snapshot_for(parent.pc);
        const ShaderResource* child_resource = exact_buffer(table, child.pc);
        if (!bytes || !child_resource ||
            static_cast<uint64_t>(parent.literal) + sizeof(uint64_t) > bytes->size())
            return false;
        uint64_t pointer = 0;
        std::memcpy(&pointer, bytes->data() + parent.literal, sizeof(pointer));
        if (pointer != child_resource->gpu_addr) return false;
    }
    for (auto& [pc, bytes] : snapshots) {
        auto* r = const_cast<ShaderResource*>(exact_buffer(table, pc));
        if (!r) return false; // no table mutation before this point
        r->host_data = bytes->data();
        r->host_data_size = bytes->size();
        if (std::binary_search(nested.begin(), nested.end(), pc))
            r->nested_raw_snapshot_admitted = true;
        table.owned_host_data.push_back(std::move(bytes));
    }
    return true;
}
} // namespace prosper::gpu
