#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/graphics_nested_wide_reader.hpp"

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
#include <utility>

namespace prosper::gpu {
namespace {
struct Range { uint64_t address, bytes; };

bool exact_storage_write_range(const ShaderResource& r, Range& out) {
    uint64_t bytes = 0;
    if (!raw_snapshot_storage_extent(r, bytes)) return false;
    out = {r.gpu_addr, bytes};
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
static GraphicsRawSourceAuthorityFn g_graphics_raw_source_authority;
void set_graphics_raw_source_authority(GraphicsRawSourceAuthorityFn fn) {
    g_graphics_raw_source_authority = std::move(fn);
}
bool graphics_raw_source_is_guest_current(const GuestMappingLease& lease,
                                          uint64_t address, uint32_t bytes) {
    return g_graphics_raw_source_authority && g_graphics_raw_source_authority(lease, address, bytes);
}
static GraphicsProducerStatusFn g_graphics_producer_status;
void set_graphics_producer_status_query(GraphicsProducerStatusFn fn) {
    g_graphics_producer_status = std::move(fn);
}
GraphicsProducerStatus graphics_producer_status() {
    return g_graphics_producer_status ? g_graphics_producer_status() : GraphicsProducerStatus{};
}

GraphicsNestedWideReader::GraphicsNestedWideReader(std::vector<RawNestedWideChain> chains,
                                                   const GraphicsRawSnapshotContext* context,
                                                   const GuestMappingLease* borrowed_lease)
    : chains_(std::move(chains)) {
    allowed_ = context && context->producers_complete && !chains_.empty();
    if (context) output_allocations_ = context->output_allocations;
    for (const auto& chain : chains_) {
        for (const auto [pc, bytes] : {std::pair{chain.parent_pc, chain.parent_bytes},
                                      std::pair{chain.child_pc, chain.child_bytes}}) {
            const auto [it, inserted] = widths_.emplace(pc, bytes);
            if (!inserted && it->second != bytes) allowed_ = false;
        }
    }
    if (allowed_) {
        if (!borrowed_lease) owned_lease_ = std::make_unique<GuestMappingLease>();
        lease_ = borrowed_lease ? borrowed_lease : owned_lease_.get();
    }
}

bool GraphicsNestedWideReader::probe(FoldProbe kind, uint32_t pc, uint64_t address,
                                     uint32_t bytes) {
    const auto width = widths_.find(pc);
    if (width == widths_.end()) return guest_readable(address, bytes);
    // Raw pointers may not be repaired by Base48/Base40 fallback or a readable host VMA.
    if (!allowed_ || !lease_ || kind != FoldProbe::Raw || bytes != width->second ||
        address <= 0x10000u || (address & 3u) || address > UINT64_MAX - bytes) return false;
    const auto prior = observations_.find(pc);
    if (prior != observations_.end())
        return prior->second.address == address && prior->second.bytes->size() == bytes;
    for (const auto& chain : chains_) {
        if (chain.child_pc != pc) continue;
        const auto parent = observations_.find(chain.parent_pc);
        if (parent == observations_.end() || address < chain.child_offset) return false;
        uint64_t pointer = 0;
        std::memcpy(&pointer, parent->second.bytes->data(), sizeof(pointer));
        if (pointer != address - chain.child_offset) return false;
    }
    if (!guest_memory_direct_range_fault_safe(*lease_, address, bytes) ||
        guest_memory_direct_allocation_relation(*lease_, address, bytes, address, bytes) !=
            GuestMemoryTopologyRelation::Overlap ||
        !guest_readable(address, bytes) ||
        !graphics_raw_source_is_guest_current(*lease_, address, bytes)) return false;
    for (const auto [output, physical_bytes] : output_allocations_)
        if (output && guest_memory_direct_allocation_relation(*lease_, address, bytes,
                output, physical_bytes) != GuestMemoryTopologyRelation::Disjoint) return false;
    auto owner = std::make_shared<std::vector<uint8_t>>(bytes);
    std::memcpy(owner->data(), reinterpret_cast<const void*>(uintptr_t(address)), bytes);
    observations_.emplace(pc, Observation{address, std::move(owner)});
    return true;
}

uint32_t GraphicsNestedWideReader::word(uint32_t pc, uint64_t address) {
    if (!owns_raw_wide(pc)) return *reinterpret_cast<const uint32_t*>(uintptr_t(address));
    const auto it = observations_.find(pc);
    if (it == observations_.end() || address < it->second.address ||
        address - it->second.address > it->second.bytes->size() - sizeof(uint32_t))
        throw std::runtime_error("nested raw fold word lacks exact observation");
    uint32_t value = 0;
    std::memcpy(&value, it->second.bytes->data() + address - it->second.address, sizeof(value));
    return value;
}

void GraphicsNestedWideReader::prefix(uint32_t pc, uint64_t address, void* destination,
                                      uint32_t bytes) {
    if (!owns_raw_wide(pc)) {
        std::memcpy(destination, reinterpret_cast<const void*>(uintptr_t(address)), bytes);
        return;
    }
    const auto it = observations_.find(pc);
    if (it == observations_.end() || address != it->second.address ||
        bytes != it->second.bytes->size())
        throw std::runtime_error("nested raw fold prefix lacks exact observation");
    std::memcpy(destination, it->second.bytes->data(), bytes);
}

bool GraphicsNestedWideReader::publish(ShaderResourceTable& table) const {
    table.owned_nested_snapshot_requirements.assign(widths_.begin(), widths_.end());
    if (!allowed_ || observations_.size() != widths_.size()) return false;
    for (const auto& chain : chains_) {
        const auto& parent = observations_.at(chain.parent_pc);
        const auto& child = observations_.at(chain.child_pc);
        uint64_t pointer = 0;
        std::memcpy(&pointer, parent.bytes->data(), sizeof(pointer));
        if (child.address < chain.child_offset || pointer != child.address - chain.child_offset)
            return false;
    }
    // Existing exact-PC resources may only be the identical owned parent from the same fold.
    // A descriptor, duplicate, or conflicting binding cannot be replaced by a later valid owner.
    for (const auto& [pc, observation] : observations_) {
        size_t matches = 0;
        for (const auto& resource : table.resources) {
            if (resource.fetch_pc != pc) continue;
            if (++matches > 1 || !valid_owned_raw_snapshot_resource(resource, widths_.at(pc)) ||
                resource.gpu_addr != observation.address ||
                std::memcmp(resource.host_data, observation.bytes->data(), widths_.at(pc)) != 0)
                return false;
        }
    }
    for (const auto& [pc, observation] : observations_) {
        ShaderResource resource;
        resource.cls = ResourceClass::ConstantBuffer;
        resource.format = DataFormat::Uint32;
        resource.num_components = 1;
        resource.gpu_addr = observation.address;
        resource.size = widths_.at(pc);
        resource.fetch_pc = pc;
        resource.owned_nested_snapshot_bytes = resource.size;
        resource.host_data = observation.bytes->data();
        resource.host_data_size = observation.bytes->size();
        const auto existing = std::find_if(table.resources.begin(), table.resources.end(),
            [pc](const ShaderResource& candidate) { return candidate.fetch_pc == pc; });
        if (existing == table.resources.end()) table.resources.push_back(resource);
        else {
            resource.owned_raw_snapshot_bytes = existing->owned_raw_snapshot_bytes;
            *existing = resource;
        }
        table.owned_host_data.push_back(observation.bytes);
    }
    return true;
}

} // namespace prosper::gpu
