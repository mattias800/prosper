// Owned x2 observations become compute inputs only after full original-program alias admission.
#include "gpu/execute/graphics_nested_wide_reader.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace prosper::gpu {
bool raw_snapshot_storage_extent(const ShaderResource& r, uint64_t& bytes) {
    if (r.cls != ResourceClass::StorageImage || !r.gpu_addr || !r.width || !r.height ||
        r.width > 16384 || r.height > 16384 || r.depth != 1 || r.img_dim != 1 ||
        r.sample_count != 1 || r.declared_mip_levels != 1 || r.in_mip_tail || r.metadata_addr ||
        r.compression_enabled || r.write_compress_enabled || !tile_mode_is_tiled(r.tile_mode))
        return false;
    const uint64_t bpe = uint64_t(data_format_bytes(r.format)) * r.num_components;
    if (!bpe || bpe > 16 || uint64_t(r.width) * r.height * bpe != r.size) return false;
    bytes = tiled_surface_bytes(r.width, r.height, r.tile_mode, 0, static_cast<uint32_t>(bpe));
    return bytes && bytes <= UINT32_MAX && r.gpu_addr <= UINT64_MAX - bytes;
}
RawSnapshotWritePlan raw_snapshot_write_plan(const std::vector<Rdna2Inst>& decoded,
                                             const std::vector<RawNestedWideChain>& chains) {
    RawSnapshotWritePlan plan;
    for (const auto& chain : chains) {
        const auto parent = std::find_if(decoded.begin(), decoded.end(),
                                         [&](const auto& in) { return in.pc == chain.parent_pc; });
        if (parent == decoded.end() || parent->src[0].kind != OperandKind::SGPR ||
            parent->src[0].value < 0)
            plan.complete = false;
        else
            plan.required_user_words = std::max(plan.required_user_words,
                                                static_cast<uint32_t>(parent->src[0].value) + 2u);
    }
    for (const auto& in : decoded) {
        if (in.fmt == Rdna2Format::Unknown || !in.len_dwords) plan.complete = false;
        if (in.fmt == Rdna2Format::MIMG) plan.descriptor_pcs.push_back(in.pc);
        if (!rdna2_instruction_may_write_memory(in)) continue;
        if (in.fmt != Rdna2Format::MIMG || in.opcode != 8u)
            plan.complete = false;
        else
            plan.storage_write_pcs.push_back(in.pc);
    }
    return plan;
}
namespace {
const ShaderResource* unique_pc(const ShaderResourceTable& table, uint32_t pc) {
    const ShaderResource* found = nullptr;
    for (const auto& r : table.resources)
        if (r.fetch_pc == pc) {
            if (found) return nullptr;
            found = &r;
        }
    return found;
}
} // namespace
bool GraphicsNestedWideReader::publish_compute_x2(ShaderResourceTable& table,
                                                  const RawSnapshotWritePlan& plan,
                                                  const std::vector<SrtUse>& uses) const {
    if (diagnostic_) diagnostic_->published = false;
    const auto reject = [&](RawSnapshotGate gate, uint32_t pc = UINT32_MAX, uint64_t address = 0,
                            uint32_t bytes = 0, uint64_t other = 0) {
        if (diagnostic_) diagnostic_->refuse(gate, pc, address, bytes, other);
        return false;
    };
    table.owned_nested_snapshot_requirements.assign(widths_.begin(), widths_.end());
    if (!allowed_ || !lease_ || observations_.size() != widths_.size())
        return reject(RawSnapshotGate::MissingObservation);
    if (!plan.complete) return reject(RawSnapshotGate::WritePlanIncomplete);
    std::vector<std::pair<uint64_t, uint64_t>> writes;
    std::vector<std::pair<uint64_t, uint64_t>> sources;
    for (const auto& [pc, observation] : observations_) {
        if (widths_.at(pc) != 8u) return reject(RawSnapshotGate::ProbeShape, pc);
        sources.emplace_back(observation.address, 8u);
    }
    for (uint32_t pc : plan.descriptor_pcs) {
        const SrtUse* descriptor = nullptr;
        for (const auto& u : uses)
            if (u.kind == 0 && u.use_pc == pc) {
                if (descriptor) return reject(RawSnapshotGate::DescriptorMissingOrAmbiguous, pc);
                descriptor = &u;
            }
        if (!descriptor) return reject(RawSnapshotGate::DescriptorMissingOrAmbiguous, pc);
        const uint64_t address = descriptor->descriptor_source_addr;
        if (!address || address > UINT64_MAX - 32u)
            return reject(RawSnapshotGate::DescriptorSourceInvalid, pc, address, 32u);
        if (!guest_memory_direct_range_fault_safe(*lease_, address, 32u))
            return reject(RawSnapshotGate::DescriptorFaultUnsafe, pc, address, 32u);
        if (!guest_readable(address, 32u))
            return reject(RawSnapshotGate::DescriptorUnreadable, pc, address, 32u);
        if (!graphics_raw_source_is_guest_current(*lease_, address, 32u))
            return reject(RawSnapshotGate::DescriptorCurrentUnproved, pc, address, 32u);
        if (std::memcmp(reinterpret_cast<const void*>(uintptr_t(address)), descriptor->t8.data(),
                        32u) != 0)
            return reject(RawSnapshotGate::DescriptorChanged, pc, address, 32u);
        sources.emplace_back(descriptor->descriptor_source_addr, 32u);
    }
    for (uint32_t pc : plan.storage_write_pcs) {
        const auto* output = unique_pc(table, pc);
        uint64_t extent = 0;
        if (!output) return reject(RawSnapshotGate::OutputMissing, pc);
        if (!raw_snapshot_storage_extent(*output, extent))
            return reject(RawSnapshotGate::OutputExtentUnproved, pc, output->gpu_addr);
        if (!guest_memory_direct_range_fault_safe(*lease_, output->gpu_addr, extent))
            return reject(RawSnapshotGate::OutputFaultUnsafe, pc, output->gpu_addr);
        writes.emplace_back(output->gpu_addr, extent);
    }
    for (const auto& source : sources)
        for (const auto& write : writes)
            if (guest_memory_topology_relation(source.first, source.second, write.first,
                                               write.second) !=
                GuestMemoryTopologyRelation::Disjoint)
                return reject(RawSnapshotGate::PhysicalNotDisjoint, UINT32_MAX, source.first,
                              static_cast<uint32_t>(source.second), write.first);
    return publish(table); // exact parent equation and transactional same-owner publication
}
} // namespace prosper::gpu
