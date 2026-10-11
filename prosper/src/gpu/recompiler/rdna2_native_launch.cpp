#include "gpu/recompiler/rdna2_native_launch.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <cstdint>

namespace prosper::gpu {
bool native_compute_has_complete_entry(const SpirvCompute& b) {
    // CONFIDENCE: HIGH. A required Vulkan subgroup width does not prove guest entry EXEC.
    // PARTIAL_TG_EN/raw presence is checked by the shared producing-register contract first.
    // Complete local groups (e.g. 65 workers) are not necessarily complete logical Wave64s.
    const auto& launch = b.guest_compute_launch;
    if (!b.is_compute || b.is_fragment || b.is_vertex || b.is_fragment_packet() ||
        b.guest_stage != GuestShaderStage::Compute ||
        b.physical_domain != PhysicalExecutionDomain::Workgroup || b.ngg_workgroup_shell ||
        b.ngg_one_lane || b.wave_size != 64 || b.native_subgroup_size != 64 ||
        launch.wave_size != b.wave_size || launch.local != b.guest_local_size ||
        !compute_dispatch_has_complete_groups(launch))
        return false;
    const uint64_t count = uint64_t{launch.local[0]} * launch.local[1] * launch.local[2];
    return count >= 64 && count % 64 == 0 && count == b.local_count;
}

bool saved_mask_physical_launch_matches_original(const SpirvCompute& b) {
    if (b.is_fragment != b.saved_mask_original_fragment_entry) return false;
    if (b.is_fragment) return true;
    // Captured before any physical consumer and retained after Bool-copy revocation. A
    // compute-shaped builder relabelled into another entry domain cannot reuse these facts.
    return !b.native_compute_physical_entry_expired && native_compute_has_complete_entry(b) &&
           native_vcc_copy_execution_domain_matches(b) &&
           b.guest_compute_launch == b.saved_mask_original_compute_launch;
}

bool native_uniform_mov_entry_sources(const RegState* initial, const Rdna2Inst& in) {
    if (!initial || initial->scc_merge_placeholder) return false;
    // Memory/descriptor footprints need a separate proof. This route covers original launch
    // scalar ALU/control only; it does not infer all-path DATA from a textual prior writer.
    if (in.fmt == Rdna2Format::SMEM || in.fmt == Rdna2Format::MUBUF ||
        in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::MIMG)
        return false;
    for (uint32_t k = 0; k < in.n_src; ++k) {
        if (in.src[k].kind != OperandKind::SGPR) continue;
        const auto width = scalar_alu_source_words(in, k);
        if (!width) return false;
        for (uint32_t half = 0; half < width; ++half) {
            const int word = in.src[k].value + static_cast<int>(half);
            uint32_t value = 0;
            if (word < 0 || word > 105 || sreg_word_may_be_fabricated(*initial, word) ||
                initial->sreg_memory_pattern.contains(word) ||
                initial->sreg_entry_m0.contains(word) ||
                !native_logical_data_word(*initial, word, value) || !value ||
                initial->lane_local_scalars.contains(value))
                return false;
        }
    }
    return true;
}

bool native_uniform_exec_mov_has_full_participation(const SpirvCompute& b, const Rdna2Inst& in) {
    if (!saved_mask_physical_launch_matches_original(b) ||
        !b.native_uniform_exec_mov_pcs.contains(in.pc) ||
        in.pc >= b.saved_mask_original_words.size())
        return false;
    auto original = rdna2_decode_one(b.saved_mask_original_words.data() + in.pc,
                                     b.saved_mask_original_words.size() - in.pc);
    original.pc = in.pc;
    return !in.synthetic_terminator && in == original;
}
} // namespace prosper::gpu
