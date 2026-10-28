#include "gpu/recompiler/rdna2_cfg_readfirstlane.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu {
CfgReadfirstlaneResult emit_cfg_readfirstlane_gather(SpirvCompute& b,
                                                     const CfgReadfirstlanePhase& phase) {
    const auto [initial_active, padded_lanes, mbcnt_lane, mbcnt_wave_index, wave_result_base, zero,
                first_pending_var, first_event_var, first_exec_var, first_source_var,
                first_dst_var] = phase;
    {
        // clang-format off
        // AMD RDNA2 ISA 70648 section 12.8: select the lowest EXEC bit; EXEC=0 selects lane 0.
        // Access and scalar destination writes ignore EXEC. Every invocation publishes its raw
        // value and event/EXEC metadata, including inactive lanes and ended waves. All barriers
        // are in this common dispatcher phase, never beneath guest-wave control flow.
        // pending/tag are reset on EACH common dispatcher iteration and the source is sampled
        // anew at its current guest visit. This barrier phase is the dynamic event instance;
        // the static tag identifies a site, never a memoized value across a backedge.
        const uint32_t pending = b.load_function(b.t_bool, first_pending_var);
        const uint32_t tag = b.load_function(b.t_u32, first_event_var);
        const uint32_t encoded = b.sel(pending,
            b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, tag, b.uconst(1)),
                   b.sel(b.load_function(b.t_bool, first_exec_var), b.uconst(1), zero)), zero);
        b.cfg_scratch_store(b.linear_localid, b.load_function(b.t_u32, first_source_var));
        b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(padded_lanes), b.linear_localid), encoded);
        b.barrier();
        const bool partial_wave = initial_active || padded_lanes != b.local_count;
        const uint32_t is_leader = partial_wave ? b.ucmp(Op_IEqual, mbcnt_lane, zero)
            : b.land(pending, b.ucmp(Op_IEqual, mbcnt_lane, zero));
        const uint32_t leader = b.id(), assembled = b.id();
        b.emit_selmerge(assembled);
        b.emit_condbranch(is_leader, leader, assembled);
        b.emit_label(leader);
        const uint32_t base = b.ibin(Op_ShiftLeftLogical, mbcnt_wave_index,
                                     b.uconst(b.wave_size == 32 ? 5u : 6u));
        const auto lane_metadata = [&](uint32_t lane) {
            const uint32_t index = b.ibin(Op_IAdd, base, b.uconst(lane));
            if (padded_lanes == b.local_count)
                return b.cfg_scratch_load(b.ibin(Op_IAdd, b.uconst(padded_lanes), index));
            const uint32_t valid = b.ucmp(Op_ULessThan, index, b.uconst(b.local_count));
            const uint32_t safe_index = b.sel(valid, index, zero);
            const uint32_t metadata = b.cfg_scratch_load(b.ibin(Op_IAdd,
                b.uconst(padded_lanes), safe_index));
            return b.sel(valid, metadata, zero);
        };
        // ACTIVE excludes padded invocations from guest instructions, not from host barriers.
        // The physical wave leader may therefore have no event: recover its wave's pending tag
        // from actual participants. Nonexistent physical lanes never supply metadata.
        uint32_t service_tag = tag;
        if (partial_wave) {
            service_tag = zero;
            for (uint32_t lane = 0; lane < b.wave_size; ++lane) {
                const uint32_t candidate = b.ibin(Op_ShiftRightLogical,
                    lane_metadata(lane), b.uconst(1));
                service_tag = b.sel(b.ucmp(Op_IEqual, service_tag, zero), candidate, service_tag);
            }
        }
        uint32_t first = b.uconst(UINT32_MAX);
        for (uint32_t lane = 0; lane < b.wave_size; ++lane) {
            const uint32_t metadata = lane_metadata(lane);
            const uint32_t include = b.land(
                b.ucmp(Op_IEqual, b.ibin(Op_ShiftRightLogical, metadata, b.uconst(1)), service_tag),
                b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, metadata, b.uconst(1)), zero));
            first = b.sel(b.land(include, b.ucmp(Op_IEqual, first, b.uconst(UINT32_MAX))),
                          b.uconst(lane), first);
        }
        const uint32_t selected_lane = b.sel(b.ucmp(Op_IEqual, first, b.uconst(UINT32_MAX)),
                                             zero, first);
        b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index),
                            b.cfg_scratch_load(b.ibin(Op_IAdd, base, selected_lane)));
        b.emit_branch(assembled);
        b.emit_label(assembled);
        b.barrier();
        const uint32_t result = b.cfg_scratch_load(
            b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index));
        const uint32_t dst = b.load_function(b.t_u32, first_dst_var);
        // clang-format on
        return {pending, dst, result};
    }
}
} // namespace prosper::gpu
