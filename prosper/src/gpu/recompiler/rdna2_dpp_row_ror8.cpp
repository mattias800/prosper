#include "gpu/recompiler/rdna2_dpp_row_ror8.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
// All workers, including completed waves, publish event/EXEC metadata and reach both barriers.
bool emit_portable_compute_dpp_row_ror8_phase(
    SpirvCompute& b, const ComputeDppRowRor8PhaseVariables& variables, uint32_t value_base,
    uint32_t metadata_base, const std::set<int>& destinations, const std::map<int, uint32_t>& vv,
    const std::map<std::pair<int, int>, uint32_t>& lv,
    const std::map<std::pair<int, int>, uint32_t>& lmv, bool floating_add) {
    const uint32_t zero = b.uconst(0), no = b.bfalse();
    const uint32_t dpp_pending = b.load_function(b.t_bool, variables.pending);
    const uint32_t dpp_active = b.load_function(b.t_bool, variables.active);
    const uint32_t dpp_src0 = b.load_function(b.t_u32, variables.source0);
    const uint32_t dpp_src1 = b.load_function(b.t_u32, variables.source1);
    const uint32_t dpp_operation = b.load_function(b.t_u32, variables.operation);
    const uint32_t dpp_event = b.load_function(b.t_u32, variables.event);
    b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(value_base), b.linear_localid), dpp_src0);
    const uint32_t dpp_metadata =
        b.sel(dpp_pending,
              b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, dpp_event, b.uconst(1)),
                     b.sel(dpp_active, b.uconst(1), zero)),
              zero);
    b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(metadata_base), b.linear_localid), dpp_metadata);
    b.barrier();

    // XOR 8 exchanges the two eight-lane halves without crossing an architectural DPP16 row.
    // This uses guest linear-local order, not the implementation-defined Vulkan subgroup lane ID.
    const uint32_t dpp_rotated_index = b.ibin(Op_BitwiseXor, b.linear_localid, b.uconst(8));
    const uint32_t dpp_source_in_bounds =
        b.ucmp(Op_ULessThan, dpp_rotated_index, b.uconst(b.local_count));
    // A partial final DPP16 row has no invocation to initialize the rotated slot. Address this
    // lane's initialized placeholder and let FI=0's validity gate supply zero for the missing peer.
    const uint32_t dpp_source_index =
        b.sel(dpp_source_in_bounds, dpp_rotated_index, b.linear_localid);
    const uint32_t dpp_rotated =
        b.cfg_scratch_load(b.ibin(Op_IAdd, b.uconst(value_base), dpp_source_index));
    const uint32_t dpp_source_metadata =
        b.cfg_scratch_load(b.ibin(Op_IAdd, b.uconst(metadata_base), dpp_source_index));
    const uint32_t dpp_source_event =
        b.ibin(Op_ShiftRightLogical, dpp_source_metadata, b.uconst(1));
    const uint32_t dpp_source_active =
        b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, dpp_source_metadata, b.uconst(1)), zero);
    const uint32_t dpp_valid_source =
        b.land(dpp_source_in_bounds,
               b.land(dpp_source_active, b.ucmp(Op_IEqual, dpp_source_event, dpp_event)));
    // FI=0 requires an EXEC-active source. BOUND_CTRL=1 supplies zero when that source is invalid,
    // but the active destination still writes the operation's result. MOV uses the bounded source;
    // MIN/MAX combine it with the destination lane's unpermuted SRC1.
    const uint32_t dpp_bounded = b.sel(dpp_valid_source, dpp_rotated, zero);
    uint32_t dpp_result = dpp_bounded;
    dpp_result = b.sel(
        b.ucmp(Op_IEqual, dpp_operation, b.uconst(static_cast<uint32_t>(DppRowRor8Op::MinF32))),
        b.fext2(Glsl_NMin, dpp_bounded, dpp_src1), dpp_result);
    dpp_result = b.sel(
        b.ucmp(Op_IEqual, dpp_operation, b.uconst(static_cast<uint32_t>(DppRowRor8Op::MaxF32))),
        b.fext2(Glsl_NMax, dpp_bounded, dpp_src1), dpp_result);
    if (floating_add)
        dpp_result = b.sel(
            b.ucmp(Op_IEqual, dpp_operation, b.uconst(static_cast<uint32_t>(DppRowRor8Op::AddF32))),
            b.fbin(Op_FAdd, dpp_bounded, dpp_src1), dpp_result);
    const uint32_t dpp_write = b.land(dpp_pending, dpp_active);
    const uint32_t dpp_dst = b.load_function(b.t_u32, variables.destination);
    for (int reg : destinations) {
        const auto kv = vv.find(reg);
        if (kv == vv.end()) return false;
        const uint32_t selected =
            b.land(dpp_write, b.ucmp(Op_IEqual, dpp_dst, b.uconst(static_cast<uint32_t>(reg))));
        const uint32_t old = b.load_function(b.t_u32, kv->second);
        b.store_function(kv->second, b.sel(selected, dpp_result, old));
    }
    // A physical destination definition invalidates scalar lane aliases even when EXEC suppresses
    // this invocation's data write, matching predicate_write and the existing DPP add phase.
    for (const auto& kv : lv) {
        if (!destinations.contains(kv.first.first)) continue;
        const uint32_t selected =
            b.land(dpp_pending,
                   b.ucmp(Op_IEqual, dpp_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        if (!destinations.contains(kv.first.first)) continue;
        const uint32_t selected =
            b.land(dpp_pending,
                   b.ucmp(Op_IEqual, dpp_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    return true;
}
}   // namespace prosper::gpu
