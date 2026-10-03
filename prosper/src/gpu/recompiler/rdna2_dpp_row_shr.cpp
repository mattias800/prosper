#include "gpu/recompiler/rdna2_dpp_row_shr.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

bool is_inplace_vmax_u32_dpp_row_shr(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::VOP2 && in.opcode == 0x14 && in.has_dpp && !in.has_modifier &&
           !in.has_sdwa && !in.dpp_bound_ctrl && in.dpp_ctrl >= 0x111u && in.dpp_ctrl <= 0x11fu &&
           in.dpp_row_mask == 0xfu && in.dpp_bank_mask == 0xfu && in.n_src == 2 && !in.src_neg[0] &&
           !in.src_neg[1] && !in.src_abs[0] && !in.src_abs[1] && !in.clamp && !in.omod &&
           in.dst.kind == OperandKind::VGPR && in.src[0].kind == OperandKind::VGPR &&
           in.src[1].kind == OperandKind::VGPR && in.dst.value == in.src[0].value &&
           in.dst.value == in.src[1].value;
}

bool emit_compute_inplace_dpp_row_shr(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                                      uint32_t source, uint32_t old_destination, bool wave_uniform,
                                      uint32_t& result) {
    if (!b.is_compute) return false;
    // Retain the established ADD ladder and its generated graph. Other ADD strides belong to
    // the existing event-isolated CFG service rather than the ordinary instruction emitter.
    const bool add =
        is_inplace_vadd_nc_u32_dpp_row_shr(in) && (in.dpp_ctrl == 0x111u || in.dpp_ctrl == 0x112u ||
                                                   in.dpp_ctrl == 0x114u || in.dpp_ctrl == 0x118u);
    const bool maximum = wave_uniform && is_inplace_vmax_u32_dpp_row_shr(in);
    if (!add && !maximum) return false;

    // DPP16 ROW_SHR selects lane-N inside its 16-lane row. BC0 keeps VDST at a row edge.
    // Preserve VDST for an invalid source under this encoded BC0/FI0 contract. For an
    // in-place unsigned maximum, max(own, 0) == own as well. Destination EXEC is a
    // separate write condition applied by the caller, including when its peer is active.
    uint32_t valid_source = 0;
    const uint32_t shifted =
        b.subgroup_row_shr(source, rs.exec, in.dpp_ctrl - 0x110u, &valid_source);
    const uint32_t combined =
        add ? b.ibin(Op_IAdd, source, shifted) : b.uext2(Glsl_UMax, source, shifted);
    result = b.sel(valid_source, combined, old_destination);
    return true;
}


// Publish event/EXEC metadata for every worker, including nonpending and completed waves.
// Two barriers bound reuse of the value plane by later wave services and loop iterations.
bool emit_portable_compute_dpp_row_shr_phase(
    SpirvCompute& b, const ComputeDppRowShrPhaseVariables& variables,
    uint32_t value_base, uint32_t metadata_base, const std::set<int>& destinations,
    const std::map<int, uint32_t>& vv,
    const std::map<std::pair<int, int>, uint32_t>& lv,
    const std::map<std::pair<int, int>, uint32_t>& lmv) {
    const uint32_t zero = b.uconst(0), no = b.bfalse();
    const uint32_t dpp_pending = b.load_function(b.t_bool, variables.pending);
    const uint32_t dpp_active = b.load_function(b.t_bool, variables.active);
    const uint32_t dpp_source = b.load_function(b.t_u32, variables.source);
    const uint32_t dpp_event = b.load_function(b.t_u32, variables.event);
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(value_base), b.linear_localid), dpp_source);
    const uint32_t dpp_metadata = b.sel(
        dpp_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, dpp_event, b.uconst(1)),
               b.sel(dpp_active, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(metadata_base), b.linear_localid),
        dpp_metadata);
    b.barrier();

    const uint32_t dpp_control = b.load_function(b.t_u32, variables.amount);
    // The bounded form belongs to the compile-only NGG probe. Preserve the established GTA
    // unbounded reduction's generated graph and invalid-source write rule outside that probe.
    const uint32_t dpp_bounded = b.ngg_workgroup_export_probe
        ? b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, dpp_control, b.uconst(0x100)), zero)
        : no;
    const uint32_t dpp_amount = b.ngg_workgroup_export_probe
        ? b.ibin(Op_BitwiseAnd, dpp_control, b.uconst(0xf)) : dpp_control;
    const uint32_t dpp_row_lane = b.ibin(
        Op_BitwiseAnd, b.linear_localid, b.uconst(15));
    const uint32_t dpp_in_bounds = b.ucmp(
        Op_UGreaterThanEqual, dpp_row_lane, dpp_amount);
    // Keep even the disabled lane's scratch address valid. BOUND_CTRL=0 uses the validity gate
    // below to preserve old VDST rather than consuming this self-addressed placeholder.
    const uint32_t dpp_source_index = b.sel(
        dpp_in_bounds,
        b.ibin(Op_ISub, b.linear_localid, dpp_amount),
        b.linear_localid);
    const uint32_t dpp_shifted = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(value_base), dpp_source_index));
    const uint32_t dpp_source_metadata = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(metadata_base), dpp_source_index));
    const uint32_t dpp_source_event = b.ibin(
        Op_ShiftRightLogical, dpp_source_metadata, b.uconst(1));
    const uint32_t dpp_source_active = b.ucmp(
        Op_INotEqual,
        b.ibin(Op_BitwiseAnd, dpp_source_metadata, b.uconst(1)), zero);
    uint32_t dpp_valid_source = b.land(
        dpp_in_bounds, dpp_source_active);
    dpp_valid_source = b.land(
        dpp_valid_source, b.ucmp(Op_IEqual, dpp_source_event, dpp_event));
    uint32_t dpp_result = b.ibin(Op_IAdd, dpp_source,
        b.ngg_workgroup_export_probe
            ? b.sel(dpp_valid_source, dpp_shifted, zero) : dpp_shifted);
    // Allocate and consume an operation field only for streams containing MAX. An ADD-only
    // stream retains its previous IDs and graph exactly. Static events isolate different sites.
    if (variables.maximum) {
        const uint32_t maximum = b.load_function(b.t_bool, variables.maximum);
        dpp_result = b.sel(maximum, b.uext2(Glsl_UMax, dpp_source, dpp_shifted), dpp_result);
    }
    const uint32_t dpp_write = b.land(b.land(dpp_pending, dpp_active),
        b.ngg_workgroup_export_probe
            ? b.lor(dpp_bounded, dpp_valid_source) : dpp_valid_source);
    const uint32_t dpp_dst = b.load_function(b.t_u32, variables.destination);
    for (int reg : destinations) {
        const auto kv = vv.find(reg);
        if (kv == vv.end()) return false;
        const uint32_t selected = b.land(
            dpp_write, b.ucmp(Op_IEqual, dpp_dst,
                              b.uconst(static_cast<uint32_t>(reg))));
        const uint32_t old = b.load_function(b.t_u32, kv->second);
        b.store_function(kv->second, b.sel(selected, dpp_result, old));
    }
    // The physical VGPR definition invalidates scalar lane-spill aliases even when EXEC or the
    // shifted source suppresses this invocation's data write.
    for (const auto& kv : lv) {
        if (!destinations.contains(kv.first.first)) continue;
        const uint32_t selected = b.land(
            dpp_pending, b.ucmp(Op_IEqual, dpp_dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        if (!destinations.contains(kv.first.first)) continue;
        const uint32_t selected = b.land(
            dpp_pending, b.ucmp(Op_IEqual, dpp_dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    return true;
}

}  // namespace prosper::gpu
