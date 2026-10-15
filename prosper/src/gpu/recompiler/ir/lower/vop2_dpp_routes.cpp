// vop2_dpp_routes.cpp — VOP2 DPP shapes: the in-place compute ROW_SHR, the partial-row v_add_nc_u32, and the SRC0 DPP transform.
// Relocated verbatim from emit_alu's VOP2 case by tools/refactor/extract_case_routes.py (ADR 0012).
#include "gpu/recompiler/ir/lower/vector_lowering_internal.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Vop2Lowering::route_dpp_row_shr_in_place() {
    if (ok &&
        emit_compute_inplace_dpp_row_shr(b, rs, in, a, old_d, allow_wave, row_result)) {
        vreg[in.dst.value] = row_result;
        predicate_write(b, rs, in.dst.value, old_d);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Vop2Lowering::route_vadd_dpp_partial_row() {
    if (b.is_compute && is_vadd_nc_u32_dpp_partial_row(in)) {
        const uint32_t lane = b.ibin(
            Op_BitwiseAnd, b.guest_lane_id(), b.uconst(b.wave_size - 1u));
        const uint32_t row = b.ibin(
            Op_ShiftRightLogical, lane, b.uconst(4));
        const uint32_t row_bit = b.ibin(
            Op_ShiftLeftLogical, b.uconst(1), row);
        const uint32_t row_selected = b.ucmp(
            Op_INotEqual,
            b.ibin(Op_BitwiseAnd, row_bit, b.uconst(in.dpp_row_mask)),
            b.uconst(0));
        const uint32_t result = b.ibin(Op_IAdd, a, c);
        vreg[in.dst.value] = b.sel(row_selected, result, old_d);
        predicate_write(b, rs, in.dst.value, old_d);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Vop2Lowering::stage_dpp_source() {
    // DPP16 quad_perm on src0 (#273): fragment FLOAT ops and compute quad swaps.  Bounded
    // ROW_SHR in the proven one-live-lane NGG projection supplies zero for lane 0.
    if (in.has_dpp) {
        const bool row_shr = in.dpp_ctrl >= 0x111u && in.dpp_ctrl <= 0x11Fu;
        uint32_t row_xor = 0;
        const bool row_ror8 = dpp_row_xor_ctrl(in.dpp_ctrl, &row_xor);
        const bool fop = in.opcode == 0x03 || in.opcode == 0x04 || in.opcode == 0x05 ||
                         in.opcode == 0x08 || in.opcode == 0x0F || in.opcode == 0x10 ||
                         in.opcode == 0x1F || in.opcode == 0x2B;
        if (row_ror8) {
            // The decoder admits only GTA V's full-mask, BC1, FI0 MIN/MAX packets here.
            // Direct subgroup shuffle is valid only when one native subgroup is exactly one
            // guest wave; portable/default-subgroup compute uses CFG scratch below.
            if (!b.is_compute || !dpp_row_xor_source_transform_ok(in) ||
                !b.native_subgroup_size) {
                ok = false; return true;
            }
            uint32_t valid_source = 0;
            const uint32_t rotated =
                b.subgroup_row_xor(a, rs.exec, row_xor, &valid_source);
            a = b.sel(valid_source, rotated, b.uconst(0));
        } else if (row_shr) {
            if (b.is_vertex) {
                if (!b.ngg_one_lane || in.opcode != 0x25 || !in.dpp_bound_ctrl) {
                    ok = false; return true;
                }
                a = b.uconst(0);
            } else if (b.is_fragment) {
                // Unbounded ROW_SHR is a pure SOURCE transform: DPP16 rewrites SRC0 only,
                // and an out-of-row or EXEC-inactive source disables the instruction,
                // preserving the old destination even when VDST and the two sources are
                // distinct. Both halves are opcode-independent, so every VOP2 whose only
                // architectural result is VDST lowers identically — Astro's material mask
                // uses v_or_b32, Syberia's Forward+ light scalarization uses the
                // v_min_u32 ROW_SHR:{1,2,4,8} wave reduction, and a per-opcode allow-list
                // silently drops every draw using any other one. The carry trio
                // (v_add_co_ci/v_sub_co_ci/v_subrev_co_ci) ALSO writes VCC, and a disabled
                // lane would have to preserve its old VCC bit too; the epilogue below only
                // restores VDST, so keep exactly that shape fail-visible.
                //
                // The generalization additionally requires `allow_wave`, which is exactly
                // "this emitter guarantees the whole guest wave is observable here". The
                // structured fragment shell passes it (every scalar branch and loop
                // condition there is subgroup-uniform, so all lanes reach this PC
                // together); the per-invocation CFG dispatcher passes false, because
                // adjacent lanes can be parked at DIFFERENT static DPP instructions and a
                // plain row shuffle would consume a neighbour published by another one.
                // That dispatcher lowers the exact V_MIN_U32 reduction itself, with a
                // static event tag carried through the identical shuffle. V_OR_B32 keeps
                // its historical unconditional admission so no stage that compiles today
                // starts failing; widening the dispatcher's coverage is separate work.
                const bool writes_carry_out = in.opcode == 0x28 ||
                                              in.opcode == 0x29 || in.opcode == 0x2A;
                const bool wave_visible = allow_wave || in.opcode == 0x1c;
                if (writes_carry_out || !wave_visible || in.dpp_bound_ctrl) {
                    ok = false; return true;
                }
                a = b.subgroup_row_shr(
                    a, rs.exec, in.dpp_ctrl - 0x110u, &dpp_active);
            } else {
                if (!b.is_compute || !fop) { ok = false; return true; }
                // ROW_SHR:N reads SRC0 from lane-N inside each architectural 16-lane row.
                // BOUND_CTRL=1 substitutes zero before the row; BOUND_CTRL=0 disables the
                // instruction for those lanes, preserving the old destination exactly.
                b.mark_subgroup_min16();
                const uint32_t shift = in.dpp_ctrl - 0x110u;
                const uint32_t lane = b.subgroup_local_id();
                const uint32_t row_lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(15));
                dpp_active = b.ucmp(Op_UGreaterThanEqual, row_lane, b.uconst(shift));
                // Never issue a shuffle with an out-of-range unsigned lane after subtraction;
                // inactive lanes address themselves and are then zeroed or masked off.
                const uint32_t source_lane = b.sel(
                    dpp_active, b.ibin(Op_ISub, lane, b.uconst(shift)), lane);
                const uint32_t shuffled = b.subgroup_shuffle(a, source_lane);
                a = in.dpp_bound_ctrl ? b.sel(dpp_active, shuffled, b.uconst(0))
                                      : shuffled;
            }
        } else {
            if (!fop) { ok = false; return true; }
            if (b.is_fragment) {
                a = b.dpp_quad(a, in.dpp_ctrl);
            } else if (b.is_compute) {
                // DPP transforms src0 only; src1 remains in this lane. A general shuffle covers
                // all 256 architectural quad_perm tables, including broadcasts and duplicates.
                a = b.subgroup_quad_permute(a, in.dpp_ctrl);
            } else {
                ok = false; return true;
            }
        }
    }
    return std::nullopt;
}

} // namespace prosper::gpu::ir
