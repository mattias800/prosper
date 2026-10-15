// vop2.cpp — VOP2 lowering: DPP routes, source modifiers, the operation, destination modifiers.
#include "gpu/recompiler/ir/lower/vector_lowering_internal.hpp"

namespace prosper::gpu::ir {

bool lower_vop2(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave) {
    return Vop2Lowering({b, rs, in, ok, allow_wave}).lower();
}

bool Vop2Lowering::lower() {
    a = val(in.src[0]);
    c = val(in.src[1]);
    old_d = vreg_old(b, rs, in.dst.value);
    if (auto routed = route_dpp_row_shr_in_place()) return *routed;
    if (auto routed = route_vadd_dpp_partial_row()) return *routed;
    if (auto routed = stage_dpp_source()) return *routed;
    // SDWA float source modifiers (only ever set on float ops by the assembler): abs then neg.
    // Packed-f16 ops apply these after selecting/unpacking the half below.
    packed_f16 = in.opcode == 0x32 || in.opcode == 0x33 || in.opcode == 0x35 ||
                 in.opcode == 0x39 || in.opcode == 0x3A;
    integer_sdwa = in.has_sdwa &&
        (in.opcode == 0x0B || (in.opcode >= 0x11 && in.opcode <= 0x14) ||
         in.opcode == 0x16 || in.opcode == 0x18 ||
         (in.opcode >= 0x1A && in.opcode <= 0x1E) ||
         (in.opcode >= 0x25 && in.opcode <= 0x2A));
    if (auto routed = stage_source_modifiers()) return *routed;
    uint32_t& d = vreg[in.dst.value];
    if (auto computed = compute(d)) return *computed;
    if (auto routed = stage_sdwa_destination(d)) return *routed;
    if (auto routed = stage_output_modifiers(d)) return *routed;
    if (ok) {
        if (dpp_active && !in.dpp_bound_ctrl) d = b.sel(dpp_active, d, old_d);
        predicate_write(b, rs, in.dst.value, old_d);
    }
    return true;
}

// The operation itself, on the already-transformed sources.
std::optional<bool> Vop2Lowering::compute(uint32_t& d) {
    // The integer and bitwise group goes through its IR definition (ir/ops/vector_alu.cpp).
    if (const VectorOpDef* def = find_vector_op(in.fmt, in.opcode)) {
        d = emit_vector_op(b, *def, a, c);
        return std::nullopt;
    }
    if (compute_float(d)) return std::nullopt;
    if (compute_f16(d)) return std::nullopt;
    switch (in.opcode) {
        case 0x01: {                                         // v_cndmask_b32: dst = vcc ? src1 : src0
            if (!vcc) { ok = false; return true; }
            if (in.sdwa_dst_sel == 6) { d = b.sel(vcc, c, a); break; }
            auto word = [&](uint32_t raw, uint8_t sel) {
                if (sel == 5) raw = b.ibin(Op_ShiftRightLogical, raw, b.uconst(16));
                return b.ibin(Op_BitwiseAnd, raw, b.uconst(0xFFFFu));
            };
            uint32_t selected = b.sel(vcc, word(c, in.sdwa_src1_sel), word(a, in.sdwa_src0_sel));
            d = in.sdwa_dst_sel == 5
                ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                         b.ibin(Op_ShiftLeftLogical, selected, b.uconst(16)))
                : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), selected);
            break;
        }

        // Carry ops (VOP2 e32 form): carry-in + carry-out are VCC. v_add_co_ci(0x28)/
        // v_sub_co_ci(0x29)/v_subrev_co_ci(0x2a). Mirrors the VOP3B 0x128/129/12A logic with VCC.
        case 0x28: case 0x29: case 0x2A: {
            if (!vcc) { ok = false; return true; }
            uint32_t cin = b.sel(vcc, b.uconst(1), b.uconst(0));
            uint32_t carry;
            if (in.opcode == 0x28) {                          // (a + c) + cin
                uint32_t s1 = b.ibin(Op_IAdd, a, c); uint32_t k1 = b.ucmp(Op_ULessThan, s1, a);
                d = b.ibin(Op_IAdd, s1, cin); uint32_t k2 = b.ucmp(Op_ULessThan, d, s1);
                carry = b.bsel(k1, b.btrue(), k2);
            } else {                                          // (x - y) - cin  (subrev swaps)
                uint32_t x = in.opcode == 0x29 ? a : c, y = in.opcode == 0x29 ? c : a;
                uint32_t s1 = b.ibin(Op_ISub, x, y); uint32_t k1 = b.ucmp(Op_ULessThan, x, y);
                d = b.ibin(Op_ISub, s1, cin); uint32_t k2 = b.ucmp(Op_ULessThan, s1, cin);
                carry = b.bsel(k1, b.btrue(), k2);
            }
            // Carry-out masks follow ISA 3.9 like compare results: an EXEC-inactive lane's
            // VCC bit is written 0, not its raw carry (wave votes must not see phantom bits).
            vcc = rs.exec_narrowed ? b.land(rs.exec, carry) : carry;
            break;
        }

        default: ok = false;
    }
    return std::nullopt;
}

} // namespace prosper::gpu::ir
