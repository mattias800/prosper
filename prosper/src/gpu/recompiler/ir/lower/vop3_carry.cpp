// vop3_carry.cpp — the VOP3B carry ops: v_add/sub/subrev_co_u32 (carry-out to SDST) and
// v_add/sub/subrev_co_ci_u32 (carry-in from SRC2, carry-out to SDST). Moved verbatim from emit_alu.
#include "gpu/recompiler/ir/lower/vop3_mask_lowering_internal.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Vop3MaskLowering::route_carry_in_out() {
    // Add/sub with carry-in + carry-out (VOP3B): v_add_co_ci_u32 (0x128), v_sub_co_ci (0x129),
    // v_subrev_co_ci (0x12A). carry-in = src2 mask (VCC or an SGPR bool); carry-out -> sdst mask.
    // dst = s0 (+/-) s1 (+/-) cin; carryout = unsigned overflow (add) / borrow (sub).
    // Carry-in from an UNTRACKED mask must reject (like cndmask above), not silently
    // default to 0 — a wrong carry produces 64-bit address math off by one in the low
    // word with no diagnostic.
    const Operand& s2 = in.src[2]; uint32_t cin_mask = 0;
    if (s2.value == 106 || s2.value == 107) {
        auto it = rs.sreg_bool.find(s2.value);
        if (rs.sreg_bool_b32.contains(s2.value) && it != rs.sreg_bool.end())
            cin_mask = it->second;
        else
            cin_mask = rs.vcc;
    }
    else if (s2.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(s2.value); if (it != rs.sreg_bool.end()) cin_mask = it->second; }
    if (!cin_mask) ok = false;
    else {
        uint32_t a = val(in.src[0]), c = val(in.src[1]);
        uint32_t cin = b.sel(cin_mask, b.uconst(1), b.uconst(0));
        uint32_t res, cout;
        if (in.opcode == 0x128) {                             // add: (a + b) + cin
            uint32_t s1 = b.ibin(Op_IAdd, a, c);
            uint32_t c1 = b.ucmp(Op_ULessThan, s1, a);        // wrap in a+b
            res = b.ibin(Op_IAdd, s1, cin);
            uint32_t c2 = b.ucmp(Op_ULessThan, res, s1);      // wrap in +cin
            cout = b.bsel(c1, b.btrue(), c2);                 // c1 || c2
        } else {                                              // sub / subrev: (a - b) - cin (borrow)
            uint32_t x = in.opcode == 0x129 ? a : c, y = in.opcode == 0x129 ? c : a;  // subrev swaps
            uint32_t s1 = b.ibin(Op_ISub, x, y);
            uint32_t b1 = b.ucmp(Op_ULessThan, x, y);         // borrow in x-y
            res = b.ibin(Op_ISub, s1, cin);
            uint32_t b2 = b.ucmp(Op_ULessThan, s1, cin);      // borrow in -cin
            cout = b.bsel(b1, b.btrue(), b2);
        }
        vreg[in.dst.value] = res;
        // carry-out -> sdst mask (VCC or a saved SGPR-pair bool). ISA 3.9: an
        // EXEC-inactive lane's mask bit is written 0, never its raw carry.
        const uint32_t cout_masked = rs.exec_narrowed ? b.land(rs.exec, cout) : cout;
        write_vop3b_carry_output(cout_masked);
    }
    return std::nullopt;
}

std::optional<bool> Vop3MaskLowering::route_carry_out() {
    // v_add_co_u32 (0x30F) / v_sub_co_u32 (0x310) / v_subrev_co_u32 (0x319): 32-bit add/
    // subtract that writes a carry/borrow-out to the VOP3B sdst mask (VCC or an SGPR pair).
    // No carry-IN — that is the _co_ci_ family (VOP2 0x28-0x2A / VOP3B 0x128-0x12A). Mirrors
    // the VOP2 e32 carry emit (0x28-0x2A above) plus the v_mad_u64_u32 carry-out sdst write.
    // GTA V (PPSA04263) UI/content compute shaders use these; the missing op dropped the whole
    // shader, so its output texture rendered as bare untextured triangles (#1163/#1165).
    // CONFIDENCE: HIGH (RDNA2 ISA 6.5/3.9; coverage test in test_recompile_coverage).
    const uint32_t a = val(in.src[0]), c = val(in.src[1]);
    uint32_t d, carry;
    if (in.opcode == 0x30F) {                     // a + c ; carry = unsigned overflow
        d = b.ibin(Op_IAdd, a, c);
        carry = b.ucmp(Op_ULessThan, d, a);
    } else {                                      // sub: a-c ; subrev: c-a ; borrow = x < y
        const uint32_t x = in.opcode == 0x310 ? a : c;
        const uint32_t y = in.opcode == 0x310 ? c : a;
        d = b.ibin(Op_ISub, x, y);
        carry = b.ucmp(Op_ULessThan, x, y);
    }
    vreg[in.dst.value] = d;
    // ISA 3.9 carry-mask rule: an EXEC-inactive lane's bit is written 0, not its raw carry.
    const uint32_t carry_masked = rs.exec_narrowed ? b.land(rs.exec, carry) : carry;
    write_vop3b_carry_output(carry_masked);
    return std::nullopt;
}

void Vop3MaskLowering::write_vop3b_carry_output(uint32_t carry_mask) {
    if (in.sdst.kind == OperandKind::SGPR && b.allow_b32_masks &&
        (b.is_fragment || (b.is_compute && b.wave_size == 32))) {
        // A Wave32 carry destination is one physical word. Preserve an independently-live
        // adjacent SGPR and register the new lifetime in the B32 mask domain used by CFG joins.
        rs.sreg_bool[in.sdst.value] = carry_mask;
        rs.sreg_bool_narrowed[in.sdst.value] = true;
        rs.sreg_bool_b32.insert(in.sdst.value);
        rs.sreg.erase(in.sdst.value);
        rs.sreg_srt.erase(in.sdst.value);
        if (in.sdst.value == 106) rs.vcc = carry_mask;
    } else if (in.sdst.value == 106 || in.sdst.value == 107) {
        rs.vcc = carry_mask;
    } else if (in.sdst.kind == OperandKind::SGPR) {
        rs.sreg_bool[in.sdst.value] = carry_mask;
    } else {
        ok = false;
    }
}

void lower_vop3_carry_out(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave) {
    Vop3MaskLowering({b, rs, in, ok, allow_wave}).route_carry_out();
}

void lower_vop3_carry_in_out(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                             bool allow_wave) {
    Vop3MaskLowering({b, rs, in, ok, allow_wave}).route_carry_in_out();
}

}   // namespace prosper::gpu::ir
