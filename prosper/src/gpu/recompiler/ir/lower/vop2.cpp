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
        case 0x03:                                          // v_add_f32
            b.observe_fragment_arithmetic_site(in.pc, FragmentArithmeticFamily::Add);
            d = b.fbin(Op_FAdd, a, c); break;
        case 0x04: d = b.fbin(Op_FSub, a, c); break;          // v_sub_f32
        case 0x05: d = b.fbin(Op_FSub, c, a); break;          // v_subrev_f32 (src1 - src0; e32 form of
                                                              // VOP3 0x105 — round-trip llvm-mc gfx1010 0x0a020702)
        case 0x08:                                          // v_mul_f32
            b.observe_fragment_arithmetic_site(in.pc, FragmentArithmeticFamily::Mul);
            d = b.fbin(Op_FMul, a, c); break;
        // v_min/v_max: hardware returns the OTHER operand when exactly one input is NaN
        // (ISA 12.7 ops 15/16: "if (S0 == NaN) D = S1; ..."), which is GLSL NMin/NMax.
        // Plain FMin/FMax are NaN-UNDEFINED — llvmpipe propagates a second-operand NaN, so
        // the ubiquitous max(min(x,K),lo) clamp idiom could yield NaN pixels where hardware
        // yields the bound.
        case 0x0F: d = b.fext2(Glsl_NMin, a, c); break;       // v_min_f32
        case 0x10: d = b.fext2(Glsl_NMax, a, c); break;       // v_max_f32
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
        // v_mac_f32 (0x1f) / v_fmac_f32 (0x2b): dst = src0*src1 + dst (accumulate into the dest).
        // mac vs fmac differ only in fused rounding — immaterial here. old_d = the dst accumulator.
        // NOTE(opcode ID): op 0x1f is v_mac_f32 on the PS5's ISA. v_mac_f32 was REMOVED on desktop
        // RDNA2/gfx1030 (where 0x1f is invalid and llvm-mc reads canonical v_dot2c at 0x02), but the
        // PS5 GPU retains the RDNA1/gfx1010 encoding — VERIFIED by round-tripping the scene VS's
        // actual op-0x1f word 0x3e261221 through `llvm-mc -mcpu=gfx1010` → `v_mac_f32_e32`. The VOP3
        // (e64) form of this same op is 0x11f, handled in the VOP3 switch below.
        case 0x1F: case 0x2B: d = b.fbin(Op_FAdd, b.fbin(Op_FMul, a, c), old_d); break;
        // The four mul-add-with-literal-K ops (K = in.literal). madmk/fmamk = src0*K + src1;
        // madak/fmaak = src0*src1 + K. (mad vs fma differ only in fused rounding — immaterial here.)
        case 0x20: case 0x2C: d = b.fbin(Op_FAdd, b.fbin(Op_FMul, a, b.uconst(in.literal)), c); break;  // v_madmk / v_fmamk
        case 0x21: case 0x2D: d = b.fbin(Op_FAdd, b.fbin(Op_FMul, a, c), b.uconst(in.literal)); break;  // v_madak / v_fmaak
        case 0x2F: d = b.pack_half2x16_rtz(a, c); break;      // v_cvt_pkrtz_f16_f32 (e32 form): RTZ clamp (#452)
        case 0x3C: {                                         // v_pk_fmac_f16: packed dst += src0*src1
            auto fmac_half = [&](uint32_t half) {
                return b.fbin(Op_FAdd,
                              b.fbin(Op_FMul, b.unpack_half(a, half), b.unpack_half(c, half)),
                              b.unpack_half(old_d, half));
            };
            uint32_t lo = b.pack_half_lo(fmac_half(0));
            uint32_t hi = b.ibin(Op_ShiftLeftLogical, b.pack_half_lo(fmac_half(1)), b.uconst(16));
            d = b.ibin(Op_BitwiseOr, lo, hi);
            break;
        }
        case 0x32: case 0x33: case 0x35: case 0x39: case 0x3A: { // v_add/sub/mul/max/min_f16
            // (SDWA WORD_1 = high 16; DWORD/WORD_0 = low 16 — an f16 op reads bits[15:0]);
            // f16xf16 products are exact in f32, so multiply in f32 and round once to f16.
            // The 16-bit result inserts into the selected dest half PRESERVING the other
            // (dst_sel WORD_1 for the SDWA pack idiom; DWORD/WORD_0 = the plain e32 form's
            // "write [15:0], preserve [31:16]" gfx10 f16-VOP2 contract). #273 (DOLL box-blur).
            auto source = [&](uint32_t raw, const Operand& operand, uint8_t sel, int k) {
                // An inline constant in a 16-bit operand carries its f16-width encoding
                // (float — exact even for 1/(2*pi)) or raw two's-complement bits (int:
                // inline 1 is the f16 denormal 0x0001, never 1.0). Register/scalar operands
                // carry packed halves and must be selected before abs/neg.
                uint32_t v = operand.kind == OperandKind::InlineFloat
                               ? b.unpack_half(b.uconst(inline_float_f16_bits(operand.value)), 0)
                           : operand.kind == OperandKind::InlineInt
                               ? b.unpack_half(b.uconst(static_cast<uint32_t>(operand.value)),
                                               sel == 5 ? 1 : 0)
                               : b.unpack_half(raw, sel == 5 ? 1 : 0);
                if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
                if (in.src_neg[k]) v = b.fneg(v);
                return v;
            };
            uint32_t x = source(a, in.src[0], in.sdwa_src0_sel, 0);
            uint32_t y = source(c, in.src[1], in.sdwa_src1_sel, 1);
            uint32_t p = in.opcode == 0x32 ? b.fbin(Op_FAdd, x, y)
                       : in.opcode == 0x33 ? b.fbin(Op_FSub, x, y)
                       : in.opcode == 0x35 ? b.fbin(Op_FMul, x, y)
                       : in.opcode == 0x39 ? b.fext2(Glsl_NMax, x, y)   // NaN -> other operand
                                           : b.fext2(Glsl_NMin, x, y);
            if (in.clamp) p = b.clamp01(p);
            uint32_t r16 = b.pack_half_lo(p);
            // dst_sel==6 covers TWO encodings the decoder now distinguishes: the SDWA
            // DWORD+UNUSED_PAD form (zero-fill, has_sdwa) and the plain e32 form, whose
            // gfx10 contract writes [15:0] and PRESERVES [31:16] (the comment above always
            // said so; the code zero-filled both, corrupting live packed high halves).
            d = in.sdwa_dst_sel == 6
                ? (in.has_sdwa
                       ? r16
                       : b.ibin(Op_BitwiseOr,
                                b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16))
              : (in.sdwa_dst_sel == 5)
                ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                         b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
            break;
        }
        default: ok = false;
    }
    return std::nullopt;
}

} // namespace prosper::gpu::ir
