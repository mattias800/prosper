// Existing VOP1 f16 unary value and placement paths, moved without changing their order.
#include "gpu/recompiler/ir/lower/vop1_unary_f16_lowering.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <optional>
namespace prosper::gpu::ir {
namespace {
// The VOP1 f16 unary family's operation, given its already-unpacked f16 operand as an f32 value.
// Both the plain e32 form and the SDWA word-select form lower through this one function, so the two
// encodings of one opcode cannot drift apart — the SDWA path previously carried its own three-op
// ternary, which is exactly how a family gains a member in one form and not the other.
// The domain is `vop1_is_f16_unary` (rdna2_decode.hpp), shared with the decoder's SDWA admission and
// asserted below rather than assumed: a caller that widens its gate without widening this switch
// would otherwise get `v_cos_f16` silently. Opcodes verified there by llvm-mc round-trip; trig input
// is in REVOLUTIONS, exactly as for the f32 forms. Returns 0 (never a valid SPIR-V id) for anything
// outside the family, which the callers treat as an unsupported instruction. CONFIDENCE: HIGH.
uint32_t emit_f16_unary(SpirvCompute& b, uint32_t opcode, uint32_t x) {
    const auto turns = [&] { return b.uconst(fbits(6.28318530717958647692f)); };
    switch (opcode) {
        case 0x54: return b.frcp(x);   // v_rcp_f16
        case 0x55: return b.fext1(Glsl_Sqrt, x);   // v_sqrt_f16
        case 0x56: return b.fext1(Glsl_InverseSqrt, x);   // v_rsq_f16
        case 0x57: return b.fext1(Glsl_Log2, x);   // v_log_f16
        case 0x58: return b.fext1(Glsl_Exp2, x);   // v_exp_f16
        case 0x5B: return b.fext1(Glsl_Floor, x);   // v_floor_f16
        case 0x5C: return b.fext1(Glsl_Ceil, x);   // v_ceil_f16
        case 0x5D: return b.fext1(Glsl_Trunc, x);   // v_trunc_f16
        case 0x5E: return b.fext1(Glsl_RoundEven, x);   // v_rndne_f16
        case 0x5F: return b.fext1(Glsl_Fract, x);   // v_fract_f16
        case 0x60: return b.fext1(Glsl_Sin, b.fbin(Op_FMul, x, turns()));   // v_sin_f16
        case 0x61: return b.fext1(Glsl_Cos, b.fbin(Op_FMul, x, turns()));   // v_cos_f16
        default: return 0;   // outside the family — fail visibly
    }
}

}   // namespace
std::optional<bool> lower_vop1_unary_f16(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                         uint32_t a, uint32_t& d, uint32_t old_d, bool& ok) {
    if (in.fmt != Rdna2Format::VOP1) return std::nullopt;
    // Packed unary f16 SDWA: select one source half, compute in f32, round back to f16, and
    // insert while preserving the opposite destination half. The admitted opcode set is the
    // whole VOP1 f16 unary family (see emit_f16_unary), matching the plain e32 lowering
    // below; the decoder gates the operand shape (WORD dst + UNUSED_PRESERVE, WORD/DWORD
    // source, no sext/neg/abs/clamp/omod).
    if (vop1_is_f16_unary(in.opcode) && in.sdwa_dst_sel != 6) {
        uint32_t raw = a;   // inline constants: f16-width encoding, not the f32 pattern
        if (in.src[0].kind == OperandKind::InlineFloat)
            raw = b.uconst(inline_float_f16_bits(in.src[0].value)
                           << (in.sdwa_src0_sel == 5 ? 16 : 0));
        else if (in.src[0].kind == OperandKind::InlineInt)
            raw = b.uconst(static_cast<uint32_t>(in.src[0].value));
        uint32_t x = b.unpack_half(raw, in.sdwa_src0_sel == 5 ? 1 : 0);
        uint32_t result = emit_f16_unary(b, in.opcode, x);
        if (!result) {
            ok = false;
            return true;
        }
        uint32_t r16 = b.pack_half_lo(result);
        d = in.sdwa_dst_sel == 5
                ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                         b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
        predicate_write(b, rs, in.dst.value, old_d);
        return true;
    }
    // PLAIN-form unary f16 transcendentals/rounders (no SDWA word select). The SDWA WORD
    // form of this family is handled above; the e32 form reaches here and used to reject —
    // Sonic Racing: CrossWorlds' post chain hits `v_rcp_f16_e32` (0x7e1ea910) three times
    // per boot (#2013). Every op in this family reads S0.f16 and writes D.f16_lo, so they
    // share one lowering: unpack the half, compute in f32, round once back to f16.
    // Bits [31:16] are PRESERVED (the gfx10 16-bit-VALU contract); only the SDWA
    // DWORD+UNUSED_PAD encoding zero-fills, which the decoder marks with has_sdwa.
    // VERIFIED(round-trip llvm-mc gfx1030, VOP1): 0x54 v_rcp_f16, 0x55 v_sqrt_f16,
    // 0x56 v_rsq_f16, 0x57 v_log_f16, 0x58 v_exp_f16, 0x5B v_floor_f16, 0x5C v_ceil_f16,
    // 0x5D v_trunc_f16, 0x5E v_rndne_f16, 0x5F v_fract_f16, 0x60 v_sin_f16, 0x61 v_cos_f16.
    // Trig input is in REVOLUTIONS, exactly as for the f32 forms. CONFIDENCE: HIGH.
    // PLAIN e32 ONLY (`!has_sdwa`). A DWORD-select SDWA form of one of these carries
    // modifiers this lowering does not model, and each would be dropped SILENTLY rather
    // than fail-visibly: the decoder's "trivial-or-modifier-only" admission at
    // rdna2_decode.cpp:64 gates neither src0 NEG/ABS nor DST_UNUSED, the generic VOP1
    // abs/neg at the top of this case applies them in the f32 domain to the packed dword
    // (wrong half for a 16-bit read), and UNUSED_SEXT would need sign extension rather than
    // the zero fill `has_sdwa` implies. CLAMP/OMOD likewise apply in the 16-bit domain.
    // Every live encoding this title uses is plain `_e32` (`7e1ea910`, `7e08a504`), so
    // rejecting the SDWA forms costs nothing and is strictly no worse than the whole-opcode
    // reject they got before. (Raised in review of PR #2067.)
    if (vop1_is_f16_unary(in.opcode) && !in.has_sdwa && in.sdwa_dst_sel == 6 && !in.clamp &&
        !in.omod) {
        uint32_t raw = a;   // inline constants: f16-width encoding, not the f32 pattern
        if (in.src[0].kind == OperandKind::InlineFloat)
            raw = b.uconst(inline_float_f16_bits(in.src[0].value));
        else if (in.src[0].kind == OperandKind::InlineInt)
            raw = b.uconst(static_cast<uint32_t>(in.src[0].value));
        // Plain e32 has no source modifiers and no half select; the SDWA forms that would
        // carry them are rejected by the `!has_sdwa` gate above.
        const uint32_t x = b.unpack_half(raw, 0);
        const uint32_t result = emit_f16_unary(b, in.opcode, x);
        if (!result) {
            ok = false;
            return true;
        }
        const uint32_t r16 = b.pack_half_lo(result);
        d = b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
        predicate_write(b, rs, in.dst.value, old_d);
        return true;
    }
    return std::nullopt;
}
}   // namespace prosper::gpu::ir
