// Existing VOP1 conversion routes and values, mechanically extracted from emit_alu.
#include "gpu/recompiler/ir/lower/vop1_conversion_lowering.hpp"
#include <cstdint>
#include <optional>
#include "gpu/recompiler/ir/lower/vop1_conversion_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
std::optional<bool> Vop1ConversionLowering::route_half_integer() {
    // 16-bit integer <-> f16 converts (plain e32 form). CrossWorlds' post chain hits
    // v_cvt_u16_f16 at ten sites per boot (#2013); the other three are the same family and
    // the same wrapper. All four read S0's low 16 bits (WORD_1 under an SDWA source select)
    // and write D's low 16 bits, PRESERVING bits [31:16].
    //   0x50 v_cvt_f16_u16   D.f16 = f16(S0.u16)
    //   0x51 v_cvt_f16_i16   D.f16 = f16(S0.i16)
    //   0x52 v_cvt_u16_f16   D.u16 = u16(S0.f16), truncated toward zero and CLAMPED
    //   0x53 v_cvt_i16_f16   D.i16 = i16(S0.f16), truncated toward zero and CLAMPED
    // The 16-bit clamp is explicit: cvt_f2u/cvt_f2i saturate at the 32-bit boundary, which
    // is the wrong boundary here, and a bare mask would WRAP an out-of-range value instead
    // of saturating it. VERIFIED(round-trip llvm-mc gfx1030, VOP1 0x50-0x53; the live
    // encoding 7e08a504 is `v_cvt_u16_f16_e32 v4, v4`). CONFIDENCE: HIGH.
    // PLAIN e32 ONLY — see the `!has_sdwa` note on the transcendental block above.
    // The 16-bit convert family, in the plain e32 form (no SDWA control word at all) or in
    // the WORD-destination SDWA form the decoder admits: dst WORD_0/1 + UNUSED_PRESERVE
    // from a full-DWORD source with no modifier. A DWORD-destination SDWA form still
    // rejects, because its control word can carry src0 NEG/ABS that the generic VOP1 path
    // would apply in the f32 domain to the packed dword — the wrong half for an op that
    // reads bits[15:0].
    const auto* def = find_vector_half_integer_conversion_op(in.fmt, in.opcode);
    if (!def) return std::nullopt;
    const bool convert16_word_sdwa =
        in.has_sdwa && in.sdwa_dst_sel != 6 && in.sdwa_src0_sel >= 4 && in.sdwa_src0_sel <= 6;
    if ((!in.has_sdwa || convert16_word_sdwa) && !in.clamp && !in.omod) {
        uint32_t raw = a;   // inline constants: 16-bit-width encoding, not the f32 pattern
        if (in.src[0].kind == OperandKind::InlineFloat)
            raw = b.uconst(inline_float_f16_bits(in.src[0].value));
        else if (in.src[0].kind == OperandKind::InlineInt)
            raw = b.uconst(static_cast<uint32_t>(in.src[0].value));
        else if (in.sdwa_src0_sel == 5)
            // SDWA src0_sel WORD_1: the 16-bit operand is bits [31:16]. DWORD and WORD_0
            // both name [15:0], which is where the reads below already look.
            raw = b.ibin(Op_ShiftRightLogical, raw, b.uconst(16));
        const uint32_t r16 = emit_conversion_op(b, *def, raw);
        // dst_sel 6 (plain e32) and 4 (WORD_0) both write bits [15:0]; 5 writes [31:16].
        // Either way the opposite half is preserved.
        d = in.sdwa_dst_sel == 5
                ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                         b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
        predicate_write(b, rs, in.dst.value, old_d);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Vop1ConversionLowering::route_pack_word() {
    // v_cvt_f16_f32_sdwa inserts the converted half into the selected destination word and
    // preserves the other word (unlike the plain form, whose result occupies the low half).
    if (in.opcode == 0x0A && in.sdwa_dst_sel != 6) {
        const auto* def = find_vector_conversion_op(in.fmt, in.opcode);
        if (!def) return std::nullopt;
        const uint32_t r16 = emit_conversion_op(b, *def, a);
        d = in.sdwa_dst_sel == 5
                ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                         b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), r16);
        predicate_write(b, rs, in.dst.value, old_d);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Vop1ConversionLowering::route_integer_subword() {
    // v_cvt_i32_f32_sdwa / v_cvt_u32_f32_sdwa convert the full source, then insert the
    // selected low/high result word or byte while preserving the other destination parts.
    // The decoder accepts BYTE_0..BYTE_3, WORD_0, WORD_1 + UNUSED_PRESERVE + DWORD-source.
    // The two opcodes differ only in the signedness of the 32-bit conversion; the inserted
    // sub-dword is masked and shifted into place, preserving all untouched bytes.
    if ((in.opcode == 0x07 || in.opcode == 0x08) && in.sdwa_dst_sel != 6) {
        const auto* def = find_vector_conversion_op(in.fmt, in.opcode);
        if (!def) return std::nullopt;
        const uint32_t result = emit_conversion_op(b, *def, a);
        if (in.sdwa_dst_sel <= 3u) {
            const uint32_t shift = in.sdwa_dst_sel * 8u;
            const uint32_t byte_val = b.ibin(Op_BitwiseAnd, result, b.uconst(0xFFu));
            const uint32_t clear_mask = ~(0xFFu << shift);
            d = b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(clear_mask)),
                       b.ibin(Op_ShiftLeftLogical, byte_val, b.uconst(shift)));
        } else if (in.sdwa_dst_sel == 5u) {
            const uint32_t word = b.ibin(Op_BitwiseAnd, result, b.uconst(0xFFFFu));
            d = b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                       b.ibin(Op_ShiftLeftLogical, word, b.uconst(16)));
        } else {
            const uint32_t word = b.ibin(Op_BitwiseAnd, result, b.uconst(0xFFFFu));
            d = b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)), word);
        }
        predicate_write(b, rs, in.dst.value, old_d);
        return true;
    }
    return std::nullopt;
}

bool Vop1ConversionLowering::values() {
    const auto* def = find_vector_conversion_op(in.fmt, in.opcode);
    if (!def) return false;
    switch (in.opcode) {
        case 0x0A:
            // Plain-form 16-bit results write D.f16_lo and PRESERVE bits [31:16] (the gfx10
            // 16-bit-VALU contract; zero-fill is only the SDWA DWORD+UNUSED_PAD behavior,
            // which the decoder marks with has_sdwa — the accepted cvt SDWA subsets all use
            // WORD dsts and take the preserve branch above, so has_sdwa here means PAD).
            d = in.has_sdwa
                    ? emit_conversion_op(b, *def, a)
                    : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)),
                             emit_conversion_op(b, *def, a));
            break;   // v_cvt_f16_f32
        case 0x0B: {   // v_cvt_f32_f16 (SDWA may select high half)
            // An inline constant in a 16-bit operand position supplies its f16-width
            // encoding (float) or raw two's-complement bits (int) — NOT the f32 pattern
            // val() materializes (unpacking THAT low half turned inline 1.0 into 0.0).
            uint32_t raw = a;
            if (in.src[0].kind == OperandKind::InlineFloat)
                raw = b.uconst(inline_float_f16_bits(in.src[0].value)
                               << (in.sdwa_src0_sel == 5 ? 16 : 0));
            else if (in.src[0].kind == OperandKind::InlineInt)
                raw = b.uconst(static_cast<uint32_t>(in.src[0].value));
            d = emit_conversion_op(b, *def, raw, in.sdwa_src0_sel == 5);
            if (in.src_abs[0]) d = b.fext1(Glsl_FAbs, d);
            if (in.src_neg[0]) d = b.fneg(d);
            break;
        }
        case 0x0C:
            // Preserve the established plain-only RPI admission. Its exceptional-value
            // convention remains the adjacent saturating conversion (CONFIDENCE: MED).
            if (in.has_sdwa || in.has_dpp || in.src_abs[0] || in.src_neg[0] || in.clamp ||
                in.omod) {
                ok = false;
                break;
            }
            d = emit_conversion_op(b, *def, a);
            break;
        default: d = emit_conversion_op(b, *def, a); break;
    }
    return true;
}
std::optional<bool> Vop1ConversionLowering::routes() {
    if (auto routed = route_half_integer()) return routed;
    if (auto routed = route_pack_word()) return routed;
    if (auto routed = route_integer_subword()) return routed;
    return std::nullopt;
}
std::optional<bool> lower_vop1_conversion_routes(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                                 uint32_t a, uint32_t& d, uint32_t old_d,
                                                 bool& ok) {
    return Vop1ConversionLowering{{b, rs, in, a, d, old_d, ok}}.routes();
}
bool lower_vop1_conversion_values(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, uint32_t a,
                                  uint32_t& d, uint32_t old_d, bool& ok) {
    return Vop1ConversionLowering{{b, rs, in, a, d, old_d, ok}}.values();
}
}   // namespace prosper::gpu::ir
