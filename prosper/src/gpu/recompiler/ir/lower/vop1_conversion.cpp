// Existing VOP1 conversion routes and values, mechanically extracted from emit_alu.
#include "gpu/recompiler/ir/lower/vop1_conversion_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
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
    const bool convert16_word_sdwa =
        in.has_sdwa && in.sdwa_dst_sel != 6 && in.sdwa_src0_sel >= 4 && in.sdwa_src0_sel <= 6;
    if (in.opcode >= 0x50 && in.opcode <= 0x53 && (!in.has_sdwa || convert16_word_sdwa) &&
        !in.clamp && !in.omod) {
        const bool from_float = in.opcode >= 0x52;
        uint32_t raw = a;   // inline constants: 16-bit-width encoding, not the f32 pattern
        if (in.src[0].kind == OperandKind::InlineFloat)
            raw = b.uconst(inline_float_f16_bits(in.src[0].value));
        else if (in.src[0].kind == OperandKind::InlineInt)
            raw = b.uconst(static_cast<uint32_t>(in.src[0].value));
        else if (in.sdwa_src0_sel == 5)
            // SDWA src0_sel WORD_1: the 16-bit operand is bits [31:16]. DWORD and WORD_0
            // both name [15:0], which is where the reads below already look.
            raw = b.ibin(Op_ShiftRightLogical, raw, b.uconst(16));
        uint32_t r16;
        if (from_float) {
            // Clamping the INTEGER result is correct and sufficient here, and an outer
            // float-domain clamp would be redundant work on a hot path. `cvt_f2u` and
            // `cvt_f2i` are not thin OpConvertFToU/OpConvertFToS wrappers: they are
            // prosper's own saturating helpers (#135/#686, defined above in this file).
            // Both select NaN to 0 BEFORE converting, bound the operand so the conversion
            // itself is always defined, and saturate at the 32-bit rail. So every f16 a
            // source can hold is already mapped to a defined 32-bit value, and the only
            // thing left to do is narrow that rail from 32 to 16 bits.
            const uint32_t x = b.unpack_half(raw, 0);
            r16 = in.opcode == 0x52
                      ? b.uext2(Glsl_UMin, b.cvt_f2u(x), b.uconst(0xFFFFu))
                      : b.sext2(Glsl_SMin, b.sext2(Glsl_SMax, b.cvt_f2i(x), b.uconst(0xFFFF8000u)),
                                b.uconst(0x7FFFu));
        } else {
            const uint32_t word = b.ibin(Op_BitwiseAnd, raw, b.uconst(0xFFFFu));
            r16 = b.pack_half_lo(in.opcode == 0x50
                                     ? b.cvt_u2f(word)
                                     : b.cvt_i2f(b.bfe_s(word, b.uconst(0), b.uconst(16))));
        }
        r16 = b.ibin(Op_BitwiseAnd, r16, b.uconst(0xFFFFu));
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
        uint32_t r16 = b.pack_half_lo(a);
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
        const uint32_t result = in.opcode == 0x07 ? b.cvt_f2u(a) : b.cvt_f2i(a);
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
    switch (in.opcode) {
        case 0x05: d = b.cvt_i2f(a); break;   // v_cvt_f32_i32
        case 0x06: d = b.cvt_u2f(a); break;   // v_cvt_f32_u32
        case 0x07: d = b.cvt_f2u(a); break;   // v_cvt_u32_f32
        case 0x08:
            d = b.cvt_f2i(a);
            break;   // v_cvt_i32_f32
        // f16<->f32 converts (#273 — DOLL's title post PSes carry f16 intermediates):
        // v_cvt_f16_f32 packs into the LOW half (high bits zero); v_cvt_f32_f16 unpacks the
        // low half. VERIFIED(round-trip llvm-mc gfx1030: 0x0a/0x0b).
        case 0x0A:
            // Plain-form 16-bit results write D.f16_lo and PRESERVE bits [31:16] (the gfx10
            // 16-bit-VALU contract; zero-fill is only the SDWA DWORD+UNUSED_PAD behavior,
            // which the decoder marks with has_sdwa — the accepted cvt SDWA subsets all use
            // WORD dsts and take the preserve branch above, so has_sdwa here means PAD).
            d = in.has_sdwa
                    ? b.pack_half_lo(a)
                    : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)),
                             b.pack_half_lo(a));
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
            d = b.unpack_half(raw, in.sdwa_src0_sel == 5 ? 1 : 0);
            if (in.src_abs[0]) d = b.fext1(Glsl_FAbs, d);
            if (in.src_neg[0]) d = b.fneg(d);
            break;
        }
        case 0x0C: {   // v_cvt_rpi_i32_f32
            // AMD RDNA2 ISA: D.i = (int)floor(S0.f + 0.5), i.e. nearest integer with
            // halfway cases rounded toward +infinity (not ceil for every fraction).
            // The guide does not restate NaN/out-of-range handling for this specialized
            // form, so use the adjacent f32->i32 conversion's deterministic convention:
            // NaN -> 0 and saturation at INT_MIN/INT_MAX. cvt_f2i also keeps the SPIR-V
            // conversion operand representable on every backend.
            //
            // Only GTA V's exact plain e32 form is established here. SDWA/DPP and source
            // or output modifiers remain fail-visible instead of being silently ignored.
            // CONFIDENCE: MED (rounding formula HIGH; exceptional-value policy follows the
            // established adjacent architectural conversion because this opcode omits it).
            if (in.has_sdwa || in.has_dpp || in.src_abs[0] || in.src_neg[0] || in.clamp ||
                in.omod) {
                ok = false;
                break;
            }
            // Do not literally add 0.5 in f32 before floor: the float immediately below
            // 0.5 would double-round to 1.0. Split off the fractional part, then apply the
            // >= tie rule; for integral large magnitudes the fraction is already zero.
            const uint32_t lower = b.fext1(Glsl_Floor, a);
            const uint32_t fraction = b.fbin(Op_FSub, a, lower);
            const uint32_t round_up =
                b.fcmp(Op_FOrdGreaterThanEqual, fraction, b.uconst(fbits(0.5f)));
            const uint32_t rounded = b.fbin(
                Op_FAdd, lower, b.sel(round_up, b.uconst(fbits(1.0f)), b.uconst(fbits(0.0f))));
            d = b.cvt_f2i(rounded);
            break;
        }
        case 0x0D:   // v_cvt_flr_i32_f32
            // AMD RDNA2 ISA: floor the f32 value before converting to signed i32. This is
            // observably different from v_cvt_i32_f32's truncation for negative fractions.
            d = b.cvt_f2i(b.fext1(Glsl_Floor, a));
            break;
        // v_cvt_off_f32_i4: sign-extend the low 4-bit integer and scale by 1/16.
        // AMD RDNA2 ISA: "4-bit signed int to 32-bit float"; LLVM's intrinsic
        // contract specifies result = 0.0625f * src_i4. This is the only opcode
        // that blocked The Messenger's 1024x32 grading-LUT producer (#527).
        case 0x0E:
            d = b.fbin(Op_FMul, b.cvt_i2f(b.bfe_s(a, b.uconst(0), b.uconst(4))),
                       b.uconst(fbits(0.0625f)));
            break;
        // AMD RDNA2 ISA: select one unsigned byte from the source dword and convert it
        // directly to f32. The opcode number selects BYTE_0 through BYTE_3.
        case 0x11:
        case 0x12:
        case 0x13:
        case 0x14:
            d = b.cvt_u2f(b.bfe_u(a, b.uconst(8u * (in.opcode - 0x11u)), b.uconst(8)));
            break;

        default: return false;
    }
    return true;
}
std::optional<bool> Vop1ConversionLowering::routes() {
    if (auto routed = route_half_integer()) return *routed;
    if (auto routed = route_pack_word()) return *routed;
    if (auto routed = route_integer_subword()) return *routed;
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
