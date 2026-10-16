// vopc.cpp — VOPC lowering (v_cmp_* / v_cmpx_*, VOPC and VOP3 encodings), moved out of emit_alu.
#include "gpu/recompiler/ir/lower/vector_alu_lowering.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_mask_write.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

namespace prosper::gpu::ir {

bool lower_vopc(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                bool allow_exec_update) {
    uint32_t& vcc = rs.vcc;
    auto val = [&](const Operand& o) { return operand_bits(b, rs, in, o, &ok); };
    const auto scalar_data_operand = [&](const Operand& source) {
        switch (source.kind) {
            case OperandKind::InlineInt:
            case OperandKind::InlineFloat:
            case OperandKind::Literal:
                return true;
            case OperandKind::SGPR:
                // #3596: "lives in a scalar register" is not the same as "is wave-uniform".
                // A `v_readfirstlane_b32` result is one lane's value in an SGPR, and a VCC
                // derived from it must NOT satisfy the wave-uniformity proof -- otherwise a
                // fragment vccz branch skips its wave vote and takes the lane's own bit.
                if (scalar_is_lane_local(rs, source.value)) return false;
                return rs.sreg.contains(source.value) ||
                    rs.sreg_input.contains(source.value);
            case OperandKind::Special:
                if (source.value == 125) return true; // SGPR_NULL
                if (source.value == 253) return rs.scc != 0;
                if (scalar_is_lane_local(rs, source.value)) return false;   // #3596
                return source.value >= 106 && source.value <= 124 &&
                    rs.sreg.contains(source.value);
            default:
                return false;
        }
    };
    const uint32_t ra = val(in.src[0]), rc = val(in.src[1]);  // raw bits (f16 compares re-derive)
    uint32_t a = ra, c = rc;
    uint32_t op = in.opcode;
    bool is_cmpx = vopc_is_cmpx(op);
    if (is_cmpx && !allow_exec_update) { ok = false; return true; }
    uint32_t eff = is_cmpx ? op - 0x10 : op;
    const bool integer64_compare =
        (eff >= 0xA1u && eff <= 0xA6u) ||
        (eff >= 0xE1u && eff <= 0xE6u);
    auto scalar_integer64_operand = [&](const Operand& source) {
        if (!scalar_data_operand(source)) return false;
        if (source.kind == OperandKind::SGPR ||
            (source.kind == OperandKind::Special && source.value >= 106 &&
             source.value <= 123)) {
            Operand high = source;
            ++high.value;
            return scalar_data_operand(high);
        }
        // Inline integers sign-extend, while literals and SGPR_NULL zero-extend.  Each
        // implied high half is a compile-time scalar.  Inline floats are rejected later
        // by integer64_halves and therefore cannot publish a proof from this path.
        return source.kind != OperandKind::InlineFloat;
    };
    const bool compare_wave_uniform = !is_cmpx && !rs.exec_narrowed &&
        (integer64_compare
            ? scalar_integer64_operand(in.src[0]) &&
              scalar_integer64_operand(in.src[1])
            : scalar_data_operand(in.src[0]) && scalar_data_operand(in.src[1]));
    // Integer compares have no ABS/NEG source semantics. A malformed e64 packet carrying
    // those bits must not be lowered as an unmodified integer operation.
    if (integer64_compare &&
        (in.src_abs[0] || in.src_abs[1] || in.src_neg[0] || in.src_neg[1])) {
        ok = false;
        return true;
    }
    // Float source modifiers (abs then neg — hardware order), set only on FLOAT compares by the
    // assembler (VOP3-encoded e64 or SDWA forms; e.g. DOLL's `v_cmp_gt_f32_sdwa vcc, |v5|, s4`).
    const bool guest_compare_fact = ok && !is_cmpx && !rs.exec_narrowed && !in.clamp && !in.omod &&
        !in.has_dpp && (!in.has_sdwa ||
            (in.sdwa_src0_sel == 6 && in.sdwa_src1_sel == 6 &&
             !in.sdwa_src0_sext && !in.sdwa_src1_sext));
    uint32_t zero_cmp = 0;
    const bool zero_compare_eligible = ok && b.is_fragment &&
        (eff == 0x02u || eff == 0x0du) && !in.clamp && !in.omod && !in.has_dpp &&
        !in.src_abs[0] && !in.src_abs[1] &&
        !in.src_neg[0] && !in.src_neg[1] &&
        (!in.has_sdwa || (in.sdwa_src0_sel == 6 && in.sdwa_src1_sel == 6 &&
                         !in.sdwa_src0_sext && !in.sdwa_src1_sext));
    if (zero_compare_eligible) {
        uint32_t literal = 0, word = 0;
        if (b.uconst_literal(ra, &literal) && (literal & 0x7fffffffu) == 0u)
            word = rc;
        else if (b.uconst_literal(rc, &literal) && (literal & 0x7fffffffu) == 0u)
            word = ra;
        if (word) {
            if (b.fragment_float_mode.available) {
                const uint32_t nonzero = b.f32_nonzero_bits(word);
                zero_cmp = eff == 0x02u ? b.logical_not(nonzero) : nonzero;
            } else {
                b.warn_fragment_float_mode_unavailable(in.pc);
            }
        }
    }
    uint32_t magnitude_cmp = 0, magnitude_word = 0;
    if (guest_compare_fact && !in.src_neg[0] && !in.src_neg[1]) {
        if (eff == 0x03 && in.src_abs[0] && !in.src_abs[1]) {
            magnitude_cmp = b.f32_abs_normal_le_bits(ra, rc);
            magnitude_word = ra;
        } else if (eff == 0x06 && !in.src_abs[0] && in.src_abs[1]) {
            // K >= ABS(U) is the same ordered relation with swapped operands.
            magnitude_cmp = b.f32_abs_normal_le_bits(rc, ra);
            magnitude_word = rc;
        }
    }
    if (!magnitude_cmp && !zero_cmp) {
        if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
        if (in.src_neg[0]) a = b.fneg(a);
        if (in.src_abs[1]) c = b.fext1(Glsl_FAbs, c);
        if (in.src_neg[1]) c = b.fneg(c);
    }
    // v_cmpx_* shares each type's compare set at base+0x10. On gfx10 it writes EXEC only,
    // preserving VCC. Map to the base compare, then narrow; vopc_is_cmpx covers the f32/f64,
    // i32/i64, and u32/u64 windows.
    const bool integer_compare =
        (eff >= 0x81u && eff <= 0x86u) ||
        (eff >= 0x89u && eff <= 0x8Eu) ||
        (eff >= 0xA9u && eff <= 0xAEu) ||
        (eff >= 0xC1u && eff <= 0xC6u);
    if (integer_compare) {
        auto sdwa_integer = [&](uint32_t raw, uint8_t sel) {
            if (sel <= 3u) return b.bfe_u(raw, b.uconst(8u * sel), b.uconst(8));
            if (sel <= 5u) return b.bfe_u(raw, b.uconst(16u * (sel - 4u)), b.uconst(16));
            return raw;
        };
        a = sdwa_integer(ra, in.sdwa_src0_sel);
        c = sdwa_integer(rc, in.sdwa_src1_sel);
    }
    struct Integer64Halves { uint32_t lo = 0, hi = 0; };
    auto integer64_halves = [&](const Operand& operand, uint32_t lo) {
        Integer64Halves result{lo, b.uconst(0)};
        if (operand.kind == OperandKind::VGPR || operand.kind == OperandKind::SGPR ||
            (operand.kind == OperandKind::Special && operand.value >= 106 &&
             operand.value <= 123)) {
            Operand high = operand;
            ++high.value;
            result.hi = val(high);
        } else if (operand.kind == OperandKind::InlineInt) {
            result.hi = b.uconst(operand.value < 0 ? 0xffffffffu : 0u);
        } else if (operand.kind == OperandKind::Literal ||
                   (operand.kind == OperandKind::Special && operand.value == 125)) {
            // Integer/untyped 32-bit literals and SGPR_NULL zero-extend in a B64 source.
            result.hi = b.uconst(0);
        } else {
            // In particular, an inline floating-point constant is a 64-bit DOUBLE operand
            // in a 64-bit compare, not an integer pair assembled from its f32 bit pattern.
            ok = false;
        }
        return result;
    };
    Integer64Halves a64, c64;
    if (integer64_compare) {
        a64 = integer64_halves(in.src[0], ra);
        c64 = integer64_halves(in.src[1], rc);
    }
    uint32_t cmp = 0;
    switch (eff) {
        case 0x00: cmp = b.bfalse(); break;                              // v_cmp_f_f32
        case 0x01: cmp = b.fcmp(Op_FOrdLessThan, a, c); break;         // v_cmp_lt_f32
        case 0x02: cmp = zero_cmp ? zero_cmp :
            b.fcmp(Op_FOrdEqual, a, c); break;                      // v_cmp_eq_f32
        case 0x03: cmp = magnitude_cmp ? magnitude_cmp :
            b.fcmp(Op_FOrdLessThanEqual, a, c); break;               // v_cmp_le_f32
        case 0x04: cmp = b.fcmp(Op_FOrdGreaterThan, a, c); break;      // v_cmp_gt_f32
        case 0x05: cmp = b.fcmp(Op_FOrdNotEqual, a, c); break;         // v_cmp_lg_f32
        case 0x06: cmp = magnitude_cmp ? magnitude_cmp :
            b.fcmp(Op_FOrdGreaterThanEqual, a, c); break;            // v_cmp_ge_f32
        case 0x07: {                                                    // v_cmp_o_f32
            const uint32_t a_ordered = b.fcmp(Op_FOrdEqual, a, a);
            const uint32_t c_ordered = b.fcmp(Op_FOrdEqual, c, c);
            cmp = b.land(a_ordered, c_ordered);
            break;
        }
        case 0x08: {                                                    // v_cmp_u_f32
            const uint32_t a_nan = b.fcmp(Op_FUnordNotEqual, a, a);
            const uint32_t c_nan = b.fcmp(Op_FUnordNotEqual, c, c);
            cmp = b.lor(a_nan, c_nan);
            break;
        }
        // NaN-inclusive f32 compares (the "n"-prefix set is the unordered negation of 0x1-0x6):
        case 0x09: cmp = b.fcmp(Op_FUnordLessThan, a, c); break;       // v_cmp_nge_f32 = !(a>=b)
        case 0x0A: cmp = b.fcmp(Op_FUnordEqual, a, c); break;          // v_cmp_nlg_f32 = !(a!=b)
        case 0x0B: cmp = b.fcmp(Op_FUnordLessThanEqual, a, c); break;  // v_cmp_ngt_f32 = !(a>b)
        case 0x0C: cmp = b.fcmp(Op_FUnordGreaterThan, a, c); break;    // v_cmp_nle_f32 = !(a<=b)
        case 0x0D: cmp = zero_cmp ? zero_cmp :
            b.fcmp(Op_FUnordNotEqual, a, c); break;                 // v_cmp_neq_f32 = !(a==b)
        case 0x0E: cmp = b.fcmp(Op_FUnordGreaterThanEqual, a, c); break;// v_cmp_nlt_f32 = !(a<b)
        case 0x0F: cmp = b.btrue(); break;                              // v_cmp_tru_f32
        case 0x81: cmp = b.scmp(Op_SLessThan, a, c); break;            // v_cmp_lt_i32
        case 0x82: cmp = b.ucmp(Op_IEqual, a, c); break;               // v_cmp_eq_i32
        case 0x83: cmp = b.scmp(Op_SLessThanEqual, a, c); break;       // v_cmp_le_i32
        case 0x84: cmp = b.scmp(Op_SGreaterThan, a, c); break;         // v_cmp_gt_i32
        case 0x85: cmp = b.ucmp(Op_INotEqual, a, c); break;            // v_cmp_ne_i32 (sign-agnostic)
        case 0x86: cmp = b.scmp(Op_SGreaterThanEqual, a, c); break;    // v_cmp_ge_i32
        // 16-bit integer compares (i16 0x89-0x8e, u16 0xa9-0xae) read bits [15:0] of each
        // source; the i16 window sign-extends and the u16 window zero-extends, after which
        // the ordinary 32-bit relational operators give the exact 16-bit answer. Both
        // windows carry the same six operations in the same order (lt/eq/le/gt/ne/ge), so
        // one shared body indexes them by the low three bits.
        // VERIFIED(round-trip llvm-mc gfx1030): 0x89 is v_cmp_lt_i16 and 0xab is
        // v_cmp_le_u16 — the encoding Sonic Racing: CrossWorlds emits (#2013).
        case 0x89: case 0x8A: case 0x8B: case 0x8C: case 0x8D: case 0x8E:
        case 0xA9: case 0xAA: case 0xAB: case 0xAC: case 0xAD: case 0xAE: {
            const bool narrow_signed = eff <= 0x8Eu;
            const uint32_t x = narrow_signed ? b.bfe_s(a, b.uconst(0), b.uconst(16))
                                             : b.bfe_u(a, b.uconst(0), b.uconst(16));
            const uint32_t y = narrow_signed ? b.bfe_s(c, b.uconst(0), b.uconst(16))
                                             : b.bfe_u(c, b.uconst(0), b.uconst(16));
            switch (eff & 7u) {
                case 1: cmp = narrow_signed ? b.scmp(Op_SLessThan, x, y)
                                            : b.ucmp(Op_ULessThan, x, y); break;
                case 2: cmp = b.ucmp(Op_IEqual, x, y); break;
                case 3: cmp = narrow_signed ? b.scmp(Op_SLessThanEqual, x, y)
                                            : b.ucmp(Op_ULessThanEqual, x, y); break;
                case 4: cmp = narrow_signed ? b.scmp(Op_SGreaterThan, x, y)
                                            : b.ucmp(Op_UGreaterThan, x, y); break;
                case 5: cmp = b.ucmp(Op_INotEqual, x, y); break;
                default: cmp = narrow_signed ? b.scmp(Op_SGreaterThanEqual, x, y)
                                             : b.ucmp(Op_UGreaterThanEqual, x, y); break;
            }
            break;
        }
        case 0xA1: case 0xA2: case 0xA3: case 0xA4: case 0xA5: case 0xA6: {
            const uint32_t lhs = b.u64_from_lohi(a64.lo, a64.hi);
            const uint32_t rhs = b.u64_from_lohi(c64.lo, c64.hi);
            // Flipping the sign bit maps two's-complement signed order onto unsigned order.
            // Keep the values in the declared u64 type: the generic `scmp` helper bitcasts
            // 32-bit operands and therefore cannot be used for a 64-bit scalar.
            const uint32_t ordered_lhs = b.u64_from_lohi(
                a64.lo, b.ibin(Op_BitwiseXor, a64.hi, b.uconst(0x80000000u)));
            const uint32_t ordered_rhs = b.u64_from_lohi(
                c64.lo, b.ibin(Op_BitwiseXor, c64.hi, b.uconst(0x80000000u)));
            switch (eff) {
                case 0xA1: cmp = b.ucmp(Op_ULessThan, ordered_lhs, ordered_rhs); break; // v_cmp_lt_i64
                case 0xA2: cmp = b.ucmp(Op_IEqual, lhs, rhs); break;               // v_cmp_eq_i64
                case 0xA3: cmp = b.ucmp(Op_ULessThanEqual, ordered_lhs, ordered_rhs); break; // v_cmp_le_i64
                case 0xA4: cmp = b.ucmp(Op_UGreaterThan, ordered_lhs, ordered_rhs); break; // v_cmp_gt_i64
                case 0xA5: cmp = b.ucmp(Op_INotEqual, lhs, rhs); break;            // v_cmp_ne_i64
                default:   cmp = b.ucmp(Op_UGreaterThanEqual, ordered_lhs, ordered_rhs); break; // v_cmp_ge_i64
            }
            break;
        }
        case 0x88: {                                                   // v_cmp_class_f32
            // CLASS tests the raw IEEE-754 category rather than doing a floating-point
            // comparison. Keep this entirely in the integer domain so signalling/quiet
            // NaNs and the sign of zero/NaN survive on every SPIR-V target.
            uint32_t class_raw = ra;
            if (in.src_abs[0])
                class_raw = b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x7fffffffu));
            if (in.src_neg[0])
                class_raw = b.ibin(Op_BitwiseXor, class_raw, b.uconst(0x80000000u));
            const uint32_t sign = b.ucmp(
                Op_INotEqual,
                b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x80000000u)), b.uconst(0));
            const uint32_t exponent =
                b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x7f800000u));
            const uint32_t mantissa =
                b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x007fffffu));
            const uint32_t exponent_zero = b.ucmp(Op_IEqual, exponent, b.uconst(0));
            const uint32_t exponent_all = b.ucmp(Op_IEqual, exponent, b.uconst(0x7f800000u));
            const uint32_t mantissa_zero = b.ucmp(Op_IEqual, mantissa, b.uconst(0));
            const uint32_t quiet_nan = b.ucmp(
                Op_INotEqual,
                b.ibin(Op_BitwiseAnd, mantissa, b.uconst(0x00400000u)), b.uconst(0));

            // AMD's mask order is sNaN, qNaN, -Inf, -normal, -subnormal, -zero,
            // +zero, +subnormal, +normal, +Inf. Select the input's one-hot class bit,
            // then test it against SRC1 (Astro's live packet uses 3 = either NaN).
            const uint32_t nan_class = b.sel(quiet_nan, b.uconst(2), b.uconst(1));
            const uint32_t inf_class = b.sel(sign, b.uconst(4), b.uconst(512));
            const uint32_t exp_all_class = b.sel(mantissa_zero, inf_class, nan_class);
            const uint32_t zero_class = b.sel(sign, b.uconst(32), b.uconst(64));
            const uint32_t subnormal_class = b.sel(sign, b.uconst(16), b.uconst(128));
            const uint32_t exp_zero_class = b.sel(mantissa_zero, zero_class, subnormal_class);
            const uint32_t normal_class = b.sel(sign, b.uconst(8), b.uconst(256));
            const uint32_t class_bit = b.sel(
                exponent_all, exp_all_class,
                b.sel(exponent_zero, exp_zero_class, normal_class));
            cmp = b.ucmp(Op_INotEqual,
                         b.ibin(Op_BitwiseAnd, rc, class_bit), b.uconst(0));
            break;
        }
        case 0xC1: cmp = b.ucmp(Op_ULessThan, a, c); break;            // v_cmp_lt_u32
        case 0xC2: cmp = b.ucmp(Op_IEqual, a, c); break;               // v_cmp_eq_u32
        case 0xC3: cmp = b.ucmp(Op_ULessThanEqual, a, c); break;       // v_cmp_le_u32
        case 0xC4: cmp = b.ucmp(Op_UGreaterThan, a, c); break;         // v_cmp_gt_u32
        case 0xC5: cmp = b.ucmp(Op_INotEqual, a, c); break;            // v_cmp_ne_u32
        case 0xC6: cmp = b.ucmp(Op_UGreaterThanEqual, a, c); break;    // v_cmp_ge_u32
        case 0xE1: case 0xE2: case 0xE3: case 0xE4: case 0xE5: case 0xE6: {
            const uint32_t lhs = b.u64_from_lohi(a64.lo, a64.hi);
            const uint32_t rhs = b.u64_from_lohi(c64.lo, c64.hi);
            switch (eff) {
                case 0xE1: cmp = b.ucmp(Op_ULessThan, lhs, rhs); break;             // v_cmp_lt_u64
                case 0xE2: cmp = b.ucmp(Op_IEqual, lhs, rhs); break;                // v_cmp_eq_u64
                case 0xE3: cmp = b.ucmp(Op_ULessThanEqual, lhs, rhs); break;        // v_cmp_le_u64
                case 0xE4: cmp = b.ucmp(Op_UGreaterThan, lhs, rhs); break;          // v_cmp_gt_u64
                case 0xE5: cmp = b.ucmp(Op_INotEqual, lhs, rhs); break;             // v_cmp_ne_u64
                default:   cmp = b.ucmp(Op_UGreaterThanEqual, lhs, rhs); break;     // v_cmp_ge_u64
            }
            break;
        }
        // f16 compares (0xC8-0xCF; cmpx at +0x10 = 0xD8-0xDF folds here too — DOLL's title
        // post PSes: `v_cmp_lt_f16_sdwa s6, 0, v7`). VERIFIED(round-trip llvm-mc gfx1030:
        // v_cmp_lt/eq/le/gt/lg/ge_f16 = VOPC 0xC9-0xCE). The f16 value lives in the source's
        // LOW half — unpack to f32 and compare there (exact: f16 order-embeds into f32); an
        // inline FLOAT constant is already an f32 value, so it is used directly. abs/neg
        // modifiers are applied after conversion (equivalent, conversion is monotone/exact).
        case 0xC9: case 0xCA: case 0xCB: case 0xCC: case 0xCD: case 0xCE: {
            auto f16v = [&](int k, uint32_t raw) -> uint32_t {
                // Inline constants supply their 16-bit operand encoding: floats the f16
                // pattern (0x3C00 for 1.0), ints raw two's-complement low bits (1 -> the
                // f16 denormal 0x0001). Reading the raw f32-pattern (float) or the value as
                // full-width f32 BITS (int) modeled a different number than hardware.
                uint32_t v = in.src[k].kind == OperandKind::InlineFloat
                                 ? b.unpack_half(b.uconst(inline_float_f16_bits(in.src[k].value)), 0)
                           : in.src[k].kind == OperandKind::InlineInt
                                 ? b.unpack_half(b.uconst(static_cast<uint32_t>(in.src[k].value)), 0)
                                 : b.unpack_half(raw, in.sdwa_src0_sel == 5u && k == 0 ? 1u :
                                                      in.sdwa_src1_sel == 5u && k == 1 ? 1u : 0u);
                if (in.src_abs[k]) v = b.fext1(Glsl_FAbs, v);
                if (in.src_neg[k]) v = b.fneg(v);
                return v;
            };
            uint32_t ha = f16v(0, ra), hc = f16v(1, rc);
            switch (eff) {
                case 0xC9: cmp = b.fcmp(Op_FOrdLessThan, ha, hc); break;          // v_cmp_lt_f16
                case 0xCA: cmp = b.fcmp(Op_FOrdEqual, ha, hc); break;             // v_cmp_eq_f16
                case 0xCB: cmp = b.fcmp(Op_FOrdLessThanEqual, ha, hc); break;     // v_cmp_le_f16
                case 0xCC: cmp = b.fcmp(Op_FOrdGreaterThan, ha, hc); break;       // v_cmp_gt_f16
                case 0xCD: cmp = b.fcmp(Op_FOrdNotEqual, ha, hc); break;          // v_cmp_lg_f16
                default:   cmp = b.fcmp(Op_FOrdGreaterThanEqual, ha, hc); break;  // v_cmp_ge_f16
            }
            break;
        }
        default: ok = false;
    }
    // A compare result is a per-lane mask. The e64/SDWAB form can target an SGPR pair (SDST)
    // instead of VCC — track it in sreg_bool so a later v_cndmask_b32_e64 / s_cselect can read it.
    // Otherwise it writes VCC; keep VCC's narrowed-state in sync (106/107 = VCC_LO/HI).
    if (ok) {
        if (magnitude_cmp && cmp == magnitude_cmp)
            b.guest_f32_abs_normal_le_words.emplace(cmp, magnitude_word);
        if (guest_compare_fact && eff == 0x02 &&
            !in.src_abs[0] && !in.src_abs[1] && !in.src_neg[0] && !in.src_neg[1]) {
            uint32_t zero_bits = 0;
            if (b.uconst_literal(ra, &zero_bits) && (zero_bits & 0x7fffffffu) == 0)
                b.guest_f32_zero_equal_words.emplace(cmp, rc);
            else if (b.uconst_literal(rc, &zero_bits) && (zero_bits & 0x7fffffffu) == 0)
                b.guest_f32_zero_equal_words.emplace(cmp, ra);
        }
        if (is_cmpx) {
            // v_cmpx writes EXEC ONLY on gfx10 (EXEC &= cmp) — it has NO VCC/SGPR destination.
            // The old shared handler fell into the `else` and set vcc = cmp for cmpx too,
            // clobbering a VCC value kept live ACROSS the cmpx, so a later v_cndmask/s_cbranch_vccz/
            // v_add_co_ci reading VCC got the compare mask instead of the real predicate (#464).
            // EXEC suppresses inactive vector comparisons. Select masks an unselected
            // comparison's poison; strict OpLogicalAnd would propagate it into EXEC.
            const uint32_t exec_before = rs.exec;
            rs.exec = b.bsel(rs.exec, cmp, b.bfalse()); rs.exec_narrowed = true;
            rs.exec_subset_of[rs.exec] = exec_before;   // EXEC' = EXEC & cmp
        } else {
            // ISA 3.9: "VCC[n] = EXEC[n] & (test passed for thread n)" — a lane inactive in
            // EXEC gets its mask bit forced to 0, never the raw test result. Masking at the
            // write keeps wave reductions (vccz votes, mbcnt publications, v_writelane mask
            // spills) from seeing phantom bits from inactive lanes. No-op when EXEC is full.
            // This is architectural predication, not scalar mask arithmetic: an inactive
            // lane writes defined zero even when its unevaluated comparison can be poison.
            const uint32_t masked = exec_masked_predicate(b, rs, cmp);
            if (in.dst.kind == OperandKind::SGPR && in.dst.value <= 105) {
                rs.sreg_bool[in.dst.value] = masked; rs.sreg_bool_narrowed[in.dst.value] = true;
                if (b.allow_b32_masks &&
                    (b.is_fragment || (b.is_compute && b.wave_size == 32)))
                    rs.sreg_bool_b32.insert(in.dst.value);
            } else if (b.allow_b32_masks && in.dst.kind == OperandKind::SGPR &&
                       (in.dst.value == 106 || in.dst.value == 107)) {
                // Wave32 SDWA/e64 compares can explicitly target either one-word VCC half as
                // an independent saved mask (Astro world-map PC1060 targets VCC_HI, then
                // consumes it with s_andn2_b32). Keep that physical destination distinct;
                // only VCC_LO is also the implicit condition used by vccz/cndmask forms.
                rs.sreg_bool[in.dst.value] = masked;
                rs.sreg_bool_narrowed[in.dst.value] = true;
                rs.sreg_bool_b32.insert(in.dst.value);
                rs.sreg.erase(in.dst.value);
                rs.sreg_srt.erase(in.dst.value);
                if (in.dst.value == 106) vcc = masked;
            } else {
                // VOPC replaces the architectural VCC predicate. In Wave32 the complete mask
                // occupies VCC_LO only; VCC_HI remains available as ordinary scalar scratch
                // (Astro's sibling traversal keeps a float there across an implicit compare).
                // Wave64 still replaces both physical words. Retain the low-word mask marker
                // so a following s_mov_b32 save sees the fresh predicate rather than stale
                // dispatcher-loaded scalar data.
                vcc = masked;
                rs.vcc_wave_uniform = compare_wave_uniform ? masked : 0;
                if (b.wave_size == 32) {
                    rs.sreg.erase(106);
                    rs.sreg_srt.erase(106);
                } else {
                    mask_write_clobbers_pair(rs, 106);
                }
                rs.sreg_bool_narrowed[106] = true;
                if (b.wave_size != 32) rs.sreg_bool_narrowed[107] = true;
                if (b.allow_b32_masks &&
                    (b.is_fragment || (b.is_compute && b.wave_size == 32))) {
                    rs.sreg_bool[106] = masked;
                    rs.sreg_bool_b32.insert(106);
                }
            }
        }
    }
    return true;
}

} // namespace prosper::gpu::ir
