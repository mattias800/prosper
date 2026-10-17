// vopc.cpp — VOPC lowering (v_cmp_* / v_cmpx_*, VOPC and VOP3 encodings): source admission and
// modifiers, the comparison value, then the shared lane-mask write.
#include "gpu/recompiler/ir/lower/vopc_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vopc_mask_write.hpp"

namespace prosper::gpu::ir {

bool lower_vopc(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave,
                bool allow_exec_update) {
    return VopcLowering({b, rs, in, ok, allow_wave}, allow_exec_update).lower();
}

bool VopcLowering::lower() {
    ra = val(in.src[0]);   // raw bits (f16 compares re-derive)
    rc = val(in.src[1]);
    a = ra;
    c = rc;
    is_cmpx = vopc_is_cmpx(in.opcode);
    if (is_cmpx && !allow_exec_update) { ok = false; return true; }
    // v_cmpx_* shares each type's compare set at base+0x10. On gfx10 it writes EXEC only,
    // preserving VCC. Map to the base compare, then narrow; vopc_is_cmpx covers the f32/f64,
    // i32/i64, and u32/u64 windows.
    eff = is_cmpx ? in.opcode - 0x10 : in.opcode;
    integer64_compare =
        (eff >= 0xA1u && eff <= 0xA6u) ||
        (eff >= 0xE1u && eff <= 0xE6u);
    compare_wave_uniform = !is_cmpx && !rs.exec_narrowed &&
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
    guest_compare_fact = ok && !is_cmpx && !rs.exec_narrowed && !in.clamp && !in.omod &&
        !in.has_dpp && (!in.has_sdwa ||
            (in.sdwa_src0_sel == 6 && in.sdwa_src1_sel == 6 &&
             !in.sdwa_src0_sext && !in.sdwa_src1_sext));
    zero_compare();
    magnitude_compare();
    if (!magnitude_cmp && !zero_cmp) source_modifiers();
    integer_sdwa_sources();
    if (integer64_compare) {
        a64 = integer64_halves(in.src[0], ra);
        c64 = integer64_halves(in.src[1], rc);
    }
    const uint32_t cmp = compare_value();
    if (ok) {
        record_guest_compare_facts(cmp);
        write_vopc_mask(b, rs, in, cmp, compare_wave_uniform);
    }
    return true;
}

bool VopcLowering::scalar_data_operand(const Operand& source) const {
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
}

bool VopcLowering::scalar_integer64_operand(const Operand& source) const {
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
}

// v_cmp_eq/neq_f32 against a literal zero in a fragment shader: the nonzero test of the other word.
void VopcLowering::zero_compare() {
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
}

// |U| <= K (and K >= |U|) as a guest magnitude fact, emitted instead of the float compare.
void VopcLowering::magnitude_compare() {
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
}

void VopcLowering::source_modifiers() {
    if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
    if (in.src_neg[0]) a = b.fneg(a);
    if (in.src_abs[1]) c = b.fext1(Glsl_FAbs, c);
    if (in.src_neg[1]) c = b.fneg(c);
}

void VopcLowering::integer_sdwa_sources() {
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
}

VopcLowering::Integer64Halves VopcLowering::integer64_halves(const Operand& operand,
                                                            uint32_t lo) {
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
}

// Guest compare facts the later proof passes read, keyed by the comparison value.
void VopcLowering::record_guest_compare_facts(uint32_t cmp) {
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
}

} // namespace prosper::gpu::ir
