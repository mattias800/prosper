// vopc_compare_values.cpp — the per-lane comparison value of each VOPC opcode not yet defined in
// ir/ops (moved verbatim from emit_alu's VOPC switch). Each group is one function; a group that
// moves onto an ir/ops definition is deleted from here (HOW_TO_ADD_A_FAMILY.md, "Mask-writing
// compares").
#include "gpu/recompiler/ir/lower/vopc_lowering_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_compare.hpp"

namespace prosper::gpu::ir {

uint32_t VopcLowering::compare_value() {
    // A value override is a guest fact the proof passes read, emitted INSTEAD of the relation
    // (zero_compare, magnitude_compare). Only the f32 EQ/NEQ and LE/GE compares produce one.
    if (zero_cmp) return zero_cmp;
    if (magnitude_cmp) return magnitude_cmp;
    // Definitions with prepared sources consume them after the value overrides.
    switch (eff) {
        case 0xC9:
        case 0xCA:
        case 0xCB:
        case 0xCC:
        case 0xCD:
        case 0xCE: return compare_f16();
        case 0x88: return compare_class_f32();
        default: break;
    }
    // Opcodes with an ir/ops definition (ops/vector_compare_*.cpp).
    if (const VectorCompareDef* def = find_vector_compare(eff))
        return emit_vector_compare(b, *def, a, c);
    uint32_t cmp = 0;
    switch (eff) {
        case 0xA1: case 0xA2: case 0xA3: case 0xA4: case 0xA5: case 0xA6:
            cmp = compare_integer64_signed();
            break;
        case 0xE1: case 0xE2: case 0xE3: case 0xE4: case 0xE5: case 0xE6:
            cmp = compare_integer64_unsigned();
            break;
        default: ok = false;
    }
    return cmp;
}

uint32_t VopcLowering::compare_integer64_signed() {
    uint32_t cmp = 0;
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
    return cmp;
}

uint32_t VopcLowering::compare_class_f32() {
    // CLASS tests the raw IEEE-754 category rather than doing a floating-point
    // comparison. Keep this entirely in the integer domain so signalling/quiet
    // NaNs and the sign of zero/NaN survive on every SPIR-V target.
    uint32_t class_raw = ra;
    if (in.src_abs[0])
        class_raw = b.ibin(Op_BitwiseAnd, class_raw, b.uconst(0x7fffffffu));
    if (in.src_neg[0])
        class_raw = b.ibin(Op_BitwiseXor, class_raw, b.uconst(0x80000000u));
    if (const VectorCompareDef* def = find_vector_compare(eff))
        return emit_vector_compare(b, *def, class_raw, rc);
    ok = false;
    return 0;
}

uint32_t VopcLowering::compare_integer64_unsigned() {
    uint32_t cmp = 0;
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
    return cmp;
}

// f16 compares (0xC8-0xCF; cmpx at +0x10 = 0xD8-0xDF folds here too — DOLL's title
// post PSes: `v_cmp_lt_f16_sdwa s6, 0, v7`). VERIFIED(round-trip llvm-mc gfx1030:
// v_cmp_lt/eq/le/gt/lg/ge_f16 = VOPC 0xC9-0xCE). The f16 value lives in the source's
// LOW half — unpack to f32 and compare there (exact: f16 order-embeds into f32); an
// inline FLOAT constant is already an f32 value, so it is used directly. abs/neg
// modifiers are applied after conversion (equivalent, conversion is monotone/exact).
uint32_t VopcLowering::compare_f16() {
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
    if (const VectorCompareDef* def = find_vector_compare(eff))
        return emit_vector_compare(b, *def, ha, hc);
    ok = false;
    return 0;
}

} // namespace prosper::gpu::ir
