// builder.hpp — typed helpers that append IR instructions to a Block.
//
// A lowering calls these in the order the instructions must be emitted. Nested calls are evaluated
// in the compiler's argument order, which is also how the pre-IR emitter's nested SpirvCompute calls
// were ordered; writing a lowering with the same nesting as the code it replaces keeps the emitted
// SPIR-V byte-identical (HOW_TO_ADD_A_FAMILY.md).
#pragma once

#include "gpu/recompiler/ir/ir.hpp"

#include <bit>

namespace prosper::gpu::ir {

class Builder {
public:
    explicit Builder(Block& block) : block_(block) {}

    Value input(Type type) {
        return block_.append({Opcode::Input, type, block_.input_count(), {}});
    }
    Value c32(uint32_t value) { return block_.append({Opcode::ConstU32, Type::U32, value, {}}); }
    Value ctrue() { return block_.append({Opcode::ConstTrue, Type::Bool, 0, {}}); }
    Value cfalse() { return block_.append({Opcode::ConstFalse, Type::Bool, 0, {}}); }

    Value iadd(Value a, Value b) { return u32(Opcode::IAdd, a, b); }
    Value isub(Value a, Value b) { return u32(Opcode::ISub, a, b); }
    Value imul(Value a, Value b) { return u32(Opcode::IMul, a, b); }
    Value udiv(Value a, Value b) { return u32(Opcode::UDiv, a, b); }
    Value umulhi(Value a, Value b) { return u32(Opcode::UMulHi, a, b); }
    Value smulhi(Value a, Value b) { return u32(Opcode::SMulHi, a, b); }
    Value and_(Value a, Value b) { return u32(Opcode::And, a, b); }
    Value or_(Value a, Value b) { return u32(Opcode::Or, a, b); }
    Value xor_(Value a, Value b) { return u32(Opcode::Xor, a, b); }
    Value not_(Value a) { return block_.append({Opcode::Not, Type::U32, 0, {a}}); }
    Value ffbh_u32(Value a) { return block_.append({Opcode::FfbhU32, Type::U32, 0, {a}}); }
    Value find_umsb_u32(Value a) { return block_.append({Opcode::FindUMsbU32, Type::U32, 0, {a}}); }
    Value bit_count(Value a) { return block_.append({Opcode::BitCount, Type::U32, 0, {a}}); }
    Value bitreverse(Value a) { return block_.append({Opcode::BitReverse, Type::U32, 0, {a}}); }
    Value shl(Value a, Value count) { return u32(Opcode::Shl, a, count); }
    Value lshr(Value a, Value count) { return u32(Opcode::LShr, a, count); }
    Value ashr(Value a, Value count) { return u32(Opcode::AShr, a, count); }
    Value smin(Value a, Value b) { return u32(Opcode::SMin, a, b); }
    Value smax(Value a, Value b) { return u32(Opcode::SMax, a, b); }
    Value umin(Value a, Value b) { return u32(Opcode::UMin, a, b); }
    Value umax(Value a, Value b) { return u32(Opcode::UMax, a, b); }
    Value booland_(Value a, Value b) {
        return block_.append({Opcode::BoolAnd, Type::Bool, 0, {a, b}});
    }
    Value boolnot_(Value a) { return block_.append({Opcode::BoolNot, Type::Bool, 0, {a}}); }
    Value boolor_(Value a, Value b) {
        return block_.append({Opcode::BoolOr, Type::Bool, 0, {a, b}});
    }
    Value f32unordne(Value a, Value b) { return cmp(Opcode::F32UnordNe, a, b); }
    Value f32eq(Value a, Value b) { return cmp(Opcode::F32Eq, a, b); }
    Value f32clamp01(Value a) { return block_.append({Opcode::F32Clamp01, Type::U32, 0, {a}}); }
    Value pack_half_lo(Value a) { return block_.append({Opcode::PackHalfLo, Type::U32, 0, {a}}); }
    Value pack_normalized_field(Value a, uint32_t width, bool is_signed) {
        const uint32_t mode =
            width >= 2 && width <= 16 ? width | (uint32_t(is_signed) << 8) : 0xffffffffu;
        return block_.append({Opcode::PackNormalizedField, Type::U32, mode, {a}});
    }
    Value pack_unsigned_float(Value a, uint32_t mantissa) {
        return block_.append({Opcode::PackUnsignedFloat, Type::U32, mantissa, {a}});
    }
    Value f32add(Value a, Value b) { return u32(Opcode::F32Add, a, b); }
    Value f32sub(Value a, Value b) { return u32(Opcode::F32Sub, a, b); }
    Value f32mul(Value a, Value b) { return u32(Opcode::F32Mul, a, b); }
    Value f32fma(Value a, Value b, Value c) {
        return block_.append({Opcode::F32Fma, Type::U32, 0, {a, b, c}});
    }
    Value f32add_no_contraction(Value a, Value b) { return u32(Opcode::F32AddNoContraction, a, b); }
    Value f32mul_no_contraction(Value a, Value b) { return u32(Opcode::F32MulNoContraction, a, b); }
    Value f32nmin(Value a, Value b) { return u32(Opcode::F32NMin, a, b); }
    Value f32nmax(Value a, Value b) { return u32(Opcode::F32NMax, a, b); }
    Value f32fract(Value a) { return block_.append({Opcode::F32Fract, Type::U32, 0, {a}}); }
    Value f32trunc(Value a) { return block_.append({Opcode::F32Trunc, Type::U32, 0, {a}}); }
    Value f32ceil(Value a) { return block_.append({Opcode::F32Ceil, Type::U32, 0, {a}}); }
    Value f32roundeven(Value a) { return block_.append({Opcode::F32RoundEven, Type::U32, 0, {a}}); }
    Value f32floor(Value a) { return block_.append({Opcode::F32Floor, Type::U32, 0, {a}}); }
    Value f32exp2(Value a) { return block_.append({Opcode::F32Exp2, Type::U32, 0, {a}}); }
    Value f32log2(Value a) { return block_.append({Opcode::F32Log2, Type::U32, 0, {a}}); }
    Value f32rcp(Value a) { return block_.append({Opcode::F32Rcp, Type::U32, 0, {a}}); }
    Value f32inversesqrt(Value a) {
        return block_.append({Opcode::F32InverseSqrt, Type::U32, 0, {a}});
    }
    Value f32sqrt(Value a) { return block_.append({Opcode::F32Sqrt, Type::U32, 0, {a}}); }
    Value f32sin(Value a) { return block_.append({Opcode::F32Sin, Type::U32, 0, {a}}); }
    Value f32cos(Value a) { return block_.append({Opcode::F32Cos, Type::U32, 0, {a}}); }
    Value i32_to_f32(Value a) { return block_.append({Opcode::I32ToF32, Type::U32, 0, {a}}); }
    Value u32_to_f32(Value a) { return block_.append({Opcode::U32ToF32, Type::U32, 0, {a}}); }
    Value f32_to_u32_sat(Value a) {
        return block_.append({Opcode::F32ToU32Sat, Type::U32, 0, {a}});
    }
    Value f32_to_i32_sat(Value a) {
        return block_.append({Opcode::F32ToI32Sat, Type::U32, 0, {a}});
    }
    Value bfe_s(Value base, Value offset, Value count) {
        return block_.append({Opcode::BfeS, Type::U32, 0, {base, offset, count}});
    }
    Value unpack_half(Value a, uint32_t component) {
        return block_.append({Opcode::UnpackHalf, Type::U32, component, {a}});
    }
    Value f32ldexp(Value a, Value exponent) { return u32(Opcode::F32Ldexp, a, exponent); }
    Value pack_norm16(Value a, bool is_signed) {
        return block_.append({Opcode::PackNorm16, Type::U32, uint32_t(is_signed), {a}});
    }
    Value normalize_field(Value a, float divisor, bool is_signed) {
        const uint32_t bits = std::bit_cast<uint32_t>(divisor);
        // Reserve the sign bit for the mode; a negative divisor is malformed metadata.
        const uint32_t mode = bits >> 31 ? 0x7fc00000u : bits | (uint32_t(is_signed) << 31);
        return block_.append({Opcode::NormalizeField, Type::U32, mode, {a}});
    }
    Value f32abs(Value a) { return block_.append({Opcode::F32Abs, Type::U32, 0, {a}}); }
    Value f32lt(Value a, Value b) { return cmp(Opcode::F32Lt, a, b); }
    Value f32ge(Value a, Value b) { return cmp(Opcode::F32Ge, a, b); }
    Value f32le(Value a, Value b) { return cmp(Opcode::F32Le, a, b); }
    Value f32gt(Value a, Value b) { return cmp(Opcode::F32Gt, a, b); }
    Value f32lg(Value a, Value b) { return cmp(Opcode::F32Lg, a, b); }
    Value f32unordlt(Value a, Value b) { return cmp(Opcode::F32UnordLt, a, b); }
    Value f32unordeq(Value a, Value b) { return cmp(Opcode::F32UnordEq, a, b); }
    Value f32unordle(Value a, Value b) { return cmp(Opcode::F32UnordLe, a, b); }
    Value f32unordgt(Value a, Value b) { return cmp(Opcode::F32UnordGt, a, b); }
    Value f32unordge(Value a, Value b) { return cmp(Opcode::F32UnordGe, a, b); }
    Value bfe_u(Value base, Value offset, Value count) {
        return block_.append({Opcode::BfeU, Type::U32, 0, {base, offset, count}});
    }
    Value select(Value cond, Value a, Value b) {
        return block_.append({Opcode::Select, Type::U32, 0, {cond, a, b}});
    }
    Value bselect(Value cond, Value a, Value b) {
        return block_.append({Opcode::BoolSelect, Type::Bool, 0, {cond, a, b}});
    }
    Value ieq(Value a, Value b) { return cmp(Opcode::IEq, a, b); }
    Value ine(Value a, Value b) { return cmp(Opcode::INe, a, b); }
    Value ult(Value a, Value b) { return cmp(Opcode::ULt, a, b); }
    Value ule(Value a, Value b) { return cmp(Opcode::ULe, a, b); }
    Value ugt(Value a, Value b) { return cmp(Opcode::UGt, a, b); }
    Value uge(Value a, Value b) { return cmp(Opcode::UGe, a, b); }
    Value slt(Value a, Value b) { return cmp(Opcode::SLt, a, b); }
    Value sle(Value a, Value b) { return cmp(Opcode::SLe, a, b); }
    Value sgt(Value a, Value b) { return cmp(Opcode::SGt, a, b); }
    Value sge(Value a, Value b) { return cmp(Opcode::SGe, a, b); }

    // 64-bit words (HOW_TO_ADD_A_FAMILY.md, "64-bit values").
    Value pack_u64(Value lo, Value hi) {
        return block_.append({Opcode::PackU64, Type::U64, 0, {lo, hi}});
    }
    // The word whose UNSIGNED order is the SIGNED order of hi:lo: the sign bit flipped. Compare two
    // of these with u64lt..u64ge for a signed 64-bit relation.
    Value pack_u64_signed_order(Value lo, Value hi) {
        return pack_u64(lo, xor_(hi, c32(0x80000000u)));
    }
    Value u64shl(Value word, Value count) {
        return block_.append({Opcode::U64Shl, Type::U64, 0, {word, count}});
    }
    Value u64lshr(Value word, Value count) {
        return block_.append({Opcode::U64LShr, Type::U64, 0, {word, count}});
    }
    Value u64ashr(Value word, Value count) {
        return block_.append({Opcode::U64AShr, Type::U64, 0, {word, count}});
    }
    Value u64lo(Value v) { return block_.append({Opcode::U64Lo, Type::U32, 0, {v}}); }
    Value u64hi(Value v) { return block_.append({Opcode::U64Hi, Type::U32, 0, {v}}); }
    Value u64bfe(Value base, Value offset, Value count) {
        return block_.append({Opcode::U64Bfe, Type::U64, 0, {base, offset, count}});
    }
    Value u64eq(Value a, Value b) { return cmp(Opcode::U64Eq, a, b); }
    Value u64ne(Value a, Value b) { return cmp(Opcode::U64Ne, a, b); }
    Value u64lt(Value a, Value b) { return cmp(Opcode::U64Lt, a, b); }
    Value u64le(Value a, Value b) { return cmp(Opcode::U64Le, a, b); }
    Value u64gt(Value a, Value b) { return cmp(Opcode::U64Gt, a, b); }
    Value u64ge(Value a, Value b) { return cmp(Opcode::U64Ge, a, b); }
    // Binary64 relations over U64 words (HOW_TO_ADD_A_FAMILY.md, "F64 values").
    Value f64lt(Value a, Value b) { return cmp(Opcode::F64Lt, a, b); }
    Value f64eq(Value a, Value b) { return cmp(Opcode::F64Eq, a, b); }
    Value f64le(Value a, Value b) { return cmp(Opcode::F64Le, a, b); }
    Value f64gt(Value a, Value b) { return cmp(Opcode::F64Gt, a, b); }
    Value f64lg(Value a, Value b) { return cmp(Opcode::F64Lg, a, b); }
    Value f64ge(Value a, Value b) { return cmp(Opcode::F64Ge, a, b); }
    Value f64unordlt(Value a, Value b) { return cmp(Opcode::F64UnordLt, a, b); }
    Value f64unordeq(Value a, Value b) { return cmp(Opcode::F64UnordEq, a, b); }
    Value f64unordle(Value a, Value b) { return cmp(Opcode::F64UnordLe, a, b); }
    Value f64unordgt(Value a, Value b) { return cmp(Opcode::F64UnordGt, a, b); }
    Value f64unordne(Value a, Value b) { return cmp(Opcode::F64UnordNe, a, b); }
    Value f64unordge(Value a, Value b) { return cmp(Opcode::F64UnordGe, a, b); }
    // Cross-lane (eval_lanes.hpp).
    Value lane_id() { return block_.append({Opcode::LaneId, Type::U32, 0, {}}); }
    Value shuffle(Value value, Value lane) { return u32(Opcode::Shuffle, value, lane); }

private:
    Value u32(Opcode op, Value a, Value b) { return block_.append({op, Type::U32, 0, {a, b}}); }
    Value cmp(Opcode op, Value a, Value b) { return block_.append({op, Type::Bool, 0, {a, b}}); }
    Block& block_;
};

} // namespace prosper::gpu::ir
