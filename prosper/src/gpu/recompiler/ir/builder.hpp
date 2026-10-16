// builder.hpp — typed helpers that append IR instructions to a Block.
//
// A lowering calls these in the order the instructions must be emitted. Nested calls are evaluated
// in the compiler's argument order, which is also how the pre-IR emitter's nested SpirvCompute calls
// were ordered; writing a lowering with the same nesting as the code it replaces keeps the emitted
// SPIR-V byte-identical (HOW_TO_ADD_A_FAMILY.md).
#pragma once

#include "gpu/recompiler/ir/ir.hpp"

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
    Value umulhi(Value a, Value b) { return u32(Opcode::UMulHi, a, b); }
    Value smulhi(Value a, Value b) { return u32(Opcode::SMulHi, a, b); }
    Value and_(Value a, Value b) { return u32(Opcode::And, a, b); }
    Value or_(Value a, Value b) { return u32(Opcode::Or, a, b); }
    Value xor_(Value a, Value b) { return u32(Opcode::Xor, a, b); }
    Value not_(Value a) { return block_.append({Opcode::Not, Type::U32, 0, {a}}); }
    Value ffbh_u32(Value a) { return block_.append({Opcode::FfbhU32, Type::U32, 0, {a}}); }
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
    Value boolor_(Value a, Value b) {
        return block_.append({Opcode::BoolOr, Type::Bool, 0, {a, b}});
    }
    Value f32unordne(Value a, Value b) { return cmp(Opcode::F32UnordNe, a, b); }
    Value f32clamp01(Value a) { return block_.append({Opcode::F32Clamp01, Type::U32, 0, {a}}); }
    Value pack_half_lo(Value a) { return block_.append({Opcode::PackHalfLo, Type::U32, 0, {a}}); }
    Value f32add(Value a, Value b) { return u32(Opcode::F32Add, a, b); }
    Value f32sub(Value a, Value b) { return u32(Opcode::F32Sub, a, b); }
    Value f32mul(Value a, Value b) { return u32(Opcode::F32Mul, a, b); }
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
    Value f32ge(Value a, Value b) { return cmp(Opcode::F32Ge, a, b); }
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

private:
    Value u32(Opcode op, Value a, Value b) { return block_.append({op, Type::U32, 0, {a, b}}); }
    Value cmp(Opcode op, Value a, Value b) { return block_.append({op, Type::Bool, 0, {a, b}}); }
    Block& block_;
};

} // namespace prosper::gpu::ir
