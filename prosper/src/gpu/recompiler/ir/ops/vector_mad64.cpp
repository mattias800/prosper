// AMD70648 VOP3B MAD64 values; preserve the pre-IR U32 graph and its append order.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <span>
#include <cstdint>
#include "gpu/recompiler/ir/ops/vector_mad64.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include <array>
namespace prosper::gpu::ir {
namespace {
Mad64Out mad(Builder& ir, const Mad64In& x, bool is_signed) {
    // CONFIDENCE: MED for signed carry: retain the old bit64 interpretation. RDNA2 names
    // {vcc_out,D.i64}; the later CDNA3 section12.11 explicitly describes a signed65-bit sum.
    const Value mul_lo = ir.imul(x.s0, x.s1);
    const Value mul_hi = is_signed ? ir.smulhi(x.s0, x.s1) : ir.umulhi(x.s0, x.s1);
    const Value result_lo = ir.iadd(mul_lo, x.add_lo);
    const Value carry_lo = ir.ult(result_lo, mul_lo);
    const Value high_sum = ir.iadd(mul_hi, x.add_hi);
    const Value carry_hi0 = ir.ult(high_sum, mul_hi);
    const Value carry_word = ir.select(carry_lo, ir.c32(1), ir.c32(0));
    const Value result_hi = ir.iadd(high_sum, carry_word);
    const Value carry_hi1 = ir.ult(result_hi, high_sum);
    Value carry = ir.boolor_(carry_hi0, carry_hi1);
    if (is_signed) {
        const Value signs_differ = ir.lshr(ir.xor_(mul_hi, x.add_hi), ir.c32(31));
        carry = ir.ine(signs_differ, ir.select(carry, ir.c32(1), ir.c32(0)));
    }
    return {result_lo, result_hi, carry};
}
Mad64Out unsigned_mad(Builder& ir, const Mad64In& x) {
    return mad(ir, x, false);
}
Mad64Out signed_mad(Builder& ir, const Mad64In& x) {
    return mad(ir, x, true);
}
constexpr VectorMad64Def kDefs[] = {
    {Rdna2Format::VOP3, 0x176, unsigned_mad},
    {Rdna2Format::VOP3, 0x177, signed_mad},
};
struct Built {
    Block block;
    Mad64Out out;
};
Built build(const VectorMad64Def& def) {
    Built built;
    Builder ir(built.block);
    const Mad64In in{ir.input(Type::U32), ir.input(Type::U32), ir.input(Type::U32),
                     ir.input(Type::U32)};
    built.out = def.build(ir, in);
    return built;
}
}   // namespace
std::span<const VectorMad64Def> vector_mad64_defs() {
    return kDefs;
}
const VectorMad64Def* find_vector_mad64(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
KnownMad64 eval_vector_mad64(const VectorMad64Def& def, Known s0, Known s1, Known add_lo,
                             Known add_hi) {
    const Built built = build(def);
    const std::array<Known, 4> inputs{s0, s1, add_lo, add_hi};
    const auto values = evaluate(built.block, inputs);
    return {values[built.out.lo.index], values[built.out.hi.index], values[built.out.carry.index]};
}
Mad64Ids emit_vector_mad64(SpirvCompute& b, const VectorMad64Def& def, uint32_t s0, uint32_t s1,
                           uint32_t add_lo, uint32_t add_hi) {
    const Built built = build(def);
    const std::array<uint32_t, 4> inputs{s0, s1, add_lo, add_hi};
    const auto ids = emit_spirv(b, built.block, inputs);
    return {ids[built.out.lo.index], ids[built.out.hi.index], ids[built.out.carry.index]};
}
}   // namespace prosper::gpu::ir
