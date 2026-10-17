// AMD 70648 scalar integer instructions: only value math at already-admitted route positions.
// Nesting, unused intermediates and lazy source resolution preserve the old emission order.
#include "gpu/recompiler/ir/ops/scalar_integer64_values.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <vector>
#include <cstdio>
#include <cstdlib>
namespace prosper::gpu::ir {
namespace {
using F = Rdna2Format;
using In = Scalar64Inputs;
Scalar64Values bfe(Builder& ir, const In& x) {
    const Value offset = ir.and_(x.s1lo(), ir.c32(0x3f));
    const Value width = ir.and_(ir.lshr(x.s1lo(), ir.c32(16)), ir.c32(0x7f));
    return {ir.u64bfe(ir.pack_u64(x.s0lo(), x.s0hi()), offset, width), {}};
}
Scalar64Values popcount(Builder& ir, const In& x) {
    return {ir.iadd(ir.bit_count(x.s0lo()), ir.bit_count(x.s0hi())), {}};
}
Scalar64Values flbit(Builder& ir, const In& x) {
    auto leading_zeros = [&](Value word) {
        return ir.isub(ir.c32(31), ir.find_umsb_u32(ir.or_(word, ir.c32(1))));
    };
    // The caller retains the old prelude's zero and none constants at their original positions.
    return {ir.select(ir.ine(x.s0hi(), ir.c32(0)), leading_zeros(x.s0hi()),
                      ir.select(ir.ine(x.s0lo(), ir.c32(0)),
                                ir.iadd(ir.c32(32), leading_zeros(x.s0lo())), ir.c32(0xffffffffu))),
            {}};
}
Scalar64Values equal(Builder& ir, const In& x) {
    return {ir.booland_(ir.ieq(x.s0lo(), x.s1lo()), ir.ieq(x.s0hi(), x.s1hi())), {}};
}
Scalar64Values different(Builder& ir, const In& x) {
    const Value eq = equal(ir, x).first;
    return {ir.boolnot_(eq), {}};
}
Scalar64Values cselect(Builder& ir, const In& x) {
    const Value lo = ir.select(x.scc(), x.s0lo(), x.s1lo());
    const Value hi = x.high_available ? ir.select(x.scc(), x.s0hi(), x.s1hi()) : ir.c32(0);
    return {lo, hi};
}
Scalar64Values bitreplicate(Builder& ir, const In& x) {
    auto replicate_16 = [&](Value input) {
        Value out = ir.c32(0);
        for (uint32_t i = 0; i < 16; ++i) {
            const Value bit = ir.and_(ir.lshr(input, ir.c32(i)), ir.c32(1u));
            const Value rep = ir.or_(ir.shl(bit, ir.c32(2u * i)), ir.shl(bit, ir.c32(2u * i + 1u)));
            out = ir.or_(out, rep);
        }
        return out;
    };
    const Value lo = replicate_16(x.s0lo());
    const Value hi = replicate_16(ir.lshr(x.s0lo(), ir.c32(16u)));
    return {lo, hi};
}
constexpr ScalarInteger64Def kDefs[] = {
    {F::SOP2, kSop2OpcodeBfeU64, bfe},
    {F::SOP1, kSop1OpcodeBcnt1I32B64, popcount},
    {F::SOP1, 0x16, flbit},
    {F::SOPC, 0x12, equal},
    {F::SOPC, 0x13, different},
    {F::SOP2, 0x0b, cselect},
    {F::SOP1, kSop1OpcodeBitreplicateB64B32, bitreplicate},
};
struct Built {
    Block block;
    std::vector<Scalar64Source> slots;
    Scalar64Values result;
};
Built build(const ScalarInteger64Def& def, bool high_available) {
    Built out;
    Builder ir(out.block);
    out.result = def.build(ir, Scalar64Inputs(ir, out.slots, high_available));
    return out;
}
}   // namespace
std::span<const ScalarInteger64Def> scalar_integer64_value_defs() {
    return kDefs;
}
const ScalarInteger64Def* find_scalar_integer64_value_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
Scalar64Known eval_scalar_integer64_value(const ScalarInteger64Def& def,
                                          const std::array<Known64, 5>& sources,
                                          bool high_available) {
    const Built out = build(def, high_available);
    std::vector<Known64> inputs;
    inputs.reserve(out.slots.size());
    for (const auto source : out.slots) inputs.push_back(sources[static_cast<size_t>(source)]);
    const auto values = evaluate_wide(out.block, inputs);
    return {values[out.result.first.index],
            out.result.second.valid() ? values[out.result.second.index] : Known64{}};
}
Scalar64Ids emit_scalar_integer64_value(SpirvCompute& b, const ScalarInteger64Def& def,
                                        const Scalar64Reader& read, bool high_available) {
    if (!read) {
        std::fprintf(stderr, "[ir] scalar integer64 source reader missing\n");
        std::abort();
    }
    const Built out = build(def, high_available);
    const auto ids = emit_spirv_resolving_inputs(
        b, out.block, [&](uint32_t slot) { return read(out.slots[slot]); });
    return {ids[out.result.first.index],
            out.result.second.valid() ? ids[out.result.second.index] : 0u};
}
}   // namespace prosper::gpu::ir
