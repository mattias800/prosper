// AMD 70648 §12.10/table87: integer dot products plus a full-width accumulator.
// Source materialization and existing modifier admission remain in their original stages.
#include "gpu/recompiler/ir/ops/vector_integer_dot.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
template <bool Signed>
Value element(Builder& ir, Value source, uint32_t offset, uint32_t width) {
    if constexpr (Signed) return ir.bfe_s(source, ir.c32(offset), ir.c32(width));
    return ir.bfe_u(source, ir.c32(offset), ir.c32(width));
}
template <bool Signed>
Value dot2(Builder& ir, const IntegerDotInputs& x) {
    const uint32_t half0_0 = (x.low >> 0) & 1u;
    const uint32_t half0_1 = (x.low >> 1) & 1u;
    const uint32_t half1_0 = (x.high >> 0) & 1u;
    const uint32_t half1_1 = (x.high >> 1) & 1u;
    const Value e0_0 = element<Signed>(ir, x.s0, half0_0 * 16u, 16);
    const Value e0_1 = element<Signed>(ir, x.s1, half0_1 * 16u, 16);
    const Value p0 = ir.imul(e0_0, e0_1);
    const Value e1_0 = element<Signed>(ir, x.s0, half1_0 * 16u, 16);
    const Value e1_1 = element<Signed>(ir, x.s1, half1_1 * 16u, 16);
    const Value p1 = ir.imul(e1_0, e1_1);
    return ir.iadd(ir.iadd(p0, p1), x.s2);
}
template <bool Signed, uint32_t Width, uint32_t Count>
Value dotN(Builder& ir, const IntegerDotInputs& x) {
    Value sum = x.s2;
    for (uint32_t k = 0; k < Count; ++k) {
        const uint32_t offset = k * Width;
        const Value e0 = element<Signed>(ir, x.s0, offset, Width);
        const Value e1 = element<Signed>(ir, x.s1, offset, Width);
        sum = ir.iadd(sum, ir.imul(e0, e1));
    }
    return sum;
}
constexpr IntegerDotDef kDefs[] = {
    {Rdna2Format::VOP3P, 0x14, dot2<true>},       {Rdna2Format::VOP3P, 0x15, dot2<false>},
    {Rdna2Format::VOP3P, 0x16, dotN<true, 8, 4>}, {Rdna2Format::VOP3P, 0x17, dotN<false, 8, 4>},
    {Rdna2Format::VOP3P, 0x18, dotN<true, 4, 8>}, {Rdna2Format::VOP3P, 0x19, dotN<false, 4, 8>}};
struct Built {
    Block block;
    Value d;
};
Built build(const IntegerDotDef& def, uint8_t low, uint8_t high) {
    Built out;
    Builder ir(out.block);
    const Value s0 = ir.input(Type::U32), s1 = ir.input(Type::U32), s2 = ir.input(Type::U32);
    out.d = def.build(ir, {s0, s1, s2, low, high});
    return out;
}
}   // namespace
std::span<const IntegerDotDef> vector_integer_dot_defs() {
    return kDefs;
}
const IntegerDotDef* find_vector_integer_dot_op(Rdna2Format fmt, uint32_t op) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == op) return &def;
    return nullptr;
}
Known eval_integer_dot_op(const IntegerDotDef& def, const std::array<Known, 3>& sources,
                          uint8_t low, uint8_t high) {
    const Built out = build(def, low, high);
    return evaluate(out.block, sources)[out.d.index];
}
uint32_t emit_integer_dot_op(SpirvCompute& b, const IntegerDotDef& def,
                             const std::array<uint32_t, 3>& sources, uint8_t low, uint8_t high) {
    const Built out = build(def, low, high);
    return emit_spirv(b, out.block, sources)[out.d.index];
}
}   // namespace prosper::gpu::ir
