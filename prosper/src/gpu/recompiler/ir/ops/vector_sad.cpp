// AMD70648 12.12: unsigned SAD, wrapping accumulation, and S1-reference masked SAD.
// Preserve the ascending field loop and original expression nesting for byte-identical emission.
#include "gpu/recompiler/ir/ops/vector_sad.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
using F = Rdna2Format;
using In = TernaryInputs;
Value accumulate(Builder& ir, Value sum, Value accumulator) {
    return ir.iadd(sum, accumulator);
}
Value subword(Builder& ir, const In& x, uint32_t bits, bool masked, bool high) {
    const Value s0 = x.s0(), s1 = x.s1(), s2 = x.s2();
    Value sum = ir.c32(0);
    for (uint32_t off = 0; off < 32; off += bits) {
        const Value a = ir.bfe_u(s0, ir.c32(off), ir.c32(bits));
        const Value b = ir.bfe_u(s1, ir.c32(off), ir.c32(bits));
        Value diff = ir.isub(ir.umax(a, b), ir.umin(a, b));
        if (masked) {
            const Value nz = ir.ine(b, ir.c32(0));
            diff = ir.select(nz, diff, ir.c32(0));
        }
        sum = ir.iadd(sum, diff);
    }
    if (high) sum = ir.shl(sum, ir.c32(16));
    return accumulate(ir, sum, s2);
}
constexpr TernaryIntegerDef kDefs[] = {
    {F::VOP3, 0x15a, [](Builder& ir, const In& x) { return subword(ir, x, 8, false, false); }},
    {F::VOP3, 0x15b, [](Builder& ir, const In& x) { return subword(ir, x, 8, false, true); }},
    {F::VOP3, 0x15c, [](Builder& ir, const In& x) { return subword(ir, x, 16, false, false); }},
    {F::VOP3, 0x15d,
     [](Builder& ir, const In& x) {
         const Value s0 = x.s0(), s1 = x.s1();
         const Value diff = ir.isub(ir.umax(s0, s1), ir.umin(s0, s1));
         return accumulate(ir, diff, x.s2());
     }},
    {F::VOP3, 0x171, [](Builder& ir, const In& x) { return subword(ir, x, 8, true, false); }},
};
}   // namespace
std::span<const TernaryIntegerDef> vector_sad_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_sad_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
