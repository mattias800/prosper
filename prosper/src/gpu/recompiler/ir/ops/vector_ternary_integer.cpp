// AMD 70648 12.12: eight existing lane-wise VOP3 integer operations, modulo 2^32.
// Input calls and nesting intentionally match the old emitter. No encoding or fold is added.
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>
namespace prosper::gpu::ir {
namespace {
using F = Rdna2Format;
using In = TernaryInputs;
constexpr TernaryIntegerDef kDefs[] = {
    {F::VOP3, kVop3OpcodeAdd3U32,
     [](Builder& ir, const In& x) { return ir.iadd(ir.iadd(x.s0(), x.s1()), x.s2()); }},
    {F::VOP3, 0x346,
     [](Builder& ir, const In& x) {
         const Value sh = ir.and_(x.s1(), ir.c32(31));
         return ir.iadd(ir.shl(x.s0(), sh), x.s2());
     }},
    {F::VOP3, kVop3OpcodeAndOrB32,
     [](Builder& ir, const In& x) {
         const Value t = ir.and_(x.s0(), x.s1());
         return ir.or_(t, x.s2());
     }},
    {F::VOP3, 0x372,
     [](Builder& ir, const In& x) {
         const Value t = ir.or_(x.s0(), x.s1());
         return ir.or_(t, x.s2());
     }},
    {F::VOP3, 0x178,
     [](Builder& ir, const In& x) {
         const Value t = ir.xor_(x.s0(), x.s1());
         return ir.xor_(t, x.s2());
     }},
    {F::VOP3, 0x36f,
     [](Builder& ir, const In& x) {
         const Value sh = ir.and_(x.s1(), ir.c32(31));
         return ir.or_(ir.shl(x.s0(), sh), x.s2());
     }},
    {F::VOP3, 0x345,
     [](Builder& ir, const In& x) {
         const Value t = ir.xor_(x.s0(), x.s1());
         return ir.iadd(t, x.s2());
     }},
    {F::VOP3, 0x347,
     [](Builder& ir, const In& x) {
         const Value sh = ir.and_(x.s2(), ir.c32(31));
         return ir.shl(ir.iadd(x.s0(), x.s1()), sh);
     }},
};
struct Built {
    Block block;
    std::vector<uint32_t> sources;
    Value d;
};
Built build(const TernaryIntegerDef& def) {
    Built out;
    Builder ir(out.block);
    out.d = def.build(ir, TernaryInputs(ir, out.sources));
    return out;
}
}   // namespace
std::span<const TernaryIntegerDef> vector_ternary_integer_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_ternary_integer_op(Rdna2Format fmt, uint32_t op) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == op) return &def;
    return nullptr;
}
Known eval_ternary_integer_op(const TernaryIntegerDef& def, const std::array<Known, 3>& sources) {
    const Built out = build(def);
    std::vector<Known> inputs;
    inputs.reserve(out.sources.size());
    for (uint32_t source : out.sources) inputs.push_back(sources[source]);
    return evaluate(out.block, inputs)[out.d.index];
}
uint32_t emit_ternary_integer_op(SpirvCompute& b, const TernaryIntegerDef& def,
                                 const TernarySourceReader& read_source) {
    if (!read_source) {
        std::fprintf(stderr, "[ir] ternary source reader missing\n");
        std::abort();
    }
    const Built out = build(def);
    return emit_spirv_resolving_inputs(
        b, out.block, [&](uint32_t slot) { return read_source(out.sources[slot]); })[out.d.index];
}
}   // namespace prosper::gpu::ir
