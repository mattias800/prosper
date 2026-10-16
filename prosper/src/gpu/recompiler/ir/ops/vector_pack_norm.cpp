// AMD70648 op872/873: normalized low/high halves. Preserve the existing RNE helper graph.
#include "gpu/recompiler/ir/ops/vector_pack_norm.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
Value pack(Builder& ir, const TernaryInputs& x, bool is_signed) {
    const Value s0 = x.s0();
    const Value lo = ir.pack_norm16(s0, is_signed);
    const Value s1 = x.s1();
    const Value hi = ir.pack_norm16(s1, is_signed);
    return ir.or_(lo, ir.shl(hi, ir.c32(16)));
}
constexpr TernaryIntegerDef kDefs[] = {
    {Rdna2Format::VOP3, 0x368,
     [](Builder& ir, const TernaryInputs& x) { return pack(ir, x, true); }},
    {Rdna2Format::VOP3, 0x369,
     [](Builder& ir, const TernaryInputs& x) { return pack(ir, x, false); }},
};
}   // namespace
std::span<const TernaryIntegerDef> vector_pack_norm_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_pack_norm_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
