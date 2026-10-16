// AMD 70648 12.12 V_PACK_B32_F16: S0 supplies low16 and S1 high16, preserving every bit.
// Keep the source reads in the old nested expression positions for byte-identical emission.
#include "gpu/recompiler/ir/ops/vector_pack16.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
constexpr TernaryIntegerDef kDefs[] = {
    {Rdna2Format::VOP3, 0x311, [](Builder& ir, const TernaryInputs& x) {
         return ir.or_(x.s0(), ir.shl(x.s1(), ir.c32(16)));
     }}};
}   // namespace
std::span<const TernaryIntegerDef> vector_pack16_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_pack16_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
