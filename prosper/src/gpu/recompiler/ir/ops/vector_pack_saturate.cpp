// AMD70648 12.12 u32_to_u16 pack; AMD's earlier ISA explicitly specifies overflow clamping.
// Keep the two source/clamp stages and the late shift constant in their original order.
#include "gpu/recompiler/ir/ops/vector_pack_saturate.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
constexpr TernaryIntegerDef kDefs[] = {
    {Rdna2Format::VOP3, 0x36a, [](Builder& ir, const TernaryInputs& x) {
         const Value lo = ir.umin(x.s0(), ir.c32(0xffff));
         const Value hi = ir.umin(x.s1(), ir.c32(0xffff));
         return ir.or_(lo, ir.shl(hi, ir.c32(16)));
     }}};
}   // namespace
std::span<const TernaryIntegerDef> vector_pack_saturate_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_pack_saturate_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
