// AMD70648 §§12.8/12.12: VOP1 + 0x180 selects the e64 encoding. Reuse the same
// operation builders; source/result modifiers and final predication belong to the caller.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include <span>
#include <cstdint>
#include "gpu/recompiler/ir/ops/vector_round_float.hpp"
#include "gpu/recompiler/ir/ops/vector_float_unary.hpp"
namespace prosper::gpu::ir {
namespace {
using F = Rdna2Format;
constexpr VectorOpDef kDefs[] = {
    {F::VOP3, 0x1a0, build_vector_fract}, {F::VOP3, 0x1a1, build_vector_trunc},
    {F::VOP3, 0x1a2, build_vector_ceil},  {F::VOP3, 0x1a3, build_vector_roundeven},
    {F::VOP3, 0x1a4, build_vector_floor},
};
}   // namespace
std::span<const VectorOpDef> vector_round_float_defs() {
    return kDefs;
}
const VectorOpDef* find_vector_round_float_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
