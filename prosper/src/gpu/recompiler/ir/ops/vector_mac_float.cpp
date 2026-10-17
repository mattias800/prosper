// Existing MAC uses the exact non-fused MAD value graph, with oldD supplied by its lowering.
#include "gpu/recompiler/ir/ops/vector_mac_float.hpp"
#include "gpu/recompiler/ir/ops/vector_mad_float.hpp"
#include "gpu/recompiler/ir/ops/vector_fma_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
constexpr TernaryIntegerDef kDefs[] = {{Rdna2Format::VOP3, 0x11f, build_nonfused_mad_float}};
// AMD70648 12.7/12.7.1: opcode43 +0x100, with the old destination as its addend.
// CONFIDENCE: MED -- inherits GLSL Fma's implementation-dependent precision (#5225).
constexpr TernaryIntegerDef kFmac{Rdna2Format::VOP3, 0x12b, build_fused_mad_float};
}
std::span<const TernaryIntegerDef> vector_mac_float_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_mac_float_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
const TernaryIntegerDef* find_vector_fmac_float_op(Rdna2Format fmt, uint32_t opcode) {
    return kFmac.fmt == fmt && kFmac.opcode == opcode ? &kFmac : nullptr;
}
}   // namespace prosper::gpu::ir
