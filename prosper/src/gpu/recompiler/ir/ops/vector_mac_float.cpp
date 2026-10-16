// Existing MAC uses the exact non-fused MAD value graph, with oldD supplied by its lowering.
#include "gpu/recompiler/ir/ops/vector_mac_float.hpp"
#include "gpu/recompiler/ir/ops/vector_mad_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
constexpr TernaryIntegerDef kDefs[] = {{Rdna2Format::VOP3, 0x11f, build_nonfused_mad_float}};
}
std::span<const TernaryIntegerDef> vector_mac_float_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_mac_float_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
