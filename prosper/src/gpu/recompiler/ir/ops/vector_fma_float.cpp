// AMD70648 sections12.7/12.12: F32 fused multiply-add. GLSL Fma is a MED portability
// approximation (#5225), not a strict-rounding capability or a half/mixed implementation.
#include "gpu/recompiler/ir/ops/vector_fma_float.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
#include <utility>
namespace prosper::gpu::ir {
Value build_fused_mad_float(Builder& ir, const TernaryInputs& x) {
    // Retain the old nested call's unspecified S0/S1 argument evaluation shape. Both
    // multiplicands are read before lazy S2; no new source0-first contract is imposed.
    const auto multiplicands = std::make_pair(x.s0(), x.s1());
    return ir.f32fma(multiplicands.first, multiplicands.second, x.s2());
}
namespace {
constexpr TernaryIntegerDef kFma{Rdna2Format::VOP3, 0x14b, build_fused_mad_float};
} // namespace
Known eval_fused_mad_float(const std::array<Known, 3>& sources) {
    return eval_ternary_integer_op(kFma, sources);
}
uint32_t emit_fused_mad_float(SpirvCompute& b, const TernarySourceReader& read) {
    return emit_ternary_integer_op(b, kFma, read);
}
uint32_t emit_fused_mad_values(SpirvCompute& b, uint32_t a, uint32_t c, uint32_t addend) {
    const std::array<uint32_t, 3> sources = {a, c, addend};
    return emit_fused_mad_float(b, [&](uint32_t slot) { return sources[slot]; });
}
} // namespace prosper::gpu::ir
