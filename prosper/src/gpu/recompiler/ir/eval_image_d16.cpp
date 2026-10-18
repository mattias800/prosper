// #5420 revised decision: every inexact MIMG D16 conversion is implementation-defined.
// Vulkan's GLSL.std.450 precision table gives PackHalf2x16 an implementation-defined rounding
// mode. Inexact values and NaN payloads stay unknown. Without a denormal-preservation profile,
// even exact binary16 subnormals are not a portable host guarantee. This image-only evaluator
// does not change the generic PackHalfLo RTE contract or folding; execution still checks the
// nominal guest's exact subnormal bits independently (#5423, discussion_r4307199058).
#include "gpu/recompiler/ir/eval_image_d16.hpp"
#include "gpu/recompiler/ir/eval_half.hpp"
#include "gpu/recompiler/ir/eval_conversion.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include <optional>
namespace prosper::gpu::ir {
Known image_half_exact(uint32_t bits) {
    const auto half = pack_half_rte(bits);
    if (!half) return std::nullopt;
    if ((*half & 0x7c00u) == 0u && (*half & 0x03ffu) != 0u) return std::nullopt;
    const auto restored = evaluate_conversion(Opcode::UnpackHalf, 0, *half, 0, 0);
    return restored && *restored == bits ? half : std::nullopt;
}
Known image_half_pair_exact(uint32_t low, uint32_t high) {
    const auto lo = image_half_exact(low), hi = image_half_exact(high);
    return lo && hi ? Known{*lo | (*hi << 16)} : std::nullopt;
}
} // namespace prosper::gpu::ir
