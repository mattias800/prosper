// Evaluate normalized16 conversion with bounded float scaling and explicit halfway-to-even integers.
#include "gpu/recompiler/ir/eval_normalized_pack.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
namespace prosper::gpu::ir {
Known evaluate_pack_norm16(uint32_t bits, uint32_t mode) {
    const float x = std::bit_cast<float>(bits);
    if (mode > 1 || std::isnan(x)) return std::nullopt;
    const float bounded = std::clamp(x, mode == 1 ? -1.0f : 0.0f, 1.0f);
    const float scaled = bounded * (mode == 1 ? 32767.0f : 65535.0f);
    const float whole = std::floor(scaled), tail = scaled - whole;
    int32_t rounded = static_cast<int32_t>(whole);
    if (tail > 0.5f || (tail == 0.5f && (rounded & 1))) ++rounded;
    return static_cast<uint32_t>(rounded) & 0xffffu;
}
}   // namespace prosper::gpu::ir
