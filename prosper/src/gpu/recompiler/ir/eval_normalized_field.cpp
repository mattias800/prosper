// Vulkan fixed-point conversion: divide by the positive field maximum, clamp SNORM at -1.
// Loaded fields are at most 16 bits, so their initial integer-to-float conversion is exact.
// The divide's reference is nominal binary32 RTE, not a portable device bit-accuracy promise.
#include "gpu/recompiler/ir/eval_normalized_field.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include <optional>

#include <algorithm>
#include <bit>
#include <cstdint>

namespace prosper::gpu::ir {
Known evaluate_normalized_field(uint32_t imm, uint32_t field) {
    const uint32_t norm_bits = imm & 0x7fffffffu;
    if (!norm_bits || norm_bits >= 0x7f800000u) return std::nullopt;
    const float norm = std::bit_cast<float>(norm_bits);
    const float integer =
        imm >> 31 ? static_cast<float>(std::bit_cast<int32_t>(field)) : static_cast<float>(field);
    const float normalized = integer / norm;
    return std::bit_cast<uint32_t>(imm >> 31 ? std::max(normalized, -1.0f) : normalized);
}
} // namespace prosper::gpu::ir
