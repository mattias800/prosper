// Per-output-word MIMG D16 values. Access, selected components and masked writes stay callers.
#pragma once
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include <cstdint>
#include <optional>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
enum class ImageD16Form : uint8_t { SelectedLow, SelectedPair, GatherPair };
struct ImageD16Block {
    Block block;
    Value word;
};
ImageD16Block build_image_d16_word(ImageD16Form form);
Known eval_image_d16_word(ImageD16Form form, Known low, Known high = std::nullopt);
uint32_t emit_image_d16_word(SpirvCompute& b, ImageD16Form form, uint32_t low, uint32_t high = 0);
} // namespace prosper::gpu::ir
