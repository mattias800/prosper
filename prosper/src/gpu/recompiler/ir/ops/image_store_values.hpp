// Pure storage-image data conversion and channel selects. Eligibility and source slots are
// established by the caller; no resource access or architectural state lives in these graphs.
#pragma once
#include "gpu/recompiler/ir/ir.hpp"
#include <cstdint>
#include <span>
#include <vector>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
enum class ImageStorageWordKind { Raw, FloatHalf, UnsignedHalf };
struct ImageStorageValue {
    Block block;
    std::vector<uint32_t> sources;
    Value value;
};
ImageStorageValue build_image_storage_word(ImageStorageWordKind kind, uint32_t half);
ImageStorageValue build_image_storage_select(int source, uint32_t constant);
uint32_t emit_image_storage_value(SpirvCompute& b, const ImageStorageValue& value,
                                  std::span<const uint32_t> words);
} // namespace prosper::gpu::ir
