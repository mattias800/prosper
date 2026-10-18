// The existing dynamic IMAGE_LOAD_MIP clamp, as a value only. The caller proves the level count
// and reads the selector; this definition adds neither admission nor a constant-folder route.
#pragma once

#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
struct ImageMipClampBlock {
    Block block;
    Value lod;
};
ImageMipClampBlock build_image_mip_clamp(uint32_t materialized_mip_levels);
Known eval_image_mip_clamp(Known requested, uint32_t materialized_mip_levels);
uint32_t emit_image_mip_clamp(SpirvCompute& b, uint32_t requested,
                              uint32_t materialized_mip_levels);
} // namespace prosper::gpu::ir
