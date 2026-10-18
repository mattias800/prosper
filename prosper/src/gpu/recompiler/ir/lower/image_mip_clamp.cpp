// IMAGE_LOAD_MIP's value-only lowering; its selector has already been read by the caller.
#include "gpu/recompiler/ir/lower/image_mip_clamp_lowering.hpp"

#include "gpu/recompiler/ir/ops/image_mip_clamp.hpp"

namespace prosper::gpu::ir {

uint32_t lower_image_mip_clamp(SpirvCompute& b, uint32_t requested,
                               uint32_t materialized_mip_levels) {
    return emit_image_mip_clamp(b, requested, materialized_mip_levels);
}

} // namespace prosper::gpu::ir
