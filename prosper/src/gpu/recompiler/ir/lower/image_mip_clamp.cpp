// IMAGE_LOAD_MIP's pure unsigned clamp, mechanically relocated before the IR migration.
#include "gpu/recompiler/ir/lower/image_mip_clamp_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu::ir {

uint32_t lower_image_mip_clamp(SpirvCompute& b, uint32_t requested,
                               uint32_t materialized_mip_levels) {
    const uint32_t lod = b.uext2(Glsl_UMin, requested, b.uconst(materialized_mip_levels - 1u));
    return lod;
}

} // namespace prosper::gpu::ir
