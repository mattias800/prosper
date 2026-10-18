// The dynamic IMAGE_LOAD_MIP clamp value only. The caller supplies a previously read selector
// and the admitted materialized-level count; admission, proof state and the access stay there.
#pragma once

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
uint32_t lower_image_mip_clamp(SpirvCompute& b, uint32_t requested,
                               uint32_t materialized_mip_levels);
} // namespace prosper::gpu::ir
