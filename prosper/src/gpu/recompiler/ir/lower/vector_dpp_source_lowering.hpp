// Bound-id adapters only; admission, subgroup contracts and write predication stay at the caller.
#pragma once
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
struct DppRowSourceIds {
    uint32_t word;
    uint32_t in_bounds;
};
DppRowSourceIds emit_dpp_row_shr_source(SpirvCompute& b, uint32_t source, uint32_t active,
                                        uint32_t shift, bool bound_control);
uint32_t emit_dpp_quad_source(SpirvCompute& b, uint32_t source, uint32_t active, uint32_t control);
} // namespace prosper::gpu::ir
