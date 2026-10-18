// Compute DPP source adapter, relocated with extract_case_routes.py.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
uint32_t SpirvCompute::subgroup_quad_permute(uint32_t value, uint32_t ctrl, uint32_t active) {
    const uint32_t lane = subgroup_local_id();
    const uint32_t quad_lane = ibin(Op_BitwiseAnd, lane, uconst(3));
    uint32_t selected = uconst(ctrl & 3u);
    for (uint32_t output_lane = 1; output_lane < 4; ++output_lane)
        selected = sel(ucmp(Op_IEqual, quad_lane, uconst(output_lane)),
                       uconst((ctrl >> (2u * output_lane)) & 3u), selected);
    const uint32_t source_lane =
        ibin(Op_BitwiseOr, ibin(Op_BitwiseAnd, lane, uconst(~3u)), selected);
    const uint32_t shuffled = subgroup_shuffle(value, source_lane);
        // AMD70648 Table90: all currently admitted DPP16 controls have FI=0.
    const uint32_t source_active = subgroup_shuffle(sel(active, uconst(1), uconst(0)), source_lane);
    return sel(ucmp(Op_INotEqual, source_active, uconst(0)), shuffled, uconst(0));
}

} // namespace prosper::gpu
