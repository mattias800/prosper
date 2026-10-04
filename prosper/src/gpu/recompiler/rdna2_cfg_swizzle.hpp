#pragma once
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <unordered_set>

namespace prosper::gpu {
void emit_cfg_swizzle_phase(
    SpirvCompute& b, const std::unordered_set<uint32_t>& swizzle_pcs,
    uint32_t swizzle_pending_var, uint32_t swizzle_active_var, uint32_t swizzle_source_var,
    uint32_t swizzle_source_lane_var, uint32_t swizzle_dst_var, uint32_t zero, uint32_t no,
    const std::map<int, uint32_t>& vv, const std::map<std::pair<int, int>, uint32_t>& lv,
    const std::map<std::pair<int, int>, uint32_t>& lmv);
} // namespace prosper::gpu
