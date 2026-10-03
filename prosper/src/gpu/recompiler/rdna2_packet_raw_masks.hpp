#pragma once
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {
// Portable complete-mask FFBH service. The status adapter preserves the dispatcher's named refusal.
bool emit_cfg_mask_ffbh_phase(SpirvCompute&, uint32_t pending_var, uint32_t mask_var,
                              uint32_t write_var, uint32_t event_var, uint32_t half_var,
                              uint32_t dst_var, uint32_t lane, uint32_t wave_index,
                              uint32_t result_base, const std::set<int>& destinations,
                              const std::map<int, uint32_t>& vector_vars,
                              const std::map<std::pair<int, int>, uint32_t>& spill_vars,
                              const std::map<std::pair<int, int>, uint32_t>& spill_mask_vars);
}   // namespace prosper::gpu
