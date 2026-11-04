#pragma once
#include <cstdint>
#include <vector>

namespace prosper::gpu {
// SPV_KHR_maximal_reconvergence, Execution Mode table. This is a device requirement.
inline constexpr uint32_t kMaximallyReconvergesKHR = 6023;
enum class FragmentReconvergence { None, Required, Malformed };
FragmentReconvergence fragment_spirv_reconvergence(const std::vector<uint32_t>& module);
struct SpirvCompute;
uint32_t fragment_readfirst_value(SpirvCompute&, uint32_t source, uint32_t exec_mask);
} // namespace prosper::gpu
