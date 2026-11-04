#pragma once
#include <cstdint>
#include <vector>

namespace prosper::gpu {
// SPV_KHR_maximal_reconvergence, Execution Mode table. This is a device requirement.
inline constexpr uint32_t kMaximallyReconvergesKHR = 6023;
inline constexpr uint32_t kRequireFullQuadsKHR = 5089;
enum class FragmentReconvergence { None = 0, Required = 1, FullQuads = 2, Both = 3, Malformed = 4 };
FragmentReconvergence fragment_spirv_reconvergence(const std::vector<uint32_t>& module);
struct SpirvCompute;
uint32_t fragment_collective_entry_exec(SpirvCompute&);
uint32_t fragment_collective_wqm(SpirvCompute&, uint32_t active);
uint32_t fragment_readfirst_value(SpirvCompute&, uint32_t source, uint32_t exec_mask);
} // namespace prosper::gpu
