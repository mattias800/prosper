#pragma once
#include <cstdint>
#include <functional>
#include <vector>
namespace prosper::gpu {
// A separate compile attempt preserves every successful legacy module, including retries.
bool native_fragment_collective_retry_active();
void request_native_fragment_collective_retry();
std::vector<uint32_t>
recompile_native_fragment_collectives(const std::function<std::vector<uint32_t>()>& compile);
} // namespace prosper::gpu
