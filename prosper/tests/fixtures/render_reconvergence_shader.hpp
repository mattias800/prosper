#pragma once
#include <cstdint>
#include <vector>

namespace prosper::test {
// Cold shader-module creation gate, after the device feature has actually been enabled.
bool render_reconvergence_supported(const std::vector<uint32_t>& module, bool device_enabled,
                                    uint64_t program = 0, uint64_t identity = 0);
} // namespace prosper::test
