#pragma once
#include <cstdint>
#include <vector>
namespace prosper::test {
struct RenderReconvergenceFeatures {
    bool maximal = false, full_quads = false;
};
// Cold module gate, using features actually enabled on the device.
bool render_reconvergence_supported(const std::vector<uint32_t>& module,
                                    RenderReconvergenceFeatures enabled, uint64_t program = 0,
                                    uint64_t identity = 0);
} // namespace prosper::test
