// A branch beyond the first END must be checked against its actual retained original trailer.
#include "gpu/recompiler/rdna2_native_parameter_base.hpp"
#include "gpu/recompiler/rdna2_fragment_terminal_tail.hpp"
#include "gpu/recompiler/pixel_parameter_launch.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace prosper::gpu {
std::shared_ptr<const NativeParameterBasePlan>
native_parameter_base_plan(const PixelParameterLaunch& launch, const uint32_t* code,
                           size_t dwords) {
    if (!code || !dwords || dwords > UINT32_MAX) return {};
    std::vector<Rdna2Inst> original;
    rdna2_walk(code, dwords, original);
    const auto tails = rdna2_fragment_terminal_tail(code, dwords, original);
    original.insert(original.end(), tails.instructions.begin(), tails.instructions.end());
    // These are immutable instruction boundaries only. Launch authority and all M0/numerical
    // observers are still checked by the original plan; no missing input value is supplied here.
    return native_parameter_base_plan(launch, original);
}
}   // namespace prosper::gpu
