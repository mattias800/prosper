// Standalone fragment tails omit their sampled/interpolated input-producing prologues. These
// owned reductions select real FragCoord x/y as v0/v1 so mask tests never consume fabricated zero.
// The optional depth input supplies v2 for the varying-loop reductions. This setup supplies
// named pixel inputs, not arbitrary register or resource authority.
#pragma once

#include "fixtures/fragment_compare_kill.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace prosper::test {
inline std::vector<uint32_t>
compile_fragment_control_flow_fixture(std::span<const uint32_t> code,
                                      prosper::gpu::RecompileDiagnosticContext diagnostic =
                                          {prosper::gpu::RecompileDiagnosticStage::Fragment, 0},
                                      bool depth_input = false) {
    auto inputs = fragment_compare_kill::position_inputs();
    if (depth_input) inputs.ena = inputs.addr = inputs.ena | (1u << 10);
    return prosper::gpu::recompile_fragment(code.data(), code.size(), nullptr, &inputs, UINT32_MAX,
                                            nullptr, false, diagnostic);
}
} // namespace prosper::test
