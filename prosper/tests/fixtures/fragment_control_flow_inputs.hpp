// Standalone fragment tails omit their sampled/interpolated input-producing prologues. These
// owned reductions select real FragCoord x/y as v0/v1 so mask tests never consume fabricated zero.
// The optional depth input supplies v2 for the varying-loop reductions. This setup supplies
// named pixel inputs, not arbitrary register or resource authority.
#pragma once

#include "fixtures/fragment_compare_kill.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace prosper::test {
// Both reduced two-source CFG fixtures retain their original ambiguous s0 join. These
// instructions supply their actual arithmetic, comparison and independent branch inputs.
inline constexpr std::array two_source_cfg_inputs{
    0xbe880380u, // s8=0
    0x7e000280u, 0x7e020281u, 0x7e040281u, 0x7e060280u, // v0/v1/v2/v3
    0x7e1e0281u, 0x7e360282u, 0x7e400280u, // v15/v27/v32
};
inline std::vector<uint32_t> declared_two_source_cfg_fixture(std::span<const uint32_t> body) {
    std::vector<uint32_t> code(two_source_cfg_inputs.begin(), two_source_cfg_inputs.end());
    code.insert(code.end(), body.begin(), body.end());
    return code;
}
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
