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
    0xbe880300u, // s8=the declared workgroup-X input in s0; neither scalar branch is folded.
    0x7e000280u,
    0x7e020281u,
    0x7e040281u,
    0x7e060280u, // v0/v1/v2/v3
    0x7e1e0281u,
    0x7e360282u,
    0x7e400280u, // v15/v27/v32
    // The current carry-source inventory conservatively asks for the adjacent physical word.
    // Define s1 explicitly to keep these fixtures focused on their ambiguous s0 join. This is
    // no claim that the Wave32 carry instruction architecturally consumes that high word.
    0xbe810380u, // s1=0; the original SDWA comparison still replaces s0 with a mask.
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
                                      bool depth_input = false, bool wave32 = false) {
    auto inputs = fragment_compare_kill::position_inputs();
    if (depth_input) inputs.ena = inputs.addr = inputs.ena | (1u << 10);
    return prosper::gpu::recompile_fragment(code.data(), code.size(), nullptr, &inputs, UINT32_MAX,
                                            nullptr, wave32, diagnostic);
}
inline std::vector<uint32_t> compile_wave32_fragment_control_flow_fixture(
    std::span<const uint32_t> code, prosper::gpu::RecompileDiagnosticContext diagnostic = {
                                        prosper::gpu::RecompileDiagnosticStage::Fragment, 0}) {
    return compile_fragment_control_flow_fixture(code, diagnostic, false, true);
}

// The isolated wave-pack/forward-if reduction tests PHI construction, not numeric VCC projection.
// Supply a real mask-only VCC producer at its original SAVEEXEC consumer; the numeric original
// remains a separate refused contract. Every original wave-pack instruction stays in place.
inline std::vector<uint32_t>
compile_ngg_phi_fixture_with_declared_predicate(std::span<const uint32_t> body) {
    if (body.size() != 30 || body[14] != 0xbe80246au) return {};
    std::vector<uint32_t> code(body.begin(), body.begin() + 14);
    code.push_back(0xbeea047eu);   // s_mov_b64 vcc, exec: genuine predicate for the forward region.
    code.insert(code.end(), body.begin() + 14, body.end());
    return prosper::gpu::recompile_vertex(code.data(), code.size());
}
} // namespace prosper::test
