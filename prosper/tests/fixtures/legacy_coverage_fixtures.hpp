// Full synthetic coverage fixtures, shared by the legacy assertions and producing regressions.
// Explicit producers precede their first consumers; packet edits retain the original instruction sizes.
#pragma once
#include "compute_coverage_inputs.hpp"
#include "checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdint>
#include <iterator>
#include <vector>

namespace prosper::test::coverage_fixtures {
inline constexpr uint32_t b32_compare_word = 1;
inline constexpr uint32_t cndmask_source_word = 13;
inline constexpr uint32_t lshl_or_source_word = 16;
inline constexpr uint32_t lshl_or_tail_word = 18;
inline std::vector<uint32_t> append_dispatch(std::vector<uint32_t> prefix) {
    prefix.insert(prefix.end(), std::begin(coverage_inputs::dispatch_tail),
                  std::end(coverage_inputs::dispatch_tail));
    return prefix;
}
inline std::vector<uint32_t> b32_alias() {
    // Compare real local-ID v0 with 1, retaining different lane outcomes and distinct sources.
    return append_dispatch({0x7e020281u, 0x7d8802f9u, 0x06068000u, 0xbe820300u});
}
inline std::vector<uint32_t> cndmask_scalar() {
    return append_dispatch({0x7e140300u, 0xbf068000u, 0xbf840003u, 0xbe920381u, 0xbe930380u,
                            0xbf820001u, 0xbe920a7eu, 0xbe910381u, 0x7e300280u, 0x7e2a0280u,
                            0x7d8414f9u, 0x06868e82u, 0xd5010015u, 0x00382318u});
}
inline std::vector<uint32_t> lshl_or_scalar() {
    auto code = append_dispatch({0x7e0e0300u, 0xbf068000u, 0xbf840003u, 0xbe880381u, 0xbe890380u,
                                 0xbf820001u, 0xbe880a7eu, 0x7e000280u, 0x7e100500u, 0xbe920384u,
                                 0xbe930382u, 0xbe940384u, 0xbe950382u, 0xbf068008u, 0x85ea1412u,
                                 0xd76f0002u, 0x01a92008u, 0xbe880480u});
    // s8 is scratch in this prefix. Remove the whole unrelated descriptor-load packet while
    // retaining the preceding saved-EXEC restoration and the tail's PC-relative branches.
    code[lshl_or_tail_word + 13] = 0xbf800000u;
    code[lshl_or_tail_word + 14] = 0xbf800000u;
    return code;
}
inline std::vector<uint32_t> trap_bypass() {
    return append_dispatch({0xbe800380u, 0xbf070000u, 0xbf840001u, 0xbf920001u});
}
inline gpu::ShaderResourceTable dispatch_resources() {
    gpu::ShaderResourceTable table;
    gpu::ShaderResource resource{};
    resource.cls = gpu::ResourceClass::VertexBuffer;
    resource.binding = 3;
    resource.sgpr_base = 8;
    resource.stride = 16;
    resource.format = gpu::DataFormat::Float32;
    resource.num_components = 4;
    table.resources.push_back(resource);
    return table;
}
inline gpu::ComputeShaderConfig wave32(bool native) {
    gpu::ComputeShaderConfig config;
    config.wave_size = 32;
    config.native_subgroup_size = native ? 32u : 0u;
    return config;
}
inline gpu::ComputeShaderConfig wave64() {
    auto config = wave32(true);
    config.wave_size = config.native_subgroup_size = 64;
    complete_compute_launch(config);
    return config;
}
// These branch fixtures consume an actual push-constant numeric word s0, separate from the
// direct descriptor s[8:11]. A runtime value retains both mask/scalar arms during compilation;
// an inline constant would fold the arm whose ambiguity the same-site negative must observe.
inline gpu::ComputeShaderConfig branch_wave64() {
    auto config = wave64();
    config.user_sgprs = {0};
    return config;
}
inline constexpr uint32_t nor_input_words = 1;
}   // namespace prosper::test::coverage_fixtures
