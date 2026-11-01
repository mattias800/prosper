#pragma once
// LLVM gfx1030: save entry EXEC, cross a counted loop, copy it to VCC, then CMPX
// and overwrite LOW only. HIGH must keep the actual launch mask even in a partial group.
#include "fixtures/gta5_compute_cfg_fixture.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace prosper::test::native_entry_exec_words {
using Words = std::vector<uint32_t>;
inline constexpr uint32_t untouched = 0xdeadbeefu, old = 0xa5a5a5a5u;
inline Words program(bool data_mask = false) {
    if (data_mask)
        return {0xbe82047eu, 0xbe840380u, 0x7e0402ffu, 0xa5a5a5a5u, 0x7e0602ffu,
                0xa5a5a5a5u, 0x80048104u, 0xbf0a8304u, 0xbf85fffdu, 0xbefe0402u,
                0xbeea047eu, 0x7da800a8u, 0x936a818du, 0xd5010002u, 0x01a90280u,
                0x7e06026bu, 0xbefe0402u, 0x8f0d870cu, 0x34020081u, 0x4a02020du,
                0xe0702000u, 0x80020201u, 0xe0702004u, 0x80020301u, 0xbf810000u};
    return {0xbe82047eu, 0xbe840380u, 0x7e0402ffu, 0xa5a5a5a5u, 0x7e0602ffu, 0xa5a5a5a5u,
            0x80048104u, 0xbf0a8304u, 0xbf85fffdu, 0xbefe0402u, 0xbeea047eu, 0x7da800a8u,
            0xbeea038du, 0x7e04026au, 0x7e06026bu, 0xbefe0402u, 0x8f0d870cu, 0x34020081u,
            0x4a02020du, 0xe0702000u, 0x80020201u, 0xe0702004u, 0x80020301u, 0xbf810000u};
}
inline Words readfirst_program(bool overwrite) {
    if (overwrite)
        return {0x7e1c0500u, 0x806a810eu, 0xbeeb0380u, 0xbeea0381u, 0xd5010002u,
                0x01a90280u, 0xe0702000u, 0x80020200u, 0xbf810000u};
    return {0x7e1c0500u, 0x806a810eu, 0xbeeb0380u, 0xd5010002u,
            0x01a90280u, 0xe0702000u, 0x80020200u, 0xbf810000u};
}
inline gpu::ComputeShaderConfig config(bool partial) {
    gpu::ComputeShaderConfig c;
    c.local_x = c.wave_size = c.native_subgroup_size = 64;
    c.user_sgprs.resize(12);
    c.tgid_x_en = true;
    c.threads_x = partial ? 4152 : 128;
    c.threads_y = c.threads_z = 1;
    c.exact_thread_extent = partial;
    c.dispatch_launch =
        partial ? gpu::test_fixture::kGta5Pc232OwnedLaunch
                : gpu::ComputeDispatchLaunchState{.known = gpu::ComputeDispatchLaunchState::kAll,
                                                  .initiator = 1,
                                                  .num_thread = {64, 1, 1},
                                                  .dimensions = {2, 1, 1}};
    return c;
}
inline Words output(bool partial) {
    const size_t workers = partial ? 4160 : 128;
    return Words(workers * 2, untouched);
}
inline gpu::ShaderResourceTable table(bool partial) {
    gpu::ShaderResource r;
    r.cls = gpu::ResourceClass::ConstantBuffer;
    r.format = gpu::DataFormat::Uint32;
    r.num_components = 1;
    r.binding = 3;
    r.sgpr_base = 8;
    r.stride = 4;
    r.size = output(partial).size() * sizeof(uint32_t);
    gpu::ShaderResourceTable result;
    result.resources.push_back(r);
    return result;
}
struct Compilation {
    Words module;
    std::string reason;
};
inline Compilation compile(const Words& code, const gpu::ComputeShaderConfig& c) {
    auto resources = table(c.exact_thread_extent);
    static std::atomic<uint64_t> next{0x5039ee0000ull};
    const auto address = next.fetch_add(0x100);
    auto module = gpu::recompile_compute(code.data(), code.size(), &resources, c,
                                         {gpu::RecompileDiagnosticStage::Compute, address});
    return {std::move(module), gpu::last_terminal_reject_reason(address)};
}
inline void expected_group(Words& values, bool partial, uint32_t group, bool data_mask = false) {
    const uint32_t live = partial && group == 64 ? 56 : 64;
    const uint32_t high = live == 64 ? UINT32_MAX : 0x00ffffffu;
    for (uint32_t lane = 0; lane < live; ++lane) {
        const uint32_t first = (group * 64 + lane) * 2;
        const uint32_t selected = lane >= 32 ? 1 : ((13u >> lane) & 1u);
        values[first] = lane < 40 ? (data_mask ? selected : 13) : old;
        values[first + 1] = lane < 40 ? high : old;
    }
}
} // namespace prosper::test::native_entry_exec_words
