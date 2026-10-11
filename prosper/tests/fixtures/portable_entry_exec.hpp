#pragma once

#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::test::portable_entry_exec {
using namespace prosper::gpu;
inline constexpr uint32_t sentinel = 0xdeadbeef, planes = 6;
inline ComputeShaderConfig config(uint32_t lanes = 256) {
    ComputeShaderConfig c;
    c.local_x = c.threads_x = lanes;
    c.local_y = c.local_z = c.threads_y = c.threads_z = 1;
    c.wave_size = 64;
    c.lds_bytes = 4;
    return c;
}
inline ShaderResourceTable table(uint32_t lanes = 256) {
    ShaderResource r;
    r.cls = ResourceClass::ConstantBuffer;
    r.format = DataFormat::Uint32;
    r.num_components = 1;
    r.binding = 3;
    r.sgpr_base = 8;
    r.stride = 4;
    r.size = uint64_t{lanes} * planes * 4;
    ShaderResourceTable t;
    t.resources.push_back(r);
    return t;
}
inline std::vector<uint32_t> program(uint32_t lanes = 256, bool phased = false) {
    // Project-owned integer program. Negating the complete entry EXEC yields low=1, high=0;
    // the high SUBB still borrows. Save those values before any later mask generation.
    std::vector<uint32_t> p{0x80927e80u, 0x82937f80u, 0x85168081u};
    const auto mov = [&](uint32_t v, uint32_t source) {
        p.push_back(0x7e000200u | (v << 17u) | source);
    };
    const auto store = [&](uint32_t v, uint32_t plane) {
        p.insert(p.end(), {0x4a0e00ffu, lanes * plane}); // v7 = v0 + plane base
        p.insert(p.end(), {0xe0702000u, 0x80020007u | (v << 8u)});
    };
    mov(1, 18);
    mov(2, 19);
    mov(3, 22);
    store(1, 0);
    store(2, 1);
    store(3, 2);
    p.push_back(0xbefe0412u); // EXEC = saved genuine numeric pair, only guest lane0
    mov(4, 137); // inline9
    store(4, 3);
    p.push_back(0xbefe04c1u); // restore all guest lanes
    mov(5, 255);
    p.push_back(0x42u);
    store(5, 4);
    p.insert(p.end(), {0xd7600014u, 256u | (159u << 9u)}); // s20 = READLANE(v0,31)
    if (phased) p.insert(p.end(), {0xbf8c0000u, 0xbf8a0000u});
    mov(6, 20);
    store(6, 5);
    p.push_back(0xbf810000u);
    return p;
}
inline std::vector<uint32_t> expected(uint32_t lanes = 256) {
    std::vector<uint32_t> data(static_cast<size_t>(lanes) * planes, sentinel);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        data[lane] = 1;
        data[lanes + lane] = 0;
        data[2 * lanes + lane] = 1; // independent 64-bit unsigned borrow: 0 < UINT64_MAX
        if (lane % 64 == 0) data[3 * lanes + lane] = 9;
        data[4 * lanes + lane] = 0x42;
        data[5 * lanes + lane] = lane / 64 * 64 + 31;
    }
    return data;
}
inline uint64_t address() {
    static std::atomic<uint64_t> next{0x4930000000ull};
    return next.fetch_add(0x100, std::memory_order_relaxed);
}
inline std::vector<uint32_t> compile(const std::vector<uint32_t>& p,
                                     const ComputeShaderConfig& c = config(),
                                     uint64_t diagnostic = address()) {
    const auto resources = table(c.local_x);
    return recompile_compute(p.data(), p.size(), &resources, c,
                             {RecompileDiagnosticStage::Compute, diagnostic});
}
} // namespace prosper::test::portable_entry_exec
