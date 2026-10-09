#pragma once
// ADR 0028 route 3: project-owned raw compute programs with a readlane inside a loop, and the
// integer oracles that go with them. Hand-assembled RDNA2 words; no title data, no emitter output.
//
// Each lane loads data[lane] (v1), runs the loop, and stores v4 at out[lane]. Inside the loop every
// guest wave reads lane 5 of its OWN wave (`v_readlane s6, v1, 5`) and accumulates it, then
// v1 += 1:
//
//     v4 = sum over i in [0, trips) of (data[wave * 64 + 5] + i)
//
// The expected values below are computed from that formula, not by running anything.
#include <cstdint>
#include <vector>

#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

namespace prosper::test::wave64_exchange {

constexpr uint32_t kSentinel = 0xdeadbeefu;

enum class Trips : uint8_t {
    Constant3,   // every wave iterates three times (s_mov s0, 3)
    PerWave,   // wave w iterates w + 1 times (s0 = readlane(v0, 0) >> 6, + 1)
};

enum class Extra : uint8_t { None, Ballot, Mbcnt };   // a second cross-lane op in the loop

struct Case {
    uint32_t local = 64;
    Trips trips = Trips::Constant3;
    Extra extra = Extra::None;
    uint32_t lane = 5;   // the lane every wave reads (0..63)
    bool sgpr_lane =
        false;   // the selector is an SGPR (s_mov s7, lane) rather than an inline constant
    uint32_t threads = 0;   // 0 = a whole number of workgroups (threads == local)
    bool barrier_in_loop =
        false;   // an s_barrier inside the loop body: no dispatcher case can hold it
};

inline uint32_t data(uint32_t lane) {
    return 0x100u + lane * 3u;
}

inline uint32_t trip_count(const Case& c, uint32_t lane) {
    return c.trips == Trips::Constant3 ? 3u : lane / 64u + 1u;
}

// Input layout (words): [0, local) data, then [local, 3 * local) unused, [3 * local, 4 * local) out.
inline std::vector<uint32_t> input(const Case& c) {
    std::vector<uint32_t> v(c.local * 4, kSentinel);
    for (uint32_t lane = 0; lane < c.local; ++lane) v[lane] = data(lane);
    return v;
}

inline std::vector<uint32_t> expected(const Case& c) {
    auto v = input(c);
    for (uint32_t lane = 0; lane < c.local; ++lane) {
        const uint32_t base = lane / 64u * 64u;
        uint32_t sum = 0;
        for (uint32_t i = 0; i < trip_count(c, lane); ++i) {
            sum += data(base + c.lane) + i;
            if (c.extra == Extra::Ballot) sum += 64u;   // every lane of the wave votes
            if (c.extra == Extra::Mbcnt)
                sum += (lane % 64u) < 32u ? lane % 64u : 32u;   // low-half lanes below
        }
        v[c.local * 3 + lane] = sum;
    }
    return v;
}

inline std::vector<uint32_t> program(const Case& c) {
    std::vector<uint32_t> p;
    // v1 = cbuf[v0]   (buffer_load_dword, idxen, stride 4)
    p.insert(p.end(), {0xe0302000u, 0x80020000u | (1u << 8) | 0u});
    if (c.trips == Trips::PerWave) {
        p.insert(p.end(), {0xd7600000u, 0x00010100u,   // v_readlane_b32 s0, v0, 0
                           0x90008600u,   // s_lshr_b32 s0, s0, 6
                           0x80008100u});   // s_add_u32 s0, s0, 1
    } else {
        p.push_back(0xbe800383u);   // s_mov_b32 s0, 3
    }
    p.push_back(0x7e080280u);   // v_mov_b32 v4, 0
    if (c.sgpr_lane) p.push_back(0xbe870380u | ((128u + c.lane) & 0xffu));   // s_mov_b32 s7, lane
    const uint32_t loop_pc = static_cast<uint32_t>(p.size());
    if (c.barrier_in_loop) p.push_back(0xbf8a0000u);   // s_barrier
    p.insert(p.end(),
             {0xd7600006u,
              c.sgpr_lane ? 0x00000f01u
                          : (0x101u | ((128u + c.lane) << 9)),   // v_readlane_b32 s6, v1, lane
              0x4a080806u,   // v_add_nc_u32 v4, s6, v4
              0x4a020281u,   // v_add_nc_u32 v1, 1, v1
              0x80808100u,   // s_sub_u32 s0, s0, 1
              0xbf078000u});   // s_cmp_lg_u32 s0, 0
    if (c.extra == Extra::Ballot)
        p.insert(p.end(), {0x7d840100u, 0xbe88106au,
                           0x4a080808u});   // v_cmp vcc ; s_bcnt1 s8, vcc ; v4 += s8
    if (c.extra == Extra::Mbcnt)
        p.insert(p.end(),
                 {0xd7650007u, 0x0001007eu, 0x4a080907u});   // v_mbcnt_lo v7, exec_lo, 0 ; v4 += v7
    const int32_t displacement = static_cast<int32_t>(loop_pc) - static_cast<int32_t>(p.size() + 1);
    p.push_back(0xbf850000u | (static_cast<uint32_t>(displacement) & 0xffffu));   // s_cbranch_scc1
    // v5 = v0 + 3 * local ; cbuf[v5] = v4
    p.insert(p.end(), {0x4a0000ffu | (5u << 17) | (0u << 9), c.local * 3u, 0xe0702000u, 0x80020405u,
                       0xbf810000u});
    return p;
}

inline prosper::gpu::ShaderResourceTable resources(const Case& c) {
    prosper::gpu::ShaderResourceTable rt;
    prosper::gpu::ShaderResource r;
    r.cls = prosper::gpu::ResourceClass::ConstantBuffer;
    r.format = prosper::gpu::DataFormat::Uint32;
    r.num_components = 1;
    r.binding = 3;
    r.stride = 4;
    r.sgpr_base = 8;
    r.size = c.local * 16;
    rt.resources.push_back(r);
    return rt;
}

// `exchange_width` is ComputeShaderConfig::wave64_exchange_width: 0 = route off (the control arm).
inline std::vector<uint32_t> compile(const Case& c, uint32_t exchange_width) {
    const auto p = program(c);
    const auto rt = resources(c);
    prosper::gpu::ComputeShaderConfig cfg;
    cfg.local_x = c.local;
    cfg.wave_size = 64;
    cfg.threads_x = c.threads ? c.threads : c.local;
    cfg.exact_thread_extent = c.threads != 0;
    cfg.threads_y = cfg.threads_z = 1;
    cfg.wave64_exchange_width = exchange_width;
    return prosper::gpu::recompile_compute(
        p.data(), p.size(), &rt, cfg, {prosper::gpu::RecompileDiagnosticStage::Compute, 0x5028u});
}

}   // namespace prosper::test::wave64_exchange
