// Project-owned two-phase compute program. Four guest waves use distinct values, paths and loop
// lengths; READLANE's EXEC-ignoring peers and guest LDS must survive the common phase schedule.
#pragma once

#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::test::readlane_barrier {
using namespace prosper::gpu;
inline constexpr uint32_t lanes = 256, sentinel = 0xdeadbeefu;
inline constexpr std::array<uint32_t, 4> trips{0, 3, 1, 4};
enum class Unsafe { None, CrossingEdge, NonuniformGuard, EarlyTrap, IndirectPc, M0, Writelane };
struct Options {
    bool loop = true, differing_peers = true, narrowed_readlane = false, barrier = true;
    Unsafe unsafe = Unsafe::None;
};
struct Program {
    std::vector<uint32_t> words;
    size_t barrier_pc = 0, peer_pc = 0;
};
inline uint32_t data(uint32_t lane) {
    return 0x51000000u + lane * 17u;
}
inline std::vector<uint32_t> input() {
    std::vector<uint32_t> result(static_cast<size_t>(lanes) * 4u, sentinel);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        result[lane] = data(lane);
        result[lanes + lane] = trips[lane / 64u];
    }
    return result;
}
inline std::vector<uint32_t> expected(Options options = {}) {
    auto result = input();
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint32_t wave = lane / 64u, base = wave * 64u;
        const uint32_t peer = options.differing_peers && wave % 2u ? 63u : 31u;
        const uint32_t added = options.loop ? trips[wave] : 0u;
        result[lanes * 3u + lane] = data(lane) + added + data(base + peer) + added +
                                    data(base + 31u) + (options.narrowed_readlane ? added : 0u);
    }
    return result;
}
inline Program program(Options options = {}) {
    Program result;
    auto& p = result.words;
    p.reserve(100);
    const auto branch = [&](uint32_t opcode) {
        const size_t at = p.size();
        p.push_back(0xbf800000u | (opcode << 16u));
        return at;
    };
    const auto target = [&](size_t at, size_t destination) {
        const auto delta = static_cast<int32_t>(destination) - static_cast<int32_t>(at + 1u);
        p[at] = (p[at] & 0xffff0000u) | static_cast<uint16_t>(delta);
    };
    const auto add = [&](uint32_t dst, uint32_t source, uint32_t value) {
        p.insert(p.end(), {0x4a0000ffu | (dst << 17u) | (source << 9u), value});
    };
    const auto readlane = [&](uint32_t dst, uint32_t source, uint32_t peer) {
        p.insert(p.end(), {0xd7600000u | dst, (256u + source) | ((128u + peer) << 9u)});
    };
    const auto load = [&](uint32_t dst, uint32_t index) {
        p.insert(p.end(), {0xe0302000u, 0x80020000u | (dst << 8u) | index});
    };
    load(1, 0);
    add(7, 0, lanes);
    load(2, 7);
    readlane(24, 0, 0);   // wave base, genuine and identical in each guest wave
    p.push_back(0x90188618u);   // s_lshr_b32 s24,s24,6
    p.push_back(0x871a8118u);   // s_and_b32 s26,s24,1
    readlane(20, 1, 31);   // retain this scalar value through the phase boundary
    readlane(22, 2, 31);   // genuine per-wave loop count
    size_t uniform_guard = 0;
    if (options.unsafe == Unsafe::NonuniformGuard) {
        p.push_back(0xbf068018u);   // s_cmp_eq_u32 s24,0: varies between guest waves
        uniform_guard = branch(5);
    }
    size_t finished = 0;
    if (options.loop) {
        p.push_back(0xbf068016u);   // s_cmp_eq_u32 s22,0
        finished = branch(5);
        const size_t loop = p.size();
        add(1, 1, 1);
        p.push_back(0x80968116u);   // s_sub_u32 s22,s22,1
        p.push_back(0xbf068016u);
        target(branch(4), loop);
        target(finished, p.size());
    }
    if (options.narrowed_readlane) {
        p.push_back(0x7da40080u);   // CMPX_EQ_U32 0,v0, only local0 active
        readlane(20, 1, 31);   // inactive selected peer still supplies its actual VGPR
        // The expected retained scalar is the post-loop peer value in this arm.
    }
    p.push_back(0xbefe04c1u);   // restore full EXEC before every LDS writer
    p.insert(p.end(), {0xd5690005u, 256u | (132u << 9u)});   // v5=v0*4 byte address
    p.insert(p.end(), {0xd8340000u, (1u << 8u) | 5u});   // DS_WRITE_B32 v5,v1
    if (options.unsafe == Unsafe::M0) p.push_back(0xbe9e037cu);   // s30=opaque entry M0
    if (options.unsafe == Unsafe::Writelane)
        p.insert(p.end(), {0xd761000au, 20u | (159u << 9u)});   // WRITELANE v10,s20,31
    if (options.unsafe == Unsafe::EarlyTrap) p.push_back(0xbf920001u);
    if (options.unsafe == Unsafe::IndirectPc) p.push_back(0xbe802018u);
    p.push_back(0xbf8c0000u);   // drain guest memory counters before the barrier
    result.barrier_pc = p.size();
    if (options.barrier) p.push_back(0xbf8a0000u);
    p.insert(p.end(), {0xd8d80000u, (3u << 24u) | 5u});   // DS_READ_B32 v3,v5
    p.push_back(0xbf8c0000u);
    p.push_back(0xbf06801au);   // s_cmp_eq_u32 s26,0
    const size_t alternate = branch(4);
    result.peer_pc = p.size();
    readlane(21, 3, 31);
    const size_t merge = branch(2);
    target(alternate, p.size());
    readlane(21, 3, options.differing_peers ? 63u : 31u);
    target(merge, p.size());
    p.push_back(0x4a080615u);   // v_add_nc_u32 v4,s21,v3
    p.push_back(0x4a080814u);   // v_add_nc_u32 v4,s20,v4
    if (options.unsafe == Unsafe::M0) p.push_back(0x4a08081eu);   // numeric s30 read after split
    add(7, 0, lanes * 3u);
    p.insert(p.end(), {0xe0702000u, 0x80020407u});
    const size_t end = p.size();
    p.push_back(0xbf810000u);
    if (options.unsafe == Unsafe::CrossingEdge && options.loop) target(finished, end);
    if (options.unsafe == Unsafe::NonuniformGuard) target(uniform_guard, end);
    return result;
}
inline ShaderResourceTable table() {
    ShaderResource resource;
    resource.cls = ResourceClass::ConstantBuffer;
    resource.format = DataFormat::Uint32;
    resource.num_components = 1;
    resource.binding = 3;
    resource.sgpr_base = 8;
    resource.stride = 4;
    resource.size = lanes * 16u;
    ShaderResourceTable result;
    result.resources.push_back(resource);
    return result;
}
inline ComputeShaderConfig config() {
    ComputeShaderConfig result;
    result.local_x = lanes;
    result.local_y = result.local_z = 1;
    result.wave_size = 64;
    result.lds_bytes = lanes * 4u;
    result.threads_x = lanes;
    result.threads_y = result.threads_z = 1;
    return result;
}
inline uint64_t address() {
    static std::atomic<uint64_t> next{0x4907000000ull};
    return next.fetch_add(0x100, std::memory_order_relaxed);
}
inline std::vector<uint32_t> compile(const Program& p, const ShaderResourceTable& resources,
                                     uint64_t diagnostic_address = address()) {
    return recompile_compute(p.words.data(), p.words.size(), &resources, config(),
                             {RecompileDiagnosticStage::Compute, diagnostic_address});
}
}   // namespace prosper::test::readlane_barrier
