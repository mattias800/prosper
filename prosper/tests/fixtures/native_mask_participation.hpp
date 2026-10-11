// Project-owned complete Wave64 programs: current physical mask words vary at each loop epoch.
#pragma once
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <vector>

namespace prosper::test::native_mask_participation {
using namespace prosper::gpu;
inline constexpr uint32_t lanes = 64, group_words = 2048, sentinel = 0xdeadbeefu;
inline constexpr std::array<uint32_t, 4> trips{0, 1, 3, 5};
enum class Unsafe { None, RelativeBranch, RelativePredicate, ReadfirstBranch, Writelane };
enum class Preservation { None, DppBoundary, PermlaneInactive, TransformedLane, ImplicitVcc };
struct Options {
    bool exec_off_peer = false, opaque_prefix = false, masked_load = false;
    Unsafe unsafe = Unsafe::None;
    Preservation preservation = Preservation::None;
    bool initialize_preserved = false;
};
struct Program {
    std::vector<uint32_t> words;
    uint32_t producer_pc = 0;
};
inline uint32_t mov(uint32_t dst, uint32_t source, bool pair = false) {
    return 0xbe800000u | (dst << 16u) | ((pair ? 4u : 3u) << 8u) | source;
}
inline Program program(Options options = {}) {
    Program result;
    auto& p = result.words;
    p.reserve(160);
    const auto scalar = [&](uint32_t op, uint32_t dst, uint32_t a, uint32_t c) {
        p.push_back(0x80000000u | (op << 23u) | (dst << 16u) | (c << 8u) | a);
    };
    const auto branch = [&](uint32_t op) {
        const size_t at = p.size();
        p.push_back(0xbf800000u | (op << 16u));
        return at;
    };
    const auto target = [&](size_t at, size_t to) {
        p[at] |= uint16_t(int32_t(to) - int32_t(at + 1u));
    };
    const auto vadd = [&](uint32_t dst, uint32_t a, uint32_t v) {
        p.push_back(0x4a000000u | (dst << 17u) | (v << 9u) | a);
    };
    const auto readlane = [&](uint32_t dst, uint32_t v, uint32_t peer) {
        p.insert(p.end(), {0xd7600000u | dst, (256u + v) | ((128u + peer) << 9u)});
    };
    const auto compare = [&](uint32_t op, uint32_t dst, uint32_t a, uint32_t v) {
        p.insert(p.end(), {0x7c0000f9u | (op << 17u) | (v << 9u),
                           (a >= 256u ? 0x06068000u : 0x06868000u) | (dst << 8u) | (a & 255u)});
    };
    scalar(0x1e, 13, 12, 139); // group ID *2048
    vadd(1, 13, 0);
    p.insert(p.end(), {0xe0302000u, 0x80020201u}); // v2=buffer[v1], real per-lane inputs
    if (options.exec_off_peer) p.push_back(0x7da40080u); // guest lane0 EXEC
    readlane(20, 2, 63);
    p.push_back(mov(126, 193, true)); // complete guest EXEC restored
    if (options.opaque_prefix || options.unsafe == Unsafe::RelativeBranch ||
        options.unsafe == Unsafe::RelativePredicate) {
        p.push_back(mov(124, 148));
        p.push_back(
            0x7e148720u); // v10=MOVRELS v32+M0: absent semantic source, not DATA authority
        readlane(22, 10, 31); // routing equality only
    }
    if (options.unsafe == Unsafe::ReadfirstBranch)
        p.push_back(0x7e2c0500u); // s22=READFIRSTLANE v0
    if (options.unsafe == Unsafe::RelativeBranch || options.unsafe == Unsafe::ReadfirstBranch) {
        p.push_back(0xbf068016u);
        const size_t conditional = branch(4);
        p.push_back(mov(23, 129));
        target(conditional, p.size());
    }
    if (options.unsafe == Unsafe::Writelane) p.insert(p.end(), {0xd761000au, 20u | (159u << 9u)});
    p.push_back(mov(21, 128));
    p.push_back(mov(30, 128)); // real preheader word; no unrelated missing-phi placeholder
    if (options.initialize_preserved) p.push_back(0x7e160280u); // v11=0, every lane
    if (options.preservation == Preservation::TransformedLane && options.initialize_preserved)
        p.push_back(0x7e180280u); // selected peer has real old v12
    if (options.preservation == Preservation::ImplicitVcc) {
        p.push_back(mov(106, 128));
        p.push_back(mov(107, 128)); // old physical VCC words must not outlive a new mask
    }
    if (options.preservation == Preservation::DppBoundary)
        p.insert(p.end(), {0x7e1602fau, 0xff011100u});   // row_shr1 BC0 preserves old v11 at edges
    if (options.preservation == Preservation::PermlaneInactive) {
        p.push_back(mov(126, 129, true));   // only lane0 active
        p.insert(p.end(), {0xd378000bu, 256u | (128u << 9u) | (128u << 18u)});
        p.push_back(mov(126, 193, true));   // peer16 was inactive, FI0/BC0 preserves old v11
    }
    if (options.preservation == Preservation::TransformedLane) {
        p.insert(p.end(), {0x7e1602fau, 0xff091100u});   // row_shr1 BC1 is not local-X identity
        p.push_back(0x7da41781u);   // CMPX v11==1: actual lane2, not lane1
        p.push_back(0x7e180281u);   // v12=1 on selected lanes only
        p.push_back(mov(126, 193, true));
        readlane(22, 12, 1);
    }
    if (options.preservation == Preservation::ImplicitVcc) {
        p.push_back(0x7d841580u);   // implicit VCC compare uses absent v10
        if (options.initialize_preserved) {
            p.push_back(mov(106, 6));
            p.push_back(mov(107, 7));   // genuine runtime words avoid constant-branch rewriting
        }
    }
    if (options.preservation == Preservation::TransformedLane ||
        options.preservation == Preservation::ImplicitVcc) {
        p.push_back(options.preservation == Preservation::ImplicitVcc ? 0xbf06806au : 0xbf068016u);
        const size_t conditional = branch(4);
        p.push_back(mov(23, 129));
        target(conditional, p.size());
        p.push_back(0xbf800000u);   // real merge instruction, distinct from the loop header
    }
    const size_t loop = p.size();
    p.push_back(0xbf0a1415u);   // top-tested loop: s21<s20, including zero iterations
    const size_t finished = branch(4);
    p.insert(p.end(), {mov(40, 255), 0x11223344u, mov(41, 255), 0x55667788u});
    if (options.masked_load) {
        // Initialize every lane, then replace only the chosen lanes with real memory words.
        p.push_back(0x7e080280u);   // v4=0
        compare(0xc3, 34, 160, 0);   // 32<=v0
        p.push_back(mov(106, 126, true));
        p.push_back(0xbeea2422u);   // AND_SAVEEXEC VCC,s34: save old full EXEC
        p.insert(p.end(), {0xe0302000u | 256u, 0x80020401u});   // real upper-lane words from plane1
        p.push_back(mov(126, 106, true));
    } else {
        // Fresh owned threshold plane at each epoch, beyond every output plane.
        scalar(0x1e, 31, 21, 134);   // iteration *64 dwords
        vadd(9, 31, 1);
        vadd(9, 255, 9);
        p.push_back(24u * lanes);
        p.insert(p.end(), {0xe0302000u, 0x80020409u});   // v4=threshold[iteration,lane]
    }
    result.producer_pc = static_cast<uint32_t>(p.size());
    compare(options.masked_load ? 0xc2 : 0xc1, 40, options.masked_load ? 130u : 256u,
            options.unsafe == Unsafe::RelativePredicate ? 10u
            : options.preservation == Preservation::DppBoundary ||
                    options.preservation == Preservation::PermlaneInactive
                ? 11u
                : 4u);
    p.push_back(0x7e0a0228u);   // v5=physical LO
    p.push_back(0x7e0c0229u);   // v6=physical HI
    p.push_back(0x7e0e7229u);   // v7=FFBH HI
    p.push_back(0x7e107228u);   // v8=FFBH LO
    scalar(0x1e, 31, 21, 136);   // 4 planes per iteration, in dwords
    vadd(9, 31, 1);
    for (uint32_t word = 0; word < 4; ++word)
        p.insert(p.end(), {0xe0702000u | (512u + word * 256u), 0x80020009u | ((5u + word) << 8u)});
    scalar(0, 21, 21, 129);
    target(branch(2), loop);
    target(finished, p.size());
    p.push_back(0xbf810000u);
    return result;
}
inline std::vector<uint32_t> input() {
    std::vector<uint32_t> result(group_words * trips.size(), sentinel);
    for (uint32_t group = 0; group < trips.size(); ++group)
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            result[group * group_words + lane] = lane == 63 ? trips[group] : 10u + lane;
            result[group * group_words + lanes + lane] = lane >= 32 && lane % 3u != 0u ? 2u : 1u;
            for (uint32_t trip = 0; trip < 5; ++trip)
                result[group * group_words + (24u + trip) * lanes + lane] = 33u + trip * 7u;
        }
    return result;
}
inline std::vector<uint32_t> expected(Options options = {}) {
    auto result = input();
    for (uint32_t group = 0; group < trips.size(); ++group)
        for (uint32_t trip = 0; trip < trips[group]; ++trip) {
            uint64_t mask = 0;
            for (uint32_t lane = 0; lane < lanes; ++lane)
                if (options.masked_load ? lane >= 32 && lane % 3u != 0u : lane < 33u + trip * 7u)
                    mask |= uint64_t{1} << lane;
            const uint32_t low = uint32_t(mask), high = uint32_t(mask >> 32u);
            const std::array<uint32_t, 4> values{
                low, high, high ? uint32_t(std::countl_zero(high)) : UINT32_MAX,
                low ? uint32_t(std::countl_zero(low)) : UINT32_MAX};
            for (uint32_t word = 0; word < 4; ++word)
                for (uint32_t lane = 0; lane < lanes; ++lane)
                    result[group * group_words + (2u + trip * 4u + word) * lanes + lane] =
                        values[word];
        }
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
    resource.size = group_words * trips.size() * 4u;
    ShaderResourceTable result;
    result.resources.push_back(resource);
    return result;
}
inline ComputeShaderConfig config() {
    ComputeShaderConfig result;
    result.local_x = result.threads_x = lanes;
    result.local_y = result.local_z = result.threads_y = result.threads_z = 1;
    result.wave_size = result.native_subgroup_size = 64;
    result.tgid_x_en = true;
    result.user_sgprs.resize(12);
    return result;
}
inline uint64_t address() {
    static std::atomic<uint64_t> next{0x4919000000ull};
    return next.fetch_add(0x100, std::memory_order_relaxed);
}
inline std::vector<uint32_t> compile(const Program& p, const ShaderResourceTable& resources,
                                     uint64_t addr, const ComputeShaderConfig& cfg = config()) {
    return recompile_compute(p.words.data(), p.words.size(), &resources, cfg,
                             {RecompileDiagnosticStage::Compute, addr});
}
}   // namespace prosper::test::native_mask_participation
