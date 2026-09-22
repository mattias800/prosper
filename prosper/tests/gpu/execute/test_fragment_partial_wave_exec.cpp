// A Wave64 pixel shader using lane id, a ballot, MBCNT and votes, EXECUTED on whatever fragment
// subgroup this device has -- including one narrower than the guest wave (#3464).
//
// On a host that cannot supply 64 lanes (NVIDIA 32..32, lavapipe 8..8) the draw runs only through
// the partial-wave tier: each host subgroup is a guest wave whose upper lanes are unpopulated. The
// checks below are therefore written so they hold for ANY assignment of pixels to waves -- the one
// thing no host reproduces -- and fail for a lowering that answers some other wave's question:
//
//   * every pixel's own bit is set in the ballot of its entry EXEC (the read-back shape GTA V uses;
//     a lane reading the wrong half, or a ballot reporting another group's lanes, clears it);
//   * rank < population, and for each population size n the ranks 0..n-1 occur equally often -- each
//     wave of n lanes contributes exactly one pixel of every rank, whatever the grouping, so a scan
//     over a different set than the ballot counts breaks the histogram;
//   * exactly one lane per wave has rank 0, and a vote over that per-lane predicate is true;
//   * on a host of at most 32 lanes the high dword of EXEC is zero -- the wave really is partial,
//     which is the premise of the admission, measured rather than assumed.
//
// What this does NOT cover, so nobody reads it as more: no helper lanes (the fullscreen draw covers
// every quad), no subgroup that is less than full on NVIDIA (every wave here has 32), and no ballot
// of a NARROWED EXEC. At S=32 the MBCNT_HI source is always zero, so only its lane>=32 gate is
// exercised (the rank histogram does catch that gate being dropped). It tests the LOWERING; it
// cannot test the premise that grouping does not matter -- its oracle is grouping-invariant.
//
// Source: tests/fixtures/fragment_partial_wave.s.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/render_runner.h"
#include <cstdio>
#include <iterator>
#include <map>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kProgram[] = {
    0xbe80047eu,              // s_mov_b64 s[0:1], exec
    0xd7640005u, 0x00010000u, // v_bcnt_u32_b32 v5, s0, 0
    0xd7640005u, 0x00020a01u, // v_bcnt_u32_b32 v5, s1, v5
    0xd7650004u, 0x00010000u, // v_mbcnt_lo_u32_b32 v4, s0, 0
    0xd7660004u, 0x00020801u, // v_mbcnt_hi_u32_b32 v4, s1, v4
    0xd4c20008u, 0x00020880u, // v_cmp_eq_u32_e64 s[8:9], 0, v4
    0xd7640007u, 0x00010008u, // v_bcnt_u32_b32 v7, s8, 0
    0xd7640007u, 0x00020e09u, // v_bcnt_u32_b32 v7, s9, v7
    0x7d840e81u,              // v_cmp_eq_u32 vcc, 1, v7
    0xd5010008u, 0x01a90480u, // v_cndmask_b32_e64 v8, 0, 2, vcc
    0xbf138008u,              // s_cmp_lg_u64 s[8:9], 0
    0x85078084u,              // s_cselect_b32 s7, 4, 0
    0xd5010006u, 0x00010280u, // v_cndmask_b32_e64 v6, 0, 1, s[0:1]
    0xd7720006u, 0x001e1106u, // v_or3_b32 v6, v6, v8, s7
    0xbf068001u,              // s_cmp_eq_u32 s1, 0
    0x8503f0f2u,              // s_cselect_b32 s3, 1.0, 0.5
    0x7e000d04u,              // v_cvt_f32_u32 v0, v4
    0x100000ffu, 0x3b808081u, // v_mul_f32 v0, 0x3b808081, v0
    0x7e020d05u,              // v_cvt_f32_u32 v1, v5
    0x100202ffu, 0x3b808081u, // v_mul_f32 v1, 0x3b808081, v1
    0x7e040d06u,              // v_cvt_f32_u32 v2, v6
    0x100404ffu, 0x3b808081u, // v_mul_f32 v2, 0x3b808081, v2
    0x7e060203u,              // v_mov_b32 v3, s3
    0xf800180fu, 0x03020100u, // exp mrt0 v0, v1, v2, v3 done vm
    0xbf810000u,              // s_endpgm
};

// A DPP row read: another lane's register by index, which the tier deliberately does NOT admit.
constexpr uint32_t kDppProgram[] = {
    0x7e0002f2u,              // v_mov_b32 v0, 1.0
    0x1e0000fau, 0xff011100u, // v_min_f32_dpp v0, v0, v0 row_shr:1 (bound_ctrl:0)
    0x7e0202f2u, 0x7e040280u, 0x7e0602f2u,
    0xf800180fu, 0x03020100u, 0xbf810000u,
};

// Fullscreen triangle from gl_VertexIndex (the vertex program test_interp_render uses). Every quad
// of the target is fully covered, so no helper invocation shares a wave with a written pixel.
constexpr uint32_t kFullscreenVs[] = {
    0x7e140d00u, 0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100202f6u, 0x100404f6u,
    0x060202f3u, 0x060404f3u, 0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xf800020fu,
    0x0403030au, 0xbf810000u,
};

std::vector<uint8_t> render(const std::vector<uint32_t>& vs, const std::vector<uint32_t>& fs,
                            bool partial_wave, uint32_t w, uint32_t h) {
    prosper::test::BackendDraw d;
    d.vs = vs;
    d.fs = fs;
    d.vcount = 3;
    d.allow_partial_wave_fragment = partial_wave;
    for (uint32_t set = 0; set < 2; ++set) {
        prosper::test::FrameResource cb; cb.binding = 2; cb.set = set; d.R.push_back(cb);
        prosper::test::FrameResource vb; vb.binding = 3; vb.set = set; d.R.push_back(vb);
    }
    return prosper::test::render_draws_rgba({d}, w, h);
}

bool is_blue_clear(const uint8_t* p) { return p[2] > 0x80 && p[0] < 0x40 && p[1] < 0x40; }

}  // namespace

int main() {
    int failures = 0;
    const auto check = [&](bool ok, const char* message) {
        std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
        if (!ok) ++failures;
    };

    const std::vector<uint32_t> fs = recompile_fragment(kProgram, std::size(kProgram));
    const std::vector<uint32_t> vs = recompile_vertex(kFullscreenVs, std::size(kFullscreenVs));
    check(!fs.empty() && !vs.empty(), "fixture shaders recompile");
    if (fs.empty() || vs.empty()) return 1;
    // Pin the class this test exercises, so it cannot silently drift to a module the tier would
    // admit for a different reason (or not at all).
    const uint32_t reasons = fragment_spirv_required_subgroup_reasons(fs);
    std::printf("reasons=0x%x size=%u\n", reasons, fragment_spirv_required_subgroup_size(fs));
    check(fragment_spirv_required_subgroup_size(fs) == 64, "module declares the Wave64 contract");
    check((reasons & kFragmentWaveReasonLaneId) && (reasons & kFragmentWaveReasonWaveBallot) &&
              (reasons & kFragmentWaveReasonWaveAny) &&
              (reasons & ~kFragmentWavePartialWaveExactReasons) == 0,
          "module carries lane id, ballot and vote and nothing outside the partial-wave class");

    const auto& ctx = prosper::test::render_vk_ctx();
    const bool subgroup_ops = ctx.subgroup_size_control &&
        (ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
        (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_VOTE_BIT) &&
        (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_BALLOT_BIT) &&
        (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
    if (!subgroup_ops) {
        std::puts("SKIP: device has no fragment subgroup vote/ballot/arithmetic");
        return failures ? 1 : 77;
    }
    const bool native64 = ctx.min_subgroup_size <= 64 && ctx.max_subgroup_size >= 64 &&
        (ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT);
    std::printf("host fragment subgroup %u..%u native64=%d\n", ctx.min_subgroup_size,
                ctx.max_subgroup_size, native64 ? 1 : 0);

    constexpr uint32_t W = 64, H = 64;
    // Fail-visible without the tier: a host that cannot supply 64 lanes must still refuse the draw.
    const std::vector<uint8_t> strict = render(vs, fs, false, W, H);
    check(strict.size() == W * H * 4, "strict arm rendered a target");
    if (!native64 && strict.size() == W * H * 4)
        check(is_blue_clear(&strict[(H / 2 * W + W / 2) * 4]),
              "without the tier a narrow host still skips the Wave64 draw fail-visible");

    const std::vector<uint8_t> px = render(vs, fs, true, W, H);
    check(px.size() == W * H * 4, "partial-wave arm rendered a target");
    if (px.size() != W * H * 4) return 1;

    unsigned bad_flags = 0, bad_rank = 0, bad_high = 0, clear = 0;
    std::map<unsigned, std::map<unsigned, unsigned>> ranks_by_population;
    for (uint32_t i = 0; i < W * H; ++i) {
        const uint8_t* p = &px[i * 4];
        const unsigned rank = p[0], population = p[1], flags = p[2], alpha = p[3];
        if (is_blue_clear(p)) { ++clear; continue; }
        if (flags != 7) ++bad_flags;
        if (population == 0 || rank >= population) ++bad_rank;
        if (!native64 && ctx.max_subgroup_size <= 32 && alpha != 255) ++bad_high;
        ++ranks_by_population[population][rank];
    }
    std::printf("clear=%u bad_flags=%u bad_rank=%u bad_high=%u populations=", clear, bad_flags,
                bad_rank, bad_high);
    for (const auto& [n, ranks] : ranks_by_population)
        std::printf("%u(x%u) ", n, ranks.empty() ? 0u : ranks.begin()->second);
    std::printf("\n");
    check(clear == 0, "every pixel of the fullscreen draw was written");
    check(bad_flags == 0,
          "own ballot bit, one rank-0 lane per wave, and the vote over it hold on every pixel");
    check(bad_rank == 0, "every rank is below its wave's population");
    check(bad_high == 0, "on a host of at most 32 lanes the high EXEC dword is zero");
    bool histogram = !ranks_by_population.empty();
    for (const auto& [n, ranks] : ranks_by_population) {
        if (ranks.size() != n) { histogram = false; break; }
        const unsigned count = ranks.begin()->second;
        for (unsigned r = 0; r < n; ++r) {
            const auto it = ranks.find(r);
            if (it == ranks.end() || it->second != count) { histogram = false; break; }
        }
    }
    check(histogram,
          "for every wave population n, ranks 0..n-1 occur equally often (one per wave)");

    // The tier must not widen past its class: a DPP row read stays refused on a narrow host.
    const std::vector<uint32_t> dpp = recompile_fragment(kDppProgram, std::size(kDppProgram));
    check(!dpp.empty() &&
              (fragment_spirv_required_subgroup_reasons(dpp) & kFragmentWaveReasonDppRow16),
          "DPP fixture carries the dpp16 reason");
    if (!native64 && !dpp.empty()) {
        const std::vector<uint8_t> dpp_px = render(vs, dpp, true, W, H);
        check(dpp_px.size() == W * H * 4 && is_blue_clear(&dpp_px[(H / 2 * W + W / 2) * 4]),
              "a DPP row read is still skipped fail-visible with the tier enabled");
    }
    return failures ? 1 : 0;
}
