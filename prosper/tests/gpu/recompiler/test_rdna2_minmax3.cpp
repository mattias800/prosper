// test_rdna2_minmax3.cpp -- the min/max/median-of-three kernel assertions (kernels 4-4g),
// moved VERBATIM out of test_rdna2_to_spirv.cpp's main(). The block was self-contained: it
// referenced no declaration from main's scope except the harness size N (replicated below),
// which is what made it movable without a rewrite.
//
// The invariant that makes the move checkable is the executed-assertion count. main() records
// the count before calling run_minmax3_checks() and requires the delta to be exactly 14, so a
// block that failed to arrive fails the test instead of quietly shrinking the suite. Verified
// by mutation: an empty run_minmax3_checks() exits 1 with "contributed 0 checks, expected 14".
//
// Below this comment is only what the moved block needs: the includes, `extern` declarations of
// main()'s `fails` and `checks` counters, and the same `CHECK` macro text main() uses. The original
// file's static helpers are not copied.

#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/compute_runner.h"
#include "fixtures/gta5_compute_cfg_fixture.hpp"
#include <algorithm>
#include <bit>
#include <set>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

using namespace prosper::gpu;

extern int fails;
extern int checks;
#define CHECK(c, m) do { ++checks; \
                         if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Harness size, mirroring main's `const uint32_t N = 128;`. The kernels size their inputs and
// launches from it; the two copies must agree or the moved block tests a different workload.
const uint32_t N = 128;

void run_minmax3_checks() {
    // Kernel 4: v_med3_f32 + v_ceil_f32. out = ceil(median(a0,a1,a2)).
    const uint32_t code4[] = { 0xD5570000u, 0x040A0300u, 0x7E004500u, 0xBF810000u };
    std::vector<uint32_t> spv4 = recompile_valu(code4, sizeof(code4)/sizeof(code4[0]), 3, 0);
    CHECK(!spv4.empty(), "recompiled kernel 4 (med3/ceil) -> SPIR-V");
    std::vector<float> in4(N * 3), exp4(N);
    for (uint32_t i = 0; i < N; i++) {
        float a0 = (float)i * 0.3f - 15.0f, a1 = (float)i * 0.1f, a2 = 5.5f;
        in4[i*3+0]=a0; in4[i*3+1]=a1; in4[i*3+2]=a2;
        float lo = std::fmin(a0,a1), hi = std::fmax(a0,a1); float med = std::fmax(lo, std::fmin(hi, a2));
        exp4[i] = std::ceil(med);
    }
    std::vector<float> got4 = prosper::test::run_compute(spv4, in4, N, N);
    uint32_t bad4 = 0; for (uint32_t i=0;i<N&&got4.size()==N;i++) if (std::fabs(got4[i]-exp4[i])>1e-3f) bad4++;
    printf("  kernel4 mismatches=%u (out[60]=%g expect=%g)\n", bad4, got4.size()==N?got4[60]:-1, exp4[60]);
    CHECK(got4.size()==N && bad4==0, "recompiled kernel 4 computes ceil(median(a0,a1,a2)) correctly");

    // Kernel 4b: unsigned median-of-three. Convert the float harness inputs to u32, find the
    // median, and convert it back so all six input orderings are checked by real Vulkan execution.
    const uint32_t code4b[] = {
        0x7E000F00u, 0x7E020F01u, 0x7E040F02u, 0xD5590003u, 0x040A0300u,
        0x7E000D03u, 0xBF810000u,
    };
    std::vector<uint32_t> spv4b = recompile_valu(code4b, std::size(code4b), 3, 0);
    CHECK(!spv4b.empty(), "recompiled kernel 4b (v_med3_u32) -> SPIR-V");
    std::vector<float> in4b(N * 3), exp4b(N);
    for (uint32_t i = 0; i < N; i++) {
        const uint32_t u0 = (i * 37u) % 211u;
        const uint32_t u1 = (i * 83u + 17u) % 211u;
        const uint32_t u2 = (i * 19u + 101u) % 211u;
        in4b[i*3+0] = (float)u0; in4b[i*3+1] = (float)u1; in4b[i*3+2] = (float)u2;
        const uint32_t mn = std::min(u0, u1), mx = std::max(u0, u1);
        exp4b[i] = (float)std::max(mn, std::min(mx, u2));
    }
    // Kernel 4c (#2013): unsigned max-of-three, the shape Sonic Racing: CrossWorlds rejects four
    // times per boot. Same u32 round-trip as 4b so real Vulkan execution checks every ordering.
    const uint32_t code4c[] = {
        0x7E000F00u, 0x7E020F01u, 0x7E040F02u, 0xD5560003u, 0x040A0300u,
        0x7E000D03u, 0xBF810000u,
    };
    std::vector<uint32_t> spv4c = recompile_valu(code4c, std::size(code4c), 3, 0);
    CHECK(!spv4c.empty(), "recompiled kernel 4c (v_max3_u32) -> SPIR-V");

    // Kernel 4d (#2013): v_min3_u32, the other half of the pair 4c covers.
    const uint32_t code4d[] = {
        0x7E000F00u, 0x7E020F01u, 0x7E040F02u, 0xD5530003u, 0x040A0300u,
        0x7E000D03u, 0xBF810000u,
    };
    std::vector<uint32_t> spv4d = recompile_valu(code4d, std::size(code4d), 3, 0);
    CHECK(!spv4d.empty(), "recompiled kernel 4d (v_min3_u32) -> SPIR-V");

    std::vector<float> got4b = prosper::test::run_compute(spv4b, in4b, N, N);
    uint32_t bad4b = 0;
    for (uint32_t i = 0; i < N && got4b.size() == N; i++)
        if (got4b[i] != exp4b[i]) bad4b++;
    printf("  kernel4b mismatches=%u (out[60]=%g expect=%g)\n",
           bad4b, got4b.size()==N ? got4b[60] : -1, exp4b[60]);
    CHECK(got4b.size()==N && bad4b==0, "recompiled kernel 4b computes unsigned median correctly");

    std::vector<float> got4c = prosper::test::run_compute(spv4c, in4b, N, N);
    uint32_t bad4c = 0;
    for (uint32_t i = 0; i < N && got4c.size() == N; i++) {
        const uint32_t u0 = (i * 37u) % 211u;
        const uint32_t u1 = (i * 83u + 17u) % 211u;
        const uint32_t u2 = (i * 19u + 101u) % 211u;
        if (got4c[i] != (float)std::max(u0, std::max(u1, u2))) bad4c++;
    }
    printf("  kernel4c mismatches=%u (out[60]=%g)\n", bad4c, got4c.size()==N ? got4c[60] : -1);
    CHECK(got4c.size()==N && bad4c==0, "recompiled kernel 4c computes unsigned max-of-three correctly");

    std::vector<float> got4d = prosper::test::run_compute(spv4d, in4b, N, N);
    uint32_t bad4d = 0;
    for (uint32_t i = 0; i < N && got4d.size() == N; i++) {
        const uint32_t u0 = (i * 37u) % 211u;
        const uint32_t u1 = (i * 83u + 17u) % 211u;
        const uint32_t u2 = (i * 19u + 101u) % 211u;
        if (got4d[i] != (float)std::min(u0, std::min(u1, u2))) bad4d++;
    }
    printf("  kernel4d mismatches=%u (out[60]=%g)\n", bad4d, got4d.size()==N ? got4d[60] : -1);
    CHECK(got4d.size()==N && bad4d==0, "recompiled kernel 4d computes unsigned min-of-three correctly");

    // Kernel 4e: signed median-of-three (v_med3_i32, 0x158).
    // Tests signed ordering with negative and positive inputs. Unlike kernels 4-4d these convert
    // through the SIGNED forms -- v_cvt_i32_f32 (VOP1 0x08) in, v_cvt_f32_i32 (0x05) out -- since
    // v_cvt_u32_f32 would clamp every negative input to 0 and never exercise a signed compare.
    const uint32_t code4e[] = {
        0x7E001100u, 0x7E021101u, 0x7E041102u, 0xD5580003u, 0x040A0300u, 0x7E000B03u, 0xBF810000u,
    };
    std::vector<uint32_t> spv4e = recompile_valu(code4e, std::size(code4e), 3, 0);
    CHECK(!spv4e.empty(), "recompiled kernel 4e (v_med3_i32) -> SPIR-V");

    // Kernel 4f: signed max-of-three (v_max3_i32, 0x155).
    const uint32_t code4f[] = {
        0x7E001100u, 0x7E021101u, 0x7E041102u, 0xD5550003u, 0x040A0300u, 0x7E000B03u, 0xBF810000u,
    };
    std::vector<uint32_t> spv4f = recompile_valu(code4f, std::size(code4f), 3, 0);
    CHECK(!spv4f.empty(), "recompiled kernel 4f (v_max3_i32) -> SPIR-V");

    // Kernel 4g: signed min-of-three (v_min3_i32, 0x152).
    const uint32_t code4g[] = {
        0x7E001100u, 0x7E021101u, 0x7E041102u, 0xD5520003u, 0x040A0300u, 0x7E000B03u, 0xBF810000u,
    };
    std::vector<uint32_t> spv4g = recompile_valu(code4g, std::size(code4g), 3, 0);
    CHECK(!spv4g.empty(), "recompiled kernel 4g (v_min3_i32) -> SPIR-V");

    std::vector<float> in4_signed(N * 3);
    for (uint32_t i = 0; i < N; i++) {
        const int32_t s0 = (int32_t)((i * 37u) % 211u) - 100;
        const int32_t s1 = (int32_t)((i * 83u + 17u) % 211u) - 100;
        const int32_t s2 = (int32_t)((i * 19u + 101u) % 211u) - 100;
        in4_signed[i * 3 + 0] = (float)s0;
        in4_signed[i * 3 + 1] = (float)s1;
        in4_signed[i * 3 + 2] = (float)s2;
    }

    std::vector<float> got4e = prosper::test::run_compute(spv4e, in4_signed, N, N);
    uint32_t bad4e = 0;
    for (uint32_t i = 0; i < N && got4e.size() == N; i++) {
        const int32_t s0 = (int32_t)((i * 37u) % 211u) - 100;
        const int32_t s1 = (int32_t)((i * 83u + 17u) % 211u) - 100;
        const int32_t s2 = (int32_t)((i * 19u + 101u) % 211u) - 100;
        const int32_t mn = std::min(s0, s1), mx = std::max(s0, s1);
        const int32_t exp = std::max(mn, std::min(mx, s2));
        if (got4e[i] != (float)exp) bad4e++;
    }
    CHECK(got4e.size() == N && bad4e == 0, "recompiled kernel 4e computes signed median correctly");

    std::vector<float> got4f = prosper::test::run_compute(spv4f, in4_signed, N, N);
    uint32_t bad4f = 0;
    for (uint32_t i = 0; i < N && got4f.size() == N; i++) {
        const int32_t s0 = (int32_t)((i * 37u) % 211u) - 100;
        const int32_t s1 = (int32_t)((i * 83u + 17u) % 211u) - 100;
        const int32_t s2 = (int32_t)((i * 19u + 101u) % 211u) - 100;
        const int32_t exp = std::max(s0, std::max(s1, s2));
        if (got4f[i] != (float)exp) bad4f++;
    }
    CHECK(got4f.size() == N && bad4f == 0,
          "recompiled kernel 4f computes signed max-of-three correctly");

    std::vector<float> got4g = prosper::test::run_compute(spv4g, in4_signed, N, N);
    uint32_t bad4g = 0;
    for (uint32_t i = 0; i < N && got4g.size() == N; i++) {
        const int32_t s0 = (int32_t)((i * 37u) % 211u) - 100;
        const int32_t s1 = (int32_t)((i * 83u + 17u) % 211u) - 100;
        const int32_t s2 = (int32_t)((i * 19u + 101u) % 211u) - 100;
        const int32_t exp = std::min(s0, std::min(s1, s2));
        if (got4g[i] != (float)exp) bad4g++;
    }
    CHECK(got4g.size() == N && bad4g == 0,
          "recompiled kernel 4g computes signed min-of-three correctly");
}
