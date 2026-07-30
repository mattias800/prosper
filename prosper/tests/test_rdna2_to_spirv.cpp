// test_rdna2_to_spirv — the payoff: recompile REAL RDNA2 instructions to SPIR-V and prove the
// result is numerically correct by execution (verification layer 4, end to end). We recompile a
// straight-line float-VALU kernel (assembled by llvm-mc for gfx1030), run it on real Vulkan compute
// with known per-invocation inputs, and assert the outputs equal what the RDNA2 ops compute.
//
// Kernel (v0..v3 = 4 inputs per invocation):
//   v_add_f32 v0, v0, v1      ; v0 = a0 + a1
//   v_mul_f32 v0, v0, v2      ; v0 = (a0+a1) * a2
//   v_fma_f32 v0, v0, v3, v1  ; v0 = v0*a3 + a1
//   s_endpgm
// => out = ((a0+a1)*a2)*a3 + a1
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/agc_shader_layout.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "compute_runner.h"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// IEEE float32 -> float16 for NORMAL, in-range, exactly-representable inputs (chosen in the tests so
// round-toward-zero (pkrtz) and round-to-nearest (packHalf2x16) agree). Not a general converter.
static uint16_t f32_to_f16_exact(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  e = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t m = (x & 0x7fffffu) >> 13;
    return (uint16_t)(sign | ((uint32_t)e << 10) | m);
}
static uint32_t bits_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint32_t bitrev32(uint32_t v) { uint32_t r = 0; for (int i = 0; i < 32; i++) { r = (r << 1) | (v & 1u); v >>= 1; } return r; }
static size_t count_spirv_opcode(const std::vector<uint32_t>& spv, uint16_t opcode) {
    size_t matches = 0;
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        if (!count || word + count > spv.size()) return 0;
        if ((spv[word] & 0xFFFFu) == opcode) ++matches;
        word += count;
    }
    return matches;
}
static bool has_spirv_builtin(const std::vector<uint32_t>& spv, uint32_t builtin) {
    constexpr uint16_t OpDecorate = 71;
    constexpr uint32_t DecorationBuiltIn = 11;
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        if (!count || word + count > spv.size()) return false;
        if ((spv[word] & 0xFFFFu) == OpDecorate && count >= 4 &&
            spv[word + 2] == DecorationBuiltIn && spv[word + 3] == builtin)
            return true;
        word += count;
    }
    return false;
}
static bool has_spirv_local_size(const std::vector<uint32_t>& spv,
                                 uint32_t x, uint32_t y, uint32_t z) {
    constexpr uint16_t OpExecutionMode = 16;
    constexpr uint32_t ExecutionModeLocalSize = 17;
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        if (!count || word + count > spv.size()) return false;
        if ((spv[word] & 0xFFFFu) == OpExecutionMode && count == 6 &&
            spv[word + 2] == ExecutionModeLocalSize && spv[word + 3] == x &&
            spv[word + 4] == y && spv[word + 5] == z)
            return true;
        word += count;
    }
    return false;
}

int main() {
    printf("== test_rdna2_to_spirv ==\n");
    // Assembled with llvm-mc -mcpu=gfx1030 (see header comment).
    const uint32_t code[] = {
        0x06000300u,               // v_add_f32 v0, v0, v1
        0x10000500u,               // v_mul_f32 v0, v0, v2
        0xD54B0000u, 0x04060700u,  // v_fma_f32 v0, v0, v3, v1
        0xBF810000u,               // s_endpgm
    };
    std::vector<uint32_t> spv = recompile_valu(code, sizeof(code)/sizeof(code[0]), /*num_inputs*/4, /*out_vgpr*/0);
    CHECK(!spv.empty() && spv[0] == 0x07230203u, "recompiled RDNA2 -> a SPIR-V module");
    if (spv.empty()) { printf("== FAIL: recompile returned empty (unsupported opcode?) ==\n"); return 1; }

    const uint32_t N = 128;
    std::vector<float> in(N * 4), expect(N);
    for (uint32_t i = 0; i < N; i++) {
        float a0 = (float)i * 0.1f - 5.0f, a1 = (float)i * 0.01f + 1.0f, a2 = 2.0f, a3 = -0.5f;
        in[i*4+0] = a0; in[i*4+1] = a1; in[i*4+2] = a2; in[i*4+3] = a3;
        expect[i] = ((a0 + a1) * a2) * a3 + a1;
    }

    std::vector<float> got = prosper::test::run_compute(spv, in, /*invocations*/N, /*out_count*/N);
    CHECK(got.size() == N, "recompiled shader compiled + ran on Vulkan");
    if (got.size() != N) { printf("== FAIL: shader did not run ==\n"); return 1; }

    uint32_t bad = 0; float worst = 0;
    for (uint32_t i = 0; i < N; i++) { float d = std::fabs(got[i] - expect[i]); if (d > 1e-3f) { bad++; worst = d > worst ? d : worst; } }
    printf("  N=%u mismatches=%u worst=%g (out[50]=%g expect=%g)\n", N, bad, worst, got[50], expect[50]);
    CHECK(bad == 0, "recompiled RDNA2 kernel computes ((a0+a1)*a2)*a3+a1 correctly");

    // Kernel 2: transcendental/min/max ops. v0=min(a0,a1); v0=max(v0,a2); v0=sqrt(v0). (a2>=0 => arg>=0)
    //   v_min_f32 v0,v0,v1 | v_max_f32 v0,v0,v2 | v_sqrt_f32 v0,v0 | s_endpgm
    const uint32_t code2[] = { 0x1E000300u, 0x20000500u, 0x7E006700u, 0xBF810000u };
    std::vector<uint32_t> spv2 = recompile_valu(code2, sizeof(code2)/sizeof(code2[0]), 3, 0);
    CHECK(!spv2.empty(), "recompiled kernel 2 (min/max/sqrt) -> SPIR-V");
    std::vector<float> in2(N * 3), exp2(N);
    for (uint32_t i = 0; i < N; i++) {
        float a0 = (float)i - 40.0f, a1 = 20.0f, a2 = (float)(i % 7);   // a2 >= 0
        in2[i*3+0] = a0; in2[i*3+1] = a1; in2[i*3+2] = a2;
        float mn = a0 < a1 ? a0 : a1; float mx = mn > a2 ? mn : a2; exp2[i] = std::sqrt(mx);
    }
    std::vector<float> got2 = prosper::test::run_compute(spv2, in2, N, N);
    uint32_t bad2 = 0;
    for (uint32_t i = 0; i < N && got2.size() == N; i++) if (std::fabs(got2[i] - exp2[i]) > 1e-3f) bad2++;
    printf("  kernel2 mismatches=%u (out[70]=%g expect=%g)\n", bad2, got2.size()==N?got2[70]:-1, exp2[70]);
    CHECK(got2.size() == N && bad2 == 0, "recompiled kernel 2 computes sqrt(max(min(a0,a1),a2)) correctly");

    // Kernel 3: integer + convert ops. u0=(uint)a0; u1=(uint)a1; u2=(u0+u1)*u0; u2 &= u1; out=(float)u2.
    //   v_cvt_u32_f32 v0,v0 | v_cvt_u32_f32 v1,v1 | v_add_nc_u32 v2,v0,v1 | v_mul_lo_u32 v2,v2,v0
    //   | v_and_b32 v2,v2,v1 | v_cvt_f32_u32 v0,v2 | s_endpgm
    const uint32_t code3[] = {
        0x7E000F00u, 0x7E020F01u, 0x4A040300u, 0xD5690002u, 0x00020102u, 0x36040302u, 0x7E000D02u, 0xBF810000u,
    };
    std::vector<uint32_t> spv3 = recompile_valu(code3, sizeof(code3)/sizeof(code3[0]), 2, 0);
    CHECK(!spv3.empty(), "recompiled kernel 3 (integer/convert ops) -> SPIR-V");
    std::vector<float> in3(N * 2), exp3(N);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t u0 = i % 20, u1 = i % 13;
        in3[i*2+0] = (float)u0; in3[i*2+1] = (float)u1;
        uint32_t u2 = (u0 + u1) * u0; u2 &= u1; exp3[i] = (float)u2;
    }
    std::vector<float> got3 = prosper::test::run_compute(spv3, in3, N, N);
    uint32_t bad3 = 0;
    for (uint32_t i = 0; i < N && got3.size() == N; i++) if (std::fabs(got3[i] - exp3[i]) > 1e-3f) bad3++;
    printf("  kernel3 mismatches=%u (out[25]=%g expect=%g)\n", bad3, got3.size()==N?got3[25]:-1, exp3[25]);
    CHECK(got3.size() == N && bad3 == 0, "recompiled kernel 3 computes ((u0+u1)*u0)&u1 correctly (int ops)");

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
    std::vector<float> got4b = prosper::test::run_compute(spv4b, in4b, N, N);
    uint32_t bad4b = 0;
    for (uint32_t i = 0; i < N && got4b.size() == N; i++)
        if (got4b[i] != exp4b[i]) bad4b++;
    printf("  kernel4b mismatches=%u (out[60]=%g expect=%g)\n",
           bad4b, got4b.size()==N ? got4b[60] : -1, exp4b[60]);
    CHECK(got4b.size()==N && bad4b==0, "recompiled kernel 4b computes unsigned median correctly");

    // Kernel 5: unsigned min/max/sub/not/and. u=(uint)a; d=(max-min) & ~u0. out=(float)d.
    const uint32_t code5[] = {
        0x7E000F00u, 0x7E020F01u, 0x26040300u, 0x28060300u, 0x4C040503u, 0x7E066F00u, 0x36040702u, 0x7E000D02u, 0xBF810000u,
    };
    std::vector<uint32_t> spv5 = recompile_valu(code5, sizeof(code5)/sizeof(code5[0]), 2, 0);
    CHECK(!spv5.empty(), "recompiled kernel 5 (uint min/max/sub/not) -> SPIR-V");
    std::vector<float> in5(N * 2), exp5(N);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t u0 = (i * 7) % 100, u1 = (i * 3) % 100;
        in5[i*2+0]=(float)u0; in5[i*2+1]=(float)u1;
        uint32_t mn=u0<u1?u0:u1, mx=u0>u1?u0:u1; exp5[i]=(float)((mx-mn) & ~u0);
    }
    std::vector<float> got5 = prosper::test::run_compute(spv5, in5, N, N);
    uint32_t bad5 = 0; for (uint32_t i=0;i<N&&got5.size()==N;i++) if (std::fabs(got5[i]-exp5[i])>1e-3f) bad5++;
    printf("  kernel5 mismatches=%u (out[33]=%g expect=%g)\n", bad5, got5.size()==N?got5[33]:-1, exp5[33]);
    CHECK(got5.size()==N && bad5==0, "recompiled kernel 5 computes (max-min)&~u0 correctly");

    // Kernel 6: compare + select. vcc = a0 > a1; v0 = vcc ? a0 : a1  (i.e. max(a0,a1)).
    //   v_cmp_gt_f32 vcc,v0,v1 | v_cndmask_b32 v0,v1,v0,vcc | s_endpgm
    const uint32_t code6[] = { 0x7C080300u, 0x02000101u, 0xBF810000u };
    std::vector<uint32_t> spv6 = recompile_valu(code6, sizeof(code6)/sizeof(code6[0]), 2, 0);
    CHECK(!spv6.empty(), "recompiled kernel 6 (v_cmp + v_cndmask) -> SPIR-V");
    std::vector<float> in6(N * 2), exp6(N);
    for (uint32_t i = 0; i < N; i++) {
        float a0 = (float)i - 60.0f, a1 = (float)((i * 37) % 90) - 30.0f;   // straddle both orders
        in6[i*2+0]=a0; in6[i*2+1]=a1; exp6[i] = a0 > a1 ? a0 : a1;
    }
    std::vector<float> got6 = prosper::test::run_compute(spv6, in6, N, N);
    uint32_t bad6 = 0; for (uint32_t i=0;i<N&&got6.size()==N;i++) if (std::fabs(got6[i]-exp6[i])>1e-3f) bad6++;
    printf("  kernel6 mismatches=%u (out[10]=%g expect=%g)\n", bad6, got6.size()==N?got6[10]:-1, exp6[10]);
    CHECK(got6.size()==N && bad6==0, "recompiled kernel 6 computes max via compare+select correctly");

    // Kernel 6b: NGG prologs merge a scalar lane-bit dword into one VCC half. Start with a false
    // VOPC predicate, OR bit 0 into VCC_LO, and select input0 only for guest lane 0. Repeat for
    // VCC_HI bit 0 (guest lane 32) to prove that a B32 write preserves the untouched half.
    auto check_vcc_half_merge = [&](bool high) {
        const uint32_t code[] = {
            0xBE940381u,                         // s_mov_b32 s20, 1
            0x7C000100u,                         // v_cmp_lt_f32 vcc, v0, v0 (false)
            high ? 0x886B6B14u : 0x886A6A14u,   // s_or_b32 vcc_{hi,lo}, s20, vcc_{hi,lo}
            0x02000101u,                         // v_cndmask_b32 v0, v1, v0, vcc
            0xBF810000u,
        };
        const auto words = recompile_valu(code, std::size(code), 2, 0);
        std::vector<float> input(64 * 2), expected(64);
        for (uint32_t lane = 0; lane < 64; ++lane) {
            input[lane * 2] = 10.0f;
            input[lane * 2 + 1] = 20.0f;
            expected[lane] = lane == (high ? 32u : 0u) ? 10.0f : 20.0f;
        }
        const auto output = prosper::test::run_compute(words, input, 64, 64);
        uint32_t bad = 0;
        for (uint32_t lane = 0; lane < output.size() && lane < expected.size(); ++lane)
            if (std::fabs(output[lane] - expected[lane]) > 1e-3f) ++bad;
        CHECK(!words.empty() && output.size() == 64 && bad == 0,
              high ? "VCC_HI scalar-mask merge updates lane 32 and preserves VCC_LO"
                   : "VCC_LO scalar-mask merge updates lane 0 and preserves VCC_HI");
    };
    check_vcc_half_merge(false);
    check_vcc_half_merge(true);

    // Kernel 7: signed min/max/ashr. i=(int)a; range=max(i0,i1)-min(i0,i1); out=(float)(range >> (i2&31)).
    const uint32_t code7[] = {
        0x7E001100u, 0x7E021101u, 0x7E041102u, 0x22060300u, 0x24080300u, 0x4C060704u, 0x30060702u, 0x7E000B03u, 0xBF810000u,
    };
    std::vector<uint32_t> spv7 = recompile_valu(code7, sizeof(code7)/sizeof(code7[0]), 3, 0);
    CHECK(!spv7.empty(), "recompiled kernel 7 (signed min/max/ashr) -> SPIR-V");
    std::vector<float> in7(N * 3), exp7(N);
    for (uint32_t i = 0; i < N; i++) {
        int i0 = (int)i - 70, i1 = (int)((i * 13) % 140) - 70, i2 = (int)(i % 8);
        in7[i*3+0]=(float)i0; in7[i*3+1]=(float)i1; in7[i*3+2]=(float)i2;
        int mn = i0 < i1 ? i0 : i1, mx = i0 > i1 ? i0 : i1; int range = mx - mn;
        exp7[i] = (float)(range >> (i2 & 31));
    }
    std::vector<float> got7 = prosper::test::run_compute(spv7, in7, N, N);
    uint32_t bad7 = 0; for (uint32_t i=0;i<N&&got7.size()==N;i++) if (std::fabs(got7[i]-exp7[i])>1e-3f) bad7++;
    printf("  kernel7 mismatches=%u (out[90]=%g expect=%g)\n", bad7, got7.size()==N?got7[90]:-1, exp7[90]);
    CHECK(got7.size()==N && bad7==0, "recompiled kernel 7 (signed min/max/ashr) correct");

    // Kernel 8: signed compare + select. out = (i0 > i1) ? i0 : i2  (signed comparison).
    const uint32_t code8[] = {
        0x7E001100u, 0x7E021101u, 0x7E041102u, 0x7D080300u, 0x02000102u, 0x7E000B00u, 0xBF810000u,
    };
    std::vector<uint32_t> spv8 = recompile_valu(code8, sizeof(code8)/sizeof(code8[0]), 3, 0);
    CHECK(!spv8.empty(), "recompiled kernel 8 (v_cmp_gt_i32 + select) -> SPIR-V");
    std::vector<float> in8(N * 3), exp8(N);
    for (uint32_t i = 0; i < N; i++) {
        int i0 = (int)(i % 50) - 25, i1 = 0, i2 = (int)i - 40;
        in8[i*3+0]=(float)i0; in8[i*3+1]=(float)i1; in8[i*3+2]=(float)i2;
        exp8[i] = (float)(i0 > i1 ? i0 : i2);
    }
    std::vector<float> got8 = prosper::test::run_compute(spv8, in8, N, N);
    uint32_t bad8 = 0; for (uint32_t i=0;i<N&&got8.size()==N;i++) if (std::fabs(got8[i]-exp8[i])>1e-3f) bad8++;
    printf("  kernel8 mismatches=%u (out[10]=%g expect=%g)\n", bad8, got8.size()==N?got8[10]:-1, exp8[10]);
    CHECK(got8.size()==N && bad8==0, "recompiled kernel 8 (signed compare/select) correct");

    // Kernel 9: scalar ALU (SOP1/SOP2) feeding a VALU op via an SGPR operand.
    //   s_mov_b32 s0,10 | s_add_u32 s0,s0,5 (=15) | s_lshl_b32 s1,s0,1 (=30)
    //   v_cvt_f32_u32 v2,s1 (=30.0) | v_add_f32 v0,v0,v2  => out = a0 + 30
    const uint32_t code9[] = {
        0xBE80038Au, 0x80008500u, 0x8F018100u, 0x7E040C01u, 0x06000500u, 0xBF810000u,
    };
    std::vector<uint32_t> spv9 = recompile_valu(code9, sizeof(code9)/sizeof(code9[0]), 1, 0);
    CHECK(!spv9.empty(), "recompiled kernel 9 (scalar ALU + SGPR operand) -> SPIR-V");
    std::vector<float> in9(N), exp9(N);
    for (uint32_t i = 0; i < N; i++) { in9[i] = (float)i; exp9[i] = (float)i + 30.0f; }
    std::vector<float> got9 = prosper::test::run_compute(spv9, in9, N, N);
    uint32_t bad9 = 0; for (uint32_t i=0;i<N&&got9.size()==N;i++) if (std::fabs(got9[i]-exp9[i])>1e-3f) bad9++;
    printf("  kernel9 mismatches=%u (out[5]=%g expect=%g)\n", bad9, got9.size()==N?got9[5]:-1, exp9[5]);
    CHECK(got9.size()==N && bad9==0, "recompiled kernel 9 (scalar s_mov/s_add/s_lshl) computes a0+30");

    // Kernel 10: VOP3 3-operand ops. u2=mad24(u0,u1,u1); u3=add3(u0,u1,u2); u3=bfi(u0,u1,u3);
    //            u3=bfe_u(u3, off=u0, cnt=u1); out=(float)u3.
    const uint32_t code10[] = {
        0x7E000F00u, 0x7E020F01u, 0xD5430002u, 0x04060300u, 0xD76D0003u, 0x040A0300u,
        0xD54A0003u, 0x040E0300u, 0xD5480003u, 0x04060103u, 0x7E000D03u, 0xBF810000u,
    };
    std::vector<uint32_t> spv10 = recompile_valu(code10, sizeof(code10)/sizeof(code10[0]), 2, 0);
    CHECK(!spv10.empty(), "recompiled kernel 10 (mad24/add3/bfi/bfe) -> SPIR-V");
    std::vector<float> in10(N * 2), exp10(N);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t u0 = i % 6, u1 = i % 9;
        in10[i*2+0] = (float)u0; in10[i*2+1] = (float)u1;
        uint32_t u2 = (u0 & 0xFFFFFFu) * (u1 & 0xFFFFFFu) + u1;
        uint32_t u3 = u0 + u1 + u2;
        u3 = (u0 & u1) | (~u0 & u3);
        uint32_t off = u0 & 31, cnt = u1 & 31;
        uint32_t mask = (cnt >= 32) ? 0xFFFFFFFFu : ((1u << cnt) - 1u);
        u3 = (u3 >> off) & mask;
        exp10[i] = (float)u3;
    }
    std::vector<float> got10 = prosper::test::run_compute(spv10, in10, N, N);
    uint32_t bad10 = 0; for (uint32_t i=0;i<N&&got10.size()==N;i++) if (std::fabs(got10[i]-exp10[i])>1e-3f) bad10++;
    printf("  kernel10 mismatches=%u (out[20]=%g expect=%g)\n", bad10, got10.size()==N?got10[20]:-1, exp10[20]);
    CHECK(got10.size()==N && bad10==0, "recompiled kernel 10 (VOP3 mad24/add3/bfi/bfe) correct");

    // Kernel 11: VOP3 bitwise/arith 3-op batch (the histogram's hottest ops — v_and_or ×74, etc.).
    //   u3=(u0&u1)|u2; u3|=u0|u1; u3=(u3<<(u0&31))|u1; u3=(u3^u0)+u2; u3=(u3+u1)<<(u0&31); out=(float)u3.
    const uint32_t code11[] = {
        0x7e000f00u, 0x7e020f01u, 0x7e040f02u, 0xd7710003u, 0x040a0300u, 0xd7720003u, 0x04060103u,
        0xd76f0003u, 0x04060103u, 0xd7450003u, 0x040a0103u, 0xd7470003u, 0x04020303u, 0x7e000d03u, 0xbf810000u,
    };
    std::vector<uint32_t> spv11 = recompile_valu(code11, sizeof(code11)/sizeof(code11[0]), 3, 0);
    CHECK(!spv11.empty(), "recompiled kernel 11 (v_and_or/or3/lshl_or/xad/add_lshl) -> SPIR-V");
    std::vector<float> in11(N * 3), exp11(N);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t u0 = i % 6, u1 = i % 9, u2 = i % 7;   // bounded so (float)u3 stays exact (< 2^24)
        in11[i*3+0]=(float)u0; in11[i*3+1]=(float)u1; in11[i*3+2]=(float)u2;
        uint32_t u3 = (u0 & u1) | u2;
        u3 = u3 | u0 | u1;
        u3 = (u3 << (u0 & 31)) | u1;
        u3 = (u3 ^ u0) + u2;
        u3 = (u3 + u1) << (u0 & 31);
        exp11[i] = (float)u3;
    }
    std::vector<float> got11 = prosper::test::run_compute(spv11, in11, N, N);
    uint32_t bad11 = 0; for (uint32_t i=0;i<N&&got11.size()==N;i++) if (std::fabs(got11[i]-exp11[i])>1e-3f) bad11++;
    printf("  kernel11 mismatches=%u (out[40]=%g expect=%g)\n", bad11, got11.size()==N?got11[40]:-1, exp11[40]);
    CHECK(got11.size()==N && bad11==0, "recompiled kernel 11 (VOP3 and_or/or3/lshl_or/xad/add_lshl) correct");

    // Kernel 12: unsigned compare + select. out = (u0 <= u1) ? a2 : a3.
    const uint32_t code12[] = { 0x7e000f00u, 0x7e020f01u, 0x7d860300u, 0x02000503u, 0xbf810000u };
    std::vector<uint32_t> spv12 = recompile_valu(code12, sizeof(code12)/sizeof(code12[0]), 4, 0);
    CHECK(!spv12.empty(), "recompiled kernel 12 (v_cmp_le_u32 + select) -> SPIR-V");
    std::vector<float> in12(N * 4), exp12(N);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t u0 = i % 11, u1 = (i * 3) % 11; float a2 = (float)i + 0.5f, a3 = -(float)i;
        in12[i*4+0]=(float)u0; in12[i*4+1]=(float)u1; in12[i*4+2]=a2; in12[i*4+3]=a3;
        exp12[i] = (u0 <= u1) ? a2 : a3;
    }
    std::vector<float> got12 = prosper::test::run_compute(spv12, in12, N, N);
    uint32_t bad12 = 0; for (uint32_t i=0;i<N&&got12.size()==N;i++) if (std::fabs(got12[i]-exp12[i])>1e-3f) bad12++;
    printf("  kernel12 mismatches=%u (out[7]=%g expect=%g)\n", bad12, got12.size()==N?got12[7]:-1, exp12[7]);
    CHECK(got12.size()==N && bad12==0, "recompiled kernel 12 (unsigned compare le_u32 + select) correct");

    // Kernel 13: signed ge compare + select = signed max. out = (i0 >= i1) ? i0 : i1.
    const uint32_t code13[] = { 0x7e001100u, 0x7e021101u, 0x7d0c0300u, 0x02000101u, 0x7e000b00u, 0xbf810000u };
    std::vector<uint32_t> spv13 = recompile_valu(code13, sizeof(code13)/sizeof(code13[0]), 2, 0);
    CHECK(!spv13.empty(), "recompiled kernel 13 (v_cmp_ge_i32 + select) -> SPIR-V");
    std::vector<float> in13(N * 2), exp13(N);
    for (uint32_t i = 0; i < N; i++) {
        int i0 = (int)i - 65, i1 = (int)((i * 17) % 130) - 65;
        in13[i*2+0]=(float)i0; in13[i*2+1]=(float)i1;
        exp13[i] = (float)(i0 >= i1 ? i0 : i1);
    }
    std::vector<float> got13 = prosper::test::run_compute(spv13, in13, N, N);
    uint32_t bad13 = 0; for (uint32_t i=0;i<N&&got13.size()==N;i++) if (std::fabs(got13[i]-exp13[i])>1e-3f) bad13++;
    printf("  kernel13 mismatches=%u (out[3]=%g expect=%g)\n", bad13, got13.size()==N?got13[3]:-1, exp13[3]);
    CHECK(got13.size()==N && bad13==0, "recompiled kernel 13 (signed compare ge_i32 + select) correct");

    // Kernel 14: v_cvt_pkrtz_f16_f32 (VOP2 e32 form, opcode 0x2f). packed = f16(a1)<<16 | f16(a0).
    // Bit-exact verification (inputs chosen f16-exact so RTZ vs RTE rounding agree).
    const uint32_t code14[] = { 0x5e000300u, 0xbf810000u };
    std::vector<uint32_t> spv14 = recompile_valu(code14, sizeof(code14)/sizeof(code14[0]), 2, 0);
    CHECK(!spv14.empty(), "recompiled kernel 14 (v_cvt_pkrtz_f16_f32 e32) -> SPIR-V");
    std::vector<float> in14(N * 2); std::vector<uint32_t> exp14(N);
    for (uint32_t i = 0; i < N; i++) {
        float a0 = (float)(i % 13) * 0.5f - 2.75f, a1 = (float)(i % 11) * 0.5f - 1.75f;  // f16-exact, nonzero
        in14[i*2+0]=a0; in14[i*2+1]=a1;
        exp14[i] = ((uint32_t)f32_to_f16_exact(a1) << 16) | f32_to_f16_exact(a0);
    }
    std::vector<float> got14 = prosper::test::run_compute(spv14, in14, N, N);
    uint32_t bad14 = 0; for (uint32_t i=0;i<N&&got14.size()==N;i++) if (bits_of(got14[i]) != exp14[i]) bad14++;
    printf("  kernel14 mismatches=%u (out[3]=0x%08x expect=0x%08x)\n", bad14, got14.size()==N?bits_of(got14[3]):0, exp14[3]);
    CHECK(got14.size()==N && bad14==0, "recompiled kernel 14 (pkrtz e32) packs f16 bit-exactly");

    // Kernel 14b: same op via the VOP3 e64 form (opcode 0x12f) — the path whose mapping was wrong.
    const uint32_t code14b[] = { 0xd52f0000u, 0x00020300u, 0xbf810000u };
    std::vector<uint32_t> spv14b = recompile_valu(code14b, sizeof(code14b)/sizeof(code14b[0]), 2, 0);
    CHECK(!spv14b.empty(), "recompiled kernel 14b (v_cvt_pkrtz_f16_f32 e64 / 0x12f) -> SPIR-V");
    std::vector<float> got14b = prosper::test::run_compute(spv14b, in14, N, N);
    uint32_t bad14b = 0; for (uint32_t i=0;i<N&&got14b.size()==N;i++) if (bits_of(got14b[i]) != exp14[i]) bad14b++;
    printf("  kernel14b mismatches=%u\n", bad14b);
    CHECK(got14b.size()==N && bad14b==0, "recompiled kernel 14b (pkrtz e64 / VOP3 0x12f) packs f16 bit-exactly");

    // Kernel 15: v_bfrev_b32 (VOP1 0x38). u0=(uint)a0; out_bits = bitreverse(u0). Bit-exact.
    const uint32_t code15[] = { 0x7e000f00u, 0x7e007100u, 0xbf810000u };
    std::vector<uint32_t> spv15 = recompile_valu(code15, sizeof(code15)/sizeof(code15[0]), 1, 0);
    CHECK(!spv15.empty(), "recompiled kernel 15 (v_bfrev_b32) -> SPIR-V");
    std::vector<float> in15(N); std::vector<uint32_t> exp15(N);
    for (uint32_t i = 0; i < N; i++) { in15[i] = (float)i; exp15[i] = bitrev32(i); }
    std::vector<float> got15 = prosper::test::run_compute(spv15, in15, N, N);
    uint32_t bad15 = 0; for (uint32_t i=0;i<N&&got15.size()==N;i++) if (bits_of(got15[i]) != exp15[i]) bad15++;
    printf("  kernel15 mismatches=%u (out[1]=0x%08x expect=0x%08x)\n", bad15, got15.size()==N?bits_of(got15[1]):0, exp15[1]);
    CHECK(got15.size()==N && bad15==0, "recompiled kernel 15 (v_bfrev_b32) reverses bits exactly");

    // Kernel 16: v_cmpx_gt_u32 (0xd4) — EXEC/predication. exec = (u0 > u1); then out = (float)(u0+u1),
    // but the store is EXEC-predicated: lanes with u0<=u1 don't write, so they keep the zero-inited slot.
    const uint32_t code16[] = {
        0x7e000f00u, 0x7e020f01u, 0x7da80300u, 0x4a040300u, 0x7e040d02u, 0xbf810000u,
    };
    std::vector<uint32_t> spv16 = recompile_valu(code16, sizeof(code16)/sizeof(code16[0]), 2, /*out_vgpr*/2);
    CHECK(!spv16.empty(), "recompiled kernel 16 (v_cmpx_gt_u32 + predicated store) -> SPIR-V");
    std::vector<float> in16(N * 2), exp16(N);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t u0 = i % 17, u1 = i % 13;
        in16[i*2+0]=(float)u0; in16[i*2+1]=(float)u1;
        exp16[i] = (u0 > u1) ? (float)(u0 + u1) : 0.0f;   // masked lanes keep the zero-inited output
    }
    std::vector<float> got16 = prosper::test::run_compute(spv16, in16, N, N);
    uint32_t bad16 = 0, active = 0, masked = 0;
    for (uint32_t i=0;i<N&&got16.size()==N;i++) { if (std::fabs(got16[i]-exp16[i])>1e-3f) bad16++;
        if ((i%17) > (i%13)) active++; else masked++; }
    printf("  kernel16 mismatches=%u (active=%u masked=%u, out[1]=%g exp=%g)\n", bad16, active, masked,
           got16.size()==N?got16[1]:-1, exp16[1]);
    CHECK(active > 0 && masked > 0, "kernel 16 exercises BOTH active and masked lanes");
    CHECK(got16.size()==N && bad16==0, "recompiled kernel 16 (EXEC-predicated store via v_cmpx) correct");

    // Kernel 17: EXEC per-write predication + restore. v3=7; v_cmpx_gt narrows EXEC to (u0>u1);
    // v_add v3=u0+u1 (masked lanes must KEEP 7, proving per-write predication); s_mov_b64 exec,-1
    // restores all lanes; cvt+store for everyone. Masked -> 7.0, active -> u0+u1.
    const uint32_t code17[] = {
        0x7e000f00u, 0x7e020f01u, 0x7e060287u, 0x7da80300u, 0x4a060300u, 0xbefe04c1u, 0x7e060d03u, 0xbf810000u,
    };
    std::vector<uint32_t> spv17 = recompile_valu(code17, sizeof(code17)/sizeof(code17[0]), 2, /*out_vgpr*/3);
    CHECK(!spv17.empty(), "recompiled kernel 17 (EXEC predication + s_mov_b64 exec restore) -> SPIR-V");
    std::vector<float> in17(N * 2), exp17(N);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t u0 = i % 17, u1 = i % 13;
        in17[i*2+0]=(float)u0; in17[i*2+1]=(float)u1;
        exp17[i] = (u0 > u1) ? (float)(u0 + u1) : 7.0f;   // masked lanes keep the pre-cmpx v_mov value
    }
    std::vector<float> got17 = prosper::test::run_compute(spv17, in17, N, N);
    uint32_t bad17 = 0, act17 = 0, msk17 = 0;
    for (uint32_t i=0;i<N&&got17.size()==N;i++){ if (std::fabs(got17[i]-exp17[i])>1e-3f) bad17++;
        if ((i%17)>(i%13)) act17++; else msk17++; }
    printf("  kernel17 mismatches=%u (active=%u masked=%u, out[1]=%g exp=%g)\n", bad17, act17, msk17,
           got17.size()==N?got17[1]:-1, exp17[1]);
    CHECK(act17>0 && msk17>0, "kernel 17 exercises both active and masked lanes");
    CHECK(got17.size()==N && bad17==0, "recompiled kernel 17 (per-write predication + EXEC restore) correct");

    // Kernel 17b: s_not_b64 complements a VCC lane mask before saveexec. DOLL's failed
    // 120x68 LUT producer uses the exact gfx1030 encoding below (`s_not_b64 vcc,vcc`).
    // v3=7; vcc=(u0>u1); vcc=!vcc; saveexec(vcc); v3=u0+u1; restore; output v3.
    const uint32_t code17b[] = {
        0x7e000f00u, 0x7e020f01u, 0x7e060287u, 0x7d880300u, 0xbeea086au,
        0xbe80246au, 0xbf880001u, 0x4a060300u, 0xbefe0400u, 0x7e060d03u, 0xbf810000u,
    };
    std::vector<uint32_t> spv17b = recompile_valu(
        code17b, sizeof(code17b)/sizeof(code17b[0]), 2, /*out_vgpr*/3);
    CHECK(!spv17b.empty(), "recompiled kernel 17b (s_not_b64 VCC complement) -> SPIR-V");
    std::vector<float> in17b(N * 2), exp17b(N);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t u0 = i % 17, u1 = i % 13;
        in17b[i*2+0]=(float)u0; in17b[i*2+1]=(float)u1;
        exp17b[i] = (u0 <= u1) ? (float)(u0 + u1) : 7.0f;
    }
    std::vector<float> got17b = prosper::test::run_compute(spv17b, in17b, N, N);
    uint32_t bad17b = 0, act17b = 0;
    for (uint32_t i=0;i<N&&got17b.size()==N;i++) {
        if (std::fabs(got17b[i]-exp17b[i])>1e-3f) bad17b++;
        if ((i%17)<=(i%13)) act17b++;
    }
    CHECK(act17b>0 && act17b<N, "kernel 17b exercises both complemented-mask outcomes");
    CHECK(got17b.size()==N && bad17b==0, "s_not_b64 complements VCC before EXEC predication");

    // Kernel 17c: compute s_cbranch_execz tests whether the whole wave's EXEC mask is empty.
    // The guarded scalar write is live after the merge, so the safe-linearization shortcut cannot
    // consume this branch. Wave 0 has no surviving lanes and keeps s0=0; wave 1 is active and writes
    // s0=5. The following EXEC restore lets every lane report its wave's scalar result.
    const uint32_t code17c[] = {
        0x7e000f00u, 0x7e020f01u, 0x7d880300u, 0xbeea086au, 0xbefe046au,
        0xbf880001u, 0xbe800385u, 0xbefe04c1u, 0x7e040c00u, 0xbf810000u,
    };
    std::vector<uint32_t> spv17c = recompile_valu(
        code17c, sizeof(code17c)/sizeof(code17c[0]), 2, /*out_vgpr*/2);
    CHECK(!spv17c.empty(), "recompiled kernel 17c (compute wave-wide execz if) -> SPIR-V");
    std::vector<float> in17c(N * 2), exp17c(N);
    for (uint32_t i = 0; i < N; i++) {
        const bool empty_after_not = i < 64;
        in17c[i*2+0] = empty_after_not ? 2.0f : 1.0f;
        in17c[i*2+1] = empty_after_not ? 1.0f : 2.0f;
        exp17c[i] = empty_after_not ? 0.0f : 5.0f;
    }
    std::vector<float> got17c = prosper::test::run_compute(spv17c, in17c, N, N);
    uint32_t bad17c = 0;
    for (uint32_t i=0;i<N&&got17c.size()==N;i++)
        if (std::fabs(got17c[i]-exp17c[i])>1e-3f) bad17c++;
    CHECK(got17c.size()==N && bad17c==0,
          "compute execz reduces exact guest-wave EXEC: empty wave skips, active wave enters");

    // Kernel 17c2: a single forward VCC branch (no extra CFG complexity) must still use the exact
    // guest-wave dispatcher. Only lane 63 sets VCC, deliberately outside a host 8/16/32-lane first
    // subgroup. The scalar VCCZ decision is therefore not taken and every lane writes one.
    const uint32_t code17c2[] = {
        0x7e040280u,              // v_mov_b32 v2, 0
        0x7c020300u,              // v_cmp_lt_f32 vcc, v0, v1
        0xbf860001u,              // s_cbranch_vccz +1
        0x7e040281u,              // v_mov_b32 v2, 1
        0x7e040d02u,              // v_cvt_f32_u32 v2, v2
        0xbf810000u,
    };
    std::vector<uint32_t> spv17c2 = recompile_valu(
        code17c2, std::size(code17c2), 2, /*out_vgpr*/2);
    CHECK(!spv17c2.empty(), "recompiled kernel 17c2 (single exact wave64 VCC vote) -> SPIR-V");
    std::vector<float> in17c2(N * 2), exp17c2(N, 1.0f);
    for (uint32_t i = 0; i < N; ++i) {
        in17c2[i * 2] = (i & 63u) == 63u ? 0.0f : 1.0f;
        in17c2[i * 2 + 1] = 0.5f;
    }
    std::vector<float> got17c2 = prosper::test::run_compute(spv17c2, in17c2, N, N);
    uint32_t bad17c2 = 0;
    for (uint32_t i = 0; i < N && got17c2.size() == N; ++i)
        if (got17c2[i] != exp17c2[i]) ++bad17c2;
    CHECK(got17c2.size() == N && bad17c2 == 0,
          "single VCC branch sees the sole set lane outside the first host subgroup");

    // Kernel 17d: force the generic CFG dispatcher (a varying VCC branch plus a VCC loop) and
    // verify that its wave vote spans all 64 emulated RDNA2 lanes, independent of the host Vulkan
    // subgroup width. Wave 0 has no matching lanes and skips v2=1; wave 1 has one matching lane at
    // its end, deliberately outside llvmpipe's first 8-lane subgroup, so every lane must enter.
    const uint32_t code17d[] = {
        0x7e040280u,              // v_mov_b32 v2, 0
        0x7c020300u,              // v_cmp_lt_f32 vcc, v0, v1
        0xbf860001u,              // s_cbranch_vccz +1
        0x7e040281u,              // v_mov_b32 v2, 1
        0x7d840100u,              // v_cmp_eq_u32 vcc, v0, v0 (always true)
        0xbf870001u,              // s_cbranch_vccnz +1
        0xbf82fffdu,              // s_branch -3
        0x7e040d02u,              // v_cvt_f32_u32 v2, v2
        0xbf810000u,              // s_endpgm
    };
    std::vector<uint32_t> spv17d = recompile_valu(
        code17d, sizeof(code17d)/sizeof(code17d[0]), 2, /*out_vgpr*/2);
    CHECK(!spv17d.empty(), "recompiled kernel 17d (wave64 CFG dispatcher) -> SPIR-V");
    std::vector<float> in17d(N * 2), exp17d(N);
    for (uint32_t i = 0; i < N; i++) {
        in17d[i*2+0] = (i == N - 1) ? 0.0f : 1.0f;
        in17d[i*2+1] = 0.5f;
        exp17d[i] = i < 64 ? 0.0f : 1.0f;
    }
    std::vector<float> got17d = prosper::test::run_compute(spv17d, in17d, N, N);
    uint32_t bad17d = 0;
    for (uint32_t i=0;i<N&&got17d.size()==N;i++)
        if (std::fabs(got17d[i]-exp17d[i])>1e-3f) bad17d++;
    printf("  kernel17d mismatches=%u (out[63]=%g out[64]=%g out[127]=%g)\n",
           bad17d, got17d.size()==N?got17d[63]:-1, got17d.size()==N?got17d[64]:-1,
           got17d.size()==N?got17d[127]:-1);
    CHECK(got17d.size()==N && bad17d==0,
          "CFG dispatcher emulates a 64-lane wave across narrower Vulkan subgroups");
    ComputeShaderConfig native_cfg17d;
    native_cfg17d.local_x = 8;
    native_cfg17d.local_y = 8;
    native_cfg17d.wave_size = 64;
    native_cfg17d.native_subgroup_size = 64;
    const std::vector<uint32_t> native_spv17d = recompile_compute(
        code17d, std::size(code17d), nullptr, native_cfg17d);
    CHECK(!native_spv17d.empty() && count_spirv_opcode(native_spv17d, 335) >= 2,
          "exact-wave CFG dispatcher lowers votes and liveness to native subgroup-any");
    CHECK(count_spirv_opcode(native_spv17d, 224) == 0,
          "exact-wave CFG dispatcher emits no workgroup control barrier");
    CHECK(has_spirv_builtin(native_spv17d, 40) &&
              has_spirv_builtin(native_spv17d, 41) &&
              count_spirv_opcode(native_spv17d, 134) >= 2 &&
              count_spirv_opcode(native_spv17d, 137) >= 2,
          "exact-wave compute remaps local coordinates in full-subgroup lane order");
    CHECK(has_spirv_local_size(native_spv17d, 64, 1, 1),
          "multidimensional guest wave flattens Vulkan LocalSize X for full subgroups");

    // Kernel 17d2: Astro's mask-priority sequence compares a VOPC-produced SGPR pair against zero,
    // then uses that wave-uniform SCC in s_cselect_b64.  Keep the irreducible prefix so this exercises
    // the generic dispatcher.  Wave 0's mask is empty and selects all lanes; wave 1 has one set bit in
    // lane 63 (outside a narrow host subgroup) and must select none.
    const uint32_t code17d2[] = {
        0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u,
        0x7d840100u, 0xbf870001u, 0xbf82fffdu,
        0x7c020300u,              // v_cmp_lt_f32 vcc, v0, v1
        0xbe82046au,              // s_mov_b64 s[2:3], vcc
        0xbf128002u,              // s_cmp_eq_u64 s[2:3], 0
        0x858480c1u,              // s_cselect_b64 s[4:5], -1, 0
        0xbefe0404u,              // s_mov_b64 exec, s[4:5]
        0x7e040287u,              // v_mov_b32 v2, 7 (predicated by selected wave mask)
        0xbefe04c1u,              // s_mov_b64 exec, -1
        0x7e040d02u, 0xbf810000u,
    };
    std::vector<uint32_t> spv17d2 = recompile_valu(
        code17d2, std::size(code17d2), 2, /*out_vgpr*/2);
    CHECK(!spv17d2.empty(), "#825: recompiled CFG wave-mask u64 compare/cselect -> SPIR-V");
    std::vector<float> got17d2 = prosper::test::run_compute(spv17d2, in17d, N, N);
    uint32_t bad17d2 = 0;
    for (uint32_t i = 0; i < N && got17d2.size() == N; ++i) {
        const float expected = i < 64 ? 7.0f : 1.0f;
        if (got17d2[i] != expected) ++bad17d2;
    }
    CHECK(got17d2.size() == N && bad17d2 == 0,
          "#825: dispatcher reduces mask==0 across the full emulated wave64");

    // Kernel 17e: the same irreducible-for-the-narrow-structurizer CFG followed by MBCNT. The
    // cross-lane pair must execute in the dispatcher's uniform common phase rather than beneath a
    // switch case, where OpControlBarrier would be invalid and could deadlock a real Vulkan driver.
    const uint32_t code17e[] = {
        0x7e040280u,              // v_mov_b32 v2, 0
        0x7c020300u,              // v_cmp_lt_f32 vcc, v0, v1
        0xbf860001u,              // s_cbranch_vccz +1
        0x7e040281u,              // v_mov_b32 v2, 1
        0x7d840100u,              // v_cmp_eq_u32 vcc, v0, v0
        0xbf870001u,              // s_cbranch_vccnz +1
        0xbf82fffdu,              // s_branch -3
        0xd7650003u, 0x0001007eu, // v_mbcnt_lo_u32_b32 v3, exec_lo, 0
        0xd7660003u, 0x0002067fu, // v_mbcnt_hi_u32_b32 v3, exec_hi, v3
        0x7e060d03u,              // v_cvt_f32_u32 v3, v3
        0xbf810000u,
    };
    std::vector<uint32_t> spv17e = recompile_valu(
        code17e, std::size(code17e), 2, /*out_vgpr*/3);
    CHECK(!spv17e.empty(), "#825: recompiled CFG-dispatch MBCNT kernel -> SPIR-V");
    std::vector<float> got17e = prosper::test::run_compute(spv17e, in17d, N, N);
    uint32_t bad17e = 0;
    for (uint32_t i = 0; i < N && got17e.size() == N; ++i)
        if (std::fabs(got17e[i] - static_cast<float>(i % 64)) > 1e-3f) ++bad17e;
    CHECK(got17e.size() == N && bad17e == 0,
          "#825: CFG dispatcher synchronizes MBCNT across each emulated wave64");
    const std::vector<uint32_t> native_spv17e = recompile_compute(
        code17e, std::size(code17e), nullptr, native_cfg17d);
    CHECK(!native_spv17e.empty() && count_spirv_opcode(native_spv17e, 349) >= 1 &&
              count_spirv_opcode(native_spv17e, 224) == 0,
          "exact-wave CFG MBCNT uses subgroup scans without workgroup barriers");

    // Kernel 17f: DS_APPEND in the same generic dispatcher. The LDS initial value is intentionally
    // unspecified; this test verifies that the generated module executes to completion with the
    // append atomic/barriers in uniform control flow. Exact append values are checked by kernel 32b5.
    const uint32_t code17f[] = {
        0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u,
        0x7d840100u, 0xbf870001u, 0xbf82fffdu,
        0xbefc0380u,                          // s_mov_b32 m0, 0
        0xd8f80008u, 0x03000000u,           // ds_append v3 offset:8 (LDS)
        0x7e060d03u, 0xbf810000u,
    };
    std::vector<uint32_t> spv17f = recompile_valu(
        code17f, std::size(code17f), 2, /*out_vgpr*/3);
    CHECK(!spv17f.empty(), "#825: recompiled CFG-dispatch DS_APPEND kernel -> SPIR-V");
    std::vector<float> got17f = prosper::test::run_compute(spv17f, in17d, N, N);
    CHECK(got17f.size() == N,
          "#825: Vulkan executes dispatcher DS_APPEND with uniform barriers");
    const std::vector<uint32_t> native_spv17f = recompile_compute(
        code17f, std::size(code17f), nullptr, native_cfg17d);
    CHECK(!native_spv17f.empty() && count_spirv_opcode(native_spv17f, 349) >= 3 &&
              count_spirv_opcode(native_spv17f, 224) == 0,
          "exact-wave CFG DS_APPEND uses subgroup reduction/broadcast without barriers");

    // Kernel 18: the REAL if-then idiom with a forward branch. v3=7; vcc=(u0>u1);
    // s_and_saveexec_b64 s[0:1],vcc (save exec, exec=vcc); s_cbranch_execz skip (forward -> no-op);
    // v_add v3=u0+u1 (predicated); skip: s_mov_b64 exec,s[0:1] (restore); cvt+store. Same result as
    // k17 but driven by s_and_saveexec + a real s_cbranch_execz (proving forward branches no-op).
    const uint32_t code18[] = {
        0x7e000f00u, 0x7e020f01u, 0x7e060287u, 0x7d880300u, 0xbe80246au, 0xbf880001u,
        0x4a060300u, 0xbefe0400u, 0x7e060d03u, 0xbf810000u,
    };
    std::vector<uint32_t> spv18 = recompile_valu(code18, sizeof(code18)/sizeof(code18[0]), 2, /*out_vgpr*/3);
    CHECK(!spv18.empty(), "recompiled kernel 18 (s_and_saveexec + s_cbranch_execz if-then) -> SPIR-V");
    std::vector<float> in18(N * 2), exp18(N);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t u0 = i % 17, u1 = i % 13;
        in18[i*2+0]=(float)u0; in18[i*2+1]=(float)u1;
        exp18[i] = (u0 > u1) ? (float)(u0 + u1) : 7.0f;
    }
    std::vector<float> got18 = prosper::test::run_compute(spv18, in18, N, N);
    uint32_t bad18 = 0, act18 = 0;
    for (uint32_t i=0;i<N&&got18.size()==N;i++){ if (std::fabs(got18[i]-exp18[i])>1e-3f) bad18++; if((i%17)>(i%13)) act18++; }
    printf("  kernel18 mismatches=%u (active=%u, out[1]=%g exp=%g)\n", bad18, act18, got18.size()==N?got18[1]:-1, exp18[1]);
    CHECK(got18.size()==N && bad18==0, "recompiled kernel 18 (real saveexec+cbranch_execz if-then) correct");

    // Astro's exact inverse-mask form: save EXEC in s[22:23], then EXEC &= ~VCC. The guarded add
    // therefore runs for u0<=u1, after which the original EXEC mask is restored from s[22:23].
    const uint32_t code18b[] = {
        0x7e000f00u, 0x7e020f01u, 0x7e060287u, 0x7d880300u,
        0xbe96376au, 0xbf880001u, 0x4a060300u, 0xbefe0416u,
        0x7e060d03u, 0xbf810000u,
    };
    std::vector<uint32_t> spv18b = recompile_valu(
        code18b, sizeof(code18b)/sizeof(code18b[0]), 2, /*out_vgpr*/3);
    CHECK(!spv18b.empty(), "recompiled kernel 18b (Astro s_andn1_saveexec_b64) -> SPIR-V");
    std::vector<float> exp18b(N);
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t u0 = i % 17, u1 = i % 13;
        exp18b[i] = (u0 <= u1) ? (float)(u0 + u1) : 7.0f;
    }
    std::vector<float> got18b = prosper::test::run_compute(spv18b, in18, N, N);
    uint32_t bad18b = 0;
    for (uint32_t i=0;i<N&&got18b.size()==N;i++)
        if (std::fabs(got18b[i]-exp18b[i])>1e-3f) ++bad18b;
    CHECK(got18b.size()==N && bad18b==0,
          "kernel 18b saves EXEC, applies ~VCC, and restores the original mask");

    // Astro Bot's world-map material PS uses the exact gfx1030 ORN2 form with VCC as the saved
    // destination. Start with EXEC=M, deliberately replace VCC with !M, then ORN2-save through an
    // all-on source: the operation must save M back into VCC and widen EXEC to all lanes. The first
    // write therefore reaches everyone; restoring EXEC from VCC restricts the second write to M.
    const uint32_t code18c[] = {
        0x7e000f00u, 0x7e020f01u, 0x7e060280u, 0x7d880300u,
        0xbe82246au, 0xbeea086au, 0xbeea2802u, 0x7e060285u,
        0xbefe046au, 0x7e060287u, 0xbefe0402u, 0x7e060d03u,
        0xbf810000u,
    };
    std::vector<uint32_t> spv18c = recompile_valu(
        code18c, std::size(code18c), 2, /*out_vgpr*/3);
    CHECK(!spv18c.empty(),
          "recompiled kernel 18c (Astro s_orn2_saveexec_b64 VCC form) -> SPIR-V");
    std::vector<float> exp18c(N);
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t u0 = i % 17, u1 = i % 13;
        exp18c[i] = (u0 > u1) ? 7.0f : 5.0f;
    }
    std::vector<float> got18c = prosper::test::run_compute(spv18c, in18, N, N);
    uint32_t bad18c = 0;
    for (uint32_t i=0;i<N&&got18c.size()==N;i++)
        if (std::fabs(got18c[i]-exp18c[i])>1e-3f) ++bad18c;
    CHECK(got18c.size()==N && bad18c==0,
          "kernel 18c widens EXEC with S0|~EXEC and saves the old mask into VCC");

    // Kernel 19: scalar s_bfe_u32. s0=0xF0; s1=bfe_u(s0, off=4,width=4)=(0xF0>>4)&0xF=0xF=15;
    // v2=(float)s1; out = a0 + 15. Proves scalar bitfield-extract (a top real-shader blocker).
    const uint32_t code19[] = {
        0xbe8003ffu, 0x000000f0u, 0x9381ff00u, 0x00040004u, 0x7e040c01u, 0x06000500u, 0xbf810000u,
    };
    std::vector<uint32_t> spv19 = recompile_valu(code19, sizeof(code19)/sizeof(code19[0]), 1, 0);
    CHECK(!spv19.empty(), "recompiled kernel 19 (s_bfe_u32) -> SPIR-V");
    std::vector<float> in19(N), exp19(N);
    for (uint32_t i = 0; i < N; i++) { in19[i] = (float)i; exp19[i] = (float)i + 15.0f; }
    std::vector<float> got19 = prosper::test::run_compute(spv19, in19, N, N);
    uint32_t bad19 = 0; for (uint32_t i=0;i<N&&got19.size()==N;i++) if (std::fabs(got19[i]-exp19[i])>1e-3f) bad19++;
    printf("  kernel19 mismatches=%u (out[3]=%g exp=%g)\n", bad19, got19.size()==N?got19[3]:-1, exp19[3]);
    CHECK(got19.size()==N && bad19==0, "recompiled kernel 19 (s_bfe_u32 offset/width extract) correct");

    // Kernel 20: SMEM scalar loads from a bound constant buffer. s0=s_load_dword(cbuf,off4)=cbuf[1];
    // s1=s_buffer_load_dword(cbuf,off8)=cbuf[2]; s0=s0+s1; out=(float)s0. With cbuf={10,20,30,40} => 50.
    const uint32_t code20[] = {
        0xf4000001u, 0xfa000004u, 0xf4200042u, 0xfa000008u, 0x80000100u, 0x7e000c00u, 0xbf810000u,
    };
    std::vector<uint32_t> spv20 = recompile_valu(code20, sizeof(code20)/sizeof(code20[0]), 1, 0);
    CHECK(!spv20.empty(), "recompiled kernel 20 (s_load_dword + s_buffer_load_dword) -> SPIR-V");
    std::vector<float> in20(N, 0.0f);
    std::vector<uint32_t> cbuf20 = { 10u, 20u, 30u, 40u };
    std::vector<float> got20 = prosper::test::run_compute(spv20, in20, N, N, cbuf20);
    uint32_t bad20 = 0; for (uint32_t i=0;i<N&&got20.size()==N;i++) if (std::fabs(got20[i]-50.0f)>1e-3f) bad20++;
    printf("  kernel20 mismatches=%u (out[0]=%g expect=50)\n", bad20, got20.size()==N?got20[0]:-1);
    CHECK(got20.size()==N && bad20==0, "recompiled kernel 20 (SMEM: cbuf[1]+cbuf[2] from constant buffer) correct");

    // Kernel 20b (#149): a register-SOFFSET s_buffer_load must be REJECTED, not silently translated
    // with the immediate alone (the register offset is unmodeled). s_buffer_load_dword s0, s[..], s8.
    // Same first-load prefix as kernel 20 so the difference is purely the register SOFFSET.
    const uint32_t code20b[] = {
        0xf4000001u, 0xfa000004u, 0xf4200002u, 0x10000008u, 0x7e000c00u, 0xbf810000u,
    };
    CHECK(recompile_valu(code20b, sizeof(code20b)/sizeof(code20b[0]), 1, 0).empty(),
          "kernel 20b (s_buffer_load with a register SOFFSET) is REJECTED (not silently mistranslated)");

    // Kernel 21: MUBUF per-lane buffer_load_dword (the vertex-fetch mechanism). v0=(uint)gid;
    // v0<<=2 (byte offset); buffer_load_dword v0, v0 offen -> cbuf[gid]; out=(float)cbuf[gid].
    const uint32_t code21[] = {
        0x7e000f00u, 0x34000082u, 0xe0301000u, 0x80020000u, 0x7e000d00u, 0xbf810000u,
    };
    std::vector<uint32_t> spv21 = recompile_valu(code21, sizeof(code21)/sizeof(code21[0]), 1, 0);
    CHECK(!spv21.empty(), "recompiled kernel 21 (MUBUF buffer_load_dword) -> SPIR-V");
    std::vector<float> in21(N); std::vector<uint32_t> cbuf21(N);
    for (uint32_t i = 0; i < N; i++) { in21[i] = (float)i; cbuf21[i] = 100u + i; }
    std::vector<float> got21 = prosper::test::run_compute(spv21, in21, N, N, cbuf21);
    uint32_t bad21 = 0; for (uint32_t i=0;i<N&&got21.size()==N;i++) if (std::fabs(got21[i]-(float)(100u+i))>1e-3f) bad21++;
    printf("  kernel21 mismatches=%u (out[7]=%g expect=107)\n", bad21, got21.size()==N?got21[7]:-1);
    CHECK(got21.size()==N && bad21==0, "recompiled kernel 21 (MUBUF per-lane buffer_load -> cbuf[gid]) correct");

    // Kernel 22: MULTI-buffer SMEM via descriptor provenance (the resource-binding contract). Two V#
    // descriptors loaded from SRT offsets 0x20 and 0x40; a ShaderResourceTable maps them to distinct
    // bindings (2 and 3); the two s_buffer_loads must route to two DIFFERENT constant buffers.
    const uint32_t code22[] = {
        0xf4080100u, 0xfa000020u, 0xf4080200u, 0xfa000040u, 0xf4200002u, 0xfa000004u,
        0xf4200044u, 0xfa000004u, 0x80000100u, 0x7e000c00u, 0xbf810000u,
    };
    ShaderResourceTable rt22;
    rt22.resources.push_back({ResourceClass::ConstantBuffer, DataFormat::Float32, 1, /*binding*/2, 0, 0, 0, /*srt*/0x20});
    rt22.resources.push_back({ResourceClass::ConstantBuffer, DataFormat::Float32, 1, /*binding*/3, 0, 0, 0, /*srt*/0x40});
    std::vector<uint32_t> spv22 = recompile_valu(code22, sizeof(code22)/sizeof(code22[0]), 1, 0, &rt22);
    CHECK(!spv22.empty(), "recompiled kernel 22 (multi-buffer SMEM via provenance) -> SPIR-V");
    std::vector<float> in22(N, 0.0f);
    std::vector<uint32_t> cbuf22a(24, 0u), cbuf22b(24, 0u); cbuf22a[1] = 20u; cbuf22b[1] = 300u;
    std::vector<float> got22 = prosper::test::run_compute(spv22, in22, N, N, cbuf22a, cbuf22b);
    uint32_t bad22 = 0; for (uint32_t i=0;i<N&&got22.size()==N;i++) if (std::fabs(got22[i]-320.0f)>1e-3f) bad22++;
    printf("  kernel22 mismatches=%u (out[0]=%g expect=320: cbuf0[1]=20 + cbuf1[1]=300)\n", bad22, got22.size()==N?got22[0]:-1);
    CHECK(got22.size()==N && bad22==0, "recompiled kernel 22 (provenance routes s_buffer_loads to bindings 2 & 3) correct");

    // DOLL scene VS: the compiler loads several V# descriptors into separate SGPR ranges, then
    // reuses one SRSRC range by copying a selected descriptor into it with s_mov_b32. Descriptor
    // provenance must follow each scalar move. Keeping the destination's previous SRT tag routes
    // every later raw MUBUF through the first constant buffer and collapses the scene triangles.
    const uint32_t code22move[] = {
        0xf4080200u, 0xfa000020u,                         // s_load_dwordx4 s[8:11],  s[0:1], 0x20
        0xf4080300u, 0xfa000040u,                         // s_load_dwordx4 s[12:15], s[0:1], 0x40
        0xbe88030cu, 0xbe89030du, 0xbe8a030eu, 0xbe8b030fu, // s_mov_b32 s[8:11], s[12:15]
        0xe0300000u, 0x84020000u,                         // buffer_load_dword v0, off, s[8:11], 4
        0x7e000d00u, 0xbf810000u,                         // v_cvt_f32_u32 v0,v0; s_endpgm
    };
    ShaderResourceTable rt22move;
    rt22move.resources.push_back({ResourceClass::ConstantBuffer, DataFormat::Uint32, 1,
                                  /*binding*/2, 0, 0, 0, /*srt*/0x20});
    rt22move.resources.push_back({ResourceClass::ConstantBuffer, DataFormat::Uint32, 1,
                                  /*binding*/3, 0, 0, 0, /*srt*/0x40});
    std::vector<uint32_t> spv22move = recompile_valu(
        code22move, sizeof(code22move) / sizeof(code22move[0]), 1, 0, &rt22move);
    std::vector<uint32_t> decoy22move(4, 0u), wanted22move(4, 0u);
    decoy22move[1] = 20u; wanted22move[1] = 300u;
    std::vector<float> got22move = prosper::test::run_compute(
        spv22move, in22, N, N, decoy22move, wanted22move);
    uint32_t bad22move = 0;
    for (uint32_t i = 0; i < N && got22move.size() == N; i++)
        if (std::fabs(got22move[i] - 300.0f) > 1e-3f) bad22move++;
    CHECK(!spv22move.empty() && got22move.size() == N && bad22move == 0,
          "descriptor provenance follows s_mov_b32 when an SRSRC range is reused");

    // #515: a stage whose decoded resources start at 32/33 must not retain dead fallback-binding-2
    // loads for the preceding raw descriptor fetches. Those s_load_dwordx4 results are provenance;
    // the actual s_buffer_load data operations below resolve to the table's high bindings.
    ShaderResourceTable rt22hi;
    rt22hi.resources.push_back({ResourceClass::ConstantBuffer, DataFormat::Float32, 1, 32, 0, 96, 0, 0x20});
    rt22hi.resources.push_back({ResourceClass::ConstantBuffer, DataFormat::Float32, 1, 33, 0, 96, 0, 0x40});
    std::vector<uint32_t> spv22hi = recompile_valu(code22, sizeof(code22)/sizeof(code22[0]), 1, 0, &rt22hi);
    auto manifest22hi = validate_spirv_descriptor_interface(
        spv22hi, &rt22hi, 0, SpirvShaderStage::Compute, false);
    bool has2 = false, has32 = false, has33 = false;
    for (const auto& d : manifest22hi.descriptors) {
        printf("  kernel22hi manifest: set=%u binding=%u type=%s required=%llu%s\n",
               d.set, d.binding, spirv_descriptor_kind_name(d.kind),
               (unsigned long long)d.required_bytes, d.dynamic_access ? "+dynamic" : "");
        has2 |= d.binding == 2; has32 |= d.binding == 32; has33 |= d.binding == 33;
    }
    CHECK(!spv22hi.empty() && !has2 && has32 && has33,
          "#515: decoded descriptor fetches add no dead binding-2 contract; data loads use 32/33");

    // #319: key-less/collided descriptor-table V#s carry exact consuming-PC provenance. Scalar
    // loads must consult it; otherwise the real resource is present while generated code reads b2.
    const uint32_t code22pc[] = {
        0xf4200002u, 0xfa000004u, 0x7e000c00u, 0xbf810000u,
    };
    ShaderResourceTable rt22pc;
    { ShaderResource cb{}; cb.cls = ResourceClass::ConstantBuffer; cb.format = DataFormat::Uint32;
      cb.num_components = 1; cb.binding = 3; cb.fetch_pc = 0; rt22pc.resources.push_back(cb); }
    std::vector<uint32_t> spv22pc = recompile_valu(
        code22pc, sizeof(code22pc)/sizeof(code22pc[0]), 1, 0, &rt22pc);
    std::vector<uint32_t> cbuf22pc0(4, 0u), cbuf22pc1(4, 0u);
    cbuf22pc0[1] = 20u; cbuf22pc1[1] = 300u;
    std::vector<float> got22pc = prosper::test::run_compute(
        spv22pc, in22, N, N, cbuf22pc0, cbuf22pc1);
    uint32_t bad22pc = 0;
    for (uint32_t i = 0; i < N && got22pc.size() == N; i++)
        if (std::fabs(got22pc[i] - 300.0f) > 1e-3f) bad22pc++;
    CHECK(!spv22pc.empty() && got22pc.size() == N && bad22pc == 0,
          "#319: pc-only cbuf provenance routes s_buffer_load off fallback binding 2");

    // A runtime resource table makes the legacy binding-2 fallback invalid. If a scalar data load
    // has no pc, SRT, or direct-SGPR provenance, reject the shader instead of emitting an interface
    // that the live renderer cannot bind.
    ShaderResourceTable rt22unresolved;
    { ShaderResource cb{}; cb.cls = ResourceClass::ConstantBuffer; cb.format = DataFormat::Uint32;
      cb.num_components = 1; cb.binding = 32; cb.srt_offset = 0x20;
      rt22unresolved.resources.push_back(cb); }
    std::vector<uint32_t> spv22unresolved = recompile_valu(
        code22pc, sizeof(code22pc)/sizeof(code22pc[0]), 1, 0, &rt22unresolved);
    CHECK(spv22unresolved.empty(),
          "unresolved scalar data load with a runtime table is REJECTED (no binding-2 fallback)");

    ShaderResourceTable rt22fallback;
    { ShaderResource cb{}; cb.cls = ResourceClass::ConstantBuffer; cb.format = DataFormat::Uint32;
      cb.num_components = 1; cb.binding = 2; rt22fallback.resources.push_back(cb); }
    std::vector<uint32_t> spv22fallback = recompile_valu(
        code22pc, sizeof(code22pc)/sizeof(code22pc[0]), 1, 0, &rt22fallback);
    std::vector<float> got22fallback = prosper::test::run_compute(
        spv22fallback, in22, N, N, cbuf22pc1);
    bool good22fallback = got22fallback.size() == N;
    for (uint32_t i = 0; i < N && good22fallback; ++i)
        good22fallback = std::fabs(got22fallback[i] - 300.0f) <= 1e-3f;
    CHECK(!spv22fallback.empty() && good22fallback,
          "unresolved scalar data load may use binding 2 when the runtime table binds it");

    // Kernel 23: buffer_load_format_x FLOAT32 VERTEX FETCH (stage 2 — the real-VS mechanism). v0=(uint)
    // gid (element index); buffer_load_format_x v1, v0, s[8:11] idxen fetches vbuf[gid]; out=v1. The V#
    // descriptor is DIRECT (in user-data SGPR s8) -> resolved via sgpr_base -> VertexBuffer binding 3,
    // stride 4. Float32 -> raw dword. Proves per-lane vertex fetch routed through the resource table.
    const uint32_t code23[] = { 0x7e000f00u, 0xe0002000u, 0x80020100u, 0xbf810000u };
    ShaderResourceTable rt23;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32;
      vb.num_components = 1; vb.binding = 3; vb.stride = 4; vb.sgpr_base = 8; rt23.resources.push_back(vb); }
    std::vector<uint32_t> spv23 = recompile_valu(code23, sizeof(code23)/sizeof(code23[0]), 1, /*out_vgpr*/1, &rt23);
    CHECK(!spv23.empty(), "recompiled kernel 23 (buffer_load_format_x float32 vertex fetch) -> SPIR-V");
    std::vector<float> in23(N), exp23(N); std::vector<uint32_t> vbuf23(N);
    for (uint32_t i = 0; i < N; i++) { in23[i] = (float)i; float f = 1.5f + (float)i;
        std::memcpy(&vbuf23[i], &f, 4); exp23[i] = f; }
    std::vector<float> got23 = prosper::test::run_compute(spv23, in23, N, N, /*cbuf0*/{}, /*cbuf1=vertex buf*/vbuf23);
    uint32_t bad23 = 0; for (uint32_t i=0;i<N&&got23.size()==N;i++) if (std::fabs(got23[i]-exp23[i])>1e-3f) bad23++;
    printf("  kernel23 mismatches=%u (out[5]=%g expect=6.5)\n", bad23, got23.size()==N?got23[5]:-1);
    CHECK(got23.size()==N && bad23==0, "recompiled kernel 23 (float32 vertex fetch via sgpr_base provenance) correct");

    // MTBUF uses the same descriptor/address path, but its gfx1030 instruction owns the combined
    // BUF_FMT. Deliberately give the resource misleading Uint8 metadata: tbuffer_load_format_x's
    // encoded format 22 (32_FLOAT) must still fetch full float dwords.
    const uint32_t code23mt[] = { 0x7e000f00u, 0xe8b02000u, 0x80020100u, 0xbf810000u };
    ShaderResourceTable rt23mt;
    { ShaderResource vb{}; vb.cls = ResourceClass::ConstantBuffer; vb.format = DataFormat::Uint8;
      vb.num_components = 1; vb.binding = 3; vb.stride = 4; vb.sgpr_base = 8; vb.fetch_pc = 1;
      rt23mt.resources.push_back(vb); }
    std::vector<uint32_t> spv23mt = recompile_valu(
        code23mt, sizeof(code23mt)/sizeof(code23mt[0]), 1, 1, &rt23mt);
    std::vector<float> got23mt = prosper::test::run_compute(
        spv23mt, in23, N, N, {}, vbuf23);
    uint32_t bad23mt = 0;
    for (uint32_t i = 0; i < N && got23mt.size() == N; ++i)
        if (std::fabs(got23mt[i] - exp23[i]) > 1e-3f) ++bad23mt;
    CHECK(!spv23mt.empty() && got23mt.size() == N && bad23mt == 0,
          "MTBUF 32_FLOAT load uses the instruction format instead of V# metadata");
    const uint32_t code23mt_badfmt[] = {
        0x7e000f00u, 0xe8002000u, 0x80020100u, 0xbf810000u,
    };
    CHECK(recompile_valu(code23mt_badfmt,
                         sizeof(code23mt_badfmt)/sizeof(code23mt_badfmt[0]),
                         1, 1, &rt23mt).empty(),
          "MTBUF unknown combined format is rejected instead of silently read as raw dwords");
    // gfx1030 MTBUF forces identity selection from its instruction format. A four-component opcode
    // reading a three-component 32_32_32_FLOAT element therefore returns XYZ0 (not MUBUF's XYZ1).
    const uint32_t code23mt_identity[] = {
        0x7e000f00u, 0xea532000u, 0x80020100u, 0xbf810000u,
    };
    ShaderResourceTable rt23mt_identity;
    { ShaderResource vb{}; vb.cls = ResourceClass::ConstantBuffer; vb.format = DataFormat::Float32;
      vb.num_components = 4; vb.binding = 3; vb.stride = 12; vb.sgpr_base = 8; vb.fetch_pc = 1;
      rt23mt_identity.resources.push_back(vb); }
    std::vector<uint32_t> vbuf23mt_identity(N * 3u);
    for (uint32_t i = 0; i < N; ++i) {
        const float xyz[3] = { (float)i, (float)i + 1.0f, (float)i + 2.0f };
        std::memcpy(vbuf23mt_identity.data() + i * 3u, xyz, sizeof(xyz));
    }
    std::vector<uint32_t> spv23mt_identity = recompile_valu(
        code23mt_identity, sizeof(code23mt_identity)/sizeof(code23mt_identity[0]),
        1, 4, &rt23mt_identity);
    std::vector<float> got23mt_identity = prosper::test::run_compute(
        spv23mt_identity, in23, N, N, {}, vbuf23mt_identity);
    bool identity_zero = got23mt_identity.size() == N;
    for (float w : got23mt_identity) identity_zero &= w == 0.0f;
    CHECK(!spv23mt_identity.empty() && identity_zero,
          "MTBUF identity selection fills absent W with zero");

    // A pc-keyed VertexBuffer is not necessarily the NGG v0 fetch prologue. DOLL's skinned scene
    // shaders use a direct structured V# through computed VADDRs (v4/v5/v7/...); the old shortcut
    // replaced all of them with gl_VertexIndex. Pin v1=1 and prove the fetch stays on record 1 for
    // every lane even though exact per-PC provenance is present.
    const uint32_t code23computed[] = {
        0x7e020281u,                 // v_mov_b32 v1, 1
        0xe0002000u, 0x80020201u,   // buffer_load_format_x v2, v1, s[8:11], 0 idxen
        0xbf810000u,
    };
    ShaderResourceTable rt23computed;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32;
      vb.num_components = 1; vb.binding = 3; vb.stride = 4; vb.sgpr_base = 8; vb.fetch_pc = 1;
      rt23computed.resources.push_back(vb); }
    std::vector<uint32_t> spv23computed = recompile_valu(
        code23computed, sizeof(code23computed)/sizeof(code23computed[0]), 1, 2, &rt23computed);
    std::vector<float> got23computed = prosper::test::run_compute(
        spv23computed, in23, N, N, {}, vbuf23);
    uint32_t bad23computed = 0;
    for (uint32_t i = 0; i < N && got23computed.size() == N; ++i)
        if (std::fabs(got23computed[i] - exp23[1]) > 1e-3f) ++bad23computed;
    CHECK(!spv23computed.empty() && got23computed.size() == N && bad23computed == 0,
          "pc-keyed structured vertex fetch preserves its computed non-v0 VADDR");

    // Conversely, a descriptor-loaded attribute remains an NGG fetch-prologue use even when the
    // incomplete prologue left its element index in a non-v0 register. The rewritten SRSRC is the
    // discriminator: the vertex module must reload gl_VertexIndex for the fetch instead of reading
    // record zero from v1. Count loads from the decorated built-in (one ABI seed + one fetch use).
    const uint32_t code23rewritten[] = {
        0xbe880380u,                 // s_mov_b32 s8, 0 (marks the SRSRC range rewritten)
        0x7e020280u,                 // v_mov_b32 v1, 0
        0xe0002000u, 0x80020201u,   // buffer_load_format_x v2, v1, s[8:11], 0 idxen
        0x7e060280u, 0x7e0802f2u,   // v3=0, v4=1.0
        0xf80008cfu, 0x04030201u,   // exp pos0 v[1:4]
        0xbf810000u,
    };
    ShaderResourceTable rt23rewritten;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32;
      vb.num_components = 1; vb.binding = 3; vb.stride = 4; vb.fetch_pc = 2;
      rt23rewritten.resources.push_back(vb); }
    std::vector<uint32_t> spv23rewritten = recompile_vertex(
        code23rewritten, sizeof(code23rewritten)/sizeof(code23rewritten[0]), &rt23rewritten);
    uint32_t vertex_index_id = 0, vertex_index_loads = 0;
    for (size_t word = 5; word < spv23rewritten.size(); ) {
        const uint32_t count = spv23rewritten[word] >> 16;
        const uint32_t opcode = spv23rewritten[word] & 0xffffu;
        if (!count || word + count > spv23rewritten.size()) break;
        // OpDecorate target BuiltIn VertexIndex (71, decoration 11, built-in 42).
        if (opcode == 71 && count >= 4 && spv23rewritten[word + 2] == 11u &&
            spv23rewritten[word + 3] == 42u)
            vertex_index_id = spv23rewritten[word + 1];
        word += count;
    }
    for (size_t word = 5; word < spv23rewritten.size(); ) {
        const uint32_t count = spv23rewritten[word] >> 16;
        const uint32_t opcode = spv23rewritten[word] & 0xffffu;
        if (!count || word + count > spv23rewritten.size()) break;
        if (opcode == 61 && count >= 4 && spv23rewritten[word + 3] == vertex_index_id)
            ++vertex_index_loads; // OpLoad result-type result-id pointer
        word += count;
    }
    CHECK(!spv23rewritten.empty() && vertex_index_id && vertex_index_loads == 2,
          "pc-keyed rewritten SRSRC keeps vertex-index recovery for non-v0 attributes");

    // Kernel 24: UNORM8x4 vertex fetch (stage 3 — packed-format conversion). buffer_load_format_xyzw
    // unpacks a dword into 4 bytes, each normalized /255 into v1..v4. out = v1 + 2*v2 + 3*v3 + 4*v4
    // (distinct weights catch a component swizzle). Proves the recompiler emits the right unpack+scale.
    const uint32_t code24[] = {
        0x7e000f00u, 0xe00c2000u, 0x80020100u, 0x7e0a02f4u, 0x100c0505u, 0x06020d01u,
        0x7e0a02f6u, 0x100c0905u, 0x06020d01u, 0x7e0a02ffu, 0x40400000u, 0x100c0705u, 0x06020d01u, 0xbf810000u,
    };
    ShaderResourceTable rt24;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Unorm8;
      vb.num_components = 4; vb.binding = 3; vb.stride = 4; vb.sgpr_base = 8; rt24.resources.push_back(vb); }
    std::vector<uint32_t> spv24 = recompile_valu(code24, sizeof(code24)/sizeof(code24[0]), 1, /*out_vgpr*/1, &rt24);
    CHECK(!spv24.empty(), "recompiled kernel 24 (buffer_load_format_xyzw unorm8x4) -> SPIR-V");
    std::vector<float> in24(N), exp24(N); std::vector<uint32_t> vbuf24(N);
    for (uint32_t i = 0; i < N; i++) {
        in24[i] = (float)i;
        uint32_t r = i & 0xFF, g = (i*3) & 0xFF, bl = (i*5) & 0xFF, a = (i*7) & 0xFF;
        vbuf24[i] = r | (g<<8) | (bl<<16) | (a<<24);
        exp24[i] = (r + 2.0f*g + 3.0f*bl + 4.0f*a) / 255.0f;
    }
    std::vector<float> got24 = prosper::test::run_compute(spv24, in24, N, N, /*cbuf0*/{}, /*cbuf1=vertex buf*/vbuf24);
    uint32_t bad24 = 0; for (uint32_t i=0;i<N&&got24.size()==N;i++) if (std::fabs(got24[i]-exp24[i])>2e-3f) bad24++;
    printf("  kernel24 mismatches=%u (out[5]=%g expect=%g)\n", bad24, got24.size()==N?got24[5]:-1, got24.size()==N?exp24[5]:-1);
    CHECK(got24.size()==N && bad24==0, "recompiled kernel 24 (unorm8x4 -> 4 normalized floats) correct");

    // The same packed conversion through MTBUF: opcode XYZW and combined format 56 are both carried
    // by the instruction. Resource metadata is intentionally Float32x1 to catch accidental reuse.
    const uint32_t code24mt[] = {
        0x7e000f00u, 0xe9c32000u, 0x80020100u, 0x7e0a02f4u, 0x100c0505u,
        0x06020d01u, 0x7e0a02f6u, 0x100c0905u, 0x06020d01u, 0x7e0a02ffu,
        0x40400000u, 0x100c0705u, 0x06020d01u, 0xbf810000u,
    };
    ShaderResourceTable rt24mt;
    { ShaderResource vb{}; vb.cls = ResourceClass::ConstantBuffer; vb.format = DataFormat::Float32;
      vb.num_components = 1; vb.binding = 3; vb.stride = 4; vb.sgpr_base = 8; vb.fetch_pc = 1;
      rt24mt.resources.push_back(vb); }
    std::vector<uint32_t> spv24mt = recompile_valu(
        code24mt, sizeof(code24mt)/sizeof(code24mt[0]), 1, 1, &rt24mt);
    std::vector<float> got24mt = prosper::test::run_compute(
        spv24mt, in24, N, N, {}, vbuf24);
    uint32_t bad24mt = 0;
    for (uint32_t i = 0; i < N && got24mt.size() == N; ++i)
        if (std::fabs(got24mt[i] - exp24[i]) > 2e-3f) ++bad24mt;
    CHECK(!spv24mt.empty() && got24mt.size() == N && bad24mt == 0,
          "MTBUF 8_8_8_8_UNORM load applies the instruction's packed conversion");

    // Store the same constant normalized vector to every lane through MTBUF. Round-to-even maps
    // 0.5*255 to 128, so each destination dword is RGBA = {0,128,255,255}.
    const uint32_t code24mt_store[] = {
        0x7e020280u, 0x7e0402f0u, 0x7e0602f2u, 0x7e0802f2u, 0x7e0a0f00u,
        0xe9c72000u, 0x80020105u, 0xbf810000u,
    };
    ShaderResourceTable rt24mt_store = rt24mt;
    rt24mt_store.resources[0].fetch_pc = 5;
    std::vector<uint32_t> spv24mt_store = recompile_valu(
        code24mt_store, sizeof(code24mt_store)/sizeof(code24mt_store[0]), 1, 0, &rt24mt_store);
    std::vector<uint32_t> packed_store_in(N, 0), packed_store_out;
    prosper::test::run_compute(spv24mt_store, in24, N, N, {}, packed_store_in,
                               &packed_store_out);
    uint32_t bad24mt_store = 0;
    for (uint32_t word : packed_store_out)
        if (word != 0xffff8000u) ++bad24mt_store;
    CHECK(!spv24mt_store.empty() && packed_store_out.size() == N && bad24mt_store == 0,
          "MTBUF 8_8_8_8_UNORM store packs and predicates instruction-typed components");

    // A wider store opcode does not extend the physical format. BUF_FMT_16_16_FLOAT has identity
    // XY00, so the Z/W source VGPRs must not spill into the next dword of each eight-byte record.
    const uint32_t code24mt_store_xy00[] = {
        0x7e020280u, 0x7e0402f0u, 0x7e0602f2u, 0x7e0802f2u, 0x7e0a0f00u,
        0xe8ef2000u, 0x80020105u, 0xbf810000u,
    };
    ShaderResourceTable rt24mt_store_xy00 = rt24mt_store;
    rt24mt_store_xy00.resources[0].stride = 8;
    std::vector<uint32_t> spv24mt_store_xy00 = recompile_valu(
        code24mt_store_xy00, sizeof(code24mt_store_xy00)/sizeof(code24mt_store_xy00[0]),
        1, 0, &rt24mt_store_xy00);
    std::vector<uint32_t> xy00_store_in(N * 2, 0xdeadbeefu), xy00_store_out;
    prosper::test::run_compute(spv24mt_store_xy00, in24, N, N, {}, xy00_store_in,
                               &xy00_store_out);
    uint32_t bad24mt_store_xy00 = 0;
    for (uint32_t i = 0; i < N && xy00_store_out.size() == N * 2; ++i) {
        if (xy00_store_out[i * 2] != 0x38000000u ||
            xy00_store_out[i * 2 + 1] != 0xdeadbeefu)
            ++bad24mt_store_xy00;
    }
    CHECK(!spv24mt_store_xy00.empty() && xy00_store_out.size() == N * 2 &&
              bad24mt_store_xy00 == 0,
          "MTBUF XYZW store honors XY00 format width and leaves the adjacent dword untouched");

    // Kernel 24atomic (#590): the MUBUF 32-bit atomic RMW family. buffer_atomic_add over a binding-3
    // storage buffer — all N dispatched lanes each add 3 to element 0, so the final memory word is 3N.
    // This exercises the opcode -> SPIR-V-atomic mapping end-to-end through the proven cbuf_atomic_rtn
    // path (real Vulkan atomic accumulation, not just recompile).
    ShaderResourceTable rt_atomic;
    { ShaderResource ab{}; ab.cls = ResourceClass::ConstantBuffer; ab.format = DataFormat::Uint32;
      ab.num_components = 1; ab.binding = 3; ab.stride = 4; ab.sgpr_base = 8; ab.fetch_pc = 1;
      rt_atomic.resources.push_back(ab); }
    const uint32_t code_atomic_add[] = {
        0x7e020283u,               // v_mov_b32 v1, 3                 (the value to add)
        0xe0c80000u, 0x80020100u,  // buffer_atomic_add v1, v0, s[8:11]  (op 0x32, element 0)
        0xbf810000u,
    };
    std::vector<uint32_t> spv_atomic_add = recompile_valu(
        code_atomic_add, std::size(code_atomic_add), 1, 0, &rt_atomic);
    std::vector<uint32_t> atomic_buf_in(1, 0), atomic_buf_out;
    std::vector<float> atomic_dummy(N, 0.0f);
    prosper::test::run_compute(spv_atomic_add, atomic_dummy, N, N, {}, atomic_buf_in, &atomic_buf_out);
    CHECK(!spv_atomic_add.empty() && atomic_buf_out.size() == 1 &&
              atomic_buf_out[0] == (uint32_t)N * 3u,
          "#590: buffer_atomic_add accumulates N lanes x 3 into element 0 (= 3N)");

    // The rest of the 32-bit RMW family recompiles to valid SPIR-V (same emit, different atomic op).
    const uint32_t atomic_op_dw0[] = {
        0xE0C00000u,   // buffer_atomic_swap (0x30)
        0xE0CC0000u,   // buffer_atomic_sub  (0x33)
        0xE0D40000u,   // buffer_atomic_smin (0x35)
        0xE0D80000u,   // buffer_atomic_umin (0x36)
        0xE0DC0000u,   // buffer_atomic_smax (0x37)
        0xE0E40000u,   // buffer_atomic_and  (0x39)
        0xE0E80000u,   // buffer_atomic_or   (0x3a)
        0xE0EC0000u,   // buffer_atomic_xor  (0x3b)
    };
    uint32_t atomic_bad = 0;
    for (uint32_t dw0 : atomic_op_dw0) {
        const uint32_t code[] = { 0x7e020283u, dw0, 0x80020100u, 0xbf810000u };
        if (recompile_valu(code, std::size(code), 1, 0, &rt_atomic).empty()) ++atomic_bad;
    }
    CHECK(atomic_bad == 0,
          "#590: swap/sub/smin/umin/smax/and/or/xor MUBUF atomics all recompile to SPIR-V");

    // Behaviorally verify a bitwise atomic too (catches an op transposition among the recompile-only
    // set): element 0 starts at 4 and every lane ORs 3, so the idempotent result is 4|3 = 7 — distinct
    // from add (4+3N), max (4), swap (3), and (0), and xor (4 for even N).
    const uint32_t code_atomic_or[] = {
        0x7e020283u,               // v_mov_b32 v1, 3
        0xe0e80000u, 0x80020100u,  // buffer_atomic_or v1, v0, s[8:11]  (op 0x3a)
        0xbf810000u,
    };
    std::vector<uint32_t> spv_atomic_or = recompile_valu(
        code_atomic_or, std::size(code_atomic_or), 1, 0, &rt_atomic);
    std::vector<uint32_t> or_buf_in(1, 4u), or_buf_out;
    prosper::test::run_compute(spv_atomic_or, atomic_dummy, N, N, {}, or_buf_in, &or_buf_out);
    CHECK(!spv_atomic_or.empty() && or_buf_out.size() == 1 && or_buf_out[0] == 7u,
          "#590: buffer_atomic_or is idempotent (4 | 3 = 7), distinct from add/max/swap");

    // Kernel 24store (#590): integer sub-dword FORMAT store at a runtime byte address. Every lane stores
    // its gid as a Uint16 into element[gid] of a stride-2 buffer, so elements 2k and 2k+1 SHARE dword k.
    // A plain read-modify-write would race; the atomicAnd(clear field)+atomicOr(set field) of each lane's
    // disjoint 16-bit field must land BOTH values -> dword k == 2k | ((2k+1)<<16). Real Vulkan run.
    ShaderResourceTable rt_store16;
    { ShaderResource sb{}; sb.cls = ResourceClass::ConstantBuffer; sb.format = DataFormat::Uint16;
      sb.num_components = 1; sb.binding = 3; sb.stride = 2; sb.sgpr_base = 8; sb.fetch_pc = 2;
      rt_store16.resources.push_back(sb); }
    const uint32_t code_store16[] = {
        0x7e000f00u,               // v_cvt_u32_f32 v0, v0   (shell input float gid -> int index)
        0x7e020300u,               // v_mov_b32 v1, v0       (value = gid)
        0xe0102000u, 0x80020100u,  // buffer_store_format_x v1, v0, s[8:11], idxen  (Uint16, stride 2)
        0xbf810000u,
    };
    std::vector<uint32_t> spv_store16 = recompile_valu(
        code_store16, std::size(code_store16), 1, 0, &rt_store16);
    std::vector<uint32_t> store16_in(N / 2, 0), store16_out;
    std::vector<float> store16_dummy(N); for (uint32_t i = 0; i < N; ++i) store16_dummy[i] = (float)i;
    prosper::test::run_compute(spv_store16, store16_dummy, N, N, {}, store16_in, &store16_out);
    uint32_t bad_store16 = 0;
    for (uint32_t k = 0; k < N / 2 && store16_out.size() == N / 2; ++k)
        if (store16_out[k] != ((2u * k) | ((2u * k + 1u) << 16))) ++bad_store16;
    CHECK(!spv_store16.empty() && store16_out.size() == N / 2 && bad_store16 == 0,
          "#590: stride-2 Uint16 buffer_store_format_x lands disjoint fields race-free (dword k = 2k|(2k+1)<<16)");

    // Kernel 24b: the SAME unorm8x4 fetch at a NON-dword-aligned inst offset (offset:2). The runtime
    // normalized-component path must begin at byte 2 of each record and join the following dword for
    // the later components. Reading v1 back proves the first component is B rather than the old R.
    const uint32_t code24b[] = { 0x7e000f00u, 0xe00c2002u, 0x80020100u, 0xbf810000u };
    std::vector<uint32_t> spv24b = recompile_valu(
        code24b, std::size(code24b), 1, /*out_vgpr*/1, &rt24);
    std::vector<uint32_t> vbuf24b = vbuf24;
    vbuf24b.push_back(0u); // final XYZW fetch straddles one dword past the last record
    const std::vector<float> got24b = prosper::test::run_compute(
        spv24b, in24, N, N, {}, vbuf24b);
    uint32_t bad24b = 0;
    for (uint32_t i = 0; i < N && got24b.size() == N; ++i) {
        const float blue = static_cast<float>((vbuf24[i] >> 16) & 0xffu) / 255.0f;
        if (std::fabs(got24b[i] - blue) > 1e-5f) ++bad24b;
    }
    CHECK(!spv24b.empty() && got24b.size() == N && bad24b == 0,
          "kernel 24b extracts normalized components from a non-dword-aligned runtime address");

    // Kernel 25: SNORM16x2 vertex fetch — signed 16-bit fields, normalized /32767 and clamped to -1.0
    // (the SNORM rule: -32768 maps to -1.0, not -1.00003). out = v1 + 10*v2. y is fixed at -32768 to
    // exercise the clamp; x varies per lane to also prove correct sign-extension of the low field.
    const uint32_t code25[] = {
        0x7e000f00u, 0xe0042000u, 0x80020100u, 0x7e0a02ffu, 0x41200000u, 0x100c0505u, 0x06020d01u, 0xbf810000u,
    };
    ShaderResourceTable rt25;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Snorm16;
      vb.num_components = 2; vb.binding = 3; vb.stride = 4; vb.sgpr_base = 8; rt25.resources.push_back(vb); }
    std::vector<uint32_t> spv25 = recompile_valu(code25, sizeof(code25)/sizeof(code25[0]), 1, /*out_vgpr*/1, &rt25);
    CHECK(!spv25.empty(), "recompiled kernel 25 (buffer_load_format_xy snorm16x2) -> SPIR-V");
    std::vector<float> in25(N), exp25(N); std::vector<uint32_t> vbuf25(N);
    auto snorm16 = [](int16_t v){ return std::fmax((float)v / 32767.0f, -1.0f); };
    for (uint32_t i = 0; i < N; i++) {
        in25[i] = (float)i;
        int16_t xs = (int16_t)((int)i*100 - 3000), ys = (int16_t)-32768;
        vbuf25[i] = (uint32_t)(uint16_t)xs | ((uint32_t)(uint16_t)ys << 16);
        exp25[i] = snorm16(xs) + 10.0f*snorm16(ys);
    }
    std::vector<float> got25 = prosper::test::run_compute(spv25, in25, N, N, /*cbuf0*/{}, /*cbuf1=vertex buf*/vbuf25);
    uint32_t bad25 = 0; for (uint32_t i=0;i<N&&got25.size()==N;i++) if (std::fabs(got25[i]-exp25[i])>2e-3f) bad25++;
    printf("  kernel25 mismatches=%u (out[40]=%g expect=%g)\n", bad25, got25.size()==N?got25[40]:-1, got25.size()==N?exp25[40]:-1);
    CHECK(got25.size()==N && bad25==0, "recompiled kernel 25 (snorm16x2 -> normalized + clamped floats) correct");

    // #370: packed-word vertex formats. All four logical components occupy one dword, so the ordinary
    // uniform-byte-component path cannot represent them. Exercise the actual translated SPIR-V on Vulkan
    // with distinct weighted components: this catches field order/width, sign extension, normalization,
    // the two-bit alpha rule, and the R11G11B10 unsigned-small-float widening.
    const uint32_t code25packed_float[] = {
        0x7e000f00u, 0xe00c2000u, 0x80020100u, // index + buffer_load_format_xyzw v[1:4]
        0x7e0a02f4u, 0x100c0505u, 0x06020d01u,
        0x7e0a02f6u, 0x100c0905u, 0x06020d01u,
        0x7e0a02ffu, 0x40400000u, 0x100c0705u, 0x06020d01u, 0xbf810000u,
    };
    const uint32_t code25packed_uint[] = {
        0x7e000f00u, 0xe00c2000u, 0x80020100u,
        0x7e020d01u, 0x7e040d02u, 0x7e060d03u, 0x7e080d04u, // uint fields -> float
        0x7e0a02f4u, 0x100c0505u, 0x06020d01u,
        0x7e0a02f6u, 0x100c0905u, 0x06020d01u,
        0x7e0a02ffu, 0x40400000u, 0x100c0705u, 0x06020d01u, 0xbf810000u,
    };
    const uint32_t code25packed_sint[] = {
        0x7e000f00u, 0xe00c2000u, 0x80020100u,
        0x7e020b01u, 0x7e040b02u, 0x7e060b03u, 0x7e080b04u, // signed fields -> float
        0x7e0a02f4u, 0x100c0505u, 0x06020d01u,
        0x7e0a02f6u, 0x100c0905u, 0x06020d01u,
        0x7e0a02ffu, 0x40400000u, 0x100c0705u, 0x06020d01u, 0xbf810000u,
    };
    auto check_packed_fetch = [&](const uint32_t* code, size_t code_dwords, DataFormat format,
                                  const std::vector<uint32_t>& words,
                                  const std::vector<float>& expected, const char* message) {
        ShaderResourceTable table;
        ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = format;
        vb.num_components = format == DataFormat::Float10_11_11 ? 3 : 4;
        vb.binding = 3; vb.stride = 4; vb.sgpr_base = 8; table.resources.push_back(vb);
        std::vector<uint32_t> spv = recompile_valu(code, code_dwords, 1, 1, &table);
        std::vector<float> input(N); for (uint32_t i = 0; i < N; ++i) input[i] = (float)i;
        std::vector<float> got = prosper::test::run_compute(spv, input, N, N, {}, words);
        uint32_t bad = 0;
        for (uint32_t i = 0; i < N && got.size() == N; ++i)
            if (std::fabs(got[i] - expected[i]) > 3e-3f) ++bad;
        CHECK(!spv.empty() && got.size() == N && bad == 0, message);
    };
    auto pack_2_10_10_10 = [](int32_t r, int32_t g, int32_t b, int32_t a) {
        return ((uint32_t)r & 0x3ffu) | (((uint32_t)g & 0x3ffu) << 10) |
               (((uint32_t)b & 0x3ffu) << 20) | (((uint32_t)a & 0x3u) << 30);
    };
    std::vector<uint32_t> packed_words(N);
    std::vector<float> packed_expected(N);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t r = i * 17u & 1023u, g = i * 29u + 3u & 1023u;
        uint32_t b = i * 43u + 7u & 1023u, a = i & 3u;
        packed_words[i] = pack_2_10_10_10(r, g, b, a);
        packed_expected[i] = r / 1023.0f + 2.0f * g / 1023.0f +
                             3.0f * b / 1023.0f + 4.0f * a / 3.0f;
    }
    check_packed_fetch(code25packed_float, sizeof(code25packed_float) / 4,
                       DataFormat::Unorm2_10_10_10, packed_words, packed_expected,
                       "packed 2_10_10_10 UNORM fetch decodes four normalized fields");

    auto snorm = [](int32_t v, float scale) { return std::fmax((float)v / scale, -1.0f); };
    for (uint32_t i = 0; i < N; ++i) {
        int32_t r = (int32_t)(i * 17u & 1023u) - 512;
        int32_t g = (int32_t)(i * 29u + 3u & 1023u) - 512;
        int32_t b = (int32_t)(i * 43u + 7u & 1023u) - 512;
        int32_t a = (int32_t)(i & 3u) - 2;
        packed_words[i] = pack_2_10_10_10(r, g, b, a);
        packed_expected[i] = snorm(r, 511.0f) + 2.0f * snorm(g, 511.0f) +
                             3.0f * snorm(b, 511.0f) + 4.0f * snorm(a, 1.0f);
    }
    check_packed_fetch(code25packed_float, sizeof(code25packed_float) / 4,
                       DataFormat::Snorm2_10_10_10, packed_words, packed_expected,
                       "packed 2_10_10_10 SNORM fetch sign-extends and clamps every field");

    for (uint32_t i = 0; i < N; ++i) {
        uint32_t r = i * 17u & 1023u, g = i * 29u + 3u & 1023u;
        uint32_t b = i * 43u + 7u & 1023u, a = i & 3u;
        packed_words[i] = pack_2_10_10_10(r, g, b, a);
        packed_expected[i] = (float)r + 2.0f * (float)g + 3.0f * (float)b + 4.0f * (float)a;
    }
    check_packed_fetch(code25packed_uint, sizeof(code25packed_uint) / 4,
                       DataFormat::Uint2_10_10_10, packed_words, packed_expected,
                       "packed 2_10_10_10 UINT fetch zero-extends four integer fields");

    for (uint32_t i = 0; i < N; ++i) {
        int32_t r = (int32_t)(i * 17u & 1023u) - 512;
        int32_t g = (int32_t)(i * 29u + 3u & 1023u) - 512;
        int32_t b = (int32_t)(i * 43u + 7u & 1023u) - 512;
        int32_t a = (int32_t)(i & 3u) - 2;
        packed_words[i] = pack_2_10_10_10(r, g, b, a);
        packed_expected[i] = (float)r + 2.0f * (float)g + 3.0f * (float)b + 4.0f * (float)a;
    }
    check_packed_fetch(code25packed_sint, sizeof(code25packed_sint) / 4,
                       DataFormat::Sint2_10_10_10, packed_words, packed_expected,
                       "packed 2_10_10_10 SINT fetch sign-extends four integer fields");

    for (uint32_t i = 0; i < N; ++i) {
        uint32_t r = (15u << 6) | (i & 63u);       // 11-bit unsigned float, [1,2)
        uint32_t g = (14u << 6) | (i * 3u & 63u);  // 11-bit unsigned float, [0.5,1)
        uint32_t b = (16u << 5) | (i * 5u & 31u);  // 10-bit unsigned float, [2,4)
        packed_words[i] = r | (g << 11) | (b << 22);
        packed_expected[i] = f11_to_float((uint16_t)r) + 2.0f * f11_to_float((uint16_t)g) +
                             3.0f * f10_to_float((uint16_t)b) + 4.0f; // absent W defaults to 1.0
    }
    check_packed_fetch(code25packed_float, sizeof(code25packed_float) / 4,
                       DataFormat::Float10_11_11, packed_words, packed_expected,
                       "packed 10_11_11 FLOAT fetch widens mini-floats and default-fills W");

    // #370 review: exercise the canonical descriptor contract, not just an injected DataFormat.
    // R11G11B10_FLOAT uses DST_SEL(X,Y,Z,1) = 0x3AC; the resulting W must be exactly 1.0.
    const uint32_t packed_float_vsharp[4] = {
        0x00020000u, 4u << 16, N, (36u << 12) | 0x3ACu,
    };
    const DecodedBufferDescriptor packed_float_desc = decode_buffer_descriptor(packed_float_vsharp);
    ShaderResourceTable packed_float_table;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer;
      vb.format = packed_float_desc.format; vb.num_components = packed_float_desc.num_components;
      vb.binding = 3; vb.stride = packed_float_desc.stride; vb.sgpr_base = 8;
      packed_float_table.resources.push_back(vb); }
    std::vector<uint32_t> packed_float_spv = recompile_valu(
        code25packed_float, sizeof(code25packed_float) / 4, 1, 1, &packed_float_table);
    std::vector<float> packed_float_input(N);
    for (uint32_t i = 0; i < N; ++i) packed_float_input[i] = (float)i;
    std::vector<float> packed_float_got = prosper::test::run_compute(
        packed_float_spv, packed_float_input, N, N, {}, packed_words);
    uint32_t packed_float_bad = 0;
    for (uint32_t i = 0; i < N && packed_float_got.size() == N; ++i)
        if (std::fabs(packed_float_got[i] - packed_expected[i]) > 3e-3f) ++packed_float_bad;
    CHECK(packed_float_desc.format == DataFormat::Float10_11_11 &&
          packed_float_desc.num_components == 3 && !packed_float_desc.forbid_unknown_fallback &&
          !packed_float_spv.empty() && packed_float_got.size() == N && packed_float_bad == 0,
          "canonical format-36 X/Y/Z/1 descriptor executes R/G/B widening with W=1");

    // Kernel 26: s_buffer_load_dwordx8 (wide scalar load). s[0:7] = cbuf[4..11] (offset 0x10>>2=4);
    // out = (float)s7 = cbuf[11]. Real shaders load whole constant blocks in one x8/x16 op; this proves
    // the wide loads emit all N dwords, not just x1/x2/x4.
    const uint32_t code26[] = { 0xf42c0004u, 0xfa000010u, 0xbf8cc07fu, 0x7e000207u, 0xbf810000u };
    std::vector<uint32_t> spv26 = recompile_valu(code26, sizeof(code26)/sizeof(code26[0]), 1, /*out_vgpr*/0);
    CHECK(!spv26.empty(), "recompiled kernel 26 (s_buffer_load_dwordx8) -> SPIR-V");
    std::vector<float> in26(N, 0.0f); std::vector<uint32_t> cbuf26(16, 0u);
    { float v = 42.5f; std::memcpy(&cbuf26[11], &v, 4); }
    std::vector<float> got26 = prosper::test::run_compute(spv26, in26, N, N, cbuf26);
    uint32_t bad26 = 0; for (uint32_t i=0;i<N&&got26.size()==N;i++) if (std::fabs(got26[i]-42.5f)>1e-3f) bad26++;
    printf("  kernel26 mismatches=%u (out[0]=%g expect=42.5)\n", bad26, got26.size()==N?got26[0]:-1);
    CHECK(got26.size()==N && bad26==0, "recompiled kernel 26 (s_buffer_load_dwordx8 loads all 8 dwords) correct");

    // Kernel 27: v_xor3_b32 (VOP3 three-way XOR) = s0 ^ s1 ^ s2. Common in hashing / address munging.
    // Convert 3 uint inputs, xor3, convert back. Verifies the VOP3-with-literal-aware decode path too.
    const uint32_t code27[] = {
        0x7e000f00u, 0x7e020f01u, 0x7e040f02u, 0xd5780000u, 0x040a0300u, 0x7e000d00u, 0xbf810000u,
    };
    std::vector<uint32_t> spv27 = recompile_valu(code27, sizeof(code27)/sizeof(code27[0]), 3, 0);
    CHECK(!spv27.empty(), "recompiled kernel 27 (v_xor3_b32) -> SPIR-V");
    std::vector<float> in27(N * 3), exp27(N);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t a = i, b = (i * 7) & 0xFFu, c = (i * 131) & 0x1FFu;
        in27[i*3+0]=(float)a; in27[i*3+1]=(float)b; in27[i*3+2]=(float)c;
        exp27[i] = (float)(a ^ b ^ c);
    }
    std::vector<float> got27 = prosper::test::run_compute(spv27, in27, N, N);
    uint32_t bad27 = 0; for (uint32_t i=0;i<N&&got27.size()==N;i++) if (std::fabs(got27[i]-exp27[i])>1e-3f) bad27++;
    printf("  kernel27 mismatches=%u (out[9]=%g expect=%g)\n", bad27, got27.size()==N?got27[9]:-1, exp27[9]);
    CHECK(got27.size()==N && bad27==0, "recompiled kernel 27 (v_xor3_b32 three-way XOR) correct");

    // Kernel 28: SCC via s_cmp + s_cselect. s_cmp_eq_u32 sets SCC; s_cselect_b32 s2 = SCC ? 42 : 7;
    // out = float(s2). Two variants prove both SCC paths (eq true -> 42, eq false -> 7).
    const uint32_t code28t[] = { 0xbe800387u, 0xbe810387u, 0xbf060100u, 0x850287aau, 0x7e000c02u, 0xbf810000u };
    const uint32_t code28f[] = { 0xbe800387u, 0xbe810388u, 0xbf060100u, 0x850287aau, 0x7e000c02u, 0xbf810000u };
    std::vector<uint32_t> spv28t = recompile_valu(code28t, sizeof(code28t)/sizeof(code28t[0]), 1, 0);
    std::vector<uint32_t> spv28f = recompile_valu(code28f, sizeof(code28f)/sizeof(code28f[0]), 1, 0);
    CHECK(!spv28t.empty() && !spv28f.empty(), "recompiled kernel 28 (s_cmp + s_cselect, both SCC paths) -> SPIR-V");
    std::vector<float> in28(N, 0.0f);
    std::vector<float> got28t = prosper::test::run_compute(spv28t, in28, N, N);
    std::vector<float> got28f = prosper::test::run_compute(spv28f, in28, N, N);
    uint32_t bad28 = 0;
    for (uint32_t i=0;i<N&&got28t.size()==N;i++) if (std::fabs(got28t[i]-42.0f)>1e-3f) bad28++;
    for (uint32_t i=0;i<N&&got28f.size()==N;i++) if (std::fabs(got28f[i]- 7.0f)>1e-3f) bad28++;
    printf("  kernel28 mismatches=%u (true=%g expect 42, false=%g expect 7)\n",
           bad28, got28t.size()==N?got28t[0]:-1, got28f.size()==N?got28f[0]:-1);
    CHECK(got28t.size()==N && got28f.size()==N && bad28==0, "recompiled kernel 28 (SCC compare -> cselect) correct");

    // Kernel 29: MUBUF STORE. v2 = (uint)gid; v3 = 2*float(gid); buffer_store_format_x v3, v2, s[8:11]
    // idxen -> binding-3 buffer[gid] = 2*gid. Reads back the stored buffer (cbuf1_out) and checks it.
    const uint32_t code29[] = { 0x7e040f00u, 0x06060100u, 0xe0102000u, 0x80020302u, 0xbf810000u };
    ShaderResourceTable rt29;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32;
      vb.num_components = 1; vb.binding = 3; vb.stride = 4; vb.sgpr_base = 8; rt29.resources.push_back(vb); }
    std::vector<uint32_t> spv29 = recompile_valu(code29, sizeof(code29)/sizeof(code29[0]), 1, 0, &rt29);
    CHECK(!spv29.empty(), "recompiled kernel 29 (buffer_store_format_x) -> SPIR-V");
    std::vector<float> in29(N); for (uint32_t i = 0; i < N; i++) in29[i] = (float)i;
    std::vector<uint32_t> stored(N, 0), stored_out;
    prosper::test::run_compute(spv29, in29, N, N, /*cbuf0*/{}, /*cbuf1=store target*/stored, &stored_out);
    uint32_t bad29 = 0;
    for (uint32_t i = 0; i < N && stored_out.size() == N; i++) {
        float f; std::memcpy(&f, &stored_out[i], 4);
        if (std::fabs(f - 2.0f*(float)i) > 1e-3f) bad29++;
    }
    { float f5 = 0; if (stored_out.size()==N) std::memcpy(&f5, &stored_out[5], 4);
      printf("  kernel29 mismatches=%u (buf[5]=%g expect=10)\n", bad29, f5); }
    CHECK(stored_out.size() == N && bad29 == 0, "recompiled kernel 29 (MUBUF store writes buffer[gid]=2*gid) correct");

    // Astro Bot world-map PS exact final MUBUF packet. Every lane writes three consecutive raw
    // dwords to one 12-byte record, proving both the unusual x3 component count and IDXEN stride.
    const uint32_t code29x3[] = {
        0x7e140f00u,                         // v10 = (uint)v0 = invocation index
        0x7e060281u, 0x7e080282u, 0x7e0a0283u, // v3/v4/v5 = 1/2/3
        0xe07c2000u, 0x8004030au,           // exact Astro buffer_store_dwordx3 packet
        0xbf810000u,
    };
    ShaderResourceTable rt29x3;
    { ShaderResource dst{}; dst.cls = ResourceClass::ConstantBuffer;
      dst.format = DataFormat::Uint32; dst.num_components = 1;
      dst.binding = 3; dst.stride = 12; dst.sgpr_base = 16;
      rt29x3.resources.push_back(dst); }
    std::vector<uint32_t> spv29x3 = recompile_valu(
        code29x3, std::size(code29x3), 1, 0, &rt29x3);
    CHECK(!spv29x3.empty(),
          "recompiled exact Astro buffer_store_dwordx3 packet -> SPIR-V");
    std::vector<uint32_t> stored29x3(N * 3, 0u), stored29x3_out;
    prosper::test::run_compute(spv29x3, in29, N, N, {}, stored29x3, &stored29x3_out);
    uint32_t bad29x3 = 0;
    for (uint32_t lane = 0; lane < N && stored29x3_out.size() == N * 3; ++lane)
        for (uint32_t component = 0; component < 3; ++component)
            bad29x3 += stored29x3_out[lane * 3 + component] != component + 1;
    CHECK(stored29x3_out.size() == N * 3 && bad29x3 == 0,
          "buffer_store_dwordx3 writes all three dwords of every indexed record");

    const uint32_t code29mt[] = {
        0x7e040f00u, 0x06060100u, 0xe8b42000u, 0x80020302u, 0xbf810000u,
    };
    ShaderResourceTable rt29mt = rt29;
    rt29mt.resources[0].format = DataFormat::Uint8; // MTBUF's encoded 32_FLOAT remains authoritative
    rt29mt.resources[0].fetch_pc = 2;
    std::vector<uint32_t> spv29mt = recompile_valu(
        code29mt, sizeof(code29mt)/sizeof(code29mt[0]), 1, 0, &rt29mt);
    std::vector<uint32_t> stored29mt(N, 0), stored29mt_out;
    prosper::test::run_compute(spv29mt, in29, N, N, {}, stored29mt, &stored29mt_out);
    uint32_t bad29mt = 0;
    for (uint32_t i = 0; i < N && stored29mt_out.size() == N; ++i) {
        float f = 0.0f; std::memcpy(&f, &stored29mt_out[i], sizeof(f));
        if (std::fabs(f - 2.0f*(float)i) > 1e-3f) ++bad29mt;
    }
    CHECK(!spv29mt.empty() && stored29mt_out.size() == N && bad29mt == 0,
          "MTBUF 32_FLOAT store writes instruction-typed dwords through descriptor provenance");

    // Kernel 30: EXEC-PREDICATED store. v_cmpx_lt_u32 narrows EXEC to lanes gid<4, then buffer_store_
    // format_x. Only active lanes must write (conditional store via a selection merge); lanes gid>=4
    // leave the buffer at its initial 0. Proves the predicated-store path (not just full-EXEC stores).
    const uint32_t code30[] = {
        0x7e040f00u, 0x06060100u, 0x7e0a0284u, 0x7da20b02u, 0xe0102000u, 0x80020302u, 0xbf810000u,
    };
    std::vector<uint32_t> spv30 = recompile_valu(code30, sizeof(code30)/sizeof(code30[0]), 1, 0, &rt29);
    CHECK(!spv30.empty(), "recompiled kernel 30 (v_cmpx + predicated MUBUF store) -> SPIR-V");
    std::vector<uint32_t> stored30(N, 0), stored30_out;
    prosper::test::run_compute(spv30, in29, N, N, /*cbuf0*/{}, /*cbuf1*/stored30, &stored30_out);
    uint32_t bad30 = 0;
    for (uint32_t i = 0; i < N && stored30_out.size() == N; i++) {
        float f; std::memcpy(&f, &stored30_out[i], 4);
        float expect = (i < 4) ? 2.0f*(float)i : 0.0f;
        if (std::fabs(f - expect) > 1e-3f) bad30++;
    }
    { float f2 = 0, f9 = 0; if (stored30_out.size()==N) { std::memcpy(&f2,&stored30_out[2],4); std::memcpy(&f9,&stored30_out[9],4); }
      printf("  kernel30 mismatches=%u (buf[2]=%g expect=4, buf[9]=%g expect=0)\n", bad30, f2, f9); }
    CHECK(stored30_out.size() == N && bad30 == 0, "recompiled kernel 30 (predicated store: only lanes gid<4 write) correct");

    // Kernel 31: the full compute grid-tail idiom — v_cmpx narrows EXEC, s_cbranch_execz skips straight
    // to s_endpgm (now linearized as a no-op for the guard-to-end shape), then a predicated store. Same
    // observable result as kernel 30 (gid<4 write 2*gid, else 0), proving the execz relaxation is safe.
    const uint32_t code31[] = {
        0x7e040f00u, 0x06060100u, 0x7e0a0284u, 0x7da20b02u, 0xbf880002u, 0xe0102000u, 0x80020302u, 0xbf810000u,
    };
    std::vector<uint32_t> spv31 = recompile_valu(code31, sizeof(code31)/sizeof(code31[0]), 1, 0, &rt29);
    CHECK(!spv31.empty(), "recompiled kernel 31 (v_cmpx + s_cbranch_execz-to-end + store) -> SPIR-V");
    std::vector<uint32_t> stored31(N, 0), stored31_out;
    prosper::test::run_compute(spv31, in29, N, N, /*cbuf0*/{}, /*cbuf1*/stored31, &stored31_out);
    uint32_t bad31 = 0;
    for (uint32_t i = 0; i < N && stored31_out.size() == N; i++) {
        float f; std::memcpy(&f, &stored31_out[i], 4);
        float expect = (i < 4) ? 2.0f*(float)i : 0.0f;
        if (std::fabs(f - expect) > 1e-3f) bad31++;
    }
    printf("  kernel31 mismatches=%u (execz-to-end guard linearized + predicated store)\n", bad31);
    CHECK(stored31_out.size() == N && bad31 == 0, "recompiled kernel 31 (guard-to-endpgm execz + store) correct");

    // Kernel 32: LDS shared memory. In ONE workgroup of 64, lane i writes lds[i]=2i+1, s_barrier, then
    // reads lds[63-i] -> out = 2*(63-i)+1 = 127-2i. Proves ds_write/ds_read + the workgroup barrier +
    // cross-lane sharing. Uses exactly 64 invocations (1 workgroup) so LDS indices stay in range.
    const uint32_t code32[] = {
        0x7e020f00u, 0x34040282u, 0x34060281u, 0x4a060681u, 0xd8340000u, 0x00000302u, 0xbf8a0000u,
        0x4c0a02bfu, 0x340c0a82u, 0xd8d80000u, 0x07000006u, 0x7e000d07u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32 = recompile_valu(code32, sizeof(code32)/sizeof(code32[0]), 1, 0);
    CHECK(!spv32.empty(), "recompiled kernel 32 (LDS ds_write/ds_read + s_barrier) -> SPIR-V");
    const uint32_t WG = 64;
    std::vector<float> in32(WG); for (uint32_t i = 0; i < WG; i++) in32[i] = (float)i;
    std::vector<float> got32 = prosper::test::run_compute(spv32, in32, WG, WG);
    uint32_t bad32 = 0; for (uint32_t i=0;i<WG&&got32.size()==WG;i++) if (std::fabs(got32[i]-(float)(127-2*(int)i))>1e-3f) bad32++;
    printf("  kernel32 mismatches=%u (out[0]=%g expect=127, out[63]=%g expect=1)\n",
           bad32, got32.size()==WG?got32[0]:-1, got32.size()==WG?got32[63]:-1);
    CHECK(got32.size()==WG && bad32==0, "recompiled kernel 32 (LDS write->barrier->cross-lane read) correct");

    // Kernel 32b: gfx10 wide LDS operations used by UE4's volume-lighting compute producer. write2_b64
    // writes v[1:2] at byte 0 and v[3:4] at byte 8; write_b64 repeats the first pair. read_b64 recovers
    // the first pair and read2_b64 recovers both; selecting the non-duplicate halves gives 1+2+3+4=10.
    const uint32_t code32b[] = {
        0x7e000280u, 0x7e0202f2u, 0x7e0402f4u, 0x7e0602ffu, 0x40400000u, 0x7e0802f6u,
        0xd9380100u, 0x00030100u, 0xd9340000u, 0x00000100u, 0xbf8a0000u,
        0xd9d80000u, 0x05000000u, 0xd9dc0100u, 0x07000000u,
        0x060a0d05u, 0x060a1305u, 0x060a1505u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32b = recompile_valu(code32b, sizeof(code32b)/sizeof(code32b[0]), 0, 5);
    CHECK(!spv32b.empty(), "recompiled kernel 32b (ds_write_b64 + ds_write2_b64) -> SPIR-V");
    std::vector<float> got32b = prosper::test::run_compute(spv32b, std::vector<float>(1), 1, 1);
    CHECK(got32b.size()==1 && std::fabs(got32b[0]-10.0f)<1e-3f,
          "kernel 32b wide LDS stores and loads preserve both dwords of both VGPR pairs");

    // Kernel 32b2: Astro Bot's loading-surface producer uses DS_READ2_B32 twice, first with
    // offset0/1=(0,1) and then (16,17). Populate those exact LDS locations with integers 1..4,
    // execute the two live instruction words, and make both return pairs observable as sum=10.
    const uint32_t code32b2[] = {
        0x7e000280u, 0x7e020281u, 0x7e040282u, 0x7e060283u, 0x7e080284u,
        0xd9340000u, 0x00000100u, 0xd9340040u, 0x00000300u, 0xbf8a0000u,
        0xd8dc0100u, 0x05000000u, 0xd8dc1110u, 0x07000000u,
        0x7e0a0d05u, 0x7e0c0d06u, 0x7e0e0d07u, 0x7e100d08u,
        0x060a0d05u, 0x060a0f05u, 0x060a1105u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32b2 = recompile_valu(code32b2, std::size(code32b2), 0, 5);
    CHECK(!spv32b2.empty(), "recompiled kernel 32b2 (Astro DS_READ2_B32 words) -> SPIR-V");
    std::vector<float> got32b2 = prosper::test::run_compute(spv32b2, std::vector<float>(1), 1, 1);
    CHECK(got32b2.size()==1 && std::fabs(got32b2[0]-10.0f)<1e-3f,
          "kernel 32b2 DS_READ2_B32 applies both packed dword offsets exactly");

    // Kernel 32b3: The Plucky Squire's title compute path writes two independent LDS dwords with
    // DS_WRITE2_B32 before reading the same pair. Use the exact adjacent-offset encoding and make
    // both values observable as 1+2=3.
    const uint32_t code32b_write2[] = {
        0x7e000280u, 0x7e0202f2u, 0x7e0402f4u,
        0xd8380100u, 0x00020100u, 0xbf8a0000u,
        0xd8dc0100u, 0x03000000u, 0x06060903u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32b_write2 = recompile_valu(
        code32b_write2, std::size(code32b_write2), 0, 3);
    CHECK(!spv32b_write2.empty(), "recompiled kernel 32b3 (DS_WRITE2_B32 + DS_READ2_B32) -> SPIR-V");
    std::vector<float> got32b_write2 = prosper::test::run_compute(
        spv32b_write2, std::vector<float>(1), 1, 1);
    CHECK(got32b_write2.size()==1 && std::fabs(got32b_write2[0]-3.0f)<1e-3f,
          "kernel 32b3 DS_WRITE2_B32 stores DATA0/1 at both packed dword offsets");

    // Kernel 32b3a: Plucky's first-gameplay compute program uses ADDTID as a per-wave LDS spill.
    // Exercise the exact live write/read encodings with M0=0x40 and immediate=0x100. Each lane
    // writes a distinct input to LDS[0x140 + TID*4] and reads its own value back. This catches both
    // accidentally ignoring TID (all lanes alias) and using DATA0/VDST from the wrong DS byte.
    const uint32_t code32b_addtid[] = {
        0xb07c0040u,                         // s_movk_i32 m0, 0x40
        0xdac00100u, 0x00000000u,           // ds_write_addtid_b32 v0 offset:0x100
        0xbf8cc07fu,                         // s_waitcnt lgkmcnt(0)
        0xdac40100u, 0x01000000u,           // ds_read_addtid_b32 v1 offset:0x100
        0xbf8cc07fu,
        0xbf810000u,
    };
    std::vector<uint32_t> spv32b_addtid = recompile_valu(
        code32b_addtid, std::size(code32b_addtid), 1, 1);
    CHECK(!spv32b_addtid.empty(),
          "recompiled kernel 32b3a (Plucky DS_WRITE/READ_ADDTID_B32) -> SPIR-V");
    std::vector<float> in32b_addtid(WG);
    for (uint32_t lane = 0; lane < WG; ++lane)
        in32b_addtid[lane] = 1000.0f + static_cast<float>(lane) * 3.0f;
    std::vector<float> got32b_addtid = prosper::test::run_compute(
        spv32b_addtid, in32b_addtid, WG, WG);
    CHECK(got32b_addtid == in32b_addtid,
          "kernel 32b3a ADDTID addresses one distinct LDS dword per hardware-wave lane");

    // Kernel 32b2w: Astro Bot's world-map reduction pairs that read with DS_WRITE2_B32. Use its
    // exact offset0/1=(0,1), DATA0=v2, DATA1=v1 instruction, then expose OFFSET0 alone. A
    // position-sensitive result proves DATA0/DATA1 are not swapped or treated as one VGPR pair.
    const uint32_t code32b2w[] = {
        0x7e000280u, 0x7e020281u, 0x7e040282u,
        0xd8380100u, 0x00010200u, 0xbf8a0000u,
        0xd8dc0100u, 0x03000000u,
        0x7e0a0d03u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32b2w = recompile_valu(
        code32b2w, std::size(code32b2w), 0, 5);
    CHECK(!spv32b2w.empty(), "recompiled kernel 32b2w (Astro DS_WRITE2_B32 word) -> SPIR-V");
    std::vector<float> got32b2w = prosper::test::run_compute(
        spv32b2w, std::vector<float>(1), 1, 1);
    CHECK(got32b2w.size()==1 && std::fabs(got32b2w[0]-2.0f)<1e-3f,
          "kernel 32b2w DS_WRITE2_B32 keeps DATA0 at packed OFFSET0");

    // Equal offsets are one access, not two ordered writes: DATA0 wins and DATA1 is ignored.
    const uint32_t code32b2weq[] = {
        0x7e000280u, 0x7e020281u, 0x7e040282u,
        0xd8380000u, 0x00010200u, 0xbf8a0000u,
        0xd8d80000u, 0x03000000u,
        0x7e0a0d03u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32b2weq = recompile_valu(
        code32b2weq, std::size(code32b2weq), 0, 5);
    CHECK(!spv32b2weq.empty(),
          "recompiled kernel 32b2weq (DS_WRITE2_B32 equal offsets) -> SPIR-V");
    std::vector<float> got32b2weq = prosper::test::run_compute(
        spv32b2weq, std::vector<float>(1), 1, 1);
    CHECK(got32b2weq.size()==1 && std::fabs(got32b2weq[0]-2.0f)<1e-3f,
          "kernel 32b2weq equal offsets perform one DATA0 access");

    // Kernel 32b2a: Astro Bot's large world-map compute program spills each wave lane through
    // DS_WRITE_ADDTID_B32 and reloads it with DS_READ_ADDTID_B32. Unique per-lane input values prove
    // the translation uses lane*4 rather than racing every invocation through one LDS dword.
    const uint32_t code32b2a[] = {
        0xb07c0000u,                // s_movk_i32 m0, 0
        0xdac00700u, 0x00000000u,   // ds_write_addtid_b32 v0 offset:0x700
        0xbf8a0000u,                // s_barrier
        0xdac40700u, 0x01000000u,   // ds_read_addtid_b32 v1 offset:0x700
        0xbf810000u,
    };
    std::vector<uint32_t> spv32b2a = recompile_valu(
        code32b2a, std::size(code32b2a), 1, 1);
    CHECK(!spv32b2a.empty(),
          "recompiled kernel 32b2a (Astro DS_WRITE/READ_ADDTID_B32 words) -> SPIR-V");
    std::vector<float> in32b2a(64);
    for (uint32_t i = 0; i < in32b2a.size(); ++i) in32b2a[i] = 1000.0f + (float)i;
    std::vector<float> got32b2a = prosper::test::run_compute(
        spv32b2a, in32b2a, (uint32_t)in32b2a.size(), (uint32_t)in32b2a.size());
    CHECK(got32b2a == in32b2a,
          "kernel 32b2a ADDTID round-trips every lane through its own LDS dword");

    // ADDTID addresses use only M0[15:0]. Cross-check the write against an ordinary DS read so the
    // test cannot pass merely because ADDTID write and read share the same incorrect formula.
    const uint32_t code32b2amw[] = {
        0xbefc03ffu, 0x00010000u,   // s_mov_b32 m0, 0x10000 (low 16 bits are zero)
        0xdac00700u, 0x00000000u,   // ds_write_addtid_b32 v0 offset:0x700
        0xbf8a0000u,
        0x7e0402ffu, 0x00000700u,   // v_mov_b32 v2, 0x700
        0xd8d80000u, 0x01000002u,   // ds_read_b32 v1, v2
        0xbf810000u,
    };
    std::vector<uint32_t> spv32b2amw = recompile_valu(
        code32b2amw, std::size(code32b2amw), 1, 1);
    CHECK(!spv32b2amw.empty(), "recompiled kernel 32b2amw (ADDTID write M0 mask) -> SPIR-V");
    std::vector<float> got32b2amw = prosper::test::run_compute(
        spv32b2amw, std::vector<float>{1234.0f}, 1, 1);
    CHECK(got32b2amw == std::vector<float>{1234.0f},
          "kernel 32b2amw masks M0 to 16 bits before ADDTID write");

    // Kernel 32b3: three other exact words from Astro's loading compute shaders. V_MUL_U32_U24
    // must mask both operands to 24 bits, while V_CVT_FLR_I32_F32 must round negative values toward
    // -infinity (not toward zero). Add the converted results: (64 * 3) + floor(-2.25) = 189.
    const uint32_t code32b3[] = {
        0x7e000f00u,               // v_cvt_u32_f32 v0, v0
        0x160600c0u,               // live: v_mul_u32_u24 v3, 64, v0
        0x7e060d03u,               // v_cvt_f32_u32 v3, v3
        0x7e681b01u,               // live: v_cvt_flr_i32_f32 v52, v1
        0x7e680b34u,               // v_cvt_f32_i32 v52, v52
        0x06686903u,               // v_add_f32 v52, v3, v52
        0xbf810000u,
    };
    std::vector<uint32_t> spv32b3 = recompile_valu(code32b3, std::size(code32b3), 2, 52);
    CHECK(!spv32b3.empty(), "recompiled kernel 32b3 (Astro U24 multiply + floor-convert words) -> SPIR-V");
    std::vector<float> got32b3 = prosper::test::run_compute(spv32b3, {3.0f, -2.25f}, 1, 1);
    CHECK(got32b3.size()==1 && std::fabs(got32b3[0]-189.0f)<1e-3f,
          "kernel 32b3 masks U24 multiplication and floors negative f32 before i32 conversion");

    // Kernel 32b4: exact DS_WRITE_B96/B128 and DS_READ_B96/B128 forms from Astro. Store 1,2,3 at
    // byte 0x510 and 4,5,6,7 at byte 0, read both vectors back, and sum all seven dwords to 28.
    const uint32_t code32b4[] = {
        0x7e000284u, 0x7e020285u, 0x7e040286u, 0x7e060287u,
        0x7e080281u, 0x7e0a0282u, 0x7e0c0283u, 0x7e0e0280u,
        0xdb780510u, 0x00000407u,   // live: ds_write_b96  v7, v[4:6] offset:0x510
        0xdb7c0000u, 0x00000007u,   // live: ds_write_b128 v7, v[0:3]
        0xbf8a0000u,
        0xdbfc0000u, 0x08000007u,   // live form: ds_read_b128 v[8:11], v7
        0xdbf80510u, 0x0c000007u,   // live form: ds_read_b96 v[12:14], v7 offset:0x510
        0x7e100d08u, 0x7e120d09u, 0x7e140d0au, 0x7e160d0bu,
        0x7e180d0cu, 0x7e1a0d0du, 0x7e1c0d0eu,
        0x06101308u, 0x06101508u, 0x06101708u,
        0x06101908u, 0x06101b08u, 0x06101d08u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32b4 = recompile_valu(code32b4, std::size(code32b4), 0, 8);
    CHECK(!spv32b4.empty(), "recompiled kernel 32b4 (Astro wide LDS read/write words) -> SPIR-V");
    std::vector<float> got32b4 = prosper::test::run_compute(spv32b4, std::vector<float>(1), 1, 1);
    CHECK(got32b4.size()==1 && std::fabs(got32b4[0]-28.0f)<1e-3f,
          "kernel 32b4 wide LDS reads/writes preserve all three/four consecutive dwords");

    // Terminator 2D's startup clear writes a device-global GDS dword. Compute GDS must use the
    // backend-owned persistent buffer, not the per-workgroup LDS array.
    const uint32_t compute_gds_write[] = {
        0xd8360000u, 0x00000100u,  // ds_write_b32 v0, v1 gds
        0xbf810000u,
    };
    ShaderResourceTable compute_gds_table;
    ShaderResource compute_gds_resource;
    compute_gds_resource.cls = ResourceClass::ConstantBuffer;
    compute_gds_resource.format = DataFormat::Uint32;
    compute_gds_resource.num_components = 1;
    compute_gds_resource.binding = kComputeInternalGdsBinding;
    compute_gds_resource.size = 64 * 1024;
    compute_gds_resource.stride = 4;
    std::array<uint8_t, 64 * 1024> compute_gds_backing{};
    compute_gds_resource.host_data = compute_gds_backing.data();
    compute_gds_resource.host_data_size = compute_gds_backing.size();
    compute_gds_table.resources.push_back(compute_gds_resource);
    ComputeShaderConfig compute_gds_config;
    std::vector<uint32_t> compute_gds_spv = recompile_compute(
        compute_gds_write, std::size(compute_gds_write), &compute_gds_table,
        compute_gds_config);
    CHECK(!compute_gds_spv.empty(), "compute GDS ds_write_b32 recompiles to SPIR-V");
    CHECK(validate_spirv_descriptor_interface(
              compute_gds_spv, &compute_gds_table, 0, SpirvShaderStage::Compute, false).ok(),
          "compute GDS module binds the persistent backend-owned buffer");

    // Astro Bot's indirect-argument producer uses device-global append counters. Routing this exact
    // GDS form through workgroup LDS made every dispatch begin with undefined counter values and
    // produced random indirect counts. The descriptor contract proves the append targets the
    // persistent backend-owned GDS buffer rather than a Workgroup array.
    const uint32_t compute_gds_append[] = {
        0xbefc0380u,              // s_mov_b32 m0, 0
        0xd8fa0010u, 0x00000000u, // ds_append v0 offset:16 gds
        0xbf810000u,
    };
    std::vector<uint32_t> compute_gds_append_spv = recompile_compute(
        compute_gds_append, std::size(compute_gds_append), &compute_gds_table,
        compute_gds_config);
    CHECK(!compute_gds_append_spv.empty(),
          "compute GDS ds_append recompiles to SPIR-V");
    CHECK(validate_spirv_descriptor_interface(
              compute_gds_append_spv, &compute_gds_table, 0,
              SpirvShaderStage::Compute, false).ok(),
          "compute GDS ds_append binds the persistent backend-owned buffer");

    // Kernel 32b5: ordinary LDS DS_APPEND. Initialize the LDS counter to 10, narrow EXEC to
    // the first 32 lanes, then append. Hardware performs one atomic +32 and broadcasts old=10 only
    // to active lanes; inactive lanes preserve their old v8=99. Reading the new counter (42) makes
    // both effects observable: active output=52, inactive output=141.
    const uint32_t code32b5[] = {
        0x7e000f00u,               // v_cvt_u32_f32 v0, v0 (lane number input)
        0x7e04028au, 0x7e060280u,  // v2=10, v3=byte address 0
        0x7e1002ffu, 0x00000063u,  // v8=99 fallback for inactive lanes
        0xbefc0380u,               // s_mov_b32 m0, 0
        0x7da40080u,               // v_cmpx_eq_u32 0, v0 (lane zero initializes)
        0xd8340010u, 0x00000203u,  // ds_write_b32 v3, v2 offset:0x10
        0xbefe04c1u, 0xbf8a0000u,  // restore EXEC; barrier
        0x7e0202a0u, 0x7da20300u,  // v1=32; v_cmpx_lt_u32 v0,v1
        0xd8f80010u, 0x08000000u,  // ds_append v8 offset:0x10 (LDS)
        0xbefe04c1u, 0xbf8a0000u,  // restore EXEC; barrier
        0xd8d80010u, 0x09000003u,  // ds_read_b32 v9, v3 offset:0x10
        0x7e100d08u, 0x7e120d09u, 0x06101308u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32b5 = recompile_valu(code32b5, std::size(code32b5), 1, 8);
    CHECK(!spv32b5.empty(), "recompiled kernel 32b5 (LDS DS_APPEND word) -> SPIR-V");
    std::vector<float> in32b5(WG); for (uint32_t i = 0; i < WG; ++i) in32b5[i] = (float)i;
    std::vector<float> got32b5 = prosper::test::run_compute(spv32b5, in32b5, WG, WG);
    uint32_t bad32b5 = 0;
    for (uint32_t i = 0; i < WG && got32b5.size() == WG; ++i) {
        const float expected = i < 32 ? 52.0f : 141.0f;
        if (std::fabs(got32b5[i] - expected) > 1e-3f) ++bad32b5;
    }
    printf("  kernel32b5 output count=%zu", got32b5.size());
    if (got32b5.size() == WG)
        printf("  kernel32b5 mismatches=%u out[0/31/32/63]=%g/%g/%g/%g expect=52/52/141/141\n",
               bad32b5, got32b5[0], got32b5[31], got32b5[32], got32b5[63]);
    else
        printf("\n");
    CHECK(got32b5.size()==WG && bad32b5==0,
          "kernel 32b5 atomically adds popcount(EXEC), broadcasts old counter, and predicates VDST");

    // Kernel 32b6: ordinary LDS DS_CONSUME is the inverse operation. Start at 50, make the
    // first 16 lanes valid, and consume once. Active lanes observe old=50 and all lanes read the new
    // counter=34, while inactive lanes preserve their fallback destination value of 99.
    const uint32_t code32b6[] = {
        0x7e000f00u,                         // v_cvt_u32_f32 v0, v0 (lane number input)
        0x7e0402ffu, 0x00000032u,            // v2=50
        0x7e060280u,                         // v3=byte address 0
        0x7e1802ffu, 0x00000063u,            // v12=99 fallback for inactive lanes
        0xbefc0380u,                         // s_mov_b32 m0, 0
        0x7da40080u,                         // v_cmpx_eq_u32 0, v0 (lane zero initializes)
        0xd8340004u, 0x00000203u,            // ds_write_b32 v3, v2 offset:4
        0xbefe04c1u, 0xbf8a0000u,            // restore EXEC; barrier
        0x7e020290u, 0x7da20300u,            // v1=16; v_cmpx_lt_u32 v0,v1
        0xd8f40004u, 0x0c000000u,            // ds_consume v12 offset:4 (LDS)
        0xbefe04c1u, 0xbf8a0000u,            // restore EXEC; barrier
        0xd8d80004u, 0x0d000003u,            // ds_read_b32 v13, v3 offset:4
        0x7e180d0cu, 0x7e1a0d0du,            // uint -> float
        0x06181b0cu, 0xbf810000u,             // v12 += v13; end
    };
    std::vector<uint32_t> spv32b6 = recompile_valu(code32b6, std::size(code32b6), 1, 12);
    CHECK(!spv32b6.empty(), "recompiled kernel 32b6 (LDS DS_CONSUME word) -> SPIR-V");
    std::vector<float> got32b6 = prosper::test::run_compute(spv32b6, in32b5, WG, WG);
    uint32_t bad32b6 = 0;
    for (uint32_t i = 0; i < WG && got32b6.size() == WG; ++i) {
        const float expected = i < 16 ? 84.0f : 133.0f;
        if (std::fabs(got32b6[i] - expected) > 1e-3f) ++bad32b6;
    }
    if (got32b6.size() == WG)
        printf("  kernel32b6 mismatches=%u out[0/15/16/63]=%g/%g/%g/%g expect=84/84/133/133\n",
               bad32b6, got32b6[0], got32b6[15], got32b6[16], got32b6[63]);
    CHECK(got32b6.size()==WG && bad32b6==0,
          "kernel 32b6 atomically subtracts popcount(EXEC), broadcasts old counter, and predicates VDST");

    // Kernel 32c: the same UE4 producer uses non-returning unsigned LDS min/max atomics. Start at
    // 10, min with 3, then max with 7; a read and uint->float conversion must report 7.
    const uint32_t code32c[] = {
        0x7e000280u, 0x7e02028au, 0x7e040283u, 0x7e060287u,
        0xd8340000u, 0x00000100u, 0xbf8a0000u,
        0xd81c0000u, 0x00000200u, 0xd8200000u, 0x00000300u, 0xbf8a0000u,
        0xd8d80000u, 0x04000000u, 0x7e080d04u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32c = recompile_valu(code32c, sizeof(code32c)/sizeof(code32c[0]), 0, 4);
    CHECK(!spv32c.empty(), "recompiled kernel 32c (ds_min_u32 + ds_max_u32) -> SPIR-V");
    std::vector<float> got32c = prosper::test::run_compute(spv32c, std::vector<float>(1), 1, 1);
    CHECK(got32c.size()==1 && std::fabs(got32c[0]-7.0f)<1e-3f,
          "kernel 32c LDS min/max atomics update shared memory exactly");

    // Plucky Squire's first-gameplay compute shaders use the non-returning DS_OR_B32 form. Seed an
    // LDS dword with 4, OR in 3 using the exact opcode-0x0a encoding, then read back 7.
    const uint32_t code32c_or[] = {
        0x7e000280u, 0x7e020284u,
        0xd8340000u, 0x00000100u,           // ds_write_b32 v0, v1
        0x7e040283u,
        0xd8280000u, 0x00000200u,           // ds_or_b32 v0, v2
        0xd8d80000u, 0x03000000u,           // ds_read_b32 v3, v0
        0x7e060d03u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32c_or = recompile_valu(
        code32c_or, std::size(code32c_or), 0, 3);
    CHECK(!spv32c_or.empty(), "recompiled kernel 32c (Plucky DS_OR_B32 word) -> SPIR-V");
    std::vector<float> got32c_or = prosper::test::run_compute(
        spv32c_or, std::vector<float>(1), 1, 1);
    CHECK(got32c_or.size()==1 && std::fabs(got32c_or[0]-7.0f)<1e-3f,
          "kernel 32c DS_OR_B32 atomically ORs the LDS dword");

    // Plucky Squire's first-gameplay compute path classifies floats before its LDS reduction. The
    // ten lanes below cover every GFX10 V_CMP_CLASS_F32 category; a second set deliberately selects
    // the next category so both the true and false VCC results are exercised. V_CNDMASK leaves SRC0
    // in v0 on true and selects the mask bits from v1 on false, making the result bit-exact even for
    // signalling/quiet NaNs and signed zero.
    const uint32_t code32c_class[] = {
        0x7d100300u,                         // v_cmp_class_f32 vcc, v0, v1
        0x02000101u,                         // v_cndmask_b32 v0, v1, v0, vcc
        0xbf810000u,
    };
    std::vector<uint32_t> spv32c_class = recompile_valu(
        code32c_class, std::size(code32c_class), 2, 0);
    CHECK(!spv32c_class.empty(),
          "recompiled kernel 32c (Plucky V_CMP_CLASS_F32 word) -> SPIR-V");
    const uint32_t class_values[10] = {
        0x7f800001u, 0x7fc00000u, 0xff800000u, 0xbf800000u, 0x80000001u,
        0x80000000u, 0x00000000u, 0x00000001u, 0x3f800000u, 0x7f800000u,
    };
    std::vector<float> in32c_class(40);
    std::vector<uint32_t> expected32c_class(20);
    for (uint32_t bit = 0; bit < 10; ++bit) {
        const uint32_t matching_mask = 1u << bit;
        const uint32_t nonmatching_mask = 1u << ((bit + 1u) % 10u);
        std::memcpy(&in32c_class[(2u * bit) * 2u], &class_values[bit], 4);
        std::memcpy(&in32c_class[(2u * bit) * 2u + 1u], &matching_mask, 4);
        std::memcpy(&in32c_class[(2u * bit + 1u) * 2u], &class_values[bit], 4);
        std::memcpy(&in32c_class[(2u * bit + 1u) * 2u + 1u], &nonmatching_mask, 4);
        expected32c_class[2u * bit] = class_values[bit];
        expected32c_class[2u * bit + 1u] = nonmatching_mask;
    }
    std::vector<float> got32c_class = prosper::test::run_compute(
        spv32c_class, in32c_class, 20, 20);
    uint32_t bad32c_class = 0;
    for (uint32_t i = 0; i < 20 && got32c_class.size() == 20; ++i)
        if (bits_of(got32c_class[i]) != expected32c_class[i]) ++bad32c_class;
    CHECK(got32c_class.size()==20 && bad32c_class==0,
          "kernel 32c V_CMP_CLASS_F32 distinguishes every IEEE-754 class and mask bit");

    // Kernel 32c2: ds_add_rtn_u32 returns the old value (10) while atomically updating LDS to 13.
    // Convert and add both observations so the single output must be 23.
    const uint32_t code32c2[] = {
        0x7e02028au, 0x7e040283u,
        0xd8340000u, 0x00000100u, 0xbf8a0000u,
        0xd8800000u, 0x03000200u, 0xbf8a0000u,
        0xd8d80000u, 0x04000000u, 0x7e060d03u, 0x7e080d04u,
        0x060a0903u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32c2 = recompile_valu(code32c2, sizeof(code32c2)/sizeof(code32c2[0]), 1, 5);
    CHECK(!spv32c2.empty(), "recompiled kernel 32c2 (ds_add_rtn_u32) -> SPIR-V");
    std::vector<float> in32c2(WG);
    for (uint32_t i = 0; i < WG; ++i) { const uint32_t addr = 4u*i; std::memcpy(&in32c2[i], &addr, 4); }
    std::vector<float> got32c2 = prosper::test::run_compute(spv32c2, in32c2, WG, WG);
    uint32_t bad32c2 = 0;
    for (uint32_t i = 0; i < WG && got32c2.size()==WG; ++i)
        if (std::fabs(got32c2[i]-23.0f) >= 1e-3f) ++bad32c2;
    printf("  kernel32c2 mismatches=%u out[0]=%g expect=23\n",
           bad32c2, got32c2.size()==WG ? got32c2[0] : -1.0f);
    CHECK(got32c2.size()==WG && bad32c2==0,
          "kernel 32c2 returns old LDS and atomically stores the sum");

    // Kernel 32c3: exact Astro DS_ADD_U32 form. Give every lane its own dword, initialize it to 10,
    // add 1 atomically without a return value, then read back 11.
    const uint32_t code32c3[] = {
        0x7e02028au,                              // v1 = 10
        0xd8340514u, 0x00000100u, 0xbf8a0000u,  // ds_write_b32 v0, v1 offset:0x514; barrier
        0x7e020281u,                              // v1 = 1
        0xd8000514u, 0x00000100u, 0xbf8a0000u,  // live: ds_add_u32 v0, v1 offset:0x514; barrier
        0xd8d80514u, 0x02000000u,                 // ds_read_b32 v2, v0 offset:0x514
        0x7e040d02u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32c3 = recompile_valu(
        code32c3, std::size(code32c3), 1, 2);
    CHECK(!spv32c3.empty(), "recompiled kernel 32c3 (Astro DS_ADD_U32 word) -> SPIR-V");
    std::vector<float> in32c3(WG);
    for (uint32_t i = 0; i < WG; ++i) {
        const uint32_t addr = 4u * i;
        std::memcpy(&in32c3[i], &addr, sizeof(addr));
    }
    std::vector<float> got32c3 = prosper::test::run_compute(spv32c3, in32c3, WG, WG);
    uint32_t bad32c3 = 0;
    for (uint32_t i = 0; i < WG && got32c3.size() == WG; ++i)
        if (std::fabs(got32c3[i] - 11.0f) > 1e-3f) ++bad32c3;
    CHECK(got32c3.size() == WG && bad32c3 == 0,
          "kernel 32c3 atomically adds DATA0 to each LDS dword");

    // Kernel 32c4: exact Astro DS_WRXCHG_RTN_B32 form. Swap 20 over 10 and observe both the
    // returned old value and the new LDS value: 10 + 20 = 30.
    const uint32_t code32c4[] = {
        0x7e02028au,
        0xd8340510u, 0x00000100u, 0xbf8a0000u,
        0x7e0402ffu, 0x00000014u,
        0xd8b40510u, 0x03000200u, 0xbf8a0000u,
        0xd8d80510u, 0x04000000u,
        0x7e060d03u, 0x7e080d04u, 0x06060903u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32c4 = recompile_valu(
        code32c4, std::size(code32c4), 1, 3);
    CHECK(!spv32c4.empty(),
          "recompiled kernel 32c4 (Astro DS_WRXCHG_RTN_B32 word) -> SPIR-V");
    std::vector<float> in32c4(WG);
    for (uint32_t i = 0; i < WG; ++i) {
        const uint32_t addr = 4u * i;
        std::memcpy(&in32c4[i], &addr, sizeof(addr));
    }
    std::vector<float> got32c4 = prosper::test::run_compute(spv32c4, in32c4, WG, WG);
    uint32_t bad32c4 = 0;
    for (uint32_t i = 0; i < WG && got32c4.size() == WG; ++i)
        if (std::fabs(got32c4[i] - 30.0f) > 1e-3f) ++bad32c4;
    CHECK(got32c4.size() == WG && bad32c4 == 0,
          "kernel 32c4 atomically exchanges LDS and returns the previous dword");

    // Kernel 32d: SDWA f16->f32 conversion selects WORD_1 before applying source negate. The packed
    // high half is -2, so `-WORD_1` converts to +2 (not a negation of the packed 32-bit payload).
    const uint32_t code32d[] = {
        0x7e0002ffu, 0xc0003c00u, 0x7e0216f9u, 0x00150600u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32d = recompile_valu(code32d, sizeof(code32d)/sizeof(code32d[0]), 0, 1);
    CHECK(!spv32d.empty(), "recompiled kernel 32d (v_cvt_f32_f16_sdwa WORD_1 negate) -> SPIR-V");
    std::vector<float> got32d = prosper::test::run_compute(spv32d, std::vector<float>(1), 1, 1);
    CHECK(got32d.size()==1 && std::fabs(got32d[0]-2.0f)<1e-3f,
          "kernel 32d selects the high f16 half before applying float modifiers");

    // Kernel 32e: packed f16 add/mul operate independently on both halves. Inputs are (1,2) and
    // (3,4); XORing add=(4,6) with mul=(3,8) makes both packed results observable as 0x0e000600.
    const uint32_t code32e[] = {
        0x7e0002ffu, 0x40003c00u, 0x7e0202ffu, 0x44004200u,
        0xcc0f4002u, 0x18020300u, 0xcc104003u, 0x18020300u,
        0x3a080702u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32e = recompile_valu(code32e, sizeof(code32e)/sizeof(code32e[0]), 0, 4);
    CHECK(!spv32e.empty(), "recompiled kernel 32e (v_pk_add_f16 + v_pk_mul_f16) -> SPIR-V");
    std::vector<float> got32e = prosper::test::run_compute(spv32e, std::vector<float>(1), 1, 1);
    CHECK(got32e.size()==1 && bits_of(got32e[0])==0x0e000600u,
          "kernel 32e packed f16 add/mul compute both halves exactly");

    // Kernel 32f: the live UE4 modifier word selects src1's high half for both results and negates
    // it in both low/high operations: (1* -4, 2* -4) -> packed (-4,-8).
    const uint32_t code32f[] = {
        0x7e0002ffu, 0x40003c00u, 0x7e0c02ffu, 0x44004200u,
        0xcc105202u, 0x58020d00u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32f = recompile_valu(code32f, sizeof(code32f)/sizeof(code32f[0]), 0, 2);
    CHECK(!spv32f.empty(), "recompiled kernel 32f (live v_pk_mul_f16 modifiers) -> SPIR-V");
    std::vector<float> got32f = prosper::test::run_compute(spv32f, std::vector<float>(1), 1, 1);
    CHECK(got32f.size()==1 && bits_of(got32f[0])==0xc800c400u,
          "kernel 32f honors packed low/high selects and negates");

    // Kernel 32g: exact live DCC-producer SDWA forms select src1 WORD_1 while src0 uses the low
    // half, write WORD_0, and preserve WORD_1. max((1,3)) -> (3,3), min((4,2)) -> (2,2).
    const uint32_t code32g[] = {
        0x7e0002ffu, 0x42003c00u, 0x7e0202ffu, 0x40004400u,
        0x720000f9u, 0x05061400u, 0x740202f9u, 0x05061401u,
        0x3a040300u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32g = recompile_valu(code32g, sizeof(code32g)/sizeof(code32g[0]), 0, 2);
    CHECK(!spv32g.empty(), "recompiled kernel 32g (live v_max/min_f16_sdwa forms) -> SPIR-V");
    std::vector<float> got32g = prosper::test::run_compute(spv32g, std::vector<float>(1), 1, 1);
    CHECK(got32g.size()==1 && bits_of(got32g[0])==0x02000200u,
          "kernel 32g selects f16 halves and preserves destinations exactly");

    // Kernel 32h: the live producer's paired v_add_f16_sdwa writes assemble a packed result in v7.
    // The first writes low=(1+3)=4; the second preserves it while writing high=(2+4)=6.
    const uint32_t code32h[] = {
        0x7e0002ffu, 0x42003c00u, 0x7e0202ffu, 0x40004400u,
        0x640e00f9u, 0x05061400u, 0x640e02f9u, 0x06051501u,
        0xbf810000u,
    };
    std::vector<uint32_t> spv32h = recompile_valu(code32h, sizeof(code32h)/sizeof(code32h[0]), 0, 7);
    CHECK(!spv32h.empty(), "recompiled kernel 32h (live paired v_add_f16_sdwa forms) -> SPIR-V");
    std::vector<float> got32h = prosper::test::run_compute(spv32h, std::vector<float>(1), 1, 1);
    CHECK(got32h.size()==1 && bits_of(got32h[0])==0x46004400u,
          "kernel 32h writes and preserves both packed f16 destination halves");

    // Kernel 32i: the live paired v_sub_f16_sdwa forms likewise assemble both halves in v0.
    // low=(1-4)=-3 and high=(5-6)=-1, with each instruction preserving the opposite half.
    const uint32_t code32i[] = {
        0x7e0002ffu, 0x44004000u, 0x7e0202ffu, 0x46004200u,
        0x7e1602ffu, 0x45003c00u, 0x660000f9u, 0x0506140bu,
        0x660002f9u, 0x0505150bu, 0xbf810000u,
    };
    std::vector<uint32_t> spv32i = recompile_valu(code32i, sizeof(code32i)/sizeof(code32i[0]), 0, 0);
    CHECK(!spv32i.empty(), "recompiled kernel 32i (live paired v_sub_f16_sdwa forms) -> SPIR-V");
    std::vector<float> got32i = prosper::test::run_compute(spv32i, std::vector<float>(1), 1, 1);
    CHECK(got32i.size()==1 && bits_of(got32i[0])==0xbc00c200u,
          "kernel 32i subtracts selected halves and preserves both destination halves");

    // Kernel 32j: v_pk_fmac_f16 accumulates both half lanes independently. Starting from (5,6),
    // (1,2)*(3,4)+(5,6) produces packed (8,14).
    const uint32_t code32j[] = {
        0x7e0002ffu, 0x40003c00u, 0x7e0202ffu, 0x44004200u,
        0x7e0402ffu, 0x46004500u, 0x78040300u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32j = recompile_valu(code32j, sizeof(code32j)/sizeof(code32j[0]), 0, 2);
    CHECK(!spv32j.empty(), "recompiled kernel 32j (v_pk_fmac_f16) -> SPIR-V");
    std::vector<float> got32j = prosper::test::run_compute(spv32j, std::vector<float>(1), 1, 1);
    CHECK(got32j.size()==1 && bits_of(got32j[0])==0x4b004800u,
          "kernel 32j accumulates both packed f16 lanes exactly");

    // Kernel 32k: exact live v_sqrt_f16_sdwa pair writes sqrt(4)=2 to a high half and sqrt(9)=3
    // to a low half, preserving the opposite destination halves in both cases.
    const uint32_t code32k[] = {
        0x7e1c02ffu, 0x45003c00u, 0x7e2c02ffu, 0x47004600u,
        0x7e2a02ffu, 0x48804400u, 0x7e1caaf9u, 0x00061515u,
        0x7e2caaf9u, 0x00051415u, 0x3a042d0eu, 0xbf810000u,
    };
    std::vector<uint32_t> spv32k = recompile_valu(code32k, sizeof(code32k)/sizeof(code32k[0]), 0, 2);
    CHECK(!spv32k.empty(), "recompiled kernel 32k (live paired v_sqrt_f16_sdwa forms) -> SPIR-V");
    std::vector<float> got32k = prosper::test::run_compute(spv32k, std::vector<float>(1), 1, 1);
    CHECK(got32k.size()==1 && bits_of(got32k[0])==0x07007e00u,
          "kernel 32k selects, roots, writes, and preserves packed f16 halves");

    // Kernel 32l: exact live v_cvt_f16_f32_sdwa forms write WORD_1 and preserve WORD_0. The first
    // aliases source/destination, while the second preserves an unrelated packed low word.
    const uint32_t code32l[] = {
        0x7e0202ffu, 0x40800000u, 0x7e0802ffu, 0x40000000u,
        0x7e0a02ffu, 0x12345678u, 0x7e0214f9u, 0x00061501u,
        0x7e0a14f9u, 0x00061504u, 0x3a040b01u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32l = recompile_valu(code32l, sizeof(code32l)/sizeof(code32l[0]), 0, 2);
    CHECK(!spv32l.empty(), "recompiled kernel 32l (live v_cvt_f16_f32_sdwa WORD_1 forms) -> SPIR-V");
    std::vector<float> got32l = prosper::test::run_compute(spv32l, std::vector<float>(1), 1, 1);
    CHECK(got32l.size()==1 && bits_of(got32l[0])==0x04005678u,
          "kernel 32l converts into high f16 halves while preserving low halves");

    // Kernel 32m: exact live f16 SDWA controls apply abs after selecting src1 WORD_1, and treat an
    // inline f32 constant as a numeric value before packing the product back into WORD_1.
    const uint32_t code32m[] = {
        0x7e0402ffu, 0xc4001234u, 0x7e0802ffu, 0x40005678u,
        0x640404f9u, 0x25861480u, 0x6a0808f9u, 0x058615f8u,
        0x3a0c0902u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32m = recompile_valu(code32m, sizeof(code32m)/sizeof(code32m[0]), 0, 6);
    CHECK(!spv32m.empty(), "recompiled kernel 32m (live f16 SDWA abs + inline float) -> SPIR-V");
    std::vector<float> got32m = prosper::test::run_compute(spv32m, std::vector<float>(1), 1, 1);
    CHECK(got32m.size()==1 && bits_of(got32m[0])==0xf1181278u,
          "kernel 32m applies packed f16 modifiers and inline constants exactly");

    // Kernel 32n: exact live cos/rcp f16 SDWA controls write WORD_1 and preserve WORD_0. cos(0
    // revolutions)=1 and rcp(2)=0.5; XOR makes both packed results observable.
    const uint32_t code32n[] = {
        0x7e0802ffu, 0x00001234u, 0x7e08c2f9u, 0x00051504u,
        0x7e0c0304u, 0x7e0802ffu, 0x40005678u, 0x7e08a8f9u,
        0x00051504u, 0x3a040906u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32n = recompile_valu(code32n, sizeof(code32n)/sizeof(code32n[0]), 0, 2);
    CHECK(!spv32n.empty(), "recompiled kernel 32n (live v_cos/rcp_f16_sdwa forms) -> SPIR-V");
    std::vector<float> got32n = prosper::test::run_compute(spv32n, std::vector<float>(1), 1, 1);
    CHECK(got32n.size()==1 && bits_of(got32n[0])==0x0400444cu,
          "kernel 32n evaluates packed f16 cosine/reciprocal and preserves low halves");

    // Kernel 32o: exact live v_fma_f16 VOP3 forms select a source high half and independently write
    // the low/high destination halves. Literals are f16 bit patterns; inline -2 remains numeric.
    const uint32_t code32o[] = {
        0x7e0602ffu, 0x12345678u, 0x7e0802ffu, 0x40005678u,
        0xd74b1003u, 0x23fe08ffu, 0x00004648u,
        0xd74b5004u, 0x03fe08f5u, 0x00004200u,
        0x3a040903u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32o = recompile_valu(code32o, sizeof(code32o)/sizeof(code32o[0]), 0, 2);
    CHECK(!spv32o.empty(), "recompiled kernel 32o (live v_fma_f16 VOP3 forms) -> SPIR-V");
    std::vector<float> got32o = prosper::test::run_compute(spv32o, std::vector<float>(1), 1, 1);
    CHECK(got32o.size()==1 && bits_of(got32o[0])==0xae349030u,
          "kernel 32o selects f16 sources and preserves the opposite destination halves");

    // Kernel 32o2: the same low-half fma with OMOD x2 must scale before f16 packing. The unscaled
    // result is -6.28125 (0xc648); x2 becomes -12.5625 (0xca48), preserving the high destination.
    const uint32_t code32o2[] = {
        0x7e0602ffu, 0x12345678u, 0x7e0802ffu, 0x40005678u,
        0xd74b1003u, 0x2bfe08ffu, 0x00004648u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32o2 = recompile_valu(code32o2, sizeof(code32o2)/sizeof(code32o2[0]), 0, 3);
    CHECK(!spv32o2.empty(), "recompiled kernel 32o2 (v_fma_f16 OMOD) -> SPIR-V");
    std::vector<float> got32o2 = prosper::test::run_compute(spv32o2, std::vector<float>(1), 1, 1);
    CHECK(got32o2.size()==1 && bits_of(got32o2[0])==0x1234ca48u,
          "kernel 32o2 applies VOP3 output scaling before f16 packing");

    // Kernel 32p: exact live v_cndmask_b32_sdwa selects src1 WORD_0 through VCC, writes WORD_1,
    // and preserves the destination's low half.
    const uint32_t code32p[] = {
        0x7e0002ffu, 0x3f800000u, 0x7e0202ffu, 0x3f800000u,
        0x7e0602ffu, 0x9999abcdu, 0x7e1c02ffu, 0x12345678u,
        0x7c040300u, 0x021c06f9u, 0x04861580u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32p = recompile_valu(code32p, sizeof(code32p)/sizeof(code32p[0]), 0, 14);
    CHECK(!spv32p.empty(), "recompiled kernel 32p (live v_cndmask_b32_sdwa WORD form) -> SPIR-V");
    std::vector<float> got32p = prosper::test::run_compute(spv32p, std::vector<float>(1), 1, 1);
    CHECK(got32p.size()==1 && bits_of(got32p[0])==0xabcd5678u,
          "kernel 32p selects and inserts a word while preserving the opposite half");

    // Kernel 32q: exact live CLAMP forms saturate before f16 packing. The first preserves WORD_0;
    // the final form reads the tracked vcc_lo scalar scratch and writes a zero-padded DWORD.
    const uint32_t code32q[] = {
        0x7e0802ffu, 0x40004000u, 0x6a0808f9u, 0x06053504u,
        0x7e0a0304u, 0xb06a385au, 0x6a0808f9u, 0x0686266au,
        0x3a040905u, 0xbf810000u,
    };
    std::vector<uint32_t> spv32q = recompile_valu(code32q, sizeof(code32q)/sizeof(code32q[0]), 0, 2);
    CHECK(!spv32q.empty(), "recompiled kernel 32q (live clamped/padded f16 SDWA forms) -> SPIR-V");
    std::vector<float> got32q = prosper::test::run_compute(spv32q, std::vector<float>(1), 1, 1);
    CHECK(got32q.size()==1 && bits_of(got32q[0])==0x3c007c00u,
          "kernel 32q clamps packed f16 results and zero-pads DWORD destinations");

    // Kernel 33: PACKED-format store. buffer_store_format_xyzw of (0,0.25,0.5,1.0) as UNORM8x4 packs to
    // bytes (0,64,128,255) = dword 0xFF804000 (inverse of the unorm8 unpack). Verifies pack_norm + the
    // tight-component packing into one dword.
    const uint32_t code33[] = {
        0x7e140f00u, 0x7e020280u, 0x7e0402ffu, 0x3e800000u, 0x7e0602f0u, 0x7e0802f2u,
        0xe01c2000u, 0x8002010au, 0xbf810000u,
    };
    ShaderResourceTable rt33;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Unorm8;
      vb.num_components = 4; vb.binding = 3; vb.stride = 4; vb.sgpr_base = 8; rt33.resources.push_back(vb); }
    std::vector<uint32_t> spv33 = recompile_valu(code33, sizeof(code33)/sizeof(code33[0]), 1, 0, &rt33);
    CHECK(!spv33.empty(), "recompiled kernel 33 (buffer_store_format_xyzw unorm8x4 pack) -> SPIR-V");
    std::vector<float> in33(N); for (uint32_t i = 0; i < N; i++) in33[i] = (float)i;
    std::vector<uint32_t> st33(N, 0), st33_out;
    prosper::test::run_compute(spv33, in33, N, N, /*cbuf0*/{}, /*cbuf1*/st33, &st33_out);
    uint32_t bad33 = 0;
    const uint8_t expB[4] = {0, 64, 128, 255};
    for (uint32_t i = 0; i < N && st33_out.size() == N; i++)
        for (int c = 0; c < 4; c++) { int got = (st33_out[i] >> (c*8)) & 0xFF; if (std::abs(got - (int)expB[c]) > 1) bad33++; }
    printf("  kernel33 mismatches=%u (buf[0]=0x%08x expect~0xff804000)\n", bad33, st33_out.size()==N?st33_out[0]:0);
    CHECK(st33_out.size() == N && bad33 == 0, "recompiled kernel 33 (unorm8x4 store packs to (0,64,128,255)) correct");

    // Kernel 34: signed scalar ALU + bitfield mask (SOP2 s_add_i32 0x02 / s_sub_i32 0x03 / s_bfm_b32 0x24).
    //   s0=20 s1=7 | s2=s0+s1=27 | s3=s0-s1=13 | s4=s_bfm(3,2)=((1<<3)-1)<<2=28 |
    //   s5=s2+s3=40 | s5=s5+s4=68 | v2=(float)68 | out=a0+68.  (dwords: llvm-mc gfx1030 round-trip)
    const uint32_t code34[] = {
        0xBE800394u, 0xBE810387u, 0x81020100u, 0x81830100u, 0x92048283u, 0x81050302u,
        0x81050405u, 0x7E040C05u, 0x06000500u, 0xBF810000u,
    };
    std::vector<uint32_t> spv34 = recompile_valu(code34, sizeof(code34)/sizeof(code34[0]), 1, 0);
    CHECK(!spv34.empty(), "recompiled kernel 34 (s_add_i32/s_sub_i32/s_bfm_b32) -> SPIR-V");
    std::vector<float> in34(N), exp34(N);
    for (uint32_t i = 0; i < N; i++) { in34[i] = (float)i; exp34[i] = (float)i + 68.0f; }
    std::vector<float> got34 = prosper::test::run_compute(spv34, in34, N, N);
    uint32_t bad34 = 0; for (uint32_t i=0;i<N&&got34.size()==N;i++) if (std::fabs(got34[i]-exp34[i])>1e-3f) bad34++;
    printf("  kernel34 mismatches=%u (out[5]=%g expect=%g)\n", bad34, got34.size()==N?got34[5]:-1, exp34[5]);
    CHECK(got34.size()==N && bad34==0, "recompiled kernel 34 (s_add_i32/s_sub_i32/s_bfm_b32) computes a0+68");

    // Kernel 35: s_wqm_b64 (SOP1 0x0a) is the identity mask op in our scalar model (helper lanes implicit).
    //   s_wqm_b64 exec,exec (no-op) | s0=5 | v2=(float)5 | out=a0+5.  Proves WQM is accepted and does NOT
    //   corrupt EXEC / spuriously predicate the following write. (dwords: llvm-mc gfx1030 round-trip)
    const uint32_t code35[] = {
        0xBEFE0A7Eu, 0xBE800385u, 0x7E040C00u, 0x06000500u, 0xBF810000u,
    };
    std::vector<uint32_t> spv35 = recompile_valu(code35, sizeof(code35)/sizeof(code35[0]), 1, 0);
    CHECK(!spv35.empty(), "recompiled kernel 35 (s_wqm_b64 identity) -> SPIR-V");
    std::vector<float> in35(N), exp35(N);
    for (uint32_t i = 0; i < N; i++) { in35[i] = (float)i; exp35[i] = (float)i + 5.0f; }
    std::vector<float> got35 = prosper::test::run_compute(spv35, in35, N, N);
    uint32_t bad35 = 0; for (uint32_t i=0;i<N&&got35.size()==N;i++) if (std::fabs(got35[i]-exp35[i])>1e-3f) bad35++;
    printf("  kernel35 mismatches=%u (out[5]=%g expect=%g)\n", bad35, got35.size()==N?got35[5]:-1, exp35[5]);
    CHECK(got35.size()==N && bad35==0, "recompiled kernel 35 (s_wqm_b64 identity, exec intact) computes a0+5");

    // Kernel 36: v_mad_f32 (VOP3 0x141) = src0*src1+src2. A gfx10.1 op removed in gfx10.3 (llvm-mc gfx1030
    //   rejects it) but emitted by the PS5 compiler. v1=2.0 | v2=3.0 | v0 = v1*v0+v2 = 2*a0+3.
    //   (dwords: llvm-mc gfx1010 round-trip)
    const uint32_t code36[] = {
        0x7E0202F4u, 0x7E0402FFu, 0x40400000u, 0xD5410000u, 0x040A0101u, 0xBF810000u,
    };
    std::vector<uint32_t> spv36 = recompile_valu(code36, sizeof(code36)/sizeof(code36[0]), 1, 0);
    CHECK(!spv36.empty(), "recompiled kernel 36 (v_mad_f32, gfx10.1 op) -> SPIR-V");
    std::vector<float> in36(N), exp36(N);
    for (uint32_t i = 0; i < N; i++) { in36[i] = (float)i; exp36[i] = 2.0f*(float)i + 3.0f; }
    std::vector<float> got36 = prosper::test::run_compute(spv36, in36, N, N);
    uint32_t bad36 = 0; for (uint32_t i=0;i<N&&got36.size()==N;i++) if (std::fabs(got36[i]-exp36[i])>1e-3f) bad36++;
    printf("  kernel36 mismatches=%u (out[5]=%g expect=%g)\n", bad36, got36.size()==N?got36[5]:-1, exp36[5]);
    CHECK(got36.size()==N && bad36==0, "recompiled kernel 36 (v_mad_f32) computes 2*a0+3");

    // Kernel 37: s_movk_i32 (SOPK 0x00) sign-extends a 16-bit immediate. s0 = sext(0xff9c) = -100;
    //   v2 = (float)(int)s0 = -100.0; out = a0 + v2 = a0 - 100. Verifies SOPK decode + sign extension.
    const uint32_t code37[] = {
        0xB000FF9Cu, 0x7E040A00u, 0x06000500u, 0xBF810000u,
    };
    std::vector<uint32_t> spv37 = recompile_valu(code37, sizeof(code37)/sizeof(code37[0]), 1, 0);
    CHECK(!spv37.empty(), "recompiled kernel 37 (s_movk_i32 sign-extend) -> SPIR-V");
    std::vector<float> in37(N), exp37(N);
    for (uint32_t i = 0; i < N; i++) { in37[i] = (float)i; exp37[i] = (float)i - 100.0f; }
    std::vector<float> got37 = prosper::test::run_compute(spv37, in37, N, N);
    uint32_t bad37 = 0; for (uint32_t i=0;i<N&&got37.size()==N;i++) if (std::fabs(got37[i]-exp37[i])>1e-3f) bad37++;
    printf("  kernel37 mismatches=%u (out[5]=%g expect=%g)\n", bad37, got37.size()==N?got37[5]:-1, exp37[5]);
    CHECK(got37.size()==N && bad37==0, "recompiled kernel 37 (s_movk_i32 -100) computes a0-100");

    // Astro Bot's world-map material kernel scales a scalar-table index with this exact SOPK word.
    // VCC_LO is used as ordinary scalar data here; 3*276 must survive as the following VALU source.
    const uint32_t code37m[] = {
        0xbeea03ffu, 0x00000003u,   // s_mov_b32 vcc_lo, 3
        0xb86a0114u,                // s_mulk_i32 vcc_lo, 276
        0x7e000a6au,                // v_cvt_f32_u32 v0, vcc_lo
        0xbf810000u,
    };
    std::vector<uint32_t> spv37m = recompile_valu(
        code37m, std::size(code37m), 0, 0);
    CHECK(!spv37m.empty(), "recompiled kernel 37m (Astro s_mulk_i32 word) -> SPIR-V");
    std::vector<float> got37m = prosper::test::run_compute(
        spv37m, std::vector<float>(1), 1, 1);
    CHECK(got37m.size()==1 && got37m[0]==828.0f,
          "kernel 37m multiplies scalar data by the signed SOPK immediate exactly");

    // Kernel 37a: s_cmpk_le_u32 compares the SGPR named by SOPK's SDST field against the raw
    // unsigned 16-bit immediate and writes SCC without changing that SGPR. Cover both sides of the
    // 0xffff boundary used by Astro Bot's NGG primitive-count setup.
    const uint32_t code37a_true[] = {
        0xB000007Bu, 0xB700FFFFu, 0x850287AAu, 0x7E000C02u, 0xBF810000u,
    };
    const uint32_t code37a_false[] = {
        0xB000FFFFu, 0xB700FFFFu, 0x850287AAu, 0x7E000C02u, 0xBF810000u,
    };
    std::vector<uint32_t> spv37a_true = recompile_valu(
        code37a_true, std::size(code37a_true), 1, 0);
    std::vector<uint32_t> spv37a_false = recompile_valu(
        code37a_false, std::size(code37a_false), 1, 0);
    CHECK(!spv37a_true.empty() && !spv37a_false.empty(),
          "recompiled kernel 37a (s_cmpk_le_u32, both SCC paths) -> SPIR-V");
    std::vector<float> got37a_true = prosper::test::run_compute(spv37a_true, in37, N, N);
    std::vector<float> got37a_false = prosper::test::run_compute(spv37a_false, in37, N, N);
    uint32_t bad37a = 0;
    for (uint32_t i=0;i<N&&got37a_true.size()==N;i++)
        if (got37a_true[i] != 42.0f) bad37a++;
    for (uint32_t i=0;i<N&&got37a_false.size()==N;i++)
        if (got37a_false[i] != 7.0f) bad37a++;
    CHECK(got37a_true.size()==N && got37a_false.size()==N && bad37a==0,
          "kernel 37a: s_cmpk_le_u32 treats 0xffff as unsigned and drives s_cselect");

    // Astro Bot's loading composite starts with s_brev_b32 s16,1 to construct 0x80000000.
    // AMD ISA 70648 also specifies that S_BREV does not set SCC. Seed SCC=true, reverse bit 0,
    // then select the reversed value through SCC so the test covers both contracts.
    const uint32_t code37b[] = {
        0xBE800381u, 0xBF068100u, 0xBE800B00u, 0x85018900u, 0x7E000C01u, 0xBF810000u,
    };
    std::vector<uint32_t> spv37b = recompile_valu(
        code37b, sizeof(code37b)/sizeof(code37b[0]), 1, 0);
    CHECK(!spv37b.empty(), "recompiled kernel 37b (s_brev_b32) -> SPIR-V");
    std::vector<float> got37b = prosper::test::run_compute(spv37b, in37, N, N);
    uint32_t bad37b = 0;
    for (uint32_t i=0;i<N&&got37b.size()==N;i++)
        if (got37b[i] != 2147483648.0f) bad37b++;
    CHECK(got37b.size()==N && bad37b==0,
          "kernel 37b reverses scalar bit 0 into bit 31 without changing SCC");

    // Kernel 38: s_cselect_b64 (SOP2 0x0b) — mask-domain select. s_cmp_eq_u32 5,5 sets SCC=1, then
    //   s_cselect_b64 vcc, exec, 0 => vcc = SCC ? exec : 0 = exec (all lanes on). v_cndmask v0,v1(42),
    //   v2(10),vcc picks src1=10 when vcc true. out = 10.0 (proves the selected mask reaches cndmask).
    const uint32_t code38[] = {
        0xBE800385u, 0xBF068500u, 0x85EA807Eu, 0x7E0202FFu, 0x42280000u, 0x7E0402FFu, 0x41200000u,
        0x02000501u, 0xBF810000u,
    };
    std::vector<uint32_t> spv38 = recompile_valu(code38, sizeof(code38)/sizeof(code38[0]), 1, 0);
    CHECK(!spv38.empty(), "recompiled kernel 38 (s_cselect_b64 mask select) -> SPIR-V");
    std::vector<float> in38(N); for (uint32_t i=0;i<N;i++) in38[i] = (float)i;
    std::vector<float> got38 = prosper::test::run_compute(spv38, in38, N, N);
    uint32_t bad38 = 0; for (uint32_t i=0;i<N&&got38.size()==N;i++) if (std::fabs(got38[i]-10.0f)>1e-3f) bad38++;
    printf("  kernel38 mismatches=%u (out[5]=%g expect=10)\n", bad38, got38.size()==N?got38[5]:-1);
    CHECK(got38.size()==N && bad38==0, "recompiled kernel 38 (s_cselect_b64 -> exec mask -> cndmask) computes 10");

    // Kernel 39: trivial SDWA (all sels = DWORD) with SGPR operands — the form the game uses to give a
    //   VOP2 two scalar sources (e32 can't). s0=100, s1=23; v_add_nc_u32_sdwa v2, s0, s1 (all DWORD) = 123;
    //   v3=(float)123; out = a0 + 123. Verifies the decoder decodes SDWA operands + un-flags the no-op case.
    const uint32_t code39[] = {
        0xBE8003FFu, 0x00000064u, 0xBE810397u, 0x4A0402F9u, 0x86860600u, 0x7E060D02u, 0x06000700u, 0xBF810000u,
    };
    std::vector<uint32_t> spv39 = recompile_valu(code39, sizeof(code39)/sizeof(code39[0]), 1, 0);
    CHECK(!spv39.empty(), "recompiled kernel 39 (trivial SDWA add, SGPR operands) -> SPIR-V");
    std::vector<float> in39(N), exp39(N);
    for (uint32_t i = 0; i < N; i++) { in39[i] = (float)i; exp39[i] = (float)i + 123.0f; }
    std::vector<float> got39 = prosper::test::run_compute(spv39, in39, N, N);
    uint32_t bad39 = 0; for (uint32_t i=0;i<N&&got39.size()==N;i++) if (std::fabs(got39[i]-exp39[i])>1e-3f) bad39++;
    printf("  kernel39 mismatches=%u (out[5]=%g expect=%g)\n", bad39, got39.size()==N?got39[5]:-1, exp39[5]);
    CHECK(got39.size()==N && bad39==0, "recompiled kernel 39 (SDWA all-DWORD add s0+s1) computes a0+123");

    // Kernel 40: v_mul_hi_u32 (VOP3 0x16a) — high 32 bits of a u32*u32 via OpUMulExtended. v1=2^31;
    //   v2 = hi(2^31 * 2^31) = hi(2^62) = 2^30 = 0x40000000; out = (float)2^30 = 1073741824. (Also proves
    //   the OpUMulExtended opcode number: a wrong one is rejected at vkCreateShaderModule -> {} -> FAIL.)
    const uint32_t code40[] = {
        0x7E0202FFu, 0x80000000u, 0xD56A0002u, 0x00020301u, 0x7E000D02u, 0xBF810000u,
    };
    std::vector<uint32_t> spv40 = recompile_valu(code40, sizeof(code40)/sizeof(code40[0]), 1, 0);
    CHECK(!spv40.empty(), "recompiled kernel 40 (v_mul_hi_u32 / OpUMulExtended) -> SPIR-V");
    std::vector<float> in40(N); for (uint32_t i=0;i<N;i++) in40[i]=(float)i;
    std::vector<float> got40 = prosper::test::run_compute(spv40, in40, N, N);
    uint32_t bad40 = 0; for (uint32_t i=0;i<N&&got40.size()==N;i++) if (std::fabs(got40[i]-1073741824.0f)>64.0f) bad40++;
    printf("  kernel40 mismatches=%u (out[5]=%g expect=1073741824)\n", bad40, got40.size()==N?got40[5]:-1);
    CHECK(got40.size()==N && bad40==0, "recompiled kernel 40 (v_mul_hi_u32) computes hi(2^62)=2^30");

    // Kernel 41: v_readfirstlane_b32 (VOP1 0x02, uniform src) + s_lshr_b32 (SOP2 0x20) + v_subrev_nc_u32
    //   (VOP2 0x27). s0=100; v1=100 (uniform); s1=readfirstlane(v1)=100; s1>>=1 => 50; v2=50; v3=8;
    //   v4 = v2 - v3 = 42 (reverse subtract); out = 42. (readfirstlane on a uniform value == hardware.)
    const uint32_t code41[] = {
        0xB0000064u, 0x7E020200u, 0x7E020501u, 0x90018101u, 0x7E040201u, 0x7E060288u, 0x4E080503u,
        0x7E000D04u, 0xBF810000u,
    };
    std::vector<uint32_t> spv41 = recompile_valu(code41, sizeof(code41)/sizeof(code41[0]), 1, 0);
    CHECK(!spv41.empty(), "recompiled kernel 41 (readfirstlane/s_lshr_b32/v_subrev_nc_u32) -> SPIR-V");
    std::vector<float> in41(N); for (uint32_t i=0;i<N;i++) in41[i]=(float)i;
    std::vector<float> got41 = prosper::test::run_compute(spv41, in41, N, N);
    uint32_t bad41 = 0; for (uint32_t i=0;i<N&&got41.size()==N;i++) if (std::fabs(got41[i]-42.0f)>1e-3f) bad41++;
    printf("  kernel41 mismatches=%u (out[5]=%g expect=42)\n", bad41, got41.size()==N?got41[5]:-1);
    CHECK(got41.size()==N && bad41==0, "recompiled kernel 41 (readfirstlane+s_lshr_b32+v_subrev) computes 42");

    // Kernel 42: s_mul_hi_u32 (SOP2 0x35) — scalar high 32 bits. s0=2^30; s1=hi(2^60)=2^28=0x10000000;
    //   out=(float)2^28=268435456. Covers the scalar mul_hi path (shares umul_hi with kernel 40's vector).
    const uint32_t code42[] = { 0xBE8003F4u, 0x9A810000u, 0x7E000C01u, 0xBF810000u };
    std::vector<uint32_t> spv42 = recompile_valu(code42, sizeof(code42)/sizeof(code42[0]), 1, 0);
    CHECK(!spv42.empty(), "recompiled kernel 42 (s_mul_hi_u32) -> SPIR-V");
    std::vector<float> in42(N); for (uint32_t i=0;i<N;i++) in42[i]=(float)i;
    std::vector<float> got42 = prosper::test::run_compute(spv42, in42, N, N);
    uint32_t bad42 = 0; for (uint32_t i=0;i<N&&got42.size()==N;i++) if (std::fabs(got42[i]-268435456.0f)>16.0f) bad42++;
    printf("  kernel42 mismatches=%u (out[5]=%g expect=268435456)\n", bad42, got42.size()==N?got42[5]:-1);
    CHECK(got42.size()==N && bad42==0, "recompiled kernel 42 (s_mul_hi_u32) computes hi(2^60)=2^28");

    // Astro Bot's NGG primitive packer combines two 16-bit population counts with
    // s_pack_ll_b32_b16. Pack 1 and 2 -> 0x00020001, then convert to an exactly representable f32.
    const uint32_t code42b[] = {
        0xB0000001u, 0xB0010002u, 0x99020100u, 0x7E000C02u, 0xBF810000u,
    };
    std::vector<uint32_t> spv42b = recompile_valu(
        code42b, sizeof(code42b) / sizeof(code42b[0]), 0, 0);
    CHECK(!spv42b.empty(), "recompiled kernel 42b (s_pack_ll_b32_b16) -> SPIR-V");
    std::vector<float> got42b = prosper::test::run_compute(spv42b, in42, N, N);
    uint32_t bad42b = 0;
    for (uint32_t i = 0; i < N && got42b.size() == N; ++i)
        if (got42b[i] != 131073.0f) ++bad42b;
    CHECK(got42b.size() == N && bad42b == 0,
          "s_pack_ll_b32_b16 packs {S1.low16,S0.low16}");

    // Kernel 43: a real COUNTED LOOP reconstructed as structured SPIR-V (OpLoopMerge + OpPhi for the
    //   loop-carried s0 and v1). sum=0; for (i=0; i<5; i++) sum+=i;  =>  sum = 0+1+2+3+4 = 10.
    //     s_movk_i32 s2,5 | s_mov s0,0 | v_mov v1,0
    //   loop: s_cmp_lt_u32 s0,s2 | s_cbranch_scc0 exit | v_add_nc_u32 v1,s0,v1 | s_add_i32 s0,s0,1 | s_branch loop
    //   exit: v_cvt_f32_u32 v0,v1 | s_endpgm.   (assembled to object by llvm-mc gfx1010; offsets resolved)
    const uint32_t code43[] = {
        0xB0020005u, 0xBE800380u, 0x7E020280u, 0xBF0A0200u, 0xBF840003u,
        0x4A020200u, 0x81008100u, 0xBF82FFFBu, 0x7E000D01u, 0xBF810000u,
    };
    std::vector<uint32_t> spv43 = recompile_valu(code43, sizeof(code43)/sizeof(code43[0]), 0, 0);
    CHECK(!spv43.empty(), "recompiled kernel 43 (counted loop -> OpLoopMerge/OpPhi) -> SPIR-V");
    std::vector<float> in43(N); for (uint32_t i=0;i<N;i++) in43[i]=(float)i;
    std::vector<float> got43 = prosper::test::run_compute(spv43, in43, N, N);
    uint32_t bad43 = 0; for (uint32_t i=0;i<N&&got43.size()==N;i++) if (std::fabs(got43[i]-10.0f)>1e-3f) bad43++;
    printf("  kernel43 mismatches=%u (out[5]=%g expect=10)\n", bad43, got43.size()==N?got43[5]:-1);
    CHECK(got43.size()==N && bad43==0, "recompiled kernel 43 (counted loop sum 0..4) computes 10");

    // Kernel 43a: one ordinary uniform if immediately before a counted loop. Astro Bot's NGG culling
    // shader uses this composition in its setup; the two constructs were individually supported, but
    // the counted-loop prelude previously admitted only if/else. Exercise both PHI inputs by changing
    // the constant comparison: the taken form seeds sum=7, while the skipped form retains sum=0.
    const uint32_t code43a_taken[] = {
        0xBE800383u, 0xBE810385u, 0xBF0A0100u, 0x7E020280u, 0xBF840001u, 0x7E020287u,
        0xB0020005u, 0xBE800380u, 0xBF0A0200u, 0xBF840003u, 0x4A020200u, 0x81008100u,
        0xBF82FFFBu, 0x7E000D01u, 0xBF810000u,
    };
    const uint32_t code43a_skipped[] = {
        0xBE800385u, 0xBE810383u, 0xBF0A0100u, 0x7E020280u, 0xBF840001u, 0x7E020287u,
        0xB0020005u, 0xBE800380u, 0xBF0A0200u, 0xBF840003u, 0x4A020200u, 0x81008100u,
        0xBF82FFFBu, 0x7E000D01u, 0xBF810000u,
    };
    const auto spv43a_taken = recompile_valu(
        code43a_taken, std::size(code43a_taken), 0, 0);
    const auto spv43a_skipped = recompile_valu(
        code43a_skipped, std::size(code43a_skipped), 0, 0);
    CHECK(!spv43a_taken.empty() && !spv43a_skipped.empty(),
          "recompiled kernel 43a (one-arm pre-loop if + counted loop) -> SPIR-V");
    const auto got43a_taken = prosper::test::run_compute(spv43a_taken, in43, N, N);
    const auto got43a_skipped = prosper::test::run_compute(spv43a_skipped, in43, N, N);
    uint32_t bad43a = 0;
    for (uint32_t i = 0; i < N && got43a_taken.size() == N && got43a_skipped.size() == N; ++i)
        if (got43a_taken[i] != 17.0f || got43a_skipped[i] != 10.0f) ++bad43a;
    CHECK(got43a_taken.size() == N && got43a_skipped.size() == N && bad43a == 0,
          "kernel 43a merges taken/skipped pre-loop values into the counted loop exactly");

    // Kernel 43b: a pre-loop conditional that jumps over the complete counted loop to s_endpgm.
    // The prelude CFG scan uses the loop header as an artificial end marker, so it must preserve the
    // early-out classification and reject rather than silently execute the loop on both outcomes.
    const uint32_t code43b[] = {
        0xBE800385u, 0xBE810383u, 0xBF0A0100u, 0xBF840009u, 0xB0020005u,
        0xBE800380u, 0x7E020280u, 0xBF0A0200u, 0xBF840003u, 0x4A020200u,
        0x81008100u, 0xBF82FFFBu, 0x7E000D01u, 0xBF810000u,
    };
    const auto spv43b = recompile_valu(code43b, std::size(code43b), 0, 0);
    CHECK(spv43b.empty(),
          "kernel 43b rejects a pre-loop early-out that skips the counted loop");

    // Kernel 44: a value advanced in the CONDITION region and read after the loop. s0 is incremented by
    //   10 at the header (which runs on the exiting iteration too), so hardware yields 40, not the phi's
    //   back-edge value 30. Regression guard for the condition-region-vs-body merge-value distinction.
    //     s0=0; s1=0; header: s0+=10; if(!(s1<3)) exit; s1++; goto header; exit: out=(float)s0  => 40
    const uint32_t code44[] = {
        0xBE800380u, 0xBE810380u, 0x80008A00u, 0xBF0A8301u, 0xBF840002u,
        0x80018101u, 0xBF82FFFBu, 0x7E000C00u, 0xBF810000u,
    };
    std::vector<uint32_t> spv44 = recompile_valu(code44, sizeof(code44)/sizeof(code44[0]), 0, 0);
    CHECK(!spv44.empty(), "recompiled kernel 44 (condition-region carry) -> SPIR-V");
    std::vector<float> in44(N); for (uint32_t i=0;i<N;i++) in44[i]=(float)i;
    std::vector<float> got44 = prosper::test::run_compute(spv44, in44, N, N);
    uint32_t bad44 = 0; for (uint32_t i=0;i<N&&got44.size()==N;i++) if (std::fabs(got44[i]-40.0f)>1e-3f) bad44++;
    printf("  kernel44 mismatches=%u (out[5]=%g expect=40)\n", bad44, got44.size()==N?got44[5]:-1);
    CHECK(got44.size()==N && bad44==0, "recompiled kernel 44 (condition-region reg read after loop) computes 40");

    // Kernel 44b: a BOTTOM-tested do-while loop whose back-edge is a conditional s_cbranch_scc1 (not
    //   an unconditional s_branch + separate forward exit). The whole body runs, THEN the SCC test at
    //   the bottom decides whether to loop — the shape Alex Kidd's (PPSA02664) light/blur-accumulation
    //   pixel shaders use, which previously rejected and dropped their draws (#320). sum=0; i=0;
    //   do { sum+=i; i++; } while (i<5)  =>  sum = 0+1+2+3+4 = 10.
    //     s_mov s0,0 | v_mov v1,0
    //   loop(header): v_add_nc_u32 v1,s0,v1 | s_add_i32 s0,s0,1 | s_cmp_lt_u32 s0,5 | s_cbranch_scc1 loop
    //   exit: v_cvt_f32_u32 v0,v1 | s_endpgm
    const uint32_t code44b[] = {
        0xBE800380u, 0x7E020280u, 0x4A020200u, 0x81008100u, 0xBF0A8500u, 0xBF85FFFCu,
        0x7E000D01u, 0xBF810000u,
    };
    std::vector<uint32_t> spv44b = recompile_valu(code44b, sizeof(code44b)/sizeof(code44b[0]), 0, 0);
    CHECK(!spv44b.empty(), "recompiled kernel 44b (do-while s_cbranch_scc1 back-edge) -> SPIR-V");
    std::vector<float> in44b(N); for (uint32_t i=0;i<N;i++) in44b[i]=(float)i;
    std::vector<float> got44b = prosper::test::run_compute(spv44b, in44b, N, N);
    uint32_t bad44b = 0; for (uint32_t i=0;i<N&&got44b.size()==N;i++) if (std::fabs(got44b[i]-10.0f)>1e-3f) bad44b++;
    printf("  kernel44b mismatches=%u (out[5]=%g expect=10)\n", bad44b, got44b.size()==N?got44b[5]:-1);
    CHECK(got44b.size()==N && bad44b==0, "recompiled kernel 44b (do-while scc1 loop sum 0..4) computes 10");

    // Kernel 44c: the same do-while with the OPPOSITE back-edge polarity — s_cbranch_scc0 continues
    //   while SCC==0. Exit test s_cmp_ge_u32 s0,5 (SCC = s0>=5), so the loop runs while s0<5. Same
    //   result (10); guards the exit_on_scc0 polarity handling for the conditional back-edge.
    const uint32_t code44c[] = {
        0xBE800380u, 0x7E020280u, 0x4A020200u, 0x81008100u, 0xBF098500u, 0xBF84FFFCu,
        0x7E000D01u, 0xBF810000u,
    };
    std::vector<uint32_t> spv44c = recompile_valu(code44c, sizeof(code44c)/sizeof(code44c[0]), 0, 0);
    CHECK(!spv44c.empty(), "recompiled kernel 44c (do-while s_cbranch_scc0 back-edge) -> SPIR-V");
    std::vector<float> in44c(N); for (uint32_t i=0;i<N;i++) in44c[i]=(float)i;
    std::vector<float> got44c = prosper::test::run_compute(spv44c, in44c, N, N);
    uint32_t bad44c = 0; for (uint32_t i=0;i<N&&got44c.size()==N;i++) if (std::fabs(got44c[i]-10.0f)>1e-3f) bad44c++;
    printf("  kernel44c mismatches=%u (out[5]=%g expect=10)\n", bad44c, got44c.size()==N?got44c[5]:-1);
    CHECK(got44c.size()==N && bad44c==0, "recompiled kernel 44c (do-while scc0 loop sum 0..4) computes 10");

    // Kernel 44d: COEXISTENCE lock — a top-tested counted loop (s_branch back-edge, kernel-43 shape)
    //   AND a readfirstlane waterfall (T19 shape, backward s_cbranch_scc1 with wave-mask SCC) in ONE
    //   kernel. The waterfall's backward scc branch must NOT count as a second loop back-edge (it is
    //   classified by waterfall_branches and linearized via safe_branches); counting it would reject
    //   the whole shader — a previously-recompilable combination (review finding on #320's do-while
    //   change). out = table[u] + sum(0..4) = table[u] + 10, table = 5/7/9 for u = 0/1/2.
    const uint32_t code44d[] = {
        0x7E020F00u,              // v_cvt_u32_f32 v1, v0            (pc 0: index u)
        0xBE86047Eu,              // s_mov_b64 s[6:7], exec          (pc 1: REMAIN = exec)
        0x7E0402FFu, 0x40A00000u, // v_mov_b32 v2, 5.0f              (pc 2)
        0x7E0602FFu, 0x40E00000u, // v_mov_b32 v3, 7.0f              (pc 4)
        0x7E0802FFu, 0x41100000u, // v_mov_b32 v4, 9.0f              (pc 6)
        0xB0020005u,              // s_movk_i32 s2, 5                (pc 8)
        0xBE800380u,              // s_mov_b32 s0, 0                 (pc 9)
        0x7E0C0280u,              // v_mov_b32 v6, 0                 (pc 10)
        0xBF0A0200u,              // loop: s_cmp_lt_u32 s0, s2       (pc 11: header)
        0xBF840003u,              // s_cbranch_scc0 exit(+3 -> 16)   (pc 12)
        0x4A0C0C00u,              // v_add_nc_u32 v6, s0, v6         (pc 13)
        0x81008100u,              // s_add_i32 s0, s0, 1             (pc 14)
        0xBF82FFFBu,              // s_branch loop(-5 -> 11)         (pc 15)
        0x7E080501u,              // exit: v_readfirstlane_b32 s4,v1 (pc 16: waterfall head)
        0x7DA40204u,              // v_cmpx_eq_u32 s4, v1            (pc 17)
        0xBEFC0304u,              // s_mov_b32 m0, s4                (pc 18)
        0x7E0A8702u,              // v_movrels_b32 v5, v2            (pc 19)
        0x8A867E06u,              // s_andn2_b64 s[6:7], s[6:7],exec (pc 20: wave-mask SCC)
        0xBEFE0406u,              // s_mov_b64 exec, s[6:7]          (pc 21)
        0xBF85FFF9u,              // s_cbranch_scc1 (-7 -> 16)       (pc 22: waterfall back-branch)
        0xBEFE04C1u,              // s_mov_b64 exec, -1              (pc 23)
        0x7E0C0D06u,              // v_cvt_f32_u32 v6, v6            (pc 24)
        0x06000D05u,              // v_add_f32 v0, v5, v6            (pc 25)
        0xBF810000u,              // s_endpgm                        (pc 26)
    };
    std::vector<uint32_t> spv44d = recompile_valu(code44d, sizeof(code44d)/sizeof(code44d[0]), 1, 0);
    CHECK(!spv44d.empty(), "recompiled kernel 44d (counted loop + waterfall coexist) -> SPIR-V");
    std::vector<float> in44d(8), exp44d(8);
    const float tab44d[3] = { 5.0f, 7.0f, 9.0f };
    for (uint32_t i = 0; i < 8; i++) { in44d[i] = (float)(i % 3); exp44d[i] = tab44d[i % 3] + 10.0f; }
    std::vector<float> got44d = prosper::test::run_compute(spv44d, in44d, 8, 8);
    uint32_t bad44d = 0; for (uint32_t i=0;i<8&&got44d.size()==8;i++) if (std::fabs(got44d[i]-exp44d[i])>1e-3f) bad44d++;
    printf("  kernel44d mismatches=%u (out[1]=%g expect=%g)\n", bad44d, got44d.size()==8?got44d[1]:-1, exp44d[1]);
    CHECK(got44d.size()==8 && bad44d==0, "kernel 44d: counted loop + waterfall compute table[u]+10");

    // Kernel 44e: NEGATIVE — a do-while whose body also has a forward scc exit (an inner uniform
    //   break). Two exit tests exceed the single-exit contract, so the detector must conservatively
    //   reject (nexit != 0 for a do-while) and the shader stays fail-visible rather than mis-lowering.
    //     s0=0; v1=0; loop: v1+=s0; s0++; if (s0==3) break; while (s0<5)
    const uint32_t code44e[] = {
        0xBE800380u,              // s_mov_b32 s0, 0                 (pc 0)
        0x7E020280u,              // v_mov_b32 v1, 0                 (pc 1)
        0x4A020200u,              // loop: v_add_nc_u32 v1, s0, v1   (pc 2: header)
        0x81008100u,              // s_add_i32 s0, s0, 1             (pc 3)
        0xBF068300u,              // s_cmp_eq_u32 s0, 3              (pc 4)
        0xBF850002u,              // s_cbranch_scc1 exit(+2 -> 8)    (pc 5: inner break)
        0xBF0A8500u,              // s_cmp_lt_u32 s0, 5              (pc 6)
        0xBF85FFFAu,              // s_cbranch_scc1 loop(-6 -> 2)    (pc 7: do-while back-edge)
        0x7E000D01u,              // exit: v_cvt_f32_u32 v0, v1      (pc 8)
        0xBF810000u,              // s_endpgm                        (pc 9)
    };
    std::vector<uint32_t> spv44e = recompile_valu(code44e, sizeof(code44e)/sizeof(code44e[0]), 0, 0);
    CHECK(spv44e.empty(), "kernel 44e: do-while with inner forward scc break conservatively rejects");

    // Kernel 45: EXEC-narrowed-state tracking regression (the review-flagged under-narrow bug). Save all-on
    //   exec to vcc; v_cmp overwrites vcc with a per-lane mask (lanes i<4); restore exec FROM vcc — exec is
    //   now narrowed, so v_mov v2,7 must be EXEC-predicated (only i<4 get 7; i>=4 keep 0). If the narrowed
    //   flag desynced from vcc's value, the write would escape predication and ALL lanes get 7. out=(i<4?7:0).
    //   (v0 = (uint)input[gid] = gid) so the per-lane compare is meaningful.
    const uint32_t code45[] = {
        0x7E000F00u, 0x7E040280u, 0xBEEA047Eu, 0x7D880084u, 0xBEFE046Au, 0x7E040287u, 0xBEFE04C1u, 0x7E000D02u, 0xBF810000u,
    };
    std::vector<uint32_t> spv45 = recompile_valu(code45, sizeof(code45)/sizeof(code45[0]), 1, 0);
    CHECK(!spv45.empty(), "recompiled kernel 45 (exec-narrowed restore-from-vcc) -> SPIR-V");
    std::vector<float> in45(N); for (uint32_t i=0;i<N;i++) in45[i]=(float)i;
    std::vector<float> got45 = prosper::test::run_compute(spv45, in45, N, N);
    uint32_t bad45 = 0; for (uint32_t i=0;i<N&&got45.size()==N;i++) if (std::fabs(got45[i]-(i<4?7.0f:0.0f))>1e-3f) bad45++;
    printf("  kernel45 mismatches=%u (out[2]=%g expect=7, out[10]=%g expect=0)\n",
           bad45, got45.size()==N?got45[2]:-1, got45.size()==N?got45[10]:-1);
    CHECK(got45.size()==N && bad45==0, "recompiled kernel 45 (restored exec stays narrowed -> predicated write)");

    // Kernel 46: v_madak_f32 (VOP2 0x21, mandatory literal K) = src0*src1 + K. v1=2.0; v0 = v0*v1 + 1.0 =
    //   2*a0 + 1. Verifies the opcode + that the decoder consumes the trailing K literal (else desync).
    const uint32_t code46[] = { 0x7E0202F4u, 0x42000300u, 0x3F800000u, 0xBF810000u };
    std::vector<uint32_t> spv46 = recompile_valu(code46, sizeof(code46)/sizeof(code46[0]), 1, 0);
    CHECK(!spv46.empty(), "recompiled kernel 46 (v_madak_f32 + literal) -> SPIR-V");
    std::vector<float> in46(N), exp46(N);
    for (uint32_t i=0;i<N;i++) { in46[i]=(float)i; exp46[i]=2.0f*(float)i+1.0f; }
    std::vector<float> got46 = prosper::test::run_compute(spv46, in46, N, N);
    uint32_t bad46 = 0; for (uint32_t i=0;i<N&&got46.size()==N;i++) if (std::fabs(got46[i]-exp46[i])>1e-3f) bad46++;
    printf("  kernel46 mismatches=%u (out[5]=%g expect=11)\n", bad46, got46.size()==N?got46[5]:-1);
    CHECK(got46.size()==N && bad46==0, "recompiled kernel 46 (v_madak_f32: 2*a0+1) correct");

    // Kernel 47: a SCALAR write inside a divergent (execz) IF-block that is DEAD at the merge — the
    // relaxation added for the tonemap/sRGB shaders (033). v3=100; vcc=(u0>u1); s_and_saveexec; execz;
    //   { s5 = 50.0; v3 = s5 + u0 }   (predicated VGPR write; s5 is a wave-uniform scalar written
    //   unconditionally under linearization); restore exec; s5 = 5.0 (redefines s5 -> the block's s5 is
    // dead); v3 = s5 + v3. Active lanes: (u0+50)+5 = u0+55; masked lanes keep v3=100 -> 105. Proves the
    // unconditional in-block scalar write does NOT corrupt masked lanes (its value is never observed past
    // the merge), i.e. the dead-at-merge liveness relaxation is sound.
    const uint32_t code47[] = {
        0x7e0602ffu, 0x42c80000u, 0x7d880300u, 0xbe80246au, 0xbf880003u, 0xbe8503ffu, 0x42480000u,
        0x06060005u, 0xbefe0400u, 0xbe8503ffu, 0x40a00000u, 0x06060605u, 0xbf810000u,
    };
    std::vector<uint32_t> spv47 = recompile_valu(code47, sizeof(code47)/sizeof(code47[0]), 2, /*out_vgpr*/3);
    CHECK(!spv47.empty(), "recompiled kernel 47 (dead scalar write in divergent block) -> SPIR-V");
    std::vector<float> in47(N * 2), exp47(N);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t u0 = i % 17, u1 = i % 13;
        in47[i*2+0]=(float)u0; in47[i*2+1]=(float)u1;
        exp47[i] = (u0 > u1) ? ((float)u0 + 55.0f) : 105.0f;
    }
    std::vector<float> got47 = prosper::test::run_compute(spv47, in47, N, N);
    uint32_t bad47 = 0, act47 = 0, msk47 = 0;
    for (uint32_t i=0;i<N&&got47.size()==N;i++){ if (std::fabs(got47[i]-exp47[i])>1e-3f) bad47++;
        if ((i%17)>(i%13)) act47++; else msk47++; }
    printf("  kernel47 mismatches=%u (active=%u masked=%u, out[1]=%g exp=%g)\n", bad47, act47, msk47,
           got47.size()==N?got47[1]:-1, exp47[1]);
    CHECK(act47>0 && msk47>0, "kernel 47 exercises both active and masked lanes");
    CHECK(got47.size()==N && bad47==0, "recompiled kernel 47 (dead scalar write in divergent block) correct");

    // Kernel 47b: same shape but the block's s5 is LIVE at the merge. The linearization shortcut must
    // not consume it; structured compute execz instead branches on subgroupAny(EXEC), making the scalar
    // write wave-uniform. Wave 0 enters and produces 102; wave 1 is empty and preserves s5=0 / v3=100.
    const uint32_t code47b[] = {
        0x7e0602ffu, 0x42c80000u, 0x7d880300u, 0xbe80246au, 0xbf880003u, 0xbe8503ffu, 0x42480000u,
        0x06060005u, 0xbefe0400u, 0x06060605u, 0xbf810000u,
    };
    std::vector<uint32_t> spv47b = recompile_valu(code47b, sizeof(code47b)/sizeof(code47b[0]), 2, /*out_vgpr*/3);
    CHECK(!spv47b.empty(), "recompiled kernel 47b (live scalar write in wave-uniform execz block) -> SPIR-V");
    std::vector<float> in47b(N * 2), exp47b(N);
    for (uint32_t i = 0; i < N; i++) {
        const bool active_wave = i < 64;
        in47b[i*2+0] = active_wave ? 2.0f : 1.0f;
        in47b[i*2+1] = active_wave ? 1.0f : 2.0f;
        exp47b[i] = active_wave ? 102.0f : 100.0f;
    }
    std::vector<float> got47b = prosper::test::run_compute(spv47b, in47b, N, N);
    uint32_t bad47b = 0;
    for (uint32_t i=0;i<N&&got47b.size()==N;i++)
        if (std::fabs(got47b[i]-exp47b[i])>1e-3f) bad47b++;
    CHECK(got47b.size()==N && bad47b==0,
          "compute execz preserves live scalar state on both wave-uniform paths");

    // Kernel 47c (NEGATIVE, soundness): the in-block "scalar move" targets vcc_lo (SDST code 106, which
    // decodes as SGPR-kind). A move into VCC/EXEC/M0 has wave-wide side effects read IMPLICITLY, so it can
    // never be proven dead — the relaxation must exclude dst > s105 and reject the block (else EXEC/VCC
    // corruption past the merge). Must return empty.
    const uint32_t code47c[] = {
        0x7e0602ffu, 0x42c80000u, 0x7d880300u, 0xbe80246au, 0xbf880003u, 0xbeea03ffu, 0x42480000u,
        0x06060100u, 0xbefe0400u, 0x02060703u, 0xbf810000u,
    };
    std::vector<uint32_t> spv47c = recompile_valu(code47c, sizeof(code47c)/sizeof(code47c[0]), 2, /*out_vgpr*/3);
    CHECK(spv47c.empty(), "kernel 47c (special-reg (vcc) write in divergent block) correctly REJECTED");

    // Kernel 47c2 (#1390): Plucky's gameplay post-process has the same divergent VCC scratch write,
    // but the value is dead after the merge. A later, unrelated forward EXECZ splits control flow;
    // BOTH paths reach a non-X VOPC that overwrites VCC before any read. The CFG liveness proof must
    // follow both successors rather than rejecting merely because lexical scanning met a branch.
    const uint32_t code47c2[] = {
        0x7e0602ffu, 0x40e00000u,            // v3=7.0 default
        0x7d880300u, 0xbe80246au,             // vcc=(u0>u1); saveexec s[0:1],vcc
        0xbf880002u,                          // execz -> pc 7 (merge)
        0xbeea0385u,                          // guarded vcc_lo=5 scalar scratch
        0x7e06026au,                          // guarded v3=vcc_lo (raw bits 5)
        0xbefe0400u,                          // pc 7: restore exec
        0xbf880001u,                          // unrelated execz: both pc 9 and pc 10 are possible
        0x7e080304u,                          // fallthrough no-op v4=v4
        0x7d820100u,                          // both paths overwrite VCC (v0<v0 = false)
        0xbf810000u,
    };
    std::vector<uint32_t> spv47c2 = recompile_valu(
        code47c2, std::size(code47c2), 2, 3);
    CHECK(!spv47c2.empty(),
          "kernel 47c2 (dead VCC scratch across post-merge branch) recompiles");
    std::vector<float> got47c2 = prosper::test::run_compute(spv47c2, in18, N, N);
    uint32_t bad47c2 = 0;
    for (uint32_t i = 0; i < N && got47c2.size() == N; ++i) {
        const uint32_t expected = (i % 17) > (i % 13) ? 5u : bits_of(7.0f);
        if (bits_of(got47c2[i]) != expected) ++bad47c2;
    }
    CHECK(got47c2.size()==N && bad47c2==0,
          "kernel 47c2 preserves predicated VGPR results while dead VCC scratch is discarded");

    // Negative sibling: the fallthrough path reads VCC through V_CNDMASK before the common overwrite.
    // Even though the branch's taken path is safe, all-successor liveness must reject the block.
    std::vector<uint32_t> code47c3(std::begin(code47c2), std::end(code47c2));
    code47c3[9] = 0x02060703u;                // v_cndmask_b32 v3,v3,v3,vcc (implicit VCC read)
    CHECK(recompile_valu(code47c3.data(), code47c3.size(), 2, 3).empty(),
          "kernel 47c3 rejects when either post-merge branch path reads VCC scratch");

    // Kernel 47d (NEGATIVE, soundness): s5 is written in the block, then an SOPK s_addk_i32 s5 (a
    // read-modify-write whose SIMM16 operand leaves n_src==0) reads s5 after the merge -> s5 is LIVE. The
    // liveness scan must NOT mistake the SOPK dst for a redefinition; the block must be rejected. Empty.
    const uint32_t code47d[] = {
        0x7e0602ffu, 0x42c80000u, 0x7d880300u, 0xbe80246au, 0xbf880003u, 0xbe8503ffu, 0x42480000u,
        0x06060005u, 0xbefe0400u, 0xb7850010u, 0x06060605u, 0xbf810000u,
    };
    std::vector<uint32_t> spv47d = recompile_valu(code47d, sizeof(code47d)/sizeof(code47d[0]), 2, /*out_vgpr*/3);
    CHECK(spv47d.empty(), "kernel 47d (SOPK RMW reads block scalar after merge) correctly REJECTED");

    // Kernel 48: v_mac_f32 (VOP2 0x1f) = src0*src1 + dst (accumulate). v3 = 3.0; v3 = a0*a1 + v3.
    // op 0x1f is v_mac_f32 on the PS5 ISA — the RDNA1/gfx1010 encoding (removed on desktop gfx1030, where
    // 0x1f is invalid). Verified by round-tripping the scene VS's op-0x1f word 0x3e261221 through
    // `llvm-mc -mcpu=gfx1010` → v_mac_f32_e32. The scene VS uses this heavily for its vertex transform.
    const uint32_t code48[] = { 0x7E0602FFu, 0x40400000u, 0x3E060300u, 0xBF810000u };
    std::vector<uint32_t> spv48 = recompile_valu(code48, sizeof(code48)/sizeof(code48[0]), 2, /*out_vgpr*/3);
    CHECK(!spv48.empty(), "recompiled kernel 48 (v_mac_f32 accumulate) -> SPIR-V");
    std::vector<float> in48(N * 2), exp48(N);
    for (uint32_t i = 0; i < N; i++) { float a0 = (float)(i % 7), a1 = (float)(i % 5);
        in48[i*2+0]=a0; in48[i*2+1]=a1; exp48[i] = a0 * a1 + 3.0f; }
    std::vector<float> got48 = prosper::test::run_compute(spv48, in48, N, N);
    uint32_t bad48 = 0; for (uint32_t i=0;i<N&&got48.size()==N;i++) if (std::fabs(got48[i]-exp48[i])>1e-3f) bad48++;
    printf("  kernel48 mismatches=%u (out[8]=%g expect=%g)\n", bad48, got48.size()==N?got48[8]:-1, exp48[8]);
    CHECK(got48.size()==N && bad48==0, "recompiled kernel 48 (v_mac_f32: a0*a1+3) correct");

    // Kernel 48b: v_mac_f32_e64 (VOP3 op 0x11f — the e64 form of VOP2 0x1f). v3 = 5.0; v3 = a0*a1 + v3.
    // The scene VS emits this e64 form (with source modifiers); verifies the VOP3 0x11f -> v_mac path.
    // Encoding via `llvm-mc -mcpu=gfx1010`: v_mac_f32_e64 v3, v0, v1 = [0xd51f0003, 0x00020300].
    const uint32_t code48b[] = { 0x7E0602FFu, 0x40A00000u, 0xd51f0003u, 0x00020300u, 0xBF810000u };
    std::vector<uint32_t> spv48b = recompile_valu(code48b, sizeof(code48b)/sizeof(code48b[0]), 2, /*out_vgpr*/3);
    CHECK(!spv48b.empty(), "recompiled kernel 48b (v_mac_f32_e64 VOP3) -> SPIR-V");
    std::vector<float> in48b(N * 2), exp48b(N);
    for (uint32_t i = 0; i < N; i++) { float a0 = (float)(i % 6), a1 = (float)(i % 4);
        in48b[i*2+0]=a0; in48b[i*2+1]=a1; exp48b[i] = a0 * a1 + 5.0f; }
    std::vector<float> got48b = prosper::test::run_compute(spv48b, in48b, N, N);
    uint32_t bad48b = 0; for (uint32_t i=0;i<N&&got48b.size()==N;i++) if (std::fabs(got48b[i]-exp48b[i])>1e-3f) bad48b++;
    printf("  kernel48b mismatches=%u (out[7]=%g expect=%g)\n", bad48b, got48b.size()==N?got48b[7]:-1, exp48b[7]);
    CHECK(got48b.size()==N && bad48b==0, "recompiled kernel 48b (v_dot2c_f32_f16 packed dot) correct");

    // Kernel 49: s_bfe_u64 (SOP2 0x29, 64-bit bitfield extract via Int64). s0=0xabcdef12, s1=0x77 ->
    // the 64-bit value 0x00000077_abcdef12; extract width=8 at offset=32 (src1=0x80020) = bits[32:39] =
    // 0x77. v0 = (float)s2 = 119. Verifies the SGPR-pair combine spans the 32-bit boundary correctly.
    const uint32_t code49[] = { 0xBE8003FFu,0xABCDEF12u,0xBE8103FFu,0x00000077u,0x9482FF00u,0x00080020u,
                                0x7E000202u,0x7E000D00u,0xBF810000u };
    std::vector<uint32_t> spv49 = recompile_valu(code49, sizeof(code49)/sizeof(code49[0]), 1, /*out_vgpr*/0);
    CHECK(!spv49.empty(), "recompiled kernel 49 (s_bfe_u64 / Int64) -> SPIR-V");
    std::vector<float> in49(N, 0.0f);
    std::vector<float> got49 = prosper::test::run_compute(spv49, in49, N, N);
    uint32_t bad49 = 0; for (uint32_t i=0;i<N&&got49.size()==N;i++) if (std::fabs(got49[i]-119.0f)>1e-3f) bad49++;
    printf("  kernel49 mismatches=%u (out[0]=%g expect=119)\n", bad49, got49.size()==N?got49[0]:-1);
    CHECK(got49.size()==N && bad49==0, "recompiled kernel 49 (s_bfe_u64: bits[32:39] of 0x77abcdef12 = 0x77) correct");

    // Kernel 50: v_cndmask_b32_e64 (VOP3 0x101) with a VCC mask: vcc=(u0>u1); v3 = vcc ? 2.0 : 1.0.
    const uint32_t code50[] = { 0x7D880300u, 0xD5010003u, 0x01A9E8F2u, 0xBF810000u };
    std::vector<uint32_t> spv50 = recompile_valu(code50, sizeof(code50)/sizeof(code50[0]), 2, /*out_vgpr*/3);
    CHECK(!spv50.empty(), "recompiled kernel 50 (v_cndmask_b32_e64) -> SPIR-V");
    std::vector<float> in50(N * 2), exp50(N);
    for (uint32_t i = 0; i < N; i++) { uint32_t u0 = i % 17, u1 = i % 13;
        in50[i*2+0]=(float)u0; in50[i*2+1]=(float)u1; exp50[i] = (u0 > u1) ? 2.0f : 1.0f; }
    std::vector<float> got50 = prosper::test::run_compute(spv50, in50, N, N);
    uint32_t bad50 = 0; for (uint32_t i=0;i<N&&got50.size()==N;i++) if (std::fabs(got50[i]-exp50[i])>1e-3f) bad50++;
    printf("  kernel50 mismatches=%u (out[1]=%g)\n", bad50, got50.size()==N?got50[1]:-1);
    CHECK(got50.size()==N && bad50==0, "recompiled kernel 50 (v_cndmask_b32_e64: vcc?2:1) correct");

    // Kernel 51: VOPC to an SGPR-pair mask (SDWAB SD form) + v_cndmask_b32_e64 reading it — the NGG-cull
    // shader 040 pattern. s[4:5]=(u0>u1); v3 = s[4:5] ? 2.0 : 1.0. Verifies v_cmp writes an SGPR mask
    // (not VCC) and that v_cndmask_e64 resolves that mask.
    const uint32_t code51[] = { 0x7D8802F9u, 0x06068400u, 0xD5010003u, 0x0011E8F2u, 0xBF810000u };
    std::vector<uint32_t> spv51 = recompile_valu(code51, sizeof(code51)/sizeof(code51[0]), 2, /*out_vgpr*/3);
    CHECK(!spv51.empty(), "recompiled kernel 51 (v_cmp->SGPR mask + v_cndmask_e64) -> SPIR-V");
    std::vector<float> in51(N * 2), exp51(N);
    for (uint32_t i = 0; i < N; i++) { uint32_t u0 = i % 17, u1 = i % 13;
        in51[i*2+0]=(float)u0; in51[i*2+1]=(float)u1; exp51[i] = (u0 > u1) ? 2.0f : 1.0f; }
    std::vector<float> got51 = prosper::test::run_compute(spv51, in51, N, N);
    uint32_t bad51 = 0; for (uint32_t i=0;i<N&&got51.size()==N;i++) if (std::fabs(got51[i]-exp51[i])>1e-3f) bad51++;
    printf("  kernel51 mismatches=%u (out[1]=%g)\n", bad51, got51.size()==N?got51[1]:-1);
    CHECK(got51.size()==N && bad51==0, "recompiled kernel 51 (v_cmp->SGPR mask + v_cndmask_e64) correct");

    // Kernel 52: VOP3 float SOURCE MODIFIERS (neg/abs) — previously silently ignored (→ wrong results).
    // v_fma_f32 v3, -v0, v1, v2  =  -a0*a1 + a2   (src0 negate: dword1 bit29). Encoding via llvm-mc gfx1010.
    const uint32_t code52[] = { 0xD54B0003u, 0x240A0300u, 0xBF810000u };
    std::vector<uint32_t> spv52 = recompile_valu(code52, sizeof(code52)/sizeof(code52[0]), 3, /*out_vgpr*/3);
    CHECK(!spv52.empty(), "recompiled kernel 52 (v_fma_f32 with -src0) -> SPIR-V");
    std::vector<float> in52(N * 3), exp52(N);
    for (uint32_t i = 0; i < N; i++) { float a0=(float)i*0.1f-6.0f, a1=(float)i*0.02f+1.0f, a2=3.0f;
        in52[i*3+0]=a0; in52[i*3+1]=a1; in52[i*3+2]=a2; exp52[i] = -a0*a1 + a2; }
    std::vector<float> got52 = prosper::test::run_compute(spv52, in52, N, N);
    uint32_t bad52 = 0; for (uint32_t i=0;i<N&&got52.size()==N;i++) if (std::fabs(got52[i]-exp52[i])>1e-3f) bad52++;
    printf("  kernel52 mismatches=%u (out[50]=%g expect=%g)\n", bad52, got52.size()==N?got52[50]:-1, exp52[50]);
    CHECK(got52.size()==N && bad52==0, "recompiled kernel 52 (VOP3 -src0: -a0*a1+a2) correct");

    // Kernel 53: VOP3 ABS source. v_fma_f32 v3, |v0|, v1, v2 = |a0|*a1 + a2 (src0 abs: dword0 bit8).
    const uint32_t code53[] = { 0xD54B0103u, 0x040A0300u, 0xBF810000u };
    std::vector<uint32_t> spv53 = recompile_valu(code53, sizeof(code53)/sizeof(code53[0]), 3, /*out_vgpr*/3);
    CHECK(!spv53.empty(), "recompiled kernel 53 (v_fma_f32 with |src0|) -> SPIR-V");
    std::vector<float> in53(N * 3), exp53(N);
    for (uint32_t i = 0; i < N; i++) { float a0=(float)i*0.1f-6.0f, a1=2.0f, a2=1.0f;
        in53[i*3+0]=a0; in53[i*3+1]=a1; in53[i*3+2]=a2; exp53[i] = std::fabs(a0)*a1 + a2; }
    std::vector<float> got53 = prosper::test::run_compute(spv53, in53, N, N);
    uint32_t bad53 = 0; for (uint32_t i=0;i<N&&got53.size()==N;i++) if (std::fabs(got53[i]-exp53[i])>1e-3f) bad53++;
    printf("  kernel53 mismatches=%u (out[10]=%g expect=%g)\n", bad53, got53.size()==N?got53[10]:-1, exp53[10]);
    CHECK(got53.size()==N && bad53==0, "recompiled kernel 53 (VOP3 |src0|: |a0|*a1+a2) correct");

    // Kernel 54: VOP3 CLAMP output modifier. v_fma_f32 v3, v0, v1, v2 clamp = saturate(a0*a1+a2) to [0,1].
    const uint32_t code54[] = { 0xD54B8003u, 0x040A0300u, 0xBF810000u };
    std::vector<uint32_t> spv54 = recompile_valu(code54, sizeof(code54)/sizeof(code54[0]), 3, /*out_vgpr*/3);
    CHECK(!spv54.empty(), "recompiled kernel 54 (v_fma_f32 clamp) -> SPIR-V");
    std::vector<float> in54(N * 3), exp54(N);
    for (uint32_t i = 0; i < N; i++) { float a0=(float)i*0.05f-2.0f, a1=1.0f, a2=0.25f; float r=a0*a1+a2;
        in54[i*3+0]=a0; in54[i*3+1]=a1; in54[i*3+2]=a2; exp54[i] = r<0.0f?0.0f:(r>1.0f?1.0f:r); }
    std::vector<float> got54 = prosper::test::run_compute(spv54, in54, N, N);
    uint32_t bad54 = 0; for (uint32_t i=0;i<N&&got54.size()==N;i++) if (std::fabs(got54[i]-exp54[i])>1e-3f) bad54++;
    printf("  kernel54 mismatches=%u (out[5]=%g expect=%g out[100]=%g)\n", bad54, got54.size()==N?got54[5]:-1, exp54[5], got54.size()==N?got54[100]:-1);
    CHECK(got54.size()==N && bad54==0, "recompiled kernel 54 (VOP3 clamp: saturate(a0*a1+a2)) correct");

    // Kernel 55: VOP3 OMOD (mul:2). v_fma_f32 v3, v0, v1, v2 mul:2 = 2*(a0*a1+a2).
    const uint32_t code55[] = { 0xD54B0003u, 0x0C0A0300u, 0xBF810000u };
    std::vector<uint32_t> spv55 = recompile_valu(code55, sizeof(code55)/sizeof(code55[0]), 3, /*out_vgpr*/3);
    CHECK(!spv55.empty(), "recompiled kernel 55 (v_fma_f32 mul:2) -> SPIR-V");
    std::vector<float> in55(N * 3), exp55(N);
    for (uint32_t i = 0; i < N; i++) { float a0=(float)i*0.1f-6.0f, a1=0.5f, a2=1.0f;
        in55[i*3+0]=a0; in55[i*3+1]=a1; in55[i*3+2]=a2; exp55[i] = 2.0f*(a0*a1+a2); }
    std::vector<float> got55 = prosper::test::run_compute(spv55, in55, N, N);
    uint32_t bad55 = 0; for (uint32_t i=0;i<N&&got55.size()==N;i++) if (std::fabs(got55[i]-exp55[i])>1e-3f) bad55++;
    printf("  kernel55 mismatches=%u (out[80]=%g expect=%g)\n", bad55, got55.size()==N?got55[80]:-1, exp55[80]);
    CHECK(got55.size()==N && bad55==0, "recompiled kernel 55 (VOP3 omod mul:2: 2*(a0*a1+a2)) correct");

    // Kernels 56-58: VOP3-encoded VOP2 float ops (were rejected) with modifiers.
    // 56: v_add_f32_e64 v3, -v0, v1  = -a0 + a1  (VOP3 v_add_f32 0x103 + neg src0).
    const uint32_t code56[] = { 0xD5030003u, 0x20020300u, 0xBF810000u };
    std::vector<uint32_t> spv56 = recompile_valu(code56, 3, 2, 3);
    CHECK(!spv56.empty(), "recompiled kernel 56 (VOP3 v_add_f32 -src0) -> SPIR-V");
    std::vector<float> in56(N*2), exp56(N);
    for (uint32_t i=0;i<N;i++){ float a0=(float)i*0.1f-3.0f,a1=(float)i*0.05f+2.0f; in56[i*2]=a0; in56[i*2+1]=a1; exp56[i]=-a0+a1; }
    std::vector<float> got56 = prosper::test::run_compute(spv56, in56, N, N);
    uint32_t bad56=0; for(uint32_t i=0;i<N&&got56.size()==N;i++) if(std::fabs(got56[i]-exp56[i])>1e-3f) bad56++;
    printf("  kernel56 mismatches=%u (out[40]=%g expect=%g)\n", bad56, got56.size()==N?got56[40]:-1, exp56[40]);
    CHECK(got56.size()==N && bad56==0, "recompiled kernel 56 (VOP3 v_add_f32 -a0+a1) correct");

    // 57: v_mul_f32_e64 v3, v0, |v1|  = a0 * |a1|  (VOP3 v_mul_f32 0x108 + abs src1).
    const uint32_t code57[] = { 0xD5080203u, 0x00020300u, 0xBF810000u };
    std::vector<uint32_t> spv57 = recompile_valu(code57, 3, 2, 3);
    CHECK(!spv57.empty(), "recompiled kernel 57 (VOP3 v_mul_f32 |src1|) -> SPIR-V");
    std::vector<float> in57(N*2), exp57(N);
    for (uint32_t i=0;i<N;i++){ float a0=(float)i*0.1f-2.0f,a1=(float)i*0.07f-4.0f; in57[i*2]=a0; in57[i*2+1]=a1; exp57[i]=a0*std::fabs(a1); }
    std::vector<float> got57 = prosper::test::run_compute(spv57, in57, N, N);
    uint32_t bad57=0; for(uint32_t i=0;i<N&&got57.size()==N;i++) if(std::fabs(got57[i]-exp57[i])>1e-3f) bad57++;
    printf("  kernel57 mismatches=%u (out[10]=%g expect=%g)\n", bad57, got57.size()==N?got57[10]:-1, exp57[10]);
    CHECK(got57.size()==N && bad57==0, "recompiled kernel 57 (VOP3 v_mul_f32 a0*|a1|) correct");

    // 58: v_max_f32_e64 v3, v0, v1  = max(a0,a1)  (VOP3 v_max_f32 0x110).
    const uint32_t code58[] = { 0xD5100003u, 0x00020300u, 0xBF810000u };
    std::vector<uint32_t> spv58 = recompile_valu(code58, 3, 2, 3);
    CHECK(!spv58.empty(), "recompiled kernel 58 (VOP3 v_max_f32) -> SPIR-V");
    std::vector<float> in58(N*2), exp58(N);
    for (uint32_t i=0;i<N;i++){ float a0=(float)i*0.1f-6.0f,a1=(float)i*-0.03f+1.0f; in58[i*2]=a0; in58[i*2+1]=a1; exp58[i]=a0>a1?a0:a1; }
    std::vector<float> got58 = prosper::test::run_compute(spv58, in58, N, N);
    uint32_t bad58=0; for(uint32_t i=0;i<N&&got58.size()==N;i++) if(std::fabs(got58[i]-exp58[i])>1e-3f) bad58++;
    printf("  kernel58 mismatches=%u (out[90]=%g expect=%g)\n", bad58, got58.size()==N?got58[90]:-1, exp58[90]);
    CHECK(got58.size()==N && bad58==0, "recompiled kernel 58 (VOP3 v_max_f32) correct");

    // Kernels 59-60: SDWA source modifiers (were rejected via has_modifier). Shader-030 pattern
    // v_mul_f32_sdwa v14,-s13,v3. 59: v_mul_f32_sdwa v3,-v0,v1 = -a0*a1 (src0 neg, DWORD sels).
    const uint32_t code59[] = { 0x100602F9u, 0x06161600u, 0xBF810000u };
    std::vector<uint32_t> spv59 = recompile_valu(code59, 3, 2, 3);
    CHECK(!spv59.empty(), "recompiled kernel 59 (v_mul_f32_sdwa -src0) -> SPIR-V");
    std::vector<float> in59(N*2), exp59(N);
    for (uint32_t i=0;i<N;i++){ float a0=(float)i*0.1f-4.0f,a1=(float)i*0.03f+1.0f; in59[i*2]=a0; in59[i*2+1]=a1; exp59[i]=-a0*a1; }
    std::vector<float> got59 = prosper::test::run_compute(spv59, in59, N, N);
    uint32_t bad59=0; for(uint32_t i=0;i<N&&got59.size()==N;i++) if(std::fabs(got59[i]-exp59[i])>1e-3f) bad59++;
    printf("  kernel59 mismatches=%u (out[30]=%g expect=%g)\n", bad59, got59.size()==N?got59[30]:-1, exp59[30]);
    CHECK(got59.size()==N && bad59==0, "recompiled kernel 59 (SDWA -src0: -a0*a1) correct");

    // 60: v_mul_f32_sdwa v3, |v0|, v1 = |a0|*a1 (src0 abs).
    const uint32_t code60[] = { 0x100602F9u, 0x06261600u, 0xBF810000u };
    std::vector<uint32_t> spv60 = recompile_valu(code60, 3, 2, 3);
    CHECK(!spv60.empty(), "recompiled kernel 60 (v_mul_f32_sdwa |src0|) -> SPIR-V");
    std::vector<float> in60(N*2), exp60(N);
    for (uint32_t i=0;i<N;i++){ float a0=(float)i*0.1f-4.0f,a1=2.0f; in60[i*2]=a0; in60[i*2+1]=a1; exp60[i]=std::fabs(a0)*a1; }
    std::vector<float> got60 = prosper::test::run_compute(spv60, in60, N, N);
    uint32_t bad60=0; for(uint32_t i=0;i<N&&got60.size()==N;i++) if(std::fabs(got60[i]-exp60[i])>1e-3f) bad60++;
    printf("  kernel60 mismatches=%u (out[10]=%g expect=%g)\n", bad60, got60.size()==N?got60[10]:-1, exp60[10]);
    CHECK(got60.size()==N && bad60==0, "recompiled kernel 60 (SDWA |src0|: |a0|*a1) correct");

    // Kernel 61: structured forward s_cbranch_scc0 (uniform if) — NEW control-flow feature.
    //   s0=bits(a); s1=bits(b); SCC=(bits(a)<bits(b)); v2=a; if SCC: v2=a+b; out v2.
    // For positive floats bits(a)<bits(b) == a<b, so v2 = (a<b) ? a+b : a. Exercises OpSelectionMerge +
    // OpBranchConditional(scc) + the merge OpPhi for v2 (live-across register written in the block).
    const uint32_t code61[] = {
        0x7E000500u,   // v_readfirstlane_b32 s0, v0
        0x7E020501u,   // v_readfirstlane_b32 s1, v1
        0xBF0A0100u,   // s_cmp_lt_u32 s0, s1     -> SCC
        0x7E040300u,   // v_mov_b32 v2, v0        (else value)
        0xBF840001u,   // s_cbranch_scc0 +1       (skip the v_add when SCC==0)
        0x06040300u,   // v_add_f32 v2, v0, v1    (then: v2 = a+b)
        0xBF810000u,   // s_endpgm
    };
    std::vector<uint32_t> spv61 = recompile_valu(code61, sizeof(code61)/sizeof(code61[0]), 2, /*out_vgpr*/2);
    CHECK(!spv61.empty(), "recompiled kernel 61 (forward s_cbranch_scc0 uniform if) -> SPIR-V");
    std::vector<float> in61(N*2), exp61(N);
    for (uint32_t i=0;i<N;i++){ float a=(float)i*0.1f+0.5f, b=6.0f; in61[i*2]=a; in61[i*2+1]=b; exp61[i]=(a<b)?(a+b):a; }
    std::vector<float> got61 = prosper::test::run_compute(spv61, in61, N, N);
    uint32_t bad61=0; for(uint32_t i=0;i<N&&got61.size()==N;i++) if(std::fabs(got61[i]-exp61[i])>1e-3f) bad61++;
    printf("  kernel61 mismatches=%u (out[10]=%g expect=%g out[100]=%g expect=%g)\n",
           bad61, got61.size()==N?got61[10]:-1, exp61[10], got61.size()==N?got61[100]:-1, exp61[100]);
    CHECK(got61.size()==N && bad61==0, "recompiled kernel 61 (structured scc0 if: (a<b)?a+b:a) correct");

    // Kernel 62: v_add_co_ci_u32 (VOP3B 0x128, add-with-carry). Convert inputs to uint, add with carry-in
    // = VCC (starts 0), convert back: v2 = (uint)a + (uint)b. Exercises the VOP3B sdst/carry decode + emit.
    const uint32_t code62[] = {
        0x7E000F00u,              // v_cvt_u32_f32 v0, v0
        0x7E020F01u,              // v_cvt_u32_f32 v1, v1
        0xD5286A02u, 0x01AA0300u, // v_add_co_ci_u32_e64 v2, vcc_lo, v0, v1, vcc_lo
        0x7E040D02u,              // v_cvt_f32_u32 v2, v2
        0xBF810000u,              // s_endpgm
    };
    std::vector<uint32_t> spv62 = recompile_valu(code62, sizeof(code62)/sizeof(code62[0]), 2, /*out_vgpr*/2);
    CHECK(!spv62.empty(), "recompiled kernel 62 (v_add_co_ci_u32 add-with-carry) -> SPIR-V");
    std::vector<float> in62(N*2), exp62(N);
    for (uint32_t i=0;i<N;i++){ float a=(float)i, b=(float)(i*2); in62[i*2]=a; in62[i*2+1]=b;
        exp62[i]=(float)((uint32_t)a + (uint32_t)b); }
    std::vector<float> got62 = prosper::test::run_compute(spv62, in62, N, N);
    uint32_t bad62=0; for(uint32_t i=0;i<N&&got62.size()==N;i++) if(std::fabs(got62[i]-exp62[i])>1e-3f) bad62++;
    printf("  kernel62 mismatches=%u (out[30]=%g expect=%g)\n", bad62, got62.size()==N?got62[30]:-1, exp62[30]);
    CHECK(got62.size()==N && bad62==0, "recompiled kernel 62 (v_add_co_ci_u32: (uint)a+(uint)b) correct");

    // Kernel 63: v_mbcnt_lo/hi (cross-lane wave-model via LDS). Full EXEC (all lanes active), so the
    // compaction index == this lane's index within its 64-wide workgroup: v1 = mbcnt_hi(exec_hi,
    // mbcnt_lo(exec_lo,0)) = localid. Verifies the LDS prefix-count + lo/hi split + LocalInvocationId.
    const uint32_t code63[] = {
        0xD7650001u, 0x0001007Eu,   // v_mbcnt_lo_u32_b32 v1, exec_lo, 0
        0xD7660001u, 0x0002027Fu,   // v_mbcnt_hi_u32_b32 v1, exec_hi, v1
        0x7E020D01u,                // v_cvt_f32_u32 v1, v1
        0xBF810000u,                // s_endpgm
    };
    std::vector<uint32_t> spv63 = recompile_valu(code63, sizeof(code63)/sizeof(code63[0]), 1, /*out_vgpr*/1);
    CHECK(!spv63.empty(), "recompiled kernel 63 (v_mbcnt wave-model) -> SPIR-V");
    std::vector<float> in63(N, 0.0f), exp63(N);
    for (uint32_t i=0;i<N;i++) exp63[i] = (float)(i % 64);   // localid within the 64-wide workgroup
    std::vector<float> got63 = prosper::test::run_compute(spv63, in63, N, N);
    uint32_t bad63=0; for(uint32_t i=0;i<N&&got63.size()==N;i++) if(std::fabs(got63[i]-exp63[i])>1e-3f) bad63++;
    printf("  kernel63 mismatches=%u (out[5]=%g exp=%g out[70]=%g exp=%g)\n", bad63,
           got63.size()==N?got63[5]:-1, exp63[5], got63.size()==N?got63[70]:-1, exp63[70]);
    CHECK(got63.size()==N && bad63==0, "recompiled kernel 63 (v_mbcnt full-exec = localid) correct");

    ComputeShaderConfig native_cfg63;
    native_cfg63.local_x = 8;
    native_cfg63.local_y = 8;
    native_cfg63.wave_size = 64;
    native_cfg63.native_subgroup_size = 64;
    const std::vector<uint32_t> native_spv63 = recompile_compute(
        code63, std::size(code63), nullptr, native_cfg63);
    CHECK(compute_shader_prefers_native_multiwave(code63, std::size(code63)) &&
              !compute_shader_prefers_native_multiwave(code62, std::size(code62)),
          "multi-wave policy recognizes scan-heavy shaders without title-specific identity");
    CHECK(!native_spv63.empty() && count_spirv_opcode(native_spv63, 349) == 2,
          "straight-line exact-wave MBCNT lowers both halves to subgroup scans");
    CHECK(count_spirv_opcode(native_spv63, 224) == 0,
          "straight-line exact-wave MBCNT emits no workgroup control barrier");

    // Kernel 64: v_mbcnt with DIVERGENT exec — the real compaction use. v_cmpx narrows exec to (a>thr);
    // then mbcnt gives each active lane its index among active lanes. Even lanes active (a=1>0.5), odd
    // inactive (a=0): active lane L -> L/2 (count of active lanes below); inactive lanes keep 0 (the
    // store is exec-predicated). Verifies the wave-model over a PARTIAL mask (not just full exec).
    const uint32_t code64[] = {
        0x7C280300u,                // v_cmpx_gt_f32 v0, v1   -> exec = (a > thr)
        0xD7650002u, 0x0001007Eu,   // v_mbcnt_lo_u32_b32 v2, exec_lo, 0
        0xD7660002u, 0x0002047Fu,   // v_mbcnt_hi_u32_b32 v2, exec_hi, v2
        0x7E040D02u,                // v_cvt_f32_u32 v2, v2
        0xBF810000u,                // s_endpgm
    };
    std::vector<uint32_t> spv64 = recompile_valu(code64, sizeof(code64)/sizeof(code64[0]), 2, /*out_vgpr*/2);
    CHECK(!spv64.empty(), "recompiled kernel 64 (v_mbcnt divergent-exec) -> SPIR-V");
    std::vector<float> in64(N*2), exp64(N);
    for (uint32_t i=0;i<N;i++){ uint32_t lane=i%64; float a=(lane%2==0)?1.0f:0.0f; in64[i*2]=a; in64[i*2+1]=0.5f;
        exp64[i] = (lane%2==0) ? (float)(lane/2) : 0.0f; }
    std::vector<float> got64 = prosper::test::run_compute(spv64, in64, N, N);
    uint32_t bad64=0; for(uint32_t i=0;i<N&&got64.size()==N;i++) if(std::fabs(got64[i]-exp64[i])>1e-3f) bad64++;
    printf("  kernel64 mismatches=%u (out[4]=%g exp=%g out[6]=%g exp=%g out[7]=%g exp=%g)\n", bad64,
           got64.size()==N?got64[4]:-1, exp64[4], got64.size()==N?got64[6]:-1, exp64[6], got64.size()==N?got64[7]:-1, exp64[7]);
    CHECK(got64.size()==N && bad64==0, "recompiled kernel 64 (v_mbcnt divergent: active lane L -> L/2) correct");

    // Kernel 65: v_mbcnt with src0 = inline -1 (all-ones mask) — the "get my lane id" idiom (shader 037).
    // mbcnt_lo(-1,0) + mbcnt_hi(-1,·) counts ALL lanes below => v1 = localid, independent of EXEC.
    const uint32_t code65[] = {
        0xD7650001u, 0x000100C1u,   // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD7660001u, 0x000202C1u,   // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x7E020D01u,                // v_cvt_f32_u32 v1, v1
        0xBF810000u,                // s_endpgm
    };
    std::vector<uint32_t> spv65 = recompile_valu(code65, sizeof(code65)/sizeof(code65[0]), 1, /*out_vgpr*/1);
    CHECK(!spv65.empty(), "recompiled kernel 65 (v_mbcnt src0=-1) -> SPIR-V");
    std::vector<float> in65(N, 0.0f), exp65(N);
    for (uint32_t i=0;i<N;i++) exp65[i] = (float)(i % 64);
    std::vector<float> got65 = prosper::test::run_compute(spv65, in65, N, N);
    uint32_t bad65=0; for(uint32_t i=0;i<N&&got65.size()==N;i++) if(std::fabs(got65[i]-exp65[i])>1e-3f) bad65++;
    printf("  kernel65 mismatches=%u (out[9]=%g exp=%g)\n", bad65, got65.size()==N?got65[9]:-1, exp65[9]);
    CHECK(got65.size()==N && bad65==0, "recompiled kernel 65 (v_mbcnt -1 = localid) correct");

    // Kernel 66: RAW MUBUF SRSRC RESOLUTION (#91). Same code as kernel 21 (buffer_load_dword v0, v0
    // offen, SRSRC s[8:11]) but WITH a resource table mapping s[8:11] -> binding 3. The load must
    // route to binding 3 (cbuf1), not the old hardcoded binding 2 — binding 2 (cbuf0) is bound with
    // DECOY values, so any binding-2 read fails the value check.
    const uint32_t code66[] = {
        0x7e000f00u, 0x34000082u, 0xe0301000u, 0x80020000u, 0x7e000d00u, 0xbf810000u,
    };
    ShaderResourceTable rt66;
    { ShaderResource rb{}; rb.cls = ResourceClass::ConstantBuffer; rb.format = DataFormat::Float32;
      rb.num_components = 1; rb.binding = 3; rb.stride = 0; rb.sgpr_base = 8; rt66.resources.push_back(rb); }
    std::vector<uint32_t> spv66 = recompile_valu(code66, sizeof(code66)/sizeof(code66[0]), 1, 0, &rt66);
    CHECK(!spv66.empty(), "recompiled kernel 66 (raw buffer_load_dword, SRSRC via table -> binding 3) -> SPIR-V");
    std::vector<float> in66(N); std::vector<uint32_t> decoy66(N, 0xDEADu), buf66(N);
    for (uint32_t i = 0; i < N; i++) { in66[i] = (float)i; buf66[i] = 500u + i; }
    std::vector<float> got66 = prosper::test::run_compute(spv66, in66, N, N, /*binding2=decoy*/decoy66, /*binding3*/buf66);
    uint32_t bad66 = 0; for (uint32_t i=0;i<N&&got66.size()==N;i++) if (std::fabs(got66[i]-(float)(500u+i))>1e-3f) bad66++;
    printf("  kernel66 mismatches=%u (out[7]=%g expect=507)\n", bad66, got66.size()==N?got66[7]:-1);
    CHECK(got66.size()==N && bad66==0, "recompiled kernel 66 (raw load resolves SRSRC -> binding 3, not binding 2) correct");

    // Kernel 66b: MUBUF idxen+offen — BOTH VADDR terms must apply (#148). v1=(uint)gid (index), v2=4
    // (per-lane byte offset), buffer_load_dword v3, v[1:2], s[8:11] idxen offen, stride 8 -> addr =
    // gid*8 + 4 => dword index 2*gid+1 => buf[2*gid+1]. The old code applied only the index term and
    // silently dropped v2, reading buf[2*gid] instead. cvt in/out (the shell's inputs are float bits).
    // Encodings llvm-mc gfx1030 verified.
    const uint32_t code66b[] = {
        0x7e020f00u, 0x7e040284u, 0xe0303000u, 0x80020301u, 0x7e060d03u, 0xbf810000u,
    };
    ShaderResourceTable rt66b;
    { ShaderResource rb{}; rb.cls = ResourceClass::ConstantBuffer; rb.format = DataFormat::Float32;
      rb.num_components = 1; rb.binding = 3; rb.stride = 8; rb.sgpr_base = 8; rt66b.resources.push_back(rb); }
    std::vector<uint32_t> spv66b = recompile_valu(code66b, sizeof(code66b)/sizeof(code66b[0]), 1, /*out_vgpr*/3, &rt66b);
    CHECK(!spv66b.empty(), "recompiled kernel 66b (MUBUF idxen+offen) -> SPIR-V");
    std::vector<float> in66b(N); std::vector<uint32_t> decoy66b(2*N, 0xDEADu), buf66b(2*N);
    for (uint32_t i = 0; i < N; i++)   in66b[i] = (float)i;
    for (uint32_t i = 0; i < 2*N; i++) buf66b[i] = 500u + i;
    std::vector<float> got66b = prosper::test::run_compute(spv66b, in66b, N, N, decoy66b, buf66b);
    uint32_t bad66b = 0;
    for (uint32_t i=0;i<N&&got66b.size()==N;i++) if (std::fabs(got66b[i]-(float)(500u + 2u*i + 1u))>1e-3f) bad66b++;
    printf("  kernel66b mismatches=%u (out[3]=%g expect=%u)\n", bad66b,
           got66b.size()==N?got66b[3]:-1, 500u + 2u*3u + 1u);
    CHECK(got66b.size()==N && bad66b==0, "kernel 66b (idxen+offen): addr = idx*stride + byteoffset (both applied)");

    // The same two-term address through an exact pc-keyed graphics VertexBuffer in Shader mode.
    // Dynamic fetch discovery publishes these resources too, but unlike ABI Vertex/Instance mode
    // their base is unshifted and the shader's complete VADDR must survive. In particular, v2 is the
    // offen byte term; dropping it selects the even dword instead of the odd one in every record.
    const uint32_t code66b_shader[] = {
        0x7e020f00u, 0x7e040284u, 0xe0003000u, 0x80020301u, 0xbf810000u,
    };
    ShaderResourceTable rt66b_shader;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32;
      vb.num_components = 1; vb.binding = 3; vb.stride = 8; vb.sgpr_base = 8; vb.fetch_pc = 2;
      vb.fetch_index_mode = VertexFetchIndexMode::Shader;
      rt66b_shader.resources.push_back(vb); }
    std::vector<uint32_t> spv66b_shader = recompile_valu(
        code66b_shader, std::size(code66b_shader), 1, 3, &rt66b_shader);
    std::vector<float> float66b_shader(2 * N), got66b_shader;
    for (uint32_t i = 0; i < N; ++i) {
        float66b_shader[2 * i] = -1000.0f - (float)i;
        float66b_shader[2 * i + 1] = 700.0f + (float)i;
    }
    std::vector<uint32_t> buf66b_shader(2 * N);
    std::memcpy(buf66b_shader.data(), float66b_shader.data(), buf66b_shader.size() * sizeof(uint32_t));
    got66b_shader = prosper::test::run_compute(
        spv66b_shader, in66b, N, N, decoy66b, buf66b_shader);
    uint32_t bad66b_shader = 0;
    for (uint32_t i = 0; i < N && got66b_shader.size() == N; ++i)
        if (got66b_shader[i] != 700.0f + (float)i) ++bad66b_shader;
    CHECK(!spv66b_shader.empty() && got66b_shader.size() == N && bad66b_shader == 0,
          "Shader-mode pc fetch preserves idxen+offen's nonzero second VADDR VGPR");

    // Raw byte/short MUBUF loads use a per-lane byte address and zero/sign-extend into one VGPR.
    // Exercise every in-dword position, including a 16-bit load beginning at byte 3 and crossing
    // into the next dword. Encodings round-tripped with llvm-mc gfx1030.
    ShaderResourceTable rt66c;
    { ShaderResource rb{}; rb.cls = ResourceClass::ConstantBuffer; rb.format = DataFormat::Uint32;
      rb.num_components = 1; rb.binding = 3; rb.sgpr_base = 8; rt66c.resources.push_back(rb); }
    std::vector<uint8_t> raw66c(N + 1);
    for (uint32_t i = 0; i <= N; ++i) raw66c[i] = static_cast<uint8_t>(i * 73u + 0x51u);
    std::vector<uint32_t> bytes66c((raw66c.size() + 3) / 4, 0u), decoy66c(bytes66c.size(), 0u);
    for (uint32_t i = 0; i < raw66c.size(); ++i)
        bytes66c[i / 4] |= static_cast<uint32_t>(raw66c[i]) << ((i & 3u) * 8u);
    auto raw_subword_ok = [&](uint32_t mubuf_word, uint32_t bits, bool is_signed) {
        const uint32_t code66c[] = {
            0x7e000f00u, mubuf_word, 0x80020000u,
            is_signed ? 0x7e000b00u : 0x7e000d00u, 0xbf810000u,
        };
        std::vector<uint32_t> spv66c = recompile_valu(code66c, std::size(code66c), 1, 0, &rt66c);
        if (spv66c.empty()) return false;
        std::vector<float> got66c = prosper::test::run_compute(
            spv66c, in66, N, N, decoy66c, bytes66c);
        uint32_t bad66c = 0;
        for (uint32_t i = 0; i < N && got66c.size() == N; ++i) {
            const uint32_t raw = bits == 8 ? raw66c[i]
                : static_cast<uint32_t>(raw66c[i]) |
                  (static_cast<uint32_t>(raw66c[i + 1]) << 8u);
            const float expected = is_signed
                ? static_cast<float>(bits == 8 ? static_cast<int8_t>(raw)
                                               : static_cast<int16_t>(raw))
                : static_cast<float>(raw);
            if (got66c[i] != expected) ++bad66c;
        }
        return got66c.size() == N && bad66c == 0;
    };
    CHECK(raw_subword_ok(0xe0201000u, 8, false),
          "buffer_load_ubyte zero-extends every byte position");
    CHECK(raw_subword_ok(0xe0241000u, 8, true),
          "buffer_load_sbyte sign-extends every byte position");
    CHECK(raw_subword_ok(0xe0281000u, 16, false),
          "buffer_load_ushort zero-extends aligned, unaligned, and cross-dword values");
    CHECK(raw_subword_ok(0xe02c1000u, 16, true),
          "buffer_load_sshort sign-extends aligned, unaligned, and cross-dword values");

    // Kernel 67: RAW MUBUF UNRESOLVABLE SRSRC -> REJECT (#91). Same code, but the table's only
    // resource lives at sgpr_base 4 — SRSRC s[8:11] resolves to nothing. With a table present the
    // recompiler must reject (empty SPIR-V), never silently fall back to binding 2.
    ShaderResourceTable rt67;
    { ShaderResource rb{}; rb.cls = ResourceClass::ConstantBuffer; rb.format = DataFormat::Float32;
      rb.num_components = 1; rb.binding = 3; rb.stride = 0; rb.sgpr_base = 4; rt67.resources.push_back(rb); }
    std::vector<uint32_t> spv67 = recompile_valu(code66, sizeof(code66)/sizeof(code66[0]), 1, 0, &rt67);
    CHECK(spv67.empty(), "kernel 67 (raw MUBUF, table present, SRSRC unresolvable) REJECTED (no binding-2 fallback)");

    // Kernel 68: RAW MUBUF STORE via the table (#91 store side). v2=(uint)gid; v3=2*float(gid);
    // buffer_store_dword v3, v2, s[8:11] idxen (op 0x1C) -> table maps s[8:11] to binding 3, stride 4
    // => buf[gid] = bits(2*gid). The old code stored these into binding 2, corrupting the wrong buffer.
    const uint32_t code68[] = { 0x7e040f00u, 0x06060100u, 0xe0702000u, 0x80020302u, 0xbf810000u };
    ShaderResourceTable rt68;
    { ShaderResource rb{}; rb.cls = ResourceClass::ConstantBuffer; rb.format = DataFormat::Float32;
      rb.num_components = 1; rb.binding = 3; rb.stride = 4; rb.sgpr_base = 8; rt68.resources.push_back(rb); }
    std::vector<uint32_t> spv68 = recompile_valu(code68, sizeof(code68)/sizeof(code68[0]), 1, 0, &rt68);
    CHECK(!spv68.empty(), "recompiled kernel 68 (raw buffer_store_dword, SRSRC via table -> binding 3) -> SPIR-V");
    std::vector<float> in68(N); for (uint32_t i = 0; i < N; i++) in68[i] = (float)i;
    std::vector<uint32_t> stored68(N, 0), stored68_out;
    prosper::test::run_compute(spv68, in68, N, N, /*binding2*/{}, /*binding3=store target*/stored68, &stored68_out);
    uint32_t bad68 = 0;
    for (uint32_t i = 0; i < N && stored68_out.size() == N; i++) {
        float f; std::memcpy(&f, &stored68_out[i], 4);
        if (std::fabs(f - 2.0f*(float)i) > 1e-3f) bad68++;
    }
    { float f6 = 0; if (stored68_out.size()==N) std::memcpy(&f6, &stored68_out[6], 4);
      printf("  kernel68 mismatches=%u (buf[6]=%g expect=12)\n", bad68, f6); }
    CHECK(stored68_out.size() == N && bad68 == 0, "recompiled kernel 68 (raw store routes to binding 3 via SRSRC) correct");

    // Kernel N: v_nop (VOP1 op 0x00) between real ALU ops must be a transparent no-op. This was the
    // #121 blocker — PPSA02664's vertex shaders contain v_nop (a common scheduling/hazard filler), and
    // the recompiler rejected it, failing the whole shader and skipping ~182 draws/frame (black screen).
    //   v_add_f32 v0,v0,v1 | v_nop | v_mul_f32 v0,v0,v2 | s_endpgm  =>  out = (a0+a1)*a2
    const uint32_t codeN[] = { 0x06000300u, 0x7E000000u, 0x10000500u, 0xBF810000u };
    std::vector<uint32_t> spvN = recompile_valu(codeN, sizeof(codeN)/sizeof(codeN[0]), 3, 0);
    CHECK(!spvN.empty(), "recompiled kernel with v_nop -> SPIR-V (v_nop no longer fails the shader)");
    std::vector<float> inN(N * 3), expN(N);
    for (uint32_t i = 0; i < N; i++) {
        float a0 = (float)i * 0.2f - 8.0f, a1 = (float)i * 0.05f + 2.0f, a2 = 1.5f;
        inN[i*3+0]=a0; inN[i*3+1]=a1; inN[i*3+2]=a2; expN[i] = (a0 + a1) * a2;
    }
    std::vector<float> gotN = prosper::test::run_compute(spvN, inN, N, N);
    uint32_t badN = 0; for (uint32_t i=0;i<N&&gotN.size()==N;i++) if (std::fabs(gotN[i]-expN[i])>1e-3f) badN++;
    printf("  kernelN(v_nop) mismatches=%u (out[33]=%g expect=%g)\n", badN, gotN.size()==N?gotN[33]:-1, expN[33]);
    CHECK(gotN.size()==N && badN==0, "v_nop is transparent: out = (a0+a1)*a2 computed correctly");

    // Kernels J1/J2/J3: forward-branch targets PAST the first s_endpgm (#129). A branch past
    // s_endpgm is a benign early-out ONLY if execution at the target immediately terminates
    // (s_nop* then s_endpgm). Real straight-line code at the target is a second terminating arm;
    // it must execute as a structured if/else rather than being silently discarded by a clamp.
    // Shape: s0=7; s1=7|8; s_cmp_eq_u32; s_cbranch_scc0 +3 (-> pc7); v0=42.0; s_endpgm; <target...>
    //   J1: target is s_endpgm            -> accept; SCC=1 runs the block (42), SCC=0 skips (input)
    //   J2: target is REAL code (v0=9.0)  -> accept; SCC=1 selects 42, SCC=0 selects 9
    //   J3: target is s_nop; s_endpgm     -> accept (nop padding before the end is still an early-out)
    // Encodings llvm-mc gfx1030 verified.
    const uint32_t codeJ1t[] = { 0xbe800387u, 0xbe810387u, 0xbf060100u, 0xbf840003u,
                                 0x7e0002ffu, 0x42280000u, 0xbf810000u, 0xbf810000u };
    const uint32_t codeJ1f[] = { 0xbe800387u, 0xbe810388u, 0xbf060100u, 0xbf840003u,
                                 0x7e0002ffu, 0x42280000u, 0xbf810000u, 0xbf810000u };
    const uint32_t codeJ2t[] = { 0xbe800387u, 0xbe810387u, 0xbf060100u, 0xbf840003u,
                                 0x7e0002ffu, 0x42280000u, 0xbf810000u,
                                 0x7e0002ffu, 0x41100000u, 0xbf810000u };
    const uint32_t codeJ2e[] = { 0xbe800387u, 0xbe810388u, 0xbf060100u, 0xbf840003u,
                                 0x7e0002ffu, 0x42280000u, 0xbf810000u,
                                 0x7e0002ffu, 0x41100000u, 0xbf810000u };
    const uint32_t codeJ3[]  = { 0xbe800387u, 0xbe810387u, 0xbf060100u, 0xbf840003u,
                                 0x7e0002ffu, 0x42280000u, 0xbf810000u, 0xbf800000u, 0xbf810000u };
    std::vector<uint32_t> spvJ1t = recompile_valu(codeJ1t, sizeof(codeJ1t)/sizeof(codeJ1t[0]), 1, 0);
    std::vector<uint32_t> spvJ1f = recompile_valu(codeJ1f, sizeof(codeJ1f)/sizeof(codeJ1f[0]), 1, 0);
    std::vector<uint32_t> spvJ2t = recompile_valu(codeJ2t, sizeof(codeJ2t)/sizeof(codeJ2t[0]), 1, 0);
    std::vector<uint32_t> spvJ2e = recompile_valu(codeJ2e, sizeof(codeJ2e)/sizeof(codeJ2e[0]), 1, 0);
    std::vector<uint32_t> spvJ3  = recompile_valu(codeJ3,  sizeof(codeJ3)/sizeof(codeJ3[0]),  1, 0);
    CHECK(!spvJ1t.empty() && !spvJ1f.empty(), "kernel J1 (early-out branch to a trailing s_endpgm) -> SPIR-V");
    CHECK(!spvJ2t.empty() && !spvJ2e.empty(),
          "kernel J2 (branch past s_endpgm) emits both terminating if/else arms");
    CHECK(rdna2_recompile_code_span(codeJ2t, sizeof(codeJ2t)/sizeof(codeJ2t[0])) ==
              sizeof(codeJ2t)/sizeof(codeJ2t[0]),
          "kernel J2 code span includes the reachable post-endpgm else arm");
    CHECK(!spvJ3.empty(), "kernel J3 (early-out through s_nop padding) -> SPIR-V");
    std::vector<float> inJ(N);
    for (uint32_t i = 0; i < N; i++) inJ[i] = (float)i * 0.25f;
    std::vector<float> gotJt = prosper::test::run_compute(spvJ1t, inJ, N, N);
    std::vector<float> gotJf = prosper::test::run_compute(spvJ1f, inJ, N, N);
    std::vector<float> gotJ2t = prosper::test::run_compute(spvJ2t, inJ, N, N);
    std::vector<float> gotJ2e = prosper::test::run_compute(spvJ2e, inJ, N, N);
    uint32_t badJ = 0, badJ2 = 0;
    for (uint32_t i = 0; i < N && gotJt.size() == N; i++) if (std::fabs(gotJt[i] - 42.0f) > 1e-3f) badJ++;
    for (uint32_t i = 0; i < N && gotJf.size() == N; i++) if (std::fabs(gotJf[i] - inJ[i]) > 1e-3f) badJ++;
    for (uint32_t i = 0; i < N && gotJ2t.size() == N; i++) if (std::fabs(gotJ2t[i] - 42.0f) > 1e-3f) badJ2++;
    for (uint32_t i = 0; i < N && gotJ2e.size() == N; i++) if (std::fabs(gotJ2e[i] - 9.0f) > 1e-3f) badJ2++;
    printf("  kernelJ mismatches=%u (scc1->%g expect 42, scc0->%g expect %g)\n", badJ,
           gotJt.size()==N?gotJt[8]:-1, gotJf.size()==N?gotJf[8]:-1, inJ[8]);
    CHECK(gotJt.size()==N && gotJf.size()==N && badJ==0,
          "kernel J1 early-out: block runs on SCC=1 (42), skipped on SCC=0 (input)");
    CHECK(gotJ2t.size()==N && gotJ2e.size()==N && badJ2==0,
          "kernel J2 terminating if/else: SCC=1 selects 42 and SCC=0 selects 9");

    // Kernels S1/S2: v_cvt_u32_f32 / v_cvt_i32_f32 SATURATION (#135). RDNA2 saturates the
    // float->int converts (NaN -> 0, negative -> 0 for u32, out-of-range clamps to the type
    // min/max); a bare OpConvertFToU/S is undefined out of range, so drivers returned garbage.
    // Round-trip through v_cvt_f32_u32 / v_cvt_f32_i32 so the output buffer holds a comparable
    // float. Encodings llvm-mc gfx1030 round-trip verified.
    //   S1: v_cvt_u32_f32 v0,v0 | v_cvt_f32_u32 v0,v0 | s_endpgm
    //   S2: v_cvt_i32_f32 v0,v0 | v_cvt_f32_i32 v0,v0 | s_endpgm
    const uint32_t codeS1[] = { 0x7E000F00u, 0x7E000D00u, 0xBF810000u };
    const uint32_t codeS2[] = { 0x7E001100u, 0x7E000B00u, 0xBF810000u };
    std::vector<uint32_t> spvS1 = recompile_valu(codeS1, sizeof(codeS1)/sizeof(codeS1[0]), 1, 0);
    std::vector<uint32_t> spvS2 = recompile_valu(codeS2, sizeof(codeS2)/sizeof(codeS2[0]), 1, 0);
    CHECK(!spvS1.empty(), "recompiled kernel S1 (v_cvt_u32_f32 saturating) -> SPIR-V");
    CHECK(!spvS2.empty(), "recompiled kernel S2 (v_cvt_i32_f32 saturating) -> SPIR-V");
    CHECK(count_spirv_opcode(spvS2, 110) == 0 && count_spirv_opcode(spvS2, 109) != 0,
          "v_cvt_i32_f32 avoids Metal's undefined signed float-conversion path");
    const float satIn[] = { 3.7f, 0.0f, -0.5f, -5.5f, 255.9f, -1e10f, 1e10f, 4294967040.0f,
                            2147483520.0f, -2147483520.0f, std::nanf(""), INFINITY, -INFINITY };
    const uint32_t NSAT = sizeof(satIn)/sizeof(satIn[0]);
    auto sat_u32 = [](float x) -> uint32_t {
        if (std::isnan(x)) return 0u;
        if (x <= 0.0f) return 0u;
        if (x >= 4294967296.0f) return 0xFFFFFFFFu;
        return (uint32_t)x;
    };
    auto sat_i32 = [](float x) -> int32_t {
        if (std::isnan(x)) return 0;
        if (x <= -2147483648.0f) return INT32_MIN;
        if (x >= 2147483648.0f) return INT32_MAX;
        return (int32_t)x;
    };
    std::vector<float> inS(N), expS1(N), expS2(N);
    for (uint32_t i = 0; i < N; i++) {
        float a = satIn[i % NSAT];
        inS[i] = a; expS1[i] = (float)sat_u32(a); expS2[i] = (float)sat_i32(a);
    }
    std::vector<float> gotS1 = prosper::test::run_compute(spvS1, inS, N, N);
    std::vector<float> gotS2 = prosper::test::run_compute(spvS2, inS, N, N);
    uint32_t badS1 = 0, badS2 = 0;
    for (uint32_t i = 0; i < N && gotS1.size() == N; i++)
        if (std::fabs(gotS1[i] - expS1[i]) > 1e-3f + 1e-6f*std::fabs(expS1[i])) badS1++;
    for (uint32_t i = 0; i < N && gotS2.size() == N; i++)
        if (std::fabs(gotS2[i] - expS2[i]) > 1e-3f + 1e-6f*std::fabs(expS2[i])) badS2++;
    printf("  kernelS1(u32 sat) mismatches=%u (nan->%g, -5.5->%g, 1e10->%g expect 0,0,%g)\n", badS1,
           gotS1.size()==N?gotS1[10]:-1, gotS1.size()==N?gotS1[3]:-1, gotS1.size()==N?gotS1[6]:-1, expS1[6]);
    printf("  kernelS2(i32 sat) mismatches=%u (nan->%g, -1e10->%g, 1e10->%g expect 0,%g,%g)\n", badS2,
           gotS2.size()==N?gotS2[10]:-1, gotS2.size()==N?gotS2[5]:-1, gotS2.size()==N?gotS2[6]:-1, expS2[5], expS2[6]);
    CHECK(gotS1.size()==N && badS1==0, "v_cvt_u32_f32 saturates (NaN/neg -> 0, >=2^32 -> UINT_MAX)");
    CHECK(gotS2.size()==N && badS2==0, "v_cvt_i32_f32 saturates (NaN -> 0, clamps to INT_MIN/INT_MAX)");

    // Kernels X1..X3: SPECIAL operands read as ALU DATA (#134). VCC/EXEC live as per-lane bools in
    // the per-invocation model (their 32-bit wave-mask value doesn't exist) and M0 isn't modeled —
    // such reads previously computed with a silent 0; they must now REJECT. SGPR_NULL (field 125)
    // is the one Special whose data value IS 0, and must keep recompiling. llvm-mc gfx1030 verified.
    const uint32_t codeX1[] = { 0x4A02007Eu, 0xBF810000u };   // v_add_nc_u32 v1, exec_lo, v0
    const uint32_t codeX2[] = { 0x4A02007Cu, 0xBF810000u };   // v_add_nc_u32 v1, m0, v0
    const uint32_t codeX3[] = { 0x4A02007Du, 0xBF810000u };   // v_add_nc_u32 v1, null, v0
    CHECK(recompile_valu(codeX1, 2, 1, 1).empty(), "kernel X1 (exec_lo read as ALU data) is REJECTED");
    CHECK(recompile_valu(codeX2, 2, 1, 1).empty(), "kernel X2 (m0 read as ALU data) is REJECTED");
    std::vector<uint32_t> spvX3 = recompile_valu(codeX3, 2, 1, /*out_vgpr*/1);
    CHECK(!spvX3.empty(), "kernel X3 (null read as ALU data) still recompiles (null == 0)");
    std::vector<float> inX(N);
    for (uint32_t i = 0; i < N; i++) inX[i] = (float)i;
    std::vector<float> gotX3 = prosper::test::run_compute(spvX3, inX, N, N);
    uint32_t badX = 0;
    for (uint32_t i = 0; i < N && gotX3.size() == N; i++) if (gotX3[i] != inX[i]) badX++;
    printf("  kernelX3(null+v0) mismatches=%u (out[9]=%g expect %g)\n", badX,
           gotX3.size()==N?gotX3[9]:-1, inX[9]);
    CHECK(gotX3.size()==N && badX==0, "kernel X3 computes null(0) + a0 = a0");

    // Kernel X4: VCC as plain scalar SCRATCH (the NGG-preamble pattern: a scalar write to vcc_lo
    // then a data read of it). s_mov_b32 vcc_lo, 5 ; v_add_nc_u32 v1, vcc_lo, v0 => out = a0 + 5.
    // The tracked write must be read back (not 0, not rejected).
    const uint32_t codeX4[] = { 0xBEEA0385u, 0x4A02006Au, 0xBF810000u };
    std::vector<uint32_t> spvX4 = recompile_valu(codeX4, 3, 1, /*out_vgpr*/1);
    CHECK(!spvX4.empty(), "kernel X4 (scalar-written vcc_lo read as data) recompiles");
    std::vector<float> gotX4 = prosper::test::run_compute(spvX4, inX, N, N);
    uint32_t badX4 = 0;   // integer add on raw VGPR bits: out bits = bits(a0) + 5
    for (uint32_t i = 0; i < N && gotX4.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotX4[i], 4);
        if (gb != bits_of(inX[i]) + 5u) badX4++;
    }
    printf("  kernelX4(vcc scratch) mismatches=%u\n", badX4);
    CHECK(gotX4.size()==N && badX4==0, "kernel X4 computes bits(a0) + vcc_lo(5) via the tracked scalar write");

    // Kernel O: VOP2 v_mul_f32 in SDWA form with OMOD ×2 output modifier. This was a #121 blocker —
    // PPSA02664's pixel shaders emit `v_mul_f32 v,v,v mul:2` (full-DWORD selects, omod=×2), which the
    // decoder rejected as an unhandled modifier, failing the whole PS. Now the ×2 is applied.
    //   w0=0x100002f9 (op 0x08, dst v0, vsrc1 v1, src0=0xF9=SDWA), w1=0x06064600 (dst/s0/s1 sel=DWORD, omod=1)
    //   => out = (a0 * a1) * 2
    const uint32_t codeO[] = { 0x100002f9u, 0x06064600u, 0xBF810000u };
    std::vector<uint32_t> spvO = recompile_valu(codeO, sizeof(codeO)/sizeof(codeO[0]), 2, 0);
    CHECK(!spvO.empty(), "recompiled v_mul_f32 SDWA omod:2 -> SPIR-V (no longer rejected)");
    std::vector<float> inO(N * 2), expO(N);
    for (uint32_t i = 0; i < N; i++) {
        float a0 = (float)i * 0.1f - 3.0f, a1 = 1.25f;
        inO[i*2+0] = a0; inO[i*2+1] = a1; expO[i] = (a0 * a1) * 2.0f;
    }
    std::vector<float> gotO = prosper::test::run_compute(spvO, inO, N, N);
    uint32_t badO = 0; for (uint32_t i=0;i<N&&gotO.size()==N;i++) if (std::fabs(gotO[i]-expO[i])>1e-3f) badO++;
    printf("  kernelO(omod:2) mismatches=%u (out[20]=%g expect=%g)\n", badO, gotO.size()==N?gotO[20]:-1, expO[20]);
    CHECK(gotO.size()==N && badO==0, "v_mul_f32 SDWA omod:2 doubles the product: out = (a0*a1)*2");

    // ---- #273 kernels: ops surfaced by DOLL's failing shaders (all llvm-mc gfx1010 verified) ----

    // Kernel T1: v_sin_f32 / v_cos_f32 — RDNA trig input is in REVOLUTIONS (sin/cos of 2π·src).
    // out = sin(2π·a0) + cos(2π·a1).
    const uint32_t codeT1[] = { 0x7e006b00u, 0x7e026d01u, 0x06000300u, 0xBF810000u };
    std::vector<uint32_t> spvT1 = recompile_valu(codeT1, sizeof(codeT1)/4, 2, 0);
    CHECK(!spvT1.empty(), "recompiled T1 (v_sin/v_cos) -> SPIR-V");
    std::vector<float> inT1(N * 2), expT1(N);
    for (uint32_t i = 0; i < N; i++) {
        float a0 = (float)i * 0.01f - 0.5f, a1 = (float)i * 0.02f;
        inT1[i*2+0] = a0; inT1[i*2+1] = a1;
        expT1[i] = std::sin(6.2831853f * a0) + std::cos(6.2831853f * a1);
    }
    std::vector<float> gotT1 = prosper::test::run_compute(spvT1, inT1, N, N);
    uint32_t badT1 = 0; for (uint32_t i=0;i<N&&gotT1.size()==N;i++) if (std::fabs(gotT1[i]-expT1[i])>2e-3f) badT1++;
    printf("  T1(sin/cos) mismatches=%u (out[30]=%g expect=%g)\n", badT1, gotT1.size()==N?gotT1[30]:-1, expT1[30]);
    CHECK(gotT1.size()==N && badT1==0, "T1: v_sin/v_cos compute sin/cos(2π·x)");

    // Kernel T2: v_subrev_f32 (e32) — out = a1 - a0.
    const uint32_t codeT2[] = { 0x0a000300u, 0xBF810000u };
    std::vector<uint32_t> spvT2 = recompile_valu(codeT2, sizeof(codeT2)/4, 2, 0);
    CHECK(!spvT2.empty(), "recompiled T2 (v_subrev_f32) -> SPIR-V");
    std::vector<float> inT2(N * 2), expT2(N);
    for (uint32_t i = 0; i < N; i++) { float a0=(float)i*0.5f, a1=100.0f-(float)i;
        inT2[i*2+0]=a0; inT2[i*2+1]=a1; expT2[i]=a1-a0; }
    std::vector<float> gotT2 = prosper::test::run_compute(spvT2, inT2, N, N);
    uint32_t badT2 = 0; for (uint32_t i=0;i<N&&gotT2.size()==N;i++) if (std::fabs(gotT2[i]-expT2[i])>1e-3f) badT2++;
    CHECK(gotT2.size()==N && badT2==0, "T2: v_subrev_f32 computes src1 - src0");

    // Kernel T3: v_min3_f32 + v_max3_f32 — out = min3(a0,a1,a2) + max3(a0,a1,a2).
    const uint32_t codeT3[] = { 0xd5510003u, 0x040a0300u, 0xd5540004u, 0x040a0300u, 0x06000903u, 0xBF810000u };
    std::vector<uint32_t> spvT3 = recompile_valu(codeT3, sizeof(codeT3)/4, 3, 0);
    CHECK(!spvT3.empty(), "recompiled T3 (v_min3/v_max3_f32) -> SPIR-V");
    std::vector<float> inT3(N * 3), expT3(N);
    for (uint32_t i = 0; i < N; i++) {
        float a0=(float)i*0.3f-10.0f, a1=5.0f-(float)i*0.1f, a2=(float)(i%9);
        inT3[i*3+0]=a0; inT3[i*3+1]=a1; inT3[i*3+2]=a2;
        expT3[i] = std::fmin(std::fmin(a0,a1),a2) + std::fmax(std::fmax(a0,a1),a2);
    }
    std::vector<float> gotT3 = prosper::test::run_compute(spvT3, inT3, N, N);
    uint32_t badT3 = 0; for (uint32_t i=0;i<N&&gotT3.size()==N;i++) if (std::fabs(gotT3[i]-expT3[i])>1e-3f) badT3++;
    CHECK(gotT3.size()==N && badT3==0, "T3: v_min3/v_max3_f32 compute 3-way min+max");

    // Kernel T4: v_cvt_pk_u16_u32 — u=(uint)a0; out bits = min(u,0xFFFF) | min(u+1,0xFFFF)<<16.
    const uint32_t codeT4[] = { 0x7e000f00u, 0x4a020081u, 0xd76a0001u, 0x00020300u, 0xBF810000u };
    std::vector<uint32_t> spvT4 = recompile_valu(codeT4, sizeof(codeT4)/4, 1, /*out_vgpr*/1);
    CHECK(!spvT4.empty(), "recompiled T4 (v_cvt_pk_u16_u32) -> SPIR-V");
    std::vector<float> inT4(N);
    for (uint32_t i = 0; i < N; i++) inT4[i] = (float)(i * 3);
    std::vector<float> gotT4 = prosper::test::run_compute(spvT4, inT4, N, N);
    uint32_t badT4 = 0;
    for (uint32_t i = 0; i < N && gotT4.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotT4[i], 4);
        uint32_t u = i * 3;
        if (gb != ((u & 0xFFFFu) | ((u + 1u) << 16))) badT4++;
    }
    CHECK(gotT4.size()==N && badT4==0, "T4: v_cvt_pk_u16_u32 packs saturated u16 halves");

    // Kernel T5: s_min_u32 (SCC=s0<s1) + s_max_i32 (SCC=s0>s1) — out bits = bits(a0) + 3 + 5.
    const uint32_t codeT5[] = { 0x83818387u, 0x840285c2u, 0x4a020001u, 0x4a020202u, 0xBF810000u };
    std::vector<uint32_t> spvT5 = recompile_valu(codeT5, sizeof(codeT5)/4, 1, 1);
    CHECK(!spvT5.empty(), "recompiled T5 (s_min_u32/s_max_i32) -> SPIR-V");
    std::vector<float> gotT5 = prosper::test::run_compute(spvT5, inX, N, N);
    uint32_t badT5 = 0;
    for (uint32_t i = 0; i < N && gotT5.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotT5[i], 4);
        if (gb != bits_of(inX[i]) + 8u) badT5++;
    }
    CHECK(gotT5.size()==N && badT5==0, "T5: s_min_u32(7,3)=3 + s_max_i32(-2,5)=5");

    // Kernel T5b (#397 -> corrected by the #879 ISA audit): s_max SCC on a TIE. The RDNA2 ISA
    // pseudocode is STRICT in both directions (S_MAX_I32: "D.i = (S0.i > S1.i) ? S0.i : S1.i;
    // SCC = (S0.i > S1.i)", doc 70648 sec 12.1 op 8) — #397's non-strict `>=` quoted a line that
    // is not in the actual document. s_max_i32 s2,5,5 -> SCC=(5>5)=0; s_cselect_b32 s3 =
    // SCC ? 11 : 4 picks 4, then v1 = a0 + s3: out bits = bits(a0) + 4. (llvm-mc gfx1010.)
    const uint32_t codeT5b[] = { 0x84028585u, 0x8503848bu, 0x4a020003u, 0xBF810000u };
    std::vector<uint32_t> spvT5b = recompile_valu(codeT5b, sizeof(codeT5b)/4, 1, 1);
    CHECK(!spvT5b.empty(), "recompiled T5b (s_max_i32 SCC on a tie + s_cselect) -> SPIR-V");
    std::vector<float> gotT5b = prosper::test::run_compute(spvT5b, inX, N, N);
    uint32_t badT5b = 0;
    for (uint32_t i = 0; i < N && gotT5b.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotT5b[i], 4);
        if (gb != bits_of(inX[i]) + 4u) badT5b++;
    }
    CHECK(gotT5b.size()==N && badT5b==0, "T5b: s_max_i32(5,5) sets SCC=0 (strict >), s_cselect picks 4");

    // Kernel T5c (#462): s_mul_hi_i32 is gfx10 SOP2 opcode 0x36 (was mapped to the invalid 0x37, so the
    // handler was dead and the op rejected -> shader dropped). s1=0x10000; s_mul_hi_i32 s2,-1,s1 = the
    // SIGNED high dword of (-1 * 0x10000) = high of 0xFFFFFFFFFFFF0000 = 0xFFFFFFFF (unsigned mul_hi would
    // be 0x0000FFFF — so this also pins the SIGNED semantics). v1 = a0 + 0xFFFFFFFF (= a0 - 1).
    const uint32_t codeT5c[] = { 0xbe8103ffu, 0x00010000u, 0x9b0201c1u, 0x4a020002u, 0xBF810000u };
    std::vector<uint32_t> spvT5c = recompile_valu(codeT5c, sizeof(codeT5c)/4, 1, 1);
    CHECK(!spvT5c.empty(), "recompiled T5c (s_mul_hi_i32 opcode 0x36) -> SPIR-V (was rejected as 0x37)");
    std::vector<float> gotT5c = prosper::test::run_compute(spvT5c, inX, N, N);
    uint32_t badT5c = 0;
    for (uint32_t i = 0; i < N && gotT5c.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotT5c[i], 4);
        if (gb != bits_of(inX[i]) + 0xFFFFFFFFu) badT5c++;
    }
    CHECK(gotT5c.size()==N && badT5c==0, "T5c: s_mul_hi_i32(-1,0x10000)=0xFFFFFFFF (signed high mul)");

    // Kernel T5d (#464): v_cmpx must NOT clobber VCC (gfx10: cmpx writes EXEC only). Keep a VCC value
    // live ACROSS a cmpx and read it via v_cndmask. v_cmp_lt_u32 vcc,v0,v0 = FALSE (a live predicate);
    // v_cmpx_eq_u32 v0,v0 leaves EXEC on (v0==v0) and must PRESERVE vcc; v_cndmask v1,v0,v2,vcc then
    // selects v0 (vcc FALSE) — the old code set vcc=cmpx result (TRUE) -> selected v2 = v0+5. Out = a0.
    const uint32_t codeT5d[] = { 0x7d820100u, 0x7da40100u, 0x4a040085u, 0x02020500u, 0xBF810000u };
    std::vector<uint32_t> spvT5d = recompile_valu(codeT5d, sizeof(codeT5d)/4, 1, 1);
    CHECK(!spvT5d.empty(), "recompiled T5d (v_cmpx VCC preservation) -> SPIR-V");
    std::vector<float> gotT5d = prosper::test::run_compute(spvT5d, inX, N, N);
    uint32_t badT5d = 0;
    for (uint32_t i = 0; i < N && gotT5d.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotT5d[i], 4);
        if (gb != bits_of(inX[i])) badT5d++;
    }
    CHECK(gotT5d.size()==N && badT5d==0, "T5d: v_cmpx preserves VCC (cndmask reads the v_cmp result, not cmpx)");

    // Kernel T6: s_add_u32 carry chain -> s_addc_u32. s0=-1+2 (carry SCC=1); s1=0+0+SCC=1.
    // out bits = bits(a0) + 1.
    const uint32_t codeT6[] = { 0x800082c1u, 0x82018080u, 0x4a020001u, 0xBF810000u };
    std::vector<uint32_t> spvT6 = recompile_valu(codeT6, sizeof(codeT6)/4, 1, 1);
    CHECK(!spvT6.empty(), "recompiled T6 (s_addc_u32) -> SPIR-V");
    std::vector<float> gotT6 = prosper::test::run_compute(spvT6, inX, N, N);
    uint32_t badT6 = 0;
    for (uint32_t i = 0; i < N && gotT6.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotT6[i], 4);
        if (gb != bits_of(inX[i]) + 1u) badT6++;
    }
    CHECK(gotT6.size()==N && badT6==0, "T6: s_addc_u32 adds the carry from s_add_u32");

    // Kernel T7: s_ashr_i32 — s1 = (-8) >> 2 (arithmetic) = -2; out bits = bits(a0) + 0xFFFFFFFE.
    const uint32_t codeT7[] = { 0xbe8003c8u, 0x91018200u, 0x4a020001u, 0xBF810000u };
    std::vector<uint32_t> spvT7 = recompile_valu(codeT7, sizeof(codeT7)/4, 1, 1);
    CHECK(!spvT7.empty(), "recompiled T7 (s_ashr_i32) -> SPIR-V");
    std::vector<float> gotT7 = prosper::test::run_compute(spvT7, inX, N, N);
    uint32_t badT7 = 0;
    for (uint32_t i = 0; i < N && gotT7.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotT7[i], 4);
        if (gb != bits_of(inX[i]) + 0xFFFFFFFEu) badT7++;
    }
    CHECK(gotT7.size()==N && badT7==0, "T7: s_ashr_i32 sign-extends the shift");

    // Kernel T7b (#455): v_bfe_u32 with offset+count > 32 must not emit UB. v_bfe_u32 v1, v0, 20, 16 has
    // off=20, count=16 -> off+count = 36 > 32 (SPIR-V OpBitFieldUExtract is UNDEFINED there). RDNA2 reads
    // bits past the MSB as 0, i.e. effective count = min(16, 32-20) = 12, so v1 = (v0 >> 20) & 0xFFF =
    // v0 >> 20 (only 12 bits exist above bit 20). The clamp makes this well-defined.
    const uint32_t codeT7b[] = { 0xd5480001u, 0x02412900u, 0xBF810000u };
    std::vector<uint32_t> spvT7b = recompile_valu(codeT7b, sizeof(codeT7b)/4, 1, 1);
    CHECK(!spvT7b.empty(), "recompiled T7b (v_bfe_u32 off+count>32) -> SPIR-V");
    std::vector<float> gotT7b = prosper::test::run_compute(spvT7b, inX, N, N);
    uint32_t badT7b = 0;
    for (uint32_t i = 0; i < N && gotT7b.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotT7b[i], 4);
        if (gb != (bits_of(inX[i]) >> 20)) badT7b++;
    }
    CHECK(gotT7b.size()==N && badT7b==0, "T7b: v_bfe_u32(v0,20,16) clamps count to 12 -> (v0>>20), not UB");

    // Kernel T7c (#452): v_cvt_pkrtz_f16_f32 is round-toward-ZERO -> an f32 above the f16 range saturates
    // to the max finite f16 (65504 = 0x7BFF), NOT +Inf (0x7C00) as PackHalf2x16's round-to-nearest-even
    // would give. s0 = 100000.0f (0x47c35000, > 65504); v1 = pkrtz(s0, 0) -> low f16 0x7BFF, high 0.
    const uint32_t codeT7c[] = { 0xbe8003ffu, 0x47c35000u, 0xd52f0001u, 0x00010000u, 0xBF810000u };
    std::vector<uint32_t> spvT7c = recompile_valu(codeT7c, sizeof(codeT7c)/4, 1, 1);
    CHECK(!spvT7c.empty(), "recompiled T7c (v_cvt_pkrtz overflow) -> SPIR-V");
    std::vector<float> gotT7c = prosper::test::run_compute(spvT7c, inX, N, N);
    uint32_t badT7c = 0;
    for (uint32_t i = 0; i < N && gotT7c.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotT7c[i], 4);
        if (gb != 0x00007BFFu) badT7c++;
    }
    CHECK(gotT7c.size()==N && badT7c==0, "T7c: pkrtz(100000) saturates to max-finite f16 0x7BFF, not Inf 0x7C00");

    // Kernel T8: s_mov_b64 as plain DATA-pair copy (not a wave mask): s2=9; s[0:1]=s[2:3];
    // out bits = bits(a0) + 9. Previously rejected (src not a recognizable mask).
    const uint32_t codeT8[] = { 0xbe820389u, 0xbe800402u, 0x4a020000u, 0xBF810000u };
    std::vector<uint32_t> spvT8 = recompile_valu(codeT8, sizeof(codeT8)/4, 1, 1);
    CHECK(!spvT8.empty(), "recompiled T8 (s_mov_b64 data-pair copy) -> SPIR-V");
    std::vector<float> gotT8 = prosper::test::run_compute(spvT8, inX, N, N);
    uint32_t badT8 = 0;
    for (uint32_t i = 0; i < N && gotT8.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotT8[i], 4);
        if (gb != bits_of(inX[i]) + 9u) badT8++;
    }
    CHECK(gotT8.size()==N && badT8==0, "T8: s_mov_b64 copies a plain scalar pair");

    // T8b: S_BCNT1_I32_B64 consumes both scalar dwords and writes a 32-bit population count.
    // This is also the data-domain fallback used when the source is not a tracked wave mask.
    const uint32_t codeT8b[] = {
        0xbe8603ffu, 0xf0f0f0f0u,
        0xbe8703ffu, 0x80000001u,
        0xbe801006u,              // s_bcnt1_i32_b64 s0, s[6:7]
        0x7e020200u,              // v_mov_b32 v1, s0
        0xBF810000u,
    };
    const auto spvT8b = recompile_valu(codeT8b, std::size(codeT8b), 1, 1);
    CHECK(!spvT8b.empty(), "recompiled T8b (s_bcnt1_i32_b64 scalar pair) -> SPIR-V");
    const auto gotT8b = prosper::test::run_compute(spvT8b, inX, N, N);
    uint32_t badT8b = 0;
    for (uint32_t i = 0; i < N && gotT8b.size() == N; ++i)
        if (bits_of(gotT8b[i]) != 18u) ++badT8b;
    CHECK(gotT8b.size() == N && badT8b == 0,
          "T8b: bcnt1 sums the low/high dword population counts exactly");

    // T8c: GFX10's complete scalar halfword pack family.  Distinct halves catch every selector.
    const uint32_t pack_ops[] = {0x99000706u, 0x99800706u, 0x9a000706u};
    const uint32_t pack_expected[] = {0x77883344u, 0x55663344u, 0x55661122u};
    for (size_t pack = 0; pack < std::size(pack_ops); ++pack) {
        const uint32_t codeT8c[] = {
            0xbe8603ffu, 0x11223344u,
            0xbe8703ffu, 0x55667788u,
            pack_ops[pack],
            0x7e020200u,
            0xBF810000u,
        };
        const auto spvT8c = recompile_valu(codeT8c, std::size(codeT8c), 1, 1);
        CHECK(!spvT8c.empty(), "recompiled T8c (scalar halfword pack) -> SPIR-V");
        const auto gotT8c = prosper::test::run_compute(spvT8c, inX, N, N);
        uint32_t badT8c = 0;
        for (uint32_t i = 0; i < N && gotT8c.size() == N; ++i)
            if (bits_of(gotT8c[i]) != pack_expected[pack]) ++badT8c;
        CHECK(gotT8c.size() == N && badT8c == 0,
              "T8c: LL/LH/HH select the documented scalar halves exactly");
    }

    // Kernel T9: TWO SEQUENTIAL forward uniform ifs (the DOLL color-grade PS shape, #273).
    // u=(uint)a0; scc=(3<5)=1 -> if1 RUNS (+10,+20); scc=(3<2)=0 -> if2 SKIPPED (+40).
    // out = (float)(u + 30). Previously rejected (single-forward-if only).
    const uint32_t codeT9[] = {
        0x7e020f00u, 0xbe800383u, 0xbf0a8500u, 0xbf840002u, 0x4a02028au, 0x4a020294u,
        0xbf0a8200u, 0xbf840001u, 0x4a0202a8u, 0x7e000d01u, 0xBF810000u,
    };
    std::vector<uint32_t> spvT9 = recompile_valu(codeT9, sizeof(codeT9)/4, 1, 0);
    CHECK(!spvT9.empty(), "recompiled T9 (two sequential forward ifs) -> SPIR-V");
    std::vector<float> inT9(N), expT9(N);
    for (uint32_t i = 0; i < N; i++) { inT9[i] = (float)i; expT9[i] = (float)(i + 30); }
    std::vector<float> gotT9 = prosper::test::run_compute(spvT9, inT9, N, N);
    uint32_t badT9 = 0; for (uint32_t i=0;i<N&&gotT9.size()==N;i++) if (gotT9[i]!=expT9[i]) badT9++;
    printf("  T9(2 seq ifs) mismatches=%u (out[5]=%g expect=%g)\n", badT9, gotT9.size()==N?gotT9[5]:-1, expT9[5]);
    CHECK(gotT9.size()==N && badT9==0, "T9: taken-if adds 10+20, skipped-if omits 40");

    // Kernel T10: NESTED forward uniform ifs (outer runs, inner skipped): u += 1 only.
    const uint32_t codeT10[] = {
        0x7e020f00u, 0xbe800383u, 0xbf0a8500u, 0xbf840004u, 0x4a020281u, 0xbf088700u,
        0xbf840001u, 0x4a020282u, 0x7e000d01u, 0xBF810000u,
    };
    std::vector<uint32_t> spvT10 = recompile_valu(codeT10, sizeof(codeT10)/4, 1, 0);
    CHECK(!spvT10.empty(), "recompiled T10 (nested forward ifs) -> SPIR-V");
    std::vector<float> gotT10 = prosper::test::run_compute(spvT10, inT9, N, N);
    uint32_t badT10 = 0; for (uint32_t i=0;i<N&&gotT10.size()==N;i++) if (gotT10[i]!=(float)(i+1)) badT10++;
    printf("  T10(nested ifs) mismatches=%u (out[5]=%g expect=%g)\n", badT10, gotT10.size()==N?gotT10[5]:-1, (float)(5+1));
    CHECK(gotT10.size()==N && badT10==0, "T10: outer if runs (+1), nested inner if skipped (+2 omitted)");

    // Kernel T11: VOPC SDWA with an ABS source modifier — |a0| > a1 ? 1.0 : 0.0. Previously the
    // decoder kept has_modifier for VOPC neg/abs and the whole compare rejected.
    const uint32_t codeT11[] = { 0x7c0802f9u, 0x06260000u, 0xd5010000u, 0x01a9e480u, 0xBF810000u };
    std::vector<uint32_t> spvT11 = recompile_valu(codeT11, sizeof(codeT11)/4, 2, 0);
    CHECK(!spvT11.empty(), "recompiled T11 (VOPC SDWA |abs| modifier) -> SPIR-V");
    std::vector<float> inT11(N * 2), expT11(N);
    for (uint32_t i = 0; i < N; i++) {
        float a0 = (float)i - 60.0f, a1 = 30.0f;
        inT11[i*2+0] = a0; inT11[i*2+1] = a1; expT11[i] = std::fabs(a0) > a1 ? 1.0f : 0.0f;
    }
    std::vector<float> gotT11 = prosper::test::run_compute(spvT11, inT11, N, N);
    uint32_t badT11 = 0; for (uint32_t i=0;i<N&&gotT11.size()==N;i++) if (gotT11[i]!=expT11[i]) badT11++;
    CHECK(gotT11.size()==N && badT11==0, "T11: v_cmp_gt_f32_sdwa applies |src0| before comparing");

    // Astro Bot's exact visibility-kernel compare: move the input bits to v63, compare integer 1
    // against its zero-extended low byte, then materialize the VCC result as float 0/1.
    const uint32_t codeT11b[] = {
        0x7e7e0300u,               // v_mov_b32 v63, v0
        0x7d8a7ef9u, 0x00860081u,  // v_cmp_ne_u32 vcc, 1, v63 src1_sel:BYTE_0
        0xd5010000u, 0x01a9e480u,  // v_cndmask_b32_e64 v0, 0.0, 1.0, vcc
        0xBF810000u,
    };
    std::vector<uint32_t> spvT11b = recompile_valu(codeT11b, sizeof(codeT11b)/4, 1, 0);
    CHECK(!spvT11b.empty(), "recompiled T11b (Astro integer VOPC SDWA BYTE_0) -> SPIR-V");
    std::vector<float> gotT11b = prosper::test::run_compute(spvT11b, inX, N, N);
    uint32_t badT11b = 0;
    for (uint32_t i = 0; i < N && gotT11b.size() == N; ++i)
        if (gotT11b[i] != (((bits_of(inX[i]) & 0xFFu) != 1u) ? 1.0f : 0.0f)) ++badT11b;
    CHECK(gotT11b.size()==N && badT11b==0,
          "T11b: unsigned SDWA compare zero-extends the selected source byte");

    // Its later paired form selects WORD_1 and updates EXEC instead of VCC. Exact live words;
    // compilation proves the CMPX opcode family is normalized before the integer SDWA admission.
    const uint32_t codeT11c[] = {
        0x7e7e0300u,               // v_mov_b32 v63, v0
        0x7daa7ef9u, 0x05860081u,  // v_cmpx_ne_u32 1, v63 src1_sel:WORD_1
        0x7e0002f2u,               // v_mov_b32 v0, 1.0 (EXEC-predicated)
        0xBF810000u,
    };
    std::vector<uint32_t> spvT11c = recompile_valu(codeT11c, sizeof(codeT11c)/4, 1, 0);
    CHECK(!spvT11c.empty(), "recompiled T11c (Astro CMPX SDWA WORD_1) -> SPIR-V");
    std::vector<float> gotT11c = prosper::test::run_compute(spvT11c, inX, N, N);
    uint32_t badT11c = 0;
    for (uint32_t i = 0; i < N && gotT11c.size() == N; ++i) {
        const bool active = ((bits_of(inX[i]) >> 16) & 0xFFFFu) != 1u;
        if (gotT11c[i] != (active ? 1.0f : inX[i])) ++badT11c;
    }
    CHECK(gotT11c.size()==N && badT11c==0,
          "T11c: CMPX compares the zero-extended high word and predicates the following write");

    // Exact Astro visibility-kernel word. Both unwritten sources are architectural undefined on
    // hardware but initialize to zero in the test shell, so XNOR(0,0) must produce all one bits.
    const uint32_t codeT11d[] = { 0x3cc0c18du, 0xbf810000u }; // v_xnor_b32 v96, v141, v96
    std::vector<uint32_t> spvT11d = recompile_valu(
        codeT11d, sizeof(codeT11d)/sizeof(codeT11d[0]), 1, /*out_vgpr*/96);
    CHECK(!spvT11d.empty(), "recompiled T11d (Astro v_xnor_b32 word) -> SPIR-V");
    std::vector<float> gotT11d = prosper::test::run_compute(spvT11d, inX, N, N);
    uint32_t badT11d = 0;
    for (uint32_t i=0;i<N&&gotT11d.size()==N;i++)
        if (bits_of(gotT11d[i]) != 0xFFFFFFFFu) ++badT11d;
    CHECK(gotT11d.size()==N && badT11d==0,
          "T11d: v_xnor_b32 is the bitwise complement of XOR");

    // Exact Astro V_BCNT packet: copy the input bits to v97, then popcount into v98 with addend 0.
    const uint32_t codeT11e[] = {
        0x7ec20300u,               // v_mov_b32 v97, v0
        0xd7640062u, 0x00010161u,  // v_bcnt_u32_b32 v98, v97, 0
        0xbf810000u,
    };
    std::vector<uint32_t> spvT11e = recompile_valu(
        codeT11e, sizeof(codeT11e)/sizeof(codeT11e[0]), 1, /*out_vgpr*/98);
    CHECK(!spvT11e.empty(), "recompiled T11e (Astro v_bcnt_u32_b32 word) -> SPIR-V");
    std::vector<float> gotT11e = prosper::test::run_compute(spvT11e, inX, N, N);
    uint32_t badT11e = 0;
    for (uint32_t i=0;i<N&&gotT11e.size()==N;i++)
        if (bits_of(gotT11e[i]) != std::popcount(bits_of(inX[i]))) ++badT11e;
    CHECK(gotT11e.size()==N && badT11e==0,
          "T11e: v_bcnt_u32_b32 adds the exact 32-bit population count");

    // Astro's next blocker is VOP3B v_mad_u64_u32. Exercise all three architectural
    // outputs with an overflowing case:
    //   0xffffffff*2 + 0xffffffffffffffff = carry:1, hi:1, lo:0xfffffffd.
    // XOR the low word to zero, convert hi to 1.0, and select 41.0 through carry => 42.0.
    const uint32_t codeT11f[] = {
        0x7eb80282u,                         // v_mov_b32 v92, 2
        0x7e0e02c1u, 0x7e1002c1u,           // v_mov_b32 v7/v8, -1
        0xd5766a05u, 0x041eb8c1u,           // v_mad_u64_u32 v[5:6],vcc,-1,v92,v[7:8]
        0x3a0a0ac3u,                         // v_xor_b32 v5, -3, v5 (validates low)
        0x7e0a0d05u, 0x7e0c0d06u,           // v_cvt_f32_u32 v5/v6, v5/v6
        0x7e1402ffu, 0x42240000u,            // v_mov_b32 v10, 41.0
        0x7e160280u,                         // v_mov_b32 v11, 0
        0x0214150bu,                         // v_cndmask_b32 v10, v11, v10, vcc
        0x06000d05u, 0x06001500u,           // v0 = v5 + v6 + v10
        0xbf810000u,
    };
    std::vector<uint32_t> spvT11f = recompile_valu(
        codeT11f, sizeof(codeT11f)/sizeof(codeT11f[0]), 1, 0);
    CHECK(!spvT11f.empty(), "recompiled T11f (Astro v_mad_u64_u32 VOP3B) -> SPIR-V");
    std::vector<float> gotT11f = prosper::test::run_compute(spvT11f, inX, N, N);
    uint32_t badT11f = 0;
    for (uint32_t i=0;i<N&&gotT11f.size()==N;i++)
        if (gotT11f[i] != 42.0f) ++badT11f;
    CHECK(gotT11f.size()==N && badT11f==0,
          "T11f: v_mad_u64_u32 writes exact low/high words and carry-out");

    // Astro's SSAO pixel shader carries dead `s_and_b64 vcc,s[0:1],vcc` operations between
    // comparisons; s[0:1] is a T# descriptor, not a wave-mask value available to SPIR-V. Prove that
    // the dead write is removed while the following, observable comparison still controls cndmask.
    const uint32_t codeT11g[] = {
        0x7c080300u,              // v_cmp_gt_f32 vcc, v0, v1
        0x87ea6a00u,              // s_and_b64 vcc, s[0:1], vcc (dead before the next compare)
        0x7c020300u,              // v_cmp_lt_f32 vcc, v0, v1
        0x02000101u,              // v_cndmask_b32 v0, v1, v0, vcc => min(v0,v1)
        0xbf810000u,
    };
    std::vector<uint32_t> spvT11g = recompile_valu(
        codeT11g, sizeof(codeT11g)/sizeof(codeT11g[0]), 2, 0);
    CHECK(!spvT11g.empty(), "recompiled T11g (dead Astro scalar-pair mask write) -> SPIR-V");
    std::vector<float> gotT11g = prosper::test::run_compute(spvT11g, in6, N, N);
    uint32_t badT11g = 0;
    for (uint32_t i=0;i<N&&gotT11g.size()==N;i++)
        if (gotT11g[i] != std::min(in6[i*2], in6[i*2+1])) ++badT11g;
    CHECK(gotT11g.size()==N && badT11g==0,
          "T11g: dead mask removal preserves the following VCC compare result");

    // Kernel T12: PC-RELATIVE EMBEDDED TABLE (#273 — DOLL's dither PS idiom). The shader builds a
    // V# with s_getpc_b64 + adds and buffer_loads from a constant table appended AFTER s_endpgm in
    // the code blob. The recompiler folds the load to a compile-time lookup.
    //   getpc s[4:5]; s4+=48; addc s5; s6=16(num_records); s7=cfg; u=(uint)a0; byteoff=u*4;
    //   buffer_load_dword v1, v1, s[4:7], 0 offen  -> table[u]; out=(float)v1.
    // Table at byte 52 of the blob = 48 bytes past the getpc-return (byte 4). llvm-mc gfx1010.
    const uint32_t codeT12[] = {
        0xbe841f00u,               // s_getpc_b64 s[4:5]
        0x800404b0u,               // s_add_u32 s4, 48, s4
        0x82050580u,               // s_addc_u32 s5, 0, s5
        0xbe860390u,               // s_mov_b32 s6, 16
        0xbe8703ffu, 0x10005004u,  // s_mov_b32 s7, 0x10005004
        0x7e020f00u,               // v_cvt_u32_f32 v1, v0
        0x34020282u,               // v_lshlrev_b32 v1, 2, v1
        0xe0301000u, 0x80010101u,  // buffer_load_dword v1, v1, s[4:7], 0 offen
        0xbf8c3f70u,               // s_waitcnt vmcnt(0)
        0x7e000d01u,               // v_cvt_f32_u32 v0, v1
        0xBF810000u,               // s_endpgm
        7u, 11u, 13u, 17u,         // the embedded table (16 bytes)
    };
    std::vector<uint32_t> spvT12 = recompile_valu(codeT12, sizeof(codeT12)/4, 1, 0);
    CHECK(!spvT12.empty(), "recompiled T12 (s_getpc_b64 embedded-table load) -> SPIR-V");
    std::vector<float> inT12(N), expT12(N);
    const uint32_t tabT12[4] = {7u, 11u, 13u, 17u};
    for (uint32_t i = 0; i < N; i++) { inT12[i] = (float)(i % 4); expT12[i] = (float)tabT12[i % 4]; }
    std::vector<float> gotT12 = prosper::test::run_compute(spvT12, inT12, N, N);
    uint32_t badT12 = 0; for (uint32_t i=0;i<N&&gotT12.size()==N;i++) if (gotT12[i]!=expT12[i]) badT12++;
    printf("  T12(pcrel table) mismatches=%u (out[2]=%g expect=%g)\n", badT12, gotT12.size()==N?gotT12[2]:-1, expT12[2]);
    CHECK(gotT12.size()==N && badT12==0, "T12: buffer_load through a getpc-built V# reads the embedded table");

    // Astro Bot emits the same proven V# shape with s_movk_i32 for num_records. Keep that descriptor
    // construction equivalent to s_mov_b32; otherwise s_getpc_b64 is rejected even though all loads
    // remain bounded to the embedded table.
    const uint32_t codeT12movk[] = {
        0xb0060010u,               // s_movk_i32 s6, 16
        0xbe841f00u,               // s_getpc_b64 s[4:5]
        0x800404acu,               // s_add_u32 s4, 44, s4 (table = next-PC byte 8 + 44)
        0x82050580u,               // s_addc_u32 s5, 0, s5
        0xbe8703ffu, 0x10005004u,  // s_mov_b32 s7, 0x10005004
        0x7e020f00u,               // v_cvt_u32_f32 v1, v0
        0x34020282u,               // v_lshlrev_b32 v1, 2, v1
        0xe0301000u, 0x80010101u,  // buffer_load_dword v1, v1, s[4:7], 0 offen
        0xbf8c3f70u,               // s_waitcnt vmcnt(0)
        0x7e000d01u,               // v_cvt_f32_u32 v0, v1
        0xBF810000u,               // s_endpgm
        7u, 11u, 13u, 17u,
    };
    std::vector<uint32_t> spvT12movk = recompile_valu(
        codeT12movk, sizeof(codeT12movk)/4, 1, 0);
    CHECK(!spvT12movk.empty(), "recompiled T12 movk V# (Astro Bot descriptor shape) -> SPIR-V");
    std::vector<float> gotT12movk = prosper::test::run_compute(spvT12movk, inT12, N, N);
    uint32_t badT12movk = 0;
    for (uint32_t i=0;i<N&&gotT12movk.size()==N;i++)
        if (gotT12movk[i]!=expT12[i]) badT12movk++;
    CHECK(gotT12movk.size()==N && badT12movk==0,
          "T12 movk: getpc-built V# reads the embedded table");

    // A PC-relative table belongs to the shader itself, not to the runtime resource table. Astro
    // Bot's loading-screen VS has no external resources and loads its position from exactly this
    // shape. The graphics-stage gate must therefore accept the proven MUBUF before requiring `rt`.
    const uint32_t codeT12vertex[] = {
        0xb0020010u,               // s_movk_i32 s2, 16 bytes
        0xbe8303ffu, 0x10005004u,  // s_mov_b32 s3, V# config
        0xbe801f00u,               // s_getpc_b64 s[0:1] (next PC byte = 16)
        0x800000ffu, 0x00000028u,  // s_add_u32 s0, 40, s0 (table byte = 56)
        0x82010180u,               // s_addc_u32 s1, 0, s1
        0x7e080280u,               // v_mov_b32 v4, 0 (table byte offset)
        0xe0381000u, 0x80000004u,  // buffer_load_dwordx4 v[0:3], v4, s[0:3], 0 offen
        0xbf8c3f70u,               // s_waitcnt vmcnt(0)
        0xf80008cfu, 0x03020100u,  // exp pos0, v0, v1, v2, v3
        0xbf810000u,               // s_endpgm
        0xbf800000u, 0xbf800000u, 0u, 0x3f800000u,
    };
    std::vector<uint32_t> spvT12vertex = recompile_vertex(
        codeT12vertex, sizeof(codeT12vertex)/4, nullptr);
    CHECK(!spvT12vertex.empty(),
          "T12 vertex: resource-free stage accepts a proven embedded-table MUBUF");

    // Astro Bot's title pixel shader builds the same bounded descriptor but consumes it with SMEM.
    // The scalar byte offset selects table[1], and x4 returns table[1..4]; summing the first two
    // loaded SGPRs proves that the fold honors both the dynamic offset and consecutive destinations.
    const uint32_t codeT12smem[] = {
        0xbe801f00u,               // s_getpc_b64 s[0:1] (next-PC byte = 4)
        0x800000ffu, 48u,          // s_add_u32 s0, 48, s0 (table byte = 52)
        0x82010180u,               // s_addc_u32 s1, 0, s1
        0xbe820394u,               // s_mov_b32 s2, 20 bytes
        0xbe8303ffu, 0x10005004u,  // s_mov_b32 s3, V# config
        0xbeea0384u,               // s_mov_b32 vcc_lo, 4 (select table[1])
        0xf4280100u, 0xd4000000u,  // s_buffer_load_dwordx4 s[4:7], s[0:3], vcc_lo
        0x80040504u,               // s_add_u32 s4, s4, s5 => 11 + 13
        0x7e000c04u,               // v_cvt_f32_u32 v0, s4
        0xbf810000u,               // s_endpgm
        7u, 11u, 13u, 17u, 19u,
    };
    std::vector<uint32_t> spvT12smem = recompile_valu(
        codeT12smem, std::size(codeT12smem), 1, 0);
    CHECK(!spvT12smem.empty(), "T12 SMEM: PC-relative scalar table recompiles without a resource");
    std::vector<float> gotT12smem = spvT12smem.empty() ? std::vector<float>{} :
        prosper::test::run_compute(spvT12smem, inT12, N, N);
    CHECK(gotT12smem.size() == N &&
              std::all_of(gotT12smem.begin(), gotT12smem.end(), [](float value) { return value == 24.0f; }),
          "T12 SMEM: dynamic byte offset and consecutive scalar table results are preserved");
    CHECK(rdna2_recompile_code_span(codeT12smem, std::size(codeT12smem)) ==
              std::size(codeT12smem),
          "T12 SMEM: the full embedded scalar table participates in the owning code span");
    std::vector<uint32_t> truncatedT12smem(
        std::begin(codeT12smem), std::end(codeT12smem) - 1);
    CHECK(recompile_valu(truncatedT12smem.data(), truncatedT12smem.size(), 1, 0).empty(),
          "T12 SMEM: a truncated embedded table remains rejected");
    std::vector<uint32_t> brokenHighT12smem(std::begin(codeT12smem), std::end(codeT12smem));
    brokenHighT12smem[3] = 0x82010181u; // s_addc_u32 s1, 1, s1: not the proven high-half chain
    CHECK(recompile_valu(brokenHighT12smem.data(), brokenHighT12smem.size(), 1, 0).empty(),
          "T12 SMEM: a broken PC-relative high-half chain remains rejected");
    std::vector<uint32_t> untrackedOffsetT12smem(std::begin(codeT12smem), std::end(codeT12smem));
    untrackedOffsetT12smem[8] = 0x10000000u; // SOFFSET=s8, which has no tracked scalar value
    CHECK(recompile_valu(untrackedOffsetT12smem.data(), untrackedOffsetT12smem.size(), 1, 0).empty(),
          "T12 SMEM: an untracked scalar table offset remains rejected");

    // Kernel T13: UINT8x1 vertex fetch (#273 — DOLL's skinned scene VS bone-index attribute class).
    // buffer_load_format_x of a Uint8 element delivers the RAW integer (no normalization) in the VGPR.
    // v1=(uint)a0 (element idx); fetch; out=(float)(uint)v1. llvm-mc gfx1010:
    //   v_cvt_u32_f32 v1,v0 | buffer_load_format_x v1, v1, s[8:11], 0 idxen | s_waitcnt | v_cvt_f32_u32 v0,v1
    const uint32_t codeT13[] = { 0x7e020f00u, 0xe0002000u, 0x80020101u, 0xbf8c3f70u, 0x7e000d01u, 0xbf810000u };
    ShaderResourceTable rtT13;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Uint8;
      vb.num_components = 1; vb.binding = 3; vb.stride = 4; vb.sgpr_base = 8; rtT13.resources.push_back(vb); }
    std::vector<uint32_t> spvT13 = recompile_valu(codeT13, sizeof(codeT13)/4, 1, 0, &rtT13);
    CHECK(!spvT13.empty(), "recompiled T13 (buffer_load_format_x UINT8) -> SPIR-V");
    std::vector<float> inT13(N), expT13(N); std::vector<uint32_t> vbufT13(N);
    for (uint32_t i = 0; i < N; i++) { inT13[i] = (float)i; uint32_t bv = (7u * i + 3u) & 0xFFu;
        vbufT13[i] = bv | 0xA5A5A500u;   // garbage in the upper bytes must NOT leak into the component
        expT13[i] = (float)bv; }
    std::vector<float> gotT13 = prosper::test::run_compute(spvT13, inT13, N, N, {}, vbufT13);
    uint32_t badT13 = 0; for (uint32_t i=0;i<N&&gotT13.size()==N;i++) if (gotT13[i]!=expT13[i]) badT13++;
    printf("  T13(uint8 fetch) mismatches=%u (out[5]=%g expect=%g)\n", badT13, gotT13.size()==N?gotT13[5]:-1, expT13[5]);
    CHECK(gotT13.size()==N && badT13==0, "T13: Uint8 attribute delivers the raw zero-extended integer");

    // Pixel-stage dynamic V#s use the same format-load instruction for structured/material buffers.
    // A stride-2 Uint16 table must retain its computed VADDR and select the runtime dword half; treating
    // this pc-keyed resource as a VertexBuffer would substitute gl_VertexIndex and reject the load.
    ShaderResourceTable rtT13structured;
    { ShaderResource cb{}; cb.cls = ResourceClass::ConstantBuffer; cb.format = DataFormat::Uint16;
      cb.num_components = 1; cb.binding = 3; cb.stride = 2; cb.sgpr_base = 8; cb.fetch_pc = 1;
      rtT13structured.resources.push_back(cb); }
    std::vector<uint32_t> spvT13structured = recompile_valu(
        codeT13, sizeof(codeT13)/4, 1, 0, &rtT13structured);
    std::vector<uint32_t> tableT13structured((N + 1) / 2, 0u);
    std::vector<float> expT13structured(N);
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t value = 1000u + i;
        tableT13structured[i / 2] |= value << ((i & 1u) * 16u);
        expT13structured[i] = static_cast<float>(value);
    }
    std::vector<float> gotT13structured = prosper::test::run_compute(
        spvT13structured, inT13, N, N, {}, tableT13structured);
    uint32_t badT13structured = 0;
    for (uint32_t i = 0; i < N && gotT13structured.size() == N; ++i)
        if (gotT13structured[i] != expT13structured[i]) ++badT13structured;
    CHECK(!spvT13structured.empty() && gotT13structured.size() == N && badT13structured == 0,
          "T13 structured: stride-2 Uint16 format load uses computed VADDR and runtime half");

    // Kernel T14: SINT8x1 vertex fetch — sign-extended integer. Same code; V# format Sint8; the
    // final convert is v_cvt_f32_i32 so -1 (0xFF) comes back as -1.0.
    const uint32_t codeT14[] = { 0x7e020f00u, 0xe0002000u, 0x80020101u, 0xbf8c3f70u, 0x7e000b01u, 0xbf810000u };
    ShaderResourceTable rtT14;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Sint8;
      vb.num_components = 1; vb.binding = 3; vb.stride = 4; vb.sgpr_base = 8; rtT14.resources.push_back(vb); }
    std::vector<uint32_t> spvT14 = recompile_valu(codeT14, sizeof(codeT14)/4, 1, 0, &rtT14);
    CHECK(!spvT14.empty(), "recompiled T14 (buffer_load_format_x SINT8) -> SPIR-V");
    std::vector<float> inT14(N), expT14(N); std::vector<uint32_t> vbufT14(N);
    for (uint32_t i = 0; i < N; i++) { inT14[i] = (float)i; int32_t sv = (int32_t)(i % 5) - 2;   // -2..2
        vbufT14[i] = ((uint32_t)sv & 0xFFu) | 0x5A5A5A00u; expT14[i] = (float)sv; }
    std::vector<float> gotT14 = prosper::test::run_compute(spvT14, inT14, N, N, {}, vbufT14);
    uint32_t badT14 = 0; for (uint32_t i=0;i<N&&gotT14.size()==N;i++) if (gotT14[i]!=expT14[i]) badT14++;
    printf("  T14(sint8 fetch) mismatches=%u (out[7]=%g expect=%g)\n", badT14, gotT14.size()==N?gotT14[7]:-1, expT14[7]);
    CHECK(gotT14.size()==N && badT14==0, "T14: Sint8 attribute delivers the sign-extended integer");

    // Kernel T15: REGISTER-SOFFSET s_buffer_load (#273 — DOLL's bloom-combine loop reads per-tap
    // weights at a computed cbuf offset). s4 = 4*(uint)a0 (readfirstlane); s_buffer_load_dword
    // s5, s[0:3], s4 offset:0x4 -> cbuf dword (4*u+4)>>2 = u+1; out=(float)s5. llvm-mc gfx1010.
    const uint32_t codeT15[] = { 0x7e020f00u, 0x34020282u, 0x7e080501u,
                                 0xf4200140u, 0x08000004u, 0xbf8cc07fu, 0x7e000c05u, 0xbf810000u };
    std::vector<uint32_t> spvT15 = recompile_valu(codeT15, sizeof(codeT15)/4, 1, 0);
    CHECK(!spvT15.empty(), "recompiled T15 (s_buffer_load with register SOFFSET) -> SPIR-V");
    std::vector<uint32_t> cbufT15 = { 100u, 7u, 11u, 13u, 17u, 23u, 29u, 31u, 37u, 41u, 43u };
    std::vector<float> inT15(8), expT15(8);
    for (uint32_t i = 0; i < 8; i++) { inT15[i] = (float)i; expT15[i] = (float)cbufT15[i + 1]; }
    std::vector<float> gotT15 = prosper::test::run_compute(spvT15, inT15, 8, 8, cbufT15);
    uint32_t badT15 = 0; for (uint32_t i=0;i<8&&gotT15.size()==8;i++) if (gotT15[i]!=expT15[i]) badT15++;
    printf("  T15(dyn s_buffer_load) mismatches=%u (out[2]=%g expect=%g)\n", badT15, gotT15.size()==8?gotT15[2]:-1, expT15[2]);
    CHECK(gotT15.size()==8 && badT15==0, "T15: register-SOFFSET s_buffer_load indexes the cbuf dynamically");

    // Kernel T16: v_movrels_b32 (#273 — DOLL UI/skinned VS: M0-relative VGPR read). m0 = (uint)a0
    // (via v_readfirstlane m0); v2=10.0 v3=20.0 v4=30.0; v_movrels_b32 v0, v2 -> VGPR[2+m0].
    const uint32_t codeT16[] = { 0x7e020f00u, 0x7ef80501u,
                                 0x7e0402ffu, 0x41200000u, 0x7e0602ffu, 0x41a00000u, 0x7e0802ffu, 0x41f00000u,
                                 0x7e008702u, 0xbf810000u };
    std::vector<uint32_t> spvT16 = recompile_valu(codeT16, sizeof(codeT16)/4, 1, 0);
    CHECK(!spvT16.empty(), "recompiled T16 (v_movrels_b32) -> SPIR-V");
    std::vector<float> inT16(8), expT16(8);
    const float relT16[3] = { 10.0f, 20.0f, 30.0f };
    for (uint32_t i = 0; i < 8; i++) { inT16[i] = (float)(i % 3); expT16[i] = relT16[i % 3]; }
    std::vector<float> gotT16 = prosper::test::run_compute(spvT16, inT16, 8, 8);
    uint32_t badT16 = 0; for (uint32_t i=0;i<8&&gotT16.size()==8;i++) if (gotT16[i]!=expT16[i]) badT16++;
    printf("  T16(v_movrels) mismatches=%u (out[4]=%g expect=%g)\n", badT16, gotT16.size()==8?gotT16[4]:-1, expT16[4]);
    CHECK(gotT16.size()==8 && badT16==0, "T16: v_movrels_b32 reads VGPR[src0+M0] per invocation");

    // Kernel T17: IF/ELSE-IF/ELSE CASCADE via common-merge s_branch arms (#273 — DOLL's color-grade
    // PS shape: every arm's s_branch jumps to the SAME outer merge). u<2 -> +10; 2<=u<5 -> +20; else +30.
    const uint32_t codeT17[] = {
        0x7e020f00u, 0x7e080501u, 0xbf0a8204u, 0xbf840002u, 0x4a02028au, 0xbf820005u,
        0xbf0a8504u, 0xbf840002u, 0x4a020294u, 0xbf820001u, 0x4a02029eu, 0x7e000d01u, 0xbf810000u,
    };
    std::vector<uint32_t> spvT17 = recompile_valu(codeT17, sizeof(codeT17)/4, 1, 0);
    CHECK(!spvT17.empty(), "recompiled T17 (if/else-if/else cascade, common merge) -> SPIR-V");
    std::vector<float> inT17(8), expT17(8);
    for (uint32_t i = 0; i < 8; i++) { inT17[i] = (float)i;
        expT17[i] = (float)(i + (i < 2 ? 10u : i < 5 ? 20u : 30u)); }
    std::vector<float> gotT17 = prosper::test::run_compute(spvT17, inT17, 8, 8);
    uint32_t badT17 = 0; for (uint32_t i=0;i<8&&gotT17.size()==8;i++) if (gotT17[i]!=expT17[i]) badT17++;
    printf("  T17(cascade) mismatches=%u (out[0]=%g out[3]=%g out[6]=%g expect 10/23/36)\n", badT17,
           gotT17.size()==8?gotT17[0]:-1, gotT17.size()==8?gotT17[3]:-1, gotT17.size()==8?gotT17[6]:-1);
    CHECK(gotT17.size()==8 && badT17==0, "T17: each cascade arm selects its own addend");

    // Kernel T18: NESTED if/else INSIDE the outer then-arm, both arms' s_branch jumping to the
    // OUTERMOST merge (DOLL ps_2086a60000's exact nesting). u<2 -> +10; 2<=u<4 -> +20; else +30.
    const uint32_t codeT18[] = {
        0x7e020f00u, 0x7e080501u, 0xbf0a8404u, 0xbf840006u, 0xbf0a8204u, 0xbf840002u,
        0x4a02028au, 0xbf820003u, 0x4a020294u, 0xbf820001u, 0x4a02029eu, 0x7e000d01u, 0xbf810000u,
    };
    std::vector<uint32_t> spvT18 = recompile_valu(codeT18, sizeof(codeT18)/4, 1, 0);
    CHECK(!spvT18.empty(), "recompiled T18 (nested if/else, escaping merge to outer) -> SPIR-V");
    std::vector<float> inT18(8), expT18(8);
    for (uint32_t i = 0; i < 8; i++) { inT18[i] = (float)i;
        expT18[i] = (float)(i + (i < 2 ? 10u : i < 4 ? 20u : 30u)); }
    std::vector<float> gotT18 = prosper::test::run_compute(spvT18, inT18, 8, 8);
    uint32_t badT18 = 0; for (uint32_t i=0;i<8&&gotT18.size()==8;i++) if (gotT18[i]!=expT18[i]) badT18++;
    printf("  T18(nested cascade) mismatches=%u (out[1]=%g out[2]=%g out[5]=%g expect 11/22/35)\n", badT18,
           gotT18.size()==8?gotT18[1]:-1, gotT18.size()==8?gotT18[2]:-1, gotT18.size()==8?gotT18[5]:-1);
    CHECK(gotT18.size()==8 && badT18==0, "T18: nested arms + outer else each select their own addend");

    // Kernel T19: READFIRSTLANE WATERFALL (#273 — DOLL's skinned scene VS bone-matrix indexing).
    // remaining=exec; L: s4=readfirstlane(v1); v_cmpx_eq(s4,v1); m0=s4; v_movrels v5, v2;
    // remaining &= ~exec; exec=remaining; s_cbranch_scc1 L; exec=-1. Per-invocation the loop runs
    // once (my lane IS the first active lane of my own iteration), so the backward mask-SCC branch
    // linearizes away. out = table[u] for u in 0..2 (table = v2..v4 = 5/7/9).
    const uint32_t codeT19[] = {
        0x7e020f00u, 0xbe86047eu, 0x7e0402ffu, 0x40a00000u, 0x7e0602ffu, 0x40e00000u,
        0x7e0802ffu, 0x41100000u, 0x7e080501u, 0x7da40204u, 0xbefc0304u, 0x7e0a8702u,
        0x8a867e06u, 0xbefe0406u, 0xbf85fff9u, 0xbefe04c1u, 0x7e000305u, 0xbf810000u,
    };
    std::vector<uint32_t> spvT19 = recompile_valu(codeT19, sizeof(codeT19)/4, 1, 0);
    CHECK(!spvT19.empty(), "recompiled T19 (readfirstlane waterfall + movrels) -> SPIR-V");
    std::vector<float> inT19(8), expT19(8);
    const float tabT19[3] = { 5.0f, 7.0f, 9.0f };
    for (uint32_t i = 0; i < 8; i++) { inT19[i] = (float)(i % 3); expT19[i] = tabT19[i % 3]; }
    std::vector<float> gotT19 = prosper::test::run_compute(spvT19, inT19, 8, 8);
    uint32_t badT19 = 0; for (uint32_t i=0;i<8&&gotT19.size()==8;i++) if (gotT19[i]!=expT19[i]) badT19++;
    printf("  T19(waterfall) mismatches=%u (out[5]=%g expect=%g)\n", badT19, gotT19.size()==8?gotT19[5]:-1, expT19[5]);
    CHECK(gotT19.size()==8 && badT19==0, "T19: waterfall loop linearizes to a per-invocation once-through");

    // Kernel T20: s_bitcmp1_b32 + s_cselect (#273 — DOLL's feature-flag test chain). u=(uint)a0;
    // readfirstlane s4; s_bitcmp1_b32 s4, 1 (SCC = bit1); s_cselect_b32 s5, 100, 200; out=(float)s5.
    // llvm-mc gfx1010: 0xbf0d8104 = s_bitcmp1_b32 s4, 1; 0x850580e4/0x8505e4... assembled below.
    const uint32_t codeT20[] = {
        0x7e020f00u,               // v_cvt_u32_f32 v1, v0
        0x7e080501u,               // v_readfirstlane_b32 s4, v1
        0xbf0d8104u,               // s_bitcmp1_b32 s4, 1
        0x850580ffu, 0x00000064u,  // s_cselect_b32 s5, 0x64(100), 0        (scc ? 100 : 0)
        0x7e000c05u,               // v_cvt_f32_u32 v0, s5
        0xbf810000u,
    };
    std::vector<uint32_t> spvT20 = recompile_valu(codeT20, sizeof(codeT20)/4, 1, 0);
    CHECK(!spvT20.empty(), "recompiled T20 (s_bitcmp1_b32 + s_cselect) -> SPIR-V");
    std::vector<float> inT20(8), expT20(8);
    for (uint32_t i = 0; i < 8; i++) { inT20[i] = (float)i; expT20[i] = (i & 2u) ? 100.0f : 0.0f; }
    std::vector<float> gotT20 = prosper::test::run_compute(spvT20, inT20, 8, 8);
    uint32_t badT20 = 0; for (uint32_t i=0;i<8&&gotT20.size()==8;i++) if (gotT20[i]!=expT20[i]) badT20++;
    printf("  T20(bitcmp1) mismatches=%u (out[2]=%g expect=100)\n", badT20, gotT20.size()==8?gotT20[2]:-1);
    CHECK(gotT20.size()==8 && badT20==0, "T20: s_bitcmp1_b32 sets SCC from the selected bit");

    // Kernel T20b: Astro's exact s_cmp_eq_u64 s[2:3], 0 form. Exercise both the equal pair and a
    // value whose low half is zero but high half is one, proving that SCC uses both encoded dwords.
    const uint32_t codeT20b_eq[] = {
        0xbe820380u, 0xbe830380u, 0xbf128002u,
        0x850580ffu, 0x00000064u, 0x7e000c05u, 0xbf810000u,
    };
    const uint32_t codeT20b_ne[] = {
        0xbe820380u, 0xbe830381u, 0xbf128002u,
        0x850580ffu, 0x00000064u, 0x7e000c05u, 0xbf810000u,
    };
    std::vector<uint32_t> spvT20b_eq = recompile_valu(codeT20b_eq, std::size(codeT20b_eq), 0, 0);
    std::vector<uint32_t> spvT20b_ne = recompile_valu(codeT20b_ne, std::size(codeT20b_ne), 0, 0);
    CHECK(!spvT20b_eq.empty() && !spvT20b_ne.empty(),
          "recompiled T20b (Astro s_cmp_eq_u64 word) -> SPIR-V");
    std::vector<float> gotT20b_eq = prosper::test::run_compute(spvT20b_eq, inT20, 8, 8);
    std::vector<float> gotT20b_ne = prosper::test::run_compute(spvT20b_ne, inT20, 8, 8);
    uint32_t badT20b = 0;
    for (uint32_t i = 0; i < 8 && gotT20b_eq.size() == 8 && gotT20b_ne.size() == 8; ++i)
        if (gotT20b_eq[i] != 100.0f || gotT20b_ne[i] != 0.0f) ++badT20b;
    CHECK(gotT20b_eq.size()==8 && gotT20b_ne.size()==8 && badT20b==0,
          "T20b: u64 equality compares both SGPR halves and drives SCC exactly");

    // Kernel T23: v_cube{id,sc,tc,ma}_f32 (#273 — DOLL's reflection-probe cube math). Direction
    // (x,y,z) = (1.0, 0.5, -2.0): |z| is the major axis and z<0, so per the GL cube table
    // id=5, sc=-x=-1, tc=-y=-0.5, ma=2z=-4. out = id*100 + sc*10 + tc + ma/1024
    //     = 500 - 10 - 0.5 - 0.00390625 = 489.4961. (Assembled by llvm-mc gfx1030.)
    const uint32_t codeT23[] = {
        0x7E0202F2u, 0x7E0402F0u, 0x7E0602F5u, 0xD5440004u, 0x040E0501u, 0xD5450005u,
        0x040E0501u, 0xD5460006u, 0x040E0501u, 0xD5470007u, 0x040E0501u, 0x100808FFu,
        0x42C80000u, 0x100A0AFFu, 0x41200000u, 0x06080B04u, 0x06080D04u, 0x100E0EFFu,
        0x3A800000u, 0x06000F04u, 0xBF810000u,
    };
    std::vector<uint32_t> spvT23 = recompile_valu(codeT23, sizeof(codeT23)/sizeof(codeT23[0]), 1, 0);
    CHECK(!spvT23.empty(), "recompiled kernel T23 (v_cube* ops) -> SPIR-V");
    std::vector<float> inT23(8, 0.f);
    std::vector<float> gotT23 = prosper::test::run_compute(spvT23, inT23, 8, 8);
    uint32_t badT23 = 0;
    for (uint32_t i = 0; i < 8 && gotT23.size() == 8; i++)
        if (std::fabs(gotT23[i] - 489.4961f) > 1e-2f) badT23++;
    printf("  kernelT23 mismatches=%u (out[0]=%g expect=489.4961)\n", badT23, gotT23.size()==8?gotT23[0]:-1);
    CHECK(gotT23.size()==8 && badT23==0, "T23: cube id/sc/tc/ma match the GL major-axis table");

    // Kernel T21: v_fma_mixlo_f16 / mixhi (#273 — DOLL's box-blur f16 packing). v1=(a0*2+3) via
    // fma_mix_f32; then mixlo packs (a0+1) into v2's low half and mixhi packs (a0+2) into its high
    // half; out = f16lo(v2) + f16hi(v2) = (a0+1)+(a0+2) recovered via unpack (v_cvt_f32_f16-free:
    // just re-add via another fma_mix on unpacked halves is overkill — compare packed halves on CPU).
    // Simpler: out = (float)(uint)v2, compared against the CPU-packed expected bits.
    const uint32_t codeT21[] = {
        0x7e020280u,               // v_mov_b32 v1, 0
        0x7e040280u,               // v_mov_b32 v2, 0
        0xcc210002u, 0x040600f2u,  // v_fma_mixlo_f16 v2, 1.0, v0, v1   (= a0 into lo half)
        0xcc220002u, 0x040600f2u,  // v_fma_mixhi_f16 v2, 1.0, v0, v1   (= a0 into hi half)
        0x7e000d02u,               // v_cvt_f32_u32 v0, v2  (raw packed bits -> float for compare)
        0xbf810000u,
    };
    // (words llvm-mc-round-tripped: 0xcc210002/0xcc220002 + 0x040600f2 = mixlo/mixhi v2, 1.0, v0, v1)
    std::vector<uint32_t> spvT21 = recompile_valu(codeT21, sizeof(codeT21)/4, 1, 0);
    CHECK(!spvT21.empty(), "recompiled T21 (v_fma_mixlo/mixhi_f16) -> SPIR-V");
    auto f2h = [](float f) -> uint32_t {   // float -> IEEE binary16 bits (round-to-nearest-even)
        union { float f; uint32_t u; } c{f};
        uint32_t s = (c.u >> 16) & 0x8000u; int32_t e = (int32_t)((c.u >> 23) & 0xFF) - 127 + 15;
        uint32_t m = c.u & 0x7FFFFFu;
        if (e <= 0) return s;                          // (test values stay normal; flush tiny to 0)
        if (e >= 31) return s | 0x7C00u;
        uint32_t h = s | ((uint32_t)e << 10) | (m >> 13);
        if ((m & 0x1FFFu) > 0x1000u || (((m & 0x1FFFu) == 0x1000u) && (h & 1u))) h++;
        return h;
    };
    std::vector<float> inT21(8), expT21(8);
    for (uint32_t i = 0; i < 8; i++) { float a = (float)i * 0.25f; inT21[i] = a;
        uint32_t packed = f2h(a) | (f2h(a) << 16); expT21[i] = (float)packed; }
    std::vector<float> gotT21 = prosper::test::run_compute(spvT21, inT21, 8, 8);
    uint32_t badT21 = 0; for (uint32_t i=0;i<8&&gotT21.size()==8;i++) if (gotT21[i]!=expT21[i]) badT21++;
    printf("  T21(fma_mix pack) mismatches=%u (out[4]=%.0f expect=%.0f)\n", badT21,
           gotT21.size()==8?gotT21[4]:-1, expT21[4]);
    CHECK(gotT21.size()==8 && badT21==0, "T21: mixlo/mixhi pack f16 halves preserving the other half");

    // Kernel T22: the DOLL box-blur f16 TAIL verbatim (#273): v0.lo = f16(a0) (fma_mixlo);
    // v_mul_f16_sdwa v0, vcc_lo(=2.0h), v0 dst_sel:WORD_1 preserve (hi = 2*f16(a0), lo kept);
    // v_mov_b32_sdwa v0, v0 dst_sel:WORD_0 preserve src0_sel:WORD_1 (lo = hi). out = packed bits.
    const uint32_t codeT22[] = {
        0x7e020280u,               // v_mov_b32 v1, 0
        0xcc210000u, 0x040600f2u,  // v_fma_mixlo_f16 v0, 1.0, v0, v1
        0xb06a4000u,               // s_movk_i32 vcc_lo, 0x4000 (f16 2.0)
        0x6a0000f9u, 0x0686156au,  // v_mul_f16_sdwa v0, vcc_lo, v0 dst:WORD_1 preserve (live blur words)
        0x7e0002f9u, 0x00051400u,  // v_mov_b32_sdwa v0, v0 dst:WORD_0 preserve src:WORD_1 (live)
        0x7e000d00u,               // v_cvt_f32_u32 v0, v0
        0xbf810000u,
    };
    std::vector<uint32_t> spvT22 = recompile_valu(codeT22, sizeof(codeT22)/4, 1, 0);
    CHECK(!spvT22.empty(), "recompiled T22 (f16 SDWA WORD-select mul/mov tail) -> SPIR-V");
    std::vector<float> inT22(8), expT22(8);
    for (uint32_t i = 0; i < 8; i++) { float a = (float)i * 0.25f; inT22[i] = a;
        uint32_t h2 = f2h(2.0f * a); expT22[i] = (float)(h2 | (h2 << 16)); }
    std::vector<float> gotT22 = prosper::test::run_compute(spvT22, inT22, 8, 8);
    uint32_t badT22 = 0; for (uint32_t i=0;i<8&&gotT22.size()==8;i++) if (gotT22[i]!=expT22[i]) badT22++;
    printf("  T22(f16 sdwa words) mismatches=%u (out[3]=%.0f expect=%.0f)\n", badT22,
           gotT22.size()==8?gotT22[3]:-1, expT22[3]);
    CHECK(gotT22.size()==8 && badT22==0, "T22: WORD-dst f16 mul + WORD-to-WORD mov preserve halves exactly");

    // Kernel T24: v_cvt_off_f32_i4 (#527). The low nibble is a signed i4 [-8,7],
    // converted to f32 and scaled by 1/16. Exercise every nibble, including both signs.
    const uint32_t codeT24[] = { 0x7e001d00u, 0xbf810000u };
    std::vector<uint32_t> spvT24 = recompile_valu(codeT24, sizeof(codeT24)/4, 1, 0);
    CHECK(!spvT24.empty(), "recompiled T24 (v_cvt_off_f32_i4) -> SPIR-V");
    std::vector<float> inT24(16), expT24(16);
    for (uint32_t i = 0; i < 16; ++i) {
        uint32_t raw = i; std::memcpy(&inT24[i], &raw, sizeof raw);
        int32_t s4 = (i & 8u) ? (int32_t)i - 16 : (int32_t)i;
        expT24[i] = (float)s4 * 0.0625f;
    }
    std::vector<float> gotT24 = prosper::test::run_compute(spvT24, inT24, 16, 16);
    uint32_t badT24 = 0;
    for (uint32_t i = 0; i < 16 && gotT24.size() == 16; ++i)
        if (gotT24[i] != expT24[i]) badT24++;
    printf("  T24(i4 offset convert) mismatches=%u (nibble 7=%g, 8=%g)\n", badT24,
           gotT24.size()==16?gotT24[7]:-9.f, gotT24.size()==16?gotT24[8]:-9.f);
    CHECK(gotT24.size()==16 && badT24==0,
          "T24: signed i4 values convert to f32 multiples of 1/16 exactly");

    // T25: Astro Bot's compute blur uses v_mov_b32_dpp quad permutations to exchange values inside
    // each 2x2 quad. These are subgroup quad swaps, not screen-space derivatives (the fragment-only
    // lowering). Exercise the exact live DPP controls: 0xb1 = lane^1 (horizontal), 0x4e = lane^2
    // (vertical), 0x1b = lane^3 (diagonal).
    const uint32_t dpp_controls[] = {0xff08b100u, 0xff084e00u, 0xff081b00u};
    const uint32_t dpp_xor[] = {1u, 2u, 3u};
    for (uint32_t k = 0; k < 3; ++k) {
        const uint32_t codeT25[] = {0x7e0002fau, dpp_controls[k], 0xbf810000u};
        std::vector<uint32_t> spvT25 = recompile_valu(codeT25, sizeof(codeT25)/4, 1, 0);
        CHECK(!spvT25.empty(), "recompiled T25 (compute DPP16 quad swap) -> SPIR-V");
        std::vector<float> inT25(128), expT25(128);
        for (uint32_t i = 0; i < 128; ++i) inT25[i] = (float)i;
        for (uint32_t i = 0; i < 128; ++i) expT25[i] = inT25[i ^ dpp_xor[k]];
        std::vector<float> gotT25 = prosper::test::run_compute(spvT25, inT25, 128, 128);
        uint32_t badT25 = 0;
        for (uint32_t i = 0; i < 128 && gotT25.size() == 128; ++i)
            if (gotT25[i] != expT25[i]) ++badT25;
        CHECK(gotT25.size() == 128 && badT25 == 0,
              "T25: compute DPP16 returns the selected neighbor from each quad");
    }

    // T25b (#1390): Plucky's gameplay reduction uses DPP ROW_SHR:1 repeatedly for a 16-lane
    // prefix min/max. With BOUND_CTRL=0, lane zero of each row is disabled and preserves VDST;
    // every other lane reads SRC0 from its left neighbor. The second kernel covers BOUND_CTRL=1,
    // where the first two lanes receive architectural zero instead.
    std::vector<float> inT25b(128), expT25b(128), expT25b_bound(128);
    for (uint32_t i = 0; i < 128; ++i) {
        inT25b[i] = (float)(i + 1u);
        const uint32_t row_lane = i & 15u;
        expT25b[i] = row_lane == 0 ? inT25b[i] : inT25b[i - 1u];
        expT25b_bound[i] = row_lane < 2 ? 0.0f : inT25b[i - 2u];
    }
    const uint32_t codeT25b[] = {
        0x1e0000fau, 0xff011100u,             // v_min_f32_dpp v0,v0,v0 row_shr:1 BC:0
        0xbf810000u,
    };
    std::vector<uint32_t> spvT25b = recompile_valu(
        codeT25b, std::size(codeT25b), 1, 0);
    CHECK(!spvT25b.empty(), "recompiled T25b (Plucky DPP ROW_SHR:1 min) -> SPIR-V");
    CHECK(compute_spirv_min_subgroup_size(spvT25b) == 16,
          "T25b: row-shuffle module advertises its 16-lane host contract");

    const uint32_t codeT25b_bound[] = {
        0x7e0002fau, 0xff091200u,             // v_mov_b32_dpp v0,v0 row_shr:2 BC:1
        0xbf810000u,
    };
    std::vector<uint32_t> spvT25b_bound = recompile_valu(
        codeT25b_bound, std::size(codeT25b_bound), 1, 0);
    CHECK(!spvT25b_bound.empty(), "recompiled T25b (bounded DPP ROW_SHR:2 mov) -> SPIR-V");
    const auto subgroupT25b = prosper::test::default_compute_subgroup_properties();
    const bool can_shuffleT25b = subgroupT25b.size &&
        (subgroupT25b.stages & VK_SHADER_STAGE_COMPUTE_BIT) &&
        (subgroupT25b.operations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT);
    if (can_shuffleT25b) {
        std::vector<float> hostExpT25b(128), hostExpT25bBound(128);
        for (uint32_t i = 0; i < 128; ++i) {
            const uint32_t lane = i % subgroupT25b.size;
            const uint32_t row_lane = lane & 15u;
            hostExpT25b[i] = row_lane == 0 ? inT25b[i] : inT25b[i - 1u];
            hostExpT25bBound[i] = row_lane < 2 ? 0.0f : inT25b[i - 2u];
        }
        const std::vector<float> gotT25b = prosper::test::run_compute(
            spvT25b, inT25b, 128, 128);
        const std::vector<float> gotT25bBound = prosper::test::run_compute(
            spvT25b_bound, inT25b, 128, 128);
        CHECK(gotT25b == hostExpT25b && gotT25bBound == hostExpT25bBound,
              "T25b: emitted shuffles and row-edge controls execute exactly at host subgroup width");
        if (subgroupT25b.size >= 16) {
            CHECK(gotT25b == expT25b && gotT25bBound == expT25b_bound,
                  "T25b: supported host preserves the architectural 16-lane DPP rows exactly");
        } else {
            std::printf("  [skip] T25b architectural execution: host subgroup %u is narrower than 16\n",
                        subgroupT25b.size);
        }
    } else {
        std::printf("  [skip] T25b execution: host lacks compute subgroup shuffle support\n");
    }

    // T25c (#1390): Plucky's second gameplay dispatch converts a permuted float after arbitrary
    // quad tables (not merely XOR swaps). Exercise its exact 0xee control: [2,3,2,3].
    const uint32_t codeT25c[] = {
        0x7e000efau, 0xff08ee00u,             // v_cvt_u32_f32_dpp v0,v0 quad_perm:[2,3,2,3]
        0xbf810000u,
    };
    const std::vector<uint32_t> spvT25c = recompile_valu(
        codeT25c, std::size(codeT25c), 1, 0);
    CHECK(!spvT25c.empty(), "recompiled T25c (Plucky DPP quad conversion) -> SPIR-V");
    CHECK(compute_spirv_min_subgroup_size(spvT25c) == 4,
          "T25c: quad-only shuffle advertises only its four-lane host contract");
    std::vector<uint32_t> spvT25cWithArithmetic = spvT25c;
    spvT25cWithArithmetic.insert(spvT25cWithArithmetic.begin() + 5,
                                 {(2u << 16) | 17u, 63u}); // OpCapability GroupNonUniformArithmetic
    CHECK(compute_spirv_min_subgroup_size(spvT25cWithArithmetic) == 4,
          "T25c: genuine subgroup capabilities do not masquerade as width metadata");
    if (can_shuffleT25b) {
        std::vector<float> inT25c(128), expectedT25c(128);
        for (uint32_t i = 0; i < 128; ++i) inT25c[i] = static_cast<float>(100u + i);
        for (uint32_t i = 0; i < 128; ++i) {
            const uint32_t selected = (0xeeu >> (2u * (i & 3u))) & 3u;
            const uint32_t converted = static_cast<uint32_t>(inT25c[(i & ~3u) | selected]);
            std::memcpy(&expectedT25c[i], &converted, sizeof(converted));
        }
        const std::vector<float> gotT25c = prosper::test::run_compute(
            spvT25c, inT25c, 128, 128);
        CHECK(gotT25c == expectedT25c,
              "T25c: arbitrary DPP quad permutation precedes float-to-uint conversion exactly");
    }

    // T25d (#1390): the reduction's final step uses the exact GFX10 V_PERMLANE16_B32 and
    // V_PERMLANEX16_B32 encodings. The first selector table duplicates 0x76543210 in both halves,
    // so lanes 8..15 gather lanes 0..7 of their own row. The cross-row form uses -1:-1, selecting
    // lane 15 of the paired row for every destination lane. Both exact words set BOUND_CTRL.
    const uint32_t codeT25d_row[] = {
        0xd7771000u, 0x03fdff00u, 0x76543210u,
        0xbf810000u,
    };
    const std::vector<uint32_t> spvT25dRow = recompile_valu(
        codeT25d_row, std::size(codeT25d_row), 1, 0);
    CHECK(!spvT25dRow.empty(),
          "recompiled T25d (Plucky V_PERMLANE16_B32 exact word) -> SPIR-V");
    CHECK(compute_spirv_min_subgroup_size(spvT25dRow) == 16,
          "T25d: within-row permlane advertises its 16-lane host contract");

    const uint32_t codeT25d_cross[] = {
        0xd7781000u, 0x03058300u,
        0xbf810000u,
    };
    const std::vector<uint32_t> spvT25dCross = recompile_valu(
        codeT25d_cross, std::size(codeT25d_cross), 1, 0);
    CHECK(!spvT25dCross.empty(),
          "recompiled T25d (Plucky V_PERMLANEX16_B32 exact word) -> SPIR-V");
    CHECK(compute_spirv_min_subgroup_size(spvT25dCross) == 32,
          "T25d: paired-row permlane advertises its 32-lane host contract");

    if (can_shuffleT25b && subgroupT25b.size >= 32) {
        std::vector<float> inT25d(128), expectedT25dRow(128), expectedT25dCross(128);
        for (uint32_t i = 0; i < 128; ++i) inT25d[i] = static_cast<float>(i);
        for (uint32_t i = 0; i < 128; ++i) {
            const uint32_t subgroup_lane = i % subgroupT25b.size;
            const uint32_t subgroup_base = i - subgroup_lane;
            const uint32_t row_base = subgroup_lane & ~15u;
            const uint32_t selected = subgroup_lane & 7u;
            expectedT25dRow[i] = inT25d[subgroup_base + row_base + selected];
            expectedT25dCross[i] = inT25d[
                subgroup_base + (row_base ^ 16u) + 15u];
        }
        const std::vector<float> gotT25dRow = prosper::test::run_compute(
            spvT25dRow, inT25d, 128, 128);
        const std::vector<float> gotT25dCross = prosper::test::run_compute(
            spvT25dCross, inT25d, 128, 128);
        CHECK(gotT25dRow == expectedT25dRow,
              "T25d: V_PERMLANE16 gathers every packed selector within its row");
        CHECK(gotT25dCross == expectedT25dCross,
              "T25d: V_PERMLANEX16 gathers lane 15 from each paired row");
    } else {
        std::printf("  [skip] T25d execution: host subgroup %u is narrower than 32\n",
                    subgroupT25b.size);
    }

    // T25e (#1390): V_READLANE_B32 reads an ordinary VGPR lane into an SGPR, independent of EXEC.
    // Plucky reads lanes 31 and 63 after its permlane reduction; exercise the exact lane-63 form and
    // move the scalar result back to v0 so the compute shell can expose it.
    const uint32_t codeT25e[] = {
        0xd7600002u, 0x00017f00u,             // v_readlane_b32 s2,v0,63
        0x7e000202u,                          // v_mov_b32 v0,s2
        0xbf810000u,
    };
    const std::vector<uint32_t> spvT25e = recompile_valu(
        codeT25e, std::size(codeT25e), 1, 0);
    CHECK(!spvT25e.empty(),
          "recompiled T25e (Plucky V_READLANE_B32 lane 63 exact word) -> SPIR-V");
    CHECK(compute_spirv_min_subgroup_size(spvT25e) == 64,
          "T25e: wave64 readlane advertises its 64-lane host contract");
    if (can_shuffleT25b && subgroupT25b.size >= 64) {
        std::vector<float> inT25e(128), expectedT25e(128);
        for (uint32_t i = 0; i < 128; ++i) inT25e[i] = static_cast<float>(i);
        for (uint32_t i = 0; i < 128; ++i) {
            const uint32_t subgroup_lane = i % subgroupT25b.size;
            const uint32_t subgroup_base = i - subgroup_lane;
            expectedT25e[i] = inT25e[subgroup_base + (subgroup_lane & ~63u) + 63u];
        }
        const std::vector<float> gotT25e = prosper::test::run_compute(
            spvT25e, inT25e, 128, 128);
        CHECK(gotT25e == expectedT25e,
              "T25e: V_READLANE_B32 broadcasts lane 63 within each guest wave64");
    } else {
        std::printf("  [skip] T25e execution: host subgroup %u is narrower than 64\n",
                    subgroupT25b.size);
    }

    // T25f (#1390): DS_SWIZZLE_B32 does not access LDS; it maps the source VGPR across active
    // lanes. The captured offset 0x020f is the architectural group32 bitmask form
    // ((lane & 15) | 16), while 0x80e4 is the identity quad selector table.
    const uint32_t codeT25f_group32[] = {
        0xd8d4020fu, 0x00000000u,             // ds_swizzle_b32 v0,v0 offset:0x020f
        0xbf810000u,
    };
    const std::vector<uint32_t> spvT25fGroup32 = recompile_valu(
        codeT25f_group32, std::size(codeT25f_group32), 1, 0);
    CHECK(!spvT25fGroup32.empty(),
          "recompiled T25f (Plucky DS_SWIZZLE_B32 group32 exact word) -> SPIR-V");
    CHECK(compute_spirv_min_subgroup_size(spvT25fGroup32) == 32,
          "T25f: group32 DS swizzle advertises its 32-lane host contract");
    const uint32_t codeT25f_quad[] = {
        0xd8d480e4u, 0x00000000u,             // ds_swizzle_b32 v0,v0 quad:[0,1,2,3]
        0xbf810000u,
    };
    const std::vector<uint32_t> spvT25fQuad = recompile_valu(
        codeT25f_quad, std::size(codeT25f_quad), 1, 0);
    CHECK(!spvT25fQuad.empty(),
          "recompiled T25f (DS_SWIZZLE_B32 quad basic mode) -> SPIR-V");
    CHECK(compute_spirv_min_subgroup_size(spvT25fQuad) == 4,
          "T25f: quad DS swizzle advertises only its four-lane host contract");
    if (can_shuffleT25b) {
        std::vector<float> inT25f(128), expectedT25f(128);
        for (uint32_t i = 0; i < 128; ++i) inT25f[i] = static_cast<float>(i);
        const std::vector<float> gotT25fQuad = prosper::test::run_compute(
            spvT25fQuad, inT25f, 128, 128);
        CHECK(gotT25fQuad == inT25f,
              "T25f: DS_SWIZZLE_B32 quad selector gathers the requested lane");
        if (subgroupT25b.size >= 32) {
            for (uint32_t i = 0; i < 128; ++i) {
                const uint32_t subgroup_lane = i % subgroupT25b.size;
                const uint32_t subgroup_base = i - subgroup_lane;
                expectedT25f[i] = inT25f[
                    subgroup_base + (subgroup_lane & ~31u) + ((subgroup_lane & 15u) | 16u)];
            }
            const std::vector<float> gotT25fGroup32 = prosper::test::run_compute(
                spvT25fGroup32, inT25f, 128, 128);
            CHECK(gotT25fGroup32 == expectedT25f,
                  "T25f: captured group32 bitmask performs the architectural lane mapping");
        } else {
            std::printf("  [skip] T25f group32 execution: host subgroup %u is narrower than 32\n",
                        subgroupT25b.size);
        }
    }

    // T26: Astro title-ship VS exact VOP3 opcode word for v_cvt_pknorm_u16_f32. The high half is
    // fixed at 0.5; the low half exercises clamping and round-to-nearest-even across [0,1].
    const uint32_t codeT26[] = {
        0xd7690002u, 0x0001e100u, // v_cvt_pknorm_u16_f32 v2, v0, 0.5
        0x7e000302u,              // v_mov_b32 v0, v2
        0xbf810000u,
    };
    std::vector<uint32_t> spvT26 = recompile_valu(codeT26, std::size(codeT26), 1, 0);
    CHECK(!spvT26.empty(), "recompiled T26 (Astro v_cvt_pknorm_u16_f32 word) -> SPIR-V");
    std::vector<float> inT26 = {-1.0f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 2.0f, 0.1f};
    std::vector<float> expT26(inT26.size());
    for (size_t i = 0; i < inT26.size(); ++i) {
        const float clamped = std::max(0.0f, std::min(1.0f, inT26[i]));
        const uint32_t lo = static_cast<uint32_t>(std::nearbyint(clamped * 65535.0f));
        const uint32_t packed = (32768u << 16) | lo;
        std::memcpy(&expT26[i], &packed, sizeof(packed));
    }
    std::vector<float> gotT26 = prosper::test::run_compute(spvT26, inT26, 8, 8);
    uint32_t badT26 = 0;
    for (size_t i = 0; i < expT26.size() && gotT26.size() == expT26.size(); ++i)
        if (gotT26[i] != expT26[i]) ++badT26;
    printf("  T26(pknorm u16) mismatches=%u (got[2]=%.0f expect=%.0f, got[3]=%.0f expect=%.0f)\n",
           badT26, gotT26.size()==8?gotT26[2]:-1.f, expT26[2],
           gotT26.size()==8?gotT26[3]:-1.f, expT26[3]);
    CHECK(gotT26.size() == expT26.size() && badT26 == 0,
          "T26: PKNORM_U16 clamps, rounds, and packs both normalized halves exactly");

    // T27: the title PS's exact SDWA operation, reduced only from v11/v5 to v0/v0 so the compute
    // harness can feed it. BYTE_0 is zero-extended before the unsigned-int-to-float conversion.
    const uint32_t codeT27[] = {0x7e000cf9u, 0x00001600u, 0xbf810000u};
    std::vector<uint32_t> spvT27 = recompile_valu(codeT27, std::size(codeT27), 1, 0);
    CHECK(!spvT27.empty(), "recompiled T27 (Astro v_cvt_f32_u32 SDWA BYTE_0) -> SPIR-V");
    const uint32_t rawT27[] = {0u, 1u, 0xffu, 0x12345678u, 0xabcdef80u, 0xffffff00u, 0x100u, 0x7fu};
    std::vector<float> inT27(std::size(rawT27)), expT27(std::size(rawT27));
    for (size_t i = 0; i < std::size(rawT27); ++i) {
        std::memcpy(&inT27[i], &rawT27[i], sizeof(uint32_t));
        expT27[i] = static_cast<float>(rawT27[i] & 0xffu);
    }
    std::vector<float> gotT27 = prosper::test::run_compute(spvT27, inT27, 8, 8);
    CHECK(gotT27 == expT27,
          "T27: SDWA BYTE_0 conversion produces the exact zero-extended byte value");

    // T27b: Plucky Squire's first gameplay compute pass uses these exact four
    // v_mul_u32_u24_sdwa packets to multiply each selected byte of v10 by the inline integer 3.
    // Verify every selector, including the scalar-source encoding in the SDWA control dword.
    const uint32_t plucky_mul_words[][2] = {
        {0x166614f9u, 0x01860683u}, // v51 = 3 * BYTE_1(v10)
        {0x161014f9u, 0x00860683u}, // v8  = 3 * BYTE_0(v10)
        {0x166214f9u, 0x02860683u}, // v49 = 3 * BYTE_2(v10)
        {0x165014f9u, 0x03860683u}, // v40 = 3 * BYTE_3(v10)
    };
    const uint32_t plucky_dst[] = {51u, 8u, 49u, 40u};
    const uint32_t plucky_cvt[] = {0x7e660d33u, 0x7e100d08u, 0x7e620d31u, 0x7e500d28u};
    const uint32_t plucky_shift[] = {8u, 0u, 16u, 24u};
    const uint32_t plucky_raw[] = {0x12345678u, 0xff008001u, 0x04030201u, 0xabcdef80u};
    for (size_t k = 0; k < std::size(plucky_mul_words); ++k) {
        const uint32_t codeT27b[] = {
            plucky_mul_words[k][0], plucky_mul_words[k][1],
            plucky_cvt[k], 0xbf810000u,
        };
        std::vector<uint32_t> spvT27b = recompile_valu(
            codeT27b, std::size(codeT27b), 11, plucky_dst[k]);
        CHECK(!spvT27b.empty(), "recompiled T27b (Plucky integer SDWA byte multiply) -> SPIR-V");
        std::vector<float> inT27b(std::size(plucky_raw) * 11u, 0.0f);
        std::vector<float> expT27b(std::size(plucky_raw));
        for (size_t i = 0; i < std::size(plucky_raw); ++i) {
            std::memcpy(&inT27b[i * 11u + 10u], &plucky_raw[i], sizeof(uint32_t));
            expT27b[i] = static_cast<float>(3u * ((plucky_raw[i] >> plucky_shift[k]) & 0xffu));
        }
        std::vector<float> gotT27b = prosper::test::run_compute(
            spvT27b, inT27b, static_cast<uint32_t>(std::size(plucky_raw)),
            static_cast<uint32_t>(std::size(plucky_raw)));
        CHECK(gotT27b == expT27b,
              "T27b: integer SDWA zero-extends the selected byte before U24 multiplication");
    }

    // T28: Astro title compute exact rejected word from the live trace. DPP applies only to src0
    // (v3); src1 (v15) stays in the current lane. With quad_perm [1,0,3,2], every lane receives its
    // horizontal neighbour plus itself.
    const uint32_t codeT28[] = {
        0x7e060300u,                         // v_mov_b32 v3, v0
        0x7e1e0300u,                         // v_mov_b32 v15, v0
        0x061e1efau, 0xff08b103u,            // exact v_add_f32_dpp v15,v3,v15 quad_perm:[1,0,3,2]
        0x7e00030fu,                         // v_mov_b32 v0, v15
        0xbf810000u,
    };
    std::vector<uint32_t> spvT28 = recompile_valu(codeT28, std::size(codeT28), 1, 0);
    CHECK(!spvT28.empty(), "recompiled T28 (Astro compute VOP2 DPP add) -> SPIR-V");
    std::vector<float> inT28(128), expT28(128);
    for (uint32_t i = 0; i < 128; ++i) inT28[i] = (float)i;
    for (uint32_t i = 0; i < 128; ++i) expT28[i] = inT28[i] + inT28[i ^ 1u];
    std::vector<float> gotT28 = prosper::test::run_compute(spvT28, inT28, 128, 128);
    uint32_t badT28 = 0;
    for (uint32_t i = 0; i < 128 && gotT28.size() == 128; ++i)
        if (gotT28[i] != expT28[i]) ++badT28;
    CHECK(gotT28.size() == 128 && badT28 == 0,
          "T28: VOP2 DPP permutes only src0 and adds current-lane src1");

    // T29: exact S_BFM_B64 word from Astro's title reduction shader.  It creates VCC with the low
    // 32 lanes set, then saveexec predicates a write.  The operation repeats independently in each
    // architectural wave64 even when the Vulkan workgroup contains two waves.
    const uint32_t codeT29[] = {
        0x7e0602ffu, 0x40e00000u,            // v_mov_b32 v3, 7.0
        0x92ea80a0u,                         // exact s_bfm_b64 vcc, 32, 0
        0xbe80246au,                         // s_and_saveexec_b64 s[0:1], vcc
        0x7e0602ffu, 0x42280000u,            // v_mov_b32 v3, 42.0 (low 32 lanes)
        0xbefe0400u,                         // s_mov_b64 exec, s[0:1]
        0x7e000303u,                         // v_mov_b32 v0, v3
        0xbf810000u,
    };
    std::vector<uint32_t> spvT29 = recompile_valu(codeT29, std::size(codeT29), 1, 0);
    CHECK(!spvT29.empty(), "recompiled T29 (Astro S_BFM_B64 VCC mask) -> SPIR-V");
    std::vector<float> inT29(128, 0.0f), expT29(128);
    for (uint32_t i = 0; i < 128; ++i) expT29[i] = (i % 64u) < 32u ? 42.0f : 7.0f;
    std::vector<float> gotT29 = prosper::test::run_compute(spvT29, inT29, 128, 128);
    CHECK(gotT29 == expT29,
          "T29: S_BFM_B64 sets exactly the requested lanes in every wave64");

    // T30: Astro's exact final reduction word narrows EXEC with inline mask 15 (lanes 0..3).
    const uint32_t codeT30[] = {
        0x7e0602ffu, 0x40e00000u,            // v_mov_b32 v3, 7.0
        0xbeea248fu,                         // exact s_and_saveexec_b64 vcc, 15
        0x7e0602ffu, 0x42280000u,            // v_mov_b32 v3, 42.0
        0xbefe046au,                         // s_mov_b64 exec, vcc (saved original EXEC)
        0x7e000303u, 0xbf810000u,
    };
    std::vector<uint32_t> spvT30 = recompile_valu(codeT30, std::size(codeT30), 1, 0);
    CHECK(!spvT30.empty(), "recompiled T30 (Astro inline B64 lane mask 15) -> SPIR-V");
    std::vector<float> inT30(128, 0.0f), expT30(128);
    for (uint32_t i = 0; i < 128; ++i) expT30[i] = (i % 64u) < 4u ? 42.0f : 7.0f;
    std::vector<float> gotT30 = prosper::test::run_compute(spvT30, inT30, 128, 128);
    CHECK(gotT30 == expT30,
          "T30: inline B64 mask 15 activates exactly four lanes per wave64");

    // T30b: Plucky Squire's exact first-gameplay compute word resets EXEC with the ORN2
    // self-complement identity while saving OLD EXEC in VCC. The following VCC restore proves the
    // destination received the saved mask rather than retaining stale compare state.
    const uint32_t codeT30b[] = {
        0x7e0602ffu, 0x40e00000u,            // v_mov_b32 v3, 7.0
        0xbeea287eu,                         // s_orn2_saveexec_b64 vcc, exec => EXEC=all-on
        0x7e0602ffu, 0x42280000u,            // v_mov_b32 v3, 42.0
        0xbefe046au,                         // s_mov_b64 exec, vcc (restore OLD EXEC)
        0x7e000303u, 0xbf810000u,
    };
    std::vector<uint32_t> spvT30b = recompile_valu(codeT30b, std::size(codeT30b), 1, 0);
    CHECK(!spvT30b.empty(),
          "recompiled T30b (Plucky s_orn2_saveexec_b64 vcc,exec) -> SPIR-V");
    std::vector<float> gotT30b = prosper::test::run_compute(spvT30b, inT30, 128, 128);
    CHECK(gotT30b == std::vector<float>(128, 42.0f),
          "T30b: ORN2 self-complement runs all lanes and VCC restores OLD EXEC");

    // Exercise the complete SAVEEXEC boolean family with a nontrivial VCC source. Each shader starts
    // with full EXEC, forms m=(lane<4), applies one boolean operation, writes 42 under the resulting
    // EXEC (7 otherwise), then restores OLD EXEC from s[4:5]. This checks both halves of the contract:
    // D receives OLD EXEC and EXEC receives the requested boolean result, for every opcode we accept.
    const uint32_t saveexec_ops[] = {0x24, 0x25, 0x26, 0x27, 0x28,
                                     0x29, 0x2a, 0x2b, 0x37, 0x38};
    const uint8_t saveexec_expect[] = {
        1, 2, 3, 0, 1, 3, 0, 1, 3, 2,
        // 0=false, 1=m, 2=true, 3=!m (OLD EXEC starts true).
    };
    std::vector<float> inT30c(128), expT30c(128);
    for (uint32_t lane = 0; lane < 128; ++lane) {
        const uint32_t local_lane = lane % 64u;
        std::memcpy(&inT30c[lane], &local_lane, 4);
    }
    uint32_t badT30c = 0;
    for (size_t k = 0; k < std::size(saveexec_ops); ++k) {
        const std::vector<uint32_t> codeT30c = {
            0x7e0402ffu, 0x40e00000u,                         // v2=7.0
            0x7e020284u,                                     // v1=4
            0x7d820300u,                                     // vcc=(v0<v1) unsigned
            0xbe84006au | (saveexec_ops[k] << 8),             // s_*_saveexec_b64 s[4:5],vcc
            0x7e0402ffu, 0x42280000u,                         // active v2=42.0
            0xbefe0404u,                                     // restore exec=s[4:5]
            0x7e000302u, 0xbf810000u,                        // output v0=v2
        };
        std::vector<uint32_t> spvT30c = recompile_valu(
            codeT30c.data(), codeT30c.size(), 1, 0);
        if (spvT30c.empty()) { ++badT30c; continue; }
        for (uint32_t lane = 0; lane < 128; ++lane) {
            const bool m = (lane % 64u) < 4u;
            const bool active = saveexec_expect[k] == 0 ? false
                              : saveexec_expect[k] == 1 ? m
                              : saveexec_expect[k] == 2 ? true : !m;
            expT30c[lane] = active ? 42.0f : 7.0f;
        }
        const std::vector<float> gotT30c = prosper::test::run_compute(
            spvT30c, inT30c, 128, 128);
        if (gotT30c != expT30c) ++badT30c;
    }
    CHECK(badT30c == 0,
          "T30c: every SAVEEXEC boolean form updates EXEC and saves OLD EXEC exactly");

    // T31: Astro's exact title-PS SDWA controls, reduced only to v0. WORD_1 is sign-extended before
    // v_cvt_f32_i32, so both positive and negative int16 values must survive the conversion.
    const uint32_t codeT31[] = {0x7e000af9u, 0x000d0600u, 0xbf810000u};
    std::vector<uint32_t> spvT31 = recompile_valu(codeT31, std::size(codeT31), 1, 0);
    CHECK(!spvT31.empty(), "recompiled T31 (Astro v_cvt_f32_i32 SDWA WORD_1+SEXT) -> SPIR-V");
    const uint32_t rawT31[] = {0x00011234u, 0x7fff0000u, 0x80000000u, 0xfffe1234u,
                               0x00000000u, 0xffffaaaau, 0x12340000u, 0xfedc0000u};
    std::vector<float> inT31(std::size(rawT31)), expT31(std::size(rawT31));
    for (size_t i = 0; i < std::size(rawT31); ++i) {
        std::memcpy(&inT31[i], &rawT31[i], sizeof(uint32_t));
        expT31[i] = static_cast<float>(static_cast<int16_t>(rawT31[i] >> 16));
    }
    std::vector<float> gotT31 = prosper::test::run_compute(spvT31, inT31, 8, 8);
    CHECK(gotT31 == expT31,
          "T31: signed SDWA conversion sign-extends the selected high word exactly");

    // T32: Astro visibility-reduction BUFFER_ATOMIC_UMAX packet — the exact live word, which has
    // GLC=0. ISA 8.1/Table 98: for atomics GLC gates "return pre-op value to VGPR"; with GLC=0
    // the memory update happens but VDATA is NOT written (it still holds the DATA operand, 42).
    // The old unconditional pre-op return silently clobbered VDATA on this live path (#882).
    const uint32_t codeT32[] = {
        0xe0e00004u, 0x80000000u,            // exact buffer_atomic_umax v0, v0, s[0:3], 0 offset:4 (GLC=0)
        0x7e000d00u,                         // v_cvt_f32_u32 v0, v0 (VDATA: still the DATA operand)
        0xbf810000u,
    };
    ShaderResourceTable rtT32;
    { ShaderResource rb{}; rb.cls = ResourceClass::ConstantBuffer; rb.format = DataFormat::Uint32;
      rb.num_components = 1; rb.binding = 3; rb.sgpr_base = 0; rtT32.resources.push_back(rb); }
    std::vector<uint32_t> spvT32 = recompile_valu(codeT32, std::size(codeT32), 1, 0, &rtT32);
    CHECK(!spvT32.empty(), "recompiled T32 (Astro buffer_atomic_umax GLC=0) -> SPIR-V");
    uint32_t forty_two = 42u; float raw_input = 0.0f;
    std::memcpy(&raw_input, &forty_two, sizeof(forty_two));
    std::vector<uint32_t> atomic_buf = {0u, 17u}, atomic_out;
    std::vector<float> gotT32 = prosper::test::run_compute(
        spvT32, {raw_input}, 1, 1, /*binding2*/{}, /*binding3*/atomic_buf, &atomic_out);
    printf("  T32 atomic got=%g memory=%u (expect 42/42)\n",
           gotT32.empty() ? -1.0f : gotT32[0], atomic_out.size() < 2 ? 0u : atomic_out[1]);
    CHECK(gotT32.size() == 1 && gotT32[0] == 42.0f && atomic_out.size() == 2 && atomic_out[1] == 42u,
          "T32: GLC=0 buffer_atomic_umax stores the max and leaves VDATA untouched");

    // T32b: the same packet with GLC=1 (bit 14 set): the pre-op memory value (17) IS returned to
    // VDATA while memory still becomes max(17, 42) = 42.
    const uint32_t codeT32b[] = {
        0xe0e04004u, 0x80000000u,            // buffer_atomic_umax v0, v0, s[0:3], 0 offset:4 glc
        0x7e000d00u,                         // v_cvt_f32_u32 v0, v0 (returned pre-op value)
        0xbf810000u,
    };
    std::vector<uint32_t> spvT32b = recompile_valu(codeT32b, std::size(codeT32b), 1, 0, &rtT32);
    CHECK(!spvT32b.empty(), "recompiled T32b (buffer_atomic_umax glc) -> SPIR-V");
    std::vector<uint32_t> atomic_buf_b = {0u, 17u}, atomic_out_b;
    std::vector<float> gotT32b = prosper::test::run_compute(
        spvT32b, {raw_input}, 1, 1, /*binding2*/{}, /*binding3*/atomic_buf_b, &atomic_out_b);
    printf("  T32b atomic got=%g memory=%u (expect 17/42)\n",
           gotT32b.empty() ? -1.0f : gotT32b[0], atomic_out_b.size() < 2 ? 0u : atomic_out_b[1]);
    CHECK(gotT32b.size() == 1 && gotT32b[0] == 17.0f && atomic_out_b.size() == 2 && atomic_out_b[1] == 42u,
          "T32b: GLC=1 buffer_atomic_umax returns the pre-op value and stores the maximum");

    // T33 (#1021): Astro title vertex shaders use the VOP1 unsigned-byte conversion family.
    // Convert all four bytes from v4 independently, then sum the exact f32 results in v0.
    const uint32_t codeT33[] = {
        0x7e002304u,                         // v_cvt_f32_ubyte0 v0, v4
        0x7e022504u,                         // v_cvt_f32_ubyte1 v1, v4
        0x7e042704u,                         // v_cvt_f32_ubyte2 v2, v4
        0x7e062904u,                         // v_cvt_f32_ubyte3 v3, v4
        0x06000300u,                         // v_add_f32 v0, v0, v1
        0x06000500u,                         // v_add_f32 v0, v0, v2
        0x06000700u,                         // v_add_f32 v0, v0, v3
        0xbf810000u,
    };
    std::vector<uint32_t> spvT33 = recompile_valu(codeT33, std::size(codeT33), 5, 0);
    CHECK(!spvT33.empty(), "recompiled T33 (Astro V_CVT_F32_UBYTE0..3) -> SPIR-V");
    const uint32_t rawT33[] = {
        0x00000000u, 0x01020304u, 0x10203040u, 0x11223344u,
        0x3f800000u, 0x80000001u, 0x7e7d7c7bu, 0xdeadbeefu,
    };
    std::vector<float> inT33(std::size(rawT33) * 5, 0.0f), expT33(std::size(rawT33));
    for (size_t i = 0; i < std::size(rawT33); ++i) {
        std::memcpy(&inT33[i * 5 + 4], &rawT33[i], sizeof(uint32_t));
        expT33[i] = static_cast<float>((rawT33[i] & 0xffu) + ((rawT33[i] >> 8) & 0xffu) +
                                       ((rawT33[i] >> 16) & 0xffu) + (rawT33[i] >> 24));
    }
    std::vector<float> gotT33 = prosper::test::run_compute(
        spvT33, inT33, static_cast<uint32_t>(std::size(rawT33)),
        static_cast<uint32_t>(std::size(rawT33)));
    CHECK(gotT33 == expT33,
          "T33: UBYTE0..3 extract the selected unsigned byte and convert every result exactly");

    // --- 2026-07 ISA-audit semantic fixes (#879/#880) — execution kernels ---

    // A1 (#879): s_not_b32 writes SCC=(result!=0). ~0 = all-ones -> SCC=1 -> cselect 11;
    // ~(-1) = 0 -> SCC=0 -> cselect 4. out = bits(a0) + 11 + 4. (llvm-mc gfx1030 encodings.)
    const uint32_t codeA1[] = { 0xbe810380u, 0xbe820701u, 0x8503848bu,
                                0xbe8103c1u, 0xbe820701u, 0x8504848bu,
                                0x4a020003u, 0x4a020204u, 0xBF810000u };
    std::vector<uint32_t> spvA1 = recompile_valu(codeA1, std::size(codeA1), 1, 1);
    CHECK(!spvA1.empty(), "recompiled A1 (s_not_b32 SCC) -> SPIR-V");
    std::vector<float> gotA1 = prosper::test::run_compute(spvA1, inX, N, N);
    uint32_t badA1 = 0;
    for (uint32_t i = 0; i < N && gotA1.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotA1[i], 4);
        if (gb != bits_of(inX[i]) + 15u) badA1++;
    }
    CHECK(gotA1.size()==N && badA1==0, "A1: s_not_b32 sets SCC=(D!=0) (1 then 0 -> +11+4)");

    // A2 (#879): s_lshl4_add_u32 writes SCC = 64-bit unsigned carry-out. 0x10000000<<4 wraps
    // (sum 0x100000001 >= 2^32) -> SCC=1 -> 11; 2<<4+1=33 in range -> SCC=0 -> 4. out = bits+15.
    const uint32_t codeA2[] = { 0xbe8103ffu, 0x10000000u, 0x98828101u, 0x8503848bu,
                                0xbe810382u, 0x98828101u, 0x8504848bu,
                                0x4a020003u, 0x4a020204u, 0xBF810000u };
    std::vector<uint32_t> spvA2 = recompile_valu(codeA2, std::size(codeA2), 1, 1);
    CHECK(!spvA2.empty(), "recompiled A2 (s_lshl4_add_u32 carry SCC) -> SPIR-V");
    std::vector<float> gotA2 = prosper::test::run_compute(spvA2, inX, N, N);
    uint32_t badA2 = 0;
    for (uint32_t i = 0; i < N && gotA2.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotA2[i], 4);
        if (gb != bits_of(inX[i]) + 15u) badA2++;
    }
    CHECK(gotA2.size()==N && badA2==0, "A2: s_lshl4_add_u32 SCC = full 64-bit carry (1 then 0)");

    // Astro world-map PC1918: s_lshl1_add_u32 uses the same full-width carry rule. A high bit in
    // 0x80000000<<1 carries out -> 11; 2<<1+1 stays in range -> 4. out = bits+15.
    const uint32_t codeA2Astro[] = { 0xbe8103ffu, 0x80000000u, 0x97028101u, 0x8503848bu,
                                 0xbe810382u, 0x97028101u, 0x8504848bu,
                                 0x4a020003u, 0x4a020204u, 0xBF810000u };
    std::vector<uint32_t> spvA2Astro = recompile_valu(
        codeA2Astro, std::size(codeA2Astro), 1, 1);
    CHECK(!spvA2Astro.empty(), "recompiled Astro s_lshl1_add_u32 carry kernel -> SPIR-V");
    std::vector<float> gotA2Astro = prosper::test::run_compute(spvA2Astro, inX, N, N);
    uint32_t badA2Astro = 0;
    for (uint32_t i = 0; i < N && gotA2Astro.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotA2Astro[i], 4);
        if (gb != bits_of(inX[i]) + 15u) badA2Astro++;
    }
    CHECK(gotA2Astro.size()==N && badA2Astro==0,
          "Astro s_lshl1_add_u32 SCC = full 64-bit carry (1 then 0)");

    // A2b (#1390): exercise the complete gfx10 s_lshl{1,2,3,4}_add_u32 opcode family. Each
    // instruction computes (1 << N) + 1 without carry; sum 3+5+9+17 = 34. The first word is the
    // exact S_LSHL1_ADD_U32 encoding used by The Plucky Squire's title fragment shader.
    const uint32_t codeA2b[] = {
        0xbe950381u,                         // s_mov_b32 s21, 1
        0x976a8115u,                         // s_lshl1_add_u32 vcc_lo, s21, 1 (live title word)
        0x97018115u,                         // s_lshl1_add_u32 s1, s21, 1
        0x97828115u,                         // s_lshl2_add_u32 s2, s21, 1
        0x98038115u,                         // s_lshl3_add_u32 s3, s21, 1
        0x98848115u,                         // s_lshl4_add_u32 s4, s21, 1
        0x7e000201u,                         // v_mov_b32 v0, s1
        0x7e020202u, 0x4a000300u,            // v0 += s2
        0x7e020203u, 0x4a000300u,            // v0 += s3
        0x7e020204u, 0x4a000300u,            // v0 += s4
        0x7e02026au, 0x4a000300u,            // v0 += live-word VCC_LO result
        0xbf810000u,
    };
    std::vector<uint32_t> spvA2b = recompile_valu(codeA2b, std::size(codeA2b), 0, 0);
    CHECK(!spvA2b.empty(), "recompiled A2b (complete S_LSHL1..4_ADD_U32 family) -> SPIR-V");
    std::vector<float> gotA2b = prosper::test::run_compute(spvA2b, std::vector<float>(1), 1, 1);
    uint32_t a2b_bits = 0; if (gotA2b.size() == 1) std::memcpy(&a2b_bits, &gotA2b[0], 4);
    CHECK(gotA2b.size() == 1 && a2b_bits == 37u,
          "A2b: live VCC_LO destination and all four shift-add results execute exactly");

    // A3 (#880): the NEG modifier is a sign-bit toggle: neg(+0.0) = -0.0. v_add_f32_e64
    // v1, -v0, -v0 with v0=+0.0 must produce -0.0 (bits 0x80000000); the old FSub(0,x)
    // lowering produced +0.0. (llvm-mc gfx1010: 0xd5030001 0x60020100.)
    const uint32_t codeA3[] = { 0xd5030001u, 0x60020100u, 0xBF810000u };
    std::vector<uint32_t> spvA3 = recompile_valu(codeA3, std::size(codeA3), 1, 1);
    CHECK(!spvA3.empty(), "recompiled A3 (VOP3 neg modifier) -> SPIR-V");
    std::vector<float> gotA3 = prosper::test::run_compute(spvA3, {0.0f}, 1, 1);
    uint32_t a3bits = 0; if (gotA3.size() == 1) std::memcpy(&a3bits, &gotA3[0], 4);
    CHECK(gotA3.size() == 1 && a3bits == 0x80000000u,
          "A3: neg(+0.0) + neg(+0.0) = -0.0 (sign-bit-exact negation)");

    // A4 (#880): v_max_f32 returns the OTHER operand when one input is NaN (NMax). With v0=NaN,
    // v_max_f32 v1, 0.5, v0 must be 0.5 — the old FMax was NaN-undefined (llvmpipe returned NaN).
    const uint32_t codeA4[] = { 0x200200f0u, 0xBF810000u };
    std::vector<uint32_t> spvA4 = recompile_valu(codeA4, std::size(codeA4), 1, 1);
    CHECK(!spvA4.empty(), "recompiled A4 (v_max_f32 NaN rule) -> SPIR-V");
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> gotA4 = prosper::test::run_compute(spvA4, {qnan}, 1, 1);
    CHECK(gotA4.size() == 1 && gotA4[0] == 0.5f,
          "A4: v_max_f32(0.5, NaN) returns the non-NaN operand");

    // A5 (#880): the CLAMP modifier maps a NaN result to 0 (DX10_CLAMP). NaN+NaN clamp -> +0.0.
    const uint32_t codeA5[] = { 0xd5038001u, 0x00020100u, 0xBF810000u };
    std::vector<uint32_t> spvA5 = recompile_valu(codeA5, std::size(codeA5), 1, 1);
    CHECK(!spvA5.empty(), "recompiled A5 (clamp NaN->0) -> SPIR-V");
    std::vector<float> gotA5 = prosper::test::run_compute(spvA5, {qnan}, 1, 1);
    uint32_t a5bits = 0xdeadbeefu; if (gotA5.size() == 1) std::memcpy(&a5bits, &gotA5[0], 4);
    CHECK(gotA5.size() == 1 && a5bits == 0u,
          "A5: v_add_f32 clamp of a NaN result clamps to +0.0 (DX10_CLAMP)");

    // A6 (#880): a PLAIN (non-SDWA) e32 f16 op writes [15:0] and PRESERVES [31:16]. v1's live
    // high half 0x1111 must survive v_add_f16_e32 v1, v0, v0 (f16 1.0+1.0=2.0 -> low 0x4000).
    // The old lowering zero-filled the high half.
    const uint32_t codeA6[] = { 0x7e0202ffu, 0x11110000u, 0x64020100u, 0xBF810000u };
    std::vector<uint32_t> spvA6 = recompile_valu(codeA6, std::size(codeA6), 1, 1);
    CHECK(!spvA6.empty(), "recompiled A6 (plain e32 f16 high-half preserve) -> SPIR-V");
    float a6in = 0.0f; { const uint32_t one_f16 = 0x00003C00u; std::memcpy(&a6in, &one_f16, 4); }
    std::vector<float> gotA6 = prosper::test::run_compute(spvA6, {a6in}, 1, 1);
    uint32_t a6bits = 0; if (gotA6.size() == 1) std::memcpy(&a6bits, &gotA6[0], 4);
    CHECK(gotA6.size() == 1 && a6bits == 0x11114000u,
          "A6: plain v_add_f16 preserves the destination's high 16 bits");

    // A7 (#879): s_bfe_u64 with an encoded WIDTH of 0 produces the architectural 0 (the old
    // lowering emitted a shift-by-64, undefined in SPIR-V). s[0:1]=~0, s4 = offset 16, width 0.
    const uint32_t codeA7[] = { 0xbe8003c1u, 0xbe8103c1u, 0xbe840390u, 0x94820400u,
                                0x4a020002u, 0xBF810000u };
    std::vector<uint32_t> spvA7 = recompile_valu(codeA7, std::size(codeA7), 1, 1);
    CHECK(!spvA7.empty(), "recompiled A7 (s_bfe_u64 width 0) -> SPIR-V");
    std::vector<float> gotA7 = prosper::test::run_compute(spvA7, inX, N, N);
    uint32_t badA7 = 0;
    for (uint32_t i = 0; i < N && gotA7.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotA7[i], 4);
        if (gb != bits_of(inX[i])) badA7++;
    }
    CHECK(gotA7.size()==N && badA7==0, "A7: s_bfe_u64 width 0 extracts 0 (defined result)");

    // A7b (#1390): NGG merged-stage prologues build an active-lane mask with s_bfe_u64 and write it
    // directly to EXEC. VCC_LO is scalar scratch here (the offset/width control 0x00080020), not a
    // VOPC mask. Offset 32 also proves the inline -1 source is correctly sign-extended to a B64
    // operand. The first eight lanes of every wave write 1.0; all other masked lanes retain 0.
    const uint32_t codeA7b[] = {
        0xbeea03ffu, 0x00080020u, // s_mov_b32 vcc_lo, width=8 offset=32
        0x94fe6ac1u,              // s_bfe_u64 exec, -1, vcc
        0x7e0202f2u,              // v_mov_b32 v1, 1.0
        0xbf810000u,
    };
    std::vector<uint32_t> spvA7b = recompile_valu(codeA7b, std::size(codeA7b), 1, 1);
    CHECK(!spvA7b.empty(), "A7b: s_bfe_u64 writing EXEC recompiles");
    std::vector<float> gotA7b = prosper::test::run_compute(spvA7b, inX, N, N);
    uint32_t badA7b = 0;
    for (uint32_t i = 0; i < N && gotA7b.size() == N; ++i)
        if (gotA7b[i] != ((i & 63u) < 8u ? 1.0f : 0.0f)) ++badA7b;
    CHECK(gotA7b.size() == N && badA7b == 0,
          "A7b: s_bfe_u64 updates EXEC with the extracted per-lane mask");

    // A7c (#1390): the same BFE destination can be VCC. Moving that mask into EXEC must consume the
    // freshly extracted VCC bits, not the stale per-lane condition that preceded the scalar write.
    const uint32_t codeA7c[] = {
        0xbeea03ffu, 0x00080000u, // s_mov_b32 vcc_lo, width=8 offset=0
        0x94ea6ac1u,              // s_bfe_u64 vcc, -1, vcc
        0xbefe046au,              // s_mov_b64 exec, vcc
        0x7e0202f2u,              // v_mov_b32 v1, 1.0
        0xbf810000u,
    };
    std::vector<uint32_t> spvA7c = recompile_valu(codeA7c, std::size(codeA7c), 1, 1);
    CHECK(!spvA7c.empty(), "A7c: s_bfe_u64 writing VCC recompiles");
    std::vector<float> gotA7c = prosper::test::run_compute(spvA7c, inX, N, N);
    uint32_t badA7c = 0;
    for (uint32_t i = 0; i < N && gotA7c.size() == N; ++i)
        if (gotA7c[i] != ((i & 63u) < 8u ? 1.0f : 0.0f)) ++badA7c;
    CHECK(gotA7c.size() == N && badA7c == 0,
          "A7c: s_bfe_u64 updates VCC and a later mask consumer observes it");

    // A8 (#880): an inline float constant in a 16-bit operand supplies the f16 encoding.
    // v_cvt_f32_f16_e32 v1, 1.0 must produce 1.0f (the old raw-f32-pattern unpack gave 0.0).
    const uint32_t codeA8[] = { 0x7e0216f2u, 0xBF810000u };
    std::vector<uint32_t> spvA8 = recompile_valu(codeA8, std::size(codeA8), 1, 1);
    CHECK(!spvA8.empty(), "recompiled A8 (v_cvt_f32_f16 inline 1.0) -> SPIR-V");
    std::vector<float> gotA8 = prosper::test::run_compute(spvA8, {0.0f}, 1, 1);
    CHECK(gotA8.size() == 1 && gotA8[0] == 1.0f,
          "A8: v_cvt_f32_f16 of inline 1.0 reads the f16 pattern 0x3C00 -> 1.0f");

    // A9 (#879): SCC POISON. A 64-bit mask op (s_and_b64 s[0:1],vcc,vcc) writes SCC = a cross-lane
    // reduction the per-invocation model cannot form, so it poisons rs.scc. A NON-ADJACENT SCC
    // consumer (s_cselect_b32, separated by an s_mov_b32 that does not touch SCC) must then reject
    // the whole shader fail-visibly instead of silently selecting on the value the (absent) earlier
    // s_cmp would have left. Recompile returns empty. (llvm-mc gfx1030 encodings.)
    const uint32_t codeA9[] = {
        0x87806a6au,   // s_and_b64 s[0:1], vcc, vcc  (poisons SCC)
        0xbe820385u,   // s_mov_b32 s2, 5             (non-adjacent filler; no SCC write)
        0x8503848bu,   // s_cselect_b32 s3, 11, 4     (reads poisoned SCC -> reject)
        0x4a020203u,   // v_add_nc_u32 v1, s3, v1
        0xBF810000u,
    };
    std::vector<uint32_t> spvA9 = recompile_valu(codeA9, std::size(codeA9), 1, 1);
    CHECK(spvA9.empty(),
          "A9: a non-adjacent SCC consumer after a 64-bit mask op is REJECTED (SCC poison)");

    // A9b: the SAME consumer after a real s_cmp (which re-arms SCC) still recompiles — the poison
    // only rejects when the last SCC writer was an unrepresentable mask op, not in general.
    const uint32_t codeA9b[] = {
        0x87806a6au,   // s_and_b64 s[0:1], vcc, vcc  (poisons SCC)
        0xbf068000u,   // s_cmp_eq_u32 s0, 0          (re-arms SCC with a representable value)
        0x8503848bu,   // s_cselect_b32 s3, 11, 4     (reads the fresh SCC -> OK)
        0x4a020203u,   // v_add_nc_u32 v1, s3, v1
        0xBF810000u,
    };
    std::vector<uint32_t> spvA9b = recompile_valu(codeA9b, std::size(codeA9b), 1, 1);
    CHECK(!spvA9b.empty(),
          "A9b: a real s_cmp between the mask op and the consumer re-arms SCC (recompiles)");

    // A10 (#458): a fixed-offset compiler spill/fill preserves the exact per-invocation VGPR bits.
    const uint32_t codeA10[] = {
        0xdc704010u, 0x00000000u, // scratch_store_dword off, v0, s0 offset:16
        0x7e000280u,              // v_mov_b32 v0, 0
        0xdc304010u, 0x00000000u, // scratch_load_dword v0, off, s0 offset:16
        0xBF810000u,
    };
    std::vector<uint32_t> spvA10 = recompile_valu(codeA10, std::size(codeA10), 1, 0);
    CHECK(!spvA10.empty(), "A10: fixed-offset private dword spill/fill recompiles");
    const std::vector<float> scratch_in = {-7.25f, 0.0f, 3.5f, 1234.0f};
    std::vector<float> gotA10 = prosper::test::run_compute(
        spvA10, scratch_in, static_cast<uint32_t>(scratch_in.size()),
        static_cast<uint32_t>(scratch_in.size()));
    CHECK(gotA10 == scratch_in,
          "A10: private dword spill/fill round-trips exact values for every invocation");

    // A11 (#458): vector forms use consecutive VGPRs while retaining one private offset range.
    const uint32_t codeA11[] = {
        0xdc744020u, 0x00060000u, // scratch_store_dwordx2 off, v[0:1], s6 offset:32
        0x7e000280u, 0x7e020280u, // clear v0/v1
        0xdc344020u, 0x00060000u, // scratch_load_dwordx2 v[0:1], off, s6 offset:32
        0xBF810000u,
    };
    std::vector<uint32_t> spvA11 = recompile_valu(codeA11, std::size(codeA11), 2, 1);
    CHECK(!spvA11.empty(), "A11: fixed-offset private dwordx2 spill/fill recompiles");
    const std::vector<float> scratch_pair_in = {
        1.0f, 11.0f, 2.0f, 12.0f, 3.0f, 13.0f, 4.0f, 14.0f,
    };
    const std::vector<float> scratch_pair_expect = {11.0f, 12.0f, 13.0f, 14.0f};
    std::vector<float> gotA11 = prosper::test::run_compute(spvA11, scratch_pair_in, 4, 4);
    CHECK(gotA11 == scratch_pair_expect,
          "A11: private dwordx2 spill/fill restores the second consecutive VGPR");

    // A12 (#458): a signed 16-bit spill crossing a dword boundary is reconstructed exactly.
    const uint32_t codeA12[] = {
        0x7e0002ffu, 0x00008001u, // v_mov_b32 v0, 0x8001
        0xdc684003u, 0x00000000u, // scratch_store_short off, v0, s0 offset:3
        0x7e000280u,              // v_mov_b32 v0, 0
        0xdc2c4003u, 0x00000000u, // scratch_load_sshort v0, off, s0 offset:3
        0x7e000b00u,              // v_cvt_f32_i32 v0, v0
        0xBF810000u,
    };
    std::vector<uint32_t> spvA12 = recompile_valu(codeA12, std::size(codeA12), 0, 0);
    CHECK(!spvA12.empty(), "A12: unaligned signed private short spill/fill recompiles");
    std::vector<float> gotA12 = prosper::test::run_compute(spvA12, {0.0f}, 1, 1);
    CHECK(gotA12.size() == 1 && gotA12[0] == -32767.0f,
          "A12: straddling signed private short reloads with correct sign extension");

    // NESTED divergent EXEC-exit loops in COMPUTE (#590 — DOLL's last post-process kernel shape:
    // an inner table loop entirely inside an outer row loop, both v_cmpx/execz-exit with backward
    // s_branch back-edges, s_barrier AFTER the loops). Per-invocation lowering: this invocation
    // iterates while its own EXEC bit holds. outer 3 iterations x inner 4 iterations accumulate
    // v3 += 0.0625 twelve times = 0.75; the outer body adds v2 += 0.25 three times = 0.75.
    {
        const uint32_t nested_cs[] = {
            0x7E020283u,               //  0: v_mov_b32 v1, 3        (outer bound)
            0x7E080284u,               //  1: v_mov_b32 v4, 4        (inner bound)
            0xBE800380u,               //  2: s_mov_b32 s0, 0        (outer counter)
            0x7E040280u,               //  3: v_mov_b32 v2, 0
            0x7E060280u,               //  4: v_mov_b32 v3, 0
            0x7E0A02F2u,               //  5: v_mov_b32 v5, 1.0
            0xBE82047Eu,               //  6: s_mov_b64 s[2:3], exec (outer save)
            0x7DA20200u,               //  7: OUTER_HDR: v_cmpx_lt_u32 s0, v1
            0xBF88000Du,               //  8: s_cbranch_execz +13 -> 22 (OUTER_EXIT)
            0xBE810380u,               //  9: s_mov_b32 s1, 0        (inner counter reset)
            0xBE84047Eu,               // 10: s_mov_b64 s[4:5], exec (inner save)
            0x7DA20801u,               // 11: INNER_HDR: v_cmpx_lt_u32 s1, v4
            0xBF880004u,               // 12: s_cbranch_execz +4 -> 17 (INNER_EXIT)
            0x060606FFu, 0x3D800000u,  // 13: v_add_f32 v3, 0.0625, v3
            0x81018101u,               // 15: s_add_i32 s1, s1, 1
            0xBF82FFFAu,               // 16: s_branch -6 -> 11 (INNER back-edge)
            0xBEFE0404u,               // 17: INNER_EXIT: s_mov_b64 exec, s[4:5]
            0x060404FFu, 0x3E800000u,  // 18: v_add_f32 v2, 0.25, v2
            0x81008100u,               // 20: s_add_i32 s0, s0, 1
            0xBF82FFF1u,               // 21: s_branch -15 -> 7 (OUTER back-edge)
            0xBEFE0402u,               // 22: OUTER_EXIT: s_mov_b64 exec, s[2:3]
            0xBF8A0000u,               // 23: s_barrier (post-loop, uniform: reconverged merge)
            0xBF810000u,               // 24: s_endpgm
        };
        std::vector<uint32_t> spvNL = recompile_valu(nested_cs, std::size(nested_cs), 1, 3);
        CHECK(!spvNL.empty(), "#590: recompiled NESTED exec-exit loops (compute) -> SPIR-V");
        std::vector<float> inNL(64, 0.0f);
        std::vector<float> gotNL = prosper::test::run_compute(spvNL, inNL, 64, 64);
        uint32_t badNL = 0;
        for (uint32_t i = 0; i < 64 && gotNL.size() == 64; ++i)
            if (std::fabs(gotNL[i] - 0.75f) > 1e-3f) ++badNL;
        if (gotNL.size() == 64)
            printf("  nested-loop out[0]=%g out[63]=%g expect 0.75\n", gotNL[0], gotNL[63]);
        CHECK(gotNL.size() == 64 && badNL == 0,
              "#590: nested loops run 3x4 iterations (inner accumulates 12 x 0.0625 = 0.75)");

        // The outer accumulator proves the OUTER body's tail (after the inner loop's exit + exec
        // restore) executes exactly once per outer iteration.
        std::vector<uint32_t> spvNL2 = recompile_valu(nested_cs, std::size(nested_cs), 1, 2);
        std::vector<float> gotNL2 = prosper::test::run_compute(spvNL2, inNL, 64, 64);
        CHECK(gotNL2.size() == 64 && std::fabs(gotNL2[0] - 0.75f) < 1e-3f,
              "#590: outer body tail runs 3 times (3 x 0.25 = 0.75) after each inner-loop exit");

        // DOLL's title kernel combines those nested loops and a post-loop workgroup barrier with a
        // top-level scalar VCC branch. The generic exact-wave dispatcher cannot contain a barrier,
        // so this shape must retain structured loops and reduce the branch through guest-wave LDS.
        // Workgroup 0 has no VCC lane; workgroup 1 has only lane 63 set, proving that the vote covers
        // the complete 64-lane guest wave rather than the host's potentially narrower subgroup.
        const uint32_t structured_wave_cs[] = {
            0x7E040280u,               //  0: v_mov_b32 v2, 0
            0x7C020300u,               //  1: v_cmp_lt_f32 vcc, v0, v1
            0xBF860001u,               //  2: s_cbranch_vccz +1 -> 4
            0x7E040281u,               //  3: v_mov_b32 v2, 1
            0x7E020283u,               //  4: v_mov_b32 v1, 3
            0x7E080284u,               //  5: v_mov_b32 v4, 4
            0xBE800380u,               //  6: s_mov_b32 s0, 0
            0x7E060280u,               //  7: v_mov_b32 v3, 0
            0x7E0A02F2u,               //  8: v_mov_b32 v5, 1.0
            0xBE82047Eu,               //  9: s_mov_b64 s[2:3], exec
            0x7DA20200u,               // 10: OUTER_HDR: v_cmpx_lt_u32 s0, v1
            0xBF88000Du,               // 11: s_cbranch_execz +13 -> 25
            0xBE810380u,               // 12: s_mov_b32 s1, 0
            0xBE84047Eu,               // 13: s_mov_b64 s[4:5], exec
            0x7DA20801u,               // 14: INNER_HDR: v_cmpx_lt_u32 s1, v4
            0xBF880004u,               // 15: s_cbranch_execz +4 -> 20
            0x060606FFu, 0x3D800000u,  // 16: v_add_f32 v3, 0.0625, v3
            0x81018101u,               // 18: s_add_i32 s1, s1, 1
            0xBF82FFFAu,               // 19: s_branch -6 -> 14
            0xBEFE0404u,               // 20: s_mov_b64 exec, s[4:5]
            0x060606FFu, 0x3E800000u,  // 21: v_add_f32 v3, 0.25, v3
            0x81008100u,               // 23: s_add_i32 s0, s0, 1
            0xBF82FFF1u,               // 24: s_branch -15 -> 10
            0xBEFE0402u,               // 25: s_mov_b64 exec, s[2:3]
            0xBF8A0000u,               // 26: s_barrier
            0x7E040D02u,               // 27: v_cvt_f32_u32 v2, v2
            0xBF810000u,               // 28: s_endpgm
        };
        std::vector<uint32_t> spvStructuredWave = recompile_valu(
            structured_wave_cs, std::size(structured_wave_cs), 2, 2);
        CHECK(!spvStructuredWave.empty(),
              "#1373: nested loops + barrier + top-level exact-wave branch recompile");
        std::vector<float> inStructuredWave(128 * 2), expStructuredWave(128);
        for (uint32_t i = 0; i < 128; ++i) {
            inStructuredWave[i * 2] = (i == 127) ? 0.0f : 1.0f;
            inStructuredWave[i * 2 + 1] = 0.5f;
            expStructuredWave[i] = i < 64 ? 0.0f : 1.0f;
        }
        const std::vector<float> gotStructuredWave = prosper::test::run_compute(
            spvStructuredWave, inStructuredWave, 128, 128);
        uint32_t badStructuredWave = 0;
        for (uint32_t i = 0; i < 128 && gotStructuredWave.size() == 128; ++i)
            if (gotStructuredWave[i] != expStructuredWave[i]) ++badStructuredWave;
        if (gotStructuredWave.size() == 128)
            printf("  structured-wave mismatches=%u out[0/63/64/127]=%g/%g/%g/%g\n",
                   badStructuredWave, gotStructuredWave[0], gotStructuredWave[63],
                   gotStructuredWave[64], gotStructuredWave[127]);
        CHECK(gotStructuredWave.size() == 128 && badStructuredWave == 0,
              "#1373: structured branch performs one exact 64-lane guest-wave vote");

        ComputeShaderConfig nativeStructuredWaveConfig;
        nativeStructuredWaveConfig.local_x = 8;
        nativeStructuredWaveConfig.local_y = 8;
        nativeStructuredWaveConfig.wave_size = 64;
        nativeStructuredWaveConfig.native_subgroup_size = 64;
        const std::vector<uint32_t> nativeStructuredWaveSpv = recompile_compute(
            structured_wave_cs, std::size(structured_wave_cs), nullptr,
            nativeStructuredWaveConfig);
        CHECK(!nativeStructuredWaveSpv.empty() &&
                  count_spirv_opcode(nativeStructuredWaveSpv, 335) >= 1,
              "structured exact-wave branch lowers its vote to subgroup-any");
        CHECK(count_spirv_opcode(nativeStructuredWaveSpv, 224) == 1,
              "structured exact-wave branch retains only its guest workgroup barrier");

        // UE4 reduction kernels also use sequential top-level EXEC tests separated by workgroup
        // barriers, with no loop in the shader. A scalar write in each arm makes the result visible
        // to every invocation in the complete guest wave, not merely the host subgroup containing
        // the one active lane. This proves the structured exact-wave path independently of loops.
        const uint32_t barrier_reduction_cs[] = {
            0xBE800380u,               //  0: s_mov_b32 s0, 0
            0xBE810380u,               //  1: s_mov_b32 s1, 0
            0x7DA20200u,               //  2: v_cmpx_lt_u32 exec, s0, v1
            0xBF880001u,               //  3: s_cbranch_execz +1 -> 5
            0xBE800381u,               //  4: s_mov_b32 s0, 1 (wave-visible arm result)
            0xBEFE04C1u,               //  5: s_mov_b64 exec, -1
            0xBF8A0000u,               //  6: s_barrier
            0x7DA20600u,               //  7: v_cmpx_lt_u32 exec, s0, v3
            0xBF880001u,               //  8: s_cbranch_execz +1 -> 10
            0xBE810382u,               //  9: s_mov_b32 s1, 2 (wave-visible arm result)
            0xBEFE04C1u,               // 10: s_mov_b64 exec, -1
            0xBF8A0000u,               // 11: s_barrier
            0x80020100u,               // 12: s_add_u32 s2, s0, s1
            0x7E040C02u,               // 13: v_cvt_f32_u32 v2, s2
            0xBF810000u,               // 14: s_endpgm
        };
        std::vector<uint32_t> spvBarrierReduction = recompile_valu(
            barrier_reduction_cs, std::size(barrier_reduction_cs), 4, 2);
        CHECK(!spvBarrierReduction.empty(),
              "barrier-separated top-level wave reductions recompile without a loop");
        std::vector<float> inBarrierReduction(128 * 4, 0.0f);
        std::vector<float> expBarrierReduction(128);
        for (uint32_t i = 0; i < 128; ++i) {
            inBarrierReduction[i * 4 + 1] = i == 63 ? 1.0f : 0.0f;   // v1
            inBarrierReduction[i * 4 + 3] = i == 127 ? 1.0f : 0.0f;  // v3
            expBarrierReduction[i] = i < 64 ? 1.0f : 2.0f;
        }
        const std::vector<float> gotBarrierReduction = prosper::test::run_compute(
            spvBarrierReduction, inBarrierReduction, 128, 128);
        uint32_t badBarrierReduction = 0;
        for (uint32_t i = 0; i < 128 && gotBarrierReduction.size() == 128; ++i)
            if (gotBarrierReduction[i] != expBarrierReduction[i]) ++badBarrierReduction;
        CHECK(gotBarrierReduction.size() == 128 && badBarrierReduction == 0,
              "barrier-separated reductions perform independent exact 64-lane guest-wave votes");

        // The automatic multi-wave policy must follow translated structure, not raw opcode counts.
        // Four valid sequential top-level wave votes each cost two synchronized scratch barriers in
        // portable lowering and none in exact-subgroup lowering. The two-vote kernel above remains
        // below the conservative threshold.
        const uint32_t vote_heavy_policy_cs[] = {
            0xBE800380u,               //  0: s_mov_b32 s0, 0
            0xBE810380u,               //  1: s_mov_b32 s1, 0
            0x7DA20200u,               //  2: v_cmpx_lt_u32 exec, s0, v1
            0xBF880001u,               //  3: s_cbranch_execz +1 -> 5
            0xBE800381u,               //  4: s_mov_b32 s0, 1
            0xBEFE04C1u,               //  5: s_mov_b64 exec, -1
            0xBF8A0000u,               //  6: s_barrier
            0x7DA20600u,               //  7: v_cmpx_lt_u32 exec, s0, v3
            0xBF880001u,               //  8: s_cbranch_execz +1 -> 10
            0xBE810382u,               //  9: s_mov_b32 s1, 2
            0xBEFE04C1u,               // 10: s_mov_b64 exec, -1
            0xBF8A0000u,               // 11: s_barrier
            0x7DA20200u,               // 12: v_cmpx_lt_u32 exec, s0, v1
            0xBF880001u,               // 13: s_cbranch_execz +1 -> 15
            0xBE800383u,               // 14: s_mov_b32 s0, 3
            0xBEFE04C1u,               // 15: s_mov_b64 exec, -1
            0xBF8A0000u,               // 16: s_barrier
            0x7DA20600u,               // 17: v_cmpx_lt_u32 exec, s0, v3
            0xBF880001u,               // 18: s_cbranch_execz +1 -> 20
            0xBE810384u,               // 19: s_mov_b32 s1, 4
            0xBEFE04C1u,               // 20: s_mov_b64 exec, -1
            0xBF8A0000u,               // 21: s_barrier
            0x80020100u,               // 22: s_add_u32 s2, s0, s1
            0x7E040C02u,               // 23: v_cvt_f32_u32 v2, s2
            0xBF810000u,               // 24: s_endpgm
        };
        ComputeShaderConfig portableVotePolicyConfig;
        portableVotePolicyConfig.local_x = 8;
        portableVotePolicyConfig.local_y = 8;
        portableVotePolicyConfig.wave_size = 64;
        ComputeShaderConfig nativeVotePolicyConfig = portableVotePolicyConfig;
        nativeVotePolicyConfig.native_subgroup_size = 64;
        const std::vector<uint32_t> portableVotePolicySpv = recompile_compute(
            vote_heavy_policy_cs, std::size(vote_heavy_policy_cs), nullptr,
            portableVotePolicyConfig);
        const std::vector<uint32_t> nativeVotePolicySpv = recompile_compute(
            vote_heavy_policy_cs, std::size(vote_heavy_policy_cs), nullptr,
            nativeVotePolicyConfig);
        const uint32_t linearized_vote_policy_cs[] = {
            0xBF8A0000u,               //  0: s_barrier (uniform, before all guarded work)
            0x7DA20200u,               //  1: v_cmpx_lt_u32 exec, s0, v1
            0xBF880007u,               //  2: s_cbranch_execz +7 -> 10 (safe guard to end)
            0x7DA20200u,               //  3: v_cmpx_lt_u32 exec, s0, v1
            0xBF880005u,               //  4: s_cbranch_execz +5 -> 10 (safe guard to end)
            0x7DA20200u,               //  5: v_cmpx_lt_u32 exec, s0, v1
            0xBF880003u,               //  6: s_cbranch_execz +3 -> 10 (safe guard to end)
            0x7DA20200u,               //  7: v_cmpx_lt_u32 exec, s0, v1
            0xBF880001u,               //  8: s_cbranch_execz +1 -> 10 (safe guard to end)
            0x7E040C00u,               //  9: v_cvt_f32_u32 v2, s0
            0xBF810000u,               // 10: s_endpgm
        };
        const std::vector<uint32_t> linearizedVotePolicySpv = recompile_compute(
            linearized_vote_policy_cs, std::size(linearized_vote_policy_cs), nullptr,
            portableVotePolicyConfig);
        CHECK(compute_shader_prefers_native_multiwave(
                  vote_heavy_policy_cs, std::size(vote_heavy_policy_cs)) &&
                  !compute_shader_prefers_native_multiwave(
                      barrier_reduction_cs, std::size(barrier_reduction_cs)) &&
                  !compute_shader_prefers_native_multiwave(
                      linearized_vote_policy_cs, std::size(linearized_vote_policy_cs)),
              "multi-wave vote policy counts accepted structured wave branches");
        CHECK(!portableVotePolicySpv.empty() && !nativeVotePolicySpv.empty() &&
                  count_spirv_opcode(portableVotePolicySpv, 224) ==
                      count_spirv_opcode(nativeVotePolicySpv, 224) + 8,
              "multi-wave vote policy proves exact lowering removes eight scratch barriers");
        CHECK(!linearizedVotePolicySpv.empty() &&
                  count_spirv_opcode(linearizedVotePolicySpv, 224) == 1,
              "raw branch density alone does not opt a linearized shader into multi-wave mode");

        std::vector<uint32_t> divergentBarrierStructuredWave(
            std::begin(structured_wave_cs), std::end(structured_wave_cs));
        divergentBarrierStructuredWave[3] = 0xBF8A0000u; // s_barrier inside the top-level VCC arm
        CHECK(recompile_valu(divergentBarrierStructuredWave.data(),
                             divergentBarrierStructuredWave.size(), 2, 2).empty(),
              "#1373: structured exact-wave path rejects a barrier in a conditional arm");
    }

    // PARTIALLY OVERLAPPING loops (back-edge into another loop's body without proper nesting) must
    // not be accepted by the STRUCTURIZER. In compute they may legitimately fall to the CFG
    // state-machine dispatcher (faithful wave emulation) — so this stream carries a mid-region
    // s_barrier, which disqualifies the dispatcher too: the only remaining outcome is a loud reject.
    {
        const uint32_t overlap_cs[] = {
            0xBE800380u,               //  0: s_mov_b32 s0, 0
            0x7E020284u,               //  1: v_mov_b32 v1, 4
            0x7DA20200u,               //  2: A_HDR: v_cmpx_lt_u32 s0, v1
            0xBF880007u,               //  3: s_cbranch_execz +7 -> 11
            0x7DA20200u,               //  4: B_HDR: v_cmpx_lt_u32 s0, v1
            0xBF880007u,               //  5: s_cbranch_execz +7 -> 13
            0xBF8A0000u,               //  6: s_barrier (disqualifies the dispatcher fallback)
            0x81008100u,               //  7: s0++
            0xBF82FFF9u,               //  8: s_branch -7 -> 2 (A back-edge; A=[2,8], exit 9)
            0x81008100u,               //  9: s0++
            0xBF82FFF9u,               // 10: s_branch -7 -> 4 (B back-edge; B=[4,10] overlaps A, not nested)
            0x7E040280u,               // 11: v_mov_b32 v2, 0
            0x7E040280u,               // 12: v_mov_b32 v2, 0
            0xBF810000u,               // 13: s_endpgm
        };
        CHECK(recompile_valu(overlap_cs, std::size(overlap_cs), 1, 2).empty(),
              "#590: partially-overlapping loops remain REJECTED (unstructured region)");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
