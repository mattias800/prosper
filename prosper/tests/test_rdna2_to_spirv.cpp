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
#include "../src/gpu/shader_resources.hpp"
#include "compute_runner.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
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

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
