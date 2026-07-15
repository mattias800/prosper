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
          "compute execz uses subgroupAny(EXEC): empty wave skips, active wave enters");

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

    // Kernel 24b (#150): the SAME unorm8x4 fetch but with a NON-dword-aligned inst offset (offset:2).
    // The packed unpack extracts components at static byte offsets from a dword-aligned base and drops
    // addr&3, so a non-aligned element base would decode the wrong bits — it must be REJECTED (the
    // alignment can't be proven) rather than silently mis-decoded. Kernel 24 (offset 0) still succeeds.
    const uint32_t code24b[] = { 0x7e000f00u, 0xe00c2002u, 0x80020100u, 0xbf810000u };
    CHECK(recompile_valu(code24b, sizeof(code24b)/sizeof(code24b[0]), 1, /*out_vgpr*/1, &rt24).empty(),
          "kernel 24b (packed unorm8x4 at a non-dword-aligned offset:2) is REJECTED (not mis-decoded)");

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
    // (s_nop* then s_endpgm). Real code at the target (an else-block) used to be silently
    // discarded by a blanket clamp — valid SPIR-V, wrong semantics; it must now REJECT.
    // Shape: s0=7; s1=7|8; s_cmp_eq_u32; s_cbranch_scc0 +3 (-> pc7); v0=42.0; s_endpgm; <target...>
    //   J1: target is s_endpgm            -> accept; SCC=1 runs the block (42), SCC=0 skips (input)
    //   J2: target is REAL code (v0=9.0)  -> REJECT (was: clamped, taken path silently lost v0=9)
    //   J3: target is s_nop; s_endpgm     -> accept (nop padding before the end is still an early-out)
    // Encodings llvm-mc gfx1030 verified.
    const uint32_t codeJ1t[] = { 0xbe800387u, 0xbe810387u, 0xbf060100u, 0xbf840003u,
                                 0x7e0002ffu, 0x42280000u, 0xbf810000u, 0xbf810000u };
    const uint32_t codeJ1f[] = { 0xbe800387u, 0xbe810388u, 0xbf060100u, 0xbf840003u,
                                 0x7e0002ffu, 0x42280000u, 0xbf810000u, 0xbf810000u };
    const uint32_t codeJ2[]  = { 0xbe800387u, 0xbe810388u, 0xbf060100u, 0xbf840003u,
                                 0x7e0002ffu, 0x42280000u, 0xbf810000u,
                                 0x7e0002ffu, 0x41100000u, 0xbf810000u };
    const uint32_t codeJ3[]  = { 0xbe800387u, 0xbe810387u, 0xbf060100u, 0xbf840003u,
                                 0x7e0002ffu, 0x42280000u, 0xbf810000u, 0xbf800000u, 0xbf810000u };
    std::vector<uint32_t> spvJ1t = recompile_valu(codeJ1t, sizeof(codeJ1t)/sizeof(codeJ1t[0]), 1, 0);
    std::vector<uint32_t> spvJ1f = recompile_valu(codeJ1f, sizeof(codeJ1f)/sizeof(codeJ1f[0]), 1, 0);
    std::vector<uint32_t> spvJ2  = recompile_valu(codeJ2,  sizeof(codeJ2)/sizeof(codeJ2[0]),  1, 0);
    std::vector<uint32_t> spvJ3  = recompile_valu(codeJ3,  sizeof(codeJ3)/sizeof(codeJ3[0]),  1, 0);
    CHECK(!spvJ1t.empty() && !spvJ1f.empty(), "kernel J1 (early-out branch to a trailing s_endpgm) -> SPIR-V");
    CHECK(spvJ2.empty(), "kernel J2 (branch past s_endpgm into REAL code) is REJECTED, not clamped");
    CHECK(!spvJ3.empty(), "kernel J3 (early-out through s_nop padding) -> SPIR-V");
    std::vector<float> inJ(N);
    for (uint32_t i = 0; i < N; i++) inJ[i] = (float)i * 0.25f;
    std::vector<float> gotJt = prosper::test::run_compute(spvJ1t, inJ, N, N);
    std::vector<float> gotJf = prosper::test::run_compute(spvJ1f, inJ, N, N);
    uint32_t badJ = 0;
    for (uint32_t i = 0; i < N && gotJt.size() == N; i++) if (std::fabs(gotJt[i] - 42.0f) > 1e-3f) badJ++;
    for (uint32_t i = 0; i < N && gotJf.size() == N; i++) if (std::fabs(gotJf[i] - inJ[i]) > 1e-3f) badJ++;
    printf("  kernelJ mismatches=%u (scc1->%g expect 42, scc0->%g expect %g)\n", badJ,
           gotJt.size()==N?gotJt[8]:-1, gotJf.size()==N?gotJf[8]:-1, inJ[8]);
    CHECK(gotJt.size()==N && gotJf.size()==N && badJ==0,
          "kernel J1 early-out: block runs on SCC=1 (42), skipped on SCC=0 (input)");

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
    const float satIn[] = { 3.7f, 0.0f, -0.5f, -5.5f, 255.9f, -1e10f, 1e10f, 4294967040.0f,
                            2147483520.0f, std::nanf(""), INFINITY, -INFINITY };
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
           gotS1.size()==N?gotS1[9]:-1, gotS1.size()==N?gotS1[3]:-1, gotS1.size()==N?gotS1[6]:-1, expS1[6]);
    printf("  kernelS2(i32 sat) mismatches=%u (nan->%g, -1e10->%g, 1e10->%g expect 0,%g,%g)\n", badS2,
           gotS2.size()==N?gotS2[9]:-1, gotS2.size()==N?gotS2[5]:-1, gotS2.size()==N?gotS2[6]:-1, expS2[5], expS2[6]);
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

    // Kernel T5b (#397): s_max SCC on a TIE. s_max_i32 s2,5,5 -> SCC=(5>=5)=1 per RDNA2 ISA (the
    // asymmetric min/max split: min uses strict `<`, max uses non-strict `>=`). s_cselect_b32 s3 =
    // SCC ? 11 : 4, then v1 = a0 + s3. Locks SCC=1 on equality: out bits = bits(a0) + 11. The old
    // strict `>` set SCC=0 on the tie -> s3=4 -> bits(a0)+4, which this catches. (llvm-mc gfx1010.)
    const uint32_t codeT5b[] = { 0x84028585u, 0x8503848bu, 0x4a020003u, 0xBF810000u };
    std::vector<uint32_t> spvT5b = recompile_valu(codeT5b, sizeof(codeT5b)/4, 1, 1);
    CHECK(!spvT5b.empty(), "recompiled T5b (s_max_i32 SCC on a tie + s_cselect) -> SPIR-V");
    std::vector<float> gotT5b = prosper::test::run_compute(spvT5b, inX, N, N);
    uint32_t badT5b = 0;
    for (uint32_t i = 0; i < N && gotT5b.size() == N; i++) {
        uint32_t gb; std::memcpy(&gb, &gotT5b[i], 4);
        if (gb != bits_of(inX[i]) + 11u) badT5b++;
    }
    CHECK(gotT5b.size()==N && badT5b==0, "T5b: s_max_i32(5,5) sets SCC=1 (>=), s_cselect picks 11 not 4");

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

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
