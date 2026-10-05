// test_rdna2_scalar_alu.cpp -- the scalar-ALU kernel assertions (kernels 5-9 plus the SOP2
// complement/SUBB contract arms), moved VERBATIM out of test_rdna2_to_spirv.cpp's main(). The
// block was self-contained: it referenced no declaration from main's scope except the harness
// size N (replicated below), which is what made it movable without a rewrite.
//
// The invariant that makes the move checkable is the executed-assertion count. main() records
// the count before calling run_scalar_alu_checks() and requires the delta to be exactly 22, so
// a block that failed to arrive fails the test instead of quietly shrinking the suite. Verified
// by mutation: an empty run_scalar_alu_checks() exits 1 with "contributed 0 checks, expected
// 22". (21 CHECK statements; the VCC half-merge arm runs twice, so 22 executions.)
//
// The declarations below this comment are the original file's preamble, replicated so this
// translation unit compiles. Most of it is unused here and is kept identical rather than pruned,
// so the two files' shared prologue stays comparable.

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

static uint32_t bits_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// Harness size, mirroring main's `const uint32_t N = 128;`.
const uint32_t N = 128;

void run_scalar_alu_checks() {
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

    // GTA V compute program 0x413cee500 first rejects on the exact pc46 S_ANDN2_B32 packet below.
    // Exercise the complete complementary B32 logical family numerically and fold SCC into bit 31
    // of the observed result so both the data operation and SCC=(D!=0) are checked by execution.
    auto run_sop2_complement = [&](uint32_t opcode_word, uint32_t a, uint32_t c) {
        const uint32_t code[] = {
            0xbe8003ffu, a,       // s_mov_b32 s0,a
            0xbe8103ffu, c,       // s_mov_b32 s1,c
            opcode_word,          // s_*_b32 s2,s0,s1
            0x85038081u,          // s_cselect_b32 s3,1,0
            0x8f039f03u,          // s_lshl_b32 s3,s3,31
            0x89020302u,          // s_xor_b32 s2,s2,s3
            0x7e000202u,          // v_mov_b32 v0,s2
            0xbf810000u,
        };
        const std::vector<uint32_t> spv = recompile_valu(code, std::size(code), 1, 0);
        const std::vector<float> got = spv.empty()
            ? std::vector<float>{}
            : prosper::test::run_compute(spv, {0.0f}, 1, 1);
        return got.size() == 1 ? bits_of(got[0]) : UINT32_MAX;
    };
    constexpr uint32_t logical_a = 0x0f0ff00fu;
    constexpr uint32_t logical_c = 0x00ff00ffu;
    auto with_scc = [](uint32_t result) {
        return result ^ (result != 0 ? 0x80000000u : 0u);
    };
    CHECK(run_sop2_complement(0x8a020100u, logical_a, logical_c) ==
              with_scc(logical_a & ~logical_c),
          "s_andn2_b32 computes scalar data and SCC exactly");
    CHECK(run_sop2_complement(0x8b020100u, logical_a, logical_c) ==
              with_scc(logical_a | ~logical_c),
          "s_orn2_b32 computes scalar data and SCC exactly");
    CHECK(run_sop2_complement(0x8c020100u, logical_a, logical_c) ==
              with_scc(~(logical_a & logical_c)),
          "s_nand_b32 computes scalar data and SCC exactly");
    CHECK(run_sop2_complement(0x8d020100u, logical_a, logical_c) ==
              with_scc(~(logical_a | logical_c)),
          "s_nor_b32 computes scalar data and SCC exactly");
    CHECK(run_sop2_complement(0x8e020100u, logical_a, logical_c) ==
              with_scc(~(logical_a ^ logical_c)),
          "s_xnor_b32 computes scalar data and SCC exactly");
    CHECK(run_sop2_complement(0x8a020100u, UINT32_MAX, UINT32_MAX) == 0,
          "s_andn2_b32 clears SCC when its scalar result is zero");

    const uint32_t gta_s_andn2_b32[] = {
        0xbe8703ffu, 0x12345678u, // establish the live s7 input
        0x8a07f507u,              // exact GTA pc46: s_andn2_b32 s7,s7,inline-float:245
        0x7e000207u,              // v_mov_b32 v0,s7
        0xbf810000u,
    };
    CHECK(!recompile_valu(gta_s_andn2_b32, std::size(gta_s_andn2_b32), 1, 0).empty(),
          "GTA V exact scalar s_andn2_b32 packet recompiles");

    // GTA V kernel 0x205b5e8600 uses s_sub_u32/s_subb_u32 as a 64-bit subtraction pair while
    // negating EXEC. Pin both halves of S_SUBB's contract independently: the incoming low-word
    // borrow changes the high-word data result, and the high-word underflow becomes its outgoing SCC.
    const uint32_t code9_subb_data[] = {
        0xbe800380u,              // s_mov_b32 s0,0
        0xbe820385u,              // s_mov_b32 s2,5
        0xbe830383u,              // s_mov_b32 s3,3
        0x80818100u,              // s_sub_u32 s1,s0,1 -> borrow-in=1
        0x82840302u,              // s_subb_u32 s4,s2,s3 -> 5-3-1 = 1, borrow-out=0
        0x7e000204u,              // v_mov_b32 v0,s4
        0x7e000d00u,              // v_cvt_f32_u32 v0,v0
        0xbf810000u,
    };
    const std::vector<uint32_t> spv9_subb_data = recompile_valu(
        code9_subb_data, std::size(code9_subb_data), 1, 0);
    const std::vector<float> got9_subb_data = spv9_subb_data.empty()
        ? std::vector<float>{}
        : prosper::test::run_compute(spv9_subb_data, {0.0f}, 1, 1);
    CHECK(!spv9_subb_data.empty() && got9_subb_data.size() == 1 &&
              got9_subb_data[0] == 1.0f,
          "s_subb_u32 subtracts the incoming low-word borrow from its data result");

    const uint32_t code9_subb_borrow[] = {
        0xbe800380u,              // s_mov_b32 s0,0
        0x80818100u,              // s_sub_u32 s1,s0,1 -> borrow-in=1
        0x82828000u,              // s_subb_u32 s2,s0,0 -> UINT_MAX, borrow-out=1
        0x85038081u,              // s_cselect_b32 s3,1,0
        0x7e000203u,              // v_mov_b32 v0,s3
        0x7e000d00u,              // v_cvt_f32_u32 v0,v0
        0xbf810000u,
    };
    const std::vector<uint32_t> spv9_subb_borrow = recompile_valu(
        code9_subb_borrow, std::size(code9_subb_borrow), 1, 0);
    const std::vector<float> got9_subb_borrow = spv9_subb_borrow.empty()
        ? std::vector<float>{}
        : prosper::test::run_compute(spv9_subb_borrow, {0.0f}, 1, 1);
    CHECK(!spv9_subb_borrow.empty() && got9_subb_borrow.size() == 1 &&
              got9_subb_borrow[0] == 1.0f,
          "s_subb_u32 publishes its high-word borrow through SCC");

    const uint32_t code9_subb_primary_borrow[] = {
        0xbe800381u,              // s_mov_b32 s0,1
        0xbe810380u,              // s_mov_b32 s1,0
        0x80838000u,              // s_sub_u32 s3,s0,0 -> borrow-in=0
        0x82820001u,              // s_subb_u32 s2,s1,s0 -> UINT_MAX, primary borrow-out=1
        0x85038081u,              // s_cselect_b32 s3,1,0
        0x7e000203u,              // v_mov_b32 v0,s3
        0x7e000d00u,              // v_cvt_f32_u32 v0,v0
        0xbf810000u,
    };
    const std::vector<uint32_t> spv9_subb_primary_borrow = recompile_valu(
        code9_subb_primary_borrow, std::size(code9_subb_primary_borrow), 1, 0);
    const std::vector<float> got9_subb_primary_borrow = spv9_subb_primary_borrow.empty()
        ? std::vector<float>{}
        : prosper::test::run_compute(spv9_subb_primary_borrow, {0.0f}, 1, 1);
    CHECK(!spv9_subb_primary_borrow.empty() && got9_subb_primary_borrow.size() == 1 &&
              got9_subb_primary_borrow[0] == 1.0f,
          "s_subb_u32 publishes a primary src0<src1 borrow through SCC");
}
