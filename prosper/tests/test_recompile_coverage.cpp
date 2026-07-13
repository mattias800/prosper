// test_recompile_coverage — recompile_coverage() reports per-instruction recompiler support without
// requiring a complete vertex/fragment. Pure (no Vulkan), so it runs in CI. It also drives the
// data-driven coverage report over the real game shaders (shader_histo).
#include "../src/gpu/rdna2_to_spirv.hpp"
#include <cstdio>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_recompile_coverage ==\n");

    // Fully-supported ALU: v_add_f32 v0,v0,v1 ; v_mul_f32 v0,v0,v2 ; s_endpgm.
    const uint32_t ok_code[] = { 0x06000300u, 0x10000500u, 0xBF810000u };
    RecompileCoverage a = recompile_coverage(ok_code, sizeof(ok_code)/sizeof(ok_code[0]));
    printf("  supported-kernel: total=%u alu=%u exports=%u unsupported=%u\n", a.total, a.alu, a.exports, a.unsupported);
    CHECK(a.total == 2 && a.alu == 2 && a.unsupported == 0 && a.first_bad_fmt < 0,
          "a fully-handled ALU kernel reports 100% coverage");

    // Contains an unsupported op: v_add_f32 ; s_branch +5 (unconditional -> rejected) ; s_endpgm.
    const uint32_t bad_code[] = { 0x06000300u, 0xbf820005u, 0xBF810000u };
    RecompileCoverage b = recompile_coverage(bad_code, sizeof(bad_code)/sizeof(bad_code[0]));
    printf("  mixed-kernel: total=%u alu=%u unsupported=%u first_bad_op=0x%x\n", b.total, b.alu, b.unsupported, b.first_bad_op);
    CHECK(b.total == 2 && b.alu == 1 && b.unsupported == 1 && b.first_bad_fmt >= 0 && b.first_bad_op == 0x02,
          "an unconditional s_branch is reported as the first unsupported instruction");

    // A forward VCC branch is not covered by EXEC predication. Treating it as a no-op would execute
    // the skipped block even when VCC says to branch, so the recompiler must reject it for now.
    const uint32_t vcc_branch[] = { 0x7da80300u, 0xbf860001u, 0x4a060300u, 0xBF810000u };
    RecompileCoverage c = recompile_coverage(vcc_branch, sizeof(vcc_branch)/sizeof(vcc_branch[0]));
    CHECK(c.unsupported == 1 && c.first_bad_fmt >= 0 && c.first_bad_op == 0x06,
          "a forward s_cbranch_vccz is rejected instead of linearized as a no-op");
    CHECK(recompile_valu(vcc_branch, sizeof(vcc_branch)/sizeof(vcc_branch[0]), 2, 3).empty(),
          "the production recompiler rejects the unsafe forward VCC branch");

    // A VCCZ-exit loop is valid in vertex/fragment stages where one SPIR-V invocation models one
    // hardware lane (#615), but not in the 64-lane compute shell: compute VCC needs a wave reduction.
    const uint32_t compute_vcc_loop[] = {
        0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7D020200u, 0xBF860004u,
        0x060000FFu, 0x3E800000u, 0x81008100u, 0xBF82FFFAu, 0xBF810000u,
    };
    CHECK(recompile_valu(compute_vcc_loop, sizeof(compute_vcc_loop)/sizeof(compute_vcc_loop[0]), 0, 0).empty(),
          "a VCCZ-exit loop remains rejected by the compute shell (wave-mask condition)");

    // A forward s_cbranch_execz that REJOINS LIVE CODE is only safe to linearize when its skipped block
    // is EXEC-predicated VGPR writes. Here the block is a scalar write (s_mov_b32 s0,1) and the branch
    // target is a live use of s0 (v_mov_b32 v0,s0) — the scalar write is NOT dead, so linearizing would
    // wrongly set s0 for lanes that should have skipped. Must reject.
    //   v_cmpx ; s_cbranch_execz +1 ; s_mov_b32 s0,1 ; v_mov_b32 v0,s0 ; s_endpgm
    const uint32_t execz_scalar[] = { 0x7da80300u, 0xbf880001u, 0xbe800381u, 0x7e000200u, 0xBF810000u };
    RecompileCoverage d = recompile_coverage(execz_scalar, sizeof(execz_scalar)/sizeof(execz_scalar[0]));
    CHECK(d.unsupported == 1 && d.first_bad_fmt >= 0 && d.first_bad_op == 0x08,
          "a narrowed EXEC branch over live scalar state (rejoining live code) is rejected");
    CHECK(recompile_valu(execz_scalar, sizeof(execz_scalar)/sizeof(execz_scalar[0]), 2, 0).empty(),
          "the production recompiler rejects execz branches that would skip live scalar writes");

    // But the SAME scalar write IS safe to linearize when the execz skips straight to s_endpgm: the
    // write is dead (nothing runs after s_endpgm) and scalar ops are wave-uniform, so running it under
    // linearization is observationally a no-op. This is the compute grid-tail guard idiom.
    //   v_cmpx ; s_cbranch_execz +1 ; s_mov_b32 s0,1 ; s_endpgm
    const uint32_t execz_end[] = { 0x7da80300u, 0xbf880001u, 0xbe800381u, 0xBF810000u };
    RecompileCoverage e = recompile_coverage(execz_end, sizeof(execz_end)/sizeof(execz_end[0]));
    CHECK(e.unsupported == 0,
          "a narrowed EXEC branch to s_endpgm over a dead scalar write is linearized (guard-to-end shape)");

    // #325: a 2D_ARRAY (SQ_RSRC_IMG_2D_ARRAY, DIM=5) image_sample is now accepted — handled as its base
    // 2D slice (array index dropped) — instead of rejecting the whole shader (previously DIM!=1&&!=2 was
    // truly-unsupported, silently skipping the draw). Same bytes as the 2D image_sample (0xF0800F08) with
    // DIM (dword0 bits[5:3]) = 5 -> 0xF0800F28. It is `table_dependent` (needs a resource table), NOT
    // `unsupported`, so a real recompile (with the T#) proceeds.
    const uint32_t arr_sample[] = { 0xF0800F28u, 0x00A30000u, 0xBF810000u };
    RecompileCoverage f = recompile_coverage(arr_sample, sizeof(arr_sample)/sizeof(arr_sample[0]));
    printf("  2d_array-sample: total=%u table_dependent=%u unsupported=%u first_bad_fmt=%d\n",
           f.total, f.table_dependent, f.unsupported, f.first_bad_fmt);
    CHECK(f.unsupported == 0 && f.table_dependent >= 1 && f.first_bad_fmt < 0,
          "#325: a 2D_ARRAY image_sample is recompilable-in-context (base slice), not unsupported");

    const uint32_t cvt_i4[] = { 0x7e001d00u, 0xBF810000u }; // v_cvt_off_f32_i4 v0,v0
    RecompileCoverage g = recompile_coverage(cvt_i4, sizeof(cvt_i4)/sizeof(cvt_i4[0]));
    CHECK(g.total == 1 && g.alu == 1 && g.unsupported == 0,
          "#527: v_cvt_off_f32_i4 is covered by the VOP1 recompiler");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
