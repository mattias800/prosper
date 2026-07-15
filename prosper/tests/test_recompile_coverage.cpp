// test_recompile_coverage — recompile_coverage() reports per-instruction recompiler support without
// requiring a complete vertex/fragment. Pure (no Vulkan), so it runs in CI. It also drives the
// data-driven coverage report over the real game shaders (shader_histo).
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
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
    CHECK(c.unsupported == 1 && c.first_bad_fmt >= 0 && c.first_bad_op == 0x06 &&
          c.first_bad_pc == 1,
          "a forward s_cbranch_vccz reports its exact rejection PC instead of becoming a no-op");
    CHECK(recompile_valu(vcc_branch, sizeof(vcc_branch)/sizeof(vcc_branch[0]), 2, 3).empty(),
          "the production recompiler rejects the unsafe forward VCC branch");

    // A PROVEN-UNIFORM VCCZ-exit loop now structurizes in the compute shell too (#590, extending
    // #615): the uniformity proof is data-provenance-based — the compare reads s0 (scalar) and v1,
    // whose nearest definition is `v_mov_b32 v1, 4` (an unmodified uniform VOP1 move from an inline
    // scalar) — so every lane's compare bool is identical and the wave-empty vccz exit lowers to
    // this invocation's bool, exactly as in the fragment shell. Body must be barrier/LDS/cross-lane
    // free (guards below).
    const uint32_t compute_vcc_loop[] = {
        0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7D020200u, 0xBF860004u,
        0x060000FFu, 0x3E800000u, 0x81008100u, 0xBF82FFFAu, 0xBF810000u,
    };
    CHECK(!recompile_valu(compute_vcc_loop, sizeof(compute_vcc_loop)/sizeof(compute_vcc_loop[0]), 0, 0).empty(),
          "a proven-uniform VCCZ-exit loop structurizes in the compute shell (#590)");
    // The SAME loop with v1 defined from a VGPR (`v_mov_b32 v1, v0`) fails the uniform-VOP1-from-
    // scalar proof clause: the compare can no longer be proven wave-uniform, so compute still
    // rejects it loudly (a varying trip count needs a real wave reduction).
    const uint32_t compute_vcc_loop_varying[] = {
        0xBE800380u, 0x7E000280u, 0x7E020300u, 0x7D020200u, 0xBF860004u,
        0x060000FFu, 0x3E800000u, 0x81008100u, 0xBF82FFFAu, 0xBF810000u,
    };
    CHECK(recompile_valu(compute_vcc_loop_varying,
                         sizeof(compute_vcc_loop_varying)/sizeof(compute_vcc_loop_varying[0]), 0, 0).empty(),
          "a VCCZ-exit loop whose compare reads a varying VGPR still rejects in the compute shell");
    // A genuinely nested/multi-branch compute CFG uses the hardware-wave-vote dispatcher fallback. This
    // is the reduced shape of UE4's volume-lighting kernel: varying VCC exit, an inner scalar branch,
    // and a backward loop edge. The simpler two-branch loop above deliberately remains rejected.
    const uint32_t compute_cfg_dispatch[] = {
        0xBE800380u, 0x7E000280u, 0x7E020300u,
        0xD7610013u, 0x00014A7Eu, 0xD7610013u, 0x0001507Fu,
        0xD760000Eu, 0x00014B13u, 0xD760000Fu, 0x00015113u, 0xBEFE040Eu,
        0xE00C2000u, 0x80020400u, 0x7DB900F9u, 0x86050007u,
        0x7D020200u, 0xBF860006u, 0xBF0A8204u, 0x360000FDu, 0xBF840001u,
        0x81008100u, 0x81008100u, 0xBF82FFF4u,
        0xBF810000u,
    };
    ShaderResourceTable dispatch_rt;
    ShaderResource dispatch_vb{}; dispatch_vb.cls = ResourceClass::VertexBuffer;
    dispatch_vb.binding = 3; dispatch_vb.sgpr_base = 8; dispatch_vb.stride = 16;
    dispatch_vb.format = DataFormat::Float32; dispatch_vb.num_components = 4;
    dispatch_rt.resources.push_back(dispatch_vb);
    CHECK(!recompile_valu(compute_cfg_dispatch,
                          sizeof(compute_cfg_dispatch)/sizeof(compute_cfg_dispatch[0]), 0, 0,
                          &dispatch_rt).empty(),
          "a nested varying-VCC compute CFG preserves spilled EXEC and lowers through the dispatcher");
    // Ordinary LDS effects do not require workgroup-uniform control flow. Keep the same dispatcher
    // shape, but make the inner SCC arm conditionally execute a ds_write_b32 before the back-edge.
    // (A barrier remains forbidden below; only the blanket rejection of raw DS is being relaxed.)
    const uint32_t compute_cfg_dispatch_lds[] = {
        0xBE800380u, 0x7E000280u, 0x7E020300u,
        0xD7610013u, 0x00014A7Eu, 0xD7610013u, 0x0001507Fu,
        0xD760000Eu, 0x00014B13u, 0xD760000Fu, 0x00015113u, 0xBEFE040Eu,
        0xE00C2000u, 0x80020400u, 0x7DB900F9u, 0x86050007u,
        0x7D020200u, 0xBF860006u, 0xBF0A8204u, 0x360000FDu, 0xBF840002u,
        0xD8340000u, 0x00000302u, 0xBF82FFF4u,
        0xBF810000u,
    };
    CHECK(!recompile_valu(compute_cfg_dispatch_lds,
                          sizeof(compute_cfg_dispatch_lds)/sizeof(compute_cfg_dispatch_lds[0]), 0, 0,
                          &dispatch_rt).empty(),
          "a complex compute CFG may execute ordinary LDS writes through the dispatcher");
    const uint32_t scc_data_source[] = {
        0xBF060000u, 0x360000FDu, 0xBF810000u, // s_cmp_eq_u32 s0,s0; v_and_b32 v0,scc,v0
    };
    CHECK(!recompile_valu(scc_data_source, sizeof(scc_data_source)/sizeof(scc_data_source[0]), 0, 0).empty(),
          "SCC is accepted as its architectural scalar 0/1 ALU source");
    const uint32_t mask_nor[] = {
        0x7C040CF9u, 0x06869880u, 0x8DEA6A18u, 0xBF810000u,
    };
    CHECK(!recompile_valu(mask_nor, sizeof(mask_nor)/sizeof(mask_nor[0]), 0, 0).empty(),
          "s_nor_b64 combines a saved comparison mask with VCC in the bool domain");
    // An s_barrier INSIDE the loop body also rejects: the uniformity proof is per-WAVE, and a
    // barrier inside a loop whose trip count could differ across the workgroup's waves would be
    // workgroup-divergent control flow (UB).
    const uint32_t compute_vcc_loop_barrier[] = {
        0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7D020200u, 0xBF860005u,
        0x060000FFu, 0x3E800000u, 0x81008100u, 0xBF8A0000u, 0xBF82FFF9u, 0xBF810000u,
    };
    CHECK(recompile_valu(compute_vcc_loop_barrier,
                         sizeof(compute_vcc_loop_barrier)/sizeof(compute_vcc_loop_barrier[0]), 0, 0).empty(),
          "a uniform VCCZ-exit loop containing s_barrier still rejects in the compute shell");

    // A FORWARD vccz if with the same proven-uniform compare also structurizes in compute (#590 —
    // DOLL's blocked lighting/fill kernels are forward vccz if/else trees, not loops). Same shape as
    // the loop fixture minus the back-edge: compare, skip-two-dwords vccz to s_endpgm (early-out).
    const uint32_t compute_vcc_if[] = {
        0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7D020200u, 0xBF860002u,
        0x060000FFu, 0x3E800000u, 0xBF810000u,
    };
    CHECK(!recompile_valu(compute_vcc_if, sizeof(compute_vcc_if)/sizeof(compute_vcc_if[0]), 0, 0).empty(),
          "a proven-uniform forward vccz if structurizes in the compute shell (#590)");
    // Varying compare (v1 <- v_mov from a VGPR): still rejects.
    const uint32_t compute_vcc_if_varying[] = {
        0xBE800380u, 0x7E000280u, 0x7E020300u, 0x7D020200u, 0xBF860002u,
        0x060000FFu, 0x3E800000u, 0xBF810000u,
    };
    CHECK(recompile_valu(compute_vcc_if_varying,
                         sizeof(compute_vcc_if_varying)/sizeof(compute_vcc_if_varying[0]), 0, 0).empty(),
          "a forward vccz if with a varying compare still rejects in the compute shell");
    // s_barrier inside the guarded region: still rejects (workgroup-divergence hazard).
    const uint32_t compute_vcc_if_barrier[] = {
        0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7D020200u, 0xBF860003u,
        0x060000FFu, 0x3E800000u, 0xBF8A0000u, 0xBF810000u,
    };
    CHECK(recompile_valu(compute_vcc_if_barrier,
                         sizeof(compute_vcc_if_barrier)/sizeof(compute_vcc_if_barrier[0]), 0, 0).empty(),
          "a uniform forward vccz if containing s_barrier still rejects in the compute shell");
    // The compare-finding walk looks past instructions that provably cannot rewrite VCC (real UE4
    // kernels schedule unrelated ALU between compare and branch): same uniform if with two plain
    // VALU ops (v_mov v2,1 ; v_mov v3,2) hoisted between the compare and the branch.
    const uint32_t compute_vcc_if_hoisted[] = {
        0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7D020200u, 0x7E040281u, 0x7E060282u,
        0xBF860002u, 0x060000FFu, 0x3E800000u, 0xBF810000u,
    };
    CHECK(!recompile_valu(compute_vcc_if_hoisted,
                          sizeof(compute_vcc_if_hoisted)/sizeof(compute_vcc_if_hoisted[0]), 0, 0).empty(),
          "the uniformity proof looks past VCC-preserving instructions between compare and branch");
    // UE4's scalar-spill lowering also schedules v_writelane_b32 between the uniform compare and
    // its VCCZ branch. The op updates one lane of a VGPR and cannot clobber VCC, so it must not stop
    // the same local proof. (Raw pair assembled and round-tripped with llvm-mc gfx1010.)
    const uint32_t compute_vcc_if_writelane[] = {
        0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7D020200u,
        0xD7610013u, 0x00012A1Bu, // v_writelane_b32 v19, s27, 21
        0xBF860002u, 0x060000FFu, 0x3E800000u, 0xBF810000u,
    };
    CHECK(!recompile_valu(compute_vcc_if_writelane,
                          sizeof(compute_vcc_if_writelane)/sizeof(compute_vcc_if_writelane[0]), 0, 0).empty(),
          "the uniformity proof looks past VCC-preserving v_writelane scalar spills");
    // But an intervening write that COULD hit VCC (s_mov_b32 s106, 0) stops the walk: reject.
    const uint32_t compute_vcc_if_clobber[] = {
        0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7D020200u, 0xBEEA0380u,
        0xBF860002u, 0x060000FFu, 0x3E800000u, 0xBF810000u,
    };
    CHECK(recompile_valu(compute_vcc_if_clobber,
                         sizeof(compute_vcc_if_clobber)/sizeof(compute_vcc_if_clobber[0]), 0, 0).empty(),
          "an intervening write to s106 (VCC) between compare and branch stops the proof");

    // A forward s_cbranch_execz that REJOINS LIVE CODE cannot be linearized when its skipped block has
    // a live scalar write. The compute structurizer now handles that case with subgroupAny(EXEC), so
    // the whole wave either executes or skips the scalar write and its live use remains exact.
    //   v_cmpx ; s_cbranch_execz +1 ; s_mov_b32 s0,1 ; v_mov_b32 v0,s0 ; s_endpgm
    const uint32_t execz_scalar[] = { 0x7da80300u, 0xbf880001u, 0xbe800381u, 0x7e000200u, 0xBF810000u };
    RecompileCoverage d = recompile_coverage(execz_scalar, sizeof(execz_scalar)/sizeof(execz_scalar[0]));
    CHECK(d.unsupported == 0,
          "a narrowed EXEC branch over live scalar state is reconstructed in compute coverage");
    CHECK(!recompile_valu(execz_scalar, sizeof(execz_scalar)/sizeof(execz_scalar[0]), 2, 0).empty(),
          "the production recompiler structures execz branches over live scalar writes");

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
