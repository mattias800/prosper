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

    // tbuffer_load_format_x v0, v0, s[8:11], 0 format:32_FLOAT ; s_endpgm.
    // MTBUF needs a resolved V# binding, so the table-less coverage pass must classify it as
    // recompilable-in-context rather than truly unsupported.
    const uint32_t mtbuf_code[] = { 0xe8b02000u, 0x80020100u, 0xBF810000u };
    RecompileCoverage mtbuf = recompile_coverage(mtbuf_code,
                                                  sizeof(mtbuf_code)/sizeof(mtbuf_code[0]));
    CHECK(mtbuf.total == 1 && mtbuf.table_dependent == 1 && mtbuf.unsupported == 0 &&
          mtbuf.first_bad_fmt < 0,
          "MTBUF typed loads are reported as resource-table-dependent coverage");
    const uint32_t mtbuf_d16_code[] = { 0xe8682000u, 0x80220000u, 0xBF810000u };
    RecompileCoverage mtbuf_d16 = recompile_coverage(
        mtbuf_d16_code, sizeof(mtbuf_d16_code)/sizeof(mtbuf_d16_code[0]));
    CHECK(mtbuf_d16.total == 1 && mtbuf_d16.table_dependent == 0 &&
          mtbuf_d16.unsupported == 1 && mtbuf_d16.first_bad_op == 8u,
          "unimplemented MTBUF D16 remains an explicit unsupported opcode");
    const uint32_t mtbuf_tfe_code[] = { 0xe8b02000u, 0x80820100u, 0xBF810000u };
    RecompileCoverage mtbuf_tfe = recompile_coverage(
        mtbuf_tfe_code, sizeof(mtbuf_tfe_code)/sizeof(mtbuf_tfe_code[0]));
    CHECK(mtbuf_tfe.total == 1 && mtbuf_tfe.table_dependent == 0 &&
          mtbuf_tfe.unsupported == 1 && mtbuf_tfe.first_bad_op == 0u,
          "MTBUF TFE remains explicit unsupported coverage until its status write is modeled");

    // Contains an unsupported op: v_add_f32 ; s_branch +5 (unconditional -> rejected) ; s_endpgm.
    const uint32_t bad_code[] = { 0x06000300u, 0xbf820005u, 0xBF810000u };
    RecompileCoverage b = recompile_coverage(bad_code, sizeof(bad_code)/sizeof(bad_code[0]));
    printf("  mixed-kernel: total=%u alu=%u unsupported=%u first_bad_op=0x%x\n", b.total, b.alu, b.unsupported, b.first_bad_op);
    CHECK(b.total == 2 && b.alu == 1 && b.unsupported == 1 && b.first_bad_fmt >= 0 && b.first_bad_op == 0x02,
          "an unconditional s_branch is reported as the first unsupported instruction");

    // A scalar conditional whose target is a straight-line second arm after the first s_endpgm is
    // structured as two terminating arms. Coverage credits the guest conditional while excluding the
    // compiler-only arm skip that replaces the first s_endpgm.
    const uint32_t terminating_if_else[] = {
        0xbe800387u, 0xbe810388u, 0xbf060100u, 0xbf840003u,
        0x7e0002ffu, 0x42280000u, 0xbf810000u,
        0x7e0002ffu, 0x41100000u, 0xbf810000u,
    };
    RecompileCoverage terminal = recompile_coverage(
        terminating_if_else, sizeof(terminating_if_else)/sizeof(terminating_if_else[0]));
    CHECK(terminal.total == 6 && terminal.alu == 6 && terminal.exports == 0 &&
              terminal.table_dependent == 0 && terminal.unsupported == 0 &&
              terminal.first_bad_fmt < 0,
          "a terminating post-endpgm if/else reports six real guest instructions as structured");

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
    // The dispatcher includes branch-target/fallthrough blocks even when no entry path can reach
    // them. This dead block overwrites half of direct V# s[8:11] and then targets the live entry;
    // provenance at the reachable fetch must not include a write hardware can never execute.
    std::vector<uint32_t> compute_cfg_dispatch_dead_descriptor = {
        0xBF820002u, // entry: s_branch live (skip the next two instructions)
        0xBE880432u, // dead: s_mov_b64 s[8:9], s[50:51]
        0xBF820000u, // dead: s_branch live
    };
    compute_cfg_dispatch_dead_descriptor.insert(
        compute_cfg_dispatch_dead_descriptor.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + sizeof(compute_cfg_dispatch)/sizeof(compute_cfg_dispatch[0]));
    CHECK(!recompile_valu(compute_cfg_dispatch_dead_descriptor.data(),
                          compute_cfg_dispatch_dead_descriptor.size(), 0, 0,
                          &dispatch_rt).empty(),
          "unreachable dispatcher writes cannot invalidate a live direct descriptor");

    // Descriptor identity is compile-time provenance, not part of the scalar value persisted by the
    // complex-CFG dispatcher's Function variables. This reduced Astro shape loads V# s[20:23] before
    // the branch tree and consumes it with a raw MUBUF at the shared exit. SRT-only lookup cannot be
    // proven at that later block; the front-half's exact per-MUBUF fetch_pc alias makes it unambiguous.
    std::vector<uint32_t> compute_cfg_dispatch_cross_block_vsharp = {
        0xF4080504u, 0xFA000010u, // s_load_dwordx4 s[20:23], s[8:9], 0x10
    };
    compute_cfg_dispatch_cross_block_vsharp.insert(
        compute_cfg_dispatch_cross_block_vsharp.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + sizeof(compute_cfg_dispatch) / sizeof(compute_cfg_dispatch[0]) - 1);
    const uint32_t cross_block_fetch_pc =
        static_cast<uint32_t>(compute_cfg_dispatch_cross_block_vsharp.size());
    compute_cfg_dispatch_cross_block_vsharp.push_back(0xE00C2000u);
    compute_cfg_dispatch_cross_block_vsharp.push_back(0x80050400u); // buffer_load_dword v4,v0,s[20:23]
    compute_cfg_dispatch_cross_block_vsharp.push_back(0xBF810000u);
    ShaderResourceTable cross_block_rt = dispatch_rt;
    ShaderResource cross_block_vsharp{};
    cross_block_vsharp.cls = ResourceClass::ConstantBuffer;
    cross_block_vsharp.binding = 4;
    cross_block_vsharp.srt_offset = 0x10;
    cross_block_vsharp.stride = 4;
    cross_block_rt.resources.push_back(cross_block_vsharp);
    CHECK(recompile_valu(compute_cfg_dispatch_cross_block_vsharp.data(),
                         compute_cfg_dispatch_cross_block_vsharp.size(), 0, 0,
                         &cross_block_rt).empty(),
          "a cross-block V# without exact fetch provenance remains unresolved");
    cross_block_rt.resources.back().fetch_pc = cross_block_fetch_pc;
    CHECK(!recompile_valu(compute_cfg_dispatch_cross_block_vsharp.data(),
                          compute_cfg_dispatch_cross_block_vsharp.size(), 0, 0,
                          &cross_block_rt).empty(),
          "exact MUBUF fetch provenance survives the complex compute CFG boundary");

    std::vector<uint32_t> compute_cfg_dispatch_live_vopc_write = {
        0x7C0400F9u, 0x06068800u, // v_cmp_eq_f32_sdwa s8, v0, v0 (writes s[8:9])
    };
    compute_cfg_dispatch_live_vopc_write.insert(
        compute_cfg_dispatch_live_vopc_write.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + sizeof(compute_cfg_dispatch)/sizeof(compute_cfg_dispatch[0]));
    CHECK(recompile_valu(compute_cfg_dispatch_live_vopc_write.data(),
                         compute_cfg_dispatch_live_vopc_write.size(), 0, 0,
                         &dispatch_rt).empty(),
          "reachable dispatcher VOPC pair writes invalidate a direct descriptor");
    // UE4 also recycles a physical v_writelane slot: first as ordinary scalar data, then as an
    // EXEC-mask spill. Force a dispatcher block boundary after each lifetime so both typed views
    // must survive the Function-variable state round trip. Consumers remain statically typed.
    const uint32_t compute_cfg_dispatch_recycled_lane[] = {
        0xBE800381u, 0x7E000280u, 0x7E020300u,
        0xD7610013u, 0x00014A00u, 0xBF820000u,       // v19[37] = s0; next block
        0xD7600002u, 0x00014B13u,                   // s2 = v19[37] (data view)
        0xD7610013u, 0x00014A7Eu, 0xD7610013u, 0x0001507Fu,
        0xBF820000u,                                // v19[37:40] = EXEC; next block
        0xD760000Eu, 0x00014B13u, 0xD760000Fu, 0x00015113u, 0xBEFE040Eu,
        0xE00C2000u, 0x80020400u, 0x7DB900F9u, 0x86050007u,
        0x7D020200u, 0xBF860006u, 0xBF0A8204u, 0x360000FDu, 0xBF840001u,
        0x81008100u, 0x81008100u, 0xBF82FFF4u,
        0xBF810000u,
    };
    CHECK(!recompile_valu(compute_cfg_dispatch_recycled_lane,
                          sizeof(compute_cfg_dispatch_recycled_lane) /
                              sizeof(compute_cfg_dispatch_recycled_lane[0]),
                          0, 0, &dispatch_rt).empty(),
          "the compute CFG dispatcher preserves a recycled scalar/mask spill lane");
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

    // A partial AGC thread-dimension workgroup needs a divergent entry guard to mask Vulkan's padded
    // invocations. That is safe for ordinary kernels, but not when the module contains a workgroup
    // barrier: every Vulkan invocation must reach OpControlBarrier uniformly. Keep exact workgroups
    // supported and reject the partial+barrier combination rather than risking a hang or undefined LDS.
    const uint32_t barrier_only[] = { 0xBF8A0000u, 0xBF810000u };
    ComputeShaderConfig exact_barrier;
    exact_barrier.local_x = 64;
    exact_barrier.exact_thread_extent = true;
    exact_barrier.threads_x = 64;
    exact_barrier.threads_y = exact_barrier.threads_z = 1;
    CHECK(!recompile_compute(barrier_only, 2, nullptr, exact_barrier).empty(),
          "an exact thread-dimension workgroup may retain a uniform barrier");
    exact_barrier.threads_x = 63;
    CHECK(recompile_compute(barrier_only, 2, nullptr, exact_barrier).empty(),
          "a partial thread-dimension workgroup with a barrier rejects fail-visibly");

    // A wave-empty saveexec/execz guard may surround a scalar counted loop when the body preserves
    // both EXEC and the guard's saved mask. This is the Evergate color-conversion shape.
    const uint32_t guarded_counted_loop[] = {
        0xBE84247Eu, 0xBF880009u,                         // save EXEC in s[4:5]; skip to restore if empty
        0xB0020005u, 0xBE800380u, 0x7E020280u,           // limit=5, i=0, sum=0
        0xBF0A0200u, 0xBF840003u, 0x4A020200u,           // loop compare/exit; sum += i
        0x81008100u, 0xBF82FFFBu,                        // i++; backedge
        0x7E000D01u, 0xBEFE0404u, 0xBF810000u,           // convert sum; restore EXEC; end
    };
    CHECK(!recompile_valu(guarded_counted_loop, std::size(guarded_counted_loop), 0, 0).empty(),
          "guarded counted loop is accepted when EXEC and the saved guard mask are preserved");

    // The loop may contain its own balanced saveexec/execz region. Unity emits this when a
    // per-iteration color contribution is lane-masked: the inner restore returns EXEC to the
    // outer guard before the backedge, so the loop-carried mask remains unchanged.
    const uint32_t guarded_counted_loop_nested_save[] = {
        0xBE84247Eu, 0xBF88000Cu,                         // outer guard -> outer restore
        0xB0020005u, 0xBE800380u, 0x7E020280u,           // limit=5, i=0, sum=0
        0xBF0A0200u, 0xBF840006u,                         // loop compare/exit
        0xBE86247Eu, 0xBF880001u,                         // inner guard -> inner restore
        0x4A020200u, 0xBEFE0406u,                         // sum += i; restore inner EXEC
        0x81008100u, 0xBF82FFF8u,                         // i++; backedge
        0x7E000D01u, 0xBEFE0404u, 0xBF810000u,           // convert; restore outer EXEC; end
    };
    CHECK(!recompile_valu(guarded_counted_loop_nested_save,
                          std::size(guarded_counted_loop_nested_save), 0, 0).empty(),
          "guarded counted loop accepts a balanced nested EXEC guard");

    const uint32_t guarded_counted_loop_cross_backedge[] = {
        0xBE84247Eu, 0xBF88000Cu,                         // outer guard -> outer restore
        0xB0020005u, 0xBE800380u, 0x7E020280u,
        0xBF0A0200u, 0xBF840005u,                         // loop compare/exit
        0xBE86247Eu, 0xBF880004u,                         // inner restore is after the backedge
        0x4A020200u, 0x81008100u, 0xBF82FFF9u,
        0x7E000D01u, 0xBEFE0406u, 0xBEFE0404u, 0xBF810000u,
    };
    CHECK(recompile_valu(guarded_counted_loop_cross_backedge,
                         std::size(guarded_counted_loop_cross_backedge), 0, 0).empty(),
          "nested EXEC guard crossing the counted-loop backedge is rejected");

    const uint32_t guarded_counted_loop_inner_save[] = {
        0xBE84247Eu, 0xBF88000Au,
        0xB0020005u, 0xBE800380u, 0x7E020280u,
        0xBF0A0200u, 0xBF840004u,
        0xBE86247Eu,                                     // inner saveexec mutates EXEC every iteration
        0x4A020200u, 0x81008100u, 0xBF82FFFAu,
        0x7E000D01u, 0xBEFE0404u, 0xBF810000u,
    };
    CHECK(recompile_valu(guarded_counted_loop_inner_save,
                         std::size(guarded_counted_loop_inner_save), 0, 0).empty(),
          "guarded counted loop rejects an uncarried inner EXEC mutation");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
