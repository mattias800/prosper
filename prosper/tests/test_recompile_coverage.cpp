// test_recompile_coverage — recompile_coverage() reports per-instruction recompiler support without
// requiring a complete vertex/fragment. Pure (no Vulkan), so it runs in CI. It also drives the
// data-driven coverage report over the real game shaders (shader_histo).
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include <algorithm>
#include <cstdlib>
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

    // v_add_co_u32 / v_sub_co_u32 / v_subrev_co_u32 (VOP3B carry-out add/subtract, ops 0x30F/0x310/
    // 0x319). GTA V's (PPSA04263) texture-decode compute kernel advances a 64-bit flat source pointer
    // with v_add_co_u32; before this op emitted, the whole kernel was rejected at first_bad_op=0x30f,
    // so the decoded texture never rendered (#1163/#1165 investigation). Encodings are byte-identical
    // to that kernel's pc=0030 instruction: op in bits[25:16], sdst=vcc, dst=v1, src0=s12, src1=v6.
    const uint32_t co_code[] = {
        0xd70f6a01u, 0x00020c0cu,   // v_add_co_u32    v1, vcc, s12, v6
        0xd7106a01u, 0x00020c0cu,   // v_sub_co_u32    v1, vcc, s12, v6
        0xd7196a01u, 0x00020c0cu,   // v_subrev_co_u32 v1, vcc, s12, v6
        0xBF810000u,                // s_endpgm
    };
    RecompileCoverage co = recompile_coverage(co_code, sizeof(co_code)/sizeof(co_code[0]));
    CHECK(co.total == 3 && co.alu == 3 && co.unsupported == 0 && co.first_bad_fmt < 0,
          "v_add/sub/subrev_co_u32 (VOP3B carry-out) recompile as supported ALU");

    // Coverage only inspects emit_alu's `ok` flag and discards the SPIR-V; also prove v_add_co_u32
    // lowers to a real, well-formed module (VGPR operands this time: v2 = v0 + v1, carry->vcc). Without
    // the emit branch recompile_valu returns {}; with it, a valid module (magic 0x07230203) is produced.
    const uint32_t co_valu[] = {
        0xd70f6a02u, 0x00020300u,   // v_add_co_u32 v2, vcc, v0, v1
        0xBF810000u,                // s_endpgm
    };
    std::vector<uint32_t> co_spv = recompile_valu(co_valu, sizeof(co_valu)/sizeof(co_valu[0]), 2, 2);
    CHECK(!co_spv.empty() && co_spv[0] == 0x07230203u,
          "v_add_co_u32 lowers to a valid compute SPIR-V module");

    // analyze_flat_loads (#1171): resolve a general (non-scratch) flat_load's 64-bit address to its base
    // user-SGPR pointer pair. Encodings are byte-identical to GTA V's texture-decode kernel
    // exec_cs_2042d47600 (pc=0030/0040): the low address dword adds base-low s12, the high dword adds
    // base-high s13, so v[1:2] = s[12:13] + offset and the resolved base is s12.
    const uint32_t flat_code[] = {
        0xd70f6a01u, 0x00020c0cu,   // v_add_co_u32    v1, vcc, s12, v6   (addr low  = s12 + v6)
        0x50041af9u, 0x86860680u,   // v_add_co_ci_u32 v2 (sdwa), addend s13  (addr high = s13 + carry)
        0xdc200000u, 0x007d0001u,   // flat_load_ubyte v0, v[1:2]
        0xBF810000u,                // s_endpgm
    };
    FlatLoadAnalysis fla = analyze_flat_loads(flat_code, sizeof(flat_code)/sizeof(flat_code[0]), 16);
    CHECK(fla.any && fla.all_resolved && fla.loads.size() == 1 &&
          fla.loads[0].base_sgpr == 12 && fla.loads[0].vaddr_lo_reg == 1 &&
          fla.loads[0].dst_reg == 0 && fla.loads[0].bits == 8 && !fla.loads[0].sign_extend,
          "analyze_flat_loads resolves v[1:2]=s[12:13]+offset flat_load_ubyte to base s12");

    // Negative: an address whose HIGH dword is a plain VGPR move (not a user-SGPR add) is unresolvable,
    // so the load stays fail-visible (all_resolved=false) rather than binding a bogus window.
    const uint32_t flat_unres[] = {
        0xd70f6a01u, 0x00020c0cu,   // v_add_co_u32    v1, vcc, s12, v6
        0x7e040300u,                // v_mov_b32       v2, v0             (high dword not from a user SGPR)
        0xdc200000u, 0x007d0001u,   // flat_load_ubyte v0, v[1:2]
        0xBF810000u,                // s_endpgm
    };
    FlatLoadAnalysis flu = analyze_flat_loads(flat_unres, sizeof(flat_unres)/sizeof(flat_unres[0]), 16);
    CHECK(flu.any && !flu.all_resolved && flu.loads.size() == 1 && flu.loads[0].base_sgpr < 0,
          "analyze_flat_loads leaves a non-pointer-arithmetic flat address unresolved (fail-visible)");

    // Negative: both dwords are valid user-SGPR adds but the base SGPRs are NON-CONSECUTIVE (low adds
    // s12, high adds s15). This must stay unresolved — resolving it to s12 would bind an s[12:13] window
    // for an address whose high dword is s15, i.e. silent wrong data. Guards the `lo_base==hi_base-1`
    // invariant (without that clause this would wrongly resolve to base s12).
    const uint32_t flat_split[] = {
        0xd70f6a01u, 0x00020c0cu,   // v_add_co_u32    v1, vcc, s12, v6   (low base = s12)
        0x5004000fu,                // v_add_co_ci_u32 v2, vcc, s15, v0   (high base = s15, NOT s13)
        0xdc200000u, 0x007d0001u,   // flat_load_ubyte v0, v[1:2]
        0xBF810000u,                // s_endpgm
    };
    FlatLoadAnalysis fls = analyze_flat_loads(flat_split, sizeof(flat_split)/sizeof(flat_split[0]), 16);
    CHECK(fls.any && !fls.all_resolved && fls.loads.size() == 1 && fls.loads[0].base_sgpr < 0,
          "analyze_flat_loads rejects a non-consecutive base SGPR pair (s12 low / s15 high)");

    // FLAT-window emit (#1171 piece 2): with a ConstantBuffer resource keyed by the flat_load's pc plus
    // flat_base_sgpr, recompile_compute LOWERS the general flat_load to a windowed SSBO read; without
    // the bound window it stays fail-visible (empty module).
    const uint32_t flat_kernel[] = {
        0xd70f6a01u, 0x00020c0cu,   // v_add_co_u32    v1, vcc, s12, v6
        0x50041af9u, 0x86860680u,   // v_add_co_ci_u32 v2 (sdwa), s13
        0xdc200000u, 0x007d0001u,   // flat_load_ubyte v0, v[1:2]
        0xBF810000u,                // s_endpgm
    };
    const size_t nfk = sizeof(flat_kernel)/sizeof(flat_kernel[0]);
    ComputeShaderConfig fcfg;
    fcfg.user_sgprs.assign(16, 0);   // 16 user SGPRs; s12/s13 hold the (per-dispatch) base pointer
    CHECK(recompile_compute(flat_kernel, nfk, nullptr, fcfg).empty(),
          "general flat_load with no bound window still rejects the shader (fail-visible)");
    FlatLoadAnalysis fka = analyze_flat_loads(flat_kernel, nfk, 16);
    ShaderResourceTable frt;
    ShaderResource fwin;
    fwin.cls = ResourceClass::ConstantBuffer;
    fwin.binding = 4;
    fwin.gpu_addr = 0x1000000000ull;
    fwin.size = 4096;
    fwin.fetch_pc = fka.loads[0].load_pc;
    fwin.flat_base_sgpr = static_cast<uint32_t>(fka.loads[0].base_sgpr);   // s12
    frt.resources.push_back(fwin);
    std::vector<uint32_t> fk_spv = recompile_compute(flat_kernel, nfk, &frt, fcfg);
    CHECK(!fk_spv.empty() && fk_spv[0] == 0x07230203u,
          "general flat_load lowers to a valid SSBO-read module when its window is bound");

    // #1183: safe_execz_branches must NOT linearize a loop-EXIT execz (one enclosed by a backward branch
    // that jumps to at-or-before it) — even when its target is s_endpgm — so detect_divergent_loops /
    // emit_divloop can reconstruct the structured loop; a plain guard-to-end execz (no enclosing back-edge)
    // still linearizes. Without the loop-awareness fix the loop-exit execz would be wrongly marked safe and
    // its back-edge would fall to a straight-line reject (GTA V's exec_cs_2042d47600 decode loop).
    auto has = [](const std::vector<uint32_t>& v, uint32_t x) {
        for (uint32_t e : v) if (e == x) return true; return false;
    };
    const uint32_t execz_guard[] = {
        0xbf880001u,   // s_cbranch_execz -> pc=2 (s_endpgm): plain guard-to-end, no loop
        0x7e000301u,   // v_mov_b32 v0, v1  (EXEC-predicated body)
        0xbf810000u,   // s_endpgm
    };
    CHECK(has(safe_execz_branches_for_test(execz_guard, sizeof(execz_guard)/sizeof(execz_guard[0])), 0u),
          "safe_execz_branches keeps a plain guard-to-end execz linearizable (no regression)");
    const uint32_t execz_loop[] = {
        0xbf880002u,   // s_cbranch_execz -> pc=3 (s_endpgm): a LOOP EXIT
        0x7e000301u,   // v_mov_b32 v0, v1  (loop body)
        0xbf82fffdu,   // s_branch -> pc=0  (backward -> encloses the execz)
        0xbf810000u,   // s_endpgm
    };
    CHECK(!has(safe_execz_branches_for_test(execz_loop, sizeof(execz_loop)/sizeof(execz_loop[0])), 0u),
          "safe_execz_branches leaves a loop-exit execz for the divergent-loop structurizer (#1183)");

    // UE4 uses VCC_LO/HI as temporary scalar data inside an execz arm, then reads only VCC_LO with
    // vector ALU before a VOPC overwrites the complete pair. The high half is dead at the merge: a
    // 32-bit vector source at VCC_LO must not be mistaken for a B64 read of VCC_HI. Once both body
    // writes are proven dead the whole-wave empty guard is a safe predication optimization.
    const uint32_t vcc_scratch_guard[] = {
        0x7C220080u,                 //  0: v_cmpx_neq_f32 exec, 0, v0 (narrow EXEC)
        0xBF880002u,                 //  1: s_cbranch_execz -> pc=4
        0xF4241A84u, 0xFA000020u,    //  2: s_buffer_load_dwordx2 vcc, s[8:11], 0x20
        0xF4201A84u, 0xFA000000u,    //  4: s_buffer_load_dword vcc_lo, s[8:11], 0
        0x100600F9u, 0x0696066Au,    //  6: v_mul_f32 v3, vcc_lo, v0 (32-bit low-half read)
        0x7D880E23u,                 //  8: v_cmp_* vcc, s35, v7 (kills both physical halves)
        0x87866A06u,                 //  9: s_and_b64 s[6:7], s[6:7], vcc
        0xBF810000u,
    };
    CHECK(has(safe_execz_branches_for_test(vcc_scratch_guard,
                                            std::size(vcc_scratch_guard)), 1u),
          "a VCC_LO vector read does not keep the dead VCC_HI scalar scratch word live");

    // The scalar inputs of a buffer packet are completely decoded: its four-dword SRSRC and
    // one-dword SOFFSET. A buffer access through unrelated s[32:35] between the merge and the VCC
    // overwrite must therefore not make a temporary VCC_HI write look live. This is the shape that
    // gates Astro Bot's world-map compute setup shaders.
    const uint32_t vcc_scratch_with_buffer[] = {
        0x7C220080u,                 //  0: v_cmpx_neq_f32 exec, 0, v0
        0xBF880002u,                 //  1: s_cbranch_execz -> pc=4
        0xF4241A84u, 0xFA000020u,    //  2: s_buffer_load_dwordx2 vcc, s[8:11], 0x20
        0xF4201A84u, 0xFA000000u,    //  4: s_buffer_load_dword vcc_lo, s[8:11], 0
        0xE0383000u, 0x80080202u,    //  6: buffer_load_dwordx4 v[2:5], v2, s[32:35], 0
        0x100600F9u, 0x0696066Au,    //  8: v_mul_f32 v3, vcc_lo, v0
        0x7D880E23u,                 // 10: v_cmp_* vcc, s35, v7 (kills both halves)
        0x87866A06u,                 // 11: s_and_b64 s[6:7], s[6:7], vcc
        0xBF810000u,
    };
    CHECK(has(safe_execz_branches_for_test(vcc_scratch_with_buffer,
                                            std::size(vcc_scratch_with_buffer)), 1u),
          "an unrelated decoded buffer descriptor does not keep VCC_HI scratch live");

    uint32_t live_vcc_hi_guard[std::size(vcc_scratch_guard)];
    std::copy(std::begin(vcc_scratch_guard), std::end(vcc_scratch_guard),
              std::begin(live_vcc_hi_guard));
    live_vcc_hi_guard[6] = 0x7E06546Bu;   // v_cvt_f32_u32 v3, vcc_hi: true direct high-half read
    live_vcc_hi_guard[7] = 0xBF800000u;   // s_nop (the replacement is one dword)
    CHECK(!has(safe_execz_branches_for_test(live_vcc_hi_guard,
                                             std::size(live_vcc_hi_guard)), 1u),
          "a real VCC_HI vector read keeps the guarded scalar write non-linearizable");

    // Wave32 kernels use one-word mask instructions around EXEC: compare into VCC_LO, invert that
    // mask, copy it to EXEC_LO, then restore EXEC_LO. These are mask-domain operations, not scalar
    // reads of an unrepresentable VCC dword.
    const uint32_t wave32_half_mask[] = {
        0x7C000000u, // v_cmp_f_f32 vcc, v0, v0
        0xBEEA076Au, // s_not_b32 vcc_lo, vcc_lo
        0xBEFE036Au, // s_mov_b32 exec_lo, vcc_lo
        0x7E000281u, // v_mov_b32 v0, 1 (EXEC-predicated)
        0xBEFE03C1u, // s_mov_b32 exec_lo, -1
        0xBF810000u,
    };
    ComputeShaderConfig wave32_half_mask_config;
    wave32_half_mask_config.wave_size = 32;
    wave32_half_mask_config.native_subgroup_size = 32;
    CHECK(!recompile_compute(wave32_half_mask, std::size(wave32_half_mask), nullptr,
                             wave32_half_mask_config).empty(),
          "Wave32 VCC_LO inversion/copy/EXEC restore recompiles in the mask domain");

    // tbuffer_load_format_x v0, v0, s[8:11], 0 format:32_FLOAT ; s_endpgm.
    // MTBUF needs a resolved V# binding, so the table-less coverage pass must classify it as
    // recompilable-in-context rather than truly unsupported.
    const uint32_t mtbuf_code[] = { 0xe8b02000u, 0x80020100u, 0xBF810000u };
#ifdef _WIN32
    _putenv_s("PROSPER_DBG", "1");
#else
    setenv("PROSPER_DBG", "1", 1);
#endif
    RecompileCoverage mtbuf = recompile_coverage(mtbuf_code,
                                                  sizeof(mtbuf_code)/sizeof(mtbuf_code[0]));
#ifdef _WIN32
    _putenv_s("PROSPER_DBG", "");
#else
    unsetenv("PROSPER_DBG");
#endif
    CHECK(mtbuf.total == 1 && mtbuf.table_dependent == 1 && mtbuf.unsupported == 0 &&
          mtbuf.first_bad_fmt < 0,
          "MTBUF typed loads remain null-safe under debug coverage and report table dependence");
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

    // Messenger's bindless VS loads a V# through a register-offset s_load_dwordx4. The production
    // emitter resolves it only with the per-draw resource table, so the table-less coverage pass must
    // classify this as context-dependent (the same contract as MUBUF/MTBUF), not unsupported.
    const uint32_t smem_descriptor[] = {
        0xF408020Cu, 0xD6000000u, // s_load_dwordx4 s[8:11], s[24:25], vcc_hi
        0xBF810000u,
    };
    RecompileCoverage smem = recompile_coverage(
        smem_descriptor, sizeof(smem_descriptor) / sizeof(smem_descriptor[0]));
    CHECK(smem.total == 1 && smem.table_dependent == 1 && smem.unsupported == 0 &&
          smem.first_bad_fmt < 0,
          "register-offset SMEM descriptor loads report resource-table dependence");

    // Astro Bot world-map PS ends with this exact raw x3 store. The table-less coverage shell uses
    // its conventional binding 2, so accepting the instruction must classify it as handled ALU/memory
    // rather than the shader's sole unsupported opcode.
    const uint32_t astro_store_x3[] = {
        0xE07C2000u, 0x8004030Au, 0xBF810000u,
    };
    RecompileCoverage x3_store = recompile_coverage(
        astro_store_x3, std::size(astro_store_x3));
    CHECK(x3_store.total == 1 && x3_store.alu == 1 &&
              x3_store.table_dependent == 0 && x3_store.unsupported == 0 &&
              x3_store.first_bad_fmt < 0,
          "Astro buffer_store_dwordx3 is covered as a supported raw memory operation");

    // The common compiler spill/fill form uses a fixed byte offset and one unmodified scalar base.
    // The host shader maps that private storage to one per-invocation Function array.
    const uint32_t scratch_static[] = {
        0xdc704010u, 0x00000000u, // scratch_store_dword off, v0, s0 offset:16
        0xdc304010u, 0x00000000u, // scratch_load_dword v0, off, s0 offset:16
        0xBF810000u,
    };
    RecompileCoverage scratch = recompile_coverage(
        scratch_static, sizeof(scratch_static) / sizeof(scratch_static[0]));
    CHECK(scratch.total == 2 && scratch.alu == 2 && scratch.unsupported == 0 &&
              scratch.first_bad_fmt < 0,
          "static private spill/fill instructions report full recompiler coverage");

    const uint32_t scratch_dynamic[] = {
        0xdc304000u, 0x007d0002u, // scratch_load_dword v0, v2, off
        0xBF810000u,
    };
    RecompileCoverage dynamic = recompile_coverage(
        scratch_dynamic, sizeof(scratch_dynamic) / sizeof(scratch_dynamic[0]));
    CHECK(dynamic.total == 1 && dynamic.unsupported == 1 && dynamic.first_bad_op == 0x0cu,
          "dynamic private addresses remain explicit unsupported coverage");

    const uint32_t global_address[] = {
        0xdc308000u, 0x007d0002u, // global_load_dword v0, v[2:3], off
        0xBF810000u,
    };
    RecompileCoverage global = recompile_coverage(
        global_address, sizeof(global_address) / sizeof(global_address[0]));
    CHECK(global.total == 1 && global.unsupported == 1 && global.first_bad_op == 0x0cu,
          "arbitrary global addresses remain explicit unsupported coverage");

    const uint32_t scratch_lds[] = {
        0xdc306000u, 0x00000000u, // scratch_load_dword off, s0 lds
        0xBF810000u,
    };
    RecompileCoverage lds_transfer = recompile_coverage(
        scratch_lds, sizeof(scratch_lds) / sizeof(scratch_lds[0]));
    CHECK(lds_transfer.total == 1 && lds_transfer.unsupported == 1 &&
              lds_transfer.first_bad_op == 0x0cu &&
              recompile_valu(scratch_lds, sizeof(scratch_lds) / sizeof(scratch_lds[0]), 0, 0).empty(),
          "scratch LDS-transfer packets remain explicit unsupported coverage");

    const uint32_t rewritten_scratch_base[] = {
        0xbe800384u,               // s_mov_b32 s0, 4
        0xdc704010u, 0x00000000u, // scratch_store_dword off, v0, s0 offset:16
        0xdc304010u, 0x00000000u, // scratch_load_dword v0, off, s0 offset:16
        0xBF810000u,
    };
    CHECK(recompile_valu(rewritten_scratch_base,
                         sizeof(rewritten_scratch_base) / sizeof(rewritten_scratch_base[0]),
                         1, 0).empty(),
          "a shader-written private-storage base rejects instead of aliasing the entry spill area");

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

    // A forward VCC branch consumes the wave-wide any(VCC) predicate. The compute shell lowers that
    // through the exact guest-wave dispatcher instead of an implementation-width native subgroup.
    const uint32_t vcc_branch[] = { 0x7da80300u, 0xbf860001u, 0x4a060300u, 0xBF810000u };
    RecompileCoverage c = recompile_coverage(vcc_branch, sizeof(vcc_branch)/sizeof(vcc_branch[0]));
    CHECK(c.unsupported == 0 && c.first_bad_fmt < 0,
          "a forward s_cbranch_vccz is covered by an explicit compute wave vote");
    CHECK(!recompile_valu(vcc_branch, sizeof(vcc_branch)/sizeof(vcc_branch[0]), 2, 3).empty(),
          "the production recompiler lowers a forward VCC branch");

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
    // The SAME loop with v1 defined from an unresolved VGPR (`v_mov_b32 v1, v2`) cannot prove the
    // compare wave-uniform, so compute still rejects it loudly (a varying trip count needs a real
    // wave reduction). Do not use v0 here: the preceding `v_mov_b32 v0, 0` makes that chain uniform.
    const uint32_t compute_vcc_loop_varying[] = {
        0xBE800380u, 0x7E000280u, 0x7E020302u, 0x7D020200u, 0xBF860004u,
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
    ComputeShaderConfig wave32_dispatch_config;
    wave32_dispatch_config.wave_size = 32;
    CHECK(!recompile_compute(compute_cfg_dispatch, std::size(compute_cfg_dispatch),
                             &dispatch_rt, wave32_dispatch_config).empty(),
          "the complex compute CFG dispatcher supports Wave32 without B32 mask aliases");
    std::vector<uint32_t> compute_cfg_dispatch_created_b32 = {
        0x7D8802F9u, 0x06068000u, // v_cmp_lt_f32_sdwa s[0:1], v0, v0
        0xBE820300u,              // s_mov_b32 s2, s0 (new one-word mask alias)
    };
    compute_cfg_dispatch_created_b32.insert(
        compute_cfg_dispatch_created_b32.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    // The generic dispatcher now carries an unambiguous one-word mask domain at every basic-block
    // boundary. This formerly-negative fixture is the smallest compute regression for the same
    // dataflow used by Astro's complex Wave32 fragment shader.
    CHECK(!recompile_compute(compute_cfg_dispatch_created_b32.data(),
                             compute_cfg_dispatch_created_b32.size(), &dispatch_rt,
                             wave32_dispatch_config).empty(),
          "a dispatcher preserves B32 aliases of masks produced inside its CFG");
    // Astro's world-map kernel materializes a Wave32 VCC predicate, then asks whether the physical
    // VCC_LO dword is greater than zero before a scalar branch. That is an exact guest-wave any vote,
    // not a per-invocation integer read. Exercise both the structured and dispatcher paths.
    const uint32_t wave32_vcc_gt_zero[] = {
        0x7C022880u, // v_cmp_eq_f32 vcc, 0, v20
        0xBF08806Au, // s_cmp_gt_u32 vcc_lo, 0
        0xBF810000u,
    };
    wave32_dispatch_config.native_subgroup_size = 32;
    CHECK(!recompile_compute(wave32_vcc_gt_zero, std::size(wave32_vcc_gt_zero), nullptr,
                             wave32_dispatch_config).empty(),
          "Wave32 VCC_LO > 0 lowers to an exact native guest-wave vote");
    std::vector<uint32_t> compute_cfg_dispatch_vcc_gt_zero(
        std::begin(wave32_vcc_gt_zero), std::end(wave32_vcc_gt_zero) - 1);
    compute_cfg_dispatch_vcc_gt_zero.insert(
        compute_cfg_dispatch_vcc_gt_zero.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_vcc_gt_zero.data(),
                             compute_cfg_dispatch_vcc_gt_zero.size(), &dispatch_rt,
                             wave32_dispatch_config).empty(),
          "the complex dispatcher preserves Wave32 VCC_LO > 0 scalar mask semantics");
    std::vector<uint32_t> compute_cfg_dispatch_ff1 = {
        0xBE800385u, // s_mov_b32 s0, 5
        0x7D840000u, // v_cmp_eq_u32 vcc, s0, v0
        0xBEEA136Au, // s_ff1_i32_b32 vcc_lo, vcc_lo
        0x7E04026Au, // v_mov_b32 v2, vcc_lo (consume the scalar-data result)
    };
    compute_cfg_dispatch_ff1.insert(
        compute_cfg_dispatch_ff1.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_ff1.data(),
                             compute_cfg_dispatch_ff1.size(), &dispatch_rt,
                             wave32_dispatch_config).empty(),
          "the complex dispatcher lowers exact Wave32 s_ff1 mask reduction to scalar data");
    // A physical VCC half may hold ordinary scalar data before a later block starts a mask lifetime
    // in the same word. Block discovery must not retroactively reinterpret the earlier comparison as
    // a wave vote after the B32 dataflow learns about that later lifetime (Astro world-map PC436).
    std::vector<uint32_t> compute_cfg_dispatch_recycled_vcc_hi = {
        0xBEEB0300u, // s_mov_b32 vcc_hi, s0 (ordinary scalar data)
        0xBF076B80u, // s_cmp_lg_u32 0, vcc_hi
        0x856B807Eu, // s_cselect_b32 vcc_hi, exec_lo, 0 (new mask lifetime)
    };
    compute_cfg_dispatch_recycled_vcc_hi.insert(
        compute_cfg_dispatch_recycled_vcc_hi.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_recycled_vcc_hi.data(),
                             compute_cfg_dispatch_recycled_vcc_hi.size(), &dispatch_rt,
                             wave32_dispatch_config).empty(),
          "the dispatcher keeps recycled VCC scalar and mask lifetimes distinct");
    std::vector<uint32_t> compute_cfg_dispatch_trap = {
        0xBF070000u, // s_cmp_lg_u32 s0, s0 (false)
        0xBF840001u, // s_cbranch_scc0 +1 (valid data skips the assertion trap)
        0xBF920001u, // s_trap 1
    };
    compute_cfg_dispatch_trap.insert(
        compute_cfg_dispatch_trap.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_trap.data(),
                             compute_cfg_dispatch_trap.size(), &dispatch_rt,
                             wave32_dispatch_config).empty(),
          "the dispatcher terminates an assertion-trap arm without rejecting its valid bypass");
    std::vector<uint32_t> compute_cfg_dispatch_bitset = {
        0xBE850380u, // s_mov_b32 s5, 0
        0xBE851D9Fu, // s_bitset1_b32 s5, 31 (in-place scalar read/modify/write)
    };
    compute_cfg_dispatch_bitset.insert(
        compute_cfg_dispatch_bitset.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_bitset.data(),
                             compute_cfg_dispatch_bitset.size(), &dispatch_rt,
                             wave32_dispatch_config).empty(),
          "the dispatcher preserves an in-place scalar bitset update");
    std::vector<uint32_t> compute_cfg_dispatch_scalar_shift = {
        0xBEEA0381u, // s_mov_b32 vcc_lo, 1 (scalar-data lifetime)
        0xBEEB0380u, // s_mov_b32 vcc_hi, 0
        0x8FEA876Au, // s_lshl_b64 vcc, vcc, 7
    };
    compute_cfg_dispatch_scalar_shift.insert(
        compute_cfg_dispatch_scalar_shift.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_scalar_shift.data(),
                             compute_cfg_dispatch_scalar_shift.size(), &dispatch_rt,
                             wave32_dispatch_config).empty(),
          "the dispatcher preserves 64-bit scalar address shifts through VCC");
    std::vector<uint32_t> compute_cfg_dispatch_saved_mask_not = {
        0xBE94037Eu, // s_mov_b32 s20, exec_lo (complete Wave32 mask)
        0xBEEA0381u, // s_mov_b32 vcc_lo, 1 (older scalar-data lifetime)
        0xBF070000u, // s_cmp_lg_u32 s0, s0 (false)
        0xBF850003u, // s_cbranch_scc1 +3 -> mask merge
        0x856B807Eu, // s_cselect_b32 vcc_hi, exec_lo, 0
        0xBEEA0714u, // s_not_b32 vcc_lo, s20
        0x8D6A6B6Au, // s_nor_b32 vcc_lo, vcc_lo, vcc_hi
    };
    compute_cfg_dispatch_saved_mask_not.insert(
        compute_cfg_dispatch_saved_mask_not.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_saved_mask_not.data(),
                             compute_cfg_dispatch_saved_mask_not.size(), &dispatch_rt,
                             wave32_dispatch_config).empty(),
          "the dispatcher inverts a saved Wave32 SGPR mask before recombining VCC");
    std::vector<uint32_t> compute_cfg_dispatch_saved_mask_not_boundary = {
        0xBE94037Eu, // s_mov_b32 s20, exec_lo (complete Wave32 mask)
        0xBE950714u, // s_not_b32 s21, s20
        0xBF060000u, // s_cmp_eq_u32 s0, s0 (true; creates a conditional CFG edge)
        0xBF850001u, // s_cbranch_scc1 +1 -> consumer
        0xBF800000u, // unreachable s_nop 0 fallthrough arm
        0xBEFE0315u, // s_mov_b32 exec_lo, s21 (mask-only consumer)
    };
    compute_cfg_dispatch_saved_mask_not_boundary.insert(
        compute_cfg_dispatch_saved_mask_not_boundary.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_saved_mask_not_boundary.data(),
                             compute_cfg_dispatch_saved_mask_not_boundary.size(), &dispatch_rt,
                             wave32_dispatch_config).empty(),
          "the dispatcher persists a saved Wave32 mask inverted before a block boundary");
    wave32_dispatch_config.native_subgroup_size = 0;
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
    // An ordinary VGPR write invalidates the compile-time v_writelane spill lifetime. Crossing a
    // dispatcher block must preserve that tombstone; reconstructing the slot as scalar zero would
    // incorrectly make the later v_readlane valid again.
    const uint32_t compute_cfg_dispatch_invalidated_lane[] = {
        0xBE800381u, 0x7E000280u, 0x7E020300u,
        0xD7610013u, 0x00014A00u,                   // v19[37] = s0
        0x7E260280u, 0xBF820000u,                   // v19 = 0; next block
        0xD7600002u, 0x00014B13u,                   // invalid s2 = v19[37]
        0xD7610013u, 0x00014A7Eu, 0xD7610013u, 0x0001507Fu,
        0xD760000Eu, 0x00014B13u, 0xD760000Fu, 0x00015113u, 0xBEFE040Eu,
        0xE00C2000u, 0x80020400u, 0x7DB900F9u, 0x86050007u,
        0x7D020200u, 0xBF860006u, 0xBF0A8204u, 0x360000FDu, 0xBF840001u,
        0x81008100u, 0x81008100u, 0xBF82FFF4u,
        0xBF810000u,
    };
    CHECK(recompile_valu(compute_cfg_dispatch_invalidated_lane,
                         sizeof(compute_cfg_dispatch_invalidated_lane) /
                             sizeof(compute_cfg_dispatch_invalidated_lane[0]),
                         0, 0, &dispatch_rt).empty(),
          "the compute CFG dispatcher preserves an invalidated spill tombstone");
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

    // The narrow loop structurizers likewise do not carry the validity bit for a one-word Wave32
    // mask alias. This loop creates an SGPR-pair mask with VOPC and copies its low word before the
    // back-edge; reject until that extra loop state has an explicit phi.
    const uint32_t compute_vcc_loop_created_b32[] = {
        0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7D020200u, 0xBF860007u,
        0x060000FFu, 0x3E800000u, 0x81008100u,
        0x7D8802F9u, 0x06068200u, // v_cmp_lt_f32_sdwa s[2:3], v0, v0
        0xBE840302u,              // s_mov_b32 s4, s2
        0xBF82FFF7u, 0xBF810000u,
    };
    CHECK(recompile_compute(compute_vcc_loop_created_b32,
                            std::size(compute_vcc_loop_created_b32), nullptr,
                            wave32_dispatch_config).empty(),
          "a structured loop rejects B32 aliases of mask pairs produced in-loop");

    // A FORWARD vccz if with the same proven-uniform compare also structurizes in compute (#590 —
    // DOLL's blocked lighting/fill kernels are forward vccz if/else trees, not loops). Same shape as
    // the loop fixture minus the back-edge: compare, skip-two-dwords vccz to s_endpgm (early-out).
    const uint32_t compute_vcc_if[] = {
        0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7D020200u, 0xBF860002u,
        0x060000FFu, 0x3E800000u, 0xBF810000u,
    };
    CHECK(!recompile_valu(compute_vcc_if, sizeof(compute_vcc_if)/sizeof(compute_vcc_if[0]), 0, 0).empty(),
          "a proven-uniform forward vccz if structurizes in the compute shell (#590)");
    // A lane-varying compare is still valid: the scalar branch consumes any(VCC), not one lane's bit.
    const uint32_t compute_vcc_if_varying[] = {
        0xBE800380u, 0x7E000280u, 0x7E020300u, 0x7D020200u, 0xBF860002u,
        0x060000FFu, 0x3E800000u, 0xBF810000u,
    };
    CHECK(!recompile_valu(compute_vcc_if_varying,
                          sizeof(compute_vcc_if_varying)/sizeof(compute_vcc_if_varying[0]), 0, 0).empty(),
          "a varying forward vccz if lowers through the compute wave vote");
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
    // An intervening VCC write changes the value consumed by the vote but no longer invalidates an
    // obsolete producer-uniformity proof: the branch always reduces the current VCC value.
    const uint32_t compute_vcc_if_clobber[] = {
        0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7D020200u, 0xBEEA0380u,
        0xBF860002u, 0x060000FFu, 0x3E800000u, 0xBF810000u,
    };
    CHECK(!recompile_valu(compute_vcc_if_clobber,
                          sizeof(compute_vcc_if_clobber)/sizeof(compute_vcc_if_clobber[0]), 0, 0).empty(),
          "a forward branch votes on the current VCC after an intervening write");

    // A forward s_cbranch_execz that REJOINS LIVE CODE cannot be linearized when its skipped block has
    // a live scalar write. The exact guest-wave dispatcher handles that case, so
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
