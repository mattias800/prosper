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

    // GTA V's world-entry compute programs 0x413ce6000 and 0x413ce6d00 contain these exact
    // V_ALIGNBYTE_B32 packets (including the literal dword).  This is a VOP3A integer result:
    // {src0,src1} is shifted right by 8*src2[4:0], with src1 as the low dword, and no VCC output.
    const uint32_t alignbyte_live[] = {
        0xd54f0006u, 0x0415fe80u, 0x3024240cu, // v_alignbyte_b32 v6, 0, 0x3024240c, v5
        0xd54f0008u, 0x0415fe80u, 0x00301818u, // v_alignbyte_b32 v8, 0, 0x00301818, v5
        0xbf810000u,
    };
    const RecompileCoverage alignbyte_coverage = recompile_coverage(
        alignbyte_live, std::size(alignbyte_live));
    CHECK(alignbyte_coverage.total == 2 && alignbyte_coverage.alu == 2 &&
              alignbyte_coverage.unsupported == 0 && alignbyte_coverage.first_bad_fmt < 0,
          "GTA V's exact V_ALIGNBYTE_B32 literal packets report supported ALU coverage");
    const std::vector<uint32_t> alignbyte_spv = recompile_valu(
        alignbyte_live, std::size(alignbyte_live), /*num_inputs*/6, /*out_vgpr*/6);
    CHECK(!alignbyte_spv.empty() && alignbyte_spv[0] == 0x07230203u,
          "GTA V's exact V_ALIGNBYTE_B32 packet lowers to a compute SPIR-V module");

    // Unsupported modifier encodings must not fall through as the unmodified integer operation.
    // These mutate the same live production packet, one decoded modifier field at a time.
    const auto alignbyte_rejects = [](uint32_t word0, uint32_t word1) {
        const uint32_t code[] = {word0, word1, 0x3024240cu, 0xbf810000u};
        const RecompileCoverage coverage = recompile_coverage(code, std::size(code));
        return coverage.unsupported == 1 && coverage.first_bad_op == 0x14f &&
               recompile_valu(code, std::size(code), 6, 6).empty();
    };
    CHECK(alignbyte_rejects(0xd54f0106u, 0x0415fe80u),
          "V_ALIGNBYTE_B32 ABS mutation remains fail-visible");
    CHECK(alignbyte_rejects(0xd54f0006u, 0x2415fe80u),
          "V_ALIGNBYTE_B32 NEG mutation remains fail-visible");
    CHECK(alignbyte_rejects(0xd54f8006u, 0x0415fe80u),
          "V_ALIGNBYTE_B32 CLAMP mutation remains fail-visible");
    CHECK(alignbyte_rejects(0xd54f0006u, 0x0c15fe80u),
          "V_ALIGNBYTE_B32 OMOD mutation remains fail-visible");

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

    // GTA V uses a scalar load into VCC_LO as temporary data inside an EXECZ guard, then crosses an
    // image access before the temporary dies. MIMG's decoded T# and S# ranges are the complete scalar
    // read set, so an unrelated resource must not stop the merge-liveness proof. Exercise the actual
    // compute structurizer/emitter path rather than only its liveness helper.
    const uint32_t mimg_vcc_scratch_guard[] = {
        0x7E080300u,                 // 0: v_mov_b32 v4, v0 (1D image coordinate)
        0x7C220080u,                 // 1: v_cmpx_neq_f32 exec, 0, v0
        0xBF880002u,                 // 2: s_cbranch_execz -> pc=5
        0xF4201A84u, 0xFA000000u,   // 3: s_buffer_load_dword vcc_lo, s[8:11], 0
        0xF0000F00u, 0x00000004u,   // 5: image_load v[0:3], v4, s[0:7], dmask:0xf dim:1D
        0xBF810000u,
    };
    ShaderResourceTable mimg_guard_rt;
    { ShaderResource image{}; image.cls = ResourceClass::StorageImage; image.binding = 4;
      image.img_dim = 0; image.sgpr_base = 0; mimg_guard_rt.resources.push_back(image); }
    { ShaderResource constants{}; constants.cls = ResourceClass::ConstantBuffer;
      constants.binding = 5; constants.sgpr_base = 8; constants.size = 64;
      mimg_guard_rt.resources.push_back(constants); }
    ComputeShaderConfig mimg_guard_config;
    mimg_guard_config.user_sgprs.resize(16);
    CHECK(!recompile_compute(mimg_vcc_scratch_guard, std::size(mimg_vcc_scratch_guard),
                             &mimg_guard_rt, mimg_guard_config).empty(),
          "an unrelated MIMG T# range is transparent to post-merge VCC liveness");

    // Same-site mutation: move the image's decoded eight-word SRSRC range to s[104:111], which
    // overlaps VCC_LO/HI. The image now observes the guarded scalar write and must stay fail-visible.
    uint32_t mimg_vcc_resource[std::size(mimg_vcc_scratch_guard)];
    std::copy(std::begin(mimg_vcc_scratch_guard), std::end(mimg_vcc_scratch_guard),
              std::begin(mimg_vcc_resource));
    mimg_vcc_resource[6] = 0x001A0004u; // image_load ..., s[104:111]
    ShaderResourceTable mimg_vcc_rt = mimg_guard_rt;
    mimg_vcc_rt.resources[0].sgpr_base = 104;
    ComputeShaderConfig mimg_vcc_config = mimg_guard_config;
    mimg_vcc_config.user_sgprs.resize(112);
    CHECK(!has(safe_execz_branches_for_test(mimg_vcc_resource,
                                             std::size(mimg_vcc_resource)), 2u),
          "an MIMG SRSRC overlap specifically blocks the dead-scalar linearization proof");
    CHECK(recompile_compute(mimg_vcc_resource, std::size(mimg_vcc_resource),
                            &mimg_vcc_rt, mimg_vcc_config).empty(),
          "an MIMG SRSRC range overlapping VCC keeps the guarded scalar write live");

    // Sampled images add a separately decoded four-word sampler source. Prove that the liveness scan
    // crosses an unrelated S# as well, while a same-site S# mutation onto VCC remains conservative.
    uint32_t sampled_mimg_guard[std::size(mimg_vcc_scratch_guard)];
    std::copy(std::begin(mimg_vcc_scratch_guard), std::end(mimg_vcc_scratch_guard),
              std::begin(sampled_mimg_guard));
    sampled_mimg_guard[5] = 0xF0800F08u; // image_sample v[0:3], v[0:1], s[0:7], s[16:19]
    sampled_mimg_guard[6] = 0x00800000u;
    ShaderResourceTable sampled_guard_rt;
    { ShaderResource texture{}; texture.cls = ResourceClass::Texture; texture.binding = 4;
      texture.img_dim = 1; texture.width = 4; texture.height = 4; texture.sgpr_base = 0;
      texture.sampler_sgpr_base = 16; sampled_guard_rt.resources.push_back(texture); }
    sampled_guard_rt.resources.push_back(mimg_guard_rt.resources[1]);
    ComputeShaderConfig sampled_guard_config = mimg_guard_config;
    sampled_guard_config.user_sgprs.resize(20);
    CHECK(!recompile_compute(sampled_mimg_guard, std::size(sampled_mimg_guard),
                             &sampled_guard_rt, sampled_guard_config).empty(),
          "unrelated MIMG T# and S# ranges are transparent to post-merge VCC liveness");
    uint32_t sampled_vcc_sampler[std::size(sampled_mimg_guard)];
    std::copy(std::begin(sampled_mimg_guard), std::end(sampled_mimg_guard),
              std::begin(sampled_vcc_sampler));
    sampled_vcc_sampler[6] = 0x03400000u; // image_sample ..., s[0:7], s[104:107]
    ShaderResourceTable sampled_vcc_rt = sampled_guard_rt;
    sampled_vcc_rt.resources[0].sampler_sgpr_base = 104;
    ComputeShaderConfig sampled_vcc_config = sampled_guard_config;
    sampled_vcc_config.user_sgprs.resize(108);
    CHECK(!has(safe_execz_branches_for_test(sampled_vcc_sampler,
                                             std::size(sampled_vcc_sampler)), 2u),
          "an MIMG sampler overlap specifically blocks the dead-scalar linearization proof");
    CHECK(recompile_compute(sampled_vcc_sampler, std::size(sampled_vcc_sampler),
                            &sampled_vcc_rt, sampled_vcc_config).empty(),
          "an MIMG sampler range overlapping VCC keeps the guarded scalar write live");

    // GTA V's texture-transfer compute kernels load two adjacent eight-word T# descriptors with one
    // immediate s_load_dwordx16. The load has one SRT offset but the halves are distinct resources,
    // so only exact consuming-PC bindings are sound. Prove both bindings are emitted and the scalar
    // load does not manufacture the legacy fallback constant buffer at binding 2.
    auto x16_texture = [](uint32_t binding, uint32_t fetch_pc) {
        ShaderResource texture{};
        texture.cls = ResourceClass::Texture;
        texture.format = DataFormat::Float32;
        texture.num_components = 4;
        texture.binding = binding;
        texture.fetch_pc = fetch_pc;
        texture.img_dim = 1;
        texture.width = texture.height = 2;
        return texture;
    };
    const uint32_t smem_x16_pair[] = {
        0xf4100300u, 0xfa000000u,   // pc0: s_load_dwordx16 s[12:27], s[0:1], 0
        0xf0800f08u, 0x01630000u,   // pc2: image_sample ..., s[12:19], s[44:47]
        0xf0800f08u, 0x01850000u,   // pc4: image_sample ..., s[20:27], s[48:51]
        0xbf810000u,
    };
    ShaderResourceTable smem_x16_pair_rt;
    smem_x16_pair_rt.resources.push_back(x16_texture(4, 2));
    smem_x16_pair_rt.resources.push_back(x16_texture(5, 4));
    const std::vector<uint32_t> smem_x16_pair_spv = recompile_valu(
        smem_x16_pair, std::size(smem_x16_pair), 2, 0, &smem_x16_pair_rt);
    ShaderResourceTable smem_x16_pair_validation_rt = smem_x16_pair_rt;
    for (uint32_t binding = 0; binding < 2; ++binding) {
        ShaderResource io{};
        io.cls = ResourceClass::VertexBuffer;
        io.binding = binding;
        smem_x16_pair_validation_rt.resources.push_back(io);
    }
    const DescriptorValidationReport smem_x16_pair_report =
        validate_spirv_descriptor_interface(smem_x16_pair_spv,
                                            &smem_x16_pair_validation_rt, 0,
                                            SpirvShaderStage::Compute, false);
    CHECK(!smem_x16_pair_spv.empty() && smem_x16_pair_report.ok() &&
              find_spirv_descriptor_binding(smem_x16_pair_report, 0, 4) &&
              find_spirv_descriptor_binding(smem_x16_pair_report, 0, 5) &&
              !find_spirv_descriptor_binding(smem_x16_pair_report, 0, 2),
          "x16 descriptor bundle routes its two T# halves by exact PC without fallback cbuf");

    // Ordinary data use makes the x16 typeless again. Replacing that SGPR read with a descriptor-only
    // admission would silently substitute zero for real scalar data.
    const uint32_t smem_x16_data[] = {
        0xf4100300u, 0xfa000000u,
        0x7e04020cu,                 // v_mov_b32 v2, s12: ordinary data read of the loaded range
        0xf0800f08u, 0x01630000u,
        0xf0800f08u, 0x01850000u,
        0xbf810000u,
    };
    ShaderResourceTable smem_x16_data_rt;
    smem_x16_data_rt.resources.push_back(x16_texture(4, 3));
    smem_x16_data_rt.resources.push_back(x16_texture(5, 5));
    CHECK(recompile_valu(smem_x16_data, std::size(smem_x16_data), 2, 0,
                         &smem_x16_data_rt).empty(),
          "x16 load with an ordinary scalar/vector consumer remains fail-visible");

    ShaderResourceTable smem_x16_missing_rt;
    smem_x16_missing_rt.resources.push_back(x16_texture(4, 2));
    CHECK(recompile_valu(smem_x16_pair, std::size(smem_x16_pair), 2, 0,
                         &smem_x16_missing_rt).empty(),
          "x16 bundle rejects when one MIMG consumer lacks an exact-PC resource");
    ShaderResourceTable smem_x16_wrong_rt = smem_x16_pair_rt;
    smem_x16_wrong_rt.resources[1].cls = ResourceClass::ConstantBuffer;
    CHECK(recompile_valu(smem_x16_pair, std::size(smem_x16_pair), 2, 0,
                         &smem_x16_wrong_rt).empty(),
          "x16 bundle rejects an exact-PC resource of the wrong descriptor class");

    // GTA V also issues two x16 loads, including a nonzero immediate, to form four independent T#s.
    const uint32_t smem_x16_two_loads[] = {
        0xf4100300u, 0xfa000040u,   // pc0: s[12:27], byte offset 0x40
        0xf4100700u, 0xfa000000u,   // pc2: s[28:43], byte offset 0
        0xf0800f08u, 0x01e30000u,   // pc4:  T# s[12:19], S# s[60:63]
        0xf0800f08u, 0x01e50000u,   // pc6:  T# s[20:27]
        0xf0800f08u, 0x01e70000u,   // pc8:  T# s[28:35]
        0xf0800f08u, 0x01e90000u,   // pc10: T# s[36:43]
        0xbf810000u,
    };
    ShaderResourceTable smem_x16_two_rt;
    for (uint32_t i = 0; i < 4; ++i)
        smem_x16_two_rt.resources.push_back(x16_texture(4 + i, 4 + i * 2));
    const std::vector<uint32_t> smem_x16_two_spv = recompile_valu(
        smem_x16_two_loads, std::size(smem_x16_two_loads), 2, 0, &smem_x16_two_rt);
    ShaderResourceTable smem_x16_two_validation_rt = smem_x16_two_rt;
    for (uint32_t binding = 0; binding < 2; ++binding) {
        ShaderResource io{};
        io.cls = ResourceClass::VertexBuffer;
        io.binding = binding;
        smem_x16_two_validation_rt.resources.push_back(io);
    }
    const DescriptorValidationReport smem_x16_two_report =
        validate_spirv_descriptor_interface(smem_x16_two_spv,
                                            &smem_x16_two_validation_rt, 0,
                                            SpirvShaderStage::Compute, false);
    bool smem_x16_four_bindings = !smem_x16_two_spv.empty() && smem_x16_two_report.ok();
    for (uint32_t binding = 4; binding < 8; ++binding)
        smem_x16_four_bindings &=
            find_spirv_descriptor_binding(smem_x16_two_report, 0, binding) != nullptr;
    CHECK(smem_x16_four_bindings &&
              !find_spirv_descriptor_binding(smem_x16_two_report, 0, 2),
          "nonzero-offset and two-x16 bundle routes four exact-PC textures without cbuf");

    // The captured patched variant clears/replaces only T#.word3's type bits through a scalar
    // temporary. Its first patch schedules one unrelated VOP between AND and OR; both exact
    // resources must remain independently routed after their live descriptor words change.
    const uint32_t smem_x16_patched[] = {
        0xf4100300u, 0xfa000000u,
        0x876aff17u, 0x0fffffffu,   // pc2:  s_and_b32 vcc_lo, s23, 0x0fffffff
        0x4a080802u,                // pc4:  independent VOP scheduled inside the patch chain
        0x8817ff6au, 0xd0000000u,   // pc5:  s_or_b32  s23, vcc_lo, 0xd0000000
        0xf0800f08u, 0x01650000u,   // pc7:  sample T# s[20:27]
        0x876aff0fu, 0x0fffffffu,   // pc9:  s_and_b32 vcc_lo, s15, 0x0fffffff
        0x880fff6au, 0xd0000000u,   // pc11: s_or_b32  s15, vcc_lo, 0xd0000000
        0xf0800f08u, 0x01830000u,   // pc13: sample T# s[12:19]
        0xbf810000u,
    };
    ShaderResourceTable smem_x16_patched_rt;
    smem_x16_patched_rt.resources.push_back(x16_texture(4, 7));
    smem_x16_patched_rt.resources.push_back(x16_texture(5, 13));
    const std::vector<uint32_t> smem_x16_patched_spv = recompile_valu(
        smem_x16_patched, std::size(smem_x16_patched), 2, 0, &smem_x16_patched_rt);
    CHECK(!smem_x16_patched_spv.empty(),
          "x16 descriptor bundle admits the captured scheduled T#.word3 patch chain");

    // The placeholder word3 makes the patch temporary provenance-only too. Any observation before
    // that temporary is overwritten would consume zero-derived scalar data instead of guest data.
    const uint32_t smem_x16_patched_temp_read[] = {
        0xf4100300u, 0xfa000000u,
        0x876aff17u, 0x0fffffffu,
        0x4a080802u,
        0x8817ff6au, 0xd0000000u,
        0x7e04026au,                // pc7: v_mov_b32 v2, vcc_lo observes the patch temporary
        0xf0800f08u, 0x01650000u,
        0x876aff0fu, 0x0fffffffu,
        0x880fff6au, 0xd0000000u,
        0xf0800f08u, 0x01830000u,
        0xbf810000u,
    };
    ShaderResourceTable smem_x16_patched_temp_rt;
    smem_x16_patched_temp_rt.resources.push_back(x16_texture(4, 8));
    smem_x16_patched_temp_rt.resources.push_back(x16_texture(5, 14));
    CHECK(recompile_valu(smem_x16_patched_temp_read,
                         std::size(smem_x16_patched_temp_read), 2, 0,
                         &smem_x16_patched_temp_rt).empty(),
          "x16 word3 patch rejects a post-join observation of its descriptor-derived temporary");

    // Same gap site, two implicit data dependencies the decoded ordinary-source walk does not
    // cover: Special253 observes the AND-produced SCC, and e32 cndmask observes VCC_LO implicitly.
    uint32_t smem_x16_patched_scc_gap[std::size(smem_x16_patched)];
    std::copy(std::begin(smem_x16_patched), std::end(smem_x16_patched),
              std::begin(smem_x16_patched_scc_gap));
    smem_x16_patched_scc_gap[4] = 0x7e0402fdu; // v_mov_b32 v2, scc
    CHECK(recompile_valu(smem_x16_patched_scc_gap, std::size(smem_x16_patched_scc_gap),
                         2, 0, &smem_x16_patched_rt).empty(),
          "x16 word3 patch gap rejects an observation of the AND-produced SCC");
    uint32_t smem_x16_patched_vcc_gap[std::size(smem_x16_patched)];
    std::copy(std::begin(smem_x16_patched), std::end(smem_x16_patched),
              std::begin(smem_x16_patched_vcc_gap));
    smem_x16_patched_vcc_gap[4] = 0x02020100u; // v_cndmask_b32 v1,v0,v0,vcc (implicit VCC read)
    CHECK(recompile_valu(smem_x16_patched_vcc_gap, std::size(smem_x16_patched_vcc_gap),
                         2, 0, &smem_x16_patched_rt).empty(),
          "x16 word3 patch gap rejects an implicit VCC observation of its temporary");

    // M0 looks like another scalar temporary to the decoded operand walk, but DS append observes it
    // implicitly. The admitted compiler shape uses VCC_LO exactly; broadening that register gate
    // would therefore turn this into a silent placeholder-derived LDS allocator input.
    const uint32_t smem_x16_patched_m0_ds[] = {
        0xf4100300u, 0xfa000000u,
        0x877cff17u, 0x0fffffffu,   // pc2:  s_and_b32 m0, s23, 0x0fffffff
        0x4a080802u,
        0x8817ff7cu, 0xd0000000u,   // pc5:  s_or_b32 s23, m0, 0xd0000000
        0xd8f80008u, 0x03000000u,   // pc7:  ds_append v3 offset:8 implicitly observes m0
        0xf0800f08u, 0x01650000u,   // pc9:  sample T# s[20:27]
        0x876aff0fu, 0x0fffffffu,
        0x880fff6au, 0xd0000000u,
        0xf0800f08u, 0x01830000u,   // pc15: sample T# s[12:19]
        0xbf810000u,
    };
    ShaderResourceTable smem_x16_patched_m0_rt;
    smem_x16_patched_m0_rt.resources.push_back(x16_texture(4, 9));
    smem_x16_patched_m0_rt.resources.push_back(x16_texture(5, 15));
    CHECK(recompile_valu(smem_x16_patched_m0_ds, std::size(smem_x16_patched_m0_ds),
                         2, 0, &smem_x16_patched_m0_rt).empty(),
          "x16 word3 patch rejects M0 as a temporary before an implicit DS observation");

    // A post-merge B64 scalar write kills both VCC words before v_cndmask consumes the new mask. The
    // high-half proof must use the central scalar write-width inventory: checking only dst==R leaves
    // VCC_HI spuriously live and rejects this otherwise-structured EXECZ guard.
    const uint32_t b64_vcc_kill_guard[] = {
        0x7E080300u,                 // 0: v_mov_b32 v4, v0 (1D image coordinate)
        0x7D840100u,                 // 1: v_cmp_eq_u32 vcc, v0, v0
        0xBF880001u,                 // 2: s_cbranch_execz -> pc=4
        0xBEEA047Eu,                 // 3: s_mov_b64 vcc, exec (guarded write)
        0xBEEA047Eu,                 // 4: s_mov_b64 vcc, exec (kills both words after merge)
        0xF0000F00u, 0x00000004u,   // 5: unrelated image_load through s[0:7]
        0x02020100u,                 // 7: v_cndmask_b32 v1, v0, v0, vcc
        0xBF810000u,
    };
    ComputeShaderConfig b64_kill_config;
    b64_kill_config.user_sgprs.resize(8);
    CHECK(has(structured_execz_branches_for_test(b64_vcc_kill_guard,
                                                  std::size(b64_vcc_kill_guard)), 2u),
          "the B64 pair kill admits the guarded write through the structured compute CFG");
    CHECK(!recompile_compute(b64_vcc_kill_guard, std::size(b64_vcc_kill_guard), &mimg_guard_rt,
                             b64_kill_config).empty(),
          "a post-merge B64 write kills both VCC halves before their next consumer");

    // Same-site mutation: a B32 low-half write cannot kill the guarded high half. The following
    // whole-VCC consumer makes that surviving value observable, so structurization must reject.
    uint32_t b32_vcc_lo_kill[std::size(b64_vcc_kill_guard)];
    std::copy(std::begin(b64_vcc_kill_guard), std::end(b64_vcc_kill_guard),
              std::begin(b32_vcc_lo_kill));
    b32_vcc_lo_kill[4] = 0xBEEA037Eu; // s_mov_b32 vcc_lo, exec_lo
    CHECK(!has(structured_execz_branches_for_test(b32_vcc_lo_kill,
                                                   std::size(b32_vcc_lo_kill)), 2u),
          "the B32 low-half mutation is rejected by the structured compute CFG proof");
    CHECK(recompile_compute(b32_vcc_lo_kill, std::size(b32_vcc_lo_kill), &mimg_guard_rt,
                            b64_kill_config).empty(),
          "a low-half-only B32 write does not kill live VCC_HI at the merge");

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
    // Astro's world-map traversal intersects three complete Wave32 predicates in VCC_LO, then
    // branches on the SCC written by the final s_and_b32. Hardware SCC is any(result mask), so an
    // exact 32-lane native subgroup can lower it without scalarizing or guessing at one lane.
    std::vector<uint32_t> compute_cfg_dispatch_mask_scc = {
        0xBE9B037Eu,              // s_mov_b32 s27, exec_lo
        0x7D8A0A15u,              // v_cmp_* vcc, s21, v5       (Astro PC1079)
        0x7D8A0AF9u, 0x0686EB19u, // v_cmp_* s107, s25, v5      (Astro PC1080)
        0x7D8A0AF9u, 0x06869C1Au, // v_cmp_* s28, s26, v5       (Astro PC1082)
        0x876A6A1Bu,              // s_and_b32 vcc_lo, s27, vcc_lo
        0x876A6B6Au,              // s_and_b32 vcc_lo, vcc_lo, vcc_hi
        0x876A1C6Au,              // s_and_b32 vcc_lo, vcc_lo, s28
        0xBF840001u,              // s_cbranch_scc0 +1
        0xBF800000u,              // fallthrough work
    };
    compute_cfg_dispatch_mask_scc.insert(
        compute_cfg_dispatch_mask_scc.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_mask_scc.data(),
                             compute_cfg_dispatch_mask_scc.size(), &dispatch_rt,
                             wave32_dispatch_config).empty(),
          "the exact Wave32 dispatcher branches on the SCC from a B32 mask intersection");
    ComputeShaderConfig portable_mask_scc_config = wave32_dispatch_config;
    portable_mask_scc_config.native_subgroup_size = 0;
    CHECK(recompile_compute(compute_cfg_dispatch_mask_scc.data(),
                            compute_cfg_dispatch_mask_scc.size(), &dispatch_rt,
                            portable_mask_scc_config).empty(),
          "a portable dispatcher rejects unsynchronized Wave32 mask SCC instead of using stale state");
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
    // GTA V reaches S_FF1_I32_B64 inside this same exact-wave dispatcher family.  Prefix the
    // irreducible fixture with the live saved-VOPC packet and its scalar shift consumer: this
    // prevents a straight-line-only implementation from claiming the terminal program fixed.
    ComputeShaderConfig wave64_dispatch_config = wave32_dispatch_config;
    wave64_dispatch_config.wave_size = 64;
    wave64_dispatch_config.native_subgroup_size = 64;
    std::vector<uint32_t> compute_cfg_dispatch_ff1_b64 = {
        0x7E0202A8u,              // v_mov_b32 v1, 40
        0x7D8402F9u, 0x06069000u, // v_cmp_eq_u32_sdwa s[16:17], v0, v1
        0xBEEA1410u,              // GTA pc1486: s_ff1_i32_b64 vcc_lo, s[16:17]
        0x8F04816Au,              // s_lshl_b32 s4, vcc_lo, 1 (scalar-data consumer)
        0x7E040204u,              // v_mov_b32 v2, s4
    };
    compute_cfg_dispatch_ff1_b64.insert(
        compute_cfg_dispatch_ff1_b64.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_ff1_b64.data(),
                             compute_cfg_dispatch_ff1_b64.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "the exact Wave64 dispatcher lowers GTA's saved-mask S_FF1_I32_B64 site");
    // The immediate producer above does not cross a dispatcher edge.  GTA's later saved-mask sites
    // do: the dispatcher persists both the Bool-domain mask and synthetic scalar placeholders for
    // its physical SGPR pair.  A mask proven live on every predecessor must still reach the exact
    // S_FF1/S_BCNT reduction packet after that save/reload boundary.
    std::vector<uint32_t> compute_cfg_dispatch_persisted_ff1_b64 = {
        0xBE900381u, // earlier scalar lifetime: s_mov_b32 s16, 1
        0xBE910382u, //                          s_mov_b32 s17, 2
        0x7E060210u, // v_mov_b32 v3, s16
        0xBE8203A8u, // s_mov_b32 s2, 40
        0x7D840002u, // v_cmp_eq_u32 vcc, s2, v0
        0xBEFE046Au, // s_mov_b64 exec, vcc
        0xBE90246Au, // s_and_saveexec_b64 s[16:17], vcc
    };
    compute_cfg_dispatch_persisted_ff1_b64.insert(
        compute_cfg_dispatch_persisted_ff1_b64.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch) - 1);
    compute_cfg_dispatch_persisted_ff1_b64.insert(
        compute_cfg_dispatch_persisted_ff1_b64.end(), {
            0xBEFE0410u, // s_mov_b64 exec, s[16:17]
            0xBEEA1410u, // s_ff1_i32_b64 vcc_lo, s[16:17] (GTA PC1486)
            0x7E04026Au, // v_mov_b32 v2, vcc_lo
            0xBF810000u, // s_endpgm
        });
    CHECK(!recompile_compute(compute_cfg_dispatch_persisted_ff1_b64.data(),
                             compute_cfg_dispatch_persisted_ff1_b64.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "the Wave64 dispatcher preserves a saved mask through GTA's exact S_FF1 reduction");
    std::vector<uint32_t> compute_cfg_dispatch_persisted_bcnt_b64 = {
        0xBE860381u, // earlier scalar lifetime: s_mov_b32 s6, 1
        0xBE870382u, //                          s_mov_b32 s7, 2
        0x7E060206u, // v_mov_b32 v3, s6
        0xBE8203A8u, // s_mov_b32 s2, 40
        0x7D840002u, // v_cmp_eq_u32 vcc, s2, v0
        0xBEFE046Au, // s_mov_b64 exec, vcc
        0xBE86246Au, // s_and_saveexec_b64 s[6:7], vcc (GTA PC2005 producer)
    };
    compute_cfg_dispatch_persisted_bcnt_b64.insert(
        compute_cfg_dispatch_persisted_bcnt_b64.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch) - 1);
    compute_cfg_dispatch_persisted_bcnt_b64.insert(
        compute_cfg_dispatch_persisted_bcnt_b64.end(), {
            0xBEFE0406u, // s_mov_b64 exec, s[6:7]
            0xBEEA1006u, // s_bcnt1_i32_b64 vcc_lo, s[6:7] (GTA PC2007)
            0x7E04026Au, // v_mov_b32 v2, vcc_lo
            0xBF810000u, // s_endpgm
        });
    CHECK(!recompile_compute(compute_cfg_dispatch_persisted_bcnt_b64.data(),
                             compute_cfg_dispatch_persisted_bcnt_b64.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "the Wave64 dispatcher preserves a saved mask through GTA's exact S_BCNT reduction");
    std::vector<uint32_t> compute_cfg_dispatch_ambiguous_ff1_b64 = {
        0xBF068008u, // s_cmp_eq_u32 s8, 0
        0xBF840003u, // s_cbranch_scc0 +3 -> mask-producing arm
        0xBE900381u, // scalar-data arm: s_mov_b32 s16, 1
        0xBE910380u, //                  s_mov_b32 s17, 0
        0xBF820001u, // s_branch +1 -> merge
        0xBE900A7Eu, // mask arm: s_wqm_b64 s[16:17], exec (no scalar words)
    };
    compute_cfg_dispatch_ambiguous_ff1_b64.insert(
        compute_cfg_dispatch_ambiguous_ff1_b64.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch) - 1);
    compute_cfg_dispatch_ambiguous_ff1_b64.insert(
        compute_cfg_dispatch_ambiguous_ff1_b64.end(), {
            0xBE841410u, // merge: s_ff1_i32_b64 s4, s[16:17]
            0x7E040204u, // v_mov_b32 v2, s4
            0xBF810000u, // s_endpgm
        });
    CHECK(recompile_compute(compute_cfg_dispatch_ambiguous_ff1_b64.data(),
                            compute_cfg_dispatch_ambiguous_ff1_b64.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "a Wave64 mask/scalar join remains ambiguous at the exact S_FF1 consumer");

    // GTA V's exec_cs_413d22d00 reaches this exact PC105 SDWA packet after s[16:17] has joined a
    // saved-mask and scalar-data lifetime. PC102 then definitely overwrites only s16. The explicit
    // SRC1 is a one-dword scalar value, so the unresolved s17 half must not poison that read; the
    // implicit VCC predicate comes from the fresh compare immediately before it. Keep a crossing
    // tail so the fixture exercises the production Wave64 dispatcher dataflow rather than SSA.
    std::vector<uint32_t> compute_cfg_dispatch_resolved_b32_half = {
        0xBF068008u,                         // pc0: scalar branch condition
        0xBF840003u,                         // pc1: branch to mask-producing arm
        0xBE900381u,                         // pc2: scalar arm s16 = 1
        0xBE910380u,                         // pc3: scalar arm s17 = 0
        0xBF820001u,                         // pc4: merge
        0xBE900A7Eu,                         // pc5: mask-only WQM s[16:17] = exec
        0x7D840000u,                         // pc6: fresh implicit VCC predicate
        0xBE900381u,                         // pc7: exact B32 overwrite corresponding to live pc102
        0x020E20F9u, 0x86860680u,            // pc8: exact live pc105 cndmask SDWA reads s16
    };
    compute_cfg_dispatch_resolved_b32_half.insert(
        compute_cfg_dispatch_resolved_b32_half.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_resolved_b32_half.data(),
                             compute_cfg_dispatch_resolved_b32_half.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "a definite B32 overwrite resolves the exact half consumed after a Wave64 domain join");

    std::vector<uint32_t> compute_cfg_dispatch_wrong_b32_half =
        compute_cfg_dispatch_resolved_b32_half;
    compute_cfg_dispatch_wrong_b32_half[7] =
        0xBE910381u;                         // overwrite s17; the consumed s16 remains ambiguous
    CHECK(recompile_compute(compute_cfg_dispatch_wrong_b32_half.data(),
                            compute_cfg_dispatch_wrong_b32_half.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "overwriting the other pair half cannot validate GTA's exact one-dword consumer");

    std::vector<uint32_t> compute_cfg_dispatch_b64_after_b32_half =
        compute_cfg_dispatch_resolved_b32_half;
    compute_cfg_dispatch_b64_after_b32_half[8] =
        0xBE841410u;                         // s_ff1_i32_b64 s4,s[16:17]
    compute_cfg_dispatch_b64_after_b32_half[9] = 0xBF800000u;
    CHECK(recompile_compute(compute_cfg_dispatch_b64_after_b32_half.data(),
                            compute_cfg_dispatch_b64_after_b32_half.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "a B32 overwrite leaves the untouched half fail-visible to a whole-pair consumer");

    std::vector<uint32_t> compute_cfg_dispatch_flbit_b64_after_b32_half =
        compute_cfg_dispatch_resolved_b32_half;
    compute_cfg_dispatch_flbit_b64_after_b32_half[8] =
        0xBE841610u;                         // s_flbit_i32_b64 s4,s[16:17]
    compute_cfg_dispatch_flbit_b64_after_b32_half[9] = 0xBF800000u;
    CHECK(recompile_compute(compute_cfg_dispatch_flbit_b64_after_b32_half.data(),
                            compute_cfg_dispatch_flbit_b64_after_b32_half.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "a B64 flbit cannot consume a pair whose high half remains ambiguous");

    // Signed/unsigned 64-bit VOPC compares read both scalar source dwords. The raw signed-i64
    // opcode sits below the u64 window, so classify the effective opcode rather than treating all
    // high VOPC numbers as pairs (which also incorrectly catches adjacent f16 compares).
    std::vector<uint32_t> compute_cfg_dispatch_i64_compare_after_b32_half =
        compute_cfg_dispatch_resolved_b32_half;
    compute_cfg_dispatch_i64_compare_after_b32_half[8] =
        0xD4A2006Au;                         // v_cmp_eq_i64_e64 vcc,s[16:17],0
    compute_cfg_dispatch_i64_compare_after_b32_half[9] = 0x00010010u;
    CHECK(recompile_compute(compute_cfg_dispatch_i64_compare_after_b32_half.data(),
                            compute_cfg_dispatch_i64_compare_after_b32_half.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "a signed-i64 VOPC source cannot consume an unresolved high scalar half");

    // exec_cs_205b66f800's other member of this family is s_getpc_b64. Decode exposes the SDST
    // field as SRC0 for that SOP1 encoding, but architecturally getpc reads no scalar register and
    // replaces both destination words. Put it directly after the same domain join and retain a
    // separate valid embedded-table chain so production getpc admission stays fail-closed.
    std::vector<uint32_t> compute_cfg_dispatch_ambiguous_getpc = {
        0xBF068008u,                         // pc0: scalar branch condition
        0xBF840003u,                         // pc1: branch to mask-producing arm
        0xBE800381u,                         // pc2: scalar arm s0 = 1
        0xBE810380u,                         // pc3: scalar arm s1 = 0
        0xBF820001u,                         // pc4: merge
        0xBE800A7Eu,                         // pc5: mask-only WQM s[0:1] = exec
        0xBE801F00u,                         // pc6: exact s_getpc_b64 s[0:1], no source read
    };
    compute_cfg_dispatch_ambiguous_getpc.insert(
        compute_cfg_dispatch_ambiguous_getpc.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch) - 1);
    compute_cfg_dispatch_ambiguous_getpc.insert(
        compute_cfg_dispatch_ambiguous_getpc.end(), {
            0xBE841F00u,                     // pc31: independent folded getpc (next byte 128)
            0x800404A8u,                     // pc32: table byte 168 = next PC + 40
            0x82050580u,                     // pc33: s_addc_u32 s5,0,s5
            0xB0060010u,                     // pc34: stride/record field
            0xBE8703FFu, 0x10005004u,        // pc35: exact V# config
            0x7E020280u,                     // pc37: byte offset v1 = 0
            0xE0301000u, 0x80010101u,        // pc38: folded embedded-table load
            0xBF8C3F70u,                     // pc40: wait for the folded load
            0xBF810000u,                     // pc41: endpgm
            7u, 11u, 13u, 17u,               // pc42: embedded table
        });
    CHECK(!recompile_compute(compute_cfg_dispatch_ambiguous_getpc.data(),
                             compute_cfg_dispatch_ambiguous_getpc.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "folded s_getpc overwrites an ambiguous Wave64 pair without reading its fake SRC0");
    std::vector<uint32_t> compute_cfg_dispatch_getpc_real_pair_read =
        compute_cfg_dispatch_ambiguous_getpc;
    compute_cfg_dispatch_getpc_real_pair_read[6] =
        0xBE800400u;                         // same site: s_mov_b64 s[0:1],s[0:1] really reads it
    CHECK(recompile_compute(compute_cfg_dispatch_getpc_real_pair_read.data(),
                            compute_cfg_dispatch_getpc_real_pair_read.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "a real B64 source read at the getpc site still rejects the ambiguous pair");

    // V_READLANE decode exposes an unused SRC2=s0 after its real VGPR and scalar lane-selector
    // operands. GTA's exact pc261 packet writes s0, so treating that synthetic operand as a read of
    // the old ambiguous pair deadlocks the very instruction that replaces it.
    std::vector<uint32_t> compute_cfg_dispatch_readlane_fake_src2 = {
        0xBF068008u,                         // pc0: scalar branch condition
        0xBF840003u,                         // pc1: branch to mask-producing arm
        0xBE800381u,                         // pc2: scalar arm s0 = 1
        0xBE810380u,                         // pc3: scalar arm s1 = 0
        0xBF820001u,                         // pc4: merge
        0xBE800A7Eu,                         // pc5: mask-only WQM s[0:1] = exec
        0x7E0C0280u,                         // pc6: v6 = 0
        0xD7600000u, 0x00013F06u,            // pc7: exact GTA v_readlane_b32 s0,v6,31
    };
    compute_cfg_dispatch_readlane_fake_src2.insert(
        compute_cfg_dispatch_readlane_fake_src2.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_readlane_fake_src2.data(),
                             compute_cfg_dispatch_readlane_fake_src2.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "v_readlane ignores its decoded fake SRC2 before replacing the ambiguous destination");

    std::vector<uint32_t> compute_cfg_dispatch_readlane_true_ambiguous_selector =
        compute_cfg_dispatch_readlane_fake_src2;
    compute_cfg_dispatch_readlane_true_ambiguous_selector[8] =
        0x00000106u;                         // v_readlane_b32 s0,v6,s0: SRC1 really is read
    CHECK(recompile_compute(compute_cfg_dispatch_readlane_true_ambiguous_selector.data(),
                            compute_cfg_dispatch_readlane_true_ambiguous_selector.size(),
                            &dispatch_rt, wave64_dispatch_config).empty(),
          "v_readlane still rejects an ambiguous SGPR used as its real dynamic selector");

    // Three ordinary VOP3A arithmetic operations account for the other live GTA source-width
    // rejects. Each scalar source is one broadcast dword. Keep VCC_HI unresolved after the join
    // while replacing VCC_LO, then execute the exact observed packets so widening any of them to
    // Pair fails here.
    std::vector<uint32_t> compute_cfg_dispatch_exact_vop3_b32_sources = {
        0xBF068008u,                         // pc0: scalar branch condition
        0xBF840003u,                         // pc1: branch to mask-producing arm
        0xBEEA0381u,                         // pc2: scalar arm vcc_lo = 1
        0xBEEB0380u,                         // pc3: scalar arm vcc_hi = 0
        0xBF820001u,                         // pc4: merge
        0xBEEA0A7Eu,                         // pc5: mask-only WQM vcc = exec
        0xBEEA0380u,                         // pc6: definite B32 overwrite of vcc_lo only
        0xBE950381u,                         // pc7: s21 = 1
        0x7E020280u,                         // pc8: v1 = 0
        0x7E060280u,                         // pc9: v3 = 0
        0x7E180280u,                         // pc10: v12 = 0
        0xD543000Eu, 0x01A82B01u,            // pc11: exact v_mad_u32_u24, VCC_LO in SRC2
        0xD5418004u, 0x01AA18FFu, 0x39D1B717u, // pc13: exact v_mad_f32, VCC_LO in SRC2
        0xD7470002u, 0x020E066Au,            // pc16: exact v_add_lshl_u32, VCC_LO in SRC0
    };
    compute_cfg_dispatch_exact_vop3_b32_sources.insert(
        compute_cfg_dispatch_exact_vop3_b32_sources.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_exact_vop3_b32_sources.data(),
                             compute_cfg_dispatch_exact_vop3_b32_sources.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "GTA's exact ordinary VOP3A packets consume only the definitely replaced scalar dword");

    std::vector<uint32_t> compute_cfg_dispatch_vop3_143_ambiguous_source =
        compute_cfg_dispatch_exact_vop3_b32_sources;
    compute_cfg_dispatch_vop3_143_ambiguous_source[12] =
        0x01AC2B01u;                         // same 0x143 site: SRC2=vcc_hi remains ambiguous
    CHECK(recompile_compute(compute_cfg_dispatch_vop3_143_ambiguous_source.data(),
                            compute_cfg_dispatch_vop3_143_ambiguous_source.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "VOP3 opcode 0x143 rejects an unresolved scalar dword at its exact source field");
    std::vector<uint32_t> compute_cfg_dispatch_vop3_141_ambiguous_source =
        compute_cfg_dispatch_exact_vop3_b32_sources;
    compute_cfg_dispatch_vop3_141_ambiguous_source[14] =
        0x01AE18FFu;                         // same 0x141 site: SRC2=vcc_hi remains ambiguous
    CHECK(recompile_compute(compute_cfg_dispatch_vop3_141_ambiguous_source.data(),
                            compute_cfg_dispatch_vop3_141_ambiguous_source.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "VOP3 opcode 0x141 rejects an unresolved scalar dword at its exact source field");
    std::vector<uint32_t> compute_cfg_dispatch_vop3_347_ambiguous_source =
        compute_cfg_dispatch_exact_vop3_b32_sources;
    compute_cfg_dispatch_vop3_347_ambiguous_source[17] =
        0x020E066Bu;                         // same 0x347 site: SRC0=vcc_hi remains ambiguous
    CHECK(recompile_compute(compute_cfg_dispatch_vop3_347_ambiguous_source.data(),
                            compute_cfg_dispatch_vop3_347_ambiguous_source.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "VOP3 opcode 0x347 rejects an unresolved scalar dword at its exact source field");

    std::vector<uint32_t> compute_cfg_dispatch_vop3_wrong_b32_half =
        compute_cfg_dispatch_exact_vop3_b32_sources;
    compute_cfg_dispatch_vop3_wrong_b32_half[6] =
        0xBEEB0380u;                         // replace vcc_hi; VCC_LO remains ambiguous
    CHECK(recompile_compute(compute_cfg_dispatch_vop3_wrong_b32_half.data(),
                            compute_cfg_dispatch_vop3_wrong_b32_half.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "GTA's exact VOP3A consumers reject when only the other VCC half is replaced");

    // exec_cs_413d22d00 pc665 is V_CNDMASK_B32_E64 with scalar SRC1=s17. An unrelated
    // mask/scalar join leaves s18 ambiguous, so widening the one-dword s17 operand to a pair
    // incorrectly rejects this packet. Keep the production words intact and mutate SRC1 at that
    // same site to s18 so the negative arm still observes the genuinely ambiguous dword.
    std::vector<uint32_t> compute_cfg_dispatch_exact_vop3_cndmask_b32_source = {
        0xBF068008u,                         // pc0: scalar branch condition
        0xBF840003u,                         // pc1: branch to mask-producing arm
        0xBE920381u,                         // pc2: scalar arm s18 = 1
        0xBE930380u,                         // pc3: scalar arm s19 = 0
        0xBF820001u,                         // pc4: merge
        0xBE920A7Eu,                         // pc5: mask-only WQM s[18:19] = exec
        0xBE910381u,                         // pc6: definite scalar s17 = 1
        0x7E300280u,                         // pc7: v24 = 0
        0x7E2A0280u,                         // pc8: v21 = 0
        0x7D8414F9u, 0x06868E82u,            // pc9: exact live compare writes mask s[14:15]
        0xD5010015u, 0x00382318u,            // pc11: exact live cndmask v21,v24,s17,s[14:15]
    };
    compute_cfg_dispatch_exact_vop3_cndmask_b32_source.insert(
        compute_cfg_dispatch_exact_vop3_cndmask_b32_source.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_exact_vop3_cndmask_b32_source.data(),
                             compute_cfg_dispatch_exact_vop3_cndmask_b32_source.size(),
                             &dispatch_rt, wave64_dispatch_config).empty(),
          "GTA's exact VOP3 cndmask consumes only its scalar source dword");

    std::vector<uint32_t> compute_cfg_dispatch_vop3_cndmask_ambiguous_source =
        compute_cfg_dispatch_exact_vop3_cndmask_b32_source;
    compute_cfg_dispatch_vop3_cndmask_ambiguous_source[12] =
        0x00382518u;                         // same pc11 packet with SRC1=s18
    CHECK(recompile_compute(compute_cfg_dispatch_vop3_cndmask_ambiguous_source.data(),
                            compute_cfg_dispatch_vop3_cndmask_ambiguous_source.size(),
                            &dispatch_rt, wave64_dispatch_config).empty(),
          "VOP3 cndmask rejects an unresolved scalar dword at its exact source field");

    // exec_cs_205b654a00 pc2232 is V_LSHL_OR_B32 with scalar SRC0=s8. The preceding
    // V_READFIRSTLANE definitely replaces s8, while the adjacent s9 remains ambiguous after a
    // mask/scalar join. Every source of this integer VOP3A operation is one dword; mutating only
    // SRC0 at the production packet to s9 must still observe and reject the unresolved value.
    std::vector<uint32_t> compute_cfg_dispatch_exact_lshl_or_b32_source = {
        0xBF068008u,                         // pc0: scalar branch condition
        0xBF840003u,                         // pc1: branch to mask-producing arm
        0xBE880381u,                         // pc2: scalar arm s8 = 1
        0xBE890380u,                         // pc3: scalar arm s9 = 0
        0xBF820001u,                         // pc4: merge
        0xBE880A7Eu,                         // pc5: mask-only WQM s[8:9] = exec
        0x7E000280u,                         // pc6: v0 = 0
        0x7E100500u,                         // pc7: exact upstream readfirstlane s8,v0
        0xBE920384u, 0xBE930382u,            // pc8: scalar s[18:19] = {4,2}
        0xBE940384u, 0xBE950382u,            // pc10: scalar s[20:21] = {4,2}
        0xBF068008u,                         // pc12: define SCC for the pair selection
        0x85EA1412u,                         // pc13: materialize scalar pair in VCC
        0xD76F0002u, 0x01A92008u,            // pc14: exact lshl_or v2,s8,16,vcc_lo
        0xBE880480u,                         // pc16: clear the unrelated pair for the tail
    };
    const size_t lshl_or_dispatch_base =
        compute_cfg_dispatch_exact_lshl_or_b32_source.size();
    compute_cfg_dispatch_exact_lshl_or_b32_source.insert(
        compute_cfg_dispatch_exact_lshl_or_b32_source.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    // This fixture deliberately overwrites s8, which is also the synthetic tail's direct V#.
    // Preserve the tail's instruction count and CFG while removing that unrelated descriptor read.
    compute_cfg_dispatch_exact_lshl_or_b32_source[lshl_or_dispatch_base + 12] = 0xBF800000u;
    compute_cfg_dispatch_exact_lshl_or_b32_source[lshl_or_dispatch_base + 13] = 0xBF800000u;
    CHECK(!recompile_compute(compute_cfg_dispatch_exact_lshl_or_b32_source.data(),
                             compute_cfg_dispatch_exact_lshl_or_b32_source.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "GTA's exact VOP3 lshl_or consumes only its definitely replaced scalar dword");

    std::vector<uint32_t> compute_cfg_dispatch_lshl_or_ambiguous_source =
        compute_cfg_dispatch_exact_lshl_or_b32_source;
    compute_cfg_dispatch_lshl_or_ambiguous_source[15] =
        0x01A92009u;                         // same pc14 packet with SRC0=s9
    CHECK(recompile_compute(compute_cfg_dispatch_lshl_or_ambiguous_source.data(),
                            compute_cfg_dispatch_lshl_or_ambiguous_source.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "VOP3 lshl_or rejects an unresolved scalar dword at its exact source field");

    // exec_cs_205b658800 pc2546 is two-source V_MAC_F32 with real SRC0=s31. A scalar load
    // replaces through s31, but s32 remains ambiguous. Keep a second ambiguity at s0 so this
    // fixture also requires the decoder to suppress V_MAC's reserved SRC2 field. The same-site
    // mutation changes only real SRC0 to s32 and must remain fail-visible.
    std::vector<uint32_t> compute_cfg_dispatch_exact_vmac_b32_source = {
        0xBF068008u,                         // pc0: first scalar branch condition
        0xBF840003u,                         // pc1: branch to mask-producing arm
        0xBE800381u,                         // pc2: scalar arm s0 = 1
        0xBE810380u,                         // pc3: scalar arm s1 = 0
        0xBF820001u,                         // pc4: merge
        0xBE800A7Eu,                         // pc5: mask-only WQM s[0:1] = exec
        0xBF068008u,                         // pc6: second scalar branch condition
        0xBF840003u,                         // pc7: branch to mask-producing arm
        0xBEA00381u,                         // pc8: scalar arm s32 = 1
        0xBEA10380u,                         // pc9: scalar arm s33 = 0
        0xBF820001u,                         // pc10: merge
        0xBEA00A7Eu,                         // pc11: mask-only WQM s[32:33] = exec
        0xBE9F0381u,                         // pc12: definite scalar s31 = 1
        0x7E040280u,                         // pc13: v2 = 0
        0x7E1E0280u,                         // pc14: old v15 = 0
        0xD51F000Fu, 0x2002041Fu,            // pc15: exact v_mac v15,-s31,v2
        0xBE800480u,                         // pc17: clear s[0:1] for the tail
        0xBEA00480u,                         // pc18: clear s[32:33] for the tail
    };
    compute_cfg_dispatch_exact_vmac_b32_source.insert(
        compute_cfg_dispatch_exact_vmac_b32_source.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_exact_vmac_b32_source.data(),
                             compute_cfg_dispatch_exact_vmac_b32_source.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "GTA's exact VOP3 v_mac ignores reserved SRC2 and reads one scalar dword");

    std::vector<uint32_t> compute_cfg_dispatch_vmac_ambiguous_source =
        compute_cfg_dispatch_exact_vmac_b32_source;
    compute_cfg_dispatch_vmac_ambiguous_source[16] =
        0x20020420u;                         // same pc15 packet with real SRC0=s32
    CHECK(recompile_compute(compute_cfg_dispatch_vmac_ambiguous_source.data(),
                            compute_cfg_dispatch_vmac_ambiguous_source.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "VOP3 v_mac rejects an unresolved scalar dword at its exact source field");

    // A B64 logical may combine a proved wave mask with a complete scalar ballot pair. Its earlier
    // VOP3B join roots an ambiguous mask pair at odd s55: the scalar operand starts at base+1 and
    // only overlaps s56, which is definitely replaced. Validate actual read-range overlap, then
    // reconstruct this lane's Bool bit from the complete scalar pair under native Wave64.
    std::vector<uint32_t> compute_cfg_dispatch_exact_scalar_pair_mask = {
        0x7E000280u,                         // pc0: v0 = 0
        0x7E020280u,                         // pc1: v1 = 0
        0xBF068008u,                         // pc2: scalar branch condition
        0xBF840003u,                         // pc3: branch to mask-producing arm
        0xBEB70381u,                         // pc4: scalar arm s55 = 1
        0xBEB80380u,                         // pc5: scalar arm s56 = 0
        0xBF820002u,                         // pc6: merge after two-dword VOP3B
        0xD70F3700u, 0x00020300u,            // pc7: mask arm v_add_co_u32 ...,s55
        0xBEB6037Eu,                         // pc9: s54 = exec_lo
        0xBEB8037Eu,                         // pc10: s56 = exec_lo
        0xBEB9037Fu,                         // pc11: s57 = exec_hi
        0xBEEA047Eu,                         // pc12: vcc = exec
        0x87EA386Au,                         // pc13: exact live s_and_b64 vcc,vcc,s[56:57]
        0xBF068008u,                         // pc14: fresh SCC condition
        0xBF840001u,                         // pc15: conditional edge forces a dispatcher reload
        0xBF800000u,                         // pc16: fallthrough arm rejoins the consumer
        0xBE84146Au,                         // pc17: s_ff1_i32_b64 s4,vcc after reload
        0x7E040204u,                         // pc18: consume scalar reduction result
    };
    compute_cfg_dispatch_exact_scalar_pair_mask.insert(
        compute_cfg_dispatch_exact_scalar_pair_mask.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_exact_scalar_pair_mask.data(),
                             compute_cfg_dispatch_exact_scalar_pair_mask.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "GTA's exact B64 logical projects a fully materialized scalar ballot pair to a mask");

    std::vector<uint32_t> compute_cfg_dispatch_shifted_unresolved_pair =
        compute_cfg_dispatch_exact_scalar_pair_mask;
    compute_cfg_dispatch_shifted_unresolved_pair[13] =
        0x87EA366Au;                         // same site reads s[54:55], overlapping unresolved s55
    CHECK(recompile_compute(compute_cfg_dispatch_shifted_unresolved_pair.data(),
                            compute_cfg_dispatch_shifted_unresolved_pair.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "a pair rooted before an odd ambiguous mask rejects its unresolved overlapping word");

    // exec_cs_205b5e8600 pc1467 has a different, exact mixed-domain VCC join. The first-entry arm
    // executes `s_mov_b64 vcc,exec`; the loop backedge reconstructs VCC_LO/HI with v_readlane.
    // Native-Wave64 S_MOV materializes ballot words, so both predecessors carry exact scalar words
    // and the live s_and_b64 can project that common pair back to its per-lane predicate. Replacing
    // only the entry producer with mask-only WQM removes those words and must reject at the join.
    std::vector<uint32_t> compute_cfg_dispatch_live_mixed_vcc_pair = {
        0xBE800380u,                         // pc0: s0 = 0
        0x7E000280u, 0x7E020280u,            // pc1: v0/v1 = 0
        0xBF068008u,                         // pc3: unresolved choice keeps both arms reachable
        0xBF840002u,                         // pc4: branch to v_readlane arm
        0xBEEA047Eu,                         // pc5: exact live producer s_mov_b64 vcc,exec
        0xBF820004u,                         // pc6: skip scalar reconstruction
        0xD760006Au, 0x00010100u,            // pc7: v_readlane_b32 vcc_lo,v0,0
        0xD760006Bu, 0x00010101u,            // pc9: v_readlane_b32 vcc_hi,v1,0
        0xBEB8037Eu, 0xBEB9037Fu,            // pc11: scalar ballot pair s[56:57]
        0x87EA386Au,                         // pc13: exact live AND vcc,vcc,s[56:57]
    };
    compute_cfg_dispatch_live_mixed_vcc_pair.insert(
        compute_cfg_dispatch_live_mixed_vcc_pair.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_live_mixed_vcc_pair.data(),
                             compute_cfg_dispatch_live_mixed_vcc_pair.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "GTA's exact mixed VCC join reaches pc1467 through native ballot words");
    std::vector<uint32_t> compute_cfg_dispatch_unmaterialized_vcc_pair =
        compute_cfg_dispatch_live_mixed_vcc_pair;
    compute_cfg_dispatch_unmaterialized_vcc_pair[5] =
        0xBEEA0A7Eu;                         // same producer: WQM retains only the mask view
    CHECK(recompile_compute(compute_cfg_dispatch_unmaterialized_vcc_pair.data(),
                            compute_cfg_dispatch_unmaterialized_vcc_pair.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "the mixed VCC join rejects when its entry producer lacks scalar ballot words");

    // The next site in exec_cs_205b5e8600 joins early exits that retain scalar EXEC ballot words
    // s[56:57] with the normal path's exact pc1798 S_ANDN2_B64 survivor mask. Under native Wave64,
    // the logical result has the same exact ballot-word representation as the early path, so the
    // pc1802 S_CMP_EQ_U64 consumer is safe. Replacing only pc1798 with mask-only WQM removes those
    // scalar words and must reject at the unchanged full-pair consumer.
    std::vector<uint32_t> compute_cfg_dispatch_logical_ballot_join = {
        0xBEB8037Eu,                         // pc0: scalar s56 = exec_lo
        0xBEB9037Fu,                         // pc1: scalar s57 = exec_hi
        0x7E400280u,                         // pc2: v32 = 0
        0xBF068008u,                         // pc3: unresolved choice keeps both paths
        0xBF840003u,                         // pc4: early path skips pc5-pc7 to join
        0x7D8A40C1u,                         // pc5: exact pc1797 compare creates VCC
        0x8AB86A38u,                         // pc6: exact pc1798 andn2 s56,s56,vcc
        0xBF800000u,                         // pc7: normal path reaches join
        0xBF128038u,                         // pc8: exact pc1802 cmp_eq_u64 s[56:57],0
    };
    compute_cfg_dispatch_logical_ballot_join.insert(
        compute_cfg_dispatch_logical_ballot_join.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_logical_ballot_join.data(),
                             compute_cfg_dispatch_logical_ballot_join.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "native Wave64 retains exact scalar ballot words for a B64 logical result");

    std::vector<uint32_t> compute_cfg_dispatch_mask_only_logical_join =
        compute_cfg_dispatch_logical_ballot_join;
    compute_cfg_dispatch_mask_only_logical_join[6] =
        0xBEB80A7Eu;                         // same pc6: WQM has no scalar-word view
    CHECK(recompile_compute(compute_cfg_dispatch_mask_only_logical_join.data(),
                            compute_cfg_dispatch_mask_only_logical_join.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "a mask-only mutation cannot supply pc1802's scalar pair");

    // A fresh VCC mask can itself overlap the high word of an older odd-rooted ambiguity. The exact
    // pc1467 packet reads VCC in the Bool domain and s[56:57] in the scalar domain, so only the latter
    // needs scalar-word facts. Mutating SRC0 at that same site to s[104:105] makes the unresolved s105
    // word a real scalar-pair read and must reject.
    std::vector<uint32_t> compute_cfg_dispatch_fresh_vcc_over_odd_ambiguity = {
        0x7E000280u,                         // pc0: v0 = 0
        0x7E020280u,                         // pc1: v1 = 0
        0xBF068008u,                         // pc2: scalar branch condition
        0xBF840003u,                         // pc3: branch to odd mask-producing arm
        0xBEE90381u,                         // pc4: scalar arm s105 = 1
        0xBEEA0380u,                         // pc5: scalar arm vcc_lo = 0
        0xBF820002u,                         // pc6: merge after two-dword VOP3B
        0xD70F6900u, 0x00020300u,            // pc7: mask arm v_add_co_u32 ...,s105
        0xBEB8037Eu,                         // pc9: s56 = exec_lo
        0xBEB9037Fu,                         // pc10: s57 = exec_hi
        0xBEEA047Eu,                         // pc11: fresh vcc = exec, overlapping old root105
        0x87EA386Au,                         // pc12: exact live s_and_b64 vcc,vcc,s[56:57]
    };
    compute_cfg_dispatch_fresh_vcc_over_odd_ambiguity.insert(
        compute_cfg_dispatch_fresh_vcc_over_odd_ambiguity.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_fresh_vcc_over_odd_ambiguity.data(),
                             compute_cfg_dispatch_fresh_vcc_over_odd_ambiguity.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "a proved VCC mask ignores an older overlapping scalar-domain ambiguity");
    std::vector<uint32_t> compute_cfg_dispatch_real_odd_scalar_source =
        compute_cfg_dispatch_fresh_vcc_over_odd_ambiguity;
    compute_cfg_dispatch_real_odd_scalar_source[12] =
        0x87EA3868u;                         // same site: SRC0=s[104:105], s105 unresolved
    CHECK(recompile_compute(compute_cfg_dispatch_real_odd_scalar_source.data(),
                            compute_cfg_dispatch_real_odd_scalar_source.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "the same B64 logical rejects when its overlapping source is a real scalar pair");

    // The next live site in exec_cs_205b654a00 is V_MBCNT_HI with mask word s1 and a fake SRC2=s0.
    // Each MBCNT half consumes one B32 mask word; widening s1 to a pair incorrectly reaches the
    // unrelated ambiguity rooted at s2. The source mutation to s3 makes that ambiguity real.
    std::vector<uint32_t> compute_cfg_dispatch_exact_mbcnt_hi_word = {
        0xBE80047Eu,                         // pc0: saved mask s[0:1] = exec
        0xBF068008u,                         // pc1: scalar branch condition
        0xBF840003u,                         // pc2: branch to mask-producing arm
        0xBE820381u,                         // pc3: scalar arm s2 = 1
        0xBE830380u,                         // pc4: scalar arm s3 = 0
        0xBF820001u,                         // pc5: merge
        0xBE820A7Eu,                         // pc6: mask-only WQM s[2:3] = exec
        0xD7660003u, 0x00010001u,            // pc7: exact pc2036 mbcnt_hi v3,s1,0
        0xBE820480u,                         // pc9: resolve s[2:3] after the observed packet
    };
    compute_cfg_dispatch_exact_mbcnt_hi_word.insert(
        compute_cfg_dispatch_exact_mbcnt_hi_word.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_exact_mbcnt_hi_word.data(),
                             compute_cfg_dispatch_exact_mbcnt_hi_word.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "GTA's exact MBCNT_HI consumes one proved mask word beside an unrelated ambiguity");
    std::vector<uint32_t> compute_cfg_dispatch_ambiguous_mbcnt_hi_word =
        compute_cfg_dispatch_exact_mbcnt_hi_word;
    compute_cfg_dispatch_ambiguous_mbcnt_hi_word[8] =
        0x00010003u;                         // same site: MBCNT_HI now names ambiguous s3
    CHECK(recompile_compute(compute_cfg_dispatch_ambiguous_mbcnt_hi_word.data(),
                            compute_cfg_dispatch_ambiguous_mbcnt_hi_word.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "MBCNT_HI rejects an unresolved mask/scalar word at the same source site");

    // The final live site joins an exact scalar load against `s_mov_b64 s[16:17], 0`. The constant
    // move has both an empty-mask view and the identical two-word scalar value in production. A
    // mask-only EXEC move at the same site is the mutation arm and must remain rejected.
    std::vector<uint32_t> compute_cfg_dispatch_dual_domain_b64_constant = {
        0xBF068008u,                         // pc0: scalar branch condition
        0xBF840003u,                         // pc1: branch to constant-mask arm
        0xBE900380u,                         // pc2: scalar arm s16 = 0
        0xBE910380u,                         // pc3: scalar arm s17 = 0
        0xBF820001u,                         // pc4: merge
        0xBE900480u,                         // pc5: s_mov_b64 s[16:17], 0 in both domains
        0x7E000210u,                         // pc6: v0 = s16
        0x7E020211u,                         // pc7: v1 = s17
    };
    compute_cfg_dispatch_dual_domain_b64_constant.insert(
        compute_cfg_dispatch_dual_domain_b64_constant.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_dual_domain_b64_constant.data(),
                             compute_cfg_dispatch_dual_domain_b64_constant.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "an inline B64 constant remains scalar-readable across a mask/scalar join");

    std::vector<uint32_t> compute_cfg_dispatch_mask_only_b64_move =
        compute_cfg_dispatch_dual_domain_b64_constant;
    compute_cfg_dispatch_mask_only_b64_move[5] =
        0xBE900A7Eu;                         // s_wqm_b64 s[16:17],exec has no scalar-word view
    CHECK(recompile_compute(compute_cfg_dispatch_mask_only_b64_move.data(),
                            compute_cfg_dispatch_mask_only_b64_move.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "a mask-only B64 move cannot masquerade as scalar data at the same consumer");

    // A definite mask lifetime (as opposed to an ambiguous join) must survive the dispatcher's
    // Function-variable round trip without its synthetic scalar-zero placeholders shadowing the
    // Bool value. Exact native Wave64 can materialize either physical word from that Bool through a
    // ballot; the portable route must still reject this generic scalar use. Changing the producer
    // to an inline constant gives both words an ordinary scalar lifetime on either route.
    std::vector<uint32_t> compute_cfg_dispatch_mask_only_scalar_reload = {
        0xBE900A7Eu,                         // pc0: mask-only s_wqm_b64 s[16:17],exec
        0xBF068008u,                         // pc1: fresh SCC condition
        0xBF840001u,                         // pc2: conditional dispatcher edge to consumer
        0xBF800000u,                         // pc3: fallthrough arm rejoins consumer
        0x7E040210u,                         // pc4: physical low-dword read v2=s16 after reload
    };
    compute_cfg_dispatch_mask_only_scalar_reload.insert(
        compute_cfg_dispatch_mask_only_scalar_reload.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_mask_only_scalar_reload.data(),
                             compute_cfg_dispatch_mask_only_scalar_reload.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "exact native Wave64 materializes a reloaded saved-mask physical dword");
    ComputeShaderConfig portable_mask_scalar_read_config = wave64_dispatch_config;
    portable_mask_scalar_read_config.native_subgroup_size = 0;
    CHECK(recompile_compute(compute_cfg_dispatch_mask_only_scalar_reload.data(),
                            compute_cfg_dispatch_mask_only_scalar_reload.size(), &dispatch_rt,
                            portable_mask_scalar_read_config).empty(),
          "portable Wave64 rejects a generic saved-mask scalar read without a synchronized phase");
    std::vector<uint32_t> compute_cfg_dispatch_dual_scalar_reload =
        compute_cfg_dispatch_mask_only_scalar_reload;
    compute_cfg_dispatch_dual_scalar_reload[0] =
        0xBE900480u;                         // same producer: inline zero has real scalar words
    CHECK(!recompile_compute(compute_cfg_dispatch_dual_scalar_reload.data(),
                             compute_cfg_dispatch_dual_scalar_reload.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "an inline B64 constant retains both scalar words across a dispatcher reload");

    // Native Wave64 S_MOV_B64 materializes an exact ballot pair as well as its Bool alias, even
    // when its saved-SGPR source has only a mask representation. That copy must retain both words
    // across a real dispatcher boundary. Replacing the copy with mask-only WQM at the exact site
    // removes the scalar representation, so CSELECT must reject instead of reading zeros.
    std::vector<uint32_t> compute_cfg_dispatch_dual_mask_move_reload = {
        0xBE900A7Eu,                         // pc0: mask-only saved s[16:17] = WQM(exec)
        0xBE920410u,                         // pc1: S_MOV materializes s[18:19] ballot words
        0xBE940381u, 0xBE950380u,            // pc2: scalar-only s[20:21] = 1
        0xBF068008u,                         // pc4: fresh SCC condition
        0xBF840001u,                         // pc5: conditional dispatcher edge
        0xBF800000u,                         // pc6: fallthrough rejoins consumer
        0x85EA1412u,                         // pc7: cselect vcc,s[18:19],s[20:21]
    };
    compute_cfg_dispatch_dual_mask_move_reload.insert(
        compute_cfg_dispatch_dual_mask_move_reload.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_dual_mask_move_reload.data(),
                             compute_cfg_dispatch_dual_mask_move_reload.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "a dual-domain S_MOV_B64 copy retains scalar words across a dispatcher reload");
    std::vector<uint32_t> compute_cfg_dispatch_mask_only_move_reload =
        compute_cfg_dispatch_dual_mask_move_reload;
    compute_cfg_dispatch_mask_only_move_reload[1] =
        0xBE920A7Eu;                         // same site: WQM writes a mask without scalar words
    CHECK(recompile_compute(compute_cfg_dispatch_mask_only_move_reload.data(),
                            compute_cfg_dispatch_mask_only_move_reload.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "the same mixed-domain consumer rejects after a mask-only producer mutation");

    // A complete scalar S_CSELECT_B64 into VCC also emits both views: its selected pair remains
    // scalar-readable while VCC consumers use the lane predicate. Force a second CSELECT after a
    // boundary to require both views. Mutating the first CSELECT to choose between two masks takes
    // its mask-only branch and makes that downstream scalar-pair use fail visibly.
    std::vector<uint32_t> compute_cfg_dispatch_dual_cselect_reload = {
        0xBE920381u, 0xBE930380u,            // pc0: scalar-only s[18:19] = 1
        0xBE940382u, 0xBE950380u,            // pc2: scalar-only s[20:21] = 2
        0xBF068008u,                         // pc4: define SCC for first cselect
        0x85EA1412u,                         // pc5: dual cselect vcc,s18,s20
        0xBF068008u,                         // pc6: fresh branch SCC
        0xBF840001u,                         // pc7: conditional dispatcher edge
        0xBF800000u,                         // pc8: fallthrough rejoins consumer
        0x8598146Au,                         // pc9: cselect s[24:25],vcc,s[20:21]
    };
    compute_cfg_dispatch_dual_cselect_reload.insert(
        compute_cfg_dispatch_dual_cselect_reload.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_dual_cselect_reload.data(),
                             compute_cfg_dispatch_dual_cselect_reload.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "a scalar-pair CSELECT into VCC retains both domains across a dispatcher reload");
    std::vector<uint32_t> compute_cfg_dispatch_mask_cselect_reload =
        compute_cfg_dispatch_dual_cselect_reload;
    compute_cfg_dispatch_mask_cselect_reload[5] =
        0x85EA807Eu;                         // same site: cselect vcc,exec,0 is mask-only
    CHECK(recompile_compute(compute_cfg_dispatch_mask_cselect_reload.data(),
                            compute_cfg_dispatch_mask_cselect_reload.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "a mask-only CSELECT mutation cannot supply scalar words after reload");

    // exec_cs_205b654a00 pc2121 stores through T# s[32:39]. IMAGE_STORE has no sampler, although
    // generic MIMG decode exposes the reserved SSAMP field as SRC2=s0. Keep s[0:1] ambiguous at the
    // exact packet: the storage form must ignore it, while a same-site IMAGE_SAMPLE mutation makes
    // S# s[0:3] real and must reject. Clear the pair only after the packet so the appended dispatcher
    // can continue without turning a successfully ignored source into a later unrelated rejection.
    std::vector<uint32_t> compute_cfg_dispatch_image_store_fake_sampler = {
        0xBF068008u,                         // pc0: scalar branch condition
        0xBF840003u,                         // pc1: branch to mask-producing arm
        0xBE800381u,                         // pc2: scalar arm s0 = 1
        0xBE810380u,                         // pc3: scalar arm s1 = 0
        0xBF820001u,                         // pc4: merge
        0xBE800A7Eu,                         // pc5: mask-only WQM s[0:1] = exec
        0x7E000280u, 0x7E020280u,            // pc6: v0/v1 = 0
        0x7E040280u, 0x7E060280u,            // pc8: v2/v3 = 0
        0x7E180280u, 0x7E1A0280u,            // pc10: v12/v13 = 0
        0xF0200F08u, 0x0008000Cu,            // pc12: exact image_store, T# s32, fake S# s0
        0xBE800480u,                         // pc14: resolve s[0:1] after the observed packet
    };
    compute_cfg_dispatch_image_store_fake_sampler.insert(
        compute_cfg_dispatch_image_store_fake_sampler.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    ShaderResourceTable wave64_store_rt = dispatch_rt;
    { ShaderResource image{}; image.cls = ResourceClass::StorageImage;
      image.format = DataFormat::Float32; image.num_components = 4;
      image.binding = 4; image.fetch_pc = 12; image.sgpr_base = 32; image.img_dim = 1;
      image.width = image.height = image.depth = 1; wave64_store_rt.resources.push_back(image); }
    ComputeShaderConfig wave64_store_config = wave64_dispatch_config;
    wave64_store_config.user_sgprs.resize(40);
    CHECK(!recompile_compute(compute_cfg_dispatch_image_store_fake_sampler.data(),
                             compute_cfg_dispatch_image_store_fake_sampler.size(), &wave64_store_rt,
                             wave64_store_config).empty(),
          "GTA's exact IMAGE_STORE ignores its decoded fake sampler source");

    std::vector<uint32_t> compute_cfg_dispatch_image_sample_real_sampler =
        compute_cfg_dispatch_image_store_fake_sampler;
    compute_cfg_dispatch_image_sample_real_sampler[12] =
        0xF0800F08u;                         // same site: image_sample now consumes S# s[0:3]
    ShaderResourceTable wave64_sample_rt = dispatch_rt;
    { ShaderResource image{}; image.cls = ResourceClass::Texture;
      image.format = DataFormat::Float32; image.num_components = 4;
      image.binding = 4; image.fetch_pc = 12; image.sgpr_base = 32;
      image.sampler_sgpr_base = 0; image.img_dim = 1;
      image.width = image.height = image.depth = 1; wave64_sample_rt.resources.push_back(image); }
    CHECK(recompile_compute(compute_cfg_dispatch_image_sample_real_sampler.data(),
                            compute_cfg_dispatch_image_sample_real_sampler.size(), &wave64_sample_rt,
                            wave64_store_config).empty(),
          "a sampled-image mutation rejects the same ambiguous SGPRs when they become a real S#");

    std::vector<uint32_t> compute_cfg_dispatch_image_store_ambiguous_resource =
        compute_cfg_dispatch_image_store_fake_sampler;
    compute_cfg_dispatch_image_store_ambiguous_resource[13] =
        0x0000000Cu;                         // same site: real eight-word T# moves onto s[0:7]
    ShaderResourceTable wave64_ambiguous_store_rt = dispatch_rt;
    { ShaderResource image{}; image.cls = ResourceClass::StorageImage;
      image.format = DataFormat::Float32; image.num_components = 4;
      image.binding = 4; image.fetch_pc = 12; image.sgpr_base = 0; image.img_dim = 1;
      image.width = image.height = image.depth = 1;
      wave64_ambiguous_store_rt.resources.push_back(image); }
    CHECK(recompile_compute(compute_cfg_dispatch_image_store_ambiguous_resource.data(),
                            compute_cfg_dispatch_image_store_ambiguous_resource.size(),
                            &wave64_ambiguous_store_rt, wave64_store_config).empty(),
          "IMAGE_STORE still rejects an ambiguous real T# while ignoring only its fake S#");

    // S_BUFFER_* reads a complete four-word descriptor. Put an unresolved pair only in the upper
    // half of s[16:19]: treating SBASE as an ordinary two-word scalar address would miss the overlap
    // and silently admit it. Moving that exact packet's SBASE to non-overlapping s[20:23] is the
    // same-site control arm and must compile with the corresponding captured resource.
    std::vector<uint32_t> compute_cfg_dispatch_sbuffer_quad_descriptor = {
        0xBF068008u,                         // pc0: scalar branch condition
        0xBF840003u,                         // pc1: branch to mask-only arm
        0xBE920381u, 0xBE930380u,            // pc2: scalar s[18:19]
        0xBF820001u,                         // pc4: merge
        0xBE920A7Eu,                         // pc5: mask-only WQM s[18:19]
        0xF4200708u, 0xFA000000u,            // pc6: s_buffer_load_dword s28,s[16:19],0
        0xBE920480u,                         // pc8: resolve the pair after the exact packet
    };
    compute_cfg_dispatch_sbuffer_quad_descriptor.insert(
        compute_cfg_dispatch_sbuffer_quad_descriptor.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    ShaderResourceTable wave64_sbuffer_rt = dispatch_rt;
    { ShaderResource constants{}; constants.cls = ResourceClass::ConstantBuffer;
      constants.binding = 5; constants.size = 64; constants.fetch_pc = 6;
      constants.sgpr_base = 16; wave64_sbuffer_rt.resources.push_back(constants); }
    ComputeShaderConfig wave64_descriptor_config = wave64_dispatch_config;
    wave64_descriptor_config.user_sgprs.resize(24);
    CHECK(recompile_compute(compute_cfg_dispatch_sbuffer_quad_descriptor.data(),
                            compute_cfg_dispatch_sbuffer_quad_descriptor.size(),
                            &wave64_sbuffer_rt, wave64_descriptor_config).empty(),
          "S_BUFFER rejects ambiguity in the upper half of its four-word descriptor");
    std::vector<uint32_t> compute_cfg_dispatch_sbuffer_nonoverlap =
        compute_cfg_dispatch_sbuffer_quad_descriptor;
    compute_cfg_dispatch_sbuffer_nonoverlap[6] =
        0xF420070Au;                         // same site: SBASE moves to s[20:23]
    ShaderResourceTable wave64_sbuffer_nonoverlap_rt = wave64_sbuffer_rt;
    wave64_sbuffer_nonoverlap_rt.resources.back().sgpr_base = 20;
    CHECK(!recompile_compute(compute_cfg_dispatch_sbuffer_nonoverlap.data(),
                             compute_cfg_dispatch_sbuffer_nonoverlap.size(),
                             &wave64_sbuffer_nonoverlap_rt,
                             wave64_descriptor_config).empty(),
          "S_BUFFER accepts the same packet when all four descriptor words are unambiguous");

    // IMAGE_BVH_INTERSECT_RAY is the one MIMG opcode whose SRSRC is a compact four-word BVH
    // descriptor rather than an eight-word T#. An ambiguity beginning immediately after s[8:11]
    // is therefore irrelevant; moving the exact packet's SRSRC to s[12:15] makes that same pair
    // observable and must reject. The reserved sampler field remains non-operational for BVH.
    std::vector<uint32_t> compute_cfg_dispatch_bvh_quad_descriptor = {
        0xBF068008u,                         // pc0: scalar branch condition
        0xBF840003u,                         // pc1: branch to mask-only arm
        0xBE8C0381u, 0xBE8D0380u,            // pc2: scalar s[12:13]
        0xBF820001u,                         // pc4: merge
        0xBE8C0A7Eu,                         // pc5: mask-only WQM s[12:13]
        0xF1989F07u, 0x00020202u,            // pc6: BVH packet, SRSRC=s[8:11]
        0x28292C23u, 0x22262725u, 0x00002A24u,
        0xBE8C0480u,                         // pc11: resolve after the packet
    };
    compute_cfg_dispatch_bvh_quad_descriptor.insert(
        compute_cfg_dispatch_bvh_quad_descriptor.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    uint32_t wave64_bvh_words[16]{};
    ShaderResourceTable wave64_bvh_rt = dispatch_rt;
    { ShaderResource bvh{}; bvh.cls = ResourceClass::ConstantBuffer;
      bvh.format = DataFormat::Uint32; bvh.num_components = 1;
      bvh.binding = 6; bvh.size = sizeof(wave64_bvh_words); bvh.fetch_pc = 6;
      bvh.sgpr_base = 8; bvh.host_data = reinterpret_cast<uint8_t*>(wave64_bvh_words);
      bvh.host_data_size = sizeof(wave64_bvh_words); wave64_bvh_rt.resources.push_back(bvh); }
    CHECK(!recompile_compute(compute_cfg_dispatch_bvh_quad_descriptor.data(),
                             compute_cfg_dispatch_bvh_quad_descriptor.size(), &wave64_bvh_rt,
                             wave64_descriptor_config).empty(),
          "BVH ignores ambiguity immediately beyond its compact four-word descriptor");
    std::vector<uint32_t> compute_cfg_dispatch_bvh_ambiguous_descriptor =
        compute_cfg_dispatch_bvh_quad_descriptor;
    compute_cfg_dispatch_bvh_ambiguous_descriptor[7] =
        0x00030202u;                         // same site: SRSRC=s[12:15]
    ShaderResourceTable wave64_bvh_ambiguous_rt = wave64_bvh_rt;
    wave64_bvh_ambiguous_rt.resources.back().sgpr_base = 12;
    CHECK(recompile_compute(compute_cfg_dispatch_bvh_ambiguous_descriptor.data(),
                            compute_cfg_dispatch_bvh_ambiguous_descriptor.size(),
                            &wave64_bvh_ambiguous_rt, wave64_descriptor_config).empty(),
          "BVH rejects ambiguity inside its exact four-word descriptor");
    // GTA V also consumes saved EXEC halves through V_MBCNT after dispatcher boundaries. LOW names
    // the pair root and HIGH names its following dword; both consumers must be admitted only where
    // the Wave64 MUST-domain proof says that same B64 mask lifetime reaches the exact PC.
    std::vector<uint32_t> compute_cfg_dispatch_persisted_mbcnt_b64 = {
        0xBE900381u, // earlier scalar lifetime: s_mov_b32 s16, 1
        0xBE910382u, //                          s_mov_b32 s17, 2
        0xBE90047Eu, // s_mov_b64 s[16:17], exec (saved-mask lifetime begins)
    };
    compute_cfg_dispatch_persisted_mbcnt_b64.insert(
        compute_cfg_dispatch_persisted_mbcnt_b64.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch) - 1);
    compute_cfg_dispatch_persisted_mbcnt_b64.insert(
        compute_cfg_dispatch_persisted_mbcnt_b64.end(), {
            0xBEFE0410u,              // s_mov_b64 exec, s[16:17]
            0xD7650003u, 0x00010010u, // v_mbcnt_lo_u32_b32 v3, s16, 0
            0xD7660003u, 0x00020611u, // v_mbcnt_hi_u32_b32 v3, s17, v3
            0xBF810000u,
        });
    CHECK(!recompile_compute(compute_cfg_dispatch_persisted_mbcnt_b64.data(),
                             compute_cfg_dispatch_persisted_mbcnt_b64.size(), &dispatch_rt,
                             wave64_dispatch_config).empty(),
          "the native Wave64 dispatcher preserves saved EXEC through LOW/even and HIGH/odd MBCNT");
    ComputeShaderConfig portable_wave64_dispatch_config = wave64_dispatch_config;
    portable_wave64_dispatch_config.native_subgroup_size = 0;
    CHECK(!recompile_compute(compute_cfg_dispatch_persisted_mbcnt_b64.data(),
                             compute_cfg_dispatch_persisted_mbcnt_b64.size(), &dispatch_rt,
                             portable_wave64_dispatch_config).empty(),
          "the portable Wave64 dispatcher preserves saved EXEC through synchronized MBCNT");

    std::vector<uint32_t> compute_cfg_dispatch_overwritten_mbcnt_b64 =
        compute_cfg_dispatch_persisted_mbcnt_b64;
    compute_cfg_dispatch_overwritten_mbcnt_b64[
        compute_cfg_dispatch_overwritten_mbcnt_b64.size() - 6] =
        0xBE910380u; // replace the restore with an s17 overwrite immediately before MBCNT
    CHECK(recompile_compute(compute_cfg_dispatch_overwritten_mbcnt_b64.data(),
                            compute_cfg_dispatch_overwritten_mbcnt_b64.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "saved-pair MBCNT rejects after an overlapping high-half scalar overwrite");

    std::vector<uint32_t> compute_cfg_dispatch_ambiguous_mbcnt_b64(
        compute_cfg_dispatch_persisted_mbcnt_b64.begin(),
        compute_cfg_dispatch_persisted_mbcnt_b64.end() - 6);
    compute_cfg_dispatch_ambiguous_mbcnt_b64.insert(
        compute_cfg_dispatch_ambiguous_mbcnt_b64.end(), {
            0xBF068014u, // s_cmp_eq_u32 s20, 0
            0xBF840001u, // s_cbranch_scc0 +1: one predecessor preserves the mask
            0xBE910380u, // other predecessor overwrites s17
        });
    compute_cfg_dispatch_ambiguous_mbcnt_b64.insert(
        compute_cfg_dispatch_ambiguous_mbcnt_b64.end(),
        compute_cfg_dispatch_persisted_mbcnt_b64.end() - 6,
        compute_cfg_dispatch_persisted_mbcnt_b64.end());
    CHECK(recompile_compute(compute_cfg_dispatch_ambiguous_mbcnt_b64.data(),
                            compute_cfg_dispatch_ambiguous_mbcnt_b64.size(), &dispatch_rt,
                            wave64_dispatch_config).empty(),
          "saved-pair MBCNT rejects a mask/scalar predecessor join at its exact consumer");

    // A forward exact-wave branch normally remains on the compact structured path. A real MBCNT
    // after its merge still needs the dispatcher's synchronized common phase (the all-ones
    // lane-index idiom remains locally scalarizable and does not take this route).
    const uint32_t compute_structured_saved_mbcnt_b64[] = {
        0xBE84047Eu,                // s_mov_b64 s[4:5], exec
        0x7E060280u,                // v_mov_b32 v3, 0 (merge live-in)
        0x7C020300u,                // v_cmp_lt_f32 vcc, v0, v1 (varying guest-wave branch)
        0xBF860001u,                // s_cbranch_vccz +1 -> merge
        0x7E060281u,                // guarded write keeps the structured branch live
        0xD7650003u, 0x00010004u,   // v_mbcnt_lo_u32_b32 v3, s4, 0
        0xD7660003u, 0x00020605u,   // v_mbcnt_hi_u32_b32 v3, s5, v3
        0xBF810000u,
    };
    const std::vector<uint32_t> native_structured_saved_mbcnt_b64 = recompile_compute(
        compute_structured_saved_mbcnt_b64,
        std::size(compute_structured_saved_mbcnt_b64), nullptr,
        wave64_dispatch_config);
    CHECK(!native_structured_saved_mbcnt_b64.empty(),
          "structured compute routes saved-mask MBCNT through the native common phase");
    const std::vector<uint32_t> portable_structured_saved_mbcnt_b64 = recompile_compute(
        compute_structured_saved_mbcnt_b64,
        std::size(compute_structured_saved_mbcnt_b64), nullptr,
        portable_wave64_dispatch_config);
    CHECK(!portable_structured_saved_mbcnt_b64.empty(),
          "structured compute routes saved-mask MBCNT through the portable common phase");
    // GTA V's exec_cs_413d88400 reaches the exact `beea147e` packet at PC336 after
    // structured divergent loops.  Their EXEC restores are represented separately from the
    // architectural EXEC mask, so loop-carried scalar placeholders for s126/s127 must not make
    // the packet look like an ambiguous ordinary SGPR-pair read.
    const uint32_t compute_structured_exec_ff1_b64[] = {
        0x7E020283u,               //  0: v_mov_b32 v1, 3        (outer bound)
        0x7E080284u,               //  1: v_mov_b32 v4, 4        (inner bound)
        0xBE800380u,               //  2: s_mov_b32 s0, 0        (outer counter)
        0x7E040280u,               //  3: v_mov_b32 v2, 0
        0x7E060280u,               //  4: v_mov_b32 v3, 0
        0xBE82047Eu,               //  5: s_mov_b64 s[2:3], exec (outer save)
        0x7DA20200u,               //  6: OUTER_HDR: v_cmpx_lt_u32 s0, v1
        0xBF88000Du,               //  7: s_cbranch_execz +13 -> 21
        0xBE810380u,               //  8: s_mov_b32 s1, 0
        0xBE84047Eu,               //  9: s_mov_b64 s[4:5], exec (inner save)
        0x7DA20801u,               // 10: INNER_HDR: v_cmpx_lt_u32 s1, v4
        0xBF880004u,               // 11: s_cbranch_execz +4 -> 16
        0x060606FFu, 0x3D800000u,  // 12: v_add_f32 v3, 0.0625, v3
        0x81018101u,               // 14: s_add_i32 s1, s1, 1
        0xBF82FFFAu,               // 15: s_branch -6 -> 10
        0xBEFE0404u,               // 16: s_mov_b64 exec, s[4:5]
        0x060404FFu, 0x3E800000u,  // 17: v_add_f32 v2, 0.25, v2
        0x81008100u,               // 19: s_add_i32 s0, s0, 1
        0xBF82FFF1u,               // 20: s_branch -15 -> 6
        0xBEFE0402u,               // 21: s_mov_b64 exec, s[2:3]
        0xBEEA147Eu,               // 22: GTA PC336: s_ff1_i32_b64 vcc_lo, exec
        0xBE86107Eu,               // 23: GTA PC337: s_bcnt1_i32_b64 s6, exec
        0x8F04816Au,               // 24: s_lshl_b32 s4, vcc_lo, 1
        0x80040604u,               // 25: s_add_u32 s4, s4, s6 (consume both results)
        0x7E0C0204u,               // 26: v_mov_b32 v6, s4
        0xBF8A0000u,               // 27: s_barrier forces the structured route
        0xBF810000u,               // 28: s_endpgm
    };
    CHECK(!recompile_compute(compute_structured_exec_ff1_b64,
                             std::size(compute_structured_exec_ff1_b64), nullptr,
                             wave64_dispatch_config).empty(),
          "GTA's exact EXEC S_FF1/S_BCNT packets survive structured loop state");
    std::vector<uint32_t> compute_cfg_dispatch_dynamic_writelane = {
        0x7E780300u,              // v_mov_b32 v60, v0
        0xBEEB0389u,              // s_mov_b32 vcc_hi, 9 (scalar data)
        0xBEAD0385u,              // s_mov_b32 s45, 5
        0xD761003Cu, 0x00005A6Bu, // v_writelane_b32 v60, vcc_hi, s45 (Astro PC1039)
        0xD7600000u, 0x00005B3Cu, // v_readlane_b32 s0, v60, s45
        0x7E040200u,              // v_mov_b32 v2, s0 (consume selected scalar)
    };
    compute_cfg_dispatch_dynamic_writelane.insert(
        compute_cfg_dispatch_dynamic_writelane.end(), compute_cfg_dispatch,
        compute_cfg_dispatch + std::size(compute_cfg_dispatch));
    CHECK(!recompile_compute(compute_cfg_dispatch_dynamic_writelane.data(),
                             compute_cfg_dispatch_dynamic_writelane.size(), &dispatch_rt,
                             wave32_dispatch_config).empty(),
          "the complex dispatcher lowers Astro's exact dynamic v_writelane packet");
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
    const uint32_t plucky_scalar_bitset0[] = {
        0xBEEB03C1u, // s_mov_b32 vcc_hi, -1 (scalar-data lifetime)
        0xBEEB1B9Fu, // s_bitset0_b32 vcc_hi, 31 (exact Plucky #1554 packet)
        0xBF810000u, // s_endpgm
    };
    CHECK(!recompile_compute(plucky_scalar_bitset0, std::size(plucky_scalar_bitset0),
                             nullptr, ComputeShaderConfig{}).empty(),
          "Plucky scalar-data s_bitset0_b32 sequence recompiles");
    const uint32_t scalar_bitset0_b64[] = {
        0xBE8404C1u, // s_mov_b64 s[4:5], -1
        0xBE841C9Fu, // s_bitset0_b64 s[4:5], 31 (distinct unsupported opcode 0x1c)
        0xBF810000u, // s_endpgm
    };
    CHECK(recompile_compute(scalar_bitset0_b64, std::size(scalar_bitset0_b64),
                            nullptr, ComputeShaderConfig{}).empty(),
          "unmodeled B64 bitset remains fail-visible instead of using B32 semantics");
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
    // v_readlane_b32 encodes its destination SGPR in VDST, so its register number lives in the SCALAR
    // file. DQ VII's title grading kernel reloads a v28 spill slot long after an unrelated
    // `v_readlane_b32 s28, v28, 9`; counting that readlane as a VGPR write made the spill array look
    // clobbered and rejected the reload as `invalidated-vgpr-lane-slot`, skipping the dispatch and
    // collapsing the title composite to black (#1483). Same shape here: the colliding readlane
    // (s19 <- v19) precedes the legitimate v19[37] reload the exec restore consumes.
    const uint32_t compute_cfg_dispatch_readlane_sgpr_collision[] = {
        0xBE800380u, 0x7E000280u, 0x7E020300u,
        0xD7610013u, 0x00014A7Eu, 0xD7610013u, 0x0001507Fu,   // v19[37:40] = EXEC
        0xD7600013u, 0x00014B13u,                             // v_readlane_b32 s19, v19, 37
        0xD760000Eu, 0x00014B13u, 0xD760000Fu, 0x00015113u, 0xBEFE040Eu,
        0xE00C2000u, 0x80020400u, 0x7DB900F9u, 0x86050007u,
        0x7D020200u, 0xBF860006u, 0xBF0A8204u, 0x360000FDu, 0xBF840001u,
        0x81008100u, 0x81008100u, 0xBF82FFF4u,
        0xBF810000u,
    };
    CHECK(!recompile_valu(compute_cfg_dispatch_readlane_sgpr_collision,
                          sizeof(compute_cfg_dispatch_readlane_sgpr_collision) /
                              sizeof(compute_cfg_dispatch_readlane_sgpr_collision[0]),
                          0, 0, &dispatch_rt).empty(),
          "a v_readlane destination SGPR number does not invalidate the same-numbered spill VGPR");
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

    // House of the Dead 2's exact IMAGE_GET_LOD packet needs both a fragment execution model and its
    // captured T#/S# pair. The table-less coverage pass must classify it as context-dependent rather
    // than keep reporting opcode 0x60 as unsupported after the production fragment lowering accepts it.
    const uint32_t get_lod_2d[] = { 0xf1800108u, 0x01480809u, 0xbf810000u };
    const RecompileCoverage get_lod = recompile_coverage(
        get_lod_2d, std::size(get_lod_2d));
    CHECK(get_lod.total == 1 && get_lod.table_dependent == 1 &&
              get_lod.unsupported == 0 && get_lod.first_bad_fmt < 0,
          "2D IMAGE_GET_LOD reports recompilable-in-context, not unsupported");

    struct UnsupportedGetLod {
        const char* name;
        std::vector<uint32_t> instruction;
    };
    const std::vector<UnsupportedGetLod> unsupported_get_lod = {
        // NSA/A16/DLC/GLC/SLC/R128/TFE/LWE are exact llvm-mc gfx1030 encodings. UNRM comes directly
        // from Table 100 because LLVM exposes no IMAGE_GET_LOD spelling for it; D16 is rejected for
        // this opcode. Their raw fields and the reserved holes must not alias the ordinary contract.
        {"NSA",      {0xf180010au, 0x01480809u, 0x0000000au}},
        {"UNRM",     {0xf1801108u, 0x01480809u}},
        {"A16",      {0xf1800108u, 0x41480809u}},
        {"DLC",      {0xf1800188u, 0x01480809u}},
        {"GLC",      {0xf1802108u, 0x01480809u}},
        {"SLC",      {0xf3800108u, 0x01480809u}},
        {"R128",     {0xf1808108u, 0x01480809u}},
        {"TFE",      {0xf1810108u, 0x01480809u}},
        {"LWE",      {0xf1820108u, 0x01480809u}},
        {"D16-raw",  {0xf1800108u, 0x81480809u}},
        {"reserved-w0-b6",  {0xf1800148u, 0x01480809u}},
        {"reserved-w0-b14", {0xf1804108u, 0x01480809u}},
        {"reserved-w1-b26", {0xf1800108u, 0x05480809u}},
        {"reserved-w1-b27", {0xf1800108u, 0x09480809u}},
        {"reserved-w1-b28", {0xf1800108u, 0x11480809u}},
        {"reserved-w1-b29", {0xf1800108u, 0x21480809u}},
    };
    for (const auto& form : unsupported_get_lod) {
        std::vector<uint32_t> code = form.instruction;
        code.push_back(0xbf810000u);
        const RecompileCoverage classified = recompile_coverage(code.data(), code.size());
        char message[160];
        std::snprintf(message, sizeof(message),
                      "IMAGE_GET_LOD %s remains unsupported in table-less coverage", form.name);
        CHECK(classified.total == 1 && classified.table_dependent == 0 &&
                  classified.unsupported == 1,
              message);
    }

    // Asterix's resolve PS mixes the ordinary and exact one-extra-word NSA 2D_MSAA IMAGE_LOAD
    // encodings. Coverage is table-less, but it can still distinguish the address shapes the real
    // resource-aware emitter accepts from superficially similar, deliberately unsupported packets.
    const uint32_t msaa_load_consecutive[] = {
        0xf0000130u, 0x00000305u, 0xbf810000u,
    };
    const uint32_t msaa_load_nsa[] = {
        0xf0000132u, 0x00000205u, 0x00000206u, 0xbf810000u,
    };
    const RecompileCoverage msaa_consecutive = recompile_coverage(
        msaa_load_consecutive, std::size(msaa_load_consecutive));
    const RecompileCoverage msaa_nsa = recompile_coverage(
        msaa_load_nsa, std::size(msaa_load_nsa));
    CHECK(msaa_consecutive.total == 1 && msaa_consecutive.table_dependent == 1 &&
              msaa_consecutive.unsupported == 0 && msaa_nsa.total == 1 &&
              msaa_nsa.table_dependent == 1 && msaa_nsa.unsupported == 0,
          "exact consecutive and NSA 2D_MSAA loads report recompilable-in-context");

    uint32_t msaa_load_nsa_unused[std::size(msaa_load_nsa)];
    std::copy(std::begin(msaa_load_nsa), std::end(msaa_load_nsa),
              msaa_load_nsa_unused);
    msaa_load_nsa_unused[2] |= 0x00010000u;
    const uint32_t msaa_array_load[] = {
        0xf0000138u, 0x00000305u, 0xbf810000u, // DIM=2D_MSAA_ARRAY remains unsupported
    };
    const RecompileCoverage msaa_unused = recompile_coverage(
        msaa_load_nsa_unused, std::size(msaa_load_nsa_unused));
    const RecompileCoverage msaa_array = recompile_coverage(
        msaa_array_load, std::size(msaa_array_load));
    CHECK(msaa_unused.total == 1 && msaa_unused.table_dependent == 0 &&
              msaa_unused.unsupported == 1 && msaa_array.total == 1 &&
              msaa_array.table_dependent == 0 && msaa_array.unsupported == 1,
          "unused NSA address bytes and 2D_MSAA_ARRAY remain honestly unsupported");

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
