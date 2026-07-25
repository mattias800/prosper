// test_recompiled_fragment — render a triangle whose PIXEL SHADER is recompiled from RDNA2.
// The vertex shader is the placeholder (positions), but the fragment shader is real RDNA2 (assembled
// by llvm-mc) recompiled to SPIR-V by recompile_fragment: it exports green via EXP MRT0. We render
// and assert the triangle is GREEN (not the placeholder's red), proving RDNA2->SPIR-V works for an
// actual graphics-stage shader wired into a real pipeline.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "render_runner.h"
#include "spirv_triangle.h"     // kTriVertSpv: placeholder vertex shader (positions)
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_recompiled_fragment ==\n");
    const uint32_t W = 64, H = 64;

    // RDNA2 green pixel shader (llvm-mc gfx1030): v0=0(r) v1=1.0(g) v2=0(b) v3=1.0(a); exp mrt0.
    //   v_mov_b32 v0,0 | v_mov_b32 v1,1.0 | v_mov_b32 v2,0 | v_mov_b32 v3,1.0 | exp mrt0 ... | s_endpgm
    const uint32_t ps[] = { 0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };
    std::vector<uint32_t> frag = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
    CHECK(!frag.empty() && frag[0] == 0x07230203u, "recompiled RDNA2 pixel shader -> SPIR-V module");
    if (frag.empty()) { printf("== FAIL ==\n"); return 1; }

    std::vector<uint32_t> vert(kTriVertSpv, kTriVertSpv + sizeof(kTriVertSpv)/sizeof(kTriVertSpv[0]));
    std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H);
    CHECK(px.size() == (size_t)W * H * 4, "rendered with the recompiled fragment shader (pipeline accepted it)");
    if (px.size() != (size_t)W * H * 4) { printf("== FAIL: render failed ==\n"); return 1; }

    auto at = [&](uint32_t x, uint32_t y) { return &px[((size_t)y * W + x) * 4]; };
    const uint8_t* c = at(W/2, H/2);   // triangle covers the center
    const uint8_t* k = at(0, 0);       // corner = clear
    printf("  center=(%u,%u,%u,%u) corner=(%u,%u,%u,%u)\n", c[0],c[1],c[2],c[3], k[0],k[1],k[2],k[3]);
    CHECK(c[1] > 0x80 && c[0] < 0x40 && c[2] < 0x40, "center pixel is GREEN (from the recompiled shader)");
    CHECK(k[2] > 0x80 && k[0] < 0x40 && k[1] < 0x40, "corner pixel is the BLUE clear");

    // Sonic Origins' large UI pixel shader writes a four-register matrix through M0-indexed
    // v_movreld_b32. Exercise the ISA rule VGPR[VDST+M0]=SRC0 with M0=1, so v1 becomes green.
    const uint32_t indexed_dest_ps[] = {
        0x7E000280u, 0x7E020280u, 0x7E0402F2u, 0x7E0602F2u,
        0xBEFC0381u,              // s_mov_b32 m0, 1
        0x7E008502u,              // v_movreld_b32 v0, v2 -> v1 = 1.0
        0xF800180Fu, 0x03000100u, 0xBF810000u,
    };
    std::vector<uint32_t> indexed_frag = recompile_fragment(
        indexed_dest_ps, std::size(indexed_dest_ps));
    std::vector<uint8_t> indexed_px = prosper::test::render_triangle_rgba(
        vert, indexed_frag, W, H);
    const uint8_t* indexed_center = indexed_px.size() == static_cast<size_t>(W) * H * 4
        ? &indexed_px[((static_cast<size_t>(H) / 2) * W + W / 2) * 4] : nullptr;
    CHECK(indexed_center && indexed_center[1] > 0x80 && indexed_center[0] < 0x40,
          "RDNA2 v_movreld_b32 writes the M0-indexed VGPR destination");

    // A second Sonic loop form has an unconditional back-edge but an interior EXECZ targeting the
    // loop exit. The interior branch must leave the loop directly: going through the back-edge would
    // restore EXEC and re-enter forever. This one-iteration fixture clears EXEC in the body, takes the
    // direct break, restores it after the loop, and exports green.
    const uint32_t direct_exec_break_ps[] = {
        0x7E000280u,              // v0 = 0
        0xBE80047Eu,              // s_mov_b64 s[0:1], exec
        0xBEFE0400u,              // loop: s_mov_b64 exec, s[0:1]
        0x7DAC0080u,              // v_cmpx_ge_u32 0, v0 (active)
        0xBF880003u,              // s_cbranch_execz exit
        0xBEFE0480u,              // s_mov_b64 exec, 0
        0xBF880001u,              // interior s_cbranch_execz exit
        0xBF82FFFAu,              // s_branch loop
        0xBEFE0400u,              // exit: s_mov_b64 exec, s[0:1]
        0x7E0202F2u, 0x7E040280u, 0x7E0602F2u,
        0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    std::vector<uint32_t> direct_break_frag = recompile_fragment(
        direct_exec_break_ps, std::size(direct_exec_break_ps));
    std::vector<uint8_t> direct_break_px = prosper::test::render_triangle_rgba(
        vert, direct_break_frag, W, H);
    const uint8_t* direct_break_center =
        direct_break_px.size() == static_cast<size_t>(W) * H * 4
            ? &direct_break_px[((static_cast<size_t>(H) / 2) * W + W / 2) * 4] : nullptr;
    CHECK(direct_break_center && direct_break_center[1] > 0x80 && direct_break_center[0] < 0x40,
          "unconditional-backedge EXECZ break exits directly and preserves loop merge state");

    // Fragment MBCNT is a real cross-lane operation. The lowering uses a native subgroup exclusive
    // sum and marks the module as requiring an exact 64-lane subgroup. RADV can
    // enforce that contract; llvmpipe is fixed at 8 and must reject the draw rather than execute
    // silently wrong wave semantics.
    const uint32_t wave_ps[] = {
        0xD7650004u, 0x000100C1u, // v_mbcnt_lo_u32_b32 v4, -1, 0
        0xD7660004u, 0x000208C1u, // v_mbcnt_hi_u32_b32 v4, -1, v4
        0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u,
        0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    std::vector<uint32_t> wave_frag = recompile_fragment(wave_ps, std::size(wave_ps));
    CHECK(!wave_frag.empty(), "recompiled fragment MBCNT through native subgroup exclusive sum");
    CHECK(fragment_spirv_required_subgroup_size(wave_frag) == 64,
          "fragment MBCNT module advertises its exact wave64 pipeline requirement");
    const auto& wave_ctx = prosper::test::render_vk_ctx();
    const bool supports_fragment_wave64 = wave_ctx.subgroup_size_control &&
        wave_ctx.min_subgroup_size <= 64 && wave_ctx.max_subgroup_size >= 64 &&
        (wave_ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
        (wave_ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
        (wave_ctx.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
    const bool supports_fragment_gds =
        supports_fragment_wave64 && wave_ctx.fragment_stores_atomics;
    std::vector<uint8_t> wave_px = prosper::test::render_triangle_rgba(vert, wave_frag, W, H);
    const bool wave_rendered = wave_px.size() == static_cast<size_t>(W) * H * 4;
    const uint8_t* wave_center = wave_rendered
        ? &wave_px[((static_cast<size_t>(H) / 2) * W + W / 2) * 4] : nullptr;
    CHECK(supports_fragment_wave64
              ? wave_rendered && wave_center[1] > 0x80 && wave_center[2] < 0x40
              : wave_rendered && wave_center[2] > 0x80 && wave_center[1] < 0x40,
          supports_fragment_wave64
              ? "device enforced wave64 and executed fragment MBCNT"
              : "device without fragment wave64 skipped MBCNT draw fail-visible");

    // Astro's compaction sequence allocates popcount(EXEC) slots from a device-global GDS
    // counter, then adds the MBCNT prefix. Export the allocated index parity so both returned
    // values are observable, and independently verify that helper invocations consumed no slots.
    const uint32_t gds_ps[] = {
        0xD7660007u, 0x0001007Fu, // v_mbcnt_hi_u32_b32 v7, exec_hi, 0
        0xBEFC0380u,              // s_mov_b32 m0, 0
        0xD8FA0014u, 0x06000000u, // ds_append v6 offset:20 gds
        0xD7650000u, 0x00020E7Eu, // v_mbcnt_lo_u32_b32 v0, exec_lo, v7
        0x4A140106u,              // v_add_nc_u32 v10, v6, v0
        0x36001481u,              // v_and_b32 v0, 1, v10
        0x7E000D00u,              // v_cvt_f32_u32 v0, v0
        0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    std::vector<uint32_t> gds_frag = recompile_fragment(gds_ps, std::size(gds_ps));
    CHECK(!gds_frag.empty(), "recompiled Astro fragment MBCNT + GDS append compaction sequence");
    CHECK(fragment_spirv_uses_internal_gds(gds_frag),
          "fragment GDS module advertises its backend-owned persistent descriptor");
    CHECK(validate_spirv_descriptor_interface(
              gds_frag, nullptr, 1, SpirvShaderStage::Fragment, false).ok(),
          "descriptor validation accepts renderer-owned GDS without a guest table entry");
    prosper::test::reset_internal_gds_for_test();
    std::vector<uint8_t> gds_px = prosper::test::render_triangle_rgba(vert, gds_frag, W, H);
    if (supports_fragment_gds && gds_px.size() == static_cast<size_t>(W) * H * 4) {
        uint32_t covered = 0, red = 0;
        for (size_t i = 0; i < gds_px.size(); i += 4) {
            if (gds_px[i + 2] < 0x40) {
                ++covered;
                if (gds_px[i] > 0x80) ++red;
            }
        }
        CHECK(prosper::test::read_internal_gds_for_test(20) == covered,
              "GDS counter advances once per covered non-helper fragment across all waves");
        printf("  GDS covered=%u red=%u counter=%u\n", covered, red,
               prosper::test::read_internal_gds_for_test(20));
        CHECK(red == covered / 2,
              "export observes unique append-base + MBCNT-prefix allocation parity");
    } else {
        CHECK(!supports_fragment_gds && !gds_px.empty(),
              "device without fragment wave64/GDS support skips compaction draw fail-visible");
    }

    // Implicit-LOD sampling forces whole-quad execution on RADV. Primitive-edge helper lanes must
    // participate in subgroup arithmetic without becoming the atomic leader or consuming slots.
    const uint32_t gds_wqm_ps[] = {
        0x7E0002FFu, 0x3E800000u, 0x7E0202FFu, 0x3E800000u,
        0xF0800F08u, 0x00820000u, // image_sample v[0:3], v[0:1], s[8:15], s[16:19]
        0xD7660007u, 0x0001007Fu, 0xBEFC0380u,
        0xD8FA0014u, 0x06000000u,
        0xD7650000u, 0x00020E7Eu, 0x4A140106u, 0x36001481u, 0x7E000D00u,
        // Keep sampled v3 live as export alpha so RADV cannot DCE the implicit-LOD sample/WQM.
        0x7E020280u, 0x7E040280u,
        0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    ShaderResourceTable gds_wqm_rt;
    ShaderResource gds_wqm_texture{};
    gds_wqm_texture.cls = ResourceClass::Texture;
    gds_wqm_texture.binding = 4; gds_wqm_texture.img_dim = 1;
    gds_wqm_texture.width = 2; gds_wqm_texture.height = 2;
    gds_wqm_texture.sgpr_base = 8;
    gds_wqm_rt.resources.push_back(gds_wqm_texture);
    std::vector<uint32_t> gds_wqm_frag = recompile_fragment(
        gds_wqm_ps, std::size(gds_wqm_ps), &gds_wqm_rt);
    const uint8_t wqm_texels[16] = {
        255,255,255,255, 255,255,255,255,
        255,255,255,255, 255,255,255,255,
    };
    prosper::test::TexDesc gds_wqm_tex{4, 2, 2, wqm_texels};
    prosper::test::reset_internal_gds_for_test();
    std::vector<uint8_t> gds_wqm_px = prosper::test::render_triangle_rgba(
        vert, gds_wqm_frag, W, H, nullptr, nullptr, nullptr, &gds_wqm_tex);
    if (supports_fragment_gds && gds_wqm_px.size() == static_cast<size_t>(W) * H * 4) {
        uint32_t covered = 0, red = 0, sampled_alpha = 0;
        for (size_t i = 0; i < gds_wqm_px.size(); i += 4) {
            if (gds_wqm_px[i + 2] < 0x40) {
                ++covered;
                if (gds_wqm_px[i] > 0x80) ++red;
                if (gds_wqm_px[i + 3] > 0x80) ++sampled_alpha;
            }
        }
        CHECK(prosper::test::read_internal_gds_for_test(20) == covered,
              "WQM helper lanes neither lead the GDS atomic nor consume counter slots");
        CHECK(red == covered / 2,
              "WQM append-base + MBCNT-prefix allocation remains unique and contiguous");
        CHECK(sampled_alpha == covered,
              "WQM regression keeps implicit-LOD sampled alpha live through export");
    } else {
        CHECK(!supports_fragment_gds && !gds_wqm_px.empty(),
              "unsupported device skips WQM GDS compaction draw fail-visible");
    }

    if (supports_fragment_gds) {
        std::vector<uint32_t> gds_consume_ps(gds_ps, gds_ps + std::size(gds_ps));
        gds_consume_ps[3] = 0xD8F60014u; // ds_consume v6 offset:20 gds
        std::vector<uint32_t> gds_consume_frag = recompile_fragment(
            gds_consume_ps.data(), gds_consume_ps.size());
        std::vector<uint8_t> gds_consume_px = prosper::test::render_triangle_rgba(
            vert, gds_consume_frag, W, H);
        CHECK(gds_consume_px.size() == static_cast<size_t>(W) * H * 4 &&
                  prosper::test::read_internal_gds_for_test(20) == 0,
              "GDS consume subtracts one covered-fragment allocation across all waves");
    }

    // Astro Bot's early foreground pass exports only R/G (EN=0x3). AMD's EXP contract preserves
    // disabled destination components; the live executor therefore intersects EN with the Vulkan
    // pipeline write mask. Prove both the exact rejected shader now recompiles and a partial draw
    // changes R/G while retaining the BLUE clear's B/A channels.
    const uint32_t astro_partial_ps[] = {
        0x7E000280u, 0xF8001803u, 0x00000000u, 0xBF810000u,
    };
    CHECK(fragment_color_export_mask(astro_partial_ps, std::size(astro_partial_ps)) == 0x3u,
          "#825: decoded Astro Bot's MRT0 R/G export-enable mask");
    CHECK(!recompile_fragment(astro_partial_ps, std::size(astro_partial_ps)).empty(),
          "#825: recompiled Astro Bot's partial-EN fragment shader");

    const uint32_t astro_ngg_vs[] = {
        0xBFA00001u, 0x93EAFF03u, 0x00080008u, 0x876BFF03u, 0x000000FFu,
        0x8F6A8C6Au, 0x887C6A6Bu, 0xBF800000u, 0xBF900009u, 0x906A8803u,
        0x81EA6A80u, 0x90FE6AC1u, 0xF8000941u, 0x00000000u, 0x81EA0380u,
        0xBF8CFF0Fu, 0x90FE6AC1u, 0x34040A81u, 0x36060AC2u, 0x7E000280u,
        0x7E0202F2u, 0x36040482u, 0x4A0606C1u, 0x4A0404C1u, 0x7E060B03u,
        0x7E040B02u, 0xF80000D4u, 0x00080000u, 0xF80008CFu, 0x01000302u,
        0xBF810000u,
    };
    std::vector<uint32_t> astro_vert = recompile_vertex(astro_ngg_vs, std::size(astro_ngg_vs));
    CHECK(!astro_vert.empty(),
          "#825: recompiled Astro Bot NGG shader with ancillary POS1 before mandatory POS0");

    const uint32_t preserve_ps[] = {
        0x7E0002F2u, 0x7E020280u, // v0=1.0 (R), v1=0.0 (G)
        0xF8001803u, 0x00000100u, 0xBF810000u,
    };
    std::vector<uint32_t> preserve_frag = recompile_fragment(preserve_ps, std::size(preserve_ps));
    ResolvedPipelineState preserve_state;
    preserve_state.topology = 3; // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    preserve_state.color_write_mask = fragment_color_export_mask(preserve_ps, std::size(preserve_ps));
    std::vector<uint8_t> preserve_px = prosper::test::render_triangle_rgba(
        vert, preserve_frag, W, H, &preserve_state);
    CHECK(preserve_px.size() == (size_t)W * H * 4,
          "#825: rendered partial-EN fragment shader with its attachment mask");
    if (preserve_px.size() == (size_t)W * H * 4) {
        const uint8_t* pc = &preserve_px[((size_t)(H/2) * W + W/2) * 4];
        printf("  partial-EN center=(%u,%u,%u,%u)\n", pc[0], pc[1], pc[2], pc[3]);
        CHECK(pc[0] > 0x80 && pc[1] < 0x40 && pc[2] > 0x80 && pc[3] > 0x80,
              "#825: partial R/G export preserves destination B/A components");
    }

    // DOLL's volume-sampling PS uses ENA=ADDR=0x702: perspective-center occupies v0:v1 and
    // POS_X/Y/Z_FLOAT occupy v2:v4. Export v2 as red; a real FragCoord.x saturates the UNORM target
    // while the old undefined-register fallback exported zero and left the triangle black.
    const uint32_t position_ps[] = {
        0x7E060280u, 0x7E080280u, 0x7E0A02F2u, // v3=0, v4=0, v5=1
        0xF800180Fu, 0x05040302u, 0xBF810000u, // exp mrt0 v2,v3,v4,v5; s_endpgm
    };
    PixelSystemInputMapping doll_inputs{0x00000702u, 0x00000702u};
    std::vector<uint32_t> position_frag = recompile_fragment(
        position_ps, std::size(position_ps), nullptr, &doll_inputs);
    CHECK(!position_frag.empty(), "recompiled DOLL's packed perspective-center + position inputs");
    if (!position_frag.empty()) {
        std::vector<uint8_t> position_px = prosper::test::render_triangle_rgba(
            vert, position_frag, W, H);
        CHECK(position_px.size() == (size_t)W * H * 4,
              "rendered a fragment shader consuming the hardware position VGPR");
        if (position_px.size() == (size_t)W * H * 4) {
            const uint8_t* pc = &position_px[((size_t)(H/2) * W + W/2) * 4];
            printf("  system-position center=(%u,%u,%u,%u)\n", pc[0],pc[1],pc[2],pc[3]);
            CHECK(pc[0] > 0x80 && pc[1] < 0x40 && pc[2] < 0x40,
                  "DOLL layout: v2 receives FragCoord.x and renders RED");
        }
    }

    // VCC kill-mask early-out (#273 — DOLL's alpha-cull PS): `v_cmp; s_andn2_b64 vcc, exec, vcc;
    // s_cbranch_scc0 <null-export>; s_mov_b64 exec, vcc; export`. The kill mask lives in VCC itself
    // (not a saved SGPR pair); mask_test_branches must recognize it so the branch linearizes and the
    // narrowed-EXEC export lowers to a per-invocation discard. Two variants (llvm-mc gfx1010):
    //  survive: v0=0.5 -> cmp(0.5<v0)=false -> survivors=all -> WHITE triangle;
    //  killed:  v0=1.0 -> cmp=true -> survivors=none -> every fragment discarded -> clear stays.
    auto killmask_ps = [](uint32_t v0mov) {
        return std::vector<uint32_t>{ v0mov, 0x7c0200f0u, 0x8aea6a7eu, 0xbf840005u, 0xbefe046au,
                                      0x7e0202f2u, 0xf800180fu, 0x01010101u, 0xbf810000u,
                                      0xbefe0480u, 0xf8001800u, 0x00000000u, 0xbf810000u };
    };
    std::vector<uint32_t> psSurvive = killmask_ps(0x7e0002f0u);   // v_mov_b32 v0, 0.5
    std::vector<uint32_t> psKilled  = killmask_ps(0x7e0002f2u);   // v_mov_b32 v0, 1.0
    std::vector<uint32_t> fragS = recompile_fragment(psSurvive.data(), psSurvive.size());
    std::vector<uint32_t> fragK = recompile_fragment(psKilled.data(),  psKilled.size());
    CHECK(!fragS.empty() && !fragK.empty(), "recompiled VCC kill-mask PS (both variants) -> SPIR-V");
    if (!fragS.empty() && !fragK.empty()) {
        std::vector<uint8_t> pxS = prosper::test::render_triangle_rgba(vert, fragS, W, H);
        std::vector<uint8_t> pxK = prosper::test::render_triangle_rgba(vert, fragK, W, H);
        CHECK(pxS.size() == (size_t)W*H*4 && pxK.size() == (size_t)W*H*4, "rendered both kill-mask variants");
        if (pxS.size() == (size_t)W*H*4 && pxK.size() == (size_t)W*H*4) {
            const uint8_t* cs = &pxS[((size_t)(H/2) * W + W/2) * 4];
            const uint8_t* ck = &pxK[((size_t)(H/2) * W + W/2) * 4];
            printf("  survive center=(%u,%u,%u,%u) killed center=(%u,%u,%u,%u)\n",
                   cs[0],cs[1],cs[2],cs[3], ck[0],ck[1],ck[2],ck[3]);
            CHECK(cs[0] > 0x80 && cs[1] > 0x80 && cs[2] > 0x80, "survive variant: center WHITE (lanes pass the mask)");
            CHECK(ck[2] > 0x80 && ck[0] < 0x40 && ck[1] < 0x40, "killed variant: center stays BLUE clear (OpKill discarded)");
        }
    }

    // Astro Bot's depth/stencil prepass PS exports only NULL. The VCC mask becomes EXEC immediately
    // before that export, so inactive fragments must still lower to OpKill even though there is no MRT
    // output. These are its exact gfx1030 control-flow/export tail (PPSA21564, 0x500540000); the
    // preceding resource sample is covered separately by the texture execution tests.
    const uint32_t null_only_ps[] = {
        0x7d840080u, 0x8aea6a7eu, 0xbf840004u, 0xbefe046au,
        0xf8001890u, 0x00000000u, 0xbf810000u,
    };
    std::vector<uint32_t> null_frag = recompile_fragment(null_only_ps, std::size(null_only_ps));
    CHECK(!null_frag.empty() && null_frag[0] == 0x07230203u,
          "#825: recompiled Astro Bot's NULL-export depth/stencil fragment shader");
    if (!null_frag.empty()) {
        std::vector<uint8_t> null_px = prosper::test::render_triangle_rgba(vert, null_frag, W, H);
        CHECK(null_px.size() == (size_t)W * H * 4,
              "#825: Vulkan accepts a fragment module with no color outputs");
        if (null_px.size() == (size_t)W * H * 4) {
            const uint8_t* nc = &null_px[((size_t)(H/2) * W + W/2) * 4];
            CHECK(nc[2] > 0x80 && nc[0] < 0x40 && nc[1] < 0x40,
                  "#825: NULL export leaves the BLUE color attachment unchanged");
        }
    }

    // Sonic Origins' early prepass uses a real MRTZ export with no color or NULL export. The
    // fragment shell already lowers target 8 to FragDepth, so the early color-output gate must
    // retain this valid depth-only module instead of rejecting it before instruction emission.
    const uint32_t depth_only_ps[] = {
        0x7e0002ffu, 0x3f000000u,       // v0 = 0.5f
        0xf8001881u, 0x00000000u,       // exp mrtz v0 (Z enabled)
        0xbf810000u,
    };
    std::vector<uint32_t> depth_frag = recompile_fragment(
        depth_only_ps, std::size(depth_only_ps));
    CHECK(!depth_frag.empty() && depth_frag[0] == 0x07230203u,
          "recompiled Sonic Origins' MRTZ-only fragment shader");

    // Sonic's bloom combine detects NaNs with `v_cmp_u_f32 vcc, v6, v6`. Opcode 0x08 is the
    // unordered predicate (true when either source is NaN), distinct from the six ordered compares.
    const uint32_t unordered_compare_ps[] = {
        0x7e0c0280u,                   // v6 = 0
        0x7c100d06u,                   // v_cmp_u_f32 vcc, v6, v6
        0xf800000fu, 0x06060606u,      // exp mrt0 v6,v6,v6,v6
        0xbf810000u,
    };
    CHECK(!recompile_fragment(unordered_compare_ps,
                              std::size(unordered_compare_ps)).empty(),
          "recompiled Sonic Origins' unordered f32 comparison");

    // DIVERGENT execz region (#273 — DOLL's FXAA PS shape): v_cmpx narrows EXEC, s_cbranch_execz
    // skips a block containing a SCALAR write read after the merge (so it is NOT safe-linearizable
    // and must go through the structured exec-if). Then-arm sets v1 = s5 = 1.0; the export runs
    // under the narrowed EXEC (surviving lanes only).
    //  taken:   v0=1.0 -> 0.5<1.0 -> all lanes survive -> WHITE;
    //  skipped: v0=0.5 -> 0.5<0.5 false -> exec 0 -> every fragment discarded -> clear stays.
    auto execz_ps = [](uint32_t v0mov) {
        return std::vector<uint32_t>{ 0x7e020280u, v0mov, 0x7c2200f0u, 0xbf880002u,
                                      0xbe8503f2u, 0x7e020205u,            // block: s5=1.0; v1=s5
                                      0x7e040205u,                          // post-merge s5 read (v2=s5)
                                      0xf800180fu, 0x01010101u, 0xbf810000u };
    };
    std::vector<uint32_t> psTaken   = execz_ps(0x7e0002f2u);   // v_mov_b32 v0, 1.0
    std::vector<uint32_t> psSkipped = execz_ps(0x7e0002f0u);   // v_mov_b32 v0, 0.5
    std::vector<uint32_t> fragT = recompile_fragment(psTaken.data(),   psTaken.size());
    std::vector<uint32_t> fragX = recompile_fragment(psSkipped.data(), psSkipped.size());
    CHECK(!fragT.empty() && !fragX.empty(), "recompiled divergent-execz PS (both variants) -> SPIR-V");
    if (!fragT.empty() && !fragX.empty()) {
        std::vector<uint8_t> pxT = prosper::test::render_triangle_rgba(vert, fragT, W, H);
        std::vector<uint8_t> pxX = prosper::test::render_triangle_rgba(vert, fragX, W, H);
        CHECK(pxT.size() == (size_t)W*H*4 && pxX.size() == (size_t)W*H*4, "rendered both execz variants");
        if (pxT.size() == (size_t)W*H*4 && pxX.size() == (size_t)W*H*4) {
            const uint8_t* ct = &pxT[((size_t)(H/2) * W + W/2) * 4];
            const uint8_t* cx = &pxX[((size_t)(H/2) * W + W/2) * 4];
            printf("  execz taken center=(%u,%u,%u,%u) skipped center=(%u,%u,%u,%u)\n",
                   ct[0],ct[1],ct[2],ct[3], cx[0],cx[1],cx[2],cx[3]);
            CHECK(ct[0] > 0x80 && ct[1] > 0x80 && ct[2] > 0x80, "execz taken: center WHITE (block ran, s5 -> v1)");
            CHECK(cx[2] > 0x80 && cx[0] < 0x40 && cx[1] < 0x40, "execz skipped: center stays BLUE clear (all lanes inactive)");
        }
    }

    // DIVERGENT EXECZ-EXIT LOOP (#273 — DOLL's title post-process accumulation PS shape).
    //   header: v_cmpx_lt_u32 s0, v1 (per-lane trip bound); s_cbranch_execz EXIT;
    //   body:   a NESTED execz if (saveexec + v_cmpx s0<2 + restore) adding 0.25 to v2 (red/blue),
    //           an unconditional 0.25 add to v3 (green), s0++; s_branch header (backward).
    //   EXIT:   restore exec; export (v2, v3, v2, 1.0).
    // 4 iterations, inner if true for the first 2 => color (0.5, 1.0, 0.5, 1.0).
    // (Assembled by llvm-mc gfx1030 from labeled source; encodings are the assembler's own.)
    {
        const uint32_t ps[] = {
            0x7E020284u, 0xBE800380u, 0x7E040280u, 0x7E060280u, 0x7E080282u, 0x7E0A02F2u,
            0xBE82047Eu, 0x7DA20200u, 0xBF88000Au, 0xBE84047Eu, 0x7DA20800u, 0xBF880002u,
            0x060404FFu, 0x3E800000u, 0xBEFE0404u, 0x060606FFu, 0x3E800000u, 0x81008100u,
            0xBF82FFF4u, 0xBEFE0402u, 0xF800180Fu, 0x05020302u, 0xBF810000u,
        };
        std::vector<uint32_t> frg = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
        CHECK(!frg.empty(), "recompiled divergent execz-exit LOOP PS -> SPIR-V");
        if (!frg.empty()) {
            std::vector<uint8_t> px2 = prosper::test::render_triangle_rgba(vert, frg, W, H);
            CHECK(px2.size() == (size_t)W*H*4, "rendered the divergent-loop PS");
            if (px2.size() == (size_t)W*H*4) {
                const uint8_t* cc = &px2[((size_t)(H/2) * W + W/2) * 4];
                printf("  divloop center=(%u,%u,%u,%u) expect ~(128,255,128,255)\n", cc[0],cc[1],cc[2],cc[3]);
                CHECK(cc[1] > 0xF0 && cc[0] > 0x70 && cc[0] < 0x90 && cc[2] > 0x70 && cc[2] < 0x90,
                      "divergent loop: 4 iterations, nested if 2 -> (0.5, 1.0, 0.5)");
            }
        }
    }

    // VCCZ-EXIT LOOP (#615 - Dead Cells' per-pixel light accumulation shape).
    // VCC is this fragment invocation's loop predicate: for (s0=0; s0<v1=4; ++s0) v0 += 0.25.
    // The hardware branch exits when the wave VCC mask is empty. Here the bound is a constant moved
    // into every lane, so VCC is provably uniform and the per-invocation branch is exact. The
    // unconditional back-edge and carried scalar/vector values lower through OpLoopMerge + OpPhi.
    {
        const uint32_t ps[] = {
            0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7E0602F2u,
            0x7D020200u, 0xBF860004u, 0x060000FFu, 0x3E800000u,
            0x81008100u, 0xBF82FFFAu, 0x7E020300u, 0x7E040300u,
            0xF800080Fu, 0x03020100u, 0xBF810000u,
        };
        std::vector<uint32_t> frg = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
        CHECK(!frg.empty(), "recompiled VCCZ-exit light-accumulation LOOP PS -> SPIR-V");
        if (!frg.empty()) {
            std::vector<uint8_t> px2 = prosper::test::render_triangle_rgba(vert, frg, W, H);
            CHECK(px2.size() == (size_t)W*H*4, "rendered the VCCZ-exit loop PS");
            if (px2.size() == (size_t)W*H*4) {
                const uint8_t* cc = &px2[((size_t)(H/2) * W + W/2) * 4];
                printf("  vccz loop center=(%u,%u,%u,%u) expect white\n", cc[0],cc[1],cc[2],cc[3]);
                CHECK(cc[0] > 0xF0 && cc[1] > 0xF0 && cc[2] > 0xF0,
                      "VCCZ loop: exactly 4 iterations of 0.25 -> 1.0 white");
            }
        }

        uint32_t varying_ps[sizeof(ps)/sizeof(ps[0])];
        std::copy(std::begin(ps), std::end(ps), varying_ps);
        varying_ps[2] = 0x7E020302u;  // v_mov_b32 v1,v2: a lane-varying/unproven loop bound
        CHECK(recompile_fragment(varying_ps, sizeof(varying_ps)/sizeof(varying_ps[0])).empty(),
              "a varying-VCC loop remains rejected (wave-empty branch is not lane-local)");

        uint32_t exec_mutating_ps[sizeof(ps)/sizeof(ps[0])];
        std::copy(std::begin(ps), std::end(ps), exec_mutating_ps);
        exec_mutating_ps[3] = 0xBEFE04C1u;  // s_mov_b64 exec,-1 between the bound definition and compare
        CHECK(recompile_fragment(exec_mutating_ps,
                                 sizeof(exec_mutating_ps)/sizeof(exec_mutating_ps[0])).empty(),
              "a VCC loop with intervening EXEC mutation remains rejected");
    }

    // EXECNZ-back-edge loop with a mid-body vccz BREAK (#273 — DOLL's scalar-indexed unroll shape):
    //   LOOP: s_cbranch_execz EXIT (header exit); s_cmp_lt_u32 s0,s1; s_cselect_b64 s[4:5],exec,0;
    //         vcc=exec=s[4:5]; s_cbranch_vccz EXIT (break); v2 += 0.25; s0++; s_cbranch_execnz LOOP.
    // 3 iterations add 0.25 => gray (0.75, 0.75, 0.75, 1.0).
    {
        const uint32_t ps[] = {
            0xBE82047Eu, 0xBE800380u, 0xBE810383u, 0x7E040280u, 0x7E0A02F2u, 0xBF880009u,
            0xBF0A0100u, 0x8584807Eu, 0xBEEA0404u, 0xBEFE0404u, 0xBF860004u, 0x060404FFu,
            0x3E800000u, 0x81008100u, 0xBF89FFF6u, 0xBEFE0402u, 0xF800180Fu, 0x05020202u,
            0xBF810000u,
        };
        std::vector<uint32_t> frg = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
        CHECK(!frg.empty(), "recompiled execnz-backedge loop + vccz break PS -> SPIR-V");
        if (!frg.empty()) {
            std::vector<uint8_t> px2 = prosper::test::render_triangle_rgba(vert, frg, W, H);
            CHECK(px2.size() == (size_t)W*H*4, "rendered the execnz-loop PS");
            if (px2.size() == (size_t)W*H*4) {
                const uint8_t* cc = &px2[((size_t)(H/2) * W + W/2) * 4];
                printf("  execnz loop center=(%u,%u,%u,%u) expect ~(191,191,191,255)\n", cc[0],cc[1],cc[2],cc[3]);
                CHECK(cc[0] > 0xA8 && cc[0] < 0xD8 && cc[1] > 0xA8 && cc[1] < 0xD8,
                      "execnz loop + break: exactly 3 iterations -> 0.75 gray");
            }
        }
    }

    // NESTED DIVERGENT EXECZ-EXIT LOOPS (#590 — DOLL's last post-process kernel shape, fragment
    // shell): an inner table loop entirely inside an outer row loop, both v_cmpx/execz-exit with
    // backward s_branch back-edges, with an exec save around the inner loop and a restore at its
    // exit. outer 3 x inner 4: v3 += 0.0625 twelve times = 0.75 (green), v2 += 0.25 three times in
    // the outer body tail = 0.75 (red/blue) -> gray (0.75, 0.75, 0.75, 1.0).
    {
        const uint32_t ps[] = {
            0x7E020283u,               //  0: v_mov_b32 v1, 3        (outer bound)
            0x7E080284u,               //  1: v_mov_b32 v4, 4        (inner bound)
            0xBE800380u,               //  2: s_mov_b32 s0, 0
            0x7E040280u,               //  3: v_mov_b32 v2, 0
            0x7E060280u,               //  4: v_mov_b32 v3, 0
            0x7E0A02F2u,               //  5: v_mov_b32 v5, 1.0
            0xBE82047Eu,               //  6: s_mov_b64 s[2:3], exec
            0x7DA20200u,               //  7: OUTER_HDR: v_cmpx_lt_u32 s0, v1
            0xBF88000Du,               //  8: s_cbranch_execz +13 -> 22 (OUTER_EXIT)
            0xBE810380u,               //  9: s_mov_b32 s1, 0
            0xBE84047Eu,               // 10: s_mov_b64 s[4:5], exec
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
            0xF800180Fu, 0x05020302u,  // 23: exp mrt0 v2, v3, v2, v5 done vm
            0xBF810000u,               // 25: s_endpgm
        };
        std::vector<uint32_t> frg = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
        CHECK(!frg.empty(), "#590: recompiled NESTED execz-exit loops PS -> SPIR-V");
        if (!frg.empty()) {
            std::vector<uint8_t> px2 = prosper::test::render_triangle_rgba(vert, frg, W, H);
            CHECK(px2.size() == (size_t)W*H*4, "rendered the nested-loop PS");
            if (px2.size() == (size_t)W*H*4) {
                const uint8_t* cc = &px2[((size_t)(H/2) * W + W/2) * 4];
                printf("  nested loops center=(%u,%u,%u,%u) expect ~(191,191,191,255)\n",
                       cc[0],cc[1],cc[2],cc[3]);
                CHECK(cc[0] > 0xA8 && cc[0] < 0xD8 && cc[1] > 0xA8 && cc[1] < 0xD8 &&
                      cc[2] > 0xA8 && cc[2] < 0xD8,
                      "#590: nested loops 3x4: inner 12 x 0.0625 and outer 3 x 0.25 both = 0.75");
            }
        }

        // Partially-overlapping loops (a back-edge into another loop's body without proper nesting)
        // remain rejected in the fragment shell (no dispatcher fallback exists there).
        const uint32_t overlap_ps[] = {
            0xBE800380u, 0x7E020284u,
            0x7DA20200u,               //  2: A_HDR: v_cmpx_lt_u32 s0, v1
            0xBF880006u,               //  3: s_cbranch_execz +6 -> 10
            0x7DA20200u,               //  4: B_HDR
            0xBF880006u,               //  5: s_cbranch_execz +6 -> 12
            0x81008100u,               //  6: s0++
            0xBF82FFFAu,               //  7: s_branch -6 -> 2 (A back-edge; A=[2,7])
            0x81008100u,               //  8: s0++
            0xBF82FFFAu,               //  9: s_branch -6 -> 4 (B back-edge; B=[4,9] overlaps A)
            0xF800180Fu, 0x05020302u,  // 10: export
            0xBF810000u,               // 12: s_endpgm
        };
        CHECK(recompile_fragment(overlap_ps, sizeof(overlap_ps)/sizeof(overlap_ps[0])).empty(),
              "#590: partially-overlapping loops remain REJECTED (fragment)");
    }

    // SCALAR-SPILL lane slots (#273): v_writelane_b32 packs two scalars into v10's lanes 3/7 and
    // v_readlane_b32 restores them; export uses the round-tripped values -> (0.25, 1.0, 0.25, 1.0).
    {
        const uint32_t ps[] = {
            0xBE8503FFu, 0x3E800000u, 0xBE8603F2u, 0xD761000Au, 0x00010605u, 0xD761000Au,
            0x00010E06u, 0xD7600007u, 0x0001070Au, 0xD7600008u, 0x00010F0Au, 0x7E000207u,
            0x7E020208u, 0xF800180Fu, 0x01000100u, 0xBF810000u,
        };
        std::vector<uint32_t> frg = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
        CHECK(!frg.empty(), "recompiled writelane/readlane scalar-spill PS -> SPIR-V");
        if (!frg.empty()) {
            std::vector<uint8_t> px2 = prosper::test::render_triangle_rgba(vert, frg, W, H);
            CHECK(px2.size() == (size_t)W*H*4, "rendered the lane-slot PS");
            if (px2.size() == (size_t)W*H*4) {
                const uint8_t* cc = &px2[((size_t)(H/2) * W + W/2) * 4];
                printf("  laneslot center=(%u,%u,%u,%u) expect ~(64,255,64,255)\n", cc[0],cc[1],cc[2],cc[3]);
                CHECK(cc[1] > 0xF0 && cc[0] > 0x30 && cc[0] < 0x50,
                      "writelane/readlane: slots round-trip (0.25, 1.0)");
            }
        }
    }

    // A compiler may recycle the spill-array VGPR for ordinary per-pixel data. The normal write
    // ends the lane-slot lifetime: an EXP must read the new value, while a later readlane from an
    // old slot is invalid. Blasphemous 2's missing world composite reuses v11 this way (#652).
    {
        const uint32_t recycled[] = {
            0xBE8503FFu, 0x3E800000u,              // s5 = 0.25
            0xD761000Au, 0x00010605u,              // v_writelane_b32 v10, s5, 3
            0x7E1402F2u,                           // v_mov_b32 v10, 1.0
            0xF800180Fu, 0x0A0A0A0Au,              // exp mrt0 v10, v10, v10, v10
            0xBF810000u,
        };
        CHECK(!recompile_fragment(recycled, sizeof(recycled)/sizeof(recycled[0])).empty(),
              "#652: ordinary VGPR write ends a scalar-spill lane-slot lifetime");

        const uint32_t stale_readlane[] = {
            0xBE8503FFu, 0x3E800000u,              // s5 = 0.25
            0xD761000Au, 0x00010605u,              // v_writelane_b32 v10, s5, 3
            0x7E1402F2u,                           // v_mov_b32 v10, 1.0 (kills spill slots)
            0xD7600007u, 0x0001070Au,              // v_readlane_b32 s7, v10, 3
            0x7E000207u, 0xF800180Fu, 0x00000000u, 0xBF810000u,
        };
        CHECK(recompile_fragment(stale_readlane,
                                 sizeof(stale_readlane)/sizeof(stale_readlane[0])).empty(),
              "#652: readlane rejects a lane slot invalidated by an ordinary VGPR write");
    }

    // MULTI-RENDER-TARGET selection (#566 — Dead Cells world/G-buffer shaders). A 4-MRT shader emits its
    // color exports in DESCENDING target order (MRT3..MRT0). Vulkan color attachment 0 must receive MRT0's
    // data; the old "first EXP with target<=7 wins" wrote MRT3 (a non-albedo G-buffer plane) into color0,
    // so the world rendered as that grayscale channel instead of the colored albedo. Here MRT3 exports RED
    // and MRT0 exports BLUE (emitted MRT3-first); the triangle must be BLUE (MRT0 chosen), not RED (MRT3).
    {
        const uint32_t ps[] = {
            0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u,   // v0=1.0,v1=0,v2=0,v3=1.0  -> RED
            0xF800183Fu, 0x03020100u,                             // exp mrt3 v0,v1,v2,v3
            0x7E080280u, 0x7E0A0280u, 0x7E0C02F2u, 0x7E0E02F2u,   // v4=0,v5=0,v6=1.0,v7=1.0  -> BLUE
            0xF800180Fu, 0x07060504u,                             // exp mrt0 v4,v5,v6,v7
            0xBF810000u,                                          // s_endpgm
        };
        std::vector<uint32_t> frg = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
        CHECK(!frg.empty(), "#566: recompiled descending-order 4-MRT (MRT3 then MRT0) PS -> SPIR-V");
        if (!frg.empty()) {
            std::vector<uint8_t> px2 = prosper::test::render_triangle_rgba(vert, frg, W, H);
            CHECK(px2.size() == (size_t)W*H*4, "rendered the MRT-order PS");
            if (px2.size() == (size_t)W*H*4) {
                const uint8_t* cc = &px2[((size_t)(H/2) * W + W/2) * 4];
                printf("  mrt-order center=(%u,%u,%u,%u) expect BLUE (MRT0), not RED (MRT3)\n",
                       cc[0],cc[1],cc[2],cc[3]);
                CHECK(cc[2] > 0x80 && cc[0] < 0x40,
                      "#566: color0 receives MRT0 (BLUE), not the first-emitted MRT3 (RED)");
            }
        }
    }

    // True MRT0+MRT1: preserve both hardware export locations and bind two simultaneous Vulkan color
    // attachments. DOLL writes its temporal scene input this way; dropping Location 1 leaves the later
    // scene sample black (#635/#719).
    {
        const uint32_t ps[] = {
            0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u,   // v0..3 = RED
            0xF800180Fu, 0x03020100u,                             // exp mrt0
            0x7E080280u, 0x7E0A02F2u, 0x7E0C0280u, 0x7E0E02F2u,   // v4..7 = GREEN
            0xF800181Fu, 0x07060504u,                             // exp mrt1
            0xBF810000u,
        };
        std::vector<uint32_t> frg = recompile_fragment(ps, std::size(ps));
        CHECK(!frg.empty(), "#635: MRT0+MRT1 fragment shader recompiles with distinct outputs");
        if (!frg.empty()) {
            prosper::test::BackendDraw draw;
            draw.vs = vert; draw.fs = frg;
            std::vector<uint8_t> mrt1;
            const float black[4] = {0, 0, 0, 1};
            std::vector<uint8_t> mrt0 = prosper::test::render_draws_rgba(
                {draw}, W, H, nullptr, black, false, nullptr, nullptr, black, &mrt1);
            CHECK(mrt0.size() == (size_t)W * H * 4 && mrt1.size() == mrt0.size(),
                  "#635: dual-attachment backend reads back both MRT surfaces");
            if (mrt0.size() == (size_t)W * H * 4 && mrt1.size() == mrt0.size()) {
                const uint8_t* c0 = &mrt0[((size_t)(H / 2) * W + W / 2) * 4];
                const uint8_t* c1 = &mrt1[((size_t)(H / 2) * W + W / 2) * 4];
                CHECK(c0[0] > 0xC0 && c0[1] < 0x40 && c0[2] < 0x40,
                      "#635: MRT0 export reaches color attachment 0");
                CHECK(c1[1] > 0xC0 && c1[0] < 0x40 && c1[2] < 0x40,
                      "#635: MRT1 export reaches color attachment 1");
            }
        }

        const uint32_t mrt3_only[] = {
            0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u,
            0xF800183Fu, 0x03020100u, 0xBF810000u,
        };
        CHECK(recompile_fragment(mrt3_only, std::size(mrt3_only)).empty(),
              "#635: unsupported MRT3-only export stays fail-visible instead of remapping to color0");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
