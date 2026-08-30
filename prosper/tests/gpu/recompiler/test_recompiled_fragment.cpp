// test_recompiled_fragment — render a triangle whose PIXEL SHADER is recompiled from RDNA2.
// The vertex shader is the placeholder (positions), but the fragment shader is real RDNA2 (assembled
// by llvm-mc) recompiled to SPIR-V by recompile_fragment: it exports green via EXP MRT0. We render
// and assert the triangle is GREEN (not the placeholder's red), proving RDNA2->SPIR-V works for an
// actual graphics-stage shader wired into a real pipeline.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"     // kTriVertSpv: placeholder vertex shader (positions)
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

    // Wave64 device capability, needed from the first EXECZ/VCCZ draw onward (#1681). The recompiler
    // lowers a wave-wide EXECZ/VCCZ test to a native subgroup vote and marks the module as requiring
    // an exact 64-lane subgroup; RADV can enforce that, while llvmpipe is fixed at 8 and the backend
    // must reject the draw rather than execute silently wrong wave semantics. Every assertion over a
    // rendered pixel from such a draw is therefore conditional on this capability — the target keeps
    // its clear colour when the device cannot supply wave64, which is correct fail-visible behaviour
    // and not a rendering difference.
    const auto& wave_ctx = prosper::test::render_vk_ctx();
    const bool supports_fragment_wave64 = wave_ctx.subgroup_size_control &&
        wave_ctx.min_subgroup_size <= 64 && wave_ctx.max_subgroup_size >= 64 &&
        (wave_ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
        (wave_ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
        (wave_ctx.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
    const bool supports_fragment_wave64_vote = supports_fragment_wave64 &&
        (wave_ctx.subgroup_operations & VK_SUBGROUP_FEATURE_VOTE_BIT);
    const bool supports_fragment_gds =
        supports_fragment_wave64 && wave_ctx.fragment_stores_atomics;
    // The BLUE clear the fixtures render against; a skipped draw must leave exactly this.
    auto is_blue_clear = [](const uint8_t* p) {
        return p && p[2] > 0x80 && p[0] < 0x40 && p[1] < 0x40;
    };
    // A clear-coloured target on its own only says "no draw landed here", which any pipeline failure
    // would also produce. Pair it with the module's own declared requirement, so the wave64-less arm
    // asserts the specific thing that should have happened: this shader demanded an exact 64-lane
    // subgroup, and the device therefore left the target untouched. Four pre-existing gated fixtures
    // already assert the required size separately; this applies the same rigour to the rest.
    auto skipped_wave64_draw = [&](const std::vector<uint32_t>& frag, const uint8_t* p) {
        return fragment_spirv_required_subgroup_size(frag) == 64 && is_blue_clear(p);
    };

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
    CHECK(direct_break_center &&
              (supports_fragment_wave64_vote
                   ? (direct_break_center[1] > 0x80 && direct_break_center[0] < 0x40)
                   : skipped_wave64_draw(direct_break_frag, direct_break_center)),
          supports_fragment_wave64_vote
              ? "unconditional-backedge EXECZ break exits directly and preserves loop merge state"
              : "device without fragment wave64 vote skips the backedge-break draw fail-visible");

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

    // Five-MRT export (#GTA5 G-buffer). EXP encoding: EN in bits[3:0], TARGET in bits[9:4],
    // COMPR bit10, DONE bit11, VM bit12; MRT0..MRT7 are targets 0..7. Each export names v0 for all
    // four channels (second dword 0), so the shader is a legal minimal five-target G-buffer writer.
    //
    // This is the regression for the defect that hid GTA V's entire 3D world: the fragment shell's
    // colour-output array, the export-mask decoder and the emit-side `exported` array were all sized
    // 2, so exports to MRT2+ were discarded. The live executor then does
    // `write_mask &= (exp_mask >> slot*4) & 0xf` per slot, which turned every slot above 1 into a
    // zero write mask -- the attachment was dropped no matter what CB_TARGET_MASK said.
    //
    // The mask below is the arm that fails without the fix: the old two-slot decoder can only ever
    // return 0x000000ff, so asserting the full five-nibble value cannot pass by coincidence.
    const uint32_t five_mrt_ps[] = {
        0x7E000280u,                 // v_mov_b32 v0, 0
        0xF800100Fu, 0x00000000u,    // exp mrt0 v0,v0,v0,v0   EN=0xf vm
        0xF800101Fu, 0x00000000u,    // exp mrt1
        0xF800102Fu, 0x00000000u,    // exp mrt2
        0xF800103Fu, 0x00000000u,    // exp mrt3
        0xF800184Fu, 0x00000000u,    // exp mrt4 ... done
        0xBF810000u,                 // s_endpgm
    };
    const uint32_t five_mrt_mask =
        fragment_color_export_mask(five_mrt_ps, std::size(five_mrt_ps));
    printf("  five-MRT export mask = 0x%08x (want 0x000fffff)\n", five_mrt_mask);
    CHECK(five_mrt_mask == 0x000fffffu,
          "five-MRT fragment reports EN for MRT0..MRT4");
    // Slots 2..4 specifically -- the ones the old array size could not represent. Stated separately
    // so a future change that widens the array but breaks the shift still fails on the right claim.
    CHECK(((five_mrt_mask >> 8) & 0xfu) == 0xfu, "five-MRT fragment: MRT2 has a full write mask");
    CHECK(((five_mrt_mask >> 12) & 0xfu) == 0xfu, "five-MRT fragment: MRT3 has a full write mask");
    CHECK(((five_mrt_mask >> 16) & 0xfu) == 0xfu, "five-MRT fragment: MRT4 has a full write mask");
    CHECK(!recompile_fragment(five_mrt_ps, std::size(five_mrt_ps)).empty(),
          "five-MRT fragment recompiles to a module with five colour outputs");

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

    // GTA V exports sample coverage through MRTZ with EN=4: VSRC2 is the sample-mask word. Preserve
    // the exact live export packet and verify both sides of its Vulkan SampleMask[0] lowering on a
    // single-sample target. A zero mask removes the covered sample; bit zero keeps it.
    auto sample_mask_ps = [](uint32_t mask_mov) {
        return std::vector<uint32_t>{
            0x7e0002f2u, 0x7e0202f2u, 0x7e0402f2u, 0x7e0602f2u, // v0..v3 = 1.0
            mask_mov,                                          // v9 = sample mask
            0xf8000084u, 0x00090000u,                          // exp mrtz off,off,v9,off
            0xf800180fu, 0x03020100u,                          // exp mrt0 v0,v1,v2,v3 done vm
            0xbf810000u,
        };
    };
    std::vector<uint32_t> sample_on_ps = sample_mask_ps(0x7e120281u);  // v9 = 1
    std::vector<uint32_t> sample_off_ps = sample_mask_ps(0x7e120280u); // v9 = 0
    std::vector<uint32_t> sample_on_frag = recompile_fragment(
        sample_on_ps.data(), sample_on_ps.size());
    std::vector<uint32_t> sample_off_frag = recompile_fragment(
        sample_off_ps.data(), sample_off_ps.size());
    CHECK(!sample_on_frag.empty() && !sample_off_frag.empty(),
          "recompiled MRTZ sample-mask export at GTA V's live EN=4 site");
    if (!sample_on_frag.empty() && !sample_off_frag.empty()) {
        std::vector<uint8_t> sample_on_px = prosper::test::render_triangle_rgba(
            vert, sample_on_frag, W, H);
        std::vector<uint8_t> sample_off_px = prosper::test::render_triangle_rgba(
            vert, sample_off_frag, W, H);
        const uint8_t* sample_on_center =
            sample_on_px.size() == static_cast<size_t>(W) * H * 4
                ? &sample_on_px[((static_cast<size_t>(H) / 2) * W + W / 2) * 4] : nullptr;
        const uint8_t* sample_off_center =
            sample_off_px.size() == static_cast<size_t>(W) * H * 4
                ? &sample_off_px[((static_cast<size_t>(H) / 2) * W + W / 2) * 4] : nullptr;
        CHECK(sample_on_center && sample_on_center[0] > 0x80 && sample_on_center[1] > 0x80 &&
                  sample_on_center[2] > 0x80,
              "MRTZ sample-mask bit zero preserves the covered sample");
        CHECK(sample_off_center && sample_off_center[2] > 0x80 && sample_off_center[0] < 0x40 &&
                  sample_off_center[1] < 0x40,
              "zero MRTZ sample mask removes the covered sample");
    }

    // Mutation arm: perturb the same live MRTZ instruction from sample-mask EN=4 to unsupported
    // stencil-reference EN=2. It must fail visibly rather than proving only the surrounding fixture.
    std::vector<uint32_t> stencil_mutation_ps = sample_on_ps;
    stencil_mutation_ps[5] = 0xf8000082u;
    CHECK(recompile_fragment(stencil_mutation_ps.data(), stencil_mutation_ps.size()).empty(),
          "same-site MRTZ stencil mutation remains fail-visible");

    const uint32_t sample_mask_only_ps[] = {
        0x7e120281u, 0xf8001884u, 0x00090000u, 0xbf810000u,
    };
    CHECK(!recompile_fragment(sample_mask_only_ps, std::size(sample_mask_only_ps)).empty(),
          "MRTZ sample-mask-only fragment passes the supported-export gate");

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
    CHECK(fragment_spirv_required_subgroup_size(fragT) == 64 &&
              (fragment_spirv_required_subgroup_features(fragT) & kFragmentSubgroupVote),
          "fragment EXECZ branch advertises its exact wave64 vote requirement");
    if (!fragT.empty() && !fragX.empty()) {
        std::vector<uint8_t> pxT = prosper::test::render_triangle_rgba(vert, fragT, W, H);
        std::vector<uint8_t> pxX = prosper::test::render_triangle_rgba(vert, fragX, W, H);
        CHECK(pxT.size() == (size_t)W*H*4 && pxX.size() == (size_t)W*H*4, "rendered both execz variants");
        if (pxT.size() == (size_t)W*H*4 && pxX.size() == (size_t)W*H*4) {
            const uint8_t* ct = &pxT[((size_t)(H/2) * W + W/2) * 4];
            const uint8_t* cx = &pxX[((size_t)(H/2) * W + W/2) * 4];
            printf("  execz taken center=(%u,%u,%u,%u) skipped center=(%u,%u,%u,%u)\n",
                   ct[0],ct[1],ct[2],ct[3], cx[0],cx[1],cx[2],cx[3]);
            CHECK(supports_fragment_wave64_vote
                      ? ct[0] > 0x80 && ct[1] > 0x80 && ct[2] > 0x80
                      : ct[2] > 0x80 && ct[0] < 0x40 && ct[1] < 0x40,
                  supports_fragment_wave64_vote
                      ? "execz taken: one wave-wide vote runs the block for the complete wave"
                      : "device without fragment wave64 vote skips the draw fail-visible");
            CHECK(cx[2] > 0x80 && cx[0] < 0x40 && cx[1] < 0x40,
                  "execz skipped: center stays BLUE clear (all lanes inactive)");
        }
    }

    // Plucky Squire's material PS compares a vector-produced VCC mask with a saved EXEC pair through
    // s_cmp_lg_u64. Model a sparse mask directly (`vcc = 1`: only guest lane zero) and prove the
    // scalar SCC result observes all 64 bits, rather than trying to read VCC_LO as scalar data or
    // comparing this invocation alone. SCC=true selects green for the complete draw.
    const uint32_t wave_mask_compare_ps[] = {
        0xBE80047Eu,              // s_mov_b64 s[0:1], exec
        0xBEEA0481u,              // s_mov_b64 vcc, 1 (only lane zero set)
        0xBF13006Au,              // s_cmp_lg_u64 vcc, s[0:1]
        0x850580F2u,              // s_cselect_b32 s5, 1.0, 0.0
        0x7E000280u, 0x7E020205u, 0x7E040280u, 0x7E0602F2u,
        0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    std::vector<uint32_t> wave_mask_compare = recompile_fragment(
        wave_mask_compare_ps, std::size(wave_mask_compare_ps));
    CHECK(!wave_mask_compare.empty() &&
              fragment_spirv_required_subgroup_size(wave_mask_compare) == 64 &&
              (fragment_spirv_required_subgroup_features(wave_mask_compare) &
               kFragmentSubgroupVote),
          "recompiled sparse VCC-vs-EXEC u64 compare through an exact wave64 vote");
    if (!wave_mask_compare.empty()) {
        std::vector<uint8_t> mask_px = prosper::test::render_triangle_rgba(
            vert, wave_mask_compare, W, H);
        const uint8_t* mask_center = mask_px.size() == static_cast<size_t>(W) * H * 4
            ? &mask_px[((static_cast<size_t>(H) / 2) * W + W / 2) * 4] : nullptr;
        if (mask_center)
            std::printf("  wave-mask compare center=(%u,%u,%u,%u)\n",
                        mask_center[0], mask_center[1], mask_center[2], mask_center[3]);
        CHECK(mask_center && (supports_fragment_wave64_vote
                  ? mask_center[1] > 0x80 && mask_center[0] < 0x40
                  : mask_center[2] > 0x80 && mask_center[0] < 0x40 && mask_center[1] < 0x40),
              supports_fragment_wave64_vote
                  ? "s_cmp_lg_u64 detects bits missing outside the sparse VCC lane"
                  : "device without fragment wave64 vote skips the mask-compare draw fail-visible");
    }
    std::vector<uint32_t> equal_mask_ps(
        std::begin(wave_mask_compare_ps), std::end(wave_mask_compare_ps));
    equal_mask_ps[1] = 0xBEEA04C1u;  // s_mov_b64 vcc, -1: exactly the full entry EXEC mask
    std::vector<uint32_t> equal_mask_compare = recompile_fragment(
        equal_mask_ps.data(), equal_mask_ps.size());
    CHECK(!equal_mask_compare.empty(),
          "recompiled equal VCC-vs-EXEC u64 comparison for the opposite SCC outcome");
    if (!equal_mask_compare.empty()) {
        std::vector<uint8_t> equal_px = prosper::test::render_triangle_rgba(
            vert, equal_mask_compare, W, H);
        const uint8_t* equal_center = equal_px.size() == static_cast<size_t>(W) * H * 4
            ? &equal_px[((static_cast<size_t>(H) / 2) * W + W / 2) * 4] : nullptr;
        CHECK(equal_center && (supports_fragment_wave64_vote
                  ? equal_center[0] < 0x20 && equal_center[1] < 0x20 && equal_center[2] < 0x20
                  : equal_center[2] > 0x80 && equal_center[0] < 0x40 && equal_center[1] < 0x40),
              supports_fragment_wave64_vote
                  ? "s_cmp_lg_u64 clears SCC when all 64 VCC and EXEC bits match"
                  : "device without fragment wave64 vote also skips the equal-mask draw");
    }

    // A second Plucky material loop uses an unconditional back-edge plus a whole-wave VCCZ break.
    // VCCZ does not clear EXEC, so the old per-lane shortcut could neither reach the loop merge nor
    // safely use the back-edge (it rejected the 2.6 KiB shader at its outer EXECZ). This reduced
    // form enters with full EXEC, clears VCC, and must break before the white write instead of looping.
    const uint32_t direct_vcc_break_ps[] = {
        0xBE80047Eu,              // s_mov_b64 s[0:1], exec
        0x7E020280u, 0x7E040280u, 0x7E0602F2u,  // black RGB seed, alpha=1
        0xBF880004u,              // loop: s_cbranch_execz exit
        0xBEEA0480u,              // s_mov_b64 vcc, 0
        0xBF860002u,              // s_cbranch_vccz exit (whole-wave direct break)
        0x7E0202F2u,              // must not run: v1 = 1.0
        0xBF82FFFBu,              // s_branch loop
        0xBEFE0400u,              // exit: restore exec
        0xF800180Fu, 0x03010101u, 0xBF810000u,
    };
    std::vector<uint32_t> direct_vcc_break = recompile_fragment(
        direct_vcc_break_ps, std::size(direct_vcc_break_ps));
    CHECK(!direct_vcc_break.empty() &&
              fragment_spirv_required_subgroup_size(direct_vcc_break) == 64,
          "recompiled unconditional-backedge loop with a direct wave-wide VCCZ break");
    if (!direct_vcc_break.empty()) {
        std::vector<uint8_t> break_px = prosper::test::render_triangle_rgba(
            vert, direct_vcc_break, W, H);
        const uint8_t* break_center = break_px.size() == static_cast<size_t>(W) * H * 4
            ? &break_px[((static_cast<size_t>(H) / 2) * W + W / 2) * 4] : nullptr;
        CHECK(break_center && (supports_fragment_wave64_vote
                  ? break_center[0] < 0x20 && break_center[1] < 0x20 && break_center[2] < 0x20
                  : break_center[2] > 0x80 && break_center[0] < 0x40 && break_center[1] < 0x40),
              supports_fragment_wave64_vote
                  ? "VCCZ exits the complete wave before the guarded write and back-edge"
                  : "device without fragment wave64 vote skips the direct-break draw");
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
                CHECK(supports_fragment_wave64_vote
                          ? (cc[1] > 0xF0 && cc[0] > 0x70 && cc[0] < 0x90 &&
                             cc[2] > 0x70 && cc[2] < 0x90)
                          : skipped_wave64_draw(frg, cc),
                      supports_fragment_wave64_vote
                          ? "divergent loop: 4 iterations, nested if 2 -> (0.5, 1.0, 0.5)"
                          : "device without fragment wave64 vote skips the divergent-loop draw");
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
                CHECK(supports_fragment_wave64_vote
                          ? (cc[0] > 0xF0 && cc[1] > 0xF0 && cc[2] > 0xF0)
                          : skipped_wave64_draw(frg, cc),
                      supports_fragment_wave64_vote
                          ? "VCCZ loop: exactly 4 iterations of 0.25 -> 1.0 white"
                          : "device without fragment wave64 vote skips the VCCZ-loop draw");
            }
        }

        // Cobra's Unity material PS derives its loop bound through scalar-fed VOP2 + VOP1 before
        // comparing it with the scalar counter. The resulting VGPR is still wave-uniform.
        const uint32_t scalar_chain_ps[] = {
            0xBE800380u, 0x7E020284u,            // s0 = 0; v1 = scalar constant 4
            0x4A020281u,                         // lane-local VOP2 v1 = v1 + 1
            0x7E021101u,                         // lane-local VOP1 v1 = f32(v1)
            0x7D020200u, 0xBF860002u,           // v_cmp_* vcc,s0,v1; vccz -> exit
            0x81008100u, 0xBF82FFFCu,           // ++s0; back to compare
            0x7E0002F2u, 0xF800180Fu, 0x00000000u, 0xBF810000u,
        };
        CHECK(!recompile_fragment(scalar_chain_ps, std::size(scalar_chain_ps)).empty(),
              "a scalar-fed VOP chain is accepted as a uniform VCC-loop bound");
        uint32_t varying_chain_ps[std::size(scalar_chain_ps)];
        std::copy(std::begin(scalar_chain_ps), std::end(scalar_chain_ps), varying_chain_ps);
        varying_chain_ps[1] = 0x7E020302u;  // v1 = unresolved lane-varying v2 before the same chain
        CHECK(!recompile_fragment(varying_chain_ps, std::size(varying_chain_ps)).empty(),
              "fragment wave64 vote accepts a lane-varying VCC loop bound exactly");

        uint32_t varying_ps[sizeof(ps)/sizeof(ps[0])];
        std::copy(std::begin(ps), std::end(ps), varying_ps);
        varying_ps[2] = 0x7E020302u;  // v_mov_b32 v1,v2: a lane-varying/unproven loop bound
        CHECK(!recompile_fragment(varying_ps, sizeof(varying_ps)/sizeof(varying_ps[0])).empty(),
              "fragment wave64 vote keeps a varying-VCC loop wave-uniform");

        uint32_t exec_mutating_ps[sizeof(ps)/sizeof(ps[0])];
        std::copy(std::begin(ps), std::end(ps), exec_mutating_ps);
        exec_mutating_ps[3] = 0xBEFE04C1u;  // s_mov_b64 exec,-1 between the bound definition and compare
        CHECK(!recompile_fragment(exec_mutating_ps,
                                  sizeof(exec_mutating_ps)/sizeof(exec_mutating_ps[0])).empty(),
              "fragment wave64 vote tracks a VCC loop across an EXEC restore");
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
                CHECK(supports_fragment_wave64_vote
                          ? (cc[0] > 0xA8 && cc[0] < 0xD8 && cc[1] > 0xA8 && cc[1] < 0xD8)
                          : skipped_wave64_draw(frg, cc),
                      supports_fragment_wave64_vote
                          ? "execnz loop + break: exactly 3 iterations -> 0.75 gray"
                          : "device without fragment wave64 vote skips the execnz-loop draw");
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
                CHECK(supports_fragment_wave64_vote
                          ? (cc[0] > 0xA8 && cc[0] < 0xD8 && cc[1] > 0xA8 && cc[1] < 0xD8 &&
                             cc[2] > 0xA8 && cc[2] < 0xD8)
                          : skipped_wave64_draw(frg, cc),
                      supports_fragment_wave64_vote
                          ? "#590: nested loops 3x4: inner 12 x 0.0625 and outer 3 x 0.25 both = 0.75"
                          : "device without fragment wave64 vote skips the nested-loop draw");
            }
        }

        // Partially-overlapping loops (a back-edge into another loop's body without proper nesting).
        // The narrow pattern structurizer still calls this unstructured and rejects it, but the
        // graphics CFG dispatcher added for Astro Bot's materials then executes the exact block graph
        // per invocation, so the region lowers rather than dropping the draw (#1474). This test owns
        // the half that needs a device: a real driver has to accept the module. The accept/reject
        // contract itself — including that the export survives the lowering — is pinned device-free in
        // test_rdna2_spirv_struct, because these Vulkan-execution tests are gated on
        // find_package(Vulkan) succeeding and every CI job that runs ctest either disables Vulkan
        // discovery (Linux, Windows MinGW, macOS) or runs a three-test seam subset (Windows App), so
        // nothing here runs in CI.
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
        std::vector<uint32_t> overlap_frag =
            recompile_fragment(overlap_ps, sizeof(overlap_ps)/sizeof(overlap_ps[0]));
        CHECK(!overlap_frag.empty() && overlap_frag[0] == 0x07230203u,
              "#1474: partially-overlapping fragment loops lower to a SPIR-V module");
        if (!overlap_frag.empty()) {
            // Pipeline creation is the assertion: the driver validates the dispatcher's output, so a
            // structurally invalid lowering fails here rather than silently producing a module.
            std::vector<uint8_t> overlap_px =
                prosper::test::render_triangle_rgba(vert, overlap_frag, W, H);
            CHECK(overlap_px.size() == (size_t)W * H * 4,
                  "#1474: a real driver accepts the dispatcher-lowered overlapping-loop shader");
        }
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
        std::vector<uint32_t> recycled_readlane = recompile_fragment(
            stale_readlane, sizeof(stale_readlane)/sizeof(stale_readlane[0]));
        CHECK(!recycled_readlane.empty(),
              "#652: readlane observes the ordinary VGPR after its spill lifetime ends");
        if (!recycled_readlane.empty()) {
            std::vector<uint8_t> px = prosper::test::render_triangle_rgba(
                vert, recycled_readlane, W, H);
            const uint8_t* center = px.empty() ? nullptr
                : &px[((size_t)(H / 2) * W + W / 2) * 4];
            CHECK(center && (supports_fragment_wave64_vote ? center[0] > 0xC0
                                                          : skipped_wave64_draw(recycled_readlane, center)),
                  supports_fragment_wave64_vote
                      ? "#652: recycled v10 lane 3 contains the new 1.0 value, not its stale spill"
                      : "device without fragment wave64 vote skips the recycled-lane draw");
        }
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
        // Rendered with FOUR attachments, which is what a shader exporting MRT3 is paired with in a
        // real frame. It used to render into a single attachment, and that was only harmless while
        // the recompiler silently dropped MRT3: once the shell carries eight outputs the module
        // legitimately writes Location 3, and a one-attachment pass makes that write unused --
        // caught by tools/vkval as `Undefined-Value-ShaderOutputNotConsumed`. Widening the pass is
        // the faithful fix and strengthens the assertion: MRT0's blue must land on attachment 0 AND
        // MRT3's red on attachment 3, which pins the mapping in both directions rather than only
        // proving color0 is not red.
        std::vector<uint32_t> frg = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
        CHECK(!frg.empty(), "#566: recompiled descending-order 4-MRT (MRT3 then MRT0) PS -> SPIR-V");
        if (!frg.empty()) {
            prosper::test::BackendDraw draw;
            draw.vs = vert; draw.fs = frg;
            const float black[4] = {0, 0, 0, 1};
            prosper::test::BackendMrtOutputs outputs;
            outputs.color_count = 4;
            std::vector<uint8_t> px2 = prosper::test::render_draws_rgba(
                {draw}, W, H, nullptr, black, false, nullptr, nullptr, black, nullptr, nullptr,
                true, &outputs);
            const size_t want = (size_t)W * H * 4;
            CHECK(px2.size() == want && outputs.colors[3].size() == want,
                  "rendered the MRT-order PS into four attachments");
            if (px2.size() == want && outputs.colors[3].size() == want) {
                const uint8_t* cc = &px2[((size_t)(H/2) * W + W/2) * 4];
                const uint8_t* c3 = &outputs.colors[3][((size_t)(H/2) * W + W/2) * 4];
                printf("  mrt-order attachment0=(%u,%u,%u,%u) attachment3=(%u,%u,%u,%u)\n",
                       cc[0],cc[1],cc[2],cc[3], c3[0],c3[1],c3[2],c3[3]);
                CHECK(cc[2] > 0x80 && cc[0] < 0x40,
                      "#566: color0 receives MRT0 (BLUE), not the first-emitted MRT3 (RED)");
                CHECK(c3[0] > 0x80 && c3[2] < 0x40,
                      "#566: MRT3's RED lands on attachment 3, not discarded and not on color0");
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

        // Slots 2..7 must RETAIN across render groups (#2550 review). Before persistence they were
        // transient images created per backend call with LOAD_OP_CLEAR, so a G-buffer assembled by
        // several groups against one set of allocations kept only the last group's work. GTA V's
        // G-buffer accumulates across roughly sixteen groups per frame, so this was the difference
        // between three complete attachments and three attachments holding one group each.
        //
        // Two groups against ONE persistent slot-2 id: the first writes red to MRT2, the second
        // writes only MRT0 and must leave MRT2 alone. The negative arm below re-runs the second
        // group with load_existing off and requires MRT2 to come back cleared -- without it, a test
        // that never loads would pass just as happily on a backend that never clears.
        const uint32_t mrt0_and_mrt2[] = {
            0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u,   // v0..3 = GREEN
            0xF800100Fu, 0x03020100u,                             // exp mrt0  (vm)
            0x7E0802F2u, 0x7E0A0280u, 0x7E0C0280u, 0x7E0E02F2u,   // v4..7 = RED
            0xF800182Fu, 0x07060504u,                             // exp mrt2  (vm, done)
            0xBF810000u,
        };
        const uint32_t mrt0_only[] = {
            0x7E000280u, 0x7E020280u, 0x7E0402F2u, 0x7E0602F2u,   // v0..3 = BLUE
            0xF800180Fu, 0x03020100u,                             // exp mrt0  (vm, done)
            0xBF810000u,
        };
        std::vector<uint32_t> frg_a = recompile_fragment(mrt0_and_mrt2, std::size(mrt0_and_mrt2));
        std::vector<uint32_t> frg_b = recompile_fragment(mrt0_only, std::size(mrt0_only));
        CHECK(!frg_a.empty() && !frg_b.empty(),
              "#2550: MRT0+MRT2 and MRT0-only fragment shaders both recompile");
        if (!frg_a.empty() && !frg_b.empty()) {
            const float black[4] = {0, 0, 0, 1};
            auto run_group = [&](const std::vector<uint32_t>& fs, uint64_t slot2_id,
                                 bool load_existing, std::vector<uint8_t>& slot2_out) {
                prosper::test::BackendDraw draw;
                draw.vs = vert; draw.fs = fs;
                prosper::test::BackendColorTarget target{};
                target.persistent_id_slots[2] = slot2_id;
                target.load_existing_slots[2] = load_existing;
                prosper::test::BackendMrtOutputs outputs;
                outputs.color_count = 3;
                prosper::test::render_draws_rgba({draw}, W, H, nullptr, black, false, &target,
                                                 nullptr, black, nullptr, nullptr, true, &outputs);
                slot2_out = std::move(outputs.colors[2]);
            };
            constexpr uint64_t kSlot2Id = 0x20aa2200ull;
            std::vector<uint8_t> first, second, cleared;
            run_group(frg_a, kSlot2Id, true, first);
            run_group(frg_b, kSlot2Id, true, second);
            const size_t px = (size_t)W * H * 4;
            CHECK(first.size() == px && second.size() == px,
                  "#2550: slot 2 reads back at full extent from both render groups");
            if (first.size() == px && second.size() == px) {
                const uint8_t* f = &first[((size_t)(H / 2) * W + W / 2) * 4];
                const uint8_t* g = &second[((size_t)(H / 2) * W + W / 2) * 4];
                printf("  slot2 group1=(%u,%u,%u) group2=(%u,%u,%u)\n",
                       f[0], f[1], f[2], g[0], g[1], g[2]);
                CHECK(f[0] > 0xC0 && f[1] < 0x40 && f[2] < 0x40,
                      "#2550: group 1 writes RED to the retained MRT2 attachment");
                CHECK(g[0] > 0xC0 && g[1] < 0x40 && g[2] < 0x40,
                      "#2550: group 2 does NOT clear MRT2 -- the first group's pixels survive");
            }
            // Negative arm: the same second group with load_existing off must come back cleared, so
            // the assertion above is demonstrably sensitive to the LOAD and not to the backend
            // simply never touching the attachment.
            run_group(frg_a, kSlot2Id + 0x1000, true, first);
            run_group(frg_b, kSlot2Id + 0x1000, false, cleared);
            // Assert the extent, do not merely gate on it. A regression that makes the no-load call
            // fail and return an EMPTY buffer would otherwise skip the only assertion in this arm
            // and take the negative control down with it, silently.
            CHECK(cleared.size() == px,
                  "#2550: the no-load arm reads back at full extent (its assertion cannot be "
                  "skipped by an empty result)");
            if (cleared.size() == px) {
                const uint8_t* c = &cleared[((size_t)(H / 2) * W + W / 2) * 4];
                printf("  slot2 no-load group2=(%u,%u,%u)\n", c[0], c[1], c[2]);
                CHECK(c[0] < 0x40,
                      "#2550: with load_existing off the second group DOES clear MRT2");
            }
        }

        // The backend colour-format mapping that frontends/shared/rtt/mrt_binding.hpp's active-binding
        // rule depends on. It is TOTAL: every unrecognised raw value, zero included, maps to
        // R8G8B8A8_UNORM, so the format term in that rule never rejects a slot. Pinned here, in a
        // test that actually links the backend, because mrt_binding's own test must model the
        // predicate rather than call it -- and a silently narrowed mapping would invalidate that
        // model instead of failing.
        CHECK(prosper::test::backend_color_format(VK_FORMAT_UNDEFINED) ==
                  VK_FORMAT_R8G8B8A8_UNORM,
              "#2550: an undefined colour format maps to the RGBA8 fallback, it is not rejected");
        CHECK(prosper::test::backend_color_format(static_cast<VkFormat>(0x7fffffff)) ==
                  VK_FORMAT_R8G8B8A8_UNORM,
              "#2550: the backend colour-format mapping is total");

        // #2550 review round 3: the depth-feedback splitter's per-segment contract. A logical
        // five-MRT pass that splits at a depth write->sample transition handed every pre-final
        // segment `mrt_outputs == nullptr`, so those segments rendered with ONE attachment and
        // discarded their MRT1..4 exports -- the same accumulation loss this PR fixes at the
        // frontend's pass groups, reappearing at the backend's own physical boundary.
        {
            prosper::test::BackendColorTarget whole{};
            whole.persistent_id = 0x2050000000ull;
            whole.persistent_id_slots[2] = 0x2083e00000ull;
            prosper::test::BackendMrtOutputs whole_mrt;
            whole_mrt.color_count = 5;

            const auto first_seg = prosper::test::split_segment_contract(
                &whole, &whole_mrt, /*first=*/true, /*final=*/false);
            const auto mid_seg = prosper::test::split_segment_contract(
                &whole, &whole_mrt, /*first=*/false, /*final=*/false);
            const auto last_seg = prosper::test::split_segment_contract(
                &whole, &whole_mrt, /*first=*/false, /*final=*/true);

            // The shape is identical across segments. This is the blocker: it used to be 1 for
            // every segment but the last.
            CHECK(first_seg.color_count == 5 && mid_seg.color_count == 5 &&
                      last_seg.color_count == 5,
                  "#2550: every segment of a split pass renders the same MRT shape");
            // Later segments LOAD every slot, or they erase their predecessors' work.
            CHECK(!first_seg.target.load_existing_slots[2] || whole.load_existing_slots[2],
                  "#2550: the first segment does not force a load it was not asked for");
            CHECK(mid_seg.target.load_existing_slots[2] && last_seg.target.load_existing_slots[2] &&
                      mid_seg.target.load_existing && mid_seg.target.load_existing1,
                  "#2550: every later segment loads every persistent slot");
            // A non-final segment discards its own slot-0 pixels but must COPY OUT everything it
            // may have to carry -- slot 1 and every active slot 2+ -- or the next segment has
            // nothing to be seeded with. Asserted on the value the helper returns, which is the
            // value the segment actually renders under: an earlier version left this decision to
            // the caller, so this assertion read `non-final segments read back no slot` and
            // documented the opposite of the effective contract.
            CHECK(!first_seg.target.readback && !mid_seg.target.readback,
                  "#2550: a non-final segment does not copy out slot 0");
            CHECK(first_seg.target.readback_slots[2] && mid_seg.target.readback_slots[2] &&
                      first_seg.target.readback1 && mid_seg.target.readback1,
                  "#2550: a non-final segment copies out every slot it may have to carry");
            // Slots outside the active prefix are not touched.
            CHECK(!first_seg.target.readback_slots[5],
                  "#2550: a slot beyond color_count is not read back");
            CHECK(last_seg.target.readback_slots[2] && last_seg.target.readback,
                  "#2550: the final segment keeps the caller's readback contract");
            // The persistent identities must survive unchanged, or later segments would retain a
            // different image than the one they are accumulating into.
            CHECK(mid_seg.target.persistent_id_slots[2] == 0x2083e00000ull &&
                      mid_seg.target.persistent_id == 0x2050000000ull,
                  "#2550: a segment keeps the whole pass's persistent identities");
        }

        // MRT3-only. This asserted `.empty()` until 2026-08-15, when the fragment shell's colour
        // outputs went from 2 to 8 -- MRT3 is now genuinely supported, so rejecting it would be the
        // defect rather than the guard. The original concern was never "MRT3 must fail"; it was that
        // MRT3 must not be silently REMAPPED onto color0, which is what a naive widening would do
        // and what the mask assertion below forbids directly: slot 3 set, slot 0 clear.
        const uint32_t mrt3_only[] = {
            0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u,
            0xF800183Fu, 0x03020100u, 0xBF810000u,
        };
        CHECK(!recompile_fragment(mrt3_only, std::size(mrt3_only)).empty(),
              "#635: MRT3-only export recompiles now that the shell carries eight colour outputs");
        const uint32_t mrt3_mask = fragment_color_export_mask(mrt3_only, std::size(mrt3_only));
        printf("  MRT3-only export mask = 0x%08x (want 0x0000f000)\n", mrt3_mask);
        CHECK(mrt3_mask == 0x0000f000u,
              "#635: MRT3-only export lands on slot 3 and does NOT remap onto color0");
    }

    {
        // #3138: v_ldexp_f32 with ABS on src0. src0 is the FLOAT operand, where ABS/NEG are the
        // ordinary VOP3 modifiers every other float op here applies; only the integer EXPONENT
        // (src1) has no meaning for them. The old gate refused all three sources together, so
        // Stray's `0x3011560000` -- 3684 dwords with exactly ONE unsupported instruction,
        // `v_ldexp_f32 v39, |v35|, -2` -- failed entirely and discarded 1536 full-screen draws.
        //
        //   v_mov_b32 v4, -1.0 | v_mov_b32 v0, 0 | v_ldexp_f32 v1, |v4|, -2
        //   v_mov_b32 v2, 0    | v_mov_b32 v3, 1.0 | exp mrt0 v0..v3 | s_endpgm
        //
        // GREEN carries |-1.0| * 2^-2 = +0.25, and the sign is the point: without ABS the result is
        // -0.25, which clamps to zero and reads as black. So the two assertions separate the three
        // outcomes that matter -- the stage does not compile at all (the old behaviour), it compiles
        // but silently drops the modifier (green 0), or it compiles and applies it (green > 0).
        const uint32_t ldexp_abs[] = {
            0x7E0802F3u,                       // v_mov_b32 v4, -1.0
            0x7E000280u,                       // v_mov_b32 v0, 0        (red)
            0xD7620101u, 0x00018504u,          // v_ldexp_f32 v1, |v4|, -2   (green)
            0x7E040280u,                       // v_mov_b32 v2, 0        (blue)
            0x7E0602F2u,                       // v_mov_b32 v3, 1.0      (alpha)
            0xF800180Fu, 0x03020100u,          // exp mrt0, v0..v3
            0xBF810000u,                       // s_endpgm
        };
        const std::vector<uint32_t> ld_frag = recompile_fragment(ldexp_abs, std::size(ldexp_abs));
        CHECK(!ld_frag.empty(),
              "#3138: v_ldexp_f32 with ABS on its float source recompiles");
        // Deliberately NOT nested under the check above: a reverted gate must redden BOTH arms, and
        // it does only if the value arm still runs and finds no pixel. Nesting it would make the
        // second arm silently SKIP on exactly the mutation it exists to catch.
        std::vector<uint32_t> ld_vert(kTriVertSpv, kTriVertSpv + std::size(kTriVertSpv));
        const std::vector<uint8_t> ld_px =
            ld_frag.empty() ? std::vector<uint8_t>()
                            : prosper::test::render_triangle_rgba(ld_vert, ld_frag, W, H);
        const uint8_t* c = ld_px.size() == (size_t)W * H * 4
            ? &ld_px[((size_t)(H / 2) * W + W / 2) * 4] : nullptr;
        printf("  #3138 centre pixel = %02x %02x %02x (want 00 40 00)\n",
               c ? c[0] : 0, c ? c[1] : 0, c ? c[2] : 0);
        // A BAND, not a floor. `> 0x20` would also accept 0xff, which is what an
        // ABS-applied-but-exponent-dropped lowering produces (|-1.0| * 2^0 = 1.0) -- the competing
        // wrong answer this arm most needs to exclude. |-1.0| * 2^-2 = 0.25 = 0x40 exactly, and this
        // is the only numeric check of ldexp_f32_bits in the tree.
        CHECK(c && c[1] > 0x30 && c[1] < 0x50 && c[0] < 0x20 && c[2] < 0x20,
              "#3138: ABS applied AND the exponent honoured (green is 0.25, not 0 and not 1.0)");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
