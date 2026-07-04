// test_pipeline_render — the Vulkan-gated half of pipeline realization: build a real VkPipeline whose
// fixed-function state comes from resolve_pipeline_state(RenderState) and prove the resolved state
// actually drives rendering. The chain under test is end-to-end: RDNA2 registers (a GpuState) ->
// extract_render_state -> resolve_pipeline_state -> VkGraphicsPipelineCreateInfo -> pixels.
//
// Deterministic proof via the color write mask: the embedded shader draws a solid-RED triangle over a
// BLUE clear. With CB_TARGET_MASK=0xF the triangle is visible (center pixel red); with CB_TARGET_MASK=0
// no channels are written, so the triangle is invisible (center stays blue). Both masks are produced
// by resolve_pipeline_state from a real register value — not hand-set — so this exercises the whole path.
#include "../src/gpu/render_state.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include "render_runner.h"
#include "spirv_triangle.h"
#include <cstdio>
#include <vector>

using namespace prosper::gpu;
using prosper::test::render_triangle_rgba;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Build a minimal GpuState whose context registers describe a triangle-list draw with the given
// per-MRT0 color write mask, then run it through the real extract -> resolve path.
static ResolvedPipelineState resolve_for(uint32_t target_mask) {
    namespace P = prosper::agc::Pm4;
    GpuState st;
    // VGT_PRIMITIVE_TYPE.PRIM_TYPE = 4 (triangle list). Field is at bit 0 for PRIM_TYPE.
    st.cx[P::VGT_PRIMITIVE_TYPE] = 4;
    st.cx[P::CB_TARGET_MASK]     = target_mask;
    return resolve_pipeline_state(extract_render_state(st));
}

int main() {
    printf("== test_pipeline_render ==\n");
    const uint32_t W = 64, H = 64;
    const size_t center = ((size_t)(H/2) * W + (W/2)) * 4;

    std::vector<uint32_t> vert(kTriVertSpv, kTriVertSpv + sizeof(kTriVertSpv)/4);
    std::vector<uint32_t> frag(kTriFragSpv, kTriFragSpv + sizeof(kTriFragSpv)/4);

    // Full write mask (CB_TARGET_MASK=0xF): the red triangle must be visible at the center.
    ResolvedPipelineState full = resolve_for(0xF);
    CHECK(full.topology == 3, "prim_type 4 resolved to VK topology TRIANGLE_LIST (3)");
    CHECK(full.color_write_mask == 0xF, "CB_TARGET_MASK 0xF resolved to RGBA write mask");
    std::vector<uint8_t> img = render_triangle_rgba(vert, frag, W, H, &full);
    CHECK(img.size() == (size_t)W*H*4, "rendered with resolved pipeline state (full mask)");
    if (img.size() == (size_t)W*H*4) {
        uint8_t r = img[center], g = img[center+1], b = img[center+2];
        printf("  center (full mask)   = (%u,%u,%u)\n", r, g, b);
        CHECK(r > 128 && b < 128, "full mask: center pixel is the red triangle (write mask honored)");
    }

    // Zero write mask (CB_TARGET_MASK=0): no channels written, so the triangle is invisible — the
    // center keeps the blue clear color. This is the same pipeline except for the resolved write mask.
    ResolvedPipelineState none = resolve_for(0x0);
    CHECK(none.color_write_mask == 0x0, "CB_TARGET_MASK 0 resolved to empty write mask");
    std::vector<uint8_t> img0 = render_triangle_rgba(vert, frag, W, H, &none);
    CHECK(img0.size() == (size_t)W*H*4, "rendered with resolved pipeline state (zero mask)");
    if (img0.size() == (size_t)W*H*4) {
        uint8_t r = img0[center], g = img0[center+1], b = img0[center+2];
        printf("  center (zero mask)   = (%u,%u,%u)\n", r, g, b);
        CHECK(b > 128 && r < 128, "zero mask: center pixel stays blue clear (write mask honored)");
    }

    // Blend honored: additive blend (src=One, dst=One, Add) of the red triangle over the blue clear
    // must produce magenta at the center. The blend fields come from resolve_pipeline_state (RenderState
    // blend enums -> Vk blend enums), so this proves the resolved blend state drives real output.
    RenderState rsb;
    rsb.prim_type      = 4;       // triangle list
    rsb.cb_target_mask = 0xF;     // write RGBA
    rsb.blend_enable   = true;
    rsb.color_src_blend = 0x01;   // One  -> VK ONE (1)
    rsb.color_dst_blend = 0x01;   // One  -> VK ONE (1)
    rsb.color_comb_fcn  = 0;      // Add  -> VK_BLEND_OP_ADD (0)
    ResolvedPipelineState blend = resolve_pipeline_state(rsb);
    CHECK(blend.blend_enable && blend.src_color_blend_factor == 1 && blend.dst_color_blend_factor == 1 &&
          blend.color_blend_op == 0, "resolved additive blend state (One/One/Add)");
    std::vector<uint8_t> imgb = render_triangle_rgba(vert, frag, W, H, &blend);
    if (imgb.size() == (size_t)W*H*4) {
        uint8_t r = imgb[center], g = imgb[center+1], b = imgb[center+2];
        printf("  center (additive)    = (%u,%u,%u)\n", r, g, b);
        CHECK(r > 128 && g < 128 && b > 128, "additive blend: red over blue -> magenta (blend state honored)");
    } else { CHECK(false, "additive blend render produced a frame"); }

    // Depth honored: the harness clears depth to 0.5 and the triangle's fragments are at z=0.0.
    // ZFUNC=LESS (1): 0.0 < 0.5 -> passes -> red visible. ZFUNC=GREATER (4): 0.0 > 0.5 -> fails ->
    // triangle rejected, center stays blue. Both compare ops come from resolve_pipeline_state.
    RenderState rsz; rsz.prim_type = 4; rsz.cb_target_mask = 0xF;
    rsz.z_enable = true; rsz.z_write_enable = true;
    rsz.zfunc = 1;   // LESS
    ResolvedPipelineState zpass = resolve_pipeline_state(rsz);
    CHECK(zpass.depth_test_enable && zpass.depth_compare_op == 1, "resolved depth test LESS");
    std::vector<uint8_t> imgzp = render_triangle_rgba(vert, frag, W, H, &zpass);
    if (imgzp.size() == (size_t)W*H*4) {
        uint8_t r = imgzp[center], b = imgzp[center+2];
        printf("  center (z LESS)      = (%u,%u,%u)\n", r, imgzp[center+1], b);
        CHECK(r > 128 && b < 128, "depth LESS: fragment z=0 < clear 0.5 passes -> red (depth honored)");
    } else { CHECK(false, "depth LESS render produced a frame"); }

    rsz.zfunc = 4;   // GREATER
    ResolvedPipelineState zfail = resolve_pipeline_state(rsz);
    CHECK(zfail.depth_compare_op == 4, "resolved depth test GREATER");
    std::vector<uint8_t> imgzf = render_triangle_rgba(vert, frag, W, H, &zfail);
    if (imgzf.size() == (size_t)W*H*4) {
        uint8_t r = imgzf[center], b = imgzf[center+2];
        printf("  center (z GREATER)   = (%u,%u,%u)\n", r, imgzf[center+1], b);
        CHECK(b > 128 && r < 128, "depth GREATER: fragment z=0 > clear 0.5 fails -> stays blue (depth honored)");
    } else { CHECK(false, "depth GREATER render produced a frame"); }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
