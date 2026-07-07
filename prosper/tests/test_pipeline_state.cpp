// test_pipeline_state — resolve_pipeline_state() turns a RenderState's RDNA2 register semantics into
// Vulkan-ready pipeline state. Pure (no Vulkan), so it runs on every host. Inputs are chosen so the
// RDNA2->Vulkan mappings are NON-identity (e.g. RDNA2 triangle-strip=6 -> VK topology 4; DstColor
// blend=8 -> VK DST_COLOR=4; Min comb=2 -> VK MIN=3) — proving translation, not passthrough.
#include "../src/gpu/render_state.hpp"
#include <cstdio>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_pipeline_state ==\n");

    RenderState rs;
    rs.prim_type          = 6;      // RDNA2 triangle strip  -> VK topology 4 (non-identity)
    rs.color0_format      = 0xA;    // COLOR_8_8_8_8
    rs.color0_number_type = 6;      // SRGB
    rs.color0_comp_swap   = 1;      // BGRA                  -> VK_FORMAT_B8G8R8A8_SRGB (50)
    rs.z_enable           = true;
    rs.z_write_enable     = true;
    rs.zfunc              = 4;       // GREATER               -> VkCompareOp 4
    rs.blend_enable       = true;
    rs.color_src_blend    = 0x08;   // DstColor              -> VK DST_COLOR (4)
    rs.color_dst_blend    = 0x05;   // OneMinusSrcAlpha      -> VK ONE_MINUS_SRC_ALPHA (7)
    rs.color_comb_fcn     = 2;      // Min                   -> VK_BLEND_OP_MIN (3)
    rs.cb_target_mask     = 0x7;    // MRT0 write mask RGB (no alpha)

    ResolvedPipelineState ps = resolve_pipeline_state(rs);

    CHECK(ps.topology == 4,               "triangle-strip prim_type 6 -> VK topology 4");
    CHECK(ps.color0_format == 50,         "8_8_8_8 SRGB BGRA -> VK_FORMAT_B8G8R8A8_SRGB (50)");
    CHECK(ps.depth_test_enable,           "depth test enabled");
    CHECK(ps.depth_write_enable,          "depth write enabled");
    CHECK(ps.depth_compare_op == 4,       "zfunc GREATER -> VkCompareOp 4");
    CHECK(ps.blend_enable,                "blend enabled");
    CHECK(ps.src_color_blend_factor == 4, "src DstColor -> VK DST_COLOR (4)");
    CHECK(ps.dst_color_blend_factor == 7, "dst OneMinusSrcAlpha -> VK ONE_MINUS_SRC_ALPHA (7)");
    CHECK(ps.color_blend_op == 3,         "comb Min -> VK_BLEND_OP_MIN (3)");
    CHECK(ps.color_write_mask == 0x7,     "CB_TARGET_MASK MRT0 nibble -> RGB write mask");

    // A default/empty RenderState resolves to a safe pipeline (point list, undefined format, no blend).
    ResolvedPipelineState def = resolve_pipeline_state(RenderState{});
    CHECK(def.topology == 0 && def.color0_format == 0 && !def.blend_enable && !def.depth_test_enable,
          "empty render-state resolves to a safe default pipeline");
    CHECK(!def.has_viewport, "empty render-state has no guest viewport (backend keeps full-target default)");

    // GNM-style 1080p viewport: xscale=960 xoffset=960, yscale=-540 yoffset=540 (+Y-up NDC). Resolves to
    // the Vulkan FLIPPED viewport {x=0, y=1080, w=1920, h=-1080} (negative height = core-1.1 Y flip).
    RenderState vv;
    vv.vport_xscale = 960.0f;  vv.vport_xoffset = 960.0f;
    vv.vport_yscale = -540.0f; vv.vport_yoffset = 540.0f;
    vv.vport_zscale = 1.0f;    vv.vport_zoffset = 0.0f;
    ResolvedPipelineState vp = resolve_pipeline_state(vv);
    CHECK(vp.has_viewport, "programmed PA_CL_VPORT -> has_viewport");
    CHECK(vp.viewport_x == 0.0f && vp.viewport_w == 1920.0f, "xscale/xoffset 960/960 -> x=0 w=1920");
    CHECK(vp.viewport_y == 1080.0f && vp.viewport_h == -1080.0f,
          "negative yscale (-540, +Y-up NDC) -> flipped Vulkan viewport y=1080 h=-1080");
    CHECK(vp.min_depth == 0.0f && vp.max_depth == 1.0f, "zscale/zoffset 1/0 -> depth range 0..1");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
