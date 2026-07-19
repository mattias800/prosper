// test_pipeline_state — resolve_pipeline_state() turns a RenderState's RDNA2 register semantics into
// Vulkan-ready pipeline state. Pure (no Vulkan), so it runs on every host. Inputs are chosen so the
// RDNA2->Vulkan mappings are NON-identity (e.g. RDNA2 triangle-strip=6 -> VK topology 4; DstColor
// blend=8 -> VK DST_COLOR=4; Min comb=2 -> VK MIN=3) — proving translation, not passthrough.
#include "../src/gpu/render_state.hpp"
#include "../src/gpu/vk_translate.hpp"
#include <array>
#include <cstdio>
#include <utility>

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

    // AMD PAL's 16 source/destination ROP3 truth tables map exactly to Vulkan's 16 logic ops,
    // but in a different order (notably AMD COPY=0xCC -> Vulkan COPY=3).
    constexpr std::array<std::pair<uint32_t, uint32_t>, 16> logic_ops = {{
        {0x00, 0}, {0x88, 1}, {0x44, 2}, {0xCC, 3},
        {0x22, 4}, {0xAA, 5}, {0x66, 6}, {0xEE, 7},
        {0x11, 8}, {0x99, 9}, {0x55, 10}, {0xDD, 11},
        {0x33, 12}, {0xBB, 13}, {0x77, 14}, {0xFF, 15},
    }};
    bool all_logic_ops_map = true;
    for (const auto& [rop3, expected] : logic_ops) {
        uint32_t actual = 99;
        all_logic_ops_map &= vk_logic_op(rop3, actual) && actual == expected;
    }
    CHECK(all_logic_ops_map, "all 16 two-input ROP3 values map to VkLogicOp");
    uint32_t unsupported_logic_op = 99;
    CHECK(!vk_logic_op(0x5A, unsupported_logic_op) && unsupported_logic_op == 3,
          "three-input ROP3 falls back visibly to VK_LOGIC_OP_COPY");

    RenderState logic;
    logic.cb_target_mask = 0xF;
    logic.cb_color_control = (1u << 4) | (0x66u << 16); // MODE=NORMAL, ROP3=XOR
    ResolvedPipelineState xor_ps = resolve_pipeline_state(logic);
    CHECK(xor_ps.logic_op_enable && xor_ps.logic_op == 6,
          "CB_COLOR_CONTROL normal/XOR resolves to VK_LOGIC_OP_XOR");
    logic.cb_color_control = (1u << 4) | (0xCCu << 16); // COPY preserves ordinary blending
    ResolvedPipelineState copy_ps = resolve_pipeline_state(logic);
    CHECK(!copy_ps.logic_op_enable && copy_ps.logic_op == 3,
          "ROP3 COPY keeps Vulkan logic ops disabled");
    logic.cb_color_control = (1u << 4) | (0x66u << 16);
    logic.disable_rop3 = true;
    CHECK(!resolve_pipeline_state(logic).logic_op_enable,
          "CB_BLEND0_CONTROL.DISABLE_ROP3 suppresses the global operation");
    logic.disable_rop3 = false;
    logic.cb_color_control = (6u << 4) | (0x66u << 16); // DCC helper mode, not a color draw
    CHECK(!resolve_pipeline_state(logic).logic_op_enable,
          "non-normal CB mode does not become a Vulkan logic op");

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
