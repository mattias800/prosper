// test_render_state — validates the render-state extractor (src/gpu/render_state.cpp) end-to-end:
// build a command buffer that sets real RDNA2 registers (shader program addresses, color target,
// primitive type, depth/blend state) via the AGC Dcb builders, replay it into a GpuState, then
// extract the semantic RenderState and assert every field (addresses use the RDNA2 <<8|<<40 rule).
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/command_processor.hpp"
#include "../src/gpu/render_state.hpp"
#include "../src/gpu/vk_translate.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper;
using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)
#define CHECK_NEAR(a, b, m) CHECK(std::fabs((a) - (b)) < 1e-4f, m)

// Mirror of render_state.cpp's sRGB->linear (VkClearColorValue floats are linear for sRGB targets).
static float srgb2lin(float c) {
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

struct Dcb { uint32_t* bottom; uint32_t* top; uint32_t* cursor_up; uint32_t* cursor_down;
             void* callback; void* user_data; uint32_t reserved_dw; uint32_t pad; };

static uint64_t rdna2_addr(uint32_t lo, uint32_t hi) {
    return ((uint64_t)lo << 8) | (((uint64_t)hi & 0xff) << 40);
}

int main() {
    printf("== test_render_state ==\n");
    register_builtin_hle();
    auto setcx = Hle::lookup("ZvwO9euwYzc");   // SetCxRegistersIndirect
    auto setsh = Hle::lookup("-HOOCn0JY48");   // SetShRegistersIndirect
    auto setuc = Hle::lookup("hvUfkUIQcOE");   // SetUcRegistersIndirect (VGT_PRIMITIVE_TYPE is uconfig)
    CHECK(setcx && setsh && setuc, "AGC Dcb builders registered");
    if (!(setcx && setsh && setuc)) { printf("== FAIL ==\n"); return 1; }

    // Context registers: color target base(+ext), color info(format), prim type, depth/blend/mask.
    ShaderReg cx_regs[] = {
        { P::CB_COLOR0_BASE,     0x00100000u },
        { P::CB_COLOR0_BASE_EXT, 0x00000012u },
        // CB_COLOR0_INFO: FORMAT=0xA (bits[6:2]) | NUMBER_TYPE=6/SRGB (bits[10:8]) | COMP_SWAP=1/BGRA (bits[12:11])
        { P::CB_COLOR0_INFO,     (0x0Au << 2) | (6u << 8) | (1u << 11) },
        { P::CB_COLOR1_BASE,     0x00110000u },
        { P::CB_COLOR1_BASE_EXT, 0x00000012u },
        { P::CB_COLOR1_INFO,     (0x0Au << 2) },
        // DB_DEPTH_CONTROL: Z_ENABLE(bit1)|Z_WRITE_ENABLE(bit2)|ZFUNC=4(bits[6:4]) = 0x46
        { P::DB_DEPTH_CONTROL,   0x00000046u },
        { P::DB_RENDER_CONTROL,  0x00000002u }, // explicit stencil clear
        { P::DB_SHADER_CONTROL,  0x00000006u }, // shader exports stencil test/op values
        { P::DB_STENCIL_CLEAR,   0x00000003u },
        { P::DB_Z_READ_BASE,     0x00200000u }, { P::DB_Z_READ_BASE_HI, 0x00000021u },
        { P::DB_Z_WRITE_BASE,    0x00210000u }, { P::DB_Z_WRITE_BASE_HI, 0x00000021u },
        { P::DB_STENCIL_READ_BASE,  0x00220000u }, { P::DB_STENCIL_READ_BASE_HI,  0x00000021u },
        { P::DB_STENCIL_WRITE_BASE, 0x00230000u }, { P::DB_STENCIL_WRITE_BASE_HI, 0x00000021u },
        { P::DB_DEPTH_VIEW, 0x04002001u },
        { P::DB_RENDER_OVERRIDE, 0x00000011u }, { P::DB_RENDER_OVERRIDE2, 0x00000022u },
        { P::DB_HTILE_DATA_BASE, 0x00240000u }, { P::DB_HTILE_DATA_BASE_HI, 0x00000021u },
        { P::DB_DEPTH_SIZE_XY, 0x01230234u }, { P::DB_DFSM_CONTROL, 0x00000033u },
        { P::DB_DEPTH_INFO, 0x00000044u }, { P::DB_Z_INFO, 0x00000055u },
        { P::DB_STENCIL_INFO, 0x00000066u }, { P::DB_DEPTH_SIZE, 0x00000077u },
        { P::DB_DEPTH_SLICE, 0x00000088u }, { P::DB_HTILE_SURFACE, 0x00000099u },
        { P::DB_RMI_L2_CACHE_CONTROL, 0x000000AAu },
        { P::CB_COLOR_CONTROL,   0x00CC0010u },
        // CB_BLEND0_CONTROL: ENABLE(bit30) | SRCBLEND=4/SrcAlpha | DESTBLEND=5/OneMinusSrcAlpha | COMB_FCN=0/Add
        { P::CB_BLEND0_CONTROL,  (1u << 30) | (4u << 0) | (5u << 8) | (0u << 5) },
        { P::CB_BLEND1_CONTROL,  (1u << 30) | (1u << 0) | (5u << 8) | (0u << 5) },
        { P::CB_TARGET_MASK,     0x000000FFu },
        // CB fast-clear (#309): CLEAR_WORD0 holds one texel in the surface's 8_8_8_8 format.
        { P::CB_COLOR0_CLEAR_WORD0, 0x11223344u },
        { P::CB_COLOR0_CLEAR_WORD1, 0x00000000u },
        { P::CB_COLOR0_ATTRIB2, ((1024u - 1u) << 14) | (32u - 1u) },
        { P::CB_COLOR1_CLEAR_WORD0, 0xFF0000FFu },
        { P::CB_COLOR1_ATTRIB2, ((1024u - 1u) << 14) | (32u - 1u) },
        { P::SPI_PS_INPUT_CNTL_0, 0x00000401u }, // PS input 0 <- PARAM1, flat
        { P::SPI_PS_INPUT_CNTL_0 + 1u, 0x00000320u }, // PS input 1 <- DEFAULT_VAL 1111
        { P::SPI_PS_INPUT_ENA, 0x00000303u }, // perspective sample/center + float position X/Y
        { P::SPI_PS_INPUT_ADDR, 0x000003CFu }, // reserve disabled centroid/pull-model slots
        // #531: effective scissor is the intersection of screen/window/generic/viewport. Window and
        // viewport apply (-10,+5); generic disables the offset. Right/bottom are exclusive.
        { P::PA_SC_SCREEN_SCISSOR_TL, 0xFFFDFFFCu }, // (-4,-3), hardware clamps negative to zero
        { P::PA_SC_SCREEN_SCISSOR_BR, 0x005A0064u }, // (100,90)
        { P::PA_SC_WINDOW_OFFSET, 0x0005FFF6u },     // (-10,+5)
        { P::PA_SC_WINDOW_SCISSOR_TL, 0x000A0014u }, // (20,10), offset enabled -> (10,15)
        { P::PA_SC_WINDOW_SCISSOR_BR, 0x0050005Au }, // (90,80), offset -> (80,85)
        { P::PA_SC_GENERIC_SCISSOR_TL, 0x8012000Cu },// (12,18), offset disabled
        { P::PA_SC_GENERIC_SCISSOR_BR, 0x004B0046u },// (70,75)
        { P::PA_SC_VPORT_SCISSOR_0_TL, 0x000E000Fu },// (15,14), offset -> (5,19)
        { P::PA_SC_VPORT_SCISSOR_0_BR, 0x00460050u },// (80,70), offset -> (70,75)
        { P::PA_SC_MODE_CNTL_0, 0x00000002u },       // VPORT_SCISSOR_ENABLE
    };
    // Shader-stage program addresses.
    ShaderReg sh_regs[] = {
        { P::SPI_SHADER_PGM_LO_PS, 0x00ABCDEFu }, { P::SPI_SHADER_PGM_HI_PS, 0x00000012u },
        { P::SPI_SHADER_PGM_LO_ES, 0x00111111u }, { P::SPI_SHADER_PGM_HI_ES, 0x00000034u },
    };

    uint32_t buffer[256]; memset(buffer, 0, sizeof buffer);
    Dcb dcb{}; dcb.bottom = buffer; dcb.top = buffer + 256; dcb.cursor_up = buffer; dcb.cursor_down = buffer + 256;
    auto D = (uint64_t)(uintptr_t)&dcb;
    ShaderReg uc_regs[] = { { P::VGT_PRIMITIVE_TYPE, 0x00000004u } };   // PRIM_TYPE = 4 (uconfig register)
    setcx(D, (uint64_t)(uintptr_t)cx_regs, sizeof(cx_regs)/sizeof(cx_regs[0]), 0, 0, 0);
    setsh(D, (uint64_t)(uintptr_t)sh_regs, sizeof(sh_regs)/sizeof(sh_regs[0]), 0, 0, 0);
    setuc(D, (uint64_t)(uintptr_t)uc_regs, sizeof(uc_regs)/sizeof(uc_regs[0]), 0, 0, 0);

    GpuState st;
    run_command_buffer(buffer, 256, st);
    RenderState rs = extract_render_state(st);

    CHECK(rs.ps_addr == rdna2_addr(0x00ABCDEFu, 0x12u), "PS shader addr = (LO<<8)|(HI<<40)");
    CHECK(rs.es_addr == rdna2_addr(0x00111111u, 0x34u), "ES shader addr = (LO<<8)|(HI<<40)");
    CHECK(rs.gs_addr == 0 && rs.hs_addr == 0, "unset GS/HS shader addrs are 0");
    CHECK(rs.ps_input_cntl_valid_mask == 0x3u &&
          rs.ps_input_cntl[0] == 0x00000401u && rs.ps_input_cntl[1] == 0x00000320u,
          "programmed SPI_PS_INPUT_CNTL words and presence mask are retained");
    CHECK(rs.ps_input_ena == 0x00000303u && rs.ps_input_addr == 0x000003CFu,
          "SPI_PS_INPUT_ENA/ADDR system-value VGPR controls are retained");
    CHECK(rs.has_scissor && rs.scissor_left == 12 && rs.scissor_top == 19 &&
          rs.scissor_right == 70 && rs.scissor_bottom == 75,
          "screen/window/generic/enabled-viewport scissors intersect with per-rectangle offsets");
    {
        ResolvedPipelineState sc = resolve_pipeline_state(rs);
        CHECK(sc.has_scissor && sc.scissor_left == 12 && sc.scissor_top == 19 &&
              sc.scissor_right == 70 && sc.scissor_bottom == 75,
              "resolved pipeline retains the effective exclusive scissor rectangle");

        GpuState no_viewport_scissor = st;
        no_viewport_scissor.cx[P::PA_SC_MODE_CNTL_0] = 0;
        const RenderState no_vport = extract_render_state(no_viewport_scissor);
        CHECK(no_vport.scissor_left == 12 && no_vport.scissor_top == 18 &&
              no_vport.scissor_right == 70 && no_vport.scissor_bottom == 75,
              "disabled VPORT_SCISSOR_ENABLE excludes the viewport rectangle");

        CHECK(!extract_render_state(GpuState{}).has_scissor,
              "completely absent guest scissor state preserves the full-target backend default");

        RenderState empty{}; empty.has_scissor = true;
        empty.scissor_left = 10; empty.scissor_top = 20;
        empty.scissor_right = 5; empty.scissor_bottom = 15;
        const ResolvedPipelineState normalized = resolve_pipeline_state(empty);
        CHECK(normalized.scissor_right == 10 && normalized.scissor_bottom == 20,
              "empty guest intersections normalize to a safe zero-area rectangle");

        ResolvedPipelineState scaled{}; scaled.has_scissor = true;
        scaled.scissor_left = -3; scaled.scissor_top = 5;
        scaled.scissor_right = 7; scaled.scissor_bottom = 9;
        scale_resolved_render_area(scaled, 0.5f, 0.5f);
        CHECK(scaled.scissor_left == -2 && scaled.scissor_top == 2 &&
              scaled.scissor_right == 4 && scaled.scissor_bottom == 5,
              "reduced-resolution scissors round outward without losing guest coverage");
    }

    CHECK(rs.color0_base == rdna2_addr(0x00100000u, 0x12u), "color0_base = (BASE<<8)|(EXT<<40)");
    CHECK(rs.color0_format == 0x0Au, "color0_format = 0x0A (FORMAT field)");
    CHECK(rs.color0_number_type == 6u, "color0_number_type = 6 (SRGB)");
    CHECK(rs.color0_comp_swap == 1u, "color0_comp_swap = 1 (BGRA)");
    CHECK(rs.color0_has_extent && rs.color0_width == 1024 && rs.color0_height == 32,
          "CB_COLOR0_ATTRIB2 dimension-minus-one fields decode to 1024x32");
    CHECK(rs.color1_base == rdna2_addr(0x00110000u, 0x12u) &&
          rs.color1_format == 0x0Au && rs.color1_number_type == 0u &&
          rs.color1_comp_swap == 0u && rs.color1_has_extent &&
          rs.color1_width == 1024 && rs.color1_height == 32,
          "MRT1 base, format, and extent decode from the second color block");
    CHECK(vk_color_format(rs.color0_format, rs.color0_number_type, rs.color0_comp_swap)
              == VkFormat::B8G8R8A8_SRGB, "color format -> VK B8G8R8A8_SRGB");
    CHECK(vk_color_format(0x0Au, 0u, 0u) == VkFormat::R8G8B8A8_UNORM, "0xA/UNORM/RGBA -> R8G8B8A8_UNORM");
    // #133: the non-8888 CB rows (standard GCN/RDNA format table; values == VkFormat enumerators).
    CHECK(vk_color_format(0xCu, 7u, 0u) == VkFormat::R16G16B16A16_SFLOAT, "0xC/FLOAT -> R16G16B16A16_SFLOAT (97)");
    CHECK(vk_color_format(0xCu, 0u, 0u) == VkFormat::R16G16B16A16_UNORM,  "0xC/UNORM -> R16G16B16A16_UNORM (91)");
    CHECK(vk_color_format(0x9u, 0u, 0u) == VkFormat::A2B10G10R10_UNORM_PACK32, "0x9/UNORM/STD -> A2B10G10R10 (64)");
    CHECK(vk_color_format(0x9u, 0u, 1u) == VkFormat::A2R10G10B10_UNORM_PACK32, "0x9/UNORM/ALT -> A2R10G10B10 (58)");
    CHECK(vk_color_format(0x7u, 7u, 0u) == VkFormat::B10G11R11_UFLOAT_PACK32,  "0x7/FLOAT -> B10G11R11 (122)");
    // #465: the aliased packed codes 0x6 (COLOR_10_11_11) and 0x8 (COLOR_10_10_10_2) resolve to the same
    // formats as 0x7 / 0x9 (shadPS4 canonicalizes 0x7->0x6, 0x8->0x9). Previously they fell to Undefined.
    CHECK(vk_color_format(0x6u, 7u, 0u) == VkFormat::B10G11R11_UFLOAT_PACK32, "0x6 (alias of 0x7)/FLOAT -> B10G11R11");
    CHECK(vk_color_format(0x8u, 0u, 0u) == VkFormat::A2B10G10R10_UNORM_PACK32, "0x8 (alias of 0x9)/UNORM/STD -> A2B10G10R10");
    CHECK(vk_color_format(0x8u, 0u, 1u) == VkFormat::A2R10G10B10_UNORM_PACK32, "0x8 (alias of 0x9)/UNORM/ALT -> A2R10G10B10");
    CHECK(vk_color_format(0x10u, 0u, 0u) == VkFormat::R5G6B5_UNORM_PACK16, "0x10/UNORM -> R5G6B5 (4)");
    CHECK(vk_color_format(0x1u, 0u, 0u) == VkFormat::R8_UNORM,   "0x1/UNORM -> R8_UNORM (9, Kyty's R8Unorm row)");
    CHECK(vk_color_format(0x4u, 7u, 0u) == VkFormat::R32_SFLOAT, "0x4/FLOAT -> R32_SFLOAT (100)");
    CHECK(vk_color_format(0x5u, 7u, 0u) == VkFormat::R16G16_SFLOAT, "0x5/FLOAT -> R16G16_SFLOAT (83)");
    CHECK(vk_color_format(0xAu, 4u, 1u) == VkFormat::B8G8R8A8_UINT, "0xA/UINT/ALT -> B8G8R8A8_UINT (48)");
    // Still-unmapped triples resolve to Undefined (logged once), never a bogus format.
    CHECK(vk_color_format(0x16u, 7u, 0u) == VkFormat::Undefined, "COLOR_X24_8_32 stays Undefined");
    CHECK(vk_color_format(0xCu, 7u, 2u)  == VkFormat::Undefined, "REV comp_swap stays Undefined");
    CHECK(rs.prim_type == 0x04u, "prim_type = 4");
    CHECK(vk_topology(rs.prim_type) == VkTopology::TriangleList, "prim_type 4 -> VK TriangleList");
    CHECK(vk_topology(6) == VkTopology::TriangleStrip && vk_topology(1) == VkTopology::PointList,
          "topology map: 6 -> TriangleStrip, 1 -> PointList");
    CHECK(vk_topology(7) == VkTopology::TriangleStrip,
          "PS5 primitive type 7 translates RectList to a Vulkan triangle strip");
    // #384: RectList/QuadList are the screen-space primitives full-screen passes emit; they must not
    // collapse to PointList (3 stray points instead of a covered target). Kyty's strip/fan approximation.
    CHECK(vk_topology(17) == VkTopology::TriangleStrip, "prim_type 17 (RectList) -> TriangleStrip");
    CHECK(vk_topology(19) == VkTopology::TriangleFan,   "prim_type 19 (QuadList) -> TriangleFan");

    // Decoded DB_DEPTH_CONTROL fields (value 0x46).
    CHECK(rs.z_enable, "z_enable = true (bit 1)");
    CHECK(rs.z_write_enable, "z_write_enable = true (bit 2)");
    CHECK(!rs.stencil_enable, "stencil_enable = false (bit 0)");
    CHECK(rs.zfunc == 4u, "zfunc = 4 (bits [6:4])");
    CHECK(vk_compare_op(rs.zfunc) == 4u, "zfunc 4 -> VkCompareOp GREATER (1:1)");
    CHECK(!rs.depth_clear_enable && rs.stencil_clear_enable && rs.db_render_control == 2u,
          "DB_RENDER_CONTROL clear intent is extracted independently");
    CHECK(rs.has_stencil_clear && rs.stencil_clear_value == 3u,
          "programmed DB_STENCIL_CLEAR is distinguished from absent/default zero");
    CHECK(rs.stencil_test_val_export_enable && rs.stencil_op_val_export_enable &&
          rs.db_shader_control == 6u, "DB_SHADER_CONTROL stencil-export intent is extracted");
    CHECK(rs.depth_read_base == rdna2_addr(0x00200000u, 0x21u) &&
          rs.depth_write_base == rdna2_addr(0x00210000u, 0x21u) &&
          rs.stencil_read_base == rdna2_addr(0x00220000u, 0x21u) &&
          rs.stencil_write_base == rdna2_addr(0x00230000u, 0x21u),
          "guest depth/stencil read/write surface identities are extracted");
    CHECK(rs.db_depth_view == 0x04002001u && rs.htile_data_base == rdna2_addr(0x00240000u, 0x21u) &&
          rs.db_depth_size_xy == 0x01230234u && rs.db_z_info == 0x55u &&
          rs.db_stencil_info == 0x66u && rs.db_htile_surface == 0x99u,
          "depth view, HTILE, format, and extent programming is retained");

    // #371: depth clear value. The sample stream programs no DB_DEPTH_CLEAR, so resolve defaults it by
    // the compare op — 0.0 for GREATER (this stream), 1.0 for LESS — never a fixed 0.5. A programmed
    // DB_DEPTH_CLEAR is used verbatim.
    CHECK(!rs.has_depth_clear, "no DB_DEPTH_CLEAR programmed in the sample stream");
    {
        ResolvedPipelineState pd = resolve_pipeline_state(rs);
        CHECK(pd.depth_compare_op == 4u && pd.depth_clear_value == 0.0f,
              "no DB_DEPTH_CLEAR + GREATER -> depth clears to 0.0 (reversed-Z near), not 0.5");
        CHECK(pd.stencil_clear_enable && pd.stencil_clear_value == 3u &&
              pd.stencil_write_base == rs.stencil_write_base,
              "resolved state preserves stencil clear intent/value/surface identity");
        CHECK(pd.stencil_test_val_export_enable && pd.stencil_op_val_export_enable,
              "resolved state preserves fragment shader stencil-export intent");
        CHECK(pd.db_depth_view == rs.db_depth_view && pd.htile_data_base == rs.htile_data_base &&
              pd.db_depth_info == rs.db_depth_info && pd.db_depth_size == rs.db_depth_size &&
              pd.db_depth_slice == rs.db_depth_slice &&
              pd.db_rmi_l2_cache_control == rs.db_rmi_l2_cache_control,
              "resolved state preserves complete depth-surface programming");
        RenderState less_rs = rs; less_rs.zfunc = 1u;   // LESS
        CHECK(resolve_pipeline_state(less_rs).depth_clear_value == 1.0f,
              "no DB_DEPTH_CLEAR + LESS -> depth clears to 1.0 (far), not 0.5");
        RenderState prog = rs; prog.has_depth_clear = true; prog.depth_clear_value = 0.25f;
        prog.stencil_clear_value = 7u;
        ResolvedPipelineState pp = resolve_pipeline_state(prog);
        CHECK(pp.depth_clear_value == 0.25f && pp.stencil_clear_value == 7u,
              "programmed DB_DEPTH_CLEAR / DB_STENCIL_CLEAR used verbatim");
    }

    // #456: PA_SU_SC_MODE_CNTL -> cull/front-face/polygon-mode. Absent (the sample stream) -> the prior
    // hardcode: CULL_NONE(0) / CCW-front(0) / FILL(0). Programmed fields resolve to the Vk enumerators.
    CHECK(rs.pa_su_sc_mode_cntl == 0u, "no PA_SU_SC_MODE_CNTL in the sample stream");
    {
        ResolvedPipelineState pd = resolve_pipeline_state(rs);
        CHECK(pd.cull_mode == 0u && pd.front_face == 0u && pd.polygon_mode == 0u,
              "absent PA_SU_SC_MODE_CNTL -> CULL_NONE + CCW + FILL (prior default preserved)");
        RenderState r2 = rs;
        r2.pa_su_sc_mode_cntl = (1u << 1);   // CULL_BACK
        CHECK(resolve_pipeline_state(r2).cull_mode == 2u, "CULL_BACK -> VkCullMode BACK(2)");
        r2.pa_su_sc_mode_cntl = (1u << 0) | (1u << 1);   // CULL_FRONT | CULL_BACK
        CHECK(resolve_pipeline_state(r2).cull_mode == 3u, "CULL_FRONT|CULL_BACK -> FRONT_AND_BACK(3)");
        r2.pa_su_sc_mode_cntl = (1u << 2);   // FACE = 1 (CW is front)
        CHECK(resolve_pipeline_state(r2).front_face == 1u, "FACE=1 -> VkFrontFace CLOCKWISE(1)");
        r2.pa_su_sc_mode_cntl = (1u << 3) | (1u << 5);   // POLY_MODE enabled + FRONT_PTYPE=1 (lines)
        CHECK(resolve_pipeline_state(r2).polygon_mode == 1u, "POLY_MODE lines -> VK_POLYGON_MODE_LINE(1)");
    }

    // #466: the three "replace-ish" stencil ops (ONES=2, REPLACE_TEST=3, REPLACE_OP=4) all resolve to
    // VK REPLACE but write DIFFERENT values via the backend's stencil_op_val reference. Build a stencil
    // state (ALWAYS compare, one pass op) and check the resolved op-val picks the right source.
    {
        RenderState s{};
        s.stencil_enable = true;
        s.db_depth_control = (1u << 0) | (7u << 8);   // STENCIL_ENABLE | STENCILFUNC=ALWAYS(7), BACKFACE_ENABLE=0
        s.db_stencilrefmask = (0x33u << 24) | 0x22u;  // STENCILOPVAL=0x33, STENCILTESTVAL=0x22
        s.db_stencil_control = (2u << 4);             // STENCILZPASS = ONES(2)
        CHECK(resolve_pipeline_state(s).stencil_op_val[0] == 0xFFu,
              "#466: ONES stencil op writes the constant 0xFF (not STENCILOPVAL)");
        s.db_stencil_control = (3u << 4);             // STENCILZPASS = REPLACE_TEST(3)
        CHECK(resolve_pipeline_state(s).stencil_op_val[0] == 0x22u,
              "#466: REPLACE_TEST writes STENCILTESTVAL (0x22)");
        s.db_stencil_control = (4u << 4);             // STENCILZPASS = REPLACE_OP(4)
        CHECK(resolve_pipeline_state(s).stencil_op_val[0] == 0x33u,
              "#466: REPLACE_OP writes STENCILOPVAL (0x33), unchanged");
        // Back face mirrors front (BACKFACE_ENABLE=0) — ONES 0xFF propagates to the back op-val too.
        s.db_stencil_control = (2u << 4);
        CHECK(resolve_pipeline_state(s).stencil_op_val[1] == 0xFFu,
              "#466: mirrored back face inherits the ONES 0xFF write value");
    }

    // #520: color-disabled draws are retained only when they change depth/stencil state.
    {
        ResolvedPipelineState p{}; p.color_write_mask = 0;
        CHECK(!has_depth_stencil_side_effect(p), "color-disabled state with no DS writes is a true no-op");
        p.depth_write_enable = true;
        CHECK(has_depth_stencil_side_effect(p), "color-disabled depth write must execute");
        p.depth_write_enable = false; p.stencil_enable = true; p.stencil_pass_op[0] = 2;
        CHECK(has_depth_stencil_side_effect(p), "color-disabled stencil REPLACE must execute");
        p.stencil_pass_op[0] = 0;
        CHECK(!has_depth_stencil_side_effect(p), "KEEP-only stencil test has no attachment side effect");
        p.stencil_clear_enable = true;
        CHECK(has_depth_stencil_side_effect(p), "explicit stencil clear must execute without color writes");
    }

    // Decoded blend state + RDNA2->Vulkan factor/op mapping.
    CHECK(rs.blend_enable, "blend_enable = true (bit 30)");
    CHECK(rs.color_src_blend == 4u && vk_blend_factor(rs.color_src_blend) == 6u,
          "src blend 4/SrcAlpha -> VK SRC_ALPHA(6)");
    CHECK(rs.color_dst_blend == 5u && vk_blend_factor(rs.color_dst_blend) == 7u,
          "dst blend 5/OneMinusSrcAlpha -> VK ONE_MINUS_SRC_ALPHA(7)");
    CHECK(rs.color_comb_fcn == 0u && vk_blend_op(rs.color_comb_fcn) == 0u, "comb_fcn 0/Add -> VK ADD(0)");
    CHECK(rs.blend1_enable && rs.color1_src_blend == 1u && rs.color1_dst_blend == 5u,
          "CB_BLEND1_CONTROL decodes independently from MRT0");
    CHECK(vk_blend_factor(0x08u) == 4u, "RDNA2 DstColor(8) -> VK DST_COLOR(4) (non-identity)");
    CHECK(vk_blend_op(2u) == 3u, "RDNA2 comb Min(2) -> VK MIN(3) (non-identity)");

    // #381: separate alpha blend. The extractor's rs (bit 29 clear) mirrors color into alpha on resolve;
    // with SEPARATE_ALPHA_BLEND set, the alpha channel resolves from its OWN factors, independent of color.
    CHECK(!rs.separate_alpha_blend, "SEPARATE_ALPHA_BLEND not set in the sample stream");
    {
        ResolvedPipelineState mirror = resolve_pipeline_state(rs);
        CHECK(mirror.src_alpha_blend_factor == mirror.src_color_blend_factor &&
              mirror.dst_alpha_blend_factor == mirror.dst_color_blend_factor &&
              mirror.alpha_blend_op == mirror.color_blend_op,
              "no SEPARATE_ALPHA_BLEND -> alpha factors mirror color");
        RenderState sep = rs;
        sep.separate_alpha_blend = true;
        sep.alpha_src_blend = 1u;  // One
        sep.alpha_dst_blend = 5u;  // OneMinusSrcAlpha
        sep.alpha_comb_fcn  = 0u;  // Add
        ResolvedPipelineState psep = resolve_pipeline_state(sep);
        CHECK(psep.src_alpha_blend_factor == vk_blend_factor(1u) &&
              psep.dst_alpha_blend_factor == vk_blend_factor(5u) &&
              psep.src_alpha_blend_factor != psep.src_color_blend_factor,
              "SEPARATE_ALPHA_BLEND -> alpha uses its own factors (One/OneMinusSrcAlpha), not color's");
    }

    CHECK(rs.db_depth_control  == 0x00000046u, "db_depth_control raw preserved");
    CHECK(rs.cb_color_control  == 0x00CC0010u, "cb_color_control raw preserved");
    CHECK(rs.cb_target_mask    == 0x000000FFu, "cb_target_mask raw preserved");

    // #309: CB fast-clear word extraction + format-aware decode in resolve_pipeline_state.
    CHECK(rs.color0_has_clear, "color0_has_clear = true (CLEAR_WORD programmed)");
    CHECK(rs.color0_clear_word0 == 0x11223344u, "color0_clear_word0 preserved");
    {
        // This target is ALT/BGRA + SRGB, so byte0=B, byte1=G, byte2=R, byte3=A, with RGB linearized.
        ResolvedPipelineState ps = resolve_pipeline_state(rs);
        CHECK(ps.has_clear_color, "resolve decodes the fast-clear color");
        CHECK_NEAR(ps.clear_color[0], srgb2lin(0x22 / 255.0f), "clear R = byte2 (ALT swap), sRGB-linearized");
        CHECK_NEAR(ps.clear_color[1], srgb2lin(0x33 / 255.0f), "clear G = byte1, sRGB-linearized");
        CHECK_NEAR(ps.clear_color[2], srgb2lin(0x44 / 255.0f), "clear B = byte0 (ALT swap), sRGB-linearized");
        CHECK_NEAR(ps.clear_color[3], 0x11 / 255.0f,           "clear A = byte3, linear (no sRGB on alpha)");
        CHECK(ps.has_clear_color1 && ps.color1_write_mask == 0xFu && ps.blend1_enable,
              "MRT1 clear, write mask, and blend state resolve independently");
        CHECK_NEAR(ps.clear_color1[0], 1.0f, "MRT1 clear R decodes from its own clear word");
        CHECK_NEAR(ps.clear_color1[1], 0.0f, "MRT1 clear G decodes from its own clear word");
        CHECK_NEAR(ps.clear_color1[2], 0.0f, "MRT1 clear B decodes from its own clear word");
        CHECK_NEAR(ps.clear_color1[3], 1.0f, "MRT1 clear A decodes from its own clear word");
    }
    {
        // STD/RGBA + UNORM: byte0=R, byte1=G, byte2=B, byte3=A, no sRGB — exact byte/255.
        RenderState r2{};
        r2.color0_format = 0xAu; r2.color0_number_type = 0u; r2.color0_comp_swap = 0u;
        r2.color0_has_clear = true; r2.color0_clear_word0 = 0x11223344u;
        ResolvedPipelineState p2 = resolve_pipeline_state(r2);
        CHECK(p2.has_clear_color, "STD/UNORM fast-clear decodes");
        CHECK_NEAR(p2.clear_color[0], 0x44 / 255.0f, "STD clear R = byte0");
        CHECK_NEAR(p2.clear_color[1], 0x33 / 255.0f, "STD clear G = byte1");
        CHECK_NEAR(p2.clear_color[2], 0x22 / 255.0f, "STD clear B = byte2");
        CHECK_NEAR(p2.clear_color[3], 0x11 / 255.0f, "STD clear A = byte3");
    }
    {
        // No fast-clear programmed -> has_clear_color false, default opaque black (never blue) (#309).
        RenderState r3{}; r3.color0_format = 0xAu; r3.color0_has_clear = false;
        ResolvedPipelineState p3 = resolve_pipeline_state(r3);
        CHECK(!p3.has_clear_color, "no fast-clear -> has_clear_color false");
        CHECK(p3.clear_color[0] == 0.0f && p3.clear_color[1] == 0.0f &&
              p3.clear_color[2] == 0.0f && p3.clear_color[3] == 1.0f,
              "no fast-clear -> default opaque-black clear color {0,0,0,1}");
    }
    {
        // Unmapped format with a clear word set -> undecoded, falls back to black (not a bogus color).
        RenderState r4{}; r4.color0_format = 0xCu; r4.color0_has_clear = true; r4.color0_clear_word0 = 0x11223344u;
        ResolvedPipelineState p4 = resolve_pipeline_state(r4);
        CHECK(!p4.has_clear_color, "unmapped format fast-clear -> has_clear_color false (black fallback)");
    }

    // #377: DB_DEPTH_CONTROL.BACKFACE_ENABLE=0 -> front stencil state applies to BOTH faces. Back must
    // NOT read the (unprogrammed) _BF regs, which would give STENCILFUNC_BF=0 -> NEVER and drop back faces.
    {
        RenderState s{};
        s.stencil_enable    = true;
        s.db_depth_control  = (1u << 0) | (4u << 8);   // STENCIL_ENABLE | STENCILFUNC=GREATER(4), BACKFACE_ENABLE=0
        s.db_stencil_control = 0x2;                     // some front op; _BF fields left 0
        s.db_stencilrefmask  = 0x1234;                 // front ref/mask; _BF left 0
        ResolvedPipelineState p = resolve_pipeline_state(s);
        CHECK(p.stencil_compare_op[0] == 4u, "front stencil func = GREATER(4)");
        CHECK(p.stencil_compare_op[1] == p.stencil_compare_op[0],
              "BACKFACE_ENABLE=0: back stencil func mirrors front (not NEVER from unprogrammed _BF)");
        CHECK(p.stencil_ref[1] == p.stencil_ref[0] && p.stencil_compare_mask[1] == p.stencil_compare_mask[0],
              "BACKFACE_ENABLE=0: back ref/mask mirror front");
        s.db_depth_control |= (1u << 7);               // BACKFACE_ENABLE=1 -> back independent (from _BF=0 -> NEVER)
        ResolvedPipelineState p2 = resolve_pipeline_state(s);
        CHECK(p2.stencil_compare_op[1] != p2.stencil_compare_op[0],
              "BACKFACE_ENABLE=1: back stencil is independent from front (_BF sourced)");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
