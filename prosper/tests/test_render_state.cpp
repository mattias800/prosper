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
        // DB_DEPTH_CONTROL: Z_ENABLE(bit1)|Z_WRITE_ENABLE(bit2)|ZFUNC=4(bits[6:4]) = 0x46
        { P::DB_DEPTH_CONTROL,   0x00000046u },
        { P::CB_COLOR_CONTROL,   0x00CC0010u },
        // CB_BLEND0_CONTROL: ENABLE(bit30) | SRCBLEND=4/SrcAlpha | DESTBLEND=5/OneMinusSrcAlpha | COMB_FCN=0/Add
        { P::CB_BLEND0_CONTROL,  (1u << 30) | (4u << 0) | (5u << 8) | (0u << 5) },
        { P::CB_TARGET_MASK,     0x0000000Fu },
        // CB fast-clear (#309): CLEAR_WORD0 holds one texel in the surface's 8_8_8_8 format.
        { P::CB_COLOR0_CLEAR_WORD0, 0x11223344u },
        { P::CB_COLOR0_CLEAR_WORD1, 0x00000000u },
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

    CHECK(rs.color0_base == rdna2_addr(0x00100000u, 0x12u), "color0_base = (BASE<<8)|(EXT<<40)");
    CHECK(rs.color0_format == 0x0Au, "color0_format = 0x0A (FORMAT field)");
    CHECK(rs.color0_number_type == 6u, "color0_number_type = 6 (SRGB)");
    CHECK(rs.color0_comp_swap == 1u, "color0_comp_swap = 1 (BGRA)");
    CHECK(vk_color_format(rs.color0_format, rs.color0_number_type, rs.color0_comp_swap)
              == VkFormat::B8G8R8A8_SRGB, "color format -> VK B8G8R8A8_SRGB");
    CHECK(vk_color_format(0x0Au, 0u, 0u) == VkFormat::R8G8B8A8_UNORM, "0xA/UNORM/RGBA -> R8G8B8A8_UNORM");
    // #133: the non-8888 CB rows (standard GCN/RDNA format table; values == VkFormat enumerators).
    CHECK(vk_color_format(0xCu, 7u, 0u) == VkFormat::R16G16B16A16_SFLOAT, "0xC/FLOAT -> R16G16B16A16_SFLOAT (97)");
    CHECK(vk_color_format(0xCu, 0u, 0u) == VkFormat::R16G16B16A16_UNORM,  "0xC/UNORM -> R16G16B16A16_UNORM (91)");
    CHECK(vk_color_format(0x9u, 0u, 0u) == VkFormat::A2B10G10R10_UNORM_PACK32, "0x9/UNORM/STD -> A2B10G10R10 (64)");
    CHECK(vk_color_format(0x9u, 0u, 1u) == VkFormat::A2R10G10B10_UNORM_PACK32, "0x9/UNORM/ALT -> A2R10G10B10 (58)");
    CHECK(vk_color_format(0x7u, 7u, 0u) == VkFormat::B10G11R11_UFLOAT_PACK32,  "0x7/FLOAT -> B10G11R11 (122)");
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

    // Decoded blend state + RDNA2->Vulkan factor/op mapping.
    CHECK(rs.blend_enable, "blend_enable = true (bit 30)");
    CHECK(rs.color_src_blend == 4u && vk_blend_factor(rs.color_src_blend) == 6u,
          "src blend 4/SrcAlpha -> VK SRC_ALPHA(6)");
    CHECK(rs.color_dst_blend == 5u && vk_blend_factor(rs.color_dst_blend) == 7u,
          "dst blend 5/OneMinusSrcAlpha -> VK ONE_MINUS_SRC_ALPHA(7)");
    CHECK(rs.color_comb_fcn == 0u && vk_blend_op(rs.color_comb_fcn) == 0u, "comb_fcn 0/Add -> VK ADD(0)");
    CHECK(vk_blend_factor(0x08u) == 4u, "RDNA2 DstColor(8) -> VK DST_COLOR(4) (non-identity)");
    CHECK(vk_blend_op(2u) == 3u, "RDNA2 comb Min(2) -> VK MIN(3) (non-identity)");

    CHECK(rs.db_depth_control  == 0x00000046u, "db_depth_control raw preserved");
    CHECK(rs.cb_color_control  == 0x00CC0010u, "cb_color_control raw preserved");
    CHECK(rs.cb_target_mask    == 0x0000000Fu, "cb_target_mask raw preserved");

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
