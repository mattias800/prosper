// test_render_state — validates the render-state extractor (src/gpu/render_state.cpp) end-to-end:
// build a command buffer that sets real RDNA2 registers (shader program addresses, color target,
// primitive type, depth/blend state) via the AGC Dcb builders, replay it into a GpuState, then
// extract the semantic RenderState and assert every field (addresses use the RDNA2 <<8|<<40 rule).
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/command_processor.hpp"
#include "../src/gpu/render_state.hpp"
#include "../src/gpu/vk_translate.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper;
using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

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
    CHECK(rs.prim_type == 0x04u, "prim_type = 4");
    CHECK(vk_topology(rs.prim_type) == VkTopology::TriangleList, "prim_type 4 -> VK TriangleList");
    CHECK(vk_topology(6) == VkTopology::TriangleStrip && vk_topology(1) == VkTopology::PointList,
          "topology map: 6 -> TriangleStrip, 1 -> PointList");

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

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
