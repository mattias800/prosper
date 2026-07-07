// test_multidraw_render — prove render_frame_rgba accumulates MANY draws into ONE framebuffer.
//
// A real frame is built from many draws into one target, not one standalone primitive. This renders
// two passes into a single framebuffer:
//   pass A: a RED triangle, opaque (no blend) -> writes red where it covers.
//   pass B: a GREEN triangle, ADDITIVE blend (src=ONE, dst=ONE, ADD) over the SAME geometry -> the
//           covered pixels become red+green = YELLOW.
// If only one draw executed (the old single-draw executor), the center would be red OR green, never
// yellow. Yellow proves BOTH draws ran, in order, into the same accumulating framebuffer.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "render_runner.h"
#include "spirv_triangle.h"     // kTriVertSpv: placeholder vertex shader (positions a centered triangle)
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_multidraw_render ==\n");
    const uint32_t W = 64, H = 64;

    // RDNA2 solid-color pixel shaders (gfx1030): v0..v3 = R,G,B,A; exp mrt0; endpgm. (0xF2=1.0, 0x80=0.0)
    const uint32_t red_ps[]   = { 0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };
    const uint32_t green_ps[] = { 0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };
    std::vector<uint32_t> red   = recompile_fragment(red_ps,   sizeof(red_ps)/4);
    std::vector<uint32_t> green = recompile_fragment(green_ps, sizeof(green_ps)/4);
    CHECK(!red.empty() && !green.empty(), "recompiled red + green pixel shaders");
    if (red.empty() || green.empty()) { printf("== FAIL ==\n"); return 1; }

    std::vector<uint32_t> vert(kTriVertSpv, kTriVertSpv + sizeof(kTriVertSpv)/sizeof(kTriVertSpv[0]));

    // pass A: opaque red (default state). pass B: additive green blend into the same framebuffer.
    ResolvedPipelineState add{};
    add.topology = 3 /*TRIANGLE_LIST*/; add.color_write_mask = 0xF;
    add.blend_enable = true;
    add.src_color_blend_factor = 1 /*VK_BLEND_FACTOR_ONE*/;
    add.dst_color_blend_factor = 1 /*VK_BLEND_FACTOR_ONE*/;
    add.color_blend_op = 0 /*VK_BLEND_OP_ADD*/;

    std::vector<prosper::test::FramePass> passes;
    { prosper::test::FramePass a; a.vs = &vert; a.fs = &red;   a.ps = nullptr; a.vcount = 3; passes.push_back(a); }
    { prosper::test::FramePass b; b.vs = &vert; b.fs = &green; b.ps = &add;    b.vcount = 3; passes.push_back(b); }

    // clear to opaque black so the covered region is purely the sum of the two draws.
    std::vector<uint8_t> px = prosper::test::render_frame_rgba(passes, W, H, VkClearColorValue{{0.0f, 0.0f, 0.0f, 1.0f}});
    CHECK(px.size() == (size_t)W * H * 4, "multi-draw frame rendered (both pipelines accepted)");
    if (px.size() != (size_t)W * H * 4) { printf("== FAIL: render failed ==\n"); return 1; }

    auto at = [&](uint32_t x, uint32_t y) { return &px[((size_t)y * W + x) * 4]; };
    const uint8_t* c = at(W/2, H/2);   // both triangles cover the center
    const uint8_t* k = at(0, 0);       // corner: neither covers -> clear (black)
    printf("  center=(%u,%u,%u,%u) corner=(%u,%u,%u,%u)\n", c[0],c[1],c[2],c[3], k[0],k[1],k[2],k[3]);
    CHECK(c[0] > 0x80 && c[1] > 0x80 && c[2] < 0x40,
          "center is YELLOW = red(pass A) + green(pass B): both draws accumulated into one framebuffer");
    CHECK(k[0] < 0x20 && k[1] < 0x20 && k[2] < 0x20, "corner is the black clear (neither draw covers it)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
