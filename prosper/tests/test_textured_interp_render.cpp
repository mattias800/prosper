// test_textured_interp_render — the full textured-draw path in one frame: a recompiled VERTEX shader
// exports per-vertex UVs (EXP PARAM0), a recompiled PIXEL shader interpolates them (v_interp) and
// samples a texture at the interpolated UV (image_sample), and the mapped texture reaches the
// framebuffer. This exercises VS param export + rasterizer interpolation + MIMG sampling together —
// the shape of a real Unity textured draw. Also writes a PPM screenshot for human inspection.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "render_runner.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_textured_interp_render ==\n");
    const uint32_t W = 128, H = 128;

    // VS: fullscreen triangle + PARAM0 = (u, v, 0, 1) with u=(vid&1)*2, v=(vid>>1)*2 (so UV spans [0,1]
    // across the visible viewport).
    const uint32_t vs[] = {
        0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100602f6u, 0x100804f6u, 0x060606f3u,
        0x060808f3u, 0x100a02f4u, 0x100c04f4u, 0x7e0e0280u, 0x7e1002f2u, 0xf80008cfu, 0x08070403u,
        0xf800020fu, 0x08070605u, 0xbf810000u,
    };
    // PS: interpolate attr0.x/.y -> (u,v); image_sample tex at (u,v); exp mrt0 RGBA.
    const uint32_t ps[] = {
        0xc8080000u, 0xc8090001u, 0xc80c0100u, 0xc80d0101u, 0xf0800f08u, 0x00820402u,
        0xf800080fu, 0x07060504u, 0xbf810000u,
    };

    std::vector<uint32_t> vert = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]));
    ShaderResourceTable rt;
    { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 1;
      t.width = 8; t.height = 8; t.sgpr_base = 8; rt.resources.push_back(t); }
    std::vector<uint32_t> frag = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]), &rt);
    CHECK(!vert.empty() && vert[0] == 0x07230203u, "recompiled VS (position + UV param export) -> SPIR-V");
    CHECK(!frag.empty() && frag[0] == 0x07230203u, "recompiled PS (interp UV + image_sample) -> SPIR-V");
    if (vert.empty() || frag.empty()) { printf("== FAIL ==\n"); return 1; }

    // 8x8 texture: a 2-axis gradient (R rises with u, G rises with v, B constant 64) — clearly mapped
    // and distinguishable from the blue clear (whose B=255).
    std::vector<uint8_t> tex(8 * 8 * 4);
    for (uint32_t y = 0; y < 8; y++) for (uint32_t x = 0; x < 8; x++) {
        uint8_t* t = &tex[(y * 8 + x) * 4];
        t[0] = (uint8_t)(x * 36); t[1] = (uint8_t)(y * 36); t[2] = 64; t[3] = 255;
    }
    prosper::test::TexDesc td{ /*binding*/4, 8, 8, tex.data() };

    std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, nullptr, nullptr, nullptr, &td);
    CHECK(px.size() == (size_t)W * H * 4, "textured+interpolated pipeline rendered a frame");
    if (px.size() != (size_t)W * H * 4) { printf("== FAIL: render failed ==\n"); return 1; }

    // Sample a grid: the frame must be a 2-axis gradient (R and G both vary), and NOT the blue clear
    // (interior B should be ~64 from the texture, not 255).
    uint8_t rmin=255,rmax=0,gmin=255,gmax=0; uint32_t interior_blue_max=0, textured=0, total=0;
    for (uint32_t y=8; y<H-8; y+=8) for (uint32_t x=8; x<W-8; x+=8) {
        const uint8_t* p=&px[((size_t)y*W+x)*4]; total++;
        if (p[2] < 200) { textured++;   // not the blue clear -> a textured fragment
            if (p[0]<rmin)rmin=p[0]; if (p[0]>rmax)rmax=p[0];
            if (p[1]<gmin)gmin=p[1]; if (p[1]>gmax)gmax=p[1];
            if (p[2]>interior_blue_max) interior_blue_max=p[2];
        }
    }
    printf("  textured samples %u/%u  R[%u..%u] G[%u..%u] texB<=%u\n",
           textured, total, rmin, rmax, gmin, gmax, interior_blue_max);
    CHECK(textured > total/2, "the triangle covers the viewport with textured (non-clear) fragments");
    CHECK(rmax - rmin > 40 && gmax - gmin > 40, "both texture axes map across the surface (R and G gradients)");

    // Screenshot: honor PROSPER_SHOTS=<dir> if set, else the current directory. BMP is Windows-native.
    { const char* dir = getenv("PROSPER_SHOTS");
      char path[1024]; snprintf(path, sizeof(path), "%s/textured_interp.bmp", dir ? dir : ".");
      if (prosper::test::dump_bmp(path, px, W, H)) printf("  wrote screenshot: %s (%ux%u)\n", path, W, H); }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
