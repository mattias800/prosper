// test_interp_render — a recompiled VERTEX shader exports a per-vertex parameter (EXP PARAM0) and a
// recompiled PIXEL shader reads it back interpolated (v_interp_p1/p2), and the interpolated value
// varies across the surface in the framebuffer.
//
// This is the pixel-shader input path (VINTRP). The VS sets PARAM0.x = float(gl_VertexIndex), so the
// three fullscreen-triangle vertices carry 0, 1, 2; the rasterizer interpolates PARAM0 across the
// triangle; the PS reads attr0.x and writes it to the red channel. A working interpolation therefore
// produces a red GRADIENT (min != max across the viewport); a broken/flat path would be uniform. Green
// and blue stay ~0, proving the output is the interpolated attribute, not garbage.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "render_runner.h"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_interp_render ==\n");
    const uint32_t W = 64, H = 64;

    // VS: fullscreen triangle from gl_VertexIndex + EXP PARAM0 = (float(vid), 0, 0, 1).
    const uint32_t vs[] = {
        0x7e140d00u, 0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100202f6u, 0x100404f6u,
        0x060202f3u, 0x060404f3u, 0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xf800020fu,
        0x0403030au, 0xbf810000u,
    };
    // PS: v_interp attr0.x -> v0; exp mrt0 (v0, 0, 0, 1) -> red = interpolated attribute.
    const uint32_t ps[] = {
        0xc8000000u, 0xc8010001u, 0x7e020280u, 0x7e0402f2u, 0xf800080fu, 0x02010100u, 0xbf810000u,
    };

    std::vector<uint32_t> vert = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]));
    std::vector<uint32_t> frag = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
    CHECK(!vert.empty() && vert[0] == 0x07230203u, "recompiled VS with EXP PARAM0 -> SPIR-V");
    CHECK(!frag.empty() && frag[0] == 0x07230203u, "recompiled PS with v_interp -> SPIR-V");
    if (vert.empty() || frag.empty()) { printf("== FAIL ==\n"); return 1; }

    std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H);
    CHECK(px.size() == (size_t)W * H * 4, "pipeline linked VS PARAM0 -> PS interpolated input + rendered");
    if (px.size() != (size_t)W * H * 4) { printf("== FAIL: render failed ==\n"); return 1; }

    // Scan the viewport: the red channel must vary (gradient), green/blue must stay near 0.
    uint8_t rmin = 255, rmax = 0; uint32_t max_gb = 0;
    for (uint32_t y = 2; y < H - 2; y += 4) for (uint32_t x = 2; x < W - 2; x += 4) {
        const uint8_t* p = &px[((size_t)y * W + x) * 4];
        if (p[0] < rmin) rmin = p[0];
        if (p[0] > rmax) rmax = p[0];
        if (p[1] > max_gb) max_gb = p[1];
        if (p[2] > max_gb) max_gb = p[2];
    }
    printf("  red range [%u..%u]  max(green,blue)=%u\n", rmin, rmax, max_gb);
    CHECK(rmax - rmin > 40, "the interpolated attribute forms a red GRADIENT across the viewport");
    CHECK(max_gb < 0x20, "green/blue stay ~0 (output is the interpolated attribute, not garbage)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
