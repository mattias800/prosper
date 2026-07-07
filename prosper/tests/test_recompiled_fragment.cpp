// test_recompiled_fragment — render a triangle whose PIXEL SHADER is recompiled from RDNA2.
// The vertex shader is the placeholder (positions), but the fragment shader is real RDNA2 (assembled
// by llvm-mc) recompiled to SPIR-V by recompile_fragment: it exports green via EXP MRT0. We render
// and assert the triangle is GREEN (not the placeholder's red), proving RDNA2->SPIR-V works for an
// actual graphics-stage shader wired into a real pipeline.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "render_runner.h"
#include "spirv_triangle.h"     // kTriVertSpv: placeholder vertex shader (positions)
#include <cstdio>
#include <cstdint>
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

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
