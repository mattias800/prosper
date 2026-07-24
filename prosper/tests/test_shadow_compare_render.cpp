// test_shadow_compare_render — IMAGE_SAMPLE_C_LZ on a plain 2D shadow map (#1271): a recompiled
// pixel shader interpolates UVs, moves a DREF constant into the vaddr slot BEFORE them (ISA 8.2.5
// "{z-compare}{body}" order), issues image_sample_c_lz dim:2D dmask:0x1, and exports the compare
// result as MRT0.R. The word0 encoding (0xf0bc0108) is byte-identical to Blue Prince's live
// packets from the #1271 reject log — this test fails (recompile reject) without the 2D dref
// lowering. The manual-compare semantics are asserted end-to-end: with compare func GREATER and
// dref 0.5, texels darker than 0.5 pass (1.0) and brighter texels fail (0.0).
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
    printf("== test_shadow_compare_render ==\n");
    const uint32_t W = 128, H = 128;

    // VS: fullscreen triangle + PARAM0 = (u, v, 0, 1) — same as test_textured_interp_render.
    const uint32_t vs[] = {
        0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100602f6u, 0x100804f6u, 0x060606f3u,
        0x060808f3u, 0x100a02f4u, 0x100c04f4u, 0x7e0e0280u, 0x7e1002f2u, 0xf80008cfu, 0x08070403u,
        0xf800020fu, 0x08070605u, 0xbf810000u,
    };
    // PS: v2 = interp attr0.x (u), v3 = interp attr0.y (v); v1 = 0.5 (DREF);
    // image_sample_c_lz v4, v[1:3], s[8:15], s[16:19] dmask:0x1 dim:2D  (word0 0xf0bc0108 — the
    // exact op/dmask/dim word Blue Prince issues); exp mrt0 v4..v7.
    const uint32_t ps[] = {
        0xc8080000u, 0xc8090001u, 0xc80c0100u, 0xc80d0101u,   // v2 = u, v3 = v
        0x7e0202f0u,                                          // v_mov_b32 v1, 0.5
        0xf0bc0108u, 0x00820401u,                             // image_sample_c_lz v4, v[1:3], s8, s16
        0xf800080fu, 0x07060504u,                             // exp mrt0 v4..v7
        0xbf810000u,                                          // s_endpgm
    };

    std::vector<uint32_t> vert = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]));
    ShaderResourceTable rt;
    { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 1;
      t.width = 8; t.height = 8; t.sgpr_base = 8;
      t.depth_compare = true; t.depth_compare_func = 4;   // GREATER: pass = dref > stored
      rt.resources.push_back(t); }
    std::vector<uint32_t> frag = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]), &rt);
    CHECK(!vert.empty() && vert[0] == 0x07230203u, "recompiled VS -> SPIR-V");
    CHECK(!frag.empty() && frag[0] == 0x07230203u,
          "recompiled PS with image_sample_c_lz dim:2D -> SPIR-V (rejects without the #1271 lowering)");
    if (vert.empty() || frag.empty()) { printf("== FAIL ==\n"); return 1; }

    // 8x8 "shadow map" sampled as a color texture: left half R=0 (depth 0.0), right half R=230
    // (depth ~0.9). dref 0.5 with GREATER -> left passes (1.0 -> R=255), right fails (0.0 -> R=0).
    std::vector<uint8_t> tex(8 * 8 * 4);
    for (uint32_t y = 0; y < 8; y++) for (uint32_t x = 0; x < 8; x++) {
        uint8_t* t = &tex[(y * 8 + x) * 4];
        t[0] = x < 4 ? 0 : 230; t[1] = 0; t[2] = 0; t[3] = 255;
    }
    prosper::test::TexDesc td{ /*binding*/4, 8, 8, tex.data() };

    std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, nullptr, nullptr, nullptr, &td);
    CHECK(px.size() == (size_t)W * H * 4, "shadow-compare pipeline rendered a frame");
    if (px.size() != (size_t)W * H * 4) { printf("== FAIL: render failed ==\n"); return 1; }

    // Left half must be lit (compare passed -> R=255), right half shadow-failed (R=0). Sample
    // interior points away from the half boundary and the viewport edges.
    uint32_t left_pass = 0, left_total = 0, right_fail = 0, right_total = 0;
    for (uint32_t y = 16; y < H - 16; y += 8) {
        for (uint32_t x = 8; x < W / 2 - 12; x += 8) {
            const uint8_t* p = &px[((size_t)y * W + x) * 4]; left_total++;
            if (p[0] > 200) left_pass++;
        }
        for (uint32_t x = W / 2 + 12; x < W - 8; x += 8) {
            const uint8_t* p = &px[((size_t)y * W + x) * 4]; right_total++;
            if (p[0] < 50) right_fail++;
        }
    }
    printf("  left pass %u/%u  right fail %u/%u\n", left_pass, left_total, right_fail, right_total);
    CHECK(left_total && left_pass == left_total,
          "dref 0.5 GREATER passes (1.0) where stored depth is 0.0");
    CHECK(right_total && right_fail == right_total,
          "dref 0.5 GREATER fails (0.0) where stored depth is ~0.9");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
