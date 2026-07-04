// test_texture_sample_render — a recompiled PIXEL shader that does MIMG image_sample reads a real
// bound texture and the sampled texel reaches the framebuffer.
//
// This is stage 4 of the resource-binding plan (textures). The pixel shader samples a 2x2 RGBA texture
// with four distinct texels; a NEAREST/normalized sampler maps coord 0.25 -> texel column/row 0 and
// coord 0.75 -> column/row 1. We render the fullscreen triangle twice with two coords and assert the
// framebuffer shows the corresponding texel color — proving MIMG decode, OpImageSampleImplicitLod,
// coordinate assembly (VADDR VGPRs -> u,v), the combined-image-sampler binding, and dmask->VDATA all
// work end to end. A broken path would leave the frame blue (clear) or the wrong texel.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "render_runner.h"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_texture_sample_render ==\n");
    const uint32_t W = 64, H = 64;

    // Fullscreen-triangle VS (from gl_VertexIndex; no resource table needed).
    const uint32_t vs[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
        0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    std::vector<uint32_t> vert = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]));
    CHECK(!vert.empty() && vert[0] == 0x07230203u, "recompiled fullscreen-triangle VS -> SPIR-V");

    // Pixel shader: u,v = literal coords; image_sample v[0:3], v[0:1], s[8:15], s[16:19] dim:2D dmask:0xf;
    // exp mrt0 v0..v3. The two literal dwords (indices 1 and 3) are the u,v coords — patched per render.
    const uint32_t ps_template[] = {
        0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u, 0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    // The T# is placed directly in user-data SGPRs (SRSRC base = s8) -> resolved via sgpr_base.
    ShaderResourceTable rt;
    { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 1; /*2D*/
      t.width = 2; t.height = 2; t.sgpr_base = 8; rt.resources.push_back(t); }

    // 2x2 RGBA8: (0,0)=red (1,0)=green (0,1)=blue (1,1)=white.
    const uint8_t texels[2*2*4] = {
        255,0,0,255,   0,255,0,255,
        0,0,255,255,   255,255,255,255,
    };
    prosper::test::TexDesc td{ /*binding*/4, /*w*/2, /*h*/2, texels };

    auto sample_center = [&](uint32_t u_bits, uint32_t v_bits, uint8_t out_rgb[3]) -> bool {
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = u_bits; ps[3] = v_bits;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        if (frag.empty() || frag[0] != 0x07230203u) return false;
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, nullptr, nullptr, nullptr, &td);
        if (px.size() != (size_t)W * H * 4) return false;
        const uint8_t* c = &px[((size_t)(H/2) * W + W/2) * 4];
        out_rgb[0] = c[0]; out_rgb[1] = c[1]; out_rgb[2] = c[2];
        return true;
    };

    // coord 0.25 -> texel index floor(0.25*2)=0; coord 0.75 -> floor(0.75*2)=1.
    const uint32_t C025 = 0x3e800000u /*0.25f*/, C075 = 0x3f400000u /*0.75f*/;
    uint8_t rgb[3];

    CHECK(recompile_fragment(ps_template, sizeof(ps_template)/sizeof(ps_template[0]), nullptr).empty(),
          "image_sample PS is rejected without a resource table");

    bool ok0 = sample_center(C025, C025, rgb);
    printf("  (0.25,0.25) center=(%u,%u,%u)\n", ok0?rgb[0]:0, ok0?rgb[1]:0, ok0?rgb[2]:0);
    CHECK(ok0 && rgb[0] > 0x80 && rgb[1] < 0x40 && rgb[2] < 0x40, "sampling texel (0,0) yields RED");

    bool ok1 = sample_center(C075, C025, rgb);
    printf("  (0.75,0.25) center=(%u,%u,%u)\n", ok1?rgb[0]:0, ok1?rgb[1]:0, ok1?rgb[2]:0);
    CHECK(ok1 && rgb[1] > 0x80 && rgb[0] < 0x40 && rgb[2] < 0x40, "sampling texel (1,0) yields GREEN (proves u routing)");

    bool ok2 = sample_center(C025, C075, rgb);
    printf("  (0.25,0.75) center=(%u,%u,%u)\n", ok2?rgb[0]:0, ok2?rgb[1]:0, ok2?rgb[2]:0);
    CHECK(ok2 && rgb[2] > 0x80 && rgb[0] < 0x40 && rgb[1] < 0x40, "sampling texel (0,1) yields BLUE (proves v routing)");

    // image_load (integer texel fetch, no sampler): x,y = inline-int coords into v0,v1; image_load
    // v[0:3], v[0:1], s[8:15]; exp mrt0. Coord (x,y) directly indexes the texel — no filtering.
    const uint32_t il_template[] = {
        0x7e000280u, 0x7e020280u, 0xf0000f08u, 0x00020000u, 0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    auto fetch_center = [&](uint32_t x, uint32_t y, uint8_t o[3]) -> bool {
        std::vector<uint32_t> ps(il_template, il_template + sizeof(il_template)/sizeof(il_template[0]));
        ps[0] = 0x7e000200u | (128u + x);   // v_mov_b32 v0, x   (inline int)
        ps[1] = 0x7e020200u | (128u + y);   // v_mov_b32 v1, y
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        if (frag.empty() || frag[0] != 0x07230203u) return false;
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, nullptr, nullptr, nullptr, &td);
        if (px.size() != (size_t)W * H * 4) return false;
        const uint8_t* c = &px[((size_t)(H/2) * W + W/2) * 4];
        o[0]=c[0]; o[1]=c[1]; o[2]=c[2]; return true;
    };
    bool okL0 = fetch_center(0, 0, rgb);
    printf("  image_load(0,0) center=(%u,%u,%u)\n", okL0?rgb[0]:0, okL0?rgb[1]:0, okL0?rgb[2]:0);
    CHECK(okL0 && rgb[0] > 0x80 && rgb[1] < 0x40 && rgb[2] < 0x40, "image_load texel (0,0) yields RED");
    bool okL1 = fetch_center(1, 1, rgb);
    printf("  image_load(1,1) center=(%u,%u,%u)\n", okL1?rgb[0]:0, okL1?rgb[1]:0, okL1?rgb[2]:0);
    CHECK(okL1 && rgb[0] > 0x80 && rgb[1] > 0x80 && rgb[2] > 0x80, "image_load texel (1,1) yields WHITE (proves integer coords)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
