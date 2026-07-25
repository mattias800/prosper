// test_texture_mip_render — generated mip chains bounded by the T#'s declared levels (#1272): a
// pixel shader samples at an explicit LOD (image_sample_l), and a 4x4 texture whose columns
// alternate red/blue is uploaded with declared_mip_levels=3. The box-filtered level 2 must sample
// as purple (the red/blue average) — impossible from any level-0 texel — while the default
// declared_mip_levels=1 keeps the historical single-level behavior (LOD clamps to 0) and level 0
// stays byte-intact through the chain upload. Fails without the mip-chain generation.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "render_runner.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_texture_mip_render ==\n");
    const uint32_t W = 64, H = 64;

    // Fullscreen-triangle VS (same words as test_texture_sample_render).
    const uint32_t vs[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
        0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    std::vector<uint32_t> vert = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]));
    CHECK(!vert.empty() && vert[0] == 0x07230203u, "recompiled fullscreen-triangle VS -> SPIR-V");

    // PS: v0 = literal u, v1 = literal v, v2 = literal LOD; image_sample_l v[4:7], v[0:2],
    // s[8:15], s[16:19] dim:2D dmask:0xf (word0 0xf0900f08 = the sample template's word with op
    // 0x24); exp mrt0 v4..v7. Literal dwords at indices 1 (u), 3 (v), 5 (lod) are patched per draw.
    const uint32_t ps_template[] = {
        0x7e0002ffu, 0x3e000000u,   // v_mov_b32 v0, <u>
        0x7e0202ffu, 0x3e000000u,   // v_mov_b32 v1, <v>
        0x7e0402ffu, 0x40000000u,   // v_mov_b32 v2, <lod>
        0xf0900f08u, 0x00820400u,   // image_sample_l v[4:7], v[0:2], s8, s16
        0xf800000fu, 0x07060504u,   // exp mrt0 v4..v7
        0xbf810000u,
    };
    ShaderResourceTable rt;
    { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 1;
      t.width = 4; t.height = 4; t.sgpr_base = 8; rt.resources.push_back(t); }

    // 4x4 RGBA8: columns alternate red/blue -> box-filtered level 1 (2x2) and level 2 (1x1) are
    // purple (128, 0, 128) — a value no level-0 texel contains.
    uint8_t texels[4 * 4 * 4];
    for (uint32_t y = 0; y < 4; y++) for (uint32_t x = 0; x < 4; x++) {
        uint8_t* t = &texels[(y * 4 + x) * 4];
        t[0] = (x & 1) ? 0 : 255; t[1] = 0; t[2] = (x & 1) ? 255 : 0; t[3] = 255;
    }

    auto render_at = [&](uint32_t u_bits, uint32_t v_bits, uint32_t lod_bits,
                         uint32_t declared_levels, uint8_t out_rgb[3]) -> bool {
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = u_bits; ps[3] = v_bits; ps[5] = lod_bits;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        if (frag.empty() || frag[0] != 0x07230203u) return false;
        prosper::test::FrameResource resource;
        resource.binding = 4; resource.set = 1;
        resource.tex_rgba = texels; resource.tw = 4; resource.th = 4;
        resource.declared_mip_levels = declared_levels;
        resource.mip_filter = 0;      // NEAREST mip select: LOD 2.0 reads exactly level 2
        resource.max_lod = 8.0f;      // S# LOD clamp above the chain; the image level count clamps
        prosper::test::BackendDraw draw;
        draw.vs = vert; draw.fs = frag; draw.R = {resource}; draw.vcount = 3;
        std::vector<uint8_t> px = prosper::test::render_draws_rgba({draw}, W, H);
        if (px.size() != (size_t)W * H * 4) return false;
        const uint8_t* c = &px[((size_t)(H / 2) * W + W / 2) * 4];
        out_rgb[0] = c[0]; out_rgb[1] = c[1]; out_rgb[2] = c[2];
        return true;
    };

    const uint32_t U = 0x3e000000u /*0.125f -> texel column 0*/, LOD0 = 0x00000000u, LOD2 = 0x40000000u;
    uint8_t rgb[3];

    bool ok = render_at(U, U, LOD2, 3, rgb);
    printf("  lod2 declared=3 center=(%u,%u,%u)\n", ok?rgb[0]:0, ok?rgb[1]:0, ok?rgb[2]:0);
    CHECK(ok && rgb[0] > 0x60 && rgb[0] < 0xA0 && rgb[1] < 0x20 && rgb[2] > 0x60 && rgb[2] < 0xA0,
          "LOD 2 with a declared 3-level chain samples the box-filtered purple (fails without #1272)");

    ok = render_at(U, U, LOD0, 3, rgb);
    printf("  lod0 declared=3 center=(%u,%u,%u)\n", ok?rgb[0]:0, ok?rgb[1]:0, ok?rgb[2]:0);
    CHECK(ok && rgb[0] > 0xC0 && rgb[2] < 0x40,
          "LOD 0 still reads the untouched base level (red) through the chain upload");

    ok = render_at(U, U, LOD2, 1, rgb);
    printf("  lod2 declared=1 center=(%u,%u,%u)\n", ok?rgb[0]:0, ok?rgb[1]:0, ok?rgb[2]:0);
    CHECK(ok && rgb[0] > 0xC0 && rgb[2] < 0x40,
          "declared_mip_levels=1 (the default) keeps the historical single-level clamp-to-0 behavior");

    // IMAGE_SAMPLE_D supplies [Ds/Dx, Dt/Dx, Ds/Dy, Dt/Dy, u, v]. Unit normalized-coordinate
    // gradients over a 4x4 texture select LOD 2, while the literal UV has zero implicit quad
    // derivatives and selects LOD 0. Purple therefore proves that the explicit Grad operands survive.
    const uint32_t ps_grad[] = {
        0x7e0002ffu, 0x3f800000u,   // v0 = Ds/Dx = 1
        0x7e020280u,                // v1 = Dt/Dx = 0
        0x7e040280u,                // v2 = Ds/Dy = 0
        0x7e0602ffu, 0x3f800000u,   // v3 = Dt/Dy = 1
        0x7e0802ffu, U,             // v4 = u = 0.125
        0x7e0a02ffu, U,             // v5 = v = 0.125
        0xf0880f08u, 0x00820600u,   // image_sample_d v[6:9], v[0:5], s8, s16, dmask:xyzw dim:2D
        0xf800000fu, 0x09080706u,   // exp mrt0 v6..v9
        0xbf810000u,
    };
    std::vector<uint32_t> grad_frag = recompile_fragment(
        ps_grad, sizeof(ps_grad) / sizeof(ps_grad[0]), &rt);
    prosper::test::FrameResource grad_resource;
    grad_resource.binding = 4; grad_resource.set = 1;
    grad_resource.tex_rgba = texels; grad_resource.tw = 4; grad_resource.th = 4;
    grad_resource.declared_mip_levels = 3;
    grad_resource.mip_filter = 0;
    grad_resource.max_lod = 8.0f;
    prosper::test::BackendDraw grad_draw;
    grad_draw.vs = vert; grad_draw.fs = grad_frag; grad_draw.R = {grad_resource}; grad_draw.vcount = 3;
    std::vector<uint8_t> grad_px = grad_frag.empty() ? std::vector<uint8_t>{}
        : prosper::test::render_draws_rgba({grad_draw}, W, H);
    const uint8_t* grad_center = grad_px.size() == (size_t)W * H * 4
        ? &grad_px[((size_t)(H / 2) * W + W / 2) * 4] : nullptr;
    printf("  sample_d unit-grad center=(%u,%u,%u)\n",
           grad_center ? grad_center[0] : 0, grad_center ? grad_center[1] : 0,
           grad_center ? grad_center[2] : 0);
    CHECK(!grad_frag.empty() && grad_center &&
              grad_center[0] > 0x60 && grad_center[0] < 0xA0 &&
              grad_center[1] < 0x20 && grad_center[2] > 0x60 && grad_center[2] < 0xA0,
          "IMAGE_SAMPLE_D unit gradients select LOD 2 instead of the literal UV's implicit LOD 0");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
