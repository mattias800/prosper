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
#include <cstdlib>
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

    // The live renderer commonly submits hundreds of draws that reference the same few decoded
    // textures. The backend must upload identical pixels once per call while preserving the legacy
    // rendered bytes and separate per-descriptor views/samplers.
    {
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = C075; ps[3] = C025;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        prosper::test::FrameResource resource;
        resource.binding = 4; resource.set = 1;
        resource.tex_rgba = texels; resource.tw = 2; resource.th = 2;
        prosper::test::BackendDraw draw;
        draw.vs = vert; draw.fs = frag; draw.R = {resource}; draw.vcount = 3;
        prosper::test::FrameResource swizzled_resource = resource;
        swizzled_resource.swizzle[0] = 5;
        swizzled_resource.swizzle[1] = 4;
        prosper::test::BackendDraw swizzled_draw = draw;
        swizzled_draw.R = {swizzled_resource};

        std::vector<uint8_t> shared =
            prosper::test::render_draws_rgba({draw, swizzled_draw}, W, H);
        const auto shared_stats = prosper::test::backend_texture_upload_stats();
        CHECK(shared_stats.references == 2 && shared_stats.unique_uploads == 1,
              "draws with separate views over shared pixels produce one backend texture upload");

#ifdef _WIN32
        _putenv_s("PROSPER_NO_BACKEND_TEXTURE_SHARE", "1");
#else
        setenv("PROSPER_NO_BACKEND_TEXTURE_SHARE", "1", 1);
#endif
        std::vector<uint8_t> legacy =
            prosper::test::render_draws_rgba({draw, swizzled_draw}, W, H);
        const auto legacy_stats = prosper::test::backend_texture_upload_stats();
#ifdef _WIN32
        _putenv_s("PROSPER_NO_BACKEND_TEXTURE_SHARE", "");
#else
        unsetenv("PROSPER_NO_BACKEND_TEXTURE_SHARE");
#endif
        CHECK(legacy_stats.references == 2 && legacy_stats.unique_uploads == 2,
              "disable switch restores one backend upload per texture reference");
        CHECK(!shared.empty() && shared == legacy,
              "shared and legacy uploads with distinct view swizzles render byte-identically");
    }

    // A non-zero content ID is an exact-validation proof supplied by the frontend. The backend may
    // retain that uploaded image across render callbacks, but the diagnostic switch must preserve
    // rendered output while forcing the old upload path for controlled A/B runs.
    {
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = C075; ps[3] = C025;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        prosper::test::FrameResource resource;
        resource.binding = 4; resource.set = 1;
        resource.tex_rgba = texels; resource.tw = 2; resource.th = 2;
        resource.persistent_texture_id = 0x7020000000000001ull;
        prosper::test::BackendDraw draw;
        draw.vs = vert; draw.fs = frag; draw.R = {resource}; draw.vcount = 3;

        std::vector<uint8_t> first = prosper::test::render_draws_rgba({draw}, W, H);
        const auto first_stats = prosper::test::backend_texture_upload_stats();
        std::vector<uint8_t> reused = prosper::test::render_draws_rgba({draw}, W, H);
        const auto reused_stats = prosper::test::backend_texture_upload_stats();
        CHECK(first_stats.persistent_misses == 1 && first_stats.unique_uploads == 1,
              "first exact-validated texture version is uploaded into the persistent cache");
        CHECK(reused_stats.persistent_hits == 1 && reused_stats.unique_uploads == 0 &&
                  reused_stats.upload_bytes == 0,
              "same exact-validated texture version skips its next callback upload");
        CHECK(!first.empty() && first == reused,
              "persistent texture reuse renders byte-identically to its initial upload");

#ifdef _WIN32
        _putenv_s("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES", "1");
#else
        setenv("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES", "1", 1);
#endif
        std::vector<uint8_t> bypassed = prosper::test::render_draws_rgba({draw}, W, H);
        const auto bypassed_stats = prosper::test::backend_texture_upload_stats();
#ifdef _WIN32
        _putenv_s("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES", "");
#else
        unsetenv("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES");
#endif
        CHECK(bypassed_stats.persistent_hits == 0 && bypassed_stats.unique_uploads == 1,
              "persistent texture disable switch restores an upload on every callback");
        CHECK(reused == bypassed,
              "persistent cache and forced-upload paths render byte-identically");

        uint8_t changed_texels[sizeof(texels)];
        std::memcpy(changed_texels, texels, sizeof(texels));
        changed_texels[4] = 255;  // sampled top-right texel: green -> yellow
        resource.tex_rgba = changed_texels;
        resource.persistent_texture_id = 0x7020000000000002ull;
        draw.R = {resource};
        std::vector<uint8_t> changed = prosper::test::render_draws_rgba({draw}, W, H);
        const auto changed_stats = prosper::test::backend_texture_upload_stats();
        CHECK(changed_stats.persistent_misses == 1 && changed_stats.unique_uploads == 1,
              "a new exact-validated content version cannot hit the prior image");
        CHECK(!changed.empty() && changed != reused,
              "a new content version uploads and renders its changed pixels");
    }

    // The backend upload key includes depth and image dimensionality. Exercise a real 3D image here
    // so depth-1 3D resources cannot accidentally regress to a 2D Vulkan image/view during sharing.
    {
        const uint32_t ps_3d[] = {
            0x7e0002ffu, C025, 0x7e0202ffu, C025, 0x7e0402ffu, C075,
            0xf0800f10u, 0x00820000u, 0xf800000fu, 0x03020100u, 0xbf810000u,
        };
        ShaderResourceTable rt_3d;
        { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 2;
          t.width = 2; t.height = 2; t.depth = 2; t.sgpr_base = 8;
          rt_3d.resources.push_back(t); }
        std::vector<uint32_t> frag_3d =
            recompile_fragment(ps_3d, sizeof(ps_3d) / sizeof(ps_3d[0]), &rt_3d);
        CHECK(!frag_3d.empty(), "recompiled a 3D image_sample fragment shader");
        if (!frag_3d.empty()) {
            const uint8_t volume[2*2*2*4] = {
                255,0,0,255, 255,0,0,255, 255,0,0,255, 255,0,0,255,
                0,255,0,255, 0,255,0,255, 0,255,0,255, 0,255,0,255,
            };
            prosper::test::FrameResource resource_3d;
            resource_3d.binding = 4; resource_3d.set = 1;
            resource_3d.tex_rgba = volume;
            resource_3d.tw = 2; resource_3d.th = 2; resource_3d.td = 2;
            resource_3d.img_dim = 2;
            prosper::test::BackendDraw draw_3d;
            draw_3d.vs = vert; draw_3d.fs = frag_3d; draw_3d.R = {resource_3d};
            draw_3d.vcount = 3;
            std::vector<uint8_t> volume_px =
                prosper::test::render_draws_rgba({draw_3d, draw_3d}, W, H);
            const auto volume_stats = prosper::test::backend_texture_upload_stats();
            bool volume_ok = volume_px.size() == static_cast<size_t>(W) * H * 4;
            if (volume_ok) {
                const uint8_t* c = &volume_px[((size_t)(H/2) * W + W/2) * 4];
                volume_ok = c[1] > 0x80 && c[0] < 0x40 && c[2] < 0x40;
            }
            CHECK(volume_ok, "shared 3D upload samples the selected depth slice");
            CHECK(volume_stats.references == 2 && volume_stats.unique_uploads == 1 &&
                      volume_stats.upload_bytes == sizeof(volume),
                  "3D texture sharing accounts for depth and uploads the volume once");
        }
    }

    bool ok2 = sample_center(C025, C075, rgb);
    printf("  (0.25,0.75) center=(%u,%u,%u)\n", ok2?rgb[0]:0, ok2?rgb[1]:0, ok2?rgb[2]:0);
    CHECK(ok2 && rgb[2] > 0x80 && rgb[0] < 0x40 && rgb[1] < 0x40, "sampling texel (0,1) yields BLUE (proves v routing)");

    // #275: anisotropy applied. A sampler built with max_aniso_ratio > 0 (here 4 -> 16x) must create
    // validly and still sample correctly — the fullscreen quad isn't minified, so aniso changes nothing
    // about the result, but a broken apply (invalid usage / wrong sampler) would blank the frame or move
    // the texel. Same (0.75,0.25) -> texel (1,0) = GREEN check, now through the anisotropic sampler.
    {
        prosper::test::TexDesc td_a{ /*binding*/4, /*w*/2, /*h*/2, texels, /*max_aniso_ratio*/4u };
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = C075; ps[3] = C025;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        bool okA = !frag.empty() && frag[0] == 0x07230203u;
        if (okA) {
            std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, nullptr, nullptr, nullptr, &td_a);
            okA = px.size() == (size_t)W*H*4;
            if (okA) { const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4]; rgb[0]=c[0]; rgb[1]=c[1]; rgb[2]=c[2]; }
        }
        printf("  aniso(16x) (0.75,0.25) center=(%u,%u,%u)\n", okA?rgb[0]:0, okA?rgb[1]:0, okA?rgb[2]:0);
        CHECK(okA && rgb[1] > 0x80 && rgb[0] < 0x40 && rgb[2] < 0x40,
              "anisotropic sampler (ratio 4/16x) still samples texel (1,0) = GREEN (valid apply)");
    }

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

    // image_sample_lz (explicit LOD 0): coords (0.75,0.25) -> texel (1,0) = green, same as image_sample.
    const uint32_t lz[] = {
        0x7e0002ffu, 0x3f400000u, 0x7e0202ffu, 0x3e800000u, 0xf09c0f08u, 0x00820000u,
        0xf800080fu, 0x03020100u, 0xbf810000u,
    };
    std::vector<uint32_t> lzf = recompile_fragment(lz, sizeof(lz)/sizeof(lz[0]), &rt);
    bool okLz = !lzf.empty() && lzf[0] == 0x07230203u;
    if (okLz) {
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, lzf, W, H, nullptr, nullptr, nullptr, &td);
        okLz = px.size() == (size_t)W*H*4;
        if (okLz) { const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4]; rgb[0]=c[0]; rgb[1]=c[1]; rgb[2]=c[2]; }
    }
    printf("  image_sample_lz(0.75,0.25) center=(%u,%u,%u)\n", okLz?rgb[0]:0, okLz?rgb[1]:0, okLz?rgb[2]:0);
    CHECK(okLz && rgb[1] > 0x80 && rgb[0] < 0x40 && rgb[2] < 0x40, "image_sample_lz (explicit LOD 0) samples texel (1,0) = GREEN");

    // image_sample_lz_o (#273 — DOLL FXAA): LOD-0 sample with a packed TEXEL offset in the first
    // vaddr (x=+1 in bits[5:0], y=+1 in bits[13:8]). Sampling (0.25,0.25) with offset (+1,+1) must
    // land on texel (1,1) = WHITE (the offset folds into the normalized coords via the level-0 size).
    const uint32_t lzo[] = {
        0x7e0002ffu, 0x00000101u, 0x7e0202ffu, 0x3e800000u, 0x7e0402ffu, 0x3e800000u,
        0xf0dc0f08u, 0x00820000u, 0xf800080fu, 0x03020100u, 0xbf810000u,
    };
    std::vector<uint32_t> lzof = recompile_fragment(lzo, sizeof(lzo)/sizeof(lzo[0]), &rt);
    bool okLzo = !lzof.empty() && lzof[0] == 0x07230203u;
    CHECK(okLzo, "recompiled image_sample_lz_o PS -> SPIR-V");
    if (okLzo) {
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, lzof, W, H, nullptr, nullptr, nullptr, &td);
        okLzo = px.size() == (size_t)W*H*4;
        if (okLzo) { const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4]; rgb[0]=c[0]; rgb[1]=c[1]; rgb[2]=c[2]; }
        printf("  image_sample_lz_o(0.25,0.25,+1,+1) center=(%u,%u,%u)\n", okLzo?rgb[0]:0, okLzo?rgb[1]:0, okLzo?rgb[2]:0);
        CHECK(okLzo && rgb[0] > 0x80 && rgb[1] > 0x80 && rgb[2] > 0x80,
              "image_sample_lz_o offset (+1,+1) from texel (0,0) samples texel (1,1) = WHITE");
    }

    // image_gather4_lz_o (locks the #296 helper's operand-ID fix): gather the ALPHA channel (dmask
    // 0x8) with a packed (+1,+1) offset — every texel's alpha is 255, so all four gathered values
    // are 1.0 and the export is WHITE. (Before the fix the emitted OpBitFieldSExtract used raw
    // integers 0/6/8 as operand IDs — an invalid module the driver rejects -> nothing renders.)
    const uint32_t g4o[] = {
        0x7e0002ffu, 0x00000101u, 0x7e0202ffu, 0x3e800000u, 0x7e0402ffu, 0x3e800000u,
        0xf15c0808u, 0x00820400u, 0xf800000fu, 0x07060504u, 0xbf810000u,
    };
    std::vector<uint32_t> g4of = recompile_fragment(g4o, sizeof(g4o)/sizeof(g4o[0]), &rt);
    bool okG4o = !g4of.empty() && g4of[0] == 0x07230203u;
    CHECK(okG4o, "recompiled image_gather4_lz_o PS -> SPIR-V");
    if (okG4o) {
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, g4of, W, H, nullptr, nullptr, nullptr, &td);
        okG4o = px.size() == (size_t)W*H*4;
        if (okG4o) { const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4]; rgb[0]=c[0]; rgb[1]=c[1]; rgb[2]=c[2]; }
        printf("  image_gather4_lz_o alpha center=(%u,%u,%u)\n", okG4o?rgb[0]:0, okG4o?rgb[1]:0, okG4o?rgb[2]:0);
        CHECK(okG4o && rgb[0] > 0x80 && rgb[1] > 0x80 && rgb[2] > 0x80,
              "image_gather4_lz_o gathers alpha=1.0 x4 -> WHITE (valid module, offset decoded)");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
