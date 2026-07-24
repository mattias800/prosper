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

    // ---- Persistent-DS sampled bridge (#1275) ----
    // A depth-only producer renders a fullscreen triangle at z=0.25 into a guest-identified
    // persistent DS surface. prosper never writes that depth back to guest memory, so a consumer
    // sampling the depth-plane address as a 2D texture must be served by the retained Vulkan depth
    // image (the bridge); without it the consumer reads zeros. The consumer samples at (0.25,0.25)
    // and exports R = stored depth: bridge -> ~0.25 (R~64), no bridge -> 0.
    {
        const uint32_t vs_z[] = {
            0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100602f6u, 0x100804f6u,
            0x060606f3u, 0x060808f3u, 0x100a02f4u, 0x100c04f4u, 0x7e0e02ffu, 0x3e800000u,
            0x7e1002f2u, 0xf80008cfu, 0x08070403u, 0xf800020fu, 0x08070605u, 0xbf810000u,
        };
        const uint32_t ps_red[] = {
            0x7e0002f2u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
            0xf800180fu, 0x03020100u, 0xbf810000u,
        };
        std::vector<uint32_t> vert_z = recompile_vertex(vs_z, sizeof(vs_z) / sizeof(vs_z[0]));
        std::vector<uint32_t> red = recompile_fragment(ps_red, sizeof(ps_red) / sizeof(ps_red[0]),
                                                       nullptr);
        CHECK(!vert_z.empty() && !red.empty(), "recompiled z=0.25 producer shaders");

        constexpr uint64_t kDepthBase = 0x20d5000000ull;
        ResolvedPipelineState producer{};
        producer.topology = 3; producer.color_write_mask = 0xF;
        producer.depth_test_enable = true; producer.depth_write_enable = true;
        producer.depth_compare_op = 7;   // ALWAYS
        producer.depth_read_base = kDepthBase; producer.depth_write_base = kDepthBase;
        prosper::test::BackendDraw pw;
        pw.vs = vert_z; pw.fs = red; pw.ps = &producer; pw.vcount = 3;
        std::vector<uint8_t> produced = prosper::test::render_draws_rgba(
            {pw}, W, H, nullptr, nullptr, /*persist_depth_stencil=*/true);
        CHECK(!produced.empty(), "depth-writing producer rendered with a persistent DS identity");

        const uint32_t ps_sample[] = {
            0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u, 0xf0800f08u, 0x00820000u,
            0xf800000fu, 0x03020100u, 0xbf810000u,
        };
        ShaderResourceTable sample_rt;
        { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 1;
          t.width = W; t.height = H; t.sgpr_base = 8; sample_rt.resources.push_back(t); }
        std::vector<uint32_t> sample_fs = recompile_fragment(
            ps_sample, sizeof(ps_sample) / sizeof(ps_sample[0]), &sample_rt);
        CHECK(!sample_fs.empty(), "recompiled depth-plane sampling consumer");

        ResolvedPipelineState opaque{};
        opaque.topology = 3; opaque.color_write_mask = 0xF;
        prosper::test::FrameResource bridged;
        bridged.binding = 4; bridged.set = 1;
        bridged.persistent_depth_target_id = kDepthBase;
        bridged.tw = W; bridged.th = H;
        prosper::test::BackendDraw consumer;
        consumer.vs = vert; consumer.fs = sample_fs; consumer.ps = &opaque;
        consumer.R = {bridged}; consumer.vcount = 3;
        std::vector<uint8_t> via_bridge = prosper::test::render_draws_rgba({consumer}, W, H);
        CHECK(via_bridge.size() == (size_t)W * H * 4, "bridged consumer rendered");

        std::vector<uint8_t> zeros((size_t)W * H * 4, 0);
        prosper::test::FrameResource unbridged = bridged;
        unbridged.persistent_depth_target_id = 0;
        unbridged.tex_rgba = zeros.data();
        prosper::test::BackendDraw control = consumer;
        control.R = {unbridged};
        std::vector<uint8_t> via_zeros = prosper::test::render_draws_rgba({control}, W, H);

        const uint8_t* center = &via_bridge[(((size_t)H / 2) * W + W / 2) * 4];
        const uint8_t* center0 = &via_zeros[(((size_t)H / 2) * W + W / 2) * 4];
        printf("  bridged center R=%u control R=%u\n", center[0], center0[0]);
        CHECK(center[0] > 56 && center[0] < 72,
              "bridged consumer reads the produced depth (~0.25) from the persistent DS image");
        CHECK(center0[0] == 0, "control without the bridge samples zeros");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
