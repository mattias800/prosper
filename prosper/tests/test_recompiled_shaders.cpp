// test_recompiled_shaders — render a frame whose BOTH shaders are recompiled from RDNA2.
// Vertex shader: a fullscreen triangle computed from gl_VertexIndex (pos.x=(vid&1)*4-1,
// pos.y=(vid>>1)*4-1) exported via EXP POS0. Pixel shader: solid green via EXP MRT0. Both are
// assembled by llvm-mc for gfx1030, recompiled to SPIR-V by recompile_vertex/recompile_fragment,
// and run through a real Vulkan pipeline. The fullscreen triangle covers the whole viewport, so we
// assert every sampled pixel is GREEN — proving RDNA2 vertex+pixel -> our SPIR-V -> rendered frame.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "render_runner.h"
#include <cstdio>
#include <cstdint>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_recompiled_shaders ==\n");
    const uint32_t W = 64, H = 64;

    // Fullscreen-triangle vertex shader (llvm-mc gfx1030): pos = ((vid&1)*4-1, (vid>>1)*4-1, 0, 1).
    const uint32_t vs[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
        0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    // Green pixel shader: exp mrt0 (0, 1.0, 0, 1.0).
    const uint32_t ps[] = { 0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };

    std::vector<uint32_t> vert = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]));
    std::vector<uint32_t> frag = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
    CHECK(!vert.empty() && vert[0] == 0x07230203u, "recompiled RDNA2 vertex shader -> SPIR-V");
    CHECK(!frag.empty() && frag[0] == 0x07230203u, "recompiled RDNA2 pixel shader  -> SPIR-V");
    if (vert.empty() || frag.empty()) { printf("== FAIL ==\n"); return 1; }

    std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H);
    CHECK(px.size() == (size_t)W * H * 4, "pipeline accepted both recompiled shaders + rendered");
    if (px.size() != (size_t)W * H * 4) { printf("== FAIL: render failed ==\n"); return 1; }

    // The fullscreen triangle covers the whole viewport -> every sampled pixel is green.
    auto isGreen = [&](uint32_t x, uint32_t y) {
        const uint8_t* p = &px[((size_t)y * W + x) * 4];
        return p[1] > 0x80 && p[0] < 0x40 && p[2] < 0x40;
    };
    const uint32_t xs[] = {0, W/2, W-1}, ys[] = {0, H/2, H-1};
    uint32_t green = 0, total = 0;
    for (uint32_t y : ys) for (uint32_t x : xs) { total++; if (isGreen(x, y)) green++; }
    const uint8_t* c = &px[((size_t)(H/2) * W + W/2) * 4];
    printf("  center=(%u,%u,%u,%u)  green samples %u/%u\n", c[0],c[1],c[2],c[3], green, total);
    CHECK(green == total, "every sampled pixel is GREEN (recompiled VS positioned the tri, PS colored it)");

    // --- NGG vertex shader (the game's shaders 004/025 pattern) ---------------------------------------
    // An NGG VS wraps the vertex work in wave-packing plumbing (s_sendmsg GS_ALLOC_REQ / exp prim /
    // s_lshr_b64 exec) that lowers to no-ops per-invocation, and carries the vertex index in v5. This
    // shader computes pos = (((v5<<1)&2)-1, (v5&-2)-1, 0, 1): v5=0,1,2 -> (-1,-1),(1,-1),(-1,1) — a
    // triangle over the lower-left half of NDC (screen verts (0,0),(63,0),(0,63)). Verifies the NGG
    // no-op lowering AND the v5=gl_VertexIndex ABI produce correct geometry, not just valid SPIR-V.
    const uint32_t nggvs[] = {
        0x93EAFF03u, 0x00080008u, 0x876BFF03u, 0x000000FFu, 0x8F6A8C6Au, 0x887C6A6Bu, 0xBF900009u,
        0x906A8803u, 0x81EA6A80u, 0x90FE6AC1u, 0xF8000941u, 0x00000000u, 0x81EA0380u, 0x90FE6AC1u,
        0x34040A81u, 0x36060AC2u, 0x7E000280u, 0x7E0202F2u, 0x36040482u, 0x4A0606C1u, 0x4A0404C1u,
        0x7E060B03u, 0x7E040B02u, 0xF80008CFu, 0x01000302u, 0xBF810000u,
    };
    std::vector<uint32_t> nvert = recompile_vertex(nggvs, sizeof(nggvs)/sizeof(nggvs[0]));
    CHECK(!nvert.empty() && nvert[0] == 0x07230203u, "recompiled NGG vertex shader -> SPIR-V");
    if (!nvert.empty()) {
        std::vector<uint8_t> npx = prosper::test::render_triangle_rgba(nvert, frag, W, H);
        // Inside the lower-left-half triangle (x+y < ~63) is green; outside (bottom-right) is the clear.
        auto grn = [&](uint32_t x, uint32_t y){ const uint8_t* p=&npx[((size_t)y*W+x)*4]; return p[1]>0x80 && p[0]<0x40 && p[2]<0x40; };
        bool inside_green = grn(16, 16), outside_not = !grn(60, 60);
        printf("  NGG: inside(16,16)green=%d outside(60,60)notgreen=%d\n", inside_green, outside_not);
        CHECK(npx.size()==(size_t)W*H*4, "pipeline accepted the recompiled NGG VS + rendered");
        CHECK(inside_green && outside_not, "NGG VS geometry correct (v5->lower-left-half triangle)");
    }

    // Terminal NGG compaction gates its POS/PARAM exports with CMPX + EXECZ. The vertex shell cannot
    // suppress a Vulkan invocation, so inactive suffix vertices must be mapped to one degenerate clip
    // point while active vertices retain their real positions. Exercise both Boolean outcomes through
    // the real rasterizer: active renders the same lower-left triangle; inactive renders no fragments.
    const uint32_t ngg_gate_active[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ
        0x34040A81u, 0x36060AC2u,           // v2/v3 from NGG v5 vertex index
        0x7E000280u, 0x7E0202F2u,
        0x36040482u, 0x4A0606C1u, 0x4A0404C1u,
        0x7E060B03u, 0x7E040B02u,           // triangle position in v2,v3,v0,v1
        0x7E280281u, 0x7E2A0280u,           // v20=1, v21=0
        0x7DA82B14u,                         // v_cmpx_gt_u32 v20, v21 -> true
        0xBF880002u,                         // s_cbranch_execz -> end
        0xF80008CFu, 0x01000302u,
        0xBF810000u,
    };
    const uint32_t ngg_gate_inactive[] = {
        0xBF900009u,
        0x34040A81u, 0x36060AC2u,
        0x7E000280u, 0x7E0202F2u,
        0x36040482u, 0x4A0606C1u, 0x4A0404C1u,
        0x7E060B03u, 0x7E040B02u,
        0x7E280280u, 0x7E2A0280u,           // v20=0, v21=0
        0x7DA82B14u,                         // v_cmpx_gt_u32 v20, v21 -> false
        0xBF880002u,
        0xF80008CFu, 0x01000302u,
        0xBF810000u,
    };
    const auto active_gate_spv = recompile_vertex(
        ngg_gate_active, std::size(ngg_gate_active));
    const auto inactive_gate_spv = recompile_vertex(
        ngg_gate_inactive, std::size(ngg_gate_inactive));
    CHECK(!active_gate_spv.empty() && !inactive_gate_spv.empty(),
          "terminal NGG output gate recompiles for active and inactive paths");
    if (!active_gate_spv.empty() && !inactive_gate_spv.empty()) {
        const auto active_px = prosper::test::render_triangle_rgba(active_gate_spv, frag, W, H);
        const auto inactive_px = prosper::test::render_triangle_rgba(inactive_gate_spv, frag, W, H);
        auto green_pixels = [&](const std::vector<uint8_t>& pixels) {
            size_t count = 0;
            for (size_t i = 0; i + 3 < pixels.size(); i += 4)
                count += pixels[i + 1] > 0x80 && pixels[i] < 0x40 && pixels[i + 2] < 0x40;
            return count;
        };
        const size_t active_green = green_pixels(active_px);
        const size_t inactive_green = green_pixels(inactive_px);
        printf("  NGG gate: active-green=%zu inactive-green=%zu\n",
               active_green, inactive_green);
        CHECK(active_green > 0, "active terminal NGG output path retains real geometry");
        CHECK(inactive_green == 0, "inactive terminal NGG output path degenerates the vertex tail");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
