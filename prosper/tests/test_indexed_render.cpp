// test_indexed_render — REAL indexed draws (issue #64): the render backend consumes a BackendDraw's
// index data via vkCmdBindIndexBuffer + vkCmdDrawIndexed, so gl_VertexIndex IS the fetched index and
// the recompiled VS's storage-buffer vertex fetch pulls the right records whatever the vertex order.
//
// Case A — the retired quad-fan heuristic's contract: a 4-vertex quad VB in perimeter order
// (BL,TL,TR,BR) drawn with the 6-entry index list [0,1,2, 2,3,0] as TRIANGLE_LIST must produce
// EXACTLY the pixels the old "4-record VB -> TRIANGLE_FAN" hack produced (byte-identical frames).
// This is the live title composite's shape: a DrawIndex quad with 6 x 16-bit indices.
//
// Case B — indices are actually applied: indices [1,2,3] select the fullscreen triangle stored at VB
// records 1..3 (record 0 is an off-screen decoy). A path that ignored the indices would draw records
// 0..2 instead and leave the sampled corners blue.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "../src/gpu/render_state.hpp"
#include "render_runner.h"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_indexed_render ==\n");
    const uint32_t W = 64, H = 64;

    // Vertex-fetch VS (llvm-mc gfx1030, same blob as test_vertex_fetch_render): fetch (x,y) from the
    // vertex buffer indexed by vid, pos=(x,y,0,1). Green PS.
    const uint32_t vs[] = {
        0x7e060280u, 0x7e0802f2u, 0xe0042000u, 0x80020100u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
    };
    const uint32_t ps[] = { 0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };

    ShaderResourceTable rt;
    ShaderResource vb{};
    vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32; vb.num_components = 2;
    vb.binding = 3; vb.stride = 8; vb.sgpr_base = 8;
    rt.resources.push_back(vb);

    std::vector<uint32_t> vert = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]), &rt);
    std::vector<uint32_t> frag = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
    CHECK(!vert.empty() && !frag.empty(), "recompiled vertex-fetch VS + green PS");
    if (vert.empty() || frag.empty()) { printf("== FAIL ==\n"); return 1; }

    auto f = [](float v) { union { float f; uint32_t u; } c; c.f = v; return c.u; };
    auto isGreen = [&](const std::vector<uint8_t>& px, uint32_t x, uint32_t y) {
        const uint8_t* p = &px[((size_t)y * W + x) * 4];
        return p[1] > 0x80 && p[0] < 0x40 && p[2] < 0x40;
    };
    auto greenAt9 = [&](const std::vector<uint8_t>& px) {   // 9-point sample grid, all green?
        const uint32_t xs[] = {0, W/2, W-1}, ys[] = {0, H/2, H-1};
        uint32_t g = 0; for (uint32_t y : ys) for (uint32_t x : xs) if (isGreen(px, x, y)) g++;
        return g == 9u;
    };
    auto draw_of = [&](const std::vector<uint32_t>& vbuf, const ResolvedPipelineState* st,
                       uint32_t vcount, std::vector<uint32_t> idx) {
        prosper::test::BackendDraw d;
        d.vs = vert; d.fs = frag; d.ps = st; d.vcount = vcount; d.indices = std::move(idx);
        prosper::test::FrameResource r; r.binding = 3; r.set = 0; r.dwords = vbuf;
        d.R.push_back(std::move(r));
        return d;
    };

    // --- Case A: indexed quad (TRIANGLE_LIST, indices [0,1,2, 2,3,0]) == old fan path, pixel-exact ---
    // Full-viewport quad, perimeter order BL,TL,TR,BR (the order the fan heuristic assumed).
    std::vector<uint32_t> quad = { f(-1.f), f(-1.f),   f(-1.f), f(1.f),
                                   f( 1.f), f( 1.f),   f( 1.f), f(-1.f) };
    ResolvedPipelineState list_ps; list_ps.topology = 3;   // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    ResolvedPipelineState fan_ps;  fan_ps.topology  = 5;   // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN

    std::vector<uint8_t> px_idx = prosper::test::render_draws_rgba(
        { draw_of(quad, &list_ps, 4, {0,1,2, 2,3,0}) }, W, H);
    std::vector<uint8_t> px_fan = prosper::test::render_draws_rgba(
        { draw_of(quad, &fan_ps, 4, {}) }, W, H);
    CHECK(px_idx.size() == (size_t)W*H*4 && px_fan.size() == (size_t)W*H*4, "both quad renders produced frames");
    if (px_idx.size() != (size_t)W*H*4 || px_fan.size() != (size_t)W*H*4) { printf("== FAIL ==\n"); return 1; }
    CHECK(greenAt9(px_idx), "indexed [0,1,2,2,3,0] TRIANGLE_LIST quad fills the viewport GREEN");
    CHECK(px_idx == px_fan, "indexed quad is byte-identical to the old perimeter-fan rendering");

    // --- Case B: indices select records 1..3 (fullscreen triangle); record 0 is an off-screen decoy ---
    std::vector<uint32_t> vbufB = { f(9.f), f(9.f),                     // rec 0: decoy (off-screen)
                                    f(-1.f), f(-1.f), f(3.f), f(-1.f), f(-1.f), f(3.f) };  // recs 1..3
    std::vector<uint8_t> px_tri = prosper::test::render_draws_rgba(
        { draw_of(vbufB, &list_ps, 4, {1,2,3}) }, W, H);
    CHECK(px_tri.size() == (size_t)W*H*4, "indexed triangle rendered");
    if (px_tri.size() != (size_t)W*H*4) { printf("== FAIL ==\n"); return 1; }
    CHECK(greenAt9(px_tri), "indices [1,2,3] fetched records 1..3 (fullscreen GREEN triangle)");
    // Negative control: the same draw WITHOUT indices draws records 0..2 (decoy included) — a different
    // picture. Guards against a backend that silently ignores the index buffer.
    std::vector<uint8_t> px_noidx = prosper::test::render_draws_rgba(
        { draw_of(vbufB, &list_ps, 3, {}) }, W, H);
    CHECK(px_noidx.size() == (size_t)W*H*4 && px_noidx != px_tri,
          "dropping the indices changes the picture (indices are really applied)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
