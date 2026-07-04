// test_vertex_fetch_render — the milestone: a vertex shader that FETCHES its positions from a bound
// vertex buffer (not computed from gl_VertexIndex) renders real geometry.
//
// This is how real game vertex shaders work: the driver puts a vertex-buffer V# in user-data SGPRs
// (s[8:11] here) and the shader does buffer_load_format_* to pull each vertex's attributes. To
// translate that load the recompiler must be told the attribute's data format — supplied via a
// ShaderResourceTable (DataFormat::Float32, binding 3). The render harness then binds the vertex
// buffer's bytes at binding 3, and the recompiled VS reads them.
//
// VS:  v3=0 ; v4=1.0 ; buffer_load_format_xy v[1:2], v0(=vid), s[8:11] idxen ; exp pos0 (x,y,0,1)
// The vertex buffer holds a fullscreen triangle {(-1,-1),(3,-1),(-1,3)}, so a correct fetch covers the
// whole viewport -> every pixel green. A broken/empty fetch collapses the triangle to the origin ->
// the frame stays blue (clear). Green-everywhere therefore proves the fetch delivered real data.
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
    printf("== test_vertex_fetch_render ==\n");
    const uint32_t W = 64, H = 64;

    // Vertex-fetch VS (llvm-mc gfx1030): fetch (x,y) from the vertex buffer indexed by vid, pos=(x,y,0,1).
    const uint32_t vs[] = {
        0x7e060280u, 0x7e0802f2u, 0xe0042000u, 0x80020100u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
    };
    // Green pixel shader: exp mrt0 (0, 1.0, 0, 1.0).
    const uint32_t ps[] = { 0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };

    // The front-half's contract: the vertex attribute is float32 x2, its V# sits directly in s[8:11].
    ShaderResourceTable rt;
    ShaderResource vb{};
    vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32; vb.num_components = 2;
    vb.binding = 3; vb.stride = 8; vb.sgpr_base = 8;
    rt.resources.push_back(vb);

    std::vector<uint32_t> vert = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]), &rt);
    std::vector<uint32_t> frag = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
    CHECK(!vert.empty() && vert[0] == 0x07230203u, "recompiled vertex-fetch VS -> SPIR-V (with resource table)");
    CHECK(!frag.empty() && frag[0] == 0x07230203u, "recompiled green pixel shader -> SPIR-V");
    if (vert.empty() || frag.empty()) { printf("== FAIL ==\n"); return 1; }

    // The vertex buffer: 3 vertices x (float x, float y), a fullscreen triangle. stride 8 bytes.
    auto f = [](float v) { union { float f; uint32_t u; } c; c.f = v; return c.u; };
    std::vector<uint32_t> vbuf = { f(-1.f), f(-1.f),  f(3.f), f(-1.f),  f(-1.f), f(3.f) };

    std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, nullptr, &vbuf);
    CHECK(px.size() == (size_t)W * H * 4, "pipeline bound the vertex buffer + rendered");
    if (px.size() != (size_t)W * H * 4) { printf("== FAIL: render failed ==\n"); return 1; }

    auto isGreen = [&](uint32_t x, uint32_t y) {
        const uint8_t* p = &px[((size_t)y * W + x) * 4];
        return p[1] > 0x80 && p[0] < 0x40 && p[2] < 0x40;
    };
    const uint32_t xs[] = {0, W/2, W-1}, ys[] = {0, H/2, H-1};
    uint32_t green = 0, total = 0;
    for (uint32_t y : ys) for (uint32_t x : xs) { total++; if (isGreen(x, y)) green++; }
    const uint8_t* c = &px[((size_t)(H/2) * W + W/2) * 4];
    printf("  center=(%u,%u,%u,%u)  green samples %u/%u\n", c[0],c[1],c[2],c[3], green, total);
    CHECK(green == total, "every sampled pixel is GREEN (VS fetched real positions from the vertex buffer)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
