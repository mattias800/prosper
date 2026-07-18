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
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <iterator>
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

    // AMD's explicit-parameter form reconstructs that same smooth value from
    //   Final = P0 + P10*I + P20*J
    // (official GFX10/RDNA2 ISA). Linux llvmpipe exposes no fragment-barycentric extension, so the
    // generated geometry stage publishes P0/P10/P20 plus perspective-center I/J. This is the exact
    // mechanism Astro Bot's title composite needs; verify it with a real Vulkan pipeline and pixels.
    const uint32_t explicit_ps[] = {
        0xc80e0000u, 0xc8120001u, 0xc8160002u,       // v3=P10, v4=P20, v5=P0, attr0.x
        0xd54b0003u, 0x04160103u,                    // v3 = P10*I + P0
        0xd54b0003u, 0x040e0304u,                    // v3 = P20*J + v3
        0x7e080280u, 0x7e0a0280u, 0x7e0c02f2u,       // G=B=0, A=1
        0xf800000fu, 0x06050403u, 0xbf810000u,
    };
    PixelSystemInputMapping perspective_center{1u << 1, 1u << 1};
    FragmentInterpolationLayout explicit_layout = fragment_interpolation_layout(
        explicit_ps, std::size(explicit_ps), &perspective_center);
    std::vector<uint32_t> explicit_frag = recompile_fragment(
        explicit_ps, std::size(explicit_ps), nullptr, &perspective_center,
        UINT32_MAX, &explicit_layout);
    std::vector<uint32_t> explicit_geom = recompile_interpolation_geometry(explicit_layout);
    CHECK(explicit_layout.valid && explicit_layout.requires_geometry &&
          !explicit_frag.empty() && !explicit_geom.empty(),
          "P0/P10/P20 + perspective-center barycentrics generate portable SPIR-V stages");
    ResolvedPipelineState explicit_state;
    explicit_state.topology = 3; // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    prosper::test::BackendDraw explicit_draw;
    explicit_draw.vs = vert; explicit_draw.gs = explicit_geom; explicit_draw.fs = explicit_frag;
    explicit_draw.ps = &explicit_state;
    std::vector<prosper::test::BackendDraw> explicit_draws;
    explicit_draws.push_back(std::move(explicit_draw));
    std::vector<uint8_t> explicit_px = prosper::test::render_draws_rgba(explicit_draws, W, H);
    CHECK(explicit_px.size() == (size_t)W * H * 4,
          "Vulkan links and renders the generated interpolation geometry stage");
    if (explicit_px.size() == (size_t)W * H * 4) {
        uint8_t explicit_min = 255, explicit_max = 0;
        uint32_t explicit_max_gb = 0;
        for (uint32_t y = 2; y < H - 2; y += 4) for (uint32_t x = 2; x < W - 2; x += 4) {
            const uint8_t* p = &explicit_px[((size_t)y * W + x) * 4];
            explicit_min = std::min(explicit_min, p[0]);
            explicit_max = std::max(explicit_max, p[0]);
            explicit_max_gb = std::max<uint32_t>(explicit_max_gb, std::max(p[1], p[2]));
        }
        printf("  explicit-parameter red range [%u..%u] max(green,blue)=%u\n",
               explicit_min, explicit_max, explicit_max_gb);
        CHECK(explicit_max - explicit_min > 40 && explicit_max_gb < 0x20,
              "P0 + P10*I + P20*J reconstructs the original smooth red gradient");
    }

    // DOLL's live linkage shape is non-identity: PS input 1 consumes producer PARAM3. Exercise the
    // generic form here by moving this fixture's export from PARAM0 to PARAM3, then remapping logical
    // PS input 0 back to source slot 3. The unchanged attr0 PS must still receive the gradient.
    std::vector<uint32_t> param3_vs(vs, vs + sizeof(vs)/sizeof(vs[0]));
    for (uint32_t& word : param3_vs) if (word == 0xf800020fu) word = 0xf800023fu;
    PixelInputMapping remap;
    remap.valid_mask = 1u;
    remap.controls[0] = 3u;
    std::vector<uint32_t> remapped_vert = recompile_vertex(
        param3_vs.data(), param3_vs.size(), nullptr, &remap);
    std::vector<uint8_t> remapped_px = prosper::test::render_triangle_rgba(
        remapped_vert, frag, W, H);
    CHECK(remapped_px.size() == (size_t)W * H * 4,
          "SPI_PS_INPUT_CNTL remap links source PARAM3 to logical PS input 0");
    if (remapped_px.size() == (size_t)W * H * 4) {
        uint8_t remapped_min = 255, remapped_max = 0;
        for (size_t i = 0; i < remapped_px.size(); i += 4) {
            remapped_min = std::min(remapped_min, remapped_px[i]);
            remapped_max = std::max(remapped_max, remapped_px[i]);
        }
        CHECK(remapped_max - remapped_min > 40,
              "remapped PARAM3 preserves the interpolated gradient at PS input 0");
    }

    // GFX10 fixed-function interpolation can replace a missing PARAM export with one of four
    // DEFAULT_VAL vectors. Vulkan has no matching pipeline state, so the vertex recompiler emits
    // the selected constant at the logical PS input location. DEFAULT_VAL=3 is (1,1,1,1): the same
    // PS that produced a gradient above must now see attr0.x=1 at every covered pixel.
    PixelInputMapping default_input;
    default_input.valid_mask = 1u;
    default_input.controls[0] = 0x00000320u; // OFFSET=0x20, DEFAULT_VAL=3 (1111)
    std::vector<uint32_t> default_vert = recompile_vertex(
        vs, sizeof(vs)/sizeof(vs[0]), nullptr, &default_input);
    std::vector<uint8_t> default_px = prosper::test::render_triangle_rgba(
        default_vert, frag, W, H);
    CHECK(default_px.size() == (size_t)W * H * 4,
          "SPI_PS_INPUT_CNTL default produces a linkable VS/PS interface");
    if (default_px.size() == (size_t)W * H * 4) {
        uint8_t default_min = 255, default_max = 0;
        for (size_t i = 0; i < default_px.size(); i += 4) {
            if (default_px[i] < default_min) default_min = default_px[i];
            if (default_px[i] > default_max) default_max = default_px[i];
        }
        CHECK(default_min == 255 && default_max == 255,
              "DEFAULT_VAL=3 materializes attr0=(1,1,1,1), not an unwritten zero varying");
    }

    // DPP quad_perm as screen-space derivatives (#273 — DOLL's manual ddx/ddy idiom). The attribute
    // attr0.x ramps linearly across the fullscreen triangle: PARAM0.x = vid with vertices
    // (-1,-1),(3,-1),(-1,3), so u(x_ndc,y_ndc) = (x_ndc+1)/4 + 2*(y_ndc+1)/4 and per-pixel
    // du/dx = 2/(4W) = 1/128 (W=64), du/dy = 4/(4H) = 1/64. The PS computes the exact live idiom
    //   ddx: v4 = u@qp[0,0,2,2]; v4 = u@qp[1,1,3,3] - v4;  out = 128*ddx -> red == 255
    //   ddy: v3 = u@qp[0,1,0,1]; v3 = u@qp[2,3,2,3] - v3;  out =  64*ddy -> red == 255
    // (words llvm-mc-round-tripped; a broken lowering leaves red 0 or garbage).
    auto dpp_ps_run = [&](std::initializer_list<uint32_t> body, const char* what) {
        std::vector<uint32_t> ps2 = { 0xc8000000u, 0xc8010001u };
        ps2.insert(ps2.end(), body);
        const uint32_t tail[] = { 0x7e040280u, 0x7e0602f2u, 0xf800080fu, 0x03020200u, 0xbf810000u };
        ps2.insert(ps2.end(), tail, tail + 5);
        std::vector<uint32_t> f2 = recompile_fragment(ps2.data(), ps2.size());
        if (f2.empty()) { printf("  [FAIL] %s: recompile rejected\n", what); fails++; return; }
        std::vector<uint8_t> p2 = prosper::test::render_triangle_rgba(vert, f2, W, H);
        if (p2.size() != (size_t)W * H * 4) { printf("  [FAIL] %s: render failed\n", what); fails++; return; }
        const uint8_t* c = &p2[((size_t)(H/2) * W + W/2) * 4];
        printf("  %s center=(%u,%u,%u)\n", what, c[0], c[1], c[2]);
        CHECK(c[0] > 0xE0 && c[1] < 0x20 && c[2] < 0x20, what);
    };
    dpp_ps_run({ 0x7e0802fau, 0xff08a000u,   // v4 = u @ qp[0,0,2,2]
                 0x080808fau, 0xff08f500u,   // v4 = u @ qp[1,1,3,3] - v4  (= ddx)
                 0x100008ffu, 0x43000000u }, // v0 = 128 * ddx
               "DPP ddx idiom reconstructs du/dx (red saturated)");
    dpp_ps_run({ 0x7e0602fau, 0xff084400u,   // v3 = u @ qp[0,1,0,1]
                 0x080606fau, 0xff08ee00u,   // v3 = u @ qp[2,3,2,3] - v3  (= ddy)
                 0x100006ffu, 0x42800000u }, // v0 = 64 * ddy
               "DPP ddy idiom reconstructs du/dy (red saturated)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
