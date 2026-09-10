// test_rect_synthesis_render -- a PS5 RectList supplies THREE corners of a rectangle and the GPU
// derives the fourth after vertex shading. Vulkan has no RectList topology, so
// recompile_interpolation_geometry(..., synthesize_rect=true) emits a geometry stage that derives it
// and outputs a four-vertex strip. This test renders that stage and asserts the whole rectangle is
// covered, with the synthesized corner carrying the right varying.
//
// WHICH of the three supplied vertices is the "shared" corner -- the one adjacent to both others --
// is a property of the draw, not a fixed index. #3507: the stage used to assume it was always v0
// (fourth = P1 + P2 - P0, emitted in the supplied order). Darksiders II's video blit shares at v2, so
// its fourth corner landed off-screen and the FMV rendered as a single triangle.
//
// The three arms below submit ONE rectangle three times, permuting the supplied order with an index
// buffer so the shared corner lands at slot 0, 1 and 2 in turn. All three describe the same rectangle
// and must render identically. Arm 0 is the positive control: it is the case the old code got right,
// so it passes either way and shows the harness itself works. Arms 1 and 2 are the discriminator --
// under the old rule both synthesize an off-screen corner and lose most of the coverage.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/render_runner.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// A vertex shader selecting one of three rectangle corners by gl_VertexIndex, exporting the position
// and one varying at Location 0 (the EXP PARAM0 slot prosper's recompiler uses, so the generated
// geometry stage and the recompiled pixel shader below link to it unchanged).
//
//   index 0 -> clip (-1,-1) top-left      varying.x 0.2     <- the shared corner
//   index 1 -> clip ( 1,-1) top-right     varying.x 0.5
//   index 2 -> clip (-1, 1) bottom-left   varying.x 0.7
//
// The fourth corner is therefore clip (1,1) bottom-right, and because the varying is affine over a
// parallelogram its value there is pinned to 0.5 + 0.7 - 0.2 = 1.0 exactly. That makes the varying a
// discriminator too: it is different for every candidate corner rule, so a stage that completes the
// rectangle geometrically but carries the wrong varying still fails.
//
// Assembled from SPIR-V assembly with spirv-as; the source is kept beside this array in
// tests/gpu/recompiler/rect_synthesis_vs.spvasm so it can be regenerated and re-read.
static const uint32_t kRectCornerVs[] = {
        0x07230203u, 0x00010600u, 0x00070000u, 0x00000027u, 0x00000000u, 0x00020011u,
        0x00000001u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu,
        0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0008000fu, 0x00000000u,
        0x00000002u, 0x6e69616du, 0x00000000u, 0x00000003u, 0x00000004u, 0x00000005u,
        0x00040047u, 0x00000003u, 0x0000000bu, 0x0000002au, 0x00050048u, 0x00000006u,
        0x00000000u, 0x0000000bu, 0x00000000u, 0x00030047u, 0x00000006u, 0x00000002u,
        0x00040047u, 0x00000005u, 0x0000001eu, 0x00000000u, 0x00020013u, 0x00000007u,
        0x00030021u, 0x00000008u, 0x00000007u, 0x00030016u, 0x00000009u, 0x00000020u,
        0x00040017u, 0x0000000au, 0x00000009u, 0x00000004u, 0x00040015u, 0x0000000bu,
        0x00000020u, 0x00000001u, 0x00020014u, 0x0000000cu, 0x00040020u, 0x0000000du,
        0x00000001u, 0x0000000bu, 0x0004003bu, 0x0000000du, 0x00000003u, 0x00000001u,
        0x0003001eu, 0x00000006u, 0x0000000au, 0x00040020u, 0x0000000eu, 0x00000003u,
        0x00000006u, 0x0004003bu, 0x0000000eu, 0x00000004u, 0x00000003u, 0x00040020u,
        0x0000000fu, 0x00000003u, 0x0000000au, 0x0004003bu, 0x0000000fu, 0x00000005u,
        0x00000003u, 0x0004002bu, 0x0000000bu, 0x00000010u, 0x00000000u, 0x0004002bu,
        0x0000000bu, 0x00000011u, 0x00000001u, 0x0004002bu, 0x00000009u, 0x00000012u,
        0x00000000u, 0x0004002bu, 0x00000009u, 0x00000013u, 0x3f800000u, 0x0004002bu,
        0x00000009u, 0x00000014u, 0xbf800000u, 0x0004002bu, 0x00000009u, 0x00000015u,
        0x3e4ccccdu, 0x0004002bu, 0x00000009u, 0x00000016u, 0x3f000000u, 0x0004002bu,
        0x00000009u, 0x00000017u, 0x3f333333u, 0x0007002cu, 0x0000000au, 0x00000018u,
        0x00000014u, 0x00000014u, 0x00000012u, 0x00000013u, 0x0007002cu, 0x0000000au,
        0x00000019u, 0x00000013u, 0x00000014u, 0x00000012u, 0x00000013u, 0x0007002cu,
        0x0000000au, 0x0000001au, 0x00000014u, 0x00000013u, 0x00000012u, 0x00000013u,
        0x0007002cu, 0x0000000au, 0x0000001bu, 0x00000015u, 0x00000012u, 0x00000012u,
        0x00000013u, 0x0007002cu, 0x0000000au, 0x0000001cu, 0x00000016u, 0x00000012u,
        0x00000012u, 0x00000013u, 0x0007002cu, 0x0000000au, 0x0000001du, 0x00000017u,
        0x00000012u, 0x00000012u, 0x00000013u, 0x00050036u, 0x00000007u, 0x00000002u,
        0x00000000u, 0x00000008u, 0x000200f8u, 0x0000001eu, 0x0004003du, 0x0000000bu,
        0x0000001fu, 0x00000003u, 0x000500aau, 0x0000000cu, 0x00000020u, 0x0000001fu,
        0x00000010u, 0x000500aau, 0x0000000cu, 0x00000021u, 0x0000001fu, 0x00000011u,
        0x000600a9u, 0x0000000au, 0x00000022u, 0x00000021u, 0x00000019u, 0x0000001au,
        0x000600a9u, 0x0000000au, 0x00000023u, 0x00000020u, 0x00000018u, 0x00000022u,
        0x000600a9u, 0x0000000au, 0x00000024u, 0x00000021u, 0x0000001cu, 0x0000001du,
        0x000600a9u, 0x0000000au, 0x00000025u, 0x00000020u, 0x0000001bu, 0x00000024u,
        0x00050041u, 0x0000000fu, 0x00000026u, 0x00000004u, 0x00000010u, 0x0003003eu,
        0x00000026u, 0x00000023u, 0x0003003eu, 0x00000005u, 0x00000025u, 0x000100fdu,
        0x00010038u,
};

int main() {
    printf("== test_rect_synthesis_render ==\n");
    const uint32_t W = 64, H = 64;

    // PS: v_interp attr0.x -> v0; exp mrt0 (v0, 0, 0, 1). Red therefore reads back the varying.
    const uint32_t ps[] = {
        0xc8000000u, 0xc8010001u, 0x7e020280u, 0x7e0402f2u, 0xf800080fu, 0x02010100u, 0xbf810000u,
    };
    const FragmentInterpolationLayout layout = fragment_interpolation_layout(ps, std::size(ps));
    const std::vector<uint32_t> frag = recompile_fragment(ps, std::size(ps));
    const std::vector<uint32_t> geom =
        recompile_interpolation_geometry(layout, /*capture_position=*/false,
                                         /*synthesize_rect=*/true);
    CHECK(layout.valid && (layout.attribute_mask & 1u),
          "the pixel shader's VINTRP read is seen as a smooth attribute 0");
    CHECK(!frag.empty() && !geom.empty() && geom[0] == 0x07230203u,
          "rect synthesis generates a geometry stage even though the PS needs no P0/P10/P20");
    if (frag.empty() || geom.empty()) { printf("== FAIL ==\n"); return 1; }

    // Expected varying at each screen corner, from the three supplied values plus the derived one.
    struct Corner { const char* name; uint32_t x, y; double expect; };
    const Corner corners[] = {
        {"top-left",     4,      4,      0.2},
        {"top-right",    W - 5,  4,      0.5},
        {"bottom-left",  4,      H - 5,  0.7},
        {"bottom-right", W - 5,  H - 5,  1.0},   // the SYNTHESIZED corner
    };

    struct Arm { const char* name; std::vector<uint32_t> indices; };
    const Arm arms[] = {
        {"shared corner at slot 0 (positive control -- the case the old rule got right)", {0, 1, 2}},
        {"shared corner at slot 1", {1, 0, 2}},
        {"shared corner at slot 2 (Darksiders II's video blit)", {1, 2, 0}},
    };

    for (const Arm& arm : arms) {
        ResolvedPipelineState state;
        state.topology = 3;   // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST -- one triangle in, a strip out
        prosper::test::BackendDraw draw;
        draw.vs.assign(std::begin(kRectCornerVs), std::end(kRectCornerVs));
        draw.gs = geom;
        draw.fs = frag;
        draw.ps = &state;
        draw.vcount = 3;
        draw.indices = arm.indices;
        std::vector<prosper::test::BackendDraw> draws;
        draws.push_back(std::move(draw));
        const std::vector<uint8_t> px = prosper::test::render_draws_rgba(draws, W, H);
        if (px.size() != (size_t)W * H * 4) {
            printf("  [FAIL] %s: render produced no image\n", arm.name);
            fails++;
            continue;
        }
        auto red_at = [&](uint32_t x, uint32_t y) { return px[((size_t)y * W + x) * 4]; };

        // Coverage. Every supplied varying is >= 0.2, so any covered pixel reads red >= ~51 while the
        // cleared background is 0. A fourth corner sent off-screen drops one half of the strip.
        uint32_t covered = 0, sampled = 0;
        uint8_t weakest = 255;
        for (uint32_t y = 2; y < H - 2; ++y) for (uint32_t x = 2; x < W - 2; ++x) {
            ++sampled;
            const uint8_t red = red_at(x, y);
            if (red >= 30) { ++covered; weakest = std::min(weakest, red); }
        }
        printf("  %s: covered %u/%u, weakest covered red %u\n",
               arm.name, covered, sampled, weakest);
        CHECK(covered == sampled,
              (std::string("the synthesized corner completes the rectangle -- full coverage, ") +
               arm.name).c_str());

        for (const Corner& corner : corners) {
            const double got = red_at(corner.x, corner.y) / 255.0;
            const bool ok = std::abs(got - corner.expect) < 0.06;
            if (!ok)
                printf("    %s corner: varying %.3f, expected %.3f\n",
                       corner.name, got, corner.expect);
            CHECK(ok, (std::string("varying is carried to the ") + corner.name + " corner, " +
                       arm.name).c_str());
        }
    }

    printf(fails ? "== FAIL ==\n" : "== PASS ==\n");
    return fails ? 1 : 0;
}
