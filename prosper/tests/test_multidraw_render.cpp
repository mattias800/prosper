// test_multidraw_render — the multi-draw backend (render_runner.h render_draws_rgba) records N draws
// into ONE framebuffer (cleared once), each with its own pipeline + blend state. Proof: a red OPAQUE
// fullscreen draw followed by a green ADDITIVE fullscreen draw composites to YELLOW at the center — a
// color that is impossible unless both draws hit the same accumulating target (a fresh clear per draw
// would leave only the last draw's green). This exercises the multi-draw spine independently of the
// game's per-draw register-snapshot resolution.
#include "../src/gpu/rdna2_to_spirv.hpp"
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
    printf("== test_multidraw_render ==\n");
    const uint32_t W = 64, H = 64;

    // Known-good fullscreen-triangle vertex shader (SPIR-V), shared by both draws.
    #include "../tools/boot_trace/refvs.inc"
    std::vector<uint32_t> vs(kRefVs, kRefVs + sizeof(kRefVs) / 4);

    // Two solid-color pixel shaders, recompiled from tiny RDNA2 EXP blobs (v_mov the 4 color VGPRs, then
    // EXP mrt0). Inline consts: 0xF2 = 1.0f, 0x80 = 0.0f.  RED = (1,0,0,1)  GREEN = (0,1,0,1).
    static const uint32_t kRedPs[]   = {0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u};
    static const uint32_t kGreenPs[] = {0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u};
    std::vector<uint32_t> red   = recompile_fragment(kRedPs,   sizeof(kRedPs)   / 4, nullptr);
    std::vector<uint32_t> green = recompile_fragment(kGreenPs, sizeof(kGreenPs) / 4, nullptr);
    CHECK(!vs.empty() && !red.empty() && !green.empty(), "fullscreen VS + red/green PS available");

    ResolvedPipelineState opaque{};
    opaque.topology = 3 /*VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST*/; opaque.color_write_mask = 0xF;
    ResolvedPipelineState additive = opaque;              // green = src*ONE + dst*ONE (accumulate onto red)
    additive.blend_enable = true;
    additive.src_color_blend_factor = 1 /*VK_BLEND_FACTOR_ONE*/;
    additive.dst_color_blend_factor = 1 /*VK_BLEND_FACTOR_ONE*/;
    additive.color_blend_op         = 0 /*VK_BLEND_OP_ADD*/;

    auto center = [&](const std::vector<uint8_t>& px) -> const uint8_t* {
        return px.empty() ? nullptr : &px[((size_t)(H / 2) * W + W / 2) * 4];
    };

    // Single opaque red draw -> red center (baseline; no accumulation).
    {
        prosper::test::BackendDraw d; d.vs = vs; d.fs = red; d.ps = &opaque; d.vcount = 3;
        std::vector<uint8_t> px = prosper::test::render_draws_rgba({d}, W, H);
        CHECK(px.size() == (size_t)W * H * 4, "single-draw path rendered a frame");
        const uint8_t* c = center(px);
        if (c) CHECK(c[0] > 0xC0 && c[1] < 0x40 && c[2] < 0x40, "one opaque red draw -> RED center");
    }

    // Red opaque THEN green additive, in one submit -> yellow center (both composited into one target).
    {
        prosper::test::BackendDraw d0; d0.vs = vs; d0.fs = red;   d0.ps = &opaque;   d0.vcount = 3;
        prosper::test::BackendDraw d1; d1.vs = vs; d1.fs = green; d1.ps = &additive; d1.vcount = 3;
        std::vector<uint8_t> px = prosper::test::render_draws_rgba({d0, d1}, W, H);
        CHECK(px.size() == (size_t)W * H * 4, "two-draw submit rendered a frame");
        const uint8_t* c = center(px);
        if (c) {
            CHECK(c[0] > 0xC0 && c[1] > 0xC0 && c[2] < 0x40,
                  "two draws composite into ONE cleared-once framebuffer -> YELLOW center (red+green)");
        }
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
