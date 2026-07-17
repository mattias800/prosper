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
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

int main() {
    printf("== test_multidraw_render ==\n");
    set_env("PROSPER_RENDER_TIMING", "1");
    const uint32_t W = 64, H = 64;

    // Known-good fullscreen-triangle vertex shader (SPIR-V), shared by both draws.
    #include "../tools/boot_trace/refvs.inc"
    std::vector<uint32_t> vs(kRefVs, kRefVs + sizeof(kRefVs) / 4);
    auto ref_vs_with_depth = [](std::vector<uint32_t> words, uint32_t depth_bits) {
        // kRefVs builds gl_Position.z from float OpConstant %37. Patch only that literal so tests can
        // model the tiny cross-shader position drift seen between UE4's depth and base passes.
        for (size_t i = 5; i < words.size();) {
            const uint32_t word_count = words[i] >> 16;
            const uint32_t opcode = words[i] & 0xffffu;
            if (word_count == 4 && opcode == 43 /* OpConstant */ &&
                words[i + 1] == 6 /* float type */ && words[i + 2] == 0x25) {
                words[i + 3] = depth_bits;
                return words;
            }
            if (!word_count || i + word_count > words.size()) break;
            i += word_count;
        }
        return std::vector<uint32_t>{};
    };

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
        const prosper::test::BackendRenderTimingStats timing =
            prosper::test::backend_render_timing_stats();
        CHECK(timing.calls == 1 && timing.draws == 2,
              "backend timing publishes the completed call and draw count");
        CHECK(timing.total_ms() > 0 && timing.draw_setup_ms > 0,
              "backend timing publishes a non-empty phase breakdown");
        CHECK(timing.draw_setup_ms + 0.001 >= timing.setup_shader_ms + timing.setup_fixed_ms +
                  timing.setup_resources_ms + timing.setup_pipeline_ms,
              "draw-setup subphases fit inside the backend draw-setup phase");
        const prosper::test::BackendPipelineCacheStats first_cache =
            prosper::test::backend_pipeline_cache_stats();
        CHECK(first_cache.references == 2 && first_cache.hits == 1 && first_cache.misses == 1,
              "pipeline cache reuses the prior opaque pipeline but separates blend state");

        std::vector<uint8_t> cached = prosper::test::render_draws_rgba({d0, d1}, W, H);
        const prosper::test::BackendPipelineCacheStats cached_stats =
            prosper::test::backend_pipeline_cache_stats();
        CHECK(cached == px, "persistent pipeline hits preserve multi-draw pixels byte-for-byte");
        CHECK(cached_stats.references == 2 && cached_stats.hits == 2 && cached_stats.misses == 0,
              "repeated draw contracts hit the persistent pipeline cache");

        set_env("PROSPER_NO_BACKEND_PIPELINE_CACHE", "1");
        std::vector<uint8_t> bypassed = prosper::test::render_draws_rgba({d0, d1}, W, H);
        const prosper::test::BackendPipelineCacheStats bypassed_stats =
            prosper::test::backend_pipeline_cache_stats();
        set_env("PROSPER_NO_BACKEND_PIPELINE_CACHE", nullptr);
        CHECK(bypassed == px, "pipeline-cache disable A/B preserves output byte-for-byte");
        CHECK(bypassed_stats.references == 2 && bypassed_stats.bypasses == 2 &&
                  bypassed_stats.hits == 0,
              "pipeline-cache disable A/B bypasses every lookup");
    }

    // A guest depth/stencil surface survives renderer calls. The first call writes stencil=2 with no
    // color; the second call LOADs the same guest-identified attachment and an EQUAL-2 draw turns green.
    // A different identity starts cleared and fails, while an explicit clear on the persisted identity
    // also rejects the reader. This is the cross-submit contract used by Messenger's level masks (#518).
    {
        ResolvedPipelineState writer = opaque;
        writer.color_write_mask = 0;
        writer.stencil_enable = true;
        writer.stencil_compare_op[0] = writer.stencil_compare_op[1] = 7; // ALWAYS
        writer.stencil_pass_op[0] = writer.stencil_pass_op[1] = 2;       // REPLACE
        writer.stencil_op_val[0] = writer.stencil_op_val[1] = 2;
        writer.stencil_write_mask[0] = writer.stencil_write_mask[1] = 0xff;
        writer.stencil_read_base = writer.stencil_write_base = 0x11110000;

        ResolvedPipelineState reader = opaque;
        reader.stencil_enable = true;
        reader.stencil_compare_op[0] = reader.stencil_compare_op[1] = 2; // EQUAL
        reader.stencil_ref[0] = reader.stencil_ref[1] = 2;
        reader.stencil_compare_mask[0] = reader.stencil_compare_mask[1] = 0xff;
        reader.stencil_read_base = reader.stencil_write_base = 0x11110000;

        prosper::test::BackendDraw w; w.vs = vs; w.fs = red; w.ps = &writer; w.vcount = 3;
        prosper::test::BackendDraw r; r.vs = vs; r.fs = green; r.ps = &reader; r.vcount = 3;
        (void)prosper::test::render_draws_rgba({w}, W, H, nullptr, nullptr, true);
        std::vector<uint8_t> loaded = prosper::test::render_draws_rgba({r}, W, H, nullptr, nullptr, true);
        const uint8_t* lc = center(loaded);
        CHECK(lc && lc[1] > 0xC0 && lc[0] < 0x40,
              "persistent guest DS identity LOADs stencil written by an earlier renderer call");

        ResolvedPipelineState fresh = reader;
        fresh.stencil_read_base = fresh.stencil_write_base = 0x22220000;
        prosper::test::BackendDraw f = r; f.ps = &fresh;
        std::vector<uint8_t> rejected = prosper::test::render_draws_rgba({f}, W, H, nullptr, nullptr, true);
        const uint8_t* fc = center(rejected);
        CHECK(fc && fc[1] < 0x80, "new guest DS identity starts cleared and rejects EQUAL-2");

        ResolvedPipelineState cleared = reader;
        cleared.stencil_clear_enable = true;
        cleared.stencil_clear_value = 0;
        prosper::test::BackendDraw c = r; c.ps = &cleared;
        std::vector<uint8_t> after_clear = prosper::test::render_draws_rgba({c}, W, H, nullptr, nullptr, true);
        const uint8_t* cc = center(after_clear);
        CHECK(cc && cc[1] < 0x80, "explicit guest stencil clear executes in order before the draw");
    }

    // Initializing one aspect of a combined D32S8 image must not make the other aspect valid. Unity
    // first uses this surface for stencil while depth is ALWAYS and read-only, then changes to GEQUAL.
    // The later draw must initialize the still-unused depth plane from its reverse-Z clear value (0),
    // rather than LOADing the depth fallback (1) used while only stencil contents mattered (#540).
    {
        ResolvedPipelineState stencil_first = opaque;
        stencil_first.color_write_mask = 0;
        stencil_first.depth_test_enable = true;
        stencil_first.depth_compare_op = 7; // ALWAYS
        stencil_first.depth_clear_value = 1.0f;
        stencil_first.depth_read_base = stencil_first.depth_write_base = 0x33330000;
        stencil_first.stencil_enable = true;
        stencil_first.stencil_compare_op[0] = stencil_first.stencil_compare_op[1] = 7; // ALWAYS
        stencil_first.stencil_pass_op[0] = stencil_first.stencil_pass_op[1] = 2;       // REPLACE
        stencil_first.stencil_op_val[0] = stencil_first.stencil_op_val[1] = 1;
        stencil_first.stencil_read_base = stencil_first.stencil_write_base = 0x44440000;

        ResolvedPipelineState depth_later = opaque;
        depth_later.depth_test_enable = true;
        depth_later.depth_compare_op = 6; // GEQUAL
        depth_later.depth_clear_value = 0.0f;
        depth_later.depth_read_base = depth_later.depth_write_base = 0x33330000;
        depth_later.stencil_read_base = depth_later.stencil_write_base = 0x44440000;

        prosper::test::BackendDraw s; s.vs = vs; s.fs = red; s.ps = &stencil_first; s.vcount = 3;
        prosper::test::BackendDraw d; d.vs = vs; d.fs = green; d.ps = &depth_later; d.vcount = 3;
        (void)prosper::test::render_draws_rgba({s}, W, H, nullptr, nullptr, true);
        std::vector<uint8_t> initialized =
            prosper::test::render_draws_rgba({d}, W, H, nullptr, nullptr, true);
        const uint8_t* c = center(initialized);
        CHECK(c && c[1] > 0xC0 && c[0] < 0x40,
              "stencil-only initialization does not poison a later reverse-Z depth plane");
    }

    // Programming DB_DEPTH_CLEAR only updates clear state; it does not clear a depth allocation until
    // DB_RENDER_CONTROL enables the clear. Treating a stale programmed word as the implicit contents of
    // a newly-created host image can reject every fragment. Astro Bot exposes this with 0x0437077f --
    // its packed 1919x1079 surface maxima, interpreted as the tiny float 2.15e-36 -- under LEQUAL.
    {
        std::vector<uint32_t> mid_depth_vs = ref_vs_with_depth(vs, 0x3f000000u); // 0.5f
        CHECK(!mid_depth_vs.empty(), "fullscreen VS can expose stale depth-clear initialization");

        ResolvedPipelineState stale = opaque;
        stale.depth_test_enable = true;
        stale.depth_compare_op = 3; // LEQUAL
        stale.has_depth_clear = true;
        uint32_t packed_extent = 0x0437077fu;
        std::memcpy(&stale.depth_clear_value, &packed_extent, sizeof(packed_extent));
        stale.depth_read_base = stale.depth_write_base = 0x77770000;

        prosper::test::BackendDraw s;
        s.vs = mid_depth_vs; s.fs = green; s.ps = &stale; s.vcount = 3;
        std::vector<uint8_t> implicit =
            prosper::test::render_draws_rgba({s}, W, H, nullptr, nullptr, true);
        const uint8_t* ic = center(implicit);
        CHECK(ic && ic[1] > 0xC0 && ic[0] < 0x40,
              "stale DB_DEPTH_CLEAR does not initialize an uncleared LEQUAL surface");

        ResolvedPipelineState explicit_clear = stale;
        explicit_clear.depth_clear_enable = true;
        explicit_clear.depth_read_base = explicit_clear.depth_write_base = 0x88880000;
        prosper::test::BackendDraw e = s; e.ps = &explicit_clear;
        std::vector<uint8_t> explicitly_rejected =
            prosper::test::render_draws_rgba({e}, W, H, nullptr, nullptr, true);
        const uint8_t* ec = center(explicitly_rejected);
        CHECK(ec && ec[1] < 0x80,
              "explicit DB_RENDER_CONTROL clear still consumes the programmed depth value");
    }

    // UE4 can emit its reverse-Z depth prepass and EQUAL base pass from different shaders. Their
    // translated clip-space positions may differ by one float ULP. On the guest this pair shades;
    // strict Vulkan EQUAL rejects it. Relax only a read-only EQUAL on an already-valid, explicitly
    // reverse-Z guest surface to GEQUAL, which continues rejecting fragments behind the prepass.
    {
        std::vector<uint32_t> depth_vs = ref_vs_with_depth(vs, 0x3f000000u); // 0.5f
        std::vector<uint32_t> drift_vs = ref_vs_with_depth(vs, 0x3f000001u); // next float above 0.5f
        CHECK(!depth_vs.empty() && !drift_vs.empty(), "fullscreen VS depth literal is patchable");

        ResolvedPipelineState writer = opaque;
        writer.color_write_mask = 0;
        writer.depth_test_enable = true;
        writer.depth_write_enable = true;
        writer.depth_compare_op = 7; // ALWAYS
        writer.has_depth_clear = true;
        writer.depth_clear_value = 0.0f;
        writer.depth_read_base = writer.depth_write_base = 0x55550000;

        ResolvedPipelineState reader = opaque;
        reader.depth_test_enable = true;
        reader.depth_compare_op = 2; // EQUAL
        reader.has_depth_clear = true;
        reader.depth_clear_value = 0.0f;
        reader.depth_read_base = reader.depth_write_base = 0x55550000;

        prosper::test::BackendDraw w; w.vs = depth_vs; w.fs = red; w.ps = &writer; w.vcount = 3;
        prosper::test::BackendDraw r; r.vs = drift_vs; r.fs = green; r.ps = &reader; r.vcount = 3;
        (void)prosper::test::render_draws_rgba({w}, W, H, nullptr, nullptr, true);
        std::vector<uint8_t> compatible =
            prosper::test::render_draws_rgba({r}, W, H, nullptr, nullptr, true);
        const uint8_t* rc = center(compatible);
        CHECK(rc && rc[1] > 0xC0 && rc[0] < 0x40,
              "persistent reverse-Z EQUAL tolerates one-ULP translated shader drift");

        ResolvedPipelineState fresh = reader;
        fresh.depth_read_base = fresh.depth_write_base = 0x66660000;
        prosper::test::BackendDraw f = r; f.ps = &fresh;
        std::vector<uint8_t> uninitialized =
            prosper::test::render_draws_rgba({f}, W, H, nullptr, nullptr, true);
        const uint8_t* fc = center(uninitialized);
        CHECK(fc && fc[1] < 0x80,
              "reverse-Z EQUAL stays strict before the guest depth surface is populated");

        ResolvedPipelineState standard_z = reader;
        standard_z.depth_clear_value = 1.0f;
        prosper::test::BackendDraw s = r; s.ps = &standard_z;
        std::vector<uint8_t> strict =
            prosper::test::render_draws_rgba({s}, W, H, nullptr, nullptr, true);
        const uint8_t* sc = center(strict);
        CHECK(sc && sc[1] < 0x80, "standard-Z EQUAL remains exact");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
