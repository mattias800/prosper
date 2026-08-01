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

    // VCC_LO/HI are architectural mask halves, but scalar instructions may also use either encoding
    // as an ordinary 32-bit scratch SGPR. Preserve that full scalar value in a fragment shader when
    // both inputs are scalar data, while still projecting its addressed lane bit into VCC. This is
    // the opening shape of Plucky Squire's captured 1x1 tonemap shader: loop_index & 3 is written to
    // VCC_LO and immediately consumed by s_cmp_eq_u32 as a complete dword.
    const uint32_t scalar_vcc_ps[] = {
        0xBE920385u,                         // s_mov_b32 s18, 5
        0x876A8312u,                         // s_and_b32 vcc_lo, s18, 3  -> 1
        0xBF066A81u,                         // s_cmp_eq_u32 1, vcc_lo   -> SCC=1
        0x850080F2u,                         // s_cselect_b32 s0, 1.0, 0.0
        0x7E000200u,                         // v_mov_b32 v0, s0
        0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    const auto scalar_vcc_frag = recompile_fragment(
        scalar_vcc_ps, std::size(scalar_vcc_ps));
    CHECK(!scalar_vcc_frag.empty(),
          "fragment scalar write/read through VCC_LO recompiles");
    if (!scalar_vcc_frag.empty()) {
        const auto scalar_vcc_px = prosper::test::render_triangle_rgba(
            vert, scalar_vcc_frag, W, H);
        const uint8_t* p = scalar_vcc_px.empty() ? nullptr
            : &scalar_vcc_px[((size_t)(H / 2) * W + W / 2) * 4];
        // #1681: this fixture writes VCC_LO from a scalar ALU chain, which the recompiler lowers
        // through a native wave64 subgroup vote, so the module requires an exact 64-lane fragment
        // subgroup. RADV enforces it; llvmpipe is fixed at 8 and the backend correctly rejects the
        // draw rather than run wrong wave semantics, leaving the BLUE clear. Gate the rendered-pixel
        // assertion on the capability the same way test_recompiled_fragment already does.
        const auto& ctx = prosper::test::render_vk_ctx();
        const bool supports_fragment_wave64_vote = ctx.subgroup_size_control &&
            ctx.min_subgroup_size <= 64 && ctx.max_subgroup_size >= 64 &&
            (ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
            (ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
            (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) &&
            (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_VOTE_BIT);
        if (p) printf("  scalar-vcc center=(%u,%u,%u,%u) wave64-vote=%d\n",
                      p[0], p[1], p[2], p[3], static_cast<int>(supports_fragment_wave64_vote));
        // The skipped arm pairs the clear colour with the module's own declared requirement, so it
        // asserts "this shader demanded wave64 and the device therefore left the target untouched"
        // rather than the weaker "no draw landed here", which any pipeline failure would satisfy.
        CHECK(p && (supports_fragment_wave64_vote
                        ? (p[0] > 0x80 && p[1] < 0x40 && p[2] < 0x40)
                        : (fragment_spirv_required_subgroup_size(scalar_vcc_frag) == 64 &&
                           p[2] > 0x80 && p[0] < 0x40 && p[1] < 0x40)),
              supports_fragment_wave64_vote
                  ? "fragment scalar VCC_LO dword reaches compare/select exactly"
                  : "device without fragment wave64 vote skips the scalar-VCC draw fail-visible");
    }

    // An NGG hardware vertex program uses LDS even when its logical output is an ordinary vertex.
    // The portable shell represents one live guest lane, so its RSRC2-sized LDS is private Function
    // memory and barriers are already synchronous.  Round-trip all four position components through
    // exact gfx10 DS_WRITE2/READ2 words; omitting the hardware allocation must remain fail-closed.
    const uint32_t lds_vs[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u,
        0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u, 0x08020D01u,
        0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u,
        0x7E0A0280u,
        0xD8380100u, 0x00020105u,       // ds_write2_b32 v5, v1, v2 offset0:0 offset1:1
        0xD8380302u, 0x00040305u,       // ds_write2_b32 v5, v3, v4 offset0:2 offset1:3
        0xBF8A0000u,                    // s_barrier
        0xD8DC0100u, 0x01000005u,       // ds_read2_b32 v[1:2], v5 offset0:0 offset1:1
        0xD8DC0302u, 0x03000005u,       // ds_read2_b32 v[3:4], v5 offset0:2 offset1:3
        0xBF8A0000u,
        0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    CHECK(recompile_vertex(lds_vs, std::size(lds_vs)).empty(),
          "graphics LDS rejects without an RSRC2_GS allocation");
    const auto lds_vert = recompile_vertex(
        lds_vs, std::size(lds_vs), nullptr, nullptr, false, 4);
    CHECK(lds_vert.empty(),
          "an allocation alone cannot authorize an unproven graphics-LDS projection");
    std::vector<uint32_t> mask_count_vs = {
        0x7D8A10F9u, 0x06868680u,       // v_cmp_* s[6:7], ... (whole-wave mask)
        0xBEEB1006u,                    // s_bcnt1_i32_b64 vcc_hi, s[6:7]
    };
    mask_count_vs.insert(mask_count_vs.end(), std::begin(vs), std::end(vs));
    CHECK(recompile_vertex(mask_count_vs.data(), mask_count_vs.size()).empty(),
          "vertex bcnt1 rejects an unproven whole-wave VOPC population count");
    std::vector<uint32_t> bounded_dpp_vs = {
        0x4A1412FAu, 0xFF091109u,       // v_add_nc_u32 v10, v9 row_shr:1 bound_ctrl:1, v9
        0x4A1414FAu, 0xFF09120Au,       // v_add_nc_u32 v10, v10 row_shr:2 bound_ctrl:1, v10
    };
    bounded_dpp_vs.insert(bounded_dpp_vs.end(), std::begin(vs), std::end(vs));
    CHECK(recompile_vertex(bounded_dpp_vs.data(), bounded_dpp_vs.size()).empty(),
          "bounded vertex ROW_SHR remains closed outside a proven one-lane projection");
    std::vector<uint32_t> projected_readlane_vs = {
        0x7E140281u,                         // v_mov_b32 v10, 1
        0x816A8102u,                         // s_add_i32 vcc_lo, s2, 1 (dynamic lane selector)
        0xD760006Au, 0x0000D50Au,            // v_readlane_b32 vcc_lo, v10, vcc_lo
        0xD7600005u, 0x0001070Au,            // v_readlane_b32 s5, v10, 3
    };
    projected_readlane_vs.insert(projected_readlane_vs.end(), std::begin(vs), std::end(vs));
    CHECK(recompile_vertex(projected_readlane_vs.data(), projected_readlane_vs.size()).empty(),
          "ordinary VGPR readlane remains closed outside a proven one-lane projection");
    std::vector<uint32_t> unbounded_dpp_vs = {
        0x4A0E0CFAu, 0xFF011106u,       // row_shr:1 without BOUND_CTRL retains old v7 on lane 0
    };
    unbounded_dpp_vs.insert(unbounded_dpp_vs.end(), std::begin(vs), std::end(vs));
    CHECK(recompile_vertex(unbounded_dpp_vs.data(), unbounded_dpp_vs.size()).empty(),
          "NGG vertex ROW_SHR without BOUND_CTRL remains fail-closed");

    // --- NGG vertex shader (the game's shaders 004/025 pattern) ---------------------------------------
    // An NGG VS wraps the vertex work in wave-mask/packing plumbing (s_bfe_u64 EXEC/VCC, s_sendmsg,
    // exp prim, s_lshr_b64 exec) and carries the vertex index in v5. The BFE prefix narrows and then
    // restores EXEC through both architectural mask destinations; each Vulkan vertex invocation is
    // the one live guest lane. The body computes pos = (((v5<<1)&2)-1, (v5&-2)-1, 0, 1): v5=0,1,2 ->
    // (-1,-1),(1,-1),(-1,1), a triangle over the lower-left half of NDC. This verifies real graphics-
    // pipeline execution, not merely that the generated SPIR-V is structurally accepted.
    const uint32_t nggvs[] = {
        0xBEEA03FFu, 0x00080000u, 0x94FE6AC1u, 0xBEFE04C1u,
        0xBEEA03FFu, 0x00080000u, 0x94EA6AC1u, 0xBEFE046Au, 0xBEFE04C1u,
        0xD7650007u, 0x000100C1u, 0xD7660009u, 0x00020EC1u,
        0x93EAFF03u, 0x00080008u, 0x876BFF03u, 0x000000FFu, 0x8F6A8C6Au, 0x887C6A6Bu, 0xBF900009u,
        0x906A8803u, 0x81EA6A80u, 0x90FE6AC1u, 0xF8000941u, 0x00000000u, 0x81EA0380u, 0x90FE6AC1u,
        0x34040A81u, 0x36060AC2u, 0x7E000280u, 0x7E0202F2u, 0x36040482u, 0x4A0606C1u, 0x4A0404C1u,
        0x7E060B03u, 0x7E040B02u, 0xF80008CFu, 0x01000302u, 0xBF810000u,
    };
    std::vector<uint32_t> nvert = recompile_vertex(nggvs, sizeof(nggvs)/sizeof(nggvs[0]));
    CHECK(nvert.empty(), "an unproven NGG wave projection remains fail-closed");

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
        0x7E280281u, 0x7E2A0280u,
        0x7C3E2B14u,                         // v_cmpx_tru_f32 -> every invocation active
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
        0x7C202B14u,                         // v_cmpx_f_f32 -> every invocation inactive
        0xBF880002u,
        0xF80008CFu, 0x01000302u,
        0xBF810000u,
    };
    const auto active_gate_spv = recompile_vertex_terminal_ngg_gate_for_test(
        ngg_gate_active, std::size(ngg_gate_active));
    const auto inactive_gate_spv = recompile_vertex_terminal_ngg_gate_for_test(
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

    // A separately-installed fetch prolog receives its continuation in s[6:7]. Increment v5 before
    // the transfer so the main program above emits the opposite half-triangle; this proves that the
    // two allocations execute as one register-preserving program rather than merely compiling the
    // main shader in isolation. Trailing CODE_END padding is outside the executable prefix.
    const uint32_t ngg_prolog[] = {
        0xBFA00003u,             // s_setprio 3
        0x4A0A0A81u,             // v_add_nc_u32 v5, 1, v5
        0xBE802006u,             // s_setpc_b64 s[6:7]
        0xBF9F0000u, 0xBF9F0000u,
    };
    const VertexPrologInfo prolog_info =
        rdna2_vertex_prolog_info(ngg_prolog, std::size(ngg_prolog));
    CHECK(prolog_info.valid && prolog_info.setpc_pc == 2 && prolog_info.prefix_dwords == 2,
          "separate vertex prolog recognizes only its bounded terminal s[6:7] transfer");
    std::vector<uint32_t> chained = recompile_vertex_chain(
        ngg_prolog, std::size(ngg_prolog), nggvs, std::size(nggvs));
    CHECK(chained.empty(),
          "a recognized split program still rejects unproven cross-lane NGG semantics");

    // Compiler-generated no-GS NGG wrappers communicate between guest lanes through LDS.  A Vulkan
    // vertex shader cannot reproduce that with Function-private LDS, but it also does not need to:
    // the host draw already launches exactly the logical vertices and assembles their primitives.
    // This split program writes {PARAM0.xy, POS0.xyzw} as one producer record; the wrapper's terminal
    // exports read the same fields from a compacted record at a different LDS base.  The chain
    // recognizer must prove the matching offsets and export the producer record directly.
    const uint32_t passthrough_producer[] = {
        0xD765000Au, 0x000100C1u,            // v_mbcnt_lo v10, -1, 0
        0xD766000Au, 0x000214C1u,            // v_mbcnt_hi v10, -1, v10
        0x34040A81u, 0x36060AC2u, 0x7E000280u, 0x7E0202F2u,
        0x36040482u, 0x4A0606C1u, 0x4A0404C1u, 0x7E060B03u, 0x7E040B02u,
        0x7E0C0280u,                         // v6 = record byte address 0 (private per invocation)
        0x7E0E0280u, 0x7E1002F2u,            // PARAM0.xy = (0, 1)
        0xD8380100u, 0x00080706u,            // record dword 0/1 = v7/v8
        0xD8380302u, 0x00030206u,            // record dword 2/3 = position x/y
        0xD8380504u, 0x00010006u,            // record dword 4/5 = position z/w
        0x7E12030Au,                         // v9 = v10 (make the MBCNT lane observable as PARAM1.x)
        0xD8340018u, 0x00000906u,            // record dword 6 = PARAM1.x
        0xBF8A0000u,
        0xBE802006u,                         // transfer to separately installed wrapper
    };
    const uint32_t passthrough_wrapper[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (no-GS NGG allocation)
        0xF8000941u, 0x00000000u,            // primitive export
        0xD5430000u, 0x03FE249Cu, 0x00000728u, // 28*v18 + compacted base + 8
        0xD5430002u, 0x03FE249Cu, 0x00000730u, // 28*v18 + compacted base + 16
        0xD8DC0100u, 0x00000000u,            // POS0.xy
        0xD8DC0100u, 0x02000002u,            // POS0.zw
        0xF80000CFu, 0x03020100u,            // exp pos0 v[0:3]
        0xD5430000u, 0x03FE249Cu, 0x00000720u, // 28*v18 + compacted base
        0xD8DC0100u, 0x00000000u,            // PARAM0.xy
        0xF8000203u, 0x00000100u,            // exp param0 v[0:1]
        0x1608249Cu,                         // v_mul_u32_u24 v4, 28, v18
        0xD8D80738u, 0x04000004u,            // v4 = LDS[28*v18 + 0x738]
        0xF8000211u, 0x00000004u,            // exp param1.x v4
        0xBF810000u,
    };
    ShaderResourceTable passthrough_rt;
    passthrough_rt.vertices_per_instance = 3;
    const auto passthrough_vert = recompile_vertex_chain(
        passthrough_producer, std::size(passthrough_producer),
        passthrough_wrapper, std::size(passthrough_wrapper),
        &passthrough_rt, nullptr, false, 7);
    CHECK(!passthrough_vert.empty(),
          "split no-GS NGG producer/wrapper accepts MAD and mul-plus-DS record addresses");
    if (!passthrough_vert.empty()) {
        const auto ppx = prosper::test::render_triangle_rgba(passthrough_vert, frag, W, H);
        auto grn = [&](uint32_t x, uint32_t y) {
            const uint8_t* p = &ppx[((size_t)y * W + x) * 4];
            return p[1] > 0x80 && p[0] < 0x40 && p[2] < 0x40;
        };
        CHECK(ppx.size() == (size_t)W * H * 4 && grn(16, 16) && !grn(60, 60),
              "no-GS NGG passthrough preserves logical vertex geometry across all host invocations");

        // Consume flat PARAM1.x in the fragment shader, convert its integer lane value to red, and
        // select each triangle's provoking vertex by rotating its indices. Repeated instances then
        // exercise both MBCNT halves at 31/32/63 and prove that the next guest wave wraps to lane 0.
        const uint32_t lane_ps[] = {
            0xC8020402u,                         // v_interp_mov_f32 v0, p0, attr1.x
            0x7E000D00u,                         // v_cvt_f32_u32 v0, v0
            0x100000FFu, 0x3C800000u,            // v_mul_f32 v0, 1/64, v0
            0x7E020280u, 0x7E040280u, 0x7E0602F2u,
            0xF800180Fu, 0x03020100u, 0xBF810000u,
        };
        const auto lane_frag = recompile_fragment(lane_ps, std::size(lane_ps));
        CHECK(!lane_frag.empty(), "flat NGG logical-lane varying is consumed by the fragment stage");
        auto rendered_lane_red = [&](uint32_t instances,
                                     std::vector<uint32_t> indices) -> int {
            prosper::test::BackendDraw draw;
            draw.vs = passthrough_vert;
            draw.fs = lane_frag;
            draw.indices = std::move(indices);
            draw.instance_count = instances;
            const auto pixels = prosper::test::render_draws_rgba({std::move(draw)}, W, H);
            return pixels.size() == static_cast<size_t>(W) * H * 4
                ? pixels[((size_t)16 * W + 16) * 4] : -1;
        };
        auto lane_matches = [&](uint32_t lane, uint32_t instances,
                                std::vector<uint32_t> indices) {
            const int red = rendered_lane_red(instances, std::move(indices));
            const int expected = static_cast<int>((lane * 255u + 32u) / 64u);
            return red >= expected - 2 && red <= expected + 2;
        };
        CHECK(!lane_frag.empty() && lane_matches(31, 11, {1, 2, 0}),
              "NGG MBCNT exposes logical lane 31 at the low-half boundary");
        CHECK(!lane_frag.empty() && lane_matches(32, 11, {2, 0, 1}),
              "NGG MBCNT combines low/high halves into logical lane 32");
        CHECK(!lane_frag.empty() && lane_matches(63, 22, {0, 1, 2}),
              "NGG MBCNT combines low/high halves into logical lane 63");
        CHECK(!lane_frag.empty() && lane_matches(0, 22, {1, 2, 0}),
              "NGG MBCNT wraps the first invocation of the second guest wave to lane 0");
    }
    std::vector<uint32_t> masked_lane_producer(
        std::begin(passthrough_producer), std::end(passthrough_producer));
    masked_lane_producer[1] = 0x00010081u;            // v_mbcnt_lo v10, 1, 0
    const auto masked_lane_vert = recompile_vertex_chain(
        masked_lane_producer.data(), masked_lane_producer.size(),
        passthrough_wrapper, std::size(passthrough_wrapper), nullptr, nullptr, false, 7);
    CHECK(masked_lane_vert.empty(),
          "no-GS logical-lane projection rejects non-canonical MBCNT masks");
    const uint32_t arbitrary_jump[] = {0xBFA00003u, 0xBE802008u};
    CHECK(!rdna2_vertex_prolog_info(arbitrary_jump, std::size(arbitrary_jump)).valid,
          "arbitrary indirect s_setpc targets remain rejected");
    const uint32_t escaping_branch[] = {
        0xBF820002u,             // s_branch pc3, beyond the transfer at pc1
        0xBE802006u,             // s_setpc_b64 s[6:7]
        0xBF9F0000u, 0xBF9F0000u,
    };
    CHECK(!rdna2_vertex_prolog_info(escaping_branch, std::size(escaping_branch)).valid,
          "vertex prolog branches into discarded padding remain rejected");
    const uint32_t vcc_data_prolog[] = {
        0x7D840100u,             // v_cmp_eq_u32 vcc, v0, v0 (true)
        0x887C6A80u,             // s_or_b32 m0, 0, vcc_lo
        0x886B6A80u,             // s_or_b32 vcc_hi, 0, vcc_lo (cross-half scalar plumbing)
        0xBE802006u,             // s_setpc_b64 s[6:7]
    };
    const auto vcc_data_chain = recompile_vertex_chain(
        vcc_data_prolog, std::size(vcc_data_prolog), nggvs, std::size(nggvs));
    CHECK(vcc_data_chain.empty(),
          "split-stage scalar mask plumbing does not bypass the NGG wave proof");
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
