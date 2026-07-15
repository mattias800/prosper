// test_gpu_execute — the GPU-executor core spine (Stage A): a GpuState (exactly what SubmitDcb folds a
// Dcb into) -> execute_gpustate() [recompile shaders from their PGM addresses + resolve pipeline] ->
// a caller-supplied Vulkan render -> present_write_frame -> present_readback. Proves the executor entry
// point that agc_driver_submit_dcb will call, and the scanout round-trip, end to end on llvmpipe.
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/videoout_present.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include "../src/gpu/vk_translate.hpp"
#include "render_runner.h"
#include <cstdio>
#include <cstdint>
#include <memory>
#include <vector>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Fullscreen-triangle VS + solid-green PS (llvm-mc gfx1030), 256-aligned so their host addresses round-trip
// through the RDNA2 (lo<<8)|((hi&0xff)<<40) SHADER_PGM encoding. (Same blobs as test_gpustate_render.)
alignas(256) static const uint32_t kVs[] = {
    0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
    0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
};
alignas(256) static const uint32_t kPs[] = {
    0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};
static void set_pgm(GpuState& st, uint32_t lo_off, uint32_t hi_off, const void* p) {
    uint64_t a = (uint64_t)(uintptr_t)p;
    st.sh[lo_off] = (uint32_t)((a >> 8) & 0xFFFFFFFFu);
    st.sh[hi_off] = (uint32_t)((a >> 40) & 0xFFu);
}

int main() {
    printf("== test_gpu_execute ==\n");
    const uint32_t W = 64, H = 64;

    // Build the GpuState the CommandProcessor produces for one green fullscreen-triangle draw.
    GpuState st;
    set_pgm(st, P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, kVs);
    set_pgm(st, P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, kPs);
    st.uc[P::VGT_PRIMITIVE_TYPE] = 4;     // triangle list
    st.cx[P::CB_TARGET_MASK]     = 0xF;   // write RGBA
    st.draws.push_back({3});

    // The executor core, with the offscreen Vulkan renderer supplied as the backend (as the HLE will
    // supply the live-device renderer). execute_gpustate does recompile + resolve + render internally.
    // The backend gets the submit's DrawItem list; this test submits a single draw, so render items[0] via
    // the single-draw wrapper (default empty-buffer resources), exactly as before.
    auto backend = [&](const std::vector<DrawItem>& items) -> std::vector<uint8_t> {
        if (items.empty()) return {};
        return prosper::test::render_triangle_rgba(items[0].vs, items[0].fs, W, H, &items[0].ps);
    };
    std::vector<uint8_t> px = execute_gpustate(st, backend);
    CHECK(px.size() == (size_t)W * H * 4, "execute_gpustate rendered a frame from the GpuState");
    if (px.size() != (size_t)W * H * 4) { printf("== FAIL: executor produced no frame ==\n"); return 1; }

    auto isGreen = [&](uint32_t x, uint32_t y){ const uint8_t* p = &px[((size_t)y*W+x)*4]; return p[1] > 0x80 && p[0] < 0x40 && p[2] < 0x40; };
    uint32_t green = 0, total = 0;
    for (uint32_t y : {0u, H/2, H-1}) for (uint32_t x : {0u, W/2, W-1}) { total++; if (isGreen(x, y)) green++; }
    CHECK(green == total, "GpuState -> executor -> GREEN frame (full recompile+resolve+render spine)");

    // Present round-trip: hand the frame to the scanout path and read it back (what videoout presents).
    prosper::gpu::present_reset();
    prosper::gpu::present_write_frame(px.data(), W, H);
    CHECK(prosper::gpu::present_has_frame(), "present accepted the executor's frame");
    // #399: the rendered-frame dims are reported separately from the display dims, so a scaled-render
    // readback consumer (screenshot) can size its buffer correctly instead of dropping every frame.
    CHECK(prosper::gpu::present_frame_width() == W && prosper::gpu::present_frame_height() == H,
          "present_frame_width/height report the rendered frame's actual dims");
    std::vector<uint8_t> scan((size_t)W * H * 4, 0);
    size_t n = prosper::gpu::present_readback(scan.data(), scan.size());
    CHECK(n == px.size() && scan == px, "present_readback returns the executor's frame byte-for-byte");

    // An empty (state-only) submit renders nothing — mirrors the game's setup Dcb (0 draws).
    GpuState empty; empty.draws.clear();
    CHECK(execute_gpustate(empty, backend).empty(), "a draw-less GpuState renders no frame (state-only submit)");

    // #584: graphics and compute are one PM4 timeline. The planner must retain the asymmetric
    // draw-before-dispatch-before-draw dependency that exposed Dead Cells' future buffer read.
    {
        GpuState mixed;
        GpuState::Draw before, after;
        before.command_order = 100;
        after.command_order = 300;
        mixed.draws = {before, after};
        GpuState::Dispatch dispatch;
        dispatch.command_order = 200;
        mixed.dispatches.push_back(dispatch);
        auto operations = plan_submit_operations(mixed);
        CHECK(operations.size() == 3 &&
              operations[0].kind == SubmitOperationKind::Draw && operations[0].index == 0 &&
              operations[1].kind == SubmitOperationKind::Dispatch && operations[1].index == 0 &&
              operations[2].kind == SubmitOperationKind::Draw && operations[2].index == 1,
              "mixed submit planner retains draw-compute-draw PM4 order");

        uint32_t aliased_value = 7;
        std::vector<uint32_t> observed;
        std::vector<LiveRenderPhase> phases;
        DrawItem first, second;
        first.draw_index = 0;
        second.draw_index = 1;
        ComputeItem fill;
        fill.dispatch_index = 0;
        LiveRenderFn observe = [&](const std::vector<DrawItem>& items, uint32_t, uint32_t) {
            for (size_t i = 0; i < items.size(); ++i) observed.push_back(aliased_value);
            phases.push_back(live_render_phase());
            return std::vector<uint8_t>(4, static_cast<uint8_t>(aliased_value));
        };
        LiveComputeFn mutate = [&](const std::vector<ComputeItem>&) {
            aliased_value = 9;
            return true;
        };
        OrderedSubmitResult ordered = execute_ordered_items(
            operations, {first, second}, {fill}, observe, mutate, 1, 1);
        CHECK(observed == std::vector<uint32_t>({7, 9}),
              "ordered executor exposes pre-compute bytes to draw 0 and post-compute bytes to draw 1");
        CHECK(ordered.compute_executed && ordered.render_spans == 2 && phases.size() == 2 &&
              phases[0].first_span && !phases[0].final_span &&
              !phases[1].first_span && phases[1].final_span,
              "ordered executor brackets one submit across two graphics spans");
    }

    // #611: a compute fast-clear writes HTILE guest memory between graphics spans. The detached
    // Vulkan depth image must be invalidated through the core's backend-agnostic write observer.
    {
        auto& cache = prosper::test::persistent_ds_cache();
        cache.clear();
        prosper::test::PersistentDsKey scene;
        scene.dr = 0x100000; scene.dw = 0x100000;
        scene.sr = 0x200000; scene.sw = 0x200000;
        scene.htile = 0x300000; scene.w = 642; scene.h = 362; scene.fmt = 1;
        auto& image = cache[scene];
        image.depth_valid = true;
        image.stencil_valid = true;
        set_guest_gpu_write_observer([](uint64_t addr, uint64_t size) {
            prosper::test::invalidate_persistent_ds_guest_write(addr, size);
        });
        notify_guest_gpu_write(0x300000, 0x8000);
        CHECK(!image.depth_valid && !image.stencil_valid,
              "HTILE guest write invalidates both cached Vulkan DS planes");
        image.depth_valid = image.stencil_valid = true;
        notify_guest_gpu_write(0x100100, 16);
        CHECK(!image.depth_valid && !image.stencil_valid,
              "partial write inside a depth plane invalidates the cached Vulkan DS image");
        image.depth_valid = image.stencil_valid = true;
        notify_guest_gpu_write(0x400000, 0x8000);
        CHECK(image.depth_valid && image.stencil_valid,
              "unrelated guest write preserves cached Vulkan DS planes");
        set_guest_gpu_write_observer({});
        cache.clear();
    }

    // --- Indexed draws through the executor (issue #64) -------------------------------------------------
    // realize_draw_item fetches a REAL 16-bit guest index buffer (index_type 0, the SetIndexType reset
    // default this title relies on) and the backend renders it with vkCmdDrawIndexed — gl_VertexIndex is
    // then the FETCHED index. kVs computes the fullscreen triangle from gl_VertexIndex, so indices
    // {0,1,2} reproduce the non-indexed green frame exactly, while degenerate indices {0,0,0} collapse
    // the triangle and leave the frame blue (clear) — proving the index data actually drives the draw.
    auto backend_idx = [&](const std::vector<DrawItem>& items) -> std::vector<uint8_t> {
        if (items.empty()) return {};
        prosper::test::BackendDraw d;
        d.vs = items[0].vs; d.fs = items[0].fs; d.ps = &items[0].ps;
        d.vcount = items[0].vertex_count; d.indices = items[0].indices;
        return prosper::test::render_draws_rgba({std::move(d)}, W, H);
    };
    static const uint16_t kIdx012[3] = {0, 1, 2};
    static const uint16_t kIdx000[3] = {0, 0, 0};
    GpuState sti = st;
    sti.index_type = 0;   // 16-bit
    auto set_indexed = [&](const uint16_t* idx, uint32_t n) {
        GpuState::Draw d; d.index_count = n; d.indexed = true;
        d.index_addr = (uint64_t)(uintptr_t)idx;
        sti.draws.clear(); sti.draws.push_back(d);
    };
    set_indexed(kIdx012, 3);
    std::vector<uint8_t> pxi = execute_gpustate(sti, backend_idx);
    CHECK(pxi.size() == px.size() && pxi == px,
          "16-bit indexed draw {0,1,2} renders the SAME green frame as the non-indexed triangle");
    set_indexed(kIdx000, 3);
    std::vector<uint8_t> pxz = execute_gpustate(sti, backend_idx);
    bool all_blue = pxz.size() == (size_t)W * H * 4;
    if (all_blue) for (uint32_t y : {0u, H/2, H-1}) for (uint32_t x : {0u, W/2, W-1}) {
        const uint8_t* p = &pxz[((size_t)y*W+x)*4];
        if (!(p[2] > 0x80 && p[0] < 0x40 && p[1] < 0x40)) all_blue = false;
    }
    CHECK(all_blue, "degenerate indices {0,0,0} collapse the triangle (frame stays clear-blue)");
    // Unknown index element size -> loud fallback to a NON-indexed draw of the hint count (never
    // misread the buffer): with kVs that is the plain fullscreen triangle again.
    set_indexed(kIdx012, 3);
    sti.index_type = 7;   // no such encoding (0=16-bit, 1=32-bit)
    std::vector<uint8_t> pxu = execute_gpustate(sti, backend_idx);
    CHECK(pxu == px, "unknown index_type falls back to a non-indexed draw of the hint count");

    // #400: a zero-vertex-count non-indexed draw is a hardware no-op. realize_draw_item must SKIP it
    // (return false), not fabricate a phantom triangle (the vcount default was 3) nor sweep the residual
    // vertex pool. A positive count still realizes — the guard is specific to zero, not a regression.
    {
        DrawItem it0;
        bool made0 = realize_draw_item(st, nullptr, /*vcount_hint*/0u, 0x10000u, /*log*/false, it0);
        CHECK(!made0, "zero vertex-count draw is skipped (no phantom triangle / VB sweep)");
        DrawItem it3;
        bool made3 = realize_draw_item(st, nullptr, /*vcount_hint*/3u, 0x10000u, /*log*/false, it3);
        CHECK(made3 && it3.vertex_count == 3u, "non-zero vertex-count draw still realizes");

        GpuState rect = st;
        rect.uc[P::VGT_PRIMITIVE_TYPE] = 7;
        DrawItem rect_item;
        bool made_rect = realize_draw_item(rect, nullptr, /*vcount_hint*/3u, 0x10000u,
                                           /*log*/false, rect_item);
        CHECK(made_rect && rect_item.vertex_count == 4u &&
              rect_item.ps.topology == static_cast<uint32_t>(VkTopology::TriangleStrip),
              "PS5 procedural RectList expands three vertices to a four-corner Vulkan strip");
    }

    // #461: an indexed draw whose fetched index buffer contains a garbage-large value (an announced
    // 32-bit index buffer, or a torn read of concurrently-freed guest memory) must NOT inflate
    // vertex_count / the VB upload unboundedly (OOM guard). realize_draw_item clamps vertex_count to the
    // index-count ceiling (1<<20) instead of max_index+1 = ~0x0F000001.
    {
        static const uint32_t kBigIdx[3] = { 0u, 1u, 0x0F000000u };   // one garbage/huge 32-bit index
        GpuState sbig = st;
        sbig.index_type = 1;   // 32-bit (announced -> the <0x10000 fingerprint is not run)
        GpuState::Draw db; db.index_count = 3; db.indexed = true; db.index_addr = (uint64_t)(uintptr_t)kBigIdx;
        sbig.draws.clear(); sbig.draws.push_back(db);
        DrawItem itb;
        bool madeb = realize_draw_item(sbig, &sbig.draws[0], sbig.draws[0].index_count, 0x10000u, /*log*/false, itb);
        CHECK(madeb && itb.vertex_count == (1u << 20),
              "#461: garbage-large index clamps vertex_count to the 1<<20 ceiling (no multi-GB VB)");
    }

    // The live-submit registry path — exactly what agc_driver_submit_dcb drives once a device is wired.
    CHECK(!have_submit_renderer(), "no live renderer registered by default (game path stays inert)");
    CHECK(!execute_and_present(st, W, H), "execute_and_present is a no-op with no renderer registered");
    std::shared_ptr<const std::vector<uint8_t>> live_storage;
    set_submit_renderer([&](const std::vector<DrawItem>& items, uint32_t w, uint32_t h) -> RenderedFrame {
        if (items.empty()) return {};
        live_storage = std::make_shared<const std::vector<uint8_t>>(
            prosper::test::render_triangle_rgba(items[0].vs, items[0].fs, w, h, &items[0].ps));
        return RenderedFrame(live_storage);
    });
    CHECK(have_submit_renderer(), "live renderer registered");
    prosper::gpu::present_reset();
    CHECK(execute_and_present(st, W, H), "execute_and_present rendered + presented the submit");
    CHECK(prosper::gpu::present_has_frame(), "the presented submit frame reached the scanout path");
    std::vector<uint8_t> submit_scan((size_t)W * H * 4, 0);
    prosper::gpu::present_readback(submit_scan.data(), submit_scan.size());
    CHECK(submit_scan == px, "live-submit path presents the same GREEN frame as the direct executor");
    prosper::gpu::PresentFrameLease submit_lease;
    CHECK(prosper::gpu::present_acquire_rendered_frame(submit_lease) && submit_lease.rgba &&
          submit_lease.rgba->data() == live_storage->data(),
          "live-submit path publishes the renderer allocation without copying");
    CHECK(!execute_and_present(empty, W, H), "execute_and_present skips a draw-less submit even with a renderer");
    set_submit_renderer({});  // restore inert default

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
