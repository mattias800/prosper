// test_gpu_execute — the GPU-executor core spine (Stage A): a GpuState (exactly what SubmitDcb folds a
// Dcb into) -> execute_gpustate() [recompile shaders from their PGM addresses + resolve pipeline] ->
// a caller-supplied Vulkan render -> present_write_frame -> present_readback. Proves the executor entry
// point that agc_driver_submit_dcb will call, and the scanout round-trip, end to end on llvmpipe.
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/gpu_capture.hpp"
#include "../src/gpu/videoout_present.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include "../src/gpu/vk_translate.hpp"
#include "../src/hle/dispatch.hpp"
#include "../src/host/guest_write_watch.hpp"
#include "render_runner.h"
#include "test_scratch.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#endif

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
// Descriptor-free clear-RG helper emitted by AGC for CB_COLOR_CONTROL.DCC_DECOMPRESS. Hardware
// interprets its export as a metadata operation; it must not run as an ordinary Vulkan color shader.
alignas(256) static const uint32_t kDccPs[] = {
    0x7E000280u, 0xF8001803u, 0x00000000u, 0xBF810000u,
};
alignas(256) static const uint32_t kNoopCs[] = {0xBF810000u};
alignas(256) static uint32_t kOrderedLateCs[] = {0u};
static void set_pgm(GpuState& st, uint32_t lo_off, uint32_t hi_off, const void* p) {
    uint64_t a = (uint64_t)(uintptr_t)p;
    st.sh[lo_off] = (uint32_t)((a >> 8) & 0xFFFFFFFFu);
    st.sh[hi_off] = (uint32_t)((a >> 40) & 0xFFu);
}

int main() {
    printf("== test_gpu_execute ==\n");
    prosper::register_agc_hle();
    const uint32_t W = 64, H = 64;

    uint64_t occurrence = 0;
    CHECK(should_log_recompile_reject(0xfeed0001, 0xfeed0002, 0, 0, 0, &occurrence) &&
          occurrence == 1,
          "recompile rejection limiter logs a shader pair's first occurrence");
    CHECK(should_log_recompile_reject(0xfeed0001, 0xfeed0002, 0, 0, 0, &occurrence) &&
          occurrence == 2,
          "recompile rejection limiter logs the second power-of-two occurrence");
    CHECK(!should_log_recompile_reject(0xfeed0001, 0xfeed0002, 0, 0, 0, &occurrence) &&
          occurrence == 3,
          "recompile rejection limiter suppresses a non-power-of-two repeat");
    CHECK(should_log_recompile_reject(0xfeed0001, 0xfeed0002, 0, 0, 0, &occurrence) &&
          occurrence == 4 &&
          !should_log_recompile_reject(0xfeed0001, 0xfeed0002, 0, 0, 0, &occurrence) &&
          occurrence == 5,
          "recompile rejection limiter reports exponential recurrence counts");
    CHECK(should_log_recompile_reject(0xfeed0001, 0xfeed0003, 0, 0, 0, &occurrence) &&
          occurrence == 1,
          "recompile rejection limiter tracks distinct shader pairs independently");

    // Build the GpuState the CommandProcessor produces for one green fullscreen-triangle draw.
    GpuState st;
    set_pgm(st, P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, kVs);
    set_pgm(st, P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, kPs);
    st.uc[P::VGT_PRIMITIVE_TYPE] = 4;     // triangle list
    st.cx[P::CB_TARGET_MASK]     = 0xF;   // write RGBA
    st.num_instances = 4;
    GpuState::Draw source_draw;
    source_draw.index_count = 3;
    source_draw.instance_count = 4;
    st.draws.push_back(source_draw);
    st.num_instances = 0; // later state must not rewrite the already-recorded draw in folded mode

    // The executor core, with the offscreen Vulkan renderer supplied as the backend (as the HLE will
    // supply the live-device renderer). execute_gpustate does recompile + resolve + render internally.
    // The backend gets the submit's DrawItem list; this test submits a single draw, so render items[0] via
    // the single-draw wrapper (default empty-buffer resources), exactly as before.
    DrawItem realized_draw;
    auto backend = [&](const std::vector<DrawItem>& items) -> std::vector<uint8_t> {
        if (items.empty()) return {};
        realized_draw = items[0];
        return prosper::test::render_triangle_rgba(
            items[0].vs_words(), items[0].fs_words(), W, H, &items[0].ps);
    };
    std::vector<uint8_t> px = execute_gpustate(st, backend);
    CHECK(px.size() == (size_t)W * H * 4, "execute_gpustate rendered a frame from the GpuState");
    if (px.size() != (size_t)W * H * 4) { printf("== FAIL: executor produced no frame ==\n"); return 1; }

    auto isGreen = [&](uint32_t x, uint32_t y){ const uint8_t* p = &px[((size_t)y*W+x)*4]; return p[1] > 0x80 && p[0] < 0x40 && p[2] < 0x40; };
    uint32_t green = 0, total = 0;
    for (uint32_t y : {0u, H/2, H-1}) for (uint32_t x : {0u, W/2, W-1}) { total++; if (isGreen(x, y)) green++; }
    CHECK(green == total, "GpuState -> executor -> GREEN frame (full recompile+resolve+render spine)");
    CHECK(realized_draw.instance_count == 4,
          "draw realization retains the folded hardware instance count");
    {
        GpuState offset_state = st;
        offset_state.uc[P::GE_INDX_OFFSET] = 37;
        offset_state.draws[0].modifier = 0x1122334455667788ull;
        DrawItem offset_item;
        const bool made_offset = realize_draw_item(
            offset_state, &offset_state.draws[0], offset_state.draws[0].index_count,
            0x10000u, false, offset_item);
        CHECK(made_offset && offset_item.vertex_offset == 37 &&
              offset_item.raw_draw_modifier == 0x1122334455667788ull,
              "draw realization retains GE_INDX_OFFSET and the raw ShaderDrawModifier");
    }

    // DCC_DECOMPRESS binds a graphics helper whose color export is interpreted by the hardware as a
    // metadata operation. Prosper stores only materialized Vulkan color, so realization substitutes
    // a NULL-export fragment module while retaining the target's ordinary write mask/cache identity.
    GpuState dcc = st;
    set_pgm(dcc, P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, kDccPs);
    dcc.cx[P::CB_COLOR_CONTROL] =
        (P::CB_COLOR_CONTROL_MODE_DCC_DECOMPRESS << P::CB_COLOR_CONTROL_MODE_SHIFT) |
        (0xCCu << P::CB_COLOR_CONTROL_ROP3_SHIFT);
    std::vector<uint8_t> dcc_px = execute_gpustate(dcc, backend);
    bool dcc_preserved = dcc_px.size() == (size_t)W * H * 4;
    if (dcc_preserved) {
        const uint8_t* center = &dcc_px[((size_t)(H / 2) * W + W / 2) * 4];
        dcc_preserved = center[2] > 0x80 && center[0] < 0x40 && center[1] < 0x40;
    }
    CHECK(dcc_preserved,
          "DCC decompress keeps the color attachment unchanged instead of running its helper export");
    bool dcc_has_output_variable = false;
    for (size_t word = 5; word < realized_draw.fs.size();) {
        const uint32_t instruction = realized_draw.fs[word];
        const uint32_t word_count = instruction >> 16;
        const uint32_t opcode = instruction & 0xffffu;
        if (!word_count || word_count > realized_draw.fs.size() - word) break;
        // OpVariable's third operand is its StorageClass; 3 is Output. A fragment module with no
        // output interface is the Vulkan equivalent of the hardware metadata operation.
        if (opcode == 59u && word_count >= 4u && realized_draw.fs[word + 3u] == 3u)
            dcc_has_output_variable = true;
        word += word_count;
    }
    CHECK(!dcc_has_output_variable,
          "DCC decompress replacement exposes no ordinary fragment-color output");

    // A folded submit can retain MODE=6 after the guest has restored normal mode. Do not classify a
    // real shader from the following post-process pass as the helper merely because of that residue.
    GpuState stale_dcc_mode = st;
    stale_dcc_mode.cx[P::CB_COLOR_CONTROL] = dcc.cx[P::CB_COLOR_CONTROL];
    const std::vector<uint8_t> stale_dcc_px = execute_gpustate(stale_dcc_mode, backend);
    bool stale_dcc_green = stale_dcc_px.size() == static_cast<size_t>(W) * H * 4;
    if (stale_dcc_green) {
        const uint8_t* center =
            &stale_dcc_px[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
        stale_dcc_green = center[1] > 0x80 && center[0] < 0x40 && center[2] < 0x40;
    }
    CHECK(stale_dcc_green,
          "stale DCC mode does not suppress the following ordinary color shader");

    // Astro's DCC helpers operate on native FP16 post-process targets. Seed one persistent target
    // with the ordinary green shader, then execute the realized metadata helper against the same
    // image with LOAD semantics. A format-specific pipeline mistake here would silently erase HDR
    // intermediates even though the RGBA8 clear-preservation check above passed.
    constexpr uint64_t dcc_fp16_target_id = 0x8250000000000016ull;
    ResolvedPipelineState fp16_state = realized_draw.ps;
    fp16_state.color0_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    prosper::test::BackendDraw fp16_seed_draw;
    fp16_seed_draw.vs = realized_draw.vs;
    fp16_seed_draw.fs = recompile_fragment(kPs, std::size(kPs));
    fp16_seed_draw.ps = &fp16_state;
    fp16_seed_draw.vcount = 3;
    prosper::test::BackendColorTarget fp16_seed_target{
        dcc_fp16_target_id, false, true, VK_FORMAT_R16G16B16A16_SFLOAT};
    const std::vector<uint8_t> fp16_seed = prosper::test::render_draws_rgba(
        {fp16_seed_draw}, W, H, nullptr, nullptr, false, &fp16_seed_target);
    auto fp16_dcc_backend = [&](const std::vector<DrawItem>& items) -> std::vector<uint8_t> {
        if (items.empty()) return {};
        ResolvedPipelineState fp16_dcc_state = items[0].ps;
        fp16_dcc_state.color0_format = VK_FORMAT_R16G16B16A16_SFLOAT;
        prosper::test::BackendDraw draw;
        draw.vs = items[0].vs_words();
        draw.fs = items[0].fs_words();
        draw.ps = &fp16_dcc_state;
        draw.vcount = items[0].vertex_count;
        prosper::test::BackendColorTarget target{
            dcc_fp16_target_id, true, true, VK_FORMAT_R16G16B16A16_SFLOAT};
        return prosper::test::render_draws_rgba(
            {std::move(draw)}, W, H, nullptr, nullptr, false, &target);
    };
    const std::vector<uint8_t> fp16_dcc = execute_gpustate(dcc, fp16_dcc_backend);
    CHECK(fp16_seed.size() == static_cast<size_t>(W) * H * 8 && fp16_dcc == fp16_seed,
          "DCC decompress preserves a native FP16 color attachment byte-for-byte");
    prosper::test::invalidate_persistent_color_target(dcc_fp16_target_id);

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
        GuestGpuWriteSnapshot validation;
        GuestGpuWriteQuery overlapping = GuestGpuWriteQuery::Unknown;
        GuestGpuWriteQuery unrelated = GuestGpuWriteQuery::Unknown;
        DrawItem first, second;
        first.draw_index = 0;
        second.draw_index = 1;
        ComputeItem fill;
        fill.dispatch_index = 0;
        LiveRenderFn observe = [&](const std::vector<DrawItem>& items, uint32_t, uint32_t) {
            for (size_t i = 0; i < items.size(); ++i) observed.push_back(aliased_value);
            phases.push_back(live_render_phase());
            if (phases.size() == 1) {
                validation = guest_gpu_write_snapshot();
            } else {
                overlapping = guest_gpu_writes_since(validation, 0x2080, 16);
                unrelated = guest_gpu_writes_since(validation, 0x4000, 16);
            }
            return std::vector<uint8_t>(4, static_cast<uint8_t>(aliased_value));
        };
        LiveComputeFn mutate = [&](const std::vector<ComputeItem>&) {
            aliased_value = 9;
            notify_guest_gpu_write(0x2000, 0x100);
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
        CHECK(phases[0].allows_deferred_scanout_readback() &&
              !phases[1].allows_deferred_scanout_readback(),
              "only a non-final graphics span may defer scanout readback across compute");
        CHECK(overlapping == GuestGpuWriteQuery::Overlap &&
              unrelated == GuestGpuWriteQuery::Unchanged,
              "write journal distinguishes overlapping and unrelated in-submit GPU writes");
        CHECK(guest_gpu_writes_since(validation, 0x4000, 16) == GuestGpuWriteQuery::Unknown,
              "write journal snapshots cannot suppress validation after submit completion");
    }

#if defined(__linux__)
    // The in-submit journal expires by design. A persistent page watch must still become Dirty when
    // the same GPU/DMA notification occurs between submits, or a cross-submit decoded-texture cache
    // could trust stale guest bytes after releasing its redundant exact source snapshot.
    {
        constexpr size_t watch_bytes = 0x1000;
        auto* watched = static_cast<uint8_t*>(mmap(
            nullptr, watch_bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        CHECK(watched != MAP_FAILED, "map GPU-write watch regression page");
        if (watched != MAP_FAILED) {
            prosper::host::guest_write_watch_set_fault_onstack(true);
            prosper::host::guest_write_watch_notify_direct_mapping_added(
                reinterpret_cast<uint64_t>(watched), watch_bytes, 0x6a0000,
                0x3 /* SCE CPU_READ|CPU_WRITE */);
            constexpr uint64_t logical_offset = 128;
            constexpr uint64_t logical_bytes = 64;
            auto watch = prosper::host::GuestWriteWatch::create(
                reinterpret_cast<uint64_t>(watched) + logical_offset, logical_bytes);
            CHECK(static_cast<bool>(watch) &&
                      watch.query() == prosper::host::GuestWriteWatchQuery::Unchanged,
                  "arm cross-submit watch before GPU notification");
            bool preserving_write_observed = false;
            set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
                preserving_write_observed =
                    addr == reinterpret_cast<uint64_t>(watched) + logical_offset + 8u &&
                    size == 16u;
            });
            notify_guest_gpu_write_preserving_bytes(
                reinterpret_cast<uint64_t>(watched) + logical_offset + 8u, 16u);
            set_guest_gpu_write_observer({});
            CHECK(preserving_write_observed,
                  "byte-preserving GPU result still notifies renderer-resident aliases");
            CHECK(watch.query() == prosper::host::GuestWriteWatchQuery::Unchanged,
                  "byte-preserving GPU result keeps the exact guest source watch clean");
            notify_guest_gpu_write(reinterpret_cast<uint64_t>(watched) + 32u, 16u);
            CHECK(watch.query() == prosper::host::GuestWriteWatchQuery::Unchanged,
                  "adjacent same-page GPU write does not dirty a nonoverlapping logical source");
            notify_guest_gpu_write(
                reinterpret_cast<uint64_t>(watched) + logical_offset + 16u, 16u);
            CHECK(watch.query() == prosper::host::GuestWriteWatchQuery::Dirty,
                  "guest GPU write notification invalidates a persistent page watch");
            watch.reset();
            prosper::host::guest_write_watch_notify_direct_mapping_removed(
                reinterpret_cast<uint64_t>(watched), watch_bytes);
            prosper::host::guest_write_watch_set_fault_onstack(false);
            munmap(watched, watch_bytes);
        }
    }
#endif

    // #189: address-backed DMA is an in-stream producer, not a completion write. It must split a
    // graphics span so an earlier consumer sees old bytes and a later consumer sees copied bytes.
    {
        uint8_t source = 0x39;
        uint8_t target = 0x17;
        GpuState mixed;
        GpuState::Draw before, after;
        before.command_order = 100;
        after.command_order = 300;
        mixed.draws = {before, after};
        mixed.dma_copies.push_back({
            (uint64_t)(uintptr_t)&target, (uint64_t)(uintptr_t)&source,
            1, 0, 200, 0});
        const auto operations = plan_submit_operations(mixed);

        DrawItem first, second;
        first.draw_index = 0;
        second.draw_index = 1;
        std::vector<uint8_t> observed;
        std::vector<LiveRenderPhase> dma_phases;
        LiveRenderFn consume = [&](const std::vector<DrawItem>& items, uint32_t, uint32_t) {
            for (size_t i = 0; i < items.size(); ++i) observed.push_back(target);
            dma_phases.push_back(live_render_phase());
            return std::vector<uint8_t>(4, target);
        };
        const OrderedSubmitResult result = execute_ordered_items(
            operations, {first, second}, {}, mixed.dma_copies, consume, {}, 1, 1);
        CHECK(observed == std::vector<uint8_t>({0x17, 0x39}) && target == source,
              "ordered DMA exposes old bytes before the copy and new bytes afterward");
        CHECK(result.render_spans == 2,
              "an ordered DMA copy splits graphics consumers into two render spans");
        CHECK(dma_phases.size() == 2 && dma_phases[0].authoritative_readback &&
              !dma_phases[1].authoritative_readback,
              "the graphics span immediately before DMA requests authoritative target readback");
        CHECK(!dma_phases[0].allows_deferred_scanout_readback() &&
              !dma_phases[1].allows_deferred_scanout_readback(),
              "DMA producers and final spans cannot defer scanout readback");
    }

    // A rendered target's authoritative bytes can live only in the backend cache. Resolve the
    // requested interior range after the preceding graphics span rather than reading stale/unmapped
    // guest backing at the target address.
    {
        constexpr uint64_t live_base = 0x100000000ull;
        const std::vector<uint8_t> live_pixels = {0x10, 0x21, 0x32, 0x43, 0x54, 0x65};
        uint8_t target[3] = {0, 0, 0};
        bool producer_rendered = false;
        set_live_target_byte_range_reader(
            [&](uint64_t addr, uint32_t bytes, std::vector<uint8_t>& output) {
                if (addr < live_base || addr >= live_base + live_pixels.size())
                    return LiveTargetByteReadResult::NotFound;
                const uint64_t offset = addr - live_base;
                if (!producer_rendered || bytes > live_pixels.size() - offset)
                    return LiveTargetByteReadResult::InvalidRange;
                output.assign(live_pixels.begin() + static_cast<size_t>(offset),
                              live_pixels.begin() + static_cast<size_t>(offset + bytes));
                return LiveTargetByteReadResult::Success;
            });
        GpuState state;
        GpuState::Draw producer;
        producer.command_order = 100;
        state.draws.push_back(producer);
        state.dma_copies.push_back({
            (uint64_t)(uintptr_t)target, live_base + 2, sizeof target, 0, 200, 0});
        DrawItem draw;
        draw.draw_index = 0;
        LiveRenderFn render = [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
            producer_rendered = true;
            return std::vector<uint8_t>(4, 0xff);
        };
        execute_ordered_items(plan_submit_operations(state), {draw}, {}, state.dma_copies,
                              render, {}, 1, 1);
        CHECK(producer_rendered &&
              std::vector<uint8_t>(target, target + sizeof target) ==
                  std::vector<uint8_t>({0x32, 0x43, 0x54}),
              "ordered DMA reads an interior range from the preceding live render target");
        set_live_target_byte_range_reader({});
    }

    // A no-draw submit still executes DMA before a later compute consumer. The old completion-FIFO
    // path left this copy pending because execute_submit_work's compute-only branch never drained it.
    {
        uint8_t source = 0xA5;
        uint8_t target = 0x4C;
        GpuState compute_only;
        compute_only.dma_copies.push_back({
            (uint64_t)(uintptr_t)&target, (uint64_t)(uintptr_t)&source,
            1, 0, 100, 0});
        GpuState::Dispatch dispatch;
        dispatch.command_order = 200;
        compute_only.dispatches.push_back(dispatch);
        const auto operations = plan_submit_operations(compute_only);
        ComputeItem consume;
        consume.dispatch_index = 0;
        uint8_t observed = 0;
        LiveComputeFn compute = [&](const std::vector<ComputeItem>&) {
            observed = target;
            return true;
        };
        const OrderedSubmitResult result = execute_ordered_items(
            operations, {}, {consume}, compute_only.dma_copies, {}, compute, 0, 0);
        CHECK(result.compute_executed && observed == source && target == source,
              "compute-only timeline executes a preceding DMA copy before its consumer");
    }

    // Overflow or otherwise incomplete write history must fall back to exact validation.
    {
        DrawItem first, second;
        first.draw_index = 0;
        second.draw_index = 1;
        ComputeItem fill;
        fill.dispatch_index = 0;
        const std::vector<SubmitOperation> operations = {
            {SubmitOperationKind::Draw, 0, 100},
            {SubmitOperationKind::Dispatch, 0, 200},
            {SubmitOperationKind::Draw, 1, 300},
        };
        GuestGpuWriteSnapshot validation;
        GuestGpuWriteQuery after_overflow = GuestGpuWriteQuery::Unchanged;
        size_t renders = 0;
        LiveRenderFn observe = [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
            if (renders++ == 0) validation = guest_gpu_write_snapshot();
            else after_overflow = guest_gpu_writes_since(validation, 0x8000, 16);
            return std::vector<uint8_t>(4, 0);
        };
        LiveComputeFn overflow = [&](const std::vector<ComputeItem>&) {
            for (size_t i = 0; i <= kGuestGpuWriteJournalCapacity; ++i)
                notify_guest_gpu_write(0x100000 + i * 0x10, 1);
            return true;
        };
        execute_ordered_items(operations, {first, second}, {fill}, observe, overflow, 1, 1);
        CHECK(after_overflow == GuestGpuWriteQuery::Unknown,
              "write journal overflow forces conservative validation");
    }

    // #611: a compute fast-clear writes HTILE guest memory between graphics spans. The detached
    // Vulkan depth image must be invalidated through the core's backend-agnostic write observer.
    // Depth and stencil live at separate guest addresses, so direct writes invalidate only the
    // corresponding Vulkan aspect; HTILE remains conservative because it can describe both.
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
        CHECK(!image.depth_valid && image.stencil_valid,
              "partial depth-plane write preserves the independent cached stencil aspect");
        image.depth_valid = image.stencil_valid = true;
        notify_guest_gpu_write(0x200100, 16);
        CHECK(image.depth_valid && !image.stencil_valid,
              "partial stencil-plane write preserves depth for a later sampled-depth pass");
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
        d.vs = items[0].vs_words(); d.fs = items[0].fs_words(); d.ps = &items[0].ps;
        d.vcount = items[0].vertex_count; d.indices = items[0].indices;
        d.vertex_offset = items[0].vertex_offset;
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
        CHECK(made3 && it3.raw_draw_count == 3u && !it3.raw_indexed,
              "#1256: realize_draw_item records the raw non-indexed draw-packet count (3) for capture");

        GpuState rect = st;
        rect.uc[P::VGT_PRIMITIVE_TYPE] = 7;
        DrawItem rect_item;
        bool made_rect = realize_draw_item(rect, nullptr, /*vcount_hint*/3u, 0x10000u,
                                           /*log*/false, rect_item);
        CHECK(made_rect && rect_item.vertex_count == 4u &&
              rect_item.ps.topology == static_cast<uint32_t>(VkTopology::TriangleStrip),
              "PS5 procedural RectList expands three vertices to a four-corner Vulkan strip");
    }

    // #1163: a non-indexed DrawIndexAuto count is the AUTHORITATIVE hardware vertex count. The bound VB's
    // record count (vb_records = size/stride) must NOT override it — a title binding a SHARED vertex pool
    // (GTA V's Scaleform UI: per-draw counts 3/6/30 against a fixed ~4096-byte pool -> vb_records
    // 146/1024/512) would otherwise sweep the whole pool, rasterizing stale-data triangles that inflate the
    // stencil masks past their EQUAL==2 clip (the black menu wedges). The VB record count is only a fallback
    // when the draw supplied NO count (draw_count == 0, already skipped as a no-op upstream). WITHOUT the
    // fix (old: `if (vb_records > count) count = vb_records`) every one of these would return the record
    // count, so each real-count assertion below would fail.
    CHECK(resolve_nonindexed_vertex_count(/*draw_count*/6u,  /*vb_records*/1024u) == 6u,
          "#1163: GTA Scaleform mask draw (6 verts vs 1024-record shared pool) keeps its real count");
    CHECK(resolve_nonindexed_vertex_count(/*draw_count*/3u,  /*vb_records*/146u)  == 3u,
          "#1163: a 3-vertex non-indexed draw is not inflated to the VB record count");
    CHECK(resolve_nonindexed_vertex_count(/*draw_count*/30u, /*vb_records*/512u)  == 30u,
          "#1163: a 30-vertex non-indexed draw is not inflated to the VB record count");
    CHECK(resolve_nonindexed_vertex_count(/*draw_count*/0u,  /*vb_records*/1024u) == 1024u,
          "#1163: a draw with NO count falls back to the VB record count (zero-count guard)");

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
        CHECK(madeb && itb.raw_draw_count == 3u && itb.raw_indexed,
              "#1256: realize_draw_item records the raw indexed draw-packet count (3) + indexed flag");
    }

    // The live-submit registry path — exactly what agc_driver_submit_dcb drives once a device is wired.
    // #189: production submit execution must realize an indexed draw only after a preceding DMA
    // supplies its index buffer. Callback ordering alone is insufficient because realization owns
    // a copied index vector; the old eager path permanently captured {0,0,0} before DMA ran.
    {
        uint16_t copied_indices[3] = {0, 0, 0};
        uint16_t source_indices[3] = {0, 1, 2};
        GpuState ordered = st;
        ordered.index_type = 0;
        ordered.draws.clear();
        ordered.dispatches.clear();
        ordered.dma_copies.clear();
        ordered.ordered_memory_effects.clear();
        GpuState::Draw draw;
        draw.index_count = 3;
        draw.indexed = true;
        draw.index_addr = (uint64_t)(uintptr_t)copied_indices;
        draw.command_order = 200;
        ordered.draws.push_back(draw);
        ordered.dma_copies.push_back({
            (uint64_t)(uintptr_t)copied_indices, (uint64_t)(uintptr_t)source_indices,
            sizeof(copied_indices), 0, 100, 0});
        std::vector<uint32_t> submitted_indices;
        set_submit_renderer([&](const std::vector<DrawItem>& items, uint32_t, uint32_t) {
            if (!items.empty()) submitted_indices = items.front().indices;
            return RenderedFrame{};
        });
        execute_ordered_and_present(ordered, W, H, 77, /*publish=*/false);
        set_submit_renderer({});
        CHECK(submitted_indices == std::vector<uint32_t>({0, 1, 2}),
              "DMA-backed index buffer is realized after the ordered copy in the production path");
    }

    // Indirect arguments are generated memory, not command-fold inputs. Supply them through an
    // earlier ordered copy and verify that the production path reads the five-dword indexed draw
    // only after that producer, including instance count and signed vertex offset.
    {
        alignas(4) uint32_t generated_args[5] = {3, 2, 0, 7, 0};
        alignas(4) uint32_t consumed_args[5] = {};
        uint16_t indirect_indices[3] = {0, 1, 2};
        GpuState ordered = st;
        ordered.index_type = 0;
        ordered.draws.clear();
        ordered.dispatches.clear();
        ordered.dma_copies.clear();
        ordered.ordered_memory_effects.clear();
        GpuState::Draw draw;
        draw.indexed = true;
        draw.index_base = reinterpret_cast<uint64_t>(indirect_indices);
        draw.indirect = true;
        draw.indirect_args_addr = reinterpret_cast<uint64_t>(consumed_args);
        draw.command_order = 200;
        ordered.draws.push_back(draw);
        ordered.dma_copies.push_back({
            reinterpret_cast<uint64_t>(consumed_args),
            reinterpret_cast<uint64_t>(generated_args), sizeof(consumed_args), 0, 100, 0});
        const std::filesystem::path capture_path =
            prosper_test::test_scratch_dir() /
            "prosper_dma_indirect_draw_capture.prgcap";
        std::error_code capture_filesystem_error;
        std::filesystem::remove(capture_path, capture_filesystem_error);
        request_interactive_gpu_capture(capture_path.string());
        std::vector<uint32_t> submitted_indices;
        uint32_t submitted_instances = 0;
        int32_t submitted_vertex_offset = 0;
        uint32_t submitted_raw_count = 0;
        set_submit_renderer([&](const std::vector<DrawItem>& items, uint32_t, uint32_t) {
            if (!items.empty()) {
                submitted_indices = items.front().indices;
                submitted_instances = items.front().instance_count;
                submitted_vertex_offset = items.front().vertex_offset;
                submitted_raw_count = items.front().raw_draw_count;
            }
            return RenderedFrame{};
        });
        execute_ordered_and_present(ordered, W, H, 78, /*publish=*/false);
        set_submit_renderer({});
        CHECK(submitted_indices == std::vector<uint32_t>({0, 1, 2}) &&
              submitted_instances == 2 && submitted_vertex_offset == 7 &&
              submitted_raw_count == 3,
              "indexed indirect arguments resolve lazily after their ordered producer");
        GpuCaptureFile indirect_capture;
        GpuReplayFrame indirect_replay;
        std::string indirect_capture_error;
        const bool captured = read_gpu_capture(
            capture_path.string(), indirect_capture, indirect_capture_error);
        const bool destination_was_zero = captured && indirect_capture.dma_copies.size() == 1 &&
            indirect_capture.dma_copies[0].destination_blob_index <
                indirect_capture.blobs.size() && [&] {
                    const auto& dma = indirect_capture.dma_copies[0];
                    const auto& blob = indirect_capture.blobs[dma.destination_blob_index];
                    if (dma.destination_blob_offset + sizeof(consumed_args) > blob.bytes.size())
                        return false;
                    return std::all_of(
                        blob.bytes.begin() + static_cast<ptrdiff_t>(dma.destination_blob_offset),
                        blob.bytes.begin() + static_cast<ptrdiff_t>(
                            dma.destination_blob_offset + sizeof(consumed_args)),
                        [](uint8_t value) { return value == 0; });
                }();
        CHECK(captured && indirect_capture.draws.size() == 1 &&
                  indirect_capture.draws[0].raw_draw_count == 3 &&
                  indirect_capture.draws[0].instance_count == 2 &&
                  indirect_capture.draws[0].vertex_offset == 7 &&
                  indirect_capture.dma_copies.size() == 1 && destination_was_zero &&
                  materialize_gpu_replay(indirect_capture, indirect_replay,
                                         indirect_capture_error) &&
                  indirect_replay.items.size() == 1 &&
                  indirect_replay.items[0].raw_draw_count == 3 &&
                  indirect_replay.items[0].instance_count == 2 &&
                  indirect_replay.items[0].vertex_offset == 7,
              "deferred capture combines exact post-DMA indirect draw fields with pre-submit bytes");
        std::filesystem::remove(capture_path, capture_filesystem_error);
    }

    // A parser stall makes later indirect packets depend on every producer in the preceding epoch.
    // Valid stale argument bytes must never leak through when that producer failed: doing so can
    // turn last frame's plausible counts into a real backend draw.
    for (bool failed_dma : {false, true}) {
        alignas(4) uint32_t stale_args[5] = {3, 1, 0, 0, 0};
        uint16_t stale_indices[3] = {0, 1, 2};
        uint8_t dma_destination = 0;
        GpuState dependent = st;
        dependent.index_type = 0;
        dependent.draws.clear();
        dependent.dispatches.clear();
        dependent.dma_copies.clear();
        dependent.parser_stalls.clear();
        dependent.ordered_memory_effects.clear();
        if (failed_dma) {
            dependent.dma_copies.push_back({
                reinterpret_cast<uint64_t>(&dma_destination), 0xdead00000000ull,
                1, 0, 50, 0});
        } else {
            GpuState::Dispatch failed_dispatch;
            failed_dispatch.threads_x = failed_dispatch.threads_y =
                failed_dispatch.threads_z = 1;
            failed_dispatch.command_order = 50;
            dependent.dispatches.push_back(failed_dispatch);
        }
        dependent.parser_stalls.push_back({100});
        dependent.parser_stalls.push_back({110});
        GpuState::Draw dependent_draw;
        dependent_draw.indexed = true;
        dependent_draw.index_base = reinterpret_cast<uint64_t>(stale_indices);
        dependent_draw.indirect = true;
        dependent_draw.indirect_args_addr = reinterpret_cast<uint64_t>(stale_args);
        dependent_draw.command_order = 150;
        dependent.draws.push_back(dependent_draw);
        uint32_t backend_draws = 0;
        set_submit_renderer([&](const std::vector<DrawItem>& items, uint32_t, uint32_t) {
            backend_draws += static_cast<uint32_t>(items.size());
            return RenderedFrame{};
        });
        set_submit_compute([](const std::vector<ComputeItem>&) { return true; });
        execute_ordered_and_present(dependent, W, H, failed_dma ? 80 : 79,
                                    /*publish=*/false);
        set_submit_renderer({});
        set_submit_compute({});
        CHECK(backend_draws == 0,
              failed_dma
                  ? "failed DMA producer poisons the stalled indirect consumer"
                  : "failed compute realization poisons the stalled indirect consumer");
    }

    // Compute-only submissions bypass presentation. Their capture must remain deferred even without
    // DMA so finish receives the exact realized/executed compute rather than eagerly materializing
    // the intentionally empty pre-execution lists as an unrealized operation.
    {
        auto create_shader = prosper::Hle::lookup("f3dg2CSgRKY");
        ShaderReg compute_registers[2] = {
            {P::COMPUTE_PGM_LO, 0}, {P::COMPUTE_PGM_HI, 0},
        };
        AgcShaderHeader compute_header{};
        compute_header.file_header = 0x34333231u;
        compute_header.version = 0x18;
        compute_header.sh_registers = compute_registers;
        compute_header.shader_size = sizeof(kNoopCs);
        compute_header.type = 0;
        compute_header.num_sh_registers = 2;
        void* registered_shader = nullptr;
        CHECK(create_shader &&
                  create_shader(reinterpret_cast<uint64_t>(&registered_shader),
                                reinterpret_cast<uint64_t>(&compute_header),
                                reinterpret_cast<uint64_t>(kNoopCs), 0, 0, 0) == 0 &&
                  registered_shader == &compute_header,
              "register a descriptor-free compute shader for non-render capture");
        GpuState nonrender;
        nonrender.sh[P::COMPUTE_PGM_LO] = compute_registers[0].value;
        nonrender.sh[P::COMPUTE_PGM_HI] = compute_registers[1].value;
        GpuState::Dispatch direct_dispatch;
        direct_dispatch.threads_x = direct_dispatch.threads_y = direct_dispatch.threads_z = 1;
        direct_dispatch.command_order = 17;
        nonrender.dispatches.push_back(direct_dispatch);
        GpuState::Dispatch unresolved_dispatch;
        unresolved_dispatch.indirect = true;
        unresolved_dispatch.indirect_args_addr = 0xdead00000000ull;
        unresolved_dispatch.command_order = 19;
        nonrender.dispatches.push_back(unresolved_dispatch);
        const std::filesystem::path path =
            prosper_test::test_scratch_dir() / "prosper_nonrender_submit_capture.prgcap";
        std::error_code filesystem_error;
        std::filesystem::remove(path, filesystem_error);
#ifdef _WIN32
        _putenv_s("PROSPER_GPU_CAPTURE", path.string().c_str());
        _putenv_s("PROSPER_GPU_CAPTURE_METADATA_ONLY", "1");
#else
        setenv("PROSPER_GPU_CAPTURE", path.string().c_str(), 1);
        setenv("PROSPER_GPU_CAPTURE_METADATA_ONLY", "1", 1);
#endif
        uint32_t compute_calls = 0;
        set_submit_compute([&](const std::vector<ComputeItem>& items) {
            compute_calls += static_cast<uint32_t>(items.size());
            return !items.empty();
        });
        const bool executed = execute_nonrender_submit_work(nonrender, 1441);
        set_submit_compute({});
#ifdef _WIN32
        _putenv_s("PROSPER_GPU_CAPTURE", "");
        _putenv_s("PROSPER_GPU_CAPTURE_METADATA_ONLY", "");
#else
        unsetenv("PROSPER_GPU_CAPTURE");
        unsetenv("PROSPER_GPU_CAPTURE_METADATA_ONLY");
#endif
        GpuCaptureFile captured;
        std::string capture_error;
        const bool captured_read = read_gpu_capture(path.string(), captured, capture_error);
        const GpuCapturedOperationFailure* unresolved_failure = captured.failure_diagnostics.empty()
            ? nullptr : &captured.failure_diagnostics.front();
        const GpuCapturedStageDiagnostic* unresolved_stage = unresolved_failure &&
            unresolved_failure->stages.size() == 1
                ? &unresolved_failure->stages.front() : nullptr;
        CHECK(executed && compute_calls == 1 && captured_read &&
                  captured.metadata.submit_index == 1441 && captured.computes.size() == 1 &&
                  captured.computes[0].code_addr == reinterpret_cast<uint64_t>(kNoopCs) &&
                  captured.operations.size() == 2 && captured.operations[0].realized &&
                  captured.operations[0].kind == SubmitOperationKind::Dispatch &&
                  captured.operations[0].source_index == 0 &&
                  captured.operations[0].command_order == direct_dispatch.command_order &&
                  !captured.operations[1].realized &&
                  captured.operations[1].kind == SubmitOperationKind::Dispatch &&
                  captured.operations[1].source_index == 1 &&
                  captured.operations[1].command_order == unresolved_dispatch.command_order &&
                  captured.failure_diagnostics.size() == 1 && unresolved_failure &&
                  unresolved_failure->kind == SubmitOperationKind::Dispatch &&
                  unresolved_failure->source_index == 1 &&
                  unresolved_failure->command_order == unresolved_dispatch.command_order &&
                  unresolved_failure->reason ==
                      RealizationFailureReason::Unknown &&
                  unresolved_failure->compute_launch.groups_x == 0 &&
                  unresolved_failure->compute_launch.groups_y == 0 &&
                  unresolved_failure->compute_launch.groups_z == 0 &&
                  unresolved_stage && !unresolved_stage->recompiled &&
                  unresolved_stage->stage == ShaderProgramStage::Compute &&
                  unresolved_stage->program_addr == reinterpret_cast<uint64_t>(kNoopCs) &&
                  unresolved_stage->raw_shader_index <
                      captured.raw_shader_versions.size() &&
                  captured.raw_shader_versions[unresolved_stage->raw_shader_index].words ==
                      std::vector<uint32_t>(std::begin(kNoopCs), std::end(kNoopCs)) &&
                  !captured.expected_output_valid,
              "ordered capture retains exact direct and unresolved indirect compute evidence");
        std::filesystem::remove(path, filesystem_error);

        // A partial exact trace can identify an operation whose source index happens to exist in
        // the semantic state but whose command order does not. Never attach that other dispatch's
        // program or resources to the exact failure.
        constexpr uint64_t kMismatchedOrder = 20;
        auto mismatched_pending = std::make_unique<PendingGpuCapture>();
        mismatched_pending->materialized = false;
        mismatched_pending->path = path.string();
        mismatched_pending->capture.metadata.submit_index = 1442;
        const std::vector<SubmitOperation> mismatched_operations = {
            {SubmitOperationKind::Dispatch, 1, kMismatchedOrder},
        };
        const std::vector<OperationRealizationFailure> mismatched_failures = {
            {SubmitOperationKind::Dispatch, 1, kMismatchedOrder,
             RealizationFailureReason::Unknown},
        };
        const std::vector<DrawItem> no_draws;
        const std::vector<ComputeItem> no_computes;
        const bool mismatched_written = finish_requested_gpu_capture(
            std::move(mismatched_pending), {}, capture_error, &no_draws, &no_computes,
            &mismatched_operations, &nonrender, &mismatched_failures);
        GpuCaptureFile mismatched_capture;
        const bool mismatched_read = mismatched_written &&
            read_gpu_capture(path.string(), mismatched_capture, capture_error);
        CHECK(mismatched_read && mismatched_capture.operations.size() == 1 &&
                  mismatched_capture.operations[0].source_index == 1 &&
                  mismatched_capture.operations[0].command_order == kMismatchedOrder &&
                  !mismatched_capture.operations[0].realized &&
                  mismatched_capture.failure_diagnostics.size() == 1 &&
                  mismatched_capture.failure_diagnostics[0].source_index == 1 &&
                  mismatched_capture.failure_diagnostics[0].command_order == kMismatchedOrder &&
                  mismatched_capture.failure_diagnostics[0].stages.empty(),
              "semantic stage enrichment requires exact dispatch command-order identity");
        std::filesystem::remove(path, filesystem_error);

        // Vulkan guarantees only 65,535 workgroups per axis, but real devices may advertise more
        // (RADV exposes UINT32_MAX in X). Astro Bot emits indirect 1D dispatches above the portable
        // minimum during world-map loading, after already using a 131,072-group direct dispatch.
        // Resolution is device-independent: retain the guest count here and let the Vulkan backend
        // compare it with the selected physical device's actual maxComputeWorkGroupCount.
        alignas(4) uint32_t large_indirect_args[3] = {80266, 1, 1};
        GpuState indirect = nonrender;
        indirect.dispatches.clear();
        GpuState::Dispatch indirect_dispatch;
        indirect_dispatch.indirect = true;
        indirect_dispatch.indirect_args_addr =
            reinterpret_cast<uint64_t>(large_indirect_args);
        indirect_dispatch.command_order = 18;
        indirect.dispatches.push_back(indirect_dispatch);
        ComputeLaunchDimensions resolved_large{};
        compute_calls = 0;
        set_submit_compute([&](const std::vector<ComputeItem>& items) {
            compute_calls += static_cast<uint32_t>(items.size());
            if (!items.empty()) resolved_large = items.front().launch;
            return !items.empty();
        });
        const bool indirect_executed = execute_nonrender_submit_work(indirect, 1442);
        set_submit_compute({});
        CHECK(indirect_executed && compute_calls == 1 &&
                  resolved_large.groups_x == large_indirect_args[0] &&
                  resolved_large.groups_y == 1 && resolved_large.groups_z == 1,
              "indirect dispatch retains workgroup counts above Vulkan's portable minimum");

        // Direct compute packets also need ordered realization. The first backend call stands in for
        // a producer that updates guest-visible shader/resource bytes; the later direct dispatch must
        // be realized only after that producer completes. Eager whole-submit realization would reject
        // kOrderedLateCs while it still contains an unsupported zero word and invoke the backend once.
        ShaderReg late_registers[2] = {
            {P::COMPUTE_PGM_LO, 0}, {P::COMPUTE_PGM_HI, 0},
        };
        AgcShaderHeader late_header{};
        late_header.file_header = 0x34333231u;
        late_header.version = 0x18;
        late_header.sh_registers = late_registers;
        late_header.shader_size = sizeof(kOrderedLateCs);
        late_header.type = 0;
        late_header.num_sh_registers = 2;
        registered_shader = nullptr;
        kOrderedLateCs[0] = 0u;
        CHECK(create_shader &&
                  create_shader(reinterpret_cast<uint64_t>(&registered_shader),
                                reinterpret_cast<uint64_t>(&late_header),
                                reinterpret_cast<uint64_t>(kOrderedLateCs), 0, 0, 0) == 0 &&
                  registered_shader == &late_header,
              "register a mutable late compute consumer for ordered realization");
        GpuState direct_ordered;
        auto first_state = std::make_shared<GpuState>();
        first_state->sh[P::COMPUTE_PGM_LO] = compute_registers[0].value;
        first_state->sh[P::COMPUTE_PGM_HI] = compute_registers[1].value;
        auto late_state = std::make_shared<GpuState>();
        late_state->sh[P::COMPUTE_PGM_LO] = late_registers[0].value;
        late_state->sh[P::COMPUTE_PGM_HI] = late_registers[1].value;
        GpuState::Dispatch first_direct;
        first_direct.threads_x = first_direct.threads_y = first_direct.threads_z = 1;
        first_direct.command_order = 10;
        first_direct.state = first_state;
        GpuState::Dispatch late_direct = first_direct;
        late_direct.command_order = 20;
        late_direct.state = late_state;
        direct_ordered.dispatches = {first_direct, late_direct};
        uint32_t ordered_compute_calls = 0;
        bool late_consumer_executed = false;
        set_submit_compute([&](const std::vector<ComputeItem>& items) {
            if (items.empty()) return false;
            ++ordered_compute_calls;
            if (items.front().code_addr == reinterpret_cast<uint64_t>(kNoopCs)) {
                kOrderedLateCs[0] = 0xBF810000u;
            } else if (items.front().code_addr ==
                       reinterpret_cast<uint64_t>(kOrderedLateCs)) {
                late_consumer_executed = true;
            }
            return true;
        });
        execute_ordered_and_present(direct_ordered, W, H, 1443, /*publish=*/false);
        set_submit_compute({});
        CHECK(ordered_compute_calls == 2 && late_consumer_executed,
              "direct compute consumer is realized after its in-submit producer completes");
        kOrderedLateCs[0] = 0u;
    }

    // Lazy realization can drop a semantic draw only after earlier spans have already executed.
    // If the dropped record was counted as the final span, send an empty terminal callback so the
    // renderer finalizes cached scanout rather than losing the successful work or hanging timing.
    for (bool failed_first : {false, true}) {
        uint8_t source = 0x7A, target = 0;
        GpuState ordered = st;
        ordered.draws.clear();
        ordered.dma_copies.clear();
        GpuState::Draw good{3};
        GpuState::Draw bad{0};
        if (failed_first) {
            bad.command_order = 100;
            good.command_order = 300;
            ordered.draws = {bad, good};
        } else {
            good.command_order = 100;
            bad.command_order = 300;
            ordered.draws = {good, bad};
        }
        ordered.dma_copies.push_back({
            (uint64_t)(uintptr_t)&target, (uint64_t)(uintptr_t)&source,
            sizeof(target), 0, 200, 0});
        std::vector<LiveRenderPhase> phases;
        size_t realized_callbacks = 0, terminal_callbacks = 0;
        set_submit_renderer([&](const std::vector<DrawItem>& items, uint32_t, uint32_t) {
            phases.push_back(live_render_phase());
            if (items.empty()) {
                terminal_callbacks++;
                return RenderedFrame(std::vector<uint8_t>(4, 0x5A));
            }
            realized_callbacks++;
            return RenderedFrame{};
        });
        execute_ordered_and_present(ordered, 1, 1, 78 + failed_first, /*publish=*/false);
        set_submit_renderer({});
        CHECK(realized_callbacks == 1 && terminal_callbacks == 1 && phases.size() == 2,
              failed_first
                  ? "failed leading span still finalizes the later successful span"
                  : "failed trailing span still finalizes the earlier successful span");
        CHECK(phases[0].first_span && !phases[0].final_span &&
              !phases[1].first_span && phases[1].final_span,
              "lazy-span terminal callback closes the submit exactly once");
        CHECK(phases[0].allows_deferred_scanout_readback() == failed_first &&
              !phases[1].allows_deferred_scanout_readback(),
              "terminal finalization preserves the prior span's readback requirement");
    }

    CHECK(!have_submit_renderer(), "no live renderer registered by default (game path stays inert)");
    CHECK(!execute_and_present(st, W, H), "execute_and_present is a no-op with no renderer registered");
    std::shared_ptr<const std::vector<uint8_t>> live_storage;
    unsigned live_calls = 0;
    set_submit_renderer([&](const std::vector<DrawItem>& items, uint32_t w, uint32_t h) -> RenderedFrame {
        if (items.empty()) return {};
        live_calls++;
        live_storage = std::make_shared<const std::vector<uint8_t>>(
            prosper::test::render_triangle_rgba(
                items[0].vs_words(), items[0].fs_words(), w, h, &items[0].ps));
        return RenderedFrame(live_storage);
    });
    CHECK(have_submit_renderer(), "live renderer registered");
    prosper::gpu::present_reset();
    CHECK(!execute_ordered_and_present(st, W, H, 1, /*publish=*/false),
          "ordered publication suppression reports no presented frame");
    CHECK(live_calls == 1, "ordered publication suppression still executes the live renderer");
    CHECK(!prosper::gpu::present_has_frame(),
          "ordered publication suppression leaves the scanout path unchanged");
    CHECK(!execute_and_present(st, W, H, /*publish=*/false),
          "publication suppression reports no presented frame");
    CHECK(live_calls == 2, "publication suppression still executes the live renderer");
    CHECK(!prosper::gpu::present_has_frame(),
          "publication suppression leaves the scanout path unchanged");
    CHECK(execute_and_present(st, W, H), "execute_and_present rendered + presented the submit");
    CHECK(live_calls == 3, "normal publication executes the live renderer again");
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

    // #1434: a shared shader value WINS over the owned vector, so assigning `vs`/`fs` on an item
    // that already carries one silently keeps rendering the ORIGINAL shader. That reads as "my
    // substitution changed nothing" rather than as an error, and it cost real time during the
    // #1427 investigation. set_vs()/set_fs() are the safe form: they drop the shared value and the
    // cache identity, so persistent backend caches compare the new words instead of hitting the
    // stale entry.
    {
        DrawItem shadowed;
        shadowed.vs_shared = std::make_shared<const std::vector<uint32_t>>(
            std::vector<uint32_t>{0x07230203u, 111u});
        shadowed.fs_shared = std::make_shared<const std::vector<uint32_t>>(
            std::vector<uint32_t>{0x07230203u, 222u});
        shadowed.vs_identity = 0xAAAA; shadowed.fs_identity = 0xBBBB;

        shadowed.vs = {0x07230203u, 999u};          // the trap: direct assignment
        CHECK(shadowed.vs_words() == std::vector<uint32_t>({0x07230203u, 111u}),
              "a direct vs assignment is shadowed by vs_shared (the #1434 trap)");
        CHECK(shadowed.has_shadowed_shader(),
              "has_shadowed_shader() reports the shadowed assignment");

        shadowed.set_vs({0x07230203u, 999u});
        CHECK(shadowed.vs_words() == std::vector<uint32_t>({0x07230203u, 999u}) &&
                  !shadowed.vs_shared && shadowed.vs_identity == 0,
              "set_vs substitutes the words and clears the shared value and cache identity");
        CHECK(shadowed.fs_words() == std::vector<uint32_t>({0x07230203u, 222u}) &&
                  shadowed.fs_identity == 0xBBBB,
              "set_vs leaves the fragment stage untouched");

        shadowed.set_fs({0x07230203u, 888u});
        CHECK(shadowed.fs_words() == std::vector<uint32_t>({0x07230203u, 888u}) &&
                  !shadowed.fs_shared && shadowed.fs_identity == 0 &&
                  !shadowed.has_shadowed_shader(),
              "set_fs substitutes the fragment stage and clears the shadow");

        DrawItem owned;                              // no shared value: unchanged behavior
        owned.set_vs({0x07230203u, 7u});
        CHECK(owned.vs_words() == std::vector<uint32_t>({0x07230203u, 7u}) &&
                  !owned.has_shadowed_shader(),
              "set_vs on an item without a shared value behaves like a plain assignment");

        // BackendDraw carries the identical accessor precedence (the line #1434 originally cited),
        // so it gets the same setters. This is the struct the frontend hands the Vulkan backend.
        prosper::test::BackendDraw backend;
        backend.vs_shared = std::make_shared<const std::vector<uint32_t>>(
            std::vector<uint32_t>{0x07230203u, 111u});
        backend.vs_identity = 0xCCCC;
        backend.vs = {0x07230203u, 999u};
        CHECK(backend.vs_words() == std::vector<uint32_t>({0x07230203u, 111u}),
              "BackendDraw shows the same shadowing before substitution (#1434)");
        backend.set_vs({0x07230203u, 999u});
        CHECK(backend.vs_words() == std::vector<uint32_t>({0x07230203u, 999u}) &&
                  !backend.vs_shared && backend.vs_identity == 0,
              "BackendDraw::set_vs substitutes and clears the shared value and identity");
    }


    // #1636: a draw that does not realize must record WHY, at the point it failed. This path used to
    // `break` without touching capture_trace->failures, so gpu_capture synthesized an empty Unknown
    // record and offline inspection reported reason=unknown with no target, extent or pipeline.
    //
    // Each case below asserts the SPECIFIC reason. Asserting merely that a record exists, or that it
    // is non-empty, would pass against a fix that reports Unknown for everything — which is the exact
    // failure mode this guards.
    {
        const auto capture_path =
            prosper_test::test_scratch_dir() / "prosper_draw_failure_reasons.prgcap";
        auto run_and_read = [&](const GpuState& state, uint64_t submit_no,
                                GpuCaptureFile& captured) -> bool {
            std::error_code ec;
            std::filesystem::remove(capture_path, ec);
            // The interactive (F9) arm deliberately bypasses every env AT/AFTER/MIN selector, so it
            // is the only trigger that reliably fires on an arbitrary submit inside a test process.
            request_interactive_gpu_capture(capture_path.string());
            set_submit_renderer([&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                return RenderedFrame{};
            });
            execute_ordered_and_present(state, W, H, submit_no, /*publish=*/false);
            set_submit_renderer({});
            std::string error;
            return read_gpu_capture(capture_path.string(), captured, error);
        };
        auto reason_for_draw = [](const GpuCaptureFile& captured, uint64_t command_order) {
            for (const auto& diagnostic : captured.failure_diagnostics)
                if (diagnostic.kind == SubmitOperationKind::Draw &&
                    diagnostic.command_order == command_order)
                    return diagnostic.reason;
            return RealizationFailureReason::None;
        };

        // (a) Unresolvable indirect arguments — returns before realize_draw_item is ever reached, so
        // it cannot inherit a shader/pipeline diagnostic and needs a reason of its own.
        {
            GpuState indirect = st;
            indirect.draws.clear();
            indirect.dispatches.clear();
            indirect.dma_copies.clear();
            indirect.ordered_memory_effects.clear();
            GpuState::Draw bad_args;
            bad_args.indirect = true;
            bad_args.indirect_args_addr = 0xdead00000000ull;   // not guest-readable
            bad_args.command_order = 300;
            indirect.draws.push_back(bad_args);
            GpuCaptureFile captured;
            const bool read = run_and_read(indirect, 2101, captured);
            const auto reason = reason_for_draw(captured, 300);
            CHECK(read && reason == RealizationFailureReason::IndirectArguments,
                  "#1636: an unresolvable indirect draw records reason=indirect-arguments");
            CHECK(read && reason != RealizationFailureReason::Unknown,
                  "#1636: ...and specifically NOT the old synthesized Unknown");
        }

        // (b) A draw whose realization genuinely fails inside realize_draw_item must FORWARD that
        // reason, not flatten it. No pixel shader is bound here, so the executor knows exactly why.
        {
            uint32_t dma_source[2] = {1, 2};
            uint32_t dma_destination[2] = {};
            GpuState no_program = st;
            no_program.draws.clear();
            no_program.dispatches.clear();
            no_program.dma_copies.clear();
            no_program.ordered_memory_effects.clear();
            no_program.sh[P::SPI_SHADER_PGM_LO_PS] = 0;
            no_program.sh[P::SPI_SHADER_PGM_HI_PS] = 0;
            no_program.sh[P::SPI_SHADER_PGM_LO_ES] = 0;
            no_program.sh[P::SPI_SHADER_PGM_HI_ES] = 0;
            GpuState::Draw plain;
            plain.index_count = 3;
            plain.command_order = 310;
            no_program.draws.push_back(plain);
            // An ordered DMA both routes the submit through the ordered path AND disables eager
            // pre-realization (can_eagerly_realize_draws = !has_ordered_dma && !has_indirect), so
            // the draw is realized by realize_retained_draw — the function this issue is about.
            // With eager realization active the reason is lost in a DIFFERENT pass, whose call
            // site passes no failures out-parameter; that is tracked separately (#1643) and is NOT
            // what this asserts.
            no_program.dma_copies.push_back({
                (uint64_t)(uintptr_t)dma_destination, (uint64_t)(uintptr_t)dma_source,
                sizeof(dma_destination), 0, 305, 0});
            GpuCaptureFile captured;
            const bool read = run_and_read(no_program, 2102, captured);
            const auto reason = reason_for_draw(captured, 310);
            // DELIBERATELY WEAK, and it PASSES against the unfixed code: gpu_capture synthesizes an
            // empty Unknown for any unrealized operation, so "a record exists" proves nothing. Kept
            // next to the specific assertion below as a standing demonstration of that trap — do not
            // "tighten" it, and do not treat it as coverage.
            CHECK(read && reason != RealizationFailureReason::None,
                  "#1636: a draw that fails inside realize_draw_item records a failure at all");
            CHECK(read && reason != RealizationFailureReason::Unknown,
                  "#1636: ...and forwards realize_draw_item's specific reason rather than Unknown");
            CHECK(read && reason == RealizationFailureReason::MissingProgram,
                  "#1636: an unbound shader program records reason=missing-program");
        }

        // (c) The new reasons must have names, or gpu_replay --inspect-only prints "unknown" for them
        // and the fix is invisible exactly where it pays off. (That a new reason survives the capture
        // validator and reader — which both bounded at Filtered — is proven by case (a) above, which
        // round-trips IndirectArguments through a real written .prgcap.)
        CHECK(std::string(realization_failure_reason_name(
                  RealizationFailureReason::IndirectArguments)) == "indirect-arguments" &&
              std::string(realization_failure_reason_name(
                  RealizationFailureReason::IndirectDependencies)) == "indirect-dependencies" &&
              std::string(realization_failure_reason_name(
                  RealizationFailureReason::RetainedDrawNotSelected)) == "retained-draw-not-selected",
              "#1636: the new reasons have names, so gpu_replay --inspect-only can print them");
        std::error_code cleanup;
        std::filesystem::remove(capture_path, cleanup);
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
