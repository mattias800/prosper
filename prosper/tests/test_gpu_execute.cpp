// test_gpu_execute — the GPU-executor core spine (Stage A): a GpuState (exactly what SubmitDcb folds a
// Dcb into) -> execute_gpustate() [recompile shaders from their PGM addresses + resolve pipeline] ->
// a caller-supplied Vulkan render -> present_write_frame -> present_readback. Proves the executor entry
// point that agc_driver_submit_dcb will call, and the scanout round-trip, end to end on llvmpipe.
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/gpu_capture.hpp"
#include "../src/gpu/videoout_present.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include "../src/gpu/vk_translate.hpp"
#include "../src/host/guest_write_watch.hpp"
#include "render_runner.h"
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
static void set_pgm(GpuState& st, uint32_t lo_off, uint32_t hi_off, const void* p) {
    uint64_t a = (uint64_t)(uintptr_t)p;
    st.sh[lo_off] = (uint32_t)((a >> 8) & 0xFFFFFFFFu);
    st.sh[hi_off] = (uint32_t)((a >> 40) & 0xFFu);
}

int main() {
    printf("== test_gpu_execute ==\n");
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

    // Compute-only and DMA-only submissions bypass presentation, but an environment capture aimed
    // at an unsupported compute program must still retain that semantic submit.  Exercise the
    // non-render hook with an ordered DMA operation here; the compute selector/failure closure is
    // covered by test_gpu_capture's semantic-dispatch cases.
    {
        uint32_t source = 0x13579BDFu, destination = 0;
        GpuState nonrender;
        nonrender.dma_copies.push_back({
            reinterpret_cast<uint64_t>(&destination), reinterpret_cast<uint64_t>(&source),
            sizeof(source), 0, 17, 0});
        const std::filesystem::path path =
            std::filesystem::temp_directory_path() / "prosper_nonrender_submit_capture.prgcap";
        std::error_code filesystem_error;
        std::filesystem::remove(path, filesystem_error);
#ifdef _WIN32
        _putenv_s("PROSPER_GPU_CAPTURE", path.string().c_str());
        _putenv_s("PROSPER_GPU_CAPTURE_METADATA_ONLY", "1");
#else
        setenv("PROSPER_GPU_CAPTURE", path.string().c_str(), 1);
        setenv("PROSPER_GPU_CAPTURE_METADATA_ONLY", "1", 1);
#endif
        const bool executed = execute_nonrender_submit_work(nonrender, 1441);
#ifdef _WIN32
        _putenv_s("PROSPER_GPU_CAPTURE", "");
        _putenv_s("PROSPER_GPU_CAPTURE_METADATA_ONLY", "");
#else
        unsetenv("PROSPER_GPU_CAPTURE");
        unsetenv("PROSPER_GPU_CAPTURE_METADATA_ONLY");
#endif
        GpuCaptureFile captured;
        std::string capture_error;
        CHECK(executed && destination == source &&
                  read_gpu_capture(path.string(), captured, capture_error) &&
                  captured.metadata.submit_index == 1441 && captured.dma_copies.size() == 1 &&
                  captured.operations.size() == 1 &&
                  captured.operations[0].kind == SubmitOperationKind::DmaCopy &&
                  !captured.expected_output_valid,
              "environment capture retains a non-render submit without inventing an output oracle");
        std::filesystem::remove(path, filesystem_error);
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

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
