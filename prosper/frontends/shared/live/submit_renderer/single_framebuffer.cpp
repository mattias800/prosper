// render_single_framebuffer -- see single_framebuffer.hpp. Moved verbatim out of live_renderer.cpp (#3892).
#include "shared/live/submit_renderer/single_framebuffer.hpp"

namespace prosper::frontend::submit_renderer {

void render_single_framebuffer(SingleFramebufferContext& ctx) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& items = ctx.items;
    auto& w = ctx.w;
    auto& h = ctx.h;
    auto& phase = ctx.phase;
    auto& pending_timing = ctx.pending_timing;
    auto& timing_enabled = ctx.timing_enabled;
    auto& backend_timing_ctx = ctx.backend_timing_ctx;
    auto& backend_draw_ctx = ctx.backend_draw_ctx;
    auto& selected_pixels = ctx.selected_pixels;
    auto& selected_source_submit = ctx.selected_source_submit;
    // Single-framebuffer path: render_draws_rgba composites every draw into ONE framebuffer.
    std::vector<const prosper::gpu::DrawItem*> all; all.reserve(items.size());
    for (const auto& it : items) all.push_back(&it);
    const bool perf_build_clock = prosper::diagnostics::perf::enabled();   // #3891
    const auto build_start = timing_enabled || perf_build_clock
        ? RenderClock::now() : RenderClock::time_point{};
    auto backend_draws = build_backend_draws(backend_draw_ctx, all, nullptr);
    const auto build_done = timing_enabled || perf_build_clock
        ? RenderClock::now() : RenderClock::time_point{};
    if (perf_build_clock) {
        prosper::diagnostics::perf::add_cost(
            prosper::diagnostics::perf::Cost::FrontendBuild,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                build_done - build_start).count()));
        prosper::diagnostics::perf::flush_thread_texture_references();
    }
    auto rendered = prosper::test::render_draws_rgba(
        backend_draws, w, h, nullptr, clear_for(all), true);
    const auto backend_done = timing_enabled
        ? RenderClock::now() : RenderClock::time_point{};
    const prosper::test::BackendRenderTimingStats backend_call_timing = timing_enabled
        ? prosper::test::backend_render_timing_stats()
        : prosper::test::BackendRenderTimingStats{};
    const prosper::test::BackendTextureUploadStats backend_texture_stats = timing_enabled
        ? prosper::test::backend_texture_upload_stats()
        : prosper::test::BackendTextureUploadStats{};
    const prosper::test::BackendPipelineCacheStats backend_pipeline_stats = timing_enabled
        ? prosper::test::backend_pipeline_cache_stats()
        : prosper::test::BackendPipelineCacheStats{};
    // Captured HERE, beside the pipeline stats, and PASSED IN -- not read inside
    // record_backend_timing. These are thread_local and published by the backend call;
    // reading them from wherever the recorder happens to run returns a different,
    // nearly empty instance. The first draft did exactly that and reported refs=11 for
    // a 2,102-draw submit -- the kind of small plausible number that gets believed.
    const prosper::test::BackendResourceReuseStats backend_reuse_stats = timing_enabled
        ? prosper::test::backend_resource_reuse_stats()
        : prosper::test::BackendResourceReuseStats{};
    if (timing_enabled) {
        pending_timing.build_resources_ms +=
            std::chrono::duration<double, std::milli>(build_done - build_start).count();
        pending_timing.backend_ms +=
            std::chrono::duration<double, std::milli>(backend_done - build_done).count();
        record_backend_timing_stats(backend_timing_ctx, backend_call_timing,
                                    backend_texture_stats, backend_pipeline_stats,
                                    backend_reuse_stats);
    }
    // RTT (#167): cache these rendered pixels under this submit's render-target base, so a later
    // composite pass that samples that address gets the scene we drew (not empty guest memory).
    selected_pixels = std::make_shared<const std::vector<uint8_t>>(std::move(rendered));
    selected_source_submit = !items.empty() ? phase.source_submit : 0;
}

} // namespace prosper::frontend::submit_renderer
