#pragma once
// The single-framebuffer (non-PROSPER_RTT) pass, carved out of the submit callback (#3892).

#include "shared/live/live_renderer_internal.hpp"        // RttCache, backend stats, render state types
#include "shared/live/submit_renderer/callback_types.hpp" // the callback types these contexts name
#include "shared/live/submit_renderer/backend_draws.hpp"  // BackendDrawContext, build_backend_draws, clear_for
#include "shared/live/submit_renderer/backend_timing.hpp" // BackendTimingContext, record_backend_timing_stats

namespace prosper::frontend::submit_renderer {

// ---- Single-framebuffer path (#3892) -----------------------------------------------------------
//
// Without PROSPER_RTT's per-target passes, every draw of the submit composites into one framebuffer,
// which becomes the present source. The callback then caches it under the first colour target (that
// store, and its PROSPER_RTTLOG report, stay at the call site). Moved out of the submit callback
// verbatim.
// The submit callback's state that render_single_framebuffer reads and writes, one reference per object.
struct SingleFramebufferContext {
    const std::vector<prosper::gpu::DrawItem> & items;
    uint32_t& w;
    uint32_t& h;
    const prosper::gpu::LiveRenderPhase& phase;
    RenderTiming& pending_timing;
    const bool& timing_enabled;
    BackendTimingContext& backend_timing_ctx;
    BackendDrawContext& backend_draw_ctx;
    std::shared_ptr<const std::vector<uint8_t>>& selected_pixels;
    uint64_t& selected_source_submit;
};
void render_single_framebuffer(SingleFramebufferContext& ctx);

} // namespace prosper::frontend::submit_renderer
