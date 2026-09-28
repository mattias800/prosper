#pragma once
// The per-target pass loop, carved out of the submit callback (#3892).

#include "shared/live/live_renderer_internal.hpp"        // RttCache, backend stats, render state types
#include "shared/live/submit_renderer/callback_types.hpp" // the callback types this context names
#include "shared/live/submit_renderer/backend_draws.hpp"  // BackendDrawContext
#include "shared/live/submit_renderer/backend_timing.hpp" // BackendTimingContext

namespace prosper::frontend::submit_renderer {

// PROSPER_RTT per-target rendering: splits the submit's draws into passes by color target,
// renders each pass into its own retained target (resolves, MRT slots, RTT publication and the
// present-candidate bookkeeping included), and leaves the composite's pixels for the present
// selection that follows. This was the body of the submit callback's `if (pertarget)` branch.
// What it shares with the rest of the callback arrives by name: build_backend_draws and
// record_backend_timing_stats over the callback's contexts, plus clear_for / print_rtt_timing.

// The submit callback's state that render_per_target_passes reads and writes, one reference per object.
struct PerTargetPassContext {
    RttCache& g_rtt;
    std::atomic<uint64_t>& g_pass_log_submit;
    prosper::frontend::DiagnosticWindow& g_pass_log_window;
    prosper::frontend::DiagnosticWindow& g_persist_window;
    const PersistentReadbackFilter& g_persist_filter;
    const std::pair<uint32_t, uint32_t>& g_persist_extent;
    std::atomic<int>& frame_no;
    const bool& live_gpu_targets;
    const bool& defer_intermediate_scanout;
    const bool& batch_backend_submits;
    const std::vector<prosper::gpu::DrawItem> & items;
    uint32_t& w;
    uint32_t& h;
    const prosper::gpu::LiveRenderPhase& phase;
    std::vector<PinnedScanout>& pinned_scanouts;
    std::vector<PinnedRendererMipTarget>& pinned_renderer_mip_targets;
    int& g_this_submit;
    const bool& rtt_log;
    RenderTiming& pending_timing;
    std::vector<RttTimingRecord>& pending_rtt_timing;
    const bool& timing_enabled;
    const bool& lightweight_rtt_timing;
    std::vector<std::vector<uint8_t>>& texstore;
    std::vector<bool>& texstore_pinned;
    size_t& texstore_used;
    std::unordered_map<TextureDecodeKey, DecodedTexture, TextureDecodeKeyHash>& decoded_textures;
    uint32_t& resource_hash_w;
    uint32_t& resource_hash_h;
    uint32_t& target_step_w;
    uint32_t& target_step_h;
    const size_t& target_step_min_draws;
    RenderClock::time_point& pass_tail_start;
    double& pass_tail_measured_before;
    const RenderClock::time_point& pass_timing_start;
    std::shared_ptr<const std::vector<uint8_t>>& selected_pixels;
    uint64_t& selected_source_submit;
    const size_t& present_extent_bytes;
    prosper::frontend::PresentSourceChoice& present_choice;
    uint32_t& px_front_w;
    uint32_t& px_front_h;
    uint32_t& px_vo_w;
    uint32_t& px_vo_h;
    uint32_t& px_last_w;
    uint32_t& px_last_h;
    uint64_t& px_front_base;
    uint64_t& px_vo_base;
    uint64_t& px_last_base;
    uint64_t& px_front_source_submit;
    uint64_t& px_vo_source_submit;
    uint64_t& px_last_source_submit;
    uint32_t& px_front_fmt;
    uint32_t& px_vo_fmt;
    uint32_t& px_last_fmt;
    BackendDrawContext& backend_draw_ctx;
    BackendTimingContext& backend_timing_ctx;
};

void render_per_target_passes(PerTargetPassContext& ctx);

} // namespace prosper::frontend::submit_renderer
