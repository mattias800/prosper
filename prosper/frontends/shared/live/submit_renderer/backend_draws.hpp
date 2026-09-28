#pragma once
// Backend draw assembly for one pass group, carved out of the submit callback (#3892).

#include "shared/live/live_renderer_internal.hpp"        // RttCache, backend stats, render state types
#include "shared/live/submit_renderer/callback_types.hpp" // the callback types this context names
#include "shared/live/submit_renderer/draw_resources.hpp" // DrawResourceContext

namespace prosper::frontend::submit_renderer {

// build_backend_draws makes one BackendDraw per realized DrawItem of a pass group: shaders (or the
// diagnostic overrides), the frame resources build_draw_frame_resources resolves, descriptor-contract
// validation, poison mode, indices, and the per-draw census/skip diagnostics. It was the submit
// callback's build_bds lambda; build_bds stays there (and in render_per_target_passes) as a
// one-line forwarder, so its call sites are unchanged. clear_for picks a group's clear colour.

// The submit callback's state that build_backend_draws reads and writes, one reference per object.
struct BackendDrawContext {
    const bool& native_fragment_vote_width;
    const bool& partial_wave_fragment;
    const prosper::gpu::LiveRenderPhase& phase;
    const bool& rtt_log;
    RenderTiming& pending_timing;
    const bool& timing_enabled;
    const bool& use_direct_index_views;
    const char *const& descriptor_validate_mode;
    DrawResourceContext& draw_resource_ctx;
    const bool& refvs;
    std::vector<uint32_t>& refvs_spv;
    std::vector<uint32_t>& ps_override;
    bool& ps_override_is_file;
    bool& ps_override_is_test;
    int& fs_match_mode;
    std::vector<uint32_t>& fs_match;
    const char *& fs_guest_addr_text;
    uint64_t& fs_guest_addr;
    const bool& fs_guest_addr_valid;
    const char *& fs_target_addr_text;
    uint64_t& fs_target_addr;
    const bool& fs_target_addr_valid;
    const char *& fs_target_dim_text;
    uint32_t& fs_target_width;
    uint32_t& fs_target_height;
    const bool& fs_target_dim_valid;
    int& testps_match_mode;
    std::vector<uint32_t>& testps_match;
    const bool& nops;
    const char *& skip_draws_env;
    prosper::gpu::DrawProgramSkipSelector& program_skip;
    const bool& program_skip_armed;
    const bool& program_census;
    prosper::gpu::DrawLinkScanSelector& link_scan;
    const bool& link_scan_armed;
};

std::vector<prosper::test::BackendDraw> build_backend_draws(BackendDrawContext& ctx,
                                                            const std::vector<const prosper::gpu::DrawItem*>& group,
                                                            prosper::test::BackendSubmissionBatch* producer_batch);
const float* clear_for(const std::vector<const prosper::gpu::DrawItem*>& g);

} // namespace prosper::frontend::submit_renderer
