#pragma once
// The per-target pass loop's diagnostics, carved out of it (#3892).

#include "shared/live/live_renderer_internal.hpp"        // RttCache, backend stats, render state types
#include "shared/live/submit_renderer/callback_types.hpp" // the callback types these contexts name
#include "shared/live/submit_renderer/backend_draws.hpp"  // BackendDrawContext (the two re-rendering dumps)

namespace prosper::frontend::submit_renderer {

// Each function is one PROSPER_* diagnostic block of render_per_target_passes, moved verbatim, with
// a context holding references to the pass-loop objects it reads. A block gated by a PROSPER_* test
// carries that test with it, so check_diag_gates sees the gate beside the report. A block gated by a
// pass-loop value keeps that gate at its call site, so a false gate builds no context.

// ---- PROSPER_RESOLVE_CENSUS (#3892) -------------------------------------------------------------
// The pass loop's state that note_resolve_census reads, one reference per object.
struct ResolveCensusContext {
    std::vector<const prosper::gpu::DrawItem *>& pass;
};
void note_resolve_census(ResolveCensusContext& ctx);

// ---- PROSPER_READBACK_WHY (#3892) ---------------------------------------------------------------
// The pass loop's state that note_readback_why reads, one reference per object.
struct ReadbackWhyContext {
    const bool & live_gpu_targets;
    const prosper::gpu::LiveRenderPhase & phase;
    const int& vo_n;
    const uint64_t& front_va;
    uint64_t& base;
    std::array<uint64_t, prosper::gpu::kColorTargetCount>& pass_bases;
    uint32_t& gw;
    uint32_t& gh;
    bool& is_vo;
    const VkFormat& pass_format;
    uint32_t& mrt_count;
    const LaterTargetConsumers& consumers0;
    const bool& sampled_exact_later;
    const bool& feedback_later;
    const bool& cpu_needed_same_batch;
};
void note_readback_why(ReadbackWhyContext& ctx);

// ---- PROSPER_RTT_RESIDENCY_TRACE for a rendered pass (#3892) ------------------------------------
// The pass loop's state that trace_pass_residency reads, one reference per object.
struct PassResidencyTraceContext {
    std::atomic<uint64_t> & g_pass_log_submit;
    const bool & live_gpu_targets;
    RttCache & g_rtt;
    size_t& pass_i;
    std::array<uint64_t, prosper::gpu::kColorTargetCount>& pass_bases;
    std::array<VkFormat, prosper::gpu::kColorTargetCount>& pass_formats;
    uint32_t& mrt_count;
    const prosper::test::BackendColorTargetStats& color_target_call;
    const DiagnosticAddressSelector & residency_trace;
};
void trace_pass_residency(PassResidencyTraceContext& ctx);

// ---- PROSPER_RESOURCE_HASH_DIM (#3892) ----------------------------------------------------------
// The pass loop's state that report_pass_resource_hash reads, one reference per object.
struct PassResourceHashContext {
    int & g_this_submit;
    uint64_t& base;
    uint32_t& native_w;
    uint32_t& native_h;
    std::vector<const prosper::gpu::DrawItem *>& render_pass;
    const uint8_t *& seed;
    const std::vector<uint8_t> & rendered_pixels;
};
void report_pass_resource_hash(PassResourceHashContext& ctx);

// ---- PROSPER_TARGET_STEP_HASH_DIM (#3892) -------------------------------------------------------
// The pass loop's state that report_target_step_hashes reads, one reference per object.
struct TargetStepHashContext {
    int & g_this_submit;
    std::vector<std::vector<uint8_t>> & texstore;
    std::vector<bool> & texstore_pinned;
    size_t & texstore_used;
    std::unordered_map<TextureDecodeKey, DecodedTexture, TextureDecodeKeyHash> & decoded_textures;
    uint64_t& base;
    uint32_t& gw;
    uint32_t& gh;
    std::vector<const prosper::gpu::DrawItem *>& render_pass;
    const uint8_t *& seed;
    BackendDrawContext& backend_draw_ctx;
};
void report_target_step_hashes(TargetStepHashContext& ctx);

// ---- PROSPER_RTTLOG pass report (#3892) ---------------------------------------------------------
// The pass loop's state that log_rendered_pass reads, one reference per object.
struct RenderedPassLogContext {
    RttCache & g_rtt;
    const uint64_t& front_va;
    uint64_t& base;
    std::vector<const prosper::gpu::DrawItem *>& pass;
    uint32_t& native_w;
    uint32_t& native_h;
    uint32_t& gw;
    uint32_t& gh;
    bool& is_vo;
    const VkFormat& pass_format;
    const std::vector<uint8_t> & rendered_pixels;
};
void log_rendered_pass(RenderedPassLogContext& ctx);

// ---- PROSPER_DUMP_PASS (#3892) ------------------------------------------------------------------
// The pass loop's state that dump_rendered_pass reads, one reference per object.
struct RenderedPassDumpContext {
    uint64_t& base;
    std::vector<const prosper::gpu::DrawItem *>& pass;
    uint32_t& gw;
    uint32_t& gh;
    const VkFormat& pass_format;
    const std::string& dump_spec;
    const int& dump_pass_every;
    const std::vector<uint8_t> & rendered_pixels;
};
void dump_rendered_pass(RenderedPassDumpContext& ctx);

// ---- PROSPER_DUMP_DRAWSTEPS (#3892) -------------------------------------------------------------
// The pass loop's state that dump_pass_draw_steps reads, one reference per object.
struct PassDrawStepsContext {
    std::atomic<int> & frame_no;
    uint64_t& base;
    std::vector<const prosper::gpu::DrawItem *>& pass;
    uint32_t& gw;
    uint32_t& gh;
    bool& is_vo;
    std::vector<const prosper::gpu::DrawItem *>& render_pass;
    const uint8_t *& seed;
    BackendDrawContext& backend_draw_ctx;
};
void dump_pass_draw_steps(PassDrawStepsContext& ctx);

// ---- PROSPER_DUMP_RTGROUPS / _RGBA (#3892) ------------------------------------------------------
// The pass loop's state that dump_rt_group reads, one reference per object.
struct RtGroupDumpContext {
    std::atomic<int> & frame_no;
    uint64_t& base;
    std::array<uint64_t, prosper::gpu::kColorTargetCount>& pass_bases;
    uint32_t& gw;
    uint32_t& gh;
    std::vector<const prosper::gpu::DrawItem *>& render_pass;
    const VkFormat& pass_format;
    const std::vector<uint8_t> & rendered_pixels;
};
void dump_rt_group(RtGroupDumpContext& ctx);

// ---- PROSPER_PASS_LOG per-pass report (#3892) ---------------------------------------------------
// The pass loop's state that log_pass_publish reads, one reference per object.
struct PassPublishLogContext {
    std::atomic<uint64_t> & g_pass_log_submit;
    prosper::frontend::DiagnosticWindow & g_pass_log_window;
    const std::vector<prosper::gpu::DrawItem> & items;
    size_t& pass_i;
    uint64_t& base;
    std::vector<const prosper::gpu::DrawItem *>& pass;
    uint32_t& gw;
    uint32_t& gh;
    bool& is_vo;
    const bool& seed_rtt0;
    const VkFormat& pass_format;
    const bool& defer_readback;
    const prosper::test::BackendColorTargetStats& color_target_call;
    const std::vector<uint8_t> & rendered_pixels;
};
void log_pass_publish(PassPublishLogContext& ctx);

// ---- PROSPER_DUMP_PERSISTENT (#3892) ------------------------------------------------------------
// The pass loop's state that dump_persistent_targets reads, one reference per object.
struct PersistentTargetDumpContext {
    prosper::frontend::DiagnosticWindow & g_persist_window;
    const PersistentReadbackFilter & g_persist_filter;
    const std::pair<uint32_t, uint32_t> & g_persist_extent;
    prosper::frontend::PresentSourceChoice & present_choice;
    RttCache & g_rtt;
    const int& vo_n;
    const int& vo_front;
    const uint64_t& front_va;
};
void dump_persistent_targets(PersistentTargetDumpContext& ctx);

// ---- PROSPER_MRT_CENSUS (#3892) ----------------------------------------------------------------
// The pass loop's state that note_mrt_census reads, one reference per object.
struct MrtCensusContext {
    const std::vector<prosper::gpu::DrawItem> & items;
    size_t& pass_i;
};
void note_mrt_census(MrtCensusContext& ctx);

} // namespace prosper::frontend::submit_renderer
