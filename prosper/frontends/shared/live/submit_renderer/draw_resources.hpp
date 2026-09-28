#pragma once
// One draw's frame resources, carved out of the submit-renderer callback in live_renderer.cpp (#3892).

#include "shared/live/live_renderer_internal.hpp"        // RttCache, texture-decode cache types
#include "shared/live/submit_renderer/callback_types.hpp" // the callback types this context names

namespace prosper::frontend::submit_renderer {

// ---- One draw's frame resources (#3892) ------------------------------------------------------
//
// Resolves every texture and buffer reference of one draw's VS (set 0) and PS (set 1) resource
// tables into backend frame resources. This was the submit callback's `build_R` lambda; its body
// moved here verbatim, and every callback-scope object it touches arrives by reference through
// DrawResourceContext (built once per callback), so the per-draw path copies nothing.
// The submit callback's state that build_draw_frame_resources reads and writes, one reference per object.
struct DrawResourceContext {
    RttCache& g_rtt;
    std::atomic<uint64_t>& g_pass_log_submit;
    const bool& invalidate_ds;
    std::atomic<int>& frame_no;
    const bool& rtt_on;
    const bool& live_gpu_targets;
    prosper::frontend::WriteWatchPromotionBudget& write_watch_promotion_budget;
    std::vector<PinnedRendererMipTarget>& pinned_renderer_mip_targets;
    int& g_this_submit;
    const bool& rtt_log;
    RenderTiming& pending_timing;
    ValidationCensusLog *& validation_census;
    const bool& timing_enabled;
    const bool& render_timing_detail;
    std::vector<std::vector<uint8_t>>& texstore;
    std::vector<bool>& texstore_pinned;
    size_t& texstore_used;
    std::unordered_map<TextureDecodeKey, DecodedTexture, TextureDecodeKeyHash>& decoded_textures;
    uint64_t& decode_span_ordinal;
    const bool& use_direct_buffer_views;
    const bool& use_tracked_buffer_membership_cache;
    std::unordered_map<TextureDecodeKey, PersistentDecodedTexture, TextureDecodeKeyHash>& persistent_decoded_textures;
    size_t& persistent_decoded_texture_bytes;
    const uint64_t& decode_generation;
    std::vector<std::shared_ptr<const std::vector<uint8_t>>>& retired_submit_pixels;
    size_t& retired_submit_bytes;
    uint64_t& persistent_texture_id;
    std::vector<uint8_t>& persistent_validation_scratch;
    const size_t& persistent_decode_limit;
    uint32_t& resource_hash_w;
    uint32_t& resource_hash_h;
    prosper::frontend::RttInjectionCache& rtt_injection_cache;
    std::unordered_map<uint64_t, ReflectMemoEntry>& reflect_memo;
    const bool& compact_buffer_resources;
    const bool& reserve_frame_resources;
    DepthArraySnapshotCensus& depth_array_census;
    std::vector<DepthArraySnapshot>& depth_array_snapshots;
    size_t& depth_array_snapshot_bytes;
    bool& depth_array_snapshot_admitted;
    const bool& reuse_depth_arrays_across_draws;
    const bool& compact_depth_array_snapshots;
    const bool& gpu_depth_array_snapshots;
    const bool& disable_guest_depth_layers;
    prosper::test::DepthArrayGuestScanMemo& depth_array_guest_scans;
    uint64_t& depth_array_guest_scan_epoch;
    const bool& gpu_depth_cube_snapshots;
    std::vector<DepthCubeGpuSnapshot>& depth_cube_gpu_snapshots;
};

BuiltFrameResources build_draw_frame_resources(DrawResourceContext& ctx,
                                               const prosper::gpu::DrawItem& draw,
                                               const prosper::gpu::ShaderResourceTable* vrt,
                                               const prosper::gpu::ShaderResourceTable* prt,
                                               prosper::test::BackendSubmissionBatch* producer_batch);

} // namespace prosper::frontend::submit_renderer
