#pragma once
// Process-lifetime prelude state shared by every live-renderer registration (#3892).
// Empty slots are constructed at registration. Each accessor constructs its value only on first
// reach of the callback's original binding; native function-static guards retain thread-safe once
// initialization and retry after an initializer throws. Keep those bindings in place and in order.
// The callback's thread_local state is deliberately not owned here.

#include "shared/live/live_renderer_internal.hpp" // decoded-cache key/value types

#include <optional>

namespace prosper::frontend::submit_renderer {

class CallbackState {
public:
    static CallbackState& instance();

    const size_t& write_watch_promotion_budget_bytes();
    std::atomic<int>& g_submit_idx();
    int& g_render_first();
    const int64_t& g_render_delay_ms();
    const std::chrono::steady_clock::time_point& g_render_delay_start();
    std::atomic<bool>& g_render_delay_announced();
    int& g_render_last();
    const int& g_rttlog_min_submit();
    const int& g_rttlog_max_submit();
    const bool& validation_census_requested();
    const uint64_t& rtt_timing_min_draws();
    const bool& submit_decode_scope_disabled();
    const bool& use_tracked_buffer_membership_cache();
    std::unordered_map<TextureDecodeKey, PersistentDecodedTexture, TextureDecodeKeyHash>& persistent_decoded_textures();
    size_t& persistent_decoded_texture_bytes();
    uint64_t& persistent_decode_generation();
    size_t& retired_submit_bytes();
    uint64_t& persistent_texture_id();
    std::vector<uint8_t>& persistent_validation_scratch();
    const size_t& persistent_decode_limit();
    const bool& reserve_frame_resources();
    const bool& disable_guest_depth_layers();

    CallbackState(const CallbackState&) = delete;
    CallbackState& operator=(const CallbackState&) = delete;

private:
    CallbackState() = default;

    std::optional<size_t> write_watch_promotion_budget_bytes_;
    std::optional<std::atomic<int>> g_submit_idx_;
    std::optional<int> g_render_first_;
    std::optional<int64_t> g_render_delay_ms_;
    std::optional<std::chrono::steady_clock::time_point> g_render_delay_start_;
    std::optional<std::atomic<bool>> g_render_delay_announced_;
    std::optional<int> g_render_last_;
    std::optional<int> g_rttlog_min_submit_;
    std::optional<int> g_rttlog_max_submit_;
    std::optional<bool> validation_census_requested_;
    std::optional<uint64_t> rtt_timing_min_draws_;
    std::optional<bool> submit_decode_scope_disabled_;
    std::optional<bool> use_tracked_buffer_membership_cache_;
    std::optional<std::unordered_map<TextureDecodeKey, PersistentDecodedTexture, TextureDecodeKeyHash>> persistent_decoded_textures_;
    std::optional<size_t> persistent_decoded_texture_bytes_;
    std::optional<uint64_t> persistent_decode_generation_;
    std::optional<size_t> retired_submit_bytes_;
    std::optional<uint64_t> persistent_texture_id_;
    std::optional<std::vector<uint8_t>> persistent_validation_scratch_;
    std::optional<size_t> persistent_decode_limit_;
    std::optional<bool> reserve_frame_resources_;
    std::optional<bool> disable_guest_depth_layers_;
};

} // namespace prosper::frontend::submit_renderer
