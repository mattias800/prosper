#pragma once
// Calling-thread prelude state shared by live-renderer registrations (#3892).
// Resolve current() inside the callback, never at registration or through a captured thread owner.
// Empty optional slots do not construct payloads. Native TLS reference guards construct each slot
// only at its original callback binding, including the conditional census site, and retry there
// after an initializer throws. Members keep the original relative first-reach order; census reach
// depends on immutable process configuration, while F8 changes its active/data fields.
// Payload destructors use their own CPU state or process device/staging/accounting ownership;
// storage-image writebacks run inline after completion, on the thread that bound their aliases.

#include "shared/live/live_renderer_internal.hpp"
#include "shared/live/submit_renderer/callback_types.hpp"
#include "shared/present/single_framebuffer_submit_policy.hpp"
#include "shared/texture/write_watch_policy.hpp"

#include <optional>

namespace prosper::frontend::submit_renderer {

class CallbackThreadState {
public:
    static CallbackThreadState& current();

    prosper::frontend::WriteWatchPromotionBudget& write_watch_promotion_budget();
    std::vector<PinnedScanout>& pinned_scanouts();
    std::vector<PinnedRendererMipTarget>& pinned_renderer_mip_targets();
    int& g_this_submit();
    bool& g_force_this_submit();
    RenderTiming& pending_timing();
    std::vector<RttTimingRecord>& pending_rtt_timing();
    uint64_t& pending_span_start_ns();
    uint64_t& pending_capture_generation();
    ValidationCensusLog& validation_census_log();
    std::vector<std::vector<uint8_t>>& texstore();
    std::vector<bool>& texstore_pinned();
    std::unordered_map<TextureDecodeKey, DecodedTexture, TextureDecodeKeyHash>& decoded_textures();
    uint64_t& decode_span_ordinal();
    int& decode_scope_submit();
    std::vector<std::shared_ptr<const std::vector<uint8_t>>>& retired_submit_pixels();
    std::unordered_map<uint64_t, ReflectMemoEntry>& reflect_memo();
    SingleFramebufferSubmitFrame& single_framebuffer_submit_frame();

    CallbackThreadState(const CallbackThreadState&) = delete;
    CallbackThreadState& operator=(const CallbackThreadState&) = delete;

private:
    CallbackThreadState() = default;
    ~CallbackThreadState() = default;

    std::optional<prosper::frontend::WriteWatchPromotionBudget> write_watch_promotion_budget_;
    std::optional<std::vector<PinnedScanout>> pinned_scanouts_;
    std::optional<std::vector<PinnedRendererMipTarget>> pinned_renderer_mip_targets_;
    std::optional<int> g_this_submit_;
    std::optional<bool> g_force_this_submit_;
    std::optional<RenderTiming> pending_timing_;
    std::optional<std::vector<RttTimingRecord>> pending_rtt_timing_;
    std::optional<uint64_t> pending_span_start_ns_;
    std::optional<uint64_t> pending_capture_generation_;
    std::optional<ValidationCensusLog> validation_census_log_;
    std::optional<std::vector<std::vector<uint8_t>>> texstore_;
    std::optional<std::vector<bool>> texstore_pinned_;
    std::optional<std::unordered_map<TextureDecodeKey, DecodedTexture, TextureDecodeKeyHash>> decoded_textures_;
    std::optional<uint64_t> decode_span_ordinal_;
    std::optional<int> decode_scope_submit_;
    std::optional<std::vector<std::shared_ptr<const std::vector<uint8_t>>>> retired_submit_pixels_;
    std::optional<std::unordered_map<uint64_t, ReflectMemoEntry>> reflect_memo_;
    std::optional<SingleFramebufferSubmitFrame> single_framebuffer_submit_frame_;
};

} // namespace prosper::frontend::submit_renderer
