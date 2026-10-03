// Calling-thread callback prelude; see callback_thread_state.hpp (#3892).
#include "shared/live/submit_renderer/callback_thread_state.hpp"

namespace prosper::frontend::submit_renderer {

CallbackThreadState& CallbackThreadState::current() {
    static thread_local CallbackThreadState state;
    return state;
}

prosper::frontend::WriteWatchPromotionBudget& CallbackThreadState::write_watch_promotion_budget() {
    static thread_local auto& value = write_watch_promotion_budget_.emplace();
    return value;
}

std::vector<PinnedScanout>& CallbackThreadState::pinned_scanouts() {
    static thread_local auto& value = pinned_scanouts_.emplace();
    return value;
}

std::vector<PinnedRendererMipTarget>& CallbackThreadState::pinned_renderer_mip_targets() {
    static thread_local auto& value = pinned_renderer_mip_targets_.emplace();
    return value;
}

int& CallbackThreadState::g_this_submit() {
    static thread_local auto& value = g_this_submit_.emplace(-1);
    return value;
}

bool& CallbackThreadState::g_force_this_submit() {
    static thread_local auto& value = g_force_this_submit_.emplace(false);
    return value;
}

RenderTiming& CallbackThreadState::pending_timing() {
    static thread_local auto& value = pending_timing_.emplace();
    return value;
}

std::vector<RttTimingRecord>& CallbackThreadState::pending_rtt_timing() {
    static thread_local auto& value = pending_rtt_timing_.emplace();
    return value;
}

uint64_t& CallbackThreadState::pending_span_start_ns() {
    static thread_local auto& value = pending_span_start_ns_.emplace(0);
    return value;
}

uint64_t& CallbackThreadState::pending_capture_generation() {
    static thread_local auto& value = pending_capture_generation_.emplace(0);
    return value;
}

ValidationCensusLog& CallbackThreadState::validation_census_log() {
    static thread_local auto& value = validation_census_log_.emplace();
    return value;
}

std::vector<std::vector<uint8_t>>& CallbackThreadState::texstore() {
    static thread_local auto& value = texstore_.emplace();
    return value;
}

std::vector<bool>& CallbackThreadState::texstore_pinned() {
    static thread_local auto& value = texstore_pinned_.emplace();
    return value;
}

std::unordered_map<TextureDecodeKey, DecodedTexture, TextureDecodeKeyHash>& CallbackThreadState::decoded_textures() {
    static thread_local auto& value = decoded_textures_.emplace();
    return value;
}

uint64_t& CallbackThreadState::decode_span_ordinal() {
    static thread_local auto& value = decode_span_ordinal_.emplace(0);
    return value;
}

int& CallbackThreadState::decode_scope_submit() {
    static thread_local auto& value = decode_scope_submit_.emplace(-1);
    return value;
}

std::vector<std::shared_ptr<const std::vector<uint8_t>>>& CallbackThreadState::retired_submit_pixels() {
    static thread_local auto& value = retired_submit_pixels_.emplace();
    return value;
}

std::unordered_map<uint64_t, ReflectMemoEntry>& CallbackThreadState::reflect_memo() {
    static thread_local auto& value = reflect_memo_.emplace();
    return value;
}

SingleFramebufferSubmitFrame& CallbackThreadState::single_framebuffer_submit_frame() {
    static thread_local auto& value = single_framebuffer_submit_frame_.emplace();
    return value;
}

} // namespace prosper::frontend::submit_renderer
