#pragma once

#include <charconv>
#include <cstdint>
#include <string_view>
#include <system_error>

namespace prosper::frontend {

// Parse the one-based host-present frame used by prosper-app's diagnostic screenshot trigger.
// Zero and malformed values disable the trigger so a typo cannot unexpectedly force a 4K readback.
inline uint64_t parse_capture_frame(const char* value) {
    if (!value || !*value) return 0;
    const std::string_view text(value);
    uint64_t frame = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), frame, 10);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && frame
        ? frame : 0;
}

// Parse a positive guest-pad flip ordinal. Unlike `parse_capture_frame`, this is signed because
// prosper_pad_flip_ordinal() reports -1 until the guest has established the pad-route origin.
// Zero and malformed input disable the trigger; values beyond INT64_MAX must not wrap into a
// negative target that can never be reached.
inline int64_t parse_capture_pad_flip(const char* value) {
    if (!value || !*value) return 0;
    const std::string_view text(value);
    int64_t ordinal = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), ordinal, 10);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && ordinal > 0
        ? ordinal : 0;
}

// `shown` counts successful presents completed before the candidate frame. If swapchain acquisition
// skipped the exact requested ordinal, keep the one-shot eligible and capture the next real present.
constexpr bool capture_frame_due(uint64_t requested, uint64_t shown, bool armed) {
    return requested && !armed && shown >= requested - 1;
}

// The pad route has no ordinal before its first guest poll (-1). Keep a due target pending in
// that state; consuming it would turn an unavailable route origin into an unexplained missed
// capture. Signed comparisons deliberately avoid the `requested - 1` unsigned-underflow shape.
constexpr bool capture_pad_flip_due(int64_t requested, int64_t observed, bool armed) {
    return requested > 0 && observed >= 0 && !armed && observed >= requested;
}

} // namespace prosper::frontend
