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

// `shown` counts successful presents completed before the candidate frame. If swapchain acquisition
// skipped the exact requested ordinal, keep the one-shot eligible and capture the next real present.
constexpr bool capture_frame_due(uint64_t requested, uint64_t shown, bool armed) {
    return requested && !armed && shown >= requested - 1;
}

} // namespace prosper::frontend
