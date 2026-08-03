#pragma once

#include <charconv>
#include <cstdint>
#include <limits>
#include <string_view>
#include <system_error>

namespace prosper::frontend {

// Parse the explicit unattended-performance-capture delay. Zero remains the disabled/invalid
// sentinel: automatic capture is opt-in, and a malformed value must never capture a later phase
// than the operator intended.
inline uint64_t parse_performance_capture_delay_ns(const char* value) {
    if (!value || !*value) return 0;
    const std::string_view text(value);
    uint64_t milliseconds = 0;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), milliseconds, 10);
    constexpr uint64_t kNanosecondsPerMillisecond = 1'000'000;
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        !milliseconds ||
        milliseconds > std::numeric_limits<uint64_t>::max() / kNanosecondsPerMillisecond)
        return 0;
    return milliseconds * kNanosecondsPerMillisecond;
}

// A due trigger is consumed before its caller attempts to arm the capture. Thus an arm failure is
// fail-visible and one-shot instead of retrying every app-loop iteration and silently capturing a
// later phase. Comparing by subtraction only after `now >= start` makes clock reversal safe.
class ElapsedPerformanceCaptureTrigger {
public:
    explicit ElapsedPerformanceCaptureTrigger(uint64_t delay_ns) : delay_ns_(delay_ns) {}

    bool take_if_due(uint64_t start_ns, uint64_t now_ns) {
        if (!delay_ns_ || attempted_ || now_ns < start_ns || now_ns - start_ns < delay_ns_)
            return false;
        attempted_ = true;
        return true;
    }

    uint64_t delay_ns() const { return delay_ns_; }

private:
    uint64_t delay_ns_ = 0;
    bool attempted_ = false;
};

} // namespace prosper::frontend
