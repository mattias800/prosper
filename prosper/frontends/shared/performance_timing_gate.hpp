#pragma once

namespace prosper::frontend {

// Header-only because render_runner.h is compiled directly into many Vulkan tests that do not link
// prosper-app's performance-capture library. The gate is thread-local because render_runner executes
// synchronously inside the live-render callback: an app-wide flag could make an unrelated concurrent
// replay/test caller pay for timestamps while F8 was active on a different thread.
inline bool& interactive_performance_timing_gate() {
    static thread_local bool enabled = false;
    return enabled;
}

inline bool interactive_performance_timing() {
    return interactive_performance_timing_gate();
}

// Scope the exceptional F8 backend clocks to exactly one live-render callback. Restoring the prior
// value supports nesting and proves that no return path can leave render_runner timing spuriously on.
class ScopedInteractivePerformanceTiming {
public:
    explicit ScopedInteractivePerformanceTiming(bool enabled)
        : previous_(interactive_performance_timing_gate()) {
        interactive_performance_timing_gate() = enabled;
    }
    ~ScopedInteractivePerformanceTiming() {
        interactive_performance_timing_gate() = previous_;
    }

    ScopedInteractivePerformanceTiming(const ScopedInteractivePerformanceTiming&) = delete;
    ScopedInteractivePerformanceTiming& operator=(const ScopedInteractivePerformanceTiming&) = delete;

private:
    bool previous_;
};

} // namespace prosper::frontend
