#pragma once

#include <chrono>
#include <cstdint>

namespace prosper::frontend {

// Per unique allocation owner, never per alias. Byte counts are the lengths supplied to exact
// comparisons/copies, not an assertion about how far an early-exiting memcmp actually read.
// States start unobserved so an interrupted dispatch cannot masquerade as a successful skip.
struct ComputeBufferTiming {
    bool enabled = false;
    bool owner_resolved = false;
    const char* cache = "unobserved";
    const char* validation = "unobserved";
    const char* baseline = "unobserved";
    const char* gpu_compare = "unobserved";
    const char* writeback = "unobserved";
    uint64_t compared_bytes = 0;
    uint64_t uploaded_bytes = 0;
    uint64_t result_compared_bytes = 0;
    uint64_t guest_copied_bytes = 0;
    double setup_ms = 0;
    double validation_ms = 0;
    double upload_compare_ms = 0;
    double upload_copy_ms = 0;
    double upload_map_ms = 0;
    double upload_watch_ms = 0;
    double writeback_ms = 0;
    double result_compare_ms = 0;
    double guest_copy_ms = 0;
    double guest_layout_ms = 0;
    double result_map_ms = 0;
    double result_watch_ms = 0;
    double baseline_ms = 0;
    double notify_ms = 0;
    double source_validation_ms = 0;
    double provenance_ms = 0;
};

// No clock reads when disabled. A scope includes failure exits and does not include log I/O.
class ComputeBufferCostScope {
public:
    ComputeBufferCostScope(bool enabled, double& milliseconds)
        : enabled_(enabled), milliseconds_(milliseconds),
          start_(enabled ? Clock::now() : Clock::time_point{}) {}
    ~ComputeBufferCostScope() {
        if (enabled_)
            milliseconds_ += std::chrono::duration<double, std::milli>(
                Clock::now() - start_).count();
    }
    ComputeBufferCostScope(const ComputeBufferCostScope&) = delete;
    ComputeBufferCostScope& operator=(const ComputeBufferCostScope&) = delete;
private:
    using Clock = std::chrono::steady_clock;
    bool enabled_;
    double& milliseconds_;
    Clock::time_point start_;
};

} // namespace prosper::frontend
