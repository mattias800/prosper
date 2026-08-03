#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace prosper::perf {

// The cheap, always-on side of F8. prosper-app samples this at 4 Hz; no renderer timing clocks,
// resource walks, screenshots, or frame dumps run until the user actually presses F8.
struct ProcessSample {
    uint64_t monotonic_ns = 0;
    std::optional<uint64_t> process_cpu_ns;
    std::optional<uint64_t> rss_bytes;
    uint64_t guest_presents = 0;
    // The CPU handoff path exposes present_frame_seq(). The shared-device GPU-present path skips
    // that handoff, so its production counter is unavailable rather than zero.
    std::optional<uint64_t> rendered_frames;
    uint64_t host_presented_frames = 0;
};

// One live-renderer callback. These are the existing PROSPER_RENDER_TIMING intervals, recorded in a
// compact structured form only during F8's post-trigger window.
struct RendererTimingRecord {
    uint64_t monotonic_ns = 0;
    uint64_t callbacks = 0;
    uint64_t draws = 0;
    uint64_t texture_bytes = 0;
    uint64_t buffer_bytes = 0;
    double total_ms = 0;
    double build_resources_ms = 0;
    double backend_ms = 0;
    double output_copy_ms = 0;
    double gpu_wait_ms = 0;
    uint64_t gpu_timestamp_samples = 0;
    double gpu_device_ms = 0;
    double readback_ms = 0;
    double setup_resources_ms = 0;
    double texture_ms = 0;
    double buffer_ms = 0;
};

// One live-compute batch. CPU-fast-path dispatches are included in `dispatches`; the cumulative
// fast-path count lets the report keep that population visible instead of silently omitting it.
struct ComputeTimingRecord {
    uint64_t monotonic_ns = 0;
    uint64_t dispatches = 0;
    uint64_t cpu_fast_total = 0;
    // Present only when every dispatch in the retained batch uses one identical program at one
    // run-local guest address. The SPIR-V hash is stable across address relocation and lets the
    // offline report group the expensive programs without enabling a whole-boot compute trace.
    std::optional<uint64_t> program_addr;
    std::optional<uint64_t> program_hash;
    double total_ms = 0;
};

struct CaptureConfig {
    uint64_t pre_window_ns = 5'000'000'000ull;
    uint64_t post_window_ns = 5'000'000'000ull;
    uint64_t sample_interval_ns = 250'000'000ull;
    size_t max_renderer_records = 4096;
    size_t max_compute_records = 4096;
};

struct CaptureArmResult {
    bool ok = false;
    unsigned index = 0;
    size_t pre_samples = 0;
    double post_seconds = 0;
    std::string error;
};

struct CaptureOutcome {
    bool ok = false;
    std::string path;
    std::string error;
    size_t pre_samples = 0;
    size_t post_samples = 0;
    size_t renderer_records = 0;
    size_t compute_records = 0;
    size_t renderer_dropped = 0;
    size_t compute_dropped = 0;
};

// Thread-safe capture state. The app thread owns process samples and finalization; renderer/compute
// threads only call the two bounded record methods while detailed_timing_active() is true.
class InteractivePerformanceCapture {
public:
    explicit InteractivePerformanceCapture(CaptureConfig config = {});
    ~InteractivePerformanceCapture();

    bool sample_due(uint64_t monotonic_ns) const;
    void observe_sample(const ProcessSample& sample);

    CaptureArmResult arm(const std::string& directory, const std::string& title_id,
                         const std::string& title_label, const std::string& revision,
                         uint64_t monotonic_ns,
                         std::chrono::system_clock::time_point wall_clock);

    bool detailed_timing_active() const {
        return detailed_active_.load(std::memory_order_relaxed);
    }
    void record_renderer(RendererTimingRecord record);
    void record_compute(ComputeTimingRecord record);
    bool take_outcome(CaptureOutcome& outcome);
    void cancel();

private:
    struct PendingCapture;

    std::unique_ptr<PendingCapture> finish_if_due_locked(uint64_t monotonic_ns);
    void publish_completed(std::unique_ptr<PendingCapture> completed);

    CaptureConfig config_;
    mutable std::mutex mutex_;
    std::deque<ProcessSample> ring_;
    std::atomic<uint64_t> next_sample_ns_{0};
    std::atomic<bool> detailed_active_{false};
    std::unique_ptr<PendingCapture> pending_;
    std::optional<CaptureOutcome> outcome_;
    unsigned arm_count_ = 0;
};

InteractivePerformanceCapture& interactive_performance_capture();

uint64_t monotonic_now_ns();
ProcessSample collect_process_sample(uint64_t monotonic_ns, uint64_t guest_presents,
                                     std::optional<uint64_t> rendered_frames,
                                     uint64_t host_presented_frames);
const char* build_revision();

} // namespace prosper::perf
