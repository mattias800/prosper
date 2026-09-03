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
    double prelude_ms = 0;
    double pass_ms = 0;
    double build_resources_ms = 0;
    double backend_ms = 0;
    double output_copy_ms = 0;
    double pass_head_ms = 0;
    double pass_loop_ms = 0;
    double pass_pre_ms = 0;
    double pass_post_ms = 0;
    double pass_tail_ms = 0;
    double post_stats_ms = 0;
    double post_slot0_ms = 0;
    double post_mrt_ms = 0;
    double post_rest_ms = 0;
    double resolve_stall_ms = 0;
    double resolve_read_ms = 0;
    double resolve_copy_stall_ms = 0;
    double resolve_copy_ms = 0;
    uint64_t resolve_count = 0;
    uint64_t resolve_read_count = 0;
    uint64_t resolve_bytes = 0;
    double gpu_wait_ms = 0;
    uint64_t gpu_timestamp_samples = 0;
    double gpu_device_ms = 0;
    double readback_ms = 0;
    double backend_target_ms = 0;
    double backend_draw_setup_ms = 0;
    double backend_record_upload_ms = 0;
    double backend_cleanup_ms = 0;
    double backend_setup_shader_ms = 0;
    double backend_setup_fixed_ms = 0;
    double setup_resources_ms = 0;
    double backend_setup_pipeline_ms = 0;
    uint64_t backend_pipeline_refs = 0;
    uint64_t backend_pipeline_hits = 0;
    uint64_t backend_pipeline_misses = 0;
    uint64_t backend_pipeline_bypasses = 0;
    uint64_t backend_pipeline_entries = 0;
    uint64_t backend_pipeline_evictions = 0;
    // FRONTEND resource time, accumulated in build_resources. These are NOT sub-buckets of
    // setup_resources_ms below: that one is the BACKEND's, and the two are different layers.
    // Subtracting these from it is a category error that produces a large, plausible, meaningless
    // residue -- I published one (#2215) before noticing, so the names now say which layer they are.
    double frontend_texture_ms = 0;
    double frontend_buffer_ms = 0;
    double frontend_tex_rtt_ms = 0;
    double frontend_tex_compute_ms = 0;
    double frontend_tex_local_ms = 0;
    double frontend_tex_persist_hit_ms = 0;
    double frontend_tex_persist_reuse_ms = 0;
    double frontend_tex_persist_miss_ms = 0;
    // The persistent entry was found and its guest bytes had CHANGED, so the surface was decoded
    // again. Split out of the unnamed residual because "the cache is missing" and "the cache is
    // working and the content really is new" call for opposite work, and a single `other` bucket
    // cannot tell them apart -- which is exactly how Stray's title screen hid 930 of 1110 ms.
    double frontend_tex_persist_invalid_ms = 0;
    uint64_t frontend_tex_persist_invalid_n = 0;
    // References that reached NONE of the named classes. A count, not a duration: paired with the
    // signed millisecond residual the report derives, the two disagree only if the classification
    // itself is wrong.
    uint64_t frontend_tex_other_n = 0;
    // Slowest texture reference that reached none of the named outcome classes in this semantic
    // submit. One fixed-size witness keeps F8 non-perturbing while still giving an offline report
    // the exact resource identity needed to explain the residual.
    double frontend_tex_other_slowest_ms = 0;
    uint64_t frontend_tex_other_addr = 0;
    uint64_t frontend_tex_other_source_bytes = 0;
    uint32_t frontend_tex_other_width = 0;
    uint32_t frontend_tex_other_height = 0;
    uint32_t frontend_tex_other_depth = 0;
    uint32_t frontend_tex_other_format = 0;
    uint32_t frontend_tex_other_components = 0;
    uint32_t frontend_tex_other_tile_mode = 0;
    uint32_t frontend_tex_other_img_dim = 0;
    uint32_t frontend_tex_other_class = 0;
    bool frontend_tex_other_compute_candidate = false;
    bool frontend_tex_other_persistent_candidate = false;
    bool frontend_tex_other_compressed = false;
    bool frontend_tex_other_depth_compare = false;
    bool frontend_tex_other_host_backed = false;
    double frontend_build_draw_ms = 0;
    double frontend_validate_ms = 0;
    double frontend_poison_ms = 0;
    double frontend_indices_ms = 0;
    double frontend_reflect_ms = 0;
    // BACKEND sub-buckets, which DO decompose setup_resources_ms:
    //     setup_resources_ms = res_texture + res_buffer + res_descriptor + other
    // `other` is deliberately not stored -- it is the remainder, and storing a number the reader can
    // derive invites the two to disagree. Without these an F8 capture cannot attribute its own
    // largest bucket, which is the reason both lanes spent a day reasoning from partial data.
    double res_texture_ms = 0;
    double res_buffer_ms = 0;
    // Leaves of res_buffer_ms. `copy` is the one the perf work kept landing on (#2215/#2231), and on
    // its own it is badly misleading: measured on Blue Prince gameplay it was 25.82 ms against a
    // res_buffer_ms of 332.08 — 5.2% of the frame, while 60.9% of the frame sat in the branch
    // unattributed. The other three name the branches that were previously timed by nothing, so a
    // reader of an F8 capture cannot repeat that. `other` stays underived here for the same reason
    // as above, but the report prints it SIGNED: a negative remainder is over-attribution, and
    // clamping it would make a broken partition look like a complete one (#2245).
    double res_buffer_copy_ms = 0;
    double res_buffer_create_ms = 0;
    double res_buffer_index_find_ms = 0;
    double res_buffer_index_insert_ms = 0;
    double res_buffer_hash_ms = 0;
    double res_descriptor_ms = 0;
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
    uint64_t gpu_timestamp_samples = 0;
    double gpu_device_ms = 0;
    double gpu_shader_ms = 0;
    double gpu_pre_ms = 0;
    double gpu_storage_copy_ms = 0;
    double gpu_compare_ms = 0;
    double gpu_restore_ms = 0;
    double setup_ms = 0;
    double pipeline_ms = 0;
    double dispatch_wait_ms = 0;
    double writeback_ms = 0;
    double cleanup_ms = 0;
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
