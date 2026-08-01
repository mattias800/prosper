// gpu_timeline.hpp - native-speed semantic submit/present timeline recording.
#pragma once

#include "command_processor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace prosper::gpu {

struct GpuTimelineMetadata {
    std::string revision;
    std::string title_id;
    std::string input_route;
};

// Compact per-submit depth/stencil programming. This is intentionally metadata-only: it lets a
// native-speed timeline prove surface/view/HTILE lifetime without snapshotting hundreds of MiB of
// shader resources for every submit (#611).
struct GpuTimelineDepthSurface {
    uint64_t depth_read_base = 0, depth_write_base = 0;
    uint64_t stencil_read_base = 0, stencil_write_base = 0;
    uint64_t htile_data_base = 0;
    uint32_t db_depth_view = 0, db_render_override = 0, db_render_override2 = 0;
    uint32_t db_depth_size_xy = 0, db_dfsm_control = 0, db_depth_info = 0;
    uint32_t db_z_info = 0, db_stencil_info = 0;
    uint32_t db_depth_size = 0, db_depth_slice = 0;
    uint32_t db_htile_surface = 0, db_rmi_l2_cache_control = 0;
    uint32_t target_width = 0, target_height = 0;
    uint32_t draw_count = 0, depth_test_count = 0, depth_write_count = 0, clear_count = 0;
    uint32_t compare_mask = 0;
    // Optional raw guest-backing hashes, enabled only for PROSPER_GPU_TIMELINE_DEPTH_HASH_DIM=WxH.
    // Bit 0/1/2 identifies valid depth/stencil/HTILE hashes respectively.
    uint32_t backing_hash_mask = 0;
    uint64_t depth_backing_hash = 0, stencil_backing_hash = 0, htile_backing_hash = 0;
    uint32_t backing_writer_kind = 0; // 0 none, otherwise GpuTimelineWriterKind
    uint64_t backing_writer_sequence = 0, backing_writer_addr = 0, backing_writer_size = 0;
    uint64_t backing_writer_order = 0, backing_writer_identity = 0;
};

// Consecutive semantic draws that use the same color-target extent. Addresses are deliberately
// excluded: they are run-local, while the extent/order signature is useful across routed boots.
struct GpuTimelineTargetSpan {
    uint32_t first_draw = 0;
    uint32_t draw_count = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct GpuTimelineDmaCopy {
    uint64_t dst = 0, src = 0;
    uint32_t bytes = 0, sels = 0;
    uint64_t command_order = 0;
    uint64_t packet_addr = 0;
};

struct GpuTimelineSubmit {
    uint64_t sequence = 0;
    uint64_t elapsed_ns = 0;
    uint64_t submit_no = 0;
    uint64_t process_command_order = 0;
    uint64_t first_command_order = 0;
    uint64_t last_command_order = 0;
    uint64_t color0_base = 0;
    uint32_t draw_count = 0;
    uint32_t dispatch_count = 0;
    uint32_t dma_copy_count = 0;
    uint32_t color0_width = 0;
    uint32_t color0_height = 0;
    std::vector<GpuTimelineDepthSurface> depth_surfaces;
    std::vector<GpuTimelineTargetSpan> target_spans;
    std::vector<GpuTimelineDmaCopy> dma_copies;
    bool target_spans_truncated = false;
    // Older timeline versions can report an operation count without the exact records required for
    // replay. New v8 writers retain this flag only when some other capture limitation is discovered.
    bool capture_incomplete = false;
};

// Shared by live detailed-capture selection and offline timeline inspection. A zero target extent
// disables target matching; all numeric ranges are inclusive.
struct GpuTimelineSelector {
    uint64_t min_submit_no = 1;
    uint32_t target_width = 0;
    uint32_t target_height = 0;
    uint32_t target_min_draw = 0;
    uint32_t target_max_draw = UINT32_MAX;
    uint32_t min_draws = 0;
    uint32_t max_draws = UINT32_MAX;
    uint32_t min_dispatches = 0;
    uint32_t max_dispatches = UINT32_MAX;
};

struct GpuTimelinePresent {
    uint64_t sequence = 0;
    uint64_t elapsed_ns = 0;
    uint64_t present_count = 0;
    uint64_t latest_submit_no = 0;
    int32_t buffer_index = -1;
    int64_t flip_arg = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct GpuTimelineDetail {
    uint64_t sequence = 0;
    uint64_t elapsed_ns = 0;
    uint64_t submit_no = 0;
    std::string capture_path;
    uint32_t semantic_draw_count = 0;
    uint32_t semantic_dispatch_count = 0;
    uint32_t realized_draw_count = 0;
    uint32_t realized_dispatch_count = 0;
    uint32_t operation_count = 0;
    uint32_t missing_operation_count = 0;
    uint32_t shader_version_count = 0;
    uint32_t resource_version_count = 0;
    uint64_t resource_bytes = 0;
};

enum class GpuTimelineWriterKind : uint32_t {
    Unknown = 0,
    Graphics = 1,
    Compute = 2,
    DmaData = 3,
    WriteData = 4,
};

// How an external image input was made reproducible, or why its producer remains unknown. A phase-
// bounded result is deliberately distinct from a generic miss: the capture only observed producers
// at/after an explicit semantic gate and must not imply that no earlier renderer writer existed.
enum class GpuTimelineProducerProvenance : uint32_t {
    Unknown = 0,
    ProducerHistory = 1,
    ExactRttSeed = 2,
    PhaseHistoryBounded = 3,
};

struct GpuTimelineProducer {
    uint64_t sequence = 0;
    uint64_t elapsed_ns = 0;
    uint64_t consumer_submit_no = 0;
    uint32_t consumer_operation = 0;
    uint32_t future_writer_operation = UINT32_MAX;
    uint64_t resource_addr = 0;
    uint64_t resource_size = 0;
    uint32_t resource_width = 0;
    uint32_t resource_height = 0;
    bool resolved = false;
    uint64_t producer_submit_no = 0;
    uint64_t producer_draw_index = 0;
    uint64_t producer_command_order = 0;
    uint64_t producer_target_addr = 0;
    uint32_t producer_width = 0;
    uint32_t producer_height = 0;
    GpuTimelineWriterKind first_writer_kind = GpuTimelineWriterKind::Unknown;
    uint64_t history_first_submit_no = 0;
    uint64_t history_first_draw_index = 0;
    uint64_t history_first_command_order = 0;
    uint64_t history_write_count = 0;
    uint64_t history_submit_count = 0;
    uint64_t history_window_first_submit_no = 0;
    uint64_t history_lower_bound_submit_no = 0;
    GpuTimelineProducerProvenance provenance = GpuTimelineProducerProvenance::Unknown;
    bool lifetime_truncated = false;
    bool history_window_truncated = false;
    bool first_color_has_clear = false;
    uint32_t first_color_clear_word0 = 0;
    uint32_t first_color_clear_word1 = 0;
    uint32_t first_color_control = 0;
    uint32_t first_color_control_mode = 0;
    uint32_t first_target_mask = 0;
    uint32_t first_color_format = 0;
};

// Cheap runtime counters for semantic detailed-capture diagnostics and focused tests. The counters
// distinguish compact phase observation from producer history and full submit materialization, so a
// long boot can prove that an AFTER_COMPUTE gate remained lightweight before it armed.
struct GpuTimelineCaptureCounters {
    uint64_t phase_observation_submits = 0;
    uint64_t phase_dispatches_scanned = 0;
    uint64_t prearm_history_submits_skipped = 0;
    uint64_t prearm_history_draws_skipped = 0;
    uint64_t prearm_bundle_submits_skipped = 0;
    uint64_t history_submits_recorded = 0;
    uint64_t bundle_submits_captured = 0;
    uint64_t detail_submits_captured = 0;
    uint64_t bundle_provenance_failures = 0;
    uint64_t history_lower_bound_submit_no = 0;
    bool history_phase_bounded = false;
};

// A read-before-write image leaf is temporal. Phase-bounded unknown provenance cannot close that
// dependency, so a requested bundle must remain an explicitly failed checkpoint instead of being
// advertised as complete. Non-temporal inputs remain ordinary guest-memory resources.
bool gpu_timeline_bundle_provenance_complete(GpuTimelineProducerProvenance provenance,
                                             uint32_t future_writer_operation);

// Bundle closure is cumulative across every retained submit. Once one constituent exposes a
// phase-bounded temporal image with no seed/producer, a later closed endpoint cannot make the
// earlier submit reproducible.
struct GpuTimelineBundleProvenanceState {
    bool complete = true;
    bool graph_unavailable = false;
    uint64_t first_incomplete_submit_no = 0;
    uint64_t bounded_unknown_leaf_count = 0;
};

void gpu_timeline_observe_bundle_provenance(
    GpuTimelineBundleProvenanceState& state, uint64_t submit_no,
    GpuTimelineProducerProvenance provenance, uint32_t future_writer_operation);

struct GpuTimelineFile {
    uint32_t version = 0;
    GpuTimelineMetadata metadata;
    std::vector<GpuTimelineSubmit> submits;
    std::vector<GpuTimelinePresent> presents;
    std::vector<GpuTimelineDetail> details;
    std::vector<GpuTimelineProducer> producers;
    bool truncated_tail = false;
};

class GpuTimelineWriter {
public:
    GpuTimelineWriter();
    ~GpuTimelineWriter();
    GpuTimelineWriter(const GpuTimelineWriter&) = delete;
    GpuTimelineWriter& operator=(const GpuTimelineWriter&) = delete;

    bool open(const std::string& path, const GpuTimelineMetadata& metadata, std::string& error);
    bool append_submit(const GpuTimelineSubmit& submit, std::string& error);
    bool append_present(const GpuTimelinePresent& present, std::string& error);
    bool append_detail(const GpuTimelineDetail& detail, std::string& error);
    bool append_producer(const GpuTimelineProducer& producer, std::string& error);
    bool flush(std::string& error);
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool read_gpu_timeline(const std::string& path, GpuTimelineFile& timeline, std::string& error);
bool gpu_timeline_submit_matches(const GpuTimelineSubmit& submit,
                                 const GpuTimelineSelector& selector);
GpuTimelineCaptureCounters gpu_timeline_capture_counters();
// True only for a fully valid detailed-capture request with a nonzero AFTER_COMPUTE phase gate.
// Frontends use this canonical parse to distinguish dormant phase capture from eager capture.
bool gpu_timeline_capture_is_after_compute_gated();
// True once the matching AFTER_COMPUTE dispatch has been observed. Published atomically because
// compute realization uses it to change capture portability policy at the semantic boundary.
bool gpu_timeline_capture_after_compute_gate_armed();

// Runtime hooks. Inert unless PROSPER_GPU_TIMELINE=<path> is set. They record folded semantic state
// before renderer sampling and never realize shaders, copy resources, or invoke Vulkan.
void begin_gpu_timeline_submit(uint64_t submit_no);
void record_gpu_timeline_submit(const GpuState& state, uint64_t submit_no);
void record_gpu_timeline_present(uint64_t present_count, int buffer_index, int64_t flip_arg,
                                 uint32_t width, uint32_t height);
void flush_gpu_timeline();
void close_gpu_timeline();

// Interactive frame-bundle capture (prosper-app F9): arm a one-shot capture of ONE complete displayed
// frame (every submit between two presents) into a replayable .prgbundle at `path`. Unlike a single
// .prgcap, the bundle re-runs the frame's producer submits on replay, so renderer-owned RTTs regenerate
// instead of replaying black (which they do for a deferred renderer). max_mb (0 = default) caps the
// deduplicated bundle size and the resource closure of each constituent submit. Armed from the app main
// thread; the frame is captured on the render thread between the next two presents. On-demand: near-zero
// cost (one atomic load per submit) until armed.
void request_interactive_capture_bundle(const std::string& path, uint32_t max_mb = 0,
                                        uint32_t delay_presents = 0);
bool interactive_capture_bundle_active();

// The outcome of the most recently COMPLETED interactive grab, so a frontend can tell the user what
// happened instead of leaving the answer in stderr among tens of thousands of lines (#1587). The user
// pressed a key; a keystroke that silently produces nothing is indistinguishable from one that never
// registered, and several agents on a 4K title concluded exactly that.
//
// Take-once: returns true and clears the pending outcome, false when nothing has completed since the
// last call. Poll it from the app loop.
struct InteractiveGrabOutcome {
    bool ok = false;              // the bundle was written
    std::string bundle_path;      // the .prgbundle that was, or would have been, written
    std::string error;            // empty when ok; otherwise the exact failure, budget numbers included
    uint64_t max_unique_bytes = 0;   // the budget in force, so a frontend can name the remedy
};
bool take_interactive_grab_outcome(InteractiveGrabOutcome& out);

// Optional guest-stdout phase gate for the same whole-frame bundle. When
// PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG is configured together with the existing
// PROSPER_CAPTURE_BUNDLE path, an exact completed guest-log line arms one capture after skipping one
// completed present. The byte-stream observer is fragment-safe and bounded; normal play pays only the
// enabled check when the environment variable is absent.
inline constexpr size_t kGuestLogCaptureMaxLineBytes = 4096;
enum class GuestLogCaptureSource : uint8_t {
    Unknown,
    Printf,
    Puts,
    Putchar,
    Fputs,
    Fwrite,
    Write,
};
bool guest_log_capture_bundle_enabled();
void observe_guest_log_for_capture(
    const char* bytes, size_t size,
    GuestLogCaptureSource source = GuestLogCaptureSource::Unknown);
// Marks bytes omitted by a bounded stdout adapter. It deliberately discards through the next observed
// line ending so an unobserved suffix can cause only a missed match, never a false exact-line match.
void observe_guest_log_capture_gap();

} // namespace prosper::gpu
