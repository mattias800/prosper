#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace prosper::screenshot {

constexpr size_t kPerceptualLumaCells = 16 * 9;

// Provenance of a captured sample, mirroring prosper::gpu::PresentSource.
//   Rendered     — prosper composited this frame from the guest's draws/dispatches.
//   GuestScanout — prosper composited NOTHING and republished the guest's own flipped display
//                  buffer. It arrives through the same renderer publish path as a composited frame,
//                  so it needs its own label: `--require-composited-frame` asserts that prosper
//                  rendered something, and folding this into Rendered makes the exact frame that
//                  assertion exists to reject satisfy it (#2026 review B1; the revert is #2044).
//   RawScanout   — no renderer frame ever existed and the reader fell back to the guest display
//                  buffer directly (the pre-renderer boot path).
enum class CaptureSource : uint8_t { Rendered, RawScanout, GuestScanout };

struct CaptureObservation {
    CaptureSource source = CaptureSource::Rendered;
    uint64_t source_seq = 0;
    uint64_t frame_seq = 0;
    uint64_t present_count = 0;
    int front_index = -1;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pixel_crc32 = 0;
    double elapsed_seconds = 0;
    uint32_t distinct_rgb_colors = 0;
    uint64_t nonblack_rgb_pixels = 0;
    uint64_t average_hash = 0;
    uint64_t difference_hash = 0;
    std::array<uint8_t, kPerceptualLumaCells> luma16x9{};
};

struct PerceptualHashes {
    uint64_t average = 0;
    uint64_t difference = 0;
};

struct PixelContentMetrics {
    uint32_t distinct_colors = 0;
    uint64_t nonblack_pixels = 0;
};

// The desktop frontend presents renderer output through an OPAQUE Vulkan swapchain. Preserve that
// exact visible contract in persisted screenshots and their metrics: render-target alpha is not
// blended into the desktop image. Raw scanout remains byte-for-byte guest evidence and is untouched.
void normalize_capture_rgba(CaptureSource source, std::vector<uint8_t>& pixels);

PixelContentMetrics measure_pixel_content_rgba(const std::vector<uint8_t>& pixels);

// Standard 8x8 average hash plus 9x8 horizontal difference hash. These intentionally discard
// fine pixel detail so small raster changes remain close while missing layers alter the signature.
PerceptualHashes perceptual_hashes_rgba(const std::vector<uint8_t>& pixels,
                                        uint32_t width, uint32_t height);

// A small spatial luminance signature suitable for SSIM comparisons. Unlike aHash/dHash,
// it preserves the magnitude and location of large missing layers.
std::array<uint8_t, kPerceptualLumaCells>
perceptual_luma16x9_rgba(const std::vector<uint8_t>& pixels,
                         uint32_t width, uint32_t height);

struct CaptureClassification {
    bool source_advanced = true;
    bool pixel_identical = false;
    double stale_seconds = 0;
    double pixel_stale_seconds = 0;
};

// Terminal state of the PRIMARY guest thread, as the sampling thread is able to observe it.
//
// Scope, and it is narrow on purpose: `run_entry` produces a `BootResult` for the one guest thread
// it entered. A worker thread that faults recovers through its own per-thread recovery point and
// never reaches here — on Linux a fatal worker fault already `_exit(90)`s the process, which every
// caller that checks an exit status can see. The primary thread was the silent one (#2007).
enum class GuestRunState : uint8_t { Running, Returned, Faulted };

struct GuestOutcome {
    GuestRunState state = GuestRunState::Running;
    int kind = 0;              // BootResult::kind — 0 returned, 2 fatal fault (SEGV/BUS/FPE), 3 SIGILL
    uint64_t fault_rip = 0;
    uint64_t fault_addr = 0;
    std::string detail;        // BootResult::detail, truncated to the channel's buffer
};

const char* guest_run_state_name(GuestRunState state);   // running | returned | faulted

// The join between the DETACHED guest thread and the sampling thread — this is the entire mechanism
// #2007 was missing, so treat it as load-bearing rather than as a convenience.
//
// `publish_guest_outcome` is called exactly once, by the guest thread, when `run_entry` returns.
// `read_guest_outcome` may be called any number of times by the sampling thread; before the first
// publish it reports `Running`. The publish is release-ordered and the read acquire-ordered, so a
// reader that observes a terminal state also observes that state's fault fields and detail text.
void publish_guest_outcome(int kind, uint64_t fault_rip, uint64_t fault_addr, const char* detail);
GuestOutcome read_guest_outcome();

// The one verdict a run reports: the `status=` token in the summary line, the `exit_code` in the
// manifest, and the process exit status all come from here so they cannot disagree.
struct RunVerdict {
    const char* status = "ok";
    int exit_code = 0;
};

// Fold the capture assertions and the guest's own terminal state into that single verdict.
//
//   guest faulted, not allowed   -> GUEST-FAULT          exit 1   (root cause outranks its symptoms)
//   assertions failed            -> FAILED               exit 1
//   guest faulted, allowed       -> GUEST-FAULT-ALLOWED  exit 0   (never "ok" — a fault stays visible)
//   otherwise                    -> ok                   exit 0
//
// `Returned` is deliberately NOT a failure. The guest's entry legitimately returns when a title
// exits, and a title that exits just as the last sample lands would otherwise make an intact run
// flaky; a truncated run is already caught by the saved/requested assertion. It is still reported.
RunVerdict decide_run_verdict(bool assertions_failed, const GuestOutcome& guest,
                              bool allow_guest_fault);

// Why the sampling loop ended. `RequestSatisfied` and `Timeout` are the two reasons the loop always
// had; `GuestFault` is the early stop (#2584).
enum class SamplingStop : uint8_t { RequestSatisfied, Timeout, GuestFault };

const char* sampling_stop_name(SamplingStop stop);   // request-satisfied | timeout | guest-fault

// Should the sampler stop before its request is satisfied, because nothing new can arrive?
//
// This deliberately does NOT stop at the fault itself, and that is the whole design. `GuestOutcome`
// describes only the thread `run_entry` entered (see the scope note above): other guest threads and
// a renderer backlog can still publish frames after the primary thread dies, and a stop keyed on the
// fault alone would discard those silently — the truncated run would look exactly like a satisfied
// one. So the stop also requires the present layer to have gone quiet, which makes the condition
// detect its own invalidity: a run that keeps producing frames keeps sampling them.
//
// Both windows are measured against the same settle duration. The one measured from the fault is
// what gives work already in flight a chance to land; the one measured from the last publication is
// what establishes that nothing is still arriving.
//
// `Returned` is deliberately NOT a stop. A title exiting on its own is not a dead guest, the tool
// already reports it, and a run cut short by it must keep tripping the saved/requested assertion —
// the contract `README.md` states and #2007 relied on.
bool should_stop_after_guest_fault(const GuestOutcome& guest, bool enabled,
                                   double seconds_since_fault_observed,
                                   double seconds_since_present_advanced,
                                   double settle_seconds);

struct CaptureRunConfig {
    std::string title;
    std::string timestamp;
    std::string output_dir;
    std::string input_route;
    std::string render_every;
    std::string render_every_for_ms;
    std::string render_scale;
    std::string render_target_dim;
    std::string render_resource_dim;
    bool time_mode = false;
    double seconds = 0;
    int every = 0;
    int requested = 0;
    int64_t warmup_ms = 0;
    int warmup_submits = 0;
    int min_distinct_frames = 0;
    double max_stale_seconds = -1;
    int min_pixel_distinct_frames = 0;
    double max_pixel_stale_seconds = -1;
    bool require_composited_frame = false;
    uint64_t min_present_count = 0;
    uint64_t min_frame_seq = 0;
    bool required_crc32_set = false;
    uint32_t required_crc32 = 0;
    bool allow_guest_fault = false;
    bool stop_after_guest_fault = true;
    double guest_fault_settle_seconds = 0;
};

class CaptureTracker {
public:
    CaptureClassification observe(const CaptureObservation& observation,
                                  const std::vector<uint8_t>& pixels);
    uint64_t distinct_source_frames() const { return distinct_source_frames_; }
    uint64_t pixel_distinct_frames() const { return pixel_distinct_frames_; }
    uint64_t rendered_samples() const { return rendered_samples_; }
    uint64_t guest_scanout_samples() const { return guest_scanout_samples_; }
    double max_stale_seconds() const { return max_stale_seconds_; }
    double max_pixel_stale_seconds() const { return max_pixel_stale_seconds_; }
    uint64_t max_frame_seq() const { return max_frame_seq_; }
    uint64_t max_present_count() const { return max_present_count_; }

private:
    bool have_previous_ = false;
    CaptureObservation previous_{};
    std::vector<uint8_t> previous_pixels_;
    double last_advance_seconds_ = 0;
    double last_pixel_change_seconds_ = 0;
    uint64_t distinct_source_frames_ = 0;
    uint64_t pixel_distinct_frames_ = 0;
    uint64_t rendered_samples_ = 0;
    uint64_t guest_scanout_samples_ = 0;
    double max_stale_seconds_ = 0;
    double max_pixel_stale_seconds_ = 0;
    uint64_t max_frame_seq_ = 0;
    uint64_t max_present_count_ = 0;
};

std::string json_escape(const std::string& value);
const char* capture_source_name(CaptureSource source);
std::string manifest_run_json(const CaptureRunConfig& config);
std::string manifest_sample_json(int index, const std::string& png_path,
                                 const CaptureObservation& observation,
                                 const CaptureClassification& classification,
                                 const std::string& input_route);
// The verdict, the guest's terminal state and the opt-out are all required parameters: a caller that
// forgets to wire the guest state must fail to compile rather than emit a summary that quietly omits
// it, which is the shape of the defect this signature exists to prevent (#2007). `stop` is required
// for the same reason: a short artifact set that does not say why it is short is the same legibility
// defect one level down (#2584). It also supplies the `timed_out` field, so the two cannot disagree.
std::string manifest_summary_json(int saved, int requested, SamplingStop stop,
                                  const CaptureTracker& tracker, const RunVerdict& verdict,
                                  const GuestOutcome& guest, bool allow_guest_fault);

} // namespace prosper::screenshot
