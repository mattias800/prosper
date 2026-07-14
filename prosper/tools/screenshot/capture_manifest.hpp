#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace prosper::screenshot {

enum class CaptureSource : uint8_t { Rendered, RawScanout };

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
};

struct CaptureClassification {
    bool source_advanced = true;
    bool pixel_identical = false;
    double stale_seconds = 0;
    double pixel_stale_seconds = 0;
};

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
};

class CaptureTracker {
public:
    CaptureClassification observe(const CaptureObservation& observation,
                                  const std::vector<uint8_t>& pixels);
    uint64_t distinct_source_frames() const { return distinct_source_frames_; }
    uint64_t pixel_distinct_frames() const { return pixel_distinct_frames_; }
    uint64_t rendered_samples() const { return rendered_samples_; }
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
std::string manifest_summary_json(int saved, int requested, bool timed_out,
                                  const CaptureTracker& tracker, int exit_code);

} // namespace prosper::screenshot
