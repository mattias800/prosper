#pragma once
// frame-flicker (#5152): detect a presented picture that ALTERNATES -- frame t looks like frame t-2
// and not like frame t-1 -- for long enough that it is a defect rather than an effect.
//
// Why it exists. Every picture instrument this project had took ONE frame (F9, scheduled
// screenshots, most PROSPER_SNAP_AT_FLIPS uses). On a build whose 3D layer flickers good / noise /
// good, that single frame lands on either parity, so a bisect or A/B labels the build "good" or
// "broken" by chance and converges on the wrong answer. Nothing named flicker, so nobody saw it
// unless a person was watching the window (Kena: Bridge of Spirits' title screen, #5146).
//
// The signature is a 32x18 grid of MEAN LUMA (0-255) of the presented image. prosper-app's GPU
// present path builds it from the --fps content sample it already knew how to take (a NEAREST
// 256x144 reduction of the swapchain's source, copied to host memory under the fence the next
// present already waits on), so it adds no GPU wait; the CPU present paths build it from the
// published pixels. 4x4 sample points per cell: a cell's mean responds to a changed layer, and the
// whole signature plus the detector costs ~18 us per presented frame on the present thread
// (measured on the Radeon 8060S host under a parallel build, from the 256x144 sample; ~23 us from
// a 4K CPU frame). 8x8 points measured ~53 us for no difference a flicker test needs.
//
// The detector. For each presented frame t, with s the grid and dist the mean absolute per-cell
// difference:
//   d1 = dist(s_t, s_t-1), d2 = dist(s_t, s_t-2)
// t is a FLICKER FRAME when d1 >= min_d1 and d2 <= max_d2_ratio * d1: it looks like the frame two
// back, not the one just before. An EPISODE is >= min_flicker_in_window of the last window_frames
// compared frames being flicker frames; it is SUSTAINED once it has lasted sustain_ns. This rejects
// by construction:
//   * a camera cut -- one spike, and d2 is as large as d1 on the cut frame;
//   * a pan or any motion -- the frame two back is further away than the one just before (d2 >= d1);
//   * fades and slow changes -- d1 below min_d1 (and d2 ~ 2*d1 anyway);
//   * 30 fps content presented twice at 60 -- a frame whose d1 is below duplicate_d1 is a REPEAT
//     and is not entered into the history at all, so the distinct frames are compared with each
//     other and an ordinary pan stays a pan;
//   * grain / dither noise -- per-cell means of independent noise are as far from t-2 as from t-1.
// Thresholds are named constants below with the measurement that set them.
//
// The pure half (LumaGrid, FlickerDetector) has no clock, I/O or globals and is unit-tested over
// synthetic signature sequences (tests/diagnostics/test_frame_flicker.cpp). The process-wide half
// (observe_presented_frame) feeds the always-on [perf-alarm] ledger, which the frame-flicker rule
// reads at window close.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace prosper::diagnostics::perf {

constexpr uint32_t kFlickerGridW = 32;
constexpr uint32_t kFlickerGridH = 18;
constexpr size_t kFlickerCells = static_cast<size_t>(kFlickerGridW) * kFlickerGridH;   // 576

// ---- thresholds ------------------------------------------------------------------------------
// Units: luma levels 0-255, mean absolute difference over the 576 cells.

// min_d1: a frame must differ from its predecessor by at least this much to be a flicker candidate.
// A noise layer against the picture it replaces differs by tens of levels per covered cell; TAA
// jitter, temporal dither and film grain stay under one level on cell means. 3 levels means a
// layer covering a tenth of the screen must change its cells by ~30 levels to count.
constexpr double kFlickerMinD1 = 3.0;
// max_d2_ratio: the frame two back must be at least twice as close as the frame just before. A pan
// or any steady motion has d2 >= d1 (further back is further away), a fade d2 ~ 2*d1.
constexpr double kFlickerMaxD2Ratio = 0.5;
// duplicate_d1: below this the presented frame repeats its predecessor (a 30 fps title at 60, a
// paused scene) and is not entered into the history.
constexpr double kFlickerDuplicateD1 = 0.25;
// The 6-of-8 vote and 1 s sustain from #5152. Eight frames rather than two: a camera shake or a
// muzzle flash alternates for a frame or two; a broken ping-pong resource alternates for as long as
// the scene runs.
constexpr uint32_t kFlickerWindowFrames = 8;
constexpr uint32_t kFlickerMinInWindow = 6;
constexpr uint64_t kFlickerSustainNs = 1'000'000'000ull;
// A CELL flickers in a flicker frame when its own c1/c2 pass the same test with this floor; the
// region a report names is the cells that flickered in at least half of the episode's flicker
// frames. 8 levels on one cell is a visible change of that 1/576 of the screen.
constexpr double kFlickerCellMinD1 = 8.0;

struct FlickerThresholds {
    double min_d1 = kFlickerMinD1;
    double max_d2_ratio = kFlickerMaxD2Ratio;
    double duplicate_d1 = kFlickerDuplicateD1;
    uint32_t window_frames = kFlickerWindowFrames;
    uint32_t min_in_window = kFlickerMinInWindow;
    uint64_t sustain_ns = kFlickerSustainNs;
    double cell_min_d1 = kFlickerCellMinD1;
};

// One signature: 32x18 cells, row-major, mean luma 0-255.
struct LumaGrid {
    std::array<float, kFlickerCells> cell{};
};

// Mean |a - b| over the cells.
double luma_grid_distance(const LumaGrid& a, const LumaGrid& b);

// The signature of an RGBA8 image (`row_bytes` per row, R first). Each cell's mean is taken over at
// most `points_per_axis` x `points_per_axis` evenly spaced pixels of the cell's rectangle (all of
// them when the cell is smaller), so a 4K frame costs the same as the 256x144 GPU sample. Luma is
// Rec.709 (0.2126 R + 0.7152 G + 0.0722 B); BGRA input only swaps R/B weights, which a change
// detector does not care about. An image smaller than the grid in either axis yields all-zero.
LumaGrid luma_grid_from_rgba(const uint8_t* rgba, uint32_t width, uint32_t height,
                             size_t row_bytes, uint32_t points_per_axis = 4);

struct FlickerFrameVerdict {
    bool duplicate = false;   // a repeat of its predecessor: not entered into the history
    bool compared = false;    // had both t-1 and t-2 to compare against
    double d1 = 0, d2 = 0;
    bool flicker = false;     // this frame passed the d1/d2 test
    bool episode = false;     // >= min_in_window of the last window_frames compared frames flicker
    bool sustained = false;   // ...and the episode has lasted sustain_ns
    bool onset = false;       // the first sustained frame of this episode
};

// The current (or, once it ended, the last) sustained episode.
struct FlickerEpisode {
    uint64_t start_ns = 0;
    uint64_t last_ns = 0;
    uint64_t compared_frames = 0;   // compared frames since the episode started
    uint64_t flicker_frames = 0;
    double d1_sum = 0, d2_sum = 0;  // over flicker frames
    // Region: cells that flickered in at least half of the flicker frames. Bounding box in cells,
    // inclusive; region_cells == 0 means no single cell passed (a diffuse change).
    uint32_t region_cells = 0;
    uint32_t x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    double seconds() const { return last_ns > start_ns ? (last_ns - start_ns) / 1e9 : 0.0; }
    double mean_d1() const { return flicker_frames ? d1_sum / flicker_frames : 0.0; }
    double mean_d2() const { return flicker_frames ? d2_sum / flicker_frames : 0.0; }
    double share() const {
        return compared_frames ? static_cast<double>(flicker_frames) / compared_frames : 0.0;
    }
};

// "x12-87%,y22-100%(212/576-cells)", or "diffuse(no-single-cell)" when no cell passed on its own.
std::string format_flicker_region(const FlickerEpisode& e);
// The region packed into one 64-bit gauge (x0, x1, y0, y1 bytes, cell count above), and its text.
uint64_t pack_flicker_region(const FlickerEpisode& e);
std::string format_flicker_region_packed(uint64_t packed);

class FlickerDetector {
public:
    explicit FlickerDetector(FlickerThresholds t = {}) : t_(t) {}

    // One presented frame at `now_ns`. Pure: the clock is the caller's.
    FlickerFrameVerdict push(const LumaGrid& s, uint64_t now_ns);

    // The episode in progress, or the last one (valid once any frame was sustained).
    const FlickerEpisode& episode() const { return episode_; }
    bool in_episode() const { return in_episode_; }
    const FlickerThresholds& thresholds() const { return t_; }

private:
    void refresh_region();

    FlickerThresholds t_;
    LumaGrid prev1_{}, prev2_{};
    uint32_t history_ = 0;          // 0, 1 or 2 frames held
    // The last window_frames compared frames, newest in bit 0, and their timestamps.
    uint32_t recent_bits_ = 0;
    uint32_t recent_count_ = 0;
    std::array<uint64_t, 32> recent_ns_{};
    uint32_t recent_head_ = 0;
    bool in_episode_ = false;
    bool sustained_ = false;
    FlickerEpisode episode_{};
    std::array<uint32_t, kFlickerCells> cell_hits_{};
};

// ---- process-wide (prosper-app's present paths) ------------------------------------------------

// Where a presented frame's pixels came from. The two sample different points of the picture, so
// the same image signs slightly differently through each: a run whose presents alternated between
// them would read as an alternating picture. The detector therefore starts over whenever the source
// changes (an episode that straddles a change is lost; a false one is never made).
enum class FlickerSource : uint8_t {
    GpuSample,   // prosper-app's GPU present: the NEAREST 256x144 --fps sample grid
    CpuFrame,    // a CPU-published frame at its own resolution
};

// One presented frame's pixels: builds the signature and feeds the process detector, which counts
// into the [perf-alarm] ledger (Counter::Flicker*, Gauge::Flicker*). `source_flip` is the guest flip
// the frame came from, recorded at onset. A no-op under PROSPER_NO_PERF_ALARMS=1.
void observe_presented_frame(const uint8_t* rgba, uint32_t width, uint32_t height,
                             size_t row_bytes, uint64_t source_flip, FlickerSource source);

// True exactly once per run: the first time a sustained episode began. The frontend's
// PROSPER_FLICKER_CAPTURE_DIR burst consumes it.
bool take_flicker_capture_request();

// Optional per-frame trace (PROSPER_FLICKER_TRACE=<path>, read by the frontend): one CSV line per
// presented frame -- t_ms,source_flip,src(gpu|cpu),dup,d1,d2,flicker,episode,sustained. For
// calibrating the thresholds on a real run; off unless the frontend sets it.
void set_flicker_trace_path(const char* path);

}  // namespace prosper::diagnostics::perf
