#include "diagnostics/perf/frame_flicker.hpp"

#include "diagnostics/perf/perf_ledger.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <mutex>

namespace prosper::diagnostics::perf {

double luma_grid_distance(const LumaGrid& a, const LumaGrid& b) {
    double sum = 0;
    for (size_t i = 0; i < kFlickerCells; ++i) sum += std::fabs(a.cell[i] - b.cell[i]);
    return sum / static_cast<double>(kFlickerCells);
}

LumaGrid luma_grid_from_rgba(const uint8_t* rgba, uint32_t width, uint32_t height,
                             size_t row_bytes, uint32_t points_per_axis) {
    LumaGrid g;
    if (!rgba || width < kFlickerGridW || height < kFlickerGridH || points_per_axis == 0 ||
        row_bytes < static_cast<size_t>(width) * 4)
        return g;
    // The sample coordinates depend only on the extent, so they are laid out once per call: the
    // centre of each of n equal strips of a cell, per axis. The pixel loop is then loads and
    // multiply-adds (this runs on the present thread for every presented frame).
    constexpr uint32_t kMaxPoints = 16;
    const uint32_t points = std::min(points_per_axis, kMaxPoints);
    std::array<uint32_t, kFlickerGridW * kMaxPoints> xs{};
    std::array<uint32_t, kFlickerGridW> nxs{};
    std::array<uint32_t, kFlickerGridH * kMaxPoints> ys{};
    std::array<uint32_t, kFlickerGridH> nys{};
    auto lay_out = [points](uint32_t cells, uint32_t extent, uint32_t* out, uint32_t* counts) {
        for (uint32_t c = 0; c < cells; ++c) {
            const uint64_t a = static_cast<uint64_t>(c) * extent / cells;
            const uint64_t b = static_cast<uint64_t>(c + 1) * extent / cells;
            const uint32_t n = static_cast<uint32_t>(std::min<uint64_t>(points, b - a));
            counts[c] = n;
            for (uint32_t i = 0; i < n; ++i)
                out[c * kMaxPoints + i] = static_cast<uint32_t>(a + (2ull * i + 1) * (b - a) / (2ull * n));
        }
    };
    lay_out(kFlickerGridW, width, xs.data(), nxs.data());
    lay_out(kFlickerGridH, height, ys.data(), nys.data());
    for (uint32_t cy = 0; cy < kFlickerGridH; ++cy) {
        for (uint32_t cx = 0; cx < kFlickerGridW; ++cx) {
            // Integer luma x 10000 per point (Rec.709 weights), summed exactly.
            uint64_t sum = 0;
            for (uint32_t j = 0; j < nys[cy]; ++j) {
                const uint8_t* row = rgba + static_cast<size_t>(ys[cy * kMaxPoints + j]) * row_bytes;
                uint32_t row_sum = 0;   // <= 16 points x 2,550,000: fits
                for (uint32_t i = 0; i < nxs[cx]; ++i) {
                    const uint8_t* p = row + static_cast<size_t>(xs[cx * kMaxPoints + i]) * 4;
                    row_sum += 2126u * p[0] + 7152u * p[1] + 722u * p[2];
                }
                sum += row_sum;
            }
            const uint64_t n = static_cast<uint64_t>(nxs[cx]) * nys[cy];
            g.cell[static_cast<size_t>(cy) * kFlickerGridW + cx] =
                n ? static_cast<float>(static_cast<double>(sum) / (10000.0 * n)) : 0.0f;
        }
    }
    return g;
}

namespace {

uint64_t pack_region(const FlickerEpisode& e) {
    return static_cast<uint64_t>(e.x0) | static_cast<uint64_t>(e.x1) << 8 |
           static_cast<uint64_t>(e.y0) << 16 | static_cast<uint64_t>(e.y1) << 24 |
           static_cast<uint64_t>(e.region_cells) << 32;
}

std::string format_region_fields(uint32_t cells, uint32_t x0, uint32_t x1, uint32_t y0,
                                 uint32_t y1) {
    if (cells == 0) return "diffuse(no-single-cell)";
    char buf[96];
    std::snprintf(buf, sizeof buf, "x%u-%u%%,y%u-%u%%(%u/%zu-cells)",
                  x0 * 100 / kFlickerGridW, (x1 + 1) * 100 / kFlickerGridW,
                  y0 * 100 / kFlickerGridH, (y1 + 1) * 100 / kFlickerGridH, cells, kFlickerCells);
    return buf;
}

}  // namespace

std::string format_flicker_region(const FlickerEpisode& e) {
    return format_region_fields(e.region_cells, e.x0, e.x1, e.y0, e.y1);
}

std::string format_flicker_region_packed(uint64_t packed) {
    return format_region_fields(static_cast<uint32_t>(packed >> 32),
                                static_cast<uint32_t>(packed & 0xff),
                                static_cast<uint32_t>((packed >> 8) & 0xff),
                                static_cast<uint32_t>((packed >> 16) & 0xff),
                                static_cast<uint32_t>((packed >> 24) & 0xff));
}

uint64_t pack_flicker_region(const FlickerEpisode& e) { return pack_region(e); }

void FlickerDetector::refresh_region() {
    const uint64_t frames = episode_.flicker_frames;
    uint32_t cells = 0, x0 = kFlickerGridW, x1 = 0, y0 = kFlickerGridH, y1 = 0;
    for (uint32_t cy = 0; cy < kFlickerGridH; ++cy)
        for (uint32_t cx = 0; cx < kFlickerGridW; ++cx) {
            const uint32_t hits = cell_hits_[static_cast<size_t>(cy) * kFlickerGridW + cx];
            if (!frames || 2ull * hits < frames || hits == 0) continue;
            ++cells;
            x0 = std::min(x0, cx); x1 = std::max(x1, cx);
            y0 = std::min(y0, cy); y1 = std::max(y1, cy);
        }
    episode_.region_cells = cells;
    if (cells) {
        episode_.x0 = x0; episode_.x1 = x1; episode_.y0 = y0; episode_.y1 = y1;
    } else {
        episode_.x0 = episode_.x1 = episode_.y0 = episode_.y1 = 0;
    }
}

FlickerFrameVerdict FlickerDetector::push(const LumaGrid& s, uint64_t now_ns) {
    FlickerFrameVerdict v;
    if (history_ >= 1) {
        v.d1 = luma_grid_distance(s, prev1_);
        if (v.d1 < t_.duplicate_d1) {
            // A repeat: neither a vote nor a history entry. The episode state stands as it was.
            v.duplicate = true;
            v.episode = in_episode_;
            v.sustained = in_episode_ && sustained_;
            return v;
        }
    }
    if (history_ >= 2) {
        v.compared = true;
        v.d2 = luma_grid_distance(s, prev2_);
        v.flicker = v.d1 >= t_.min_d1 && v.d2 <= t_.max_d2_ratio * v.d1;

        const uint32_t window = std::clamp<uint32_t>(t_.window_frames, 1u, 32u);
        const uint32_t mask = window == 32 ? 0xffffffffu : ((1u << window) - 1u);
        recent_bits_ = ((recent_bits_ << 1) | (v.flicker ? 1u : 0u)) & mask;
        recent_ns_[recent_head_] = now_ns;
        recent_head_ = (recent_head_ + 1) % window;
        recent_count_ = std::min(recent_count_ + 1, window);
        const bool holds = recent_count_ >= window &&
            static_cast<uint32_t>(std::popcount(recent_bits_)) >= t_.min_in_window;

        if (holds) {
            if (!in_episode_) {
                in_episode_ = true;
                sustained_ = false;
                episode_ = FlickerEpisode{};
                cell_hits_.fill(0);
                // The oldest frame of the window that formed the vote: the slot about to be
                // overwritten next, since the ring is full.
                episode_.start_ns = recent_ns_[recent_head_];
            }
            episode_.last_ns = now_ns;
            ++episode_.compared_frames;
            if (v.flicker) {
                ++episode_.flicker_frames;
                episode_.d1_sum += v.d1;
                episode_.d2_sum += v.d2;
                for (size_t i = 0; i < kFlickerCells; ++i) {
                    const double c1 = std::fabs(s.cell[i] - prev1_.cell[i]);
                    const double c2 = std::fabs(s.cell[i] - prev2_.cell[i]);
                    if (c1 >= t_.cell_min_d1 && c2 <= t_.max_d2_ratio * c1) ++cell_hits_[i];
                }
            }
            if (!sustained_ && now_ns - episode_.start_ns >= t_.sustain_ns) {
                sustained_ = true;
                v.onset = true;
            }
            if (sustained_) refresh_region();
            v.episode = true;
            v.sustained = sustained_;
        } else {
            in_episode_ = false;
            sustained_ = false;
        }
    }
    prev2_ = prev1_;
    prev1_ = s;
    history_ = std::min<uint32_t>(history_ + 1, 2);
    return v;
}

// ---- process-wide ------------------------------------------------------------------------------

namespace {

struct ProcessFlicker {
    std::mutex mutex;
    FlickerDetector detector;
    bool request_pending = false;
    bool request_issued = false;
    FILE* trace = nullptr;
    uint64_t origin_ns = 0;
    bool have_source = false;
    FlickerSource source = FlickerSource::GpuSample;
};

ProcessFlicker& process_flicker() {
    static ProcessFlicker* const instance = new ProcessFlicker();   // never destroyed
    return *instance;
}

}  // namespace

void set_flicker_trace_path(const char* path) {
    if (!path || !*path) return;
    ProcessFlicker& p = process_flicker();
    std::lock_guard<std::mutex> lock(p.mutex);
    if (p.trace) return;
    p.trace = std::fopen(path, "w");
    if (!p.trace) {
        std::fprintf(stderr, "[flicker] cannot open PROSPER_FLICKER_TRACE=%s; no trace written\n",
                     path);
        return;
    }
    std::fputs("t_ms,source_flip,src,dup,d1,d2,flicker,episode,sustained\n", p.trace);
}

void observe_presented_frame(const uint8_t* rgba, uint32_t width, uint32_t height,
                             size_t row_bytes, uint64_t source_flip, FlickerSource source) {
    if (!enabled() || !rgba) return;
    // Built outside the lock: it depends only on the caller's pixels.
    const LumaGrid grid = luma_grid_from_rgba(rgba, width, height, row_bytes);
    const uint64_t now = now_ns();
    ProcessFlicker& p = process_flicker();
    std::lock_guard<std::mutex> lock(p.mutex);
    if (!p.origin_ns) p.origin_ns = now;
    if (p.have_source && p.source != source) {
        // Two sampling grids are not comparable frame to frame (see FlickerSource).
        p.detector = FlickerDetector(p.detector.thresholds());
    }
    p.have_source = true;
    p.source = source;
    const FlickerFrameVerdict v = p.detector.push(grid, now);
    add(Counter::FlickerPresentedFrames);
    if (v.compared) add(Counter::FlickerComparedFrames);
    if (v.flicker) add(Counter::FlickerFrames);
    if (v.sustained && !v.duplicate) {
        add(Counter::FlickerSustainedFrames);
        const FlickerEpisode& e = p.detector.episode();
        set(Gauge::FlickerEpisodeD1Milli, static_cast<uint64_t>(e.mean_d1() * 1000.0 + 0.5));
        set(Gauge::FlickerEpisodeD2Milli, static_cast<uint64_t>(e.mean_d2() * 1000.0 + 0.5));
        set(Gauge::FlickerEpisodeShareMilli, static_cast<uint64_t>(e.share() * 1000.0 + 0.5));
        set(Gauge::FlickerEpisodeMs, static_cast<uint64_t>(e.seconds() * 1000.0 + 0.5));
        set(Gauge::FlickerEpisodeRegion, pack_flicker_region(e));
    }
    if (v.onset) {
        set(Gauge::FlickerOnsetSourceFlip, source_flip);
        if (!p.request_issued) p.request_pending = p.request_issued = true;
    }
    if (p.trace) {
        std::fprintf(p.trace, "%.3f,%llu,%s,%d,%.3f,%.3f,%d,%d,%d\n", (now - p.origin_ns) / 1e6,
                     static_cast<unsigned long long>(source_flip),
                     source == FlickerSource::GpuSample ? "gpu" : "cpu", v.duplicate ? 1 : 0, v.d1,
                     v.d2, v.flicker ? 1 : 0, v.episode ? 1 : 0, v.sustained ? 1 : 0);
        std::fflush(p.trace);
    }
}

bool take_flicker_capture_request() {
    ProcessFlicker& p = process_flicker();
    std::lock_guard<std::mutex> lock(p.mutex);
    const bool pending = p.request_pending;
    p.request_pending = false;
    return pending;
}

}  // namespace prosper::diagnostics::perf
