// Burst capture (#5152): N CONSECUTIVE presented frames as screenshots, for a picture that
// alternates.
//
// Every other picture instrument takes one frame, and one frame of a good / broken / good flicker
// lands on one parity: the build is called good or broken by chance, and a bisect over that oracle
// converges on the wrong commit. A burst shows both parities side by side.
//
// Two triggers share this: Shift+F9 (a person who sees the flicker), and PROSPER_FLICKER_CAPTURE_DIR
// (the frame-flicker [perf-alarm] detector, once per run, for a headless or unattended run).
// PROSPER_SNAP_AT_FLIPS ranges (snap_author.hpp) are the third way to get consecutive frames, aimed
// at a pad flip ordinal instead.
//
// Frames are held in memory and written only when the burst is complete. Writing a 4K BMP (~24 MiB)
// between two presents would stall the presenter long enough to skip guest flips -- and a burst that
// skips every other flip shows a period-2 flicker as a steady picture. The file names carry the
// guest flip each frame came from, and the completion line says whether they were consecutive.
//
// Pure: no SDL, Vulkan or filesystem here, so it is unit-tested (tests/frontend/test_burst_capture.cpp);
// main.cpp owns the hotkey, the pixels and the writing.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace prosper::frontend {

inline constexpr uint32_t kBurstFramesDefault = 8;
// 32 frames of 4K RGBA is ~1 GiB held until the burst completes.
inline constexpr uint32_t kBurstFramesMax = 32;

// PROSPER_BURST_FRAMES. Unset or empty -> the default. A positive decimal up to kBurstFramesMax ->
// that. Anything else (a typo, 0, more than the cap) -> nullopt: the caller says so and disables
// burst capture, so a typo costs a capture rather than taking a different number of frames.
inline std::optional<uint32_t> parse_burst_frames(const char* value) {
    if (!value || !*value) return kBurstFramesDefault;
    uint64_t n = 0;
    for (const char* p = value; *p; ++p) {
        if (*p < '0' || *p > '9') return std::nullopt;
        n = n * 10 + static_cast<uint64_t>(*p - '0');
        if (n > kBurstFramesMax) return std::nullopt;
    }
    if (n < 2) return std::nullopt;   // a "burst" of one frame cannot show an alternation
    return static_cast<uint32_t>(n);
}

// "<stem>_03_flip1203.bmp": index first so a listing sorts in capture order, then the guest flip.
inline std::string burst_file_name(const std::string& stem, uint32_t index, uint64_t source_flip) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "_%02u_flip%llu.bmp", index,
                  static_cast<unsigned long long>(source_flip));
    return stem + buf;
}

// Whether the frames came from consecutive guest flips. An empty answer for fewer than two frames.
// The gap list is what a reader needs when the burst shows no alternation: if the presenter
// skipped every other flip, a period-2 flicker is invisible in it.
inline std::string burst_continuity(const std::vector<uint64_t>& flips) {
    if (flips.size() < 2) return "";
    std::string gaps;
    uint32_t gap_count = 0;
    for (size_t i = 1; i < flips.size(); ++i) {
        if (flips[i] == flips[i - 1] + 1) continue;
        ++gap_count;
        if (gap_count <= 4) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%s%llu->%llu", gaps.empty() ? "" : ",",
                          static_cast<unsigned long long>(flips[i - 1]),
                          static_cast<unsigned long long>(flips[i]));
            gaps += buf;
        }
    }
    char buf[192];
    if (!gap_count) {
        std::snprintf(buf, sizeof buf, "consecutive guest flips %llu-%llu",
                      static_cast<unsigned long long>(flips.front()),
                      static_cast<unsigned long long>(flips.back()));
        return buf;
    }
    std::snprintf(buf, sizeof buf,
                  "NOT consecutive: %u gap(s) %s%s -- the presenter skipped guest flips, so a "
                  "period-2 alternation can be hidden in this burst",
                  gap_count, gaps.c_str(), gap_count > 4 ? ",..." : "");
    return buf;
}

struct BurstFrame {
    std::vector<uint8_t> rgba;   // tightly packed width*height*4
    uint32_t width = 0, height = 0;
    uint64_t source_flip = 0;   // the guest flip the presented frame came from
};

struct BurstResult {
    std::string dir;
    std::string stem;
    std::string reason;
    std::vector<BurstFrame> frames;
};

class BurstCapture {
public:
    // Starts a burst of `frames` presented frames. Refused (false) while one is running, or for
    // fewer than two frames.
    bool arm(std::string dir, std::string stem, uint32_t frames, std::string reason) {
        if (active() || frames < 2) return false;
        result_ = BurstResult{std::move(dir), std::move(stem), std::move(reason), {}};
        result_.frames.reserve(frames);
        target_ = frames;
        return true;
    }

    // True while frames are still wanted: the present path must read back the next frame.
    bool active() const { return target_ != 0 && result_.frames.size() < target_; }
    // True once every frame is held and take() has not yet been called.
    bool complete() const { return target_ != 0 && result_.frames.size() >= target_; }
    uint32_t target() const { return target_; }
    size_t held() const { return result_.frames.size(); }

    // One presented frame (tightly packed RGBA). Ignored unless active. A frame from the same or an
    // older guest flip than the last one held is ignored too: a re-presented frame is not a new one.
    bool add(const uint8_t* rgba, uint32_t width, uint32_t height, uint64_t source_flip) {
        if (!active() || !rgba || !width || !height) return false;
        if (!result_.frames.empty() && source_flip <= result_.frames.back().source_flip)
            return false;
        BurstFrame f;
        f.width = width;
        f.height = height;
        f.source_flip = source_flip;
        f.rgba.assign(rgba, rgba + static_cast<size_t>(width) * height * 4);
        result_.frames.push_back(std::move(f));
        return true;
    }

    // Hands over a complete burst and returns to idle.
    BurstResult take() {
        BurstResult out = std::move(result_);
        result_ = BurstResult{};
        target_ = 0;
        return out;
    }

    // Abandons a burst (shutdown). Returns how many frames were held.
    size_t cancel() {
        const size_t n = result_.frames.size();
        result_ = BurstResult{};
        target_ = 0;
        return n;
    }

private:
    BurstResult result_;
    uint32_t target_ = 0;
};

}   // namespace prosper::frontend
