// present_frame_rate.hpp — how fast is this title actually running?
//
// WHY THIS IS NOT A PRESENT COUNTER, AND WHY THAT DISTINCTION IS THE WHOLE POINT
// ------------------------------------------------------------------------------
// The obvious framerate is "publications per second": count present_write_frame calls, divide by
// wall time. That number is WRONG in the one case anybody needs a framerate for, and wrong in the
// direction that hides the defect — it reads FULL SPEED for a completely frozen title.
//
// The reason is `RetainedFrameAction::ServeRetained` (frontends/shared/live/live_renderer.cpp):
// when a submit produces no publishable present source, the renderer re-serves the frame it
// retained, and that re-serve goes through the ordinary publish path. The guest keeps flipping,
// the renderer keeps publishing, `present_frame_seq()` keeps climbing — and the screen never
// changes. That is instrument trap 90 stated as a counter, and it is exactly the R-Type Delta
// (PPSA26414) regression #2783: for nine days the guest reached stage 1 while every presented frame
// was the same retained one, with `no pass produced a 1920x1080 present source … offered 0 bytes`
// on every flip. A present-rate fps would have read ~60 through the entire failure. The A/B that
// finally diagnosed it (#2799) counted DISTINCT late `pixel_crc32` values — 46 of 46 with the fix,
// 2 without — precisely because a rate alone could not see it.
//
// So this module counts two things and every consumer reports BOTH:
//
//   presented fps — publications accepted by the present layer, per second of wall clock.
//   distinct  fps — publications whose CONTENT differed from the immediately preceding
//                   publication, per second of wall clock.
//
// A healthy title has the two roughly equal. A title where they diverge is a title with a problem,
// and the divergence is itself the diagnostic: `distinct 0.0 / presented 59.8` is a frozen picture
// being re-served at vblank, not a game running at 60 fps.
//
// WHAT "DIFFERENT CONTENT" MEANS HERE, EXACTLY
// --------------------------------------------
// Two publications are the same frame when `frame_content_signature` agrees on them. The signature
// is deliberately SUB-LINEAR (see the constants below) because it runs on every publication of a
// frame that may be 33 MB, and a full hash of that at 60 Hz would cost more than the render.
//
// The sampling error is therefore real, and it is one-sided in the safe direction:
//   * it can NEVER report a re-served frame as new  — identical bytes give an identical signature,
//     so the failure mode this module exists to catch cannot be laundered by the sampling;
//   * it CAN miss a change too small and too scattered to hit a sampled window, which UNDER-reports
//     the distinct rate.
// Concretely: any change to a contiguous run of at least
// kFrameSignatureBlockBytes + kFrameSignatureBytesPerBlock bytes is guaranteed to be seen. The
// distinct rate is thus a lower bound on the true rate of new frames, and it is documented as one
// wherever it is printed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace prosper::gpu {

// One 16-byte window is hashed out of every 4 KiB. At 3840x2160 RGBA that reads 128 KiB of a 33 MB
// frame (0.39%) and costs tens of microseconds; at 1280x720 it reads 14 KiB of 3.7 MB. Any change
// to a contiguous run of >= 4096 + 16 bytes must contain a whole window and is therefore detected.
constexpr size_t kFrameSignatureBlockBytes = 4096;
constexpr size_t kFrameSignatureBytesPerBlock = 16;

// A bounded-cost content signature of one frame. Dimensions and byte count are folded in, so a
// resolution change is a content change even in the (impossible in practice) event that the sampled
// bytes agree. `pixels` may be null only when `bytes` is 0.
uint64_t frame_content_signature(const uint8_t* pixels, size_t bytes, uint32_t width, uint32_t height);

// The pure accumulator: publications in, counters out. Holds no clock and no lock of its own, so it
// is unit-testable without a present layer, a renderer or a Vulkan device — which is what lets the
// frozen-title arm be a fast, deterministic ctest case rather than a live run somebody has to
// remember to do.
class FrameRateCounter {
public:
    // `at_seconds` must come from a monotonic clock; the counter only ever subtracts two of them.
    void observe(uint64_t signature, double at_seconds);
    void reset();

    uint64_t published() const { return published_; }
    uint64_t distinct() const { return distinct_; }
    double first_publication_seconds() const { return first_; }
    double last_publication_seconds() const { return last_; }

private:
    uint64_t published_ = 0;
    uint64_t distinct_ = 0;
    uint64_t last_signature_ = 0;
    double first_ = 0;
    double last_ = 0;
};

// A reading of the process-wide counters, taken at `now_seconds` on the module's own monotonic
// epoch. Two snapshots subtract cleanly, so any window can be measured after the fact.
struct PresentRateSnapshot {
    uint64_t published = 0;
    uint64_t distinct = 0;
    double now_seconds = 0;                  // when this snapshot was taken
    double first_publication_seconds = 0;    // 0 when nothing has been published
    double last_publication_seconds = 0;
};

// Both rates over one window, plus the raw counts they were derived from. The counts travel with
// the rates on purpose: a rate with no population behind it is the thing this project's instrument
// traps are mostly made of, and "0.4 fps" from three frames is not the same claim as "0.4 fps" from
// four hundred.
struct FrameRate {
    bool measured = false;        // false => the window was empty or degenerate; the rates are 0
    double window_seconds = 0;
    uint64_t published = 0;
    uint64_t distinct = 0;
    double presented_fps = 0;
    double distinct_fps = 0;
    double distinct_fraction = 0; // distinct / published, 0 when nothing was published
};

// Window = [first publication, the moment the snapshot was taken]. Wall clock, not "time between
// the first and last publication": a title that STOPS publishing must decay toward zero rather than
// freeze at whatever it last managed.
FrameRate frame_rate_since_first_publication(const PresentRateSnapshot& snapshot);

// Window = [earlier, later]. Returns an unmeasured result if the counters moved backwards, which
// can only mean a reset happened between the two readings.
FrameRate frame_rate_between(const PresentRateSnapshot& earlier, const PresentRateSnapshot& later);

// "distinct 3.4 fps / presented 59.8 fps over 60.0 s (204 of 3590 published frames carried new
// content, 5.7%)". Distinct comes FIRST in every rendering of this in the project, because a reader
// skimming a log line takes the first number, and the first number must be the honest one.
std::string format_frame_rate(const FrameRate& rate);

// A compact form for burning into an image or drawing in a HUD:
// "3.4 fps  (59.8 presented)  3840x2160".
std::string format_frame_rate_short(const FrameRate& rate, uint32_t width, uint32_t height);

// True when publications kept arriving but almost none of them carried new content — i.e. the
// reader is looking at the R-Type Delta shape and MUST NOT quote the presented rate.
//
// This is a labelling heuristic layered on two exact numbers, never a substitute for them: the
// thresholds are named parameters precisely so nobody has to guess what "almost none" meant. It
// deliberately keys on the FRACTION rather than on an absolute distinct rate, because titles in
// this project legitimately run at ~1 fps and an absolute threshold would call those frozen.
bool frame_rate_is_mostly_retained(const FrameRate& rate,
                                   uint64_t min_published = 30,
                                   double max_distinct_fraction = 0.5);

// ---- process-wide counters, fed by the present layer ---------------------------------------
//
// Called by present_write_frame for every accepted publication, before it takes the present mutex:
// the signature is computed from the caller's own bytes and needs none of the present layer's
// state, so it must not lengthen that critical section.
void note_present_publication(const uint8_t* pixels, size_t bytes, uint32_t width, uint32_t height);

// Cleared by present_reset(), so a test that resets the present layer resets the rate with it.
void reset_present_rate();

PresentRateSnapshot present_rate_snapshot();

} // namespace prosper::gpu
