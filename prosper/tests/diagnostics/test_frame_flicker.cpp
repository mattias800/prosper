// frame-flicker (#5152): the detector over synthetic signature sequences, the signature itself, and
// the [perf-alarm] rule over hand-built windows.
//
// Every negative arm is a picture that changes a lot frame to frame for a legitimate reason -- a
// pan, a fade, a cut, 30 fps content shown twice -- because those are what an alternation detector
// would misread. Each positive arm is the alternation built by hand, so a quiet result there is a
// detector that cannot see the defect at all.
#include "diagnostics/perf/frame_flicker.hpp"
#include "diagnostics/perf/perf_alarm_rules.hpp"
#include "diagnostics/perf/perf_ledger.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

using namespace prosper::diagnostics::perf;

namespace {

constexpr uint64_t kFrame60 = 16'666'667;   // ns per frame at 60 Hz
constexpr double kPi = 3.14159265358979323846;

LumaGrid grid_from(const std::function<double(uint32_t x, uint32_t y)>& f) {
    LumaGrid g;
    for (uint32_t y = 0; y < kFlickerGridH; ++y)
        for (uint32_t x = 0; x < kFlickerGridW; ++x)
            g.cell[y * kFlickerGridW + x] = static_cast<float>(std::clamp(f(x, y), 0.0, 255.0));
    return g;
}

// A smooth, textured picture: what the "good" parity looks like.
LumaGrid picture() {
    return grid_from([](uint32_t x, uint32_t y) {
        return 128.0 + 60.0 * std::sin(2 * kPi * x / 13.0) + 40.0 * std::cos(2 * kPi * y / 7.0);
    });
}

// Deterministic per-cell garbage: what the broken parity looks like.
LumaGrid noise(uint32_t seed) {
    uint32_t s = seed * 2654435761u + 12345u;
    LumaGrid g;
    for (float& c : g.cell) {
        s = s * 1664525u + 1013904223u;
        c = static_cast<float>((s >> 24) & 0xff);
    }
    return g;
}

// A picture scrolled by `offset` cells horizontally (a camera pan).
LumaGrid panned(double offset) {
    return grid_from([offset](uint32_t x, uint32_t y) {
        const double u = x + offset;
        return 128.0 + 60.0 * std::sin(2 * kPi * u / 13.0) + 40.0 * std::cos(2 * kPi * y / 7.0) +
               20.0 * std::sin(2 * kPi * u / 29.0);
    });
}

struct RunResult {
    uint32_t compared = 0, flicker = 0, sustained = 0, onsets = 0;
    uint64_t first_sustained_ns = 0;
};

// Pushes `frames` frames at 60 Hz. `frame(t)` builds frame t.
RunResult run(FlickerDetector& d, uint32_t frames, const std::function<LumaGrid(uint32_t)>& frame) {
    RunResult r;
    for (uint32_t t = 0; t < frames; ++t) {
        const uint64_t now = 1'000'000'000ull + t * kFrame60;
        const FlickerFrameVerdict v = d.push(frame(t), now);
        r.compared += v.compared;
        r.flicker += v.flicker;
        if (v.sustained && !v.duplicate) {
            ++r.sustained;
            if (!r.first_sustained_ns) r.first_sustained_ns = now - 1'000'000'000ull;
        }
        r.onsets += v.onset;
    }
    return r;
}

}   // namespace

// ---- the signature ---------------------------------------------------------------------------

TEST(FrameFlickerSignature, UniformImageGivesItsLumaInEveryCell) {
    const uint32_t w = 320, h = 180;
    std::vector<uint8_t> px(static_cast<size_t>(w) * h * 4);
    for (size_t i = 0; i < px.size(); i += 4) {
        px[i] = 200;
        px[i + 1] = 100;
        px[i + 2] = 50;
    }
    const LumaGrid g = luma_grid_from_rgba(px.data(), w, h, static_cast<size_t>(w) * 4);
    const double expected = 0.2126 * 200 + 0.7152 * 100 + 0.0722 * 50;
    for (float c : g.cell) EXPECT_NEAR(c, expected, 0.01);
}

TEST(FrameFlickerSignature, CellsFollowTheImageAndRespectRowPitch) {
    // Left half white, right half black, with 16 bytes of row padding that must never be read as
    // pixels (it is filled white to show up if it is).
    const uint32_t w = 256, h = 144;
    const size_t pitch = static_cast<size_t>(w) * 4 + 16;
    std::vector<uint8_t> px(pitch * h, 255);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = w / 2; x < w; ++x)
            std::memset(&px[y * pitch + static_cast<size_t>(x) * 4], 0, 4);
    const LumaGrid g = luma_grid_from_rgba(px.data(), w, h, pitch);
    for (uint32_t y = 0; y < kFlickerGridH; ++y) {
        EXPECT_NEAR(g.cell[y * kFlickerGridW + 0], 255.0, 0.01);
        EXPECT_NEAR(g.cell[y * kFlickerGridW + 15], 255.0, 0.01);
        EXPECT_NEAR(g.cell[y * kFlickerGridW + 16], 0.0, 0.01);
        EXPECT_NEAR(g.cell[y * kFlickerGridW + 31], 0.0, 0.01);
    }
}

TEST(FrameFlickerSignature, TooSmallOrNullImageIsAllZero) {
    std::vector<uint8_t> px(size_t{16} * 16 * 4, 255);
    const LumaGrid small = luma_grid_from_rgba(px.data(), 16, 16, 64);
    const LumaGrid null = luma_grid_from_rgba(nullptr, 256, 144, 1024);
    for (size_t i = 0; i < kFlickerCells; ++i) {
        EXPECT_EQ(small.cell[i], 0.0f);
        EXPECT_EQ(null.cell[i], 0.0f);
    }
    EXPECT_EQ(luma_grid_distance(small, null), 0.0);
}

// ---- the detector: alternation fires ----------------------------------------------------------

TEST(FrameFlickerDetector, StrictAlternationFiresAfterOneSecond) {
    FlickerDetector d;
    const LumaGrid a = picture(), b = noise(1);
    const RunResult r = run(d, 180, [&](uint32_t t) { return t % 2 ? b : a; });
    EXPECT_EQ(r.onsets, 1u);
    EXPECT_GT(r.sustained, 100u);
    // The episode is dated from the OLDEST frame of the window that formed the vote -- frame 2, the
    // first one with two predecessors -- so it becomes sustained one second after that frame, not
    // one second after the vote passed (which would be 7 frames later).
    const FlickerEpisode& e = d.episode();
    EXPECT_EQ(e.start_ns, 1'000'000'000ull + 2 * kFrame60);
    EXPECT_GE(r.first_sustained_ns, kFlickerSustainNs);
    // Onset additionally needs two fresh flicker frames, with one frame of clock quantization.
    EXPECT_LE(r.first_sustained_ns, kFlickerSustainNs + 4 * kFrame60);
    EXPECT_GT(e.mean_d1(), 30.0);
    EXPECT_LT(e.mean_d2(), 0.01);
    EXPECT_NEAR(e.share(), 1.0, 1e-9);
    EXPECT_GT(e.region_cells, kFlickerCells / 2);
}

TEST(FrameFlickerDetector, AlternationPresentedTwicePerFrameStillFires) {
    // A 30 fps title at 60 Hz: A A B B A A ... Each repeat is not a new frame; without dropping
    // repeats, the second A compares 0 against its twin and the first B compares against two As,
    // so the alternation would be invisible.
    FlickerDetector d;
    const LumaGrid a = picture(), b = noise(2);
    const RunResult r = run(d, 240, [&](uint32_t t) { return (t / 2) % 2 ? b : a; });
    EXPECT_EQ(r.onsets, 1u);
    EXPECT_GT(r.sustained, 0u);
}

TEST(FrameFlickerDetector, AlternationPresentedThreeTimesPerFrameStillFires) {
    const LumaGrid a = picture(), b = noise(2);
    for (const uint32_t presentations : {3u, 6u}) {
        FlickerDetector d;
        const RunResult r =
            run(d, 360, [&](uint32_t t) { return (t / presentations) % 2 ? b : a; });
        EXPECT_EQ(r.onsets, 1u) << "presentations per content frame " << presentations;
        EXPECT_GT(r.sustained, 0u);
    }
}

TEST(FrameFlickerDetector, SlowDistinctAlternationStillFires) {
    // Preserve sparse alternation, including intervals up to 0.9 s between distinct frames.
    const LumaGrid a = picture(), b = noise(2);
    for (const uint64_t interval : {250'000'000ull, 900'000'000ull}) {
        FlickerDetector d;
        uint32_t onsets = 0;
        for (uint32_t t = 0; t < 24; ++t)
            onsets += d.push(t % 2 ? b : a, 1'000'000'000ull + t * interval).onset;
        EXPECT_EQ(onsets, 1u) << "interval " << interval;
    }
}

TEST(FrameFlickerDetector, RegionNamesTheAlternatingQuarter) {
    // Only the top-left quarter flickers (a broken 3D layer under a steady UI, say); the rest of
    // the picture is still. The region must name that quarter, not the screen.
    FlickerDetector d;
    const LumaGrid a = picture(), n = noise(3);
    LumaGrid b = a;
    for (uint32_t y = 0; y < kFlickerGridH / 2; ++y)
        for (uint32_t x = 0; x < kFlickerGridW / 2; ++x)
            b.cell[y * kFlickerGridW + x] = n.cell[y * kFlickerGridW + x];
    const RunResult r = run(d, 180, [&](uint32_t t) { return t % 2 ? b : a; });
    ASSERT_EQ(r.onsets, 1u);
    const FlickerEpisode& e = d.episode();
    EXPECT_EQ(e.x0, 0u);
    EXPECT_LE(e.x1, kFlickerGridW / 2 - 1);
    EXPECT_EQ(e.y0, 0u);
    EXPECT_LE(e.y1, kFlickerGridH / 2 - 1);
    EXPECT_LE(e.region_cells, kFlickerCells / 4);
    EXPECT_GT(e.region_cells, kFlickerCells / 8);
    EXPECT_EQ(format_flicker_region(e).rfind("x0-", 0), 0u) << format_flicker_region(e);
}

// ---- the detector: what must NOT fire ---------------------------------------------------------

TEST(FrameFlickerDetector, SteadyPictureFiresNothing) {
    FlickerDetector d;
    const LumaGrid a = picture();
    const RunResult r = run(d, 300, [&](uint32_t) { return a; });
    EXPECT_EQ(r.compared, 0u);   // every frame after the first is a repeat
    EXPECT_EQ(r.flicker, 0u);
    EXPECT_EQ(r.sustained, 0u);
}

TEST(FrameFlickerDetector, PansDoNotFire) {
    // Slow, medium and fast pans: the frame two back is always further away than the one before.
    for (const double speed : {0.25, 1.0, 2.5}) {
        FlickerDetector d;
        const RunResult r = run(d, 300, [&](uint32_t t) { return panned(speed * t); });
        EXPECT_GT(r.compared, 250u) << "speed " << speed;
        EXPECT_EQ(r.flicker, 0u) << "speed " << speed;
        EXPECT_EQ(r.sustained, 0u) << "speed " << speed;
    }
}

TEST(FrameFlickerDetector, PanAtThirtyShownTwiceDoesNotFire) {
    FlickerDetector d;
    const RunResult r = run(d, 300, [&](uint32_t t) {
        const uint32_t content_frame = t / 2;
        return panned(1.5 * static_cast<double>(content_frame));
    });
    EXPECT_EQ(r.sustained, 0u);
    EXPECT_EQ(r.flicker, 0u);
}

TEST(FrameFlickerDetector, FadeDoesNotFire) {
    // A half-second fade to black, and a slow two-second one.
    for (const uint32_t frames : {30u, 120u}) {
        FlickerDetector d;
        const LumaGrid a = picture();
        const RunResult r = run(d, frames + 60, [&](uint32_t t) {
            const double k = t >= frames ? 0.0 : 1.0 - static_cast<double>(t) / frames;
            LumaGrid g = a;
            for (float& c : g.cell) c = static_cast<float>(c * k);
            return g;
        });
        EXPECT_EQ(r.flicker, 0u) << "fade over " << frames;
        EXPECT_EQ(r.sustained, 0u) << "fade over " << frames;
    }
}

TEST(FrameFlickerDetector, SingleCutDoesNotFire) {
    FlickerDetector d;
    const LumaGrid a = picture(), b = noise(4);
    const RunResult r = run(d, 240, [&](uint32_t t) { return t < 120 ? a : b; });
    EXPECT_EQ(r.flicker, 0u);
    EXPECT_EQ(r.sustained, 0u);
}

TEST(FrameFlickerDetector, AlternationShorterThanASecondDoesNotFire) {
    // 0.8 s of strict alternation (a muzzle flash, a hit strobe), then a steady picture. It is an
    // episode -- the vote passes -- but never a sustained one.
    FlickerDetector d;
    const LumaGrid a = picture(), b = noise(5);
    bool episode_seen = false;
    uint32_t sustained = 0;
    for (uint32_t t = 0; t < 240; ++t) {
        const LumaGrid& g = (t < 48 && (t % 2)) ? b : a;
        const FlickerFrameVerdict v = d.push(g, 1'000'000'000ull + t * kFrame60);
        episode_seen |= v.episode;
        sustained += v.sustained;
    }
    EXPECT_TRUE(episode_seen) << "the vote should pass: this arm is about the SUSTAIN";
    EXPECT_EQ(sustained, 0u);
}

TEST(FrameFlickerDetector, StrobeThenStillThenChangeDoesNotFire) {
    // A confirm flash (10 frames of alternation), a still load screen, then a cut -- or a slow
    // fade. The votes from the flash must not survive the still stretch: the second of stillness
    // would otherwise pay for the 1 s sustain, and the first new frame would fire.
    const LumaGrid a = picture(), b = noise(7), c = noise(8);
    for (const bool fade : {false, true}) {
        FlickerDetector d;
        const RunResult r = run(d, 300, [&](uint32_t t) {
            if (t < 10) return t % 2 ? b : a;
            if (t < 108) return a;
            if (!fade) return c;
            LumaGrid g = a;
            const double k = std::max(0.0, 1.0 - (t - 108) / 120.0);
            for (float& v : g.cell) v = static_cast<float>(v * k);
            return g;
        });
        EXPECT_EQ(r.onsets, 0u) << (fade ? "then a fade" : "then a cut");
        EXPECT_EQ(r.sustained, 0u) << (fade ? "then a fade" : "then a cut");
    }
}

TEST(FrameFlickerDetector, StrobeThenPresenterStallThenCutDoesNotFire) {
    // As above, but the pause presents NOTHING (a load that stops flipping), so no repeat ever
    // arrives to clear the votes. The 2 s gap must still not pay for the sustain: the episode has
    // to contain alternation of its own.
    FlickerDetector d;
    const LumaGrid a = picture(), b = noise(9);
    uint32_t onsets = 0, sustained = 0;
    uint64_t now = 1'000'000'000ull;
    for (uint32_t t = 0; t < 10; ++t, now += kFrame60) {
        const FlickerFrameVerdict v = d.push(t % 2 ? b : a, now);
        onsets += v.onset;
        sustained += v.sustained;
    }
    now += 2'000'000'000ull;
    for (uint32_t t = 0; t < 120; ++t, now += kFrame60) {
        const FlickerFrameVerdict v = d.push(panned(0.5 * t), now);
        onsets += v.onset;
        sustained += v.sustained;
    }
    EXPECT_EQ(onsets, 0u);
    EXPECT_EQ(sustained, 0u);
}

TEST(FrameFlickerDetector, LongerStrobeThenPresenterStallCannotFinishSustain) {
    // 0.8 s passes both the vote and the episode-comparison floor. No repeat is presented during
    // the pause, so neither repeat ageing nor that floor prevents the first new frame firing.
    FlickerDetector d;
    const LumaGrid a = picture(), b = noise(9);
    for (uint32_t t = 0; t < 48; ++t)
        EXPECT_FALSE(d.push(t % 2 ? b : a, 1'000'000'000ull + t * kFrame60).sustained);
    ASSERT_TRUE(d.in_episode());
    const auto resumed = d.push(a, 4'000'000'000ull);
    EXPECT_FALSE(resumed.onset);
    EXPECT_FALSE(resumed.sustained);
    EXPECT_FALSE(d.in_episode());
}

TEST(FrameFlickerDetector, ShortPresenterPauseThenSingleChangeDoesNotFire) {
    // A pause can be shorter than the allowed sparse-presentation interval. Resuming on one
    // old parity still cannot turn the preceding 0.8 s flash into a sustained defect.
    const LumaGrid a = picture(), b = noise(9);
    for (const uint64_t pause : {400'000'000ull, 900'000'000ull}) {
        FlickerDetector d;
        uint64_t now = 1'000'000'000ull;
        for (uint32_t t = 0; t < 48; ++t, now += kFrame60)
            EXPECT_FALSE(d.push(t % 2 ? b : a, now).onset);
        now += pause;
        for (uint32_t t = 0; t < 60; ++t, now += kFrame60)
            EXPECT_FALSE(d.push(a, now).onset) << "pause " << pause << " resumed frame " << t;
    }
}

TEST(FrameFlickerDetector, SlowHardBlinkDoesNotFire) {
    // A tenth of the screen blinks at 1 Hz (+60 luma, half a second on, half off). Every distinct
    // frame is a perfect d2 = 0 vote, but the repeats between them are a still picture, not an
    // alternation at the presented cadence.
    const LumaGrid a = picture();
    LumaGrid lit = a;
    for (size_t i = 0; i < kFlickerCells / 10; ++i)
        lit.cell[i] = std::min(255.0f, lit.cell[i] + 60.0f);
    for (const uint32_t half_period : {30u, 15u}) {
        FlickerDetector d;
        const RunResult r =
            run(d, 600, [&](uint32_t t) { return (t / half_period) % 2 ? lit : a; });
        EXPECT_EQ(r.onsets, 0u) << "blink half-period in presented frames " << half_period;
        EXPECT_EQ(r.sustained, 0u);
    }
}

TEST(FrameFlickerDetector, ShakeOfAFewFramesDoesNotFire) {
    // A camera shake: the picture jumps back and forth for 5 frames inside an ongoing pan.
    FlickerDetector d;
    const RunResult r = run(d, 300, [&](uint32_t t) {
        const double shake = (t >= 100 && t < 105) ? ((t % 2) ? 4.0 : -4.0) : 0.0;
        return panned(0.5 * t + shake);
    });
    EXPECT_EQ(r.sustained, 0u);
}

TEST(FrameFlickerDetector, OneGlitchFrameInEightDoesNotFire) {
    // A garbage frame every 8th frame inside a slow pan, for 5 s: the frame AFTER each glitch looks
    // like the one two back (the pan barely moved), so each glitch makes one flicker frame. That is
    // a defect, but not an alternation: one vote in eight must never sustain an episode.
    FlickerDetector d;
    const LumaGrid n = noise(6);
    const RunResult r = run(d, 300, [&](uint32_t t) { return t % 8 == 7 ? n : panned(0.25 * t); });
    EXPECT_GT(r.flicker, 20u) << "the glitches should produce single flicker frames";
    EXPECT_EQ(r.sustained, 0u);
}

// ---- the process feed -------------------------------------------------------------------------

TEST(FrameFlickerProcess, ChangingSignatureSourceRestartsTheDetector) {
    // The GPU-present sample grid and a CPU frame sample different points of the same picture, so a
    // run whose presents alternate between the two paths must not read as an alternating picture.
    // Positive half first: the same two images through ONE source are compared and vote.
    const uint32_t w = 64, h = 36;
    std::vector<uint8_t> dark(static_cast<size_t>(w) * h * 4, 20), bright(dark.size(), 220);
    auto count = [](Counter c) {
        return ledger().counters[static_cast<size_t>(c)].load(std::memory_order_relaxed);
    };
    ASSERT_TRUE(enabled()) << "PROSPER_NO_PERF_ALARMS must be unset for this test";
    const uint64_t compared0 = count(Counter::FlickerComparedFrames);
    const uint64_t flicker0 = count(Counter::FlickerFrames);
    for (int i = 0; i < 12; ++i)
        observe_presented_frame(i % 2 ? bright.data() : dark.data(), w, h,
                                static_cast<size_t>(w) * 4, 100 + i, FlickerSource::CpuFrame);
    EXPECT_EQ(count(Counter::FlickerComparedFrames) - compared0, 10u);
    EXPECT_EQ(count(Counter::FlickerFrames) - flicker0, 10u);
    // The same alternation, but each frame through the other source: never compared at all.
    const uint64_t compared1 = count(Counter::FlickerComparedFrames);
    for (int i = 0; i < 12; ++i)
        observe_presented_frame(i % 2 ? bright.data() : dark.data(), w, h,
                                static_cast<size_t>(w) * 4, 200 + i,
                                i % 2 ? FlickerSource::CpuFrame : FlickerSource::GpuSample);
    EXPECT_EQ(count(Counter::FlickerComparedFrames) - compared1, 0u);
}

// ---- the rule ---------------------------------------------------------------------------------

namespace {
WindowSample flicker_window(uint64_t compared, uint64_t flicker, uint64_t sustained) {
    WindowSample w;
    w.seconds = 5.0;
    w.flips = 300;
    w.target_hz = 60;
    w.counters[static_cast<size_t>(Counter::FlickerPresentedFrames)] = compared + 2;
    w.counters[static_cast<size_t>(Counter::FlickerComparedFrames)] = compared;
    w.counters[static_cast<size_t>(Counter::FlickerFrames)] = flicker;
    w.counters[static_cast<size_t>(Counter::FlickerSustainedFrames)] = sustained;
    return w;
}
bool fired(const std::vector<AlarmFiring>& v, const char* rule) {
    for (const auto& a : v)
        if (std::strcmp(a.rule, rule) == 0) return true;
    return false;
}
}   // namespace

TEST(FrameFlickerRule, FiresOnSustainedFramesAndNamesTheEpisode) {
    FlickerEpisode e;
    e.region_cells = 144;
    e.x0 = 0;
    e.x1 = 15;
    e.y0 = 0;
    e.y1 = 8;
    WindowSample w = flicker_window(298, 290, 230);
    w.gauges[static_cast<size_t>(Gauge::FlickerEpisodeD1Milli)] = 42500;
    w.gauges[static_cast<size_t>(Gauge::FlickerEpisodeD2Milli)] = 300;
    w.gauges[static_cast<size_t>(Gauge::FlickerEpisodeShareMilli)] = 970;
    w.gauges[static_cast<size_t>(Gauge::FlickerEpisodeMs)] = 3900;
    w.gauges[static_cast<size_t>(Gauge::FlickerEpisodeRegion)] = pack_flicker_region(e);
    w.gauges[static_cast<size_t>(Gauge::FlickerOnsetSourceFlip)] = 1234;
    const auto out = evaluate_rules(w, RuleThresholds{});
    ASSERT_TRUE(fired(out, "frame-flicker"));
    for (const auto& a : out) {
        if (std::strcmp(a.rule, "frame-flicker") != 0) continue;
        EXPECT_EQ(a.value, 230.0);
        EXPECT_NE(a.detail.find("period=2"), std::string::npos) << a.detail;
        EXPECT_NE(a.detail.find("d1=42.50 d2=0.30"), std::string::npos) << a.detail;
        EXPECT_NE(a.detail.find("onset-guest-flip=1234"), std::string::npos) << a.detail;
        EXPECT_NE(a.detail.find("region=x0-50%,y0-50%(144/576-cells)"), std::string::npos)
            << a.detail;
        EXPECT_NE(std::string(a.hint).find("Shift+F9"), std::string::npos);
    }
    EXPECT_TRUE(rule_has_data("frame-flicker", w));
    EXPECT_EQ(sustain_windows("frame-flicker"), 1u);
}

TEST(FrameFlickerRule, QuietWithoutASustainedFrameAndNoDataWithoutAFullVote) {
    // Flicker frames that never formed a sustained episode (a strobe) are not an alarm.
    EXPECT_FALSE(
        fired(evaluate_rules(flicker_window(298, 120, 0), RuleThresholds{}), "frame-flicker"));
    // A frontend that compared fewer frames than one vote window has no data, not a quiet answer.
    EXPECT_FALSE(rule_has_data("frame-flicker", flicker_window(kFlickerWindowFrames - 1, 0, 0)));
    EXPECT_TRUE(rule_has_data("frame-flicker", flicker_window(kFlickerWindowFrames, 0, 0)));
    // The threshold-scale knob does not make it more sensitive: it has none.
    EXPECT_FALSE(fired(evaluate_rules(flicker_window(298, 290, 0), RuleThresholds::scaled(1)),
                       "frame-flicker"));
}
