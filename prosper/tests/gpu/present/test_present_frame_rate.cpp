// test_present_frame_rate.cpp — the framerate metric must not read a FROZEN title as full speed.
//
// This file exists for one arm, and the rest of it is there to stop that arm passing for the wrong
// reason. The arm is `frozen_title_does_not_report_full_speed`: 600 publications at 60 Hz that all
// carry the same picture must report a presented rate near 60 and a distinct rate near 0. That is
// the R-Type Delta (#2783) failure shape — the guest reached stage 1 while every presented frame
// was the renderer's re-served retained one, and a present-counting fps would have read ~60 through
// the whole nine-day regression.
//
// A frozen-arm on its own is satisfied by a broken metric that answers "identical" to everything,
// so the discriminator is pinned from BOTH sides here:
//
//   * `signature_sees_a_changed_block` and `live_title_reports_its_real_rate` fail if the signature
//     ever collapses to a constant. Note that the changed block is constructed BY HAND, outside the
//     accumulator that consumes it — a positive control drawn from the same source as the null it
//     validates tests the discriminator and not the domain (CLAUDE.md; instrument trap 122).
//   * `frozen_title_*` fails if the signature becomes always-different.
//
// Neither direction can be removed without a red test, which is the property that makes the number
// quotable. Pure: no Vulkan, no game dump, no renderer.
#include "gpu/present/present_frame_rate.hpp"
#include "gpu/present/videoout_present.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

namespace {

constexpr uint32_t kW = 64, kH = 64;

// Everything the counter knows, as the snapshot a consumer would take. Kept in one place so an arm
// cannot accidentally leave the interval fields at zero and then "prove" the typical rate is absent.
PresentRateSnapshot snapshot_of(const FrameRateCounter& counter, double now_seconds) {
    PresentRateSnapshot s;
    s.published = counter.published();
    s.distinct = counter.distinct();
    s.first_publication_seconds = counter.first_publication_seconds();
    s.last_publication_seconds = counter.last_publication_seconds();
    s.typical_interval_seconds = counter.typical_interval_seconds();
    s.active_seconds = counter.active_seconds();
    s.interval_samples = counter.interval_samples();
    s.now_seconds = now_seconds;
    return s;
}

constexpr size_t kBytes = static_cast<size_t>(kW) * kH * 4;   // 16384

// A frame whose bytes are a deterministic function of `variant`. Every byte differs between
// variants, so "these two frames are different" needs no argument about sampling.
std::vector<uint8_t> frame(uint8_t variant) {
    std::vector<uint8_t> px(kBytes);
    for (size_t i = 0; i < px.size(); i++)
        px[i] = static_cast<uint8_t>((i * 31u + variant * 97u) & 0xff);
    return px;
}

// A change confined to one contiguous run, sized at the guarantee the header states: any run of at
// least kFrameSignatureBlockBytes + kFrameSignatureBytesPerBlock bytes must contain a whole sampled
// window. Built here, by hand, rather than by asking the metric to produce a case it likes.
std::vector<uint8_t> frame_with_changed_block(const std::vector<uint8_t>& base, size_t at) {
    std::vector<uint8_t> px = base;
    const size_t run = kFrameSignatureBlockBytes + kFrameSignatureBytesPerBlock;
    for (size_t i = at; i < at + run && i < px.size(); i++) px[i] = static_cast<uint8_t>(~px[i]);
    return px;
}

uint64_t signature_of(const std::vector<uint8_t>& px) {
    return frame_content_signature(px.data(), px.size(), kW, kH);
}

void signature_discriminates() {
    const std::vector<uint8_t> a = frame(0);
    std::vector<uint8_t> copy = a;

    CHECK(signature_of(a) == signature_of(copy),
          "byte-identical frames share a signature");
    CHECK(signature_of(a) != signature_of(frame(1)),
          "wholly different frames have different signatures");

    // The header's guarantee, checked at three offsets: start, middle, and a run that straddles the
    // final partial block (which the block loop samples only from its start).
    for (size_t at : {size_t{0}, kBytes / 2, kBytes - (kFrameSignatureBlockBytes + kFrameSignatureBytesPerBlock)}) {
        const uint64_t changed = signature_of(frame_with_changed_block(a, at));
        char message[128];
        std::snprintf(message, sizeof message,
                      "a %zu-byte contiguous change at offset %zu is seen",
                      kFrameSignatureBlockBytes + kFrameSignatureBytesPerBlock, at);
        CHECK(changed != signature_of(a), message);
    }

    // Dimensions are content: a resolution change must never be mistaken for a repeat. 128x32 and
    // 32x128 hold the same byte count as 64x64, so this arm varies ONLY the declared geometry.
    CHECK(frame_content_signature(a.data(), a.size(), 128, 32) !=
              frame_content_signature(a.data(), a.size(), 32, 128),
          "the same bytes at different dimensions signature differently");
    CHECK(frame_content_signature(a.data(), a.size(), kW, kH) !=
              frame_content_signature(a.data(), a.size(), 128, 32),
          "...and differently again from the square geometry the rest of this file uses");
    CHECK(frame_content_signature(nullptr, 0, 0, 0) ==
              frame_content_signature(nullptr, 0, 0, 0),
          "an empty frame has a stable signature rather than reading uninitialised memory");

    // THE TAIL BRANCH. Every frame above is an exact multiple of the block size, so the explicit
    // final-16-bytes read never mattered to any arm -- the block loop happened to cover the end.
    // A frame whose byte count is NOT a multiple of the block size leaves the last partial block
    // sampled only from its start, and without the tail read a change in the last few KiB would be
    // invisible. That is the bottom-right of the image, which is exactly where a padded-footprint
    // or a partially-composited frame differs (videoout_present.hpp's `padded_footprint`).
    {
        const size_t odd = 3 * kFrameSignatureBlockBytes + 777;   // deliberately not a multiple
        std::vector<uint8_t> base(odd, 0xa5);
        std::vector<uint8_t> tail_changed = base;
        for (size_t i = odd - 8; i < odd; i++) tail_changed[i] = 0x5a;
        CHECK(frame_content_signature(base.data(), odd, 1, 1) !=
                  frame_content_signature(tail_changed.data(), odd, 1, 1),
              "a change in the FINAL bytes of a non-block-multiple frame is seen");

        // ...and a frame shorter than one sample window must not read past its end. Correctness
        // here is memory safety, not sensitivity: the assertion is that this terminates and is
        // stable, which under ASan/UBSan is a real check.
        std::vector<uint8_t> tiny(kFrameSignatureBytesPerBlock - 3, 0x11);
        CHECK(frame_content_signature(tiny.data(), tiny.size(), 1, 1) ==
                  frame_content_signature(tiny.data(), tiny.size(), 1, 1),
              "a frame shorter than one sample window signatures without reading past its end");
    }
}

// 60 publications per second for 10 seconds, each carrying new content.
void live_title_reports_its_real_rate() {
    FrameRateCounter counter;
    for (int i = 0; i < 600; i++)
        counter.observe(signature_of(frame(static_cast<uint8_t>(i % 251))), i / 60.0);

    const FrameRate rate = frame_rate_since_first_publication(snapshot_of(counter, 600 / 60.0));
    CHECK(rate.measured, "a live title's rate is measured");
    CHECK(rate.published == 600, "every publication is counted");
    CHECK(rate.distinct == 600, "every publication carried new content");
    CHECK(rate.presented_fps > 55 && rate.presented_fps < 65, "presented rate is ~60 fps");
    CHECK(rate.distinct_fps > 55 && rate.distinct_fps < 65, "distinct rate is ~60 fps");
    CHECK(rate.typical_measured && rate.typical_fps > 55 && rate.typical_fps < 65,
          "the TYPICAL rate is ~60 fps too -- with no idling, headline and average agree");
    CHECK(rate.active_fraction > 0.95,
          "a title producing frames throughout is ~100% active");
    CHECK(!frame_rate_is_mostly_unchanged(rate), "a live title is not labelled mostly-retained");
}

// THE ARM. Identical timing and identical publication count to the live case above; the ONLY
// difference is that the content never changes.
void frozen_title_does_not_report_full_speed() {
    const std::vector<uint8_t> retained = frame(7);
    FrameRateCounter counter;
    for (int i = 0; i < 600; i++)
        counter.observe(signature_of(retained), i / 60.0);

    const FrameRate rate = frame_rate_since_first_publication(snapshot_of(counter, 600 / 60.0));
    CHECK(rate.published == 600, "a frozen title still PUBLISHES at full rate");
    CHECK(rate.presented_fps > 55 && rate.presented_fps < 65,
          "...so its presented rate is ~60 fps, which is exactly the trap");
    CHECK(rate.distinct == 1, "only the first publication carried new content");
    CHECK(rate.distinct_fps < 1.0, "the distinct rate does NOT report full speed");
    CHECK(frame_rate_is_mostly_unchanged(rate), "a frozen title is labelled mostly-retained");

    // The headline must be ABSENT, not zero and not fast. One distinct frame yields no interval, so
    // there is nothing to take a median of -- and reporting "0.0 fps" would be a measurement where
    // none exists. The 0% beside it is what makes the absence legible.
    CHECK(!rate.typical_measured && rate.typical_fps == 0,
          "a frozen title has NO typical rate rather than a zero one");
    CHECK(rate.active_fraction == 0, "...and is 0% active");

    const std::string text = format_frame_rate(rate);
    CHECK(text.find("-- fps while producing frames, 0% of the") == 0,
          "the formatted headline reads '-- fps ... 0% active', never a number");
    CHECK(text.find("there is no framerate to report") != std::string::npos,
          "...and says so in words");
    CHECK(format_frame_rate_short(rate, 1920, 1080) == "-- fps  0% active  1920x1080",
          "the compact form a HUD or an overlay burns says the same thing");
}

// The same freeze, but with a DIFFERENT buffer holding identical bytes each time. The renderer's
// re-serve path hands back the same shared_ptr, so pointer identity alone would pass the arm above;
// this arm is what proves the metric compares CONTENT.
void frozen_title_with_distinct_buffers_is_still_frozen() {
    FrameRateCounter counter;
    for (int i = 0; i < 120; i++) {
        const std::vector<uint8_t> fresh_copy = frame(7);   // new allocation, same picture
        counter.observe(signature_of(fresh_copy), i / 60.0);
    }
    CHECK(counter.published() == 120, "every copy was published");
    CHECK(counter.distinct() == 1, "identical pictures in different buffers are one distinct frame");
}

// THE ARM FOR THE HEADLINE. This is The Messenger's measured shape, reproduced exactly: bursts of
// real frames around a long idle stretch, where the run AVERAGE describes neither mode.
//
// A title that renders at 20 fps and then sits on a menu is a 20 fps title. Averaging it to 4 puts
// it in the "we have work to do" bucket it does not belong in -- and that bucket decision is what
// the number is for. So the headline is a median over intervals, which weights by FRAME: the
// 120-second pause below is ONE interval, not 120 seconds of pull.
void a_title_that_pauses_reports_its_producing_rate() {
    FrameRateCounter counter;
    double t = 0;
    auto burst = [&](int frames) {
        for (int i = 0; i < frames; i++) {
            counter.observe(signature_of(frame(static_cast<uint8_t>(i % 251))), t);
            t += 1.0 / 20.0;                       // 20 fps while producing
        }
    };
    burst(300);                                    // 15 s of real frames
    const std::vector<uint8_t> held = frame(200);
    for (int i = 0; i < 2400; i++) {               // 120 s of a frozen picture, republished at 20 Hz
        counter.observe(signature_of(held), t);
        t += 1.0 / 20.0;
    }
    burst(300);                                    // 15 s more

    const FrameRate rate = frame_rate_since_first_publication(snapshot_of(counter, t));

    // What the average says, and why it is not quotable: 601 distinct frames over 150 seconds.
    CHECK(rate.distinct_fps > 3.5 && rate.distinct_fps < 4.5,
          "the run AVERAGE is ~4 fps -- a true number that describes neither mode");

    // What the headline says.
    CHECK(rate.typical_measured, "a title that produced frames has a typical rate");
    CHECK(rate.typical_fps > 17 && rate.typical_fps < 23,
          "the TYPICAL rate is ~20 fps: the rate it runs at while it is running");
    CHECK(rate.active_fraction > 0.12 && rate.active_fraction < 0.30,
          "...and it was active for ~20% of the run, which is what stops 20 fps being quoted bare");

    // The three cases the pair has to separate, checked against each other rather than in isolation.
    // A slow-but-healthy title is the one an absolute threshold would misfile, so it is here.
    FrameRateCounter slow;
    for (int i = 0; i < 150; i++) slow.observe(signature_of(frame(static_cast<uint8_t>(i % 251))), i * 1.0);
    const FrameRate slow_rate = frame_rate_since_first_publication(snapshot_of(slow, 150.0));
    CHECK(slow_rate.typical_measured &&
              std::fabs(slow_rate.typical_fps - 1.0) <= kIntervalRelativeError * 1.0 + 1e-9,
          "a genuinely 1 fps title reports 1 fps, within the documented bucket tolerance");
    CHECK(slow_rate.active_fraction > 0.95,
          "...at ~100% active -- LOW AND ACTIVE is the 'we have work to do' bucket, and it is "
          "distinguishable from fast-then-idle only because both numbers are reported");
    CHECK(slow_rate.typical_fps < rate.typical_fps && slow_rate.active_fraction > rate.active_fraction,
          "the slow title and the paused title differ in BOTH fields, in opposite directions");

    // THE C++ HALF OF A CROSS-LANGUAGE CONTRACT. gen_progress_tracker.py's grammar accepts
    // "<n> fps while producing frames, <n>% active" because that is what this function prints, and
    // its selftest pins the ACCEPTING side. Nothing there can observe this formatter, so if the
    // wording changed the Python arm would keep passing while the grammar silently stopped accepting
    // the tool's own output -- which is exactly the defect that grammar was fixed for. This is the
    // side that can move, so this is where it is pinned.
    CHECK(format_frame_rate(rate).find(" fps while producing frames,") != std::string::npos,
          "the summary phrasing gen_progress_tracker.py's FPS_RECORD_RE accepts is still emitted");
}

// The recovered rate must match a known input to within the tolerance the header states. Without
// this, the bucket constants could be widened for memory and the numbers would quietly get worse --
// and these figures are filed in game trackers and compared across releases.
void interval_estimator_is_accurate() {
    for (double fps : {0.5, 1.0, 5.0, 20.0, 30.0, 60.0, 144.0}) {
        FrameRateCounter counter;
        const double dt = 1.0 / fps;
        for (int i = 0; i < 200; i++)
            counter.observe(signature_of(frame(static_cast<uint8_t>(i % 251))), i * dt);
        const double got = 1.0 / counter.typical_interval_seconds();
        char message[128];
        std::snprintf(message, sizeof message,
                      "a known %.1f fps signal is recovered as %.2f fps (within %.0f%%)",
                      fps, got, kIntervalRelativeError * 100.0);
        CHECK(std::fabs(got - fps) <= kIntervalRelativeError * fps + 1e-9, message);
    }
}

void window_math() {
    PresentRateSnapshot a, b;
    a.published = 100; a.distinct = 100; a.now_seconds = 10.0;
    b.published = 400; b.distinct = 105; b.now_seconds = 15.0;

    const FrameRate window = frame_rate_between(a, b);
    CHECK(window.measured, "a window between two snapshots is measured");
    CHECK(!window.typical_measured && window.typical_fps == 0 && window.active_fraction == 0,
          "a differenced window claims NO typical rate: a cumulative histogram cannot be "
          "subtracted, and a plausible wrong number is worse than an absent one");
    CHECK(window.published == 300 && window.distinct == 5, "the window subtracts both counters");
    CHECK(window.presented_fps > 59.9 && window.presented_fps < 60.1, "300 publications / 5 s");
    CHECK(window.distinct_fps > 0.9 && window.distinct_fps < 1.1, "5 distinct frames / 5 s");
    CHECK(frame_rate_is_mostly_unchanged(window),
          "a title that froze DURING the window is caught by the window, not only by the run total");

    // A reset between the readings must produce "no measurement", never a wrapped one.
    const FrameRate backwards = frame_rate_between(b, a);
    CHECK(!backwards.measured && backwards.published == 0,
          "counters running backwards report no measurement rather than a wrapped window");

    PresentRateSnapshot empty;
    CHECK(!frame_rate_since_first_publication(empty).measured,
          "nothing published yet is not a measured 0 fps");
}

// The wiring, not the arithmetic: does present_write_frame actually feed the counter, and does
// present_frame_seq keep climbing while the distinct count does not?
void present_layer_wiring() {
    present_reset();
    const PresentRateSnapshot start = present_rate_snapshot();
    CHECK(start.published == 0 && start.distinct == 0, "present_reset clears the rate counters");

    auto retained = std::make_shared<const std::vector<uint8_t>>(frame(3));
    const uint64_t seq_before = present_frame_seq();
    for (int i = 0; i < 64; i++) present_write_frame(retained, kW, kH);

    const PresentRateSnapshot frozen = present_rate_snapshot();
    CHECK(present_frame_seq() - seq_before == 64,
          "the present layer counts all 64 publications (this is what a present-rate fps sees)");
    CHECK(frozen.published == 64, "the rate counter saw all 64 publications");
    CHECK(frozen.distinct == 1,
          "re-serving one retained frame 64 times is ONE distinct frame -- the #2783 shape");

    // EIGHT DIFFERENT pictures, not two alternating ones: the whole frame varies per iteration, so
    // "8 more distinct" is the property under test rather than "at least one of them registered".
    // The assertion is exact for the same reason -- `>` would pass if seven of the eight were
    // silently dropped, and an off-by-one in the counter is precisely what it should catch.
    for (int i = 0; i < 8; i++) {
        auto fresh = std::make_shared<const std::vector<uint8_t>>(frame(static_cast<uint8_t>(10 + i)));
        present_write_frame(fresh, kW, kH);
    }
    const PresentRateSnapshot after = present_rate_snapshot();
    CHECK(after.published == 72, "the eight new frames were published");
    CHECK(after.distinct == frozen.distinct + 8,
          "all eight advance the distinct counter through the real present path -- exactly eight");

    // A rejected publication (wrong byte count for the extent) must count as neither.
    auto wrong_extent = std::make_shared<const std::vector<uint8_t>>(std::vector<uint8_t>(16));
    present_write_frame(wrong_extent, kW, kH);
    CHECK(present_rate_snapshot().published == 72,
          "a publication the present layer rejects is not counted as a frame");
    present_reset();
}

// The GPU present path signs a SMALL sample densely instead of a large frame sparsely (#3010).
// Both halves of that choice are pinned here, and the sample buffers are built BY HAND rather than
// drawn from whatever the accumulator would produce (instrument trap 122).
void dense_signature_pins_the_gpu_path() {
    // The grid prosper-app actually uses: 256x144 RGBA8.
    constexpr size_t kBytes = 256u * 144u * 4u;
    std::vector<uint8_t> a(kBytes, 0x40);
    std::vector<uint8_t> b = a;
    // ONE pixel, in the middle, changed by one step. This is the case the counter has to see: a
    // title whose picture changes only in a small region must not read as frozen.
    const size_t px = ((144u / 2) * 256u + 128u) * 4u;
    b[px + 1] = 0x41;

    CHECK(dense_content_signature(a.data(), a.size()) !=
          dense_content_signature(b.data(), b.size()),
          "the dense signature sees a single changed pixel in the sample grid");
    CHECK(dense_content_signature(a.data(), a.size()) ==
          dense_content_signature(a.data(), a.size()),
          "...and is stable for identical bytes");
    CHECK(dense_content_signature(nullptr, 0) == dense_content_signature(nullptr, 0),
          "an empty sample signs consistently rather than trapping");

    // WHY dense, ASSERTED rather than printed: at this buffer size the sparse whole-frame signature
    // reads only a few hundred bytes and misses the same change. This was a printf, which made it an
    // experiment that could not fail -- if the sparse signature ever became sensitive enough here the
    // dense variant would have lost its justification and nobody would have been told. Now the day
    // that changes, this goes red and someone removes dense_content_signature on purpose.
    CHECK(frame_content_signature(a.data(), a.size(), 256, 144) ==
              frame_content_signature(b.data(), b.size(), 256, 144),
          "the sparse whole-frame signature MISSES it at this size -- which is why dense exists");

    // And the publication accounting is the same as the pixel entry point's.
    present_reset();
    const uint64_t sig_a = dense_content_signature(a.data(), a.size());
    const uint64_t sig_b = dense_content_signature(b.data(), b.size());
    note_present_publication_signature(sig_a);
    note_present_publication_signature(sig_a);
    note_present_publication_signature(sig_b);
    const PresentRateSnapshot s = present_rate_snapshot();
    CHECK(s.published == 3, "three signatures published");
    CHECK(s.distinct == 2, "the repeat is not distinct; the changed one is");
    present_reset();
}

} // namespace

int main() {
    std::printf("== content signature ==\n");            signature_discriminates();
    std::printf("== live title ==\n");                   live_title_reports_its_real_rate();
    std::printf("== FROZEN title (the arm) ==\n");       frozen_title_does_not_report_full_speed();
    std::printf("== frozen, distinct buffers ==\n");     frozen_title_with_distinct_buffers_is_still_frozen();
    std::printf("== a title that pauses (the arm) ==\n"); a_title_that_pauses_reports_its_producing_rate();
    std::printf("== estimator accuracy ==\n");           interval_estimator_is_accurate();
    std::printf("== window arithmetic ==\n");            window_math();
    std::printf("== present-layer wiring ==\n");         present_layer_wiring();
    std::printf("== dense signature (GPU path) ==\n"); dense_signature_pins_the_gpu_path();
    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
