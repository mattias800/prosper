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

    PresentRateSnapshot snapshot;
    snapshot.published = counter.published();
    snapshot.distinct = counter.distinct();
    snapshot.first_publication_seconds = counter.first_publication_seconds();
    snapshot.now_seconds = 600 / 60.0;

    const FrameRate rate = frame_rate_since_first_publication(snapshot);
    CHECK(rate.measured, "a live title's rate is measured");
    CHECK(rate.published == 600, "every publication is counted");
    CHECK(rate.distinct == 600, "every publication carried new content");
    CHECK(rate.presented_fps > 55 && rate.presented_fps < 65, "presented rate is ~60 fps");
    CHECK(rate.distinct_fps > 55 && rate.distinct_fps < 65, "distinct rate is ~60 fps");
    CHECK(!frame_rate_is_mostly_retained(rate), "a live title is not labelled mostly-retained");
}

// THE ARM. Identical timing and identical publication count to the live case above; the ONLY
// difference is that the content never changes.
void frozen_title_does_not_report_full_speed() {
    const std::vector<uint8_t> retained = frame(7);
    FrameRateCounter counter;
    for (int i = 0; i < 600; i++)
        counter.observe(signature_of(retained), i / 60.0);

    PresentRateSnapshot snapshot;
    snapshot.published = counter.published();
    snapshot.distinct = counter.distinct();
    snapshot.first_publication_seconds = counter.first_publication_seconds();
    snapshot.now_seconds = 600 / 60.0;

    const FrameRate rate = frame_rate_since_first_publication(snapshot);
    CHECK(rate.published == 600, "a frozen title still PUBLISHES at full rate");
    CHECK(rate.presented_fps > 55 && rate.presented_fps < 65,
          "...so its presented rate is ~60 fps, which is exactly the trap");
    CHECK(rate.distinct == 1, "only the first publication carried new content");
    CHECK(rate.distinct_fps < 1.0, "the distinct rate does NOT report full speed");
    CHECK(frame_rate_is_mostly_retained(rate), "a frozen title is labelled mostly-retained");

    const std::string text = format_frame_rate(rate);
    CHECK(text.find("distinct 0.1 fps") == 0,
          "the formatted line leads with the distinct rate, not the presented one");
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

void window_math() {
    PresentRateSnapshot a, b;
    a.published = 100; a.distinct = 100; a.now_seconds = 10.0;
    b.published = 400; b.distinct = 105; b.now_seconds = 15.0;

    const FrameRate window = frame_rate_between(a, b);
    CHECK(window.measured, "a window between two snapshots is measured");
    CHECK(window.published == 300 && window.distinct == 5, "the window subtracts both counters");
    CHECK(window.presented_fps > 59.9 && window.presented_fps < 60.1, "300 publications / 5 s");
    CHECK(window.distinct_fps > 0.9 && window.distinct_fps < 1.1, "5 distinct frames / 5 s");
    CHECK(frame_rate_is_mostly_retained(window),
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

} // namespace

int main() {
    std::printf("== content signature ==\n");            signature_discriminates();
    std::printf("== live title ==\n");                   live_title_reports_its_real_rate();
    std::printf("== FROZEN title (the arm) ==\n");       frozen_title_does_not_report_full_speed();
    std::printf("== frozen, distinct buffers ==\n");     frozen_title_with_distinct_buffers_is_still_frozen();
    std::printf("== window arithmetic ==\n");            window_math();
    std::printf("== present-layer wiring ==\n");         present_layer_wiring();
    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
