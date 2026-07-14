#include "../tools/screenshot/capture_manifest.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace prosper::screenshot;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

int main() {
    std::printf("== test_capture_manifest ==\n");
    CaptureTracker tracker;
    std::vector<uint8_t> pixels(16 * 8 * 4, 0x11);
    CaptureObservation a{CaptureSource::Rendered, 10, 10, 100, 1, 16, 8, 0x12345678, 1.0};
    auto ca = tracker.observe(a, pixels);
    CHECK(ca.source_advanced && !ca.pixel_identical && ca.stale_seconds == 0,
          "first sample is a new source frame");

    auto b = a; b.elapsed_seconds = 3.5;
    auto cb = tracker.observe(b, pixels);
    CHECK(!cb.source_advanced && cb.pixel_identical && cb.stale_seconds == 2.5 &&
          cb.pixel_stale_seconds == 2.5,
          "same source identity is classified stale");

    auto c = b; c.source_seq = c.frame_seq = 11; c.pixel_crc32 = 0x87654321; c.elapsed_seconds = 4.0;
    c.distinct_rgb_colors = 37;
    c.nonblack_rgb_pixels = 101;
    c.average_hash = 0x0123456789abcdef;
    c.difference_hash = 0xfedcba9876543210;
    c.luma16x9.fill(0x7f);
    pixels[0] = 0x22;
    auto cc = tracker.observe(c, pixels);
    CHECK(cc.source_advanced && !cc.pixel_identical && cc.stale_seconds == 0,
          "new renderer publication resets stale duration");
    auto d = c; d.source_seq = d.frame_seq = 12; d.elapsed_seconds = 6.0;
    auto cd = tracker.observe(d, pixels);
    CHECK(cd.source_advanced && cd.pixel_identical && cd.stale_seconds == 0 &&
          cd.pixel_stale_seconds == 2.0,
          "new source publication can still be visually stale");
    CHECK(tracker.distinct_source_frames() == 3 && tracker.pixel_distinct_frames() == 2 &&
          tracker.rendered_samples() == 4, "tracker reports source, pixel, and composited counts");
    CHECK(tracker.max_stale_seconds() == 2.5 && tracker.max_frame_seq() == 12 &&
          tracker.max_present_count() == 100 && tracker.max_pixel_stale_seconds() == 2.5,
          "tracker preserves assertion maxima");

    const std::string line = manifest_sample_json(2, "a\\b\"c.png", c, cc, "@route\nname");
    CHECK(line.find("a\\\\b\\\"c.png") != std::string::npos, "manifest JSON escapes paths");
    CHECK(line.find("@route\\nname") != std::string::npos, "manifest JSON escapes route text");
    CHECK(line.find("\"source\":\"composited\"") != std::string::npos,
          "manifest names the capture source");
    CHECK(line.find("\"distinct_rgb_colors\":37") != std::string::npos,
          "manifest records a coarse content metric for presented pixels");
    CHECK(line.find("\"nonblack_rgb_pixels\":101") != std::string::npos,
          "manifest records black-frame coverage");
    CHECK(line.find("\"average_hash\":\"0123456789abcdef\"") != std::string::npos &&
          line.find("\"difference_hash\":\"fedcba9876543210\"") != std::string::npos,
          "manifest records tolerant perceptual fingerprints");
    CHECK(line.find("\"luma16x9\":\"7f7f7f7f") != std::string::npos,
          "manifest records a structural luminance signature");

    std::vector<uint8_t> gradient(9 * 8 * 4, 255);
    for (uint32_t y = 0; y < 8; ++y) for (uint32_t x = 0; x < 9; ++x) {
        const size_t offset = (y * 9 + x) * 4;
        gradient[offset] = gradient[offset + 1] = gradient[offset + 2] = static_cast<uint8_t>(x * 28);
    }
    const auto increasing = perceptual_hashes_rgba(gradient, 9, 8);
    CHECK(increasing.difference == 0, "difference hash follows the standard left-greater-than-right rule");
    for (uint32_t y = 0; y < 8; ++y) for (uint32_t x = 0; x < 9; ++x) {
        const size_t offset = (y * 9 + x) * 4;
        gradient[offset] = gradient[offset + 1] = gradient[offset + 2] = static_cast<uint8_t>((8 - x) * 28);
    }
    const auto decreasing = perceptual_hashes_rgba(gradient, 9, 8);
    CHECK(decreasing.difference == UINT64_MAX,
          "difference hash detects the opposite horizontal structure in all 64 cells");
    const auto luma = perceptual_luma16x9_rgba(gradient, 9, 8);
    CHECK(luma.size() == kPerceptualLumaCells && luma.front() > luma.back(),
          "16x9 luminance signature preserves broad spatial structure");
    for (size_t i = 3; i < gradient.size(); i += 4) gradient[i] = 0;
    const auto transparent_luma = perceptual_luma16x9_rgba(gradient, 9, 8);
    CHECK(transparent_luma.front() == 0 && transparent_luma.back() == 0,
          "structural signature ignores invisible RGB beneath zero alpha");
    const std::string long_route(4096, 'x');
    const std::string long_line = manifest_sample_json(3, "long.png", c, cc, long_route);
    CHECK(long_line.size() > long_route.size() && long_line.find(long_route) != std::string::npos,
          "manifest serialization does not truncate long routes");

    CaptureRunConfig config;
    config.title = "PPSA15552";
    config.input_route = "@scripts/dead-cells/route.pad";
    config.time_mode = true;
    config.seconds = 1;
    config.requested = 120;
    config.render_every = "1000";
    config.render_every_for_ms = "120000";
    config.min_distinct_frames = 10;
    config.min_pixel_distinct_frames = 8;
    config.max_pixel_stale_seconds = 3.5;
    config.require_composited_frame = true;
    config.required_crc32_set = true;
    config.required_crc32 = 0x1234abcd;
    const std::string run = manifest_run_json(config);
    CHECK(run.find("\"capture_mode\":\"wall_seconds\"") != std::string::npos &&
          run.find("\"render_every\":\"1000\"") != std::string::npos &&
          run.find("\"render_every_for_ms\":\"120000\"") != std::string::npos,
          "run header records cadence and renderer policy");
    CHECK(run.find("\"required_crc32\":\"1234abcd\"") != std::string::npos,
          "run header records checkpoint assertions");
    CHECK(run.find("\"min_pixel_distinct_frames\":8") != std::string::npos &&
          run.find("\"max_pixel_stale_seconds\":3.500000") != std::string::npos,
          "run header records pixel-progress assertions");

    const std::string summary = manifest_summary_json(3, 5, true, tracker, 1);
    CHECK(summary.find("\"timed_out\":true") != std::string::npos &&
          summary.find("\"pixel_distinct_frames\":2") != std::string::npos &&
          summary.find("\"max_pixel_stale_seconds\":2.500000") != std::string::npos &&
          summary.find("\"exit_code\":1") != std::string::npos,
          "summary records incomplete failure");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
