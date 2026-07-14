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
