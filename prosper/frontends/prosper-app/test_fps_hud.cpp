// test_fps_hud.cpp — the `--fps` HUD must not be readable as "this title runs at 60 fps" when the
// title is frozen. Pure: no SDL, no Vulkan, no ImGui.
#include "fps_hud.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace prosper::frontend;
using prosper::gpu::FrameRate;
using prosper::gpu::PresentRateSnapshot;
using prosper::gpu::frame_rate_between;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

namespace {

FrameRate window(uint64_t published, uint64_t distinct, double seconds) {
    PresentRateSnapshot a, b;
    a.now_seconds = 0;
    b.published = published;
    b.distinct = distinct;
    b.now_seconds = seconds;
    return frame_rate_between(a, b);
}

bool contains(const std::vector<std::string>& lines, const std::string& needle) {
    for (const std::string& line : lines)
        if (line.find(needle) != std::string::npos) return true;
    return false;
}

} // namespace

int main() {
    // A healthy title. The two rates are deliberately DIFFERENT (55 distinct of 60 published): with
    // both at 60 the headline check could not tell "the headline is the distinct rate" from "the
    // headline is the presented rate", and the claim would rest entirely on the frozen arm below.
    {
        const std::vector<std::string> lines = fps_hud_lines(window(60, 55, 1.0), 1920, 1080, 3600);
        CHECK(lines.size() == 2, "a healthy title gets two lines");
        CHECK(lines[0] == "55.0 fps", "the headline is the DISTINCT rate, not the presented one");
        CHECK(contains(lines, "60.0 presented"), "the presented rate is shown and labelled");
        CHECK(contains(lines, "1920x1080"), "the resolution is shown -- an fps has no meaning without it");
        CHECK(contains(lines, "3600 frames"), "the population behind the rate is shown");
        CHECK(!contains(lines, "not changing"), "a healthy title is not warned about");
    }

    // THE ARM. Identical publication rate; the content never changes. The headline must NOT read 60.
    {
        const std::vector<std::string> lines = fps_hud_lines(window(60, 0, 1.0), 3840, 2160, 1);
        CHECK(lines[0] == "0.0 fps",
              "a frozen title's HEADLINE is 0.0 fps, not the 60 it is publishing");
        CHECK(contains(lines, "60.0 presented"),
              "...the presented rate is still shown, so the divergence is visible rather than hidden");
        CHECK(contains(lines, "picture not changing"),
              "...and the reader is told in words that the picture is not advancing");
        CHECK(!contains(lines, "retained") && !contains(lines, "RETAINED"),
              "...without asserting the RENDERER as the cause: a static menu is indistinguishable "
              "from a re-served retained frame here, and naming the second manufactures a defect");
    }

    // A title genuinely running slowly is NOT a frozen one, and must not be labelled as one. This is
    // the arm that stops the warning being keyed on an absolute rate: prosper has titles that
    // legitimately run at ~1 fps, and calling those frozen would make the warning noise.
    {
        const std::vector<std::string> lines = fps_hud_lines(window(1, 1, 1.0), 3840, 2160, 42);
        CHECK(lines[0] == "1.0 fps", "a genuinely 1 fps title reports 1.0 fps");
        CHECK(!contains(lines, "not changing"),
              "a slow-but-live title is not warned about -- every publication carried content");
    }

    // Before anything is published there is no rate. "-- fps" is a different claim from "0.0 fps",
    // and a HUD that showed 0.0 during boot would look like a hang.
    {
        const std::vector<std::string> lines = fps_hud_lines(FrameRate{}, 0, 0, 0);
        CHECK(lines[0] == "-- fps", "nothing published yet shows no rate rather than zero");
        CHECK(contains(lines, "no frames published yet"), "...and says why");
    }

    // The rolling window: due at the boundary, not before.
    CHECK(!fps_window_due(10.0, 10.5, 1.0), "the window is not due before it elapses");
    CHECK(fps_window_due(10.0, 11.0, 1.0), "the window is due exactly at the boundary");
    CHECK(fps_window_due(10.0, 40.0, 1.0), "a long gap is due");

    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
