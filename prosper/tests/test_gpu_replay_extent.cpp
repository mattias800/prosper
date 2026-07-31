#include "../tools/gpu_replay/replay_output_extent.hpp"

#include <cstdio>

using namespace prosper::gpu;
using namespace prosper::gpu::replay_tool;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_gpu_replay_extent ==\n");

    GpuReplayFrame replay;
    replay.metadata.width = 3840;
    replay.metadata.height = 2160;

    DrawItem offscreen;
    offscreen.draw_index = 23;
    offscreen.color0_width = 642;
    offscreen.color0_height = 362;
    replay.items.push_back(offscreen);

    OutputExtent extent = replay_output_extent(
        replay, OutputExtentMode::Capture, static_cast<size_t>(3840) * 2160 * 4);
    CHECK(extent.width == 3840 && extent.height == 2160,
          "full replay retains the capture output extent");

    extent = replay_output_extent(
        replay, OutputExtentMode::SelectedDraws, static_cast<size_t>(642) * 362 * 4);
    CHECK(extent.width == 642 && extent.height == 362,
          "single selected draw uses its offscreen target extent");

    DrawItem later = offscreen;
    later.draw_index = 24;
    later.color0_width = 320;
    later.color0_height = 180;
    replay.items.push_back(later);
    extent = replay_output_extent(
        replay, OutputExtentMode::SelectedDraws, static_cast<size_t>(320) * 180 * 4);
    CHECK(extent.width == 320 && extent.height == 180,
          "selected draw range uses the final selected target extent");

    replay.operations = {
        {SubmitOperationKind::Draw, 23, 100, true},
        {SubmitOperationKind::Dispatch, 7, 200, true},
        {SubmitOperationKind::Draw, 24, 300, true},
    };
    extent = replay_output_extent(
        replay, OutputExtentMode::OrderedPrefix, static_cast<size_t>(642) * 362 * 4, 1);
    CHECK(extent.width == 642 && extent.height == 362,
          "ordered prefix uses its last realized draw extent");
    extent = replay_output_extent(
        replay, OutputExtentMode::OrderedPrefix, static_cast<size_t>(642) * 362 * 4, 2);
    CHECK(extent.width == 642 && extent.height == 362,
          "compute operations do not change the output extent");
    extent = replay_output_extent(
        replay, OutputExtentMode::OrderedPrefix, static_cast<size_t>(320) * 180 * 4, 3);
    CHECK(extent.width == 320 && extent.height == 180,
          "later realized draw replaces the ordered-prefix extent");

    extent = replay_output_extent(
        replay, OutputExtentMode::Capture, static_cast<size_t>(320) * 180 * 4);
    CHECK(extent.width == 320 && extent.height == 180,
          "full replay uses the final realized target when bytes disprove the capture-header extent");
    extent = replay_output_extent(
        replay, OutputExtentMode::Capture, static_cast<size_t>(3840) * 2160 * 4);
    CHECK(extent.width == 3840 && extent.height == 2160,
          "full replay preserves capture resolution when its exact byte count matches");

    replay.items.back().color0_width = 0;
    replay.items.back().color0_height = 0;
    extent = replay_output_extent(
        replay, OutputExtentMode::SelectedDraws, static_cast<size_t>(3840) * 2160 * 4);
    CHECK(extent.width == 3840 && extent.height == 2160,
          "missing target dimensions fall back to the capture extent");

    replay.items = {offscreen};
    extent = replay_output_extent(
        replay, OutputExtentMode::SelectedDraws, static_cast<size_t>(3840) * 2160 * 4);
    CHECK(extent.width == 3840 && extent.height == 2160,
          "capture-sized renderer output overrides stale native draw dimensions");

    // Regression for the #1486 class of false bisect boundary. Two adjacent ordered-prefix cutoffs
    // that resolve to DIFFERENT color targets must report different addresses, so that comparing
    // their images is visibly not like-for-like. Reporting only an extent is what let op116
    // (a 1920x1080 post-process input) and op117 (the 3840x2160 graded scene) read as a single
    // surface being corrupted between two cutoffs, when they were never the same buffer.
    {
        GpuReplayFrame bisect;
        bisect.metadata.width = 3840;
        bisect.metadata.height = 2160;

        DrawItem post;                       // the post-process input a later draw samples
        post.draw_index = 88;
        post.color0_base = 0x3083db0000ull;
        post.color0_width = 1920;
        post.color0_height = 1080;

        DrawItem scene;                      // the full-resolution graded scene
        scene.draw_index = 90;
        scene.color0_base = 0x30867e0000ull;
        scene.color0_width = 3840;
        scene.color0_height = 2160;

        bisect.items = {post, scene};
        bisect.operations = {
            {SubmitOperationKind::Draw, 88, 100, true},
            {SubmitOperationKind::Dispatch, 26, 200, true},
            {SubmitOperationKind::Draw, 90, 300, true},
        };

        const OutputExtent earlier = replay_output_extent(
            bisect, OutputExtentMode::OrderedPrefix, static_cast<size_t>(1920) * 1080 * 4, 2);
        const OutputExtent later_cut = replay_output_extent(
            bisect, OutputExtentMode::OrderedPrefix, static_cast<size_t>(3840) * 2160 * 4, 3);

        CHECK(earlier.target_addr == 0x3083db0000ull && earlier.target_draw_index == 88,
              "prefix ending at a dispatch reports the draw target it actually rendered");
        CHECK(later_cut.target_addr == 0x30867e0000ull && later_cut.target_draw_index == 90,
              "the next cutoff reports the full-resolution target");
        CHECK(earlier.target_addr != later_cut.target_addr,
              "adjacent cutoffs on different surfaces report different targets");
    }

    // The capture-extent fallback must not claim a draw target whose dimensions it does not report;
    // an address next to the wrong extent would be worse than reporting none.
    {
        GpuReplayFrame plain;
        plain.metadata.width = 64;
        plain.metadata.height = 64;
        DrawItem odd;
        odd.draw_index = 3;
        odd.color0_base = 0x1234000ull;
        odd.color0_width = 8;
        odd.color0_height = 8;
        plain.items = {odd};
        plain.operations = {{SubmitOperationKind::Draw, 3, 10, true}};

        const OutputExtent fallback = replay_output_extent(
            plain, OutputExtentMode::OrderedPrefix, static_cast<size_t>(64) * 64 * 4, 1);
        CHECK(fallback.width == 64 && fallback.height == 64,
              "byte count disproving the draw extent falls back to the capture extent");
        CHECK(fallback.target_addr == 0,
              "capture-extent fallback reports no target rather than a mismatched address");
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
