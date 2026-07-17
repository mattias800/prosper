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

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
