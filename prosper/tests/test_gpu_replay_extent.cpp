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

    // Exact post-operation target selection is a write proof, not an address/binding lookup. This
    // is the CPU-only policy used by --output-target-after before it asks Vulkan for a readback.
    {
        GpuReplayFrame exact;
        DrawItem writer;
        writer.draw_index = 41;
        writer.color_targets[0] = {0x2010000000ull, 1920, 1080};
        writer.ps.color_targets[0].format = 37; // VK_FORMAT_R8G8B8A8_UNORM
        writer.ps.color_targets[0].write_mask = 0xf;
        exact.items = {writer};
        exact.operations = {{SubmitOperationKind::Draw, 41, 100, true}};

        const auto selected = replay_output_target_after_operation(exact, 0, 0x2010000000ull);
        CHECK(selected.status == OutputTargetAfterStatus::Selected &&
                  selected.target.operation_index == 0 && selected.target.draw_index == 41 &&
                  selected.target.slot == 0 && selected.target.guest_addr == 0x2010000000ull &&
                  selected.target.width == 1920 && selected.target.height == 1080 &&
                  selected.target.format == 37 && !selected.target.fixed_function_resolve,
              "exact selector retains operation/draw/slot/address/extent/format write proof");

        CHECK(replay_output_target_after_operation(exact, 1, 0x2010000000ull).status ==
                  OutputTargetAfterStatus::InvalidOperation,
              "exact selector distinguishes an invalid operation index");
        exact.operations[0].kind = SubmitOperationKind::Dispatch;
        CHECK(replay_output_target_after_operation(exact, 0, 0x2010000000ull).status ==
                  OutputTargetAfterStatus::NonDrawOperation,
              "exact selector distinguishes a non-draw operation");
        exact.operations[0].kind = SubmitOperationKind::Draw;
        exact.operations[0].realized = false;
        CHECK(replay_output_target_after_operation(exact, 0, 0x2010000000ull).status ==
                  OutputTargetAfterStatus::UnrealizedOperation,
              "exact selector distinguishes an unrealized draw operation");
        exact.operations[0].realized = true;
        exact.operations[0].source_index = 42;
        CHECK(replay_output_target_after_operation(exact, 0, 0x2010000000ull).status ==
                  OutputTargetAfterStatus::DrawUnavailable,
              "exact selector distinguishes missing realized draw state");
        exact.operations[0].source_index = 41;
        exact.items[0].ps.color_targets[0].write_mask = 0;
        CHECK(replay_output_target_after_operation(exact, 0, 0x2010000000ull).status ==
                  OutputTargetAfterStatus::NotWrittenByOperation,
              "a merely bound target with zero effective write mask is not an output");
        exact.items[0].ps.color_targets[0].write_mask = 0xf;
        exact.items[0].color_targets[0].width = 0;
        CHECK(replay_output_target_after_operation(exact, 0, 0x2010000000ull).status ==
                  OutputTargetAfterStatus::TargetIdentityUnavailable,
              "a written target with incomplete extent fails as unavailable identity");
    }

    // Fixed-function resolve is the deliberate exception to shader write-mask selection. Its raw
    // color1 is the destination while color0 is an input, even though both shader masks are zero.
    // Mutation check: removing the cb_resolve branch above must fail the first named check here.
    {
        GpuReplayFrame resolve;
        DrawItem operation;
        operation.draw_index = 77;
        operation.color0_base = 0x2011000000ull;
        operation.color0_width = 1280;
        operation.color0_height = 720;
        operation.color1_base = 0x2011800000ull;
        operation.color1_width = 1280;
        operation.color1_height = 720;
        operation.ps.color0_format = 97; // VK_FORMAT_R16G16B16A16_SFLOAT
        operation.ps.color1_format = 97;
        operation.ps.color_write_mask = 0;
        operation.ps.color1_write_mask = 0;
        operation.ps.cb_resolve = true;
        resolve.items = {operation};
        resolve.operations = {{SubmitOperationKind::Draw, 77, 600, true}};

        const auto destination =
            replay_output_target_after_operation(resolve, 0, 0x2011800000ull);
        CHECK(destination.status == OutputTargetAfterStatus::Selected &&
                  destination.target.slot == 1 && destination.target.format == 97 &&
                  destination.target.fixed_function_resolve,
              "MODE=RESOLVE selects raw color1 destination despite zero shader write mask");
        CHECK(replay_output_target_after_operation(resolve, 0, 0x2011000000ull).status ==
                  OutputTargetAfterStatus::NotWrittenByOperation,
              "MODE=RESOLVE never reports raw color0 source as an output");
    }

    // Direct callers and captures older than complete MRT arrays keep slot 0/1 in named aliases.
    {
        GpuReplayFrame legacy;
        DrawItem writer;
        writer.draw_index = 5;
        writer.color0_base = 0x12340000ull;
        writer.color0_width = 320;
        writer.color0_height = 180;
        writer.ps.color0_format = 37;
        writer.ps.color_write_mask = 0xb;
        legacy.items = {writer};
        legacy.operations = {{SubmitOperationKind::Draw, 5, 10, true}};
        const auto selected = replay_output_target_after_operation(legacy, 0, 0x12340000ull);
        CHECK(selected.status == OutputTargetAfterStatus::Selected &&
                  selected.target.width == 320 && selected.target.height == 180 &&
                  selected.target.format == 37 && selected.target.slot == 0,
              "exact selector preserves legacy named target aliases");
    }

    // Bundle operation ordinals are local to each submit. An earlier retained predecessor may have
    // an identical op/address and still must not receive the final submit's prefix selector.
    {
        GpuReplayFrame earlier;
        DrawItem old_writer;
        old_writer.draw_index = 11;
        old_writer.color_targets[0] = {0x2011800000ull, 64, 64};
        old_writer.ps.color_targets[0].format = 37;
        old_writer.ps.color_targets[0].write_mask = 0xf;
        earlier.items = {old_writer};
        earlier.operations = {{SubmitOperationKind::Draw, 11, 10, true}};

        GpuReplayFrame final = earlier;
        final.items[0].draw_index = 99;
        final.items[0].color_targets[0].width = 1280;
        final.items[0].color_targets[0].height = 720;
        final.operations[0].source_index = 99;

        const auto predecessor = replay_bundle_output_target_after_operation(
            earlier, 3, 4, 0, 0x2011800000ull);
        const auto selected_final = replay_bundle_output_target_after_operation(
            final, 4, 4, 0, 0x2011800000ull);
        CHECK(!predecessor.applies_to_submit,
              "bundle exact selector leaves retained predecessor submit unbounded");
        CHECK(selected_final.applies_to_submit &&
                  selected_final.selection.status == OutputTargetAfterStatus::Selected &&
                  selected_final.selection.target.draw_index == 99 &&
                  selected_final.selection.target.width == 1280 &&
                  selected_final.selection.target.height == 720,
              "bundle exact selector plans against selected final submit only");
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
