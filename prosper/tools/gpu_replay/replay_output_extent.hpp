#pragma once

#include "gpu/gpu_capture.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace prosper::gpu::replay_tool {

enum class OutputExtentMode {
    Capture,
    SelectedDraws,
    OrderedPrefix,
};

struct OutputExtent {
    uint32_t width = 0;
    uint32_t height = 0;
    // Guest base address of the color target the reported extent came from, and the draw that wrote
    // it. Zero when the extent came from the capture's presentation metadata rather than a draw.
    //
    // Reporting WHICH surface was selected matters because a prefix replay renders the last executed
    // draw target, so two adjacent `--through-operation` cutoffs can display two completely different
    // buffers. Without the address, that reads as one surface changing between the cutoffs -- which is
    // how #1486's "exact corruption boundary" was derived from op116 (1920x1080, a post-process input)
    // versus op117 (3840x2160, the graded scene). They were never the same surface. Printing the
    // address makes that class of false boundary self-evident instead of load-bearing.
    uint64_t target_addr = 0;
    uint64_t target_draw_index = 0;
};

struct OutputTarget {
    uint64_t guest_addr = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t draw_index = 0;

    explicit operator bool() const { return guest_addr && width && height; }
};

// Return the last realized color target with the requested native extent in an ordered prefix.
// A frame-ending submit may finish with an auxiliary 1x1 draw after producing its 4K scanout, so
// byte-count inference alone cannot identify the displayed surface. Keep the choice explicit and
// address-backed: callers can read this exact renderer-owned target after replaying the full prefix.
inline OutputTarget replay_last_target_matching_extent(const GpuReplayFrame& replay,
                                                        uint32_t width, uint32_t height,
                                                        size_t operation_limit = SIZE_MAX) {
    OutputTarget selected;
    const size_t count = std::min(operation_limit, replay.operations.size());
    for (size_t operation_index = 0; operation_index < count; ++operation_index) {
        const auto& operation = replay.operations[operation_index];
        if (!operation.realized || operation.kind != SubmitOperationKind::Draw) continue;
        const auto draw = std::find_if(replay.items.begin(), replay.items.end(),
            [&](const auto& item) { return item.draw_index == operation.source_index; });
        if (draw == replay.items.end() || !draw->color0_base ||
            draw->color0_width != width || draw->color0_height != height)
            continue;
        selected = {draw->color0_base, width, height, draw->draw_index};
    }
    return selected;
}

inline OutputExtent replay_output_extent(const GpuReplayFrame& replay, OutputExtentMode mode,
                                         size_t output_bytes,
                                         size_t operation_limit = SIZE_MAX) {
    const OutputExtent capture_extent{replay.metadata.width, replay.metadata.height};
    OutputExtent selected_extent = capture_extent;
    auto select_draw = [&](const DrawItem& draw) {
        if (draw.color0_width && draw.color0_height)
            selected_extent = {draw.color0_width, draw.color0_height,
                               draw.color0_base, draw.draw_index};
    };

    if (mode == OutputExtentMode::SelectedDraws) {
        if (!replay.items.empty()) select_draw(replay.items.back());
    } else {
        const size_t count = mode == OutputExtentMode::OrderedPrefix
            ? std::min(operation_limit, replay.operations.size())
            : replay.operations.size();
        for (size_t operation_index = 0; operation_index < count; ++operation_index) {
            const auto& operation = replay.operations[operation_index];
            if (!operation.realized || operation.kind != SubmitOperationKind::Draw) continue;
            const auto draw = std::find_if(replay.items.begin(), replay.items.end(),
                [&](const auto& item) { return item.draw_index == operation.source_index; });
            if (draw != replay.items.end()) select_draw(*draw);
        }
    }

    // Some captures retain a presentation/capture extent even though this submit's last realized
    // operation writes an intermediate target. Prefer the capture dimensions whenever their exact
    // RGBA byte count matches. Otherwise switch only when the last realized target's byte count
    // proves its dimensions; this cannot silently scale or reinterpret a full-resolution frame.
    const uint64_t capture_bytes = static_cast<uint64_t>(capture_extent.width) *
                                   capture_extent.height * 4;
    const uint64_t selected_bytes = static_cast<uint64_t>(selected_extent.width) *
                                    selected_extent.height * 4;
    // Attach the target identity only when the returned extent is the one that draw actually
    // produced. Reporting an address next to dimensions it does not describe would be worse than
    // reporting none, since the whole point is to identify the surface on screen.
    auto with_target = [&](OutputExtent chosen) {
        if (chosen.width == selected_extent.width && chosen.height == selected_extent.height) {
            chosen.target_addr = selected_extent.target_addr;
            chosen.target_draw_index = selected_extent.target_draw_index;
        }
        return chosen;
    };
    if (capture_bytes == output_bytes) return with_target(capture_extent);
    return with_target(selected_bytes == output_bytes ? selected_extent : capture_extent);
}

} // namespace prosper::gpu::replay_tool
