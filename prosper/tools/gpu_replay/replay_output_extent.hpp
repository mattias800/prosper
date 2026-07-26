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
};

inline OutputExtent replay_output_extent(const GpuReplayFrame& replay, OutputExtentMode mode,
                                         size_t output_bytes,
                                         size_t operation_limit = SIZE_MAX) {
    const OutputExtent capture_extent{replay.metadata.width, replay.metadata.height};
    OutputExtent selected_extent = capture_extent;
    auto select_draw = [&](const DrawItem& draw) {
        if (draw.color0_width && draw.color0_height)
            selected_extent = {draw.color0_width, draw.color0_height};
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
    if (capture_bytes == output_bytes) return capture_extent;
    return selected_bytes == output_bytes ? selected_extent : capture_extent;
}

} // namespace prosper::gpu::replay_tool
