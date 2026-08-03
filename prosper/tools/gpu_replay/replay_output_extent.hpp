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

// Exact render-target identity selected for post-operation inspection. Unlike OutputTarget's
// extent-only search, this plan is tied to one semantic operation and retains the VkFormat-valued
// raw pipeline format plus the attachment slot which proved the write. The renderer readback must
// compare every field; looking the format up again by address would allow a later alias/version to
// masquerade as the requested surface.
struct OutputTargetAfterOperation {
    size_t operation_index = SIZE_MAX;
    uint64_t guest_addr = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t format = 0;
    uint64_t draw_index = 0;
    uint32_t slot = UINT32_MAX;
    bool fixed_function_resolve = false;

    explicit operator bool() const {
        return operation_index != SIZE_MAX && guest_addr && width && height && format &&
               draw_index != UINT64_MAX && slot < kColorTargetCount;
    }
};

enum class OutputTargetAfterStatus {
    Selected,
    InvalidOperation,
    NonDrawOperation,
    UnrealizedOperation,
    DrawUnavailable,
    NotWrittenByOperation,
    TargetIdentityUnavailable,
};

struct OutputTargetAfterSelection {
    OutputTargetAfterStatus status = OutputTargetAfterStatus::InvalidOperation;
    OutputTargetAfterOperation target;
};

struct BundleOutputTargetAfterSelection {
    bool applies_to_submit = false;
    OutputTargetAfterSelection selection;
};

inline DrawItem::ColorTargetBinding replay_color_binding(const DrawItem& draw, uint32_t slot) {
    auto binding = draw.color_targets[slot];
    // Capture versions through v33 and direct synthetic callers retain MRT0/MRT1 in named aliases.
    if (!binding.base && !binding.width && !binding.height && slot == 0)
        binding = {draw.color0_base, draw.color0_width, draw.color0_height};
    else if (!binding.base && !binding.width && !binding.height && slot == 1)
        binding = {draw.color1_base, draw.color1_width, draw.color1_height};
    return binding;
}

inline uint32_t replay_color_format(const DrawItem& draw, uint32_t slot) {
    uint32_t format = draw.ps.color_targets[slot].format;
    if (!format && slot == 0) format = draw.ps.color0_format;
    else if (!format && slot == 1) format = draw.ps.color1_format;
    return format;
}

inline uint32_t replay_color_write_mask(const DrawItem& draw, uint32_t slot) {
    const auto& target = draw.ps.color_targets[slot];
    uint32_t write_mask = target.write_mask;
    // Mirror the live renderer's legacy-alias discriminator. A complete-MRT binding with a zero
    // mask is intentionally not rescued by a named legacy field.
    if (slot == 0 && !target.format && !draw.color_targets[slot].base)
        write_mask = draw.ps.color_write_mask;
    else if (slot == 1 && !target.format && !draw.color_targets[slot].base)
        write_mask = draw.ps.color1_write_mask;
    return write_mask;
}

// Select ADDR only when semantic operation OP actually writes it. An ordinary draw needs a nonzero
// effective target write mask. CB_COLOR_CONTROL.MODE=RESOLVE is different fixed-function work: its
// raw color1 binding is the destination even though the pixel shader exports nothing and therefore
// has a zero shader/write mask; raw color0 is only the source and must never be reported as output.
inline OutputTargetAfterSelection replay_output_target_after_operation(
    const GpuReplayFrame& replay, size_t operation_index, uint64_t guest_addr) {
    if (operation_index >= replay.operations.size())
        return {OutputTargetAfterStatus::InvalidOperation, {}};
    const auto& operation = replay.operations[operation_index];
    if (operation.kind != SubmitOperationKind::Draw)
        return {OutputTargetAfterStatus::NonDrawOperation, {}};
    if (!operation.realized)
        return {OutputTargetAfterStatus::UnrealizedOperation, {}};
    const auto found = std::find_if(replay.items.begin(), replay.items.end(),
        [&](const DrawItem& draw) { return draw.draw_index == operation.source_index; });
    if (found == replay.items.end())
        return {OutputTargetAfterStatus::DrawUnavailable, {}};

    const DrawItem& draw = *found;
    uint32_t selected_slot = UINT32_MAX;
    if (draw.ps.cb_resolve) {
        // Fixed-function resolve has exactly one color output: raw color1. Do not consider color0,
        // and do not consult color1_write_mask (it is expected to be zero for this operation).
        if (replay_color_binding(draw, 1).base == guest_addr) selected_slot = 1;
    } else {
        for (uint32_t slot = 0; slot < kColorTargetCount; ++slot) {
            if (replay_color_binding(draw, slot).base == guest_addr &&
                replay_color_write_mask(draw, slot)) {
                selected_slot = slot;
                break;
            }
        }
    }
    if (selected_slot == UINT32_MAX)
        return {OutputTargetAfterStatus::NotWrittenByOperation, {}};

    const auto binding = replay_color_binding(draw, selected_slot);
    const uint32_t format = replay_color_format(draw, selected_slot);
    if (!binding.base || !binding.width || !binding.height || !format)
        return {OutputTargetAfterStatus::TargetIdentityUnavailable, {}};
    OutputTargetAfterOperation target;
    target.operation_index = operation_index;
    target.guest_addr = binding.base;
    target.width = binding.width;
    target.height = binding.height;
    target.format = format;
    target.draw_index = draw.draw_index;
    target.slot = selected_slot;
    target.fixed_function_resolve = draw.ps.cb_resolve;
    return {OutputTargetAfterStatus::Selected, target};
}

// Bundle operation ordinals are submit-local. Apply an exact selector only to the final submit left
// after the caller's tail/through-submit selection; an earlier retained predecessor must execute in
// full even if it happens to contain the same operation/address pair.
inline BundleOutputTargetAfterSelection replay_bundle_output_target_after_operation(
    const GpuReplayFrame& replay, size_t current_submit_index, size_t selected_final_submit_index,
    size_t operation_index, uint64_t guest_addr) {
    if (current_submit_index != selected_final_submit_index) return {};
    return {true, replay_output_target_after_operation(replay, operation_index, guest_addr)};
}

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
