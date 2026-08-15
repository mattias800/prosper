#pragma once
#include <cstdint>

#include "../../src/gpu/gpu_execute.hpp"
#include "../../src/gpu/render_state.hpp"
#include "mrt_extent.hpp"

// THE definition of an active colour binding.
//
// Two things ask it and they must not disagree: pass grouping, which decides how many attachments a
// render pass has and which surfaces they are, and same-pass feedback detection, which decides
// whether a sampled surface is one this pass is writing. A second, looser copy of this rule
// classified stale named state as a live binding, which denies the authoritative direct-GPU path and
// — when no CPU snapshot exists — degrades to guest bytes rather than to a slower correct source.
//
// The rule: a slot is active when it has a base, a defined format, and a non-zero write mask. The
// named `color0_*` / `color1_*` fields are consulted ONLY when the array representation is absent,
// because `DrawItem` predates the complete array and capture versions through v33 carry the first
// two attachments in those fields. Falling back whenever the array mask merely reads zero is a
// different and wrong rule: a genuinely masked-off slot then inherits a stale named mask.
namespace prosper::frontend {

inline prosper::gpu::DrawItem::ColorTargetBinding mrt_color_binding(
    const prosper::gpu::DrawItem& draw, uint32_t slot) {
    auto binding = draw.color_targets[slot];
    if (!binding.base && !binding.width && !binding.height && slot == 0)
        binding = prosper::gpu::DrawItem::ColorTargetBinding{
            draw.color0_base, draw.color0_width, draw.color0_height};
    else if (!binding.base && !binding.width && !binding.height && slot == 1)
        binding = prosper::gpu::DrawItem::ColorTargetBinding{
            draw.color1_base, draw.color1_width, draw.color1_height};
    return binding;
}

// Raw guest format for a slot, with the same named-field fallback. Left as the raw value so callers
// that need a backend VkFormat can map it themselves without this header depending on the backend.
inline uint32_t mrt_raw_format(const prosper::gpu::DrawItem& draw, uint32_t slot) {
    uint32_t raw = draw.ps.color_targets[slot].format;
    if (slot == 0 && !raw) raw = draw.ps.color0_format;
    if (slot == 1 && !raw) raw = draw.ps.color1_format;
    return raw;
}

// The write mask that governs this slot, with the named fallback taken only when the array
// representation is ABSENT -- matching pass grouping exactly.
inline uint32_t mrt_write_mask(const prosper::gpu::DrawItem& draw, uint32_t slot) {
    const auto& target = draw.ps.color_targets[slot];
    if (slot == 0 && !target.format && !draw.color_targets[slot].base)
        return draw.ps.color_write_mask;
    if (slot == 1 && !target.format && !draw.color_targets[slot].base)
        return draw.ps.color1_write_mask;
    return target.write_mask;
}

// The base this slot is actively writing, or 0. `format_defined` decides whether a raw guest format
// counts as defined; callers pass their own mapping so this header stays backend-free.
template <typename FormatDefined>
uint64_t mrt_active_color(const prosper::gpu::DrawItem& draw, uint32_t slot,
                          FormatDefined format_defined) {
    const auto binding = mrt_color_binding(draw, slot);
    if (!binding.base) return 0;
    if (!mrt_write_mask(draw, slot)) return 0;
    if (!format_defined(mrt_raw_format(draw, slot))) return 0;
    return binding.base;
}

// The active attachment prefix: one past the highest active slot, never less than 1.
template <typename FormatDefined>
uint32_t mrt_active_color_count(const prosper::gpu::DrawItem& draw, FormatDefined format_defined) {
    uint32_t count = 1;
    for (uint32_t slot = 1; slot < prosper::gpu::kColorTargetCount; ++slot)
        if (mrt_active_color(draw, slot, format_defined)) count = slot + 1;
    return count;
}

// Does this draw bind `addr` as any ACTIVE colour target? The feedback question, answered through
// the same rule pass grouping uses rather than a second interpretation of it.
template <typename FormatDefined>
bool mrt_draw_binds_target(const prosper::gpu::DrawItem& draw, uint64_t addr,
                           FormatDefined format_defined) {
    if (!addr) return false;
    uint64_t bases[prosper::gpu::kColorTargetCount]{};
    bool active[prosper::gpu::kColorTargetCount]{};
    for (uint32_t slot = 0; slot < prosper::gpu::kColorTargetCount; ++slot) {
        bases[slot] = mrt_active_color(draw, slot, format_defined);
        active[slot] = bases[slot] != 0u;
    }
    return mrt_target_feedback(bases, active, prosper::gpu::kColorTargetCount, addr);
}

}  // namespace prosper::frontend
