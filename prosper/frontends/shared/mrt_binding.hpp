#pragma once
#include <cstdint>

#include "gpu/execute/gpu_execute.hpp"
#include "gpu/state/render_state.hpp"
#include "mrt_extent.hpp"

// THE definition of an active colour binding.
//
// Two things ask it and they must not disagree: pass grouping, which decides how many attachments a
// render pass has and which surfaces they are, and same-pass feedback detection, which decides
// whether a sampled surface is one this pass is writing. A second, looser copy of this rule
// classified stale named state as a live binding, which denies the authoritative direct-GPU path and
// — when no CPU snapshot exists — degrades to guest bytes rather than to a slower correct source.
//
// The rule: a slot is active when it has a base, a non-zero write mask, and a format the backend
// accepts. That last term is TOTAL in the current backend -- `backend_color_format` maps every
// unrecognised value, zero included, onto R8G8B8A8_UNORM, so it never rejects anything and a zero
// format means "use the fallback" rather than "undefined". It is kept as a parameter because it is
// the backend's decision to make, not this header's, and a backend that ever narrows it must narrow
// grouping and feedback together. The
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
// counts as accepted; callers pass their own mapping so this header stays backend-free. Note it is
// total in today's backend -- see the file comment; do not read this parameter as evidence that a
// zero format is rejected anywhere.
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

// The two materialization decisions that key on feedback, as seams that OWN their gate.
//
// They exist because the gate is the interesting part and a helper called beside it is not: with the
// comparison written inline at each call site, reverting one to `sampled != draw.color0_base` left
// every test green, since the tests exercised the helper rather than the decision.
//
// The two are coupled, which is why they must agree. `mrt_direct_serves` deciding TRUE suppresses
// the lazy CPU materialisation for that resource; if the later bind then refuses the direct image --
// which it must, when the sample really is one of this pass's targets -- there is no snapshot left
// and the resource degrades to guest bytes. So a feedback collision has to be seen by BOTH, and it
// used to be seen by neither above slot 1.

// May the retained GPU image serve this sample directly?
template <typename FormatDefined>
bool mrt_direct_serves(const prosper::gpu::DrawItem& draw, uint64_t sampled,
                       bool is_storage_image, uint32_t img_dim, bool extent_compatible,
                       bool has_persistent_target, FormatDefined format_defined) {
    return !is_storage_image && img_dim == 1u && extent_compatible && has_persistent_target &&
           !mrt_draw_binds_target(draw, sampled, format_defined);
}

// May the uniform-colour fast path serve this sample? `preconditions` folds the caller's own
// non-feedback terms (not a storage image, a plain 2D view, not in a mip tail, a uniform cache entry
// with a usable extent) so this seam owns exactly the feedback gate and nothing it cannot see.
template <typename FormatDefined>
bool mrt_uniform_live_serves(const prosper::gpu::DrawItem& draw, uint64_t sampled,
                             bool preconditions, FormatDefined format_defined) {
    return preconditions && !mrt_draw_binds_target(draw, sampled, format_defined);
}

}  // namespace prosper::frontend
