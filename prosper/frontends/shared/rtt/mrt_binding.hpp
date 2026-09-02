#pragma once
#include <cstdint>

#include "gpu/execute/gpu_execute.hpp"
#include "gpu/state/render_state.hpp"
#include "shared/rtt/mrt_extent.hpp"

// THE definition of an active colour binding.
//
// Two things ask it and they must not disagree: pass grouping, which decides how many attachments a
// render pass has and which surfaces they are, and same-pass feedback detection, which decides
// whether a sampled surface is one this pass is writing. A second, looser copy of this rule
// classified stale named state as a live binding, which denies the authoritative direct-GPU path and
// — when no CPU snapshot exists — degrades to guest bytes rather than to a slower correct source.
//
// One asymmetry this header does NOT itself resolve: for SLOT 0 the two read different
// representations. Grouping takes slot 0's identity from the named fields (see
// `mrt_same_color_pass`), because that is where live_renderer takes the address it renders to,
// while feedback stays array-first through `mrt_active_color`. They agree because every producer
// mirrors `color_targets[0]` from the named triple, not because anything here enforces it — so a
// producer that ever filled the two independently would break the agreement without this header
// changing.
//
// The rule: a slot is active when it has a base, a non-zero write mask, and a format the backend
// accepts. That last term is TOTAL in the current backend -- `backend_color_format` maps every
// unrecognised value, zero included, onto R8G8B8A8_UNORM, so it never rejects anything and a zero
// format means "use the fallback" rather than "undefined". It is kept as a parameter because it is
// the backend's decision to make, not this header's, and a backend that ever narrows it must narrow
// grouping and feedback together. For the ACTIVE-BINDING rule above, the
// named `color0_*` / `color1_*` fields are consulted ONLY when the array representation is absent,
// because `DrawItem` predates the complete array and capture versions through v33 carry the first
// two attachments in those fields. Falling back whenever the array mask merely reads zero is a
// different and wrong rule: a genuinely masked-off slot then inherits a stale named mask.
//
// That "ONLY" scopes to the active-binding rule and does not extend to slot 0's PASS IDENTITY,
// which is named-first for the reason given at `mrt_same_color_pass`.
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

// Can two consecutive draws share one backend colour pass?
//
// A target address is not a complete attachment identity.  Packed mip tails legitimately give two
// rendered levels the same guest address while their extents still differ.  Grouping only on
// address and format made GTA V's 64x32 and 32x16 R11G11B10F levels share
// one 64x32 Vulkan attachment.  The missing last level was then sampled as undefined max-float data
// by deferred lighting.  Unknown 0x0 extents remain non-conflicting, matching mrt_extent_conflicts.
template <typename FormatDefined, typename FormatAt>
bool mrt_same_color_pass(const prosper::gpu::DrawItem& first,
                         const prosper::gpu::DrawItem& candidate,
                         FormatDefined format_defined, FormatAt format_at) {
    // Slot 0 is compared on the NAMED triple whenever it names a surface, because that is the
    // surface the pass actually renders to: live_renderer takes `pass_bases[0]` from `color0_base`,
    // while every slot above 0 goes through the array. Reading slot 0's identity from the array
    // instead makes GROUPING and TARGET SELECTION consult different representations of one
    // attachment -- and two draws that render to DIFFERENT addresses then share a pass, so the
    // second draw's target is silently discarded and its pixels are never published.
    //
    // On captured and live draws the two representations are identical, so this is a no-op there.
    // For captures a divergence is not merely absent but INEXPRESSIBLE: the wire format carries
    // slots 2 and up only, and `restore_legacy_color_target_aliases` re-derives slots 0/1 from the
    // named triple on every load. For live draws the same mirror sits at the single success exit of
    // `realize_draw_item`. Neither is asserted anywhere, which is the one soft spot -- they are
    // conventions held by every producer rather than a checked invariant.
    //
    // The two differ only for a caller that builds a DrawItem directly and populates the named
    // aliases alone: exactly the shape `realize_draw_item`'s mirror exists to repair, and the shape
    // the render fixtures construct. Falling back to `mrt_color_binding` when `color0_base` is 0
    // keeps the previous answer for a draw that names no slot-0 surface at all.
    auto pass_binding = [](const prosper::gpu::DrawItem& draw, uint32_t slot) {
        if (slot == 0 && draw.color0_base)
            return prosper::gpu::DrawItem::ColorTargetBinding{
                draw.color0_base, draw.color0_width, draw.color0_height};
        return mrt_color_binding(draw, slot);
    };
    const uint32_t count = mrt_active_color_count(first, format_defined);
    if (mrt_active_color_count(candidate, format_defined) != count) return false;
    for (uint32_t slot = 0; slot < count; ++slot) {
        const auto a = pass_binding(first, slot);
        const auto b = pass_binding(candidate, slot);
        const uint64_t a_base = slot == 0 ? a.base
            : mrt_active_color(first, slot, format_defined);
        const uint64_t b_base = slot == 0 ? b.base
            : mrt_active_color(candidate, slot, format_defined);
        if (a_base != b_base || format_at(first, slot) != format_at(candidate, slot))
            return false;
        const bool active = slot == 0 || a_base || b_base;
        if (active && mrt_extent_conflicts(a.width, a.height, b.width, b.height))
            return false;
    }
    return true;
}

// Can two consecutive draws share one pass given their FIXED-FUNCTION RESOLVE identity?
//
// A CB_COLOR_CONTROL.MODE=RESOLVE draw is not an ordinary draw: live_renderer answers it with a
// straight copy of the already-rendered `color0_base` surface into `color1_base` and then ends the
// group, because a resolve is a copy and not a render. Two consequences follow, and the predicate
// has to carry both.
//
// 1. A resolve must never group with an ordinary draw. It shares `color0_base` with the scene it
//    resolves and reports `mrt_active_color(draw, 1) == 0` -- a fixed-function resolve exports
//    nothing, so its `color1_write_mask` is 0 and slot 1 contributes no attachment -- so on colour
//    identity alone it would merge with those scene draws, and the copy's `continue` would then
//    silently drop every ordinary draw grouped after it.
//
// 2. Two resolves must not group unless they resolve into the SAME destination. `cb_resolve` alone
//    is a boolean and cannot separate one resolve from another, while slot 1 -- the slot the
//    destination lives in -- is inactive for every resolve and so contributes no discriminating
//    term to `mrt_same_color_pass`. Consecutive resolves sharing a source and naming DIFFERENT
//    `color1_base` destinations therefore satisfied every grouping term, became one pass, and had
//    one copy performed from the group's first draw; every other destination was discarded with no
//    copy, no render and no diagnostic (#3025). The destination the copy acts with is the raw
//    `color1_base`, so that is the field the group must be keyed on -- the same principle #3023
//    established for slot 0: THE IDENTITY A PASS GROUPS ON MUST BE THE IDENTITY IT ACTS WITH.
//
// With the destination in the key, every member of a resolve group names one source and one
// destination, so the single copy the group performs is exactly right for all of them and no member
// needs its own.
//
// Deliberately keyed on the RAW named field rather than on `mrt_active_color(draw, 1, ...)`, for
// the same reason the copy reads it raw: a fixed-function resolve's slot 1 is never active, so the
// active-binding rule reports 0 for every resolve and would key the group on a constant.
inline bool mrt_same_resolve_pass(const prosper::gpu::DrawItem& first,
                                  const prosper::gpu::DrawItem& candidate) {
    if (first.ps.cb_resolve != candidate.ps.cb_resolve) return false;
    if (!first.ps.cb_resolve) return true;
    return first.color1_base == candidate.color1_base;
}

// Does this draw bind the sampled VIEW as any ACTIVE colour target? Address alone is insufficient:
// packed mip tails may give two levels the same guest base even though they are separate Vulkan
// images. Known, conflicting extents therefore prove that the sampled view is not the attachment.
// Unknown extents remain conservative and count an address match as feedback -- unlike pass
// grouping, this decision protects us from binding one image simultaneously for sampling and
// rendering, so absence of evidence cannot make the operation safe.
template <typename FormatDefined>
bool mrt_draw_binds_target_view(const prosper::gpu::DrawItem& draw, uint64_t addr,
                                uint32_t sampled_width, uint32_t sampled_height,
                                FormatDefined format_defined) {
    if (!addr) return false;
    for (uint32_t slot = 0; slot < prosper::gpu::kColorTargetCount; ++slot) {
        if (mrt_active_color(draw, slot, format_defined) != addr) continue;
        const auto binding = mrt_color_binding(draw, slot);
        if (mrt_extent_conflicts(binding.width, binding.height,
                                 sampled_width, sampled_height))
            continue;
        return true;
    }
    return false;
}

// Address-only compatibility wrapper. With an unknown sampled extent it deliberately keeps the
// historical conservative behaviour.
template <typename FormatDefined>
bool mrt_draw_binds_target(const prosper::gpu::DrawItem& draw, uint64_t addr,
                           FormatDefined format_defined) {
    return mrt_draw_binds_target_view(draw, addr, 0u, 0u, format_defined);
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

// May the retained GPU image serve this sample without materializing guest/CPU bytes? An ordinary
// non-feedback sample borrows the target directly. When `feedback_copy_supported` is true, an exact
// attachment collision is also admitted because the backend snapshots the prior attachment version
// into a distinct sampled image before beginning the render pass. The default preserves the
// conservative contract for callers that do not implement that GPU copy.
template <typename FormatDefined>
bool mrt_direct_serves(const prosper::gpu::DrawItem& draw, uint64_t sampled,
                       uint32_t sampled_width, uint32_t sampled_height,
                       bool is_storage_image, uint32_t img_dim, bool extent_compatible,
                       bool has_persistent_target, FormatDefined format_defined,
                       bool feedback_copy_supported = false) {
    const bool feedback = mrt_draw_binds_target_view(
        draw, sampled, sampled_width, sampled_height, format_defined);
    return !is_storage_image && img_dim == 1u && extent_compatible && has_persistent_target &&
           (!feedback || feedback_copy_supported);
}

// May the uniform-colour fast path serve this sample? `preconditions` folds the caller's own
// non-feedback terms (not a storage image, a plain 2D view, not in a mip tail, a uniform cache entry
// with a usable extent) so this seam owns exactly the feedback gate and nothing it cannot see.
template <typename FormatDefined>
bool mrt_uniform_live_serves(const prosper::gpu::DrawItem& draw, uint64_t sampled,
                             uint32_t sampled_width, uint32_t sampled_height,
                             bool preconditions, FormatDefined format_defined) {
    return preconditions &&
           !mrt_draw_binds_target_view(draw, sampled, sampled_width, sampled_height,
                                       format_defined);
}

}  // namespace prosper::frontend
