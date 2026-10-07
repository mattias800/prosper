// Same-TU fixture: how a logical draw batch is cut into physical render-pass segments
// (depth-feedback transitions, merged-NGG draws) and the contract each segment renders
// under. Moved verbatim out of render_runner.h, which includes it at the same point, to keep
// that file under its size ratchet.
#pragma once

// clang-format off: moved verbatim; reformatting would hide the move from review.
// A Vulkan render pass has one depth layout for every draw. If an earlier draw writes a persistent
// guest depth plane and a later draw samples that same plane, keeping both in one pass makes the
// attachment writable and the sampled-depth bridge must reject the alias. Split exactly at those
// write/sample transitions (in either direction): queue order plus the persistent color/DS caches
// preserve guest draw order, while each resulting pass has a legal, unambiguous depth layout.
inline size_t depth_feedback_split_index(std::span<const BackendDraw> draws,
                                         uint32_t W, uint32_t H) {
    std::unordered_set<uint64_t> sampled_depth;
    std::unordered_set<uint64_t> written_depth;
    for (size_t i = 0; i < draws.size(); ++i) {
        const BackendDraw& draw = draws[i];
        const bool samples_prior_write = std::any_of(
            draw.R.begin(), draw.R.end(), [&](const FrameResource& resource) {
                return resource.persistent_depth_target_id && resource.img_dim == 1u &&
                    resource.tw == W && resource.th == H &&
                    written_depth.contains(resource.persistent_depth_target_id);
            });
        const bool writes_depth = draw.ps && persistent_ds_pass_may_write_depth(
            draw.ps->depth_clear_enable, draw.ps->depth_test_enable,
            draw.ps->depth_write_enable, draw.ps->depth_compare_op);
        const bool writes_prior_sample = writes_depth &&
            ((draw.ps->depth_read_base && sampled_depth.contains(draw.ps->depth_read_base)) ||
             (draw.ps->depth_write_base && sampled_depth.contains(draw.ps->depth_write_base)));
        if (i && (samples_prior_write || writes_prior_sample)) return i;

        for (const FrameResource& resource : draw.R)
            if (resource.persistent_depth_target_id && resource.img_dim == 1u &&
                resource.tw == W && resource.th == H)
                sampled_depth.insert(resource.persistent_depth_target_id);
        if (writes_depth) {
            if (draw.ps->depth_read_base) written_depth.insert(draw.ps->depth_read_base);
            if (draw.ps->depth_write_base) written_depth.insert(draw.ps->depth_write_base);
        }
    }
    return draws.size();
}

// Where the first segment ends: a depth feedback transition (persistent depth only) or a merged-NGG
// draw, which is always a segment of its own (#3135 P4, ngg_subgroup_gpu.h).
inline size_t backend_segment_split_index(std::span<const BackendDraw> draws, uint32_t W,
                                          uint32_t H, bool persist_depth_stencil) {
    const size_t ngg = ngg_segment_split_index(draws);
    return persist_depth_stencil ? std::min(ngg, depth_feedback_split_index(draws, W, H)) : ngg;
}

// The contract one SEGMENT of a depth-feedback-split pass renders under.
//
// Extracted so the segment decisions are testable at the site that makes them. Every field here was
// once derived inline, and two of them were wrong in ways no end-to-end assertion caught: non-final
// segments were handed `mrt_outputs == nullptr`, so their colour_count collapsed to one attachment
// and every MRT1..7 export in them was discarded; and later segments never asked to LOAD the higher
// slots, so a fresh slot cleared at the second segment.
//
// The invariant the tests pin: the SHAPE is identical across segments and only the pixel
// destination and the load/readback flags differ.
struct SplitSegmentContract {
    BackendColorTarget target{};
    uint32_t color_count = 1;
    bool has_target = false;
};
inline SplitSegmentContract split_segment_contract(
    const BackendColorTarget* whole, const BackendMrtOutputs* whole_mrt, bool first, bool final,
    const std::array<const uint8_t*, prosper::gpu::kColorTargetCount>& carried_slots = {}) {
    SplitSegmentContract out;
    // The attachment shape never varies by segment. A pass is five-MRT for all of its segments or
    // none of them; splitting is a synchronisation boundary, not a change of render target.
    out.color_count = whole_mrt ? whole_mrt->color_count : 1u;
    // A caller that named no persistent identities still needs a target when higher slots have to
    // be carried between segments: the seed and readback flags live on it, and they are the only
    // channel by which a non-persistent MRT2+ survives a physical pass boundary. A synthesised
    // target is behaviourally identical to no target -- every identity is zero, so `persistent_color`
    // and its siblings stay false and no cached image is consulted -- except that it can carry.
    // > 1, not > 2: slot 1 needs carrying too when the caller takes it through BackendMrtOutputs.
    // The threshold was 2 and a two-attachment split therefore synthesised no carrier at all, so
    // MRT1 was cleared by the next segment even with every other part of the carry correct.
    const bool carries_slots = out.color_count > 1u;
    if (!whole && !carries_slots) return out;
    if (whole) out.target = *whole;
    out.has_target = true;
    if (!first) {
        // Every later segment LOADS everything, or it erases what its predecessors drew.
        out.target.load_existing = true;
        out.target.load_existing1 = true;
        out.target.load_existing_slots.fill(true);
        // LOADing is only sufficient when the slot's image actually SURVIVES, and a non-zero
        // identity does not establish that: persistence also requires
        // persistent_color_targets_enabled (an env switch) and a successful cache/budget
        // allocation, and the slot-creation path explicitly falls back to a transient image while
        // leaving the identity non-zero. Seeding whatever the previous segment produced is correct
        // in every one of those cases and merely redundant when the image did survive, so the
        // decision is made conservatively rather than from an identity that cannot answer it.
        for (uint32_t slot = 2; slot < prosper::gpu::kColorTargetCount; ++slot)
            out.target.seed_slots[slot] = carried_slots[slot];
        // Slot 1 travels the same way when the caller takes it through BackendMrtOutputs rather
        // than out_rgba1 -- see the carry below.
        if (carried_slots[1]) out.target.seed_rgba1_slot = carried_slots[1];
    }
    if (!final) {
        // A non-final segment's slot-0 pixels are discarded, so do not copy them out.
        out.target.readback = false;
        out.target.readback1 = false;
        out.target.readback_slots.fill(false);
        // ...but every slot this segment may have to CARRY must be copied out, or there is nothing
        // to seed the next segment with. That is slot 1 (which the live renderer takes through
        // BackendMrtOutputs) and every active slot 2+, unconditionally: whether a slot's image
        // actually survives depends on persistent_color_targets_enabled and on a cache/budget
        // allocation that can fall back to a transient image while leaving the identity non-zero,
        // and none of that is visible from here.
        //
        // Decided HERE rather than by the caller amending the result. It was written as an
        // override immediately after this function returned, which left the helper's own test
        // asserting `non-final segments read back no slot` -- the exact opposite of the effective
        // contract, in the test whose job is to document it.
        for (uint32_t slot = 2; slot < out.color_count; ++slot)
            out.target.readback_slots[slot] = true;
        if (out.color_count > 1) out.target.readback1 = true;
    } else if (!whole) {
        // A synthesised carrier target on the final segment represents a caller with whole == nullptr.
        // It must read back the requested outputs just as a whole == nullptr non-split pass does.
        out.target.readback = true;
        out.target.readback1 = out.color_count > 1;
        for (uint32_t slot = 2; slot < out.color_count; ++slot)
            out.target.readback_slots[slot] = true;
    }
    return out;
}
// clang-format on
