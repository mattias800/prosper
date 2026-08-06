// guest_scanout_present.hpp — when prosper may publish the guest's OWN flipped display buffer (#1968).
#pragma once

#include <cstddef>

namespace prosper::frontend {

// Every present source the live renderer normally chooses among is something PROSPER rendered — a
// pass, or a persistent target it owns. A title whose final composite into the display buffer is not
// a draw has no such source at all, and no amount of selection among passes finds one: Sonic
// Frontiers (PPSA03831) runs ~3,700 passes across a boot without a single one whose target is a
// registered VideoOut buffer, and prosper's render-target map therefore has no entry at the flipped
// address on any submit. What it does have is the frame itself, sitting in the guest buffer it
// flips — de-swizzled, an exact SEGA logo and an exact intro shot. VideoOut's contract is "display
// THIS buffer", so when prosper has written nothing there, the guest's own bytes are not a fallback
// approximation: they are the only description of the frame that exists, and publishing them is the
// literal meaning of the flip.
//
// This header holds the decision, not the mechanism, for the reason present_extent.hpp exists next
// door: the branch it replaced was 81 lines of renderer code with no test at all, and the case that
// reverted it (#2044) is a pure predicate question — "does an untouched buffer decline?" — that
// should cost a unit test rather than a 180-second run of another title.

// Why the flipped guest buffer was or was not published for this submit. Every value except
// `Publish` is a reason the branch declined, and they are distinct so a diagnostic can say WHICH —
// a negative control that reports only "it did not fire" cannot tell "correctly declined" from
// "never reached".
//
// **Only the stage-2 reasons are currently observable at runtime.** The renderer's `[rtt] GUEST
// SCANOUT` report sits inside the stage-1 `Publish` branch, so a stage-1 skip prints nothing and the
// four names below it (`SkipPublishedGpu`, `SkipRendererSource`, `SkipNoPresentContract`,
// `SkipScaledPresent`) can never appear in a log as things stand. That is why Bendy's (PPSA27616)
// zero-line control is honestly reported as **"never reached"** rather than as a decline: the
// instrument cannot tell those apart today, and the reused-memory case it was meant to cover is
// asserted by unit test instead. Giving stage 1 its own low-budget report would close that gap.
enum class GuestScanoutDecision {
    Publish,
    SkipPublishedGpu,        // prosper already published a GPU frame for this guest flip
    SkipRendererSource,      // prosper has a present source of its own; it always wins
    SkipNoPresentContract,   // no present extent applies (gpu_replay / render_submit_items)
    SkipScaledPresent,       // PROSPER_RENDER_SCALE: the display extent is not the present extent
    SkipUnreadable,          // no flipped buffer, or it could not be read
    SkipWrongExtent,         // the guest buffer is not the size the caller will publish
    SkipRendererOwnsTarget,  // prosper owns a render target at this address: prosper is the writer
    SkipNotAuthored,         // the guest has not written this buffer since it registered it
};

constexpr const char* guest_scanout_decision_name(GuestScanoutDecision decision) {
    switch (decision) {
        case GuestScanoutDecision::Publish:               return "publish";
        case GuestScanoutDecision::SkipPublishedGpu:      return "published-gpu";
        case GuestScanoutDecision::SkipRendererSource:    return "renderer-source";
        case GuestScanoutDecision::SkipNoPresentContract: return "no-present-contract";
        case GuestScanoutDecision::SkipScaledPresent:     return "scaled-present";
        case GuestScanoutDecision::SkipUnreadable:        return "unreadable";
        case GuestScanoutDecision::SkipWrongExtent:       return "wrong-extent";
        case GuestScanoutDecision::SkipRendererOwnsTarget:return "renderer-owns-target";
        case GuestScanoutDecision::SkipNotAuthored:       return "not-authored";
    }
    return "unknown";
}

// Stage 1 — everything answerable BEFORE paying for the read. A 4K de-swizzle is not free and this
// callback runs for every span of every submit (eleven per frame on Frontiers), so the cheap tests
// come first; `Publish` here means only "the read is warranted", not "publish it".
//
// The display-extent test belongs here rather than after the read: under PROSPER_RENDER_SCALE the
// guest buffer can never match the reduced present extent, and most snapshot guards run scaled —
// checking afterwards would make them pay a full-resolution de-swizzle per flip only to discard it.
constexpr GuestScanoutDecision guest_scanout_read_warranted(bool published_gpu,
                                                            bool renderer_scanout,
                                                            bool have_selected_pixels,
                                                            size_t present_extent_bytes,
                                                            size_t display_bytes) {
    // prosper's own image for the flipped address always wins when it has one: it is unscaled and
    // needs no CPU round trip. `have_selected_pixels` covers the rest of the renderer's selection,
    // which has already refused every wrong-extent candidate — so a survivor is a frame prosper can
    // publish and keeps its priority. Without this test the guest buffer would OVERRIDE a
    // qualifying candidate, which is a cross-title present-priority change and not what this is for.
    if (published_gpu) return GuestScanoutDecision::SkipPublishedGpu;
    if (renderer_scanout || have_selected_pixels) return GuestScanoutDecision::SkipRendererSource;
    if (present_extent_bytes == 0) return GuestScanoutDecision::SkipNoPresentContract;
    if (display_bytes != present_extent_bytes) return GuestScanoutDecision::SkipScaledPresent;
    return GuestScanoutDecision::Publish;
}

// Stage 2 — the tests that need the read to have happened. `linear_bytes` is 0 when the read failed.
constexpr GuestScanoutDecision guest_scanout_publishable(size_t linear_bytes,
                                                         size_t present_extent_bytes,
                                                         bool have_address,
                                                         bool renderer_owns_target,
                                                         bool guest_authored) {
    if (!linear_bytes || !have_address) return GuestScanoutDecision::SkipUnreadable;
    if (linear_bytes != present_extent_bytes) return GuestScanoutDecision::SkipWrongExtent;
    // A render-target entry that merely failed to materialize (wrong extent mid-resize, a readback
    // that did not come back) still means prosper IS the writer and is between states; overriding
    // it with guest memory would publish a surface prosper is midway through producing. Testing the
    // address the READER returned, rather than the one looked up before it, also closes the gap
    // where the guest flips between the two.
    if (renderer_owns_target) return GuestScanoutDecision::SkipRendererOwnsTarget;
    // The load-bearing one. "Not all bytes are zero" is NOT this test: it holds only for a freshly
    // zeroed allocation, and a title re-registering over reused memory passes it while prosper's
    // target map misses precisely BECAUSE the address is new — publishing stale garbage in place of
    // the retained good frame. See videoout_read_front_linear for how authorship is established.
    // Note what this is not: it is "was anything written", not "is it not black". A frame the guest
    // deliberately clears to opaque black publishes as black, which is what the console shows.
    if (!guest_authored) return GuestScanoutDecision::SkipNotAuthored;
    return GuestScanoutDecision::Publish;
}

} // namespace prosper::frontend
