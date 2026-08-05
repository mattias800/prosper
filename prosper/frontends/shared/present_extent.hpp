// present_extent.hpp — the extent contract between the live renderer and its caller (#1986).
#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper::frontend {

// `execute_ordered_and_present` publishes a rendered frame only when its byte count equals the
// requested width*height*4; anything else is discarded and nothing reaches the screen. The renderer
// therefore has an extent contract with its caller, and until #1986 it had none: the present source
// was chosen by target IDENTITY alone (the flipped front buffer > any registered scanout > the last
// non-empty RGBA8 pass of the submit, whatever that pass happened to be). Sonic Frontiers (#1968)
// renders internally at 1920x1080 and presents at 3840x2160, so post-intro submits in which no pass
// targets a scanout returned an internal 1920x1080 (or 1024x1024, or 3840x3072) pass as "the frame"
// and every one of them was dropped — 13,028 submits with no publish and no explanation.
//
// The contract is expressed in BYTES rather than an extent pair because bytes are what the caller
// actually compares, and because that is also what the retained-frame net below holds.
//
// `want_bytes == 0` means "no extent contract applies", which is a real and load-bearing case, not a
// degenerate one: gpu_replay's ordered-prefix inspection (--draw / --draw-steps /
// --through-operation) and the render_submit_items tests consume the last pass AT ITS OWN EXTENT and
// infer the extent from the byte count afterwards. tests/test_gpu_capture_render.cpp asserts exactly
// that ("per-target chain returns the last pass at its declared 16x16 extent"). Neither goes through
// the publish gate, which is what tells the two apart — see PresentSubmitScope in gpu_execute.hpp.

// A submit's present-source candidates, in the historical identity priority order.
enum class PresentSourceChoice { None, Front, Vo, Last };

// The names the renderer's own provenance diagnostics have always used for these candidates.
constexpr const char* present_source_name(PresentSourceChoice choice) {
    switch (choice) {
        case PresentSourceChoice::Front: return "px_front";
        case PresentSourceChoice::Vo:    return "px_vo";
        case PresentSourceChoice::Last:  return "px_last";
        case PresentSourceChoice::None:  break;
    }
    return "none";
}

// Pick a present source. Under an extent contract, a candidate whose byte count disagrees with
// `want_bytes` is not a present source at all, so a correctly sized lower-priority candidate wins
// over an incorrectly sized higher-priority one; when nothing qualifies the answer is None, never a
// frame the caller must throw away. A candidate's `*_bytes` is 0 when it is absent or empty.
constexpr PresentSourceChoice select_present_source(size_t front_bytes, size_t vo_bytes,
                                                    size_t last_bytes, size_t want_bytes) {
    if (want_bytes == 0)
        return front_bytes ? PresentSourceChoice::Front
             : vo_bytes    ? PresentSourceChoice::Vo
             : last_bytes  ? PresentSourceChoice::Last
                           : PresentSourceChoice::None;
    if (front_bytes == want_bytes) return PresentSourceChoice::Front;
    if (vo_bytes    == want_bytes) return PresentSourceChoice::Vo;
    if (last_bytes  == want_bytes) return PresentSourceChoice::Last;
    return PresentSourceChoice::None;
}

// True when the extent contract passed over a non-empty HIGHER-priority candidate to reach `choice`
// — the one new semantic path this rule introduces, and therefore the one worth a report. It is
// unreachable on current master: a pass whose target is the flipped front buffer or any registered
// scanout has its extent pinned to the present extent before it renders (`live_renderer.cpp:5060-5063`
// sets `native_w/native_h` from `present_width()/present_height()` for `is_vo`), and the contract only
// applies where that extent is non-zero, so `px_front`/`px_vo` always fit. That makes this a tripwire
// on that pin rather than a live branch: if the pin is ever relaxed, a demotion must not be silent —
// a silent substitution is the whole defect #1986 is about.
constexpr bool present_source_demoted(PresentSourceChoice choice, size_t front_bytes,
                                      size_t vo_bytes) {
    if (choice == PresentSourceChoice::Vo)   return front_bytes != 0;
    if (choice == PresentSourceChoice::Last) return front_bytes != 0 || vo_bytes != 0;
    return false;  // Front is already the top priority; None is the shortfall report's business
}

// What the retained-frame net does with this submit's frame. The net exists because Bendy and the
// Ink Machine (PPSA27616) rapidly re-registers its scanout buffers: on the frames where the guest
// has registered new buffers but not yet rendered and flipped them, none is valid and the present
// would be a black flicker, so the previous frame is presented instead. Before #1986 the net tested
// only emptiness, so a non-empty WRONG-EXTENT frame took the store branch and overwrote the retained
// good frame — which is why Frontiers' wall was permanent rather than a one-frame flicker.
enum class RetainedFrameAction {
    StoreCurrent,   // this submit's frame is publishable: publish it, and retain it for later
    ServeRetained,  // this submit's frame is unusable: publish the retained frame, keep retaining it
    NoUsableFrame,  // neither is publishable: there is nothing to publish this submit
};

// A frame is publishable when it is non-empty and, under an extent contract, exactly `want_bytes`.
// Retaining is deliberately the same predicate as publishing: a frame that could not be published
// must never become the frame served on a later miss.
constexpr bool present_frame_publishable(size_t frame_bytes, size_t want_bytes) {
    return frame_bytes != 0 && (want_bytes == 0 || frame_bytes == want_bytes);
}

constexpr RetainedFrameAction retained_frame_action(size_t current_bytes, size_t retained_bytes,
                                                    size_t want_bytes) {
    if (present_frame_publishable(current_bytes, want_bytes))
        return RetainedFrameAction::StoreCurrent;
    if (present_frame_publishable(retained_bytes, want_bytes))
        return RetainedFrameAction::ServeRetained;
    return RetainedFrameAction::NoUsableFrame;
}

} // namespace prosper::frontend
