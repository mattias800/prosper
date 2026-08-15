#pragma once
#include <cstdint>

// Shared predicate for the live renderer's MRT-prefix truncation.
//
// A render pass binds one colour attachment per exported MRT slot, and the live renderer records
// EVERY slot's surface at the single extent the pass renders at (live_renderer.cpp writes each
// slot's RttSurf as gw x gh). A bound slot whose own surface is really a different size therefore
// cannot join that pass, and the MRT prefix has to stop before it.
//
// That is a statement about two MEASURED extents, which is the whole point of the guard below.
// A colour target's extent is only ever populated from CB_COLORn_ATTRIB2, whose MIP0_WIDTH and
// MIP0_HEIGHT fields are biased by one: render_state.cpp decodes any register the guest actually
// wrote to at least 1x1, and leaves the extent at 0x0 exclusively when prosper has never seen that
// register. Zero is a "not measured" sentinel, never a real surface size -- and the two registers
// are independent (0x3B0 and 0x3B1, a dense per-slot array), so either side can be absent on its
// own.
//
// Comparing an unmeasured extent for equality manufactures a conflict out of missing data. It is
// also silent in the worst way: mrt_count truncates, the second colour attachment is dropped, the
// shader's second output goes nowhere, and nothing logs and nothing fails -- the frame simply
// renders without whatever that target carried (a normal buffer, velocity, an ID target). #2114.
namespace prosper::frontend {

// Whether a colour target's extent was actually measured. Both axes come from one register write,
// so a half-populated extent does not occur; requiring both mirrors live_renderer.cpp's existing
// `if (native_w && native_h)` guards on the same values.
constexpr bool mrt_extent_known(uint32_t w, uint32_t h) {
    return w != 0u && h != 0u;
}

// Does this pass bind ANY colour slot within its active prefix?
//
// Two decisions in the live renderer key on this and both were once written as "is colour-0 bound":
// whether to hand the backend a BackendColorTarget at all, and whether to ask for colour pixels
// back. A pass with colour-0 unbound and a higher slot bound is reachable by construction --
// `mrt_count` does not depend on colour-0 -- and on `base != 0` alone such a pass populated the
// per-slot persistence contract and then handed the backend a nullptr, so those active slots stayed
// transient and were cleared by the next render group. Depth-only passes have every slot zero, so
// the "skip the work" win the readback flag was added for is unchanged.
//
// Takes a pointer and a count rather than a container so the production array and a test fixture
// exercise the identical expression.
constexpr bool mrt_any_slot_bound(const uint64_t* slot_bases, uint32_t count) {
    if (!slot_bases) return false;
    for (uint32_t slot = 0; slot < count; ++slot)
        if (slot_bases[slot] != 0u) return true;
    return false;
}

// Does a sampled surface collide with a colour target this pass is actively writing?
//
// `bases[i]` is slot i's bound base and `active[i]` whether that slot is really being written. Both
// the backend's descriptor borrow and the frontend's direct-live sampling decision ask this, and
// both once asked it of slots 0 and 1 only -- complete while higher slots could not be persistent,
// and wrong the moment they could. Sampling an ACTIVE MRT2+ target then borrows the very VkImage the
// pass is writing, using one image as shader-read and colour-attachment at once and bypassing the
// CPU fallback that exists for exactly this case.
//
// An inactive slot does not collide: a stale base with a zero write mask is not a target.
constexpr bool mrt_target_feedback(const uint64_t* bases, const bool* active, uint32_t count,
                                   uint64_t sampled) {
    if (!bases || !active || !sampled) return false;
    for (uint32_t slot = 0; slot < count; ++slot)
        if (active[slot] && bases[slot] == sampled) return true;
    return false;
}

// True only when a bound slot's extent is KNOWN to disagree with the extent the pass renders at.
// An unknown extent on either side yields false: absence of a measurement is not evidence of a
// mismatch, and the fail-safe direction is to keep a real, actively-written attachment rather than
// to drop its contents silently.
constexpr bool mrt_extent_conflicts(uint32_t slot_w, uint32_t slot_h,
                                    uint32_t pass_w, uint32_t pass_h) {
    if (!mrt_extent_known(pass_w, pass_h)) return false;
    if (!mrt_extent_known(slot_w, slot_h)) return false;
    return slot_w != pass_w || slot_h != pass_h;
}

} // namespace prosper::frontend
