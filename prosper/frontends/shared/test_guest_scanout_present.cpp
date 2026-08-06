// test_guest_scanout_present — when prosper may publish the guest's OWN flipped buffer (#1968).
//
// The defect this guards is the one that reverted the first attempt (#2026 -> #2044): the branch was
// 81 lines of renderer code with no test at all, and its load-bearing premise — "the guest wrote
// this buffer" — was approximated by "not all its bytes are zero". That approximation holds only for
// a freshly zeroed allocation. A title re-registering scanouts over REUSED memory (Bendy and the Ink
// Machine, PPSA27616) hands back unrelated bytes that pass it trivially, while prosper's
// render-target map misses precisely BECAUSE the address is new — so stale garbage would be
// published in place of the retained good frame. Authorship is now established differentially (see
// videoout_read_front_linear); this file asserts the decision that consumes it.
//
// Every predicate here is a pure function of the renderer's own booleans and byte counts, so no
// Vulkan device, guest or renderer registration is involved. Each arm names the outcome it forbids,
// and each is written so that flipping the input flips the verdict — a test that cannot show its
// lever moved proves nothing about the mechanism.
#include "guest_scanout_present.hpp"

#include <cstdio>
#include <string_view>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); ++fails; } \
                              else         { std::printf("  [ok]   %s\n", msg); } } while (0)

namespace {

using prosper::frontend::GuestScanoutDecision;
using prosper::frontend::guest_scanout_decision_name;
using prosper::frontend::guest_scanout_publishable;
using prosper::frontend::guest_scanout_read_warranted;

constexpr size_t bytes_for(size_t w, size_t h) { return w * h * 4; }

// Sonic Frontiers' measured configuration: a 3840x2160 present with no renderer source at all.
constexpr size_t k4k = bytes_for(3840, 2160);
// The same title under a snapshot guard's PROSPER_RENDER_SCALE=4: the display buffer is still 4K
// while the renderer presents a quarter-linear-scale frame.
constexpr size_t kScaled = bytes_for(960, 540);

} // namespace

int main() {
    std::printf("== test_guest_scanout_present ==\n");

    // ---- stage 1: is the read even warranted? ------------------------------------------------
    CHECK(guest_scanout_read_warranted(false, false, false, k4k, k4k) ==
              GuestScanoutDecision::Publish,
          "no GPU publish, no renderer scanout, no selected frame, full-scale present -> read it");

    // Priority. This branch must never change which of prosper's own sources wins; it only decides
    // what happens when there is none. Each of the three arms below is a source that already works.
    CHECK(guest_scanout_read_warranted(true, false, false, k4k, k4k) ==
              GuestScanoutDecision::SkipPublishedGpu,
          "a GPU-published frame for this flip is never overridden by guest memory");
    CHECK(guest_scanout_read_warranted(false, true, false, k4k, k4k) ==
              GuestScanoutDecision::SkipRendererSource,
          "prosper's own image for the flipped address wins: unscaled, no CPU round trip");
    CHECK(guest_scanout_read_warranted(false, false, true, k4k, k4k) ==
              GuestScanoutDecision::SkipRendererSource,
          "a surviving selected frame keeps its priority — selection already refused wrong extents");

    // The paths that must stay byte-for-byte unaffected.
    CHECK(guest_scanout_read_warranted(false, false, false, 0, k4k) ==
              GuestScanoutDecision::SkipNoPresentContract,
          "no present contract (gpu_replay / render_submit_items) never takes this branch");
    CHECK(guest_scanout_read_warranted(false, false, false, kScaled, k4k) ==
              GuestScanoutDecision::SkipScaledPresent,
          "PROSPER_RENDER_SCALE: a 4K guest buffer can never be the reduced present extent");
    // …and the scale test is BEFORE the read on purpose: a guard running at scale 4 would otherwise
    // pay a full-resolution copy and de-swizzle per guest flip only to discard the result.
    CHECK(guest_scanout_read_warranted(false, false, false, kScaled, kScaled) ==
              GuestScanoutDecision::Publish,
          "a scaled present whose display extent happens to match is not excluded by the pre-test");

    // ---- stage 2: may the frame that came back be published? ---------------------------------
    CHECK(guest_scanout_publishable(k4k, k4k, true, false, true) == GuestScanoutDecision::Publish,
          "a readable, present-extent, unowned, guest-authored buffer publishes");

    // B2, the finding that caused the revert. This is the Bendy case stated as a predicate: a
    // re-registered buffer over reused memory is non-empty, prosper owns no target at its new
    // address, and it is still NOT the guest's frame.
    CHECK(guest_scanout_publishable(k4k, k4k, true, false, false) ==
              GuestScanoutDecision::SkipNotAuthored,
          "an unauthored buffer declines even though it is readable, sized and unowned (B2)");
    // The lever: authorship is the ONLY difference between the two arms above. If the predicate
    // stopped consulting it, the first arm would still pass — so the pair is what makes this a test
    // of the mechanism rather than of the happy path.

    CHECK(guest_scanout_publishable(0, k4k, true, false, true) == GuestScanoutDecision::SkipUnreadable,
          "a read that failed publishes nothing, whatever else is true");
    CHECK(guest_scanout_publishable(k4k, k4k, false, false, true) ==
              GuestScanoutDecision::SkipUnreadable,
          "a read with no buffer address publishes nothing");
    CHECK(guest_scanout_publishable(bytes_for(1920, 1080), k4k, true, false, true) ==
              GuestScanoutDecision::SkipWrongExtent,
          "a guest buffer that is not the present extent is refused, exactly as a pass would be");
    CHECK(guest_scanout_publishable(k4k, k4k, true, true, true) ==
              GuestScanoutDecision::SkipRendererOwnsTarget,
          "a renderer target at the flipped address means prosper is the writer, mid-state or not");

    // Ordering matters where two reasons apply at once: the cheaper, more fundamental one must win,
    // so a diagnostic never reports "not authored" for a buffer that was never read.
    CHECK(guest_scanout_publishable(0, k4k, true, true, false) ==
              GuestScanoutDecision::SkipUnreadable,
          "an unreadable buffer reports unreadable, not the conditions it never got to evaluate");

    // ---- the reasons are distinguishable ------------------------------------------------------
    // A negative control that can only observe silence cannot tell "correctly declined" from "never
    // reached", which is exactly the evidence Bendy's control has to produce.
    CHECK(std::string_view(guest_scanout_decision_name(GuestScanoutDecision::SkipNotAuthored)) ==
              "not-authored" &&
          std::string_view(guest_scanout_decision_name(GuestScanoutDecision::Publish)) == "publish" &&
          std::string_view(guest_scanout_decision_name(GuestScanoutDecision::SkipScaledPresent)) !=
              std::string_view(guest_scanout_decision_name(GuestScanoutDecision::SkipNotAuthored)),
          "each decline reports its own reason, so a zero count is readable as a correct decline");

    // ---- constexpr: the whole decision is compile-time evaluable ------------------------------
    static_assert(guest_scanout_read_warranted(false, false, false, k4k, k4k) ==
                  GuestScanoutDecision::Publish);
    static_assert(guest_scanout_publishable(k4k, k4k, true, false, false) ==
                  GuestScanoutDecision::SkipNotAuthored);

    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
