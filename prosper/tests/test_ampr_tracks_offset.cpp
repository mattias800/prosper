// #2384: an AMPR command buffer whose constructor passes a PACKED `a2` must still track its cursor.
//
// `sceAmprCommandBufferGetCurrentOffset` answers from a per-buffer cursor that only advances when
// the constructor recorded `tracks_offset`. That flag was inferred from `a2 >= 0x20 && a2 <=
// 0x100000` -- "is a2 itself a plausible capacity". Grand Theft Auto V (PPSA04263) passes a packed
// `(count << 32) | capacity`, which fails that test, so its cursor never moved, GetCurrentOffset
// answered 0 forever, and the guest's own assertion at eboot+0x2b2c463 trapped on it.
//
// WHAT THIS FILE GUARDS IS THE ADDITIVITY, NOT THE NEW CASE.
//
// The new case is one line and a boot proves it. The dangerous half is the other direction: this
// predicate is shared by every title that constructs an AMPR command buffer, and turning tracking ON
// for a buffer that previously had it OFF changes when that title submits (UE4 batches until
// `GetSize - GetCurrentOffset` can no longer hold the next packet). None of those titles' dumps are
// available to CI, so a boot test cannot speak for them -- but the predicate is a pure function of a
// 64-bit argument, and the shapes they pass are recorded. Asserting over those shapes is the only
// check that can run here and the only one that would fail if the widening leaked.

#include "hle/guest_memory_topology.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); ++failures; }
    else std::fprintf(stderr, "ok: %s\n", what.c_str());
}

static std::string hex(uint64_t v) {
    char b[32]; std::snprintf(b, sizeof b, "0x%llx", (unsigned long long)v); return b;
}

int main() {
    std::fprintf(stderr, "== test_ampr_tracks_offset ==\n");
    const auto tracks = &prosper::ampr_cb_tracks_offset_arg_for_test;

    // ---- the new case: GTA V's ten constructor calls from one live boot -------------------------
    //
    // Recorded verbatim under PROSPER_AMPRLOG=1, not synthesized. The four consecutive entries in
    // the middle are the evidence that `a2` decomposes rather than merely happening to have a
    // nonzero high half: as the high field falls 95, 94, 93, 92 the low field rises by exactly 0x70
    // each step. That stride is asserted below, because it is the whole basis for reading the low
    // half as a capacity -- if a future edit "fixes" these constants, the arm that made them
    // meaningful should break too.
    const uint64_t gta[] = {
        0x6400001370ull, 0x5f00001610ull, 0x5e00001680ull, 0x5d000016f0ull,
        0x5c00001760ull, 0x9c000022b0ull, 0xe7000008a0ull, 0x7500000c70ull,
        0x6700001290ull, 0x5700001990ull,
    };
    for (uint64_t a2 : gta)
        check(tracks(a2), "GTA V packed a2 " + hex(a2) + " tracks its offset");

    // The stride, over the run of four -- which is entries [1]..[4], NOT [0]..[3]. This arm caught
    // that off-by-one when it was first written: [0] -> [1] is count -5 and capacity +0x2a0, so the
    // pair is not on the same carve, and a loop that included it failed. Left as a bounded loop over
    // the exact run rather than widened to fit, because the claim being guarded is about those four
    // and admitting [0] would have meant weakening the assertion to make it pass.
    //
    // This is a property of the RECORDING, so it can only fail if someone edits the constants --
    // which is exactly when the comment above stops being true.
    for (int i = 2; i <= 4; ++i) {
        const uint64_t prev = gta[i - 1], cur = gta[i];
        check((prev >> 32) - (cur >> 32) == 1 &&
                  (cur & 0xffffffffull) - (prev & 0xffffffffull) == 0x70,
              "consecutive GTA V pair " + std::to_string(i) +
                  ": count falls by 1 while capacity rises by 0x70");
    }

    // ---- ADDITIVITY: everything that answered yes before still answers yes ----------------------
    // Pathless's PS5 3.20 IoStore batch passes the capacity directly.
    check(tracks(0x560ull), "Pathless plain a2=0x560 still tracks (unchanged)");
    check(tracks(0x20ull),  "the low boundary 0x20 still tracks (unchanged)");
    check(tracks(0x100000ull), "the high boundary 0x100000 still tracks (unchanged)");

    // ---- ADDITIVITY: everything that answered no before still answers no, where it must ---------
    // DOLL passes a2 = 0 or 1 as a mode flag and carries capacity in a5. If the widening reached
    // these, DOLL would start tracking a cursor for a buffer that has none.
    check(!tracks(0ull), "DOLL a2=0 still does NOT track");
    check(!tracks(1ull), "DOLL a2=1 still does NOT track");
    check(!tracks(0x1full), "0x1f, just below the plain floor, still does NOT track");
    check(!tracks(0x100001ull), "0x100001, just above the plain ceiling, still does NOT track");

    // THE STRUCTURAL ARGUMENT, asserted rather than asserted-in-a-comment. Additivity comes from the
    // ORDER: the plain arm runs first and returns true for everything it accepts, so no
    // previously-accepted value ever reaches the packed arm to be re-decided. An earlier version of
    // this comment credited the `count != 0` guard instead; mutation testing showed that guard is
    // REDUNDANT -- deleting it survives, because when count is zero a2 IS cap and the plain arm has
    // already answered. The guard stays as a statement of intent, but it is not what makes this safe.
    //
    // A hand-picked list of examples could not establish this; a sweep can. The claim is a
    // DISJOINTNESS one, and no finite list of examples is evidence for disjointness.
    for (uint64_t cap = 0; cap <= 0x100100ull; cap += 0x37ull) {
        const bool plain_says_yes = (cap >= 0x20 && cap <= 0x100000);
        if (plain_says_yes && !tracks(cap)) {
            check(false, "sweep: plain-accepted " + hex(cap) + " must still track");
            break;
        }
        if (!plain_says_yes && cap <= 0x100000ull && tracks(cap)) {
            check(false, "sweep: plain-rejected " + hex(cap) + " must NOT start tracking");
            break;
        }
    }
    check(true, "sweep over 0..0x100100 found no value whose plain-arm answer changed");

    // ---- the packed arm's own bounds ------------------------------------------------------------
    // A pointer-shaped a2 must not be mistaken for a packed descriptor. Guest pointers in these
    // captures look like 0x20003f13f0 -- high half 0x20, low half 0x003f13f0, which is far above the
    // capacity ceiling and so already excluded. Asserted because "the ceiling happens to exclude
    // pointers" is load-bearing and invisible.
    check(!tracks(0x20003f13f0ull),
          "a guest POINTER 0x20003f13f0 is not mistaken for a packed descriptor");
    check(!tracks(0xffffffffffffffffull), "the -1 sentinel does not track");
    check(!tracks(0x100000000ull), "count=1 with capacity=0 does not track");

    // The `count <= 0xffff` bound needs a case where the COUNT is the only thing rejecting the
    // value -- i.e. the capacity half is perfectly plausible. Mutation testing is what forced this
    // arm: deleting the bound SURVIVED the first version of this file, because every large-count
    // case here also had an out-of-range capacity and was rejected twice over. An assertion that
    // two guards jointly reject something says nothing about either guard.
    check(!tracks((0x10000ull << 32) | 0x1000ull),
          "count=0x10000 with a PLAUSIBLE capacity 0x1000 is still rejected -- the count bound is "
          "load-bearing and not merely shadowed by the capacity check");
    check(tracks((0xffffull << 32) | 0x1000ull),
          "...while count=0xffff with the same capacity IS accepted, so the bound is a boundary "
          "rather than a blanket rejection");

    std::fprintf(stderr, failures ? "== FAIL ==\n" : "== PASS ==\n");
    return failures ? 1 : 0;
}
