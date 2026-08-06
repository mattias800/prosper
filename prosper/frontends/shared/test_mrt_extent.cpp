// Unit test for mrt_extent_conflicts (live-renderer MRT-prefix truncation).
//
// The predicate decides whether a bound colour slot may join the pass or whether the MRT prefix
// truncates before it. Truncating drops the attachment with no log and no failure, so the boundary
// between "measured extents that really disagree" (truncate) and "an extent nobody measured"
// (keep) is pinned here.
//
// `legacy_extent_conflicts` below is the predicate this file replaced, kept as an explicit
// counter-arm: several cases assert that the two DISAGREE, so the test cannot pass unless the guard
// is actually present. A test that only asserted the new contract could stay green against an
// implementation that never grew the guard at all.

#include "mrt_extent.hpp"

#include <cstdint>
#include <cstdio>

using prosper::frontend::mrt_extent_conflicts;
using prosper::frontend::mrt_extent_known;

// The pre-#2114 comparison, verbatim: a bare inequality against the pass extent.
static constexpr bool legacy_extent_conflicts(uint32_t slot_w, uint32_t slot_h,
                                              uint32_t pass_w, uint32_t pass_h) {
    return slot_w != pass_w || slot_h != pass_h;
}

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    // --- mrt_extent_known: 0 on either axis is the "never measured" sentinel. ---
    CHECK(mrt_extent_known(1920, 1080));
    CHECK(mrt_extent_known(1, 1));            // CB_COLORn_ATTRIB2 is biased by one: a written
                                              // register decodes to at least 1x1, never 0.
    CHECK(!mrt_extent_known(0, 0));
    CHECK(!mrt_extent_known(1920, 0));
    CHECK(!mrt_extent_known(0, 1080));

    // --- Preserved behaviour: two MEASURED extents that really differ still truncate. ---
    // This is the positive control. If the guard were written too broadly (e.g. returning false
    // whenever any axis compares unequal) these would flip and the truncation would stop working.
    CHECK(mrt_extent_conflicts(1024, 32, 1920, 1080));     // small LUT bound beside a scene target
    CHECK(mrt_extent_conflicts(1920, 1080, 3840, 2160));
    CHECK(mrt_extent_conflicts(1920, 540, 1920, 1080));     // height alone differs
    CHECK(mrt_extent_conflicts(960, 1080, 1920, 1080));     // width alone differs
    CHECK(legacy_extent_conflicts(1024, 32, 1920, 1080));   // old predicate agreed here
    CHECK(legacy_extent_conflicts(1920, 540, 1920, 1080));

    // --- Preserved behaviour: two measured extents that agree never truncate. ---
    CHECK(!mrt_extent_conflicts(1920, 1080, 1920, 1080));
    CHECK(!mrt_extent_conflicts(1, 1, 1, 1));
    CHECK(!mrt_extent_conflicts(16384, 16384, 16384, 16384));  // MIP0_WIDTH is 14 bits + 1

    // --- #2114: the pass extent was never measured. ---
    // CB_COLOR0_ATTRIB2 unwritten leaves native_w/native_h at 0x0 while MRT1 carries a real
    // extent. The old predicate reported a conflict against 0 and truncated mrt_count to 1,
    // silently dropping the second colour attachment.
    CHECK(!mrt_extent_conflicts(1920, 1080, 0, 0));
    CHECK(!mrt_extent_conflicts(1024, 32, 0, 0));
    CHECK(!mrt_extent_conflicts(1, 1, 0, 0));
    // The lever, made visible: old and new disagree on exactly this input.
    CHECK(legacy_extent_conflicts(1920, 1080, 0, 0));
    CHECK(legacy_extent_conflicts(1920, 1080, 0, 0) != mrt_extent_conflicts(1920, 1080, 0, 0));
    CHECK(legacy_extent_conflicts(1024, 32, 0, 0) != mrt_extent_conflicts(1024, 32, 0, 0));

    // --- The same sentinel on the slot side. ---
    // CB_COLORn_ATTRIB2 is a dense per-slot array (0x3B0, 0x3B1, ...), so a slot can carry a base,
    // a format and a write mask -- everything that makes it a real, actively written target -- while
    // its own extent register has not been seen. That is equally not evidence of a mismatch.
    CHECK(!mrt_extent_conflicts(0, 0, 1920, 1080));
    CHECK(!mrt_extent_conflicts(0, 0, 1024, 32));
    CHECK(legacy_extent_conflicts(0, 0, 1920, 1080));
    CHECK(legacy_extent_conflicts(0, 0, 1920, 1080) != mrt_extent_conflicts(0, 0, 1920, 1080));

    // A half-populated extent cannot come from one register write, but must not be trusted if seen.
    CHECK(!mrt_extent_conflicts(1920, 0, 1920, 1080));
    CHECK(!mrt_extent_conflicts(0, 1080, 1920, 1080));
    CHECK(!mrt_extent_conflicts(1920, 1080, 1920, 0));
    CHECK(!mrt_extent_conflicts(1920, 1080, 0, 1080));

    // --- Neither side measured: already benign before the fix (0 == 0), pinned so it stays so. ---
    CHECK(!mrt_extent_conflicts(0, 0, 0, 0));
    CHECK(!legacy_extent_conflicts(0, 0, 0, 0));

    // --- Symmetry: the predicate is about disagreement, so measured pairs commute. ---
    CHECK(mrt_extent_conflicts(1024, 32, 1920, 1080) ==
          mrt_extent_conflicts(1920, 1080, 1024, 32));
    CHECK(mrt_extent_conflicts(1920, 1080, 1920, 1080) ==
          mrt_extent_conflicts(1920, 1080, 1920, 1080));

    // --- Usable in constant expressions, like the rest of the frontend's extent predicates. ---
    static_assert(!mrt_extent_conflicts(1920, 1080, 0, 0), "#2114: unknown pass extent never conflicts");
    static_assert(!mrt_extent_conflicts(0, 0, 1920, 1080), "unknown slot extent never conflicts");
    static_assert(mrt_extent_conflicts(1024, 32, 1920, 1080), "measured disagreement still truncates");
    static_assert(!mrt_extent_known(0, 0), "0x0 is the not-measured sentinel");

    if (failures == 0) std::printf("mrt_extent: OK\n");
    return failures == 0 ? 0 : 1;
}
