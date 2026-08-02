// #1761: the rate-limiting contract for capped diagnostics.
//
// The defect this guards is not a crash — it is a number. #1226 spent multiple sessions comparing
// "the forge tripwire fires 64 times" against "64 suppressed MB3 writes" as if they described the
// same population. Both were print artifacts. This suite pins the two properties that make such a
// number trustworthy: the tail exists (so the ceiling is not the finding), and each key gets its
// own budget (so a noisy key cannot starve the one under investigation).

#include "../src/gpu/diag_ratelimit.hpp"

#include <cstdint>
#include <cstdio>

using namespace prosper;

static int failures = 0;
static void check(bool ok, const char* name) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) failures++;
}

int main() {
    // --- the print window ------------------------------------------------------------------
    bool all_first = true;
    for (uint64_t o = 1; o <= 64; o++) if (!diag_should_print(o)) all_first = false;
    check(all_first, "ordinals 1..64 all print");

    bool gap_silent = true;
    for (uint64_t o = 65; o <= 127; o++) if (diag_should_print(o)) gap_silent = false;
    check(gap_silent, "ordinals 65..127 are suppressed");

    check(diag_should_print(128) && diag_should_print(256) && diag_should_print(4096),
          "powers of two print past the cap");

    bool between_quiet = true;
    for (uint64_t o = 129; o <= 255; o++) if (diag_should_print(o)) between_quiet = false;
    check(between_quiet, "non-powers between 128 and 256 stay suppressed");

    check(!diag_should_print(0), "ordinal 0 is not a valid 1-based ordinal and does not print");

    // --- the #1226 discriminator -----------------------------------------------------------
    // The old mb3_freelist_report rule was `i < 64 || (i & 4095) == 0`, which put the 65th printed
    // line on the 4,097th event. That is precisely why "64 suppressed live-protocol writes"
    // bounded the population only BELOW 4,096 instead of measuring it. The 65th line must land on
    // ordinal 128 — close enough behind the window that the tail is actually informative.
    uint64_t printed = 0, sixty_fifth = 0;
    for (uint64_t o = 1; o <= 8192 && !sixty_fifth; o++)
        if (diag_should_print(o)) { printed++; if (printed == 65) sixty_fifth = o; }
    check(sixty_fifth == 128, "the 65th printed line lands on ordinal 128, not 4096");
    if (sixty_fifth != 128) printf("       got %llu\n", (unsigned long long)sixty_fifth);

    // A 4,096-event population is now legible from the log alone: the LAST ordinal is a true lower
    // bound on the population, where the line count never was.
    printed = 0;
    uint64_t last = 0;
    for (uint64_t o = 1; o <= 4096; o++) if (diag_should_print(o)) { printed++; last = o; }
    check(printed == 70, "4,096 events produce 70 lines (64 + six powers of two)");
    check(last == 4096, "the last line of a 4,096-event population carries ordinal 4096");

    // Growth is logarithmic, so the tail is affordable on an unbounded population.
    printed = 0;
    for (uint64_t o = 1; o <= (1u << 20); o++) if (diag_should_print(o)) printed++;
    check(printed == 78, "2^20 events produce 78 lines (64 + fourteen powers of two)");

    // --- caller-supplied windows -------------------------------------------------------------
    // report_suspect_write budgets 192 and the generation reporters budget 256; the window must
    // follow first_n rather than a hardcoded 64.
    check(diag_should_print(192, 192) && !diag_should_print(193, 192) && diag_should_print(256, 192),
          "first_n=192 window is honored, with the tail resuming at 256");
    check(diag_should_print(256, 256) && !diag_should_print(257, 256) && diag_should_print(512, 256),
          "first_n=256 window is honored, with the tail resuming at 512");

    // --- per-key budgets ---------------------------------------------------------------------
    static const char* const kKinds[] = {"REL1-LIVE", "REL1-NOINIT", "REL1-FORGE"};
    check(diag_key_slot("REL1-LIVE", kKinds, 3) == 0, "first key maps to slot 0");
    check(diag_key_slot("REL1-FORGE", kKinds, 3) == 2, "last key maps to its own slot");
    check(diag_key_slot("WDATA", kKinds, 3) == 3, "an unlisted key gets the overflow slot");
    check(diag_key_slot(nullptr, kKinds, 3) == 3, "a null key gets the overflow slot");

    // The property that blocked finding B2 on #1754: with ONE shared counter, a noisy kind
    // exhausts the budget before the kind under investigation ever prints. Per-key budgets must
    // make the quiet kind's output independent of the noisy kind's volume.
    uint64_t counters[4] = {0, 0, 0, 0};
    for (int i = 0; i < 10000; i++)
        (void)diag_should_print(++counters[diag_key_slot("REL1-LIVE", kKinds, 3)]);
    uint64_t quiet_printed = 0;
    for (int i = 0; i < 3; i++)
        if (diag_should_print(++counters[diag_key_slot("REL1-FORGE", kKinds, 3)])) quiet_printed++;
    check(quiet_printed == 3, "10,000 events on a noisy key do not starve a quiet key's budget");

    // Trailing summary, present iff the run completed — an abort partway through must not read as
    // a pass.
    printf("%s (%d failed)\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
