#pragma once
// Rate-limiting contract for capped diagnostics (#1761; instrument-trap 49).
//
// A capped diagnostic must never let its own ceiling masquerade as the measurement. #1226 spent
// multiple sessions comparing "the forge tripwire fires 64 times" against "64 suppressed MB3
// writes" as if they were the same population. Both numbers were print artifacts — the forge
// population is above 1,024, and the MB3 figure was bounded only *below* 4,096 — and the apparent
// numerical coincidence shaped the investigation.
//
// The contract has two halves, and only both together are sufficient:
//
//   1. Carry the 1-based ordinal on EVERY line. The last line's ordinal is then a lower bound on
//      the population; the line COUNT never is. Without this a reader cannot tell the two apart,
//      which is the whole defect.
//   2. Keep printing past the cap, sparsely, so the tail exists at all. Powers of two cost
//      O(log n) lines for an unbounded population.
//
// Budget PER KEY wherever a diagnostic has several (kinds, values, addresses): with one shared
// counter a noisy key exhausts the log before the key actually under investigation ever gets a
// line. That is not hypothetical — it blocked finding B2 on #1754.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace prosper {

// True if the event with 1-based ordinal `ord` should be printed: the first `first_n`, then every
// power of two. Pure and total, so it is unit-testable without any GPU state.
inline bool diag_should_print(uint64_t ord, uint64_t first_n = 64) {
    if (ord == 0) return false;                    // 0 is not a valid 1-based ordinal
    return ord <= first_n || (ord & (ord - 1)) == 0;
}

// Index of `key` within `keys[0..n)`, or `n` (one past the end) as a shared overflow slot for an
// unlisted key — so an unexpected kind still prints rather than silently vanishing. Callers size
// their counter array `n + 1`.
inline size_t diag_key_slot(const char* key, const char* const* keys, size_t n) {
    if (!key) return n;
    for (size_t i = 0; i < n; i++)
        if (keys[i] && std::strcmp(key, keys[i]) == 0) return i;
    return n;
}

}  // namespace prosper
