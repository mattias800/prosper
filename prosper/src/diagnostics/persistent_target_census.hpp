#pragma once
// Is the persistent colour-target cache at its bound, or is it simply not holding these targets?
//
// Those two states produce the SAME symptom -- `find_persistent_color_target` returns null, the
// compute result cannot be mirrored back onto the GPU, and the render target latches to CPU-only
// authority -- and they have opposite fixes: raise the budget, or create the allocation. Nothing
// distinguished them, and the obvious experiment could not: `PROSPER_BACKEND_TARGET_CACHE_MB`
// changes `persistent_color_target_limit()` while the only line printed about the budget comes
// from `init_persistent_color_target_device_budget`, which never consults the variable. An A/B on
// that variable therefore cannot show its own lever moved, which makes it void rather than
// negative, whichever way the numbers fall.
//
// Residency and eviction, reported directly, settle it without a lever. A cache far below its
// bound with zero evictions cannot be short of capacity.

#include <cstdint>

namespace prosper::diagnostics {

// Called on every entry to the eviction routine, whether or not a victim is found.
//
// A zero here does NOT mean "the cache was never asked to make room": the admission paths skip the
// eviction loop entirely while a submission batch is pending (`avoid_cache_eviction` /
// `eviction_deferred`), so a target can be refused for capacity without this ever incrementing.
// That is why the verdict below rests on the high-water marks rather than on this counter
// (#3872 review).
void note_persistent_target_eviction_attempt();
// Called once per target actually destroyed, with the bytes reclaimed.
void note_persistent_target_evicted(uint64_t bytes);
// Sampled wherever the cache is consulted. Reports a HIGH-WATER MARK, not the last sample: a
// gauge read at exit says nothing about whether the cache was ever under pressure, and "was it
// ever near a bound?" is the entire question. Both bounds are reported because the admission gate
// has two -- a byte budget and an entry count -- and either can refuse a target on its own.
void note_persistent_target_residency(uint64_t entries, uint64_t entry_limit,
                                      uint64_t bytes, uint64_t limit_bytes);

}  // namespace prosper::diagnostics
