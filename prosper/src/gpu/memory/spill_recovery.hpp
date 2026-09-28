// spill_recovery.hpp — when may a retained resource that spilled to system memory be rebuilt in
// device-local memory? (#3905)
//
// Since #3897 a GPU-only allocation that gets VK_ERROR_OUT_OF_DEVICE_MEMORY lands on the next memory
// type the resource allows, which on a discrete card is system memory. Transient allocations recover
// by themselves: the next request tries device-local first again. RETAINED resources (persistent
// texture images, persistent colour targets) do not -- once created they stay where they landed for
// as long as they are cached, so after one VRAM spike a title could sample them over PCIe for the
// rest of the session.
//
// The renderer recovers them by EVICTION: a spilled entry that no pending command references is
// dropped, and its next use recreates it through the ordinary allocation path, which tries
// device-local first. Textures are rebuilt from guest memory; a colour target is handed to the
// existing eviction sink (a readback into the frontend's CPU copy) exactly as a budget eviction is.
// This header decides only HOW MANY spilled bytes may be dropped now:
//
//   - at most once per kSpillRecoveryIntervalMs, and not at all while nothing is spilled (so a
//     healthy run pays one compare per pass);
//   - with VK_EXT_memory_budget: min(spilled, headroom, kSpillRecoveryMaxBytesPerStep), where
//     headroom = heap_budget * 80% - heap_usage (saturating) on the heap device-local allocations use.
//     No headroom -> nothing, so a card that is still full is not churned;
//   - without a live budget: a probe of kSpillRecoveryProbeBytes per interval;
//   - BACKOFF: when a step granted bytes and, at the next due step, the spilled total has not dropped
//     below what it was when the grant was made (what was evicted came back spilled, or more spilled),
//     the interval doubles up to kSpillRecoveryMaxBackoffMs. Any progress resets it. This bounds the
//     re-upload cost of a budget that lies, or of a device without one, to one step per minute.
//
// CONFIDENCE: HIGH for the arithmetic; MED for the constants (headroom choices, not measurements; the
// 80% matches texture_cache_budget's target so both policies agree on what "room" means).
#pragma once
#include <algorithm>
#include <cstdint>

namespace prosper::gpu {

inline constexpr uint64_t kSpillRecoveryIntervalMs = 250;
inline constexpr uint64_t kSpillRecoveryMaxBackoffMs = 60'000;
inline constexpr uint64_t kSpillRecoveryMaxBytesPerStep = 256ull * 1024ull * 1024ull;
inline constexpr uint64_t kSpillRecoveryProbeBytes = 64ull * 1024ull * 1024ull;
inline constexpr uint64_t kSpillRecoveryTargetPercentOfBudget = 80;

struct SpillRecoveryInputs {
    uint64_t now_ms = 0;
    uint64_t spilled_bytes = 0;  // bytes of this cache currently in non-device-local memory
    bool have_budget = false;    // VK_EXT_memory_budget figures valid
    uint64_t heap_budget = 0;    // heapBudget of the device-local heap
    uint64_t heap_usage = 0;     // heapUsage of the same heap
};

struct SpillRecoveryState {
    uint64_t next_ms = 0;
    uint64_t interval_ms = kSpillRecoveryIntervalMs;
    uint64_t spilled_at_grant = 0;  // spilled total when the last non-zero grant was made
    bool granted = false;           // the last due step granted bytes
};

// Is a decision due? Callers use this to skip the (driver) budget query entirely.
inline bool spill_recovery_due(const SpillRecoveryState& s, uint64_t now_ms,
                               uint64_t spilled_bytes) {
    return spilled_bytes != 0 && now_ms >= s.next_ms;
}

// Bytes of spilled entries the caller may evict now; 0 = not now. Advances `s`.
inline uint64_t spill_recovery_allowance(const SpillRecoveryInputs& in, SpillRecoveryState& s) {
    if (!spill_recovery_due(s, in.now_ms, in.spilled_bytes)) return 0;
    if (s.granted) {
        s.interval_ms = in.spilled_bytes >= s.spilled_at_grant
            ? std::min(s.interval_ms * 2, kSpillRecoveryMaxBackoffMs)
            : kSpillRecoveryIntervalMs;
    }
    s.next_ms = in.now_ms + s.interval_ms;
    uint64_t allowance = 0;
    if (in.have_budget) {
        const uint64_t target = in.heap_budget / 100 * kSpillRecoveryTargetPercentOfBudget;
        const uint64_t headroom = target > in.heap_usage ? target - in.heap_usage : 0;
        allowance = std::min({in.spilled_bytes, headroom, kSpillRecoveryMaxBytesPerStep});
    } else {
        allowance = std::min(in.spilled_bytes, kSpillRecoveryProbeBytes);
    }
    s.granted = allowance != 0;
    if (s.granted) s.spilled_at_grant = in.spilled_bytes;
    return allowance;
}

}  // namespace prosper::gpu
