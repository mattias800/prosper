#pragma once
#include <cstdint>
#include <cstdlib>

namespace prosper::frontend {

// A write-only shader can map invocations to texels in swizzled or vectorized ways (Astro Bot's
// bloom chain launches width*8 by height/8). Per-axis comparison therefore rejects valid full-image
// fills. The only shape-independent prerequisite is enough total invocations; the poison proving
// pass below establishes actual coverage before any seed is skipped.
constexpr bool dispatch_has_enough_threads_for_texels(uint32_t threads_x, uint32_t threads_y,
                                                       uint32_t threads_z, uint32_t width,
                                                       uint32_t height, uint32_t depth) {
    if (!threads_x || !threads_y || !threads_z || !width || !height || !depth) return false;
    return static_cast<uint64_t>(threads_x) * threads_y * threads_z >=
           static_cast<uint64_t>(width) * height * depth;
}

// #1127: the #1122 seed-skip caches a "this write-only shader covers every texel" verdict per
// (shader, binding, extent) and trusts it forever. That is unsound for a DATA-DEPENDENT store (full
// on the proving frame, partial on a later frame with different input): the untouched texels then pack
// undefined device memory to the guest. Bound the exposure by periodically re-proving a Full verdict.
//
// Advance the per-key fast-skip counter and report whether this dispatch is due to re-prove. `skips`
// counts fast-skips taken since the last (re-)proof; on the interval-th one it fires (true) and resets
// to 0, so the next re-prove is `interval` fast-skips later. `interval == 0` disables re-proving
// (the old prove-once behavior). Pure and lock-free — the caller holds the verdict-map mutex.
inline bool seed_reprove_due(uint32_t& skips, uint32_t interval) {
    if (interval == 0) return false;
    if (++skips >= interval) { skips = 0; return true; }
    return false;
}

// Parse the PROSPER_SEED_REPROVE re-prove interval, failing SAFE. `env` is the raw getenv() result
// (nullptr when unset). Only a fully-parsed, non-negative, in-uint32-range integer overrides `dflt`;
// unset, empty, non-numeric ("foo"), trailing junk ("256x"), negative, or overflow ("4294967296")
// all keep `dflt` -- so a typo in the safety knob can never silently DISABLE the safety (which an
// atol-style parse would, by returning 0). An explicit, exact "0" is honored: it intentionally
// disables re-proving (restores the old prove-once behavior). Uses strtoll so the range check is
// correct on both LP64 (Linux) and LLP64 (MinGW, where long is 32-bit).
inline uint32_t seed_reprove_interval_from_env(const char* env, uint32_t dflt) {
    if (!env || !*env) return dflt;
    char* end = nullptr;
    long long v = std::strtoll(env, &end, 10);
    if (end == env || *end != '\0' || v < 0 || v > (long long)UINT32_MAX) return dflt;
    return (uint32_t)v;
}

}  // namespace prosper::frontend
