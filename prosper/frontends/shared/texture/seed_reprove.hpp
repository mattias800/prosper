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
                                                       uint32_t height, uint32_t depth,
                                                       uint32_t max_texels_per_thread = 1) {
    if (!threads_x || !threads_y || !threads_z || !width || !height || !depth || !max_texels_per_thread) return false;
    return static_cast<uint64_t>(threads_x) * threads_y * threads_z * max_texels_per_thread >=
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

// Coverage classification for storage image writes proven by poison pattern verification.
// Full: every texel is overwritten by the shader (survived == 0).
// Partial: some texels are overwritten and some remain untouched (0 < survived < texels).
// None: zero texels are overwritten by the shader (survived == texels).
enum class SeedCoverage : uint8_t { Full, Partial, None };

constexpr SeedCoverage classify_seed_coverage(size_t survived, size_t texels) {
    if (survived == 0) return SeedCoverage::Full;
    if (survived >= texels) return SeedCoverage::None;
    return SeedCoverage::Partial;
}

constexpr const char* seed_coverage_name(SeedCoverage cov) {
    switch (cov) {
        case SeedCoverage::Full: return "full-coverage (seed-skip proven)";
        case SeedCoverage::None: return "NONE-COVERAGE (untouched, setup/writeback skip proven)";
        case SeedCoverage::Partial: return "PARTIAL-COVERAGE (will always seed)";
    }
    return "unknown";
}

// Near-full coverage classification: permits up to 0.2% (1/500) unwritten texels for targets with >= 1000 texels.
// This accounts for small unwritten UI margins or minimap cutouts on 4K post-processing surfaces
// (e.g. ~16,000 unwritten margin texels on an ~8.3M 4K target is ~0.19%, just under the 0.2% bound).
constexpr bool classify_near_full_coverage(size_t survived, size_t texels) {
    return (survived == 0) || (texels >= 1000 && survived * 500 <= texels);
}

// Determines whether a cached seed verdict should participate in periodic re-proving.
// In addition to Full and None, any Partial verdict that carries an active optimization
// (a non-default layer mask or near-full coverage bypass) must periodically re-prove so
// that data-dependent stores changing which layers/texels are written do not remain stale
// beyond one reprove interval (#3328 B1/N2).
constexpr bool seed_verdict_reprove_eligible(SeedCoverage cov, uint64_t written_layers, bool near_full) {
    if (cov == SeedCoverage::Full || cov == SeedCoverage::None) return true;
    if (cov == SeedCoverage::Partial && (written_layers != ~0ULL || near_full)) return true;
    return false;
}

// Per-layer written mask for an array storage image, and whether that mask is EXACT.
//
// The mask is a 64-bit bitmap, so it cannot name a layer above 63. The first version of this
// computation fell back to `written_layers = (survived < texels) ? 1 : 0` for such resources -- bit 0
// used as a boolean -- and that is not a conservative approximation, it is a wrong mask: the retile
// and pack consumers read bit `layer`, so a fully-written 100-layer array had layers 1..63 skipped as
// "untouched" while layers 64..99 fell through the `layer < 64` guard and were written from scratch
// that is deliberately allocated without zero-fill. Silent guest-memory corruption, default-on.
//
// There is no honest per-layer answer above 64 layers, so this returns the NO-MASKING sentinel
// `~0ULL` -- the value every consumer already understands as "the optimization is off" -- and marks
// the result inexact. `exact` exists because a sentinel mask must never be mistaken for a proof:
// `~0ULL` would otherwise compare equal to "every layer written" and promote coverage to Full on a
// surface nothing was known about. The same applies when per-layer counts are unavailable
// (`layer_texels == 0`), which had the identical defect for any depth.
struct ArrayLayerCoverage {
    uint64_t written_layers = ~0ULL;   // ~0ULL == no masking
    bool any_written_partial = false;
    bool exact = false;                // false => the bits are a sentinel, not a measurement
};

inline ArrayLayerCoverage classify_array_layer_coverage(uint32_t depth,
                                                        const size_t* layer_survived,
                                                        size_t layer_texels,
                                                        size_t survived, size_t texels) {
    ArrayLayerCoverage out{};
    if (depth > 1 && depth <= 64 && layer_texels > 0 && layer_survived) {
        out.written_layers = 0;
        for (uint32_t l = 0; l < depth; ++l) {
            if (layer_survived[l] < layer_texels) {
                out.written_layers |= (1ULL << l);
                if (layer_survived[l] > 0) out.any_written_partial = true;
            }
        }
        out.exact = true;
        return out;
    }
    out.any_written_partial = (survived > 0 && survived < texels);
    if (depth > 1) {          // > 64 layers, or no per-layer counts: disable the optimization
        out.written_layers = ~0ULL;
        out.exact = false;
        return out;
    }
    out.written_layers = (survived < texels) ? 1ULL : 0ULL;   // depth == 1: one bit IS the truth
    out.exact = true;
    return out;
}

// Whether every layer of an array is known written. Requires an EXACT mask: an inexact one is the
// `~0ULL` sentinel, which would otherwise satisfy the all-ones comparison and promote a surface that
// was never measured per layer. `depth == 64` needs the explicit all-ones constant because
// `1ULL << 64` is undefined.
constexpr bool array_all_layers_written(uint32_t depth, uint64_t written_layers, bool exact) {
    if (!exact || depth <= 1 || depth > 64) return false;
    return written_layers == ((depth == 64) ? ~0ULL : ((1ULL << depth) - 1));
}

}  // namespace prosper::frontend
