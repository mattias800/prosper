// Which renderer-cache entries can hold an unpublished VOLUME footprint (#3873 plan item 2).
//
// Once any renderer-produced 3D target has existed, every sampled texture reference asks "does any
// volume target's outstanding guest footprint overlap this source?" -- and so do the compute
// unpublished-volume query and the live byte-range reader. Answering that by walking the whole RTT
// cache made the question cost O(cache size) per reference: on Grand Theft Auto V's gameplay route
// the walk alone was ~18% of the render thread (4-5 us on EVERY texture reference, hits included),
// although only a handful of entries are volumes.
//
// This index keeps the keys that MAY carry a nonzero footprint. It is a SUPERSET, and that is the
// whole contract:
//   * every site that can make an entry's footprint nonzero calls note(key) -- the renderer routes
//     all such writes through one helper, so the obligation lives in one place;
//   * a query looks each candidate up in the live cache and drops candidates that are gone or no
//     longer carry a footprint. Dropping is safe because a footprint can only reappear through a
//     site that notes it again.
// Because every candidate is re-read from the cache at query time, the answer is exactly the full
// walk's answer (the predicate is an `any`, so visit order cannot matter), never a cached verdict.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::frontend {

class VolumeTargetIndex {
public:
    void note(uint64_t base) {
        if (!base) return;
        if (std::find(bases_.begin(), bases_.end(), base) == bases_.end()) bases_.push_back(base);
    }

    // `visit(base, surface)` returns true to stop (an overlap was found). Only entries that exist in
    // `cache` with a nonzero `volume_guest_bytes` are visited. `lookups`, when given, counts cache
    // lookups -- the cost this index exists to bound.
    template <class Cache, class Visit>
    bool any_of(Cache& cache, Visit&& visit, size_t* lookups = nullptr) {
        for (size_t i = 0; i < bases_.size();) {
            if (lookups) ++*lookups;
            const auto it = cache.find(bases_[i]);
            if (it == cache.end() || !it->second.volume_guest_bytes) {
                bases_[i] = bases_.back();
                bases_.pop_back();
                continue;
            }
            if (visit(it->first, it->second)) return true;
            ++i;
        }
        return false;
    }

    bool contains(uint64_t base) const {
        return std::find(bases_.begin(), bases_.end(), base) != bases_.end();
    }
    size_t candidates() const { return bases_.size(); }
    void clear() { bases_.clear(); }

private:
    std::vector<uint64_t> bases_;
};

// The reference answer the index must reproduce: the full cache walk it replaces. Kept beside it so
// the same-binary control (PROSPER_NO_VOLUME_RTT_INDEX) and the audit mode share one definition.
template <class Cache, class Visit>
bool any_volume_target_full_scan(Cache& cache, Visit&& visit, size_t* visits = nullptr) {
    for (auto& entry : cache) {
        if (visits) ++*visits;
        if (!entry.second.volume_guest_bytes) continue;
        if (visit(entry.first, entry.second)) return true;
    }
    return false;
}

}  // namespace prosper::frontend
