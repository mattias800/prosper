// VolumeTargetIndex must answer exactly what the full RTT-cache walk answers, while visiting only
// volume candidates (#3873 plan item 2). The cache here is a stand-in with the one field the index
// reads, so the test needs no device and no renderer.
#include "shared/rtt/volume_target_index.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <unordered_map>

namespace {

struct Surf {
    uint64_t volume_guest_bytes = 0;
};
using Cache = std::unordered_map<uint64_t, Surf>;

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", what);
    failures += !ok;
}

bool overlaps(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) { return a < b + bn && b < a + an; }

}  // namespace

int main() {
    using prosper::frontend::VolumeTargetIndex;
    using prosper::frontend::any_volume_target_full_scan;

    // A GTA-V-shaped cache: thousands of 2D targets, a couple of renderer-produced volumes.
    Cache cache;
    VolumeTargetIndex index;
    constexpr uint64_t kStride = 0x100000;
    for (uint64_t i = 0; i < 10000; ++i) cache[0x200000000ull + i * kStride] = {};
    const uint64_t vol_a = 0x200000000ull + 17 * kStride;
    const uint64_t vol_b = 0x200000000ull + 9000 * kStride;
    cache[vol_a].volume_guest_bytes = 4 * kStride;    // spans the next three 2D bases
    cache[vol_b].volume_guest_bytes = kStride / 2;
    index.note(vol_a);
    index.note(vol_b);
    index.note(vol_a);   // idempotent

    // 1. Exactness over many queries, including ones that start inside a volume (interior alias),
    //    ones at the volume's own base (excluded by the caller's predicate) and ones far away.
    std::mt19937_64 rng(3873);
    size_t mismatches = 0, positives = 0;
    for (int q = 0; q < 20000; ++q) {
        const uint64_t addr = 0x200000000ull + (rng() % (10002 * kStride));
        const uint64_t bytes = 1 + rng() % (2 * kStride);
        auto visit = [&](uint64_t base, const Surf& s) {
            return base != addr && overlaps(base, s.volume_guest_bytes, addr, bytes);
        };
        const bool expected = any_volume_target_full_scan(cache, visit);
        const bool got = index.any_of(cache, visit);
        mismatches += expected != got;
        positives += got;
    }
    check(mismatches == 0, "indexed answer equals the full walk on 20000 queries");
    check(positives > 0, "the query mix reaches the positive (overlap) case");

    // 2. Cost: the index looks up its candidates only; the walk visits every entry.
    size_t lookups = 0, visits = 0;
    auto never = [](uint64_t, const Surf&) { return false; };
    index.any_of(cache, never, &lookups);
    any_volume_target_full_scan(cache, never, &visits);
    std::printf("      index lookups=%zu, full walk visits=%zu\n", lookups, visits);
    check(lookups == 2, "index visits only the two volume candidates");
    check(visits == cache.size(), "control: the full walk visits the whole cache");

    // 3. A candidate whose footprint is gone (erased, or reset to a 2D target) is dropped, and the
    //    answer still matches the walk.
    cache.erase(vol_b);
    auto hits_b = [&](uint64_t base, const Surf& s) {
        return overlaps(base, s.volume_guest_bytes, vol_b, 1);
    };
    check(!index.any_of(cache, hits_b) && !any_volume_target_full_scan(cache, hits_b),
          "an erased volume no longer answers");
    check(index.candidates() == 1, "the erased candidate was pruned");
    cache[vol_a].volume_guest_bytes = 0;
    check(!index.any_of(cache, never) && index.candidates() == 0,
          "a footprint reset to zero prunes its candidate");

    // 4. The superset contract: a footprint that reappears must be noted again, and then answers.
    cache[vol_a].volume_guest_bytes = kStride;
    auto hits_a = [&](uint64_t base, const Surf& s) {
        return overlaps(base, s.volume_guest_bytes, vol_a + 16, 16);
    };
    check(any_volume_target_full_scan(cache, hits_a), "control: the walk sees the new footprint");
    check(!index.any_of(cache, hits_a),
          "negative control: an UN-noted footprint is invisible to the index (why every writer notes)");
    index.note(vol_a);
    check(index.any_of(cache, hits_a), "re-noted footprint answers again");

    // 5. Key zero is never a target.
    index.note(0);
    check(!index.contains(0), "address zero is never indexed");

    return failures ? 1 : 0;
}
