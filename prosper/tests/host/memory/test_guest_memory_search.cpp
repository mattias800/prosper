// memory_search_ranges(): the chunking, overlap and de-duplication that a guest-memory content
// scan depends on, exercised against a synthetic address space so no live guest is needed.
//
// The case that matters is the STRADDLING one. A chunked scanner that forgets its overlap finds
// every needle except the ones that cross a chunk boundary, and on a real scan that is a silent
// partial blindness: the result is still a plausible-looking hit list, so nothing about the output
// says the scan was incomplete. It is exactly the shape that makes a "not found" wrong. Each case
// below therefore places a needle deliberately, and the fixture reports where it put it.
//
// The fetch callback counts its calls and can refuse a range, so the unreadable accounting and the
// budget cut-off are asserted too -- both produce an empty hit list, and a caller that cannot tell
// them apart from a genuine absence has no result at all.

#include "../../../src/host/memory/guest_memory_search.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper::host;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// A synthetic address space: one contiguous blob mapped at `base`.
struct FakeSpace {
    uint64_t base;
    std::vector<uint8_t> bytes;
    mutable int fetches = 0;
    mutable bool refuse = false;

    MemoryFetch fetch() const {
        return [this](uint64_t addr, uint8_t* out, size_t len) -> bool {
            ++fetches;
            if (refuse) return false;
            if (addr < base || addr + len > base + bytes.size()) return false;
            memcpy(out, bytes.data() + (addr - base), len);
            return true;
        };
    }
    std::vector<MemorySearchRange> ranges() const {
        return {{base, base + bytes.size()}};
    }
};

// Deterministic filler that never accidentally contains the needle: a byte stream with no 0xA5.
std::vector<uint8_t> filler(size_t n) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        uint8_t b = (uint8_t)((i * 31u + 7u) & 0xff);
        if (b == 0xA5) b = 0x00;
        v[i] = b;
    }
    return v;
}

std::vector<uint8_t> needle_bytes(size_t n) {
    std::vector<uint8_t> v(n);
    v[0] = 0xA5;
    for (size_t i = 1; i < n; ++i) v[i] = (uint8_t)(0xA5u ^ (i * 17u));
    return v;
}

}  // namespace

int main() {
    const size_t kChunk = 4u << 20;          // must match the implementation's chunk size
    const size_t kNeedle = 256;
    const std::vector<uint8_t> needle = needle_bytes(kNeedle);

    // --- 1. A needle in the middle of one chunk is found, once, with full=true ------------------
    {
        FakeSpace s{0x400000000ull, filler(kChunk * 3)};
        const size_t at = 1000;
        memcpy(s.bytes.data() + at, needle.data(), kNeedle);
        std::vector<MemorySearchHit> hits;
        MemorySearchScope sc = memory_search_ranges(s.ranges(), s.fetch(), needle.data(),
                                                    kNeedle, kNeedle, 0, 64, hits);
        check(hits.size() == 1, "mid-chunk: exactly one hit, got " + std::to_string(hits.size()));
        check(!hits.empty() && hits[0].addr == s.base + at, "mid-chunk: hit address");
        check(!hits.empty() && hits[0].full, "mid-chunk: full match");
        check(sc.ranges_scanned == 1 && sc.ranges_unreadable == 0, "mid-chunk: scope");
        check(!sc.budget_exhausted && !sc.hits_capped, "mid-chunk: no truncation");
    }

    // --- 2. THE STRADDLING CASE: a needle spanning a chunk boundary is still found, exactly once
    // A scanner without overlap reports zero here; one that overlaps but does not de-duplicate
    // reports two. Both are wrong in ways a live run cannot distinguish from the truth.
    {
        FakeSpace s{0x400000000ull, filler(kChunk * 2)};
        const size_t at = kChunk - kNeedle / 2;      // half in chunk 0, half in chunk 1
        memcpy(s.bytes.data() + at, needle.data(), kNeedle);
        std::vector<MemorySearchHit> hits;
        memory_search_ranges(s.ranges(), s.fetch(), needle.data(), kNeedle, kNeedle, 0, 64, hits);
        check(hits.size() == 1, "straddle: exactly one hit, got " + std::to_string(hits.size()));
        check(!hits.empty() && hits[0].addr == s.base + at, "straddle: hit address");
        check(!hits.empty() && hits[0].full, "straddle: full match");
    }

    // --- 3. A needle ending exactly at the last byte of the range is found ----------------------
    {
        FakeSpace s{0x400000000ull, filler(kChunk + 4096)};
        const size_t at = s.bytes.size() - kNeedle;
        memcpy(s.bytes.data() + at, needle.data(), kNeedle);
        std::vector<MemorySearchHit> hits;
        memory_search_ranges(s.ranges(), s.fetch(), needle.data(), kNeedle, kNeedle, 0, 64, hits);
        check(hits.size() == 1, "range-end: exactly one hit, got " + std::to_string(hits.size()));
        check(!hits.empty() && hits[0].addr == s.base + at, "range-end: hit address");
    }

    // --- 4. A prefix-only copy is reported as a hit with full=false -----------------------------
    // This is the half that makes a partial relocation visible: the needle's first 32 bytes are
    // present and the rest is not, which is what a block-granular re-tiling looks like.
    {
        FakeSpace s{0x400000000ull, filler(kChunk)};
        const size_t at = 4096;
        memcpy(s.bytes.data() + at, needle.data(), 32);
        std::vector<MemorySearchHit> hits;
        memory_search_ranges(s.ranges(), s.fetch(), needle.data(), 32, kNeedle, 0, 64, hits);
        check(hits.size() == 1, "prefix: exactly one hit, got " + std::to_string(hits.size()));
        check(!hits.empty() && !hits[0].full, "prefix: full=false");
        // And with prefix_len == needle_len the same space reports nothing at all, which is the
        // arm proving the prefix search is what found it rather than a looser comparison.
        std::vector<MemorySearchHit> strict;
        memory_search_ranges(s.ranges(), s.fetch(), needle.data(), kNeedle, kNeedle, 0, 64, strict);
        check(strict.empty(), "prefix: strict search finds nothing");
    }

    // --- 5. A needle that is absent is absent (the null this instrument exists to publish) ------
    {
        FakeSpace s{0x400000000ull, filler(kChunk * 2)};
        std::vector<MemorySearchHit> hits;
        MemorySearchScope sc = memory_search_ranges(s.ranges(), s.fetch(), needle.data(),
                                                    kNeedle, kNeedle, 0, 64, hits);
        check(hits.empty(), "absent: no hits");
        check(sc.bytes_scanned >= s.bytes.size(), "absent: the whole range was actually read");
        check(!sc.budget_exhausted, "absent: the null is not a budget artifact");
    }

    // --- 6. An unreadable range is accounted, never treated as zeros ----------------------------
    {
        FakeSpace s{0x400000000ull, filler(kChunk)};
        memcpy(s.bytes.data() + 128, needle.data(), kNeedle);
        s.refuse = true;
        std::vector<MemorySearchHit> hits;
        MemorySearchScope sc = memory_search_ranges(s.ranges(), s.fetch(), needle.data(),
                                                    kNeedle, kNeedle, 0, 64, hits);
        check(hits.empty(), "unreadable: no hits");
        check(sc.ranges_unreadable == 1, "unreadable: range counted once");
        check(sc.bytes_unreadable > 0, "unreadable: bytes counted");
        check(sc.bytes_scanned == 0, "unreadable: nothing counted as scanned");
    }

    // --- 7. The budget stops the scan and SAYS SO ----------------------------------------------
    {
        FakeSpace s{0x400000000ull, filler(kChunk * 4)};
        const size_t at = kChunk * 3 + 77;          // beyond the budget
        memcpy(s.bytes.data() + at, needle.data(), kNeedle);
        std::vector<MemorySearchHit> hits;
        MemorySearchScope sc = memory_search_ranges(s.ranges(), s.fetch(), needle.data(),
                                                    kNeedle, kNeedle, kChunk, 64, hits);
        check(hits.empty(), "budget: needle beyond the budget is not found");
        check(sc.budget_exhausted, "budget: the empty result is marked as truncated");
        // ...and the same space with no budget finds it, which is what makes the case above a
        // statement about the budget rather than about the needle.
        std::vector<MemorySearchHit> full;
        MemorySearchScope sc2 = memory_search_ranges(s.ranges(), s.fetch(), needle.data(),
                                                    kNeedle, kNeedle, 0, 64, full);
        check(full.size() == 1 && full[0].addr == s.base + at, "budget: unbounded scan finds it");
        check(!sc2.budget_exhausted, "budget: unbounded scan is not marked truncated");
    }

    // --- 8. Two copies are both reported, and max_hits caps visibly ----------------------------
    {
        FakeSpace s{0x400000000ull, filler(kChunk * 2)};
        memcpy(s.bytes.data() + 512, needle.data(), kNeedle);
        memcpy(s.bytes.data() + kChunk + 512, needle.data(), kNeedle);
        std::vector<MemorySearchHit> hits;
        memory_search_ranges(s.ranges(), s.fetch(), needle.data(), kNeedle, kNeedle, 0, 64, hits);
        check(hits.size() == 2, "two copies: both found, got " + std::to_string(hits.size()));

        std::vector<MemorySearchHit> capped;
        MemorySearchScope sc = memory_search_ranges(s.ranges(), s.fetch(), needle.data(),
                                                    kNeedle, kNeedle, 0, 1, capped);
        check(capped.size() == 1 && sc.hits_capped, "two copies: cap is reported");
    }

    // --- 9. A range shorter than the needle is skipped rather than mis-read --------------------
    {
        FakeSpace s{0x400000000ull, filler(64)};
        std::vector<MemorySearchHit> hits;
        MemorySearchScope sc = memory_search_ranges(s.ranges(), s.fetch(), needle.data(),
                                                    kNeedle, kNeedle, 0, 64, hits);
        check(hits.empty() && sc.ranges_scanned == 0, "short range: skipped");
        check(s.fetches == 0, "short range: never fetched");
    }

    if (g_failures) {
        fprintf(stderr, "guest_memory_search: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("guest_memory_search: all cases passed\n");
    return 0;
}
