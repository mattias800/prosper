// memory_search_ranges(): the chunking, overlap and range-end handling that a guest-memory content
// scan depends on, exercised against a synthetic address space so no live guest is needed.
//
// The case that matters is the STRADDLING one. A chunked scanner that forgets its overlap finds
// every needle except the ones that cross a chunk boundary, and on a real scan that is a silent
// partial blindness: the result is still a plausible-looking hit list, so nothing about the output
// says the scan was incomplete. It is exactly the shape that makes a "not found" wrong.
//
// That case is therefore built against `memory_search_chunk_bytes()` rather than a copied constant,
// and it ASSERTS that the placement really straddles. Review of #3243 measured why: with the chunk
// size raised to 8 MiB and a hard-coded 4 MiB in the test, the needle lands in the middle of chunk 0,
// the case silently becomes a duplicate of the mid-chunk one, and the overlap can then be deleted
// outright with the suite still green.
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
    // NOT a copied constant: see the header note. A straddle case that stops straddling still
    // passes, and then nothing tests the overlap at all.
    const size_t kChunk = memory_search_chunk_bytes();
    const size_t kNeedle = 256;
    const std::vector<uint8_t> needle = needle_bytes(kNeedle);
    check(kChunk > kNeedle * 4, "fixture: the chunk is large enough for these placements");

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
        check(!sc.enumeration_failed, "mid-chunk: enumeration not flagged");
    }

    // --- 2. THE STRADDLING CASE: a needle spanning a chunk boundary is still found, exactly once
    // A scanner without overlap reports zero here. The placement is asserted to straddle, so this
    // case cannot quietly degrade into a second copy of case 1 if the chunk size changes.
    {
        FakeSpace s{0x400000000ull, filler(kChunk * 2)};
        const size_t at = kChunk - kNeedle / 2;      // half in chunk 0, half in chunk 1
        check(at < kChunk && at + kNeedle > kChunk, "straddle: the placement really straddles");
        memcpy(s.bytes.data() + at, needle.data(), kNeedle);
        std::vector<MemorySearchHit> hits;
        memory_search_ranges(s.ranges(), s.fetch(), needle.data(), kNeedle, kNeedle, 0, 64, hits);
        check(hits.size() == 1, "straddle: exactly one hit, got " + std::to_string(hits.size()));
        check(!hits.empty() && hits[0].addr == s.base + at, "straddle: hit address");
        check(!hits.empty() && hits[0].full, "straddle: full match");
    }

    // --- 2b. Every placement across the boundary, not just one --------------------------------
    // One placement can be right by luck; the whole crossing window cannot. Each offset is asserted
    // to be found exactly once at the right address.
    //
    // It does NOT test the `accept_from` de-duplication, and saying it did was wrong: with the
    // bounds as written the searched start windows of consecutive chunks are contiguous and
    // disjoint, so no placement can be reported twice and removing the guard leaves this sweep
    // green. Measured in review of #3243. The guard is an invariant against a future change to
    // `limit` or `advance`, not a fix for a hazard this arithmetic can produce -- and it is
    // deliberately unpinned, because a test that could redden without it would have to introduce
    // the overlap the design excludes.
    {
        const size_t lo = kChunk - kNeedle - 2, hi = kChunk + 2;
        for (size_t at = lo; at <= hi; ++at) {
            FakeSpace s{0x400000000ull, filler(kChunk * 2)};
            memcpy(s.bytes.data() + at, needle.data(), kNeedle);
            std::vector<MemorySearchHit> hits;
            memory_search_ranges(s.ranges(), s.fetch(), needle.data(), kNeedle, kNeedle, 0, 64, hits);
            if (hits.size() != 1 || hits[0].addr != s.base + at || !hits[0].full) {
                check(false, "boundary sweep at +" + std::to_string(at) + ": hits=" +
                                 std::to_string(hits.size()));
                break;
            }
        }
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

    // --- 3b. A needle at offset ZERO of a range ------------------------------------------------
    {
        FakeSpace s{0x400000000ull, filler(kChunk / 2)};
        memcpy(s.bytes.data(), needle.data(), kNeedle);
        std::vector<MemorySearchHit> hits;
        memory_search_ranges(s.ranges(), s.fetch(), needle.data(), kNeedle, kNeedle, 0, 64, hits);
        check(hits.size() == 1 && hits[0].addr == s.base, "range-start: found at offset 0");
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

    // --- 4b. PREFIX MODE AT A RANGE END, the case a needle-length bound silently loses ---------
    // In prefix mode a hit needs `prefix_len` bytes, not `needle_len`. A scanner that bounds its
    // last chunk by the needle cannot see the final needle_len-1 bytes of any range -- 255 bytes per
    // mapping, invisible, and contradicting what the header says a match is. Every placement in the
    // window is swept, and each must be reported with full=false since the needle cannot fit.
    {
        const size_t kPrefix = 16;
        FakeSpace s{0x400000000ull, filler(kChunk / 4)};
        const size_t end = s.bytes.size();
        for (size_t at = end - kNeedle + 1; at + kPrefix <= end; ++at) {
            std::vector<uint8_t> saved(s.bytes.begin() + (long)at, s.bytes.end());
            memcpy(s.bytes.data() + at, needle.data(), std::min(kNeedle, end - at));
            std::vector<MemorySearchHit> hits;
            memory_search_ranges(s.ranges(), s.fetch(), needle.data(), kPrefix, kNeedle, 0, 64, hits);
            bool ok = hits.size() == 1 && hits[0].addr == s.base + at && !hits[0].full;
            memcpy(s.bytes.data() + at, saved.data(), saved.size());
            if (!ok) {
                check(false, "prefix-at-range-end at +" + std::to_string(at) + ": hits=" +
                                 std::to_string(hits.size()));
                break;
            }
        }
        // The `full` flag must require the needle to be PRESENT, not merely to compare equal
        // against whatever the fetch buffer happens to hold beyond the range. This arm pins that:
        // the needle's tail is all zeros and the fetch buffer is zero-initialised, so a full-needle
        // memcmp that ran past `want` would SUCCEED and report full=true. Only the
        // `off + needle_len <= want` guard makes the answer false here.
        {
            std::vector<uint8_t> zero_tail = needle;
            for (size_t k = kPrefix; k < kNeedle; ++k) zero_tail[k] = 0;
            FakeSpace z{0x600000000ull, filler(kChunk / 8)};
            const size_t at = z.bytes.size() - kPrefix;      // only the prefix fits
            memcpy(z.bytes.data() + at, zero_tail.data(), kPrefix);
            std::vector<MemorySearchHit> zh;
            memory_search_ranges(z.ranges(), z.fetch(), zero_tail.data(), kPrefix, kNeedle, 0, 64, zh);
            check(zh.size() == 1 && zh[0].addr == z.base + at, "zero-tail: prefix hit found");
            check(!zh.empty() && !zh[0].full,
                  "zero-tail: full=false -- the needle does not FIT, whatever the buffer holds");
        }
        // ...and a range too short for the whole needle but long enough for the prefix is still
        // scanned rather than skipped.
        FakeSpace tiny{0x500000000ull, filler(kNeedle / 2)};
        memcpy(tiny.bytes.data() + 8, needle.data(), kPrefix);
        std::vector<MemorySearchHit> th;
        MemorySearchScope tsc = memory_search_ranges(tiny.ranges(), tiny.fetch(), needle.data(),
                                                     kPrefix, kNeedle, 0, 64, th);
        check(th.size() == 1 && th[0].addr == tiny.base + 8 && !th[0].full,
              "short range: prefix hit found");
        check(tsc.ranges_scanned == 1, "short range: counted as scanned");
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

    // --- 7b. A range the budget abandons before its first fetch is not counted as scanned ------
    {
        FakeSpace a{0x400000000ull, filler(kChunk)};
        FakeSpace b{0x500000000ull, filler(kChunk)};
        memcpy(b.bytes.data() + 64, needle.data(), kNeedle);
        std::vector<MemorySearchRange> two{{a.base, a.base + a.bytes.size()},
                                           {b.base, b.base + b.bytes.size()}};
        const MemoryFetch fetch = [&](uint64_t addr, uint8_t* out, size_t len) -> bool {
            const FakeSpace& s = addr >= b.base ? b : a;
            if (addr < s.base || addr + len > s.base + s.bytes.size()) return false;
            memcpy(out, s.bytes.data() + (addr - s.base), len);
            return true;
        };
        std::vector<MemorySearchHit> hits;
        MemorySearchScope sc = memory_search_ranges(two, fetch, needle.data(), kNeedle, kNeedle,
                                                    /*byte_budget=*/1, 64, hits);
        check(hits.empty() && sc.budget_exhausted, "budget/2-ranges: stopped and said so");
        check(sc.ranges_scanned == 1,
              "budget/2-ranges: only the range actually read is counted, got " +
                  std::to_string(sc.ranges_scanned));
        // Unbounded, both ranges are scanned and the second range's needle is found -- which is what
        // makes the count above a statement about the budget rather than about the range list.
        std::vector<MemorySearchHit> all;
        MemorySearchScope sc2 = memory_search_ranges(two, fetch, needle.data(), kNeedle, kNeedle,
                                                     0, 64, all);
        check(all.size() == 1 && all[0].addr == b.base + 64, "2-ranges: second range is searched");
        check(sc2.ranges_scanned == 2, "2-ranges: both counted");
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

    // --- 9. A range shorter than the PREFIX is skipped rather than mis-read --------------------
    {
        FakeSpace s{0x400000000ull, filler(8)};
        std::vector<MemorySearchHit> hits;
        MemorySearchScope sc = memory_search_ranges(s.ranges(), s.fetch(), needle.data(),
                                                    16, kNeedle, 0, 64, hits);
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
