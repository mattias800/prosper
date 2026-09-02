#include "guest_memory_search.hpp"

#include <cstdio>
#include <cstring>
#include <algorithm>

#ifndef _WIN32
#include <sys/uio.h>
#include <unistd.h>
// Darwin has no process_vm_readv; the shim supplies it (and is a no-op on Linux).
#include "host/platform/posix_shim.hpp"
#endif

namespace prosper::host {

namespace {

// 4 MiB per fetch. Large enough that the per-call overhead of process_vm_readv disappears against
// the copy, small enough that the transient buffer is not itself a memory event on a title that is
// already streaming gigabytes. Reachable through memory_search_chunk_bytes() so the regression test
// can place a needle on a real boundary instead of on a hard-coded guess at this number.
constexpr size_t kChunkBytes = 4u << 20;

}  // namespace

size_t memory_search_chunk_bytes() { return kChunkBytes; }

MemorySearchScope memory_search_ranges(const std::vector<MemorySearchRange>& ranges,
                                       const MemoryFetch& fetch,
                                       const uint8_t* needle, size_t prefix_len, size_t needle_len,
                                       uint64_t byte_budget, size_t max_hits,
                                       std::vector<MemorySearchHit>& out) {
    MemorySearchScope scope;
    if (!needle || !prefix_len || prefix_len > needle_len || !max_hits) return scope;

    // The overlap is the FULL needle minus one, not the prefix minus one. With advance =
    // chunk - overlap, every hit accepted in an INTERIOR chunk has its whole needle inside that same
    // chunk, so the full-match test never needs a second fetch and never silently degrades to a
    // prefix-only answer at a chunk boundary.
    const size_t overlap = needle_len - 1;
    const size_t chunk = std::max(kChunkBytes, needle_len * 2);
    const size_t advance = chunk - overlap;

    std::vector<uint8_t> buf(chunk);

    for (const MemorySearchRange& r : ranges) {
        if (r.begin >= r.end) continue;
        // The bound is the PREFIX, not the needle: a prefix match is a hit by contract, so a range
        // too short for the whole needle can still contain one. Testing needle_len here made the
        // header's own definition of a match untrue at the small end.
        if (r.end - r.begin < prefix_len) continue;
        bool range_counted = false, range_unreadable = false;
        // Matches are found in ascending address order, so a single high-water mark would
        // de-duplicate an overlap region without a set. As the arithmetic stands it can never fire:
        // an interior chunk k searches starts up to pos_k + want - needle_len and chunk k+1 begins
        // at exactly pos_k + chunk - needle_len + 1, so the searched START windows are contiguous
        // and disjoint, and the last chunk only ever extends FORWARD of them. Kept as a cheap
        // invariant so a future change to `limit` or `advance` cannot silently begin reporting the
        // same address twice -- not because the present design needs it.
        uint64_t accept_from = r.begin;
        for (uint64_t pos = r.begin; pos < r.end;) {
            if (byte_budget && scope.bytes_scanned >= byte_budget) {
                scope.budget_exhausted = true;
                return scope;
            }
            const size_t want = (size_t)std::min<uint64_t>(chunk, r.end - pos);
            const bool last = (pos + want >= r.end);
            // An interior chunk searches only where a FULL needle fits, because everything past that
            // is covered by the next chunk's overlap. The LAST chunk has no next chunk, so it
            // searches everywhere the PREFIX fits -- otherwise the final needle_len-1 bytes of every
            // range are unsearchable in prefix mode, which contradicts what a hit is defined to be.
            const size_t need = last ? prefix_len : needle_len;
            if (want < need) break;
            // Counted here rather than at the top of the range, so a range the budget abandons
            // before reading a single byte is not reported as one the scan looked at.
            if (!range_counted) { range_counted = true; ++scope.ranges_scanned; }
            if (!fetch(pos, buf.data(), want)) {
                scope.bytes_unreadable += want;
                if (!range_unreadable) { range_unreadable = true; ++scope.ranges_unreadable; }
                if (last) break;
                pos += advance;
                continue;
            }
            scope.bytes_scanned += want;
            const size_t limit = want - need;
            for (size_t i = 0; i <= limit;) {
                const uint8_t* hit = (const uint8_t*)memchr(buf.data() + i, needle[0], want - i);
                if (!hit) break;
                const size_t off = (size_t)(hit - buf.data());
                if (off > limit) break;
                if (memcmp(hit, needle, prefix_len) == 0) {
                    const uint64_t addr = pos + off;
                    if (addr >= accept_from) {
                        MemorySearchHit h;
                        h.addr = addr;
                        // `full` needs the whole needle to be PRESENT as well as equal. At a range
                        // end it may not be, and claiming a full match over bytes that were never
                        // read is the one answer this instrument must never give. It is also a
                        // BOUNDS guard, not only a correctness one: when a range's last chunk is
                        // exactly `chunk` bytes, an unguarded memcmp of needle_len from `off` reads
                        // past the end of `buf`.
                        h.full = off + needle_len <= want &&
                                 memcmp(hit, needle, needle_len) == 0;
                        out.push_back(h);
                        accept_from = addr + 1;
                        if (out.size() >= max_hits) { scope.hits_capped = true; return scope; }
                    }
                }
                i = off + 1;
            }
            // The last chunk of a range: everything reachable has been searched, and advancing would
            // re-scan it (harmlessly, thanks to `accept_from`, but at full cost) until the remainder
            // fell below the needle length.
            if (last) break;
            pos += advance;
        }
    }
    return scope;
}

#ifndef _WIN32

MemorySearchScope guest_memory_search(uint64_t lo, uint64_t hi,
                                      uint64_t skip_begin, uint64_t skip_end,
                                      const uint8_t* needle, size_t prefix_len, size_t needle_len,
                                      uint64_t byte_budget, size_t max_hits,
                                      std::vector<MemorySearchHit>& out) {
    std::vector<MemorySearchRange> ranges;
    FILE* f = fopen("/proc/self/maps", "r");
    if (!f) {
        // With no ranges the scan below would report zero of everything and read as a thorough null.
        // Say so in the scope AND in the log: this is the unconditional outcome on a host with no
        // /proc, which this file is now compiled for.
        MemorySearchScope failed;
        failed.enumeration_failed = true;
        fprintf(stderr, "[memsearch] /proc/self/maps could not be opened -- NO ranges enumerated, "
                        "so this scan looked at nothing and any \"not found\" from it is void.\n");
        return failed;
    }
    char line[512];
    while (fgets(line, sizeof line, f)) {
        unsigned long long b = 0, e = 0;
        char perms[8] = {0};
        if (sscanf(line, "%llx-%llx %7s", &b, &e, perms) != 3) continue;
        if (perms[0] != 'r') continue;
        uint64_t begin = std::max<uint64_t>(b, lo);
        uint64_t end = std::min<uint64_t>(e, hi);
        if (begin >= end) continue;
        // Excluding the caller's own buffer is what keeps a control arm honest: without it a scan
        // for "is this payload anywhere else" would always succeed by finding the copy the scanner
        // was handed.
        if (skip_begin < skip_end && begin < skip_end && end > skip_begin) {
            if (begin < skip_begin) ranges.push_back({begin, skip_begin});
            if (end > skip_end) ranges.push_back({skip_end, end});
            continue;
        }
        ranges.push_back({begin, end});
    }
    fclose(f);
    const MemoryFetch fetch = [](uint64_t addr, uint8_t* dst, size_t len) -> bool {
        struct iovec l { dst, len }, r { (void*)(uintptr_t)addr, len };
        return process_vm_readv(getpid(), &l, 1, &r, 1, 0) == (ssize_t)len;
    };
    MemorySearchScope sc = memory_search_ranges(ranges, fetch, needle, prefix_len, needle_len,
                                                byte_budget, max_hits, out);
    // An empty enumeration is not the same as an empty result: /proc parsed but yielded nothing
    // inside [lo, hi) is still "nothing was looked at".
    if (ranges.empty()) sc.enumeration_failed = true;
    return sc;
}

#endif  // !_WIN32

}  // namespace prosper::host
