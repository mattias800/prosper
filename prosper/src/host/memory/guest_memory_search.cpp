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
// already streaming gigabytes.
constexpr size_t kChunkBytes = 4u << 20;

}  // namespace

MemorySearchScope memory_search_ranges(const std::vector<MemorySearchRange>& ranges,
                                       const MemoryFetch& fetch,
                                       const uint8_t* needle, size_t prefix_len, size_t needle_len,
                                       uint64_t byte_budget, size_t max_hits,
                                       std::vector<MemorySearchHit>& out) {
    MemorySearchScope scope;
    if (!needle || !prefix_len || prefix_len > needle_len || !max_hits) return scope;

    // The overlap is the FULL needle minus one, not the prefix minus one. With advance =
    // chunk - overlap, every hit accepted inside a chunk has its whole needle inside that same
    // chunk, so the full-match test never needs a second fetch and never silently degrades to a
    // prefix-only answer at a chunk boundary.
    const size_t overlap = needle_len - 1;
    const size_t chunk = std::max(kChunkBytes, needle_len * 2);
    const size_t advance = chunk - overlap;

    std::vector<uint8_t> buf(chunk);

    for (const MemorySearchRange& r : ranges) {
        if (r.begin >= r.end) continue;
        if (r.end - r.begin < needle_len) continue;
        ++scope.ranges_scanned;
        bool range_unreadable = false;
        // Matches are found in ascending address order, so a single high-water mark de-duplicates
        // the overlap region without a set: a repeat of an already-reported hit necessarily has a
        // lower address than the mark.
        uint64_t accept_from = r.begin;
        for (uint64_t pos = r.begin; pos < r.end;) {
            if (byte_budget && scope.bytes_scanned >= byte_budget) {
                scope.budget_exhausted = true;
                return scope;
            }
            const size_t want = (size_t)std::min<uint64_t>(chunk, r.end - pos);
            if (want < needle_len) break;
            if (!fetch(pos, buf.data(), want)) {
                scope.bytes_unreadable += want;
                if (!range_unreadable) { range_unreadable = true; ++scope.ranges_unreadable; }
                pos += advance;
                continue;
            }
            scope.bytes_scanned += want;
            // Search only where a FULL needle still fits; the overlap on the next chunk covers the
            // tail. On the last chunk of a range there is no next chunk, and the bound is the same
            // one -- a needle that would run past the range end is not present in the range.
            const size_t limit = want - needle_len;
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
                        h.full = memcmp(hit, needle, needle_len) == 0;
                        out.push_back(h);
                        accept_from = addr + 1;
                        if (out.size() >= max_hits) { scope.hits_capped = true; return scope; }
                    }
                }
                i = off + 1;
            }
            // A short chunk is the tail of the range: everything left has been searched, and
            // advancing by `advance` would re-scan it (harmlessly, thanks to `accept_from`, but at
            // full cost) until the remainder fell below the needle length.
            if (want < chunk) break;
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
    if (FILE* f = fopen("/proc/self/maps", "r")) {
        char line[512];
        while (fgets(line, sizeof line, f)) {
            unsigned long long b = 0, e = 0;
            char perms[8] = {0};
            if (sscanf(line, "%llx-%llx %7s", &b, &e, perms) != 3) continue;
            if (perms[0] != 'r') continue;
            uint64_t begin = std::max<uint64_t>(b, lo);
            uint64_t end = std::min<uint64_t>(e, hi);
            if (begin >= end) continue;
            // Excluding the caller's own buffer is what keeps the control arm honest: without it a
            // scan for "is this payload anywhere else" would always succeed by finding the copy the
            // scanner was handed.
            if (skip_begin < skip_end && begin < skip_end && end > skip_begin) {
                if (begin < skip_begin) ranges.push_back({begin, skip_begin});
                if (end > skip_end) ranges.push_back({skip_end, end});
                continue;
            }
            ranges.push_back({begin, end});
        }
        fclose(f);
    }
    const MemoryFetch fetch = [](uint64_t addr, uint8_t* dst, size_t len) -> bool {
        struct iovec l { dst, len }, r { (void*)(uintptr_t)addr, len };
        return process_vm_readv(getpid(), &l, 1, &r, 1, 0) == (ssize_t)len;
    };
    return memory_search_ranges(ranges, fetch, needle, prefix_len, needle_len,
                                byte_budget, max_hits, out);
}

#endif  // !_WIN32

}  // namespace prosper::host
