#pragma once

// Search the guest's own address space for a KNOWN byte pattern.
//
// Why this exists. Several investigations reach the same shape: prosper knows exactly what an asset
// looked like when it was delivered (the APR reader holds the source buffer), the address it was
// delivered to later reads as something else, and the question becomes "did the guest MOVE those
// bytes somewhere, or did it consume them into a form that no longer contains them?". Both answers
// are actionable and they need different fixes -- a relocation means prosper is reading a stale
// descriptor, an absence means the engine transformed the data and prosper missed the operation
// that did it -- so an instrument that separates them is worth more than another two-point sample
// of the address that went quiet. #3142 (Stray) is the case it was written for.
//
// The scan is split in two so the interesting half is testable without a live guest:
//   * memory_search_ranges() is pure -- it takes the ranges, a fetch callback and the needle, and
//     owns the chunking, the overlap that keeps a needle straddling a chunk boundary from being
//     missed, the hit de-duplication that overlap otherwise creates, and the budget accounting.
//   * guest_memory_search() is the thin POSIX half: enumerate this process's readable mappings from
//     /proc/self/maps and read them with process_vm_readv (never a raw dereference -- a mapping can
//     disappear under a diagnostic, and a diagnostic must not fault the run).
//
// EVERY RESULT CARRIES ITS OWN SCOPE. A "not found" from a scanner is worthless without knowing how
// much was actually looked at, and the two failure modes -- a budget that stopped the scan early and
// a range that could not be read -- both produce exactly the same empty hit list as a real absence.
// MemorySearchScope reports both, and callers are expected to print it beside the verdict.
//
// WHAT IT STRUCTURALLY CANNOT SEE, stated here rather than left to be discovered: a byte-identical
// copy. If the guest re-tiled, swizzled, decompressed or otherwise transformed the payload, the
// bytes are no longer the same bytes and no needle length will find them. An empty result therefore
// means "no verbatim copy in the scanned ranges", never "the data does not exist".

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace prosper::host {

struct MemorySearchRange {
    uint64_t begin = 0;
    uint64_t end = 0;
};

struct MemorySearchHit {
    uint64_t addr = 0;
    // The whole needle matched here, not merely its prefix. Reporting the two separately is what
    // makes a partial relocation visible: a block-granular re-tiling can preserve a short run and
    // break a long one, and a scanner that only ever reports full matches calls that "absent".
    bool full = false;
};

struct MemorySearchScope {
    uint64_t bytes_scanned = 0;
    uint64_t bytes_unreadable = 0;
    uint32_t ranges_scanned = 0;
    uint32_t ranges_unreadable = 0;
    bool budget_exhausted = false;   // the scan stopped early: an empty result proves nothing
    bool hits_capped = false;        // more hits exist than were reported
};

// Read [addr, addr+len) into `out`. Returns false if the range cannot be read, in which case the
// chunk is accounted as unreadable and skipped -- it is never treated as zeros.
using MemoryFetch = std::function<bool(uint64_t addr, uint8_t* out, size_t len)>;

// Scan `ranges` for `needle`. A location matches when its first `prefix_len` bytes equal the
// needle's; `full` is set on the hit when all `needle_len` bytes match too. `prefix_len` must be
// >= 1 and <= `needle_len`.
//
// `byte_budget` bounds the work (0 = unbounded). `max_hits` bounds the output; when it is reached
// the scan stops and `hits_capped` is set.
MemorySearchScope memory_search_ranges(const std::vector<MemorySearchRange>& ranges,
                                       const MemoryFetch& fetch,
                                       const uint8_t* needle, size_t prefix_len, size_t needle_len,
                                       uint64_t byte_budget, size_t max_hits,
                                       std::vector<MemorySearchHit>& out);

#ifndef _WIN32
// Enumerate this process's readable mappings intersected with [lo, hi) and scan them. `skip_begin`
// / `skip_end` excludes one range from the scan -- the caller's own buffer, so a hit is never the
// needle finding itself.
MemorySearchScope guest_memory_search(uint64_t lo, uint64_t hi,
                                      uint64_t skip_begin, uint64_t skip_end,
                                      const uint8_t* needle, size_t prefix_len, size_t needle_len,
                                      uint64_t byte_budget, size_t max_hits,
                                      std::vector<MemorySearchHit>& out);
#endif

}  // namespace prosper::host
