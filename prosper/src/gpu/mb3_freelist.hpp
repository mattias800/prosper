// mb3_freelist.hpp -- non-trapping MallocBinned3 free-block membership (#312).
#pragma once

#include <cstdint>

namespace prosper::gpu {

struct Mb3FreelistMatch {
    uint64_t pool_base = 0;
    uint64_t head = 0;
    uint32_t hops = 0;
    uint8_t list = 0;          // 1 = primary head, 2 = secondary head
};

// MallocBinned3 keeps one 64 KiB-aligned per-thread pool-array in pthread TLS. The libkernel HLE
// feeds every aligned TLS value here; exact freelist traversal later distinguishes the real pool
// arrays from unrelated aligned TLS allocations without title-specific addresses.
void mb3_note_tls_pool_candidate(uint64_t base);
bool mb3_tls_tracking_enabled();

// Is `block` currently linked into size-class idx=1 (Malloc(0x20)) in any learned per-thread cache?
// A positive result is observed twice so a concurrent allocator pop cannot turn a stale first read
// into a false "free" verdict. False negatives merely leave the existing #505/#510 guards in place.
bool mb3_freelist_contains_stable(uint64_t block, Mb3FreelistMatch* match = nullptr);

// Test isolation for the process-global candidate registry.
void mb3_reset_pool_candidates_for_test();

} // namespace prosper::gpu
