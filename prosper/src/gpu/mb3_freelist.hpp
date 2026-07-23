// mb3_freelist.hpp -- non-trapping MallocBinned3 free-block membership (#312).
#pragma once

#include <cstdint>

namespace prosper::gpu {

struct Mb3FreelistMatch {
    uint64_t pool_base = 0;
    uint64_t head = 0;
    uint32_t hops = 0;
    uint8_t list = 0;          // 1/2 = TLS; 3..10 = global recycler; 11 = central free run
};

// MallocBinned3 keeps one 64 KiB-aligned per-thread pool-array in pthread TLS. The libkernel HLE
// feeds every aligned TLS value here; exact freelist traversal later distinguishes the real pool
// arrays from unrelated aligned TLS allocations without title-specific addresses.
void mb3_note_tls_pool_candidate(uint64_t base);
bool mb3_tls_tracking_enabled();

// A full per-thread bundle is transferred into one of eight lock-free global recycler slots. The
// bin address is learned from the guest allocator code; registering it extends membership beyond
// the two per-thread roots without trapping the allocator's hot free path.
void mb3_note_global_recycler_bin(uint64_t base);

// Is `block` currently free in size-class idx=1 (Malloc(0x20))? This checks the learned per-thread
// and global bundle chains plus the allocator's in-page central FFreeBlock runs. A positive result is
// observed twice so a concurrent allocator pop cannot turn a stale first read into a false verdict.
bool mb3_freelist_contains_stable(uint64_t block, Mb3FreelistMatch* match = nullptr);

// #1226 POOLSHIFT window scan: walk every learned TLS pool array's bin heads (both bundle heads
// per 0x20-stride size class) AND the eight global recycler slots, reporting any head whose value
// is a byte-shifted (>>8-encoded) pointer to MAPPED guest memory — the poisoned-bin signature.
// Called per submit (env-gated by the caller). LIMITS (do NOT read a found=0 as "no poison in this
// window"): it samples head RESIDENCE only — a poison buried in a free-chain INTERIOR (below a
// healthy head) is invisible, and a head that only surfaces in a microsecond sliver on the
// allocator hot path can fall between two per-submit samples. So a state change gives a WEAK bound
// (the poison was at a scanned head between these submits), never the free()-to-fault lifetime; the
// mapped-target requirement additionally drops a poisoned head whose <<8 target was unmapped by
// scan time. The watchpoint path (PROSPER_MB3WATCH) is what actually traps the store. Fault-safe;
// returns the hit count and formats up to `cap` bytes into `out` (NUL-terminated) when out non-null.
int mb3_poolshift_window_scan(char* out, unsigned cap);

// Test isolation for the process-global candidate registry.
void mb3_reset_pool_candidates_for_test();

} // namespace prosper::gpu
