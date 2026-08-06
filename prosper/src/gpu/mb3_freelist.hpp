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

// #1226 POSITIVE CONTROL for mb3_freelist_contains_stable(). A `false` from that walk is only
// evidence if the walk can return `true` at all in this run — and it silently cannot when the
// pool-candidate registry is empty, when the recycler bin was never discovered, or when the
// allocator layout has drifted. A learned bin HEAD is by construction the first node of its own
// chain, so asking membership about a head must answer yes; and because a head matches at hop 0,
// each head's SUCCESSOR is probed too, since only traversal can reach an interior node.
// Reports `pools/probes/positives/deep/deep_positives` into `out` and names the three failure
// modes inline: `NO HEADS` (nothing to answer about), `BLIND` (a head is not in its own chain),
// `SHALLOW` (heads found, no interior node reached).
// RETURNS the interior-node count when traversal was testable (deep > 0), else the head count —
// i.e. the strongest property actually demonstrated. Zero always means "do not trust a null from
// this walk right now". Fault-safe and read-only.
//
// LIMIT: this controls the WALK, not the CALL SITE. A caller that early-returns before its own
// probe for exactly the member==true population (as honor_dma_data does under
// PROSPER_MB3_FREELIST_GUARD) will still see 100% member=0 beside a green self-test.
int mb3_freelist_selftest(char* out, unsigned cap);

// #1945 INTERIOR poison scan — what mb3_poolshift_window_scan structurally cannot see.
//
// The terminal fault of this family is a bundle-list POP that dereferences a head which is not an
// FBundleNode. By then the poison has already been copied out of the chain into the bin, so the
// node it came from is gone from every register and every log. This walks each learned per-thread
// pool array's size-class-1 bundle chains (both bundle heads; see SCOPE below) up to `max_hops`
// nodes deep and reports any node whose `next` link is NOT a plausible FBundleNode — i.e. the poison while it is
// still IN the chain, together with the address of the node that holds it.
//
// That node address is the whole point: it is the exact guest block a prosper label write would
// have had to target to be the author, so the caller can put it straight to the label-history
// table. Unlike a value-shape test, the predicate is structural — a head/next slot may only hold 0
// or a 0x20-aligned pointer into mapped guest memory — so it cannot be fooled by a poison that
// happens not to look byte-shifted (measured on PPSA07809: 0xff000000ff000000, 0x0002400100024001).
//
// SCOPE: size class **idx=1 only** (bin offset 0x20 — the 32-byte class the consumed-marker labels
// are allocated from, and the only class ever observed corrupt in this family). The other classes
// share the bin shape but not the node alignment, so the structural rule below would report their
// healthy heads as poison; widening this needs each class's block size read out of the guest first.
//
// The walk is performed TWICE and only hits seen by both passes are reported — the same contract
// `mb3_freelist_contains_stable` has, and for the same reason: the owning thread mutates these
// chains concurrently, so one pass can read a head and its successor from two different generations
// and manufacture a link that never existed. A real poison survives to the second pass.
//
// SELF-VALIDATING: the walked-pools/heads/nodes census is written to `out` on EVERY call (with each
// pass's raw count), so a found=0 from a scan that walked nothing is never mistakable for "the
// chains are clean".
// Fault-safe (process_vm_readv), read-only, never gates a write. Returns the poison count.
struct Mb3PoisonHit {
    uint64_t pool_base = 0;
    uint32_t class_off = 0;    // byte offset of the size class within the pool array
    uint8_t  list = 0;         // 1 = partial bundle head, 2 = full bundle head
    uint32_t hops = 0;         // distance from the head
    uint64_t node = 0;         // the node whose next-link is poisoned
    uint64_t bad_next = 0;     // the poisoned value
};
int mb3_poison_scan(Mb3PoisonHit* hits, int max_hits, uint32_t max_hops, char* out, unsigned cap);

// Test isolation for the process-global candidate registry.
void mb3_reset_pool_candidates_for_test();

} // namespace prosper::gpu
