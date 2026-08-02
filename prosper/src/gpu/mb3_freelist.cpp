#include "mb3_freelist.hpp"
#include "gpu_execute.hpp"
#include "../host/boot_program.hpp"   // #1659: BOOT_EBOOT (the real mapped base)

#include <atomic>
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <sys/uio.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace prosper::gpu {
namespace {

// Session-10 captures found 21 caches/run. Keep ample fixed storage with no allocation or locks on
// the pthread TLS hot path; an overflow only reduces guard coverage and cannot change guest state.
constexpr uint32_t kMaxPoolCandidates = 256;
constexpr uint32_t kMaxChainHops = 4096;
constexpr uint64_t kMb3PageSize = 0x10000;
constexpr uint32_t kMb3BlockSize = 0x20;
// FFreeBlock { uint16 BlockSizeShifted; uint8 PoolIndex; uint8 Canary; ... } for pool idx=1.
// DOLL's UE build uses 0xe3 (confirmed at eboot+0x230ca5c); newer UE sources use 0xe7.
constexpr uint32_t kMb3Idx1FreeTagE3 = 0xe3010002;
constexpr uint32_t kMb3Idx1FreeTagE7 = 0xe7010002;
std::atomic<uint64_t> g_pool_candidates[kMaxPoolCandidates];
std::atomic<uint32_t> g_pool_candidate_count{0};
std::atomic<uint64_t> g_global_recycler_bin{0};

bool safe_read(uint64_t addr, void* out, uint32_t bytes) {
#if defined(__linux__)
    // The allocator may decommit a candidate/node between a page-map probe and memcpy. Ask the
    // kernel to copy from our own address space instead: process_vm_readv returns EFAULT for that
    // race and never delivers SIGSEGV to the GPU worker.
    struct iovec local { out, bytes };
    struct iovec remote { (void*)(uintptr_t)addr, bytes };
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == (ssize_t)bytes;
#elif defined(_WIN32)
    SIZE_T copied = 0;
    return ReadProcessMemory(GetCurrentProcess(), (const void*)(uintptr_t)addr, out, bytes, &copied) &&
           copied == bytes;
#else
    if (!guest_readable(addr, bytes)) return false;
    memcpy(out, (const void*)(uintptr_t)addr, bytes);
    return true;
#endif
}

bool plausible_node(uint64_t addr) {
    // MB3 0x20-byte blocks are naturally 0x20-aligned. Rejecting a corrupt/tagged pointer before
    // dereference is both a traversal terminator and protection against the historic 0x...0001 node.
    return addr >= 0x10000 && !(addr & 0x1full);
}

bool scan_chain(uint64_t head, uint64_t block, uint8_t list, uint64_t pool_base,
                Mb3FreelistMatch* match) {
    uint64_t node = head;
    for (uint32_t hops = 0; node && hops < kMaxChainHops; ++hops) {
        if (node == block) {
            if (match) *match = {pool_base, head, hops, list};
            return true;
        }
        if (!plausible_node(node)) return false;
        uint64_t next = 0;
        if (!safe_read(node, &next, sizeof next) || next == node) return false;
        node = next;
    }
    return false;
}

bool central_scan_enabled() {
    static const bool enabled = [] {
        const char* e = getenv("PROSPER_MB3_CENTRAL_SCAN");
        return !e || strtol(e, nullptr, 0) != 0;
    }();
    return enabled;
}

bool scan_central_run(uint64_t block, Mb3FreelistMatch* match) {
    // A central FFreeBlock is an in-place header at the low end of a contiguous free run. Scanning
    // one 64 KiB MB3 page avoids depending on the title's FPoolInfoSmall table address/layout. The
    // exact pool/canary tag and the bounded run containing `block` make unrelated data an extremely
    // unlikely match. safe_read keeps a concurrent decommit from faulting the GPU worker.
    const uint64_t page_base = block & ~(kMb3PageSize - 1);
    const uint32_t block_off = (uint32_t)(block - page_base);
    alignas(64) static thread_local uint8_t page[kMb3PageSize];
    if (!safe_read(page_base, page, sizeof page)) return false;

    for (uint32_t off = 0; off <= kMb3PageSize - 12; off += kMb3BlockSize) {
        uint32_t tag = 0, num_free = 0;
        memcpy(&tag, page + off, sizeof tag);
        if (tag != kMb3Idx1FreeTagE3 && tag != kMb3Idx1FreeTagE7) continue;
        memcpy(&num_free, page + off + 4, sizeof num_free);
        const uint32_t max_run = (uint32_t)((kMb3PageSize - off) / kMb3BlockSize);
        if (!num_free || num_free > max_run) continue;
        const uint32_t run_end = off + num_free * kMb3BlockSize;
        if (block_off >= off && block_off < run_end) {
            if (match) *match = {page_base, page_base + off,
                                 (block_off - off) / kMb3BlockSize, 11};
            return true;
        }
    }
    return false;
}

uint64_t discover_global_recycler_bin() {
    // UE's compiled bundle-spill path scales PoolIndex by 0x40 immediately before a RIP-relative
    // LEA of the eight-slot recycler table. Locate that instruction shape once instead of baking in
    // DOLL's eboot address. The signature includes the following per-thread count transfer and slot
    // probe; the displacement itself is the only wildcard.
    static const uint64_t discovered = [] {
        // #1659: this SCANS guest memory for an instruction signature, so a stale base is not a
        // labelling problem — 0x400000000 is the pre-#825 eboot address and is now a direct-memory
        // aperture, meaning the scan swept 64 MiB of the wrong region and could never match.
        constexpr uint64_t kImageBase = prosper::BOOT_EBOOT;
        constexpr uint64_t kScanBytes = 0x04000000ull;
        constexpr uint32_t kChunk = 0x10000;
        constexpr uint32_t kOverlap = 0x40;
        static const uint8_t prefix[] = {0xc1, 0xe0, 0x06, 0x4c, 0x8d, 0x0d};
        static const uint8_t suffix[] = {
            0x41, 0xc1, 0xe0, 0x05, 0x4a, 0x8d, 0x74, 0x01, 0x18,
            0x46, 0x8b, 0x44, 0x01, 0x18, 0x44, 0x89, 0x47, 0x08,
            0x4d, 0x8d, 0x04, 0x01, 0x4a, 0x83, 0x3c, 0x08, 0x00,
        };
        alignas(64) static uint8_t code[kChunk];
        for (uint64_t chunk_addr = kImageBase;
             chunk_addr < kImageBase + kScanBytes; chunk_addr += kChunk - kOverlap) {
            if (!safe_read(chunk_addr, code, sizeof code)) break;
            constexpr uint32_t kPatternBytes = sizeof prefix + 4 + sizeof suffix;
            for (uint32_t off = 0; off + kPatternBytes <= kChunk; ++off) {
                if (memcmp(code + off, prefix, sizeof prefix) ||
                    memcmp(code + off + sizeof prefix + 4, suffix, sizeof suffix)) continue;
                int32_t disp = 0;
                memcpy(&disp, code + off + sizeof prefix, sizeof disp);
                uint64_t signature = chunk_addr + off;
                uint64_t table = signature + sizeof prefix + 4 + (int64_t)disp;
                uint8_t slots[64] = {};
                if (table < kImageBase || table >= 0x4c0000000ull || (table & 0x3full) ||
                    !safe_read(table + 0x40, slots, sizeof slots)) continue;
                fprintf(stderr, "[agc] MB3 global recycler discovered: table=0x%llx "
                                "idx1=0x%llx signature=0x%llx\n",
                        (unsigned long long)table, (unsigned long long)(table + 0x40),
                        (unsigned long long)signature);
                return (uint64_t)(table + 0x40); // size-class idx=1 (Malloc(0x20))
            }
        }
        return uint64_t{0};
    }();
    return discovered;
}

bool scan_once(uint64_t block, Mb3FreelistMatch* match) {
    if (!plausible_node(block)) return false;
    uint32_t count = g_pool_candidate_count.load(std::memory_order_acquire);
    if (count > kMaxPoolCandidates) count = kMaxPoolCandidates;
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t base = g_pool_candidates[i].load(std::memory_order_acquire);
        if (!base) continue;

        // Per-size-class descriptor layout recovered at eboot+0x2316a20:
        // {head,count,secondaryHead,secondaryCount}, 0x20 stride. idx=1 is Malloc(0x20).
        uint64_t bin[4] = {};
        if (!safe_read(base + 0x20, bin, sizeof bin)) continue;
        if (scan_chain(bin[0], block, 1, base, match) ||
            scan_chain(bin[2], block, 2, base, match)) return true;
    }

    uint64_t recycler = g_global_recycler_bin.load(std::memory_order_acquire);
    if (!recycler) {
        // Diagnostic override used to validate a newly recovered allocator layout before making
        // automatic discovery production behavior. The value is the exact idx=1 eight-slot bin,
        // not the beginning of the all-size-class recycler table.
        static const uint64_t env_recycler = [] {
            const char* e = getenv("PROSPER_MB3_GLOBAL_BIN");
            return e ? strtoull(e, nullptr, 0) : 0ull;
        }();
        recycler = env_recycler;
    }
    if (!recycler) recycler = discover_global_recycler_bin();
    if (recycler >= 0x10000 && !(recycler & 7ull)) {
        for (uint8_t slot = 0; slot < 8; ++slot) {
            uint64_t head = 0;
            if (!safe_read(recycler + (uint64_t)slot * 8, &head, sizeof head)) break;
            if (scan_chain(head, block, (uint8_t)(3 + slot), recycler, match)) return true;
        }
    }
    if (central_scan_enabled() && scan_central_run(block, match)) return true;
    return false;
}

} // namespace

bool mb3_tls_tracking_enabled() {
    static const bool enabled = [] {
        const char* e = getenv("PROSPER_MB3_TRACK_TLS");
        return !e || strtol(e, nullptr, 0) != 0;
    }();
    return enabled;
}

void mb3_note_tls_pool_candidate(uint64_t base) {
    // The observed MB3 pool arrays are 64 KiB allocations. This cheap shape gate makes the usual
    // pthread_getspecific traffic a no-op and keeps unrelated TLS pointers out of the scan set.
    if (base < 0x10000 || (base & 0xffffull)) return;
    uint32_t count = g_pool_candidate_count.load(std::memory_order_acquire);
    if (count > kMaxPoolCandidates) count = kMaxPoolCandidates;
    for (uint32_t i = 0; i < count; ++i)
        if (g_pool_candidates[i].load(std::memory_order_acquire) == base) return;

    uint32_t slot = g_pool_candidate_count.fetch_add(1, std::memory_order_acq_rel);
    if (slot < kMaxPoolCandidates)
        g_pool_candidates[slot].store(base, std::memory_order_release);
}

void mb3_note_global_recycler_bin(uint64_t base) {
    if (base < 0x10000 || (base & 7ull)) return;
    g_global_recycler_bin.store(base, std::memory_order_release);
}

// See mb3_freelist.hpp. Probes the learned idx=1 bin heads — each is trivially a member of its own
// chain, so a walk that answers "no" to one of those is blind rather than truthful.
int mb3_freelist_selftest(char* out, unsigned cap) {
    uint32_t pools = g_pool_candidate_count.load(std::memory_order_acquire);
    if (pools > kMaxPoolCandidates) pools = kMaxPoolCandidates;
    int probes = 0, positives = 0, deep = 0, deep_pos = 0;
    uint64_t first_head = 0;
    for (uint32_t i = 0; i < pools; ++i) {
        const uint64_t base = g_pool_candidates[i].load(std::memory_order_acquire);
        if (!base) continue;
        uint64_t bin[4] = {};
        if (!safe_read(base + 0x20, bin, sizeof bin)) continue;
        for (const uint64_t head : {bin[0], bin[2]}) {
            if (!plausible_node(head)) continue;
            if (!first_head) first_head = head;
            probes++;
            Mb3FreelistMatch m{};
            if (scan_once(head, &m)) positives++;
            // A head matches at hop 0, so the probe above proves the walk is ARMED and reaches the
            // chains — but never that it can TRAVERSE one. Probe the head's successor too: an
            // interior node is only findable by following a link, so a walk whose traversal is
            // broken (a mis-shaped node terminating the loop, say) shows up here and nowhere else.
            uint64_t next = 0;
            if (!safe_read(head, &next, sizeof next) || !plausible_node(next)) continue;
            deep++;
            Mb3FreelistMatch dm{};
            if (scan_once(next, &dm) && dm.hops >= 1) deep_pos++;
        }
    }
    if (out && cap)
        snprintf(out, cap, "pools=%u probes=%d positives=%d deep=%d deep_positives=%d first_head=0x%llx%s",
                 pools, probes, positives, deep, deep_pos, (unsigned long long)first_head,
                 probes == 0            ? "  [NO HEADS — walk has nothing to answer about]"
                 : positives == 0       ? "  [BLIND — a head is not found in its own chain]"
                 : (deep && !deep_pos)  ? "  [SHALLOW — heads found, but no interior node reached]"
                                        : "");
    // Traversal is the stronger property, so report it when it was testable at all.
    return deep ? deep_pos : positives;
}

bool mb3_freelist_contains_stable(uint64_t block, Mb3FreelistMatch* match) {
    Mb3FreelistMatch first{};
    if (!scan_once(block, &first)) return false;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    Mb3FreelistMatch second{};
    if (!scan_once(block, &second)) return false;
    if (match) *match = second;
    return true;
}

// See mb3_freelist.hpp. Bin region: the first 0x400 bytes of each learned pool array cover 32
// size classes of {head, count32, secondaryHead, secondaryCount32} at 0x20 stride; heads sit at
// +0x00 and +0x10 of each class (observed live: the #1226 poisoned bin at base+0x20 between
// healthy raw-pointer bins). A legitimate head is 0 or a full 0x20xx/0x30xx pointer (nonzero top
// dword), so the byte-shifted test (top32==0, <<8 lands in mapped [0x20..0x40) guest memory) has
// no overlap with healthy state; safe_read doubles as the fault-safe mappedness probe.
int mb3_poolshift_window_scan(char* out, unsigned cap) {
    unsigned off = 0; int found = 0;
    uint32_t count = g_pool_candidate_count.load(std::memory_order_acquire);
    if (count > kMaxPoolCandidates) count = kMaxPoolCandidates;
    for (uint32_t i = 0; i < count; i++) {
        uint64_t base = g_pool_candidates[i].load(std::memory_order_acquire);
        if (!base) continue;
        uint8_t bins[0x400];
        if (!safe_read(base, bins, sizeof bins)) continue;
        for (uint32_t o = 0; o + 8 <= sizeof bins; o += 0x10) {
            uint64_t q; memcpy(&q, bins + o, sizeof q);
            if (q >> 32 || q < 0x100000) continue;
            uint64_t real = q << 8;
            if (real < 0x2000000000ull || real >= 0x4000000000ull) continue;
            uint64_t probe;
            if (!safe_read(real, &probe, sizeof probe)) continue;
            if (out && off + 96 < cap) {
                int m = snprintf(out + off, cap - off,
                                 "[poolshift-window] pool=0x%llx +0x%x head=0x%llx (<<8=0x%llx)\n",
                                 (unsigned long long)base, o, (unsigned long long)q,
                                 (unsigned long long)real);
                if (m > 0) off += (unsigned)m;
            }
            found++;
        }
    }
    // #1254-review B1: the eight global recycler slots hold bundle HEAD values directly, so a
    // poisoned bundle parked there (not in any TLS bin) carries the same byteshift shape. Scan them
    // too, else a parked poison scores found=0 for its whole residence. Chain INTERIORS remain a
    // structural blind spot (the poison buried below a healthy head is invisible to a head-only
    // scan) — see the header comment; the watchpoint path is what actually closes that.
    uint64_t recycler = g_global_recycler_bin.load(std::memory_order_acquire);
    if (!recycler) recycler = discover_global_recycler_bin();
    if (recycler >= 0x10000 && !(recycler & 7ull)) {
        for (uint8_t slot = 0; slot < 8; ++slot) {
            uint64_t head = 0;
            if (!safe_read(recycler + (uint64_t)slot * 8, &head, sizeof head)) break;
            if (head >> 32 || head < 0x100000) continue;
            uint64_t real = head << 8;
            if (real < 0x2000000000ull || real >= 0x4000000000ull) continue;
            uint64_t probe;
            if (!safe_read(real, &probe, sizeof probe)) continue;
            if (out && off + 96 < cap) {
                int m = snprintf(out + off, cap - off,
                                 "[poolshift-window] recycler=0x%llx slot=%u head=0x%llx (<<8=0x%llx)\n",
                                 (unsigned long long)recycler, slot, (unsigned long long)head,
                                 (unsigned long long)real);
                if (m > 0) off += (unsigned)m;
            }
            found++;
        }
    }
    if (out && off < cap) out[off] = 0;
    return found;
}

// #1226: is `base` a learned TLS pool array? Async-signal-safe (atomic loads only) — the
// PROSPER_FAULTOBJ dump uses it to report whether the faulting pool was even visible to the
// window probe (an unregistered pool explains a probe miss without implicating scan timing).
extern "C" int prosper_mb3_is_pool_candidate(uint64_t base) {
    uint32_t count = g_pool_candidate_count.load(std::memory_order_acquire);
    if (count > kMaxPoolCandidates) count = kMaxPoolCandidates;
    for (uint32_t i = 0; i < count; i++)
        if (g_pool_candidates[i].load(std::memory_order_acquire) == base) return 1;
    return 0;
}

void mb3_reset_pool_candidates_for_test() {
    for (auto& base : g_pool_candidates) base.store(0, std::memory_order_relaxed);
    g_pool_candidate_count.store(0, std::memory_order_release);
    g_global_recycler_bin.store(0, std::memory_order_release);
}

} // namespace prosper::gpu
