#include "mb3_freelist.hpp"
#include "gpu_execute.hpp"

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
std::atomic<uint64_t> g_pool_candidates[kMaxPoolCandidates];
std::atomic<uint32_t> g_pool_candidate_count{0};

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

bool mb3_freelist_contains_stable(uint64_t block, Mb3FreelistMatch* match) {
    Mb3FreelistMatch first{};
    if (!scan_once(block, &first)) return false;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    Mb3FreelistMatch second{};
    if (!scan_once(block, &second)) return false;
    if (match) *match = second;
    return true;
}

void mb3_reset_pool_candidates_for_test() {
    for (auto& base : g_pool_candidates) base.store(0, std::memory_order_relaxed);
    g_pool_candidate_count.store(0, std::memory_order_release);
}

} // namespace prosper::gpu
