// hle_kernel_mem.cpp — HLE of libkernel virtual/direct memory (Linux backing).
// PS5 memory model: reserve a virtual range, allocate "direct" (physical) memory as
// an opaque offset, then map it (or flexible memory) into VA. We back it with host
// mmap and TRACK every mapping so VirtualQuery is truthful and so we can log/debug the
// guest's address-space construction. Guarded to Linux.
#include "dispatch.hpp"
#include "nid.hpp"
#include "sync_futex.hpp"   // shared futex wake + waiter registration (also used by the GPU's label wake)

#ifdef __linux__
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <unistd.h>
#include <fcntl.h>          // fallocate
#include <linux/falloc.h>  // FALLOC_FL_PUNCH_HOLE / FALLOC_FL_KEEP_SIZE
#include <climits>
#include <cerrno>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <vector>

namespace prosper {

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)

namespace {
    bool memlog() { static int v = getenv("PROSPER_MEMLOG") ? 1 : 0; return v; }
    #define MLOG(...) do { if (memlog()) fprintf(stderr, "[memhle] " __VA_ARGS__); } while (0)

    struct Mapping { uint64_t base, size; int prot; bool committed; char name[32]; };
    std::mutex g_mx;
    std::vector<Mapping> g_maps;
    // Direct ("physical") memory: a bump allocator over a FINITE pool. The pool size is what
    // sceKernelGetDirectMemorySize advertises, and exhaustion MUST fail with ENOMEM like real
    // hardware: guests rely on it — UE4 (PPSA17942) sizes its pool requests from
    // GetDirectMemorySize and allocates chunks in a loop until ENOMEM ends it. A never-failing
    // allocator handed out offsets past the pool, the guest's 512GB-arena block bitmap indexed
    // out of range, and user_malloc_init crashed on the bitmap read.
    constexpr uint64_t kDmemBase  = 0x10000000;
    constexpr uint64_t kDmemTotal = 8ull * 1024 * 1024 * 1024;   // 8 GiB pool
    // Direct ("physical") memory allocations, kept SORTED by start (first-fit allocation walks the
    // gaps). Also serves sceKernelDirectMemoryQuery.
    struct DMem { uint64_t start, end; int type; };
    std::mutex g_dmx;
    std::vector<DMem> g_dmem;

    // Claim `sz` bytes of direct memory at `align`, first-fit over the pool's free gaps WITHIN the
    // caller's [lo, hi) search window (default = the whole pool). False (no state change) when
    // nothing fits — callers translate that to SCE_KERNEL_ERROR_ENOMEM (0x8002000C, Kyty Errno.h).
    // sceKernelAllocateDirectMemory passes searchStart/searchEnd here: a guest that partitions
    // physical memory by window (GPU pool above a CPU pool, etc.) must get an offset inside its
    // window, not wherever the base-first walk lands. First-fit (not bump) because guests genuinely
    // RELEASE ranges: UE4 (PPSA17942) probes the pool full with halving allocations, releases the
    // probes, then allocates its real pools — a bump cursor would be permanently exhausted.
    bool dmem_take(uint64_t sz, uint64_t align, int type, uint64_t& off_out,
                   uint64_t lo = 0, uint64_t hi = ~0ull) {
        if (!align) align = 0x4000;
        if (lo < kDmemBase) lo = kDmemBase;
        if (hi > kDmemBase + kDmemTotal) hi = kDmemBase + kDmemTotal;
        if (lo >= hi) return false;
        std::lock_guard<std::mutex> lk(g_dmx);
        uint64_t cursor = kDmemBase;
        size_t insert_at = 0;
        for (size_t i = 0; i <= g_dmem.size(); i++) {
            uint64_t gap_beg = cursor;
            uint64_t gap_end = (i < g_dmem.size()) ? g_dmem[i].start : kDmemBase + kDmemTotal;
            // Clip the free gap to the search window, then align the start.
            uint64_t beg = gap_beg < lo ? lo : gap_beg;
            uint64_t end = gap_end > hi ? hi : gap_end;
            uint64_t off = (beg + align - 1) & ~(align - 1);
            if (off + sz <= end) { insert_at = i; off_out = off; goto found; }
            if (i < g_dmem.size()) cursor = g_dmem[i].end;
        }
        return false;
    found:
        g_dmem.insert(g_dmem.begin() + insert_at, { off_out, off_out + sz, type });
        return true;
    }

    // Largest FREE aligned block within [lo, hi) of the direct-memory pool (for
    // sceKernelAvailableDirectMemorySize). Walks the same sorted free gaps dmem_take allocates from,
    // clipping each gap to the requested [lo, hi) window and aligning its start. Returns the biggest
    // such block's aligned offset + size; size 0 means nothing fits (caller -> ENOMEM).
    void dmem_largest_free(uint64_t lo, uint64_t hi, uint64_t align, uint64_t& off_out, uint64_t& size_out) {
        if (!align) align = 0x4000;
        if (lo < kDmemBase) lo = kDmemBase;
        if (hi > kDmemBase + kDmemTotal) hi = kDmemBase + kDmemTotal;
        off_out = 0; size_out = 0;
        std::lock_guard<std::mutex> lk(g_dmx);
        uint64_t cursor = kDmemBase;
        for (size_t i = 0; i <= g_dmem.size(); i++) {
            uint64_t gap_beg = cursor;
            uint64_t gap_end = (i < g_dmem.size()) ? g_dmem[i].start : kDmemBase + kDmemTotal;
            // Clip the free gap to the caller's search window, then align its start.
            uint64_t beg = gap_beg < lo ? lo : gap_beg;
            uint64_t end = gap_end > hi ? hi : gap_end;
            uint64_t aligned = (beg + align - 1) & ~(align - 1);
            if (end > aligned && end - aligned > size_out) { off_out = aligned; size_out = end - aligned; }
            if (i < g_dmem.size()) cursor = g_dmem[i].end;
        }
    }

    // Release [start, start+len): remove or trim every overlapping allocation (kernel semantics —
    // the range is freed regardless of how it was carved into allocations).
    void dmem_release(uint64_t start, uint64_t len) {
        uint64_t end = start + len;
        std::lock_guard<std::mutex> lk(g_dmx);
        std::vector<DMem> out;
        out.reserve(g_dmem.size());
        for (auto& d : g_dmem) {
            if (d.end <= start || d.start >= end) { out.push_back(d); continue; }
            if (d.start < start) out.push_back({ d.start, start, d.type });
            if (d.end > end)     out.push_back({ end, d.end, d.type });
        }
        g_dmem.swap(out);
    }

    void track(uint64_t base, uint64_t size, int prot, bool committed, const char* nm) {
        std::lock_guard<std::mutex> lk(g_mx);
        Mapping m{ base, size, prot, committed, {0} };
        if (nm) { strncpy(m.name, nm, sizeof m.name - 1); }
        g_maps.push_back(m);
    }
    // Trim/split tracked mappings overlapping [base, base+len). munmap/BatchMap-UNMAP must remove
    // their tracking (this never happened before — g_maps only ever grew): stale "committed"
    // records made VirtualQuery report freed VAs as live, made the fault handler's lazy-commit
    // probe (prosper_reserved_range_state) mis-decide on re-reserved ranges, and let the render
    // thread's safe_copy memcpy a texture whose backing the game had batch-unmapped — the exact
    // SIGSEGV class safe_copy exists to prevent (same overlap-trim shape as dmem_release above).
    void untrack(uint64_t base, uint64_t len) {
        if (!len) return;
        uint64_t end = base + len;
        std::lock_guard<std::mutex> lk(g_mx);
        std::vector<Mapping> out;
        out.reserve(g_maps.size() + 1);
        for (auto& m : g_maps) {
            uint64_t me = m.base + m.size;
            if (me <= base || m.base >= end) { out.push_back(m); continue; }
            if (m.base < base) { Mapping lo = m; lo.size = base - m.base; out.push_back(lo); }
            if (me > end)      { Mapping hi = m; hi.base = end; hi.size = me - end; out.push_back(hi); }
        }
        g_maps.swap(out);
    }
    // Re-tag [base, base+len) with a new protection (mprotect): replace the overlapped span so
    // VirtualQuery/the fault probe report the CURRENT prot, not the one from map time.
    void retrack_prot(uint64_t base, uint64_t len, int prot, const char* nm) {
        if (!len) return;
        untrack(base, len);
        track(base, len, prot, true, nm);
    }
    const Mapping* find(uint64_t addr) {
        std::lock_guard<std::mutex> lk(g_mx);
        const Mapping* best = nullptr;
        for (auto& m : g_maps)
            if (addr >= m.base && addr < m.base + m.size)
                if (!best || m.base > best->base) best = &m;   // most specific (latest overlay)
        return best;
    }
    // Is [base, base+len) entirely covered by our OWN UNCOMMITTED (PROT_NONE reservation) mappings?
    // Only then is a MAP_FIXED replace safe: the guest is committing a range it reserved (#137). A
    // committed span (live guest memory) or a gap (an untracked host mapping / loaded image) means a
    // MAP_FIXED would silently destroy it — so map_at/map_phys_at must fail instead. Walks boundary to
    // boundary (not page by page) so a 512 GiB reservation is checked in a few steps.
    bool range_is_free_reservation(uint64_t base, uint64_t len) {
        if (!len) return false;
        std::lock_guard<std::mutex> lk(g_mx);
        uint64_t cur = base, end = base + len;
        while (cur < end) {
            const Mapping* cover = nullptr;      // most-specific mapping containing cur
            uint64_t next_start = end;           // nearest mapping that STARTS after cur (a coverage gap boundary)
            for (auto& m : g_maps) {
                uint64_t me = m.base + m.size;
                if (cur >= m.base && cur < me) { if (!cover || m.base > cover->base) cover = &m; }
                else if (m.base > cur && m.base < next_start) next_start = m.base;
            }
            if (!cover || cover->committed) return false;   // gap (untracked) or committed = unsafe to replace
            uint64_t adv = cover->base + cover->size;
            if (next_start < adv) adv = next_start;         // don't skip a mapping that starts inside `cover`
            cur = adv;
        }
        return true;
    }
    // Smallest mapping base strictly greater than addr (0 if none) — for hole reporting.
    uint64_t next_base(uint64_t addr) {
        std::lock_guard<std::mutex> lk(g_mx);
        uint64_t n = 0;
        for (auto& m : g_maps)
            if (m.base > addr && (n == 0 || m.base < n)) n = m.base;
        return n;
    }

    int host_prot(uint64_t p) {
        int hp = 0;
        if (p & 0x1) hp |= PROT_READ;
        if (p & 0x2) hp |= PROT_READ | PROT_WRITE;
        if (p & 0x4) hp |= PROT_EXEC;
        if (!hp) hp = PROT_READ | PROT_WRITE;
        return hp;
    }
    uint64_t align_up(uint64_t v, uint64_t a) { return a ? (v + a - 1) & ~(a - 1) : v; }
    void* map_at(uint64_t hint, uint64_t len, int prot) {
        if (!hint) {
            void* p = mmap(nullptr, len, prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            return p == MAP_FAILED ? nullptr : p;
        }
        // Non-zero hint: claim it WITHOUT clobbering (#137). MAP_FIXED_NOREPLACE fails if anything
        // is already there; only if the occupant is entirely our own uncommitted reservation do we
        // replace it with MAP_FIXED (the guest is committing a range it reserved). A hint that
        // overlaps a committed mapping or an untracked host mapping fails instead of destroying it.
        void* p = mmap((void*)hint, len, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p != MAP_FAILED) return p;
        if (range_is_free_reservation(hint, len)) {
            p = mmap((void*)hint, len, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
            return p == MAP_FAILED ? nullptr : p;
        }
        return nullptr;
    }

    // The direct-memory pool is backed by ONE shared memfd so every mapping of a phys offset
    // aliases the SAME bytes — the console's unified-physical-memory contract. Independent
    // anonymous pages per mapping broke aliasing: UE4's MallocBinned3 re-maps a committed phys
    // page at a second VA and expects its block canaries there ("MallocBinned3 Corruption
    // Canary" fatal without this). Sized to cover [0, kDmemBase + kDmemTotal) so offset == phys;
    // the file is sparse, so unused ranges cost nothing.
    int dmem_fd() {
        static int fd = [] {
            int f = (int)syscall(SYS_memfd_create, "prosper-dmem", 1 /*MFD_CLOEXEC*/);
            if (f >= 0 && ftruncate(f, (off_t)(kDmemBase + kDmemTotal)) != 0) { close(f); f = -1; }
            return f;
        }();
        return fd;
    }
    // Zero a phys range in the memfd — real hardware hands out ZEROED pages on a fresh direct-memory
    // allocation, but our memfd RETAINS bytes across release/reuse (a released phys re-allocated to a
    // new buffer would otherwise expose stale content). Punch a hole (reads back as zeros, keeps the
    // range sparse). Called only at ALLOCATION time, so it never disturbs the aliasing contract
    // (which is about mapping an already-allocated phys at a second VA). CONFIDENCE: HIGH (matches
    // the console's fresh-page-zeroed semantics).
    void dmem_zero(uint64_t phys, uint64_t sz) {
        int fd = dmem_fd();
        if (fd >= 0 && phys < kDmemBase + kDmemTotal && sz)
            fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, (off_t)phys, (off_t)sz);
    }
    // Map `len` bytes of the phys pool at `hint` (0 = anywhere). Falls back to anonymous memory
    // if the memfd is unavailable (still boots; loses aliasing).
    void* map_phys_at(uint64_t hint, uint64_t len, int prot, uint64_t phys) {
        int fd = dmem_fd();
        if (fd >= 0 && phys < kDmemBase + kDmemTotal) {
            if (!hint) {
                void* p = mmap(nullptr, len, prot, MAP_SHARED, fd, (off_t)phys);
                if (p != MAP_FAILED) return p;
            } else {
                // Same no-clobber discipline as map_at (#137): NOREPLACE first, MAP_FIXED replace
                // only over our own uncommitted reservation, else refuse rather than destroy a live
                // (committed / untracked) mapping — the exact clobber class of issues #88 / #107.
                void* p = mmap((void*)hint, len, prot, MAP_SHARED | MAP_FIXED_NOREPLACE, fd, (off_t)phys);
                if (p != MAP_FAILED) return p;
                if (range_is_free_reservation(hint, len)) {
                    p = mmap((void*)hint, len, prot, MAP_SHARED | MAP_FIXED, fd, (off_t)phys);
                    if (p != MAP_FAILED) return p;
                } else {
                    return nullptr;
                }
            }
        }
        return map_at(hint, len, prot);
    }
}

// sceKernelReserveVirtualRange(void** addrInOut, size_t len, int flags, size_t align)
// MUST return an address aligned to `align`: guest allocators round the returned base
// down to their requested alignment and would otherwise land below the mapping.
//
// FLAG SEMANTICS (issue #161): SCE_KERNEL_MAP_FIXED (0x10) means "exactly this address".
// WITHOUT it, the hint is only a SEARCH START — the kernel finds the first free range at or
// above the hint (shadPS4 MemoryManager::MapMemory SearchFree; BSD mmap contract). Treating
// every hinted reserve as fixed — and, worse, blessing a hinted reserve that lands inside an
// EXISTING reservation as "OK, same base" (the old #115 workaround below) — handed the SAME
// base 0x1000000000 to two DIFFERENT guest VM spaces: UE4 PPSA17942 reserves its 512 GiB
// MallocBinned3 arena (len=0x8000000000, flags=0) and then a 64 MiB allocator-metadata pool
// (len=0x4000000, flags=0) with the SAME hint. With both spaces based at the same VA, MB3's
// class-0 pool-info-table pointer array and its block-of-blocks BIT TREE were carved at the
// SAME address (0x1000000000): when pools 0-63 of size-class 0 filled up, the bit tree's
// root-propagation write (Bits[0] |= 1, caught live by a HW watchpoint at eboot+0x231c0ca)
// flipped the low byte of the stored table pointer 0x20015f0000 -> 0x20015f0001, all
// subsequent FPoolInfoSmall reads came back misaligned, and the boot died in a
// "MallocBinned3 Corruption Canary" spam-loop. CONFIDENCE: HIGH (reserve args live-captured;
// the corrupting write watched in-place; search semantics cross-checked against shadPS4).
HLE(k_reserve_vrange) {
    uint64_t hint = a0 ? *(uint64_t*)a0 : 0;
    uint64_t align = a3 ? a3 : 0x4000;
    const bool fixed = (a2 & 0x10) != 0;   // SCE_KERNEL_MAP_FIXED
    MLOG("reserve ENTRY hint=0x%llx len=0x%llx flags=0x%llx align=0x%llx\n",
         (unsigned long long)hint, (unsigned long long)a1, (unsigned long long)a2,
         (unsigned long long)a3);
    if (!a1) return 0x80020016ull;         // SCE_KERNEL_ERROR_EINVAL (len 0; shadPS4 rejects too)
    if (hint && fixed) {
        void* p = mmap((void*)hint, a1, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == MAP_FAILED) {
            // The FIXED hint region is already mapped. If it's ALREADY one of OUR reservations
            // (uncommitted), the guest is re-claiming its own recorded range — idempotent success
            // (#115: an error here made the guest allocator hand back NULL, which the level1
            // asset-load FileCacher then memcpy'd from).
            bool ours = false;
            { std::lock_guard<std::mutex> lk(g_mx);
              for (auto& m : g_maps)
                  if (!m.committed && hint >= m.base && hint < m.base + m.size) { ours = true; break; } }
            if (ours) {
                if (a0) *(uint64_t*)a0 = hint;
                MLOG("reserve hint=0x%llx re-reserve-of-own-range -> OK\n", (unsigned long long)hint);
                return 0;
            }
            MLOG("reserve FIXED hint=0x%llx FAILED\n", (unsigned long long)hint); return 0x16;
        }
        if (a0) *(uint64_t*)a0 = (uint64_t)p;
        track((uint64_t)p, a1, 0, false, "reserved");
        MLOG("reserve(fixed) -> 0x%llx len=0x%llx align=0x%llx\n", (unsigned long long)p, (unsigned long long)a1, (unsigned long long)align);
        return 0;
    }
    if (hint) {
        // Non-fixed hint: search for a free range starting at the hint. Probe candidates with
        // MAP_FIXED_NOREPLACE (also catches host mappings the tracker doesn't know); on a miss,
        // skip past whichever TRACKED mapping covers the candidate (fast-forwards the search past
        // the 512 GiB arena in one step), else advance one alignment granule.
        uint64_t cand = align_up(hint, align);
        const uint64_t kSearchLimit = 0x40000000000ull;   // 4 TiB — far above any guest range
        while (cand < kSearchLimit) {
            void* p = mmap((void*)cand, a1, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (p != MAP_FAILED) {
                if (a0) *(uint64_t*)a0 = (uint64_t)p;
                track((uint64_t)p, a1, 0, false, "reserved");
                MLOG("reserve(search from 0x%llx) -> 0x%llx len=0x%llx align=0x%llx\n",
                     (unsigned long long)hint, (unsigned long long)p,
                     (unsigned long long)a1, (unsigned long long)align);
                return 0;
            }
            uint64_t next = cand + (align > 0x10000 ? align : 0x10000);
            { std::lock_guard<std::mutex> lk(g_mx);
              for (auto& m : g_maps)
                  if (cand >= m.base && cand < m.base + m.size) {
                      uint64_t past = align_up(m.base + m.size, align);
                      if (past > next) next = past;
                  } }
            cand = next;
        }
        MLOG("reserve search from 0x%llx FAILED (no free range)\n", (unsigned long long)hint);
        return 0x8002000cull;   // SCE_KERNEL_ERROR_ENOMEM
    }
    // Over-map by `align`, then trim the head/tail slack to yield an aligned span.
    uint64_t total = a1 + align;
    void* raw = mmap(nullptr, total, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) { MLOG("reserve len=0x%llx FAILED\n", (unsigned long long)a1); return 0x16; }
    uint64_t base = align_up((uint64_t)raw, align);
    if (base > (uint64_t)raw) munmap(raw, base - (uint64_t)raw);
    uint64_t used_end = base + a1, raw_end = (uint64_t)raw + total;
    if (raw_end > used_end) munmap((void*)used_end, raw_end - used_end);
    if (a0) *(uint64_t*)a0 = base;
    track(base, a1, 0, false, "reserved");
    MLOG("reserve -> 0x%llx len=0x%llx align=0x%llx (raw 0x%llx)\n",
         (unsigned long long)base, (unsigned long long)a1, (unsigned long long)align, (unsigned long long)raw);
    return 0;
}

// sceKernelMapNamedFlexibleMemory(void** addrInOut, size_t len, int prot, int flags, const char* name)
HLE(k_map_flexible) {
    uint64_t hint = a0 ? *(uint64_t*)a0 : 0;
    void* p = map_at(hint, a1, host_prot(a2));
    if (!p) { MLOG("mapflexible hint=0x%llx len=0x%llx FAILED\n", (unsigned long long)hint, (unsigned long long)a1); return 0x16; }
    if (a0) *(uint64_t*)a0 = (uint64_t)p;
    track((uint64_t)p, a1, host_prot(a2), true, a4 ? (const char*)a4 : "flexible");
    MLOG("mapflexible -> 0x%llx len=0x%llx prot=0x%llx name=%s\n",
         (unsigned long long)p, (unsigned long long)a1, (unsigned long long)a2, a4 ? (const char*)a4 : "");
    return 0;
}

// sceKernelAllocateDirectMemory(off_t start, off_t end, size_t len, size_t align, int memType, off_t* physOut)
HLE(k_alloc_dmem) {   // (searchStart, searchEnd, len, alignment, memoryType, physAddrOut)
    uint64_t align = a3 ? a3 : 0x4000;
    uint64_t sz = align_up(a2, align);
    uint64_t off;
    // Honor the [searchStart, searchEnd) window (a0/a1) — dropping it handed the guest an offset
    // outside the window it asked for.
    if (!dmem_take(sz, align, (int)a4, off, a0, a1 ? a1 : ~0ull)) {
        MLOG("alloc_dmem len=0x%llx in [0x%llx,0x%llx) -> ENOMEM\n",
             (unsigned long long)a2, (unsigned long long)a0, (unsigned long long)a1);
        return 0x8002000Cull;   // SCE_KERNEL_ERROR_ENOMEM
    }
    dmem_zero(off, sz);                       // fresh allocation -> zeroed pages (console semantics)
    if (a5 > 0xffff) *(uint64_t*)a5 = off;   // only write through a plausible out-pointer
    MLOG("alloc_dmem len=0x%llx -> phys=0x%llx\n", (unsigned long long)a2, (unsigned long long)off);
    return 0;
}

// sceKernelAllocateMainDirectMemory(size_t len, size_t align, int memType, off_t* physOut) — a
// DIFFERENT signature (4 args) from AllocateDirectMemory: physOut is arg3, not arg5. Aliasing them
// to one handler wrote the result through arg5 (uninitialized garbage, e.g. 0xa) -> crash.
HLE(k_alloc_main_dmem) {
    uint64_t align = a1 ? a1 : 0x4000;
    uint64_t sz = align_up(a0, align);
    uint64_t off;
    if (!dmem_take(sz, align, (int)a2, off)) {
        MLOG("alloc_main_dmem len=0x%llx -> ENOMEM (pool exhausted)\n", (unsigned long long)a0);
        return 0x8002000Cull;   // SCE_KERNEL_ERROR_ENOMEM
    }
    dmem_zero(off, sz);                       // fresh allocation -> zeroed pages (console semantics)
    if (a3 > 0xffff) *(uint64_t*)a3 = off;
    MLOG("alloc_main_dmem len=0x%llx -> phys=0x%llx\n", (unsigned long long)a0, (unsigned long long)off);
    return 0;
}

// sceKernelDirectMemoryQuery(off_t offset, int flags, SceKernelDirectMemoryQueryInfo* info, size_t infoSize)
//   info: 0x00 off_t start; 0x08 off_t end; 0x10 i32 memoryType. flags&1 = find next.
HLE(k_direct_memory_query) {
    if (!a2) return 0x16;
    uint8_t* info = (uint8_t*)a2;
    uint64_t sz = a3 ? (a3 > 0x18 ? 0x18 : a3) : 0x18;
    memset(info, 0, sz);
    std::lock_guard<std::mutex> lk(g_dmx);
    const DMem* hit = nullptr; const DMem* next = nullptr;
    for (auto& d : g_dmem) {
        if (a0 >= d.start && a0 < d.end) { if (!hit || d.start > hit->start) hit = &d; }
        if (d.start > a0 && (!next || d.start < next->start)) next = &d;
    }
    const DMem* r = hit ? hit : ((a1 & 1) ? next : nullptr);
    if (!r) { MLOG("dmem_query(0x%llx) -> none\n", (unsigned long long)a0); return 0x8002000e; }
    if (sz >= 0x08) *(uint64_t*)(info + 0x00) = r->start;
    if (sz >= 0x10) *(uint64_t*)(info + 0x08) = r->end;
    if (sz >= 0x14) *(int32_t*)(info + 0x10) = r->type;
    MLOG("dmem_query(0x%llx,f=0x%llx) -> [0x%llx,0x%llx) type=%d\n",
         (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)r->start, (unsigned long long)r->end, r->type);
    return 0;
}

// sceKernelMapDirectMemory(void** addrInOut, size_t len, int prot, int flags, off_t phys, size_t align)
HLE(k_map_dmem) {
    uint64_t hint = a0 ? *(uint64_t*)a0 : 0;
    void* p = map_phys_at(hint, a1, host_prot(a2), a4);
    if (!p) { MLOG("map_dmem hint=0x%llx len=0x%llx FAILED\n", (unsigned long long)hint, (unsigned long long)a1); return 0x16; }
    if (a0) *(uint64_t*)a0 = (uint64_t)p;
    track((uint64_t)p, a1, host_prot(a2), true, "direct");
    MLOG("map_dmem -> 0x%llx len=0x%llx phys=0x%llx prot=0x%llx\n",
         (unsigned long long)p, (unsigned long long)a1, (unsigned long long)a4, (unsigned long long)a2);
    return 0;
}

// sceKernelVirtualQuery(const void* addr, int flags, SceKernelVirtualQueryInfo* info, size_t infoSize)
//   0x00 start; 0x08 end; 0x10 offset; 0x18 i32 prot; 0x1C i32 memType; 0x20 u32 flags; 0x24 name[32]
HLE(k_virtual_query) {
    if (!a2) return 0x16;
    uint8_t* info = (uint8_t*)a2;
    uint64_t sz = a3 ? (a3 > 0x48 ? 0x48 : a3) : 0x48;
    memset(info, 0, sz);
    const Mapping* m = find(a0);
    uint64_t start, end; int prot; uint32_t flags;
    const char* how;
    if (m) {                                   // inside a real mapping
        // Report the mapping's REAL commit state: a reserved-but-uncommitted range has NO access
        // (prot 0). Lying prot=RW for the whole 512GB reservation made UE4's allocator skip its
        // BatchMap commit for pages VirtualQuery claimed were already writable -> first-touch crash.
        start = m->base; end = m->base + m->size;
        prot = m->committed ? 0x3 : 0x0; flags = m->committed ? 0x10 : 0x0; how = "tracked";
    } else {                                   // unmapped: report the whole hole to the next mapping
        uint64_t nb = next_base(a0);
        if (!nb) { MLOG("virtual_query(0x%llx) -> end-of-space (EACCES)\n", (unsigned long long)a0); return 0x8002000e; }
        start = a0 & ~(uint64_t)0x3fff; end = nb; prot = 0; flags = 0; how = "hole";
    }
    if (sz >= 0x08) *(uint64_t*)(info + 0x00) = start;
    if (sz >= 0x10) *(uint64_t*)(info + 0x08) = end;
    if (sz >= 0x1c) *(int32_t*)(info + 0x18) = prot;
    if (sz >= 0x24) *(uint32_t*)(info + 0x20) = flags;
    if (sz >= 0x44 && m && m->name[0]) memcpy(info + 0x24, m->name, 32);
    MLOG("virtual_query(0x%llx,f=0x%llx) -> [0x%llx,0x%llx) %s\n",
         (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)start, (unsigned long long)end, how);
    return 0;
}

// libkernel_sync_on_address — PS5 futex. Guest and host share the address space, so the
// wait address is a real host address we can pass to Linux futex(2). Signature inferred
// as wait(addr, expectedValue32, ...) / wake(addr, count). First calls log their args so
// we can confirm the ABI.
namespace { bool synclog() { static int v = getenv("PROSPER_SYNCLOG") ? 1 : 0; return v; } }
HLE(k_wait_on_address) {
    if (!a0) return 0;
    // Args (verified against the game's wait wrapper eboot+0x19b4050, called as (addr, 0, &timeout, 0)):
    //   a0 = wait address, a1 = expected 32-bit value (blocks while *addr == a1 — confirmed: the waits
    //   that block do so with a1 == *addr), a2 = pointer to the timeout.
    // CONFIDENCE: HIGH addr/expected (empirically correct — 80+ waits block+wake normally).
    // CONFIDENCE: MED  a2 = timeout pointer, *a2 = uint32 microseconds (the game stores it via a 32-bit
    //   `mov %eax` after a double→int convert). We previously IGNORED the timeout and passed nullptr to
    //   futex → the guest's *bounded* semaphore-wait blocked FOREVER and could never reach its
    //   timeout-exhausted branch. Honor it so timed waits actually time out and the guest re-evaluates.
    struct timespec ts, *pts = nullptr;
    // Honor the guest's timeout by DEFAULT (#139): ignoring it made a bounded semaphore-wait block
    // FOREVER and return 0 (=signaled), so the guest consumed a resource never produced (phantom item
    // -> crash). The original reason for keeping this gated off — the guest's timeout branch threw a
    // C++ exception whose unwind crashed via the then-unimplemented sceKernelGetModuleInfoForUnwind —
    // was fixed the same day (ca17aa9) and is stale: a 240 s run honoring the timeout on The Messenger
    // showed no unwind/eh_frame fault. PROSPER_NO_WAIT_TIMEOUT restores infinite waits for bisection.
    static const bool honor_timeout = getenv("PROSPER_NO_WAIT_TIMEOUT") == nullptr;
    if (a2 && honor_timeout) {
        uint32_t us = *(volatile uint32_t*)(uintptr_t)a2;
        // FUTEX_WAIT's timeout is a RELATIVE duration. Honor the guest's EXACT value (uint32 µs, so at
        // most ~71 min — no timespec overflow); the old hardcoded 5 s cap turned a longer legitimate
        // timeout into an early ETIMEDOUT. Only 0 ("poll") is nudged to a minimal wait so we re-check.
        if (us == 0) us = 1;
        ts.tv_sec = us / 1000000u; ts.tv_nsec = (long)(us % 1000000u) * 1000L; pts = &ts;
    }
    // --- DIAGNOSTIC punch-through (PROSPER_PUNCH=<secs>) -----------------------------------------
    // The render loop deadlocks: 3 threads block on sync_on_address semaphores whose producer never
    // runs (it's gated on GPU completion we don't yet provide). To learn what's DOWNSTREAM of the
    // deadlock — does clearing it lead to draws/rendering, or another wall? — this makes an INFINITE
    // wait (a2==0) that stays blocked past `secs` fabricate a signal: bump *addr and return success, as
    // if the resource were produced. This is a deliberate FAKE for observation only (gated, off by
    // default). CONFIDENCE: LOW — the punched thread proceeds on a phantom resource; a resulting crash
    // location is itself the diagnostic (it names what the producer was supposed to set up).
    static const long punch_secs = getenv("PROSPER_PUNCH") ? atol(getenv("PROSPER_PUNCH")) : 0;
    static std::atomic<int> punch_budget{64};   // cap total fabricated signals so it can't run away
    // Only punch the specific deadlocked wait sites (return addresses in eboot, from the gdb thread map),
    // NOT the job pool (0xae1af9 / 0x18ab088) — those need real jobs and crash on a phantom. Whitelist:
    // main PreloadManager 0x18a83b5, GfxDevice work-queue 0xb0672a, worker 0x9385d7.
    uint64_t gra = ((uint64_t*)__builtin_frame_address(0))[1];   // guest return address (stub tail-jumped)
    uint64_t goff = gra - 0x400000000ull;
    bool punch_site = (goff == 0x18a83b5 || goff == 0xb0672a || goff == 0x9385d7);
    if (punch_secs > 0 && a2 == 0 && !pts && punch_site) {
        ts.tv_sec = punch_secs; ts.tv_nsec = 0; pts = &ts;   // bound this otherwise-infinite wait
    }
    if (synclog()) fprintf(stderr, "[sync] T%ld WAIT.enter  addr=0x%llx *addr=0x%x exp=0x%llx timo_us=%lld caller=eboot+0x%llx\n",
                           (long)syscall(SYS_gettid), (unsigned long long)a0, *(uint32_t*)a0,
                           (unsigned long long)a1, pts ? (long long)(ts.tv_sec*1000000 + ts.tv_nsec/1000) : -1,
                           (unsigned long long)goff);
    futex_wait_enter();   // registers this waiter so GPU-side label wakes know someone is blocked
    long r = syscall(SYS_futex, (uint32_t*)a0, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, (uint32_t)a1, pts, nullptr, 0);
    int e = errno;
    futex_wait_exit();
    if (synclog()) fprintf(stderr, "[sync] T%ld WAIT.exit   addr=0x%llx r=%ld errno=%d\n",
                           (long)syscall(SYS_gettid), (unsigned long long)a0, r, r < 0 ? e : 0);
    // Return the RIGHT status to the guest. Previously we always returned 0 (=woken/success), so on a
    // timeout the guest believed its semaphore was signaled → it consumed a resource that was never
    // produced → phantom item → crash in the exception unwinder. On a futex timeout, report the Sony
    // timeout error so the guest's bounded wait re-loops / handles it instead of proceeding on garbage.
    // CONFIDENCE: HIGH on the semantics (must distinguish timeout from wake).
    // CONFIDENCE: MED on the exact code value 0x80020060 (SCE_KERNEL_ERROR_ETIMEDOUT).
    if (r < 0 && e == ETIMEDOUT) {
        if (punch_secs > 0 && punch_site && punch_budget.load() > 0 && *(volatile uint32_t*)(uintptr_t)a0 == (uint32_t)a1) {
            // Fabricate the awaited signal: bump the count so the guest's loop acquires and proceeds.
            punch_budget.fetch_sub(1);
            *(volatile uint32_t*)(uintptr_t)a0 = (uint32_t)a1 + 1;
            futex_wake(a0, INT_MAX);
            fprintf(stderr, "[punch] T%ld addr=0x%llx exp=0x%llx -> fabricated signal (ra=eboot+0x%llx, budget=%d)\n",
                    (long)syscall(SYS_gettid), (unsigned long long)a0, (unsigned long long)a1,
                    (unsigned long long)(gra - 0x400000000ull), punch_budget.load());
            return 0;
        }
        return 0x80020060ull;   // SCE_KERNEL_ERROR_ETIMEDOUT
    }
    return 0;   // woken, or EAGAIN (value already changed) — treat as success; guest re-checks the count
}
HLE(k_wake_by_address) {
    if (!a0) return 0;
    int n = a1 ? (int)a1 : INT_MAX;
    futex_wake(a0, n);
    if (synclog()) {
        uint64_t wgoff = ((uint64_t*)__builtin_frame_address(0))[1] - 0x400000000ull;
        fprintf(stderr, "[sync] T%ld WAKE       addr=0x%llx *addr=0x%x n=%d caller=eboot+0x%llx\n",
                (long)syscall(SYS_gettid), (unsigned long long)a0, *(uint32_t*)a0, n, (unsigned long long)wgoff);
    }
    return 0;
}

HLE(k_munmap)   { if (a0) { munmap((void*)a0, a1); untrack(a0, a1); } return 0; }
HLE(k_mprotect) { if (a0) { mprotect((void*)a0, a1, host_prot(a2)); retrack_prot(a0, a1, host_prot(a2), "mprotect"); } return 0; }
HLE(k_dmem_size){ return kDmemTotal; }   // 8 GiB pool (allocation failures enforce this bound)
// sceKernelAvailableDirectMemorySize(searchStart, searchEnd, alignment, off_t* physAddrOut,
// size_t* sizeOut) — report the LARGEST free aligned direct-memory block in [searchStart, searchEnd).
// Signature per shadPS4 memory.cpp:120 (NID C0f7TJcbfac, PS4-inherited). Was aliased to
// k_dmem_size, which returned the pool TOTAL and never wrote the out-params — a caller sizing an
// allocation from *sizeOut or seeding a search from *physAddrOut read uninitialized stack after a
// "success". Now walks the real free gaps.
HLE(k_avail_dmem) {   // a0=searchStart a1=searchEnd a2=alignment a3=physAddrOut a4=sizeOut
    if (!a3 || !a4) return 0x80020016ull;                 // EINVAL: out-params required
    uint64_t off = 0, size = 0;
    dmem_largest_free(a0, a1 ? a1 : (kDmemBase + kDmemTotal), a2, off, size);
    if (!size) return 0x8002000Cull;                      // ENOMEM: nothing available in the window
    *(uint64_t*)(uintptr_t)a3 = off;
    *(uint64_t*)(uintptr_t)a4 = size;
    MLOG("avail-dmem [0x%llx,0x%llx) align=0x%llx -> phys=0x%llx size=0x%llx\n",
         (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
         (unsigned long long)off, (unsigned long long)size);
    return 0;
}

// --- libSceAmpr (PS5 async memory-programming engine) -------------------------------------------
// The UE4 title (PPSA17942) commits one class of its allocator pool pages through an Ampr command
// buffer instead of BatchMap. Live-captured sequence per page (gdb at the import stubs):
//   8aI7R7WaOlc(ctx, 0, 1, 0, -1, 0x720)          — command-buffer init (0x720 = size?)
//   a8uLzYY--tM(ctx, ctx+0x18, ctx+0x20, 0, …)    — begin/get-cursors (outputs unused before push)
//   N-FSPA4S3nI(ctx, va, 0x10000, 0, flags, 0)    — push "map 64KB page at va"
// We execute the mapping SYNCHRONOUSLY at push time (AMM allocates backing physical pages
// internally on real hardware; anonymous host pages model that). Names unknown — registered by
// raw NID. CONFIDENCE: MED (semantics from live arg capture at all three call sites).
HLE(k_ampr_ok) { return 0; }
// DIAGNOSTIC (PROSPER_AMPRLOG=1): log every Ampr init/begin call with args and, for begin, the
// CURRENT contents of the two out-cursor slots (a1/a2). Purpose: test whether the engine derives
// the APR read destination from a begin-cursor we never populate (k_ampr_ok returns 0 without
// writing outputs) — i.e. whether the stale value in *(a1)/*(a2) is where the bogus
// req+0x20 = freed-pool-block pointer comes from.
namespace { bool amprlog() { static int v = getenv("PROSPER_AMPRLOG") ? 1 : 0; return v; } }
// Last Ampr command-buffer (address/size) seen at init: the APR read-submit path inits its request
// with (req, cbSize, 0, cbBuf, poolCtx, 3) — the actual read COMMAND (file offset / dest / size)
// lives inside cbBuf (the "CB offset 40" of the guest's error message is a byte offset into it).
// Exposed so the read-submit HLE can locate and decode the real command. CONFIDENCE: MED.
uint64_t g_apr_last_cb = 0, g_apr_last_cb_size = 0;
// Per-cb capacity, recorded at init (issue #208 follow-up). Two live-captured init flavors carry
// the size in different args: the APR read flow inits (req, cbSize=a1, 0, cbBuf=a3, poolCtx, 3),
// and the IoStore batch-append cb inits (cb=a0, 0, 0, 0, -1, size=a5 — observed a5=0x720). The
// guest's append loop (eboot+0x227e2c0) polls GetSize(cb) - GetUsed(cb) and only appends the next
// command when the difference exceeds 0xff — with the old global "last size" (0 for this cb) the
// loop spun forever, parking the IoStore thread and stalling the whole post-shader-map load.
namespace {
    std::mutex g_ampr_cb_mx;
    std::unordered_map<uint64_t, uint64_t> g_ampr_cb_size;   // cb ctx -> byte capacity
}
HLE(k_ampr_init) {
    if (a3 > 0xffff && a1 && a1 <= 0x10000) { g_apr_last_cb = a3; g_apr_last_cb_size = a1; }
    if (a0 > 0xffff) {
        uint64_t sz = (a1 && a1 <= 0x100000) ? a1 : (a5 && a5 <= 0x100000) ? a5 : 0;
        if (sz) {
            std::lock_guard<std::mutex> lk(g_ampr_cb_mx);
            g_ampr_cb_size[a0] = sz;
            if (g_ampr_cb_size.size() > 4096) g_ampr_cb_size.clear();   // bound (ctxs recycle)
        }
    }
    if (amprlog()) {
        fprintf(stderr, "[amprlog] init  a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx a4=0x%llx a5=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
        if (a0 > 0xffff) for (int o = 0; o < 0x40; o += 8)
            fprintf(stderr, "[amprlog]   ctx+0x%02x = 0x%016llx\n", o, (unsigned long long)*(uint64_t*)(a0 + o));
    }
    return 0;
}
// DIAGNOSTIC (PROSPER_AMPRLOG=1): arg capture for the four other libSceAmpr NIDs the APR read flow
// touches (baQO9ez2gL4 / ULvXMDz56po / Qs1xtplKo0U / GuchCTefuZw, previously anonymous unimpl
// stubs). One of these likely carries the READ RANGE: the pak reads want the FPakInfo footer at
// filesize-0xdd (a5 of the submit = 0xdd = footer size, 0x90 = utoc header size), so a per-read
// {offset,size} must flow through some pre-submit call. CONFIDENCE: LOW until captured.
static uint64_t ampr_arglog(const char* tag, uint64_t a0, uint64_t a1, uint64_t a2,
                            uint64_t a3, uint64_t a4, uint64_t a5) {
    if (amprlog()) {
        fprintf(stderr, "[amprlog] %s a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx a4=0x%llx a5=0x%llx\n",
                tag, (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
        for (uint64_t p : { a0, a1, a2 }) if (p > 0xffff && !(p & 7))
            fprintf(stderr, "[amprlog]   [0x%llx] = 0x%016llx 0x%016llx\n", (unsigned long long)p,
                    (unsigned long long)*(uint64_t*)p, (unsigned long long)*(uint64_t*)(p + 8));
    }
    return 0;
}
HLE(k_ampr_x1) { return ampr_arglog("baQO9ez2gL4", a0, a1, a2, a3, a4, a5); }
HLE(k_ampr_x2) { return ampr_arglog("ULvXMDz56po", a0, a1, a2, a3, a4, a5); }
// sceAmprCommandBufferGetSize (NID tZDDEo2tE5k, recovered by brute-forcing nid_hash over a
// generated libSceAmpr corpus). Returns the command buffer's byte CAPACITY. Live contract
// (issue #208 follow-up, guest append loop eboot+0x227e2c0, wrapper 0x59b5dd0): the IoStore
// batch builder polls `GetSize(cb) - <companion>(cb) > 0xff` before appending the next command
// packet — GetSize is the fixed capacity and the companion (0x59b5e00, currently a 0-returning
// stub) is the used/pending byte count. prosper serves every appended command synchronously at
// ReadFile time, so "used" is truthfully always 0 and the free space is the full capacity.
// Returns: the per-cb capacity recorded at init; falls back to the legacy "last cb size" global,
// then to a roomy 0x10000 (a starved 0 here parks the IoStore thread in a permanent poll spin —
// the post-shader-map wall this fixes). CONFIDENCE: MED (semantics inferred from the decompiled
// append loop + live spin -> unblock flip; name from the verified NID corpus).
HLE(k_ampr_getsize) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    ampr_arglog("tZDDEo2tE5k(GetSize)", a0, a1, a2, a3, a4, a5);
    {
        std::lock_guard<std::mutex> lk(g_ampr_cb_mx);
        auto it = g_ampr_cb_size.find(a0);
        if (it != g_ampr_cb_size.end()) return it->second;
    }
    return g_apr_last_cb_size ? g_apr_last_cb_size : 0x10000;
}
HLE(k_ampr_x3) { return ampr_arglog("Qs1xtplKo0U", a0, a1, a2, a3, a4, a5); }
HLE(k_ampr_x4) { return ampr_arglog("GuchCTefuZw", a0, a1, a2, a3, a4, a5); }
// --- APR completion-event plumbing (issues #115/#180/#208). Live-captured surface:
//   sSAUCCU1dv4(eq=APREventQueue, id=0x74fe+ring, 0, 0, 0x43, 0)  — register APR ids on an equeue
//     (called 6x, rings 0..5, by the guest listener-ctx ctor eboot+0x22a0670)
//   H896Pt-yB4I(cbCtx, eq, id, tag=(ring<<58)|counter, 0, 7)      — bind a command buffer to the
//     eq; the tag is the GUEST-CHOSEN completion token (counter from the ctor-seeded per-ring
//     sequence starting at 1000)
//   ASoW5WE-UPo(cb, ring_1based, u64* out1, u64* out2)            — libkernel: SUBMIT the APR cb;
//     nonzero return = error (test eax,eax at eboot+0x22a1d55); the completion handler
//     (eboot+0x229dcb0) resubmits the next queued cb with ring = (data>>58)+1 (1-based ring).
// prosper's reads complete synchronously inside the builders, so submit == complete: for a bound
// cb, post its H896 tag as the completion event (deferred); for an unbound cb, hand a counter
// token through the out slots and post nothing (record-polled). Full contract write-up in
// hle_kernel_time.cpp. CONFIDENCE: HIGH (static disassembly of the guest submit/listener/handler
// + live tag-echo captures).
uint64_t prosper_apr_next_token(unsigned ring);                          // hle_kernel_time.cpp
void prosper_eq_add_apr(uint64_t eq, int64_t id);                        // "
void prosper_eq_post_apr_token(uint64_t eq, int64_t id, uint64_t token); // " (tag echo, #208)
namespace {
    // Command-buffer ctxs bound to the APR event queue via H896Pt-yB4I, with the binding's a3 TAG
    // and target equeue. The tag IS the guest-chosen completion token for that cb ((ring<<58)|n,
    // observed n starting at 0x3e8=1000 and incrementing per binding — live capture 2026-07-09):
    // the guest's listener context (an eboot GLOBAL, ctx+0xa8+ring*0x28 in-flight slot, expected
    // token at [slot+0x10] — read live at the issue-#180 stall: ring-4 slot expecting exactly
    // 0x10000000000003e8, the first H896 tag) tracks the cb under this exact value. The submit
    // must ECHO it through the out slots and the completion event must carry it verbatim — the
    // pre-#180 code posted an invented per-ring counter (seq 1,2,...) that could never match, so
    // the engine believed batch #1 was in flight forever and the whole async IO pipeline jammed
    // behind it (the CreateGlobalShaderMap precache stall).
    struct AprBoundCb { uint64_t cb, tag, eq; int64_t id; };
    std::mutex g_apr_bound_mx;
    std::vector<AprBoundCb> g_apr_bound_cbs;
    bool apr_cb_bound(uint64_t cb, AprBoundCb* out = nullptr) {
        std::lock_guard<std::mutex> lk(g_apr_bound_mx);
        for (auto& b : g_apr_bound_cbs)
            if (b.cb == cb) { if (out) *out = b; return true; }
        return false;
    }
    // Per-request "eventful" marks for UNBOUND one-shot cbs (issue #180). The engine drives two
    // flavors of sceAmprAprCommandBufferReadFile through ONE wrapper (identical guest-RA chains):
    //   - sync reads: the caller consumes the completion RECORD inline; NO equeue event may be
    //     posted for them (the listener's hash-miss path faults at eboot+0x229df3e reading 0x10 —
    //     re-verified 2026-07-09: posting events for every submit crashes there within seconds);
    //   - async streaming reads: the submitting worker returns to its pool immediately (its frame
    //     is dead at stall time — verified live) and the ONLY completion path is the APREventQueue
    //     event -> FAPREventQueueListener -> completion handler -> FEvent. Suppressing their event
    //     stalls the boot forever in IAsyncReadRequest::WaitCompletion (spin RA eboot+0x22ea954).
    // The observable that separates them (only ReadFile-level distinction found across 4 runs /
    // 90 reads each): the async wrapper passes a live 8th argument — a pointer into its OWN stack
    // frame just above the request object (arg8 - req == +0x94 in every capture, 3 independent
    // runs) — while sync callsites leave frame residue there (small ints, code addresses, heap
    // garbage). f_apr_read_submit marks each request eventful/sync at ReadFile time; the ASoW
    // submit consumes the mark. CONFIDENCE: MED — the discriminator is empirical (single eventful
    // specimen at implementation time, deterministic across runs; every subsequent streaming read
    // of a full boot validates or refutes it loudly: a missed eventful read re-stalls, a false
    // positive crashes the listener).
    std::mutex g_apr_eventful_mx;
    std::unordered_map<uint64_t, bool> g_apr_eventful;   // req/cb ptr -> eventful (updated per ReadFile)
}
void prosper_apr_mark_eventful(uint64_t req, bool eventful) {
    std::lock_guard<std::mutex> lk(g_apr_eventful_mx);
    g_apr_eventful[req] = eventful;   // stack frames are reused: update, don't accumulate stale marks
    if (g_apr_eventful.size() > 4096) g_apr_eventful.clear();   // bound (frames recycle constantly)
}
namespace {
    bool apr_req_eventful(uint64_t req) {
        std::lock_guard<std::mutex> lk(g_apr_eventful_mx);
        auto it = g_apr_eventful.find(req);
        return it != g_apr_eventful.end() && it->second;
    }
}
HLE(k_apr_set_equeue) {          // sSAUCCU1dv4 (eq, id, 0, 0, 0x43, 0)
    ampr_arglog("sSAUCCU1dv4(SetEqueue)", a0, a1, a2, a3, a4, a5);
    // Called 6x by the guest listener-ctx ctor (eboot+0x22a0670) with id = 0x74fe + ring for
    // rings 0..5, right after it creates its own equeue and seeds the per-ring counters.
    if (a0) prosper_eq_add_apr(a0, (int64_t)a1);
    return 0;
}
HLE(k_apr_cb_set_equeue) {       // H896Pt-yB4I (cb, eq, id, tag, 0, flags)
    ampr_arglog("H896Pt-yB4I(CbSetEqueue)", a0, a1, a2, a3, a4, a5);
    if (a1) prosper_eq_add_apr(a1, (int64_t)a2);
    if (a0) {
        std::lock_guard<std::mutex> lk(g_apr_bound_mx);
        bool seen = false;
        for (auto& b : g_apr_bound_cbs)
            if (b.cb == a0) { b.tag = a3; b.eq = a1; b.id = (int64_t)a2; seen = true; break; }
        if (!seen) g_apr_bound_cbs.push_back({ a0, a3, a1, (int64_t)a2 });
    }
    return 0;
}
HLE(k_apr_submit) {              // libkernel ASoW5WE-UPo (cb, ring_1based, out1, out2)
    unsigned ring = a1 ? (unsigned)(a1 - 1) & 0x3f : 0;
    // Completion-token contract (issues #180/#208 — guest submit path eboot+0x22a02b0, handler
    // +0x229dcb0, listener +0x22740b0, listener-ctx ctor +0x22a0670; full write-up in
    // hle_kernel_time.cpp and docs/UE4_APR_IOSTORE_BRINGUP.md):
    //   - H896-BOUND cb (the batched/streaming channel): the completion token IS the binding tag,
    //     (ring<<58)|counter with the counter drawn from the guest's own per-ring sequence seeded
    //     at 1000 ([ctx+0xc0+ring*0x28], ctor-initialized 0x3e8). The guest tracks it at
    //     [slot+0x10] AND in a {token -> callback} hash; the listener's walk is ctor-seeded to
    //     start exactly at 1000 (last-processed = 0x3e7). We post the tag VERBATIM (deferred) on
    //     the binding's own equeue — nothing else. The out slots are NOT written: a2/a3 alias the
    //     request's completion RECORD ({status@req+0x28, bytes@req+0x30}, already completed
    //     {0,size} by ReadFile) — writing a token there marks the record FAILED (nonzero status,
    //     checked at eboot+0x22738a5; live-verified "GEngineLoop.PreInit Failed!").
    //   - UNBOUND cb (mount-era sync flow, ring_1b=6 async-archive flow): hand out our own
    //     per-ring counter token through the out slots (the engine stores it verbatim where it
    //     tracks the read) and post NO event — these flows are completion-record-polled, and any
    //     invented-counter event would REGRESS the listener's ctor-seeded last-processed via its
    //     unconditional last:=cnt store (+0x2274143), setting up the fatal +0x229df3e walk (the
    //     #180 residual fault). CONFIDENCE: HIGH (static disassembly + live tag-echo runs).
    AprBoundCb bc{};
    const bool bound = apr_cb_bound(a0, &bc);
    const bool tag_echo = bound && bc.tag && bc.eq;
    uint64_t token = tag_echo ? bc.tag : prosper_apr_next_token(ring);
    if (!bound) {
        if (a2 > 0xffff) *(uint64_t*)(uintptr_t)a2 = token;
        if (a3 > 0xffff) *(uint64_t*)(uintptr_t)a3 = token;
    }
    if (amprlog()) fprintf(stderr, "[amprlog] ASoW5WE-UPo(Submit) cb=0x%llx ring1b=%llu out1=0x%llx out2=0x%llx -> token=0x%llx%s%s\n",
                           (unsigned long long)a0, (unsigned long long)a1,
                           (unsigned long long)a2, (unsigned long long)a3, (unsigned long long)token,
                           bound ? " (bound)" : "", apr_req_eventful(a0) ? " (arg8-async)" : "");
    if (tag_echo) prosper_eq_post_apr_token(bc.eq, bc.id, bc.tag);
    return 0;
}
// Still-unnamed mount-time Ampr NID (state query?); arg-logging no-op under PROSPER_AMPRLOG.
HLE(k_ampr_u3) { return ampr_arglog("GnxKOHEawhk", a0, a1, a2, a3, a4, a5); }
HLE(k_ampr_begin) {
    if (amprlog()) {
        fprintf(stderr, "[amprlog] begin a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx a4=0x%llx a5=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
        if (a1 > 0xffff) fprintf(stderr, "[amprlog]   *out1 = 0x%016llx\n", (unsigned long long)*(uint64_t*)a1);
        if (a2 > 0xffff) fprintf(stderr, "[amprlog]   *out2 = 0x%016llx\n", (unsigned long long)*(uint64_t*)a2);
        if (a0 > 0xffff) for (int o = 0; o < 0x40; o += 8)
            fprintf(stderr, "[amprlog]   ctx+0x%02x = 0x%016llx\n", o, (unsigned long long)*(uint64_t*)(a0 + o));
    }
    // POPULATE the out-slots with a library-owned staging buffer. Root cause of the eboot+0x2316c91
    // freelist crash (UE4 PPSA17942 IoStore): the APR read flow is
    //   init(req, cbSize, 0, pathStruct, allocCtx, 3); begin(req, &req->0x18, &req->0x20); submit(req,…)
    // and after completion the ENGINE reads the file bytes through *(req+0x20) — i.e. begin's second
    // out-slot is the data pointer the library chooses. Returning 0 WITHOUT writing the slots left
    // stale stack garbage in req+0x20 that happened to point at a FREED 0x50-byte block of the
    // engine's own live path-string pool (freelist head 0x2001c10080, blocks 0x1080e10xxx — proven by
    // a full-run HW write-watch: the block was carve-linked, churned as UTF-16 path storage, and was
    // sitting FREE on the freelist when the APR read wrote the 645-byte TOC over it, clobbering the
    // free nodes' next pointers with the TOC magic). Both we and the engine dereferenced the same
    // stale slot, which is why the TOC still parsed while the pool corrupted. On real hardware the
    // Ampr library returns its own staging pointer here; model that with a prosper-owned buffer.
    // The engine consumes the data synchronously after submit, so one buffer serves consecutive
    // reads. CONFIDENCE: MED (out-slot semantics inferred from live capture + the NOWRITE/WRITELEN
    // experiments; staging size generous for the .ucas chunk reads that follow).
    static void* staging = [] {
        void* p = mmap(nullptr, 16ull << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return p == MAP_FAILED ? nullptr : p;
    }();
    if (staging) {
        if (a1 > 0xffff) *(uint64_t*)a1 = (uint64_t)staging;
        if (a2 > 0xffff) *(uint64_t*)a2 = (uint64_t)staging;
    }
    return 0;
}
HLE(k_ampr_push_map) {
    MLOG("ampr SetBuffer args a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx a4=0x%llx a5=0x%llx\n",
         (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
         (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
    if (a1 && a2) {
        // Back the page from the shared phys pool so BOTH pool views alias the same bytes: the
        // guest WRITES its MallocBinned pool-page headers through this (high) view and READS them
        // through a second view exactly 0x540000000 lower (empirically pinned — the two views'
        // canary reads returned zeros with independent anonymous pages). Map the mirror too.
        // CONFIDENCE: LOW on the 0x540000000 rule (observed constant, one title) — refine by
        // finding the guest's own second-view creation; the aliasing itself is the HW contract.
        //
        // TWO FLAVORS of SetBuffer, discriminated by live arg capture (issue #88 root cause):
        //   MAP flavor      — (cb, va, len, a3!=va, a4=0xffffffff/-1 sentinel, a5=0/0x720):
        //                     "map fresh direct memory at va" (the AMM pool flow; the guest
        //                     expects FRESH ZEROED pages, like sceKernelMapDirectMemory).
        //   BUFFER flavor   — (cb, va, len, a3==va, a4=allocCtx e.g. 0x2001c10060, a5=3/0x2d):
        //                     "use this ALREADY-EXISTING buffer" (the APR read flow's 0x40-byte
        //                     completion records and its 0x4000 descriptor buffer). Real HW does
        //                     NOT touch the buffer's memory here.
        // The old handler treated BOTH as map requests. For the BUFFER flavor the target
        // (va=0x15a0dfc000 len=0x4000) — and worse, the va-0x540000000 "second view" mirror —
        // landed on LIVE MallocBinned heap holding the console manager's registered-CVar name
        // strings; MAP_FIXED silently replaced them with fresh zero pages (invisible to HW
        // watchpoints: no CPU store ever happened). The zeroed key made FEngineModule::
        // StartupModule's FindConsoleVariable("r.Shadow.CacheWPOPrimitives") return null ->
        // null virtual call -> rip=0: the issue-#88 crash. Registering the buffer is a no-op for
        // memory state, so the BUFFER flavor now leaves memory strictly untouched.
        // CONFIDENCE: HIGH on the discriminator (a3==a1 in all 13 captured buffer-flavor calls,
        // a3!=a1 + a4 sentinel in all 3 captured map-flavor calls) and on the clobber diagnosis
        // (single-run reg-time vs crash-time key dump + silent HW write-watch).
        if (a3 == a1) {
            MLOG("ampr set-buffer va=0x%llx len=0x%llx ctx=0x%llx flags=0x%llx -> no-op (existing buffer)\n",
                 (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)a4,
                 (unsigned long long)a5);
            return 0;
        }
        // Back the page from the shared phys pool so BOTH pool views alias the same bytes: the
        // guest WRITES its MallocBinned pool-page headers through this (high) view and READS them
        // through a second view exactly 0x540000000 lower (empirically pinned — the two views'
        // canary reads returned zeros with independent anonymous pages). Map the mirror too.
        // CONFIDENCE: LOW on the 0x540000000 rule (observed constant, one title) — refine by
        // finding the guest's own second-view creation; the aliasing itself is the HW contract.
        uint64_t phys = 0;
        bool have_phys = dmem_take(a2, 0x10000, 0x0c, phys);
        void* p = have_phys ? map_phys_at(a1, a2, PROT_READ | PROT_WRITE, phys)
                            : map_at(a1, a2, PROT_READ | PROT_WRITE);
        if (p) track((uint64_t)p, a2, PROT_READ | PROT_WRITE, true, "ampr-map");
        if (p && have_phys && a1 > 0x540000000ull + 0x1000000000ull) {
            uint64_t mirror = a1 - 0x540000000ull;
            // issue #107: the mirror is a LOW-confidence heuristic (the 0x540000000 rule was pinned
            // on one title, The Messenger). On UE4 PPSA17942 the mirror VA lands squarely on LIVE
            // MallocBinned heap that the guest already lazy-committed and filled (e.g. mirror
            // 0x11e0df0000 aliases Ampr buffer phys 0x10220000 but was committed as heap earlier):
            // MAP_FIXED'ing the aliased page there DESTROYS the heap page, corrupting the pool so it
            // later carves two blocks 0x10 apart. That overlap is what makes FConfigCacheIni's
            // teardown free a bookkeeping word (0xd) as a pointer -> "FMallocBinned3 Attempt to free
            // an unrecognized block". This is the same clobber class as issue #88 (SetBuffer over
            // live heap), here via the map-flavor mirror. Only create the mirror when its target is
            // NOT already backed guest memory (mincore succeeds == fully mapped == live): that keeps
            // the Messenger's genuine two-view pool working (its mirror target is unmapped at map
            // time) while never overwriting a live heap page. CONFIDENCE: HIGH (the collision is
            // proven by MEMLOG: page 0x11e0df0000 appears as both [lazy-commit] heap and ampr
            // mirror; the mirror map is strictly later, so it clobbers the live page).
            unsigned char vec;
            bool mirror_live = (mincore((void*)(uintptr_t)mirror, 1, &vec) == 0);
            void* q = nullptr;
            if (mirror_live) {
                MLOG("ampr push-map va=0x%llx mirror=0x%llx SKIPPED (target is live guest memory — "
                     "map-flavor mirror would clobber MallocBinned heap, issue #107)\n",
                     (unsigned long long)a1, (unsigned long long)mirror);
            } else {
                q = map_phys_at(mirror, a2, PROT_READ | PROT_WRITE, phys);
                if (q) track((uint64_t)q, a2, PROT_READ | PROT_WRITE, true, "ampr-mirror");
            }
            MLOG("ampr push-map va=0x%llx len=0x%llx phys=0x%llx mirror=0x%llx -> %s/%s\n",
                 (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)phys,
                 (unsigned long long)mirror, p ? "ok" : "FAIL",
                 mirror_live ? "skip" : (q ? "ok" : "FAIL"));
        } else {
            MLOG("ampr push-map va=0x%llx len=0x%llx -> %s\n",
                 (unsigned long long)a1, (unsigned long long)a2, p ? "ok" : "FAILED");
        }
    }
    return 0;
}

// sceKernelReleaseDirectMemory(off_t start, size_t len) — return a range to the pool. Any VA still
// mapped to it is the guest's problem (same as the real kernel); UE4 releases only unmapped probe
// allocations here.
HLE(k_release_dmem) {
    dmem_release(a0, a1);
    MLOG("release_dmem [0x%llx,0x%llx)\n", (unsigned long long)a0, (unsigned long long)(a0 + a1));
    return 0;
}

// Most-specific tracked-mapping state at `addr` for the fault handler's lazy-commit probe:
// 0 = untracked, 1 = reserved-but-uncommitted, 2 = committed. Called from a signal handler on the
// FAULTING thread — that thread is in guest code, so it cannot itself hold g_mx (only HLE memory
// entry points take it, briefly); a contended lock just waits for the other thread's release.
extern "C" int prosper_reserved_range_state(uint64_t addr) {
    std::lock_guard<std::mutex> lk(g_mx);
    const Mapping* best = nullptr;
    for (auto& m : g_maps)
        if (addr >= m.base && addr < m.base + m.size)
            if (!best || m.base > best->base) best = &m;
    return best ? (best->committed ? 2 : 1) : 0;
}

// sceKernelBatchMap(SceKernelBatchMapEntry* entries, int numberOfEntries, int* numberOfEntriesOut)
// Entry (0x20 bytes, Orbis ABI): start@0x00, physOffset@0x08, length@0x10, protection@0x18 (char),
// type@0x19 (char), operation@0x1c (int). Operations: 0=MAP_DIRECT, 1=UNMAP, 2=PROTECT,
// 3=MAP_FLEXIBLE, 4=TYPE_PROTECT. The UE4 title (PPSA17942) commits EVERY 64KB page of its
// allocator arena through 1-entry batches (identified from the eboot's own assert strings:
// "sceKernelBatchMap failed ..."); the previous unimplemented stub returned "success" having
// mapped NOTHING, so the engine's memory never materialized and boot wedged before any file IO.
// CONFIDENCE: MED on the exact entry ABI (PS4-documented layout; call sites disassembled match:
// 0x4000-byte entry buffer = 512 * 0x20, count compared against numberOfEntriesOut).
HLE(k_batch_map) {
    const uint8_t* e = (const uint8_t*)(uintptr_t)a0;
    int n = (int)(int64_t)a1, done = 0;
    if (!e || n < 0) return 0x80020016ull;   // SCE_KERNEL_ERROR_EINVAL
    uint64_t ret = 0;
    for (int i = 0; i < n; i++, e += 0x20) {
        uint64_t start = *(const uint64_t*)(e + 0x00);
        uint64_t phys  = *(const uint64_t*)(e + 0x08);
        uint64_t len   = *(const uint64_t*)(e + 0x10);
        uint8_t  prot  = e[0x18];
        int32_t  op    = *(const int32_t*)(e + 0x1c);
        bool ok = true;
        switch (op) {
            case 0: {                               // MAP_DIRECT: phys-backed (aliasing preserved)
                void* p = map_phys_at(start, len, host_prot(prot), phys);
                ok = (p != nullptr);
                if (ok) track((uint64_t)p, len, host_prot(prot), true, "batch-direct");
                break;
            }
            case 3: {                               // MAP_FLEXIBLE: anonymous
                void* p = map_at(start, len, host_prot(prot));
                ok = (p != nullptr);
                if (ok) track((uint64_t)p, len, host_prot(prot), true, "batch-flex");
                break;
            }
            case 1: if (start) { munmap((void*)(uintptr_t)start, len); untrack(start, len); } break;  // UNMAP
            case 2: case 4:                                                              // PROTECT / TYPE_PROTECT
                if (start) { mprotect((void*)(uintptr_t)start, len, host_prot(prot));
                             retrack_prot(start, len, host_prot(prot), "batch-prot"); } break;
            default: ok = false; ret = 0x80020016ull; break;
        }
        if (!ok) { if (!ret) ret = 0x8002000Cull; break; }   // ENOMEM on a failed map
        done++;
    }
    MLOG("batchmap n=%d done=%d first{op=%d start=0x%llx len=0x%llx prot=0x%x} -> 0x%llx\n",
         n, done, n ? *(const int32_t*)((const uint8_t*)(uintptr_t)a0 + 0x1c) : -1,
         n ? (unsigned long long)*(const uint64_t*)(uintptr_t)a0 : 0ull,
         n ? (unsigned long long)*(const uint64_t*)((const uint8_t*)(uintptr_t)a0 + 0x10) : 0ull,
         n ? *((const uint8_t*)(uintptr_t)a0 + 0x18) : 0, (unsigned long long)ret);
    if (a2) *(int32_t*)(uintptr_t)a2 = done;
    return ret;
}

void register_kernel_mem_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceKernelReserveVirtualRange", k_reserve_vrange);
    R("sceKernelMapNamedFlexibleMemory", k_map_flexible);
    R("sceKernelMapFlexibleMemory", k_map_flexible);
    R("sceKernelAllocateDirectMemory", k_alloc_dmem);
    R("sceKernelAllocateMainDirectMemory", k_alloc_main_dmem);  // 4-arg signature (physOut at arg3)
    R("sceKernelMapDirectMemory", k_map_dmem);
    R("sceKernelMapNamedDirectMemory", k_map_dmem);
    R("sceKernelMunmap", k_munmap);
    R("sceKernelMprotect", k_mprotect);
    R("sceKernelReleaseDirectMemory", k_release_dmem);
    R("sceKernelCheckedReleaseDirectMemory", k_release_dmem);
    R("sceKernelBatchMap", k_batch_map);
    R("sceKernelBatchMap2", k_batch_map);   // (entries, num, out, flags) — extra flags arg ignored
    R("sceKernelVirtualQuery", k_virtual_query);
    R("sceKernelDirectMemoryQuery", k_direct_memory_query);
    R("sceKernelGetDirectMemorySize", k_dmem_size);
    R("sceKernelAvailableDirectMemorySize", k_avail_dmem);
    #undef R
    // sync_on_address futex — registered by raw NID (names not in any public DB yet).
    Hle::register_fn("Hc4CaR6JBL0", (HleFn)k_wait_on_address, "sceKernelWaitOnAddress?");
    Hle::register_fn("q2y-wDIVWZA", (HleFn)k_wake_by_address, "sceKernelWakeByAddress?");
    // libSceAmpr command-buffer trio. NID names recovered by brute-forcing nid_hash() over a
    // generated libSceAmpr corpus (see hle_file.cpp block comment above f_apr_read_submit).
    Hle::register_fn("8aI7R7WaOlc", (HleFn)k_ampr_init,     "sceAmprCommandBufferConstructor");
    Hle::register_fn("a8uLzYY--tM", (HleFn)k_ampr_begin,    "sceAmprAprCommandBufferConstructor");
    Hle::register_fn("N-FSPA4S3nI", (HleFn)k_ampr_push_map, "sceAmprCommandBufferSetBuffer");
    // The rest of the APR read flow: Reset (pre-submit) + the two Destructors (post-submit
    // teardown). Modeled as no-ops (arg capture under PROSPER_AMPRLOG); the read itself is
    // sceAmprAprCommandBufferReadFile in hle_file.cpp.
    Hle::register_fn("baQO9ez2gL4", (HleFn)k_ampr_x1, "sceAmprCommandBufferReset");
    // ULvXMDz56po and tZDDEo2tE5k names recovered by brute-forcing nid_hash() over a generated
    // libSceAmpr corpus (sceAmprCommandBuffer<Verb>): ClearBuffer and GetSize.
    Hle::register_fn("ULvXMDz56po", (HleFn)k_ampr_x2, "sceAmprCommandBufferClearBuffer");
    Hle::register_fn("tZDDEo2tE5k", (HleFn)k_ampr_getsize, "sceAmprCommandBufferGetSize");
    Hle::register_fn("Qs1xtplKo0U", (HleFn)k_ampr_x3, "sceAmprAprCommandBufferDestructor");
    Hle::register_fn("GuchCTefuZw", (HleFn)k_ampr_x4, "sceAmprCommandBufferDestructor");
    // APR completion-event plumbing (issue #115). vWU-odnS+fU (the direct async file read) is
    // registered in hle_file.cpp next to the other APR read path.
    Hle::register_fn("sSAUCCU1dv4", (HleFn)k_apr_set_equeue,    "AprSetEventQueue?");
    Hle::register_fn("H896Pt-yB4I", (HleFn)k_apr_cb_set_equeue, "AprCbSetEventQueue?");
    Hle::register_fn("ASoW5WE-UPo", (HleFn)k_apr_submit,        "AprSubmitCommandBuffer?");
    Hle::register_fn("GnxKOHEawhk", (HleFn)k_ampr_u3,           "AmprUnknown3");
}

} // namespace prosper

#else
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace prosper {

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)

namespace {
std::mutex g_sync_mx;
std::condition_variable g_sync_cv;
}

HLE(k_wait_on_address) {
    if (!a0) return 0;
    auto& raw = *(uint32_t*)(uintptr_t)a0;
    std::atomic_ref<uint32_t> addr(raw);
    uint32_t expected = (uint32_t)a1;
    // Honor the guest timeout by default (#142) — the same fix the Linux futex path got (#139).
    // Previously this global-cv fallback IGNORED the timeout arg and blocked FOREVER while
    // *addr == expected, returning 0 (=signaled), so a Windows-build timed wait could never take its
    // timeout branch. Parse *a2 as uint32 microseconds, wait to that deadline, and return SCE
    // ETIMEDOUT on expiry. PROSPER_NO_WAIT_TIMEOUT restores infinite waits (parity with Linux).
    // The 32-bit compare is a shared limitation with the Linux FUTEX_WAIT path (WaitOnAddress can
    // wait on 1/2/4/8-byte values; only 4 is modeled) — unchanged here.
    static const bool honor_timeout = getenv("PROSPER_NO_WAIT_TIMEOUT") == nullptr;
    std::unique_lock<std::mutex> lk(g_sync_mx);
    if (a2 && honor_timeout) {
        uint32_t us = *(volatile uint32_t*)(uintptr_t)a2;
        if (us == 0) us = 1;   // 0 == "poll" -> a minimal wait so we re-check
        auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(us);
        while (addr.load(std::memory_order_acquire) == expected)
            if (g_sync_cv.wait_until(lk, deadline) == std::cv_status::timeout &&
                addr.load(std::memory_order_acquire) == expected)
                return 0x80020060ull;   // SCE_KERNEL_ERROR_ETIMEDOUT
        return 0;
    }
    while (addr.load(std::memory_order_acquire) == expected) g_sync_cv.wait(lk);
    return 0;
}

HLE(k_wake_by_address) {
    std::lock_guard<std::mutex> lk(g_sync_mx);
    int n = a1 ? (int)a1 : INT_MAX;
    if (n == 1) g_sync_cv.notify_one();
    else        g_sync_cv.notify_all();
    return 0;
}

void register_kernel_mem_hle() {
    Hle::register_fn("Hc4CaR6JBL0", (HleFn)k_wait_on_address, "sceKernelWaitOnAddress?");
    Hle::register_fn("q2y-wDIVWZA", (HleFn)k_wake_by_address, "sceKernelWakeByAddress?");
}

} // namespace prosper
#endif
