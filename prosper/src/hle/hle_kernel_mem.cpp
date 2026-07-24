// hle_kernel_mem.cpp — HLE of libkernel virtual/direct memory.
// PS5 memory model: reserve a virtual range, allocate "direct" (physical) memory as
// an opaque offset, then map it (or flexible memory) into VA. We back it with host
// native VM primitives and TRACK every mapping so VirtualQuery is truthful and so we can
// log/debug the guest's address-space construction.
#include "dispatch.hpp"
#include "nid.hpp"
#include "sync_futex.hpp"   // shared futex wake + waiter registration (also used by the GPU's label wake)
#include "../host/guest_memory_map.hpp"
#include "../host/guest_write_watch.hpp"

namespace prosper {
// #312/#946 huge-reserve redirect threshold, shared by the Linux and Windows reserve paths (one
// definition so the platforms cannot drift): a non-fixed hinted reservation of at least this size
// (only UE's 512 GiB MallocBinned3 flex arena qualifies) is steered off its low hint into the
// guest auto window. See k_reserve_vrange (POSIX) / win_reserve (Windows).
inline constexpr uint64_t kHugeReserveLen = 0x2000000000ull;   // 128 GiB
}

#if defined(__linux__) || defined(__APPLE__)
#include "../host/posix_shim.hpp"
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>   // process_vm_writev — fault-safe APR completion write (#1149)
#include <unistd.h>
#include <fcntl.h>
#ifdef __linux__
#include <linux/futex.h>
#include <linux/falloc.h>  // FALLOC_FL_PUNCH_HOLE / FALLOC_FL_KEEP_SIZE
#endif
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
#include <utility>
#include <vector>

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define HLE7(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6)

namespace {
    bool memlog() { static int v = getenv("PROSPER_MEMLOG") ? 1 : 0; return v; }
    #define MLOG(...) do { if (memlog()) fprintf(stderr, "[memhle] " __VA_ARGS__); } while (0)

    constexpr uint32_t kVirtualQueryFlexible = 0x01;
    constexpr uint32_t kVirtualQueryDirect   = 0x02;
    constexpr uint32_t kVirtualQueryCommitted = 0x10;
    constexpr uint64_t kGuestPageSize = 0x4000;

    bool align_up_multiple(uint64_t value, uint64_t alignment, uint64_t& out) {
        if (!alignment) { out = value; return true; }
        const uint64_t remainder = value % alignment;
        const uint64_t delta = remainder ? alignment - remainder : 0;
        if (delta > UINT64_MAX - value) return false;
        out = value + delta;
        return true;
    }

    bool normalize_guest_page_range(uint64_t addr, uint64_t len,
                                    uint64_t& base_out, uint64_t& len_out) {
        constexpr uint64_t mask = kGuestPageSize - 1;
        if (len > UINT64_MAX - addr) return false;
        const uint64_t raw_end = addr + len;
        if (raw_end > UINT64_MAX - mask) return false;
        const uint64_t base = addr & ~mask;
        const uint64_t end = (raw_end + mask) & ~mask;
        if (end < base) return false;
        base_out = base;
        len_out = end - base;
        return true;
    }

    struct Mapping {
        uint64_t base, size, offset;
        int prot;                  // normalized host CPU-access mask
        uint32_t guest_prot;       // exact SCE protection enum/bits returned to the guest
        int32_t memory_type;        // direct-memory type; meaningful only with is_direct
        uint32_t query_flags;       // is_flexible / is_direct bits (commit state is separate)
        bool committed;
        char name[32];
    };
    std::mutex g_mx;
    std::vector<Mapping> g_maps;
    // Direct ("physical") memory: a bump allocator over a FINITE pool. The pool size is what
    // sceKernelGetDirectMemorySize advertises, and exhaustion MUST fail with ENOMEM like real
    // hardware: guests rely on it — UE4 (PPSA17942) sizes its pool requests from
    // GetDirectMemorySize and allocates chunks in a loop until ENOMEM ends it. A never-failing
    // allocator handed out offsets past the pool, the guest's 512GB-arena block bitmap indexed
    // out of range, and user_malloc_init crashed on the bitmap read.
    constexpr uint64_t kDmemBase  = 0x10000000;
    // Direct-memory budget the pool holds AND sceKernelGetDirectMemorySize advertises. Real PS5
    // reports the GAME budget (aperture minus OS reservation), not the raw 16 GiB aperture; UE
    // sizes allocator pools from this value (#1213 investigation). PROSPER_DMEM_BUDGET_MB
    // overrides for A/B experiments; the default stays the historical 16 GiB pending
    // cross-title verification of a hardware-truthful default.
    const uint64_t kDmemTotal = [] {
        if (const char* v = getenv("PROSPER_DMEM_BUDGET_MB")) {
            const uint64_t mib = strtoull(v, nullptr, 10);
            if (mib >= 1024) return mib * 1024ull * 1024ull;
        }
        return 16ull * 1024 * 1024 * 1024;
    }();
    // Direct ("physical") memory allocations, kept SORTED by start (first-fit allocation walks the
    // gaps). Also serves sceKernelDirectMemoryQuery.
    struct DMem { uint64_t start, end; int type; };
    std::mutex g_dmx;
    std::vector<DMem> g_dmem;

    void rebase_mapping(Mapping& mapping, uint64_t new_base) {
        if ((mapping.query_flags & kVirtualQueryDirect) && new_base > mapping.base)
            mapping.offset += new_base - mapping.base;
        mapping.base = new_base;
    }

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
            uint64_t off = 0;
            if (align_up_multiple(beg, align, off) && off <= end && sz <= end - off) {
                insert_at = i; off_out = off; goto found;
            }
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
            uint64_t aligned = 0;
            if (align_up_multiple(beg, align, aligned) && end > aligned &&
                end - aligned > size_out) {
                off_out = aligned; size_out = end - aligned;
            }
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

    bool dmem_type_at(uint64_t offset, int32_t& type_out) {
        std::lock_guard<std::mutex> lk(g_dmx);
        for (const auto& d : g_dmem) {
            if (offset >= d.start && offset < d.end) {
                type_out = d.type;
                return true;
            }
        }
        return false;
    }

    // Change the memory type of an allocated physical subrange. Carving the allocation keeps
    // GetDirectMemoryType/DirectMemoryQuery truthful after Mtypeprotect or MapDirectMemory2.
    void dmem_retype(uint64_t start, uint64_t len, int32_t type) {
        if (!len || start > UINT64_MAX - len) return;
        const uint64_t end = start + len;
        std::lock_guard<std::mutex> lk(g_dmx);
        std::vector<DMem> out;
        out.reserve(g_dmem.size() + 2);
        for (const auto& d : g_dmem) {
            if (d.end <= start || d.start >= end) {
                out.push_back(d);
                continue;
            }
            if (d.start < start) out.push_back({d.start, start, d.type});
            const uint64_t changed_start = d.start < start ? start : d.start;
            const uint64_t changed_end = d.end > end ? end : d.end;
            out.push_back({changed_start, changed_end, type});
            if (d.end > end) out.push_back({end, d.end, d.type});
        }
        g_dmem.swap(out);
    }

    // A successful host map replaces any reservation record under it. Keeping both as overlays
    // lets the older uncommitted record win same-base queries after the real commit.
    void track(uint64_t base, uint64_t size, int prot, uint32_t guest_prot,
               bool committed, const char* nm, uint32_t query_flags = 0,
               uint64_t offset = 0, int32_t memory_type = 0) {
        if (!size || base > UINT64_MAX - size) return;
        const uint64_t end = base + size;
        {
            std::lock_guard<std::mutex> lk(g_mx);
            std::vector<Mapping> out;
            out.reserve(g_maps.size() + 2);
            for (const auto& old : g_maps) {
                const uint64_t old_end = old.base + old.size;
                if (old_end <= base || old.base >= end) { out.push_back(old); continue; }
                if (old.base < base) {
                    Mapping lo = old; lo.size = base - old.base; out.push_back(lo);
                }
                if (old_end > end) {
                    Mapping hi = old;
                    rebase_mapping(hi, end);
                    hi.size = old_end - end;
                    out.push_back(hi);
                }
            }
            Mapping m{};
            m.base = base;
            m.size = size;
            m.offset = offset;
            m.prot = prot;
            m.guest_prot = guest_prot;
            m.memory_type = memory_type;
            m.query_flags = query_flags;
            m.committed = committed;
            if (nm) { strncpy(m.name, nm, sizeof m.name - 1); }
            out.push_back(m);
            g_maps.swap(out);
        }
        host::notify_guest_mapping_added(base, size, committed && (prot & 0x1));
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
            if (me > end) {
                Mapping hi = m;
                rebase_mapping(hi, end);
                hi.size = me - end;
                out.push_back(hi);
            }
        }
        g_maps.swap(out);
        host::notify_guest_mapping_removed(base, len);
    }
    // Re-tag [base, base+len) with a new protection (mprotect) without changing whether each
    // covered span is guest-committed. A reservation remains a reservation until a MAP operation
    // commits it; VirtualQuery and the lazy-commit probe rely on that distinction (#343).
    void retrack_prot(uint64_t base, uint64_t len, int prot, uint32_t guest_prot,
                      const char* nm) {
        if (!len || base > UINT64_MAX - len) return;
        const uint64_t end = base + len;
        std::vector<Mapping> retagged;
        {
            std::lock_guard<std::mutex> lk(g_mx);
            std::vector<Mapping> out;
            out.reserve(g_maps.size() + 2);
            for (const auto& m : g_maps) {
                const uint64_t me = m.base + m.size;
                if (me <= base || m.base >= end) { out.push_back(m); continue; }
                if (m.base < base) {
                    Mapping lo = m; lo.size = base - m.base; out.push_back(lo);
                }
                Mapping changed = m;
                rebase_mapping(changed, m.base < base ? base : m.base);
                const uint64_t changed_end = me > end ? end : me;
                changed.size = changed_end - changed.base;
                changed.prot = prot;
                changed.guest_prot = guest_prot;
                memset(changed.name, 0, sizeof changed.name);
                if (nm) strncpy(changed.name, nm, sizeof changed.name - 1);
                out.push_back(changed);
                retagged.push_back(changed);
                if (me > end) {
                    Mapping hi = m;
                    rebase_mapping(hi, end);
                    hi.size = me - end;
                    out.push_back(hi);
                }
            }
            // Preserve the prior treatment of a successful protection change over an otherwise
            // untracked host mapping: begin tracking it as committed.
            if (retagged.empty()) {
                Mapping changed{};
                changed.base = base;
                changed.size = len;
                changed.prot = prot;
                changed.guest_prot = guest_prot;
                changed.committed = true;
                if (nm) strncpy(changed.name, nm, sizeof changed.name - 1);
                out.push_back(changed);
                retagged.push_back(changed);
            }
            g_maps.swap(out);
        }
        host::notify_guest_mapping_removed(base, len);
        for (const auto& m : retagged)
            host::notify_guest_mapping_added(m.base, m.size, m.committed && (prot & 0x1));
        // Protection chokepoint for the texture write-watch (#1144 B3): a guest mprotect/mtypeprotect
        // (or batch PROTECT) that flips a watched page's writability out from under our read-only arming
        // must invalidate the watch — otherwise a subsequent CPU store would not fault yet the renderer
        // would still read Unchanged. `guest_prot` carries the Sony bits the watch decodes.
        host::guest_write_watch_notify_direct_mapping_protection(base, len, guest_prot);
    }

    // Re-tag direct mappings with a new memory type while preserving virtual-to-physical offset
    // correspondence across every carved prefix/suffix. Flexible/reserved mappings have no type.
    void retrack_type(uint64_t base, uint64_t len, int32_t memory_type) {
        if (!len || base > UINT64_MAX - len) return;
        const uint64_t end = base + len;
        std::vector<std::pair<uint64_t, uint64_t>> physical_ranges;
        {
            std::lock_guard<std::mutex> lk(g_mx);
            std::vector<Mapping> out;
            out.reserve(g_maps.size() + 2);
            for (const auto& mapping : g_maps) {
                const uint64_t mapping_end = mapping.base + mapping.size;
                if (mapping_end <= base || mapping.base >= end ||
                    !(mapping.query_flags & kVirtualQueryDirect)) {
                    out.push_back(mapping);
                    continue;
                }
                if (mapping.base < base) {
                    Mapping prefix = mapping;
                    prefix.size = base - mapping.base;
                    out.push_back(prefix);
                }
                Mapping changed = mapping;
                rebase_mapping(changed, mapping.base < base ? base : mapping.base);
                const uint64_t changed_end = mapping_end > end ? end : mapping_end;
                changed.size = changed_end - changed.base;
                changed.memory_type = memory_type;
                physical_ranges.emplace_back(changed.offset, changed.size);
                out.push_back(changed);
                if (mapping_end > end) {
                    Mapping suffix = mapping;
                    rebase_mapping(suffix, end);
                    suffix.size = mapping_end - end;
                    out.push_back(suffix);
                }
            }
            g_maps.swap(out);
        }
        for (const auto& range : physical_ranges)
            dmem_retype(range.first, range.second, memory_type);
    }
    bool snapshot_mapping(uint64_t addr, Mapping& out) {
        std::lock_guard<std::mutex> lk(g_mx);
        const Mapping* best = nullptr;
        for (auto& m : g_maps)
            if (addr >= m.base && addr < m.base + m.size)
                if (!best || m.base > best->base) best = &m;   // most specific (latest overlay)
        if (!best) return false;
        out = *best;
        return true;
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

    // Map guest Orbis protection bits (READ=1, WRITE=2 [implies read], EXEC=4) to host PROT_* flags.
    // prot 0 is SCE_KERNEL_PROT_NONE — a LEGITIMATE "no access" request (guard pages, reserved
    // redzones), so it must map to PROT_NONE, not a default RW. Every caller passes the guest's real
    // prot arg (never a "please default me" sentinel), so there is no case where 0 should mean RW; the
    // old `if(!hp) hp = RW` fallback silently turned an explicit guard page into a writable one, so an
    // overrun never faulted and allocator/stack tripwires never fired (#342).
    int host_prot(uint64_t p) {
        int hp = 0;
        if (p & 0x1) hp |= PROT_READ;
        if (p & 0x2) hp |= PROT_READ | PROT_WRITE;
        if (p & 0x4) hp |= PROT_EXEC;
        return hp;   // p == 0 -> PROT_NONE (no access), NOT read-write
    }
    uint64_t align_up(uint64_t v, uint64_t a) { return a ? (v + a - 1) & ~(a - 1) : v; }

    constexpr uint64_t kMaxDirectMemoryType = 0x7fffffffull;
    bool valid_dmem_allocation(uint64_t len, uint64_t alignment,
                               uint64_t memory_type, uint64_t phys_out) {
        // The direct-memory ABI is 16 KiB-granular. Do not round a malformed request up: doing so
        // consumes a different physical range than the guest requested. Alignment zero selects the
        // default page alignment; the allocator supports every explicit 16 KiB multiple.
        if (!len || (len & (kGuestPageSize - 1)) != 0) return false;
        if (alignment && (alignment & (kGuestPageSize - 1)) != 0) return false;
        // memoryType is a signed int in the ABI. Preserve every non-negative value it can carry:
        // PS5 titles use values above the older type-10 ceiling (Blasphemous 2 requests type 12),
        // and the mapping/type-query paths already preserve types 11 and 13.
        if (memory_type > kMaxDirectMemoryType) return false;
        // Low values are not plausible guest pointers. The old code accepted them, allocated from
        // the pool, then skipped the guarded write -- false success plus leaked physical capacity.
        return phys_out > 0xffff;
    }

    // The console chooses automatic mappings from its guest user-VA space.  Letting mmap(nullptr)
    // choose instead leaks a host VA (commonly 0x7f...) to the guest.  Besides being outside the PS5
    // address model, Sony libc explicitly rejects such an address as mspace backing.  Search a quiet
    // guest range atomically with MAP_FIXED_NOREPLACE; all HLE-created occupants are tracked, so a
    // collision can skip directly past them instead of probing every 64 KiB page.
    constexpr uint64_t kGuestAutoMapBase  = 0x2000000000ull;
    constexpr uint64_t kGuestAutoMapLimit = 0x40000000000ull;
    // Monotonic placement cursor for auto-mapped guest VA (#983). Restarting the linear probe in
    // map_guest_from() from kGuestAutoMapBase on EVERY auto-map is O(N) in the live-mapping count,
    // and on macOS each collided probe is a full mmap+munmap under Rosetta's VM tracking
    // (prosper_mmap_noreplace emulates MAP_FIXED_NOREPLACE that way) — a level's tens of thousands
    // of sceKernelMapDirectMemory calls degrade to O(N^2) Rosetta VM churn (minutes). Advancing a
    // cursor past each successful placement makes the common case a single successful mmap.
    // Correctness is unchanged: map_guest_from still uses NOREPLACE (never clobbers) and still scans
    // g_maps for a free slot; the cursor only picks the STARTING hint, and map_guest_auto wraps back
    // to the base to reclaim VA freed below the cursor. Atomic (lock-free) — concurrent mappers stay
    // correct via NOREPLACE; the cursor is a hint, not a lock.
    uint64_t g_auto_map_cursor = kGuestAutoMapBase;
    void* map_guest_from(uint64_t start, uint64_t len, int prot, uint64_t align) {
        const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
        if (align < page) align = page;
        if (!len || (align & (align - 1)) != 0 ||
            start > UINT64_MAX - (align - 1))
            return nullptr;
        const uint64_t step = align > 0x10000 ? align : 0x10000;
        uint64_t cand = align_up(start, align);
        if (cand >= kGuestAutoMapLimit || len > kGuestAutoMapLimit - cand)
            return nullptr;
        while (cand <= kGuestAutoMapLimit - len) {
            void* p = prosper_mmap_noreplace((void*)cand, len, prot,
                                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p != MAP_FAILED) return p;

            uint64_t next = cand + step;
            {
                std::lock_guard<std::mutex> lk(g_mx);
                const uint64_t end = cand + len;
                for (const auto& m : g_maps) {
                    const uint64_t me = m.base + m.size;
                    if (m.base < end && me > cand) {
                        const uint64_t past = align_up(me, align);
                        if (past > next) next = past;
                    }
                }
            }
            if (next <= cand) return nullptr;
            cand = next;
        }
        return nullptr;
    }

    void* map_guest_auto(uint64_t len, int prot, uint64_t align) {
        // Start probing past the last successful placement (see g_auto_map_cursor, #983) instead of
        // rescanning from kGuestAutoMapBase every call. If the tail is exhausted, wrap once to the
        // base to reclaim VA freed below the cursor.
        uint64_t hint = __atomic_load_n(&g_auto_map_cursor, __ATOMIC_RELAXED);
        if (hint < kGuestAutoMapBase) hint = kGuestAutoMapBase;
        void* p = map_guest_from(hint, len, prot, align);
        if (!p && hint > kGuestAutoMapBase)
            p = map_guest_from(kGuestAutoMapBase, len, prot, align);
        if (p) {
            // Advance the cursor monotonically to the end of this placement (only forward; VA below
            // the cursor is reclaimed by the wrap above, never leaked).
            uint64_t end = (uint64_t)p + len;
            uint64_t cur = __atomic_load_n(&g_auto_map_cursor, __ATOMIC_RELAXED);
            while (end > cur && !__atomic_compare_exchange_n(&g_auto_map_cursor, &cur, end, true,
                                                             __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
        }
        return p;
    }

    void* map_at(uint64_t hint, uint64_t len, int prot) {
        if (!hint) {
            return map_guest_auto(len, prot, 0x4000);
        }
        // Non-zero hint: claim it WITHOUT clobbering (#137). MAP_FIXED_NOREPLACE fails if anything
        // is already there; only if the occupant is entirely our own uncommitted reservation do we
        // replace it with MAP_FIXED (the guest is committing a range it reserved). A hint that
        // overlaps a committed mapping or an untracked host mapping fails instead of destroying it.
        void* p = prosper_mmap_noreplace((void*)hint, len, prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
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
            int f = prosper_memfd_create("prosper-dmem");
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
        if (fd < 0 || phys >= kDmemBase + kDmemTotal || !sz) return;
#ifdef __linux__
        fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, (off_t)phys, (off_t)sz);
#else
        // Darwin shm has no hole punching; zero the range through a scratch mapping. Loses
        // sparseness but preserves the guest-visible fresh-pages-read-zero contract.
        void* z = mmap(nullptr, sz, PROT_WRITE, MAP_SHARED, fd, (off_t)phys);
        if (z != MAP_FAILED) { memset(z, 0, sz); munmap(z, sz); }
#endif
    }
    // Reserve an anonymous span whose returned base satisfies `align`. Linux mmap only promises
    // host-page alignment, while Orbis direct-memory callers can require larger alignment.
    void* reserve_aligned(uint64_t len, uint64_t align) {
        return map_guest_auto(len, PROT_NONE, align);
    }

    // Map `len` bytes of the phys pool at `hint` (0 = anywhere). Falls back to anonymous memory
    // if the memfd is unavailable (still boots; loses aliasing). A zero-hint mapping honors the
    // caller's requested VA alignment; the old host-page-aligned mmap violated that ABI contract.
    void* map_phys_at_impl(uint64_t hint, uint64_t len, int prot, uint64_t phys, uint64_t align,
                           bool fixed) {
        int fd = dmem_fd();
        if (fd >= 0 && phys < kDmemBase + kDmemTotal) {
            if (!hint) {
                void* reserved = reserve_aligned(len, align ? align : 0x4000);
                if (reserved) {
                    void* p = mmap(reserved, len, prot, MAP_SHARED | MAP_FIXED, fd, (off_t)phys);
                    if (p != MAP_FAILED) return p;
                    munmap(reserved, len);
                }
            } else if (range_is_free_reservation(hint, len)) {
                // A prior sceKernelReserveVirtualRange owns this exact span. Replacing that
                // PROT_NONE placeholder is safe even when MAP_FIXED was not requested.
                void* p = mmap((void*)hint, len, prot, MAP_SHARED | MAP_FIXED, fd, (off_t)phys);
                if (p != MAP_FAILED) return p;
            } else if (fixed) {
                // Same no-clobber discipline as map_at (#137): NOREPLACE first, MAP_FIXED replace
                // only over our own uncommitted reservation, else refuse rather than destroy a live
                // (committed / untracked) mapping — the exact clobber class of issues #88 / #107.
                void* p = prosper_mmap_noreplace((void*)hint, len, prot, MAP_SHARED, fd, (off_t)phys);
                if (p != MAP_FAILED) return p;
                return nullptr;
            } else {
                // Without SCE_KERNEL_MAP_FIXED the input address is a search hint, not a demand
                // to replace that VA. Try the hint atomically, then relocate within the *guest* VA
                // range. Plain mmap(hint) may silently return a 0x7f... host address when the hint is
                // occupied; Sony libc rejects such a pointer as mspace backing.
                void* p = prosper_mmap_noreplace((void*)hint, len, prot, MAP_SHARED, fd,
                                                  (off_t)phys);
                if (p != MAP_FAILED &&
                    (!align || (((uint64_t)p & (align - 1)) == 0))) return p;
                if (p != MAP_FAILED) munmap(p, len);
                void* reserved = reserve_aligned(len, align ? align : 0x4000);
                if (reserved) {
                    p = mmap(reserved, len, prot, MAP_SHARED | MAP_FIXED, fd, (off_t)phys);
                    if (p != MAP_FAILED) return p;
                    munmap(reserved, len);
                }
            }
        }
        if (hint && fixed) return map_at(hint, len, prot);
        void* reserved = reserve_aligned(len, align ? align : 0x4000);
        if (!reserved) return nullptr;
        void* p = mmap(reserved, len, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        if (p == MAP_FAILED) { munmap(reserved, len); return nullptr; }
        return p;
    }

    // Single chokepoint for physical-dmem mapping. EVERY dmem alias (map_dmem, BatchMap MAP_DIRECT, Ampr
    // push-map + its same-phys mirror) funnels through here, so the texture write-watch (#1144) learns
    // the VA<->phys alias from one place — it can then arm every VA that maps a watched physical page,
    // rather than only the ones a specific high-level API remembered to report (review B3). `prot` is
    // host protection whose bits (READ=1/WRITE=2/EXEC=4) coincide with the Sony bits the watch decodes.
    void* map_phys_at(uint64_t hint, uint64_t len, int prot, uint64_t phys, uint64_t align = 0,
                      bool fixed = true) {
        void* p = map_phys_at_impl(hint, len, prot, phys, align, fixed);
        if (p)
            host::guest_write_watch_notify_direct_mapping_added(
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)), len, phys,
                static_cast<uint32_t>(prot));
        return p;
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
        void* p = prosper_mmap_noreplace((void*)hint, a1, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            // The FIXED hint region is already mapped. If it's ALREADY one of OUR reservations
            // (uncommitted), the guest is re-claiming its own recorded range — idempotent success
            // (#115: an error here made the guest allocator hand back NULL, which the level1
            // asset-load FileCacher then memcpy'd from).
            bool ours = false;
            { std::lock_guard<std::mutex> lk(g_mx);
              for (auto& m : g_maps)
                  // Idempotent only when the WHOLE requested span is contained: matching on the
                  // hint alone let a large re-reserve false-succeed backed by a smaller range at
                  // the same base (e.g. the 64 MiB metadata pool at the arena's old 0x1000000000
                  // hint after #312's huge-reserve redirect) — success with mostly-unreserved VA.
                  if (!m.committed && hint >= m.base && hint < m.base + m.size &&
                      a1 <= m.base + m.size - hint) { ours = true; break; } }
            if (ours) {
                if (a0) *(uint64_t*)a0 = hint;
                MLOG("reserve hint=0x%llx re-reserve-of-own-range -> OK\n", (unsigned long long)hint);
                return 0;
            }
            MLOG("reserve FIXED hint=0x%llx FAILED\n", (unsigned long long)hint); return 0x8002000cull; // ENOMEM
        }
        if (a0) *(uint64_t*)a0 = (uint64_t)p;
        track((uint64_t)p, a1, 0, 0, false, "reserved");
        MLOG("reserve(fixed) -> 0x%llx len=0x%llx align=0x%llx\n", (unsigned long long)p, (unsigned long long)a1, (unsigned long long)align);
        return 0;
    }
    // #312/#946: a HUGE non-fixed reservation (UE MallocBinned3's 512 GiB arena — hint
    // 0x1000000000, len 0x8000000000, align 0x200000, live-captured) must NOT be searched from
    // its low hint. A flag-less hint is only a search start, and honoring it literally races
    // prosper's own low guest-VA occupants: with the arena based at 0x1000000000 the boot
    // corrupts MallocBinned3 metadata (#312). Only the flex arena is >= 128 GiB, so steer such
    // reservations into the guest auto-map region (map_guest_auto below, placement A/B-validated
    // in the #982 investigation), leaving the low hint free for the small metadata pool the
    // guest reserves next with the same hint. The auto window bounds are untouched.
    if (hint && a1 < kHugeReserveLen) {
        // Non-fixed hint: search for a free range starting at the hint. Probe candidates with
        // MAP_FIXED_NOREPLACE (also catches host mappings the tracker doesn't know); on a miss,
        // skip past whichever TRACKED mapping covers the candidate (fast-forwards the search past
        // the 512 GiB arena in one step), else advance one alignment granule.
        uint64_t cand = align_up(hint, align);
        const uint64_t kSearchLimit = 0x40000000000ull;   // 4 TiB — far above any guest range
        while (cand < kSearchLimit) {
            void* p = prosper_mmap_noreplace((void*)cand, a1, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p != MAP_FAILED) {
                if (a0) *(uint64_t*)a0 = (uint64_t)p;
                track((uint64_t)p, a1, 0, 0, false, "reserved");
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
    void* raw = map_guest_auto(a1, PROT_NONE, align);
    if (!raw) { MLOG("reserve len=0x%llx FAILED\n", (unsigned long long)a1); return 0x8002000cull; } // ENOMEM
    uint64_t base = (uint64_t)raw;
    if (a0) *(uint64_t*)a0 = base;
    track(base, a1, 0, 0, false, "reserved");
    MLOG("reserve -> 0x%llx len=0x%llx align=0x%llx (raw 0x%llx)\n",
         (unsigned long long)base, (unsigned long long)a1, (unsigned long long)align, (unsigned long long)raw);
    return 0;
}

// sceKernelMapNamedFlexibleMemory(void** addrInOut, size_t len, int prot, int flags, const char* name)
HLE(k_map_flexible) {
    uint64_t hint = a0 ? *(uint64_t*)a0 : 0;
    const bool fixed = (a3 & 0x10) != 0;   // SCE_KERNEL_MAP_FIXED
    if (fixed && !hint) return 0x80020016ull;   // SCE_KERNEL_ERROR_EINVAL
    void* p = map_at(hint, a1, host_prot(a2));
    if (!p && hint && !fixed)
        p = map_guest_from(hint, a1, host_prot(a2), 0x4000);
    if (!p) { MLOG("mapflexible hint=0x%llx len=0x%llx FAILED\n", (unsigned long long)hint, (unsigned long long)a1); return 0x8002000cull; } // ENOMEM
    if (a0) *(uint64_t*)a0 = (uint64_t)p;
    track((uint64_t)p, a1, host_prot(a2), static_cast<uint32_t>(a2), true,
          a4 ? (const char*)a4 : "flexible", kVirtualQueryFlexible);
    MLOG("mapflexible -> 0x%llx len=0x%llx prot=0x%llx name=%s\n",
         (unsigned long long)p, (unsigned long long)a1, (unsigned long long)a2, a4 ? (const char*)a4 : "");
    return 0;
}
// sceKernelMapFlexibleMemory(addr, len, prot, flags) has NO name arg (that is the separate Named variant),
// so a4 (r8) is caller-indeterminate scratch -- k_map_flexible reads it as a name pointer and track()
// strncpy's from it, so a non-zero garbage a4 dereferences a wild address. Force name=null for the
// non-named entry point.
HLE(k_map_flexible_noname) { return k_map_flexible(a0, a1, a2, a3, 0, 0); }

// sceKernelAvailableFlexibleMemorySize(size_t* sizeOut): report the available flexible-memory budget.
// Was MISSING -> the return-0 stub left *sizeOut uninitialized, so the guest read garbage as its budget
// (Unity/allocator sizing) -> either a wild over-commit or a refusal to allocate. We don't pool-account
// flexible memory, so report the configured 512 MiB pool (shadPS4 parity); a later map that exceeds host
// memory still fails cleanly with ENOMEM.
HLE(k_avail_flexible) { if (!a0) return 0x80020016ull; *(uint64_t*)(uintptr_t)a0 = 512ull * 1024 * 1024; return 0; }

// sceKernelAllocateDirectMemory(off_t start, off_t end, size_t len, size_t align, int memType, off_t* physOut)
HLE(k_alloc_dmem) {   // (searchStart, searchEnd, len, alignment, memoryType, physAddrOut)
    if (!valid_dmem_allocation(a2, a3, a4, a5))
        return 0x80020016ull;   // SCE_KERNEL_ERROR_EINVAL
    uint64_t align = a3 ? a3 : 0x4000;
    uint64_t sz = a2;
    uint64_t off;
    // Honor the [searchStart, searchEnd) window (a0/a1) — dropping it handed the guest an offset
    // outside the window it asked for.
    if (!dmem_take(sz, align, (int)a4, off, a0, a1 ? a1 : ~0ull)) {
        MLOG("alloc_dmem len=0x%llx align=0x%llx type=0x%llx in [0x%llx,0x%llx) -> ENOMEM\n",
             (unsigned long long)a2, (unsigned long long)a3, (unsigned long long)a4,
             (unsigned long long)a0, (unsigned long long)a1);
        return 0x8002000Cull;   // SCE_KERNEL_ERROR_ENOMEM
    }
    dmem_zero(off, sz);                       // fresh allocation -> zeroed pages (console semantics)
    *(uint64_t*)a5 = off;
    MLOG("alloc_dmem range=[0x%llx,0x%llx) len=0x%llx align=0x%llx type=0x%llx -> phys=0x%llx\n",
         (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
         (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)off);
    return 0;
}

// sceKernelAllocateMainDirectMemory(size_t len, size_t align, int memType, off_t* physOut) — a
// DIFFERENT signature (4 args) from AllocateDirectMemory: physOut is arg3, not arg5. Aliasing them
// to one handler wrote the result through arg5 (uninitialized garbage, e.g. 0xa) -> crash.
HLE(k_alloc_main_dmem) {
    if (!valid_dmem_allocation(a0, a1, a2, a3))
        return 0x80020016ull;   // SCE_KERNEL_ERROR_EINVAL
    uint64_t align = a1 ? a1 : 0x4000;
    uint64_t sz = a0;
    uint64_t off;
    if (!dmem_take(sz, align, (int)a2, off)) {
        MLOG("alloc_main_dmem len=0x%llx -> ENOMEM (pool exhausted)\n", (unsigned long long)a0);
        return 0x8002000Cull;   // SCE_KERNEL_ERROR_ENOMEM
    }
    dmem_zero(off, sz);                       // fresh allocation -> zeroed pages (console semantics)
    *(uint64_t*)a3 = off;
    MLOG("alloc_main_dmem len=0x%llx -> phys=0x%llx\n", (unsigned long long)a0, (unsigned long long)off);
    return 0;
}

// sceKernelDirectMemoryQuery(off_t offset, int flags, SceKernelDirectMemoryQueryInfo* info, size_t infoSize)
//   info: 0x00 off_t start; 0x08 off_t end; 0x10 i32 memoryType. flags&1 = find next.
HLE(k_direct_memory_query) {
    if (!a2) return 0x80020016ull;   // EINVAL (null out-param)
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
    // No region at/after the offset: EACCES is the enumeration terminator. GTA V (PPSA04263)
    // walks query(offset,1)/offset=info.end and its ONLY loop exit compares against 0x8002000d;
    // any other value reads as success with a zeroed info block and the walk spins forever (#1129).
    if (!r) { MLOG("dmem_query(0x%llx) -> none (EACCES)\n", (unsigned long long)a0); return 0x8002000dull; }
    if (sz >= 0x08) *(uint64_t*)(info + 0x00) = r->start;
    if (sz >= 0x10) *(uint64_t*)(info + 0x08) = r->end;
    if (sz >= 0x14) *(int32_t*)(info + 0x10) = r->type;
    MLOG("dmem_query(0x%llx,f=0x%llx) -> [0x%llx,0x%llx) type=%d\n",
         (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)r->start, (unsigned long long)r->end, r->type);
    return 0;
}

static uint64_t map_dmem_impl(uint64_t addr_in_out, uint64_t len, uint64_t prot,
                              uint64_t flags, uint64_t phys, uint64_t align,
                              int32_t memory_type, bool set_memory_type) {
    uint64_t hint = addr_in_out ? *(uint64_t*)addr_in_out : 0;
    const bool fixed = (flags & 0x10) != 0;
    void* p = map_phys_at(hint, len, host_prot(prot), phys, align, fixed);
    if (!p) { MLOG("map_dmem hint=0x%llx len=0x%llx flags=0x%llx phys=0x%llx align=0x%llx FAILED\n",
                   (unsigned long long)hint, (unsigned long long)len,
                   (unsigned long long)flags, (unsigned long long)phys,
                   (unsigned long long)align); return 0x8002000cull; } // ENOMEM
    if (set_memory_type) dmem_retype(phys, len, memory_type);
    if (addr_in_out) *(uint64_t*)addr_in_out = (uint64_t)p;
    track((uint64_t)p, len, host_prot(prot), static_cast<uint32_t>(prot), true,
          "direct", kVirtualQueryDirect, phys, memory_type);
    // (The texture write-watch alias for this mapping is registered at the map_phys_at chokepoint above,
    // which also covers BatchMap MAP_DIRECT and Ampr — see #1144 B3.)
    MLOG("map_dmem -> 0x%llx len=0x%llx phys=0x%llx prot=0x%llx flags=0x%llx align=0x%llx\n",
         (unsigned long long)p, (unsigned long long)len, (unsigned long long)phys,
         (unsigned long long)prot, (unsigned long long)flags, (unsigned long long)align);
    return 0;
}

// sceKernelMapDirectMemory(void** addrInOut, size_t len, int prot, int flags, off_t phys, size_t align)
HLE(k_map_dmem) {
    int32_t memory_type = 0;
    dmem_type_at(a4, memory_type);
    return map_dmem_impl(a0, a1, a2, a3, a4, a5, memory_type, false);
}

// sceKernelVirtualQuery(const void* addr, int flags, SceKernelVirtualQueryInfo* info, size_t infoSize)
//   0x00 start; 0x08 end; 0x10 offset; 0x18 i32 prot; 0x1C i32 memType;
//   0x20 u8 classification; 0x21 name[32]
HLE(k_virtual_query) {
    if (!a2) return 0x80020016ull;   // EINVAL (null out-param)
    uint8_t* info = (uint8_t*)a2;
    uint64_t sz = a3 ? (a3 > 0x41 ? 0x41 : a3) : 0x41;
    memset(info, 0, sz);
    Mapping mapping{};
    const bool found = snapshot_mapping(a0, mapping);
    uint64_t start, end, offset = 0; int prot, memory_type = 0; uint32_t flags;
    const char* how;
    if (found) {                               // inside a real mapping
        // Report the mapping's REAL commit state: a reserved-but-uncommitted range has NO access
        // (prot 0). Lying prot=RW for the whole 512GB reservation made UE4's allocator skip its
        // BatchMap commit for pages VirtualQuery claimed were already writable -> first-touch crash.
        start = mapping.base; end = mapping.base + mapping.size;
        // Keep Sony's exact protection value: CPU_RW is enum 0x02 (not host R|W 0x03), and
        // GPU-only bits have no host-page-protection equivalent but remain guest-visible state.
        prot = mapping.committed ? static_cast<int>(mapping.guest_prot) : 0x0;
        offset = (mapping.query_flags & kVirtualQueryDirect) ? mapping.offset : 0;
        memory_type = (mapping.query_flags & kVirtualQueryDirect) ? mapping.memory_type : 0;
        flags = mapping.query_flags |
                (mapping.committed ? kVirtualQueryCommitted : 0); how = "tracked";
    } else {                                   // unmapped: report the whole hole to the next mapping
        uint64_t nb = next_base(a0);
        if (!nb) { MLOG("virtual_query(0x%llx) -> end-of-space (EACCES)\n", (unsigned long long)a0); return 0x8002000e; }
        start = a0 & ~(uint64_t)0x3fff; end = nb; prot = 0; flags = 0; how = "hole";
    }
    if (sz >= 0x08) *(uint64_t*)(info + 0x00) = start;
    if (sz >= 0x10) *(uint64_t*)(info + 0x08) = end;
    if (sz >= 0x18) *(uint64_t*)(info + 0x10) = offset;
    if (sz >= 0x1c) *(int32_t*)(info + 0x18) = prot;
    if (sz >= 0x20) *(int32_t*)(info + 0x1c) = memory_type;
    if (sz >= 0x21) info[0x20] = static_cast<uint8_t>(flags);
    if (sz >= 0x41 && found && mapping.name[0]) memcpy(info + 0x21, mapping.name, 32);
    MLOG("virtual_query(0x%llx,f=0x%llx) -> [0x%llx,0x%llx) %s\n",
         (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)start, (unsigned long long)end, how);
    return 0;
}

// sceKernelGetDirectMemoryType(off_t offset, int* typeOut, off_t* startOut, off_t* endOut)
// queries the allocated physical range containing `offset`. All three outputs are required by the
// ABI; a miss must fail without changing them rather than returning success with stale guest data.
HLE(k_get_direct_memory_type) {
    if (!a1 || !a2 || !a3) return 0x80020016ull;   // SCE_KERNEL_ERROR_EINVAL
    DMem result{};
    {
        std::lock_guard<std::mutex> lk(g_dmx);
        bool found = false;
        for (const auto& d : g_dmem) {
            if (a0 >= d.start && a0 < d.end) {
                result = d;
                found = true;
                break;
            }
        }
        if (!found) return 0x80020002ull;          // SCE_KERNEL_ERROR_ENOENT
    }
    *(int32_t*)(uintptr_t)a1 = result.type;
    *(uint64_t*)(uintptr_t)a2 = result.start;
    *(uint64_t*)(uintptr_t)a3 = result.end;
    return 0;
}

// sceKernelQueryMemoryProtection(addr, startOut, endOut, protOut): query the exact tracked VMA
// without VirtualQuery's larger result structure. Every output is optional; an untracked address
// fails before touching any of them. Copying the Mapping under g_mx avoids retaining a vector
// element pointer while another thread splits or replaces the tracker.
HLE(k_query_memory_protection) {
    Mapping mapping{};
    if (!snapshot_mapping(a0, mapping)) return 0x8002000dull;  // SCE_KERNEL_ERROR_EACCES
    if (a1) *(uint64_t*)(uintptr_t)a1 = mapping.base;
    if (a2) *(uint64_t*)(uintptr_t)a2 = mapping.base + mapping.size;
    if (a3) *(uint32_t*)(uintptr_t)a3 = mapping.guest_prot;
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
                           (long)prosper_gettid(), (unsigned long long)a0, *(uint32_t*)a0,
                           (unsigned long long)a1, pts ? (long long)(ts.tv_sec*1000000 + ts.tv_nsec/1000) : -1,
                           (unsigned long long)goff);
    WaitRegistration registration = futex_wait_enter(a0); // lets labels and GC wake this waiter
    long r = prosper_futex_wait((uint32_t*)a0, (uint32_t)a1, pts);
    int e = errno;
    futex_wait_exit(registration);
    if (synclog()) fprintf(stderr, "[sync] T%ld WAIT.exit   addr=0x%llx r=%ld errno=%d\n",
                           (long)prosper_gettid(), (unsigned long long)a0, r, r < 0 ? e : 0);
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
                    (long)prosper_gettid(), (unsigned long long)a0, (unsigned long long)a1,
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
                (long)prosper_gettid(), (unsigned long long)a0, *(uint32_t*)a0, n, (unsigned long long)wgoff);
    }
    return 0;
}

HLE(k_munmap) {
    constexpr uint64_t mask = kGuestPageSize - 1;
    if (!a0 || !a1 || (a0 & mask) || (a1 & mask) || a1 > UINT64_MAX - a0)
        return 0x80020016ull;
    // Retire any texture write-watch over this VA BEFORE the pages disappear (#1144): invalidate the
    // watch (its next query reads Dirty) and drop the alias so a later create can't mprotect a stale
    // or reused mapping. Harmless for non-dmem ranges (no matching alias).
    host::guest_write_watch_notify_direct_mapping_removed(a0, a1);
    if (munmap((void*)a0, a1) != 0) return 0x80020016ull;
    untrack(a0, a1);
    return 0;
}
static uint64_t sce_mprotect_error(int error) {
    switch (error) {
        case EACCES: return 0x8002000dull;
        case ENOMEM: return 0x8002000cull;
        case EINVAL:
        default:     return 0x80020016ull;
    }
}
HLE(k_mprotect) {
    if (!a0) return 0x80020016ull;
    const int prot = host_prot(a2);
    if (mprotect((void*)a0, a1, prot) != 0) return sce_mprotect_error(errno);
    retrack_prot(a0, a1, prot, static_cast<uint32_t>(a2), "mprotect");
    return 0;
}

// sceKernelMapDirectMemory2 inserts a memory-type argument before protection. A successful map
// applies that explicit type to both this VMA and the corresponding physical-allocation range.
HLE7(k_map_dmem2) {
    return map_dmem_impl(a0, a1, a3, a4, a5, a6, static_cast<int32_t>(a2), true);
}
// sceKernelMtypeprotect(addr, size, mtype, prot): apply the CPU protection (arg a3) then set the direct-
// memory type (a2). Only publish either change after the host protection operation succeeds.
HLE(k_mtypeprotect) {
    if (!a0) return 0x80020016ull;
    uint64_t base = 0, len = 0;
    if (!normalize_guest_page_range(a0, a1, base, len)) return 0x80020016ull;
    if (!len) return 0;
    const int prot = host_prot(a3);
    if (mprotect((void*)base, len, prot) != 0) return sce_mprotect_error(errno);
    retrack_prot(base, len, prot, static_cast<uint32_t>(a3), "mtypeprotect");
    retrack_type(base, len, static_cast<int32_t>(a2));
    return 0;
}
HLE(k_dmem_size){ return kDmemTotal; }   // sparse-backed; allocation failures enforce this bound
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
        for (uint64_t p : { a0, a1, a2 }) if (p > 0xffff && !(p & 7)) {
            // Fault-safe read (#1154): a passed-in guest pointer that is aligned and > 0xffff can still
            // be UNMAPPED — GTA V's baQO9ez2gL4 passes a2=0x302200, which clears both guards but was
            // never mapped. A raw *(uint64_t*)p deref SIGSEGVs in host HLE code and kills the very run
            // this diagnostic is meant to observe. process_vm_readv returns EFAULT instead of faulting
            // (all-or-nothing per iovec, so a partially-mapped pair reports UNMAPPED — acceptable here).
            uint64_t v[2];
            struct iovec l { v, sizeof v }, r { (void*)(uintptr_t)p, sizeof v };
            if (process_vm_readv(getpid(), &l, 1, &r, 1, 0) == (ssize_t)sizeof v)
                fprintf(stderr, "[amprlog]   [0x%llx] = 0x%016llx 0x%016llx\n", (unsigned long long)p,
                        (unsigned long long)v[0], (unsigned long long)v[1]);
            else
                fprintf(stderr, "[amprlog]   [0x%llx] = UNMAPPED\n", (unsigned long long)p);
        }
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

// --- APR command-buffer WriteAddress (NID j0+3uJMxYJY) — the APR completion-notification primitive
// (issue #1149). The NID is taken verbatim from GTA V (PPSA04263)'s import table; its exact firmware
// name is not in the PS5 3.20 dump (and does NOT hash from Kyty's guessed
// "sceAmprCommandBufferWriteAddress"), so the name is unverified — the behavior below is re-derived
// from the guest's use and confirmed against a live capture, NOT copied from a secondary source.
// Signature: (cb, u64* address, u64 value, u32 flags).
// It appends a "write value -> *address" command to the APR command buffer; the APR engine performs
// that write when the buffer executes, in command-buffer order (i.e. AFTER the reads queued before
// it). The guest uses it as a load-completion signal: it pre-sets *address to a "pending" sentinel,
// appends WriteAddress(address, doneValue), then spin-polls *address for doneValue on a waiter thread.
//
// prosper serves the APR ReadFile commands SYNCHRONOUSLY at append time (see f_apr_read_submit /
// mQ16-QdKv7k in hle_file.cpp): every read queued before this WriteAddress is already complete when
// this call is made. The completion condition the write represents is therefore already satisfied, so
// we perform the write here, immediately — consistent with prosper's eager-read model and correct in
// command order (the write only ever follows its reads). *value* is written verbatim (the guest polls
// for the exact token it queued — GTA increments it per submit: 0, then 4, 5, 6, ...).
//
// Residual hazard (documented, not hit by the observed protocol): because the write fires at
// append-time rather than at an explicit submit, a title that re-armed *address to its pending
// sentinel AFTER appending WriteAddress but BEFORE submitting would lose this completion. GTA pre-arms
// the sentinel FIRST (verified: *addr is already 0xffffffffffffffff when this call is made), so the
// ordering is safe here; revisit if a future title's WriteAddress completion is dropped.
//
// The write is fault-safe (never faults the HLE on a bad guest VA — see issue #1154) AND commits
// lazy-reserved pages the way the SIGSEGV fault handler does: process_vm_writev cannot fault through
// prosper's lazy-commit handler, so a target in a guest-RESERVED-but-untouched range would EFAULT even
// though a real guest write would succeed — so we commit-and-retry, exactly like apr_write_guest_dst
// in hle_file.cpp. A genuinely undeliverable completion write is logged LOUDLY (unconditionally): a
// silently dropped completion is precisely the invisible stall this fix exists to remove.
//
// Live GTA V evidence: post-intro streaming appends one such command per small metadata buffer, e.g.
//   j0+3uJMxYJY(cb=0x20001f13f0, addr=cb+0x40, value=0)   with *addr pre-set to 0xffffffffffffffff.
// prosper previously stubbed the NID (returned 0, wrote nothing), so *address stayed pending forever
// and the RAGE main thread busy-waited (eboot+0x2b5f0e0 sched_yield loop) on the load future's ready
// flag gated behind this write — the title-screen-frontier stall. Performing the write lets the load
// complete and the game advances past the intro into further streaming. CONFIDENCE: HIGH (ABI +
// address/value verified live; the write clears the stall and the boot progresses).
namespace { bool wa_log() { static int v = getenv("PROSPER_FILELOG") ? 1 : 0; return v; } }
extern "C" int prosper_reserved_range_state(uint64_t addr);   // defined below (lazy-commit tracking)
HLE(k_ampr_write_address) {   // j0+3uJMxYJY (cb, address, value, flags)
    (void)a0;
    if (!a1) return 0;
    uint64_t value = a2;
    struct iovec local { &value, sizeof(value) };
    struct iovec remote { (void*)(uintptr_t)a1, sizeof(value) };
    bool ok = process_vm_writev(getpid(), &local, 1, &remote, 1, 0) == (ssize_t)sizeof(value);
    if (!ok) {
        // Commit any covering 64 KiB pages that are lazy-reserved-but-untouched, then retry once.
        bool committed = false;
        for (uint64_t p = a1 & ~0xffffull; p < a1 + sizeof(value); p += 0x10000) {
            unsigned char vec;
            if (prosper_mincore((void*)(uintptr_t)p, 1, &vec) != 0 &&
                prosper_reserved_range_state(p) == 1 &&
                mmap((void*)(uintptr_t)p, 0x10000, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == (void*)(uintptr_t)p)
                committed = true;
        }
        if (committed)
            ok = process_vm_writev(getpid(), &local, 1, &remote, 1, 0) == (ssize_t)sizeof(value);
    }
    if (!ok)
        fprintf(stderr, "[apr] write-address *0x%llx = 0x%llx UNMAPPED — completion write dropped\n",
                (unsigned long long)a1, (unsigned long long)value);
    else if (wa_log())
        fprintf(stderr, "[apr] write-address *0x%llx = 0x%llx (a3=0x%llx) OK\n",
                (unsigned long long)a1, (unsigned long long)value, (unsigned long long)a3);
    return 0;
}
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
        if (p) track((uint64_t)p, a2, PROT_READ | PROT_WRITE, 0x2, true, "ampr-map",
                     have_phys ? kVirtualQueryDirect : 0, have_phys ? phys : 0,
                     have_phys ? 0x0c : 0);
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
            bool mirror_live = (prosper_mincore((void*)(uintptr_t)mirror, 1, &vec) == 0);
            void* q = nullptr;
            if (mirror_live) {
                MLOG("ampr push-map va=0x%llx mirror=0x%llx SKIPPED (target is live guest memory — "
                     "map-flavor mirror would clobber MallocBinned heap, issue #107)\n",
                     (unsigned long long)a1, (unsigned long long)mirror);
            } else {
                q = map_phys_at(mirror, a2, PROT_READ | PROT_WRITE, phys);
                if (q) track((uint64_t)q, a2, PROT_READ | PROT_WRITE, 0x2, true,
                             "ampr-mirror", kVirtualQueryDirect, phys, 0x0c);
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
        int32_t  type  = e[0x19];
        int32_t  op    = *(const int32_t*)(e + 0x1c);
        // Per-entry phys trace (#312 alias hunt): log map/unmap WITH the phys offset so an offline
        // pass can prove/disprove two live VAs aliasing one phys range through the shared memfd.
        MLOG("bm op=%d va=0x%llx phys=0x%llx len=0x%llx prot=0x%x\n",
             op, (unsigned long long)start, (unsigned long long)phys,
             (unsigned long long)len, prot);
        if (!len) {
            ret = 0x80020016ull;
            break;
        }
        bool ok = true;
        switch (op) {
            case 0: {                               // MAP_DIRECT: phys-backed (aliasing preserved)
                void* p = map_phys_at(start, len, host_prot(prot), phys);
                ok = (p != nullptr);
                int32_t memory_type = 0;
                dmem_type_at(phys, memory_type);
                if (ok) track((uint64_t)p, len, host_prot(prot), prot, true,
                              "batch-direct", kVirtualQueryDirect, phys, memory_type);
                break;
            }
            case 3: {                               // MAP_FLEXIBLE: anonymous
                void* p = map_at(start, len, host_prot(prot));
                ok = (p != nullptr);
                if (ok) track((uint64_t)p, len, host_prot(prot), prot, true,
                              "batch-flex", kVirtualQueryFlexible);
                break;
            }
            case 1: if (start) {                                                             // UNMAP
                        host::guest_write_watch_notify_direct_mapping_removed(start, len);    // #1144 B3/B4
                        munmap((void*)(uintptr_t)start, len); untrack(start, len);
                    } break;
            case 2: case 4: {                                                            // PROTECT / TYPE_PROTECT
                if (start) {
                    uint64_t protect_start = start, protect_len = len;
                    if (op == 4 && !normalize_guest_page_range(
                                       start, len, protect_start, protect_len)) {
                        ok = false;
                        ret = 0x80020016ull;
                        break;
                    }
                    if (!protect_len) break;
                    if (mprotect((void*)(uintptr_t)protect_start, protect_len,
                                 host_prot(prot)) != 0) {
                        ok = false;
                        ret = sce_mprotect_error(errno);
                    } else {
                        retrack_prot(protect_start, protect_len, host_prot(prot), prot,
                                     "batch-prot");
                        if (op == 4) retrack_type(protect_start, protect_len, type);
                    }
                }
                break;
            }
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
    R("sceKernelMapFlexibleMemory", k_map_flexible_noname);   // no name arg: a4 is garbage r8, don't deref it
    Hle::register_fn("4h6F1LLbTiw", (HleFn)k_map_flexible_noname,
                     "sceKernelMapFlexibleMemoryInternal");
    R("sceKernelAvailableFlexibleMemorySize", k_avail_flexible);   // was MISSING -> uninitialized budget
    R("sceKernelAllocateDirectMemory", k_alloc_dmem);
    R("sceKernelAllocateMainDirectMemory", k_alloc_main_dmem);  // 4-arg signature (physOut at arg3)
    R("sceKernelMapDirectMemory", k_map_dmem);
    R("sceKernelMapDirectMemory2", k_map_dmem2);
    R("sceKernelMapNamedDirectMemory", k_map_dmem);
    R("sceKernelMunmap", k_munmap);
    // sceKernelReleaseFlexibleMemory(addr, len): same shape as munmap (unmap + untrack the flexible range).
    // Was MISSING -> the stub returned success without releasing, so the range stayed mapped and its
    // tracking entry lingered -> VirtualQuery / the fault-handler probe report a freed VA as live, and host
    // VA leaks on churn. Reuse k_munmap (identical (addr, len) contract).
    R("sceKernelReleaseFlexibleMemory", k_munmap);
    R("sceKernelMprotect", k_mprotect);
    R("sceKernelMtypeprotect", k_mtypeprotect);   // was MISSING -> silently dropped the protection change
    R("sceKernelReleaseDirectMemory", k_release_dmem);
    R("sceKernelCheckedReleaseDirectMemory", k_release_dmem);
    R("sceKernelBatchMap", k_batch_map);
    R("sceKernelBatchMap2", k_batch_map);   // (entries, num, out, flags) — extra flags arg ignored
    R("sceKernelVirtualQuery", k_virtual_query);
    R("sceKernelQueryMemoryProtection", k_query_memory_protection);
    R("sceKernelDirectMemoryQuery", k_direct_memory_query);
    R("sceKernelGetDirectMemoryType", k_get_direct_memory_type);
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
    // APR completion-notification write, performed eagerly (#1149). NID from GTA V's import table; the
    // exact firmware name is unknown (not in the 3.20 dump, and it does not hash from Kyty's guessed
    // "sceAmprCommandBufferWriteAddress"), so the label carries a "?" per the unverified-name convention.
    Hle::register_fn("j0+3uJMxYJY", (HleFn)k_ampr_write_address, "sceAmprCommandBufferWriteAddress?");
}

} // namespace prosper

#else
// ============================================================================================
// Windows backing (Win32 VirtualAlloc/VirtualProtect). Sibling of the POSIX impl above.
//
// PS5 memory model (identical to the Linux path): reserve a virtual range, allocate "direct"
// (physical) memory as an opaque pool offset, then map it — or flexible memory — into VA. The
// direct-memory POOL BOOKKEEPING (dmem_take / g_dmem) and the VA TRACKER (g_maps) are pure
// logic COPIED VERBATIM from the Linux path so VirtualQuery/DirectMemoryQuery stay truthful and
// the allocator's first-fit/window semantics match byte-for-byte.
//
// What DIFFERS: the OS primitives. Linux uses mmap over a shared memfd so a phys offset can be
// mapped at two VAs that ALIAS the same bytes (unified-physical-memory contract). One sparse
// temporary file backs the entire pool, matching the POSIX memfd implementation without requiring
// guest-time access violations to commit SEC_RESERVE pages. Windows
// ordinary file views require 64 KiB-aligned offsets and bases. Modern placeholder replacement
// removes that restriction: MapViewOfFile3 accepts page-aligned offsets/bases when replacing an
// exact placeholder, so every 16 KiB guest mapping can preserve physical aliasing. The legacy
// MapViewOfFileEx path remains for systems without the modern APIs, but never fakes a fixed direct
// mapping with private memory.
// ============================================================================================
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00   // Win10: WaitOnAddress/WakeByAddress* need >= 0x0602 (Win8)
#endif
#include <windows.h>
#include <winioctl.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define HLE7(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6)

namespace {
    bool memlog() { static int v = getenv("PROSPER_MEMLOG") ? 1 : 0; return v; }
    #define MLOG(...) do { if (memlog()) fprintf(stderr, "[memhle] " __VA_ARGS__); } while (0)

    // --- VA tracker + direct-memory pool: PURE logic, copied verbatim from the Linux path -----
    constexpr uint32_t kVirtualQueryFlexible = 0x01;
    constexpr uint32_t kVirtualQueryDirect   = 0x02;
    constexpr uint32_t kVirtualQueryCommitted = 0x10;
    constexpr uint64_t kGuestPageSize = 0x4000;

    bool align_up_multiple(uint64_t value, uint64_t alignment, uint64_t& out) {
        if (!alignment) { out = value; return true; }
        const uint64_t remainder = value % alignment;
        const uint64_t delta = remainder ? alignment - remainder : 0;
        if (delta > UINT64_MAX - value) return false;
        out = value + delta;
        return true;
    }

    bool normalize_guest_page_range(uint64_t addr, uint64_t len,
                                    uint64_t& base_out, uint64_t& len_out) {
        constexpr uint64_t mask = kGuestPageSize - 1;
        if (len > UINT64_MAX - addr) return false;
        const uint64_t raw_end = addr + len;
        if (raw_end > UINT64_MAX - mask) return false;
        const uint64_t base = addr & ~mask;
        const uint64_t end = (raw_end + mask) & ~mask;
        if (end < base) return false;
        base_out = base;
        len_out = end - base;
        return true;
    }

    struct Mapping {
        uint64_t base, size, offset;
        int prot;                  // normalized host CPU mask
        uint32_t guest_prot;       // exact SCE protection enum/bits returned to the guest
        int32_t memory_type;        // direct-memory type; meaningful only with is_direct
        uint32_t query_flags;       // is_flexible / is_direct bits (commit state is separate)
        bool committed;
        char name[32];
    };
    std::mutex g_mx;
    std::vector<Mapping> g_maps;
    bool sparse_dmem_view_overlaps(uint64_t begin, uint64_t end);
    constexpr uint64_t kDmemBase  = 0x10000000;
    // Direct-memory budget the pool holds AND sceKernelGetDirectMemorySize advertises. Real PS5
    // reports the GAME budget (aperture minus OS reservation), not the raw 16 GiB aperture; UE
    // sizes allocator pools from this value (#1213 investigation). PROSPER_DMEM_BUDGET_MB
    // overrides for A/B experiments; the default stays the historical 16 GiB pending
    // cross-title verification of a hardware-truthful default.
    const uint64_t kDmemTotal = [] {
        if (const char* v = getenv("PROSPER_DMEM_BUDGET_MB")) {
            const uint64_t mib = strtoull(v, nullptr, 10);
            if (mib >= 1024) return mib * 1024ull * 1024ull;
        }
        return 16ull * 1024 * 1024 * 1024;
    }();
    struct DMem { uint64_t start, end; int type; };
    std::mutex g_dmx;
    std::vector<DMem> g_dmem;

    void rebase_mapping(Mapping& mapping, uint64_t new_base) {
        if ((mapping.query_flags & kVirtualQueryDirect) && new_base > mapping.base)
            mapping.offset += new_base - mapping.base;
        mapping.base = new_base;
    }

    // First-fit claim of `sz` bytes at `align` within [lo,hi) — identical to the Linux allocator.
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
            uint64_t beg = gap_beg < lo ? lo : gap_beg;
            uint64_t end = gap_end > hi ? hi : gap_end;
            uint64_t off = 0;
            if (align_up_multiple(beg, align, off) && off <= end && sz <= end - off) {
                insert_at = i; off_out = off; goto found;
            }
            if (i < g_dmem.size()) cursor = g_dmem[i].end;
        }
        return false;
    found:
        g_dmem.insert(g_dmem.begin() + insert_at, { off_out, off_out + sz, type });
        return true;
    }
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
            uint64_t beg = gap_beg < lo ? lo : gap_beg;
            uint64_t end = gap_end > hi ? hi : gap_end;
            uint64_t aligned = 0;
            if (align_up_multiple(beg, align, aligned) && end > aligned &&
                end - aligned > size_out) {
                off_out = aligned; size_out = end - aligned;
            }
            if (i < g_dmem.size()) cursor = g_dmem[i].end;
        }
    }
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

    bool dmem_type_at(uint64_t offset, int32_t& type_out) {
        std::lock_guard<std::mutex> lk(g_dmx);
        for (const auto& d : g_dmem) {
            if (offset >= d.start && offset < d.end) {
                type_out = d.type;
                return true;
            }
        }
        return false;
    }

    void dmem_retype(uint64_t start, uint64_t len, int32_t type) {
        if (!len || start > UINT64_MAX - len) return;
        const uint64_t end = start + len;
        std::lock_guard<std::mutex> lk(g_dmx);
        std::vector<DMem> out;
        out.reserve(g_dmem.size() + 2);
        for (const auto& d : g_dmem) {
            if (d.end <= start || d.start >= end) {
                out.push_back(d);
                continue;
            }
            if (d.start < start) out.push_back({d.start, start, d.type});
            const uint64_t changed_start = d.start < start ? start : d.start;
            const uint64_t changed_end = d.end > end ? end : d.end;
            out.push_back({changed_start, changed_end, type});
            if (d.end > end) out.push_back({end, d.end, d.type});
        }
        g_dmem.swap(out);
    }
    // Keep tracking non-overlapping when a commit replaces part of an existing reservation.
    void track(uint64_t base, uint64_t size, int prot, uint32_t guest_prot,
               bool committed, const char* nm, uint32_t query_flags = 0,
               uint64_t offset = 0, int32_t memory_type = 0,
               bool host_readable = true) {
        if (!size || base > UINT64_MAX - size) return;
        const uint64_t end = base + size;
        {
            std::lock_guard<std::mutex> lk(g_mx);
            std::vector<Mapping> out;
            out.reserve(g_maps.size() + 2);
            for (const auto& old : g_maps) {
                const uint64_t old_end = old.base + old.size;
                if (old_end <= base || old.base >= end) { out.push_back(old); continue; }
                if (old.base < base) {
                    Mapping lo = old; lo.size = base - old.base; out.push_back(lo);
                }
                if (old_end > end) {
                    Mapping hi = old;
                    rebase_mapping(hi, end);
                    hi.size = old_end - end;
                    out.push_back(hi);
                }
            }
            Mapping m{};
            m.base = base;
            m.size = size;
            m.offset = offset;
            m.prot = prot;
            m.guest_prot = guest_prot;
            m.memory_type = memory_type;
            m.query_flags = query_flags;
            m.committed = committed;
            if (nm) { strncpy(m.name, nm, sizeof m.name - 1); }
            out.push_back(m);
            g_maps.swap(out);
        }
        host::notify_guest_mapping_added(base, size,
                                         committed && host_readable && (prot & 0x1));
    }
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
            if (me > end) {
                Mapping hi = m;
                rebase_mapping(hi, end);
                hi.size = me - end;
                out.push_back(hi);
            }
        }
        g_maps.swap(out);
        host::notify_guest_mapping_removed(base, len);
    }
    void retrack_prot(uint64_t base, uint64_t len, int prot, uint32_t guest_prot,
                      const char* nm) {
        if (!len || base > UINT64_MAX - len) return;
        const uint64_t end = base + len;
        std::vector<Mapping> retagged;
        {
            std::lock_guard<std::mutex> lk(g_mx);
            std::vector<Mapping> out;
            out.reserve(g_maps.size() + 2);
            for (const auto& m : g_maps) {
                const uint64_t me = m.base + m.size;
                if (me <= base || m.base >= end) { out.push_back(m); continue; }
                if (m.base < base) {
                    Mapping lo = m; lo.size = base - m.base; out.push_back(lo);
                }
                Mapping changed = m;
                rebase_mapping(changed, m.base < base ? base : m.base);
                const uint64_t changed_end = me > end ? end : me;
                changed.size = changed_end - changed.base;
                changed.prot = prot;
                changed.guest_prot = guest_prot;
                memset(changed.name, 0, sizeof changed.name);
                if (nm) strncpy(changed.name, nm, sizeof changed.name - 1);
                out.push_back(changed);
                retagged.push_back(changed);
                if (me > end) {
                    Mapping hi = m;
                    rebase_mapping(hi, end);
                    hi.size = me - end;
                    out.push_back(hi);
                }
            }
            if (retagged.empty()) {
                Mapping changed{};
                changed.base = base;
                changed.size = len;
                changed.prot = prot;
                changed.guest_prot = guest_prot;
                changed.committed = true;
                if (nm) strncpy(changed.name, nm, sizeof changed.name - 1);
                out.push_back(changed);
                retagged.push_back(changed);
            }
            g_maps.swap(out);
        }
        host::notify_guest_mapping_removed(base, len);
        for (const auto& m : retagged) {
            // Sparse protection overlays remain submit-local; fully committed mappings retain the
            // generation-guarded cross-submit readability optimization from #737.
            const bool host_readable = !sparse_dmem_view_overlaps(m.base, m.base + m.size);
            host::notify_guest_mapping_added(
                m.base, m.size, m.committed && host_readable && (prot & 0x1));
        }
    }

    void retrack_type(uint64_t base, uint64_t len, int32_t memory_type) {
        if (!len || base > UINT64_MAX - len) return;
        const uint64_t end = base + len;
        std::vector<std::pair<uint64_t, uint64_t>> physical_ranges;
        {
            std::lock_guard<std::mutex> lk(g_mx);
            std::vector<Mapping> out;
            out.reserve(g_maps.size() + 2);
            for (const auto& mapping : g_maps) {
                const uint64_t mapping_end = mapping.base + mapping.size;
                if (mapping_end <= base || mapping.base >= end ||
                    !(mapping.query_flags & kVirtualQueryDirect)) {
                    out.push_back(mapping);
                    continue;
                }
                if (mapping.base < base) {
                    Mapping prefix = mapping;
                    prefix.size = base - mapping.base;
                    out.push_back(prefix);
                }
                Mapping changed = mapping;
                rebase_mapping(changed, mapping.base < base ? base : mapping.base);
                const uint64_t changed_end = mapping_end > end ? end : mapping_end;
                changed.size = changed_end - changed.base;
                changed.memory_type = memory_type;
                physical_ranges.emplace_back(changed.offset, changed.size);
                out.push_back(changed);
                if (mapping_end > end) {
                    Mapping suffix = mapping;
                    rebase_mapping(suffix, end);
                    suffix.size = mapping_end - end;
                    out.push_back(suffix);
                }
            }
            g_maps.swap(out);
        }
        for (const auto& range : physical_ranges)
            dmem_retype(range.first, range.second, memory_type);
    }
    bool snapshot_mapping(uint64_t addr, Mapping& out) {
        std::lock_guard<std::mutex> lk(g_mx);
        const Mapping* best = nullptr;
        for (auto& m : g_maps)
            if (addr >= m.base && addr < m.base + m.size)
                if (!best || m.base > best->base) best = &m;
        if (!best) return false;
        out = *best;
        return true;
    }
    uint64_t next_base(uint64_t addr) {
        std::lock_guard<std::mutex> lk(g_mx);
        uint64_t n = 0;
        for (auto& m : g_maps)
            if (m.base > addr && (n == 0 || m.base < n)) n = m.base;
        return n;
    }
    uint64_t align_up(uint64_t v, uint64_t a) { return a ? (v + a - 1) & ~(a - 1) : v; }

    constexpr uint64_t kMaxDirectMemoryType = 0x7fffffffull;
    bool valid_dmem_allocation(uint64_t len, uint64_t alignment,
                               uint64_t memory_type, uint64_t phys_out) {
        if (!len || (len & (kGuestPageSize - 1)) != 0) return false;
        if (alignment && (alignment & (kGuestPageSize - 1)) != 0) return false;
        // memoryType is a signed int in the ABI. PS5 titles legitimately use values above 10;
        // reject only values that cannot represent a non-negative int.
        if (memory_type > kMaxDirectMemoryType) return false;
        return phys_out > 0xffff;
    }

    // Guest Orbis protection bits (READ=1, WRITE=2 [implies read], EXEC=4). WRITE implies READ,
    // and prot 0 is a legitimate no-access guard page (matches the Linux host_prot #342 fix).
    enum { HP_R = 1, HP_W = 2, HP_X = 4 };
    int host_prot(uint64_t p) {
        int hp = 0;
        if (p & 0x1) hp |= HP_R;
        if (p & 0x2) hp |= HP_R | HP_W;
        if (p & 0x4) hp |= HP_X;
        return hp;   // p == 0 -> no access
    }
    // --- Win32 OS primitives (the platform-specific half) -------------------------------------
    DWORD win_page_prot(int hp) {
        bool w = hp & HP_W, r = hp & HP_R, x = hp & HP_X;
        if (x) return w ? PAGE_EXECUTE_READWRITE : (r ? PAGE_EXECUTE_READ : PAGE_EXECUTE);
        if (w) return PAGE_READWRITE;
        if (r) return PAGE_READONLY;
        return PAGE_NOACCESS;
    }
    constexpr uint64_t kWinAllocationGranularity = 0x10000;
    // PS5 libc accepts caller-supplied mspace storage in two virtual-address apertures. Keep
    // automatically placed guest mappings in the low aperture and above the fixed module/direct
    // ranges. Letting Windows choose from its full user VA space can land a valid mapping in the
    // rejected 1-8 TiB gap (Astro observed 0x2d980000000), after which sceLibcMspaceCreate returns
    // null even though the pages are accessible.
    constexpr uint64_t kGuestAutoVaMin = 0x2000000000ull;   // 128 GiB
    constexpr uint64_t kGuestAutoVaMax = 0xfbffffffffull;  // inclusive; one byte below a 64 KiB boundary

    // A paging-file SEC_RESERVE section avoids a 16 GiB commit charge, but every first guest touch
    // raises a Windows access violation. Windows builds its exception-dispatch frame below RSP
    // before the VEH runs, overwriting the 128-byte SysV red zone used by unmodified PS5 code.
    // Back the aperture with a delete-on-close sparse file instead: mapped pages are ordinary
    // demand-paged file data (MEM_COMMIT from the guest's perspective), untouched ranges consume no
    // disk clusters, aliases remain coherent, and guest execution never needs a lazy-commit fault.
    struct DmemSectionState {
        HANDLE file = INVALID_HANDLE_VALUE;
        HANDLE section = nullptr;
    };

    const DmemSectionState& dmem_section_state() {
        static const DmemSectionState state = [] {
            wchar_t temp_dir[MAX_PATH + 1]{};
            wchar_t temp_path[MAX_PATH + 1]{};
            const DWORD temp_len = GetTempPathW(MAX_PATH, temp_dir);
            if (temp_len && temp_len <= MAX_PATH &&
                GetTempFileNameW(temp_dir, L"ps5", 0, temp_path)) {
                HANDLE file = CreateFileW(
                    temp_path, GENERIC_READ | GENERIC_WRITE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                    OPEN_EXISTING,
                    FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE |
                        FILE_FLAG_RANDOM_ACCESS,
                    nullptr);
                if (file != INVALID_HANDLE_VALUE) {
                    DWORD ignored = 0;
                    LARGE_INTEGER size{};
                    size.QuadPart = static_cast<LONGLONG>(kDmemTotal);
                    if (DeviceIoControl(file, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0,
                                        &ignored, nullptr) &&
                        SetFilePointerEx(file, size, nullptr, FILE_BEGIN) && SetEndOfFile(file)) {
                        HANDLE section = CreateFileMappingW(
                            file, nullptr, PAGE_READWRITE,
                            static_cast<DWORD>(kDmemTotal >> 32),
                            static_cast<DWORD>(kDmemTotal & 0xffffffffu), nullptr);
                        if (section) return DmemSectionState{file, section};
                    }
                    CloseHandle(file); // FILE_FLAG_DELETE_ON_CLOSE removes the temporary file.
                } else {
                    DeleteFileW(temp_path);
                }
            }

            std::fprintf(stderr,
                         "[memhle] Windows direct memory requires a sparse temporary file\n");
            return DmemSectionState{};
        }();
        return state;
    }

    HANDLE dmem_section() { return dmem_section_state().section; }

    bool ensure_section_pages_committed(uint64_t base, uint64_t len, int hp) {
        if (!len || base > UINT64_MAX - len) return false;
        const uint64_t end = base + len;
        uint64_t cursor = base;
        while (cursor < end) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (!VirtualQuery(reinterpret_cast<void*>(static_cast<uintptr_t>(cursor)),
                              &mbi, sizeof(mbi))) return false;
            const uint64_t region_end = std::min<uint64_t>(
                end, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mbi.BaseAddress)) +
                         mbi.RegionSize);
            if (region_end <= cursor || mbi.State == MEM_FREE) return false;
            if (mbi.State == MEM_RESERVE && !VirtualAlloc(
                    reinterpret_cast<void*>(static_cast<uintptr_t>(cursor)),
                    static_cast<SIZE_T>(region_end - cursor), MEM_COMMIT,
                    win_page_prot(hp))) return false;
            cursor = region_end;
        }
        return true;
    }

    struct DmemView {
        uint64_t guest_base;
        uint64_t guest_size;
        uint64_t phys;
        void* view_base;
        uint64_t view_size;
        bool sparse;
        bool placeholder;
        uint64_t page_cache_generation = 0;
        std::vector<uint64_t> committed_pages;
    };
    struct PlaceholderSpan { uint64_t base, size; };
    struct PrivatePlaceholderView { uint64_t base, size; };
    enum class PlaceholderOwner { Free, Guest };
    struct AcquiredPlaceholder {
        void* address = nullptr;
        PlaceholderOwner owner = PlaceholderOwner::Free;
    };
    std::mutex g_dview_mx;
    std::vector<DmemView> g_dviews;
    std::vector<PlaceholderSpan> g_free_placeholders;
    // Host placeholders owned by an uncommitted guest reservation. Keeping these separate from
    // guest-free placeholders prevents a zero-hint map from consuming another allocation while
    // still allowing a fixed map to replace its own reservation.
    std::vector<PlaceholderSpan> g_guest_placeholders;
    // Private allocations created with MEM_REPLACE_PLACEHOLDER must be returned to placeholders
    // on unmap; VirtualFree(..., MEM_DECOMMIT) would leave a non-replaceable private reservation.
    std::vector<PrivatePlaceholderView> g_private_placeholder_views;

    using VirtualAlloc2Fn = PVOID (WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG,
                                            MEM_EXTENDED_PARAMETER*, ULONG);
    using MapViewOfFile3Fn = PVOID (WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T,
                                             ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using UnmapViewOfFile2Fn = BOOL (WINAPI*)(HANDLE, PVOID, ULONG);

    struct PlaceholderApis {
        VirtualAlloc2Fn virtual_alloc2 = nullptr;
        MapViewOfFile3Fn map_view_of_file3 = nullptr;
        UnmapViewOfFile2Fn unmap_view_of_file2 = nullptr;
    };

    const PlaceholderApis& placeholder_apis() {
        static const PlaceholderApis apis = [] {
            HMODULE module = GetModuleHandleW(L"kernelbase.dll");
            if (!module) module = GetModuleHandleW(L"kernel32.dll");
            PlaceholderApis out;
            if (module) {
                out.virtual_alloc2 = reinterpret_cast<VirtualAlloc2Fn>(
                    GetProcAddress(module, "VirtualAlloc2"));
                out.map_view_of_file3 = reinterpret_cast<MapViewOfFile3Fn>(
                    GetProcAddress(module, "MapViewOfFile3"));
                out.unmap_view_of_file2 = reinterpret_cast<UnmapViewOfFile2Fn>(
                    GetProcAddress(module, "UnmapViewOfFile2"));
            }
            return out;
        }();
        return apis;
    }

    // Store one exact Windows placeholder and coalesce only adjacent spans with the same guest
    // ownership. Coalescing free and guest-reserved spans together would lose the boundary needed
    // to stop an automatic map from consuming someone else's reservation.
    void remember_placeholder_locked(std::vector<PlaceholderSpan>& spans,
                                     uint64_t base, uint64_t size) {
        if (!size) return;
        spans.push_back({base, size});
        std::sort(spans.begin(), spans.end(),
                  [](const PlaceholderSpan& a, const PlaceholderSpan& b) {
                      return a.base < b.base;
                  });
        for (size_t i = 1; i < spans.size();) {
            PlaceholderSpan& left = spans[i - 1];
            const PlaceholderSpan right = spans[i];
            if (left.base + left.size != right.base) { ++i; continue; }
            if (VirtualFree(reinterpret_cast<void*>(static_cast<uintptr_t>(left.base)),
                            static_cast<SIZE_T>(left.size + right.size),
                            MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) {
                left.size += right.size;
                spans.erase(spans.begin() + i);
            } else {
                ++i;
            }
        }
    }

    void remember_free_placeholder_locked(uint64_t base, uint64_t size) {
        remember_placeholder_locked(g_free_placeholders, base, size);
    }

    void remember_guest_placeholder_locked(uint64_t base, uint64_t size) {
        remember_placeholder_locked(g_guest_placeholders, base, size);
    }

    // Remove one exact page-aligned subrange from a free placeholder. VirtualFree with
    // MEM_PRESERVE_PLACEHOLDER splits the Windows placeholder without releasing either side.
    void* take_placeholder_locked(std::vector<PlaceholderSpan>& spans,
                                  uint64_t hint, uint64_t len, uint64_t align) {
        if (!len) return nullptr;
        const uint64_t requested_align = align ? align : 0x1000;
        for (size_t i = 0; i < spans.size(); ++i) {
            const PlaceholderSpan span = spans[i];
            uint64_t base = hint ? hint : align_up(span.base, requested_align);
            if ((base & (requested_align - 1)) || base < span.base ||
                base > UINT64_MAX - len ||
                base + len > span.base + span.size) continue;
            if (!hint && (base < kGuestAutoVaMin ||
                          base + len - 1 > kGuestAutoVaMax)) continue;

            const uint64_t prefix = base - span.base;
            const uint64_t suffix = span.base + span.size - (base + len);
            spans.erase(spans.begin() + i);
            if (prefix && !VirtualFree(
                    reinterpret_cast<void*>(static_cast<uintptr_t>(span.base)),
                    static_cast<SIZE_T>(prefix),
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
                remember_placeholder_locked(spans, span.base, span.size);
                return nullptr;
            }
            if (suffix && !VirtualFree(
                    reinterpret_cast<void*>(static_cast<uintptr_t>(base)),
                    static_cast<SIZE_T>(len),
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
                if (prefix) {
                    remember_placeholder_locked(spans, span.base, prefix);
                    remember_placeholder_locked(spans, base, len + suffix);
                } else {
                    remember_placeholder_locked(spans, span.base, span.size);
                }
                return nullptr;
            }
            if (prefix) remember_placeholder_locked(spans, span.base, prefix);
            if (suffix) remember_placeholder_locked(spans, base + len, suffix);
            return reinterpret_cast<void*>(static_cast<uintptr_t>(base));
        }
        return nullptr;
    }

    void* take_free_placeholder_locked(uint64_t hint, uint64_t len, uint64_t align) {
        return take_placeholder_locked(g_free_placeholders, hint, len, align);
    }

    void* take_guest_placeholder_locked(uint64_t hint, uint64_t len, uint64_t align) {
        return take_placeholder_locked(g_guest_placeholders, hint, len, align);
    }

    void restore_placeholder_owner_locked(PlaceholderOwner owner, uint64_t base, uint64_t size) {
        if (owner == PlaceholderOwner::Guest)
            remember_guest_placeholder_locked(base, size);
        else
            remember_free_placeholder_locked(base, size);
    }

    // OS-chosen placeholder inside the guest auto window, with a caller-narrowed lowest bound.
    // The upper bound is ALWAYS kGuestAutoVaMax: the shared window is never widened (raising it
    // into the PS5-libc-rejected 1-8 TiB gap is what sank #982); `lowest` only narrows placement
    // WITHIN the window (the huge-reserve top band in win_reserve).
    AcquiredPlaceholder acquire_placeholder_window_locked(uint64_t lowest, uint64_t len,
                                                          uint64_t align) {
        const PlaceholderApis& apis = placeholder_apis();
        if (!apis.virtual_alloc2 || !apis.map_view_of_file3 ||
            !apis.unmap_view_of_file2 || !len || (len & 0xfff) ||
            (align && (align & (align - 1)))) return {};
        if (lowest < kGuestAutoVaMin || lowest > kGuestAutoVaMax) return {};
        MEM_ADDRESS_REQUIREMENTS requirements{};
        requirements.LowestStartingAddress =
            reinterpret_cast<void*>(static_cast<uintptr_t>(lowest));
        requirements.HighestEndingAddress =
            reinterpret_cast<void*>(static_cast<uintptr_t>(kGuestAutoVaMax));
        requirements.Alignment = static_cast<SIZE_T>(
            std::max<uint64_t>(align ? align : 0x4000, kWinAllocationGranularity));
        MEM_EXTENDED_PARAMETER parameter{};
        parameter.Type = MemExtendedParameterAddressRequirements;
        parameter.Pointer = &requirements;
        if (len > UINT64_MAX - (kWinAllocationGranularity - 1)) return {};
        const uint64_t cover_size = align_up(len, kWinAllocationGranularity);
        void* cover = apis.virtual_alloc2(
            GetCurrentProcess(), nullptr, static_cast<SIZE_T>(cover_size),
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, &parameter, 1);
        if (!cover) return {};
        const uint64_t base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cover));
        remember_free_placeholder_locked(base, cover_size);
        return {take_free_placeholder_locked(base, len, align), PlaceholderOwner::Free};
    }

    AcquiredPlaceholder acquire_placeholder_locked(uint64_t hint, uint64_t len,
                                                   uint64_t align, bool allow_guest) {
        const PlaceholderApis& apis = placeholder_apis();
        if (!apis.virtual_alloc2 || !apis.map_view_of_file3 ||
            !apis.unmap_view_of_file2 || !len || (len & 0xfff) ||
            (align && (align & (align - 1)))) return {};

        if (void* recycled = take_free_placeholder_locked(hint, len, align))
            return {recycled, PlaceholderOwner::Free};
        if (hint && allow_guest) {
            if (void* reserved = take_guest_placeholder_locked(hint, len, align))
                return {reserved, PlaceholderOwner::Guest};
        }

        if (hint) {
            if (hint & 0xfff) return {};
            const uint64_t cover_base = hint & ~(kWinAllocationGranularity - 1);
            const uint64_t end = hint + len;
            if (end < hint) return {};
            const uint64_t cover_end = align_up(end, kWinAllocationGranularity);
            void* cover = apis.virtual_alloc2(
                GetCurrentProcess(),
                reinterpret_cast<void*>(static_cast<uintptr_t>(cover_base)),
                static_cast<SIZE_T>(cover_end - cover_base),
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
            if (!cover) return {};
            remember_free_placeholder_locked(cover_base, cover_end - cover_base);
            return {take_free_placeholder_locked(hint, len, align), PlaceholderOwner::Free};
        }

        return acquire_placeholder_window_locked(kGuestAutoVaMin, len, align);
    }

    bool protect_committed_regions(uint64_t base, uint64_t len, int hp) {
        if (!len || base > UINT64_MAX - len) return false;
        const uint64_t end = base + len;
        uint64_t cursor = base;
        while (cursor < end) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (!VirtualQuery(reinterpret_cast<void*>(static_cast<uintptr_t>(cursor)),
                              &mbi, sizeof(mbi))) return false;
            const uint64_t region_end = std::min<uint64_t>(
                end, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mbi.BaseAddress)) +
                         mbi.RegionSize);
            if (region_end <= cursor || mbi.State == MEM_FREE) return false;
            if (mbi.State == MEM_COMMIT) {
                DWORD old = 0;
                if (!VirtualProtect(reinterpret_cast<void*>(static_cast<uintptr_t>(cursor)),
                                    static_cast<SIZE_T>(region_end - cursor),
                                    win_page_prot(hp), &old)) return false;
            }
            cursor = region_end;
        }
        return true;
    }

    void* map_placeholder_section_view(HANDLE section, uint64_t hint, uint64_t rel,
                                       uint64_t len, uint64_t align, int hp) {
        const PlaceholderApis& apis = placeholder_apis();
        if (!apis.map_view_of_file3 || (rel & 0xfff)) return nullptr;
        std::lock_guard<std::mutex> lk(g_dview_mx);
        const AcquiredPlaceholder acquired =
            acquire_placeholder_locked(hint, len, align, true);
        if (!acquired.address) return nullptr;
        void* view = apis.map_view_of_file3(
            section, GetCurrentProcess(), acquired.address, rel,
            static_cast<SIZE_T>(len), MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE, nullptr, 0);
        if (!view) {
            const DWORD error = GetLastError();
            restore_placeholder_owner_locked(
                acquired.owner,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(acquired.address)), len);
            SetLastError(error);
            return nullptr;
        }
        const uint64_t base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(view));
        if (!protect_committed_regions(base, len, hp)) {
            const DWORD error = GetLastError();
            if (!apis.unmap_view_of_file2(GetCurrentProcess(), view,
                                          MEM_PRESERVE_PLACEHOLDER)) {
                std::fprintf(stderr,
                             "[memhle] fatal: could not restore placeholder after protection "
                             "failure\n");
                std::abort();
            }
            restore_placeholder_owner_locked(acquired.owner, base, len);
            SetLastError(error);
            return nullptr;
        }
        return view;
    }

    void* map_section_view(uint64_t hint, uint64_t len, int hp, uint64_t phys, uint64_t align) {
        HANDLE section = dmem_section();
        if (!section || !len || phys < kDmemBase || phys - kDmemBase > kDmemTotal ||
            len > kDmemTotal - (phys - kDmemBase)) return nullptr;

        const uint64_t rel = phys - kDmemBase;
        const uint64_t file_off = rel & ~(kWinAllocationGranularity - 1);
        const uint64_t delta = rel - file_off;
        uint64_t view_size = align_up(delta + len, kWinAllocationGranularity);
        const uint64_t requested_align = align ? align : 0x4000;
        bool sparse = false;
        bool placeholder = false;
        void* view = nullptr;
        // Placeholder replacement is page-granular, unlike MapViewOfFileEx. Map the guest range
        // directly at its physical-section offset, with no 64 KiB congruence requirement.
        view = map_placeholder_section_view(section, hint, rel, len, requested_align, hp);
        placeholder = view != nullptr;
        sparse = false;
        if (!view) {
            void* requested = nullptr;
            if (hint) {
                if (hint < delta || ((hint - delta) & (kWinAllocationGranularity - 1)) != 0) {
                    MLOG("section map incompatible hint=0x%llx phys=0x%llx delta=0x%llx\n",
                         (unsigned long long)hint, (unsigned long long)phys,
                         (unsigned long long)delta);
                    return nullptr;
                }
                requested = (void*)(uintptr_t)(hint - delta);
            }
            view = MapViewOfFileEx(section, FILE_MAP_ALL_ACCESS,
                                   (DWORD)(file_off >> 32), (DWORD)(file_off & 0xffffffffu),
                                   (SIZE_T)view_size, requested);
        }
        if (!view) return nullptr;
        uint8_t* guest = placeholder ? static_cast<uint8_t*>(view)
                                     : static_cast<uint8_t*>(view) + delta;
        if (placeholder) view_size = len;

        if (!hint && requested_align &&
            ((uint64_t)(uintptr_t)guest & (requested_align - 1)) != 0) {
            if (placeholder) {
                const PlaceholderApis& apis = placeholder_apis();
                if (!apis.unmap_view_of_file2 ||
                    !apis.unmap_view_of_file2(GetCurrentProcess(), view,
                                              MEM_PRESERVE_PLACEHOLDER)) {
                    std::fprintf(stderr,
                                 "[memhle] fatal: could not restore misaligned placeholder "
                                 "view\n");
                    std::abort();
                }
                std::lock_guard<std::mutex> lk(g_dview_mx);
                remember_free_placeholder_locked(
                    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(view)), len);
            } else {
                UnmapViewOfFile(view);
            }
            return nullptr;
        }
        if (!sparse && !ensure_section_pages_committed(
                           static_cast<uint64_t>(reinterpret_cast<uintptr_t>(guest)),
                           len, HP_R | HP_W)) {
            MLOG("section commit va=0x%llx len=0x%llx failed error=%lu\n",
                 (unsigned long long)(uintptr_t)guest, (unsigned long long)len, GetLastError());
            UnmapViewOfFile(view);
            return nullptr;
        }
        DWORD old = 0;
        if (!sparse && !VirtualProtect(guest, (SIZE_T)len, win_page_prot(hp), &old)) {
            MLOG("section protect va=0x%llx len=0x%llx failed error=%lu\n",
                 (unsigned long long)(uintptr_t)guest, (unsigned long long)len, GetLastError());
            UnmapViewOfFile(view);
            return nullptr;
        }
        {
            std::lock_guard<std::mutex> lk(g_dview_mx);
            DmemView tracked{
                (uint64_t)(uintptr_t)guest, len, phys, view, view_size, sparse, placeholder
            };
            if (sparse) {
                const uint64_t pages = (len + 0x3fff) / 0x4000;
                tracked.committed_pages.resize((pages + 63) / 64);
                tracked.page_cache_generation = host::guest_mapping_generation();
            }
            g_dviews.push_back(std::move(tracked));
        }
        host::guest_write_watch_notify_direct_mapping_added(
            (uint64_t)(uintptr_t)guest, len, phys, win_page_prot(hp));
        if (sparse)
            MLOG("map_dmem sparse placeholder view va=0x%llx len=0x%llx phys=0x%llx align=0x%llx\n",
                 (unsigned long long)(uintptr_t)guest, (unsigned long long)len,
                 (unsigned long long)phys, (unsigned long long)requested_align);
        return guest;
    }

    bool sparse_dmem_view_contains(uint64_t begin, uint64_t end, DmemView* out = nullptr) {
        if (begin >= end) return false;
        std::lock_guard<std::mutex> lk(g_dview_mx);
        for (const DmemView& view : g_dviews) {
            if (!view.sparse || begin < view.guest_base ||
                end > view.guest_base + view.guest_size) continue;
            if (out) *out = view;
            return true;
        }
        return false;
    }

    bool sparse_dmem_view_overlaps(uint64_t begin, uint64_t end) {
        if (begin >= end) return false;
        std::lock_guard<std::mutex> lk(g_dview_mx);
        for (const DmemView& view : g_dviews)
            if (view.sparse && begin < view.guest_base + view.guest_size &&
                end > view.guest_base) return true;
        return false;
    }

    bool tracked_mapping_access(uint64_t begin, uint64_t end, bool write, int& hp) {
        if (begin >= end) return false;
        struct AccessCache {
            uint64_t generation = 0;
            uint64_t begin = 0;
            uint64_t end = 0;
            int prot = 0;
        };
        static thread_local AccessCache cache;
        static const bool cache_disabled = getenv("PROSPER_NO_SPARSE_DMEM_ACCESS_CACHE") != nullptr;
        const uint64_t generation = host::guest_mapping_generation();
        if (!cache_disabled && cache.generation == generation &&
            begin >= cache.begin && end <= cache.end) {
            if (!(cache.prot & HP_R) || (write && !(cache.prot & HP_W))) return false;
            hp = cache.prot;
            return true;
        }
        std::lock_guard<std::mutex> lk(g_mx);
        const Mapping* best = nullptr;
        for (const Mapping& mapping : g_maps) {
            if (!mapping.committed || begin < mapping.base ||
                begin >= mapping.base + mapping.size) continue;
            if (!best || mapping.base > best->base) best = &mapping;
        }
        if (!best || end > best->base + best->size || !(best->prot & HP_R) ||
            (write && !(best->prot & HP_W))) return false;
        hp = best->prot;
        if (!cache_disabled) {
            cache.generation = host::guest_mapping_generation();
            cache.begin = best->base;
            cache.end = best->base + best->size;
            cache.prot = best->prot;
        }
        return true;
    }

    bool tracked_mapping_protection(uint64_t begin, uint64_t end, int& hp) {
        if (begin >= end) return false;
        std::lock_guard<std::mutex> lk(g_mx);
        const Mapping* best = nullptr;
        for (const Mapping& mapping : g_maps) {
            if (!mapping.committed || begin < mapping.base ||
                begin >= mapping.base + mapping.size) continue;
            if (!best || mapping.base > best->base) best = &mapping;
        }
        if (!best || end > best->base + best->size) return false;
        hp = best->prot;
        return true;
    }

    // SEC_RESERVE commitment is shared by every view of the physical page, but each view keeps
    // its own protection contract. A page first committed through an RW alias can therefore make
    // an untouched RO alias committed too; immediately apply every alias's tracked protection.
    bool apply_dmem_page_protections_locked(uint64_t phys_page) {
        if (phys_page > UINT64_MAX - 0x4000) return false;
        const uint64_t generation = host::guest_mapping_generation();
        for (DmemView& view : g_dviews) {
            if (!view.sparse || phys_page < view.phys ||
                phys_page + 0x4000 > view.phys + view.guest_size) continue;
            const uint64_t alias_page = view.guest_base + (phys_page - view.phys);
            int hp = 0;
            if (!tracked_mapping_protection(alias_page, alias_page + 0x4000, hp)) return false;
            MEMORY_BASIC_INFORMATION mbi{};
            if (!VirtualQuery(reinterpret_cast<void*>(static_cast<uintptr_t>(alias_page)),
                              &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT) return false;
            DWORD old = 0;
            if (!VirtualProtect(reinterpret_cast<void*>(static_cast<uintptr_t>(alias_page)),
                                0x4000, win_page_prot(hp), &old)) return false;

            if (view.page_cache_generation != generation) {
                std::fill(view.committed_pages.begin(), view.committed_pages.end(), 0);
                view.page_cache_generation = generation;
            }
            const uint64_t index = (alias_page - view.guest_base) / 0x4000;
            if (index / 64 >= view.committed_pages.size()) return false;
            view.committed_pages[index / 64] |= 1ull << (index & 63);
        }
        return true;
    }

    int commit_sparse_dmem(uint64_t addr, uint64_t len, bool write) {
        if (!len || addr > UINT64_MAX - len ||
            !sparse_dmem_view_contains(addr, addr + len)) return 0;
        int hp = 0;
        if (!tracked_mapping_access(addr, addr + len, write, hp)) return 0;
        const uint64_t first = addr & ~0x3fffull;
        const uint64_t last = (addr + len - 1) & ~0x3fffull;
        static const bool page_cache_disabled =
            getenv("PROSPER_NO_SPARSE_DMEM_PAGE_CACHE") != nullptr;
        if (!page_cache_disabled) {
            std::lock_guard<std::mutex> lk(g_dview_mx);
            DmemView* selected = nullptr;
            for (DmemView& view : g_dviews) {
                if (!view.sparse || addr < view.guest_base ||
                    addr + len > view.guest_base + view.guest_size) continue;
                selected = &view;
                break;
            }
            if (!selected) return 0;
            const uint64_t generation = host::guest_mapping_generation();
            if (selected->page_cache_generation != generation) {
                std::fill(selected->committed_pages.begin(),
                          selected->committed_pages.end(), 0);
                selected->page_cache_generation = generation;
            }
            const uint64_t view_first = selected->guest_base & ~0x3fffull;
            for (uint64_t page = first;; page += 0x4000) {
                const uint64_t index = (page - view_first) / 0x4000;
                if (index / 64 >= selected->committed_pages.size()) return 0;
                const uint64_t mask = 1ull << (index & 63);
                if (!(selected->committed_pages[index / 64] & mask)) {
                    if (!VirtualAlloc((void*)(uintptr_t)page, 0x4000, MEM_COMMIT,
                                      win_page_prot(hp))) {
                        MEMORY_BASIC_INFORMATION mbi{};
                        if (!VirtualQuery((const void*)(uintptr_t)page, &mbi, sizeof(mbi)) ||
                            mbi.State != MEM_COMMIT) return 0;
                        const DWORD blocked = PAGE_NOACCESS | PAGE_GUARD;
                        const DWORD readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                                               PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                                               PAGE_EXECUTE_WRITECOPY;
                        const DWORD writable = PAGE_READWRITE | PAGE_WRITECOPY |
                                               PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
                        if ((mbi.Protect & blocked) || !(mbi.Protect & readable) ||
                            (write && !(mbi.Protect & writable))) return 0;
                    }
                    const uint64_t phys_page = selected->phys + (page - selected->guest_base);
                    if (!apply_dmem_page_protections_locked(phys_page)) return 0;
                }
                if (page == last) break;
            }
            return 1;
        }
        uint64_t page = first;
        for (;;) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (!VirtualQuery((const void*)(uintptr_t)page, &mbi, sizeof(mbi))) return 0;
            if (mbi.State != MEM_COMMIT) {
                if (!VirtualAlloc((void*)(uintptr_t)page, 0x4000, MEM_COMMIT,
                                  win_page_prot(hp))) {
                    MLOG("sparse dmem commit va=0x%llx failed error=%lu\n",
                         (unsigned long long)page, GetLastError());
                    return 0;
                }
            } else {
                const DWORD blocked = PAGE_NOACCESS | PAGE_GUARD;
                const DWORD readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                                       PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                                       PAGE_EXECUTE_WRITECOPY;
                const DWORD writable = PAGE_READWRITE | PAGE_WRITECOPY |
                                       PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
                if ((mbi.Protect & blocked) || !(mbi.Protect & readable) ||
                    (write && !(mbi.Protect & writable))) return 0;
            }
            {
                std::lock_guard<std::mutex> lk(g_dview_mx);
                DmemView* selected = nullptr;
                for (DmemView& view : g_dviews) {
                    if (!view.sparse || page < view.guest_base ||
                        page + 0x4000 > view.guest_base + view.guest_size) continue;
                    selected = &view;
                    break;
                }
                if (!selected || !apply_dmem_page_protections_locked(
                        selected->phys + (page - selected->guest_base))) return 0;
            }
            if (page == last) break;
            page += 0x4000;
        }
        return 1;
    }

    bool sparse_dmem_page_uncommitted(uint64_t addr) {
        if (!sparse_dmem_view_contains(addr, addr + 1)) return false;
        MEMORY_BASIC_INFORMATION mbi{};
        return VirtualQuery((const void*)(uintptr_t)addr, &mbi, sizeof(mbi)) &&
               mbi.State != MEM_COMMIT;
    }

    // A released physical range can be allocated again while old virtual aliases still exist.
    // Zeroing the shared section at allocation time gives every alias the console's fresh-page view.
    void dmem_zero(uint64_t phys, uint64_t len) {
        // A temporary physical-section alias bypasses protections installed on persistent guest
        // aliases. Invalidate first so a recycled direct-memory allocation cannot retain a clean
        // texture watch for bytes this zeroing operation is about to replace.
        host::guest_write_watch_notify_physical_write(phys, len);
        HANDLE section = dmem_section();
        if (!section || !len || phys < kDmemBase || phys - kDmemBase > kDmemTotal ||
            len > kDmemTotal - (phys - kDmemBase)) return;
        const uint64_t rel = phys - kDmemBase;
        const uint64_t file_off = rel & ~(kWinAllocationGranularity - 1);
        const uint64_t delta = rel - file_off;
        const uint64_t view_size = align_up(delta + len, kWinAllocationGranularity);
        void* view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS,
                                   (DWORD)(file_off >> 32), (DWORD)(file_off & 0xffffffffu),
                                   (SIZE_T)view_size);
        if (!view) {
            MLOG("dmem zero map phys=0x%llx len=0x%llx failed error=%lu\n",
                 (unsigned long long)phys, (unsigned long long)len, GetLastError());
            return;
        }
        uint8_t* bytes = (uint8_t*)view + delta;
        if (ensure_section_pages_committed(
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bytes)),
                len, HP_R | HP_W)) {
            memset(bytes, 0, (size_t)len);
            bool protections_ok = true;
            {
                std::lock_guard<std::mutex> lk(g_dview_mx);
                for (uint64_t page = phys; page < phys + len; page += 0x4000) {
                    if (!apply_dmem_page_protections_locked(page)) {
                        protections_ok = false;
                        break;
                    }
                }
            }
            if (!protections_ok)
                MLOG("dmem zero could not restore alias protections phys=0x%llx len=0x%llx\n",
                     (unsigned long long)phys, (unsigned long long)len);
        } else {
            MLOG("dmem zero commit phys=0x%llx len=0x%llx failed error=%lu\n",
                 (unsigned long long)phys, (unsigned long long)len, GetLastError());
        }
        UnmapViewOfFile(view);
    }

    struct PhysRange { uint64_t start, end; };
    std::mutex g_dmem_seen_mx;
    std::vector<PhysRange> g_dmem_seen;

    // Paging-file sections are zero-filled on first commit, so touching every first-time allocation
    // would needlessly charge and write several GiB during title startup. Only ranges overlapping a
    // previous allocation can contain stale bytes. Keep a compact union of ever-allocated ranges and
    // zero on reuse; sequential first-time allocations normally collapse to one interval.
    void dmem_prepare_allocation(uint64_t phys, uint64_t len) {
        if (!len) return;
        const uint64_t end = phys + len;
        bool reused = false;
        {
            std::lock_guard<std::mutex> lk(g_dmem_seen_mx);
            for (const auto& r : g_dmem_seen) {
                if (r.start < end && r.end > phys) { reused = true; break; }
            }
            g_dmem_seen.push_back({ phys, end });
            std::sort(g_dmem_seen.begin(), g_dmem_seen.end(),
                      [](const PhysRange& a, const PhysRange& b) { return a.start < b.start; });
            std::vector<PhysRange> merged;
            merged.reserve(g_dmem_seen.size());
            for (const auto& r : g_dmem_seen) {
                if (merged.empty() || merged.back().end < r.start) merged.push_back(r);
                else if (r.end > merged.back().end) merged.back().end = r.end;
            }
            g_dmem_seen.swap(merged);
        }
        if (reused) dmem_zero(phys, len);
    }

    PrivatePlaceholderView* private_placeholder_view_containing_locked(uint64_t begin,
                                                                       uint64_t end) {
        for (PrivatePlaceholderView& view : g_private_placeholder_views) {
            if (begin >= view.base && end >= begin && end <= view.base + view.size)
                return &view;
        }
        return nullptr;
    }

    void* replace_placeholder_with_private_locked(uint64_t base, uint64_t len, int hp,
                                                  PlaceholderOwner owner) {
        const PlaceholderApis& apis = placeholder_apis();
        if (!apis.virtual_alloc2) return nullptr;
        void* result = apis.virtual_alloc2(
            GetCurrentProcess(),
            reinterpret_cast<void*>(static_cast<uintptr_t>(base)),
            static_cast<SIZE_T>(len),
            MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
            win_page_prot(hp), nullptr, 0);
        if (!result) {
            restore_placeholder_owner_locked(owner, base, len);
            return nullptr;
        }
        g_private_placeholder_views.push_back({base, len});
        return result;
    }

    enum class FlexibleHintState { Unavailable, Free, GuestReservation };

    bool hle_reservation_covers(uint64_t begin, uint64_t end) {
        if (begin >= end) return false;
        {
            std::lock_guard<std::mutex> lk(g_mx);
            uint64_t cursor = begin;
            while (cursor < end) {
                const Mapping* cover = nullptr;
                for (const auto& m : g_maps) {
                    if (cursor < m.base || cursor >= m.base + m.size) continue;
                    if (!cover || m.base > cover->base) cover = &m;
                }
                if (!cover || cover->committed) break;
                cursor = std::min<uint64_t>(end, cover->base + cover->size);
            }
            if (cursor == end) return true;
        }
        std::lock_guard<std::mutex> lk(g_dview_mx);
        if (private_placeholder_view_containing_locked(begin, end)) return true;
        for (const PlaceholderSpan& span : g_guest_placeholders)
            if (begin >= span.base && end <= span.base + span.size) return true;
        // A partial direct-memory unmap returns its hole to the HLE-owned free-placeholder
        // registry. Flexible memory may claim that exact hole just like direct memory can.
        for (const PlaceholderSpan& span : g_free_placeholders)
            if (begin >= span.base && end <= span.base + span.size) return true;
        return false;
    }

    // Classify an exact flexible-memory hint without treating arbitrary host reservations as guest
    // storage. In particular, VirtualAlloc(..., MEM_COMMIT) succeeds on already committed pages;
    // checking only the HLE tracker would therefore let an untracked image/host allocation be
    // adopted. MEM_RESERVE is eligible only when the guest owns the reservation or placeholder.
    FlexibleHintState flexible_hint_state(uint64_t base, uint64_t len) {
        if (!len || base > UINT64_MAX - len) return FlexibleHintState::Unavailable;
        const uint64_t end = base + len;
        {
            std::lock_guard<std::mutex> lk(g_mx);
            for (const auto& m : g_maps)
                if (m.committed && m.base < end && m.base + m.size > base)
                    return FlexibleHintState::Unavailable;
        }

        bool saw_reservation = false;
        uint64_t cursor = base;
        while (cursor < end) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (!VirtualQuery(reinterpret_cast<void*>(static_cast<uintptr_t>(cursor)),
                              &mbi, sizeof(mbi)))
                return FlexibleHintState::Unavailable;
            const uint64_t region_base =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mbi.BaseAddress));
            const uint64_t region_end =
                region_base > UINT64_MAX - mbi.RegionSize
                    ? UINT64_MAX
                    : region_base + static_cast<uint64_t>(mbi.RegionSize);
            if (region_end <= cursor || mbi.State == MEM_COMMIT)
                return FlexibleHintState::Unavailable;
            if (mbi.State == MEM_RESERVE) {
                if (!hle_reservation_covers(cursor, std::min<uint64_t>(end, region_end)))
                    return FlexibleHintState::Unavailable;
                saw_reservation = true;
            }
            cursor = std::min<uint64_t>(end, region_end);
        }
        return saw_reservation ? FlexibleHintState::GuestReservation
                               : FlexibleHintState::Free;
    }

    // Commit `len` at `hint` (0 = OS chooses). Placeholder-backed guest reservations and holes
    // are replaced with ordinary private pages so the same 16 KiB address can later be reused by
    // either flexible or direct memory. Legacy VirtualAlloc reservations retain their old path.
    void* win_commit(uint64_t hint, uint64_t len, int hp) {
        DWORD pp = win_page_prot(hp);
        void* result = nullptr;
        if (!hint) {
            result = VirtualAlloc(nullptr, (SIZE_T)len, MEM_RESERVE | MEM_COMMIT, pp);
        } else {
            bool placeholder_owned = false;
            bool direct_collision = false;
            if (hint <= UINT64_MAX - len) {
                std::lock_guard<std::mutex> lk(g_dview_mx);
                if (private_placeholder_view_containing_locked(hint, hint + len)) {
                    placeholder_owned = true;
                    bool needs_commit = false;
                    uint64_t cursor = hint;
                    while (cursor < hint + len) {
                        MEMORY_BASIC_INFORMATION mbi{};
                        if (!VirtualQuery(
                                reinterpret_cast<void*>(static_cast<uintptr_t>(cursor)),
                                &mbi, sizeof(mbi)) || mbi.State == MEM_FREE) break;
                        if (mbi.State == MEM_RESERVE) needs_commit = true;
                        const uint64_t region_end = std::min<uint64_t>(
                            hint + len,
                            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mbi.BaseAddress)) +
                                mbi.RegionSize);
                        if (region_end <= cursor) break;
                        cursor = region_end;
                    }
                    if (cursor == hint + len) {
                        if (!needs_commit || VirtualAlloc(
                                reinterpret_cast<void*>(static_cast<uintptr_t>(hint)),
                                static_cast<SIZE_T>(len), MEM_COMMIT, pp)) {
                            DWORD old = 0;
                            if (VirtualProtect(
                                    reinterpret_cast<void*>(static_cast<uintptr_t>(hint)),
                                    static_cast<SIZE_T>(len), pp, &old))
                                result = reinterpret_cast<void*>(static_cast<uintptr_t>(hint));
                        }
                    }
                } else {
                    PlaceholderOwner owner = PlaceholderOwner::Guest;
                    void* placeholder = take_guest_placeholder_locked(hint, len, 0x1000);
                    if (!placeholder) {
                        owner = PlaceholderOwner::Free;
                        placeholder = take_free_placeholder_locked(hint, len, 0x1000);
                    }
                    if (placeholder) {
                        placeholder_owned = true;
                        result = replace_placeholder_with_private_locked(hint, len, hp, owner);
                    } else {
                        for (const DmemView& view : g_dviews) {
                            if (hint < view.guest_base + view.guest_size &&
                                hint + len > view.guest_base) {
                                direct_collision = true;
                                break;
                            }
                        }
                    }
                }
            }
            if (!result && !placeholder_owned && !direct_collision) {
                result = VirtualAlloc(reinterpret_cast<void*>(static_cast<uintptr_t>(hint)),
                                      static_cast<SIZE_T>(len), MEM_COMMIT, pp);
                if (!result)
                    result = VirtualAlloc(reinterpret_cast<void*>(static_cast<uintptr_t>(hint)),
                                          static_cast<SIZE_T>(len),
                                          MEM_RESERVE | MEM_COMMIT, pp);
            }
        }
        if (result)
            host::guest_write_watch_notify_direct_mapping_added(
                (uint64_t)(uintptr_t)result, len, (uint64_t)(uintptr_t)result, pp);
        return result;
    }

    void* win_commit_flexible_exact(uint64_t hint, uint64_t len, int hp) {
        const FlexibleHintState state = flexible_hint_state(hint, len);
        if (state == FlexibleHintState::GuestReservation)
            return win_commit(hint, len, hp);
        if (state != FlexibleHintState::Free) return nullptr;

        const DWORD pp = win_page_prot(hp);
        void* result = nullptr;
        {
            // VirtualAlloc reservations round a free address down to 64 KiB. Use the placeholder
            // splitter first so a fixed 16 KiB-aligned guest hint is either mapped exactly or not
            // at all; the retained prefix/suffix become reusable HLE-owned placeholders.
            std::lock_guard<std::mutex> lk(g_dview_mx);
            AcquiredPlaceholder acquired =
                acquire_placeholder_locked(hint, len, 0x4000, false);
            if (acquired.address ==
                reinterpret_cast<void*>(static_cast<uintptr_t>(hint)))
                result = replace_placeholder_with_private_locked(
                    hint, len, hp, acquired.owner);
            else if (acquired.address)
                restore_placeholder_owner_locked(
                    acquired.owner,
                    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(acquired.address)), len);
        }
        if (!result && (hint & (kWinAllocationGranularity - 1)) == 0)
            result = VirtualAlloc(
                reinterpret_cast<void*>(static_cast<uintptr_t>(hint)),
                static_cast<SIZE_T>(len), MEM_RESERVE | MEM_COMMIT, pp);
        if (result)
            host::guest_write_watch_notify_direct_mapping_added(
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result)), len,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result)), pp);
        return result;
    }

    // Map flexible memory at the first host-free address at or above a guest hint. Windows'
    // default VirtualAlloc(nullptr, ...) placement is not a search-from-hint operation and may
    // also land in the 1-8 TiB aperture rejected by Sony libc. Probe the valid low guest aperture
    // explicitly; VirtualAlloc at the selected candidate makes each probe race-safe.
    void* win_commit_from(uint64_t start, uint64_t len, int hp) {
        if (!len || len > kGuestAutoVaMax - kGuestAutoVaMin + 1)
            return nullptr;
        const uint64_t first = std::max<uint64_t>(start, kGuestAutoVaMin);
        if (first > UINT64_MAX - (kWinAllocationGranularity - 1))
            return nullptr;
        uint64_t cursor = align_up(first, kWinAllocationGranularity);
        const uint64_t last = kGuestAutoVaMax - len + 1;
        const DWORD pp = win_page_prot(hp);
        while (cursor <= last) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (!VirtualQuery(reinterpret_cast<void*>(static_cast<uintptr_t>(cursor)),
                              &mbi, sizeof(mbi)))
                return nullptr;
            const uint64_t region_base =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mbi.BaseAddress));
            const uint64_t region_end =
                region_base > UINT64_MAX - mbi.RegionSize
                    ? UINT64_MAX
                    : region_base + static_cast<uint64_t>(mbi.RegionSize);
            if (mbi.State == MEM_FREE) {
                const uint64_t candidate = align_up(std::max(cursor, region_base),
                                                    kWinAllocationGranularity);
                if (candidate <= last && candidate <= region_end &&
                    len <= region_end - candidate) {
                    if (void* result = VirtualAlloc(
                            reinterpret_cast<void*>(static_cast<uintptr_t>(candidate)),
                            static_cast<SIZE_T>(len), MEM_RESERVE | MEM_COMMIT, pp)) {
                        host::guest_write_watch_notify_direct_mapping_added(
                            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result)), len,
                            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result)), pp);
                        return result;
                    }
                }
            }
            uint64_t next = region_end > cursor
                ? region_end
                : cursor + kWinAllocationGranularity;
            if (next <= cursor || next > UINT64_MAX - (kWinAllocationGranularity - 1))
                return nullptr;
            cursor = align_up(next, kWinAllocationGranularity);
        }
        return nullptr;
    }
    void* win_map_phys(uint64_t hint, uint64_t len, int hp, uint64_t phys, uint64_t align,
                       bool fixed) {
        if (void* p = map_section_view(hint, len, hp, phys, align)) return p;
        // Without SCE_KERNEL_MAP_FIXED, addrInOut is a search hint. If Windows cannot extend a
        // run of adjacent section views at that exact VA, relocate the mapping and return the
        // chosen address instead of silently changing it into non-aliasing private memory.
        if (hint && !fixed) {
            if (void* p = map_section_view(0, len, hp, phys, align)) return p;
        }
        // A private fixed mapping would report success while severing the physical alias contract.
        // Older Windows versions without placeholder replacement fail visibly instead.
        if (fixed) return nullptr;
        MLOG("map_dmem private fallback hint=0x%llx len=0x%llx phys=0x%llx align=0x%llx error=%lu\n",
             (unsigned long long)hint, (unsigned long long)len,
             (unsigned long long)phys, (unsigned long long)align, GetLastError());
        return win_commit(fixed ? hint : 0, len, hp);
    }
    // Reserve (no commit). Modern Windows placeholders can be partitioned at the guest's 16 KiB
    // page boundary and later replaced by either a section view or private pages.
    void* win_reserve(uint64_t hint, uint64_t len, bool fixed, uint64_t align) {
        // #946/#312: a HUGE non-fixed reservation (UE MallocBinned3's 512 GiB arena, hint
        // 0x1000000000) must not be placed at its literal low hint — a flag-less hint is a
        // search start, and the low base overlaps prosper's low guest-VA occupants (the #946
        // __cxa_guard deadlock; the Linux face is the #312 MB3 corruption). It must also never
        // fall to an unconstrained VirtualAlloc: an out-of-window arena makes the guest's
        // arena-relative batchmaps exceed the window and ENOMEM (a second wedge). Place it via
        // window-bounded placeholders ONLY, preferring the TOP of the window so the low window
        // stays contiguous for ordinary auto-maps; the shared window bounds themselves are
        // untouched (widening kGuestAutoVaMax into the PS5-libc-rejected 1-8 TiB gap sank #982).
        // On placeholder-less hosts this returns ENOMEM rather than a dangerous base.
        if (!fixed && hint && len >= kHugeReserveLen) {
            std::lock_guard<std::mutex> lk(g_dview_mx);
            const uint64_t granule =
                std::max<uint64_t>(align ? align : 0x4000, kWinAllocationGranularity);
            const uint64_t span = align_up(len, kWinAllocationGranularity);
            // Recycle a freed placeholder first (window-bounded via the hint-less take path): an
            // unmapped huge arena's VA stays OS-reserved as a free placeholder, so without this a
            // reserve->unmap->re-reserve cycle exhausts the window and ENOMEMs (review finding on
            // #1084; Linux reuses freed VA naturally via the cursor wrap).
            AcquiredPlaceholder acquired{};
            if (void* recycled = take_free_placeholder_locked(0, len, align))
                acquired = {recycled, PlaceholderOwner::Free};
            if (!acquired.address && span <= kGuestAutoVaMax + 1 - kGuestAutoVaMin) {
                const uint64_t band_low = (kGuestAutoVaMax + 1 - span) & ~(granule - 1);
                if (band_low >= kGuestAutoVaMin)
                    acquired = acquire_placeholder_window_locked(band_low, len, align);
            }
            if (!acquired.address) {  // band contended/undersized: anywhere in-window
                acquired = acquire_placeholder_window_locked(kGuestAutoVaMin, len, align);
                if (acquired.address)
                    MLOG("reserve(huge) top band unavailable — whole-window fallback -> 0x%llx\n",
                         (unsigned long long)(uintptr_t)acquired.address);
            }
            if (!acquired.address) return nullptr;
            const uint64_t base =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(acquired.address));
            remember_guest_placeholder_locked(base, len);
            return acquired.address;
        }
        {
            std::lock_guard<std::mutex> lk(g_dview_mx);
            AcquiredPlaceholder acquired =
                acquire_placeholder_locked(hint, len, align, false);
            if (!acquired.address && hint && !fixed)
                acquired = acquire_placeholder_locked(0, len, align, false);
            if (acquired.address) {
                const uint64_t base = static_cast<uint64_t>(
                    reinterpret_cast<uintptr_t>(acquired.address));
                remember_guest_placeholder_locked(base, len);
                return acquired.address;
            }
        }
        if (hint) {
            if (void* p = VirtualAlloc((void*)hint, (SIZE_T)len, MEM_RESERVE, PAGE_NOACCESS)) return p;
            if (fixed) return nullptr;   // SCE_KERNEL_MAP_FIXED: must be exactly this address
        }
        // VirtualAlloc only guarantees 64 KiB (allocation-granularity) base alignment. Guest allocators
        // reserve with LARGER alignment (e.g. 0x40000 = 256 KiB) and then compute chunk/metadata offsets
        // by rounding a pointer DOWN to that alignment — so a base that is 64 KiB- but not 256 KiB-aligned
        // makes those computations land below the reservation and corrupt/wedge the allocator. Honor a
        // >64 KiB request by over-reserving len+align and returning the aligned sub-base; commits within
        // it still succeed (it is one reserved region) and the slack stays reserved (only wasted address
        // space — trimming a reservation would race a concurrent reserve). CONFIDENCE: HIGH.
        if (align <= 0x10000)
            return VirtualAlloc(nullptr, (SIZE_T)len, MEM_RESERVE, PAGE_NOACCESS);
        void* raw = VirtualAlloc(nullptr, (SIZE_T)(len + align), MEM_RESERVE, PAGE_NOACCESS);
        if (!raw) return nullptr;
        return (void*)(((uint64_t)raw + (align - 1)) & ~(align - 1));
    }
    struct TrackedMappingSlice {
        uint64_t base;
        uint64_t size;
        int prot;
        bool committed;
    };

    // Snapshot the tracker coverage for a protection transaction. VirtualQuery describes every
    // committed host allocation in the process, including allocations that do not belong to the
    // guest. A host MEM_COMMIT result therefore cannot prove that sceKernelMprotect owns the span.
    bool tracked_mapping_slices(uint64_t begin, uint64_t end,
                                std::vector<TrackedMappingSlice>& slices) {
        slices.clear();
        if (begin >= end) return false;
        std::lock_guard<std::mutex> lk(g_mx);
        uint64_t cursor = begin;
        while (cursor < end) {
            const Mapping* best = nullptr;
            for (const auto& mapping : g_maps) {
                if (cursor < mapping.base || cursor >= mapping.base + mapping.size) continue;
                if (!best || mapping.base > best->base) best = &mapping;
            }
            if (!best) return false;
            const uint64_t slice_end = std::min<uint64_t>(end, best->base + best->size);
            slices.push_back({cursor, slice_end - cursor, best->prot, best->committed});
            cursor = slice_end;
        }
        return true;
    }

    bool tracked_slices_have_commit_state(const std::vector<TrackedMappingSlice>& slices,
                                          uint64_t begin, uint64_t end, bool committed) {
        uint64_t cursor = begin;
        for (const TrackedMappingSlice& slice : slices) {
            const uint64_t slice_end = slice.base + slice.size;
            if (slice_end <= cursor || slice.base >= end) continue;
            if (slice.base > cursor || slice.committed != committed) return false;
            cursor = std::min<uint64_t>(end, slice_end);
            if (cursor == end) return true;
        }
        return false;
    }

    bool tracked_slices_back_host_reservation(
        const std::vector<TrackedMappingSlice>& slices, uint64_t begin, uint64_t end) {
        uint64_t cursor = begin;
        for (const TrackedMappingSlice& slice : slices) {
            const uint64_t slice_end = slice.base + slice.size;
            if (slice_end <= cursor || slice.base >= end) continue;
            if (slice.base > cursor) return false;
            const uint64_t overlap_end = std::min<uint64_t>(end, slice_end);
            // A committed tracker entry may still be MEM_RESERVE when it is a sparse section view;
            // every other committed entry must have committed host pages.
            if (slice.committed && !sparse_dmem_view_contains(cursor, overlap_end)) return false;
            cursor = overlap_end;
            if (cursor == end) return true;
        }
        return false;
    }

    bool win_protect(uint64_t addr, uint64_t len, int hp, DWORD* error_out = nullptr) {
        if (error_out) *error_out = ERROR_SUCCESS;
        if (!addr || !len || addr > UINT64_MAX - len) {
            if (error_out) *error_out = ERROR_INVALID_PARAMETER;
            return false;
        }

        // Validate the complete span before changing either host protection or write-watch state.
        // Reserved (uncommitted) pages accept a tracking-only protection change, matching the POSIX
        // PROT_NONE reservation path; MEM_FREE makes the whole operation fail (#387 F3).
        const uint64_t end = addr + len;
        std::vector<TrackedMappingSlice> tracked;
        if (!tracked_mapping_slices(addr, end, tracked)) {
            if (error_out) *error_out = ERROR_INVALID_ADDRESS;
            return false;
        }

        struct CommittedRegion { uint64_t base, size; DWORD old_protection; };
        std::vector<CommittedRegion> committed;
        uint64_t cursor = addr;
        while (cursor < end) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (!VirtualQuery((const void*)(uintptr_t)cursor, &mbi, sizeof(mbi))) {
                if (error_out) *error_out = GetLastError();
                return false;
            }
            const uint64_t region_end = std::min<uint64_t>(
                end, (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize);
            if (region_end <= cursor || mbi.State == MEM_FREE) {
                if (error_out) *error_out = ERROR_INVALID_ADDRESS;
                return false;
            }
            if (mbi.State == MEM_RESERVE) {
                if (!tracked_slices_back_host_reservation(tracked, cursor, region_end)) {
                    if (error_out) *error_out = ERROR_INVALID_ADDRESS;
                    return false;
                }
            } else if (mbi.State == MEM_COMMIT) {
                if (!tracked_slices_have_commit_state(tracked, cursor, region_end, true)) {
                    if (error_out) *error_out = ERROR_INVALID_ADDRESS;
                    return false;
                }
                // Split at tracker boundaries so rollback restores the guest protection recorded
                // before write-watch temporarily made any pages read-only.
                for (const TrackedMappingSlice& slice : tracked) {
                    const uint64_t overlap_begin = std::max(cursor, slice.base);
                    const uint64_t overlap_end = std::min(region_end, slice.base + slice.size);
                    if (overlap_begin < overlap_end)
                        committed.push_back({overlap_begin, overlap_end - overlap_begin,
                                             win_page_prot(slice.prot)});
                }
            }
            cursor = region_end;
        }

        // Update the alias registry before changing committed pages so an overlapping watch first
        // restores its old writable pages. Unrelated mappings leave existing watches armed.
        host::guest_write_watch_notify_direct_mapping_protection(
            addr, len, win_page_prot(hp));
        size_t changed = 0;
        for (const auto& region : committed) {
            DWORD old = 0;
            if (!VirtualProtect((void*)(uintptr_t)region.base, (SIZE_T)region.size,
                                win_page_prot(hp), &old)) {
                const DWORD error = GetLastError();
                for (size_t i = changed; i > 0; --i) {
                    const CommittedRegion& prior = committed[i - 1];
                    DWORD ignored = 0;
                    if (!VirtualProtect((void*)(uintptr_t)prior.base,
                                        (SIZE_T)prior.size, prior.old_protection, &ignored)) {
                        std::fprintf(stderr,
                                     "[memhle] fatal: could not roll back mprotect at 0x%llx\n",
                                     (unsigned long long)prior.base);
                        std::abort();
                    }
                }
                for (const TrackedMappingSlice& slice : tracked) {
                    if (slice.committed)
                        host::guest_write_watch_notify_direct_mapping_protection(
                            slice.base, slice.size, win_page_prot(slice.prot));
                }
                if (error_out) *error_out = error;
                return false;
            }
            changed++;
        }
        return true;
    }
    struct ProtectionSlice { uint64_t base, size; int prot; };

    std::vector<ProtectionSlice> tracked_protection_slices(uint64_t begin, uint64_t end) {
        std::vector<ProtectionSlice> slices;
        std::lock_guard<std::mutex> lk(g_mx);
        for (const Mapping& mapping : g_maps) {
            const uint64_t mapping_end = mapping.base + mapping.size;
            const uint64_t overlap_begin = std::max(begin, mapping.base);
            const uint64_t overlap_end = std::min(end, mapping_end);
            if (overlap_begin < overlap_end && mapping.committed)
                slices.push_back({overlap_begin, overlap_end - overlap_begin, mapping.prot});
        }
        std::sort(slices.begin(), slices.end(),
                  [](const ProtectionSlice& a, const ProtectionSlice& b) {
                      return a.base < b.base;
                  });
        return slices;
    }

    bool restore_view_protections(const DmemView& view) {
        const auto slices = tracked_protection_slices(
            view.guest_base, view.guest_base + view.guest_size);
        for (const ProtectionSlice& slice : slices) {
            uint64_t cursor = slice.base;
            const uint64_t end = slice.base + slice.size;
            while (cursor < end) {
                MEMORY_BASIC_INFORMATION mbi{};
                if (!VirtualQuery(reinterpret_cast<void*>(static_cast<uintptr_t>(cursor)),
                                  &mbi, sizeof(mbi))) {
                    MLOG("partial-unmap VirtualQuery failed va=0x%llx error=%lu\n",
                         (unsigned long long)cursor, GetLastError());
                    return false;
                }
                const uint64_t region_end = std::min<uint64_t>(
                    end, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mbi.BaseAddress)) +
                             mbi.RegionSize);
                if (region_end <= cursor || mbi.State == MEM_FREE) return false;
                if (mbi.State == MEM_COMMIT) {
                    DWORD old = 0;
                    if (!VirtualProtect(
                            reinterpret_cast<void*>(static_cast<uintptr_t>(cursor)),
                            static_cast<SIZE_T>(region_end - cursor),
                            win_page_prot(slice.prot), &old)) {
                        MLOG("partial-unmap VirtualProtect failed va=0x%llx len=0x%llx error=%lu\n",
                             (unsigned long long)cursor,
                             (unsigned long long)(region_end - cursor), GetLastError());
                        return false;
                    }
                }
                cursor = region_end;
            }
        }
        return true;
    }

    void notify_view_watch_added(const DmemView& view) {
        const auto slices = tracked_protection_slices(
            view.guest_base, view.guest_base + view.guest_size);
        for (const ProtectionSlice& slice : slices) {
            host::guest_write_watch_notify_direct_mapping_added(
                slice.base, slice.size,
                view.phys + (slice.base - view.guest_base), win_page_prot(slice.prot));
        }
    }

    void notify_private_watch_added(const PrivatePlaceholderView& view) {
        const auto slices = tracked_protection_slices(view.base, view.base + view.size);
        for (const ProtectionSlice& slice : slices) {
            host::guest_write_watch_notify_direct_mapping_added(
                slice.base, slice.size, slice.base, win_page_prot(slice.prot));
        }
    }

    bool replace_free_placeholder_with_view_locked(uint64_t base, uint64_t size, uint64_t phys,
                                                   DmemView& out) {
        const PlaceholderApis& apis = placeholder_apis();
        HANDLE section = dmem_section();
        if (!apis.map_view_of_file3 || !section || phys < kDmemBase) return false;
        void* placeholder = take_free_placeholder_locked(base, size, 0x1000);
        if (!placeholder) return false;
        void* mapped = apis.map_view_of_file3(
            section, GetCurrentProcess(), placeholder, phys - kDmemBase,
            static_cast<SIZE_T>(size), MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE, nullptr, 0);
        if (!mapped) {
            remember_free_placeholder_locked(base, size);
            return false;
        }
        const bool sparse = false;
        out = DmemView{base, size, phys, mapped, size, sparse, true};
        if (sparse) {
            const uint64_t pages = (size + 0x3fff) / 0x4000;
            out.committed_pages.resize((pages + 63) / 64);
            out.page_cache_generation = host::guest_mapping_generation();
        }
        return true;
    }

    void invalidate_view_page_cache_locked(const DmemView& original) {
        for (DmemView& view : g_dviews) {
            if (view.view_base != original.view_base ||
                view.guest_base != original.guest_base ||
                view.guest_size != original.guest_size || view.phys != original.phys) continue;
            std::fill(view.committed_pages.begin(), view.committed_pages.end(), 0);
            view.page_cache_generation = 0;
            return;
        }
    }

    bool restore_legacy_view_locked(const DmemView& original) {
        HANDLE section = dmem_section();
        if (!section || original.phys < kDmemBase) return false;
        const uint64_t rel = original.phys - kDmemBase;
        const uint64_t file_off = rel & ~(kWinAllocationGranularity - 1);
        void* mapped = MapViewOfFileEx(
            section, FILE_MAP_ALL_ACCESS,
            static_cast<DWORD>(file_off >> 32), static_cast<DWORD>(file_off & 0xffffffffu),
            static_cast<SIZE_T>(original.view_size), original.view_base);
        if (!mapped || mapped != original.view_base) {
            if (mapped) UnmapViewOfFile(mapped);
            return false;
        }
        if (!ensure_section_pages_committed(original.guest_base, original.guest_size,
                                            HP_R | HP_W) ||
            !restore_view_protections(original)) {
            UnmapViewOfFile(mapped);
            return false;
        }
        return true;
    }

    struct UnmapChange {
        DmemView original;
        std::vector<DmemView> replacements;
    };

    bool rollback_unmap_change_locked(const UnmapChange& change) {
        const PlaceholderApis& apis = placeholder_apis();
        for (auto it = change.replacements.rbegin(); it != change.replacements.rend(); ++it) {
            if (!apis.unmap_view_of_file2 ||
                !apis.unmap_view_of_file2(GetCurrentProcess(), it->view_base,
                                          MEM_PRESERVE_PLACEHOLDER)) return false;
            remember_free_placeholder_locked(it->guest_base, it->guest_size);
        }

        bool restored = false;
        if (change.original.placeholder) {
            DmemView replacement;
            restored = replace_free_placeholder_with_view_locked(
                           change.original.guest_base, change.original.guest_size,
                           change.original.phys, replacement) &&
                       restore_view_protections(replacement);
        } else {
            restored = restore_legacy_view_locked(change.original);
        }
        if (restored) invalidate_view_page_cache_locked(change.original);
        return restored;
    }

    bool split_placeholder_view_locked(const DmemView& old, uint64_t hole_begin,
                                       uint64_t hole_end,
                                       std::vector<DmemView>& replacements) {
        const PlaceholderApis& apis = placeholder_apis();
        if (!old.placeholder || !apis.unmap_view_of_file2 ||
            !apis.unmap_view_of_file2(GetCurrentProcess(), old.view_base,
                                      MEM_PRESERVE_PLACEHOLDER)) return false;
        remember_free_placeholder_locked(old.guest_base, old.guest_size);

        auto rollback = [&]() {
            UnmapChange change{old, replacements};
            const bool restored = rollback_unmap_change_locked(change);
            replacements.clear();
            if (!restored) {
                std::fprintf(stderr,
                             "[memhle] fatal: could not restore direct-memory view after "
                             "partial-unmap failure\n");
                std::abort();
            }
        };

        const uint64_t left_size = hole_begin - old.guest_base;
        const uint64_t right_size = old.guest_base + old.guest_size - hole_end;
        if (left_size) {
            DmemView left;
            if (!replace_free_placeholder_with_view_locked(
                    old.guest_base, left_size, old.phys, left)) {
                rollback();
                return false;
            }
            replacements.push_back(std::move(left));
        }
        if (right_size) {
            DmemView right;
            if (!replace_free_placeholder_with_view_locked(
                    hole_end, right_size, old.phys + (hole_end - old.guest_base), right)) {
                rollback();
                return false;
            }
            replacements.push_back(std::move(right));
        }
        for (const DmemView& replacement : replacements) {
            if (!restore_view_protections(replacement)) {
                rollback();
                return false;
            }
        }
        return true;
    }

    void normalize_spans(std::vector<PlaceholderSpan>& spans) {
        std::sort(spans.begin(), spans.end(),
                  [](const PlaceholderSpan& a, const PlaceholderSpan& b) {
                      return a.base < b.base;
                  });
        size_t out = 0;
        for (const PlaceholderSpan& span : spans) {
            if (!span.size) continue;
            if (!out || spans[out - 1].base + spans[out - 1].size < span.base) {
                spans[out++] = span;
                continue;
            }
            const uint64_t merged_end = std::max(
                spans[out - 1].base + spans[out - 1].size, span.base + span.size);
            spans[out - 1].size = merged_end - spans[out - 1].base;
        }
        spans.resize(out);
    }

    bool span_is_covered(uint64_t begin, uint64_t end,
                         const std::vector<PlaceholderSpan>& covered) {
        if (begin >= end) return false;
        uint64_t cursor = begin;
        for (const PlaceholderSpan& span : covered) {
            if (span.base + span.size <= cursor) continue;
            if (span.base > cursor) break;
            cursor = std::min(end, span.base + span.size);
            if (cursor == end) return true;
        }
        return false;
    }

    bool tracked_mappings_covered_by_spans(uint64_t begin, uint64_t end,
                                           const std::vector<PlaceholderSpan>& covered) {
        std::lock_guard<std::mutex> lk(g_mx);
        for (const Mapping& mapping : g_maps) {
            const uint64_t overlap_begin = std::max(begin, mapping.base);
            const uint64_t overlap_end = std::min(end, mapping.base + mapping.size);
            if (overlap_begin >= overlap_end) continue;
            if (!span_is_covered(overlap_begin, overlap_end, covered)) return false;
        }
        return true;
    }

    // Shared section views must be released with the view APIs. Placeholder-backed views can be
    // split at the guest's 16 KiB boundary: unmap the old view back to a placeholder, replace the
    // untouched prefix/suffix with views of their original physical offsets, and retain the hole as
    // an inaccessible free placeholder for an exact future remap.
    bool win_unmap(uint64_t addr, uint64_t len) {
        if (!addr || !len || addr > UINT64_MAX - len) return false;
        const uint64_t end = addr + len;
        std::vector<DmemView> added;
        bool special_overlap = false;
        {
            std::lock_guard<std::mutex> lk(g_dview_mx);
            struct PrivateUnmapTarget {
                PrivatePlaceholderView view;
                uint64_t cut_base;
                uint64_t cut_size;
                bool whole;
            };
            std::vector<DmemView> targets;
            std::vector<PrivateUnmapTarget> private_targets;
            std::vector<PlaceholderSpan> guest_cuts;
            std::vector<PlaceholderSpan> covered;
            for (const DmemView& view : g_dviews) {
                if (addr < view.guest_base + view.guest_size && end > view.guest_base) {
                    targets.push_back(view);
                    const uint64_t cut_begin = std::max(addr, view.guest_base);
                    const uint64_t cut_end = std::min(end, view.guest_base + view.guest_size);
                    covered.push_back({cut_begin, cut_end - cut_begin});
                    if ((cut_begin != view.guest_base || cut_end != view.guest_base + view.guest_size) &&
                        !view.placeholder) return false;
                }
            }
            for (const PrivatePlaceholderView& view : g_private_placeholder_views) {
                const uint64_t cut_begin = std::max(addr, view.base);
                const uint64_t cut_end = std::min(end, view.base + view.size);
                if (cut_begin >= cut_end) continue;
                const bool whole = cut_begin == view.base &&
                                   cut_end == view.base + view.size;
                // A whole replacement returns to a reusable placeholder. A partial unmap retains
                // the private reservation and decommits only the requested guest pages, matching
                // the former VirtualAlloc path without discarding prefix/suffix contents.
                if (!whole && ((cut_begin & 0x3fff) || ((cut_end - cut_begin) & 0x3fff)))
                    return false;
                private_targets.push_back(
                    {view, cut_begin, cut_end - cut_begin, whole});
                covered.push_back({cut_begin, cut_end - cut_begin});
            }
            for (const PlaceholderSpan& span : g_guest_placeholders) {
                const uint64_t cut_begin = std::max(addr, span.base);
                const uint64_t cut_end = std::min(end, span.base + span.size);
                if (cut_begin >= cut_end) continue;
                guest_cuts.push_back({cut_begin, cut_end - cut_begin});
                covered.push_back({cut_begin, cut_end - cut_begin});
            }
            for (const PlaceholderSpan& span : g_free_placeholders) {
                const uint64_t cut_begin = std::max(addr, span.base);
                const uint64_t cut_end = std::min(end, span.base + span.size);
                if (cut_begin < cut_end)
                    covered.push_back({cut_begin, cut_end - cut_begin});
            }
            std::sort(targets.begin(), targets.end(),
                      [](const DmemView& a, const DmemView& b) {
                          return a.guest_base < b.guest_base;
                      });
            normalize_spans(covered);
            special_overlap = !covered.empty();
            if (special_overlap &&
                (!span_is_covered(addr, end, covered) ||
                 !tracked_mappings_covered_by_spans(addr, end, covered))) return false;
            if (!special_overlap) {
                // No placeholder-owned range participates; retain the legacy private allocation
                // path outside the registry lock.
            } else {

                // Disarm active physical write watches while every old direct alias is still
                // mapped. On failure, the transaction restores views and alias registrations.
                for (const DmemView& target : targets)
                    host::guest_write_watch_notify_direct_mapping_removed(
                        target.guest_base, target.guest_size);

                std::vector<UnmapChange> changes;
                changes.reserve(targets.size());
                auto rollback_views = [&]() {
                    for (auto it = changes.rbegin(); it != changes.rend(); ++it) {
                        if (!rollback_unmap_change_locked(*it)) {
                            std::fprintf(stderr,
                                         "[memhle] fatal: could not roll back direct-memory "
                                         "unmap\n");
                            std::abort();
                        }
                    }
                    for (const DmemView& target : targets) notify_view_watch_added(target);
                };

                for (const DmemView& target : targets) {
                    const uint64_t cut_begin = std::max(addr, target.guest_base);
                    const uint64_t cut_end = std::min(end, target.guest_base + target.guest_size);
                    UnmapChange change{target, {}};
                    bool ok = false;
                    if (cut_begin == target.guest_base &&
                        cut_end == target.guest_base + target.guest_size) {
                        if (target.placeholder) {
                            const PlaceholderApis& apis = placeholder_apis();
                            ok = apis.unmap_view_of_file2 &&
                                 apis.unmap_view_of_file2(GetCurrentProcess(), target.view_base,
                                                         MEM_PRESERVE_PLACEHOLDER);
                            if (ok) remember_free_placeholder_locked(
                                target.guest_base, target.guest_size);
                        } else {
                            ok = UnmapViewOfFile(target.view_base) != 0;
                        }
                    } else {
                        ok = split_placeholder_view_locked(
                            target, cut_begin, cut_end, change.replacements);
                    }
                    if (!ok) {
                        rollback_views();
                        return false;
                    }
                    changes.push_back(std::move(change));
                }

                std::vector<PlaceholderSpan> converted_guest;
                auto rollback_guest = [&]() {
                    for (auto it = converted_guest.rbegin();
                         it != converted_guest.rend(); ++it) {
                        if (!take_free_placeholder_locked(it->base, it->size, 0x1000)) {
                            std::fprintf(stderr,
                                         "[memhle] fatal: could not restore guest placeholder "
                                         "ownership\n");
                            std::abort();
                        }
                        remember_guest_placeholder_locked(it->base, it->size);
                    }
                };
                for (const PlaceholderSpan& cut : guest_cuts) {
                    if (!take_guest_placeholder_locked(cut.base, cut.size, 0x1000)) {
                        rollback_guest();
                        rollback_views();
                        return false;
                    }
                    remember_free_placeholder_locked(cut.base, cut.size);
                    converted_guest.push_back(cut);
                }

                size_t private_unmapped = 0;
                for (const PrivateUnmapTarget& target : private_targets) {
                    // The notification API invalidates every overlapping AliasRange rather than
                    // splitting one. Remove the complete private allocation deliberately, then
                    // rebuild retained tracked slices after a successful partial decommit (or the
                    // complete allocation if the host operation fails).
                    host::guest_write_watch_notify_direct_mapping_removed(
                        target.view.base, target.view.size);
                    const bool released = target.whole
                        ? VirtualFree(
                              reinterpret_cast<void*>(
                                  static_cast<uintptr_t>(target.view.base)),
                              static_cast<SIZE_T>(target.view.size),
                              MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != 0
                        : VirtualFree(
                              reinterpret_cast<void*>(
                                  static_cast<uintptr_t>(target.cut_base)),
                              static_cast<SIZE_T>(target.cut_size), MEM_DECOMMIT) != 0;
                    if (!released) {
                        MLOG("private placeholder release va=0x%llx len=0x%llx failed "
                             "error=%lu\n",
                             (unsigned long long)target.cut_base,
                             (unsigned long long)target.cut_size, GetLastError());
                        if (private_unmapped) {
                            std::fprintf(stderr,
                                         "[memhle] fatal: partial private-placeholder unmap\n");
                            std::abort();
                        }
                        notify_private_watch_added(target.view);
                        rollback_guest();
                        rollback_views();
                        return false;
                    }
                    if (target.whole) {
                        remember_free_placeholder_locked(
                            target.view.base, target.view.size);
                    } else {
                        if (target.view.base < target.cut_base)
                            notify_private_watch_added(
                                {target.view.base, target.cut_base - target.view.base});
                        const uint64_t cut_end = target.cut_base + target.cut_size;
                        const uint64_t view_end = target.view.base + target.view.size;
                        if (cut_end < view_end)
                            notify_private_watch_added({cut_end, view_end - cut_end});
                    }
                    ++private_unmapped;
                }

                for (const UnmapChange& change : changes) {
                    const DmemView& target = change.original;
                    auto it = std::find_if(g_dviews.begin(), g_dviews.end(),
                        [&](const DmemView& view) {
                            return view.view_base == target.view_base &&
                                   view.guest_base == target.guest_base &&
                                   view.guest_size == target.guest_size &&
                                   view.phys == target.phys;
                        });
                    if (it != g_dviews.end()) g_dviews.erase(it);
                    for (const DmemView& replacement : change.replacements) {
                        added.push_back(replacement);
                        g_dviews.push_back(replacement);
                    }
                }
                for (const PrivateUnmapTarget& target : private_targets) {
                    if (!target.whole) continue;
                    auto it = std::find_if(
                        g_private_placeholder_views.begin(),
                        g_private_placeholder_views.end(),
                        [&](const PrivatePlaceholderView& view) {
                            return view.base == target.view.base &&
                                   view.size == target.view.size;
                        });
                    if (it != g_private_placeholder_views.end())
                        g_private_placeholder_views.erase(it);
                }
            }
        }
        if (!special_overlap) {
            host::guest_write_watch_notify_direct_mapping_removed(addr, len);
            return VirtualFree(reinterpret_cast<void*>(static_cast<uintptr_t>(addr)),
                               static_cast<SIZE_T>(len), MEM_DECOMMIT) != 0;
        }
        for (const DmemView& view : added) notify_view_watch_added(view);
        return true;
    }
} // namespace

// Materialize untouched zero pages in a sparse direct-memory view before a host-side read/write.
// Returns 0 for an unrelated, inaccessible, or invalid range so callers retain their normal guard.
extern "C" int prosper_try_commit_dmem(uint64_t addr, uint64_t len, int write) {
    return commit_sparse_dmem(addr, len, write != 0);
}

// Materialize one page of an uncommitted guest reservation on first touch. The legacy VEH path
// still handles ordinary VirtualAlloc reservations; this helper owns only modern placeholders.
extern "C" int prosper_try_commit_reserved_placeholder(uint64_t addr, uint64_t len) {
    if (!len || (addr & 0xfff) || (len & 0xfff) || addr > UINT64_MAX - len) return 0;
    std::lock_guard<std::mutex> lk(g_dview_mx);
    if (private_placeholder_view_containing_locked(addr, addr + len)) return 1;
    void* placeholder = take_guest_placeholder_locked(addr, len, 0x1000);
    if (!placeholder) return 0;
    return replace_placeholder_with_private_locked(
               addr, len, HP_R | HP_W, PlaceholderOwner::Guest) != nullptr;
}

// Fault-handler lazy-commit probe parity (Linux exports this for its SIGSEGV handler). The Windows
// VEH uses this to back reserved guest pages on first touch: 0 = untracked, 1 = reserved,
// 2 = committed, 3 = guest-committed sparse direct page awaiting host commitment.
extern "C" int prosper_reserved_range_state(uint64_t addr) {
    bool tracked = false;
    bool committed = false;
    {
        std::lock_guard<std::mutex> lk(g_mx);
        const Mapping* best = nullptr;
        for (auto& m : g_maps)
            if (addr >= m.base && addr < m.base + m.size)
                if (!best || m.base > best->base) best = &m;
        tracked = best != nullptr;
        committed = best && best->committed;
    }
    if (!tracked) return 0;
    if (!committed) return 1;
    return addr != UINT64_MAX && sparse_dmem_page_uncommitted(addr) ? 3 : 2;
}

// --- Handlers: Win32 versions of the Linux memory HLE, same Sony contracts -------------------

// sceKernelReserveVirtualRange(void** addrInOut, size_t len, int flags, size_t align)
HLE(k_reserve_vrange) {
    uint64_t hint = a0 ? *(uint64_t*)a0 : 0;
    uint64_t align = a3 ? a3 : 0x4000;
    const bool fixed = (a2 & 0x10) != 0;   // SCE_KERNEL_MAP_FIXED
    if (!a1) return 0x80020016ull;         // EINVAL (len 0)
    // Idempotent re-reserve of our own uncommitted range (Linux #115 parity).
    if (hint) {
        bool ours = false;
        { std::lock_guard<std::mutex> lk(g_mx);
          for (auto& m : g_maps)
              // Idempotent only when the WHOLE requested span is contained (mirrors the Linux
              // FIXED branch): a hint-only match let a large re-reserve false-succeed backed by
              // a smaller range at the same base (the 64 MiB metadata pool at the arena's old
              // 0x1000000000 hint after #312's huge-reserve redirect).
              if (!m.committed && hint >= m.base && hint < m.base + m.size &&
                  a1 <= m.base + m.size - hint) { ours = true; break; } }
        if (ours) { if (a0) *(uint64_t*)a0 = hint;
                    MLOG("reserve hint=0x%llx re-reserve-of-own-range -> OK\n", (unsigned long long)hint);
                    return 0; }
    }
    void* p = win_reserve(hint, a1, fixed, align);
    if (!p) { MLOG("reserve hint=0x%llx len=0x%llx FAILED\n",
                   (unsigned long long)hint, (unsigned long long)a1); return 0x8002000cull; } // ENOMEM
    if (a0) *(uint64_t*)a0 = (uint64_t)p;
    track((uint64_t)p, a1, 0, 0, false, "reserved");
    MLOG("reserve -> 0x%llx len=0x%llx flags=0x%llx align=0x%llx\n",
         (unsigned long long)p, (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)align);
    return 0;
}

// sceKernelMapNamedFlexibleMemory(void** addrInOut, size_t len, int prot, int flags, const char* name)
HLE(k_map_flexible) {
    uint64_t hint = a0 ? *(uint64_t*)a0 : 0;
    const bool fixed = (a3 & 0x10) != 0;   // SCE_KERNEL_MAP_FIXED
    if (fixed && !hint) return 0x80020016ull;   // SCE_KERNEL_ERROR_EINVAL
    void* p = hint ? win_commit_flexible_exact(hint, a1, host_prot(a2)) : nullptr;
    if (!p && !fixed)
        p = win_commit_from(hint ? hint : kGuestAutoVaMin, a1, host_prot(a2));
    if (!p) { MLOG("mapflexible hint=0x%llx len=0x%llx FAILED\n",
                   (unsigned long long)hint, (unsigned long long)a1); return 0x8002000cull; }
    if (a0) *(uint64_t*)a0 = (uint64_t)p;
    track((uint64_t)p, a1, host_prot(a2), static_cast<uint32_t>(a2), true,
          a4 ? (const char*)a4 : "flexible", kVirtualQueryFlexible);
    MLOG("mapflexible -> 0x%llx len=0x%llx prot=0x%llx\n",
         (unsigned long long)p, (unsigned long long)a1, (unsigned long long)a2);
    return 0;
}
HLE(k_map_flexible_noname) { return k_map_flexible(a0, a1, a2, a3, 0, 0); }

// sceKernelAvailableFlexibleMemorySize(size_t* sizeOut) — 512 MiB budget (shadPS4 parity).
HLE(k_avail_flexible) { if (!a0) return 0x80020016ull; *(uint64_t*)(uintptr_t)a0 = 512ull * 1024 * 1024; return 0; }

// sceKernelAllocateDirectMemory(start, end, len, align, memType, off_t* physOut)
HLE(k_alloc_dmem) {
    if (!valid_dmem_allocation(a2, a3, a4, a5))
        return 0x80020016ull;   // SCE_KERNEL_ERROR_EINVAL
    uint64_t align = a3 ? a3 : 0x4000;
    uint64_t sz = a2;
    uint64_t off;
    if (!dmem_take(sz, align, (int)a4, off, a0, a1 ? a1 : ~0ull)) {
        MLOG("alloc_dmem len=0x%llx in [0x%llx,0x%llx) -> ENOMEM\n",
             (unsigned long long)a2, (unsigned long long)a0, (unsigned long long)a1);
        return 0x8002000Cull;
    }
    dmem_prepare_allocation(off, sz);
    *(uint64_t*)a5 = off;
    MLOG("alloc_dmem len=0x%llx -> phys=0x%llx\n", (unsigned long long)a2, (unsigned long long)off);
    return 0;
}
// sceKernelAllocateMainDirectMemory(len, align, memType, off_t* physOut) — physOut at arg3.
HLE(k_alloc_main_dmem) {
    if (!valid_dmem_allocation(a0, a1, a2, a3))
        return 0x80020016ull;   // SCE_KERNEL_ERROR_EINVAL
    uint64_t align = a1 ? a1 : 0x4000;
    uint64_t sz = a0;
    uint64_t off;
    if (!dmem_take(sz, align, (int)a2, off)) {
        MLOG("alloc_main_dmem len=0x%llx -> ENOMEM\n", (unsigned long long)a0);
        return 0x8002000Cull;
    }
    dmem_prepare_allocation(off, sz);
    *(uint64_t*)a3 = off;
    MLOG("alloc_main_dmem len=0x%llx -> phys=0x%llx\n", (unsigned long long)a0, (unsigned long long)off);
    return 0;
}

// sceKernelDirectMemoryQuery(off_t offset, int flags, info*, size_t infoSize)
HLE(k_direct_memory_query) {
    if (!a2) return 0x80020016ull;
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
    // EACCES terminates guest enumeration walks — same contract as the Linux variant (#1129).
    if (!r) return 0x8002000dull;
    if (sz >= 0x08) *(uint64_t*)(info + 0x00) = r->start;
    if (sz >= 0x10) *(uint64_t*)(info + 0x08) = r->end;
    if (sz >= 0x14) *(int32_t*)(info + 0x10) = r->type;
    return 0;
}

static uint64_t map_dmem_impl(uint64_t addr_in_out, uint64_t len, uint64_t prot,
                              uint64_t flags, uint64_t phys, uint64_t align,
                              int32_t memory_type, bool set_memory_type) {
    uint64_t hint = addr_in_out ? *(uint64_t*)addr_in_out : 0;
    const bool fixed = (flags & 0x10) != 0;
    void* p = win_map_phys(hint, len, host_prot(prot), phys, align, fixed);
    if (!p) { MLOG("map_dmem hint=0x%llx len=0x%llx FAILED\n",
                   (unsigned long long)hint, (unsigned long long)len); return 0x8002000cull; }
    if (set_memory_type) dmem_retype(phys, len, memory_type);
    if (addr_in_out) *(uint64_t*)addr_in_out = (uint64_t)p;
    const bool sparse = sparse_dmem_view_contains((uint64_t)p, (uint64_t)p + len);
    track((uint64_t)p, len, host_prot(prot), static_cast<uint32_t>(prot), true,
          "direct", kVirtualQueryDirect, phys, memory_type, !sparse);
    MLOG("map_dmem -> 0x%llx len=0x%llx phys=0x%llx prot=0x%llx flags=0x%llx align=0x%llx\n",
         (unsigned long long)p, (unsigned long long)len, (unsigned long long)phys,
         (unsigned long long)prot, (unsigned long long)flags, (unsigned long long)align);
    return 0;
}

// sceKernelMapDirectMemory(void** addrInOut, len, prot, flags, off_t phys, size_t align)
HLE(k_map_dmem) {
    int32_t memory_type = 0;
    dmem_type_at(a4, memory_type);
    return map_dmem_impl(a0, a1, a2, a3, a4, a5, memory_type, false);
}

// sceKernelMapDirectMemory2 inserts `type` before prot/flags/phys/align and applies it to the
// mapped VMA plus the corresponding physical-allocation range.
HLE7(k_map_dmem2) {
    return map_dmem_impl(a0, a1, a3, a4, a5, a6, static_cast<int32_t>(a2), true);
}

HLE(k_munmap) {
    constexpr uint64_t mask = kGuestPageSize - 1;
    if (!a0 || !a1 || (a0 & mask) || (a1 & mask) || a1 > UINT64_MAX - a0)
        return 0x80020016ull;
    if (!win_unmap(a0, a1)) return 0x80020016ull;
    untrack(a0, a1);
    return 0;
}
static uint64_t sce_win_mprotect_error(DWORD error) {
    switch (error) {
        case ERROR_ACCESS_DENIED:
        case ERROR_NOACCESS:          return 0x8002000dull;
        case ERROR_NOT_ENOUGH_MEMORY:
        case ERROR_OUTOFMEMORY:       return 0x8002000cull;
        default:                      return 0x80020016ull;
    }
}
HLE(k_mprotect) {
    if (!a0) return 0x80020016ull;
    const int prot = host_prot(a2);
    DWORD error = ERROR_SUCCESS;
    if (!win_protect(a0, a1, prot, &error)) return sce_win_mprotect_error(error);
    retrack_prot(a0, a1, prot, static_cast<uint32_t>(a2), "mprotect");
    return 0;
}
HLE(k_mtypeprotect) {
    if (!a0) return 0x80020016ull;
    uint64_t base = 0, len = 0;
    if (!normalize_guest_page_range(a0, a1, base, len)) return 0x80020016ull;
    if (!len) return 0;
    const int prot = host_prot(a3);
    DWORD error = ERROR_SUCCESS;
    if (!win_protect(base, len, prot, &error)) return sce_win_mprotect_error(error);
    retrack_prot(base, len, prot, static_cast<uint32_t>(a3), "mtypeprotect");
    retrack_type(base, len, static_cast<int32_t>(a2));
    return 0;
}
HLE(k_dmem_size){ return kDmemTotal; }
// sceKernelAvailableDirectMemorySize(searchStart, searchEnd, align, off_t* physOut, size_t* sizeOut)
HLE(k_avail_dmem) {
    if (!a3 || !a4) return 0x80020016ull;
    uint64_t off = 0, size = 0;
    dmem_largest_free(a0, a1 ? a1 : (kDmemBase + kDmemTotal), a2, off, size);
    if (!size) return 0x8002000Cull;
    *(uint64_t*)(uintptr_t)a3 = off;
    *(uint64_t*)(uintptr_t)a4 = size;
    return 0;
}
HLE(k_release_dmem) {
    MLOG("release_dmem phys=0x%llx len=0x%llx\n",
         (unsigned long long)a0, (unsigned long long)a1);
    dmem_release(a0, a1);
    return 0;
}

// sceKernelBatchMap(entries, num, int* numOut) — entry 0x20 bytes: start@0, phys@8, len@0x10,
// prot@0x18(char), type@0x19, op@0x1c. Ops: 0=MAP_DIRECT 1=UNMAP 2=PROTECT 3=MAP_FLEXIBLE 4=TYPE_PROTECT.
HLE(k_batch_map) {
    const uint8_t* e = (const uint8_t*)(uintptr_t)a0;
    int n = (int)(int64_t)a1, done = 0;
    if (!e || n < 0) return 0x80020016ull;
    uint64_t ret = 0;
    for (int i = 0; i < n; i++, e += 0x20) {
        uint64_t start = *(const uint64_t*)(e + 0x00);
        uint64_t phys  = *(const uint64_t*)(e + 0x08);
        uint64_t len   = *(const uint64_t*)(e + 0x10);
        uint8_t  prot  = e[0x18];
        int32_t  type  = e[0x19];
        int32_t  op    = *(const int32_t*)(e + 0x1c);
        MLOG("bm op=%d va=0x%llx phys=0x%llx len=0x%llx prot=0x%x\n",
             op, (unsigned long long)start, (unsigned long long)phys,
             (unsigned long long)len, prot);
        if (!len) {
            ret = 0x80020016ull;
            break;
        }
        bool ok = true;
        switch (op) {
            case 0: {                               // MAP_DIRECT: shared physical backing
                void* p = win_map_phys(start, len, host_prot(prot), phys, 0, start != 0);
                ok = (p != nullptr);
                if (ok) {
                    const bool sparse = sparse_dmem_view_contains(
                        (uint64_t)p, (uint64_t)p + len);
                    int32_t memory_type = 0;
                    dmem_type_at(phys, memory_type);
                    track((uint64_t)p, len, host_prot(prot), prot, true,
                          "batch-direct", kVirtualQueryDirect, phys, memory_type, !sparse);
                }
                break;
            }
            case 3: {                               // MAP_FLEXIBLE: anonymous/private
                void* p = win_commit(start, len, host_prot(prot));
                ok = (p != nullptr);
                if (ok) track((uint64_t)p, len, host_prot(prot), prot, true,
                              "batch-flex", kVirtualQueryFlexible);
                break;
            }
            case 1: {                               // UNMAP
                ok = !start || win_unmap(start, len);
                if (ok && start) untrack(start, len);
                break;
            }
            case 2: case 4: {                                                            // PROTECT / TYPE_PROTECT
                if (start) {
                    uint64_t protect_start = start, protect_len = len;
                    if (op == 4 && !normalize_guest_page_range(
                                       start, len, protect_start, protect_len)) {
                        ok = false;
                        ret = 0x80020016ull;
                        break;
                    }
                    if (!protect_len) break;
                    DWORD error = ERROR_SUCCESS;
                    ok = win_protect(protect_start, protect_len, host_prot(prot), &error);
                    if (ok) {
                        retrack_prot(protect_start, protect_len, host_prot(prot), prot,
                                     "batch-prot");
                        if (op == 4) retrack_type(protect_start, protect_len, type);
                    } else {
                        ret = sce_win_mprotect_error(error);
                    }
                }
                break;
            }
            default: ok = false; ret = 0x80020016ull; break;
        }
        if (!ok) { if (!ret) ret = 0x8002000Cull; break; }
        done++;
    }
    if (a2) *(int32_t*)(uintptr_t)a2 = done;
    MLOG("batchmap n=%d done=%d -> 0x%llx\n", n, done, (unsigned long long)ret);
    return ret;
}

// sceKernelVirtualQuery(addr, flags, info*, infoSize): 0x00 start; 0x08 end; 0x10
// physical offset; 0x18 protection; 0x1c memory type; 0x20 u8 classification; 0x21 name[32].
HLE(k_virtual_query) {
    if (!a2) return 0x80020016ull;
    uint8_t* info = (uint8_t*)a2;
    uint64_t sz = a3 ? (a3 > 0x41 ? 0x41 : a3) : 0x41;
    memset(info, 0, sz);
    Mapping mapping{};
    const bool found = snapshot_mapping(a0, mapping);
    uint64_t start, end, offset = 0; int prot, memory_type = 0; uint32_t flags;
    if (found) {
        start = mapping.base; end = mapping.base + mapping.size;
        // CPU read/write is an SCE enum (RW is 0x2, not host READ|WRITE 0x3), and the GPU bits
        // have no host-page equivalent. Return the original guest protection value verbatim.
        prot = mapping.committed ? static_cast<int>(mapping.guest_prot) : 0x0;
        offset = (mapping.query_flags & kVirtualQueryDirect) ? mapping.offset : 0;
        memory_type = (mapping.query_flags & kVirtualQueryDirect) ? mapping.memory_type : 0;
        flags = mapping.query_flags |
                (mapping.committed ? kVirtualQueryCommitted : 0);
    } else {
        uint64_t nb = next_base(a0);
        if (!nb) return 0x8002000eull;
        start = a0 & ~(uint64_t)0x3fff; end = nb; prot = 0; flags = 0;
    }
    if (sz >= 0x08) *(uint64_t*)(info + 0x00) = start;
    if (sz >= 0x10) *(uint64_t*)(info + 0x08) = end;
    if (sz >= 0x18) *(uint64_t*)(info + 0x10) = offset;
    if (sz >= 0x1c) *(int32_t*)(info + 0x18) = prot;
    if (sz >= 0x20) *(int32_t*)(info + 0x1c) = memory_type;
    if (sz >= 0x21) info[0x20] = static_cast<uint8_t>(flags);
    if (sz >= 0x41 && found && mapping.name[0]) memcpy(info + 0x21, mapping.name, 32);
    return 0;
}

// Physical-allocation query counterpart to sceKernelDirectMemoryQuery. Keep the implementation
// identical to the POSIX path so a missing range and required-output validation have one contract.
HLE(k_get_direct_memory_type) {
    if (!a1 || !a2 || !a3) return 0x80020016ull;   // SCE_KERNEL_ERROR_EINVAL
    DMem result{};
    {
        std::lock_guard<std::mutex> lk(g_dmx);
        bool found = false;
        for (const auto& d : g_dmem) {
            if (a0 >= d.start && a0 < d.end) {
                result = d;
                found = true;
                break;
            }
        }
        if (!found) return 0x80020002ull;          // SCE_KERNEL_ERROR_ENOENT
    }
    *(int32_t*)(uintptr_t)a1 = result.type;
    *(uint64_t*)(uintptr_t)a2 = result.start;
    *(uint64_t*)(uintptr_t)a3 = result.end;
    return 0;
}

HLE(k_query_memory_protection) {
    Mapping mapping{};
    if (!snapshot_mapping(a0, mapping)) return 0x8002000dull;
    if (a1) *(uint64_t*)(uintptr_t)a1 = mapping.base;
    if (a2) *(uint64_t*)(uintptr_t)a2 = mapping.base + mapping.size;
    if (a3) *(uint32_t*)(uintptr_t)a3 = mapping.guest_prot;
    return 0;
}

// libSceAmpr / APR command-buffer trio + teardown. These commit UE4 (PPSA17942, area:ue4)
// allocator pages; that title does not boot on Windows yet, so no-op stubs are correct here
// (the Linux path models them). Registered by raw NID for parity.
HLE(k_ampr_ok) { return 0; }

// APR command-buffer WriteAddress (NID j0+3uJMxYJY) — the completion-notification write (#1149).
// Windows sibling of the POSIX handler above (full contract documented there): write `value` (a2)
// verbatim to `*address` (a1). Fault-safe via VirtualQuery — an uncommitted / non-writable / partially
// mapped target is skipped rather than faulting the emulator (mirrors windows_prepare_guest_write).
HLE(k_ampr_write_address) {   // j0+3uJMxYJY (cb, address, value, flags)
    (void)a0; (void)a3;
    if (a1) {
        MEMORY_BASIC_INFORMATION mbi{};
        constexpr DWORD kWritable = PAGE_READWRITE | PAGE_WRITECOPY |
                                    PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
        if (VirtualQuery(reinterpret_cast<void*>(static_cast<uintptr_t>(a1)), &mbi, sizeof(mbi)) &&
            mbi.State == MEM_COMMIT && (mbi.Protect & kWritable) &&
            static_cast<uintptr_t>(a1) + sizeof(uint64_t) <=
                reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize) {
            *reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(a1)) = a2;
        }
    }
    return 0;
}

namespace {
std::mutex g_sync_mx;
std::condition_variable g_sync_cv;
bool wsynclog() { static int v = getenv("PROSPER_SYNCLOG") ? 1 : 0; return v; }
// Best-effort guest call site of the current HLE call. The Windows import stub `call`s the handler
// (no frame pointer), so an [rbp+8] walk misses the guest return address; instead scan our stack for
// the first value in the guest code range (eboot..libc = [0x400000000, 0x700000000)). The guest's
// `call <stub>` pushed its return address just below, so the first in-range hit is the caller. Coarse
// (a stack data word could coincidentally look like a guest PC), but good enough to name a wait site.
uint64_t sync_guest_caller() {
    uint64_t* sp = (uint64_t*)__builtin_frame_address(0);
    for (int i = 0; i < 160; i++) {
        uint64_t v = sp[i];
        if (v < 0x400000000ull || v >= 0x600000000ull) continue;   // guest MODULES only (exclude 0x6.. import stubs)
        if ((v & 0xfff) < 8) continue;                             // keep the 6-byte look-back on v's page
        // A stack word in the guest range may be data, not a return address, and could point at an
        // UNMAPPED gap between modules — so confirm the page is committed+readable before dereferencing
        // (this scan runs on every logged wait; an unguarded read faulted intermittently).
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery((LPCVOID)(uintptr_t)v, &mbi, sizeof mbi) || mbi.State != MEM_COMMIT) continue;
        const DWORD rd = PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE
                       | PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY;
        if (!(mbi.Protect & rd) || (mbi.Protect & PAGE_GUARD)) continue;
        const uint8_t* p = (const uint8_t*)(uintptr_t)v;
        // Accept only a real return address: the preceding instruction must be a CALL.
        if (p[-5] == 0xE8) return v;                                        // call rel32
        if (p[-2] == 0xFF && ((p[-1] >> 3) & 7) == 2) return v;             // call r/m (2-byte)
        if (p[-3] == 0xFF && ((p[-2] >> 3) & 7) == 2) return v;             // call r/m (3-byte, modrm+sib/disp8)
        if (p[-6] == 0xFF && ((p[-5] >> 3) & 7) == 2) return v;             // call r/m (rip-rel disp32)
    }
    return 0;
}
}

HLE(k_wait_on_address) {
    if (!a0) return 0;
    auto& raw = *(uint32_t*)(uintptr_t)a0;
    std::atomic_ref<uint32_t> addr(raw);
    uint32_t expected = (uint32_t)a1;
    if (wsynclog()) fprintf(stderr, "[sync] T%lu WAIT.enter addr=0x%llx *addr=0x%x exp=0x%x timo_ptr=0x%llx caller=0x%llx\n",
                            (unsigned long)GetCurrentThreadId(),
                            (unsigned long long)a0, (unsigned)addr.load(std::memory_order_acquire),
                            (unsigned)expected, (unsigned long long)a2, (unsigned long long)sync_guest_caller());
    // Back this with the NATIVE Win32 futex (WaitOnAddress/WakeByAddress*, Win8+): a true PER-ADDRESS
    // wait, exactly like the Linux FUTEX_WAIT path. The previous global-`condition_variable` shared one
    // cv across every waited address, so WakeByAddress(n=1) -> notify_one() could wake a waiter parked
    // on a DIFFERENT address (it re-checks its own word, finds it unchanged, re-sleeps) while the
    // intended waiter was never woken — a lost wakeup that made the boot nondeterministically deadlock.
    // Native WaitOnAddress blocks while *addr == compare and is woken only by WakeByAddress* on THIS
    // addr, so n=1 semantics are correct and there is no thundering herd. The 32-bit compare is a
    // shared limitation with the Linux path (WaitOnAddress supports 1/2/4/8-byte; only 4 is modeled).
    static const bool honor_timeout = getenv("PROSPER_NO_WAIT_TIMEOUT") == nullptr;
    ULONGLONG deadline = 0;   // 0 = infinite
    if (a2 && honor_timeout) {
        uint32_t us = *(volatile uint32_t*)(uintptr_t)a2;
        // Widen before rounding. UINT32_MAX is a common "effectively infinite" bounded wait;
        // adding 999 in uint32_t wrapped it to 998 and turned a ~71-minute wait into 1 ms.
        ULONGLONG ms = us == 0 ? 1 : ((ULONGLONG)us + 999) / 1000;   // us -> ms, round up; 0 == poll
        deadline = GetTickCount64() + ms;
    }
    volatile uint32_t* wa = (volatile uint32_t*)(uintptr_t)a0;
    // Register as a futex waiter so the GPU command processor's RELEASE_MEM/EOP wake (wake_label_waiters,
    // which only fires when g_waiters>0) reaches this thread. RAII so every return path unregisters.
    WaitRegistration registration = futex_wait_enter(a0);
    struct WaiterGuard {
        WaitRegistration registration;
        ~WaiterGuard() { futex_wait_exit(registration); }
    } _waiter_guard{registration};
    while (*wa == expected) {
        DWORD wait_ms = INFINITE;
        if (deadline) {
            ULONGLONG now = GetTickCount64();
            if (now >= deadline) {
                if (wsynclog()) fprintf(stderr, "[sync] WAIT.exit  addr=0x%llx TIMEOUT\n", (unsigned long long)a0);
                return 0x80020060ull;   // SCE_KERNEL_ERROR_ETIMEDOUT
            }
            wait_ms = (DWORD)(deadline - now);
        }
        uint32_t cmp = expected;
        const bool woke = WaitOnAddress(wa, &cmp, sizeof(cmp), wait_ms) != FALSE;
        const DWORD wait_error = woke ? ERROR_SUCCESS : GetLastError();
        // A cooperative guest exception wakes this address without first changing the futex word.
        // Accept the queued handler at the native-wait boundary before re-checking the predicate;
        // otherwise the thread immediately parks again and can never run the handler that releases it.
        dispatch_pending_guest_exception();
        if (!woke && wait_error == ERROR_TIMEOUT) {
            if (*wa == expected) {
                if (wsynclog()) fprintf(stderr, "[sync] WAIT.exit  addr=0x%llx TIMEOUT\n", (unsigned long long)a0);
                return 0x80020060ull;
            }
        }
        // else: woken or spurious — loop re-checks *addr
    }
    if (wsynclog()) fprintf(stderr, "[sync] WAIT.exit  addr=0x%llx woke\n", (unsigned long long)a0);
    return 0;
}

HLE(k_wake_by_address) {
    if (wsynclog()) fprintf(stderr, "[sync] T%lu WAKE       addr=0x%llx *addr=0x%x n=%lld caller=0x%llx\n",
                            (unsigned long)GetCurrentThreadId(),
                            (unsigned long long)a0, a0 ? *(uint32_t*)(uintptr_t)a0 : 0, (long long)a1,
                            (unsigned long long)sync_guest_caller());
    if (!a0) return 0;
    // Native per-address wake: n==1 wakes ONE waiter on THIS address (correct semaphore-release
    // semantics), otherwise wake all. No lost wakeup and no global thundering herd (unlike the old
    // single shared condition_variable).
    if (a1 == 1) WakeByAddressSingle((PVOID)(uintptr_t)a0);
    else         WakeByAddressAll((PVOID)(uintptr_t)a0);
    return 0;
}

void register_kernel_mem_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceKernelReserveVirtualRange", k_reserve_vrange);
    R("sceKernelMapNamedFlexibleMemory", k_map_flexible);
    R("sceKernelMapFlexibleMemory", k_map_flexible_noname);
    Hle::register_fn("4h6F1LLbTiw", (HleFn)k_map_flexible_noname, "sceKernelMapFlexibleMemoryInternal");
    R("sceKernelAvailableFlexibleMemorySize", k_avail_flexible);
    R("sceKernelAllocateDirectMemory", k_alloc_dmem);
    R("sceKernelAllocateMainDirectMemory", k_alloc_main_dmem);
    R("sceKernelMapDirectMemory", k_map_dmem);
    R("sceKernelMapDirectMemory2", k_map_dmem2);
    R("sceKernelMapNamedDirectMemory", k_map_dmem);
    R("sceKernelMunmap", k_munmap);
    R("sceKernelReleaseFlexibleMemory", k_munmap);
    R("sceKernelMprotect", k_mprotect);
    R("sceKernelMtypeprotect", k_mtypeprotect);
    R("sceKernelReleaseDirectMemory", k_release_dmem);
    R("sceKernelCheckedReleaseDirectMemory", k_release_dmem);
    R("sceKernelBatchMap", k_batch_map);
    R("sceKernelBatchMap2", k_batch_map);
    R("sceKernelVirtualQuery", k_virtual_query);
    R("sceKernelQueryMemoryProtection", k_query_memory_protection);
    R("sceKernelDirectMemoryQuery", k_direct_memory_query);
    R("sceKernelGetDirectMemoryType", k_get_direct_memory_type);
    R("sceKernelGetDirectMemorySize", k_dmem_size);
    R("sceKernelAvailableDirectMemorySize", k_avail_dmem);
    #undef R
    Hle::register_fn("Hc4CaR6JBL0", (HleFn)k_wait_on_address, "sceKernelWaitOnAddress?");
    Hle::register_fn("q2y-wDIVWZA", (HleFn)k_wake_by_address, "sceKernelWakeByAddress?");
    // libSceAmpr / APR command-buffer trio + teardown — no-op stubs on Windows (area:ue4).
    Hle::register_fn("8aI7R7WaOlc", (HleFn)k_ampr_ok, "sceAmprCommandBufferConstructor");
    Hle::register_fn("a8uLzYY--tM", (HleFn)k_ampr_ok, "sceAmprAprCommandBufferConstructor");
    Hle::register_fn("N-FSPA4S3nI", (HleFn)k_ampr_ok, "sceAmprCommandBufferSetBuffer");
    Hle::register_fn("baQO9ez2gL4", (HleFn)k_ampr_ok, "sceAmprCommandBufferReset");
    Hle::register_fn("ULvXMDz56po", (HleFn)k_ampr_ok, "sceAmprCommandBufferClearBuffer");
    Hle::register_fn("tZDDEo2tE5k", (HleFn)k_ampr_ok, "sceAmprCommandBufferGetSize");
    Hle::register_fn("Qs1xtplKo0U", (HleFn)k_ampr_ok, "sceAmprAprCommandBufferDestructor");
    Hle::register_fn("GuchCTefuZw", (HleFn)k_ampr_ok, "sceAmprCommandBufferDestructor");
    Hle::register_fn("sSAUCCU1dv4", (HleFn)k_ampr_ok, "AprSetEventQueue?");
    Hle::register_fn("H896Pt-yB4I", (HleFn)k_ampr_ok, "AprCbSetEventQueue?");
    Hle::register_fn("ASoW5WE-UPo", (HleFn)k_ampr_ok, "AprSubmitCommandBuffer?");
    Hle::register_fn("GnxKOHEawhk", (HleFn)k_ampr_ok, "AmprUnknown3");
    // APR completion-notification write (#1149) — real behavior on Windows too (see handler above).
    Hle::register_fn("j0+3uJMxYJY", (HleFn)k_ampr_write_address, "sceAmprCommandBufferWriteAddress?");
}

} // namespace prosper
#endif
