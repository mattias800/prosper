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

    // Claim `sz` bytes of direct memory at `align` (first-fit over the pool's free gaps). False (no
    // state change) when nothing fits — callers translate that to SCE_KERNEL_ERROR_ENOMEM
    // (0x8002000C, Kyty Errno.h). First-fit (not bump) because guests genuinely RELEASE ranges:
    // UE4 (PPSA17942) probes the pool full with halving allocations, releases the probes, then
    // allocates its real pools — a bump cursor would be permanently exhausted.
    bool dmem_take(uint64_t sz, uint64_t align, int type, uint64_t& off_out) {
        if (!align) align = 0x4000;
        std::lock_guard<std::mutex> lk(g_dmx);
        uint64_t cursor = kDmemBase;
        size_t insert_at = 0;
        for (size_t i = 0; i <= g_dmem.size(); i++) {
            uint64_t gap_end = (i < g_dmem.size()) ? g_dmem[i].start : kDmemBase + kDmemTotal;
            uint64_t off = (cursor + align - 1) & ~(align - 1);
            if (off + sz <= gap_end) { insert_at = i; off_out = off; goto found; }
            if (i < g_dmem.size()) cursor = g_dmem[i].end;
        }
        return false;
    found:
        g_dmem.insert(g_dmem.begin() + insert_at, { off_out, off_out + sz, type });
        return true;
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
    const Mapping* find(uint64_t addr) {
        std::lock_guard<std::mutex> lk(g_mx);
        const Mapping* best = nullptr;
        for (auto& m : g_maps)
            if (addr >= m.base && addr < m.base + m.size)
                if (!best || m.base > best->base) best = &m;   // most specific (latest overlay)
        return best;
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
        int flags = MAP_PRIVATE | MAP_ANONYMOUS | (hint ? MAP_FIXED : 0);
        void* p = mmap((void*)hint, len, prot, flags, -1, 0);
        return p == MAP_FAILED ? nullptr : p;
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
    // Map `len` bytes of the phys pool at `hint` (0 = anywhere). Falls back to anonymous memory
    // if the memfd is unavailable (still boots; loses aliasing).
    void* map_phys_at(uint64_t hint, uint64_t len, int prot, uint64_t phys) {
        int fd = dmem_fd();
        if (fd >= 0 && phys < kDmemBase + kDmemTotal) {
            void* p = mmap((void*)hint, len, prot, MAP_SHARED | (hint ? MAP_FIXED : 0), fd, (off_t)phys);
            if (p != MAP_FAILED) return p;
        }
        return map_at(hint, len, prot);
    }
}

// sceKernelReserveVirtualRange(void** addrInOut, size_t len, int flags, size_t align)
// MUST return an address aligned to `align`: guest allocators round the returned base
// down to their requested alignment and would otherwise land below the mapping.
HLE(k_reserve_vrange) {
    uint64_t hint = a0 ? *(uint64_t*)a0 : 0;
    uint64_t align = a3 ? a3 : 0x4000;
    if (hint) {
        void* p = mmap((void*)hint, a1, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == MAP_FAILED) {
            // The hint region is already mapped. If it's ALREADY one of OUR reservations (uncommitted),
            // the guest is re-reserving its own fixed range — PS5's reserve is idempotent for the caller's
            // own range (the game's allocator no-hint-reserves, records the base, then re-claims it with a
            // fixed hint). Returning an error here made the guest allocator hand back NULL, which the level1
            // asset-load FileCacher then memcpy'd from (crash: memcpy(dst, NULL, n) during deserialization).
            // Treat a re-reserve of our own reservation as success. CONFIDENCE: HIGH (matches the observed
            // no-hint@line23 -> fixed-hint@line143 collision that precedes the loader-thread SIGSEGV).
            bool ours = false;
            { std::lock_guard<std::mutex> lk(g_mx);
              for (auto& m : g_maps)
                  if (!m.committed && hint >= m.base && hint < m.base + m.size) { ours = true; break; } }
            if (ours) {
                if (a0) *(uint64_t*)a0 = hint;
                MLOG("reserve hint=0x%llx re-reserve-of-own-range -> OK\n", (unsigned long long)hint);
                return 0;
            }
            MLOG("reserve hint=0x%llx FAILED\n", (unsigned long long)hint); return 0x16;
        }
        if (a0) *(uint64_t*)a0 = (uint64_t)p;
        track((uint64_t)p, a1, 0, false, "reserved");
        MLOG("reserve(hint) -> 0x%llx len=0x%llx align=0x%llx\n", (unsigned long long)p, (unsigned long long)a1, (unsigned long long)align);
        return 0;
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
HLE(k_alloc_dmem) {
    uint64_t align = a3 ? a3 : 0x4000;
    uint64_t sz = align_up(a2, align);
    uint64_t off;
    if (!dmem_take(sz, align, (int)a4, off)) {
        MLOG("alloc_dmem len=0x%llx -> ENOMEM (pool exhausted)\n", (unsigned long long)a2);
        return 0x8002000Cull;   // SCE_KERNEL_ERROR_ENOMEM
    }
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
    // NOTE: honoring the timeout is gated behind PROSPER_WAIT_TIMEOUT for now. It is the correct
    // behavior (we were ignoring a timeout the guest passes → bounded semaphore-waits blocked forever),
    // and it demonstrably changes execution — but it makes the guest take its *timeout* branch, which
    // throws a C++ exception whose unwind then crashes because sceKernelGetModuleInfoForUnwind is
    // unimplemented (libunwind reads garbage eh_frame → "Unsupported .eh_frame_hdr version" → stack
    // smash). Until the unwinder is fed real eh_frame info, default to infinite waits (stable) and let
    // the flag exercise the exception path during bring-up. See docs/RENDER_LOOP.md.
    static const bool honor_timeout = getenv("PROSPER_WAIT_TIMEOUT") != nullptr;
    if (a2 && honor_timeout) {
        uint32_t us = *(volatile uint32_t*)(uintptr_t)a2;
        // FUTEX_WAIT's timeout is a RELATIVE duration; guard against absurd values.
        if (us == 0) us = 1;                 // 0 == "poll" — a minimal wait keeps us re-checking
        if (us > 5000000u) us = 5000000u;    // cap 5s so a huge/garbage value can't hang the thread
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

HLE(k_munmap)   { if (a0) munmap((void*)a0, a1); return 0; }
HLE(k_mprotect) { if (a0) mprotect((void*)a0, a1, host_prot(a2)); return 0; }
HLE(k_dmem_size){ return kDmemTotal; }   // 8 GiB pool (allocation failures enforce this bound)

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
HLE(k_ampr_push_map) {
    if (a1 && a2) {
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
            void* q = map_phys_at(mirror, a2, PROT_READ | PROT_WRITE, phys);
            if (q) track((uint64_t)q, a2, PROT_READ | PROT_WRITE, true, "ampr-mirror");
            MLOG("ampr push-map va=0x%llx len=0x%llx phys=0x%llx mirror=0x%llx -> %s/%s\n",
                 (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)phys,
                 (unsigned long long)mirror, p ? "ok" : "FAIL", q ? "ok" : "FAIL");
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
            case 1: if (start) munmap((void*)(uintptr_t)start, len); break;              // UNMAP
            case 2: case 4:                                                              // PROTECT / TYPE_PROTECT
                if (start) mprotect((void*)(uintptr_t)start, len, host_prot(prot)); break;
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
    R("sceKernelAvailableDirectMemorySize", k_dmem_size);
    #undef R
    // sync_on_address futex — registered by raw NID (names not in any public DB yet).
    Hle::register_fn("Hc4CaR6JBL0", (HleFn)k_wait_on_address, "sceKernelWaitOnAddress?");
    Hle::register_fn("q2y-wDIVWZA", (HleFn)k_wake_by_address, "sceKernelWakeByAddress?");
    // libSceAmpr command-buffer trio (raw NIDs; see the block comment above k_ampr_push_map).
    Hle::register_fn("8aI7R7WaOlc", (HleFn)k_ampr_ok,       "sceAmprCommandBufferInit?");
    Hle::register_fn("a8uLzYY--tM", (HleFn)k_ampr_ok,       "sceAmprCommandBufferBegin?");
    Hle::register_fn("N-FSPA4S3nI", (HleFn)k_ampr_push_map, "sceAmprPushMapPages?");
}

} // namespace prosper

#else
#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstdint>
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
    std::unique_lock<std::mutex> lk(g_sync_mx);
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
