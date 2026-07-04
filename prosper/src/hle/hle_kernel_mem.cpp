// hle_kernel_mem.cpp — HLE of libkernel virtual/direct memory (Linux backing).
// PS5 memory model: reserve a virtual range, allocate "direct" (physical) memory as
// an opaque offset, then map it (or flexible memory) into VA. We back it with host
// mmap and TRACK every mapping so VirtualQuery is truthful and so we can log/debug the
// guest's address-space construction. Guarded to Linux.
#include "dispatch.hpp"
#include "nid.hpp"

#ifdef __linux__
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <unistd.h>
#include <climits>
#include <cstdint>
#include <cstdio>
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
    std::atomic<uint64_t> g_dmem_off{0x10000000};   // "physical" bump allocator
    // Direct ("physical") memory allocations, for sceKernelDirectMemoryQuery.
    struct DMem { uint64_t start, end; int type; };
    std::mutex g_dmx;
    std::vector<DMem> g_dmem;

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
}

// sceKernelReserveVirtualRange(void** addrInOut, size_t len, int flags, size_t align)
// MUST return an address aligned to `align`: guest allocators round the returned base
// down to their requested alignment and would otherwise land below the mapping.
HLE(k_reserve_vrange) {
    uint64_t hint = a0 ? *(uint64_t*)a0 : 0;
    uint64_t align = a3 ? a3 : 0x4000;
    if (hint) {
        void* p = mmap((void*)hint, a1, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == MAP_FAILED) { MLOG("reserve hint=0x%llx FAILED\n", (unsigned long long)hint); return 0x16; }
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
    uint64_t off = align_up(g_dmem_off.fetch_add(sz), align);
    { std::lock_guard<std::mutex> lk(g_dmx); g_dmem.push_back({ off, off + sz, (int)a4 }); }
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
    uint64_t off = align_up(g_dmem_off.fetch_add(sz), align);
    { std::lock_guard<std::mutex> lk(g_dmx); g_dmem.push_back({ off, off + sz, (int)a2 }); }
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
    void* p = map_at(hint, a1, host_prot(a2));
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
        start = m->base; end = m->base + m->size; prot = 0x3; flags = 0x10; how = "tracked";
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
    if (synclog()) fprintf(stderr, "[sync] T%ld WAIT.enter  addr=0x%llx *addr=0x%x exp=0x%llx\n",
                           (long)syscall(SYS_gettid), (unsigned long long)a0, *(uint32_t*)a0, (unsigned long long)a1);
    long r = syscall(SYS_futex, (uint32_t*)a0, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, (uint32_t)a1, nullptr, nullptr, 0);
    if (synclog()) fprintf(stderr, "[sync] T%ld WAIT.exit   addr=0x%llx r=%ld\n",
                           (long)syscall(SYS_gettid), (unsigned long long)a0, r);
    return 0;
}
HLE(k_wake_by_address) {
    if (!a0) return 0;
    int n = a1 ? (int)a1 : INT_MAX;
    long w = syscall(SYS_futex, (uint32_t*)a0, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, n, nullptr, nullptr, 0);
    if (synclog()) fprintf(stderr, "[sync] T%ld WAKE       addr=0x%llx n=%d woke=%ld\n",
                           (long)syscall(SYS_gettid), (unsigned long long)a0, n, w);
    return 0;
}

HLE(k_munmap)   { if (a0) munmap((void*)a0, a1); return 0; }
HLE(k_mprotect) { if (a0) mprotect((void*)a0, a1, host_prot(a2)); return 0; }
HLE(k_dmem_size){ return 8ull * 1024 * 1024 * 1024; }   // 8 GiB pool

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
    R("sceKernelVirtualQuery", k_virtual_query);
    R("sceKernelDirectMemoryQuery", k_direct_memory_query);
    R("sceKernelGetDirectMemorySize", k_dmem_size);
    R("sceKernelAvailableDirectMemorySize", k_dmem_size);
    #undef R
    // sync_on_address futex — registered by raw NID (names not in any public DB yet).
    Hle::register_fn("Hc4CaR6JBL0", (HleFn)k_wait_on_address, "sceKernelWaitOnAddress?");
    Hle::register_fn("q2y-wDIVWZA", (HleFn)k_wake_by_address, "sceKernelWakeByAddress?");
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
