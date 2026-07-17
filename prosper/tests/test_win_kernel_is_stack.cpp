#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include "../src/host/exec_image.hpp"
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <windows.h>
#include <pthread.h>
#include <cstdint>
#include <cstdio>
#include <thread>

using namespace prosper;

struct WorkerStackProbe {
    volatile LONG ready = 0;
    volatile LONG release = 0;
    void* base = nullptr;
    size_t size = 0;
};

struct GuestStackProbe {
    bool registered = false;
    bool fully_committed = false;
    size_t registered_size = 0;
};

// k_pthread_create enters guest functions with the SysV ABI even in the native Windows build.
// Inspect the resulting reservation from inside that exact trampoline, after its PS5 stack setup.
static void* __attribute__((sysv_abi)) prepared_guest_worker(void* raw) {
    auto* probe = static_cast<GuestStackProbe*>(raw);
    void* registered_base = nullptr;
    probe->registered = guest_stack_for_current_thread(&registered_base, &probe->registered_size);

    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    MEMORY_BASIC_INFORMATION top{};
    if (!high || !VirtualQuery((void*)(high - 1), &top, sizeof(top)) || !top.AllocationBase)
        return nullptr;

    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const uintptr_t page = si.dwPageSize ? si.dwPageSize : 0x1000;
    uintptr_t cursor = (uintptr_t)top.AllocationBase + page; // bottom page remains inaccessible
    bool committed = cursor < high;
    while (committed && cursor < high) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery((void*)cursor, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT) {
            committed = false;
            break;
        }
        uintptr_t next = (uintptr_t)mbi.BaseAddress + mbi.RegionSize;
        if (next <= cursor) { committed = false; break; }
        cursor = next;
    }
    probe->fully_committed = committed;
    return nullptr;
}

static void* registered_worker(void* raw) {
    auto* probe = static_cast<WorkerStackProbe*>(raw);
    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    probe->base = (void*)low;
    probe->size = (size_t)(high - low);
    register_thread_stack((uint64_t)GetCurrentThreadId(), probe->base, probe->size);
    InterlockedExchange(&probe->ready, 1);
    while (!probe->release) Sleep(1);
    unregister_thread_stack((uint64_t)GetCurrentThreadId());
    return nullptr;
}

int main() {
    register_builtin_hle();
    auto is_stack = Hle::lookup(nid_hash("sceKernelIsStack"));
    auto attr_init = Hle::lookup(nid_hash("scePthreadAttrInit"));
    auto attr_get = Hle::lookup(nid_hash("scePthreadAttrGet"));
    auto attr_getaddr = Hle::lookup(nid_hash("scePthreadAttrGetstackaddr"));
    auto attr_getsize = Hle::lookup(nid_hash("scePthreadAttrGetstacksize"));
    auto attr_setsize = Hle::lookup(nid_hash("scePthreadAttrSetstacksize"));
    auto attr_destroy = Hle::lookup(nid_hash("scePthreadAttrDestroy"));
    auto thread_create = Hle::lookup(nid_hash("scePthreadCreate"));
    auto thread_join = Hle::lookup(nid_hash("scePthreadJoin"));
    if (!is_stack || !attr_init || !attr_get || !attr_getaddr || !attr_getsize || !attr_setsize ||
        !attr_destroy || !thread_create || !thread_join) {
        std::fprintf(stderr, "required stack HLE is not registered\n");
        return 1;
    }

    uint8_t local = 0;
    uint64_t addr = (uint64_t)(uintptr_t)&local;
    uint64_t base = addr - 0x1000;
    register_thread_stack((uint64_t)GetCurrentThreadId(), (void*)(uintptr_t)base, 0x2000);

    bool inside = is_stack(addr, 0, 0, 0, 0, 0) == 1;
    bool below = is_stack(base - 1, 0, 0, 0, 0, 0) == 0;
    bool at_end = is_stack(base + 0x2000, 0, 0, 0, 0, 0) == 0;

    void* attr = nullptr;
    void* reported_base = nullptr;
    size_t reported_size = 0;
    attr_init((uint64_t)(uintptr_t)&attr, 0, 0, 0, 0, 0);
    attr_get((uint64_t)pthread_self(), (uint64_t)(uintptr_t)&attr, 0, 0, 0, 0);
    attr_getaddr((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&reported_base, 0, 0, 0, 0);
    attr_getsize((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&reported_size, 0, 0, 0, 0);
    bool attr_bounds = reported_base == (void*)(uintptr_t)base && reported_size == 0x2000;

    // The guest passes a winpthreads pthread_t handle for another thread, while Windows stack
    // registration is keyed by GetCurrentThreadId(). The registry must bridge those identities.
    WorkerStackProbe probe;
    pthread_t worker = 0;
    bool worker_created = pthread_create(&worker, nullptr, registered_worker, &probe) == 0;
    for (int i = 0; worker_created && i < 2000 && !probe.ready; ++i) Sleep(1);
    void* resolved_base = nullptr;
    size_t resolved_size = 0;
    bool resolved_worker = worker_created && probe.ready &&
        guest_stack_for_thread((uint64_t)worker, &resolved_base, &resolved_size) &&
        resolved_base == probe.base && resolved_size == probe.size;

    reported_base = nullptr;
    reported_size = 0;
    attr_get((uint64_t)worker, (uint64_t)(uintptr_t)&attr, 0, 0, 0, 0);
    attr_getaddr((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&reported_base, 0, 0, 0, 0);
    attr_getsize((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&reported_size, 0, 0, 0, 0);
    bool target_attr_bounds = reported_base == probe.base && reported_size == probe.size;

    InterlockedExchange(&probe.release, 1);
    if (worker_created) pthread_join(worker, nullptr);

    // screenshot and prosper-app initialize guest modules from a frontend-owned std::thread.
    // Verify its implicit pthread handle resolves through the native-ID registration and that the
    // thread-local owner removes the record when the frontend thread exits.
    bool std_thread_self_resolved = false;
    uint64_t std_thread_native_id = 0;
    std::thread frontend_guest([&] {
        uint8_t stack_byte = 0;
        std_thread_native_id = (uint64_t)GetCurrentThreadId();
        run_guest_inits({});
        void* self_base = nullptr;
        size_t self_size = 0;
        std_thread_self_resolved =
            guest_stack_for_thread((uint64_t)pthread_self(), &self_base, &self_size) &&
            (uintptr_t)&stack_byte >= (uintptr_t)self_base &&
            (uintptr_t)&stack_byte < (uintptr_t)self_base + self_size;
    });
    frontend_guest.join();
    void* exited_base = nullptr;
    size_t exited_size = 0;
    bool std_thread_registration_cleaned = !guest_stack_for_thread(
        std_thread_native_id, &exited_base, &exited_size);

    // A stale/unknown target must leave the caller-provided attr unchanged. Falling back to the
    // querying thread here is the exact wrong-stack substitution that can corrupt a GC scan.
    void* sentinel_stack = VirtualAlloc(nullptr, 1024 * 1024,
                                        MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    bool sentinel_set = sentinel_stack &&
        pthread_attr_setstack((pthread_attr_t*)attr, sentinel_stack, 1024 * 1024) == 0;
    reported_base = nullptr;
    reported_size = 0;
    attr_get((uint64_t)worker, (uint64_t)(uintptr_t)&attr, 0, 0, 0, 0);
    attr_getaddr((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&reported_base, 0, 0, 0, 0);
    attr_getsize((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&reported_size, 0, 0, 0, 0);
    bool missing_target_unchanged = sentinel_set && reported_base == sentinel_stack &&
                                    reported_size == 1024 * 1024;
    attr_destroy((uint64_t)(uintptr_t)&attr, 0, 0, 0, 0, 0);
    if (sentinel_stack) VirtualFree(sentinel_stack, 0, MEM_RELEASE);

    // A tiny Sony request is floored to leave room for the emulator boundary, and every usable
    // page is committed before guest entry.  This specifically guards PS5 code that makes a large
    // stack jump without the Windows page-by-page probes emitted by native Windows compilers.
    void* guest_attr = nullptr;
    attr_init((uint64_t)(uintptr_t)&guest_attr, 0, 0, 0, 0, 0);
    attr_setsize((uint64_t)(uintptr_t)&guest_attr, 64 * 1024, 0, 0, 0, 0);
    GuestStackProbe guest_probe;
    uint64_t guest_tid = 0;
    bool guest_created = thread_create((uint64_t)(uintptr_t)&guest_tid,
                                       (uint64_t)(uintptr_t)&guest_attr,
                                       (uint64_t)(uintptr_t)&prepared_guest_worker,
                                       (uint64_t)(uintptr_t)&guest_probe, 0, 0) == 0;
    if (guest_created) thread_join(guest_tid, 0, 0, 0, 0, 0);
    attr_destroy((uint64_t)(uintptr_t)&guest_attr, 0, 0, 0, 0, 0);
    bool guest_stack_prepared = guest_created && guest_probe.registered &&
                                guest_probe.registered_size >= 1024 * 1024 &&
                                guest_probe.fully_committed;

    unregister_thread_stack((uint64_t)GetCurrentThreadId());
    bool unregistered = is_stack(addr, 0, 0, 0, 0, 0) == 0;

    if (!inside || !below || !at_end || !attr_bounds || !worker_created || !resolved_worker ||
        !target_attr_bounds || !std_thread_self_resolved || !std_thread_registration_cleaned ||
        !missing_target_unchanged || !guest_stack_prepared || !unregistered) {
        std::fprintf(stderr,
                     "stack HLE mismatch: inside=%d below=%d at_end=%d attr_bounds=%d "
                     "worker_created=%d resolved_worker=%d target_attr_bounds=%d std_self=%d "
                     "std_cleaned=%d "
                     "missing_unchanged=%d guest_prepared=%d guest_size=%zu committed=%d "
                     "reported=%p/%zu expected=%p/%zu unregistered=%d\n",
                     inside, below, at_end, attr_bounds, worker_created, resolved_worker,
                     target_attr_bounds, std_thread_self_resolved,
                     std_thread_registration_cleaned, missing_target_unchanged,
                     guest_stack_prepared, guest_probe.registered_size, guest_probe.fully_committed,
                     reported_base, reported_size,
                     probe.base, probe.size, unregistered);
        return 1;
    }
    return 0;
}
