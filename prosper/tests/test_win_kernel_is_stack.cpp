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
    auto attr_destroy = Hle::lookup(nid_hash("scePthreadAttrDestroy"));
    if (!is_stack || !attr_init || !attr_get || !attr_getaddr || !attr_getsize || !attr_destroy) {
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

    // prosper-app and screenshot enter the guest from std::thread rather than a thread created
    // through our pthread HLE. winpthreads cannot always translate that implicit handle with
    // pthread_gethandle(), so a self-query must fall back to the current native thread id.
    bool std_thread_self_resolved = false;
    std::thread frontend_guest([&] {
        uint8_t stack_byte = 0;
        run_guest_inits({});
        void* self_base = nullptr;
        size_t self_size = 0;
        std_thread_self_resolved =
            guest_stack_for_thread((uint64_t)pthread_self(), &self_base, &self_size) &&
            (uintptr_t)&stack_byte >= (uintptr_t)self_base &&
            (uintptr_t)&stack_byte < (uintptr_t)self_base + self_size;
        unregister_thread_stack((uint64_t)GetCurrentThreadId());
    });
    frontend_guest.join();

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

    unregister_thread_stack((uint64_t)GetCurrentThreadId());
    bool unregistered = is_stack(addr, 0, 0, 0, 0, 0) == 0;

    if (!inside || !below || !at_end || !attr_bounds || !worker_created || !resolved_worker ||
        !target_attr_bounds || !std_thread_self_resolved || !missing_target_unchanged ||
        !unregistered) {
        std::fprintf(stderr,
                     "stack HLE mismatch: inside=%d below=%d at_end=%d attr_bounds=%d "
                     "worker_created=%d resolved_worker=%d target_attr_bounds=%d std_self=%d "
                     "missing_unchanged=%d reported=%p/%zu expected=%p/%zu unregistered=%d\n",
                     inside, below, at_end, attr_bounds, worker_created, resolved_worker,
                     target_attr_bounds, std_thread_self_resolved, missing_target_unchanged,
                     reported_base, reported_size,
                     probe.base, probe.size, unregistered);
        return 1;
    }
    return 0;
}
