#include "../src/host/exec_image.hpp"
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <windows.h>
#include <pthread.h>
#include <cstdint>
#include <cstdio>

using namespace prosper;

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
    attr_destroy((uint64_t)(uintptr_t)&attr, 0, 0, 0, 0, 0);
    bool attr_bounds = reported_base == (void*)(uintptr_t)base && reported_size == 0x2000;

    unregister_thread_stack((uint64_t)GetCurrentThreadId());
    bool unregistered = is_stack(addr, 0, 0, 0, 0, 0) == 0;

    if (!inside || !below || !at_end || !attr_bounds || !unregistered) {
        std::fprintf(stderr,
                     "stack HLE mismatch: inside=%d below=%d at_end=%d attr_bounds=%d "
                     "reported=%p/%zu unregistered=%d\n",
                     inside, below, at_end, attr_bounds, reported_base, reported_size, unregistered);
        return 1;
    }
    return 0;
}
