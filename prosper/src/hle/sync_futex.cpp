// sync_futex.cpp — see sync_futex.hpp.
#include "sync_futex.hpp"
#include <atomic>
#include <climits>
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <sys/syscall.h>
#include "../host/posix_shim.hpp"
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00   // WakeByAddressAll needs >= 0x0602 (Win8)
#endif
#include <windows.h>   // native futex: WakeByAddressAll (pairs with WaitOnAddress in hle_kernel_mem.cpp)
#include <pthread.h>
#include <mutex>
#include <unordered_map>
#endif

namespace prosper {
namespace {
std::atomic<int> g_waiters{0};
#ifdef _WIN32
std::mutex g_wait_mx;
std::unordered_map<uint64_t, uint64_t> g_thread_waits;  // pthread_t -> guest wait address
#endif
}

void futex_wait_enter(uint64_t addr) {
#ifdef _WIN32
    { std::lock_guard<std::mutex> lk(g_wait_mx); g_thread_waits[(uint64_t)pthread_self()] = addr; }
#else
    (void)addr;
#endif
    g_waiters.fetch_add(1, std::memory_order_seq_cst);
}
void futex_wait_exit() {
    g_waiters.fetch_sub(1, std::memory_order_seq_cst);
#ifdef _WIN32
    std::lock_guard<std::mutex> lk(g_wait_mx);
    g_thread_waits.erase((uint64_t)pthread_self());
#endif
}

bool interrupt_futex_wait(uint64_t thread) {
#ifdef _WIN32
    uint64_t addr = 0;
    { std::lock_guard<std::mutex> lk(g_wait_mx);
      auto it = g_thread_waits.find(thread);
      if (it != g_thread_waits.end()) addr = it->second; }
    if (addr) WakeByAddressAll((PVOID)(uintptr_t)addr);
    return addr != 0;
#else
    (void)thread;
    return false;
#endif
}

void futex_wake(uint64_t addr, int n) {
#if defined(__linux__) || defined(__APPLE__)
    if (!addr) return;
    prosper_futex_wake((uint32_t*)(uintptr_t)addr, n > 1);
#elif defined(_WIN32)
    // Native Win32 futex wake, pairing with sceKernelWaitOnAddress's WaitOnAddress (hle_kernel_mem.cpp).
    // The GPU command processor calls this from RELEASE_MEM/EOP to wake a guest render/producer thread
    // parked on the label the GPU just wrote — without it the guest's WaitOnAddress never wakes on
    // Windows and rendering wedges after the first submit. Wake ALL (n is a hint; label waiters are few).
    if (!addr) return;
    (void)n;
    WakeByAddressAll((PVOID)(uintptr_t)addr);
#else
    (void)addr; (void)n;
#endif
}

void wake_label_waiters(uint64_t addr) {
    if (!addr) return;
    // The label was just stored with a PLAIN write; x86 permits store->load reordering, so without a
    // full fence this counter load could execute before that store is visible and a registering waiter
    // could then block on the stale value with nobody left to wake. Fence, THEN check.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (g_waiters.load(std::memory_order_seq_cst) == 0) return;
    futex_wake(addr, INT_MAX);
    futex_wake(addr + 4, INT_MAX);   // 64-bit labels: a waiter may block on the high dword too
}

} // namespace prosper
