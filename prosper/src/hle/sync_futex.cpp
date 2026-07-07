// sync_futex.cpp — see sync_futex.hpp.
#include "sync_futex.hpp"
#include <atomic>
#include <climits>
#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#endif

namespace prosper {
namespace { std::atomic<int> g_waiters{0}; }

void futex_wait_enter() { g_waiters.fetch_add(1, std::memory_order_seq_cst); }
void futex_wait_exit()  { g_waiters.fetch_sub(1, std::memory_order_seq_cst); }

void futex_wake(uint64_t addr, int n) {
#if defined(__linux__)
    if (!addr) return;
    syscall(SYS_futex, (uint32_t*)(uintptr_t)addr, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, n, nullptr, nullptr, 0);
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
