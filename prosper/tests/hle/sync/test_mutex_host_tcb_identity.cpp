// A guest mutex must be unlockable by the thread that locked it, even when prosper's import stub
// restored the WRONG host TCB for the call (#3615 / #3623).
//
// THE DEFECT, precisely. prosper stashes each host thread's `%fs` inside the GUEST TCB it builds for
// a guest thread, and the import stubs restore `%fs` from that stash on every HLE call. The stash is
// per-guest-TCB; the value is per-host-thread. A guest TCB that becomes active on a second host
// thread therefore makes the stub install the FIRST thread's host TCB.
//
// glibc does not resolve a mutex's owner from a syscall-derived thread id -- it compares against the
// tid cached in the TCB that `%fs` points at. So the unlocking thread is judged to be a different
// thread and `pthread_mutex_unlock` returns EPERM, even though the same host thread locked it
// microseconds earlier. Measured on PPSA05684: identical tid, identical mutex, `owner` equal to the
// caller, EPERM anyway, the two calls differing only in `%fs`.
//
// WHAT THIS TEST DOES. It reproduces that exact condition without a guest: lock through the slot API,
// point `%fs` at a REAL second host thread's TCB (the thing the stale stash would have installed),
// unlock through the slot API, restore. `pthread_self()` on glibc returns the TCB address, so the
// helper thread hands back precisely the value a stale stash holds.
//
// Two properties are load-bearing:
//
//  * The mutex is explicitly PTHREAD_MUTEX_ERRORCHECK. On Linux `ensure_mutex` leaves static-slot
//    mutexes at the host default (NORMAL), which performs NO ownership check on unlock and so cannot
//    fail this way -- a test that let it default would pass against the unfixed code.
//  * Nothing between the `%fs` swap and the call may touch TLS: with a foreign TCB installed, errno,
//    thread_local and the stack canary all belong to the other thread. The window is exactly one call.
#include "hle/sync/pthread_slot.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <pthread.h>
#include <thread>

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

#if defined(__linux__) && defined(__x86_64__)
static inline uint64_t rd_fs() { uint64_t v; __asm__ volatile("rdfsbase %0" : "=r"(v)); return v; }
static inline void     wr_fs(uint64_t v) { __asm__ volatile("wrfsbase %0" : : "r"(v)); }
#endif

int main() {
    printf("== test_mutex_host_tcb_identity ==\n");
#if !defined(__linux__) || !defined(__x86_64__)
    printf("  [skip] guest-%%fs swapping is x86-64 Linux only\n== PASS ==\n");
    return 0;
#else
    // A second REAL host thread, parked, lending us its TCB address — the value a stale stash holds.
    std::atomic<uint64_t> other_tcb{0};
    std::atomic<bool> done{false};
    std::thread helper([&] {
        other_tcb.store((uint64_t)pthread_self(), std::memory_order_release);
        while (!done.load(std::memory_order_acquire)) std::this_thread::yield();
    });
    while (other_tcb.load(std::memory_order_acquire) == 0) std::this_thread::yield();
    const uint64_t foreign = other_tcb.load(std::memory_order_acquire);
    const uint64_t mine    = rd_fs();
    CHECK(foreign != 0 && foreign != mine,
          "CONTROL: the helper thread's TCB really is a different TCB from this thread's -- "
          "otherwise the swap below would be a no-op and the test would prove nothing");

    // Prime the registry the fix consults, exactly as guest_tls_activate_thread would.
    prosper::guest_tls_record_host_fs(mine);

    // An ERRORCHECK mutex reached through the slot API: ensure_mutex returns a non-sentinel slot
    // value as-is, so this is the same code path a guest-initialised mutex takes.
    pthread_mutex_t m;
    pthread_mutexattr_t at; pthread_mutexattr_init(&at);
    pthread_mutexattr_settype(&at, PTHREAD_MUTEX_ERRORCHECK);
    pthread_mutex_init(&m, &at);
    pthread_mutexattr_destroy(&at);
    void* slot_value = &m;
    const uint64_t slot_addr = (uint64_t)&slot_value;

    const uint64_t lock_rc = prosper::guest_mutex_lock_slot(slot_addr);
    CHECK(lock_rc == 0, "CONTROL: the lock itself succeeds on this thread");

    // The defect's exact condition: a foreign host TCB installed for the duration of the unlock.
    wr_fs(foreign);
    const uint64_t unlock_rc = prosper::guest_mutex_unlock_slot(slot_addr);
    wr_fs(mine);

    CHECK(unlock_rc == 0,
          "#3615: the locking thread can unlock even though the call arrived on ANOTHER host "
          "thread's TCB -- without the fix glibc reads the foreign TCB's tid and returns EPERM");

    // The mutex must really be released, not merely reported released: a fix that swallowed the
    // error would pass the arm above and leave the lock held forever.
    const int relock = pthread_mutex_trylock(&m);
    CHECK(relock == 0,
          "#3615: ...and the mutex is genuinely unlocked afterwards, so the unlock happened rather "
          "than its failure being hidden");
    if (relock == 0) pthread_mutex_unlock(&m);
    pthread_mutex_destroy(&m);

    done.store(true, std::memory_order_release);
    helper.join();

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
#endif
}
