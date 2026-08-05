// test_rwlock_once — functional guard for the scePthreadRwlock* / scePthreadOnce HLE. These back
// real libc.prx's internal locking; a no-op stub (the old behavior) leaves libc state unlocked under
// the multithreaded IL2CPP pool (data races). This calls the handlers through the NID registry and
// asserts REAL semantics: a write-lock actually excludes (trywrlock fails while held), and once runs
// its init exactly once.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <atomic>
#include <chrono>
#include <thread>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static int g_once_count = 0;
static void once_init() { g_once_count++; }

int main() {
    printf("== test_rwlock_once ==\n");
    register_builtin_hle();

    auto RWinit  = Hle::lookup(nid_hash("scePthreadRwlockInit"));
    auto RWrd    = Hle::lookup(nid_hash("scePthreadRwlockRdlock"));
    auto RWwr    = Hle::lookup(nid_hash("scePthreadRwlockWrlock"));
    auto RWun    = Hle::lookup(nid_hash("scePthreadRwlockUnlock"));
    auto RWtrywr = Hle::lookup(nid_hash("scePthreadRwlockTrywrlock"));
    auto RWdes   = Hle::lookup(nid_hash("scePthreadRwlockDestroy"));
    auto ONCE    = Hle::lookup(nid_hash("scePthreadOnce"));
    CHECK(RWinit && RWrd && RWwr && RWun && RWtrywr && RWdes && ONCE, "rwlock + once registered");
    if (!(RWinit && RWrd && RWwr && RWun && RWtrywr && RWdes && ONCE)) { printf("== FAIL ==\n"); return 1; }

    auto call1 = [](HleFn f, uint64_t a) { return f(a, 0, 0, 0, 0, 0); };
    auto call2 = [](HleFn f, uint64_t a, uint64_t b) { return f(a, b, 0, 0, 0, 0); };

    void* h = nullptr;
    call1(RWinit, (uint64_t)(uintptr_t)&h);
    CHECK(h != nullptr, "rwlock init allocated a handle");
    CHECK(call1(RWrd, (uint64_t)(uintptr_t)&h) == 0, "rdlock ok");
    CHECK(call1(RWun, (uint64_t)(uintptr_t)&h) == 0, "unlock after rdlock ok");
    CHECK(call1(RWwr, (uint64_t)(uintptr_t)&h) == 0, "wrlock ok");
    // The lock is REAL: a trywrlock while the write lock is held must fail (nonzero). A no-op stub
    // would wrongly return 0 here.
    CHECK(call1(RWtrywr, (uint64_t)(uintptr_t)&h) != 0, "trywrlock fails while write-held (real exclusion)");
    CHECK(call1(RWun, (uint64_t)(uintptr_t)&h) == 0, "unlock after wrlock ok");
    CHECK(call1(RWtrywr, (uint64_t)(uintptr_t)&h) == 0, "trywrlock succeeds when free");
    call1(RWun, (uint64_t)(uintptr_t)&h);

    // #2012: an UNMATCHED unlock — the calling thread holds neither a read nor a write hold — must
    // be refused, not forwarded. FreeBSD (the guest's contract) returns EPERM and leaves the lock
    // untouched; glibc's is undefined and subtracts a reader, driving the count NEGATIVE so the
    // lock can never be write-acquired again. The lock still working afterwards is the assertion
    // that matters: without the guard, `trywrlock` below sees a phantom reader and returns EBUSY.
    const uint64_t unmatched = call1(RWun, (uint64_t)(uintptr_t)&h);
    CHECK(unmatched == 0x80020001ull, "unmatched scePthreadRwlockUnlock returns encoded EPERM");
    CHECK(call1(RWtrywr, (uint64_t)(uintptr_t)&h) == 0,
          "the lock still write-acquires after an unmatched unlock (reader count not corrupted)");
    call1(RWun, (uint64_t)(uintptr_t)&h);
    // The POSIX spelling shares the body and keeps the bare errno (the #1984 split).
    auto RWunPosix = Hle::lookup(nid_hash("pthread_rwlock_unlock"));
    CHECK(RWunPosix != nullptr, "pthread_rwlock_unlock registered");
    if (RWunPosix)
        CHECK(call1(RWunPosix, (uint64_t)(uintptr_t)&h) == 1,
              "unmatched pthread_rwlock_unlock returns bare EPERM");
    // A read hold taken by THIS thread still releases normally afterwards.
    CHECK(call1(RWrd, (uint64_t)(uintptr_t)&h) == 0, "rdlock ok after the refusals");
    CHECK(call1(RWun, (uint64_t)(uintptr_t)&h) == 0, "matched unlock still returns success");

    // #2012: libScePosix exports POSIX spellings of the try/timed acquisitions too. Unregistered,
    // they fell to the dispatcher's default 0 — "the lock is yours" for a lock nobody took — which
    // is how a guest ends up unlocking a lock it never held. They must exist AND really fail while
    // the lock is write-held; a 0 here is the stub answer, not an acquisition.
    auto RWtryrdP = Hle::lookup(nid_hash("pthread_rwlock_tryrdlock"));
    auto RWtrywrP = Hle::lookup(nid_hash("pthread_rwlock_trywrlock"));
    auto RWtimedrdP = Hle::lookup(nid_hash("pthread_rwlock_timedrdlock"));
    auto RWtimedwrP = Hle::lookup(nid_hash("pthread_rwlock_timedwrlock"));
    auto RWreltimedrdP = Hle::lookup(nid_hash("pthread_rwlock_reltimedrdlock_np"));
    auto RWreltimedwrP = Hle::lookup(nid_hash("pthread_rwlock_reltimedwrlock_np"));
    CHECK(RWtryrdP && RWtrywrP && RWtimedrdP && RWtimedwrP && RWreltimedrdP && RWreltimedwrP,
          "POSIX rwlock try/timed spellings are registered (not the default-0 stub)");
    if (RWtryrdP && RWtrywrP && RWtimedrdP && RWtimedwrP && RWreltimedrdP && RWreltimedwrP) {
        // The write hold must belong to ANOTHER thread: a timed acquisition attempted by the thread
        // that already holds the write lock reports EDEADLK, not ETIMEDOUT, so a same-thread arm
        // would test the deadlock detector instead of the timeout path.
        std::atomic<bool> held{false}, release{false};
        std::thread holder([&]{
            call1(RWwr, (uint64_t)(uintptr_t)&h);
            held.store(true);
            while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            call1(RWun, (uint64_t)(uintptr_t)&h);
        });
        for (int i = 0; i < 2000 && !held.load(); i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(held.load(), "helper thread holds the write lock for the POSIX try/timed arms");
        CHECK(call1(RWtrywrP, (uint64_t)(uintptr_t)&h) != 0,
              "pthread_rwlock_trywrlock FAILS while another thread write-holds (a 0 is the stub lie)");
        CHECK(call1(RWtryrdP, (uint64_t)(uintptr_t)&h) != 0,
              "pthread_rwlock_tryrdlock FAILS while another thread write-holds");
        // POSIX timed variants take an ABSOLUTE timespec*; a deadline already in the past must time
        // out rather than acquire. The Sony spellings take a RELATIVE microsecond scalar, so reading
        // this argument the Sony way would dereference the pointer value as a µs count instead.
        timespec past{};
        clock_gettime(CLOCK_REALTIME, &past);
        past.tv_sec -= 1;
        CHECK(call2(RWtimedwrP, (uint64_t)(uintptr_t)&h, (uint64_t)(uintptr_t)&past) == 60,
              "pthread_rwlock_timedwrlock with a past absolute deadline reports ETIMEDOUT(60)");
        CHECK(call2(RWtimedrdP, (uint64_t)(uintptr_t)&h, (uint64_t)(uintptr_t)&past) == 60,
              "pthread_rwlock_timedrdlock with a past absolute deadline reports ETIMEDOUT(60)");
        // The _np variants take a RELATIVE timespec: zero means "do not wait".
        timespec zero{0, 0};
        CHECK(call2(RWreltimedwrP, (uint64_t)(uintptr_t)&h, (uint64_t)(uintptr_t)&zero) == 60,
              "pthread_rwlock_reltimedwrlock_np with a zero relative timeout reports ETIMEDOUT(60)");
        CHECK(call2(RWreltimedrdP, (uint64_t)(uintptr_t)&h, (uint64_t)(uintptr_t)&zero) == 60,
              "pthread_rwlock_reltimedrdlock_np with a zero relative timeout reports ETIMEDOUT(60)");
        release.store(true);
        holder.join();
        // The lock is free and undamaged once the helper releases it.
        CHECK(call1(RWtrywrP, (uint64_t)(uintptr_t)&h) == 0,
              "pthread_rwlock_trywrlock succeeds once the helper releases");
        CHECK(call1(RWun, (uint64_t)(uintptr_t)&h) == 0, "unlock the write hold from the POSIX arms");
    }
    call1(RWdes, (uint64_t)(uintptr_t)&h);

    // once: init runs exactly once no matter how many times it's called.
    int ctl = 0; g_once_count = 0;
    for (int i = 0; i < 3; i++) call2(ONCE, (uint64_t)(uintptr_t)&ctl, (uint64_t)(uintptr_t)&once_init);
    CHECK(g_once_count == 1, "scePthreadOnce runs init exactly once across 3 calls");

    // Cross-control independence (#69): init routine for control X blocks until ANOTHER thread
    // completes once(Y). The old one-global-recursive-mutex implementation deadlocked here (Y's
    // caller parked behind X's global lock; X waited on Y forever). Per-control serialization
    // (real pthread_once semantics) must let Y proceed while X's init is still running.
    {
        static HleFn s_once = ONCE;
        static int s_ctl_y = 0;
        static std::atomic<bool> s_y_done{false};
        static auto s_init_y = +[]() { s_y_done.store(true); };
        static auto s_init_x = +[]() {
            // From inside X's init: run once(Y) on a worker and WAIT for it — cross-thread,
            // cross-control dependency, legal on real hardware.
            std::thread t([]{ s_once((uint64_t)(uintptr_t)&s_ctl_y, (uint64_t)(uintptr_t)+s_init_y, 0, 0, 0, 0); });
            t.join();
        };
        int ctl_x = 0;
        std::atomic<bool> x_returned{false};
        std::thread xt([&]{ s_once((uint64_t)(uintptr_t)&ctl_x, (uint64_t)(uintptr_t)+s_init_x, 0, 0, 0, 0);
                            x_returned.store(true); });
        // Deadlock watchdog: the whole thing must finish well within 5 s.
        for (int i = 0; i < 500 && !x_returned.load(); i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(x_returned.load(), "once(X) whose init depends on another thread's once(Y) completes (no global-lock deadlock)");
        CHECK(s_y_done.load(), "the dependent once(Y) actually ran");
        if (x_returned.load()) xt.join(); else xt.detach();   // don't hang the test binary on failure
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
