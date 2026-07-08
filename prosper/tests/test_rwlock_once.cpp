// test_rwlock_once — functional guard for the scePthreadRwlock* / scePthreadOnce HLE. These back
// real libc.prx's internal locking; a no-op stub (the old behavior) leaves libc state unlocked under
// the multithreaded IL2CPP pool (data races). This calls the handlers through the NID registry and
// asserts REAL semantics: a write-lock actually excludes (trywrlock fails while held), and once runs
// its init exactly once.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
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
