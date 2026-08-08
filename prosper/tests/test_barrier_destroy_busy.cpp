// #2168 (barrier half): scePthreadBarrierDestroy must refuse a barrier that still has waiters.
//
// FreeBSD's pthread_barrier_destroy returns EBUSY when `bar->b_waiters > 0` and does NOT free the
// object. prosper returned 0 unconditionally, so a guest destroying a barrier out from under parked
// threads was told it succeeded and had its slot cleared.
//
// The barrier is worse than the condvar half (#2359) in one specific way, and the test is shaped
// around it: every waiter is parked until the count is reached, so a destroy underneath them strands
// ALL of them, not one. That means a regression here HANGS rather than fails, which is why the joins
// below are watchdogged — a CI timeout with no message is strictly worse than a failure.

#include "hle/dispatch.hpp"
#include "hle/nid.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
    else std::fprintf(stderr, "ok: %s\n", what);
}

using prosper::Hle;
using prosper::HleFn;

static uint64_t call(HleFn fn, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0) {
    return fn(a0, a1, a2, 0, 0, 0);
}

int main() {
    std::fprintf(stderr, "== test_barrier_destroy_busy ==\n");
    prosper::register_kernel_hle();

    const auto by_name = [](const char* n) { return Hle::lookup(prosper::nid_hash(n)); };
    HleFn init    = by_name("scePthreadBarrierInit");
    HleFn wait    = by_name("scePthreadBarrierWait");
    HleFn destroy = by_name("scePthreadBarrierDestroy");
    check(init && wait && destroy, "the barrier entry points are registered");
    if (failures) { std::fprintf(stderr, "== FAIL ==\n"); return 1; }

    // --- the control FIRST: a barrier with no waiters must still destroy cleanly ----------------
    // Asserted before the busy case so a change returning EBUSY unconditionally cannot pass this
    // file. Its own barrier, so it cannot disturb the one below.
    {
        uint64_t spare = 0;
        call(init, (uint64_t)(uintptr_t)&spare, 0, 2);
        check(call(destroy, (uint64_t)(uintptr_t)&spare) == 0,
              "control: destroying a barrier with no waiters returns 0");
    }

    // A barrier of 2: one worker parks, and the main thread never arrives, so it stays parked.
    uint64_t slot = 0;
    call(init, (uint64_t)(uintptr_t)&slot, 0, 2);

    std::atomic<bool> entered{false}, released{false};
    std::thread waiter([&] {
        entered.store(true, std::memory_order_release);
        call(wait, (uint64_t)(uintptr_t)&slot);
        released.store(true, std::memory_order_release);
    });

    // ONE probe, after a delay — never a poll. `entered` says the thread reached the call, not that
    // it is inside it, and there is no portable way to observe the moment a barrier parks.
    //
    // I first wrote this as a polling loop with a comment claiming it was safe "because destroy is
    // side-effect-free while it refuses". That is true only WHILE it refuses. The FIRST probe runs
    // before the waiter has parked, so it does not refuse — it destroys the barrier, and every later
    // probe then peeks a retired slot and also returns 0. The loop cannot converge, and it reported
    // this fix as broken when the fix was fine.
    //
    // That is the identical defect recorded in test_cond_destroy_busy.cpp, reproduced here by me
    // after writing that comment. A probe that mutates what it probes is not made safe by being
    // repeated; it is made worse. So: sleep past the park, then probe exactly once.
    while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const uint64_t busy_rc = call(destroy, (uint64_t)(uintptr_t)&slot);

    // The Sony encoding, not the bare errno: this body is Sony-only (there is no
    // pthread_barrier_destroy registration) and its sibling scePthreadBarrierInit already returns
    // kSceKernelError* directly, so a bare 16 here would be the #1984 split defect.
    check(busy_rc == 0x80020010ull,
          "scePthreadBarrierDestroy with a thread parked returns ENCODED EBUSY (0x80020010)");
    check(slot != 0, "a refused destroy leaves the barrier slot intact");

    // Release the waiter by arriving ourselves, and confirm the object still works after refusal.
    call(wait, (uint64_t)(uintptr_t)&slot);
    for (int spin = 0; spin < 1000 && !released.load(std::memory_order_acquire); ++spin)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (!released.load(std::memory_order_acquire)) {
        check(false, "the waiter did not wake within 5 s — the barrier was destroyed out from "
                     "under a parked thread, so arriving reached nothing (#2168 regressed)");
        waiter.detach();
        std::fprintf(stderr, "== FAIL ==\n");
        return 1;
    }
    waiter.join();
    check(call(destroy, (uint64_t)(uintptr_t)&slot) == 0,
          "once the waiter has left, the destroy succeeds");

    std::fprintf(stderr, failures ? "== FAIL ==\n" : "== PASS ==\n");
    return failures ? 1 : 0;
}
