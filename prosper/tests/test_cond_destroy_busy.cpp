// #2168: scePthreadCondDestroy / pthread_cond_destroy must refuse a condvar that still has waiters.
//
// FreeBSD's `_thr_cond_destroy` returns EBUSY when `cvp->__has_user_waiters || cvp->kcond.
// c_has_waiters`, and it does NOT free the object. prosper returned 0 unconditionally, so a guest
// that destroyed a condvar out from under a parked thread was told it succeeded and had its slot
// cleared -- where hardware hands back EBUSY and leaves a working object behind.
//
// The two spellings differ in ENCODING only: the POSIX name returns the bare errno, the Sony name
// the libkernel form (0x80020010). Both are asserted, because the whole point of the #1984 split is
// that the answer must not depend on which spelling the guest happened to call.

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

static uint64_t call(HleFn fn, uint64_t a0 = 0, uint64_t a1 = 0) {
    return fn(a0, a1, 0, 0, 0, 0);
}

int main() {
    std::fprintf(stderr, "== test_cond_destroy_busy ==\n");

    // The dispatch table is populated by the registrar, not statically -- without this every
    // lookup below returns nullptr and the test would report "not registered" for functions that
    // are perfectly fine.
    prosper::register_kernel_hle();

    // Looked up by NAME through nid_hash rather than by a hard-coded NID: the registration is what
    // this test guards, and a literal NID would silently stop referring to it if the mapping moved.
    const auto by_name = [](const char* name) {
        return Hle::lookup(prosper::nid_hash(name));
    };
    HleFn cond_init      = by_name("pthread_cond_init");
    HleFn cond_wait      = by_name("pthread_cond_wait");
    HleFn cond_broadcast = by_name("pthread_cond_broadcast");
    HleFn cond_destroy   = by_name("pthread_cond_destroy");
    HleFn mutex_init     = by_name("pthread_mutex_init");
    HleFn mutex_lock     = by_name("pthread_mutex_lock");
    HleFn mutex_unlock   = by_name("pthread_mutex_unlock");

    check(cond_init && cond_wait && cond_broadcast && cond_destroy &&
          mutex_init && mutex_lock && mutex_unlock,
          "every pthread entry point this test needs is registered");
    if (failures) { std::fprintf(stderr, "== FAIL ==\n"); return 1; }

    // Guest-visible slots: the HLE takes the ADDRESS of a pointer-sized slot, not the object.
    uint64_t cond_slot = 0, mutex_slot = 0;
    call(cond_init, (uint64_t)(uintptr_t)&cond_slot, 0);
    call(mutex_init, (uint64_t)(uintptr_t)&mutex_slot, 0);

    // --- the control, first: with no waiters a destroy must still SUCCEED --------------------
    // Asserted before the busy case rather than after, so a change that made destroy return EBUSY
    // unconditionally could not pass this file. Its own condvar, so it cannot disturb the one below.
    {
        uint64_t spare = 0;
        call(cond_init, (uint64_t)(uintptr_t)&spare, 0);
        check(call(cond_destroy, (uint64_t)(uintptr_t)&spare) == 0,
              "control: destroying a condvar with no waiters returns 0");
    }

    std::atomic<bool> waiter_running{false};
    std::atomic<bool> waiter_done{false};
    std::thread waiter([&] {
        call(mutex_lock, (uint64_t)(uintptr_t)&mutex_slot);
        waiter_running.store(true, std::memory_order_release);
        call(cond_wait, (uint64_t)(uintptr_t)&cond_slot, (uint64_t)(uintptr_t)&mutex_slot);
        call(mutex_unlock, (uint64_t)(uintptr_t)&mutex_slot);
        waiter_done.store(true, std::memory_order_release);
    });

    // Wait until the waiter is genuinely PARKED before attempting the destroy, using the mutex as
    // the handshake: `pthread_cond_wait` releases it on entry, so the main thread can only acquire
    // it once the waiter is inside the wait. `waiter_running` alone is not enough -- it says the
    // thread reached the lock, not that it reached the wait.
    //
    // An earlier version of this test polled `cond_destroy` in a loop instead, and it FAILED for a
    // reason worth recording: the first iteration ran before the waiter had parked, so it destroyed
    // the condvar and returned 0, and every later iteration then peeked an already-retired slot and
    // also returned 0. A polling loop whose first probe MUTATES the thing it is polling cannot
    // converge, and it reported the fix as broken when the fix was fine.
    while (!waiter_running.load(std::memory_order_acquire))
        std::this_thread::yield();
    call(mutex_lock, (uint64_t)(uintptr_t)&mutex_slot);
    call(mutex_unlock, (uint64_t)(uintptr_t)&mutex_slot);

    const uint64_t busy_rc = call(cond_destroy, (uint64_t)(uintptr_t)&cond_slot);

    // 16 is FreeBSD EBUSY, and it is the guest's platform value rather than the host's -- on MinGW
    // EBUSY is 16 too, but on Linux it is 16 and on some hosts it is not, so the constant is
    // asserted against the FreeBSD number the guest actually expects.
    check(busy_rc == 16,
          "pthread_cond_destroy on a condvar with a waiter returns bare FreeBSD EBUSY (16)");

    // The refusal must not have retired the slot: FreeBSD does not free on EBUSY, and a guest that
    // gets EBUSY is entitled to keep using the object -- which is exactly what the broadcast below
    // does. If the slot had been claimed, this would wake nobody and the join would hang.
    check(cond_slot != 0, "a refused destroy leaves the condvar slot intact");

    // The Sony spelling: same behaviour, libkernel encoding. Asserted through the real dispatch
    // table rather than by calling the alias directly, because the defect this guards against is a
    // REGISTRATION one -- scePthreadCondDestroy used to point straight at the always-0 body.
    {
        HleFn sce = by_name("scePthreadCondDestroy");
        check(sce != nullptr, "scePthreadCondDestroy is registered");
        if (sce) {
            const uint64_t rc = call(sce, (uint64_t)(uintptr_t)&cond_slot);
            check(rc == 0x80020010ull,
                  "scePthreadCondDestroy returns the ENCODED EBUSY (0x80020010), not the bare 16");
        }
    }

    // Release the waiter and confirm the object still works after two refused destroys.
    call(mutex_lock, (uint64_t)(uintptr_t)&mutex_slot);
    call(cond_broadcast, (uint64_t)(uintptr_t)&cond_slot);
    call(mutex_unlock, (uint64_t)(uintptr_t)&mutex_slot);

    // A WATCHDOG, not a nicety, and mutation testing is what showed it was needed. With the busy
    // check removed, the destroy above succeeds, the condvar is retired out from under the parked
    // thread, and this broadcast then wakes nobody -- so a regression makes this test HANG rather
    // than fail. A hang is strictly worse than a failure: CI reports a timeout with no message,
    // and the first person to see it has to reproduce the deadlock to learn what broke.
    //
    // So the join is bounded. Detaching a still-parked waiter is safe here because the process is
    // about to exit and the slots outlive it; the point is to FAIL LOUDLY with the diagnosis
    // already written down.
    for (int spin = 0; spin < 1000 && !waiter_done.load(std::memory_order_acquire); ++spin)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (!waiter_done.load(std::memory_order_acquire)) {
        check(false, "the waiter did not wake within 5 s -- the condvar was destroyed out from "
                     "under a parked thread, so the broadcast reached nothing (#2168 regressed)");
        waiter.detach();
        std::fprintf(stderr, "== FAIL ==\n");
        return 1;
    }
    waiter.join();
    check(waiter_done.load(std::memory_order_acquire),
          "the waiter was woken through a condvar whose destroy had been refused");

    check(call(cond_destroy, (uint64_t)(uintptr_t)&cond_slot) == 0,
          "once the waiter has left, the destroy succeeds");

    // --- the sibling entry point, which review found uncounted -----------------------------------
    //
    // `pthread_cond_timedwait(cond, mutex, NULL)` parks INDEFINITELY through the identical call that
    // `pthread_cond_wait` makes. The first version of this fix bracketed the waiter count on
    // `k_cond_wait` alone, so a thread parked here was invisible to the busy check and the destroy
    // retired the slot out from under it -- the exact bug this file guards, reached through the
    // sibling function.
    //
    // The arms above could not see it, because they park via `pthread_cond_wait`. Same lesson as the
    // polling loop earlier in this file: a test cannot catch a state it never produces.
    {
        HleFn timedwait = by_name("pthread_cond_timedwait");
        check(timedwait != nullptr, "pthread_cond_timedwait is registered");
        uint64_t cond2 = 0, mutex2 = 0;
        call(cond_init, (uint64_t)(uintptr_t)&cond2, 0);
        call(mutex_init, (uint64_t)(uintptr_t)&mutex2, 0);

        std::atomic<bool> parked{false}, released{false};
        std::thread waiter2([&] {
            call(mutex_lock, (uint64_t)(uintptr_t)&mutex2);
            parked.store(true, std::memory_order_release);
            timedwait((uint64_t)(uintptr_t)&cond2, (uint64_t)(uintptr_t)&mutex2, 0, 0, 0, 0);
            call(mutex_unlock, (uint64_t)(uintptr_t)&mutex2);
            released.store(true, std::memory_order_release);
        });
        while (!parked.load(std::memory_order_acquire)) std::this_thread::yield();
        call(mutex_lock, (uint64_t)(uintptr_t)&mutex2);      // granted only once it is inside
        call(mutex_unlock, (uint64_t)(uintptr_t)&mutex2);

        check(call(cond_destroy, (uint64_t)(uintptr_t)&cond2) == 16,
              "a thread parked via pthread_cond_timedwait(NULL) also counts as a waiter");

        call(mutex_lock, (uint64_t)(uintptr_t)&mutex2);
        call(cond_broadcast, (uint64_t)(uintptr_t)&cond2);
        call(mutex_unlock, (uint64_t)(uintptr_t)&mutex2);
        for (int spin = 0; spin < 1000 && !released.load(std::memory_order_acquire); ++spin)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (!released.load(std::memory_order_acquire)) {
            check(false, "the timedwait waiter did not wake within 5 s (#2168 regressed)");
            waiter2.detach();
            std::fprintf(stderr, "== FAIL ==\n");
            return 1;
        }
        waiter2.join();
        check(call(cond_destroy, (uint64_t)(uintptr_t)&cond2) == 0,
              "and once it has left, that condvar destroys cleanly too");
    }

    std::fprintf(stderr, failures ? "== FAIL ==\n" : "== PASS ==\n");
    return failures ? 1 : 0;
}
