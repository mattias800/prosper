// One condition variable must resolve to ONE slot, for the life of that condvar (#2139).
//
// The Windows condition layer maps each guest `pthread_cond_t*` onto a fixed-size table of slots,
// and the waiter and the signaller find their slot independently. If a lookup can ever return a
// DIFFERENT slot for the same condvar, the signal is delivered to a sequence nobody is waiting on
// and the waiter sleeps forever. There is no error and no log -- the only evidence afterwards is a
// process in which every thread is parked, which is exactly how Blue Prince deadlocked.
//
// The trigger is slot reuse: `cond_slot_for` used to look for an existing owner and claim a free
// slot in the SAME pass, so a slot freed EARLIER in the table was taken in preference to the slot
// this condvar already owned.
//
// Two things about the shape of this test are load-bearing:
//
//  * It gates on `snapshot_guest_wait` rather than on a sleep. The defect only shows if the waiter
//    is already REGISTERED on its slot when the signal is sent; a sleep-based gate let the waiter
//    reach the predicate after it was set and pass without ever testing anything. Measured: with a
//    30 ms sleep the case passed 3/3 against the unfixed code. Once the wait is registered the
//    outcome is timing-independent, because the waiter captured its expected sequence beforehand.
//  * It does not join a waiter it failed to wake. On the unfixed path the missed signal is
//    unrecoverable from inside the test -- a rescue broadcast resolves to the same wrong slot -- so
//    joining would hang the suite instead of reporting a failure.
#include "../src/hle/sync_futex.hpp"

#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); fails++; } \
                              else std::printf("  [ok]   %s\n", msg); } while (0)

namespace {

constexpr int kWakeBudgetMs = 5000;   // reaching this IS the defect, not a tolerance

pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
volatile int g_predicate = 0;
volatile unsigned long g_waiter_tid = 0;
volatile int g_woke = 0;

void sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    timespec ts{ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, nullptr);
#endif
}

void* wait_on(void* raw) {
    auto* cond = (pthread_cond_t*)raw;
#ifdef _WIN32
    g_waiter_tid = (unsigned long)GetCurrentThreadId();
#endif
    pthread_mutex_lock(&g_mutex);
    while (!g_predicate) interruptible_cond_wait(cond, &g_mutex);
    g_woke = 1;
    pthread_mutex_unlock(&g_mutex);
    return nullptr;
}

// Block until the waiter's condition wait is REGISTERED, which is the state the defect needs. Any
// weaker gate makes this test able to pass without exercising anything.
bool await_registered_wait() {
#ifdef _WIN32
    for (int i = 0; i < 3000; ++i) {
        GuestWaitSnapshot snapshot{};
        if (g_waiter_tid &&
            snapshot_guest_wait((uint64_t)g_waiter_tid, snapshot) &&
            snapshot.kind == GuestWaitKind::ConditionSequence)
            return true;
        sleep_ms(1);
    }
    return false;
#else
    return true;
#endif
}

// Park a thread on `target`, free `earlier` while it sleeps, then signal `target`. The signal must
// still land on the slot the sleeper is parked on.
bool signal_survives_slot_reuse(pthread_cond_t* earlier, pthread_cond_t* target,
                                const char* gate_message) {
    g_predicate = 0;
    g_waiter_tid = 0;
    g_woke = 0;

    pthread_t thread{};
    if (pthread_create(&thread, nullptr, wait_on, target) != 0) return false;
    CHECK(await_registered_wait(), gate_message);

    interruptible_cond_forget(earlier);   // frees a slot BEFORE the target's own

    pthread_mutex_lock(&g_mutex);
    g_predicate = 1;
    interruptible_cond_signal(target);
    pthread_mutex_unlock(&g_mutex);

    for (int i = 0; i < kWakeBudgetMs && !g_woke; ++i) sleep_ms(1);
    if (!g_woke) return false;            // do NOT join: unwakeable on the buggy path
    pthread_join(thread, nullptr);
    return true;
}

}  // namespace

int main() {
#ifndef _WIN32
    std::printf("  [skip] the condition-slot table is Windows-only\n");
    std::printf("== PASS ==\n");
    return 0;
#else
    // `early` claims a slot before `target`, so it sits earlier in the table. Signalling with nobody
    // waiting is the cheapest way to force the claim and is otherwise a no-op.
    static pthread_cond_t early = PTHREAD_COND_INITIALIZER;
    static pthread_cond_t target = PTHREAD_COND_INITIALIZER;
    interruptible_cond_signal(&early);
    interruptible_cond_signal(&target);
    const bool first = signal_survives_slot_reuse(
        &early, &target, "the waiter's condition wait is registered before the slot is freed");
    CHECK(first, "a signal still reaches the waiter after an EARLIER slot is freed");

    if (first) {
        // Same property for a condvar that itself claimed a recycled slot. Only meaningful if the
        // case above passed -- otherwise a thread is still parked on the wrong slot.
        static pthread_cond_t filler = PTHREAD_COND_INITIALIZER;
        static pthread_cond_t second_filler = PTHREAD_COND_INITIALIZER;
        static pthread_cond_t recycled = PTHREAD_COND_INITIALIZER;
        interruptible_cond_signal(&filler);
        interruptible_cond_signal(&second_filler);
        interruptible_cond_forget(&filler);       // free a slot...
        interruptible_cond_signal(&recycled);     // ...which `recycled` now claims
        CHECK(signal_survives_slot_reuse(&second_filler, &recycled,
                                         "the second waiter's condition wait is registered"),
              "a condvar holding a recycled slot keeps it when another slot is freed");
    } else {
        std::printf("  [skip] recycled-slot case: a thread is stuck on the wrong slot\n");
    }

    if (fails) {
        std::printf("== FAIL: %d ==\n", fails);
        std::fflush(stdout);
        // A waiter may still be parked on a slot nothing will ever signal, so a normal return would
        // block in thread teardown. Report first, then leave immediately.
        std::_Exit(1);
    }
    std::printf("== PASS ==\n");
    return 0;
#endif
}
