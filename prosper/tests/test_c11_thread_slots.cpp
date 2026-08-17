// test_c11_thread_slots — the C11 `_Mtx_*` / `_Cnd_*` handlers must resolve a guest sync slot the
// same way the libkernel entry point they forward to on hardware resolves it (#2596).
//
// THE DEFECT. Each of these handlers opened with a private `if (a0 && *(void**)a0)` guard, so a slot
// holding a STATIC INITIALISER — NULL for `PTHREAD_MUTEX_INITIALIZER`, and every other sub-page
// sentinel FreeBSD uses — read as "there is no object here" and the operation was SKIPPED. For
// `_Cnd_wait` that means the handler reported success for a wait that never happened: a predicate
// loop is told its condition fired when nothing was ever waited on, so it spins or proceeds on
// unsatisfied state. For `_Mtx_lock` it means the critical section ran unguarded — the same shape as
// the bdwgc `GC_allocate_ml` static-mutex bug that was the ROOT CAUSE of the level-1 heap
// corruption (#793), reached through the other spelling.
//
// WHAT IS NOT THE DEFECT, and this test must not be "fixed" into asserting it: `_Cnd_wait` returning
// 0 unconditionally is CORRECT. The shipped guest libc.prx's own `_Cnd_wait` is
// `push rbp; mov rbp,rsp; call <plt scePthreadCondWait>; xor eax,eax; pop rbp; ret` — it discards
// the result on every path, so an unconditional 0 is the contract rather than a defect (#1983,
// recorded in the code above `k_cond_wait`). Every arm below therefore asserts what the handler DID,
// never what it returned.
//
// WHAT EACH ARM KILLS. Every row below was MEASURED — the mutation applied, rebuilt, re-run — not
// predicted, so a future reader can check they still bite:
//   N1  restore `if (a0 && *(void**)P(a0) && a1 && *(void**)P(a1))` on m_cnd_wait
//                                          -> FAIL (2): §1's "still WAITING 200 ms later" arm and
//                                             its "both slots SELF-INITIALISED" companion. §1's
//                                             wake arm PASSES under N1, because a wait that never
//                                             happened has already "returned".
//   N2  restore the guard on m_mtx_lock    -> FAIL (2): §2's "self-initialised into a real host
//                                             mutex" and "a second thread BLOCKS"
//   N3  make m_cnd_broadcast never broadcast AT ALL (`if (false && a0)`)
//                                          -> FAIL (3): the wake arm of §1 AND of §3 — nobody is
//                                             ever woken, and the bounded retry reports it as a
//                                             failure rather than hanging — PLUS §4's broadcast arm,
//                                             because a broadcast that never fires never resolves
//   N4  make guest_cond_from_slot refuse an ALREADY-INITIALISED slot (sentinels only)
//                                          -> FAIL (2): §1's wake arm and §3's "an initialised
//                                             _Cnd_wait waits too". This is the arm §3 exists for:
//                                             §1 and §2 both pass under it, so without §3 a change
//                                             that fixed the sentinel path by breaking the ordinary
//                                             one would land green.
//   N5  restore the OLD GUARD on m_cnd_broadcast (`if (a0 && *(void**)P(a0))`)
//                                          -> FAIL (1): §4's `_Cnd_broadcast` arm, and ONLY that
//                                             one. §1, §2 and §3 all pass under it — see §4.
//   N6  restore the OLD GUARD on m_mtx_unlock
//                                          -> FAIL (1): §4's `_Mtx_unlock` arm, and only that one.
//   N7  restore the OLD GUARD on m_cnd_signal
//                                          -> FAIL (1): §4's `_Cnd_signal` arm, and only that one.
//                                             `_Cnd_signal` is one of the five handlers this change
//                                             moves and had NO arm at all until a reviewer pointed
//                                             out that restoring its guard failed nothing.
//
// N3 AND N5 ARE DIFFERENT MUTATIONS, and an earlier revision of this header ran N3 while describing
// N5. A reviewer caught the mismatch by reading, and measuring the arm they described produced
// something worse than either of us predicted: with the old guard genuinely restored the suite
// passed COMPLETELY. §4 exists because of that, and the header now names the exact lever each row
// pulls rather than a paraphrase of it — a row describing a mutation nobody ran is how coverage
// that does not exist gets believed (the shape #2573 corrected in its own M5 row).
//
// No single arm covers another: N1/N2 are blind to N3-N6 (all of those leave the wait blocking,
// which is what N1/N2 assert), N3/N4 are blind to N1/N2 (a skipped wait "returns" immediately,
// satisfying every wake arm), and N5/N6 are invisible to every section except §4. N3 kills §4's
// broadcast arm as well as the two wake arms, which is why its count is 3 — a broadcast that never
// fires also never resolves.
//
// HANG SAFETY. Every worker here is DETACHED and every wait is bounded, deliberately: a regression
// in this area is a thread that never wakes, and a test that expresses that as a hang burns ctest's
// timeout and gets reported as an infrastructure problem — on a box that regularly runs at load 30+
// a real regression would then be blamed on machine load. Each arm fails and the process exits.

#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <pthread.h>
#include <thread>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

HleFn mtx_lock = nullptr, mtx_unlock = nullptr, mtx_init = nullptr;
HleFn cnd_wait = nullptr, cnd_broadcast = nullptr, cnd_init = nullptr, cnd_signal = nullptr;

// A statically-initialised guest cnd_t / mtx_t pair: the slot holds NULL, which is exactly what
// `PTHREAD_MUTEX_INITIALIZER` puts there on the guest's platform. File-scope so a detached worker
// can never outlive the storage (the hazard §9 of test_pthread_error_encoding records).
void* g_static_cnd = nullptr;
void* g_static_mtx = nullptr;
std::atomic<int> g_waiter_entered{0};
std::atomic<int> g_wait_returned{0};

void* static_waiter(void*) {
    mtx_lock((uint64_t)(uintptr_t)&g_static_mtx, 0, 0, 0, 0, 0);
    g_waiter_entered.store(1, std::memory_order_release);
    cnd_wait((uint64_t)(uintptr_t)&g_static_cnd, (uint64_t)(uintptr_t)&g_static_mtx, 0, 0, 0, 0);
    g_wait_returned.store(1, std::memory_order_release);
    mtx_unlock((uint64_t)(uintptr_t)&g_static_mtx, 0, 0, 0, 0, 0);
    return nullptr;
}

void* g_contended_mtx = nullptr;
std::atomic<int> g_second_locked{0};

void* second_locker(void*) {
    mtx_lock((uint64_t)(uintptr_t)&g_contended_mtx, 0, 0, 0, 0, 0);
    g_second_locked.store(1, std::memory_order_release);
    mtx_unlock((uint64_t)(uintptr_t)&g_contended_mtx, 0, 0, 0, 0, 0);
    return nullptr;
}

void* g_init_cnd = nullptr;
void* g_init_mtx = nullptr;
std::atomic<int> g_init_entered{0};
std::atomic<int> g_init_returned{0};

void* initialised_waiter(void*) {
    mtx_lock((uint64_t)(uintptr_t)&g_init_mtx, 0, 0, 0, 0, 0);
    g_init_entered.store(1, std::memory_order_release);
    cnd_wait((uint64_t)(uintptr_t)&g_init_cnd, (uint64_t)(uintptr_t)&g_init_mtx, 0, 0, 0, 0);
    g_init_returned.store(1, std::memory_order_release);
    mtx_unlock((uint64_t)(uintptr_t)&g_init_mtx, 0, 0, 0, 0, 0);
    return nullptr;
}

// Deliberately GENEROUS, and only ever used for CONTROLS — "did the worker start", "did it acquire
// after the release". None of them is a timing assertion, so a tight bound here would only turn
// machine load into a red build on a box that routinely runs at load 30+. The one assertion in this
// file that IS about timing — "still waiting 200 ms later" — does not go through this helper.
bool spin_until(const std::atomic<int>& flag, int seconds) {
    for (int i = 0; i < seconds * 1000 && !flag.load(std::memory_order_acquire); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return flag.load(std::memory_order_acquire) != 0;
}

// Broadcast in a bounded retry loop rather than once. A single broadcast races the waiter's own
// entry into the wait, and a lost wakeup would express itself as a HANG — which is the one failure
// mode this file must not have. The retry turns it into a failed assertion.
bool broadcast_until_awake(void* cnd_slot, void* mtx_slot, const std::atomic<int>& flag) {
    for (int i = 0; i < 300 && !flag.load(std::memory_order_acquire); ++i) {
        mtx_lock((uint64_t)(uintptr_t)mtx_slot, 0, 0, 0, 0, 0);
        cnd_broadcast((uint64_t)(uintptr_t)cnd_slot, 0, 0, 0, 0, 0);
        mtx_unlock((uint64_t)(uintptr_t)mtx_slot, 0, 0, 0, 0, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return flag.load(std::memory_order_acquire) != 0;
}

bool spawn_detached(void* (*body)(void*)) {
    pthread_t t{};
    if (pthread_create(&t, nullptr, body, nullptr) != 0) return false;
    pthread_detach(t);
    return true;
}

}  // namespace

int main() {
    printf("== test_c11_thread_slots ==\n");
    register_builtin_hle();

    mtx_init      = Hle::lookup(nid_hash("_Mtx_init"));
    mtx_lock      = Hle::lookup(nid_hash("_Mtx_lock"));
    mtx_unlock    = Hle::lookup(nid_hash("_Mtx_unlock"));
    cnd_init      = Hle::lookup(nid_hash("_Cnd_init"));
    cnd_wait      = Hle::lookup(nid_hash("_Cnd_wait"));
    cnd_broadcast = Hle::lookup(nid_hash("_Cnd_broadcast"));
    cnd_signal    = Hle::lookup(nid_hash("_Cnd_signal"));
    CHECK(mtx_init && mtx_lock && mtx_unlock && cnd_init && cnd_wait && cnd_broadcast && cnd_signal,
          "the C11 _Mtx_* / _Cnd_* handlers are registered");
    if (!(mtx_init && mtx_lock && mtx_unlock && cnd_init && cnd_wait && cnd_broadcast && cnd_signal)) {
        printf("== FAIL (%d) ==\n", ++fails);
        return 1;
    }

    // ===== 1. a _Cnd_wait on a STATICALLY INITIALISED cnd_t must actually wait ==================
    // The arm is "did it block", not "what did it return", because the return is 0 either way and
    // always will be — see the header. Before #2596 the null slot short-circuited the whole body,
    // so the wait returned within microseconds of being entered.
    printf("-- _Cnd_wait on a static cnd_t waits instead of returning immediately --\n");
    {
        CHECK(spawn_detached(static_waiter), "control: the waiter thread starts");
        CHECK(spin_until(g_waiter_entered, 5), "control: the waiter reached its _Cnd_wait");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        CHECK(g_wait_returned.load(std::memory_order_acquire) == 0,
              "_Cnd_wait on a statically-initialised cnd_t is still WAITING 200 ms later -- it used "
              "to skip the wait entirely and report success for a wait that never happened");
        CHECK(g_static_cnd != nullptr && g_static_mtx != nullptr,
              "...and both slots SELF-INITIALISED: each now names a real host object, the same way "
              "scePthreadCondWait resolves them (#793 / #2170), which is the mechanism the arm "
              "above observes rather than a second assertion of it");
        CHECK(broadcast_until_awake(&g_static_cnd, &g_static_mtx, g_wait_returned),
              "...and the wait RETURNS once the condition is broadcast (a wait that blocks forever "
              "would be a worse defect than the one being fixed)");
    }

    // ===== 2. _Mtx_lock on a static mtx_t must really take the lock ============================
    // The same guard, on the handler where skipping it is the #793 heap-corruption shape: a guest
    // whose lock silently does nothing runs its critical section unguarded and reports success.
    printf("-- _Mtx_lock on a static mtx_t really excludes a second thread --\n");
    {
        mtx_lock((uint64_t)(uintptr_t)&g_contended_mtx, 0, 0, 0, 0, 0);
        CHECK(g_contended_mtx != nullptr,
              "control: the static slot self-initialised into a real host mutex");
        CHECK(spawn_detached(second_locker), "control: a second locker thread starts");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        CHECK(g_second_locked.load(std::memory_order_acquire) == 0,
              "a second thread's _Mtx_lock BLOCKS while this one holds it -- it used to no-op and "
              "let both threads into the critical section");
        mtx_unlock((uint64_t)(uintptr_t)&g_contended_mtx, 0, 0, 0, 0, 0);
        CHECK(spin_until(g_second_locked, 5),
              "...and it acquires as soon as the holder releases");
    }

    // ===== 3. the ORDINARY initialised path is unchanged =======================================
    // A regression guard rather than a discriminator for #2596: these slots never hold a sentinel,
    // so they exercise the branch the change does not touch. It is here because the obvious way to
    // get §1 and §2 green is to make the handlers unconditional, and that would break nothing
    // visible without this section.
    printf("-- _Mtx_init / _Cnd_init objects still lock, wait and wake --\n");
    {
        CHECK(mtx_init((uint64_t)(uintptr_t)&g_init_mtx, 0, 0, 0, 0, 0) == 0 && g_init_mtx,
              "control: _Mtx_init produces an object");
        CHECK(cnd_init((uint64_t)(uintptr_t)&g_init_cnd, 0, 0, 0, 0, 0) == 0 && g_init_cnd,
              "control: _Cnd_init produces an object");
        CHECK(spawn_detached(initialised_waiter), "control: the waiter thread starts");
        CHECK(spin_until(g_init_entered, 5), "control: the waiter reached its _Cnd_wait");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        CHECK(g_init_returned.load(std::memory_order_acquire) == 0,
              "an initialised _Cnd_wait waits too");
        CHECK(broadcast_until_awake(&g_init_cnd, &g_init_mtx, g_init_returned),
              "...and wakes on _Cnd_broadcast");
    }

    // ===== 4. the SIGNAL side, on a slot no wait has resolved yet ==============================
    // THIS SECTION EXISTS BECAUSE ITS ABSENCE WAS INVISIBLE. A reviewer questioned the N3 row and
    // derived that restoring the old guard on `m_cnd_broadcast` should fail only §1's wake arm.
    // Measuring it produced something worse than either reading: restoring the guard made the suite
    // pass COMPLETELY. The reason is structural — §1 and §3 both WAIT before they broadcast, and the
    // wait self-initialises the slot, so by the time any broadcast runs every slot in this file
    // already holds a real pointer and the old guard is satisfied. Those sections cannot see the
    // signal/broadcast half of the change at all, however the mutation is phrased. A positive
    // control that has already resolved the thing under test is not a control.
    //
    // WHAT THESE THREE ARMS ARE FOR, stated carefully because the first revision of this block gave
    // a reason that CANNOT HAPPEN. It said a skipped signal "would be signalling an object no waiter
    // will ever use". There is no such waiter: `ensure_cond` installs the object with a CAS before
    // returning it, and `m_cnd_wait` can only park on what it returns, so a slot still holding a
    // sentinel has nobody parked on it by construction. On a NULL sentinel a skipped broadcast
    // therefore loses NOTHING. A second reviewer caught that, and it is recorded rather than quietly
    // replaced because a plausible-but-impossible failure mode in a test's rationale is exactly what
    // stops the next reader questioning the arm.
    //
    // WHAT THE THREE ARMS ESTABLISH -- one thing, not two:
    //   CROSS-SPELLING PARITY, which is this PR's whole thesis (#1873). `k_cond_signal`,
    //   `k_cond_broadcast` and `k_mutex_unlock` resolve the identical guest slot for the identical
    //   input. §4 is the only place in the suite that asserts the C11 spelling agrees, and the only
    //   place the resolver is exercised on the signal side at all.
    //
    // WHAT THEY DO NOT ESTABLISH, though the change fixes it: THE CRASH PATH. The old guard tested
    // the slot's VALUE, so it was TRUE for every sentinel except NULL -- sentinel 1 and the destroyed
    // poison kPtDestroyed (0xDEA) were dereferenced as object pointers, and routing through the
    // resolver turns `interruptible_cond_broadcast((pthread_cond_t*)0xDEA)` into a refusal. Every
    // clause of that is true and NO ARM BELOW SHOWS IT: all three slots here hold nullptr, so nothing
    // in this file exercises a non-NULL sentinel. Correctness by construction, not by measurement.
    // An arm that stored `(void*)1` would not fix that -- under the mutation it would SIGSEGV rather
    // than fail an assertion, which is a worse instrument than none. The third reviewer caught this
    // attribution, and it is the SAME class as the retraction three paragraphs up: a true statement
    // filed under an arm that cannot support it.
    // The observable is the resolution itself, because a broadcast to a condvar with no waiters has
    // no other effect to assert.
    printf("-- _Cnd_signal / _Cnd_broadcast / _Mtx_unlock resolve a slot no wait has touched --\n");
    {
        static void* untouched_signal_cnd = nullptr;   // static sentinels, never waited on
        static void* untouched_bcast_cnd = nullptr;
        static void* untouched_mtx = nullptr;
        // `_Cnd_signal` gets its own slot rather than sharing the broadcast one: it is one of the
        // five handlers this change moves, and without an arm of its own restoring its old guard
        // would fail nothing in this file.
        cnd_signal((uint64_t)(uintptr_t)&untouched_signal_cnd, 0, 0, 0, 0, 0);
        CHECK(untouched_signal_cnd != nullptr,
              "_Cnd_signal RESOLVES a statically-initialised cnd_t rather than skipping it -- the "
              "same slot, and the same answer, scePthreadCondSignal gives for this input");
        cnd_broadcast((uint64_t)(uintptr_t)&untouched_bcast_cnd, 0, 0, 0, 0, 0);
        CHECK(untouched_bcast_cnd != nullptr,
              "_Cnd_broadcast likewise resolves rather than skipping");
        // NOTE for anyone reading this arm as evidence of more than it shows: unlocking a mutex this
        // thread never locked is UNDEFINED under POSIX for a non-ERRORCHECK mutex. It is benign on
        // glibc, and it is exactly what k_mutex_unlock already does for the same input, which is why
        // it is acceptable here — but the arm asserts that the slot was RESOLVED, not that the
        // unlock is defined.
        mtx_unlock((uint64_t)(uintptr_t)&untouched_mtx, 0, 0, 0, 0, 0);
        CHECK(untouched_mtx != nullptr,
              "_Mtx_unlock likewise resolves rather than skipping, so the C11 family agrees with "
              "itself about what a static sentinel means");
    }

    if (fails) printf("== FAIL (%d) ==\n", fails);
    else       printf("== PASS ==\n");
    return fails ? 1 : 0;
}
