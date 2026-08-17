// test_pthread_error_encoding — #2178's sweep, made executable.
//
// THE RULE. A `scePthread*` handler goes through `SCE_PTHREAD_ALIAS` (or encodes in place) **if and
// only if** its body can fail. The alias wraps the body in `sce_pthread_rc`, which passes 0 and
// already-encoded values through and otherwise returns `0x80020000 | errno`. A Sony-spelled entry
// point registered straight onto a fallible body hands the guest a **bare FreeBSD errno** instead —
// #1984's defect, the one that killed The Oregon Trail until 38619f29, and the one #2158 fixed for
// `scePthreadRwlockRdlock`/`Wrlock` alone.
//
// WHY A SINGLE-SPELLING ASSERTION CANNOT SEE THIS, which is the whole reason this file exists.
// The interesting failure is not "the handler returned zero when it should have failed" — every
// row below returns *something* non-zero either way. It is that the VALUE is wrong. So an arm
// written as `CHECK(sony(...) != 0)` passes identically under the bug and under the fix, and an arm
// that checks only the Sony spelling cannot distinguish a correctly aliased handler from one wired
// straight to the POSIX body when the two happen to agree. Every dual-registered row therefore
// asserts BOTH spellings, and asserts them to *different* values: Sony encoded, POSIX bare. That
// pairing is the discriminator. Nothing weaker is.
//
// WHAT EACH ARM KILLS. Three mutations, named here so a future reader can check they still bite:
//   M1  drop the alias (register the Sony name onto the POSIX body)      -> Sony arm fails, POSIX passes
//   M2  alias BOTH spellings (encode the POSIX one too)                  -> POSIX arm fails, Sony passes
//   M3  return the host errno instead of the FreeBSD one (#1612)         -> the EAGAIN row fails on both
//   M4  apply the encoding to a non-error return                         -> the barrier serial-thread arm fails
//
//   M5  make the CondWait body report 0 again (discard its result)       -> its kRows row fails on
//                                                                           BOTH spellings (#1983)
//   M6  make the CondWait body REFUSE unconditionally                    -> §7's signalled-wait
//                                                                           control fails (#1983)
//
//   M7  drop SCE_PTHREAD_ALIAS(k_sce_pthread_join, …)                    -> §9's Sony self-join arm
//   M8  alias `pthread_join` as well                                     -> §9's POSIX self-join arm
//   M9  make k_pthread_join report 0 again (discard its result)          -> §9's self-join arms, both
//   M10 write value_ptr on `if (a1)` rather than `if (rc == 0 && a1)`    -> §9's untouched arm ONLY
//   M11 return the raw HOST errno from k_pthread_join instead of mapping -> §9's self-join arms, BOTH
//   M12 drop the value_ptr write entirely                                -> §9's exit-value control ONLY
//   M13 drop SCE_PTHREAD_ALIAS(k_sce_pthread_detach, …)                  -> §9's Sony detach arm
//   M14 make k_pthread_detach report 0 again                             -> §9's detach arms
//   M15 make k_pthread_detach report 0 WITHOUT calling through           -> §9's "it REALLY detached"
//                                                                           control, plus both detach
//                                                                           arms
//
// M11 kills BOTH self-join arms, not just the POSIX one, and an earlier revision of this table said
// otherwise while the PR body's measured row said `FAIL (2)`. The reason it must: an unmapped host
// EDEADLK is 35, and `sce_pthread_rc(35)` is 0x80020023 — encoded FreeBSD EAGAIN — so the Sony arm
// sees a wrong-but-encoded value and fails too. Caught in review; the same class this file corrects
// five lines above for M5/M6, which is why it is stated rather than quietly fixed.
//
// M7-M15 are the #2575 set and they are NOT interchangeable, for the same reason M5 and M6 are not.
// M10 and M12 are invisible to every return-value arm in the file — a body that reports EDEADLK
// correctly and still nulls (or never writes) the guest's `value_ptr` satisfies all of them — and
// conversely the value_ptr arms cannot see M7/M8/M11/M13, which are all about the number. M11 is the
// only one that can tell a FreeBSD errno from a host one, because EDEADLK is the ONLY errno reachable
// in §9 where the two numberings disagree (11 on FreeBSD, 35 on Linux, and 35 is FreeBSD's EAGAIN).
//
// M9 AND M14 ARE THE ARMS THAT REVERT THE DEFECT ITSELF, and the first draft of §9 could not see
// either. Its skip guards asked the HANDLER whether the host refuses a self-join / a second detach;
// under M9 and M14 the handler answers 0, the guard read that as "this host permits it" and skipped,
// and the section passed under the exact bug it exists to catch. Both guards now come from a direct
// host call on a thread the handler never touches. Recorded because the failure was silent and
// green: the same "a positive control drawn from the same source as the null tests the
// DISCRIMINATOR, never the DOMAIN" shape the charter records, inside a mutation harness.
//
// M5 and M6 are COMPLEMENTARY, and an earlier revision of this header got it wrong in the way that
// matters most here: it claimed M5 kills §7's positive control too. It cannot. M5 makes the body
// return 0 and §7 asserts 0, so §7 passes under M5 — the measured `FAIL (2)` is the kRows row's two
// halves, Sony and POSIX, not §7. Conversely no kRows row can see M6: they all provoke a refusal, so
// a body that refuses unconditionally satisfies every one of them. Neither arm covers the other, and
// a row here claiming otherwise is worse than no row at all, because this header is exactly where a
// future reader checks whether an arm still bites — and a false claim of coverage stops them looking.
//
// The M3 and M4 arms matter most. M1/M2 are provoked with a null object, which yields EINVAL — and
// EINVAL is 22 on FreeBSD, on Linux, on Darwin and on MinGW, so a null-slot arm alone cannot tell a
// FreeBSD number from a host number. The EAGAIN row can: FreeBSD EAGAIN is 35 while the host's is
// 11, and 11 is FreeBSD's EDEADLK, so "would block, retry" arrives as "deadlock" if the mapping is
// skipped. The M4 arm guards the opposite error — `scePthreadBarrierWait` returns -1 for the serial
// thread, which is not an errno at all, and over-applying the encoding would corrupt it.

#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/sce_errno.hpp"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <pthread.h>
#include <thread>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

// Written as literals on purpose. Deriving them from sce_errno.hpp would make the arms agree with
// whatever the header says, including a mutated header — these are the numbers the guest's own
// libc.prx compares against, so they are pinned here independently.
constexpr uint64_t kEncodedEINVAL = 0x80020016ull;
constexpr uint64_t kBareEINVAL    = 22;              // FreeBSD EINVAL == every host's EINVAL
static_assert(kEncodedEINVAL ==
                  prosper::hle::sce_kernel_error(prosper::hle::FreeBsdErrno::EInval),
              "the literal and the encoder must agree; if this fires, one of them moved");
static_assert(kBareEINVAL == static_cast<uint64_t>(prosper::hle::FreeBsdErrno::EInval),
              "FreeBSD EINVAL is 22");

uint64_t call0(HleFn f) { return f(0, 0, 0, 0, 0, 0); }

// One row of the sweep: a Sony spelling, and the POSIX spelling registered on the same body (or
// null when the body carries no POSIX name, in which case the Sony spelling encodes in place).
struct Row {
    const char* sony;
    const char* posix;    // nullptr = Sony-only body
    const char* note;
    // The POSIX spelling's RETURN CONVENTION, which is not uniform across these families (#2182).
    // pthread_* genuinely returns the error number, so those rows assert a bare errno. The sem_*
    // family does not: C11 7.26 and FreeBSD sem_wait(3) specify "return -1, number left in errno",
    // so those rows assert -1 AND read the number back out of errno. Getting this wrong in either
    // direction is silent at the guest, which is why it is a per-row property and not a global one.
    bool posix_returns_minus1 = false;
};

// Every fallible entry point in the synchronisation-object families, in registration order. A null
// object provokes EINVAL in all of them (each body opens with a `!a0` / `ensure_*` guard), so one
// zero-argument call exercises every row uniformly.
const Row kRows[] = {
    // --- thread attributes: the two stack setters (#2183) ---
    // scePthreadAttrSetstacksize was previously INFALLIBLE, and for the wrong reason: it swallowed
    // a null attr and an undersized request and reported success, while scePthreadAttrSetstack
    // rejected the same three conditions. Making it fallible and giving it its alias are one
    // change, not two -- splitting them would have started handing the Sony spelling a bare errno.
    {"scePthreadAttrSetstacksize",   "pthread_attr_setstacksize",   "#2183"},
    {"scePthreadAttrSetstack",       "pthread_attr_setstack",       "#2183, alias was missing"},
    // --- mutex attributes ---
    {"scePthreadMutexattrInit",      "pthread_mutexattr_init",      "#2178"},
    {"scePthreadMutexattrSettype",   "pthread_mutexattr_settype",   "#2178"},
    {"scePthreadMutexattrGettype",   nullptr,                       "#2178, in place"},
    // --- mutex ---
    {"scePthreadMutexInit",          "pthread_mutex_init",          "#2178"},
    {"scePthreadMutexLock",          "pthread_mutex_lock",          "#1945, regression guard"},
    {"scePthreadMutexTrylock",       "pthread_mutex_trylock",       "#1945, regression guard"},
    {"scePthreadMutexTimedlock",     nullptr,                       "#1945, in place"},
    {"scePthreadMutexUnlock",        "pthread_mutex_unlock",        "#2178, the sharpest row"},
    // --- condition-variable attributes ---
    {"scePthreadCondattrInit",       "pthread_condattr_init",       "#2178"},
    {"scePthreadCondattrSetclock",   "pthread_condattr_setclock",   "#2178"},
    {"scePthreadCondattrGetclock",   "pthread_condattr_getclock",   "#2178"},
    {"scePthreadCondattrSetpshared", "pthread_condattr_setpshared", "#2178"},
    {"scePthreadCondattrGetpshared", "pthread_condattr_getpshared", "#2178"},
    // --- condition variables ---
    {"scePthreadCondInit",           "pthread_cond_init",           "#2178"},
    {"scePthreadCondTimedwait",      nullptr,                       "#2178, in place"},
    // Moved OUT of kInfallible by #1983. It was correctly listed there when this file was written:
    // the body discarded its wait result and returned 0 on every path, including the path where a
    // null slot meant no wait was attempted at all. That is a silent success, strictly worse than
    // the bare-errno defect this file sweeps for, because a guest testing `rc == 0` is misled too.
    // Now the body reports the wait and refuses an unusable pair, so the alias rule applies to it.
    {"scePthreadCondWait",           "pthread_cond_wait",           "#1983, was infallible"},
    // --- read/write locks (already correct: #1984 / #2158 — these are regression guards) ---
    {"scePthreadRwlockInit",         "pthread_rwlock_init",         "#1984"},
    {"scePthreadRwlockRdlock",       "pthread_rwlock_rdlock",       "#2158"},
    {"scePthreadRwlockWrlock",       "pthread_rwlock_wrlock",       "#2158"},
    // Re-added as the note above instructed: #2159 has landed, so a null slot now answers EINVAL
    // here as it already did at every sibling entry point, and the row asserts the pairing on a real
    // failure rather than enshrining the old 0. Before the fix this row is red for exactly the right
    // reason — the Sony spelling returned 0 instead of the encoded 0x80020016.
    {"scePthreadRwlockUnlock",       "pthread_rwlock_unlock",       "#2159"},
    {"scePthreadRwlockTryrdlock",    "pthread_rwlock_tryrdlock",    "#1984"},
    {"scePthreadRwlockTrywrlock",    "pthread_rwlock_trywrlock",    "#1984"},
    {"scePthreadRwlockTimedrdlock",  "pthread_rwlock_timedrdlock",  "#1984, separate POSIX body"},
    {"scePthreadRwlockTimedwrlock",  "pthread_rwlock_timedwrlock",  "#1984, separate POSIX body"},
    // --- read/write lock attributes ---
    {"scePthreadRwlockattrInit",     "pthread_rwlockattr_init",     "#2178"},
    {"scePthreadRwlockattrDestroy",  "pthread_rwlockattr_destroy",  "#2178"},
    // --- semaphores ---
    {"scePthreadSemInit",            "sem_init",                    "#2178", true},
    {"scePthreadSemWait",            "sem_wait",                    "#2178", true},
    {"scePthreadSemTrywait",         "sem_trywait",                 "#2178", true},
    {"scePthreadSemTimedwait",       nullptr,                       "#2178, in place"},
    {"scePthreadSemPost",            "sem_post",                    "#2178", true},
    {"scePthreadSemGetvalue",        "sem_getvalue",                "#2178", true},
    // --- barriers (no POSIX spelling registered on either body) ---
    {"scePthreadBarrierInit",        nullptr,                       "#2178, in place"},
    {"scePthreadBarrierWait",        nullptr,                       "#2178, in place"},
    // scePthreadJoin / scePthreadDetach became fallible in #2575 and so now need the alias, but they
    // deliberately have NO row here and that is not an oversight: kRows provokes its refusal with an
    // all-zero call, and `pthread_join(0, …)` is not a defined refusal. A `pthread_t` is an opaque
    // descriptor POINTER on both hosts prosper builds for, so a null one is dereferenced before any
    // validity check — the sweep would segfault rather than report EINVAL. Their arms are §9, where
    // the failure comes from a real thread.
};

// --- §9 thread bodies. Plain HOST pthreads on purpose: these arms are about what k_pthread_join and
// k_pthread_detach do with the host result, so nothing about prosper's own create trampoline is
// under test and routing through it would only add failure modes the arms cannot attribute.
void* join_worker(void* arg) { return arg; }        // hands its own argument back as the exit value

// The park gate and its counters are FILE-SCOPE STATICS, not locals of main, and that is a
// correctness requirement rather than a style choice. §9's workers are detached, so nothing joins
// them; a gate living in main's frame would be touched by a worker still inside pthread_mutex_lock
// after that frame had gone away. It reproduced: the first mutation run aborted with SIGABRT in one
// arm out of eight, which is exactly how this presents — intermittently, in whichever arm happens to
// lose the race, and attributable to anything. §9 also waits for every worker to finish below, so
// the statics are belt and braces.
pthread_mutex_t g_park_gate = PTHREAD_MUTEX_INITIALIZER;
std::atomic<int> g_park_running{0};
std::atomic<int> g_park_done{0};
void* park_worker(void*) {
    g_park_running.fetch_add(1, std::memory_order_release);
    pthread_mutex_lock(&g_park_gate);      // main holds this for the whole of §9's detach block
    pthread_mutex_unlock(&g_park_gate);
    g_park_done.fetch_add(1, std::memory_order_release);
    return nullptr;
}
// Bounded spin so a lost wake fails the run instead of burning ctest's timeout, which reports as an
// infrastructure problem and gets blamed on machine load (the hazard #1983's review caught in §7).
bool wait_for(const std::atomic<int>& counter, int target) {
    for (int i = 0; i < 20000 && counter.load(std::memory_order_acquire) < target; ++i)
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    return counter.load(std::memory_order_acquire) >= target;
}

// Entry points whose body returns 0 on every path. They correctly have NO alias, and #2158
// established that its absence is the right answer rather than an oversight.
const char* const kInfallible[] = {
    "scePthreadMutexattrSetprotocol", "scePthreadMutexattrSetpshared", "scePthreadMutexattrDestroy",
    "scePthreadMutexDestroy", "scePthreadCondattrDestroy",
    "scePthreadCondSignal", "scePthreadCondBroadcast",
    "scePthreadRwlockDestroy", "scePthreadSemDestroy",
    "scePthreadBarrierattrInit", "scePthreadBarrierattrDestroy", "scePthreadBarrierattrSetpshared",
};

// Bodies that CAN fail, but answer 0 for a NULL slot — so the sweep's null probe cannot reach their
// failure path and they do not belong in either list above.
//
// They were listed as infallible when this file was written, and that was true then. #2359 (condvar)
// and #2379 (barrier) landed #2168's busy-destroy contract: each now refuses an object that still
// has waiters, and the guest sees `SCE_KERNEL_ERROR_EBUSY` (0x80020010) from BOTH Sony spellings.
// They get there by opposite routes, and the difference is the axis #2182 polices, so do not
// collapse it:
//   * `k_barrier_destroy` encodes IN PLACE (hle_kernel.cpp:3045). It is Sony-only — nothing
//     registers `pthread_barrier_destroy` onto that body — so there is no POSIX caller to mislead.
//   * `k_cond_destroy` returns the BARE FreeBSD errno (hle_kernel.cpp:1033-1034) precisely BECAUSE
//     `pthread_cond_destroy` is registered onto the same body (:4810). The Sony spelling gets its
//     0x8002xxxx from the `SCE_PTHREAD_ALIAS` at :1049. Making this one encode in place would hand
//     every `pthread_cond_destroy` caller a Sony-encoded value — #1984's shape — and
//     `test_cond_destroy_busy.cpp` asserts the bare 16 for exactly that reason.
// The `kInfallible` ARM still passed unchanged after both landings, because a null slot never
// reaches the busy check — so nothing failed, and the list's stated rationale ("returns 0 on every
// path") silently went false while the assertion stayed green. That is the failure this split
// exists to prevent: a passing arm is not evidence that the sentence justifying it is still true.
//
// The busy path itself is covered where it can actually be constructed — a thread must really be
// parked in the object — by `test_cond_destroy_busy.cpp` and `test_barrier_destroy_busy.cpp`. What
// is pinned HERE is only the null-slot half: these report success for an absent object rather than
// inventing a failure. That is all this arm can see: `sce_pthread_rc(0)` is 0, so it reads the same
// through the condvar's existing alias as it would through the barrier's direct registration, and
// it would not notice an alias being added to or removed from either.
// #2168's remaining third, `scePthreadMutexDestroy`, is still genuinely infallible above: it needs
// owner tracking that deliberately does not exist on POSIX hosts (#719/#793).
const char* const kFallibleButNullSucceeds[] = {
    "scePthreadCondDestroy",      // #2359: EBUSY with waiters — bare from the body, encoded by the alias
    "scePthreadBarrierDestroy",   // #2379: EBUSY with threads parked — encoded in the body itself
};

}  // namespace

int main() {
    printf("== test_pthread_error_encoding ==\n");
    register_builtin_hle();

    // ===== 1. the sweep: Sony encoded, POSIX bare, one row per fallible entry point ============
    printf("-- the alias rule, per entry point (null object -> EINVAL) --\n");
    for (const Row& r : kRows) {
        char msg[256];
        HleFn sony = Hle::lookup(nid_hash(r.sony));
        if (!sony) {
            snprintf(msg, sizeof msg, "%s is registered", r.sony);
            CHECK(false, msg);
            continue;
        }
        const uint64_t got = call0(sony);
        snprintf(msg, sizeof msg, "%-30s -> encoded EINVAL 0x80020016 (got 0x%llx)  [%s]",
                 r.sony, (unsigned long long)got, r.note);
        CHECK(got == kEncodedEINVAL, msg);

        if (!r.posix) continue;
        HleFn posix = Hle::lookup(nid_hash(r.posix));
        if (!posix) {
            snprintf(msg, sizeof msg, "%s is registered", r.posix);
            CHECK(false, msg);
            continue;
        }
        // The half a single-spelling test cannot do: the POSIX name on the same body must NOT
        // encode. Without this line, aliasing both spellings would look like a clean pass.
        errno = 0;
        const uint64_t got_posix = call0(posix);
        if (r.posix_returns_minus1) {
            snprintf(msg, sizeof msg,
                     "%-30s -> -1 with EINVAL 22 in errno (got %lld, errno %d)",
                     r.posix, (long long)(int64_t)got_posix, errno);
            CHECK(got_posix == (uint64_t)(int64_t)-1 && errno == (int)kBareEINVAL, msg);
        } else {
            snprintf(msg, sizeof msg, "%-30s -> bare EINVAL 22 (got %llu)",
                     r.posix, (unsigned long long)got_posix);
            CHECK(got_posix == kBareEINVAL, msg);
        }
    }

    // ===== 2. M3: a forwarded HOST errno, where host and FreeBSD numbering disagree ============
    // scePthreadSemTrywait on an empty semaphore fails with EAGAIN. FreeBSD EAGAIN is 35; the host's
    // is 11, which is FreeBSD's EDEADLK. This row is the only one that can tell a correct mapping
    // from a passed-through host number, because every other row's errno is EINVAL(22), which every
    // platform agrees on. It also carries the full contract at once: Sony gets 0x80020023, POSIX
    // gets -1 with 35 in errno, and neither gets 11. Since #2182 the POSIX half is the strongest
    // arm in this file: it is the only place that can show WHICH numbering reaches the guest's
    // errno, because it is the only errno in the suite where host and FreeBSD disagree.
    printf("-- a forwarded host errno (EAGAIN: FreeBSD 35, host 11) --\n");
    {
        HleFn sem_init    = Hle::lookup(nid_hash("scePthreadSemInit"));
        HleFn sem_trywait = Hle::lookup(nid_hash("scePthreadSemTrywait"));
        HleFn sem_destroy = Hle::lookup(nid_hash("scePthreadSemDestroy"));
        HleFn posix_trywait = Hle::lookup(nid_hash("sem_trywait"));
        CHECK(sem_init && sem_trywait && sem_destroy && posix_trywait,
              "scePthreadSem* and sem_trywait are registered");
        if (sem_init && sem_trywait && sem_destroy && posix_trywait) {
            alignas(16) uint8_t slot[32]{};
            CHECK(sem_init((uint64_t)(uintptr_t)slot, 0, 0, 0, 0, 0) == 0,
                  "a zero-count guest semaphore initializes");
            CHECK(sem_trywait((uint64_t)(uintptr_t)slot, 0, 0, 0, 0, 0) == 0x80020023ull,
                  "scePthreadSemTrywait on an empty semaphore reports encoded FreeBSD EAGAIN "
                  "(0x80020023) -- not a bare 35, and not the host's 11");
            // The POSIX half, and the sharpest arm in the file: -1 is the C contract, and the
            // 35 in errno is FreeBSD's EAGAIN where the host's is 11. An implementation that
            // published the host number would still return -1 here and would still look right to
            // any check that only tested the return value (#2182).
            errno = 0;
            const uint64_t posix_rc = posix_trywait((uint64_t)(uintptr_t)slot, 0, 0, 0, 0, 0);
            CHECK(posix_rc == (uint64_t)(int64_t)-1,
                  "sem_trywait on the same semaphore returns -1, per C11 7.26 / sem_wait(3)");
            CHECK(errno == 35,
                  "sem_trywait leaves FreeBSD EAGAIN (35) in errno -- not the host's 11, which is "
                  "FreeBSD's EDEADLK and would turn a retryable wait into a deadlock report");
            sem_destroy((uint64_t)(uintptr_t)slot, 0, 0, 0, 0, 0);
        }
    }

    // ===== 3. the sharpest row, on a REAL failure rather than a null =========================
    // scePthreadMutexLock/Trylock/Timedlock all encoded and Unlock did not, so the answer a guest
    // got to "did this mutex operation fail, and how" depended on which of the four it called.
    // A guest mutex initialized with no attribute is ERRORCHECK (Sony's default), so releasing one
    // this thread does not hold is refused with EPERM instead of being undefined behaviour.
    printf("-- scePthreadMutexUnlock on a mutex this thread does not hold (EPERM) --\n");
    {
        HleFn mtx_init   = Hle::lookup(nid_hash("scePthreadMutexInit"));
        HleFn mtx_unlock = Hle::lookup(nid_hash("scePthreadMutexUnlock"));
        HleFn mtx_lock   = Hle::lookup(nid_hash("scePthreadMutexLock"));
        HleFn mtx_destroy = Hle::lookup(nid_hash("scePthreadMutexDestroy"));
        HleFn posix_unlock = Hle::lookup(nid_hash("pthread_mutex_unlock"));
        CHECK(mtx_init && mtx_unlock && mtx_lock && mtx_destroy && posix_unlock,
              "scePthreadMutex* and pthread_mutex_unlock are registered");
        if (mtx_init && mtx_unlock && mtx_lock && mtx_destroy && posix_unlock) {
            alignas(16) void* slot = nullptr;
            const uint64_t s = (uint64_t)(uintptr_t)&slot;
            CHECK(mtx_init(s, 0, 0, 0, 0, 0) == 0, "guest mutex initializes (ERRORCHECK by default)");
            const uint64_t sony_rc = mtx_unlock(s, 0, 0, 0, 0, 0);
            if (sony_rc == 0) {
                // A host whose ERRORCHECK unlock is permissive cannot run this arm. Say so rather
                // than passing vacuously — an assertion that held while the mechanism never ran is
                // the trap this repository keeps rediscovering.
                printf("  [SKIP] this host permits unlocking an unheld ERRORCHECK mutex -- the"
                       " EPERM half of the #2178 mutex arm cannot run here\n");
            } else {
                CHECK(sony_rc == 0x80020001ull,
                      "scePthreadMutexUnlock of an unheld mutex reports encoded EPERM (0x80020001)");
                CHECK(posix_unlock(s, 0, 0, 0, 0, 0) == 1,
                      "pthread_mutex_unlock of the same mutex reports bare EPERM (1)");
                // The pair above is the #1873 shape closed: ask the same question through the two
                // spellings of the SAME family and the answers now differ only in encoding.
                CHECK(mtx_lock(s, 0, 0, 0, 0, 0) == 0, "the mutex still locks after the refusals");
                CHECK(mtx_unlock(s, 0, 0, 0, 0, 0) == 0, "and the matched unlock still returns 0");
            }
            mtx_destroy(s, 0, 0, 0, 0, 0);
        }
    }

    // ===== 4. M4: the encoding must NOT eat a non-error return ================================
    // scePthreadBarrierWait answers -1 (PTHREAD_BARRIER_SERIAL_THREAD) to exactly one of the
    // arriving threads. That is a sentinel, not an errno, and `sce_pthread_rc` passes it through
    // only because its high bits are set. This arm is what stops the sweep from being applied by
    // reflex to a handler whose return is a value: over-encode it and the guest's designated
    // serial thread stops recognising itself.
    printf("-- the barrier serial-thread sentinel survives the encoding --\n");
    {
        HleFn b_init = Hle::lookup(nid_hash("scePthreadBarrierInit"));
        HleFn b_wait = Hle::lookup(nid_hash("scePthreadBarrierWait"));
        HleFn b_destroy = Hle::lookup(nid_hash("scePthreadBarrierDestroy"));
        CHECK(b_init && b_wait && b_destroy, "scePthreadBarrier* are registered");
        if (b_init && b_wait && b_destroy) {
            alignas(16) void* slot = nullptr;
            const uint64_t s = (uint64_t)(uintptr_t)&slot;
            // count 1: this thread is the whole rendezvous, so it is the serial thread.
            CHECK(b_init(s, 0, 1, 0, 0, 0) == 0, "a one-participant barrier initializes");
            const uint64_t rc = b_wait(s, 0, 0, 0, 0, 0);
            CHECK(rc == (uint64_t)(int64_t)-1,
                  "scePthreadBarrierWait still returns -1 to the serial thread, unencoded");
            b_destroy(s, 0, 0, 0, 0, 0);
        }
    }

    // ===== 5. the "iff" is two-sided: three handlers that must NEVER be encoded ===============
    // These return a VALUE, not an error code. Aliasing scePthreadEqual would turn its `1` into
    // 0x80020001 — "equal" delivered as EPERM — and aliasing Self or Getspecific would mangle any
    // pointer-sized result whose top bits happen to be clear. The rule is "alias iff fallible"
    // precisely so that a mechanical sweep cannot reach them.
    printf("-- entry points that return a value, not an error code --\n");
    {
        HleFn eq   = Hle::lookup(nid_hash("scePthreadEqual"));
        HleFn self = Hle::lookup(nid_hash("scePthreadSelf"));
        CHECK(eq && self, "scePthreadEqual / scePthreadSelf are registered");
        if (eq && self) {
            CHECK(eq(0x1234, 0x1234, 0, 0, 0, 0) == 1,
                  "scePthreadEqual answers 1 for equal ids (NOT 0x80020001)");
            CHECK(eq(0x1234, 0x5678, 0, 0, 0, 0) == 0, "scePthreadEqual answers 0 for different ids");
            const uint64_t me = call0(self);
            CHECK(me != 0 && (me & ~0xffull) != 0x80020000ull,
                  "scePthreadSelf returns a thread id, not an encoded error");
        }
    }

    // ===== 6. #1983: the TLS-key pair, which cannot use the null-object row convention ==========
    // kRows provokes EINVAL with an all-zero call. That is unsafe here: a key argument of 0 is a
    // plausible LIVE key in this process (the C++ runtime allocates keys before main), so a null-arg
    // row could delete somebody else's key and pass by accident. Use an index far above any
    // implementation's table instead — every host range-checks before touching anything, so the
    // refusal is defined behaviour rather than a lucky read.
    //
    // These two were #2178's "raw host errno first" rows: the body returned the HOST number through
    // BOTH spellings, so aliasing alone would have produced `0x80020000 | wrong_errno`. The
    // fbsd_errno mapping and the alias therefore land together, as #2189 did for k_mutexattr_init.
    //
    // The C11 layer is what makes the pairing concrete rather than merely symmetric: in the shipped
    // guest libc.prx (PPSA24651), `_Tss_create`, `_Tss_delete` and `_Tss_set` are 11-byte,
    // five-instruction forwarders onto scePthreadKeyCreate / KeyDelete / Setspecific --
    // `push rbp; mov rbp,rsp; call <plt>; pop rbp; ret`, returning eax unexamined (resolved through
    // the PLT relocations, NIDs geDaqgH9lTg / PrdHuuDekhY / +BzXYkqYeLE). KeyCreate already encoded, so
    // one C11 family forwarded an encoded value from one member and a bare errno from the other two.
    // CONFIDENCE: MED on the encoded form itself — the forwarders do not COMPARE, so no guest has
    // been observed testing an encoded result from any of the three.
    printf("-- TLS keys: encoded on the Sony spelling, bare on the POSIX one (#1983) --\n");
    {
        HleFn sce_kcreate = Hle::lookup(nid_hash("scePthreadKeyCreate"));
        HleFn sce_kdelete = Hle::lookup(nid_hash("scePthreadKeyDelete"));
        HleFn pos_kdelete = Hle::lookup(nid_hash("pthread_key_delete"));
        HleFn sce_setspec = Hle::lookup(nid_hash("scePthreadSetspecific"));
        HleFn pos_setspec = Hle::lookup(nid_hash("pthread_setspecific"));
        HleFn pos_getspec = Hle::lookup(nid_hash("pthread_getspecific"));
        HleFn sce_getspec = Hle::lookup(nid_hash("scePthreadGetspecific"));
        CHECK(sce_kcreate && sce_kdelete && pos_kdelete && sce_setspec && pos_setspec &&
                  pos_getspec && sce_getspec,
              "the TLS-key entry points are registered under both spellings");
        if (sce_kcreate && sce_kdelete && pos_kdelete && sce_setspec && pos_setspec &&
            pos_getspec && sce_getspec) {
            const uint64_t bogus = 0x7ffffffful;
            const uint64_t posix_del = pos_kdelete(bogus, 0, 0, 0, 0, 0);
            // Stated as a precondition rather than assumed: if a host ever accepted this, both arms
            // below would be vacuously satisfied and would read as coverage.
            CHECK(posix_del != 0,
                  "precondition: this host refuses an out-of-range TLS key at all");
            CHECK(posix_del == kBareEINVAL,
                  "pthread_key_delete on a bogus key keeps the bare FreeBSD EINVAL (22)");
            CHECK(sce_kdelete(bogus, 0, 0, 0, 0, 0) == kEncodedEINVAL,
                  "scePthreadKeyDelete on a bogus key reports encoded EINVAL (0x80020016)");
            CHECK(pos_setspec(bogus, 0x1234, 0, 0, 0, 0) == kBareEINVAL,
                  "pthread_setspecific on a bogus key keeps the bare EINVAL (22)");
            CHECK(sce_setspec(bogus, 0x1234, 0, 0, 0, 0) == kEncodedEINVAL,
                  "scePthreadSetspecific on a bogus key reports encoded EINVAL");

            // POSITIVE CONTROL. Without it, a body broken into "refuse everything" satisfies all
            // four arms above.
            //
            // The stored value is deliberately 0x2a, BELOW 256. sce_pthread_rc passes anything with
            // bits outside the low byte through untouched, so a large value would be identical
            // whether or not Getspecific were aliased — and section 5's arms cannot see this either,
            // because they never store a value. 0x2a is the only shape that discriminates: aliasing
            // scePthreadGetspecific turns it into 0x8002002a, i.e. a guest's TLS pointer delivered
            // as an errno. Read through BOTH spellings, since only the Sony one can be aliased.
            uint32_t key = 0xffffffffu;
            CHECK(sce_kcreate((uint64_t)(uintptr_t)&key, 0, 0, 0, 0, 0) == 0 && key != 0xffffffffu,
                  "control: scePthreadKeyCreate produces a key");
            CHECK(sce_setspec(key, 0x2a, 0, 0, 0, 0) == 0,
                  "control: scePthreadSetspecific stores a value and reports 0");
            CHECK(pos_getspec(key, 0, 0, 0, 0, 0) == 0x2a,
                  "control: pthread_getspecific reads the small value back unencoded");
            CHECK(sce_getspec && sce_getspec(key, 0, 0, 0, 0, 0) == 0x2a,
                  "control: scePthreadGetspecific reads back 0x2a, NOT 0x8002002a -- the arm that "
                  "catches a mechanical sweep reaching a value-returning handler");
            CHECK(sce_kdelete(key, 0, 0, 0, 0, 0) == 0,
                  "control: scePthreadKeyDelete of a real key reports 0");
        }
    }

    // ===== 7. #1983: scePthreadCondWait must still report SUCCESS for a real signalled wait =====
    // The kRows row above provokes the refusal; this is its positive control, and it is the arm that
    // separates "the handler reports its wait" from "the handler was broken into rejecting". It also
    // covers the second defect the row cannot see: before #1983 the body returned 0 for a wait that
    // FAILED as well as for one that succeeded, so only a success arm plus a failure arm together
    // establish that the result is now the wait's own.
    //
    // No lost-wakeup race: the signaller can only take the mutex once the wait has released it.
    printf("-- a real signalled condition wait still reports 0 through both spellings --\n");
    {
        HleFn sce_wait   = Hle::lookup(nid_hash("scePthreadCondWait"));
        HleFn posix_wait = Hle::lookup(nid_hash("pthread_cond_wait"));
        HleFn cond_init  = Hle::lookup(nid_hash("scePthreadCondInit"));
        HleFn cond_sig   = Hle::lookup(nid_hash("scePthreadCondSignal"));
        HleFn cond_del   = Hle::lookup(nid_hash("scePthreadCondDestroy"));
        HleFn mtx_init   = Hle::lookup(nid_hash("scePthreadMutexInit"));
        HleFn mtx_lock   = Hle::lookup(nid_hash("scePthreadMutexLock"));
        HleFn mtx_unlock = Hle::lookup(nid_hash("scePthreadMutexUnlock"));
        HleFn mtx_del    = Hle::lookup(nid_hash("scePthreadMutexDestroy"));
        CHECK(sce_wait && posix_wait && cond_init && cond_sig && cond_del && mtx_init && mtx_lock &&
                  mtx_unlock && mtx_del,
              "the condition-variable family is registered under both spellings");
        if (sce_wait && posix_wait && cond_init && cond_sig && cond_del && mtx_init && mtx_lock &&
            mtx_unlock && mtx_del) {
            void* cond = nullptr;
            void* mutex = nullptr;
            const uint64_t c = (uint64_t)(uintptr_t)&cond;
            const uint64_t m = (uint64_t)(uintptr_t)&mutex;
            CHECK(cond_init(c, 0, 0, 0, 0, 0) == 0 && cond, "control: a condition variable initializes");
            CHECK(mtx_init(m, 0, 0, 0, 0, 0) == 0 && mutex, "control: a mutex initializes");
            for (int arm = 0; arm < 2; ++arm) {
                HleFn wait_fn = arm == 0 ? sce_wait : posix_wait;
                CHECK(mtx_lock(m, 0, 0, 0, 0, 0) == 0, "control: the waiter takes the mutex first");
                std::thread signaller([&] {
                    mtx_lock(m, 0, 0, 0, 0, 0);
                    cond_sig(c, 0, 0, 0, 0, 0);
                    mtx_unlock(m, 0, 0, 0, 0, 0);
                });
                const uint64_t rc = wait_fn(c, m, 0, 0, 0, 0);
                // Release BEFORE joining, and the order is load-bearing rather than stylistic. The
                // wait returns holding the mutex. On a SPURIOUS wake it can re-acquire before the
                // signaller's blocked lock is ever granted, so joining first would strand the
                // signaller inside mtx_lock while this thread holds the mutex it is waiting for --
                // the test would HANG rather than fail. Under ctest that burns the timeout and
                // reports as an infrastructure problem, so a real regression would be attributed to
                // machine load instead of to this assertion.
                CHECK(mtx_unlock(m, 0, 0, 0, 0, 0) == 0,
                      "control: the wait reacquired the mutex before returning");
                signaller.join();
                CHECK(rc == 0, arm == 0
                          ? "scePthreadCondWait reports 0 for a wait that was really signalled"
                          : "pthread_cond_wait reports 0 for a wait that was really signalled");
            }
            cond_del(c, 0, 0, 0, 0, 0);
            mtx_del(m, 0, 0, 0, 0, 0);
        }
    }

    // ===== 8. the other side of "iff": handlers that correctly have no alias ==================
    // HONEST LABEL: these arms document the contract, they do NOT discriminate. Each body returns 0
    // on every path, and `sce_pthread_rc(0)` is 0, so a speculatively added alias would be
    // invisible here. They are kept because they pin the *observable* half — that these entry
    // points report success for a null object rather than inventing a failure — which is what a
    // future author would break by "fixing" them to return EINVAL without also aliasing them.
    printf("-- handlers with no failure path report success (contract, not a discriminator) --\n");
    for (const char* name : kInfallible) {
        char msg[192];
        HleFn f = Hle::lookup(nid_hash(name));
        if (!f) { snprintf(msg, sizeof msg, "%s is registered", name); CHECK(false, msg); continue; }
        const uint64_t got = call0(f);
        snprintf(msg, sizeof msg, "%-32s -> 0 (no failure path, so no alias)  got 0x%llx",
                 name, (unsigned long long)got);
        CHECK(got == 0, msg);
    }

    // The #2168 busy-destroy pair. Same observable assertion, deliberately a separate group and a
    // separate label: these bodies DO have a failure path, it is simply unreachable through a null
    // slot. Keeping them in kInfallible would keep asserting the right thing for a reason that
    // stopped being true when #2359 and #2379 landed. See kFallibleButNullSucceeds.
    printf("-- #2168 busy-destroy bodies: a NULL slot still reports success (the busy path is in "
           "test_cond_destroy_busy / test_barrier_destroy_busy) --\n");
    for (const char* name : kFallibleButNullSucceeds) {
        char msg[192];
        HleFn f = Hle::lookup(nid_hash(name));
        if (!f) { snprintf(msg, sizeof msg, "%s is registered", name); CHECK(false, msg); continue; }
        const uint64_t got = call0(f);
        snprintf(msg, sizeof msg, "%-32s -> 0 for a null slot (fallible, but not here)  got 0x%llx",
                 name, (unsigned long long)got);
        CHECK(got == 0, msg);
    }

    // ===== 9. #2575: join and detach must report their result, and only a real join writes back ===
    // Both bodies DISCARDED the host result and returned 0 on every path, so ESRCH, EINVAL and
    // EDEADLK all reached the guest as "the thread was joined, and here is its result" — with
    // `*value_ptr` overwritten by the `nullptr` the discarded call left behind. That second half is
    // what makes this worse than an ignored return code: "the worker returned NULL" and "there was
    // no such thread" became the same observation, and a SELF-join read as a worker that finished.
    //
    // The self-join arm carries the #1612 half as well, and it is the only join failure in this file
    // that can: EDEADLK is 11 on FreeBSD and 35 on Linux, and 35 is FreeBSD's EAGAIN — so a
    // passed-through host number would tell the guest "would block, retry" where the truth is "you
    // just tried to join yourself", which is a loop that never ends. Every other errno these two
    // produce is EINVAL(22), which every platform agrees on and proves nothing about the mapping.
    //
    // EVERY SKIP BELOW IS DECIDED BY A DIRECT HOST CALL, NEVER BY THE HANDLER UNDER TEST, and that
    // is the whole design of this section rather than a detail. The first draft asked the handler
    // itself whether the host refuses a self-join — and the mutation that reverts the fix (make the
    // body report 0 again) then produced `0`, which the draft read as "this host permits a
    // self-join" and SKIPPED. The arm passed under the exact defect it exists to catch, in both the
    // join and the detach block. A guard drawn from the same source as the null it is validating
    // tests nothing; the host probes here are the "construct a positive instance BY HAND, outside
    // whatever produced the null" rule applied literally.
    printf("-- join/detach report their result; only a real join writes value_ptr (#2575) --\n");
    {
        HleFn sce_join     = Hle::lookup(nid_hash("scePthreadJoin"));     // onNY9Byn-W8
        HleFn posix_join   = Hle::lookup(nid_hash("pthread_join"));
        HleFn sce_detach   = Hle::lookup(nid_hash("scePthreadDetach"));   // 4qGrR6eoP9Y
        HleFn posix_detach = Hle::lookup(nid_hash("pthread_detach"));
        CHECK(sce_join && posix_join && sce_detach && posix_detach,
              "scePthreadJoin/Detach and pthread_join/detach are registered under both spellings");
        if (sce_join && posix_join && sce_detach && posix_detach) {
            // --- POSITIVE CONTROL. Without it, a body broken into "refuse everything" satisfies
            // every failure arm below, and a body that never joins at all satisfies the rc == 0 half.
            // The exit-value readback separates them: 0x5eed can only reach `out` through a join
            // that really completed and really read the worker's return.
            pthread_t worker{};
            const int created = pthread_create(&worker, nullptr, join_worker, (void*)0x5eed);
            CHECK(created == 0, "control: a worker thread starts");
            if (created == 0) {
                void* out = (void*)0xC0FFEE;   // a sentinel a real join MUST overwrite
                const uint64_t rc = sce_join((uint64_t)worker, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0);
                CHECK(rc == 0, "control: scePthreadJoin of a real worker reports 0");
                CHECK(out == (void*)0x5eed,
                      "control: the worker's own exit value (0x5eed) reaches value_ptr -- an arm that "
                      "only checked rc == 0 would pass under the old always-0 body too");
            }

            // --- the join failure arm: a self-join, through both spellings.
            // POSIX specifies EDEADLK here ("a deadlock was detected, or the value of thread
            // specifies the calling thread"), so the host probe is a precondition rather than a
            // capability test — but it is still taken from the host, for the reason in the header.
            char msg[224];
            const int host_self_join = pthread_join(pthread_self(), nullptr);
            if (host_self_join == 0) {
                printf("  [SKIP] this host's own pthread_join permits a thread to join itself -- the "
                       "#2575 EDEADLK arms cannot run here\n");
            } else {
                snprintf(msg, sizeof msg,
                         "precondition: this HOST refuses a self-join with EDEADLK (host EDEADLK is "
                         "%d here; got %d) -- established by a direct host call, so no handler can "
                         "manufacture the skip", EDEADLK, host_self_join);
                CHECK(host_self_join == EDEADLK, msg);

                void* untouched = (void*)0xC0FFEE;
                const uint64_t sony_rc =
                    sce_join((uint64_t)pthread_self(), (uint64_t)(uintptr_t)&untouched, 0, 0, 0, 0);
                const uint64_t posix_rc = posix_join((uint64_t)pthread_self(), 0, 0, 0, 0, 0);
                snprintf(msg, sizeof msg,
                         "pthread_join(self) reports bare FreeBSD EDEADLK (11), not this host's %d "
                         "-- %d is FreeBSD's EAGAIN and would arrive as a retry hint (got %llu)",
                         EDEADLK, EDEADLK, (unsigned long long)posix_rc);
                CHECK(posix_rc == 11, msg);
                snprintf(msg, sizeof msg,
                         "scePthreadJoin(self) reports encoded FreeBSD EDEADLK 0x8002000b (got 0x%llx)",
                         (unsigned long long)sony_rc);
                CHECK(sony_rc == 0x8002000bull, msg);
                CHECK(untouched == (void*)0xC0FFEE,
                      "a REFUSED join leaves value_ptr untouched -- it used to be overwritten with "
                      "NULL, which a guest reads as a legitimate NULL exit value");
            }

            // --- detach. Its only reachable refusal is EINVAL, so it cannot carry the mapping half;
            // what it carries is the alias split, which is the defect it actually had.
            //
            // Both workers park on g_park_gate, which main holds for this whole block, so their
            // descriptors stay valid throughout — a detached thread that had already exited would be
            // a wild pointer, not a defined EINVAL. The PROBE thread is detached by direct host
            // calls only, so the host's answer to a second detach is ground truth; the SUBJECT
            // thread is the handler's.
            pthread_mutex_lock(&g_park_gate);
            pthread_t probe{}, subject{};
            const int probe_created   = pthread_create(&probe, nullptr, park_worker, nullptr);
            const int subject_created = pthread_create(&subject, nullptr, park_worker, nullptr);
            CHECK(probe_created == 0 && subject_created == 0, "control: two parked workers start");
            if (probe_created == 0 && subject_created == 0) {
                CHECK(wait_for(g_park_running, 2), "control: both parked workers reached the gate");
                CHECK(pthread_detach(probe) == 0, "control: the HOST detaches a live joinable thread");
                const int host_double = pthread_detach(probe);
                if (host_double == 0) {
                    printf("  [SKIP] this host's own pthread_detach accepts a second detach of the "
                           "same thread -- the #2575 detach refusal arms cannot run here\n");
                } else {
                    snprintf(msg, sizeof msg,
                             "precondition: this HOST refuses a second detach with EINVAL (got %d)",
                             host_double);
                    CHECK(host_double == EINVAL, msg);

                    CHECK(sce_detach((uint64_t)subject, 0, 0, 0, 0, 0) == 0,
                          "control: scePthreadDetach of a live joinable thread reports 0");
                    CHECK(pthread_detach(subject) == EINVAL,
                          "control: ...and it REALLY detached -- the host now refuses a second detach. "
                          "A body that reported 0 without calling through passes the line above and "
                          "fails this one");
                    CHECK(sce_detach((uint64_t)subject, 0, 0, 0, 0, 0) == kEncodedEINVAL,
                          "scePthreadDetach of an already-detached thread reports encoded EINVAL "
                          "(0x80020016)");
                    CHECK(posix_detach((uint64_t)subject, 0, 0, 0, 0, 0) == kBareEINVAL,
                          "pthread_detach of the same thread keeps the bare FreeBSD EINVAL (22)");
                }
            }
            pthread_mutex_unlock(&g_park_gate);
            // Both workers are detached, so nothing joins them; wait for them to leave the gate
            // before returning from main rather than racing process teardown against a live thread.
            if (probe_created == 0 && subject_created == 0)
                CHECK(wait_for(g_park_done, 2), "control: both parked workers finished");
        }
    }

    if (fails) printf("== FAIL (%d) ==\n", fails);
    else       printf("== PASS ==\n");
    return fails ? 1 : 0;
}
