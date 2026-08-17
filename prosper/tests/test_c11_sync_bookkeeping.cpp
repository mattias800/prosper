// test_c11_sync_bookkeeping — the C11 `_Mtx_*` / `_Cnd_*` handlers must keep the SAME PER-OBJECT
// BOOKKEEPING their libkernel counterparts keep, not merely share their slot resolution
// (#2619, #2623).
//
// WHY THIS IS A SEPARATE FILE FROM test_c11_thread_slots. That file guards #2596: the handlers must
// RESOLVE a statically-initialised slot instead of skipping it. This one guards what #2596 did not
// give them — the state each libkernel body maintains ALONGSIDE the operation. Every arm here passes
// on the #2596 code and fails on the bookkeeping being absent, which is exactly why the two must not
// be merged: a suite that resolves slots correctly can be completely blind to the accounting.
//
// THE PRIMARY EVIDENCE, because it is what decided the return values below and it settles the one
// question #2619 left open. The shipped guest `libc.prx` of PPSA24651 was flattened
// (`tools/il2cpp/prx_to_elf.py`), each C11 export located by NID in its dynsym, each body decoded
// (`tools/re/edis.py`), and each PLT thunk taken to its import through the JMPREL slot
// (`tools/re/stub_nid_map.py`) rather than by adjacency:
//
//   _Mtx_destroy   @0x5e60, 11 bytes:  push rbp; mov rbp,rsp; call -> scePthreadMutexDestroy;
//                                      pop rbp; ret
//   _Cnd_destroy   @0x5660, 11 bytes:  push rbp; mov rbp,rsp; call -> scePthreadCondDestroy;
//                                      pop rbp; ret
//   _Cnd_wait      @0x5670, 13 bytes:  … call -> scePthreadCondWait; XOR EAX,EAX; pop rbp; ret
//   _Thrd_yield    @0x4ee0, 11 bytes:  push rbp; mov rbp,rsp; call -> scePthreadYield; pop rbp; ret
//
// The `_Cnd_wait` line is the recorded #2596 reading, re-derived here byte for byte, and it is this
// method's positive control: a decode that reproduces a result someone else measured independently
// is a decode that resolved the right symbol.
//
// THE DESTROY PAIR'S RETURN VALUE IS NOT A CONTRACT, and this file used to say the opposite. The
// two are ELEVEN bytes, not thirteen — there is no `xor eax,eax` — and an earlier revision read
// that absence as "THE DESTROY PAIR FORWARDS libkernel's answer". It does not follow, and the module
// settles it in BOTH directions with two functions sixteen bytes apart:
//
//   _Thrd_yield  @0x4ee0  554889e5 e8f7061100 5dc3  ->  scePthreadYield   ISO: void thrd_yield(void)
//   _Thrd_equal  @0x4ef0  554889e5 e8f7061100 5dc3  ->  scePthreadEqual   ISO: int  thrd_equal(...)
//
// Byte-for-byte identical bodies, opposite ISO C11 return types — and `_Thrd_equal` can only return
// its value by leaving `scePthreadEqual`'s result in eax, so one of these forwards and the other has
// nothing to forward, with nothing in the bytes to tell them apart. `_Tss_create` / `_Tss_set` /
// `_Tss_get` are the same shape and also forward (already recorded in hle_kernel.cpp, above the
// `k_sce_key_delete` alias). pthread_slot.hpp carries the full table (#2636).
//
// A SECOND EARLIER REVISION claimed the byte shape tracked the published C11 return type across
// "all thirteen" of the module's C11 exports. That is withdrawn too: the thirteen were the twelve
// mutex/condvar primitives plus the one control that supported the conclusion, and the exports left
// outside the set contain at least six counterexamples. It is recorded rather than deleted because
// it is the failure mode worth remembering — a universal asserted over a set assembled alongside the
// conclusion tests the discriminator, not the domain.
//
// What survives is narrower and is labelled as such: the ISO C11 signatures of `mtx_destroy` and
// `cnd_destroy`, plus the fact that 0 of the destroy pair's 30 call sites in `.text` read the value.
//
// So both destroy handlers answer 0, and what #2619 asked to be established is established on the
// OBJECT rather than on a number: the guest wrapper's single call is to `scePthreadCondDestroy`, so
// #2168's EBUSY refusal is part of `_Cnd_destroy`'s behaviour too, and §4 asserts it — as a
// non-retirement, which is the form in which it is observable at all.
//
// WHAT EACH ARM KILLS. Every row below was MEASURED — mutation applied, rebuilt, re-run — and the
// counts are what the run printed, not what the author expected:
//   M1  restore the old private `m_mtx_destroy` body (bare `if (a0 && *(void**)P(a0))` +
//       `retire_sync_object`, no `pt_claim_slot`)
//                                    -> FAIL (4): all four discriminating arms of §1. §2/§3/§4 pass.
//   M2  restore the old private `m_cnd_destroy` body (no claim, no busy check)
//                                    -> FAIL (4): §2's three, plus §4's "none of the three retired
//                                       anything" arm. §4's Sony and POSIX refusals still PASS —
//                                       only the C11 body lost the check, which is the divergence,
//                                       and a suite that could not show that would not be testing
//                                       parity. §4's `_Cnd_destroy` VALUE arm also passes, because
//                                       the old body returned 0 as well; that arm is M2b's, not
//                                       M2's, and the pair is what separates the two questions.
//   M2b make `m_cnd_destroy` publish `kSceKernelErrorEBUSY` on a refusal instead of 0 — the
//       pre-correction #2636 behaviour, and the "right number, wrong convention" failure this area
//       keeps producing
//                                    -> FAIL (1): §4's `_Cnd_destroy` value arm, and ONLY that one.
//                                       So the value arm and the retirement arm are measurably
//                                       independent: M2 moves one and not the other, M2b the
//                                       reverse. Neither is a second spelling of the other, and a
//                                       suite asserting only "the object survived" would accept a
//                                       libkernel-encoded value through a C11-spelled door.
//   M3  drop `guest_cond_advance` from `guest_cond_signal_slot`/`guest_cond_broadcast_slot` (the
//       SHARED bodies, so both spellings lose it)
//                                    -> FAIL (3): all three of §3, the Sony parity arm included.
//   M4  route `m_cnd_signal`/`m_cnd_broadcast` back through the bare resolver — advance kept for the
//       Sony spelling only, i.e. the exact #2623(1) divergence
//                                    -> FAIL (2): §3's two C11 arms. The Sony parity arm PASSES.
//   M5  drop `GuestCondWaiterScope` from the shared `guest_cond_wait_slot`
//                                    -> FAIL (4): §4's Sony and POSIX refusals, the "none of the
//                                       three retired anything" arm, and the wake control — with no
//                                       waiter counted the condvar is retired out from under the
//                                       parked thread, so the later broadcast resolves a poisoned
//                                       slot and nobody is ever woken. That cascade IS the bug,
//                                       expressed as a failed assertion instead of a hang. The
//                                       `_Cnd_destroy` value arm passes here BY DESIGN: 0 is the
//                                       answer whether the destroy refused or retired, which is
//                                       precisely why the refusal is asserted on the object.
//   M6  give `m_cnd_wait` back its own body (resolve + `interruptible_cond_wait`, no scope) — the
//       exact #2623(2) divergence
//                                    -> FAIL (4): the same four. §1, §2 and §3 pass.
//   M7  make `guest_cond_waiter_leave` a no-op, so the count leaks
//                                    -> FAIL (1): §4's last arm, and only that one.
//
// M3/M4 are a deliberate pair, and the pair is the evidence: the first breaks the shared body so
// BOTH spellings lose the behaviour, the second breaks only the C11 one. §3 separates them, so §3
// is testing cross-spelling parity rather than merely that the feature exists.
//
// M5/M6 ARE NOT SEPARATED BY THIS FILE, and saying so is the honest form of the same claim: §4's
// only waiter parks through the C11 spelling, so both mutations produce the identical four
// failures. What separates them is the pre-existing `cond_destroy_busy` suite, which parks a SONY
// waiter — measured: it fails 3 arms under M5 and passes completely under M6. So the shared-body
// regression is caught there and the C11-only one is caught here, and neither file is redundant.
//
// M7 exists because every other arm in §4 survives it: without it, §4's closing arm would be a row
// no mutation kills, which is a row that has never been shown to test anything.
//
// HANG SAFETY. §4 parks a real thread on a condvar. It is DETACHED and every wait is bounded: a
// regression in this area is a thread that never wakes, and expressing that as a hang burns ctest's
// timeout and gets misread as machine load on a box that routinely runs at load 30+.

#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/pthread_slot.hpp"
#include "../src/hle/sce_errno.hpp"
#include "../src/hle/sync_retire.hpp"

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

HleFn mtx_init = nullptr, mtx_lock = nullptr, mtx_unlock = nullptr, mtx_destroy = nullptr;
HleFn cnd_init = nullptr, cnd_signal = nullptr, cnd_broadcast = nullptr, cnd_wait = nullptr,
      cnd_destroy = nullptr;
HleFn sce_mtx_lock = nullptr, sce_mtx_destroy = nullptr;
HleFn sce_cnd_signal = nullptr, sce_cnd_destroy = nullptr, posix_cnd_destroy = nullptr;

uint64_t call1(HleFn fn, void* slot) { return fn((uint64_t)(uintptr_t)slot, 0, 0, 0, 0, 0); }
uint64_t call2(HleFn fn, void* a, void* b) {
    return fn((uint64_t)(uintptr_t)a, (uint64_t)(uintptr_t)b, 0, 0, 0, 0);
}

// §4's parked waiter. File-scope storage so a detached worker can never outlive it.
void* g_park_cnd = nullptr;
void* g_park_mtx = nullptr;
std::atomic<int> g_parked{0};
std::atomic<int> g_woke{0};

void* park_in_c11_wait(void*) {
    mtx_lock((uint64_t)(uintptr_t)&g_park_mtx, 0, 0, 0, 0, 0);
    g_parked.store(1, std::memory_order_release);
    cnd_wait((uint64_t)(uintptr_t)&g_park_cnd, (uint64_t)(uintptr_t)&g_park_mtx, 0, 0, 0, 0);
    g_woke.store(1, std::memory_order_release);
    mtx_unlock((uint64_t)(uintptr_t)&g_park_mtx, 0, 0, 0, 0, 0);
    return nullptr;
}

// Generous, and only ever used for CONTROLS ("did the worker reach its wait"). Not a timing
// assertion — a tight bound would only turn machine load into a red build.
bool spin_until(const std::atomic<int>& flag, int seconds) {
    for (int i = 0; i < seconds * 1000 && !flag.load(std::memory_order_acquire); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return flag.load(std::memory_order_acquire) != 0;
}

bool broadcast_until_awake(void* cnd_slot, void* mtx_slot, const std::atomic<int>& flag) {
    for (int i = 0; i < 300 && !flag.load(std::memory_order_acquire); ++i) {
        mtx_lock((uint64_t)(uintptr_t)mtx_slot, 0, 0, 0, 0, 0);
        cnd_broadcast((uint64_t)(uintptr_t)cnd_slot, 0, 0, 0, 0, 0);
        mtx_unlock((uint64_t)(uintptr_t)mtx_slot, 0, 0, 0, 0, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return flag.load(std::memory_order_acquire) != 0;
}

}  // namespace

int main() {
    printf("== test_c11_sync_bookkeeping ==\n");
    register_builtin_hle();

    const auto by_name = [](const char* n) { return Hle::lookup(nid_hash(n)); };
    mtx_init          = by_name("_Mtx_init");
    mtx_lock          = by_name("_Mtx_lock");
    mtx_unlock        = by_name("_Mtx_unlock");
    mtx_destroy       = by_name("_Mtx_destroy");
    cnd_init          = by_name("_Cnd_init");
    cnd_signal        = by_name("_Cnd_signal");
    cnd_broadcast     = by_name("_Cnd_broadcast");
    cnd_wait          = by_name("_Cnd_wait");
    cnd_destroy       = by_name("_Cnd_destroy");
    sce_mtx_lock      = by_name("scePthreadMutexLock");
    sce_mtx_destroy   = by_name("scePthreadMutexDestroy");
    sce_cnd_signal    = by_name("scePthreadCondSignal");
    sce_cnd_destroy   = by_name("scePthreadCondDestroy");
    posix_cnd_destroy = by_name("pthread_cond_destroy");
    const bool all = mtx_init && mtx_lock && mtx_unlock && mtx_destroy && cnd_init && cnd_signal &&
                     cnd_broadcast && cnd_wait && cnd_destroy && sce_mtx_lock && sce_mtx_destroy &&
                     sce_cnd_signal && sce_cnd_destroy && posix_cnd_destroy;
    CHECK(all, "control: both spellings of every handler under test are registered");
    if (!all) { printf("== FAIL (%d) ==\n", fails); return 1; }

    // ===== 1. _Mtx_destroy CLAIMS and POISONS the slot (#2619) ================================
    // The old body read the slot with a bare truthiness test and never wrote to it, so the slot went
    // on naming quarantined storage. Three consequences, all asserted here: a second destroy retired
    // the SAME block again (a double free ~30 s later, from an unrelated call), a later lock through
    // either spelling resolved a destroyed object, and a slot already carrying the destroyed poison
    // 0xDEA passed the truthiness test and had 0xDEA itself handed to `free`.
    printf("-- _Mtx_destroy claims the slot, so a repeat destroy retires nothing --\n");
    {
        void* slot = nullptr;
        CHECK(mtx_init((uint64_t)(uintptr_t)&slot, 0, 0, 0, 0, 0) == 0 && slot != nullptr,
              "control: _Mtx_init produces an object");
        void* const object = slot;

        const uint64_t before = retired_sync_object_count();
        call1(mtx_destroy, &slot);
        CHECK(retired_sync_object_count() == before + 1,
              "control: the first _Mtx_destroy retires the object exactly once (true before this "
              "change too -- it is the positive control the arms below are measured against)");
        CHECK(slot != object && slot != nullptr,
              "the slot NO LONGER NAMES the retired storage, and was not merely cleared to NULL -- "
              "it holds the destroyed poison, so a later use is refused rather than self-initialised");

        // #2619 scenario 2, in its cross-spelling form, and it must come BEFORE any Sony destroy
        // touches this slot. An earlier revision put it last, where `scePthreadMutexDestroy` had
        // already claimed and poisoned the slot itself — so the refusal was produced by the arm
        // above it rather than by the handler under test, and it passed under the mutation. Order
        // is load-bearing here, not stylistic.
        CHECK(call1(sce_mtx_lock, &slot) == prosper::hle::kSceKernelErrorEINVAL,
              "scePthreadMutexLock REFUSES a slot the C11 spelling destroyed -- the use-after-destroy "
              "#2170 closed for the Sony spelling was still open through _Mtx_destroy");

        const uint64_t after_first = retired_sync_object_count();
        call1(mtx_destroy, &slot);
        CHECK(retired_sync_object_count() == after_first,
              "a SECOND _Mtx_destroy on the same slot retires NOTHING -- it used to hand the same "
              "block to the quarantine twice, which frees it twice ~30 s apart (#2619 scenario 1)");

        call1(sce_mtx_destroy, &slot);
        CHECK(retired_sync_object_count() == after_first,
              "...and a scePthreadMutexDestroy after it retires nothing either, so the two spellings "
              "agree about a slot that is already destroyed");
    }

    // ===== 2. _Cnd_destroy does the same ======================================================
    // Same defect, same shape, separate arm: the two handlers had separate private bodies, so a fix
    // to one proves nothing about the other. (Measured: M1 leaves every arm of §2 green.)
    printf("-- _Cnd_destroy claims the slot too --\n");
    {
        void* slot = nullptr;
        CHECK(cnd_init((uint64_t)(uintptr_t)&slot, 0, 0, 0, 0, 0) == 0 && slot != nullptr,
              "control: _Cnd_init produces an object");
        void* const object = slot;

        const uint64_t before = retired_sync_object_count();
        CHECK(call1(cnd_destroy, &slot) == 0, "control: destroying an idle condvar succeeds");
        CHECK(retired_sync_object_count() == before + 1 && slot != object && slot != nullptr,
              "the slot no longer names the retired condvar");

        const uint64_t after_first = retired_sync_object_count();
        call1(cnd_destroy, &slot);
        CHECK(retired_sync_object_count() == after_first,
              "a second _Cnd_destroy retires NOTHING (it used to double-retire the same block)");
        call1(sce_cnd_destroy, &slot);
        CHECK(retired_sync_object_count() == after_first,
              "...and neither does a scePthreadCondDestroy behind it");
    }

    // ===== 3. the missed-wakeup generation, through BOTH spellings (#2623 divergence 1) =========
    // `guest_cond_advance` is the guard that lets `interruptible_cond_clock_timedwait` recover a
    // signal that lands between two slices of a converted non-realtime deadline. Its only production
    // reader is that slice-boundary race, which no test can schedule — so the counter is read
    // directly. That is not a weaker assertion than a behavioural one: the counter is written by
    // exactly one function, whose only callers are the two bodies under test, so nothing else in
    // this process can move it.
    printf("-- _Cnd_signal / _Cnd_broadcast bump the missed-wakeup generation --\n");
    {
        void* slot = nullptr;   // a static sentinel: self-initialises on first use, like the guest's
        call1(cnd_signal, &slot);
        CHECK(slot != nullptr, "control: the static cnd_t slot self-initialised (#2596)");
        auto* cond = (pthread_cond_t*)slot;

        const uint64_t g0 = guest_cond_generation_for_test(cond);
        call1(cnd_signal, &slot);
        CHECK(guest_cond_generation_for_test(cond) == g0 + 1,
              "_Cnd_signal ADVANCES the generation -- a guest signalling through the C11 spelling and "
              "waiting through the Sony one saw a stale generation and could lose the wakeup");
        const uint64_t g1 = guest_cond_generation_for_test(cond);
        call1(cnd_broadcast, &slot);
        CHECK(guest_cond_generation_for_test(cond) == g1 + 1,
              "_Cnd_broadcast advances it too");
        // The parity arm, and the reason M3 and M4 are separable: it asserts the answer does not
        // depend on which spelling asked (#1873). It passes under M4 and fails under M3.
        const uint64_t g2 = guest_cond_generation_for_test(cond);
        call1(sce_cnd_signal, &slot);
        CHECK(guest_cond_generation_for_test(cond) == g2 + 1,
              "...and scePthreadCondSignal advances it by exactly the same amount on the same object");
    }

    // ===== 4. a thread parked in _Cnd_wait blocks EVERY spelling of cond destroy (#2623 div. 2) ==
    // #2168 made cond-destroy refuse a condvar with waiters, and #2596 opened a new route to a
    // waiter the check could not see: a null cond slot could not park at all before it, and can
    // after. So a destroy retired the object out from under a parked thread — #2168's exact failure
    // through a different door. All three destroy spellings are asserted, because the whole thesis
    // is that they agree.
    printf("-- a C11 waiter is visible to the cond-destroy busy check, through every spelling --\n");
    {
        pthread_t t{};
        CHECK(pthread_create(&t, nullptr, park_in_c11_wait, nullptr) == 0,
              "control: the C11 waiter thread starts");
        pthread_detach(t);
        CHECK(spin_until(g_parked, 5), "control: the waiter reached its _Cnd_wait");
        // The waiter has published `g_parked` before entering the wait, so give it a moment to get
        // inside. `GuestCondWaiterScope` is entered before `interruptible_cond_wait`, so this is a
        // bound on the scope being taken, not on the park completing.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // `before` and `object` are captured BEFORE the first destroy attempt, deliberately. An
        // earlier revision took the count just above the `_Cnd_destroy` arm, and when the two
        // preceding destroys were the ones that failed they had already retired the condvar and
        // poisoned the slot — so `_Cnd_destroy` found nothing to retire and the "retired NOTHING"
        // arm PASSED, in the exact run where the behaviour it names was broken. A count that can be
        // satisfied by the object having already gone is not a count of anything.
        const uint64_t before = retired_sync_object_count();
        void* const object = g_park_cnd;
        CHECK(object != nullptr, "control: the waiter self-initialised the static cnd_t slot");

        CHECK(call1(sce_cnd_destroy, &g_park_cnd) == prosper::hle::kSceKernelErrorEBUSY,
              "scePthreadCondDestroy refuses with the encoded EBUSY while a C11 wait is parked on it");
        CHECK(call1(posix_cnd_destroy, &g_park_cnd) ==
                  static_cast<uint64_t>(prosper::hle::FreeBsdErrno::EBusy),
              "...pthread_cond_destroy refuses with the bare EBUSY, same object, same instant");
        // The C11 spelling answers 0 — and this arm pins the CONVENTION rather than the refusal.
        // Its guest wrapper's eleven bytes do not establish a forwarded result (see the block at the
        // head of this file and pthread_slot.hpp, #2636), so there is no value here with which to
        // express a refusal; publishing the encoded EBUSY instead would put a libkernel-encoded
        // value through a C11-spelled door, which is what this arm fails on. It is NOT the arm that
        // catches a missing busy check — the old private body returned 0 too — and saying so is the
        // point: the refusal is observable only on the OBJECT, which is the next arm's job.
        CHECK(call1(cnd_destroy, &g_park_cnd) == 0,
              "...and _Cnd_destroy answers 0, the same as its _Mtx_destroy sibling and not a "
              "libkernel-encoded EBUSY");
        // THE load-bearing arm of the section, for all three spellings and for the C11 one in
        // particular: a destroy that refused must not have retired the condvar or cleared the slot.
        CHECK(retired_sync_object_count() == before && g_park_cnd == object,
              "...and NONE of the three retired anything: the slot still names the same live condvar "
              "the thread is parked in, rather than storage sitting in the quarantine with its "
              "clock started");

        CHECK(broadcast_until_awake(&g_park_cnd, &g_park_mtx, g_woke),
              "control: the waiter still wakes on a broadcast (a wait that never returns would be a "
              "worse defect than the one being fixed)");
        // Once the waiter has left, the refusal must clear — otherwise the fix would be a permanent
        // EBUSY, which is the leak `GuestCondWaiterScope`'s RAII exists to prevent.
        for (int i = 0; i < 500 && call1(sce_cnd_destroy, &g_park_cnd) != 0; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(retired_sync_object_count() > before,
              "...and the destroy SUCCEEDS once the waiter has left: the refusal is a live waiter "
              "count, not a one-way latch");
    }

    // ===== 5. the Windows guest-mutex ownership map (#2623 divergence 3) ========================
    // NOT EXECUTED ON THIS HOST, and said plainly rather than skipped silently. The ownership map is
    // `#ifdef _WIN32` (native POSIX implements the ERRORCHECK contract in pthreads, and routing every
    // guest lock through one process-global map there throttles synchronisation-heavy titles —
    // #719/#793). The C11 lock/unlock reached `interruptible_mutex_lock` / `pthread_mutex_unlock`
    // directly and never touched `guest_mutex_acquired` / `guest_mutex_released`, so on Windows a
    // static mutex first locked through the C11 spelling stayed registered with `owner == 0`. The fix
    // is structural — `m_mtx_lock`/`m_mtx_unlock` now call the same body `scePthreadMutexLock`/
    // `Unlock` call — and this arm confirms it on the platform where it is observable.
    printf("-- the C11 lock records ownership in the Windows guest-mutex map --\n");
#ifdef _WIN32
    {
        void* slot = nullptr;
        call1(mtx_lock, &slot);
        CHECK(slot != nullptr, "control: the static mtx_t slot self-initialised");
        CHECK(win_guest_mutex_owned_by_current_thread_for_test(slot),
              "_Mtx_lock records this thread as the owner, so a same-thread relock is refused by "
              "prosper's own check rather than by winpthreads' raw EDEADLK");
        call1(mtx_unlock, &slot);
        CHECK(!win_guest_mutex_owned_by_current_thread_for_test(slot),
              "_Mtx_unlock clears it, so a later scePthreadMutexLock is not refused against a stale "
              "owner");
    }
#else
    printf("  [skip] Windows-only: the guest-mutex ownership map does not exist on this host, and "
           "an arm that cannot fail is not an arm\n");
#endif

    if (fails) printf("== FAIL (%d) ==\n", fails);
    else       printf("== PASS ==\n");
    return fails ? 1 : 0;
}
