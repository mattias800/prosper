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
#include <cerrno>      // EDEADLK — the host's number, which is NOT the FreeBSD one the guest sees
#include <pthread.h>   // the #2024 arm probes the host primitive directly before asserting on it
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

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

    // #2012: an UNMATCHED unlock — nothing is held on the lock at all — must be refused, not
    // forwarded. FreeBSD (the guest's contract) returns EPERM and leaves the lock untouched;
    // glibc's is undefined and subtracts a reader, driving the count NEGATIVE so the lock can never
    // be write-acquired again. The lock still working afterwards is the assertion that matters:
    // without the guard, `trywrlock` below sees a phantom reader and returns EBUSY.
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
    // A read hold taken by THIS thread still releases normally afterwards. Since #2024 the rdlock
    // line is an assertion in its own right rather than mere setup — the handler no longer returns 0
    // unconditionally, so a 0 here means the host really granted the hold — and the unlock is the
    // second half: it proves the acquisition was RECORDED, because an unrecorded one draws EPERM.
    CHECK(call1(RWrd, (uint64_t)(uintptr_t)&h) == 0, "a granted rdlock reports success");
    CHECK(call1(RWun, (uint64_t)(uintptr_t)&h) == 0, "matched unlock still returns success");

    // ===== #2024: an acquisition the host REFUSED must be reported as a failure ==================
    //
    // Rdlock and Wrlock used to `return 0` unconditionally — `rc` fed only a diagnostic line — so a
    // guest handed a real EDEADLK was told "the lock is yours" and entered the critical section
    // unguarded. Its later unlock, entirely correct from its own point of view, then arrived at a
    // lock with no hold to release: #2012's corruption manufactured inside prosper rather than by
    // the guest. Silent, non-deterministic, and attributable to whatever corrupts next.
    //
    // Two arms, because they need different things from the host.
    //
    //   (a) NULL SLOT — deterministic everywhere, no host cooperation needed. A null rwlock cannot
    //       be acquired, so 0 is a lie the host was never even consulted about.
    //   (b) HOST REFUSAL — the real lever: make the host lock refuse an acquisition the handler
    //       forwards to it. A recursive write acquire does it; glibc, Darwin and FreeBSD all detect
    //       it and answer EDEADLK instead of self-deadlocking.
    //
    // Arm (b) PROBES the host first, on a private pthread_rwlock_t of its own, and asserts nothing
    // unless the probe proves this host really refuses. That is deliberate rather than defensive:
    // an arm that cannot reach the failure path it exists for must SAY so instead of passing
    // vacuously — the trap this repository keeps rediscovering is an assertion that held while the
    // mechanism never ran. The probe is watchdog'd because a host WITHOUT deadlock detection would
    // self-deadlock rather than answer, and a hung ctest is a worse signal than a loud skip.
    //
    // Both arms also assert the SONY and POSIX spellings separately, and that pairing is load-
    // bearing: it is what distinguishes a correctly aliased handler from one wired straight to the
    // POSIX body. Sony must report the libkernel-encoded form, POSIX the bare FreeBSD errno
    // (#1984's split). A single-spelling arm passes under either wiring.
    auto RWrdPosix = Hle::lookup(nid_hash("pthread_rwlock_rdlock"));
    auto RWwrPosix = Hle::lookup(nid_hash("pthread_rwlock_wrlock"));
    CHECK(RWrdPosix && RWwrPosix, "POSIX rdlock/wrlock spellings are registered");
    if (RWrdPosix && RWwrPosix) {
        // (a) A null slot: EINVAL, encoded for Sony and bare for POSIX.
        CHECK(call1(RWrd, 0) == 0x80020016ull, "scePthreadRwlockRdlock(null) reports encoded EINVAL");
        CHECK(call1(RWwr, 0) == 0x80020016ull, "scePthreadRwlockWrlock(null) reports encoded EINVAL");
        CHECK(call1(RWrdPosix, 0) == 22, "pthread_rwlock_rdlock(null) reports bare EINVAL(22)");
        CHECK(call1(RWwrPosix, 0) == 22, "pthread_rwlock_wrlock(null) reports bare EINVAL(22)");

        // (b) Probe this host. Statics, not stack objects: on a host that self-deadlocks the probe
        // thread is detached and never joined, so anything it can still touch must outlive main's
        // frame. Each stage publishes its own result, and a stage that never returns leaves the
        // stages after it at -1, which reads as "unknown" and skips the arm that depends on it.
        static pthread_rwlock_t probe_rw;
        static std::atomic<int> probe_rd{-1}, probe_wr{-1};
        static std::atomic<bool> probe_done{false};
        if (pthread_rwlock_init(&probe_rw, nullptr) == 0) {
            std::thread p([]{
                if (pthread_rwlock_wrlock(&probe_rw) != 0) { probe_done.store(true); return; }
                probe_rd.store(pthread_rwlock_rdlock(&probe_rw));    // read while SELF write-holds
                probe_wr.store(pthread_rwlock_wrlock(&probe_rw));    // recursive write acquire
                probe_done.store(true);
            });
            for (int i = 0; i < 3000 && !probe_done.load(); i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (probe_done.load()) p.join();
            else                   p.detach();   // this host blocks instead of refusing
        }
        const int rd_rc = probe_rd.load(), wr_rc = probe_wr.load();
        printf("  [info] host refusal probe: rdlock-while-self-write-held rc=%d, recursive wrlock"
               " rc=%d (this host's EDEADLK=%d)\n", rd_rc, wr_rc, EDEADLK);

        // FreeBSD EDEADLK is 11 and the encoded form is 0x8002000b. Neither is the host's number:
        // EDEADLK is 35 on Linux and 36 on MinGW, and 35 is FreeBSD's EAGAIN — so a handler that
        // returned the bare host `rc` would tell the guest "try again" where the host said
        // "deadlock". These two literals are exactly what `fbsd_errno` exists to produce (#1612).
        if (wr_rc == EDEADLK) {
            CHECK(call1(RWwr, (uint64_t)(uintptr_t)&h) == 0, "wrlock granted (setup for the refusal)");
            CHECK(call1(RWwr, (uint64_t)(uintptr_t)&h) == 0x8002000bull,
                  "scePthreadRwlockWrlock reports encoded EDEADLK when the host REFUSED");
            CHECK(call1(RWwrPosix, (uint64_t)(uintptr_t)&h) == 11,
                  "pthread_rwlock_wrlock reports bare FreeBSD EDEADLK(11), not the host's number");
            // No phantom hold: the two refusals recorded nothing, so exactly ONE hold is
            // outstanding. The SECOND unlock being refused is the proof — had either refused
            // acquire reached rw_acquired, COUNT would still be positive and this would return 0.
            CHECK(call1(RWun, (uint64_t)(uintptr_t)&h) == 0, "the one real write hold releases");
            CHECK(call1(RWun, (uint64_t)(uintptr_t)&h) == 0x80020001ull,
                  "the refused wrlocks recorded NO hold (the next unlock is unmatched -> EPERM)");
            CHECK(call1(RWtrywr, (uint64_t)(uintptr_t)&h) == 0,
                  "the lock is undamaged after the refused wrlocks");
            call1(RWun, (uint64_t)(uintptr_t)&h);
        } else {
            printf("  [SKIP] this host does not refuse a recursive write acquire (rc=%d) — the"
                   " wrlock half of the #2024 arm cannot run here\n", wr_rc);
        }
        if (rd_rc == EDEADLK) {
            CHECK(call1(RWwr, (uint64_t)(uintptr_t)&h) == 0, "wrlock granted (setup for the refusal)");
            CHECK(call1(RWrd, (uint64_t)(uintptr_t)&h) == 0x8002000bull,
                  "scePthreadRwlockRdlock reports encoded EDEADLK when the host REFUSED");
            CHECK(call1(RWrdPosix, (uint64_t)(uintptr_t)&h) == 11,
                  "pthread_rwlock_rdlock reports bare FreeBSD EDEADLK(11), not the host's number");
            CHECK(call1(RWun, (uint64_t)(uintptr_t)&h) == 0, "the one real write hold releases");
            CHECK(call1(RWun, (uint64_t)(uintptr_t)&h) == 0x80020001ull,
                  "the refused rdlocks recorded NO hold (the next unlock is unmatched -> EPERM)");
            CHECK(call1(RWtrywr, (uint64_t)(uintptr_t)&h) == 0,
                  "the lock is undamaged after the refused rdlocks");
            call1(RWun, (uint64_t)(uintptr_t)&h);
        } else {
            printf("  [SKIP] this host grants a read acquire while the caller write-holds (rc=%d) —"
                   " the rdlock half of the #2024 arm cannot run here\n", rd_rc);
        }
    }

    // #2012: the accounting is PER LOCK, not per thread, and that distinction is load-bearing.
    // FreeBSD's umtx path refuses an unlock only when nothing is held (or when a non-owner tries to
    // release a WRITE hold) — it makes no thread-identity check for readers, so releasing a read
    // hold from a different thread is legal there. prosper's own fibers make it reachable:
    // sceFiberRun resumes a suspended fiber on whichever host thread calls it, so a fiber that
    // read-locks, yields and is resumed elsewhere releases from another host thread. A per-thread
    // rule would refuse that and hang the title exactly the way this fix exists to prevent.
    {
        std::atomic<bool> locked{false};
        std::thread reader([&]{ call1(RWrd, (uint64_t)(uintptr_t)&h); locked.store(true); });
        reader.join();
        CHECK(locked.load(), "a read hold was taken on another thread");
        CHECK(call1(RWun, (uint64_t)(uintptr_t)&h) == 0,
              "a read hold taken on ANOTHER thread releases here (FreeBSD reader rule, not per-thread)");
        CHECK(call1(RWtrywr, (uint64_t)(uintptr_t)&h) == 0,
              "the lock is free after the cross-thread release");
        call1(RWun, (uint64_t)(uintptr_t)&h);
    }
    // A WRITE hold is different: FreeBSD checks the owner in userland and refuses a non-owner
    // release, so the write lock must still be held afterwards.
    {
        std::atomic<bool> locked{false}, release{false}, released{false};
        std::thread writer([&]{
            call1(RWwr, (uint64_t)(uintptr_t)&h);
            locked.store(true);
            while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            call1(RWun, (uint64_t)(uintptr_t)&h);
            released.store(true);
        });
        for (int i = 0; i < 2000 && !locked.load(); i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(locked.load(), "a write hold was taken on another thread");
        CHECK(call1(RWun, (uint64_t)(uintptr_t)&h) == 0x80020001ull,
              "releasing another thread's WRITE hold is refused with encoded EPERM");
        CHECK(call1(RWtrywr, (uint64_t)(uintptr_t)&h) != 0,
              "the write hold survived the refused cross-thread release");
        release.store(true);
        writer.join();
        CHECK(released.load(), "the owning thread released its own write hold");
        CHECK(call1(RWtrywr, (uint64_t)(uintptr_t)&h) == 0, "the lock is free once its owner releases");
        call1(RWun, (uint64_t)(uintptr_t)&h);
    }

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
        // out rather than acquire. NOTE how this arm fails if the shape is wrong: under a relative
        // or Sony scalar-µs reading the deadline lands far in the FUTURE while the helper still
        // holds the lock, so a wrong implementation produces a ctest TIMEOUT rather than a [FAIL].
        timespec past{};
        clock_gettime(CLOCK_REALTIME, &past);
        past.tv_sec -= 1;
        CHECK(call2(RWtimedwrP, (uint64_t)(uintptr_t)&h, (uint64_t)(uintptr_t)&past) == 60,
              "pthread_rwlock_timedwrlock with a past absolute deadline reports ETIMEDOUT(60)");
        CHECK(call2(RWtimedrdP, (uint64_t)(uintptr_t)&h, (uint64_t)(uintptr_t)&past) == 60,
              "pthread_rwlock_timedrdlock with a past absolute deadline reports ETIMEDOUT(60)");
        // The _np variants take a RELATIVE timespec. A zero timeout would NOT discriminate — {0,0}
        // is also a past deadline when read as absolute — so use a real 50 ms relative timeout and
        // assert the WAIT HAPPENED. Read as absolute, {0, 50'000'000} is a 1970 deadline and
        // returns immediately, so the elapsed-time bound is the only lever that moves.
        timespec rel50ms{0, 50L * 1000L * 1000L};
        auto elapsed_ms = [](auto begin) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - begin).count();
        };
        auto t0 = std::chrono::steady_clock::now();
        const uint64_t relwr = call2(RWreltimedwrP, (uint64_t)(uintptr_t)&h,
                                     (uint64_t)(uintptr_t)&rel50ms);
        const auto relwr_ms = elapsed_ms(t0);
        CHECK(relwr == 60 && relwr_ms >= 40,
              "pthread_rwlock_reltimedwrlock_np waits its RELATIVE timeout, then ETIMEDOUT(60)");
        t0 = std::chrono::steady_clock::now();
        const uint64_t relrd = call2(RWreltimedrdP, (uint64_t)(uintptr_t)&h,
                                     (uint64_t)(uintptr_t)&rel50ms);
        const auto relrd_ms = elapsed_ms(t0);
        CHECK(relrd == 60 && relrd_ms >= 40,
              "pthread_rwlock_reltimedrdlock_np waits its RELATIVE timeout, then ETIMEDOUT(60)");
        release.store(true);
        holder.join();
        // The lock is free and undamaged once the helper releases it.
        CHECK(call1(RWtrywrP, (uint64_t)(uintptr_t)&h) == 0,
              "pthread_rwlock_trywrlock succeeds once the helper releases");
        CHECK(call1(RWun, (uint64_t)(uintptr_t)&h) == 0, "unlock the write hold from the POSIX arms");
    }
    // #2012: CONCURRENT cross-thread traffic against stray unlocks. The two ordering windows this
    // guard can get wrong — publishing the hold count before the owner on acquire, or clearing the
    // owner before the count on release — are each a few instructions wide, so a single-shot arm
    // passes by luck. This LOOPS: balanced writers and readers on several threads while another
    // thread hammers stray unlocks at the same lock.
    //
    // The invariant is NOT "a balanced pair always releases". FreeBSD makes no identity check for
    // readers, so a stray unlock that arrives while readers are present legitimately consumes one
    // reader's release — on hardware too. What must hold is sharper, and is exactly what the store
    // ordering buys:
    //   * a WRITE hold can never be stolen: the write flag and the hold count live in ONE atomic
    //     word (bit 63 WRITE, bits 0..31 COUNT), so the unlock decision is a single CAS over the
    //     whole value. A stray either sees WRITE set (owner check -> EPERM) or sees COUNT == 0
    //     (-> EPERM). There is no pair of loads to straddle, by construction.
    //     This is NOT the two-store ordering argument, which was tried and falsified. With
    //     `writer` and `holds` as separate atomics, a stray could straddle the pre-check/post-check
    //     pair across two stalls: the owner clears `writer` before decrementing `holds`, so the
    //     stray reads writer == 0 with holds == 1, takes the reader path, and forwards an unlock
    //     that glibc services as a WRITER release — freeing the real owner's lock. The owner then
    //     decrements to -1 and its own unlock takes the reader path, driving __readers negative:
    //     #2012 reproduced by its own fix. Publishing the owner before the count and clearing it
    //     after is necessary and NOT sufficient; only the single-word CAS closes it.
    //   * a READ hold can only be stolen by a stray that was ACCEPTED, so the victims are bounded
    //     by the accepted strays.
    //   * and the lock is never corrupted: it still write-acquires afterwards.
    // The workers take WRITE holds only, deliberately. A write-held lock always has WRITE set in
    // the same word the stray must CAS, and an idle lock has COUNT == 0, so **no stray is ever
    // accepted** — which makes the arm deterministic and gives it a crisp invariant.
    //
    // Read this before acting on a failure: the invariant follows from the single-word CAS, NOT
    // from this test. A two-stall interleaving is not observable by any amount of hammering here,
    // so a clean run is not evidence of closure — the CAS's atomicity is the argument, and the arm
    // exists to catch a future regression that reintroduces a straddle. If `strays_accepted != 0`
    // ever fires, that is the signature of a residual ordering window in `rw_release`. Do not retry
    // it; go read `rw_release`.
    //
    // Reader stealing is deliberately NOT in this loop. It is legal (the serialized cross-thread
    // read arm above covers it), but forwarding a non-owner read release into glibc under heavy
    // contention is undefined territory in the HOST lock rather than in this accounting, and a test
    // that manufactures it is testing glibc, not prosper — a real guest issues balanced pairs plus
    // the occasional stray, not 4,000 adversarial ones.
    {
        std::atomic<int> refused_after_write{0};
        std::atomic<int> strays_accepted{0}, strays_refused{0}, rounds{0};
        std::vector<std::thread> workers;
        for (int t = 0; t < 3; t++) {
            workers.emplace_back([&]{
                for (int i = 0; i < 2000; i++) {
                    call1(RWwr, (uint64_t)(uintptr_t)&h);
                    if (call1(RWun, (uint64_t)(uintptr_t)&h) != 0) refused_after_write.fetch_add(1);
                    rounds.fetch_add(1);
                }
            });
        }
        workers.emplace_back([&]{
            // The attacker: unlocks it never took. Bounded on BOTH sides — it stops when the
            // workers are done, and never spins more than this many times, because an unbounded
            // "while the workers run" loop reaches half a billion attempts on a fast host and buys
            // no additional interleavings.
            for (int i = 0; i < 200000 && rounds.load(std::memory_order_relaxed) < 6000; i++) {
                if (call1(RWun, (uint64_t)(uintptr_t)&h) != 0) strays_refused.fetch_add(1);
                else strays_accepted.fetch_add(1);
            }
        });
        for (auto& w : workers) w.join();
        CHECK(rounds.load() == 6000 && strays_refused.load() > 1000,
              "the concurrency arm actually ran (6000 balanced write rounds; strays were refused)");
        // If this ever fires, it is NOT flakiness — it is the signature of a residual ordering
        // window between the write flag and the hold count, and it is the only assertion here that
        // can see one (`refused_after_write` cannot: a stray that steals the hold returns 0 to
        // ITSELF, and the owner's own release still succeeds). Do not retry it; go read rw_release.
        CHECK(strays_accepted.load() == 0,
              "no stray unlock is EVER accepted against a write-only workload (the ordering invariant)");
        CHECK(refused_after_write.load() == 0,
              "no WRITE hold was ever stolen, so every owner released its own hold");
        CHECK(call1(RWtrywr, (uint64_t)(uintptr_t)&h) == 0,
              "the lock still write-acquires after the concurrency arm (never corrupted)");
        call1(RWun, (uint64_t)(uintptr_t)&h);
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
