// #1612: every errno the guest observes must be a FreeBSD errno.
//
// The PS5 kernel is FreeBSD-derived. Linux and FreeBSD agree across most of 1..34 and diverge above
// it, so a constant copied out of the host's <errno.h> looks right in review and reaches the guest
// meaning something else. 11 is the exception inside the low range, and it is the headline case.
//
// The pair that matters most is EAGAIN/EDEADLK, which the two systems *swap*:
// FreeBSD EAGAIN=35 / EDEADLK=11, Linux EAGAIN=11 / EDEADLK=35. Handing back the host number turns
// "would block, retry" into "deadlock avoided" and vice versa — and a guest that retries on EAGAIN
// can spin forever on a condition that will never change.
//
// These assertions are written against the FreeBSD numbers the guest expects, so on a Linux host
// they fail against the pre-#1612 code and pass after it. On a BSD-numbered host (macOS) several
// were already correct; the contract they pin is the same either way.

#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/sce_errno.hpp"
#include "../src/hle/sync_retire.hpp"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

using namespace prosper;
using prosper::hle::FreeBsdErrno;

static int failures;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++failures; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_sce_errno ==\n");

    // ---- the vocabulary itself -------------------------------------------------------------
    // Spelled as literals on purpose: if someone "simplifies" the table by pointing it at the host
    // headers, these are the assertions that catch it.
    CHECK(prosper::hle::sce_kernel_error(FreeBsdErrno::EDeadlk) == 0x8002000bull,
          "SCE encoding of FreeBSD EDEADLK is 0x8002000b (11), not the Linux 35");
    CHECK(prosper::hle::sce_kernel_error(FreeBsdErrno::EAgain) == 0x80020023ull,
          "SCE encoding of FreeBSD EAGAIN is 0x80020023 (35), not the Linux 11");
    CHECK(prosper::hle::sce_kernel_error(FreeBsdErrno::ENoSys) == 0x8002004eull,
          "SCE encoding of FreeBSD ENOSYS is 0x8002004e (78), not the Linux 38");
    CHECK(prosper::hle::sce_kernel_error(FreeBsdErrno::ETimedOut) == 0x8002003cull,
          "SCE encoding of FreeBSD ETIMEDOUT is 0x8002003c (60), not the Linux 110");
    CHECK(prosper::hle::sce_kernel_error(FreeBsdErrno::EInval) == 0x80020016ull,
          "the values Linux and FreeBSD share are unchanged (EINVAL 22)");
    CHECK(prosper::hle::sce_kernel_error(FreeBsdErrno::EOwnerDead) == 0x80020060ull,
          "0x80020060 is EOWNERDEAD (96) — the value a decimal-60 typo produces, NOT a timeout");

    // ---- host -> FreeBSD translation -------------------------------------------------------
    CHECK(prosper::hle::freebsd_errno_from_host(EAGAIN) == FreeBsdErrno::EAgain,
          "host EAGAIN maps to FreeBSD 35 whatever this host numbers it");
    CHECK(prosper::hle::freebsd_errno_from_host(EDEADLK) == FreeBsdErrno::EDeadlk,
          "host EDEADLK maps to FreeBSD 11 whatever this host numbers it");
    CHECK(prosper::hle::freebsd_errno_from_host(ETIMEDOUT) == FreeBsdErrno::ETimedOut,
          "host ETIMEDOUT maps to FreeBSD 60");
    CHECK(prosper::hle::freebsd_errno_from_host(ENOSYS) == FreeBsdErrno::ENoSys,
          "host ENOSYS maps to FreeBSD 78");
    CHECK(prosper::hle::freebsd_errno_from_host(EINVAL) == FreeBsdErrno::EInval,
          "host EINVAL maps to FreeBSD 22");
    // An unknown errno must NOT be passed through raw — passing it through is the original defect.
    CHECK(prosper::hle::freebsd_errno_from_host(0x7fffffff) == FreeBsdErrno::EInval,
          "an unrecognized host errno degrades to EINVAL rather than leaking a raw number");
    // Round trip: the file layer converts an SCE result back into a host errno for its libc alias.
    CHECK(prosper::hle::host_errno_from_freebsd(FreeBsdErrno::EAgain) == EAGAIN &&
          prosper::hle::host_errno_from_freebsd(FreeBsdErrno::EDeadlk) == EDEADLK &&
          prosper::hle::host_errno_from_freebsd(FreeBsdErrno::ETimedOut) == ETIMEDOUT,
          "FreeBSD -> host is the exact inverse for the divergent values");

    // The file layer's 45-line hand-rolled ladder was folded onto this table (#1612). Its values are
    // reproduced exactly, including the deliberate EIO fallback rather than EINVAL — an independent
    // check on the table, since that ladder was written separately and already had them right.
    {
        struct Row { int host; uint32_t freebsd; };
        const Row rows[] = {
            {EPERM,1},{ENOENT,2},{EINTR,4},{EIO,5},{ENXIO,6},{EBADF,9},{ENOMEM,12},{EACCES,13},
            {EFAULT,14},{EBUSY,16},{EEXIST,17},{EXDEV,18},{ENODEV,19},{ENOTDIR,20},{EISDIR,21},
            {EINVAL,22},{ENFILE,23},{EMFILE,24},{ENOTTY,25},
#ifdef ETXTBSY
            {ETXTBSY,26},
#endif
            {EFBIG,27},{ENOSPC,28},
            {ESPIPE,29},{EROFS,30},{EMLINK,31},{EPIPE,32},{EAGAIN,35},{ELOOP,62},
            {ENAMETOOLONG,63},
#ifdef EDQUOT
            {EDQUOT,69},
#endif
            {EOVERFLOW,84},
        };
        bool all = true;
        for (const Row& r : rows)
            if (static_cast<uint32_t>(prosper::hle::freebsd_errno_from_host(r.host)) != r.freebsd) {
                std::printf("    mismatch: host %d -> %u, expected %u\n", r.host,
                            (unsigned)prosper::hle::freebsd_errno_from_host(r.host), r.freebsd);
                all = false;
            }
        CHECK(all, "the shared table reproduces every value of the file layer's original ladder");
        CHECK(prosper::hle::freebsd_errno_from_host(0x7ffffffe, FreeBsdErrno::EIo) == FreeBsdErrno::EIo,
              "the file layer keeps its EIO fallback for an unmapped host failure");
        // The original ladder had an explicit EWOULDBLOCK arm. On Linux/macOS it aliases EAGAIN, so
        // a host-value comparison cannot tell whether the row survived the fold — but on MinGW/UCRT
        // EWOULDBLOCK is 140, a distinct value that would silently take the fallback. Assert by
        // NAME so the check is meaningful on the platform where it can actually regress.
        CHECK(prosper::hle::freebsd_errno_from_host(EWOULDBLOCK) == FreeBsdErrno::EAgain,
              "EWOULDBLOCK maps to FreeBSD EAGAIN even where it is not an alias of host EAGAIN");
        CHECK(prosper::hle::freebsd_errno_from_host(EOPNOTSUPP) == FreeBsdErrno::EOpNotSupp &&
              prosper::hle::freebsd_errno_from_host(ENOTSUP) == FreeBsdErrno::EOpNotSupp,
              "both spellings of 'not supported' reach FreeBSD EOPNOTSUPP");
        // Rows MinGW/UCRT does not define at all must still work where the host does define them,
        // and must not have been dropped from the table to make Windows compile.
#ifdef EDQUOT
        CHECK(prosper::hle::freebsd_errno_from_host(EDQUOT) == FreeBsdErrno::EDQuot,
              "EDQUOT survives the per-row #ifdef guarding");
#endif
#ifdef ESTALE
        CHECK(prosper::hle::freebsd_errno_from_host(ESTALE) == FreeBsdErrno::EStale,
              "ESTALE survives the per-row #ifdef guarding");
#endif
#ifdef EMULTIHOP
        CHECK(prosper::hle::freebsd_errno_from_host(EMULTIHOP) == FreeBsdErrno::EMultiHop,
              "EMULTIHOP survives the per-row #ifdef guarding");
#endif
    }

    FreeBsdErrno decoded{};
    CHECK(prosper::hle::sce_kernel_error_errno(0x8002000bull, decoded) &&
          decoded == FreeBsdErrno::EDeadlk,
          "an encoded error decodes back to its FreeBSD errno");
    CHECK(!prosper::hle::sce_kernel_error_errno(0x80020100ull, decoded),
          "a value outside the errno byte is not treated as a member of the family");

    register_builtin_hle();

    // ---- behavioral: scePthreadSemTrywait on an empty semaphore ---------------------------
    // sem_trywait fails with EAGAIN ("would block"). Unmapped, the guest received the HOST 11,
    // which on the PS5 is EDEADLK: a retryable poll reported as a deadlock.
    {
        auto sem_init    = Hle::lookup(nid_hash("scePthreadSemInit"));
        auto sem_trywait = Hle::lookup(nid_hash("scePthreadSemTrywait"));
        auto sem_destroy = Hle::lookup(nid_hash("scePthreadSemDestroy"));
        if (sem_init && sem_trywait && sem_destroy) {
            alignas(16) uint8_t slot[32]{};
            CHECK(sem_init((uint64_t)slot, 0, 0, 0, 0, 0) == 0, "zero-count guest semaphore initializes");
            // #2178: this is the SONY spelling, so the FreeBSD number arrives libkernel-encoded.
            // The two halves of the claim are independent and both matter — 0x8002_0000 is the
            // encoding, and 0x23 (35) is the FreeBSD numbering, where the host would have said 11.
            const uint64_t rc = sem_trywait((uint64_t)slot, 0, 0, 0, 0, 0);
            CHECK(rc == sce_kernel_error(FreeBsdErrno::EAgain),
                  "scePthreadSemTrywait on an empty semaphore reports encoded FreeBSD EAGAIN "
                  "(0x80020023) — not a bare 35, and not host 11");
            // The POSIX spelling shares the body and must NOT encode. Without this line the arm
            // above cannot distinguish a correct alias from both spellings being encoded.
            auto posix_trywait = Hle::lookup(nid_hash("sem_trywait"));
            errno = 0;
            const uint64_t posix_rc =
                posix_trywait ? posix_trywait((uint64_t)slot, 0, 0, 0, 0, 0) : 0;
            CHECK(posix_trywait && posix_rc == (uint64_t)(int64_t)-1 &&
                      errno == static_cast<int>(FreeBsdErrno::EAgain),
                  "sem_trywait returns -1 and leaves FreeBSD EAGAIN (35) in errno (#2182) — the "
                  "POSIX contract, and the FreeBSD numbering the guest compares against");
            sem_destroy((uint64_t)slot, 0, 0, 0, 0, 0);
        } else {
            std::printf("  [skip] scePthreadSem* not registered on this platform\n");
        }
    }

    // ===== scePthreadAttrSetstacksize: fallible, and encoded on the Sony spelling (#2183) =======
    // It used to accept a null attr and any size, silently, reporting success -- while its sibling
    // scePthreadAttrSetstack rejected the SAME three conditions with EINVAL. A guest asking for an
    // 8 KiB stack was told yes by one entry point and no by the other, and its thread was then
    // created with whatever size the attribute already carried.
    {
        auto attr_init    = Hle::lookup(nid_hash("scePthreadAttrInit"));
        auto sce_setsize  = Hle::lookup(nid_hash("scePthreadAttrSetstacksize"));
        auto pos_setsize  = Hle::lookup(nid_hash("pthread_attr_setstacksize"));
        auto getsize      = Hle::lookup(nid_hash("scePthreadAttrGetstacksize"));
        auto attr_destroy = Hle::lookup(nid_hash("scePthreadAttrDestroy"));
        CHECK(attr_init && sce_setsize && pos_setsize && getsize && attr_destroy,
              "the pthread-attr stack entry points are registered");
        if (attr_init && sce_setsize && pos_setsize && getsize && attr_destroy) {
            void* at = nullptr;
            const uint64_t slot = (uint64_t)(uintptr_t)&at;
            attr_init(slot, 0, 0, 0, 0, 0);
            CHECK(at != nullptr, "the attribute initializes");

            // The pair, on the same undersized request. Neither spelling alone can see the other
            // drifting, which is the whole reason #2178 asserts both halves per entry point.
            CHECK(sce_setsize(slot, 4096, 0, 0, 0, 0) == 0x80020016ull,
                  "scePthreadAttrSetstacksize refuses a 4 KiB stack with encoded EINVAL "
                  "(0x80020016), the same answer its scePthreadAttrSetstack sibling already gave");
            CHECK(pos_setsize(slot, 4096, 0, 0, 0, 0) ==
                      static_cast<uint64_t>(FreeBsdErrno::EInval),
                  "pthread_attr_setstacksize refuses the same request with the bare errno (22)");
            CHECK(sce_setsize(0, 256 * 1024, 0, 0, 0, 0) == 0x80020016ull,
                  "a null attribute handle is refused even for a valid size");

            // The arm that stops "reject everything" from passing: a VALID size must still be
            // stored and must read back. Without it the three arms above are equally satisfied by a
            // handler broken in the other direction, which is the cheaper mistake to make here.
            CHECK(sce_setsize(slot, 256 * 1024, 0, 0, 0, 0) == 0,
                  "a 256 KiB stack is still accepted");
            size_t stored = 0;
            getsize(slot, (uint64_t)(uintptr_t)&stored, 0, 0, 0, 0);
            CHECK(stored == 256 * 1024,
                  "scePthreadAttrGetstacksize reads back the size that was accepted");

            attr_destroy(slot, 0, 0, 0, 0, 0);
        }
    }

    // ===== use-after-destroy is EINVAL, not a silently fresh object (#2170) ====================
    // FreeBSD writes a DESTROYED poison through the caller's handle and every later operation on it
    // returns EINVAL. prosper wrote NULL, which pt_static_sentinel reads as "statically initialised,
    // self-init me" -- so ensure_mutex/cond/rwlock manufactured a brand-new UNHELD object and the
    // operation SUCCEEDED. Thread A holds rwlock R, thread B destroys it, A's next rdlock gets a
    // fresh lock and enters the critical section alongside A. No error, no log line.
    //
    // Each arm below is destroy-then-USE. A test that only checked destroy's own return value could
    // not see this at all: destroy returned 0 before and returns 0 now.
    {
        struct { const char* init; const char* destroy; const char* use; const char* what; } objs[] = {
            { "scePthreadMutexInit",  "scePthreadMutexDestroy",  "scePthreadMutexLock",   "mutex"  },
            { "scePthreadRwlockInit", "scePthreadRwlockDestroy", "scePthreadRwlockRdlock","rwlock" },
        };
        for (const auto& o : objs) {
            auto init = Hle::lookup(nid_hash(o.init));
            auto destroy = Hle::lookup(nid_hash(o.destroy));
            auto use = Hle::lookup(nid_hash(o.use));
            char msg[220];
            snprintf(msg, sizeof msg, "%s / %s / %s are registered", o.init, o.destroy, o.use);
            CHECK(init && destroy && use, msg);
            if (!init || !destroy || !use) continue;

            alignas(16) uint8_t slot[32]{};
            const uint64_t h = (uint64_t)(uintptr_t)slot;
            snprintf(msg, sizeof msg, "a guest %s initialises", o.what);
            CHECK(init(h, 0, 0, 0, 0, 0) == 0, msg);

            // Positive control, and it is the arm that stops "reject everything" from passing: the
            // object must WORK before it is destroyed. Without it, an ensure_* broken to always
            // return nullptr would satisfy every assertion below.
            snprintf(msg, sizeof msg, "the %s is usable before destroy", o.what);
            CHECK(use(h, 0, 0, 0, 0, 0) == 0, msg);

            CHECK(destroy(h, 0, 0, 0, 0, 0) == 0, "destroy succeeds");
            // THE BUG: this used to return 0, having silently built a fresh unheld object.
            snprintf(msg, sizeof msg,
                     "using a destroyed %s reports encoded EINVAL rather than self-initialising",
                     o.what);
            CHECK(use(h, 0, 0, 0, 0, 0) == 0x80020016ull, msg);

            // The interaction #2170 flags: re-Init on a destroyed slot must be ACCEPTED. A poison
            // that made the slot permanently unusable would break every guest that reuses storage.
            snprintf(msg, sizeof msg, "a destroyed %s slot can be re-initialised", o.what);
            CHECK(init(h, 0, 0, 0, 0, 0) == 0, msg);
            snprintf(msg, sizeof msg, "the re-initialised %s works again", o.what);
            CHECK(use(h, 0, 0, 0, 0, 0) == 0, msg);
            destroy(h, 0, 0, 0, 0, 0);
        }
    }

    // ===== destroy CLAIMS the slot, so a double destroy retires once (#2176) ==================
    // Two threads racing destroy on one slot both used to read the non-sentinel pointer before
    // either cleared it, and both handed the SAME storage to retire_sync_object. Before the #2042
    // quarantine that double-freed immediately; with it, the second free lands ~30 s later from an
    // unrelated call -- strictly harder to diagnose, which cuts against the quarantine's own
    // argument. The fix claims the slot with an atomic exchange, so exactly one caller gets a
    // pointer back.
    {
        auto init = Hle::lookup(nid_hash("scePthreadMutexInit"));
        auto destroy = Hle::lookup(nid_hash("scePthreadMutexDestroy"));
        CHECK(init && destroy, "mutex init/destroy are registered");
        if (init && destroy) {
            // Sequential arm. Deterministic, and it is the case #2176 notes is reachable with no
            // race at all on the C11 spellings.
            alignas(16) uint8_t slot[32]{};
            const uint64_t h = (uint64_t)(uintptr_t)slot;
            init(h, 0, 0, 0, 0, 0);
            const uint64_t before = prosper::retired_sync_object_count();
            destroy(h, 0, 0, 0, 0, 0);
            const uint64_t after_one = prosper::retired_sync_object_count();
            CHECK(after_one == before + 1, "the first destroy retires the object exactly once");
            destroy(h, 0, 0, 0, 0, 0);
            CHECK(prosper::retired_sync_object_count() == after_one,
                  "a second destroy of the same slot retires NOTHING (the slot was claimed)");

            // There is deliberately NO concurrent arm here, and that is worth stating rather
            // than leaving as an omission. I wrote one -- two threads racing destroy on one slot,
            // asserting exactly one retire, which holds for every interleaving -- and could not make
            // it DISCRIMINATE: against a deliberately non-atomic read-then-write claim (with a yield
            // opened in the window) it fired once in 40 runs, then not once in 246 more across 200
            // rounds per run. An arm that cannot show its lever moved is not evidence, and a 200-
            // round thread-pair loop that proves nothing is worse than none: it reads as coverage.
            //
            // The concurrent case is guarded BY CONSTRUCTION instead: __atomic_exchange_n returns
            // the previous value to exactly one caller, so at most one racer can ever receive a
            // non-sentinel pointer to retire. The arm above tests the same claim through the path a
            // test can observe deterministically.
        }
    }

    // ---- behavioral: scePthreadMutexTimedlock timeout --------------------------------------
    {
        auto mtx_init = Hle::lookup(nid_hash("scePthreadMutexInit"));
        auto mtx_lock = Hle::lookup(nid_hash("scePthreadMutexLock"));
        auto mtx_timedlock = Hle::lookup(nid_hash("scePthreadMutexTimedlock"));
        auto mtx_unlock = Hle::lookup(nid_hash("scePthreadMutexUnlock"));
        if (mtx_init && mtx_lock && mtx_timedlock && mtx_unlock) {
            alignas(16) uint8_t slot[32]{};
            CHECK(mtx_init((uint64_t)slot, 0, 0, 0, 0, 0) == 0, "guest mutex initializes");
            // Hold it on another thread so the timed lock genuinely expires.
            std::atomic<bool> held{false}, release{false};
            std::thread holder([&] {
                mtx_lock((uint64_t)slot, 0, 0, 0, 0, 0);
                held.store(true, std::memory_order_release);
                while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
                mtx_unlock((uint64_t)slot, 0, 0, 0, 0, 0);
            });
            while (!held.load(std::memory_order_acquire)) std::this_thread::yield();
            const uint64_t rc = mtx_timedlock((uint64_t)slot, 20000 /* µs */, 0, 0, 0, 0);
            // #1945: the Sony spelling reports the libkernel-encoded form. The guest libc.prx's
            // `_Mtx_timedlock` compares against 0x8002003c exactly; a bare 60 became `_Thrd_error`
            // and threw. The FreeBSD numbering (60, not host 110) is still what gets encoded.
            CHECK(rc == prosper::hle::kSceKernelErrorETIMEDOUT,
                  "scePthreadMutexTimedlock expiry reports SCE-encoded FreeBSD ETIMEDOUT "
                  "(0x8002003c), not host 110 and not a bare 60");
            release.store(true, std::memory_order_release);
            holder.join();
        } else {
            std::printf("  [skip] scePthreadMutex* not registered on this platform\n");
        }
    }

    // ---- behavioral: sceKernelWaitOnAddress timeout ----------------------------------------
    // This returned 0x80020060 — decimal 60 written as hex, i.e. EOWNERDEAD(96) — while every other
    // timeout in the emulator encodes 0x8002003c. A guest comparing against the real
    // SCE_KERNEL_ERROR_ETIMEDOUT never matched it.
    {
        // Registered by raw NID: the 3.20 firmware symbol dump has no name for it, so prosper labels
        // it "sceKernelWaitOnAddress?" and there is no nid_hash() spelling to look it up by.
        auto wait_on_address = Hle::lookup("Hc4CaR6JBL0");
        if (wait_on_address) {
            // a0 = wait address, a1 = expected value (blocks while *addr == a1),
            // a2 = pointer to a uint32 microsecond timeout. Nothing ever writes `word`, so the wait
            // can only end by expiring.
            alignas(8) uint32_t word = 0x5a5a5a5au;
            uint32_t timeout_us = 20000;
            const auto started = std::chrono::steady_clock::now();
            const uint64_t rc = wait_on_address((uint64_t)(uintptr_t)&word, 0x5a5a5a5au,
                                                (uint64_t)(uintptr_t)&timeout_us, 0, 0, 0);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            CHECK(rc == prosper::hle::kSceKernelErrorETIMEDOUT,
                  "sceKernelWaitOnAddress timeout is SCE_KERNEL_ERROR_ETIMEDOUT (0x8002003c)");
            CHECK(rc != 0x80020060ull,
                  "sceKernelWaitOnAddress no longer reports EOWNERDEAD(96) for a timeout");
            CHECK(elapsed >= std::chrono::milliseconds(5),
                  "the timeout result came from a real expiry, not an immediate rejection");
        } else {
            std::printf("  [skip] sceKernelWaitOnAddress not registered on this platform\n");
        }
    }

    // ---- behavioral: pthread_setcancelstate is a POSIX contract, not an SCE one -------------
    {
        auto setcancelstate = Hle::lookup(nid_hash("pthread_setcancelstate"));
        if (setcancelstate) {
            const uint64_t rc = setcancelstate(7 /* neither ENABLE nor DISABLE */, 0, 0, 0, 0, 0);
            CHECK(rc == static_cast<uint64_t>(FreeBsdErrno::EInval),
                  "POSIX pthread_setcancelstate reports a plain errno (22), not 0x80020016");
        } else {
            std::printf("  [skip] pthread_setcancelstate not registered on this platform\n");
        }
    }

    std::printf(failures ? "== FAIL: %d ==\n" : "== PASS ==\n", failures);
    return failures ? 1 : 0;
}
