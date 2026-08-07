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
// The M3 and M4 arms matter most. M1/M2 are provoked with a null object, which yields EINVAL — and
// EINVAL is 22 on FreeBSD, on Linux, on Darwin and on MinGW, so a null-slot arm alone cannot tell a
// FreeBSD number from a host number. The EAGAIN row can: FreeBSD EAGAIN is 35 while the host's is
// 11, and 11 is FreeBSD's EDEADLK, so "would block, retry" arrives as "deadlock" if the mapping is
// skipped. The M4 arm guards the opposite error — `scePthreadBarrierWait` returns -1 for the serial
// thread, which is not an errno at all, and over-applying the encoding would corrupt it.

#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/sce_errno.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

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
};

// Every fallible entry point in the synchronisation-object families, in registration order. A null
// object provokes EINVAL in all of them (each body opens with a `!a0` / `ensure_*` guard), so one
// zero-argument call exercises every row uniformly.
const Row kRows[] = {
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
    {"scePthreadSemInit",            "sem_init",                    "#2178"},
    {"scePthreadSemWait",            "sem_wait",                    "#2178"},
    {"scePthreadSemTrywait",         "sem_trywait",                 "#2178"},
    {"scePthreadSemTimedwait",       nullptr,                       "#2178, in place"},
    {"scePthreadSemPost",            "sem_post",                    "#2178"},
    {"scePthreadSemGetvalue",        "sem_getvalue",                "#2178"},
    // --- barriers (no POSIX spelling registered on either body) ---
    {"scePthreadBarrierInit",        nullptr,                       "#2178, in place"},
    {"scePthreadBarrierWait",        nullptr,                       "#2178, in place"},
};

// Entry points whose body returns 0 on every path. They correctly have NO alias, and #2158
// established that its absence is the right answer rather than an oversight.
const char* const kInfallible[] = {
    "scePthreadMutexattrSetprotocol", "scePthreadMutexattrSetpshared", "scePthreadMutexattrDestroy",
    "scePthreadMutexDestroy", "scePthreadCondattrDestroy", "scePthreadCondDestroy",
    "scePthreadCondSignal", "scePthreadCondBroadcast", "scePthreadCondWait",
    "scePthreadRwlockDestroy", "scePthreadSemDestroy", "scePthreadBarrierDestroy",
    "scePthreadBarrierattrInit", "scePthreadBarrierattrDestroy", "scePthreadBarrierattrSetpshared",
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
        const uint64_t bare = call0(posix);
        snprintf(msg, sizeof msg, "%-30s -> bare EINVAL 22 (got %llu)",
                 r.posix, (unsigned long long)bare);
        CHECK(bare == kBareEINVAL, msg);
    }

    // ===== 2. M3: a forwarded HOST errno, where host and FreeBSD numbering disagree ============
    // scePthreadSemTrywait on an empty semaphore fails with EAGAIN. FreeBSD EAGAIN is 35; the host's
    // is 11, which is FreeBSD's EDEADLK. This row is the only one that can tell a correct mapping
    // from a passed-through host number, because every other row's errno is EINVAL(22), which every
    // platform agrees on. It also carries the full contract at once: Sony gets 0x80020023, POSIX
    // gets 35, and neither gets 11.
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
                  "(0x80020023) — not a bare 35, and not the host's 11");
            CHECK(posix_trywait((uint64_t)(uintptr_t)slot, 0, 0, 0, 0, 0) == 35,
                  "sem_trywait on the same semaphore reports bare FreeBSD EAGAIN (35)");
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
                printf("  [SKIP] this host permits unlocking an unheld ERRORCHECK mutex — the"
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

    // ===== 6. the other side of "iff": handlers that correctly have no alias ==================
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

    if (fails) printf("== FAIL (%d) ==\n", fails);
    else       printf("== PASS ==\n");
    return fails ? 1 : 0;
}
