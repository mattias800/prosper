// test_kernel_sem_init_error (#3068) - scePthreadSemInit must forward a real sem_init() failure
// rather than reporting success and publishing a pointer to an object that was never initialised.
//
// Before the fix, k_sem_init (hle_kernel.cpp) DISCARDED sem_init()'s return value, always returned
// SCE_OK, and always wrote the (possibly-uninitialised) sem_t pointer through the guest's handle
// slot. Every other member of the family -- k_sem_wait, k_sem_trywait, k_sem_timedwait, k_sem_post,
// k_sem_getvalue -- reads that pointer straight back out via ensure_sem() and operates on it, so a
// failed init was silently turned into undefined behaviour on every LATER call instead of one
// honest, traceable failure at init time.
//
// This matters most on Darwin, where unnamed POSIX semaphores are not implemented at all --
// sem_init() there is unconditionally ENOSYS -- but that specific errno cannot be produced on this
// host (Linux, where unnamed semaphores work). What CAN be produced here, and is a real, portable
// sem_init() failure rather than a Linux quirk, is POSIX's own SEM_VALUE_MAX check: sem_init(3)
// must fail EINVAL when the requested initial value exceeds SEM_VALUE_MAX. glibc enforces this
// (measured: sem_init(&s, 0, UINT32_MAX) -> rc=-1, errno=22).
//
// So rather than assuming that check exists on whatever platform this runs on, the test PROBES the
// host sem_init() directly with the exact value the guest call below also uses, and only asserts
// the guest-visible failure contract when the host call actually failed. That is the "prefer an
// experiment that detects its own invalidity" rule: on a hypothetical platform whose libc accepts
// an absurd value, this test explains why it has nothing to check instead of either flaking or
// passing for the wrong reason. On every platform this suite's CI actually runs (Linux: EINVAL,
// confirmed; Darwin: ENOSYS on any value, per Apple's documented sem_init) the probe fails and the
// regression arms below run for real.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <semaphore.h>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_kernel_sem_init_error ==\n");
    register_builtin_hle();

    HleFn sem_init_fn = Hle::lookup(nid_hash("scePthreadSemInit").c_str());
    CHECK(sem_init_fn != nullptr, "scePthreadSemInit is registered");
    if (!sem_init_fn) { printf("== FAIL (unresolved) ==\n"); return 1; }

    // UINT32_MAX: exceeds SEM_VALUE_MAX (INT_MAX on glibc, and Darwin rejects everything anyway)
    // on every platform this project targets.
    const unsigned kOutOfRange = 4294967295u;

    sem_t probe;
    errno = 0;
    const int host_rc = sem_init(&probe, 0, kOutOfRange);
    const int host_errno = errno;
    if (host_rc == 0) sem_destroy(&probe);
    printf("         (host sem_init(%u) -> rc=%d errno=%d)\n", kOutOfRange, host_rc, host_errno);

    if (host_rc == 0) {
        // Self-invalidating rather than silently green: this platform's sem_init doesn't reject
        // the probe value, so this run cannot exercise the failure path at all.
        printf("== SKIP (this platform's sem_init() accepted an out-of-range value; the failure "
               "path cannot be exercised here) ==\n");
        return 0;
    }

    // GUEST SIDE: same value, through the HLE entry point. The handle is a POINTER CELL (see the
    // note in test_kernel_sem_timedwait.cpp -- k_sem_init does `*(void**)a0 = s`, an 8-byte write),
    // pre-seeded with a sentinel that is neither nullptr nor a plausible heap pointer, so "the slot
    // was published" and "the slot was left alone" are distinguishable afterwards.
    void* slot = (void*)(uintptr_t)0x1;
    const uint64_t handle = (uint64_t)(uintptr_t)&slot;
    const uint64_t rc = sem_init_fn(handle, 0, kOutOfRange, 0, 0, 0);

    // THE MECHANISM ARM: pre-fix, k_sem_init discards sem_init's return and this is 0 (SCE_OK)
    // unconditionally regardless of what the host call did -- this is what reddens without #3068.
    CHECK(rc != 0, "scePthreadSemInit reports the real failure instead of manufactured success");

    // Sony spellings encode via sce_kernel_error() (0x8002_0000 | FreeBSD errno) -- decode it back
    // and check it names EINVAL, the FreeBSD errno for an out-of-range semaphore value.
    hle::FreeBsdErrno decoded{};
    const bool decodes = hle::sce_kernel_error_errno(rc, decoded);
    CHECK(decodes, "the failure is encoded the libkernel way (0x8002_0000 | errno), not a bare number");
    if (decodes)
        CHECK(decoded == hle::FreeBsdErrno::EInval,
              "...and names EINVAL, the FreeBSD errno for an out-of-range initial value");

    // THE OTHER HALF OF THE FIX: nothing was published into the guest's slot. Pre-fix, the handler
    // wrote `*(void**)a0 = s` unconditionally, so a failed init still handed the guest a pointer to
    // a sem_t that sem_init() never successfully touched -- and every later scePthreadSem* call
    // resolves that exact pointer out of the slot (ensure_sem) and dereferences it.
    CHECK(slot == (void*)(uintptr_t)0x1,
          "the guest slot is left untouched, not published with a pointer to storage that was "
          "never initialised");

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
