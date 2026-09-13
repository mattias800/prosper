// host_tcb_scope.hpp -- correct %fs to the CALLING host thread's own TCB for one call (#3615/#3623).
//
// Shared rather than local to one translation unit because the defect is not local: every guest
// synchronisation primitive whose host implementation resolves ownership through the TCB is exposed,
// and those live in more than one file. hle_ult.cpp is the second (its mutexes are ERRORCHECK and its
// condvar wait DISCARDS the return, so an EPERM there degrades silently into an unthrottled spin).
#pragma once

#include "hle/dispatch/dispatch.hpp"

#include <cstdint>

namespace prosper {

// #3623 -- CORRECT %fs FOR THE DURATION OF AN OWNERSHIP-SENSITIVE CALL.
//
// glibc does not resolve a mutex's owner from a syscall-derived thread id; it compares against the
// tid cached in the TCB that %fs points at. prosper's import stubs restore %fs from a stash held
// in the GUEST TCB, and that stash records the host %fs of whichever host thread first activated
// that TCB -- so an HLE reached after a guest TCB migrated between host threads runs on ANOTHER
// thread's host TCB, and glibc judges the caller to be a different thread. Measured on PPSA05684
// (#3615): identical tid, identical mutex, `owner` equal to the caller, and unlock still EPERM,
// with the two calls differing only in %fs.
//
// This also repairs prosper's OWN ownership map in the same stroke, because `pthread_self()` on
// glibc IS the TCB address -- so the map (guest_mutex_acquired and friends) was misreading the
// owner exactly as glibc was.
//
// DECLARE IT FIRST IN EACH BODY, which is not merely style. Destroyed last, the scope also covers the
// `ensure_mutex` / `ensure_cond` / `ensure_rwlock` calls that precede the host primitive -- so their
// allocations run under the CORRECTED TCB rather than against a foreign thread's malloc tcache. That
// is a second, independent reason the correction belongs at the top of the body rather than immediately
// around the host call (raised in review of #3624).
//
// Deliberately a SCOPE at the sync call sites, not a fix to the stub: repairing the stub is
// #3623, it sits on the hottest path in the emulator, and it needs its own design and cross-title
// validation. A no-op when guest TLS is off, when the thread never activated it, or when %fs is
// already this thread's own -- which is every call on every title that does not move a TCB.
struct HostTcbScope {
    HostTcbScope(const HostTcbScope&) = delete;
    HostTcbScope& operator=(const HostTcbScope&) = delete;
#if defined(__linux__) && !defined(__APPLE__) && defined(__x86_64__)
    uint64_t prev = 0;
    static uint64_t rd() { uint64_t v; __asm__ volatile("rdfsbase %0" : "=r"(v)); return v; }
    static void     wr(uint64_t v) { __asm__ volatile("wrfsbase %0" : : "r"(v)); }
    HostTcbScope() {
        // The order here is the safety argument: `rd()` is unreachable unless the registry has
        // an entry for this thread, and the registry's only writer runs AFTER `rd_fsbase()` has
        // already retired on this same thread. So rdfsbase/wrfsbase never execute on a build or
        // machine where guest %fs is not in use.
        const uint64_t want = guest_tls_host_fs_for_current_thread();
        if (!want) return;                       // guest TLS off, or thread never activated it
        const uint64_t cur = rd();
        if (cur && cur != want) { prev = cur; wr(want); }
    }
    ~HostTcbScope() { if (prev) wr(prev); }      // leave %fs exactly as the stub left it
#else
    // Every other platform: a well-formed no-op. The declarations above suppress the implicit
    // default constructor, so this arm is REQUIRED -- without it `HostTcbScope x;` does not
    // compile on Windows or macOS, which is exactly how the first version broke both builds.
    HostTcbScope() = default;
#endif
};

}  // namespace prosper
