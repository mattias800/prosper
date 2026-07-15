// hle_kernel.cpp — HLE of libkernel primitives. Threading/sync are backed by host
// pthreads (guest ABI == host SysV ABI). Sony pthread types are opaque pointer
// handles: scePthreadMutexInit(&handle, &attr, name) allocates the object and stores
// the pointer through the caller's handle slot, returning 0 on success.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE   // pthread_getattr_np
#endif
#ifdef _WIN32
// Must precede any header that transitively includes windows.h (e.g. winpthreads <pthread.h>) so
// GetCurrentThreadStackLimits (needs _WIN32_WINNT >= 0x0602 / Win8) is declared.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#endif
#include "dispatch.hpp"
#include "nid.hpp"
#include "sync_futex.hpp"
#include "../gpu/mb3_freelist.hpp"
#include "../host/exec_image.hpp"
#include <pthread.h>
#include <semaphore.h>   // scePthreadSem* -> host sem_t
#include "../host/posix_shim.hpp"   // Darwin: sem/barrier/timedlock/getattr_np/sigqueue compat
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <new>
#ifdef _WIN32
#include <windows.h>   // GetCurrentThreadStackLimits/GetCurrentThreadId for the guest-thread trampoline
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <signal.h>
#ifdef __linux__
#include <ucontext.h>
#endif
#endif

namespace prosper {
namespace { bool sclog() { static int v = getenv("PROSPER_SYNCLOG") ? 1 : 0; return v; }
    long sctid() {
#if defined(__linux__) || defined(__APPLE__)
        return (long)prosper_gettid();
#else
        return 0;
#endif
    }
}
}
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)

// --- mutex attributes (opaque; we back with host pthread_mutexattr_t) ---
HLE(k_mutexattr_init) {
    if (!a0) return 0x16; // EINVAL-ish
    auto* at = (pthread_mutexattr_t*)calloc(1, sizeof(pthread_mutexattr_t));
    pthread_mutexattr_init(at);
    // A FRESH attr's type must be the FreeBSD/Sony DEFAULT — ERRORCHECK — not the host default
    // (glibc: NORMAL). Kyty's PthreadMutexattrInit does exactly this (settype(1) on init). The
    // #183 change covered the no-attr init path but left attr-without-settype at glibc NORMAL,
    // which self-deadlocks guests that build their own recursion on the EDEADLK contract: UE4's
    // PS5 lock wrapper (PPSA17942 eboot+0x24ca4b6) does
    //   err = mutex_lock(obj); if (err) /* EDEADLK: already mine */ skip-acquire; depth++;
    // — with a NORMAL mutex the relock BLOCKS forever instead of returning EDEADLK (live-verified:
    // single-thread wedge in k_mutex_lock, mutex __owner == self, ~10 s into the DOLL boot).
    // CONFIDENCE: HIGH (FreeBSD contract + Kyty + live wedge disassembly).
    pthread_mutexattr_settype(at, PTHREAD_MUTEX_ERRORCHECK);
    *(void**)a0 = at;                     // store handle through caller's slot
    return 0;
}
// Sony/FreeBSD mutex types: 1=ERRORCHECK (the FreeBSD/Sony DEFAULT), 2=RECURSIVE, 3=NORMAL,
// 4=ADAPTIVE_NP. The guest-visible SELF-LOCK contract is what matters for the host mapping
// (FreeBSD libthr mutex_self_lock): ERRORCHECK and ADAPTIVE_NP return EDEADLK; only NORMAL
// hard-deadlocks. So type 4 maps to host ERRORCHECK (adaptive is errorcheck + a spin heuristic —
// pure performance), NOT to host NORMAL: glibc ADAPTIVE/NORMAL self-lock blocks forever. Kyty
// maps 3/4 -> NORMAL, which self-deadlocks any guest that builds recursion on the EDEADLK
// contract — exactly what UE4's PS5 lock wrapper does (PPSA17942 eboot+0x24ca4b6:
//   err = mutex_lock(obj); if (err) /* EDEADLK: already mine */ skip-acquire; depth++;
// its mutexes are created with settype(4) — live-captured via PROSPER_MUTEXLOG at the #208-era
// DOLL boot wedge: single-thread self-deadlock in k_mutex_lock, mutex __owner == self).
// Weight Kyty DOWN here: no PS4 title it runs exercises adaptive self-lock; FreeBSD libthr is
// the platform contract. CONFIDENCE: HIGH (FreeBSD source + the live wedge -> unwedge flip).
HLE(k_mutexattr_settype) {
    if (!a0 || !*(void**)a0) return 0x16;
    int host;
    switch ((int)a1) {
        case 1: host = PTHREAD_MUTEX_ERRORCHECK; break;
        case 2: host = PTHREAD_MUTEX_RECURSIVE;  break;
        case 3: host = PTHREAD_MUTEX_NORMAL; break;
        case 4: host = PTHREAD_MUTEX_ERRORCHECK; break;   // ADAPTIVE_NP: EDEADLK on self-lock
        default: return 0x16;   // EINVAL
    }
    static const bool mtxlog = getenv("PROSPER_MUTEXLOG") != nullptr;
    if (mtxlog) fprintf(stderr, "[mtx] settype attr=%p type=%d\n", *(void**)a0, (int)a1);
    pthread_mutexattr_settype((pthread_mutexattr_t*)*(void**)a0, host);
    return 0;
}
// scePthreadMutexattrGettype: the read-back inverse of settype. Was MISSING -> the generic stub
// returned 0 while leaving the caller's `int* type` (a1) unwritten, so the guest read stack garbage as
// the mutex type (the harmful-Get-stub class the affinity/sched paths already guard against). Translate
// host type -> Sony (ERRORCHECK->1, RECURSIVE->2, NORMAL->3), the inverse of k_mutexattr_settype above.
HLE(k_mutexattr_gettype) {
    if (!a0 || !*(void**)a0 || !a1) return 0x16;   // EINVAL
    int host = PTHREAD_MUTEX_NORMAL;
    pthread_mutexattr_gettype((pthread_mutexattr_t*)*(void**)a0, &host);
    int sony = (host == PTHREAD_MUTEX_ERRORCHECK) ? 1 : (host == PTHREAD_MUTEX_RECURSIVE) ? 2 : 3;
    *(int*)(uintptr_t)a1 = sony;
    return 0;
}
HLE(k_mutexattr_setprotocol) { return 0; }

// __stack_chk_fail (Ou3iL1abvng): the guest's -fstack-protector epilogue jumps here when its stack
// canary check fails; the [[noreturn]] contract lets the compiler place a trailing UD2, so our old
// blanket "return 0" stub fell into that UD2 -> SIGILL crash on the New Game path (#163-progress). Log
// the current guest canary at %fs:0x28: a normal high-entropy value (low byte 0x00) => the TLS is fine
// and a real stack overflow corrupted the on-stack copy; 0 / a weird value => a guest-TCB canary
// divergence. Still returns (the crash is unavoidable here) — this is diagnostic-only for now.
HLE(k_stack_chk_fail) {
    uint64_t c = 0;
    __asm__ volatile ("movq %%fs:0x28, %0" : "=r"(c));
    static int n = 0;
    if (n++ < 4) fprintf(stderr, "[stackchk] __stack_chk_fail: guest canary @fs:0x28 = 0x%016llx\n",
                         (unsigned long long)c);
    return 0;
}
HLE(k_mutexattr_setpshared)  { return 0; }
HLE(k_mutexattr_destroy)     { if (a0 && *(void**)a0) { free(*(void**)a0); *(void**)a0 = nullptr; } return 0; }

// --- mutexes ---
// FreeBSD/PS5 pthread objects are POINTERS, and a STATICALLY-INITIALIZED object is a small
// sentinel (PTHREAD_MUTEX_INITIALIZER == NULL, adaptive == 1, ...). Real libthr SELF-INITIALIZES
// such an object on first use (Kyty models the same with its PthreadStaticObject lazy creation).
// We previously treated a NULL handle as a no-op that RETURNED SUCCESS — which silently voided
// every statically-initialized lock. ROOT CAUSE of the level1 scene-load heap corruption: the
// IL2CPP/bdwgc GC allocation lock (GC_allocate_ml, a static PTHREAD_MUTEX_INITIALIZER in
// Il2cppUserAssemblies' .bss, locked via libScePosix pthread_mutex_trylock/lock) never locked,
// so the collector raced concurrently-allocating / lazy-sweeping mutator threads and corrupted
// the GC free lists ("Rewired_" ASCII popped as a free-list link, marker ABORT "Unexpected
// state", varying downstream SIGSEGVs). Self-init closes that class for mutex/cond/rwlock.
// CONFIDENCE: HIGH (semantics cross-checked against FreeBSD libthr + the Kyty reference).
namespace {
    inline bool pt_static_sentinel(void* v) { return (uintptr_t)v < 0x1000; }  // NULL/1/2/3... = static initializer
    pthread_mutex_t* ensure_mutex(uint64_t slot_addr) {
        if (!slot_addr) return nullptr;
        void** slot = (void**)(uintptr_t)slot_addr;
        void* cur = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
        if (!pt_static_sentinel(cur)) return (pthread_mutex_t*)cur;
        auto* m = (pthread_mutex_t*)calloc(1, sizeof(pthread_mutex_t));
        pthread_mutexattr_t at; pthread_mutexattr_init(&at);
        // The FreeBSD static sentinel ENCODES the type: NULL = PTHREAD_MUTEX_INITIALIZER (the
        // DEFAULT type, which is ERRORCHECK on FreeBSD), 1 = PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP.
        // BOTH map to host ERRORCHECK: FreeBSD's self-lock contract returns EDEADLK for the
        // default AND adaptive types (libthr mutex_self_lock; adaptive's spin is pure
        // performance). Previously every sentinel self-init forced RECURSIVE (#145).
        pthread_mutex_init(m, &at);
        pthread_mutexattr_destroy(&at);
        if (__atomic_compare_exchange_n(slot, &cur, (void*)m, false,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            return m;
        pthread_mutex_destroy(m); free(m);          // lost the init race: use the winner's object
        return (pthread_mutex_t*)cur;
    }
    pthread_cond_t* ensure_cond(uint64_t slot_addr) {
        if (!slot_addr) return nullptr;
        void** slot = (void**)(uintptr_t)slot_addr;
        void* cur = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
        if (!pt_static_sentinel(cur)) return (pthread_cond_t*)cur;
        auto* c = (pthread_cond_t*)calloc(1, sizeof(pthread_cond_t));
        pthread_cond_init(c, nullptr);
        if (__atomic_compare_exchange_n(slot, &cur, (void*)c, false,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            return c;
        pthread_cond_destroy(c); free(c);
        return (pthread_cond_t*)cur;
    }
    pthread_rwlock_t* ensure_rwlock(uint64_t slot_addr) {
        if (!slot_addr) return nullptr;
        void** slot = (void**)(uintptr_t)slot_addr;
        void* cur = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
        if (!pt_static_sentinel(cur)) return (pthread_rwlock_t*)cur;
        auto* rw = (pthread_rwlock_t*)calloc(1, sizeof(pthread_rwlock_t));
        pthread_rwlock_init(rw, nullptr);
        if (__atomic_compare_exchange_n(slot, &cur, (void*)rw, false,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            return rw;
        pthread_rwlock_destroy(rw); free(rw);
        return (pthread_rwlock_t*)cur;
    }
    // scePthreadSem*: read the sem_t created by scePthreadSemInit (no static SEM_INITIALIZER, so no lazy
    // self-init -- Init is the sole creator). Returns null if the guest never called Init.
    sem_t* ensure_sem(uint64_t slot_addr) {
        if (!slot_addr) return nullptr;
        void* cur = __atomic_load_n((void**)(uintptr_t)slot_addr, __ATOMIC_ACQUIRE);
        return pt_static_sentinel(cur) ? nullptr : (sem_t*)cur;
    }
    // scePthreadBarrier*: read the pthread_barrier_t created by BarrierInit (Init is the sole creator).
    pthread_barrier_t* ensure_barrier(uint64_t slot_addr) {
        if (!slot_addr) return nullptr;
        void* cur = __atomic_load_n((void**)(uintptr_t)slot_addr, __ATOMIC_ACQUIRE);
        return pt_static_sentinel(cur) ? nullptr : (pthread_barrier_t*)cur;
    }
}
// scePthreadMutexInit(mutex_slot, attr_slot, name) / pthread_mutex_init(mutex_slot, attr_slot):
// honor the caller's attr (type set via k_mutexattr_settype); no attr means Sony's default,
// ERRORCHECK — FreeBSD PTHREAD_MUTEX_DEFAULT, and what Kyty's attr-init settype(1) produces (#145).
HLE(k_mutex_init) {
    if (!a0) return 0x16;
    auto* m = (pthread_mutex_t*)calloc(1, sizeof(pthread_mutex_t));
    pthread_mutexattr_t* at = (a1 && !pt_static_sentinel(*(void**)a1)) ? (pthread_mutexattr_t*)*(void**)a1 : nullptr;
    pthread_mutexattr_t def;
    if (!at) { pthread_mutexattr_init(&def); pthread_mutexattr_settype(&def, PTHREAD_MUTEX_ERRORCHECK); at = &def; }
    pthread_mutex_init(m, at);
    static const bool mtxlog = getenv("PROSPER_MUTEXLOG") != nullptr;
    if (mtxlog) {
        int t = -1; pthread_mutexattr_gettype(at, &t);
        fprintf(stderr, "[mtx] init m=%p slot=0x%llx attr=%p type=%d\n", (void*)m,
                (unsigned long long)a0, at == &def ? nullptr : (void*)at, t);
    }
    if (at == &def) pthread_mutexattr_destroy(&def);
    *(void**)a0 = m;
    return 0;
}
// The guest sees FreeBSD errno values; EBUSY(16)/EPERM(1)/EINVAL(22) coincide with both Linux and
// MinGW, but EDEADLK differs on every host (FreeBSD 11, Linux 35, MinGW/winpthreads 36) — an
// ERRORCHECK relock must report the FreeBSD value. `EDEADLK` is the HOST's own constant, so the
// comparison remaps whatever the host returns on ALL platforms (the Windows CI build hit exactly
// this: winpthreads returned its EDEADLK=36 and the old __linux__-only guard left it unremapped).
namespace { inline uint64_t fbsd_errno(int host) {
    if (host == EDEADLK) return 11;
    return (uint64_t)(unsigned)host;
} }
HLE(k_mutex_destroy) { if (a0 && !pt_static_sentinel(*(void**)a0)) { pthread_mutex_destroy((pthread_mutex_t*)*(void**)a0); free(*(void**)a0); } if (a0) *(void**)a0 = nullptr; return 0; }
// PROSPER_MUTEX_FAILLOG: report any mutex op that returns a non-zero (EINVAL/EDEADLK/...) result,
// with the guest slot address and the host object it resolved to. Diagnostic for the macOS
// guest-side "std::mutex lock failed: Invalid argument" terminate — a host EINVAL(22) surfaces as
// a FreeBSD EINVAL(0x16) to the guest's libc++, which throws and terminates. This pinpoints the
// exact slot/host-pointer producing the bad lock without needing a debugger (unusable under Rosetta).
namespace {
    inline uint64_t mtx_report(const char* op, uint64_t slot, pthread_mutex_t* m, int host) {
        static const bool on = getenv("PROSPER_MUTEX_FAILLOG") != nullptr;
        if (on && host != 0)
            fprintf(stderr, "[mtx-fail] %s slot=0x%llx host_m=%p rc=%d(%s)\n", op,
                    (unsigned long long)slot, (void*)m, host, strerror(host));
        return fbsd_errno(host);
    }
}
HLE(k_mutex_lock)    { auto* m = ensure_mutex(a0); return m ? mtx_report("lock",    a0, m, pthread_mutex_lock(m))    : 0x16; }
HLE(k_mutex_trylock) { auto* m = ensure_mutex(a0); return m ? mtx_report("trylock", a0, m, pthread_mutex_trylock(m)) : 0x16; }
HLE(k_mutex_unlock)  { auto* m = ensure_mutex(a0); return m ? mtx_report("unlock",  a0, m, pthread_mutex_unlock(m))  : 0x16; }

// --- condition variables ---
HLE(k_condattr_init)    { if (a0) { auto* c = (pthread_condattr_t*)calloc(1, sizeof(pthread_condattr_t)); pthread_condattr_init(c); *(void**)a0 = c; } return 0; }
HLE(k_condattr_destroy) { if (a0 && *(void**)a0) { free(*(void**)a0); *(void**)a0 = nullptr; } return 0; }
HLE(k_cond_init)      { if (!a0) return 0x16; auto* c = (pthread_cond_t*)calloc(1, sizeof(pthread_cond_t)); pthread_cond_init(c, nullptr); *(void**)a0 = c; return 0; }
HLE(k_cond_destroy)   { if (a0 && !pt_static_sentinel(*(void**)a0)) { pthread_cond_destroy((pthread_cond_t*)*(void**)a0); free(*(void**)a0); } if (a0) *(void**)a0 = nullptr; return 0; }
HLE(k_cond_signal)    { if (sclog()) fprintf(stderr, "[sync2] T%ld COND.signal    cond=0x%llx\n", sctid(), a0 ? (unsigned long long)*(void**)a0 : 0); if (auto* c = ensure_cond(a0)) pthread_cond_signal(c); return 0; }
HLE(k_cond_broadcast) { if (sclog()) fprintf(stderr, "[sync2] T%ld COND.broadcast cond=0x%llx\n", sctid(), a0 ? (unsigned long long)*(void**)a0 : 0); if (auto* c = ensure_cond(a0)) pthread_cond_broadcast(c); return 0; }
HLE(k_cond_wait)      { if (sclog()) fprintf(stderr, "[sync2] T%ld COND.wait.ent  cond=0x%llx\n", sctid(), a0 ? (unsigned long long)*(void**)a0 : 0);
    { auto* c = ensure_cond(a0); auto* m = ensure_mutex(a1);
      if (c && m) pthread_cond_wait(c, m); }
    if (sclog()) fprintf(stderr, "[sync2] T%ld COND.wait.exit cond=0x%llx\n", sctid(), a0 ? (unsigned long long)*(void**)a0 : 0); return 0; }
// POSIX pthread_cond_timedwait(cond_slot, mutex_slot, const timespec* abstime) — abstime is an
// ABSOLUTE CLOCK_REALTIME deadline ({i64 sec, i64 nsec}, FreeBSD == Linux x86-64 layout), and the
// POSIX shim returns the errno VALUE directly (FreeBSD ETIMEDOUT = 60). This was an unimplemented
// stub returning 0 = "signaled": UE's IAsyncReadRequest::WaitCompletion(timeout) loop (guest
// eboot+0x22ea8ca, live-caught spinning at 100% CPU with RA 0x4022ea954 inside prosper_on_unimpl)
// re-checked its predicate and retried instantly, forever. The cond/mutex SLOTS are the same
// pointer-slot scheme as scePthreadCond* (the same objects flow through k_cond_wait's infinite
// branch), so ensure_cond/ensure_mutex map them to the identical host objects.
// CONFIDENCE: HIGH (POSIX semantics; spin diagnosed live; FreeBSD errno per Kyty Errno.h).
HLE(k_cond_timedwait) {
    auto* c = ensure_cond(a0); auto* m = ensure_mutex(a1);
    if (!c || !m) return 22;                                   // EINVAL
    if (!a2) { pthread_cond_wait(c, m); return 0; }
    const int64_t* gts = (const int64_t*)(uintptr_t)a2;
    struct timespec dl { (time_t)gts[0], (long)gts[1] };
    int rc = pthread_cond_timedwait(c, m, &dl);
    if (rc == ETIMEDOUT) return 60;                            // FreeBSD ETIMEDOUT
    return (uint64_t)rc;
}

// --- read/write locks (opaque handle -> host pthread_rwlock_t). Real libc.prx uses these for its
// internal state (locale, stdio, malloc arenas); stubbing them to no-ops leaves that state UNLOCKED
// under the 15-thread IL2CPP pool -> data races. Real locking is a genuine correctness fix. ---
HLE(k_rwlock_init) {
    if (!a0) return 0x16;
    auto* rw = (pthread_rwlock_t*)calloc(1, sizeof(pthread_rwlock_t));
    pthread_rwlock_init(rw, nullptr);
    *(void**)a0 = rw;   // store handle through caller's slot (a1=attr, a2=name ignored)
    return 0;
}
HLE(k_rwlock_destroy) { if (a0 && !pt_static_sentinel(*(void**)a0)) { pthread_rwlock_destroy((pthread_rwlock_t*)*(void**)a0); free(*(void**)a0); } if (a0) *(void**)a0 = nullptr; return 0; }
HLE(k_rwlock_rdlock)  { if (auto* rw = ensure_rwlock(a0)) pthread_rwlock_rdlock(rw); return 0; }
HLE(k_rwlock_wrlock)  { if (auto* rw = ensure_rwlock(a0)) pthread_rwlock_wrlock(rw); return 0; }
HLE(k_rwlock_unlock)  { if (auto* rw = ensure_rwlock(a0)) pthread_rwlock_unlock(rw); return 0; }
HLE(k_rwlock_tryrdlock){ auto* rw = ensure_rwlock(a0); return rw ? (uint64_t)pthread_rwlock_tryrdlock(rw) : 0x16; }
HLE(k_rwlock_trywrlock){ auto* rw = ensure_rwlock(a0); return rw ? (uint64_t)pthread_rwlock_trywrlock(rw) : 0x16; }

// scePthreadOnce(once_control*, init_routine): run init exactly once, PER CONTROL. The old
// implementation held ONE process-global recursive mutex across the guest init routine — if init
// routine A blocked on another thread's progress and that thread touched ANY other once-control,
// both deadlocked (impossible on real hardware, which serializes per control). Now the control
// word itself is the state machine (0 = not run, 2 = running, 1 = done — same shape as
// h_execute_once in hle_libc.cpp), losers park on a condvar, and the init routine runs OUTSIDE
// the lock so independent once-inits never serialize. Same-thread re-entry (an init calling
// scePthreadOnce on ANOTHER control) works naturally; re-entry on the SAME control would be a
// guest bug on real hardware too (pthread_once self-deadlocks there).
HLE(k_pthread_once) {
    auto* ctl = (std::atomic<int>*)(uintptr_t)a0;
    auto init = (void (*)())(uintptr_t)a1;
    if (!ctl || !init) return 0x16;
    static std::mutex om;                    // guards state TRANSITIONS only, never held during init()
    static std::condition_variable ocv;
    for (;;) {
        if (ctl->load(std::memory_order_acquire) == 1) return 0;   // fast path: already done
        std::unique_lock<std::mutex> lk(om);
        int v = ctl->load(std::memory_order_acquire);
        if (v == 1) return 0;
        if (v == 2) { ocv.wait(lk); continue; }   // another thread runs this control's init
        ctl->store(2, std::memory_order_relaxed);
        lk.unlock();
        init();
        lk.lock();
        ctl->store(1, std::memory_order_release);
        lk.unlock();
        ocv.notify_all();
        return 0;
    }
}

// --- thread identity ---
// Return the real host thread handle as the Sony ScePthread — unique and stable per
// thread, used as an opaque id (stored/compared, not dereferenced). A constant here
// collides across threads and breaks per-thread lookups (GC, TLS).
HLE(k_pthread_self) { return (uint64_t)pthread_self(); }
HLE(k_pthread_equal){ return (uint64_t)(a0 == a1); }
HLE(k_pthread_yield){ sched_yield(); return 0; }

// --- thread attributes ---
HLE(k_attr_init)        { if (a0) { auto* at = (pthread_attr_t*)calloc(1, sizeof(pthread_attr_t)); pthread_attr_init(at); *(void**)a0 = at; } return 0; }
HLE(k_attr_destroy)     { if (a0 && *(void**)a0) { pthread_attr_destroy((pthread_attr_t*)*(void**)a0); free(*(void**)a0); *(void**)a0 = nullptr; } return 0; }
HLE(k_attr_setstacksize){ if (a0 && *(void**)a0 && a1 >= 16384) pthread_attr_setstacksize((pthread_attr_t*)*(void**)a0, a1); return 0; }
HLE(k_attr_noop)        { return 0; }

// sceKernelGetSanitizerMallocReplaceExternal returns the process replacement table, even when no
// sanitizer callbacks are installed. Returning nullptr made runtimes dereference null while checking
// the table (Dead Cells reads +0x18 during module initialization). Layout is shared by Kyty/SharpEmu:
// one size qword followed by 13 function pointers.
struct SanitizerMallocReplace {
    uint64_t size = sizeof(SanitizerMallocReplace);
    uint64_t callbacks[13]{};
};
static_assert(sizeof(SanitizerMallocReplace) == 0x70);
HLE(k_get_sanitizer_malloc_replace) {
    static SanitizerMallocReplace table;
    return (uint64_t)(uintptr_t)&table;
}
// scePthreadAttrSetdetachstate: store DETACHED/JOINABLE into the host attr so k_pthread_create (which
// reads pthread_attr_getdetachstate and applies it, line ~465) actually honors a detached-thread request.
// This was a no-op, so a thread the guest marked DETACHED was created JOINABLE and, never joined, leaked
// its TCB/stack -> OOM under thread churn. Sony DETACHED=1/JOINABLE=0 map 1:1 to the host enum.
HLE(k_attr_setdetachstate) {
    if (!a0 || !*(void**)a0) return 0x16;   // EINVAL
    pthread_attr_setdetachstate((pthread_attr_t*)*(void**)a0,
                                a1 ? PTHREAD_CREATE_DETACHED : PTHREAD_CREATE_JOINABLE);
    return 0;
}
HLE(k_attr_getdetachstate) {
    if (!a0 || !*(void**)a0 || !a1) return 0x16;   // EINVAL (was MISSING -> left *state uninitialized)
    int st = PTHREAD_CREATE_JOINABLE;
    pthread_attr_getdetachstate((pthread_attr_t*)*(void**)a0, &st);
    *(int*)(uintptr_t)a1 = (st == PTHREAD_CREATE_DETACHED) ? 1 : 0;
    return 0;
}

// Query the CURRENT thread's real attributes into the caller's attr object (the GC needs
// accurate stack bounds to scan roots; bad bounds make IL2CPP's GC init assert).
HLE(k_attr_get) {
    // scePthreadAttrGet(ScePthread thread, ScePthreadAttr* attr): fill *attr with the given
    // thread's attributes. a0 = thread handle (== the host pthread_t we store), a1 = attr handle.
    // (Bug fixed: the attr is arg1, not arg0 — reading arg0 left the real attr empty, so the GC's
    // GC_get_stack_base got a 0 stack base -> "Bad stack base in GC_register_my_thread".)
    if (a1 && *(void**)a1) {
        auto* at = (pthread_attr_t*)*(void**)a1;
        void* base = nullptr; size_t sz = 0;
        bool ok = (a0 && guest_stack_for_thread(a0, &base, &sz)) ||
                  guest_stack_for_current_thread(&base, &sz);
        if (ok) pthread_attr_setstack(at, base, sz);   // real, tracked stack for that thread
        // else: leave the attr as-is (avoid the fragile pthread_getattr_np)
    }
    return 0;
}
// scePthreadAttrGetstackaddr(attr, void** addr) — Sony reports the stack *base* (low addr).
HLE(k_attr_getstackaddr) {
    if (a0 && *(void**)a0 && a1) {
        void* base = nullptr; size_t sz = 0;
        pthread_attr_getstack((pthread_attr_t*)*(void**)a0, &base, &sz);
        *(void**)(uintptr_t)a1 = base;
    }
    return 0;
}
HLE(k_attr_getstacksize) {
    if (a0 && *(void**)a0 && a1) {
        void* base = nullptr; size_t sz = 0;
        pthread_attr_getstack((pthread_attr_t*)*(void**)a0, &base, &sz);
        *(size_t*)(uintptr_t)a1 = sz;
    }
    return 0;
}
// scePthreadAttrGetaffinity(attr, SceKernelCpumask* mask): report all 8 PS5 cores available.
// Returning 0 (the old stub) yields an EMPTY mask -> the guest may conclude no CPUs are usable.
// PROSPER_ONE_CPU (default off): report a SINGLE core (0x01). Unity sizes its job-system worker
// pool from the available-core mask; a 1-core mask makes it run jobs on the main thread instead of
// spawning ~8 worker threads. That eliminates the async-load thread races (e.g. a WorkerThread whose
// Stopwatch/+0x40 isn't created yet when the main thread times it — see docs/CUTSCENE_PROGRESSION.md).
// CONFIDENCE: MED — the mask→worker-count coupling is the standard Unity behavior; gated so default
// boot (0xff) is unchanged.
HLE(k_attr_getaffinity) {
    static const bool one = getenv("PROSPER_ONE_CPU") != nullptr;
    if (a1) *(uint64_t*)(uintptr_t)a1 = one ? 0x01 : 0xff;
    return 0;
}
// Scheduling Get* handlers. The Set* side is a legitimate no-op (we don't re-prioritize host
// threads), but a Get* that returns SUCCESS while never writing its out-params hands the caller
// uninitialized stack memory — the exact harmful-stub class k_attr_getaffinity above was fixed
// for. Report deterministic defaults: policy SCHED_OTHER(=1)-equivalent RR, priority 700 (the
// Sony default thread priority; Kyty Pthread.cpp uses the same 700 midpoint).
HLE(k_getschedparam)      { // scePthread/pthread_getschedparam(thread, int* policy, SchedParam* param)
    if (a1) *(int32_t*)(uintptr_t)a1 = 1;
    if (a2) *(int32_t*)(uintptr_t)a2 = 700;   // SceKernelSchedParam = { int sched_priority }
    return 0;
}
HLE(k_attr_getschedparam) { // scePthreadAttrGetschedparam(attr, SchedParam* param)
    if (a1) *(int32_t*)(uintptr_t)a1 = 700;
    return 0;
}
HLE(k_getprio)            { // scePthreadGetprio(thread, int* prio)
    if (a1) *(int32_t*)(uintptr_t)a1 = 700;
    return 0;
}

// --- thread creation: run the guest entry on a real host thread (ABI matches) ---
// Worker stacks are GLIBC-OWNED (#138): create passes only a stacksize (honoring the guest attr),
// so glibc allocates the stack and — crucially — RECLAIMS it when the thread is joined or a
// detached thread exits. The old code mmap'd a fixed 8 MiB per thread via pthread_attr_setstack
// and never munmapped it (glibc never frees a CALLER-owned stack): a thread-churning title leaked
// 8 MiB per exited thread. The trampoline still TRACKS the bounds (via pthread_getattr_np, keyed
// by our tid) so GC/thread-stack queries stay accurate, and unregisters them on exit — pthread
// ids are recycled, so a stale entry would serve the next thread the dead thread's bounds.
#if defined(__linux__) || defined(__APPLE__)
namespace {
struct ThreadStart { void* (*entry)(void*); void* arg; char name[16]; };
// Runs first on the new thread: register our own stack (keyed by our tid) BEFORE any guest code,
// so an early GC_register_my_thread / stack-base query from this thread finds it. Closes a race
// where a fast-starting worker ran before the parent's post-create registration → "Bad stack base".
void* thread_trampoline(void* p) {
    auto* ts = (ThreadStart*)p;
    {
        pthread_attr_t sa;
        if (pthread_getattr_np(pthread_self(), &sa) == 0) {
            void* sb = nullptr; size_t ss = 0;
            if (pthread_attr_getstack(&sa, &sb, &ss) == 0 && sb)
                register_thread_stack((uint64_t)pthread_self(), sb, ss);
            pthread_attr_destroy(&sa);
        }
    }
    install_sigaltstack();   // so a guest stack overflow on this worker is still catchable
    // Adopt the guest's thread name (scePthreadCreate/pthread_create_name_np arg) as the HOST
    // thread name so debugger/procfs views (gdb, /proc/PID/task/*/comm) show the engine's own
    // role names (GameThread, RenderThread, RHIThread, ...) instead of the binary name. Kernel
    // limit is 15 chars + NUL; host libc call, so it must run on the host %fs (we are).
#ifdef __APPLE__
    if (ts->name[0]) pthread_setname_np(ts->name);   // Darwin can only name the CURRENT thread (which this is)
#else
    if (ts->name[0]) pthread_setname_np(pthread_self(), ts->name);
#endif
    auto entry = ts->entry; void* arg = ts->arg; free(ts);   // all host libc — MUST run on the host %fs
    // gated (PROSPER_GUEST_FS): give this guest worker its own guest TCB + static TLS and switch %fs to it
    // as the LAST host action before entering guest code (so guest initial-exec TLS — incl. libc.prx's
    // allocator arena/tcache — resolves to real guest storage, not the aliased host glibc TCB). The import
    // stubs swap back to host %fs per HLE call. No-op when the gate is off. Order matters: the free() above
    // is host glibc (host-TLS tcache) — running it under the guest %fs corrupts the host heap.
    arm_hwbp_this_thread();   // no-op unless PROSPER_HWBP_ALLTHREADS; MUST run on host %fs (host libc calls)
    guest_tls_activate_thread();
    void* rv = entry ? entry(arg) : nullptr;
    // Normal-return exit path: purge this thread's __tls_get_addr DTV entries and free the blocks
    // (#68 — glibc recycles pthread ids, so a stale entry would hand the NEXT thread on this id the
    // dead thread's dirty TLS). The guest entry can return with %fs still = the guest TP
    // (PROSPER_GUEST_FS), and the purge is host libc (mutex/free), so switch permanently to host
    // %fs before cleanup. This is a terminal guest boundary: restoring guest %fs before returning
    // makes glibc's start_thread cleanup read its TLS through the guest TCB (#644). Threads that
    // exit via scePthreadExit never get here — its import gate already restored host %fs and
    // k_pthread_exit purges before host pthread_exit.
    (void)guest_fs_to_host_scoped();
    tls_dtv_purge_current_thread();
    unregister_thread_stack((uint64_t)pthread_self());   // ids recycle; stale bounds = wrong bounds (#138)
    return rv;
}
}
#elif defined(_WIN32)
namespace {
// Windows guest-thread trampoline. The bare `pthread_create(entry, arg)` the Windows path used
// before had two fatal bugs: (1) winpthreads calls `entry(arg)` with the MS x64 ABI (arg in rcx),
// but the guest entry is System V (reads rdi) — so the worker got a garbage arg and crashed on a
// null-derived pointer; (2) it never ran guest_tls_activate_thread(), so the worker had no guest
// %fs TCB and its initial-exec TLS was wrong. This trampoline fixes both: activate the guest %fs
// TCB, then call the guest entry through the SysV marshalling shim, and purge DTV on exit.
struct WinThreadStart { uint64_t entry; void* arg; };
extern "C" uint64_t prosper_call_guest_sysv(uint64_t fn, uint64_t a0, uint64_t a1);
void* win_thread_trampoline(void* p) {
    auto* ts = (WinThreadStart*)p;
    uint64_t entry = ts->entry; void* arg = ts->arg; free(ts);   // host libc: run on host %fs (before activate)
    // Register this worker's stack bounds FIRST, keyed by the Windows thread id (== exec_image_win's
    // cur_tid) so a guest GC that queries its own stack base (GC_register_my_thread / stack scanning —
    // e.g. Unity's AssetGarbageCollectorHelper) finds real bounds instead of wedging on "Bad stack
    // base". Mirrors the Linux thread_trampoline's register-first ordering. GetCurrentThreadStackLimits
    // gives the committed stack range (Win8+).
    ULONG_PTR stk_lo = 0, stk_hi = 0;
    GetCurrentThreadStackLimits(&stk_lo, &stk_hi);
    if (stk_lo && stk_hi > stk_lo)
        register_thread_stack((uint64_t)GetCurrentThreadId(), (void*)stk_lo, (uint64_t)(stk_hi - stk_lo));
    guest_tls_activate_thread();   // this worker's guest %fs TCB (FSGSBASE); no-op if the gate is off
    void* rv = entry ? (void*)(uintptr_t)prosper_call_guest_sysv(entry, (uint64_t)(uintptr_t)arg, 0) : nullptr;
    tls_dtv_purge_current_thread();
    unregister_thread_stack((uint64_t)GetCurrentThreadId());   // ids recycle; stale bounds = wrong bounds
    return rv;
}
}
#endif

HLE(k_pthread_create) {
    auto entry = (void* (*)(void*))(uintptr_t)a2;
    void* arg  = (void*)(uintptr_t)a3;
    if (getenv("PROSPER_SYNCLOG"))
        fprintf(stderr, "[thread] create entry=0x%llx arg=0x%llx name=%s\n",
                (unsigned long long)a2, (unsigned long long)a3, a4 ? (const char*)(uintptr_t)a4 : "");
    pthread_t tid;
#if defined(__linux__) || defined(__APPLE__)
    // Stack size: honor the guest attr's stacksize (previously ignored — everything got a fixed
    // 8 MiB), floored at 1 MiB because our HLE + host libc frames run deeper than Sony's runtime
    // would on the same stack. No attr (or no explicit size) keeps the 8 MiB default.
    constexpr size_t kStackFloor = 1 * 1024 * 1024, kStackDefault = 8 * 1024 * 1024;
    size_t ssz = kStackDefault;
    int detach = PTHREAD_CREATE_JOINABLE;
    if (a1 && *(void**)a1) {
        auto* gat = (pthread_attr_t*)*(void**)a1;
        size_t req = 0;
        if (pthread_attr_getstacksize(gat, &req) == 0 && req)
            ssz = req < kStackFloor ? kStackFloor : req;
        pthread_attr_getdetachstate(gat, &detach);
    }
    // A local attr with setstackSIZE (not setstack): glibc allocates AND RECLAIMS the stack
    // (join / detached exit) — the old caller-owned mmap was never freed (#138). Also stops
    // mutating the guest's own attr object (setstack used to be applied to it in place).
    pthread_attr_t la; pthread_attr_init(&la);
    pthread_attr_setstacksize(&la, ssz);
    pthread_attr_setdetachstate(&la, detach);
    auto* ts = (ThreadStart*)malloc(sizeof(ThreadStart));
    ts->entry = entry; ts->arg = arg; ts->name[0] = 0;
    if (a4) { strncpy(ts->name, (const char*)(uintptr_t)a4, sizeof(ts->name) - 1); ts->name[sizeof(ts->name) - 1] = 0; }
    int r = pthread_create(&tid, &la, thread_trampoline, ts);   // trampoline registers the stack first
    pthread_attr_destroy(&la);
    if (r) { free(ts); return (uint64_t)r; }
#else
    // Windows: route through win_thread_trampoline so the guest entry is called with the SysV ABI and
    // gets its guest %fs TCB — a bare pthread_create(entry, arg) mis-passes the arg (MS x64 vs SysV) and
    // skips TLS activation, crashing the worker on a null-derived pointer.
    pthread_attr_t* at = (a1 && *(void**)a1) ? (pthread_attr_t*)*(void**)a1 : nullptr;
    auto* ts = (WinThreadStart*)malloc(sizeof(WinThreadStart));
    ts->entry = (uint64_t)(uintptr_t)entry; ts->arg = arg;
    int r = pthread_create(&tid, at, win_thread_trampoline, ts);
    if (r) { free(ts); return (uint64_t)r; }
#endif
    if (a0) *(uint64_t*)a0 = (uint64_t)tid;
    return 0;
}
// Plain POSIX pthread_create(thread, attr, start, arg) has only 4 args, so a4 (r8) is caller-indeterminate
// scratch. k_pthread_create reads a4 as a thread-name pointer -- legitimate for scePthreadCreate /
// pthread_create_name_np (5-arg), but for the 4-arg POSIX form a non-zero garbage a4 makes it strncpy from
// a wild/unmapped address -> intermittent SIGSEGV (IL2CPP bdwgc's GC_pthread_create calls the 4-arg form).
// Force name=null for the 4-arg entry point.
HLE(k_pthread_create_noname) { return k_pthread_create(a0, a1, a2, a3, 0, 0); }
HLE(k_pthread_join)   { void* rv = nullptr; pthread_join((pthread_t)a0, a1 ? &rv : nullptr); if (a1) *(void**)(uintptr_t)a1 = rv; return 0; }
HLE(k_pthread_detach) { pthread_detach((pthread_t)a0); return 0; }
HLE(k_pthread_exit)   {
    // Host pthread_exit unwinds without ever returning through thread_trampoline, so this is its own
    // thread-exit path: purge the exiting thread's __tls_get_addr DTV first (#68) and drop its stack
    // registration (#138 — pthread ids recycle). HLE handlers run under the HOST %fs (the import
    // stubs swap), so the host libc below is safe. No-op for threads that never touched either.
    tls_dtv_purge_current_thread();
#if defined(__linux__) || defined(__APPLE__)
    unregister_thread_stack((uint64_t)pthread_self());
#endif
    pthread_exit((void*)(uintptr_t)a0);
    return 0;
}

// --- thread-local storage keys (IL2CPP uses these heavily) -> host pthread keys ---
HLE(k_key_create) {
    if (!a0) return 0x16;
    pthread_key_t k;
    int r = pthread_key_create(&k, (void (*)(void*))(uintptr_t)a1);
    if (r) return (uint64_t)r;
    *(uint32_t*)(uintptr_t)a0 = (uint32_t)k;   // hand the guest our host key
    return 0;
}
HLE(k_key_delete)    { pthread_key_delete((pthread_key_t)a0); return 0; }
HLE(k_getspecific)   {
    uint64_t rv = (uint64_t)(uintptr_t)pthread_getspecific((pthread_key_t)a0);
    // #312: learn the non-trapping MB3 pool-array address even when the hardware-watch diagnostic
    // is off. Only 64 KiB-aligned values enter the bounded lock-free registry. Deliberately do not
    // use host thread_local state here: guest threads own %fs between import-stub swaps.
    if (gpu::mb3_tls_tracking_enabled()) gpu::mb3_note_tls_pool_candidate(rv);
    // #312: the MallocBinned3 per-thread free-block cache base is fetched via getspecific; arm a
    // per-thread head watch on it the moment this (owning) thread receives it (no-op unless armed).
    if (g_mb3_arm_hook && rv) g_mb3_arm_hook(rv);
    return rv;
}
HLE(k_setspecific)   {
    // #312: catch the cache the instant it is FIRST installed (base passed to setspecific), which is
    // right after allocation — before any head store — the earliest "descriptor established" moment.
    if (g_mb3_arm_hook && a1) g_mb3_arm_hook(a1);
    int r = pthread_setspecific((pthread_key_t)a0, (void*)(uintptr_t)a1);
    if (!r && gpu::mb3_tls_tracking_enabled()) gpu::mb3_note_tls_pool_candidate(a1);
    return (uint64_t)(int64_t)r;
}

// --- event flags (SceKernelEventFlag): a bit pattern with wait/set/clear ---
namespace {
    // `deleted` + `waiters` make sceKernelDeleteEventFlag safe against blocked waiters: delete marks
    // the flag, wakes everyone, and defers destroy+free to the last waiter leaving (or frees now if
    // none are parked). Freeing under a live pthread_cond_wait is a UAF — destroying a condvar with
    // waiters is explicitly UB.
    struct EventFlag { pthread_mutex_t m; pthread_cond_t c; uint64_t bits; bool deleted; int waiters; };
    bool evf_match(uint64_t bits, uint64_t pat, uint32_t mode) {
        return (mode & 0x1) ? ((bits & pat) == pat) : ((bits & pat) != 0);  // AND vs OR
    }
    void ef_destroy(EventFlag* e) { pthread_mutex_destroy(&e->m); pthread_cond_destroy(&e->c); free(e); }
}
HLE(k_ef_create) {   // (ef*, name, attr, initPattern, opt)
    auto* e = (EventFlag*)calloc(1, sizeof(EventFlag));
    pthread_mutex_init(&e->m, nullptr); pthread_cond_init(&e->c, nullptr); e->bits = a3;
    if (a0) *(void**)(uintptr_t)a0 = e;
    return 0;
}
HLE(k_ef_delete)  {
    auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0;
    pthread_mutex_lock(&e->m);
    e->deleted = true;
    pthread_cond_broadcast(&e->c);                 // wake every waiter -> they return EACCES
    bool has_waiters = e->waiters > 0;
    pthread_mutex_unlock(&e->m);
    if (!has_waiters) ef_destroy(e);               // none parked -> free now; else the last waiter frees
    return 0;
}
HLE(k_ef_set)     { auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0; if (sclog()) fprintf(stderr, "[sync2] T%ld EF.set       ef=0x%llx bits|=0x%llx\n", sctid(), (unsigned long long)a0, (unsigned long long)a1); pthread_mutex_lock(&e->m); e->bits |= a1; pthread_cond_broadcast(&e->c); pthread_mutex_unlock(&e->m); return 0; }
HLE(k_ef_clear)   { auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0; pthread_mutex_lock(&e->m); e->bits &= a1; pthread_mutex_unlock(&e->m); return 0; }
// Absolute CLOCK_REALTIME deadline `usec` microseconds from now (for pthread_cond_timedwait
// on default-attr condvars, which time against CLOCK_REALTIME).
static timespec abs_deadline_us(uint64_t usec) {
    timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec  += (time_t)(usec / 1000000u);
    ts.tv_nsec += (long)(usec % 1000000u) * 1000L;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    return ts;
}
// scePthreadMutexTimedlock(mutex, SceKernelUseconds usec): acquire with a RELATIVE microsecond timeout,
// returning 0 only when actually locked. Was MISSING -> the generic stub returned 0 (= "lock held")
// without taking the lock, so the guest ran its critical section unguarded (the heap/GC-corruption class
// root-caused for null static locks). ETIMEDOUT -> FreeBSD 60 (fbsd_errno doesn't remap it).
HLE(k_mutex_timedlock) {
    auto* m = ensure_mutex(a0); if (!m) return 0x16;   // EINVAL
    timespec dl = abs_deadline_us(a1);
    int rc = pthread_mutex_timedlock(m, &dl);
    return rc == ETIMEDOUT ? 60u : fbsd_errno(rc);
}
// scePthreadCondTimedwait(cond, mutex, SceKernelUseconds usec): the Sony 3rd arg is a RELATIVE µs scalar,
// NOT an abstime pointer -> it must NOT be aliased to the POSIX k_cond_timedwait (which reads a2 as a
// timespec*, so a small µs integer there derefs a bogus pointer). Was MISSING -> the stub returned 0
// (= "woke"), so a timed cond-wait loop busy-spun (the same #115 class fixed for the POSIX variant).
HLE(k_cond_timedwait_sce) {
    auto* c = ensure_cond(a0); auto* m = ensure_mutex(a1);
    if (!c || !m) return 0x16;                          // EINVAL
    timespec dl = abs_deadline_us(a2);
    int rc = pthread_cond_timedwait(c, m, &dl);
    return rc == ETIMEDOUT ? 60u : (uint64_t)rc;        // FreeBSD ETIMEDOUT
}
// scePthreadRwlockTimedrd/wrlock(rwlock, SceKernelUseconds usec): acquire with a RELATIVE µs timeout.
// Were MISSING -> the generic stub returned 0 (= "lock held") without taking the lock, so the guest ran
// its critical section unguarded (the same silent-unsync / heap-race class as k_mutex_timedlock above).
HLE(k_rwlock_timedrdlock) {
    auto* rw = ensure_rwlock(a0); if (!rw) return 0x16;   // EINVAL
    timespec dl = abs_deadline_us(a1);
    int rc = pthread_rwlock_timedrdlock(rw, &dl);
    return rc == ETIMEDOUT ? 60u : (uint64_t)rc;
}
HLE(k_rwlock_timedwrlock) {
    auto* rw = ensure_rwlock(a0); if (!rw) return 0x16;   // EINVAL
    timespec dl = abs_deadline_us(a1);
    int rc = pthread_rwlock_timedwrlock(rw, &dl);
    return rc == ETIMEDOUT ? 60u : (uint64_t)rc;
}
// scePthreadSem* -- POSIX-style counting semaphores (DISTINCT from sceKernelCreateSema). Were MISSING ->
// the generic stub returned 0: SemInit created nothing, SemWait returned immediately (never blocked),
// SemGetvalue left *value uninitialized -> a producer/consumer or gate built on these got NO
// synchronization (the silent-unsync / UAF class). Back them with host sem_t.
HLE(k_sem_init)      { if (!a0) return 0x16; auto* s = (sem_t*)calloc(1, sizeof(sem_t)); sem_init(s, 0, (unsigned)a2); *(void**)(uintptr_t)a0 = s; return 0; }
HLE(k_sem_wait)      { auto* s = ensure_sem(a0); if (!s) return 0x16; return sem_wait(s) == 0 ? 0 : (uint64_t)(unsigned)errno; }
HLE(k_sem_trywait)   { auto* s = ensure_sem(a0); if (!s) return 0x16; return sem_trywait(s) == 0 ? 0 : (uint64_t)(unsigned)errno; }
HLE(k_sem_timedwait) { auto* s = ensure_sem(a0); if (!s) return 0x16; timespec dl = abs_deadline_us(a1); int rc = sem_timedwait(s, &dl); return rc == 0 ? 0 : (errno == ETIMEDOUT ? 60u : (uint64_t)(unsigned)errno); }
HLE(k_sem_post)      { auto* s = ensure_sem(a0); if (!s) return 0x16; sem_post(s); return 0; }
HLE(k_sem_getvalue)  { auto* s = ensure_sem(a0); if (!s) return 0x16; int v = 0; sem_getvalue(s, &v); if (a1) *(int*)(uintptr_t)a1 = v; return 0; }
HLE(k_sem_destroy)   { if (a0) { void** slot = (void**)(uintptr_t)a0; if (!pt_static_sentinel(*slot)) { sem_destroy((sem_t*)*slot); free(*slot); *slot = nullptr; } } return 0; }
// scePthreadBarrier* -- were MISSING -> the generic stub returned 0, so BarrierWait let every thread sail
// past a rendezvous before its peers arrived: the barrier's downstream invariant (all-arrived) is false ->
// reads of not-yet-produced data (the async-load race class). Back with host pthread_barrier_t; the serial
// thread gets -1 (PTHREAD_BARRIER_SERIAL_THREAD, FreeBSD's value), the rest 0. Barrierattr are no-ops.
HLE(k_barrier_init)    { if (!a0 || a2 == 0) return 0x16; auto* b = (pthread_barrier_t*)calloc(1, sizeof(pthread_barrier_t)); pthread_barrier_init(b, nullptr, (unsigned)a2); *(void**)(uintptr_t)a0 = b; return 0; }
HLE(k_barrier_wait)    { auto* b = ensure_barrier(a0); if (!b) return 0x16; int rc = pthread_barrier_wait(b); return rc == PTHREAD_BARRIER_SERIAL_THREAD ? (uint64_t)(int64_t)-1 : (uint64_t)(unsigned)rc; }
HLE(k_barrier_destroy) { if (a0) { void** slot = (void**)(uintptr_t)a0; if (!pt_static_sentinel(*slot)) { pthread_barrier_destroy((pthread_barrier_t*)*slot); free(*slot); *slot = nullptr; } } return 0; }
HLE(k_ef_wait)    { // (ef, pattern, waitMode, resultPat*, SceKernelUseconds* timeout)
    // The timeout arg (a4) was previously IGNORED — a bounded guest wait blocked forever (the
    // same class of silent hang root-caused for wait_on_address, hle_kernel_mem.cpp). Sony
    // semantics: *timeout is a u32 microsecond budget, updated on return with the time left;
    // expiry returns KERNEL_ERROR_ETIMEDOUT (Kyty EventFlag.cpp / Errno.h 0x8002003C).
    auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0;
    uint64_t ret = 0;
    if (sclog()) fprintf(stderr, "[sync2] T%ld EF.wait.ent  ef=0x%llx pat=0x%llx mode=0x%llx bits=0x%llx\n",
                         sctid(), (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                         (unsigned long long)e->bits);
    pthread_mutex_lock(&e->m);
    e->waiters++;
    if (a4) {
        uint32_t usec = *(uint32_t*)(uintptr_t)a4;
        timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
        timespec dl = abs_deadline_us(usec);
        int rc = 0;
        while (!evf_match(e->bits, a1, (uint32_t)a2) && rc != ETIMEDOUT && !e->deleted)
            rc = pthread_cond_timedwait(&e->c, &e->m, &dl);
        timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        int64_t spent_i = (int64_t)(t1.tv_sec - t0.tv_sec) * 1000000
                        + ((int64_t)t1.tv_nsec - (int64_t)t0.tv_nsec) / 1000;
        uint64_t spent = spent_i < 0 ? 0u : (uint64_t)spent_i;
        *(uint32_t*)(uintptr_t)a4 = spent >= usec ? 0u : (uint32_t)(usec - spent);
        if (!e->deleted && !evf_match(e->bits, a1, (uint32_t)a2)) ret = 0x8002003Cull;  // ETIMEDOUT
    } else {
        while (!evf_match(e->bits, a1, (uint32_t)a2) && !e->deleted) pthread_cond_wait(&e->c, &e->m);
    }
    bool deleted = e->deleted;                     // deleted under us -> EACCES (Kyty EventFlag.cpp)
    if (deleted) ret = 0x8002000Dull;
    uint64_t res = e->bits;
    if (!ret) { if (a2 & 0x10) e->bits = 0; else if (a2 & 0x20) e->bits &= ~a1; }  // CLEAR_ALL / CLEAR_PAT
    bool last = (--e->waiters == 0) && deleted;    // last waiter out of a deleted flag frees it
    pthread_mutex_unlock(&e->m);
    if (last) ef_destroy(e);                        // safe: `res` was copied before the free
    if (a3) *(uint64_t*)(uintptr_t)a3 = res;
    if (sclog()) fprintf(stderr, "[sync2] T%ld EF.wait.exit ef=0x%llx res=0x%llx ret=0x%llx\n",
                         sctid(), (unsigned long long)a0, (unsigned long long)res, (unsigned long long)ret);
    return ret;
}
HLE(k_ef_poll)    { // (ef, pattern, waitMode, resultPat*)
    auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0;
    pthread_mutex_lock(&e->m);
    bool ok = evf_match(e->bits, a1, (uint32_t)a2); uint64_t res = e->bits;
    if (ok) { if (a2 & 0x10) e->bits = 0; else if (a2 & 0x20) e->bits &= ~a1; }
    pthread_mutex_unlock(&e->m);
    if (a3) *(uint64_t*)(uintptr_t)a3 = res;
    return ok ? 0 : 0x80020023;   // SCE_KERNEL_ERROR_EBUSY-ish when not matched
}

// --- semaphores (SceKernelSema): counting sem with wait/signal ---
// deleted/waiters: same defer-free-to-last-waiter scheme as EventFlag above (delete under a blocked
// waiter would otherwise free the mutex/condvar out from under it — UAF).
namespace {
    struct Sema { pthread_mutex_t m; pthread_cond_t c; int64_t count; bool deleted; int waiters; };
    void sema_destroy(Sema* s) { pthread_mutex_destroy(&s->m); pthread_cond_destroy(&s->c); free(s); }
}
HLE(k_sema_create) { // (sema*, name, attr, initCount, maxCount, opt)
    auto* s = (Sema*)calloc(1, sizeof(Sema));
    pthread_mutex_init(&s->m, nullptr); pthread_cond_init(&s->c, nullptr); s->count = (int64_t)(int32_t)a3;
    if (a0) *(void**)(uintptr_t)a0 = s;
    if (sclog()) fprintf(stderr, "[sync2] T%ld SEMA.create  sema=0x%llx name='%s' init=%lld max=%lld\n",
                         sctid(), (unsigned long long)(uintptr_t)s, a1 ? (const char*)(uintptr_t)a1 : "",
                         (long long)(int32_t)a3, (long long)(int32_t)a4);
    return 0;
}
HLE(k_sema_delete) {
    auto* s = (Sema*)(uintptr_t)a0; if (!s) return 0;
    pthread_mutex_lock(&s->m);
    s->deleted = true;
    pthread_cond_broadcast(&s->c);
    bool has_waiters = s->waiters > 0;
    pthread_mutex_unlock(&s->m);
    if (!has_waiters) sema_destroy(s);
    return 0;
}
HLE(k_sema_wait)   { // (sema, need, SceKernelUseconds* timeout) — timeout honored like k_ef_wait
    auto* s = (Sema*)(uintptr_t)a0; if (!s) return 0; int64_t need = a1 ? (int64_t)a1 : 1;
    if (sclog()) fprintf(stderr, "[sync2] T%ld SEMA.wait     sema=0x%llx need=%lld\n", sctid(), (unsigned long long)a0, (long long)need);
    uint64_t ret = 0;
    pthread_mutex_lock(&s->m);
    s->waiters++;
    if (a2) {
        uint32_t usec = *(uint32_t*)(uintptr_t)a2;
        timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
        timespec dl = abs_deadline_us(usec);
        int rc = 0;
        while (s->count < need && rc != ETIMEDOUT && !s->deleted) rc = pthread_cond_timedwait(&s->c, &s->m, &dl);
        timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        int64_t spent_i = (int64_t)(t1.tv_sec - t0.tv_sec) * 1000000
                        + ((int64_t)t1.tv_nsec - (int64_t)t0.tv_nsec) / 1000;
        uint64_t spent = spent_i < 0 ? 0u : (uint64_t)spent_i;
        *(uint32_t*)(uintptr_t)a2 = spent >= usec ? 0u : (uint32_t)(usec - spent);
        if (!s->deleted && s->count < need) ret = 0x8002003Cull;    // KERNEL_ERROR_ETIMEDOUT
    } else {
        while (s->count < need && !s->deleted) pthread_cond_wait(&s->c, &s->m);
    }
    bool deleted = s->deleted;                       // deleted under us -> EACCES
    if (deleted) ret = 0x8002000Dull;
    if (!ret) s->count -= need;
    bool last = (--s->waiters == 0) && deleted;
    pthread_mutex_unlock(&s->m);
    if (last) sema_destroy(s);
    return ret; }
HLE(k_sema_signal) { auto* s = (Sema*)(uintptr_t)a0; if (!s) return 0; int64_t n = a1 ? (int64_t)a1 : 1;
    if (sclog()) fprintf(stderr, "[sync2] T%ld SEMA.signal   sema=0x%llx n=%lld\n", sctid(), (unsigned long long)a0, (long long)n);
    pthread_mutex_lock(&s->m); s->count += n; pthread_cond_broadcast(&s->c); pthread_mutex_unlock(&s->m); return 0; }
HLE(k_sema_poll)   { auto* s = (Sema*)(uintptr_t)a0; if (!s) return 0; int64_t need = a1 ? (int64_t)a1 : 1;
    pthread_mutex_lock(&s->m); bool ok = s->count >= need; if (ok) s->count -= need; pthread_mutex_unlock(&s->m); return ok ? 0 : 0x80020023; }

// ---- C++ exception unwinding: sceKernelGetModuleInfoForUnwind (libkernel RpQJJVKTiFM) ----
// The guest's libunwind calls this per code address to locate that module's .eh_frame_hdr for stack
// unwinding. It was an unimplemented stub returning 0 with the caller's struct left uninitialised, so
// libunwind read a garbage eh_frame_hdr address ("Unsupported .eh_frame_hdr version" → stack smash) and
// ANY thrown C++ exception was fatal. Now we fill the real info from the loaded modules' program headers.
namespace {
    std::vector<UnwindModuleDesc> g_unwind_mods;
    std::mutex g_unwind_mx;   // guards g_unwind_mods: set_unwind_modules (assign, may realloc) can run on a
                              // runtime module load while a worker is unwinding here (#344).
    // Decode the eh_frame pointer out of an .eh_frame_hdr. Layout: [0]=version(1), [1]=eh_frame_ptr_enc,
    // [2]=fde_count_enc, [3]=table_enc, [4..]=eh_frame_ptr (encoded). The common encoding is 0x1B
    // (DW_EH_PE_pcrel|sdata4): a signed 32-bit offset from the field's own address.
    // CONFIDENCE: MED — handles the common 0x1B encoding; other encodings fall back to 0.
    uint64_t decode_eh_frame_ptr(uint64_t hdr_va) {
        if (!hdr_va) return 0;
        const uint8_t* h = (const uint8_t*)(uintptr_t)hdr_va;
        if (h[0] != 1) return 0;                       // unknown version
        if (h[1] == 0x1B) {                            // pcrel | sdata4
            int32_t off = *(const int32_t*)(h + 4);
            return (hdr_va + 4) + (int64_t)off;
        }
        if (h[1] == 0x03) return *(const uint32_t*)(h + 4);           // absptr (udata4)
        if (h[1] == 0x0C) return *(const uint64_t*)(h + 4);           // udata8-ish
        return 0;
    }
}
// ModuleInfoForUnwind (orbis, 0x130 bytes): st_size@0x00 (caller-set), name[256]@0x08,
// eh_frame_hdr_addr@0x108, eh_frame_addr@0x110, eh_frame_size@0x118, seg0_addr@0x120, seg0_size@0x128.
// CONFIDENCE: MED on the struct layout (matches shadPS4's OrbisModuleInfoForUnwind) — verify by whether
// the guest's libunwind stops erroring after this fills real values.
HLE(k_get_module_info_for_unwind) {   // (VAddr addr, int flags, ModuleInfoForUnwind* info)
    uint8_t* info = (uint8_t*)(uintptr_t)a2;
    if (!info) return 0x80020016;                      // SCE_KERNEL_ERROR_EINVAL
    UnwindModuleDesc found{};
    { std::lock_guard<std::mutex> lk(g_unwind_mx);
      const UnwindModuleDesc* hit = nullptr;
      for (auto& d : g_unwind_mods) if (a0 >= d.lo && a0 < d.hi) { hit = &d; break; }
      if (!hit) return 0x80020003;                     // SCE_KERNEL_ERROR_ESRCH: addr not in any module
      found = *hit; }   // copy out under the lock — the vector may be reassigned after we unlock (#344)
    const UnwindModuleDesc* m = &found;
    char* nm = (char*)(info + 0x08);
    const char* src = m->name ? m->name : "";
    size_t n = 0; for (; src[n] && n < 255; n++) nm[n] = src[n]; nm[n] = 0;
    uint64_t eh = decode_eh_frame_ptr(m->ehframe_hdr);
    *(uint64_t*)(info + 0x108) = m->ehframe_hdr;
    *(uint64_t*)(info + 0x110) = eh;
    // eh_frame extends from its start to the end of the text segment (an over-estimate is safe —
    // libunwind reads FDEs on demand via the hdr's binary-search table). 0 if we couldn't decode.
    *(uint64_t*)(info + 0x118) = (eh && eh < m->seg0 + m->seg0_sz) ? (m->seg0 + m->seg0_sz - eh) : 0;
    *(uint64_t*)(info + 0x120) = m->seg0;
    *(uint64_t*)(info + 0x128) = m->seg0_sz;
    if (getenv("PROSPER_UNWINDLOG"))
        fprintf(stderr, "[unwind] addr=0x%llx -> %s eh_frame_hdr=0x%llx eh_frame=0x%llx seg0=0x%llx\n",
                (unsigned long long)a0, nm, (unsigned long long)m->ehframe_hdr,
                (unsigned long long)eh, (unsigned long long)m->seg0);
    return 0;
}
void set_unwind_modules(const UnwindModuleDesc* d, size_t c) {
    std::lock_guard<std::mutex> lk(g_unwind_mx); g_unwind_mods.assign(d, d + c);
}

// ---- Async exception delivery = the IL2CPP GC's stop-the-world thread suspension ----
// The runtime installs a handler (sceKernelInstallExceptionHandler) for exception type 0x1e,
// then to stop the world it calls sceKernelRaiseException(thread, 0x1e) on each thread. On
// real hardware that asynchronously interrupts the target thread and runs its handler ON that
// thread; the handler captures the thread's registers (for GC root scanning) and blocks until
// resumed. POSIX hosts use a targeted signal. Windows suspends the target, redirects its CONTEXT
// through a small aligned thunk, and restores the interrupted CONTEXT after the guest handler
// returns. Both paths synthesize the same FreeBSD amd64 mcontext and run the real guest handler
// on the target thread. A stubbed RaiseException left every thread un-acked -> deadlock.
namespace {
uint64_t g_exc_handlers[128] = {0};   // guest handler fn ptr, indexed by exception type
bool g_exc_log = false;               // set once (outside signal ctx) from PROSPER_SYNCLOG
// PROSPER_EXCLOG: dedicated, cheap GC stop-the-world tracing (raise/deliver/ack) — used to correlate
// GC suspend cycles with the level1 loader heap-corruption crash. Independent of the very verbose
// PROSPER_SYNCLOG. Read once (getenv is not signal-safe).
bool g_exc_log2 = false;
volatile int* g_exc_counter = nullptr; // optional fork-safe raise counter (tests)
#if defined(__linux__) || defined(__APPLE__)
int  g_exc_sig = -1;
#ifdef __APPLE__
// Darwin has no pthread_sigqueue / RT-signal payload, so the exception TYPE travels through a
// small async-signal-safe pending table keyed by the target pthread instead of si_value. A raise
// claims a slot with CAS before pthread_kill; the handler (on the target thread) takes it back.
// GC suspend/resume raises are sequential per target, so 16 slots is generous.
struct ExcPending { std::atomic<uint64_t> tid{0}; std::atomic<int> type{0}; };
ExcPending g_exc_pending[16];
bool exc_pending_put(uint64_t tid, int type) {
    for (auto& s : g_exc_pending) {
        uint64_t z = 0;
        if (s.tid.compare_exchange_strong(z, tid)) { s.type.store(type); return true; }
    }
    return false;
}
int exc_pending_take(uint64_t tid) {
    for (auto& s : g_exc_pending)
        if (s.tid.load() == tid) { int t = s.type.load(); s.tid.store(0); return t; }
    return -1;
}
#endif
void exc_delivery_handler(int, siginfo_t* si, void* uc_) {
#ifdef __APPLE__
    (void)si;
    int type = exc_pending_take((uint64_t)pthread_self());   // pthread_self reads %gs TSD, never %fs
#else
    int type = si->si_value.sival_int;
#endif
    if (type < 0 || type >= 128 || !g_exc_handlers[type]) {
        if (g_exc_log2) {   // a suspend request that silently does NOTHING = an unstopped thread
            // %fs-safe: this handler can interrupt guest code running on the GUEST %fs, where host
            // libc TLS (locale, stack canary) reads garbage — swap to host %fs for the logging.
            uint64_t sfs = guest_fs_to_host_scoped();
            char b[96]; int n = snprintf(b, sizeof b, "[exc2] DROPPED type=%d tid=%ld (no handler)\n",
                                         type, (long)prosper_gettid());
            (void)!write(2, b, n);
            guest_fs_restore_scoped(sfs);
        }
        return;
    }
    if (g_exc_log2) {
        // Log the interrupted context: rip (classified guest module+off / host) and whether the
        // thread was on the guest or host %fs at interrupt time. %fs-safe (scoped host swap).
        uint64_t irip = (uint64_t)PROSPER_GREGS((ucontext_t*)uc_)[REG_RIP];
        uint64_t sfs = guest_fs_to_host_scoped();
        const char* fss = sfs ? "guest" : "host";
        char b[160]; int n;
        if (irip >= 0x400000000ull && irip < 0x420000000ull)
            n = snprintf(b, sizeof b, "[exc2] ENTER tid=%ld type=%d fs=%s rip=eboot+0x%llx\n",
                         (long)prosper_gettid(), type, fss, (unsigned long long)(irip - 0x400000000ull));
        else if (irip >= 0x440000000ull && irip < 0x443000000ull)
            n = snprintf(b, sizeof b, "[exc2] ENTER tid=%ld type=%d fs=%s rip=il2cpp+0x%llx\n",
                         (long)prosper_gettid(), type, fss, (unsigned long long)(irip - 0x440000000ull));
        else if (irip >= 0x500000000ull && irip < 0x501000000ull)
            n = snprintf(b, sizeof b, "[exc2] ENTER tid=%ld type=%d fs=%s rip=libc+0x%llx\n",
                         (long)prosper_gettid(), type, fss, (unsigned long long)(irip - 0x500000000ull));
        else
            n = snprintf(b, sizeof b, "[exc2] ENTER tid=%ld type=%d fs=%s rip=host:0x%llx\n",
                         (long)prosper_gettid(), type, fss, (unsigned long long)irip);
        (void)!write(2, b, n);
        guest_fs_restore_scoped(sfs);
    }
    auto* uc = (ucontext_t*)uc_;
    auto g = PROSPER_GREGS(uc);
    // FreeBSD amd64 mcontext_t: rdi@0x08 rsi@0x10 rdx@0x18 rcx@0x20 r8@0x28 r9@0x30 rax@0x38
    // rbx@0x40 rbp@0x48 r10@0x50 r11@0x58 r12@0x60 r13@0x68 r14@0x70 r15@0x78 rip@0xA0 rsp@0xB8
    // NOT `static __thread`: guest threads run under a GUEST %fs, so a host thread_local resolves into
    // the GUEST TLS block — and this buffer's negative tpoff aliased the guest's thread-local GfxDevice
    // pointer at [fs-0x80], so zeroing it nulled Unity's GfxDevice object graph (the 0xba6e08 wall). A
    // plain stack buffer lives on the per-thread sigaltstack (SA_ONSTACK, above), clear of the guest TLS.
    uint8_t ctx[0x400];
    memset(ctx, 0, sizeof ctx);
    auto WQ = [&](int off, uint64_t v) { *(uint64_t*)(ctx + off) = v; };
    WQ(0x08, g[REG_RDI]); WQ(0x10, g[REG_RSI]); WQ(0x18, g[REG_RDX]); WQ(0x20, g[REG_RCX]);
    WQ(0x28, g[REG_R8]);  WQ(0x30, g[REG_R9]);  WQ(0x38, g[REG_RAX]); WQ(0x40, g[REG_RBX]);
    WQ(0x48, g[REG_RBP]); WQ(0x50, g[REG_R10]); WQ(0x58, g[REG_R11]); WQ(0x60, g[REG_R12]);
    WQ(0x68, g[REG_R13]); WQ(0x70, g[REG_R14]); WQ(0x78, g[REG_R15]);
    WQ(0xA0, g[REG_RIP]); WQ(0xB8, g[REG_RSP]);
    // The Sony exception context carries the thread's stack pointer at offset 0xf8; the GC's
    // suspend handler copies context[0xf8] into the thread's GC-table entry as the sp it scans
    // from (GC_push_all_stacks aborts "sp not set!" if it's 0). Populate it with the real rsp.
    WQ(0xF8, g[REG_RSP]);
    // Run the guest handler on this (the target) thread: handler(type, &mcontext). It captures
    // the registers, acks via SuspendSemaphore, and blocks on ResumeSemaphore until resumed.
    // Log via raw SYS_write, NOT glibc write(): this handler keeps the GUEST %fs (to run the guest GC
    // handler), and glibc's write() is a cancellation point whose prologue reads THREAD_SELF at %fs:0x10
    // (== 0 on our guest TCB) and dereferences self->cancelhandling at +0x308 -> null+0x308 fault. The raw
    // syscall touches no TLS. (This bit us only under PROSPER_SYNCLOG, but it was a latent guest-%fs landmine.)
    if (g_exc_log) { const char m[] = "[exc] handler ENTER on target\n"; (void)syscall(SYS_write, 2, m, sizeof m - 1); }
    ((void (*)(uint64_t, void*))(uintptr_t)g_exc_handlers[type])((uint64_t)type, ctx);
    if (g_exc_log) { const char m[] = "[exc] handler EXIT (resumed)\n";   (void)syscall(SYS_write, 2, m, sizeof m - 1); }
    if (g_exc_log2) {
        uint64_t sfs = guest_fs_to_host_scoped();   // %fs-safe logging (see ENTER)
        char b[96]; int n = snprintf(b, sizeof b, "[exc2] RESUME tid=%ld type=%d\n",
                                     (long)prosper_gettid(), type);
        (void)!write(2, b, n);
        guest_fs_restore_scoped(sfs);
    }
}
void ensure_exc_sig() {
    if (g_exc_sig != -1) return;
    g_exc_log = getenv("PROSPER_SYNCLOG") != nullptr;
    g_exc_log2 = getenv("PROSPER_EXCLOG") != nullptr;
#ifdef __APPLE__
    g_exc_sig = SIGUSR2;                         // no RT signals on Darwin; SIGUSR2 is otherwise unused here
#else
    g_exc_sig = SIGRTMIN + 4;                    // free RT signal (not our SIGSEGV/ILL/BUS)
#endif
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = exc_delivery_handler;
    // SA_ONSTACK: run on the per-thread sigaltstack (installed for the main thread and every worker in
    // thread_trampoline), NOT the interrupted guest stack. Critical: the guest thread's TLS (%fs base)
    // can sit only a few hundred bytes below its live stack pointer, so running this handler — and the
    // guest exception handler it invokes — on the guest stack overruns and ZEROES the guest TLS,
    // including the thread-local GfxDevice pointer at [fs-0x80]. That is what made Unity's GfxDevice
    // object graph come up systematically null (the eboot+0xba6e08 / 0x95c823 bring-up wall). Unlike the
    // fault handler, this handler never siglongjmps, so SA_ONSTACK is safe here.
    sa.sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaction(g_exc_sig, &sa, nullptr);
}
#elif defined(_WIN32)
struct WinExcDelivery {
    CONTEXT saved{};
    uint64_t type = 0;
    uint64_t handler = 0;
};
using RtlRestoreContextFn = VOID (WINAPI*)(PCONTEXT, PEXCEPTION_RECORD);
RtlRestoreContextFn g_rtl_restore_context = nullptr;

extern "C" uint64_t prosper_call_guest_sysv(uint64_t fn, uint64_t a0, uint64_t a1);
extern "C" __attribute__((noreturn)) void prosper_win_exc_delivery(WinExcDelivery* delivery);
extern "C" void prosper_win_exc_thunk();

// SetThreadContext reliably redirects RIP/RSP, but Windows may rewrite volatile integer registers
// while resuming a suspended syscall. Pass the delivery pointer on the target stack, not in RCX.
// Entry RSP is 8 mod 16; pop makes it call-site aligned, and the 32-byte subtraction supplies the
// Microsoft-x64 home area before the compiled helper call. The injected RSP is placed below the
// guest SysV red zone so an asynchronous delivery cannot overwrite a leaf function's live locals.
__asm__(
    ".text\n"
    ".p2align 4\n"
    ".globl prosper_win_exc_thunk\n"
    "prosper_win_exc_thunk:\n"
    "    popq %rcx\n"
    "    subq $32, %rsp\n"
    "    call prosper_win_exc_delivery\n"
    "    ud2\n"
);

void win_exc_log(const char* msg) {
    if (!g_exc_log && !g_exc_log2) return;
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), msg, (DWORD)strlen(msg), &written, nullptr);
}

extern "C" __attribute__((noinline, noreturn))
void prosper_win_exc_delivery(WinExcDelivery* delivery) {
    CONTEXT saved = delivery->saved;
    uint64_t type = delivery->type;
    uint64_t handler = delivery->handler;

    // FreeBSD amd64 mcontext layout; keep in lockstep with the POSIX signal path above.
    alignas(16) uint8_t ctx[0x400]{};
    auto WQ = [&](int off, uint64_t v) { *(uint64_t*)(ctx + off) = v; };
    WQ(0x08, saved.Rdi); WQ(0x10, saved.Rsi); WQ(0x18, saved.Rdx); WQ(0x20, saved.Rcx);
    WQ(0x28, saved.R8);  WQ(0x30, saved.R9);  WQ(0x38, saved.Rax); WQ(0x40, saved.Rbx);
    WQ(0x48, saved.Rbp); WQ(0x50, saved.R10); WQ(0x58, saved.R11); WQ(0x60, saved.R12);
    WQ(0x68, saved.R13); WQ(0x70, saved.R14); WQ(0x78, saved.R15);
    WQ(0xA0, saved.Rip); WQ(0xB8, saved.Rsp); WQ(0xF8, saved.Rsp);

    win_exc_log("[exc] handler ENTER on Windows target\n");
    prosper_call_guest_sysv(handler, type, (uint64_t)(uintptr_t)ctx);
    win_exc_log("[exc] handler EXIT on Windows target (resumed)\n");

    delete delivery;
    g_rtl_restore_context(&saved, nullptr);
    TerminateProcess(GetCurrentProcess(), ERROR_INVALID_STATE);   // RtlRestoreContext never returns
    __builtin_unreachable();
}

uint64_t win_raise_exception(uint64_t target, uint64_t type) {
    if (!target || type >= 128 || !g_exc_handlers[type]) return 0;
    if (pthread_equal((pthread_t)target, pthread_self())) return 0x80020023ull; // EDEADLK-ish
    if (!g_rtl_restore_context) return 0x80020026ull;                         // ENOSYS-ish

    HANDLE thread = (HANDLE)pthread_gethandle((pthread_t)target);
    if (!thread || thread == INVALID_HANDLE_VALUE) return 0x80020003ull;       // ESRCH

    // Allocate before suspending: the target could currently own the process heap lock.
    auto* delivery = new (std::nothrow) WinExcDelivery;
    if (!delivery) return 0x8002000cull;                                      // ENOMEM
    delivery->type = type;
    delivery->handler = g_exc_handlers[type];

    DWORD previous_suspend = SuspendThread(thread);
    if (previous_suspend == (DWORD)-1) { DWORD e = GetLastError(); delete delivery; return 0x80020000ull | (e & 0xffff); }
    if (previous_suspend != 0) {
        ResumeThread(thread);
        delete delivery;
        return 0x80020010ull;                                                 // EBUSY
    }

    delivery->saved.ContextFlags = CONTEXT_ALL;
    if (!GetThreadContext(thread, &delivery->saved)) {
        DWORD e = GetLastError(); ResumeThread(thread); delete delivery;
        return 0x80020000ull | (e & 0xffff);
    }

    CONTEXT injected = delivery->saved;
    injected.ContextFlags = CONTEXT_FULL;
    injected.Rip = (DWORD64)(uintptr_t)&prosper_win_exc_thunk;
    constexpr DWORD64 kGuestRedZone = 128;
    injected.Rsp = (injected.Rsp & ~(DWORD64)0xf) - kGuestRedZone - 8;
    DWORD64 delivery_ptr = (DWORD64)(uintptr_t)delivery;
    SIZE_T written = 0;
    if (!WriteProcessMemory(GetCurrentProcess(), (void*)(uintptr_t)injected.Rsp,
                            &delivery_ptr, sizeof(delivery_ptr), &written) ||
        written != sizeof(delivery_ptr)) {
        DWORD e = GetLastError(); ResumeThread(thread); delete delivery;
        return 0x80020000ull | (e & 0xffff);
    }
    if (!SetThreadContext(thread, &injected)) {
        DWORD e = GetLastError(); ResumeThread(thread); delete delivery;
        return 0x80020000ull | (e & 0xffff);
    }
    if (ResumeThread(thread) == (DWORD)-1) {
        DWORD e = GetLastError();
        bool restored = SetThreadContext(thread, &delivery->saved) != FALSE;
        if (restored && ResumeThread(thread) != (DWORD)-1) delete delivery;
        return 0x80020000ull | (e & 0xffff);
    }
    // A redirected Windows CONTEXT is not dispatched until a blocking syscall returns. Current
    // IL2CPP targets commonly sit in our infinite WaitOnAddress HLE; wake its registered address
    // after dropping the suspend count so the target enters prosper_win_exc_thunk immediately.
    interrupt_futex_wait(target);
    return 0;
}

void ensure_exc_sig() {
    if (g_rtl_restore_context) return;
    g_exc_log = getenv("PROSPER_SYNCLOG") != nullptr;
    g_exc_log2 = getenv("PROSPER_EXCLOG") != nullptr;
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    g_rtl_restore_context = ntdll
        ? (RtlRestoreContextFn)GetProcAddress(ntdll, "RtlRestoreContext") : nullptr;
}
#else
void ensure_exc_sig() {}
#endif
} // namespace

HLE(k_install_exc_handler) {   // (exceptionType, handler, ...)
    ensure_exc_sig();
    if (a0 < 128) g_exc_handlers[a0] = a1;
    if (getenv("PROSPER_SYNCLOG"))
        fprintf(stderr, "[exc] install type=0x%llx handler=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1);
    return 0;
}
HLE(k_raise_exception) {       // (targetThread /*host pthread_t*/, exceptionType, arg)
    ensure_exc_sig();
    if (g_exc_counter) (*g_exc_counter)++;
    if (getenv("PROSPER_SYNCLOG"))
        fprintf(stderr, "[exc] T%ld raise target=0x%llx type=0x%llx\n",
                sctid(), (unsigned long long)a0, (unsigned long long)a1);
#if defined(__linux__) || defined(__APPLE__)
    if (a0 && a1 < 128 && g_exc_handlers[a1]) {
#ifdef __APPLE__
        int qr = exc_pending_put((uint64_t)a0, (int)a1) ? pthread_kill((pthread_t)a0, g_exc_sig)
                                                        : EAGAIN;   // pending table full
#else
        union sigval sv; sv.sival_int = (int)a1;
        int qr = pthread_sigqueue((pthread_t)a0, g_exc_sig, sv);
#endif
        if (g_exc_log2)
            fprintf(stderr, "[exc2] RAISE by tid=%ld target=0x%llx type=0x%llx sigqueue=%d\n",
                    sctid(), (unsigned long long)a0, (unsigned long long)a1, qr);
        if (qr != 0) return 0x80020000u | (uint32_t)qr;   // deliverance FAILED — report, don't lie
    } else if (g_exc_log2) {
        fprintf(stderr, "[exc2] RAISE-NOOP by tid=%ld target=0x%llx type=0x%llx (no handler)\n",
                sctid(), (unsigned long long)a0, (unsigned long long)a1);
    }
#elif defined(_WIN32)
    return win_raise_exception(a0, a1);
#endif
    return 0;
}

void set_exc_raise_counter(volatile int* counter) { g_exc_counter = counter; }

HLE(k_is_stack) {   // sceKernelIsStack(void* addr): is addr within the current thread's stack?
    void* base = nullptr; size_t sz = 0;
    if (guest_stack_for_current_thread(&base, &sz) &&
        a0 >= (uint64_t)(uintptr_t)base && a0 < (uint64_t)(uintptr_t)base + sz)
        return 1;
    return 0;
}
// Global export table (NID -> guest addr) registered by the loader, so dlsym can resolve
// exported symbols by name against loaded modules (e.g. PSN.prx's plugin/init exports).
namespace {
    const std::unordered_map<std::string, uint64_t>* g_exports = nullptr;
    std::vector<ModuleExportTable> g_mod_exports;   // per-module tables (#147)
    // Real module handles occupy 0x10000+index — far above the synthetic success counter
    // (k_load_start_mod's g_module_handle, which starts at 1), so the ranges can't collide.
    constexpr uint64_t kModuleHandleBase = 0x10000;
    const char* path_basename(const char* p) {
        const char* b = p;
        for (const char* c = p; *c; c++) if (*c == '/' || *c == '\\') b = c + 1;
        return b;
    }
    // Case-INSENSITIVE basename compare: the guest's runtime sceKernelLoadStartModule path can differ
    // in case from the linker's preload path (observed: the Messenger loads "Il2CppUserAssemblies.prx"
    // but the module is preloaded as "Il2cppUserAssemblies.prx"), and PS5 module paths are effectively
    // case-insensitive. Without this the game's own PRX miss the match and — with #146's ENOENT — the
    // load fails, so IL2CPP never bootstraps.
    inline bool ieq(const char* a, const char* b) {
        for (; *a && *b; a++, b++) if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return false;
        return *a == *b;
    }
}
void set_module_exports(const std::unordered_map<std::string, uint64_t>* exports) { g_exports = exports; }
void set_module_export_tables(std::vector<ModuleExportTable> tables) { g_mod_exports = std::move(tables); }
uint64_t module_handle_for_path(const char* path) {
    if (!path || !*path) return 0;
    const char* want = path_basename(path);
    for (size_t i = 0; i < g_mod_exports.size(); i++)
        if (ieq(path_basename(g_mod_exports[i].path.c_str()), want))
            return kModuleHandleBase + i;
    return 0;
}

HLE(k_dlsym) {   // sceKernelDlsym(SceKernelModule handle, const char* name, void** funcAddr)
    // Resolve the requested symbol by name against the HANDLE'S module first (#147) — a real
    // handle from k_load_start_mod names one linked module, and two modules exporting the same
    // NID must not alias to the global table's first definition — then fall back to the global
    // export table (Unity's native-plugin loader dlsym's UnityPluginLoad / PSN_PrxInitialize by
    // name; synthetic handles for unknown paths land here). We hash the name to its Sony NID
    // (nid_hash); a hit is written through *funcAddr and reported as success.
    const char* name = a1 ? (const char*)(uintptr_t)a1 : nullptr;
    if (name && a0 >= kModuleHandleBase && a0 - kModuleHandleBase < g_mod_exports.size()) {
        if (const auto* nids = g_mod_exports[a0 - kModuleHandleBase].nids) {
            auto it = nids->find(nid_hash(name));
            if (it != nids->end()) {
                if (a2) *(uint64_t*)(uintptr_t)a2 = it->second;
                if (getenv("PROSPER_SYNCLOG"))
                    fprintf(stderr, "[dlsym] '%s' (module handle 0x%llx) -> 0x%llx\n",
                            name, (unsigned long long)a0, (unsigned long long)it->second);
                return 0;
            }
        }
    }
    if (name && g_exports) {
        auto it = g_exports->find(nid_hash(name));
        if (it != g_exports->end()) {
            if (a2) *(uint64_t*)(uintptr_t)a2 = it->second;   // *funcAddr = export
            if (getenv("PROSPER_SYNCLOG"))
                fprintf(stderr, "[dlsym] '%s' -> 0x%llx\n", name, (unsigned long long)it->second);
            return 0;
        }
    }
    // Not an exported symbol of any loaded module. Return ESRCH ("not found") but leave *funcAddr
    // untouched — callers pre-seed it with a fallback and keep that on failure. (The old path
    // returned success; nulling the out pointer here broke a caller that then invoked it.)
    if (getenv("PROSPER_SYNCLOG"))
        fprintf(stderr, "[dlsym] unresolved '%s' -> ESRCH (fallback kept)\n", name ? name : "?");
    return 0x80020003;   // SCE_KERNEL_ERROR_ESRCH
}

// --- General-dynamic TLS (__tls_get_addr) for loaded modules (e.g. the real libc.prx). ----------
// PS5 .prx shared libs access thread-locals via __tls_get_addr(tls_index*), where the tls_index
// {module_id, offset} was patched by our DTPMOD64/DTPOFF64 relocs. We keep a per-thread block per
// module id, lazily allocated from the module's PT_TLS template (memsz block, filesz copied from
// the init image, tbss zeroed). This is the general-dynamic model — no %fs needed (that's only for
// the main exe's initial-exec TLS, which the current boot already tolerates).
namespace {
std::vector<TlsModuleDesc> g_tls_mods;
std::mutex g_tls_mods_mx;   // guards g_tls_mods: a runtime sceKernelLoadStartModule re-runs set_tls_modules
                            // (assign, may realloc) while a worker is in k_tls_get_addr (#344).
// Per-thread DTV (thread -> module id -> TLS block). MUST NOT use a host `thread_local`: guest threads
// run under the GUEST %fs, and host thread_local storage is %fs-relative, so it ALIASES guest memory —
// reads come back as garbage (an unordered_map whose bucket_count reads 0 → `hash % 0` → SIGFPE).
// This is the same host↔guest %fs-aliasing landmine as the GfxDevice boot wall. Key by
// std::thread::id in a mutex-guarded global map instead. CONFIDENCE: HIGH (root-caused via gdb:
// SIGFPE in k_tls_get_addr with a corrupt thread_local map under a guest %fs).
// The map is PURGED on every HLE-controlled thread-exit path (thread_trampoline return +
// scePthreadExit/pthread_exit) via tls_dtv_purge_current_thread(): glibc recycles pthread ids, so a
// stale entry would hand a NEW thread the dead thread's dirty TLS blocks instead of the fresh
// zero/tdata-initialized state the ABI guarantees — and the blocks would leak per thread churn (#68).
std::mutex g_dtv_mx;
std::unordered_map<std::thread::id, std::unordered_map<uint64_t, void*>> g_dtv;
}
void tls_dtv_purge_current_thread() {
    std::unordered_map<uint64_t, void*> mine;
    { std::lock_guard<std::mutex> lk(g_dtv_mx);
      auto it = g_dtv.find(std::this_thread::get_id());
      if (it == g_dtv.end()) return;   // main/host threads may never have touched __tls_get_addr
      mine = std::move(it->second);
      g_dtv.erase(it);
    }
    for (auto& kv : mine) free(kv.second);   // free the blocks outside the lock
}
size_t tls_dtv_thread_count() {   // test/diagnostic introspection: threads with live DTV entries
    std::lock_guard<std::mutex> lk(g_dtv_mx);
    return g_dtv.size();
}
void set_tls_modules(const TlsModuleDesc* descs, size_t count) {
    { std::lock_guard<std::mutex> lk(g_tls_mods_mx); g_tls_mods.assign(descs, descs + count); }
    if (getenv("PROSPER_TLSLOG"))
        for (size_t i = 0; i < count; i++)
            fprintf(stderr, "[tls] module %zu: init_va=0x%llx filesz=0x%llx memsz=0x%llx\n", i,
                    (unsigned long long)descs[i].init_va, (unsigned long long)descs[i].filesz,
                    (unsigned long long)descs[i].memsz);
}
HLE(k_tls_get_addr) {   // __tls_get_addr(tls_index* ti): ti[0]=module id, ti[1]=offset-in-block
    const uint64_t* ti = (const uint64_t*)(uintptr_t)a0;
    if (!ti) return 0;
    uint64_t modid = ti[0], off = ti[1];
    std::thread::id tid = std::this_thread::get_id();   // portable per-OS-thread key (no %fs, no syscall)
    { std::lock_guard<std::mutex> lk(g_dtv_mx);
      auto& dtv = g_dtv[tid];
      auto it = dtv.find(modid);
      if (it != dtv.end()) return (uint64_t)(uintptr_t)it->second + off;
    }
    size_t memsz = 64, filesz = 0; uint64_t init_va = 0;
    { std::lock_guard<std::mutex> lk(g_tls_mods_mx);   // #344: safe vs a concurrent set_tls_modules realloc
      if (modid < g_tls_mods.size()) {
          memsz  = g_tls_mods[modid].memsz ? g_tls_mods[modid].memsz : 64;
          filesz = g_tls_mods[modid].filesz;
          init_va = g_tls_mods[modid].init_va;
      } }
    void* blk = calloc(1, memsz);   // zero-init (tbss), then copy the tdata image
    if (init_va && filesz) memcpy(blk, (const void*)(uintptr_t)init_va, filesz);
    { std::lock_guard<std::mutex> lk(g_dtv_mx); g_dtv[tid][modid] = blk; }
    return (uint64_t)(uintptr_t)blk + off;
}

// sceKernelGetProcParam: guest address of the main module's SCE_PROCPARAM. Real libc reads its
// heap/malloc config (sceLibcParam) from here; without it, libc's heap never inits and malloc/
// memalign return null (the branch's eboot+0x8065ee crash).
namespace { uint64_t g_proc_param = 0; }
void set_proc_param(uint64_t guest_va) { g_proc_param = guest_va; }
HLE(k_get_proc_param) { return g_proc_param; }

void register_kernel_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    // ELF TLS accessor used by loaded .prx modules (raw NID + name).
    Hle::register_fn("vNe1w4diLCs", (HleFn)k_tls_get_addr, "__tls_get_addr");
    R("__tls_get_addr", k_tls_get_addr);
    R("sceKernelGetProcParam", k_get_proc_param);
    R("sceKernelInstallExceptionHandler", k_install_exc_handler);
    R("sceKernelRaiseException", k_raise_exception);
    R("sceKernelDlsym", k_dlsym);
    R("sceKernelIsStack", k_is_stack);
    R("scePthreadMutexattrInit", k_mutexattr_init);
    R("scePthreadMutexattrSettype", k_mutexattr_settype);
    R("scePthreadMutexattrGettype", k_mutexattr_gettype);   // was MISSING -> left *type uninitialized
    R("scePthreadMutexattrSetprotocol", k_mutexattr_setprotocol);
    R("scePthreadMutexattrSetpshared", k_mutexattr_setpshared);
    R("scePthreadMutexattrDestroy", k_mutexattr_destroy);
    R("scePthreadMutexInit", k_mutex_init);
    R("scePthreadMutexDestroy", k_mutex_destroy);
    R("scePthreadMutexLock", k_mutex_lock);
    R("scePthreadMutexTrylock", k_mutex_trylock);
    R("scePthreadMutexTimedlock", k_mutex_timedlock);   // was MISSING -> faked "locked" without locking
    R("scePthreadMutexUnlock", k_mutex_unlock);
    R("scePthreadCondattrInit", k_condattr_init);
    R("scePthreadCondattrDestroy", k_condattr_destroy);
    R("scePthreadCondInit", k_cond_init);
    R("scePthreadCondDestroy", k_cond_destroy);
    R("scePthreadCondSignal", k_cond_signal);
    R("scePthreadCondBroadcast", k_cond_broadcast);
    R("scePthreadCondWait", k_cond_wait);
    R("scePthreadCondTimedwait", k_cond_timedwait_sce);   // Sony: relative µs, NOT the POSIX abstime form
    // read/write locks + once (Sony + POSIX names) — real host primitives (thread-safety fix).
    R("scePthreadRwlockInit", k_rwlock_init);        R("pthread_rwlock_init", k_rwlock_init);
    R("scePthreadRwlockDestroy", k_rwlock_destroy);  R("pthread_rwlock_destroy", k_rwlock_destroy);
    R("scePthreadRwlockRdlock", k_rwlock_rdlock);    R("pthread_rwlock_rdlock", k_rwlock_rdlock);
    R("scePthreadRwlockWrlock", k_rwlock_wrlock);    R("pthread_rwlock_wrlock", k_rwlock_wrlock);
    R("scePthreadRwlockUnlock", k_rwlock_unlock);    R("pthread_rwlock_unlock", k_rwlock_unlock);
    R("scePthreadRwlockTryrdlock", k_rwlock_tryrdlock);
    R("scePthreadRwlockTrywrlock", k_rwlock_trywrlock);
    R("scePthreadRwlockTimedrdlock", k_rwlock_timedrdlock);   // were MISSING -> faked "locked" without locking
    R("scePthreadRwlockTimedwrlock", k_rwlock_timedwrlock);
    // scePthreadSem* counting semaphores (were MISSING -> SemWait never blocked)
    R("scePthreadSemInit", k_sem_init);         R("scePthreadSemDestroy", k_sem_destroy);
    R("scePthreadSemWait", k_sem_wait);         R("scePthreadSemTrywait", k_sem_trywait);
    R("scePthreadSemTimedwait", k_sem_timedwait); R("scePthreadSemPost", k_sem_post);
    R("scePthreadSemGetvalue", k_sem_getvalue);
    // scePthreadBarrier* (were MISSING -> BarrierWait let threads pass a false rendezvous)
    R("scePthreadBarrierInit", k_barrier_init);   R("scePthreadBarrierWait", k_barrier_wait);
    R("scePthreadBarrierDestroy", k_barrier_destroy);
    R("scePthreadBarrierattrInit", k_attr_noop);  R("scePthreadBarrierattrDestroy", k_attr_noop);
    R("scePthreadBarrierattrSetpshared", k_attr_noop);
    R("scePthreadOnce", k_pthread_once);             R("pthread_once", k_pthread_once);
    R("scePthreadSelf", k_pthread_self);
    R("scePthreadEqual", k_pthread_equal);
    R("scePthreadYield", k_pthread_yield);
    R("__stack_chk_fail", k_stack_chk_fail);   // diagnostic: log the guest canary on a canary-check failure
    R("scePthreadCreate", k_pthread_create);
    R("scePthreadJoin", k_pthread_join);
    R("scePthreadDetach", k_pthread_detach);
    R("scePthreadExit", k_pthread_exit);
    R("scePthreadAttrInit", k_attr_init);
    R("scePthreadAttrDestroy", k_attr_destroy);
    R("scePthreadAttrSetstacksize", k_attr_setstacksize);
    R("scePthreadAttrSetinheritsched", k_attr_noop);
    R("scePthreadAttrSetschedpolicy", k_attr_noop);
    R("scePthreadAttrSetschedparam", k_attr_noop);
    R("scePthreadAttrSetdetachstate", k_attr_setdetachstate);   // was no-op -> detached threads leaked
    R("scePthreadAttrGetdetachstate", k_attr_getdetachstate);   // was MISSING -> uninitialized *state
    R("scePthreadAttrGetschedparam", k_attr_getschedparam);
    R("scePthreadAttrGet", k_attr_get);
    R("scePthreadAttrGetstackaddr", k_attr_getstackaddr);
    R("scePthreadAttrGetstacksize", k_attr_getstacksize);
    R("scePthreadAttrGetaffinity", k_attr_getaffinity);  // report 8 cores (not an empty mask)
    R("scePthreadAttrSetaffinity", k_attr_noop);         // accept affinity requests (we don't pin)
    R("scePthreadGetaffinity", k_attr_getaffinity);      R("scePthreadSetaffinity", k_attr_noop);
    R("scePthreadGetschedparam", k_getschedparam);  R("pthread_getschedparam", k_getschedparam);
    R("scePthreadSetschedparam", k_attr_noop);  R("scePthreadSetprio", k_attr_noop);
    R("scePthreadGetprio", k_getprio);
    R("scePthreadGetstack", k_attr_getstackaddr);
    // TLS keys (POSIX + Sony names -> host pthread keys)
    R("pthread_key_create", k_key_create);   R("scePthreadKeyCreate", k_key_create);
    R("pthread_key_delete", k_key_delete);   R("scePthreadKeyDelete", k_key_delete);
    R("pthread_getspecific", k_getspecific); R("scePthreadGetspecific", k_getspecific);
    R("pthread_setspecific", k_setspecific); R("scePthreadSetspecific", k_setspecific);
    R("pthread_self", k_pthread_self);
    // POSIX pthread_equal — the GC compares thread ids with this while searching its
    // thread table; unimplemented (always "not equal") made every thread look unknown.
    R("pthread_equal", k_pthread_equal);
    // POSIX pthread_* names. The guest's libc is FreeBSD-derived (pointer/opaque pthread
    // types), so these have the same semantics as our Sony handlers — alias them.
    R("pthread_create", k_pthread_create_noname);   R("pthread_join", k_pthread_join);   // 4-arg: no name arg (a4=garbage r8)
    // libScePosix variant with a trailing name arg — same (tid*, attr, entry, arg, name) shape as
    // scePthreadCreate. Unimplemented-0 silently created NO thread (UE4's IO stack starves).
    R("pthread_create_name_np", k_pthread_create);
    R("pthread_detach", k_pthread_detach);    R("pthread_exit", k_pthread_exit);
    R("pthread_yield", k_pthread_yield);      R("sched_yield", k_pthread_yield);
    R("pthread_mutex_init", k_mutex_init);    R("pthread_mutex_destroy", k_mutex_destroy);
    R("pthread_mutex_lock", k_mutex_lock);    R("pthread_mutex_trylock", k_mutex_trylock);
    R("pthread_mutex_unlock", k_mutex_unlock);
    R("pthread_mutexattr_init", k_mutexattr_init); R("pthread_mutexattr_settype", k_mutexattr_settype);
    R("pthread_mutexattr_destroy", k_mutexattr_destroy); R("pthread_mutexattr_setprotocol", k_mutexattr_setprotocol);
    R("pthread_cond_init", k_cond_init);      R("pthread_cond_destroy", k_cond_destroy);
    R("pthread_cond_signal", k_cond_signal);  R("pthread_cond_broadcast", k_cond_broadcast);
    R("pthread_cond_wait", k_cond_wait);
    R("pthread_cond_timedwait", k_cond_timedwait);   // issue #115: unimpl-0 spun WaitCompletion loops
    R("pthread_condattr_init", k_condattr_init); R("pthread_condattr_destroy", k_condattr_destroy);
    R("pthread_attr_init", k_attr_init);      R("pthread_attr_destroy", k_attr_destroy);
    R("pthread_attr_setstacksize", k_attr_setstacksize);
    R("pthread_attr_setdetachstate", k_attr_setdetachstate); R("pthread_attr_getdetachstate", k_attr_getdetachstate);
    R("pthread_attr_setinheritsched", k_attr_noop);
    R("pthread_attr_setschedpolicy", k_attr_noop);  R("pthread_attr_setschedparam", k_attr_noop);
    R("pthread_attr_getstacksize", k_attr_getstacksize);
    R("scePthreadAttrSetaffinity", k_attr_noop); R("pthread_setname_np", k_attr_noop);
    // event flags + semaphores (engine thread synchronization)
    R("sceKernelCreateEventFlag", k_ef_create); R("sceKernelDeleteEventFlag", k_ef_delete);
    R("sceKernelSetEventFlag", k_ef_set);       R("sceKernelClearEventFlag", k_ef_clear);
    R("sceKernelWaitEventFlag", k_ef_wait);     R("sceKernelPollEventFlag", k_ef_poll);
    R("sceKernelCreateSema", k_sema_create);    R("sceKernelDeleteSema", k_sema_delete);
    R("sceKernelWaitSema", k_sema_wait);        R("sceKernelSignalSema", k_sema_signal);
    R("sceKernelPollSema", k_sema_poll);
    R("sceKernelGetModuleInfoForUnwind", k_get_module_info_for_unwind);   // C++ exception unwinding
    // sceSysmoduleGetModuleInfoForUnwind (libSceSysmodule, NID 4fU5yvOkVG4): same contract — shadPS4's
    // implementation just delegates to sceKernelGetModuleInfoForUnwind (sysmodule.cpp:31). It was an
    // unimplemented stub returning 0 (= SUCCESS) with the caller's 0x130-byte info struct left
    // uninitialized, so any exception unwound through a sysmodule-resolved frame read a garbage
    // eh_frame_hdr — the same stack-smash failure mode the kernel variant had before it was implemented.
    Hle::register_fn("4fU5yvOkVG4", (HleFn)k_get_module_info_for_unwind, "sceSysmoduleGetModuleInfoForUnwind");
    // Registration / hook / debug libkernel calls the app makes at startup that have no observable
    // effect in our headless boot — returning OK without side effects is the correct behavior:
    //  - SetThreadDtors / SetThreadAtexitCount / SetThreadAtexitReport: per-thread exit bookkeeping;
    //    our guest threads never cleanly exit (process is torn down), so there's nothing to run.
    //  - RtldSetApplicationHeapAPI: lets the app give the dynamic linker a custom malloc; unset -> the
    //    linker uses its default heap, which is what we already run on.
    //  - GetSanitizerNewReplaceExternal: reports whether a sanitizer replaced operator new; 0 = none.
    //  - SetVirtualRangeName: debug label for a VA range; no runtime effect.
    // Registered by raw NID (guaranteed match; names not in our NidDb).
    Hle::register_fn("rNhWz+lvOMU", (HleFn)k_attr_noop, "sceKernelSetThreadDtors");
    Hle::register_fn("pB-yGZ2nQ9o", (HleFn)k_attr_noop, "sceKernelSetThreadAtexitCount");
    Hle::register_fn("WhCc1w3EhSI", (HleFn)k_attr_noop, "sceKernelSetThreadAtexitReport");
    Hle::register_fn("p5EcQeEeJAE", (HleFn)k_attr_noop, "sceKernelRtldSetApplicationHeapAPI");
    Hle::register_fn("bnZxYgAFeA0", (HleFn)k_attr_noop, "sceKernelGetSanitizerNewReplaceExternal");
    Hle::register_fn("py6L8jiVAN8", (HleFn)k_get_sanitizer_malloc_replace,
                     "sceKernelGetSanitizerMallocReplaceExternal");
    Hle::register_fn("DGMG3JshrZU", (HleFn)k_attr_noop, "sceKernelSetVirtualRangeName");
    #undef R
    register_kernel_mem_hle();    // virtual/direct memory
    register_kernel_time_hle();   // time/clock + C11 threads + stubs
}

} // namespace prosper
