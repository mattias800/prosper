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
#include "sce_errno.hpp"
#include "../host/boot_program.hpp"   // #1659: shared guest-module labelling
#include "../host/exec_image.hpp"      // describe_code_address (host frame naming)
#include "sync_futex.hpp"
#include "sync_retire.hpp"   // #2042: a destroyed guest sync object's storage is retired, not freed
#include "../gpu/mb3_freelist.hpp"
#include "../host/exec_image.hpp"
#include <pthread.h>
#include <semaphore.h>   // scePthreadSem* -> host sem_t
#include "../host/posix_shim.hpp"   // Darwin: sem/barrier/timedlock/getattr_np/sigqueue compat
// raw_fmt_len: snprintf's would-be length clamped to what actually landed (#2050), used by the
// [exc2] handler below.
//
// Guarded, and at FILE SCOPE, for two separate reasons that must both hold:
//   * raw_syscall.hpp's non-Linux branch includes <sys/mman.h>/<unistd.h> unconditionally, so it is
//     not MinGW-clean -- hence the guard. The [exc2] handler is POSIX-only, so Windows needs none
//     of this.
//   * it declares `namespace prosper`, so including it anywhere nested produces
//     `prosper::{anonymous}::prosper` and every later `prosper::` lookup resolves into the INNER
//     namespace and fails. The first attempt put it beside the handler, inside an anonymous
//     namespace inside `namespace prosper`, and Linux/macOS CI rejected it with
//     "'guest_module_name' is not a member of 'prosper::{anonymous}::prosper'". A header that opens
//     a namespace can only be included where that namespace means what the file expects.
#if defined(__linux__) || defined(__APPLE__)
#include "../host/raw_syscall.hpp"
#endif
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
#include <array>
#include <cinttypes>
#include <condition_variable>
#include <chrono>
#include <new>
#ifdef _WIN32
#include <windows.h>   // GetCurrentThreadStackLimits/GetCurrentThreadId for the guest-thread trampoline
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <signal.h>
#ifdef __linux__
#include <ucontext.h>
#endif
#endif

namespace prosper {
uint64_t sync_trace_tid_value(uint64_t native_tid) {
    return native_tid;
}

namespace {
    uint64_t sclog_pthread_filter() {
        static const uint64_t filter = [] {
            const char* value = getenv("PROSPER_SYNCLOG_PTHREAD");
            return value ? strtoull(value, nullptr, 0) : 0ull;
        }();
        return filter;
    }
    bool sclog_time_enabled() {
        static const bool enabled = getenv("PROSPER_SYNCLOG") != nullptr;
        if (!enabled) return false;
        static const uint64_t delay_ms = [] {
            const char* value = getenv("PROSPER_SYNCLOG_DELAY_MS");
            return value ? strtoull(value, nullptr, 10) : 0ull;
        }();
        if (!delay_ms) return true;
        static const auto start = std::chrono::steady_clock::now();
        return std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(delay_ms);
    }
    bool sclog_thread_enabled() {
        const uint64_t filter = sclog_pthread_filter();
        return sclog_time_enabled() && (!filter || (uint64_t)pthread_self() == filter);
    }
    bool sclog_condition() {
        static const bool semaphore_only = getenv("PROSPER_SYNCLOG_SEMA_ONLY") != nullptr;
        const bool thread_enabled = sclog_thread_enabled();
        return thread_enabled && !semaphore_only;
    }
    bool sclog_semaphore() {
        static const bool condition_only = getenv("PROSPER_SYNCLOG_COND_ONLY") != nullptr;
        const bool thread_enabled = sclog_thread_enabled();
        return thread_enabled && !condition_only;
    }
    bool sclog() {
        static const bool condition_only = getenv("PROSPER_SYNCLOG_COND_ONLY") != nullptr;
        static const bool semaphore_only = getenv("PROSPER_SYNCLOG_SEMA_ONLY") != nullptr;
        const bool thread_enabled = sclog_thread_enabled();
        return thread_enabled && !condition_only && !semaphore_only;
    }
    std::atomic<uint64_t> g_sclog_focus_sema{0};
    bool sclog_focused_semaphore(uint64_t sema) {
        return sclog_time_enabled() && sema &&
               g_sclog_focus_sema.load(std::memory_order_acquire) == sema;
    }
    uint64_t sctid() {
#if defined(__linux__) || defined(__APPLE__)
        return sync_trace_tid_value(static_cast<uint64_t>(prosper_gettid()));
#elif defined(_WIN32)
        return sync_trace_tid_value(static_cast<uint32_t>(GetCurrentThreadId()));
#else
        return 0;
#endif
    }
#ifdef _WIN32
    std::mutex g_thread_handle_mx;
    std::unordered_map<uint64_t, HANDLE> g_thread_handles;
    void register_current_thread_handle(uint64_t guest_thread) {
        HANDLE duplicate = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &duplicate,
                             0, FALSE, DUPLICATE_SAME_ACCESS)) {
            // Fail LOUDLY. An unregistered thread is unreachable by sceKernelRaiseException, and on
            // the primary thread that is the #2139 heap corruption -- a defect whose symptom appears
            // arbitrarily far from here. A silent return would make the one condition that produces
            // it indistinguishable from normal operation.
            std::fprintf(stderr,
                         "[thread] register_current_thread_handle: DuplicateHandle failed for guest "
                         "thread 0x%llx (GetLastError=%lu) -- this thread is NOT reachable by "
                         "sceKernelRaiseException\n",
                         (unsigned long long)guest_thread, (unsigned long)GetLastError());
            return;
        }
        std::lock_guard<std::mutex> lk(g_thread_handle_mx);
        auto [it, inserted] = g_thread_handles.emplace(guest_thread, duplicate);
        if (!inserted) { CloseHandle(it->second); it->second = duplicate; }
    }
    void unregister_current_thread_handle(uint64_t guest_thread) {
        HANDLE handle = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_thread_handle_mx);
            auto it = g_thread_handles.find(guest_thread);
            if (it != g_thread_handles.end()) { handle = it->second; g_thread_handles.erase(it); }
        }
        if (handle) CloseHandle(handle);
    }
    // #2139: the ONLY caller of register_current_thread_handle was win_thread_trampoline, so the
    // primary guest thread -- which never runs through it -- had no entry here, and in a live title
    // win_raise_exception's pthread_gethandle fallback did not resolve it either: every raise at it
    // took the esrch-no-handle branch. The IL2CPP GC therefore could not stop the primary thread and
    // marked the heap while it kept mutating, which is heap corruption, not a missed suspend.
    // Measured on Blue Prince: 45/45 raises at the primary refused, 739/739 at workers accepted;
    // after registering it, 0 refusals and the title streams its 784 MB asset file for the first
    // time on Windows.
    //
    // CONFIDENCE: HIGH that registering the primary is correct (it is exactly the symmetry the
    // worker path already has, and the live measurement is unambiguous). CONFIDENCE: LOW on WHY the
    // fallback fails: standalone probes on this toolchain show pthread_gethandle naming the process
    // main thread, a pthread_create worker, AND a raw CreateThread thread, so the emulator-specific
    // reason is NOT established. Do not repeat the retracted explanation that winpthreads cannot
    // name a foreign thread -- that was measured false. The registry no longer depends on it.
    HANDLE duplicate_registered_thread_handle(uint64_t guest_thread) {
        HANDLE duplicate = nullptr;
        std::lock_guard<std::mutex> lk(g_thread_handle_mx);
        auto it = g_thread_handles.find(guest_thread);
        if (it != g_thread_handles.end())
            DuplicateHandle(GetCurrentProcess(), it->second, GetCurrentProcess(), &duplicate,
                            0, FALSE, DUPLICATE_SAME_ACCESS);
        return duplicate;
    }
#endif
}
}
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)

// --- mutex attributes (opaque; retain Sony type identity beside the host attribute) ---
// Type 4 intentionally maps to host ERRORCHECK for FreeBSD-compatible self-lock behavior, so the
// host attribute alone cannot distinguish it from Sony type 1 when Gettype reads it back.
struct GuestMutexAttr {
    pthread_mutexattr_t host;
    int sony_type;
};

// PS5 condition attributes carry clock ids that are not all representable by host
// pthread_condattr_t (notably Virtual/Prof, and macOS lacks condattr_setclock). Keep the guest
// identity explicitly and translate its deadline when a timed wait is performed.
struct GuestCondAttr {
    int32_t clock_id = 0; // 0=Realtime, 1=Virtual, 2=Prof, 4=Monotonic
    int32_t pshared = 0;  // only process-private condition variables are supported
};

// Declared here, defined with its full rationale beside the mutex handlers below: the attribute
// handlers in this section forward host pthread failures too, and a raw host errno reaching the
// guest is the #1612 defect regardless of which spelling it arrives through.
namespace { inline uint64_t fbsd_errno(int host); }

HLE(k_mutexattr_init) {
    if (!a0) return 0x16; // EINVAL-ish
    auto* at = (GuestMutexAttr*)calloc(1, sizeof(GuestMutexAttr));
    if (!at) return 0x0c; // ENOMEM
    int result = pthread_mutexattr_init(&at->host);
    if (result != 0) { free(at); return fbsd_errno(result); }
    // A FRESH attr's type must be the FreeBSD/Sony DEFAULT — ERRORCHECK — not the host default
    // (glibc: NORMAL). Kyty's PthreadMutexattrInit does exactly this (settype(1) on init). The
    // #183 change covered the no-attr init path but left attr-without-settype at glibc NORMAL,
    // which self-deadlocks guests that build their own recursion on the EDEADLK contract: UE4's
    // PS5 lock wrapper (PPSA17942 eboot+0x24ca4b6) does
    //   err = mutex_lock(obj); if (err) /* EDEADLK: already mine */ skip-acquire; depth++;
    // — with a NORMAL mutex the relock BLOCKS forever instead of returning EDEADLK (live-verified:
    // single-thread wedge in k_mutex_lock, mutex __owner == self, ~10 s into the DOLL boot).
    // CONFIDENCE: HIGH (FreeBSD contract + Kyty + live wedge disassembly).
    result = pthread_mutexattr_settype(&at->host, PTHREAD_MUTEX_ERRORCHECK);
    if (result != 0) { pthread_mutexattr_destroy(&at->host); free(at); return fbsd_errno(result); }
    at->sony_type = 1;
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
    auto* at = (GuestMutexAttr*)*(void**)a0;
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
    int result = pthread_mutexattr_settype(&at->host, host);
    if (result != 0) return fbsd_errno(result);
    at->sony_type = (int)a1;
    return 0;
}
// scePthreadMutexattrGettype: the read-back inverse of settype. Was MISSING -> the generic stub
// returned 0 while leaving the caller's `int* type` (a1) unwritten, so the guest read stack garbage as
// the mutex type (the harmful-Get-stub class the affinity/sched paths already guard against). Preserve
// the original Sony type because host ERRORCHECK represents both Sony ERRORCHECK(1) and ADAPTIVE_NP(4).
// scePthreadMutexattrGettype has no POSIX spelling registered, so — like scePthreadMutexTimedlock —
// it encodes in place rather than through SCE_PTHREAD_ALIAS. Registering `pthread_mutexattr_gettype`
// onto this body later would need the alias split instead (#2178).
HLE(k_mutexattr_gettype) {
    if (!a0 || !*(void**)a0 || !a1) return prosper::hle::kSceKernelErrorEINVAL;
    auto* at = (GuestMutexAttr*)*(void**)a0;
    *(int*)(uintptr_t)a1 = at->sony_type;
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
HLE(k_mutexattr_destroy)     { if (a0 && *(void**)a0) { auto* at = (GuestMutexAttr*)*(void**)a0;
                               pthread_mutexattr_destroy(&at->host); free(at); *(void**)a0 = nullptr; } return 0; }

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

    // A DESTROYED slot, distinct from an uninitialised one (#2170). FreeBSD writes its own poison
    // through the caller's handle on *_destroy -- THR_MUTEX_DESTROYED (2), THR_COND_DESTROYED and
    // THR_RWLOCK_DESTROYED (1) -- and every later operation on it returns EINVAL, loudly. prosper
    // wrote NULL, which pt_static_sentinel above reads as "statically initialised, self-init me", so
    // ensure_mutex/cond/rwlock SILENTLY MANUFACTURED a fresh, unheld object and the operation
    // succeeded. Thread A holds rwlock R, thread B destroys it, A's next rdlock gets a brand-new
    // lock and enters the critical section alongside A -- no error, no log line. Same silent-unsync
    // class as #2012, through a different door.
    //
    // FreeBSD's own poison values cannot be reused as-is: 1 and 2 are both below the 0x1000
    // threshold, so pt_static_sentinel swallows them exactly as it swallowed NULL. This needs a
    // value of its own. Sub-page so it can never be mistaken for a real object pointer, and not
    // 0/1/2/3 so it cannot collide with an initializer the guest legitimately writes.
    inline constexpr uintptr_t kPtDestroyed = 0xDEA;
    inline bool pt_destroyed_sentinel(void* v) { return (uintptr_t)v == kPtDestroyed; }

    // Loud, because this is a BEHAVIOUR CHANGE: the operation used to succeed. A guest that
    // use-after-destroys is buggy, but it was buggy silently, and a title that somehow depended on
    // the old self-init must announce itself rather than fail somewhere else later.
    // Take the slot ATOMICALLY and return whatever it held (#2176). Two threads racing destroy on
    // the same slot both used to read the non-sentinel pointer before either cleared it, and both
    // handed the SAME storage to retire_sync_object. Before #2042 that double-freed immediately;
    // with the quarantine it double-frees ~30 s later from an unrelated call, which is strictly
    // harder to diagnose -- it cuts against the very argument the quarantine was made for.
    //
    // The exchange installs the destroyed sentinel, so it also closes the second edge in the same
    // move: every handler used to retire BEFORE clearing the slot, leaving a window where another
    // thread could still resolve the object out of it after its quarantine clock had started. Now
    // the object is unreachable the instant the clock starts, and exactly one racer gets a
    // non-sentinel pointer back to retire.
    inline void* pt_claim_slot(uint64_t slot_addr) {
        if (!slot_addr) return nullptr;
        void* prev = __atomic_exchange_n((void**)(uintptr_t)slot_addr,
                                         (void*)kPtDestroyed, __ATOMIC_ACQ_REL);
        return pt_static_sentinel(prev) ? nullptr : prev;
    }

    inline void pt_report_destroyed(const char* what) {
        static std::atomic<unsigned> seen{0};
        if (seen.fetch_add(1) < 16)
            fprintf(stderr, "[pthread] use-after-destroy on a %s: refused with EINVAL "
                            "(it used to self-initialise a fresh object and succeed) -- #2170\n", what);
    }

    struct GuestCondState {
        int32_t clock_id = 0;
        uint64_t generation = 0;
    };
    std::mutex g_guest_cond_state_mutex;
    std::unordered_map<pthread_cond_t*, GuestCondState> g_guest_cond_states;

    void guest_cond_register(pthread_cond_t* cond, int32_t clock_id) {
        std::lock_guard<std::mutex> lock(g_guest_cond_state_mutex);
        g_guest_cond_states[cond] = GuestCondState{clock_id, 0};
    }
    void guest_cond_unregister(pthread_cond_t* cond) {
        std::lock_guard<std::mutex> lock(g_guest_cond_state_mutex);
        g_guest_cond_states.erase(cond);
    }
    GuestCondState guest_cond_snapshot(pthread_cond_t* cond) {
        std::lock_guard<std::mutex> lock(g_guest_cond_state_mutex);
        auto found = g_guest_cond_states.find(cond);
        return found == g_guest_cond_states.end() ? GuestCondState{} : found->second;
    }
    void guest_cond_advance(pthread_cond_t* cond) {
        std::lock_guard<std::mutex> lock(g_guest_cond_state_mutex);
        auto found = g_guest_cond_states.find(cond);
        if (found != g_guest_cond_states.end()) ++found->second.generation;
    }
#ifdef _WIN32
    // winpthreads' cooperative try-lock loop cannot rely on pthread_mutex_lock to report a
    // same-thread ERRORCHECK relock before it starts polling. Track ownership explicitly on Windows
    // for that path. Native POSIX hosts already implement the ERRORCHECK contract in pthreads; putting
    // their every guest lock/unlock through this one process-global map mutex serializes unrelated UE4
    // locks and severely throttles synchronization-heavy titles (#719/#793 regression).
    struct GuestMutexState { int type = PTHREAD_MUTEX_NORMAL; uint64_t owner = 0; uint32_t depth = 0; };
    std::mutex g_guest_mutex_state_mutex;
    std::unordered_map<pthread_mutex_t*, GuestMutexState> g_guest_mutex_states;
    void guest_mutex_register(pthread_mutex_t* mutex, int type) {
        std::lock_guard<std::mutex> lock(g_guest_mutex_state_mutex);
        g_guest_mutex_states[mutex] = {type, 0, 0};
    }
    void guest_mutex_unregister(pthread_mutex_t* mutex) {
        std::lock_guard<std::mutex> lock(g_guest_mutex_state_mutex);
        g_guest_mutex_states.erase(mutex);
    }
    bool guest_mutex_self_deadlock(pthread_mutex_t* mutex) {
        std::lock_guard<std::mutex> lock(g_guest_mutex_state_mutex);
        auto found = g_guest_mutex_states.find(mutex);
        return found != g_guest_mutex_states.end() &&
               found->second.type == PTHREAD_MUTEX_ERRORCHECK &&
               found->second.owner == (uint64_t)pthread_self();
    }
    void guest_mutex_acquired(pthread_mutex_t* mutex) {
        std::lock_guard<std::mutex> lock(g_guest_mutex_state_mutex);
        auto found = g_guest_mutex_states.find(mutex);
        if (found == g_guest_mutex_states.end()) return;
        const uint64_t self = (uint64_t)pthread_self();
        if (found->second.owner == self) {
            ++found->second.depth;
        } else {
            found->second.owner = self;
            found->second.depth = 1;
        }
    }
    bool guest_mutex_released(pthread_mutex_t* mutex) {
        std::lock_guard<std::mutex> lock(g_guest_mutex_state_mutex);
        auto found = g_guest_mutex_states.find(mutex);
        if (found == g_guest_mutex_states.end() ||
            found->second.owner != (uint64_t)pthread_self()) return false;
        if (found->second.depth > 1) {
            --found->second.depth;
        } else {
            found->second.owner = 0;
            found->second.depth = 0;
        }
        return true;
    }
#else
    inline void guest_mutex_register(pthread_mutex_t*, int) {}
    inline void guest_mutex_unregister(pthread_mutex_t*) {}
    inline bool guest_mutex_self_deadlock(pthread_mutex_t*) { return false; }
    inline void guest_mutex_acquired(pthread_mutex_t*) {}
    inline bool guest_mutex_released(pthread_mutex_t*) { return false; }
#endif
#ifdef _WIN32
    const CondWaitMutexBookkeeping kGuestMutexCondWaitBookkeepingHooks{
        guest_mutex_released, guest_mutex_acquired};
    const CondWaitMutexBookkeeping* const kGuestMutexCondWaitBookkeeping =
        &kGuestMutexCondWaitBookkeepingHooks;
#else
    constexpr const CondWaitMutexBookkeeping* kGuestMutexCondWaitBookkeeping = nullptr;
#endif

    pthread_mutex_t* ensure_mutex(uint64_t slot_addr) {
        if (!slot_addr) return nullptr;
        void** slot = (void**)(uintptr_t)slot_addr;
        void* cur = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
        if (!pt_static_sentinel(cur)) return (pthread_mutex_t*)cur;
        if (pt_destroyed_sentinel(cur)) { pt_report_destroyed("mutex"); return nullptr; }
        auto* m = (pthread_mutex_t*)calloc(1, sizeof(pthread_mutex_t));
        pthread_mutexattr_t at; pthread_mutexattr_init(&at);
#ifdef _WIN32
        // winpthreads needs the explicit type so cooperative polling can reproduce FreeBSD's
        // same-thread EDEADLK result. Preserve the native-POSIX initializer used before #793 on
        // Linux/macOS; changing every UE4 static lock there was outside the Windows fix's scope and
        // materially changes title progression.
        pthread_mutexattr_settype(&at, PTHREAD_MUTEX_ERRORCHECK);
#endif
        // The FreeBSD static sentinel ENCODES the type: NULL = PTHREAD_MUTEX_INITIALIZER and
        // 1 = PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP. Windows maps both to ERRORCHECK above because
        // its cooperative poller needs an explicit same-owner result. Native POSIX retains the
        // host-default self-init behavior used before #793; forcing every static UE4 mutex to
        // ERRORCHECK there changed both scheduling and title progression.
        pthread_mutex_init(m, &at);
        pthread_mutexattr_destroy(&at);
        if (__atomic_compare_exchange_n(slot, &cur, (void*)m, false,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            guest_mutex_register(m, PTHREAD_MUTEX_ERRORCHECK);
            return m;
        }
        pthread_mutex_destroy(m); free(m);          // lost the init race: use the winner's object
        return (pthread_mutex_t*)cur;
    }
    pthread_cond_t* ensure_cond(uint64_t slot_addr) {
        if (!slot_addr) return nullptr;
        void** slot = (void**)(uintptr_t)slot_addr;
        void* cur = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
        if (!pt_static_sentinel(cur)) return (pthread_cond_t*)cur;
        if (pt_destroyed_sentinel(cur)) { pt_report_destroyed("condition variable"); return nullptr; }
        auto* c = (pthread_cond_t*)calloc(1, sizeof(pthread_cond_t));
        if (!c) return nullptr;
        if (pthread_cond_init(c, nullptr) != 0) { free(c); return nullptr; }
        guest_cond_register(c, 0);
        if (__atomic_compare_exchange_n(slot, &cur, (void*)c, false,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            return c;
        guest_cond_unregister(c);
        pthread_cond_destroy(c); free(c);
        return (pthread_cond_t*)cur;
    }
    // A guest rwlock is the host lock PLUS the outstanding-hold accounting FreeBSD keeps in its
    // umtx word, because glibc keeps none that survives an unmatched unlock (#2012, and the long
    // comment above the handlers). The accounting belongs to the LOCK, not to a thread: FreeBSD
    // permits a read hold to be released by a different thread, and prosper's own fibers can move
    // between host threads across a yield.
    // The write flag and the hold count live in ONE atomic word, so "is this a decrementable
    // non-write hold?" is decided and committed by a single compare-exchange. Two separate atomics
    // cannot express that: the unlocking thread's loads of the two can straddle a writer's acquire,
    // and no amount of re-checking fixes it, because a re-check is an ABA test — it distinguishes
    // "an owner is here now" from "no owner", never "an owner came and went". See rw_release.
    //   bit 63       WRITE — a write hold is outstanding; `owner` names the thread holding it
    //   bits 32..62  must remain ZERO — see the static_assert below
    //   bits 0..31   COUNT — outstanding acquisitions (each reader 1, a writer 1)
    //
    // A NON-OWNER CAN NEITHER SET NOR CLEAR BIT 63, which is what makes every claim below hold
    // rather than merely being likely: the write path runs only behind an exclusive host lock, the
    // write branch of rw_release is gated on `owner == self`, and the reader path's `s - 1` is
    // guarded by `(s & kRwCount) != 0` so it can never borrow out of bit 31 into the flag space.
    // A write hold therefore cannot be stolen by anything.
    struct GuestRwlock {
        pthread_rwlock_t rw;
        std::atomic<uint64_t> state;   // WRITE | COUNT
        std::atomic<uint64_t> owner;   // owning tid while WRITE is set; only read when WRITE is set
    };
    constexpr uint64_t kRwWrite = 1ull << 63;
    constexpr uint64_t kRwCount = 0xffffffffull;
    // The gap between COUNT and WRITE is unused and MUST stay that way: the layout has no
    // structural defence, so an arithmetic edit written without the guards above (a bare `s + 1`,
    // or widening COUNT) walks straight into the flag space and silently turns a hold count into a
    // write hold. Nothing reachable spills into these bits today — COUNT is bounded by the number
    // of concurrent guest holds — and this assert is the reminder, not the protection.
    static_assert((kRwWrite & kRwCount) == 0, "the WRITE flag must not overlap the hold count");
    static_assert(kRwCount == 0xffffffffull && kRwWrite == (1ull << 63),
                  "bits 32..62 are reserved and must remain zero");
    GuestRwlock* guest_rwlock_create() {
        void* mem = calloc(1, sizeof(GuestRwlock));
        if (!mem) return nullptr;
        auto* g = new (mem) GuestRwlock{};
        if (pthread_rwlock_init(&g->rw, nullptr) != 0) { g->~GuestRwlock(); free(mem); return nullptr; }
        return g;
    }
    // ONLY for an object that was never published through a guest slot: the init-failure path above
    // and the lost-init-race path below. No other thread can hold this pointer, so the storage
    // really can go back to the allocator — which matters, because the lost-race path is the
    // CONTENDED one and must stay allocation-neutral. A published object is retired instead
    // (`retire_sync_object`, #2042); do not reuse this for one.
    void guest_rwlock_free_unpublished(GuestRwlock* g) {
        if (!g) return;
        pthread_rwlock_destroy(&g->rw);
        g->~GuestRwlock();
        free(g);
    }
    GuestRwlock* ensure_rwlock(uint64_t slot_addr) {
        if (!slot_addr) return nullptr;
        void** slot = (void**)(uintptr_t)slot_addr;
        void* cur = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
        if (!pt_static_sentinel(cur)) return (GuestRwlock*)cur;
        if (pt_destroyed_sentinel(cur)) { pt_report_destroyed("rwlock"); return nullptr; }
        auto* g = guest_rwlock_create();
        if (!g) return nullptr;
        if (__atomic_compare_exchange_n(slot, &cur, (void*)g, false,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            return g;
        guest_rwlock_free_unpublished(g);   // lost the init race: use the winner's object
        return (GuestRwlock*)cur;
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

#ifdef _WIN32
bool win_guest_mutex_owned_by_current_thread_for_test(void* mutex) {
    return guest_mutex_self_deadlock(static_cast<pthread_mutex_t*>(mutex));
}
#endif

// scePthreadMutexInit(mutex_slot, attr_slot, name) / pthread_mutex_init(mutex_slot, attr_slot):
// honor the caller's attr (type set via k_mutexattr_settype); no attr means Sony's default,
// ERRORCHECK — FreeBSD PTHREAD_MUTEX_DEFAULT, and what Kyty's attr-init settype(1) produces (#145).
HLE(k_mutex_init) {
    if (!a0) return 0x16;
    auto* m = (pthread_mutex_t*)calloc(1, sizeof(pthread_mutex_t));
    GuestMutexAttr* guest_at = (a1 && !pt_static_sentinel(*(void**)a1)) ? (GuestMutexAttr*)*(void**)a1 : nullptr;
    pthread_mutexattr_t* at = guest_at ? &guest_at->host : nullptr;
    pthread_mutexattr_t def;
    if (!at) { pthread_mutexattr_init(&def); pthread_mutexattr_settype(&def, PTHREAD_MUTEX_ERRORCHECK); at = &def; }
    int host_type = PTHREAD_MUTEX_NORMAL;
    pthread_mutexattr_gettype(at, &host_type);
    pthread_mutex_init(m, at);
    static const bool mtxlog = getenv("PROSPER_MUTEXLOG") != nullptr;
    if (mtxlog) {
        int t = -1; pthread_mutexattr_gettype(at, &t);
        fprintf(stderr, "[mtx] init m=%p slot=0x%llx attr=%p type=%d\n", (void*)m,
                (unsigned long long)a0, at == &def ? nullptr : (void*)at, t);
    }
    if (at == &def) pthread_mutexattr_destroy(&def);
    *(void**)a0 = m;
    guest_mutex_register(m, host_type);
    return 0;
}
// The guest sees FreeBSD errno values. EBUSY(16)/EPERM(1)/EINVAL(22) coincide with Linux and MinGW,
// which is why this went unnoticed for so long — but the values that differ are the ones these
// wrappers actually return on their interesting paths: EDEADLK is FreeBSD 11 / Linux 35 /
// MinGW-winpthreads 36, and EAGAIN is the mirror image (FreeBSD 35 / Linux 11). Handing back the
// host number swaps those two, so "would block, try again" arrives as "deadlock" and vice versa.
//
// This used to special-case EDEADLK alone and leave every other divergent value raw, which is why
// callers below had to bolt on their own `rc == ETIMEDOUT ? 60 : ...` (Linux ETIMEDOUT is 110,
// FreeBSD's is 60). The complete host->FreeBSD table now lives in sce_errno.hpp; keep this thin
// alias so existing call sites read the same, and never return a bare host errno to a guest (#1612).
namespace { inline uint64_t fbsd_errno(int host) {
    if (host == 0) return 0;
    return (uint64_t)static_cast<uint32_t>(prosper::hle::freebsd_errno_from_host(host));
} }
// --- DESTROYING A GUEST SYNC OBJECT (#2042) — read this before editing any *Destroy handler ------
//
// Every one of these handlers used to `free()` the object outright. That is a use-after-free waiting
// for a guest that destroys an object another thread is still using: the handlers resolve the
// pointer from a guest slot and then dereference it, sometimes while PARKED INSIDE it
// (`interruptible_mutex_lock`, `pthread_rwlock_rdlock`), so the freed block is handed to the next
// `calloc` and prosper's own heap is corrupted. The storage now goes into a time-bounded QUARANTINE
// instead, and the host `pthread_*_destroy` runs at reclaim rather than here — which keeps the
// guest's bug the guest's bug. `sync_retire.hpp` carries the full argument, the measurement that
// sized the window, and what the window does not close. `sceKernelDeleteEventFlag` and
// `sceKernelDeleteSema` below already solved the same hazard their own way (defer the free to the
// last waiter) and are left alone.
//
// WHAT THESE HANDLERS DELIBERATELY STILL DO NOT DO: refuse a busy destroy. FreeBSD libthr — the
// platform the guest was built against — is NOT uniform here, and the differences are not
// guessable, so each spelling follows its own source rather than a house rule:
//
//   pthread_mutex_destroy   EBUSY when an owner is recorded (`PMUTEX_OWNER_ID(m) != 0`); frees only
//                           when unowned. EINVAL on an already-destroyed object.
//   pthread_cond_destroy    EBUSY when `__has_user_waiters || kcond.c_has_waiters`. EINVAL likewise.
//   pthread_barrier_destroy EBUSY when `b_waiters > 0`, or when another destroy is in progress; it
//                           also waits out `b_refcount` on the barrier's own condition.
//   pthread_rwlock_destroy  NO busy check of any kind — it frees unconditionally and returns 0,
//                           whatever is held or blocked. EINVAL only for a double destroy.
//   sem_destroy             no busy check; it does not free at all (the sem_t is caller storage).
//
// So the EBUSY that is obvious for a *rwlock* is exactly the one the platform does not have, and
// inventing it here would be a guest-visible divergence introduced to paper over an emulator-side
// memory-safety problem — the wrong trade in both directions. The three EBUSY results FreeBSD DOES
// specify are a separate defect (a contract divergence, not a lifetime bug): prosper has no owner
// tracking on POSIX hosts and no waiter counts for cond/barrier, so each needs new accounting and
// is guest-visible on every title. Filed as #2168 rather than folded in here. The slot prosper
// clears to NULL is a third divergence — FreeBSD writes a DESTROYED poison and fails EINVAL, where
// prosper's sentinel test self-initializes a fresh unheld object instead — filed as #2170.
// CONFIDENCE: HIGH on the table above (read out of libthr and libc, the guest's own platform);
// CONFIDENCE: MED that Sony's libkernel did not re-write these bodies, which no evidence in the
// corpus can currently settle either way — a title observed testing a Destroy result would.
HLE(k_mutex_destroy) { auto* m = (pthread_mutex_t*)pt_claim_slot(a0);   // #2176: claim, then retire
    if (m) { guest_mutex_unregister(m); retire_sync_object(m, SyncObjectKind::Mutex,
                                    [](void* p) { pthread_mutex_destroy((pthread_mutex_t*)p); }); }
    return 0; }
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
// PROSPER_MUTEX_WAITLOG: name the HOLDER of a contended guest mutex when a thread is about to block on
// it — a deadlock/lock-contention probe. POSIX-only (Linux/macOS): Windows already has the guest-mutex
// ownership map above, and winpthreads lacks pthread_getname_np. Fully gated: the owner map is only
// touched when the env var is set, so there is zero cost by default.
#ifndef _WIN32
namespace {
    struct MtxOwner { long tid = 0; char name[16] = {0}; };
    std::mutex g_mtx_waitlog_mx;
    std::unordered_map<pthread_mutex_t*, MtxOwner> g_mtx_waitlog_owner;
    inline bool mtx_waitlog() { static const bool on = getenv("PROSPER_MUTEX_WAITLOG") != nullptr; return on; }
    void mtx_waitlog_record(pthread_mutex_t* m) {
        if (!mtx_waitlog()) return;
        MtxOwner o; o.tid = (long)prosper_gettid();
        pthread_getname_np(pthread_self(), o.name, sizeof o.name);
        std::lock_guard<std::mutex> lk(g_mtx_waitlog_mx); g_mtx_waitlog_owner[m] = o;
    }
    void mtx_waitlog_clear(pthread_mutex_t* m) {
        if (!mtx_waitlog()) return;
        std::lock_guard<std::mutex> lk(g_mtx_waitlog_mx); g_mtx_waitlog_owner.erase(m);
    }
    void mtx_waitlog_report_block(pthread_mutex_t* m, uint64_t slot) {
        if (!mtx_waitlog()) return;
        if (pthread_mutex_trylock(m) == 0) { pthread_mutex_unlock(m); return; } // uncontended
        MtxOwner o{}; bool have = false;
        { std::lock_guard<std::mutex> lk(g_mtx_waitlog_mx);
          auto it = g_mtx_waitlog_owner.find(m); if (it != g_mtx_waitlog_owner.end()) { o = it->second; have = true; } }
        char self[16] = {0}; pthread_getname_np(pthread_self(), self, sizeof self);
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 256)
            fprintf(stderr, "[mtx-wait] '%s'(tid=%ld) BLOCKING on guest mutex slot=0x%llx held by %s\n",
                    self, (long)prosper_gettid(), (unsigned long long)slot,
                    have ? ([&]{ static char b[48]; snprintf(b, sizeof b, "'%s'(tid=%ld)", o.name, o.tid); return b; }())
                         : "<owner unknown / acquired via cond_wait>");
    }
}
#else
namespace {
    inline void mtx_waitlog_record(pthread_mutex_t*) {}
    inline void mtx_waitlog_clear(pthread_mutex_t*) {}
    inline void mtx_waitlog_report_block(pthread_mutex_t*, uint64_t) {}
}
#endif
HLE(k_mutex_lock)    {
    auto* m = ensure_mutex(a0); if (!m) return 0x16;
    if (guest_mutex_self_deadlock(m)) return mtx_report("lock", a0, m, EDEADLK);
    mtx_waitlog_report_block(m, a0);
    const int result = interruptible_mutex_lock(m);
    if (result == 0) { guest_mutex_acquired(m); mtx_waitlog_record(m); }
    return mtx_report("lock", a0, m, result);
}
HLE(k_mutex_trylock) {
    auto* m = ensure_mutex(a0); if (!m) return 0x16;
    const int result = pthread_mutex_trylock(m);
    if (result == 0) { guest_mutex_acquired(m); mtx_waitlog_record(m); }
    return mtx_report("trylock", a0, m, result);
}
HLE(k_mutex_unlock)  {
    auto* m = ensure_mutex(a0); if (!m) return 0x16;
    const int result = pthread_mutex_unlock(m);
    if (result == 0) { guest_mutex_released(m); mtx_waitlog_clear(m); }
    return mtx_report("unlock", a0, m, result);
}

// --- Sony vs POSIX failure encoding for the shared pthread bodies (#1945, family of #1612) -------
// The handler bodies above are registered under BOTH spellings. They are not the same contract:
//   * POSIX `pthread_mutex_trylock` returns the bare errno (EBUSY).
//   * Sony's `scePthreadMutexTrylock` returns the libkernel-encoded SCE_KERNEL_ERROR_* form
//     (0x8002_0000 | FreeBSD errno), the same encoding every sceKernel* entry point uses.
// The shipped guest `libc.prx` proves the Sony side directly: its C11 `_Mtx_*`/`_Cnd_*` wrappers
// call the scePthread* entry points and compare the result against the ENCODED constants —
// `_Mtx_trylock` tests 0x80020010 (EBUSY), `_Mtx_lock` tests 0x8002000b (EDEADLK),
// `_Mtx_timedlock`/`_Cnd_timedwait` test 0x8002003c (ETIMEDOUT), `_Cnd_timedwait` also tests
// 0x80020001 (EPERM). A bare errno matches none of them, so the wrapper maps the result to
// Dinkumware's `_Thrd_error` (4), and `std::_Throw_C_error(4)` raises an *uncaught*
// std::system_error("invalid argument") that terminates the process. A perfectly ordinary
// "the mutex was busy" therefore killed the guest.
// CONFIDENCE: HIGH (guest disassembly of the shipped libc.prx is the primary evidence).
namespace {
    inline uint64_t sce_pthread_rc(uint64_t posix_rc) {
        // Pass through success, and anything a body already encoded or that is not an errno.
        if (posix_rc == 0 || (posix_rc & ~0xffull) != 0) return posix_rc;
        return prosper::hle::sce_kernel_error(
            static_cast<prosper::hle::FreeBsdErrno>(static_cast<uint32_t>(posix_rc)));
    }
}
// One Sony-spelling alias per shared body: same behaviour, libkernel error encoding.
#define SCE_PTHREAD_ALIAS(sce_name, posix_body) \
    HLE(sce_name) { return sce_pthread_rc(posix_body(a0, a1, a2, a3, a4, a5)); }
SCE_PTHREAD_ALIAS(k_sce_mutex_lock,    k_mutex_lock)
SCE_PTHREAD_ALIAS(k_sce_mutex_trylock, k_mutex_trylock)
// #2178: Unlock belongs here as much as Lock and Trylock do. It was the sharpest instance of the
// gap — `scePthreadMutexUnlock` reports EPERM for a mutex this thread does not hold, and it was the
// ONLY member of the lock/trylock/timedlock/unlock quartet handing that back bare, so the encoding
// a guest saw depended on which of the four it called. Init and the two fallible attribute setters
// travel with them for the same reason: each has a failure path, so each needs the split.
SCE_PTHREAD_ALIAS(k_sce_mutex_unlock,      k_mutex_unlock)
SCE_PTHREAD_ALIAS(k_sce_mutex_init,        k_mutex_init)
SCE_PTHREAD_ALIAS(k_sce_mutexattr_init,    k_mutexattr_init)
SCE_PTHREAD_ALIAS(k_sce_mutexattr_settype, k_mutexattr_settype)

// --- condition variables ---
namespace {
    constexpr int32_t kSonyClockRealtime = 0;
    constexpr int32_t kSonyClockVirtual = 1;
    constexpr int32_t kSonyClockProf = 2;
    constexpr int32_t kSonyClockMonotonic = 4;
    constexpr int64_t kCondClockSliceNs = 50'000'000;

    bool valid_cond_clock(int32_t clock_id) {
        return clock_id == kSonyClockRealtime || clock_id == kSonyClockVirtual ||
               clock_id == kSonyClockProf || clock_id == kSonyClockMonotonic;
    }
    GuestCondAttr* guest_condattr(uint64_t slot_addr) {
        if (!slot_addr) return nullptr;
        void* attr = *(void**)(uintptr_t)slot_addr;
        return pt_static_sentinel(attr) ? nullptr : (GuestCondAttr*)attr;
    }

#ifdef _WIN32
    uint64_t filetime_ticks(const FILETIME& value) {
        ULARGE_INTEGER ticks{};
        ticks.LowPart = value.dwLowDateTime;
        ticks.HighPart = value.dwHighDateTime;
        return ticks.QuadPart;
    }
#endif

    // Sony's Virtual clock is process user CPU time; Prof adds kernel CPU time. Neither is a
    // portable pthread condattr clock, so collect them explicitly on every bounded wait slice.
    int guest_cond_clock_now(int32_t clock_id, timespec& result) {
        if (clock_id == kSonyClockRealtime || clock_id == kSonyClockMonotonic) {
            const clockid_t host_clock = clock_id == kSonyClockRealtime ? CLOCK_REALTIME : CLOCK_MONOTONIC;
            return clock_gettime(host_clock, &result) == 0 ? 0 : (errno ? errno : EINVAL);
        }
#ifdef _WIN32
        FILETIME created{}, exited{}, kernel{}, user{};
        if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return EINVAL;
        uint64_t ticks = filetime_ticks(user);
        if (clock_id == kSonyClockProf) ticks += filetime_ticks(kernel);
        result.tv_sec = (time_t)(ticks / 10'000'000ull);
        result.tv_nsec = (long)((ticks % 10'000'000ull) * 100ull);
        return 0;
#else
        rusage usage{};
        // HOST errno on purpose: this helper's result flows up to k_cond_timedwait, which applies
        // fbsd_errno() itself. Translating here would translate twice — and the second pass reads a
        // FreeBSD number as a host one, so ETIMEDOUT (60) would come back out as EINVAL. Its sibling
        // return paths are host errnos for the same reason.
        if (getrusage(RUSAGE_SELF, &usage) != 0) return errno ? errno : EINVAL;
        int64_t sec = usage.ru_utime.tv_sec;
        int64_t usec = usage.ru_utime.tv_usec;
        if (clock_id == kSonyClockProf) {
            sec += usage.ru_stime.tv_sec;
            usec += usage.ru_stime.tv_usec;
        }
        sec += usec / 1'000'000;
        usec %= 1'000'000;
        result.tv_sec = (time_t)sec;
        result.tv_nsec = (long)(usec * 1000);
        return 0;
#endif
    }

    bool timespec_reached(const timespec& now, const timespec& deadline) {
        return now.tv_sec > deadline.tv_sec ||
               (now.tv_sec == deadline.tv_sec && now.tv_nsec >= deadline.tv_nsec);
    }

    timespec stable_slice_duration(const timespec& selected_now,
                                   const timespec& selected_deadline) {
        int64_t wait_ns = kCondClockSliceNs;
        if (selected_deadline.tv_sec == selected_now.tv_sec) {
            wait_ns = selected_deadline.tv_nsec - selected_now.tv_nsec;
        } else if (selected_deadline.tv_sec == selected_now.tv_sec + 1) {
            const int64_t remainder = 1'000'000'000ll - selected_now.tv_nsec +
                                      selected_deadline.tv_nsec;
            if (remainder < wait_ns) wait_ns = remainder;
        }
        return timespec{(time_t)(wait_ns / 1'000'000'000ll),
                        (long)(wait_ns % 1'000'000'000ll)};
    }

    int interruptible_cond_clock_timedwait(pthread_cond_t* cond, pthread_mutex_t* mutex,
                                           const timespec& deadline, int32_t clock_id,
                                           CondWaitMutexState* mutex_state,
                                           const CondWaitMutexBookkeeping* bookkeeping) {
        if (mutex_state) *mutex_state = {};
        if (clock_id == kSonyClockRealtime)
            return interruptible_cond_timedwait(cond, mutex, &deadline,
                                                GuestWaitKind::ConditionSequence, 0,
                                                mutex_state, bookkeeping);
        if (!valid_cond_clock(clock_id)) return EINVAL;

        const uint64_t initial_generation = guest_cond_snapshot(cond).generation;
        for (;;) {
            timespec selected_now{};
            int result = guest_cond_clock_now(clock_id, selected_now);
            if (result != 0) return result;
            if (timespec_reached(selected_now, deadline)) return ETIMEDOUT;

            const timespec slice = stable_slice_duration(selected_now, deadline);
            result = interruptible_cond_timedwait_relative(
                cond, mutex, &slice, GuestWaitKind::ConditionSequence, 0, mutex_state,
                bookkeeping);
            if (result != ETIMEDOUT) return result;
            // A signal can land between two host timed waits. Treat a generation change as a
            // permitted spurious wake instead of losing the signal during clock conversion.
            if (guest_cond_snapshot(cond).generation != initial_generation) return 0;
        }
    }
}

HLE(k_condattr_init) {
    if (!a0) return 0x16;
    auto* attr = (GuestCondAttr*)calloc(1, sizeof(GuestCondAttr));
    if (!attr) return 0x0c;
    *(void**)a0 = attr;
    return 0;
}
HLE(k_condattr_destroy) {
    if (a0 && !pt_static_sentinel(*(void**)a0)) free(*(void**)a0);
    if (a0) *(void**)a0 = nullptr;
    return 0;
}
HLE(k_condattr_setclock) {
    auto* attr = guest_condattr(a0);
    if (!attr || !valid_cond_clock((int32_t)a1)) return 0x16;
    attr->clock_id = (int32_t)a1;
    return 0;
}
HLE(k_condattr_getclock) {
    auto* attr = guest_condattr(a0);
    if (!attr || !a1) return 0x16;
    *(int32_t*)(uintptr_t)a1 = attr->clock_id;
    return 0;
}
HLE(k_condattr_setpshared) {
    auto* attr = guest_condattr(a0);
    if (!attr || a1 != 0) return 0x16;
    attr->pshared = 0;
    return 0;
}
HLE(k_condattr_getpshared) {
    auto* attr = guest_condattr(a0);
    if (!attr || !a1) return 0x16;
    *(int32_t*)(uintptr_t)a1 = attr->pshared;
    return 0;
}
HLE(k_cond_init) {
    if (!a0) return 0x16;
    const GuestCondAttr* attr = guest_condattr(a1);
    auto* cond = (pthread_cond_t*)calloc(1, sizeof(pthread_cond_t));
    if (!cond) return 0x0c;
    const int result = pthread_cond_init(cond, nullptr);
    // #1612, not #1984: this is the BARE value both spellings report, and a raw host number is
    // wrong through either. pthread_cond_init's EAGAIN is host 11 on Linux, which is FreeBSD's
    // EDEADLK — "out of resources, retry" delivered to the guest as "deadlock".
    if (result != 0) { free(cond); return fbsd_errno(result); }
    guest_cond_register(cond, attr ? attr->clock_id : kSonyClockRealtime);
    *(void**)a0 = cond;
    return 0;
}
// The Sony spellings of the fallible condition-variable entry points (#2178). Destroy, Signal,
// Broadcast and Wait are deliberately absent: each returns 0 unconditionally, so an alias would
// change nothing today and misstate the contract tomorrow. scePthreadCondTimedwait encodes in
// place instead — it has no POSIX spelling registered on its body.
SCE_PTHREAD_ALIAS(k_sce_cond_init,             k_cond_init)
SCE_PTHREAD_ALIAS(k_sce_condattr_init,         k_condattr_init)
SCE_PTHREAD_ALIAS(k_sce_condattr_setclock,     k_condattr_setclock)
SCE_PTHREAD_ALIAS(k_sce_condattr_getclock,     k_condattr_getclock)
SCE_PTHREAD_ALIAS(k_sce_condattr_setpshared,   k_condattr_setpshared)
SCE_PTHREAD_ALIAS(k_sce_condattr_getpshared,   k_condattr_getpshared)
HLE(k_cond_destroy) {
    if (auto* cond = (pthread_cond_t*)pt_claim_slot(a0)) {   // #2176: claim, then retire
        // Quarantined, not freed, and the host `pthread_cond_destroy` runs at reclaim rather than
        // here — see the contract block above k_mutex_destroy and sync_retire.hpp (#2042). A thread
        // parked in interruptible_cond_wait is inside this very object.
        interruptible_cond_forget(cond);
        guest_cond_unregister(cond);
        retire_sync_object(cond, SyncObjectKind::Cond,
                           [](void* p) { pthread_cond_destroy((pthread_cond_t*)p); });
    }
    return 0;
}
HLE(k_cond_signal)    { if (sclog_condition()) fprintf(stderr, "[sync2] T%" PRIu64 " COND.signal    cond=0x%llx\n", sctid(), a0 ? (unsigned long long)*(void**)a0 : 0); if (auto* c = ensure_cond(a0)) { guest_cond_advance(c); interruptible_cond_signal(c); } return 0; }
HLE(k_cond_broadcast) { if (sclog_condition()) fprintf(stderr, "[sync2] T%" PRIu64 " COND.broadcast cond=0x%llx\n", sctid(), a0 ? (unsigned long long)*(void**)a0 : 0); if (auto* c = ensure_cond(a0)) { guest_cond_advance(c); interruptible_cond_broadcast(c); } return 0; }
HLE(k_cond_wait)      { if (sclog_condition()) fprintf(stderr, "[sync2] T%" PRIu64 " COND.wait.ent  cond=0x%llx\n", sctid(), a0 ? (unsigned long long)*(void**)a0 : 0);
    { auto* c = ensure_cond(a0); auto* m = ensure_mutex(a1);
      if (c && m) {
          (void)interruptible_cond_wait(c, m, GuestWaitKind::ConditionSequence, 0,
                                        nullptr, kGuestMutexCondWaitBookkeeping);
      } }
    if (sclog_condition()) fprintf(stderr, "[sync2] T%" PRIu64 " COND.wait.exit cond=0x%llx\n", sctid(), a0 ? (unsigned long long)*(void**)a0 : 0); return 0; }
// POSIX pthread_cond_timedwait(cond_slot, mutex_slot, const timespec* abstime) — abstime is an
// absolute deadline in the condition attribute's selected clock ({i64 sec, i64 nsec}, FreeBSD ==
// Linux x86-64 layout), and the
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
    if (!a2) {
        int rc = interruptible_cond_wait(c, m, GuestWaitKind::ConditionSequence, 0,
                                         nullptr, kGuestMutexCondWaitBookkeeping);
        return fbsd_errno(rc);
    }
    const int64_t* gts = (const int64_t*)(uintptr_t)a2;
    if (gts[0] < 0 || gts[1] < 0 || gts[1] >= 1'000'000'000ll) return 22;
    struct timespec dl { (time_t)gts[0], (long)gts[1] };
    const int32_t clock_id = guest_cond_snapshot(c).clock_id;
    int rc = interruptible_cond_clock_timedwait(
        c, m, dl, clock_id, nullptr, kGuestMutexCondWaitBookkeeping);
    return fbsd_errno(rc);   // host ETIMEDOUT (Linux 110) -> FreeBSD 60
}

// --- read/write locks (opaque handle -> host pthread_rwlock_t). Real libc.prx uses these for its
// internal state (locale, stdio, malloc arenas); stubbing them to no-ops leaves that state UNLOCKED
// under the 15-thread IL2CPP pool -> data races. Real locking is a genuine correctness fix. ---
HLE(k_rwlock_init) {
    if (!a0) return 0x16;
    auto* g = guest_rwlock_create();
    if (!g) return 12;   // ENOMEM
    *(void**)a0 = g;    // store handle through caller's slot (a1=attr, a2=name ignored)
    return 0;
}
// #2042. Destroying a rwlock while another thread is inside one of the handlers below used to free
// the object under it — and the rwlock is the worst of the family, because `k_rwlock_rdlock` parks
// the caller INSIDE `pthread_rwlock_rdlock(&g->rw)` with the raw pointer live for as long as the
// lock stays write-held. The storage is quarantined instead (`sync_retire.hpp`), and the host
// `pthread_rwlock_destroy` runs at reclaim, since a parked thread may be sitting in it right now.
//
// The hold accounting from #2012 is NOT consulted here, and that is the interesting part: FreeBSD's
// own `pthread_rwlock_destroy` performs no busy check at all and returns 0 whatever is held, so an
// EBUSY refusal would be a divergence rather than a fix. See the contract table above
// k_mutex_destroy. The return value, and the cleared slot, are exactly what they were before.
HLE(k_rwlock_destroy) {
    if (void* g = pt_claim_slot(a0))            // #2176: claim, then retire
        retire_sync_object(g, SyncObjectKind::Rwlock,
            [](void* p) { auto* r = (GuestRwlock*)p; pthread_rwlock_destroy(&r->rw); r->~GuestRwlock(); });
    return 0;
}
// An UNMATCHED unlock must never reach the host lock (#2012).
//
// glibc's pthread_rwlock_unlock is *undefined* when the lock is not held, and what it actually does
// is unrecoverable: with the lock in read phase it takes the reader-release path and subtracts one
// reader from `__readers`, so an unmatched unlock drives the reader count NEGATIVE. Every later
// writer then blocks forever waiting for a reader population that can never drain, and every later
// reader inherits a corrupt lock. There is no error, no signal, and no way back — one stray unlock
// permanently bricks that rwlock.
//
// Measured on Sonic Racing: CrossWorlds (PPSA08804, UE5): the guest's own main thread parks in
// pthread_rwlock_wrlock forever while the host lock reads __readers = 0xfffffff2 — WRLOCKED set and
// a reader count of exactly -2, i.e. two more read-unlocks than read-locks. Nothing else in the
// 64-thread process is spinning; the whole engine is waiting on that one thread.
//
// FreeBSD, the platform the guest was built against, cannot get into that state, and its rule is
// specifically NOT "the caller must hold it":
//   * `_pthread_rwlock_unlock` (libthr) checks the WRITE owner in userland -> EPERM for a non-owner
//     write unlock.
//   * everything else defers to `_umtx_op(UMTX_OP_RW_UNLOCK)`, which returns EPERM only when there
//     is no write owner AND the reader count is zero. With readers present it decrements and
//     succeeds — with no thread-identity check at all.
// So a read hold taken on one thread and released on another is LEGAL there, and prosper must not
// refuse it: `sceFiberRun` resumes a suspended fiber on whichever host thread calls it, so a fiber
// that read-locks, yields, and is resumed elsewhere would release from a different host thread. A
// per-thread rule would deadlock exactly the way this fix exists to prevent. The accounting
// therefore lives WITH THE LOCK, not with the thread, and reproduces FreeBSD's rule directly.
// CONFIDENCE: HIGH (libthr + the umtx path; the glibc corruption was read out of a live process).
//
// PROSPER_RWLOCKLOG=1 reports refusals, =2 every rwlock operation — both rate-limited per report
// kind, so a flood of one kind cannot starve another (see rw_report). PROSPER_RWLOCK_UNSAFE_UNLOCK
// (set to ANY value, including 0) restores the old forwarding so the A/B stays reproducible.
namespace {
    int rwlocklog() {
        static const int lvl = []{ const char* e = getenv("PROSPER_RWLOCKLOG");
                                   return e ? atoi(e) : 0; }();
        return lvl;
    }
    // Under this hatch the accounting is KNOWINGLY UNSOUND, and that is the point: a refused unlock
    // is forwarded to the host lock anyway while COUNT keeps its value, so COUNT stops tracking the
    // outstanding host holds and later EPERMs and hangs are the reproduction, not new bugs. It
    // exists so the A/B that established #2012 stays runnable. Never diagnose under it.
    bool rwlock_unsafe_unlock() {
        static const bool on = getenv("PROSPER_RWLOCK_UNSAFE_UNLOCK") != nullptr;
        return on;
    }
    // The OS thread id, in the same vocabulary as /proc/<pid>/task and guest_bt. prosper_gettid()
    // does not exist on Windows (posix_shim.hpp is entirely #ifndef _WIN32), so this mirrors the
    // sctid() split above. Any live thread's id is non-zero on every platform, which is what makes
    // `writer != 0` usable as "write-held".
    uint64_t rw_self_tid() {
#if defined(__linux__) || defined(__APPLE__)
        return (uint64_t)prosper_gettid();
#elif defined(_WIN32)
        return (uint64_t)GetCurrentThreadId();
#else
        return (uint64_t)(uintptr_t)pthread_self();
#endif
    }
    // WHY THE WRITE FLAG AND THE COUNT SHARE ONE WORD, written down because two earlier versions of
    // this function got it wrong in two different ways and the second one looked airtight.
    //
    // A stray unlock — the thing this accounting exists to refuse — runs concurrently with real
    // acquisitions, so any decision it makes by reading TWO atomics can be invalidated between the
    // reads. Version 2 read `writer` then `holds`, which let the pair straddle a writer's acquire.
    // Version 3 re-read `writer` after the decrement and undid it if an owner had appeared; that is
    // an ABA test, and it only narrowed the window. It answers "is an owner here NOW?", never "did
    // an owner come and go?", so this interleaving still slipped through (S = stray, W' and W" =
    // balanced writers, all values permitted by the memory model):
    //
    //   1. idle: count=0, writer=0
    //   2. S loads writer -> 0, takes the reader path
    //   3. W' acquires: writer=W', count 0->1
    //   4. S loads count -> 1
    //   5. W' releases: count 1->0, writer=0, forwards its host unlock — the lock is free
    //   6. S pre-checks writer -> 0, proceeds
    //   7. W" acquires the free lock: writer=W", count 0->1
    //   8. S's CAS expects count==1, and count IS 1 (W"'s) -> succeeds, count->0
    //   9. W" releases: owner==self, decrement-if-positive finds 0, forwards its own host unlock
    //  10. S post-checks writer -> 0 -> returns 0 -> forwards a SECOND host unlock on a free lock
    //      -> glibc takes the reader path -> __readers negative -> #2012, reproduced by its own fix
    //
    // Reaching that needs S to be interrupted inside two ~2-instruction gaps, so a clean stress run
    // does not refute it — which is exactly why it must be closed by construction rather than by
    // testing. Folding WRITE into the counted word does that, and it kills the trace at step 4
    // rather than step 8: W' publishes `WRITE|1`, so S's load sees bit 63 and takes the write
    // branch, where `owner != self` refuses it. Even holding a reader-era `s`, S's CAS compares the
    // whole 64-bit value against one that now reads `WRITE|1` — never equal — so it re-reads and
    // re-decides. There is no interleaving in which a reader-path decrement commits against a
    // write-held lock.
    void rw_acquired(GuestRwlock* g, bool write) {
        if (!write) { g->state.fetch_add(1, std::memory_order_acq_rel); return; }
        // The owner store is sequenced-before the acq_rel CAS below, so any thread that observes
        // WRITE through an acquire load of `state` has synchronized-with that CAS and therefore
        // sees THIS owner value. That pairing — not any claim that `owner` is stable — is what
        // makes reading `owner` before the CAS in rw_release sound; `owner` is genuinely NOT
        // stable, because this store runs for the next writer before its WRITE becomes visible.
        g->owner.store(rw_self_tid(), std::memory_order_release);
        // `s` is 0 here, and the tight reason is worth recording: COUNT > 0 while the host lock is
        // acquirable would require a host-side release that did not decrement COUNT, and the only
        // construct that produces one is the PROSPER_RWLOCK_UNSAFE_UNLOCK escape hatch. The CAS
        // loop rather than a plain store is a cheap concession to that hatch and to future callers.
        uint64_t s = g->state.load(std::memory_order_acquire);
        while (!g->state.compare_exchange_weak(s, (s | kRwWrite) + 1,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {}
    }
    // 0 = released (the caller may now unlock the host lock); EPERM = FreeBSD refuses this unlock
    // and the host lock must not be touched.
    int rw_release(GuestRwlock* g) {
        const uint64_t self = rw_self_tid();
        uint64_t s = g->state.load(std::memory_order_acquire);
        for (;;) {
            if (s & kRwWrite) {
                // Only the owner may release a write hold (FreeBSD checks this in libthr). Reading
                // `owner` before the CAS is sound by release/acquire pairing, not by stability: the
                // acquire load of `state` that produced this WRITE synchronizes-with the owner's
                // acq_rel CAS in rw_acquired, which its `owner.store(release)` is sequenced-before,
                // so the value read here is that writer's. Read-read coherence also rules out the
                // tid-reuse case — a later `owner` value cannot be observed before an earlier one.
                if (g->owner.load(std::memory_order_acquire) != self) return EPERM;
                // Clear WRITE and drop the count in ONE commit. Decrement-if-positive, so a count
                // that somehow reached 0 cannot wrap to 4 billion and refuse every future unlock.
                const uint64_t ns = (s & ~kRwWrite) - ((s & kRwCount) ? 1u : 0u);
                if (!g->state.compare_exchange_weak(s, ns, std::memory_order_acq_rel,
                                                    std::memory_order_acquire))
                    continue;                              // raced: re-read and re-decide
                // Cleared before the caller forwards its host unlock, so the next writer to acquire
                // cannot observe this owner.
                g->owner.store(0, std::memory_order_release);
                return 0;
            }
            if ((s & kRwCount) == 0) return EPERM;         // nothing held: refuse, lock untouched
            // A read hold is released, and it need not be THIS thread's — nor even the same hold
            // that was outstanding when `s` was loaded. That residual ABA is unobservable rather
            // than merely tolerable: COUNT is a pure counter with no identity, nothing downstream
            // of this CAS reads the value of `s`, and the caller forwards exactly one host unlock.
            // The invariant that matters is COUNT == outstanding host holds, and a steal preserves
            // it exactly — glibc goes N -> N-1 either way, and *which* reader was robbed is not a
            // distinction this system can express. The robbed reader's own release then finds COUNT
            // one short, takes a spurious EPERM and does NOT forward, so glibc lands at 0 rather
            // than negative. (FreeBSD accepts the same unlock, but that is a coincidence of a
            // matching design, not the reason this is safe — do not reason from it.)
            if (g->state.compare_exchange_weak(s, s - 1, std::memory_order_acq_rel,
                                               std::memory_order_acquire))
                return 0;
        }
    }
    // One counter per report KIND: a flood of one must not consume another's budget (the trap this
    // repository records as asymmetric cap exhaustion). Each line carries its own kind's ordinal.
    enum class RwReport { Failed, Unmatched, Trace, Count_ };
    void rw_report(RwReport kind, const char* what, uint64_t slot, const GuestRwlock* g, int rc) {
        static std::atomic<int> printed[(size_t)RwReport::Count_];
        const int n = printed[(size_t)kind].fetch_add(1);
        if (n >= 64 && (n & (n - 1)) != 0) return;   // first 64 of this kind, then powers of two
        const uint64_t s = g ? g->state.load(std::memory_order_relaxed) : 0;
        fprintf(stderr, "[rwlock] %s slot=0x%llx rw=%p rc=%d tid=%llu holds=%llu%s #%d\n", what,
                (unsigned long long)slot, (const void*)g, rc,
                (unsigned long long)rw_self_tid(),
                (unsigned long long)(s & kRwCount), (s & kRwWrite) ? " WRITE" : "", n);
    }
    // A lock call that FAILED records no hold, and since #2024 it also reports that failure to the
    // guest. Both halves matter and they are the same invariant seen from two sides: COUNT must
    // equal the outstanding host holds, and the guest must not believe it holds what it does not.
    // A handler that returned 0 on a refusal broke the second half, and the guest's own eventual
    // unlock — correct from its point of view — then arrived unmatched at a lock with no hold to
    // release, which is the #2012 corruption originating INSIDE prosper rather than in the guest.
    void rw_note_lock_result(const char* what, uint64_t slot, GuestRwlock* g, int rc) {
        if (rc == 0) { if (rwlocklog() >= 2) rw_report(RwReport::Trace, what, slot, g, 0); return; }
        rw_report(RwReport::Failed, what, slot, g, rc);
    }
}
// A REFUSED acquisition must be reported as one (#2024). These two returned 0 unconditionally, so a
// guest told "the lock is yours" after a real EDEADLK/EAGAIN entered the critical section unguarded
// and its later, entirely correct unlock arrived unmatched — the #2012 corruption manufactured by
// prosper itself. `rc` fed only a diagnostic line.
//
// `fbsd_errno`, never the bare `rc`, and the reason is specific rather than stylistic: EDEADLK and
// EAGAIN are the two values that SWAP between the host and the platform the guest was built against
// (FreeBSD EDEADLK 11 / EAGAIN 35; Linux the mirror image). These handlers' interesting refusals are
// exactly those two, so returning the host number would hand a guest "try again" where the host said
// "deadlock" — the #1612 defect, on the paths that produce it most.
//
// The `!g` case (a null slot, or a lock object that could not be created) reports EINVAL, matching
// `tryrdlock`, `trywrlock` and all four timed forms. It DELIBERATELY folds the out-of-memory
// sub-case into EINVAL rather than reporting ENOMEM: `ensure_rwlock` does not distinguish them, and
// answering one question two ways across the eight rwlock acquisition handlers would be its own
// defect. Reachable only for a guest passing a null rwlock, where EINVAL is the platform answer.
HLE(k_rwlock_rdlock)  {
    auto* g = ensure_rwlock(a0);
    if (!g) return 0x16;   // EINVAL
    const int rc = pthread_rwlock_rdlock(&g->rw);
    if (rc == 0) rw_acquired(g, false);
    rw_note_lock_result("rdlock", a0, g, rc);
    return fbsd_errno(rc);
}
HLE(k_rwlock_wrlock)  {
    auto* g = ensure_rwlock(a0);
    if (!g) return 0x16;   // EINVAL
    const int rc = pthread_rwlock_wrlock(&g->rw);
    if (rc == 0) rw_acquired(g, true);
    rw_note_lock_result("wrlock", a0, g, rc);
    return fbsd_errno(rc);
}
HLE(k_rwlock_unlock)  {
    auto* g = ensure_rwlock(a0);
    // EINVAL, matching every sibling rwlock entry point (rd/wr/tryrd/trywr and all four timed
    // forms). ensure_rwlock returns null only for a null slot address or a failed allocation;
    // neither is a successful unlock, and FreeBSD answers EINVAL for the first. Reporting 0 here
    // told a guest that an unlock of a null handle had succeeded, hiding its bug behind ours —
    // and made this the one rwlock call whose answer depended on which entry point was used.
    // The two causes fold together because ensure_rwlock does not distinguish them (#2159).
    if (!g) return 0x16;
    const int rc = rw_release(g);
    if (rc != 0) {
        if (rwlocklog() || !rwlock_unsafe_unlock())
            rw_report(RwReport::Unmatched,
                      "UNMATCHED unlock: the lock is unheld, or write-held by another thread — "
                      "refused (FreeBSD returns EPERM); forwarding it would corrupt the lock",
                      a0, g, rc);
        // POSIX spelling reports the bare FreeBSD errno; the Sony spelling is registered through
        // SCE_PTHREAD_ALIAS below, which encodes it the way libkernel does (#1984's split).
        if (!rwlock_unsafe_unlock()) return fbsd_errno(rc);
        pthread_rwlock_unlock(&g->rw);   // PROSPER_RWLOCK_UNSAFE_UNLOCK: the pre-#2012 behaviour
        return 0;
    }
    pthread_rwlock_unlock(&g->rw);
    if (rwlocklog() >= 2) rw_report(RwReport::Trace, "unlock", a0, g, 0);
    return 0;
}
HLE(k_rwlock_tryrdlock){
    auto* g = ensure_rwlock(a0); if (!g) return 0x16;
    const int rc = pthread_rwlock_tryrdlock(&g->rw);
    if (rc == 0) rw_acquired(g, false);
    // EAGAIN (host 11) is FreeBSD EDEADLK unmapped — the #1612 defect. Never a bare host errno.
    return fbsd_errno(rc);
}
HLE(k_rwlock_trywrlock){
    auto* g = ensure_rwlock(a0); if (!g) return 0x16;
    const int rc = pthread_rwlock_trywrlock(&g->rw);
    if (rc == 0) rw_acquired(g, true);
    return fbsd_errno(rc);
}
// The Sony spellings report failure in the libkernel encoding; the POSIX ones keep the bare errno.
// Same split, and the same reason, as the mutex aliases above — but note the evidence boundary:
// #1984's proof is the shipped guest libc.prx comparing MUTEX and CONDVAR results against the
// encoded constants. No guest code has been observed testing an encoded RWLOCK result, so applying
// it here is an extrapolation from a consistent libkernel-wide convention rather than a direct
// observation. It is the safer of the two errors (a guest that only tests == 0 is unaffected either
// way), and it makes the family self-consistent. CONFIDENCE: MED.
// Init is aliased too: it can fail with ENOMEM (a failed allocation), so it is not a can't-fail
// entry point.
//
// Rdlock and Wrlock are aliased as of #2024, and their previous ABSENCE from this list was not an
// oversight — it was consistent with handlers that could not fail. Registering the Sony spelling
// straight onto the POSIX body is only correct while that body returns nothing but 0; the moment it
// can return a failure, the Sony spelling starts handing the guest a BARE FreeBSD errno where every
// other libkernel entry point hands it the encoded form. Making a handler fallible and giving it its
// alias are therefore one change, not two, and splitting them would produce exactly the #1984 defect
// this alias exists to prevent.
SCE_PTHREAD_ALIAS(k_sce_rwlock_init,      k_rwlock_init)
SCE_PTHREAD_ALIAS(k_sce_rwlock_rdlock,    k_rwlock_rdlock)
SCE_PTHREAD_ALIAS(k_sce_rwlock_wrlock,    k_rwlock_wrlock)
SCE_PTHREAD_ALIAS(k_sce_rwlock_unlock,    k_rwlock_unlock)
SCE_PTHREAD_ALIAS(k_sce_rwlock_tryrdlock, k_rwlock_tryrdlock)
SCE_PTHREAD_ALIAS(k_sce_rwlock_trywrlock, k_rwlock_trywrlock)
HLE(k_rwlockattr_init) {
    if (!a0) return 0x16;
    auto* attr = (pthread_rwlockattr_t*)calloc(1, sizeof(pthread_rwlockattr_t));
    if (!attr) return 12;
    const int rc = pthread_rwlockattr_init(attr);
    if (rc) { free(attr); return fbsd_errno(rc); }
    *(void**)(uintptr_t)a0 = attr;
    return 0;
}
HLE(k_rwlockattr_destroy) {
    if (!a0) return 0x16;
    void** slot = (void**)(uintptr_t)a0;
    if (*slot) { pthread_rwlockattr_destroy((pthread_rwlockattr_t*)*slot); free(*slot); }
    *slot = nullptr;
    return 0;
}
// Both attribute entry points reject a null slot, and Init also forwards an allocation or host
// failure, so both are fallible and both need the split (#2178). They were left raw when the rest
// of the rwlock family was aliased.
SCE_PTHREAD_ALIAS(k_sce_rwlockattr_init,    k_rwlockattr_init)
SCE_PTHREAD_ALIAS(k_sce_rwlockattr_destroy, k_rwlockattr_destroy)

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
struct GuestPthreadAttr {
    pthread_attr_t host;
    void* supplied_stack = nullptr;
    size_t supplied_stack_size = 0;
};
static_assert(offsetof(GuestPthreadAttr, host) == 0);
HLE(k_attr_init) {
    if (a0) {
        auto* at = (GuestPthreadAttr*)calloc(1, sizeof(GuestPthreadAttr));
        if (!at) return 12;
        pthread_attr_init(&at->host); *(void**)a0 = at;
    }
    return 0;
}
HLE(k_attr_destroy) {
    if (a0 && *(void**)a0) {
        auto* at = (GuestPthreadAttr*)*(void**)a0;
        pthread_attr_destroy(&at->host); free(at); *(void**)a0 = nullptr;
    }
    return 0;
}
// The guard MIRRORS k_attr_setstack below: same three rejected conditions, same answer (#2183).
// It used to wrap the success path in `if (a0 && *a0 && a1 >= 16384)` and return 0 regardless, so a
// guest asking for an 8 KiB stack was told YES here and NO by scePthreadAttrSetstack -- the same
// question answered two ways (the #1873 shape) -- and the thread was then created with whatever size
// the attribute already carried. The guest believed it had configured a stack it had not.
//
// Rejecting rather than clamping, because k_attr_setstack already assumes rejection and FreeBSD's
// pthread_attr_setstacksize returns EINVAL below PTHREAD_STACK_MIN. CONFIDENCE: MED on the boundary
// itself -- no live title evidence pins the PS5's behaviour at exactly 16 KiB, and the 16384 here is
// inherited from the sibling rather than independently established.
//
// The rejection LOGS unconditionally rather than under PROSPER_SYNCLOG. This handler previously
// could not fail, so any title relying on the silent acceptance changes behaviour here, and that
// must not be invisible: a refused stack request is exactly the kind of thing that surfaces later as
// an unexplained thread fault with nothing connecting it back to this call.
HLE(k_attr_setstacksize) {
    if (!a0 || !*(void**)(uintptr_t)a0 || a1 < 16384) {
        static std::atomic<unsigned> refused{0};
        if (refused.fetch_add(1) < 16)
            fprintf(stderr, "[thread] attr setstacksize REFUSED attr=0x%llx size=0x%llx "
                            "(null/uninitialised attr, or below the 16 KiB minimum) -- #2183\n",
                    (unsigned long long)a0, (unsigned long long)a1);
        return 0x16;   // bare EINVAL; the Sony spelling encodes it through the alias below
    }
    auto* at = (GuestPthreadAttr*)*(void**)a0;
    int rc = pthread_attr_setstacksize(&at->host, a1);
    if (getenv("PROSPER_SYNCLOG"))
        fprintf(stderr, "[thread] attr setstacksize attr=%p size=0x%llx rc=%d\n",
                (void*)at, (unsigned long long)a1, rc);
    if (rc == 0) {
        at->supplied_stack = nullptr; at->supplied_stack_size = 0;
    }
    return fbsd_errno(rc);
}
HLE(k_attr_setstack) {
    // Sony and every host backend we support require at least 16 KiB here.  MinGW's winpthreads
    // headers do not consistently expose PTHREAD_STACK_MIN, so keep this check aligned with the
    // setstacksize path above instead of depending on that optional libc macro.
    if (!a0 || !*(void**)(uintptr_t)a0 || !a1 || a2 < 16384) return 0x16;
#ifdef _WIN32
    // The host pthread remains on a winpthreads-owned stack, while win_thread_trampoline switches
    // guest entry to this exact range. Preserve the range explicitly instead of asking CreateThread
    // to choose an address (which Windows cannot do).
    auto* at = (GuestPthreadAttr*)*(void**)(uintptr_t)a0;
    int rc = pthread_attr_setstacksize(&at->host, (size_t)a2);
    if (!rc) { at->supplied_stack = (void*)(uintptr_t)a1; at->supplied_stack_size = (size_t)a2; }
    return fbsd_errno(rc);
#else
    auto* at = (GuestPthreadAttr*)*(void**)(uintptr_t)a0;
    int rc = pthread_attr_setstack(&at->host, (void*)(uintptr_t)a1, (size_t)a2);
    if (getenv("PROSPER_SYNCLOG"))
        fprintf(stderr, "[thread] attr setstack attr=%p base=%p size=0x%llx rc=%d\n",
                (void*)at, (void*)(uintptr_t)a1, (unsigned long long)a2, rc);
    if (!rc) { at->supplied_stack = (void*)(uintptr_t)a1; at->supplied_stack_size = (size_t)a2; }
    return fbsd_errno(rc);
#endif
}
// Both stack-setting entry points are fallible, so both need the Sony/POSIX split (#1984, #2178).
// setstacksize needs it because #2183 just MADE it fallible: landing that without the alias would
// have started handing scePthreadAttrSetstacksize a bare FreeBSD errno, which is exactly the defect
// the alias exists to prevent -- the two are one change, not two.
//
// setstack gets one in the same commit because it was ALREADY returning a bare errno from both its
// 0x16 and its fbsd_errno(rc) paths. Fixing only its sibling would have left the pair still
// disagreeing about the ENCODING, i.e. moved the "same question answered two ways" defect one step
// rather than removing it.
SCE_PTHREAD_ALIAS(k_sce_attr_setstacksize, k_attr_setstacksize)
SCE_PTHREAD_ALIAS(k_sce_attr_setstack,     k_attr_setstack)
HLE(k_attr_setguardsize) {
    if (!a0 || !*(void**)(uintptr_t)a0) return 0x16;   // SCE_KERNEL_ERROR_EINVAL
    auto* at = (GuestPthreadAttr*)*(void**)(uintptr_t)a0;
#ifdef _WIN32
    // winpthreads returns EINVAL for guard sizes it cannot apply to a CreateThread-owned stack.
    // Windows already supplies the host stack's guard page, while an explicitly supplied guest
    // stack is switched in the trampoline and retains the guest mapping's own protection. Accept
    // the valid guest attribute without claiming winpthreads can reshape either backing stack.
    (void)at; (void)a1;
    return 0;
#else
    int rc = pthread_attr_setguardsize(&at->host, (size_t)a1);
    return fbsd_errno(rc);
#endif
}
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
        // A supplied handle names a specific thread. Never replace a failed target lookup with
        // the caller's stack: a collector querying a parked worker would then scan itself.
        bool ok = a0 ? guest_stack_for_thread(a0, &base, &sz)
                     : guest_stack_for_current_thread(&base, &sz);
#ifdef _WIN32
        const uint64_t self = (uint64_t)pthread_self();
        if (!ok || (a0 && a0 != self))
            trace_guest_stack_query(a0, ok, base, sz);
#endif
        if (ok) pthread_attr_setstack(at, base, sz);   // real, tracked stack for that thread
        // else: leave the attr as-is (avoid the fragile pthread_getattr_np)
    }
    return 0;
}
// scePthreadAttrGetstackaddr(attr, void** addr) — Sony reports the stack *base* (low addr).
HLE(k_attr_getstackaddr) {
    if (a0 && *(void**)a0 && a1) {
        auto* at = (GuestPthreadAttr*)*(void**)a0;
        void* base = at->supplied_stack; size_t sz = at->supplied_stack_size;
        if (!base) pthread_attr_getstack(&at->host, &base, &sz);
        *(void**)(uintptr_t)a1 = base;
    }
    return 0;
}
HLE(k_attr_getstacksize) {
    if (a0 && *(void**)a0 && a1) {
        auto* at = (GuestPthreadAttr*)*(void**)a0;
        void* base = at->supplied_stack; size_t sz = at->supplied_stack_size;
        if (!base) pthread_attr_getstack(&at->host, &base, &sz);
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

// Guest-visible names need their own registry: Linux truncates host names to 15 bytes, Darwin can
// rename only the current host thread, and Windows uses thread descriptions. Generation tags keep
// an exiting thread from erasing the name of a later thread that reuses the same pthread_t.
namespace {
constexpr size_t kGuestThreadNameSize = 32;
struct GuestThreadName {
    uint64_t generation = 0;
    std::array<char, kGuestThreadNameSize> value{};
};
std::mutex g_guest_thread_name_mutex;
std::unordered_map<uint64_t, GuestThreadName> g_guest_thread_names;
std::atomic<uint64_t> g_guest_thread_name_generation{1};

uint64_t next_guest_thread_name_generation() {
    uint64_t generation = g_guest_thread_name_generation.fetch_add(1, std::memory_order_relaxed);
    if (generation == 0)
        generation = g_guest_thread_name_generation.fetch_add(1, std::memory_order_relaxed);
    return generation;
}

std::array<char, kGuestThreadNameSize> bounded_guest_thread_name(const char* name) {
    std::array<char, kGuestThreadNameSize> result{};
    if (name) {
        strncpy(result.data(), name, result.size() - 1);
        result.back() = '\0';
    }
    return result;
}

bool publish_guest_thread_name(uint64_t thread, uint64_t generation, const char* name) noexcept {
    std::lock_guard<std::mutex> lock(g_guest_thread_name_mutex);
    try {
        g_guest_thread_names[thread] = GuestThreadName{generation, bounded_guest_thread_name(name)};
        return true;
    } catch (...) {
        return false;
    }
}

void retire_guest_thread_name(uint64_t thread, uint64_t generation) {
    std::lock_guard<std::mutex> lock(g_guest_thread_name_mutex);
    const auto it = g_guest_thread_names.find(thread);
    if (it != g_guest_thread_names.end() && (!generation || it->second.generation == generation))
        g_guest_thread_names.erase(it);
}

bool get_guest_thread_name(uint64_t thread, std::array<char, kGuestThreadNameSize>& name) {
    std::lock_guard<std::mutex> lock(g_guest_thread_name_mutex);
    const auto it = g_guest_thread_names.find(thread);
    if (it != g_guest_thread_names.end()) {
        name = it->second.value;
        return true;
    }
    if (thread == (uint64_t)(uintptr_t)pthread_self()) {
        name.fill('\0');
        return true;
    }
    return false;
}

void set_host_thread_name(uint64_t thread, const char* name) {
    char short_name[16]{};
    if (name) strncpy(short_name, name, sizeof(short_name) - 1);
#ifdef __APPLE__
    if (thread == (uint64_t)(uintptr_t)pthread_self()) pthread_setname_np(short_name);
#elif defined(__linux__)
    pthread_setname_np((pthread_t)(uintptr_t)thread, short_name);
#elif defined(_WIN32)
    if (thread == (uint64_t)(uintptr_t)pthread_self()) {
        wchar_t wide_name[kGuestThreadNameSize]{};
        MultiByteToWideChar(CP_UTF8, 0, name ? name : "", -1, wide_name,
                            static_cast<int>(kGuestThreadNameSize));
        SetThreadDescription(GetCurrentThread(), wide_name);
    }
#endif
}

bool set_guest_thread_name(uint64_t thread, const char* name) {
    std::lock_guard<std::mutex> lock(g_guest_thread_name_mutex);
    auto it = g_guest_thread_names.find(thread);
    if (it == g_guest_thread_names.end()) {
        if (thread != (uint64_t)(uintptr_t)pthread_self()) return false;
        try {
            it = g_guest_thread_names.emplace(thread, GuestThreadName{}).first;
        } catch (...) {
            return false;
        }
    }
    it->second.value = bounded_guest_thread_name(name);
    // Serialize the host rename with registry retirement so a target cannot exit between the
    // lifetime check above and pthread_setname_np receiving its host pthread_t.
    set_host_thread_name(thread, name);
    return true;
}
}

// PROSPER_PRIOLOG (#1195/#1264 investigation): the priority SET calls are otherwise k_attr_noop (prosper
// does not re-prioritize host threads). These gated handlers log the priority a title requests so we can
// see whether it DIFFERENTIATES thread priorities (a PS5 priority-scheduling dependency prosper ignores
// could explain the suspend-point/safe-flag livelock). Lower Sony priority value = higher scheduling
// priority (256 highest .. 767 lowest, default 700). Behaviour is unchanged (still return 0/noop); only
// prints under the env var. SceKernelSchedParam = { int sched_priority } (k_getschedparam writes the same
// single int32), so the *schedparam forms dereference one int from the caller's param pointer.
namespace {
// The gate must be checked BEFORE any caller dereferences a guest SchedParam pointer: an argument
// expression like *(int32_t*)a2 evaluates even when logging is off, which would make the default
// path deref a pointer k_attr_noop never touched (review finding on #1264).
bool priolog_enabled() {
    static const bool on = getenv("PROSPER_PRIOLOG") != nullptr;
    return on;
}
void priolog(const char* api, uint64_t target_thread, long policy, long prio) {
    if (!priolog_enabled()) return;
    char self[16] = {0};
#ifndef _WIN32
    pthread_getname_np(pthread_self(), self, sizeof self);   // winpthreads has no getname_np
#endif
    std::array<char, kGuestThreadNameSize> tname{};
    const bool have_target = target_thread && get_guest_thread_name(target_thread, tname);
    fprintf(stderr, "[prio] %s caller='%s' target=0x%llx'%s'%s policy=%ld prio=%ld\n",
            api, self, (unsigned long long)target_thread,
            have_target ? tname.data() : "?",
            target_thread == (uint64_t)(uintptr_t)pthread_self() ? "(self)" : "",
            policy, prio);
}
}
HLE(k_log_setprio) {          // scePthreadSetprio(thread, prio)
    priolog("setprio", a0, -1, (long)(int32_t)a1);
    return 0;
}
HLE(k_log_setschedparam) {    // scePthread/pthread_setschedparam(thread, policy, SchedParam* param)
    if (priolog_enabled())
        priolog("setschedparam", a0, (long)(int32_t)a1, a2 ? (long)*(int32_t*)(uintptr_t)a2 : -1);
    return 0;
}
HLE(k_log_attr_setschedparam) { // scePthreadAttrSetschedparam(attr, SchedParam* param)
    if (priolog_enabled())
        priolog("attr_setschedparam", 0, -1, a1 ? (long)*(int32_t*)(uintptr_t)a1 : -1);
    return 0;
}

HLE(k_pthread_getname) {
    if (!a0) return 3;    // ESRCH
    if (!a1) return 14;   // EFAULT
    std::array<char, kGuestThreadNameSize> name{};
    if (!get_guest_thread_name(a0, name)) return 3;
    memcpy((void*)(uintptr_t)a1, name.data(), name.size());
    return 0;
}

HLE(k_pthread_rename) {
    if (!a0) return 22;  // EINVAL
    if (!a1) return 0;   // Sony accepts a null name as a no-op
    const char* name = (const char*)(uintptr_t)a1;
    if (!set_guest_thread_name(a0, name)) return 3;
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

// Host-side hooks for the NEXT guest_thread_spawn on this thread. Deliberately a thread-local
// rather than an argument: k_pthread_create also serves scePthreadCreate, whose 5-argument form
// leaves a5 (r9) caller-indeterminate, so reading hooks out of a guest register would dereference
// garbage on every ordinary thread creation (the same class as the a4 name-pointer hazard that
// k_pthread_create_noname exists to avoid). Creation is serialised per calling thread, so a
// thread-local is exactly the right scope.
namespace { thread_local const GuestThreadHooks* t_pending_guest_thread_hooks = nullptr; }

#if defined(__linux__) || defined(__APPLE__)
namespace {
extern "C" uint64_t prosper_call_guest_on_stack(uintptr_t stack_top, uint64_t fn,
                                                  uint64_t a0, uint64_t a1);
struct ThreadStart {
    void* (*entry)(void*);
    void* arg;
    void* guest_stack;
    size_t guest_stack_size;
    char name[kGuestThreadNameSize];
    uint64_t name_generation;
    std::atomic<bool> name_published{false};
    // Optional host-side hooks, both invoked on the HOST %fs — on_enter as the last host action
    // before the guest entry, on_exit as the first after it returns. Null for every existing caller,
    // so this changes nothing for scePthreadCreate. libSceUlt uses them to count RUNNING ulthreads at
    // the real guest-code boundary and to measure guest stack usage (#1603).
    GuestThreadHooks hooks{};
};
// Runs first on the new thread: register our own stack (keyed by our tid) BEFORE any guest code,
// so an early GC_register_my_thread / stack-base query from this thread finds it. Closes a race
// where a fast-starting worker ran before the parent's post-create registration → "Bad stack base".
void* thread_trampoline(void* p) {
    auto* ts = (ThreadStart*)p;
    while (!ts->name_published.load(std::memory_order_acquire)) std::this_thread::yield();
    const uint64_t name_generation = ts->name_generation;
    void* registered_stack = ts->guest_stack;
    size_t registered_stack_size = ts->guest_stack_size;
    {
        pthread_attr_t sa;
        if (pthread_getattr_np(pthread_self(), &sa) == 0) {
            void* sb = nullptr; size_t ss = 0;
            if (!registered_stack && pthread_attr_getstack(&sa, &sb, &ss) == 0 && sb) {
                registered_stack = sb;
                registered_stack_size = ss;
            }
            pthread_attr_destroy(&sa);
        }
    }
    if (registered_stack)
        register_thread_stack((uint64_t)pthread_self(), registered_stack, registered_stack_size);
    install_sigaltstack();   // so a guest stack overflow on this worker is still catchable
    // Adopt the guest's thread name (scePthreadCreate/pthread_create_name_np arg) as the HOST
    // thread name so debugger/procfs views (gdb, /proc/PID/task/*/comm) show the engine's own
    // role names (GameThread, RenderThread, RHIThread, ...) instead of the binary name. Kernel
    // limit is 15 chars + NUL; host libc call, so it must run on the host %fs (we are).
#ifdef __APPLE__
    if (ts->name[0]) {
        char host_name[16]{}; strncpy(host_name, ts->name, sizeof(host_name) - 1);
        pthread_setname_np(host_name);
    }
#else
    if (ts->name[0]) {
        char host_name[16]{}; strncpy(host_name, ts->name, sizeof(host_name) - 1);
        pthread_setname_np(pthread_self(), host_name);
    }
#endif
    auto entry = ts->entry; void* arg = ts->arg;
    void* guest_stack = ts->guest_stack; size_t guest_stack_size = ts->guest_stack_size;
    const GuestThreadHooks hooks = ts->hooks;
    delete ts;   // all host libc; MUST run on the host %fs
    // gated (PROSPER_GUEST_FS): give this guest worker its own guest TCB + static TLS and switch %fs to it
    // as the LAST host action before entering guest code (so guest initial-exec TLS — incl. libc.prx's
    // allocator arena/tcache — resolves to real guest storage, not the aliased host glibc TCB). The import
    // stubs swap back to host %fs per HLE call. No-op when the gate is off. Order matters: the free() above
    // is host glibc (host-TLS tcache) — running it under the guest %fs corrupts the host heap.
    guest_execution_thread_enter(/*primary=*/false); // ALLTHREADS-gated; MUST run on host %fs
    if (hooks.on_enter) hooks.on_enter(hooks.opaque);   // host %fs; see ThreadStart::hooks
    guest_tls_activate_thread();
    void* rv = nullptr;
    if (entry) {
        // glibc reserves part of a native pthread stack for its static TLS and rejects valid Sony
        // stacks as small as 32 KiB. Keep the pthread runtime on its host-owned stack, but enter the
        // guest at the exact caller-supplied base/size.
        rv = guest_stack
            ? (void*)(uintptr_t)prosper_call_guest_on_stack(
                  (uintptr_t)guest_stack + guest_stack_size, (uint64_t)(uintptr_t)entry,
                  (uint64_t)(uintptr_t)arg, 0)
            : entry(arg);
    }
    // Normal-return exit path: purge this thread's __tls_get_addr DTV entries and free the blocks
    // (#68 — glibc recycles pthread ids, so a stale entry would hand the NEXT thread on this id the
    // dead thread's dirty TLS). The guest entry can return with %fs still = the guest TP
    // (PROSPER_GUEST_FS), and the purge is host libc (mutex/free), so switch permanently to host
    // %fs before cleanup. This is a terminal guest boundary: restoring guest %fs before returning
    // makes glibc's start_thread cleanup read its TLS through the guest TCB (#644). Threads that
    // exit via scePthreadExit never get here — its import gate already restored host %fs and
    // k_pthread_exit purges before host pthread_exit.
    (void)guest_fs_to_host_scoped();
    if (hooks.on_exit) hooks.on_exit(hooks.opaque, (uint64_t)(uintptr_t)rv);   // back on host %fs
    tls_dtv_purge_current_thread();
    unregister_thread_stack((uint64_t)pthread_self());   // ids recycle; stale bounds = wrong bounds (#138)
    retire_guest_thread_name((uint64_t)(uintptr_t)pthread_self(), name_generation);
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
struct WinThreadStart {
    uint64_t entry;
    void* arg;
    void* guest_stack;
    size_t guest_stack_size;
    char name[kGuestThreadNameSize];
    uint64_t name_generation;
    std::atomic<bool> name_published{false};
    GuestThreadHooks hooks{};   // see ThreadStart::hooks
};
extern "C" uint64_t prosper_call_guest_sysv(uint64_t fn, uint64_t a0, uint64_t a1);
extern "C" uint64_t prosper_call_guest_on_stack(uintptr_t stack_top, uint64_t fn,
                                                  uint64_t a0, uint64_t a1);

// PS5 code is compiled for a stack whose entire usable range is writable.  Windows instead
// initially commits only the top of a reserved thread stack and relies on a moving guard page;
// native Windows compilers probe each intervening page before a large stack allocation.  Guest
// code has no reason to emit those Windows probes, so an otherwise-valid `sub rsp, large_size`
// followed by a call can jump past the guard and fault in still-reserved memory.  Commit the
// guest worker's usable reservation before entering it, retaining one guard page above the
// permanently inaccessible bottom page so a real stack overflow is still caught.
bool prepare_guest_windows_stack(ULONG_PTR* out_low, ULONG_PTR* out_high) {
    ULONG_PTR api_low = 0, high = 0;
    GetCurrentThreadStackLimits(&api_low, &high);
    if (!high) return false;

    MEMORY_BASIC_INFORMATION top{};
    if (!VirtualQuery((void*)(high - 1), &top, sizeof(top)) || !top.AllocationBase)
        return false;

    const ULONG_PTR reserve_low = (ULONG_PTR)top.AllocationBase;
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const SIZE_T page = si.dwPageSize ? si.dwPageSize : 0x1000;
    if (high <= reserve_low + 2 * page) return false;

    // VirtualAlloc also succeeds for pages that are already committed.  Supplying the entire
    // usable interval therefore both fills the reserved gap and leaves the existing top frames
    // intact.  PAGE_GUARD on an already-committed page is cleared explicitly below.
    void* usable = (void*)(reserve_low + page);
    const SIZE_T usable_size = (SIZE_T)(high - (reserve_low + page));
    if (!VirtualAlloc(usable, usable_size, MEM_COMMIT, PAGE_READWRITE)) return false;

    DWORD old_protect = 0;
    if (!VirtualProtect((void*)(reserve_low + page), page,
                        PAGE_READWRITE | PAGE_GUARD, &old_protect))
        return false;

    if (out_low) *out_low = reserve_low + page;
    if (out_high) *out_high = high;
    return true;
}

#ifdef _WIN32
// #997: a guest worker thread that calls scePthreadExit must NOT reach host pthread_exit — winpthreads'
// pthread_exit unwinds (RtlUnwind) back through the hand-written prosper_call_guest_* bridge and the
// guest frames, none of which carry x64 .pdata unwind info, so RtlUnwind fails and RtlRaiseStatus
// (STATUS_UNSUCCESSFUL) terminates the whole PROCESS kernel-side — no catchable user-mode exception,
// exit ~0xC00000FF, first observed as DOLL's ~4s death when its first "TAsync" async-loader worker
// exits. Escape back to win_thread_trampoline via __builtin_longjmp instead (a plain SP/register
// restore, no SEH unwind — the same technique exec_image_win.cpp's t_jb uses for guest faults), and
// let the trampoline run the ordinary cleanup + winpthreads exit, exactly like a normal-return worker.
static thread_local void*    t_thread_exit_jb[5];
static thread_local int      t_thread_exit_armed = 0;
static thread_local uint64_t t_thread_exit_rv = 0;
static thread_local void*    t_saved_teb_base = nullptr;
static thread_local void*    t_saved_teb_limit = nullptr;
static thread_local int      t_teb_swapped = 0;
// True while the current thread is inside a win_thread_trampoline guest call with the longjmp armed.
bool win_thread_exit_armed() { return t_thread_exit_armed != 0; }
[[noreturn]] void win_thread_exit_longjmp(uint64_t rv) {
    t_thread_exit_rv = rv;
    t_thread_exit_armed = 0;
    __builtin_longjmp(t_thread_exit_jb, 1);
}
#endif

void* win_thread_trampoline(void* p) {
    auto* ts = (WinThreadStart*)p;
    while (!ts->name_published.load(std::memory_order_acquire)) std::this_thread::yield();
    const uint64_t name_generation = ts->name_generation;
    uint64_t entry = ts->entry; void* arg = ts->arg;
    void* guest_stack = ts->guest_stack; size_t guest_stack_size = ts->guest_stack_size;
    char name[sizeof(ts->name)]; memcpy(name, ts->name, sizeof(name));
    const GuestThreadHooks hooks = ts->hooks;
    delete ts;   // host libc: run on host %fs (before activate)
    if (name[0]) {
        wchar_t wname[sizeof(name)]{};
        MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, (int)(sizeof(wname) / sizeof(wname[0])));
        SetThreadDescription(GetCurrentThread(), wname);
    }
    uint64_t guest_thread = (uint64_t)pthread_self();
    register_current_thread_handle(guest_thread);
    // Register this worker's stack bounds FIRST, keyed by the Windows thread id (== exec_image_win's
    // cur_tid) so a guest GC that queries its own stack base (GC_register_my_thread / stack scanning —
    // e.g. Unity's AssetGarbageCollectorHelper) finds real bounds instead of wedging on "Bad stack
    // base". Mirrors the Linux thread_trampoline's register-first ordering. GetCurrentThreadStackLimits
    // gives the committed stack range (Win8+).
    ULONG_PTR stk_lo = 0, stk_hi = 0;
    if (!prepare_guest_windows_stack(&stk_lo, &stk_hi))
        GetCurrentThreadStackLimits(&stk_lo, &stk_hi);
    void* registered_base = guest_stack ? guest_stack : (void*)stk_lo;
    size_t registered_size = guest_stack ? guest_stack_size : (size_t)(stk_hi - stk_lo);
    if (registered_base && registered_size) {
        register_thread_stack((uint64_t)GetCurrentThreadId(), registered_base, registered_size);
        register_thread_stack(guest_thread, registered_base, registered_size);
    }
    trace_guest_thread_lifecycle(true, (uint64_t)pthread_self(),
                                 (uint64_t)GetCurrentThreadId(), registered_base, registered_size);
    guest_execution_thread_enter(/*primary=*/false); // Windows boundary; HWBP backend unavailable
    if (hooks.on_enter) hooks.on_enter(hooks.opaque);   // host %fs; see ThreadStart::hooks
    guest_tls_activate_thread();   // this worker's guest %fs TCB (FSGSBASE); no-op if the gate is off
    void* rv = nullptr;
    if (entry) {
        // Arm the scePthreadExit escape (see win_thread_exit_longjmp / #997). __builtin_setjmp returns
        // 0 on the direct path and non-zero when k_pthread_exit longjmps back here.
        if (__builtin_setjmp(t_thread_exit_jb) == 0) {
            t_thread_exit_armed = 1;
            if (guest_stack) {
                NT_TIB* tib = (NT_TIB*)NtCurrentTeb();
                t_saved_teb_base = tib->StackBase; t_saved_teb_limit = tib->StackLimit;
                t_teb_swapped = 1;
                tib->StackLimit = guest_stack;
                tib->StackBase = (void*)((uintptr_t)guest_stack + guest_stack_size);
                rv = (void*)(uintptr_t)prosper_call_guest_on_stack(
                    (uintptr_t)guest_stack + guest_stack_size, entry, (uint64_t)(uintptr_t)arg, 0);
                tib->StackLimit = t_saved_teb_limit; tib->StackBase = t_saved_teb_base;
                t_teb_swapped = 0;
            } else {
                rv = (void*)(uintptr_t)prosper_call_guest_sysv(entry, (uint64_t)(uintptr_t)arg, 0);
            }
            t_thread_exit_armed = 0;
        } else {
            // Returned via scePthreadExit's __builtin_longjmp: recover the exit value and undo any TEB
            // stack swap the guest_stack path left in place (the longjmp skipped its inline restore).
            rv = (void*)(uintptr_t)t_thread_exit_rv;
            if (t_teb_swapped) {
                NT_TIB* tib = (NT_TIB*)NtCurrentTeb();
                tib->StackBase = t_saved_teb_base; tib->StackLimit = t_saved_teb_limit;
                t_teb_swapped = 0;
            }
        }
    }
    if (hooks.on_exit) hooks.on_exit(hooks.opaque, (uint64_t)(uintptr_t)rv);   // back on host %fs
    tls_dtv_purge_current_thread();
    trace_guest_thread_lifecycle(false, (uint64_t)pthread_self(),
                                 (uint64_t)GetCurrentThreadId(), registered_base, registered_size);
    unregister_thread_stack((uint64_t)GetCurrentThreadId());   // ids recycle; stale bounds = wrong bounds
    unregister_thread_stack(guest_thread);
    unregister_current_thread_handle(guest_thread);
    retire_guest_thread_name(guest_thread, name_generation);
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
    void* supplied_stack = nullptr;
    size_t supplied_stack_size = 0;
    int detach = PTHREAD_CREATE_JOINABLE;
    if (a1 && *(void**)a1) {
        auto* guest_attr = (GuestPthreadAttr*)*(void**)a1;
        auto* gat = &guest_attr->host;
        size_t req = 0;
        if (pthread_attr_getstacksize(gat, &req) == 0 && req)
            ssz = req < kStackFloor ? kStackFloor : req;
        supplied_stack = guest_attr->supplied_stack;
        supplied_stack_size = guest_attr->supplied_stack_size;
        pthread_attr_getdetachstate(gat, &detach);
        if (getenv("PROSPER_SYNCLOG"))
            fprintf(stderr, "[thread] attr create attr=%p requested=0x%zx supplied=%p+0x%zx detach=%d\n",
                    (void*)guest_attr, req, supplied_stack, supplied_stack_size, detach);
    }
    // Size-only attrs keep a host-owned/reclaimed stack; an explicit base stays caller-owned and
    // must be copied exactly into the local attr.
    pthread_attr_t la; pthread_attr_init(&la);
    // The trampoline enters guest code on an explicit Sony stack. The host pthread itself retains a
    // glibc-owned stack large enough for static TLS and native start/exit bookkeeping.
    int attr_rc = pthread_attr_setstacksize(&la, supplied_stack ? kStackFloor : ssz);
    if (attr_rc) { pthread_attr_destroy(&la); return (uint64_t)(unsigned)attr_rc; }
    pthread_attr_setdetachstate(&la, detach);
    auto* ts = new (std::nothrow) ThreadStart{};
    if (!ts) { pthread_attr_destroy(&la); return 12; }         // ENOMEM (FreeBSD and host agree)
    ts->entry = entry; ts->arg = arg;
    ts->guest_stack = supplied_stack; ts->guest_stack_size = supplied_stack_size;
    ts->name_generation = next_guest_thread_name_generation();
    ts->name[0] = 0;
    if (t_pending_guest_thread_hooks) ts->hooks = *t_pending_guest_thread_hooks;
    if (a4) { strncpy(ts->name, (const char*)(uintptr_t)a4, sizeof(ts->name) - 1); ts->name[sizeof(ts->name) - 1] = 0; }
    int r = pthread_create(&tid, &la, thread_trampoline, ts);   // trampoline registers the stack first
    if (getenv("PROSPER_SYNCLOG")) fprintf(stderr, "[thread] pthread_create rc=%d\n", r);
    pthread_attr_destroy(&la);
    if (r) { delete ts; return fbsd_errno(r); }   // EAGAIN (out of threads) is 11 here, 35 on the PS5
    (void)publish_guest_thread_name((uint64_t)(uintptr_t)tid, ts->name_generation, ts->name);
    ts->name_published.store(true, std::memory_order_release);
#else
    // Windows: route through win_thread_trampoline so the guest entry is called with the SysV ABI and
    // gets its guest %fs TCB — a bare pthread_create(entry, arg) mis-passes the arg (MS x64 vs SysV) and
    // skips TLS activation, crashing the worker on a null-derived pointer.
    constexpr size_t kStackFloor = 1 * 1024 * 1024, kStackDefault = 8 * 1024 * 1024;
    size_t ssz = kStackDefault;
    void* supplied_stack = nullptr;
    size_t supplied_stack_size = 0;
    int detach = PTHREAD_CREATE_JOINABLE;
    if (a1 && *(void**)a1) {
        auto* guest_attr = (GuestPthreadAttr*)*(void**)a1;
        auto* gat = &guest_attr->host;
        size_t req = 0;
        if (pthread_attr_getstacksize(gat, &req) == 0 && req)
            ssz = req < kStackFloor ? kStackFloor : req;
        supplied_stack = guest_attr->supplied_stack;
        supplied_stack_size = guest_attr->supplied_stack_size;
        pthread_attr_getdetachstate(gat, &detach);
    }
    // Keep winpthreads on a host-owned stack; the trampoline switches guest entry to an explicit
    // Sony range after preparing the native reservation used by Windows runtime cleanup.
    pthread_attr_t la; pthread_attr_init(&la);
    pthread_attr_setstacksize(&la, supplied_stack ? kStackFloor : ssz);
    pthread_attr_setdetachstate(&la, detach);
    auto* ts = new (std::nothrow) WinThreadStart{};
    if (!ts) { pthread_attr_destroy(&la); return 12; }          // ENOMEM (FreeBSD and host agree)
    ts->entry = (uint64_t)(uintptr_t)entry; ts->arg = arg;
    ts->guest_stack = supplied_stack; ts->guest_stack_size = supplied_stack_size;
    ts->name_generation = next_guest_thread_name_generation();
    ts->name[0] = 0;
    if (t_pending_guest_thread_hooks) ts->hooks = *t_pending_guest_thread_hooks;
    if (a4) { strncpy(ts->name, (const char*)(uintptr_t)a4, sizeof(ts->name) - 1); ts->name[sizeof(ts->name) - 1] = 0; }
    int r = pthread_create(&tid, &la, win_thread_trampoline, ts);
    pthread_attr_destroy(&la);
    if (r) { delete ts; return fbsd_errno(r); }   // EAGAIN (out of threads) is 11 here, 35 on the PS5
    (void)publish_guest_thread_name((uint64_t)(uintptr_t)tid, ts->name_generation, ts->name);
    ts->name_published.store(true, std::memory_order_release);
#endif
    if (a0) *(uint64_t*)a0 = (uint64_t)tid;
    return 0;
}
// Spawn a guest-code thread through the EXACT path scePthreadCreate uses, with an explicit
// caller-supplied guest stack and optional host-side entry/exit hooks.
//
// This exists so libSceUlt's ulthreads (#1603) reuse the trampoline rather than reimplementing it.
// That trampoline carries a lot of hard-won behaviour a second copy would silently lose: the guest
// %fs TCB activation and its terminal restore before glibc's thread cleanup (#644), stack-bounds
// registration keyed by tid and its removal on exit because pthread ids recycle (#138), the
// __tls_get_addr DTV purge (#68), sigaltstack for a catchable guest stack overflow, and host thread
// naming. `guest_stack` is honoured exactly: the host pthread keeps a glibc-owned stack for its own
// static TLS and bookkeeping, and the guest entry is called on the caller's range — which is what
// _sceUltUlthreadCreate's `context`/`sizeContext` pair is.
int guest_thread_spawn(uint64_t entry, uint64_t arg, const char* name,
                       void* guest_stack, size_t guest_stack_size,
                       const GuestThreadHooks* hooks, uint64_t* out_thread) {
    GuestPthreadAttr attr{};
    pthread_attr_init(&attr.host);
    pthread_attr_setdetachstate(&attr.host, PTHREAD_CREATE_JOINABLE);
    attr.supplied_stack = guest_stack;
    attr.supplied_stack_size = guest_stack_size;
    void* attr_ptr = &attr;
    uint64_t tid = 0;
    t_pending_guest_thread_hooks = hooks;
    const uint64_t rc = k_pthread_create((uint64_t)(uintptr_t)&tid, (uint64_t)(uintptr_t)&attr_ptr,
                                         entry, arg, (uint64_t)(uintptr_t)name, 0);
    t_pending_guest_thread_hooks = nullptr;
    pthread_attr_destroy(&attr.host);
    if (rc == 0 && out_thread) *out_thread = tid;
    return (int)rc;
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
#ifdef _WIN32
    // #997: for a guest WORKER thread (entered through win_thread_trampoline), escape back to the
    // trampoline via __builtin_longjmp rather than host pthread_exit, whose SEH unwind through the
    // un-unwindable guest / prosper_call_guest_* frames raises RtlRaiseStatus and kills the process.
    // The trampoline then runs the single normal-exit cleanup path. The main guest thread is not
    // armed (it never runs through win_thread_trampoline), so it keeps the pthread_exit fallback.
    if (win_thread_exit_armed()) {
        win_thread_exit_longjmp(a0);   // does not return
    }
#endif
    tls_dtv_purge_current_thread();
    retire_guest_thread_name((uint64_t)(uintptr_t)pthread_self(), 0);
#if defined(__linux__) || defined(__APPLE__)
    unregister_thread_stack((uint64_t)pthread_self());
#elif defined(_WIN32)
    void* stack_base = nullptr; size_t stack_size = 0;
    guest_stack_for_current_thread(&stack_base, &stack_size);
    trace_guest_thread_lifecycle(false, (uint64_t)pthread_self(),
                                 (uint64_t)GetCurrentThreadId(), stack_base, stack_size);
    unregister_thread_stack((uint64_t)GetCurrentThreadId());
    unregister_thread_stack((uint64_t)pthread_self());
    unregister_current_thread_handle((uint64_t)pthread_self());
#endif
    pthread_exit((void*)(uintptr_t)a0);
    return 0;
}

// --- thread-local storage keys (IL2CPP uses these heavily) -> host pthread keys ---
#ifdef _WIN32
namespace {
std::mutex g_win_key_thunks_mutex;
std::unordered_map<uint64_t, void*> g_win_key_thunks;
std::atomic<void (*)(uint64_t)> g_win_key_delete_after_host_hook{nullptr};

// winpthreads invokes key destructors with the Microsoft x64 ABI, but every guest callback uses
// the PS5/FreeBSD SysV ABI. Emit one tiny Microsoft-ABI thunk per live key so winpthreads can retain
// its normal destructor iteration semantics while the callback value is delivered in guest RDI.
void* win_make_key_destructor_thunk(uint64_t guest_destructor) {
    auto* code = static_cast<uint8_t*>(VirtualAlloc(
        nullptr, 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!code) return nullptr;

    size_t offset = 0;
    auto byte = [&](uint8_t value) { code[offset++] = value; };
    auto qword = [&](uint64_t value) {
        std::memcpy(code + offset, &value, sizeof value);
        offset += sizeof value;
    };

    // winpthreads leaves its per-key "used" flag set after pthread_setspecific(key, nullptr),
    // then invokes the destructor with a null value at thread exit. POSIX requires callbacks only
    // for non-null values, so discard that spurious invocation before crossing into guest code.
    byte(0x48); byte(0x85); byte(0xc9);                 // test rcx, rcx
    byte(0x74); byte(0x24);                             // je final ret

    // rcx=value -> prosper_call_guest_sysv(guest_destructor, value, 0).
    byte(0x48); byte(0x89); byte(0xca);                 // mov rdx, rcx
    byte(0x48); byte(0xb9); qword(guest_destructor);    // mov rcx, guest_destructor
    byte(0x45); byte(0x31); byte(0xc0);                 // xor r8d, r8d
    byte(0x48); byte(0xb8);
    qword((uint64_t)(uintptr_t)&prosper_call_guest_sysv); // mov rax, ABI bridge
    byte(0x48); byte(0x83); byte(0xec); byte(0x28);     // shadow space + call alignment
    byte(0xff); byte(0xd0);                             // call rax
    byte(0x48); byte(0x83); byte(0xc4); byte(0x28);
    byte(0xc3);                                         // ret

    DWORD old_protect = 0;
    if (!VirtualProtect(code, 0x1000, PAGE_EXECUTE_READ, &old_protect)) {
        VirtualFree(code, 0, MEM_RELEASE);
        return nullptr;
    }
    FlushInstructionCache(GetCurrentProcess(), code, offset);
    return code;
}

}

void win_set_key_delete_after_host_hook_for_test(void (*hook)(uint64_t)) {
    g_win_key_delete_after_host_hook.store(hook, std::memory_order_release);
}

size_t win_key_destructor_thunk_count_for_test() {
    std::lock_guard<std::mutex> lock(g_win_key_thunks_mutex);
    return g_win_key_thunks.size();
}
#endif

HLE(k_key_create) {
    if (!a0) return 0x16;
    pthread_key_t k;
#ifdef _WIN32
    void* thunk = nullptr;
    int r = 0;
    {
        // Keep host key allocation and registry publication atomic with deletion. winpthreads can
        // immediately reuse a deleted numeric key, so exposing it before the old map entry is gone
        // can make the new emplace lose and leave its RX thunk permanently untracked.
        std::lock_guard<std::mutex> lock(g_win_key_thunks_mutex);
        thunk = a1 ? win_make_key_destructor_thunk(a1) : nullptr;
        if (a1 && !thunk) return ENOMEM;
        r = pthread_key_create(&k, (void (*)(void*))thunk);
        if (!r && thunk) {
            try {
                if (!g_win_key_thunks.emplace((uint64_t)k, thunk).second) r = EAGAIN;
            } catch (const std::bad_alloc&) {
                r = ENOMEM;
            }
            if (r) pthread_key_delete(k);
        }
    }
    if (r) {
        if (thunk) VirtualFree(thunk, 0, MEM_RELEASE);
        return fbsd_errno(r);
    }
#else
    int r = pthread_key_create(&k, (void (*)(void*))(uintptr_t)a1);
    if (r) return fbsd_errno(r);   // EAGAIN (keys exhausted) is 11 here, 35 on the PS5
#endif
    *(uint32_t*)(uintptr_t)a0 = (uint32_t)k;   // hand the guest our host key
    return 0;
}
HLE(k_key_delete)    {
    const pthread_key_t key = (pthread_key_t)a0;
#ifdef _WIN32
    void* thunk = nullptr;
    int result = 0;
    {
        // Do not release the host key until creation is excluded: winpthreads may recycle its
        // numeric value before the corresponding thunk entry has otherwise been erased.
        std::lock_guard<std::mutex> lock(g_win_key_thunks_mutex);
        result = pthread_key_delete(key);
        if (!result) {
            if (auto hook = g_win_key_delete_after_host_hook.load(std::memory_order_acquire))
                hook((uint64_t)key);
            auto it = g_win_key_thunks.find((uint64_t)key);
            if (it != g_win_key_thunks.end()) {
                thunk = it->second;
                g_win_key_thunks.erase(it);
            }
        }
    }
    if (thunk) VirtualFree(thunk, 0, MEM_RELEASE);
#else
    const int result = pthread_key_delete(key);
#endif
    return (uint64_t)(int64_t)result;
}
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
    void ef_destroy(EventFlag* e) {
        pthread_mutex_destroy(&e->m);
        pthread_cond_destroy(&e->c);
        interruptible_cond_forget(&e->c);
        free(e);
    }
}
HLE(k_ef_create) {   // (ef*, name, attr, initPattern, opt)
    auto* e = (EventFlag*)calloc(1, sizeof(EventFlag));
    pthread_mutex_init(&e->m, nullptr); pthread_cond_init(&e->c, nullptr); e->bits = a3;
    if (a0) *(void**)(uintptr_t)a0 = e;
    return 0;
}
HLE(k_ef_delete)  {
    auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0;
    interruptible_mutex_lock(&e->m);
    e->deleted = true;
    interruptible_cond_broadcast(&e->c);           // wake every waiter -> they return EACCES
    bool has_waiters = e->waiters > 0;
    pthread_mutex_unlock(&e->m);
    if (!has_waiters) ef_destroy(e);               // none parked -> free now; else the last waiter frees
    return 0;
}
HLE(k_ef_set)     { auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0; if (sclog()) fprintf(stderr, "[sync2] T%" PRIu64 " EF.set       ef=0x%llx bits|=0x%llx\n", sctid(), (unsigned long long)a0, (unsigned long long)a1); interruptible_mutex_lock(&e->m); e->bits |= a1; interruptible_cond_broadcast(&e->c); pthread_mutex_unlock(&e->m); return 0; }
HLE(k_ef_clear)   { auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0; interruptible_mutex_lock(&e->m); e->bits &= a1; pthread_mutex_unlock(&e->m); return 0; }
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
// scePthreadMutexTimedlock has no POSIX spelling registered, so it encodes in place. The shipped
// guest libc.prx's `_Mtx_timedlock` compares this result against 0x8002003c (encoded ETIMEDOUT) and
// maps anything else non-zero to Dinkumware `_Thrd_error`, which throws — so a plain 60 turned an
// ordinary lock timeout into an uncaught std::system_error. Same contract as the mutex lock/trylock
// aliases above (#1945).
HLE(k_mutex_timedlock) {
    auto* m = ensure_mutex(a0); if (!m) return prosper::hle::kSceKernelErrorEINVAL;
    if (guest_mutex_self_deadlock(m)) return prosper::hle::kSceKernelErrorEDEADLK;
    timespec dl = abs_deadline_us(a1);
    int rc = pthread_mutex_timedlock(m, &dl);
    if (rc == 0) guest_mutex_acquired(m);
    return sce_pthread_rc(fbsd_errno(rc));   // the table maps host ETIMEDOUT -> FreeBSD 60
}
// scePthreadCondTimedwait(cond, mutex, SceKernelUseconds usec): the Sony 3rd arg is a RELATIVE µs scalar,
// NOT an abstime pointer -> it must NOT be aliased to the POSIX k_cond_timedwait (which reads a2 as a
// timespec*, so a small µs integer there derefs a bogus pointer). Unlike pthread_cond_timedwait, the
// Sony entry point returns the encoded SCE_KERNEL_ERROR_ETIMEDOUT (0x8002003c), not the positive
// FreeBSD errno 60. Middleware commonly compares that exact value before returning from its bounded
// wait; returning 60 makes it retry the timed wait forever instead of running its periodic work.
// This one encoded only its TIMEOUT, and left EINVAL and every forwarded host failure bare — the
// same question answered two ways inside a single function (#2178). It has no POSIX spelling
// registered on its body (pthread_cond_timedwait takes the absolute form and has its own handler),
// so it encodes in place, exactly as k_mutex_timedlock above does.
HLE(k_cond_timedwait_sce) {
    auto* c = ensure_cond(a0); auto* m = ensure_mutex(a1);
    if (!c || !m) return prosper::hle::kSceKernelErrorEINVAL;
    timespec dl = abs_deadline_us(a2);
    int rc = interruptible_cond_timedwait(c, m, &dl, GuestWaitKind::ConditionSequence, 0,
                                          nullptr, kGuestMutexCondWaitBookkeeping);
    // sce_pthread_rc passes the already-encoded ETIMEDOUT through untouched, and encodes the rest.
    return sce_pthread_rc(rc == ETIMEDOUT ? prosper::hle::kSceKernelErrorETIMEDOUT : fbsd_errno(rc));
}
// scePthreadRwlockTimedrd/wrlock(rwlock, SceKernelUseconds usec): acquire with a RELATIVE µs timeout.
// Were MISSING -> the generic stub returned 0 (= "lock held") without taking the lock, so the guest ran
// its critical section unguarded (the same silent-unsync / heap-race class as k_mutex_timedlock above).
HLE(k_rwlock_timedrdlock) {
    auto* g = ensure_rwlock(a0); if (!g) return 0x16;   // EINVAL
    timespec dl = abs_deadline_us(a1);
    int rc = pthread_rwlock_timedrdlock(&g->rw, &dl);
    if (rc == 0) rw_acquired(g, false);   // record the hold so its unlock is not refused (#2012)
    // EAGAIN here means "too many concurrent readers"; unmapped it reached the guest as
    // FreeBSD EDEADLK, turning a retryable condition into a deadlock report (#1612).
    return fbsd_errno(rc);
}
HLE(k_rwlock_timedwrlock) {
    auto* g = ensure_rwlock(a0); if (!g) return 0x16;   // EINVAL
    timespec dl = abs_deadline_us(a1);
    int rc = pthread_rwlock_timedwrlock(&g->rw, &dl);
    if (rc == 0) rw_acquired(g, true);    // record the hold so its unlock is not refused (#2012)
    return fbsd_errno(rc);
}
// The POSIX spellings libScePosix exports alongside the Sony ones. They were NOT registered, so the
// dispatcher answered its default 0 — "you hold the lock" — for a lock nobody took (#2012). That is
// the silent-unsync class, and here it was also the *source* of the unmatched unlocks that corrupt
// the host rwlock: Sonic Racing: CrossWorlds calls `pthread_rwlock_trywrlock` (`XhWHn6P5R7U`, one of
// exactly 21 unimplemented NIDs in its boot), is told it succeeded, runs the critical section and
// then unlocks a lock it never held.
//
// Two argument shapes, and they must NOT share a body:
//   * scePthreadRwlockTimedrd/wrlock take a RELATIVE µs scalar   (handled above)
//   * pthread_rwlock_timedrd/wrlock  take an ABSOLUTE timespec*  (POSIX, CLOCK_REALTIME)
//   * pthread_rwlock_reltimedrd/wrlock_np take a RELATIVE timespec*
// Aliasing the first two is the same defect the k_cond_timedwait note above records.
namespace {
    // Read a guest timespec; FreeBSD and Linux agree on {time_t tv_sec; long tv_nsec} for x86-64.
    bool read_guest_timespec(uint64_t addr, timespec& out) {
        if (!addr) return false;
        const auto* ts = (const timespec*)(uintptr_t)addr;
        out.tv_sec = ts->tv_sec;
        out.tv_nsec = ts->tv_nsec;
        return true;
    }
    // A relative timeout expressed against CLOCK_REALTIME, because that is the clock the host's
    // timed rwlock waits on; a wall-clock jump therefore shifts it, which is the same exposure
    // every other abs_deadline_us() caller in this file already has.
    timespec rel_to_abs(const timespec& rel) {
        timespec now{};
        clock_gettime(CLOCK_REALTIME, &now);
        timespec dl{now.tv_sec + rel.tv_sec, now.tv_nsec + rel.tv_nsec};
        // Normalize by division, not by subtraction in a loop: an out-of-spec tv_nsec near LONG_MAX
        // would be ~9.2e9 iterations. Without normalizing at all, glibc answers EINVAL instead of
        // timing out.
        if (dl.tv_nsec >= 1000000000L) {
            dl.tv_sec += dl.tv_nsec / 1000000000L;
            dl.tv_nsec %= 1000000000L;
        }
        return dl;
    }
}
HLE(k_posix_rwlock_timedrdlock) {
    auto* g = ensure_rwlock(a0); if (!g) return 0x16;
    timespec dl{}; if (!read_guest_timespec(a1, dl)) return 0x16;
    const int rc = pthread_rwlock_timedrdlock(&g->rw, &dl);
    if (rc == 0) rw_acquired(g, false);
    return fbsd_errno(rc);
}
HLE(k_posix_rwlock_timedwrlock) {
    auto* g = ensure_rwlock(a0); if (!g) return 0x16;
    timespec dl{}; if (!read_guest_timespec(a1, dl)) return 0x16;
    const int rc = pthread_rwlock_timedwrlock(&g->rw, &dl);
    if (rc == 0) rw_acquired(g, true);
    return fbsd_errno(rc);
}
// The `_np` pair takes a RELATIVE timespec*. No title in the corpus is yet known to call these, so
// the shape follows the documented `_np` convention rather than live evidence; registering them is
// still strictly better than the dispatcher's default 0, which claims the lock was acquired.
// CONFIDENCE: MED.
HLE(k_posix_rwlock_reltimedrdlock) {
    auto* g = ensure_rwlock(a0); if (!g) return 0x16;
    timespec rel{}; if (!read_guest_timespec(a1, rel)) return 0x16;
    timespec dl = rel_to_abs(rel);
    const int rc = pthread_rwlock_timedrdlock(&g->rw, &dl);
    if (rc == 0) rw_acquired(g, false);
    return fbsd_errno(rc);
}
HLE(k_posix_rwlock_reltimedwrlock) {
    auto* g = ensure_rwlock(a0); if (!g) return 0x16;
    timespec rel{}; if (!read_guest_timespec(a1, rel)) return 0x16;
    timespec dl = rel_to_abs(rel);
    const int rc = pthread_rwlock_timedwrlock(&g->rw, &dl);
    if (rc == 0) rw_acquired(g, true);
    return fbsd_errno(rc);
}
// Sony spellings of the timed acquisitions encode their failure the libkernel way (#1984).
SCE_PTHREAD_ALIAS(k_sce_rwlock_timedrdlock, k_rwlock_timedrdlock)
SCE_PTHREAD_ALIAS(k_sce_rwlock_timedwrlock, k_rwlock_timedwrlock)
// scePthreadSem* -- POSIX-style counting semaphores (DISTINCT from sceKernelCreateSema). Were MISSING ->
// the generic stub returned 0: SemInit created nothing, SemWait returned immediately (never blocked),
// SemGetvalue left *value uninitialized -> a producer/consumer or gate built on these got NO
// synchronization (the silent-unsync / UAF class). Back them with host sem_t.
namespace { bool semlog() { static const bool on = getenv("PROSPER_SEMLOG") != nullptr; return on; } }
HLE(k_sem_init)      { if (!a0) return 0x16; auto* s = (sem_t*)calloc(1, sizeof(sem_t)); sem_init(s, 0, (unsigned)a2); *(void**)(uintptr_t)a0 = s; if (semlog()) fprintf(stderr, "[sem] init slot=%p value=%u\n", (void*)(uintptr_t)a0, (unsigned)a2); return 0; }
HLE(k_sem_wait)      { auto* s = ensure_sem(a0); if (!s) return 0x16; if (semlog()) fprintf(stderr, "[sem] wait slot=%p enter\n", (void*)(uintptr_t)a0); int rc = sem_wait(s); if (semlog()) fprintf(stderr, "[sem] wait slot=%p exit rc=%d\n", (void*)(uintptr_t)a0, rc); return rc == 0 ? 0 : fbsd_errno(errno); }
HLE(k_sem_trywait)   { auto* s = ensure_sem(a0); if (!s) return 0x16; return sem_trywait(s) == 0 ? 0 : fbsd_errno(errno); }   // EAGAIN (would block) is 11 here, 35 on the PS5
// scePthreadSemTimedwait is the one member of the family with no POSIX spelling registered on its
// body, so it encodes in place; the other six go through SCE_PTHREAD_ALIAS below (#2178).
HLE(k_sem_timedwait) { auto* s = ensure_sem(a0); if (!s) return prosper::hle::kSceKernelErrorEINVAL; timespec dl = abs_deadline_us(a1); int rc = sem_timedwait(s, &dl); return rc == 0 ? 0 : sce_pthread_rc(fbsd_errno(errno)); }
HLE(k_sem_post)      { auto* s = ensure_sem(a0); if (!s) return 0x16; int rc = sem_post(s); if (semlog()) fprintf(stderr, "[sem] post slot=%p rc=%d\n", (void*)(uintptr_t)a0, rc); return rc == 0 ? 0 : fbsd_errno(errno); }
HLE(k_sem_getvalue)  { auto* s = ensure_sem(a0); if (!s) return 0x16; int v = 0; sem_getvalue(s, &v); if (a1) *(int*)(uintptr_t)a1 = v; return 0; }
// Quarantined, not freed, and `sem_destroy` deferred to reclaim — a thread parked in `sem_wait` is
// inside this object. See the contract block above k_mutex_destroy and sync_retire.hpp (#2042).
HLE(k_sem_destroy)   { if (void* s = pt_claim_slot(a0)) retire_sync_object(s, SyncObjectKind::Sem, [](void* p) { sem_destroy((sem_t*)p); }); return 0; }   // #2176
// The Sony spellings of the fallible semaphore entry points (#2178). SemDestroy is absent because
// it returns 0 unconditionally — #2042 made it quarantine its object rather than free it, which
// changes what it does but not what it reports, so it still needs no alias. NOTE that the POSIX
// names registered on these same bodies are handled just below, along the OTHER
// axis (#2182): the return CONVENTION rather than the error ENCODING.
SCE_PTHREAD_ALIAS(k_sce_sem_init,     k_sem_init)
SCE_PTHREAD_ALIAS(k_sce_sem_wait,     k_sem_wait)
SCE_PTHREAD_ALIAS(k_sce_sem_trywait,  k_sem_trywait)
SCE_PTHREAD_ALIAS(k_sce_sem_post,     k_sem_post)
SCE_PTHREAD_ALIAS(k_sce_sem_getvalue, k_sem_getvalue)
// The same bodies under their POSIX spellings, split along the RETURN-CONVENTION axis rather than
// the encoding one (#2182). k_sem_* return the error NUMBER, which is right for scePthreadSem* and
// wrong for sem_*: C11 7.26 and FreeBSD sem_wait(3) both specify "return -1, number left in errno".
//
// The failure this fixes is quiet in both idioms. A guest writing `if (sem_wait(s) < 0)` -- the form
// the C standard prescribes -- saw SUCCESS on every failure, because a positive errno is not
// negative. One writing `if (sem_wait(s) != 0)` saw the failure but could not read it out of errno,
// because nothing had been put there.
//
// The value published is the FreeBSD-numbered errno the body already returned, NOT the host's, and
// that distinction is load-bearing: the guest is a FreeBSD-ABI binary comparing against FreeBSD
// constants, and it reads this through __error, which hle_libc.cpp resolves to the host thread's
// &errno. The two numbering schemes are not interchangeable -- EAGAIN is 35 on the PS5 and 11 here,
// which is exactly why fbsd_errno() exists (see the k_sem_trywait comment above).
//
// sem_timedwait has no POSIX spelling registered, so it is absent here and encodes in place.
// sem_destroy is absent because returning 0 unconditionally already IS its POSIX contract.
namespace {
    inline uint64_t posix_sem_rc(uint64_t fbsd_rc) {
        if (fbsd_rc == 0) return 0;
        errno = (int)(uint32_t)fbsd_rc;
        return (uint64_t)(int64_t)-1;
    }
}
#define POSIX_SEM_ALIAS(posix_name, body) HLE(posix_name) { return posix_sem_rc(body(a0, a1, a2, a3, a4, a5)); }
POSIX_SEM_ALIAS(k_posix_sem_init,     k_sem_init)
POSIX_SEM_ALIAS(k_posix_sem_wait,     k_sem_wait)
POSIX_SEM_ALIAS(k_posix_sem_trywait,  k_sem_trywait)
POSIX_SEM_ALIAS(k_posix_sem_post,     k_sem_post)
POSIX_SEM_ALIAS(k_posix_sem_getvalue, k_sem_getvalue)
#undef POSIX_SEM_ALIAS
// scePthreadBarrier* -- were MISSING -> the generic stub returned 0, so BarrierWait let every thread sail
// past a rendezvous before its peers arrived: the barrier's downstream invariant (all-arrived) is false ->
// reads of not-yet-produced data (the async-load race class). Back with host pthread_barrier_t; the serial
// thread gets -1 (PTHREAD_BARRIER_SERIAL_THREAD, FreeBSD's value), the rest 0. Barrierattr are no-ops.
// No POSIX spelling is registered on either body, so both encode in place (#2178). The serial
// thread's -1 is NOT an errno and must survive untouched: sce_pthread_rc passes it through because
// its high bits are set, which is exactly the case the pass-through guard exists for.
HLE(k_barrier_init)    { if (!a0 || a2 == 0) return prosper::hle::kSceKernelErrorEINVAL; auto* b = (pthread_barrier_t*)calloc(1, sizeof(pthread_barrier_t)); pthread_barrier_init(b, nullptr, (unsigned)a2); *(void**)(uintptr_t)a0 = b; return 0; }
HLE(k_barrier_wait)    { auto* b = ensure_barrier(a0); if (!b) return prosper::hle::kSceKernelErrorEINVAL; int rc = pthread_barrier_wait(b); return rc == PTHREAD_BARRIER_SERIAL_THREAD ? (uint64_t)(int64_t)-1 : sce_pthread_rc(fbsd_errno(rc)); }
// Quarantined, not freed (#2042) — `pthread_barrier_wait` parks every arriving thread inside the
// object, the widest window in the family. See the contract block above k_mutex_destroy.
HLE(k_barrier_destroy) { if (void* b = pt_claim_slot(a0)) retire_sync_object(b, SyncObjectKind::Barrier, [](void* p) { pthread_barrier_destroy((pthread_barrier_t*)p); }); return 0; }   // #2176
HLE(k_ef_wait)    { // (ef, pattern, waitMode, resultPat*, SceKernelUseconds* timeout)
    // The timeout arg (a4) was previously IGNORED — a bounded guest wait blocked forever (the
    // same class of silent hang root-caused for wait_on_address, hle_kernel_mem.cpp). Sony
    // semantics: *timeout is a u32 microsecond budget, updated on return with the time left;
    // expiry returns KERNEL_ERROR_ETIMEDOUT (Kyty EventFlag.cpp / Errno.h 0x8002003C).
    auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0;
    uint64_t ret = 0;
    if (sclog()) fprintf(stderr, "[sync2] T%" PRIu64 " EF.wait.ent  ef=0x%llx pat=0x%llx mode=0x%llx bits=0x%llx\n",
                         sctid(), (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                         (unsigned long long)e->bits);
    interruptible_mutex_lock(&e->m);
    e->waiters++;
    if (a4) {
        uint32_t usec = *(uint32_t*)(uintptr_t)a4;
        timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
        timespec dl = abs_deadline_us(usec);
        int rc = 0;
        while (!evf_match(e->bits, a1, (uint32_t)a2) && rc != ETIMEDOUT && !e->deleted)
            rc = interruptible_cond_timedwait(&e->c, &e->m, &dl,
                                              GuestWaitKind::EventFlag, (uintptr_t)e);
        timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        int64_t spent_i = (int64_t)(t1.tv_sec - t0.tv_sec) * 1000000
                        + ((int64_t)t1.tv_nsec - (int64_t)t0.tv_nsec) / 1000;
        uint64_t spent = spent_i < 0 ? 0u : (uint64_t)spent_i;
        *(uint32_t*)(uintptr_t)a4 = spent >= usec ? 0u : (uint32_t)(usec - spent);
        if (!e->deleted && !evf_match(e->bits, a1, (uint32_t)a2)) ret = 0x8002003Cull;  // ETIMEDOUT
    } else {
        while (!evf_match(e->bits, a1, (uint32_t)a2) && !e->deleted)
            interruptible_cond_wait(&e->c, &e->m, GuestWaitKind::EventFlag, (uintptr_t)e);
    }
    bool deleted = e->deleted;                     // deleted under us -> EACCES (Kyty EventFlag.cpp)
    if (deleted) ret = 0x8002000Dull;
    uint64_t res = e->bits;
    if (!ret) { if (a2 & 0x10) e->bits = 0; else if (a2 & 0x20) e->bits &= ~a1; }  // CLEAR_ALL / CLEAR_PAT
    bool last = (--e->waiters == 0) && deleted;    // last waiter out of a deleted flag frees it
    pthread_mutex_unlock(&e->m);
    if (last) ef_destroy(e);                        // safe: `res` was copied before the free
    if (a3) *(uint64_t*)(uintptr_t)a3 = res;
    if (sclog()) fprintf(stderr, "[sync2] T%" PRIu64 " EF.wait.exit ef=0x%llx res=0x%llx ret=0x%llx\n",
                         sctid(), (unsigned long long)a0, (unsigned long long)res, (unsigned long long)ret);
    return ret;
}
HLE(k_ef_poll)    { // (ef, pattern, waitMode, resultPat*)
    auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0;
    interruptible_mutex_lock(&e->m);
    bool ok = evf_match(e->bits, a1, (uint32_t)a2); uint64_t res = e->bits;
    if (ok) { if (a2 & 0x10) e->bits = 0; else if (a2 & 0x20) e->bits &= ~a1; }
    pthread_mutex_unlock(&e->m);
    if (a3) *(uint64_t*)(uintptr_t)a3 = res;
    // PollEventFlag reports EBUSY when the requested pattern is not currently satisfied.
    // Sonic Origins' RsdxWaitEvent checks this exact ABI value before falling back to a
    // blocking WaitEventFlag; returning 0x80020023 made every normal poll miss an assertion.
    return ok ? 0 : 0x80020010;   // SCE_KERNEL_ERROR_EBUSY
}

// --- semaphores (SceKernelSema): counting sem with wait/signal ---
// deleted/waiters: same defer-free-to-last-waiter scheme as EventFlag above (delete under a blocked
// waiter would otherwise free the mutex/condvar out from under it — UAF).
namespace {
    struct Sema { pthread_mutex_t m; pthread_cond_t c; int64_t count; bool deleted; int waiters; };
    void sema_destroy(Sema* s) {
        pthread_mutex_destroy(&s->m);
        pthread_cond_destroy(&s->c);
        interruptible_cond_forget(&s->c);
        free(s);
    }
}
HLE(k_sema_create) { // (sema*, name, attr, initCount, maxCount, opt)
    auto* s = (Sema*)calloc(1, sizeof(Sema));
    pthread_mutex_init(&s->m, nullptr); pthread_cond_init(&s->c, nullptr); s->count = (int64_t)(int32_t)a3;
    if (a0) *(void**)(uintptr_t)a0 = s;
    if (sclog_semaphore()) fprintf(stderr, "[sync2] T%" PRIu64 " SEMA.create  sema=0x%llx name='%s' init=%lld max=%lld\n",
                         sctid(), (unsigned long long)(uintptr_t)s, a1 ? (const char*)(uintptr_t)a1 : "",
                         (long long)(int32_t)a3, (long long)(int32_t)a4);
    return 0;
}
HLE(k_sema_delete) {
    auto* s = (Sema*)(uintptr_t)a0; if (!s) return 0;
    uint64_t focused = a0;
    g_sclog_focus_sema.compare_exchange_strong(focused, 0, std::memory_order_acq_rel,
                                                std::memory_order_acquire);
    interruptible_mutex_lock(&s->m);
    s->deleted = true;
    interruptible_cond_broadcast(&s->c);
    bool has_waiters = s->waiters > 0;
    pthread_mutex_unlock(&s->m);
    if (!has_waiters) sema_destroy(s);
    return 0;
}
HLE(k_sema_wait)   { // (sema, need, SceKernelUseconds* timeout) — timeout honored like k_ef_wait
    auto* s = (Sema*)(uintptr_t)a0; if (!s) return 0; int64_t need = a1 ? (int64_t)a1 : 1;
    const bool log_wait = sclog_semaphore();
    if (log_wait && sclog_pthread_filter())
        g_sclog_focus_sema.store(a0, std::memory_order_release);
    if (log_wait) fprintf(stderr, "[sync2] T%" PRIu64 " SEMA.wait     sema=0x%llx need=%lld\n", sctid(), (unsigned long long)a0, (long long)need);
    uint64_t ret = 0;
    interruptible_mutex_lock(&s->m);
    s->waiters++;
    if (a2) {
        uint32_t usec = *(uint32_t*)(uintptr_t)a2;
        timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
        timespec dl = abs_deadline_us(usec);
        int rc = 0;
        while (s->count < need && rc != ETIMEDOUT && !s->deleted)
            rc = interruptible_cond_timedwait(&s->c, &s->m, &dl,
                                              GuestWaitKind::Semaphore, (uintptr_t)s);
        timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        int64_t spent_i = (int64_t)(t1.tv_sec - t0.tv_sec) * 1000000
                        + ((int64_t)t1.tv_nsec - (int64_t)t0.tv_nsec) / 1000;
        uint64_t spent = spent_i < 0 ? 0u : (uint64_t)spent_i;
        *(uint32_t*)(uintptr_t)a2 = spent >= usec ? 0u : (uint32_t)(usec - spent);
        if (!s->deleted && s->count < need) ret = 0x8002003Cull;    // KERNEL_ERROR_ETIMEDOUT
    } else {
        while (s->count < need && !s->deleted)
            interruptible_cond_wait(&s->c, &s->m, GuestWaitKind::Semaphore, (uintptr_t)s);
    }
    bool deleted = s->deleted;                       // deleted under us -> EACCES
    if (deleted) ret = 0x8002000Dull;
    if (!ret) s->count -= need;
    bool last = (--s->waiters == 0) && deleted;
    pthread_mutex_unlock(&s->m);
    if (last) sema_destroy(s);
    return ret; }
HLE(k_sema_signal) { auto* s = (Sema*)(uintptr_t)a0; if (!s) return 0; int64_t n = a1 ? (int64_t)a1 : 1;
    if (sclog_semaphore() || sclog_focused_semaphore(a0))
        fprintf(stderr, "[sync2] T%" PRIu64 " SEMA.signal   sema=0x%llx n=%lld\n", sctid(), (unsigned long long)a0, (long long)n);
    interruptible_mutex_lock(&s->m); s->count += n; interruptible_cond_broadcast(&s->c); pthread_mutex_unlock(&s->m); return 0; }
HLE(k_sema_poll)   { auto* s = (Sema*)(uintptr_t)a0; if (!s) return 0; int64_t need = a1 ? (int64_t)a1 : 1;
    interruptible_mutex_lock(&s->m); bool ok = s->count >= need; if (ok) s->count -= need; pthread_mutex_unlock(&s->m); return ok ? 0 : 0x80020023; }

// ---- C++ exception unwinding: sceKernelGetModuleInfoForUnwind (libkernel RpQJJVKTiFM) ----
// The guest's libunwind calls this per code address to locate that module's .eh_frame_hdr for stack
// unwinding. It was an unimplemented stub returning 0 with the caller's struct left uninitialised, so
// libunwind read a garbage eh_frame_hdr address ("Unsupported .eh_frame_hdr version" → stack smash) and
// ANY thrown C++ exception was fatal. Now we fill the real info from the loaded modules' program headers.
namespace {
    std::vector<UnwindModuleDesc> g_unwind_mods;
    std::mutex g_unwind_mx;   // guards g_unwind_mods: set_unwind_modules (assign, may realloc) can run on a
                              // runtime module load while a worker is unwinding here (#344).
    // Real linked-module handles are stable by load order. The unwind descriptors and per-module
    // export tables are built from that same order, so address lookup can publish the handle expected
    // by sceKernelGetModuleInfoFromAddr and later consume it in sceKernelDlsym.
    constexpr uint64_t kModuleHandleBase = kSceModuleHandleBase;
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
// PS5 SDK 3.20 ModuleInfo: size@0, opaque info[32]@8, int32 module handle@0x108. Native
// libraries use this to turn a return/program address into a handle for module-local lookup.
// Returning success without writing the handle sends later dlsym calls to an arbitrary module.
HLE(k_get_module_info_from_addr) {   // (VAddr addr, int n, ModuleInfo* info)
    uint8_t* info = (uint8_t*)(uintptr_t)a2;
    if (!info || (int)a1 != 2) return 0x80020016;      // SCE_KERNEL_ERROR_EINVAL
    int32_t handle = 0;
    {
        std::lock_guard<std::mutex> lk(g_unwind_mx);
        for (size_t i = 0; i < g_unwind_mods.size(); ++i) {
            const auto& d = g_unwind_mods[i];
            if (a0 >= d.lo && a0 < d.hi) {
                handle = (int32_t)(kModuleHandleBase + i);
                break;
            }
        }
    }
    *(int32_t*)(info + 0x108) = handle;
    return handle ? 0 : (uint64_t)-1;
}
void set_unwind_modules(const UnwindModuleDesc* d, size_t c) {
    std::lock_guard<std::mutex> lk(g_unwind_mx); g_unwind_mods.assign(d, d + c);
}
// #639: a runtime-loaded module appends here. The index it lands on is the same index the module's
// export table lands on in g_mod_exports below, because k_get_module_info_from_addr derives a
// handle from THIS vector while sceKernelDlsym consumes it against THAT one — the two must stay
// parallel or an address-derived handle names a different module than it resolves through. The
// runtime loader appends to both, once per module, and checks the two indices agree.
size_t add_unwind_module(const UnwindModuleDesc& d) {
    std::lock_guard<std::mutex> lk(g_unwind_mx);
    g_unwind_mods.push_back(d);
    return g_unwind_mods.size() - 1;
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
            // snprintf returns the length it WOULD have written, so handing it to write() reads
            // past the end of this stack buffer whenever the line truncates -- #2050's class, and
            // instrument trap 109's. raw_fmt_len() is the clamp #2050 introduced for exactly this,
            // already unit-tested (tests/test_raw_syscall.cpp), and it encodes the right choice for
            // the truncated case: write the cap-1 bytes that really landed, so the shortening stays
            // visible rather than being papered over.
            //
            // Integers only, so this one cannot truncate today -- clamped anyway, because "cannot
            // truncate today" is a property of the format string, and the next person to add a %s
            // will not re-derive it.
            char b[96]; int n = snprintf(b, sizeof b, "[exc2] DROPPED type=%d tid=%ld (no handler)\n",
                                         type, (long)prosper_gettid());
            (void)!write(2, b, prosper::raw_fmt_len(n, sizeof b));
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
        // #1659: was a hand-rolled three-branch dispatcher whose eboot (0x400000000) and libc
        // (0x500000000) bases were both stale; only the il2cpp one was current. One shared,
        // module-aware call replaces all three and cannot drift again.
        // This is the reachable one of the three (#2162): the format embeds guest_module_name(),
        // which is NOT length-bounded at the call site, into a 160-byte buffer whose fixed part is
        // ~50. A module label over ~90 characters truncates and the unclamped write over-reads --
        // the same shape that made the [labelhist] site the first one caught in trap 109.
        n = snprintf(b, sizeof b, "[exc2] ENTER tid=%ld type=%d fs=%s rip=%s+0x%llx\n",
                     (long)prosper_gettid(), type, fss, prosper::guest_module_name(irip),
                     (unsigned long long)prosper::guest_module_offset(irip));
        // guest_module_name/offset already degrade to "mapped/host" + the verbatim address, so the
        // former host: fallback branch is folded into the single call above.
        (void)!write(2, b, prosper::raw_fmt_len(n, sizeof b));
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
        (void)!write(2, b, prosper::raw_fmt_len(n, sizeof b));
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
struct WinExcStackSlot {
    std::atomic<uint64_t> thread{0};
    std::atomic<uintptr_t> base{0};
    std::atomic<uint32_t> active{0};
    std::mutex raise_mutex;
};
constexpr DWORD kWinExcContextFlags = CONTEXT_ALL | CONTEXT_XSTATE;
constexpr size_t kWinExcContextBufferSize = 64 * 1024;
struct WinExcDelivery {
    alignas(64) std::array<uint8_t, kWinExcContextBufferSize> context_storage{};
    PCONTEXT saved = nullptr;
    uint64_t type = 0;
    uint64_t handler = 0;
};
constexpr size_t kWinExcStackSize = 256 * 1024;
std::array<WinExcStackSlot, 1024> g_win_exc_stacks;
std::atomic<uint32_t> g_win_exc_active_busy{0};
std::atomic<uint32_t> g_win_exc_suspended_busy{0};
enum class WinExcTraceKind : uint32_t {
    Redirect = 1, Enter = 2, Exit = 3, Reject = 4,
    StackQuery = 5, ThreadStart = 6, ThreadExit = 7,
    CoopQueue = 8, CoopEnter = 9, CoopExit = 10,
};
struct WinExcTraceEvent {
    std::atomic<uint64_t> published{0};
    uint64_t sequence = 0;
    WinExcTraceKind kind = WinExcTraceKind::Redirect;
    uint32_t windows_tid = 0;
    uint64_t target = 0;
    uint64_t rip = 0;
    uint64_t rsp = 0;
    uint64_t extra = 0;
};
std::array<WinExcTraceEvent, 1024> g_win_exc_trace;
std::atomic<uint64_t> g_win_exc_trace_sequence{0};
bool g_win_exc_trace_enabled = true;

enum class WinCoopState : uint32_t { Idle, Publishing, Queued, Running };
struct WinCoopSlot {
    std::atomic<uint64_t> thread{0};
    std::atomic<WinCoopState> state{WinCoopState::Idle};
    std::atomic<uint64_t> type{0};
    std::atomic<uintptr_t> stack_base{0};
    alignas(16) CONTEXT captured{};
    alignas(16) std::array<uint8_t, 0x400> guest_context{};
};
std::array<WinCoopSlot, 1024> g_win_coop_slots;
std::atomic<uint32_t> g_win_coop_queued{0};

struct WinGuestThreadSlot {
    std::atomic<uint64_t> publication{0};
    uint64_t next_publication = 0; // written only while publication holds the exclusive sentinel
    std::atomic<uintptr_t> thread_handle{0};
    std::atomic<uint32_t> native_id{0};
    std::atomic<uint64_t> pthread_id{0};
    std::atomic<uintptr_t> stack_base{0};
    std::atomic<size_t> stack_size{0};
};
std::array<WinGuestThreadSlot, 1024> g_win_guest_threads;
constexpr uint64_t kWinGuestThreadSlotPublishing = UINT64_MAX;
std::atomic<GuestThreadTraceTestHook> g_guest_thread_trace_test_hook{nullptr};
std::atomic<void*> g_guest_thread_trace_test_opaque{nullptr};

bool win_guest_thread_snapshot(const WinGuestThreadSlot& slot, GuestThreadSnapshot& snapshot,
                               uint64_t* generation = nullptr,
                               HANDLE* registered_handle = nullptr) {
    if (generation) *generation = 0;
    if (registered_handle) *registered_handle = nullptr;
    const uint64_t publication = slot.publication.load(std::memory_order_acquire);
    if (publication == 0 || publication == kWinGuestThreadSlotPublishing) return false;
    const HANDLE handle = (HANDLE)slot.thread_handle.load(std::memory_order_relaxed);
    const uint32_t native_id = slot.native_id.load(std::memory_order_relaxed);
    const uint64_t pthread_id = slot.pthread_id.load(std::memory_order_relaxed);
    const uintptr_t stack_base = slot.stack_base.load(std::memory_order_relaxed);
    const size_t stack_size = slot.stack_size.load(std::memory_order_relaxed);
    if (slot.publication.load(std::memory_order_acquire) != publication) return false;
    snapshot = {native_id, pthread_id, stack_base, stack_size};
    if (generation) *generation = publication;
    if (registered_handle) *registered_handle = handle;
    return true;
}

void win_guest_thread_register(uint64_t pthread_id, uint32_t native_id,
                               void* stack_base, size_t stack_size) {
    if (native_id == 0) return;
    HANDLE thread_handle = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                         &thread_handle, 0, FALSE, DUPLICATE_SAME_ACCESS))
        return;
    for (WinGuestThreadSlot& slot : g_win_guest_threads) {
        GuestThreadSnapshot current{};
        if (win_guest_thread_snapshot(slot, current) && current.native_id == native_id &&
            current.pthread_id == pthread_id) {
            CloseHandle(thread_handle);
            return; // idempotent lifecycle notification
        }
        uint64_t publication = 0;
        if (!slot.publication.compare_exchange_strong(
                publication, kWinGuestThreadSlotPublishing, std::memory_order_acq_rel))
            continue;
        // A unique publication token validates the complete payload and detects unregister/re-register
        // ABA even when Windows promptly reuses the same native thread id.
        slot.thread_handle.store((uintptr_t)thread_handle, std::memory_order_relaxed);
        slot.native_id.store(native_id, std::memory_order_relaxed);
        slot.pthread_id.store(pthread_id, std::memory_order_relaxed);
        slot.stack_base.store((uintptr_t)stack_base, std::memory_order_relaxed);
        slot.stack_size.store(stack_size, std::memory_order_relaxed);
        uint64_t token;
        do {
            token = ++slot.next_publication;
        } while (token == 0 || token == kWinGuestThreadSlotPublishing);
        slot.publication.store(token, std::memory_order_release);
        return;
    }
    CloseHandle(thread_handle);
}

void win_guest_thread_unregister(uint32_t native_id) {
    for (WinGuestThreadSlot& slot : g_win_guest_threads) {
        uint64_t publication = slot.publication.load(std::memory_order_acquire);
        if (publication == 0 || publication == kWinGuestThreadSlotPublishing ||
            slot.native_id.load(std::memory_order_relaxed) != native_id ||
            slot.publication.load(std::memory_order_acquire) != publication ||
            !slot.publication.compare_exchange_strong(
                publication, kWinGuestThreadSlotPublishing, std::memory_order_acq_rel))
            continue;
        HANDLE thread_handle = (HANDLE)slot.thread_handle.exchange(0, std::memory_order_relaxed);
        slot.native_id.store(0, std::memory_order_relaxed);
        slot.pthread_id.store(0, std::memory_order_relaxed);
        slot.stack_base.store(0, std::memory_order_relaxed);
        slot.stack_size.store(0, std::memory_order_relaxed);
        slot.publication.store(0, std::memory_order_release);
        if (thread_handle) CloseHandle(thread_handle);
        return;
    }
}

void win_exc_trace(WinExcTraceKind kind, uint64_t target, uint32_t windows_tid,
                   uint64_t rip, uint64_t rsp, uint64_t extra = 0);

WinCoopSlot* win_coop_slot_for(uint64_t thread, bool create) {
    for (;;) {
        bool initializing = false;
        for (WinCoopSlot& slot : g_win_coop_slots) {
            uint64_t owner = slot.thread.load(std::memory_order_acquire);
            if (owner == thread) {
                if (slot.stack_base.load(std::memory_order_acquire)) return &slot;
                initializing = true;
                continue;
            }
            if (!create || owner != 0 || !slot.thread.compare_exchange_strong(
                    owner, thread, std::memory_order_acq_rel))
                continue;

            void* base = VirtualAlloc(nullptr, kWinExcStackSize,
                                      MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (!base) {
                slot.thread.store(0, std::memory_order_release);
                return nullptr;
            }
            slot.stack_base.store((uintptr_t)base, std::memory_order_release);
            return &slot;
        }
        if (initializing) {
            SwitchToThread();
            continue;
        }
        return nullptr;
    }
}

enum class WinCoopQueueResult { Fallback, Queued, Busy };
WinCoopQueueResult try_queue_wait_exception(uint64_t target, uint64_t type,
                                            HANDLE target_handle) {
    if (getenv("PROSPER_WIN_LEGACY_EXC")) return WinCoopQueueResult::Fallback;
    WinCoopSlot* slot = win_coop_slot_for(target, true);
    if (!slot) return WinCoopQueueResult::Fallback;
    WinCoopState expected = WinCoopState::Idle;
    if (!slot->state.compare_exchange_strong(expected, WinCoopState::Publishing,
                                              std::memory_order_acq_rel))
        return WinCoopQueueResult::Busy;
    slot->type.store(type, std::memory_order_relaxed);
    slot->state.store(WinCoopState::Queued, std::memory_order_release);
    g_win_coop_queued.fetch_add(1, std::memory_order_release);
    const bool interrupted = interrupt_guest_wait(target);
    win_exc_trace(WinExcTraceKind::CoopQueue, target,
                  GetThreadId(target_handle),
                  type, 0, interrupted ? 1 : 0);

    // The target claims Queued at the wrapper checkpoint before executing the handler. A bounded
    // handoff closes the race where it left the wait just before our wake. A request that remains
    // unclaimed cannot be reported as delivered: withdraw it atomically so the caller can use forced
    // CONTEXT delivery. The target's Queued->Running claim races this Queued->Idle withdrawal; exactly
    // one side wins and therefore exactly one side decrements the global queued count.
    for (unsigned spin = 0; spin < 20; ++spin) {
        if (slot->state.load(std::memory_order_acquire) != WinCoopState::Queued)
            return WinCoopQueueResult::Queued;
        Sleep(1);
    }
    expected = WinCoopState::Queued;
    if (slot->state.compare_exchange_strong(expected, WinCoopState::Idle,
                                            std::memory_order_acq_rel)) {
        g_win_coop_queued.fetch_sub(1, std::memory_order_acq_rel);
        return WinCoopQueueResult::Fallback;
    }
    return WinCoopQueueResult::Queued;
}

void win_exc_trace(WinExcTraceKind kind, uint64_t target, uint32_t windows_tid,
                   uint64_t rip, uint64_t rsp, uint64_t extra) {
    if (!g_win_exc_trace_enabled) return;
    const uint64_t sequence = g_win_exc_trace_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    WinExcTraceEvent& event = g_win_exc_trace[(sequence - 1) % g_win_exc_trace.size()];
    event.published.store(0, std::memory_order_relaxed);
    event.sequence = sequence;
    event.kind = kind;
    event.windows_tid = windows_tid;
    event.target = target;
    event.rip = rip;
    event.rsp = rsp;
    event.extra = extra;
    event.published.store(sequence, std::memory_order_release);
}

bool win_init_xstate_context(void* storage, size_t storage_size, PCONTEXT* context) {
    DWORD length = static_cast<DWORD>(storage_size);
    if (!InitializeContext(storage, kWinExcContextFlags, context, &length)) return false;
    return SetXStateFeaturesMask(*context, GetEnabledXStateFeatures()) != FALSE;
}

// #2139: sceKernelRaiseException reports ESRCH for a target that is demonstrably alive (Blue Prince:
// 45/45 refusals target one thread while 739/739 raises at every other thread succeed). Three
// distinct branches return that one code, so the log cannot say which. Name them.
std::atomic<uint32_t> g_win_exc_esrch_no_handle{0};
std::atomic<uint32_t> g_win_exc_esrch_exited{0};
std::atomic<uint32_t> g_win_exc_esrch_exited_late{0};

void win_exc_log_rejected(const char* reason, std::atomic<uint32_t>& counter,
                          uint64_t target, uint64_t type, DWORD suspend_count = 0) {
    const uint32_t occurrence = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    if (occurrence > 16) return;
    win_exc_trace(WinExcTraceKind::Reject, target, GetCurrentThreadId(), type, suspend_count);
    char message[192];
    const int length = std::snprintf(
        message, sizeof(message),
        "[exc] Windows delivery rejected reason=%s count=%u target=0x%llx type=0x%llx suspend=%lu\n",
        reason, occurrence, (unsigned long long)target, (unsigned long long)type,
        (unsigned long)suspend_count);
    if (length <= 0) return;
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), message,
              static_cast<DWORD>(std::min<int>(length, sizeof(message) - 1)), &written, nullptr);
}

WinExcStackSlot* win_exc_stack_for(uint64_t thread) {
    for (;;) {
        bool initializing = false;
        for (WinExcStackSlot& slot : g_win_exc_stacks) {
            uint64_t owner = slot.thread.load(std::memory_order_acquire);
            if (owner == thread) {
                const uintptr_t base = slot.base.load(std::memory_order_acquire);
                if (base) return &slot;
                initializing = true;
                continue;
            }
            if (owner != 0 || !slot.thread.compare_exchange_strong(
                    owner, thread, std::memory_order_acq_rel))
                continue;

            void* base = VirtualAlloc(nullptr, kWinExcStackSize,
                                      MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (!base) {
                slot.thread.store(0, std::memory_order_release);
                return nullptr;
            }
            slot.base.store((uintptr_t)base, std::memory_order_release);
            return &slot;
        }
        // Another raiser may be publishing this target's slot. It completes without depending on
        // the target thread, so a scheduler handoff is sufficient and avoids duplicate stacks.
        if (initializing) {
            SwitchToThread();
            continue;
        }
        return nullptr;
    }
}
using RtlRestoreContextFn = VOID (WINAPI*)(PCONTEXT, PEXCEPTION_RECORD);
RtlRestoreContextFn g_rtl_restore_context = nullptr;

extern "C" uint64_t prosper_call_guest_sysv(uint64_t fn, uint64_t a0, uint64_t a1);
extern "C" uint64_t prosper_call_guest_on_stack(uintptr_t stack_top, uint64_t fn,
                                                  uint64_t a0, uint64_t a1);
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

// Cooperative delivery must still use an alternate stack. The guest TLS allocation can sit only
// a few hundred bytes below its normal RSP, so placing the 0x400-byte context and GC handler frames
// there corrupts TLS even though no Windows CONTEXT redirection is involved. Unlike the injected
// path, this helper returns to the wait wrapper normally after the guest handler is released.
__asm__(
    ".text\n"
    ".p2align 4\n"
    ".globl prosper_call_guest_on_stack\n"
    "prosper_call_guest_on_stack:\n"
    "    movq %rcx, %r10\n"
    "    movq %rdx, %rcx\n"
    "    movq %r8, %rdx\n"
    "    movq %r9, %r8\n"
    "    pushq %r12\n"
    "    movq %rsp, %r12\n"
    "    movq %r10, %rsp\n"
    "    andq $-16, %rsp\n"
    "    subq $32, %rsp\n"
    "    call prosper_call_guest_sysv\n"
    "    movq %r12, %rsp\n"
    "    popq %r12\n"
    "    ret\n"
);

void win_exc_log(const char* msg) {
    if (!g_exc_log && !g_exc_log2) return;
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), msg, (DWORD)strlen(msg), &written, nullptr);
}

// The guest-visible result is a FreeBSD errno, so the Win32 status that produced it is not
// recoverable from the return value (#1612). Keep the raw code where a developer can still see it —
// it is the part that actually names what the host refused to do.
static uint64_t win_exc_sce_error(const char* where, DWORD win32_error) {
    if (g_exc_log || g_exc_log2) {
        char msg[160];
        std::snprintf(msg, sizeof(msg), "[exc] %s failed: Win32 %lu -> 0x%llx\n", where,
                      (unsigned long)win32_error,
                      (unsigned long long)prosper::hle::sce_error_from_win32(win32_error));
        win_exc_log(msg);
    }
    return prosper::hle::sce_error_from_win32(win32_error);
}

extern "C" __attribute__((noinline, noreturn))
void prosper_win_exc_delivery(WinExcDelivery* delivery) {
    PCONTEXT captured = delivery->saved;
    uint64_t type = delivery->type;
    uint64_t handler = delivery->handler;
    alignas(64) std::array<uint8_t, kWinExcContextBufferSize> restore_storage{};
    PCONTEXT saved = nullptr;
    if (!win_init_xstate_context(restore_storage.data(), restore_storage.size(), &saved) ||
        !CopyContext(saved, kWinExcContextFlags, captured)) {
        TerminateProcess(GetCurrentProcess(), ERROR_INVALID_STATE);
    }
    // FreeBSD amd64 mcontext layout; keep in lockstep with the POSIX signal path above.
    alignas(16) uint8_t ctx[0x400]{};
    auto WQ = [&](int off, uint64_t v) { *(uint64_t*)(ctx + off) = v; };
    WQ(0x08, saved->Rdi); WQ(0x10, saved->Rsi); WQ(0x18, saved->Rdx); WQ(0x20, saved->Rcx);
    WQ(0x28, saved->R8);  WQ(0x30, saved->R9);  WQ(0x38, saved->Rax); WQ(0x40, saved->Rbx);
    WQ(0x48, saved->Rbp); WQ(0x50, saved->R10); WQ(0x58, saved->R11); WQ(0x60, saved->R12);
    WQ(0x68, saved->R13); WQ(0x70, saved->R14); WQ(0x78, saved->R15);
    WQ(0xA0, saved->Rip); WQ(0xB8, saved->Rsp); WQ(0xF8, saved->Rsp);

    win_exc_trace(WinExcTraceKind::Enter, 0, GetCurrentThreadId(), saved->Rip, saved->Rsp);
    win_exc_log("[exc] handler ENTER on Windows target\n");
    prosper_call_guest_sysv(handler, type, (uint64_t)(uintptr_t)ctx);
    win_exc_log("[exc] handler EXIT on Windows target (resumed)\n");
    win_exc_trace(WinExcTraceKind::Exit, 0, GetCurrentThreadId(), saved->Rip, saved->Rsp);

    VirtualFree(delivery, 0, MEM_RELEASE);
    // Keep the slot active through RtlRestoreContext. Clearing it here lets another raiser reuse
    // this stack while the target is still executing on it. The next raiser reclaims the slot only
    // after suspending the target and proving that RSP has left the dedicated exception stack.
    g_rtl_restore_context(saved, nullptr);
    TerminateProcess(GetCurrentProcess(), ERROR_INVALID_STATE);   // RtlRestoreContext never returns
    __builtin_unreachable();
}

uint64_t win_raise_exception(uint64_t target, uint64_t type) {
    if (!target || type >= 128 || !g_exc_handlers[type]) return 0;
    // FreeBSD numbering, not this host's: 0x...23 (35) is EAGAIN on the PS5, and EAGAIN is a RETRY
    // hint, so a guest told "try again" for "you raised an exception on yourself" can spin on a
    // condition that can never change. 0x...26 (38) is likewise ENOTSOCK, not ENOSYS (#1612).
    if (pthread_equal((pthread_t)target, pthread_self()))
        return prosper::hle::kSceKernelErrorEDEADLK;   // FreeBSD EDEADLK = 11
    if (!g_rtl_restore_context)
        return prosper::hle::kSceKernelErrorENOSYS;    // FreeBSD ENOSYS = 78

    // Hold a real duplicate for the complete queue/fallback transaction. This both validates the
    // pthread before publishing cooperative state and keeps a detached target's kernel object alive
    // while delivery is in flight.
    HANDLE thread = duplicate_registered_thread_handle(target);
    if (!thread) {
        HANDLE borrowed = (HANDLE)pthread_gethandle((pthread_t)target);
        if (borrowed && borrowed != INVALID_HANDLE_VALUE)
            DuplicateHandle(GetCurrentProcess(), borrowed, GetCurrentProcess(), &thread,
                            0, FALSE, DUPLICATE_SAME_ACCESS);
    }
    struct OwnedHandle { HANDLE value; ~OwnedHandle() { if (value) CloseHandle(value); } }
        owned{thread};
    if (!thread || thread == INVALID_HANDLE_VALUE) {
        // Neither the trampoline registry nor winpthreads could name this thread. A guest thread
        // that never ran through win_thread_trampoline is registry-invisible by construction.
        win_exc_log_rejected("esrch-no-handle", g_win_exc_esrch_no_handle, target, type);
        return 0x80020003ull;                                                     // ESRCH
    }
    DWORD exit_code = 0;
    if (!GetExitCodeThread(thread, &exit_code) || exit_code != STILL_ACTIVE) {
        // A handle was found but reports a dead thread -- e.g. a stale registry entry left by an
        // abnormal exit, reachable again once pthread ids recycle (#138).
        win_exc_log_rejected("esrch-exited", g_win_exc_esrch_exited, target, type,
                             (DWORD)exit_code);
        return 0x80020003ull;                                                     // ESRCH
    }

    const WinCoopQueueResult cooperative = try_queue_wait_exception(target, type, thread);
    if (cooperative == WinCoopQueueResult::Queued) return 0;
    if (cooperative == WinCoopQueueResult::Busy) return 0x80020010ull;

    // The target may have exited during the bounded cooperative handoff. Do not allocate a delivery
    // record or turn a failed SuspendThread into an opaque Win32 error for a dead pthread.
    if (!GetExitCodeThread(thread, &exit_code) || exit_code != STILL_ACTIVE) {
        win_exc_log_rejected("esrch-exited-late", g_win_exc_esrch_exited_late, target, type,
                             (DWORD)exit_code);
        return 0x80020003ull;                                                     // ESRCH
    }

    // Do not use the process heap for delivery records. The target can be interrupted while it owns
    // that heap's lock, and freeing a record from its injected handler would then deadlock before the
    // original context gets a chance to release it.
    auto* delivery = (WinExcDelivery*)VirtualAlloc(nullptr, sizeof(WinExcDelivery),
                                                   MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!delivery) return 0x8002000cull;                                      // ENOMEM
    std::memset(delivery, 0, sizeof(*delivery));
    delivery->type = type;
    delivery->handler = g_exc_handlers[type];
    WinExcStackSlot* stack_slot = win_exc_stack_for(target);
    if (!stack_slot) {
        VirtualFree(delivery, 0, MEM_RELEASE);
        return 0x8002000cull;
    }
    std::lock_guard<std::mutex> raise_lock(stack_slot->raise_mutex);
    const uintptr_t exception_stack = stack_slot->base.load(std::memory_order_acquire);

    CONTEXT captured{};
    bool target_suspended = false;
    if (stack_slot->active.load(std::memory_order_acquire)) {
        DWORD previous_suspend = SuspendThread(thread);
        if (previous_suspend == (DWORD)-1) {
            DWORD e = GetLastError();
            VirtualFree(delivery, 0, MEM_RELEASE);
            return win_exc_sce_error("SuspendThread", e);
        }
        if (previous_suspend != 0) {
            ResumeThread(thread);
            win_exc_log_rejected("already-suspended", g_win_exc_suspended_busy,
                                 target, type, previous_suspend);
            VirtualFree(delivery, 0, MEM_RELEASE);
            return 0x80020010ull;
        }
        target_suspended = true;
        captured.ContextFlags = CONTEXT_CONTROL;
        if (!GetThreadContext(thread, &captured)) {
            DWORD e = GetLastError();
            ResumeThread(thread);
            VirtualFree(delivery, 0, MEM_RELEASE);
            return win_exc_sce_error("GetThreadContext", e);
        }
        const uintptr_t rsp = static_cast<uintptr_t>(captured.Rsp);
        if (rsp >= exception_stack && rsp < exception_stack + kWinExcStackSize) {
            ResumeThread(thread);
            win_exc_log_rejected("active", g_win_exc_active_busy, target, type);
            VirtualFree(delivery, 0, MEM_RELEASE);
            return 0x80020010ull;
        }
        // The prior handler completed and restored ordinary guest execution. Its active bit was
        // intentionally retained to protect the exception stack during RtlRestoreContext; this
        // suspended context proves the stack can now be reused for the new delivery.
    } else {
        stack_slot->active.store(1, std::memory_order_release);
    }

    DWORD previous_suspend = target_suspended ? 0 : SuspendThread(thread);
    if (previous_suspend == (DWORD)-1) { DWORD e = GetLastError(); stack_slot->active.store(0, std::memory_order_release); VirtualFree(delivery, 0, MEM_RELEASE); return win_exc_sce_error("raise-exception host call", e); }
    if (previous_suspend != 0) {
        ResumeThread(thread);
        win_exc_log_rejected("already-suspended", g_win_exc_suspended_busy,
                             target, type, previous_suspend);
        stack_slot->active.store(0, std::memory_order_release);
        VirtualFree(delivery, 0, MEM_RELEASE);
        return 0x80020010ull;                                                 // EBUSY
    }

    if (!win_init_xstate_context(delivery->context_storage.data(),
                                 delivery->context_storage.size(), &delivery->saved)) {
        ResumeThread(thread);
        stack_slot->active.store(0, std::memory_order_release);
        VirtualFree(delivery, 0, MEM_RELEASE);
        return 0x8002000cull;
    }
    if (!GetThreadContext(thread, delivery->saved)) {
        DWORD e = GetLastError(); ResumeThread(thread); stack_slot->active.store(0, std::memory_order_release); VirtualFree(delivery, 0, MEM_RELEASE);
        return win_exc_sce_error("raise-exception host call", e);
    }
    win_exc_trace(WinExcTraceKind::Redirect, target, GetThreadId(thread),
                  delivery->saved->Rip, delivery->saved->Rsp);

    CONTEXT injected = *delivery->saved;
    injected.ContextFlags = CONTEXT_FULL;
    injected.Rip = (DWORD64)(uintptr_t)&prosper_win_exc_thunk;
    // Match the POSIX SA_ONSTACK path: Unity's guest TLS can sit only a few hundred bytes below the
    // live guest RSP, so even respecting the 128-byte red zone is insufficient for the 0x400-byte
    // exception context and the guest GC handler's call frames.
    injected.Rsp = ((exception_stack + kWinExcStackSize) & ~(DWORD64)0xf) - 8;
    DWORD64 delivery_ptr = (DWORD64)(uintptr_t)delivery;
    SIZE_T written = 0;
    if (!WriteProcessMemory(GetCurrentProcess(), (void*)(uintptr_t)injected.Rsp,
                            &delivery_ptr, sizeof(delivery_ptr), &written) ||
        written != sizeof(delivery_ptr)) {
        DWORD e = GetLastError(); ResumeThread(thread); stack_slot->active.store(0, std::memory_order_release); VirtualFree(delivery, 0, MEM_RELEASE);
        return win_exc_sce_error("raise-exception host call", e);
    }
    if (!SetThreadContext(thread, &injected)) {
        DWORD e = GetLastError(); ResumeThread(thread); stack_slot->active.store(0, std::memory_order_release); VirtualFree(delivery, 0, MEM_RELEASE);
        return win_exc_sce_error("raise-exception host call", e);
    }
    // Wake while the target is still suspended. Its registration is stable until resume, and the
    // wait will be ready to return as soon as Windows restores the redirected context. This avoids
    // racing the target's wait-unregistration and the lifetime of the condition object.
    const bool interrupted = interrupt_guest_wait(target);
    if (ResumeThread(thread) == (DWORD)-1) {
        DWORD e = GetLastError();
        bool restored = SetThreadContext(thread, delivery->saved) != FALSE;
        if (restored && ResumeThread(thread) != (DWORD)-1) {
            stack_slot->active.store(0, std::memory_order_release);
            VirtualFree(delivery, 0, MEM_RELEASE);
        }
        return win_exc_sce_error("raise-exception host call", e);
    }
    // A redirected Windows CONTEXT is not dispatched until a blocking syscall returns. Current
    // IL2CPP targets commonly sit in a condition-variable or WaitOnAddress HLE; waking its registered
    // object lets the target enter prosper_win_exc_thunk immediately after it is resumed.
    if (g_exc_log2)
        win_exc_log(interrupted ? "[exc2] Windows wait-interrupted=1\n"
                                : "[exc2] Windows wait-interrupted=0\n");
    return 0;
}

void ensure_exc_sig() {
    if (g_rtl_restore_context) return;
    g_exc_log = getenv("PROSPER_SYNCLOG") != nullptr;
    g_exc_log2 = getenv("PROSPER_EXCLOG") != nullptr;
    g_win_exc_trace_enabled = getenv("PROSPER_EXC_RING_DISABLE") == nullptr;
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    g_rtl_restore_context = ntdll
        ? (RtlRestoreContextFn)GetProcAddress(ntdll, "RtlRestoreContext") : nullptr;
}
#else
void ensure_exc_sig() {}
#endif
} // namespace

void dump_guest_exception_trace() {
#ifdef _WIN32
    const uint64_t end = g_win_exc_trace_sequence.load(std::memory_order_acquire);
    const uint64_t begin = end > 128 ? end - 127 : 1;
    std::fprintf(stderr, "[exc-trace] recent events begin=%llu end=%llu queued=%u\n",
                 (unsigned long long)begin, (unsigned long long)end,
                 g_win_coop_queued.load(std::memory_order_acquire));
    for (uint64_t sequence = begin; sequence <= end; sequence++) {
        const WinExcTraceEvent& event = g_win_exc_trace[(sequence - 1) % g_win_exc_trace.size()];
        if (event.published.load(std::memory_order_acquire) != sequence) continue;
        const char* kind = event.kind == WinExcTraceKind::Redirect ? "redirect"
                         : event.kind == WinExcTraceKind::Enter ? "enter"
                         : event.kind == WinExcTraceKind::Exit ? "exit"
                         : event.kind == WinExcTraceKind::Reject ? "reject"
                         : event.kind == WinExcTraceKind::StackQuery ? "stack-query"
                         : event.kind == WinExcTraceKind::ThreadStart ? "thread-start"
                         : event.kind == WinExcTraceKind::ThreadExit ? "thread-exit"
                         : event.kind == WinExcTraceKind::CoopQueue ? "coop-queue"
                         : event.kind == WinExcTraceKind::CoopEnter ? "coop-enter"
                         : "coop-exit";
        std::fprintf(stderr,
                     "[exc-trace] seq=%llu kind=%s tid=%lu target=0x%llx "
                     "a=0x%llx b=0x%llx extra=0x%llx\n",
                     (unsigned long long)sequence, kind, (unsigned long)event.windows_tid,
                     (unsigned long long)event.target, (unsigned long long)event.rip,
                     (unsigned long long)event.rsp, (unsigned long long)event.extra);
    }
#endif
}

void dump_guest_thread_trace(const char* path, uint64_t pthread_filter) {
#ifdef _WIN32
    HANDLE output = GetStdHandle(STD_ERROR_HANDLE);
    bool close_output = false;
    if (path && *path) {
        HANDLE file = CreateFileA(path, FILE_APPEND_DATA,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                  nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file != INVALID_HANDLE_VALUE) {
            output = file;
            close_output = true;
        }
    }
    auto trace = [output]<typename... Args>(const char* format, Args... args) {
        char message[512];
        const int length = std::snprintf(message, sizeof(message), format, args...);
        if (length <= 0) return;
        DWORD written = 0;
        const DWORD bytes = static_cast<DWORD>(
            length < (int)sizeof(message) ? length : (int)sizeof(message) - 1);
        WriteFile(output, message, bytes, &written, nullptr);
    };
    std::vector<UnwindModuleDesc> modules;
    {
        std::lock_guard<std::mutex> lock(g_unwind_mx);
        modules = g_unwind_mods;
    }
    unsigned live = 0;
    for (const WinGuestThreadSlot& slot : g_win_guest_threads) {
        GuestThreadSnapshot registered{};
        live += win_guest_thread_snapshot(slot, registered);
    }
    size_t condition_slots_used = 0;
    size_t condition_slots_capacity = 0;
    snapshot_guest_wait_registry(condition_slots_used, condition_slots_capacity);
    trace("[thread-trace] live guest threads=%u condition-slots=%zu/%zu\n", live,
          condition_slots_used, condition_slots_capacity);
    for (const WinGuestThreadSlot& slot : g_win_guest_threads) {
        GuestThreadSnapshot registered{};
        uint64_t registration_generation = 0;
        HANDLE registered_handle = nullptr;
        if (!win_guest_thread_snapshot(
                slot, registered, &registration_generation, &registered_handle))
            continue;
        const uint32_t native_id = registered.native_id;
        const uint64_t pthread_id = registered.pthread_id;
        if (pthread_filter && pthread_id != pthread_filter) continue;
        const uintptr_t stack_base = registered.stack_base;
        const size_t stack_size = registered.stack_size;
        HANDLE thread = nullptr;
        DWORD pin_error = registered_handle ? ERROR_SUCCESS : ERROR_INVALID_HANDLE;
        if (registered_handle &&
            !DuplicateHandle(GetCurrentProcess(), registered_handle, GetCurrentProcess(),
                             &thread, 0, FALSE, DUPLICATE_SAME_ACCESS))
            pin_error = GetLastError();
        if (thread && GetThreadId(thread) != native_id)
            pin_error = ERROR_INVALID_THREAD_ID;
        if (pin_error == ERROR_SUCCESS && thread) {
            GuestThreadTraceTestHook hook =
                g_guest_thread_trace_test_hook.load(std::memory_order_acquire);
            if (hook)
                hook(native_id, g_guest_thread_trace_test_opaque.load(std::memory_order_acquire));
        }
        if (thread &&
            slot.publication.load(std::memory_order_acquire) != registration_generation)
            pin_error = ERROR_RETRY;
        if (pin_error != ERROR_SUCCESS || !thread) {
            if (thread) CloseHandle(thread);
            trace("[thread-trace] tid=%lu pthread=0x%llx unavailable error=%lu\n",
                  (unsigned long)native_id, (unsigned long long)pthread_id,
                  (unsigned long)pin_error);
            continue;
        }
        const DWORD prior_suspend = SuspendThread(thread);
        if (prior_suspend != (DWORD)-1 &&
            slot.publication.load(std::memory_order_acquire) != registration_generation) {
            ResumeThread(thread);
            CloseHandle(thread);
            trace("[thread-trace] tid=%lu pthread=0x%llx registration-retired\n",
                  (unsigned long)native_id, (unsigned long long)pthread_id);
            continue;
        }
        CONTEXT context{};
        context.ContextFlags = CONTEXT_CONTROL;
        const bool captured = prior_suspend != (DWORD)-1 && GetThreadContext(thread, &context);
        const DWORD capture_error = captured ? ERROR_SUCCESS : GetLastError();
        std::array<uint64_t, 2048> stack_words{};
        SIZE_T stack_bytes = 0;
        if (captured && context.Rsp >= stack_base &&
            context.Rsp - stack_base < stack_size) {
            const size_t available = stack_size - ((uintptr_t)context.Rsp - stack_base);
            const size_t wanted = std::min(available, sizeof(stack_words));
            ReadProcessMemory(GetCurrentProcess(), (void*)(uintptr_t)context.Rsp,
                              stack_words.data(), wanted, &stack_bytes);
        }
        std::array<GuestWaitSnapshot, 16> captured_waits{};
        const size_t captured_wait_count = captured
            ? snapshot_guest_waits(native_id, captured_waits.data(), captured_waits.size())
            : 0;
        if (prior_suspend != (DWORD)-1) ResumeThread(thread);
        CloseHandle(thread);
        if (!captured) {
            trace("[thread-trace] tid=%lu pthread=0x%llx capture-failed error=%lu\n",
                  (unsigned long)native_id, (unsigned long long)pthread_id,
                  (unsigned long)capture_error);
            continue;
        }
        const uint64_t rip = context.Rip;
        const char* module_name = "host";
        char guest_module_name[32]{};
        uint64_t offset = rip;
        for (size_t module_index = 0; module_index < modules.size(); ++module_index) {
            const UnwindModuleDesc& guest_module = modules[module_index];
            if (rip < guest_module.lo || rip >= guest_module.hi) continue;
            std::snprintf(guest_module_name, sizeof(guest_module_name), "guest[%zu]", module_index);
            module_name = guest_module_name;
            offset = rip - guest_module.lo;
            break;
        }
        char guest_returns[224] = "-";
        size_t guest_returns_used = 0;
        unsigned guest_return_count = 0;
        for (size_t i = 0; i < stack_bytes / sizeof(uint64_t) && guest_return_count < 8; ++i) {
            const uint64_t candidate = stack_words[i];
            // #1659: module range, not the pre-#825 literal.
            if (!prosper::guest_va_in_module_code(candidate)) continue;
            bool in_guest_module = false;
            for (const UnwindModuleDesc& guest_module : modules)
                in_guest_module |= candidate >= guest_module.lo && candidate < guest_module.hi;
            if (!in_guest_module || !guest_trace_page_executable((uintptr_t)candidate)) continue;
            bool duplicate = false;
            for (size_t j = 0; j < i; ++j)
                duplicate |= stack_words[j] == candidate;
            if (duplicate) continue;
            const int appended = std::snprintf(
                guest_returns + guest_returns_used,
                sizeof(guest_returns) - guest_returns_used,
                guest_return_count ? ",0x%llx" : "0x%llx",
                (unsigned long long)candidate);
            if (appended <= 0 || (size_t)appended >= sizeof(guest_returns) - guest_returns_used)
                break;
            guest_returns_used += (size_t)appended;
            ++guest_return_count;
        }
        // Host-side companion to the guest scan above: name the PROSPER frames on the same stack.
        // A sampled rip inside a system DLL (getenv, memcpy, a wait) says WHAT the thread is doing
        // and nothing about WHY; the prosper frame below it is the answer, and without this the
        // reader is left inferring it from the guest frame, which is often several calls away.
        char host_returns[288] = "-";
        size_t host_returns_used = 0;
        unsigned host_return_count = 0;
        for (size_t i = 0; i < stack_bytes / sizeof(uint64_t) && host_return_count < 6; ++i) {
            const uint64_t candidate = stack_words[i];
            if (candidate < 0x10000) continue;
            if (!guest_trace_page_executable((uintptr_t)candidate)) continue;
            const std::string described = describe_code_address(candidate);
            if (described.rfind("prosper+", 0) != 0) continue;   // ours only; DLLs are noise here
            bool duplicate = false;
            for (size_t j = 0; j < i; ++j) duplicate |= stack_words[j] == candidate;
            if (duplicate) continue;
            const int appended = std::snprintf(
                host_returns + host_returns_used, sizeof(host_returns) - host_returns_used,
                host_return_count ? ",%s" : "%s", described.c_str());
            if (appended <= 0 || (size_t)appended >= sizeof(host_returns) - host_returns_used) break;
            host_returns_used += (size_t)appended;
            ++host_return_count;
        }
        char wait_description[256] = "-";
        if (captured_wait_count) {
            wait_description[0] = 0;
            size_t used = 0;
            const size_t stored_wait_count = std::min(captured_wait_count, captured_waits.size());
            for (size_t wait_index = 0; wait_index < stored_wait_count; ++wait_index) {
                const GuestWaitSnapshot& wait = captured_waits[wait_index];
                const char* kind = wait.kind == GuestWaitKind::Address ? "address" :
                                   wait.kind == GuestWaitKind::ConditionSequence ? "condition" :
                                   wait.kind == GuestWaitKind::EventFlag ? "event-flag" :
                                   wait.kind == GuestWaitKind::Semaphore ? "semaphore" :
                                   "unknown";
                const int appended = wait.source
                    ? std::snprintf(wait_description + used, sizeof(wait_description) - used,
                                    wait_index ? ";%s@0x%llx(source=0x%llx)" :
                                                 "%s@0x%llx(source=0x%llx)",
                                    kind, (unsigned long long)wait.object,
                                    (unsigned long long)wait.source)
                    : std::snprintf(wait_description + used, sizeof(wait_description) - used,
                                    wait_index ? ";%s@0x%llx" : "%s@0x%llx", kind,
                                    (unsigned long long)wait.object);
                if (appended <= 0 || (size_t)appended >= sizeof(wait_description) - used) break;
                used += (size_t)appended;
            }
            if (captured_wait_count > captured_waits.size() && used < sizeof(wait_description))
                std::snprintf(wait_description + used, sizeof(wait_description) - used,
                              ";+%zu-more", captured_wait_count - captured_waits.size());
        }
        trace("[thread-trace] tid=%lu pthread=0x%llx rip=%s "
              "raw=0x%llx rsp=0x%llx suspend=%lu waits=%s guest-stack=%s host-stack=%s\n",
              (unsigned long)native_id, (unsigned long long)pthread_id,
              describe_code_address(rip).c_str(), (unsigned long long)rip,
              (unsigned long long)context.Rsp, (unsigned long)prior_suspend,
              wait_description, guest_returns, host_returns);
    }
    if (close_output) CloseHandle(output);
#else
    (void)path;
    (void)pthread_filter;
#endif
}

bool snapshot_guest_thread_registration(uint32_t native_id, GuestThreadSnapshot& snapshot) {
    snapshot = {};
#ifdef _WIN32
    for (const WinGuestThreadSlot& slot : g_win_guest_threads) {
        GuestThreadSnapshot candidate{};
        if (win_guest_thread_snapshot(slot, candidate) && candidate.native_id == native_id) {
            snapshot = candidate;
            return true;
        }
    }
#else
    (void)native_id;
#endif
    return false;
}

bool guest_trace_page_executable(uintptr_t address) {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION memory{};
    if (!address || VirtualQuery((const void*)address, &memory, sizeof(memory)) != sizeof(memory) ||
        memory.State != MEM_COMMIT || (memory.Protect & (PAGE_GUARD | PAGE_NOACCESS)))
        return false;
    switch (memory.Protect & 0xffu) {
        case PAGE_EXECUTE:
        case PAGE_EXECUTE_READ:
        case PAGE_EXECUTE_READWRITE:
        case PAGE_EXECUTE_WRITECOPY:
            return true;
        default:
            return false;
    }
#else
    (void)address;
    return false;
#endif
}

void set_guest_thread_trace_test_hook(GuestThreadTraceTestHook hook, void* opaque) {
#ifdef _WIN32
    g_guest_thread_trace_test_opaque.store(opaque, std::memory_order_release);
    g_guest_thread_trace_test_hook.store(hook, std::memory_order_release);
#else
    (void)hook;
    (void)opaque;
#endif
}

void dispatch_pending_guest_exception() {
#ifdef _WIN32
    if (g_win_coop_queued.load(std::memory_order_acquire) == 0) return;
    const uint64_t self = (uint64_t)pthread_self();
    WinCoopSlot* slot = win_coop_slot_for(self, false);
    if (!slot) return;
    WinCoopState expected = WinCoopState::Queued;
    if (!slot->state.compare_exchange_strong(expected, WinCoopState::Running,
                                              std::memory_order_acq_rel))
        return;
    g_win_coop_queued.fetch_sub(1, std::memory_order_acq_rel);

    const uint64_t type = slot->type.load(std::memory_order_relaxed);
    const uint64_t handler = type < 128 ? g_exc_handlers[type] : 0;
    if (!handler) {
        slot->state.store(WinCoopState::Idle, std::memory_order_release);
        return;
    }

    CONTEXT& captured = slot->captured;
    RtlCaptureContext(&captured);
    auto& ctx = slot->guest_context;
    ctx.fill(0);
    auto WQ = [&](int off, uint64_t value) { *(uint64_t*)(ctx.data() + off) = value; };
    WQ(0x08, captured.Rdi); WQ(0x10, captured.Rsi); WQ(0x18, captured.Rdx);
    WQ(0x20, captured.Rcx); WQ(0x28, captured.R8);  WQ(0x30, captured.R9);
    WQ(0x38, captured.Rax); WQ(0x40, captured.Rbx); WQ(0x48, captured.Rbp);
    WQ(0x50, captured.R10); WQ(0x58, captured.R11); WQ(0x60, captured.R12);
    WQ(0x68, captured.R13); WQ(0x70, captured.R14); WQ(0x78, captured.R15);
    WQ(0xA0, captured.Rip); WQ(0xB8, captured.Rsp); WQ(0xF8, captured.Rsp);

    win_exc_trace(WinExcTraceKind::CoopEnter, self, GetCurrentThreadId(),
                  captured.Rip, captured.Rsp);
    const uintptr_t stack_top = slot->stack_base.load(std::memory_order_acquire)
                              + kWinExcStackSize;
    prosper_call_guest_on_stack(stack_top, handler, type,
                                (uint64_t)(uintptr_t)ctx.data());
    win_exc_trace(WinExcTraceKind::CoopExit, self, GetCurrentThreadId(),
                  captured.Rip, captured.Rsp);
    slot->state.store(WinCoopState::Idle, std::memory_order_release);
#endif
}

void register_guest_execution_thread_handle() {
#ifdef _WIN32
    register_current_thread_handle((uint64_t)pthread_self());
#endif
}

uint32_t pending_guest_exception_count() {
#ifdef _WIN32
    return g_win_coop_queued.load(std::memory_order_acquire);
#else
    return 0;
#endif
}

void trace_guest_stack_query(uint64_t target, bool found, void* base, size_t size) {
#ifdef _WIN32
    win_exc_trace(WinExcTraceKind::StackQuery, target, GetCurrentThreadId(),
                  (uint64_t)(uintptr_t)base, (uint64_t)size, found ? 1 : 0);
#else
    (void)target; (void)found; (void)base; (void)size;
#endif
}

void trace_guest_thread_lifecycle(bool starting, uint64_t pthread_id, uint64_t native_id,
                                  void* stack_base, size_t stack_size) {
#ifdef _WIN32
    if (starting)
        win_guest_thread_register(pthread_id, (uint32_t)native_id, stack_base, stack_size);
    else
        win_guest_thread_unregister((uint32_t)native_id);
    win_exc_trace(starting ? WinExcTraceKind::ThreadStart : WinExcTraceKind::ThreadExit,
                  pthread_id, (uint32_t)native_id, (uint64_t)(uintptr_t)stack_base,
                  (uint64_t)stack_size);
#else
    (void)starting; (void)pthread_id; (void)native_id; (void)stack_base; (void)stack_size;
#endif
}

HLE(k_install_exc_handler) {   // (exceptionType, handler, ...)
    ensure_exc_sig();
    if (a0 < 128) g_exc_handlers[a0] = a1;
    if (g_exc_log)
        fprintf(stderr, "[exc] install type=0x%llx handler=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1);
    return 0;
}
HLE(k_raise_exception) {       // (targetThread /*host pthread_t*/, exceptionType, arg)
    ensure_exc_sig();
    if (g_exc_counter) (*g_exc_counter)++;
    if (g_exc_log)
        fprintf(stderr, "[exc] T%" PRIu64 " raise target=0x%llx type=0x%llx\n",
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
        // qr is a HOST errno from pthread_sigqueue/pthread_kill; the guest reads FreeBSD numbering,
        // where EAGAIN and EDEADLK are swapped relative to Linux (#1612).
        if (qr != 0) return prosper::hle::sce_error_from_host_errno(qr);   // FAILED — report, don't lie
    } else if (g_exc_log2) {
        fprintf(stderr, "[exc2] RAISE-NOOP by tid=%ld target=0x%llx type=0x%llx (no handler)\n",
                sctid(), (unsigned long long)a0, (unsigned long long)a1);
    }
#elif defined(_WIN32)
    {
        uint64_t ret = win_raise_exception(a0, a1);
        if (g_exc_log2) {
            char msg[192];
            snprintf(msg, sizeof msg,
                     "[exc2] RAISE by tid=%ld self=0x%llx target=0x%llx type=0x%llx ret=0x%llx\n",
                     sctid(), (unsigned long long)(uint64_t)pthread_self(),
                     (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)ret);
            win_exc_log(msg); // raw Win32 stderr write: target may be suspended while owning FILE's lock
        }
        return ret;
    }
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
    // Guards g_mod_exports only. Before #639 the vector was written once, before the guest ran;
    // real runtime PRX loading appends to it from a guest thread while other guest threads are
    // inside sceKernelDlsym / sceKernelLoadStartModule. The registered maps themselves are never
    // mutated after publication, so the lock covers the vector and the lookup that borrows into it.
    std::mutex g_mod_exports_mx;
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
void set_module_export_tables(std::vector<ModuleExportTable> tables) {
    std::lock_guard<std::mutex> lk(g_mod_exports_mx);
    g_mod_exports = std::move(tables);
}
uint64_t add_module_export_table(std::string path,
                                 const std::unordered_map<std::string, uint64_t>* nids) {
    std::lock_guard<std::mutex> lk(g_mod_exports_mx);
    g_mod_exports.push_back({ std::move(path), nids });
    return kModuleHandleBase + (g_mod_exports.size() - 1);
}
uint64_t module_handle_for_path(const char* path) {
    if (!path || !*path) return 0;
    const char* want = path_basename(path);
    std::lock_guard<std::mutex> lk(g_mod_exports_mx);
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
    const std::string want = name ? nid_hash(name) : std::string();
    if (name) {
        uint64_t hit = 0;
        {
            std::lock_guard<std::mutex> lk(g_mod_exports_mx);
            if (a0 >= kModuleHandleBase && a0 - kModuleHandleBase < g_mod_exports.size())
                if (const auto* nids = g_mod_exports[a0 - kModuleHandleBase].nids) {
                    auto it = nids->find(want);
                    if (it != nids->end()) hit = it->second;
                }
        }
        if (hit) {
            if (a2) *(uint64_t*)(uintptr_t)a2 = hit;
            if (getenv("PROSPER_SYNCLOG"))
                fprintf(stderr, "[dlsym] '%s' (module handle 0x%llx) -> 0x%llx\n",
                        name, (unsigned long long)a0, (unsigned long long)hit);
            return 0;
        }
    }
    if (name && g_exports) {
        auto it = g_exports->find(want);
        if (it != g_exports->end()) {
            if (a2) *(uint64_t*)(uintptr_t)a2 = it->second;   // *funcAddr = export
            if (getenv("PROSPER_SYNCLOG"))
                fprintf(stderr, "[dlsym] '%s' -> 0x%llx\n", name, (unsigned long long)it->second);
            return 0;
        }
    }
    // #639: g_exports covers only the pre-linked program, so a name-only dlsym (no handle, or a
    // handle the caller never obtained from LoadStartModule) would miss a module the guest loaded
    // at runtime. Scan the registered per-module tables in registration order — for a pre-linked
    // module this can only reproduce what g_exports already answered above, so the pre-linked
    // first-definition-wins order is preserved and only runtime modules are added, last.
    if (name) {
        uint64_t hit = 0;
        std::string from;   // COPY, not c_str(): a concurrent add_module_export_table can realloc
        {                   // g_mod_exports and free the string buffer the moment the lock drops.
            std::lock_guard<std::mutex> lk(g_mod_exports_mx);
            for (const auto& me : g_mod_exports) {
                if (!me.nids) continue;
                auto it = me.nids->find(want);
                if (it != me.nids->end()) { hit = it->second; from = me.path; break; }
            }
        }
        if (hit) {
            if (a2) *(uint64_t*)(uintptr_t)a2 = hit;
            if (getenv("PROSPER_SYNCLOG"))
                fprintf(stderr, "[dlsym] '%s' -> 0x%llx (runtime-loaded %s)\n", name,
                        (unsigned long long)hit, from.c_str());
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
    // Sony spellings report failure in the libkernel-encoded form; the POSIX aliases further down
    // keep the bare errno. See sce_pthread_rc() above for the guest-side evidence (#1945). The rule
    // is "alias IFF the body can fail": a k_sce_* name below means the body has a failure path, and
    // a raw k_* name means it returns 0 unconditionally and an alias would state a contract it does
    // not have. #2178 swept the family for spellings on the wrong side of that.
    R("scePthreadMutexattrInit", k_sce_mutexattr_init);
    R("scePthreadMutexattrSettype", k_sce_mutexattr_settype);
    R("scePthreadMutexattrGettype", k_mutexattr_gettype);   // encodes in place: no POSIX spelling
    R("scePthreadMutexattrSetprotocol", k_mutexattr_setprotocol);   // always 0
    R("scePthreadMutexattrSetpshared", k_mutexattr_setpshared);     // always 0
    R("scePthreadMutexattrDestroy", k_mutexattr_destroy);           // always 0
    R("scePthreadMutexInit", k_sce_mutex_init);
    R("scePthreadMutexDestroy", k_mutex_destroy);                   // always 0
    R("scePthreadMutexLock", k_sce_mutex_lock);
    R("scePthreadMutexTrylock", k_sce_mutex_trylock);
    R("scePthreadMutexTimedlock", k_mutex_timedlock);   // was MISSING -> faked "locked" without locking
    R("scePthreadMutexUnlock", k_sce_mutex_unlock);
    R("scePthreadCondattrInit", k_sce_condattr_init);
    R("scePthreadCondattrDestroy", k_condattr_destroy);             // always 0
    R("scePthreadCondattrSetclock", k_sce_condattr_setclock);
    R("scePthreadCondattrGetclock", k_sce_condattr_getclock);
    R("scePthreadCondattrSetpshared", k_sce_condattr_setpshared);
    R("scePthreadCondattrGetpshared", k_sce_condattr_getpshared);
    R("scePthreadCondInit", k_sce_cond_init);
    R("scePthreadCondDestroy", k_cond_destroy);                     // always 0
    R("scePthreadCondSignal", k_cond_signal);
    R("scePthreadCondBroadcast", k_cond_broadcast);
    R("scePthreadCondWait", k_cond_wait);
    R("scePthreadCondTimedwait", k_cond_timedwait_sce);   // Sony: relative µs, NOT the POSIX abstime form
    // read/write locks + once (Sony + POSIX names) — real host primitives (thread-safety fix).
    R("scePthreadRwlockInit", k_sce_rwlock_init);    R("pthread_rwlock_init", k_rwlock_init);
    R("scePthreadRwlockDestroy", k_rwlock_destroy);  R("pthread_rwlock_destroy", k_rwlock_destroy);
    R("scePthreadRwlockRdlock", k_sce_rwlock_rdlock); R("pthread_rwlock_rdlock", k_rwlock_rdlock);
    R("scePthreadRwlockWrlock", k_sce_rwlock_wrlock); R("pthread_rwlock_wrlock", k_rwlock_wrlock);
    R("scePthreadRwlockUnlock", k_sce_rwlock_unlock); R("pthread_rwlock_unlock", k_rwlock_unlock);
    R("scePthreadRwlockTryrdlock", k_sce_rwlock_tryrdlock);
    R("scePthreadRwlockTrywrlock", k_sce_rwlock_trywrlock);
    // The POSIX spellings libScePosix exports. Unregistered they fell to the dispatcher's default 0,
    // i.e. "the lock is yours" for a lock nobody took (#2012).
    R("pthread_rwlock_tryrdlock", k_rwlock_tryrdlock);
    R("pthread_rwlock_trywrlock", k_rwlock_trywrlock);
    R("pthread_rwlock_timedrdlock", k_posix_rwlock_timedrdlock);      // ABSOLUTE timespec*
    R("pthread_rwlock_timedwrlock", k_posix_rwlock_timedwrlock);      // ABSOLUTE timespec*
    R("pthread_rwlock_reltimedrdlock_np", k_posix_rwlock_reltimedrdlock);   // RELATIVE timespec*
    R("pthread_rwlock_reltimedwrlock_np", k_posix_rwlock_reltimedwrlock);   // RELATIVE timespec*
    R("scePthreadRwlockTimedrdlock", k_sce_rwlock_timedrdlock);   // were MISSING -> faked "locked" without locking
    R("scePthreadRwlockTimedwrlock", k_sce_rwlock_timedwrlock);
    R("scePthreadRwlockattrInit", k_sce_rwlockattr_init);
    R("scePthreadRwlockattrDestroy", k_sce_rwlockattr_destroy);
    R("pthread_rwlockattr_init", k_rwlockattr_init);
    R("pthread_rwlockattr_destroy", k_rwlockattr_destroy);
    // scePthreadSem* counting semaphores (were MISSING -> SemWait never blocked)
    R("scePthreadSemInit", k_sce_sem_init);       R("scePthreadSemDestroy", k_sem_destroy);   // Destroy: always 0
    R("scePthreadSemWait", k_sce_sem_wait);       R("scePthreadSemTrywait", k_sce_sem_trywait);
    R("scePthreadSemTimedwait", k_sem_timedwait); R("scePthreadSemPost", k_sce_sem_post);     // Timedwait: encodes in place
    R("scePthreadSemGetvalue", k_sce_sem_getvalue);
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
    R("scePthreadAttrSetstack", k_sce_attr_setstack);            // encoded (#2183)
    R("scePthreadAttrSetstacksize", k_sce_attr_setstacksize);    // encoded (#2183)
    R("scePthreadAttrSetguardsize", k_attr_setguardsize);
    R("scePthreadAttrSetinheritsched", k_attr_noop);
    R("scePthreadAttrSetschedpolicy", k_attr_noop);
    R("scePthreadAttrSetschedparam", k_log_attr_setschedparam);
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
    R("scePthreadSetschedparam", k_log_setschedparam);  R("scePthreadSetprio", k_log_setprio);
    R("scePthreadGetprio", k_getprio);
    R("scePthreadGetname", k_pthread_getname);
    R("scePthreadRename", k_pthread_rename);
    R("scePthreadSetName", k_pthread_rename);
    R("pthread_getname_np", k_pthread_getname);
    R("pthread_rename_np", k_pthread_rename);
    R("pthread_set_name_np", k_pthread_rename);
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
    R("pthread_condattr_setclock", k_condattr_setclock); R("pthread_condattr_getclock", k_condattr_getclock);
    R("pthread_condattr_setpshared", k_condattr_setpshared); R("pthread_condattr_getpshared", k_condattr_getpshared);
    R("pthread_attr_init", k_attr_init);      R("pthread_attr_destroy", k_attr_destroy);
    R("pthread_attr_setstack", k_attr_setstack);
    R("pthread_attr_setstacksize", k_attr_setstacksize);
    R("pthread_attr_setguardsize", k_attr_setguardsize);
    R("pthread_attr_setdetachstate", k_attr_setdetachstate); R("pthread_attr_getdetachstate", k_attr_getdetachstate);
    R("pthread_attr_setinheritsched", k_attr_noop);
    R("pthread_attr_setschedpolicy", k_attr_noop);  R("pthread_attr_setschedparam", k_attr_noop);
    R("pthread_attr_getstacksize", k_attr_getstacksize);
    R("scePthreadAttrSetaffinity", k_attr_noop); R("pthread_setname_np", k_attr_noop);
    // Plain libScePosix sem_* imports use the same guest object representation as scePthreadSem*.
    // Registering only the Sony-prefixed spellings left successful no-op imports in native engines.
    // POSIX spellings take the -1/errno wrappers (#2182). sem_destroy keeps the shared body:
    // returning 0 unconditionally is already its POSIX contract.
    R("sem_init", k_posix_sem_init);       R("sem_destroy", k_sem_destroy);
    R("sem_wait", k_posix_sem_wait);       R("sem_trywait", k_posix_sem_trywait);
    R("sem_post", k_posix_sem_post);       R("sem_getvalue", k_posix_sem_getvalue);
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
    Hle::register_fn("f7KBOafysXo", (HleFn)k_get_module_info_from_addr, "sceKernelGetModuleInfoFromAddr");
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
    Hle::register_fn("g0VTBxfJyu0", (HleFn)k_attr_noop, "sceKernelGetCurrentCpu");
    Hle::register_fn("Tz4RNUCBbGI", (HleFn)k_attr_noop, "_sceKernelRtldThreadAtexitIncrement");
    Hle::register_fn("jh+8XiK4LeE", (HleFn)k_attr_noop, "sceKernelIsAddressSanitizerEnabled");
    Hle::register_fn("El+cQ20DynU", (HleFn)k_attr_setguardsize, "scePthreadAttrSetguardsize");
    #undef R
    register_kernel_mem_hle();    // virtual/direct memory
    register_kernel_time_hle();   // time/clock + C11 threads + stubs
}

} // namespace prosper
