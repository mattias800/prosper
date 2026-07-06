// hle_kernel.cpp — HLE of libkernel primitives. Threading/sync are backed by host
// pthreads (guest ABI == host SysV ABI). Sony pthread types are opaque pointer
// handles: scePthreadMutexInit(&handle, &attr, name) allocates the object and stores
// the pointer through the caller's handle slot, returning 0 on success.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE   // pthread_getattr_np
#endif
#include "dispatch.hpp"
#include "nid.hpp"
#include "../host/exec_image.hpp"
#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#ifdef __linux__
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <signal.h>
#include <ucontext.h>
#endif

namespace prosper {
namespace { bool sclog() { static int v = getenv("PROSPER_SYNCLOG") ? 1 : 0; return v; }
    long sctid() {
#ifdef __linux__
        return (long)syscall(SYS_gettid);
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

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)

// --- mutex attributes (opaque; we back with host pthread_mutexattr_t) ---
HLE(k_mutexattr_init) {
    if (!a0) return 0x16; // EINVAL-ish
    auto* at = (pthread_mutexattr_t*)calloc(1, sizeof(pthread_mutexattr_t));
    pthread_mutexattr_init(at);
    *(void**)a0 = at;                     // store handle through caller's slot
    return 0;
}
HLE(k_mutexattr_settype)     { return 0; } // type/protocol/pshared ignored for now
HLE(k_mutexattr_setprotocol) { return 0; }
HLE(k_mutexattr_setpshared)  { return 0; }
HLE(k_mutexattr_destroy)     { if (a0 && *(void**)a0) { free(*(void**)a0); *(void**)a0 = nullptr; } return 0; }

// --- mutexes ---
HLE(k_mutex_init) {
    if (!a0) return 0x16;
    auto* m = (pthread_mutex_t*)calloc(1, sizeof(pthread_mutex_t));
    // Default to recursive: game code often locks re-entrantly and Sony's default differs.
    pthread_mutexattr_t at; pthread_mutexattr_init(&at);
    pthread_mutexattr_settype(&at, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(m, &at);
    pthread_mutexattr_destroy(&at);
    *(void**)a0 = m;
    return 0;
}
HLE(k_mutex_destroy) { if (a0 && *(void**)a0) { pthread_mutex_destroy((pthread_mutex_t*)*(void**)a0); free(*(void**)a0); *(void**)a0 = nullptr; } return 0; }
HLE(k_mutex_lock)    { if (a0 && *(void**)a0) pthread_mutex_lock((pthread_mutex_t*)*(void**)a0); return 0; }
HLE(k_mutex_trylock) { return (a0 && *(void**)a0) ? (uint64_t)pthread_mutex_trylock((pthread_mutex_t*)*(void**)a0) : 0x16; }
HLE(k_mutex_unlock)  { if (a0 && *(void**)a0) pthread_mutex_unlock((pthread_mutex_t*)*(void**)a0); return 0; }

// --- condition variables ---
HLE(k_condattr_init)    { if (a0) { auto* c = (pthread_condattr_t*)calloc(1, sizeof(pthread_condattr_t)); pthread_condattr_init(c); *(void**)a0 = c; } return 0; }
HLE(k_condattr_destroy) { if (a0 && *(void**)a0) { free(*(void**)a0); *(void**)a0 = nullptr; } return 0; }
HLE(k_cond_init)      { if (!a0) return 0x16; auto* c = (pthread_cond_t*)calloc(1, sizeof(pthread_cond_t)); pthread_cond_init(c, nullptr); *(void**)a0 = c; return 0; }
HLE(k_cond_destroy)   { if (a0 && *(void**)a0) { pthread_cond_destroy((pthread_cond_t*)*(void**)a0); free(*(void**)a0); *(void**)a0 = nullptr; } return 0; }
HLE(k_cond_signal)    { if (sclog()) fprintf(stderr, "[sync2] T%ld COND.signal    cond=0x%llx\n", sctid(), a0 ? (unsigned long long)*(void**)a0 : 0); if (a0 && *(void**)a0) pthread_cond_signal((pthread_cond_t*)*(void**)a0); return 0; }
HLE(k_cond_broadcast) { if (sclog()) fprintf(stderr, "[sync2] T%ld COND.broadcast cond=0x%llx\n", sctid(), a0 ? (unsigned long long)*(void**)a0 : 0); if (a0 && *(void**)a0) pthread_cond_broadcast((pthread_cond_t*)*(void**)a0); return 0; }
HLE(k_cond_wait)      { if (sclog()) fprintf(stderr, "[sync2] T%ld COND.wait.ent  cond=0x%llx\n", sctid(), a0 ? (unsigned long long)*(void**)a0 : 0);
    if (a0 && *(void**)a0 && a1 && *(void**)a1) pthread_cond_wait((pthread_cond_t*)*(void**)a0, (pthread_mutex_t*)*(void**)a1);
    if (sclog()) fprintf(stderr, "[sync2] T%ld COND.wait.exit cond=0x%llx\n", sctid(), a0 ? (unsigned long long)*(void**)a0 : 0); return 0; }

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
HLE(k_rwlock_destroy) { if (a0 && *(void**)a0) { pthread_rwlock_destroy((pthread_rwlock_t*)*(void**)a0); free(*(void**)a0); *(void**)a0 = nullptr; } return 0; }
HLE(k_rwlock_rdlock)  { if (a0 && *(void**)a0) pthread_rwlock_rdlock((pthread_rwlock_t*)*(void**)a0); return 0; }
HLE(k_rwlock_wrlock)  { if (a0 && *(void**)a0) pthread_rwlock_wrlock((pthread_rwlock_t*)*(void**)a0); return 0; }
HLE(k_rwlock_unlock)  { if (a0 && *(void**)a0) pthread_rwlock_unlock((pthread_rwlock_t*)*(void**)a0); return 0; }
HLE(k_rwlock_tryrdlock){ return (a0 && *(void**)a0) ? (uint64_t)pthread_rwlock_tryrdlock((pthread_rwlock_t*)*(void**)a0) : 0x16; }
HLE(k_rwlock_trywrlock){ return (a0 && *(void**)a0) ? (uint64_t)pthread_rwlock_trywrlock((pthread_rwlock_t*)*(void**)a0) : 0x16; }

// scePthreadOnce(once_control*, init_routine): run init exactly once. We run it UNDER a lock so all
// callers see it complete before returning (pthread_once semantics). A recursive mutex avoids
// self-deadlock if an init routine itself calls scePthreadOnce. The once-control's first int is the
// done-flag. init is a guest fn (guest ABI == host SysV) so it's callable directly.
HLE(k_pthread_once) {
    auto* ctl = (volatile int*)(uintptr_t)a0;
    auto init = (void (*)())(uintptr_t)a1;
    if (!ctl || !init) return 0x16;
    static pthread_mutex_t om = []{ pthread_mutex_t m; pthread_mutexattr_t a;
        pthread_mutexattr_init(&a); pthread_mutexattr_settype(&a, PTHREAD_MUTEX_RECURSIVE);
        pthread_mutex_init(&m, &a); pthread_mutexattr_destroy(&a); return m; }();
    pthread_mutex_lock(&om);
    if (*ctl == 0) { init(); *ctl = 1; }
    pthread_mutex_unlock(&om);
    return 0;
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

// Query the CURRENT thread's real attributes into the caller's attr object (the GC needs
// accurate stack bounds to scan roots; bad bounds make IL2CPP's GC init assert).
HLE(k_attr_get) {
    // scePthreadAttrGet(ScePthread thread, ScePthreadAttr* attr): fill *attr with the given
    // thread's attributes. a0 = thread handle (== the host pthread_t we store), a1 = attr handle.
    // (Bug fixed: the attr is arg1, not arg0 — reading arg0 left the real attr empty, so the GC's
    // GC_get_stack_base got a 0 stack base -> "Bad stack base in GC_register_my_thread".)
#ifdef __linux__
    if (a1 && *(void**)a1) {
        auto* at = (pthread_attr_t*)*(void**)a1;
        void* base = nullptr; size_t sz = 0;
        bool ok = (a0 && guest_stack_for_thread(a0, &base, &sz)) ||
                  guest_stack_for_current_thread(&base, &sz);
        if (ok) pthread_attr_setstack(at, base, sz);   // real, tracked stack for that thread
        // else: leave the attr as-is (avoid the fragile pthread_getattr_np)
    }
#endif
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
HLE(k_attr_getaffinity) { if (a1) *(uint64_t*)(uintptr_t)a1 = 0xff; return 0; }

// --- thread creation: run the guest entry on a real host thread (ABI matches) ---
// We give each worker a stack we allocate and TRACK, so GC/thread-stack queries get
// accurate bounds (see k_attr_get) without relying on pthread_getattr_np.
#ifdef __linux__
namespace {
struct ThreadStart { void* (*entry)(void*); void* arg; void* sbase; uint64_t ssz; };
// Runs first on the new thread: register our own stack (keyed by our tid) BEFORE any guest code,
// so an early GC_register_my_thread / stack-base query from this thread finds it. Closes a race
// where a fast-starting worker ran before the parent's post-create registration → "Bad stack base".
void* thread_trampoline(void* p) {
    auto* ts = (ThreadStart*)p;
    if (ts->sbase) register_thread_stack((uint64_t)pthread_self(), ts->sbase, ts->ssz);
    install_sigaltstack();   // so a guest stack overflow on this worker is still catchable
    auto entry = ts->entry; void* arg = ts->arg; free(ts);   // all host libc — MUST run on the host %fs
    // gated (PROSPER_GUEST_FS): give this guest worker its own guest TCB + static TLS and switch %fs to it
    // as the LAST host action before entering guest code (so guest initial-exec TLS — incl. libc.prx's
    // allocator arena/tcache — resolves to real guest storage, not the aliased host glibc TCB). The import
    // stubs swap back to host %fs per HLE call. No-op when the gate is off. Order matters: the free() above
    // is host glibc (host-TLS tcache) — running it under the guest %fs corrupts the host heap.
    guest_tls_activate_thread();
    return entry ? entry(arg) : nullptr;
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
#ifdef __linux__
    pthread_attr_t la; bool own = false;
    pthread_attr_t* at;
    if (a1 && *(void**)a1) at = (pthread_attr_t*)*(void**)a1;
    else { pthread_attr_init(&la); at = &la; own = true; }
    size_t ssz = 8 * 1024 * 1024;
    void* sbase = mmap(nullptr, ssz, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
    int r;
    if (sbase != MAP_FAILED) {
        pthread_attr_setstack(at, sbase, ssz);
        auto* ts = (ThreadStart*)malloc(sizeof(ThreadStart));
        ts->entry = entry; ts->arg = arg; ts->sbase = sbase; ts->ssz = ssz;
        r = pthread_create(&tid, at, thread_trampoline, ts);   // trampoline registers the stack first
        if (r) free(ts);
    } else {
        r = pthread_create(&tid, at, entry, arg);
    }
    if (own) pthread_attr_destroy(&la);
    if (r) { if (sbase != MAP_FAILED) munmap(sbase, ssz); return (uint64_t)r; }
    if (sbase != MAP_FAILED) register_thread_stack((uint64_t)tid, sbase, ssz);  // redundant safety net
#else
    pthread_attr_t* at = (a1 && *(void**)a1) ? (pthread_attr_t*)*(void**)a1 : nullptr;
    int r = pthread_create(&tid, at, entry, arg);
    if (r) return (uint64_t)r;
#endif
    if (a0) *(uint64_t*)a0 = (uint64_t)tid;
    return 0;
}
HLE(k_pthread_join)   { void* rv = nullptr; pthread_join((pthread_t)a0, a1 ? &rv : nullptr); if (a1) *(void**)(uintptr_t)a1 = rv; return 0; }
HLE(k_pthread_detach) { pthread_detach((pthread_t)a0); return 0; }
HLE(k_pthread_exit)   { pthread_exit((void*)(uintptr_t)a0); return 0; }

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
HLE(k_getspecific)   { return (uint64_t)(uintptr_t)pthread_getspecific((pthread_key_t)a0); }
HLE(k_setspecific)   { return (uint64_t)(int64_t)pthread_setspecific((pthread_key_t)a0, (void*)(uintptr_t)a1); }

// --- event flags (SceKernelEventFlag): a bit pattern with wait/set/clear ---
namespace {
    struct EventFlag { pthread_mutex_t m; pthread_cond_t c; uint64_t bits; };
    bool evf_match(uint64_t bits, uint64_t pat, uint32_t mode) {
        return (mode & 0x1) ? ((bits & pat) == pat) : ((bits & pat) != 0);  // AND vs OR
    }
}
HLE(k_ef_create) {   // (ef*, name, attr, initPattern, opt)
    auto* e = (EventFlag*)calloc(1, sizeof(EventFlag));
    pthread_mutex_init(&e->m, nullptr); pthread_cond_init(&e->c, nullptr); e->bits = a3;
    if (a0) *(void**)(uintptr_t)a0 = e;
    return 0;
}
HLE(k_ef_delete)  { if (a0) free((void*)(uintptr_t)a0); return 0; }
HLE(k_ef_set)     { auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0; pthread_mutex_lock(&e->m); e->bits |= a1; pthread_cond_broadcast(&e->c); pthread_mutex_unlock(&e->m); return 0; }
HLE(k_ef_clear)   { auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0; pthread_mutex_lock(&e->m); e->bits &= a1; pthread_mutex_unlock(&e->m); return 0; }
HLE(k_ef_wait)    { // (ef, pattern, waitMode, resultPat*, timeout*)
    auto* e = (EventFlag*)(uintptr_t)a0; if (!e) return 0;
    pthread_mutex_lock(&e->m);
    while (!evf_match(e->bits, a1, (uint32_t)a2)) pthread_cond_wait(&e->c, &e->m);
    uint64_t res = e->bits;
    if (a2 & 0x10) e->bits = 0; else if (a2 & 0x20) e->bits &= ~a1;   // CLEAR_ALL / CLEAR_PAT
    pthread_mutex_unlock(&e->m);
    if (a3) *(uint64_t*)(uintptr_t)a3 = res;
    return 0;
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
namespace { struct Sema { pthread_mutex_t m; pthread_cond_t c; int64_t count; }; }
HLE(k_sema_create) { // (sema*, name, attr, initCount, maxCount, opt)
    auto* s = (Sema*)calloc(1, sizeof(Sema));
    pthread_mutex_init(&s->m, nullptr); pthread_cond_init(&s->c, nullptr); s->count = (int64_t)(int32_t)a3;
    if (a0) *(void**)(uintptr_t)a0 = s;
    if (sclog()) fprintf(stderr, "[sync2] T%ld SEMA.create  sema=0x%llx name='%s' init=%lld max=%lld\n",
                         sctid(), (unsigned long long)(uintptr_t)s, a1 ? (const char*)(uintptr_t)a1 : "",
                         (long long)(int32_t)a3, (long long)(int32_t)a4);
    return 0;
}
HLE(k_sema_delete) { if (a0) free((void*)(uintptr_t)a0); return 0; }
HLE(k_sema_wait)   { auto* s = (Sema*)(uintptr_t)a0; if (!s) return 0; int64_t need = a1 ? (int64_t)a1 : 1;
    if (sclog()) fprintf(stderr, "[sync2] T%ld SEMA.wait     sema=0x%llx need=%lld\n", sctid(), (unsigned long long)a0, (long long)need);
    pthread_mutex_lock(&s->m); while (s->count < need) pthread_cond_wait(&s->c, &s->m); s->count -= need; pthread_mutex_unlock(&s->m); return 0; }
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
    const UnwindModuleDesc* m = nullptr;
    for (auto& d : g_unwind_mods) if (a0 >= d.lo && a0 < d.hi) { m = &d; break; }
    if (!m) return 0x80020003;                         // SCE_KERNEL_ERROR_ESRCH: addr not in any module
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
void set_unwind_modules(const UnwindModuleDesc* d, size_t c) { g_unwind_mods.assign(d, d + c); }

// ---- Async exception delivery = the IL2CPP GC's stop-the-world thread suspension ----
// The runtime installs a handler (sceKernelInstallExceptionHandler) for exception type 0x1e,
// then to stop the world it calls sceKernelRaiseException(thread, 0x1e) on each thread. On
// real hardware that asynchronously interrupts the target thread and runs its handler ON that
// thread; the handler captures the thread's registers (for GC root scanning) and blocks until
// resumed. We reproduce this exactly with a real-time signal: pthread_sigqueue delivers it to
// the target thread, and our SA_SIGINFO handler synthesises a FreeBSD amd64 mcontext from the
// interrupted ucontext (so the guest handler sees the real registers) and runs the guest
// handler on that thread. A stubbed RaiseException left every thread un-acked -> deadlock.
namespace {
uint64_t g_exc_handlers[128] = {0};   // guest handler fn ptr, indexed by exception type
bool g_exc_log = false;               // set once (outside signal ctx) from PROSPER_SYNCLOG
volatile int* g_exc_counter = nullptr; // optional fork-safe raise counter (tests)
#ifdef __linux__
int  g_exc_sig = -1;
void exc_delivery_handler(int, siginfo_t* si, void* uc_) {
    int type = si->si_value.sival_int;
    if (type < 0 || type >= 128 || !g_exc_handlers[type]) return;
    auto* uc = (ucontext_t*)uc_;
    greg_t* g = uc->uc_mcontext.gregs;
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
    if (g_exc_log) { const char m[] = "[exc] handler ENTER on target\n"; (void)!write(2, m, sizeof m - 1); }
    ((void (*)(uint64_t, void*))(uintptr_t)g_exc_handlers[type])((uint64_t)type, ctx);
    if (g_exc_log) { const char m[] = "[exc] handler EXIT (resumed)\n";   (void)!write(2, m, sizeof m - 1); }
}
void ensure_exc_sig() {
    if (g_exc_sig != -1) return;
    g_exc_log = getenv("PROSPER_SYNCLOG") != nullptr;
    g_exc_sig = SIGRTMIN + 4;                    // free RT signal (not our SIGSEGV/ILL/BUS)
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
#ifdef __linux__
    if (a0 && a1 < 128 && g_exc_handlers[a1]) {
        union sigval sv; sv.sival_int = (int)a1;
        pthread_sigqueue((pthread_t)a0, g_exc_sig, sv);
    }
#endif
    return 0;
}

void set_exc_raise_counter(volatile int* counter) { g_exc_counter = counter; }

HLE(k_is_stack) {   // sceKernelIsStack(void* addr): is addr within the current thread's stack?
#ifdef __linux__
    void* base = nullptr; size_t sz = 0;
    if (guest_stack_for_current_thread(&base, &sz) &&
        a0 >= (uint64_t)(uintptr_t)base && a0 < (uint64_t)(uintptr_t)base + sz)
        return 1;
#endif
    return 0;
}
HLE(k_dlsym) {   // sceKernelDlsym(SceKernelModule handle, const char* name, void** funcAddr)
    // We don't resolve dynamic symbols through the HLE layer. Return ESRCH ("not found") but leave
    // *funcAddr untouched — callers pre-seed it with a fallback and keep that on failure. (The old
    // path returned success; nulling the out pointer here broke a caller that then invoked it.)
    if (getenv("PROSPER_SYNCLOG"))
        fprintf(stderr, "[dlsym] unresolved '%s' -> ESRCH (fallback kept)\n", a1 ? (const char*)(uintptr_t)a1 : "?");
    return 0x80020003;   // SCE_KERNEL_ERROR_ESRCH
}

// --- General-dynamic TLS (__tls_get_addr) for loaded modules (e.g. the real libc.prx). ----------
// PS5 .prx shared libs access thread-locals via __tls_get_addr(tls_index*), where the tls_index
// {module_id, offset} was patched by our DTPMOD64/DTPOFF64 relocs. We keep a per-thread block per
// module id, lazily allocated from the module's PT_TLS template (memsz block, filesz copied from
// the init image, tbss zeroed). This is the general-dynamic model — no %fs needed (that's only for
// the main exe's initial-exec TLS, which the current boot already tolerates).
namespace { std::vector<TlsModuleDesc> g_tls_mods; }
void set_tls_modules(const TlsModuleDesc* descs, size_t count) {
    g_tls_mods.assign(descs, descs + count);
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
    // Per-thread DTV (module id -> TLS block). MUST NOT use a host `thread_local` here: guest threads run
    // under the GUEST %fs, and host thread_local storage is %fs-relative, so it ALIASES guest memory —
    // reads come back as garbage (an unordered_map whose bucket_count reads 0 → `hash % 0` → SIGFPE).
    // This is the same host↔guest %fs-aliasing landmine as the GfxDevice boot wall. Key by the host tid
    // (a syscall, %fs-independent) in a mutex-guarded global map instead. CONFIDENCE: HIGH (root-caused
    // via gdb: SIGFPE in k_tls_get_addr with a corrupt thread_local map under a guest %fs).
    static std::mutex s_dtv_mx;
    static std::unordered_map<std::thread::id, std::unordered_map<uint64_t, void*>> s_dtv;   // per-thread DTV
    std::thread::id tid = std::this_thread::get_id();   // portable per-OS-thread key (no %fs, no syscall)
    { std::lock_guard<std::mutex> lk(s_dtv_mx);
      auto& dtv = s_dtv[tid];
      auto it = dtv.find(modid);
      if (it != dtv.end()) return (uint64_t)(uintptr_t)it->second + off;
    }
    size_t memsz = 64, filesz = 0; uint64_t init_va = 0;
    if (modid < g_tls_mods.size()) {
        memsz  = g_tls_mods[modid].memsz ? g_tls_mods[modid].memsz : 64;
        filesz = g_tls_mods[modid].filesz;
        init_va = g_tls_mods[modid].init_va;
    }
    void* blk = calloc(1, memsz);
    if (init_va && filesz) memcpy(blk, (const void*)(uintptr_t)init_va, filesz);
    { std::lock_guard<std::mutex> lk(s_dtv_mx); s_dtv[tid][modid] = blk; }
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
    R("scePthreadMutexattrSetprotocol", k_mutexattr_setprotocol);
    R("scePthreadMutexattrSetpshared", k_mutexattr_setpshared);
    R("scePthreadMutexattrDestroy", k_mutexattr_destroy);
    R("scePthreadMutexInit", k_mutex_init);
    R("scePthreadMutexDestroy", k_mutex_destroy);
    R("scePthreadMutexLock", k_mutex_lock);
    R("scePthreadMutexTrylock", k_mutex_trylock);
    R("scePthreadMutexUnlock", k_mutex_unlock);
    R("scePthreadCondattrInit", k_condattr_init);
    R("scePthreadCondattrDestroy", k_condattr_destroy);
    R("scePthreadCondInit", k_cond_init);
    R("scePthreadCondDestroy", k_cond_destroy);
    R("scePthreadCondSignal", k_cond_signal);
    R("scePthreadCondBroadcast", k_cond_broadcast);
    R("scePthreadCondWait", k_cond_wait);
    // read/write locks + once (Sony + POSIX names) — real host primitives (thread-safety fix).
    R("scePthreadRwlockInit", k_rwlock_init);        R("pthread_rwlock_init", k_rwlock_init);
    R("scePthreadRwlockDestroy", k_rwlock_destroy);  R("pthread_rwlock_destroy", k_rwlock_destroy);
    R("scePthreadRwlockRdlock", k_rwlock_rdlock);    R("pthread_rwlock_rdlock", k_rwlock_rdlock);
    R("scePthreadRwlockWrlock", k_rwlock_wrlock);    R("pthread_rwlock_wrlock", k_rwlock_wrlock);
    R("scePthreadRwlockUnlock", k_rwlock_unlock);    R("pthread_rwlock_unlock", k_rwlock_unlock);
    R("scePthreadRwlockTryrdlock", k_rwlock_tryrdlock);
    R("scePthreadRwlockTrywrlock", k_rwlock_trywrlock);
    R("scePthreadOnce", k_pthread_once);             R("pthread_once", k_pthread_once);
    R("scePthreadSelf", k_pthread_self);
    R("scePthreadEqual", k_pthread_equal);
    R("scePthreadYield", k_pthread_yield);
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
    R("scePthreadAttrSetdetachstate", k_attr_noop);
    R("scePthreadAttrGetschedparam", k_attr_noop);
    R("scePthreadAttrGet", k_attr_get);
    R("scePthreadAttrGetstackaddr", k_attr_getstackaddr);
    R("scePthreadAttrGetstacksize", k_attr_getstacksize);
    R("scePthreadAttrGetaffinity", k_attr_getaffinity);  // report 8 cores (not an empty mask)
    R("scePthreadAttrSetaffinity", k_attr_noop);         // accept affinity requests (we don't pin)
    R("scePthreadGetaffinity", k_attr_getaffinity);      R("scePthreadSetaffinity", k_attr_noop);
    R("scePthreadGetschedparam", k_attr_noop);  R("pthread_getschedparam", k_attr_noop);
    R("scePthreadSetschedparam", k_attr_noop);  R("scePthreadSetprio", k_attr_noop);
    R("scePthreadGetprio", k_attr_noop);
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
    R("pthread_create", k_pthread_create);   R("pthread_join", k_pthread_join);
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
    R("pthread_condattr_init", k_condattr_init); R("pthread_condattr_destroy", k_condattr_destroy);
    R("pthread_attr_init", k_attr_init);      R("pthread_attr_destroy", k_attr_destroy);
    R("pthread_attr_setstacksize", k_attr_setstacksize);
    R("pthread_attr_setdetachstate", k_attr_noop); R("pthread_attr_setinheritsched", k_attr_noop);
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
    Hle::register_fn("DGMG3JshrZU", (HleFn)k_attr_noop, "sceKernelSetVirtualRangeName");
    #undef R
    register_kernel_mem_hle();    // virtual/direct memory
    register_kernel_time_hle();   // time/clock + C11 threads + stubs
}

} // namespace prosper
