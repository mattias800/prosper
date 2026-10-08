// mutex_diagnostics.hpp -- env-gated diagnostics for guest mutexes, moved out of hle_kernel.cpp.
//
//   PROSPER_MUTEX_TRACE    name every lock/trylock/unlock on one guest mutex slot (mtx_trace)
//   PROSPER_MUTEX_WAITLOG  name the holder of a contended mutex when a thread is about to block on it
//
// Both are diagnostics: they observe and never change what the guest sees. The bodies are moved
// verbatim from hle_kernel.cpp (internal linkage, in an unnamed namespace, so there is exactly one copy
// per translation unit that includes this -- today only hle_kernel.cpp). `mtx_report`, which also turns a
// host errno into the guest's, stays in hle_kernel.cpp because it needs that file's `fbsd_errno`.
#pragma once

#include "host/platform/posix_shim.hpp"   // prosper_gettid

#include <pthread.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

// Not self-contained by design: it relies on windows.h (GetCurrentThreadId) and on `arch_prctl` /
// ARCH_GET_FS being declared by the includes hle_kernel.cpp already has, so that moving the code here
// adds no host-platform #if of its own (the architecture ratchet counts those).

namespace {
// PROSPER_MUTEX_TRACE -- name every lock/trylock/unlock on ONE guest mutex slot, with the host
// thread and the host TCB that performed it. Set it to the slot address in hex
// (`PROSPER_MUTEX_TRACE=1500001e80`) or to `all`. Zero cost when unset.
//
// Why this exists rather than reusing what was here (#3615). `PROSPER_MUTEX_FAILLOG` prints the
// failing call and nothing else, so it cannot say who held the lock; `PROSPER_SYNCLOG` prints
// acquisitions only and is hardcoded to the AkSoundEngine slot range, so it answers nothing for
// any other title. An ownership question needs the SEQUENCE, the thread, and -- this is the part
// that actually cracked #3615 -- the %fs base, because on glibc a mutex's owner check is resolved
// through the TCB that %fs points at, NOT through a syscall-derived thread id. A trace showing
// "same thread, same mutex, EPERM anyway" is unreadable until you can also see that the two calls
// ran on two different TCBs.
//
// `tid` is `prosper_gettid()`, a raw syscall, deliberately: anything TLS-derived would be fooled
// by the very TCB swap this is built to expose. `guest_tcb` says whether %fs is one of OUR guest
// TCBs (the "PROS" magic guest_tls.cpp stamps) or a host one, which separates "the import stub
// restored the wrong host TCB" from "the stub did not swap back at all" -- two different bugs
// that present identically in every other field.
inline bool mtx_trace_enabled(uint64_t slot) {
    static const char* const env = getenv("PROSPER_MUTEX_TRACE");
    if (!env) return false;
    static const bool all = std::strcmp(env, "all") == 0;
    if (all) return true;
    static const uint64_t want = std::strtoull(env, nullptr, 16);
    return slot == want;
}
// `prosper_gettid()` does not exist on Windows -- posix_shim.hpp is entirely #ifndef _WIN32 --
// so this mirrors the rw_self_tid()/sctid() split already in this file. Syscall-derived wherever
// one exists, because a TLS-derived id is exactly what this trace is built to catch going wrong.
inline uint64_t mtx_trace_tid() {
#if defined(__linux__) || defined(__APPLE__)
    return (uint64_t)prosper_gettid();
#elif defined(_WIN32)
    return (uint64_t)GetCurrentThreadId();
#else
    return (uint64_t)(uintptr_t)pthread_self();
#endif
}
inline void mtx_trace(const char* op, uint64_t slot, pthread_mutex_t* m, int rc) {
    if (!mtx_trace_enabled(slot)) return;
    int kind = -1, owner = -1, count = -1;
#if defined(__GLIBC__)
    kind = m->__data.__kind;
    owner = m->__data.__owner;
    count = (int)m->__data.__count;
#else
    (void)m;
#endif
    unsigned long fs = 0;
    int guest_tcb = -1;
#if defined(__linux__)
    if (arch_prctl(ARCH_GET_FS, (unsigned long)&fs) != 0) fs = 0;
    // guest_tls.cpp stamps GUEST_TCB_MAGIC at TP+0x108 on every TCB it manufactures. Reading it
    // on a host TCB is safe -- it is a positive offset into the allocated TLS block -- and simply
    // does not match. Keep these two constants in sync with guest_tls.cpp.
    if (fs) guest_tcb = (*(volatile unsigned*)(fs + 0x108) == 0x50524F53u) ? 1 : 0;
#endif
    std::fprintf(stderr,
                 "[mtx-trace] %-8s slot=0x%llx m=%p tid=%ld rc=%d kind=%d owner=%d count=%d "
                 "fs=0x%lx guest_tcb=%d\n",
                 op, (unsigned long long)slot, (void*)m, (long)mtx_trace_tid(), rc, kind, owner,
                 count, fs, guest_tcb);
}
}   // namespace

// PROSPER_MUTEX_WAITLOG: name the HOLDER of a contended guest mutex when a thread is about to block on
// it — a deadlock/lock-contention probe. POSIX-only (Linux/macOS): Windows already has the guest-mutex
// ownership map above, and winpthreads lacks pthread_getname_np. Fully gated: the owner map is only
// touched when the env var is set, so there is zero cost by default.
#ifndef _WIN32
namespace {
struct MtxOwner {
    long tid = 0;
    char name[16] = {0};
};
std::mutex g_mtx_waitlog_mx;
std::unordered_map<pthread_mutex_t*, MtxOwner> g_mtx_waitlog_owner;
inline bool mtx_waitlog() {
    static const bool on = getenv("PROSPER_MUTEX_WAITLOG") != nullptr;
    return on;
}
void mtx_waitlog_record(pthread_mutex_t* m) {
    if (!mtx_waitlog()) return;
    MtxOwner o;
    o.tid = (long)prosper_gettid();
    pthread_getname_np(pthread_self(), o.name, sizeof o.name);
    std::lock_guard<std::mutex> lk(g_mtx_waitlog_mx);
    g_mtx_waitlog_owner[m] = o;
}
void mtx_waitlog_clear(pthread_mutex_t* m) {
    if (!mtx_waitlog()) return;
    std::lock_guard<std::mutex> lk(g_mtx_waitlog_mx);
    g_mtx_waitlog_owner.erase(m);
}
void mtx_waitlog_report_block(pthread_mutex_t* m, uint64_t slot) {
    if (!mtx_waitlog()) return;
    if (pthread_mutex_trylock(m) == 0) {
        pthread_mutex_unlock(m);
        return;
    }   // uncontended
    MtxOwner o{};
    bool have = false;
    {
        std::lock_guard<std::mutex> lk(g_mtx_waitlog_mx);
        auto it = g_mtx_waitlog_owner.find(m);
        if (it != g_mtx_waitlog_owner.end()) {
            o = it->second;
            have = true;
        }
    }
    char self[16] = {0};
    pthread_getname_np(pthread_self(), self, sizeof self);
    static std::atomic<int> n{0};
    if (n.fetch_add(1) < 256)
        fprintf(stderr, "[mtx-wait] '%s'(tid=%ld) BLOCKING on guest mutex slot=0x%llx held by %s\n",
                self, (long)prosper_gettid(), (unsigned long long)slot,
                have ? ([&] {
                    static char b[48];
                    snprintf(b, sizeof b, "'%s'(tid=%ld)", o.name, o.tid);
                    return b;
                }())
                     : "<owner unknown / acquired via cond_wait>");
}
}   // namespace
#else
namespace {
inline void mtx_waitlog_record(pthread_mutex_t*) {}
inline void mtx_waitlog_clear(pthread_mutex_t*) {}
inline void mtx_waitlog_report_block(pthread_mutex_t*, uint64_t) {}
}   // namespace
#endif
