#pragma once
// One fault's own context, snapshotted from ITS siginfo/ucontext.
//
// exec_image's signal handler keeps the interrupted registers in process-global variables
// (`g_fault_addr`, `g_fault_rip`, `g_rax` … `g_r15`) because a guest thread may be running on a
// guest %fs, which makes `__thread` storage unusable inside the handler — see the `g_jb` comment in
// exec_image_linux.cpp. Those globals are correct for the armed/main-thread recovery path, which is
// single-threaded by construction, but they are NOT a per-fault record: the guest is heavily
// multithreaded and its faults are correlated (one heap corruption takes several worker threads
// within milliseconds), so a second thread faulting while the first is still printing overwrites
// every global underneath it. The first thread's report then continues with the second thread's
// registers and reads as one complete fault (#2018).
//
// A stack-local snapshot has neither problem: it needs no TLS, and it cannot be rewritten by another
// thread. Capture once at handler entry and drive the whole report from it.
//
// Async-signal-safe: plain field copies out of the ucontext, no allocation and no libc calls.
#include <atomic>
#include <cstdint>

#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE   // glibc gates the REG_* gregs[] indices behind __USE_GNU
#endif
#include <signal.h>
#include <ucontext.h>
#endif
// Darwin deliberately gets no <ucontext.h>: that header is `#error`-guarded behind _XOPEN_SOURCE
// on macOS. posix_shim.hpp's Darwin branch already pulls in <sys/ucontext.h>, which is where
// ucontext_t/mcontext_t come from there.

#include "posix_shim.hpp"   // PROSPER_GREGS + the REG_* indices (Linux gregs[] / Darwin __ss)

namespace prosper::host {

struct FaultContext {
    uint64_t addr = 0;   // siginfo_t::si_addr — the address that faulted
    uint64_t rip = 0, rbp = 0, rsp = 0;
    uint64_t rax = 0, rbx = 0, rcx = 0, rdx = 0, rsi = 0, rdi = 0;
    uint64_t r8 = 0, r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0;
    int  sig = 0;
    long tid = 0;        // kernel tid, so a report names the thread it actually describes
};

// `uc` is the handler's own third argument and `fault_addr` its siginfo's `si_addr` (spelled
// differently because glibc defines `si_addr` as a macro, so it cannot be a parameter name).
// Deliberately takes no default and reads no global, so a caller cannot accidentally build a
// context out of shared state.
inline FaultContext capture_fault_context(int sig, long tid, const void* fault_addr,
                                          ucontext_t* uc) {
    FaultContext f;
    f.sig = sig;
    f.tid = tid;
    f.addr = (uint64_t)(uintptr_t)fault_addr;
    if (!uc) return f;
    auto g = PROSPER_GREGS(uc);
    f.rip = (uint64_t)g[REG_RIP]; f.rbp = (uint64_t)g[REG_RBP]; f.rsp = (uint64_t)g[REG_RSP];
    f.rax = (uint64_t)g[REG_RAX]; f.rbx = (uint64_t)g[REG_RBX]; f.rcx = (uint64_t)g[REG_RCX];
    f.rdx = (uint64_t)g[REG_RDX]; f.rsi = (uint64_t)g[REG_RSI]; f.rdi = (uint64_t)g[REG_RDI];
    f.r8  = (uint64_t)g[REG_R8];  f.r9  = (uint64_t)g[REG_R9];  f.r10 = (uint64_t)g[REG_R10];
    f.r11 = (uint64_t)g[REG_R11]; f.r12 = (uint64_t)g[REG_R12]; f.r13 = (uint64_t)g[REG_R13];
    f.r14 = (uint64_t)g[REG_R14]; f.r15 = (uint64_t)g[REG_R15];
    return f;
}

// Who owns the fatal fault report *right now*. A coherent snapshot stops a report from carrying
// another thread's registers; it does not stop two coherent reports from being written to stderr at
// the same time and arriving spliced together. So the first faulting thread claims the report and a
// thread that arrives while one is in flight says so in one line instead of printing over it.
//
// This call itself never blocks — it is one compare-exchange. Whether a loser then WAITS for the
// owner is the caller's decision, and the caller must bound it: the owner may die inside its own
// dump, and a signal handler that waited unconditionally on another *faulting* thread would hang a
// run that used to terminate.
//
// Ownership is scoped to one report, not to the process: `release_fault_report` frees it when the
// dump completes, so a fault arriving LATER — with no concurrency at all — gets its own full
// report. Holding it for the life of the process would be sound only if a fatal fault always ended
// the process, and PROSPER_WORKER_PARK exists precisely so that it does not.
//
// Returns true when this thread owns the dump. `already` is set to the owning tid either way, so a
// losing thread can name who is reporting and can watch that value for the release.
inline bool claim_fault_report(std::atomic<long>& owner, long tid, long* already = nullptr) {
    long expected = 0;
    const bool won = owner.compare_exchange_strong(expected, tid);
    if (already) *already = won ? tid : expected;
    // Re-entering on the same thread (a fault raised inside the dump) is not a competing report:
    // it is the owner, and refusing it there would silently truncate the record.
    return won || expected == tid;
}

// Release a claim taken by `claim_fault_report`. Compare-exchange rather than a plain store, so a
// thread that never owned the gate cannot hand it to a third thread in the middle of someone
// else's report.
//
// Deliberately NOT nesting-counted, and that is safe only because of a property of the caller:
// `claim_fault_report` grants re-entry to the owning tid (a fault raised inside the dump), and a
// re-entrant call would then release the gate while the outer report is still open. It cannot,
// because the fatal report block never returns — its every exit is `_exit` or an unbounded park —
// so an inner report ends the process rather than unwinding back into the outer one. If that ever
// stops being true, this needs a depth count, not a bool.
inline void release_fault_report(std::atomic<long>& owner, long tid) {
    long expected = tid;
    owner.compare_exchange_strong(expected, 0);
}

}  // namespace prosper::host
