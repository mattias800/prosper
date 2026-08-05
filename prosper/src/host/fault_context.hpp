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

#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE   // glibc gates the REG_* gregs[] indices behind __USE_GNU
#endif
#include <signal.h>
#include <ucontext.h>
#endif

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

// Who owns the one fatal fault report. A coherent snapshot stops a report from carrying another
// thread's registers; it does not stop two coherent reports from being written to stderr at the same
// time and arriving spliced together. So the first faulting thread claims the report and the others
// say so in one line instead of printing over it.
//
// Never blocks. A signal handler that waited for another *faulting* thread to finish could wait
// forever — the owner may itself die inside its dump — so a loser reports and leaves rather than
// queueing. `owner` starts at 0 (no owner) and is never cleared: the process is on its way to
// `_exit`, and a second claim after the first report is finished would still splice into a log the
// reader is treating as one crash.
//
// Returns true when this thread owns the full dump. `already` is set to the owning tid either way,
// so a losing thread can name who is reporting.
inline bool claim_fault_report(std::atomic<long>& owner, long tid, long* already = nullptr) {
    long expected = 0;
    const bool won = owner.compare_exchange_strong(expected, tid);
    if (already) *already = won ? tid : expected;
    // Re-entering on the same thread (a fault raised inside the dump) is not a competing report:
    // it is the owner, and refusing it there would silently truncate the record.
    return won || expected == tid;
}

}  // namespace prosper::host
