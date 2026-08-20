#pragma once
// SIGTRAP arbitration between prosper's own in-process debug instruments (#1932).
//
// The Linux fault_handler is ONE SIGTRAP consumer chain shared by several independent instruments.
// Two of them are code breakpoints, and they trap at different places relative to their address:
//
//   * PROSPER_HWBP / PROSPER_HWWATCH — perf hardware break/watchpoints, one fd per owning TID. An
//     execute breakpoint traps BEFORE the instruction at g_hwbp_addr runs, so its RIP is exactly
//     g_hwbp_addr. Recovery is: disable the fd, set TF, single-step off the address, re-enable on
//     the completion trap. That in-flight step is tracked PER TID (HwbpThreadState::stepping) with
//     a single-thread fallback (g_hwbp_stepping).
//   * PROSPER_BP — a software int3. The 0xCC replaces the first byte at g_bp_addr, so a hit traps
//     AFTER it with RIP == g_bp_addr + 1. Recovery is: restore the original byte, back RIP up to
//     g_bp_addr, set TF to re-execute the real instruction, and re-insert the 0xCC on the
//     completion trap. That sequence is marked by g_bp_stepping.
//
// The HWBP branch runs first in the chain and used to accept ANY SIGTRAP while an HWBP or HWWATCH
// fd was live. With both instruments armed the int3 hit was therefore consumed by the WRONG state
// machine: a plausible-looking "[hwbp] rip=<bp addr>+1" record was fabricated, BP's restore /
// back-up-RIP / step / re-arm sequence never ran, the guest's original instruction was skipped, and
// the title died on the resulting corruption. The instrument manufactured data and then crashed the
// subject it was measuring.
//
// The decision below is deliberately architectural — RIP, the trap flag, and each instrument's own
// step state — rather than kernel-reported siginfo. si_code/si_fd DO separate these producers on
// current Linux (a perf-delivered SIGTRAP arrives as SI_SIGIO with a valid si_fd; an int3 arrives as
// TRAP_BRKPT and a TF completion as TRAP_TRACE), but the handler already carries a note that perf's
// asynchronous delivery does not preserve TRAP_TRACE on every host/kernel combination, and
// posix_shim.hpp defines PROSPER_SI_FD as a constant -1 on Darwin. RIP and TF are reported by the
// CPU, not by the delivery path, so they hold wherever the handler runs.
//
// CONFIDENCE: HIGH — every clause is decided by state this process itself set, and the regression
// test tests/host/fault/test_trap_arbitration.cpp drives the real production shape (a live perf execute
// breakpoint plus a real int3) through both this predicate and the pre-fix condition it replaces.

#include <cstdint>

namespace prosper::host {

// What the arbiter needs from the PROSPER_BP software-breakpoint state machine.
struct SoftwareBreakpointTrapState {
    bool     armed = false;      // g_bp_on: PROSPER_BP was parsed (the 0xCC is installed or mid-step)
    bool     stepping = false;   // g_bp_stepping: original byte restored, TF set, completion pending
    uint64_t addr = 0;           // g_bp_addr: the address holding the patched byte
};

// What the arbiter needs from the PROSPER_HWBP / PROSPER_HWWATCH instruments.
struct HardwareBreakpointTrapState {
    bool armed = false;                 // g_hwbp_on
    int  breakpoint_fd = -1;            // g_hwbp_fd: the always-live anchor fd, or -1
    int  watchpoint_fd = -1;            // g_hwwatch_fd: the chained data watchpoint fd, or -1
    // HwbpThreadState::stepping for THIS tid, falling back to g_hwbp_stepping. Per-TID ownership is
    // the whole point of PROSPER_HWBP_ALLTHREADS and must survive this arbitration untouched.
    bool stepping_this_thread = false;
};

// True when at least one HWBP/HWWATCH fd is live — the original (pre-#1932) entry condition of the
// hardware-breakpoint branch, and still the cheap gate that decides whether the branch can own
// anything at all.
inline bool hardware_breakpoint_instrument_live(const HardwareBreakpointTrapState& hw) {
    return (hw.armed && hw.breakpoint_fd >= 0) || hw.watchpoint_fd >= 0;
}

// True when PROSPER_BP's OWN state proves this SIGTRAP belongs to it.
//
// `rip` and `trap_flag_set` come from the trapping thread's ucontext (RIP and EFLAGS.TF). TF is
// still set in the context of a single-step completion — both state machines clear it by hand — so
// it distinguishes "this thread was mid-step" from "an asynchronous perf hit landed here".
inline bool software_breakpoint_owns_sigtrap(const SoftwareBreakpointTrapState& bp,
                                             const HardwareBreakpointTrapState& hw,
                                             uint64_t rip,
                                             bool trap_flag_set) {
    if (!bp.armed) return false;
    // HWBP's own single step, tracked per TID, always completes in the HWBP branch. Checking it
    // first keeps the RIP proof below from claiming a step completion that merely happens to land
    // one byte past the patched address — reachable only when the two instruments are armed within
    // one instruction of each other, which is a user error either way.
    if (hw.stepping_this_thread) return false;
    // (b) The int3 hit itself. A perf EXECUTE breakpoint traps before its instruction, so its RIP is
    //     g_hwbp_addr; only the trap after our own 0xCC lands on g_bp_addr + 1. This also covers the
    //     completion of a step over a ONE-byte original instruction, which lands on the same RIP —
    //     harmlessly, because the BP branch checks g_bp_stepping before the RIP case.
    if (rip == bp.addr + 1) return true;
    // (a) The completion of the single step BP itself scheduled, for an original instruction longer
    //     than one byte. Requiring TF keeps a genuine perf hit on ANOTHER thread from being stolen
    //     while this global flag is set: that thread's context has TF clear.
    return bp.stepping && trap_flag_set;
}

// The hardware-breakpoint branch's entry condition: an HWBP/HWWATCH fd is live AND the trap is not
// one PROSPER_BP has proven is its own.
inline bool hardware_breakpoint_claims_sigtrap(const HardwareBreakpointTrapState& hw,
                                               const SoftwareBreakpointTrapState& bp,
                                               uint64_t rip,
                                               bool trap_flag_set) {
    return hardware_breakpoint_instrument_live(hw) &&
           !software_breakpoint_owns_sigtrap(bp, hw, rip, trap_flag_set);
}

}  // namespace prosper::host
