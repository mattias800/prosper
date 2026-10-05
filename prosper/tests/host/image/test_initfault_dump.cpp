// test_initfault_dump — guest init faults must be recovered and their reports must not fault (#128).
// This also drives a real integer #DE/SIGFPE through the installed handler (#1160); without the
// SIGFPE registration the test process is killed before run_guest_inits can recover.
// run_guest_inits
// tolerates a faulting init fn (sigsetjmp recovery) and then prints a diagnostic that includes
// the bytes around the faulting rip and the entry. With a WILD or NULL init-fn pointer — the
// exact failure this diagnostic exists for — the faulting rip is unmapped; the old report
// mprotect'd it unchecked and deref'd it with the recovery guard already disarmed, turning a
// tolerated, logged failure into hard process death. Exit code is truth: merely surviving both
// calls proves the reporter is fault-safe (the dump prints unmapped markers instead). In GoogleTest
// that survival IS the assertion — this file reaching the end of a TEST is what "the process did not
// die" means, and a reporter that faulted would take the whole ctest case down with it.
#include "host/image/exec_image.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#ifndef _WIN32
__attribute__((noinline)) static void divide_by_zero(uint64_t, uint64_t) {
    __asm__ volatile("mov $1, %%eax; xor %%edx, %%edx; xor %%ecx, %%ecx; idiv %%ecx"
                     : : : "rax", "rcx", "rdx", "cc");
}
#endif

TEST(InitFaultDump, WildAndNullInitFunctionsAreReportedAndRecovered) {
    prosper::install_trap_handler();
    // A canonical-but-unmapped target (wild jump: rip = target, rip-8 also unmapped), a null call,
    // and on hosts with the handler a real integer divide. All must be tolerated AND survive the
    // byte-dump diagnostic that follows.
    std::vector<uint64_t> faulting = { 0xdead0000ull, 0ull };
#ifndef _WIN32
    faulting.push_back(reinterpret_cast<uint64_t>(&divide_by_zero));
#endif
    ASSERT_EQ(prosper::run_guest_inits(faulting), 0u)
        << "a faulting init fn must be recovered, never counted as a succeeded init";
#ifdef _WIN32
    // The ABI half (#633) belongs to the SAME recovery, not to an independent fact: the VEH records
    // the RSP it called the compiled recovery thunk with, and that slot is only populated once a
    // recovery has actually run in this process (it reads -1, "never recorded", until then). Split
    // into its own TEST it would run in a fresh process, assert against nothing, and pass.
    EXPECT_EQ(prosper::recovery_thunk_call_rsp_mod16(), 0)
        << "the recovery thunk called compiled code with RSP%16 != 0 (expected 0)";
#endif
}