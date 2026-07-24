// test_initfault_dump — guest init faults must be recovered and their reports must not fault (#128).
// This also drives a real integer #DE/SIGFPE through the installed handler (#1160); without the
// SIGFPE registration the test process is killed before run_guest_inits can recover.
// run_guest_inits
// tolerates a faulting init fn (sigsetjmp recovery) and then prints a diagnostic that includes
// the bytes around the faulting rip and the entry. With a WILD or NULL init-fn pointer — the
// exact failure this diagnostic exists for — the faulting rip is unmapped; the old report
// mprotect'd it unchecked and deref'd it with the recovery guard already disarmed, turning a
// tolerated, logged failure into hard process death. Exit code is truth: merely surviving both
// calls proves the reporter is fault-safe (the dump prints unmapped markers instead).
#include "../src/host/exec_image.hpp"
#include <cstdio>

#ifndef _WIN32
__attribute__((noinline)) static void divide_by_zero(uint64_t, uint64_t) {
    __asm__ volatile("mov $1, %%eax; xor %%edx, %%edx; xor %%ecx, %%ecx; idiv %%ecx"
                     : : : "rax", "rcx", "rdx", "cc");
}
#endif

int main() {
    printf("== test_initfault_dump ==\n");
    prosper::install_trap_handler();
    // A canonical-but-unmapped target (wild jump: rip = target, rip-8 also unmapped) and a null
    // call. Both must be tolerated AND survive the byte-dump diagnostic that follows.
    std::vector<uint64_t> faulting = { 0xdead0000ull, 0ull };
#ifndef _WIN32
    faulting.push_back((uint64_t)(uintptr_t)&divide_by_zero);
#endif
    size_t ok = prosper::run_guest_inits(faulting);
    if (ok != 0) {
        printf("  [FAIL] wild init fns were counted as succeeded (ok=%zu)\n", ok);
        return 1;
    }
#ifdef _WIN32
    int call_alignment = prosper::recovery_thunk_call_rsp_mod16();
    if (call_alignment != 0) {
        printf("  [FAIL] recovery thunk called compiled code with RSP%%16=%d (expected 0)\n",
               call_alignment);
        return 1;
    }
    printf("  [ok]   recovery thunk provided aligned MS-x64 shadow-space call frame\n");
#endif
#ifdef _WIN32
    printf("  [ok]   wild and null faults were reported and recovered\n");
#else
    printf("  [ok]   wild, null, and integer-divide faults were reported and recovered\n");
#endif
    printf("== PASS ==\n");
    return 0;
}
