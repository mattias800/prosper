// test_initfault_dump — the init-fault REPORT must never fault itself (#128). run_guest_inits
// tolerates a faulting init fn (sigsetjmp recovery) and then prints a diagnostic that includes
// the bytes around the faulting rip and the entry. With a WILD or NULL init-fn pointer — the
// exact failure this diagnostic exists for — the faulting rip is unmapped; the old report
// mprotect'd it unchecked and deref'd it with the recovery guard already disarmed, turning a
// tolerated, logged failure into hard process death. Exit code is truth: merely surviving both
// calls proves the reporter is fault-safe (the dump prints unmapped markers instead).
#include "../src/host/exec_image.hpp"
#include <cstdio>

int main() {
    printf("== test_initfault_dump ==\n");
    prosper::install_trap_handler();
    // A canonical-but-unmapped target (wild jump: rip = target, rip-8 also unmapped) and a null
    // call. Both must be tolerated AND survive the byte-dump diagnostic that follows.
    size_t ok = prosper::run_guest_inits({ 0xdead0000ull, 0ull });
    if (ok != 0) {
        printf("  [FAIL] wild init fns were counted as succeeded (ok=%zu)\n", ok);
        return 1;
    }
    printf("  [ok]   wild + null init fns tolerated; the fault report did not kill the process\n");
    printf("== PASS ==\n");
    return 0;
}
