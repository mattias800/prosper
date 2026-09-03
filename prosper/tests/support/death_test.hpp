// death_test.hpp — suppress the core dump of a child this suite kills ON PURPOSE.
//
// Several cases here prove a contract by watching a child die: `test_raw_syscall` faults an
// unclamped write into a guard page, `test_libc_alloc` re-execs itself so a throwing operator new
// can abort. The death is the assertion, and it is correct.
//
// What is not correct is what the death costs everyone else. This box pipes `core_pattern` to
// `systemd-coredump`, so each deliberate crash spawns a handler that compresses and journals a dump
// of a process that was killed on purpose and whose corpse nobody will ever open. With several
// agents running suites concurrently those handlers pile up, and the resulting stall lands on
// whichever lane happens to be running a THIRD suite — as an unexplained hang in an unrelated test,
// which reads as "my change deadlocks something" (#3269).
//
// `ulimit -c 0` DOES NOT FIX THIS, which is the whole reason this header exists rather than a line
// in the charter. Measured 2026-09-03 on this box: with `ulimit -c` verified at 0, a crashing
// process still produced a dump — one, exactly as with `ulimit -c unlimited`. When `core_pattern`
// begins with `|`, the kernel invokes the handler and leaves the limit for the handler to
// interpret. `PR_SET_DUMPABLE` is decided in the kernel before that, so it is the one that holds:
// same experiment, 0 dumps with it against 1 for the control.
//
// Call it in the CHILD, as early as possible and always before the deliberate death. It is
// deliberately not called for the parent: a test process that crashes UNEXPECTEDLY should still
// leave a dump, because that one is worth opening.
#pragma once

#if defined(__linux__)
#include <sys/prctl.h>
#endif

namespace prosper::test {

// Mark this process undumpable. No-op where the concept does not exist. Returns true if the
// suppression was actually applied, so a caller that wants to assert it can.
inline bool suppress_coredump() {
#if defined(__linux__)
    return prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) == 0;
#else
    // macOS and Windows: no piped-handler pile-up of the kind #3269 describes, and no equivalent
    // one-call switch. Left as a no-op rather than faked, so the return value stays honest.
    return false;
#endif
}

}  // namespace prosper::test
