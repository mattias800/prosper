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
// `ulimit -c 0` DOES NOT FIX THIS, and the reason is narrower than "it does nothing" — that phrasing
// was in an earlier draft of this comment and is wrong. Measured 2026-09-03 on this box, 20 SIGSEGVs
// per arm:
//
//     arm                        handler invoked   core files stored   wall
//     ulimit -c 0                     20 / 20              0           0.78 s
//     ulimit -c unlimited             20 / 20             20           8.53 s
//     prctl(PR_SET_DUMPABLE, 0)        0 / 20              0           0.009 s
//
// So `ulimit -c 0` really does stop the core file being written — systemd-coredump logs "Resource
// limits disable core dumping" and declines to save. What it does NOT stop is the handler being
// invoked at all: when `core_pattern` begins with `|`, the kernel spawns the handler regardless and
// leaves RLIMIT_CORE for it to interpret, so the process spawn and the journal commit are still paid
// for every crash. THAT is the #3269 stall, which is why the flag is not the fix even though it
// looks like one and is 11x faster than doing nothing.
//
// `PR_SET_DUMPABLE` is decided in the kernel before the handler is reached, so nothing is spawned:
// 87x faster again, and the only arm that leaves no journal entry.
//
// (Checking this the cheap way — crash under `ulimit -c 0` and look in /var/lib/systemd/coredump —
// finds no file and reads as though the flag worked. `coredumpctl list`'s COREFILE column shows
// `none` for those entries, which is the distinction the earlier draft missed.)
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
