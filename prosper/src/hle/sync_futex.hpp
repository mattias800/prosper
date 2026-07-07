// sync_futex.hpp — the guest-address futex wake/wait primitives, shared by the sync HLE
// (sceKernelWaitOnAddress / sceKernelWakeByAddress in hle_kernel_mem.cpp) and the GPU command
// processor's completion-label wake (command_processor.cpp). ONE implementation, so a waker can
// never silently diverge from the blocking primitive it must pair with (a future non-futex port —
// e.g. Windows WaitOnAddress — changes both sides together).
#pragma once
#include <cstdint>

namespace prosper {

// Bracket a blocking guest futex wait: enter before FUTEX_WAIT, exit after it returns. Lets
// wake_label_waiters skip its wake syscalls entirely while no thread is blocked.
void futex_wait_enter();
void futex_wait_exit();

// FUTEX_WAKE up to n waiters blocked on the 32-bit word at addr. No-op when addr==0 or non-Linux.
void futex_wake(uint64_t addr, int n);

// Wake sync_on_address waiters on a completion label that was just written: both dwords of a
// potentially-64-bit label. Skips the syscalls when nothing is (or is about to be) blocked — safe
// because a racing waiter that registers AFTER the check has its label value re-validated atomically
// inside FUTEX_WAIT, which returns EAGAIN on the already-written value instead of blocking.
void wake_label_waiters(uint64_t addr);

} // namespace prosper
