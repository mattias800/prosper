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
void futex_wait_enter(uint64_t addr);
void futex_wait_exit();

// Windows applies SetThreadContext only after a blocked syscall returns. If `thread` is currently
// inside the HLE WaitOnAddress path, wake that exact address so an asynchronous guest exception can
// enter its redirected context. Returns true when a registered futex wait was woken.
bool interrupt_futex_wait(uint64_t thread);

// FUTEX_WAKE up to n waiters blocked on the 32-bit word at addr. No-op when addr==0 or non-Linux.
void futex_wake(uint64_t addr, int n);

// Wake sync_on_address waiters on a completion label that was just written: both dwords of a
// potentially-64-bit label. Skips the syscalls when nothing is (or is about to be) blocked — safe
// because a racing waiter that registers AFTER the check has its label value re-validated atomically
// inside FUTEX_WAIT, which returns EAGAIN on the already-written value instead of blocking.
void wake_label_waiters(uint64_t addr);

} // namespace prosper
