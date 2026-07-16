// sync_futex.hpp — the guest-address futex wake/wait primitives, shared by the sync HLE
// (sceKernelWaitOnAddress / sceKernelWakeByAddress in hle_kernel_mem.cpp) and the GPU command
// processor's completion-label wake (command_processor.cpp). ONE implementation, so a waker can
// never silently diverge from the blocking primitive it must pair with (a future non-futex port —
// e.g. Windows WaitOnAddress — changes both sides together).
#pragma once
#include <cstdint>
#include <pthread.h>

namespace prosper {

enum class GuestWaitKind : uint32_t { None, Address, ConditionSequence };
struct GuestWaitSnapshot {
    GuestWaitKind kind = GuestWaitKind::None;
    uintptr_t object = 0;
    uintptr_t source = 0;
};

// Bracket a blocking guest futex wait: enter before FUTEX_WAIT, exit after it returns. Lets
// wake_label_waiters skip its wake syscalls entirely while no thread is blocked.
using WaitRegistration = void*;
WaitRegistration futex_wait_enter(uint64_t addr);
void futex_wait_exit(WaitRegistration registration);

// Register pthread condition waits that may need interruption for asynchronous guest exception
// delivery on Windows. Other hosts call pthread directly through these wrappers.
int interruptible_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex);
int interruptible_cond_timedwait(pthread_cond_t* cond, pthread_mutex_t* mutex,
                                 const timespec* deadline);
int interruptible_cond_signal(pthread_cond_t* cond);
int interruptible_cond_broadcast(pthread_cond_t* cond);
int interruptible_mutex_lock(pthread_mutex_t* mutex);

// Windows applies SetThreadContext only after a blocked syscall returns. Wake the target's exact
// registered WaitOnAddress or pthread-condition wait so it can enter its redirected context.
// Returns true when a registered wait was found and woken.
bool interrupt_guest_wait(uint64_t thread);

// Read a Windows guest thread's currently registered interruptible wait. Used by the
// app checkpoint to distinguish futex/label waits from pthread condition waits.
bool snapshot_guest_wait(uint64_t windows_tid, GuestWaitSnapshot& snapshot);

// FUTEX_WAKE up to n waiters blocked on the 32-bit word at addr. No-op when addr==0 or non-Linux.
void futex_wake(uint64_t addr, int n);

// Wake sync_on_address waiters on a completion label that was just written: both dwords of a
// potentially-64-bit label. Skips the syscalls when nothing is (or is about to be) blocked — safe
// because a racing waiter that registers AFTER the check has its label value re-validated atomically
// inside FUTEX_WAIT, which returns EAGAIN on the already-written value instead of blocking.
void wake_label_waiters(uint64_t addr);

} // namespace prosper
