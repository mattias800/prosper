// sync_futex.hpp — the guest-address futex wake/wait primitives, shared by the sync HLE
// (sceKernelWaitOnAddress / sceKernelWakeByAddress in hle_kernel_mem.cpp) and the GPU command
// processor's completion-label wake (command_processor.cpp). ONE implementation, so a waker can
// never silently diverge from the blocking primitive it must pair with (a future non-futex port —
// e.g. Windows WaitOnAddress — changes both sides together).
#pragma once
#include <cstddef>
#include <cstdint>
#include <pthread.h>

namespace prosper {

enum class GuestWaitKind : uint32_t { None, Address, ConditionSequence, EventFlag, Semaphore };
struct GuestWaitSnapshot {
    GuestWaitKind kind = GuestWaitKind::None;
    uintptr_t object = 0;
    uintptr_t source = 0;
};

// Describes what a condition wait actually did to its host mutex. Windows condition waits are
// assembled from explicit unlock / WaitOnAddress / relock operations, so callers cannot infer
// ownership from the final errno alone (an error may happen on either side of the unlock).
struct CondWaitMutexState {
    bool unlocked = false;
    bool reacquired = false;
};

// Optional ownership-map transaction paired with the helper's explicit host unlock/relock. The
// release callback reports whether it actually removed bookkeeping, so an unlock error cannot
// invent ownership for a caller that did not own the mutex.
struct CondWaitMutexBookkeeping {
    bool (*release)(pthread_mutex_t*) = nullptr;
    void (*acquire)(pthread_mutex_t*) = nullptr;
};

// Bracket a blocking guest futex wait: enter before FUTEX_WAIT, exit after it returns. Lets
// wake_label_waiters skip its wake syscalls entirely while no thread is blocked.
using WaitRegistration = void*;
WaitRegistration futex_wait_enter(uint64_t addr);
void futex_wait_exit(WaitRegistration registration);

// Register pthread condition waits that may need interruption for asynchronous guest exception
// delivery on Windows. Other hosts call pthread directly through these wrappers.
int interruptible_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex,
                            GuestWaitKind kind = GuestWaitKind::ConditionSequence,
                            uintptr_t source = 0,
                            CondWaitMutexState* mutex_state = nullptr,
                            const CondWaitMutexBookkeeping* bookkeeping = nullptr);
int interruptible_cond_timedwait(pthread_cond_t* cond, pthread_mutex_t* mutex,
                                 const timespec* deadline,
                                 GuestWaitKind kind = GuestWaitKind::ConditionSequence,
                                 uintptr_t source = 0,
                                 CondWaitMutexState* mutex_state = nullptr,
                                 const CondWaitMutexBookkeeping* bookkeeping = nullptr);
// Wait for a relative duration measured by a stable host clock. Used when a guest condition's
// absolute deadline belongs to a non-realtime clock and must not inherit wall-clock jumps.
int interruptible_cond_timedwait_relative(pthread_cond_t* cond, pthread_mutex_t* mutex,
                                          const timespec* duration,
                                          GuestWaitKind kind = GuestWaitKind::ConditionSequence,
                                          uintptr_t source = 0,
                                          CondWaitMutexState* mutex_state = nullptr,
                                          const CondWaitMutexBookkeeping* bookkeeping = nullptr);
int interruptible_cond_signal(pthread_cond_t* cond);
int interruptible_cond_broadcast(pthread_cond_t* cond);
// Release the Windows WaitOnAddress registry entry after a condition variable is destroyed.
// No-op on native POSIX hosts.
void interruptible_cond_forget(pthread_cond_t* cond);
int interruptible_mutex_lock(pthread_mutex_t* mutex);

// Windows applies SetThreadContext only after a blocked syscall returns. Wake the target's exact
// registered WaitOnAddress or pthread-condition wait so it can enter its redirected context.
// Returns true when a registered wait was found and woken.
bool interrupt_guest_wait(uint64_t thread);

// Read every Windows interruptible wait currently registered by a guest thread. Nested exception
// delivery can retain more than one; callers must not claim an arbitrary slot is the current wait.
// Returns the total validated count and stores up to capacity entries. Other hosts return zero.
size_t snapshot_guest_waits(uint64_t windows_tid, GuestWaitSnapshot* snapshots, size_t capacity);
// Convenience for callers that require exactly one active wait. Returns false for zero or nested waits.
bool snapshot_guest_wait(uint64_t windows_tid, GuestWaitSnapshot& snapshot);
void snapshot_guest_wait_registry(size_t& condition_slots_used,
                                  size_t& condition_slots_capacity);

#ifdef _WIN32
// One-shot failure seams for the Windows ownership regression. BeforeUnlock leaves the host mutex
// owned; BeforeRelock fires after a successful unlock and leaves it unowned.
enum class CondWaitFailurePointForTest : uint32_t { None, BeforeUnlock, BeforeRelock };
void win_set_cond_wait_failure_for_test(CondWaitFailurePointForTest point, int error);
#endif

// FUTEX_WAKE up to n waiters blocked on the 32-bit word at addr. No-op when addr==0 or non-Linux.
void futex_wake(uint64_t addr, int n);

// Wake sync_on_address waiters on a completion label that was just written: both dwords of a
// potentially-64-bit label. Skips the syscalls when nothing is (or is about to be) blocked — safe
// because a racing waiter that registers AFTER the check has its label value re-validated atomically
// inside FUTEX_WAIT, which returns EAGAIN on the already-written value instead of blocking.
void wake_label_waiters(uint64_t addr);

} // namespace prosper
