// exc_pending.hpp -- the macOS sceKernelRaiseException request table (#4324).
//
// Darwin has no pthread_sigqueue / RT-signal payload, so on macOS the exception TYPE travels
// through this async-signal-safe table keyed by the target pthread instead of si_value: a raise
// claims a slot before pthread_kill, and the delivery handler (on the target thread) takes it back.
// Lock-free (atomics only), so both sides are safe in a signal handler. Platform-neutral code; only
// the macOS raise path uses it.
//
// SIZED FOR A WHOLE STOP-THE-WORLD BURST, not for one target. IL2CPP's stop-the-world raises every
// attached thread back to back and only then waits for one acknowledgement per raise, and under
// Rosetta the raiser outruns delivery: Silksong had 16 requests pending at once, the 17th raise
// returned EAGAIN, the guest still counted that thread, and il2cpp_stop_gc_world waited forever (an
// intermittent scene-load hang, ~1 boot in 8 unattended). Silksong's bursts raised ~24 threads; its
// process runs ~100 threads in all. One slot per thread that can have a request in flight is the
// real bound, so the table is sized far above either.
#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper {

inline constexpr size_t kExcPendingSlots = 1024;

// Record a request of `type` for thread `tid`. Returns a non-zero handle, or 0 when every slot is
// in flight (the caller reports EAGAIN).
size_t exc_pending_put(uint64_t tid, int type);

// Undo a put whose signal was never sent (pthread_kill failed), so the slot is not leaked. A no-op
// if the request was already taken.
void exc_pending_release(size_t handle, uint64_t tid);

// Take one request pending for `tid`: its type, or -1 when none. Called by the target thread itself
// (its delivery handler); a request is handed out at most once (CAS), and never before its type is
// published (exc_pending_put is two-phase).
int exc_pending_take(uint64_t tid);

}   // namespace prosper
