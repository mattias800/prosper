// sync_retire.hpp — storage lifetime for destroyed guest synchronization objects (#2042).
//
// THE HAZARD. A guest sync object (mutex, condvar, rwlock, semaphore, barrier) is reached through a
// POINTER SLOT in guest memory. Every handler re-reads that slot, resolves the host object, and then
// dereferences it — including across a call that BLOCKS: `interruptible_mutex_lock` and
// `pthread_rwlock_rdlock` park the calling thread *inside* the object. A `Destroy` that returns the
// storage to the allocator therefore frees it under a thread sitting in prosper's own handler, and
// the block goes straight to the next `calloc`. So a guest bug — destroying an object another thread
// is still using — stops corrupting the guest's own state and starts corrupting PROSPER's heap, in
// prosper's own allocator, with no diagnosable link back to the guest call that caused it. That
// asymmetry is the whole argument: the guest's bug stays the guest's bug on hardware and must stay
// the guest's bug here rather than being amplified into an emulator defect.
//
// THE FIX. A destroyed object's storage goes into a QUARANTINE and is reclaimed only after it has
// been unreachable for `PROSPER_SYNC_RETIRE_SECONDS` (default 30 s). A stale pointer inside an
// in-flight handler therefore still names a live object of the right type.
//
// WHY A TIME WINDOW AND NOT PERMANENT RETENTION. #2042 proposed never freeing at all, on the premise
// that guest sync objects are few. Measured on *The Messenger* (`PPSA24651`), 90 s of headless boot,
// `PROSPER_SYNC_RETIRELOG=1`: **131,066 mutex destroys** against 6 cond, 0 rwlock, 0 sem, 0 barrier
// and 0 of either C11 spelling. Permanent retention is ~5 MB per 90 s on a guarded title — hundreds
// of megabytes in a play session — so the premise is false for the family as a whole, and true only
// for the six low-volume kinds. A window bounds the retained set by RATE rather than by total, which
// is the variable that actually varies: at the measured churn the default holds ~44,000 mutexes,
// about 1.7 MB.
//
// WHAT THE WINDOW DOES AND DOES NOT CLOSE. It closes the window in the issue's own failure scenario
// completely — "thread A is descheduled" between resolving the pointer and dereferencing it is
// bounded by scheduler latency, four orders of magnitude inside the default. It does NOT close the
// unbounded case: a thread PARKED inside a destroyed object (blocked on a lock the guest then
// destroyed) can outlive any window, and after it expires the object is reclaimed under that thread.
// Closing that needs hazard-pointer or epoch reclamation bracketing every sync handler — real, and
// far larger than this defect justifies; filed as #2169. Note that FreeBSD frees a rwlock
// under a parked waiter too, so prosper is not worse than the platform there.
//
// WHY THE HOST PRIMITIVE IS DESTROYED ONLY AT RECLAIM. `pthread_*_destroy` on an object another
// thread is blocked in is undefined, and on winpthreads it frees an inner allocation the parked
// thread still holds a handle to — reintroducing exactly the use-after-free this exists to remove.
// So the caller does NOT destroy the host primitive; it hands over a callback that runs once the
// object leaves quarantine, which is also what keeps winpthreads' inner allocation from leaking.
//
// DIAGNOSTICS AND THE COUNTER-ARM.
//   PROSPER_SYNC_RETIRELOG=1        report the retained high-water and a per-kind census.
//   PROSPER_SYNC_RETIRE_SECONDS=N   override the window. **N=0 restores the pre-#2042 immediate
//                                   free**, so the A/B stays reproducible on a shipped binary — the
//                                   `sync_destroy_lifetime_counter_arm` ctest entry is exactly that.
#pragma once
#include <cstdint>

namespace prosper {

// Which spelling retired the object. Only the diagnostic census distinguishes them; the policy does
// not. Keep in sync with kSyncObjectKindNames in sync_retire.cpp.
enum class SyncObjectKind : uint32_t {
    Mutex, Cond, Rwlock, Sem, Barrier, StlMutex, StlCond, Count_
};

// Runs once the object leaves quarantine and immediately before its storage is freed: the caller's
// matching `pthread_*_destroy` / `sem_destroy`, plus any C++ destructor the type needs. Kept at the
// call site so this file needs no knowledge of the object layouts.
using SyncObjectHostDestroy = void (*)(void*);

// Hand a destroyed guest sync object's storage over to the quarantine. Null is ignored. Safe to call
// from any thread. The caller must NOT already have destroyed the host primitive — see above.
void retire_sync_object(void* storage, SyncObjectKind kind, SyncObjectHostDestroy host_destroy);

// Total objects ever retired, and how many are held in quarantine right now. For tests and
// diagnostics.
uint64_t retired_sync_object_count();
uint64_t quarantined_sync_object_count();

}  // namespace prosper
