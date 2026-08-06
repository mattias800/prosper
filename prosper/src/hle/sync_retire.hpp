// sync_retire.hpp — storage lifetime for destroyed guest synchronization objects (#2042).
//
// THE HAZARD. A guest sync object (mutex, condvar, rwlock, semaphore, barrier) is reached through a
// POINTER SLOT in guest memory. Every handler re-reads that slot, resolves the host object, and then
// dereferences it — including across a call that BLOCKS: `pthread_rwlock_rdlock(&g->rw)` parks the
// calling thread *inside* the object, and so does `interruptible_cond_wait`. A concurrent `Destroy`
// that returned the storage to the allocator would free it under a thread parked in prosper's own
// handler, and the block would then be handed to the next `calloc`. So a guest bug — destroying an
// object another thread is still using — becomes corruption of PROSPER's heap, in prosper's own
// allocator, with no diagnosable link back to the guest call that caused it. That asymmetry is the
// whole argument: the guest's bug stays the guest's bug on hardware, and must stay the guest's bug
// here rather than being amplified into an emulator defect.
//
// THE FIX. A destroyed object's storage is RETIRED: kept alive, and never reused for anything. A
// stale pointer then still names a live object of the right type, so the worst outcome is an
// operation on a lock nothing can reach any more — which is the guest's own bug, contained.
//
// WHY THE POINTERS ARE KEPT IN A LIVE CONTAINER. Simply not calling `free` would be indistinguish-
// able from a leak to LSan/valgrind and would bury real leaks under this one. A reachable owner
// makes the retention a deliberate, auditable policy instead of an omission.
//
// WHY THE HOST PRIMITIVE IS NOT DESTROYED FIRST. `pthread_*_destroy` on an object another thread is
// blocked in is undefined, and on some hosts (winpthreads) it frees an inner allocation the parked
// thread still holds a handle to — reintroducing exactly the use-after-free this exists to remove.
// Callers therefore retire the object WITHOUT calling the matching host destroy, and without running
// its C++ destructor. On glibc and Apple libc these destroys own no kernel resource, so nothing is
// lost; on winpthreads the inner allocation is retained along with the object.
//
// COST, AND HOW IT ANNOUNCES ITSELF. One object per guest `Destroy`, never reclaimed. Bounding that
// needs epoch or hazard-pointer reclamation, which is a much larger change than this defect
// justifies (#2042) — so instead the retained count is REPORTED at powers of two from 65,536 upward
// (and from 1 under `PROSPER_SYNC_RETIRELOG=1`). A title that churns sync objects therefore
// announces itself rather than growing silently, and the follow-up is then justified by measurement
// rather than by argument.
#pragma once
#include <cstdint>

namespace prosper {

// Take ownership of a destroyed guest sync object's storage: never freed, never reused. Null is
// ignored. Safe to call from any thread. The caller must NOT have called the matching host
// `pthread_*_destroy` / `sem_destroy`, and must not run the object's C++ destructor — see above.
void retire_sync_object(void* storage);

// How many objects have been retired so far. For tests and diagnostics.
uint64_t retired_sync_object_count();

}  // namespace prosper
