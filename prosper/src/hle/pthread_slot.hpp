// pthread_slot.hpp — resolving a GUEST pthread object slot to the host object it names.
//
// A FreeBSD/PS5 pthread object is a POINTER held in guest storage, and a STATICALLY INITIALISED one
// is a small sentinel (`PTHREAD_MUTEX_INITIALIZER` == NULL, adaptive == 1, …). Real libthr
// SELF-INITIALISES such an object on first use. prosper does the same in `ensure_mutex` /
// `ensure_cond` (hle_kernel.cpp), which also REFUSE a destroyed slot (#2170) instead of
// manufacturing a fresh unheld object, and which install the created object atomically so two
// threads racing the first use share one host object (#793, #2176).
//
// These two declarations exist so the C11 `_Mtx_*` / `_Cnd_*` handlers in hle_kernel_time.cpp reach
// the SAME resolution rather than carrying a second, weaker copy of it. That is not tidiness: on
// hardware the C11 spelling resolves its slots through exactly this code, because the guest's own
// wrappers do not resolve anything themselves. Read out of the shipped libc.prx of PPSA24651 for
// #2596, each PLT thunk taken to its import through the JMPREL slot rather than by adjacency:
//
//   _Cnd_wait  @0x5670 (13 bytes)  push rbp; mov rbp,rsp; call 0x115640; xor eax,eax; pop rbp; ret
//                                  0x115640 -> GOT 0x192378 -> WKAXJ4XBPQ4 (scePthreadCondWait)
//   _Mtx_lock  @0x5e80 (29 bytes)  … call 0x115510; cmp eax,0x8002000b; setne cl; add ecx,3; …
//                                  0x115510 -> GOT 0x1922e0 -> 9UK1vLZQft4 (scePthreadMutexLock)
//
// The two differ in what they do with the RESULT — `_Cnd_wait` discards it, `_Mtx_lock` maps it
// onto a `_Thrd_*` code — and that difference is not the point here. The point is what neither of
// them does: **inspect the slot**. Each is a single call with the guest's slot pointer passed
// straight through, so every question about "is this object initialised, destroyed, or a static
// sentinel" is answered inside libkernel. A private `if (*slot)` guard in prosper's C11 handler
// answers a DIFFERENT question: it reads a statically initialised object as absent and skips the
// operation entirely (#2596).
#pragma once

#include <cstdint>
#include <pthread.h>

namespace prosper {

// `slot_addr` is the GUEST address of the pointer-sized slot, not the object. Returns nullptr when
// there is nothing to operate on: a null slot address, or a slot holding the destroyed sentinel
// (which also logs, once per prefix — see pt_report_destroyed). Any other value either already
// names a host object or is a static initialiser, and both come back as a usable object.
pthread_mutex_t* guest_mutex_from_slot(uint64_t slot_addr);
pthread_cond_t*  guest_cond_from_slot(uint64_t slot_addr);

}  // namespace prosper
