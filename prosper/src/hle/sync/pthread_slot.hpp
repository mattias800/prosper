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
// onto a `_Thrd_*` code (that mapping is `thrd_rc_from_mutex_lock` below, landed by #2626; it was
// still a discard when this paragraph was written) — and that difference is not the point here,
// which is why #2596 deferred it. The point is what neither of
// them does: **inspect the slot**. Each is a single call with the guest's slot pointer passed
// straight through, so every question about "is this object initialised, destroyed, or a static
// sentinel" is answered inside libkernel. A private `if (*slot)` guard in prosper's C11 handler
// answers a DIFFERENT question: it reads a statically initialised object as absent and skips the
// operation entirely (#2596).
//
// SHARING THE RESOLUTION WAS NOT ENOUGH (#2619, #2623). #2596 gave the C11 handlers the same slot
// RESOLUTION as their Sony counterparts and left each one keeping its own (absent) per-object
// BOOKKEEPING: the destroy-busy waiter count (#2168), the missed-wakeup generation, the Windows
// ownership map, and the claim-and-poison of a destroyed slot (#2176/#2170 -- the half #2619 is
// about). Every one of those is the #1873 shape, the same question answered differently depending
// on which spelling the guest happened to use, and the C11 spelling could reach them all the moment
// its slots started resolving.
//
// So the family below publishes WHOLE OPERATIONS rather than resolvers, and both spellings call
// them. That is not tidiness either: the guest's own wrappers ARE the Sony entry points. Read out
// of the shipped libc.prx of PPSA24651, each PLT thunk taken to its import through the JMPREL slot
// (`tools/re/stub_nid_map.py`) rather than by adjacency, and each body decoded with
// `tools/re/edis.py`:
//
//   export         addr    size  forwards to                what it does with the result
//   _Mtx_init      0x5dc0  0x73  scePthreadMutexattr{Init,Settype,Destroy}, scePthreadMutexInit
//                                                           maps onto a _Thrd_* code
//   _Mtx_lock      0x5e80  0x1d  scePthreadMutexLock        maps: 0->0, 0x8002000b->3, else 4
//   _Mtx_unlock    0x5e70  0x0d  scePthreadMutexUnlock      xor eax,eax -- DISCARDS
//   _Mtx_destroy   0x5e60  0x0b  scePthreadMutexDestroy     nothing after the call touches eax
//   _Cnd_init      0x5560  0x9d  scePthreadCondInit         maps onto a _Thrd_* code
//   _Cnd_signal    0x56d0  0x0d  scePthreadCondSignal       xor eax,eax -- DISCARDS
//   _Cnd_broadcast 0x56e0  0x0d  scePthreadCondBroadcast    xor eax,eax -- DISCARDS
//   _Cnd_wait      0x5670  0x0d  scePthreadCondWait         xor eax,eax -- DISCARDS
//   _Cnd_destroy   0x5660  0x0b  scePthreadCondDestroy      nothing after the call touches eax
//   _Thrd_yield    0x4ee0  0x0b  scePthreadYield            nothing after the call touches eax
//                                                           (not a sync primitive -- it is here as
//                                                            the CONTROL for the block below)
//
// Every wrapper is a single call with the guest's slot pointer passed straight through, and not one
// of them inspects the slot. So a C11 handler whose BODY differs from the Sony handler's body is a
// divergence by construction, whatever it returns.
//
// THE DESTROY PAIR IS THE ROW THAT SETTLES #2619's OPEN QUESTION, and the answer is that their
// RESULT IS NOT A CONTRACT: prosper answers 0 through both spellings, and nothing in the lifetime
// behaviour below depends on the question either way.
//
// An earlier revision of this block read the eleven bytes as "they FORWARD the libkernel result",
// reasoning that the missing `xor eax,eax` had to mean the callee's value was being passed on. That
// inference is VOID, and the counterexample is in the table above: an eleven-byte
// `push rbp; mov rbp,rsp; call; pop rbp; ret` is exactly what a VOID-RETURNING wrapper compiles
// to. The compiler emits no `xor eax,eax` for a function with no return value, so the absence --
// which was the entire argument -- is equally well explained by there being nothing to return.
// Recorded rather than quietly deleted because the shape is genuinely suggestive and the next
// reader will re-derive it (#2636).
//
// THE ELEVEN-BYTE SHAPE CARRIES NO INFORMATION ABOUT THE RETURN TYPE AT ALL, and this module proves
// it in BOTH directions with two functions sixteen bytes apart:
//
//     _Thrd_yield   @0x4ee0  554889e5 e8f7061100 5dc3  -> 0x1155e0  T72hz6ffq08  scePthreadYield
//     _Thrd_equal   @0x4ef0  554889e5 e8f7061100 5dc3  -> 0x1155f0  3PtV6p3QNX4  scePthreadEqual
//
// BYTE-FOR-BYTE IDENTICAL bodies (the call displacements coincide because both the wrapper pair and
// the stub pair are 0x10 apart, so both encode disp 0x1106f7). ISO C11 spells one
// `void thrd_yield(void)` and the other `int thrd_equal(thrd_t, thrd_t)` -- and `_Thrd_equal` has no
// way to return its value except by leaving `scePthreadEqual`'s result in eax. So one of these
// eleven-byte bodies has nothing to forward and the other certainly does forward, and nothing in the
// bytes distinguishes them. That is a complete proof rather than a counterexample, and it is why the
// inference above is void.
//
// THE OBVIOUS COMPETING READING -- "the compiler just emits identical bytes for every eleven-byte
// forwarder, so the identity means nothing" -- IS FALSIFIED BY THE NEXT PAIR DOWN, and the
// arithmetic above predicts it. `_Thrd_current` @0x4f10 and `_Thrd_id` @0x4f20 are also 0x10 apart,
// but they SHARE one stub (both call 0x115600 = `aI+OeCz8xrQ` scePthreadSelf), so their
// displacements differ by exactly 0x10 and their bodies are NOT identical:
//
//     _Thrd_current 0x4f10  554889e5 e8e70611 00 5dc3   disp 0x1106e7  -> 0x115600
//     _Thrd_id      0x4f20  554889e5 e8d70611 00 5dc3   disp 0x1106d7  -> 0x115600
//
// Same prologue, same shape, different bytes. So the yield/equal identity is not a compiler
// signature that all these bodies share -- it is a consequence of the two wrappers and their two
// distinct stubs being equally spaced, which is a prediction this module tests and confirms.
//
// It is not a lone oddity. Every eleven-byte forwarder in the module's C11 family, with its ISO
// counterpart's return type -- these are the ones checked, listed so the claim is falsifiable rather
// than asserted over a set someone assembled:
//
//     _Thrd_yield   0x4ee0 -> scePthreadYield         void          (nothing to forward)
//     _Thrd_equal   0x4ef0 -> scePthreadEqual         int           FORWARDS
//     _Thrd_current 0x4f10 -> scePthreadSelf          thrd_t        FORWARDS
//     _Thrd_id      0x4f20 -> scePthreadSelf          (Dinkumware)  forwards
//     _Mtx_destroy  0x5e60 -> scePthreadMutexDestroy  void
//     _Cnd_destroy  0x5660 -> scePthreadCondDestroy   void
//     _Tss_create  0x7f3a0 -> scePthreadKeyCreate     int           FORWARDS
//     _Tss_delete  0x7f3b0 -> scePthreadKeyDelete     void          (nothing to forward)
//     _Tss_set     0x7f3c0 -> scePthreadSetspecific   int           FORWARDS
//     _Tss_get     0x7f3d0 -> scePthreadGetspecific   void*         FORWARDS
//
// THE `_Tss_*` HALF WAS ALREADY IN THIS REPOSITORY. `hle_kernel.cpp` (above
// `SCE_PTHREAD_ALIAS(k_sce_key_delete, ...)`) records that `_Tss_create`, `_Tss_delete` and
// `_Tss_set` are "11-byte, five-instruction forwarders ... returning eax unexamined", verified by
// PLT relocation for PPSA24651. Two of those return `int` in ISO C11. The counterexample never
// required leaving the tree, and an earlier revision of THIS block nevertheless asserted a universal
// that the same source file contradicts 1,600 lines away (#2636).
//
// WHAT DOES SURVIVE, and it is the load-bearing line: NO CALLER READS THE DESTROY PAIR'S RESULT.
// Over every direct `E8` in `.text` (0..0x115bf0) of the flattened module, `_Cnd_destroy` has 14
// call sites and `_Mtx_destroy` 16, and NOT ONE of those 30 is followed by `test eax,eax`; the
// first, at 0x504f, does `mov rax,[rip+0x18d1ad]` and clobbers eax before any use. The wrappers
// whose result IS a contract are read at most of theirs -- `_Cnd_wait` 7 of 8, `_Mtx_unlock` 12 of
// 25. That is an absence rather than a proof, and it is labelled as one below.
//
// WHAT THE ELEVEN BYTES DO ESTABLISH is the half #2168 needs, and it is not about a return value at
// all: the wrapper's single call is to `scePthreadCondDestroy`, so whatever that entry point does
// TO THE OBJECT is what happens on hardware through the C11 spelling. The EBUSY refusal is
// therefore part of `_Cnd_destroy`'s behaviour too, and a C11 destroy that retires an object a
// thread is parked in is wrong for the same reason the Sony one was. That is what
// `guest_cond_destroy_slot` enforces and what the tests assert -- on the OBJECT, not on a number.
//
// CONFIDENCE: HIGH that the eleven-byte shape does not establish forwarding. This is now a PROOF,
// not a counterexample: `_Thrd_yield` and `_Thrd_equal` are byte-identical with opposite ISO return
// types, so the shape is demonstrably ambiguous in both directions inside one module. Also HIGH on
// every byte, address, NID and count quoted above -- all re-derived from the shipped `libc.prx` with
// the tools named at the head of this block.
// CONFIDENCE: MED that the destroy pair is declared `void`, and the grounds are narrower than an
// earlier revision claimed. What supports it is exactly two things: the ISO C11 signatures of
// `mtx_destroy` and `cnd_destroy` themselves, which are `void`; and the call-site census above,
// which is an ABSENCE over one module rather than a proof. A partition over the module's C11 exports
// is NOT among the grounds -- that argument was asserted over a set assembled alongside the
// conclusion and is refuted by the table above. Nothing prosper does needs it settled: both readings
// permit the 0 both handlers answer, because a void result may be anything and a forwarded one is
// read by no caller here.
#pragma once

#include "hle/kernel/sce_errno.hpp"     // the libkernel encoding the Sony spelling and the C11 wrapper both use
#include "hle/sync/sync_retire.hpp"   // SyncObjectKind — the destroy pair keeps its own census bucket

#include <cstdint>
#include <pthread.h>

namespace prosper {

// `slot_addr` is the GUEST address of the pointer-sized slot, not the object. Returns nullptr when
// there is nothing to operate on: a null slot address, or a slot holding the destroyed sentinel
// (which also logs, once per prefix — see pt_report_destroyed). Any other value either already
// names a host object or is a static initialiser, and both come back as a usable object.
pthread_mutex_t* guest_mutex_from_slot(uint64_t slot_addr);
pthread_cond_t*  guest_cond_from_slot(uint64_t slot_addr);

// --- whole operations, shared by the Sony spelling and the C11 spelling (#2619 / #2623) ----------
// Each is the ENTIRE body of the scePthread* / pthread_* handler of the same name, bookkeeping
// included, so the two spellings cannot drift again. Results are the BARE FreeBSD errno; the
// libkernel encoding is applied by SCE_PTHREAD_ALIAS at the Sony spelling, and by the C11 wrapper's
// own result transform (the table above) at the C11 one.
uint64_t guest_mutex_lock_slot(uint64_t slot_addr);
uint64_t guest_mutex_unlock_slot(uint64_t slot_addr);
// `kind` selects the retirement CENSUS bucket only (`_Mtx` vs `mutex`); the lifetime policy is
// identical. Keeping the buckets apart is what let #2619 establish that no title on this machine
// reaches the C11 spelling at all.
uint64_t guest_mutex_destroy_slot(uint64_t slot_addr, SyncObjectKind kind);
void     guest_cond_signal_slot(uint64_t slot_addr);
void     guest_cond_broadcast_slot(uint64_t slot_addr);
// The ONE body that parks on a condition variable without a deadline, so `GuestCondWaiterScope`
// (#2168) is taken here rather than remembered by each caller. EINVAL(22) if either slot is unusable.
uint64_t guest_cond_wait_slot(uint64_t cond_slot, uint64_t mutex_slot);
// EBUSY(16) when the condvar still has waiters — checked BEFORE the slot is claimed, so a refusal
// leaves the guest a handle that still works (#2168).
uint64_t guest_cond_destroy_slot(uint64_t slot_addr, SyncObjectKind kind);

// --- the three result conventions those bodies feed (#2626) --------------------------------------
// THREE spellings reach the identical body and each reports its failure differently. Getting this
// wrong is the #1612/#1945 family, and it has cost this project a title:
//
//   POSIX  `pthread_mutex_lock`   -> the BARE FreeBSD errno, exactly what the bodies above return.
//   Sony   `scePthreadMutexLock`  -> `sce_pthread_rc` below: 0x8002_0000 | that errno.
//   C11    `_Mtx_lock`            -> a Dinkumware `_Thrd_*` code, `thrd_rc_from_mutex_lock` below.
//
// The C11 transform is DEFINED IN TERMS OF the Sony one rather than beside it, because that is how
// the guest's own wrapper computes it: it compares `scePthreadMutexLock`'s ENCODED result against
// the encoded constant. Re-derived independently for #2626 from the shipped `libc.prx` of
// PPSA24651 — flattened with `tools/il2cpp/prx_to_elf.py`, the export located by NID
// (`_Mtx_lock` = `iS4aWbUonl0`) in the dynsym at 0x5e80 with st_size 29, the body decoded with
// `tools/re/edis.py`, and the call taken to its import through the JMPREL slot with
// `tools/re/stub_nid_map.py` (0x115510 -> `9UK1vLZQft4` scePthreadMutexLock, libkernel):
//
//     push rbp; mov rbp,rsp
//     call   0x115510            ; scePthreadMutexLock
//     xor    ecx,ecx
//     cmp    eax,0x8002000b      ; SCE_KERNEL_ERROR_EDEADLK — the ENCODED form, not a bare 11
//     setne  cl
//     add    ecx,0x3             ; ecx = 3 if EDEADLK, 4 otherwise
//     test   eax,eax
//     cmovne eax,ecx             ; success passes through untouched
//     pop rbp; ret
//
// Two independent controls came with that decode and both fired: `_Mtx_unlock` (0x5e70, 13 bytes)
// reproduced its recorded `xor eax,eax` DISCARD byte for byte, so this method resolves the right
// symbol; and `_Mtx_trylock` (0x5ea0, 29 bytes) carries the SAME transform against 0x80020010
// (encoded EBUSY). That second one is what settles the NAME rather than only the number: two
// different libkernel refusals — "would deadlock" and "already held" — both land on 3, which is
// what `_Thrd_busy` means and nothing else in the enum does.
// CONFIDENCE: HIGH on the numbers (they are the guest's own bytes, quoted above) and on the names
// (Dinkumware's `_Thrd_*` ordering, already relied on by the `_Thrd_error`(4) reading recorded
// above SCE_PTHREAD_ALIAS in hle_kernel.cpp).
inline uint64_t sce_pthread_rc(uint64_t posix_rc) {
    // Pass through success, and anything a body already encoded or that is not an errno.
    if (posix_rc == 0 || (posix_rc & ~0xffull) != 0) return posix_rc;
    return hle::sce_kernel_error(static_cast<hle::FreeBsdErrno>(static_cast<uint32_t>(posix_rc)));
}

// Dinkumware's C11 result enum, in its declaration order. Only the three the mutex-lock transform
// can produce are reachable from here; the other two are spelled out so a future `_Mtx_timedlock`
// transform does not have to re-guess the ordering. Every value below is PINNED BY THE GUEST'S OWN
// BYTES rather than by analogy to a published header — each is the constant a decoded wrapper in the
// shipped `libc.prx` of PPSA24651 actually produces (#2626):
//
//   0 success   every wrapper: `test eax,eax` then `cmove`/`je` — success passes through untouched
//   1 nomem     `_Cnd_init`      0x5560: cmp ecx,0x8002000c (ENOMEM)    -> lea eax,[rax+rax*2+1]
//   2 timedout  `_Mtx_timedlock` 0x5ec0: cmp eax,0x8002003c (ETIMEDOUT) -> lea ecx,[rcx+rcx*1+2]
//   3 busy      `_Mtx_trylock`   0x5ea0: cmp eax,0x80020010 (EBUSY)     -> add ecx,3
//   4 error     all four: the failure none of them discriminates
//
// THIS ENUM IS NOT THE WHOLE RANGE, and the promise above is deliberately narrower than it once was.
// An earlier revision offered these values to a future `_Cnd_timedwait` transform as well, and that
// reader would have been one value short: `_Cnd_timedwait` (0x5680, 73 bytes, its call taken through
// the JMPREL slot to 0x115650 -> `BmMjYxmew1w` scePthreadCondTimedwait) produces a value outside
// {0..4}.
//
//     56a2: cmp eax,0x8002003c   ; ETIMEDOUT -> mov eax,2
//     56ad: test ecx,ecx         ; success   -> 0
//     56b1: cmp ecx,0x80020001   ; EPERM
//     56b7: sete al
//     56ba: or   eax,0x4         ; EPERM -> 5, anything else -> 4
//
// So its full transform is `0->0, ETIMEDOUT->2, EPERM->5, else->4`. That it discriminates EPERM at
// all agrees with the #2327 note above `guest_mutex_not_owned_by_self` in hle_kernel.cpp, which
// records Sony's own `_Cnd_timedwait` testing for it — two independent readings of the same
// behaviour. The 5 is recorded here so the follow-up does not re-derive it, and deliberately NOT
// added to the enum below: the bytes give the NUMBER, and Dinkumware's spelling for it has not been
// traced in this repository. Naming it would be the same unforced guess the block above
// `thrd_rc_from_mutex_lock` declines to make about which exception `_Thrd_busy` raises.
// CONFIDENCE: HIGH on every number, address and byte in this block (the guest's own bytes,
// re-derived with prx_to_elf.py / edis.py / stub_nid_map.py). The names carry the confidence stated
// above `sce_pthread_rc`, which is where they come from.
enum : uint64_t {
    kThrdSuccess  = 0,
    kThrdNomem    = 1,
    kThrdTimedout = 2,
    kThrdBusy     = 3,
    kThrdError    = 4,
};

// `_Mtx_lock`'s result transform, applied to a BARE body result. Note what it is NOT: "zero stays
// zero, non-zero becomes an error". The guest's wrapper spends four instructions singling EDEADLK
// out, so the two refusals are meant to be distinguishable and collapsing them is a defect even
// though both are non-zero. What the difference costs is recorded rather than inferred: the
// `_Thrd_error`(4) path is the one this repository has already traced to
// `std::_Throw_C_error(4)` -> an uncaught `std::system_error("invalid argument")` that terminates
// the guest (see the note above SCE_PTHREAD_ALIAS in hle_kernel.cpp, and #1945 where it killed
// *The Oregon Trail*). The exact exception Dinkumware raises for `_Thrd_busy` has NOT been traced
// here and is deliberately not stated — what is established is that it is a different path from the
// one that terminates.
inline uint64_t thrd_rc_from_mutex_lock(uint64_t posix_rc) {
    const uint64_t sce = sce_pthread_rc(posix_rc);
    if (sce == 0) return kThrdSuccess;
    return sce == hle::kSceKernelErrorEDEADLK ? kThrdBusy : kThrdError;
}

// Test-only. The missed-wakeup generation counter is bumped by exactly one function
// (`guest_cond_advance`, hle_kernel.cpp) whose only callers are the signal/broadcast bodies, and its
// only PRODUCTION reader is the retry loop in `interruptible_cond_clock_timedwait` — a
// slice-boundary race no test can schedule. Reading the counter is therefore the only way to assert
// that a signal was ACCOUNTED FOR, which is the first divergence #2623 records.
uint64_t guest_cond_generation_for_test(pthread_cond_t* cond);

}  // namespace prosper
