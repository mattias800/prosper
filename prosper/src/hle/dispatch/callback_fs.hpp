#pragma once

#include <cstdint>

namespace prosper {

// The guest import stub's frame, as seen from an HLE entry's %rsp. This describes ONE of the two
// shapes prosper emits -- `emit_swap_stub` (exec_image_linux.cpp): return-to-stub at +0, the four
// forwarded stack arguments 7/8/9/10 at +8/+0x10/+0x18/+0x20, saved r11 (the CALLING host thread's
// guest %fs) at +0x28, and the guest return address at +0x30. Its five pushes are what align the
// handler's entry; there is no dead slot in it.
//
// The OTHER shape is `prosper_hle_hook_swap_trampoline` (same file), used by imports that carry a
// return hook. Its real guest-%fs slot is at +0x38, and +0x28 holds a COPY of it -- so the +0x28
// test alone does NOT distinguish them. What excludes a hooked frame here is the aperture check
// below: the trampoline reaches the handler through `callq *%r10`, so the qword at +0 is a return
// address in prosper's own text rather than in the emitted-stub aperture. Today no fiber NID has a
// return hook (only the AGC submit NIDs do), so nothing reaches that path; if one ever gains one,
// this must grow a shape discriminator rather than silently repairing the spill slot.
//
// Host-context tail calls have no such frame at all.
//
// VERIFIED BY EXECUTION, not by reading: in review of #3637 both stubs' emitted bytes were run in a
// standalone program against a fake guest TCB and the frame captured live inside the handler --
// +0x28 carries the TCB for the swap stub, and the epilogue restores %fs from that slot. Note that
// agreement is NOT machine-checked by any test; see test_fiber_guest_tp_stash.cpp.
inline constexpr uint64_t kStubGuestFsSlotOffset = 0x28;

// Address of the slot holding the stashed guest %fs, or 0 when this is not a swap-stub frame. The
// checks mirror the stub's own TCB test so a malformed frame fails closed instead of letting a
// forwarded argument be read -- or written -- as a thread pointer.
inline uint64_t callback_guest_fs_slot_from_entry_stack(uint64_t entry_rsp) {
    uint64_t shim_ret = entry_rsp ? *(const uint64_t*)(uintptr_t)entry_rsp : 0;
    if (shim_ret < 0x600000000ull || shim_ret >= 0x700000000ull) return 0;

    const uint64_t slot = entry_rsp + kStubGuestFsSlotOffset;
    uint64_t fs = *(const uint64_t*)(uintptr_t)slot;
    if (fs < 0x10000 || (fs & 0xfull) || *(const uint64_t*)(uintptr_t)fs != fs ||
        *(const uint32_t*)(uintptr_t)(fs + 0x108) != 0x50524F53u)
        return 0;
    return slot;
}

// Recover the guest %fs saved by exec_image_linux.cpp's guest import-stub path. Returns 0 for a
// host-context tail call or a malformed frame.
inline uint64_t callback_guest_fs_from_entry_stack(uint64_t entry_rsp) {
    const uint64_t slot = callback_guest_fs_slot_from_entry_stack(entry_rsp);
    return slot ? *(const uint64_t*)(uintptr_t)slot : 0;
}

// Re-point a stub frame's stashed guest %fs at `own_tp` (#3615).
//
// The stash is per-HOST-THREAD; the storage it lives in is the guest stack. Those coincide only while
// the thread that enters an HLE call is the thread that returns from it, and a fiber breaks exactly
// that: a fiber suspended inside sceFiberReturnToThread/sceFiberSwitch can be resumed by ANOTHER host
// thread, which then returns through the suspending thread's frame and executes its `pop r11;
// wrfsbase r11` epilogue -- installing a foreign guest TCB. Every guest initial-exec TLS read on the
// resuming thread would then resolve to another thread's storage. Measured on Uncharted (PPSA05684),
// whose NdJob fibers all start on the main thread and are resumed by worker threads: 13 of 13 resumes
// returned on a different tid than they entered on, and the workers' `get_worker_index()` -- a plain
// `%fs:-0x80` read -- answered -1 instead of their registered index.
//
// Returns true when the slot was rewritten. A no-op (false) when there is no frame, when this thread
// has no guest TCB of its own, or when the fiber resumed on the thread that suspended it.
inline bool callback_repair_guest_fs_slot(uint64_t slot, uint64_t own_tp) {
    if (!slot || !own_tp) return false;
    uint64_t* p = (uint64_t*)(uintptr_t)slot;
    if (*p == own_tp) return false;
    *p = own_tp;
    return true;
}

} // namespace prosper
