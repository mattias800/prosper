#pragma once
// The kevent `data` word of a VideoOut FLIP event, and how sceVideoOutGetEventData decodes it (#3668).
//
// Producer (prosper_eq_trigger_flip, hle_kernel_time.cpp) and accessor (sceVideoOutGetEventData,
// hle_graphics.cpp) both go through this header, so the two halves of the contract cannot drift.
//
// CONTRACT: the submitted flipArg lives in bits 16..63 of the kernel event's data; bits 0..15 carry
// no flipArg bits. sceVideoOutGetEventData returns `data >> 16`, arithmetic, so a negative flipArg
// (the common -1 "no argument") round-trips, and a flipArg is reported through 48 significant bits.
//
// EVIDENCE (guest disassembly, the top of the charter's hierarchy):
//   Uncharted: Legacy of Thieves Collection (PPSA05684) reads the raw kernel data of its flip event
//   and decodes it inline, exactly as the VideoOut accessor would:
//     eboot+0x15b31e4  call sceKernelGetEventFilter ; cmp eax,0xfffffff3  -- -13, EVFILT_VIDEO_OUT
//     eboot+0x15b3229  call sceKernelGetEventData   ; test rax,rax ; js retry
//     eboot+0x15b3239  sar  r14,0x10                ; completed frame = data >> 16 (ARITHMETIC)
//     eboot+0x15b3279  cmp  rbx,0x10000 ; jl retry  ; data below bit 16 carries no frame
//   With the flipArg in the low bits (prosper's previous layout) `data >> 16` is always 0 for a
//   title whose flipArgs are small ordinals, its completed-frame high-water advances once and sticks,
//   and it deadlocks after three flips.
//   The firmware exports sceVideoOutGetEventCount beside GetEventData/GetEventId
//   (PS5-3.20_Libs/libSceVideoOut.c, Mt4QHHkxkOc), i.e. the event word carries more than the argument,
//   which is what a flipArg displaced into the high bits leaves room for.
// CROSS-TITLE (static, 2026-09-22): of the 38 local dumps that call sceVideoOutAddFlipEvent, none
//   imports sceVideoOutGetEventData, and 11 import sceKernelGetEventData. Only Uncharted has a
//   VideoOut filter compare (`cmp eax,0xfffffff3`) beside its call site. Eight of the other ten
//   (Unreal titles) decode the word as `queue << 58 | counter`, with no such compare nearby; their
//   queue is not traced here. PPSA05143 and PPSA17168 were not decoded.
//
// CONFIDENCE: HIGH on bits 16+ and the arithmetic decode (the guest's own arithmetic pins both).
// CONFIDENCE: LOW on bits 0..15. The guest above requires only that they carry no frame; what Sony
// stores there (a coalesced-event count for sceVideoOutGetEventCount is the likely candidate) is not
// established by any title here, so prosper leaves them zero rather than invent a value.
#include <cstdint>

namespace prosper::videoout {

inline constexpr int kFlipEventDataShift = 16;

// kevent.data for a completed flip carrying `flip_arg`.
inline constexpr int64_t pack_flip_event_data(int64_t flip_arg) {
    return (int64_t)((uint64_t)flip_arg << kFlipEventDataShift);
}

// sceVideoOutGetEventData's answer for a flip event's kevent.data.
inline constexpr int64_t flip_arg_from_event_data(int64_t data) {
    return data >> kFlipEventDataShift;   // arithmetic on int64_t (C++20), as the guest's `sar`
}

} // namespace prosper::videoout
