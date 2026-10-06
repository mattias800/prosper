// cond_indirect_buffer.hpp -- execution helpers for sceAgcCbBranch's packet, for refused Jump
// segments (#4540) and for the predication decision of a packet-predicated Jump. Kept beside
// command_processor.cpp, which is past the ratchet's line cap.
#pragma once

#include "gpu/pm4/pm4_decode.hpp"

#include <cstdint>

namespace prosper::gpu {

// Packet modes (dword1 bits 1:0), as AMD's COND_INDIRECT_BUFFER defines them: 1 = if-then (a false
// condition runs nothing), 2 = if-then-else. 0 and 3 are reserved. Every branch traced on Kena
// (PPSA01802, 331 calls) is mode 1, function 0.
inline constexpr uint32_t kCibModeIfThen = 1;
inline constexpr uint32_t kCibModeIfThenElse = 2;

// The Jump command a CondIndirectBuffer resolves to when the command processor reaches it: the
// then-target when the condition holds; otherwise the else-target in mode 2 and nothing in mode 1.
// A reserved mode runs neither target and is reported. A Jump with no address or no dwords is a
// no-op. Function 0 is "always"; functions 1-6 compare (mem64 & mask) <, <=, ==, !=, >=, >
// reference, read at this moment, and are reported because no title has been observed using them
// (CONFIDENCE: LOW for functions 1-6 and for refusing the reserved modes; HIGH for mode 1/2, func 0).
Pm4Command cond_indirect_buffer_jump(const Pm4Command& c);

// The Jump segment guards, each reporting a refusal before the caller drops the segment: every draw
// and fence inside a refused segment is lost, so it must not disappear silently.
//   within limits: at most 0x40000 dwords (1 MiB, far past any real segment) and nesting depth < 8.
//   readable: the whole segment is mapped guest memory.
bool jump_segment_within_limits(uint64_t addr, uint32_t dwords, uint32_t depth);
bool jump_segment_readable(uint64_t addr, uint32_t dwords);

// The control word of prosper's SetPredication packet (dword 3). sceAgcDcbSetPredication(dcb, a1, op,
// a3, cond) carries the PRED_OP in `op` and two flag arguments that hold the hardware's PRED_BOOL
// (DRAW_VISIBLE / DRAW_NOT_VISIBLE) and HINT -- which one is which is not identified, because every
// observed call passes 1 for both. All three are kept: op in bits 0-7, a1 in 8-15, a3 in 16-23
// (a flag wider than 8 bits saturates to 0xff, an unobserved shape), and bit 31 marks a word that
// carries the flags at all. A word without bit 31 is a packet from a capture recorded before the
// flags were kept.
inline constexpr uint32_t kSetPredicationFlagsPresent = 0x80000000u;
inline constexpr uint32_t pack_set_predication_control(uint64_t op, uint64_t a1, uint64_t a3) {
    const auto byte = [](uint64_t v) { return static_cast<uint32_t>(v > 0xffu ? 0xffu : v); };
    return byte(op) | (byte(a1) << 8) | (byte(a3) << 16) | kSetPredicationFlagsPresent;
}

// Whether a packet-predicated Jump is skipped under a SetPredication window with control word
// `control` (above), given the 64-bit condition word read at fold time. The polarity comes from the
// PRED_BOOL flag, not from the op: see the evidence block in cond_indirect_buffer.cpp. Every shape
// other than the observed BOOL64 (1, 3, 1) is reported once per distinct shape.
bool predicated_jump_skips(uint32_t control, uint64_t cond);

// A predicated Jump whose condition word is misaligned or unreadable runs (the fold cannot read a
// word to decide on). Reported once per address, because under BOOL64 that is a fail-open choice.
void report_unreadable_predicate_condition(uint64_t addr);

}   // namespace prosper::gpu
