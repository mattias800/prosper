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

// Whether a packet-predicated Jump is skipped under a SetPredication window with raw `pred_op`, given
// the 64-bit condition word read at fold time. Op 3 (BOOL64) runs the segment only on a non-zero
// condition; other ops keep the previous rule and are reported once.
bool predicated_jump_skips(uint32_t pred_op, uint64_t cond);

}   // namespace prosper::gpu
