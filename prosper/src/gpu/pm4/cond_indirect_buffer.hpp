// cond_indirect_buffer.hpp -- execution helpers for sceAgcCbBranch's packet and for refused Jump
// segments (#4540). Kept beside command_processor.cpp, which is past the ratchet's line cap.
#pragma once

#include "gpu/pm4/pm4_decode.hpp"

#include <cstdint>

namespace prosper::gpu {

// The Jump command a CondIndirectBuffer resolves to when the command processor reaches it: the
// then-target when the condition holds, otherwise the else-target (which may be empty: a Jump with
// no address or no dwords is a no-op). Function 0 is "always"; functions 1-6 compare
// (mem64 & mask) <, <=, ==, !=, >=, > reference, read at this moment, and are reported because no
// title has been observed using them (CONFIDENCE: LOW).
Pm4Command cond_indirect_buffer_jump(const Pm4Command& c);

// The Jump segment guards, each reporting a refusal before the caller drops the segment: every draw
// and fence inside a refused segment is lost, so it must not disappear silently.
//   within limits: at most 0x40000 dwords (1 MiB, far past any real segment) and nesting depth < 8.
//   readable: the whole segment is mapped guest memory.
bool jump_segment_within_limits(uint64_t addr, uint32_t dwords, uint32_t depth);
bool jump_segment_readable(uint64_t addr, uint32_t dwords);

}   // namespace prosper::gpu
