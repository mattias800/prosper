#pragma once

// The opaque handle allocator the PS5 service HLE hands to guest code.
//
// Every service that opens something -- libSceVideodec2, libSceAvPlayer, libSceMouse,
// libSceSaveData -- draws its handle from this one counter, so the handles a title holds at any
// moment are distinct whichever library produced them. It lives in a header because those services
// are being split out of hle_service.cpp into their own translation units (#3735) and the counter
// has to stay ONE counter across all of them.
//
// `inline`, deliberately and load-bearingly. The declaration this replaces was
// `namespace { std::atomic<uint64_t> g_handle{1}; }`, which was correct while there was exactly one
// translation unit and becomes a per-TU copy the moment a header carries it -- two libraries would
// then hand out the same number, with nothing to report it. promote_internal.py now refuses that
// shape rather than emitting `inline namespace { ... }`, which compiles and is still per-TU.
//
// Monotonic and never reset: fetch_add only. Some callers bias the result (libSceVideodec2 adds
// 0x10000) so a handle's numeric range identifies its family; that is the caller's business, not
// this counter's.

#include <atomic>
#include <cstdint>

namespace prosper {

inline std::atomic<uint64_t> g_handle{1};

}  // namespace prosper
