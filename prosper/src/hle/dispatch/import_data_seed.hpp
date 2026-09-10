// import_data_seed.hpp — the initial value of an unresolved Sony DATA import.
//
// The dispatcher answers unresolved FUNCTION imports; this answers the handful of unresolved
// *variables*. Both are the same question — "prosper is this Sony library, what does it provide?" —
// asked about an object instead of an entry point, which is why it lives beside the dispatcher.
//
// Everything not listed here stays ZERO. Zero is the honest default for a variable whose contents
// prosper does not know: it is stable, it is what a reader gets from a fresh page, and for a
// pointer-shaped object it makes the guest take its own null branch instead of dereferencing
// whatever bytes happened to be there.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace prosper {

// The stack canary prosper supplies through `__stack_chk_guard`.
//
// WHY A VALUE AT ALL, AND WHY THIS ONE. `__stack_chk_guard` is a libkernel *variable*, and libkernel
// is prosper — so supplying it is reimplementation, not invention. Every function a title compiled
// with `-fstack-protector` stores it into its frame on entry and compares on return, so the only
// property the contract needs is that both reads agree; the platform additionally makes it non-zero
// and per-process random.
//
// Non-zero rather than zero is the choice that PRESERVES measured behaviour. Before #3529 this
// import was bound into the code-stub aperture, so every title in the local corpus — all 60 of
// them, 728 bindings — has been reading prosper's own machine code here and booting with it. That
// value is non-zero and stable, which is exactly why a defect this universal stayed invisible.
// Leaving the slot zero would therefore have been the novel change, and it would also silently
// disable `-fstack-protector` in every guest.
//
// The low byte is 0x00 on purpose (the "terminator canary" convention): a string-based overflow that
// runs through the canary cannot copy past it without writing the NUL. The remaining bytes spell
// "PROSPER" in memory order, so a canary slot is recognisable on sight in a stack dump. Fixed rather
// than randomised so a run is reproducible — prosper has no attacker model, and a deterministic
// boot is worth more here than an unpredictable canary.
inline constexpr uint64_t kStackGuardValue = 0x52455053'4F525000ull;
// little-endian bytes, in memory order: 00 'P' 'R' 'O' 'S' 'P' 'E' 'R'

// Write the initial bytes of the data import `nid` into `slot` (`size` bytes available).
// Returns true when a value was written, false when the slot should be left as it is (zero).
// `slot` is assumed zero-filled on entry, so a partial write is still well-defined.
bool seed_import_data(const std::string& nid, uint8_t* slot, size_t size);

} // namespace prosper
