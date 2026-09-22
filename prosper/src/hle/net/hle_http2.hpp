#pragma once

#include <cstdint>

// libSceHttp2 error constants (#2894).
//
// The v2 facility is 0x817b____, NOT v1's 0x8043____. The low code words are shared between the
// two libraries, so the libSceHttp constants in hle_http.hpp are *not* reusable here -- reusing
// them would hand the guest an error from the wrong facility, which its own classifier buckets
// differently. Derivation recorded on #2894: every error-shaped immediate in the shipped
// libSceHttp2.sprx carries 0x817b, and the blocking title's classifier compares against
// 0x817b1220 and dispatches 0x817b1064..0x817b1084 through a 33-entry jump table.
//
// Two of the code words are pinned to a meaning by code rather than by a name table:
//   0x_1076  insufficient stack -- every libSceHttp2 export shares a prologue that returns it
//   0x_1100  invalid id         -- v1's context-id validator (+0xb070) answers 0x80431100 after
//                                  range-checking the id and cross-checking the table slot
// The rest are named from their v1 siblings, whose code words match.
namespace prosper::http2 {

// Called before the library was initialised / out of order. v1's 0x80431001 is the most common
// error immediate in libSceHttp and is documented in hle_http.cpp as BEFORE_INIT.
// CONFIDENCE: MED (code word attested in the v2 module; the exact name is read across from v1).
constexpr uint32_t kErrorBeforeInit = 0x817b1001u;

// No free slot in one of prosper's object tables. CONFIDENCE: MED (v1 sibling 0x80431022 is
// SCE_HTTP_ERROR_OUT_OF_MEMORY, and 0x817b1022 is the most frequent immediate in libSceHttp2).
constexpr uint32_t kErrorOutOfMemory = 0x817b1022u;

// An id that was never allocated, or 0. CONFIDENCE: HIGH on the semantic (read off v1's validator
// on the create path), MED on the facility carry-across.
constexpr uint32_t kErrorInvalidId = 0x817b1100u;

// A malformed argument, and also prosper's answer for a call whose result it cannot model without
// fabricating data (see hle_http2.cpp's "not modelled" note). CONFIDENCE: MED.
constexpr uint32_t kErrorInvalidValue = 0x817b11feu;

// What the send path answers offline is NOT an HTTP error: libSceHttp2 propagates the raw
// libSceNet error out of a failed connect, so it is net::kNetErrorNetUnreach from
// hle/net/sce_net_errors.hpp, which owns that facility and records the evidence for it (#3545).

} // namespace prosper::http2
