// Parse a comma-separated list of HEX guest addresses for a `PROSPER_*_WATCH` diagnostic.
//
// It exists because `strtoull(spec, &end, 0)` is the wrong tool and reads as the right one. Base 0 means
// "0x prefix selects hex, otherwise DECIMAL" — so a bare `2063380000` parses happily as decimal
// 2,063,380,000, the watch reports itself armed, and the routed run produces a confident zero-hit result
// about an address nobody asked about. A watch whose null is meaningless is worse than no watch.
//
// This was recorded as an instrument trap for `PROSPER_TARGET_WATCH`, and then reproduced verbatim by a
// later diagnostic whose own comment claimed `0x` was required. It was not: nothing checked the prefix.
// Hence one strict parser, used and tested rather than restated.
//
// Strict means every one of these is a REJECTION, not a best-effort parse:
//   * a token without an explicit `0x` / `0X` prefix
//   * trailing or embedded text that is not consumed
//   * an empty token, including from a stray or doubled comma
//   * a value that overflows 64 bits
//   * a zero address, which is never a guest surface and silently matches nothing
// A rejected spec yields NO addresses and returns false, so the caller reports "not armed" instead of
// arming a lie.
#pragma once

#include <cstdint>
#include <vector>

namespace prosper::gpu {

// True only when the whole spec parsed and produced at least one address. On false, `out` is empty.
bool parse_hex_watch_list(const char* spec, std::vector<uint64_t>& out);

}  // namespace prosper::gpu
