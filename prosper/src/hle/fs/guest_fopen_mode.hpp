// guest_fopen_mode.hpp — a guest `fopen` mode string, read by the guest's rules and re-spelled for
// the host C library.
//
// The guest's libc is FreeBSD's. Its mode grammar (fopen(3)): a leading 'r', 'w' or 'a'; then, in
// order, any of '+' (read and write), 'b' (ignored — FreeBSD has no text mode), 'x' (fail if the file
// exists) and 'e' (close-on-exec). Anything else as the first character is EINVAL.
//
// Forwarding that string to the host verbatim is wrong on Windows in three ways, all silent until a
// title hits them: without 'b' the Microsoft CRT opens in TEXT mode, so CRLF collapses to LF and a
// 0x1A byte reads as end-of-file in what the guest wrote as binary data (the same defect `open()`
// already guards against with O_BINARY); 'e' is not a Microsoft mode letter (its spelling is 'N'),
// and a mode the Microsoft CRT rejects raises its invalid-parameter handler, which terminates the
// process rather than returning EINVAL. So the mode is parsed once, here, and the host is handed a
// canonical spelling it is known to accept.
//
// CONFIDENCE: HIGH for the documented letters. MED for two corners fopen(3) does not state, taken
// from the FreeBSD implementation as a cross-check: an unrecognised letter after the first ENDS the
// parse rather than being skipped or rejected, and 'x' on a read-only stream is EINVAL. LOW for 'a'
// with 'x' on Windows, which the Microsoft CRT cannot spell: it is refused with EINVAL rather than
// opened without the exclusivity the guest asked for.
#pragma once
#include "host/platform/host_crt.hpp"
#include <cstdint>

namespace prosper {

enum class FopenHostDialect : uint8_t {
    Posix,   // glibc / Darwin: 'e' is close-on-exec, 'b' is accepted and ignored
    MicrosoftCrt,   // UCRT / msvcrt: 'b' selects binary mode, 'N' is non-inheritable
};

inline constexpr FopenHostDialect kHostFopenDialect = host::kCrtFamily == host::CrtFamily::Microsoft
                                                          ? FopenHostDialect::MicrosoftCrt
                                                          : FopenHostDialect::Posix;

struct GuestFopenMode {
    bool valid = false;   // false: the guest's fopen fails with EINVAL and opens nothing
    bool readable = false;
    bool writable = false;
    bool append = false;
    bool exclusive = false;
    bool cloexec = false;
    char host[8] = {};   // the mode to hand the host's fopen; empty when !valid
};

GuestFopenMode parse_guest_fopen_mode(const char* mode, FopenHostDialect dialect);

}   // namespace prosper
