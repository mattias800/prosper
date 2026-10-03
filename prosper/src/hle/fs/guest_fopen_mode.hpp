// guest_fopen_mode.hpp — a guest `fopen` mode string, read by the guest's rules and re-spelled for
// the host C library.
//
// Scope: the linker binds each title's `fopen` import to its OWN sce_module/libc.prx export first
// (linker.cpp pass 2, "Cross-module export beats a local slot"), and every local dump ships one. So
// this parser is reached only by a title that ships no libc.prx; the others run the guest's own
// fopen, which reaches prosper at `_open`, not here.
//
// The guest's stdio is Dinkumware's, not FreeBSD's. Evidence, in the project's evidence order:
//   - Guest disassembly. PPSA24651's sce_module/libc.prx: `fopen` (+0x562a0) is
//     `_Locksyslock; _Fofind; _Foprep(name, mode, str, -1, 0, 0)`, and `_Foprep` (+0x562f0) parses
//     the mode as follows.
//       * mode[0] (+0x563a5, read with no null check): 'r' is read; 'w' is write|truncate|create;
//         'a' is write|append|create. Anything else stores EINVAL in errno (+0x5643a), frees the
//         stream and returns NULL.
//       * Then a loop (+0x56448) over the following characters: '+' (+0x56450) adds read|write
//         unless both are already set; 'b' (+0x56455) sets a binary bit unless it is already set.
//         ANY other character — a second '+', a second 'b', an unknown letter, NUL — ends the
//         loop, and nothing after it is read.
//       * The character that ended the loop sets the exclusive flag iff it is 'x' (+0x5646a,
//         `cmp $0x78; sete; shl $6`). No mode is ever refused because of 'x'.
//       * `_Fopen` (+0x56560) maps the flags to open(2) flags — read/write through a table
//         {O_RDONLY, O_WRONLY, O_RDWR} (+0x15a190), create and exclusive `<<5` (O_CREAT 0x200,
//         O_EXCL 0x800), truncate `<<7` (O_TRUNC 0x400), append `<<1` (O_APPEND 0x8) — and calls
//         `_open` with mode 0666. So "rx" opens read-only with O_EXCL and no O_CREAT.
//   - Firmware symbol data. PS5 3.20 libSceLibcInternal exports Dinkumware's stdio internals:
//     `_Foprep` (dREVnZkAKRE), `_Fopen` (sQL8D-jio7U), `_Fofind`, `_Frprep`, `_Fwprep`.
//   - Census. Across the 72 local dumps (all ship libc.prx), `_Foprep`'s loop compares 'x' in 47
//     copies and not at all in 25, which otherwise run the same loop; no copy compares 'e'. This
//     parser follows the 47: in the other 25 an 'x' merely ends the mode, the same as here except
//     that no exclusivity is asked for.
//
// prosper's own choices, which are NOT the guest's contract:
//   - A null mode is EINVAL. The guest dereferences it unconditionally; refusing is the lenient
//     answer an HLE can give instead of faulting.
//   - 'e' is not a guest letter: like any unknown letter it ends the mode. No close-on-exec is set
//     on the host stream (a PS5 guest cannot exec, and `_open`'s handler sets none either).
//   - The host spelling ALWAYS carries 'b': the guest has no text mode. Forwarding a mode verbatim
//     is wrong on Windows, where without 'b' the Microsoft CRT translates CR LF and stops at 0x1A,
//     and where a mode it rejects raises its invalid-parameter handler instead of returning EINVAL.
//   - Exclusivity is spelled only where the open creates ('w' and 'a'). On 'r' the guest passes
//     O_EXCL without O_CREAT, which POSIX leaves undefined and which neither the guest's nor the
//     host's open uses to refuse an existing file; the Microsoft CRT rejects 'x' after 'r'.
//   - "ax"/"a+x" on Windows: the Microsoft CRT accepts 'x' only with 'w', so the mode is refused
//     with EINVAL rather than opened without the exclusivity the guest asked for.
//
// CONFIDENCE: HIGH for the grammar (read from the guest's own disassembly at the offsets above).
// MED that a title without libc.prx would use the 'x'-aware variant rather than the older one.
// LOW for the Windows "ax" refusal.
#pragma once
#include "host/platform/host_crt.hpp"
#include <cstdint>

namespace prosper {

enum class FopenHostDialect : uint8_t {
    Posix,          // glibc / Darwin: 'b' is accepted and ignored
    MicrosoftCrt,   // UCRT / msvcrt: 'b' selects binary mode; 'x' only after 'w'
};

inline constexpr FopenHostDialect kHostFopenDialect = host::kCrtFamily == host::CrtFamily::Microsoft
                                                          ? FopenHostDialect::MicrosoftCrt
                                                          : FopenHostDialect::Posix;

struct GuestFopenMode {
    bool valid = false;   // false: fopen fails with EINVAL and opens nothing
    bool readable = false;
    bool writable = false;
    bool append = false;
    bool create = false;      // 'w' or 'a': the guest's open carries O_CREAT
    bool exclusive = false;   // the loop ended on 'x': the guest's open carries O_EXCL
    char host[8] = {};        // the mode to hand the host's fopen; empty when !valid
};

GuestFopenMode parse_guest_fopen_mode(const char* mode, FopenHostDialect dialect);

}   // namespace prosper
