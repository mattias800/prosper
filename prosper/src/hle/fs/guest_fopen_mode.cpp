#include "hle/fs/guest_fopen_mode.hpp"

namespace prosper {

// The grammar is the guest libc's `_Foprep` mode loop; see the header for the disassembly.
GuestFopenMode parse_guest_fopen_mode(const char* mode, FopenHostDialect dialect) {
    GuestFopenMode m;
    if (!mode) return m;   // prosper's choice: the guest would dereference it (header)
    const char base = mode[0];
    if (base != 'r' && base != 'w' && base != 'a') return m;
    m.readable = base == 'r';
    m.writable = base != 'r';
    m.append = base == 'a';
    m.create = base != 'r';

    // '+' once (it stops being accepted once read AND write are both set), 'b' once; any other
    // character ends the loop, and only that terminating character can be 'x'.
    bool binary = false;
    const char* p = mode + 1;
    for (;; ++p) {
        if (*p == '+' && !(m.readable && m.writable)) {
            m.readable = m.writable = true;
        } else if (*p == 'b' && !binary) {
            binary = true;
        } else {
            break;
        }
    }
    m.exclusive = *p == 'x';

    if (m.exclusive && base == 'a' && dialect == FopenHostDialect::MicrosoftCrt)
        return GuestFopenMode{};   // CONFIDENCE: LOW — see the header

    // Canonical order: base, '+', 'b', 'x'. Every host accepts these letters in this order. 'b' is
    // always present because the guest never has text mode; 'x' only where the open creates.
    char* out = m.host;
    *out++ = base;
    if (m.readable && m.writable) *out++ = '+';
    *out++ = 'b';
    if (m.exclusive && m.create) *out++ = 'x';
    *out = '\0';
    m.valid = true;
    return m;
}

}   // namespace prosper
