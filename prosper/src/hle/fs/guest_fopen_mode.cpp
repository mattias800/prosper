#include "hle/fs/guest_fopen_mode.hpp"

namespace prosper {

GuestFopenMode parse_guest_fopen_mode(const char* mode, FopenHostDialect dialect) {
    GuestFopenMode m;
    if (!mode) return m;
    const char base = mode[0];
    if (base != 'r' && base != 'w' && base != 'a') return m;
    m.readable = base == 'r';
    m.writable = base != 'r';
    m.append = base == 'a';

    bool done = false;
    for (const char* p = mode + 1; *p && !done; ++p) {
        switch (*p) {
            case '+': m.readable = m.writable = true; break;
            case 'b': break;
            case 'x':
                if (!m.writable) return GuestFopenMode{};
                m.exclusive = true;
                break;
            case 'e': m.cloexec = true; break;
            default: done = true; break;
        }
    }
    if (m.exclusive && base == 'a' && dialect == FopenHostDialect::MicrosoftCrt)
        return GuestFopenMode{};

    // Canonical order: base, '+', 'b', 'x', then the close-on-exec letter. Every host accepts
    // these letters in this order; 'b' is always present because the guest never has text mode.
    // 'x' is only spelled where it creates: on an "r+" stream the file must already exist, so
    // exclusivity has nothing to refuse, and the Microsoft CRT rejects 'x' after 'r'.
    char* out = m.host;
    *out++ = base;
    if (m.readable && m.writable) *out++ = '+';
    *out++ = 'b';
    if (m.exclusive && base != 'r') *out++ = 'x';
    if (m.cloexec) *out++ = dialect == FopenHostDialect::MicrosoftCrt ? 'N' : 'e';
    *out = '\0';
    m.valid = true;
    return m;
}

}   // namespace prosper
