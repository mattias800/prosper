#include "watch_list.hpp"

#include <cctype>
#include <cstdlib>

namespace prosper::gpu {
namespace {

bool hex_digit(char c, uint64_t& value) {
    if (c >= '0' && c <= '9') { value = static_cast<uint64_t>(c - '0'); return true; }
    if (c >= 'a' && c <= 'f') { value = static_cast<uint64_t>(c - 'a') + 10u; return true; }
    if (c >= 'A' && c <= 'F') { value = static_cast<uint64_t>(c - 'A') + 10u; return true; }
    return false;
}

}  // namespace

bool parse_hex_watch_list(const char* spec, std::vector<uint64_t>& out) {
    out.clear();
    if (!spec || !*spec) return false;

    const char* p = spec;
    while (true) {
        // Leading spaces are tolerated so a shell-quoted list stays usable; everything else is not.
        while (*p == ' ' || *p == '\t') ++p;

        // An explicit 0x/0X prefix is REQUIRED. This is the whole point: without it a decimal token
        // parses silently into an unrelated address.
        if (p[0] != '0' || (p[1] != 'x' && p[1] != 'X')) { out.clear(); return false; }
        p += 2;

        // At least one hex digit, accumulated with an overflow check rather than trusting strtoull's
        // errno dance.
        uint64_t value = 0;
        uint64_t digit = 0;
        size_t digits = 0;
        while (hex_digit(*p, digit)) {
            if (value > (UINT64_MAX >> 4)) { out.clear(); return false; }   // would shift out the top
            value = (value << 4) | digit;
            ++digits;
            ++p;
        }
        if (!digits) { out.clear(); return false; }

        while (*p == ' ' || *p == '\t') ++p;

        // A zero address is never a guest surface; accepting it arms a watch that matches nothing while
        // reporting itself armed.
        if (!value) { out.clear(); return false; }
        out.push_back(value);

        if (*p == '\0') return true;                 // whole spec consumed
        if (*p != ',') { out.clear(); return false; } // trailing junk after a valid token
        ++p;
        if (*p == '\0') { out.clear(); return false; } // trailing comma: an empty final token
    }
}

}  // namespace prosper::gpu
