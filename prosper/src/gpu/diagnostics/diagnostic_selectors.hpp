#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace prosper::gpu {

// Diagnostic draw selectors are semantic capture IDs.  Keep their spelling and range rules shared
// between the replay front-end and the render backend so a selector cannot recompile one draw while
// probing another.  Match the command-line tools' base-0 convention without strtoull's permissive
// whitespace/sign handling.
inline bool parse_diagnostic_uint64(std::string_view text, uint64_t& value) {
    if (text.empty() || text.front() == '+' || text.front() == '-') return false;

    unsigned base = 10;
    size_t offset = 0;
    if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        base = 16;
        offset = 2;
    } else if (text.size() > 1 && text[0] == '0') {
        base = 8;
        offset = 1;
    }
    if (offset == text.size()) return false;

    uint64_t parsed = 0;
    for (; offset < text.size(); ++offset) {
        const char c = text[offset];
        unsigned digit = 0;
        if (c >= '0' && c <= '9') digit = static_cast<unsigned>(c - '0');
        else if (c >= 'a' && c <= 'f') digit = static_cast<unsigned>(c - 'a') + 10;
        else if (c >= 'A' && c <= 'F') digit = static_cast<unsigned>(c - 'A') + 10;
        else return false;
        if (digit >= base || parsed > (std::numeric_limits<uint64_t>::max() - digit) / base)
            return false;
        parsed = parsed * base + digit;
    }
    value = parsed;
    return true;
}

inline bool parse_diagnostic_draw_id(std::string_view text, uint64_t& value) {
    return parse_diagnostic_uint64(text, value);
}

inline bool parse_diagnostic_draw_range(std::string_view text,
                                        uint64_t& first, uint64_t& last) {
    const size_t colon = text.find(':');
    if (colon == std::string_view::npos) {
        if (!parse_diagnostic_draw_id(text, first)) return false;
        last = first;
        return true;
    }
    if (text.find(':', colon + 1) != std::string_view::npos ||
        !parse_diagnostic_draw_id(text.substr(0, colon), first) ||
        !parse_diagnostic_draw_id(text.substr(colon + 1), last))
        return false;
    return true;
}

inline bool parse_fragment_tap_selector(std::string_view text,
                                        uint64_t& draw, uint32_t& pc) {
    const size_t colon = text.find(':');
    uint64_t parsed_pc = 0;
    if (colon == std::string_view::npos ||
        text.find(':', colon + 1) != std::string_view::npos ||
        !parse_diagnostic_draw_id(text.substr(0, colon), draw) ||
        !parse_diagnostic_uint64(text.substr(colon + 1), parsed_pc) ||
        parsed_pc > std::numeric_limits<uint32_t>::max())
        return false;
    pc = static_cast<uint32_t>(parsed_pc);
    return true;
}

} // namespace prosper::gpu
