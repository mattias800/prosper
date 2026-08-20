#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <system_error>

namespace prosper::frontend {

enum class ComputeTimingSelectorParseError : uint8_t {
    None,
    Unset,
    Empty,
    Sign,
    MissingDigits,
    InvalidDigit,
    TrailingCharacters,
    Overflow,
};

struct ComputeTimingSelectorParseResult {
    uint64_t value = 0;
    ComputeTimingSelectorParseError error = ComputeTimingSelectorParseError::Unset;

    constexpr bool accepted() const {
        return error == ComputeTimingSelectorParseError::None;
    }
};

// Strict unsigned parsing for diagnostic selectors. std::strtoull accepts signs and leading
// whitespace, both of which make a mistyped stable identity look armed. Decimal and 0x-prefixed
// hexadecimal are the only accepted forms; every byte must belong to the value.
inline ComputeTimingSelectorParseResult parse_compute_timing_selector_u64(
    const char* text) {
    if (!text) return {0, ComputeTimingSelectorParseError::Unset};
    const std::string_view input(text);
    if (input.empty()) return {0, ComputeTimingSelectorParseError::Empty};
    if (input.front() == '+' || input.front() == '-')
        return {0, ComputeTimingSelectorParseError::Sign};

    int base = 10;
    std::size_t prefix = 0;
    if (input.size() >= 2 && input[0] == '0' &&
        (input[1] == 'x' || input[1] == 'X')) {
        base = 16;
        prefix = 2;
    }
    if (prefix == input.size())
        return {0, ComputeTimingSelectorParseError::MissingDigits};

    uint64_t value = 0;
    const char* begin = input.data() + prefix;
    const char* end = input.data() + input.size();
    const auto parsed = std::from_chars(begin, end, value, base);
    if (parsed.ec == std::errc::result_out_of_range)
        return {0, ComputeTimingSelectorParseError::Overflow};
    if (parsed.ec == std::errc::invalid_argument)
        return {0, ComputeTimingSelectorParseError::InvalidDigit};
    if (parsed.ptr != end)
        return {0, ComputeTimingSelectorParseError::TrailingCharacters};
    return {value, ComputeTimingSelectorParseError::None};
}

inline const char* compute_timing_selector_parse_error_name(
    ComputeTimingSelectorParseError error) {
    switch (error) {
        case ComputeTimingSelectorParseError::None: return "none";
        case ComputeTimingSelectorParseError::Unset: return "unset";
        case ComputeTimingSelectorParseError::Empty: return "empty";
        case ComputeTimingSelectorParseError::Sign: return "sign-not-allowed";
        case ComputeTimingSelectorParseError::MissingDigits: return "missing-digits";
        case ComputeTimingSelectorParseError::InvalidDigit: return "invalid-digit";
        case ComputeTimingSelectorParseError::TrailingCharacters: return "trailing-characters";
        case ComputeTimingSelectorParseError::Overflow: return "overflow";
    }
    return "unknown";
}

struct ComputeTimingSelector {
    bool address_enabled = false;
    bool address_valid = true;
    uint64_t address = 0;
    bool hash_requested = false;
    bool hash_valid = true;
    uint64_t hash = 0;
};

// Address and stable hash are an AND when both are present. An invalid explicitly requested filter
// fails closed; silently falling back to an unfiltered trace would turn an apparatus error into a
// very large, plausibly useful-looking log.
constexpr bool compute_timing_selector_matches(const ComputeTimingSelector& selector,
                                                uint64_t code_addr,
                                                uint64_t program_hash) {
    if (!selector.address_valid || !selector.hash_valid) return false;
    return (!selector.address_enabled || code_addr == selector.address) &&
           (!selector.hash_requested || program_hash == selector.hash);
}

struct ComputeTimingSelectorCounters {
    uint64_t seen = 0;
    uint64_t matched = 0;
    bool summary_reported = false;
};

struct ComputeTimingSelectorObservation {
    bool matched = false;
    bool first_match = false;
};

constexpr uint64_t saturating_increment(uint64_t value) {
    return value == std::numeric_limits<uint64_t>::max() ? value : value + 1;
}

inline ComputeTimingSelectorObservation observe_compute_timing_selector(
    const ComputeTimingSelector& selector,
    ComputeTimingSelectorCounters& counters,
    uint64_t code_addr,
    uint64_t program_hash) {
    counters.seen = saturating_increment(counters.seen);
    const bool matched = compute_timing_selector_matches(selector, code_addr, program_hash);
    const bool first_match = matched && counters.matched == 0;
    if (matched) counters.matched = saturating_increment(counters.matched);
    return {matched, first_match};
}

constexpr bool compute_timing_zero_match_is_invalid(
    const ComputeTimingSelectorCounters& counters) {
    return counters.matched == 0;
}

// Runtime frontends do not all reach static destruction: prosper-app deliberately uses _Exit while
// its detached guest thread is still alive. Keep the exactly-once decision in this pure policy seam
// so an explicit frontend report and the ordinary destructor fallback cannot both publish a verdict.
inline bool claim_compute_timing_selector_summary(
    ComputeTimingSelectorCounters& counters) {
    if (counters.summary_reported) return false;
    counters.summary_reported = true;
    return true;
}

} // namespace prosper::frontend
