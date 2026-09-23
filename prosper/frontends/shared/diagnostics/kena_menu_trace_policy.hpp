// One passive Kena menu observation. The time is measured from the renderer's first callback.
#pragma once

#include <charconv>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace prosper::frontend {

struct KenaMenuTraceSpec {
    static constexpr uint64_t kLatestMs = 120'000;
    bool armed = false;
    uint64_t after_ms = 0;
};

// A malformed request must never turn into an immediate or unbounded GPU readback.
inline KenaMenuTraceSpec parse_kena_menu_trace(const char* value) {
    if (!value) return {};
    const std::string_view spec(value);
    if (!spec.starts_with("ms:") || spec.size() == 3) return {};
    uint64_t ms = 0;
    const auto digits = spec.substr(3);
    const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), ms);
    if (error != std::errc{} || end != digits.data() + digits.size() ||
        ms > KenaMenuTraceSpec::kLatestMs)
        return {};
    return {true, ms};
}

inline bool kena_menu_trace_callback(KenaMenuTraceSpec spec, uint64_t elapsed_ms,
                                     bool already_fired, bool final_span,
                                     bool render_admitted) {
    return spec.armed && render_admitted && final_span && !already_fired &&
           elapsed_ms >= spec.after_ms;
}

struct KenaGuestPeekRange {
    uint64_t address = 0;
    uint32_t bytes = 0;
};

// Strict, bounded, opt-in guest-memory census at the same callback as the menu trace.
// Every range must be explicit: 0xADDRESS:0xBYTES[,0xADDRESS:0xBYTES].
inline bool parse_kena_guest_peek_ranges(const char* value,
                                         std::vector<KenaGuestPeekRange>& ranges) {
    ranges.clear();
    if (!value || !*value) return false;
    const std::string_view spec(value);
    auto parse_hex = [](std::string_view token, uint64_t& number) {
        if (token.size() < 3 || token.substr(0, 2) != "0x") return false;
        const auto digits = token.substr(2);
        const auto [end, error] = std::from_chars(
            digits.data(), digits.data() + digits.size(), number, 16);
        return error == std::errc{} && end == digits.data() + digits.size() && number != 0;
    };
    uint64_t total = 0;
    size_t begin = 0;
    while (begin < spec.size()) {
        const size_t comma = spec.find(',', begin);
        const auto entry = spec.substr(begin, comma == std::string_view::npos
                                                ? comma : comma - begin);
        const size_t colon = entry.find(':');
        uint64_t address = 0, bytes = 0;
        if (colon == std::string_view::npos || entry.find(':', colon + 1) != std::string_view::npos ||
            !parse_hex(entry.substr(0, colon), address) ||
            !parse_hex(entry.substr(colon + 1), bytes) ||
            bytes > (64ull << 20) || total + bytes > (96ull << 20) ||
            address > std::numeric_limits<uint64_t>::max() - bytes || ranges.size() == 2) {
            ranges.clear();
            return false;
        }
        ranges.push_back({address, static_cast<uint32_t>(bytes)});
        total += bytes;
        if (comma == std::string_view::npos) return true;
        begin = comma + 1;
    }
    ranges.clear();
    return false;
}

struct KenaMenuGpuSource {
    const char* kind = "none";
    uint64_t address = 0;
    uint64_t flip = 0;
    uint64_t publication_id = 0;
};

inline KenaMenuGpuSource kena_menu_gpu_source(bool published, bool new_flip,
                                              uint64_t front_address, uint64_t flip,
                                              uint64_t publication_id) {
    if (!published) return {};
    if (!front_address || !publication_id)
        return {"gpu-source-unknown", 0, flip, publication_id};
    return {new_flip ? "gpu-front" : "gpu-front-same-flip", front_address, flip,
            publication_id};
}

} // namespace prosper::frontend
