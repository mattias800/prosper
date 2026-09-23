// One passive Kena menu observation. The time is measured from the renderer's first callback.
#pragma once

#include <charconv>
#include <cstdint>
#include <string_view>

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
                                     bool already_fired, bool final_span) {
    return spec.armed && final_span && !already_fired && elapsed_ms >= spec.after_ms;
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
    if (!published || !front_address || !publication_id) return {};
    return {new_flip ? "gpu-front" : "gpu-front-existing-slot", front_address, flip,
            publication_id};
}

} // namespace prosper::frontend
