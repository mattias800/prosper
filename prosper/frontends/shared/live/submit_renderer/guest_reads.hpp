#pragma once
// Bounds-checked guest reads the submit-renderer callback (live_renderer.cpp) and the modules
// carved out of it share (#3892). They were captureless lambdas local to the callback and moved
// here verbatim; only `auto` became `inline constexpr auto` so they can live at namespace scope.

#include "gpu/execute/gpu_execute.hpp"          // guest_readable
#include "shared/live/guest_source_read.hpp"    // guest_source_readable_prefix / copy / equal
#include "shared/live/submit_renderer/validation_mapping_census.hpp"   // PROSPER_VALIDATION_MAPPING_CENSUS

#include <cstddef>
#include <cstdint>

namespace prosper::frontend::submit_renderer {

// A declared resource can extend past its real committed guest mapping. The shared
// prefix helper checks exact mapping boundaries before a host read can lazy-commit a
// reservation; it also prepares Windows sparse direct pages within the proven extent.
// Callers keep their existing zero-filled short-read fallback.
inline constexpr auto safe_span = [](uint64_t a, size_t n) -> size_t {
    return guest_source_readable_prefix(a, n, prosper::gpu::guest_readable);
};
inline constexpr auto safe_copy = [](uint8_t* dst, uint64_t a, size_t n) -> size_t {
    return copy_guest_source(dst, a, n, prosper::gpu::guest_readable);
};
inline constexpr auto safe_equal = [](const uint8_t* expected, uint64_t a, size_t n,
                                                size_t& compared,
                                                GuestSourceComparisonObservation* observation = nullptr) -> bool {
    const bool equal = equal_guest_source_prefix(
        expected, a, n, compared, prosper::gpu::guest_readable, observation);
    note_validation_mapping(a, n, equal);
    return equal;
};

} // namespace prosper::frontend::submit_renderer
