// diag_clock.hpp — one process-wide time source shared by runtime diagnostics.
//
// Why this exists: diagnostics in different subsystems write to the same stderr stream from
// DIFFERENT threads, and stderr interleaving is not a happens-before relation. Comparing two
// diagnostics by their line order in a log is therefore unsound, and it produced a wrong published
// conclusion on #3142 — "the read completes 1,428 log lines before the sample" was read as a
// timeline when it establishes no ordering at all.
//
// A shared monotonic stamp makes the comparison sound. steady_clock is process-wide and monotonic,
// so values taken independently in unrelated translation units are directly comparable without an
// anchor, a registry, or any shared mutable state.
//
// Deliberately NOT inside the PROSPER_DIAGNOSTICS compile-time split of diagnostics.hpp: the
// diagnostics that need this are gated at RUNTIME by environment variables and must work in the
// default build. A time source that vanishes unless a build flag is set is exactly the failure that
// header's own banner records.

#pragma once

#include <chrono>
#include <cstdint>

namespace prosper::diagnostics {

// Microseconds on the process-wide monotonic clock. Comparable across threads and translation
// units; the absolute value is meaningless, only differences are.
inline uint64_t diag_now_us() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace prosper::diagnostics
