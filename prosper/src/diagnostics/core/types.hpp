// types.hpp — Core type definitions for diagnostics (PROSPER_DIAGNOSTICS build only).

#pragma once

#include <string>
#include <cstdint>
#include <chrono>

namespace prosper::diagnostics {

// Boot phases tracked during PS4 program startup.
// These align with milestones in boot_program().
enum class BootPhase : unsigned {
    PROCESS_START = 0,     // boot_program() entered
    LINKING,              // link_program() complete
    HLE_REGISTERED,       // register_builtin_hle() complete
    MODULES_MAPPED,       // all images mapped into guest VA
    STUBS_INSTALLED,      // trap handler + stubs active
    GUEST_INITS_RUNNING,  // run_guest_inits() called
    BOOT_COMPLETE,        // boot_program() returning true
    _COUNT                // sentinel — must be last
};

// Convert BootPhase to human-readable string.
inline const char* phase_name(BootPhase p) {
    static const char* names[] = {
        "PROCESS_START", "LINKING", "HLE_REGISTERED", "MODULES_MAPPED",
        "STUBS_INSTALLED", "GUEST_INITS_RUNNING", "BOOT_COMPLETE"
    };
    unsigned i = static_cast<unsigned>(p);
    return i < static_cast<unsigned>(BootPhase::_COUNT) ? names[i] : "UNKNOWN";
}

// A single timestamped boot event.
struct BootEvent {
    BootPhase phase;
    double timestamp_ms;  // milliseconds from process start (monotonic)

    BootEvent(BootPhase p, double ts) : phase(p), timestamp_ms(ts) {}
};

// Severity levels for future diagnostic categories (reserved).
enum class Severity : unsigned { INFO, WARNING, ERROR };

// Get monotonic time in milliseconds since first call.
inline double monotonic_ms() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

} // namespace prosper::diagnostics
