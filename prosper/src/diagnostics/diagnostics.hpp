// diagnostics.hpp — Single include entry point for prosper diagnostics.
//
// When PROSPER_DIAGNOSTICS is OFF (default), this header compiles to zero-cost
// stubs: all calls inline to empty functions with no overhead.
//
// When ON, the full observer-only diagnostics layer is available for boot
// timeline capture, event subscription, and JSON report generation.

#pragma once

#ifdef PROSPER_DIAGNOSTICS
#include "core/types.hpp"
#include "core/event_bus.hpp"
#include "core/context.hpp"
#include "storage/json_writer.hpp"
#else

namespace prosper::diagnostics {

// Boot phases tracked during program startup.
enum class BootPhase : unsigned {
    PROCESS_START = 0,
    LINKING,
    HLE_REGISTERED,
    MODULES_MAPPED,
    STUBS_INSTALLED,
    GUEST_INITS_RUNNING,
    BOOT_COMPLETE,
    _COUNT  // sentinel
};

// Event emitted when a boot phase is reached.
struct BootEvent {
    BootPhase phase;
    double timestamp_ms;  // monotonic clock
};

// Stub context — all methods are no-ops when diagnostics disabled.
class DiagnosticContext {
public:
    static DiagnosticContext& instance() {
        static DiagnosticContext ctx;
        return ctx;
    }

    bool is_enabled() const { return false; }
    void enable() {}
    void disable() {}

    void record_phase(BootPhase) {}
    void emit(const BootEvent&) {}

    size_t event_count() const { return 0; }
    void clear() {}
};

// Inline stubs — compiler optimizes these away entirely.
inline void record_boot_phase(BootPhase) {}

} // namespace prosper::diagnostics

#endif // PROSPER_DIAGNOSTICS
