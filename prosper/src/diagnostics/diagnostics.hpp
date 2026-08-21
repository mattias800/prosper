// diagnostics.hpp — Single include entry point for prosper diagnostics.
//
// When PROSPER_DIAGNOSTICS is OFF (default), the event-history and JSON-report layer compiles
// away: recording a phase stores nothing and publishes nothing.
//
// When ON, the full observer-only diagnostics layer is available for boot
// timeline capture, event subscription, and JSON report generation.
//
// `record_boot_phase()` is the ONE exception to that split, and it is deliberate. In BOTH builds it
// forwards to `log_boot_phase()`, whose only cost when unselected is a call and a cached bool test,
// seven times in the lifetime of the process. This header used to promise "zero-cost stubs", and
// keeping that promise literally is what made the boot phases unobservable in every shipped build:
// a `boot_program()` that hangs inside `run_guest_inits()` printed nothing, and no runtime switch
// could make it print, because the instrumentation had been compiled out months earlier. See
// boot_phase_log.hpp. The recording/reporting layer is still fully compiled out; only the one
// stderr line survives, and only when PROSPER_BOOTPHASE asks for it.

#pragma once

#include "boot_phase_log.hpp"

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

// The history/reporting side is stubbed away; the phase LINE is not (see the note at the top).
inline void record_boot_phase(BootPhase p) { log_boot_phase(static_cast<unsigned>(p)); }

} // namespace prosper::diagnostics

#endif // PROSPER_DIAGNOSTICS
