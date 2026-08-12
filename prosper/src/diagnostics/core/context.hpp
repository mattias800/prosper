// context.hpp — DiagnosticContext coordinator (PROSPER_DIAGNOSTICS build only).
//
// Singleton context that:
// - Enables/disables diagnostics globally
// - Records boot phase events
// - Stores event history for JSON export

#pragma once

#include "types.hpp"
#include <vector>
#include <mutex>

namespace prosper::diagnostics {

class DiagnosticContext {
public:
    static DiagnosticContext& instance();

    // Enable/disable recording. When disabled, record_phase() is a no-op.
    bool is_enabled() const;
    void enable();
    void disable();

    // Record a boot phase transition (only if enabled).
    void record_phase(BootPhase phase);

    // Manually emit an event (for custom instrumentation).
    void emit(const BootEvent& event);

    // Query recorded events.
    size_t event_count() const;
    const std::vector<BootEvent>& events() const;

    // Clear all recorded events.
    void clear();

private:
    DiagnosticContext() = default;
    mutable std::mutex mutex_;
    bool enabled_ = false;
    std::vector<BootEvent> events_;
};

// Convenience: record current phase to global context.
void record_boot_phase(BootPhase phase);

} // namespace prosper::diagnostics
