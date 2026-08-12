// context.cpp — DiagnosticContext implementation.

#include "context.hpp"
#include "event_bus.hpp"

namespace prosper::diagnostics {

DiagnosticContext& DiagnosticContext::instance() {
    static DiagnosticContext ctx;
    return ctx;
}

bool DiagnosticContext::is_enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}

void DiagnosticContext::enable() {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = true;
}

void DiagnosticContext::disable() {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = false;
}

void DiagnosticContext::record_phase(BootPhase phase) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) return;
    BootEvent event(phase, monotonic_ms());
    events_.push_back(event);
    // Publish outside lock to avoid potential deadlock with subscribers.
    mutex_.unlock();
    event_bus().publish(event);
    mutex_.lock();
}

void DiagnosticContext::emit(const BootEvent& event) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) return;
    events_.push_back(event);
    mutex_.unlock();
    event_bus().publish(event);
    mutex_.lock();
}

size_t DiagnosticContext::event_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_.size();
}

const std::vector<BootEvent>& DiagnosticContext::events() const {
    return events_;  // Caller should hold their own sync if iterating while recording
}

void DiagnosticContext::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.clear();
}

void record_boot_phase(BootPhase phase) {
    DiagnosticContext::instance().record_phase(phase);
}

} // namespace prosper::diagnostics
