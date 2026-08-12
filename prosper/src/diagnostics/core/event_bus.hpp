// event_bus.hpp — Thread-safe event bus for diagnostics (PROSPER_DIAGNOSTICS build only).
//
// Subscribers receive events via callable signature void(const BootEvent&).
// Bus owns no state beyond the subscriber list; all events are passed by const ref.

#pragma once

#include "types.hpp"
#include <vector>
#include <functional>
#include <mutex>

namespace prosper::diagnostics {

class EventBus {
public:
    using Subscriber = std::function<void(const BootEvent&)>;

    // Subscribe a callback. Returns handle for unsubscribe (0 on error).
    size_t subscribe(Subscriber cb);

    // Remove subscriber by handle.
    void unsubscribe(size_t handle);

    // Publish event to all current subscribers.
    void publish(const BootEvent& event);

    // Current subscriber count.
    size_t subscriber_count() const;

    // Clear all subscribers.
    void clear();

private:
    mutable std::mutex mutex_;
    std::vector<Subscriber> subscribers_;
    size_t next_handle_ = 1;
};

// Global accessor (created on first use when diagnostics enabled).
EventBus& event_bus();

} // namespace prosper::diagnostics
