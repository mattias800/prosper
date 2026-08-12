// event_bus.cpp — EventBus implementation.

#include "event_bus.hpp"
#include <algorithm>

namespace prosper::diagnostics {

size_t EventBus::subscribe(Subscriber cb) {
    if (!cb) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    size_t h = next_handle_++;
    subscribers_.push_back(std::move(cb));
    return h;
}

void EventBus::unsubscribe(size_t handle) {
    // Handles are 1-based indices; subscriber at index handle-1.
    // We null-out rather than erase to preserve other handles.
    std::lock_guard<std::mutex> lock(mutex_);
    if (handle == 0 || handle > subscribers_.size()) return;
    subscribers_[handle - 1] = nullptr;
}

void EventBus::publish(const BootEvent& event) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& sub : subscribers_) {
        if (sub) sub(event);
    }
}

size_t EventBus::subscriber_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto& sub : subscribers_)
        if (sub) ++count;
    return count;
}

void EventBus::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    subscribers_.clear();
    next_handle_ = 1;
}

EventBus& event_bus() {
    static EventBus bus;
    return bus;
}

} // namespace prosper::diagnostics
