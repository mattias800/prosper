// lifecycle.cpp — see lifecycle.hpp.
#include "lifecycle.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace prosper {

namespace {
std::atomic<bool> g_stop{false};
std::atomic<bool> g_paused{false};
std::mutex g_lifecycle_mx;
std::condition_variable g_lifecycle_cv;
}

void prosper_request_stop() {
    {
        // Use the waiter's mutex for the transition so a waiter cannot miss the notification
        // between checking the predicate and sleeping.
        std::lock_guard<std::mutex> lk(g_lifecycle_mx);
        g_stop.store(true, std::memory_order_release);
    }
    g_lifecycle_cv.notify_all();
}

bool prosper_stop_requested() { return g_stop.load(std::memory_order_acquire); }

void prosper_set_paused(bool paused) {
    {
        std::lock_guard<std::mutex> lk(g_lifecycle_mx);
        g_paused.store(paused, std::memory_order_release);
    }
    if (!paused) g_lifecycle_cv.notify_all();
}

bool prosper_paused() { return g_paused.load(std::memory_order_acquire); }

bool prosper_wait_while_paused() {
    // This runs once per flip/audio grain. Keep the normal (never paused) path lock-free; a pause
    // racing this check is observed at the next cooperative boundary.
    if (!g_paused.load(std::memory_order_acquire))
        return !g_stop.load(std::memory_order_acquire);
    std::unique_lock<std::mutex> lk(g_lifecycle_mx);
    g_lifecycle_cv.wait(lk, [] {
        return !g_paused.load(std::memory_order_acquire) ||
               g_stop.load(std::memory_order_acquire);
    });
    return !g_stop.load(std::memory_order_acquire);
}

void prosper_reset_stop() {
    {
        std::lock_guard<std::mutex> lk(g_lifecycle_mx);
        g_stop.store(false, std::memory_order_release);
        g_paused.store(false, std::memory_order_release);
    }
    g_lifecycle_cv.notify_all();
}

} // namespace prosper
