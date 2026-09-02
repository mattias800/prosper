// gpu_submit_gate.cpp — see gpu_submit_gate.hpp (#3225).
#include "host/platform/gpu_submit_gate.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace prosper {

namespace {

// One word carries both halves so admission is a single atomic decision. Bit 0 is the shutdown
// flag; the rest is the admitted-region count. Packing them is what makes "admit only while open"
// atomic: a compare-exchange that succeeds with bit 0 clear cannot interleave with the fetch_or
// that sets it, so after begin_shutdown() no further region can appear. Keeping them in two
// separate atomics would need a Dekker-style argument instead, and would let a REFUSED enter
// transiently bump the count a drain could see.
constexpr uint64_t kShutdownBit = 1u;
constexpr uint64_t kOneRegion   = 2u;   // count lives in bits 1..63

std::atomic<uint64_t> g_state{0};

// Waiters exist only during shutdown, so the admitted path never touches the mutex. The count is
// what lets leave() skip the lock when nobody is draining; it is read with seq_cst against the
// waiter's seq_cst store for the usual store-buffer reason (see the comment in notify_if_idle).
std::atomic<int> g_waiters{0};
std::mutex g_mx;
std::condition_variable g_cv;

int region_count(uint64_t state) { return static_cast<int>(state >> 1); }

void notify_if_idle() {
    // A leaving region and an arriving drain race: the leaver writes the count then reads the
    // waiter count; the drain writes the waiter count then reads the region count. Both pairs are
    // sequentially consistent, so at least one side observes the other and the wakeup cannot be
    // lost. Taking the mutex (even empty) before notifying closes the remaining window where the
    // waiter has evaluated its predicate but has not yet slept.
    if (g_waiters.load(std::memory_order_seq_cst) == 0) return;
    if (region_count(g_state.load(std::memory_order_seq_cst)) != 0) return;
    { std::lock_guard<std::mutex> lk(g_mx); }
    g_cv.notify_all();
}

} // namespace

bool gpu_submit_gate_try_enter() {
    uint64_t state = g_state.load(std::memory_order_acquire);
    for (;;) {
        if (state & kShutdownBit) return false;
        if (g_state.compare_exchange_weak(state, state + kOneRegion,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire))
            return true;
    }
}

void gpu_submit_gate_leave() {
    // Saturating rather than a bare fetch_sub. A mispaired leave would otherwise wrap the count to
    // ~2^62, after which every drain in the process times out (or, with a negative timeout, never
    // returns) — a silent, permanent breakage of the one thing this file exists to do. Unreachable
    // today (GpuSubmitRegion is the only caller and it leaves only when it was admitted), so this
    // guards a future misuse rather than a live defect. The successful exchange is seq_cst because
    // notify_if_idle()'s argument depends on it.
    uint64_t state = g_state.load(std::memory_order_acquire);
    for (;;) {
        if (region_count(state) == 0) return;   // mispaired leave: refuse to underflow
        if (g_state.compare_exchange_weak(state, state - kOneRegion,
                                          std::memory_order_seq_cst,
                                          std::memory_order_acquire))
            break;
    }
    notify_if_idle();
}

void gpu_submit_gate_begin_shutdown() {
    // Under the waiter's mutex so a drain cannot check the flag, decide to sleep, and miss this
    // transition — the same reason prosper_request_stop() takes the lifecycle mutex.
    {
        std::lock_guard<std::mutex> lk(g_mx);
        g_state.fetch_or(kShutdownBit, std::memory_order_seq_cst);
    }
    g_cv.notify_all();
}

bool gpu_submit_gate_shutting_down() {
    return (g_state.load(std::memory_order_acquire) & kShutdownBit) != 0;
}

int gpu_submit_gate_in_flight() {
    return region_count(g_state.load(std::memory_order_acquire));
}

bool gpu_submit_gate_drain(int timeout_ms) {
    const auto idle = [] { return region_count(g_state.load(std::memory_order_seq_cst)) == 0; };
    if (idle()) return true;
    if (timeout_ms == 0) return false;

    // Published before the predicate is first evaluated under the lock, so any leaver that runs
    // from here on will take the notify path.
    g_waiters.fetch_add(1, std::memory_order_seq_cst);
    bool drained = false;
    {
        std::unique_lock<std::mutex> lk(g_mx);
        if (timeout_ms < 0) {
            g_cv.wait(lk, idle);
            drained = true;
        } else {
            drained = g_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), idle);
        }
    }
    g_waiters.fetch_sub(1, std::memory_order_seq_cst);
    return drained;
}

void gpu_submit_gate_reset() {
    {
        std::lock_guard<std::mutex> lk(g_mx);
        g_state.store(0, std::memory_order_seq_cst);
    }
    g_cv.notify_all();
}

} // namespace prosper
