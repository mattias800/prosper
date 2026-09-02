// gpu_submit_gate.hpp — admit/refuse GPU submissions across process shutdown (#3225).
//
// The problem this exists for. prosper-app cannot join the guest thread (run_entry() does not yet
// observe the cooperative stop flag), so it detaches it and calls std::_Exit. std::_Exit becomes
// exit_group(), and a thread that is *inside a syscall* at that moment cannot be torn down until it
// returns from the kernel. When that syscall is an amdgpu command submission, the task parks in
// __drm_exec_lock_obj holding/awaiting a GEM reservation, the process cannot be reaped (it shows as
// a zombie whose SIGKILL is a no-op), and the compositor's own DRM work blocks behind it — a frozen
// desktop that only a root-forced GPU reset clears. It cost the developer two sessions in one day.
//
// What this gate does. Every guest-thread GPU submission enters a counted region first. Once
// gpu_submit_gate_begin_shutdown() has run, no NEW region is admitted, so the in-flight count can
// actually reach zero; gpu_submit_gate_drain() then waits, bounded, for the submissions that were
// already inside to come out. The frontend runs both before its _Exit.
//
// What it does NOT do, stated plainly so nobody mistakes it for the fix. It BOUNDS the window; it
// does not close it. A thread that is already inside vkQueueSubmit when shutdown begins is still
// waited for, and if it never returns the drain times out and we _Exit exactly as before — no worse
// than today, but no better either. The real fix is the cooperative stop the frontend's own comment
// names as unimplemented: run_entry() observes the stop flag, stops at a submission boundary, and
// the thread is JOINED, after which an ordinary drain and teardown are safe.
//
// Deliberately dependency-free (no Vulkan, no OS/windowing headers) so it lives in prosper_core and
// is callable from the offscreen backend, the live compute path, the present blit and the app
// alike. It is a sibling of lifecycle.hpp rather than part of it: that file carries boolean
// cooperative signals, this one is a counted region with a drain.
#pragma once

#include <cstdint>

namespace prosper {

// Try to enter a GPU-submission region. Returns true when admitted — the caller MUST then pair it
// with exactly one gpu_submit_gate_leave(). Returns false once shutdown has begun, in which case
// the caller must not submit and nothing needs to be released. Thread-safe, lock-free on the
// admitted path. Nesting is allowed (the count is a depth, not a flag), though an inner region
// entered after shutdown began is refused while its outer one stays admitted — which is the
// intended behaviour: refuse new work, wait for what is already running.
bool gpu_submit_gate_try_enter();

// Leave a region admitted by gpu_submit_gate_try_enter(). Never call after a refusal.
void gpu_submit_gate_leave();

// RAII wrapper for the pair above. Declare one immediately before a queue submission and check
// admitted():
//
//     GpuSubmitRegion region;
//     if (!region.admitted()) return VK_ERROR_DEVICE_LOST;   // shutting down; the device is going
//     return vkQueueSubmit(...);
class GpuSubmitRegion {
public:
    GpuSubmitRegion() : admitted_(gpu_submit_gate_try_enter()) {}
    ~GpuSubmitRegion() { if (admitted_) gpu_submit_gate_leave(); }
    GpuSubmitRegion(const GpuSubmitRegion&) = delete;
    GpuSubmitRegion& operator=(const GpuSubmitRegion&) = delete;
    GpuSubmitRegion(GpuSubmitRegion&&) = delete;
    GpuSubmitRegion& operator=(GpuSubmitRegion&&) = delete;

    // True when this region was admitted, i.e. the caller may submit.
    bool admitted() const { return admitted_; }

private:
    bool admitted_;
};

// Close the gate: after this, gpu_submit_gate_try_enter() always refuses. Idempotent, thread-safe.
// Without this the count cannot be relied on to reach zero, so a drain would be a race rather than
// a wait.
void gpu_submit_gate_begin_shutdown();

// True once gpu_submit_gate_begin_shutdown() has run.
bool gpu_submit_gate_shutting_down();

// Wait for every admitted region to leave. Returns true when the count reached zero, false when
// `timeout_ms` expired first (a caller that is about to _Exit should say so on stderr rather than
// wait longer — an unbounded wait here is the freeze this file exists to avoid, moved earlier).
// A negative timeout waits indefinitely; 0 polls once. Call gpu_submit_gate_begin_shutdown() first.
bool gpu_submit_gate_drain(int timeout_ms);

// Number of regions currently admitted. Exact (a refused enter never appears in it), but only a
// snapshot — for diagnostics and tests, not for control flow.
int gpu_submit_gate_in_flight();

// Reopen the gate and clear the count (tests, and an in-process session restart). Mirrors
// prosper_reset_stop(). Not for use while regions are live.
void gpu_submit_gate_reset();

} // namespace prosper
