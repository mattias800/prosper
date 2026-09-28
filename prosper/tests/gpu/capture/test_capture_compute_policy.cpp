#include "gpu/capture/capture_compute_policy.hpp"

#include <cstdio>

using prosper::gpu::env_capture_requires_portable_compute;
using prosper::gpu::EnvCaptureWindow;
using prosper::gpu::timeline_capture_requires_portable_compute;

namespace {
bool env_window_policy_holds() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
    };
    EnvCaptureWindow w;
    check(!env_capture_requires_portable_compute(w), "no capture requested: native compute");
    w.requested = true;
    check(env_capture_requires_portable_compute(w),
          "an ungated capture is armed from the first submit (unchanged)");
    // #3895: a capture aimed late must not change the compute modules the run executes before it.
    w.after = 1000; w.invocations = 999;
    check(!env_capture_requires_portable_compute(w),
          "before PROSPER_GPU_CAPTURE_AFTER the run compiles native compute (#3895)");
    w.invocations = 1000;
    check(env_capture_requires_portable_compute(w), "the AFTER gate opens the window");
    w.after_ms = 5000;
    check(!env_capture_requires_portable_compute(w),
          "AFTER_MS keeps the window shut until its clock has started");
    w.clock_started = true; w.elapsed_ms = 4999;
    check(!env_capture_requires_portable_compute(w), "AFTER_MS not yet elapsed");
    w.elapsed_ms = 5000;
    check(env_capture_requires_portable_compute(w), "AFTER_MS elapsed opens the window");
    w.finished = true;
    check(!env_capture_requires_portable_compute(w),
          "the window closes once the one-shot capsule is written");
    EnvCaptureWindow legacy;
    legacy.requested = true; legacy.legacy_readback = true; legacy.after = 1000;
    legacy.finished = true;
    check(env_capture_requires_portable_compute(legacy),
          "PROSPER_GPU_CAPTURE_READBACK restores the whole-run portable compute arm");
    legacy.requested = false;
    check(!env_capture_requires_portable_compute(legacy),
          "the legacy arm is inert without a capture");
    return failures == 0;
}
}  // namespace

int main() {
    if (!env_window_policy_holds()) return 1;
    if (timeline_capture_requires_portable_compute(false, false, false)) {
        std::fprintf(stderr, "normal compute unexpectedly required portable capture storage\n");
        return 1;
    }
    if (timeline_capture_requires_portable_compute(true, true, false)) {
        std::fprintf(stderr, "dormant phase gate unexpectedly required portable capture storage\n");
        return 1;
    }
    if (!timeline_capture_requires_portable_compute(true, true, true)) {
        std::fprintf(stderr, "armed phase gate unexpectedly retained native storage formats\n");
        return 1;
    }
    if (!timeline_capture_requires_portable_compute(true, false, false)) {
        std::fprintf(stderr, "immediate capture unexpectedly retained native storage formats\n");
        return 1;
    }
    return 0;
}
