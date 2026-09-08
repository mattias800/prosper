#pragma once
#include "gpu/timeline/gpu_timeline.hpp"
#include <chrono>
#include <thread>

// Bound a test's wait for the production background writer; never infer completion from a file
// merely appearing, since installation and result publication are separate operations.
inline bool wait_for_interactive_grab(prosper::gpu::InteractiveGrabOutcome& outcome) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
        if (prosper::gpu::take_interactive_grab_outcome(outcome)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}
