#include "capture_renderer_policy.hpp"

#include <cstdio>

using prosper::frontend::timeline_capture_allows_persistent_targets;

int main() {
    if (!timeline_capture_allows_persistent_targets(false, false)) {
        std::fprintf(stderr, "normal rendering unexpectedly disabled persistent targets\n");
        return 1;
    }
    if (!timeline_capture_allows_persistent_targets(true, true)) {
        std::fprintf(stderr, "phase-gated timeline capture unexpectedly disabled persistent targets\n");
        return 1;
    }
    if (timeline_capture_allows_persistent_targets(true, false)) {
        std::fprintf(stderr, "immediate timeline capture unexpectedly retained persistent targets\n");
        return 1;
    }
    return 0;
}
