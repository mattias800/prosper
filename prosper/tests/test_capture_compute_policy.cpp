#include "gpu/capture_compute_policy.hpp"

#include <cstdio>

using prosper::gpu::timeline_capture_requires_portable_compute;

int main() {
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
