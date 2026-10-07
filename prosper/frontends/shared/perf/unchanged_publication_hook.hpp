#pragma once
// unchanged_publication_hook.hpp -- run the unchanged-publication census on the renderer's
// "a compute result was published to a render target" notifier, but only while an F8 capture window is
// active.
//
// The census hashes every published byte (about 8 MB for a 1080p target), which is cheap for one
// publication and not something to pay on every submit of every title. F8 is the project's on-demand
// "measure now" gesture and already gates every other expensive observer (`detailed_timing_active()`), so
// the census rides on it: outside a capture window the hook is one predictable branch, there is no
// environment switch, and nothing is registered.
//
// Observes only: the wrapped notifier is always called first, with the same write, whatever the census
// does. Only writes that carry CPU pixels are counted (see unchanged_publication_census.hpp, SCOPE). The
// hash runs on the publishing thread inside the span F8 is timing; the census reports it as `hash_ms=`.
#include <cstdio>
#include <functional>
#include <string>
#include <utility>

#include "diagnostics/exit_census.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "shared/perf/performance_capture.hpp"
#include "shared/perf/unchanged_publication_census.hpp"

namespace prosper::perf {

// Never destroyed: the end-of-run report reads it after static destruction has begun elsewhere.
inline UnchangedPublicationCensus& unchanged_publication_census() {
    static UnchangedPublicationCensus* const census = new UnchangedPublicationCensus();
    return *census;
}

// The census is observation only and must never stop the publication, so a failure inside it (an
// allocation, the first registration) is swallowed. `capture` and `census` are parameters so the gate can be
// tested without a renderer; the defaults used by the wrapper are the process-wide instances.
inline void note_published_target(const prosper::gpu::LiveTargetImageWrite& write,
                                  const InteractivePerformanceCapture& capture,
                                  UnchangedPublicationCensus& census) {
    const uint64_t generation = capture.active_generation();
    if (generation == 0) return;
    if (!write.linear_pixels || write.linear_pixels->empty()) return;
    try {
        // Registered on first use, never during static initialisation (see exit_census.hpp). Prints
        // nothing unless something was counted.
        static const bool registered = (prosper::diagnostics::register_census(nullptr, [] {
            const std::string text = unchanged_publication_census().format();
            if (text.empty()) return false;
            std::fputs(text.c_str(), stderr);
            return true;
        }), true);
        (void)registered;
        census.note(write.gpu_addr, write.width, write.height, static_cast<uint32_t>(write.format),
                    write.linear_pixels->data(), write.linear_pixels->size(), generation);
    } catch (...) {
    }
}

// Wraps `inner` so it runs first, unconditionally and unchanged, and the census sees the write after it.
inline std::function<void(const prosper::gpu::LiveTargetImageWrite&)> with_unchanged_publication_census(
    std::function<void(const prosper::gpu::LiveTargetImageWrite&)> inner,
    const InteractivePerformanceCapture& capture = interactive_performance_capture(),
    UnchangedPublicationCensus& census = unchanged_publication_census()) {
    return [inner = std::move(inner), &capture, &census](const prosper::gpu::LiveTargetImageWrite& write) {
        inner(write);
        note_published_target(write, capture, census);
    };
}

}  // namespace prosper::perf
