// pipeline_observe_hook.hpp -- the live compute backend's side of the pipelining observer (ADR 0009
// Stage 1, step 2). Diagnostic only: with PROSPER_PIPELINE_OBSERVE unset the call below is one
// cached boolean test; with it set it records each completed dispatch's guest ranges, never alters
// them, and prints the observer's report every 256 dispatches and at exit.
#pragma once

#include <chrono>
#include <optional>
#include <vector>

#include "gpu/execute/gpu_execute.hpp"
#include "shared/compute/pipeline_observer.hpp"
#include "shared/live/live_compute_bound_resources.hpp"

namespace prosper::frontend {

// PROSPER_PIPELINE_OBSERVE, sampled once.
bool pipeline_observe_enabled();

// Report one completed dispatch with the backend's own phase markers: `start` (execute_item entry),
// `pipeline_start` (before command recording), `dispatch_end` (after the fence wait) and `loop_exit`
// (after writeback). A missing optional marker counts that span as zero. setup = start..pipeline_start,
// gpu = pipeline_start..dispatch_end (record + submit + fence wait), writeback = dispatch_end..loop_exit.
// The first call also installs the executor's ordered-operation hook, so graphics spans and DMA
// copies between dispatches are seen from then on.
void pipeline_observe_dispatch(
    const prosper::gpu::ComputeItem& item, const std::vector<BoundBuffer>& buffers,
    const std::vector<BoundImage>& images, std::chrono::steady_clock::time_point start,
    const std::optional<std::chrono::steady_clock::time_point>& pipeline_start,
    const std::optional<std::chrono::steady_clock::time_point>& dispatch_end,
    std::chrono::steady_clock::time_point loop_exit);

// The ranges a dispatch's bindings occupy in guest memory, and how it uses them. Pure; the hook
// above is this plus a call into the observer. Conservative on purpose: every buffer counts as read
// (a writable one may read its previous contents), a storage image counts as read unless it is
// write-only, and only writable buffers and written-back storage images count as written.
std::vector<BindingRange> dispatch_binding_ranges(const std::vector<BoundBuffer>& buffers,
                                                  const std::vector<BoundImage>& images);

}  // namespace prosper::frontend
