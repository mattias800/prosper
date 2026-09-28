#pragma once
// The fixed-function resolve, carved out of the per-target pass loop (#3892).

#include "shared/live/live_renderer_internal.hpp"        // RttCache, backend stats, render state types
#include "shared/live/submit_renderer/callback_types.hpp" // the callback types these contexts name

namespace prosper::frontend::submit_renderer {

// ---- Fixed-function resolve (#3892) --------------------------------------------------------------
//
// One CB_COLOR_CONTROL.MODE=RESOLVE pass: copy the rendered color0 surface into color1's
// retained target (GPU copy, batched or out of band, with the CPU-mirror fallback), publish the
// destination in g_rtt, and flush the ordered batch when the resolve is the submit's last pass.
// This was the body of the pass loop's cb_resolve branch, moved here verbatim; the loop's
// `continue` stays at the call site.
// The pass loop's state that resolve_pass reads and writes, one reference per object.
struct ResolvePassContext {
    const bool & batch_backend_submits;
    const std::vector<prosper::gpu::DrawItem> & items;
    RenderTiming & pending_timing;
    const bool & timing_enabled;
    RttCache& g_rtt;
    size_t& pass_i;
    prosper::test::BackendSubmissionBatch& backend_submission;
    std::vector<const prosper::gpu::DrawItem *>& pass;
};
void resolve_pass(ResolvePassContext& ctx);

} // namespace prosper::frontend::submit_renderer
