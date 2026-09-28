#pragma once
// Backend timing accumulation and PROSPER_RTT_TIMING records, carved out of the submit callback (#3892).

#include "shared/live/live_renderer_internal.hpp"        // RttCache, backend stats, render state types
#include "shared/live/submit_renderer/callback_types.hpp" // the callback types this context names

namespace prosper::frontend::submit_renderer {

// record_backend_timing_stats folds one backend call's thread-local timing, texture, pipeline and
// reuse counters into the submit's RenderTiming (and, under timing logging, prints the cumulative
// buffer-reuse report). append_rtt_timing / print_rtt_timing format one PROSPER_RTT_TIMING record.
// These were lambdas in the submit callback; record_backend_timing stays there as a forwarder.

// The submit callback's state that record_backend_timing_stats reads and writes, one reference per object.
struct BackendTimingContext {
    RenderTiming& pending_timing;
    const prosper::frontend::PerformanceTimingMode& timing_mode;
};

void record_backend_timing_stats(BackendTimingContext& ctx,
                                 const prosper::test::BackendRenderTimingStats& backend,
                                 const prosper::test::BackendTextureUploadStats& textures,
                                 const prosper::test::BackendPipelineCacheStats& pipelines,
                                 const prosper::test::BackendResourceReuseStats& reuse);
void append_rtt_timing(std::string& output, const RttTimingRecord& record);
void print_rtt_timing(const RttTimingRecord& record);

} // namespace prosper::frontend::submit_renderer
