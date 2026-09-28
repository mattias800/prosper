#pragma once
// The PROSPER_RENDER_TIMING aggregate report, carved out of the submit-renderer callback in live_renderer.cpp (#3892).

#include "shared/live/live_renderer_internal.hpp"        // RttCache, texture-decode cache types
#include "shared/live/submit_renderer/callback_types.hpp" // the callback types this context names

namespace prosper::frontend::submit_renderer {

// ---- PROSPER_RENDER_TIMING aggregate report (#3892) ---------------------------------------------
//
// Folds one complete semantic submit's RenderTiming into the process-lifetime and windowed
// totals and prints the periodic [render-timing] summaries. This was a block inside the submit
// callback, reached only when a timing log is active; it moved here verbatim.
// The submit callback's state that report_render_timing_aggregates reads and resets.
struct RenderTimingReportContext {
    RttCache& g_rtt;
    RenderTiming& pending_timing;
    std::vector<std::vector<uint8_t>>& texstore;
    const bool& submit_decode_scope_disabled;
    std::unordered_map<TextureDecodeKey, PersistentDecodedTexture, TextureDecodeKeyHash>& persistent_decoded_textures;
    size_t& persistent_decoded_texture_bytes;
    std::vector<uint8_t>& persistent_validation_scratch;
};

void report_render_timing_aggregates(RenderTimingReportContext& ctx);

// ---- F8 renderer timing record (#3892) ----------------------------------------------------------
//
// Copies one complete semantic submit's RenderTiming into the interactive performance capture's
// RendererTimingRecord. Moved out of the submit callback verbatim; its gate stays at the call site.
// The submit callback's state that publish_renderer_timing_record reads and writes, one reference per object.
struct RendererTimingRecordContext {
    RenderTiming& pending_timing;
    uint64_t& pending_span_start_ns;
    uint64_t& pending_capture_generation;
};
void publish_renderer_timing_record(RendererTimingRecordContext& ctx);

} // namespace prosper::frontend::submit_renderer
