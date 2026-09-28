#pragma once
// Final-span scanout and present selection, carved out of the submit callback (#3892).

#include "shared/live/live_renderer_internal.hpp"        // RttCache, present policy and backend types
#include "shared/live/submit_renderer/callback_types.hpp" // the callback types these contexts name

namespace prosper::frontend::submit_renderer {

// ---- Final-span scanout / present selection (#3892) --------------------------------------------
//
// At a submit's final render span: GPU-present the flipped front buffer, or pick the CPU fallback
// frame (a cached scanout target, the guest's own display buffer, or the retained frame), and report
// a present-extent shortfall. Moved out of the submit callback verbatim.
// The submit callback's state that select_final_span_present reads and writes, one reference per object.
struct FinalSpanPresentContext {
    RttCache& g_rtt;
    uint32_t& w;
    uint32_t& h;
    std::shared_ptr<const std::vector<uint8_t>>& selected_pixels;
    uint64_t& selected_source_submit;
    prosper::gpu::PresentFrameOrigin& frame_origin;
    bool& published_gpu;
    const size_t& present_extent_bytes;
    prosper::frontend::PresentSourceChoice& present_choice;
    uint32_t& px_front_w;
    uint32_t& px_front_h;
    uint32_t& px_vo_w;
    uint32_t& px_vo_h;
    uint32_t& px_last_w;
    uint32_t& px_last_h;
    uint64_t& px_front_base;
    uint64_t& px_vo_base;
    uint64_t& px_last_base;
    uint32_t& px_front_fmt;
    uint32_t& px_vo_fmt;
    uint32_t& px_last_fmt;
};
void select_final_span_present(FinalSpanPresentContext& ctx);

} // namespace prosper::frontend::submit_renderer
