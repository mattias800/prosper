#pragma once
// The pass loop's MRT slot queries, shared by the loop and its diagnostics (#3892).

#include "shared/live/live_renderer_internal.hpp" // mrt_format_defined, mrt_binding.hpp, backend_color_format

namespace prosper::frontend::submit_renderer {

// Pass grouping and same-pass feedback detection must not disagree about what an
// active binding is, so both go through frontends/shared/rtt/mrt_binding.hpp. These
// were duplicated lambdas; a second, looser copy in the feedback path classified
// stale named state as a live binding (#2550 review).
inline auto color_binding(const prosper::gpu::DrawItem& draw, uint32_t slot) {
    return prosper::frontend::mrt_color_binding(draw, slot);
}

inline auto active_format(const prosper::gpu::DrawItem& draw, uint32_t slot) {
    return prosper::test::backend_color_format(static_cast<VkFormat>(
        prosper::frontend::mrt_raw_format(draw, slot)));
}

inline auto active_color(const prosper::gpu::DrawItem& draw, uint32_t slot) {
    return prosper::frontend::mrt_active_color(draw, slot, mrt_format_defined);
}

inline auto active_color_count(const prosper::gpu::DrawItem& draw) {
    return prosper::frontend::mrt_active_color_count(draw, mrt_format_defined);
}

} // namespace prosper::frontend::submit_renderer
