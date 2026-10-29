// Capture and restore values, without retaining a cache entry or claiming live guest layout.
#include "shared/live/submit_renderer/rtt_dcc_capture.hpp"
#include "shared/live/live_renderer_internal.hpp"

namespace prosper::frontend::submit_renderer {
prosper::gpu::GpuCaptureRttDccMetadata capture_rtt_dcc_metadata(const RttSurf& s) {
    return {s.dcc_metadata_addr,   s.dcc_metadata_bytes, s.dcc_num_components,
            s.dcc_alpha_is_on_msb, s.dcc_width,          s.dcc_height};
}
void restore_rtt_dcc_metadata(RttSurf& s, const prosper::gpu::GpuCaptureRttDccMetadata& m,
                              bool same_target) {
    // A pixel-only temporal seed does not revoke a descriptor already learned for this target.
    // A fresh entry, changed identity or an explicitly supplied association cannot borrow it.
    if (same_target && m == prosper::gpu::GpuCaptureRttDccMetadata{}) return;
    s.dcc_metadata_addr = m.metadata_addr;
    s.dcc_metadata_bytes = m.metadata_bytes;
    s.dcc_num_components = m.num_components;
    s.dcc_alpha_is_on_msb = m.alpha_is_on_msb;
    s.dcc_width = m.guest_width;
    s.dcc_height = m.guest_height;
    // Preserve old unpublished-origin history, as the live descriptor path does.
    if (m.metadata_addr && m.metadata_bytes)
        s.dcc_guest_origins.observe(m.metadata_addr, m.metadata_bytes);
}
} // namespace prosper::frontend::submit_renderer
