#pragma once
#include "gpu/capture/rtt_dcc_metadata.hpp"

namespace prosper::frontend {
struct RttSurf;
namespace submit_renderer {
prosper::gpu::GpuCaptureRttDccMetadata capture_rtt_dcc_metadata(const RttSurf&);
void restore_rtt_dcc_metadata(RttSurf&, const prosper::gpu::GpuCaptureRttDccMetadata&);
} // namespace submit_renderer
} // namespace prosper::frontend
