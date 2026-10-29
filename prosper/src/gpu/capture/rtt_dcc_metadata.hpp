// Descriptor facts retained with pixels so replay can identify later DCC metadata writes.
// Zero components/dimensions retain a known plane whose descriptor interpretation is unavailable.
#pragma once
#include <cstdint>

namespace prosper::gpu {
struct GpuCaptureRttDccMetadata {
    uint64_t metadata_addr = 0;
    uint64_t metadata_bytes = 0;
    uint32_t num_components = 0;
    bool alpha_is_on_msb = false;
    uint32_t guest_width = 0;
    uint32_t guest_height = 0;
    bool operator==(const GpuCaptureRttDccMetadata&) const = default;
};
} // namespace prosper::gpu
