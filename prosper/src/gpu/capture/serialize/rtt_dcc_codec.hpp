// v79 append-only tail: u32 seed count; for each seed, u64 metadata VA, u64 plane bytes,
// u32 components, u32 alpha-MSB (0/1), u32 guest width, u32 guest height. All little-endian.
// An all-zero record means unavailable. A known range with zero components/alpha/dimensions
// retains plane-only registration without inventing its interpretation. Other records need all
// interpretation fields. Entries follow the original RTT-seed vector order.
#pragma once
#include "gpu/capture/gpu_capture.hpp"
#include <string>
#include <vector>

namespace prosper::gpu {
struct Writer;
struct Reader;
bool write_rtt_dcc_metadata(Writer&, const std::vector<GpuCaptureRttSeed>&, std::string&);
bool read_rtt_dcc_metadata(Reader&, std::vector<GpuCaptureRttSeed>&, std::string&);
} // namespace prosper::gpu
