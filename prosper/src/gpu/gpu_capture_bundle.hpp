// gpu_capture_bundle.hpp - content-deduplicated ordered submit captures.
#pragma once

#include "gpu_capture.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {

struct GpuCaptureBundleSubmit {
    uint64_t submit_index = 0;
    uint64_t logical_bytes = 0;
    std::vector<uint32_t> chunk_indices;
};

struct GpuCaptureBundle {
    uint32_t version = 1;
    uint32_t chunk_bytes = 256 * 1024;
    uint64_t logical_bytes = 0;
    std::vector<uint64_t> chunk_hashes;
    std::vector<std::vector<uint8_t>> chunks;
    std::vector<GpuCaptureBundleSubmit> submits;
};

bool append_gpu_capture_bundle(GpuCaptureBundle& bundle, const GpuCaptureFile& capture,
                               std::string& error);
bool materialize_gpu_capture_bundle_submit(const GpuCaptureBundle& bundle, size_t submit_index,
                                           GpuCaptureFile& capture, std::string& error);
bool write_gpu_capture_bundle(const std::string& path, const GpuCaptureBundle& bundle,
                              std::string& error);
bool read_gpu_capture_bundle(const std::string& path, GpuCaptureBundle& bundle,
                             std::string& error);
uint64_t gpu_capture_bundle_unique_bytes(const GpuCaptureBundle& bundle);

} // namespace prosper::gpu
