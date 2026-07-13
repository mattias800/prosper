// gpu_capture_bundle.hpp - content-deduplicated ordered submit captures.
#pragma once

#include "gpu_capture.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace prosper::gpu {

struct GpuCaptureBundleSubmit {
    uint64_t submit_index = 0;
    uint64_t logical_bytes = 0;
    uint64_t manifest_bytes = 0;
    std::vector<uint32_t> chunk_indices;
    std::vector<uint32_t> blob_resource_indices;
    std::vector<uint64_t> blob_bytes_read;
};

struct GpuCaptureBundleResource {
    uint64_t content_hash = 0;
    uint64_t logical_bytes = 0;
    std::vector<uint32_t> chunk_indices;
};

struct GpuCaptureBundle {
    uint32_t version = 2;
    uint32_t chunk_bytes = 256 * 1024;
    uint64_t logical_bytes = 0;
    std::vector<uint64_t> chunk_hashes;
    std::vector<std::vector<uint8_t>> chunks;
    std::vector<GpuCaptureBundleResource> resources;
    std::vector<GpuCaptureBundleSubmit> submits;
    // Runtime append indexes. They are rebuilt on read and are not serialized.
    std::unordered_map<uint64_t, std::vector<uint32_t>> chunk_indices_by_hash;
    std::unordered_map<uint64_t, std::vector<uint32_t>> resource_indices_by_hash;
};

struct GpuCaptureBundleStats {
    uint64_t manifest_unique_bytes = 0;
    uint64_t resource_unique_bytes = 0;
    uint64_t resource_logical_bytes = 0;
    uint64_t resource_reference_count = 0;
    uint64_t exact_reuse_count = 0;
};

bool append_gpu_capture_bundle(GpuCaptureBundle& bundle, const GpuCaptureFile& capture,
                               std::string& error);
bool materialize_gpu_capture_bundle_manifest(const GpuCaptureBundle& bundle, size_t submit_index,
                                             GpuCaptureFile& capture, std::string& error);
bool materialize_gpu_capture_bundle_submit(const GpuCaptureBundle& bundle, size_t submit_index,
                                           GpuCaptureFile& capture, std::string& error);
bool compact_gpu_capture_bundle(GpuCaptureBundle& bundle, std::string& error);
bool write_gpu_capture_bundle(const std::string& path, const GpuCaptureBundle& bundle,
                              std::string& error);
bool read_gpu_capture_bundle(const std::string& path, GpuCaptureBundle& bundle,
                             std::string& error);
uint64_t gpu_capture_bundle_unique_bytes(const GpuCaptureBundle& bundle);
GpuCaptureBundleStats gpu_capture_bundle_stats(const GpuCaptureBundle& bundle);

} // namespace prosper::gpu
