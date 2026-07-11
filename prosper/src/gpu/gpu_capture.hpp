// gpu_capture.hpp - versioned, local-only capture of realized GPU draws.
#pragma once

#include "gpu_execute.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace prosper::gpu {

struct GpuCaptureMetadata {
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t submit_index = 0;
    std::string revision;
    std::string title_id;
    std::string input_route;
    std::string savedata_dir;
    std::vector<std::pair<std::string, std::string>> renderer_env;
};

struct GpuCaptureBlob {
    uint64_t guest_addr = 0;
    uint64_t bytes_read = 0;
    std::vector<uint8_t> bytes;
};

struct GpuCapturedResource {
    ShaderResource resource;
    uint32_t blob_index = 0xFFFFFFFFu;
    uint64_t blob_offset = 0;
};

struct GpuCapturedTable {
    bool present = false;
    std::vector<GpuCapturedResource> resources;
};

struct GpuCapturedDraw {
    std::vector<uint32_t> vs;
    std::vector<uint32_t> fs;
    ResolvedPipelineState ps;
    GpuCapturedTable vrt;
    GpuCapturedTable prt;
    uint32_t vertex_count = 3;
    std::vector<uint32_t> indices;
    uint64_t color0_base = 0;
};

struct GpuCaptureFile {
    GpuCaptureMetadata metadata;
    std::vector<GpuCaptureBlob> blobs;
    std::vector<GpuCapturedDraw> draws;
    uint64_t expected_output_hash = 0;
    uint64_t expected_output_bytes = 0;
};

// Reader returns the number of bytes copied. The capture zero-fills the unread suffix, matching the
// live renderer's guarded-copy behavior for partially committed guest resources.
using CaptureMemoryReader = std::function<size_t(uint64_t guest_addr, uint8_t* dst, size_t bytes)>;

bool capture_draw_items(const std::vector<DrawItem>& items, const GpuCaptureMetadata& metadata,
                        const CaptureMemoryReader& reader, GpuCaptureFile& out, std::string& error);
bool write_gpu_capture(const std::string& path, const GpuCaptureFile& capture, std::string& error);
bool read_gpu_capture(const std::string& path, GpuCaptureFile& capture, std::string& error);

struct GpuReplayFrame {
    GpuCaptureMetadata metadata;
    std::vector<GpuCaptureBlob> blobs;
    std::vector<DrawItem> items;
    uint64_t expected_output_hash = 0;
    uint64_t expected_output_bytes = 0;
};

bool materialize_gpu_replay(const GpuCaptureFile& capture, GpuReplayFrame& replay, std::string& error);
uint64_t gpu_capture_hash(const uint8_t* data, size_t size);
inline uint64_t gpu_capture_hash(const std::vector<uint8_t>& data) {
    return gpu_capture_hash(data.data(), data.size());
}

// Runtime hook used by execute_and_present. PROSPER_GPU_CAPTURE=<path> captures exactly one realized
// submit. MIN_DRAWS/MAX_DRAWS select a semantic candidate class; PROSPER_GPU_CAPTURE_AT=N then selects
// the zero-based matching invocation (default first match).
struct PendingGpuCapture {
    std::string path;
    GpuCaptureFile capture;
};
std::unique_ptr<PendingGpuCapture> begin_requested_gpu_capture(const std::vector<DrawItem>& items,
                                                               uint32_t width, uint32_t height);
bool finish_requested_gpu_capture(std::unique_ptr<PendingGpuCapture> pending,
                                  const std::vector<uint8_t>& output, std::string& error);

} // namespace prosper::gpu
