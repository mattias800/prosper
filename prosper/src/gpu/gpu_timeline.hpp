// gpu_timeline.hpp - native-speed semantic submit/present timeline recording.
#pragma once

#include "command_processor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace prosper::gpu {

struct GpuTimelineMetadata {
    std::string revision;
    std::string title_id;
    std::string input_route;
};

struct GpuTimelineSubmit {
    uint64_t sequence = 0;
    uint64_t elapsed_ns = 0;
    uint64_t submit_no = 0;
    uint64_t process_command_order = 0;
    uint64_t first_command_order = 0;
    uint64_t last_command_order = 0;
    uint64_t color0_base = 0;
    uint32_t draw_count = 0;
    uint32_t dispatch_count = 0;
    uint32_t color0_width = 0;
    uint32_t color0_height = 0;
};

struct GpuTimelinePresent {
    uint64_t sequence = 0;
    uint64_t elapsed_ns = 0;
    uint64_t present_count = 0;
    uint64_t latest_submit_no = 0;
    int32_t buffer_index = -1;
    int64_t flip_arg = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct GpuTimelineFile {
    GpuTimelineMetadata metadata;
    std::vector<GpuTimelineSubmit> submits;
    std::vector<GpuTimelinePresent> presents;
    bool truncated_tail = false;
};

class GpuTimelineWriter {
public:
    GpuTimelineWriter();
    ~GpuTimelineWriter();
    GpuTimelineWriter(const GpuTimelineWriter&) = delete;
    GpuTimelineWriter& operator=(const GpuTimelineWriter&) = delete;

    bool open(const std::string& path, const GpuTimelineMetadata& metadata, std::string& error);
    bool append_submit(const GpuTimelineSubmit& submit, std::string& error);
    bool append_present(const GpuTimelinePresent& present, std::string& error);
    bool flush(std::string& error);
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool read_gpu_timeline(const std::string& path, GpuTimelineFile& timeline, std::string& error);

// Runtime hooks. Inert unless PROSPER_GPU_TIMELINE=<path> is set. They record folded semantic state
// before renderer sampling and never realize shaders, copy resources, or invoke Vulkan.
void begin_gpu_timeline_submit(uint64_t submit_no);
void record_gpu_timeline_submit(const GpuState& state, uint64_t submit_no);
void record_gpu_timeline_present(uint64_t present_count, int buffer_index, int64_t flip_arg,
                                 uint32_t width, uint32_t height);
void flush_gpu_timeline();
void close_gpu_timeline();

} // namespace prosper::gpu
