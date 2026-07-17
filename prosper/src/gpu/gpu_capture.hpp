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
    uint64_t content_hash = 0;
    std::vector<uint8_t> bytes;
};

struct GpuCaptureShaderVersion {
    uint64_t content_hash = 0;
    std::vector<uint32_t> words;
};

// Raw RDNA2 retained only for failed operations. Each version is capped at 64 KiB and ends at the
// first decoded s_endpgm/unknown instruction or that cap. Equal byte streams share one version.
struct GpuCaptureRawShaderVersion {
    uint64_t content_hash = 0;
    bool has_endpgm = false;
    std::vector<uint32_t> words;
};

struct GpuCapturedResource {
    ShaderResource resource;
    uint32_t blob_index = 0xFFFFFFFFu;
    uint64_t blob_offset = 0;
    uint64_t metadata_size = 0;
    uint32_t metadata_blob_index = 0xFFFFFFFFu;
    uint64_t metadata_blob_offset = 0;
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
    uint32_t color0_width = 0, color0_height = 0;
    uint64_t color1_base = 0;
    uint32_t color1_width = 0, color1_height = 0;
    uint64_t draw_index = 0;
    uint64_t command_order = 0;
};

struct GpuCapturedCompute {
    std::vector<uint32_t> spirv;
    GpuCapturedTable resources;
    ComputeLaunchDimensions launch;
    uint64_t code_addr = 0;
    uint64_t dispatch_index = 0;
    uint64_t submit_no = 0;
    uint64_t command_order = 0;
};

struct GpuCapturedOperation {
    SubmitOperationKind kind = SubmitOperationKind::Draw;
    uint64_t source_index = 0;
    uint64_t command_order = 0;
    bool realized = false;
};

struct GpuCapturedStageDiagnostic {
    ShaderProgramStage stage = ShaderProgramStage::Vertex;
    uint64_t program_addr = 0;
    uint32_t raw_shader_index = 0xFFFFFFFFu;
    bool recompiled = false;
    bool resource_table_present = false;
    uint32_t resource_count = 0;
    RecompileCoverage coverage;
    uint32_t descriptor_issue_count = 0;
    uint32_t first_descriptor_issue = 0xFFFFFFFFu;
};

struct GpuCapturedOperationFailure {
    SubmitOperationKind kind = SubmitOperationKind::Draw;
    uint64_t source_index = 0;
    uint64_t command_order = 0;
    RealizationFailureReason reason = RealizationFailureReason::None;
    bool pipeline_present = false;
    ResolvedPipelineState pipeline;
    uint64_t color0_base = 0;
    uint32_t color0_width = 0;
    uint32_t color0_height = 0;
    uint64_t color1_base = 0;
    uint32_t color1_width = 0;
    uint32_t color1_height = 0;
    uint32_t vertex_count = 0;
    ComputeLaunchDimensions compute_launch;
    std::vector<GpuCapturedStageDiagnostic> stages;
};

// Host-rendered pixels retained across submits. The HLE renderer does not write these pixels back to
// guest memory, so a standalone replay must seed its RTT cache explicitly to reproduce consumers whose
// producer was in an earlier submit.
enum class GpuCaptureColorFormat : uint32_t {
    Rgba8Unorm = 1,
    Rgba16Float = 2,
};

struct GpuCaptureRttSeed {
    uint64_t guest_addr = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    GpuCaptureColorFormat format = GpuCaptureColorFormat::Rgba8Unorm;
    std::vector<uint8_t> rgba;
};

enum class GpuCaptureDsFormat : uint32_t {
    D32Float = 1,
    D32FloatS8 = 2,
};

// Exact host depth/stencil attachment contents retained across submits. These planes are detached
// from guest tiled memory, just like RTT pixels, so a final-submit capsule needs an explicit copy of
// every valid plane selected by the persistent renderer cache.
struct GpuCaptureDsSeed {
    uint64_t depth_read_base = 0;
    uint64_t depth_write_base = 0;
    uint64_t stencil_read_base = 0;
    uint64_t stencil_write_base = 0;
    uint64_t htile_data_base = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    GpuCaptureDsFormat format = GpuCaptureDsFormat::D32Float;
    bool depth_valid = false;
    bool stencil_valid = false;
    std::vector<uint8_t> depth;
    std::vector<uint8_t> stencil;
};

struct GpuCaptureFile {
    GpuCaptureMetadata metadata;
    std::vector<GpuCaptureBlob> blobs;
    std::vector<GpuCaptureShaderVersion> shader_versions;
    std::vector<GpuCaptureRawShaderVersion> raw_shader_versions;
    std::vector<GpuCaptureRttSeed> rtt_seeds;
    std::vector<GpuCaptureDsSeed> ds_seeds;
    std::vector<GpuCapturedDraw> draws;
    std::vector<GpuCapturedCompute> computes;
    std::vector<GpuCapturedOperation> operations;
    std::vector<GpuCapturedOperationFailure> failure_diagnostics;
    bool failure_diagnostics_available = false;
    bool expected_output_valid = false;
    uint64_t expected_output_hash = 0;
    uint64_t expected_output_bytes = 0;
};

// Reader returns the number of bytes copied. The capture zero-fills the unread suffix, matching the
// live renderer's guarded-copy behavior for partially committed guest resources.
using CaptureMemoryReader = std::function<size_t(uint64_t guest_addr, uint8_t* dst, size_t bytes)>;
using CaptureRttSeedReader = std::function<bool(uint64_t guest_addr, GpuCaptureRttSeed& seed)>;
using CaptureRttSeedSnapshotReader =
    std::function<bool(std::vector<GpuCaptureRttSeed>& seeds, std::string& error)>;
using CaptureDsSeedSnapshotReader =
    std::function<bool(std::vector<GpuCaptureDsSeed>& seeds, std::string& error)>;

bool capture_draw_items(const std::vector<DrawItem>& items, const GpuCaptureMetadata& metadata,
                        const CaptureMemoryReader& reader, GpuCaptureFile& out, std::string& error,
                        const CaptureRttSeedReader& rtt_reader = {});
bool capture_submit_items(const std::vector<DrawItem>& draws,
                          const std::vector<ComputeItem>& computes,
                          const std::vector<SubmitOperation>& operations,
                          const GpuCaptureMetadata& metadata,
                          const CaptureMemoryReader& reader, GpuCaptureFile& out,
                          std::string& error, const CaptureRttSeedReader& rtt_reader = {},
                          const std::vector<OperationRealizationFailure>& failures = {});
bool capture_gpustate_submit(const GpuState& state, uint64_t submit_no,
                             uint32_t width, uint32_t height,
                             const GpuCaptureMetadata& metadata,
                             GpuCaptureFile& out, std::string& error);
bool capture_gpustate_target_submit(const GpuState& state, uint64_t submit_no,
                                    uint32_t width, uint32_t height,
                                    uint32_t target_width, uint32_t target_height,
                                    const GpuCaptureMetadata& metadata,
                                    GpuCaptureFile& out, std::string& error);
bool serialize_gpu_capture(const GpuCaptureFile& capture, std::vector<uint8_t>& bytes,
                           std::string& error);
bool deserialize_gpu_capture(const std::vector<uint8_t>& bytes, GpuCaptureFile& capture,
                             std::string& error);
bool write_gpu_capture(const std::string& path, const GpuCaptureFile& capture, std::string& error);
bool read_gpu_capture(const std::string& path, GpuCaptureFile& capture, std::string& error);

struct GpuReplayFrame {
    struct ResourceInstance {
        uint64_t guest_addr = 0;
        uint32_t blob_index = 0xFFFFFFFFu;
        std::vector<uint8_t> bytes;
    };
    GpuCaptureMetadata metadata;
    std::vector<GpuCaptureBlob> blobs;
    std::vector<ResourceInstance> resource_instances;
    std::vector<GpuCaptureRttSeed> rtt_seeds;
    std::vector<GpuCaptureDsSeed> ds_seeds;
    std::vector<DrawItem> items;
    std::vector<ComputeItem> computes;
    std::vector<GpuCapturedOperation> operations;
    std::vector<GpuCaptureRawShaderVersion> raw_shader_versions;
    std::vector<GpuCapturedOperationFailure> failure_diagnostics;
    bool failure_diagnostics_available = false;
    bool expected_output_valid = false;
    uint64_t expected_output_hash = 0;
    uint64_t expected_output_bytes = 0;
};

bool materialize_gpu_replay(const GpuCaptureFile& capture, GpuReplayFrame& replay, std::string& error);

// Byte range the capture planner reserves for one descriptor. This includes decoded/tiled image
// storage when it is larger than the descriptor's declared byte count.
uint64_t gpu_capture_resource_footprint(const ShaderResource& resource);

// Exact supported GFX10 DCC control-surface span for one descriptor, or zero when the descriptor is
// uncompressed or outside the currently validated SW_64KB_R_X single-sample/base-level contract.
uint64_t gpu_capture_dcc_metadata_footprint(const ShaderResource& resource);

// The Vulkan frontend owns the RTT and persistent DS caches and registers these hooks when its renderer
// is installed. Normal capture queries only RTT addresses referenced by the selected submit; final bundle
// export snapshots both complete caches. Replay restores serialized surfaces before draw zero.
using ReplayRttSeedWriter = std::function<bool(const GpuCaptureRttSeed& seed, std::string& error)>;
using ReplayDsSeedWriter = std::function<bool(const GpuCaptureDsSeed& seed, std::string& error)>;
void set_gpu_capture_rtt_seed_reader(CaptureRttSeedReader reader);
bool read_gpu_capture_rtt_seed(uint64_t guest_addr, GpuCaptureRttSeed& seed, std::string& error);
void set_gpu_capture_rtt_seed_snapshot_reader(CaptureRttSeedSnapshotReader reader);
bool read_all_gpu_capture_rtt_seeds(std::vector<GpuCaptureRttSeed>& seeds, std::string& error);
void set_gpu_replay_rtt_seed_writer(ReplayRttSeedWriter writer);
bool restore_gpu_replay_rtt_seeds(const std::vector<GpuCaptureRttSeed>& seeds, std::string& error);
void set_gpu_capture_ds_seed_snapshot_reader(CaptureDsSeedSnapshotReader reader);
bool read_all_gpu_capture_ds_seeds(std::vector<GpuCaptureDsSeed>& seeds, std::string& error);
void set_gpu_replay_ds_seed_writer(ReplayDsSeedWriter writer);
bool restore_gpu_replay_ds_seeds(const std::vector<GpuCaptureDsSeed>& seeds, std::string& error);
uint64_t gpu_capture_hash(const uint8_t* data, size_t size);
inline uint64_t gpu_capture_hash(const std::vector<uint8_t>& data) {
    return gpu_capture_hash(data.data(), data.size());
}
const char* shader_program_stage_name(ShaderProgramStage stage);
const char* realization_failure_reason_name(RealizationFailureReason reason);

// Runtime hook used by execute_and_present. PROSPER_GPU_CAPTURE=<path> captures exactly one realized
// submit. MIN_DRAWS/MAX_DRAWS select a semantic candidate class; PROSPER_GPU_CAPTURE_AT=N then selects
// the zero-based matching invocation (default first match).
struct PendingGpuCapture {
    std::string path;
    GpuCaptureFile capture;
};
std::unique_ptr<PendingGpuCapture> begin_requested_gpu_capture(
    const std::vector<DrawItem>& draws, const std::vector<ComputeItem>& computes,
    const std::vector<SubmitOperation>& operations, uint32_t width, uint32_t height,
    bool has_ordered_dma = false);
bool finish_requested_gpu_capture(std::unique_ptr<PendingGpuCapture> pending,
                                  const std::vector<uint8_t>& output, std::string& error);

} // namespace prosper::gpu
