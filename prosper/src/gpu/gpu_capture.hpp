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

inline constexpr const char* kGpuReplayScanoutAddressEnv =
    "PROSPER_GPU_REPLAY_SCANOUT_ADDR";

// Preserve the exact registered front-buffer identity as replay metadata. The replay process has no
// live VideoOut registry, so extent alone cannot distinguish scanout from same-sized intermediates.
void annotate_gpu_capture_scanout(GpuCaptureMetadata& metadata);

// Parse the replay-only front-buffer identity. Bundle replay updates this value at each captured
// submit, so consumers must not cache the result across renderer callbacks.
uint64_t parse_gpu_replay_scanout_address(const char* value);

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

// Raw RDNA2 retained for realized graphics stages and failed operations. Each version is capped at
// 64 KiB and ends at the first decoded s_endpgm/unknown instruction or that cap. Equal byte streams
// share one version.
struct GpuCaptureRawShaderVersion {
    uint64_t content_hash = 0;
    bool has_endpgm = false;
    std::vector<uint32_t> words;
};

struct GpuCapturedResource {
    ShaderResource resource;
    // Exact base-resource byte span planned by the capture that produced this record. v1-v27 leave
    // this zero and materialize with their historical footprint rules.
    uint64_t captured_size = 0;
    uint32_t blob_index = 0xFFFFFFFFu;
    uint64_t blob_offset = 0;
    uint64_t metadata_size = 0;
    uint32_t metadata_blob_index = 0xFFFFFFFFu;
    uint64_t metadata_blob_offset = 0;
    // GPU-internal resources have no guest address. Capture their initial contents explicitly;
    // ordered replay materializes one shared instance so mutations remain visible to later work.
    std::vector<uint8_t> internal_bytes;
};

struct GpuCapturedTable {
    bool present = false;
    std::vector<GpuCapturedResource> resources;
};

struct GpuCapturedDraw {
    std::vector<uint32_t> vs;
    std::vector<uint32_t> gs;
    std::vector<uint32_t> fs;
    ResolvedPipelineState ps;
    GpuCapturedTable vrt;
    GpuCapturedTable prt;
    uint32_t vertex_count = 3;
    uint32_t instance_count = 1;
    // #1256 (v23): the RAW draw-packet count (DrawIndexAuto/DrawIndex index_count) and indexed flag as
    // decoded from the guest, BEFORE realization. For a non-indexed draw the realized vertex_count must
    // equal raw_draw_count; a divergence is a decode/realization bug (--inspect-only flags it). 0 = unknown.
    uint32_t raw_draw_count = 0;
    bool raw_indexed = false;
    uint64_t raw_draw_modifier = 0;
    int32_t vertex_offset = 0;
    std::vector<uint32_t> indices;
    uint64_t color0_base = 0;
    uint32_t color0_width = 0, color0_height = 0;
    uint64_t color1_base = 0;
    uint32_t color1_width = 0, color1_height = 0;
    // v34: complete RDNA2 color-target identity. Slots 0/1 mirror the named legacy fields.
    std::array<DrawItem::ColorTargetBinding, kColorTargetCount> color_targets{};
    uint64_t draw_index = 0;
    uint64_t command_order = 0;
    uint32_t vs_raw_shader_index = 0xFFFFFFFFu;
    uint32_t fs_raw_shader_index = 0xFFFFFFFFu;
    // v31: optional separately-installed vertex main stage and its exact graphics-LDS allocation.
    uint32_t vs_chain_raw_shader_index = 0xFFFFFFFFu;
    uint32_t vertex_lds_dwords = 0;
    // v36: exact fixed-function pixel-stage ABI required to recompile the retained raw graphics
    // programs. Older captures deliberately leave both unavailable.
    PixelInputMapping pixel_inputs{};
    PixelSystemInputMapping system_inputs{};
    bool has_pixel_inputs = false;
    bool has_system_inputs = false;
};

struct GpuCapturedCompute {
    std::vector<uint32_t> spirv;
    GpuCapturedTable resources;
    ComputeLaunchDimensions launch;
    uint64_t code_addr = 0;
    uint64_t dispatch_index = 0;
    uint64_t submit_no = 0;
    uint64_t command_order = 0;
    // v37: exact VkPipeline required-subgroup/full-subgroups contract used by the stored module.
    // Zero retains the historical portable shader path.
    uint32_t required_subgroup_size = 0;
    // v39: source and semantic launch ABI for deterministic current-translator replay. The optional
    // device capability fields describe the capture host; rendering intersects/replaces them with
    // the replay device's actual capabilities before recompiling.
    uint32_t raw_shader_index = 0xFFFFFFFFu;
    ComputeShaderConfig recompile_config{};
    bool recompile_config_available = false;
};

struct GpuCapturedOperation {
    SubmitOperationKind kind = SubmitOperationKind::Draw;
    uint64_t source_index = 0;
    uint64_t command_order = 0;
    bool realized = false;
};

// Ordered address-backed DMA_DATA. Guest addresses preserve the semantic identity; blob references
// bind the source and destination to their exact pre-submit backing versions for standalone replay.
struct GpuCapturedDmaCopy {
    uint64_t dst = 0, src = 0;
    uint32_t bytes = 0, sels = 0;
    uint64_t command_order = 0;
    uint64_t packet_addr = 0;
    uint32_t destination_blob_index = 0xFFFFFFFFu;
    uint64_t destination_blob_offset = 0;
    uint32_t source_blob_index = 0xFFFFFFFFu;
    uint64_t source_blob_offset = 0;
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
    // v35: the exact descriptor metadata supplied to the failed recompiler invocation. Earlier
    // captures retained only presence/count, which was enough to describe the failure but not to
    // retry it offline: table-dependent SMEM/MUBUF/MIMG lowering rejected before reaching the real
    // unsupported instruction. Resource bytes remain deliberately absent from this diagnostic
    // table; once a fix realizes the operation, an ordinary follow-up capture retains its renderable
    // resource closure through GpuCapturedDraw/GpuCapturedCompute.
    GpuCapturedTable resource_table;
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
    // v34: complete failed-draw target identity. Slots 0/1 mirror the named legacy fields.
    std::array<DrawItem::ColorTargetBinding, kColorTargetCount> color_targets{};
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
    R11G11B10Float = 3,
    R8Unorm = 4,
    R32Uint = 5,
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
    // In-memory provenance for version-dependent materialization. This is populated by capture/read;
    // the on-disk header remains the single serialized version field.
    uint32_t format_version = 0;
    GpuCaptureMetadata metadata;
    std::vector<GpuCaptureBlob> blobs;
    std::vector<GpuCaptureShaderVersion> shader_versions;
    std::vector<GpuCaptureRawShaderVersion> raw_shader_versions;
    std::vector<GpuCaptureRttSeed> rtt_seeds;
    std::vector<GpuCaptureDsSeed> ds_seeds;
    std::vector<GpuCapturedDraw> draws;
    std::vector<GpuCapturedCompute> computes;
    std::vector<GpuCapturedDmaCopy> dma_copies;
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
                          const std::vector<OperationRealizationFailure>& failures = {},
                          const std::vector<GpuState::DmaCopy>& dma_copies = {},
                          uint64_t resource_limit_bytes = 0);
bool capture_gpustate_submit(const GpuState& state, uint64_t submit_no,
                             uint32_t width, uint32_t height,
                             const GpuCaptureMetadata& metadata,
                             GpuCaptureFile& out, std::string& error,
                             uint64_t resource_limit_bytes = 0);
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
    std::vector<ReplayDmaCopy> dma_copies;
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
// Add one exact live renderer target to a capture unless it is already dependency-seeded. Used by
// frame-boundary capture for a held VideoOut scanout that may predate every submit in the window.
bool capture_gpu_rtt_seed(GpuCaptureFile& capture, uint64_t guest_addr, std::string& error);
void set_gpu_capture_rtt_seed_snapshot_reader(CaptureRttSeedSnapshotReader reader);
bool read_all_gpu_capture_rtt_seeds(std::vector<GpuCaptureRttSeed>& seeds, std::string& error);
void set_gpu_replay_rtt_seed_writer(ReplayRttSeedWriter writer);
bool restore_gpu_replay_rtt_seeds(const std::vector<GpuCaptureRttSeed>& seeds, std::string& error);
void set_gpu_capture_ds_seed_snapshot_reader(CaptureDsSeedSnapshotReader reader);
bool read_all_gpu_capture_ds_seeds(std::vector<GpuCaptureDsSeed>& seeds, std::string& error);
bool gpu_capture_ds_seed_snapshot_available();
// Add only live depth/stencil checkpoints referenced by this capsule's realized draws. Standalone
// pre-render captures otherwise replay a read-only depth pass against a newly-cleared attachment.
bool capture_referenced_gpu_ds_seeds(GpuCaptureFile& capture, std::string& error);
void set_gpu_replay_ds_seed_writer(ReplayDsSeedWriter writer);
bool restore_gpu_replay_ds_seeds(const std::vector<GpuCaptureDsSeed>& seeds, std::string& error);
uint64_t gpu_capture_hash(const uint8_t* data, size_t size);
inline uint64_t gpu_capture_hash(const std::vector<uint8_t>& data) {
    return gpu_capture_hash(data.data(), data.size());
}
bool gpu_capture_output_nonzero_matches(const std::vector<uint8_t>& output, size_t min_nonzero,
                                        size_t max_nonzero, size_t* observed = nullptr);
const char* shader_program_stage_name(ShaderProgramStage stage);
const char* realization_failure_reason_name(RealizationFailureReason reason);

// Runtime hook used by execute_and_present. PROSPER_GPU_CAPTURE=<path> captures exactly one realized
// submit. MIN_DRAWS/MAX_DRAWS, COMPUTE_ADDR, SHADER_ADDR, and TARGET_DIM select a candidate class;
// PROSPER_GPU_CAPTURE_AT=N then selects the zero-based matching invocation (default first match).
struct PendingGpuCapture {
    struct MemorySnapshot {
        uint64_t guest_addr = 0;
        size_t bytes_read = 0;
        std::vector<uint8_t> bytes;
    };
    std::string path;
    GpuCaptureFile capture;
    // Hand-built pending captures used by tests/tools already carry a complete capture. Runtime
    // begin_requested_gpu_capture explicitly clears this while it owns metadata only.
    bool materialized = true;
    bool output_triggered = false;
    size_t output_min_nonzero = 0;
    size_t output_max_nonzero = 0;
    // Deferred captures finish after ordered producers have mutated memory. Preserve the known
    // pre-submit DMA endpoints and backend-owned GDS here, then combine them with the exact
    // post-producer operation realization during finish.
    std::vector<MemorySnapshot> pre_submit_memory;
    std::vector<uint8_t> pre_submit_compute_gds;
};
void snapshot_pending_gpu_capture_compute_gds(PendingGpuCapture* pending,
                                              const uint8_t* data, size_t bytes);
std::unique_ptr<PendingGpuCapture> begin_requested_gpu_capture(
    const std::vector<DrawItem>& draws, const std::vector<ComputeItem>& computes,
    const std::vector<SubmitOperation>& operations, uint32_t width, uint32_t height,
    const GpuState* semantic_state = nullptr, uint64_t submit_no = 0,
    uint64_t semantic_draw_count = UINT64_MAX,
    const std::vector<OperationRealizationFailure>* exact_failures = nullptr,
    bool defer_materialization = false);
bool finish_requested_gpu_capture(std::unique_ptr<PendingGpuCapture> pending,
                                  const std::vector<uint8_t>& output, std::string& error,
                                  const std::vector<DrawItem>* draws = nullptr,
                                  const std::vector<ComputeItem>* computes = nullptr,
                                  const std::vector<SubmitOperation>* operations = nullptr,
                                  const GpuState* semantic_state = nullptr,
                                  const std::vector<OperationRealizationFailure>* exact_failures = nullptr);

// Interactive one-shot capture (frame_grab tool): arm a capture from a hotkey WITHOUT predicting a
// submit index ahead of time. The next drawing invocation reaching begin_requested_gpu_capture() writes
// a .prgcap to `path` and disarms, bypassing the env AT/AFTER/MIN selectors. Thread-safe: arm on the
// app's main thread, consumed on the render thread. Off unless armed; the env PROSPER_GPU_CAPTURE path
// is unaffected. interactive_gpu_capture_armed() lets the caller (e.g. the app) know a grab is pending
// so it can also snapshot the presented pixels for that frame.
void request_interactive_gpu_capture(const std::string& path);
bool interactive_gpu_capture_armed();

} // namespace prosper::gpu
