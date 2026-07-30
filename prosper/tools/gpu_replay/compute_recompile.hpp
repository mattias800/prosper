#pragma once

#include "gpu/gpu_capture.hpp"

namespace prosper::tools {

// gpu_replay has several one-capsule diagnostic modes that return before renderer execution. Keep
// that command-flow decision explicit and testable: every other invocation reaches the renderer,
// including the ordinary one-positional `gpu_replay capture.prgcap` form and stored-SPIR-V/resource
// dumps which historically continue into replay.
struct ReplayRenderIntent {
    bool metadata_only = false;
    bool graph_only = false;
    bool validate_only = false;
    bool inspect_only = false;
    bool inspect = false;
    size_t positional_count = 0;
    bool realized_shader_dump = false;
    bool failed_shader_dump = false;
    bool retry_failed_chain = false;
    bool retry_failed_stage = false;
};

inline bool replay_will_render(const ReplayRenderIntent& intent) {
    if (intent.metadata_only || intent.graph_only || intent.validate_only || intent.inspect_only)
        return false;
    const bool terminal_one_capsule_diagnostic = intent.positional_count == 1 && !intent.inspect &&
        (intent.realized_shader_dump || intent.failed_shader_dump ||
         intent.retry_failed_chain || intent.retry_failed_stage);
    return !terminal_one_capsule_diagnostic;
}

// Replace one capture v39 compute module with output from the current translator. A rendering
// replay supplies its initialized device so optional storage formats and subgroup contracts are
// selected from capabilities that are actually enabled there. Inspection/dump callers may omit the
// device and reproduce the capture host's recorded translator policy without initializing Vulkan.
inline bool recompile_captured_compute(
    gpu::ComputeItem& compute,
    const std::vector<gpu::GpuCaptureRawShaderVersion>& raw_shader_versions,
    const gpu::SharedVulkanContext* replay_device,
    bool force_native_multiwave,
    bool disable_native_subgroup,
    bool packed_r11_storage) {
    if (!compute.recompile_config_available ||
        compute.raw_shader_index >= raw_shader_versions.size())
        return false;

    const auto& raw = raw_shader_versions[compute.raw_shader_index];
    gpu::ComputeShaderConfig config = compute.recompile_config;
    if (replay_device && replay_device->valid()) {
        const bool adoptable = replay_device->compute_queue_supported &&
            replay_device->storage_image_read_without_format &&
            replay_device->storage_image_write_without_format;
        config.native_storage_format_support = adoptable
            ? replay_device->native_storage_format_support : 0;
        const bool native_multiwave = force_native_multiwave ||
            gpu::compute_shader_prefers_native_multiwave(raw.words.data(), raw.words.size());
        config.native_subgroup_size = gpu::select_native_compute_subgroup_size(
            *replay_device, config, native_multiwave, disable_native_subgroup);
    }
    config.packed_r11_storage = packed_r11_storage;
    std::vector<uint32_t> spirv = gpu::recompile_compute_shader_cached(
        raw.words.data(), raw.words.size(), compute.resources.get(), config);
    if (spirv.empty()) return false;

    compute.spirv = std::move(spirv);
    compute.user_sgprs = config.user_sgprs;
    compute.required_subgroup_size = config.native_subgroup_size;
    return true;
}

} // namespace prosper::tools
