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
    const gpu::RecompileDiagnosticContext diagnostic{
        gpu::RecompileDiagnosticStage::Compute, compute.code_addr};
    if (replay_device && replay_device->valid()) {
        const bool adoptable = replay_device->compute_queue_supported &&
            replay_device->storage_image_read_without_format &&
            replay_device->storage_image_write_without_format;
        config.native_storage_format_support = adoptable
            ? replay_device->native_storage_format_support : 0;
        config.storage_buffer_int64_atomics = adoptable &&
            replay_device->storage_buffer_int64_atomics;
        const bool native_multiwave = force_native_multiwave ||
            gpu::compute_shader_prefers_native_multiwave(
                raw.words.data(), raw.words.size(), diagnostic);
        config.native_subgroup_size = gpu::select_native_compute_subgroup_size(
            *replay_device, config, native_multiwave, disable_native_subgroup);
    }
    config.packed_r11_storage = packed_r11_storage;
    std::vector<uint32_t> spirv = gpu::recompile_compute_shader_cached(
        raw.words.data(), raw.words.size(), compute.resources.get(), config, nullptr, diagnostic);
    if (spirv.empty()) return false;

    compute.spirv = std::move(spirv);
    compute.user_sgprs = config.user_sgprs;
    compute.required_subgroup_size = config.native_subgroup_size;
    return true;
}

// Retry one failed compute translation without Vulkan. A successful return means the capsule had
// complete, internally consistent retry inputs; an empty SPIR-V vector remains the translator's
// fail-visible rejection. Older captures and incomplete diagnostics return false with a precise
// metadata error instead of silently substituting default launch or user-SGPR state.
inline bool recompile_failed_compute_stage(
    const gpu::GpuCapturedStageDiagnostic& stage,
    const std::vector<gpu::GpuCaptureRawShaderVersion>& raw_shader_versions,
    std::vector<uint32_t>& spirv,
    std::string& error) {
    error.clear();
    spirv.clear();
    if (stage.stage != gpu::ShaderProgramStage::Compute) {
        error = "failed stage is not compute";
        return false;
    }
    if (stage.raw_shader_index >= raw_shader_versions.size()) {
        error = "failed compute stage has no captured raw stream";
        return false;
    }
    if (stage.resource_table_present != stage.resource_table.present ||
        stage.resource_count != stage.resource_table.resources.size()) {
        error = "failed compute stage lacks exact resource metadata (capture predates v35)";
        return false;
    }
    if (!stage.recompile_config_available) {
        error = "failed compute stage lacks exact specialization "
                "(capture predates v42 or diagnostic is incomplete)";
        return false;
    }

    gpu::ShaderResourceTable table;
    table.resources.reserve(stage.resource_table.resources.size());
    for (const auto& captured : stage.resource_table.resources)
        table.resources.push_back(captured.resource);
    const gpu::ShaderResourceTable* resources = stage.resource_table.present ? &table : nullptr;
    const auto& raw = raw_shader_versions[stage.raw_shader_index];
    const gpu::RecompileDiagnosticContext diagnostic{
        gpu::RecompileDiagnosticStage::Compute, stage.program_addr};
    spirv = gpu::recompile_compute_shader_cached(
        raw.words.data(), raw.words.size(), resources, stage.recompile_config, nullptr,
        diagnostic);
    return true;
}

// A retry is diagnostic recompilation, not a bypass around the live descriptor contract. Validate
// the resulting module with the same stage/set mapping used by execution so a capsule cannot report
// "recompiled" while the live backend will immediately reject its resources.
inline gpu::DescriptorValidationReport validate_recompiled_failed_stage_contract(
    gpu::ShaderProgramStage stage,
    const std::vector<uint32_t>& spirv,
    const gpu::ShaderResourceTable* resources) {
    uint32_t set = 0;
    gpu::SpirvShaderStage spirv_stage = gpu::SpirvShaderStage::Unknown;
    switch (stage) {
        case gpu::ShaderProgramStage::Vertex:
            spirv_stage = gpu::SpirvShaderStage::Vertex;
            break;
        case gpu::ShaderProgramStage::Fragment:
            set = 1;
            spirv_stage = gpu::SpirvShaderStage::Fragment;
            break;
        case gpu::ShaderProgramStage::Compute:
            spirv_stage = gpu::SpirvShaderStage::Compute;
            break;
    }
    return gpu::validate_spirv_descriptor_interface(
        spirv, resources, set, spirv_stage, false);
}

} // namespace prosper::tools
