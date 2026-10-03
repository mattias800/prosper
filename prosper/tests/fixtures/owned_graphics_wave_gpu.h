// SHIPPING companion of render_runner.h, inside prosper::test. Uses that renderer's actual
// device, queue, allocation pool and completion leases; it creates no independent compute device.
#pragma once

struct OwnedGraphicsWaveGpuOwner {
    const RenderVkCtx* context = nullptr;
    std::vector<RenderHostBuffer> buffers;
    std::vector<VkShaderModule> modules;
    std::vector<VkPipeline> pipelines;
    VkDescriptorSetLayout descriptors_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    RenderCommandPoolLease commands{};
    ~OwnedGraphicsWaveGpuOwner() {
        if (!context) return;
        for (auto pipeline : pipelines) vkDestroyPipeline(context->dev, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(context->dev, layout, nullptr);
        if (descriptors) vkDestroyDescriptorPool(context->dev, descriptors, nullptr);
        if (descriptors_layout)
            vkDestroyDescriptorSetLayout(context->dev, descriptors_layout, nullptr);
        for (auto module : modules) vkDestroyShaderModule(context->dev, module, nullptr);
        if (commands.pool) release_render_command_pool(context->dev, context->qfi, commands);
        for (auto buffer : buffers) release_render_host_buffer(context->dev, buffer);
    }
};

// Test-owned, same-thread observation of the exact completed programs. Shipping never installs
// this callback: no environment/path reads, file I/O, extra dispatch or owner authority live here.
using OwnedGraphicsWaveModuleObserver =
    void (*)(const std::vector<prosper::gpu::FragmentPacketProgram>&);
inline thread_local OwnedGraphicsWaveModuleObserver owned_graphics_wave_module_observer = nullptr;

// Caller owns BackendPersistentResourceGuard and an ordered immutable input plan. Private scratch
// effects are not guest producer publication. All waves complete and validate before the caller
// may bind any export to a guest attachment; an indeterminate wait retains every actual GPU owner.
inline bool execute_owned_graphics_waves(const RenderVkCtx& ctx,
                                         const prosper::gpu::GraphicsWaveStagePlan& plan,
                                         BackendSubmissionBatch& prior,
                                         prosper::gpu::GraphicsWaveOutputTransaction& transaction,
                                         std::string& refusal) {
    using namespace prosper::gpu;
    transaction = {};
    refusal.clear();
    const auto reject = [&](const char* reason) {
        refusal = reason;
        return false;
    };
    const auto& limits = ctx.detile_limits;
    if (!ctx.ok || !ctx.queue_supports_compute || backend_has_unproven_submission() ||
        ctx.max_compute_workgroup_size_x < 64u || ctx.max_compute_workgroup_invocations < 64u ||
        limits.maxBoundDescriptorSets < 1u || limits.maxPerStageDescriptorStorageBuffers < 2u ||
        limits.maxDescriptorSetStorageBuffers < 2u || limits.maxPerStageResources < 2u)
        return reject("graphics-wave-executing-device-unavailable");
    if (plan.packets.empty() || plan.packets.size() > 256u ||
        plan.packets.size() != plan.invocations.size())
        return reject("graphics-wave-executing-plan-incomplete");
    std::vector<FragmentPacketProgram> programs;
    uint64_t total_bytes = 0;
    for (const auto& packet : plan.packets) {
        if (!complete_graphics_packet_locals(packet, refusal, false) ||
            !validate_graphics_raw_wave_windows(packet.guest_code, packet.sgprs, packet.raw_windows,
                                                refusal))
            return false;
        auto program = recompile_fragment_packet(packet);
        if (program.spirv.empty()) {
            refusal = program.rejection;
            return false;
        }
        const uint64_t input_bytes = program.input_words.size() * uint64_t(4);
        const uint64_t output_bytes = program.output_words.size() * uint64_t(4);
        if (!input_bytes || !output_bytes || input_bytes > limits.maxStorageBufferRange ||
            output_bytes > limits.maxStorageBufferRange ||
            input_bytes + output_bytes > (128u << 20u) - total_bytes)
            return reject("graphics-wave-executing-buffer-domain-unavailable");
        total_bytes += input_bytes + output_bytes;
        programs.push_back(std::move(program));
    }
    if (prior.pending()) {
        const auto completed = prior.submit_and_wait(ctx.dev, ctx.queue, false);
        if (completed.submit_result != VK_SUCCESS || completed.wait_result != VK_SUCCESS ||
            backend_has_unproven_submission())
            return reject("graphics-wave-prior-producer-incomplete");
    }
    auto owner = std::make_shared<OwnedGraphicsWaveGpuOwner>();
    owner->context = &ctx;
    const VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo descriptor_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor_info.bindingCount = 2;
    descriptor_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(ctx.dev, &descriptor_info, nullptr,
                                    &owner->descriptors_layout) != VK_SUCCESS)
        return reject("graphics-wave-descriptor-layout-failed");
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &owner->descriptors_layout;
    if (vkCreatePipelineLayout(ctx.dev, &layout_info, nullptr, &owner->layout) != VK_SUCCESS)
        return reject("graphics-wave-pipeline-layout-failed");
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         uint32_t(programs.size() * 2u)};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = uint32_t(programs.size());
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(ctx.dev, &pool_info, nullptr, &owner->descriptors) != VK_SUCCESS)
        return reject("graphics-wave-descriptor-pool-failed");
    std::vector<VkDescriptorSetLayout> layouts(programs.size(), owner->descriptors_layout);
    std::vector<VkDescriptorSet> sets(programs.size());
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = owner->descriptors;
    allocation.descriptorSetCount = uint32_t(sets.size());
    allocation.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(ctx.dev, &allocation, sets.data()) != VK_SUCCESS)
        return reject("graphics-wave-descriptor-allocation-failed");
    for (size_t wave = 0; wave < programs.size(); ++wave) {
        const auto& program = programs[wave];
        VkDescriptorBufferInfo infos[2]{};
        VkWriteDescriptorSet writes[2]{};
        for (uint32_t binding = 0; binding < 2u; ++binding) {
            const auto& words = binding ? program.output_words : program.input_words;
            owner->buffers.push_back(acquire_render_host_buffer(ctx, words.size() * uint64_t(4)));
            auto& buffer = owner->buffers.back();
            if (!buffer.mapped) return reject("graphics-wave-buffer-allocation-failed");
            std::memcpy(buffer.mapped, words.data(), words.size() * sizeof(uint32_t));
            infos[binding] = {buffer.buffer, 0, words.size() * uint64_t(4)};
            writes[binding] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[binding].dstSet = sets[wave];
            writes[binding].dstBinding = binding;
            writes[binding].descriptorCount = 1;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].pBufferInfo = &infos[binding];
        }
        vkUpdateDescriptorSets(ctx.dev, 2, writes, 0, nullptr);
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = program.spirv.size() * sizeof(uint32_t);
        module_info.pCode = program.spirv.data();
        VkShaderModule module = VK_NULL_HANDLE;
        if (create_render_shader_module_checked(ctx.dev, module_info, &module) != VK_SUCCESS)
            return reject("graphics-wave-device-module-refused");
        owner->modules.push_back(module);
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.layout = owner->layout;
        pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(ctx.dev, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                     &pipeline) != VK_SUCCESS) {
            if (pipeline) vkDestroyPipeline(ctx.dev, pipeline, nullptr);
            return reject("graphics-wave-compute-pipeline-failed");
        }
        owner->pipelines.push_back(pipeline);
    }
    owner->commands = acquire_render_command_pool(ctx.dev, ctx.qfi);
    if (!owner->commands.command) return reject("graphics-wave-command-pool-failed");
    const auto command = owner->commands.command;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS)
        return reject("graphics-wave-command-begin-failed");
    for (size_t wave = 0; wave < programs.size(); ++wave) {
        VkBufferMemoryBarrier uploads[2]{};
        for (uint32_t binding = 0; binding < 2u; ++binding) {
            auto& barrier = uploads[binding];
            barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask =
                binding ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = owner->buffers[wave * 2u + binding].buffer;
            barrier.size = VK_WHOLE_SIZE;
        }
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 2, uploads, 0,
                             nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, owner->pipelines[wave]);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, owner->layout, 0, 1,
                                &sets[wave], 0, nullptr);
        vkCmdDispatch(command, 1, 1,
                      1); // one real64-worker rendezvous, independent of host subgroups
        record_host_read_barrier(command, owner->buffers[wave * 2u + 1u].buffer,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    }
    if (vkEndCommandBuffer(command) != VK_SUCCESS)
        return reject("graphics-wave-command-end-failed");
    BackendSubmissionBatch execution;
    execution.enqueue(command);
    execution.add_cleanup([owner] {});
    const auto completed = execution.submit_and_wait(ctx.dev, ctx.queue, false);
    if (completed.submit_result != VK_SUCCESS || completed.wait_result != VK_SUCCESS ||
        backend_has_unproven_submission())
        return reject("graphics-wave-submission-completion-unproved");
    // HOST_READ availability was recorded separately. This pool requires HOST_COHERENT; a future
    // allocator change must also add mapped-memory invalidation before these actual byte reads.
    std::vector<std::vector<uint32_t>> records;
    for (size_t wave = 0; wave < programs.size(); ++wave) {
        records.emplace_back(programs[wave].output_words.size());
        std::memcpy(records.back().data(), owner->buffers[wave * 2u + 1u].mapped,
                    records.back().size() * sizeof(uint32_t));
    }
    if (!validate_graphics_wave_outputs(plan, records, transaction, refusal)) return false;
    if (owned_graphics_wave_module_observer) owned_graphics_wave_module_observer(programs);
    return true;
}

// Replace only the selected stage's binding bank. The new module reads completed immutable
// exports, never a nominal guest address or a warm shader-cache entry. Other-stage bindings keep
// their original owners. Binding order has no authority beyond the already unique contract.
inline void bind_owned_graphics_exports(BackendDraw& draw, uint32_t set, uint32_t binding,
                                        std::vector<uint32_t> words) {
    std::erase_if(draw.R, [set](const auto& resource) { return resource.set == set; });
    std::erase_if(draw.B, [set](const auto& resource) { return resource.set == set; });
    FrameBufferResource resource;
    resource.set = set;
    resource.binding = binding;
    resource.dwords = std::move(words);
    draw.B.push_back(std::move(resource));
    draw.resource_order.clear();
    for (uint32_t index = 0; index < draw.R.size(); ++index) draw.resource_order.push_back(index);
    for (uint32_t index = 0; index < draw.B.size(); ++index)
        draw.resource_order.push_back(0x80000000u | index);
}

// A real live consumer of the owned input plans. The scratch transaction and its output
// validation finish BEFORE normal guest colour/depth/blend work is recorded. The ordinary pass
// still owns fixed-function order and publication; collecting inputs is not a guest producer.
inline bool materialize_owned_graphics_draw(const RenderVkCtx& ctx, BackendDraw& draw,
                                            uint32_t width, uint32_t height,
                                            BackendSubmissionBatch& prior, std::string& refusal) {
    using namespace prosper::gpu;
    refusal.clear();
    const auto owner = draw.owned_waves;
    if (!owner) return true;
    const auto reject = [&](const char* reason) {
        refusal = reason;
        return false;
    };
    if (draw.mesh_draw || draw.raster_quad_contract_modified ||
        (!owner->vertex_pending && !owner->fragment_pending))
        return reject("graphics-wave-live-contract-unavailable");
    if (owner->vertex_pending) {
        GraphicsWaveOutputTransaction completed;
        GraphicsVertexExportCommit commit;
        if (!execute_owned_graphics_waves(ctx, owner->vertex, prior, completed, refusal) ||
            !prepare_owned_vertex_export_commit(
                owner->vertex, completed, owner->has_pixel_inputs ? &owner->pixel_inputs : nullptr,
                commit, refusal))
            return false;
        draw.set_vs(std::move(commit.shader));
        bind_owned_graphics_exports(draw, 0u, 0u, std::move(commit.words));
        // This native pass consumes occurrence-ordered completed exports. Guest indexed identity
        // was already supplied to the full64 execution; do not apply the index/base twice.
        draw.indices.clear();
        draw.borrowed_indices = {};
        draw.has_borrowed_indices = false;
        draw.vcount = commit.vertices_per_instance;
        draw.instance_count = commit.instances;
        draw.vertex_offset = 0;
    }
    if (owner->fragment_pending) {
        if (!draw.ps || !owner->fragment_raster_inputs || !owner->fragment_code ||
            owner->fragment_code->empty() || draw.ps->depth_test_enable ||
            draw.ps->depth_write_enable || draw.ps->stencil_enable || draw.ps->depth_clear_enable ||
            draw.ps->stencil_clear_enable)
            return reject("graphics-wave-live-predepth-domain-unavailable");
        auto inputs = std::make_shared<RasterQuadInputs>(*owner->fragment_raster_inputs);
        inputs->source_vs = std::make_shared<const std::vector<uint32_t>>(draw.vs_words());
        inputs->source_gs = std::make_shared<const std::vector<uint32_t>>(draw.gs_words());
        // Only this typed original-code contract admits collection without a producing native PS.
        // It preserves the incoming entry/resource observations without promoting preparation.ready.
        inputs->source_fs = std::make_shared<const std::vector<uint32_t>>();
        inputs->owned_wave_pending = true;
        auto collected = std::make_shared<RasterQuadCollection>();
        collected->inputs = inputs;
        collected->max_quads = 4096u;
        draw.raster_quads = collected;
        collect_backend_raster_quads(ctx, draw, width, height, prior);
        RasterQuadResult raster;
        {
            std::lock_guard lock(collected->mutex);
            raster = collected->result;
        }
        if (!raster.complete) {
            refusal = raster.rejection;
            return false;
        }
        GraphicsWaveStagePlan plan;
        GraphicsWaveOutputTransaction completed;
        GraphicsFragmentExportCommit commit;
        if (!prepare_owned_fragment_waves(*inputs, raster, owner->fragment_scalars,
                                          owner->fragment_windows, owner->fragment_float_mode,
                                          owner->fragment_float_flags, plan, refusal) ||
            !execute_owned_graphics_waves(ctx, plan, prior, completed, refusal) ||
            !prepare_owned_fragment_export_commit(plan, completed, commit, refusal))
            return false;
        draw.set_fs(std::move(commit.shader));
        // Match the commit module's storage identity without occupying internal GDS at1/0.
        bind_owned_graphics_exports(draw, 1u, 1u, std::move(commit.words));
        // The collector is an inspectable immutable input witness, not a second request on the
        // commit shader. Normal rasterization now consumes only authenticated completed rows.
        draw.raster_quads.reset();
    }
    draw.owned_waves.reset();
    return true;
}
