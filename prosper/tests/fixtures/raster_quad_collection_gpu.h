// Included by render_runner.h inside prosper::test after the shared Vulkan/ownership facilities.
// This input-only pass does not bind a guest color/depth image, clear an attachment, or execute PS.
#pragma once

inline bool raster_quad_raster_state_overridden() {
    // These are presence switches in the selected normal renderer, including the value "0".
    // Refuse rather than collect raw guest state while the normal path executes a different state.
    return PROSPER_ENV_ON("PROSPER_NO_CULL") || PROSPER_ENV_ON("PROSPER_FLIP_FRONT_FACE") ||
           PROSPER_ENV_ON("PROSPER_NO_DEPTH_BIAS") || PROSPER_ENV_ON("PROSPER_IGNORE_EMPTY_SCISSOR");
}

inline const char* raster_quad_descriptor_limits(const VkPhysicalDeviceLimits& limits,
                                                uint64_t vertex_storage_buffers) {
    // Set0 is the exact read-only VS buffer inventory; Set1 has one fragment-only collector SSBO.
    // Geometry is descriptor-free. Validate API limits BEFORE creating either layout or pipeline.
    if (limits.maxBoundDescriptorSets < 2) return "quad-collector-descriptor-set-limit";
    if (limits.maxPerStageDescriptorStorageBuffers < 1 ||
        vertex_storage_buffers > limits.maxPerStageDescriptorStorageBuffers)
        return "quad-collector-per-stage-storage-limit";
    if (limits.maxPerStageResources < 1 || vertex_storage_buffers > limits.maxPerStageResources)
        return "quad-collector-per-stage-resource-limit";
    if (limits.maxDescriptorSetStorageBuffers < 1 ||
        vertex_storage_buffers > uint64_t(limits.maxDescriptorSetStorageBuffers) - 1)
        return "quad-collector-aggregate-storage-limit";
    return nullptr;
}

inline bool raster_quad_pre_raster_readonly(const std::vector<uint32_t>& words) {
    using namespace prosper::gpu;
    const auto report = validate_spirv_descriptor_interface(words, nullptr, 0, SpirvShaderStage::Unknown, false);
    if (!spirv_descriptor_reflection_complete(report) || !report.storage_buffer_writes_complete ||
        !report.storage_image_writes_complete) return false;
    for (const auto& binding : report.descriptors)
        if (binding.writable || binding.atomic_access || binding.kind != SpirvDescriptorKind::StorageBuffer)
            return false;
    std::map<uint32_t, uint32_t> pointers, builtins;
    std::map<uint32_t, std::vector<uint32_t>> member_builtins;
    std::set<uint32_t> outputs;
    // Capability whitelist excludes clocks, XFB, pipes, interlocks and subgroup-dependent
    // pre-raster programs. Reflection supplies a separate conservative pointer/extended-write
    // proof; absence of descriptor writes alone is not the effect contract.
    for (size_t p = 5; p < words.size();) {
        const uint32_t n = words[p] >> 16, op = words[p] & 0xffff;
        if (!n || p + n > words.size()) return false;
        if (op == 17 && (n != 2 || (words[p + 1] != 1 && words[p + 1] != 2 &&
            words[p + 1] != 4466 && words[p + 1] != 6029))) return false;
        if ((op == 16 && n >= 3 && words[p + 2] == 11) ||
            (op >= 227 && op <= 242) || op == 259 || op >= 4096) return false;
        if (op == 32 && n == 4 && words[p + 2] == 3) pointers[words[p + 1]] = words[p + 3];
        if (op == 59 && n >= 4 && words[p + 3] == 3) {
            const auto pointer = pointers.find(words[p + 1]);
            if (pointer == pointers.end()) return false;
            outputs.insert(words[p + 2]); outputs.insert(pointer->second);
        }
        if (op == 71 && n == 4 && words[p + 2] == 11) builtins[words[p + 1]] = words[p + 3];
        if (op == 72 && n == 5 && words[p + 3] == 11)
            member_builtins[words[p + 1]].push_back(words[p + 4]);
        p += n;
    }
    for (uint32_t output : outputs) {
        if (const auto builtin = builtins.find(output); builtin != builtins.end() &&
            builtin->second != 0 && builtin->second != 7) return false;
        if (const auto members = member_builtins.find(output); members != member_builtins.end())
            for (uint32_t builtin : members->second) if (builtin != 0) return false;
    }
    return true;
}

inline bool raster_quad_varying_interface(const std::vector<uint32_t>& producer,
        const prosper::gpu::RasterQuadInputs& inputs, const prosper::gpu::RasterQuadCollector& contract) {
    // Check the Vulkan interface ABI plus NECESSARY whole-vector writer evidence. The native
    // compiler intentionally declares consumed-but-unwritten outputs (#3416), so a declaration
    // alone must not authorize collecting a definitely unwritten value. A direct OpStore somewhere
    // still DOES NOT prove it executes on every path or establish initialized guest registers.
    std::set<uint32_t> f32, v4f;
    std::map<uint32_t, uint32_t> pointers, variables, locations;
    std::set<uint32_t> nondefault_components, whole_vector_writers;
    for (size_t p = 5; p < producer.size();) {
        const uint32_t n = producer[p] >> 16, op = producer[p] & 0xffff;
        if (!n || p + n > producer.size()) return false;
        if (op == 22 && n == 3 && producer[p + 2] == 32) f32.insert(producer[p + 1]);
        if (op == 23 && n == 4 && f32.count(producer[p + 2]) && producer[p + 3] == 4)
            v4f.insert(producer[p + 1]);
        if (op == 32 && n == 4 && producer[p + 2] == 3) pointers[producer[p + 1]] = producer[p + 3];
        if (op == 59 && n >= 4 && producer[p + 3] == 3) variables[producer[p + 2]] = producer[p + 1];
        if (op == 71 && n == 4 && producer[p + 2] == 30) {
            if (!locations.emplace(producer[p + 1], producer[p + 3]).second) return false;
        }
        if (op == 71 && n >= 3 && (producer[p + 2] == 31 || producer[p + 2] == 32))
            nondefault_components.insert(producer[p + 1]);
        // This first producer supports the existing emitted whole-vec4 stores. Partial stores,
        // access-chain/copy writers and unknown output provenance conservatively decline.
        if (op == 62 && n == 3) whole_vector_writers.insert(producer[p + 1]);
        p += n;
    }
    for (const auto& field : contract.fields) {
        using K = prosper::gpu::RasterQuadFieldKind;
        const uint32_t required = field.kind == K::Interpolant ? field.index : field.kind == K::Parameter
            ? inputs.interpolation.parameter_locations[field.index][field.selector]
            : inputs.interpolation.system_locations[field.index];
        uint32_t matches = 0;
        for (const auto& [variable, location] : locations) {
            if (location != required || !variables.count(variable)) continue;
            const auto pointer = pointers.find(variables.at(variable));
            if (pointer == pointers.end() || !v4f.count(pointer->second) || nondefault_components.count(variable) ||
                !whole_vector_writers.count(variable))
                return false;
            ++matches;
        }
        if (matches != 1) return false;
    }
    return true;
}

inline bool raster_quad_producing_modules_match(const BackendDraw& draw) {
    const auto& owner = draw.raster_quads;
    return !draw.raster_quad_contract_modified && owner && owner->inputs &&
        owner->inputs->source_vs && owner->inputs->source_gs && owner->inputs->source_fs &&
        *owner->inputs->source_vs == draw.vs_words() &&
        *owner->inputs->source_gs == draw.gs_words() &&
        *owner->inputs->source_fs == draw.fs_words();
}

struct RasterQuadGpuOwner {
    const RenderVkCtx* ctx = nullptr;
    std::vector<RenderHostBuffer> buffers;
    std::vector<VkShaderModule> modules;
    std::array<VkDescriptorSetLayout, 2> layouts{};
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkRenderPass pass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    RenderCommandPoolLease commands{};
    ~RasterQuadGpuOwner() {
        if (!ctx) return;
        if (pipeline) vkDestroyPipeline(ctx->dev, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(ctx->dev, layout, nullptr);
        if (framebuffer) vkDestroyFramebuffer(ctx->dev, framebuffer, nullptr);
        if (pass) vkDestroyRenderPass(ctx->dev, pass, nullptr);
        if (descriptors) vkDestroyDescriptorPool(ctx->dev, descriptors, nullptr);
        for (auto value : layouts) if (value) vkDestroyDescriptorSetLayout(ctx->dev, value, nullptr);
        for (auto value : modules) vkDestroyShaderModule(ctx->dev, value, nullptr);
        if (commands.pool) release_render_command_pool(ctx->dev, ctx->qfi, commands);
        for (auto& buffer : buffers) release_render_host_buffer(ctx->dev, buffer);
    }
};

// Caller holds BackendPersistentResourceGuard. Nothing escapes except immutable CPU snapshots and
// the batch's completion lease. Pending/lost submissions retain their GPU owners through the
// existing abandoned-cleanup mechanism instead of destroying resources the device might own.
inline void collect_backend_raster_quads(const RenderVkCtx& ctx, const BackendDraw& draw,
        uint32_t width, uint32_t height, BackendSubmissionBatch& prior) {
    using namespace prosper::gpu;
    const auto sink = draw.raster_quads;
    if (!sink) return;
    std::lock_guard result_lock(sink->mutex);
    RasterQuadResult result;
    result.attempted = true; result.source_submit = draw.source_submit;
    result.draw_index = draw.draw_index; result.command_order = draw.command_order;
    result.width = width; result.height = height;
    if (sink->inputs) result.launch = sink->inputs->launch;
    auto refuse = [&](const char* reason) {
        result.complete = false; result.quads.clear(); result.rejection = reason;
        sink->result = std::move(result);
        // Bounded reports, while each actual request retains its own named result in its owner.
        static std::set<std::pair<uint64_t, std::string>> reported;
        if (reported.size() < 1024 && reported.emplace(draw.fs_identity, reason).second)
            std::fprintf(stderr, "[raster-quad] refused reason=%s draw=%llu; no logical64/raster admission\n",
                         reason, static_cast<unsigned long long>(draw.draw_index));
    };
    if (!ctx.ok || backend_has_unproven_submission()) return refuse("quad-collector-device-unavailable");
    if (draw.raster_quad_contract_modified) return refuse("quad-collector-shader-state-override");
    if (!raster_quad_producing_modules_match(draw))
        return refuse("quad-collector-producing-module-mismatch");
    if (!ctx.fragment_stores_atomics || !ctx.geometry_shader_enabled ||
        !(ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
        !(ctx.subgroup_operations & VK_SUBGROUP_FEATURE_QUAD_BIT) ||
        !(ctx.subgroup_operations & VK_SUBGROUP_FEATURE_BASIC_BIT))
        return refuse("quad-collector-enabled-device-features-unavailable");
    if (!ctx.float_transport.explicit_nonfinite32())
        return refuse("quad-collector-enabled-transport-unavailable");
    if (raster_quad_raster_state_overridden())
        return refuse("quad-collector-raster-state-override");
    if (!draw.ps || draw.mesh_draw || draw.instance_count != 1 ||
        draw.ps->topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || draw.ps->polygon_mode != VK_POLYGON_MODE_FILL)
        return refuse("quad-collector-raster-shape-unimplemented");
    for (const auto& target : draw.ps->color_targets)
        if (target.log2_samples) return refuse("quad-collector-guest-msaa-unimplemented");
    if (!width || !height || width > ctx.detile_limits.maxFramebufferWidth ||
        height > ctx.detile_limits.maxFramebufferHeight)
        return refuse("quad-collector-framebuffer-extent-invalid");
    if (draw.ps->depth_bias_enable && draw.ps->depth_bias_clamp && !ctx.depth_bias_clamp_enabled)
        return refuse("quad-collector-depth-bias-clamp-unavailable");
    if (!draw.gs_words().empty() && !sink->inputs->generated_interpolation_geometry)
        return refuse("quad-collector-primitive-id-producer-unavailable");
    if (!raster_quad_pre_raster_readonly(draw.vs_words()) ||
        (!draw.gs_words().empty() && !raster_quad_pre_raster_readonly(draw.gs_words())))
        return refuse("quad-collector-pre-raster-effects-unproved");
    RasterQuadCollector collector;
    const auto collector_words = build_raster_quad_collector(*sink->inputs, sink->max_quads, collector);
    if (!collector.rejection.empty()) return refuse(collector.rejection.c_str());
    const VkDeviceSize output_bytes = (uint64_t(kRasterQuadBufferHeaderWords) +
        uint64_t(collector.record_words) * collector.max_quads) * 4;
    if (output_bytes > ctx.detile_limits.maxStorageBufferRange)
        return refuse("quad-collector-device-buffer-budget");
    result.fields = collector.fields; result.lane_words = collector.lane_words;
    result.vertex_source = std::make_shared<const std::vector<uint32_t>>(draw.vs_words());
    result.collector_source = std::make_shared<const std::vector<uint32_t>>(collector_words);
    std::vector<uint32_t> geometry;
    if (!draw.gs_words().empty()) {
        // The optional ID publisher may only replace this exact project-generated module, not an
        // arbitrary GS which merely carries a guessed boolean from a caller.
        const auto original = recompile_interpolation_geometry(sink->inputs->interpolation,
            false, false, sink->inputs->float_transport, false);
        if (original != draw.gs_words()) return refuse("quad-collector-geometry-producing-module-mismatch");
        // Generated GS reads exactly these VS attribute locations. Interface presence is only ABI
        // compatibility, NOT MUST-defined-per-invocation or guest initialization authority.
        RasterQuadCollector vs_inputs;
        for (uint32_t attr = 0; attr < 32; ++attr)
            if (sink->inputs->interpolation.attribute_mask & (1u << attr))
                vs_inputs.fields.push_back({RasterQuadFieldKind::Interpolant, attr, 0, 4});
        if (!raster_quad_varying_interface(draw.vs_words(), *sink->inputs, vs_inputs))
            return refuse("quad-collector-geometry-input-interface-unavailable");
        geometry = recompile_interpolation_geometry(sink->inputs->interpolation, false, false,
                                                    sink->inputs->float_transport, true);
        if (geometry.empty()) return refuse("quad-collector-primitive-id-emission-refused");
        result.geometry_source = std::make_shared<const std::vector<uint32_t>>(geometry);
    }
    if (!raster_quad_varying_interface(geometry.empty() ? draw.vs_words() : geometry, *sink->inputs, collector))
        return refuse("quad-collector-varying-interface-unavailable");
    const auto reflected = validate_spirv_descriptor_interface(draw.vs_words(), nullptr, 0,
                                                               SpirvShaderStage::Vertex, false);
    uint64_t vertex_bytes = 0;
    if (reflected.descriptors.size() > 32)
        return refuse("quad-collector-vertex-buffer-budget");
    if (const char* limit = raster_quad_descriptor_limits(ctx.detile_limits, reflected.descriptors.size()))
        return refuse(limit);
    for (const auto& binding : reflected.descriptors) {
        if (binding.set != 0 || binding.descriptor_count != 1)
            return refuse("quad-collector-vertex-descriptor-layout-unimplemented");
        const FrameBufferResource* source = nullptr;
        for (const auto& value : draw.B) if (value.set == 0 && value.binding == binding.binding) {
            if (source) return refuse("quad-collector-duplicate-vertex-binding"); source = &value;
        }
        for (const auto& value : draw.R) if (value.set == 0 && value.binding == binding.binding) {
            if (source) return refuse("quad-collector-duplicate-vertex-binding"); source = &value;
        }
        if (!source || source->is_internal_gds || !source->table_entries.empty() ||
            !source->buffer_words_data() || !source->buffer_word_count() ||
            source->buffer_word_count() > ctx.detile_limits.maxStorageBufferRange / 4 ||
            binding.required_bytes > source->buffer_word_count() * uint64_t(4))
            return refuse("quad-collector-vertex-buffer-source-unavailable");
        const uint64_t bytes = source->buffer_word_count() * uint64_t(4);
        if (bytes > (8u << 20) - vertex_bytes)
            return refuse("quad-collector-vertex-buffer-budget");
        vertex_bytes += bytes;
        result.vertex_buffers.push_back({binding.binding,
            {source->buffer_words_data(), source->buffer_words_data() + source->buffer_word_count()}});
    }
    // Preserve ordering against already-recorded producer work before a synchronous collection.
    if (prior.pending()) {
        const auto completion = prior.submit_and_wait(ctx.dev, ctx.queue, false);
        if (completion.submit_result != VK_SUCCESS || completion.wait_result != VK_SUCCESS ||
            backend_has_unproven_submission()) return refuse("quad-collector-prior-completion-unproved");
    }
    auto owner = std::make_shared<RasterQuadGpuOwner>(); owner->ctx = &ctx;
    owner->buffers.push_back(acquire_render_host_buffer(ctx, output_bytes));
    if (!owner->buffers[0].mapped) return refuse("quad-collector-buffer-allocation-failed");
    auto* output = static_cast<uint32_t*>(owner->buffers[0].mapped);
    std::fill(output, output + output_bytes / 4, 0);
    output[2] = collector.record_words; output[3] = kRasterQuadMagic;
    std::array<std::vector<VkDescriptorSetLayoutBinding>, 2> bindings;
    for (const auto& source : result.vertex_buffers) {
        owner->buffers.push_back(acquire_render_host_buffer(ctx, source.words.size() * uint64_t(4)));
        if (!owner->buffers.back().mapped) return refuse("quad-collector-vertex-upload-allocation-failed");
        std::memcpy(owner->buffers.back().mapped, source.words.data(), source.words.size() * 4);
        bindings[0].push_back({source.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr});
    }
    bindings[1].push_back({0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
    for (uint32_t set = 0; set < 2; ++set) {
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = static_cast<uint32_t>(bindings[set].size()); info.pBindings = bindings[set].data();
        if (vkCreateDescriptorSetLayout(ctx.dev, &info, nullptr, &owner->layouts[set]) != VK_SUCCESS)
            return refuse("quad-collector-descriptor-layout-failed");
    }
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 2; layout_info.pSetLayouts = owner->layouts.data();
    if (vkCreatePipelineLayout(ctx.dev, &layout_info, nullptr, &owner->layout) != VK_SUCCESS)
        return refuse("quad-collector-pipeline-layout-failed");
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<uint32_t>(owner->buffers.size())};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 2; pool_info.poolSizeCount = 1; pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(ctx.dev, &pool_info, nullptr, &owner->descriptors) != VK_SUCCESS)
        return refuse("quad-collector-descriptor-pool-failed");
    std::array<VkDescriptorSet, 2> sets{};
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = owner->descriptors; set_info.descriptorSetCount = 2;
    set_info.pSetLayouts = owner->layouts.data();
    if (vkAllocateDescriptorSets(ctx.dev, &set_info, sets.data()) != VK_SUCCESS)
        return refuse("quad-collector-descriptor-allocation-failed");
    std::vector<VkDescriptorBufferInfo> buffer_infos(owner->buffers.size());
    std::vector<VkWriteDescriptorSet> writes(owner->buffers.size());
    for (size_t index = 0; index < owner->buffers.size(); ++index) {
        const uint32_t set = index ? 0 : 1;
        buffer_infos[index] = {owner->buffers[index].buffer, 0,
            index ? result.vertex_buffers[index - 1].words.size() * uint64_t(4) : output_bytes};
        writes[index] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[index].dstSet = sets[set]; writes[index].dstBinding = index ? result.vertex_buffers[index - 1].binding : 0;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[index].descriptorCount = 1;
        writes[index].pBufferInfo = &buffer_infos[index];
    }
    vkUpdateDescriptorSets(ctx.dev, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    const auto stage = [&](const std::vector<uint32_t>& words, VkShaderStageFlagBits type) {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = words.size() * 4; info.pCode = words.data(); VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(ctx.dev, &info, nullptr, &module) != VK_SUCCESS) return false;
        owner->modules.push_back(module);
        VkPipelineShaderStageCreateInfo shader{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        shader.stage = type; shader.module = module; shader.pName = "main"; stages.push_back(shader); return true;
    };
    if (!stage(draw.vs_words(), VK_SHADER_STAGE_VERTEX_BIT) ||
        (!geometry.empty() && !stage(geometry, VK_SHADER_STAGE_GEOMETRY_BIT)) ||
        !stage(collector_words, VK_SHADER_STAGE_FRAGMENT_BIT)) return refuse("quad-collector-shader-module-failed");
    VkSubpassDescription subpass{}; subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    VkRenderPassCreateInfo pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass_info.subpassCount = 1; pass_info.pSubpasses = &subpass;
    if (vkCreateRenderPass(ctx.dev, &pass_info, nullptr, &owner->pass) != VK_SUCCESS)
        return refuse("quad-collector-render-pass-failed");
    VkFramebufferCreateInfo framebuffer_info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer_info.renderPass = owner->pass; framebuffer_info.width = width;
    framebuffer_info.height = height; framebuffer_info.layers = 1;
    if (vkCreateFramebuffer(ctx.dev, &framebuffer_info, nullptr, &owner->framebuffer) != VK_SUCCESS)
        return refuse("quad-collector-framebuffer-failed");
    VkPipelineVertexInputStateCreateInfo vertex{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    const auto& ps = *draw.ps;
    VkViewport viewport{0, 0, float(width), float(height), 0, 1};
    if (ps.has_viewport) viewport = {ps.viewport_x, ps.viewport_y, ps.viewport_w, ps.viewport_h, ps.min_depth, ps.max_depth};
    if (!std::isfinite(viewport.x) || !std::isfinite(viewport.y) || !std::isfinite(viewport.width) ||
        !std::isfinite(viewport.height) || viewport.width <= 0 || viewport.height == 0 ||
        !std::isfinite(viewport.minDepth) || !std::isfinite(viewport.maxDepth) ||
        viewport.minDepth < 0 || viewport.minDepth > 1 || viewport.maxDepth < 0 || viewport.maxDepth > 1 ||
        viewport.width > ctx.detile_limits.maxViewportDimensions[0] ||
        std::abs(viewport.height) > ctx.detile_limits.maxViewportDimensions[1] ||
        viewport.x < ctx.detile_limits.viewportBoundsRange[0] ||
        viewport.y < ctx.detile_limits.viewportBoundsRange[0] ||
        viewport.x + viewport.width > ctx.detile_limits.viewportBoundsRange[1] ||
        viewport.y + viewport.height < ctx.detile_limits.viewportBoundsRange[0] ||
        viewport.y + viewport.height > ctx.detile_limits.viewportBoundsRange[1] ||
        viewport.y > ctx.detile_limits.viewportBoundsRange[1])
        return refuse("quad-collector-viewport-unavailable");
    if ((ps.cull_mode & ~VK_CULL_MODE_FRONT_AND_BACK) || ps.front_face > VK_FRONT_FACE_CLOCKWISE ||
        !std::isfinite(ps.depth_bias_constant) || !std::isfinite(ps.depth_bias_slope) ||
        !std::isfinite(ps.depth_bias_clamp)) return refuse("quad-collector-raster-state-unavailable");
    VkRect2D scissor{{0, 0}, {width, height}};
    if (ps.has_scissor) {
        const int64_t left = std::clamp<int64_t>(ps.scissor_left, 0, width);
        const int64_t top = std::clamp<int64_t>(ps.scissor_top, 0, height);
        const int64_t right = std::clamp<int64_t>(ps.scissor_right, left, width);
        const int64_t bottom = std::clamp<int64_t>(ps.scissor_bottom, top, height);
        scissor = {{int32_t(left), int32_t(top)}, {uint32_t(right - left), uint32_t(bottom - top)}};
    }
    VkPipelineViewportStateCreateInfo viewport_state{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport_state.viewportCount = 1; viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1; viewport_state.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL; raster.cullMode = ps.cull_mode;
    raster.frontFace = static_cast<VkFrontFace>(ps.front_face); raster.lineWidth = 1;
    raster.depthBiasEnable = ps.depth_bias_enable != 0; raster.depthBiasConstantFactor = ps.depth_bias_constant;
    raster.depthBiasSlopeFactor = ps.depth_bias_slope; raster.depthBiasClamp = ps.depth_bias_clamp;
    VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline_info.stageCount = static_cast<uint32_t>(stages.size()); pipeline_info.pStages = stages.data();
    pipeline_info.pVertexInputState = &vertex; pipeline_info.pInputAssemblyState = &assembly;
    pipeline_info.pViewportState = &viewport_state; pipeline_info.pRasterizationState = &raster;
    pipeline_info.pMultisampleState = &samples; pipeline_info.pColorBlendState = &blend;
    pipeline_info.pDepthStencilState = &depth; pipeline_info.layout = owner->layout; pipeline_info.renderPass = owner->pass;
    if (vkCreateGraphicsPipelines(ctx.dev, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &owner->pipeline) != VK_SUCCESS)
        return refuse("quad-collector-pipeline-failed");
    const auto indices = draw.index_words();
    if (indices.size() > UINT32_MAX ||
        (indices.empty() && (draw.vertex_offset < 0 || !draw.vcount)) ||
        (!indices.empty() && indices.size_bytes() > ctx.detile_limits.maxStorageBufferRange))
        return refuse("quad-collector-index-domain-unavailable");
    const uint32_t primitive_count = static_cast<uint32_t>((indices.empty() ? draw.vcount : indices.size()) / 3);
    size_t index_buffer = 0;
    if (!indices.empty()) {
        index_buffer = owner->buffers.size();
        owner->buffers.push_back(acquire_render_host_buffer(ctx, indices.size_bytes()));
        if (!owner->buffers.back().mapped) return refuse("quad-collector-index-upload-failed");
        std::memcpy(owner->buffers.back().mapped, indices.data(), indices.size_bytes());
    }
    owner->commands = acquire_render_command_pool(ctx.dev, ctx.qfi);
    if (!owner->commands.command) return refuse("quad-collector-command-pool-failed");
    const auto command = owner->commands.command;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) return refuse("quad-collector-command-begin-failed");
    VkBufferMemoryBarrier initialized{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    initialized.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    initialized.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    initialized.srcQueueFamilyIndex = initialized.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    initialized.buffer = owner->buffers[0].buffer; initialized.size = output_bytes;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 1, &initialized, 0, nullptr);
    VkRenderPassBeginInfo render_begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    render_begin.renderPass = owner->pass; render_begin.framebuffer = owner->framebuffer;
    render_begin.renderArea = {{0, 0}, {width, height}};
    vkCmdBeginRenderPass(command, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, owner->pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, owner->layout, 0, 2, sets.data(), 0, nullptr);
    if (!indices.empty()) {
        vkCmdBindIndexBuffer(command, owner->buffers[index_buffer].buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(command, static_cast<uint32_t>(indices.size()), 1, 0, draw.vertex_offset, 0);
    } else vkCmdDraw(command, draw.vcount, 1, static_cast<uint32_t>(draw.vertex_offset), 0);
    vkCmdEndRenderPass(command);
    record_host_read_barrier(command, owner->buffers[0].buffer,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    if (vkEndCommandBuffer(command) != VK_SUCCESS) return refuse("quad-collector-command-end-failed");
    BackendSubmissionBatch collection;
    collection.enqueue(command); collection.add_cleanup([owner] {});
    const auto completion = collection.submit_and_wait(ctx.dev, ctx.queue, false);
    if (completion.submit_result != VK_SUCCESS || completion.wait_result != VK_SUCCESS || backend_has_unproven_submission())
        return refuse("quad-collector-submission-completion-unproved");
    // acquire_render_host_buffer requires HOST_COHERENT; the separate HOST_READ barrier above is
    // still required. Never weaken that allocator without adding noncoherent invalidation here.
    const auto decode_error = decode_raster_quad_records(collector, output,
        static_cast<size_t>(output_bytes / 4), primitive_count, result.quads);
    if (!decode_error.empty()) return refuse(decode_error.c_str());
    result.complete = true;
    result.host_raster_domain_available = true;
    result.host_sample_count = 1; result.host_sample_index = 0;
    result.host_layer = 0; result.host_view_index = 0;
    result.host_instance_count = 1; result.host_first_instance = 0;
    static uint32_t completion_reports = 0;
    if (completion_reports < 32) {
        ++completion_reports;
        std::fprintf(stderr, "[raster-quad] completed host-input collection draw=%llu quads=%zu; coverage is pre-depth/pre-guest-PS, guest64 composition/commit unavailable\n",
                     static_cast<unsigned long long>(draw.draw_index), result.quads.size());
    }
    sink->result = std::move(result);
}
