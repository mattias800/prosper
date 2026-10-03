// Same-TU shipping collector. Uses the normal draw's already-uploaded vertex/index leases,
// one set from its shared descriptor pool, and its command buffer. No attachment or CPU wait.
#pragma once

inline const char* fragment_draw_replay_descriptor_limits(const VkPhysicalDeviceLimits& limits,
                                                          uint64_t vertex_bindings) {
    if (limits.maxBoundDescriptorSets < 2) return "fragment-draw-replay-descriptor-set-limit";
    if (vertex_bindings > UINT64_MAX - 5) return "fragment-draw-replay-storage-count-overflow";
    const uint64_t descriptors = vertex_bindings + 5;
    // The normal backend uses the same V|F stage flags for raw storage bindings. Count that
    // actual layout, including all five private replay planes, before any Vulkan allocation.
    if (descriptors > limits.maxPerStageDescriptorStorageBuffers)
        return "fragment-draw-replay-per-stage-storage-limit";
    if (descriptors > limits.maxPerStageResources)
        return "fragment-draw-replay-per-stage-resource-limit";
    if (descriptors > limits.maxDescriptorSetStorageBuffers)
        return "fragment-draw-replay-aggregate-storage-limit";
    return nullptr;
}

class FragmentDrawCollectGpuProgram {
public:
    static std::shared_ptr<const FragmentDrawCollectGpuProgram>
    acquire(const RenderVkCtx& context,
            std::shared_ptr<const prosper::gpu::FragmentDrawProgramPlan> program,
            prosper::gpu::SharedShaderWords vertex, std::vector<uint32_t> bindings, bool depth_bias,
            std::string& refusal) {
        refusal.clear();
        if (!context.ok || !context.geometry_shader_enabled || !context.fragment_stores_atomics ||
            !(context.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
            !(context.subgroup_operations & VK_SUBGROUP_FEATURE_BASIC_BIT) ||
            !(context.subgroup_operations & VK_SUBGROUP_FEATURE_QUAD_BIT) || !program ||
            !program->rejection_reason().empty() || program->collect_words().empty() || !vertex ||
            vertex->empty()) {
            refusal = "fragment-draw-collector-enabled-profile-unavailable";
            return {};
        }
        std::sort(bindings.begin(), bindings.end());
        if (std::adjacent_find(bindings.begin(), bindings.end()) != bindings.end()) {
            refusal = "fragment-draw-collector-duplicate-vertex-binding";
            return {};
        }
        if (const auto reason =
                raster_quad_descriptor_limits(context.detile_limits, bindings.size())) {
            refusal = reason;
            return {};
        }
        if (const auto reason =
                fragment_draw_replay_descriptor_limits(context.detile_limits, bindings.size())) {
            refusal = reason;
            return {};
        }
        using Key = std::tuple<uintptr_t, const prosper::gpu::FragmentDrawProgramPlan*,
                               const std::vector<uint32_t>*, std::vector<uint32_t>, bool>;
        static thread_local std::map<Key, std::shared_ptr<const FragmentDrawCollectGpuProgram>>
            cache;
        const Key key{reinterpret_cast<uintptr_t>(context.dev), program.get(), vertex.get(),
                      bindings, depth_bias};
        if (const auto found = cache.find(key); found != cache.end()) return found->second;
        if (!raster_quad_pre_raster_readonly(*vertex)) {
            refusal = "fragment-draw-collector-pre-raster-effects-unproved";
            return {};
        }
        auto result = std::shared_ptr<FragmentDrawCollectGpuProgram>(
            new FragmentDrawCollectGpuProgram(context, std::move(program), std::move(vertex)));
        const auto reflected = prosper::gpu::validate_spirv_descriptor_interface(
            *result->vertex_, nullptr, 0, prosper::gpu::SpirvShaderStage::Vertex, false);
        if (!prosper::gpu::spirv_descriptor_reflection_complete(reflected)) {
            refusal = "fragment-draw-collector-vertex-interface-unavailable";
            return {};
        }
        for (const auto& binding : reflected.descriptors) {
            if (binding.set != 0 || binding.descriptor_count != 1 ||
                !std::binary_search(bindings.begin(), bindings.end(), binding.binding)) {
                refusal = "fragment-draw-collector-vertex-layout-unimplemented";
                return {};
            }
            result->required_bytes_.emplace_back(binding.binding, binding.required_bytes);
        }
        std::array<std::vector<VkDescriptorSetLayoutBinding>, 2> layouts;
        for (uint32_t binding : bindings)
            layouts[0].push_back({binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                  nullptr});
        layouts[1].push_back({0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
        for (uint32_t set = 0; set < layouts.size(); ++set) {
            VkDescriptorSetLayoutCreateInfo info{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            info.bindingCount = uint32_t(layouts[set].size());
            info.pBindings = layouts[set].data();
            if (vkCreateDescriptorSetLayout(context.dev, &info, nullptr,
                                            &result->descriptors_[set]) != VK_SUCCESS) {
                refusal = "fragment-draw-collector-descriptor-layout-failed";
                return {};
            }
        }
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = uint32_t(result->descriptors_.size());
        layout.pSetLayouts = result->descriptors_.data();
        if (vkCreatePipelineLayout(context.dev, &layout, nullptr, &result->layout_) != VK_SUCCESS) {
            refusal = "fragment-draw-collector-pipeline-layout-failed";
            return {};
        }
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        pass.subpassCount = 1;
        pass.pSubpasses = &subpass;
        if (vkCreateRenderPass(context.dev, &pass, nullptr, &result->pass_) != VK_SUCCESS) {
            refusal = "fragment-draw-collector-render-pass-failed";
            return {};
        }
        std::array<VkShaderModule, 2> modules{};
        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        const std::array<const std::vector<uint32_t>*, 2> sources{
            result->vertex_.get(), &result->program_->collect_words()};
        bool modules_ready = true;
        for (uint32_t index = 0; index < sources.size(); ++index) {
            VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            module.codeSize = sources[index]->size() * 4;
            module.pCode = sources[index]->data();
            if (create_render_shader_module_checked(context.dev, module, &modules[index]) !=
                VK_SUCCESS) {
                modules_ready = false;
                break;
            }
            stages[index] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stages[index].stage = index ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT;
            stages[index].module = modules[index];
            stages[index].pName = "main";
        }
        VkResult created = VK_ERROR_INITIALIZATION_FAILED;
        if (modules_ready) {
            VkPipelineVertexInputStateCreateInfo vertex_input{
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo assembly{
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo viewport{
                VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewport.viewportCount = viewport.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo raster{
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.lineWidth = 1;
            raster.depthBiasEnable = depth_bias;
            VkPipelineMultisampleStateCreateInfo samples{
                VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo depth{
                VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            VkPipelineColorBlendStateCreateInfo blend{
                VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            const VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                          VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_CULL_MODE,
                                          VK_DYNAMIC_STATE_FRONT_FACE};
            VkPipelineDynamicStateCreateInfo dynamic{
                VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dynamic.dynamicStateCount = uint32_t(std::size(states));
            dynamic.pDynamicStates = states;
            VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pipeline.stageCount = uint32_t(stages.size());
            pipeline.pStages = stages.data();
            pipeline.pVertexInputState = &vertex_input;
            pipeline.pInputAssemblyState = &assembly;
            pipeline.pViewportState = &viewport;
            pipeline.pRasterizationState = &raster;
            pipeline.pMultisampleState = &samples;
            pipeline.pDepthStencilState = &depth;
            pipeline.pColorBlendState = &blend;
            pipeline.pDynamicState = &dynamic;
            pipeline.layout = result->layout_;
            pipeline.renderPass = result->pass_;
            const std::lock_guard driver_guard(graphics_driver_cache_mutex());
            created = vkCreateGraphicsPipelines(context.dev, context.driver_pipeline_cache, 1,
                                                &pipeline, nullptr, &result->pipeline_);
        }
        for (auto module : modules)
            if (module) vkDestroyShaderModule(context.dev, module, nullptr);
        if (!modules_ready || created != VK_SUCCESS || !result->pipeline_) {
            refusal = "fragment-draw-collector-pipeline-failed";
            return {};
        }
        // Every cached object owns its exact immutable sources; completion additionally owns it.
        if (cache.size() >= 16) cache.clear();
        cache.emplace(key, result);
        return result;
    }
    ~FragmentDrawCollectGpuProgram() {
        if (pipeline_) vkDestroyPipeline(context_->dev, pipeline_, nullptr);
        if (layout_) vkDestroyPipelineLayout(context_->dev, layout_, nullptr);
        if (pass_) vkDestroyRenderPass(context_->dev, pass_, nullptr);
        for (auto value : descriptors_)
            if (value) vkDestroyDescriptorSetLayout(context_->dev, value, nullptr);
    }
    VkRenderPass pass() const { return pass_; }
    VkPipeline pipeline() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }
    VkDescriptorSetLayout collector_descriptors() const { return descriptors_[1]; }
    const auto& required_bytes() const { return required_bytes_; }

private:
    FragmentDrawCollectGpuProgram(
        const RenderVkCtx& context,
        std::shared_ptr<const prosper::gpu::FragmentDrawProgramPlan> program,
        prosper::gpu::SharedShaderWords vertex)
        : context_(&context), program_(std::move(program)), vertex_(std::move(vertex)) {}
    const RenderVkCtx* const context_;
    const std::shared_ptr<const prosper::gpu::FragmentDrawProgramPlan> program_;
    const prosper::gpu::SharedShaderWords vertex_;
    std::vector<std::pair<uint32_t, uint64_t>> required_bytes_;
    std::array<VkDescriptorSetLayout, 2> descriptors_{};
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkRenderPass pass_ = VK_NULL_HANDLE;
};

class FragmentDrawCollectFramebuffer {
public:
    static std::shared_ptr<const FragmentDrawCollectFramebuffer>
    acquire(const RenderVkCtx& context,
            std::shared_ptr<const FragmentDrawCollectGpuProgram> program, uint32_t width,
            uint32_t height) {
        if (!program || !width || !height || width > context.detile_limits.maxFramebufferWidth ||
            height > context.detile_limits.maxFramebufferHeight)
            return {};
        using Key = std::tuple<uintptr_t, const FragmentDrawCollectGpuProgram*, uint32_t, uint32_t>;
        static thread_local std::map<Key, std::shared_ptr<const FragmentDrawCollectFramebuffer>>
            cache;
        const Key key{reinterpret_cast<uintptr_t>(context.dev), program.get(), width, height};
        if (const auto found = cache.find(key); found != cache.end()) return found->second;
        auto result = std::shared_ptr<FragmentDrawCollectFramebuffer>(
            new FragmentDrawCollectFramebuffer(context.dev, std::move(program)));
        VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebuffer.renderPass = result->program_->pass();
        framebuffer.width = width;
        framebuffer.height = height;
        framebuffer.layers = 1;
        if (vkCreateFramebuffer(context.dev, &framebuffer, nullptr, &result->framebuffer_) !=
            VK_SUCCESS)
            return {};
        if (cache.size() >= 16) cache.clear();
        cache.emplace(key, result);
        return result;
    }
    ~FragmentDrawCollectFramebuffer() {
        if (framebuffer_) vkDestroyFramebuffer(device_, framebuffer_, nullptr);
    }
    VkFramebuffer framebuffer() const { return framebuffer_; }
    const FragmentDrawCollectGpuProgram& program() const { return *program_; }

private:
    FragmentDrawCollectFramebuffer(VkDevice device,
                                   std::shared_ptr<const FragmentDrawCollectGpuProgram> program)
        : device_(device), program_(std::move(program)) {}
    const VkDevice device_;
    const std::shared_ptr<const FragmentDrawCollectGpuProgram> program_;
    VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
};

inline bool allocate_fragment_draw_collect_set(VkDevice device, VkDescriptorPool pool,
                                               const FragmentDrawCollectGpuProgram& program,
                                               const FragmentDrawGpuOwner& owner,
                                               VkDescriptorSet& set) {
    const auto layout = program.collector_descriptors();
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &layout;
    if (vkAllocateDescriptorSets(device, &allocate, &set) != VK_SUCCESS) return false;
    const auto output = owner.plane(FragmentDrawGpuOwner::Collector);
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &output;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    return true;
}

// DV supplies the exact same normal-backend vertex set, index slice, draw parameters and
// resolved dynamic raster state used again by the ordered attachment consumer.
template <class Draw>
inline void record_fragment_draw_collect(VkCommandBuffer command,
                                         const FragmentDrawCollectFramebuffer& target,
                                         VkDescriptorSet collector_set, const Draw& draw,
                                         uint32_t width, uint32_t height) {
    const auto& program = target.program();
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = program.pass();
    begin.framebuffer = target.framebuffer();
    begin.renderArea = {{0, 0}, {width, height}};
    vkCmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, program.pipeline());
    const VkDescriptorSet sets[]{draw.dsets[0], collector_set};
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, program.layout(), 0, 2, sets,
                            0, nullptr);
    vkCmdSetViewport(command, 0, 1, &draw.viewport);
    vkCmdSetScissor(command, 0, 1, &draw.scissor);
    vkCmdSetDepthBias(command, draw.depth_bias_constant, draw.depth_bias_clamp,
                      draw.depth_bias_slope);
    vkCmdSetCullMode(command, draw.cull_mode);
    vkCmdSetFrontFace(command, draw.front_face);
    if (draw.icount) {
        vkCmdBindIndexBuffer(command, draw.ibuf, draw.ioffset, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(command, draw.icount, 1, 0, draw.vertex_offset, 0);
    } else
        vkCmdDraw(command, draw.vcount, 1, uint32_t(draw.vertex_offset), 0);
    vkCmdEndRenderPass(command);
}
