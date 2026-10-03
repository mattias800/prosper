// Shipping same-TU companion. Cold original-code/profile pipelines are shared by transactions;
// a changing wave count, entry word or observed pixel never creates a pipeline or shader module.
#pragma once

class FragmentDrawComputeGpuProgram {
public:
    enum Stage : uint32_t { Count, Assemble, OriginalPs, Validate, Stages };
    static std::shared_ptr<const FragmentDrawComputeGpuProgram>
    acquire(const RenderVkCtx& context,
            std::shared_ptr<const prosper::gpu::FragmentDrawProgramPlan> program,
            std::string& refusal) {
        refusal.clear();
        if (!context.ok || !context.queue_supports_compute || !context.shader_int64_enabled ||
            context.max_compute_workgroup_size_x < 64 ||
            context.max_compute_workgroup_invocations < 64 ||
            context.detile_limits.maxBoundDescriptorSets < 1 ||
            context.detile_limits.maxPerStageDescriptorStorageBuffers < 5 ||
            context.detile_limits.maxDescriptorSetStorageBuffers < 5 ||
            context.detile_limits.maxPerStageResources < 5 || !program ||
            !program->rejection_reason().empty() || !program->capacity_owner() ||
            program->device_contract().device_identity !=
                reinterpret_cast<uintptr_t>(context.dev)) {
            refusal = "fragment-draw-compute-device-profile-unavailable";
            return {};
        }
        using Key = std::pair<uintptr_t, const prosper::gpu::FragmentDrawProgramPlan*>;
        static thread_local std::map<Key, std::shared_ptr<const FragmentDrawComputeGpuProgram>>
            cache;
        const Key key{reinterpret_cast<uintptr_t>(context.dev), program.get()};
        if (const auto found = cache.find(key); found != cache.end()) return found->second;
        auto result = std::shared_ptr<FragmentDrawComputeGpuProgram>(
            new FragmentDrawComputeGpuProgram(context, std::move(program)));
        std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
        for (uint32_t binding = 0; binding < bindings.size(); ++binding)
            bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                 VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo descriptors{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descriptors.bindingCount = uint32_t(bindings.size());
        descriptors.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(context.dev, &descriptors, nullptr,
                                        &result->descriptors_) != VK_SUCCESS) {
            refusal = "fragment-draw-compute-descriptor-layout-failed";
            return {};
        }
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &result->descriptors_;
        if (vkCreatePipelineLayout(context.dev, &layout, nullptr, &result->layout_) != VK_SUCCESS) {
            refusal = "fragment-draw-compute-pipeline-layout-failed";
            return {};
        }
        const auto& input = *result->program_;
        const std::array<const std::vector<uint32_t>*, Stages> sources{
            &input.count_words(), &input.assembly_words(),
            &input.capacity_owner()->kernel()->program.packet.spirv, &input.validation_words()};
        for (uint32_t stage = 0; stage < Stages; ++stage) {
            if (sources[stage]->empty()) {
                refusal = "fragment-draw-compute-original-source-unavailable";
                return {};
            }
            VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            module.codeSize = sources[stage]->size() * sizeof(uint32_t);
            module.pCode = sources[stage]->data();
            VkShaderModule shader = VK_NULL_HANDLE;
            if (create_render_shader_module_checked(context.dev, module, &shader) != VK_SUCCESS) {
                refusal = "fragment-draw-compute-device-module-refused";
                return {};
            }
            VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline.layout = result->layout_;
            pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipeline.stage.module = shader;
            pipeline.stage.pName = "main";
            VkResult created;
            {
                const std::lock_guard driver_guard(graphics_driver_cache_mutex());
                created = vkCreateComputePipelines(context.dev, context.driver_pipeline_cache, 1,
                                                   &pipeline, nullptr, &result->pipelines_[stage]);
            }
            vkDestroyShaderModule(context.dev, shader, nullptr);
            if (created != VK_SUCCESS || !result->pipelines_[stage]) {
                refusal = "fragment-draw-compute-pipeline-failed";
                return {};
            }
        }
        // Source-plan and completion owners prevent pointer reuse/cold-cache eviction from
        // destroying a live pipeline. A finite cache does not turn per-wave data into keys.
        if (cache.size() >= 16) cache.clear();
        cache.emplace(key, result);
        return result;
    }
    ~FragmentDrawComputeGpuProgram() {
        for (auto value : pipelines_)
            if (value) vkDestroyPipeline(context_->dev, value, nullptr);
        if (layout_) vkDestroyPipelineLayout(context_->dev, layout_, nullptr);
        if (descriptors_) vkDestroyDescriptorSetLayout(context_->dev, descriptors_, nullptr);
    }
    VkDescriptorSetLayout descriptors() const { return descriptors_; }
    VkPipelineLayout layout() const { return layout_; }
    VkPipeline pipeline(Stage stage) const { return pipelines_[stage]; }

private:
    FragmentDrawComputeGpuProgram(
        const RenderVkCtx& context,
        std::shared_ptr<const prosper::gpu::FragmentDrawProgramPlan> program)
        : context_(&context), program_(std::move(program)) {}
    const RenderVkCtx* const context_;
    const std::shared_ptr<const prosper::gpu::FragmentDrawProgramPlan> program_;
    VkDescriptorSetLayout descriptors_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    std::array<VkPipeline, Stages> pipelines_{};
};

// Allocations come from the normal backend call's one descriptor pool. Caller budgets these
// four sets/twenty descriptors before constructing that pool and retains it through completion.
inline bool allocate_fragment_draw_compute_sets(
    VkDevice device, VkDescriptorPool pool, const FragmentDrawComputeGpuProgram& program,
    const FragmentDrawGpuOwner& owner,
    std::array<VkDescriptorSet, FragmentDrawComputeGpuProgram::Stages>& sets) {
    sets.fill(VK_NULL_HANDLE);
    std::array<VkDescriptorSetLayout, FragmentDrawComputeGpuProgram::Stages> layouts{};
    layouts.fill(program.descriptors());
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool;
    allocate.descriptorSetCount = uint32_t(layouts.size());
    allocate.pSetLayouts = layouts.data();
    if (!device || !pool || vkAllocateDescriptorSets(device, &allocate, sets.data()) != VK_SUCCESS)
        return false;
    using O = FragmentDrawGpuOwner;
    const auto collector = owner.plane(O::Collector), input = owner.plane(O::Input),
               output = owner.plane(O::Output), commit = owner.plane(O::Commit),
               authority = owner.upload(0), entry = owner.upload(1);
    const std::array<std::array<VkDescriptorBufferInfo, 5>, 4> bindings{{
        {collector, input, authority, entry, commit},
        {collector, input, authority, entry, commit},
        {input, output, authority, entry, commit},
        {input, commit, authority, output, collector},
    }};
    std::array<VkWriteDescriptorSet, 20> writes{};
    for (uint32_t stage = 0; stage < bindings.size(); ++stage)
        for (uint32_t binding = 0; binding < bindings[stage].size(); ++binding) {
            auto& write = writes[stage * 5 + binding];
            write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = sets[stage];
            write.dstBinding = binding;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &bindings[stage][binding];
        }
    vkUpdateDescriptorSets(device, uint32_t(writes.size()), writes.data(), 0, nullptr);
    return true;
}

// Collector writes precede this recorder. Every phase uses independent retained private planes;
// no readback/CPU count/per-wave compile/wait occurs. Validation gate publication is followed by
// a compute→fragment barrier before normal ordered graphics replay reads immutable outputs.
inline void record_fragment_draw_compute_transaction(
    VkCommandBuffer command, const FragmentDrawComputeGpuProgram& program,
    const FragmentDrawGpuOwner& owner,
    const std::array<VkDescriptorSet, FragmentDrawComputeGpuProgram::Stages>& sets) {
    using O = FragmentDrawGpuOwner;
    const auto barrier = [&](O::Plane plane, VkPipelineStageFlags source_stage,
                             VkAccessFlags source, VkPipelineStageFlags destination_stage,
                             VkAccessFlags destination) {
        const auto info = owner.plane(plane);
        VkBufferMemoryBarrier visibility{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        visibility.srcAccessMask = source;
        visibility.dstAccessMask = destination;
        visibility.srcQueueFamilyIndex = visibility.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        visibility.buffer = info.buffer;
        visibility.size = info.range;
        vkCmdPipelineBarrier(command, source_stage, destination_stage, 0, 0, nullptr, 1,
                             &visibility, 0, nullptr);
    };
    const auto dispatch = [&](FragmentDrawComputeGpuProgram::Stage stage, bool indirect) {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, program.pipeline(stage));
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, program.layout(), 0, 1,
                                &sets[stage], 0, nullptr);
        if (indirect)
            vkCmdDispatchIndirect(command, owner.plane(O::Input).buffer,
                                  prosper::gpu::kFragmentDrawIndirectWord * sizeof(uint32_t));
        else
            vkCmdDispatch(command, 1, 1, 1);
    };
    barrier(O::Collector, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    dispatch(FragmentDrawComputeGpuProgram::Count, false);
    barrier(O::Input, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    dispatch(FragmentDrawComputeGpuProgram::Assemble, true);
    barrier(O::Input, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    dispatch(FragmentDrawComputeGpuProgram::OriginalPs, true);
    barrier(O::Output, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    dispatch(FragmentDrawComputeGpuProgram::Validate, false);
    for (O::Plane plane : {O::Input, O::Output, O::Commit, O::Collector})
        barrier(plane, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
}
