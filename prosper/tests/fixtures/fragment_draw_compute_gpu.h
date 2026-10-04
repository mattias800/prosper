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
        const uint32_t storage_bindings = program && program->requires_scalar_bank() ? 6u : 5u;
        if (!context.ok || !context.queue_supports_compute || !context.shader_int64_enabled ||
            context.max_compute_workgroup_size_x < 64 ||
            context.max_compute_workgroup_invocations < 64 ||
            context.detile_limits.maxBoundDescriptorSets < 1 ||
            context.detile_limits.maxPerStageDescriptorStorageBuffers < storage_bindings ||
            context.detile_limits.maxDescriptorSetStorageBuffers < storage_bindings ||
            context.detile_limits.maxPerStageResources < storage_bindings || !program ||
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
        auto& stats = prosper::gpu::fragment_draw_cache_stats();
        prosper::gpu::retire_dead_fragment_draw_entries(cache, stats.compute_retired);
        ++stats.compute_cold_builds;
        auto result = std::shared_ptr<FragmentDrawComputeGpuProgram>(
            new FragmentDrawComputeGpuProgram(context, std::move(program)));
        std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
        for (uint32_t binding = 0; binding < storage_bindings; ++binding)
            bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                 VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo descriptors{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descriptors.bindingCount = storage_bindings;
        descriptors.pBindings = bindings.data();
        ++stats.vk_object_create_calls;
        if (vkCreateDescriptorSetLayout(context.dev, &descriptors, nullptr,
                                        &result->descriptors_) != VK_SUCCESS) {
            refusal = "fragment-draw-compute-descriptor-layout-failed";
            return {};
        }
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &result->descriptors_;
        ++stats.vk_object_create_calls;
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
            ++stats.checked_shader_module_calls;
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
                ++stats.vk_object_create_calls;
                ++stats.vk_pipeline_create_calls;
                created = vkCreateComputePipelines(context.dev, context.driver_pipeline_cache, 1,
                                                   &pipeline, nullptr, &result->pipelines_[stage]);
            }
            vkDestroyShaderModule(context.dev, shader, nullptr);
            if (created != VK_SUCCESS || !result->pipelines_[stage]) {
                refusal = "fragment-draw-compute-pipeline-failed";
                return {};
            }
        }
        // This strong copied-plan payload owns no original analysis generation. Retirement
        // consults weak source authority; live working sets have no synthetic 16-key limit.
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
    uint32_t storage_bindings() const { return program_->requires_scalar_bank() ? 6u : 5u; }
    bool source_live() const { return program_->source_live(); }

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
// four sets/twenty descriptors (twenty-four with the shared bank) before constructing that pool
// and retains it through completion.
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
               authority = owner.upload(0), entry = owner.upload(1), bank = owner.upload(2);
    const std::array<std::array<VkDescriptorBufferInfo, 6>, 4> bindings{{
        {collector, input, authority, entry, commit, bank},
        {collector, input, authority, entry, commit, bank},
        {input, output, authority, entry, commit, bank},
        {input, commit, authority, output, collector, bank},
    }};
    const uint32_t storage_bindings = program.storage_bindings();
    if (storage_bindings == 6 && (!bank.buffer || !bank.range)) return false;
    std::array<VkWriteDescriptorSet, 24> writes{};
    for (uint32_t stage = 0; stage < bindings.size(); ++stage)
        for (uint32_t binding = 0; binding < storage_bindings; ++binding) {
            auto& write = writes[stage * storage_bindings + binding];
            write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = sets[stage];
            write.dstBinding = binding;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &bindings[stage][binding];
        }
    vkUpdateDescriptorSets(device, uint32_t(bindings.size()) * storage_bindings, writes.data(), 0,
                           nullptr);
    return true;
}

// PRIVATE input-wire calibration, not a guest read/entry authority. Its only mutation is one
// actual scalar word in logical wave-index2 after assembly. It cannot access count/status/header
// words or replace the retained code/capacity/guest bank. Normally null, scoped by test callers.
class FragmentScalarWaveWireCalibration {
public:
    uint32_t wave_index() const { return 2; }
    uint32_t scalar_register() const { return 3; }
    uint64_t target_word() const { return offset_ / sizeof(uint32_t); }
    uint32_t wave_input_base() const { return input_base_; }
    uint32_t scalar_relative_word() const { return scalar_offset_; }
    bool replace_word3(uint32_t value) const {
        if (!input_.buffer || offset_ > input_.range || input_.range - offset_ < 4) return false;
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.buffer = input_.buffer;
        barrier.offset = input_.offset;
        barrier.size = input_.range;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        // Assembly writes and reads of the unchanged indirect header finish before the private
        // transfer. This also orders the retained input plane's read/write hazards explicitly.
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(
            command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        vkCmdUpdateBuffer(command_, input_.buffer, input_.offset + offset_, sizeof(value), &value);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                             0, 0, nullptr, 1, &barrier, 0, nullptr);
        return true;
    }

private:
    FragmentScalarWaveWireCalibration(VkCommandBuffer command, VkDescriptorBufferInfo input,
                                      VkDeviceSize offset, uint32_t input_base,
                                      uint32_t scalar_offset)
        : command_(command), input_(input), offset_(offset), input_base_(input_base),
          scalar_offset_(scalar_offset) {}
    friend void record_fragment_draw_compute_transaction(
        VkCommandBuffer, const FragmentDrawComputeGpuProgram&, const FragmentDrawGpuOwner&,
        const std::array<VkDescriptorSet, FragmentDrawComputeGpuProgram::Stages>&);
    const VkCommandBuffer command_;
    const VkDescriptorBufferInfo input_;
    const VkDeviceSize offset_;
    const uint32_t input_base_, scalar_offset_;
};
using FragmentScalarWaveWireCallback =
    std::function<void(const FragmentScalarWaveWireCalibration&)>;
inline FragmentScalarWaveWireCallback& fragment_scalar_wave_wire_calibration_for_test() {
    static thread_local FragmentScalarWaveWireCallback callback;
    return callback;
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
    // Count also reassembles split quad scopes in place in the collector (#4277), so it writes
    // the plane, and every later stage that reads collector records must see those writes. It
    // borrows the commit plane's pixel slots as a scratch table and zeroes them again, so
    // validation's pixel table must see those writes too.
    barrier(O::Collector, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    dispatch(FragmentDrawComputeGpuProgram::Count, false);
    barrier(O::Collector, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    barrier(O::Commit, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    barrier(O::Input, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    dispatch(FragmentDrawComputeGpuProgram::Assemble, true);
    barrier(O::Input, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    if (const auto& callback = fragment_scalar_wave_wire_calibration_for_test()) {
        const auto& capacity = *owner.transaction().program()->capacity_owner();
        const auto& layout = capacity.kernel()->layout;
        const auto scalar = std::find(layout.sgprs.begin(), layout.sgprs.end(), 3u);
        if (owner.transaction().scalar_bank() && capacity.max_waves() > 2 &&
            scalar != layout.sgprs.end()) {
            const auto index = size_t(scalar - layout.sgprs.begin());
            if (index < layout.scalar_offsets.size()) {
                const uint64_t word =
                    uint64_t(capacity.placement(2).input_base) + 2u + layout.scalar_offsets[index];
                const auto input = owner.plane(O::Input);
                const VkDeviceSize offset = word * sizeof(uint32_t);
                if (offset <= input.range && input.range - offset >= sizeof(uint32_t))
                    callback(FragmentScalarWaveWireCalibration(command, input, offset,
                                                               capacity.placement(2).input_base,
                                                               layout.scalar_offsets[index]));
            }
        }
    }
    dispatch(FragmentDrawComputeGpuProgram::OriginalPs, true);
    barrier(O::Output, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    dispatch(FragmentDrawComputeGpuProgram::Validate, false);
    for (O::Plane plane : {O::Input, O::Output, O::Commit, O::Collector})
        barrier(plane, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
}
