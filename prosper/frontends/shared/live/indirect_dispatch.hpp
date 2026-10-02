// Device-resolved indirect compute dispatch (#3656).
//
// The ordered executor used to read a dispatch's 12-byte workgroup-count triplet on the CPU, copy it
// into direct dimensions and clear `indirect`. For an eligible dispatch it now hands the backend the
// argument ADDRESS instead (ComputeItem::indirect_args_addr), and this file is the backend half:
//
//   1. find a retained GPU buffer that is authoritative for those 12 bytes (a persistent compute
//      buffer, still equal to guest memory by the cache's own journal proof) and pin it;
//   2. copy the triplet into a small device-side scratch record;
//   3. run a one-invocation validation pass that zeroes the launch when any count exceeds ITS AXIS's
//      device limit (vkCmdDispatchIndirect on an out-of-range count is undefined, and a clamped
//      kernel would silently compute part of its domain);
//   4. dispatch with vkCmdDispatchIndirect from the scratch record.
//
// Nothing in steps 1-4 reads the counts on the CPU. A dispatch that cannot get a pin, or a device
// that cannot build the validation pass, falls back to the host reading the triplet and issuing a
// direct dispatch, which is exactly what the executor did before.
#pragma once
#include "gpu/recompiler/spirv_builder.hpp"
#include "gpu/diagnostics/vk_object_names.hpp"
#include "gpu/execute/host_read_barrier.hpp"
#include <vulkan/vulkan.h>
#include <atomic>
#include <cstdint>

namespace prosper::frontend {

// Observable without a log. `device_dispatches` counts vkCmdDispatchIndirect calls recorded;
// `host_fallbacks` counts dispatches that arrived device-resolved but were resolved on the host
// because no authoritative buffer or validation pass was available; `rejected` counts launches the
// validation pass zeroed for exceeding the device limit.
struct IndirectDispatchBackendStats {
    std::atomic<uint64_t> device_dispatches{0};
    std::atomic<uint64_t> host_fallbacks{0};
    std::atomic<uint64_t> rejected{0};
};
inline IndirectDispatchBackendStats& indirect_dispatch_backend_stats() {
    static IndirectDispatchBackendStats stats;
    return stats;
}

// Owned by the compute context, which destroys it before releasing its device.
struct IndirectDispatchValidator {
    // x, y, z, rejected flag, then maxComputeWorkGroupCount[0..2] (host-written once), one spare.
    static constexpr VkDeviceSize kRecordBytes = 8u * sizeof(uint32_t);

    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    bool attempted = false;
    VkResult setup_result = VK_ERROR_INITIALIZATION_FAILED;

    bool ready() const { return pipeline != VK_NULL_HANDLE && set != VK_NULL_HANDLE; }

    void destroy() {
        if (pool) vkDestroyDescriptorPool(device, pool, nullptr);   // frees `set`
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (descriptors) vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
        pool = VK_NULL_HANDLE; set = VK_NULL_HANDLE; pipeline = VK_NULL_HANDLE;
        layout = VK_NULL_HANDLE; descriptors = VK_NULL_HANDLE;
    }

    // Builds the pipeline and binds `scratch` (kRecordBytes, STORAGE_BUFFER usage) once. The scratch
    // buffer is owned by the context and lives as long as this object, so the set is never rewritten.
    VkResult initialize(VkDevice dev, VkPipelineCache cache, VkBuffer scratch) {
        if (attempted) return setup_result;
        attempted = true; device = dev;
        const VkDescriptorSetLayoutBinding binding{
            0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 1; dci.pBindings = &binding;
        setup_result = vkCreateDescriptorSetLayout(device, &dci, nullptr, &descriptors);
        if (setup_result != VK_SUCCESS) { destroy(); return setup_result; }
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount = 1; lci.pSetLayouts = &descriptors;
        lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &push;
        setup_result = vkCreatePipelineLayout(device, &lci, nullptr, &layout);
        if (setup_result != VK_SUCCESS) { destroy(); return setup_result; }
        const auto words = prosper::gpu::build_compute_indirect_dispatch_validate();
        VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sci.codeSize = words.size() * sizeof(uint32_t); sci.pCode = words.data();
        VkShaderModule shader = VK_NULL_HANDLE;
        setup_result = vkCreateShaderModule(device, &sci, nullptr, &shader);
        if (setup_result != VK_SUCCESS) { destroy(); return setup_result; }
        prosper::gpu::vk_name_object(device, VK_OBJECT_TYPE_SHADER_MODULE, (uint64_t)shader,
                                     "prosper indirect_dispatch_validate");
        VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pci.stage.module = shader; pci.stage.pName = "main"; pci.layout = layout;
        setup_result = vkCreateComputePipelines(device, cache, 1, &pci, nullptr, &pipeline);
        vkDestroyShaderModule(device, shader, nullptr);
        if (setup_result != VK_SUCCESS) { destroy(); return setup_result; }
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo poolci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolci.maxSets = 1; poolci.poolSizeCount = 1; poolci.pPoolSizes = &size;
        setup_result = vkCreateDescriptorPool(device, &poolci, nullptr, &pool);
        if (setup_result != VK_SUCCESS) { destroy(); return setup_result; }
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &descriptors;
        setup_result = vkAllocateDescriptorSets(device, &ai, &set);
        if (setup_result != VK_SUCCESS) { destroy(); return setup_result; }
        const VkDescriptorBufferInfo info{scratch, 0, kRecordBytes};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set; write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo = &info;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        return setup_result;
    }

    // Records everything up to, but not including, the vkCmdDispatchIndirect: producer write ->
    // transfer read of the argument range, the copy into `scratch`, transfer write -> validation
    // shader, and validation write -> INDIRECT_COMMAND_READ. The caller binds its own pipeline
    // afterwards (this rebinds the compute pipeline and descriptor set 0) and dispatches indirectly.
    void record(VkCommandBuffer command, VkBuffer arguments, VkDeviceSize argument_offset,
                VkBuffer scratch, uint32_t group_count_limit) const {
        // Every earlier producer of the argument range is a previous submission that already
        // completed, but the dependency is stated rather than assumed: a later change that keeps
        // the producer in the same command buffer (#3948) inherits a correct barrier.
        VkBufferMemoryBarrier produced{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        produced.srcQueueFamilyIndex = produced.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        produced.buffer = arguments; produced.offset = argument_offset;
        produced.size = 3u * sizeof(uint32_t);
        produced.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                 VK_ACCESS_HOST_WRITE_BIT;
        produced.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        // The scratch record is rewritten by every indirect dispatch: order the previous use (a
        // host read of the rejected flag, a prior indirect read) before the overwrite.
        VkBufferMemoryBarrier recycled{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        recycled.srcQueueFamilyIndex = recycled.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        recycled.buffer = scratch; recycled.offset = 0; recycled.size = kRecordBytes;
        recycled.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        recycled.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        const VkBufferMemoryBarrier before[2] = {produced, recycled};
        vkCmdPipelineBarrier(command,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 2, before, 0, nullptr);
        const VkBufferCopy copy{argument_offset, 0, 3u * sizeof(uint32_t)};
        vkCmdCopyBuffer(command, arguments, scratch, 1, &copy);
        VkBufferMemoryBarrier staged{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        staged.srcQueueFamilyIndex = staged.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        staged.buffer = scratch; staged.offset = 0; staged.size = kRecordBytes;
        staged.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        staged.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &staged,
                             0, nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set,
                                0, nullptr);
        // The shader reads its limits from the record; the push block exists only because the
        // shared in-place header declares one, and is pushed so the layout is fully populated.
        vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(group_count_limit), &group_count_limit);
        vkCmdDispatch(command, 1, 1, 1);
        VkBufferMemoryBarrier validated = staged;
        validated.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        validated.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0,
                             0, nullptr, 1, &validated, 0, nullptr);
    }
};

} // namespace prosper::frontend
