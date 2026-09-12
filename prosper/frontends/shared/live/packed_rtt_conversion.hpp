// Device conversion of an authoritative RGBA8 image into the sampled packed-10-bit layout.
#pragma once
#include "gpu/recompiler/spirv_builder.hpp"
#include "gpu/diagnostics/vk_object_names.hpp"   // #3578
#include "gpu/execute/host_read_barrier.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>

namespace prosper::frontend {
// Owned by the compute context, which destroys it before releasing its device.
struct PackedRttConversion {
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPhysicalDeviceLimits limits{};
    bool attempted = false;
    VkResult setup_result = VK_ERROR_FORMAT_NOT_SUPPORTED;

    void destroy() {
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (descriptors) vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
        pipeline = VK_NULL_HANDLE; layout = VK_NULL_HANDLE; descriptors = VK_NULL_HANDLE;
    }
    VkResult initialize(VkPhysicalDevice physical, VkDevice dev, VkPipelineCache cache) {
        if (attempted) return setup_result;
        attempted = true; device = dev;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical, &properties);
        limits = properties.limits;
        VkFormatProperties format{};
        vkGetPhysicalDeviceFormatProperties(physical, VK_FORMAT_A2B10G10R10_UNORM_PACK32, &format);
        const auto needed = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if ((format.optimalTilingFeatures & needed) != needed ||
            limits.maxComputeWorkGroupSize[0] < 128 || limits.maxComputeWorkGroupInvocations < 128)
            return setup_result;
        const VkDescriptorSetLayoutBinding binding{
            0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 1; dci.pBindings = &binding;
        setup_result = vkCreateDescriptorSetLayout(device, &dci, nullptr, &descriptors);
        if (setup_result != VK_SUCCESS) return setup_result;
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount = 1; lci.pSetLayouts = &descriptors;
        lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &push;
        setup_result = vkCreatePipelineLayout(device, &lci, nullptr, &layout);
        if (setup_result != VK_SUCCESS) {
            destroy(); return setup_result;
        }
        const auto words = prosper::gpu::build_compute_rgba8_to_packed10();
        VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sci.codeSize = words.size() * sizeof(uint32_t); sci.pCode = words.data();
        VkShaderModule shader = VK_NULL_HANDLE;
        setup_result = vkCreateShaderModule(device, &sci, nullptr, &shader);
        if (setup_result == VK_SUCCESS)   // #3578
            prosper::gpu::vk_name_object(device, VK_OBJECT_TYPE_SHADER_MODULE, (uint64_t)shader,
                                         "prosper packed_rtt_conversion");
        if (setup_result != VK_SUCCESS) {
            destroy(); return setup_result;
        }
        VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pci.stage.module = shader; pci.stage.pName = "main"; pci.layout = layout;
        setup_result = vkCreateComputePipelines(device, cache, 1, &pci, nullptr, &pipeline);
        vkDestroyShaderModule(device, shader, nullptr);
        if (setup_result != VK_SUCCESS) destroy();
        return setup_result;
    }
    bool fits(uint64_t texels) const {
        return texels && texels <= UINT32_MAX - 127u &&
            texels * 4 <= limits.maxStorageBufferRange &&
            (texels + 127) / 128 <= limits.maxComputeWorkGroupCount[0];
    }
    VkResult allocate_binding(VkDescriptorPool& pool, VkDescriptorSet& set) const {
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &size;
        VkResult result = vkCreateDescriptorPool(device, &pci, nullptr, &pool);
        if (result != VK_SUCCESS) return result;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &descriptors;
        result = vkAllocateDescriptorSets(device, &ai, &set);
        if (result == VK_SUCCESS) return result;
        vkDestroyDescriptorPool(device, pool, nullptr); pool = VK_NULL_HANDLE;
        return result;
    }
    void bind_buffer(VkDescriptorSet set, VkBuffer buffer, VkDeviceSize bytes) const {
        const VkDescriptorBufferInfo info{buffer, 0, bytes};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set; write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo = &info;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
    void record(VkCommandBuffer command, VkImage source, VkImageLayout saved,
                VkImage destination, VkBuffer scratch, VkDescriptorSet set,
                uint32_t width, uint32_t height) const {
        VkImageMemoryBarrier images[2]{};
        for (auto& b : images) {
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        images[0].image = source; images[0].oldLayout = saved;
        images[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        images[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        images[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        images[1].image = destination; images[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        images[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        images[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        // The allocation may contain a prior dispatch's device writes under another
        // buffer handle. Order those writes before overwriting the recycled storage.
        VkMemoryBarrier recycled{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        recycled.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        recycled.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &recycled, 0, nullptr, 2, images);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(command, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              scratch, 1, &copy);
        VkBufferMemoryBarrier buffer{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        buffer.srcQueueFamilyIndex = buffer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer.buffer = scratch; buffer.size = VK_WHOLE_SIZE;
        buffer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        buffer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &buffer, 0, nullptr);
        const uint32_t count = width * height;
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                                0, 1, &set, 0, nullptr);
        vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count), &count);
        vkCmdDispatch(command, (count + 127u) / 128u, 1, 1);
        buffer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        buffer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &buffer, 0, nullptr);
        vkCmdCopyBufferToImage(command, scratch, destination,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        images[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        images[0].newLayout = saved; images[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        images[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        images[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        images[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        images[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        images[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 2, images);
        // The scratch allocation returns to the host-visible memory pool. Its next
        // owner can map and read it even though this binding never maps it.
        prosper::gpu::record_host_read_barrier(command, scratch,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    }
};
} // namespace prosper::frontend
