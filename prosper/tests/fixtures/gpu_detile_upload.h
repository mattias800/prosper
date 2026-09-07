// Submission-owned tile27 RGBA16F -> RGBA8 upload. Used by the live render backend.
#pragma once
#include "gpu/recompiler/spirv_builder.hpp"
#include "gpu/texture/tile.hpp"
#include "diagnostics/env_cache.hpp"
#include "mapped_staging.h"
#include <vulkan/vulkan.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

namespace prosper::test {

inline std::atomic<uint64_t>& gpu_detile_recordings() {
    static std::atomic<uint64_t> count{0};
    return count;
}

struct GpuDetilePipeline {
    GpuDetilePipeline() = default;
    GpuDetilePipeline(const GpuDetilePipeline&) = delete;
    GpuDetilePipeline& operator=(const GpuDetilePipeline&) = delete;
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    ~GpuDetilePipeline() {
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (descriptors) vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
    }
    bool initialize(VkDevice dev) {
        device = dev;
        const VkDescriptorSetLayoutBinding bindings[]{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 2;
        dci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(dev, &dci, nullptr, &descriptors) != VK_SUCCESS)
            return false;
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &descriptors;
        if (vkCreatePipelineLayout(dev, &lci, nullptr, &layout) != VK_SUCCESS) return false;
        const auto words = prosper::gpu::build_compute_detile_rgba16f();
        VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sci.codeSize = words.size() * sizeof(uint32_t);
        sci.pCode = words.data();
        VkShaderModule shader = VK_NULL_HANDLE;
        if (vkCreateShaderModule(dev, &sci, nullptr, &shader) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pci.stage.module = shader;
        pci.stage.pName = "main";
        pci.layout = layout;
        const VkResult result = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pci,
                                                         nullptr, &pipeline);
        vkDestroyShaderModule(dev, shader, nullptr);
        return result == VK_SUCCESS;
    }
};

struct GpuDetileUpload {
    GpuDetileUpload() = default;
    GpuDetileUpload(const GpuDetileUpload&) = delete;
    GpuDetileUpload& operator=(const GpuDetileUpload&) = delete;
    const GpuDetilePipeline* program = nullptr; // retained with the renderer's device
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer input = VK_NULL_HANDLE, output = VK_NULL_HANDLE;
    VkDeviceMemory input_memory = VK_NULL_HANDLE, output_memory = VK_NULL_HANDLE;
    void* input_mapped = nullptr;
    bool input_reused = false;
    uint64_t input_lease = 0;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptors = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0, faces = 0, count = 0;
    VkDeviceSize source_bytes = 0, output_bytes = 0;
    ~GpuDetileUpload() {
        if (pool) vkDestroyDescriptorPool(device, pool, nullptr);
        // Submission cleanup retains this object until completion is proven. Reuse
        // is safe only here, after the LAST owner releases the immutable snapshot.
        if (!release_mapped_staging(device, input, input_memory, input_mapped, input_lease,
                !PROSPER_ENV_ON("PROSPER_NO_GPU_DETILE_INPUT_REUSE"))) {
            if (input_mapped) vkUnmapMemory(device, input_memory);
            if (input) vkDestroyBuffer(device, input, nullptr);
            if (input_memory) vkFreeMemory(device, input_memory, nullptr);
        }
        if (output) vkDestroyBuffer(device, output, nullptr);
        if (output_memory) vkFreeMemory(device, output_memory, nullptr);
    }
    // Re-recording the same immutable snapshot is allowed across ordered render
    // passes. Order previous copies/dispatches before overwriting its output.
    void record(VkCommandBuffer cmd) const {
        const auto ordinal = gpu_detile_recordings().fetch_add(1, std::memory_order_relaxed) + 1;
        if (PROSPER_ENV_ON("PROSPER_GPU_DETILE_LOG"))
            std::fprintf(stderr, "[gpu-detile] record=%llu extent=%ux%ux%u input=%llu output=%llu\n",
                         static_cast<unsigned long long>(ordinal), width, height, faces,
                         static_cast<unsigned long long>(source_bytes),
                         static_cast<unsigned long long>(output_bytes));
        VkBufferMemoryBarrier before[2]{};
        for (auto& barrier : before) {
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.size = VK_WHOLE_SIZE;
        }
        before[0].buffer = input;
        before[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        before[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        before[1].buffer = output;
        before[1].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        before[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 2, before, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, program->pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, program->layout,
                                0, 1, &descriptors, 0, nullptr);
        vkCmdDispatch(cmd, (count + 127u) / 128u, 1, 1);
        VkBufferMemoryBarrier after{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        after.srcQueueFamilyIndex = after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        after.buffer = output;
        after.size = VK_WHOLE_SIZE;
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 1, &after, 0, nullptr);
    }
};

// Called under the backend's resource lock. Validate the entire source layout
// before allocation or guest reads; snapshot directly into an immutable mapped
// input buffer. The mapping stays alive through submission ownership and idle pool
// reuse, on the retained renderer device. No guest pointer or prior pixel content is
// reused; failed preparation returns its allocation without publishing an upload.
inline std::shared_ptr<GpuDetileUpload> prepare_gpu_detile_upload(
    const GpuDetilePipeline& program, VkPhysicalDevice phys,
    const VkPhysicalDeviceLimits& limits, uint32_t width, uint32_t height,
    uint32_t faces, uint64_t guest_base, uint64_t face_stride, uint64_t mip_offset,
    const std::function<size_t(uint8_t*, uint64_t, size_t)>& copy_source) {
    if (!width || !height || !faces || width > limits.maxImageDimension2D ||
        uint64_t(height) * faces > limits.maxImageDimension2D ||
        limits.maxComputeWorkGroupSize[0] < 128 || limits.maxComputeWorkGroupInvocations < 128)
        return {};
    const uint64_t count = uint64_t(width) * height * faces;
    const uint64_t face_bytes = ((uint64_t(width) + 127) / 128) *
                               ((uint64_t(height) + 63) / 64) * 65536;
    if (count > UINT32_MAX - 127u || (count + 127) / 128 > limits.maxComputeWorkGroupCount[0] ||
        face_bytes > UINT32_MAX || face_bytes > (UINT32_MAX - 80u) / faces ||
        count * 4 > limits.maxStorageBufferRange ||
        80 + face_bytes * faces > limits.maxStorageBufferRange)
        return {};
    if (!face_stride) face_stride = face_bytes;
    if (mip_offset > face_stride || face_bytes > face_stride - mip_offset ||
        mip_offset > UINT64_MAX - guest_base)
        return {};
    const uint64_t first = guest_base + mip_offset;
    if (face_stride > (UINT64_MAX - first) / faces ||
        face_bytes > UINT64_MAX - (first + uint64_t(faces - 1) * face_stride))
        return {};
    VkImageFormatProperties image_properties{};
    if (vkGetPhysicalDeviceImageFormatProperties(
            phys, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 0,
            &image_properties) != VK_SUCCESS || width > image_properties.maxExtent.width ||
        uint64_t(height) * faces > image_properties.maxExtent.height)
        return {};
    auto upload = std::make_shared<GpuDetileUpload>();
    upload->program = &program;
    upload->device = program.device;
    upload->width = width; upload->height = height; upload->faces = faces;
    upload->count = static_cast<uint32_t>(count);
    upload->source_bytes = 80 + face_bytes * faces;
    upload->output_bytes = count * 4;
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(phys, &memory);
    auto buffer = [&](VkDeviceSize bytes, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags flags, VkBuffer& buf, VkDeviceMemory& mem) {
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.size = bytes; ci.usage = usage;
        if (vkCreateBuffer(program.device, &ci, nullptr, &buf) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(program.device, buf, &requirements);
        uint32_t type = 0;
        while (type < memory.memoryTypeCount &&
               (!(requirements.memoryTypeBits & (1u << type)) ||
                (memory.memoryTypes[type].propertyFlags & flags) != flags)) ++type;
        if (type == memory.memoryTypeCount) return false;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = requirements.size; ai.memoryTypeIndex = type;
        return vkAllocateMemory(program.device, &ai, nullptr, &mem) == VK_SUCCESS &&
               vkBindBufferMemory(program.device, buf, mem, 0) == VK_SUCCESS;
    };
    const auto input = acquire_mapped_staging(program.device, upload->source_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, [&](uint32_t bits) {
            constexpr auto required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            for (uint32_t type = 0; type < memory.memoryTypeCount; ++type)
                if ((bits & (1u << type)) &&
                    (memory.memoryTypes[type].propertyFlags & required) == required) return type;
            return UINT32_MAX;
        }, &upload->input_reused);
    upload->input = input.buffer;
    upload->input_memory = input.memory;
    upload->input_mapped = input.mapped;
    upload->input_lease = input.lease;
    if (!input.mapped || !buffer(upload->output_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, upload->output, upload->output_memory))
        return {};
    void* mapped = input.mapped;
    const uint32_t header[]{width, height, static_cast<uint32_t>(face_bytes / 4),
                            static_cast<uint32_t>(count)};
    std::memcpy(mapped, header, sizeof(header));
    const auto equation = prosper::gpu::tile27_rgba16f_equation();
    std::memcpy(static_cast<uint8_t*>(mapped) + 16, equation.data(), sizeof(equation));
    bool copied = true;
    for (uint32_t face = 0; face < faces && copied; ++face)
        copied = copy_source(static_cast<uint8_t*>(mapped) + 80 + face * face_bytes,
                             first + face * face_stride, face_bytes) == face_bytes;
    if (!copied) return {};
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(program.device, &pci, nullptr, &upload->pool) != VK_SUCCESS)
        return {};
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = upload->pool; dai.descriptorSetCount = 1;
    dai.pSetLayouts = &program.descriptors;
    if (vkAllocateDescriptorSets(program.device, &dai, &upload->descriptors) != VK_SUCCESS)
        return {};
    const VkDescriptorBufferInfo infos[]{
        {upload->input, 0, upload->source_bytes}, {upload->output, 0, upload->output_bytes}};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = upload->descriptors; writes[i].dstBinding = i;
        writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(program.device, 2, writes, 0, nullptr);
    if (PROSPER_ENV_ON("PROSPER_GPU_DETILE_LOG"))
        std::fprintf(stderr, "[gpu-detile] prepared extent=%ux%ux%u guest=0x%llx input=%llu output=%llu input-reused=%d\n",
                     width, height, faces, static_cast<unsigned long long>(guest_base),
                     static_cast<unsigned long long>(upload->source_bytes),
                     static_cast<unsigned long long>(upload->output_bytes), int(upload->input_reused));
    return upload;
}
} // namespace prosper::test
