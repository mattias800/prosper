#pragma once
// Optional companion of the EXISTING compute runner, not a second Vulkan device/queue harness.
// It admits one deliberately closed immutable nearest RGBA32F upload domain. Feature witnesses
// are delivered to the compiler callback only AFTER successful enablement on this exact owner.
#include <vulkan/vulkan.h>
#include <array>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

namespace prosper::test {
struct ComputeEnabledContract {
    uint64_t device_identity = 0;
    bool shader_int64_enabled = false, rgba32_sfloat_sampled = false;
};
struct ComputeSampledMip {
    uint32_t width = 0, height = 0;
    std::vector<std::array<uint32_t, 4>> texels;
};
struct ComputeOwnedPlan {
    std::vector<uint32_t> spirv;
    std::vector<float> input;
    uint32_t output_words = 0;
    uint32_t wave_count =
        1; // explicit owned logical64 batches, never inferred guest raster waves
    std::vector<uint32_t> initial_output; // optional complete raw guard/record initialization
    // Optional binding2 raw-u32 immutable upload owner, retained through actual completion.
    // A packet caller aliases its private validated authority, never a mutable transport copy.
    std::shared_ptr<const std::vector<uint32_t>> readonly_words;
    std::vector<std::vector<ComputeSampledMip>> images; // fixed binding16+i, nearest/clamp-edge
};
struct ComputeOwnedDispatch {
    std::function<bool(const ComputeEnabledContract&, ComputeOwnedPlan&)> prepare;
    ComputeEnabledContract enabled;
    ComputeOwnedPlan plan;
    uint32_t dispatch_attempts = 0; // actual vkQueueSubmit attempts, not callback preparation
    bool completion_and_host_availability = false;
};
inline uint64_t next_compute_owner_identity() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}
struct ComputeSampledUpload {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory image_memory = VK_NULL_HANDLE, staging_memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkBuffer staging = VK_NULL_HANDLE;
    std::vector<VkBufferImageCopy> regions;
    void release(VkDevice device) {
        if (sampler) vkDestroySampler(device, sampler, nullptr);
        if (view) vkDestroyImageView(device, view, nullptr);
        if (image) vkDestroyImage(device, image, nullptr);
        if (image_memory) vkFreeMemory(device, image_memory, nullptr);
        if (staging) vkDestroyBuffer(device, staging, nullptr);
        if (staging_memory) vkFreeMemory(device, staging_memory, nullptr);
        *this = {};
    }
    bool prepare(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
                 const std::vector<ComputeSampledMip>& mips) {
        if (mips.empty() || mips.size() > 16 || !mips[0].width || !mips[0].height ||
            (mips[0].width & (mips[0].width - 1)) || (mips[0].height & (mips[0].height - 1))) return false;
        uint32_t extent = std::max(mips[0].width, mips[0].height), levels = 0;
        do { ++levels; extent >>= 1; } while (extent);
        if (mips.size() > levels) return false;
        std::vector<std::array<uint32_t, 4>> texels;
        for (uint32_t level = 0; level < mips.size(); ++level) {
            const auto& mip = mips[level];
            if (mip.width != std::max(1u, mips[0].width >> level) ||
                mip.height != std::max(1u, mips[0].height >> level) ||
                mip.texels.size() != uint64_t(mip.width) * mip.height ||
                texels.size() + mip.texels.size() > 262144) return false;
            VkBufferImageCopy region{};
            region.bufferOffset = texels.size() * sizeof(texels[0]);
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
            region.imageExtent = {mip.width, mip.height, 1};
            regions.push_back(region);
            texels.insert(texels.end(), mip.texels.begin(), mip.texels.end());
        }
        const auto type = [&](uint32_t bits, VkMemoryPropertyFlags required) {
            for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
                if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & required) == required) return i;
            return UINT32_MAX;
        };
        const auto allocate = [&](const VkMemoryRequirements& req, VkMemoryPropertyFlags properties,
                                  VkDeviceMemory& out) {
            const auto index = type(req.memoryTypeBits, properties);
            if (index == UINT32_MAX) return false;
            VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            a.allocationSize = req.size; a.memoryTypeIndex = index;
            return vkAllocateMemory(device, &a, nullptr, &out) == VK_SUCCESS;
        };
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D; info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        info.extent = {mips[0].width, mips[0].height, 1}; info.mipLevels = static_cast<uint32_t>(mips.size());
        info.arrayLayers = 1; info.samples = VK_SAMPLE_COUNT_1_BIT; info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if (vkCreateImage(device, &info, nullptr, &image) != VK_SUCCESS) return false;
        VkMemoryRequirements req{}; vkGetImageMemoryRequirements(device, image, &req);
        if (!allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image_memory) ||
            vkBindImageMemory(device, image, image_memory, 0) != VK_SUCCESS) return false;
        VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer.size = texels.size() * sizeof(texels[0]); buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(device, &buffer, nullptr, &staging) != VK_SUCCESS) return false;
        vkGetBufferMemoryRequirements(device, staging, &req);
        if (!allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_memory) ||
            vkBindBufferMemory(device, staging, staging_memory, 0) != VK_SUCCESS) return false;
        void* mapped = nullptr;
        if (vkMapMemory(device, staging_memory, 0, buffer.size, 0, &mapped) != VK_SUCCESS || !mapped) return false;
        std::memcpy(mapped, texels.data(), static_cast<size_t>(buffer.size));
        vkUnmapMemory(device, staging_memory);
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = image; view_info.viewType = VK_IMAGE_VIEW_TYPE_2D; view_info.format = info.format;
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, info.mipLevels, 0, 1};
        if (vkCreateImageView(device, &view_info, nullptr, &view) != VK_SUCCESS) return false;
        VkSamplerCreateInfo s{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        s.magFilter = s.minFilter = VK_FILTER_NEAREST; s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        s.addressModeU = s.addressModeV = s.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        s.maxLod = static_cast<float>(info.mipLevels - 1);
        return vkCreateSampler(device, &s, nullptr, &sampler) == VK_SUCCESS;
    }
    void record(VkCommandBuffer cmd) const {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image; barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, static_cast<uint32_t>(regions.size()), 0, 1};
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
        vkCmdCopyBufferToImage(cmd, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               static_cast<uint32_t>(regions.size()), regions.data());
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
};
} // namespace prosper::test
