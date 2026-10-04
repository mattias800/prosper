// Device seed of a storage binding from a BGRA render target that the renderer holds as canonical
// RGBA8 (#4291 follow-up).
//
// Compute must read such a target in guest (BGRA) byte order, and the identity storage view cannot
// be swizzled. #4291 therefore declined the raw seed and fell back to a CPU snapshot plus an R<->B
// swap of the whole surface -- on GTA V's bank scene a 31.6 MiB round trip per binding. Instead,
// blit the canonical image into this B8G8R8A8 scratch -- a per-channel conversion -- and copy its
// bytes raw into the RGBA8 binding, whose identity view then reads byte 0 (B) as R: guest order.
// The UNORM8 -> float -> UNORM8 round trip is exact where the driver rounds to nearest (RADV,
// NVIDIA and lavapipe do); Vulkan only says it SHOULD, so a truncating driver would be one LSB low
// on some values (#4428 review). One image, regrown on demand; it is never shrunk. Every live
// compute item waits on its fence, so no use outlives the item that recorded it; within an item
// each use starts with a barrier.
#pragma once
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace prosper::frontend {
// Owned by the compute context, which releases it before its device.
struct BgraSeedScratch {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
    int blit_supported = -1;   // -1 unprobed

    // True when `image` can hold a width x height destination. A grown image is created before the
    // old one is released, so a failure leaves earlier admissions in the same item valid.
    // `allocate(requirements)` returns bound-ready device memory or null; `release(memory)` frees it.
    template <class Allocate, class Release>
    bool prepare(VkPhysicalDevice physical, VkDevice device, uint32_t queue_family, bool disabled,
                 uint32_t w, uint32_t h, Allocate&& allocate, Release&& release) {
        if (blit_supported < 0) {
            blit_supported = 0;
            // vkCmdBlitImage needs a graphics-capable queue; the compute context normally adopts
            // the renderer's graphics family, but a private compute-only device would not.
            uint32_t families = 0;
            if (physical) vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
            std::vector<VkQueueFamilyProperties> properties(families);
            if (families)
                vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, properties.data());
            const bool graphics_queue = queue_family < families &&
                (properties[queue_family].queueFlags & VK_QUEUE_GRAPHICS_BIT);
            if (physical && graphics_queue && !disabled) {
                VkFormatProperties canonical{}, swapped{};
                vkGetPhysicalDeviceFormatProperties(physical, VK_FORMAT_R8G8B8A8_UNORM, &canonical);
                vkGetPhysicalDeviceFormatProperties(physical, VK_FORMAT_B8G8R8A8_UNORM, &swapped);
                const VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_BLIT_DST_BIT |
                    VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
                blit_supported =
                    (canonical.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) &&
                    (swapped.optimalTilingFeatures & needed) == needed;
            }
        }
        if (!blit_supported || !w || !h) return false;
        if (image && w <= width && h <= height) return true;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_B8G8R8A8_UNORM;
        ici.extent = {std::max(w, width), std::max(h, height), 1};
        ici.mipLevels = ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage grown = VK_NULL_HANDLE;
        if (vkCreateImage(device, &ici, nullptr, &grown) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, grown, &requirements);
        VkDeviceMemory grown_memory = allocate(requirements);
        if (!grown_memory || vkBindImageMemory(device, grown, grown_memory, 0) != VK_SUCCESS) {
            if (grown_memory) release(grown_memory);
            vkDestroyImage(device, grown, nullptr);
            return false;
        }
        destroy(device, release);
        image = grown;
        memory = grown_memory;
        width = ici.extent.width;
        height = ici.extent.height;
        return true;
    }

    template <class Release>
    void destroy(VkDevice device, Release&& release) {
        if (image) vkDestroyImage(device, image, nullptr);
        if (memory) release(memory);
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        width = height = 0;
    }

    // `source` is in TRANSFER_SRC_OPTIMAL and `destination` in TRANSFER_DST_OPTIMAL; both stay so.
    // Same extent, so NEAREST samples texel centres and no texel is filtered.
    void record(VkCommandBuffer command, VkImage source, VkImage destination, uint32_t w,
                uint32_t h) const {
        VkImageMemoryBarrier to_blit{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_blit.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;   // an earlier use in this item
        to_blit.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_blit.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_blit.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_blit.srcQueueFamilyIndex = to_blit.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_blit.image = image;
        to_blit.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_blit);
        VkImageBlit blit{};
        blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = blit.dstOffsets[1] = {int32_t(w), int32_t(h), 1};
        vkCmdBlitImage(command, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
        VkImageMemoryBarrier to_copy = to_blit;
        to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_copy.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_copy);
        VkImageCopy copy{};
        copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.extent = {w, h, 1};
        vkCmdCopyImage(command, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    }
};
} // namespace prosper::frontend
