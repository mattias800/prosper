#pragma once
// staging_mirror_copy.hpp -- copy a compute result's canonical staging bytes into a borrowed renderer image.
//
// The exact-result mirror (live_compute.cpp) reaches the renderer image in one of two ways. When the
// staging buffer already holds the destination's own texel encoding (RGBA8, RGBA16F, R8, packed R11) a
// buffer-to-image copy moves the bits unchanged, even when the private storage image was typed differently
// (R11 is stored as R32_UINT). A result whose encoding differs from the destination's goes through a
// converting blit instead (unorm10_mirror.hpp). These helpers hold the first path's two commands so the
// caller reads as the choice between the two, plus the bit copy back into a seeding import.
#include <cstdint>

#include <vulkan/vulkan.h>

namespace prosper::frontend {

// Orders the writes that produced `staging` (a retile shader or an image-to-buffer transfer) before the
// transfer that reads it.
inline void record_staging_ready_for_transfer(VkCommandBuffer command, VkBuffer staging,
                                              VkDeviceSize bytes) {
    VkBufferMemoryBarrier ready{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ready.srcQueueFamilyIndex = ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ready.buffer = staging;
    ready.size = bytes;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &ready, 0, nullptr);
}

// Copies width x height tightly packed texels from `staging` into mip 0 / layer 0 of `image`, which the
// caller has moved to TRANSFER_DST_OPTIMAL.
inline void record_staging_to_image_copy(VkCommandBuffer command, VkBuffer staging, VkImage image,
                                         uint32_t width, uint32_t height) {
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(command, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

// Copies a dispatch's result (`result`, already in TRANSFER_SRC_OPTIMAL) into the imported image it was
// seeded from (`imported`, GENERAL and only read by the dispatch), returning `imported` to GENERAL for the
// next compute reader. Same texel format on both sides, so a bit copy.
inline void record_result_to_imported_copy(VkCommandBuffer command, VkImage result, VkImage imported,
                                           uint32_t width, uint32_t height, uint32_t depth) {
    VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    to_dst.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_dst.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.image = imported;
    to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &to_dst);
    VkImageCopy copy{};
    copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {width, height, depth};
    vkCmdCopyImage(command, result, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imported,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    VkImageMemoryBarrier to_general = to_dst;
    to_general.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_general.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &to_general);
}

}  // namespace prosper::frontend
