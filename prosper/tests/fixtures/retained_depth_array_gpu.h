// GPU-resident snapshot of a retained Float32 depth array (#3873 plan item 5).
//
// A guest 2D-array T# can name a depth allocation whose layers prosper rendered as separate
// retained D32 images (one per DB_DEPTH_VIEW slice). Nothing can bind several images as one array,
// so the layers must be gathered into one sampled image. The original bridge
// (`read_persistent_ds_depth_array` in render_runner.h) does that on the CPU: it flushes the ordered
// producer batch, waits, reads every layer back through mapped memory, copies it into a zero-filled
// float vector, expands that into the sampled representation and hands the bytes to the backend,
// which allocates a staging buffer and uploads them again. Sonic Frontiers pays that for ~13 layers
// of 4 MiB per frame.
//
// This keeps the same data on the GPU: one command buffer, recorded into the SAME ordered batch as
// the producer and consumer passes, copies each selected layer's depth aspect into a device-local
// buffer and from there into a sampled image. Queue order replaces the fence wait. The values are
// bit-identical: a depth-aspect copy of D32_SFLOAT / D32_SFLOAT_S8_UINT yields the stored 32-bit
// float, which is exactly what the CPU path read. The expanded RGBA32 representation (kept for
// border and fetch users, whose results differ between R32 and RGBA32) is produced with a
// same-extent NEAREST blit from an R32 staging image; blit conversion fills the absent components
// with (0, 0, 1), which is the CPU path's `{d, 0, 0, 1.0f}` expansion. CONFIDENCE: HIGH for the
// compact path (a bit copy), MED-HIGH for the blit (the Vulkan conversion rules define missing
// components as 0/0/1, and an exact float-to-float blit at identical extents does not filter;
// the execution test compares every component against the CPU path).
//
// Selection is shared with the CPU path (`select_persistent_ds_depth_array`), so both refuse and
// accept exactly the same bindings. Whatever this path cannot express -- no ordered batch, an
// abandoned batch, a format without blit support, an allocation failure -- reports `Fallback` and
// the caller runs the CPU path unchanged. `PROSPER_NO_GPU_DEPTH_ARRAY=1` forces the CPU path for
// same-binary A/B and recovery.
#pragma once

#include "fixtures/render_runner.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace prosper::test {

inline bool gpu_depth_array_snapshots_enabled() {
    static const bool on = std::getenv("PROSPER_NO_GPU_DEPTH_ARRAY") == nullptr;
    return on;
}

// Work actually recorded by this path, for tests and censuses. `output_bytes` counts the sampled
// image's texels at its real format, so a compact snapshot and an expanded one are distinguishable.
struct DepthArrayGpuCopyStats {
    uint64_t copies = 0;
    uint64_t layers = 0;
    uint64_t output_bytes = 0;
    uint64_t pool_hits = 0;
    uint64_t pool_misses = 0;
};
inline DepthArrayGpuCopyStats& depth_array_gpu_copy_stats() {
    static DepthArrayGpuCopyStats stats;
    return stats;
}

// Owned Vulkan objects for one snapshot shape. Recycled through a small pool: a snapshot is
// released only when its last owner drops it, and every GPU user (the copy command's batch
// cleanup, the backend's texture-upload lease) holds a reference until its submission completed.
struct DepthArrayGpuSlot {
    VkDevice device = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0, layers = 0;
    VkImage image = VK_NULL_HANDLE;           // sampled result (format)
    VkDeviceMemory image_memory = VK_NULL_HANDLE;
    VkImage staging = VK_NULL_HANDLE;         // R32 intermediate, expanded format only
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;         // depth-aspect copy target, one packed plane per layer
    VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
    VkDeviceSize bytes = 0;                   // device allocation total, for the pool budget
    // #3905: some allocation landed off device-local memory (the #3897 fallback). Such a slot is
    // never recycled, so the next snapshot of this shape tries device-local memory again.
    bool off_device = false;

    ~DepthArrayGpuSlot() {
        if (!device) return;
        if (image) vkDestroyImage(device, image, nullptr);
        if (staging) vkDestroyImage(device, staging, nullptr);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (image_memory) prosper::gpu::free_device_memory(device, image_memory);
        if (staging_memory) prosper::gpu::free_device_memory(device, staging_memory);
        if (buffer_memory) prosper::gpu::free_device_memory(device, buffer_memory);
    }
};

struct DepthArrayGpuPool {
    std::mutex mutex;
    std::vector<std::unique_ptr<DepthArrayGpuSlot>> free;
    VkDeviceSize free_bytes = 0;
    static constexpr size_t kMaxFreeSlots = 16;
    static constexpr VkDeviceSize kMaxFreeBytes = 256ull << 20;
};
inline DepthArrayGpuPool& depth_array_gpu_pool() {
    // Deliberately leaked: the device may already be gone at static destruction. Free slots are
    // matched by VkDevice handle value; that is sound only because the renderer context is created
    // once per process -- a recreated device could reuse a stale handle value.
    static auto* pool = new DepthArrayGpuPool;
    return *pool;
}

// One GPU snapshot. Its image holds meaningful texels only for command buffers recorded later in
// `owner`, the ordered batch that carries the copy: a consumer submitted any other way (a direct
// backend submission, another batch) may run before that copy, and would see an UNDEFINED image or
// a recycled slot's previous depth. `valid` drops if the owning batch fails.
struct PersistentDsDepthArrayGpuImage {
    std::unique_ptr<DepthArrayGpuSlot> slot;
    std::atomic<bool> valid{true};
    const BackendSubmissionBatch* owner = nullptr;
    // The only admission test a memo may use to serve this snapshot to a later consumer.
    bool servable_to(const BackendSubmissionBatch* consumer_batch) const {
        return consumer_batch && consumer_batch == owner && valid.load();
    }
    VkImage image() const { return slot ? slot->image : VK_NULL_HANDLE; }
    VkFormat format() const { return slot ? slot->format : VK_FORMAT_UNDEFINED; }
    ~PersistentDsDepthArrayGpuImage() {
        if (!slot) return;
        DepthArrayGpuPool& pool = depth_array_gpu_pool();
        std::lock_guard lock(pool.mutex);
        // A snapshot whose batch failed has undefined contents and possibly an unknown layout;
        // the slot is reinitialised from UNDEFINED on reuse, so recycling it is still correct.
        if (!slot->off_device && pool.free.size() < DepthArrayGpuPool::kMaxFreeSlots &&
            slot->bytes <= DepthArrayGpuPool::kMaxFreeBytes - pool.free_bytes) {
            pool.free_bytes += slot->bytes;
            pool.free.push_back(std::move(slot));
        }
    }
};

inline bool depth_array_gpu_allocate(const RenderVkCtx& ctx, VkMemoryRequirements requirements,
                                     VkDeviceMemory& memory, bool& off_device) {
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    if (render_allocate_gpu_only_memory(ctx.phys, ctx.dev,
                                        prosper::gpu::GpuOnlyMemoryClass::RetainedDepthArray,
                                        requirements.memoryTypeBits, allocation,
                                        &memory) != VK_SUCCESS) {
        memory = VK_NULL_HANDLE;
        return false;
    }
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(ctx.phys, &props);
    if (!prosper::gpu::memory_type_is_device_local(props, allocation.memoryTypeIndex))
        off_device = true;
    return true;
}

inline std::unique_ptr<DepthArrayGpuSlot> depth_array_gpu_slot(
        const RenderVkCtx& ctx, VkFormat format, uint32_t width, uint32_t height,
        uint32_t layers) {
    DepthArrayGpuPool& pool = depth_array_gpu_pool();
    {
        std::lock_guard lock(pool.mutex);
        for (size_t i = 0; i < pool.free.size(); ++i) {
            DepthArrayGpuSlot& candidate = *pool.free[i];
            if (candidate.device != ctx.dev || candidate.format != format ||
                candidate.width != width || candidate.height != height ||
                candidate.layers != layers)
                continue;
            auto slot = std::move(pool.free[i]);
            pool.free.erase(pool.free.begin() + static_cast<std::ptrdiff_t>(i));
            pool.free_bytes -= slot->bytes;
            ++depth_array_gpu_copy_stats().pool_hits;
            return slot;
        }
    }
    ++depth_array_gpu_copy_stats().pool_misses;
    auto slot = std::make_unique<DepthArrayGpuSlot>();
    slot->device = ctx.dev;
    slot->format = format;
    slot->width = width; slot->height = height; slot->layers = layers;
    const bool expanded = format != VK_FORMAT_R32_SFLOAT;
    auto make_image = [&](VkFormat image_format, VkImageUsageFlags usage, VkImage& image,
                          VkDeviceMemory& memory) {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = image_format;
        info.extent = {width, height, 1};
        info.mipLevels = 1;
        info.arrayLayers = layers;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        if (vkCreateImage(ctx.dev, &info, nullptr, &image) != VK_SUCCESS || !image) {
            image = VK_NULL_HANDLE;
            return false;
        }
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(ctx.dev, image, &requirements);
        if (!depth_array_gpu_allocate(ctx, requirements, memory, slot->off_device)) return false;
        slot->bytes += requirements.size;
        return vkBindImageMemory(ctx.dev, image, memory, 0) == VK_SUCCESS;
    };
    if (!make_image(format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    slot->image, slot->image_memory))
        return nullptr;
    if (expanded &&
        !make_image(VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    slot->staging, slot->staging_memory))
        return nullptr;
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = static_cast<VkDeviceSize>(width) * height * layers * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(ctx.dev, &buffer_info, nullptr, &slot->buffer) != VK_SUCCESS ||
        !slot->buffer) {
        slot->buffer = VK_NULL_HANDLE;
        return nullptr;
    }
    VkMemoryRequirements buffer_requirements{};
    vkGetBufferMemoryRequirements(ctx.dev, slot->buffer, &buffer_requirements);
    if (!depth_array_gpu_allocate(ctx, buffer_requirements, slot->buffer_memory,
                                  slot->off_device)) return nullptr;
    slot->bytes += buffer_requirements.size;
    if (vkBindBufferMemory(ctx.dev, slot->buffer, slot->buffer_memory, 0) != VK_SUCCESS)
        return nullptr;
    return slot;
}

enum class DepthArrayGpuResult { NoIdentity, Ready, Unavailable, Fallback };

// Record the gather into `batch`. On Ready, `output` owns an image that the batch will fill before
// any later command buffer in it runs, resting in SHADER_READ_ONLY_OPTIMAL. Unavailable carries the
// same refusal the CPU path would have produced for this identity; Fallback means "use the CPU
// path" and has changed nothing. The caller must not hold BackendPersistentResourceGuard.
inline DepthArrayGpuResult copy_persistent_ds_depth_array_gpu(
        uint64_t base, uint32_t width, uint32_t height, uint32_t first_layer,
        uint32_t layer_count, VkFormat output_format, BackendSubmissionBatch& batch,
        std::shared_ptr<PersistentDsDepthArrayGpuImage>& output, std::string& error,
        const DepthArrayGuestSource* guest_source = nullptr) {
    output.reset();
    if (batch.retains_pending_resources()) return DepthArrayGpuResult::Fallback;
    if (output_format != VK_FORMAT_R32_SFLOAT && output_format != VK_FORMAT_R32G32B32A32_SFLOAT)
        return DepthArrayGpuResult::Fallback;
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) return DepthArrayGpuResult::Fallback;
    if (output_format != VK_FORMAT_R32_SFLOAT) {
        VkFormatProperties source{}, target{};
        vkGetPhysicalDeviceFormatProperties(ctx.phys, VK_FORMAT_R32_SFLOAT, &source);
        vkGetPhysicalDeviceFormatProperties(ctx.phys, output_format, &target);
        if (!(source.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ||
            !(target.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
            return DepthArrayGpuResult::Fallback;
    }

    const BackendPersistentResourceGuard guard;
    // The same one-attempt failure injection the CPU path honours, so a test can drive the
    // frontend's transactional refusal through either route.
    int fail_after = depth_array_readback_failure_after_layers();
    depth_array_readback_failure_after_layers() = -1;
    std::vector<PersistentDsImage*> selected;
    std::vector<DepthArrayGuestLayer> guest_layers;
    uint32_t source_format = VK_FORMAT_UNDEFINED;
    const PersistentDsDepthArrayStatus selection = select_persistent_ds_depth_array(
        base, width, height, first_layer, layer_count, selected, source_format, error, &batch,
        /*flush_producer=*/false, guest_source, &guest_layers);
    if (selection == PersistentDsDepthArrayStatus::NoIdentity) return DepthArrayGpuResult::NoIdentity;
    if (selection == PersistentDsDepthArrayStatus::Unavailable) return DepthArrayGpuResult::Unavailable;
    if (fail_after >= 0 && static_cast<uint32_t>(fail_after) < layer_count) {
        error = "injected retained depth array readback failure";
        return DepthArrayGpuResult::Unavailable;
    }

    // A Fallback from here on hands the attempt to the CPU reader, which must still see any
    // injected failure this call consumed.
    const auto fallback = [fail_after] {
        depth_array_readback_failure_after_layers() = fail_after;
        return DepthArrayGpuResult::Fallback;
    };
    // A guest-authoritative layer (#3893) is written into the packed plane buffer with
    // vkCmdFillBuffer, which is exact only for a uniform layer -- the case the guest leaves when it
    // never writes the layer, or clears it. Anything else needs a detile, which the CPU route does.
    for (const DepthArrayGuestLayer& guest : guest_layers)
        if (!guest.uniform) return fallback();
    auto slot = depth_array_gpu_slot(ctx, output_format, width, height, layer_count);
    if (!slot) return fallback();
    const RenderCommandPoolLease lease = acquire_render_command_pool(ctx.dev, ctx.qfi);
    if (!lease) return fallback();
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(lease.command, &begin) != VK_SUCCESS) {
        release_render_command_pool(ctx.dev, ctx.qfi, lease);
        return fallback();
    }
    const VkCommandBuffer command = lease.command;
    const bool expanded = output_format != VK_FORMAT_R32_SFLOAT;
    const VkImage copy_target = expanded ? slot->staging : slot->image;
    const VkImageAspectFlags ds_aspects = source_format == VK_FORMAT_D32_SFLOAT_S8_UINT
        ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VK_IMAGE_ASPECT_DEPTH_BIT;
    auto image_barrier = [](VkImage image, VkImageAspectFlags aspects, VkImageLayout from,
                            VkImageLayout to, VkAccessFlags src, VkAccessFlags dst,
                            uint32_t layers) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = from; barrier.newLayout = to;
        barrier.srcAccessMask = src; barrier.dstAccessMask = dst;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {aspects, 0, 1, 0, layers};
        return barrier;
    };
    // Distinct retained images, in layer order. Every retained DS image rests in
    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL between passes (the CPU readback relies on the same contract).
    std::vector<VkImage> sources;
    for (const PersistentDsImage* image : selected)
        if (image && std::find(sources.begin(), sources.end(), image->image) == sources.end())
            sources.push_back(image->image);
    std::vector<VkImageMemoryBarrier> barriers;
    for (VkImage source : sources)
        barriers.push_back(image_barrier(source, ds_aspects,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, 1));
    // A recycled slot's previous users completed before it returned to the pool, so its old
    // contents need no ordering; UNDEFINED discards them.
    barriers.push_back(image_barrier(copy_target, VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
        VK_ACCESS_TRANSFER_WRITE_BIT, layer_count));
    if (expanded)
        barriers.push_back(image_barrier(slot->image, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, layer_count));
    vkCmdPipelineBarrier(command,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
        static_cast<uint32_t>(barriers.size()), barriers.data());

    const VkDeviceSize plane_bytes = static_cast<VkDeviceSize>(width) * height * sizeof(float);
    for (const DepthArrayGuestLayer& guest : guest_layers)
        vkCmdFillBuffer(command, slot->buffer, plane_bytes * guest.layer, plane_bytes, guest.word);
    for (uint32_t layer = 0; layer < layer_count; ++layer) {
        if (!selected[layer]) continue;   // a guest layer, filled above
        VkBufferImageCopy copy{};
        copy.bufferOffset = plane_bytes * layer;
        copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        copy.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(command, selected[layer]->image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot->buffer, 1, &copy);
    }
    VkBufferMemoryBarrier buffer_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    buffer_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    buffer_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    buffer_barrier.srcQueueFamilyIndex = buffer_barrier.dstQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;
    buffer_barrier.buffer = slot->buffer;
    buffer_barrier.size = VK_WHOLE_SIZE;
    barriers.clear();
    for (VkImage source : sources)
        barriers.push_back(image_barrier(source, ds_aspects,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, 1));
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        0, 0, nullptr, 1, &buffer_barrier, static_cast<uint32_t>(barriers.size()),
        barriers.data());
    VkBufferImageCopy upload{};
    upload.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layer_count};
    upload.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(command, slot->buffer, copy_target,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &upload);
    if (expanded) {
        const VkImageMemoryBarrier to_source = image_barrier(copy_target,
            VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, layer_count);
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &to_source);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layer_count};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layer_count};
        blit.srcOffsets[1] = {static_cast<int32_t>(width), static_cast<int32_t>(height), 1};
        blit.dstOffsets[1] = blit.srcOffsets[1];
        vkCmdBlitImage(command, copy_target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot->image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    }
    const VkImageMemoryBarrier to_sampled = image_barrier(slot->image, VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, layer_count);
    // Every stage the backend may sample from, including the optional mesh stage when the device
    // enabled it (the bit is invalid otherwise) -- the same set render_draw_pass_rgba uses.
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             (ctx.mesh_shader_enabled ? VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT : 0u),
                         0, 0, nullptr, 0, nullptr, 1, &to_sampled);
    if (vkEndCommandBuffer(command) != VK_SUCCESS) {
        // Recorded but never enqueued: nothing can reference the slot, so it may be recycled.
        release_render_command_pool(ctx.dev, ctx.qfi, lease);
        return fallback();
    }

    auto snapshot = std::make_shared<PersistentDsDepthArrayGpuImage>();
    const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(width) * height * layer_count *
        (expanded ? 4u : 1u) * sizeof(float);
    snapshot->slot = std::move(slot);
    snapshot->owner = &batch;
    batch.enqueue(command);
    batch.add_cleanup([dev = ctx.dev, qfi = ctx.qfi, lease, snapshot]() {
        release_render_command_pool(dev, qfi, lease);
    });
    std::weak_ptr<PersistentDsDepthArrayGpuImage> weak = snapshot;
    batch.add_failure_cleanup([weak]() {
        if (auto failed = weak.lock()) failed->valid.store(false);
    });
    DepthArrayGpuCopyStats& stats = depth_array_gpu_copy_stats();
    ++stats.copies;
    stats.layers += layer_count;
    stats.output_bytes += output_bytes;
    output = std::move(snapshot);
    return DepthArrayGpuResult::Ready;
}

}  // namespace prosper::test
