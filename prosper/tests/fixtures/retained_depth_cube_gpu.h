// GPU-resident snapshot of a retained depth cube (#3873 plan item 5, the cube half).
//
// A guest depth cube is one six-face allocation whose faces prosper renders as separate retained
// D32 images (one per DB_DEPTH_VIEW slice). The recompiler lowers a cube sample onto a single
// vertically stacked `w x 6h` RGBA8 image, so the six faces must be gathered, converted and
// restacked. The original bridge (the cube-depth block in live_renderer.cpp) does that on the CPU:
// six synchronous readbacks, a CPU quantise into replicated RGBA8 (depth_cube_quantize.hpp), then
// the backend stages and uploads the result again. Outer Wilds pays that for its shadow cube about
// once per frame.
//
// This keeps the same data on the GPU, in one command buffer recorded into the SAME ordered batch
// as the producer and consumer passes (queue order replaces the readback's fence wait):
//   1. copy each face's depth aspect into one device-local buffer, one packed plane per face, in
//      face order -- exactly the byte layout of the CPU's stacked payload before conversion;
//   2. a compute pass converts every float in place to `q | q<<8 | q<<16 | 0xff000000` with
//      q = trunc(clamp(d, 0, 1) * 255 + 0.5), reproducing the CPU's separate round-to-even
//      float32 multiply and add using integer operations (build_compute_depth_to_rgba8);
//   3. copy the buffer into an R8G8B8A8_UNORM `w x 6h` image that rests in
//      SHADER_READ_ONLY_OPTIMAL, bound by the backend's borrowed-image route.
// CONFIDENCE: HIGH for finite depth: step 1 copies stored float bits and step 2 implements the
// CPU's two float32 rounding steps without depending on Vulkan's optional rounding-mode property.
// tests/shared/live/test_depth_cube_gpu.cpp compares every texel near quantisation boundaries and
// random depths. NaN maps to zero by an explicit GPU policy; the CPU's NaN-to-uint8 cast is
// undefined and is not used as an oracle.
//
// Scope is deliberately the fully renderer-owned cube (all six faces retained). A mixed cube
// (missing faces decoded from guest bytes) and the compute/DS hybrid keep the CPU path unchanged,
// as does anything this path cannot express -- no ordered batch, an abandoned batch, a device
// limit, an allocation failure: those report `Fallback`. `PROSPER_NO_GPU_DEPTH_CUBE=1` forces the
// CPU path for same-binary A/B and recovery.
#pragma once

#include "fixtures/retained_depth_array_gpu.h"
#include "gpu/recompiler/spirv_builder.hpp"

#include <array>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace prosper::test {

inline bool gpu_depth_cube_snapshots_enabled() {
    static const bool on = std::getenv("PROSPER_NO_GPU_DEPTH_CUBE") == nullptr;
    return on;
}

// Work actually recorded by this path, for tests and censuses.
struct DepthCubeGpuCopyStats {
    uint64_t copies = 0;
    uint64_t output_bytes = 0;
    uint64_t pool_hits = 0;
    uint64_t pool_misses = 0;
    uint64_t fallbacks = 0;
};
inline DepthCubeGpuCopyStats& depth_cube_gpu_copy_stats() {
    static DepthCubeGpuCopyStats stats;
    return stats;
}

// Test-only one-shot: make the next attempt report Fallback after selection, so a test can prove
// the CPU route takes over unchanged. Never armed by the live frontend.
inline bool& depth_cube_gpu_force_fallback_once() {
    static thread_local bool armed = false;
    return armed;
}

// The conversion pipeline, created once per process for the renderer's device. Deliberately
// leaked for the same reason as the snapshot pool: the device may be gone at static destruction.
struct DepthCubeGpuPipeline {
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPhysicalDeviceLimits limits{};
    bool attempted = false;
    bool ok = false;

    // One invocation per texel on the X axis; a cube too large for one dispatch or one storage
    // binding stays on the CPU path rather than growing a second code path nobody exercises.
    bool fits(uint64_t texels) const {
        return ok && texels && texels <= UINT32_MAX - 127u &&
            texels * sizeof(uint32_t) <= limits.maxStorageBufferRange &&
            (texels + 127u) / 128u <= limits.maxComputeWorkGroupCount[0];
    }
};

inline const DepthCubeGpuPipeline* depth_cube_gpu_pipeline(const RenderVkCtx& ctx) {
    static std::mutex mutex;
    static auto* state = new DepthCubeGpuPipeline;
    std::lock_guard lock(mutex);
    if (state->attempted) return state->ok && state->device == ctx.dev ? state : nullptr;
    state->attempted = true;
    state->device = ctx.dev;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(ctx.phys, &properties);
    state->limits = properties.limits;
    if (state->limits.maxComputeWorkGroupSize[0] < 128 ||
        state->limits.maxComputeWorkGroupInvocations < 128)
        return nullptr;
    const VkDescriptorSetLayoutBinding binding{
        0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 1;
    set_info.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(ctx.dev, &set_info, nullptr, &state->set_layout) != VK_SUCCESS)
        return nullptr;
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &state->set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(ctx.dev, &layout_info, nullptr, &state->layout) != VK_SUCCESS)
        return nullptr;
    const std::vector<uint32_t> words = prosper::gpu::build_compute_depth_to_rgba8();
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = words.size() * sizeof(uint32_t);
    module_info.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.dev, &module_info, nullptr, &module) != VK_SUCCESS)
        return nullptr;
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = state->layout;
    const VkResult created = vkCreateComputePipelines(ctx.dev, VK_NULL_HANDLE, 1, &pipeline_info,
                                                      nullptr, &state->pipeline);
    vkDestroyShaderModule(ctx.dev, module, nullptr);
    state->ok = created == VK_SUCCESS && state->pipeline;
    return state->ok ? state : nullptr;
}

// Owned Vulkan objects for one cube extent. The descriptor set is bound to the slot's own buffer
// once, at creation, so recording a gather performs no descriptor update.
struct DepthCubeGpuSlot {
    VkDevice device = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
    VkImage image = VK_NULL_HANDLE;           // sampled R8G8B8A8_UNORM, width x (6 * height)
    VkDeviceMemory image_memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;         // six depth planes in, packed RGBA8 texels out
    VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkDeviceSize bytes = 0;

    ~DepthCubeGpuSlot() {
        if (!device) return;
        if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        if (image) vkDestroyImage(device, image, nullptr);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (image_memory) prosper::gpu::free_device_memory(device, image_memory);
        if (buffer_memory) prosper::gpu::free_device_memory(device, buffer_memory);
    }
};

struct DepthCubeGpuPool {
    std::mutex mutex;
    std::vector<std::unique_ptr<DepthCubeGpuSlot>> free;
    VkDeviceSize free_bytes = 0;
    static constexpr size_t kMaxFreeSlots = 8;
    static constexpr VkDeviceSize kMaxFreeBytes = 256ull << 20;
};
inline DepthCubeGpuPool& depth_cube_gpu_pool() {
    static auto* pool = new DepthCubeGpuPool;   // leaked: see DepthArrayGpuPool
    return *pool;
}

// One GPU snapshot. The image holds meaningful texels only for command buffers recorded later in
// `owner`, the ordered batch that carries the gather -- the same admission rule as the retained
// depth-array snapshot (PersistentDsDepthArrayGpuImage::servable_to). `valid` drops if the owning
// batch fails. `faces`/`generations` are the exact retained identity the gather read.
struct PersistentDsDepthCubeGpuImage {
    std::unique_ptr<DepthCubeGpuSlot> slot;
    std::atomic<bool> valid{true};
    const BackendSubmissionBatch* owner = nullptr;
    std::array<VkImage, 6> faces{};
    std::array<uint64_t, 6> generations{};
    bool servable_to(const BackendSubmissionBatch* consumer_batch) const {
        return consumer_batch && consumer_batch == owner && valid.load();
    }
    VkImage image() const { return slot ? slot->image : VK_NULL_HANDLE; }
    ~PersistentDsDepthCubeGpuImage() {
        if (!slot) return;
        DepthCubeGpuPool& pool = depth_cube_gpu_pool();
        std::lock_guard lock(pool.mutex);
        // Contents are reinitialised from UNDEFINED on reuse, so even a failed batch's slot is
        // safe to recycle once its last owner (including the backend lease) has let go.
        if (pool.free.size() < DepthCubeGpuPool::kMaxFreeSlots &&
            slot->bytes <= DepthCubeGpuPool::kMaxFreeBytes - pool.free_bytes) {
            pool.free_bytes += slot->bytes;
            pool.free.push_back(std::move(slot));
        }
    }
};

inline std::unique_ptr<DepthCubeGpuSlot> depth_cube_gpu_slot(
        const RenderVkCtx& ctx, const DepthCubeGpuPipeline& pipeline, uint32_t width,
        uint32_t height) {
    DepthCubeGpuPool& pool = depth_cube_gpu_pool();
    {
        std::lock_guard lock(pool.mutex);
        for (size_t i = 0; i < pool.free.size(); ++i) {
            const DepthCubeGpuSlot& candidate = *pool.free[i];
            if (candidate.device != ctx.dev || candidate.width != width ||
                candidate.height != height)
                continue;
            auto slot = std::move(pool.free[i]);
            pool.free.erase(pool.free.begin() + static_cast<std::ptrdiff_t>(i));
            pool.free_bytes -= slot->bytes;
            ++depth_cube_gpu_copy_stats().pool_hits;
            return slot;
        }
    }
    ++depth_cube_gpu_copy_stats().pool_misses;
    auto slot = std::make_unique<DepthCubeGpuSlot>();
    slot->device = ctx.dev;
    slot->width = width;
    slot->height = height;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R8G8B8A8_UNORM;
    info.extent = {width, height * 6u, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT;   // TRANSFER_SRC: tests read the result back
    if (vkCreateImage(ctx.dev, &info, nullptr, &slot->image) != VK_SUCCESS || !slot->image) {
        slot->image = VK_NULL_HANDLE;
        return nullptr;
    }
    VkMemoryRequirements image_requirements{};
    vkGetImageMemoryRequirements(ctx.dev, slot->image, &image_requirements);
    if (!depth_array_gpu_allocate(ctx, image_requirements, slot->image_memory)) return nullptr;
    slot->bytes += image_requirements.size;
    if (vkBindImageMemory(ctx.dev, slot->image, slot->image_memory, 0) != VK_SUCCESS)
        return nullptr;
    const VkDeviceSize buffer_bytes =
        static_cast<VkDeviceSize>(width) * height * 6u * sizeof(uint32_t);
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = buffer_bytes;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (vkCreateBuffer(ctx.dev, &buffer_info, nullptr, &slot->buffer) != VK_SUCCESS ||
        !slot->buffer) {
        slot->buffer = VK_NULL_HANDLE;
        return nullptr;
    }
    VkMemoryRequirements buffer_requirements{};
    vkGetBufferMemoryRequirements(ctx.dev, slot->buffer, &buffer_requirements);
    if (!depth_array_gpu_allocate(ctx, buffer_requirements, slot->buffer_memory)) return nullptr;
    slot->bytes += buffer_requirements.size;
    if (vkBindBufferMemory(ctx.dev, slot->buffer, slot->buffer_memory, 0) != VK_SUCCESS)
        return nullptr;
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(ctx.dev, &pool_info, nullptr, &slot->descriptor_pool) != VK_SUCCESS)
        return nullptr;
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = slot->descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &pipeline.set_layout;
    if (vkAllocateDescriptorSets(ctx.dev, &set_info, &slot->descriptor_set) != VK_SUCCESS)
        return nullptr;
    const VkDescriptorBufferInfo descriptor{slot->buffer, 0, buffer_bytes};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = slot->descriptor_set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &descriptor;
    vkUpdateDescriptorSets(ctx.dev, 1, &write, 0, nullptr);
    return slot;
}

// The in-place conversion of `count` float depth words in the slot's buffer to packed RGBA8.
// The caller orders the buffer's writes before and its reads after this dispatch. Exposed so the
// execution test can drive it with depth values no attachment could hold (NaN, negative, > 1).
inline void record_depth_cube_quantize(VkCommandBuffer command,
                                       const DepthCubeGpuPipeline& pipeline,
                                       const DepthCubeGpuSlot& slot, uint32_t count) {
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1,
                            &slot.descriptor_set, 0, nullptr);
    vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count),
                       &count);
    vkCmdDispatch(command, (count + 127u) / 128u, 1, 1);
}

enum class DepthCubeGpuResult { Ready, Fallback };

// Record the gather of a fully retained cube into `batch`. On Ready, `output` owns a `w x 6h`
// RGBA8 image that the batch fills before any later command buffer in it runs. Fallback means
// "use the CPU path" and has changed nothing -- including when the selection is not six retained
// faces, which the CPU path handles by mixing in guest-decoded faces. The caller must not hold
// BackendPersistentResourceGuard.
inline DepthCubeGpuResult copy_persistent_ds_cube_depth_gpu(
        uint64_t base, uint32_t width, uint32_t height, BackendSubmissionBatch& batch,
        std::shared_ptr<PersistentDsDepthCubeGpuImage>& output) {
    output.reset();
    DepthCubeGpuCopyStats& stats = depth_cube_gpu_copy_stats();
    const auto fallback = [&stats] {
        ++stats.fallbacks;
        return DepthCubeGpuResult::Fallback;
    };
    if (batch.retains_pending_resources()) return fallback();
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) return fallback();
    const DepthCubeGpuPipeline* pipeline = depth_cube_gpu_pipeline(ctx);
    const uint64_t texels = static_cast<uint64_t>(width) * height * 6u;
    if (!pipeline || !pipeline->fits(texels)) return fallback();

    const BackendPersistentResourceGuard guard;
    const PersistentDsCubeSelection selected = select_persistent_ds_cube_depth(base, width, height);
    if (selected.present_mask != 0x3fu) return fallback();
    if (std::exchange(depth_cube_gpu_force_fallback_once(), false)) return fallback();
    // Every retained DS image rests in DEPTH_STENCIL_ATTACHMENT_OPTIMAL between passes (the CPU
    // readback relies on the same contract). A combined depth/stencil format must transition both
    // aspects together.
    std::array<VkImageAspectFlags, 6> aspects{};
    for (uint32_t face = 0; face < 6u; ++face) {
        aspects[face] = VK_IMAGE_ASPECT_DEPTH_BIT;
        for (const auto& [key, image] : persistent_ds_cache())
            if (&image == selected.faces[face] &&
                key.fmt == static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT_S8_UINT))
                aspects[face] |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    auto slot = depth_cube_gpu_slot(ctx, *pipeline, width, height);
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
    auto image_barrier = [](VkImage image, VkImageAspectFlags aspect_mask, VkImageLayout from,
                            VkImageLayout to, VkAccessFlags src, VkAccessFlags dst) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = from; barrier.newLayout = to;
        barrier.srcAccessMask = src; barrier.dstAccessMask = dst;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {aspect_mask, 0, 1, 0, 1};
        return barrier;
    };
    // Distinct face images only: one image named by two slices must not be transitioned twice.
    std::vector<std::pair<VkImage, VkImageAspectFlags>> sources;
    for (uint32_t face = 0; face < 6u; ++face) {
        const VkImage image = selected.faces[face]->image;
        bool seen = false;
        for (const auto& source : sources) seen |= source.first == image;
        if (!seen) sources.emplace_back(image, aspects[face]);
    }
    std::vector<VkImageMemoryBarrier> barriers;
    for (const auto& [image, aspect_mask] : sources)
        barriers.push_back(image_barrier(image, aspect_mask,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT));
    vkCmdPipelineBarrier(command,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
        static_cast<uint32_t>(barriers.size()), barriers.data());
    const VkDeviceSize plane_bytes = static_cast<VkDeviceSize>(width) * height * sizeof(float);
    for (uint32_t face = 0; face < 6u; ++face) {
        VkBufferImageCopy copy{};
        copy.bufferOffset = plane_bytes * face;
        copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        copy.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(command, selected.faces[face]->image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot->buffer, 1, &copy);
    }
    VkBufferMemoryBarrier buffer_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    buffer_barrier.srcQueueFamilyIndex = buffer_barrier.dstQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;
    buffer_barrier.buffer = slot->buffer;
    buffer_barrier.size = VK_WHOLE_SIZE;
    buffer_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    buffer_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barriers.clear();
    for (const auto& [image, aspect_mask] : sources)
        barriers.push_back(image_barrier(image, aspect_mask,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        0, 0, nullptr, 1, &buffer_barrier, static_cast<uint32_t>(barriers.size()),
        barriers.data());
    record_depth_cube_quantize(command, *pipeline, *slot, static_cast<uint32_t>(texels));
    buffer_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    buffer_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    // A recycled slot's previous users completed before it returned to the pool, so its old
    // contents need no ordering; UNDEFINED discards them.
    const VkImageMemoryBarrier to_destination = image_barrier(slot->image,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &buffer_barrier, 1,
                         &to_destination);
    VkBufferImageCopy upload{};
    upload.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    upload.imageExtent = {width, height * 6u, 1};
    vkCmdCopyBufferToImage(command, slot->buffer, slot->image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &upload);
    const VkImageMemoryBarrier to_sampled = image_barrier(slot->image, VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    // Every stage the backend may sample from -- the same set render_draw_pass_rgba uses.
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             (ctx.mesh_shader_enabled ? VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT : 0u),
                         0, 0, nullptr, 0, nullptr, 1, &to_sampled);
    if (vkEndCommandBuffer(command) != VK_SUCCESS) {
        // Recorded but never enqueued: nothing references the slot, so it may be recycled.
        release_render_command_pool(ctx.dev, ctx.qfi, lease);
        return fallback();
    }

    auto snapshot = std::make_shared<PersistentDsDepthCubeGpuImage>();
    for (uint32_t face = 0; face < 6u; ++face) {
        snapshot->faces[face] = selected.faces[face]->image;
        snapshot->generations[face] = selected.faces[face]->last_depth_write;
    }
    snapshot->slot = std::move(slot);
    snapshot->owner = &batch;
    batch.enqueue(command);
    batch.add_cleanup([dev = ctx.dev, qfi = ctx.qfi, lease, snapshot]() {
        release_render_command_pool(dev, qfi, lease);
    });
    std::weak_ptr<PersistentDsDepthCubeGpuImage> weak = snapshot;
    batch.add_failure_cleanup([weak]() {
        if (auto failed = weak.lock()) failed->valid.store(false);
    });
    ++stats.copies;
    stats.output_bytes += texels * sizeof(uint32_t);
    output = std::move(snapshot);
    return DepthCubeGpuResult::Ready;
}

// Current retained identity of a fully renderer-owned cube, for the callback-local memo: the
// same six images at the same depth-write generations mean a snapshot of them is still exact.
inline bool persistent_ds_cube_identity_matches(uint64_t base, uint32_t width, uint32_t height,
                                                const PersistentDsDepthCubeGpuImage& snapshot) {
    const BackendPersistentResourceGuard guard;
    const PersistentDsCubeSelection selected = select_persistent_ds_cube_depth(base, width, height);
    if (selected.present_mask != 0x3fu) return false;
    for (uint32_t face = 0; face < 6u; ++face)
        if (selected.faces[face]->image != snapshot.faces[face] ||
            selected.faces[face]->last_depth_write != snapshot.generations[face])
            return false;
    return true;
}

}  // namespace prosper::test
