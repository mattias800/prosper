// test_present_blit — the GPU scanout handoff (#1270) must reproduce the source front-buffer pixels
// EXACTLY, under a concurrent producer/consumer on the shared render queue.
//
// A producer thread fills a source image with a per-frame, spatially-varied pattern and calls
// present_blit_publish (the renderer's role: front image -> scanout slot). A consumer thread acquires the
// published scanout image and copies it to a host-visible buffer (standing in for the app's swapchain
// blit), then asserts every pixel equals the pattern for that frame. This validates: (a) blit fidelity --
// the scanout image is a byte-exact copy of the front buffer, so GPU-present == the CPU readback path; and
// (b) the slot handoff -- with fence-gated publish and release, the consumer never reads a half-written or
// reused image even while the producer runs ahead. Run it under VK_LAYER_KHRONOS_validation in CI to catch
// sync/layout hazards. No window/surface is needed, so it runs headlessly.
#include "render_runner.h"
#include "../frontends/shared/live_renderer.hpp"
#include "../frontends/shared/present_blit.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

using namespace prosper;

namespace {
constexpr uint32_t W = 96, H = 54;
constexpr uint64_t FRAMES = 48;

int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } } while (0)

// The per-frame source pattern (spatially varied + frame-varied so a torn/stale/reused frame is visible).
inline void fill_pattern(uint8_t* p, uint32_t w, uint32_t h, uint64_t seq) {
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            uint8_t* px = p + ((size_t)y * w + x) * 4;
            px[0] = (uint8_t)((x + seq) & 0xFF);
            px[1] = (uint8_t)((y * 3 + seq) & 0xFF);
            px[2] = (uint8_t)((seq * 5) & 0xFF);
            px[3] = 255;
        }
}

uint32_t mem_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

// Submit one command buffer on the shared render queue and wait its fence, holding the shared present
// submit mutex around the CALL exactly as the real renderer/app do.
VkResult locked_submit_wait(const test::RenderVkCtx& ctx, VkCommandBuffer cb, VkFence fence) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkResetFences(ctx.dev, 1, &fence);
    VkResult sr;
    {
        std::unique_lock<std::mutex> lk(gpu::shared_present_submit_mutex(), std::defer_lock);
        if (gpu::shared_present_active()) lk.lock();
        sr = vkQueueSubmit(ctx.queue, 1, &si, fence);
    }
    if (sr != VK_SUCCESS) return sr;
    return vkWaitForFences(ctx.dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000);
}

void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
             VkAccessFlags sa, VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from; b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask = sa; b.dstAccessMask = da;
    vkCmdPipelineBarrier(cb, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
}
} // namespace

int main() {
    frontend::register_live_renderer(std::string(), false);
    const test::RenderVkCtx& ctx = test::render_vk_ctx();
    if (!ctx.ok) { printf("test_present_blit: no Vulkan device, skipping\n"); return 0; }

    gpu::set_shared_present_active(true);   // exercise the shared-queue submit-mutex path

    const size_t bytes = (size_t)W * H * 4;

    // Producer resources: a source image (front-buffer stand-in) + a host-visible staging buffer to fill it.
    VkImage src = VK_NULL_HANDLE; VkDeviceMemory srcMem = VK_NULL_HANDLE;
    {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D; ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {W, H, 1}; ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        CHECK(vkCreateImage(ctx.dev, &ici, nullptr, &src) == VK_SUCCESS, "create source image");
        VkMemoryRequirements req{}; vkGetImageMemoryRequirements(ctx.dev, src, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = mem_type(ctx.phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        CHECK(vkAllocateMemory(ctx.dev, &mai, nullptr, &srcMem) == VK_SUCCESS, "alloc source image");
        vkBindImageMemory(ctx.dev, src, srcMem, 0);
    }
    auto make_host_buffer = [&](VkBufferUsageFlags usage, VkBuffer& buf, VkDeviceMemory& mem, void** map) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes; bci.usage = usage; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(ctx.dev, &bci, nullptr, &buf);
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(ctx.dev, buf, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = mem_type(ctx.phys, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(ctx.dev, &mai, nullptr, &mem);
        vkBindBufferMemory(ctx.dev, buf, mem, 0);
        if (map) vkMapMemory(ctx.dev, mem, 0, bytes, 0, map);
    };
    VkBuffer stage = VK_NULL_HANDLE; VkDeviceMemory stageMem = VK_NULL_HANDLE; void* stageMap = nullptr;
    make_host_buffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stage, stageMem, &stageMap);

    // Per-thread command pools (pools are externally synchronized -> one per submitting thread).
    auto make_pool = [&](VkCommandPool& pool, VkCommandBuffer& cb) {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pci.queueFamilyIndex = ctx.qfi;
        vkCreateCommandPool(ctx.dev, &pci, nullptr, &pool);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        vkAllocateCommandBuffers(ctx.dev, &ai, &cb);
    };
    VkCommandPool prodPool, consPool; VkCommandBuffer prodCb, consCb;
    make_pool(prodPool, prodCb); make_pool(consPool, consCb);
    VkFence prodFence, consFence;
    { VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      vkCreateFence(ctx.dev, &fci, nullptr, &prodFence);
      vkCreateFence(ctx.dev, &fci, nullptr, &consFence); }

    std::atomic<bool> producer_done{false};
    std::atomic<uint64_t> published{0};
    std::atomic<uint64_t> consumed{0};
    std::atomic<int> mismatches{0};
    std::atomic<uint64_t> last_frame_seq{~0ull};

    // Producer (renderer role): fill src with pattern(seq), transition it to SHADER_READ_ONLY (front-buffer
    // layout), then publish. present_blit_publish fence-waits its blit, so src is free to refill afterward.
    std::thread producer([&] {
        for (uint64_t seq = 0; seq < FRAMES; seq++) {
            fill_pattern((uint8_t*)stageMap, W, H, seq);
            vkResetCommandBuffer(prodCb, 0);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(prodCb, &bi);
            barrier(prodCb, src, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferImageCopy cp{}; cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            cp.imageExtent = {W, H, 1};
            vkCmdCopyBufferToImage(prodCb, stage, src, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
            barrier(prodCb, src, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            vkEndCommandBuffer(prodCb);
            locked_submit_wait(ctx, prodCb, prodFence);

            // Retry publish a few times if the consumer is momentarily behind (no free slot).
            bool ok = false;
            for (int tries = 0; tries < 1000 && !ok; tries++) {
                ok = frontend::present_blit_publish(src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                    VK_FORMAT_R8G8B8A8_UNORM, W, H, seq);
                if (!ok) std::this_thread::yield();
            }
            if (ok) published.fetch_add(1);
        }
        producer_done.store(true);
    });

    // Consumer (app role): acquire the latest scanout image, copy it to a host buffer, verify the pixels
    // equal the pattern for that frame_seq, then release the slot.
    std::thread consumer([&] {
        VkBuffer rb = VK_NULL_HANDLE; VkDeviceMemory rbMem = VK_NULL_HANDLE; void* rbMap = nullptr;
        make_host_buffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, rb, rbMem, &rbMap);
        std::vector<uint8_t> expected(bytes);
        // Drain until the producer is done AND no newer frame is available. The consumer legitimately
        // SKIPS superseded frames (the renderer runs ahead), so it must not wait for consumed==published.
        while (true) {
            frontend::GpuScanoutFrame f;
            if (!frontend::present_blit_acquire(f)) {
                if (producer_done.load()) break;
                std::this_thread::yield();
                continue;
            }
            if (last_frame_seq.load() == f.frame_seq) { frontend::present_blit_release(f.slot); continue; }
            last_frame_seq.store(f.frame_seq);

            vkResetCommandBuffer(consCb, 0);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(consCb, &bi);
            VkBufferImageCopy cp{}; cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            cp.imageExtent = {W, H, 1};
            vkCmdCopyImageToBuffer(consCb, f.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &cp);
            vkEndCommandBuffer(consCb);
            locked_submit_wait(ctx, consCb, consFence);

            fill_pattern(expected.data(), W, H, f.frame_seq);
            if (memcmp(rbMap, expected.data(), bytes) != 0) mismatches.fetch_add(1);
            consumed.fetch_add(1);
            frontend::present_blit_release(f.slot);
        }
        vkDestroyBuffer(ctx.dev, rb, nullptr); vkFreeMemory(ctx.dev, rbMem, nullptr);
    });

    producer.join();
    consumer.join();

    CHECK(published.load() == FRAMES, "producer published every frame");
    CHECK(consumed.load() >= 1, "consumer observed at least one frame");
    CHECK(mismatches.load() == 0, "every consumed scanout image is a byte-exact copy of its source");
    printf("test_present_blit: published=%llu consumed=%llu mismatches=%d\n",
           (unsigned long long)published.load(), (unsigned long long)consumed.load(),
           mismatches.load());

    vkDeviceWaitIdle(ctx.dev);
    frontend::present_blit_reset();
    gpu::set_shared_present_active(false);
    vkDestroyFence(ctx.dev, prodFence, nullptr); vkDestroyFence(ctx.dev, consFence, nullptr);
    vkDestroyCommandPool(ctx.dev, prodPool, nullptr); vkDestroyCommandPool(ctx.dev, consPool, nullptr);
    vkDestroyBuffer(ctx.dev, stage, nullptr); vkFreeMemory(ctx.dev, stageMem, nullptr);
    vkDestroyImage(ctx.dev, src, nullptr); vkFreeMemory(ctx.dev, srcMem, nullptr);

    printf(fails ? "test_present_blit: %d FAILURE(S)\n" : "test_present_blit: all ok\n", fails);
    return fails ? 1 : 0;
}
