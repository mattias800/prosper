// test_present_blit — the GPU scanout handoff (#1270) must reproduce the source front-buffer pixels
// EXACTLY, under a concurrent producer/consumer on the shared render queue.
//
// A producer thread fills a source image with a per-frame, spatially-varied pattern and calls
// present_blit_publish (the renderer's role: front image -> scanout slot). A consumer thread acquires the
// published scanout image and copies it to a host-visible buffer (standing in for the app's swapchain
// blit), then asserts every pixel equals the pattern for that frame. This validates: (a) blit fidelity --
// the scanout image is a byte-exact copy of the front buffer, so GPU-present == the CPU readback path; and
// (b) the slot handoff -- with fence-gated publish and release, the consumer never reads a half-written or
// reused image even while the producer runs ahead. No window/surface is needed, so it runs headlessly.
//
// Source fixtures carry SAMPLED_BIT because they model the renderer's shader-read layout.
// Strict core/synchronization validation is the regression guard for that usage contract (#1716).
#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"
#include "shared/live/live_renderer.hpp"
#include "shared/present/present_blit.hpp"
#include "shared/perf/performance_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/present/videoout_present.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
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

    // #1270 Finding 3: the real front buffer can be R16G16B16A16_SFLOAT; present_blit converts it to the
    // RGBA8 scanout format. Verify that conversion (including the HDR >1.0 clamp) with exactly-representable
    // half values, single-threaded (the concurrent test above already covers the handoff mechanics).
    {
        const uint32_t rw = 32, rh = 16;
        const size_t rsrcBytes = (size_t)rw * rh * 8, rdstBytes = (size_t)rw * rh * 4;
        VkImage r16 = VK_NULL_HANDLE; VkDeviceMemory r16Mem = VK_NULL_HANDLE;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D; ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ici.extent = {rw, rh, 1}; ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCreateImage(ctx.dev, &ici, nullptr, &r16);
        VkMemoryRequirements rq{}; vkGetImageMemoryRequirements(ctx.dev, r16, &rq);
        VkMemoryAllocateInfo rmai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; rmai.allocationSize = rq.size;
        rmai.memoryTypeIndex = mem_type(ctx.phys, rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(ctx.dev, &rmai, nullptr, &r16Mem); vkBindImageMemory(ctx.dev, r16, r16Mem, 0);
        auto hbuf = [&](VkDeviceSize sz, VkBufferUsageFlags u, VkBuffer& b, VkDeviceMemory& m, void** mp) {
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = sz; bci.usage = u; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            vkCreateBuffer(ctx.dev, &bci, nullptr, &b);
            VkMemoryRequirements q{}; vkGetBufferMemoryRequirements(ctx.dev, b, &q);
            VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; a.allocationSize = q.size;
            a.memoryTypeIndex = mem_type(ctx.phys, q.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(ctx.dev, &a, nullptr, &m); vkBindBufferMemory(ctx.dev, b, m, 0);
            if (mp) vkMapMemory(ctx.dev, m, 0, sz, 0, mp);
        };
        VkBuffer rstage, rrb; VkDeviceMemory rstageMem, rrbMem; void* rstageMap = nullptr; void* rrbMap = nullptr;
        hbuf(rsrcBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, rstage, rstageMem, &rstageMap);
        hbuf(rdstBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, rrb, rrbMem, &rrbMap);
        struct HCase { uint16_t half; uint8_t expect; };
        const HCase cases[] = {{0x0000, 0}, {0x3C00, 255}, {0x4000, 255}};   // 0.0->0, 1.0->255, 2.0->255(clamp)
        int r16_mismatches = 0; uint64_t seq = 1000;
        for (const HCase& c : cases) {
            uint16_t* pmap = (uint16_t*)rstageMap;
            for (size_t i = 0; i < (size_t)rw * rh * 4; i++) pmap[i] = c.half;
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkResetCommandBuffer(prodCb, 0); vkBeginCommandBuffer(prodCb, &bi);
            barrier(prodCb, r16, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferImageCopy cp{}; cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp.imageExtent = {rw, rh, 1};
            vkCmdCopyBufferToImage(prodCb, rstage, r16, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
            barrier(prodCb, r16, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            vkEndCommandBuffer(prodCb); locked_submit_wait(ctx, prodCb, prodFence);
            CHECK(frontend::present_blit_publish(r16, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_FORMAT_R16G16B16A16_SFLOAT, rw, rh, seq++), "R16F publish succeeded");
            frontend::GpuScanoutFrame gf; bool acq = false;
            for (int t = 0; t < 1000 && !acq; t++) { acq = frontend::present_blit_acquire(gf); if (!acq) std::this_thread::yield(); }
            CHECK(acq, "R16F acquire succeeded");
            if (acq) {
                vkResetCommandBuffer(prodCb, 0); vkBeginCommandBuffer(prodCb, &bi);
                VkBufferImageCopy cp2{}; cp2.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp2.imageExtent = {rw, rh, 1};
                vkCmdCopyImageToBuffer(prodCb, gf.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rrb, 1, &cp2);
                vkEndCommandBuffer(prodCb); locked_submit_wait(ctx, prodCb, prodFence);
                const uint8_t* out = (const uint8_t*)rrbMap;
                for (size_t i = 0; i < rdstBytes; i++) if (out[i] != c.expect) { r16_mismatches++; break; }
                frontend::present_blit_release(gf.slot);
            }
        }
        CHECK(r16_mismatches == 0, "R16G16B16A16_SFLOAT source converts + clamps to the expected RGBA8");
        printf("test_present_blit: r16f_conversion mismatches=%d\n", r16_mismatches);
        vkDeviceWaitIdle(ctx.dev);
        vkDestroyBuffer(ctx.dev, rstage, nullptr); vkFreeMemory(ctx.dev, rstageMem, nullptr);
        vkDestroyBuffer(ctx.dev, rrb, nullptr); vkFreeMemory(ctx.dev, rrbMem, nullptr);
        vkDestroyImage(ctx.dev, r16, nullptr); vkFreeMemory(ctx.dev, r16Mem, nullptr);
    }

    // #1270 Finding 1 (direct guard): an ACQUIRED (in-flight) slot's image must survive a different-size
    // publish. The resize UAF this replaces would free the held image here. Publish+acquire frame A at
    // WxH; publish several frames at a DIFFERENT size (which recreate OTHER free slots); the held image
    // must still be valid and still contain A's pixels.
    {
        const uint32_t w2 = 48, h2 = 27;
        const size_t bytes2 = (size_t)w2 * h2 * 4;
        VkImage src2 = VK_NULL_HANDLE; VkDeviceMemory src2Mem = VK_NULL_HANDLE;
        { VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
          ici.imageType = VK_IMAGE_TYPE_2D; ici.format = VK_FORMAT_R8G8B8A8_UNORM; ici.extent = {w2, h2, 1};
          ici.mipLevels = 1; ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT;
          ici.tiling = VK_IMAGE_TILING_OPTIMAL;
          ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
          ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
          vkCreateImage(ctx.dev, &ici, nullptr, &src2);
          VkMemoryRequirements q{}; vkGetImageMemoryRequirements(ctx.dev, src2, &q);
          VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; a.allocationSize = q.size;
          a.memoryTypeIndex = mem_type(ctx.phys, q.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
          vkAllocateMemory(ctx.dev, &a, nullptr, &src2Mem); vkBindImageMemory(ctx.dev, src2, src2Mem, 0); }
        VkBuffer st2, rbA; VkDeviceMemory st2Mem, rbAMem; void* st2Map = nullptr; void* rbAMap = nullptr;
        auto hbuf2 = [&](VkDeviceSize sz, VkBufferUsageFlags u, VkBuffer& b, VkDeviceMemory& m, void** mp) {
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = sz; bci.usage = u; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            vkCreateBuffer(ctx.dev, &bci, nullptr, &b);
            VkMemoryRequirements q{}; vkGetBufferMemoryRequirements(ctx.dev, b, &q);
            VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; a.allocationSize = q.size;
            a.memoryTypeIndex = mem_type(ctx.phys, q.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(ctx.dev, &a, nullptr, &m); vkBindBufferMemory(ctx.dev, b, m, 0);
            if (mp) vkMapMemory(ctx.dev, m, 0, sz, 0, mp);
        };
        hbuf2(bytes2, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, st2, st2Mem, &st2Map);
        hbuf2((size_t)W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, rbA, rbAMem, &rbAMap);
        auto upload_and_publish = [&](VkImage img, VkBuffer stg, void* stgMap, uint32_t iw, uint32_t ih,
                                      uint64_t seq) -> bool {
            fill_pattern((uint8_t*)stgMap, iw, ih, seq);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkResetCommandBuffer(prodCb, 0); vkBeginCommandBuffer(prodCb, &bi);
            barrier(prodCb, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferImageCopy cp{}; cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp.imageExtent = {iw, ih, 1};
            vkCmdCopyBufferToImage(prodCb, stg, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
            barrier(prodCb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            vkEndCommandBuffer(prodCb); locked_submit_wait(ctx, prodCb, prodFence);
            return frontend::present_blit_publish(img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_FORMAT_R8G8B8A8_UNORM, iw, ih, seq,
                    {seq < 800 ? 77u : 88u, {seq < 800 ? 77u : 88u, seq}});
        };
        // Record the actual held-slot/supersession path in the opt-in CTest variant. Recording
        // synthetic "published" rows cannot prove the production handoff emitted its decisions.
        auto& handoffs = perf::interactive_performance_capture();
        const bool trace_requested = std::getenv("PROSPER_PRESENT_HANDOFF_TRACE") != nullptr;
        const auto trace_start = perf::monotonic_now_ns();
        if (trace_requested)
            CHECK(handoffs.arm("present_blit_trace_test", "fixture", "fixture", "test",
                               trace_start, std::chrono::system_clock::now()).ok,
                  "held-slot: arm production handoff recording");
        // Frame A at WxH; hold it.
        CHECK(upload_and_publish(src, stage, stageMap, W, H, 777), "held-slot: publish A (WxH)");
        frontend::GpuScanoutFrame gfA; bool gotA = false;
        for (int t = 0; t < 1000 && !gotA; t++) { gotA = frontend::present_blit_acquire(gfA); if (!gotA) std::this_thread::yield(); }
        CHECK(gotA && gfA.width == W && gfA.height == H, "held-slot: acquire A");
        CHECK(gotA && gfA.producer.completed.work == 777 &&
                  gfA.producer.image_registration == 77,
              "held-slot: acquired lineage belongs to the held image");
        // Publish several DIFFERENT-size frames; these recreate other free slots, never the held one.
        for (uint64_t k = 0; k < 4; k++) upload_and_publish(src2, st2, st2Map, w2, h2, 800 + k);
        // The held image must still be valid AND still contain A's pixels.
        int held_mismatch = 1;
        if (gotA) {
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkResetCommandBuffer(consCb, 0); vkBeginCommandBuffer(consCb, &bi);
            VkBufferImageCopy cp{}; cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp.imageExtent = {W, H, 1};
            vkCmdCopyImageToBuffer(consCb, gfA.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rbA, 1, &cp);
            vkEndCommandBuffer(consCb); locked_submit_wait(ctx, consCb, consFence);
            std::vector<uint8_t> expA((size_t)W * H * 4); fill_pattern(expA.data(), W, H, 777);
            held_mismatch = memcmp(rbAMap, expA.data(), expA.size()) != 0 ? 1 : 0;
            frontend::present_blit_release(gfA.slot);
        }
        CHECK(held_mismatch == 0, "held-slot: acquired image survives a different-size publish, content intact");
        CHECK(gfA.producer.completed.work == 777 && gfA.producer.image_registration == 77,
              "held-slot: resize and slot replacement cannot mutate the held provenance");
        if (trace_requested) {
            perf::ProcessSample finish;
            finish.monotonic_ns = trace_start + 5'000'000'000ull;
            handoffs.observe_sample(finish);
            perf::CaptureOutcome outcome;
            CHECK(handoffs.take_outcome(outcome) && outcome.ok && outcome.present_dropped == 0,
                  "held-slot: production trace completes without truncation");
            std::ifstream file(outcome.path);
            std::string line;
            unsigned replacements = 0, published = 0, acquired_a = 0, released_a = 0;
            while (std::getline(file, line)) {
                if (line.find("\"type\":\"present-handoff\"") == std::string::npos) continue;
                if (line.find("\"event\":\"published\"") != std::string::npos) ++published;
                for (unsigned source = 800; source < 803; ++source)
                    if (line.find("\"event\":\"superseded\"") != std::string::npos &&
                        line.find("\"source_seq\":" + std::to_string(source) + ",") != std::string::npos)
                        ++replacements;
                const auto id = "\"publication_id\":" + std::to_string(gfA.publication_id) + ",";
                if (line.find(id) != std::string::npos &&
                    line.find("\"source_seq\":777,") != std::string::npos) {
                    acquired_a += line.find("\"event\":\"acquired\"") != std::string::npos;
                    released_a += line.find("\"event\":\"released\"") != std::string::npos;
                }
            }
            CHECK(published == 5 && replacements == 3 && acquired_a == 1 && released_a == 1,
                  "held-slot: trace distinguishes five publications, three replacements and A's lease");
            CHECK(gfA.publication_id != 0 && gfA.publication_id != gfA.frame_seq,
                  "held-slot: publication identity is not substituted with the guest flip");
            std::error_code ignored;
            std::filesystem::remove_all("present_blit_trace_test", ignored);
        }
        vkDeviceWaitIdle(ctx.dev);
        vkDestroyBuffer(ctx.dev, st2, nullptr); vkFreeMemory(ctx.dev, st2Mem, nullptr);
        vkDestroyBuffer(ctx.dev, rbA, nullptr); vkFreeMemory(ctx.dev, rbAMem, nullptr);
        vkDestroyImage(ctx.dev, src2, nullptr); vkFreeMemory(ctx.dev, src2Mem, nullptr);
    }

    // Completed backend draw -> exact retained image -> production GPU scanout slot. The draw is
    // repeated with identical shader/pixels: new completed work must remain new even when pixels
    // match. Guest flip and publication IDs deliberately change independently of producer work.
    {
        constexpr uint64_t a = 0x3486001000ull, b = 0x3486002000ull,
                           c = 0x3486003000ull, d = 0x3486004000ull;
        auto render_target = [&](uint64_t id) -> test::PersistentColorTargetImage* {
            test::BackendDraw draw;
            draw.vs.assign(kTriVertSpv, kTriVertSpv + sizeof(kTriVertSpv) / 4);
            draw.fs.assign(kTriFragSpv, kTriFragSpv + sizeof(kTriFragSpv) / 4);
            test::BackendColorTarget target;
            target.persistent_id = id;
            target.load_existing = false;
            target.readback = false;
            test::render_draws_rgba({draw}, W, H, nullptr, nullptr, false, &target,
                                    nullptr, nullptr, nullptr, nullptr, true, nullptr, false);
            return test::find_persistent_color_target(id, W, H, VK_FORMAT_R8G8B8A8_UNORM);
        };
        auto publish_target = [&](test::PersistentColorTargetImage* target, uint64_t flip,
                                  bool acquire = true) -> frontend::GpuScanoutFrame {
            frontend::GpuScanoutFrame frame;
            if (!target || !target->image) return frame;
            const bool published = frontend::present_blit_publish(
                target->image, target->layout, VK_FORMAT_R8G8B8A8_UNORM, W, H, flip,
                test::persistent_color_producer_source(*target));
            CHECK(published, "lineage: retained renderer image published through GPU scanout");
            if (published && acquire)
                CHECK(frontend::present_blit_acquire(frame), "lineage: acquired exact scanout slot");
            return frame;
        };
        frontend::ProducerLineageCounters deliveries;
        auto* ta = render_target(a);
        CHECK(ta && test::persistent_color_producer_source(*ta).known(),
              "lineage: real renderer pass has a completed producer");
        const auto first_a = ta ? test::persistent_color_producer_source(*ta)
                                : frontend::ProducerSource{};
        auto fa = publish_target(ta, 4001);
        CHECK(fa.producer.completed == first_a.completed && fa.producer.image_registration ==
                  first_a.image_registration,
              "lineage: acquired slot carries its exact source image registration and producer");
        if (fa.valid()) {
            CHECK(deliveries.presentation(true, fa.producer) == frontend::ProducerDelivery::New,
                  "lineage: first successful present delivers new completed work");
            frontend::present_blit_release(fa.slot);
        }
        // Deliberately supply a front identity that cannot be live. The copy still yields usable
        // pixels and a source-flip token, but its completed producer must be revoked *after* the
        // fence and *before* acquisition. Removing that post-copy check turns this arm green-to-red.
        VideoOutBufferSnapshot stale_front;
        stale_front.address = a;
        stale_front.generation = UINT64_MAX;
        stale_front.source_flip_seq = 4901;
        stale_front.buffer_index = 0;
        if (ta && ta->image) {
            CHECK(frontend::present_blit_publish(
                      ta->image, ta->layout, VK_FORMAT_R8G8B8A8_UNORM, W, H, 4901,
                      first_a, &stale_front),
                  "lineage: stale-front probe still publishes copied pixels");
            frontend::GpuScanoutFrame stale_frame;
            CHECK(frontend::present_blit_acquire(stale_frame) && stale_frame.frame_seq == 4901 &&
                  !stale_frame.producer.known(),
                  "lineage: changed VideoOut front revokes producer proof before lease acquisition");
            if (stale_frame.valid()) frontend::present_blit_release(stale_frame.slot);
        }
        ta = render_target(a);
        const auto second_a = ta ? test::persistent_color_producer_source(*ta)
                                 : frontend::ProducerSource{};
        CHECK(second_a.known() && second_a.image_registration == first_a.image_registration &&
                  second_a.completed != first_a.completed,
              "lineage: identical rendering on the same image creates a new completed version");
        fa = publish_target(ta, 4002);
        if (fa.valid()) {
            CHECK(deliveries.presentation(true, fa.producer) == frontend::ProducerDelivery::New,
                  "lineage: identical pixels from new work are new");
            frontend::present_blit_release(fa.slot);
        }
        auto* tb = render_target(b);
        const auto source_b = tb ? test::persistent_color_producer_source(*tb)
                                  : frontend::ProducerSource{};
        auto fb = publish_target(tb, 4003);
        if (fb.valid()) {
            CHECK(deliveries.presentation(true, fb.producer) == frontend::ProducerDelivery::New,
                  "lineage: second output buffer delivers its own completed work");
            frontend::present_blit_release(fb.slot);
        }
        fa = publish_target(ta, 4004);
        if (fa.valid()) {
            CHECK(deliveries.presentation(true, fa.producer) == frontend::ProducerDelivery::Repeat,
                  "lineage: newer guest flip of stale A repeats despite intervening B");
            frontend::present_blit_release(fa.slot);
        }
        fb = publish_target(tb, 4005);
        if (fb.valid()) {
            CHECK(deliveries.presentation(true, fb.producer) == frontend::ProducerDelivery::Repeat,
                  "lineage: alternating stale B repeats despite intervening A");
            frontend::present_blit_release(fb.slot);
        }
        std::string copy_error;
        CHECK(test::copy_persistent_color_target(a, c, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                                                 copy_error),
              "lineage: pure GPU resolve copy completed");
        auto* tc = test::find_persistent_color_target(c, W, H, VK_FORMAT_R8G8B8A8_UNORM);
        CHECK(tc && tc->registration != second_a.image_registration &&
                  tc->completed_producer == second_a.completed,
              "lineage: representation copy retains producer across a new allocation");
        auto fc = publish_target(tc, 4006);
        if (fc.valid()) {
            CHECK(deliveries.presentation(true, fc.producer) == frontend::ProducerDelivery::Repeat,
                  "lineage: pure copy does not manufacture a new producer");
            frontend::present_blit_release(fc.slot);
        }
        auto* td = render_target(d);
        const auto before_supersede = frontend::producer_lineage_counters().snapshot();
        publish_target(td, 4007, false); // publication D is replaced before acquisition
        fa = publish_target(ta, 4008);
        const auto after_supersede = frontend::producer_lineage_counters().snapshot();
        CHECK(after_supersede.producer_publications ==
                  before_supersede.producer_publications + 2,
              "lineage: both completed scanout publications are counted");
        CHECK(fa.valid() && fa.producer.completed == second_a.completed,
              "lineage: supersession acquires A's exact metadata, never D's");
        if (fa.valid()) {
            CHECK(deliveries.presentation(true, fa.producer) == frontend::ProducerDelivery::Repeat,
                  "lineage: superseded D was not delivered");
            frontend::present_blit_release(fa.slot);
        }
        auto fd = publish_target(td, 4009);
        const auto before_failed = deliveries.snapshot();
        CHECK(!deliveries.presentation(false, fd.producer).has_value() &&
                  deliveries.snapshot().producer_delivered_new == before_failed.producer_delivered_new,
              "lineage: failed or skipped presentation delivers nothing");
        if (fd.valid()) frontend::present_blit_release(fd.slot);
        fd = publish_target(td, 4010);
        if (fd.valid()) {
            CHECK(deliveries.presentation(true, fd.producer) == frontend::ProducerDelivery::New,
                  "lineage: previously failed presentation remains undelivered until success");
            frontend::present_blit_release(fd.slot);
        }
        CHECK(deliveries.presentation(true, {}) == frontend::ProducerDelivery::Unknown,
              "lineage: CPU fallback has unknown producer provenance");
        CHECK(source_b.known(), "lineage: second output source remained known");
        test::invalidate_persistent_color_target(a);
        CHECK(ta && !test::persistent_color_producer_source(*ta).known(),
              "lineage: invalidation clears a retained image's completed producer");
        ta = render_target(a);
        CHECK(ta && test::persistent_color_producer_source(*ta).known() &&
                  ta->completed_producer != second_a.completed,
              "lineage: new work after invalidation gets a distinct producer");
        if (ta) {
            // A CPU upload into a retained image followed by a partial draw cannot inherit the
            // earlier renderer version, even when the allocation and guest target stay the same.
            std::vector<uint8_t> seed(bytes, 0x6b);
            test::BackendDraw partial;
            partial.vs.assign(kTriVertSpv, kTriVertSpv + sizeof(kTriVertSpv) / 4);
            partial.fs.assign(kTriFragSpv, kTriFragSpv + sizeof(kTriFragSpv) / 4);
            test::BackendColorTarget seeded_target;
            seeded_target.persistent_id = a;
            seeded_target.load_existing = true;
            seeded_target.readback = false;
            test::render_draws_rgba({partial}, W, H, seed.data(), nullptr, false,
                                    &seeded_target, nullptr, nullptr, nullptr, nullptr, true,
                                    nullptr, false);
            CHECK(!test::persistent_color_producer_source(*ta).known(),
                  "lineage: CPU-seeded partial draw cannot inherit old completed work");
        }
        if (ta) {
            const auto ticket = test::begin_color_producer_write(*ta);
            test::BackendSubmissionBatch discarded;
            test::queue_color_producer_write(discarded, *ta, ticket, true);
            discarded.discard();
            CHECK(!test::persistent_color_producer_source(*ta).known(),
                  "lineage: discarded submission cannot commit recorded work");
            // Simulate a cache registration reset on the same address and VkImage handle. The old
            // ticket must not authorize a replacement image, even if a mutation value repeats.
            test::PersistentColorTargetImage replacement = *ta;
            replacement.registration = frontend::next_producer_identity();
            replacement.mutation = ticket.mutation;
            replacement.completed_producer = {};
            test::complete_color_producer_write(replacement, ticket, true);
            CHECK(!replacement.completed_producer.known(),
                  "lineage: allocation registration prevents address/handle ABA");
            const auto earlier = test::begin_color_producer_write(replacement);
            const auto later = test::begin_color_producer_write(replacement);
            test::complete_color_producer_write(replacement, earlier, true);
            CHECK(!replacement.completed_producer.known(),
                  "lineage: older completion cannot label a later pending write");
            test::complete_color_producer_write(replacement, later, true);
            CHECK(replacement.completed_producer.known(),
                  "lineage: latest matching mutation can commit after completion");
            // Exercise the batch's unproven-completion branch without submitting a doomed GPU
            // command. It poisons this fixture's backend only after every rendering assertion.
            const auto unproven_ticket = test::begin_color_producer_write(*ta);
            test::BackendSubmissionBatch unproven;
            test::queue_color_producer_write(unproven, *ta, unproven_ticket, true);
            unproven.abandon_pending_resources();
            CHECK(!test::persistent_color_producer_source(*ta).known() &&
                      test::backend_has_unproven_submission(),
                  "lineage: unproven submission cannot advertise a new completed version");
        }
        frontend::ProducerDeliveryHistory bounded_history;
        const auto first_old = frontend::CompletedProducer{1, 1};
        CHECK(bounded_history.classify(first_old) == frontend::ProducerDelivery::New,
              "lineage: first delivered producer enters bounded history");
        for (uint64_t work = 2; work <= frontend::ProducerDeliveryHistory::kCapacity + 1;
             ++work)
            bounded_history.classify({1, work});
        CHECK(bounded_history.classify(first_old) == frontend::ProducerDelivery::Unknown,
              "lineage: evicted history never mislabels an old version fresh");
    }

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
