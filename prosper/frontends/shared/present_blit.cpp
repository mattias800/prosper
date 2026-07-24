// present_blit.cpp — see present_blit.hpp (#1270).
#include "present_blit.hpp"
#include "present_blit_policy.hpp"
#include "render_runner.h"            // render_vk_ctx()
#include "gpu/gpu_execute.hpp"        // shared_present_submit_mutex / shared_present_active
#include <mutex>
#include <cstdio>
#include <cstring>

namespace prosper::frontend {
namespace {

constexpr int kSlots = 3;   // 1 in-flight (consumer) + 1 published (untaken) + >=1 free
constexpr VkFormat kScanoutFormat = VK_FORMAT_R8G8B8A8_UNORM;

enum class SlotState { Free, Published, InFlight };

struct Slot {
    VkImage        image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    SlotState      state = SlotState::Free;
    uint32_t       w = 0, h = 0;            // the size of THIS slot's current image (per-slot, #1270)
    uint64_t       frame_seq = 0;
};

struct PresentBlitState {
    std::mutex      mx;                     // guards the slot bookkeeping below
    bool            init_tried = false;
    bool            ok = false;             // device + pool + fence created
    VkDevice        dev = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkQueue         queue = VK_NULL_HANDLE;
    uint32_t        qfi = UINT32_MAX;
    VkCommandPool   pool = VK_NULL_HANDLE;
    VkFence         blit_fence = VK_NULL_HANDLE;   // publish waits its own blit on this (single producer)
    Slot            slots[kSlots];
    int             latest = -1;            // slot index of the newest published frame
    bool            latest_taken = false;   // has the consumer acquired `latest`
};

PresentBlitState& S() { static PresentBlitState s; return s; }

uint32_t find_mem_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

// #1270: an optimal-tiling image can only be blit source/dest if the driver advertises the feature for
// that format. On any conformant driver RGBA8/RGBA16F/BGRA8 all support BLIT_SRC/BLIT_DST, so this is
// belt-and-suspenders: an unsupported combination declines to the CPU present path instead of producing a
// validation error / garbage. vkGetPhysicalDeviceFormatProperties is a cheap driver-table lookup.
bool blit_supported(VkPhysicalDevice phys, VkFormat src_format) {
    auto has = [&](VkFormat f, VkFormatFeatureFlags bit) {
        VkFormatProperties fp{}; vkGetPhysicalDeviceFormatProperties(phys, f, &fp);
        return (fp.optimalTilingFeatures & bit) == bit;
    };
    return has(src_format, VK_FORMAT_FEATURE_BLIT_SRC_BIT) &&
           has(kScanoutFormat, VK_FORMAT_FEATURE_BLIT_DST_BIT);
}

void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
                   VkAccessFlags src_access, VkAccessFlags dst_access,
                   VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from; b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask = src_access; b.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

void destroy_slot_image(PresentBlitState& s, Slot& sl) {
    if (sl.image)  vkDestroyImage(s.dev, sl.image, nullptr);
    if (sl.memory) vkFreeMemory(s.dev, sl.memory, nullptr);
    sl.image = VK_NULL_HANDLE; sl.memory = VK_NULL_HANDLE; sl.w = sl.h = 0;
}

// Ensure THIS slot owns an image of exactly (w,h). Caller holds mx AND the slot is Free -- a Free slot's
// last GPU use (its publish blit, fence-waited; or the consumer's read, released only after its present
// fence) is provably complete, so destroying/recreating its image needs no device wait and can never
// touch an image the consumer still references (#1270 Finding 1). Returns false on allocation failure.
bool ensure_slot_image(PresentBlitState& s, Slot& sl, uint32_t w, uint32_t h) {
    if (sl.image && sl.w == w && sl.h == h) return true;
    destroy_slot_image(s, sl);
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = kScanoutFormat;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(s.dev, &ici, nullptr, &sl.image) != VK_SUCCESS) { sl.image = VK_NULL_HANDLE; return false; }
    VkMemoryRequirements req{}; vkGetImageMemoryRequirements(s.dev, sl.image, &req);
    uint32_t mt = find_mem_type(s.phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == UINT32_MAX) { destroy_slot_image(s, sl); return false; }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size; mai.memoryTypeIndex = mt;
    if (vkAllocateMemory(s.dev, &mai, nullptr, &sl.memory) != VK_SUCCESS) { destroy_slot_image(s, sl); return false; }
    if (vkBindImageMemory(s.dev, sl.image, sl.memory, 0) != VK_SUCCESS) { destroy_slot_image(s, sl); return false; }
    sl.w = w; sl.h = h;
    return true;
}

// One-time device/pool/fence/command-buffer setup. Caller holds mx.
bool ensure_init(PresentBlitState& s) {
    if (s.init_tried) return s.ok;
    s.init_tried = true;
    const prosper::test::RenderVkCtx& ctx = prosper::test::render_vk_ctx();
    if (!ctx.ok) return false;   // no render device -> GPU present declines -> caller keeps CPU path
    s.dev = ctx.dev; s.phys = ctx.phys; s.queue = ctx.queue; s.qfi = ctx.qfi;

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = s.qfi;
    if (vkCreateCommandPool(s.dev, &pci, nullptr, &s.pool) != VK_SUCCESS) return false;

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(s.dev, &fci, nullptr, &s.blit_fence) != VK_SUCCESS) return false;

    for (auto& sl : s.slots) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = s.pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(s.dev, &ai, &sl.cmd) != VK_SUCCESS) return false;
    }
    s.ok = true;
    return true;
}

int pick_free_slot(PresentBlitState& s) {
    for (int i = 0; i < kSlots; i++) if (s.slots[i].state == SlotState::Free) return i;
    return -1;
}

} // namespace

bool present_blit_publish(VkImage src, VkImageLayout src_layout, VkFormat src_format,
                          uint32_t w, uint32_t h, uint64_t frame_seq) {
    if (!src || !w || !h) return false;
    PresentBlitState& s = S();
    std::lock_guard<std::mutex> lk(s.mx);
    if (!ensure_init(s)) return false;
    if (!blit_supported(s.phys, src_format)) return false;   // decline -> caller keeps CPU present path

    const int slot = pick_free_slot(s);
    if (slot < 0) return false;   // consumer is behind; drop this frame's publish (it keeps the last one)
    Slot& sl = s.slots[slot];
    // Size the (Free) slot's image to this frame. A display-size change (or dynamic resolution) recreates
    // ONLY this Free slot's image -- never an in-flight one -- so it cannot free an image the consumer
    // still holds, and needs no device-wide wait (#1270 Finding 1).
    if (!ensure_slot_image(s, sl, w, h)) return false;

    VkCommandBuffer cb = sl.cmd;
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) return false;

    // src (front-buffer image) -> TRANSFER_SRC; scanout slot -> TRANSFER_DST (contents discarded).
    image_barrier(cb, src, src_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    image_barrier(cb, sl.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  0, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {(int32_t)w, (int32_t)h, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {(int32_t)w, (int32_t)h, 1};
    vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   sl.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

    // scanout slot -> TRANSFER_SRC (consumer reads it); restore src to its original layout.
    image_barrier(cb, sl.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    image_barrier(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, src_layout,
                  VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    if (vkEndCommandBuffer(cb) != VK_SUCCESS) return false;

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;

    vkResetFences(s.dev, 1, &s.blit_fence);
    VkResult sr;
    {
        // Serialize the host submit CALL against prosper-app's present submits on the shared queue.
        std::unique_lock<std::mutex> qlk(prosper::gpu::shared_present_submit_mutex(), std::defer_lock);
        if (prosper::gpu::shared_present_active()) qlk.lock();
        sr = vkQueueSubmit(s.queue, 1, &si, s.blit_fence);
    }
    if (sr != VK_SUCCESS) return false;
    // Wait the blit's completion here (a fast GPU->GPU copy) so the slot is fully written before it can be
    // acquired -- this is what lets the consumer read it with no cross-thread semaphore, and is strictly
    // cheaper than the GPU->CPU readback + CPU->GPU re-upload it replaces. Does not touch the queue, so it
    // is outside the submit mutex.
    const VkResult wait_result =
        vkWaitForFences(s.dev, 1, &s.blit_fence, VK_TRUE, 5ull * 1000 * 1000 * 1000);
    if (!present_blit_wait_completed(wait_result)) {
        // The command buffer and destination slot may still be in flight. Fail the GPU-present path
        // closed for this session so neither can be published or reused; present_blit_reset() retains
        // the resources until its device-idle barrier and callers fall back to CPU presentation.
        std::fprintf(stderr, "[present-blit] fence wait failed (%d); disabling GPU present\n",
                     static_cast<int>(wait_result));
        s.ok = false;
        return false;
    }

    sl.frame_seq = frame_seq;
    sl.state = SlotState::Published;

    // A previously published-but-untaken frame is now stale; free its slot for reuse.
    if (s.latest >= 0 && s.latest != slot && !s.latest_taken &&
        s.slots[s.latest].state == SlotState::Published)
        s.slots[s.latest].state = SlotState::Free;
    s.latest = slot;
    s.latest_taken = false;
    return true;
}

bool present_blit_acquire(GpuScanoutFrame& out) {
    PresentBlitState& s = S();
    std::lock_guard<std::mutex> lk(s.mx);
    if (!s.ok || s.latest < 0 || s.latest_taken) return false;
    Slot& sl = s.slots[s.latest];
    if (sl.state != SlotState::Published) return false;
    sl.state = SlotState::InFlight;
    s.latest_taken = true;
    out.image = sl.image;
    out.width = sl.w; out.height = sl.h;
    out.frame_seq = sl.frame_seq;
    out.slot = s.latest;
    return true;
}

void present_blit_release(int slot) {
    if (slot < 0 || slot >= kSlots) return;
    PresentBlitState& s = S();
    std::lock_guard<std::mutex> lk(s.mx);
    Slot& sl = s.slots[slot];
    if (sl.state == SlotState::InFlight) sl.state = SlotState::Free;
}

void present_blit_reset() {
    PresentBlitState& s = S();
    std::lock_guard<std::mutex> lk(s.mx);
    if (s.dev) vkDeviceWaitIdle(s.dev);
    if (s.dev)
        for (auto& sl : s.slots) { destroy_slot_image(s, sl); sl.state = SlotState::Free; sl.cmd = VK_NULL_HANDLE; }
    if (s.dev) {
        if (s.blit_fence) { vkDestroyFence(s.dev, s.blit_fence, nullptr); s.blit_fence = VK_NULL_HANDLE; }
        if (s.pool) { vkDestroyCommandPool(s.dev, s.pool, nullptr); s.pool = VK_NULL_HANDLE; }
    }
    s.init_tried = false; s.ok = false;
    s.dev = VK_NULL_HANDLE; s.phys = VK_NULL_HANDLE; s.queue = VK_NULL_HANDLE; s.qfi = UINT32_MAX;
    s.latest = -1; s.latest_taken = false;
}

bool present_blit_ready() {
    PresentBlitState& s = S();
    std::lock_guard<std::mutex> lk(s.mx);
    return s.ok;
}

} // namespace prosper::frontend
