// prosper-app (P0a) — the OS-integration frontend: an SDL3 window + Vulkan swapchain that presents
// the frame prosper's renderer hands to the present layer (present_readback). See
// docs/FRONTEND_APP.md and issue #164.
//
// P0a scope (this file): the whole present half, decoupled from the guest boot. It pulls finished
// frames from prosper_core's present layer (present_readback) and blits them to a real swapchain,
// and closes on SDL_QUIT/Esc via the cooperative stop signal (prosper_request_stop). To verify the
// pipeline without a game dump it can also FEED the present layer a synthetic animated pattern
// (--test-pattern): frame -> present_write_frame -> present_readback -> swapchain, exactly the path
// a real guest frame takes. P0b wires the actual guest boot in front of this (the same present
// path, no changes here).
//
// Two Vulkan contexts by design (docs/FRONTEND_APP.md): the core keeps its headless render device;
// THIS is a separate presentation device, and frames cross as CPU pixels via present_readback.
#include "gpu/videoout_present.hpp"   // present_readback / present_write_frame / present_width/height/count
#include "host/lifecycle.hpp"          // prosper_request_stop / prosper_stop_requested

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <thread>

using namespace prosper;

namespace {

// ---- tiny Vulkan error helper -----------------------------------------------------------------
#define VKCHECK(x, msg) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "[app] Vulkan error %d at %s\n", (int)_r, msg); return false; } } while (0)

struct Vk {
    VkInstance       instance = VK_NULL_HANDLE;
    VkSurfaceKHR     surface  = VK_NULL_HANDLE;
    VkPhysicalDevice phys     = VK_NULL_HANDLE;
    VkDevice         device   = VK_NULL_HANDLE;
    uint32_t         qfamily  = 0;
    VkQueue          queue    = VK_NULL_HANDLE;

    VkSwapchainKHR   swapchain = VK_NULL_HANDLE;
    VkFormat         scFormat  = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D       scExtent  {};
    std::vector<VkImage> scImages;

    // A host-visible staging buffer + a device-local staging image; each frame we copy the guest
    // RGBA into the buffer, copy buffer->image, then BLIT (scaling) image->swapchain image.
    VkBuffer        stageBuf   = VK_NULL_HANDLE;
    VkDeviceMemory  stageMem   = VK_NULL_HANDLE;
    void*           stageMapped = nullptr;
    VkDeviceSize    stageCap   = 0;
    VkImage         stageImg   = VK_NULL_HANDLE;
    VkDeviceMemory  stageImgMem = VK_NULL_HANDLE;
    uint32_t        stageW = 0, stageH = 0;

    VkCommandPool   cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd     = VK_NULL_HANDLE;
    VkSemaphore     acquireSem = VK_NULL_HANDLE, presentSem = VK_NULL_HANDLE;
    VkFence         inFlight   = VK_NULL_HANDLE;
};

uint32_t find_mem(VkPhysicalDevice p, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties m; vkGetPhysicalDeviceMemoryProperties(p, &m);
    for (uint32_t i = 0; i < m.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) && (m.memoryTypes[i].propertyFlags & props) == props) return i;
    return UINT32_MAX;
}

bool create_instance(Vk& vk, SDL_Window* win) {
    uint32_t extCount = 0;
    const char* const* sdlExts = SDL_Vulkan_GetInstanceExtensions(&extCount);
    if (!sdlExts) { fprintf(stderr, "[app] SDL_Vulkan_GetInstanceExtensions: %s\n", SDL_GetError()); return false; }
    std::vector<const char*> exts(sdlExts, sdlExts + extCount);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "prosper-app"; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();
    VKCHECK(vkCreateInstance(&ci, nullptr, &vk.instance), "vkCreateInstance");

    if (!SDL_Vulkan_CreateSurface(win, vk.instance, nullptr, &vk.surface)) {
        fprintf(stderr, "[app] SDL_Vulkan_CreateSurface: %s\n", SDL_GetError()); return false;
    }
    return true;
}

bool pick_device(Vk& vk) {
    uint32_t n = 0; vkEnumeratePhysicalDevices(vk.instance, &n, nullptr);
    if (!n) { fprintf(stderr, "[app] no Vulkan device\n"); return false; }
    std::vector<VkPhysicalDevice> devs(n); vkEnumeratePhysicalDevices(vk.instance, &n, devs.data());
    for (auto d : devs) {
        uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> q(qn); vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, q.data());
        for (uint32_t i = 0; i < qn; i++) {
            VkBool32 present = VK_FALSE; vkGetPhysicalDeviceSurfaceSupportKHR(d, i, vk.surface, &present);
            if ((q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) { vk.phys = d; vk.qfamily = i; break; }
        }
        if (vk.phys) break;
    }
    if (!vk.phys) { fprintf(stderr, "[app] no graphics+present queue family\n"); return false; }

    float pr = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = vk.qfamily; qi.queueCount = 1; qi.pQueuePriorities = &pr;
    const char* devExts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    di.enabledExtensionCount = 1; di.ppEnabledExtensionNames = devExts;
    VKCHECK(vkCreateDevice(vk.phys, &di, nullptr, &vk.device), "vkCreateDevice");
    vkGetDeviceQueue(vk.device, vk.qfamily, 0, &vk.queue);

    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(vk.phys, &pp);
    fprintf(stderr, "[app] Vulkan device: %s\n", pp.deviceName);
    return true;
}

bool create_swapchain(Vk& vk, uint32_t w, uint32_t h) {
    VkSurfaceCapabilitiesKHR caps; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.phys, vk.surface, &caps);
    vk.scExtent = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : VkExtent2D{w, h};
    if (vk.scExtent.width == 0 || vk.scExtent.height == 0) return false;   // minimized

    uint32_t fmtN = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(vk.phys, vk.surface, &fmtN, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtN); vkGetPhysicalDeviceSurfaceFormatsKHR(vk.phys, vk.surface, &fmtN, fmts.data());
    vk.scFormat = fmts[0].format; VkColorSpaceKHR cs = fmts[0].colorSpace;
    for (auto& f : fmts) if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { vk.scFormat = f.format; cs = f.colorSpace; break; }

    uint32_t imgCount = caps.minImageCount + 1;
    if (caps.maxImageCount && imgCount > caps.maxImageCount) imgCount = caps.maxImageCount;
    VkSwapchainCreateInfoKHR si{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    si.surface = vk.surface; si.minImageCount = imgCount; si.imageFormat = vk.scFormat;
    si.imageColorSpace = cs; si.imageExtent = vk.scExtent; si.imageArrayLayers = 1;
    si.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;   // we blit into the swapchain image
    si.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    si.preTransform = caps.currentTransform;
    si.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    si.presentMode = VK_PRESENT_MODE_FIFO_KHR;   // vsync
    si.clipped = VK_TRUE;
    VKCHECK(vkCreateSwapchainKHR(vk.device, &si, nullptr, &vk.swapchain), "vkCreateSwapchainKHR");
    uint32_t n = 0; vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &n, nullptr);
    vk.scImages.resize(n); vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &n, vk.scImages.data());
    return true;
}

// (Re)create the staging buffer + image sized to the guest frame (w*h RGBA).
bool ensure_stage(Vk& vk, uint32_t w, uint32_t h) {
    if (vk.stageW == w && vk.stageH == h && vk.stageBuf) return true;
    if (vk.stageBuf)   { vkDestroyBuffer(vk.device, vk.stageBuf, nullptr); vkFreeMemory(vk.device, vk.stageMem, nullptr); }
    if (vk.stageImg)   { vkDestroyImage(vk.device, vk.stageImg, nullptr);  vkFreeMemory(vk.device, vk.stageImgMem, nullptr); }
    vk.stageBuf = VK_NULL_HANDLE; vk.stageImg = VK_NULL_HANDLE; vk.stageMapped = nullptr;
    vk.stageW = w; vk.stageH = h; vk.stageCap = (VkDeviceSize)w * h * 4;

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = vk.stageCap; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKCHECK(vkCreateBuffer(vk.device, &bi, nullptr, &vk.stageBuf), "stage buffer");
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(vk.device, vk.stageBuf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = find_mem(vk.phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VKCHECK(vkAllocateMemory(vk.device, &ai, nullptr, &vk.stageMem), "stage buffer mem");
    vkBindBufferMemory(vk.device, vk.stageBuf, vk.stageMem, 0);
    vkMapMemory(vk.device, vk.stageMem, 0, vk.stageCap, 0, &vk.stageMapped);

    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = {w, h, 1}; ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VKCHECK(vkCreateImage(vk.device, &ii, nullptr, &vk.stageImg), "stage image");
    vkGetImageMemoryRequirements(vk.device, vk.stageImg, &mr);
    ai.allocationSize = mr.size; ai.memoryTypeIndex = find_mem(vk.phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VKCHECK(vkAllocateMemory(vk.device, &ai, nullptr, &vk.stageImgMem), "stage image mem");
    vkBindImageMemory(vk.device, vk.stageImg, vk.stageImgMem, 0);
    return true;
}

void barrier(VkCommandBuffer c, VkImage img, VkImageLayout from, VkImageLayout to,
             VkAccessFlags srcA, VkAccessFlags dstA, VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from; b.newLayout = to; b.image = img;
    b.srcAccessMask = srcA; b.dstAccessMask = dstA;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(c, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// Present one guest RGBA frame (w*h, 4 bytes/pixel) to the window, scaling to the swapchain extent.
bool present_frame(Vk& vk, const uint8_t* rgba, uint32_t w, uint32_t h) {
    if (!ensure_stage(vk, w, h)) return false;
    memcpy(vk.stageMapped, rgba, (size_t)w * h * 4);

    vkWaitForFences(vk.device, 1, &vk.inFlight, VK_TRUE, UINT64_MAX);
    uint32_t imgIndex = 0;
    VkResult acq = vkAcquireNextImageKHR(vk.device, vk.swapchain, UINT64_MAX, vk.acquireSem, VK_NULL_HANDLE, &imgIndex);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) return false;   // caller recreates
    vkResetFences(vk.device, 1, &vk.inFlight);
    vkResetCommandBuffer(vk.cmd, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(vk.cmd, &bi);

    // staging buffer -> staging image (TRANSFER_DST), then image -> swapchain image (blit, scaled).
    barrier(vk.cmd, vk.stageImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy bic{}; bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bic.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(vk.cmd, vk.stageBuf, vk.stageImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
    barrier(vk.cmd, vk.stageImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    barrier(vk.cmd, vk.scImages[imgIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; blit.srcOffsets[1] = {(int32_t)w, (int32_t)h, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {(int32_t)vk.scExtent.width, (int32_t)vk.scExtent.height, 1};
    vkCmdBlitImage(vk.cmd, vk.stageImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   vk.scImages[imgIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    barrier(vk.cmd, vk.scImages[imgIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    vkEndCommandBuffer(vk.cmd);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo su{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    su.waitSemaphoreCount = 1; su.pWaitSemaphores = &vk.acquireSem; su.pWaitDstStageMask = &waitStage;
    su.commandBufferCount = 1; su.pCommandBuffers = &vk.cmd;
    su.signalSemaphoreCount = 1; su.pSignalSemaphores = &vk.presentSem;
    vkQueueSubmit(vk.queue, 1, &su, vk.inFlight);

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &vk.presentSem;
    pi.swapchainCount = 1; pi.pSwapchains = &vk.swapchain; pi.pImageIndices = &imgIndex;
    VkResult pr = vkQueuePresentKHR(vk.queue, &pi);
    return pr == VK_SUCCESS;
}

// Synthetic animated frame (no guest): a moving gradient fed through the REAL present layer, so the
// window + swapchain + readback path is exercised end-to-end without a game dump.
void feed_test_pattern(uint32_t w, uint32_t h, uint64_t frame) {
    static std::vector<uint8_t> px; px.resize((size_t)w * h * 4);
    uint8_t t = (uint8_t)(frame * 2);
    for (uint32_t y = 0; y < h; y++) for (uint32_t x = 0; x < w; x++) {
        uint8_t* p = &px[((size_t)y * w + x) * 4];
        p[0] = (uint8_t)(x * 255 / w + t);   // R
        p[1] = (uint8_t)(y * 255 / h);       // G
        p[2] = t;                            // B
        p[3] = 255;
    }
    gpu::present_write_frame(px.data(), w, h);
}

} // namespace

int main(int argc, char** argv) {
    bool testPattern = false; int exitAfter = 0; uint32_t winW = 1280, winH = 720;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--test-pattern") testPattern = true;
        else if (a == "--frames" && i + 1 < argc) exitAfter = atoi(argv[++i]);   // present N frames then exit 0 (CI/smoke)
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) { fprintf(stderr, "[app] SDL_Init: %s\n", SDL_GetError()); return 1; }
    SDL_Window* win = SDL_CreateWindow("prosper", (int)winW, (int)winH, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!win) { fprintf(stderr, "[app] SDL_CreateWindow: %s\n", SDL_GetError()); return 1; }

    Vk vk;
    if (!create_instance(vk, win) || !pick_device(vk)) return 1;
    // Initial swapchain sized to the window; recreated on resize / out-of-date.
    { int dw = 0, dh = 0; SDL_GetWindowSizeInPixels(win, &dw, &dh);
      if (!create_swapchain(vk, (uint32_t)dw, (uint32_t)dh)) return 1; }

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi.queueFamilyIndex = vk.qfamily;
    vkCreateCommandPool(vk.device, &cpi, nullptr, &vk.cmdPool);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = vk.cmdPool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(vk.device, &cai, &vk.cmd);
    VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(vk.device, &semi, nullptr, &vk.acquireSem);
    vkCreateSemaphore(vk.device, &semi, nullptr, &vk.presentSem);
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(vk.device, &fci, nullptr, &vk.inFlight);

    fprintf(stderr, "[app] window up (%s). Close the window or press Esc to quit.\n",
            testPattern ? "test-pattern" : "waiting for guest frames");

    uint64_t shown = 0, lastCount = ~0ull, patFrame = 0;
    bool running = true;
    while (running && !prosper_stop_requested()) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_EVENT_QUIT) running = false;
            else if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.key == SDLK_ESCAPE) running = false;
        }
        if (!running) break;

        static const uint32_t kPatW = 1920, kPatH = 1080;
        if (testPattern) feed_test_pattern(kPatW, kPatH, patFrame++);

        // Present the latest finished frame from the core's present layer.
        // The frame dimensions: from the guest's registered VideoOut display in normal use; in
        // test-pattern mode there is no guest, so use the dims we feed (present_width/height report
        // the VideoOut registry, which is empty without a guest). Either way, readback needs a
        // buffer sized to the frame it holds — guard zero dims so we never present a 0-extent image.
        // In normal use a frame is "new" when the guest flips (present_count advances); test-pattern
        // has no flips, so treat every iteration as new.
        bool newFrame = testPattern || (gpu::present_count() != lastCount);
        if (gpu::present_has_frame() && newFrame) {
            uint32_t w = testPattern ? kPatW : gpu::present_width();
            uint32_t h = testPattern ? kPatH : gpu::present_height();
            if (w == 0 || h == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
            static std::vector<uint8_t> buf;
            buf.resize((size_t)w * h * 4);
            size_t n = gpu::present_readback(buf.data(), buf.size());
            if (n == buf.size()) {
                if (!present_frame(vk, buf.data(), w, h)) {
                    // out-of-date / resize: recreate the swapchain and retry next iteration.
                    vkDeviceWaitIdle(vk.device);
                    vkDestroySwapchainKHR(vk.device, vk.swapchain, nullptr); vk.swapchain = VK_NULL_HANDLE;
                    int dw = 0, dh = 0; SDL_GetWindowSizeInPixels(win, &dw, &dh);
                    create_swapchain(vk, (uint32_t)dw, (uint32_t)dh);
                } else {
                    lastCount = gpu::present_count(); shown++;
                    if (exitAfter && (int)shown >= exitAfter) running = false;
                }
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));   // no new frame — don't spin
        }
    }

    prosper_request_stop();   // tell any guest run-loop (P0b) to wind down
    fprintf(stderr, "[app] shutting down after %llu presented frame(s)\n", (unsigned long long)shown);
    vkDeviceWaitIdle(vk.device);
    // (P0a leaves teardown to process exit; the resources are process-lifetime. P0b joins the guest thread here.)
    SDL_DestroyWindow(win); SDL_Quit();
    return (exitAfter && (int)shown < exitAfter) ? 1 : 0;
}
