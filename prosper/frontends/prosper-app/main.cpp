// prosper-app (P0a) — the OS-integration frontend: an SDL3 window + Vulkan swapchain that presents
// the frame prosper's renderer hands to the present layer. See
// docs/FRONTEND_APP.md and issue #164.
//
// P0a scope (this file): the whole present half, decoupled from the guest boot. It pulls finished
// frames from prosper_core's present layer and blits them to a real swapchain,
// and handles SDL_QUIT/Esc via the shared stop request. run_entry does not consume that request yet,
// so a booted guest uses direct process exit after the present loop (issue #352). To verify the
// pipeline without a game dump the app can FEED the present layer a synthetic animated pattern
// (--test-pattern): frame -> present_write_frame -> shared frame lease -> swapchain, exactly the path
// a real guest frame takes. P0b wires the actual guest boot in front of this (the same present
// path, no changes here).
//
// Two Vulkan contexts by design (docs/FRONTEND_APP.md): the core keeps its headless render device;
// THIS is a separate presentation device, and frames cross as shared immutable CPU pixels.
#include "gpu/videoout_present.hpp"   // present_acquire_rendered_frame / present_write_frame
#include "host/lifecycle.hpp"          // prosper_request_stop / prosper_stop_requested
#include "host/boot_program.hpp"       // boot_program (shared guest-boot path, also used by boot_trace)
#include "host/exec_image.hpp"         // run_entry
#include "loader/linker.hpp"           // Program
#include "input/pad.hpp"               // keyboard -> libScePad (HostPadState / PadBackend)
#include "pad_overlay.hpp"              // keyboard pad 0 composed over the physical controller backend
#include "present_mode.hpp"             // explicit swapchain latency/vsync policy, pure regression seam
#include "window_controls.hpp"           // debounced app-window shortcuts, pure regression seam
#ifdef PROSPER_HAVE_LIVE_RENDERER
#include "live_renderer.hpp"           // shared DrawItem->Vulkan compositor (register_live_renderer)
#endif
#ifdef PROSPER_AUDIO_SDL3
#include "audio_sdl3.hpp"              // install_sdl3_audio_sink (route sceAudioOut to the host)
#endif
#ifdef PROSPER_PAD_SDL3
#include "pad_sdl3.hpp"                // install_sdl3_pad_backend (route a host controller to libScePad)
#endif
#ifdef PROSPER_HAVE_DIALOG_SDL3
#include "dialog_sdl3.hpp"             // install_sdl3_platform_ui (real SDL message boxes for dialogs)
#endif
#ifdef PROSPER_VIDEO_MF
#include "media_foundation_backend.hpp" // native Windows AvPlayer demux + hardware decode
#endif
#ifdef PROSPER_VIDEO_VAAPI
#include "vaapi_backend.hpp"            // native Linux FFmpeg demux + VA-API hardware decode
#endif

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>

using namespace prosper;

namespace {

bool set_environment(const char* name, const char* value) {
#ifdef _WIN32
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

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

    // macOS: SDL adds VK_KHR_portability_enumeration to its required extensions because it assumes
    // the Khronos loader. This build links MoltenVK DIRECTLY (no loader), where MoltenVK is the sole
    // driver — no enumeration is needed and MoltenVK rejects the extension with
    // VK_ERROR_EXTENSION_NOT_PRESENT. Strip it (and never set the enumerate flag). The device-level
    // VK_KHR_portability_subset (below) is the piece that actually matters. If a future build routes
    // through the loader, keep SDL's extension and add the ENUMERATE_PORTABILITY flag instead.
#ifdef __APPLE__
    exts.erase(std::remove_if(exts.begin(), exts.end(),
                   [](const char* e){ return std::strcmp(e, "VK_KHR_portability_enumeration") == 0; }),
               exts.end());
#endif
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
    std::vector<const char*> devExts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
#ifdef __APPLE__
    // The Vulkan spec requires enabling VK_KHR_portability_subset on any device that advertises it
    // (all MoltenVK devices do), or vkCreateDevice fails with VK_ERROR_EXTENSION_NOT_PRESENT.
    { uint32_t n = 0; vkEnumerateDeviceExtensionProperties(vk.phys, nullptr, &n, nullptr);
      std::vector<VkExtensionProperties> dp(n);
      vkEnumerateDeviceExtensionProperties(vk.phys, nullptr, &n, dp.data());
      for (auto& e : dp) if (!strcmp(e.extensionName, "VK_KHR_portability_subset")) {
          devExts.push_back("VK_KHR_portability_subset"); break; } }
#endif
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    di.enabledExtensionCount = (uint32_t)devExts.size(); di.ppEnabledExtensionNames = devExts.data();
    VKCHECK(vkCreateDevice(vk.phys, &di, nullptr, &vk.device), "vkCreateDevice");
    vkGetDeviceQueue(vk.device, vk.qfamily, 0, &vk.queue);

    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(vk.phys, &pp);
    fprintf(stderr, "[app] Vulkan device: %s\n", pp.deviceName);
    return true;
}

bool create_swapchain(Vk& vk, uint32_t w, uint32_t h,
                      prosper::frontend::AppPresentMode requestedPresentMode) {
    VkSurfaceCapabilitiesKHR caps; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.phys, vk.surface, &caps);
    vk.scExtent = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : VkExtent2D{w, h};
    if (vk.scExtent.width == 0 || vk.scExtent.height == 0) return false;   // minimized

    uint32_t fmtN = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(vk.phys, vk.surface, &fmtN, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtN); vkGetPhysicalDeviceSurfaceFormatsKHR(vk.phys, vk.surface, &fmtN, fmts.data());
    vk.scFormat = fmts[0].format; VkColorSpaceKHR cs = fmts[0].colorSpace;
    for (auto& f : fmts) if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { vk.scFormat = f.format; cs = f.colorSpace; break; }

    uint32_t imgCount = caps.minImageCount + 1;
    if (caps.maxImageCount && imgCount > caps.maxImageCount) imgCount = caps.maxImageCount;

    uint32_t modeN = 0;
    VKCHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(vk.phys, vk.surface, &modeN, nullptr),
            "vkGetPhysicalDeviceSurfacePresentModesKHR(count)");
    std::vector<VkPresentModeKHR> modes(modeN);
    VKCHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(vk.phys, vk.surface, &modeN, modes.data()),
            "vkGetPhysicalDeviceSurfacePresentModesKHR(list)");
    const bool hasMailbox = std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end();
    const bool hasImmediate = std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.end();
    const auto selectedPresentMode = prosper::frontend::select_present_mode(
        requestedPresentMode, hasMailbox, hasImmediate);
    VkPresentModeKHR vkPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (selectedPresentMode.mode == prosper::frontend::AppPresentMode::mailbox)
        vkPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    else if (selectedPresentMode.mode == prosper::frontend::AppPresentMode::immediate)
        vkPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    if (selectedPresentMode.fell_back) {
        fprintf(stderr, "[app] requested present mode %s is unsupported; falling back to fifo\n",
                prosper::frontend::present_mode_name(requestedPresentMode));
    }
    fprintf(stderr, "[app] present mode: %s\n",
            prosper::frontend::present_mode_name(selectedPresentMode.mode));

    VkSwapchainCreateInfoKHR si{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    si.surface = vk.surface; si.minImageCount = imgCount; si.imageFormat = vk.scFormat;
    si.imageColorSpace = cs; si.imageExtent = vk.scExtent; si.imageArrayLayers = 1;
    si.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;   // we blit into the swapchain image
    si.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    si.preTransform = caps.currentTransform;
    si.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    si.presentMode = vkPresentMode;
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

// ---- keyboard -> virtual DualSense (pad 0) ----------------------------------------------------
// A controller over WSL passthrough is flaky, so the app maps the keyboard onto the same
// HostPadState the SDL gamepad backend fills. The event loop updates g_keyboard_pad on the main
// thread; the guest's scePadReadState reads the composed keyboard/physical state on a guest thread.
prosper::frontend::KeyboardPadOverlay g_keyboard_pad;

// Snapshot the current keyboard into the overlay. Call from the thread that pumps SDL events.
void poll_keyboard(const bool* keyboard, bool enter_maps_to_options) {
    using namespace prosper::input;
    auto d = [&](SDL_Scancode s){ return keyboard[s]; };
    bool up    = d(SDL_SCANCODE_UP)   || d(SDL_SCANCODE_W);
    bool down  = d(SDL_SCANCODE_DOWN) || d(SDL_SCANCODE_S);
    bool left  = d(SDL_SCANCODE_LEFT) || d(SDL_SCANCODE_A);
    bool right = d(SDL_SCANCODE_RIGHT)|| d(SDL_SCANCODE_D);
    uint32_t b = 0;
    if (up)    b |= SCE_PAD_BUTTON_UP;
    if (down)  b |= SCE_PAD_BUTTON_DOWN;
    if (left)  b |= SCE_PAD_BUTTON_LEFT;
    if (right) b |= SCE_PAD_BUTTON_RIGHT;
    if (d(SDL_SCANCODE_SPACE) || d(SDL_SCANCODE_J)) b |= SCE_PAD_BUTTON_CROSS;    // jump
    if (d(SDL_SCANCODE_K))     b |= SCE_PAD_BUTTON_SQUARE;                        // attack
    if (d(SDL_SCANCODE_L))     b |= SCE_PAD_BUTTON_CIRCLE;
    if (d(SDL_SCANCODE_I))     b |= SCE_PAD_BUTTON_TRIANGLE;
    if (d(SDL_SCANCODE_U))     b |= SCE_PAD_BUTTON_L1;
    if (d(SDL_SCANCODE_O))     b |= SCE_PAD_BUTTON_R1;
    if (d(SDL_SCANCODE_Y))     b |= SCE_PAD_BUTTON_L2;
    if (d(SDL_SCANCODE_H))     b |= SCE_PAD_BUTTON_R2;
    const bool enter_down = d(SDL_SCANCODE_RETURN) || d(SDL_SCANCODE_RETURN2) ||
                            d(SDL_SCANCODE_KP_ENTER);
    if (enter_down && enter_maps_to_options)
        b |= SCE_PAD_BUTTON_OPTIONS;   // menu/start; Alt+Enter belongs to the host window
    HostPadState st;
    st.buttons = b;
    st.left_x  = left ? 0x00 : (right ? 0xff : 0x80);   // also drive the left stick, for stick-only titles
    st.left_y  = up   ? 0x00 : (down  ? 0xff : 0x80);
    st.l2 = d(SDL_SCANCODE_Y) ? 255 : 0;
    st.r2 = d(SDL_SCANCODE_H) ? 255 : 0;
    st.connected = true;
    g_keyboard_pad.set_keyboard_state(st);
}

} // namespace

int main(int argc, char** argv) {
    bool testPattern = false; int exitAfter = 0; uint32_t winW = 1280, winH = 720;
    prosper::frontend::AppPresentMode requestedPresentMode = prosper::frontend::AppPresentMode::fifo;
    std::string dump;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--test-pattern") testPattern = true;
        else if (a == "--frames" && i + 1 < argc) exitAfter = atoi(argv[++i]);   // present N frames then exit (CI/smoke)
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];                // boot the game at this app0 dir
        else if (a == "--present-mode") {
            if (i + 1 >= argc ||
                !prosper::frontend::parse_present_mode(argv[++i], requestedPresentMode)) {
                fprintf(stderr, "prosper-app: --present-mode requires fifo, mailbox, or immediate\n");
                return 2;
            }
        }
        else if (a == "--record" && i + 1 < argc) {
            if (!set_environment("PROSPER_PAD_RECORD", argv[++i])) {
                fprintf(stderr, "prosper-app: failed to set PROSPER_PAD_RECORD\n");
                return 2;
            }
        }
        else if (a[0] != '-' && dump.empty()) dump = a;                          // positional dump path
    }

    // Boot the game (unless test-pattern): register the shared live renderer so the guest's GPU
    // submits composite to frames on the present layer, then the shared boot_program path sets the
    // guest up and it runs on its own thread while this thread owns the window + present. Reaching
    // the frame loop needs PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct in the environment
    // (same as boot_trace).
    std::thread guestThread;
    if (!testPattern && !dump.empty()) {
#ifdef PROSPER_HAVE_LIVE_RENDERER
        prosper::frontend::register_live_renderer(
            getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".",
            getenv("PROSPER_APP_DUMP_FRAMES") != nullptr);
#else
        fprintf(stderr, "[app] built without the live renderer; the window will stay blank.\n");
#endif
        static Program prog; std::string err;
        // Install host frontends (audio out, controller in) at the same point boot_trace does —
        // right after the built-in HLE is registered, before the guest runs. Built in only when the
        // corresponding SDL3 frontend is enabled; a window app wants both on by default.
        auto install_backends = []{
#ifdef PROSPER_VIDEO_MF
            if (!getenv("PROSPER_APP_DISABLE_VIDEO")) {
                if (prosper::video::install_media_foundation_backend())
                    fprintf(stderr, "[app] Media Foundation video backend installed.\n");
                else
                    fprintf(stderr, "[app] Media Foundation video backend unavailable.\n");
            } else {
                fprintf(stderr, "[app] native video backend disabled.\n");
            }
#endif
#ifdef PROSPER_VIDEO_VAAPI
            if (!getenv("PROSPER_APP_DISABLE_VIDEO")) {
                if (prosper::video::install_vaapi_backend())
                    fprintf(stderr, "[app] FFmpeg/VA-API video backend installed.\n");
                else
                    fprintf(stderr, "[app] FFmpeg/VA-API video backend unavailable.\n");
            } else {
                fprintf(stderr, "[app] native video backend disabled.\n");
            }
#endif
#ifdef PROSPER_AUDIO_SDL3
            if (!getenv("PROSPER_APP_DISABLE_AUDIO")) {
                prosper::install_sdl3_audio_sink();
            } else {
                fprintf(stderr, "[app] SDL audio backend disabled; using the realtime silent sink.\n");
            }
#endif
#ifdef PROSPER_PAD_SDL3
            if (!getenv("PROSPER_APP_DISABLE_PAD")) {
                if (prosper::install_sdl3_pad_backend()) {
                    g_keyboard_pad.set_fallback(prosper::input::pad_backend());
                    fprintf(stderr, "[app] controller backend installed.\n");
                }
            } else {
                fprintf(stderr, "[app] SDL controller backend disabled; keyboard and scripted input remain available.\n");
            }
#endif
#ifdef PROSPER_HAVE_DIALOG_SDL3
            if (!getenv("PROSPER_APP_DISABLE_DIALOG")) {
                prosper::install_sdl3_platform_ui();   // real SDL message boxes for MsgDialog/ErrorDialog (#347)
                fprintf(stderr, "[app] dialog backend installed.\n");
            } else {
                fprintf(stderr, "[app] SDL dialog backend disabled; using headless auto-dismiss.\n");
            }
#endif
        };
        if (!boot_program(dump, prog, &err, install_backends)) { fprintf(stderr, "[app] boot failed: %s\n", err.c_str()); return 1; }
        guestThread = std::thread([]{
            const BootResult result = run_entry(prog.imgs[0]);
            fprintf(stderr,
                    "[app] guest thread ended: kind=%d detail=%s rip=0x%llx addr=0x%llx "
                    "rax=0x%llx rbx=0x%llx rdi=0x%llx rsi=0x%llx rdx=0x%llx "
                    "rbp=0x%llx rsp=0x%llx\n",
                    result.kind, result.detail.c_str(),
                    static_cast<unsigned long long>(result.fault_rip),
                    static_cast<unsigned long long>(result.fault_addr),
                    static_cast<unsigned long long>(result.rax),
                    static_cast<unsigned long long>(result.rbx),
                    static_cast<unsigned long long>(result.rdi),
                    static_cast<unsigned long long>(result.rsi),
                    static_cast<unsigned long long>(result.rdx),
                    static_cast<unsigned long long>(result.rbp),
                    static_cast<unsigned long long>(result.rsp));
            if (result.kind != 0)
                dump_guest_exception_trace();
            for (uint64_t address : result.backtrace)
                fprintf(stderr, "[app] guest backtrace: 0x%llx\n",
                        static_cast<unsigned long long>(address));
        });   // runs the guest frame loop
        fprintf(stderr, "[app] guest booted; presenting its frames.\n");
    } else if (!testPattern) {
        fprintf(stderr, "[app] no dump given and not --test-pattern; waiting for external present frames.\n");
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) { fprintf(stderr, "[app] SDL_Init: %s\n", SDL_GetError()); return 1; }
#ifdef __APPLE__
    // There is no system Vulkan loader on macOS; point SDL at MoltenVK so SDL_Vulkan_* uses the same
    // driver this binary links. PROSPER_VULKAN_LIB overrides the path; default resolves via the
    // executable's rpath (CMake links MoltenVK with an rpath, so the dylib sits beside/near the app).
    if (!SDL_Vulkan_LoadLibrary(getenv("PROSPER_VULKAN_LIB") ? getenv("PROSPER_VULKAN_LIB") : "libMoltenVK.dylib")) {
        fprintf(stderr, "[app] SDL_Vulkan_LoadLibrary(MoltenVK): %s\n", SDL_GetError());
        fprintf(stderr, "[app] set PROSPER_VULKAN_LIB=/path/to/libMoltenVK.dylib\n");
        return 1;
    }
#endif
    // Title: "prosper — <app0 basename>" for a booted game, else a plain label.
    std::string title = "prosper";
    if (!dump.empty()) { auto sl = dump.find_last_of("/\\"); title += " — " + (sl == std::string::npos ? dump : dump.substr(sl + 1)); }
    else if (testPattern) title += " — test pattern";
    SDL_Window* win = SDL_CreateWindow(title.c_str(), (int)winW, (int)winH, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!win) { fprintf(stderr, "[app] SDL_CreateWindow: %s\n", SDL_GetError()); return 1; }

    Vk vk;
    if (!create_instance(vk, win) || !pick_device(vk)) return 1;
    // Initial swapchain sized to the window; recreated on resize / out-of-date.
    { int dw = 0, dh = 0; SDL_GetWindowSizeInPixels(win, &dw, &dh);
      if (!create_swapchain(vk, (uint32_t)dw, (uint32_t)dh, requestedPresentMode)) return 1; }

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

    // Keyboard controls augment SDL pad 0; the fallback keeps physical pads and their analog state.
    prosper::input::pad_set_backend(&g_keyboard_pad);
    fprintf(stderr, "[app] keyboard: WASD/Arrows=move  J/Space=Cross(jump)  K=Square(attack)  L=Circle  "
                    "I=Triangle  U/O=L1/R1  Y/H=L2/R2  Enter=Options  "
                    "F11/Alt+Enter=fullscreen  Esc=quit\n");

    const bool frameTrace = getenv("PROSPER_APP_FRAME_TRACE") != nullptr;
    const char* stallDumpEnv = getenv("PROSPER_APP_STALL_DUMP_MS");
    const int stallDumpMs = stallDumpEnv ? std::max(0, atoi(stallDumpEnv)) : 0;
    const char* timedDumpEnv = getenv("PROSPER_APP_GUEST_DUMP_MS");
    const int timedDumpMs = timedDumpEnv ? std::max(0, atoi(timedDumpEnv)) : 0;
    const char* timedDumpIntervalEnv = getenv("PROSPER_APP_GUEST_DUMP_INTERVAL_MS");
    const int timedDumpIntervalMs = timedDumpIntervalEnv ?
        std::max(0, atoi(timedDumpIntervalEnv)) : 0;
    const char* timedDumpPath = getenv("PROSPER_APP_GUEST_DUMP_PATH");
    const char* timedDumpPthreadEnv = getenv("PROSPER_APP_GUEST_DUMP_PTHREAD");
    const uint64_t timedDumpPthread = timedDumpPthreadEnv ?
        strtoull(timedDumpPthreadEnv, nullptr, 0) : 0;
    const auto loopStarted = std::chrono::steady_clock::now();
    auto lastFrameProgress = loopStarted;
    auto nextTimedDump = loopStarted + std::chrono::milliseconds(timedDumpMs);
    uint64_t shown = 0, lastFrameSeq = ~0ull, patFrame = 0;
    unsigned timedDumpCount = 0;
    bool timedDumpPending = timedDumpMs > 0;
    bool running = true;
    bool swapchainDirty = false;
    const SDL_WindowID appWindowId = SDL_GetWindowID(win);
    prosper::frontend::AppWindowControls windowControls;
    const SDL_WindowFlags initialWindowFlags = SDL_GetWindowFlags(win);
    bool fullscreenRequested = (initialWindowFlags & SDL_WINDOW_FULLSCREEN) != 0;
    windowControls.set_app_focus((initialWindowFlags & SDL_WINDOW_INPUT_FOCUS) != 0);
    while (running && !prosper_stop_requested()) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_EVENT_QUIT) running = false;
            else if (ev.type == SDL_EVENT_KEY_DOWN || ev.type == SDL_EVENT_KEY_UP) {
                prosper::frontend::AppWindowKey key{};
                key.app_window = ev.key.windowID == appWindowId;
                key.pressed = ev.key.down;
                key.repeat = ev.key.repeat;
                key.escape = ev.key.key == SDLK_ESCAPE;
                key.f11 = ev.key.key == SDLK_F11;
                key.enter = ev.key.key == SDLK_RETURN || ev.key.key == SDLK_KP_ENTER;
                key.alt = (ev.key.mod & SDL_KMOD_ALT) != 0;
                switch (windowControls.handle_key(key)) {
                case prosper::frontend::AppWindowCommand::quit:
                    running = false;
                    break;
                case prosper::frontend::AppWindowCommand::toggle_fullscreen: {
                    // SDL may apply fullscreen requests asynchronously. Toggle the last accepted
                    // target instead of reading a flag that can still describe the old state.
                    const bool targetFullscreen = !fullscreenRequested;
                    if (!SDL_SetWindowFullscreen(win, targetFullscreen)) {
                        fprintf(stderr, "[app] fullscreen toggle failed: %s\n", SDL_GetError());
                    } else {
                        fullscreenRequested = targetFullscreen;
                        swapchainDirty = true;
                        fprintf(stderr, "[app] fullscreen %s requested\n",
                                targetFullscreen ? "on" : "off");
                    }
                    break;
                }
                case prosper::frontend::AppWindowCommand::none:
                    break;
                }
            } else if (ev.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED &&
                       ev.window.windowID == appWindowId) {
                swapchainDirty = true;
            } else if (ev.type == SDL_EVENT_WINDOW_FOCUS_LOST &&
                       ev.window.windowID == appWindowId) {
                windowControls.set_app_focus(false);
            } else if (ev.type == SDL_EVENT_WINDOW_FOCUS_GAINED &&
                       ev.window.windowID == appWindowId) {
                windowControls.set_app_focus(true);
            } else if (ev.type == SDL_EVENT_WINDOW_ENTER_FULLSCREEN &&
                       ev.window.windowID == appWindowId) {
                fullscreenRequested = true;
            } else if (ev.type == SDL_EVENT_WINDOW_LEAVE_FULLSCREEN &&
                       ev.window.windowID == appWindowId) {
                fullscreenRequested = false;
            }
        }
        if (!running) break;

        // Snapshot input before any minimized-window early exit. This clears released guest
        // buttons even while swapchain recreation has to wait for a non-zero pixel extent.
        const bool* keyboard = SDL_GetKeyboardState(nullptr);
        const bool enterDown = keyboard[SDL_SCANCODE_RETURN] ||
                               keyboard[SDL_SCANCODE_RETURN2] ||
                               keyboard[SDL_SCANCODE_KP_ENTER];
        windowControls.reconcile_enter(enterDown);
        poll_keyboard(keyboard, windowControls.guest_options_allowed());
#ifdef PROSPER_HAVE_DIALOG_SDL3
        prosper::sdl_platform_ui_pump();   // run a pending ImeDialog text-entry modal on this (main) thread
#endif

        if (swapchainDirty) {
            int dw = 0, dh = 0;
            SDL_GetWindowSizeInPixels(win, &dw, &dh);
            if (dw <= 0 || dh <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;   // minimized or between fullscreen modes; retry after the next event
            }
            vkDeviceWaitIdle(vk.device);
            if (vk.swapchain) vkDestroySwapchainKHR(vk.device, vk.swapchain, nullptr);
            vk.swapchain = VK_NULL_HANDLE;
            if (!create_swapchain(vk, static_cast<uint32_t>(dw), static_cast<uint32_t>(dh),
                                  requestedPresentMode)) {
                fprintf(stderr, "[app] could not recreate the swapchain after a window-size change\n");
                running = false;
                break;
            }
            swapchainDirty = false;
        }
        const auto loopNow = std::chrono::steady_clock::now();
        if (timedDumpPending && loopNow >= nextTimedDump) {
            ++timedDumpCount;
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                loopNow - loopStarted).count();
            fprintf(stderr, "[app] timed guest-state dump #%u after %lld ms at frame %llu\n",
                    timedDumpCount, (long long)elapsed, (unsigned long long)shown);
            if (timedDumpCount == 1) dump_guest_exception_trace();
            dump_guest_thread_trace(timedDumpPath, timedDumpPthread);
            if (timedDumpIntervalMs > 0)
                nextTimedDump = loopNow + std::chrono::milliseconds(timedDumpIntervalMs);
            else
                timedDumpPending = false;
        }

        static const uint32_t kPatW = 1920, kPatH = 1080;
        if (testPattern) feed_test_pattern(kPatW, kPatH, patFrame++);

        // Present the latest finished frame from the core's present layer.
        // The frame dimensions: from the guest's registered VideoOut display in normal use; in
        // test-pattern mode there is no guest, so use the dims we feed (present_width/height report
        // the VideoOut registry, which is empty without a guest). Either way, readback needs a
        // buffer sized to the frame it holds — guard zero dims so we never present a 0-extent image.
        // Render completion and guest flips are separate clocks: the command stream can flip before
        // the renderer publishes its CPU frame. Key this loop to the completed-frame sequence so a
        // late renderer publication is not missed or marked handled while only the previous frame exists.
        bool newFrame = testPattern || (gpu::present_frame_seq() != lastFrameSeq);
        gpu::PresentFrameLease frame;
        if (newFrame && gpu::present_acquire_rendered_frame(frame)) {
            uint32_t w = frame.width;
            uint32_t h = frame.height;
            if (w == 0 || h == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
            if (frame.rgba && frame.rgba->size() == (size_t)w * h * 4) {
                if (!present_frame(vk, frame.rgba->data(), w, h)) {
                    // Out-of-date/suboptimal: share the resize/fullscreen recreation path next loop.
                    swapchainDirty = true;
                } else {
                    lastFrameSeq = frame.frame_seq; shown++;
                    lastFrameProgress = std::chrono::steady_clock::now();
                    // Periodic present-rate log (every 60 presented frames).
                    static auto t0 = std::chrono::steady_clock::now(); static uint64_t mark = 0;
                    if (shown - mark >= 60) {
                        auto now = std::chrono::steady_clock::now();
                        double s = std::chrono::duration<double>(now - t0).count();
                        if (frameTrace) {
                            size_t nonzeroRgbBytes = 0;
                            for (size_t i = 0; i + 3 < frame.rgba->size(); i += 4) {
                                nonzeroRgbBytes += (*frame.rgba)[i] != 0;
                                nonzeroRgbBytes += (*frame.rgba)[i + 1] != 0;
                                nonzeroRgbBytes += (*frame.rgba)[i + 2] != 0;
                            }
                            fprintf(stderr,
                                    "[app] %.1f fps (%llu frames) render_seq=%llu flips=%llu "
                                    "nonzero_rgb_bytes=%zu/%zu\n",
                                    (shown - mark) / (s > 0 ? s : 1),
                                    (unsigned long long)shown,
                                    (unsigned long long)frame.frame_seq,
                                    (unsigned long long)gpu::present_count(), nonzeroRgbBytes,
                                    (size_t)w * h * 3);
                        } else {
                            fprintf(stderr, "[app] %.1f fps (%llu frames)\n",
                                    (shown - mark) / (s > 0 ? s : 1),
                                    (unsigned long long)shown);
                        }
                        t0 = now; mark = shown;
                    }
                    if (exitAfter && (int)shown >= exitAfter) running = false;
                }
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));   // no new frame — don't spin
        }
        if (stallDumpMs > 0 &&
            std::chrono::steady_clock::now() - lastFrameProgress >=
                std::chrono::milliseconds(stallDumpMs)) {
            fprintf(stderr, "[app] no presented-frame progress for %d ms at frame %llu\n",
                    stallDumpMs, (unsigned long long)shown);
            dump_guest_exception_trace();
            lastFrameProgress = std::chrono::steady_clock::now();
        }
    }

    prosper_request_stop();   // signal the guest run-loop to wind down at its next boundary
    fprintf(stderr, "[app] shutting down after %llu presented frame(s)\n", (unsigned long long)shown);
    const int exitCode = (exitAfter && (int)shown < exitAfter) ? 1 : 0;

    // run_entry does not yet observe the frontend stop flag, so a booted guest cannot be joined.
    // Returning from main after detaching it is unsafe: C++ static teardown destroys HLE/renderer
    // state while guest threads still use it (a short --frames run reliably ended in 0xC0000005 on
    // Windows). Until the flip-boundary cooperative stop is implemented, terminate directly and
    // let the OS reclaim process state without running destructors under the live guest.
    if (guestThread.joinable()) {
        guestThread.detach();
        fflush(nullptr);
        std::_Exit(exitCode);
    }

    vkDeviceWaitIdle(vk.device);
    SDL_DestroyWindow(win); SDL_Quit();
    return exitCode;
}
