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
// Real game boots normally adopt the renderer's Vulkan device and pass its front image directly to
// the swapchain. Test-pattern boots, an explicit override, or failed adoption retain the original
// two-device path, where frames cross as shared immutable CPU pixels.
#include "gpu/videoout_present.hpp"   // present_acquire_rendered_frame / present_write_frame
#include "gpu/gpu_execute.hpp"         // shared_vulkan_context / gpu-present activation (#1270)
#include "gpu/gpu_capture.hpp"         // request_interactive_gpu_capture (F9 frame grab)
#include "gpu/gpu_timeline.hpp"        // request_interactive_capture_bundle (F9 whole-frame grab)
#include "capture_schedule.hpp"        // exact host-frame screenshot calibration trigger
#include "present_blit.hpp"           // GPU scanout handoff: acquire/release the renderer's front image
#include "host/lifecycle.hpp"          // frontend-owned stop/pause gates
#include "host/boot_program.hpp"       // boot_program (shared guest-boot path, also used by boot_trace)
#include "host/exec_image.hpp"         // run_entry
#include "loader/linker.hpp"           // Program
#include "input/pad.hpp"               // keyboard -> libScePad (HostPadState / PadBackend)
#include "pad_overlay.hpp"              // keyboard pad 0 composed over the physical controller backend
#include "hle/ime_input.hpp"           // #1093: forward host keyboard keys to the guest IME path
#include "present_mode.hpp"             // explicit swapchain latency/vsync policy, pure regression seam
#include "present_policy.hpp"           // bounded-acquire present classification (#1182), pure seam
#include "window_controls.hpp"           // debounced app-window shortcuts, pure regression seam
#ifdef PROSPER_HAVE_LIVE_RENDERER
#include "live_renderer.hpp"           // shared DrawItem->Vulkan compositor (register_live_renderer)
#endif
#ifdef PROSPER_AUDIO_SDL3
#include "audio_sdl3.hpp"              // install_sdl3_audio_sink (route sceAudioOut to the host)
#endif
#ifdef PROSPER_AUDIO_FFMPEG
#include "ajm_ffmpeg.hpp"              // install AJM MP3 decoder before guest instance creation
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
#include <mutex>

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

    // Present unification (#1270): set when the app adopted the renderer's shared device and presents by
    // GPU-blitting its front-buffer image (no CPU round-trip). queue_shared means the present queue aliases
    // the render queue, so present submits serialize through gpu::shared_present_submit_mutex().
    bool            gpu_present = false;
    bool            queue_shared = false;
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
    bi.size = vk.stageCap;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
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

// A failed queue submit leaves inFlight unsignaled and the acquired-image semaphore unusable for a
// second acquire. Replace the whole sync trio before asking the main loop to rebuild the swapchain;
// otherwise the next frame would wait forever on a fence that no submit can signal. The abandoned
// handles are deliberately retained until process teardown: the acquire signal operation may still
// be pending, so destroying them here would itself violate Vulkan lifetime rules. This path is rare
// and bounded by a fatal device/driver error.
bool replace_present_sync(Vk& vk) {
    VkSemaphore acquire = VK_NULL_HANDLE;
    VkSemaphore present = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateSemaphore(vk.device, &semi, nullptr, &acquire) != VK_SUCCESS ||
        vkCreateSemaphore(vk.device, &semi, nullptr, &present) != VK_SUCCESS ||
        vkCreateFence(vk.device, &fci, nullptr, &fence) != VK_SUCCESS) {
        if (fence) vkDestroyFence(vk.device, fence, nullptr);
        if (present) vkDestroySemaphore(vk.device, present, nullptr);
        if (acquire) vkDestroySemaphore(vk.device, acquire, nullptr);
        return false;
    }
    vk.acquireSem = acquire;
    vk.presentSem = present;
    vk.inFlight = fence;
    return true;
}

prosper::frontend::PresentAttempt recover_submit_failure(Vk& vk, VkResult result) {
    if (prosper::frontend::classify_submit_failure(result) ==
        prosper::frontend::PresentAttempt::failed) {
        fprintf(stderr, "[app] vkQueueSubmit failed (%d); device lost, stopping\n",
                static_cast<int>(result));
        return prosper::frontend::PresentAttempt::failed;
    }
    fprintf(stderr, "[app] vkQueueSubmit failed (%d); replacing present synchronization\n",
            static_cast<int>(result));
    if (replace_present_sync(vk)) return prosper::frontend::PresentAttempt::out_of_date;
    fprintf(stderr, "[app] could not replace present synchronization; stopping\n");
    return prosper::frontend::PresentAttempt::failed;
}

// Interactive frame grab (F9): write a small BGR bottom-up BMP of a presented RGBA frame next to the
// .prgcap so the user gets a visible "this is the frame I captured" alongside the replayable capsule.
// A local writer (the shared dump_bmp lives in the Vulkan test harness, not linkable here) — the .prgcap
// oracle remains the authoritative pixels; this is the convenience screenshot.
static bool write_frame_bmp(const std::string& path, const uint8_t* rgba, uint32_t w, uint32_t h) {
    if (!rgba || !w || !h) return false;
    const uint32_t row = w * 3, pad = (4 - (row & 3)) & 3, stride = row + pad;
    const uint32_t pixels = stride * h, size = 54 + pixels;
    std::vector<uint8_t> f(size, 0);
    auto put16 = [&](uint32_t o, uint16_t v) { f[o] = v & 0xff; f[o + 1] = v >> 8; };
    auto put32 = [&](uint32_t o, uint32_t v) { for (int i = 0; i < 4; i++) f[o + i] = (v >> (8 * i)) & 0xff; };
    f[0] = 'B'; f[1] = 'M'; put32(2, size); put32(10, 54);
    put32(14, 40); put32(18, w); put32(22, h); put16(26, 1); put16(28, 24); put32(34, pixels);
    for (uint32_t y = 0; y < h; y++) {
        uint8_t* dst = f.data() + 54 + (size_t)(h - 1 - y) * stride;   // BMP rows are bottom-up
        const uint8_t* src = rgba + (size_t)y * w * 4;
        for (uint32_t x = 0; x < w; x++) { dst[x * 3] = src[x * 4 + 2]; dst[x * 3 + 1] = src[x * 4 + 1]; dst[x * 3 + 2] = src[x * 4]; }
    }
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) return false;
    const bool ok = fwrite(f.data(), 1, f.size(), fp) == f.size();
    fclose(fp);
    return ok;
}

// Present one guest RGBA frame (w*h, 4 bytes/pixel) to the window, scaling to the swapchain extent.
using prosper::frontend::PresentAttempt;

prosper::frontend::PresentAttempt present_frame(Vk& vk, const uint8_t* rgba, uint32_t w, uint32_t h) {
    if (!ensure_stage(vk, w, h)) return PresentAttempt::out_of_date;
    memcpy(vk.stageMapped, rgba, (size_t)w * h * 4);

    const VkResult waitResult = vkWaitForFences(
        vk.device, 1, &vk.inFlight, VK_TRUE, UINT64_MAX);
    if (waitResult != VK_SUCCESS) {
        fprintf(stderr, "[app] present fence wait failed (%d); stopping\n",
                static_cast<int>(waitResult));
        return PresentAttempt::failed;
    }
    uint32_t imgIndex = 0;
    // Bound the acquire (#1182): an occluded/minimized window releases no swapchain image, and an
    // infinite wait here would block the app main thread — freezing visible output, stalling SDL
    // event/close handling, and (on the shared physical GPU) potentially the guest. A timeout is a
    // benign SKIP: return without touching the swapchain and retry next loop. The guest keeps running
    // on its own device regardless.
    constexpr uint64_t kAcquireTimeoutNs = 100ull * 1000 * 1000;   // 100 ms — never hit while visible
    VkResult acq = vkAcquireNextImageKHR(vk.device, vk.swapchain, kAcquireTimeoutNs, vk.acquireSem,
                                         VK_NULL_HANDLE, &imgIndex);
    switch (prosper::frontend::classify_acquire(acq)) {
    case prosper::frontend::AcquireAction::skip:     return PresentAttempt::skipped;
    case prosper::frontend::AcquireAction::recreate: return PresentAttempt::out_of_date;
    case prosper::frontend::AcquireAction::fail:     return PresentAttempt::failed;
    case prosper::frontend::AcquireAction::proceed:  break;
    }
    // Load-bearing ordering: vkResetFences MUST stay after the skip/recreate early-returns above. A skip
    // leaves inFlight signaled (from the last real present) so the next frame's vkWaitForFences returns
    // immediately; hoisting this reset above the acquire would leave inFlight unsignaled on a skip with no
    // paired submit to re-signal it, and the next vkWaitForFences would hang. The reset is always paired
    // with the vkQueueSubmit(..., inFlight) below.
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

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &vk.presentSem;
    pi.swapchainCount = 1; pi.pSwapchains = &vk.swapchain; pi.pImageIndices = &imgIndex;
    VkResult submitResult;
    VkResult pr = VK_SUCCESS;
    {
        // #1270: when this CPU present runs on the renderer's shared queue (the GPU-present miss fallback),
        // serialize the submit + present CALLS against the renderer's submits. No-op on a private device.
        std::unique_lock<std::mutex> lk(gpu::shared_present_submit_mutex(), std::defer_lock);
        if (gpu::shared_present_active()) lk.lock();
        submitResult = vkQueueSubmit(vk.queue, 1, &su, vk.inFlight);
        if (submitResult == VK_SUCCESS) pr = vkQueuePresentKHR(vk.queue, &pi);
    }
    if (submitResult != VK_SUCCESS) return recover_submit_failure(vk, submitResult);
    return prosper::frontend::classify_present(pr);
}

// Present unification (#1270): try to present on the RENDERER's Vulkan device instead of a private one,
// so the app can GPU-blit the renderer's front-buffer image straight to the swapchain (no 4K CPU
// round-trip). Returns false at any unmet precondition, leaving the caller to create its own device (the
// unchanged CPU path). Called only when PROSPER_APP_GPU_PRESENT is set and a game is booting.
bool try_adopt_shared_present(Vk& vk, SDL_Window* win) {
    const gpu::SharedVulkanContext ctx = gpu::shared_vulkan_context();
    if (!ctx.valid() || !ctx.present_capable || !ctx.present_queue) {
        fprintf(stderr, "[app] GPU present: shared device is not present-capable; using own device\n");
        return false;
    }
    VkInstance inst = (VkInstance)ctx.instance;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(win, inst, nullptr, &surface)) {
        // The shared instance was created before SDL_Init and enabled every AVAILABLE platform surface
        // extension blind; if SDL needs one we did not get, fall back to a private device.
        fprintf(stderr, "[app] GPU present: surface on shared instance failed (%s); using own device\n",
                SDL_GetError());
        return false;
    }
    VkPhysicalDevice phys = (VkPhysicalDevice)ctx.physical;
    VkBool32 present = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(phys, ctx.queue_family, surface, &present);
    if (!present) {
        fprintf(stderr, "[app] GPU present: shared queue family cannot present here; using own device\n");
        vkDestroySurfaceKHR(inst, surface, nullptr);
        return false;
    }
    vk.instance = inst; vk.surface = surface; vk.phys = phys;
    vk.device = (VkDevice)ctx.device; vk.qfamily = ctx.queue_family;
    vk.queue = (VkQueue)ctx.present_queue;
    vk.gpu_present = true;
    vk.queue_shared = ctx.present_queue_shared;
    // The renderer must now publish the front-buffer image (and skip the CPU readback); if the present
    // queue aliases the render queue, both threads must serialize their submits.
    gpu::set_gpu_present_active(true);
    if (ctx.present_queue_shared) gpu::set_shared_present_active(true);
    fprintf(stderr, "[app] GPU present: adopted the renderer's device (%s present queue)\n",
            ctx.present_queue_shared ? "shared" : "dedicated");
    return true;
}

// Present one GPU scanout frame (the renderer's front-buffer image, already in TRANSFER_SRC_OPTIMAL) by
// blitting it straight to the swapchain -- no CPU pixels. `prevSlot` is the slot presented last frame; its
// GPU read is complete once inFlight signals below, so it is released then. Updates prevSlot to the slot
// now in flight and returns the acquire/present outcome.
prosper::frontend::PresentAttempt present_frame_gpu(Vk& vk, const prosper::frontend::GpuScanoutFrame& gf,
                                                    int& prevSlot, bool requestReadback,
                                                    bool& readbackReady) {
    readbackReady = false;
    const VkResult previousWait = vkWaitForFences(
        vk.device, 1, &vk.inFlight, VK_TRUE, UINT64_MAX);   // previous present's read complete
    if (previousWait != VK_SUCCESS) {
        prosper::frontend::present_blit_release(gf.slot);
        if (prevSlot >= 0) {
            prosper::frontend::present_blit_release(prevSlot);
            prevSlot = -1;
        }
        fprintf(stderr, "[app] gpu-present fence wait failed (%d); stopping\n",
                static_cast<int>(previousWait));
        return prosper::frontend::PresentAttempt::failed;
    }
    if (prevSlot >= 0) { prosper::frontend::present_blit_release(prevSlot); prevSlot = -1; }
    if (requestReadback && !ensure_stage(vk, gf.width, gf.height)) {
        prosper::frontend::present_blit_release(gf.slot);
        fprintf(stderr, "[grab] could not allocate the gpu-present readback buffer\n");
        return prosper::frontend::PresentAttempt::failed;
    }

    uint32_t imgIndex = 0;
    constexpr uint64_t kAcquireTimeoutNs = 100ull * 1000 * 1000;
    VkResult acq = vkAcquireNextImageKHR(vk.device, vk.swapchain, kAcquireTimeoutNs, vk.acquireSem,
                                         VK_NULL_HANDLE, &imgIndex);
    switch (prosper::frontend::classify_acquire(acq)) {
    case prosper::frontend::AcquireAction::skip:
        prosper::frontend::present_blit_release(gf.slot); return prosper::frontend::PresentAttempt::skipped;
    case prosper::frontend::AcquireAction::recreate:
        prosper::frontend::present_blit_release(gf.slot); return prosper::frontend::PresentAttempt::out_of_date;
    case prosper::frontend::AcquireAction::fail:
        prosper::frontend::present_blit_release(gf.slot); return prosper::frontend::PresentAttempt::failed;
    case prosper::frontend::AcquireAction::proceed: break;
    }
    vkResetFences(vk.device, 1, &vk.inFlight);
    vkResetCommandBuffer(vk.cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(vk.cmd, &bi);
    // gf.image is already TRANSFER_SRC_OPTIMAL (left there by present_blit); swapchain image
    // UNDEFINED -> TRANSFER_DST -> (scaled blit) -> PRESENT_SRC.
    barrier(vk.cmd, vk.scImages[imgIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {(int32_t)gf.width, (int32_t)gf.height, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {(int32_t)vk.scExtent.width, (int32_t)vk.scExtent.height, 1};
    vkCmdBlitImage(vk.cmd, gf.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   vk.scImages[imgIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    if (requestReadback) {
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {gf.width, gf.height, 1};
        vkCmdCopyImageToBuffer(vk.cmd, gf.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               vk.stageBuf, 1, &copy);
        VkBufferMemoryBarrier hostBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        hostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostBarrier.buffer = vk.stageBuf;
        hostBarrier.offset = 0;
        hostBarrier.size = vk.stageCap;
        vkCmdPipelineBarrier(vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &hostBarrier,
                             0, nullptr);
    }
    barrier(vk.cmd, vk.scImages[imgIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    vkEndCommandBuffer(vk.cmd);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo su{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    su.waitSemaphoreCount = 1; su.pWaitSemaphores = &vk.acquireSem; su.pWaitDstStageMask = &waitStage;
    su.commandBufferCount = 1; su.pCommandBuffers = &vk.cmd;
    su.signalSemaphoreCount = 1; su.pSignalSemaphores = &vk.presentSem;
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &vk.presentSem;
    pi.swapchainCount = 1; pi.pSwapchains = &vk.swapchain; pi.pImageIndices = &imgIndex;
    VkResult submitResult;
    VkResult pr = VK_SUCCESS;
    {
        // Serialize the present submit + present CALL against the renderer's submits on a shared queue.
        std::unique_lock<std::mutex> lk(gpu::shared_present_submit_mutex(), std::defer_lock);
        if (vk.queue_shared) lk.lock();
        submitResult = vkQueueSubmit(vk.queue, 1, &su, vk.inFlight);
        if (submitResult == VK_SUCCESS) pr = vkQueuePresentKHR(vk.queue, &pi);
    }
    if (submitResult != VK_SUCCESS) {
        prosper::frontend::present_blit_release(gf.slot);
        return recover_submit_failure(vk, submitResult);
    }
    prevSlot = gf.slot;
    if (requestReadback) {
        const VkResult waitResult = vkWaitForFences(vk.device, 1, &vk.inFlight, VK_TRUE, UINT64_MAX);
        if (waitResult != VK_SUCCESS) {
            fprintf(stderr, "[grab] gpu-present readback wait failed (%d)\n",
                    static_cast<int>(waitResult));
            return prosper::frontend::PresentAttempt::failed;
        }
        prosper::frontend::present_blit_release(prevSlot);
        prevSlot = -1;
        readbackReady = true;
    }
    return prosper::frontend::classify_present(pr);
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

// Read the human-readable game name from the dump's PS5 param.json so the window title shows e.g.
// "Bendy and the Ink Machine" instead of the PPSA content-id directory. PS5 param.json stores the name under
// localizedParameters.<lang>.titleName; we prefer the defaultLanguage's entry and fall back to the first
// titleName found. Dependency-free string scan (no JSON lib in-tree); returns "" if the file/field is absent.
static std::string read_game_title(const std::string& dump) {
    std::string path = dump + "/sce_sys/param.json";
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "";
    std::string s; char buf[8192]; size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    std::fclose(f);
    // Extract the JSON string value that follows the first "titleName" at or after `from`.
    auto title_after = [&](size_t from) -> std::string {
        size_t k = s.find("\"titleName\"", from);
        if (k == std::string::npos) return "";
        k = s.find(':', k); if (k == std::string::npos) return "";
        k = s.find('"', k); if (k == std::string::npos) return "";
        std::string out;
        for (size_t e = k + 1; e < s.size() && s[e] != '"'; ++e) {
            if (s[e] == '\\' && e + 1 < s.size()) { ++e; out += s[e]; }  // unescape \" \\ etc. (title is plain ASCII)
            else out += s[e];
        }
        return out;
    };
    // Prefer the default language's titleName: find defaultLanguage's value, then that language object's key.
    std::string title;
    size_t dl = s.find("\"defaultLanguage\"");
    if (dl != std::string::npos) {
        size_t c = s.find(':', dl);
        size_t q = (c == std::string::npos) ? std::string::npos : s.find('"', c);
        if (q != std::string::npos) {
            size_t qe = s.find('"', q + 1);
            if (qe != std::string::npos) {
                std::string lang = s.substr(q + 1, qe - q - 1);
                // The language OBJECT key ("en-US":{ ... }) appears before the defaultLanguage line, so search
                // from the top for the key and take the titleName inside its object.
                size_t lk = s.find("\"" + lang + "\"");
                if (lk != std::string::npos && lk < dl) title = title_after(lk);
            }
        }
    }
    if (title.empty()) title = title_after(0);
    return title;
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
        else if (a == "--record") {
            if (i + 1 >= argc) {
                fprintf(stderr, "prosper-app: --record requires a path\n");
                return 2;
            }
            if (!set_environment("PROSPER_PAD_RECORD", argv[++i])) {
                fprintf(stderr, "prosper-app: failed to set PROSPER_PAD_RECORD\n");
                return 2;
            }
        }
        else if (a == "--record-axis") {
            if (i + 1 >= argc) {
                fprintf(stderr, "prosper-app: --record-axis requires flip or pad-read\n");
                return 2;
            }
            const std::string axis = argv[++i];
            if (axis != "flip" && axis != "pad-read") {
                fprintf(stderr, "prosper-app: --record-axis requires flip or pad-read\n");
                return 2;
            }
            if (!set_environment("PROSPER_PAD_RECORD_AXIS", axis.c_str())) {
                fprintf(stderr, "prosper-app: failed to set PROSPER_PAD_RECORD_AXIS\n");
                return 2;
            }
        }
        else if (a == "--hdr") {
            // Advertise an HDR-capable display to the guest (sceVideoOut capability + output
            // status). Default is SDR — the mode most users' displays and captures expect, and
            // the path where titles apply their own tonemapping. Presentation itself is
            // unchanged (SDR swapchain): a title that commits to PQ output will look wrong —
            // this is an A/B knob, not an HDR presentation path.
            if (!set_environment("PROSPER_HDR", "1")) {
                fprintf(stderr, "prosper-app: failed to set PROSPER_HDR\n");
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
#ifdef PROSPER_AUDIO_FFMPEG
            if (prosper::ajm::install_ffmpeg_decoder_backend())
                fprintf(stderr, "[app] FFmpeg AJM audio decoder installed.\n");
            else
                fprintf(stderr, "[app] FFmpeg AJM audio decoder unavailable.\n");
#endif
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
    // Title: "prosper — <game name>" for a booted game (name from param.json; falls back to the app0 basename),
    // else a plain label.
    std::string title = "prosper";
    if (!dump.empty()) {
        std::string name = read_game_title(dump);
        if (name.empty()) { auto sl = dump.find_last_of("/\\"); name = (sl == std::string::npos ? dump : dump.substr(sl + 1)); }
        title += " — " + name;
    }
    else if (testPattern) title += " — test pattern";
    fprintf(stderr, "[app] window title: \"%s\"\n", title.c_str());
    SDL_Window* win = SDL_CreateWindow(title.c_str(), (int)winW, (int)winH, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!win) { fprintf(stderr, "[app] SDL_CreateWindow: %s\n", SDL_GetError()); return 1; }

    Vk vk;
    // Present unification (#1270): prefer the renderer's shared device for real game boots so we can
    // GPU-blit its front-buffer image. The test pattern, PROSPER_APP_GPU_PRESENT=0, and any adoption
    // failure fall back to a private device + CPU pixels.
    const bool wantGpuPresent = prosper::frontend::request_gpu_present(
        getenv("PROSPER_APP_GPU_PRESENT"), testPattern, !dump.empty());
    if (!wantGpuPresent || !try_adopt_shared_present(vk, win)) {
        if (!create_instance(vk, win) || !pick_device(vk)) return 1;
    }
    // Initial swapchain sized to the window; recreated on resize / out-of-date.
    { int dw = 0, dh = 0; SDL_GetWindowSizeInPixels(win, &dw, &dh);
      if (!create_swapchain(vk, (uint32_t)dw, (uint32_t)dh, requestedPresentMode)) return 1; }

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi.queueFamilyIndex = vk.qfamily;
    vkCreateCommandPool(vk.device, &cpi, nullptr, &vk.cmdPool);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = vk.cmdPool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(vk.device, &cai, &vk.cmd);
    if (!replace_present_sync(vk)) {
        fprintf(stderr, "[app] could not create present synchronization\n");
        return 1;
    }

    fprintf(stderr, "[app] window up (%s). Close the window or press Esc to quit.\n",
            testPattern ? "test-pattern" : "waiting for guest frames");

    // Keyboard controls augment SDL pad 0; the fallback keeps physical pads and their analog state.
    prosper::input::pad_set_backend(&g_keyboard_pad);
    fprintf(stderr, "[app] keyboard: WASD/Arrows=move  J/Space=Cross(jump)  K=Square(attack)  L=Circle  "
                    "I=Triangle  U/O=L1/R1  Y/H=L2/R2  Enter=Options  "
                    "Pause/F10=pause  F11/Alt+Enter=fullscreen  Esc=quit\n");
    fprintf(stderr, "[app] F9 = grab the current frame: writes a replayable .prgbundle + a .bmp "
                    "screenshot (to PROSPER_CAPTURE_DIR, default cwd) for gpu_replay debugging "
                    "(brief hitch on press; PROSPER_CAPTURE_FRAMES>1 grabs an animation).\n");

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
    int gpuPrevSlot = -1;   // #1270: the GPU scanout slot presented last frame (released after its read)
    unsigned timedDumpCount = 0;
    // Interactive frame grab (F9): base output dir + a per-session counter; when a grab was armed we
    // also snapshot the next presented CPU frame to a BMP next to its .prgcap.
    const std::string grabDir = getenv("PROSPER_CAPTURE_DIR") ? getenv("PROSPER_CAPTURE_DIR") : ".";
    unsigned grabCounter = 0;
    std::string pendingGrabScreenshot;   // non-empty => write the next presented CPU frame to this path
    uint64_t pendingGrabGuestPresent = 0;
    auto flushGrabScreenshot = [&](const uint8_t* rgba, uint32_t w, uint32_t h) {
        if (pendingGrabScreenshot.empty()) return;
        const bool ok = write_frame_bmp(pendingGrabScreenshot, rgba, w, h);
        std::fprintf(stderr,
                     "[grab] screenshot%s armed at guest present %llu "
                     "(written at guest present %llu) -> %s\n",
                     ok ? "" : " write FAILED",
                     static_cast<unsigned long long>(pendingGrabGuestPresent),
                     static_cast<unsigned long long>(gpu::present_count()),
                     pendingGrabScreenshot.c_str());
        pendingGrabScreenshot.clear();
        pendingGrabGuestPresent = 0;
    };
    const uint64_t scheduledScreenshotFrame = prosper::frontend::parse_capture_frame(
        getenv("PROSPER_CAPTURE_SCREENSHOT_AT_FRAME"));
    std::string scheduledScreenshotPath;
    if (scheduledScreenshotFrame) {
        if (const char* configured = getenv("PROSPER_CAPTURE_SCREENSHOT")) {
            scheduledScreenshotPath = configured;
        } else {
            scheduledScreenshotPath = grabDir + "/scheduled_frame_" +
                std::to_string(scheduledScreenshotFrame) + ".bmp";
        }
        fprintf(stderr, "[grab] screenshot scheduled at host frame %llu -> %s\n",
                static_cast<unsigned long long>(scheduledScreenshotFrame),
                scheduledScreenshotPath.c_str());
    }
    bool scheduledScreenshotArmed = false;
    bool timedDumpPending = timedDumpMs > 0;
    bool running = true;
    bool paused = false;
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
            else if (ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
                     ev.window.windowID == appWindowId) {
                // A progress utility window means the app window is not necessarily SDL's last
                // window, so closing it no longer guarantees a synthesized SDL_EVENT_QUIT.
                running = false;
            } else if (ev.type == SDL_EVENT_KEY_DOWN || ev.type == SDL_EVENT_KEY_UP) {
                prosper::frontend::AppWindowKey key{};
                key.app_window = ev.key.windowID == appWindowId;
                key.pressed = ev.key.down;
                key.repeat = ev.key.repeat;
                key.escape = ev.key.key == SDLK_ESCAPE;
                key.pause = ev.key.key == SDLK_PAUSE || ev.key.key == SDLK_F10;
                key.f11 = ev.key.key == SDLK_F11;
                key.enter = ev.key.key == SDLK_RETURN || ev.key.key == SDLK_KP_ENTER;
                key.alt = (ev.key.mod & SDL_KMOD_ALT) != 0;
                // Interactive frame grab: F9 arms a one-shot capture of the next COMPLETE frame (every
                // submit between the next two presents) into a replayable .prgbundle, plus a screenshot.
                // The whole-frame bundle re-runs the frame's producer submits on replay, so renderer-owned
                // RTTs regenerate instead of replaying black (as a single-submit .prgcap does for a
                // deferred renderer). It is a HOST hotkey, NOT forwarded to the guest IME/pad. On-demand
                // only — near-zero cost until pressed, so it never distorts the FPS you are observing.
                if (ev.type == SDL_EVENT_KEY_DOWN && key.app_window && !ev.key.repeat &&
                    ev.key.key == SDLK_F9) {
                    char base[512];
                    std::snprintf(base, sizeof base, "%s/frame_grab_%03u", grabDir.c_str(), ++grabCounter);
                    prosper::gpu::request_interactive_capture_bundle(std::string(base) + ".prgbundle");
                    pendingGrabScreenshot = std::string(base) + ".bmp";
                    pendingGrabGuestPresent = gpu::present_count();
                    std::fprintf(stderr,
                        "[grab] F9: arming a whole-frame capture -> %s.prgbundle (+ .bmp screenshot)\n",
                        base);
                    continue;
                }
                // #1093: forward app-window keys to the guest's IME keyboard path. Titles like
                // PPSA02664 read input through sceImeUpdate, not libScePad. SDL3 scancodes ARE USB
                // HID usage ids for the keyboard page — exactly the keycode the guest event wants.
                // Deliver clean press/release edges (skip auto-repeat).
                if (key.app_window && !ev.key.repeat && !key.pause)
                    prosper::ime_push_key((uint16_t)ev.key.scancode, ev.key.down);
                switch (windowControls.handle_key(key)) {
                case prosper::frontend::AppWindowCommand::quit:
                    running = false;
                    break;
                case prosper::frontend::AppWindowCommand::toggle_pause:
                    paused = !paused;
                    if (paused) {
                        // Close the producer gate before freezing queued device audio.
                        prosper_set_paused(true);
#ifdef PROSPER_AUDIO_SDL3
                        prosper::set_sdl3_audio_paused(true);
#endif
                    } else {
                        // Start the device before releasing producers so the first resumed grain
                        // cannot queue behind a device that is still paused.
#ifdef PROSPER_AUDIO_SDL3
                        prosper::set_sdl3_audio_paused(false);
#endif
                        prosper_set_paused(false);
                    }
                    SDL_SetWindowTitle(win, paused ? (title + " — paused").c_str() : title.c_str());
                    fprintf(stderr, "[app] %s at guest flip boundary\n",
                            paused ? "pause requested" : "resumed");
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
            {
                // #1270: on a shared present queue, hold the submit mutex across the device drain so no
                // guest submit races vkDeviceWaitIdle (which waits on all queues). Only the wait needs it;
                // the swapchain destroy/create below do not touch the render queue.
                std::unique_lock<std::mutex> lk(gpu::shared_present_submit_mutex(), std::defer_lock);
                if (vk.gpu_present && vk.queue_shared) lk.lock();
                vkDeviceWaitIdle(vk.device);
            }
            if (vk.gpu_present && gpuPrevSlot >= 0) {
                prosper::frontend::present_blit_release(gpuPrevSlot); gpuPrevSlot = -1;
            }
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
        if (testPattern && !paused) feed_test_pattern(kPatW, kPatH, patFrame++);

        // Cheap calibration companion to a heavyweight F9 bundle. Read back exactly one presented
        // frame, with no GPU command capture, so long routes can locate a visual checkpoint before
        // scheduling the replayable bundle there. Do not overwrite an interactive F9 screenshot;
        // if both coincide, the scheduled shot remains eligible for the following real present.
        if (pendingGrabScreenshot.empty() && prosper::frontend::capture_frame_due(
                scheduledScreenshotFrame, shown, scheduledScreenshotArmed)) {
            pendingGrabScreenshot = scheduledScreenshotPath;
            pendingGrabGuestPresent = gpu::present_count();
            scheduledScreenshotArmed = true;
            fprintf(stderr,
                    "[grab] scheduled screenshot armed at host frame %llu (guest present %llu)\n",
                    static_cast<unsigned long long>(shown + 1),
                    static_cast<unsigned long long>(pendingGrabGuestPresent));
        }

        // Present the latest finished frame from the core's present layer.
        // The frame dimensions: from the guest's registered VideoOut display in normal use; in
        // test-pattern mode there is no guest, so use the dims we feed (present_width/height report
        // the VideoOut registry, which is empty without a guest). Either way, readback needs a
        // buffer sized to the frame it holds — guard zero dims so we never present a 0-extent image.
        // Render completion and guest flips are separate clocks: the command stream can flip before
        // the renderer publishes its CPU frame. Key this loop to the completed-frame sequence so a
        // late renderer publication is not missed or marked handled while only the previous frame exists.
        if (vk.gpu_present) {
            // GPU present (#1270): blit the renderer's front-buffer image straight to the swapchain.
            prosper::frontend::GpuScanoutFrame gf;
            if (prosper::frontend::present_blit_acquire(gf)) {
                bool grabReady = false;
                PresentAttempt attempt = present_frame_gpu(
                    vk, gf, gpuPrevSlot, !pendingGrabScreenshot.empty(), grabReady);
                if (grabReady)
                    flushGrabScreenshot(static_cast<const uint8_t*>(vk.stageMapped),
                                        gf.width, gf.height);
                if (attempt == PresentAttempt::out_of_date) {
                    swapchainDirty = true;
                } else if (attempt == PresentAttempt::skipped) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(4));
                } else if (attempt == PresentAttempt::failed) {
                    running = false;
                } else {
                    shown++; lastFrameProgress = std::chrono::steady_clock::now();
                    static auto t0 = std::chrono::steady_clock::now(); static uint64_t mark = 0;
                    if (shown - mark >= 60) {
                        auto now = std::chrono::steady_clock::now();
                        double s = std::chrono::duration<double>(now - t0).count();
                        fprintf(stderr, "[app] %.1f fps (%llu frames, gpu-present)\n",
                                (shown - mark) / (s > 0 ? s : 1), (unsigned long long)shown);
                        t0 = now; mark = shown;
                    }
                    if (exitAfter && (int)shown >= exitAfter) running = false;
                }
            } else if (gpu::present_frame_seq() != lastFrameSeq) {
                // #1270 Finding 2: no GPU frame was published this iteration. On a publish MISS (front
                // target evicted/invalidated, or no free slot) the renderer still did the CPU readback, so
                // present that CPU frame rather than stranding the window on a stale/black image. Also
                // covers startup before the first publish. present_frame serializes on the shared queue.
                gpu::PresentFrameLease cf;
                if (gpu::present_acquire_rendered_frame(cf) && cf.width && cf.height && cf.rgba &&
                    cf.rgba->size() == (size_t)cf.width * cf.height * 4) {
                    PresentAttempt a = present_frame(vk, cf.rgba->data(), cf.width, cf.height);
                    if (gpuPrevSlot >= 0) { prosper::frontend::present_blit_release(gpuPrevSlot); gpuPrevSlot = -1; }
                    if (a == PresentAttempt::out_of_date) swapchainDirty = true;
                    else if (a == PresentAttempt::skipped) std::this_thread::sleep_for(std::chrono::milliseconds(4));
                    else if (a == PresentAttempt::failed) running = false;
                    else { lastFrameSeq = cf.frame_seq; shown++; lastFrameProgress = std::chrono::steady_clock::now();
                           flushGrabScreenshot(cf.rgba->data(), cf.width, cf.height); }
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));   // no new GPU or CPU frame yet
            }
        } else {
        bool newFrame = testPattern || (gpu::present_frame_seq() != lastFrameSeq);
        gpu::PresentFrameLease frame;
        if (newFrame && gpu::present_acquire_rendered_frame(frame)) {
            uint32_t w = frame.width;
            uint32_t h = frame.height;
            if (w == 0 || h == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
            if (frame.rgba && frame.rgba->size() == (size_t)w * h * 4) {
                PresentAttempt attempt = present_frame(vk, frame.rgba->data(), w, h);
                if (attempt == PresentAttempt::out_of_date) {
                    // Out-of-date/suboptimal: share the resize/fullscreen recreation path next loop.
                    swapchainDirty = true;
                } else if (attempt == PresentAttempt::skipped) {
                    // Window occluded/minimized (#1182): no swapchain image within the bounded acquire.
                    // Leave lastFrameSeq unchanged so we present the freshest frame once the window is
                    // visible again, and do not mark the swapchain dirty. The guest keeps running.
                    std::this_thread::sleep_for(std::chrono::milliseconds(4));
                } else if (attempt == PresentAttempt::failed) {
                    running = false;
                } else {
                    lastFrameSeq = frame.frame_seq; shown++;
                    lastFrameProgress = std::chrono::steady_clock::now();
                    flushGrabScreenshot(frame.rgba->data(), w, h);
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
        }   // end CPU-present branch (#1270)
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
