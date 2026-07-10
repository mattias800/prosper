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
#include "host/boot_program.hpp"       // boot_program (shared guest-boot path, also used by boot_trace)
#include "host/exec_image.hpp"         // run_entry
#include "loader/linker.hpp"           // Program
#include "input/pad.hpp"               // keyboard -> libScePad (HostPadState / PadBackend)
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
#include <mutex>

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

// ---- keyboard -> virtual DualSense (pad 0) ----------------------------------------------------
// A controller over WSL passthrough is flaky, so the app maps the keyboard onto the same
// HostPadState the SDL gamepad backend fills. The event loop snapshots the current key state into
// g_kb_state (main thread); the guest's scePadReadState reads it through this backend (guest thread).
std::mutex g_kb_mx;
prosper::input::HostPadState g_kb_state;

struct KeyboardPad : prosper::input::PadBackend {
    bool poll(int index, prosper::input::HostPadState& out) override {
        if (index != 0) return false;                 // single virtual pad
        std::lock_guard<std::mutex> lk(g_kb_mx);
        out = g_kb_state;
        return true;
    }
};
KeyboardPad g_keyboard_pad;

// Snapshot the current keyboard into g_kb_state. Call from the thread that pumps SDL events.
void poll_keyboard() {
    using namespace prosper::input;
    const bool* k = SDL_GetKeyboardState(nullptr);
    auto d = [&](SDL_Scancode s){ return k[s]; };
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
    if (d(SDL_SCANCODE_RETURN) || d(SDL_SCANCODE_RETURN2)) b |= SCE_PAD_BUTTON_OPTIONS;   // menu/start
    HostPadState st;
    st.buttons = b;
    st.left_x  = left ? 0x00 : (right ? 0xff : 0x80);   // also drive the left stick, for stick-only titles
    st.left_y  = up   ? 0x00 : (down  ? 0xff : 0x80);
    st.l2 = d(SDL_SCANCODE_Y) ? 255 : 0;
    st.r2 = d(SDL_SCANCODE_H) ? 255 : 0;
    st.connected = true;
    std::lock_guard<std::mutex> lk(g_kb_mx);
    g_kb_state = st;
}

} // namespace

int main(int argc, char** argv) {
    bool testPattern = false; int exitAfter = 0; uint32_t winW = 1280, winH = 720;
    std::string dump;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--test-pattern") testPattern = true;
        else if (a == "--frames" && i + 1 < argc) exitAfter = atoi(argv[++i]);   // present N frames then exit (CI/smoke)
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];                // boot the game at this app0 dir
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
        prosper::frontend::register_live_renderer(".", /*dump_bmps=*/false);   // composite to the present layer, no disk spam
#else
        fprintf(stderr, "[app] built without the live renderer; the window will stay blank.\n");
#endif
        static Program prog; std::string err;
        // Install host frontends (audio out, controller in) at the same point boot_trace does —
        // right after the built-in HLE is registered, before the guest runs. Built in only when the
        // corresponding SDL3 frontend is enabled; a window app wants both on by default.
        auto install_backends = []{
#ifdef PROSPER_AUDIO_SDL3
            prosper::install_sdl3_audio_sink();
#endif
#ifdef PROSPER_PAD_SDL3
            if (prosper::install_sdl3_pad_backend()) fprintf(stderr, "[app] controller backend installed.\n");
#endif
#ifdef PROSPER_HAVE_DIALOG_SDL3
            prosper::install_sdl3_platform_ui();   // real SDL message boxes for MsgDialog/ErrorDialog (#347)
            fprintf(stderr, "[app] dialog backend installed.\n");
#endif
        };
        if (!boot_program(dump, prog, &err, install_backends)) { fprintf(stderr, "[app] boot failed: %s\n", err.c_str()); return 1; }
        guestThread = std::thread([]{ run_entry(prog.imgs[0]); });   // runs the guest frame loop
        fprintf(stderr, "[app] guest booted; presenting its frames.\n");
    } else if (!testPattern) {
        fprintf(stderr, "[app] no dump given and not --test-pattern; waiting for external present frames.\n");
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) { fprintf(stderr, "[app] SDL_Init: %s\n", SDL_GetError()); return 1; }
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

    // Keyboard controls (installed last, so it's the active pad even if the SDL gamepad backend ran).
    prosper::input::pad_set_backend(&g_keyboard_pad);
    fprintf(stderr, "[app] keyboard: WASD/Arrows=move  J/Space=Cross(jump)  K=Square(attack)  L=Circle  "
                    "I=Triangle  U/O=L1/R1  Y/H=L2/R2  Enter=Options  Esc=quit\n");

    uint64_t shown = 0, lastCount = ~0ull, patFrame = 0;
    bool running = true;
    while (running && !prosper_stop_requested()) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_EVENT_QUIT) running = false;
            else if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.key == SDLK_ESCAPE) running = false;
        }
        if (!running) break;
        poll_keyboard();   // snapshot key state for the guest's pad reads
#ifdef PROSPER_HAVE_DIALOG_SDL3
        prosper::sdl_platform_ui_pump();   // run a pending ImeDialog text-entry modal on this (main) thread
#endif

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
                    // Periodic present-rate log (every 60 presented frames).
                    static auto t0 = std::chrono::steady_clock::now(); static uint64_t mark = 0;
                    if (shown - mark >= 60) {
                        auto now = std::chrono::steady_clock::now();
                        double s = std::chrono::duration<double>(now - t0).count();
                        fprintf(stderr, "[app] %.1f fps (%llu frames)\n", (shown - mark) / (s > 0 ? s : 1), (unsigned long long)shown);
                        t0 = now; mark = shown;
                    }
                    if (exitAfter && (int)shown >= exitAfter) running = false;
                }
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));   // no new frame — don't spin
        }
    }

    prosper_request_stop();   // signal the guest run-loop to wind down at its next boundary
    fprintf(stderr, "[app] shutting down after %llu presented frame(s)\n", (unsigned long long)shown);
    vkDeviceWaitIdle(vk.device);
    // The guest runs guest code with no cooperative yield point yet (a flip-boundary stop check is a
    // refinement), so we detach and let process exit reclaim it rather than block on a join.
    if (guestThread.joinable()) guestThread.detach();
    SDL_DestroyWindow(win); SDL_Quit();
    return (exitAfter && (int)shown < exitAfter) ? 1 : 0;
}
