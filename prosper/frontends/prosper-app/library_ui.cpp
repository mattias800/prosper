// library_ui.cpp — see library_ui.hpp. Draws the library grid with Dear ImGui on the app's existing
// Vulkan device, and decodes cover art with stb_image (#1471).
#include "library_ui.hpp"
#include "library_nav.hpp"

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG            // the only format sce_sys ships for icons; keeps the surface narrow
#define STBI_NO_STDIO            // we hand it bytes we already read, so it never opens files itself
#define STBI_NO_FAILURE_STRINGS
#include "stb_image.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace prosper::frontend {
namespace {

// Cover art is square (icon0.png is 512x512 in every dump), so 320 still samples the source DOWN and
// cannot show resampling artefacts. The previous 160 read as very small on a desktop monitor.
constexpr float kCoverSize   = 320.0f;
// Wide gutters are what let the focused title's background art actually be seen between the cards
// (#1630) — at the old spacing the grid covered almost all of it.
constexpr float kCellPadding = 96.0f;
constexpr float kLabelHeight = 84.0f;    // up to three lines of wrapped title at the larger font
constexpr float kTitleFontPx = 26.0f;    // 2x ImGui's 13px default; integer scale keeps the bitmap crisp
constexpr float kCellWidth   = kCoverSize + kCellPadding;
constexpr float kCellHeight  = kCoverSize + kLabelHeight + kCellPadding;

std::string read_file_bytes(const std::string& path) {
    std::string out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    char buf[16384]; size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

// ImGui's Vulkan backend ignores every VkResult unless given this hook, so a failed pipeline or
// descriptor allocation would otherwise show up only as a window that draws nothing.
void imgui_vk_result(VkResult r) {
    if (r != VK_SUCCESS) fprintf(stderr, "[library] imgui vulkan error: %d\n", static_cast<int>(r));
}

} // namespace

LibraryUi::~LibraryUi() { shutdown(); }

bool LibraryUi::init(SDL_Window* window, VkInstance instance, VkPhysicalDevice phys, VkDevice device,
                     uint32_t queue_family, VkQueue queue, VkSwapchainKHR swapchain,
                     VkFormat swapchain_format, const std::vector<VkImage>& swapchain_images,
                     VkExtent2D extent) {
    window_ = window; device_ = device; phys_ = phys; queue_ = queue; qfamily_ = queue_family;
    swapchain_ = swapchain;

    // A pool large enough for ImGui's font atlas plus one descriptor per cover. Sized from the grid,
    // not guessed: exceeding it would silently drop covers.
    const uint32_t kMaxTextures = 256;
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxTextures },
    };
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = kMaxTextures;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(device_, &dpi, nullptr, &pool_) != VK_SUCCESS) {
        fprintf(stderr, "[library] descriptor pool creation failed\n");
        shutdown();
        return false;
    }

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = qfamily_;
    if (vkCreateCommandPool(device_, &cpi, nullptr, &cmdPool_) != VK_SUCCESS) { shutdown(); return false; }
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cmdPool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device_, &cai, &cmd_) != VK_SUCCESS) { shutdown(); return false; }

    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateSemaphore(device_, &si, nullptr, &acquireSem_) != VK_SUCCESS ||
        vkCreateSemaphore(device_, &si, nullptr, &renderSem_) != VK_SUCCESS ||
        vkCreateFence(device_, &fi, nullptr, &inFlight_) != VK_SUCCESS) { shutdown(); return false; }

    VkSamplerCreateInfo sam{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sam.magFilter = VK_FILTER_LINEAR; sam.minFilter = VK_FILTER_LINEAR;
    sam.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sam.addressModeU = sam.addressModeV = sam.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sam.maxLod = 1.0f;
    if (vkCreateSampler(device_, &sam, nullptr, &sampler_) != VK_SUCCESS) { shutdown(); return false; }

    if (!create_render_target(swapchain_format, swapchain_images, extent)) { shutdown(); return false; }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    imguiCtx_ = true;   // set immediately: shutdown() must destroy the context even if a backend fails
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;   // no imgui.ini beside the binary: the app has its own settings file
    // ImGui's own keyboard/gamepad nav is deliberately NOT enabled: this screen drives selection with
    // the unit-tested grid rules in library_nav.hpp, and letting both run means the arrow keys move a
    // widget focus as well as the selection, and Enter activates whatever widget that focus landed on
    // rather than launching the highlighted game.
    io.ConfigFlags &= ~(ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad);
    ImGui::StyleColorsDark();

    // Game titles are drawn at twice the UI font size. A SECOND atlas entry rather than
    // SetWindowFontScale: ImGui's default font is a bitmap designed for 13 px, and scaling it at draw
    // time blurs it, while baking it at an exact 2x integer multiple stays sharp. The first font added
    // remains the default, so the header, footer and buttons are unchanged.
    io.Fonts->AddFontDefault();
    ImFontConfig titleCfg;
    titleCfg.SizePixels = kTitleFontPx;
    titleFont_ = io.Fonts->AddFontDefault(&titleCfg);

    if (!ImGui_ImplSDL3_InitForVulkan(window_)) { fprintf(stderr, "[library] SDL3 backend init failed\n"); shutdown(); return false; }
    sdlInit_ = true;
    ImGui_ImplVulkan_InitInfo vi{};
    vi.Instance = instance;
    vi.PhysicalDevice = phys_;
    vi.Device = device_;
    vi.QueueFamily = qfamily_;
    vi.Queue = queue_;
    vi.DescriptorPool = pool_;
    vi.RenderPass = renderPass_;
    vi.MinImageCount = static_cast<uint32_t>(swapchain_images.size());
    vi.ImageCount = static_cast<uint32_t>(swapchain_images.size());
    vi.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    vi.CheckVkResultFn = imgui_vk_result;
    if (!ImGui_ImplVulkan_Init(&vi)) { fprintf(stderr, "[library] Vulkan backend init failed\n"); shutdown(); return false; }
    vulkanInit_ = true;
    ready_ = true;

    // Background art and focus music (#1630). Must come after the ImGui Vulkan backend, which owns the
    // descriptor allocation the backgrounds register through. A failure here is not fatal: the library
    // then looks exactly as it did before this feature existed.
    const char* statsEnv = SDL_getenv("PROSPER_LIBRARY_STATS");
    stats_ = statsEnv && *statsEnv && statsEnv[0] != '0';

    // PROSPER_LIBRARY_STATS marks a measurement/automation run — a person browsing their games does not
    // ask for a frame-time histogram. Such a run is silent unless PROSPER_LAUNCHER_MUSIC=1 explicitly
    // asks for sound, so screenshot captures, timing sweeps and scripted routes cannot play audio on
    // the developer's desktop by default.
    const char* musicEnv = SDL_getenv("PROSPER_LAUNCHER_MUSIC");
    musicToggle_ = resolve_launcher_music(musicEnv, musicToggle_, /*automated=*/stats_);
    const float gain = resolve_launcher_music_gain(SDL_getenv("PROSPER_LAUNCHER_MUSIC_VOLUME"));
    if (!media_.init(phys_, device_, queue_, qfamily_, sampler_, musicToggle_, gain))
        fprintf(stderr, "[library] background art and music unavailable\n");
    // Mirror the EFFECTIVE state, not the request: if the audio device could not be opened the checkbox
    // must show unticked, or set_music_enabled() sees no change on the first click and swallows it.
    musicToggle_ = media_.music_enabled();
    return true;
}

bool LibraryUi::create_render_target(VkFormat format, const std::vector<VkImage>& images,
                                     VkExtent2D extent) {
    format_ = format;
    extent_ = extent;

    VkAttachmentDescription color{};
    color.format = format_;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;   // the library owns the whole frame; nothing under it
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 1; rpi.pAttachments = &color;
    rpi.subpassCount = 1;    rpi.pSubpasses = &sub;
    rpi.dependencyCount = 1; rpi.pDependencies = &dep;
    if (renderPass_ == VK_NULL_HANDLE &&
        vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_) != VK_SUCCESS) {
        fprintf(stderr, "[library] render pass creation failed\n");
        return false;
    }

    views_.resize(images.size(), VK_NULL_HANDLE);
    framebuffers_.resize(images.size(), VK_NULL_HANDLE);
    for (size_t i = 0; i < images.size(); i++) {
        VkImageViewCreateInfo ivi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivi.image = images[i];
        ivi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivi.format = format_;
        ivi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device_, &ivi, nullptr, &views_[i]) != VK_SUCCESS) return false;
        VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbi.renderPass = renderPass_;
        fbi.attachmentCount = 1; fbi.pAttachments = &views_[i];
        fbi.width = extent_.width; fbi.height = extent_.height; fbi.layers = 1;
        if (vkCreateFramebuffer(device_, &fbi, nullptr, &framebuffers_[i]) != VK_SUCCESS) return false;
    }
    return true;
}

void LibraryUi::destroy_render_target() {
    for (VkFramebuffer fb : framebuffers_) if (fb) vkDestroyFramebuffer(device_, fb, nullptr);
    for (VkImageView v : views_) if (v) vkDestroyImageView(device_, v, nullptr);
    framebuffers_.clear();
    views_.clear();
}

bool LibraryUi::recreate_swapchain(VkSwapchainKHR swapchain, VkFormat format,
                                   const std::vector<VkImage>& images, VkExtent2D extent) {
    if (!ready_) return false;
    vkDeviceWaitIdle(device_);
    destroy_render_target();
    swapchain_ = swapchain;
    if (!create_render_target(format, images, extent)) { ready_ = false; return false; }
    return true;
}

void LibraryUi::shutdown() {
    if (stats_ && frameCount_ > 0 && !statsReported_) {
        statsReported_ = true;   // shutdown() is idempotent and can run twice; the report should not
        std::vector<double> s = frameSamples_;
        std::sort(s.begin(), s.end());
        const auto pct = [&](double p) {
            if (s.empty()) return 0.0;
            size_t i = static_cast<size_t>(p * (s.size() - 1));
            return s[i];
        };
        fprintf(stderr,
                "[library] frames=%llu mean=%.2fms median=%.2fms p99=%.2fms max=%.2fms(frame #%llu) "
                "over-16.7ms=%llu over-33.3ms=%llu loads=%llu discarded=%llu\n",
                (unsigned long long)frameCount_, frameTotalMs_ / (double)frameCount_,
                pct(0.50), pct(0.99), frameMaxMs_, (unsigned long long)frameMaxIndex_,
                (unsigned long long)frameOver16_, (unsigned long long)frameOver33_,
                (unsigned long long)media_.loads_started(),
                (unsigned long long)media_.loads_discarded());
    }
    if (device_) vkDeviceWaitIdle(device_);
    // Before the ImGui Vulkan backend goes away: the backgrounds hold descriptor sets allocated from
    // it, and it also stops the worker and closes the audio device, so no music survives into a boot.
    media_.shutdown();
    destroy_covers();
    // Each step is separately conditional: a failure part-way through init must still unwind whatever
    // was created, and shutdown() is called from those failure paths.
    if (vulkanInit_) { ImGui_ImplVulkan_Shutdown(); vulkanInit_ = false; }
    if (sdlInit_)    { ImGui_ImplSDL3_Shutdown();   sdlInit_ = false; }
    if (imguiCtx_)   { ImGui::DestroyContext();     imguiCtx_ = false; }
    destroy_render_target();
    if (renderPass_) { vkDestroyRenderPass(device_, renderPass_, nullptr); renderPass_ = VK_NULL_HANDLE; }
    if (sampler_)    { vkDestroySampler(device_, sampler_, nullptr);       sampler_ = VK_NULL_HANDLE; }
    if (inFlight_)   { vkDestroyFence(device_, inFlight_, nullptr);        inFlight_ = VK_NULL_HANDLE; }
    if (acquireSem_) { vkDestroySemaphore(device_, acquireSem_, nullptr);  acquireSem_ = VK_NULL_HANDLE; }
    if (renderSem_)  { vkDestroySemaphore(device_, renderSem_, nullptr);   renderSem_ = VK_NULL_HANDLE; }
    if (cmdPool_)    { vkDestroyCommandPool(device_, cmdPool_, nullptr);   cmdPool_ = VK_NULL_HANDLE; cmd_ = VK_NULL_HANDLE; }
    if (pool_)       { vkDestroyDescriptorPool(device_, pool_, nullptr);   pool_ = VK_NULL_HANDLE; }
    ready_ = false;
}

void LibraryUi::destroy_covers() {
    for (Cover& c : covers_) {
        if (c.set)    ImGui_ImplVulkan_RemoveTexture(c.set);
        if (c.view)   vkDestroyImageView(device_, c.view, nullptr);
        if (c.image)  vkDestroyImage(device_, c.image, nullptr);
        if (c.memory) vkFreeMemory(device_, c.memory, nullptr);
    }
    covers_.clear();
}

void LibraryUi::set_games(std::vector<GameEntry> games, const std::string& games_dir) {
    // Hold the selection on the same title across a rescan, so adding a game elsewhere in the list
    // does not move the cursor under the user's hands.
    std::string keep;
    if (selected_ >= 0 && selected_ < static_cast<int>(games_.size()))
        keep = games_[static_cast<size_t>(selected_)].app0_root;

    if (device_) vkDeviceWaitIdle(device_);   // covers may still be referenced by an in-flight frame
    destroy_covers();
    games_ = std::move(games);
    gamesDir_ = games_dir;
    covers_.resize(games_.size());

    selected_ = 0;
    if (!keep.empty())
        for (size_t i = 0; i < games_.size(); i++)
            if (games_[i].app0_root == keep) { selected_ = static_cast<int>(i); break; }
    firstRow_ = 0;
}

bool LibraryUi::handle_event(const SDL_Event& ev) {
    if (!ready_) return false;
    ImGui_ImplSDL3_ProcessEvent(&ev);
    const ImGuiIO& io = ImGui::GetIO();
    // Report only what ImGui actually claimed, so the app still sees close/resize and its own hotkeys.
    if (ev.type == SDL_EVENT_KEY_DOWN || ev.type == SDL_EVENT_KEY_UP) return io.WantCaptureKeyboard;
    if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN || ev.type == SDL_EVENT_MOUSE_BUTTON_UP ||
        ev.type == SDL_EVENT_MOUSE_MOTION || ev.type == SDL_EVENT_MOUSE_WHEEL)
        return io.WantCaptureMouse;
    return false;
}

VkDescriptorSet LibraryUi::cover_for(const GameEntry& game) {
    const size_t index = static_cast<size_t>(&game - games_.data());
    if (index >= covers_.size()) return VK_NULL_HANDLE;
    Cover& cover = covers_[index];
    if (cover.tried) return cover.set;
    cover.tried = true;   // one attempt per title: a corrupt icon must not be re-decoded every frame

    if (game.icon_path.empty()) return VK_NULL_HANDLE;
    const std::string bytes = read_file_bytes(game.icon_path);
    if (bytes.empty()) return VK_NULL_HANDLE;

    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()),
                                            static_cast<int>(bytes.size()), &w, &h, &channels, 4);
    if (!pixels || w <= 0 || h <= 0) {
        if (pixels) stbi_image_free(pixels);
        fprintf(stderr, "[library] could not decode %s\n", game.icon_path.c_str());
        return VK_NULL_HANDLE;
    }
    const VkDeviceSize bytesNeeded = static_cast<VkDeviceSize>(w) * h * 4;

    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
    ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    bool ok = vkCreateImage(device_, &ii, nullptr, &cover.image) == VK_SUCCESS;
    if (ok) {
        VkMemoryRequirements mr{};
        vkGetImageMemoryRequirements(device_, cover.image, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = find_memory_type(phys_, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        ok = ai.memoryTypeIndex != UINT32_MAX &&
             vkAllocateMemory(device_, &ai, nullptr, &cover.memory) == VK_SUCCESS &&
             vkBindImageMemory(device_, cover.image, cover.memory, 0) == VK_SUCCESS;
    }
    if (ok) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytesNeeded;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ok = vkCreateBuffer(device_, &bi, nullptr, &staging) == VK_SUCCESS;
        if (ok) {
            VkMemoryRequirements mr{};
            vkGetBufferMemoryRequirements(device_, staging, &mr);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = mr.size;
            ai.memoryTypeIndex = find_memory_type(phys_, mr.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            ok = ai.memoryTypeIndex != UINT32_MAX &&
                 vkAllocateMemory(device_, &ai, nullptr, &stagingMem) == VK_SUCCESS &&
                 vkBindBufferMemory(device_, staging, stagingMem, 0) == VK_SUCCESS;
        }
        if (ok) {
            void* mapped = nullptr;
            ok = vkMapMemory(device_, stagingMem, 0, bytesNeeded, 0, &mapped) == VK_SUCCESS;
            if (ok) { std::memcpy(mapped, pixels, static_cast<size_t>(bytesNeeded)); vkUnmapMemory(device_, stagingMem); }
        }
    }
    stbi_image_free(pixels);

    if (ok) {
        // A one-shot upload on the app's queue. The library is idle-state UI drawn a few times a
        // second, and covers are decoded once each, so a serialized upload is not worth a transfer
        // queue and its ownership transfers.
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cmdPool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        VkCommandBuffer up = VK_NULL_HANDLE;
        ok = vkAllocateCommandBuffers(device_, &cai, &up) == VK_SUCCESS;
        if (ok) {
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(up, &bi);
            VkImageMemoryBarrier toDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toDst.srcQueueFamilyIndex = toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.image = cover.image;
            toDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(up, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &toDst);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
            vkCmdCopyBufferToImage(up, staging, cover.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            VkImageMemoryBarrier toRead = toDst;
            toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(up, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &toRead);
            vkEndCommandBuffer(up);
            VkSubmitInfo su{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            su.commandBufferCount = 1; su.pCommandBuffers = &up;
            ok = vkQueueSubmit(queue_, 1, &su, VK_NULL_HANDLE) == VK_SUCCESS;
            if (ok) vkQueueWaitIdle(queue_);
            vkFreeCommandBuffers(device_, cmdPool_, 1, &up);
        }
    }
    if (staging)    vkDestroyBuffer(device_, staging, nullptr);
    if (stagingMem) vkFreeMemory(device_, stagingMem, nullptr);

    if (ok) {
        VkImageViewCreateInfo ivi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivi.image = cover.image;
        ivi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivi.format = VK_FORMAT_R8G8B8A8_UNORM;
        ivi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        ok = vkCreateImageView(device_, &ivi, nullptr, &cover.view) == VK_SUCCESS;
    }
    if (ok) {
        cover.set = ImGui_ImplVulkan_AddTexture(sampler_, cover.view,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    if (!ok || !cover.set) {
        // Leave the entry usable without art rather than dropping a bootable title from the list.
        fprintf(stderr, "[library] could not upload cover for %s\n", game.title_name.c_str());
        if (cover.view)   { vkDestroyImageView(device_, cover.view, nullptr); cover.view = VK_NULL_HANDLE; }
        if (cover.image)  { vkDestroyImage(device_, cover.image, nullptr);    cover.image = VK_NULL_HANDLE; }
        if (cover.memory) { vkFreeMemory(device_, cover.memory, nullptr);     cover.memory = VK_NULL_HANDLE; }
        cover.set = VK_NULL_HANDLE;
    }
    return cover.set;
}

// Consume an already-signalled acquire semaphore without drawing. Used when an image was acquired but
// cannot be rendered to; leaving the semaphore signalled would corrupt every later frame's pairing.
void LibraryUi::discard_acquired_frame() {
    vkResetFences(device_, 1, &inFlight_);
    vkResetCommandBuffer(cmd_, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_, &bi);
    vkEndCommandBuffer(cmd_);
    const VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo su{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    su.waitSemaphoreCount = 1; su.pWaitSemaphores = &acquireSem_; su.pWaitDstStageMask = &wait;
    su.commandBufferCount = 1; su.pCommandBuffers = &cmd_;
    if (vkQueueSubmit(queue_, 1, &su, inFlight_) != VK_SUCCESS) ready_ = false;
}

// Paint the focused title's key art full-bleed behind everything, then a scrim over it.
//
// The art is COVER-fitted, not stretched: pic1 is 16:9 and the window need not be, so the image is
// scaled to fill and the overflow is cropped by sampling a sub-rectangle. Stretching 4K key art to an
// arbitrary window shape looks obviously wrong, and letterboxing it would leave bars that read as a
// rendering bug rather than a choice.
//
// The scrim exists because the artwork is arbitrary — some titles' pic1 is bright enough that white
// UI text on it is unreadable. It darkens toward the app's normal background colour rather than
// tinting, so the grid keeps the contrast it was designed with whatever is behind it.
void LibraryUi::draw_backdrop() {
    const Backdrop bd = media_.backdrop();
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const ImVec2 p0 = vp->Pos;
    const ImVec2 p1 = ImVec2(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y);

    if (bd.texture && bd.alpha > 0.0f && bd.width > 0 && bd.height > 0 &&
        vp->Size.x > 0 && vp->Size.y > 0) {
        const float imageAspect = static_cast<float>(bd.width) / static_cast<float>(bd.height);
        const float viewAspect = vp->Size.x / vp->Size.y;
        ImVec2 uv0(0.0f, 0.0f), uv1(1.0f, 1.0f);
        if (viewAspect > imageAspect) {
            // The window is wider than the art: use the full width and crop top and bottom.
            const float keep = imageAspect / viewAspect;
            const float margin = (1.0f - keep) * 0.5f;
            uv0.y = margin; uv1.y = 1.0f - margin;
        } else if (viewAspect < imageAspect) {
            const float keep = viewAspect / imageAspect;
            const float margin = (1.0f - keep) * 0.5f;
            uv0.x = margin; uv1.x = 1.0f - margin;
        }
        const int a = static_cast<int>(bd.alpha * 255.0f + 0.5f);
        dl->AddImage(reinterpret_cast<ImTextureID>(bd.texture), p0, p1, uv0, uv1,
                     IM_COL32(255, 255, 255, a < 0 ? 0 : (a > 255 ? 255 : a)));
        // Scaled with the art so a title with no background does not flash a dark panel over the
        // app's normal colour partway through a fade.
        const int scrim = static_cast<int>(bd.alpha * 0.62f * 255.0f + 0.5f);
        dl->AddRectFilled(p0, p1, IM_COL32(18, 18, 21, scrim));
    }
}

LibraryAction LibraryUi::render_frame(const std::string& status) {
    LibraryAction action;
    if (!ready_) return action;

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    // Drive the media layer before anything is laid out, so the alpha the backdrop draws with and the
    // gain the music ramped to were both evaluated at the same instant.
    const uint64_t nowMs = SDL_GetTicks();
    if (stats_) {
        const uint64_t ns = SDL_GetTicksNS();
        if (lastFrameNs_ != 0) {
            const double ms = static_cast<double>(ns - lastFrameNs_) / 1e6;
            ++frameCount_;
            frameTotalMs_ += ms;
            if (ms > frameMaxMs_) { frameMaxMs_ = ms; frameMaxIndex_ = frameCount_; }
            if (ms > 16.7) {
                ++frameOver16_;
                // Named individually rather than only counted: a stutter is a tail event, and knowing
                // WHICH frame was slow is what distinguishes one-off startup cost from a load hitch
                // that would repeat every time the user moves the selection.
                fprintf(stderr, "[library] slow frame #%llu: %.2f ms\n",
                        (unsigned long long)frameCount_, ms);
            }
            if (ms > 33.3) ++frameOver33_;
            // Stops recording rather than evicting: percentiles over a bounded prefix are honest
            // about what they cover, whereas a ring buffer silently redefines the window mid-run.
            if (frameSamples_.size() < kMaxFrameSamples) frameSamples_.push_back(ms);
        }
        lastFrameNs_ = ns;
    }
    if (!games_.empty() && selected_ >= 0 && selected_ < static_cast<int>(games_.size()))
        media_.set_focus(games_[static_cast<size_t>(selected_)].app0_root, nowMs);
    media_.update(nowMs);
    draw_backdrop();

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    // NoBackground so the title's art is visible behind the grid (#1630). With no background loaded
    // this reveals the render pass's clear colour, which is the same flat dark the window used to
    // paint, so the view is unchanged for a title with no usable art.
    ImGui::Begin("prosper", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                 ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground);

    ImGui::TextUnformatted(games_.empty() ? "prosper" : "Choose a game");
    if (!gamesDir_.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%s)", gamesDir_.c_str());
    }
    ImGui::Separator();

    if (games_.empty()) {
        ImGui::Spacing();
        if (gamesDir_.empty()) {
            ImGui::TextWrapped("No games folder is set yet. Choose the folder that holds your PS5 game "
                               "directories - the ones containing eboot.bin and sce_sys.");
        } else {
            ImGui::TextWrapped("No PS5 games found in this folder. Each game is its own directory "
                               "containing eboot.bin and sce_sys.");
        }
        ImGui::Spacing();
        // Keyboard-reachable too: ImGui's own nav is off (it fights the grid rules), so without this
        // the only way out of the empty state would be the mouse or a drop.
        if (ImGui::Button("Choose folder...") ||
            ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter) ||
            ImGui::IsKeyPressed(ImGuiKey_Space))
            action.kind = LibraryAction::Kind::browse;
        ImGui::SameLine();
        ImGui::TextDisabled("or press Enter, or drop a game folder on this window");
    } else {
        const float avail = ImGui::GetContentRegionAvail().x;
        const int columns = library_columns_for_width(avail, kCellWidth);
        const int count = static_cast<int>(games_.size());

        // Selection. ImGui's own nav does not know about the grid, so movement is decided by the
        // unit-tested rules in library_nav.hpp and the scroll follows the selection.
        // Keyboard only. Controller navigation is NOT implemented: ImGui's gamepad nav is off (it
        // fights these rules), and nothing initialises SDL's gamepad subsystem while the library is
        // alive — the pad backend does that inside start_guest, by which point the library is gone.
        // Tracked separately rather than shipped as code that cannot fire.
        LibraryNavKey key = LibraryNavKey::none;
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))       key = LibraryNavKey::left;
        else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) key = LibraryNavKey::right;
        else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow))    key = LibraryNavKey::up;
        else if (ImGui::IsKeyPressed(ImGuiKey_DownArrow))  key = LibraryNavKey::down;
        else if (ImGui::IsKeyPressed(ImGuiKey_Home))     key = LibraryNavKey::home;
        else if (ImGui::IsKeyPressed(ImGuiKey_End))      key = LibraryNavKey::end;
        else if (ImGui::IsKeyPressed(ImGuiKey_PageUp))   key = LibraryNavKey::page_up;
        else if (ImGui::IsKeyPressed(ImGuiKey_PageDown)) key = LibraryNavKey::page_down;

        const float gridHeight = ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing() * 2.0f;
        const int rowsPerPage = static_cast<int>(gridHeight / kCellHeight);
        const int before = selected_;
        selected_ = library_nav_apply(key, selected_, count, columns, rowsPerPage);
        if (stats_ && selected_ != before)
            fprintf(stderr, "[library] selection %d -> %d (key=%d) at frame %llu\n", before, selected_,
                    (int)key, (unsigned long long)frameCount_);
        firstRow_ = library_scroll_row_for(selected_, columns, rowsPerPage, firstRow_);

        if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter) ||
            ImGui::IsKeyPressed(ImGuiKey_Space)) {
            action.kind = LibraryAction::Kind::open;
            action.app0_root = games_[static_cast<size_t>(selected_)].app0_root;
        }

        ImGui::BeginChild("grid", ImVec2(0, gridHeight), false);
        // Apply the computed scroll, but only when a key moved the selection: doing it every frame
        // would fight the mouse wheel. Without this the selection can move below the visible rows —
        // the pure test covers library_scroll_row_for, only this line makes it reach the screen.
        if (key != LibraryNavKey::none) ImGui::SetScrollY(static_cast<float>(firstRow_) * kCellHeight);
        for (int i = 0; i < count; i++) {
            if (i % columns != 0) ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::PushID(i);
            const GameEntry& game = games_[static_cast<size_t>(i)];
            const bool isSelected = (i == selected_);
            if (isSelected)
                ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));

            VkDescriptorSet cover = cover_for(game);
            bool clicked = false;
            if (cover) {
                clicked = ImGui::ImageButton("cover", reinterpret_cast<ImTextureID>(cover),
                                             ImVec2(kCoverSize, kCoverSize));
            } else {
                // No art: a labelled button of the same size keeps the grid aligned and the entry
                // launchable, which matters more than the picture.
                clicked = ImGui::Button(game.title_id.empty() ? "(no art)" : game.title_id.c_str(),
                                        ImVec2(kCoverSize, kCoverSize));
            }
            if (isSelected) ImGui::PopStyleColor();
            if (clicked) {
                selected_ = i;
                action.kind = LibraryAction::Kind::open;
                action.app0_root = game.app0_root;
            }
            // The title and its FOLDER NAME, not the absolute path. The folder name is what identifies
            // the dump (the content id is in it), while the rest of the path is the user's home
            // directory — which would otherwise end up in every screenshot anyone shares of their
            // library. The games directory is already shown once in the header for orientation.
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s\n%s", game.title_name.c_str(),
                                  path_basename(game.app0_root).c_str());

            // Wrapped to the cover's width so a long localized name ("Space Adventure Cobra - The
            // Awakening") breaks onto further lines instead of running into the next column.
            if (titleFont_) ImGui::PushFont(titleFont_);
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + kCoverSize);
            ImGui::TextUnformatted(game.title_name.c_str());
            ImGui::PopTextWrapPos();
            if (titleFont_) ImGui::PopFont();

            ImGui::PopID();
            ImGui::EndGroup();
        }
        ImGui::EndChild();

        ImGui::Separator();
        ImGui::Text("%d game%s", count, count == 1 ? "" : "s");
        ImGui::SameLine();
        if (ImGui::Button("Change folder...")) action.kind = LibraryAction::Kind::browse;
        ImGui::SameLine();
        // Discoverable rather than env-only: someone who does not want a launcher making noise should
        // not have to find a variable name to stop it. Reported to the caller so it is persisted.
        if (ImGui::Checkbox("Music", &musicToggle_)) {
            media_.set_music_enabled(musicToggle_, nowMs);
            action.kind = LibraryAction::Kind::set_music;
            action.music_on = musicToggle_;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("Enter opens  |  arrows move  |  Esc quits");
    }

    if (!status.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("%s", status.c_str());
    }

    ImGui::End();
    ImGui::Render();

    // --- present -------------------------------------------------------------------------------
    vkWaitForFences(device_, 1, &inFlight_, VK_TRUE, UINT64_MAX);
    uint32_t imageIndex = 0;
    const VkResult acq = vkAcquireNextImageKHR(device_, swapchain_, 100ull * 1000 * 1000,
                                               acquireSem_, VK_NULL_HANDLE, &imageIndex);
    // Split by whether an image was actually acquired, NOT by success-vs-failure. These three leave
    // acquireSem_ untouched, so returning is clean:
    if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_TIMEOUT || acq == VK_NOT_READY) {
        needsRecreate_ |= (acq == VK_ERROR_OUT_OF_DATE_KHR);   // sticky: never clobber a pending request
        return action;
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
        // Device or surface lost. Nothing was signalled, but returning straight into a loop with no
        // wait anywhere would spin a core, so stop drawing and let the app fall back.
        fprintf(stderr, "[library] acquire failed (%d); closing the library view\n", (int)acq);
        ready_ = false;
        return action;
    }
    // From here an image WAS acquired and acquireSem_ WILL be signalled, so every path below has to
    // consume it. VK_SUBOPTIMAL_KHR is a SUCCESS code — dropping this frame would leave the semaphore
    // signalled, so the next acquire would reuse a signalled semaphore and the wait/signal pairing
    // would be permanently off by one. Present it and ask the app to recreate afterwards.
    if (acq == VK_SUBOPTIMAL_KHR) needsRecreate_ = true;
    if (imageIndex >= framebuffers_.size()) {
        // Stale framebuffers (the swapchain changed under us). The image is already acquired, so drain
        // the semaphore with an empty submit rather than stranding it, then ask for a rebuild.
        discard_acquired_frame();
        needsRecreate_ = true;
        return action;
    }
    vkResetFences(device_, 1, &inFlight_);
    vkResetCommandBuffer(cmd_, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_, &bi);
    VkClearValue clear{};
    clear.color = {{0.07f, 0.07f, 0.08f, 1.0f}};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = renderPass_;
    rp.framebuffer = framebuffers_[imageIndex];
    rp.renderArea.extent = extent_;
    rp.clearValueCount = 1; rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd_, &rp, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd_);
    vkCmdEndRenderPass(cmd_);
    vkEndCommandBuffer(cmd_);

    const VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo su{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    su.waitSemaphoreCount = 1;   su.pWaitSemaphores = &acquireSem_;  su.pWaitDstStageMask = &wait;
    su.commandBufferCount = 1;   su.pCommandBuffers = &cmd_;
    su.signalSemaphoreCount = 1; su.pSignalSemaphores = &renderSem_;
    if (vkQueueSubmit(queue_, 1, &su, inFlight_) != VK_SUCCESS) {
        // inFlight_ was just reset and nothing will signal it, so the next frame's infinite wait would
        // hang the event loop. Stop drawing instead; the app falls back to the flat idle colour.
        fprintf(stderr, "[library] submit failed; closing the library view\n");
        ready_ = false;
        return action;
    }

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &renderSem_;
    pi.swapchainCount = 1;     pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &imageIndex;
    const VkResult pres = vkQueuePresentKHR(queue_, &pi);
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) needsRecreate_ = true;
    return action;
}

} // namespace prosper::frontend
