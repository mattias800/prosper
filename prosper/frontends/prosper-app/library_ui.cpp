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

#include <cstdio>
#include <cstring>

namespace prosper::frontend {
namespace {

constexpr float kCoverSize   = 160.0f;   // cover art is square (icon0.png is 512x512 in every dump)
constexpr float kCellPadding = 24.0f;
constexpr float kLabelHeight = 42.0f;    // two lines of wrapped title under each cover
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

// One frame's worth of freshly-pressed pad inputs. Edges only: holding a direction must not sweep the
// whole library, matching how the keyboard repeat is handled by ImGui::IsKeyPressed.
LibraryUi::PadEdge LibraryUi::poll_pad_edges() {
    PadEdge edge;
    bool down[PadEdge::kCount] = {};
    int count = 0;
    SDL_JoystickID* pads = SDL_GetGamepads(&count);
    if (pads) {
        for (int i = 0; i < count; i++) {
            SDL_Gamepad* pad = SDL_GetGamepadFromID(pads[i]);
            if (!pad) continue;   // not opened by us; the app's pad backend owns the guest's device
            down[0] = down[0] || SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_DPAD_LEFT);
            down[1] = down[1] || SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_DPAD_RIGHT);
            down[2] = down[2] || SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_DPAD_UP);
            down[3] = down[3] || SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_DPAD_DOWN);
            down[4] = down[4] || SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_SOUTH);   // Cross
        }
        SDL_free(pads);
    }
    edge.left    = down[0] && !padDown_[0];
    edge.right   = down[1] && !padDown_[1];
    edge.up      = down[2] && !padDown_[2];
    edge.down    = down[3] && !padDown_[3];
    edge.confirm = down[4] && !padDown_[4];
    for (int i = 0; i < PadEdge::kCount; i++) padDown_[i] = down[i];
    return edge;
}

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
    if (device_) vkDeviceWaitIdle(device_);
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

LibraryAction LibraryUi::render_frame(const std::string& status) {
    LibraryAction action;
    if (!ready_) return action;

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("prosper", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                 ImGuiWindowFlags_NoNavFocus);

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
        const PadEdge pad = poll_pad_edges();
        if (ImGui::Button("Choose folder...") ||
            ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter) ||
            ImGui::IsKeyPressed(ImGuiKey_Space) || pad.confirm)
            action.kind = LibraryAction::Kind::browse;
        ImGui::SameLine();
        ImGui::TextDisabled("or press Enter, or drop a game folder on this window");
    } else {
        const float avail = ImGui::GetContentRegionAvail().x;
        const int columns = library_columns_for_width(avail, kCellWidth);
        const int count = static_cast<int>(games_.size());

        // Keyboard/controller selection. ImGui's own nav does not know about the grid, so movement is
        // decided by the unit-tested rules and the scroll follows the selection.
        // ImGui nav is off (see init), which also silences the SDL3 backend's gamepad feed — so the
        // pad is polled directly here rather than through ImGuiKey_Gamepad*, which would never be set.
        const PadEdge pad = poll_pad_edges();
        LibraryNavKey key = LibraryNavKey::none;
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) || pad.left)   key = LibraryNavKey::left;
        else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) || pad.right) key = LibraryNavKey::right;
        else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) || pad.up)  key = LibraryNavKey::up;
        else if (ImGui::IsKeyPressed(ImGuiKey_DownArrow) || pad.down) key = LibraryNavKey::down;
        else if (ImGui::IsKeyPressed(ImGuiKey_Home))     key = LibraryNavKey::home;
        else if (ImGui::IsKeyPressed(ImGuiKey_End))      key = LibraryNavKey::end;
        else if (ImGui::IsKeyPressed(ImGuiKey_PageUp))   key = LibraryNavKey::page_up;
        else if (ImGui::IsKeyPressed(ImGuiKey_PageDown)) key = LibraryNavKey::page_down;

        const float gridHeight = ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing() * 2.0f;
        const int rowsPerPage = static_cast<int>(gridHeight / kCellHeight);
        selected_ = library_nav_apply(key, selected_, count, columns, rowsPerPage);
        firstRow_ = library_scroll_row_for(selected_, columns, rowsPerPage, firstRow_);

        if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter) ||
            ImGui::IsKeyPressed(ImGuiKey_Space) || pad.confirm) {
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
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n%s", game.title_name.c_str(),
                                                          game.app0_root.c_str());

            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + kCoverSize);
            ImGui::TextUnformatted(game.title_name.c_str());
            ImGui::PopTextWrapPos();

            ImGui::PopID();
            ImGui::EndGroup();
        }
        ImGui::EndChild();

        ImGui::Separator();
        ImGui::Text("%d game%s", count, count == 1 ? "" : "s");
        ImGui::SameLine();
        if (ImGui::Button("Change folder...")) action.kind = LibraryAction::Kind::browse;
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
        needsRecreate_ = (acq == VK_ERROR_OUT_OF_DATE_KHR);
        return action;
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) return action;   // hard error: nothing signalled
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
