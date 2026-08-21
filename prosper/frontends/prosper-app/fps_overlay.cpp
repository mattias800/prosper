// fps_overlay.cpp — see fps_overlay.hpp.
#include "fps_overlay.hpp"

#include "imgui.h"
#include "imgui_impl_vulkan.h"

#include "gpu/execute/gpu_execute.hpp"   // shared_present_submit_mutex

#include <algorithm>
#include <cstdio>
#include <mutex>

namespace prosper::frontend {
namespace {

// ImGui_ImplVulkan_CreateFontsTexture() returns `true` unconditionally: every one of its ~15 error
// sites goes through CheckVkResultFn and then falls through to `return true`
// (imgui_impl_vulkan.cpp:194). So a `if (!ImGui_ImplVulkan_CreateFontsTexture())` guard is DEAD
// CODE, and a real upload failure would leave this class believing the atlas exists.
//
// That is not a cosmetic dead branch. On failure `io.Fonts->SetTexID()` is never reached, so
// `bd->FontTexture.DescriptorSet` stays null -- and `ImGui_ImplVulkan_NewFrame()` retries the
// creation on EVERY recorded frame (imgui_impl_vulkan.cpp:1201), which is precisely the unlocked
// vkQueueSubmit on a shared queue that building the atlas in init() exists to prevent. A silent
// failure would therefore re-open the race rather than merely losing the counter.
//
// The callback has no user pointer, so the failure is latched here. `font_upload_failed` is cleared
// immediately before the call and read immediately after, on one thread, which is the whole extent
// of the contract.
bool g_overlay_vk_failed = false;

void overlay_vk_result(VkResult r) {
    if (r == VK_SUCCESS) return;
    g_overlay_vk_failed = true;
    std::fprintf(stderr, "[fps] Vulkan error %d in the overlay\n", (int)r);
}

// Build (or rebuild) the font atlas, and actually establish that it worked.
//
// Two independent checks, because neither alone is sufficient: the latched VkResult catches a driver
// error, and the texture id catches any path that returned without uploading. The id is the real
// POSTCONDITION -- it is what NewFrame() tests before deciding to retry -- so checking it is what
// guarantees the lazy path stays closed.
bool upload_font_atlas() {
    g_overlay_vk_failed = false;
    ImGui_ImplVulkan_CreateFontsTexture();
    if (g_overlay_vk_failed) return false;
    return ImGui::GetIO().Fonts->TexID != 0;
}

// ImGui's default font is rasterized at 13 px. At 4K that is unreadable, so the atlas is built at a
// size scaled to the swapchain rather than magnified afterwards -- a scaled 13 px atlas is blurry,
// and a HUD that reports a measurement has to be legible enough to read off a screenshot.
float scale_for(VkExtent2D extent) {
    return std::max(1.0f, static_cast<float>(extent.height) / 720.0f);
}

// Restores whatever context was current, so the overlay can never leave ImGui pointing at its own
// context for somebody else's code. The library view and the overlay never coexist today (the
// library is torn down before a guest boots); this makes that a property of the code rather than of
// the current call order.
class ScopedContext {
public:
    explicit ScopedContext(ImGuiContext* ctx) : previous_(ImGui::GetCurrentContext()) {
        ImGui::SetCurrentContext(ctx);
    }
    ~ScopedContext() { ImGui::SetCurrentContext(previous_); }
private:
    ImGuiContext* previous_;
};

} // namespace

FpsOverlay::~FpsOverlay() { shutdown(); }

bool FpsOverlay::init(VkInstance instance, VkPhysicalDevice phys, VkDevice device,
                      uint32_t queue_family, VkQueue queue, VkFormat swapchain_format,
                      const std::vector<VkImage>& swapchain_images, VkExtent2D extent) {
    if (ready_) return true;
    if (!device || swapchain_images.empty() || !extent.width || !extent.height) return false;
    device_ = device;

    // ImGui_ImplVulkan wants a pool it can allocate the font texture's descriptor from. The overlay
    // owns exactly one texture, so this is deliberately tiny.
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = 4;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &size;
    if (vkCreateDescriptorPool(device_, &dpi, nullptr, &pool_) != VK_SUCCESS) {
        std::fprintf(stderr, "[fps] descriptor pool creation failed; the counter is off\n");
        shutdown();
        return false;
    }

    if (!create_render_target(swapchain_format, swapchain_images, extent)) {
        std::fprintf(stderr, "[fps] render target creation failed; the counter is off\n");
        shutdown();
        return false;
    }

    context_ = ImGui::CreateContext();
    ScopedContext current(context_);
    ImGuiIO& io = ImGui::GetIO();
    // No .ini file: the HUD has no state worth persisting, and writing one into the user's cwd from
    // a game frontend would be a surprise.
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NoMouse | ImGuiConfigFlags_NoMouseCursorChange;
    scale_ = scale_for(extent);
    ImFontConfig font{};
    font.SizePixels = 13.0f * scale_;
    io.Fonts->AddFontDefault(&font);
    ImGui::StyleColorsDark();
    ImGui::GetStyle().ScaleAllSizes(scale_);

    ImGui_ImplVulkan_InitInfo vi{};
    vi.Instance = instance;
    vi.PhysicalDevice = phys;
    vi.Device = device_;
    vi.QueueFamily = queue_family;
    vi.Queue = queue;
    vi.DescriptorPool = pool_;
    vi.RenderPass = renderPass_;
    vi.MinImageCount = static_cast<uint32_t>(swapchain_images.size());
    vi.ImageCount = static_cast<uint32_t>(swapchain_images.size());
    vi.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    vi.CheckVkResultFn = overlay_vk_result;
    if (!ImGui_ImplVulkan_Init(&vi)) {
        std::fprintf(stderr, "[fps] ImGui Vulkan backend init failed; the counter is off\n");
        shutdown();
        return false;
    }

    // Build the font atlas HERE, under the present submit mutex, rather than letting
    // ImGui_ImplVulkan_NewFrame() do it lazily on the first recorded frame.
    //
    // This is not tidiness. `ImGui_ImplVulkan_NewFrame` calls `ImGui_ImplVulkan_CreateFontsTexture`
    // when no font descriptor exists yet (imgui_impl_vulkan.cpp:1201), and that function ends in
    // `vkQueueSubmit(v->Queue, …)` followed by `vkQueueWaitIdle(v->Queue)` (:819, :822). Under
    // PROSPER_APP_GPU_PRESENT `v->Queue` is the queue the app ADOPTED from the live renderer
    // (main.cpp:574) and may alias the render queue, which is why both present paths serialize their
    // submits through this mutex. `record()` runs while the present command buffer is being built,
    // BEFORE that lock is taken -- so a lazy upload there would submit to a queue the renderer
    // thread can be using concurrently, and vkQueueSubmit requires external synchronisation of the
    // VkQueue. Doing it once, here, under the lock, closes that window; `NewFrame` then finds the
    // descriptor already present and submits nothing for the rest of the process.
    //
    // Locked unconditionally rather than only when the queue is shared: init happens once, off the
    // hot path, and a lock whose necessity depends on a flag read elsewhere is the kind of
    // conditional correctness that stops being true when the flag moves.
    {
        std::lock_guard<std::mutex> lk(gpu::shared_present_submit_mutex());
        if (!upload_font_atlas()) {
            std::fprintf(stderr, "[fps] font atlas upload failed; the counter is off\n");
            ImGui_ImplVulkan_Shutdown();
            shutdown();
            return false;
        }
    }
    ready_ = true;
    std::fprintf(stderr, "[fps] overlay on (%ux%u, %.0fx text scale)\n",
                 extent.width, extent.height, scale_);
    return true;
}

bool FpsOverlay::create_render_target(VkFormat format, const std::vector<VkImage>& images,
                                      VkExtent2D extent) {
    format_ = format;
    extent_ = extent;

    // LOAD, not CLEAR: the frame is already on this image and the overlay draws OVER it. The
    // initial layout is the one the present path's blit leaves behind, and the final layout is the
    // one the present path would otherwise have transitioned to itself -- so the render pass
    // replaces that barrier exactly.
    VkAttachmentDescription color{};
    color.format = format_;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;
    // The blit that produced these contents is a TRANSFER write; the load must not begin until it
    // has completed, or the overlay would composite over an undefined image.
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dep.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 1; rpi.pAttachments = &color;
    rpi.subpassCount = 1;    rpi.pSubpasses = &sub;
    rpi.dependencyCount = 1; rpi.pDependencies = &dep;
    if (renderPass_ == VK_NULL_HANDLE &&
        vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_) != VK_SUCCESS)
        return false;

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

void FpsOverlay::destroy_render_target() {
    for (VkFramebuffer fb : framebuffers_) if (fb) vkDestroyFramebuffer(device_, fb, nullptr);
    for (VkImageView v : views_) if (v) vkDestroyImageView(device_, v, nullptr);
    framebuffers_.clear();
    views_.clear();
}

bool FpsOverlay::recreate_swapchain(VkFormat format, const std::vector<VkImage>& images,
                                    VkExtent2D extent) {
    if (!ready_) return false;
    // The caller has already idled the device before recreating the swapchain, so the old views are
    // not in flight. The format cannot change under a swapchain recreation in this app; if it ever
    // does, the render pass is incompatible and the overlay turns itself off rather than drawing
    // through a mismatched pass.
    if (format != format_) {
        std::fprintf(stderr, "[fps] swapchain format changed; the counter is off\n");
        shutdown();
        return false;
    }
    destroy_render_target();
    if (!create_render_target(format, images, extent)) { shutdown(); return false; }

    // Re-rasterize the font when the swapchain changes size class. Without this a run that starts
    // windowed at 720p and goes fullscreen 4K keeps a 13 px HUD on a 2160 px surface, which defeats
    // the point of a counter you are meant to be able to read off a screenshot.
    //
    // Safe to destroy and re-upload here because the caller has already drained the device before
    // recreating the swapchain (main.cpp's swapchainDirty block calls vkDeviceWaitIdle), so nothing
    // in flight still references the old atlas. The upload takes the present submit mutex for the
    // same reason init() does: it ends in a vkQueueSubmit + vkQueueWaitIdle on a queue that may
    // alias the renderer's.
    const float wanted = scale_for(extent);
    if (wanted != scale_) {
        ScopedContext current(context_);
        ImGui_ImplVulkan_DestroyFontsTexture();
        ImGuiIO& io = ImGui::GetIO();
        io.Fonts->Clear();
        ImFontConfig font{};
        font.SizePixels = 13.0f * wanted;
        io.Fonts->AddFontDefault(&font);
        // Rebuild the style from DEFAULTS and scale once, rather than applying a ratio to the
        // already-scaled one. ImGuiStyle::ScaleAllSizes truncates every field (imgui.cpp:
        // `ImTrunc(x * scale_factor)`), so scaling down and back up does not round-trip: a 1 px
        // border scaled to 0 is 0 for the rest of the process, and a window that goes 4K -> 720p ->
        // 4K would permanently lose its padding and borders.
        ImGui::GetStyle() = ImGuiStyle();
        ImGui::StyleColorsDark();
        ImGui::GetStyle().ScaleAllSizes(wanted);
        scale_ = wanted;
        std::lock_guard<std::mutex> lk(gpu::shared_present_submit_mutex());
        if (!upload_font_atlas()) {
            std::fprintf(stderr, "[fps] font atlas rebuild failed after a resize; the counter is off\n");
            shutdown();
            return false;
        }
    }
    return true;
}

void FpsOverlay::shutdown() {
    if (context_) {
        ScopedContext current(context_);
        if (ready_) ImGui_ImplVulkan_Shutdown();
        ImGui::DestroyContext(context_);
        context_ = nullptr;
    }
    ready_ = false;
    if (device_) {
        destroy_render_target();
        if (renderPass_) { vkDestroyRenderPass(device_, renderPass_, nullptr); renderPass_ = VK_NULL_HANDLE; }
        if (pool_) { vkDestroyDescriptorPool(device_, pool_, nullptr); pool_ = VK_NULL_HANDLE; }
    }
    device_ = VK_NULL_HANDLE;
}

bool FpsOverlay::record(VkCommandBuffer cmd, uint32_t index, const std::vector<std::string>& lines) {
    if (!ready_ || lines.empty() || index >= framebuffers_.size()) return false;

    ScopedContext current(context_);
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(extent_.width), static_cast<float>(extent_.height));
    // The HUD has no animation, so the exact delta does not matter -- but it must be positive, or
    // ImGui asserts in a debug build.
    io.DeltaTime = 1.0f / 60.0f;

    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    const float margin = 12.0f * scale_;
    ImGui::SetNextWindowPos(ImVec2(margin, margin), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.55f);
    ImGui::Begin("##prosper-fps", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                 ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoMove);
    for (const std::string& line : lines) ImGui::TextUnformatted(line.c_str());
    ImGui::End();
    ImGui::Render();

    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = renderPass_;
    rp.framebuffer = framebuffers_[index];
    rp.renderArea.extent = extent_;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
    return true;
}

} // namespace prosper::frontend
