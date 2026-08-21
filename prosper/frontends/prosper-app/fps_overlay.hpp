#pragma once
// fps_overlay.hpp — the `--fps` HUD drawn over a RUNNING title (#2843).
//
// WHY THIS IS NOT LibraryUi
// -------------------------
// The library view is the app's IDLE state: it owns the whole window, does its own acquire and
// present, and is torn down the moment a guest boots so two presenters never fight over the
// swapchain (library_ui.hpp). An fps counter is the opposite shape — it must draw over a frame the
// game present path already put on the swapchain image, inside that path's command buffer and
// submit. So it shares nothing with the library beyond ImGui itself.
//
// The contract with the present path is one call:
//
//     blit into the swapchain image (leaves it TRANSFER_DST_OPTIMAL)
//     record(cmd, image_index, lines)         <-- LOAD, draw, and leave it PRESENT_SRC_KHR
//     submit
//
// The render pass loads the existing contents, so what was blitted stays; without the overlay the
// present path does the identical layout transition with a plain barrier. Nothing else about the
// present path changes, which is deliberate: this is an off-by-default annotation and it must not be
// able to alter how a title presents when it is off.
//
// NO PLATFORM BACKEND, AND THAT IS DELIBERATE
// -------------------------------------------
// ImGui_ImplSDL3 is NOT initialised here. The overlay takes no input and must not: prosper-app
// forwards SDL events to the guest pad, and an ImGui platform backend that captured keyboard or
// gamepad state would change how the game plays whenever the counter was on. DisplaySize and
// DeltaTime are set by hand instead, which is all a static HUD needs.
//
// WHAT THE OVERLAY DOES NOT TOUCH
// -------------------------------
// F9 frame grabs and every scheduled screenshot read the RENDERER's pixels (the CPU frame, or
// `gf.image` in the GPU-present path), not the swapchain image the overlay draws into. So a capture
// taken while `--fps` is on carries no annotation. If a burned-in one is wanted, that is
// `tools/screenshot --fps-overlay`, which is explicit about it.

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

struct ImGuiContext;

namespace prosper::frontend {

class FpsOverlay {
public:
    ~FpsOverlay();

    // Bring ImGui up on the app's existing device and swapchain. Returns false with everything
    // released if any object could not be created -- a driver that cannot host the HUD costs the
    // user a counter, never the frame under it.
    bool init(VkInstance instance, VkPhysicalDevice phys, VkDevice device, uint32_t queue_family,
              VkQueue queue, VkFormat swapchain_format, const std::vector<VkImage>& swapchain_images,
              VkExtent2D extent);

    // Re-point at a recreated swapchain (resize, fullscreen). Keeps the ImGui context and the font
    // atlas; only the views and framebuffers are rebuilt.
    bool recreate_swapchain(VkFormat format, const std::vector<VkImage>& images, VkExtent2D extent);

    void shutdown();
    bool ready() const { return ready_; }

    // Record the HUD into `cmd` for swapchain image `index`.
    //
    // PRECONDITION: the image is in VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and its contents are the
    // frame to annotate. POSTCONDITION: the image is in VK_IMAGE_LAYOUT_PRESENT_SRC_KHR -- i.e. this
    // REPLACES the present path's final barrier rather than adding to it. Returns false if it
    // recorded nothing, in which case the caller must still make that transition itself.
    bool record(VkCommandBuffer cmd, uint32_t index, const std::vector<std::string>& lines);

private:
    bool create_render_target(VkFormat format, const std::vector<VkImage>& images, VkExtent2D extent);
    void destroy_render_target();

    bool ready_ = false;
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    std::vector<VkImageView> views_;
    std::vector<VkFramebuffer> framebuffers_;
    ImGuiContext* context_ = nullptr;
    float scale_ = 1.0f;
};

} // namespace prosper::frontend
