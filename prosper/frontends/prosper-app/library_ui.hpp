#pragma once
// library_ui.hpp — the game library screen: a grid of cover art the user picks a title from (#1471).
//
// This is the app's idle state. Once a guest boots, the library is torn down and the window goes back
// to presenting game frames; prosper runs one game per launch (#352), so the library never draws over a
// running title and never competes with the present path for the swapchain.
//
// It renders with Dear ImGui through the app's EXISTING Vulkan device and swapchain (third_party/imgui),
// adding only a render pass and per-image framebuffers — no second device, no second window. Cover art
// is decoded from each dump's sce_sys/icon0.png with stb_image.
//
// The selection rules live in library_nav.hpp, which is pure and unit-tested; this file owns pixels,
// resources, and event translation.

#include "game_library.hpp"

#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace prosper::frontend {

// What the user did on this frame.
struct LibraryAction {
    enum class Kind {
        none,
        open,          // launch `app0_root`
        browse,        // asked for the folder picker (no games directory, or wants another folder)
        set_games_dir, // chose `path` as the games directory; persist it and rescan
        quit,
    };
    Kind kind = Kind::none;
    std::string app0_root;   // Kind::open
    std::string path;        // Kind::set_games_dir
};

class LibraryUi {
public:
    ~LibraryUi();

    // Bring up ImGui on an already-created device/swapchain. `queue_family`/`queue` must be the same
    // ones the app presents with. Returns false (with everything released) if any Vulkan object could
    // not be created — the caller then falls back to the flat idle colour, so a driver that cannot host
    // the UI costs the user a library, not the app.
    //
    // The library owns its own command buffer, semaphores and fence and does its own acquire/present.
    // That is deliberate rather than sharing the app's: the two never run at the same time (the library
    // is the idle state and disappears the moment a guest boots), so separate sync objects keep the
    // game present path exactly as it was instead of threading a second mode through it.
    bool init(SDL_Window* window, VkInstance instance, VkPhysicalDevice phys, VkDevice device,
              uint32_t queue_family, VkQueue queue, VkSwapchainKHR swapchain,
              VkFormat swapchain_format, const std::vector<VkImage>& swapchain_images,
              VkExtent2D extent);

    // Release every Vulkan object. Safe to call twice, and safe to call without a successful init.
    void shutdown();

    bool ready() const { return ready_; }

    // Re-point at a recreated swapchain (resize, or a fullscreen transition). Keeps the ImGui context,
    // the font atlas and the decoded covers — only the framebuffers and views are rebuilt.
    bool recreate_swapchain(VkSwapchainKHR swapchain, VkFormat format,
                            const std::vector<VkImage>& images, VkExtent2D extent);

    // Hand ImGui one SDL event. Returns true when ImGui consumed it (so the caller does not also act
    // on it). The app still sees window close, resize and its own hotkeys.
    bool handle_event(const SDL_Event& ev);

    // Replace the displayed titles. Keeps the selection on the same app0 root when it is still present,
    // so a rescan does not jump the cursor somewhere else under the user.
    void set_games(std::vector<GameEntry> games, const std::string& games_dir);

    // Build and draw one frame, and present it. `status` is shown under the grid — used for the reason
    // a title failed to boot, so the message is visible in the UI rather than only on stderr.
    LibraryAction render_frame(const std::string& status);

private:
    bool create_render_target(VkFormat format, const std::vector<VkImage>& images, VkExtent2D extent);
    void destroy_render_target();
    // Decode icon0.png and upload it. Returns VK_NULL_HANDLE on any failure; the grid then draws a
    // placeholder, because a missing or corrupt icon must not remove a bootable title from the list.
    VkDescriptorSet cover_for(const GameEntry& game);
    void destroy_covers();

    struct Cover {
        VkImage        image   = VK_NULL_HANDLE;
        VkDeviceMemory memory  = VK_NULL_HANDLE;
        VkImageView    view    = VK_NULL_HANDLE;
        VkDescriptorSet set    = VK_NULL_HANDLE;
        bool           tried   = false;   // decoded once; failures are not retried every frame
    };

    SDL_Window*      window_   = nullptr;
    VkDevice         device_   = VK_NULL_HANDLE;
    VkPhysicalDevice phys_     = VK_NULL_HANDLE;
    VkQueue          queue_    = VK_NULL_HANDLE;
    uint32_t         qfamily_  = 0;
    VkFormat         format_   = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D       extent_   {};

    VkDescriptorPool pool_       = VK_NULL_HANDLE;
    VkRenderPass     renderPass_ = VK_NULL_HANDLE;
    VkCommandPool    cmdPool_    = VK_NULL_HANDLE;
    VkCommandBuffer  cmd_        = VK_NULL_HANDLE;
    VkSemaphore      acquireSem_ = VK_NULL_HANDLE;
    VkSemaphore      renderSem_  = VK_NULL_HANDLE;
    VkFence          inFlight_   = VK_NULL_HANDLE;
    VkSwapchainKHR   swapchain_  = VK_NULL_HANDLE;   // borrowed; the app owns it
    VkSampler        sampler_    = VK_NULL_HANDLE;
    std::vector<VkImageView>   views_;
    std::vector<VkFramebuffer> framebuffers_;

    std::vector<GameEntry> games_;
    std::vector<Cover>     covers_;
    std::string            gamesDir_;
    int                    selected_  = 0;
    int                    firstRow_  = 0;
    bool                   ready_     = false;
    bool                   imguiInit_ = false;
};

} // namespace prosper::frontend
