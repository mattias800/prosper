#pragma once
// library_ui.hpp — the game library screen: a shadPS4-style list (cover, name, serial,
// region, version, path) the user picks a title from, plus the settings tab (#1471).
//
// This is the app's idle state. Once a guest boots, the library is torn down and the window goes back
// to presenting game frames; prosper runs one game per launch (#352), so the library never draws over a
// running title and never competes with the present path for the swapchain.
//
// It renders with Dear ImGui through the app's EXISTING Vulkan device and swapchain (third_party/imgui),
// adding only a render pass and per-image framebuffers — no second device, no second window. Cover art
// is decoded from each dump's sce_sys/icon0.png with stb_image.
//
// Filtering is pure (game_entry_matches_filter, unit-tested); this file owns pixels, resources,
// and event translation.

#include "game_library.hpp"
#include "library_descriptor_budget.hpp"
#include "library_media.hpp"

#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

// ImGui's type, declared at global scope so the member below refers to the real one rather than
// introducing prosper::frontend::ImFont. This header deliberately does not include imgui.h: only the
// implementation needs it, and the app's other translation units should not pick it up.
struct ImFont;

namespace prosper {
namespace frontend {
// The Game Log line buffer (log_ring.hpp); owned by main.cpp, only observed here. Forward
// declared so this header stays free of <mutex>/<deque>.
class LogRing;
} // namespace frontend
} // namespace prosper

namespace prosper::frontend {

// What the user did on this frame.
struct LibraryAction {
    enum class Kind {
        none,
        open,              // launch `app0_root`
        browse,            // asked for the folder picker (no games directory, or wants another folder)
        pick_game,         // asked for the folder picker for ONE game to boot right away (Ctrl+O path)
        rescan,            // re-read the games directory now
        set_games_dir,     // chose `path` as the games directory; persist it and rescan
        set_music,         // toggled launcher music to `music_on`; persist it
        set_present_mode,  // picked swapchain policy `value` (fifo|mailbox|immediate); persist + apply
        set_display_mode,  // picked guest display policy `value`; persist + apply to the next boot
        set_savedata_dir,  // picked save location `value` ("" clears); persist + apply to the next boot
        set_restore_imports, // toggled patched-import repair to `restore_on`; persist + apply to next boot
        set_volume,          // dragged the volume slider to `value` (0-100); persist + apply live
        toggle_fullscreen,   // the Full Screen toolbar button (same path as F11)
        show_in_explorer,    // reveal `app0_root` in the OS file manager
        remove_game,         // delete `app0_root` from disk after the inline confirm
        quit,
    };
    Kind kind = Kind::none;
    std::string app0_root;   // Kind::open
    std::string path;        // Kind::set_games_dir
    std::string value;       // Kind::set_present_mode/set_display_mode/set_savedata_dir
    bool music_on = true;    // Kind::set_music
    bool restore_on = false; // Kind::set_restore_imports
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

    // Seed the launcher-music preference from the persisted settings, before init() brings the media
    // layer up. PROSPER_LAUNCHER_MUSIC still overrides this, matching how every other host setting in
    // app_config.hpp is layered. Has no effect once init() has run.
    void set_music_preference(bool on) { musicToggle_ = on; }

    // Seed the settings panel with the run's EFFECTIVE host settings (flag > env > file, resolved
    // by main.cpp at startup). The panel edits these live and reports changes as actions; main.cpp
    // persists them. Seeded once before the first frame.
    void set_host_settings(const std::string& present_mode, const std::string& display_mode,
                           const std::string& savedata_dir, bool restore_imports) {
        presentMode_ = present_mode.empty() ? "fifo" : present_mode;
        displayMode_ = display_mode.empty() ? "legacy" : display_mode;
        savedataDir_ = savedata_dir;
        savedataApplied_ = savedata_dir;
        std::snprintf(savedataBuf_, sizeof savedataBuf_, "%s", savedata_dir.c_str());
        restoreToggle_ = restore_imports;
    }

    // One File > Recent games row: the root to boot, and the label to show for it. The label is
    // resolved where the filesystem is available (main.cpp at seed); the menu only displays.
    struct RecentGame {
        std::string root;
        std::string label;
    };

    // Seed File > Recent games from the persisted settings. Seeded once before the first frame;
    // boots only happen from here pre-guest, after which the menu is gone with the library.
    void set_recent(std::vector<RecentGame> recent) { recentGames_ = std::move(recent); }

    // Point the Game Log panel at main.cpp's capture ring (null: capture unavailable, the panel
    // says so). Seeded once before the first frame; the ring outlives the UI.
    void set_log_ring(const LogRing* ring) { logRing_ = ring; }

    // prosper-app's `--volume`, as a linear factor in [0,1] (#3499). It attenuates the launcher music
    // as well as the title (launcher_music_output_gain). Live once init() has run: the media layer
    // reads its gain per chunk on this same thread.
    void set_output_volume(float volume);

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

public:
    // Set when the swapchain went out of date (acquire or present said so). The app clears it by
    // recreating; without this a swapchain invalidated with no resize event leaves the view dead.
    bool needs_recreate() const { return needsRecreate_; }
    void clear_needs_recreate() { needsRecreate_ = false; }

private:
    // Consume an acquired-but-unusable frame's semaphore instead of stranding it signalled.
    void discard_acquired_frame();

    bool create_render_target(VkFormat format, const std::vector<VkImage>& images, VkExtent2D extent);
    void destroy_render_target();

    // The library's single descriptor pool, shared by ImGui's font atlas, one set per cover, and the
    // background art (#1630). Sized from the library rather than fixed at 256 (#1649).
    VkDescriptorPool create_descriptor_pool(uint32_t sets) const;
    // Make sure the pool can hold `title_count` covers plus the background allowance, growing it if not.
    // A grow releases every set prosper holds and re-initialises ImGui's Vulkan backend, which is the
    // only way to point it at a different pool without patching the vendored backend — so the caller
    // must have destroyed the covers and waited for the device first. Returns false when the pool could
    // not be grown; the view keeps working at its current capacity and the budget absorbs the shortfall.
    bool ensure_descriptor_capacity(size_t title_count);
    // Everything ImGui_ImplVulkan_Init needs, rebuilt from the members so init() and a grow agree.
    bool init_imgui_vulkan();

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

    // Upload decoded RGBA as a sampled image plus an ImGui descriptor set. Handles out via
    // the refs; null set on any failure, partial handles freed.
    VkDescriptorSet upload_rgba(int w, int h, const unsigned char* rgba, const char* what,
                                VkImage& image, VkDeviceMemory& memory, VkImageView& view);

    SDL_Window*      window_   = nullptr;
    VkInstance       instance_ = VK_NULL_HANDLE;   // kept only so a pool grow can re-init ImGui's backend
    VkDevice         device_   = VK_NULL_HANDLE;
    VkPhysicalDevice phys_     = VK_NULL_HANDLE;
    VkQueue          queue_    = VK_NULL_HANDLE;
    uint32_t         qfamily_  = 0;
    VkFormat         format_   = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D       extent_   {};
    uint32_t         imageCount_ = 2;              // swapchain image count, for ImGui's init info

    VkDescriptorPool pool_       = VK_NULL_HANDLE;
    uint32_t         poolSets_   = 0;              // maxSets the live pool was created with
    // How many of pool_'s sets prosper itself may allocate — covers here, backgrounds in LibraryMedia.
    // Consulted BEFORE ImGui_ImplVulkan_AddTexture, which would otherwise write to a VK_NULL_HANDLE set
    // when the pool is full (#1649); see library_descriptor_budget.hpp.
    //
    // MUST stay declared ABOVE media_: media_ holds a pointer to this and gives its sets back from
    // ~LibraryMedia, and members are destroyed in reverse declaration order.
    DescriptorBudget budget_;
    // PROSPER_LIBRARY_POOL_SETS caps the pool, so #1649's exhaustion path is reproducible on a small
    // library. 0 = unset. Not a tuning knob: capping it below the library deliberately costs covers.
    uint32_t         poolCap_    = 0;
    bool             poolFullWarned_ = false;   // the exhaustion notice is per run, not per title
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

    // Paint the focused title's own art behind the grid, and darken it enough that the UI stays
    // readable over arbitrary artwork (#1630).
    void draw_backdrop();

    // The settings tab content: host settings a terminal-free launch could never reach. Drawn
    // inline in the main window (a floating dialog over a game library is the wrong shape — the
    // library IS the window, so settings is one of its tabs, not a popup above it).
    void draw_settings_content(LibraryAction& action);
    // Read-only keyboard map. The mapping itself lives in keyboard_pad_map.hpp — this table
    // mirrors its documented layout, so update both when the mapping changes.
    void draw_controls_content();
    // Right-click detector for one row item: arms the menu below. Called at each clickable
    // item of the row so the whole row answers.
    void note_row_right_click(int fi);
    // The row menu itself (Play, Show in Explorer, Remove from disk with confirm), drawn once
    // per frame after the table at window scope, where OpenPopup/BeginPopup always agree.
    void draw_row_menu(LibraryAction& action, int shown);
    int contextFi_ = -1;      // row the menu was armed from
    bool contextArmed_ = false;

    // Which view is showing. Games, settings and controls share the menu bar; only one draws.
    enum class LibraryTab { games, settings, controls };

    ImFont*                boldFont_ = nullptr;    // section headers; null while on the bitmap fallback

    LibraryMedia           media_;
    bool                   musicToggle_ = true;   // mirrors the persisted setting for the in-UI switch
    float                  outputVolume_ = 1.0f;  // --volume, applied on top of the music's own level
    float                  musicLevel_ = kDefaultMusicGain;  // the launcher's own mix level under --volume
    int                    volumePercent_ = 100;  // toolbar slider position, mirrors outputVolume_

    // The settings tab (same exe, same window). The radio state mirrors the run's effective
    // policy; changing one emits an action main.cpp persists and applies live to the not-yet-
    // booted guest. The save path edits a buffer and applies explicitly, so half-typed paths
    // never reach the config; `savedataApplied_` is what the last Apply wrote.
    LibraryTab             tab_ = LibraryTab::games;
    bool                   restoreToggle_ = false;  // mirrors the repair opt-in for the in-UI switch
    std::string            presentMode_ = "fifo";
    std::string            displayMode_ = "legacy";
    std::string            savedataDir_;
    std::string            savedataApplied_;
    char                   savedataBuf_[1024] = {};

    // PROSPER_LIBRARY_STATS=1: per-frame timing for the library view, reported at shutdown. A mean
    // cannot detect a stutter — a 40 ms frame among 5 ms ones averages away — so what is kept is the
    // maximum and the count over one and two vsync intervals, which is what "is the UI smooth" means.
    bool                   stats_        = false;
    uint64_t               frameCount_   = 0;
    uint64_t               frameOver16_  = 0;
    uint64_t               frameOver33_  = 0;
    double                 frameTotalMs_ = 0.0;
    double                 frameMaxMs_   = 0.0;
    uint64_t               frameMaxIndex_ = 0;   // which frame was the worst — startup or a load?
    uint64_t               lastFrameNs_  = 0;
    bool                   statsReported_ = false;
    // For percentiles. Bounded, and recording simply STOPS at the cap: the reported percentiles
    // then describe the first kMaxFrameSamples frames, which is stated rather than disguised.
    static constexpr size_t kMaxFrameSamples = 200000;
    std::vector<double>    frameSamples_;

    // Rebuild the visible rows: every game matching the search box, as indices into games_.
    // Selection is a position in this list, reset whenever it is rebuilt.
    void apply_filter();

    std::vector<GameEntry> games_;
    std::vector<Cover>     covers_;
    std::vector<int>       filtered_;
    // Row hovered last frame. The highlight is painted at row start, but hover is only
    // knowable after the row's items — so it trails by one frame, which is imperceptible and
    // always spans the whole row (unlike a Selectable's own cell-sized hover paint).
    int                    hovered_ = -1;
    std::vector<RecentGame> recentGames_;
    const LogRing*         logRing_ = nullptr;  // observed only; owned by main.cpp
    bool                   logFollow_ = true;
    std::string            pendingRemoveRoot_;  // delete-confirm modal open for this root, "" = none
    std::string            gamesDir_;
    char                   filterBuf_[256] = {};
    std::string            filterApplied_;
    int                    selected_  = 0;
    bool                   ready_     = false;
    bool                   imguiCtx_   = false;   // unwound independently so a part-failed init leaks nothing
    bool                   sdlInit_    = false;
    bool                   vulkanInit_ = false;
    bool                   needsRecreate_ = false;
};

} // namespace prosper::frontend
