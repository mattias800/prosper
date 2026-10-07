// library_ui.cpp — see library_ui.hpp. Draws the library list with Dear ImGui on the app's existing
// Vulkan device, and decodes cover art with stb_image (#1471).
#include "library_ui.hpp"
#include "list_nav.hpp"   // pure keyboard gate: typing in the search box never drives the list
#include "gpu/diagnostics/gpu_memory_budget_vk.hpp"  // #3533: count what we hold on each heap

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG            // the only format sce_sys ships for icons; keeps the surface narrow
#define STBI_NO_STDIO            // we hand it bytes we already read, so it never opens files itself
#define STBI_NO_FAILURE_STRINGS
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace prosper::frontend {
namespace {

// Table thumbnails sample the square icon0.png down, so they cannot show resampling artefacts.
constexpr float kThumbSize = 56.0f;

// Toolbar icons, drawn as vectors: no font file, no license baggage, crisp at any scale.
// Each is centered on `c` within roughly radius `s`.
enum class ToolbarIcon { play, folder, rescan, fullscreen, keyboard, music_on, music_off };
void draw_toolbar_icon(ImDrawList* dl, ImVec2 c, float s, ToolbarIcon icon, ImU32 col) {
    const float t = s / 11.0f;   // icon units: every glyph lives in an 11-unit box
    switch (icon) {
    case ToolbarIcon::play: {
        // Right-pointing triangle.
        dl->AddTriangleFilled(ImVec2(c.x - 4 * t, c.y - 6 * t), ImVec2(c.x - 4 * t, c.y + 6 * t),
                              ImVec2(c.x + 6 * t, c.y), col);
        break;
    }
    case ToolbarIcon::folder: {
        // Outline body with a tab nub: one stroke weight throughout, no fill tricks.
        dl->AddRect(ImVec2(c.x - 8 * t, c.y - 3 * t), ImVec2(c.x + 8 * t, c.y + 6 * t), col,
                    1.5f * t, 0, 2 * t);
        dl->AddLine(ImVec2(c.x - 8 * t, c.y - 3 * t), ImVec2(c.x - 8 * t, c.y - 6 * t), col,
                    2 * t);
        dl->AddLine(ImVec2(c.x - 8 * t, c.y - 6 * t), ImVec2(c.x - 1 * t, c.y - 6 * t), col,
                    2 * t);
        dl->AddLine(ImVec2(c.x - 1 * t, c.y - 6 * t), ImVec2(c.x - 1 * t, c.y - 3 * t), col,
                    2 * t);
        break;
    }
    case ToolbarIcon::rescan: {
        // Circular arrow. The head sits at the arc's end, pointing along the tangent, so the
        // eye reads rotation rather than a blob.
        const float end = 5.0f;
        dl->PathArcTo(c, 6 * t, 0.8f, end, 24);
        dl->PathStroke(col, 0, 2.2f * t);
        const ImVec2 dir(std::cos(end), std::sin(end));    // radial
        const ImVec2 tan(-dir.y, dir.x);                   // direction of travel
        const ImVec2 base(c.x + dir.x * 6 * t, c.y + dir.y * 6 * t);
        dl->AddTriangleFilled(ImVec2(base.x + tan.x * 5.5f * t, base.y + tan.y * 5.5f * t),
                              ImVec2(base.x - tan.x * 0.5f * t + dir.x * 3.5f * t,
                                     base.y - tan.y * 0.5f * t + dir.y * 3.5f * t),
                              ImVec2(base.x - tan.x * 0.5f * t - dir.x * 3.5f * t,
                                     base.y - tan.y * 0.5f * t - dir.y * 3.5f * t),
                              col);
        break;
    }
    case ToolbarIcon::fullscreen: {
        // Four corner brackets pushing outward.
        const float e = 7 * t, l = 4 * t, w = 2 * t;
        dl->AddLine(ImVec2(c.x - e, c.y - e + l), ImVec2(c.x - e, c.y - e), col, w);
        dl->AddLine(ImVec2(c.x - e, c.y - e), ImVec2(c.x - e + l, c.y - e), col, w);
        dl->AddLine(ImVec2(c.x + e - l, c.y - e), ImVec2(c.x + e, c.y - e), col, w);
        dl->AddLine(ImVec2(c.x + e, c.y - e), ImVec2(c.x + e, c.y - e + l), col, w);
        dl->AddLine(ImVec2(c.x - e, c.y + e - l), ImVec2(c.x - e, c.y + e), col, w);
        dl->AddLine(ImVec2(c.x - e, c.y + e), ImVec2(c.x - e + l, c.y + e), col, w);
        dl->AddLine(ImVec2(c.x + e - l, c.y + e), ImVec2(c.x + e, c.y + e), col, w);
        dl->AddLine(ImVec2(c.x + e, c.y + e), ImVec2(c.x + e, c.y + e - l), col, w);
        break;
    }
    case ToolbarIcon::keyboard: {
        // Key well with three key rows.
        dl->AddRect(ImVec2(c.x - 9 * t, c.y - 5 * t), ImVec2(c.x + 9 * t, c.y + 5 * t), col,
                    2 * t, 0, 1.8f * t);
        for (int row = 0; row < 3; row++)
            for (int k = 0; k < 5; k++)
                dl->AddRectFilled(ImVec2(c.x - 7 * t + k * 3 * t, c.y - 3 * t + row * 2.6f * t),
                                  ImVec2(c.x - 5 * t + k * 3 * t, c.y - 1.4f * t + row * 2.6f * t),
                                  col);
        break;
    }
    case ToolbarIcon::music_on:
    case ToolbarIcon::music_off: {
        // One beamed note, drawn large: the stem lands through the head's middle, so the two
        // are one mark at any size. The off state adds the universal slash.
        dl->AddLine(ImVec2(c.x + 2 * t, c.y - 7 * t), ImVec2(c.x + 2 * t, c.y + 4 * t), col,
                    2.6f * t);
        dl->AddEllipseFilled(ImVec2(c.x + 1 * t, c.y + 4 * t), ImVec2(4 * t, 3.2f * t),
                             col);
        dl->AddTriangleFilled(ImVec2(c.x + 2 * t, c.y - 7 * t),
                              ImVec2(c.x + 7.5f * t, c.y - 4.5f * t),
                              ImVec2(c.x + 2 * t, c.y - 1.5f * t), col);
        if (icon == ToolbarIcon::music_off)
            dl->AddLine(ImVec2(c.x - 9 * t, c.y - 7 * t), ImVec2(c.x + 9 * t, c.y + 7 * t), col,
                        2.4f * t);
        break;
    }
    }
}

// An icon-over-label toolbar button. Behaves like a button (hover/active painting included);
// `active` pins the active paint for toggle state.
bool icon_button(const char* id, ToolbarIcon icon, const char* label, bool active = false) {
    ImGui::BeginGroup();
    ImGui::PushID(id);
    const ImVec2 size(64.0f, 52.0f);
    const bool clicked = ImGui::InvisibleButton("btn", size);
    const bool hovered = ImGui::IsItemHovered();
    const bool pressed = ImGui::IsItemActive();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p0 = ImGui::GetItemRectMin();
    if (hovered || pressed || active)
        dl->AddRectFilled(p0, ImVec2(p0.x + size.x, p0.y + size.y),
                          ImGui::GetColorU32(pressed || active ? ImGuiCol_ButtonActive
                                                              : ImGuiCol_ButtonHovered),
                          6.0f);
    const ImU32 glyph =
        ImGui::GetColorU32((hovered || pressed || active) ? ImGuiCol_Text : ImGuiCol_TextDisabled);
    draw_toolbar_icon(dl, ImVec2(p0.x + size.x * 0.5f, p0.y + 20.0f), 11.0f, icon, glyph);
    const ImVec2 textSize = ImGui::CalcTextSize(label);
    dl->AddText(ImVec2(p0.x + (size.x - textSize.x) * 0.5f, p0.y + 34.0f), glyph, label);
    ImGui::PopID();
    ImGui::EndGroup();
    return clicked;
}

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

// One combined-image-sampler set per texture, and FREE_DESCRIPTOR_SET because covers and backgrounds are
// individually released (ImGui_ImplVulkan_RemoveTexture calls vkFreeDescriptorSets).
VkDescriptorPool LibraryUi::create_descriptor_pool(uint32_t sets) const {
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sets};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = sets;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &size;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(device_, &dpi, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
    return pool;
}

bool LibraryUi::init_imgui_vulkan() {
    ImGui_ImplVulkan_InitInfo vi{};
    vi.Instance = instance_;
    vi.PhysicalDevice = phys_;
    vi.Device = device_;
    vi.QueueFamily = qfamily_;
    vi.Queue = queue_;
    vi.DescriptorPool = pool_;
    vi.RenderPass = renderPass_;
    vi.MinImageCount = imageCount_;
    vi.ImageCount = imageCount_;
    vi.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    vi.CheckVkResultFn = imgui_vk_result;
    return ImGui_ImplVulkan_Init(&vi);
}

bool LibraryUi::init(SDL_Window* window, VkInstance instance, VkPhysicalDevice phys, VkDevice device,
                     uint32_t queue_family, VkQueue queue, VkSwapchainKHR swapchain,
                     VkFormat swapchain_format, const std::vector<VkImage>& swapchain_images,
                     VkExtent2D extent) {
    window_ = window; instance_ = instance; device_ = device; phys_ = phys; queue_ = queue;
    qfamily_ = queue_family;
    swapchain_ = swapchain;
    imageCount_ = static_cast<uint32_t>(swapchain_images.size());

    // PROSPER_LIBRARY_POOL_SETS forces a small pool so #1649's exhaustion can be reproduced without
    // owning 250 games. Deliberately a HARD cap that a grow cannot lift: the point of the knob is to
    // reach the path where a cover has no descriptor left, and see that it costs that cover and nothing
    // else. Unset (the normal case) leaves the sizing entirely to the library's title count.
    if (const char* cap = SDL_getenv("PROSPER_LIBRARY_POOL_SETS")) {
        const long v = std::strtol(cap, nullptr, 10);
        if (v > 0) {
            poolCap_ = static_cast<uint32_t>(v);
            fprintf(stderr, "[library] PROSPER_LIBRARY_POOL_SETS=%u: descriptor pool capped\n", poolCap_);
        }
    }

    // Sized from the library, not fixed (#1649). The games are not scanned yet at this point, so this is
    // the floor; the first set_games() grows it to fit the real title count.
    poolSets_ = library_descriptor_pool_sets(0);
    if (poolCap_ && poolSets_ > poolCap_) poolSets_ = poolCap_;
    pool_ = create_descriptor_pool(poolSets_);
    if (pool_ == VK_NULL_HANDLE) {
        fprintf(stderr, "[library] descriptor pool creation failed\n");
        shutdown();
        return false;
    }
    budget_.reset(library_texture_budget(poolSets_));

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
    // ImGui's own keyboard/gamepad nav is deliberately NOT enabled: this screen drives selection
    // itself (Up/Down/Home/End plus Enter, gated by the pure list_nav.hpp rules), and letting both
    // run means the arrow keys move a widget focus as well as the selection, and Enter activates
    // whatever widget that focus landed on rather than launching the highlighted game.
    io.ConfigFlags &= ~(ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad);
    ImGui::StyleColorsDark();
    // Console-shelf polish: roomier touch targets and one accent family. The library window sits
    // flush against the OS window edge, so rounded content corners would frame a square window
    // with gaps — widgets keep their rounding, windows do not.
    {
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding = 0.0f;
        style.FrameRounding = 6.0f;
        style.GrabRounding = 6.0f;
        style.WindowPadding = ImVec2(16, 14);
        style.FramePadding = ImVec2(10, 6);
        style.ItemSpacing = ImVec2(10, 8);
        style.ItemInnerSpacing = ImVec2(8, 6);
        ImVec4* c = style.Colors;
        c[ImGuiCol_Button] = ImVec4(0.10f, 0.32f, 0.68f, 1.00f);
        c[ImGuiCol_ButtonHovered] = ImVec4(0.16f, 0.42f, 0.82f, 1.00f);
        c[ImGuiCol_ButtonActive] = ImVec4(0.07f, 0.25f, 0.55f, 1.00f);
        c[ImGuiCol_Header] = ImVec4(0.10f, 0.32f, 0.68f, 1.00f);
        c[ImGuiCol_HeaderHovered] = ImVec4(0.16f, 0.42f, 0.82f, 1.00f);
        c[ImGuiCol_CheckMark] = ImVec4(0.45f, 0.75f, 1.00f, 1.00f);
        c[ImGuiCol_FrameBg] = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
        c[ImGuiCol_FrameBgHovered] = ImVec4(0.18f, 0.20f, 0.24f, 1.00f);
    }

    // Type comes from the host's own UI font, baked at the exact sizes it is drawn at — the same
    // crispness rule the old 2x bitmap title font followed. A system font is data, not code:
    // nothing guest-controlled touches it, and when none is found the bitmap default keeps every
    // label readable, only less pretty.
    {
        const char* regular = nullptr;
        const char* bold = nullptr;
#ifdef _WIN32
        static const char kWinRegular[] = "C:\\Windows\\Fonts\\segoeui.ttf";
        static const char kWinBold[] = "C:\\Windows\\Fonts\\segoeuib.ttf";
        FILE* probe = std::fopen(kWinRegular, "rb");
        if (probe) {
            std::fclose(probe);
            regular = kWinRegular;
            probe = std::fopen(kWinBold, "rb");
            if (probe) { std::fclose(probe); bold = kWinBold; }
        }
#else
        static const struct { const char* regular; const char* bold; } kLinuxFonts[] = {
            {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"},
            {"/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
             "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"},
            {"/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"},
            {"/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
             "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"},
        };
#ifdef __APPLE__
        static const struct { const char* regular; const char* bold; } kMacFonts[] = {
            {"/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Helvetica.ttc"},
        };
#endif
        for (const auto& candidate : kLinuxFonts) {
            FILE* probe = std::fopen(candidate.regular, "rb");
            if (!probe) continue;
            std::fclose(probe);
            regular = candidate.regular;
            probe = std::fopen(candidate.bold, "rb");
            if (probe) { std::fclose(probe); bold = candidate.bold; }
            break;
        }
#ifdef __APPLE__
        if (!regular) {
            for (const auto& candidate : kMacFonts) {
                FILE* probe = std::fopen(candidate.regular, "rb");
                if (!probe) continue;
                std::fclose(probe);
                regular = candidate.regular;
                bold = candidate.bold;
                break;
            }
        }
#endif
#endif
        if (regular) {
            // 16 px UI text and 16 px bold headers. All-or-nothing: a half-built atlas is
            // discarded so the fallback below owns everything. A bold path identical to the
            // regular one (macOS Helvetica.ttc, whose faces need an index this loader does not
            // pick) is skipped rather than baking the same font twice.
            ImFont* ui = io.Fonts->AddFontFromFileTTF(regular, 16.0f);
            if (ui && bold && bold != regular)
                boldFont_ = io.Fonts->AddFontFromFileTTF(bold, 16.0f);
            if (!ui) io.Fonts->Clear();
        }
    }
    if (io.Fonts->Fonts.empty()) {
        // No system font: the bitmap default at 13 px. Scaling it at draw time would blur, so the
        // list simply draws smaller rather than rescaling what it has.
        io.Fonts->AddFontDefault();
    }

    if (!ImGui_ImplSDL3_InitForVulkan(window_)) { fprintf(stderr, "[library] SDL3 backend init failed\n"); shutdown(); return false; }
    sdlInit_ = true;
    if (!init_imgui_vulkan()) { fprintf(stderr, "[library] Vulkan backend init failed\n"); shutdown(); return false; }
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
    musicLevel_ = resolve_launcher_music_gain(SDL_getenv("PROSPER_LAUNCHER_MUSIC_VOLUME"));
    const float gain = launcher_music_output_gain(musicLevel_, outputVolume_);
    if (!media_.init(phys_, device_, queue_, qfamily_, sampler_, &budget_, musicToggle_, gain))
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
    imageCount_ = static_cast<uint32_t>(images.size());
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
    poolSets_ = 0;
    budget_.reset(0);   // nothing may be allocated until a later init() creates a pool again
    ready_ = false;
}

void LibraryUi::destroy_covers() {
    for (Cover& c : covers_) {
        release_texture_set(budget_, c.set, VkDescriptorSet(VK_NULL_HANDLE),
                            [](VkDescriptorSet s) { ImGui_ImplVulkan_RemoveTexture(s); });
        if (c.view)   vkDestroyImageView(device_, c.view, nullptr);
        if (c.image)  vkDestroyImage(device_, c.image, nullptr);
        if (c.memory) prosper::gpu::free_device_memory(device_, c.memory);
    }
    covers_.clear();
}

// Grow the descriptor pool to fit `title_count` covers. The pool handle is baked into ImGui's Vulkan
// backend at init time and the vendored backend offers no way to swap it, so growing means creating the
// new pool, releasing every set prosper holds out of the old one, and re-running
// ImGui_ImplVulkan_Init — which rebuilds the font atlas and the UI pipeline against the new pool. That
// is affordable here because this only runs when the library actually got bigger: once at startup for a
// library above the floor, and again if the user points the app at a larger folder.
//
// Never shrinks. A rescan that finds FEWER games would otherwise churn the backend for nothing, and the
// spare sets cost a few hundred bytes.
bool LibraryUi::ensure_descriptor_capacity(size_t title_count) {
    if (!ready_ || device_ == VK_NULL_HANDLE) return false;
    uint32_t want = library_descriptor_pool_sets(title_count);
    if (poolCap_ && want > poolCap_) want = poolCap_;
    if (want <= poolSets_) return true;

    // Create first, swap second: if the driver refuses the bigger pool we still have a working one, and
    // the budget below simply keeps costing an image per title past capacity.
    VkDescriptorPool fresh = create_descriptor_pool(want);
    if (fresh == VK_NULL_HANDLE) {
        fprintf(stderr, "[library] could not grow the descriptor pool to %u sets; %zu titles share %u\n",
                want, title_count, poolSets_);
        return false;
    }

    // Both holders of sets from the old pool must let go before it is destroyed. The caller already
    // destroyed the covers; the backgrounds are released here, and are reloaded on the next focus change.
    media_.release_backgrounds();
    if (vulkanInit_) { ImGui_ImplVulkan_Shutdown(); vulkanInit_ = false; }
    vkDestroyDescriptorPool(device_, pool_, nullptr);
    pool_ = fresh;
    poolSets_ = want;
    poolFullWarned_ = false;   // a bigger pool may well hold everything; let it say so again if not
    budget_.reset(library_texture_budget(poolSets_));
    if (!init_imgui_vulkan()) {
        // Nothing can draw without the backend, and the pool it would have used is already gone. Same
        // outcome as a failed init(): the app falls back to the flat idle colour.
        fprintf(stderr, "[library] Vulkan backend re-init failed after growing the descriptor pool\n");
        ready_ = false;
        return false;
    }
    vulkanInit_ = true;
    // Worth a line: it is the only place the pool changes size, and it says out loud how many covers the
    // run can actually hold — which is exactly the number #1649 was silently wrong about.
    fprintf(stderr, "[library] descriptor pool grown to %u sets for %zu titles\n", poolSets_, title_count);
    return true;
}

void LibraryUi::apply_filter() {
    filterApplied_ = filterBuf_;
    filtered_.clear();
    for (int i = 0; i < static_cast<int>(games_.size()); i++)
        if (game_entry_matches_filter(games_[static_cast<size_t>(i)], filterApplied_))
            filtered_.push_back(i);
    selected_ = 0;
    hovered_ = -1;
    contextFi_ = -1;
    contextArmed_ = false;
}

void LibraryUi::set_output_volume(float volume) {
    if (volume < 0.0f) volume = 0.0f;
    if (volume > 1.0f) volume = 1.0f;
    outputVolume_ = volume;
    volumePercent_ = static_cast<int>(volume * 100.0f + 0.5f);
    // Live once the media layer is up; before init() this only seeds what init() will use.
    if (media_.ready())
        media_.set_output_gain(launcher_music_output_gain(musicLevel_, outputVolume_));
}

void LibraryUi::set_games(std::vector<GameEntry> games, const std::string& games_dir) {
    if (device_) vkDeviceWaitIdle(device_);   // covers may still be referenced by an in-flight frame
    destroy_covers();
    // After destroy_covers() and the device wait, so the grow finds the old pool free of cover sets and
    // no frame in flight that could still be sampling one.
    ensure_descriptor_capacity(games.size());
    games_ = std::move(games);
    gamesDir_ = games_dir;
    covers_.resize(games_.size());
    apply_filter();
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
             prosper::gpu::allocate_device_memory(device_, &ai, &cover.memory) == VK_SUCCESS &&
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
                 prosper::gpu::allocate_device_memory(device_, &ai, &stagingMem) == VK_SUCCESS &&
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
    if (stagingMem) prosper::gpu::free_device_memory(device_, stagingMem);

    if (ok) {
        VkImageViewCreateInfo ivi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivi.image = cover.image;
        ivi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivi.format = VK_FORMAT_R8G8B8A8_UNORM;
        ivi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        ok = vkCreateImageView(device_, &ivi, nullptr, &cover.view) == VK_SUCCESS;
    }
    bool poolFull = false;
    if (ok) {
        // Through the budget, NOT straight to AddTexture: on a full pool that helper hands the
        // VK_NULL_HANDLE from vkAllocateDescriptorSets to vkUpdateDescriptorSets, so checking its return
        // value would be checking after the undefined behaviour (#1649).
        poolFull = budget_.available() == 0;
        cover.set = acquire_texture_set(budget_, VkDescriptorSet(VK_NULL_HANDLE), [&] {
            return ImGui_ImplVulkan_AddTexture(sampler_, cover.view,
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
    }
    if (!ok || !cover.set) {
        // Leave the entry usable without art rather than dropping a bootable title from the list.
        // Said once for an exhausted pool: every remaining title would repeat it, and the cause is a
        // property of the pool rather than of this icon.
        if (poolFull) {
            if (!poolFullWarned_) {
                poolFullWarned_ = true;
                fprintf(stderr,
                        "[library] descriptor pool full at %u sets; titles past this show no cover art\n",
                        poolSets_);
            }
        } else {
            fprintf(stderr, "[library] could not upload cover for %s\n", game.title_name.c_str());
        }
        if (cover.view)   { vkDestroyImageView(device_, cover.view, nullptr); cover.view = VK_NULL_HANDLE; }
        if (cover.image)  { vkDestroyImage(device_, cover.image, nullptr);    cover.image = VK_NULL_HANDLE; }
        if (cover.memory) { prosper::gpu::free_device_memory(device_, cover.memory);     cover.memory = VK_NULL_HANDLE; }
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

void LibraryUi::draw_settings_content(LibraryAction& action) {
    if (ImGui::Button("< Back to games")) tab_ = LibraryTab::games;
    ImGui::Separator();
    const auto section = [&](const char* title) {
        // Bold section headers when the system font supplied one; the bitmap fallback has no
        // bold, and faking it with a second size would blur.
        ImGui::Spacing();
        if (boldFont_) ImGui::PushFont(boldFont_);
        ImGui::TextUnformatted(title);
        if (boldFont_) ImGui::PopFont();
        ImGui::Separator();
    };
    const auto radio_row = [&](const char* label, const char* hint, const std::string& current,
                               const char* value, LibraryAction::Kind kind) {
        // The label names the choice, the dim line below names its consequence; the stored value
        // stays the bare policy name the config file and flags use.
        const bool selected = (current == value);
        if (ImGui::RadioButton(label, selected) && !selected &&
            action.kind == LibraryAction::Kind::none) {
            action.kind = kind;
            action.value = value;
        }
        if (hint) {
            ImGui::Indent();
            ImGui::TextDisabled("%s", hint);
            ImGui::Unindent();
        }
    };

    ImGui::TextWrapped("These apply to games you open from here — never to a scripted launch. "
                       "A command-line flag or an environment variable still wins over each one.");
    section("Presentation");
    radio_row("Vsync", "Smoothest picture, a little more input lag.", presentMode_, "fifo",
              LibraryAction::Kind::set_present_mode);
    radio_row("Low-latency vsync", "Still tear-free, wakes the game sooner.", presentMode_,
              "mailbox", LibraryAction::Kind::set_present_mode);
    radio_row("Tearing allowed", "Fastest response, the image can shear mid-frame.", presentMode_,
              "immediate", LibraryAction::Kind::set_present_mode);
    // Mirror the choice locally: the persisted value comes back through main.cpp, but the dot
    // must move on the click, not on the next rescan.
    if (action.kind == LibraryAction::Kind::set_present_mode) presentMode_ = action.value;

    section("Display the game is told it is plugged into");
    radio_row("Always 1080p", "What prosper has always answered. Safest.", displayMode_, "legacy",
              LibraryAction::Kind::set_display_mode);
    radio_row("Match this display", "EXPERIMENTAL, per title: lets games offer modes beyond 1080p, "
                                   "but some pace themselves by this.",
              displayMode_, "host", LibraryAction::Kind::set_display_mode);
    radio_row("Match, including high refresh", "EXPERIMENTAL: some games run too fast past 60 Hz.",
              displayMode_, "host-high-refresh", LibraryAction::Kind::set_display_mode);
    if (action.kind == LibraryAction::Kind::set_display_mode) displayMode_ = action.value;

    section("Save data folder");
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputText("##savedata", savedataBuf_, sizeof savedataBuf_);
    ImGui::SameLine();
    if (ImGui::Button("Apply") && action.kind == LibraryAction::Kind::none) {
        savedataDir_ = savedataBuf_;
        savedataApplied_ = savedataDir_;
        action.kind = LibraryAction::Kind::set_savedata_dir;
        action.value = savedataDir_;
    }
    if (savedataApplied_.empty())
        ImGui::TextDisabled("Using the default location.");
    else
        ImGui::TextDisabled("Now: %s", savedataApplied_.c_str());
}

void LibraryUi::draw_controls_content() {
    if (ImGui::Button("< Back to games")) tab_ = LibraryTab::games;
    ImGui::Separator();
    ImGui::TextWrapped("Keyboard controls. A connected controller just works as pad 0 -- "
                       "this map is composed over it.");
    const ImGuiTableFlags flags =
        ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV;
    if (ImGui::BeginTable("controls", 2, flags)) {
        ImGui::TableSetupColumn("Keys", ImGuiTableColumnFlags_WidthFixed, 220.0f);
        ImGui::TableSetupColumn("Guest control", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        // Mirrors keyboard_pad_map.hpp's documented layout; update both together.
        static const char* kRows[][2] = {
            {"W A S D / arrows", "D-pad"},
            {"T F G H", "Left stick (up / left / down / right)"},
            {"I J K L", "Right stick (up / left / down / right)"},
            {"N M , .", "Square / Cross / Circle / Triangle"},
            {"Space", "Cross"},
            {"Z X C V", "L1 / L2 / R1 / R2"},
            {"B / Slash", "L3 / R3 (stick clicks)"},
            {"Enter", "Options"},
            {"Pause / F10", "Pause / resume at a flip boundary"},
            {"F11 / Alt+Enter", "Fullscreen"},
            {"F8", "Performance capture"},
            {"F9", "Frame capture for offline replay"},
            {"Esc", "Exit"},
        };
        for (const auto (&row)[2] : kRows) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(row[0]);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(row[1]);
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled("One cluster per hand at a time: WASD or TFGH left, IJKL or N M , . right.");
}

// Right-click detector for one row item: arms the menu below. Called at each clickable item
// of the row so the whole row answers; the menu itself is opened and drawn once per frame
// after the table (draw_row_menu), where OpenPopup/BeginPopup always agree.
void LibraryUi::note_row_right_click(int fi) {
    if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        selected_ = fi;
        contextFi_ = fi;
        contextArmed_ = true;
    }
}

void LibraryUi::draw_row_menu(LibraryAction& action, int shown) {
    if (contextArmed_) {
        ImGui::OpenPopup("rowmenu");
        contextArmed_ = false;
    }
    if (!ImGui::BeginPopup("rowmenu")) return;
    // The filter may have moved under an open menu; a stale row closes it rather than acting
    // on the wrong game.
    if (contextFi_ < 0 || contextFi_ >= shown) {
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        return;
    }
    const GameEntry& game =
        games_[static_cast<size_t>(filtered_[static_cast<size_t>(contextFi_)])];
    if (ImGui::MenuItem("Play")) {
        selected_ = contextFi_;
        action.kind = LibraryAction::Kind::open;
        action.app0_root = game.app0_root;
    }
    if (ImGui::MenuItem("Show in Explorer")) {
        action.kind = LibraryAction::Kind::show_in_explorer;
        action.app0_root = game.app0_root;
    }
    ImGui::EndPopup();
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
    if (!filtered_.empty() && selected_ >= 0 && selected_ < static_cast<int>(filtered_.size()))
        media_.set_focus(
            games_[static_cast<size_t>(filtered_[static_cast<size_t>(selected_)])].app0_root,
            nowMs);
    media_.update(nowMs);
    draw_backdrop();

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    // NoBackground so the title's art is visible behind the list (#1630). With no background loaded
    // this reveals the render pass's clear colour, which is the same flat dark the window used to
    // paint, so the view is unchanged for a title with no usable art.
    ImGui::Begin("prosper", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                 ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground |
                 ImGuiWindowFlags_MenuBar);

    // Menu bar: File (folders, recent games, exit). No PKG install item — prosper runs
    // user-supplied unpacked dumps, it installs nothing.
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Add games folder...")) action.kind = LibraryAction::Kind::browse;
            if (ImGui::MenuItem("Open game folder...")) action.kind = LibraryAction::Kind::pick_game;
            if (ImGui::BeginMenu("Recent games", !recentGames_.empty())) {
                for (size_t ri = 0; ri < recentGames_.size(); ri++) {
                    const RecentGame& recent = recentGames_[ri];
                    ImGui::PushID(static_cast<int>(ri));
                    // The name on the line, the path on hover: same screenshot rule as the table.
                    if (ImGui::MenuItem(recent.label.c_str())) {
                        action.kind = LibraryAction::Kind::open;
                        action.app0_root = recent.root;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", recent.root.c_str());
                    ImGui::PopID();
                }
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) action.kind = LibraryAction::Kind::quit;
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Settings")) {
            if (ImGui::MenuItem("Open settings")) tab_ = LibraryTab::settings;
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    if (tab_ == LibraryTab::settings) {
        draw_settings_content(action);
    } else if (tab_ == LibraryTab::controls) {
        draw_controls_content();
    } else if (games_.empty()) {
        ImGui::Spacing();
        if (gamesDir_.empty()) {
            ImGui::TextWrapped("No games folder is set yet. Choose the folder that holds your PS5 game "
                               "directories - the ones containing eboot.bin and sce_sys.");
        } else {
            ImGui::TextWrapped("No PS5 games found in this folder. Each game is its own directory "
                               "containing eboot.bin and sce_sys.");
        }
        ImGui::Spacing();
        // Keyboard-reachable too: ImGui's own nav is off, so without this the only way out of the
        // empty state would be the mouse or a drop. No text field exists here, so Enter is safe.
        if (ImGui::Button("Choose folder...") ||
            ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter) ||
            ImGui::IsKeyPressed(ImGuiKey_Space))
            action.kind = LibraryAction::Kind::browse;
        ImGui::SameLine();
        ImGui::TextDisabled("or press Enter, or drop a game folder on this window");
    } else {
        // Toolbar: icon buttons left, search right.
        const float toolbarTop = ImGui::GetCursorPosY();
        bool playClicked = false;
        if (icon_button("play", ToolbarIcon::play, "Play")) playClicked = true;
        ImGui::SameLine();
        if (icon_button("folder", ToolbarIcon::folder, "Add folder"))
            action.kind = LibraryAction::Kind::browse;
        ImGui::SameLine();
        if (icon_button("rescan", ToolbarIcon::rescan, "Refresh list"))
            action.kind = LibraryAction::Kind::rescan;
        ImGui::SameLine();
        if (icon_button("fullscreen", ToolbarIcon::fullscreen, "Full screen"))
            action.kind = LibraryAction::Kind::toggle_fullscreen;
        ImGui::SameLine();
        if (icon_button("keyboard", ToolbarIcon::keyboard, "Keyboard")) tab_ = LibraryTab::controls;
        ImGui::SameLine();
        // Discoverable rather than env-only: someone who does not want a launcher making noise should
        // not have to find a variable name to stop it. Reported to the caller so it is persisted.
        // No active paint: the icon itself shows the state (slashed when off), and music defaults
        // on, so an active highlight would read as permanently pressed.
        if (icon_button("music", musicToggle_ ? ToolbarIcon::music_on : ToolbarIcon::music_off,
                        "Music", false)) {
            musicToggle_ = !musicToggle_;
            media_.set_music_enabled(musicToggle_, nowMs);
            action.kind = LibraryAction::Kind::set_music;
            action.music_on = musicToggle_;
        }
        // The volume slider rides with the icon buttons; the search box rides the row's
        // right edge. Both vertically centered against the 52 px buttons, not their top edge.
        ImGui::SameLine();
        ImGui::SetCursorPosY(toolbarTop + 14.0f);
        ImGui::SetNextItemWidth(120.0f);
        int vol = volumePercent_;
        // Live while dragging (the gain applies on this thread), persisted on release: writing
        // the settings file on every dragged frame would be a write per frame.
        if (ImGui::SliderInt("##volume", &vol, 0, 100, "%d%%") && vol != volumePercent_) {
            volumePercent_ = vol;
            set_output_volume(static_cast<float>(vol) / 100.0f);
        }
        if (ImGui::IsItemDeactivatedAfterEdit() &&
            action.kind == LibraryAction::Kind::none) {
            action.kind = LibraryAction::Kind::set_volume;
            action.value = std::to_string(volumePercent_);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Volume");
        ImGui::SameLine(ImGui::GetContentRegionMax().x - 260.0f);
        ImGui::SetCursorPosY(toolbarTop + 14.0f);
        ImGui::SetNextItemWidth(260.0f);
        if (ImGui::InputTextWithHint("##search", "Search...", filterBuf_, sizeof filterBuf_))
            apply_filter();

        // Keyboard. Controller navigation is NOT implemented: ImGui's gamepad nav is off, and
        // nothing initialises SDL's gamepad subsystem while the library is alive — the pad backend
        // does that inside start_guest, by which point the library is gone. Tracked separately
        // rather than shipped as code that cannot fire.
        // Gated on text input: with the search box focused every key belongs to it — typing a
        // space once booted the wrong game (list_nav.hpp pins the gate).
        const bool textActive = ImGui::GetIO().WantTextInput;
        const int shown = static_cast<int>(filtered_.size());
        bool moveKey = false;
        switch (list_nav_move(ImGui::IsKeyPressed(ImGuiKey_UpArrow),
                              ImGui::IsKeyPressed(ImGuiKey_DownArrow),
                              ImGui::IsKeyPressed(ImGuiKey_Home),
                              ImGui::IsKeyPressed(ImGuiKey_End), textActive)) {
        case ListNavMove::up:
            if (selected_ > 0) {
                selected_--;
                moveKey = true;
            }
            break;
        case ListNavMove::down:
            if (selected_ + 1 < shown) {
                selected_++;
                moveKey = true;
            }
            break;
        case ListNavMove::home:
            if (shown > 0) {
                selected_ = 0;
                moveKey = true;
            }
            break;
        case ListNavMove::end:
            if (shown > 0) {
                selected_ = shown - 1;
                moveKey = true;
            }
            break;
        case ListNavMove::none:
            break;
        }
        if ((playClicked ||
             list_nav_open(ImGui::IsKeyPressed(ImGuiKey_Enter) ||
                               ImGui::IsKeyPressed(ImGuiKey_KeypadEnter) ||
                               ImGui::IsKeyPressed(ImGuiKey_Space),
                           textActive)) &&
            selected_ >= 0 && selected_ < shown) {
            action.kind = LibraryAction::Kind::open;
            action.app0_root =
                games_[static_cast<size_t>(filtered_[static_cast<size_t>(selected_)])].app0_root;
        }

        const ImGuiTableFlags tableFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter |
                                           ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                                           ImGuiTableFlags_ScrollY;
        float tableH = ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing();
        if (tableH < 0.0f) tableH = 0.0f;
        if (ImGui::BeginTable("games", 6, tableFlags, ImVec2(0, tableH))) {
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, kThumbSize);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Serial", ImGuiTableColumnFlags_WidthFixed, 110.0f);
            ImGui::TableSetupColumn("Region", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableSetupColumn("Version", ImGuiTableColumnFlags_WidthFixed, 90.0f);
            ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            // Single-line cells center against the thumbnail row; without this every text column
            // rides the row's top edge while the art fills it.
            const float textPadY = (kThumbSize + 8.0f - ImGui::GetTextLineHeight()) * 0.5f;
            int hoverThisFrame = -1;
            for (int fi = 0; fi < shown; fi++) {
                const GameEntry& game =
                    games_[static_cast<size_t>(filtered_[static_cast<size_t>(fi)])];
                ImGui::TableNextRow(0, kThumbSize + 8.0f);
                // One highlight per row, painted up front so it always spans the whole width:
                // solid for selected, brightest for selected+hovered, translucent wash for
                // hovered-only. The hover trails one frame (see hovered_), which is
                // imperceptible and avoids a cell-sized Selectable hover patch.
                if (fi == selected_ && fi == hovered_)
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                           ImGui::GetColorU32(ImGuiCol_ButtonHovered));
                else if (fi == selected_)
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                           ImGui::GetColorU32(ImGuiCol_ButtonActive));
                else if (fi == hovered_)
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                           ImGui::GetColorU32(ImGuiCol_ButtonHovered, 0.45f));
                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(fi);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4.0f);
                const VkDescriptorSet cover = cover_for(game);
                if (cover)
                    ImGui::Image(reinterpret_cast<ImTextureID>(cover),
                                 ImVec2(kThumbSize, kThumbSize));
                else
                    ImGui::TextDisabled("--");
                if (ImGui::IsItemClicked()) selected_ = fi;
                note_row_right_click(fi);
                ImGui::TableSetColumnIndex(1);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + textPadY);
                // SpanAllColumns makes the whole row one click target; the double-click opens.
                // The Selectable itself paints nothing — no selected, no hover — because the row
                // background above is the single highlight.
                ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0, 0, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0, 0, 0, 0));
                const bool rowClicked = ImGui::Selectable(
                    game.title_name.c_str(), false, ImGuiSelectableFlags_SpanAllColumns);
                if (ImGui::IsItemHovered()) hoverThisFrame = fi;
                ImGui::PopStyleColor(2);
                if (rowClicked) {
                    selected_ = fi;
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        action.kind = LibraryAction::Kind::open;
                        action.app0_root = game.app0_root;
                    }
                }
                ImGui::TableSetColumnIndex(2);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + textPadY);
                ImGui::TextUnformatted(game.title_id.c_str());
                ImGui::TableSetColumnIndex(3);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + textPadY);
                ImGui::TextUnformatted(game.region.c_str());
                ImGui::TableSetColumnIndex(4);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + textPadY);
                ImGui::TextUnformatted(game.version.c_str());
                ImGui::TableSetColumnIndex(5);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + textPadY);
                // The FOLDER NAME, not the absolute path: the folder name is what identifies the
                // dump while the rest of the path is the user's home directory — which would
                // otherwise end up in every screenshot anyone shares of their library. A drive
                // root has no basename, so there the root itself is the name.
                const std::string folder = path_basename(game.app0_root);
                ImGui::TextUnformatted((folder.empty() ? game.app0_root : folder).c_str());
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", path_basename(game.app0_root).c_str());
                note_row_right_click(fi);
                ImGui::PopID();
                // Follow a keyboard move; a mouse scroll is left alone so the two never fight.
                if (moveKey && fi == selected_) ImGui::SetScrollHereY(0.5f);
            }
            hovered_ = hoverThisFrame;
            ImGui::EndTable();
            draw_row_menu(action, shown);
        }
        if (shown == 0)
            ImGui::TextDisabled("No games match.");
        ImGui::Separator();
        ImGui::Text("%d game%s", shown, shown == 1 ? "" : "s");
        ImGui::SameLine();
        ImGui::TextDisabled("Play / Enter opens  |  double-click opens  |  Esc quits");
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
