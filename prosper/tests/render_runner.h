// render_runner.h — inline helper to render a single triangle (3 vertices, no vertex input) with a
// given vertex + fragment SPIR-V pair into a WxH RGBA8 image, clearing to blue first, and return the
// pixels. Used to verify recompiled shaders end-to-end (render -> readback -> pixel asserts). The
// including test links Vulkan::Vulkan.
#pragma once
#include <vulkan/vulkan.h>
#include "../src/gpu/render_state.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace prosper::test {

// Write an RGBA8 framebuffer to a binary PPM (P6) so a rendered frame is viewable as an image. PPM is
// trivially convertible to PNG (e.g. `magick in.ppm out.png`). Used by the render demos to leave
// screenshots on disk. Returns true on success.
inline bool dump_ppm(const char* path, const std::vector<uint8_t>& px, uint32_t W, uint32_t H) {
    if (px.size() != (size_t)W * H * 4) return false;
    FILE* f = fopen(path, "wb"); if (!f) return false;
    fprintf(f, "P6\n%u %u\n255\n", W, H);
    for (size_t i = 0; i < (size_t)W * H; i++) fwrite(&px[i * 4], 1, 3, f);   // RGB (drop alpha)
    fclose(f); return true;
}

// Write an RGBA8 framebuffer to a 24-bit BMP — natively viewable on Windows (double-click). BMP rows
// are bottom-up and BGR, padded to a 4-byte boundary.
inline bool dump_bmp(const char* path, const std::vector<uint8_t>& px, uint32_t W, uint32_t H) {
    if (px.size() != (size_t)W * H * 4) return false;
    FILE* f = fopen(path, "wb"); if (!f) return false;
    const uint32_t rowpad = (4 - (W * 3) % 4) % 4;
    const uint32_t dataSize = (W * 3 + rowpad) * H;
    const uint32_t fileSize = 54 + dataSize;
    auto u16 = [&](uint32_t v){ uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)}; fwrite(b, 1, 2, f); };
    auto u32 = [&](uint32_t v){ uint8_t b[4] = {(uint8_t)v, (uint8_t)(v>>8), (uint8_t)(v>>16), (uint8_t)(v>>24)}; fwrite(b, 1, 4, f); };
    fputc('B', f); fputc('M', f); u32(fileSize); u32(0); u32(54);                 // file header
    u32(40); u32(W); u32(H); u16(1); u16(24); u32(0); u32(dataSize); u32(2835); u32(2835); u32(0); u32(0);  // info header
    for (int y = (int)H - 1; y >= 0; y--) {                                       // bottom-up rows
        for (uint32_t x = 0; x < W; x++) {
            const uint8_t* p = &px[((size_t)y * W + x) * 4];
            fputc(p[2], f); fputc(p[1], f); fputc(p[0], f);                        // BGR
        }
        for (uint32_t k = 0; k < rowpad; k++) fputc(0, f);
    }
    fclose(f); return true;
}

// A texture to bind for a recompiled shader's image_sample: `rgba` points to w*h*4 RGBA8 bytes,
// bound as a COMBINED_IMAGE_SAMPLER (nearest filter, clamp) at descriptor-set 0, `binding`.
struct TexDesc { uint32_t binding; uint32_t w; uint32_t h; const uint8_t* rgba;
                 uint32_t max_aniso_ratio = 0; };   // #275: S# anisotropy ratio (0 = isotropic)

// One resource for the general N-binding path (render_triangle_rgba's `gres`): either a storage buffer
// (dwords non-empty, tex_rgba null) or a combined image sampler (tex_rgba set) at `binding`. Lets a
// real game shader that declares several constant/vertex buffers + textures have each bound distinctly.
struct FrameResource {
    uint32_t binding = 0;
    uint32_t set = 0;               // descriptor set: VS resources -> 0, PS resources -> 1 (they must not
                                    // share a set — both stages number bindings from 2, so one set would
                                    // collide binding 2/3 between stages and make the layout invalid).
    std::vector<uint32_t> dwords;   // storage-buffer contents (empty -> a 1-dword zero buffer)
    const uint8_t* tex_rgba = nullptr;   // non-null => a texture; then tw/th are its dimensions
    uint32_t tw = 0, th = 0;
    // Sampler state (Texture only). Defaults = LINEAR + clamp-to-edge — the harness's prior fixed
    // sampler — so render tests that build FrameResources directly stay byte-identical. The live path
    // fills these from the decoded S# (shader_resources.hpp). filter: 0=nearest, 1=linear; addr = Gen5
    // SQ_TEX CLAMP enum (0=wrap, 1=mirror, 2=clamp-last-texel, 6/7=border).
    uint32_t mag_filter = 1, min_filter = 1, mip_filter = 0;
    uint32_t addr_uvw[3] = {2, 2, 2};
    // Remaining S# sampler fields (#262). Defaults reproduce the current Vulkan sampler exactly (border
    // transparent-black, LOD 0..0, no bias), so FrameResources built directly by tests are byte-identical.
    uint32_t border_color_type = 0;
    float    min_lod = 0.0f, max_lod = 0.0f, lod_bias = 0.0f;
    // Anisotropy ratio (S# WORD0[11:9]; maxAnisotropy = 1<<ratio). 0 = isotropic (the default) -> the
    // sampler is byte-identical to before, so tests building FrameResources directly are unaffected (#275).
    uint32_t max_aniso_ratio = 0;
    // T# DST_SEL channel swizzle (SQ_SEL per channel: 0=0,1=1,4=R,5=G,6=B,7=A). Default = identity
    // (R,G,B,A) == a no-op VkComponentMapping, so tests that build FrameResources directly are unchanged.
    uint32_t swizzle[4] = {4, 5, 6, 7};
    bool is_texture() const { return tex_rgba != nullptr; }
};

// Returns W*H*4 RGBA bytes, or {} on any Vulkan failure (incl. a rejected SPIR-V module). When `ps`
// is non-null, the pipeline's fixed-function state (topology, blend, color write mask) is taken from
// the resolved RDNA2 render-state — this is how the back-half realizes a GpuState as a real VkPipeline.
//
// When `vbuf` and/or `cbuf` are non-null, a descriptor set is bound with the constant buffer at
// binding 2 and the vertex buffer at binding 3 (as storage buffers), matching declare_cbufs() in the
// recompiler — this is how a table-recompiled vertex shader fetches real vertex/constant data. Each is
// the raw dword contents of the buffer to bind. When both are null, the pipeline layout is empty and
// the color-only path above is taken unchanged.
//
// When `tex` is non-null, its RGBA8 texels are uploaded to a sampled VkImage and bound as a combined
// image sampler at tex->binding — how a recompiled pixel shader's image_sample reaches a real texture.
// One draw for the multi-draw backend: recompiled VS+PS SPIR-V, its resolved fixed-function state, its
// set-tagged resources, and its vertex count. render_draws_rgba records ALL of a submit's draws into ONE
// render pass (clear once, then per-draw pipeline+descriptors+draw) so a multi-draw frame composites
// correctly. render_triangle_rgba is a thin single-draw wrapper (below).
struct BackendDraw {
    std::vector<uint32_t> vs, fs;
    const prosper::gpu::ResolvedPipelineState* ps = nullptr;   // null -> triangle-list, write RGBA, no depth
    std::vector<FrameResource> R;                              // set-tagged resources (empty -> no descriptors)
    uint32_t vcount = 3;
    // Indexed draw: 32-bit index data (the executor widens guest 16-bit indices). Non-empty -> the draw
    // is recorded as vkCmdBindIndexBuffer + vkCmdDrawIndexed(indices.size()), so gl_VertexIndex is the
    // fetched index — exactly what the recompiled VS's storage-buffer vertex fetch expects. Empty ->
    // plain vkCmdDraw(vcount).
    std::vector<uint32_t> indices;
};

// `seed_rgba` (optional): W*H*4 RGBA8 pixels to PRELOAD the color attachment with before the draws
// run (loadOp LOAD instead of the blue clear). This is real render-target memory semantics: a game
// pass that draws into a target it (or an earlier submit) already rendered composites OVER that
// content — without it every pass starts from the diagnostic blue clear, so cross-submit
// accumulation (UE4's UI-onto-backbuffer after a separate composite submit) is lost. Null (the
// default) keeps the blue-clear behavior byte-identical for every existing caller.
// `clear_rgba` (optional): 4 floats (RGBA, Vulkan order) to clear the color attachment to when no
// seed is supplied. Null keeps the legacy diagnostic blue — every test harness caller passes null,
// so their behavior is byte-identical. The live renderer passes the game's decoded fast-clear color
// (or opaque black when none), so real frames no longer start from blue (#309). PROSPER_CLEAR_DEBUG
// forces the blue back on regardless, so unrendered areas can still be spotted during development.
// Persistent Vulkan context. Creating a fresh instance+device PER render_draws_rgba call dominated
// wall-clock — every submit paid full device init — which made a many-draw frame (real gameplay is
// hundreds of draws/submit) impossibly slow and blocked headless scene investigation (#320). Create the
// instance/physical-device/device/queue ONCE (lazy, thread-safe static init) and reuse it across every
// call; per-call resources (images/pipelines/descriptors/command buffer/fence/pool) are still created and
// freed per call. The context intentionally leaks at process exit — fine for a headless/diagnostic tool.
struct RenderVkCtx {
    VkInstance inst = VK_NULL_HANDLE; VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE; VkQueue queue = VK_NULL_HANDLE; uint32_t qfi = UINT32_MAX;
    bool aniso_enabled = false; float max_aniso_limit = 1.0f; bool ok = false;
};
inline const RenderVkCtx& render_vk_ctx() {
    static RenderVkCtx c = [] {
        RenderVkCtx r;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &r.inst) != VK_SUCCESS || !r.inst) return r;
        uint32_t nd = 0; vkEnumeratePhysicalDevices(r.inst, &nd, nullptr);
        if (!nd) return r;
        std::vector<VkPhysicalDevice> devs(nd); vkEnumeratePhysicalDevices(r.inst, &nd, devs.data());
        r.phys = devs[0];
        uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &nqf, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nqf); vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &nqf, qf.data());
        for (uint32_t i = 0; i < nqf; i++) if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { r.qfi = i; break; }
        if (r.qfi == UINT32_MAX) return r;
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = r.qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
        // robustBufferAccess: OOB storage-buffer accesses are well-defined (predicated ops on
        // narrowed-EXEC lanes can't fault).
        VkPhysicalDeviceFeatures feats{}; feats.robustBufferAccess = VK_TRUE; dci.pEnabledFeatures = &feats;
        // samplerAnisotropy (#275): enable only if advertised; maxSamplerAnisotropy is the clamp ceiling.
        VkPhysicalDeviceFeatures supported{}; vkGetPhysicalDeviceFeatures(r.phys, &supported);
        VkPhysicalDeviceProperties phys_props{}; vkGetPhysicalDeviceProperties(r.phys, &phys_props);
        r.aniso_enabled = supported.samplerAnisotropy;
        r.max_aniso_limit = phys_props.limits.maxSamplerAnisotropy;
        if (r.aniso_enabled) feats.samplerAnisotropy = VK_TRUE;
        // robustImageAccess (VK_EXT_image_robustness): OpImageRead OOB must return zero (#131). Guarded.
        VkPhysicalDeviceImageRobustnessFeaturesEXT irf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT};
        const char* img_robust_ext[] = { "VK_EXT_image_robustness" };
        { uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(r.phys, nullptr, &ne, nullptr);
          std::vector<VkExtensionProperties> de(ne);
          vkEnumerateDeviceExtensionProperties(r.phys, nullptr, &ne, de.data());
          for (uint32_t i = 0; i < ne; i++) if (!strcmp(de[i].extensionName, "VK_EXT_image_robustness")) {
              VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
              f2.pNext = &irf; vkGetPhysicalDeviceFeatures2(r.phys, &f2);
              if (irf.robustImageAccess) { dci.pNext = &irf;
                  dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = img_robust_ext; }
              break;
          } }
        if (vkCreateDevice(r.phys, &dci, nullptr, &r.dev) != VK_SUCCESS || !r.dev) return r;
        vkGetDeviceQueue(r.dev, r.qfi, 0, &r.queue);
        r.ok = true;
        return r;
    }();
    return c;
}

inline std::vector<uint8_t> render_draws_rgba(const std::vector<BackendDraw>& draws, uint32_t W, uint32_t H,
                                              const uint8_t* seed_rgba = nullptr,
                                              const float* clear_rgba = nullptr) {
    std::vector<uint8_t> out;
    if (draws.empty()) return out;
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) return out;
    VkInstance inst = ctx.inst; (void)inst; VkPhysicalDevice phys = ctx.phys;
    VkDevice dev = ctx.dev; VkQueue queue = ctx.queue; uint32_t qfi = ctx.qfi;
    const bool aniso_enabled = ctx.aniso_enabled; const float max_aniso_limit = ctx.max_aniso_limit;
    VkPhysicalDeviceMemoryProperties memp; vkGetPhysicalDeviceMemoryProperties(phys, &memp);
    auto pick = [&](uint32_t bits, VkMemoryPropertyFlags want) -> uint32_t {
        for (uint32_t i = 0; i < memp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memp.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX; };
    const VkFormat FMT = VK_FORMAT_R8G8B8A8_UNORM;
    // Depth attachment is created if ANY draw enables the depth test (the shared render pass has one
    // fixed attachment set); each draw's pipeline sets its own depthTest/Write/CompareOp. A frame with
    // no depth-using draw takes the color-only path unchanged.
    bool use_depth = false, use_stencil = false;
    // Depth/stencil clear values for the render pass's LOAD_OP_CLEAR (#371). Resolve set
    // depth_clear_value from the guest's DB_DEPTH_CLEAR, or a compare-op-appropriate default (1.0 for
    // LESS, 0.0 for GREATER) — never the old fixed 0.5. Latch the depth clear from the first DEPTH-testing
    // draw and the stencil clear from the first STENCIL-testing draw INDEPENDENTLY: coupling them (one
    // latch off the first depth-OR-stencil draw) let a stencil-only first draw — e.g. the title-shimmer
    // stencil prime with depth off — force depth_clear to that draw's default 1.0, so a later reverse-Z
    // (GREATER) draw in the same submit cleared to 1.0 and rejected every fragment (#457).
    float    depth_clear   = 1.0f;
    uint32_t stencil_clear = 0;
    bool     got_depth_clear = false, got_stencil_clear = false;
    for (const auto& d : draws) {
        if (!d.ps) continue;
        if (d.ps->depth_test_enable) { use_depth = true;
            if (!got_depth_clear) { depth_clear = d.ps->depth_clear_value; got_depth_clear = true; } }
        if (d.ps->stencil_enable) { use_stencil = true;
            if (!got_stencil_clear) { stencil_clear = d.ps->stencil_clear_value; got_stencil_clear = true; } }
    }
    if (getenv("PROSPER_NO_DEPTH"))   use_depth = false;     // diag: isolate depth-test rejection
    if (getenv("PROSPER_NO_STENCIL")) use_stencil = false;   // diag: isolate stencil masking
    const bool use_ds = use_depth || use_stencil;
    // Use a stencil-capable depth format ONLY when a draw actually uses stencil (a UI mask). The
    // depth-only path keeps the original D32 depth-only format + aspect, so existing render tests are
    // byte-identical (#264).
    const VkFormat DFMT = use_stencil ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
    const VkImageAspectFlags DASPECT = VK_IMAGE_ASPECT_DEPTH_BIT | (use_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
    VkImage dimg = VK_NULL_HANDLE; VkDeviceMemory dmem = VK_NULL_HANDLE; VkImageView dview = VK_NULL_HANDLE;

    VkImageCreateInfo imgci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgci.imageType = VK_IMAGE_TYPE_2D; imgci.format = FMT; imgci.extent = {W, H, 1};
    imgci.mipLevels = 1; imgci.arrayLayers = 1; imgci.samples = VK_SAMPLE_COUNT_1_BIT;
    imgci.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  (seed_rgba ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0u);
    VkImage img; vkCreateImage(dev, &imgci, nullptr, &img);
    VkMemoryRequirements ir; vkGetImageMemoryRequirements(dev, img, &ir);
    VkMemoryAllocateInfo iai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    iai.allocationSize = ir.size; iai.memoryTypeIndex = pick(ir.memoryTypeBits, 0);
    VkDeviceMemory imem; vkAllocateMemory(dev, &iai, nullptr, &imem); vkBindImageMemory(dev, img, imem, 0);
    VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ivci.image = img; ivci.viewType = VK_IMAGE_VIEW_TYPE_2D; ivci.format = FMT;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view; vkCreateImageView(dev, &ivci, nullptr, &view);

    if (use_ds) {
        VkImageCreateInfo dci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        dci.imageType = VK_IMAGE_TYPE_2D; dci.format = DFMT; dci.extent = {W, H, 1};
        dci.mipLevels = 1; dci.arrayLayers = 1; dci.samples = VK_SAMPLE_COUNT_1_BIT;
        dci.tiling = VK_IMAGE_TILING_OPTIMAL; dci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        vkCreateImage(dev, &dci, nullptr, &dimg);
        VkMemoryRequirements dr; vkGetImageMemoryRequirements(dev, dimg, &dr);
        VkMemoryAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        dai.allocationSize = dr.size; dai.memoryTypeIndex = pick(dr.memoryTypeBits, 0);
        vkAllocateMemory(dev, &dai, nullptr, &dmem); vkBindImageMemory(dev, dimg, dmem, 0);
        VkImageViewCreateInfo dvci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        dvci.image = dimg; dvci.viewType = VK_IMAGE_VIEW_TYPE_2D; dvci.format = DFMT;
        dvci.subresourceRange = {DASPECT, 0, 1, 0, 1};
        vkCreateImageView(dev, &dvci, nullptr, &dview);
    }

    VkAttachmentDescription att[2]{};
    att[0].format = FMT; att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    // Seeded: the attachment already holds the preloaded pixels (uploaded before the pass below), so
    // LOAD them instead of clearing, and declare the matching initial layout.
    att[0].loadOp = seed_rgba ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = seed_rgba ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    att[1].format = DFMT; att[1].samples = VK_SAMPLE_COUNT_1_BIT;
    att[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // Clear the stencil at pass start when a mask uses it (so the mask draw defines the stenciled
    // region from 0); within the pass the stencil persists across draws, so store-op can be DONT_CARE.
    att[1].stencilLoadOp = use_stencil ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference ar{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference dar{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{}; sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1; sub.pColorAttachments = &ar;
    if (use_ds) sub.pDepthStencilAttachment = &dar;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = use_ds ? 2 : 1; rpci.pAttachments = att; rpci.subpassCount = 1; rpci.pSubpasses = &sub;
    VkRenderPass rp; vkCreateRenderPass(dev, &rpci, nullptr, &rp);
    VkImageView fbviews[2] = {view, dview};
    VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbci.renderPass = rp; fbci.attachmentCount = use_ds ? 2 : 1; fbci.pAttachments = fbviews; fbci.width = W; fbci.height = H; fbci.layers = 1;
    VkFramebuffer fb; vkCreateFramebuffer(dev, &fbci, nullptr, &fb);

    auto mkmod = [&](const std::vector<uint32_t>& c) -> VkShaderModule {
        VkShaderModuleCreateInfo s{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        s.codeSize = c.size() * 4; s.pCode = c.data(); VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(dev, &s, nullptr, &m); return m; };
    // Per-draw Vulkan objects — kept alive until after the queue submit, freed at the end.
    struct DV {
        VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
        std::vector<FrameResource> R;
        std::vector<VkDescriptorSetLayout> dsls; std::vector<VkDescriptorSet> dsets;
        VkDescriptorPool dpool = VK_NULL_HANDLE;
        std::vector<VkBuffer> sbuf, tstage; std::vector<VkDeviceMemory> sbmem, tmem, tstagemem;
        std::vector<VkImage> timg; std::vector<VkImageView> tview; std::vector<VkSampler> tsamp;
        VkPipelineLayout layout = VK_NULL_HANDLE; VkPipeline pipe = VK_NULL_HANDLE;
        VkBuffer ibuf = VK_NULL_HANDLE; VkDeviceMemory ibmem = VK_NULL_HANDLE;   // index buffer (indexed draws)
        uint32_t n_sets = 1, vcount = 3, icount = 0; bool use_desc = false, ok = false;
    };
    std::vector<DV> dv(draws.size());
    // Pass 1: create each draw's shader modules, descriptors (with texture staging upload), and pipeline.
    for (size_t di = 0; di < draws.size(); di++) {
        const BackendDraw& bd = draws[di];
        DV& v = dv[di];
        v.vs = mkmod(bd.vs); v.fs = mkmod(bd.fs);
        if (!v.vs || !v.fs) continue;   // rejected SPIR-V -> skip this draw
        const prosper::gpu::ResolvedPipelineState* ps = bd.ps;
        v.vcount = bd.vcount;
        // Indexed draw: upload the 32-bit index data to a host-visible VkIndexBuffer now; the record
        // pass binds it and issues vkCmdDrawIndexed instead of vkCmdDraw.
        if (!bd.indices.empty()) {
            VkDeviceSize isz = (VkDeviceSize)bd.indices.size() * 4;
            VkBufferCreateInfo ibci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            ibci.size = isz; ibci.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            vkCreateBuffer(dev, &ibci, nullptr, &v.ibuf);
            VkMemoryRequirements imr; vkGetBufferMemoryRequirements(dev, v.ibuf, &imr);
            VkMemoryAllocateInfo imai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; imai.allocationSize = imr.size;
            imai.memoryTypeIndex = pick(imr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(dev, &imai, nullptr, &v.ibmem); vkBindBufferMemory(dev, v.ibuf, v.ibmem, 0);
            void* ip = nullptr; vkMapMemory(dev, v.ibmem, 0, isz, 0, &ip);
            std::memcpy(ip, bd.indices.data(), (size_t)isz); vkUnmapMemory(dev, v.ibmem);
            v.icount = (uint32_t)bd.indices.size();
        }
        VkPipelineShaderStageCreateInfo st[2]{};
        st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = v.vs; st[0].pName = "main";
        st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = v.fs; st[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = ps ? (VkPrimitiveTopology)ps->topology : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        // Default: full-target viewport. When the resolved state carries the guest's PA_CL_VPORT transform,
        // honor it — a guest yscale < 0 arrives as a negative viewport_h (Vulkan core-1.1 flipped viewport),
        // reproducing the hardware's Y orientation (#38; each draw item keeps its own resolved viewport).
        VkViewport vp{0, 0, (float)W, (float)H, 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
        if (ps && ps->has_viewport)
            vp = {ps->viewport_x, ps->viewport_y, ps->viewport_w, ps->viewport_h, ps->min_depth, ps->max_depth};
        VkPipelineViewportStateCreateInfo vpst{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vpst.viewportCount = 1; vpst.pViewports = &vp; vpst.scissorCount = 1; vpst.pScissors = &sc;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        // Honor the guest's PA_SU_SC_MODE_CNTL cull/front-face/polygon mode (#456). Resolve encodes these
        // as the Vk enumerators; an absent register resolves to the same NONE/CCW/FILL default above, so
        // the null-ps (test) path and any draw that never programs it are byte-identical. PROSPER_NO_CULL
        // forces CULL_NONE back on (diag escape hatch, matching PROSPER_NO_DEPTH/NO_STENCIL/NO_BLEND) —
        // if honoring the guest cull ever culls the wrong faces on a live render (a winding/Y-flip edge),
        // this isolates it without a rebuild.
        if (ps) { rs.cullMode  = getenv("PROSPER_NO_CULL") ? VK_CULL_MODE_NONE : (VkCullModeFlags)ps->cull_mode;
                  rs.frontFace = (VkFrontFace)ps->front_face;
                  rs.polygonMode = (VkPolygonMode)ps->polygon_mode; }
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
        if (ps) {
            cba.colorWriteMask = ps->color_write_mask;
            cba.blendEnable    = ps->blend_enable ? VK_TRUE : VK_FALSE;
            if (getenv("PROSPER_NO_BLEND")) cba.blendEnable = VK_FALSE;   // diag: isolate blend compositing
            cba.srcColorBlendFactor = (VkBlendFactor)ps->src_color_blend_factor;
            cba.dstColorBlendFactor = (VkBlendFactor)ps->dst_color_blend_factor;
            cba.colorBlendOp        = (VkBlendOp)ps->color_blend_op;
            // Alpha channel uses its OWN resolved factors (#381): resolve set these from the separate
            // ALPHA_* blend fields when SEPARATE_ALPHA_BLEND was programmed, else it already mirrored the
            // color factors — so this is correct in both cases without guessing here.
            cba.srcAlphaBlendFactor = (VkBlendFactor)ps->src_alpha_blend_factor;
            cba.dstAlphaBlendFactor = (VkBlendFactor)ps->dst_alpha_blend_factor;
            cba.alphaBlendOp        = (VkBlendOp)ps->alpha_blend_op;
        }
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1; cb.pAttachments = &cba;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        if (ps && ps->depth_test_enable) {
            dss.depthTestEnable  = VK_TRUE;
            dss.depthWriteEnable = ps->depth_write_enable ? VK_TRUE : VK_FALSE;
            dss.depthCompareOp   = (VkCompareOp)ps->depth_compare_op;
            if (getenv("PROSPER_DEPTH_ALWAYS")) dss.depthCompareOp = VK_COMPARE_OP_ALWAYS;   // diag
        }
        if (ps && ps->stencil_enable) {
            // Wire the front/back stencil op-state so masks clip (e.g. the title shimmer tests the
            // stencil the logo draw wrote). ref/compareMask/writeMask are baked (not dynamic).
            dss.stencilTestEnable = VK_TRUE;
            auto mkop = [&](int fb) {
                VkStencilOpState s{};
                s.failOp      = (VkStencilOp)ps->stencil_fail_op[fb];
                s.passOp      = (VkStencilOp)ps->stencil_pass_op[fb];
                s.depthFailOp = (VkStencilOp)ps->stencil_depth_fail_op[fb];
                s.compareOp   = (VkCompareOp)ps->stencil_compare_op[fb];
                s.compareMask = ps->stencil_compare_mask[fb];
                s.writeMask   = ps->stencil_write_mask[fb];
                // AMD splits the stencil reference: STENCILTESTVAL is the COMPARE reference, but a REPLACE
                // op writes STENCILOPVAL. Vulkan has one `reference` for both. When this draw REPLACEs
                // (its whole purpose is to WRITE a mask value, and its compare is typically ALWAYS so the
                // compare-ref is irrelevant), use STENCILOPVAL so the mask is written with the value the
                // game intended — else a mask-write with TESTVAL=0/OPVAL=1 writes 0 and every later
                // test==1 draw is wrongly culled (PPSA02664's whole UI vanished; #270).
                const uint32_t REPLACE = 2;   // VK_STENCIL_OP_REPLACE
                bool does_replace = (ps->stencil_pass_op[fb] == REPLACE || ps->stencil_fail_op[fb] == REPLACE ||
                                     ps->stencil_depth_fail_op[fb] == REPLACE);
                s.reference   = does_replace ? ps->stencil_op_val[fb] : ps->stencil_ref[fb];
                return s;
            };
            dss.front = mkop(0); dss.back = mkop(1);
            if (getenv("PROSPER_STENCILLOG"))
                fprintf(stderr, "[stencil] front{cmp=%u ref=%u opval=%u cmask=0x%x wmask=0x%x fail=%u pass=%u zfail=%u} vkref=%u depth_test=%d\n",
                        ps->stencil_compare_op[0], ps->stencil_ref[0], ps->stencil_op_val[0], ps->stencil_compare_mask[0], ps->stencil_write_mask[0],
                        ps->stencil_fail_op[0], ps->stencil_pass_op[0], ps->stencil_depth_fail_op[0], dss.front.reference, (int)ps->depth_test_enable);
        }
        // Descriptor resources for this draw (two-set: VS=set0, PS=set1 — same layout as the single path).
        v.R = bd.R; auto& R = v.R;
        v.use_desc = !R.empty();
        for (auto& r : R) v.n_sets = std::max(v.n_sets, r.set + 1);
        v.dsls.assign(v.n_sets, VK_NULL_HANDLE); v.dsets.assign(v.n_sets, VK_NULL_HANDLE);
        v.sbuf.assign(R.size(), VK_NULL_HANDLE); v.tstage.assign(R.size(), VK_NULL_HANDLE);
        v.sbmem.assign(R.size(), VK_NULL_HANDLE); v.tmem.assign(R.size(), VK_NULL_HANDLE); v.tstagemem.assign(R.size(), VK_NULL_HANDLE);
        v.timg.assign(R.size(), VK_NULL_HANDLE); v.tview.assign(R.size(), VK_NULL_HANDLE); v.tsamp.assign(R.size(), VK_NULL_HANDLE);
        if (v.use_desc) {
            std::vector<VkDescriptorSetLayoutBinding> lb(R.size());
            std::vector<VkDescriptorBufferInfo> dbi(R.size());
            std::vector<VkDescriptorImageInfo> dii(R.size());
            std::vector<VkWriteDescriptorSet> wr(R.size());
            uint32_t n_storage = 0, n_sampler = 0;
            for (size_t i = 0; i < R.size(); i++) {
                const FrameResource& r = R[i];
                lb[i] = {}; lb[i].binding = r.binding; lb[i].descriptorCount = 1;
                if (r.is_texture()) {
                    n_sampler++;
                    lb[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    // A set-0 texture belongs to the VERTEX shader (build_R tags VS resources into set 0,
                    // PS into set 1). stageFlags must include every stage that reads the binding, so a
                    // vertex texture fetch (displacement/heightmap, GPU vertex animation) needs
                    // VERTEX_BIT — a fragment-only hardcode made set-0 textures invisible to the VS,
                    // yielding undefined samples / a validation error (#376). Match the storage-buffer path.
                    lb[i].stageFlags = (r.set == 0)
                        ? (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                        : VK_SHADER_STAGE_FRAGMENT_BIT;
                    VkImageCreateInfo tci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                    tci.imageType = VK_IMAGE_TYPE_2D; tci.format = VK_FORMAT_R8G8B8A8_UNORM; tci.extent = {r.tw, r.th, 1};
                    tci.mipLevels = 1; tci.arrayLayers = 1; tci.samples = VK_SAMPLE_COUNT_1_BIT; tci.tiling = VK_IMAGE_TILING_OPTIMAL;
                    tci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                    vkCreateImage(dev, &tci, nullptr, &v.timg[i]);
                    VkMemoryRequirements tr; vkGetImageMemoryRequirements(dev, v.timg[i], &tr);
                    VkMemoryAllocateInfo tai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; tai.allocationSize = tr.size;
                    tai.memoryTypeIndex = pick(tr.memoryTypeBits, 0); vkAllocateMemory(dev, &tai, nullptr, &v.tmem[i]);
                    vkBindImageMemory(dev, v.timg[i], v.tmem[i], 0);
                    VkImageViewCreateInfo tvci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                    // NOTE(#263): r.srgb carries whether the T# is a gamma-encoded (sRGB) surface, but we
                    // deliberately keep the view UNORM. This whole renderer works in gamma/sRGB space
                    // end-to-end (this target is UNORM, the frontend blit + swapchain are UNORM), with NO
                    // linear->sRGB encode at present. Sampling an sRGB texture as UNORM passes its encoded
                    // bytes straight through, which MATCHES real-hardware output for pass-through content.
                    // Flipping this to VK_FORMAT_R8G8B8A8_SRGB would apply sRGB->linear on sample with no
                    // matching encode on store -> linear values into a UNORM swapchain -> too dark. A
                    // correct sRGB fix is a coordinated linear-working-space + output-encode change (see the
                    // #263 discussion), NOT a per-view format flip. r.srgb is decoded now as groundwork.
                    tvci.image = v.timg[i]; tvci.viewType = VK_IMAGE_VIEW_TYPE_2D; tvci.format = VK_FORMAT_R8G8B8A8_UNORM;
                    // T# DST_SEL channel remap (#261): map each SQ_SEL to a VkComponentSwizzle. Identity
                    // (the default, and the narrow/font path) yields IDENTITY == a no-op. PROSPER_NO_SWIZZLE
                    // forces identity for A/B testing against the pre-swizzle behavior.
                    auto vkswz = [](uint32_t s) -> VkComponentSwizzle {
                        switch (s) {
                            case 0:  return VK_COMPONENT_SWIZZLE_ZERO;
                            case 1:  return VK_COMPONENT_SWIZZLE_ONE;
                            case 4:  return VK_COMPONENT_SWIZZLE_R;
                            case 5:  return VK_COMPONENT_SWIZZLE_G;
                            case 6:  return VK_COMPONENT_SWIZZLE_B;
                            case 7:  return VK_COMPONENT_SWIZZLE_A;
                            default: return VK_COMPONENT_SWIZZLE_IDENTITY;
                        }
                    };
                    if (!getenv("PROSPER_NO_SWIZZLE"))
                        tvci.components = {vkswz(r.swizzle[0]), vkswz(r.swizzle[1]), vkswz(r.swizzle[2]), vkswz(r.swizzle[3])};
                    tvci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; vkCreateImageView(dev, &tvci, nullptr, &v.tview[i]);
                    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
                    // Honor the game's decoded S# (r.mag/min/mip_filter, r.addr_uvw) instead of a fixed
                    // LINEAR/clamp sampler — point-sampled art (pixel-art titles) no longer gets a blurred
                    // per-texel outline, and real wrap modes work. Gen5 CLAMP enum -> Vk address mode.
                    auto vkflt  = [](uint32_t f){ return f ? VK_FILTER_LINEAR : VK_FILTER_NEAREST; };
                    auto vkaddr = [](uint32_t c) -> VkSamplerAddressMode {
                        switch (c) {
                            case 0:  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
                            case 1:  return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                            case 6: case 7: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                            default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   // 2,3,4,5: clamp-ish
                        }
                    };
                    sci.magFilter = vkflt(r.mag_filter); sci.minFilter = vkflt(r.min_filter);
                    sci.mipmapMode = r.mip_filter ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    sci.addressModeU = vkaddr(r.addr_uvw[0]);
                    sci.addressModeV = vkaddr(r.addr_uvw[1]);
                    sci.addressModeW = vkaddr(r.addr_uvw[2]);
                    // Remaining S# fields (#262), applied where valid on this color combined-image-sampler.
                    // Defaults (border 0 / LOD 0,0 / bias 0) reproduce the previous fixed sampler exactly.
                    //   border color: only bites with CLAMP_TO_BORDER wrap. 3 = register/custom (needs
                    //     VK_EXT_custom_border_color); fall back to opaque-black.
                    //   LOD min/max/bias: honored; harmless with our single uploaded mip.
                    // Anisotropy (#275): applied when the S# requests a ratio, the device feature is
                    // enabled, and filtering is linear (Vulkan requires anisotropyEnable only with linear
                    // mag/min filters). maxAnisotropy = 1<<ratio, clamped to the device ceiling. ratio 0
                    // (isotropic) leaves anisotropyEnable false -> the sampler is unchanged.
                    if (r.max_aniso_ratio > 0 && aniso_enabled &&
                        sci.magFilter == VK_FILTER_LINEAR && sci.minFilter == VK_FILTER_LINEAR) {
                        sci.anisotropyEnable = VK_TRUE;
                        float want = (float)(1u << r.max_aniso_ratio);
                        sci.maxAnisotropy = want < max_aniso_limit ? want : max_aniso_limit;
                    }
                    // NOT applied here (need machinery the current path lacks — decoded under GFXLOG only):
                    //   depth_compare_func (needs a depth/shadow sampler over a depth image),
                    //   unnormalized coords (strict validity rules + recompiler coord semantics).
                    switch (r.border_color_type) {
                        case 1:  sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK; break;
                        case 2:  sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; break;
                        case 3:  sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK; break;   // custom unsupported
                        default: sci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK; break;
                    }
                    sci.minLod = r.min_lod; sci.maxLod = r.max_lod; sci.mipLodBias = r.lod_bias;
                    vkCreateSampler(dev, &sci, nullptr, &v.tsamp[i]);
                    VkDeviceSize tbytes = (VkDeviceSize)r.tw * r.th * 4;
                    VkBufferCreateInfo stci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; stci.size = tbytes; stci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                    vkCreateBuffer(dev, &stci, nullptr, &v.tstage[i]);
                    VkMemoryRequirements sr; vkGetBufferMemoryRequirements(dev, v.tstage[i], &sr);
                    VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; sai.allocationSize = sr.size;
                    sai.memoryTypeIndex = pick(sr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    vkAllocateMemory(dev, &sai, nullptr, &v.tstagemem[i]); vkBindBufferMemory(dev, v.tstage[i], v.tstagemem[i], 0);
                    void* sp = nullptr; vkMapMemory(dev, v.tstagemem[i], 0, tbytes, 0, &sp);
                    std::memcpy(sp, r.tex_rgba, (size_t)tbytes); vkUnmapMemory(dev, v.tstagemem[i]);
                    dii[i] = {v.tsamp[i], v.tview[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                    wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstBinding = r.binding; wr[i].descriptorCount = 1;
                    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wr[i].pImageInfo = &dii[i];
                } else {
                    n_storage++;
                    lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; lb[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
                    VkDeviceSize sz = r.dwords.empty() ? 4 : (VkDeviceSize)r.dwords.size() * 4;
                    VkBufferCreateInfo sbci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; sbci.size = sz; sbci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                    vkCreateBuffer(dev, &sbci, nullptr, &v.sbuf[i]);
                    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, v.sbuf[i], &mr);
                    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = mr.size;
                    mai.memoryTypeIndex = pick(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    vkAllocateMemory(dev, &mai, nullptr, &v.sbmem[i]); vkBindBufferMemory(dev, v.sbuf[i], v.sbmem[i], 0);
                    void* p = nullptr; vkMapMemory(dev, v.sbmem[i], 0, sz, 0, &p);
                    if (r.dwords.empty()) ((uint32_t*)p)[0] = 0; else std::memcpy(p, r.dwords.data(), r.dwords.size() * 4);
                    vkUnmapMemory(dev, v.sbmem[i]);
                    dbi[i] = {v.sbuf[i], 0, VK_WHOLE_SIZE};
                    wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstBinding = r.binding; wr[i].descriptorCount = 1;
                    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i];
                }
            }
            for (uint32_t s = 0; s < v.n_sets; s++) {
                std::vector<VkDescriptorSetLayoutBinding> slb;
                for (size_t i = 0; i < R.size(); i++) if (R[i].set == s) slb.push_back(lb[i]);
                VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
                dslci.bindingCount = (uint32_t)slb.size(); dslci.pBindings = slb.data();
                vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &v.dsls[s]);
            }
            VkDescriptorPoolSize psz[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, n_storage ? n_storage : 1},
                                           {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, n_sampler ? n_sampler : 1}};
            VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            dpci.maxSets = v.n_sets; dpci.poolSizeCount = n_sampler ? 2 : 1; dpci.pPoolSizes = psz; vkCreateDescriptorPool(dev, &dpci, nullptr, &v.dpool);
            VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dsai.descriptorPool = v.dpool; dsai.descriptorSetCount = v.n_sets; dsai.pSetLayouts = v.dsls.data();
            vkAllocateDescriptorSets(dev, &dsai, v.dsets.data());
            for (size_t i = 0; i < R.size(); i++) wr[i].dstSet = v.dsets[R[i].set];
            vkUpdateDescriptorSets(dev, (uint32_t)wr.size(), wr.data(), 0, nullptr);
        }
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        if (v.use_desc) { plci.setLayoutCount = v.n_sets; plci.pSetLayouts = v.dsls.data(); }
        vkCreatePipelineLayout(dev, &plci, nullptr, &v.layout);
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.stageCount = 2; gp.pStages = st; gp.pVertexInputState = &vin; gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vpst; gp.pRasterizationState = &rs; gp.pMultisampleState = &ms;
        gp.pColorBlendState = &cb; gp.layout = v.layout; gp.renderPass = rp; gp.subpass = 0;
        if (ps && (ps->depth_test_enable || ps->stencil_enable)) gp.pDepthStencilState = &dss;
        if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &v.pipe) == VK_SUCCESS) v.ok = true;
    }

    VkDeviceSize bytes = (VkDeviceSize)W * H * 4;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer rb; vkCreateBuffer(dev, &bci, nullptr, &rb);
    VkMemoryRequirements br; vkGetBufferMemoryRequirements(dev, rb, &br);
    VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; bai.allocationSize = br.size;
    bai.memoryTypeIndex = pick(br.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory bmem; vkAllocateMemory(dev, &bai, nullptr, &bmem); vkBindBufferMemory(dev, rb, bmem, 0);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = qfi;
    VkCommandPool pool; vkCreateCommandPool(dev, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cbai, &cmd);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);
    // Preload the color attachment with the seed pixels (render-target persistence): staging upload +
    // transition to COLOR_ATTACHMENT_OPTIMAL, matching att[0]'s LOAD/initialLayout above.
    VkBuffer seedbuf = VK_NULL_HANDLE; VkDeviceMemory seedmem = VK_NULL_HANDLE;
    if (seed_rgba) {
        VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sci.size = bytes; sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        vkCreateBuffer(dev, &sci, nullptr, &seedbuf);
        VkMemoryRequirements sr; vkGetBufferMemoryRequirements(dev, seedbuf, &sr);
        VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; sai.allocationSize = sr.size;
        sai.memoryTypeIndex = pick(sr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(dev, &sai, nullptr, &seedmem); vkBindBufferMemory(dev, seedbuf, seedmem, 0);
        void* sp = nullptr; vkMapMemory(dev, seedmem, 0, bytes, 0, &sp);
        memcpy(sp, seed_rgba, (size_t)bytes); vkUnmapMemory(dev, seedmem);
        VkImageMemoryBarrier s0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        s0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; s0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        s0.image = img; s0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        s0.srcAccessMask = 0; s0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &s0);
        VkBufferImageCopy sc{}; sc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; sc.imageExtent = {W, H, 1};
        vkCmdCopyBufferToImage(cmd, seedbuf, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &sc);
        VkImageMemoryBarrier s1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        s1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; s1.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        s1.image = img; s1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        s1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        s1.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &s1);
    }
    // Upload each draw's textures: UNDEFINED -> TRANSFER_DST, copy staging buffer, TRANSFER_DST -> SHADER_READ.
    for (auto& v : dv) for (size_t i = 0; i < v.R.size(); i++) {
        if (!v.R[i].is_texture()) continue;
        VkImageMemoryBarrier b0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b0.image = v.timg[i]; b0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b0.srcAccessMask = 0; b0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b0);
        VkBufferImageCopy tc{}; tc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; tc.imageExtent = {v.R[i].tw, v.R[i].th, 1};
        vkCmdCopyBufferToImage(cmd, v.tstage[i], v.timg[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &tc);
        VkImageMemoryBarrier b1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b1.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b1.image = v.timg[i]; b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        // The dst stage must cover EVERY stage that samples this image. #376 made set-0 textures
        // VS-visible (stageFlags VERTEX|FRAGMENT), so a vertex texture fetch reads it in the VERTEX
        // stage — a FRAGMENT-only barrier leaves the transfer-write→shader-read dependency unordered
        // for that stage (SYNC-HAZARD-READ-AFTER-WRITE; garbage vertex fetch on GPUs that don't
        // over-synchronize). Include the vertex stage to match the binding's stageFlags (#454).
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b1);
    }
    // Clear color: the caller's clear_rgba (game fast-clear / black on the live path), else the
    // legacy diagnostic blue. PROSPER_CLEAR_DEBUG forces blue back on even when a color is passed.
    float cc[4] = {0.0f, 0.0f, 1.0f, 1.0f};   // diagnostic blue
    if (clear_rgba && getenv("PROSPER_CLEAR_DEBUG") == nullptr)
        for (int i = 0; i < 4; i++) cc[i] = clear_rgba[i];
    VkClearValue clear[2]{}; clear[0].color = {{cc[0], cc[1], cc[2], cc[3]}};
    clear[1].depthStencil = {depth_clear, stencil_clear};   // guest DB_DEPTH_CLEAR / compare-op default (#371)
    VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpbi.renderPass = rp; rpbi.framebuffer = fb; rpbi.renderArea = {{0, 0}, {W, H}}; rpbi.clearValueCount = use_ds ? 2 : 1; rpbi.pClearValues = clear;
    if (getenv("PROSPER_PIPELOG")) {   // diag: how many draws' pipelines built + will be recorded
        int nok = 0; for (auto& v : dv) if (v.ok) nok++;
        fprintf(stderr, "[pipe] %zu draws, %d pipelines OK, use_depth=%d use_stencil=%d; counts:", dv.size(), nok, (int)use_depth, (int)use_stencil);
        for (auto& v : dv) fprintf(stderr, " %s%u", v.ok ? "" : "SKIP", v.icount ? v.icount : v.vcount);
        fprintf(stderr, "\n");
    }
    // ONE render pass (cleared once): record every realized draw with its own pipeline + descriptors.
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    for (auto& v : dv) {
        if (!v.ok) continue;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v.pipe);
        if (v.use_desc) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v.layout, 0, v.n_sets, v.dsets.data(), 0, nullptr);
        if (v.icount) {
            vkCmdBindIndexBuffer(cmd, v.ibuf, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, v.icount, 1, 0, 0, 0);
        } else {
            vkCmdDraw(cmd, v.vcount, 1, 0, 0);
        }
    }
    vkCmdEndRenderPass(cmd);
    VkBufferImageCopy cp{}; cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &cp);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; vkCreateFence(dev, &fci, nullptr, &fence);
    vkQueueSubmit(queue, 1, &si, fence); vkWaitForFences(dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000);

    out.resize(bytes);
    void* mp = nullptr; vkMapMemory(dev, bmem, 0, bytes, 0, &mp);
    for (VkDeviceSize i = 0; i < bytes; i++) out[i] = ((const uint8_t*)mp)[i];
    vkUnmapMemory(dev, bmem);

    // PROSPER_DRAW_ISO + PROSPER_ISO_AT="x,y": per-draw kill isolation (generalizes the #240 title harness
    // to any submit / any target pixel). On the FIRST submit whose rendered pixel at (x,y) is lit
    // (non-background), re-render THIS exact submit once per killed-draw index and report which draw lights
    // that pixel — the kill index that turns (x,y) dark is the culprit. Reuses the built pipelines/
    // descriptors; a fresh clear each pass. Env-gated, no default behavior. Used to locate a stray primitive
    // such as the #298 menu focus-ring sliver. Dumps iso_kill_<k>.bmp to PROSPER_FRAME_DIR.
    if (getenv("PROSPER_DRAW_ISO") && getenv("PROSPER_ISO_AT")) {
        static bool iso_done = false;
        int tx = -1, ty = -1; sscanf(getenv("PROSPER_ISO_AT"), "%d,%d", &tx, &ty);
        // Optional PROSPER_ISO_RGB="r,g,b" (+ PROSPER_ISO_TOL, default 45): the target submit is the first
        // whose pixel at (x,y) matches that color within tol — robust against an earlier full-screen submit
        // (e.g. the intro cutscene) that merely lights the pixel a different color. Unset -> any non-background.
        int wr = -1, wg = 0, wb = 0, tol = getenv("PROSPER_ISO_TOL") ? atoi(getenv("PROSPER_ISO_TOL")) : 45;
        if (getenv("PROSPER_ISO_RGB")) sscanf(getenv("PROSPER_ISO_RGB"), "%d,%d,%d", &wr, &wg, &wb);
        auto lit_at = [&](const std::vector<uint8_t>& buf) -> bool {
            if (tx < 0 || ty < 0 || (uint32_t)tx >= W || (uint32_t)ty >= H) return false;
            const uint8_t* p = &buf[((size_t)ty * W + tx) * 4];
            if (wr >= 0) return abs((int)p[0]-wr) <= tol && abs((int)p[1]-wg) <= tol && abs((int)p[2]-wb) <= tol;
            return p[0] > 40 || p[1] > 40 || p[2] > 40;
        };
        // Optional second reference pixel: reported alongside the target so we can tell whether the culprit
        // draw ALSO paints a legit element (e.g. the active focus ring) or only the stray pixel.
        int rx = -1, ry = -1; if (getenv("PROSPER_ISO_AT2")) sscanf(getenv("PROSPER_ISO_AT2"), "%d,%d", &rx, &ry);
        if (!iso_done && lit_at(out)) {
            iso_done = true;
            const char* fd = getenv("PROSPER_FRAME_DIR"); std::string dir = fd ? fd : ".";
            fprintf(stderr, "[iso] submit lights (%d,%d): %zu draws; re-rendering per killed draw\n", tx, ty, dv.size());
            // Characterize every draw in the target submit (blend/write-mask/viewport/textures/vertex count).
            for (size_t di = 0; di < dv.size(); di++) {
                const prosper::gpu::ResolvedPipelineState* ps = draws[di].ps; DV& v = dv[di];
                fprintf(stderr, "[iso]  draw#%zu %s cnt=%u", di, v.ok ? "OK" : "SKIP", v.icount ? v.icount : v.vcount);
                if (ps) fprintf(stderr, " blend=%d src=%u dst=%u cwm=0x%x vp_y=%.0f vp_h=%.0f depth=%d/%d",
                                (int)ps->blend_enable, ps->src_color_blend_factor, ps->dst_color_blend_factor,
                                ps->color_write_mask, ps->viewport_y, ps->viewport_h,
                                (int)ps->depth_test_enable, (int)ps->depth_write_enable);
                int nt = 0; for (auto& r : v.R) if (r.is_texture()) { fprintf(stderr, " tex%d=%ux%u", nt, r.tw, r.th); nt++; }
                fprintf(stderr, "\n");
            }
            VkBufferImageCopy cp2{}; cp2.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp2.imageExtent = {W, H, 1};
            for (int kk = -1; kk < (int)dv.size(); kk++) {
                VkCommandBuffer c2; VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
                vkAllocateCommandBuffers(dev, &ai, &c2); vkBeginCommandBuffer(c2, &cbbi);
                vkCmdBeginRenderPass(c2, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
                for (size_t di = 0; di < dv.size(); di++) { auto& v = dv[di]; if (!v.ok) continue; if ((int)di == kk) continue;
                    vkCmdBindPipeline(c2, VK_PIPELINE_BIND_POINT_GRAPHICS, v.pipe);
                    if (v.use_desc) vkCmdBindDescriptorSets(c2, VK_PIPELINE_BIND_POINT_GRAPHICS, v.layout, 0, v.n_sets, v.dsets.data(), 0, nullptr);
                    if (v.icount) { vkCmdBindIndexBuffer(c2, v.ibuf, 0, VK_INDEX_TYPE_UINT32); vkCmdDrawIndexed(c2, v.icount, 1, 0, 0, 0); }
                    else vkCmdDraw(c2, v.vcount, 1, 0, 0);
                }
                vkCmdEndRenderPass(c2);
                vkCmdCopyImageToBuffer(c2, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &cp2);
                vkEndCommandBuffer(c2); vkResetFences(dev, 1, &fence);
                VkSubmitInfo si2{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si2.commandBufferCount = 1; si2.pCommandBuffers = &c2;
                vkQueueSubmit(queue, 1, &si2, fence); vkWaitForFences(dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000);
                std::vector<uint8_t> px(bytes); void* m2 = nullptr; vkMapMemory(dev, bmem, 0, bytes, 0, &m2);
                for (VkDeviceSize i = 0; i < bytes; i++) px[i] = ((const uint8_t*)m2)[i]; vkUnmapMemory(dev, bmem);
                const uint8_t* tp = &px[((size_t)ty * W + tx) * 4];
                bool lit = tp[0] > 40 || tp[1] > 40 || tp[2] > 40;
                // Total gold pixels (ring color) in the frame: if killing a draw drops this only by the
                // sliver's ~size, that draw paints ONLY the sliver; a large drop means it also paints the ring.
                size_t gold_n = 0;
                for (size_t q = 0; q < (size_t)W * H; q++) { const uint8_t* g = &px[q * 4];
                    if (g[0] > 140 && g[1] > 100 && g[2] < 90 && (int)g[0] - (int)g[2] > 60) gold_n++; }
                char ref[96] = "";
                if (rx >= 0 && ry >= 0 && (uint32_t)rx < W && (uint32_t)ry < H) {
                    const uint8_t* rp = &px[((size_t)ry * W + rx) * 4];
                    snprintf(ref, sizeof ref, "  ref(%d,%d)=%u,%u,%u %s", rx, ry, rp[0], rp[1], rp[2],
                             (rp[0] > 40 || rp[1] > 40 || rp[2] > 40) ? "lit" : "DARK");
                }
                fprintf(stderr, "[iso]  kill=%d -> (%d,%d) rgb=%u,%u,%u %s  gold_px=%zu%s\n", kk, tx, ty, tp[0], tp[1], tp[2],
                        lit ? "LIT" : "dark <<< THIS DRAW paints the pixel", gold_n, ref);
                char fn[512]; snprintf(fn, sizeof fn, "%s/iso_kill_%d.bmp", dir.c_str(), kk); dump_bmp(fn, px, W, H);
                vkFreeCommandBuffers(dev, pool, 1, &c2);
            }
            fprintf(stderr, "[iso] done: the kill index marked 'dark' is the draw painting (%d,%d)\n", tx, ty);
        }
    }

    vkDestroyFence(dev, fence, nullptr); vkDestroyCommandPool(dev, pool, nullptr);
    for (auto& v : dv) {   // per-draw objects
        if (v.pipe)   vkDestroyPipeline(dev, v.pipe, nullptr);
        if (v.layout) vkDestroyPipelineLayout(dev, v.layout, nullptr);
        if (v.ibuf)   vkDestroyBuffer(dev, v.ibuf, nullptr);
        if (v.ibmem)  vkFreeMemory(dev, v.ibmem, nullptr);
        if (v.vs)     vkDestroyShaderModule(dev, v.vs, nullptr);
        if (v.fs)     vkDestroyShaderModule(dev, v.fs, nullptr);
        if (v.dpool)  vkDestroyDescriptorPool(dev, v.dpool, nullptr);
        for (auto d : v.dsls) if (d) vkDestroyDescriptorSetLayout(dev, d, nullptr);
        for (size_t i = 0; i < v.R.size(); i++) {
            if (v.sbuf[i])     vkDestroyBuffer(dev, v.sbuf[i], nullptr);
            if (v.sbmem[i])    vkFreeMemory(dev, v.sbmem[i], nullptr);
            if (v.tsamp[i])    vkDestroySampler(dev, v.tsamp[i], nullptr);
            if (v.tview[i])    vkDestroyImageView(dev, v.tview[i], nullptr);
            if (v.timg[i])     vkDestroyImage(dev, v.timg[i], nullptr);
            if (v.tmem[i])     vkFreeMemory(dev, v.tmem[i], nullptr);
            if (v.tstage[i])   vkDestroyBuffer(dev, v.tstage[i], nullptr);
            if (v.tstagemem[i])vkFreeMemory(dev, v.tstagemem[i], nullptr);
        }
    }
    if (seedbuf) vkDestroyBuffer(dev, seedbuf, nullptr);
    if (seedmem) vkFreeMemory(dev, seedmem, nullptr);
    vkDestroyBuffer(dev, rb, nullptr); vkFreeMemory(dev, bmem, nullptr);
    vkDestroyFramebuffer(dev, fb, nullptr); vkDestroyRenderPass(dev, rp, nullptr); vkDestroyImageView(dev, view, nullptr);
    vkDestroyImage(dev, img, nullptr); vkFreeMemory(dev, imem, nullptr);
    if (use_depth) { vkDestroyImageView(dev, dview, nullptr); vkDestroyImage(dev, dimg, nullptr); vkFreeMemory(dev, dmem, nullptr); }
    // NB: dev/instance are the persistent RenderVkCtx — do NOT destroy them here (reused across calls).
    return out;
}

// Single-draw entry — a thin wrapper over render_draws_rgba, preserving the exact signature/behavior the
// recompiled-shader render tests rely on. Builds one draw's set-tagged resources (`gres`, or the legacy
// cbuf/vbuf@bindings 2/3 mirrored into both sets + optional `tex` in set 1).
inline std::vector<uint8_t> render_triangle_rgba(const std::vector<uint32_t>& vert,
                                                 const std::vector<uint32_t>& frag,
                                                 uint32_t W, uint32_t H,
                                                 const prosper::gpu::ResolvedPipelineState* ps = nullptr,
                                                 const std::vector<uint32_t>* vbuf = nullptr,
                                                 const std::vector<uint32_t>* cbuf = nullptr,
                                                 const TexDesc* tex = nullptr,
                                                 const std::vector<FrameResource>* gres = nullptr,
                                                 uint32_t vcount = 3) {
    BackendDraw d; d.vs = vert; d.fs = frag; d.ps = ps; d.vcount = vcount;
    if (gres && !gres->empty()) { d.R = *gres; }
    else {
        for (uint32_t s2 = 0; s2 < 2; s2++) {
            FrameResource b2; b2.binding = 2; b2.set = s2; if (cbuf) b2.dwords = *cbuf; d.R.push_back(std::move(b2));
            FrameResource b3; b3.binding = 3; b3.set = s2; if (vbuf) b3.dwords = *vbuf; d.R.push_back(std::move(b3));
        }
        if (tex) { FrameResource t; t.binding = tex->binding; t.set = 1; t.tex_rgba = tex->rgba; t.tw = tex->w; t.th = tex->h; t.max_aniso_ratio = tex->max_aniso_ratio; d.R.push_back(std::move(t)); }
    }
    return render_draws_rgba({std::move(d)}, W, H);
}

} // namespace prosper::test
