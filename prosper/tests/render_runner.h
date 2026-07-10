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
struct TexDesc { uint32_t binding; uint32_t w; uint32_t h; const uint8_t* rgba; };

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

inline std::vector<uint8_t> render_draws_rgba(const std::vector<BackendDraw>& draws, uint32_t W, uint32_t H) {
    std::vector<uint8_t> out;
    if (draws.empty()) return out;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS || !inst) return out;
    uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, nullptr);
    if (!nd) { vkDestroyInstance(inst, nullptr); return out; }
    std::vector<VkPhysicalDevice> devs(nd); vkEnumeratePhysicalDevices(inst, &nd, devs.data());
    VkPhysicalDevice phys = devs[0];
    uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nqf); vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qf.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nqf; i++) if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qfi = i; break; }
    if (qfi == UINT32_MAX) { vkDestroyInstance(inst, nullptr); return out; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    // robustBufferAccess: out-of-range storage-buffer accesses are well-defined, so a predicated memory
    // op run by an inactive lane (narrowed EXEC) can't fault.
    VkPhysicalDeviceFeatures feats{}; feats.robustBufferAccess = VK_TRUE; dci.pEnabledFeatures = &feats;
    // robustImageAccess (VK_EXT_image_robustness; core in 1.3): the recompiled storage-image load
    // path issues OpImageRead for ALL invocations — including EXEC-inactive lanes whose coordinates
    // can be out of range — relying on OOB image reads returning zero (#131). This is the LIVE
    // render backend (boot_trace registers render_draws_rgba as the submit renderer), so real game
    // shaders run here on non-multiple image sizes. Feature-query guarded: a device without it
    // still creates (visibly logged risk is preferable to failing device creation outright).
    VkPhysicalDeviceImageRobustnessFeaturesEXT irf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT};
    const char* img_robust_ext[] = { "VK_EXT_image_robustness" };
    { uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(phys, nullptr, &ne, nullptr);
      std::vector<VkExtensionProperties> de(ne);
      vkEnumerateDeviceExtensionProperties(phys, nullptr, &ne, de.data());
      for (uint32_t i = 0; i < ne; i++) if (!strcmp(de[i].extensionName, "VK_EXT_image_robustness")) {
          VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
          f2.pNext = &irf; vkGetPhysicalDeviceFeatures2(phys, &f2);
          if (irf.robustImageAccess) { dci.pNext = &irf;
              dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = img_robust_ext; }
          break;
      } }
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS || !dev) { vkDestroyInstance(inst, nullptr); return out; }
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);
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
    for (const auto& d : draws) { if (d.ps && d.ps->depth_test_enable) use_depth = true;
                                  if (d.ps && d.ps->stencil_enable)    use_stencil = true; }
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
    imgci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
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
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; att[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
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
            cba.srcAlphaBlendFactor = (VkBlendFactor)ps->src_color_blend_factor;   // mirror color for alpha
            cba.dstAlphaBlendFactor = (VkBlendFactor)ps->dst_color_blend_factor;
            cba.alphaBlendOp        = (VkBlendOp)ps->color_blend_op;
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
                s.reference   = ps->stencil_ref[fb];
                return s;
            };
            dss.front = mkop(0); dss.back = mkop(1);
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
                    lb[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; lb[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
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
                    // NOT applied here (need machinery the current path lacks — decoded under GFXLOG only):
                    //   anisotropy (needs the samplerAnisotropy device feature), depth_compare_func (needs a
                    //   depth/shadow sampler over a depth image), unnormalized coords (strict validity rules).
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
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b1);
    }
    VkClearValue clear[2]{}; clear[0].color = {{0.0f, 0.0f, 1.0f, 1.0f}};   // blue
    clear[1].depthStencil = {0.5f, 0};   // depth cleared to 0.5 (fragments at z=0.0 pass LESS, fail GREATER)
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
    vkDestroyBuffer(dev, rb, nullptr); vkFreeMemory(dev, bmem, nullptr);
    vkDestroyFramebuffer(dev, fb, nullptr); vkDestroyRenderPass(dev, rp, nullptr); vkDestroyImageView(dev, view, nullptr);
    vkDestroyImage(dev, img, nullptr); vkFreeMemory(dev, imem, nullptr);
    if (use_depth) { vkDestroyImageView(dev, dview, nullptr); vkDestroyImage(dev, dimg, nullptr); vkFreeMemory(dev, dmem, nullptr); }
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);
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
        if (tex) { FrameResource t; t.binding = tex->binding; t.set = 1; t.tex_rgba = tex->rgba; t.tw = tex->w; t.th = tex->h; d.R.push_back(std::move(t)); }
    }
    return render_draws_rgba({std::move(d)}, W, H);
}

} // namespace prosper::test
