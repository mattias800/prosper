// render_runner.h — inline offscreen Vulkan render helpers used by the tests and the live renderer.
// render_draws_rgba(): record N draws (each with its own SPIR-V pair, fixed-function state, and
// descriptor resources) into ONE render pass over a WxH RGBA8 image (cleared to blue), and return the
// pixels — this is how a real multi-draw submit composites (e.g. Unity's background + composite pair).
// render_triangle_rgba(): the original single-draw entry, now a thin wrapper over render_draws_rgba
// (all existing tests keep their exact behavior). The including test links Vulkan::Vulkan.
#pragma once
#include <vulkan/vulkan.h>
#include "../src/gpu/render_state.hpp"
#include <cstdint>
#include <cstdio>
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

// One resource for the general N-binding path: either a storage buffer (dwords non-empty, tex_rgba
// null) or a combined image sampler (tex_rgba set) at `binding`. Lets a real game shader that declares
// several constant/vertex buffers + textures have each bound distinctly.
struct FrameResource {
    uint32_t binding = 0;
    std::vector<uint32_t> dwords;   // storage-buffer contents (empty -> a 1-dword zero buffer)
    const uint8_t* tex_rgba = nullptr;   // non-null => a texture; then tw/th are its dimensions
    uint32_t tw = 0, th = 0;
    bool is_texture() const { return tex_rgba != nullptr; }
};

// One draw for the multi-draw backend: a SPIR-V pair, the draw's OWN resolved fixed-function state
// (null -> defaults: triangle list, no blend, mask 0xF, no depth), the descriptor resources to bind at
// set 0, and the vertex count.
struct BackendDraw {
    const std::vector<uint32_t>* vs = nullptr;
    const std::vector<uint32_t>* fs = nullptr;
    const prosper::gpu::ResolvedPipelineState* ps = nullptr;
    std::vector<FrameResource> res;
    uint32_t vcount = 3;
};

// Record every draw into ONE render pass over a WxH RGBA8 image (cleared to blue) and return the
// pixels, or {} if no draw could be realized. Each draw gets its own pipeline + descriptor set, so
// per-draw blend/mask/topology/shaders are honored — the requirement for a real multi-draw submit.
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
    VkDeviceCreateInfo dcinfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dcinfo.queueCreateInfoCount = 1; dcinfo.pQueueCreateInfos = &qci;
    // robustBufferAccess: out-of-range storage-buffer accesses are well-defined, so a predicated memory
    // op run by an inactive lane (narrowed EXEC) can't fault.
    VkPhysicalDeviceFeatures feats{}; feats.robustBufferAccess = VK_TRUE; dcinfo.pEnabledFeatures = &feats;
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &dcinfo, nullptr, &dev) != VK_SUCCESS || !dev) { vkDestroyInstance(inst, nullptr); return out; }
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);
    VkPhysicalDeviceMemoryProperties memp; vkGetPhysicalDeviceMemoryProperties(phys, &memp);
    auto pick = [&](uint32_t bits, VkMemoryPropertyFlags want) -> uint32_t {
        for (uint32_t i = 0; i < memp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memp.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX; };
    const VkFormat FMT = VK_FORMAT_R8G8B8A8_UNORM;
    // Depth is opt-in: only when SOME draw's resolved state enables the depth test. Draws that don't
    // simply build their pipeline with depthTestEnable=FALSE against the same pass.
    bool use_depth = false;
    for (const auto& d : draws) if (d.ps && d.ps->depth_test_enable) use_depth = true;
    const VkFormat DFMT = VK_FORMAT_D32_SFLOAT;
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

    if (use_depth) {
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
        dvci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCreateImageView(dev, &dvci, nullptr, &dview);
    }

    VkAttachmentDescription att[2]{};
    att[0].format = FMT; att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; att[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    att[1].format = DFMT; att[1].samples = VK_SAMPLE_COUNT_1_BIT;
    att[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference ar{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference dar{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{}; sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1; sub.pColorAttachments = &ar;
    if (use_depth) sub.pDepthStencilAttachment = &dar;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = use_depth ? 2 : 1; rpci.pAttachments = att; rpci.subpassCount = 1; rpci.pSubpasses = &sub;
    VkRenderPass rp; vkCreateRenderPass(dev, &rpci, nullptr, &rp);
    VkImageView fbviews[2] = {view, dview};
    VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbci.renderPass = rp; fbci.attachmentCount = use_depth ? 2 : 1; fbci.pAttachments = fbviews; fbci.width = W; fbci.height = H; fbci.layers = 1;
    VkFramebuffer fb; vkCreateFramebuffer(dev, &fbci, nullptr, &fb);

    auto mkmod = [&](const std::vector<uint32_t>& c) -> VkShaderModule {
        VkShaderModuleCreateInfo s{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        s.codeSize = c.size() * 4; s.pCode = c.data(); VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(dev, &s, nullptr, &m); return m; };

    // Per-draw Vulkan objects (freed at the end). `valid` marks draws that fully realized — an
    // invalid draw (rejected SPIR-V / failed pipeline) is skipped, not fatal to the frame.
    struct ItemVk {
        bool valid = false, use_desc = false;
        VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE; VkDescriptorPool dpool = VK_NULL_HANDLE;
        VkDescriptorSet dset = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE; VkPipeline pipe = VK_NULL_HANDLE;
        std::vector<VkBuffer> sbuf, tstage;
        std::vector<VkDeviceMemory> sbmem, tmem, tstagemem;
        std::vector<VkImage> timg; std::vector<VkImageView> tview; std::vector<VkSampler> tsamp;
    };
    std::vector<ItemVk> iv(draws.size());

    for (size_t di = 0; di < draws.size(); di++) {
        const BackendDraw& d = draws[di]; ItemVk& v = iv[di];
        if (!d.vs || !d.fs) continue;
        v.vs = mkmod(*d.vs); v.fs = mkmod(*d.fs);
        if (!v.vs || !v.fs) continue;                       // rejected SPIR-V -> skip this draw
        const std::vector<FrameResource>& R = d.res;
        v.use_desc = !R.empty();
        v.sbuf.assign(R.size(), VK_NULL_HANDLE);  v.tstage.assign(R.size(), VK_NULL_HANDLE);
        v.sbmem.assign(R.size(), VK_NULL_HANDLE); v.tmem.assign(R.size(), VK_NULL_HANDLE);
        v.tstagemem.assign(R.size(), VK_NULL_HANDLE);
        v.timg.assign(R.size(), VK_NULL_HANDLE);  v.tview.assign(R.size(), VK_NULL_HANDLE);
        v.tsamp.assign(R.size(), VK_NULL_HANDLE);
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
                    tvci.image = v.timg[i]; tvci.viewType = VK_IMAGE_VIEW_TYPE_2D; tvci.format = VK_FORMAT_R8G8B8A8_UNORM;
                    tvci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; vkCreateImageView(dev, &tvci, nullptr, &v.tview[i]);
                    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
                    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR; sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
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
            VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dslci.bindingCount = (uint32_t)lb.size(); dslci.pBindings = lb.data(); vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &v.dsl);
            VkDescriptorPoolSize psz[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, n_storage ? n_storage : 1},
                                           {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, n_sampler ? n_sampler : 1}};
            VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            dpci.maxSets = 1; dpci.poolSizeCount = n_sampler ? 2 : 1; dpci.pPoolSizes = psz; vkCreateDescriptorPool(dev, &dpci, nullptr, &v.dpool);
            VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dsai.descriptorPool = v.dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &v.dsl; vkAllocateDescriptorSets(dev, &dsai, &v.dset);
            for (auto& x : wr) x.dstSet = v.dset;
            vkUpdateDescriptorSets(dev, (uint32_t)wr.size(), wr.data(), 0, nullptr);
        }
        VkPipelineShaderStageCreateInfo st[2]{};
        st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = v.vs; st[0].pName = "main";
        st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = v.fs; st[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = d.ps ? (VkPrimitiveTopology)d.ps->topology : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkViewport vp{0, 0, (float)W, (float)H, 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
        VkPipelineViewportStateCreateInfo vpst{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vpst.viewportCount = 1; vpst.pViewports = &vp; vpst.scissorCount = 1; vpst.pScissors = &sc;
        VkPipelineRasterizationStateCreateInfo rst{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rst.polygonMode = VK_POLYGON_MODE_FILL; rst.cullMode = VK_CULL_MODE_NONE; rst.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rst.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
        if (d.ps) {
            cba.colorWriteMask = d.ps->color_write_mask;
            cba.blendEnable    = d.ps->blend_enable ? VK_TRUE : VK_FALSE;
            cba.srcColorBlendFactor = (VkBlendFactor)d.ps->src_color_blend_factor;
            cba.dstColorBlendFactor = (VkBlendFactor)d.ps->dst_color_blend_factor;
            cba.colorBlendOp        = (VkBlendOp)d.ps->color_blend_op;
            cba.srcAlphaBlendFactor = (VkBlendFactor)d.ps->src_color_blend_factor;   // mirror color for alpha
            cba.dstAlphaBlendFactor = (VkBlendFactor)d.ps->dst_color_blend_factor;
            cba.alphaBlendOp        = (VkBlendOp)d.ps->color_blend_op;
        }
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1; cb.pAttachments = &cba;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        if (use_depth && d.ps && d.ps->depth_test_enable) {
            dss.depthTestEnable  = VK_TRUE;
            dss.depthWriteEnable = d.ps->depth_write_enable ? VK_TRUE : VK_FALSE;
            dss.depthCompareOp   = (VkCompareOp)d.ps->depth_compare_op;
        }
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        if (v.use_desc) { plci.setLayoutCount = 1; plci.pSetLayouts = &v.dsl; }
        vkCreatePipelineLayout(dev, &plci, nullptr, &v.layout);
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.stageCount = 2; gp.pStages = st; gp.pVertexInputState = &vin; gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vpst; gp.pRasterizationState = &rst; gp.pMultisampleState = &ms;
        gp.pColorBlendState = &cb; gp.layout = v.layout; gp.renderPass = rp; gp.subpass = 0;
        if (use_depth) gp.pDepthStencilState = &dss;
        if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &v.pipe) != VK_SUCCESS) continue;
        v.valid = true;
    }

    bool any_valid = false; for (auto& v : iv) any_valid |= v.valid;
    if (any_valid) {
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
        // Upload every valid draw's textures: UNDEFINED -> TRANSFER_DST, copy staging, -> SHADER_READ.
        for (size_t di = 0; di < draws.size(); di++) {
            ItemVk& v = iv[di]; if (!v.valid) continue;
            for (size_t i = 0; i < draws[di].res.size(); i++) {
                if (!draws[di].res[i].is_texture()) continue;
                VkImageMemoryBarrier b0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                b0.image = v.timg[i]; b0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                b0.srcAccessMask = 0; b0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b0);
                VkBufferImageCopy tc{}; tc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                tc.imageExtent = {draws[di].res[i].tw, draws[di].res[i].th, 1};
                vkCmdCopyBufferToImage(cmd, v.tstage[i], v.timg[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &tc);
                VkImageMemoryBarrier b1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b1.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b1.image = v.timg[i]; b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b1);
            }
        }
        VkClearValue clear[2]{}; clear[0].color = {{0.0f, 0.0f, 1.0f, 1.0f}};   // blue
        clear[1].depthStencil = {0.5f, 0};   // depth cleared to 0.5 (fragments at z=0.0 pass LESS, fail GREATER)
        VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rpbi.renderPass = rp; rpbi.framebuffer = fb; rpbi.renderArea = {{0, 0}, {W, H}}; rpbi.clearValueCount = use_depth ? 2 : 1; rpbi.pClearValues = clear;
        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
        for (size_t di = 0; di < draws.size(); di++) {      // clear once, then every draw in submit order
            ItemVk& v = iv[di]; if (!v.valid) continue;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v.pipe);
            if (v.use_desc) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v.layout, 0, 1, &v.dset, 0, nullptr);
            vkCmdDraw(cmd, draws[di].vcount, 1, 0, 0);
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
        std::memcpy(out.data(), mp, (size_t)bytes);
        vkUnmapMemory(dev, bmem);

        vkDestroyFence(dev, fence, nullptr); vkDestroyCommandPool(dev, pool, nullptr);
        vkDestroyBuffer(dev, rb, nullptr); vkFreeMemory(dev, bmem, nullptr);
    }

    for (auto& v : iv) {
        if (v.pipe)   vkDestroyPipeline(dev, v.pipe, nullptr);
        if (v.layout) vkDestroyPipelineLayout(dev, v.layout, nullptr);
        if (v.vs)     vkDestroyShaderModule(dev, v.vs, nullptr);
        if (v.fs)     vkDestroyShaderModule(dev, v.fs, nullptr);
        if (v.dpool)  vkDestroyDescriptorPool(dev, v.dpool, nullptr);
        if (v.dsl)    vkDestroyDescriptorSetLayout(dev, v.dsl, nullptr);
        for (size_t i = 0; i < v.sbuf.size(); i++) {
            if (v.sbuf[i])      vkDestroyBuffer(dev, v.sbuf[i], nullptr);
            if (v.sbmem[i])     vkFreeMemory(dev, v.sbmem[i], nullptr);
            if (v.tsamp[i])     vkDestroySampler(dev, v.tsamp[i], nullptr);
            if (v.tview[i])     vkDestroyImageView(dev, v.tview[i], nullptr);
            if (v.timg[i])      vkDestroyImage(dev, v.timg[i], nullptr);
            if (v.tmem[i])      vkFreeMemory(dev, v.tmem[i], nullptr);
            if (v.tstage[i])    vkDestroyBuffer(dev, v.tstage[i], nullptr);
            if (v.tstagemem[i]) vkFreeMemory(dev, v.tstagemem[i], nullptr);
        }
    }
    vkDestroyFramebuffer(dev, fb, nullptr); vkDestroyRenderPass(dev, rp, nullptr); vkDestroyImageView(dev, view, nullptr);
    vkDestroyImage(dev, img, nullptr); vkFreeMemory(dev, imem, nullptr);
    if (use_depth) { vkDestroyImageView(dev, dview, nullptr); vkDestroyImage(dev, dimg, nullptr); vkFreeMemory(dev, dmem, nullptr); }
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);
    return out;
}

// The original single-draw entry — a thin wrapper over render_draws_rgba, preserving its exact
// behavior and defaults ({} on rejected SPIR-V / failed pipeline; cbuf@2 / vbuf@3 zero-filled storage
// buffers when `gres` is absent; optional single texture).
inline std::vector<uint8_t> render_triangle_rgba(const std::vector<uint32_t>& vert,
                                                 const std::vector<uint32_t>& frag,
                                                 uint32_t W, uint32_t H,
                                                 const prosper::gpu::ResolvedPipelineState* ps = nullptr,
                                                 const std::vector<uint32_t>* vbuf = nullptr,
                                                 const std::vector<uint32_t>* cbuf = nullptr,
                                                 const TexDesc* tex = nullptr,
                                                 const std::vector<FrameResource>* gres = nullptr,
                                                 uint32_t vcount = 3) {
    BackendDraw d; d.vs = &vert; d.fs = &frag; d.ps = ps; d.vcount = vcount;
    if (gres && !gres->empty()) { d.res = *gres; }
    else {
        FrameResource b2; b2.binding = 2; if (cbuf) b2.dwords = *cbuf; d.res.push_back(std::move(b2));
        FrameResource b3; b3.binding = 3; if (vbuf) b3.dwords = *vbuf; d.res.push_back(std::move(b3));
        if (tex) { FrameResource t; t.binding = tex->binding; t.tex_rgba = tex->rgba; t.tw = tex->w; t.th = tex->h; d.res.push_back(std::move(t)); }
    }
    return render_draws_rgba({d}, W, H);
}

} // namespace prosper::test
