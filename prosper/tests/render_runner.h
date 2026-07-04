// render_runner.h — inline helper to render a single triangle (3 vertices, no vertex input) with a
// given vertex + fragment SPIR-V pair into a WxH RGBA8 image, clearing to blue first, and return the
// pixels. Used to verify recompiled shaders end-to-end (render -> readback -> pixel asserts). The
// including test links Vulkan::Vulkan.
#pragma once
#include <vulkan/vulkan.h>
#include "../src/gpu/render_state.hpp"
#include <cstdint>
#include <cstdio>
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
inline std::vector<uint8_t> render_triangle_rgba(const std::vector<uint32_t>& vert,
                                                 const std::vector<uint32_t>& frag,
                                                 uint32_t W, uint32_t H,
                                                 const prosper::gpu::ResolvedPipelineState* ps = nullptr,
                                                 const std::vector<uint32_t>* vbuf = nullptr,
                                                 const std::vector<uint32_t>* cbuf = nullptr,
                                                 const TexDesc* tex = nullptr) {
    std::vector<uint8_t> out;
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
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS || !dev) { vkDestroyInstance(inst, nullptr); return out; }
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);
    VkPhysicalDeviceMemoryProperties memp; vkGetPhysicalDeviceMemoryProperties(phys, &memp);
    auto pick = [&](uint32_t bits, VkMemoryPropertyFlags want) -> uint32_t {
        for (uint32_t i = 0; i < memp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memp.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX; };
    const VkFormat FMT = VK_FORMAT_R8G8B8A8_UNORM;
    // Depth is opt-in: only when the resolved state enables the depth test. Every caller that passes
    // no state (or depth disabled) takes the exact color-only path below — unchanged.
    const bool use_depth = ps && ps->depth_test_enable;
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
    VkShaderModule vs = mkmod(vert), fs = mkmod(frag);
    if (!vs || !fs) return out;   // rejected SPIR-V
    VkPipelineShaderStageCreateInfo st[2]{};
    st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = vs; st[0].pName = "main";
    st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fs; st[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = ps ? (VkPrimitiveTopology)ps->topology : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{0, 0, (float)W, (float)H, 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
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
    if (use_depth) {
        dss.depthTestEnable  = VK_TRUE;
        dss.depthWriteEnable = ps->depth_write_enable ? VK_TRUE : VK_FALSE;
        dss.depthCompareOp   = (VkCompareOp)ps->depth_compare_op;
    }
    // Optional resource descriptor set: constant buffer at binding 2, vertex buffer at binding 3, both
    // as storage buffers — the layout declare_cbufs() emits into a table-recompiled shader. Whichever
    // is null is bound as a 1-dword zero buffer so the (always-declared) binding is still satisfied.
    const bool use_desc = (vbuf || cbuf || tex);
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE; VkDescriptorPool dpool = VK_NULL_HANDLE; VkDescriptorSet dset = VK_NULL_HANDLE;
    VkBuffer sb[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE}; VkDeviceMemory sbmem[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    // Texture (combined image sampler) objects — created here, uploaded in the command buffer below.
    VkImage timg = VK_NULL_HANDLE; VkDeviceMemory tmem = VK_NULL_HANDLE; VkImageView tview = VK_NULL_HANDLE;
    VkSampler tsamp = VK_NULL_HANDLE; VkBuffer tstage = VK_NULL_HANDLE; VkDeviceMemory tstagemem = VK_NULL_HANDLE;
    if (use_desc) {
        auto mkstore = [&](const std::vector<uint32_t>* data, VkBuffer& buf, VkDeviceMemory& mem) {
            VkDeviceSize sz = (data && !data->empty()) ? (VkDeviceSize)data->size() * 4 : 4;
            VkBufferCreateInfo sbci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            sbci.size = sz; sbci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            vkCreateBuffer(dev, &sbci, nullptr, &buf);
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = mr.size;
            mai.memoryTypeIndex = pick(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(dev, &mai, nullptr, &mem); vkBindBufferMemory(dev, buf, mem, 0);
            void* p = nullptr; vkMapMemory(dev, mem, 0, sz, 0, &p);
            if (data && !data->empty()) for (size_t i = 0; i < data->size(); i++) ((uint32_t*)p)[i] = (*data)[i];
            else ((uint32_t*)p)[0] = 0;
            vkUnmapMemory(dev, mem);
        };
        mkstore(cbuf, sb[0], sbmem[0]);   // binding 2
        mkstore(vbuf, sb[1], sbmem[1]);   // binding 3

        // Storage bindings 2 & 3 are always present (declare_cbufs emits them whenever rt != null); the
        // texture, if any, adds one COMBINED_IMAGE_SAMPLER at tex->binding.
        VkDescriptorSetLayoutBinding lb[3]{}; uint32_t nb = 2;
        lb[0].binding = 2; lb[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; lb[0].descriptorCount = 1;
        lb[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        lb[1].binding = 3; lb[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; lb[1].descriptorCount = 1;
        lb[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        if (tex) {
            lb[2].binding = tex->binding; lb[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            lb[2].descriptorCount = 1; lb[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; nb = 3;

            VkImageCreateInfo tci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            tci.imageType = VK_IMAGE_TYPE_2D; tci.format = VK_FORMAT_R8G8B8A8_UNORM; tci.extent = {tex->w, tex->h, 1};
            tci.mipLevels = 1; tci.arrayLayers = 1; tci.samples = VK_SAMPLE_COUNT_1_BIT; tci.tiling = VK_IMAGE_TILING_OPTIMAL;
            tci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            vkCreateImage(dev, &tci, nullptr, &timg);
            VkMemoryRequirements tr; vkGetImageMemoryRequirements(dev, timg, &tr);
            VkMemoryAllocateInfo tai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; tai.allocationSize = tr.size;
            tai.memoryTypeIndex = pick(tr.memoryTypeBits, 0); vkAllocateMemory(dev, &tai, nullptr, &tmem);
            vkBindImageMemory(dev, timg, tmem, 0);
            VkImageViewCreateInfo tvci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            tvci.image = timg; tvci.viewType = VK_IMAGE_VIEW_TYPE_2D; tvci.format = VK_FORMAT_R8G8B8A8_UNORM;
            tvci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; vkCreateImageView(dev, &tvci, nullptr, &tview);
            VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            sci.magFilter = sci.minFilter = VK_FILTER_NEAREST; sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            vkCreateSampler(dev, &sci, nullptr, &tsamp);
            // Staging buffer with the RGBA texels (uploaded into the image in the command buffer below).
            VkDeviceSize tbytes = (VkDeviceSize)tex->w * tex->h * 4;
            VkBufferCreateInfo stci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; stci.size = tbytes; stci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            vkCreateBuffer(dev, &stci, nullptr, &tstage);
            VkMemoryRequirements sr; vkGetBufferMemoryRequirements(dev, tstage, &sr);
            VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; sai.allocationSize = sr.size;
            sai.memoryTypeIndex = pick(sr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(dev, &sai, nullptr, &tstagemem); vkBindBufferMemory(dev, tstage, tstagemem, 0);
            void* sp = nullptr; vkMapMemory(dev, tstagemem, 0, tbytes, 0, &sp);
            for (VkDeviceSize i = 0; i < tbytes; i++) ((uint8_t*)sp)[i] = tex->rgba[i];
            vkUnmapMemory(dev, tstagemem);
        }
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = nb; dslci.pBindings = lb; vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);
        VkDescriptorPoolSize psz[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1; dpci.poolSizeCount = tex ? 2 : 1; dpci.pPoolSizes = psz; vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool);
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl; vkAllocateDescriptorSets(dev, &dsai, &dset);
        VkDescriptorBufferInfo dbi[2] = {{sb[0], 0, VK_WHOLE_SIZE}, {sb[1], 0, VK_WHOLE_SIZE}};
        VkDescriptorImageInfo dii{tsamp, tview, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w[3]{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[0].dstSet = dset; w[0].dstBinding = 2;
        w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &dbi[0];
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[1].dstSet = dset; w[1].dstBinding = 3;
        w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &dbi[1];
        if (tex) {
            w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[2].dstSet = dset; w[2].dstBinding = tex->binding;
            w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[2].pImageInfo = &dii;
        }
        vkUpdateDescriptorSets(dev, nb, w, 0, nullptr);
    }
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    if (use_desc) { plci.setLayoutCount = 1; plci.pSetLayouts = &dsl; }
    VkPipelineLayout layout; vkCreatePipelineLayout(dev, &plci, nullptr, &layout);
    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = 2; gp.pStages = st; gp.pVertexInputState = &vin; gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vpst; gp.pRasterizationState = &rs; gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cb; gp.layout = layout; gp.renderPass = rp; gp.subpass = 0;
    if (use_depth) gp.pDepthStencilState = &dss;
    VkPipeline pipe;
    if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe) != VK_SUCCESS) return out;

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
    if (tex) {
        // Upload the texels: UNDEFINED -> TRANSFER_DST, copy staging buffer, TRANSFER_DST -> SHADER_READ.
        VkImageMemoryBarrier b0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b0.image = timg; b0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b0.srcAccessMask = 0; b0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b0);
        VkBufferImageCopy tc{}; tc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; tc.imageExtent = {tex->w, tex->h, 1};
        vkCmdCopyBufferToImage(cmd, tstage, timg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &tc);
        VkImageMemoryBarrier b1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b1.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b1.image = timg; b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b1);
    }
    VkClearValue clear[2]{}; clear[0].color = {{0.0f, 0.0f, 1.0f, 1.0f}};   // blue
    clear[1].depthStencil = {0.5f, 0};   // depth cleared to 0.5 (fragments at z=0.0 pass LESS, fail GREATER)
    VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpbi.renderPass = rp; rpbi.framebuffer = fb; rpbi.renderArea = {{0, 0}, {W, H}}; rpbi.clearValueCount = use_depth ? 2 : 1; rpbi.pClearValues = clear;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    if (use_desc) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &dset, 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
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
    vkDestroyPipeline(dev, pipe, nullptr); vkDestroyPipelineLayout(dev, layout, nullptr);
    vkDestroyShaderModule(dev, vs, nullptr); vkDestroyShaderModule(dev, fs, nullptr);
    vkDestroyBuffer(dev, rb, nullptr); vkFreeMemory(dev, bmem, nullptr);
    vkDestroyFramebuffer(dev, fb, nullptr); vkDestroyRenderPass(dev, rp, nullptr); vkDestroyImageView(dev, view, nullptr);
    vkDestroyImage(dev, img, nullptr); vkFreeMemory(dev, imem, nullptr);
    if (use_depth) { vkDestroyImageView(dev, dview, nullptr); vkDestroyImage(dev, dimg, nullptr); vkFreeMemory(dev, dmem, nullptr); }
    if (use_desc) {
        vkDestroyDescriptorPool(dev, dpool, nullptr); vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        for (int i = 0; i < 2; i++) { vkDestroyBuffer(dev, sb[i], nullptr); vkFreeMemory(dev, sbmem[i], nullptr); }
    }
    if (tex) {
        vkDestroySampler(dev, tsamp, nullptr); vkDestroyImageView(dev, tview, nullptr);
        vkDestroyImage(dev, timg, nullptr); vkFreeMemory(dev, tmem, nullptr);
        vkDestroyBuffer(dev, tstage, nullptr); vkFreeMemory(dev, tstagemem, nullptr);
    }
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);
    return out;
}

} // namespace prosper::test
