// render_runner.h — inline helper to render a single triangle (3 vertices, no vertex input) with a
// given vertex + fragment SPIR-V pair into a WxH RGBA8 image, clearing to blue first, and return the
// pixels. Used to verify recompiled shaders end-to-end (render -> readback -> pixel asserts). The
// including test links Vulkan::Vulkan.
#pragma once
#include <vulkan/vulkan.h>
#include "../src/gpu/render_state.hpp"
#include <cstdint>
#include <vector>

namespace prosper::test {

// Returns W*H*4 RGBA bytes, or {} on any Vulkan failure (incl. a rejected SPIR-V module). When `ps`
// is non-null, the pipeline's fixed-function state (topology, blend, color write mask) is taken from
// the resolved RDNA2 render-state — this is how the back-half realizes a GpuState as a real VkPipeline.
inline std::vector<uint8_t> render_triangle_rgba(const std::vector<uint32_t>& vert,
                                                 const std::vector<uint32_t>& frag,
                                                 uint32_t W, uint32_t H,
                                                 const prosper::gpu::ResolvedPipelineState* ps = nullptr) {
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
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS || !dev) { vkDestroyInstance(inst, nullptr); return out; }
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);
    VkPhysicalDeviceMemoryProperties memp; vkGetPhysicalDeviceMemoryProperties(phys, &memp);
    auto pick = [&](uint32_t bits, VkMemoryPropertyFlags want) -> uint32_t {
        for (uint32_t i = 0; i < memp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memp.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX; };
    const VkFormat FMT = VK_FORMAT_R8G8B8A8_UNORM;

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

    VkAttachmentDescription att{}; att.format = FMT; att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference ar{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{}; sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1; sub.pColorAttachments = &ar;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1; rpci.pAttachments = &att; rpci.subpassCount = 1; rpci.pSubpasses = &sub;
    VkRenderPass rp; vkCreateRenderPass(dev, &rpci, nullptr, &rp);
    VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbci.renderPass = rp; fbci.attachmentCount = 1; fbci.pAttachments = &view; fbci.width = W; fbci.height = H; fbci.layers = 1;
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
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout; vkCreatePipelineLayout(dev, &plci, nullptr, &layout);
    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = 2; gp.pStages = st; gp.pVertexInputState = &vin; gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vpst; gp.pRasterizationState = &rs; gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cb; gp.layout = layout; gp.renderPass = rp; gp.subpass = 0;
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
    VkClearValue clear{}; clear.color = {{0.0f, 0.0f, 1.0f, 1.0f}};   // blue
    VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpbi.renderPass = rp; rpbi.framebuffer = fb; rpbi.renderArea = {{0, 0}, {W, H}}; rpbi.clearValueCount = 1; rpbi.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
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
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);
    return out;
}

} // namespace prosper::test
