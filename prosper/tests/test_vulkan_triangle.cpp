// test_vulkan_triangle — first REAL rendered output through the translation path. Builds a Vulkan
// graphics pipeline whose primitive topology and color-attachment format are derived from a prosper
// GpuState/RenderState via vk_translate (proving the CommandProcessor->Vulkan bridge works), draws a
// red triangle over a blue clear with embedded placeholder SPIR-V, reads the framebuffer back, and
// asserts the center pixel is the triangle color and a corner is the clear color. Agentic-first: no
// window, pixels checked programmatically. The RDNA2->SPIR-V recompiler will later replace the
// placeholder shaders; the fixed-function state already comes from the real register translation.
#include <vulkan/vulkan.h>
#include "../src/gpu/render_state.hpp"
#include "../src/gpu/vk_translate.hpp"
#include "spirv_triangle.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

static uint32_t crc32(const uint8_t* p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) { c ^= p[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (~(c & 1) + 1)); }
    return ~c;
}
// Write the framebuffer to a PPM if PROSPER_DUMP_PPM is set — for optional human/debug inspection.
// The test never *requires* this: verification is the programmatic checks below. (No <cstdlib>
// getenv include needed beyond what's here; guarded so it's a no-op in normal/CI runs.)
static void maybe_dump_ppm(const uint8_t* px, uint32_t W, uint32_t H);

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)
#define VKCHECK(e, m) CHECK((e) == VK_SUCCESS, m)

int main() {
    printf("== test_vulkan_triangle ==\n");
    const uint32_t W = 64, H = 64;

    // A render-state as the CommandProcessor would produce it: triangle list, RGBA8_UNORM color
    // target, no blend, no depth. The pipeline below is driven by translating THIS.
    RenderState rs;
    rs.prim_type = 4;                 // VGT triangle list
    rs.color0_format = 0x0A; rs.color0_number_type = 0; rs.color0_comp_swap = 0;   // 8_8_8_8 UNORM RGBA
    // vk_translate returns prosper::gpu enums whose values match the Vulkan enums; cast through
    // uint32_t to the (global) Vulkan types. Qualify ::VkFormat to avoid the prosper::gpu::VkFormat clash.
    const VkPrimitiveTopology topo = (VkPrimitiveTopology)(uint32_t)vk_topology(rs.prim_type);
    const ::VkFormat cfmt = (::VkFormat)(uint32_t)vk_color_format(rs.color0_format, rs.color0_number_type, rs.color0_comp_swap);
    CHECK(topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, "render-state topology -> VK triangle list");
    CHECK(cfmt == VK_FORMAT_R8G8B8A8_UNORM, "render-state color format -> VK R8G8B8A8_UNORM");

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "prosper-triangle"; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    VKCHECK(vkCreateInstance(&ici, nullptr, &inst), "vkCreateInstance");
    if (!inst) { printf("== FAIL: no instance ==\n"); return 1; }

    uint32_t ndev = 0; vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    if (!ndev) { printf("== FAIL: no device ==\n"); return 1; }
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs.data());
    VkPhysicalDevice phys = devs[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
    printf("  device: %s\n", props.deviceName);

    uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qf.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nqf; i++) if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qfi = i; break; }
    if (qfi == UINT32_MAX) { printf("== FAIL: no graphics queue ==\n"); return 1; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev = VK_NULL_HANDLE;
    VKCHECK(vkCreateDevice(phys, &dci, nullptr, &dev), "vkCreateDevice");
    if (!dev) { printf("== FAIL ==\n"); return 1; }
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);

    VkPhysicalDeviceMemoryProperties memp; vkGetPhysicalDeviceMemoryProperties(phys, &memp);
    auto pickMem = [&](uint32_t bits, VkMemoryPropertyFlags want) -> uint32_t {
        for (uint32_t i = 0; i < memp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memp.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX;
    };

    // Color attachment image (format from the render-state translation).
    VkImageCreateInfo imgci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgci.imageType = VK_IMAGE_TYPE_2D; imgci.format = cfmt; imgci.extent = {W, H, 1};
    imgci.mipLevels = 1; imgci.arrayLayers = 1; imgci.samples = VK_SAMPLE_COUNT_1_BIT;
    imgci.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imgci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage img; VKCHECK(vkCreateImage(dev, &imgci, nullptr, &img), "vkCreateImage");
    VkMemoryRequirements ireq; vkGetImageMemoryRequirements(dev, img, &ireq);
    VkMemoryAllocateInfo iai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    iai.allocationSize = ireq.size; iai.memoryTypeIndex = pickMem(ireq.memoryTypeBits, 0);
    VkDeviceMemory imem; VKCHECK(vkAllocateMemory(dev, &iai, nullptr, &imem), "alloc image mem");
    vkBindImageMemory(dev, img, imem, 0);

    VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ivci.image = img; ivci.viewType = VK_IMAGE_VIEW_TYPE_2D; ivci.format = cfmt;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view; VKCHECK(vkCreateImageView(dev, &ivci, nullptr, &view), "vkCreateImageView");

    // Render pass: one color attachment, clear -> store, ends in TRANSFER_SRC for readback.
    VkAttachmentDescription att{}; att.format = cfmt; att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference ar{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{}; sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1; sub.pColorAttachments = &ar;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1; rpci.pAttachments = &att; rpci.subpassCount = 1; rpci.pSubpasses = &sub;
    VkRenderPass rp; VKCHECK(vkCreateRenderPass(dev, &rpci, nullptr, &rp), "vkCreateRenderPass");

    VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbci.renderPass = rp; fbci.attachmentCount = 1; fbci.pAttachments = &view;
    fbci.width = W; fbci.height = H; fbci.layers = 1;
    VkFramebuffer fb; VKCHECK(vkCreateFramebuffer(dev, &fbci, nullptr, &fb), "vkCreateFramebuffer");

    // Shader modules (embedded placeholder SPIR-V).
    auto makeModule = [&](const uint32_t* code, size_t bytes) {
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = bytes; smci.pCode = code;
        VkShaderModule m; vkCreateShaderModule(dev, &smci, nullptr, &m); return m;
    };
    VkShaderModule vs = makeModule(kTriVertSpv, sizeof kTriVertSpv);
    VkShaderModule fs = makeModule(kTriFragSpv, sizeof kTriFragSpv);
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vs; stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = topo;                                    // <-- from vk_topology(render-state)
    VkViewport vp{0, 0, (float)W, (float)H, 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vpst{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vpst.viewportCount = 1; vpst.pViewports = &vp; vpst.scissorCount = 1; vpst.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs2{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs2.polygonMode = VK_POLYGON_MODE_FILL; rs2.cullMode = VK_CULL_MODE_NONE;
    rs2.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs2.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = rs.blend_enable ? VK_TRUE : VK_FALSE;   // <-- from render-state
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout; vkCreatePipelineLayout(dev, &plci, nullptr, &layout);
    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.stageCount = 2; gpci.pStages = stages; gpci.pVertexInputState = &vin;
    gpci.pInputAssemblyState = &ia; gpci.pViewportState = &vpst; gpci.pRasterizationState = &rs2;
    gpci.pMultisampleState = &ms; gpci.pColorBlendState = &cb; gpci.layout = layout;
    gpci.renderPass = rp; gpci.subpass = 0;
    VkPipeline pipe; VKCHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipe),
                             "vkCreateGraphicsPipelines (topology/format from render-state)");

    // Readback buffer.
    VkDeviceSize bytes = (VkDeviceSize)W * H * 4;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer rb; vkCreateBuffer(dev, &bci, nullptr, &rb);
    VkMemoryRequirements breq; vkGetBufferMemoryRequirements(dev, rb, &breq);
    VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    bai.allocationSize = breq.size;
    bai.memoryTypeIndex = pickMem(breq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory bmem; vkAllocateMemory(dev, &bai, nullptr, &bmem); vkBindBufferMemory(dev, rb, bmem, 0);

    // Record: render pass (clear blue) -> draw triangle -> copy image to buffer.
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = qfi;
    VkCommandPool pool; vkCreateCommandPool(dev, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cbai, &cmd);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);
    VkClearValue clear{}; clear.color = {{0.0f, 0.0f, 1.0f, 1.0f}};   // blue
    VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpbi.renderPass = rp; rpbi.framebuffer = fb; rpbi.renderArea = {{0, 0}, {W, H}};
    rpbi.clearValueCount = 1; rpbi.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    VkBufferImageCopy copy{}; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &copy);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence; vkCreateFence(dev, &fci, nullptr, &fence);
    VKCHECK(vkQueueSubmit(queue, 1, &si, fence), "vkQueueSubmit (draw)");
    VKCHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000), "vkWaitForFences");

    void* mapped = nullptr; vkMapMemory(dev, bmem, 0, bytes, 0, &mapped);
    const uint8_t* px = (const uint8_t*)mapped;
    auto at = [&](uint32_t x, uint32_t y) { return px + ((size_t)y * W + x) * 4; };
    const uint8_t* center = at(W / 2, H / 2);
    const uint8_t* corner = at(0, 0);
    printf("  center=(%u,%u,%u,%u) corner=(%u,%u,%u,%u)\n",
           center[0], center[1], center[2], center[3], corner[0], corner[1], corner[2], corner[3]);
    CHECK(center[0] > 0x80 && center[1] < 0x40 && center[2] < 0x40, "center pixel is the RED triangle");
    CHECK(corner[2] > 0x80 && corner[0] < 0x40 && corner[1] < 0x40, "corner pixel is the BLUE clear");

    // Golden-hash regression gate: llvmpipe (software raster) is deterministic, so the whole
    // framebuffer hashes to a stable value. Any change to the pipeline/translation that alters a
    // single pixel flips this hash and fails the test — automated regression detection, no eyeballing.
    // (On a hardware GPU the raster may differ; this gate assumes the CI's software rasterizer.)
    const uint32_t crc = crc32(px, (size_t)bytes);
    printf("  framebuffer CRC32 = 0x%08x\n", crc);
    constexpr uint32_t kGolden = 0x87160F1Eu;   // captured from llvmpipe; see note above
    CHECK(crc == kGolden, "framebuffer matches golden hash (whole-image regression gate)");
    maybe_dump_ppm(px, W, H);
    vkUnmapMemory(dev, bmem);

    vkDestroyFence(dev, fence, nullptr); vkDestroyCommandPool(dev, pool, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr); vkDestroyPipelineLayout(dev, layout, nullptr);
    vkDestroyShaderModule(dev, vs, nullptr); vkDestroyShaderModule(dev, fs, nullptr);
    vkDestroyFramebuffer(dev, fb, nullptr); vkDestroyRenderPass(dev, rp, nullptr);
    vkDestroyImageView(dev, view, nullptr);
    vkDestroyBuffer(dev, rb, nullptr); vkFreeMemory(dev, bmem, nullptr);
    vkDestroyImage(dev, img, nullptr); vkFreeMemory(dev, imem, nullptr);
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

static void maybe_dump_ppm(const uint8_t* px, uint32_t W, uint32_t H) {
    if (!getenv("PROSPER_DUMP_PPM")) return;
    if (FILE* f = fopen("triangle.ppm", "wb")) {
        fprintf(f, "P6\n%u %u\n255\n", W, H);
        for (uint32_t i = 0; i < W * H; i++) fwrite(px + i * 4, 1, 3, f);   // RGB, drop alpha
        fclose(f);
        printf("  wrote triangle.ppm\n");
    }
}
