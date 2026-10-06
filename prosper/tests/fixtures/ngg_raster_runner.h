// ngg_raster_runner.h -- rasterize a merged-NGG export record buffer offline through the P3 raster
// commit (gpu/recompiler/ngg_raster_commit.hpp) into a layered RGBA32F target, and read back every
// layer plus the violation counters (#3135 P3).
//
// The caller supplies the vertex stage, an optional geometry stage, and a fragment stage; this
// runner binds the export buffer at set 2 binding 1 and, when the vertex stage declares it, the
// counter buffer at set 2 binding 2 (zeroed), draws `vertex_count` non-indexed vertices with no
// vertex input, and returns layer-major RGBA32F pixels. The viewport has a NEGATIVE height, the
// guest's convention: clip y = +1 is the top row. Header-only; the including test links
// Vulkan::Vulkan.
#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <vector>

namespace prosper::test {

struct NggRasterRun {
    std::vector<uint32_t> vertex, geometry, fragment;
    std::vector<uint32_t> export_words;   // set 2 binding 1
    bool counters = true;   // set 2 binding 2, three words
    uint32_t vertex_count = 0;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCullModeFlags cull = VK_CULL_MODE_NONE;
    uint32_t width = 16, height = 16, layers = 32;
    bool shader_output_layer = false;   // enable VK_EXT_shader_viewport_index_layer
    float clear = -1.0f;
};

struct NggRasterResult {
    std::vector<float> pixels;   // [layer][y][x][rgba]
    std::array<uint32_t, 3> counters{};
    uint32_t width = 0, height = 0;
    const float* at(uint32_t layer, uint32_t x, uint32_t y) const {
        return &pixels[((static_cast<size_t>(layer) * height + y) * width + x) * 4u];
    }
};

struct NggRasterDevice {
    bool present = false;
    bool geometry = false;
    bool vertex_stores = false;
    bool shader_output_layer = false;
};

inline bool ngg_raster_has_extension(VkPhysicalDevice phys, const char* name) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, extensions.data());
    for (const auto& extension : extensions)
        if (!std::strcmp(extension.extensionName, name)) return true;
    return false;
}

// What the first physical device can do for these tests.
inline NggRasterDevice ngg_raster_device() {
    NggRasterDevice device;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return device;
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count) {
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        VkPhysicalDeviceFeatures features{};
        vkGetPhysicalDeviceFeatures(devices[0], &features);
        device.present = true;
        device.geometry = features.geometryShader;
        device.vertex_stores = features.vertexPipelineStoresAndAtomics;
        // As production gates it (render_runner.h): the extension AND Vulkan 1.2 shaderOutputLayer.
        VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12};
        vkGetPhysicalDeviceFeatures2(devices[0], &features2);
        device.shader_output_layer =
            v12.shaderOutputLayer &&
            ngg_raster_has_extension(devices[0], VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME);
    }
    vkDestroyInstance(instance, nullptr);
    return device;
}

// A fragment stage that writes the vec4 input at `location` to color 0, Flat when `flat`.
inline std::vector<uint32_t> ngg_param_fragment(uint32_t location, bool flat) {
    // ids: 1 void, 2 fn, 3 f32, 4 v4f, 5 ptr in v4, 6 ptr out v4, 7 input, 8 output, 9 main,
    // 10 label, 11 value.
    std::vector<uint32_t> m = {0x07230203u, 0x00010300u, 0u, 12u, 0u};
    const auto op = [&](uint32_t code, std::initializer_list<uint32_t> operands) {
        m.push_back(static_cast<uint32_t>(operands.size() + 1u) << 16 | code);
        m.insert(m.end(), operands);
    };
    op(17, {1});   // OpCapability Shader
    op(14, {0, 1});   // OpMemoryModel Logical GLSL450
    op(15, {4, 9, 0x6e69616du, 0u, 7, 8});   // OpEntryPoint Fragment %main "main" %in %out
    op(16, {9, 7});   // OpExecutionMode OriginUpperLeft
    op(71, {7, 30, location});   // OpDecorate %in Location
    if (flat) op(71, {7, 14});   // OpDecorate %in Flat
    op(71, {8, 30, 0});   // OpDecorate %out Location 0
    op(19, {1});   // OpTypeVoid
    op(33, {2, 1});   // OpTypeFunction
    op(22, {3, 32});   // OpTypeFloat 32
    op(23, {4, 3, 4});   // OpTypeVector
    op(32, {5, 1, 4});   // OpTypePointer Input
    op(32, {6, 3, 4});   // OpTypePointer Output
    op(59, {5, 7, 1});   // OpVariable Input
    op(59, {6, 8, 3});   // OpVariable Output
    op(54, {1, 9, 0, 2});   // OpFunction
    op(248, {10});   // OpLabel
    op(61, {4, 11, 7});   // OpLoad
    op(62, {8, 11});   // OpStore
    op(253, {});   // OpReturn
    op(56, {});   // OpFunctionEnd
    return m;
}

inline std::optional<NggRasterResult> run_ngg_raster(const NggRasterRun& run) {
    if (run.vertex.empty() || run.fragment.empty() || run.export_words.empty() ||
        !run.vertex_count || !run.layers)
        return std::nullopt;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return std::nullopt;
    VkDevice dev = VK_NULL_HANDLE;
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };
    std::vector<Buffer> buffers;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory image_memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkRenderPass pass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkDescriptorSetLayout layouts[3] = {};
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    std::vector<VkShaderModule> modules;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    const auto finish = [&](const char* failure) -> std::optional<NggRasterResult> {
        if (failure) std::fprintf(stderr, "ngg_raster_runner: %s\n", failure);
        if (dev) {
            vkDeviceWaitIdle(dev);
            if (command_pool) vkDestroyCommandPool(dev, command_pool, nullptr);
            if (pipeline) vkDestroyPipeline(dev, pipeline, nullptr);
            for (VkShaderModule module : modules) vkDestroyShaderModule(dev, module, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
            if (pool) vkDestroyDescriptorPool(dev, pool, nullptr);
            for (VkDescriptorSetLayout layout : layouts)
                if (layout) vkDestroyDescriptorSetLayout(dev, layout, nullptr);
            if (framebuffer) vkDestroyFramebuffer(dev, framebuffer, nullptr);
            if (pass) vkDestroyRenderPass(dev, pass, nullptr);
            if (view) vkDestroyImageView(dev, view, nullptr);
            if (image) vkDestroyImage(dev, image, nullptr);
            if (image_memory) vkFreeMemory(dev, image_memory, nullptr);
            for (const Buffer& b : buffers) {
                vkDestroyBuffer(dev, b.buffer, nullptr);
                vkFreeMemory(dev, b.memory, nullptr);
            }
            vkDestroyDevice(dev, nullptr);
        }
        vkDestroyInstance(inst, nullptr);
        return std::nullopt;
    };

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(inst, &device_count, nullptr);
    if (!device_count) return finish("no physical device");
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(inst, &device_count, devices.data());
    const VkPhysicalDevice phys = devices[0];
    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(phys, &supported);
    if (!run.geometry.empty() && !supported.geometryShader) return finish("no geometryShader");
    if (run.counters && !supported.vertexPipelineStoresAndAtomics)
        return finish("no vertexPipelineStoresAndAtomics");
    if (run.shader_output_layer &&
        !ngg_raster_has_extension(phys, VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME))
        return finish("no VK_EXT_shader_viewport_index_layer");
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, families.data());
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; ++i)
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            family = i;
            break;
        }
    if (family == UINT32_MAX) return finish("no graphics queue");
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures features{};
    features.robustBufferAccess = VK_TRUE;
    features.geometryShader = run.geometry.empty() ? VK_FALSE : VK_TRUE;
    features.vertexPipelineStoresAndAtomics = run.counters ? VK_TRUE : VK_FALSE;
    const char* extension = VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &features;
    dci.enabledExtensionCount = run.shader_output_layer ? 1u : 0u;
    dci.ppEnabledExtensionNames = &extension;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS) return finish("vkCreateDevice");
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev, family, 0, &queue);

    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(phys, &memory_properties);
    const auto memory_type = [&](uint32_t bits, VkMemoryPropertyFlags flags) {
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
            if ((bits & (1u << i)) &&
                (memory_properties.memoryTypes[i].propertyFlags & flags) == flags)
                return i;
        return UINT32_MAX;
    };
    constexpr VkMemoryPropertyFlags kHost =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const auto make_buffer = [&](VkDeviceSize bytes, VkBufferUsageFlags usage,
                                 const void* data) -> Buffer {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes;
        bci.usage = usage;
        Buffer out;
        if (vkCreateBuffer(dev, &bci, nullptr, &out.buffer) != VK_SUCCESS) return {};
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(dev, out.buffer, &requirements);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = requirements.size;
        ai.memoryTypeIndex = memory_type(requirements.memoryTypeBits, kHost);
        if (ai.memoryTypeIndex == UINT32_MAX ||
            vkAllocateMemory(dev, &ai, nullptr, &out.memory) != VK_SUCCESS) {
            vkDestroyBuffer(dev, out.buffer, nullptr);
            return {};
        }
        buffers.push_back(out);
        void* mapped = nullptr;
        if (vkBindBufferMemory(dev, out.buffer, out.memory, 0) != VK_SUCCESS ||
            vkMapMemory(dev, out.memory, 0, bytes, 0, &mapped) != VK_SUCCESS)
            return {};
        if (data)
            std::memcpy(mapped, data, bytes);
        else
            std::memset(mapped, 0, bytes);
        vkUnmapMemory(dev, out.memory);
        return out;
    };
    const Buffer exports = make_buffer(run.export_words.size() * 4u,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, run.export_words.data());
    const Buffer counters =
        run.counters ? make_buffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr) : Buffer{};
    const VkDeviceSize pixel_bytes =
        static_cast<VkDeviceSize>(run.width) * run.height * run.layers * 16u;
    const Buffer readback = make_buffer(pixel_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, nullptr);
    if (!exports.buffer || (run.counters && !counters.buffer) || !readback.buffer)
        return finish("buffer allocation");

    constexpr VkFormat kFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization): every field is set below.
    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = kFormat;
    image_info.extent = {run.width, run.height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = run.layers;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateImage(dev, &image_info, nullptr, &image) != VK_SUCCESS) return finish("image");
    VkMemoryRequirements image_requirements;
    vkGetImageMemoryRequirements(dev, image, &image_requirements);
    VkMemoryAllocateInfo image_alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    image_alloc.allocationSize = image_requirements.size;
    image_alloc.memoryTypeIndex = memory_type(image_requirements.memoryTypeBits, 0);
    if (vkAllocateMemory(dev, &image_alloc, nullptr, &image_memory) != VK_SUCCESS ||
        vkBindImageMemory(dev, image, image_memory, 0) != VK_SUCCESS)
        return finish("image memory");
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    view_info.format = kFormat;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, run.layers};
    if (vkCreateImageView(dev, &view_info, nullptr, &view) != VK_SUCCESS) return finish("view");

    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization): every field is set below.
    VkAttachmentDescription attachment{};
    attachment.format = kFormat;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;
    VkSubpassDependency dependency{};
    dependency.srcSubpass = 0;
    dependency.dstSubpass = VK_SUBPASS_EXTERNAL;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependency.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1;
    rpci.pAttachments = &attachment;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies = &dependency;
    if (vkCreateRenderPass(dev, &rpci, nullptr, &pass) != VK_SUCCESS) return finish("render pass");
    VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fci.renderPass = pass;
    fci.attachmentCount = 1;
    fci.pAttachments = &view;
    fci.width = run.width;
    fci.height = run.height;
    fci.layers = run.layers;
    if (vkCreateFramebuffer(dev, &fci, nullptr, &framebuffer) != VK_SUCCESS)
        return finish("framebuffer");

    const VkShaderStageFlags stages =
        VK_SHADER_STAGE_VERTEX_BIT | (run.geometry.empty() ? 0u : VK_SHADER_STAGE_GEOMETRY_BIT);
    std::vector<VkDescriptorSetLayoutBinding> set2;
    set2.push_back({1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stages, nullptr});
    if (run.counters) set2.push_back({2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stages, nullptr});
    for (uint32_t set = 0; set < 3; ++set) {
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        if (set == 2) {
            lci.bindingCount = static_cast<uint32_t>(set2.size());
            lci.pBindings = set2.data();
        }
        if (vkCreateDescriptorSetLayout(dev, &lci, nullptr, &layouts[set]) != VK_SUCCESS)
            return finish("descriptor set layout");
    }
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 3;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(dev, &pci, nullptr, &pool) != VK_SUCCESS) return finish("pool");
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool;
    dai.descriptorSetCount = 3;
    dai.pSetLayouts = layouts;
    VkDescriptorSet sets[3] = {};
    if (vkAllocateDescriptorSets(dev, &dai, sets) != VK_SUCCESS) return finish("descriptor sets");
    const VkDescriptorBufferInfo infos[2] = {{exports.buffer, 0, VK_WHOLE_SIZE},
                                             {counters.buffer, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[2] = {};
    for (uint32_t k = 0; k < (run.counters ? 2u : 1u); ++k) {
        writes[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[k].dstSet = sets[2];
        writes[k].dstBinding = 1 + k;
        writes[k].descriptorCount = 1;
        writes[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[k].pBufferInfo = &infos[k];
    }
    vkUpdateDescriptorSets(dev, run.counters ? 2u : 1u, writes, 0, nullptr);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 3;
    plci.pSetLayouts = layouts;
    if (vkCreatePipelineLayout(dev, &plci, nullptr, &pipeline_layout) != VK_SUCCESS)
        return finish("pipeline layout");

    std::vector<VkPipelineShaderStageCreateInfo> stage_infos;
    const auto add_stage = [&](const std::vector<uint32_t>& words, VkShaderStageFlagBits stage) {
        if (words.empty()) return true;
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = words.size() * 4u;
        smci.pCode = words.data();
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(dev, &smci, nullptr, &module) != VK_SUCCESS) return false;
        modules.push_back(module);
        // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization): every field is set below.
        VkPipelineShaderStageCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage = stage;
        info.module = module;
        info.pName = "main";
        stage_infos.push_back(info);
        return true;
    };
    if (!add_stage(run.vertex, VK_SHADER_STAGE_VERTEX_BIT) ||
        !add_stage(run.geometry, VK_SHADER_STAGE_GEOMETRY_BIT) ||
        !add_stage(run.fragment, VK_SHADER_STAGE_FRAGMENT_BIT))
        return finish("shader module");
    VkPipelineVertexInputStateCreateInfo vertex_input{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = run.topology;
    // The guest's convention: a negative height puts clip y = +1 on the top row.
    const VkViewport viewport{0.0f,
                              static_cast<float>(run.height),
                              static_cast<float>(run.width),
                              -static_cast<float>(run.height),
                              0.0f,
                              1.0f};
    const VkRect2D scissor{{0, 0}, {run.width, run.height}};
    VkPipelineViewportStateCreateInfo viewport_state{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = run.cull;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization): every field is set below.
    VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.stageCount = static_cast<uint32_t>(stage_infos.size());
    gpci.pStages = stage_infos.data();
    gpci.pVertexInputState = &vertex_input;
    gpci.pInputAssemblyState = &assembly;
    gpci.pViewportState = &viewport_state;
    gpci.pRasterizationState = &raster;
    gpci.pMultisampleState = &multisample;
    gpci.pColorBlendState = &blend;
    gpci.layout = pipeline_layout;
    gpci.renderPass = pass;
    if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline) != VK_SUCCESS)
        return finish("graphics pipeline");

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = family;
    if (vkCreateCommandPool(dev, &cpi, nullptr, &command_pool) != VK_SUCCESS)
        return finish("command pool");
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = command_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(dev, &cbai, &cmd) != VK_SUCCESS) return finish("command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &begin);
    VkClearValue clear{};
    for (float& channel : clear.color.float32) channel = run.clear;
    VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpbi.renderPass = pass;
    rpbi.framebuffer = framebuffer;
    rpbi.renderArea = scissor;
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 3, sets, 0,
                            nullptr);
    vkCmdDraw(cmd, run.vertex_count, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, run.layers};
    copy.imageExtent = {run.width, run.height, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1,
                           &copy);
    VkMemoryBarrier host_read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    host_read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_read, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS ||
        vkQueueWaitIdle(queue) != VK_SUCCESS)
        return finish("submit");

    NggRasterResult result;
    result.width = run.width;
    result.height = run.height;
    result.pixels.resize(pixel_bytes / 4u);
    void* mapped = nullptr;
    if (vkMapMemory(dev, readback.memory, 0, pixel_bytes, 0, &mapped) != VK_SUCCESS)
        return finish("readback map");
    std::memcpy(result.pixels.data(), mapped, pixel_bytes);
    vkUnmapMemory(dev, readback.memory);
    if (run.counters) {
        if (vkMapMemory(dev, counters.memory, 0, 12, 0, &mapped) != VK_SUCCESS)
            return finish("counter map");
        std::memcpy(result.counters.data(), mapped, 12);
        vkUnmapMemory(dev, counters.memory);
    }
    finish(nullptr);
    return result;
}

}   // namespace prosper::test
