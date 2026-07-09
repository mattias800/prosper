// image_compute_runner.h — inline helper to run a compute SPIR-V module that copies one 1D storage
// image to another (OpImageRead binding 4 -> OpImageWrite binding 5), and return the destination
// texels. This is the execution harness for the storage-image recompiler path: it mirrors the
// compute shell's descriptor layout (4 storage buffers at bindings 0..3 + 2 storage images at 4/5)
// so a recompiled kernel binds cleanly. Header-only; the including test links Vulkan::Vulkan.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace prosper::test {

// Runs `spirv` (a compute module) which reads storage image at (set=0, binding=4) and writes
// storage image at (set=0, binding=5), both 1D VK_FORMAT_R32G32B32A32_UINT of `width` texels.
// `src_rgba` has width*4 uint32 (R,G,B,A per texel, tightly packed). Dispatches ceil(width/64)
// workgroups (local_size_x=64). Returns the dst image texels (width*4 uint32), or {} on any failure.
inline std::vector<uint32_t> run_image_copy(const std::vector<uint32_t>& spirv,
                                            uint32_t width,
                                            const std::vector<uint32_t>& src_rgba) {
    std::vector<uint32_t> out;
    if (width == 0) return out;
    const VkDeviceSize texels = (VkDeviceSize)width * 4;              // uint32 per RGBA channel
    const VkDeviceSize imgBytes = texels * sizeof(uint32_t);
    if (src_rgba.size() < texels) return out;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS || !inst) return out;

    uint32_t ndev = 0; vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    if (!ndev) { vkDestroyInstance(inst, nullptr); return out; }
    std::vector<VkPhysicalDevice> devs(ndev); vkEnumeratePhysicalDevices(inst, &ndev, devs.data());
    VkPhysicalDevice phys = devs[0];

    uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qf.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nqf; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    if (qfi == UINT32_MAX) { vkDestroyInstance(inst, nullptr); return out; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    // robustBufferAccess: out-of-range storage-buffer reads/writes are well-defined (return 0 / no-op)
    // rather than UB. shaderStorageImageRead/WriteWithoutFormat: the module declares the matching
    // StorageImageReadWithoutFormat / WriteWithoutFormat capabilities (the image views carry no format
    // hint in the shader), so these features must be enabled for the SPIR-V to be accepted.
    VkPhysicalDeviceFeatures feats{};
    feats.robustBufferAccess = VK_TRUE;
    feats.shaderStorageImageReadWithoutFormat = VK_TRUE;
    feats.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    dci.pEnabledFeatures = &feats;
    // robustImageAccess (VK_EXT_image_robustness; core in 1.3): the recompiled storage-image load
    // path issues OpImageRead for ALL invocations — including EXEC-inactive/grid-tail lanes whose
    // coordinates can be out of range — relying on OOB image reads returning zero (#131). Enable it
    // whenever the device offers it (feature-query guarded, so a device without it still creates —
    // then only exact-multiple dispatches are safe, which the fixed-width tests are).
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
    const VkMemoryPropertyFlags HOST = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    // --- Fail helper: destroy device+instance and return {} on any create failure. Every object below
    // is null-initialized so a partial teardown at any point is safe. ---
    auto fail = [&]() -> std::vector<uint32_t> {
        vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr); return {};
    };

    // --- Four host-visible storage buffers (bindings 0..3). The recompiler's compute shell always
    // declares 4 storage buffers, so the descriptor layout must be complete. Binding 0 is the shell's
    // per-invocation INPUT array: we fill it with the linear index [0,1,..,width-1] so a copy kernel can
    // read `v0 = input[gid] = gid` as its texel coordinate (the compute thread-ID ABI — s16=wg.id /
    // v0=local.id — is modeled separately; this harness supplies the per-lane index via buffer 0).
    // Bindings 1..3 are unused here (4 bytes each). ---
    VkBuffer sb[4] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory sbmem[4] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    for (int i = 0; i < 4; i++) {
        VkDeviceSize bsz = (i == 0) ? (VkDeviceSize)width * 4 : 4;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bsz; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(dev, &bci, nullptr, &sb[i]) != VK_SUCCESS) return fail();
        VkMemoryRequirements r; vkGetBufferMemoryRequirements(dev, sb[i], &r);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = r.size; ai.memoryTypeIndex = pick(r.memoryTypeBits, HOST);
        if (ai.memoryTypeIndex == UINT32_MAX) return fail();
        if (vkAllocateMemory(dev, &ai, nullptr, &sbmem[i]) != VK_SUCCESS) return fail();
        vkBindBufferMemory(dev, sb[i], sbmem[i], 0);
        if (i == 0) {   // per-lane linear index [0..width-1] (read as raw bits by the shell's load_input)
            void* ip = nullptr; vkMapMemory(dev, sbmem[0], 0, bsz, 0, &ip);
            for (uint32_t t = 0; t < width; t++) ((uint32_t*)ip)[t] = t;
            vkUnmapMemory(dev, sbmem[0]);
        }
    }

    // --- Two 1D storage images (binding 4 = src/read, binding 5 = dst/write), device-local. ---
    const VkFormat FMT = VK_FORMAT_R32G32B32A32_UINT;
    VkImage srcImg = VK_NULL_HANDLE, dstImg = VK_NULL_HANDLE;
    VkDeviceMemory srcMem = VK_NULL_HANDLE, dstMem = VK_NULL_HANDLE;
    VkImageView srcView = VK_NULL_HANDLE, dstView = VK_NULL_HANDLE;
    auto makeImg = [&](VkImage& img, VkDeviceMemory& mem, VkImageView& view) -> bool {
        VkImageCreateInfo ici2{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici2.imageType = VK_IMAGE_TYPE_1D; ici2.format = FMT; ici2.extent = {width, 1, 1};
        ici2.mipLevels = 1; ici2.arrayLayers = 1; ici2.samples = VK_SAMPLE_COUNT_1_BIT;
        ici2.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici2.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici2.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(dev, &ici2, nullptr, &img) != VK_SUCCESS) return false;
        VkMemoryRequirements r; vkGetImageMemoryRequirements(dev, img, &r);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = r.size; ai.memoryTypeIndex = pick(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX) return false;
        if (vkAllocateMemory(dev, &ai, nullptr, &mem) != VK_SUCCESS) return false;
        vkBindImageMemory(dev, img, mem, 0);
        VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivci.image = img; ivci.viewType = VK_IMAGE_VIEW_TYPE_1D; ivci.format = FMT;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        if (vkCreateImageView(dev, &ivci, nullptr, &view) != VK_SUCCESS) return false;
        return true;
    };
    if (!makeImg(srcImg, srcMem, srcView)) return fail();
    if (!makeImg(dstImg, dstMem, dstView)) return fail();

    // --- Host-visible staging buffer holding src_rgba, uploaded into srcImg below. ---
    VkBuffer stage = VK_NULL_HANDLE; VkDeviceMemory stageMem = VK_NULL_HANDLE;
    {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = imgBytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(dev, &bci, nullptr, &stage) != VK_SUCCESS) return fail();
        VkMemoryRequirements r; vkGetBufferMemoryRequirements(dev, stage, &r);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = r.size; ai.memoryTypeIndex = pick(r.memoryTypeBits, HOST);
        if (ai.memoryTypeIndex == UINT32_MAX) return fail();
        if (vkAllocateMemory(dev, &ai, nullptr, &stageMem) != VK_SUCCESS) return fail();
        vkBindBufferMemory(dev, stage, stageMem, 0);
        void* p = nullptr; vkMapMemory(dev, stageMem, 0, imgBytes, 0, &p);
        for (VkDeviceSize i = 0; i < texels; i++) ((uint32_t*)p)[i] = src_rgba[(size_t)i];
        vkUnmapMemory(dev, stageMem);
    }

    // --- Host-visible readback buffer for the dst image texels. ---
    VkBuffer readback = VK_NULL_HANDLE; VkDeviceMemory readbackMem = VK_NULL_HANDLE;
    {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = imgBytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(dev, &bci, nullptr, &readback) != VK_SUCCESS) return fail();
        VkMemoryRequirements r; vkGetBufferMemoryRequirements(dev, readback, &r);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = r.size; ai.memoryTypeIndex = pick(r.memoryTypeBits, HOST);
        if (ai.memoryTypeIndex == UINT32_MAX) return fail();
        if (vkAllocateMemory(dev, &ai, nullptr, &readbackMem) != VK_SUCCESS) return fail();
        vkBindBufferMemory(dev, readback, readbackMem, 0);
    }

    // --- Descriptor set 0: bindings 0..3 storage buffers, 4/5 storage images. ---
    VkDescriptorSetLayoutBinding binds[6]{};
    for (int i = 0; i < 4; i++) {
        binds[i].binding = i; binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1; binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    binds[4].binding = 4; binds[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[4].descriptorCount = 1; binds[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[5].binding = 5; binds[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[5].descriptorCount = 1; binds[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 6; dslci.pBindings = binds;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl) != VK_SUCCESS) return fail();
    VkDescriptorPoolSize psz[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = psz;
    VkDescriptorPool dp = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(dev, &dpci, nullptr, &dp) != VK_SUCCESS) { vkDestroyDescriptorSetLayout(dev, dsl, nullptr); return fail(); }
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(dev, &dsai, &dset) != VK_SUCCESS) {
        vkDestroyDescriptorPool(dev, dp, nullptr); vkDestroyDescriptorSetLayout(dev, dsl, nullptr); return fail();
    }
    VkDescriptorBufferInfo dbi[4] = {{sb[0], 0, VK_WHOLE_SIZE}, {sb[1], 0, VK_WHOLE_SIZE},
                                     {sb[2], 0, VK_WHOLE_SIZE}, {sb[3], 0, VK_WHOLE_SIZE}};
    // Storage images are accessed in VK_IMAGE_LAYOUT_GENERAL.
    VkDescriptorImageInfo dii[2] = {{VK_NULL_HANDLE, srcView, VK_IMAGE_LAYOUT_GENERAL},
                                    {VK_NULL_HANDLE, dstView, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[6]{};
    for (int i = 0; i < 4; i++) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[i].dstSet = dset; w[i].dstBinding = i;
        w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &dbi[i];
    }
    w[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[4].dstSet = dset; w[4].dstBinding = 4;
    w[4].descriptorCount = 1; w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[4].pImageInfo = &dii[0];
    w[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[5].dstSet = dset; w[5].dstBinding = 5;
    w[5].descriptorCount = 1; w[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[5].pImageInfo = &dii[1];
    vkUpdateDescriptorSets(dev, 6, w, 0, nullptr);

    // --- Shader module + compute pipeline. ---
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirv.size() * 4; smci.pCode = spirv.data();
    VkShaderModule sm = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &smci, nullptr, &sm) != VK_SUCCESS || !sm) {   // rejected SPIR-V
        vkDestroyDescriptorPool(dev, dp, nullptr); vkDestroyDescriptorSetLayout(dev, dsl, nullptr); return fail();
    }
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(dev, &plci, nullptr, &layout) != VK_SUCCESS) {
        vkDestroyShaderModule(dev, sm, nullptr); vkDestroyDescriptorPool(dev, dp, nullptr);
        vkDestroyDescriptorSetLayout(dev, dsl, nullptr); return fail();
    }
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main";
    cpci.layout = layout;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe) != VK_SUCCESS) {
        vkDestroyPipelineLayout(dev, layout, nullptr); vkDestroyShaderModule(dev, sm, nullptr);
        vkDestroyDescriptorPool(dev, dp, nullptr); vkDestroyDescriptorSetLayout(dev, dsl, nullptr); return fail();
    }

    // --- Single command buffer: upload src, transition both images to GENERAL, dispatch, then
    // transition dst to TRANSFER_SRC and copy it to the readback buffer. ---
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = qfi;
    VkCommandPool pool = VK_NULL_HANDLE; vkCreateCommandPool(dev, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cbai, &cmd);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);

    auto barrier = [&](VkImage img, VkImageLayout oldL, VkImageLayout newL,
                       VkAccessFlags srcA, VkAccessFlags dstA,
                       VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = oldL; b.newLayout = newL; b.image = img;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.srcAccessMask = srcA; b.dstAccessMask = dstA;
        vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    // (a) src: UNDEFINED -> TRANSFER_DST, copy staging, TRANSFER_DST -> GENERAL.
    barrier(srcImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy up{}; up.bufferRowLength = 0; up.bufferImageHeight = 0;
    up.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; up.imageExtent = {width, 1, 1};
    vkCmdCopyBufferToImage(cmd, stage, srcImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &up);
    barrier(srcImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    // (b) dst: UNDEFINED -> GENERAL (written by the shader).
    barrier(dstImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
    vkCmdDispatch(cmd, (width + 63) / 64, 1, 1);

    // Readback: dst GENERAL -> TRANSFER_SRC, copy image to host-visible buffer.
    barrier(dstImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy down{}; down.bufferRowLength = 0; down.bufferImageHeight = 0;
    down.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; down.imageExtent = {width, 1, 1};
    vkCmdCopyImageToBuffer(cmd, dstImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &down);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; vkCreateFence(dev, &fci, nullptr, &fence);
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000);

    out.resize((size_t)texels);
    void* mp = nullptr; vkMapMemory(dev, readbackMem, 0, imgBytes, 0, &mp);
    for (VkDeviceSize i = 0; i < texels; i++) out[(size_t)i] = ((const uint32_t*)mp)[i];
    vkUnmapMemory(dev, readbackMem);

    vkDestroyFence(dev, fence, nullptr); vkDestroyCommandPool(dev, pool, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr); vkDestroyPipelineLayout(dev, layout, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyDescriptorPool(dev, dp, nullptr); vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyBuffer(dev, readback, nullptr); vkFreeMemory(dev, readbackMem, nullptr);
    vkDestroyBuffer(dev, stage, nullptr); vkFreeMemory(dev, stageMem, nullptr);
    vkDestroyImageView(dev, srcView, nullptr); vkDestroyImage(dev, srcImg, nullptr); vkFreeMemory(dev, srcMem, nullptr);
    vkDestroyImageView(dev, dstView, nullptr); vkDestroyImage(dev, dstImg, nullptr); vkFreeMemory(dev, dstMem, nullptr);
    for (int i = 0; i < 4; i++) { vkDestroyBuffer(dev, sb[i], nullptr); vkFreeMemory(dev, sbmem[i], nullptr); }
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);
    return out;
}

} // namespace prosper::test
