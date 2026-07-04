// compute_runner.h — inline helper to run a compute SPIR-V module over an input float buffer and
// return the output float buffer, via real Vulkan compute (2 storage buffers: binding 0 = in,
// binding 1 = out; local_size_x = 64). Shared by the execution-differential tests so the Vulkan
// boilerplate lives in one place. Header-only; the including test links Vulkan::Vulkan.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

namespace prosper::test {

// Run `spirv` over storage buffer 0 = `input`, storage buffer 1 = output. `invocations` compute
// threads are dispatched (default = input.size()); the output buffer holds `out_count` floats
// (default = input.size()). Returns the output, or {} on any Vulkan failure (incl. rejected SPIR-V).
inline std::vector<float> run_compute(const std::vector<uint32_t>& spirv, const std::vector<float>& input,
                                      uint32_t invocations = 0, uint32_t out_count = 0,
                                      const std::vector<uint32_t>& cbuf = {},
                                      const std::vector<uint32_t>& cbuf1 = {}) {
    const uint32_t IN_N = (uint32_t)input.size();
    if (invocations == 0) invocations = IN_N;
    if (out_count == 0)   out_count = IN_N;
    std::vector<float> out;

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
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS || !dev) { vkDestroyInstance(inst, nullptr); return out; }
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);

    VkPhysicalDeviceMemoryProperties memp; vkGetPhysicalDeviceMemoryProperties(phys, &memp);
    auto hostMem = [&](uint32_t bits) -> uint32_t {
        for (uint32_t i = 0; i < memp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memp.memoryTypes[i].propertyFlags &
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) return i;
        return UINT32_MAX;
    };
    const VkDeviceSize inBytes = (VkDeviceSize)IN_N * sizeof(float);
    const VkDeviceSize outBytes = (VkDeviceSize)out_count * sizeof(float);
    VkBuffer inBuf, outBuf, cbBuf, cbBuf1; VkDeviceMemory inMem, outMem, cbMem, cbMem1;
    auto makeBuf = [&](VkBuffer& b, VkDeviceMemory& m, VkDeviceSize sz) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = sz; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        vkCreateBuffer(dev, &bci, nullptr, &b);
        VkMemoryRequirements r; vkGetBufferMemoryRequirements(dev, b, &r);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = r.size; ai.memoryTypeIndex = hostMem(r.memoryTypeBits);
        vkAllocateMemory(dev, &ai, nullptr, &m); vkBindBufferMemory(dev, b, m, 0);
    };
    // Constant buffer (binding 2, always present since the shell declares it; >=1 dword). Holds the
    // scalar memory an SMEM load reads. Shaders without SMEM never read it.
    const uint32_t CB_N  = cbuf.empty()  ? 1u : (uint32_t)cbuf.size();
    const uint32_t CB1_N = cbuf1.empty() ? 1u : (uint32_t)cbuf1.size();
    const VkDeviceSize cbBytes  = (VkDeviceSize)CB_N  * sizeof(uint32_t);
    const VkDeviceSize cbBytes1 = (VkDeviceSize)CB1_N * sizeof(uint32_t);
    makeBuf(inBuf, inMem, inBytes); makeBuf(outBuf, outMem, outBytes);
    makeBuf(cbBuf, cbMem, cbBytes); makeBuf(cbBuf1, cbMem1, cbBytes1);
    void* p = nullptr; vkMapMemory(dev, inMem, 0, inBytes, 0, &p);
    for (uint32_t i = 0; i < IN_N; i++) ((float*)p)[i] = input[i];
    vkUnmapMemory(dev, inMem);
    void* cp = nullptr; vkMapMemory(dev, cbMem, 0, cbBytes, 0, &cp);
    for (uint32_t i = 0; i < CB_N; i++) ((uint32_t*)cp)[i] = (i < cbuf.size()) ? cbuf[i] : 0u;
    vkUnmapMemory(dev, cbMem);
    void* cp1 = nullptr; vkMapMemory(dev, cbMem1, 0, cbBytes1, 0, &cp1);
    for (uint32_t i = 0; i < CB1_N; i++) ((uint32_t*)cp1)[i] = (i < cbuf1.size()) ? cbuf1[i] : 0u;
    vkUnmapMemory(dev, cbMem1);
    // Zero the output buffer so results are deterministic — in particular, EXEC-masked lanes (which
    // do not store) must observe a defined prior value, not uninitialized memory.
    void* zp = nullptr; vkMapMemory(dev, outMem, 0, outBytes, 0, &zp);
    for (uint32_t i = 0; i < out_count; i++) ((float*)zp)[i] = 0.0f;
    vkUnmapMemory(dev, outMem);

    VkDescriptorSetLayoutBinding binds[4]{};
    for (int i = 0; i < 4; i++) { binds[i].binding = i; binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1; binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 4; dslci.pBindings = binds;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);
    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &psz;
    VkDescriptorPool dp; vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset; vkAllocateDescriptorSets(dev, &dsai, &dset);
    VkDescriptorBufferInfo bi0{inBuf, 0, VK_WHOLE_SIZE}, bi1{outBuf, 0, VK_WHOLE_SIZE},
                           bi2{cbBuf, 0, VK_WHOLE_SIZE}, bi3{cbBuf1, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[4]{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[0].dstSet = dset; w[0].dstBinding = 0;
    w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &bi0;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[1].dstSet = dset; w[1].dstBinding = 1;
    w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &bi1;
    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[2].dstSet = dset; w[2].dstBinding = 2;
    w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[2].pBufferInfo = &bi2;
    w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[3].dstSet = dset; w[3].dstBinding = 3;
    w[3].descriptorCount = 1; w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[3].pBufferInfo = &bi3;
    vkUpdateDescriptorSets(dev, 4, w, 0, nullptr);

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirv.size() * 4; smci.pCode = spirv.data();
    VkShaderModule sm = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &smci, nullptr, &sm) != VK_SUCCESS) return out;   // invalid SPIR-V
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout layout; vkCreatePipelineLayout(dev, &plci, nullptr, &layout);
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main";
    cpci.layout = layout;
    VkPipeline pipe;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe) != VK_SUCCESS) return out;

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = qfi;
    VkCommandPool pool; vkCreateCommandPool(dev, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cbai, &cmd);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
    vkCmdDispatch(cmd, (invocations + 63) / 64, 1, 1);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; vkCreateFence(dev, &fci, nullptr, &fence);
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000);

    out.resize(out_count);
    void* op = nullptr; vkMapMemory(dev, outMem, 0, outBytes, 0, &op);
    for (uint32_t i = 0; i < out_count; i++) out[i] = ((float*)op)[i];
    vkUnmapMemory(dev, outMem);

    vkDestroyFence(dev, fence, nullptr); vkDestroyCommandPool(dev, pool, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr); vkDestroyPipelineLayout(dev, layout, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyDescriptorPool(dev, dp, nullptr); vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyBuffer(dev, inBuf, nullptr); vkFreeMemory(dev, inMem, nullptr);
    vkDestroyBuffer(dev, outBuf, nullptr); vkFreeMemory(dev, outMem, nullptr);
    vkDestroyBuffer(dev, cbBuf, nullptr); vkFreeMemory(dev, cbMem, nullptr);
    vkDestroyBuffer(dev, cbBuf1, nullptr); vkFreeMemory(dev, cbMem1, nullptr);
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);
    return out;
}

} // namespace prosper::test
