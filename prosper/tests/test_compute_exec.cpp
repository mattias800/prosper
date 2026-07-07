// test_compute_exec — the execution-differential verification harness (verification layer 4, see
// docs/VERIFICATION.md). It runs a shader on real Vulkan compute with known input values and asserts
// the numeric outputs against the expected math — the pattern that proves shader CORRECTNESS (not
// just structural plausibility) with zero manual inspection. Today it runs a placeholder glslang
// compute shader (b[i] = a[i]*2 + 1); once the RDNA2->SPIR-V recompiler lands, its emitted SPIR-V
// swaps in here and the same assertions verify the recompile is numerically correct.
#include <vulkan/vulkan.h>
#include "spirv_compute.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)
#define VKCHECK(e, m) CHECK((e) == VK_SUCCESS, m)

int main() {
    printf("== test_compute_exec ==\n");
    const uint32_t N = 256;   // 4 groups of local_size_x=64

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "prosper-compute"; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    VKCHECK(vkCreateInstance(&ici, nullptr, &inst), "vkCreateInstance");
    if (!inst) { printf("== FAIL: no instance ==\n"); return 1; }

    uint32_t ndev = 0; vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    if (!ndev) { printf("== FAIL: no device ==\n"); return 1; }
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs.data());
    VkPhysicalDevice phys = devs[0];

    uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qf.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nqf; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    if (qfi == UINT32_MAX) { printf("== FAIL: no compute queue ==\n"); return 1; }
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
    auto hostMem = [&](uint32_t bits) -> uint32_t {
        for (uint32_t i = 0; i < memp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memp.memoryTypes[i].propertyFlags &
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) return i;
        return UINT32_MAX;
    };
    const VkDeviceSize bytes = (VkDeviceSize)N * sizeof(float);
    auto makeBuf = [&](VkBuffer& b, VkDeviceMemory& m) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        vkCreateBuffer(dev, &bci, nullptr, &b);
        VkMemoryRequirements r; vkGetBufferMemoryRequirements(dev, b, &r);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = r.size; ai.memoryTypeIndex = hostMem(r.memoryTypeBits);
        vkAllocateMemory(dev, &ai, nullptr, &m); vkBindBufferMemory(dev, b, m, 0);
    };
    VkBuffer inBuf, outBuf; VkDeviceMemory inMem, outMem;
    makeBuf(inBuf, inMem); makeBuf(outBuf, outMem);

    // Known inputs a[i] = i * 0.5 - 3; expected b[i] = a[i]*2 + 1.
    float* ain = nullptr; vkMapMemory(dev, inMem, 0, bytes, 0, (void**)&ain);
    std::vector<float> expect(N);
    for (uint32_t i = 0; i < N; i++) { float a = (float)i * 0.5f - 3.0f; ain[i] = a; expect[i] = a * 2.0f + 1.0f; }
    vkUnmapMemory(dev, inMem);

    // Descriptor set layout: 2 storage buffers.
    VkDescriptorSetLayoutBinding binds[2]{};
    for (int i = 0; i < 2; i++) { binds[i].binding = i; binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1; binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 2; dslci.pBindings = binds;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkDescriptorPool dp; vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds; vkAllocateDescriptorSets(dev, &dsai, &ds);
    VkDescriptorBufferInfo bi0{inBuf, 0, VK_WHOLE_SIZE}, bi1{outBuf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[2]{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[0].dstSet = ds; w[0].dstBinding = 0;
    w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &bi0;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[1].dstSet = ds; w[1].dstBinding = 1;
    w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &bi1;
    vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = sizeof kComputeSpv; smci.pCode = kComputeSpv;
    VkShaderModule sm; VKCHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm), "vkCreateShaderModule");
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout layout; vkCreatePipelineLayout(dev, &plci, nullptr, &layout);
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main";
    cpci.layout = layout;
    VkPipeline pipe; VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe),
                             "vkCreateComputePipelines");

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = qfi;
    VkCommandPool pool; vkCreateCommandPool(dev, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cbai, &cmd);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
    vkCmdDispatch(cmd, N / 64, 1, 1);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; vkCreateFence(dev, &fci, nullptr, &fence);
    VKCHECK(vkQueueSubmit(queue, 1, &si, fence), "vkQueueSubmit (dispatch)");
    VKCHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000), "vkWaitForFences");

    float* bout = nullptr; vkMapMemory(dev, outMem, 0, bytes, 0, (void**)&bout);
    uint32_t bad = 0; float worst = 0;
    for (uint32_t i = 0; i < N; i++) { float d = std::fabs(bout[i] - expect[i]); if (d > 1e-4f) { bad++; worst = d > worst ? d : worst; } }
    printf("  N=%u mismatches=%u worst=%g (sample out[10]=%g expect=%g)\n", N, bad, worst, bout[10], expect[10]);
    CHECK(bad == 0, "all compute outputs match expected b[i] = a[i]*2 + 1 (execution-differential)");
    vkUnmapMemory(dev, outMem);

    vkDestroyFence(dev, fence, nullptr); vkDestroyCommandPool(dev, pool, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr); vkDestroyPipelineLayout(dev, layout, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyDescriptorPool(dev, dp, nullptr); vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyBuffer(dev, inBuf, nullptr); vkFreeMemory(dev, inMem, nullptr);
    vkDestroyBuffer(dev, outBuf, nullptr); vkFreeMemory(dev, outMem, nullptr);
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
