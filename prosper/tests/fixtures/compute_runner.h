// compute_runner.h — inline helper to run a compute SPIR-V module over an input float buffer and
// return the output float buffer, via real Vulkan compute (2 storage buffers: binding 0 = in,
// binding 1 = out; local_size_x defaults to 64). Shared by the execution-differential tests so the
// Vulkan boilerplate lives in one place. Header-only; the including test links Vulkan::Vulkan.
#pragma once
#include <vulkan/vulkan.h>
#include "gpu/recompiler/rdna2_to_spirv.hpp"   // kComputeInternalGdsBinding
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace prosper::test {

// FLOAT CONTROLS: this fixture deliberately does NOT publish a SignedZeroInfNanPreserve verdict to
// the recompiler (#3479/#3561), so modules run here carry the neutral, undeclared form. The device
// below is created on a Vulkan 1.1 instance, and VUID-VkShaderModuleCreateInfo-pCode-08742 requires
// 1.2 or VK_KHR_shader_float_controls before the extension may be declared at all -- declaring it
// here is what produced 2 VUIDs x475 across four binaries in CI while every test passed. Publishing
// nothing is not an oversight to repair: it is the fail-safe state, and the property this fixture's
// tests assert (RDNA2 numeric results) does not depend on the declaration, because every device CI
// and this project run on preserves Inf regardless of whether it was asked to. The declaration's
// own guard is the structural `float_controls` ctest plus spirv-val in `tools/spv_validate`.
//
// The hazard this creates, named so it is not discovered the hard way: if a future test process
// creates BOTH one of these 1.1 devices and a 1.2+ device that DOES publish, the published verdict
// is process-wide and the modules that 1.1 device compiles would carry the declaration again.
// Measured today, no test file includes both this fixture and `render_runner.h`. If one ever does,
// make this fixture publish `false` explicitly (publishers are ANDed, so the weaker device wins) --
// and note the failure is loud rather than silent: the validation scan reports it immediately, at
// x475 messages, which is exactly how it was found the first time.

struct ComputeSubgroupProperties {
    uint32_t size = 0;
    VkShaderStageFlags stages = 0;
    VkSubgroupFeatureFlags operations = 0;
};

// Query the same first physical device run_compute() uses. Tests for native subgroup lowering can
// then distinguish a valid host-width result from the stricter guest architectural requirement.
inline ComputeSubgroupProperties default_compute_subgroup_properties() {
    ComputeSubgroupProperties result;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS || !instance) return result;
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count) {
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        VkPhysicalDeviceSubgroupProperties subgroup{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &subgroup;
        vkGetPhysicalDeviceProperties2(devices[0], &properties);
        result = {subgroup.subgroupSize, subgroup.supportedStages, subgroup.supportedOperations};
    }
    vkDestroyInstance(instance, nullptr);
    return result;
}

inline bool default_compute_buffer_int64_atomics_supported() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS || !instance) return false;
    bool result = false;
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count) {
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        VkPhysicalDeviceFeatures core{};
        vkGetPhysicalDeviceFeatures(devices[0], &core);
        uint32_t extension_count = 0;
        vkEnumerateDeviceExtensionProperties(devices[0], nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(
            devices[0], nullptr, &extension_count, extensions.data());
        for (const auto& extension : extensions) {
            if (std::strcmp(extension.extensionName,
                            VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME))
                continue;
            VkPhysicalDeviceShaderAtomicInt64Features atomics{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES};
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.pNext = &atomics;
            vkGetPhysicalDeviceFeatures2(devices[0], &features);
            result = core.shaderInt64 && atomics.shaderBufferInt64Atomics;
            break;
        }
    }
    vkDestroyInstance(instance, nullptr);
    return result;
}

// Run `spirv` over storage buffer 0 = `input`, storage buffer 1 = output. `invocations` compute
// threads are dispatched (default = input.size()) in groups of `local_size_x` (default 64); the
// output buffer holds `out_count` floats (default = input.size()). Returns the output, or {} on any
// Vulkan failure (including rejected SPIR-V).
inline std::vector<float> run_compute(const std::vector<uint32_t>& spirv, const std::vector<float>& input,
                                      uint32_t invocations = 0, uint32_t out_count = 0,
                                      const std::vector<uint32_t>& cbuf = {},
                                      const std::vector<uint32_t>& cbuf1 = {},
                                      std::vector<uint32_t>* cbuf1_out = nullptr,
                                      uint32_t local_size_x = 64,
                                      std::vector<uint32_t>* cbuf_out = nullptr,
                                      // Internal-GDS binding (kComputeInternalGdsBinding = 127).
                                      // Declared and bound UNCONDITIONALLY below: a recompiled module
                                      // that statically uses it -- anything carrying the
                                      // PROSPER_CFG_TRIP_BOUND witness does -- is an invalid pipeline
                                      // without it (VUID-VkComputePipelineCreateInfo-layout-07988 and
                                      // VUID-vkCmdDispatch-None-08114), and a test that dispatches an
                                      // invalid pipeline is measuring undefined behaviour however
                                      // confidently its assertions print. Pass this to read the buffer
                                      // back; leave it null to bind the storage and ignore it.
                                      std::vector<uint32_t>* gds_out = nullptr,
                                      const std::array<std::vector<uint32_t>, 3>* extra_cbufs = nullptr) {
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
    // robustBufferAccess: out-of-range storage-buffer reads/writes are well-defined (return 0 / no-op)
    // rather than UB — so a predicated memory op executed by an inactive lane (narrowed EXEC) is safe.
    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(phys, &supported);
    VkPhysicalDeviceFeatures feats{};
    // Preserve the harness's pre-existing fail-closed requirement: if robustBufferAccess is not
    // supported, vkCreateDevice must fail rather than allowing OOB-sensitive tests to run with UB.
    feats.robustBufferAccess = VK_TRUE;
    feats.shaderInt64 = supported.shaderInt64;
    dci.pEnabledFeatures = &feats;
    VkPhysicalDeviceShaderAtomicInt64Features atomic_int64_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES};
    std::vector<const char*> device_extensions;
    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(
        phys, nullptr, &extension_count, available_extensions.data());
    for (const auto& extension : available_extensions) {
        if (std::strcmp(extension.extensionName,
                        VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME))
            continue;
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &atomic_int64_features;
        vkGetPhysicalDeviceFeatures2(phys, &features);
        if (supported.shaderInt64 && atomic_int64_features.shaderBufferInt64Atomics) {
            VkPhysicalDeviceShaderAtomicInt64Features want{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES};
            want.shaderBufferInt64Atomics = VK_TRUE;
            atomic_int64_features = want;
            dci.pNext = &atomic_int64_features;
            device_extensions.push_back(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
        }
        break;
    }
    dci.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    dci.ppEnabledExtensionNames = device_extensions.empty() ? nullptr : device_extensions.data();
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
    VkBuffer inBuf, outBuf, cbBuf, cbBuf1, gdsBuf;
    VkDeviceMemory inMem, outMem, cbMem, cbMem1, gdsMem;
    std::array<VkBuffer, 3> extra_bufs{};
    std::array<VkDeviceMemory, 3> extra_mems{};
    std::array<VkDeviceSize, 3> extra_bytes{};
    // Matches the live backend's internal GDS allocation (gpu_executor.cpp's g_compute_gds).
    constexpr VkDeviceSize kGdsBytes = 64 * 1024;
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
    if (extra_cbufs)
        for (size_t slot = 0; slot < extra_cbufs->size(); ++slot) {
            const auto& words = (*extra_cbufs)[slot];
            extra_bytes[slot] = std::max<size_t>(words.size(), 1u) * sizeof(uint32_t);
            makeBuf(extra_bufs[slot], extra_mems[slot], extra_bytes[slot]);
            void* mapped = nullptr;
            vkMapMemory(dev, extra_mems[slot], 0, extra_bytes[slot], 0, &mapped);
            std::memset(mapped, 0, extra_bytes[slot]);
            if (!words.empty())
                std::memcpy(mapped, words.data(), words.size() * sizeof(uint32_t));
            vkUnmapMemory(dev, extra_mems[slot]);
        }
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

    makeBuf(gdsBuf, gdsMem, kGdsBytes);
    void* gp = nullptr; vkMapMemory(dev, gdsMem, 0, kGdsBytes, 0, &gp);
    std::memset(gp, 0, kGdsBytes);
    vkUnmapMemory(dev, gdsMem);

    VkDescriptorSetLayoutBinding binds[8]{};
    for (int i = 0; i < 4; i++) { binds[i].binding = i; binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1; binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    const uint32_t extra_count = extra_cbufs ? 3u : 0u;
    for (uint32_t slot = 0; slot < extra_count; ++slot) {
        binds[4 + slot].binding = 4 + slot;
        binds[4 + slot].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[4 + slot].descriptorCount = 1;
        binds[4 + slot].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    binds[4 + extra_count].binding = prosper::gpu::kComputeInternalGdsBinding;
    binds[4 + extra_count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[4 + extra_count].descriptorCount = 1;
    binds[4 + extra_count].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    const uint32_t binding_count = 5u + extra_count;
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = binding_count; dslci.pBindings = binds;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);
    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, binding_count};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &psz;
    VkDescriptorPool dp; vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset; vkAllocateDescriptorSets(dev, &dsai, &dset);
    VkDescriptorBufferInfo bi0{inBuf, 0, VK_WHOLE_SIZE}, bi1{outBuf, 0, VK_WHOLE_SIZE},
                           bi2{cbBuf, 0, VK_WHOLE_SIZE}, bi3{cbBuf1, 0, VK_WHOLE_SIZE},
                           bi4{gdsBuf, 0, VK_WHOLE_SIZE};
    std::array<VkDescriptorBufferInfo, 3> extra_infos{};
    VkWriteDescriptorSet w[8]{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[0].dstSet = dset; w[0].dstBinding = 0;
    w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &bi0;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[1].dstSet = dset; w[1].dstBinding = 1;
    w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &bi1;
    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[2].dstSet = dset; w[2].dstBinding = 2;
    w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[2].pBufferInfo = &bi2;
    w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[3].dstSet = dset; w[3].dstBinding = 3;
    w[3].descriptorCount = 1; w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[3].pBufferInfo = &bi3;
    for (uint32_t slot = 0; slot < extra_count; ++slot) {
        extra_infos[slot] = {extra_bufs[slot], 0, VK_WHOLE_SIZE};
        w[4 + slot] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[4 + slot].dstSet = dset;
        w[4 + slot].dstBinding = 4 + slot;
        w[4 + slot].descriptorCount = 1;
        w[4 + slot].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[4 + slot].pBufferInfo = &extra_infos[slot];
    }
    w[4 + extra_count] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[4 + extra_count].dstSet = dset;
    w[4 + extra_count].dstBinding = prosper::gpu::kComputeInternalGdsBinding;
    w[4 + extra_count].descriptorCount = 1;
    w[4 + extra_count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[4 + extra_count].pBufferInfo = &bi4;
    vkUpdateDescriptorSets(dev, binding_count, w, 0, nullptr);

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
    vkCmdDispatch(cmd, (invocations + local_size_x - 1) / local_size_x, 1, 1);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; vkCreateFence(dev, &fci, nullptr, &fence);
    if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS ||
        vkWaitForFences(dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000) != VK_SUCCESS) {
        // The resources may still be in flight after a timeout. This test process exits after
        // reporting failure; leaking the device is safer than mapping incomplete output or freeing
        // resources that an unfinished dispatch may still access.
        std::fprintf(stderr, "compute_runner: dispatch submission or fence wait failed\n");
        return {};
    }

    out.resize(out_count);
    void* op = nullptr; vkMapMemory(dev, outMem, 0, outBytes, 0, &op);
    for (uint32_t i = 0; i < out_count; i++) out[i] = ((float*)op)[i];
    vkUnmapMemory(dev, outMem);
    // Optional: read back the internal-GDS buffer, so a test can assert what a module wrote there
    // (the PROSPER_CFG_TRIP_BOUND witness lives at its top). The binding itself is always declared
    // and bound, whether or not anyone reads it back.
    if (gds_out) {
        void* g2 = nullptr; vkMapMemory(dev, gdsMem, 0, kGdsBytes, 0, &g2);
        gds_out->assign(reinterpret_cast<const uint32_t*>(g2),
                        reinterpret_cast<const uint32_t*>(g2) + kGdsBytes / sizeof(uint32_t));
        vkUnmapMemory(dev, gdsMem);
    }
    // Optional: read back binding-3 (cbuf1) contents — for verifying MUBUF stores that target it.
    if (cbuf1_out) {
        cbuf1_out->resize(CB1_N);
        void* rp = nullptr; vkMapMemory(dev, cbMem1, 0, cbBytes1, 0, &rp);
        for (uint32_t i = 0; i < CB1_N; i++) (*cbuf1_out)[i] = ((uint32_t*)rp)[i];
        vkUnmapMemory(dev, cbMem1);
    }
    if (cbuf_out) {
        cbuf_out->resize(CB_N);
        void* rp = nullptr; vkMapMemory(dev, cbMem, 0, cbBytes, 0, &rp);
        for (uint32_t i = 0; i < CB_N; i++) (*cbuf_out)[i] = ((uint32_t*)rp)[i];
        vkUnmapMemory(dev, cbMem);
    }

    vkDestroyFence(dev, fence, nullptr); vkDestroyCommandPool(dev, pool, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr); vkDestroyPipelineLayout(dev, layout, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyDescriptorPool(dev, dp, nullptr); vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyBuffer(dev, inBuf, nullptr); vkFreeMemory(dev, inMem, nullptr);
    vkDestroyBuffer(dev, outBuf, nullptr); vkFreeMemory(dev, outMem, nullptr);
    vkDestroyBuffer(dev, cbBuf, nullptr); vkFreeMemory(dev, cbMem, nullptr);
    vkDestroyBuffer(dev, cbBuf1, nullptr); vkFreeMemory(dev, cbMem1, nullptr);
    for (uint32_t slot = 0; slot < extra_count; ++slot) {
        vkDestroyBuffer(dev, extra_bufs[slot], nullptr);
        vkFreeMemory(dev, extra_mems[slot], nullptr);
    }
    vkDestroyBuffer(dev, gdsBuf, nullptr); vkFreeMemory(dev, gdsMem, nullptr);
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);
    return out;
}

} // namespace prosper::test
