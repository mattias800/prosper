// ngg_subgroup_runner.h -- execute a merged-NGG subgroup shell module (gpu/recompiler/
// ngg_subgroup_shell.hpp) offline on real Vulkan and read back its export record buffer.
//
// The module's descriptor bindings are reflected from its own decorations: every (set, binding)
// it declares gets a host-visible storage buffer. Set 2 binding 0 is the launch buffer, set 2
// binding 1 the export buffer; the caller supplies the contents of any set-0 guest buffer and the
// rest are zero-filled. The export buffer is poisoned on the host and then cleared with
// vkCmdFillBuffer in the same command buffer as the dispatch, which is the clearing contract the
// shell's consumer relies on (an unwritten flag means "not exported"). Header-only; the including
// test links Vulkan::Vulkan.
#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace prosper::test {

struct NggSubgroupDispatch {
    std::vector<uint32_t> spirv;
    uint32_t local_size = 64;   // 64 * waves
    uint32_t workgroups = 1;
    std::vector<uint32_t> launch;   // set 2 binding 0
    uint32_t export_words = 0;      // set 2 binding 1
    std::map<uint32_t, std::vector<uint32_t>> guest_buffers;   // set 0 binding -> words
    std::vector<uint32_t> push_constants;
    uint32_t required_subgroup_size = 0;   // 64 = the native Wave64 shell
};

// The set/binding pairs a module declares, or nullopt for a module this runner cannot bind
// (images, samplers, or a descriptor array).
inline std::optional<std::set<std::pair<uint32_t, uint32_t>>>
ngg_module_bindings(const std::vector<uint32_t>& spirv) {
    std::map<uint32_t, uint32_t> set_of, binding_of;
    bool unsupported = false;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t count = spirv[word] >> 16, opcode = spirv[word] & 0xffffu;
        if (!count || word + count > spirv.size()) return std::nullopt;
        if (opcode == 71 && count >= 4) {   // OpDecorate
            if (spirv[word + 2] == 34) set_of[spirv[word + 1]] = spirv[word + 3];
            if (spirv[word + 2] == 33) binding_of[spirv[word + 1]] = spirv[word + 3];
        }
        if (opcode == 25 || opcode == 26 || opcode == 27) unsupported = true;   // images/samplers
        word += count;
    }
    if (unsupported) return std::nullopt;
    std::set<std::pair<uint32_t, uint32_t>> bindings;
    for (const auto& [id, set] : set_of) {
        const auto binding = binding_of.find(id);
        if (binding == binding_of.end()) return std::nullopt;
        bindings.insert({set, binding->second});
    }
    return bindings;
}

inline bool ngg_required_subgroup_supported(VkPhysicalDevice phys, uint32_t size, uint32_t local) {
    VkPhysicalDeviceProperties basic{};
    vkGetPhysicalDeviceProperties(phys, &basic);
    if (basic.apiVersion < VK_API_VERSION_1_3 || local % size) return false;
    VkPhysicalDeviceVulkan13Features features13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &features13;
    vkGetPhysicalDeviceFeatures2(phys, &features);
    VkPhysicalDeviceSubgroupProperties subgroup{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceVulkan13Properties properties13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    properties13.pNext = &subgroup;
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &properties13;
    vkGetPhysicalDeviceProperties2(phys, &properties);
    constexpr VkSubgroupFeatureFlags kNeeded =
        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
        VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    return features13.subgroupSizeControl && features13.computeFullSubgroups &&
           properties13.minSubgroupSize <= size && size <= properties13.maxSubgroupSize &&
           (properties13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
           local / size <= properties13.maxComputeWorkgroupSubgroups &&
           (subgroup.supportedOperations & kNeeded) == kNeeded;
}

// Whether the first physical device can run the native Wave64 shell for `waves` guest waves.
inline bool ngg_native_wave64_supported(uint32_t waves) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return false;
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    bool supported = false;
    if (count) {
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        supported = ngg_required_subgroup_supported(devices[0], 64, 64 * waves);
    }
    vkDestroyInstance(instance, nullptr);
    return supported;
}

// Returns the export buffer, or nullopt on any Vulkan failure (the reason goes to stderr).
inline std::optional<std::vector<uint32_t>> run_ngg_subgroup(const NggSubgroupDispatch& run) {
    const auto bindings = ngg_module_bindings(run.spirv);
    if (!bindings || run.spirv.empty() || !run.workgroups || !run.export_words ||
        run.launch.empty()) {
        std::fprintf(stderr, "ngg_subgroup_runner: unsupported module or empty dispatch\n");
        return std::nullopt;
    }
    for (const auto& [set, binding] : *bindings)
        if (set != 0 && set != 2) {
            std::fprintf(stderr, "ngg_subgroup_runner: unexpected descriptor set %u\n", set);
            return std::nullopt;
        }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return std::nullopt;
    VkDevice dev = VK_NULL_HANDLE;
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> buffers;
    VkDescriptorSetLayout layouts[3] = {};
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    const auto finish = [&](const char* failure) -> std::optional<std::vector<uint32_t>> {
        if (failure) std::fprintf(stderr, "ngg_subgroup_runner: %s\n", failure);
        if (dev) {
            if (command_pool) vkDestroyCommandPool(dev, command_pool, nullptr);
            if (pipeline) vkDestroyPipeline(dev, pipeline, nullptr);
            if (module) vkDestroyShaderModule(dev, module, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
            if (pool) vkDestroyDescriptorPool(dev, pool, nullptr);
            for (VkDescriptorSetLayout layout : layouts)
                if (layout) vkDestroyDescriptorSetLayout(dev, layout, nullptr);
            for (auto& [buffer, memory] : buffers) {
                vkDestroyBuffer(dev, buffer, nullptr);
                vkFreeMemory(dev, memory, nullptr);
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
    if (run.required_subgroup_size &&
        !ngg_required_subgroup_supported(phys, run.required_subgroup_size, run.local_size))
        return finish("required subgroup size unsupported");
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, families.data());
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; ++i)
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            family = i;
            break;
        }
    if (family == UINT32_MAX) return finish("no compute queue");
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(phys, &supported);
    VkPhysicalDeviceFeatures features{};
    features.robustBufferAccess = VK_TRUE;
    // The recompiler declares Int64 for some lowerings (64-bit scalar pairs); enable it where the
    // device has it, exactly as the compute runner does.
    features.shaderInt64 = supported.shaderInt64;
    VkPhysicalDeviceVulkan13Features features13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.subgroupSizeControl = run.required_subgroup_size ? VK_TRUE : VK_FALSE;
    features13.computeFullSubgroups = run.required_subgroup_size ? VK_TRUE : VK_FALSE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &features;
    if (run.required_subgroup_size) dci.pNext = &features13;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS) return finish("vkCreateDevice");
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev, family, 0, &queue);

    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(phys, &memory_properties);
    constexpr VkMemoryPropertyFlags kHost =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const auto make_buffer = [&](const std::vector<uint32_t>& words, uint32_t size_words,
                                 uint32_t fill) -> VkBuffer {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = static_cast<VkDeviceSize>(std::max<size_t>(size_words, 1)) * 4u;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkBuffer buffer = VK_NULL_HANDLE;
        if (vkCreateBuffer(dev, &bci, nullptr, &buffer) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(dev, buffer, &requirements);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memory_properties.memoryTypes[i].propertyFlags & kHost) == kHost) {
                type = i;
                break;
            }
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = requirements.size;
        ai.memoryTypeIndex = type;
        if (type == UINT32_MAX || vkAllocateMemory(dev, &ai, nullptr, &memory) != VK_SUCCESS) {
            vkDestroyBuffer(dev, buffer, nullptr);
            return VK_NULL_HANDLE;
        }
        buffers.push_back({buffer, memory});
        void* mapped = nullptr;
        if (vkBindBufferMemory(dev, buffer, memory, 0) != VK_SUCCESS ||
            vkMapMemory(dev, memory, 0, bci.size, 0, &mapped) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        auto* out = static_cast<uint32_t*>(mapped);
        for (size_t i = 0; i < bci.size / 4u; ++i) out[i] = i < words.size() ? words[i] : fill;
        vkUnmapMemory(dev, memory);
        return buffer;
    };

    std::map<std::pair<uint32_t, uint32_t>, VkBuffer> bound;
    std::map<std::pair<uint32_t, uint32_t>, VkDeviceSize> bound_bytes;
    VkBuffer export_buffer = VK_NULL_HANDLE;
    for (const auto& key : *bindings) {
        VkBuffer buffer = VK_NULL_HANDLE;
        uint32_t words = 64;
        if (key == std::make_pair(2u, 0u)) {
            words = static_cast<uint32_t>(run.launch.size());
            buffer = make_buffer(run.launch, words, 0);
        } else if (key == std::make_pair(2u, 1u)) {
            words = run.export_words;
            buffer = make_buffer({}, words, 0xcdcdcdcdu);   // poisoned; the fill clears it
            export_buffer = buffer;
        } else {
            const auto it = run.guest_buffers.find(key.second);
            const std::vector<uint32_t> empty;
            const auto& data = key.first == 0 && it != run.guest_buffers.end() ? it->second : empty;
            words = std::max<uint32_t>(64u, static_cast<uint32_t>(data.size()));
            buffer = make_buffer(data, words, 0);
        }
        if (!buffer) return finish("buffer allocation");
        bound[key] = buffer;
        bound_bytes[key] = static_cast<VkDeviceSize>(std::max(words, 1u)) * 4u;
    }
    if (!export_buffer || !bound.contains({2u, 0u})) return finish("module lacks shell bindings");

    for (uint32_t set = 0; set < 3; ++set) {
        std::vector<VkDescriptorSetLayoutBinding> entries;
        for (const auto& key : *bindings) {
            if (key.first != set) continue;
            VkDescriptorSetLayoutBinding entry{};
            entry.binding = key.second;
            entry.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            entry.descriptorCount = 1;
            entry.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            entries.push_back(entry);
        }
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = static_cast<uint32_t>(entries.size());
        lci.pBindings = entries.empty() ? nullptr : entries.data();
        if (vkCreateDescriptorSetLayout(dev, &lci, nullptr, &layouts[set]) != VK_SUCCESS)
            return finish("descriptor set layout");
    }
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                   static_cast<uint32_t>(bindings->size())};
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
    std::vector<VkDescriptorBufferInfo> infos;
    infos.reserve(bound.size());
    std::vector<VkWriteDescriptorSet> writes;
    for (const auto& [key, buffer] : bound) {
        infos.push_back({buffer, 0, bound_bytes[key]});
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = sets[key.first];
        write.dstBinding = key.second;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &infos.back();
        writes.push_back(write);
    }
    vkUpdateDescriptorSets(dev, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             static_cast<uint32_t>(run.push_constants.size() * 4u)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 3;
    plci.pSetLayouts = layouts;
    plci.pushConstantRangeCount = run.push_constants.empty() ? 0u : 1u;
    plci.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(dev, &plci, nullptr, &pipeline_layout) != VK_SUCCESS)
        return finish("pipeline layout");
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = run.spirv.size() * 4u;
    smci.pCode = run.spirv.data();
    if (vkCreateShaderModule(dev, &smci, nullptr, &module) != VK_SUCCESS) return finish("module");
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    required.requiredSubgroupSize = run.required_subgroup_size;
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization): every stage field is set below.
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    if (run.required_subgroup_size) {
        cpci.stage.pNext = &required;
        cpci.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }
    cpci.layout = pipeline_layout;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline) != VK_SUCCESS)
        return finish("compute pipeline");

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
    vkCmdFillBuffer(cmd, export_buffer, 0, VK_WHOLE_SIZE, 0);
    VkMemoryBarrier fill_done{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    fill_done.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fill_done.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &fill_done, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 3, sets, 0,
                            nullptr);
    if (!run.push_constants.empty())
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push.size,
                           run.push_constants.data());
    vkCmdDispatch(cmd, run.workgroups, 1, 1);
    VkMemoryBarrier host_read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host_read.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    host_read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                         1, &host_read, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS ||
        vkQueueWaitIdle(queue) != VK_SUCCESS)
        return finish("submit");

    std::vector<uint32_t> result(run.export_words);
    for (auto& [buffer, memory] : buffers) {
        if (buffer != export_buffer) continue;
        void* mapped = nullptr;
        if (vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
            return finish("export map");
        std::memcpy(result.data(), mapped, result.size() * 4u);
        vkUnmapMemory(dev, memory);
    }
    finish(nullptr);
    return result;
}

}   // namespace prosper::test
