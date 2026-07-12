#include "live_compute.hpp"

#include "gpu/gpu_execute.hpp"
#include "gpu/shader_resources.hpp"
#include "gpu/writer_provenance.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

namespace prosper::frontend {
namespace {

struct VulkanComputeContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = UINT32_MAX;
    VkPhysicalDeviceMemoryProperties memory{};

    ~VulkanComputeContext() {
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    bool init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return false;

        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
        if (!device_count) return false;
        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
        physical = devices[0];

        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
        for (uint32_t i = 0; i < family_count; i++) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queue_family = i;
                break;
            }
        }
        if (queue_family == UINT32_MAX) return false;

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        VkPhysicalDeviceFeatures supported{};
        vkGetPhysicalDeviceFeatures(physical, &supported);
        if (!supported.robustBufferAccess) {
            std::fprintf(stderr, "[compute] device lacks robustBufferAccess\n");
            return false;
        }
        VkPhysicalDeviceFeatures enabled{};
        enabled.robustBufferAccess = VK_TRUE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.pEnabledFeatures = &enabled;
        if (vkCreateDevice(physical, &dci, nullptr, &device) != VK_SUCCESS) return false;
        vkGetDeviceQueue(device, queue_family, 0, &queue);
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        return true;
    }

    uint32_t host_memory_type(uint32_t bits) const {
        const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < memory.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & wanted) == wanted)
                return i;
        return UINT32_MAX;
    }
};

struct BoundBuffer {
    const prosper::gpu::ShaderResource* resource = nullptr;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint64_t before_hash = 0, after_hash = 0;
    uint64_t changed_bytes = 0;
};

uint64_t fnv1a(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

bool execute_item(VulkanComputeContext& ctx, const prosper::gpu::ComputeItem& item) {
    using namespace prosper::gpu;
    const bool trace = std::getenv("PROSPER_COMPUTELOG") != nullptr;
    auto report = validate_spirv_descriptor_interface(
        item.spirv, item.resources.get(), 0, SpirvShaderStage::Compute, false);
    if (!report.ok()) return false;

    std::vector<SpirvDescriptorBinding> descriptors;
    for (const auto& descriptor : report.descriptors) {
        if (descriptor.kind != SpirvDescriptorKind::StorageBuffer) {
            std::fprintf(stderr, "[compute] program 0x%llx uses unsupported %s binding %u\n",
                         (unsigned long long)item.code_addr,
                         spirv_descriptor_kind_name(descriptor.kind), descriptor.binding);
            return false;
        }
        descriptors.push_back(descriptor);
    }
    if (descriptors.empty()) return false;
    std::sort(descriptors.begin(), descriptors.end(), [](const auto& a, const auto& b) {
        return a.binding < b.binding;
    });

    std::vector<BoundBuffer> buffers(descriptors.size());
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool ok = false;

    auto cleanup = [&] {
        if (fence) vkDestroyFence(ctx.device, fence, nullptr);
        if (command_pool) vkDestroyCommandPool(ctx.device, command_pool, nullptr);
        if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
        if (shader) vkDestroyShaderModule(ctx.device, shader, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
        if (descriptor_layout) vkDestroyDescriptorSetLayout(ctx.device, descriptor_layout, nullptr);
        for (auto& buffer : buffers) {
            if (buffer.buffer) vkDestroyBuffer(ctx.device, buffer.buffer, nullptr);
            if (buffer.memory) vkFreeMemory(ctx.device, buffer.memory, nullptr);
        }
    };
    auto resource_bytes = [](const ShaderResource* resource) -> uint8_t* {
        if (resource->host_data && resource->host_data_size >= resource->size)
            return resource->host_data;
        return reinterpret_cast<uint8_t*>(uintptr_t(resource->gpu_addr));
    };

    do {
        std::vector<VkDescriptorSetLayoutBinding> layout_bindings(descriptors.size());
        for (size_t i = 0; i < descriptors.size(); i++) {
            const ShaderResource* resource = item.resources->by_binding(descriptors[i].binding);
            if (!resource || !resource->size ||
                ((!resource->host_data || resource->host_data_size < resource->size) &&
                 !guest_readable(resource->gpu_addr, resource->size))) break;
            buffers[i].resource = resource;
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = resource->size;
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            if (vkCreateBuffer(ctx.device, &bci, nullptr, &buffers[i].buffer) != VK_SUCCESS) break;
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(ctx.device, buffers[i].buffer, &requirements);
            const uint32_t memory_type = ctx.host_memory_type(requirements.memoryTypeBits);
            if (memory_type == UINT32_MAX) break;
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = requirements.size;
            mai.memoryTypeIndex = memory_type;
            if (vkAllocateMemory(ctx.device, &mai, nullptr, &buffers[i].memory) != VK_SUCCESS) break;
            if (vkBindBufferMemory(ctx.device, buffers[i].buffer, buffers[i].memory, 0) != VK_SUCCESS) break;
            void* mapped = nullptr;
            if (vkMapMemory(ctx.device, buffers[i].memory, 0, resource->size, 0, &mapped) != VK_SUCCESS) break;
            const uint8_t* source = resource_bytes(resource);
            if (trace) buffers[i].before_hash = fnv1a(source, resource->size);
            std::memcpy(mapped, source, resource->size);
            vkUnmapMemory(ctx.device, buffers[i].memory);

            layout_bindings[i].binding = descriptors[i].binding;
            layout_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            layout_bindings[i].descriptorCount = 1;
            layout_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        bool buffers_ready = true;
        for (const auto& buffer : buffers) buffers_ready &= buffer.resource && buffer.memory;
        if (!buffers_ready) break;

        VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dlci.bindingCount = static_cast<uint32_t>(layout_bindings.size());
        dlci.pBindings = layout_bindings.data();
        if (vkCreateDescriptorSetLayout(ctx.device, &dlci, nullptr, &descriptor_layout) != VK_SUCCESS) break;
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       static_cast<uint32_t>(buffers.size())};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &pool_size;
        if (vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &descriptor_pool) != VK_SUCCESS) break;
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = descriptor_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &descriptor_layout;
        if (vkAllocateDescriptorSets(ctx.device, &dsai, &descriptor_set) != VK_SUCCESS) break;

        std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size());
        for (size_t i = 0; i < buffers.size(); i++) {
            buffer_infos[i] = {buffers[i].buffer, 0, buffers[i].resource->size};
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = descriptors[i].binding;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[i];
        }
        vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = item.spirv.size() * sizeof(uint32_t);
        smci.pCode = item.spirv.data();
        if (vkCreateShaderModule(ctx.device, &smci, nullptr, &shader) != VK_SUCCESS) break;
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &descriptor_layout;
        if (vkCreatePipelineLayout(ctx.device, &plci, nullptr, &pipeline_layout) != VK_SUCCESS) break;
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = shader;
        cpci.stage.pName = "main";
        cpci.layout = pipeline_layout;
        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline) != VK_SUCCESS) break;

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = ctx.queue_family;
        if (vkCreateCommandPool(ctx.device, &pci, nullptr, &command_pool) != VK_SUCCESS) break;
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = command_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(ctx.device, &cbai, &command) != VK_SUCCESS) break;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) break;
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                                0, 1, &descriptor_set, 0, nullptr);
        vkCmdDispatch(command, item.launch.groups_x, item.launch.groups_y, item.launch.groups_z);
        if (vkEndCommandBuffer(command) != VK_SUCCESS) break;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(ctx.device, &fci, nullptr, &fence) != VK_SUCCESS) break;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        if (vkQueueSubmit(ctx.queue, 1, &submit, fence) != VK_SUCCESS) break;
        if (vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, 30ull * 1000 * 1000 * 1000) != VK_SUCCESS) break;

        bool readback_ok = true;
        for (auto& buffer : buffers) {
            void* mapped = nullptr;
            if (vkMapMemory(ctx.device, buffer.memory, 0, buffer.resource->size, 0, &mapped) != VK_SUCCESS) {
                readback_ok = false;
                break;
            }
            uint8_t* destination = resource_bytes(buffer.resource);
            const auto* result = static_cast<const uint8_t*>(mapped);
            if (trace) {
                buffer.after_hash = fnv1a(result, buffer.resource->size);
                for (uint32_t i = 0; i < buffer.resource->size; i++)
                    buffer.changed_bytes += destination[i] != result[i];
            }
            std::memcpy(destination, result, buffer.resource->size);
            vkUnmapMemory(ctx.device, buffer.memory);
            if (!buffer.resource->host_data && writer_provenance_enabled())
                record_guest_write(GuestWriterKind::ComputeBuffer,
                                   buffer.resource->gpu_addr, buffer.resource->size,
                                   item.submit_no, item.dispatch_index,
                                   item.command_order, item.code_addr);
        }
        if (!readback_ok) break;
        ok = true;
    } while (false);

    if (trace)
        std::fprintf(stderr, "[compute] execute submit=%llu dispatch=%llu code=0x%llx "
                     "groups=%ux%ux%u buffers=%zu result=%s\n",
                     (unsigned long long)item.submit_no, (unsigned long long)item.dispatch_index,
                     (unsigned long long)item.code_addr, item.launch.groups_x, item.launch.groups_y,
                     item.launch.groups_z, buffers.size(), ok ? "ok" : "failed");
    if (trace) {
        for (const auto& buffer : buffers) {
            if (!buffer.resource) continue;
            const uint8_t* bytes = resource_bytes(buffer.resource);
            std::fprintf(stderr,
                         "[compute]   writeback binding=%u addr=0x%llx size=%u changed=%llu "
                         "hash=%016llx->%016llx first=%08x,%08x,%08x,%08x\n",
                         buffer.resource->binding, (unsigned long long)buffer.resource->gpu_addr,
                         buffer.resource->size, (unsigned long long)buffer.changed_bytes,
                         (unsigned long long)buffer.before_hash, (unsigned long long)buffer.after_hash,
                         buffer.resource->size >= 4 ? reinterpret_cast<const uint32_t*>(bytes)[0] : 0,
                         buffer.resource->size >= 8 ? reinterpret_cast<const uint32_t*>(bytes)[1] : 0,
                         buffer.resource->size >= 12 ? reinterpret_cast<const uint32_t*>(bytes)[2] : 0,
                         buffer.resource->size >= 16 ? reinterpret_cast<const uint32_t*>(bytes)[3] : 0);
        }
    }
    cleanup();
    return ok;
}

} // namespace

bool execute_live_compute_items(const std::vector<prosper::gpu::ComputeItem>& items) {
    VulkanComputeContext context;
    if (!context.init()) {
        std::fprintf(stderr, "[compute] Vulkan initialization failed\n");
        return false;
    }
    for (const auto& item : items)
        if (!execute_item(context, item)) return false;
    return true;
}

void register_live_compute() {
    static bool attempted = false;
    if (attempted) return;
    attempted = true;
    const char* enabled = std::getenv("PROSPER_COMPUTE");
    if (enabled && (!std::strcmp(enabled, "0") || !std::strcmp(enabled, "off"))) {
        std::fprintf(stderr, "[compute] live execution disabled by PROSPER_COMPUTE=%s\n", enabled);
        return;
    }
    prosper::gpu::set_submit_compute(execute_live_compute_items);
    std::fprintf(stderr, "[compute] live Vulkan compute backend registered\n");
}

} // namespace prosper::frontend
