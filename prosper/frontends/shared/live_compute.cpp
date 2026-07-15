#include "live_compute.hpp"

#include "gpu/bc_decode.hpp"
#include "gpu/gpu_execute.hpp"
#include "gpu/shader_resources.hpp"
#include "gpu/tile.hpp"
#include "gpu/writer_provenance.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace prosper::frontend {
namespace {

constexpr VkDeviceSize kMaxComputeImageBytes = 256ull << 20;

struct ComputeMemoryKey {
    VkDeviceSize bytes = 0;
    uint32_t memory_type = UINT32_MAX;
    bool operator==(const ComputeMemoryKey& other) const {
        return bytes == other.bytes && memory_type == other.memory_type;
    }
};

struct ComputeMemoryKeyHash {
    size_t operator()(const ComputeMemoryKey& key) const {
        return std::hash<uint64_t>{}(static_cast<uint64_t>(key.bytes)) ^
               (std::hash<uint32_t>{}(key.memory_type) << 1);
    }
};

struct ComputeMemoryPool {
    std::mutex mutex;
    std::unordered_map<ComputeMemoryKey, std::vector<VkDeviceMemory>, ComputeMemoryKeyHash> available;
    std::unordered_map<VkDeviceMemory, ComputeMemoryKey> active;
    VkDeviceSize cached_bytes = 0;
    size_t cached_allocations = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t discarded = 0;
};

struct ComputeMemoryPoolStats {
    VkDeviceSize cached_bytes = 0;
    size_t cached_allocations = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t discarded = 0;
};

bool compute_memory_pool_enabled() {
    static const bool enabled = std::getenv("PROSPER_NO_MEMORY_POOL") == nullptr;
    return enabled;
}

VkDeviceSize compute_memory_pool_limit() {
    static const VkDeviceSize limit = []() -> VkDeviceSize {
        const char* value = std::getenv("PROSPER_COMPUTE_MEMORY_POOL_MB");
        const uint64_t mib = value ? std::strtoull(value, nullptr, 10) : 256ull;
        if (mib > UINT64_MAX / (1024ull * 1024ull)) return VkDeviceSize{UINT64_MAX};
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    return limit;
}

struct VulkanComputeContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = UINT32_MAX;
    VkPhysicalDeviceMemoryProperties memory{};
    ComputeMemoryPool memory_pool;
    // Storage-image support (#590): the recompiler's storage path declares the
    // StorageImageRead/WriteWithoutFormat capabilities (raw uvec4 texel model — see
    // tests/image_compute_runner.h, the exec-diff harness for that contract). When the device lacks
    // the features, image-binding dispatches are skipped loudly instead of creating an invalid device.
    bool image_support = false;

    ~VulkanComputeContext() {
        release_cached_memory();
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    VkDeviceMemory allocate_memory(VkDeviceSize bytes, uint32_t memory_type) {
        if (memory_type == UINT32_MAX) return VK_NULL_HANDLE;
        const ComputeMemoryKey key{bytes, memory_type};
        if (compute_memory_pool_enabled()) {
            std::lock_guard<std::mutex> lock(memory_pool.mutex);
            auto found = memory_pool.available.find(key);
            if (found != memory_pool.available.end() && !found->second.empty()) {
                const VkDeviceMemory allocation = found->second.back();
                found->second.pop_back();
                if (found->second.empty()) memory_pool.available.erase(found);
                memory_pool.cached_bytes -= bytes;
                --memory_pool.cached_allocations;
                ++memory_pool.hits;
                memory_pool.active.emplace(allocation, key);
                return allocation;
            }
            ++memory_pool.misses;
        }

        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = bytes;
        allocation.memoryTypeIndex = memory_type;
        VkDeviceMemory result = VK_NULL_HANDLE;
        if (vkAllocateMemory(device, &allocation, nullptr, &result) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        if (compute_memory_pool_enabled()) {
            std::lock_guard<std::mutex> lock(memory_pool.mutex);
            memory_pool.active.emplace(result, key);
        }
        return result;
    }

    void release_memory(VkDeviceMemory allocation) {
        if (!allocation) return;
        if (!compute_memory_pool_enabled()) {
            vkFreeMemory(device, allocation, nullptr);
            return;
        }
        std::lock_guard<std::mutex> lock(memory_pool.mutex);
        auto found = memory_pool.active.find(allocation);
        if (found == memory_pool.active.end()) {
            vkFreeMemory(device, allocation, nullptr);
            return;
        }
        const ComputeMemoryKey key = found->second;
        memory_pool.active.erase(found);
        constexpr size_t max_cached_allocations = 2048;
        const VkDeviceSize limit = compute_memory_pool_limit();
        const VkDeviceSize remaining = memory_pool.cached_bytes < limit
            ? limit - memory_pool.cached_bytes : 0;
        if (memory_pool.cached_allocations >= max_cached_allocations || key.bytes > remaining) {
            ++memory_pool.discarded;
            vkFreeMemory(device, allocation, nullptr);
            return;
        }
        memory_pool.available[key].push_back(allocation);
        memory_pool.cached_bytes += key.bytes;
        ++memory_pool.cached_allocations;
    }

    ComputeMemoryPoolStats memory_pool_stats() {
        std::lock_guard<std::mutex> lock(memory_pool.mutex);
        return {memory_pool.cached_bytes, memory_pool.cached_allocations, memory_pool.hits,
                memory_pool.misses, memory_pool.discarded};
    }

    void release_cached_memory() {
        if (!device) return;
        std::lock_guard<std::mutex> lock(memory_pool.mutex);
        for (const auto& [key, allocations] : memory_pool.available) {
            (void)key;
            for (VkDeviceMemory allocation : allocations)
                vkFreeMemory(device, allocation, nullptr);
        }
        for (const auto& [allocation, key] : memory_pool.active) {
            (void)key;
            vkFreeMemory(device, allocation, nullptr);
        }
        memory_pool.available.clear();
        memory_pool.active.clear();
        memory_pool.cached_bytes = 0;
        memory_pool.cached_allocations = 0;
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
        // Image bindings (#590): enable the format-free storage-image features when available.
        image_support = supported.shaderStorageImageReadWithoutFormat &&
                        supported.shaderStorageImageWriteWithoutFormat;
        if (image_support) {
            enabled.shaderStorageImageReadWithoutFormat = VK_TRUE;
            enabled.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        }
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.pEnabledFeatures = &enabled;
        std::vector<const char*> dev_exts;
#ifdef __APPLE__
        // Spec-mandated on MoltenVK: enable VK_KHR_portability_subset when advertised (always is).
        { uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(physical, nullptr, &ne, nullptr);
          std::vector<VkExtensionProperties> de(ne);
          vkEnumerateDeviceExtensionProperties(physical, nullptr, &ne, de.data());
          for (auto& e : de) if (!std::strcmp(e.extensionName, "VK_KHR_portability_subset")) {
              dev_exts.push_back("VK_KHR_portability_subset"); break; } }
        dci.enabledExtensionCount = (uint32_t)dev_exts.size();
        dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
#endif
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

// One image binding (#590): a sampled texture (usually RGBA8, with native UINT8x4 and R11G11B10F
// views where shader-visible numeric semantics require them) or a storage image (R32G32B32A32_UINT
// texels — the recompiler's format-free contract, exec-diff proven by tests/image_compute_runner.h).
struct BoundImage {
    const prosper::gpu::ShaderResource* resource = nullptr;
    uint32_t binding = 0;
    bool storage = false;               // storage image: read back + pack to guest after the dispatch
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE; // combined image sampler only
    VkDeviceSize row_pitch = 0;         // LINEAR-tiling row pitch (bytes), from vkGetImageSubresourceLayout
    size_t guest_bytes = 0;             // real linear/tiled guest backing footprint
    size_t alias_of = SIZE_MAX;         // identical storage binding sharing an earlier image/view
    uint64_t before_hash = 0, after_hash = 0; // trace-only storage-image writeback evidence
    uint64_t nonzero_channels = 0;
};

// --- Storage-image channel model (#590) -------------------------------------------------------------
// The recompiler moves image texels as RAW 32-bit VGPR channel values (uvec4 per texel; the VkImage is
// R32G32B32A32_UINT with format-free reads/writes). Real hardware format-converts per the T#, so the
// guest surface's bytes must be UNPACKED to channel dwords on upload and PACKED back on writeback:
//   Unorm8  -> float(b/255) bits         <- clamp(bitcast float,0,1)*255 rounded
//   Float16 -> half->float bits          <- round-to-nearest-even float->half
//   Float32/Uint32/Sint32 -> raw 4-byte move both ways.
// Missing channels read the hardware default (0,0,0,1.0f). Anything else is unsupported -> the caller
// skips the dispatch loudly (never a silent wrong-layout write — correctness-first).
bool storage_unpack_supported(prosper::gpu::DataFormat f) {
    using DF = prosper::gpu::DataFormat;
    return f == DF::Unorm8 || f == DF::Float16 || f == DF::Float32 || f == DF::Uint32 || f == DF::Sint32;
}
bool storage_pack_supported(prosper::gpu::DataFormat f) {
    using DF = prosper::gpu::DataFormat;
    return f == DF::Unorm8 || f == DF::Float16 || f == DF::Float32 ||
           f == DF::Uint32 || f == DF::Sint32;
}
void storage_unpack_texel(const uint8_t* src, prosper::gpu::DataFormat f, uint32_t ncomp, uint32_t out[4]) {
    using DF = prosper::gpu::DataFormat;
    const uint32_t one_f32 = 0x3f800000u;                 // hardware default: missing channels = (0,0,0,1.0f)
    out[0] = out[1] = out[2] = 0; out[3] = one_f32;
    for (uint32_t c = 0; c < ncomp && c < 4; c++) {
        switch (f) {
            case DF::Unorm8: { float v = src[c] / 255.0f; std::memcpy(&out[c], &v, 4); break; }
            case DF::Float16: { float v = prosper::gpu::half_to_float(
                                    (uint16_t)(src[c * 2] | (src[c * 2 + 1] << 8)));
                                std::memcpy(&out[c], &v, 4); break; }
            default: std::memcpy(&out[c], src + c * 4, 4); break;   // 32-bit raw
        }
    }
}
void storage_pack_texel(const uint32_t in[4], prosper::gpu::DataFormat f, uint32_t ncomp, uint8_t* dst) {
    using DF = prosper::gpu::DataFormat;
    for (uint32_t c = 0; c < ncomp && c < 4; c++) {
        switch (f) {
            case DF::Unorm8: { float v; std::memcpy(&v, &in[c], 4);
                               if (std::isnan(v)) v = 0.f;
                               v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                               dst[c] = (uint8_t)std::lround(v * 255.0f); break; }
            case DF::Float16: { float v; std::memcpy(&v, &in[c], 4);
                                const uint16_t h = prosper::gpu::float_to_half(v);
                                dst[c * 2] = static_cast<uint8_t>(h);
                                dst[c * 2 + 1] = static_cast<uint8_t>(h >> 8); break; }
            default: std::memcpy(dst + c * 4, &in[c], 4); break;    // 32-bit raw
        }
    }
}

uint64_t fnv1a(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

bool trace_compute_item(const prosper::gpu::ComputeItem& item) {
    if (std::getenv("PROSPER_COMPUTELOG")) return true;
    const char* code_env = std::getenv("PROSPER_COMPUTELOG_CODE");
    const char* size_env = std::getenv("PROSPER_COMPUTELOG_SIZE");
    if ((!code_env || !*code_env) && (!size_env || !*size_env)) return false;
    if (code_env && *code_env) {
        char* end = nullptr;
        const uint64_t wanted = std::strtoull(code_env, &end, 0);
        if (!end || *end || item.code_addr != wanted) return false;
    }
    if (size_env && *size_env) {
        char* end = nullptr;
        const unsigned long wanted = std::strtoul(size_env, &end, 0);
        if (!end || *end || !item.resources) return false;
        const auto found = std::find_if(item.resources->resources.begin(),
                                       item.resources->resources.end(),
            [&](const auto& resource) { return resource.size == wanted; });
        if (found == item.resources->resources.end()) return false;
    }
    return true;
}

bool execute_item(VulkanComputeContext& ctx, const prosper::gpu::ComputeItem& item) {
    using namespace prosper::gpu;
    const bool trace = trace_compute_item(item);
    auto report = validate_spirv_descriptor_interface(
        item.spirv, item.resources.get(), 0, SpirvShaderStage::Compute, false);
    if (!report.ok()) return false;

    std::vector<SpirvDescriptorBinding> descriptors;       // storage buffers
    std::vector<SpirvDescriptorBinding> image_descriptors; // sampled + storage images (#590)
    for (const auto& descriptor : report.descriptors) {
        switch (descriptor.kind) {
            case SpirvDescriptorKind::StorageBuffer:
                descriptors.push_back(descriptor);
                break;
            case SpirvDescriptorKind::CombinedImageSampler:
            case SpirvDescriptorKind::StorageImage:
                image_descriptors.push_back(descriptor);
                break;
            default:
                std::fprintf(stderr, "[compute] program 0x%llx uses unsupported %s binding %u\n",
                             (unsigned long long)item.code_addr,
                             spirv_descriptor_kind_name(descriptor.kind), descriptor.binding);
                return false;
        }
    }
    if (descriptors.empty() && image_descriptors.empty()) return false;
    const bool has_storage_images = std::any_of(
        image_descriptors.begin(), image_descriptors.end(), [](const auto& descriptor) {
            return descriptor.kind == SpirvDescriptorKind::StorageImage;
        });
    if (has_storage_images && !ctx.image_support) {
        static bool warned = false;
        if (!warned) { warned = true;
            std::fprintf(stderr, "[compute] device lacks shaderStorageImageRead/WriteWithoutFormat; "
                                 "image-binding dispatches are skipped\n"); }
        return false;
    }
    std::sort(descriptors.begin(), descriptors.end(), [](const auto& a, const auto& b) {
        return a.binding < b.binding;
    });
    std::sort(image_descriptors.begin(), image_descriptors.end(), [](const auto& a, const auto& b) {
        return a.binding < b.binding;
    });

    std::vector<BoundBuffer> buffers(descriptors.size());
    std::vector<BoundImage> images(image_descriptors.size());
    std::vector<VkBuffer> staging(image_descriptors.size(), VK_NULL_HANDLE);          // upload/readback
    std::vector<VkDeviceMemory> staging_memory(image_descriptors.size(), VK_NULL_HANDLE);
    std::vector<VkDeviceSize> staging_bytes(image_descriptors.size(), 0);
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool ok = false;
    auto vk_ok = [&](VkResult result, const char* stage) {
        if (result == VK_SUCCESS) return true;
        if (trace) std::fprintf(stderr, "[compute]   Vulkan failure stage=%s result=%d\n",
                                stage, static_cast<int>(result));
        return false;
    };
    auto vk_handle_ok = [&](auto handle, const char* stage) {
        if (handle != VK_NULL_HANDLE) return true;
        if (trace) std::fprintf(stderr, "[compute]   Vulkan failure stage=%s result=null-handle\n", stage);
        return false;
    };

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
            if (buffer.memory) ctx.release_memory(buffer.memory);
        }
        for (size_t i = 0; i < images.size(); i++) {
            if (images[i].alias_of != SIZE_MAX) continue;
            if (images[i].sampler) vkDestroySampler(ctx.device, images[i].sampler, nullptr);
            if (images[i].view) vkDestroyImageView(ctx.device, images[i].view, nullptr);
            if (images[i].image) vkDestroyImage(ctx.device, images[i].image, nullptr);
            if (images[i].memory) ctx.release_memory(images[i].memory);
            if (staging[i]) vkDestroyBuffer(ctx.device, staging[i], nullptr);
            if (staging_memory[i]) ctx.release_memory(staging_memory[i]);
        }
    };
    auto resource_bytes = [](const ShaderResource* resource) -> uint8_t* {
        if (resource->host_data && resource->host_data_size >= resource->size)
            return resource->host_data;
        return reinterpret_cast<uint8_t*>(uintptr_t(resource->gpu_addr));
    };
    auto resource_bytes_for = [](const ShaderResource* resource, size_t required) -> uint8_t* {
        if (resource->host_data && resource->host_data_size >= required)
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
            buffers[i].memory = ctx.allocate_memory(requirements.size, memory_type);
            if (!buffers[i].memory) break;
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

        // --- Image bindings (#590): sampled textures use RGBA8 unless integer/packed-float semantics
        // require a native view; storage images are R32G32B32A32_UINT raw-channel texels (the recompiler's
        // format-free contract, exec-diff proven by tests/image_compute_runner.h) with per-format
        // pack/unpack against the guest surface. Everything not provably correct skips LOUDLY. ---
        bool images_ready = !ctx.device ? false : true;
        auto device_memory_type = [&](uint32_t bits) -> uint32_t {
            for (uint32_t i = 0; i < ctx.memory.memoryTypeCount; i++)
                if ((bits & (1u << i)) &&
                    (ctx.memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                    return i;
            for (uint32_t i = 0; i < ctx.memory.memoryTypeCount; i++)
                if (bits & (1u << i)) return i;
            return UINT32_MAX;
        };
        auto skip_image = [&](const prosper::gpu::ShaderResource* r, const char* why) {
            static std::vector<uint64_t> warned;
            const uint64_t key = r ? r->gpu_addr : 0;
            if (std::find(warned.begin(), warned.end(), key) == warned.end()) {
                warned.push_back(key);
                std::fprintf(stderr, "[compute] program 0x%llx image 0x%llx %ux%ux%u fmt=%u: %s -> "
                                     "dispatch skipped (#590)\n",
                             (unsigned long long)item.code_addr, (unsigned long long)key,
                             r ? r->width : 0, r ? r->height : 0, r ? r->depth : 0,
                             r ? (unsigned)r->format : 0u, why);
            }
            images_ready = false;
        };
        for (size_t i = 0; i < image_descriptors.size() && images_ready; i++) {
            const ShaderResource* r = item.resources->by_binding(image_descriptors[i].binding);
            if (!r || !r->width || !r->height) { skip_image(r, "no/degenerate resource"); break; }
            BoundImage& bi = images[i];
            bi.resource = r;
            bi.binding = image_descriptors[i].binding;
            bi.storage = image_descriptors[i].kind == SpirvDescriptorKind::StorageImage;
            // A surface whose CURRENT pixels live in the renderer's RTT cache must not be read from
            // raw guest memory (empty/stale — the Dead Cells 642x362 lesson).
            if (is_live_render_target(r->gpu_addr)) { skip_image(r, "renderer-owned RTT surface"); break; }
            const bool dim_1d = r->img_dim == 0;
            const bool dim_3d = r->img_dim == 2;
            if (!dim_1d && r->img_dim != 1 && !dim_3d) {
                skip_image(r, "layered image deferred to #657"); break;
            }
            if (dim_1d && r->height != 1) { skip_image(r, "1D image has non-unit height"); break; }
            if (!r->depth || (!dim_3d && r->depth != 1)) {
                skip_image(r, "image depth does not match its dimensionality"); break;
            }
            if (dim_3d && r->depth > 1 && r->tile_mode &&
                !tile_mode_supports_volume(r->tile_mode)) {
                skip_image(r, "3D tile mode has no volume address pattern"); break;
            }
            // Multiple U# bindings may intentionally name the same guest surface (read/modify/write
            // views of one UE4 volume). They must see one Vulkan image and produce one guest
            // writeback; independent images let an unwritten alias clobber the written result.
            if (bi.storage) {
                for (size_t j = 0; j < i; j++) {
                    const BoundImage& prior = images[j];
                    const ShaderResource* p = prior.resource;
                    if (!prior.storage || !p || p->gpu_addr != r->gpu_addr ||
                        p->width != r->width || p->height != r->height || p->depth != r->depth ||
                        p->format != r->format || p->num_components != r->num_components ||
                        p->tile_mode != r->tile_mode || p->img_dim != r->img_dim)
                        continue;
                    bi.alias_of = prior.alias_of == SIZE_MAX ? j : prior.alias_of;
                    const BoundImage& owner = images[bi.alias_of];
                    bi.image = owner.image; bi.memory = owner.memory; bi.view = owner.view;
                    bi.guest_bytes = owner.guest_bytes;
                    staging_bytes[i] = staging_bytes[bi.alias_of];
                    if (trace)
                        std::fprintf(stderr, "[compute]   image-alias binding=%u -> binding=%u addr=0x%llx\n",
                                     bi.binding, owner.binding, (unsigned long long)r->gpu_addr);
                    break;
                }
                if (bi.alias_of != SIZE_MAX) continue;
            }
            const VkDeviceSize volume_texels = static_cast<VkDeviceSize>(r->width) * r->height * r->depth;
            const uint32_t texel_bytes = bi.storage ? 16u : 4u;   // uvec4 raw channels / RGBA8
            const VkDeviceSize sbytes = volume_texels * texel_bytes;
            if (!sbytes || sbytes > kMaxComputeImageBytes) {
                skip_image(r, "expanded image exceeds the 256 MiB backend bound"); break;
            }
            staging_bytes[i] = sbytes;

            // Fill the upload staging bytes.
            std::vector<uint8_t> upload((size_t)sbytes, 0);
            if (bi.storage) {
                if (r->tile_mode && !dim_3d) {
                    skip_image(r, "tiled 1D/2D storage writeback deferred"); break;
                }
                if (!storage_unpack_supported(r->format) || !storage_pack_supported(r->format)) {
                    skip_image(r, "storage format has no channel pack/unpack yet"); break; }
                const uint32_t cb = data_format_bytes(r->format);
                const uint32_t nc = r->num_components ? r->num_components : 1;
                const size_t guest_texel = (size_t)cb * nc;
                const size_t texels = (size_t)volume_texels;
                const uint64_t linear_guest_bytes = static_cast<uint64_t>(texels) * guest_texel;
                const size_t guest_bytes = r->tile_mode
                    ? (r->depth > 1
                           ? tiled_volume_bytes(r->width, r->height, r->depth, r->tile_mode,
                                                static_cast<uint32_t>(guest_texel))
                           : tiled_surface_bytes(r->width, r->height, r->tile_mode, 0,
                                                 static_cast<uint32_t>(guest_texel)))
                    : static_cast<size_t>(linear_guest_bytes);
                if (!linear_guest_bytes || linear_guest_bytes > SIZE_MAX || !guest_bytes ||
                    guest_bytes > UINT32_MAX || (!r->tile_mode && guest_bytes > r->size)) {
                    skip_image(r, "storage backing size is invalid"); break;
                }
                bi.guest_bytes = guest_bytes;
                const uint8_t* src = resource_bytes_for(r, guest_bytes);
                const bool readable = (r->host_data && r->host_data_size >= guest_bytes) ||
                                      guest_readable(r->gpu_addr, static_cast<uint32_t>(guest_bytes));
                if (!readable) { skip_image(r, "storage backing unreadable"); break; }
                if (trace) bi.before_hash = fnv1a(src, guest_bytes);
                std::vector<uint8_t> linear((size_t)linear_guest_bytes, 0);
                if (r->tile_mode && r->depth > 1) {
                    if (!detile_volume(linear.data(), src, guest_bytes, r->width, r->height,
                                       r->depth, r->tile_mode, static_cast<uint32_t>(guest_texel))) {
                        skip_image(r, "storage volume detile failed"); break;
                    }
                } else if (r->tile_mode) {
                    detile_surface(linear.data(), src, r->width, r->height, r->tile_mode, 0,
                                   static_cast<uint32_t>(guest_texel));
                } else {
                    std::memcpy(linear.data(), src, linear.size());
                }
                for (size_t t = 0; t < texels; t++)
                    storage_unpack_texel(linear.data() + t * guest_texel, r->format, nc,
                                         reinterpret_cast<uint32_t*>(upload.data()) + t * 4);
            } else {
                const uint32_t bpb = bc_block_bytes(r->format);
                const uint32_t cb = data_format_bytes(r->format);
                const uint32_t nc = r->num_components ? r->num_components : 1;
                const bool rgba8 = r->format == DataFormat::Unorm8 && nc == 4;
                const bool uint8 = r->format == DataFormat::Uint8 && nc == 4;
                const bool r8 = r->format == DataFormat::Unorm8 && nc == 1;
                const bool f16 = r->format == DataFormat::Float16 && nc >= 1 && nc <= 4;
                const bool r11g11b10 = r->format == DataFormat::Float10_11_11 && nc == 3;
                size_t need;                                       // guest bytes the decode reads
                if (bpb && dim_3d) {
                    skip_image(r, "block-compressed 3D texture deferred"); break;
                } else if (bpb) {
                    const uint32_t bw = (r->width + 3) / 4, bh = (r->height + 3) / 4;
                    need = r->tile_mode ? tiled_elements_bytes(bw, bh, bpb, r->tile_mode)
                                        : (size_t)bw * bh * bpb;
                } else if (rgba8 || uint8 || r8 || f16 || r11g11b10) {
                    const uint32_t bpt = r11g11b10 ? 4u : cb * nc;
                    need = r->tile_mode
                        ? (dim_3d && r->depth > 1
                               ? tiled_volume_bytes(r->width, r->height, r->depth,
                                                    r->tile_mode, bpt)
                                  : tiled_surface_bytes(r->width, r->height, r->tile_mode, 0, bpt))
                        : (size_t)volume_texels * bpt;
                } else { skip_image(r, "sampled format not decodable yet"); break; }
                if (!need || need > kMaxComputeImageBytes || need > UINT32_MAX) {
                    skip_image(r, "sampled backing exceeds the 256 MiB backend bound"); break;
                }
                bi.guest_bytes = need;
                const uint8_t* src = resource_bytes_for(r, need);
                const bool readable = (r->host_data && r->host_data_size >= need) ||
                                      guest_readable(r->gpu_addr, (uint32_t)need);
                if (!readable) { skip_image(r, "sampled surface unreadable"); break; }
                if (bpb) {                                          // BCn: (block-detile ->) decode
                    const uint32_t bw = (r->width + 3) / 4, bh = (r->height + 3) / 4;
                    std::vector<uint8_t> lin((size_t)bw * bh * bpb, 0);
                    if (r->tile_mode)
                        detile_elements(lin.data(), src, need, bw, bh, bpb, r->tile_mode);
                    else
                        std::memcpy(lin.data(), src, lin.size());
                    if (!bc_decode_surface(upload.data(), lin.data(), lin.size(),
                                           r->width, r->height, r->format)) {
                        skip_image(r, "BC decode unsupported"); break; }
                } else {
                    const uint32_t bpt = r11g11b10 ? 4u : cb * nc;
                    std::vector<uint8_t> lin((size_t)volume_texels * bpt, 0);
                    if (r->tile_mode && dim_3d && r->depth > 1) {
                        if (!detile_volume(lin.data(), src, need, r->width, r->height, r->depth,
                                           r->tile_mode, bpt)) {
                            skip_image(r, "sampled volume detile failed"); break;
                        }
                    } else if (r->tile_mode) {
                        detile_surface(lin.data(), src, r->width, r->height, r->tile_mode, 0, bpt);
                    } else {
                        std::memcpy(lin.data(), src, lin.size());
                    }
                    const size_t texels = (size_t)volume_texels;
                    if (rgba8 || uint8 || r11g11b10) {              // Native 4-byte sampled texels
                        std::memcpy(upload.data(), lin.data(), lin.size());
                    } else if (r8) {                                // R8: broadcast coverage to RGBA
                        for (size_t t = 0; t < texels; t++) {
                            const uint8_t v = lin[t];
                            upload[t * 4 + 0] = v; upload[t * 4 + 1] = v;
                            upload[t * 4 + 2] = v; upload[t * 4 + 3] = v;
                        }
                    } else {                                        // Float16: half -> UNORM8 + default fill
                        const uint16_t* hp = reinterpret_cast<const uint16_t*>(lin.data());
                        for (size_t t = 0; t < texels; t++) {
                            for (uint32_t c = 0; c < 4; c++) {
                                if (c >= nc) {
                                    upload[t * 4 + c] = c == 3 ? 255 : 0;
                                    continue;
                                }
                                float v = half_to_float(hp[t * nc + c]);
                                if (std::isnan(v)) v = 0.f;
                                v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                                upload[t * 4 + c] = (uint8_t)std::lround(v * 255.0f);
                            }
                        }
                    }
                }
            }

            // Staging buffer (host-visible) holding the upload bytes; reused for storage readback.
            VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            sci.size = sbytes;
            sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            if (!vk_ok(vkCreateBuffer(ctx.device, &sci, nullptr, &staging[i]),
                       "image-staging-buffer")) { images_ready = false; break; }
            VkMemoryRequirements sreq{};
            vkGetBufferMemoryRequirements(ctx.device, staging[i], &sreq);
            const uint32_t staging_memory_type = ctx.host_memory_type(sreq.memoryTypeBits);
            staging_memory[i] = ctx.allocate_memory(sreq.size, staging_memory_type);
            if (!vk_handle_ok(staging_memory[i], "image-staging-memory") ||
                !vk_ok(vkBindBufferMemory(ctx.device, staging[i], staging_memory[i], 0),
                       "image-staging-bind")) {
                images_ready = false; break; }
            { void* mapped = nullptr;
              if (!vk_ok(vkMapMemory(ctx.device, staging_memory[i], 0, sbytes, 0, &mapped),
                         "image-staging-map")) {
                  images_ready = false; break; }
              std::memcpy(mapped, upload.data(), (size_t)sbytes);
              vkUnmapMemory(ctx.device, staging_memory[i]); }

            // Device-local image.
            VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType = dim_1d ? VK_IMAGE_TYPE_1D : (dim_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D);
            const bool sampled_uint8 = !bi.storage && r->format == DataFormat::Uint8 &&
                                       (r->num_components ? r->num_components : 1) == 4;
            const bool sampled_r11g11b10 = !bi.storage &&
                r->format == DataFormat::Float10_11_11 &&
                (r->num_components ? r->num_components : 1) == 3;
            ici.format = bi.storage ? VK_FORMAT_R32G32B32A32_UINT
                                    : (sampled_uint8 ? VK_FORMAT_R8G8B8A8_UINT
                                       : sampled_r11g11b10 ? VK_FORMAT_B10G11R11_UFLOAT_PACK32
                                                           : VK_FORMAT_R8G8B8A8_UNORM);
            ici.extent = {r->width, r->height, r->depth};
            ici.mipLevels = 1;
            ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = (bi.storage ? (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                                    : VK_IMAGE_USAGE_SAMPLED_BIT) |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            if (!vk_ok(vkCreateImage(ctx.device, &ici, nullptr, &bi.image),
                       "image-create")) { images_ready = false; break; }
            VkMemoryRequirements ireq{};
            vkGetImageMemoryRequirements(ctx.device, bi.image, &ireq);
            const uint32_t image_memory_type = device_memory_type(ireq.memoryTypeBits);
            bi.memory = ctx.allocate_memory(ireq.size, image_memory_type);
            if (!vk_handle_ok(bi.memory, "image-memory") ||
                !vk_ok(vkBindImageMemory(ctx.device, bi.image, bi.memory, 0), "image-bind")) {
                images_ready = false; break; }
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = bi.image;
            vci.viewType = dim_1d ? VK_IMAGE_VIEW_TYPE_1D
                                  : (dim_3d ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D);
            vci.format = ici.format;
            if (!bi.storage) {
                // T# DST_SEL channel routing (SQ_SEL: 0=0, 1=1, 4=R, 5=G, 6=B, 7=A) — same mapping
                // the renderer applies on its sampled views.
                auto sel = [&](uint32_t s) {
                    switch (s) {
                        case 0: return VK_COMPONENT_SWIZZLE_ZERO;
                        case 1: return VK_COMPONENT_SWIZZLE_ONE;
                        case 4: return VK_COMPONENT_SWIZZLE_R;
                        case 5: return VK_COMPONENT_SWIZZLE_G;
                        case 6: return VK_COMPONENT_SWIZZLE_B;
                        case 7: return VK_COMPONENT_SWIZZLE_A;
                        default: return VK_COMPONENT_SWIZZLE_IDENTITY;
                    }
                };
                vci.components = {sel(r->swizzle[0]), sel(r->swizzle[1]),
                                  sel(r->swizzle[2]), sel(r->swizzle[3])};
            }
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, ici.arrayLayers};
            if (!vk_ok(vkCreateImageView(ctx.device, &vci, nullptr, &bi.view),
                       "image-view")) { images_ready = false; break; }
            if (!bi.storage) {
                // Sampler from the decoded S# (mag/min filter, SQ_TEX CLAMP wrap enums) — the same
                // fields the renderer honors; defaults reproduce LINEAR/clamp.
                auto wrap = [&](uint32_t m) {
                    switch (m) {
                        case 0: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
                        case 1: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                        case 6: case 7: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                        default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                    }
                };
                VkSamplerCreateInfo smci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
                // Integer sampled formats cannot be linearly filtered in Vulkan. DOLL's UINT8x4
                // volume is consumed by image_load (texel fetch), so the sampler is unused there;
                // nearest also preserves integer semantics if a shader samples such a descriptor.
                smci.magFilter = sampled_uint8 ? VK_FILTER_NEAREST
                                               : (r->mag_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
                smci.minFilter = sampled_uint8 ? VK_FILTER_NEAREST
                                               : (r->min_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
                smci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                smci.addressModeU = wrap(r->addr_uvw[0]);
                smci.addressModeV = wrap(r->addr_uvw[1]);
                smci.addressModeW = wrap(r->addr_uvw[2]);
                smci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
                if (!vk_ok(vkCreateSampler(ctx.device, &smci, nullptr, &bi.sampler), "image-sampler")) {
                    images_ready = false; break; }
            }
        }
        if (!images_ready) break;

        // Layout: the buffer bindings (filled above) + one entry per image binding (#590).
        for (size_t i = 0; i < images.size(); i++) {
            VkDescriptorSetLayoutBinding b{};
            b.binding = images[i].binding;
            b.descriptorType = images[i].storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                 : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            layout_bindings.push_back(b);
        }
        VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dlci.bindingCount = static_cast<uint32_t>(layout_bindings.size());
        dlci.pBindings = layout_bindings.data();
        if (!vk_ok(vkCreateDescriptorSetLayout(ctx.device, &dlci, nullptr, &descriptor_layout),
                   "descriptor-layout")) break;
        uint32_t sampled_count = 0, storage_image_count = 0;
        for (const auto& im : images) (im.storage ? storage_image_count : sampled_count)++;
        VkDescriptorPoolSize pool_sizes[3]; uint32_t pool_size_count = 0;
        if (!buffers.empty())
            pool_sizes[pool_size_count++] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                             static_cast<uint32_t>(buffers.size())};
        if (sampled_count)
            pool_sizes[pool_size_count++] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampled_count};
        if (storage_image_count)
            pool_sizes[pool_size_count++] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storage_image_count};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = pool_size_count;
        dpci.pPoolSizes = pool_sizes;
        if (!vk_ok(vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &descriptor_pool),
                   "descriptor-pool")) break;
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = descriptor_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &descriptor_layout;
        if (!vk_ok(vkAllocateDescriptorSets(ctx.device, &dsai, &descriptor_set),
                   "descriptor-set")) break;

        std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
        std::vector<VkDescriptorImageInfo> image_infos(images.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size() + images.size());
        for (size_t i = 0; i < buffers.size(); i++) {
            buffer_infos[i] = {buffers[i].buffer, 0, buffers[i].resource->size};
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = descriptors[i].binding;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[i];
        }
        for (size_t i = 0; i < images.size(); i++) {
            image_infos[i] = {images[i].sampler, images[i].view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet& w = writes[buffers.size() + i];
            w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = descriptor_set;
            w.dstBinding = images[i].binding;
            w.descriptorCount = 1;
            w.descriptorType = images[i].storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                 : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &image_infos[i];
        }
        vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = item.spirv.size() * sizeof(uint32_t);
        smci.pCode = item.spirv.data();
        if (!vk_ok(vkCreateShaderModule(ctx.device, &smci, nullptr, &shader), "shader-module")) break;
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &descriptor_layout;
        if (!vk_ok(vkCreatePipelineLayout(ctx.device, &plci, nullptr, &pipeline_layout),
                   "pipeline-layout")) break;
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = shader;
        cpci.stage.pName = "main";
        cpci.layout = pipeline_layout;
        if (!vk_ok(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline),
                   "compute-pipeline")) break;

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = ctx.queue_family;
        if (!vk_ok(vkCreateCommandPool(ctx.device, &pci, nullptr, &command_pool), "command-pool")) break;
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = command_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        if (!vk_ok(vkAllocateCommandBuffers(ctx.device, &cbai, &command), "command-buffer")) break;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (!vk_ok(vkBeginCommandBuffer(command, &begin), "command-begin")) break;
        // Upload every image: UNDEFINED -> TRANSFER_DST, copy the staged texels in, -> GENERAL.
        for (size_t i = 0; i < images.size(); i++) {
            const BoundImage& bi = images[i];
            if (bi.alias_of != SIZE_MAX) continue;
            const ShaderResource* r = bi.resource;
            VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_dst.srcAccessMask = 0;
            to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_dst.image = bi.image;
            to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_dst);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {r->width, r->height, r->depth};
            vkCmdCopyBufferToImage(command, staging[i], bi.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            VkImageMemoryBarrier to_general = to_dst;
            to_general.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_general.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                       (bi.storage ? VK_ACCESS_SHADER_WRITE_BIT : 0u);
            to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                                 1, &to_general);
        }
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                                0, 1, &descriptor_set, 0, nullptr);
        vkCmdDispatch(command, item.launch.groups_x, item.launch.groups_y, item.launch.groups_z);
        // Storage images: copy the written texels back into the staging buffer for guest writeback.
        for (size_t i = 0; i < images.size(); i++) {
            const BoundImage& bi = images[i];
            if (!bi.storage || bi.alias_of != SIZE_MAX) continue;
            const ShaderResource* r = bi.resource;
            VkImageMemoryBarrier to_src{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_src.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            to_src.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            to_src.srcQueueFamilyIndex = to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_src.image = bi.image;
            to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_src);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {r->width, r->height, r->depth};
            vkCmdCopyImageToBuffer(command, bi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   staging[i], 1, &region);
        }
        if (!vk_ok(vkEndCommandBuffer(command), "command-end")) break;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (!vk_ok(vkCreateFence(ctx.device, &fci, nullptr, &fence), "fence")) break;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        if (!vk_ok(vkQueueSubmit(ctx.queue, 1, &submit, fence), "queue-submit")) break;
        if (!vk_ok(vkWaitForFences(ctx.device, 1, &fence, VK_TRUE,
                                   30ull * 1000 * 1000 * 1000), "queue-wait")) break;

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
            notify_guest_gpu_write(buffer.resource->gpu_addr, buffer.resource->size);
            if (!buffer.resource->host_data && writer_provenance_enabled())
                record_guest_write(GuestWriterKind::ComputeBuffer,
                                   buffer.resource->gpu_addr, buffer.resource->size,
                                   item.submit_no, item.dispatch_index,
                                   item.command_order, item.code_addr);
        }
        if (!readback_ok) break;
        // Storage-image writeback (#590): pack the kernel's raw uvec4 channel texels back into the
        // guest surface's real format, then restore its linear or 3D tiled address layout and notify
        // the render side exactly like the buffer path.
        for (size_t i = 0; i < images.size() && readback_ok; i++) {
            BoundImage& bi = images[i];
            if (!bi.storage || bi.alias_of != SIZE_MAX) continue;
            const ShaderResource* r = bi.resource;
            void* mapped = nullptr;
            if (vkMapMemory(ctx.device, staging_memory[i], 0, staging_bytes[i], 0, &mapped) != VK_SUCCESS) {
                readback_ok = false;
                break;
            }
            const uint32_t cb = data_format_bytes(r->format);
            const uint32_t nc = r->num_components ? r->num_components : 1;
            const size_t guest_texel = (size_t)cb * nc;
            const size_t texels = (size_t)r->width * r->height * r->depth;
            std::vector<uint8_t> linear(texels * guest_texel, 0);
            const uint32_t* channels = static_cast<const uint32_t*>(mapped);
            if (trace) {
                for (size_t t = 0; t < texels; t++)
                    for (uint32_t c = 0; c < 4; c++)
                        bi.nonzero_channels += channels[t * 4 + c] != 0;
            }
            for (size_t t = 0; t < texels; t++)
                storage_pack_texel(channels + t * 4, r->format, nc,
                                   linear.data() + t * guest_texel);
            uint8_t* destination = resource_bytes_for(r, bi.guest_bytes);
            if (r->tile_mode && r->depth > 1) {
                if (!tile_volume(destination, bi.guest_bytes, linear.data(), r->width, r->height,
                                 r->depth, r->tile_mode, static_cast<uint32_t>(guest_texel))) {
                    readback_ok = false;
                    vkUnmapMemory(ctx.device, staging_memory[i]);
                    break;
                }
            } else if (r->tile_mode) {
                tile_surface(destination, linear.data(), r->width, r->height, r->tile_mode, 0,
                             static_cast<uint32_t>(guest_texel));
            } else {
                std::memcpy(destination, linear.data(), linear.size());
            }
            if (trace) bi.after_hash = fnv1a(destination, bi.guest_bytes);
            notify_guest_gpu_write(r->gpu_addr, bi.guest_bytes);
            if (!r->host_data && writer_provenance_enabled())
                record_guest_write(GuestWriterKind::ComputeBuffer,
                                   r->gpu_addr, bi.guest_bytes,
                                   item.submit_no, item.dispatch_index,
                                   item.command_order, item.code_addr);
            vkUnmapMemory(ctx.device, staging_memory[i]);
        }
        if (!readback_ok) break;
        ok = true;
    } while (false);

    if (trace)
        std::fprintf(stderr, "[compute] execute submit=%llu dispatch=%llu code=0x%llx "
                     "threads=%ux%ux%u local=%ux%ux%u groups=%ux%ux%u "
                     "buffers=%zu images=%zu result=%s\n",
                     (unsigned long long)item.submit_no, (unsigned long long)item.dispatch_index,
                     (unsigned long long)item.code_addr, item.launch.threads_x,
                     item.launch.threads_y, item.launch.threads_z, item.launch.local_x,
                     item.launch.local_y, item.launch.local_z, item.launch.groups_x,
                     item.launch.groups_y, item.launch.groups_z, buffers.size(), images.size(),
                     ok ? "ok" : "failed");
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
        for (const auto& image : images) {
            if (!image.storage || !image.resource || image.alias_of != SIZE_MAX) continue;
            const uint8_t* bytes = resource_bytes_for(image.resource, image.guest_bytes);
            std::fprintf(stderr,
                         "[compute]   image-writeback binding=%u addr=0x%llx size=%zu "
                         "hash=%016llx->%016llx nonzero-ch=%llu "
                         "first=%08x,%08x,%08x,%08x\n",
                         image.binding, (unsigned long long)image.resource->gpu_addr,
                         image.guest_bytes, (unsigned long long)image.before_hash,
                         (unsigned long long)image.after_hash,
                         (unsigned long long)image.nonzero_channels,
                         image.guest_bytes >= 4 ? reinterpret_cast<const uint32_t*>(bytes)[0] : 0,
                         image.guest_bytes >= 8 ? reinterpret_cast<const uint32_t*>(bytes)[1] : 0,
                         image.guest_bytes >= 12 ? reinterpret_cast<const uint32_t*>(bytes)[2] : 0,
                         image.guest_bytes >= 16 ? reinterpret_cast<const uint32_t*>(bytes)[3] : 0);
        }
    }
    cleanup();
    return ok;
}

} // namespace

bool execute_live_compute_items(const std::vector<prosper::gpu::ComputeItem>& items) {
    // Keep the Vulkan device alive across dispatch spans. Constructing an instance, device, and queue for
    // every callback cost roughly 25 ms/frame on the native Windows frontend before any kernel work ran.
    // Function-local static initialization is thread-safe; AGC submit execution serializes subsequent use.
    static VulkanComputeContext context;
    static const bool context_ready = context.init();
    if (!context_ready) {
        std::fprintf(stderr, "[compute] Vulkan initialization failed\n");
        return false;
    }
    const bool timing_enabled = std::getenv("PROSPER_RENDER_TIMING") != nullptr;
    const auto timing_start = timing_enabled
        ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // Dispatches are independent PM4-order operations: one item failing (e.g. an image shape the
    // backend can't bind yet, #590) must not abort the rest of the batch — that would regress
    // dispatches that executed before image bindings existed. Run all; report all-succeeded.
    bool all_ok = true;
    for (const auto& item : items)
        all_ok &= execute_item(context, item);
    if (timing_enabled) {
        struct TimingTotals { uint64_t calls = 0, dispatches = 0; double milliseconds = 0; };
        static TimingTotals totals;
        static TimingTotals window;
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - timing_start).count();
        auto accumulate = [&](TimingTotals& timing) {
            timing.calls++;
            timing.dispatches += items.size();
            timing.milliseconds += elapsed;
        };
        accumulate(totals);
        accumulate(window);
        if (totals.calls % 25 == 0) {
            std::fprintf(stderr,
                         "[render-timing] compute calls=%llu dispatches=%llu avg_ms=%.2f\n",
                         (unsigned long long)totals.calls,
                         (unsigned long long)totals.dispatches,
                         totals.milliseconds / static_cast<double>(totals.calls));
            const ComputeMemoryPoolStats pool = context.memory_pool_stats();
            std::fprintf(stderr,
                         "[render-timing] compute_memory_pool hits=%llu misses=%llu cached=%zu "
                         "%.1f MiB discarded=%llu\n",
                         (unsigned long long)pool.hits, (unsigned long long)pool.misses,
                         pool.cached_allocations,
                         static_cast<double>(pool.cached_bytes) / (1024.0 * 1024.0),
                         (unsigned long long)pool.discarded);
            std::fprintf(stderr,
                         "[render-window] compute calls=%llu dispatches=%.1f avg_ms=%.2f\n",
                         (unsigned long long)window.calls,
                         window.dispatches / static_cast<double>(window.calls),
                         window.milliseconds / static_cast<double>(window.calls));
            window = {};
        }
    }
    return all_ok;
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
