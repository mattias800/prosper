#include "live_compute.hpp"
#include "vulkan_device_select.hpp"

#include "gpu/bc_decode.hpp"
#include "gpu/gpu_capture.hpp"
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
#include <string>
#include <unordered_map>
#include <vector>

namespace prosper::frontend {

uint8_t storage_pack_unorm8(uint32_t float_bits) {
    float value;
    std::memcpy(&value, &float_bits, sizeof(value));
    if (!(value > 0.0f)) return 0; // Includes negative values and NaN.
    if (value >= 1.0f) return 255;
    const float scaled = value * 255.0f;
    const uint32_t whole = static_cast<uint32_t>(scaled);
    return static_cast<uint8_t>(whole + (scaled - static_cast<float>(whole) >= 0.5f));
}

namespace {

constexpr VkDeviceSize kMaxComputeImageBytes = 256ull << 20;
constexpr size_t kMaxCachedComputePipelines = 4096;

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

struct CachedComputePipeline {
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
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
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    std::unordered_map<std::string, CachedComputePipeline> pipelines;
    uint32_t queue_family = UINT32_MAX;
    VkPhysicalDeviceMemoryProperties memory{};
    ComputeMemoryPool memory_pool;
    // Storage-image support (#590): the recompiler's storage path declares the
    // StorageImageRead/WriteWithoutFormat capabilities (raw uvec4 texel model — see
    // tests/image_compute_runner.h, the exec-diff harness for that contract). When the device lacks
    // the features, image-binding dispatches are skipped loudly instead of creating an invalid device.
    bool image_support = false;
    // True when instance/device/queue were ADOPTED from the live renderer (#1091). A borrowed
    // context is owned by the renderer: destroy our own pipelines/pools/memory, never its device.
    bool borrowed = false;

    ~VulkanComputeContext() {
        release_cached_memory();
        for (const auto& [key, cached] : pipelines) {
            (void)key;
            if (cached.pipeline) vkDestroyPipeline(device, cached.pipeline, nullptr);
            if (cached.pipeline_layout)
                vkDestroyPipelineLayout(device, cached.pipeline_layout, nullptr);
            if (cached.shader) vkDestroyShaderModule(device, cached.shader, nullptr);
            if (cached.descriptor_layout)
                vkDestroyDescriptorSetLayout(device, cached.descriptor_layout, nullptr);
        }
        if (pipeline_cache) vkDestroyPipelineCache(device, pipeline_cache, nullptr);
        if (borrowed) return;                       // renderer owns the device/instance
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

    // A GRAPHICS queue family is not required by spec to also advertise COMPUTE, and the renderer
    // selects its family on GRAPHICS alone. Dispatching on a family without COMPUTE is invalid usage,
    // so verify before adopting rather than assuming the common family-0 layout.
    static bool queue_family_supports_compute(VkPhysicalDevice phys, uint32_t family) {
        if (!phys || family == UINT32_MAX) return false;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
        if (family >= count) return false;
        std::vector<VkQueueFamilyProperties> props(count);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, props.data());
        return (props[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
    }

    bool init() {
        // Adopt the live renderer's device when it published one (#1091). Sharing a device is what
        // makes it possible for a dispatch to bind a renderer-owned image at all; without it every
        // such binding must round-trip through host memory. Declined when the renderer's device
        // lacks the storage-image features this backend needs, and absent entirely in headless
        // compute-only use (tests/test_game_compute.cpp), where the private device below is created
        // exactly as before.
        const prosper::gpu::SharedVulkanContext shared = prosper::gpu::shared_vulkan_context();
        if (shared.valid() && shared.storage_image_read_without_format &&
            shared.storage_image_write_without_format &&
            queue_family_supports_compute(static_cast<VkPhysicalDevice>(shared.physical),
                                          shared.queue_family)) {
            instance = static_cast<VkInstance>(shared.instance);
            physical = static_cast<VkPhysicalDevice>(shared.physical);
            device = static_cast<VkDevice>(shared.device);
            queue = static_cast<VkQueue>(shared.queue);
            queue_family = shared.queue_family;
            borrowed = true;
            image_support = true;
            VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
            if (vkCreatePipelineCache(device, &pcci, nullptr, &pipeline_cache) == VK_SUCCESS) {
                vkGetPhysicalDeviceMemoryProperties(physical, &memory);
                std::fprintf(stderr, "[compute] Vulkan device: adopted the renderer's device "
                                     "(shared, queue family %u)\n", queue_family);
                return true;
            }
            // Anything failing here must fall through to a private device rather than killing the
            // whole compute backend for the run: adoption is an optimization, never a requirement.
            // Release only what we created (nothing yet: the pipeline cache is what failed) and drop
            // the borrowed handles so the private path below starts from a clean context.
            instance = VK_NULL_HANDLE; physical = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE; queue = VK_NULL_HANDLE;
            queue_family = UINT32_MAX; borrowed = false; image_support = false;
            pipeline_cache = VK_NULL_HANDLE;
        }
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
        const auto selection = select_vulkan_device(devices, VK_QUEUE_COMPUTE_BIT);
        physical = selection.device;
        queue_family = selection.queue_family;
        if (!physical || queue_family == UINT32_MAX) return false;
        std::fprintf(stderr, "[compute] Vulkan device: %s (%s)\n",
                     selection.properties.deviceName,
                     vulkan_device_type_name(selection.properties.deviceType));

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
        VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        if (vkCreatePipelineCache(device, &pcci, nullptr, &pipeline_cache) != VK_SUCCESS)
            return false;
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        return true;
    }

    uint32_t host_memory_type(uint32_t bits) const {
        const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const VkMemoryPropertyFlags cached = wanted | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        for (uint32_t i = 0; i < memory.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & cached) == cached)
                return i;
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
    size_t alias_of = SIZE_MAX;         // exact guest range sharing an earlier storage buffer
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
    size_t alias_of = SIZE_MAX;         // exact sampled/storage binding sharing an earlier image/view
    uint8_t* dcc_metadata = nullptr;    // DCC control bytes to mark uncompressed after writeback
    size_t dcc_metadata_bytes = 0;
    // Borrowed renderer-owned image bound in place (#1095). `image` is then owned by the live
    // renderer: it must not be destroyed here, its layout must be restored, and the pin taken at
    // import time must be released.
    bool imported = false;
    bool seed_skip = false;             // #1122: write-only full-coverage storage image; no seed needed
    bool poison_verify = false;         // #1122: PROSPER_VERIFY_SEED_SKIP -- seed poison, prove full coverage
    uint64_t imported_addr = 0;
    uint32_t imported_saved_layout = 0;      // VkImageLayout the renderer left the image in
    // Several bindings can borrow the SAME renderer image without being folded together, because
    // the import contract is looser than the alias contract (it ignores sampler state, T# size and
    // img_dim 1-vs-5). Exactly one of them must emit the layout transitions: a second barrier pair
    // would declare oldLayout=saved on an image already in GENERAL, which is an invalid transition a
    // driver may treat as a discard. Ownership is derived at the barrier loops rather than stored
    // here -- see imported_barrier_owner().
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
//   Uint8/Sint8/Uint16/Sint16 -> integer channel widen (sign-extend for Sint) <- truncate to width.
//     A UINT/SINT image_load returns the stored integer directly and image_store writes the low
//     N bits with no normalization or saturation (Vulkan integer-format store contract), so the
//     host move is a width-aware zero/sign extend on upload and a low-bit truncation on writeback.
//   Unorm2_10_10_10 -> per-field float(bits/max) bits <- clamp(bitcast float,0,1)*max rounded, packed
//     high-to-low A2/B10/G10/R10 (GFX10 IMG_FMT 50 layout, matching unorm2_10_10_10_to_rgba8).
// UE4's post-process color-grading writes its 3D LUT / exposure volumes as these formats (DOLL: a
// 32x32x32 Uint8/Unorm2_10_10_10 LUT + 1x1x1 exposure + a 16x16x16 Uint16 volume); skipping the
// dispatch left the tonemap sampling an unproduced volume -> a near-zero grade -> black title (#590).
// Missing channels read the hardware default (0,0,0,1.0f). Anything else is unsupported -> the caller
// skips the dispatch loudly (never a silent wrong-layout write — correctness-first).
bool storage_unpack_supported(prosper::gpu::DataFormat f) {
    using DF = prosper::gpu::DataFormat;
    return f == DF::Unorm8 || f == DF::Float16 || f == DF::Float32 || f == DF::Uint32 ||
           f == DF::Sint32 || f == DF::Float10_11_11 ||
           f == DF::Uint8 || f == DF::Sint8 || f == DF::Uint16 || f == DF::Sint16 ||
           f == DF::Unorm2_10_10_10;
}
bool storage_pack_supported(prosper::gpu::DataFormat f) {
    using DF = prosper::gpu::DataFormat;
    return f == DF::Unorm8 || f == DF::Float16 || f == DF::Float32 ||
           f == DF::Uint32 || f == DF::Sint32 || f == DF::Float10_11_11 ||
           f == DF::Uint8 || f == DF::Sint8 || f == DF::Uint16 || f == DF::Sint16 ||
           f == DF::Unorm2_10_10_10;
}
void storage_unpack_texel(const uint8_t* src, prosper::gpu::DataFormat f, uint32_t ncomp, uint32_t out[4]) {
    using DF = prosper::gpu::DataFormat;
    const uint32_t one_f32 = 0x3f800000u;                 // hardware default: missing channels = (0,0,0,1)
    // A missing alpha reads 1 — as the FLOAT 1.0 bits for float/unorm formats, but the INTEGER 1 for
    // UINT/SINT formats (an integer image_load returns raw integer channels, not normalized floats).
    const bool integer_fmt = f == DF::Uint8 || f == DF::Sint8 || f == DF::Uint16 ||
                             f == DF::Sint16 || f == DF::Uint32 || f == DF::Sint32;
    out[0] = out[1] = out[2] = 0; out[3] = integer_fmt ? 1u : one_f32;
    if (f == DF::Float10_11_11) {
        uint32_t packed = 0; std::memcpy(&packed, src, sizeof(packed));
        const float values[3] = { prosper::gpu::f11_to_float(static_cast<uint16_t>(packed)),
                                  prosper::gpu::f11_to_float(static_cast<uint16_t>(packed >> 11)),
                                  prosper::gpu::f10_to_float(static_cast<uint16_t>(packed >> 22)) };
        for (uint32_t c = 0; c < 3; ++c) std::memcpy(&out[c], &values[c], sizeof(values[c]));
        return;
    }
    if (f == DF::Unorm2_10_10_10) {
        uint32_t packed = 0; std::memcpy(&packed, src, sizeof(packed));
        const float values[4] = { ((packed >>  0) & 0x3ffu) / 1023.0f,
                                  ((packed >> 10) & 0x3ffu) / 1023.0f,
                                  ((packed >> 20) & 0x3ffu) / 1023.0f,
                                  ((packed >> 30) & 0x3u)   / 3.0f };
        for (uint32_t c = 0; c < 4; ++c) std::memcpy(&out[c], &values[c], sizeof(values[c]));
        return;
    }
    for (uint32_t c = 0; c < ncomp && c < 4; c++) {
        switch (f) {
            case DF::Unorm8: { float v = src[c] / 255.0f; std::memcpy(&out[c], &v, 4); break; }
            case DF::Float16: { float v = prosper::gpu::half_to_float(
                                    (uint16_t)(src[c * 2] | (src[c * 2 + 1] << 8)));
                                std::memcpy(&out[c], &v, 4); break; }
            // Integer formats carry the raw channel value; a UINT/SINT image_load reads the stored
            // integer directly (zero-extend for Uint, sign-extend for Sint). No normalization.
            case DF::Uint8:  out[c] = src[c]; break;
            case DF::Sint8:  out[c] = static_cast<uint32_t>(static_cast<int32_t>(
                                          static_cast<int8_t>(src[c]))); break;
            case DF::Uint16: out[c] = static_cast<uint32_t>(src[c * 2] | (src[c * 2 + 1] << 8)); break;
            case DF::Sint16: out[c] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(
                                          src[c * 2] | (src[c * 2 + 1] << 8)))); break;
            default: std::memcpy(&out[c], src + c * 4, 4); break;   // 32-bit raw
        }
    }
}
// Unpack `count` consecutive texels (source stride `src_stride`) into `count` RGBA32 quads.
//
// storage_unpack_texel re-derives the format for every texel AND re-enters a switch for every
// component, so a full-resolution storage image pays ~texels function calls plus ~4x texels switch
// dispatches. That dominated compute image setup (measured: ~56 ms per image-bearing dispatch on
// Blasphemous 2's menu, against 0.39 ms of actual GPU dispatch). This hoists the format dispatch out
// of the loop so each specialized path is a tight typed loop.
//
// Every specialized path is bit-identical to storage_unpack_texel by construction; formats without a
// specialization fall through to the per-texel helper, and PROSPER_VERIFY_UNPACK=1 checks the two
// against each other at runtime.
void storage_unpack_range(const uint8_t* src, size_t src_stride, prosper::gpu::DataFormat f,
                          uint32_t ncomp, size_t count, uint32_t* out) {
    using DF = prosper::gpu::DataFormat;
    const uint32_t one_f32 = 0x3f800000u;
    const bool integer_fmt = f == DF::Uint8 || f == DF::Sint8 || f == DF::Uint16 ||
                             f == DF::Sint16 || f == DF::Uint32 || f == DF::Sint32;
    const uint32_t alpha_default = integer_fmt ? 1u : one_f32;
    const uint32_t n = ncomp < 4u ? ncomp : 4u;
    auto defaults = [&](uint32_t* o) { o[0] = o[1] = o[2] = 0; o[3] = alpha_default; };
    switch (f) {
        case DF::Unorm8:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c) { float v = p[c] / 255.0f; std::memcpy(&o[c], &v, 4); }
            }
            return;
        case DF::Float16:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c) {
                    float v = prosper::gpu::half_to_float(
                        static_cast<uint16_t>(p[c * 2] | (p[c * 2 + 1] << 8)));
                    std::memcpy(&o[c], &v, 4);
                }
            }
            return;
        case DF::Uint8:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c) o[c] = p[c];
            }
            return;
        case DF::Sint8:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c)
                    o[c] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(p[c])));
            }
            return;
        case DF::Uint16:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c)
                    o[c] = static_cast<uint32_t>(p[c * 2] | (p[c * 2 + 1] << 8));
            }
            return;
        case DF::Sint16:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c)
                    o[c] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(
                        p[c * 2] | (p[c * 2 + 1] << 8))));
            }
            return;
        case DF::Float32: case DF::Uint32: case DF::Sint32:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c) std::memcpy(&o[c], p + c * 4, 4);
            }
            return;
        default:                                  // packed formats keep the general per-texel path
            for (size_t t = 0; t < count; ++t)
                storage_unpack_texel(src + t * src_stride, f, ncomp, out + t * 4);
            return;
    }
}
// Range-specialized pack (#1101): the writeback mirror of storage_unpack_range (#1092). Hoists the
// per-texel format dispatch out of the loop with per-format inner loops; packed formats keep the
// general per-texel path. Semantics are IDENTICAL to storage_pack_texel over the range -- asserted
// format-by-format by test_storage_pack_range.
void storage_pack_texel(const uint32_t in[4], prosper::gpu::DataFormat f, uint32_t ncomp, uint8_t* dst);
void storage_pack_range(const uint32_t* channels, prosper::gpu::DataFormat f, uint32_t ncomp,
                        size_t count, uint8_t* dst, size_t dst_stride) {
    using DF = prosper::gpu::DataFormat;
    const uint32_t n = ncomp < 4u ? ncomp : 4u;
    switch (f) {
        case DF::Unorm8:
            for (size_t t = 0; t < count; ++t) {
                const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                for (uint32_t c = 0; c < n; ++c) p[c] = storage_pack_unorm8(in[c]);
            }
            return;
        case DF::Float16:
            for (size_t t = 0; t < count; ++t) {
                const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                for (uint32_t c = 0; c < n; ++c) {
                    float v; std::memcpy(&v, &in[c], 4);
                    const uint16_t h = prosper::gpu::float_to_half(v);
                    p[c * 2] = static_cast<uint8_t>(h);
                    p[c * 2 + 1] = static_cast<uint8_t>(h >> 8);
                }
            }
            return;
        case DF::Uint8: case DF::Sint8:
            for (size_t t = 0; t < count; ++t) {
                const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                for (uint32_t c = 0; c < n; ++c) p[c] = static_cast<uint8_t>(in[c]);
            }
            return;
        case DF::Uint16: case DF::Sint16:
            for (size_t t = 0; t < count; ++t) {
                const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                for (uint32_t c = 0; c < n; ++c) {
                    p[c * 2] = static_cast<uint8_t>(in[c]);
                    p[c * 2 + 1] = static_cast<uint8_t>(in[c] >> 8);
                }
            }
            return;
        case DF::Float32: case DF::Uint32: case DF::Sint32:
            for (size_t t = 0; t < count; ++t) {
                const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                for (uint32_t c = 0; c < n; ++c) std::memcpy(p + c * 4, &in[c], 4);
            }
            return;
        default:                                  // packed formats keep the general per-texel path
            for (size_t t = 0; t < count; ++t)
                storage_pack_texel(channels + t * 4, f, ncomp, dst + t * dst_stride);
            return;
    }
}
void storage_pack_texel(const uint32_t in[4], prosper::gpu::DataFormat f, uint32_t ncomp, uint8_t* dst) {
    using DF = prosper::gpu::DataFormat;
    if (f == DF::Float10_11_11) {
        float values[3];
        for (uint32_t c = 0; c < 3; ++c) std::memcpy(&values[c], &in[c], sizeof(values[c]));
        const uint32_t packed = static_cast<uint32_t>(prosper::gpu::float_to_f11(values[0])) |
                                (static_cast<uint32_t>(prosper::gpu::float_to_f11(values[1])) << 11) |
                                (static_cast<uint32_t>(prosper::gpu::float_to_f10(values[2])) << 22);
        std::memcpy(dst, &packed, sizeof(packed));
        return;
    }
    if (f == DF::Unorm2_10_10_10) {
        auto q = [](const uint32_t bits, float scale) -> uint32_t {
            float v; std::memcpy(&v, &bits, 4);
            v = !(v > 0.0f) ? 0.0f : (v > 1.0f ? 1.0f : v);   // NaN and negatives clamp to 0
            return static_cast<uint32_t>(v * scale + 0.5f);
        };
        const uint32_t packed = (q(in[0], 1023.0f) & 0x3ffu)        |
                                ((q(in[1], 1023.0f) & 0x3ffu) << 10) |
                                ((q(in[2], 1023.0f) & 0x3ffu) << 20) |
                                ((q(in[3], 3.0f)    & 0x3u)   << 30);
        std::memcpy(dst, &packed, sizeof(packed));
        return;
    }
    for (uint32_t c = 0; c < ncomp && c < 4; c++) {
        switch (f) {
            case DF::Unorm8: dst[c] = storage_pack_unorm8(in[c]); break;
            case DF::Float16: { float v; std::memcpy(&v, &in[c], 4);
                                const uint16_t h = prosper::gpu::float_to_half(v);
                                dst[c * 2] = static_cast<uint8_t>(h);
                                dst[c * 2 + 1] = static_cast<uint8_t>(h >> 8); break; }
            // Integer image_store writes the low N bits with no saturation (mirrors the 32-bit raw
            // move truncated to the format width).
            case DF::Uint8: case DF::Sint8:
                dst[c] = static_cast<uint8_t>(in[c]); break;
            case DF::Uint16: case DF::Sint16:
                dst[c * 2] = static_cast<uint8_t>(in[c]);
                dst[c * 2 + 1] = static_cast<uint8_t>(in[c] >> 8); break;
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
    using ComputeClock = std::chrono::steady_clock;
    const auto phase_start = ComputeClock::now();
    auto phase_setup = phase_start;
    auto phase_pipeline = phase_start;
    auto phase_dispatch = phase_start;
    auto phase_writeback = phase_start;
    double pack_ms = 0.0;
    double layout_ms = 0.0;
    const bool trace = trace_compute_item(item);
    // #1122: a compute post-process that only image_stores its output (no image_load) never reads the
    // seed, so materializing the renderer RTT's prior pixels into it is wasted. Detect it cheaply: an
    // OpImageRead-free SPIR-V reads no storage image at all. Combined with full dispatch coverage of
    // the target, the seed (a GPU readback + rgba8->uvec4 expand + upload, ~20 ms/frame on Blasphemous
    // 2's full-screen composite) can be skipped entirely. PROSPER_NO_SKIP_SEED forces the seed.
    const bool shader_reads_no_image = [&] {
        if (std::getenv("PROSPER_NO_SKIP_SEED")) return false;
        const auto& w = item.spirv;
        for (size_t k = 5; k < w.size();) {
            const uint32_t wc = w[k] >> 16, op = w[k] & 0xffff;
            if (!wc) break;
            if (op == 98u /* OpImageRead */) return false;
            k += wc;
        }
        return true;
    }();
    const auto setup_validate_start = ComputeClock::now();
    auto report = validate_spirv_descriptor_interface(
        item.spirv, item.resources.get(), 0, SpirvShaderStage::Compute, false);
    if (!report.ok()) return false;
    double setup_validate_ms = std::chrono::duration<double, std::milli>(
        ComputeClock::now() - setup_validate_start).count();
    double setup_buffers_ms = 0.0;

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
    // The phase timer is also useful for buffer-only kernels. Astro Bot's heaviest dispatcher writes
    // buffers exclusively, so the historical storage-image gate hid the actual frame bottleneck.
    const bool phase_timing = std::getenv("PROSPER_COMPUTE_PHASE_TIMING") != nullptr;
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
    bool pipeline_cached = false;
    std::string pipeline_key;
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
        if (!pipeline_cached) {
            if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
            if (shader) vkDestroyShaderModule(ctx.device, shader, nullptr);
            if (descriptor_layout)
                vkDestroyDescriptorSetLayout(ctx.device, descriptor_layout, nullptr);
        }
        if (descriptor_pool) vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
        for (auto& buffer : buffers) {
            if (buffer.alias_of != SIZE_MAX) continue;
            if (buffer.buffer) vkDestroyBuffer(ctx.device, buffer.buffer, nullptr);
            if (buffer.memory) ctx.release_memory(buffer.memory);
        }
        for (size_t i = 0; i < images.size(); i++) {
            // A pin is taken per successful import, so it is released per import -- including for a
            // binding that a later alias check folded into an earlier one (#1095).
            if (images[i].imported) release_live_render_target_image(images[i].imported_addr);
            if (images[i].alias_of != SIZE_MAX) continue;
            if (images[i].sampler) vkDestroySampler(ctx.device, images[i].sampler, nullptr);
            if (images[i].view) vkDestroyImageView(ctx.device, images[i].view, nullptr);
            // An imported image belongs to the live renderer: release the pin, destroy nothing.
            if (images[i].image && !images[i].imported)
                vkDestroyImage(ctx.device, images[i].image, nullptr);
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
            for (size_t j = 0; j < i; ++j) {
                const ShaderResource* prior = buffers[j].resource;
                if (!prior || prior->gpu_addr != resource->gpu_addr ||
                    prior->size != resource->size ||
                    prior->host_data != resource->host_data ||
                    prior->host_data_size != resource->host_data_size)
                    continue;
                buffers[i].alias_of = buffers[j].alias_of == SIZE_MAX ? j : buffers[j].alias_of;
                const BoundBuffer& owner = buffers[buffers[i].alias_of];
                buffers[i].buffer = owner.buffer;
                buffers[i].memory = owner.memory;
                break;
            }
            if (buffers[i].alias_of == SIZE_MAX) {
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
                // Pooled host-visible allocations retain their previous contents. Compare them with
                // current guest memory before uploading: an exact match is safe to reuse even when a
                // different resource owned the allocation previously, while any guest CPU mutation
                // still takes the ordinary full-copy path.
                if (std::memcmp(mapped, source, resource->size) != 0)
                    std::memcpy(mapped, source, resource->size);
                vkUnmapMemory(ctx.device, buffers[i].memory);
            }

            layout_bindings[i].binding = descriptors[i].binding;
            layout_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            layout_bindings[i].descriptorCount = 1;
            layout_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        bool buffers_ready = true;
        for (const auto& buffer : buffers) buffers_ready &= buffer.resource && buffer.memory;
        if (!buffers_ready) break;
        setup_buffers_ms = std::chrono::duration<double, std::milli>(
            ComputeClock::now() - setup_validate_start).count() - setup_validate_ms;

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
                std::fprintf(stderr, "[compute] program 0x%llx image 0x%llx %ux%ux%u fmt=%u comps=%u "
                                     "tile=%u size=%u: %s -> "
                                     "dispatch skipped (#590)\n",
                             (unsigned long long)item.code_addr, (unsigned long long)key,
                             r ? r->width : 0, r ? r->height : 0, r ? r->depth : 0,
                             r ? (unsigned)r->format : 0u, r ? r->num_components : 0u,
                             r ? r->tile_mode : 0u, r ? r->size : 0u, why);
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
            const bool dim_1d = r->img_dim == 0;
            const bool dim_3d = r->img_dim == 2;
            const bool dim_2d_array = r->img_dim == 5 && r->depth_compare;
            // A SINGLE-LAYER 2D array (img_dim==5, depth==1, no depth-compare) is byte-identical to a
            // plain 2D image (one layer, same tiling), so it flows through the 2D path below unchanged.
            // The general multi-layer/array case stays deferred to #657. DOLL's post-process compute
            // declares its mask/LUT surfaces this way; skipping them left the tonemap sampling an empty
            // surface -> a zero mask -> black composite even though the scene renders (#319/#657).
            const bool dim_2d_single = r->img_dim == 5 && !r->depth_compare && r->depth == 1;
            if (!dim_1d && r->img_dim != 1 && !dim_3d && !dim_2d_array && !dim_2d_single) {
                skip_image(r, "layered image deferred to #657"); break;
            }
            if (dim_1d && r->height != 1) { skip_image(r, "1D image has non-unit height"); break; }
            if (!r->depth || (!dim_3d && !dim_2d_array && r->depth != 1)) {
                skip_image(r, "image depth does not match its dimensionality"); break;
            }
            if (dim_3d && r->depth > 1 && r->tile_mode &&
                !tile_mode_supports_volume(r->tile_mode)) {
                skip_image(r, "3D tile mode has no volume address pattern"); break;
            }
            // The backend writes an ordinary tiled base allocation, not hardware-compressed blocks.
            // A compressed U# therefore also needs writable DCC metadata so successful writeback can
            // publish the hardware's 0xff (uncompressed) state before a later sampled descriptor sees it.
            if (bi.storage && r->compression_enabled) {
                const uint64_t metadata_bytes = gpu_capture_dcc_metadata_footprint(*r);
                if (!r->metadata_addr || !metadata_bytes || metadata_bytes > SIZE_MAX ||
                    metadata_bytes > UINT32_MAX) {
                    skip_image(r, "DCC metadata extent is unsupported"); break;
                }
                bi.dcc_metadata_bytes = static_cast<size_t>(metadata_bytes);
                if (r->dcc_metadata_host_data) {
                    if (r->dcc_metadata_host_data_size < metadata_bytes) {
                        skip_image(r, "replay DCC metadata backing is truncated"); break;
                    }
                    bi.dcc_metadata = r->dcc_metadata_host_data;
                } else {
                    if (!guest_readable(r->metadata_addr, static_cast<uint32_t>(metadata_bytes))) {
                        skip_image(r, "live DCC metadata backing is unreadable"); break;
                    }
                    bi.dcc_metadata = reinterpret_cast<uint8_t*>(uintptr_t(r->metadata_addr));
                }
            }
            // A renderer-owned target's current pixels are not in raw guest memory. Import its
            // immutable CPU snapshot for sampled and storage bindings; a successful storage
            // writeback publishes ordinary guest bytes and invalidates the renderer-owned copy.
            LiveTargetSnapshot live_target;
            bool renderer_owned = !r->in_mip_tail && is_live_render_target(r->gpu_addr);
            // Bind the renderer's own image when it is the authoritative copy and this binding
            // matches it EXACTLY (#1095, phase 2 of #1091). Restricted to a sampled Unorm8x4 view of
            // an rgba8 target because that is the one shape whose host path is a plain memcpy into a
            // VK_FORMAT_R8G8B8A8_UNORM image -- the direct bind is then byte-identical, and it skips
            // a full readback plus a re-upload of those same pixels to the same device. Every other
            // shape (notably rgba16f, which the host path converts down to UNORM8) keeps that path.
            if (renderer_owned && !bi.storage && !dim_1d && !dim_3d && !dim_2d_array &&
                r->depth == 1 && !r->depth_compare && r->format == DataFormat::Unorm8 &&
                (r->num_components ? r->num_components : 1) == 4) {
                LiveTargetImageImport import;
                if (import_live_render_target_image(r->gpu_addr, import)) {
                    if (import.format == LiveTargetPixelFormat::Rgba8Unorm &&
                        import.width == r->width && import.height == r->height &&
                        import.device == static_cast<void*>(ctx.device)) {
                        bi.imported = true;
                        bi.imported_addr = r->gpu_addr;
                        bi.imported_saved_layout = import.layout;
                        bi.image = static_cast<VkImage>(import.image);
                        live_target.width = import.width;
                        live_target.height = import.height;
                        live_target.format = import.format;
                        if (trace)
                            std::fprintf(stderr,
                                         "[compute]   bound renderer RTT in place binding=%u "
                                         "addr=0x%llx extent=%ux%u format=rgba8\n",
                                         bi.binding, (unsigned long long)r->gpu_addr,
                                         r->width, r->height);
                    } else {
                        // Not our exact contract (a different device or a stale aliased view).
                        release_live_render_target_image(r->gpu_addr);
                    }
                }
            }
            // #1122: skip the seed entirely for a write-only storage image whose dispatch fully
            // covers the target (every texel is image_stored, so the prior content is never observed
            // -- not by the shader, and not by the writeback, which packs only written texels). One
            // thread per texel with global size >= extent is the standard full-screen post-process
            // pattern; require it per dimension so a partially-covered write still seeds correctly.
            const bool covers_extent =
                item.launch.threads_x >= r->width && item.launch.threads_y >= r->height &&
                item.launch.threads_z >= r->depth;
            static const bool verify_seed_skip = std::getenv("PROSPER_VERIFY_SEED_SKIP") != nullptr;
            if (bi.storage && shader_reads_no_image && covers_extent && verify_seed_skip) {
                // Prove coverage instead of skipping: seed the image with poison and confirm the
                // write-only shader overwrites every texel (no poison survives in the result).
                bi.poison_verify = true;
            } else if (bi.storage && shader_reads_no_image && covers_extent) {
                bi.seed_skip = true;   // guest-backed OR renderer-owned: a fully-covered write ignores the seed
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   seed-skip write-only storage binding=%u addr=0x%llx "
                                 "extent=%ux%u threads=%ux%ux%u renderer_owned=%d\n",
                                 bi.binding, (unsigned long long)r->gpu_addr, r->width, r->height,
                                 item.launch.threads_x, item.launch.threads_y, item.launch.threads_z,
                                 renderer_owned ? 1 : 0);
            }
            if (renderer_owned && !bi.imported && !bi.seed_skip) {
                if (dim_3d || r->depth != 1 ||
                    !read_live_render_target(r->gpu_addr, live_target) || !live_target.pixels) {
                    skip_image(r, "renderer-owned RTT has no readable snapshot"); break;
                }
                if (live_target.width != r->width || live_target.height != r->height) {
                    // Address-only ownership is insufficient when the guest reuses or aliases an
                    // allocation with another image view. Astro Bot renders a 960x540 R11G11B10
                    // target at a base, then dispatches a 1216x684 R32 Z_X view at the same base
                    // after changing dynamic resolution. That stale cache entry does not own the
                    // new view; import its ordinary guest backing and let successful writeback
                    // invalidate the overlapping renderer entry.
                    renderer_owned = false;
                    if (trace)
                        std::fprintf(stderr,
                                     "[compute]   renderer RTT view miss binding=%u addr=0x%llx "
                                     "cached=%ux%u requested=%ux%u -> guest backing\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     live_target.width, live_target.height, r->width, r->height);
                }
                if (renderer_owned) {
                    const uint64_t bpp = live_target.format == LiveTargetPixelFormat::Rgba16Float ? 8u : 4u;
                    const uint64_t texels = static_cast<uint64_t>(r->width) * r->height;
                    if (texels > UINT64_MAX / bpp) {
                        skip_image(r, "renderer-owned RTT snapshot size overflow"); break;
                    }
                    const uint64_t expected = texels * bpp;
                    if (expected != live_target.pixels->size()) {
                        skip_image(r, "renderer-owned RTT snapshot byte count mismatch"); break;
                    }
                }
                if (renderer_owned && bi.storage) {
                    const uint32_t nc = r->num_components ? r->num_components : 1;
                    const bool compatible =
                        (live_target.format == LiveTargetPixelFormat::Rgba8Unorm &&
                         r->format == DataFormat::Unorm8 && nc == 4) ||
                        (live_target.format == LiveTargetPixelFormat::Rgba16Float &&
                         r->format == DataFormat::Float16 && nc == 4);
                    if (!compatible) {
                        // The same allocation can carry another target view before this compute
                        // operation (Astro Bot uses R8G8 and RGBA16F views at one base). A snapshot
                        // in the wrong storage format is not current bytes for this view.
                        renderer_owned = false;
                        if (trace)
                            std::fprintf(stderr,
                                         "[compute]   renderer RTT format miss binding=%u "
                                         "addr=0x%llx cached=%s requested=f%u/c%u -> guest backing\n",
                                         bi.binding, (unsigned long long)r->gpu_addr,
                                         live_target.format == LiveTargetPixelFormat::Rgba16Float
                                             ? "rgba16f" : "rgba8",
                                         (unsigned)r->format, nc);
                    }
                }
            }
            // Exact descriptor aliases share one Vulkan image. Storage aliases must observe the same
            // read/modify/write state and produce one guest writeback; sampled aliases avoid repeatedly
            // detiling and uploading Astro Bot's full-resolution source through dozens of SRT slots.
            for (size_t j = 0; j < i; j++) {
                const BoundImage& prior = images[j];
                const ShaderResource* p = prior.resource;
                const bool same_dcc_identity = p &&
                    p->max_uncompressed_block_size == r->max_uncompressed_block_size &&
                    p->max_compressed_block_size == r->max_compressed_block_size &&
                    p->meta_pipe_aligned == r->meta_pipe_aligned &&
                    p->write_compress_enabled == r->write_compress_enabled &&
                    p->compression_enabled == r->compression_enabled &&
                    p->alpha_is_on_msb == r->alpha_is_on_msb &&
                    p->color_transform == r->color_transform &&
                    p->metadata_addr == r->metadata_addr &&
                    p->dcc_metadata_size == r->dcc_metadata_size &&
                    p->dcc_metadata_host_data == r->dcc_metadata_host_data &&
                    p->dcc_metadata_host_data_size == r->dcc_metadata_host_data_size;
                const bool same_view = p && prior.storage == bi.storage &&
                    p->gpu_addr == r->gpu_addr && p->size == r->size &&
                    p->width == r->width && p->height == r->height && p->depth == r->depth &&
                    p->format == r->format && p->num_components == r->num_components &&
                    p->tile_mode == r->tile_mode && p->img_dim == r->img_dim &&
                    p->srgb == r->srgb && p->host_data == r->host_data &&
                    p->host_data_size == r->host_data_size && same_dcc_identity;
                bool same_sampler = true;
                if (!bi.storage && same_view) {
                    same_sampler = p->mag_filter == r->mag_filter &&
                        p->min_filter == r->min_filter && p->mip_filter == r->mip_filter &&
                        p->addr_uvw[0] == r->addr_uvw[0] && p->addr_uvw[1] == r->addr_uvw[1] &&
                        p->addr_uvw[2] == r->addr_uvw[2] &&
                        p->border_color_type == r->border_color_type &&
                        p->min_lod == r->min_lod && p->max_lod == r->max_lod &&
                        p->lod_bias == r->lod_bias && p->max_aniso_ratio == r->max_aniso_ratio &&
                        p->depth_compare_func == r->depth_compare_func &&
                        p->depth_compare == r->depth_compare && p->unnormalized == r->unnormalized &&
                        p->swizzle[0] == r->swizzle[0] && p->swizzle[1] == r->swizzle[1] &&
                        p->swizzle[2] == r->swizzle[2] && p->swizzle[3] == r->swizzle[3];
                }
                if (!same_view || !same_sampler) continue;
                bi.alias_of = prior.alias_of == SIZE_MAX ? j : prior.alias_of;
                const BoundImage& owner = images[bi.alias_of];
                bi.image = owner.image; bi.memory = owner.memory; bi.view = owner.view;
                bi.sampler = owner.sampler; bi.guest_bytes = owner.guest_bytes;
                bi.dcc_metadata = owner.dcc_metadata;
                bi.dcc_metadata_bytes = owner.dcc_metadata_bytes;
                staging_bytes[i] = staging_bytes[bi.alias_of];
                if (trace)
                    std::fprintf(stderr, "[compute]   image-alias binding=%u -> binding=%u addr=0x%llx\n",
                                 bi.binding, owner.binding, (unsigned long long)r->gpu_addr);
                break;
            }
            if (bi.alias_of != SIZE_MAX) continue;
            const VkDeviceSize volume_texels = static_cast<VkDeviceSize>(r->width) * r->height * r->depth;
            const uint32_t sampled_components = r->num_components ? r->num_components : 1;
            const bool sampled_depth = !bi.storage && r->depth_compare && dim_2d_array &&
                                       sampled_components == 1 &&
                                       (r->format == DataFormat::Float32 ||
                                        r->format == DataFormat::Unorm16);
            if (r->depth_compare && !sampled_depth) {
                skip_image(r, "depth-compare image format/dimension unsupported"); break;
            }
            const bool sampled_float32 = !bi.storage && !sampled_depth &&
                                         r->format == DataFormat::Float32 &&
                                         sampled_components >= 1 && sampled_components <= 4;
            const bool sampled_unorm8x2 = !bi.storage && r->format == DataFormat::Unorm8 &&
                                          sampled_components == 2;
            const bool sampled_unorm16_native = !bi.storage && r->format == DataFormat::Unorm16 &&
                                                (sampled_components == 1 || sampled_components == 2 ||
                                                 sampled_components == 4);
            const bool sampled_uint8_native = !bi.storage && r->format == DataFormat::Uint8 &&
                                              (sampled_components == 1 || sampled_components == 2 ||
                                               sampled_components == 4);
            const uint32_t texel_bytes = bi.storage || sampled_float32 ? 16u
                                         : sampled_depth ? data_format_bytes(r->format)
                                         : sampled_unorm16_native ? sampled_components * 2u
                                         : sampled_uint8_native ? sampled_components
                                         : sampled_unorm8x2 ? 2u : 4u;
            // Storage images use raw uvec4 channels. Most sampled formats are normalized to RGBA8,
            // but Float32 stays native so exposure/HDR kernels do not lose sign or dynamic range.
            const VkDeviceSize sbytes = volume_texels * texel_bytes;
            if (!sbytes || sbytes > kMaxComputeImageBytes) {
                skip_image(r, "expanded image exceeds the 256 MiB backend bound"); break;
            }
            staging_bytes[i] = sbytes;

            // Fill the upload staging bytes.
            std::vector<uint8_t> upload((bi.imported || bi.seed_skip) ? size_t{0} : (size_t)sbytes, 0);
            if (bi.imported) {
                // The renderer's image is the source: there is nothing to convert or stage.
                bi.guest_bytes = 0;
            } else if (bi.storage) {
                if (r->tile_mode && !tile_mode_is_tiled(r->tile_mode)) {
                    skip_image(r, "storage tile mode has no supported address pattern"); break;
                }
                if (r->tile_mode && !dim_3d &&
                    std::getenv("PROSPER_DISABLE_COMPUTE_TILED_2D_STORAGE")) {
                    skip_image(r, "tiled 1D/2D storage writeback disabled"); break;
                }
                if (!storage_unpack_supported(r->format) || !storage_pack_supported(r->format)) {
                    skip_image(r, "storage format has no channel pack/unpack yet"); break; }
                const uint32_t cb = data_format_bytes(r->format);
                const uint32_t nc = r->num_components ? r->num_components : 1;
                const size_t guest_texel = (r->format == DataFormat::Float10_11_11 ||
                                            r->format == DataFormat::Unorm2_10_10_10)
                    ? 4u : (size_t)cb * nc;
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
                const uint8_t* src = (renderer_owned || bi.seed_skip)
                    ? nullptr : resource_bytes_for(r, guest_bytes);
                const bool readable = renderer_owned || bi.seed_skip ||
                                      (r->host_data && r->host_data_size >= guest_bytes) ||
                                      guest_readable(r->gpu_addr, static_cast<uint32_t>(guest_bytes));
                if (!readable) { skip_image(r, "storage backing unreadable"); break; }
                std::vector<uint8_t> linear(bi.seed_skip ? size_t{0}
                                            : (size_t)linear_guest_bytes, 0);
                if (bi.seed_skip) {
                    // #1122: write-only full-coverage target -- the shader overwrites every texel, so
                    // the image is created but never seeded, uploaded, or read. Nothing to fill.
                } else if (renderer_owned) {
                    std::memcpy(linear.data(), live_target.pixels->data(), linear.size());
                    if (trace) {
                        bi.before_hash = fnv1a(linear.data(), linear.size());
                        std::fprintf(stderr,
                                     "[compute]   imported writable renderer RTT binding=%u "
                                     "addr=0x%llx extent=%ux%u format=%s\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     r->width, r->height,
                                     live_target.format == LiveTargetPixelFormat::Rgba16Float
                                         ? "rgba16f" : "rgba8");
                    }
                } else if (r->tile_mode && r->depth > 1) {
                    if (trace) bi.before_hash = fnv1a(src, guest_bytes);
                    if (!detile_volume(linear.data(), src, guest_bytes, r->width, r->height,
                                       r->depth, r->tile_mode, static_cast<uint32_t>(guest_texel))) {
                        skip_image(r, "storage volume detile failed"); break;
                    }
                } else if (r->tile_mode && r->in_mip_tail) {
                    if (trace) bi.before_hash = fnv1a(src, guest_bytes);
                    detile_surface_level(linear.data(), src, guest_bytes,
                                         r->width, r->height, r->tile_mode,
                                         static_cast<uint32_t>(guest_texel),
                                         r->mip_tail_x, r->mip_tail_y);
                } else if (r->tile_mode) {
                    if (trace) bi.before_hash = fnv1a(src, guest_bytes);
                    detile_surface(linear.data(), src, r->width, r->height, r->tile_mode, 0,
                                   static_cast<uint32_t>(guest_texel));
                } else {
                    if (trace) bi.before_hash = fnv1a(src, guest_bytes);
                    std::memcpy(linear.data(), src, linear.size());
                }
                if (!bi.seed_skip)
                    storage_unpack_range(linear.data(), guest_texel, r->format, nc, texels,
                                         reinterpret_cast<uint32_t*>(upload.data()));
                if (bi.poison_verify) {   // #1122 coverage proof: poison every uvec4 channel
                    uint32_t* pp = reinterpret_cast<uint32_t*>(upload.data());
                    for (size_t t = 0; t < texels * 4; ++t) pp[t] = 0xDEADBEEFu;
                }
                static const bool verify_unpack =
                    std::getenv("PROSPER_VERIFY_UNPACK") != nullptr && !bi.seed_skip;
                if (verify_unpack) {
                    // Fail-visible A/B: the specialized range unpack must be bit-identical to the
                    // per-texel path it replaces. Reports the whole divergence (count + first texel)
                    // and identifies the binding, and logs the clean case too so a verified run is
                    // self-proving rather than merely silent.
                    uint32_t expect[4];
                    const uint32_t* got = reinterpret_cast<const uint32_t*>(upload.data());
                    size_t bad = 0, first_bad = 0;
                    for (size_t t = 0; t < texels; ++t) {
                        storage_unpack_texel(linear.data() + t * guest_texel, r->format, nc, expect);
                        if (std::memcmp(expect, got + t * 4, sizeof expect) != 0) {
                            if (!bad) first_bad = t;
                            ++bad;
                        }
                    }
                    std::fprintf(stderr,
                                 "[compute] unpack-verify binding=%u addr=0x%llx fmt=%u nc=%u "
                                 "texels=%zu mismatches=%zu%s\n",
                                 bi.binding, (unsigned long long)r->gpu_addr,
                                 (unsigned)r->format, nc, texels, bad,
                                 bad ? " MISMATCH" : "");
                    if (bad)
                        std::fprintf(stderr, "[compute]   first mismatching texel=%zu\n", first_bad);
                }
            } else {
                const uint32_t bpb = bc_block_bytes(r->format);
                const uint32_t cb = data_format_bytes(r->format);
                const uint32_t nc = r->num_components ? r->num_components : 1;
                const bool rgba8 = r->format == DataFormat::Unorm8 && nc == 4;
                const bool uint8 = r->format == DataFormat::Uint8 && nc == 4;
                const bool r8 = r->format == DataFormat::Unorm8 && nc == 1;
                const bool f16 = r->format == DataFormat::Float16 && nc >= 1 && nc <= 4;
                const bool f32 = sampled_float32;
                const bool r11g11b10 = r->format == DataFormat::Float10_11_11 && nc == 3;
                if (renderer_owned) {
                    if (r11g11b10) {
                        skip_image(r, "renderer RTT cannot be reconstructed as packed R11G11B10");
                        break;
                    }
                    const std::vector<uint8_t>& pixels = *live_target.pixels;
                    if (sampled_uint8_native) {
                        const size_t texels = static_cast<size_t>(volume_texels);
                        for (size_t t = 0; t < texels; ++t) {
                            for (uint32_t c = 0; c < sampled_components; ++c) {
                                if (live_target.format == LiveTargetPixelFormat::Rgba8Unorm) {
                                    upload[t * sampled_components + c] = pixels[t * 4 + c];
                                } else {
                                    uint16_t half = 0;
                                    std::memcpy(&half, pixels.data() + t * 8 + c * 2, sizeof(half));
                                    float value = half_to_float(half);
                                    if (!std::isfinite(value) || value <= 0.0f) value = 0.0f;
                                    else if (value >= 255.0f) value = 255.0f;
                                    upload[t * sampled_components + c] = static_cast<uint8_t>(
                                        std::lround(value));
                                }
                            }
                        }
                    } else if (sampled_unorm8x2) {
                        const size_t texels = static_cast<size_t>(volume_texels);
                        for (size_t t = 0; t < texels; ++t) {
                            for (uint32_t c = 0; c < 2; ++c) {
                                if (live_target.format == LiveTargetPixelFormat::Rgba8Unorm) {
                                    upload[t * 2 + c] = pixels[t * 4 + c];
                                } else {
                                    uint16_t half = 0;
                                    std::memcpy(&half, pixels.data() + t * 8 + c * 2, sizeof(half));
                                    float value = half_to_float(half);
                                    if (!std::isfinite(value) || value <= 0.0f) value = 0.0f;
                                    else if (value >= 1.0f) value = 1.0f;
                                    upload[t * 2 + c] = static_cast<uint8_t>(
                                        std::lround(value * 255.0f));
                                }
                            }
                        }
                    } else if (sampled_unorm16_native) {
                        const size_t texels = static_cast<size_t>(volume_texels);
                        for (size_t t = 0; t < texels; ++t) {
                            for (uint32_t c = 0; c < sampled_components; ++c) {
                                uint16_t value = 0;
                                if (live_target.format == LiveTargetPixelFormat::Rgba8Unorm) {
                                    value = static_cast<uint16_t>(pixels[t * 4 + c]) * 257u;
                                } else {
                                    uint16_t half = 0;
                                    std::memcpy(&half, pixels.data() + t * 8 + c * 2, sizeof(half));
                                    float f = half_to_float(half);
                                    if (!std::isfinite(f) || f <= 0.0f) f = 0.0f;
                                    else if (f >= 1.0f) f = 1.0f;
                                    value = static_cast<uint16_t>(std::lround(f * 65535.0f));
                                }
                                std::memcpy(upload.data() +
                                                (t * sampled_components + c) * sizeof(value),
                                            &value, sizeof(value));
                            }
                        }
                    } else if (f32) {
                        const size_t texels = static_cast<size_t>(volume_texels);
                        for (size_t t = 0; t < texels; ++t) {
                            for (uint32_t c = 0; c < 4; ++c) {
                                float value = 0.0f;
                                if (live_target.format == LiveTargetPixelFormat::Rgba8Unorm) {
                                    value = pixels[t * 4 + c] / 255.0f;
                                } else {
                                    uint16_t half = 0;
                                    std::memcpy(&half, pixels.data() + t * 8 + c * 2, sizeof(half));
                                    value = half_to_float(half);
                                    if (std::isnan(value)) value = 0.0f;
                                }
                                std::memcpy(upload.data() + (t * 4 + c) * sizeof(float),
                                            &value, sizeof(value));
                            }
                        }
                    } else if (live_target.format == LiveTargetPixelFormat::Rgba8Unorm) {
                        std::memcpy(upload.data(), pixels.data(), upload.size());
                    } else {
                        const size_t texels = static_cast<size_t>(volume_texels);
                        for (size_t t = 0; t < texels; ++t) {
                            for (uint32_t c = 0; c < 4; ++c) {
                                uint16_t half = 0;
                                std::memcpy(&half, pixels.data() + t * 8 + c * 2, sizeof(half));
                                float value = half_to_float(half);
                                if (!std::isfinite(value) || value <= 0.0f) value = 0.0f;
                                else if (value >= 1.0f) value = 1.0f;
                                upload[t * 4 + c] = static_cast<uint8_t>(
                                    std::lround(value * 255.0f));
                            }
                        }
                    }
                    if (trace) {
                        const uint64_t snapshot_hash = fnv1a(live_target.pixels->data(),
                                                             live_target.pixels->size());
                        const size_t nonzero_bytes = static_cast<size_t>(std::count_if(
                            live_target.pixels->begin(), live_target.pixels->end(),
                            [](uint8_t value) { return value != 0; }));
                        std::fprintf(stderr,
                                     "[compute]   imported renderer RTT binding=%u addr=0x%llx "
                                     "extent=%ux%u format=%s hash=%016llx nonzero-bytes=%zu\n",
                                     bi.binding, (unsigned long long)r->gpu_addr, r->width, r->height,
                                     live_target.format == LiveTargetPixelFormat::Rgba16Float
                                         ? "rgba16f" : "rgba8",
                                     (unsigned long long)snapshot_hash, nonzero_bytes);
                    }
                    bi.guest_bytes = 0;
                } else {
                    size_t need;                                   // guest bytes the decode reads
                    if (bpb && dim_3d) {
                        skip_image(r, "block-compressed 3D texture deferred"); break;
                    } else if (bpb) {
                        const uint32_t bw = (r->width + 3) / 4, bh = (r->height + 3) / 4;
                        const size_t slice = r->tile_mode
                            ? tiled_elements_bytes(bw, bh, bpb, r->tile_mode)
                            : (size_t)bw * bh * bpb;
                        need = dim_2d_array ? slice * r->depth : slice;
                    } else if (rgba8 || uint8 || r8 || f16 || f32 || r11g11b10 ||
                               sampled_unorm8x2 || sampled_unorm16_native || sampled_uint8_native ||
                               sampled_depth) {
                        const uint32_t bpt = r11g11b10 ? 4u : cb * nc;
                        if (r->tile_mode && dim_3d && r->depth > 1) {
                            need = tiled_volume_bytes(r->width, r->height, r->depth,
                                                      r->tile_mode, bpt);
                        } else if (r->tile_mode && dim_2d_array) {
                            need = tiled_surface_bytes(r->width, r->height,
                                                       r->tile_mode, 0, bpt) * r->depth;
                        } else {
                            need = r->tile_mode
                                ? tiled_surface_bytes(r->width, r->height, r->tile_mode, 0, bpt)
                                : (size_t)volume_texels * bpt;
                        }
                    } else { skip_image(r, "sampled format not decodable yet"); break; }
                    if (!need || need > kMaxComputeImageBytes || need > UINT32_MAX) {
                        skip_image(r, "sampled backing exceeds the 256 MiB backend bound"); break;
                    }
                    bi.guest_bytes = need;
                    const uint8_t* src = resource_bytes_for(r, need);
                    const bool readable = (r->host_data && r->host_data_size >= need) ||
                                          guest_readable(r->gpu_addr, (uint32_t)need);
                    if (!readable) { skip_image(r, "sampled surface unreadable"); break; }
                    if (bpb) {                                      // BCn: (block-detile ->) decode
                        const uint32_t bw = (r->width + 3) / 4, bh = (r->height + 3) / 4;
                        if (dim_2d_array) {
                            skip_image(r, "block-compressed sampled arrays deferred"); break;
                        }
                        std::vector<uint8_t> lin((size_t)bw * bh * bpb, 0);
                        if (r->tile_mode && r->in_mip_tail)
                            detile_elements_level(lin.data(), src, need, bw, bh, bpb,
                                                  r->tile_mode, r->mip_tail_x,
                                                  r->mip_tail_y);
                        else if (r->tile_mode)
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
                        } else if (r->tile_mode && dim_2d_array) {
                            const size_t tiled_slice = tiled_surface_bytes(
                                r->width, r->height, r->tile_mode, 0, bpt);
                            const size_t linear_slice = static_cast<size_t>(r->width) * r->height * bpt;
                            for (uint32_t layer = 0; layer < r->depth; ++layer)
                                detile_surface(lin.data() + linear_slice * layer,
                                               src + tiled_slice * layer,
                                               r->width, r->height, r->tile_mode, 0, bpt);
                        } else if (r->tile_mode && r->in_mip_tail) {
                            detile_surface_level(lin.data(), src, need, r->width, r->height,
                                                 r->tile_mode, bpt, r->mip_tail_x,
                                                 r->mip_tail_y);
                        } else if (r->tile_mode) {
                            detile_surface(lin.data(), src, r->width, r->height, r->tile_mode, 0, bpt);
                        } else {
                            std::memcpy(lin.data(), src, lin.size());
                        }
                        const size_t texels = (size_t)volume_texels;
                        if (rgba8 || uint8 || r11g11b10 ||
                            sampled_unorm8x2 || sampled_unorm16_native ||
                            sampled_uint8_native || sampled_depth) {   // Native sampled texels
                            std::memcpy(upload.data(), lin.data(), lin.size());
                        } else if (f32) {                           // Native float channels + default fill
                            for (size_t t = 0; t < texels; ++t) {
                                for (uint32_t c = 0; c < 4; ++c) {
                                    float value = c == 3 ? 1.0f : 0.0f;
                                    if (c < nc)
                                        std::memcpy(&value, lin.data() + (t * nc + c) * sizeof(float),
                                                    sizeof(value));
                                    std::memcpy(upload.data() + (t * 4 + c) * sizeof(float),
                                                &value, sizeof(value));
                                }
                            }
                        } else if (r8) {                            // R8: broadcast coverage to RGBA
                            for (size_t t = 0; t < texels; t++) {
                                const uint8_t v = lin[t];
                                upload[t * 4 + 0] = v; upload[t * 4 + 1] = v;
                                upload[t * 4 + 2] = v; upload[t * 4 + 3] = v;
                            }
                        } else {                                    // Float16: half -> UNORM8 + default fill
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
            }

            // Staging buffer (host-visible) holding the upload bytes; reused for storage
            // readback. An imported renderer image supplies its own pixels, so it needs none.
            if (!bi.imported) {
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
                if (!bi.seed_skip) {   // seed-skip (#1122): image is never uploaded, staging stays for writeback
                  void* mapped = nullptr;
                  if (!vk_ok(vkMapMemory(ctx.device, staging_memory[i], 0, sbytes, 0, &mapped),
                             "image-staging-map")) {
                      images_ready = false; break; }
                  std::memcpy(mapped, upload.data(), (size_t)sbytes);
                  vkUnmapMemory(ctx.device, staging_memory[i]); }
            }

            // Device-local image.
            const auto img_t1 = ComputeClock::now();
            VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType = dim_1d ? VK_IMAGE_TYPE_1D : (dim_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D);
            const bool sampled_r11g11b10 = !bi.storage &&
                r->format == DataFormat::Float10_11_11 &&
                (r->num_components ? r->num_components : 1) == 3;
            ici.format = bi.storage ? VK_FORMAT_R32G32B32A32_UINT
                                    : (sampled_depth
                                           ? (r->format == DataFormat::Float32
                                                  ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_D16_UNORM)
                                       : sampled_uint8_native
                                           ? (sampled_components == 1 ? VK_FORMAT_R8_UINT
                                              : sampled_components == 2 ? VK_FORMAT_R8G8_UINT
                                                                         : VK_FORMAT_R8G8B8A8_UINT)
                                       : sampled_r11g11b10 ? VK_FORMAT_B10G11R11_UFLOAT_PACK32
                                         : sampled_unorm8x2 ? VK_FORMAT_R8G8_UNORM
                                           : sampled_unorm16_native
                                               ? (sampled_components == 1 ? VK_FORMAT_R16_UNORM
                                                  : sampled_components == 2 ? VK_FORMAT_R16G16_UNORM
                                                                            : VK_FORMAT_R16G16B16A16_UNORM)
                                         : sampled_float32 ? VK_FORMAT_R32G32B32A32_SFLOAT
                                                           : VK_FORMAT_R8G8B8A8_UNORM);
            ici.extent = {r->width, r->height, r->depth};
            ici.mipLevels = 1;
            if (dim_2d_array) ici.extent.depth = 1;
            ici.arrayLayers = dim_2d_array ? r->depth : 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = (bi.storage ? (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                                    : VK_IMAGE_USAGE_SAMPLED_BIT) |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            // An imported binding already holds the renderer's image; only the view/sampler below
            // are ours to create. `ici` is still filled above so the view matches its format/layers.
            if (!bi.imported) {
                if (!vk_ok(vkCreateImage(ctx.device, &ici, nullptr, &bi.image),
                           "image-create")) { images_ready = false; break; }
                VkMemoryRequirements ireq{};
                vkGetImageMemoryRequirements(ctx.device, bi.image, &ireq);
                const uint32_t image_memory_type = device_memory_type(ireq.memoryTypeBits);
                bi.memory = ctx.allocate_memory(ireq.size, image_memory_type);
                if (!vk_handle_ok(bi.memory, "image-memory") ||
                    !vk_ok(vkBindImageMemory(ctx.device, bi.image, bi.memory, 0), "image-bind")) {
                    images_ready = false; break; }
            }
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = bi.image;
            vci.viewType = dim_1d ? VK_IMAGE_VIEW_TYPE_1D
                                  : (dim_3d ? VK_IMAGE_VIEW_TYPE_3D
                                     : dim_2d_array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                    : VK_IMAGE_VIEW_TYPE_2D);
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
            const VkImageAspectFlags image_aspect = sampled_depth
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange = {image_aspect, 0, 1, 0, ici.arrayLayers};
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
                smci.magFilter = sampled_uint8_native ? VK_FILTER_NEAREST
                                               : (r->mag_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
                smci.minFilter = sampled_uint8_native ? VK_FILTER_NEAREST
                                               : (r->min_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
                smci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                smci.addressModeU = wrap(r->addr_uvw[0]);
                smci.addressModeV = wrap(r->addr_uvw[1]);
                smci.addressModeW = wrap(r->addr_uvw[2]);
                smci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
                smci.compareEnable = sampled_depth ? VK_TRUE : VK_FALSE;
                smci.compareOp = static_cast<VkCompareOp>(r->depth_compare_func & 0x7u);
                if (!vk_ok(vkCreateSampler(ctx.device, &smci, nullptr, &bi.sampler), "image-sampler")) {
                    images_ready = false; break; }
            }
        }
        if (!images_ready) break;
        phase_setup = ComputeClock::now();

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
        pipeline_key.clear();
        auto append_pipeline_key_u32 = [&](uint32_t value) {
            pipeline_key.append(reinterpret_cast<const char*>(&value), sizeof(value));
        };
        append_pipeline_key_u32(static_cast<uint32_t>(item.spirv.size()));
        pipeline_key.append(reinterpret_cast<const char*>(item.spirv.data()),
                            item.spirv.size() * sizeof(uint32_t));
        append_pipeline_key_u32(static_cast<uint32_t>(item.user_sgprs.size()));
        append_pipeline_key_u32(static_cast<uint32_t>(layout_bindings.size()));
        for (const auto& binding : layout_bindings) {
            append_pipeline_key_u32(binding.binding);
            append_pipeline_key_u32(static_cast<uint32_t>(binding.descriptorType));
            append_pipeline_key_u32(binding.descriptorCount);
            append_pipeline_key_u32(binding.stageFlags);
        }
        if (const auto found = ctx.pipelines.find(pipeline_key); found != ctx.pipelines.end()) {
            descriptor_layout = found->second.descriptor_layout;
            shader = found->second.shader;
            pipeline_layout = found->second.pipeline_layout;
            pipeline = found->second.pipeline;
            pipeline_cached = true;
        } else {
            VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dlci.bindingCount = static_cast<uint32_t>(layout_bindings.size());
            dlci.pBindings = layout_bindings.data();
            if (!vk_ok(vkCreateDescriptorSetLayout(ctx.device, &dlci, nullptr, &descriptor_layout),
                       "descriptor-layout")) break;
        }
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

        if (!pipeline_cached) {
            VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smci.codeSize = item.spirv.size() * sizeof(uint32_t);
            smci.pCode = item.spirv.data();
            if (!vk_ok(vkCreateShaderModule(ctx.device, &smci, nullptr, &shader), "shader-module"))
                break;
            VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &descriptor_layout;
            VkPushConstantRange push_range{};
            if (!item.user_sgprs.empty()) {
                push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                push_range.size = static_cast<uint32_t>(item.user_sgprs.size() * sizeof(uint32_t));
                plci.pushConstantRangeCount = 1;
                plci.pPushConstantRanges = &push_range;
            }
            if (!vk_ok(vkCreatePipelineLayout(ctx.device, &plci, nullptr, &pipeline_layout),
                       "pipeline-layout")) break;
            VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module = shader;
            cpci.stage.pName = "main";
            cpci.layout = pipeline_layout;
            if (trace) std::fprintf(stderr, "[compute]   creating compute pipeline\n");
            if (!vk_ok(vkCreateComputePipelines(ctx.device, ctx.pipeline_cache, 1, &cpci, nullptr,
                                                &pipeline), "compute-pipeline")) break;
            if (trace) std::fprintf(stderr, "[compute]   compute pipeline ready\n");
            if (ctx.pipelines.size() < kMaxCachedComputePipelines) {
                ctx.pipelines.emplace(std::move(pipeline_key), CachedComputePipeline{
                    descriptor_layout, shader, pipeline_layout, pipeline});
                pipeline_cached = true;
            }
        }
        phase_pipeline = ComputeClock::now();

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
        // Exactly one binding emits the layout transitions for each borrowed renderer image. The
        // hazard is per-VkImage, so ownership is keyed on the handle, and it is derived over the
        // bindings that actually REACH these loops (non-aliased ones) rather than decided at import
        // time -- an owner chosen earlier could later be folded into an alias, leaving the image
        // transitioned zero times while descriptors still declare GENERAL.
        auto imported_barrier_owner = [&](size_t index) {
            const BoundImage& self = images[index];
            for (size_t prior = 0; prior < index; prior++)
                if (images[prior].imported && images[prior].alias_of == SIZE_MAX &&
                    images[prior].image == self.image)
                    return false;
            return true;
        };
        // Upload every image: UNDEFINED -> TRANSFER_DST, copy the staged texels in, -> GENERAL.
        for (size_t i = 0; i < images.size(); i++) {
            const BoundImage& bi = images[i];
            if (bi.alias_of != SIZE_MAX) continue;
            const ShaderResource* r = bi.resource;
            if (bi.seed_skip) {
                // #1122: nothing uploaded -- take the never-seeded image straight to GENERAL for the
                // write-only shader (which overwrites every texel). The result is read back below.
                VkImageMemoryBarrier to_general{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                to_general.srcAccessMask = 0;
                to_general.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
                to_general.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                to_general.srcQueueFamilyIndex = to_general.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                to_general.image = bi.image;
                to_general.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                                     1, &to_general);
                continue;
            }
            if (bi.imported) {
                if (!imported_barrier_owner(i)) continue;   // another binding transitions this image
                // Borrowed renderer image (#1095): nothing to upload. Make the renderer's writes
                // visible to this dispatch and move it into the GENERAL layout the descriptors
                // declare. The matching barrier after the dispatch puts it back.
                VkImageMemoryBarrier to_general{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                to_general.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                           VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                to_general.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                to_general.oldLayout = static_cast<VkImageLayout>(bi.imported_saved_layout);
                to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                to_general.srcQueueFamilyIndex = to_general.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                to_general.image = bi.image;
                to_general.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                                     1, &to_general);
                continue;
            }
            VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_dst.srcAccessMask = 0;
            to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_dst.image = bi.image;
            const bool array_depth = r->img_dim == 5 && r->depth_compare;
            const VkImageAspectFlags aspect = r->depth_compare
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            to_dst.subresourceRange = {aspect, 0, 1, 0, array_depth ? r->depth : 1u};
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_dst);
            VkBufferImageCopy region{};
            region.imageSubresource = {aspect, 0, 0, array_depth ? r->depth : 1u};
            region.imageExtent = {r->width, r->height, array_depth ? 1u : r->depth};
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
        if (!item.user_sgprs.empty())
            vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               static_cast<uint32_t>(item.user_sgprs.size() * sizeof(uint32_t)),
                               item.user_sgprs.data());
        vkCmdDispatch(command, item.launch.groups_x, item.launch.groups_y, item.launch.groups_z);
        // Hand every borrowed renderer image back in the layout its owner left it in (#1095), so the
        // renderer's own layout tracking stays true whether or not a dispatch consumed the target.
        for (size_t i = 0; i < images.size(); i++) {
            const BoundImage& bi = images[i];
            if (!bi.imported || bi.alias_of != SIZE_MAX || !imported_barrier_owner(i)) continue;
            VkImageMemoryBarrier restore{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            restore.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            // The renderer's next use of a persistent target is frequently vkCmdCopyImageToBuffer
            // or a scanout blit, so make the transition visible to transfer access as well.
            restore.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            restore.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            restore.newLayout = static_cast<VkImageLayout>(bi.imported_saved_layout);
            restore.srcQueueFamilyIndex = restore.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            restore.image = bi.image;
            restore.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                                 1, &restore);
        }
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
        if (trace) std::fprintf(stderr, "[compute]   submitting dispatch\n");
        if (!vk_ok(vkQueueSubmit(ctx.queue, 1, &submit, fence), "queue-submit")) break;
        if (trace) std::fprintf(stderr, "[compute]   waiting for dispatch\n");
        if (!vk_ok(vkWaitForFences(ctx.device, 1, &fence, VK_TRUE,
                                   30ull * 1000 * 1000 * 1000), "queue-wait")) {
            // cleanup() destroys the command buffer and releases the borrowed image's pin. With a
            // renderer-owned image bound, in-flight work still references it and its layout
            // transitions, so drain the queue first rather than freeing it underneath the GPU.
            // readback_persistent_color_target uses the same drain but PROMOTES a successful drain
            // to success; this item stays failed either way, which is the conservative choice for a
            // dispatch whose results we can no longer trust.
            if (vkQueueWaitIdle(ctx.queue) != VK_SUCCESS && trace)
                std::fprintf(stderr, "[compute]   queue drain after fence timeout failed\n");
            break;
        }
        if (trace) std::fprintf(stderr, "[compute]   dispatch complete\n");
        phase_dispatch = ComputeClock::now();

        bool readback_ok = true;
        for (auto& buffer : buffers) {
            if (buffer.alias_of != SIZE_MAX) continue;
            void* mapped = nullptr;
            if (vkMapMemory(ctx.device, buffer.memory, 0, buffer.resource->size, 0, &mapped) != VK_SUCCESS) {
                readback_ok = false;
                break;
            }
            uint8_t* destination = resource_bytes(buffer.resource);
            const auto* result = static_cast<const uint8_t*>(mapped);
            const bool changed = std::memcmp(destination, result, buffer.resource->size) != 0;
            if (trace) {
                buffer.after_hash = fnv1a(result, buffer.resource->size);
                for (uint32_t i = 0; i < buffer.resource->size; i++)
                    buffer.changed_bytes += destination[i] != result[i];
            }
            // Synchronous Unity maintenance kernels commonly rewrite a large persistent buffer with
            // the values it already contains. Terminator 2D's startup kernel binds 8,847,360 bytes;
            // after its first dispatch all later readbacks are identical. Avoiding the redundant host
            // write removes one full pass over both source and destination. Cache invalidation and
            // writer provenance remain unconditional: renderer-resident state can differ from guest
            // RAM even when consecutive compute readbacks contain identical bytes.
            if (changed)
                std::memcpy(destination, result, buffer.resource->size);
            vkUnmapMemory(ctx.device, buffer.memory);
            if (buffer.resource->gpu_addr)
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
            const size_t guest_texel = (r->format == DataFormat::Float10_11_11 ||
                                        r->format == DataFormat::Unorm2_10_10_10)
                ? 4u : (size_t)cb * nc;
            const size_t texels = (size_t)r->width * r->height * r->depth;
            std::vector<uint8_t> linear(texels * guest_texel, 0);
            const uint32_t* channels = static_cast<const uint32_t*>(mapped);
            if (bi.poison_verify) {
                size_t survived = 0;
                for (size_t t = 0; t < texels; ++t) {
                    bool all = true;
                    for (uint32_t c = 0; c < 4; ++c) if (channels[t * 4 + c] != 0xDEADBEEFu) all = false;
                    if (all) ++survived;
                }
                std::fprintf(stderr,
                             "[seed-skip-verify] code=0x%llx binding=%u texels=%zu poison_survived=%zu %s\n",
                             (unsigned long long)item.code_addr, bi.binding, texels, survived,
                             survived ? "PARTIAL-COVERAGE-UNSAFE" : "full-coverage-ok");
            }
            if (trace) {
                for (size_t t = 0; t < texels; t++)
                    for (uint32_t c = 0; c < 4; c++)
                        bi.nonzero_channels += channels[t * 4 + c] != 0;
            }
            const auto pack_start = ComputeClock::now();
            static const bool pack_range_enabled = !std::getenv("PROSPER_NO_PACK_RANGE");
            if (pack_range_enabled) {
                storage_pack_range(channels, r->format, nc, texels, linear.data(), guest_texel);
            } else {
                for (size_t t = 0; t < texels; t++)
                    storage_pack_texel(channels + t * 4, r->format, nc,
                                       linear.data() + t * guest_texel);
            }
            const auto pack_done = ComputeClock::now();
            static const bool verify_pack = std::getenv("PROSPER_VERIFY_PACK") != nullptr;
            if (verify_pack) {
                // Fail-visible A/B (mirrors PROSPER_VERIFY_UNPACK): the specialized range pack must
                // be bit-identical to the per-texel path it replaces, verified against the real
                // workload's texels. Logs the clean case too, so a verified run is self-proving.
                std::vector<uint8_t> expect(guest_texel);
                size_t bad = 0, first_bad = 0;
                for (size_t t = 0; t < texels; ++t) {
                    std::memset(expect.data(), 0, expect.size());
                    storage_pack_texel(channels + t * 4, r->format, nc, expect.data());
                    if (std::memcmp(expect.data(), linear.data() + t * guest_texel,
                                    guest_texel) != 0) {
                        if (!bad) first_bad = t;
                        ++bad;
                    }
                }
                std::fprintf(stderr,
                             "[compute] pack-verify binding=%u addr=0x%llx fmt=%u nc=%u "
                             "texels=%zu mismatches=%zu%s\n",
                             bi.binding, (unsigned long long)r->gpu_addr, (unsigned)r->format,
                             nc, texels, bad, bad ? " MISMATCH" : "");
                if (bad)
                    std::fprintf(stderr, "[compute] pack-verify first mismatch texel=%zu\n",
                                 first_bad);
            }
            uint8_t* destination = resource_bytes_for(r, bi.guest_bytes);
            if (r->tile_mode && r->depth > 1) {
                if (!tile_volume(destination, bi.guest_bytes, linear.data(), r->width, r->height,
                                 r->depth, r->tile_mode, static_cast<uint32_t>(guest_texel))) {
                    readback_ok = false;
                    vkUnmapMemory(ctx.device, staging_memory[i]);
                    break;
                }
            } else if (r->tile_mode && r->in_mip_tail) {
                tile_surface_level(destination, bi.guest_bytes, linear.data(),
                                   r->width, r->height, r->tile_mode,
                                   static_cast<uint32_t>(guest_texel),
                                   r->mip_tail_x, r->mip_tail_y);
            } else if (r->tile_mode) {
                tile_surface(destination, linear.data(), r->width, r->height, r->tile_mode, 0,
                             static_cast<uint32_t>(guest_texel));
            } else {
                std::memcpy(destination, linear.data(), linear.size());
            }
            const auto layout_done = ComputeClock::now();
            pack_ms += std::chrono::duration<double, std::milli>(pack_done - pack_start).count();
            layout_ms += std::chrono::duration<double, std::milli>(layout_done - pack_done).count();
            if (trace) bi.after_hash = fnv1a(destination, bi.guest_bytes);
            notify_guest_gpu_write(r->gpu_addr, bi.guest_bytes);
            if (!r->host_data && writer_provenance_enabled())
                record_guest_write(GuestWriterKind::ComputeBuffer,
                                   r->gpu_addr, bi.guest_bytes,
                                   item.submit_no, item.dispatch_index,
                                   item.command_order, item.code_addr);
            if (bi.dcc_metadata && bi.dcc_metadata_bytes) {
                std::memset(bi.dcc_metadata, 0xff, bi.dcc_metadata_bytes);
                notify_guest_gpu_write(r->metadata_addr, bi.dcc_metadata_bytes);
                if (!r->dcc_metadata_host_data && writer_provenance_enabled())
                    record_guest_write(GuestWriterKind::ComputeBuffer,
                                       r->metadata_addr, bi.dcc_metadata_bytes,
                                       item.submit_no, item.dispatch_index,
                                       item.command_order, item.code_addr);
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   DCC uncompressed binding=%u meta=0x%llx bytes=%zu code=0xff\n",
                                 bi.binding, (unsigned long long)r->metadata_addr,
                                 bi.dcc_metadata_bytes);
            }
            vkUnmapMemory(ctx.device, staging_memory[i]);
        }
        if (!readback_ok) break;
        ok = true;
        phase_writeback = ComputeClock::now();
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
            if (!buffer.resource || buffer.alias_of != SIZE_MAX) continue;
            const uint8_t* bytes = resource_bytes(buffer.resource);
            std::fprintf(stderr,
                         "[compute]   writeback binding=%u addr=0x%llx size=%u changed=%llu "
                         "hash=%016llx->%016llx first=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
                         buffer.resource->binding, (unsigned long long)buffer.resource->gpu_addr,
                         buffer.resource->size, (unsigned long long)buffer.changed_bytes,
                         (unsigned long long)buffer.before_hash, (unsigned long long)buffer.after_hash,
                         buffer.resource->size >= 4 ? reinterpret_cast<const uint32_t*>(bytes)[0] : 0,
                         buffer.resource->size >= 8 ? reinterpret_cast<const uint32_t*>(bytes)[1] : 0,
                         buffer.resource->size >= 12 ? reinterpret_cast<const uint32_t*>(bytes)[2] : 0,
                         buffer.resource->size >= 16 ? reinterpret_cast<const uint32_t*>(bytes)[3] : 0,
                         buffer.resource->size >= 20 ? reinterpret_cast<const uint32_t*>(bytes)[4] : 0,
                         buffer.resource->size >= 24 ? reinterpret_cast<const uint32_t*>(bytes)[5] : 0,
                         buffer.resource->size >= 28 ? reinterpret_cast<const uint32_t*>(bytes)[6] : 0,
                         buffer.resource->size >= 32 ? reinterpret_cast<const uint32_t*>(bytes)[7] : 0);
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
    if (phase_timing) {
        const auto phase_cleanup = ComputeClock::now();
        auto milliseconds = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        std::fprintf(stderr,
                     "[compute-phase] submit=%llu code=0x%llx ok=%u "
                     "setup_ms=%.2f setup_validate_ms=%.2f setup_buffers_ms=%.2f "
                     "pipeline_ms=%.2f dispatch_ms=%.2f "
                     "writeback_ms=%.2f pack_ms=%.2f layout_ms=%.2f "
                     "cleanup_ms=%.2f total_ms=%.2f\n",
                     (unsigned long long)item.submit_no, (unsigned long long)item.code_addr,
                     ok ? 1u : 0u, milliseconds(phase_start, phase_setup),
                     setup_validate_ms, setup_buffers_ms,
                     milliseconds(phase_setup, phase_pipeline),
                     milliseconds(phase_pipeline, phase_dispatch),
                     milliseconds(phase_dispatch, phase_writeback),
                     pack_ms, layout_ms,
                     milliseconds(phase_writeback, phase_cleanup),
                     milliseconds(phase_start, phase_cleanup));
    }
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
