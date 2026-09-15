// EXECUTION coverage for build_compute_compare_uvec4() -- the shader whose changed-flag decides
// whether a compute storage writeback is skipped.
//
// What was missing, stated precisely: the module is already emitted and `spirv-val`-checked by
// `tools/spv_validate` (registered unconditionally; CI hard-fails when spirv-val is absent), and the
// Vulkan validation scan covers it at runtime. What nothing did was RUN it and look at what it
// computed.
//
// That gap grew teeth recently. While the ceiling admitting the comparison was 2 MiB almost nothing
// reached it; #3685 derives that ceiling from the memory topology and admits results up to 128 MiB on
// a unified-memory device, so this shader now gates the publication of every 4K compute target on the
// platform this project develops on. A false "unchanged" does not merely skip a baseline update --
// live_compute.cpp reads the flag as `gpu_result_unchanged` and skips writing the dispatch's result
// to guest memory at all, which surfaces as a frame silently keeping stale pixels.
//
// **This test does not guard #1711**, and an earlier version of this header claimed it did. That
// defect declared the flag as a uvec4 runtime array with ArrayStride 16 over a 4-byte binding, making
// element 0 out of bounds under robustBufferAccess -- which a driver is PERMITTED to discard. The fix
// commit records that neither RADV nor lavapipe actually did, so the module behaved correctly for the
// whole life of the bug and any behavioural assertion would have been green throughout. What catches
// that class is `spv_validate` and the validation scan, not this file.
//
// Two properties here are worth more than the rest, because they fail against a real wrong
// implementation rather than re-walking the happy path:
//
//   * a difference in ANY of the four components raises the flag -- asserted per lane, because a
//     comparison testing only .x passes three of the four;
//   * words at or beyond the push-constant count are untouched even when they differ, and a
//     difference beyond the count does not raise the flag. The count is deliberately 300, not a
//     multiple of 256, so the final workgroup launches 256 invocations of which 44 are in range.
//
// One note on the skip, so it is not misread as "no GPU". The device is requested at
// VK_API_VERSION_1_4 and selected through `select_vulkan_device`, whose runtime-version gate is the
// contract live_compute.cpp ships against; robustBufferAccess is required for the same reason. That
// is a real narrowing over the 1.1 an earlier version accepted -- on a 1.3-only environment this
// test now skips where it would once have run. Testing against a looser contract than production's
// would be testing a different shader environment, so the narrowing is deliberate.
//
// And one property is deliberately NOT claimed. The shader stores the baseline only on the differing
// branch, but writing `a[i]` where `a[i] == b[i]` is byte-idempotent, so an implementation that
// stored unconditionally is indistinguishable through buffer contents. No assertion below pins it,
// and none pretends to.

#include "gpu/recompiler/spirv_builder.hpp"
#include "gpu/execute/host_read_barrier.hpp"
#include "shared/device/vulkan_device_select.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/live/gpu_retile.hpp"

#include <vulkan/vulkan.h>

#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const std::string& message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message.c_str());
    failures += !ok;
}

struct Uvec4 { uint32_t x, y, z, w; };

// The device is built here rather than through the live backend so a failure indicts the SHADER
// rather than the backend's resource machinery. The SELECTION, though, is the shared one production
// uses: Vulkan does not promise physical devices are enumerated in performance order, and a
// hand-rolled "first with a compute queue" can land on a software implementation on a box that also
// has a real GPU -- which for a test about driver-dependent behaviour is the wrong device entirely.
struct Device {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory{};
    VkQueryPool timestamps = VK_NULL_HANDLE;
    uint32_t timestamp_bits = 0;
    int failure_exit = 1;
    const char* failure = nullptr;   // which step failed, so a skip names its own cause

    bool init(bool benchmark) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_4;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        const auto instance_result = vkCreateInstance(&ici, nullptr, &instance);
        if (instance_result != VK_SUCCESS) {
            failure_exit = instance_result == VK_ERROR_INCOMPATIBLE_DRIVER ? 77 : 1;
            failure = "vkCreateInstance failed"; return false;
        }
        uint32_t count = 0;
        if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS) {
            failure = "physical-device enumeration failed"; return false;
        }
        if (!count) { failure_exit = 77; failure = "no Vulkan physical devices"; return false; }
        std::vector<VkPhysicalDevice> devices(count);
        if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) {
            failure = "physical-device enumeration changed or failed"; return false;
        }
        devices.resize(count);
        const auto selection =
            prosper::frontend::select_vulkan_device(devices, VK_QUEUE_COMPUTE_BIT);
        if (!selection.device || selection.queue_family == UINT32_MAX) {
            failure_exit = 77;
            failure = "no device with the runtime version, a compute queue and robustBufferAccess";
            return false;
        }
        physical = selection.device;
        family = selection.queue_family;
        properties = selection.properties;

        uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
        std::vector<VkQueueFamilyProperties> queues(families);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, queues.data());
        timestamp_bits = queues.at(family).timestampValidBits;
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = family; qci.queueCount = 1; qci.pQueuePriorities = &priority;
        // Requested because live_compute.cpp requires it; testing without it would exercise a
        // different device contract than the one this shader ships against.
        VkPhysicalDeviceFeatures enabled{};
        enabled.robustBufferAccess = VK_TRUE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pEnabledFeatures = &enabled;
        if (vkCreateDevice(physical, &dci, nullptr, &device) != VK_SUCCESS) {
            failure = "vkCreateDevice failed"; return false;
        }
        vkGetDeviceQueue(device, family, 0, &queue);
        if (benchmark && timestamp_bits && properties.limits.timestampPeriod > 0) {
            VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            info.queryCount = 2;
            if (vkCreateQueryPool(device, &info, nullptr, &timestamps) != VK_SUCCESS) {
                failure = "timestamp query pool creation failed"; return false;
            }
        }
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.queueFamilyIndex = family;
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(device, &cpi, nullptr, &pool) != VK_SUCCESS) {
            failure = "vkCreateCommandPool failed"; return false;
        }
        return true;
    }
    void destroy() {
        if (timestamps) vkDestroyQueryPool(device, timestamps, nullptr);
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
        pool = VK_NULL_HANDLE; device = VK_NULL_HANDLE; instance = VK_NULL_HANDLE;
    }
    uint32_t host_memory_type(uint32_t bits) const {
        const VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        // Match the live allocator: prefer cached coherent host memory, then coherent.
        const auto cached = want | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & cached) == cached) return i;
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX;
    }
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize bytes = 0;

    bool create(Device& d, VkDeviceSize size) {
        bytes = size;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(d.device, &bci, nullptr, &buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(d.device, buffer, &requirements);
        const uint32_t type = d.host_memory_type(requirements.memoryTypeBits);
        if (type == UINT32_MAX) { destroy(d); return false; }
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = requirements.size; mai.memoryTypeIndex = type;
        if (vkAllocateMemory(d.device, &mai, nullptr, &memory) != VK_SUCCESS) { destroy(d); return false; }
        if (vkBindBufferMemory(d.device, buffer, memory, 0) != VK_SUCCESS) { destroy(d); return false; }
        if (vkMapMemory(d.device, memory, 0, size, 0, &mapped) != VK_SUCCESS) { destroy(d); return false; }
        return true;
    }
    void destroy(Device& d) {
        if (mapped) vkUnmapMemory(d.device, memory);
        if (memory) vkFreeMemory(d.device, memory, nullptr);
        if (buffer) vkDestroyBuffer(d.device, buffer, nullptr);
        buffer = VK_NULL_HANDLE; memory = VK_NULL_HANDLE; mapped = nullptr;
    }
};

// Three storage bindings and a one-uint push constant, matching the production layout.
struct ComparePipeline {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module_ = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

    bool create(Device& d, uint32_t max_sets) {
        VkDescriptorSetLayoutBinding bindings[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        sli.bindingCount = 3; sli.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(d.device, &sli, nullptr, &set_layout) != VK_SUCCESS) return false;
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &set_layout;
        pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(d.device, &pli, nullptr, &layout) != VK_SUCCESS) return false;

        const std::vector<uint32_t> spirv = prosper::gpu::build_compute_compare_uvec4();
        if (spirv.empty()) return false;
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = spirv.size() * sizeof(uint32_t); smi.pCode = spirv.data();
        if (vkCreateShaderModule(d.device, &smi, nullptr, &module_) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = module_; cpi.stage.pName = "main";
        cpi.layout = layout;
        if (vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline) != VK_SUCCESS)
            return false;
        // Created WITHOUT VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT and never freed
        // individually -- which is production's shape too (it recycles with vkResetDescriptorPool or
        // destroys the pool). Calling vkFreeDescriptorSets on a pool lacking that bit is undefined
        // behaviour, and the Vulkan validation scan fails the build for it; the first version of this
        // file did exactly that and turned the Linux job red.
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * max_sets};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = max_sets; dpi.poolSizeCount = 1; dpi.pPoolSizes = &size;
        return vkCreateDescriptorPool(d.device, &dpi, nullptr, &descriptor_pool) == VK_SUCCESS;
    }
    void destroy(Device& d) {
        if (descriptor_pool) vkDestroyDescriptorPool(d.device, descriptor_pool, nullptr);
        if (pipeline) vkDestroyPipeline(d.device, pipeline, nullptr);
        if (module_) vkDestroyShaderModule(d.device, module_, nullptr);
        if (layout) vkDestroyPipelineLayout(d.device, layout, nullptr);
        if (set_layout) vkDestroyDescriptorSetLayout(d.device, set_layout, nullptr);
    }
};

struct RetileStep {
    prosper::frontend::GpuRetilePipeline* pipeline;
    prosper::frontend::GpuRetileParameters parameters;
    VkBuffer output;
    VkDescriptorSet set;
};

// One comparison. `count` is in uvec4 units and goes in the push constant exactly as production sets
// it; the buffers may be longer than count, which is how the out-of-range cases are built.
// `flag_offset` binds the changed flag at a nonzero offset the way production does -- it slices ONE
// shared buffer at `target_index * compare_flag_stride()`, so the dword beside a flag is another
// target's flag rather than allocation padding.
bool run_compare(Device& d, ComparePipeline& p, Buffer& a, Buffer& b, Buffer& flag,
                 uint32_t count, VkDeviceSize flag_offset = 0, double* elapsed_ms = nullptr, double* gpu_ms = nullptr, const RetileStep* retile = nullptr) {
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = p.descriptor_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &p.set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(d.device, &dai, &set) != VK_SUCCESS) return false;
    VkDescriptorBufferInfo infos[3] = {
        {a.buffer, 0, a.bytes}, {b.buffer, 0, b.bytes}, {flag.buffer, flag_offset, sizeof(uint32_t)}};
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(d.device, 3, writes, 0, nullptr);

    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = d.pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(d.device, &cbi, &cmd) != VK_SUCCESS) return false;
    // Every failure from here on frees the command buffer. The descriptor set is NOT reclaimed --
    // the pool has no FREE bit, by design -- so a failing path permanently consumes one of kRuns;
    // bounded, and every such path has already reddened a check.
    const auto give_up = [&] {
        vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
        return false;
    };
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) return give_up();
    // Production's own group count rather than a local replica, so a sizing mistake here would be the
    // backend's too. (The previous version hardcoded `(count + 255) / 256` under a comment claiming
    // exactly this coupling, which it did not have.)
    const uint32_t groups = prosper::frontend::compute_result_compare_group_count(
        VkDeviceSize(count) * sizeof(Uvec4), d.properties.limits.maxStorageBufferRange,
        d.properties.limits.maxComputeWorkGroupCount[0]);
    if (!groups) { vkEndCommandBuffer(cmd); return give_up(); }
    if (gpu_ms) {
        if (!d.timestamps) { vkEndCommandBuffer(cmd); return give_up(); }
        vkCmdResetQueryPool(cmd, d.timestamps, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, d.timestamps, 0);
    }
    if (retile)
        retile->pipeline->record(cmd, a.buffer, retile->output, retile->set, retile->parameters,
                                VK_NULL_HANDLE, b.buffer, flag.buffer, flag_offset);
    if (!retile || !retile->pipeline->compare_result) {
        if (retile) {
            // Match the production separate-pass dependency, including retile's compute stage.
            VkBufferMemoryBarrier barriers[3]{};
            for (auto& barrier : barriers) {
                barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                        VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            }
            barriers[0].buffer = a.buffer; barriers[0].size = a.bytes;
            barriers[1].buffer = b.buffer; barriers[1].size = b.bytes;
            barriers[2].buffer = flag.buffer; barriers[2].offset = flag_offset;
            barriers[2].size = sizeof(uint32_t);
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 3, barriers, 0, nullptr);
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count), &count);
        vkCmdDispatch(cmd, groups, 1, 1);
    }
    if (gpu_ms && !retile)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, d.timestamps, 1);
    // The availability operation into the host domain. A fence orders EXECUTION; it does not move a
    // shader write into the host domain, and HOST_COHERENT does not exempt it. Production records
    // exactly this on the flag and on every baseline before reading them back. Without it this test
    // stays green on these drivers, which is why it is here deliberately rather than by luck --
    // frontends/shared/live/AGENTS.md and #2944.
    if (!retile || !retile->pipeline->compare_result) {
        prosper::gpu::record_host_read_barrier(cmd, b.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                               VK_ACCESS_SHADER_WRITE_BIT);
        prosper::gpu::record_host_read_barrier(cmd, flag.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                               VK_ACCESS_SHADER_WRITE_BIT);
    }
    if (gpu_ms && retile)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, d.timestamps, 1);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return give_up();

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(d.device, &fci, nullptr, &fence) != VK_SUCCESS) return give_up();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd;
    const auto started = std::chrono::steady_clock::now();
    bool ok = vkQueueSubmit(d.queue, 1, &submit, fence) == VK_SUCCESS &&
                    vkWaitForFences(d.device, 1, &fence, VK_TRUE, 30ull * 1000 * 1000 * 1000) == VK_SUCCESS;
    if (!ok) {
        // Completion is unproved. Do not destroy possibly pending commands or their resources.
        std::fprintf(stderr, "FAIL: comparison submission/completion failed\n");
        std::fflush(nullptr);
        std::_Exit(1);
    }
    if (elapsed_ms)
        *elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    if (ok && gpu_ms) {
        uint64_t ticks[2]{};
        ok = vkGetQueryPoolResults(d.device, d.timestamps, 0, 2, sizeof(ticks), ticks,
                                  sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
        if (ok) {
            const uint64_t mask = d.timestamp_bits == 64 ? UINT64_MAX :
                                  (uint64_t{1} << d.timestamp_bits) - 1;
            *gpu_ms = double((ticks[1] - ticks[0]) & mask) *
                      d.properties.limits.timestampPeriod / 1e6;
        }
    }
    vkDestroyFence(d.device, fence, nullptr);
    vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
    // The descriptor set is deliberately NOT freed; see the pool's creation flags.
    return ok;
}

// Same command-buffer boundary for separate and fused work. Inputs/baselines use the
// production host-visible allocation preference; this does not simulate a preceding guest shader.
void compare_retile(Device& d, bool measured) {
    using namespace prosper::frontend;
    GpuRetilePipeline pipelines[2];
    ComparePipeline comparator;
    bool ready = comparator.create(d, 256) &&
        pipelines[0].initialize(d.device, VK_NULL_HANDLE) == VK_SUCCESS &&
        pipelines[1].initialize(d.device, VK_NULL_HANDLE, prosper::gpu::RetileShaderKind::Words2D,
            prosper::gpu::RetileImageSource::LinearBuffer, false, d.physical, true) == VK_SUCCESS;
    check(ready, "retile benchmark pipelines created");
    const auto flag_stride = std::max(VkDeviceSize(4), d.properties.limits.minStorageBufferOffsetAlignment);
    for (const auto [width, height] : {std::pair{260u, 129u}, std::pair{2048u, 1024u},
                                    std::pair{4096u, 2048u}}) {
        if (!ready) break;
        if (!measured && width > 260) continue;
        GpuRetileParameters params;
        ready = params.initialize(width, height, 4, 27, d.properties.limits);
        check(ready, "retile benchmark shape admitted");
        if (!ready) break;
        Buffer input, baseline, output, flag;
        VkDescriptorPool pools[2]{};
        VkDescriptorSet sets[2]{};
        ready = input.create(d, params.linear_bytes) && baseline.create(d, params.linear_bytes) &&
                output.create(d, params.tiled_bytes) && flag.create(d, flag_stride * 5);
        check(ready, "retile benchmark buffers allocated");
        if (ready) {
            const uint32_t words = uint32_t(params.linear_bytes / 4);
            auto* in = static_cast<uint32_t*>(input.mapped);
            auto* prior = static_cast<uint32_t*>(baseline.mapped);
            auto* flags = static_cast<uint8_t*>(flag.mapped);
            for (uint32_t i = 0; i < words; ++i) in[i] = i * 2654435761u;
            std::vector<uint8_t> expected(params.tiled_bytes, 0);
            prosper::gpu::tile_surface(expected.data(), reinterpret_cast<uint8_t*>(in),
                                      width, height, 27, 0, 4);
            for (unsigned arm = 0; arm < 2; ++arm)
                ready = ready && pipelines[arm].bind(input.buffer, output.buffer, params, pools[arm],
                    sets[arm], VK_NULL_HANDLE, baseline.buffer, flag.buffer, flag_stride) == VK_SUCCESS;
            check(ready, "retile benchmark bindings created");
            for (unsigned pattern = 0; pattern < 3 && ready; ++pattern) {
                for (int rep = measured ? -2 : 0; rep < (measured ? 12 : 1) && ready; ++rep) {
                    for (unsigned order = 0; order < 2; ++order) {
                        const unsigned arm = order ^ unsigned((rep + 2) & 1);
                        std::memcpy(prior, in, params.linear_bytes);
                        if (pattern) {
                            for (uint32_t i = pattern == 1 ? words - 1 : 0; i < words; ++i)
                                prior[i] ^= 0x80000000u;
                        }
                        // Poison padding on every dispatch, including the fully equal comparison.
                        std::memset(output.mapped, 0xa5, params.tiled_bytes);
                        std::memset(flags, 0xa5, flag.bytes);
                        *reinterpret_cast<uint32_t*>(flags + flag_stride) = 0;
                        RetileStep step{&pipelines[arm], params, output.buffer, sets[arm]};
                        double cpu_ms = 0, gpu_ms = 0;
                        ready = run_compare(d, comparator, input, baseline, flag, words / 4,
                                            flag_stride, measured ? &cpu_ms : nullptr, measured ? &gpu_ms : nullptr, &step) &&
                            *reinterpret_cast<uint32_t*>(flags + flag_stride) == unsigned(pattern != 0) &&
                            std::memcmp(prior, in, params.linear_bytes) == 0 &&
                            std::memcmp(output.mapped, expected.data(), params.tiled_bytes) == 0;
                        for (uint32_t slot = 0; slot < 5; ++slot)
                            if (slot != 1) ready = ready &&
                                *reinterpret_cast<uint32_t*>(flags + flag_stride * slot) == 0xa5a5a5a5u;
                        if (!ready) { check(false, "retile benchmark completion/bytes/padding/flag"); break; }
                        if (measured && rep >= 0)
                            std::printf("[retile-benchmark] width=%u height=%u pattern=%u rep=%d "
                                        "order=%u fused=%u gpu_ms=%.6f cpu_ms=%.6f\n",
                                        width, height, pattern, rep, order, arm, gpu_ms, cpu_ms);
                    }
                }
            }
        }
        for (auto pool : pools) if (pool) vkDestroyDescriptorPool(d.device, pool, nullptr);
        input.destroy(d); baseline.destroy(d); output.destroy(d); flag.destroy(d);
    }
    comparator.destroy(d);
    for (auto& pipeline : pipelines) pipeline.destroy();
}

}  // namespace

int main(int argc, char** argv) {
    bool benchmark = false, retile = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--retile-benchmark") == 0) benchmark = retile = true;
        else if (std::strcmp(argv[i], "--retile") == 0) retile = true;
        else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 1; }
    }
    // The pure eligibility contract -- which byte counts and device limits admit a comparison at
    // all -- is NOT restated here. `tests/gpu/recompiler/test_game_compute.cpp` already covers it,
    // and covers it better, because it separates each refusal clause instead of merely reaching
    // one: `!(4112, 4096, 2)` is the storage-range clause alone (2 groups is within the 2
    // permitted) and `!(4112, 4112, 1)` is the max-groups clause alone, alongside the zero,
    // not-a-multiple-of-16, zero-limit and overflow arms and the 65,535 dispatch boundary pair.
    //
    // An earlier version of this file restated four of those here and got one wrong in exactly the
    // way that block existed to prevent: `(1ull << 28, 1u << 27, 65535)` lands one workgroup past
    // BOTH limits at once, so deleting the storage-range clause entirely left the assertion green.
    // A correct assertion that cannot fail for the reason it names is worse than no assertion. What
    // this file demonstrates about that function instead is the coupling a pure test cannot: every
    // dispatch below is sized by calling it, so a sizing mistake here would be the backend's too.
    Device d;
    if (!d.init(benchmark)) {
        std::printf("[%s] %s\n", d.failure_exit == 77 ? "skip" : "FAIL",
                    d.failure ? d.failure : "device initialization failed");
        d.destroy();
        return d.failure_exit;
    }
    std::printf("[info] device: %s (type %u)\n", d.properties.deviceName,
                (unsigned)d.properties.deviceType);

    if (benchmark && !d.timestamps) {
        std::printf("[skip] queue timestamps unavailable\n"); d.destroy(); return 77;
    }
    // Deliberate upper bound; descriptor exhaustion fails a dispatch assertion.
    constexpr uint32_t kRuns = 128;
    ComparePipeline p;
    if (!p.create(d, kRuns)) { std::printf("FAIL: could not build the compare pipeline\n"); return 1; }

    constexpr uint32_t kWords = 1024;         // uvec4s; 4 full workgroups
    const VkDeviceSize bytes = kWords * sizeof(Uvec4);
    // Flag slots, bound the way production binds them: it slices ONE buffer at
    // `target_index * compare_flag_stride()`, so the dwords beside a flag are OTHER TARGETS' flags
    // rather than allocation padding. The flag under test is slot 1, leaving one neighbour behind it
    // and three ahead, and the DIRECTION is the point. An over-wide STORE through the flag chain can
    // only smear forward: a SPIR-V access chain cannot produce a negative byte offset from the
    // binding base, so a 16-byte write from that base spans slots 1 through 4 at the 4-byte stride
    // RADV and lavapipe report here -- the flag itself plus all three canaries ahead of it, which is
    // why the buffer is five slots and not two. (A uint OpAtomicExchange through such a chain writes
    // only its own four bytes and does not smear at all; the over-wide store is the case worth
    // guarding.) Production smears the same way -- `{compare_flags, j * stride, 4}` puts target j's
    // overrun on j+1, j+2, j+3. An earlier version of this file placed its only canary BEHIND the
    // binding, where no device write can reach it by any route.
    //
    // The honest limit, because a guard that cannot fail is worse than none: the descriptor's range
    // is 4 bytes and robustBufferAccess is on, so on RADV the byte-granular buffer bound is what
    // stops such a write, and a shader carrying that defect would very likely leave these canaries
    // green on this driver anyway. They are a cheap structural guard that CAN fail, not a
    // discriminator for the class -- `spv_validate` and the Vulkan validation scan are what catch
    // it. Slot 0 earns its place on a different property: paired with the flag at slot 1 reading
    // back as expected, an untouched slot 0 shows the device honoured the descriptor's nonzero
    // offset rather than writing at the buffer base.
    const VkDeviceSize flag_stride =
        d.properties.limits.minStorageBufferOffsetAlignment > sizeof(uint32_t)
            ? d.properties.limits.minStorageBufferOffsetAlignment : sizeof(uint32_t);
    constexpr uint32_t kFlagSlots = 5;
    constexpr uint32_t kFlagSlot = 1;         // slots 0 and 2..4 are canaries
    Buffer a, b, flags;
    if (!a.create(d, bytes) || !b.create(d, bytes) || !flags.create(d, kFlagSlots * flag_stride)) {
        std::printf("FAIL: buffer allocation\n");
        return 1;
    }
    auto* av = static_cast<Uvec4*>(a.mapped);
    auto* bv = static_cast<Uvec4*>(b.mapped);
    const auto slot = [&](uint32_t i) {
        return reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(flags.mapped) + i * flag_stride);
    };
    uint32_t* const fv = slot(kFlagSlot);
    const auto fill = [&](Uvec4* dst, uint32_t seed) {
        for (uint32_t i = 0; i < kWords; ++i)
            dst[i] = {seed + i, seed + i * 3u, seed + i * 5u, seed + i * 7u};
    };
    const auto compare = [&](uint32_t count) {
        return run_compare(d, p, a, b, flags, count, kFlagSlot * flag_stride);
    };
    constexpr uint32_t kCanary = 0xA5A5A5A5u;
    // Re-armed before every case, not once: a case that ran after a corrupting one would otherwise
    // read an already-clobbered slot and attribute it to itself.
    const auto arm_canaries = [&] {
        for (uint32_t i = 0; i < kFlagSlots; ++i)
            if (i != kFlagSlot) *slot(i) = kCanary;
    };
    const auto canaries_intact = [&] {
        for (uint32_t i = 0; i < kFlagSlots; ++i)
            if (i != kFlagSlot && *slot(i) != kCanary) return false;
        return true;
    };

    // ---- equal: the flag stays clear ------------------------------------------------------------
    fill(av, 11); fill(bv, 11); *fv = 0; arm_canaries();
    check(compare(kWords), "equal: dispatch completed");
    check(*fv == 0, "equal: flag stays clear");
    check(canaries_intact(), "equal: the neighbouring flag slots are untouched");

    // ---- one differing component, in each lane --------------------------------------------------
    // A comparison that tested only .x would pass three of these four.
    for (uint32_t lane = 0; lane < 4; ++lane) {
        fill(av, 11); fill(bv, 11); *fv = 0; arm_canaries();
        uint32_t* component = &reinterpret_cast<uint32_t*>(&av[500])[lane];
        *component += 1;
        check(compare(kWords), "single-component: dispatch completed");
        check(*fv == 1, "a difference in lane " + std::to_string(lane) + " alone raises the flag");
        check(std::memcmp(&bv[500], &av[500], sizeof(Uvec4)) == 0,
              "single-component: that word's baseline adopts the new value");
        check(canaries_intact(), "single-component: the neighbouring flag slots are untouched");
    }

    // ---- every word differs ---------------------------------------------------------------------
    fill(av, 11); fill(bv, 22); *fv = 0; arm_canaries();
    check(compare(kWords), "all-different: dispatch completed");
    check(*fv == 1, "all-different: flag raised");
    check(std::memcmp(bv, av, bytes) == 0, "all-different: the whole baseline adopts the new result");
    check(canaries_intact(), "all-different: the neighbouring flag slots are untouched");

    // ---- out of range: words at or beyond `count` are untouched even when they differ ------------
    // 300 is deliberately not a multiple of 256, so the final workgroup launches 256 invocations of
    // which 44 are in range -- the case a missing or off-by-one bounds check gets wrong in production
    // and never in a tidy fixture. 300 * 16 = 4800 bytes is a multiple of 16, so the group-count
    // contract admits it.
    constexpr uint32_t kInRange = 300;
    fill(av, 11); fill(bv, 22); *fv = 0; arm_canaries();
    std::vector<Uvec4> tail(bv + kInRange, bv + kWords);
    check(compare(kInRange), "partial workgroup: dispatch completed");
    check(*fv == 1, "partial workgroup: an in-range difference raises the flag");
    check(std::memcmp(bv, av, kInRange * sizeof(Uvec4)) == 0,
          "partial workgroup: every in-range word is updated");
    check(std::memcmp(bv + kInRange, tail.data(), tail.size() * sizeof(Uvec4)) == 0,
          "partial workgroup: words at and beyond count are UNTOUCHED");
    check(canaries_intact(), "partial workgroup: the neighbouring flag slots are untouched");

    // ---- a difference only beyond count must not raise the flag ---------------------------------
    fill(av, 11); fill(bv, 11); *fv = 0; arm_canaries();
    av[kInRange].x += 1;                              // the first word past the end
    check(compare(kInRange), "beyond-count: dispatch completed");
    check(*fv == 0, "a difference beyond count does not raise the flag");
    check(canaries_intact(), "beyond-count: the neighbouring flag slots are untouched");

    // A single changed invocation must publish even when it is not a subgroup leader.
    // Boundaries cover typical wave sizes without assuming any of them on this device.
    for (const uint32_t index : {1u, 31u, 32u, 63u, 64u, 255u, 256u, 299u}) {
        fill(av, 11); fill(bv, 11); *fv = 0; arm_canaries();
        av[index].z ^= 0x80000000u;
        check(compare(kInRange), "singleton: dispatch completed");
        check(*fv == 1, "singleton " + std::to_string(index) + ": flag raised");
        check(std::memcmp(bv, av, bytes) == 0, "singleton: complete baseline adopted");
        *fv = 0;
        check(compare(kInRange) && *fv == 0, "singleton: repeat after adoption is equal");
        check(canaries_intact(), "singleton: flag neighbours untouched");
    }
    fill(av, 11); fill(bv, 11); *fv = 0; arm_canaries();
    av[0].w ^= 1;
    check(compare(1) && *fv == 1 && std::memcmp(bv, av, bytes) == 0,
          "count one: sole valid invocation adopts and publishes");
    check(canaries_intact(), "count one: flag neighbours untouched");
    fill(av, 11); fill(bv, 11); *fv = 0; arm_canaries();
    for (uint32_t i = 1; i < kInRange; i += 2) av[i].y ^= 1;
    check(compare(kInRange) && *fv == 1 && std::memcmp(bv, av, bytes) == 0,
          "alternating differences: all changed invocations adopt their baseline");
    check(canaries_intact(), "alternating: flag neighbours untouched");

    if (retile) compare_retile(d, benchmark);

    a.destroy(d); b.destroy(d); flags.destroy(d);
    p.destroy(d);
    d.destroy();
    if (failures) { std::printf("compute_result_compare: %d FAILED\n", failures); return 1; }
    std::printf("compute_result_compare: OK\n");
    return 0;
}
