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

#include <vulkan/vulkan.h>

#include <chrono>
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
    const char* failure = nullptr;   // which step failed, so a skip names its own cause

    bool init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_4;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
            failure = "vkCreateInstance failed"; return false;
        }
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (!count) { failure = "no Vulkan physical devices"; return false; }
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        const auto selection =
            prosper::frontend::select_vulkan_device(devices, VK_QUEUE_COMPUTE_BIT);
        if (!selection.device || selection.queue_family == UINT32_MAX) {
            failure = "no device with the runtime version, a compute queue and robustBufferAccess";
            return false;
        }
        physical = selection.device;
        family = selection.queue_family;
        properties = selection.properties;
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
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.queueFamilyIndex = family;
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(device, &cpi, nullptr, &pool) != VK_SUCCESS) {
            failure = "vkCreateCommandPool failed"; return false;
        }
        return true;
    }
    void destroy() {
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
        pool = VK_NULL_HANDLE; device = VK_NULL_HANDLE; instance = VK_NULL_HANDLE;
    }
    uint32_t host_memory_type(uint32_t bits) const {
        const VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
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
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
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

// One comparison. `count` is in uvec4 units and goes in the push constant exactly as production sets
// it; the buffers may be longer than count, which is how the out-of-range cases are built.
// `flag_offset` binds the changed flag at a nonzero offset the way production does -- it slices ONE
// shared buffer at `target_index * compare_flag_stride()`, so the dword beside a flag is another
// target's flag rather than allocation padding.
bool run_compare(Device& d, ComparePipeline& p, Buffer& a, Buffer& b, Buffer& flag,
                 uint32_t count, VkDeviceSize flag_offset = 0, double* elapsed_ms = nullptr) {
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
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count), &count);
    // Production's own group count rather than a local replica, so a sizing mistake here would be the
    // backend's too. (The previous version hardcoded `(count + 255) / 256` under a comment claiming
    // exactly this coupling, which it did not have.)
    const uint32_t groups = prosper::frontend::compute_result_compare_group_count(
        VkDeviceSize(count) * sizeof(Uvec4), d.properties.limits.maxStorageBufferRange,
        d.properties.limits.maxComputeWorkGroupCount[0]);
    if (!groups) { vkEndCommandBuffer(cmd); return give_up(); }
    vkCmdDispatch(cmd, groups, 1, 1);
    // The availability operation into the host domain. A fence orders EXECUTION; it does not move a
    // shader write into the host domain, and HOST_COHERENT does not exempt it. Production records
    // exactly this on the flag and on every baseline before reading them back. Without it this test
    // stays green on these drivers, which is why it is here deliberately rather than by luck --
    // frontends/shared/live/AGENTS.md and #2944.
    prosper::gpu::record_host_read_barrier(cmd, b.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                           VK_ACCESS_SHADER_WRITE_BIT);
    prosper::gpu::record_host_read_barrier(cmd, flag.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                           VK_ACCESS_SHADER_WRITE_BIT);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return give_up();

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(d.device, &fci, nullptr, &fence) != VK_SUCCESS) return give_up();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd;
    const auto started = std::chrono::steady_clock::now();
    const bool ok = vkQueueSubmit(d.queue, 1, &submit, fence) == VK_SUCCESS &&
                    vkWaitForFences(d.device, 1, &fence, VK_TRUE, 30ull * 1000 * 1000 * 1000) == VK_SUCCESS;
    if (elapsed_ms)
        *elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    vkDestroyFence(d.device, fence, nullptr);
    vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
    // The descriptor set is deliberately NOT freed; see the pool's creation flags.
    return ok;
}

}  // namespace

int main() {
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
    if (!d.init()) {
        // Name the cause. Several of the failures reachable here are environment or driver DEFECTS,
        // and reporting them as a benign absence is how a green run comes to mean nothing. Nothing
        // above asserts, so a skip is a clean exit rather than a suppressed failure.
        std::printf("[skip] %s\n", d.failure ? d.failure : "device initialization failed");
        d.destroy();
        return 0;
    }
    std::printf("[info] device: %s (type %u)\n", d.properties.deviceName,
                (unsigned)d.properties.deviceType);

    // A hand-counted ceiling on the 19 dispatches below, not a derived one; the pool is sized from
    // it, and exceeding it fails loudly (VK_ERROR_OUT_OF_POOL_MEMORY -> run_compare false -> a red
    // "dispatch completed") rather than silently.
    constexpr uint32_t kRuns = 32;
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

    // ---- how much more a changed result costs than an unchanged one ------------------------------
    // Reported, never asserted, as a MINIMUM over repetitions. The question is whether the
    // all-different case is dominated by contention on the changed-flag dword: every differing
    // invocation issues a Device-scope OpAtomicExchange against the same uint, while an all-equal
    // target issues none. Extra baseline traffic alone predicts about 1.5x.
    //
    // It was measured and it does NOT settle that. Guarding the exchange with a plain load (sound --
    // the flag is only ever set, and the stored value is the constant 1) gave all-different times of
    // 1.449/1.741/1.807 ms against 2.012/1.784/2.280 ms unguarded: the spread WITHIN each arm is as
    // large as the difference between them. The guard was not adopted. This harness times a
    // submit-plus-fence on the CPU, so it carries queue scheduling noise and needs GPU timestamps to
    // answer the question properly. The figure is printed because whoever fuses this comparison into
    // the retile shader will want a before/after, and a fixed fixture beats a game route.
    {
        Buffer big_a, big_b;
        constexpr uint32_t kBig = 1u << 19;           // 8 MiB of uvec4
        const VkDeviceSize big_bytes = VkDeviceSize(kBig) * sizeof(Uvec4);
        check(big_a.create(d, big_bytes) && big_b.create(d, big_bytes),
              "timing fixture: 8 MiB buffers allocated");
        if (big_a.mapped && big_b.mapped) {
            auto* ba = static_cast<Uvec4*>(big_a.mapped);
            auto* bb = static_cast<Uvec4*>(big_b.mapped);
            for (uint32_t i = 0; i < kBig; ++i) ba[i] = {i, i, i, i};
            std::memcpy(bb, ba, big_bytes);
            constexpr int kReps = 5;
            double equal_ms = 1e9, different_ms = 1e9, sample = 0;
            run_compare(d, p, big_a, big_b, flags, kBig, kFlagSlot * flag_stride, &sample);   // warm
            bool equal_clean = true;
            for (int i = 0; i < kReps; ++i) {
                *fv = 0;
                run_compare(d, p, big_a, big_b, flags, kBig, kFlagSlot * flag_stride, &sample);
                equal_clean = equal_clean && *fv == 0;
                if (sample < equal_ms) equal_ms = sample;
            }
            bool different_raised = true;
            for (int i = 0; i < kReps; ++i) {
                // Diverge at the TOP of every repetition, and only here. The baseline equals `a` on
                // entry -- from the equal loop, or because the previous repetition's comparison
                // adopted `a` into it -- so one XOR per iteration is what makes each run measure the
                // changed path. Diverging before the loop as well would cancel this one on the first
                // iteration and silently measure the EQUAL path, which is what the first version did;
                // the per-repetition flag assertion below is what caught it.
                for (uint32_t j = 0; j < kBig; ++j) bb[j].w ^= 0xffffffffu;
                *fv = 0;
                run_compare(d, p, big_a, big_b, flags, kBig, kFlagSlot * flag_stride, &sample);
                different_raised = different_raised && *fv == 1;
                if (sample < different_ms) different_ms = sample;
            }
            std::printf("[info] 8 MiB compare, min of %d: all-equal %.3f ms, all-different %.3f ms "
                        "(ratio %.2f) -- CPU-timed, carries scheduling noise\n",
                        kReps, equal_ms, different_ms, equal_ms > 0 ? different_ms / equal_ms : 0.0);
            check(equal_clean, "8 MiB all-equal comparison leaves the flag clear, every repetition");
            check(different_raised, "8 MiB all-different comparison raises the flag, every repetition");
        }
        big_a.destroy(d); big_b.destroy(d);
    }

    a.destroy(d); b.destroy(d); flags.destroy(d);
    p.destroy(d);
    d.destroy();
    if (failures) { std::printf("compute_result_compare: %d FAILED\n", failures); return 1; }
    std::printf("compute_result_compare: OK\n");
    return 0;
}
