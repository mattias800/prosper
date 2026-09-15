// Execution coverage for build_compute_compare_uvec4() -- the shader that decides whether a compute
// storage writeback is SKIPPED.
//
// It had none. That mattered less when the ceiling admitting it was 2 MiB, because almost nothing
// reached it; #3685 derives that ceiling from the memory topology and admits results up to 128 MiB on
// a unified-memory device, so this shader now gates the publication of every 4K compute target on the
// platform this project develops on. A false "unchanged" here does not merely skip a baseline update
// -- live_compute.cpp reads the flag as `gpu_result_unchanged` and skips writing the dispatch's result
// back to guest memory at all, which surfaces as a frame that silently keeps stale pixels.
//
// The cases are chosen for what a wrong implementation would get wrong, not to re-walk the happy path:
//
//   * a difference in ANY of the four components of ANY word must raise the flag -- a comparison that
//     tested only .x would pass an equal-x/different-y case;
//   * the baseline must be updated to the new value, and ONLY where it differed;
//   * words at or beyond the push-constant count must be untouched even when they differ. That is the
//     bounds property the shader's own comment calls load-bearing, and it is the one a partial final
//     workgroup exercises for real: 256 invocations launch, fewer than 256 are in range;
//   * the flag buffer is bound as FOUR BYTES. The #1711 comment records that declaring it as a
//     uvec4 runtime array gave it ArrayStride 16 over a 4-byte range, making element 0 out of bounds
//     under robustBufferAccess -- which a conformant driver may discard, leaving the flag stuck at
//     zero. This test binds it the way production does and asserts the flag actually changes, so that
//     regression cannot come back silently.
//
// robustBufferAccess is requested because live_compute.cpp requires it; testing this shader without it
// would test a different device contract than the one it ships against.

#include "gpu/recompiler/spirv_builder.hpp"
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
bool operator==(const Uvec4& a, const Uvec4& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

// A minimal compute device. Deliberately standalone rather than reaching into the live backend: the
// point is to exercise the SHADER against a plain Vulkan contract, so a failure here indicts the
// module rather than the backend's resource machinery.
struct Device {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};

    bool init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return false;
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (!count) return false;
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceFeatures features{};
            vkGetPhysicalDeviceFeatures(candidate, &features);
            if (!features.robustBufferAccess) continue;
            uint32_t families = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, nullptr);
            std::vector<VkQueueFamilyProperties> properties(families);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, properties.data());
            for (uint32_t i = 0; i < families; ++i)
                if (properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { physical = candidate; family = i; break; }
            if (physical) break;
        }
        if (!physical) return false;
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = family; qci.queueCount = 1; qci.pQueuePriorities = &priority;
        VkPhysicalDeviceFeatures enabled{};
        enabled.robustBufferAccess = VK_TRUE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pEnabledFeatures = &enabled;
        if (vkCreateDevice(physical, &dci, nullptr, &device) != VK_SUCCESS) return false;
        vkGetDeviceQueue(device, family, 0, &queue);
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.queueFamilyIndex = family;
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        return vkCreateCommandPool(device, &cpi, nullptr, &pool) == VK_SUCCESS;
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
        if (type == UINT32_MAX) return false;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = requirements.size; mai.memoryTypeIndex = type;
        if (vkAllocateMemory(d.device, &mai, nullptr, &memory) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(d.device, buffer, memory, 0) != VK_SUCCESS) return false;
        return vkMapMemory(d.device, memory, 0, size, 0, &mapped) == VK_SUCCESS;
    }
    void destroy(Device& d) {
        if (memory) { vkUnmapMemory(d.device, memory); vkFreeMemory(d.device, memory, nullptr); }
        if (buffer) vkDestroyBuffer(d.device, buffer, nullptr);
        buffer = VK_NULL_HANDLE; memory = VK_NULL_HANDLE; mapped = nullptr;
    }
};

// The production pipeline shape: three storage bindings and a one-uint push constant.
struct ComparePipeline {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module_ = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

    bool create(Device& d) {
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
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * 64};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 64; dpi.poolSizeCount = 1; dpi.pPoolSizes = &size;
        return vkCreateDescriptorPool(d.device, &dpi, nullptr, &descriptor_pool) == VK_SUCCESS;
    }
};

// Runs the comparison once. `count` is in uvec4 units and goes in the push constant, exactly as
// live_compute.cpp sets it; the buffers may be LONGER than count, which is how the out-of-range cases
// are built. Returns the elapsed device-visible wall time so the same helper can answer the timing
// question without a second harness.
bool run_compare(Device& d, ComparePipeline& p, Buffer& a, Buffer& b, Buffer& flag,
                 uint32_t count, double* elapsed_ms = nullptr) {
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = p.descriptor_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &p.set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(d.device, &dai, &set) != VK_SUCCESS) return false;
    VkDescriptorBufferInfo infos[3] = {
        {a.buffer, 0, a.bytes}, {b.buffer, 0, b.bytes}, {flag.buffer, 0, flag.bytes}};
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
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count), &count);
    // The same group count the backend computes, so a sizing mistake here would be the backend's too.
    const uint32_t groups = (count + 255u) / 256u;
    vkCmdDispatch(cmd, groups, 1, 1);
    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(d.device, &fci, nullptr, &fence);
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
    vkFreeDescriptorSets(d.device, p.descriptor_pool, 1, &set);
    return ok;
}

}  // namespace

int main() {
    Device d;
    if (!d.init()) {
        // No conformant compute device with robustBufferAccess. Say so rather than passing silently:
        // a green run here must mean the shader was exercised.
        std::printf("[skip] no Vulkan compute device with robustBufferAccess\n");
        return 0;
    }
    ComparePipeline p;
    if (!p.create(d)) { std::printf("FAIL: could not build the compare pipeline\n"); return 1; }

    constexpr uint32_t kWords = 1024;          // uvec4s; 4 full workgroups
    const VkDeviceSize bytes = kWords * sizeof(Uvec4);
    Buffer a, b, flag;
    if (!a.create(d, bytes) || !b.create(d, bytes) || !flag.create(d, sizeof(uint32_t))) {
        std::printf("FAIL: buffer allocation\n");
        return 1;
    }
    auto* av = static_cast<Uvec4*>(a.mapped);
    auto* bv = static_cast<Uvec4*>(b.mapped);
    auto* fv = static_cast<uint32_t*>(flag.mapped);
    const auto fill = [&](Uvec4* dst, uint32_t seed) {
        for (uint32_t i = 0; i < kWords; ++i)
            dst[i] = {seed + i, seed + i * 3u, seed + i * 5u, seed + i * 7u};
    };

    // ---- equal: the flag must stay clear and the baseline must not move -------------------------
    fill(av, 11); fill(bv, 11); *fv = 0;
    std::vector<Uvec4> before(bv, bv + kWords);
    check(run_compare(d, p, a, b, flag, kWords), "equal: dispatch completed");
    check(*fv == 0, "equal: flag stays clear");
    check(std::memcmp(bv, before.data(), bytes) == 0, "equal: baseline untouched");

    // ---- one differing component, in each lane, must raise the flag ------------------------------
    // A comparison that tested only .x would pass three of these four.
    for (uint32_t lane = 0; lane < 4; ++lane) {
        fill(av, 11); fill(bv, 11); *fv = 0;
        uint32_t* component = &reinterpret_cast<uint32_t*>(&av[500])[lane];
        *component += 1;
        check(run_compare(d, p, a, b, flag, kWords), "single-component: dispatch completed");
        check(*fv == 1, "single-component difference in lane " + std::to_string(lane) + " raises the flag");
        check(bv[500] == av[500], "single-component: that word's baseline is updated");
        check(bv[499] == av[499] && bv[501] == av[501], "single-component: neighbours still equal");
    }

    // ---- only the differing word is written ------------------------------------------------------
    fill(av, 11); fill(bv, 22); *fv = 0;             // every word differs
    check(run_compare(d, p, a, b, flag, kWords), "all-different: dispatch completed");
    check(*fv == 1, "all-different: flag raised");
    check(std::memcmp(bv, av, bytes) == 0, "all-different: whole baseline adopts the new result");

    // ---- out of range: words at or beyond `count` are untouched even when they differ -------------
    // The bounds property the shader's own #1711 comment calls load-bearing. 300 is deliberately not
    // a multiple of 256, so the final workgroup launches 256 invocations of which only 44 are in
    // range -- the case a missing bounds check gets wrong in production and never in a tidy fixture.
    constexpr uint32_t kInRange = 300;
    fill(av, 11); fill(bv, 22); *fv = 0;
    std::vector<Uvec4> tail(bv + kInRange, bv + kWords);
    check(run_compare(d, p, a, b, flag, kInRange), "partial workgroup: dispatch completed");
    check(*fv == 1, "partial workgroup: in-range difference raises the flag");
    check(std::memcmp(bv, av, kInRange * sizeof(Uvec4)) == 0,
          "partial workgroup: every in-range word is updated");
    check(std::memcmp(bv + kInRange, tail.data(), tail.size() * sizeof(Uvec4)) == 0,
          "partial workgroup: words at and beyond count are UNTOUCHED");

    // ---- a difference only beyond count must NOT raise the flag ----------------------------------
    fill(av, 11); fill(bv, 11); *fv = 0;
    av[kInRange].x += 1;                              // the first word past the end
    check(run_compare(d, p, a, b, flag, kInRange), "beyond-count: dispatch completed");
    check(*fv == 0, "a difference beyond count does not raise the flag");

    // ---- count 0: nothing runs, nothing changes ---------------------------------------------------
    fill(av, 11); fill(bv, 22); *fv = 0;
    before.assign(bv, bv + kWords);
    check(run_compare(d, p, a, b, flag, 0), "zero count: dispatch completed");
    check(*fv == 0 && std::memcmp(bv, before.data(), bytes) == 0,
          "zero count: flag clear and baseline untouched");

    // ---- how much more a changed result costs than an unchanged one ------------------------------
    // Reported, never asserted, and reported as a MINIMUM over repetitions rather than a single pair.
    //
    // The question this answers is whether the all-different case is dominated by contention on the
    // changed-flag dword: every differing invocation issues a Device-scope OpAtomicExchange against
    // the SAME uint, so an all-different target could serialize one atomic per uvec4 on one cache
    // line, while an all-equal target issues none. Extra baseline traffic alone predicts about 1.5x.
    //
    // It was measured, and it does not settle that question. Guarding the exchange with a plain load
    // (skip when the flag already reads 1 -- sound, since the flag is only ever set and the stored
    // value is the constant 1) produced all-different times of 1.449/1.741/1.807 ms against
    // 2.012/1.784/2.280 ms unguarded: the spread WITHIN each arm is as large as the difference
    // between them. The guard was not adopted. Do not re-derive that from a single pair -- this
    // harness times a submit-plus-fence on the CPU, so it carries queue scheduling noise and needs
    // GPU timestamps to answer the question properly.
    //
    // The number is printed because whoever fuses this comparison into the retile shader will want a
    // before/after, and a min-of-N from a fixed fixture is a better baseline than a game route.
    {
        Buffer big_a, big_b;
        constexpr uint32_t kBig = 1u << 19;           // 8 MiB of uvec4
        const VkDeviceSize big_bytes = VkDeviceSize(kBig) * sizeof(Uvec4);
        if (big_a.create(d, big_bytes) && big_b.create(d, big_bytes)) {
            auto* ba = static_cast<Uvec4*>(big_a.mapped);
            auto* bb = static_cast<Uvec4*>(big_b.mapped);
            for (uint32_t i = 0; i < kBig; ++i) ba[i] = {i, i, i, i};
            std::memcpy(bb, ba, big_bytes);
            constexpr int kReps = 5;
            double equal_ms = 1e9, different_ms = 1e9, sample = 0;
            run_compare(d, p, big_a, big_b, flag, kBig, &sample);            // warm
            bool equal_clean = true;
            for (int i = 0; i < kReps; ++i) {
                *fv = 0;
                run_compare(d, p, big_a, big_b, flag, kBig, &sample);
                equal_clean = equal_clean && *fv == 0;
                if (sample < equal_ms) equal_ms = sample;
            }
            bool different_raised = true;
            for (int i = 0; i < kReps; ++i) {
                // Diverge at the TOP of every repetition, and only here. The baseline is equal to `a`
                // on entry -- either from the equal loop above, or because the previous repetition's
                // comparison adopted `a` into it -- so one XOR per iteration is what makes each run
                // measure the changed path. Diverging once before the loop as well would cancel this
                // one on the first iteration and silently measure the EQUAL path instead, which is
                // what the first version of this did; the flag assertion below caught it.
                for (uint32_t j = 0; j < kBig; ++j) bb[j].w ^= 0xffffffffu;
                *fv = 0;
                run_compare(d, p, big_a, big_b, flag, kBig, &sample);
                different_raised = different_raised && *fv == 1;
                if (sample < different_ms) different_ms = sample;
            }
            std::printf("[info] 8 MiB compare, min of %d: all-equal %.3f ms, all-different %.3f ms "
                        "(ratio %.2f) -- CPU-timed, carries scheduling noise\n",
                        kReps, equal_ms, different_ms, equal_ms > 0 ? different_ms / equal_ms : 0.0);
            check(equal_clean, "8 MiB all-equal comparison leaves the flag clear, every repetition");
            check(different_raised, "8 MiB all-different comparison raises the flag, every repetition");
            big_a.destroy(d); big_b.destroy(d);
        }
    }

    a.destroy(d); b.destroy(d); flag.destroy(d);
    if (failures) { std::printf("compute_result_compare: %d FAILED\n", failures); return 1; }
    std::printf("compute_result_compare: OK\n");
    return 0;
}
