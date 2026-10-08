#pragma once
// ADR 0028 glue for live_compute.cpp: build the Wave64 route facts from this device and a compute
// item, so the two `[wave64-unsupported]` decline sites can name the route the program's analysis
// selects, and so a module compiled through the exchange dispatcher is checked against the device
// limits only this frontend knows (shared memory, workgroup size).
//
// A header, templated on the context type, because VulkanComputeContext lives in live_compute.cpp
// (already at its architecture-ratchet line cap) and the limits are read from its physical device.
#include <vulkan/vulkan.h>

#include <cstdio>
#include <map>
#include <mutex>

#include "diagnostics/perf/wave64_refusal.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/compute_wave_route.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

namespace prosper::frontend {

struct ComputeWaveRouteText {
    char text[96] = "unanalyzed";
};

// maxComputeSharedMemorySize and maxComputeWorkGroupInvocations of `physical`, queried once.
inline void compute_wave_device_limits(VkPhysicalDevice physical, uint32_t& shared_bytes,
                                       uint32_t& invocations) {
    static std::mutex mutex;
    static std::map<VkPhysicalDevice, std::pair<uint32_t, uint32_t>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(physical);
    if (it == cache.end()) {
        VkPhysicalDeviceProperties properties{};
        if (physical) vkGetPhysicalDeviceProperties(physical, &properties);
        it =
            cache
                .emplace(physical, std::make_pair(properties.limits.maxComputeSharedMemorySize,
                                                  properties.limits.maxComputeWorkGroupInvocations))
                .first;
    }
    shared_bytes = it->second.first;
    invocations = it->second.second;
}

struct ComputeWaveLimits {
    uint32_t shared_bytes = 0, invocations = 0;
};

inline ComputeWaveLimits compute_wave_device_limits(VkPhysicalDevice physical) {
    ComputeWaveLimits limits;
    compute_wave_device_limits(physical, limits.shared_bytes, limits.invocations);
    return limits;
}

template <class Ctx>
prosper::gpu::ComputeWaveHost compute_wave_host(const Ctx& ctx,
                                                const prosper::gpu::ComputeItem& item,
                                                const ComputeWaveLimits& limits) {
    prosper::gpu::ComputeWaveHost host;
    host.guest_wave = item.recompile_config_available ? item.recompile_config.wave_size : 64u;
    host.host_subgroup_min = ctx.min_native_subgroup_size;
    host.host_subgroup_max = ctx.max_native_subgroup_size;
    if (!host.host_subgroup_min && ctx.subgroup_size)
        host.host_subgroup_min = host.host_subgroup_max = ctx.subgroup_size;
    if (item.recompile_config_available) {
        const auto& c = item.recompile_config;
        host.local_x = c.local_x;
        host.local_y = c.local_y;
        host.local_z = c.local_z;
        host.guest_lds_bytes = c.lds_bytes;
        if (c.exact_thread_extent)
            host.partial_workgroup =
                (c.threads_x % c.local_x) || (c.threads_y % c.local_y) || (c.threads_z % c.local_z);
    }
    host.max_shared_bytes = limits.shared_bytes;
    host.max_workgroup_invocations = limits.invocations;
    host.native_contract = item.required_subgroup_size && ctx.native_subgroup_contract &&
                           item.required_subgroup_size >= ctx.min_native_subgroup_size &&
                           item.required_subgroup_size <= ctx.max_native_subgroup_size;
    return host;
}

// `<route>:<reason>` for the `route=` field of a refusal line. Analysis only. An item without
// analysis facts (capture replay, hand-built records) reports `unanalyzed`.
template <class Ctx>
ComputeWaveRouteText compute_wave_route_text(const Ctx& ctx, const prosper::gpu::ComputeItem& item,
                                             const ComputeWaveLimits& limits) {
    ComputeWaveRouteText out;
    if (!item.wave_ops) return out;
    const auto decision = prosper::gpu::select_compute_wave_route(
        *item.wave_ops, compute_wave_host(ctx, item, limits));
    std::snprintf(out.text, sizeof out.text, "%s:%s",
                  prosper::gpu::compute_wave_route_name(decision.route), decision.reason);
    return out;
}

// A module compiled through the exchange dispatcher keeps its scratch in workgroup memory beside
// the guest's own LDS, and its barriers need whole guest waves in every workgroup. Those are
// limits this frontend knows and the recompiler does not: returns the decline reason, after
// printing the refusal, or nullptr when the dispatch may proceed.
template <class Ctx>
const char* exchange_limit(const Ctx& ctx, const prosper::gpu::ComputeItem& item,
                           const ComputeWaveLimits& limits) {
    if (!prosper::gpu::compute_spirv_wave64_exchange(item.spirv)) return nullptr;
    const char* why =
        prosper::gpu::compute_exchange_launch_refusal(compute_wave_host(ctx, item, limits));
    if (!why) return nullptr;
    std::fprintf(
        stderr,
        "[compute] program 0x%llx compiled for the Wave64 exchange but %s -> dispatch skipped\n",
        static_cast<unsigned long long>(item.code_addr), why);
    char route[96];
    std::snprintf(route, sizeof route, "refused:%s", why);
    prosper::diagnostics::perf::note_unsupported_wave64(
        prosper::diagnostics::perf::Wave64Refusal::ComputeSubgroup,
        item.recompile_config_available ? item.recompile_config.wave_size : 64u, item.code_addr, 0,
        UINT32_MAX, ctx.min_native_subgroup_size, ctx.max_native_subgroup_size, {}, route);
    return "wave64-exchange-limit";
}

// The two entry points live_compute.cpp calls: the limits come from the context's own device.
template <class Ctx>
ComputeWaveRouteText compute_wave_route_text(const Ctx& ctx,
                                             const prosper::gpu::ComputeItem& item) {
    return compute_wave_route_text(ctx, item, compute_wave_device_limits(ctx.physical));
}

template <class Ctx>
const char* exchange_limit(const Ctx& ctx, const prosper::gpu::ComputeItem& item) {
    return exchange_limit(ctx, item, compute_wave_device_limits(ctx.physical));
}

}  // namespace prosper::frontend
