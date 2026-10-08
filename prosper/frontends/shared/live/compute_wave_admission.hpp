#pragma once
// ADR 0028 glue for live_compute.cpp: name the route the Wave64 analysis would select for a refused
// compute dispatch, as the `candidate-route=` / `candidate-reason=` fields of `[wave64-unsupported]`
// (the single `route=` field and its vocabulary belong to #4754; a declined dispatch is `refused`).
//
// Everything here runs only when the refusal line will actually print: the line's dedupe happens
// inside note_unsupported_wave64, which calls back into this code afterwards. The refusal path
// itself therefore pays nothing per dispatch.
//
// A header, templated on the context type, because VulkanComputeContext lives in live_compute.cpp
// (at its architecture-ratchet line cap) and the limits are read from its physical device.
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

#include "diagnostics/perf/wave64_refusal.hpp"
#include "gpu/execute/compute_program_facts.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/compute_wave_route.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

namespace prosper::frontend {

struct ComputeWaveLimits {
    uint32_t shared_bytes = 0, invocations = 0;
};

// maxComputeSharedMemorySize and maxComputeWorkGroupInvocations. A direct query, no cache or lock:
// it runs only when a refusal line prints.
inline ComputeWaveLimits query_compute_wave_limits(VkPhysicalDevice physical) {
    VkPhysicalDeviceProperties properties{};
    if (physical) vkGetPhysicalDeviceProperties(physical, &properties);
    return {properties.limits.maxComputeSharedMemorySize,
            properties.limits.maxComputeWorkGroupInvocations};
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

// The candidate for given facts. A `native` decision is NOT printed on a declined dispatch: it
// means "runs as is", which contradicts the refusal beside it (instrument trap 291 in reverse), so
// the candidate is left empty and the reason is the whole story.
template <class Ctx>
prosper::diagnostics::perf::Wave64Candidate
compute_wave_candidate(const Ctx& ctx, const prosper::gpu::ComputeItem& item,
                       const prosper::gpu::ComputeWaveOpFacts& facts,
                       const ComputeWaveLimits& limits) {
    prosper::diagnostics::perf::Wave64Candidate out;
    const auto decision =
        prosper::gpu::select_compute_wave_route(facts, compute_wave_host(ctx, item, limits));
    if (decision.route == prosper::gpu::ComputeWaveRoute::Native) return out;
    std::snprintf(out.route, sizeof out.route, "%s",
                  prosper::gpu::compute_wave_route_name(decision.route));
    std::snprintf(out.reason, sizeof out.reason, "%s", decision.reason);
    return out;
}

template <class Ctx>
struct ComputeWaveCandidateArg {
    const Ctx* ctx;
    const prosper::gpu::ComputeItem* item;
};

// The callback note_unsupported_wave64 invokes once per printed line. The inventory comes from the
// memoized program facts, computed here on first use; an item without a program length reports
// `unanalyzed` rather than an empty inventory.
template <class Ctx>
prosper::diagnostics::perf::Wave64Candidate compute_wave_candidate_thunk(const void* p) {
    const auto& arg = *static_cast<const ComputeWaveCandidateArg<Ctx>*>(p);
    const auto& item = *arg.item;
    prosper::diagnostics::perf::Wave64Candidate out;
    if (!item.code_dwords || !item.code_addr) {
        std::snprintf(out.route, sizeof out.route, "refused");
        std::snprintf(out.reason, sizeof out.reason, "unanalyzed");
        return out;
    }
    // Runs under note_unsupported_wave64's mutex, once per printed line. The peek is side-effect free.
    const auto facts = prosper::gpu::compute_program_facts_peek(
        reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(item.code_addr)), item.code_dwords,
        item.code_addr);
    return compute_wave_candidate(*arg.ctx, item, facts->wave_ops(),
                                  query_compute_wave_limits(arg.ctx->physical));
}

// Both compute decline sites of live_compute.cpp.
template <class Ctx>
void note_compute_wave_refusal(const Ctx& ctx, const prosper::gpu::ComputeItem& item,
                               uint32_t guest_wave) {
    const ComputeWaveCandidateArg<Ctx> arg{&ctx, &item};
    prosper::diagnostics::perf::note_unsupported_wave64(
        prosper::diagnostics::perf::Wave64Refusal::ComputeSubgroup, guest_wave, item.code_addr, 0,
        UINT32_MAX, ctx.min_native_subgroup_size, ctx.max_native_subgroup_size, {},
        &compute_wave_candidate_thunk<Ctx>, &arg);
}

// ---- ADR 0028 route 3 admission (PROSPER_WAVE64_EXCHANGE, #4753) ----

// A module compiled through the exchange dispatcher keeps its scratch in workgroup memory beside the
// guest's own LDS, and its barriers need whole guest waves per workgroup. The route analysis decides
// whether the program's cross-lane operations may be exchanged at all (here with the dispatcher's
// own semantics, see ComputeWaveHost::exchange_dispatcher); the launch limits are the device's.
// Returns the REASON the dispatch must be declined (a literal naming the condition that fired), or
// nullptr. `facts` null means unanalyzed, which is itself a refusal.
template <class Ctx>
const char* exchange_limit(const Ctx& ctx, const prosper::gpu::ComputeItem& item,
                           const prosper::gpu::ComputeWaveOpFacts* facts,
                           const ComputeWaveLimits& limits) {
    if (!prosper::gpu::compute_spirv_wave64_exchange(item.spirv)) return nullptr;
    auto host = compute_wave_host(ctx, item, limits);
    host.exchange_dispatcher = true;
    const char* why = nullptr;
    if (!facts) {
        why = "unanalyzed";
    } else {
        const auto decision = prosper::gpu::select_compute_wave_route(*facts, host);
        if (decision.route == prosper::gpu::ComputeWaveRoute::Refused ||
            decision.route == prosper::gpu::ComputeWaveRoute::NeedsNLanes)
            why = decision.reason;
    }
    if (!why) return nullptr;
    std::fprintf(
        stderr,
        "[compute] program 0x%llx compiled for the Wave64 exchange but %s -> dispatch skipped\n",
        static_cast<unsigned long long>(item.code_addr), why);
    return why;
}

// The entry point live_compute.cpp calls on each dispatch. Ordinary modules return immediately. The
// device limits are read once (one device per process).
template <class Ctx>
const char* exchange_admit(const Ctx& ctx, const prosper::gpu::ComputeItem& item,
                           const ComputeWaveLimits& limits) {
    if (!item.exchange_facts) return nullptr;
    if (!prosper::gpu::compute_spirv_wave64_exchange(item.spirv)) return nullptr;
    const char* why = exchange_limit(ctx, item, &item.exchange_facts->wave_ops(), limits);
    if (why) return why;
    // Admitted: count it (lock-free, per dispatch) and name the route once per program, so the A/B
    // in #4753 can see which dispatches took it. The announcement takes a mutex, so only the first
    // sighting of a program reaches it.
    prosper::diagnostics::perf::note_wave64_route(
        prosper::diagnostics::perf::Wave64Route::WorkgroupExchange, true, 64);
    if (!item.exchange_facts->exchange_announced.exchange(true, std::memory_order_relaxed))
        prosper::diagnostics::perf::announce_wave64_route(
            prosper::diagnostics::perf::Wave64Route::WorkgroupExchange, true, 64, item.code_addr);
    return nullptr;
}

template <class Ctx>
const char* exchange_limit(const Ctx& ctx, const prosper::gpu::ComputeItem& item) {
    // FIRST and cheap: only a dispatch the executor gave an exchange width (the switch is on) carries
    // facts. With the switch off this is a null-pointer test, never a walk of the SPIR-V module.
    if (!item.exchange_facts) return nullptr;
    static const ComputeWaveLimits limits = query_compute_wave_limits(ctx.physical);   // one device
    return exchange_admit(ctx, item, limits);
}

}   // namespace prosper::frontend
