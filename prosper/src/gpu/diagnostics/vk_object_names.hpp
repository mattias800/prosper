#pragma once
// Vulkan object naming (#3578) -- give a guest shader a name that survives into RenderDoc, RGP and
// the validation layer.
//
// Why this exists. prosper named NO Vulkan object, so every external tool identified a guest shader
// by its `VkShaderModule` handle: a number that changes run to run and means nothing outside the
// process that produced it. The project routes real investigation through exactly those tools --
// `docs/GPU_PROFILING_EXTERNAL.md` is their recipe book, `tools/pixel_history/` walks a RenderDoc
// pixel history, and #3321 made captures schedulable so an AGENT takes them with nobody at the
// keyboard to remember which handle was which. The identity to name them with already existed and was
// already being used as a memo key (`fs_identity`, `compute_dispatch_code_addr`); it just never
// reached Vulkan.
//
// Two properties this must have, both of them stated as requirements rather than discovered later:
//
//  1. It must work WITHOUT the validation layer. `VK_EXT_debug_utils` used to be requested only
//     inside the `if (validation_enabled)` branch, so naming would have silently done nothing in an
//     ordinary run -- which is the run people actually capture. The instance now requests it
//     unconditionally and tolerates its absence.
//  2. It must never fail a run. The entry point is resolved once and is allowed to be null; every
//     call through a null pointer is a no-op, and no result code is propagated anywhere. An object
//     name is a convenience, and a run that died because it could not name a shader module would be
//     a worse tool than no names at all.
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <vulkan/vulkan.h>

namespace prosper::gpu {

// Resolved once per device. `vkGetDeviceProcAddr` returns null when the instance did not enable
// VK_EXT_debug_utils, which is the ordinary case on a driver or loader that does not offer it --
// not an error, and deliberately not logged on every call.
inline PFN_vkSetDebugUtilsObjectNameEXT vk_object_name_fn(VkDevice device) {
    // Atomic because prosper runs two guest submit threads and this is an inline function shared
    // across translation units, so the cache is genuinely reachable concurrently. The worst outcome
    // of a race here would be a missed name rather than a crash -- but this file's contract is that
    // naming must NEVER fail a run, and "probably benign" is not that. Resolution is idempotent, so
    // two threads racing to resolve simply store the same pointer; relaxed ordering is enough because
    // neither value guards any other memory.
    static std::atomic<VkDevice> cached_device{VK_NULL_HANDLE};
    static std::atomic<PFN_vkSetDebugUtilsObjectNameEXT> cached_fn{nullptr};
    if (device != cached_device.load(std::memory_order_relaxed)) {
        const PFN_vkSetDebugUtilsObjectNameEXT resolved =
            device ? (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(
                         device, "vkSetDebugUtilsObjectNameEXT")
                   : nullptr;
        cached_fn.store(resolved, std::memory_order_relaxed);
        cached_device.store(device, std::memory_order_relaxed);
        return resolved;
    }
    return cached_fn.load(std::memory_order_relaxed);
}

// Name one object. Silently does nothing when the extension is absent.
inline void vk_name_object(VkDevice device, VkObjectType type, uint64_t handle, const char* name) {
    if (!device || !handle || !name) return;
    const PFN_vkSetDebugUtilsObjectNameEXT fn = vk_object_name_fn(device);
    if (!fn) return;
    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name;
    (void)fn(device, &info);   // a naming failure is never a run failure
}

// printf-style convenience. The format happens at OBJECT CREATION, never in a draw or dispatch hot
// path -- a name is built once per module, not once per use.
inline void vk_name_objectf(VkDevice device, VkObjectType type, uint64_t handle,
                            const char* fmt, ...) {
    if (!device || !handle || !fmt) return;
    if (!vk_object_name_fn(device)) return;   // build no string when nothing is listening
    char buf[192];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    vk_name_object(device, type, handle, buf);
}

// A content digest of a SPIR-V module, FOR NAMING ONLY -- never an identity anything depends on.
//
// It is here because the identities prosper already carries are not sufficient on their own.
// `DrawItem::vs_identity`/`fs_identity` are a monotonic counter from the exact shader-recompile cache
// (`gpu_executor.cpp`, `cache.next_identity++`): process-unique and never reused, which is exactly
// what a memo key needs, but it is NOT a content hash and NOT the guest code address. Two
// consequences for a name. It differs between two runs of the same title, so it cannot correlate a
// capture taken today against a log taken yesterday; and it is documented as **zero** for the
// external/replay paths -- `gpu_execute.hpp:83-85` -- which is precisely how `gpu_replay` opens the
// bundles this naming exists to make readable. A name carrying only the counter would read `id=0` for
// every module in a replay.
//
// So a name carries both: the counter, which ties a capture to the same run's `PROSPER_DBG` lines,
// and this digest, which is stable across runs and still identifies the module when the counter is 0.
inline uint64_t vk_name_spirv_digest(const uint32_t* words, size_t count) {
    uint64_t h = 1469598103934665603ull;                 // FNV-1a 64 offset basis
    for (size_t i = 0; i < count; ++i) {
        h ^= words[i];
        h *= 1099511628211ull;
    }
    return h;
}

}  // namespace prosper::gpu
