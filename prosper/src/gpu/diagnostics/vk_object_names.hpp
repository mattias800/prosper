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
//  1. It must work WITHOUT the validation layer, on EVERY instance that creates nameable objects --
//     not just the renderer's. prosper builds more than one: the live compute backend creates its own
//     private instance when it does not adopt the renderer's device, and naming there was silently
//     inert until #3597 armed it too. `VK_EXT_debug_utils` used to be requested only
//     inside the `if (validation_enabled)` branch, so naming would have silently done nothing in an
//     ordinary run -- which is the run people actually capture. The instance now requests it
//     unconditionally and tolerates its absence.
//  2. It must never fail a run. The entry point is resolved once and is allowed to be null; every
//     call through a null pointer is a no-op, and no result code is propagated anywhere. An object
//     name is a convenience, and a run that died because it could not name a shader module would be
//     a worse tool than no names at all.
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <vulkan/vulkan.h>

namespace prosper::gpu {

// Resolved once per device. `vkGetDeviceProcAddr` returns null when the instance did not enable
// VK_EXT_debug_utils, which is the ordinary case on a driver or loader that does not offer it --
// not an error, and deliberately not logged on every call.
inline PFN_vkSetDebugUtilsObjectNameEXT vk_object_name_fn(VkDevice device) {
    // THREAD_LOCAL, which removes the race rather than synchronising it. An earlier version kept the
    // device and the function pointer in two independent atomics, and that pair TEARS: a thread
    // matching the stale device could read a pointer another thread had already replaced for a
    // different device, then call a foreign dispatch entry. Benign with one VkDevice, not benign in
    // general, and this file's contract is that naming must NEVER fail a run (#3597).
    //
    // Holding both halves in one atomic was the obvious repair and does not work here: two pointers
    // is 16 bytes, and `std::atomic<>::is_always_lock_free` is FALSE for that on this target -- the
    // static_assert that said so is why this is thread_local instead. A locking atomic on a
    // diagnostic path would be a worse trade than a per-thread copy.
    //
    // Cost is one pointer pair per thread that ever names an object, and a re-resolve the first time
    // each thread sees a device. Naming happens once per object CREATION, never per draw or dispatch,
    // so even resolving every call would be affordable; the cache is politeness, not necessity.
    thread_local VkDevice cached_device = VK_NULL_HANDLE;
    thread_local PFN_vkSetDebugUtilsObjectNameEXT cached_fn = nullptr;
    if (device != cached_device) {
        cached_fn = device ? (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(
                                 device, "vkSetDebugUtilsObjectNameEXT")
                           : nullptr;
        cached_device = device;
    }
    return cached_fn;
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
