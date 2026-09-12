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
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <vulkan/vulkan.h>

namespace prosper::gpu {

// Resolved once per device. `vkGetDeviceProcAddr` returns null when the instance did not enable
// VK_EXT_debug_utils, which is the ordinary case on a driver or loader that does not offer it --
// not an error, and deliberately not logged on every call.
inline PFN_vkSetDebugUtilsObjectNameEXT vk_object_name_fn(VkDevice device) {
    static VkDevice cached_device = VK_NULL_HANDLE;
    static PFN_vkSetDebugUtilsObjectNameEXT cached_fn = nullptr;
    if (device != cached_device) {
        cached_device = device;
        cached_fn = device ? (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(
                                 device, "vkSetDebugUtilsObjectNameEXT")
                           : nullptr;
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

}  // namespace prosper::gpu
