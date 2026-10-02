// gpu_labels_vk.hpp -- guest-meaningful command labels around every draw and dispatch (D5).
//
// WHY. prosper already names guest SHADER MODULES (vk_object_names.hpp), but a capture in RenderDoc or
// RGP still shows an anonymous stream of draws: nothing says which guest submit, which draw index or
// which program a given event is. `PROSPER_GPU_LABELS` brackets every instrumented draw and dispatch
// with a VK_EXT_debug_utils label whose text is `format_breadcrumb_site`: the SAME spelling a
// breadcrumb verdict uses, so a marker in an external tool and a `[gpu-breadcrumb]` line can be matched
// by eye. Numbers and hashes only; no guest string ever reaches a label.
//
// OFF BY DEFAULT AND FREE WHEN OFF. The switch is cached, the entry points are not loaded, no text is
// formatted and no site is built unless it is on. An absent extension makes every call a no-op: a label
// is a convenience, and a run that failed for want of one would be a worse tool than none (the same
// contract as vk_object_names.hpp). VK_EXT_debug_utils is already requested on every instance.
//
// THE DISPATCH TABLE IS INJECTABLE so the exact sequence of calls and the label text are unit-tested
// without a device (tests/gpu/diagnostics/test_gpu_labels.cpp). That proves what is recorded, not how
// RenderDoc or RGP display it: that is a local look at a real capture and is stated in the PR.
#pragma once

#include "diagnostics/env_cache.hpp"
#include "gpu/diagnostics/gpu_breadcrumbs.hpp"

#include <vulkan/vulkan.h>

#include <string>

namespace prosper::gpu {

inline bool gpu_labels_requested() { return PROSPER_ENV_ON("PROSPER_GPU_LABELS"); }

struct LabelDispatch {
    PFN_vkCmdBeginDebugUtilsLabelEXT cmd_begin = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT cmd_end = nullptr;
    bool usable() const { return cmd_begin && cmd_end; }
};

// Resolved once per device per thread. Null entry points when the instance did not enable
// VK_EXT_debug_utils: not an error, and deliberately not logged.
inline LabelDispatch label_dispatch_for(VkDevice device) {
    thread_local VkDevice cached_device = VK_NULL_HANDLE;
    thread_local LabelDispatch cached{};
    if (device != cached_device) {
        cached = {};
        if (device) {
            cached.cmd_begin = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
                vkGetDeviceProcAddr(device, "vkCmdBeginDebugUtilsLabelEXT"));
            cached.cmd_end = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
                vkGetDeviceProcAddr(device, "vkCmdEndDebugUtilsLabelEXT"));
        }
        cached_device = device;
    }
    return cached;
}

// Draws read blue and dispatches green, so the two are told apart at a glance in a timeline.
inline void gpu_label_begin(const LabelDispatch& dispatch, VkCommandBuffer command,
                            const BreadcrumbSite& site) {
    if (!dispatch.usable() || !command) return;
    const std::string text = format_breadcrumb_site(site);
    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = text.c_str();
    const bool is_draw = site.kind == BreadcrumbKind::draw;
    label.color[0] = 0.25f;
    label.color[1] = is_draw ? 0.55f : 0.80f;
    label.color[2] = is_draw ? 0.95f : 0.35f;
    label.color[3] = 1.0f;
    dispatch.cmd_begin(command, &label);
}

inline void gpu_label_end(const LabelDispatch& dispatch, VkCommandBuffer command) {
    if (!dispatch.usable() || !command) return;
    dispatch.cmd_end(command);
}

// Convenience for the call sites: resolve from the device, then begin. Returns whether a label was
// opened, so the matching end is recorded only when its begin was.
inline bool gpu_label_begin(VkDevice device, VkCommandBuffer command, const BreadcrumbSite& site) {
    const LabelDispatch dispatch = label_dispatch_for(device);
    if (!dispatch.usable()) return false;
    gpu_label_begin(dispatch, command, site);
    return true;
}
inline void gpu_label_end(VkDevice device, VkCommandBuffer command) {
    gpu_label_end(label_dispatch_for(device), command);
}

} // namespace prosper::gpu
