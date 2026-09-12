// float_controls_probe.hpp — the Vulkan half of the SignedZeroInfNanPreserve contract (#3479/#3561).
//
// The contract itself — what the recompiler declares, why, and what the neutral state means — is in
// gpu/recompiler/rdna2_to_spirv.hpp beside publish_float_controls_support(). This header is only the
// part that has to speak Vulkan: ask a physical device whether it can execute a module declaring the
// mode, and publish that answer to the recompiler.
//
// It lives here, next to host_read_barrier.hpp, for the same reason that one does: the offscreen
// render backend (tests/fixtures/render_runner.h) and the live compute backend
// (frontends/shared/live/live_compute.cpp) both own Vulkan devices that execute the SAME recompiled
// modules, and two spellings of a device query is exactly how the two paths diverge. #3531's
// image-robustness helper was written after that had already happened once.
//
// A caller that gets this wrong FAILS SAFE. Publishing false (or never publishing) costs the Inf
// guarantee and emits a module every device can legally compile; the failure this guards against is
// the other direction — a module no device may legally compile, which drivers honour anyway and no
// test reports.
//
// ONE RESIDUAL LIMIT, stated rather than hidden: the verdict is process-global and read at emit
// time, so a module emitted before a device publishes gets the neutral answer. That direction is
// harmless — it loses the guarantee, never the legality. The harmful direction would need a
// publisher to RAISE the verdict after a module was emitted, and the AND is what makes that
// impossible: `supported` only ever goes true -> false. Publish at device creation, which is what
// both call sites do and what puts every publisher ahead of the first guest draw.
#pragma once

#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>

namespace prosper::gpu {

// Measure `physical` and publish the verdict. `instance_api_version` is the VkApplicationInfo
// apiVersion this device's instance was created with, and `float_controls_extension_enabled` is
// whether VK_KHR_shader_float_controls was ENABLED at vkCreateDevice (not merely advertised).
// Returns what was published: true only when a guest module may declare the mode on this device.
//
// Both halves of the Vulkan requirement are checked, because they fail independently:
//   -08742  the extension may be declared at all: effective API >= 1.2, or the extension enabled.
//   -08740  the device actually preserves signed zero / Inf / NaN for 32-bit float.
// The effective version is min(instance, physical device) — the same quantity the validation layer
// computes — so an instance that asked for 1.4 on a 1.1 physical device does not pass the first.
inline bool publish_device_float_controls(const char* owner, VkPhysicalDevice physical,
                                          uint32_t instance_api_version,
                                          bool float_controls_extension_enabled) {
    if (!physical) return false;
    VkPhysicalDeviceProperties base{};
    vkGetPhysicalDeviceProperties(physical, &base);
    const uint32_t effective = std::min(instance_api_version, base.apiVersion);
    const bool permitted =
        effective >= VK_API_VERSION_1_2 || float_controls_extension_enabled;

    bool preserve_float32 = false;
    if (permitted) {
        // Chain the struct only when the device can answer it. A pNext an implementation does not
        // recognize is skipped and leaves the struct as written — i.e. reads back as VK_FALSE — so
        // querying unconditionally would be safe but would also be indistinguishable from a genuine
        // "no", and the two have different diagnoses.
        VkPhysicalDeviceFloatControlsProperties float_controls{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
        VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties2.pNext = &float_controls;
        vkGetPhysicalDeviceProperties2(physical, &properties2);
        preserve_float32 = float_controls.shaderSignedZeroInfNanPreserveFloat32 == VK_TRUE;
    }

    publish_float_controls_support(preserve_float32, permitted);
    const bool usable = preserve_float32 && permitted;
    // Both directions, as everywhere else in this family: a line that prints only on success makes
    // the failing case silent, and a silent failing case is how #3561's unmeasured claim survived a
    // full green test run. publish_float_controls_support() prints the diagnosis when the answer is
    // no; this line is the per-device record of what was asked and answered.
    std::fprintf(stderr,
                 "[%s] SignedZeroInfNanPreserve %s (api %u.%u, extension %s, "
                 "shaderSignedZeroInfNanPreserveFloat32 %s)\n",
                 owner, usable ? "ENABLED" : "unavailable",
                 VK_API_VERSION_MAJOR(effective), VK_API_VERSION_MINOR(effective),
                 float_controls_extension_enabled ? "enabled" : "not enabled",
                 permitted ? (preserve_float32 ? "VK_TRUE" : "VK_FALSE") : "not queryable");
    return usable;
}

} // namespace prosper::gpu
