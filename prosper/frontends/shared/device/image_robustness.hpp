#pragma once

// Vulkan acquisition of the storage-image device contract (#3531). The contract itself -- what the
// recompiler needs and why -- is in storage_image_contract.hpp; this header is only the part that
// must speak Vulkan: query the physical device and chain the feature into VkDeviceCreateInfo.

#include "shared/device/storage_image_contract.hpp"

#include <vulkan/vulkan.h>

#include <cstdio>

namespace prosper::frontend {

// Query `physical` for image robustness, chain VkPhysicalDeviceImageRobustnessFeatures into `dci`
// when it is advertised, and report what the resulting device will have. `storage` must outlive the
// vkCreateDevice call -- it becomes part of the pNext chain.
//
// The feature is core in Vulkan 1.3 and the runtime floor is 1.4 (kVulkanRuntimeVersion), so the
// promoted extension name is deliberately NOT requested: a 1.4 device need not still advertise
// `VK_EXT_image_robustness`, and gating on the string would disable the feature on devices that
// have it. This matches how the neighbouring descriptor-indexing and int64-atomics features are
// acquired.
//
// Logs in BOTH directions. A line that prints only on success makes the failing case silent, which
// is precisely how #3531 survived: there was no run anywhere whose log could contradict the
// assumption that robustness was on.
inline StorageImageDeviceFeatures acquire_storage_image_device_features(
    const char* owner, VkPhysicalDevice physical, bool read_without_format,
    bool write_without_format, VkPhysicalDeviceImageRobustnessFeatures& storage,
    VkDeviceCreateInfo& dci) {
    storage = VkPhysicalDeviceImageRobustnessFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES};
    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &storage;
    vkGetPhysicalDeviceFeatures2(physical, &features2);
    const StorageImageDeviceFeatures features = storage_image_device_features(
        storage.robustImageAccess == VK_TRUE, read_without_format, write_without_format);
    if (features.robust_image_access) {
        // Request exactly the one bit in use; the query struct carries no others.
        storage.pNext = const_cast<void*>(dci.pNext);
        dci.pNext = &storage;
    }
    std::fprintf(stderr,
                 "[%s] image robustness %s (core feature; recompiled storage-image reads are "
                 "out-of-bounds by design and need it)\n",
                 owner, features.robust_image_access ? "ENABLED" : "unavailable");
    if (!features.robust_image_access && read_without_format && write_without_format)
        std::fprintf(stderr,
                     "[%s] refusing recompiled storage-image work on this device: without "
                     "robustImageAccess an inactive lane's out-of-range OpImageRead is undefined\n",
                     owner);
    return features;
}

} // namespace prosper::frontend
