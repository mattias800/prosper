#pragma once
#include "gpu/recompiler/float_transport_config.hpp"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdio>

namespace prosper::frontend {

constexpr gpu::FloatTransportConfig select_float_transport_config(
        uint32_t effective_api, bool feature_supported, bool preserve_float32,
        bool request_feature = true) {
    return gpu::select_float_transport_request(effective_api >= VK_API_VERSION_1_4,
                                               feature_supported, preserve_float32, request_feature);
}

// The result describes the REQUEST. Retain/publish only after vkCreateDevice succeeds.
// The runtime requires Vulkan 1.4; older standalone probes retain the implicit profile.
inline gpu::FloatTransportConfig acquire_float_transport_device_features(
        const char* owner, VkPhysicalDevice physical, uint32_t instance_api,
        VkPhysicalDeviceShaderFloatControls2Features& requested, VkDeviceCreateInfo& create,
        bool request_feature = true) {
    requested = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES};
    VkPhysicalDeviceProperties base{};
    vkGetPhysicalDeviceProperties(physical, &base);
    const uint32_t effective = std::min(instance_api, base.apiVersion);
    bool feature = false, preserve = false;
    if (effective >= VK_API_VERSION_1_4) {
        VkPhysicalDeviceShaderFloatControls2Features supported{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &supported;
        vkGetPhysicalDeviceFeatures2(physical, &features);
        feature = supported.shaderFloatControls2 == VK_TRUE;
        VkPhysicalDeviceFloatControlsProperties properties{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
        VkPhysicalDeviceProperties2 queried{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        queried.pNext = &properties;
        vkGetPhysicalDeviceProperties2(physical, &queried);
        preserve = properties.shaderSignedZeroInfNanPreserveFloat32 == VK_TRUE;
    }
    const auto profile = select_float_transport_config(effective, feature, preserve, request_feature);
    if (profile.explicit_nonfinite32()) {
        requested.shaderFloatControls2 = VK_TRUE;
        requested.pNext = const_cast<void*>(create.pNext);
        create.pNext = &requested;
    }
    std::fprintf(stderr,
        "[%s] float transport profile=%s REQUESTED (api=%u.%u shaderFloatControls2=%d "
        "shaderSignedZeroInfNanPreserveFloat32=%d request=%d); enabled witness requires "
        "successful device creation\n",
        owner, profile.explicit_nonfinite32() ? "explicit-nonfinite32" : "implicit",
        VK_API_VERSION_MAJOR(effective), VK_API_VERSION_MINOR(effective), feature, preserve,
        request_feature);
    return profile;
}

using gpu::float_transport_module_supported;

} // namespace prosper::frontend
