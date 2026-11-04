#pragma once
#include <vulkan/vulkan.h>
#include <vector>

namespace prosper::test {
// Owns the enabled feature until vkCreateDevice returns. The linked object cannot move.
class RenderReconvergenceDevice {
public:
    RenderReconvergenceDevice() = default;
    RenderReconvergenceDevice(const RenderReconvergenceDevice&) = delete;
    RenderReconvergenceDevice& operator=(const RenderReconvergenceDevice&) = delete;
    bool prepare(VkPhysicalDevice physical, bool advertised, VkDeviceCreateInfo& device,
                 std::vector<const char*>& extensions,
                 PFN_vkGetPhysicalDeviceFeatures2 query = vkGetPhysicalDeviceFeatures2);

private:
#ifdef VK_KHR_shader_maximal_reconvergence
    VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR feature_{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR};
#endif
    bool prepared_ = false;
};
} // namespace prosper::test
