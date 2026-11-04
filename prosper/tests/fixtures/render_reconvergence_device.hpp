#pragma once
#include "fixtures/render_reconvergence_shader.hpp"
#include <vulkan/vulkan.h>
#include <vector>
namespace prosper::test {
// Both linked feature objects outlive vkCreateDevice; they cannot be copied or moved.
class RenderReconvergenceDevice {
public:
    RenderReconvergenceDevice() = default;
    RenderReconvergenceDevice(const RenderReconvergenceDevice&) = delete;
    RenderReconvergenceDevice& operator=(const RenderReconvergenceDevice&) = delete;
    void note_extension(const char* name);
    RenderReconvergenceFeatures prepare(VkPhysicalDevice physical, VkDeviceCreateInfo& device,
        std::vector<const char*>& extensions,
        PFN_vkGetPhysicalDeviceFeatures2 query = vkGetPhysicalDeviceFeatures2);
private:
#ifdef VK_KHR_shader_maximal_reconvergence
    VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR maximal_{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR};
#endif
#ifdef VK_KHR_shader_quad_control
    VkPhysicalDeviceShaderQuadControlFeaturesKHR quads_{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR};
#endif
    bool maximal_advertised_ = false, quads_advertised_ = false, prepared_ = false;
};
} // namespace prosper::test
