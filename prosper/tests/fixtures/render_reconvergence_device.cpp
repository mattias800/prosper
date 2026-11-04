#include "fixtures/render_reconvergence_device.hpp"
#include "fixtures/render_reconvergence_shader.hpp"
#include <cstring>
#include <vector>
namespace prosper::test {
void RenderReconvergenceDevice::note_extension(const char* name) {
    if (!name) return;
    maximal_advertised_ |= std::strcmp(name, "VK_KHR_shader_maximal_reconvergence") == 0;
    quads_advertised_ |= std::strcmp(name, "VK_KHR_shader_quad_control") == 0;
}
RenderReconvergenceFeatures
RenderReconvergenceDevice::prepare(VkPhysicalDevice physical, VkDeviceCreateInfo& device,
                                   std::vector<const char*>& extensions,
                                   PFN_vkGetPhysicalDeviceFeatures2 query) {
    RenderReconvergenceFeatures enabled;
    if (prepared_) return enabled;
    prepared_ = true;
    if (!physical || !query) return enabled;
#ifdef VK_KHR_shader_maximal_reconvergence
    if (maximal_advertised_) {
        VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                            &maximal_};
        query(physical, &supported);
        if (maximal_.shaderMaximalReconvergence) {
            maximal_ = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR};
            maximal_.shaderMaximalReconvergence = VK_TRUE;
            maximal_.pNext = const_cast<void*>(device.pNext);
            device.pNext = &maximal_;
            extensions.push_back(VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME);
            enabled.maximal = true;
        }
    }
#endif
#ifdef VK_KHR_shader_quad_control
    if (enabled.maximal && quads_advertised_) {
        VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &quads_};
        query(physical, &supported);
        if (quads_.shaderQuadControl) {
            quads_ = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR};
            quads_.shaderQuadControl = VK_TRUE;
            quads_.pNext = const_cast<void*>(device.pNext);
            device.pNext = &quads_;
            extensions.push_back(VK_KHR_SHADER_QUAD_CONTROL_EXTENSION_NAME);
            enabled.full_quads = true;
        }
    }
#endif
    (void)device;
    (void)extensions;
    return enabled;
}
} // namespace prosper::test
