#include <vector>
#include "fixtures/render_reconvergence_device.hpp"

namespace prosper::test {
bool RenderReconvergenceDevice::prepare(VkPhysicalDevice physical, bool advertised,
                                        VkDeviceCreateInfo& device,
                                        std::vector<const char*>& extensions,
                                        PFN_vkGetPhysicalDeviceFeatures2 query) {
    if (prepared_) return false;
    prepared_ = true;
#ifdef VK_KHR_shader_maximal_reconvergence
    if (!physical || !advertised || !query) return false;
    VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &feature_};
    query(physical, &supported);
    if (!feature_.shaderMaximalReconvergence) return false;
    feature_ = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR};
    feature_.shaderMaximalReconvergence = VK_TRUE;
    feature_.pNext = const_cast<void*>(device.pNext);
    device.pNext = &feature_;
    extensions.push_back(VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME);
    return true;
#else
    (void)physical;
    (void)advertised;
    (void)device;
    (void)extensions;
    (void)query;
    return false;
#endif
}
} // namespace prosper::test
