// The first physical device, selected by compute_runner, without publishing a float profile.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
namespace prosper::test::fma_rounding {
inline std::string device_receipt() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) return {};
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || !count) {
        vkDestroyInstance(instance, nullptr);
        return {};
    }
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        return {};
    }
    // run_compute also selects the first physical device. Query without publishing a new
    // float-transport profile or changing any execution capability.
    VkPhysicalDeviceProperties basic{};
    vkGetPhysicalDeviceProperties(devices[0], &basic);
    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(devices[0], nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(devices[0], nullptr, &extension_count, extensions.data());
    bool driver_properties = basic.apiVersion >= VK_API_VERSION_1_2;
    for (const auto& extension : extensions)
        driver_properties |=
            std::strcmp(extension.extensionName, VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME) == 0;
    VkPhysicalDeviceDriverProperties driver{.sType =
                                                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
                                            .driverID = VK_DRIVER_ID_MAX_ENUM};
    if (driver_properties) {
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &driver;
        vkGetPhysicalDeviceProperties2(devices[0], &properties);
    }
    std::string out = "device=" + std::string(basic.deviceName) +
                      " vendor=" + std::to_string(basic.vendorID) +
                      " device_id=" + std::to_string(basic.deviceID) +
                      " api=" + std::to_string(basic.apiVersion) +
                      " driver_version=" + std::to_string(basic.driverVersion) +
                      " driver_id=" + std::to_string(driver.driverID) +
                      " driver_name=" + std::string(driver.driverName) +
                      " driver_info=" + std::string(driver.driverInfo);
    vkDestroyInstance(instance, nullptr);
    return out;
}
}   // namespace prosper::test::fma_rounding
