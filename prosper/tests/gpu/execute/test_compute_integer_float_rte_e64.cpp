// The interpreter cannot establish device rounding. Execute the actual RTE32 modules as well.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstdint>
#include "fixtures/compute_runner.h"
#include "gpu/execute/float_controls_probe.hpp"
#include "gpu/recompiler/float32_rte_support.hpp"
#include "gpu/recompiler/ir/integer_float_rte_e64_cases.hpp"
#include <gtest/gtest.h>
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace prosper::gpu;
namespace cases = prosper::test::ir::integer_float_rte;
namespace {
struct Qualification {
    bool eligible = false, published = false;
};
bool publish_on_created_device(VkPhysicalDevice physical) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
    uint32_t family = 0;
    while (family < count && !(queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
    if (family == count) return false;
    float priority = 1;
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    const char* extension = VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME;
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &queue;
    ci.enabledExtensionCount = 1;
    ci.ppEnabledExtensionNames = &extension;
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical, &ci, nullptr, &device) != VK_SUCCESS) return false;
    publish_device_float_controls("integer-float-rte-test", physical, VK_API_VERSION_1_1, true);
    const bool admitted = float32_rte_admitted();
    vkDestroyDevice(device, nullptr);
    return admitted;
}
Qualification device_support() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) return {};
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    if (count) vkEnumeratePhysicalDevices(instance, &count, devices.data());
    Qualification result;
    if (count) {
        // run_compute uses this same first physical device and enables this advertised extension.
        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(devices[0], nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> extensions(n);
        vkEnumerateDeviceExtensionProperties(devices[0], nullptr, &n, extensions.data());
        bool extension = false;
        for (const auto& e : extensions)
            extension |= !std::strcmp(e.extensionName, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
        VkPhysicalDeviceFloatControlsProperties controls{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
        VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        if (extension) p.pNext = &controls;
        vkGetPhysicalDeviceProperties2(devices[0], &p);
        result.eligible = extension && float32_rte_controls_compatible(controls);
        if (result.eligible) result.published = publish_on_created_device(devices[0]);
        std::printf("EXECUTING_DRIVER device=%s driver_version=%u RTE32=%u "
                    "rounding_independence=%u extension=%u\n",
                    p.properties.deviceName, p.properties.driverVersion,
                    controls.shaderRoundingModeRTEFloat32, controls.roundingModeIndependence,
                    extension);
    }
    vkDestroyInstance(instance, nullptr);
    return result;
}
}   // namespace
TEST(ComputeIntegerFloatRteE64, EveryAdmittedFormRoundsAndPreservesInactiveWordsOnTheDevice) {
    struct Reset {
        ~Reset() {
            reset_float32_rte_support_for_test();
            reset_float_controls_support_for_test();
        }
    } reset;
    reset_float32_rte_support_for_test();
    reset_float_controls_support_for_test();
    const auto device = device_support();
    if (!device.eligible)
        GTEST_SKIP() << "device lacks independent RTE32 or float-controls extension";
    ASSERT_TRUE(device.published) << "production publisher must retain the measured RTE32 property";
    size_t modules = 0, lanes = 0;
    for (const auto& c : cases::cases())
        if (c.accepted) {
            const auto p = cases::program(c);
            const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
            ASSERT_FALSE(module.empty());
            ++modules;
            for (uint32_t mask : {0u, 1u, 2u}) {
                const auto in = cases::inputs(5394, mask);
                std::vector<uint32_t> actual;
                const auto done = prosper::test::run_compute(module, std::vector<float>(64), 64, 64,
                                                             {}, cases::buffer(in), &actual);
                ASSERT_EQ(done.size(), 64u);
                ASSERT_EQ(actual, cases::expected(c, in));
                lanes += 64;
            }
        }
    RecordProperty("modules", modules);
    RecordProperty("lanes_checked", lanes);
    reset_float32_rte_support_for_test();
}
