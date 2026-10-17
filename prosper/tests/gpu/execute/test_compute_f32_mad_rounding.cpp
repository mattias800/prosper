// #5225: execute the real guest modules on Vulkan, because interpreter agreement alone
// cannot prove that a driver preserves the separately rounded product.
#include "fixtures/f32_mad_rounding.hpp"
#include "fixtures/compute_runner.h"
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include <gtest/gtest.h>
#include <vulkan/vulkan.h>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::mad_rounding;
namespace {
std::filesystem::path artifact_root() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
    const char* configured = std::getenv("PROSPER_F32_MAD_GPU_ARTIFACTS");
    const auto root = configured ? std::filesystem::path(configured)
                                 : prosper_test::test_scratch_dir() / "f32-mad-gpu";
    std::filesystem::create_directories(root);
    return root;
}
std::string device_receipt() {
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
std::vector<uint32_t> gpu_run(const Encoding& e, uint32_t wave) {
    auto p = program(e, {}, wave, 0);
    const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
    EXPECT_FALSE(module.empty());
    if (module.empty()) return {};
    const auto stem = std::string(e.name) + "_wave" + std::to_string(wave);
    const auto root = artifact_root();
    std::ofstream spv(root / (stem + ".spv"), std::ios::binary);
    spv.write(reinterpret_cast<const char*>(module.data()),
              static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
    spv.close();
    if (!spv.good()) return {};
    std::ofstream guest(root / (stem + ".bin"), std::ios::binary);
    guest.write(reinterpret_cast<const char*>(p.code.data()),
                static_cast<std::streamsize>(p.code.size() * sizeof(uint32_t)));
    guest.close();
    if (!guest.good() || !write_refused_compute_inputs(root / (stem + ".inputs"), 0, &p.table,
                                                       p.config, "synthetic-f32-mad-gpu-witness"))
        return {};
    const auto in = inputs(&e);
    std::vector<uint32_t> output;
    const auto dispatch =
        prosper::test::run_compute(module, std::vector<float>(64), 64, 64, {}, buffer(in), &output);
    EXPECT_EQ(dispatch.size(), 64u) << "dispatch, fence, and host availability must complete";
    output.resize(std::min(output.size(), size_t(64)));
    return output;
}
} // namespace
TEST(ComputeF32MadRounding, FiveNonfusedGuestEncodingsRetainTheirRoundingOnTheExecutingDriver) {
    const auto receipt = device_receipt();
    ASSERT_FALSE(receipt.empty()) << "a real Vulkan dispatch is required for this witness";
    RecordProperty("executing_driver", receipt);
    std::printf("F32_MAD_EXECUTING_DRIVER %s\n", receipt.c_str());
    std::ofstream driver(artifact_root() / "driver.txt");
    driver << receipt << '\n';
    driver.close();
    ASSERT_TRUE(driver.good());
    for (const auto& e : kEncodings)
        for (uint32_t wave : {32u, 64u}) {
            SCOPED_TRACE(std::string(e.name) + " wave " + std::to_string(wave));
            const auto output = gpu_run(e, wave);
            const auto in = inputs(&e);
            ASSERT_EQ(output.size(), 64u);
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(output[lane], expected(e, {}, in, lane, 0)) << "lane " << lane;
            EXPECT_EQ(output[4], 0u);
            EXPECT_EQ(output[0], in.old_d[0]) << "inactive witness retains old VDST";
        }
}
