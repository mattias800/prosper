// #5225: actual guest half FMA modules; integer exact results are compared against a
// bit-serial rational oracle. Legacy fallback is compared to a separate old-graph dispatch.
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <ios>
#include "fixtures/half_fma_finite.hpp"
#include "fixtures/compute_runner.h"
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
using namespace prosper::test::half_fma;
using namespace prosper::gpu;
namespace {
std::filesystem::path artifacts() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
    const char* configured = std::getenv("PROSPER_HALF_FMA_GPU_DIR");
    const auto root = configured ? std::filesystem::path(configured)
                                 : prosper_test::test_scratch_dir() / "half-fma-gpu";
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
std::vector<uint32_t> dispatch(const std::vector<uint32_t>& module, const LaneInputs& in) {
    std::vector<uint32_t> out;
    const auto done =
        prosper::test::run_compute(module, std::vector<float>(64), 64, 64, {}, buffer(in), &out);
    EXPECT_EQ(done.size(), 64u) << "queue, fence and host availability must finish";
    out.resize(std::min(out.size(), size_t(64)));
    return out;
}
void save(Kind kind, Shape shape, uint32_t wave, uint32_t offset,
          const std::vector<uint32_t>& module) {
    const auto stem =
        std::string(name(kind)) + "_w" + std::to_string(wave) + "_o" + std::to_string(offset);
    const auto p = program(kind, shape, wave);
    std::ofstream spv(artifacts() / (stem + ".spv"), std::ios::binary);
    spv.write(reinterpret_cast<const char*>(module.data()), module.size() * 4);
    spv.close();
    ASSERT_TRUE(spv.good());
    std::ofstream guest(artifacts() / (stem + ".bin"), std::ios::binary);
    guest.write(reinterpret_cast<const char*>(p.code.data()), p.code.size() * 4);
    guest.close();
    ASSERT_TRUE(guest.good());
    ASSERT_TRUE(write_refused_compute_inputs(artifacts() / (stem + ".inputs"), 0, &p.table,
                                             p.config, "synthetic-half-fma-gpu-witness"));
}
}   // namespace
TEST(ComputeHalfFmaFinite, ThreeExistingEncodingsPreserveTheExactResidueOnTheExecutingDevice) {
    const auto receipt = device_receipt();
    ASSERT_FALSE(receipt.empty()) << "real Vulkan execution is required";
    RecordProperty("executing_driver", receipt);
    std::printf("HALF_FMA_EXECUTING_DRIVER %s\n", receipt.c_str());
    std::ofstream driver(artifacts() / "driver.txt");
    driver << receipt << '\n';
    driver.close();
    ASSERT_TRUE(driver.good());
    size_t old_wrong = 0, new_wrong = 0;
    for (Kind kind : kKinds)
        for (uint32_t wave : {32u, 64u})
            for (uint32_t offset : {0u, 64u}) {
                Shape shape;
                if (kind == Kind::Native) shape.low = 7;
                const auto in = inputs(kind, offset);
                const auto old_module = compile(kind, shape, wave, 0);
                const auto module = compile(kind, shape, wave);
                ASSERT_FALSE(old_module.empty());
                ASSERT_FALSE(module.empty());
                save(kind, shape, wave, offset, module);
                const auto old = dispatch(old_module, in);
                const auto out = dispatch(module, in);
                ASSERT_EQ(old.size(), 64u);
                ASSERT_EQ(out.size(), 64u);
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto want = expected(kind, shape, in, lane, old[lane]);
                    old_wrong += old[lane] != want;
                    new_wrong += out[lane] != want;
                    EXPECT_EQ(out[lane], want) << name(kind) << " wave=" << wave
                                               << " offset=" << offset << " lane=" << lane;
                }
                if (kind != Kind::Native)
                    EXPECT_EQ(out[4], 0x3e0136dbu);
                else
                    EXPECT_EQ(out[4] & 0xffffu, 0x3e01u);
            }
    EXPECT_GT(old_wrong, 0u) << "the immutable legacy graph is a numeric positive control";
    EXPECT_EQ(new_wrong, 0u);
    RecordProperty("old_graph_wrong_lanes", old_wrong);
    RecordProperty("corrected_wrong_lanes", new_wrong);
}
TEST(ComputeHalfFmaFinite, DynamicZeroUnderflowOverflowAndNonfiniteUseTheOldGraphPerHalf) {
    const uint16_t boundary[][3] = {{0, 0x3c00, 0},
                                    {0x8000, 0x3c00, 0},
                                    {0x3c00, 0x3c00, 0xbc00},
                                    {1, 1, 0},
                                    {0x8001, 1, 0},
                                    {0x7bff, 0x4000, 0},
                                    {0x7c00, 0x3c00, 0},
                                    {0xfc00, 0, 0},
                                    {0x7e01, 0x3c00, 0},
                                    {0x4000, 0x4400, 0x2000},
                                    {0x4400, 0x4400, 0x2400},
                                    {0x7000, 0x6c00, 0},
                                    {0x7000, 0x7000, 0},
                                    {1, 1, 0x03ff},
                                    {1, 1, 0x0400}};
    for (Kind kind : {Kind::Packed, Kind::PackedMac})
        for (bool high : {false, true}) {
            auto in = inputs(kind);
            const auto shift = high ? 16 : 0;
            for (size_t lane = 0; lane < 64; ++lane) {
                const auto& v = boundary[lane % std::size(boundary)];
                in.s0[lane] = (in.s0[lane] & ~(0xffffu << shift)) | (uint32_t(v[0]) << shift);
                in.s1[lane] = (in.s1[lane] & ~(0xffffu << shift)) | (uint32_t(v[1]) << shift);
                auto& c = kind == Kind::PackedMac ? in.old_d[lane] : in.s2[lane];
                c = (c & ~(0xffffu << shift)) | (uint32_t(v[2]) << shift);
            }
            const auto old = dispatch(compile(kind, {}, 64, 0), in);
            const auto out = dispatch(compile(kind, {}, 64), in);
            ASSERT_EQ(old.size(), 64u);
            ASSERT_EQ(out.size(), 64u);
            for (size_t lane = 0; lane < 64; ++lane)
                EXPECT_EQ(out[lane], expected(kind, {}, in, lane, old[lane]));
        }
}
