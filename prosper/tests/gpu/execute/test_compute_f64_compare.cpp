// #5325: the admitted V_CMP(X)_*_F64 modules on the executing Vulkan device. Each definition, in
// all five forms and both wave sizes, under the preserving and the flushing FP_DENORM[7:6] mode,
// must write the hand-worked IEEE binary64 lane mask: NaNs (quiet and signalling), signed zeros,
// the infinities, subnormals on both sides of zero and at the normal boundary, and a pair that
// differs only in the low word. The interpreter tests (test_ir_vector_compare_f64_execution) say
// the SPIR-V means this; this test says a real driver executes it so, with DenormPreserve 64.
#include "fixtures/compute_runner.h"
#include "gpu/execute/float_controls_probe.hpp"
#include "gpu/recompiler/float64_support.hpp"
#include "gpu/recompiler/ir/ops/vector_compare.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {
constexpr uint32_t kFlush = kDefaultComputePgmRsrc1;
constexpr uint32_t kPreserve = kDefaultComputePgmRsrc1 | (0xc0u << 12);

// The same rows and hand-worked truth as test_ir_vector_compare_f64_execution (AMD 70648 12.9
// Table 61; FP_DENORM 6.4 Table 24). One character per pair: preserved, then flushed inputs.
constexpr uint64_t kPairs[][2] = {
    {0, 0x8000000000000000ull},
    {0x3ff0000000000000ull, 0x7ff8000000000000ull},
    {0x7ff8000000000000ull, 0x7ff8000000000000ull},
    {0x7ff0000000000001ull, 0x3ff0000000000000ull},
    {0xfff0000000000000ull, 0x7ff0000000000000ull},
    {0x7fefffffffffffffull, 0x7ff0000000000000ull},
    {1, 0},
    {0x8000000000000001ull, 0},
    {0x000fffffffffffffull, 0x0010000000000000ull},
    {0x3ff0000000000000ull, 0x3ff0000000000001ull},
    {0xbff0000000000000ull, 0xbff0000000000000ull},
    {0x4000000000000000ull, 0xbfe0000000000000ull},
    {1, 0x8000000000000001ull},
};
struct TruthRow {
    uint32_t opcode;
    const char* preserve;
    const char* flush;
};
constexpr TruthRow kRows[] = {
    {0x20, "0000000000000", "0000000000000"}, {0x21, "0000110111000", "0000110011000"},
    {0x22, "1000000000100", "1000001100101"}, {0x23, "1000110111100", "1000111111101"},
    {0x24, "0000001000011", "0000000000010"}, {0x25, "0000111111011", "0000110011010"},
    {0x26, "1000001000111", "1000001100111"}, {0x27, "1000111111111", "1000111111111"},
    {0x28, "0111000000000", "0111000000000"}, {0x29, "0111110111000", "0111110011000"},
    {0x2a, "1111000000100", "1111001100101"}, {0x2b, "1111110111100", "1111111111101"},
    {0x2c, "0111001000011", "0111000000010"}, {0x2d, "0111111111011", "0111110011010"},
    {0x2e, "1111001000111", "1111001100111"}, {0x2f, "1111111111111", "1111111111111"},
};

struct Device {
    bool float64 = false, float_controls = false, float_controls2 = false,
         compatible_controls = false;
    std::string receipt;
};
// run_compute's device: the first physical device on a Vulkan 1.1 instance.
Device query_device() {
    Device d;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) return d;
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    if (count) vkEnumeratePhysicalDevices(instance, &count, devices.data());
    if (count) {
        VkPhysicalDeviceFeatures features{};
        vkGetPhysicalDeviceFeatures(devices[0], &features);
        d.float64 = features.shaderFloat64 == VK_TRUE;
        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(devices[0], nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> extensions(n);
        vkEnumerateDeviceExtensionProperties(devices[0], nullptr, &n, extensions.data());
        bool driver_properties = false;
        for (const auto& e : extensions) {
            d.float_controls |=
                !std::strcmp(e.extensionName, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
            d.float_controls2 |=
                !std::strcmp(e.extensionName, VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME);
            driver_properties |=
                !std::strcmp(e.extensionName, VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME);
        }
        if (d.float_controls2) {
            VkPhysicalDeviceShaderFloatControls2Features controls2{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES};
            VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            query.pNext = &controls2;
            vkGetPhysicalDeviceFeatures2(devices[0], &query);
            d.float_controls2 = controls2.shaderFloatControls2 == VK_TRUE;
        }
        VkPhysicalDeviceFloatControlsProperties controls{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
        VkPhysicalDeviceDriverProperties driver{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        if (driver_properties) controls.pNext = &driver;
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &controls;
        vkGetPhysicalDeviceProperties2(devices[0], &properties);
        d.compatible_controls =
            d.float_controls && d.float_controls2 && float64_controls_compatible(controls);
        d.receipt = std::string("device=") + properties.properties.deviceName +
                    " api=" + std::to_string(properties.properties.apiVersion) +
                    " driver_version=" + std::to_string(properties.properties.driverVersion) +
                    (driver_properties
                         ? std::string(" driver=") + driver.driverName + " " + driver.driverInfo
                         : std::string());
    }
    vkDestroyInstance(instance, nullptr);
    return d;
}

std::vector<uint32_t> buffer(const LaneInputs& in) {
    std::vector<uint32_t> words;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        words.insert(words.end(), array->begin(), array->end());
    for (const bool active : in.active) words.push_back(active ? 1u : 0u);
    words.insert(words.end(), in.s2.begin(), in.s2.end());
    words.insert(words.end(), in.s3.begin(), in.s3.end());
    return words;
}
} // namespace

// The emitter declares DenormPreserve only for 64 bits. Restricted independence would require
// matching modes for other widths (Vulkan runtime SPIR-V rules 06289/06290), which it does not emit.
// These synthetic properties exercise the production publication without creating a Vulkan device.
TEST(Float64DeviceGate, RestrictedIndependenceRefusesAndCannotBeRaisedByAnotherPublisher) {
    VkPhysicalDeviceFloatControlsProperties properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    properties.shaderDenormPreserveFloat64 = VK_TRUE;
    properties.shaderSignedZeroInfNanPreserveFloat64 = VK_TRUE;
    for (const auto independence : {VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE,
                                    VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY}) {
        reset_float64_support_for_test();
        properties.denormBehaviorIndependence = independence;
        EXPECT_FALSE(publish_float64_device_support(true, properties, true, true));
        EXPECT_FALSE(float64_admitted());
        properties.denormBehaviorIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL;
        EXPECT_TRUE(publish_float64_device_support(true, properties, true, true));
        EXPECT_FALSE(float64_admitted()) << "the weaker earlier publisher remains authoritative";
    }
    reset_float64_support_for_test();
}

TEST(Float64DeviceGate, IndependentPreservationStillRequiresEveryEnabledCapability) {
    VkPhysicalDeviceFloatControlsProperties properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    properties.denormBehaviorIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL;
    for (const bool feature : {false, true})
        for (const bool preserve : {false, true})
            for (const bool nonfinite : {false, true})
                for (const bool controls2 : {false, true})
                    for (const bool permitted : {false, true}) {
                        reset_float64_support_for_test();
                        properties.shaderDenormPreserveFloat64 = preserve ? VK_TRUE : VK_FALSE;
                        properties.shaderSignedZeroInfNanPreserveFloat64 =
                            nonfinite ? VK_TRUE : VK_FALSE;
                        const bool want =
                            feature && preserve && nonfinite && controls2 && permitted;
                        EXPECT_EQ(publish_float64_device_support(feature, properties, permitted,
                                                                 controls2),
                                  want);
                        EXPECT_EQ(float64_admitted(), want);
                    }
    reset_float64_support_for_test();
}

TEST(ComputeF64Compare, EveryDefinitionAndFormMatchesTheHandWorkedTruthOnTheDevice) {
    const Device device = query_device();
    ASSERT_FALSE(device.receipt.empty()) << "real Vulkan execution is required";
    RecordProperty("executing_driver", device.receipt);
    std::printf("F64_COMPARE_EXECUTING_DRIVER %s\n", device.receipt.c_str());
    if (!device.float64 || !device.compatible_controls)
        GTEST_SKIP()
            << "device lacks Float64 or compatible nonfinite/denormal controls: F64 refuses";
    reset_float64_support_for_test();
    publish_float64_support(true, true, true);
    size_t lanes_checked = 0;
    for (const auto& row : kRows)
        for (const Form form : kForms)
            for (const uint32_t wave : {64u, 32u})
                for (const uint32_t rsrc1 : {kPreserve, kFlush}) {
                    auto in = lane_inputs(row.opcode * 31 + wave);
                    in.s2.resize(in.lanes());
                    in.s3.resize(in.lanes());
                    for (size_t lane = 0; lane < in.lanes(); ++lane) {
                        const auto& p = kPairs[lane % std::size(kPairs)];
                        in.s0[lane] = uint32_t(p[0]), in.s1[lane] = uint32_t(p[0] >> 32);
                        in.s2[lane] = uint32_t(p[1]), in.s3[lane] = uint32_t(p[1] >> 32);
                    }
                    auto program = lane_program(compare_program(row.opcode, form, 0, 0, kVsrc2), 64,
                                                wave, true, true);
                    program.config.compute_pgm_rsrc1 = rsrc1;
                    const auto module = recompile_compute(program.code.data(), program.code.size(),
                                                          &program.table, program.config);
                    ASSERT_FALSE(module.empty()) << std::hex << row.opcode;
                    std::vector<uint32_t> out;
                    const auto done = prosper::test::run_compute(module, std::vector<float>(64), 64,
                                                                 64, {}, buffer(in), &out);
                    ASSERT_EQ(done.size(), 64u) << "queue, fence and host availability must finish";
                    ASSERT_GE(out.size(), 64u);
                    const char* truth = rsrc1 == kFlush ? row.flush : row.preserve;
                    for (size_t lane = 0; lane < 64; ++lane) {
                        // v1 is stored over the S0-low array: the harness's out[lane] slot.
                        const uint32_t want = expected_mask_read(
                            form, in.active[lane], truth[lane % std::size(kPairs)] == '1',
                            in.old_d[lane]);
                        EXPECT_EQ(out[lane], want)
                            << "op " << std::hex << row.opcode << " form " << int(form) << " wave "
                            << std::dec << wave << " rsrc1 " << std::hex << rsrc1 << " lane "
                            << std::dec << lane;
                        ++lanes_checked;
                    }
                }
    RecordProperty("lanes_checked", lanes_checked);
    reset_float64_support_for_test();
}
