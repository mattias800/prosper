// #5225: execute actual guest GLSL Fma modules. This proves this driver's finite RTE
// witness only; Vulkan allows Mul+Add precision and the old RADV graph also contracted.
// #5331: keep guest-to-native mapping active when an independent driver control lacks
// strict fused accuracy; qualify only the strict witness and retain the #5225 gap.
#include "fixtures/f32_fma_rounding.hpp"
#include "fixtures/f32_mad_rounding.hpp"
#include "fixtures/compute_runner.h"
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <vulkan/vulkan.h>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::fma_rounding;
namespace {
// Independent, hand-built SPIR-V: buffer3 contains A, B, C and result arrays of 64
// raw words. Dynamic storage loads prevent constant folding of the accuracy probe.
// Do not use SpirvCompute::fma here: a wrong guest emission must not change its oracle.
std::vector<uint32_t> native_fma_control() {
    std::vector<uint32_t> module{0x07230203, 0x00010300, 0, 35, 0};
    const auto emit = [&](uint32_t opcode, std::initializer_list<uint32_t> operands) {
        module.push_back((uint32_t(operands.size() + 1) << 16) | opcode);
        module.insert(module.end(), operands);
    };
    emit(Op_Capability, {1}); // Shader
    emit(Op_ExtInstImport, {1, 0x4c534c47, 0x6474732e, 0x3035342e, 0}); // GLSL.std.450
    emit(Op_MemoryModel, {0, 1}); // Logical, GLSL450
    emit(Op_EntryPoint, {5, 16, 0x6e69616d, 0, 14}); // GLCompute main, GlobalInvocationId
    emit(Op_ExecutionMode, {16, 17, 64, 1, 1}); // LocalSize
    emit(Op_Decorate, {8, 6, 4}); // ArrayStride
    emit(Op_Decorate, {9, 2}); // Block
    emit(Op_MemberDecorate, {9, 0, 35, 0}); // Offset
    emit(Op_Decorate, {15, 34, 0}); // DescriptorSet
    emit(Op_Decorate, {15, 33, 3}); // Binding
    emit(Op_Decorate, {14, 11, 28}); // BuiltIn GlobalInvocationId
    emit(Op_TypeVoid, {2});
    emit(Op_TypeFunction, {3, 2});
    emit(Op_TypeInt, {4, 32, 0});
    emit(Op_TypeFloat, {5, 32});
    emit(Op_TypeVector, {6, 4, 3});
    emit(Op_TypePointer, {7, 1, 6}); // Input uvec3
    emit(Op_TypeRuntimeArray, {8, 4});
    emit(Op_TypeStruct, {9, 8});
    emit(Op_TypePointer, {10, 12, 9}); // StorageBuffer block
    emit(Op_TypePointer, {11, 12, 4}); // StorageBuffer uint
    emit(Op_Constant, {4, 12, 0});
    emit(Op_Constant, {4, 13, 64});
    emit(Op_Variable, {7, 14, 1});
    emit(Op_Variable, {10, 15, 12});
    emit(Op_Function, {2, 16, 0, 3});
    emit(Op_Label, {17});
    emit(Op_Load, {6, 18, 14});
    emit(Op_CompositeExtract, {4, 19, 18, 0});
    emit(Op_IAdd, {4, 20, 19, 13});
    emit(Op_IAdd, {4, 21, 20, 13});
    emit(Op_IAdd, {4, 22, 21, 13});
    for (uint32_t k = 0; k < 3; ++k) {
        emit(Op_AccessChain, {11, 23 + k, 15, 12, 19 + k});
        emit(Op_Load, {4, 26 + k, 23 + k});
        emit(Op_Bitcast, {5, 29 + k, 26 + k});
    }
    emit(Op_ExtInst, {5, 32, 1, 50, 29, 30, 31}); // GLSL.std.450 Fma, directly from SPIR-V
    emit(Op_Bitcast, {4, 33, 32});
    emit(Op_AccessChain, {11, 34, 15, 12, 22});
    emit(Op_Store, {34, 33});
    emit(Op_Return, {});
    emit(Op_FunctionEnd, {});
    return module;
}

std::vector<uint32_t> native_fma_inputs(const Encoding& encoding, const LaneInputs& in) {
    std::vector<uint32_t> data(4 * in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto source = prosper::test::mad_rounding::sources(encoding, {}, in, lane);
        for (size_t k = 0; k < source.size(); ++k) data[k * in.lanes() + lane] = source[k];
    }
    return data;
}

std::vector<uint32_t> native_fma_results(const Encoding& encoding, const LaneInputs& in) {
    const auto data = native_fma_inputs(encoding, in);
    std::vector<uint32_t> result;
    const auto dispatch = prosper::test::run_compute(native_fma_control(), std::vector<float>(64),
                                                     64, 64, {}, data, &result);
    EXPECT_EQ(dispatch.size(), 64u);
    if (result.size() != data.size()) return {};
    return {result.begin() + 192, result.end()};
}

std::filesystem::path artifact_root() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
    const char* configured = std::getenv("PROSPER_F32_FMA_GPU_ARTIFACTS");
    const auto root = configured ? std::filesystem::path(configured)
                                 : prosper_test::test_scratch_dir() / "f32-fma-gpu";
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
std::vector<uint32_t> gpu_run(const Encoding& e, uint32_t wave,
                              const LaneInputs* independent_inputs = nullptr) {
    auto p = program(e, {}, wave, 0);
    const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
    EXPECT_FALSE(module.empty());
    if (module.empty()) return {};
    const auto stem = std::string(e.name) + "_wave" + std::to_string(wave) +
                      (independent_inputs ? "_distinct" : "");
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
                                                       p.config, "synthetic-f32-fma-gpu-witness"))
        return {};
    const auto in = independent_inputs ? *independent_inputs : inputs(&e);
    std::vector<uint32_t> output;
    const auto dispatch =
        prosper::test::run_compute(module, std::vector<float>(64), 64, 64, {}, buffer(in), &output);
    EXPECT_EQ(dispatch.size(), 64u) << "dispatch, fence, and host availability must complete";
    output.resize(std::min(output.size(), size_t(64)));
    return output;
}
} // namespace
TEST(ComputeF32FmaRounding, IndependentControlKeepsTheHandWorkedFusedWitness) {
    const auto data = native_fma_inputs(kEncodings.front(), inputs(&kEncodings.front()));
    const auto module = native_fma_control();
    std::ofstream saved(artifact_root() / "native-fma-control.spv", std::ios::binary);
    saved.write(reinterpret_cast<const char*>(module.data()),
                static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
    saved.close();
    ASSERT_TRUE(saved.good());
    bpermute_oracle::Interpreter interpreter(module);
    const auto output = interpreter.run_buffers(64, {{3, data}}, 3);
    ASSERT_TRUE(interpreter.error.empty()) << interpreter.error;
    ASSERT_EQ(output.size(), data.size());
    // (1+2^-23)*(1-2^-23)-1 = -2^-46, independent of the guest IR evaluator.
    EXPECT_EQ(output[3 * 64 + 4], 0xa8800000u);
    EXPECT_TRUE(std::equal(data.begin(), data.begin() + 192, output.begin()));
}

TEST(ComputeF32FmaRounding, FourFusedGuestGraphsMatchTheFiniteRteReferenceOnTheExecutingDriver) {
    const auto receipt = device_receipt();
    ASSERT_FALSE(receipt.empty()) << "a real Vulkan dispatch is required for this witness";
    RecordProperty("executing_driver", receipt);
    std::printf("F32_FMA_EXECUTING_DRIVER %s\n", receipt.c_str());
    std::ofstream driver(artifact_root() / "driver.txt");
    driver << receipt << '\n';
    driver.close();
    ASSERT_TRUE(driver.good());
    const auto control = native_fma_results(kEncodings.front(), inputs(&kEncodings.front()));
    ASSERT_EQ(control.size(), 64u);
    RecordProperty("native_fma_witness", control[4]);
    if (control[4] == 0)
        GTEST_SKIP() << "Independent native GLSL Fma control also rounds separately; "
                        "strict guest accuracy is unsupported here (#5225). "
                     << receipt;
    ASSERT_EQ(control[4], 0xa8800000u) << "unexpected native-control accuracy";
    for (const auto& e : kEncodings)
        for (uint32_t wave : {32u, 64u}) {
            SCOPED_TRACE(std::string(e.name) + " wave " + std::to_string(wave));
            const auto output = gpu_run(e, wave);
            const auto in = inputs(&e);
            ASSERT_EQ(output.size(), 64u);
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(output[lane], fused_expected(e, {}, in, lane, 0)) << "lane " << lane;
            EXPECT_EQ(output[4], 0xa8800000u);
            EXPECT_EQ(output[0], in.old_d[0]) << "inactive witness retains old VDST";
        }
}

TEST(ComputeF32FmaRounding, EveryGuestFormMatchesAnIndependentNativeFmaControl) {
    for (const auto& encoding : kEncodings) {
        auto in = inputs(&encoding);
        // Distinguish accumulator, SRC1 and SRC2. The cancellation dataset deliberately
        // aliases these values, so it cannot detect selecting the wrong addend register.
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            in.old_d[lane] = lane % 2 ? 0xc0500000u : 0x40500000u; // +/-3.25
            in.s2[lane] = lane % 2 ? 0xbf400000u : 0x3f400000u; // +/-0.75
        }
        const auto reference = native_fma_results(encoding, in);
        ASSERT_EQ(reference.size(), 64u);
        for (uint32_t wave : {32u, 64u}) {
            SCOPED_TRACE(std::string(encoding.name) + " wave " + std::to_string(wave));
            const auto output = gpu_run(encoding, wave, &in);
            ASSERT_EQ(output.size(), 64u);
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(output[lane], in.active[lane] ? reference[lane] : in.old_d[lane])
                    << "lane " << lane;
        }
    }
}
