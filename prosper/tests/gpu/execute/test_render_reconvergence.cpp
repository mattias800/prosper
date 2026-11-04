// Mutations: advertise without enabling the feature, or read a mode number from literal data.
#include "fixtures/render_reconvergence_device.hpp"
#include "fixtures/render_reconvergence_shader.hpp"
#include "fixtures/native_fragment_readfirst.hpp"
#include "gpu/recompiler/spirv/fragment_reconvergence.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace {
#ifdef VK_KHR_shader_maximal_reconvergence
thread_local bool feature_supported = false;
thread_local unsigned queries = 0;
VKAPI_ATTR void VKAPI_CALL query(VkPhysicalDevice, VkPhysicalDeviceFeatures2* output) {
    ++queries;
    auto* feature =
        static_cast<VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR*>(output->pNext);
    ASSERT_EQ(feature->sType,
              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR);
    feature->shaderMaximalReconvergence = feature_supported;
}
#endif
using Requirement = prosper::gpu::FragmentReconvergence;
std::vector<uint32_t> module(std::initializer_list<uint32_t> instructions) {
    std::vector<uint32_t> result{0x07230203, 0x00010300, 0, 8, 0};
    result.insert(result.end(), instructions);
    return result;
}
} // namespace

TEST(RenderReconvergence, DeviceRequestRequiresAdvertisementAndQueriedSupport) {
#ifdef VK_KHR_shader_maximal_reconvergence
    for (bool advertised : {false, true})
        for (bool supported : {false, true}) {
            prosper::test::RenderReconvergenceDevice request;
            VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            VkPhysicalDeviceFeatures2 previous{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            device.pNext = &previous;
            std::vector<const char*> extensions;
            feature_supported = supported;
            queries = 0;
            const auto physical = reinterpret_cast<VkPhysicalDevice>(uintptr_t{1});
            const bool enabled = request.prepare(physical, advertised, device, extensions, query);
            EXPECT_EQ(queries, advertised ? 1u : 0u);
            EXPECT_EQ(enabled, advertised && supported);
            ASSERT_EQ(extensions.size(), enabled ? 1u : 0u);
            if (enabled) {
                const auto* feature =
                    static_cast<const VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR*>(
                        device.pNext);
                ASSERT_EQ(
                    feature->sType,
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR);
                EXPECT_EQ(feature->shaderMaximalReconvergence, VK_TRUE);
                EXPECT_EQ(feature->pNext, &previous);
                EXPECT_STREQ(extensions[0], VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME);
            } else {
                EXPECT_EQ(device.pNext, &previous);
            }
            EXPECT_FALSE(request.prepare(physical, advertised, device, extensions, query));
            EXPECT_EQ(queries, advertised ? 1u : 0u);
        }
#endif
}

TEST(RenderReconvergence, MissingQueryCannotEnableAFeature) {
    prosper::test::RenderReconvergenceDevice request;
    VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    std::vector<const char*> extensions;
    EXPECT_FALSE(request.prepare(reinterpret_cast<VkPhysicalDevice>(uintptr_t{1}), true, device,
                                 extensions, nullptr));
    EXPECT_EQ(device.pNext, nullptr);
    EXPECT_TRUE(extensions.empty());
}

TEST(RenderReconvergence, OnlyAnExecutionModeRequestsTheFeature) {
    using prosper::gpu::fragment_spirv_reconvergence;
    const auto literal = module({(4u << 16) | 43, 1, 2, 6023}); // OpConstant payload
    const auto mode = module({(3u << 16) | 16, 1, 6023});
    EXPECT_EQ(fragment_spirv_reconvergence(literal), Requirement::None);
    EXPECT_EQ(fragment_spirv_reconvergence(mode), Requirement::Required);
    EXPECT_TRUE(prosper::test::render_reconvergence_supported(literal, false));
    EXPECT_FALSE(prosper::test::render_reconvergence_supported(mode, false));
    EXPECT_TRUE(prosper::test::render_reconvergence_supported(mode, true));
}

TEST(RenderReconvergence, TruncatedOrMalformedMetadataCannotBypassTheGate) {
    using prosper::gpu::fragment_spirv_reconvergence;
    for (const auto& malformed :
         {std::vector<uint32_t>{}, module({0}), module({(3u << 16) | 16, 1}),
          module({(4u << 16) | 16, 1, 6023, 0})}) {
        EXPECT_EQ(fragment_spirv_reconvergence(malformed), Requirement::Malformed);
        EXPECT_FALSE(prosper::test::render_reconvergence_supported(malformed, true));
    }
}

TEST(RenderReconvergence, NewBroadcastDeclaresTheContractAndFixedLaneControlDoesNot) {
    namespace fixture = prosper::test::native_fragment_readfirst;
    for (bool first : {false, true}) {
        const auto result = prosper::test::native_pixel_parameter::compile(fixture::program(first),
                                                                           fixture::mapping());
        ASSERT_FALSE(result.module.empty()) << result.reason;
        EXPECT_EQ(prosper::gpu::fragment_spirv_reconvergence(result.module),
                  first ? Requirement::Required : Requirement::None);
        EXPECT_EQ(prosper::test::render_reconvergence_supported(result.module, false), !first);
    }
}
