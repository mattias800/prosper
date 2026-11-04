// Mutations: advertise without enabling the feature, or read a mode number from literal data.
#include "fixtures/native_pixel_parameter.hpp"
#include "fixtures/render_reconvergence_device.hpp"
#include "fixtures/render_reconvergence_shader.hpp"
#include "fixtures/native_fragment_readfirst.hpp"
#include "gpu/recompiler/spirv/fragment_reconvergence.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace {
#if defined(VK_KHR_shader_maximal_reconvergence) && defined(VK_KHR_shader_quad_control)
thread_local bool support_maximal = false, support_quads = false;
thread_local unsigned maximal_queries = 0, quad_queries = 0;
VKAPI_ATTR void VKAPI_CALL query(VkPhysicalDevice, VkPhysicalDeviceFeatures2* output) {
    auto* header = static_cast<VkBaseOutStructure*>(output->pNext);
    if (header->sType ==
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR) {
        ++maximal_queries;
        reinterpret_cast<VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR*>(header)
            ->shaderMaximalReconvergence = support_maximal;
    } else if (header->sType ==
               VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR) {
        ++quad_queries;
        reinterpret_cast<VkPhysicalDeviceShaderQuadControlFeaturesKHR*>(header)->shaderQuadControl =
            support_quads;
    } else {
        ADD_FAILURE() << "unexpected feature query";
    }
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
#if defined(VK_KHR_shader_maximal_reconvergence) && defined(VK_KHR_shader_quad_control)
    for (bool max_advertised : {false, true})
        for (bool quad_advertised : {false, true})
            for (bool max_supported : {false, true})
                for (bool quad_supported : {false, true}) {
                    prosper::test::RenderReconvergenceDevice request;
                    if (max_advertised)
                        request.note_extension(VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME);
                    if (quad_advertised)
                        request.note_extension(VK_KHR_SHADER_QUAD_CONTROL_EXTENSION_NAME);
                    VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
                    VkPhysicalDeviceFeatures2 previous{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                    device.pNext = &previous;
                    std::vector<const char*> extensions;
                    support_maximal = max_supported;
                    support_quads = quad_supported;
                    maximal_queries = quad_queries = 0;
                    const auto physical = reinterpret_cast<VkPhysicalDevice>(uintptr_t{1});
                    const auto enabled = request.prepare(physical, device, extensions, query);
                    EXPECT_EQ(enabled.maximal, max_advertised && max_supported);
                    EXPECT_EQ(enabled.full_quads,
                              enabled.maximal && quad_advertised && quad_supported);
                    EXPECT_EQ(maximal_queries, max_advertised ? 1u : 0u);
                    EXPECT_EQ(quad_queries, enabled.maximal && quad_advertised ? 1u : 0u);
                    ASSERT_EQ(extensions.size(),
                              size_t(enabled.maximal) + size_t(enabled.full_quads));
                    const auto* chain = static_cast<const VkBaseOutStructure*>(device.pNext);
                    if (enabled.full_quads) {
                        ASSERT_EQ(
                            chain->sType,
                            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR);
                        EXPECT_EQ(
                            reinterpret_cast<const VkPhysicalDeviceShaderQuadControlFeaturesKHR*>(
                                chain)
                                ->shaderQuadControl,
                            VK_TRUE);
                        chain = chain->pNext;
                    }
                    if (enabled.maximal) {
                        ASSERT_EQ(
                            chain->sType,
                            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR);
                        EXPECT_EQ(
                            reinterpret_cast<
                                const VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR*>(chain)
                                ->shaderMaximalReconvergence,
                            VK_TRUE);
                        chain = chain->pNext;
                    }
                    EXPECT_EQ(static_cast<const void*>(chain), static_cast<const void*>(&previous));
                    const auto again = request.prepare(physical, device, extensions, query);
                    EXPECT_FALSE(again.maximal);
                    EXPECT_FALSE(again.full_quads);
                }
#else
    GTEST_SKIP() << "SDK does not describe both fragment collective features";
#endif
}

TEST(RenderReconvergence, MissingQueryCannotEnableAFeature) {
    prosper::test::RenderReconvergenceDevice request;
    request.note_extension("VK_KHR_shader_maximal_reconvergence");
    request.note_extension("VK_KHR_shader_quad_control");
    VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    std::vector<const char*> extensions;
    const auto result = request.prepare(reinterpret_cast<VkPhysicalDevice>(uintptr_t{1}), device,
                                        extensions, nullptr);
    EXPECT_FALSE(result.maximal);
    EXPECT_FALSE(result.full_quads);
    EXPECT_EQ(device.pNext, nullptr);
    EXPECT_TRUE(extensions.empty());
}

TEST(RenderReconvergence, OnlyAnExecutionModeRequestsTheFeature) {
    using prosper::gpu::fragment_spirv_reconvergence;
    const auto literal = module({(4u << 16) | 43, 1, 2, 6023});
    const auto maximal = module({(3u << 16) | 16, 1, 6023});
    const auto full = module({(3u << 16) | 16, 1, 5089});
    EXPECT_EQ(fragment_spirv_reconvergence(literal), Requirement::None);
    EXPECT_EQ(fragment_spirv_reconvergence(maximal), Requirement::Required);
    EXPECT_EQ(fragment_spirv_reconvergence(full), Requirement::Both);
    for (bool max_enabled : {false, true})
        for (bool quad_enabled : {false, true}) {
            const prosper::test::RenderReconvergenceFeatures features{max_enabled, quad_enabled};
            EXPECT_TRUE(prosper::test::render_reconvergence_supported(literal, features));
            EXPECT_EQ(prosper::test::render_reconvergence_supported(maximal, features),
                      max_enabled);
            EXPECT_EQ(prosper::test::render_reconvergence_supported(full, features),
                      max_enabled && quad_enabled);
        }
}

TEST(RenderReconvergence, TruncatedOrMalformedMetadataCannotBypassTheGate) {
    using prosper::gpu::fragment_spirv_reconvergence;
    for (const auto& malformed :
         {std::vector<uint32_t>{}, module({0}), module({(3u << 16) | 16, 1}),
          module({(4u << 16) | 16, 1, 6023, 0}), module({(4u << 16) | 16, 1, 5089, 0})}) {
        EXPECT_EQ(fragment_spirv_reconvergence(malformed), Requirement::Malformed);
        EXPECT_FALSE(prosper::test::render_reconvergence_supported(malformed, {true, true}));
    }
}

TEST(RenderReconvergence, NewBroadcastDeclaresTheContractAndFixedLaneControlDoesNot) {
    namespace fixture = prosper::test::native_fragment_readfirst;
    for (bool first : {false, true}) {
        const auto result = prosper::test::native_pixel_parameter::compile(fixture::program(first),
                                                                           fixture::mapping());
        ASSERT_FALSE(result.module.empty()) << result.reason;
        EXPECT_EQ(prosper::gpu::fragment_spirv_reconvergence(result.module),
                  first ? Requirement::Both : Requirement::None);
        EXPECT_EQ(prosper::test::render_reconvergence_supported(result.module, {}), !first);
    }
}
