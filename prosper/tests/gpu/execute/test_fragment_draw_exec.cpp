// Actual registered guest PS -> same backend collector/compute transaction -> normal attachments.
// These finite input-free rails prove the first recipe only, never Kena/helper/console packing.
#include "fixtures/fragment_draw_fixture.hpp"
#include "fixtures/render_runner.h"
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

namespace {
namespace g = prosper::gpu;
namespace r = prosper::test;
namespace f = r::fragment_draw;
r::BackendDraw backend(const g::DrawItem& draw) {
    r::BackendDraw value;
    value.vs = draw.vs;
    value.gs = draw.gs;
    value.fs = draw.fs;
    value.vs_shared = draw.vs_shared;
    value.fs_shared = draw.fs_shared;
    value.fragment_draw_inputs = draw.fragment_draw_inputs;
    value.vcount = draw.vertex_count;
    value.vertex_offset = draw.vertex_offset;
    value.instance_count = 1;
    value.ps = &draw.ps;
    return value;
}
void pixels(const std::vector<uint8_t>& raw, uint32_t width, uint32_t height,
            const std::array<float, 4>& expected) {
    ASSERT_EQ(raw.size(), size_t(width) * height * sizeof(expected));
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
            std::array<float, 4> actual{};
            std::memcpy(actual.data(), raw.data() + (size_t(y) * width + x) * sizeof(actual),
                        sizeof(actual));
            for (uint32_t channel = 0; channel < 4; ++channel) {
                EXPECT_TRUE(std::isfinite(actual[channel]));
                EXPECT_NEAR(actual[channel], expected[channel], 1e-6)
                    << "pixel=" << x << ',' << y << " channel=" << channel;
            }
        }
}
class FragmentDrawExec : public ::testing::Test {
protected:
    void SetUp() override {
        const auto& context = r::render_vk_ctx();
        ASSERT_TRUE(context.ok) << "a missing device is not execution evidence";
        ASSERT_TRUE(context.shader_int64_enabled);
        ASSERT_TRUE(context.float_transport.explicit_nonfinite32());
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(context.phys, VK_FORMAT_R32G32B32A32_SFLOAT,
                                            &properties);
        ASSERT_TRUE(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);
        ASSERT_TRUE(properties.optimalTilingFeatures &
                    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT);
        target.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        target.load_existing = false;
    }
    std::vector<uint8_t> render(const std::vector<r::BackendDraw>& draws, uint32_t width = f::width,
                                const std::array<float, 4>& clear = seed) {
        const auto result =
            r::render_draws_rgba(draws, width, f::height, nullptr, clear.data(), false, &target);
        const auto stats = r::fragment_draw_backend_stats();
        EXPECT_EQ(stats.planned, draws.size());
        EXPECT_EQ(stats.recorded, draws.size())
            << "actual same-backend collection/compute commands, not native fallback";
        EXPECT_EQ(stats.refused, 0u);
        std::fprintf(stderr,
                     "[fragment-draw-gpu] original_transactions=%llu recorded=%llu extent=%ux%u "
                     "raw_bytes=%zu\n",
                     static_cast<unsigned long long>(stats.planned),
                     static_cast<unsigned long long>(stats.recorded), width, f::height,
                     result.size());
        return result;
    }
    static constexpr std::array<float, 4> seed{.125f, .25f, .5f, 1.f};
    r::BackendColorTarget target;
};
TEST_F(FragmentDrawExec, ThreeLogicalWavesCommitAllOriginalRawPixelComponents) {
    g::DrawItem draw;
    ASSERT_TRUE(f::realize(draw));
    ASSERT_TRUE(draw.fragment_draw_inputs);
    ASSERT_EQ(*draw.fragment_draw_inputs->raw_code, f::fragment_words());
    pixels(render({backend(draw)}), f::width, f::height, f::color_a);
}
TEST_F(FragmentDrawExec, OriginalOrderedReplayUsesNormalFixedFunctionBlend) {
    g::DrawItem a, b;
    ASSERT_TRUE(f::realize(a, f::color_a));
    ASSERT_TRUE(f::realize(b, f::color_b));
    for (auto* draw : {&a, &b}) {
        draw->ps.blend_enable = true;
        draw->ps.src_color_blend_factor = VK_BLEND_FACTOR_SRC_ALPHA;
        draw->ps.dst_color_blend_factor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        draw->ps.color_blend_op = VK_BLEND_OP_ADD;
        draw->ps.src_alpha_blend_factor = VK_BLEND_FACTOR_ONE;
        draw->ps.dst_alpha_blend_factor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        draw->ps.alpha_blend_op = VK_BLEND_OP_ADD;
    }
    const auto expected = [&](const auto& first, const auto& second) {
        std::array<float, 4> result{};
        for (uint32_t channel = 0; channel < 3; ++channel)
            result[channel] =
                second[channel] * .5f + (first[channel] * .5f + seed[channel] * .5f) * .5f;
        result[3] = 1;
        return result;
    };
    const auto ab = expected(f::color_a, f::color_b), ba = expected(f::color_b, f::color_a);
    ASSERT_NE(ab, ba) << "order is a discriminatory numerical input";
    pixels(render({backend(a), backend(b)}), f::width, f::height, ab);
    pixels(render({backend(b), backend(a)}), f::width, f::height, ba);
}
TEST_F(FragmentDrawExec, APartialHostQuadRefusesTheWholeDrawAfterAPooledSuccess) {
    g::DrawItem draw;
    ASSERT_TRUE(f::realize(draw));
    const auto original = backend(draw);
    pixels(render({original}), f::width, f::height, f::color_a);
    // Same compiled code/capacity/private plane sizes after accepted gate1. An odd framebuffer
    // edge introduces genuine helper/partial topology. Until its backing authority is supported,
    // no earlier full quad may mutate ANY attachment pixel; stale pooled gate1 must not survive.
    pixels(render({original}, f::width - 1), f::width - 1, f::height, seed);
    pixels(render({original}), f::width, f::height, f::color_a);
}
TEST_F(FragmentDrawExec, SeventeenOriginalProgramsKeepVulkanObjectsWarmAcrossFrames) {
    std::array<g::DrawItem, 17> draws;
    std::vector<r::BackendDraw> batch;
    for (uint32_t index = 0; index < draws.size(); ++index) {
        ASSERT_TRUE(f::realize(draws[index], f::color_a, f::distinct_fragment_words(index + 200)));
        ASSERT_TRUE(draws[index].fragment_draw_inputs);
        ASSERT_EQ(*draws[index].fragment_draw_inputs->raw_code,
                  f::distinct_fragment_words(index + 200));
        batch.push_back(backend(draws[index]));
    }
    const auto before = g::fragment_draw_cache_stats();
    pixels(render(batch), f::width, f::height, f::color_a);
    const auto warm = g::fragment_draw_cache_stats();
    EXPECT_EQ(warm.program_compile_calls - before.program_compile_calls, 17u);
    EXPECT_EQ(warm.compute_cold_builds - before.compute_cold_builds, 17u);
    EXPECT_EQ(warm.collector_cold_builds - before.collector_cold_builds, 17u);
    EXPECT_EQ(warm.framebuffer_cold_builds - before.framebuffer_cold_builds, 17u);
    EXPECT_GT(warm.vk_object_create_calls, before.vk_object_create_calls);
    EXPECT_GT(warm.vk_pipeline_create_calls, before.vk_pipeline_create_calls);
    EXPECT_GT(warm.checked_shader_module_calls, before.checked_shader_module_calls);
    for (uint32_t frame = 0; frame < 3; ++frame)
        pixels(render(batch), f::width, f::height, f::color_a);
    const auto after = g::fragment_draw_cache_stats();
    EXPECT_EQ(after.program_compile_calls, warm.program_compile_calls);
    EXPECT_EQ(after.compute_cold_builds, warm.compute_cold_builds);
    EXPECT_EQ(after.collector_cold_builds, warm.collector_cold_builds);
    EXPECT_EQ(after.framebuffer_cold_builds, warm.framebuffer_cold_builds);
    EXPECT_EQ(after.vk_object_create_calls, warm.vk_object_create_calls)
        << "counts actual shipping companion Vulkan call sites, including failed attempts";
    EXPECT_EQ(after.vk_pipeline_create_calls, warm.vk_pipeline_create_calls);
    EXPECT_EQ(after.checked_shader_module_calls, warm.checked_shader_module_calls);
}
TEST_F(FragmentDrawExec, DeadSourceRetirementPreservesRetainedPipelinePayloads) {
    const auto& context = r::render_vk_ctx();
    const g::FragmentPacketDeviceContract device{reinterpret_cast<uintptr_t>(context.dev),
                                                 context.shader_int64_enabled, false};
    std::shared_ptr<const r::FragmentDrawComputeGpuProgram> compute;
    std::shared_ptr<const r::FragmentDrawCollectGpuProgram> collect;
    std::shared_ptr<const r::FragmentDrawCollectFramebuffer> framebuffer;
    std::weak_ptr<const std::vector<uint32_t>> source;
    VkPipeline original_compute = VK_NULL_HANDLE, original_collect = VK_NULL_HANDLE;
    VkFramebuffer original_framebuffer = VK_NULL_HANDLE;
    {
        g::DrawItem draw;
        ASSERT_TRUE(f::realize(draw, f::color_a, f::distinct_fragment_words(400)));
        const auto prepared = f::prepare(draw);
        ASSERT_TRUE(prepared && draw.fragment_draw_inputs && draw.vs_shared);
        source = draw.fragment_draw_inputs->raw_code;
        const auto plan =
            g::cached_fragment_draw_program(*draw.fragment_draw_inputs, *prepared, device, 51);
        ASSERT_TRUE(plan && plan->rejection_reason().empty());
        std::string refusal;
        compute = r::FragmentDrawComputeGpuProgram::acquire(context, plan, refusal);
        ASSERT_TRUE(compute) << refusal;
        collect = r::FragmentDrawCollectGpuProgram::acquire(context, plan, draw.vs_shared, {},
                                                            false, refusal);
        ASSERT_TRUE(collect) << refusal;
        framebuffer =
            r::FragmentDrawCollectFramebuffer::acquire(context, collect, f::width, f::height);
        ASSERT_TRUE(framebuffer);
        original_compute = compute->pipeline(r::FragmentDrawComputeGpuProgram::OriginalPs);
        original_collect = collect->pipeline();
        original_framebuffer = framebuffer->framebuffer();
        ASSERT_NE(original_compute, VK_NULL_HANDLE);
        ASSERT_NE(original_collect, VK_NULL_HANDLE);
        ASSERT_NE(original_framebuffer, VK_NULL_HANDLE);
        g::clear_shader_analysis_cache();
        EXPECT_FALSE(source.expired()) << "the real producing draw still owns its analysis";
    }
    EXPECT_TRUE(source.expired()) << "copied pipeline/VS payloads must not retain analysis";
    EXPECT_FALSE(compute->source_live());
    EXPECT_FALSE(collect->source_live());
    EXPECT_FALSE(framebuffer->source_live());
    const auto before = g::fragment_draw_cache_stats();
    g::DrawItem next;
    ASSERT_TRUE(f::realize(next, f::color_a, f::distinct_fragment_words(401)));
    const auto prepared = f::prepare(next);
    ASSERT_TRUE(prepared && next.fragment_draw_inputs && next.vs_shared);
    const auto plan =
        g::cached_fragment_draw_program(*next.fragment_draw_inputs, *prepared, device, 51);
    ASSERT_TRUE(plan && plan->rejection_reason().empty());
    std::string refusal;
    const auto next_compute = r::FragmentDrawComputeGpuProgram::acquire(context, plan, refusal);
    ASSERT_TRUE(next_compute) << refusal;
    const auto next_collect = r::FragmentDrawCollectGpuProgram::acquire(
        context, plan, next.vs_shared, {}, false, refusal);
    ASSERT_TRUE(next_collect) << refusal;
    const auto next_framebuffer =
        r::FragmentDrawCollectFramebuffer::acquire(context, next_collect, f::width, f::height);
    ASSERT_TRUE(next_framebuffer);
    const auto after = g::fragment_draw_cache_stats();
    EXPECT_GT(after.compute_retired, before.compute_retired);
    EXPECT_GT(after.collector_retired, before.collector_retired);
    EXPECT_GT(after.framebuffer_retired, before.framebuffer_retired);
    // Strong completion-style payload leases survive actual residence retirement. This case
    // creates/retains objects but does not claim that an unsubmitted completion ran on the GPU.
    EXPECT_EQ(compute->pipeline(r::FragmentDrawComputeGpuProgram::OriginalPs), original_compute);
    EXPECT_EQ(collect->pipeline(), original_collect);
    EXPECT_EQ(framebuffer->framebuffer(), original_framebuffer);
}
TEST(FragmentDrawReplayLimits, AllFivePrivatePlanesCountBeforeLayoutCreation) {
    VkPhysicalDeviceLimits limits{};
    limits.maxBoundDescriptorSets = 2;
    limits.maxPerStageDescriptorStorageBuffers = 7;
    limits.maxPerStageResources = 7;
    limits.maxDescriptorSetStorageBuffers = 7;
    EXPECT_EQ(r::fragment_draw_replay_descriptor_limits(limits, 2), nullptr);
    auto bad = limits;
    bad.maxPerStageDescriptorStorageBuffers = 6;
    EXPECT_STREQ(r::fragment_draw_replay_descriptor_limits(bad, 2),
                 "fragment-draw-replay-per-stage-storage-limit");
    bad = limits;
    bad.maxPerStageResources = 6;
    EXPECT_STREQ(r::fragment_draw_replay_descriptor_limits(bad, 2),
                 "fragment-draw-replay-per-stage-resource-limit");
    bad = limits;
    bad.maxDescriptorSetStorageBuffers = 6;
    EXPECT_STREQ(r::fragment_draw_replay_descriptor_limits(bad, 2),
                 "fragment-draw-replay-aggregate-storage-limit");
    bad = limits;
    bad.maxBoundDescriptorSets = 1;
    EXPECT_STREQ(r::fragment_draw_replay_descriptor_limits(bad, 2),
                 "fragment-draw-replay-descriptor-set-limit");
    EXPECT_STREQ(r::fragment_draw_replay_descriptor_limits(limits, UINT64_MAX),
                 "fragment-draw-replay-storage-count-overflow");
}
}   // namespace
