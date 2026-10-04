// Genuine registered original SMEM -> ordered checked seal -> normal frontend/shared bank GPU
// service -> WAT2 attachment replay. Every pixel and upload count below is an execution oracle,
// not a fabricated reflection or standalone bank. These finite input-free recipes are not Kena.
#include "fixtures/fragment_scalar_bank_fixture.hpp"
#include "fixtures/render_runner.h"
#include "shared/live/live_renderer.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace {
namespace g = prosper::gpu;
namespace r = prosper::test;
namespace f = r::scalar_bank;
namespace p = prosper::agc::Pm4;
class FragmentScalarBankGpu : public testing::Test {
protected:
    f::Scene scene;
    uint64_t submit = 9301;
    void SetUp() override {
        const auto& context = r::render_vk_ctx();
        ASSERT_TRUE(context.ok) << "a missing device is not scalar-bank execution evidence";
        ASSERT_TRUE(context.shader_int64_enabled);
        ASSERT_TRUE(context.float_transport.explicit_nonfinite32());
        ASSERT_EQ(r::backend_pending_submission_batches().load(), 0);
        ASSERT_FALSE(r::backend_has_unproven_submission());
        ASSERT_TRUE(scene.create());
        prosper::frontend::register_live_renderer(".", false);
    }
    void TearDown() override {
        g::retire_deferred_graphics();
        g::set_submit_renderer({});
        g::set_submit_compute({});
        g::set_deferred_graphics_retirer(nullptr);
        g::set_graphics_raw_allocation_authority({});
        g::set_graphics_raw_source_authority({});
        g::set_graphics_producer_status_query({});
        EXPECT_EQ(r::backend_pending_submission_batches().load(), 0);
        EXPECT_TRUE(scene.data.close());
        EXPECT_TRUE(scene.color.close());
    }
    g::GpuState state() const {
        auto value = scene.state();
        // The submitted target is genuinely RGBA8_UNORM; input buffer components stay raw FP32.
        // This exercises build_backend_draws and the shipping selected-frame/present transport.
        value.cx[p::CB_COLOR0_INFO] = 0xau << p::CB_COLOR0_INFO_FORMAT_SHIFT;
        return value;
    }
    std::vector<uint8_t> render(const g::GpuState& value, uint64_t bank_draws = 1) {
        const auto before = g::fragment_draw_cache_stats();
        const auto generation = r::backend_failed_publication_generation().load();
        const bool published = g::execute_ordered_and_present(
            value, r::fragment_draw::width, r::fragment_draw::height, submit++, true);
        EXPECT_TRUE(published) << "the real frontend must publish this complete normal submit";
        g::PresentFrameLease frame;
        if (!published || !g::present_acquire_rendered_frame(frame) || !frame.rgba) {
            ADD_FAILURE() << "missing actual selected frame";
            return {};
        }
        EXPECT_EQ(frame.width, r::fragment_draw::width);
        EXPECT_EQ(frame.height, r::fragment_draw::height);
        const auto stats = r::fragment_draw_backend_stats();
        EXPECT_EQ(stats.planned, bank_draws);
        EXPECT_EQ(stats.recorded, bank_draws) << "not a native fallback or guest-memory republish";
        EXPECT_EQ(stats.refused, 0u);
        const auto after = g::fragment_draw_cache_stats();
        EXPECT_EQ(after.scalar_bank_uploads - before.scalar_bank_uploads, bank_draws);
        EXPECT_EQ(after.scalar_bank_payload_bytes - before.scalar_bank_payload_bytes,
                  bank_draws * 16u)
            << "one small bank payload per draw, shared by every wave";
        EXPECT_EQ(r::backend_failed_publication_generation().load(), generation);
        return *frame.rgba;
    }
    static void pixels(const std::vector<uint8_t>& raw, const std::array<float, 4>& expected) {
        ASSERT_EQ(raw.size(), size_t(r::fragment_draw::width) * r::fragment_draw::height * 4);
        for (uint32_t y = 0; y < r::fragment_draw::height; ++y)
            for (uint32_t x = 0; x < r::fragment_draw::width; ++x)
                for (uint32_t c = 0; c < 4; ++c)
                    EXPECT_NEAR(raw[(size_t(y) * r::fragment_draw::width + x) * 4 + c],
                                std::lround(expected[c] * 255.f), 1)
                        << "pixel=" << x << ',' << y << " channel=" << c;
    }
    static void append(g::GpuState& sequence, const g::GpuState& actual, uint64_t order) {
        auto draw = sequence.draws[0];
        draw.command_order = order;
        draw.state = std::make_shared<g::GpuState>(actual);
        sequence.draws.push_back(std::move(draw));
    }
};

TEST_F(FragmentScalarBankGpu, ThreeLogicalWavesUseOneRealBufferUploadAndAllRawComponents) {
    static_assert(r::fragment_draw::width * r::fragment_draw::height == 3 * 64);
    ASSERT_NE(scene.descriptor[3], 0u)
        << "raw SMEM ignores format fields; identity still includes them";
    pixels(render(state()), r::fragment_draw::color_a);
}
TEST_F(FragmentScalarBankGpu, ObservedZeroScalarMemoryIsDataRatherThanAbsentBacking) {
    const std::array<float, 4> zero{};
    scene.write(zero);
    pixels(render(state()), zero);
    scene.write(r::fragment_draw::color_b);
    pixels(render(state()), r::fragment_draw::color_b);
}
TEST_F(FragmentScalarBankGpu, ChangedBytesAddressAndDeclaredExtentReuseTheWarmCodeSchema) {
    auto value = state();
    pixels(render(value), r::fragment_draw::color_a);
    const auto warm = g::fragment_draw_cache_stats();
    scene.write(r::fragment_draw::color_b, 0x100);
    const uint64_t address = scene.data.address + 0x100;
    value.sh[p::SPI_SHADER_USER_DATA_PS_0] = uint32_t(address);
    value.sh[p::SPI_SHADER_USER_DATA_PS_0 + 1] = uint32_t(address >> 32u);
    value.sh[p::SPI_SHADER_USER_DATA_PS_0 + 2] = 16;
    value.sh[p::SPI_SHADER_USER_DATA_PS_0 + 3] = 0x31000000u;
    pixels(render(value), r::fragment_draw::color_b);
    const auto after = g::fragment_draw_cache_stats();
    EXPECT_EQ(after.program_compile_calls, warm.program_compile_calls);
    EXPECT_EQ(after.compute_cold_builds, warm.compute_cold_builds);
    EXPECT_EQ(after.collector_cold_builds, warm.collector_cold_builds);
    EXPECT_EQ(after.framebuffer_cold_builds, warm.framebuffer_cold_builds);
    EXPECT_EQ(after.vk_pipeline_create_calls, warm.vk_pipeline_create_calls);
    EXPECT_EQ(after.checked_shader_module_calls, warm.checked_shader_module_calls);
}
TEST_F(FragmentScalarBankGpu, GenuineLargeUnusedDeclaredTailDoesNotUploadUnusedGuestBytes) {
    auto value = state();
    value.sh[p::SPI_SHADER_USER_DATA_PS_0 + 1] |= 16u << 16u;
    value.sh[p::SPI_SHADER_USER_DATA_PS_0 + 2] = UINT32_MAX;
    pixels(render(value), r::fragment_draw::color_a);
}
TEST_F(FragmentScalarBankGpu, AdjacentMixedNativeAndBankDrawsCommitThroughOneNormalPass) {
    auto value = state();
    auto native = state();
    auto* source = f::register_original(false, r::fragment_draw::fragment_words());
    ASSERT_TRUE(source);
    for (const auto& reg : source->registers) native.sh[reg.offset] = reg.value;
    native.cx[p::SPI_PS_IN_CONTROL] = 1u << p::SPI_PS_IN_CONTROL_PS_W32_EN_SHIFT;
    for (uint32_t word = 0; word < 4; ++word)
        native.sh[p::SPI_SHADER_USER_DATA_PS_0 + word] =
            std::bit_cast<uint32_t>(r::fragment_draw::color_b[word]);
    append(value, native, 200);
    append(value, state(), 300);
    pixels(render(value, 2), r::fragment_draw::color_a);
}
}   // namespace
