// #4894: distinct exact Float16 layers expose a dropped/clobbered layer coordinate. Execute the
// actual live backend and compare every gather word against an independent spatial/layer oracle.
#include "fixtures/compute_array_gather.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"
#include "shared/live/live_compute.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;
namespace fixture = prosper::test::compute_array_gather;
namespace live = prosper::frontend;
namespace {
class ComputeArrayGatherExecution : public testing::Test {
protected:
    void SetUp() override {
        live::execute_live_compute_items({});
        ASSERT_TRUE(live::live_compute_storage_image_device().initialized)
            << "production Vulkan initialization failed";
    }
};

void execute(fixture::Options options, uint32_t tile, bool selected_view, bool query = false,
             uint32_t count = fixture::layers) {
    const size_t linear_bytes =
        static_cast<size_t>(fixture::width) * fixture::height * sizeof(uint16_t);
    const size_t slice =
        tile ? tiled_surface_bytes(fixture::width, fixture::height, tile, 0, 2) : linear_bytes;
    ASSERT_GT(slice, 0u);
    const size_t offset = selected_view ? slice : 0;
    const size_t stride = slice + offset;
    const uint32_t first_layer = selected_view ? 2 : 0;
    std::vector<uint8_t> source(stride * (count + first_layer + 1), 0xa5);
    std::vector<uint16_t> linear(static_cast<size_t>(fixture::width) * fixture::height);
    for (uint32_t layer = 0; layer < count + first_layer + 1; ++layer) {
        for (uint32_t y = 0; y < fixture::height; ++y)
            for (uint32_t x = 0; x < fixture::width; ++x)
                linear[y * fixture::width + x] = float_to_half(fixture::texel(layer, x, y));
        auto* destination = source.data() + layer * stride + offset;
        if (tile)
            tile_surface(destination, reinterpret_cast<const uint8_t*>(linear.data()),
                         fixture::width, fixture::height, tile, 0, 2);
        else
            std::memcpy(destination, linear.data(), linear_bytes);
    }
    const auto original = source;
    constexpr size_t result_words = static_cast<size_t>(fixture::layers) * 4;
    std::array<uint32_t, result_words + 4> result;
    result.fill(0xcccccccc);
    auto resources = fixture::table(DataFormat::Float16, count);
    auto& image = resources.resources[0];
    image.gpu_addr = reinterpret_cast<uint64_t>(source.data() + first_layer * stride);
    image.size = static_cast<uint32_t>(source.size() - first_layer * stride);
    image.tile_mode = tile;
    image.layer_stride_bytes = static_cast<uint32_t>(stride);
    image.layer_mip_offset_bytes = offset;
    image.linear_row_pitch_bytes = tile ? 0u : fixture::width * sizeof(uint16_t);
    image.unnormalized = options.unnormalized;
    resources.resources[1].gpu_addr = reinterpret_cast<uint64_t>(result.data());
    const auto code = query ? fixture::query() : fixture::program(options);
    ComputeItem item;
    item.code_addr = fixture::diagnostic_address();
    item.spirv = fixture::compile(code, resources, item.code_addr);
    ASSERT_FALSE(item.spirv.empty());
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.user_sgprs = fixture::config().user_sgprs;
    item.launch.local_x = item.launch.threads_x = fixture::layers;
    item.launch.local_y = item.launch.local_z = item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    ASSERT_TRUE(live::execute_live_compute_items({item}))
        << last_terminal_reject_reason(item.code_addr);
    constexpr std::array<uint32_t, 4> dx{0, 1, 1, 0}, dy{1, 1, 0, 0};
    for (uint32_t invocation = 0; invocation < fixture::layers; ++invocation) {
        const auto layer = static_cast<uint32_t>(
            std::clamp(std::floor(static_cast<float>(invocation) + options.layer_bias + 0.5f), 0.0f,
                       static_cast<float>(count - 1)));
        for (uint32_t channel = 0; channel < 4; ++channel) {
            const uint32_t x = 2 + dx[channel] + (options.opcode == 0x57 ? 1 : 0);
            const uint32_t y = 3 + dy[channel] - (options.opcode == 0x57 ? 1 : 0);
            const uint32_t expected =
                query ? std::array<uint32_t, 4>{fixture::width, fixture::height, count, 1}[channel]
                      : std::bit_cast<uint32_t>(fixture::texel(first_layer + layer, x, y));
            EXPECT_EQ(result[invocation * 4 + channel], expected)
                << "invocation=" << invocation << " channel=" << channel;
        }
    }
    EXPECT_EQ(source, original) << "sampling must not alter guest image backing";
    for (size_t word = result_words; word < result.size(); ++word)
        EXPECT_EQ(result[word], 0xccccccccu) << "output guard";
}
} // namespace

TEST_F(ComputeArrayGatherExecution, EveryDynamicFloat16LayerHasDistinctGatherValues) {
    execute({}, 0, false);
}
TEST_F(ComputeArrayGatherExecution, Tile27DetilesEveryLayer) {
    execute({}, 27, false);
}
TEST_F(ComputeArrayGatherExecution, SelectedTile27ViewKeepsBaseStrideAndMipOffset) {
    execute({.opcode = 0x57, .nsa = true}, 27, true);
}
TEST_F(ComputeArrayGatherExecution, UnnormalizedSpatialOffsetDoesNotScaleLayer) {
    execute({.opcode = 0x57, .nsa = true, .unnormalized = true}, 0, false);
}
TEST_F(ComputeArrayGatherExecution, LayerRoundingAndClampingUseSelectedView) {
    execute({.layer_bias = -1.51f}, 0, true, false, 5);
    execute({.layer_bias = 0.51f}, 0, true, false, 5);
}
TEST_F(ComputeArrayGatherExecution, ResinfoReportsSelectedLayerCount) {
    execute({}, 27, true, true, 5);
}
TEST_F(ComputeArrayGatherExecution, SingleLayerArrayStillHasMatchingViewAndQuery) {
    execute({}, 0, false, false, 1);
    execute({}, 0, false, true, 1);
}
