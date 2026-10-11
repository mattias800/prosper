// #4900: execute actual packed-half stores and compare every guest backing byte. Zero-only
// captured stores cannot distinguish half-float unpacking, signed extension or a wrong high word.
#include "fixtures/uint_d16_store.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/texture/tile.hpp"
#include "shared/live/live_compute.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;
namespace fixture = prosper::test::uint_d16_store;
namespace live = prosper::frontend;
namespace {
class UintD16StoreExecution : public testing::Test {
protected:
    void SetUp() override {
        live::execute_live_compute_items({});
        const auto device = live::live_compute_storage_image_device();
        ASSERT_TRUE(device.initialized) << "production Vulkan initialization failed";
        if (!device.features.storage_image_capable())
            GTEST_SKIP() << "device lacks the production storage-image contract";
    }
};

void execute(DataFormat format, uint32_t channels, fixture::Options options = {},
             bool native = false, bool selected = false, uint32_t tile = 0,
             std::array<uint32_t, 4> swizzle = {4, 5, 6, 7}) {
    const uint32_t bytes = data_format_bytes(format), texel_bytes = bytes * channels;
    const uint32_t layers = options.dim == 5 ? fixture::width : 1u;
    const size_t linear_bytes = static_cast<size_t>(fixture::width) * fixture::height * texel_bytes;
    const size_t slice =
        tile ? tiled_surface_bytes(fixture::width, fixture::height, tile, 0, texel_bytes)
             : linear_bytes;
    ASSERT_GT(slice, 0u);
    const size_t offset = selected ? slice : 0u, stride = slice + offset;
    const uint32_t first_layer = selected ? 2u : 0u;
    const uint32_t total_layers = first_layer + layers + 1u;
    std::vector<uint8_t> destination(stride * total_layers + 32u, 0xa5), oracle(destination);
    std::vector<uint8_t> linear(linear_bytes, 0x5a);
    auto pack_slice = [&](uint8_t* into, const std::vector<uint8_t>& from) {
        if (tile)
            tile_surface(into, from.data(), fixture::width, fixture::height, tile, 0, texel_bytes);
        else
            std::memcpy(into, from.data(), linear_bytes);
    };
    for (uint32_t layer = 0; layer < total_layers; ++layer) {
        pack_slice(destination.data() + layer * stride + offset, linear);
        pack_slice(oracle.data() + layer * stride + offset, linear);
    }
    for (uint32_t layer = 0; layer < layers; ++layer) {
        std::fill(linear.begin(), linear.end(), 0x5a);
        for (uint32_t x = 0; x < fixture::width; ++x) {
            if (options.predicated && x != 1) continue;
            const uint32_t written_layer = options.dim == 5 ? (options.dynamic_layer ? x : 1u) : 0u;
            if (written_layer != layer) continue;
            std::array<uint32_t, 4> values{};
            uint32_t rank = 0;
            for (uint32_t c = 0; c < 4; ++c) {
                if (!(options.mask & (1u << c))) continue;
                values[c] = options.d16 ? (options.data[rank / 2] >> ((rank & 1u) * 16u)) & 0xffffu
                                        : options.data[rank];
                ++rank;
            }
            for (uint32_t c = 0; c < channels; ++c) {
                uint32_t value = 0;
                for (uint32_t source = 0; source < 4; ++source)
                    if (swizzle[source] == c + 4u) value = values[source];
                for (uint32_t byte = 0; byte < bytes; ++byte)
                    linear[(fixture::width + x) * texel_bytes + c * bytes + byte] =
                        static_cast<uint8_t>(value >> (byte * 8u));
            }
        }
        pack_slice(oracle.data() + (first_layer + layer) * stride + offset, linear);
    }
    auto resources = fixture::table(format, channels, options.dim, layers);
    auto& image = resources.resources.front();
    image.gpu_addr = reinterpret_cast<uint64_t>(destination.data() + first_layer * stride);
    image.size = static_cast<uint32_t>(stride * layers);
    image.tile_mode = tile;
    image.linear_row_pitch_bytes = tile ? 0u : fixture::width * texel_bytes;
    if (selected) {
        image.layer_stride_bytes = static_cast<uint32_t>(stride);
        image.layer_mip_offset_bytes = offset;
    }
    std::copy(swizzle.begin(), swizzle.end(), image.swizzle);
    const uint32_t capabilities = native ? native_storage_format_support_bit(format, channels) : 0;
    const auto code = fixture::program(options);
    ComputeItem item;
    item.code_addr = fixture::diagnostic_address();
    item.spirv = fixture::compile(code, resources, capabilities, item.code_addr);
    ASSERT_FALSE(item.spirv.empty()) << last_terminal_reject_reason(item.code_addr);
    const auto report = validate_spirv_descriptor_interface(item.spirv, &resources, 0,
                                                            SpirvShaderStage::Compute, false);
    ASSERT_TRUE(report.ok());
    ASSERT_EQ(report.descriptors.size(), 1u);
    EXPECT_EQ(report.descriptors.front().storage_image_format,
              native ? kSpirvImageFormatR32ui : 0u);
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.user_sgprs = fixture::config(capabilities).user_sgprs;
    item.launch.local_x = item.launch.threads_x = fixture::width;
    item.launch.local_y = item.launch.local_z = item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    const auto cpu_fills = live::live_compute_cpu_fill_dispatches();
    ASSERT_TRUE(live::execute_live_compute_items({item}));
    EXPECT_EQ(live::live_compute_cpu_fill_dispatches(), cpu_fills)
        << "partial-row execution must use the actual device path";
    EXPECT_EQ(destination, oracle) << "all selected/unselected layers, padding and tail bytes";
}
} // namespace

TEST_F(UintD16StoreExecution, RawUint32StoresBothPackedWordsWithoutSignExtension) {
    execute(DataFormat::Uint32, 4);
}
TEST_F(UintD16StoreExecution, RawUint16StoresBothHalvesAtTheOriginalMemoryWidth) {
    execute(DataFormat::Uint16, 4);
}
TEST_F(UintD16StoreExecution, RawUint8TruncatesTheUnsigned16NumericValues) {
    execute(DataFormat::Uint8, 4);
}
TEST_F(UintD16StoreExecution, NativeR32UsesOnlyTheLowHalfForOneComponent) {
    execute(DataFormat::Uint32, 1, {.mask = 1}, true);
}
TEST_F(UintD16StoreExecution, Tile27SelectedArrayRetainsDynamicLayerAndOffsets) {
    execute(DataFormat::Uint32, 4, {.dim = 5, .dynamic_layer = true}, false, true, 27);
}
TEST_F(UintD16StoreExecution, NarrowedExecLeavesEveryInactiveTexelUntouched) {
    execute(DataFormat::Uint16, 4, {.mask = 7, .predicated = true});
}
TEST_F(UintD16StoreExecution, FloatSparseD16StoreKeepsItsExistingHalfConversion) {
    execute(DataFormat::Float16, 4, {.mask = 9, .data = {0x40003c00u, 0x44004200u, 0u, 0u}});
}
TEST_F(UintD16StoreExecution, NonD16StoreKeepsAllThirtyTwoBitsPerComponent) {
    execute(DataFormat::Uint32, 4, {.d16 = false});
}
TEST_F(UintD16StoreExecution, ExistingDstSelInverseStillRoutesAfterUnpacking) {
    execute(DataFormat::Uint32, 4, {}, false, false, 0, {6, 5, 4, 7});
}
