// #4885: execute native unsigned volumes through the production compute backend. A wrong width or
// z/slice layout silently changes guest bytes. Independent whole-volume oracles cover selected
// rows, untouched texels and narrow-store high-bit truncation, with the raw interchange control.
#include "fixtures/native_uint_3d.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/texture/tile.hpp"
#include "shared/live/live_compute.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace prosper::gpu;
namespace fixture = prosper::test::native_uint_3d;
namespace live = prosper::frontend;

namespace {
constexpr uint32_t width = 64, height = 3, depth = 3, mode = 5;

ComputeItem compile(const fixture::Format& format, const std::vector<uint32_t>& code,
                    const ShaderResourceTable& resources, bool native) {
    ComputeItem item;
    const auto config = fixture::config(native ? format.capability : 0);
    item.spirv = recompile_compute(code.data(), code.size(), &resources, config);
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.user_sgprs = config.user_sgprs;
    item.launch.threads_x = item.launch.local_x = width;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    item.code_addr = 0x4885000000ull + format.capability;
    return item;
}

void expect_declarations(const ComputeItem& item, uint32_t format, size_t count) {
    ASSERT_FALSE(item.spirv.empty());
    const auto report = validate_spirv_descriptor_interface(item.spirv, item.resources.get(), 0,
                                                            SpirvShaderStage::Compute, false);
    ASSERT_TRUE(report.ok());
    ASSERT_EQ(report.descriptors.size(), count);
    for (const auto& descriptor : report.descriptors) {
        EXPECT_EQ(descriptor.kind, SpirvDescriptorKind::StorageImage);
        EXPECT_EQ(descriptor.image_dim, 2u);
        EXPECT_EQ(descriptor.storage_image_format, format);
    }
}

class NativeUint3dExecution : public testing::TestWithParam<fixture::Format> {
protected:
    void SetUp() override {
        live::execute_live_compute_items({});
        const auto device = live::live_compute_storage_image_device();
        ASSERT_TRUE(device.initialized) << "production Vulkan initialization failed";
        if (!device.features.storage_image_capable())
            GTEST_SKIP() << "device cannot execute the production storage-image contract";
        const auto format = GetParam();
        if (!live::live_compute_native_storage_3d_supported(format.format, format.components, width,
                                                            height, depth))
            GTEST_SKIP() << "selected device does not support this exact 3D storage format";
    }
};

TEST_P(NativeUint3dExecution, CopySelectedRowsAndRetainUntouchedTexels) {
    const auto format = GetParam();
    const size_t bytes = static_cast<size_t>(width) * height * depth * format.bytes;
    const size_t tiled = tiled_volume_bytes(width, height, depth, mode, format.bytes);
    ASSERT_GT(tiled, 0u);
    std::vector<uint8_t> source_linear(bytes), expected(bytes);
    for (size_t i = 0; i < bytes; ++i) {
        source_linear[i] = static_cast<uint8_t>(i * 197u + (i / 17u) * 31u + 0x81u);
        expected[i] = static_cast<uint8_t>(i * 97u + 0x3bu);
    }
    const auto initial = expected;
    std::vector<uint8_t> source(tiled), destination(tiled + 32, 0xa5), oracle(tiled);
    ASSERT_TRUE(tile_volume(source.data(), tiled, source_linear.data(), width, height, depth, mode,
                            format.bytes));
    for (bool native : {false, true}) {
        expected = initial;
        ASSERT_TRUE(tile_volume(destination.data(), tiled, expected.data(), width, height, depth,
                                mode, format.bytes));
        ShaderResourceTable resources;
        resources.resources.reserve(2);
        resources.resources.push_back(fixture::image(format, 4, 0, width, height, depth,
                                                     reinterpret_cast<uint64_t>(source.data()),
                                                     static_cast<uint32_t>(tiled), mode));
        resources.resources.push_back(fixture::image(format, 5, 8, width, height, depth,
                                                     reinterpret_cast<uint64_t>(destination.data()),
                                                     static_cast<uint32_t>(tiled), mode));
        for (const auto row : std::array<std::array<uint32_t, 2>, 3>{{{0, 0}, {2, 2}, {1, 1}}}) {
            const size_t offset =
                (static_cast<size_t>(row[1]) * height + row[0]) * width * format.bytes;
            std::copy_n(source_linear.data() + offset, width * format.bytes,
                        expected.data() + offset);
            ASSERT_TRUE(tile_volume(oracle.data(), tiled, expected.data(), width, height, depth,
                                    mode, format.bytes));
            const auto item = compile(format, fixture::copy_row(row[0], row[1]), resources, native);
            expect_declarations(item, native ? format.spirv_format : 0u, 2);
            ASSERT_TRUE(live::execute_live_compute_items({item}));
            EXPECT_TRUE(std::equal(oracle.begin(), oracle.end(), destination.begin()))
                << "all physical bytes, native=" << native << " y=" << row[0] << " z=" << row[1];
            EXPECT_TRUE(std::all_of(destination.begin() + tiled, destination.end(),
                                    [](uint8_t value) { return value == 0xa5; }));
        }
    }
}

TEST_P(NativeUint3dExecution, StoresDiscardHighBitsInEveryWrittenChannel) {
    const auto format = GetParam();
    const size_t bytes = static_cast<size_t>(width) * height * depth * format.bytes;
    const size_t tiled = tiled_volume_bytes(width, height, depth, mode, format.bytes);
    ASSERT_GT(tiled, 0u);
    std::vector<uint8_t> expected(bytes, 0x5a), destination(tiled), oracle(tiled);
    constexpr std::array<uint32_t, 4> values{0x89abcdefu, 0xffffffffu, 0x12345678u, 0x87654321u};
    // Coordinates are retained in v4:6 before VDATA is overwritten by arbitrary B32 values.
    std::vector<uint32_t> code{0x7e080300u, 0x7e0a0281u, 0x7e0c0282u};
    code.reserve(14);
    for (uint32_t channel = 0; channel < 4; ++channel) {
        code.push_back(0x7e0002ffu | (channel << 17));
        code.push_back(values[channel]);
    }
    code.insert(code.end(), {0xf0200f10u, 0x00020004u, 0xbf810000u});
    for (bool native : {false, true}) {
        std::fill(expected.begin(), expected.end(), 0x5a);
        ASSERT_TRUE(tile_volume(destination.data(), tiled, expected.data(), width, height, depth,
                                mode, format.bytes));
        for (uint32_t x = 0; x < width; ++x)
            for (uint32_t channel = 0; channel < format.components; ++channel)
                for (uint32_t byte = 0; byte < format.bytes / format.components; ++byte)
                    expected[((size_t{2} * height + 1) * width + x) * format.bytes +
                             size_t{channel} * (format.bytes / format.components) + byte] =
                        static_cast<uint8_t>(values[channel] >> (byte * 8));
        ASSERT_TRUE(tile_volume(oracle.data(), tiled, expected.data(), width, height, depth, mode,
                                format.bytes));
        ShaderResourceTable resources;
        resources.resources.push_back(fixture::image(format, 5, 8, width, height, depth,
                                                     reinterpret_cast<uint64_t>(destination.data()),
                                                     static_cast<uint32_t>(tiled), mode));
        const auto item = compile(format, code, resources, native);
        expect_declarations(item, native ? format.spirv_format : 0u, 1);
        ASSERT_TRUE(live::execute_live_compute_items({item}));
        EXPECT_EQ(destination, oracle) << "native=" << native;
    }
}

INSTANTIATE_TEST_SUITE_P(Formats, NativeUint3dExecution, testing::ValuesIn(fixture::formats),
                         [](const testing::TestParamInfo<fixture::Format>& info) {
                             return std::string(info.param.name);
                         });
} // namespace
