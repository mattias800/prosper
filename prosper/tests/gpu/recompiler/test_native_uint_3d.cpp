// #4885: a supported R16_UINT volume must not expand into an oversized RGBA32 storage image.
// Exact format/type capabilities select native storage; absent or 2D-only support keeps raw storage.
// These tests inspect production compiler declarations, not a second implementation of admission.
#include "fixtures/native_uint_3d.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace prosper::gpu;
namespace fixture = prosper::test::native_uint_3d;

namespace {
ShaderResourceTable table(const fixture::Format& format) {
    ShaderResourceTable result;
    result.resources.reserve(2);
    result.resources.push_back(fixture::image(format, 4, 0, 64, 3, 2, 0x10000, 4096));
    result.resources.push_back(fixture::image(format, 5, 8, 64, 3, 2, 0x20000, 4096));
    return result;
}

void expect_format(const fixture::Format& format, uint32_t capability, uint32_t expected,
                   ShaderResourceTable resources = {}) {
    if (resources.resources.empty()) resources = table(format);
    const auto code = fixture::copy_row();
    const auto module =
        recompile_compute(code.data(), code.size(), &resources, fixture::config(capability));
    ASSERT_FALSE(module.empty());
    const auto report = validate_spirv_descriptor_interface(module, &resources, 0,
                                                            SpirvShaderStage::Compute, false);
    ASSERT_TRUE(report.ok());
    ASSERT_EQ(report.descriptors.size(), 2u);
    for (const auto& descriptor : report.descriptors) {
        EXPECT_EQ(descriptor.kind, SpirvDescriptorKind::StorageImage);
        EXPECT_EQ(descriptor.image_dim, 2u);
        EXPECT_FALSE(descriptor.image_arrayed);
        EXPECT_FALSE(descriptor.image_multisampled);
        EXPECT_EQ(descriptor.image_numeric_class, SpirvImageNumericClass::Uint);
        EXPECT_EQ(descriptor.storage_image_format, expected);
    }
}

class NativeUint3d : public testing::TestWithParam<fixture::Format> {};
TEST_P(NativeUint3d, Exact3dCapabilitySelectsNativeFormat) {
    const auto format = GetParam();
    expect_format(format, format.capability, format.spirv_format);
}
TEST_P(NativeUint3d, TwoDimensionalCapabilityDoesNotAuthorizeVolume) {
    const auto format = GetParam();
    expect_format(format, native_storage_format_support_bit(format.format, format.components), 0);
}
TEST_P(NativeUint3d, MissingCapabilityKeepsRawInterchange) {
    expect_format(GetParam(), 0, 0);
}
TEST_P(NativeUint3d, Different3dFormatDoesNotAuthorizeVolume) {
    const auto format = GetParam();
    expect_format(format, format.capability == (1u << 25) ? 1u << 27 : 1u << 25, 0);
}
TEST_P(NativeUint3d, GammaEncodedStorageKeepsSemanticExclusion) {
    const auto format = GetParam();
    auto resources = table(format);
    for (auto& resource : resources.resources) resource.srgb = true;
    expect_format(format, format.capability, 0, resources);
}
TEST_P(NativeUint3d, MultisampleMetadataCannotSelectNativeVolume) {
    const auto format = GetParam();
    auto resources = table(format);
    for (auto& resource : resources.resources) resource.sample_count = 2;
    expect_format(format, format.capability, 0, resources);
}
INSTANTIATE_TEST_SUITE_P(Formats, NativeUint3d, testing::ValuesIn(fixture::formats),
                         [](const testing::TestParamInfo<fixture::Format>& info) {
                             return std::string(info.param.name);
                         });
}   // namespace

TEST(NativeUint3dContract, CapabilitiesPreserveCaptureBitsAndSeparateDimensions) {
    uint32_t seen = (1u << 25) - 1u;
    for (const auto& format : fixture::formats) {
        const uint32_t actual =
            native_storage_3d_format_support_bit(format.format, format.components);
        EXPECT_EQ(actual, format.capability);
        EXPECT_EQ(actual & seen, 0u);
        seen |= actual;
    }
    EXPECT_EQ(seen, kNativeStorageFormatSupportMask);
    EXPECT_EQ(native_storage_format_support_bit(DataFormat::Unorm2_10_10_10, 4), 1u << 24);
    EXPECT_EQ(native_storage_3d_format_support_bit(DataFormat::Float16, 4), 1u << 15);
    EXPECT_EQ(native_storage_3d_format_support_bit(DataFormat::Uint16, 4), 0u);
    EXPECT_EQ(native_storage_3d_format_support_bit(DataFormat::Uint8, 2), 0u);
    EXPECT_EQ(native_storage_3d_format_support_bit(DataFormat::Unorm2_10_10_10, 4), 0u);
    EXPECT_EQ(kNativeStorageFormatSupportMask & (7u << 29), 0u);
}

TEST(NativeUint3dContract, InstructionsReallyUseThreeCoordinates) {
    const auto code = fixture::copy_row(2, 1);
    std::vector<Rdna2Inst> decoded;
    ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
    uint32_t images = 0;
    for (const auto& instruction : decoded)
        if (instruction.fmt == Rdna2Format::MIMG) {
            ++images;
            EXPECT_EQ(instruction.mimg_dim, 2u);
            EXPECT_EQ(instruction.src[0].value, 4);
            EXPECT_EQ(instruction.mimg_dmask, 15u);
        }
    EXPECT_EQ(images, 2u);
}
