// #4894: compute Float16 DIM5 gathers must consume their actual layer and declare full arrays.
// A graphics upload predicate previously declared plain 2D, causing a correct backend refusal.
#include "fixtures/compute_array_gather.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_sampled_array.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;
namespace fixture = prosper::test::compute_array_gather;
namespace {
void expect_array(const std::vector<uint32_t>& module, const ShaderResourceTable& table) {
    ASSERT_FALSE(module.empty());
    const auto report =
        validate_spirv_descriptor_interface(module, &table, 0, SpirvShaderStage::Compute, false);
    for (const auto& issue : report.issues)
        EXPECT_FALSE(issue.error) << "binding=" << issue.binding
                                  << " code=" << static_cast<uint32_t>(issue.code);
    ASSERT_TRUE(report.ok());
    bool found = false;
    for (const auto& descriptor : report.descriptors)
        if (descriptor.binding == 4) {
            found = true;
            EXPECT_EQ(descriptor.kind, SpirvDescriptorKind::CombinedImageSampler);
            EXPECT_EQ(descriptor.image_dim, 1u);
            EXPECT_TRUE(descriptor.image_arrayed);
            EXPECT_EQ(descriptor.image_numeric_class, SpirvImageNumericClass::Float);
        }
    EXPECT_TRUE(found);
}

bool dynamic_gather_layer(const std::vector<uint32_t>& module) {
    uint32_t coordinate = 0, layer = 0;
    for (size_t at = 5; at < module.size();) {
        const uint32_t count = module[at] >> 16, opcode = module[at] & 0xffffu;
        if (!count || at + count > module.size()) return false;
        if (opcode == 96 && count >= 6) coordinate = module[at + 4]; // OpImageGather
        at += count;
    }
    for (size_t at = 5; at < module.size();) {
        const uint32_t count = module[at] >> 16, opcode = module[at] & 0xffffu;
        if (opcode == 80 && count == 6 && module[at + 2] == coordinate)
            layer = module[at + 5]; // vec3's independent layer operand
        at += count;
    }
    if (!layer) return false;
    // The actual layer is converted from a local ID, not substituted with a zero constant.
    for (uint32_t depth = 0; depth < 4; ++depth) {
        bool bitcast = false;
        for (size_t at = 5; at < module.size();) {
            const uint32_t count = module[at] >> 16, opcode = module[at] & 0xffffu;
            if (count == 4 && module[at + 2] == layer) {
                if (opcode == 112) return true; // OpConvertUToF
                if (opcode == 124) {
                    layer = module[at + 3];
                    bitcast = true;
                    break;
                }
            }
            at += count;
        }
        if (!bitcast) return false;
    }
    return false;
}
} // namespace

TEST(ComputeArrayGather, Float16AndFloat32RetainDynamicLayer) {
    for (const auto format : {DataFormat::Float16, DataFormat::Float32}) {
        const auto table = fixture::table(format);
        const auto module = fixture::compile(fixture::program(), table);
        expect_array(module, table);
        EXPECT_TRUE(dynamic_gather_layer(module));
    }
}

TEST(ComputeArrayGather, NsaAndOffsetRetainIndependentLayer) {
    for (bool nsa : {false, true})
        for (uint32_t opcode : {0x47u, 0x40u, 0x57u}) {
            const auto table = fixture::table();
            const auto code = fixture::program({.opcode = opcode, .nsa = nsa});
            std::vector<Rdna2Inst> decoded;
            ASSERT_EQ(rdna2_walk(code.words.data(), code.words.size(), decoded), code.words.size());
            const auto module = fixture::compile(code, table);
            expect_array(module, table);
            EXPECT_TRUE(dynamic_gather_layer(module)) << "opcode=" << opcode << " nsa=" << nsa;
        }
}

TEST(ComputeArrayGather, ResinfoUsesSelectedArrayIncludingSingleLayer) {
    for (uint32_t layers : {1u, fixture::layers}) {
        const auto table = fixture::table(DataFormat::Float16, layers);
        const auto module = fixture::compile(fixture::query(), table);
        expect_array(module, table);
        bool queried = false;
        for (size_t at = 5; at < module.size();) {
            const uint32_t count = module[at] >> 16;
            ASSERT_GT(count, 0u);
            ASSERT_LE(at + count, module.size());
            queried |= (module[at] & 0xffffu) == 103u; // OpImageQuerySizeLod
            at += count;
        }
        EXPECT_TRUE(queried);
    }
}

TEST(ComputeArrayGather, UnmodeledModifiersRefuseBesideSupportedControl) {
    const auto table = fixture::table();
    const auto code = fixture::program({.opcode = 0x57, .nsa = true});
    ASSERT_FALSE(fixture::compile(code, table).empty());
    struct Modifier {
        uint32_t word, bit;
    };
    constexpr std::array<Modifier, 10> modifiers{
        {{1, 30}, {1, 31}, {0, 15}, {0, 16}, {0, 17}, {0, 7}, {0, 13}, {0, 25}, {0, 6}, {0, 9}}};
    for (const auto modifier : modifiers) {
        auto bad = code;
        bad.words[bad.image_pc + modifier.word] |= 1u << modifier.bit;
        EXPECT_TRUE(fixture::compile(bad, table).empty())
            << "word=" << modifier.word << " bit=" << modifier.bit;
    }
}

TEST(ComputeArrayGather, GraphicsBaseSliceAndOtherFormatsKeepTheirOwnContract) {
    const auto code = fixture::program();
    std::vector<Rdna2Inst> decoded;
    ASSERT_EQ(rdna2_walk(code.words.data(), code.words.size(), decoded), code.words.size());
    const auto image = decoded[decoded.size() - 4];
    ASSERT_EQ(image.fmt, Rdna2Format::MIMG);
    auto resource = fixture::table(DataFormat::Float32).resources.front();
    auto ordinary = image;
    ordinary.mimg_dim = 1u;
    EXPECT_FALSE(sampled_array_contract(ordinary, resource, false).declared_arrayed);
    EXPECT_TRUE(sampled_array_contract(image, resource, false).declared_arrayed);
    resource.format = DataFormat::Float16;
    EXPECT_FALSE(sampled_array_contract(image, resource, false).declared_arrayed);
    resource.format = DataFormat::Unorm8;
    EXPECT_FALSE(sampled_array_contract(image, resource, true).float_array_gather);
    resource.format = DataFormat::Float16;
    resource.cls = ResourceClass::StorageImage;
    EXPECT_FALSE(sampled_array_contract(image, resource, true).float_array_gather);
}
