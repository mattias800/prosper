// #4900: legal unsigned D16 stores previously refused. Inspect actual production modules and
// their constant image-write values so wrong sign/half conversion fails without a GPU.
#include "fixtures/uint_d16_store.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace prosper::gpu;
namespace fixture = prosper::test::uint_d16_store;
namespace {
// Closed integer-SSA observation, not a reimplementation of the guest instruction. Unknown
// values/opcodes cannot become zero; an unevaluable image texel returns no observation.
std::optional<std::array<uint32_t, 4>> stored_constants(const std::vector<uint32_t>& module) {
    std::map<uint32_t, uint32_t> words;
    std::map<uint32_t, std::array<uint32_t, 4>> vectors;
    for (size_t at = 5; at < module.size();) {
        const uint32_t count = module[at] >> 16, opcode = module[at] & 0xffffu;
        if (!count || at + count > module.size()) return std::nullopt;
        if (opcode == 43 && count == 4) words[module[at + 2]] = module[at + 3]; // Constant
        if ((opcode == 83 || opcode == 124) && count == 4 && words.contains(module[at + 3]))
            words[module[at + 2]] = words.at(module[at + 3]); // CopyObject/Bitcast
        if ((opcode == 194 || opcode == 199) && count == 5 && words.contains(module[at + 3]) &&
            words.contains(module[at + 4])) {
            const uint32_t left = words.at(module[at + 3]), right = words.at(module[at + 4]);
            if (opcode == 199 || right < 32)
                words[module[at + 2]] = opcode == 199 ? left & right : left >> right;
        }
        if (opcode == 80 && count == 7) {
            std::array<uint32_t, 4> value{};
            bool known = true;
            for (uint32_t c = 0; c < 4; ++c) {
                const auto it = words.find(module[at + 3 + c]);
                if (it == words.end())
                    known = false;
                else
                    value[c] = it->second;
            }
            if (known) vectors[module[at + 2]] = value;
        }
        if (opcode == 99 && count >= 4 && vectors.contains(module[at + 3]))
            return vectors.at(module[at + 3]); // ImageWrite texel
        at += count;
    }
    return std::nullopt;
}

void expect_values(fixture::Options options, std::array<uint32_t, 4> expected) {
    const auto resources = fixture::table();
    const auto module = fixture::compile(fixture::program(options), resources);
    ASSERT_FALSE(module.empty());
    const auto observed = stored_constants(module);
    if (observed.has_value())
        EXPECT_EQ(*observed, expected);
    else
        ADD_FAILURE() << "actual image-write texel must be independently evaluable";
}

void expect_site_reject(const fixture::Program& code, const ShaderResourceTable& resources) {
    const auto address = fixture::diagnostic_address();
    EXPECT_TRUE(fixture::compile(code, resources, 0, address).empty());
    EXPECT_NE(
        last_terminal_reject_reason(address).find("pc=" + std::to_string(code.image_pc) + " "),
        std::string::npos)
        << last_terminal_reject_reason(address);
}
} // namespace

TEST(UintD16Store, UnsignedFormatsAndLowMasksCompileWithExactImageTypes) {
    for (const auto format : {DataFormat::Uint8, DataFormat::Uint16, DataFormat::Uint32})
        for (uint32_t channels : {1u, 4u})
            for (uint32_t dim : {1u, 5u})
                for (uint32_t mask : {1u, 3u, 7u, 15u}) {
                    const auto resources = fixture::table(format, channels, dim, dim == 5 ? 2 : 1);
                    for (bool native : {false, true}) {
                        const auto module = fixture::compile(
                            fixture::program({.dim = dim, .mask = mask}), resources,
                            native ? native_storage_format_support_bit(format, channels) : 0);
                        ASSERT_FALSE(module.empty())
                            << "format=" << static_cast<uint32_t>(format)
                            << " channels=" << channels << " dim=" << dim << " mask=" << mask;
                        const auto report = validate_spirv_descriptor_interface(
                            module, &resources, 0, SpirvShaderStage::Compute, false);
                        ASSERT_TRUE(report.ok());
                        ASSERT_EQ(report.descriptors.size(), 1u);
                        const auto& image = report.descriptors.front();
                        EXPECT_EQ(image.kind, SpirvDescriptorKind::StorageImage);
                        EXPECT_EQ(image.image_numeric_class, SpirvImageNumericClass::Uint);
                        EXPECT_EQ(image.image_dim, 1u);
                        EXPECT_EQ(image.image_arrayed, dim == 5);
                        EXPECT_TRUE(image.writable);
                    }
                }
}
TEST(UintD16Store, PackedHalvesZeroExtendAndOddCountsIgnoreTheUnusedHalf) {
    for (uint32_t mask : {1u, 3u, 7u, 15u}) {
        auto expected = fixture::components;
        for (uint32_t channel = 0; channel < 4; ++channel)
            if (!(mask & (1u << channel))) expected[channel] = 0;
        expect_values({.mask = mask}, expected);
    }
}
TEST(UintD16Store, NonD16UnsignedStoresKeepTheirFullWords) {
    expect_values({.d16 = false}, fixture::packed);
}
TEST(UintD16Store, FloatD16AndSparseFloatMasksRetainExistingLowering) {
    for (uint32_t mask : {1u, 3u, 5u, 9u, 15u})
        ASSERT_FALSE(
            fixture::compile(fixture::program({.mask = mask}), fixture::table(DataFormat::Float16))
                .empty());
}
TEST(UintD16Store, SparseUnsignedMasksStayRefusedBesideAValidControl) {
    const auto resources = fixture::table();
    ASSERT_FALSE(fixture::compile(fixture::program(), resources).empty());
    for (uint32_t mask : {0u, 2u, 5u, 9u, 14u})
        expect_site_reject(fixture::program({.mask = mask}), resources);
}
TEST(UintD16Store, SignedAndLoadFormsDoNotGainUnsignedStoreAuthority) {
    ASSERT_FALSE(fixture::compile(fixture::program(), fixture::table()).empty());
    expect_site_reject(fixture::program(), fixture::table(DataFormat::Sint32));
    expect_site_reject(fixture::program({.opcode = 0}), fixture::table());
}
TEST(UintD16Store, AddressStatusAndCacheModifiersRemainUnmodeled) {
    const auto resources = fixture::table();
    const auto good = fixture::program();
    ASSERT_FALSE(fixture::compile(good, resources).empty());
    constexpr std::array<std::array<uint32_t, 2>, 10> modifiers{
        {{1, 30}, {0, 15}, {0, 16}, {0, 17}, {0, 7}, {0, 13}, {0, 25}, {0, 6}, {0, 14}, {1, 26}}};
    for (const auto modifier : modifiers) {
        auto bad = good;
        bad.words[bad.image_pc + modifier[0]] |= 1u << modifier[1];
        expect_site_reject(bad, resources);
    }
}
TEST(UintD16Store, MipAndAtomicRestrictionsArePreserved) {
    ASSERT_FALSE(fixture::compile(fixture::program(), fixture::table()).empty());
    expect_site_reject(fixture::program({.opcode = 9}), fixture::table());
    expect_site_reject(fixture::program({.mask = 1, .opcode = 0x11}),
                       fixture::table(DataFormat::Uint32, 1));
}
TEST(UintD16Store, DecoderRetainsD16AndThePackedDataFootprint) {
    for (uint32_t mask : {1u, 3u, 7u, 15u}) {
        const auto code = fixture::program({.dim = 5, .mask = mask});
        std::vector<Rdna2Inst> instructions;
        ASSERT_EQ(rdna2_walk(code.words.data(), code.words.size(), instructions),
                  code.words.size());
        const auto& image = instructions[instructions.size() - 2];
        EXPECT_EQ(image.opcode, 8u);
        EXPECT_TRUE(image.mimg_d16);
        EXPECT_FALSE(image.mimg_a16);
        EXPECT_EQ(image.mimg_dmask, mask);
        EXPECT_EQ(image.mimg_dim, 5u);
        EXPECT_EQ(image.dst.value, 0);
    }
}
