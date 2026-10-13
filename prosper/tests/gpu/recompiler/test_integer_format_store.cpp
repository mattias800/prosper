// #5007: aligned subword stores must retain the existing atomic field update, and direct
// compute fetches must publish the descriptor's routing rather than default identity.
// These are CPU compiler/publication guards, not device execution or title output evidence.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::gpu {
namespace {

std::vector<uint32_t> store(DataFormat format, uint32_t stride = 4u, uint32_t first = 0xe0102000u,
                            uint32_t second = 0x80020100u, bool routed = false) {
    ShaderResourceTable table;
    ShaderResource resource;
    resource.cls = ResourceClass::ConstantBuffer;
    resource.format = format;
    resource.num_components = 1;
    resource.binding = 3;
    resource.stride = stride;
    resource.size = 64;
    resource.sgpr_base = 8;
    resource.fetch_pc = 2;
    if (routed) resource.swizzle[0] = 5;
    table.resources.push_back(resource);
    const std::array<uint32_t, 5> code{0x7e000280u, 0x7e020281u, first, second, 0xbf810000u};
    return recompile_valu(code.data(), code.size(), 1, 0, &table);
}

size_t opcode_count(const std::vector<uint32_t>& module, uint16_t opcode) {
    size_t count = 0;
    for (size_t i = 5; i < module.size();) {
        const size_t words = module[i] >> 16u;
        if (!words || words > module.size() - i) return SIZE_MAX;
        if ((module[i] & 0xffffu) == opcode) ++count;
        i += words;
    }
    return count;
}

void expect_atomic_pair(const std::vector<uint32_t>& module) {
    ASSERT_FALSE(module.empty());
    EXPECT_EQ(opcode_count(module, 240), 1u) << "OpAtomicAnd clears only the selected field";
    EXPECT_EQ(opcode_count(module, 241), 1u) << "OpAtomicOr writes the selected field";
}

void expect_swizzle(uint32_t selectors, const std::array<uint8_t, 4>& expected) {
    std::array<uint32_t, 16> backing{};
    const auto base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(backing.data()));
    std::array<uint32_t, 12> seed{};
    seed[8] = static_cast<uint32_t>(base);
    seed[9] = (static_cast<uint32_t>(base >> 32u) & 0xffffu) | (4u << 16u);
    seed[10] = 16;
    seed[11] = (2u << 12u) | selectors;
    const std::array<uint32_t, 3> code{0xe00c2000u, 0x80020100u, 0xbf810000u};
    ShaderResourceTable table;
    const auto uses =
        add_compute_buffer_resources(table, code.data(), code.size(), seed.data(), seed.size());
    EXPECT_TRUE(uses.empty());
    ASSERT_EQ(table.resources.size(), 1u);
    const auto& resource = table.resources.front();
    EXPECT_EQ(resource.gpu_addr, base);
    EXPECT_EQ(resource.fetch_pc, 0u);
    EXPECT_EQ(resource.srt_offset, UINT32_MAX);
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(resource.swizzle[i], expected[i]) << "component " << i;
}

TEST(ComputeBufferSwizzle, NonidentityDirectFetch) {
    expect_swizzle(0xf66u, {6, 4, 5, 7});
}

TEST(ComputeBufferSwizzle, IdentityDirectFetch) {
    expect_swizzle(0xfacu, {4, 5, 6, 7});
}

TEST(IntegerFormatStore, AlignedUint16UsesAtomicPair) {
    expect_atomic_pair(store(DataFormat::Uint16));
}

TEST(IntegerFormatStore, AlignedUint8UsesAtomicPair) {
    expect_atomic_pair(store(DataFormat::Uint8));
}

TEST(IntegerFormatStore, PackedUint16StrideRetainsAtomicPair) {
    expect_atomic_pair(store(DataFormat::Uint16, 2));
}

TEST(IntegerFormatStore, MisalignedUint16OffsetRefuses) {
    EXPECT_TRUE(store(DataFormat::Uint16, 4, 0xe0102001u).empty());
}

TEST(IntegerFormatStore, RegisterSoffsetRefuses) {
    EXPECT_TRUE(store(DataFormat::Uint16, 4, 0xe0102000u, 0x00020100u).empty());
}

TEST(IntegerFormatStore, PerLaneByteOffsetRefuses) {
    EXPECT_TRUE(store(DataFormat::Uint16, 4, 0xe0103000u).empty());
}

TEST(IntegerFormatStore, RoutedMubufStoreRefuses) {
    EXPECT_TRUE(store(DataFormat::Uint16, 4, 0xe0102000u, 0x80020100u, true).empty());
}

} // namespace
} // namespace prosper::gpu
