// Store packing must match independently derived memory bytes, including inactive and guard words.
// This uses actual Vulkan for the existing atomic clear/set path; it assumes no atomic CPU model.
#include "gpu/recompiler/ir/buffer_store_cases.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;
namespace fixture = prosper::test::ir::buffer_store;
namespace {
TEST(IrBufferStoreExecution, FullFormatsMatchIndependentBytesInBothWaveSizes) {
    size_t index = 0;
    for (const auto& c : fixture::cases()) {
        SCOPED_TRACE(::testing::Message()
                     << "case=" << index << " fmt=" << c.format << " typed=" << c.typed
                     << " offset=" << c.offset << " wave=" << c.wave);
        const auto p = fixture::program(c);
        const auto in = rdna2_decode_one(p.code.data() + 14, p.code.size() - 14);
        ASSERT_EQ(in.fmt, c.typed ? Rdna2Format::MTBUF : Rdna2Format::MUBUF);
        ASSERT_EQ(in.opcode, 7u);
        const auto module = recompile_compute(
            p.code.data(), p.code.size(), &p.table, p.config,
            {RecompileDiagnosticStage::Compute, 0x5385000000ull + index * 0x1000});
        ASSERT_FALSE(module.empty())
            << last_terminal_reject_reason(0x5385000000ull + index * 0x1000);
        const auto d = fixture::data(c);
        std::vector<uint32_t> got;
        const auto dummy = prosper::test::run_compute(module, std::vector<float>(c.wave), c.wave,
                                                      c.wave, d.input, d.initial, &got, c.wave);
        ASSERT_EQ(dummy.size(), c.wave);
        ASSERT_EQ(got, d.expected);
        ++index;
    }
    EXPECT_EQ(index, 88u);
}
TEST(IrBufferStoreExecution, PartialPackedWordAndRoutedStoresRemainRefused) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t f : {9u, 10u, 11u, 12u, 13u})
            for (bool typed : {false, true}) {
                const auto p = fixture::program({f, wave, 0, typed}, 5);
                EXPECT_TRUE(
                    recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
            }
    for (uint32_t wave : {32u, 64u}) {
        auto p = fixture::program({0, wave, 0, false});
        p.table.resources[1].swizzle[0] = 6;
        p.table.resources[1].swizzle[2] = 4;
        EXPECT_TRUE(recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
    }
}
TEST(IrBufferStoreExecution, MisalignedNonIntegerAndShortFieldsRemainRefused) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t f : {2u, 3u, 4u, 5u, 6u, 7u, 8u})
            for (bool typed : {false, true}) {
                const auto p = fixture::program({f, wave, 1, typed});
                EXPECT_TRUE(
                    recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
            }
}
TEST(IrBufferStoreExecution, SamplesPinIndependentHalfwayAndClampingAnswers) {
    fixture::Case c{5, 32, 0, false};
    // lane8/component0 selects eighths[1]=-1, lane4 selects index7=0.5.
    EXPECT_EQ(fixture::sample_index(4, 0), 7u);
    EXPECT_EQ(fixture::field(c, 4, 0), 128u);
    EXPECT_EQ(fixture::field(c, 8, 0), 0u);
    c.format = 6;
    EXPECT_EQ(fixture::field(c, 4, 0), 64u);
    EXPECT_EQ(fixture::field(c, 8, 0), 129u); // -127 in two's complement, not -128
    c.format = 4;
    EXPECT_EQ(fixture::field(c, 4, 0), 0x3800u);
    EXPECT_EQ(fixture::field(c, 8, 0), 0xbc00u);
}
} // namespace
