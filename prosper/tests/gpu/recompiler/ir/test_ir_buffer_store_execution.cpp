// Store packing must match independently derived memory bytes, including inactive and guard words.
// This uses actual Vulkan for the existing atomic clear/set path; it assumes no atomic CPU model.
#include "gpu/recompiler/ir/buffer_store_cases.hpp"
#include "gpu/recompiler/ir/buffer_store_witness.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <string>
#include "gpu/recompiler/rdna2_decode.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

using namespace prosper::gpu;
namespace fixture = prosper::test::ir::buffer_store;
namespace {
TEST(IrBufferStoreExecution, ExportsReplaysAndExecutesWitnessWithMatchedRefusals) {
    namespace fs = std::filesystem;
    using namespace prosper::test::ir;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): the launch environment is immutable in this process.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const fs::path root = compare_witness_root("ir_buffer_store_witness");
    if (!configured) {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    const fs::path dump = compare_witness_dump("ir_buffer_store_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream attributes(root / "attributes.jsonl");
    size_t index = 0, accepted = 0, refused = 0;
    for (const auto& x : fixture::witness_cases()) {
        SCOPED_TRACE(index);
        auto p = fixture::program(x.c, x.opcode);
        if (x.routed) {
            p.table.resources[1].swizzle[0] = 6;
            p.table.resources[1].swizzle[2] = 4;
        }
        const uint64_t site = 0x5385100000ull + index * 0x1000;
        const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                              {RecompileDiagnosticStage::Compute, site});
        ASSERT_EQ(!module.empty(), x.admitted) << last_terminal_reject_reason(site);
        const std::string reason =
            x.admitted ? "synthetic-witness (no live refusal)" : last_terminal_reject_reason(site);
        const fs::path stem = dump / ("cs_" + std::to_string(site) + "_" + std::to_string(index));
        ASSERT_TRUE(fixture::save_words(stem.string() + ".bin", p.code));
        ASSERT_TRUE(write_refused_compute_inputs(stem.string() + ".inputs", site, &p.table,
                                                 p.config, reason));
        const auto replay = replay_refused_shader_inputs(
            p.code, read_refused_shader_inputs(stem.string() + ".inputs"));
        ASSERT_EQ(replay.produced, x.admitted) << replay.reject << " " << replay.incomplete;
        if (x.admitted) {
            ASSERT_EQ(replay.spirv, module);
            ASSERT_TRUE(fixture::save_words(stem.string() + ".spv", module));
            const auto reopened = fixture::reopen_words(stem.string() + ".spv", module.size());
            ASSERT_EQ(reopened, module);
            const auto d = fixture::data(x.c);
            std::vector<uint32_t> got;
            const auto dummy =
                prosper::test::run_compute(reopened, std::vector<float>(x.c.wave), x.c.wave,
                                           x.c.wave, d.input, d.initial, &got, x.c.wave);
            ASSERT_EQ(dummy.size(), x.c.wave);
            ASSERT_TRUE(fixture::save_words(stem.string() + ".buffer2", d.input));
            ASSERT_TRUE(fixture::save_words(stem.string() + ".buffer3", d.initial));
            ASSERT_TRUE(fixture::save_words(stem.string() + ".actual", got));
            ASSERT_TRUE(fixture::save_words(stem.string() + ".expected", d.expected));
            ASSERT_EQ(got, d.expected);
            ++accepted;
        } else {
            ASSERT_FALSE(reason.empty());
            ASSERT_EQ(replay.reject, reason);
            ++refused;
        }
        fixture::attributes(attributes, index++, x, stem);
    }
    attributes.close();
    ASSERT_TRUE(attributes.good());
    EXPECT_EQ(index, 146u);
    EXPECT_EQ(accepted, 96u);
    EXPECT_EQ(refused, 50u);
}
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
    EXPECT_EQ(fixture::field(c, 8, 0), 129u);   // -127 in two's complement, not -128
    c.format = 4;
    EXPECT_EQ(fixture::field(c, 4, 0), 0x3800u);
    EXPECT_EQ(fixture::field(c, 8, 0), 0xbc00u);
}
}   // namespace
