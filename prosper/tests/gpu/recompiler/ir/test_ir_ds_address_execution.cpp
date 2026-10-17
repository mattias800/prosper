// Reference DS accesses use host-computed addresses, so matching read/write offset mistakes cannot
// cancel. Inactive stores are checked by all-active reads of independently initialized LDS words.
#include "gpu/recompiler/ir/ds_address_cases.hpp"
#include "gpu/recompiler/ir/ds_address_witness.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>
using namespace prosper::gpu;
namespace fixture = prosper::test::ir::ds_address;
namespace {
TEST(IrDsAddressExecution, ExportsReplaysAndExecutesWitnessWithMatchedRefusals) {
    namespace fs = std::filesystem;
    using namespace prosper::test::ir;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): the launch environment is immutable in this process.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const fs::path root = compare_witness_root("ir_ds_address_witness");
    if (!configured) {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    const fs::path dump = compare_witness_dump("ir_ds_address_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream attributes(root / "attributes.jsonl");
    size_t index = 0, accepted = 0, refused = 0;
    for (const auto& x : fixture::witness_cases()) {
        SCOPED_TRACE(index);
        const auto p = fixture::witness_program(x);
        const uint64_t site = 0x5387100000ull + index * 0x1000;
        const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                              {RecompileDiagnosticStage::Compute, site});
        ASSERT_EQ(!module.empty(), x.admitted()) << last_terminal_reject_reason(site);
        const std::string reason = x.admitted() ? "synthetic-witness (no live refusal)"
                                                : last_terminal_reject_reason(site);
        const fs::path stem = dump / ("cs_" + std::to_string(site) + "_" + std::to_string(index));
        ASSERT_TRUE(fixture::save_words(stem.string() + ".bin", p.code));
        ASSERT_TRUE(write_refused_compute_inputs(stem.string() + ".inputs", site, &p.table,
                                                 p.config, reason));
        const auto replay = replay_refused_shader_inputs(
            p.code, read_refused_shader_inputs(stem.string() + ".inputs"));
        ASSERT_EQ(replay.produced, x.admitted()) << replay.reject << " " << replay.incomplete;
        if (x.admitted()) {
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
    EXPECT_EQ(index, 74u);
    EXPECT_EQ(accepted, 68u);
    EXPECT_EQ(refused, 6u);
}
TEST(IrDsAddressExecution, OrdinaryAndPairedAccessesMatchIndependentPhysicalWordsInBothWaveSizes) {
    size_t index = 0;
    for (const auto& c : fixture::cases()) {
        SCOPED_TRACE(::testing::Message() << "case=" << index << " form=" << c.form
                                          << " literal=" << c.literal << " wave=" << c.wave);
        const auto p = fixture::program(c);
        const uint64_t site = 0x5387000000ull + index * 0x1000;
        const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                              {RecompileDiagnosticStage::Compute, site});
        ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(site);
        const auto d = fixture::data(c);
        std::vector<uint32_t> got;
        const auto dummy = prosper::test::run_compute(module, std::vector<float>(c.wave), c.wave,
                                                      c.wave, d.input, d.initial, &got, c.wave);
        ASSERT_EQ(dummy.size(), c.wave);
        ASSERT_EQ(got, d.expected);
        ++index;
    }
    EXPECT_EQ(index, 68u);
}
TEST(IrDsAddressExecution, OversizedSt64OffsetsAndGdsPairsStayRefused) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t form : {10u, 11u}) {
            auto p = fixture::program({form, wave, 0x0100});
            p.config.lds_bytes = 256;   // OFFSET1=1 means dword64, exactly beyond this allocation.
            EXPECT_TRUE(
                recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
        }
    for (uint32_t wave : {32u, 64u}) {
        auto p = fixture::program({9, wave, 0x0100});
        for (size_t pc = 0; pc < p.code.size();) {
            const auto in = rdna2_decode_one(p.code.data() + pc, p.code.size() - pc);
            ASSERT_GT(in.len_dwords, 0u);
            if (in.fmt == Rdna2Format::DS && in.opcode == 0x37) p.code[pc] |= 1u << 17;
            pc += in.len_dwords;
        }
        EXPECT_TRUE(recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
    }
}
TEST(IrDsAddressExecution, EqualOffsetsHaveDistinctSecondDataAndShareTheFirstPhysicalPair) {
    const fixture::Case c{12, 64, 0x0202};
    EXPECT_EQ(fixture::address(c, 3, 0), 208u);
    EXPECT_EQ(fixture::address(c, 3, 1), 212u);
    EXPECT_EQ(fixture::address(c, 3, 2), 208u);
    EXPECT_EQ(fixture::address(c, 3, 3), 212u);
    EXPECT_NE(fixture::word(3, 0, false), fixture::word(3, 2, false));
    EXPECT_NE(fixture::word(3, 1, false), fixture::word(3, 3, false));
    const fixture::Case scaled{10, 64, 0x0103};
    EXPECT_EQ(fixture::address(scaled, 4, 0), 4864u);
    EXPECT_EQ(fixture::address(scaled, 4, 1), 4352u);
}
}   // namespace
