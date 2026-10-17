// MAD64 results, all three outputs and source/destination ownership through real guest modules.
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <ios>
#include "gpu/recompiler/ir/mad64_execution_harness.hpp"
#include "gpu/recompiler/ir/ops/vector_mad64.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <system_error>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace m = prosper::test::ir::mad64;
namespace {
struct Row {
    m::Raw x;
    m::Answer answer;
};
const std::array<Row, 10> signed_rows{{
    {{0xffffffff, 1, 0, 0}, {0xffffffff, 0xffffffff, 1}},
    {{0x80000000, 0x80000000, 0, 0x80000000}, {0, 0xc0000000, 1}},
    {{0x7fffffff, 0x7fffffff, 0xffffffff, 0x7fffffff}, {0, 0xbfffffff, 0}},
    {{2, 3, 0xffffffff, 0xffffffff}, {5, 0, 0}},
    {{0xfffffffb, 7, 0x10, 0}, {0xffffffed, 0xffffffff, 1}},
    {{0x3039, 0xfffffffe, 0, 1}, {0xffff9f8e, 0, 0}},
    {{0x80000000, 0xffffffff, 0xffffffff, 0x7fffffff}, {0x7fffffff, 0x80000000, 0}},
    {{0x80000000, 0x7fffffff, 0, 0x80000000}, {0x80000000, 0x40000000, 1}},
    {{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}, {0, 0, 0}},
    {{0, 0, 0, 0}, {0, 0, 0}},
}};
void compare(m::Answer got, m::Answer want) {
    EXPECT_EQ(got.lo, want.lo);
    EXPECT_EQ(got.hi, want.hi);
    EXPECT_EQ(got.carry, want.carry);
}
void check_module(const std::vector<uint32_t>& module, const LaneInputs& in, uint32_t opcode,
                  m::Form form, uint32_t output) {
    ASSERT_FALSE(module.empty());
    const auto run = m::execute(module, in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = m::reference(m::raw(in, lane, form), opcode == 0x177);
        const uint32_t word = output == 0 ? want.lo : output == 1 ? want.hi : want.carry;
        const uint32_t inactive =
            output == 2 ? 0 : m::old_word(in, lane, m::destination(form) + output);
        EXPECT_EQ(run.d[lane],
                  (form == m::Form::UncachedFullExec || in.active[lane]) ? word : inactive)
            << "op " << opcode << " form " << static_cast<uint32_t>(form) << " output " << output
            << " lane " << lane;
    }
}
std::vector<uint32_t> read_words(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        ADD_FAILURE() << path;
        return {};
    }
    const auto bytes = file.tellg();
    if (bytes <= 0 || static_cast<uint64_t>(bytes) % sizeof(uint32_t)) {
        ADD_FAILURE() << path;
        return {};
    }
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    if (!file) {
        ADD_FAILURE() << path;
        return {};
    }
    return words;
}
std::vector<uint32_t> compile(const LaneProgram& lp) {
    const auto address = lane_harness_address();
    const auto out = recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config,
                                       {RecompileDiagnosticStage::Compute, address});
    EXPECT_FALSE(out.empty()) << last_terminal_reject_reason(address);
    return out;
}
}   // namespace
TEST(IrVectorMad64Execution, HandWorkedSignedBit64IsDistinctFromSignedOverflow) {
    const auto* def = find_vector_mad64(Rdna2Format::VOP3, 0x177);
    ASSERT_NE(def, nullptr);
    for (const auto& row : signed_rows) {
        compare(m::reference(row.x, true), row.answer);
        const auto got = eval_vector_mad64(*def, row.x.a, row.x.b, row.x.lo, row.x.hi);
        if (!got.lo || !got.hi || !got.carry) {
            ADD_FAILURE() << "all three MAD64 results must be Known";
            return;
        }
        compare({*got.lo, *got.hi, *got.carry}, row.answer);
    }
}
TEST(IrVectorMad64Execution, HandWorkedUnsignedCarriesIncludeZeroEquality) {
    const auto* def = find_vector_mad64(Rdna2Format::VOP3, 0x176);
    ASSERT_NE(def, nullptr);
    const std::array<Row, 5> rows{{
        {{0xffffffff, 2, 0xffffffff, 0xffffffff}, {0xfffffffd, 1, 1}},
        {{0, 0, 0, 0}, {0, 0, 0}},
        {{0xffffffff, 0xffffffff, 0, 0}, {1, 0xfffffffe, 0}},
        {{1, 1, 0xffffffff, 0xffffffff}, {0, 0, 1}},
        {{1, 0xffffffff, 1, 0}, {0, 1, 0}},
    }};
    for (const auto& row : rows) {
        compare(m::reference(row.x, false), row.answer);
        const auto got = eval_vector_mad64(*def, row.x.a, row.x.b, row.x.lo, row.x.hi);
        if (!got.lo || !got.hi || !got.carry) {
            ADD_FAILURE() << "all three MAD64 results must be Known";
            return;
        }
        compare({*got.lo, *got.hi, *got.carry}, row.answer);
    }
}
TEST(IrVectorMad64Execution, HandWorkedSignedRowsExecuteInActiveAndInactiveLanes) {
    auto in = m::inputs(5354);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto& row = signed_rows[lane % signed_rows.size()];
        in.s0[lane] = row.x.a;
        in.s1[lane] = row.x.b;
        in.s2[lane] = row.x.lo;
        in.s3[lane] = row.x.hi;
        in.active[lane] = (lane / signed_rows.size()) % 2 != 0;
    }
    for (uint32_t sdst : {106u, 20u})
        for (uint32_t wave : {32u, 64u})
            for (uint32_t output : {0u, 1u, 2u}) {
                const auto run =
                    m::execute(compile(m::program(0x177, m::Form::Plain, sdst, wave, output)), in);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto& want = signed_rows[lane % signed_rows.size()].answer;
                    const uint32_t active = output == 0   ? want.lo
                                            : output == 1 ? want.hi
                                                          : want.carry;
                    const uint32_t inactive = output == 2 ? 0 : m::old_word(in, lane, 1 + output);
                    EXPECT_EQ(run.d[lane], in.active[lane] ? active : inactive) << "lane " << lane;
                }
            }
}
TEST(IrVectorMad64Execution, EvalMatchesIndependent65BitAdditionAndRequiredUnknownsStayIncomplete) {
    for (const auto& def : vector_mad64_defs()) {
        const auto in = m::inputs(5354);
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            const auto x = m::raw(in, lane, m::Form::Plain);
            const auto want = m::reference(x, def.opcode == 0x177);
            const auto got = eval_vector_mad64(def, x.a, x.b, x.lo, x.hi);
            if (!got.lo || !got.hi || !got.carry) {
                ADD_FAILURE() << "all three MAD64 results must be Known";
                return;
            }
            compare({*got.lo, *got.hi, *got.carry}, want);
        }
        for (uint32_t absent = 0; absent < 4; ++absent) {
            std::array<Known, 4> x = {1, 2, 3, 4};
            x[absent] = std::nullopt;
            const auto got = eval_vector_mad64(def, x[0], x[1], x[2], x[3]);
            EXPECT_FALSE(got.lo && got.hi && got.carry);
        }
    }
}
TEST(IrVectorMad64Execution, DefinitionsUseCentralSourceAndDestinationWidths) {
    ASSERT_EQ(vector_mad64_defs().size(), 2u);
    for (const auto& def : vector_mad64_defs()) {
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_TRUE(effects.describes_sources());
        EXPECT_TRUE(effects.describes_vector_result());
        EXPECT_EQ(effects.ssrc_dwords[0], 1u);
        EXPECT_EQ(effects.ssrc_dwords[1], 1u);
        EXPECT_EQ(effects.ssrc_dwords[2], 2u);
        EXPECT_EQ(effects.vdst_dwords, 2u);
        EXPECT_EQ(find_vector_mad64(def.fmt, def.opcode), &def);
    }
    EXPECT_EQ(find_vector_mad64(Rdna2Format::VOP3, 0x175), nullptr);
    EXPECT_EQ(find_vector_mad64(Rdna2Format::VOP2, 0x176), nullptr);
}
TEST(IrVectorMad64Execution, EveryFormObservesBothWordsAndCarryWithInactiveAndAliasCanaries) {
    for (const auto& def : vector_mad64_defs())
        for (const auto form : m::forms(def.opcode))
            for (uint32_t sdst : {106u, 20u})
                for (uint32_t wave : {32u, 64u})
                    for (uint32_t output : {0u, 1u, 2u})
                        for (uint32_t seed : {5354u, 17531u}) {
                            const auto lp = m::program(def.opcode, form, sdst, wave, output);
                            check_module(compile(lp), m::inputs(seed), def.opcode, form, output);
                        }
}
TEST(IrVectorMad64Execution, FullExecReplacesBothWordsWithoutDependingOnOldDestinations) {
    for (const auto& def : vector_mad64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t output : {0u, 1u, 2u}) {
                auto in = m::inputs(5381);
                std::fill(in.active.begin(), in.active.end(), true);
                auto lp = m::program(def.opcode, m::Form::UncachedFullExec, 20, wave, output);
                check_module(compile(lp), in, def.opcode, m::Form::UncachedFullExec, output);
            }
}
TEST(IrVectorMad64Execution, BothCarryValuesAndInactiveRawCarryOneExerciseTheHarness) {
    for (uint32_t opcode : {0x176u, 0x177u}) {
        auto in = m::inputs(17531);
        size_t active_zero = 0, active_one = 0, inactive_one = 0, changed_high = 0;
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            const auto out = m::reference(m::raw(in, lane, m::Form::Plain), opcode == 0x177);
            if (in.active[lane]) {
                active_zero += out.carry == 0;
                active_one += out.carry == 1;
            } else
                inactive_one += out.carry == 1;
            changed_high += out.hi != m::old_word(in, lane, 2);
        }
        ASSERT_GT(active_zero, 0u);
        ASSERT_GT(active_one, 0u);
        ASSERT_GT(inactive_one, 0u);
        ASSERT_GT(changed_high, 0u);
        for (uint32_t wave : {32u, 64u})
            check_module(compile(m::program(opcode, m::Form::Plain, 106, wave, 2)), in, opcode,
                         m::Form::Plain, 2);
    }
}
TEST(IrVectorMad64Execution, SignedLiteralAddendRefusesWhileUnsignedKeepsItsExistingZeroExtension) {
    for (uint32_t wave : {32u, 64u}) {
        const auto unsigned_program = m::program(0x176, m::Form::AddLiteral, 106, wave, 0);
        ASSERT_FALSE(compile(unsigned_program).empty());
        const auto lp = m::program(0x177, m::Form::AddLiteral, 106, wave, 0);
        const auto address = lane_harness_address();
        const auto out = recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config,
                                           {RecompileDiagnosticStage::Compute, address});
        EXPECT_TRUE(out.empty());
        EXPECT_NE(last_terminal_reject_reason(address).find("op=0x177"), std::string::npos);
    }
}
TEST(IrVectorMad64Execution, HighOnlyFabricatedScalarAddendCannotBecomeAFreshCarry) {
    for (uint32_t opcode : {0x176u, 0x177u})
        for (uint32_t wave : {32u, 64u}) {
            const auto good = m::program(opcode, m::Form::PairSgpr, 20, wave, 2);
            ASSERT_FALSE(compile(good).empty());
            auto bad = good; // omit only S_MOV_B32 s55, literal; low addend remains genuine
            bad.code.erase(bad.code.begin() + 16, bad.code.begin() + 18);
            const auto address = lane_harness_address();
            const auto out =
                recompile_compute(bad.code.data(), bad.code.size(), &bad.table, bad.config,
                                  {RecompileDiagnosticStage::Compute, address});
            EXPECT_TRUE(out.empty());
            const auto reason = last_terminal_reject_reason(address);
            EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
            EXPECT_NE(reason.find("pc=16 "), std::string::npos) << reason;
        }
}
TEST(IrVectorMad64Execution, ExportsEveryAcceptedAttributeAndExecutesTheSavedModules) {
    namespace fs = std::filesystem;
    const auto root = compare_witness_root("mad64-values-witness");
 // NOLINTNEXTLINE(concurrency-mt-unsafe): launch environment is immutable.
    if (!std::getenv("PROSPER_IR_WITNESS_DIR")) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory: " << root;
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream attributes(dump / "attributes.tsv");
    ASSERT_TRUE(attributes);
    attributes << "index\topcode\tform\tsdst\twave\toutput\n";
    size_t count = 0;
    for (const auto& def : vector_mad64_defs())
        for (const auto form : m::forms(def.opcode))
            for (uint32_t sdst : {106u, 20u})
                for (uint32_t wave : {32u, 64u})
                    for (uint32_t output : {0u, 1u, 2u}) {
                        const auto lp = m::program(def.opcode, form, sdst, wave, output);
                        const uint64_t address = 0x5354000000ull + count * 0x100;
                        write_compare_witness_program(dump, address, count, lp);
                        const auto stem =
                            dump / ("cs_" + std::to_string(address) + "_" + std::to_string(count));
                        const auto code = read_words(stem.string() + ".bin");
                        ASSERT_EQ(code, lp.code);
                        const auto inputs = read_refused_shader_inputs(stem.string() + ".inputs");
                        ASSERT_TRUE(inputs.incomplete.empty()) << inputs.incomplete;
                        const auto replay = replay_refused_shader_inputs(code, inputs);
                        ASSERT_TRUE(replay.produced) << replay.reject;
                        const auto saved_path = dump / ("module-" + std::to_string(count) + ".spv");
                        std::ofstream saved(saved_path, std::ios::binary);
                        ASSERT_TRUE(saved);
                        saved.write(
                            reinterpret_cast<const char*>(replay.spirv.data()),
                            static_cast<std::streamsize>(replay.spirv.size() * sizeof(uint32_t)));
                        saved.close();
                        ASSERT_TRUE(saved.good());
                        const auto loaded = read_words(saved_path);
                        ASSERT_EQ(loaded, replay.spirv);
                        check_module(loaded, m::inputs(5354), def.opcode, form, output);
                        attributes << count << '\t' << def.opcode << '\t'
                                   << static_cast<uint32_t>(form) << '\t' << sdst << '\t' << wave
                                   << '\t' << output << '\n';
                        ++count;
                    }
    ASSERT_TRUE(attributes);
    EXPECT_EQ(count, 396u);
}
