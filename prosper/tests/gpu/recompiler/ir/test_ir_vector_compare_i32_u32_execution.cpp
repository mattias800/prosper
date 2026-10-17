// I32/U32 comparison values must keep the old EXEC-masked lane masks in every existing form.
// Truth rows are hand-worked two's-complement/unsigned ordering from AMD 70648, 12.9 Table 61.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_compare.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {
constexpr uint32_t kPairs[][2] = {
    {0x80000000u, 0x7fffffffu}, {0xffffffffu, 0u}, {0u, 0xffffffffu}, {0x80000000u, 0x80000000u},
    {0x7fffffffu, 0x80000000u}, {0u, 0u},          {1u, 2u},          {2u, 1u}};
struct TruthRow {
    uint32_t opcode;
    const char* truth;
};
constexpr TruthRow kRows[] = {
    {0x81, "11000010"}, {0x82, "00010100"}, {0x83, "11010110"}, {0x84, "00101001"},
    {0x85, "11101011"}, {0x86, "00111101"}, {0xc1, "00101010"}, {0xc2, "00010100"},
    {0xc3, "00111110"}, {0xc4, "11000001"}, {0xc5, "11101011"}, {0xc6, "11010101"},
};

LaneInputs integer_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        if (lane < 2 * std::size(kPairs)) {
            in.s0[lane] = kPairs[lane % std::size(kPairs)][0];
            in.s1[lane] = kPairs[lane % std::size(kPairs)][1];
        } else if (lane % 3 == 0) {
            in.s1[lane] = in.s0[lane];
        }
    }
    return in;
}

// Keep mutation failures bounded to their first differing lane without weakening the input set.
void check_integer_run(const VectorCompareDef& def, Form form, uint32_t wave, const LaneInputs& in,
                       const std::vector<uint32_t>& words) {
    const auto run = run_vector_op_per_lane(words, in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const Known test = eval_vector_compare(def, in.s0[lane], in.s1[lane]);
        ASSERT_TRUE(test.has_value());
        ASSERT_EQ(run.d[lane],
                  expected_mask_read(form, in.active[lane], *test != 0, in.old_d[lane]))
            << "op " << std::hex << def.opcode << " form " << int(form) << " wave " << wave
            << " lane " << lane << " s0 " << in.s0[lane] << " s1 " << in.s1[lane];
    }
}

std::vector<uint32_t> sdwa_program(uint32_t op, Form form, uint32_t select0, uint32_t select1) {
    const uint32_t opcode = is_cmpx(form) ? op + 0x10 : op;
    const bool scalar_destination = form == Form::Vop3Sgpr;
    const uint32_t control = kVsrc0 | (select0 << 16) | (select1 << 24) |
                             (scalar_destination ? (1u << 15) | (kSgprPair << 8) : 0u);
    std::vector<uint32_t> words{0x7c000000u | (opcode << 17) | (kVsrc1 << 9) | 0xf9u, control};
    if (is_cmpx(form)) {
        words.push_back(kMovV1One);
    } else {
        const uint32_t mask = scalar_destination ? kSgprPair : kVcc;
        words.insert(words.end(), {kSMovExecAll, 0xd5010001u, 0x80u | (0x81u << 9) | (mask << 18)});
    }
    return words;
}

constexpr std::array<std::pair<uint32_t, uint32_t>, 13> kSelects{{{0, 6},
                                                                  {1, 6},
                                                                  {2, 6},
                                                                  {3, 6},
                                                                  {4, 6},
                                                                  {5, 6},
                                                                  {6, 6},
                                                                  {6, 0},
                                                                  {6, 1},
                                                                  {6, 2},
                                                                  {6, 3},
                                                                  {6, 4},
                                                                  {6, 5}}};
constexpr Form kSdwaForms[] = {Form::VopcVcc, Form::Vop3Sgpr, Form::CmpxVopc};
uint32_t selected_bits(uint32_t bits, uint32_t select) {
    if (select < 4) return (bits >> (8 * select)) & 0xffu;
    if (select < 6) return (bits >> (16 * (select - 4))) & 0xffffu;
    return bits;
}

void check_sdwa_run(const VectorCompareDef& def, Form form, uint32_t wave, uint32_t select0,
                    uint32_t select1) {
    const auto in = integer_inputs(def.opcode + 0x5330);
    const auto words = sdwa_program(def.opcode, form, select0, select1);
    const auto decoded = rdna2_decode_one(words.data(), words.size());
    ASSERT_FALSE(decoded.has_modifier);
    ASSERT_TRUE(decoded.has_sdwa);
    ASSERT_EQ(decoded.sdwa_src0_sel, select0);
    ASSERT_EQ(decoded.sdwa_src1_sel, select1);
    const auto run = run_vector_op_per_lane(words, in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        // ISA SDWA zero extension is independent of the relation's evaluation.
        const auto truth = eval_vector_compare(def, selected_bits(in.s0[lane], select0),
                                               selected_bits(in.s1[lane], select1));
        ASSERT_TRUE(truth.has_value());
        ASSERT_EQ(run.d[lane],
                  expected_mask_read(form, in.active[lane], *truth != 0, in.old_d[lane]))
            << "op " << def.opcode << " sels " << select0 << ',' << select1 << " lane " << lane;
    }
}

size_t append_sdwa_witnesses() {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this test never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "vopc-i32-u32-witness";
    const fs::path dump = root / "refused_shaders_20000101-000000Z_synthetic";
    if (!fs::is_directory(dump)) return 0;
    size_t count = 0;
    for (const auto& def : vector_compare_i32_u32_defs())
        for (const auto [select0, select1] : kSelects)
            for (const Form form : kSdwaForms)
                for (const uint32_t wave : {64u, 32u}) {
                    const auto lp =
                        lane_program(sdwa_program(def.opcode, form, select0, select1), 64, wave);
                    const uint64_t address = 0x5330100000ull + count * 0x100;
                    const fs::path stem =
                        dump / ("cs_" + std::to_string(address) + "_sdwa_" + std::to_string(count));
                    std::ofstream bin(stem.string() + ".bin", std::ios::binary);
                    bin.write(reinterpret_cast<const char*>(lp.code.data()),
                              static_cast<std::streamsize>(lp.code.size() * sizeof(uint32_t)));
                    bin.close();
                    EXPECT_TRUE(bin.good());
                    const auto inputs = stem.string() + ".inputs";
                    EXPECT_TRUE(
                        write_refused_compute_inputs(inputs, address, &lp.table, lp.config,
                                                     "synthetic-witness (no live refusal)"));
                    const auto read = read_refused_shader_inputs(inputs);
                    const auto replay = replay_refused_shader_inputs(lp.code, read);
                    EXPECT_TRUE(replay.produced) << replay.reject << " op " << def.opcode;
                    ++count;
                }
    return count;
}
} // namespace

TEST(IrVectorCompareI32U32Execution, EveryDefinitionWritesTheExecMaskedLaneMask) {
    for (const auto& def : vector_compare_i32_u32_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode).name);
        for (const Form form : kForms)
            for (const uint32_t wave : {64u, 32u})
                for (uint32_t seed = 1; seed <= 3; ++seed)
                    check_integer_run(def, form, wave, integer_inputs(seed * 0x5330),
                                      compare_program(def.opcode, form));
    }
}

TEST(IrVectorCompareI32U32Execution, HandWorkedSignedAndUnsignedIsaRelations) {
    ASSERT_EQ(std::size(kRows), vector_compare_i32_u32_defs().size());
    for (const auto& row : kRows) {
        const auto* def = find_vector_compare(row.opcode);
        ASSERT_NE(def, nullptr);
        auto in = integer_inputs(row.opcode);
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            in.s0[lane] = kPairs[lane % std::size(kPairs)][0];
            in.s1[lane] = kPairs[lane % std::size(kPairs)][1];
        }
        for (size_t p = 0; p < std::size(kPairs); ++p)
            ASSERT_EQ(eval_vector_compare(*def, kPairs[p][0], kPairs[p][1]),
                      Known(row.truth[p] == '1'))
                << "op " << row.opcode << " pair " << p;
        for (const Form form : kForms)
            for (const uint32_t wave : {64u, 32u}) {
                const auto run =
                    run_vector_op_per_lane(compare_program(row.opcode, form), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    ASSERT_EQ(run.d[lane],
                              expected_mask_read(form, in.active[lane],
                                                 row.truth[lane % std::size(kPairs)] == '1',
                                                 in.old_d[lane]))
                        << "op " << row.opcode << " lane " << lane;
            }
    }
}

TEST(IrVectorCompareI32U32Execution, InactiveLanesReadZeroAndActiveLanesDiffer) {
    for (const uint32_t opcode : {0x81u, 0xc1u}) {
        auto in = integer_inputs(0x5330);
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            in.s0[lane] = kPairs[lane % std::size(kPairs)][0];
            in.s1[lane] = kPairs[lane % std::size(kPairs)][1];
            in.active[lane] = lane % 5 != 0;
        }
        const char* truth = opcode == 0x81 ? kRows[0].truth : kRows[6].truth;
        for (const uint32_t wave : {64u, 32u}) {
            const auto run =
                run_vector_op_per_lane(compare_program(opcode, Form::VopcVcc), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), in.lanes());
            std::set<uint32_t> active;
            size_t inactive_true = 0;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                if (in.active[lane])
                    active.insert(run.d[lane]);
                else {
                    inactive_true += truth[lane % std::size(kPairs)] == '1';
                    ASSERT_EQ(run.d[lane], 0u);
                }
            }
            EXPECT_EQ(active, (std::set<uint32_t>{0u, 1u}));
            EXPECT_GT(inactive_true, 0u) << "the control must observe EXEC independently of truth";
        }
    }
}

TEST(IrVectorCompareI32U32Execution, SdwaByteWordAndDwordSelectionsKeepTheSameRelations) {
    for (const auto& def : vector_compare_i32_u32_defs())
        for (const auto [select0, select1] : kSelects)
            for (const Form form : kSdwaForms)
                for (const uint32_t wave : {64u, 32u})
                    check_sdwa_run(def, form, wave, select0, select1);
}

TEST(IrVectorCompareI32U32Execution, EmptyExecWritesNoMaskBitsAndCmpxKeepsEveryOldDestination) {
    auto in = integer_inputs(0x5330);
    for (size_t lane = 0; lane < in.lanes(); ++lane) in.active[lane] = false;
    for (const auto& def : vector_compare_i32_u32_defs())
        for (const Form form : kForms)
            for (const uint32_t wave : {64u, 32u})
                check_integer_run(def, form, wave, in, compare_program(def.opcode, form));
}

TEST(IrVectorCompareI32U32Execution, SdwaSignExtensionStillRefusesBesideZeroExtendedSuccess) {
    for (const uint32_t opcode : {0x81u, 0xc1u}) {
        const auto* def = find_vector_compare(opcode);
        ASSERT_NE(def, nullptr);
        const auto good = sdwa_program(opcode, Form::VopcVcc, 4, 6);
        check_sdwa_run(*def, Form::VopcVcc, 64, 4, 6);
        auto bad = good;
        bad[1] |=
            1u << 19; // the existing decoder rejects unsupported SEXT, before relation lowering
        ASSERT_TRUE(rdna2_decode_one(bad.data(), bad.size()).has_modifier);
        const auto lp = lane_program(bad, 64, 64);
        EXPECT_TRUE(
            recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config).empty());
    }
}

TEST(IrVectorCompareI32U32Execution, DefinitionsAgreeWithEffectsAndUseAllIntegerRelations) {
    std::set<Opcode> seen;
    for (const auto& def : vector_compare_i32_u32_defs()) {
        const auto& cmp = rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode);
        const auto& cmpx = rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode + 0x10);
        ASSERT_NE(cmp.name, nullptr);
        ASSERT_NE(cmpx.name, nullptr);
        EXPECT_EQ(cmp.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(cmp.vdst_dwords, 0u);
        EXPECT_EQ(cmp.ssrc_dwords[0], 1u);
        EXPECT_EQ(cmp.ssrc_dwords[1], 1u);
        const std::string name = cmp.name;
        EXPECT_EQ(name.rfind("v_cmp_", 0), 0u);
        EXPECT_TRUE(name.ends_with("_i32") || name.ends_with("_u32"));
        EXPECT_EQ(std::string(cmpx.name), "v_cmpx_" + name.substr(6));
        EXPECT_TRUE(vopc_is_cmpx(def.opcode + 0x10));
        EXPECT_FALSE(vopc_is_cmpx(def.opcode));
        EXPECT_EQ(find_vector_compare(def.opcode + 0x10), nullptr);
        Block block;
        Builder ir(block);
        const auto result = def.build(ir, {ir.input(Type::U32), ir.input(Type::U32)});
        EXPECT_EQ(block.at(result).type, Type::Bool);
        for (const auto& instruction : block.insts()) seen.insert(instruction.op);
    }
    for (const auto op : {Opcode::SLt, Opcode::SLe, Opcode::SGt, Opcode::SGe, Opcode::IEq,
                          Opcode::INe, Opcode::ULt, Opcode::ULe, Opcode::UGt, Opcode::UGe})
        EXPECT_TRUE(seen.contains(op)) << opcode_info(op).name;
}

TEST(IrVectorCompareI32U32Execution, IntegerConstantComparesRemainUnsupported) {
    for (const uint32_t op : {0x80u, 0x87u, 0xc0u, 0xc7u}) {
        EXPECT_EQ(find_vector_compare(op), nullptr);
        for (const Form form : kForms) {
            const auto lp = lane_program(compare_program(op, form), 64, 64);
            const auto produced =
                recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config);
            EXPECT_TRUE(produced.empty()) << "op " << op << " form " << int(form);
        }
    }
}

TEST(IrVectorCompareI32U32Execution, ExportsEveryPlainAndSdwaFormForTheCorpusGate) {
    const size_t plain = export_compare_witness(vector_compare_i32_u32_defs(),
                                                "vopc-i32-u32-witness", 0x5330000000ull);
    ASSERT_EQ(plain, vector_compare_i32_u32_defs().size() * std::size(kForms) * 2u);
    const size_t sdwa = append_sdwa_witnesses();
    EXPECT_EQ(sdwa,
              vector_compare_i32_u32_defs().size() * kSelects.size() * std::size(kSdwaForms) * 2u);
}
