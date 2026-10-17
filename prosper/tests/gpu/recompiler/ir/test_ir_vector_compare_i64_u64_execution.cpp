// I64/U64 comparison values (64-bit words, HOW_TO_ADD_A_FAMILY.md "64-bit values") must keep the
// old EXEC-masked lane masks in every form. Sources are the VGPR pairs v[2:3] (S0) and v[5:6] (S1).
// Truth rows are hand-worked two's-complement/unsigned 64-bit order from AMD 70648, 12.9.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_compare.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <iterator>
#include <set>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {
constexpr uint64_t kMin = 0x8000000000000000ull, kMax = 0x7fffffffffffffffull;
// S0, S1 as 64-bit values. Rows 4, 9 and 10 differ only in the high word, row 6 only in the low
// word with the low sign bit set, and rows 0-3 and 9 order differently signed and unsigned.
constexpr uint64_t kPairs[][2] = {
    {kMin, kMax},
    {kMax, kMin},
    {~0ull, 0},
    {0, ~0ull},
    {0x0000000100000000ull, 0x00000000ffffffffull},
    {0x0000000500000007ull, 0x0000000500000007ull},
    {0x0000000580000000ull, 0x000000057fffffffull},
    {0x8000000000000001ull, kMin},
    {0xfffffffeffffffffull, 0xffffffff00000000ull},
    {kMin, 0},
    {0x0000000112345678ull, 0x0000000212345678ull},
};
struct TruthRow {
    uint32_t opcode;
    const char* truth;
};
constexpr TruthRow kRows[] = {
    {0xa1, "10100000111"}, {0xa2, "00000100000"}, {0xa3, "10100100111"}, {0xa4, "01011011000"},
    {0xa5, "11111011111"}, {0xa6, "01011111000"}, {0xe1, "01010000101"}, {0xe2, "00000100000"},
    {0xe3, "01010100101"}, {0xe4, "10101011010"}, {0xe5, "11111011111"}, {0xe6, "10101111010"},
};

void set_pair(LaneInputs& in, size_t lane, uint64_t s0, uint64_t s1) {
    in.s0[lane] = uint32_t(s0);
    in.s1[lane] = uint32_t(s0 >> 32);
    in.s2[lane] = uint32_t(s1);
    in.s3[lane] = uint32_t(s1 >> 32);
}

// lane_inputs' S0/S1 words become S0's low/high words; S1's pair is fresh, with the high words
// equal in a third of the lanes (the low words decide) and whole values equal in some.
LaneInputs pair_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    in.s2.resize(in.lanes());
    in.s3.resize(in.lanes());
    uint32_t state = seed ^ 0x6464u;
    const auto next = [&] { return state = state * 1664525u + 1013904223u; };
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s2[lane] = next();
        in.s3[lane] = next();
        if (lane < 2 * std::size(kPairs))
            set_pair(in, lane, kPairs[lane % std::size(kPairs)][0],
                     kPairs[lane % std::size(kPairs)][1]);
        else if (lane % 7 == 0)
            in.s2[lane] = in.s0[lane], in.s3[lane] = in.s1[lane];
        else if (lane % 3 == 0)
            in.s3[lane] = in.s1[lane];
        else if (lane % 5 == 0)
            in.s2[lane] = in.s0[lane];
    }
    return in;
}

Known lane_truth(const VectorCompareDef& def, const LaneInputs& in, size_t lane) {
    return eval_vector_compare_pairs(def, in.s0[lane], in.s1[lane], in.s2[lane], in.s3[lane]);
}

void check_run(const VectorCompareDef& def, Form form, uint32_t wave, const LaneInputs& in) {
    const auto run =
        run_vector_op_per_lane(compare_program(def.opcode, form, 0, 0, kVsrc2), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const Known test = lane_truth(def, in, lane);
        ASSERT_TRUE(test.has_value());
        ASSERT_EQ(run.d[lane],
                  expected_mask_read(form, in.active[lane], *test != 0, in.old_d[lane]))
            << "op " << std::hex << def.opcode << " form " << int(form) << " wave " << wave
            << " lane " << lane << " s0 " << in.s1[lane] << ':' << in.s0[lane] << " s1 "
            << in.s3[lane] << ':' << in.s2[lane];
    }
}

// SRC0 as a constant operand of a VOPC e32 compare against v[5:6]: the lowering's 64-bit halves
// sign-extend an inline integer and zero-extend a literal (integer64_halves).
struct ConstantSource {
    uint32_t operand;   // the SRC0 field
    uint32_t literal;   // follows the instruction when operand == 0xff
    uint32_t lo, hi;   // the 64-bit value the compare must see
};
constexpr ConstantSource kConstants[] = {
    {0xc1, 0, 0xffffffffu, 0xffffffffu},   // inline -1
    {0x85, 0, 5u, 0u},   // inline 5
    {0x80, 0, 0u, 0u},   // inline 0
    {0xff, 0x80000000u, 0x80000000u, 0u},   // a 32-bit literal: zero-extended
};
std::vector<uint32_t> constant_program(uint32_t opcode, const ConstantSource& source) {
    std::vector<uint32_t> words{0x7c000000u | (opcode << 17) | (kVsrc2 << 9) | source.operand};
    if (source.operand == 0xff) words.push_back(source.literal);
    words.insert(words.end(), {kSMovExecAll, 0xd5010001u, 0x80u | (0x81u << 9) | (kVcc << 18)});
    return words;
}
}   // namespace

TEST(IrVectorCompareI64U64Execution, EveryDefinitionWritesTheExecMaskedLaneMask) {
    for (const auto& def : vector_compare_i64_u64_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode).name);
        for (const Form form : kForms)
            for (const uint32_t wave : {64u, 32u})
                for (uint32_t seed = 1; seed <= 3; ++seed)
                    check_run(def, form, wave, pair_inputs(seed * 0x6464));
    }
}

TEST(IrVectorCompareI64U64Execution, HandWorkedSignedAndUnsignedIsaRelations) {
    ASSERT_EQ(std::size(kRows), vector_compare_i64_u64_defs().size());
    for (const auto& row : kRows) {
        const auto* def = find_vector_compare(row.opcode);
        ASSERT_NE(def, nullptr);
        auto in = pair_inputs(row.opcode);
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            set_pair(in, lane, kPairs[lane % std::size(kPairs)][0],
                     kPairs[lane % std::size(kPairs)][1]);
        for (size_t p = 0; p < std::size(kPairs); ++p)
            ASSERT_EQ(lane_truth(*def, in, p), Known(row.truth[p] == '1'))
                << "op " << std::hex << row.opcode << " pair " << p;
        for (const Form form : kForms)
            for (const uint32_t wave : {64u, 32u}) {
                const auto run = run_vector_op_per_lane(
                    compare_program(row.opcode, form, 0, 0, kVsrc2), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    ASSERT_EQ(run.d[lane],
                              expected_mask_read(form, in.active[lane],
                                                 row.truth[lane % std::size(kPairs)] == '1',
                                                 in.old_d[lane]))
                        << "op " << std::hex << row.opcode << " form " << int(form) << " lane "
                        << lane;
            }
    }
}

TEST(IrVectorCompareI64U64Execution, InactiveLanesReadZeroAndActiveLanesDiffer) {
    for (const size_t row : {size_t{0}, size_t{6}}) {   // lt_i64, lt_u64
        const uint32_t opcode = kRows[row].opcode;
        auto in = pair_inputs(0x6464);
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            set_pair(in, lane, kPairs[lane % std::size(kPairs)][0],
                     kPairs[lane % std::size(kPairs)][1]);
            in.active[lane] = lane % 5 != 0;
        }
        for (const uint32_t wave : {64u, 32u}) {
            const auto run = run_vector_op_per_lane(
                compare_program(opcode, Form::VopcVcc, 0, 0, kVsrc2), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), in.lanes());
            std::set<uint32_t> active;
            size_t inactive_true = 0;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                if (in.active[lane]) {
                    active.insert(run.d[lane]);
                } else {
                    inactive_true += kRows[row].truth[lane % std::size(kPairs)] == '1';
                    ASSERT_EQ(run.d[lane], 0u);
                }
            }
            EXPECT_EQ(active, (std::set<uint32_t>{0u, 1u}));
            EXPECT_GT(inactive_true, 0u) << "the control must observe EXEC independently of truth";
        }
    }
}

TEST(IrVectorCompareI64U64Execution, ConstantSourcesExtendToSixtyFourBits) {
    for (const auto& def : vector_compare_i64_u64_defs())
        for (const auto& source : kConstants) {
            const auto in = pair_inputs(def.opcode ^ source.operand);
            const auto run = run_vector_op_per_lane(constant_program(def.opcode, source), in, 64);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), in.lanes());
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                const Known test =
                    eval_vector_compare_pairs(def, source.lo, source.hi, in.s2[lane], in.s3[lane]);
                ASSERT_TRUE(test.has_value());
                ASSERT_EQ(run.d[lane], in.active[lane] && *test ? 1u : 0u)
                    << "op " << std::hex << def.opcode << " src0 " << source.operand << " lane "
                    << lane;
            }
        }
}

TEST(IrVectorCompareI64U64Execution, RefusalsStay) {
    // An inline FLOAT constant is a 64-bit double in a 64-bit compare, not an integer pair.
    const ConstantSource one_f{0xf2, 0, 0, 0};
    // ABS/NEG bits on an integer compare's e64 encoding.
    const auto abs_words = compare_program(0xa1, Form::Vop3Vcc, 1, 0, kVsrc2);
    const auto neg_words = compare_program(0xe1, Form::Vop3Vcc, 0, 1, kVsrc2);
    // V_CMP_F/T_I64/U64 have no definition.
    for (const auto& words :
         {constant_program(0xa1, one_f), constant_program(0xe4, one_f), abs_words, neg_words,
          compare_program(0xa0, Form::VopcVcc, 0, 0, kVsrc2),
          compare_program(0xe7, Form::Vop3Sgpr, 0, 0, kVsrc2)}) {
        const auto lp = lane_program(words, 64, 64, true, true);
        EXPECT_TRUE(recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config).empty())
            << std::hex << words[0];
    }
    for (const uint32_t op : {0xa0u, 0xa7u, 0xe0u, 0xe7u})
        EXPECT_EQ(find_vector_compare(op), nullptr);
}

TEST(IrVectorCompareI64U64Execution, DefinitionsAgreeWithEffectsAndUseEverySixtyFourBitRelation) {
    std::set<Opcode> seen;
    for (const auto& def : vector_compare_i64_u64_defs()) {
        const auto& cmp = rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode);
        const auto& cmpx = rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode + 0x10);
        ASSERT_NE(cmp.name, nullptr);
        ASSERT_NE(cmpx.name, nullptr);
        EXPECT_TRUE(def.pair_sources);
        EXPECT_EQ(cmp.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(cmp.ssrc_dwords[0], 2u);
        EXPECT_EQ(cmp.ssrc_dwords[1], 2u);
        const std::string name = cmp.name;
        EXPECT_EQ(name.rfind("v_cmp_", 0), 0u);
        EXPECT_TRUE(name.ends_with("_i64") || name.ends_with("_u64"));
        EXPECT_EQ(std::string(cmpx.name), "v_cmpx_" + name.substr(6));
        EXPECT_TRUE(vopc_is_cmpx(def.opcode + 0x10));
        EXPECT_EQ(find_vector_compare(def.opcode + 0x10), nullptr);
        Block block;
        Builder ir(block);
        VectorCompareIn in{ir.input(Type::U32), ir.input(Type::U32)};
        in.s0_hi = ir.input(Type::U32);
        in.s1_hi = ir.input(Type::U32);
        const auto result = def.build(ir, in);
        EXPECT_EQ(block.at(result).type, Type::Bool);
        for (const auto& instruction : block.insts()) seen.insert(instruction.op);
    }
    for (const auto op : {Opcode::PackU64, Opcode::Xor, Opcode::U64Eq, Opcode::U64Ne, Opcode::U64Lt,
                          Opcode::U64Le, Opcode::U64Gt, Opcode::U64Ge})
        EXPECT_TRUE(seen.contains(op)) << opcode_info(op).name;
}

TEST(IrVectorCompareI64U64Execution, ExportsEveryFormAndConstantSourceForTheCorpusGate) {
    const char* name = "vopc-i64-u64-witness";
    const size_t plain =
        export_compare_witness(vector_compare_i64_u64_defs(), name, 0x6464000000ull);
    ASSERT_EQ(plain, vector_compare_i64_u64_defs().size() * std::size(kForms) * 2u);
    size_t count = 0;
    for (const auto& def : vector_compare_i64_u64_defs())
        for (const auto& source : kConstants) {
            const auto lp = lane_program(constant_program(def.opcode, source), 64, 64, true, true);
            write_compare_witness_program(compare_witness_dump(name),
                                          0x6464100000ull + count * 0x100, count, lp);
            ++count;
        }
    EXPECT_EQ(count, vector_compare_i64_u64_defs().size() * std::size(kConstants));
}
