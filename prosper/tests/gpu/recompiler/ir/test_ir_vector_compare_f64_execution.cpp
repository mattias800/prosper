// F64 comparison values (an ADMISSION, HOW_TO_ADD_A_FAMILY.md "F64 values"): every V_CMP(X)_*_F64
// form writes the EXEC-masked lane mask of the IEEE binary64 relation. Sources are the VGPR pairs
// v[2:3] (S0) and v[5:6] (S1). Truth rows are hand-worked from AMD 70648 12.9 Table 61 and the
// binary64 encodings, under both FP_DENORM[7:6] input modes (6.4 Table 24): preserved subnormals,
// and flushed ones that compare as signed zeros.
#include "gpu/recompiler/float64_support.hpp"
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
#include <iterator>
#include <set>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {
// FLOAT_MODE[19:12] of COMPUTE_PGM_RSRC1: the default keeps F32 denormals and flushes F64 ones
// (FP_DENORM[7:6] = 0); kPreserve sets [7:6] = 3, so binary64 subnormals reach the relation.
constexpr uint32_t kFlush = kDefaultComputePgmRsrc1;
constexpr uint32_t kPreserve = kDefaultComputePgmRsrc1 | (0xc0u << 12);

// S0, S1 as binary64 words: signed zeros, NaNs (quiet and signalling), the infinities and the
// largest finite value, subnormals on both sides of zero and at the normal boundary, a pair that
// differs only in the low word, equal negatives, and a pair of opposite-signed subnormals.
constexpr uint64_t kPairs[][2] = {
    {0, 0x8000000000000000ull},
    {0x3ff0000000000000ull, 0x7ff8000000000000ull},
    {0x7ff8000000000000ull, 0x7ff8000000000000ull},
    {0x7ff0000000000001ull, 0x3ff0000000000000ull},
    {0xfff0000000000000ull, 0x7ff0000000000000ull},
    {0x7fefffffffffffffull, 0x7ff0000000000000ull},
    {1, 0},
    {0x8000000000000001ull, 0},
    {0x000fffffffffffffull, 0x0010000000000000ull},
    {0x3ff0000000000000ull, 0x3ff0000000000001ull},
    {0xbff0000000000000ull, 0xbff0000000000000ull},
    {0x4000000000000000ull, 0xbfe0000000000000ull},
    {1, 0x8000000000000001ull},
};
struct TruthRow {
    uint32_t opcode;
    const char* preserve;   // one character per kPairs row
    const char* flush;
};
constexpr TruthRow kRows[] = {
    {0x20, "0000000000000", "0000000000000"}, {0x21, "0000110111000", "0000110011000"},
    {0x22, "1000000000100", "1000001100101"}, {0x23, "1000110111100", "1000111111101"},
    {0x24, "0000001000011", "0000000000010"}, {0x25, "0000111111011", "0000110011010"},
    {0x26, "1000001000111", "1000001100111"}, {0x27, "1000111111111", "1000111111111"},
    {0x28, "0111000000000", "0111000000000"}, {0x29, "0111110111000", "0111110011000"},
    {0x2a, "1111000000100", "1111001100101"}, {0x2b, "1111110111100", "1111111111101"},
    {0x2c, "0111001000011", "0111000000010"}, {0x2d, "0111111111011", "0111110011010"},
    {0x2e, "1111001000111", "1111001100111"}, {0x2f, "1111111111111", "1111111111111"},
};

// The device gate is process-global: publish a capable device for each test, restore after.
class IrVectorCompareF64Execution : public ::testing::Test {
protected:
    void SetUp() override {
        reset_float64_support_for_test();
        publish_float64_support(true, true, true);
    }
    void TearDown() override { reset_float64_support_for_test(); }
};

void set_pair(LaneInputs& in, size_t lane, uint64_t s0, uint64_t s1) {
    in.s0[lane] = uint32_t(s0);
    in.s1[lane] = uint32_t(s0 >> 32);
    in.s2[lane] = uint32_t(s1);
    in.s3[lane] = uint32_t(s1 >> 32);
}

// Edge pairs first, then random words with high words shared in some lanes (the low words
// decide), whole values equal in some, and subnormal high words in some.
LaneInputs pair_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    in.s2.resize(in.lanes());
    in.s3.resize(in.lanes());
    uint32_t state = seed ^ 0xf64u;
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
            in.s1[lane] &= 0x800fffffu;   // a subnormal S0
    }
    return in;
}

// FP_DENORM input flush, written independently of the lowering: a zero exponent field leaves
// only the sign.
uint64_t flushed(uint32_t lo, uint32_t hi, bool flush) {
    if (flush && (hi & 0x7ff00000u) == 0) return uint64_t(hi & 0x80000000u) << 32;
    return uint64_t(hi) << 32 | lo;
}
Known lane_truth(const VectorCompareDef& def, const LaneInputs& in, size_t lane, bool flush) {
    const uint64_t a = flushed(in.s0[lane], in.s1[lane], flush);
    const uint64_t c = flushed(in.s2[lane], in.s3[lane], flush);
    return eval_vector_compare_pairs(def, uint32_t(a), uint32_t(a >> 32), uint32_t(c),
                                     uint32_t(c >> 32));
}

void check_run(const VectorCompareDef& def, Form form, uint32_t wave, const LaneInputs& in,
               uint32_t rsrc1) {
    const auto run = run_vector_op_per_lane(compare_program(def.opcode, form, 0, 0, kVsrc2), in,
                                            wave, false, rsrc1);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const Known test = lane_truth(def, in, lane, rsrc1 == kFlush);
        ASSERT_TRUE(test.has_value());
        ASSERT_EQ(run.d[lane],
                  expected_mask_read(form, in.active[lane], *test != 0, in.old_d[lane]))
            << "op " << std::hex << def.opcode << " form " << int(form) << " wave " << wave
            << " rsrc1 " << rsrc1 << " lane " << lane << " s0 " << in.s1[lane] << ':'
            << in.s0[lane] << " s1 " << in.s3[lane] << ':' << in.s2[lane];
    }
}

// SRC0 as a constant operand of a VOPC e32 compare against v[5:6].
struct ConstantSource {
    uint32_t operand;   // the SRC0 field
    uint32_t literal;   // follows the instruction when operand == 0xff
    uint64_t value;     // the binary64 value the compare must see
};
constexpr ConstantSource kConstants[] = {
    {0xf0, 0, 0x3fe0000000000000ull},            // inline 0.5 as a double
    {0xf7, 0, 0xc010000000000000ull},            // inline -4.0
    {0xf8, 0, 0x3fc45f306dc9c882ull},            // inline 1/(2*pi), Table 20's double
    {0x80, 0, 0},                                // inline integer 0: +0.0
    {0x7d, 0, 0},                                // SGPR_NULL: +0.0
    {0xff, 0x3ff80000u, 0x3ff8000000000000ull},  // literal: the HIGH word (6.2.1), so 1.5
};
std::vector<uint32_t> constant_program(uint32_t opcode, uint32_t operand, uint32_t literal) {
    std::vector<uint32_t> words{0x7c000000u | (opcode << 17) | (kVsrc2 << 9) | operand};
    if (operand == 0xff) words.push_back(literal);
    words.insert(words.end(), {kSMovExecAll, 0xd5010001u, 0x80u | (0x81u << 9) | (kVcc << 18)});
    return words;
}
uint64_t modified(uint64_t v, bool abs, bool neg) {
    if (abs) v &= ~0x8000000000000000ull;
    if (neg) v ^= 0x8000000000000000ull;
    return v;
}
} // namespace

TEST_F(IrVectorCompareF64Execution, EveryDefinitionWritesTheExecMaskedLaneMask) {
    for (const auto& def : vector_compare_f64_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode).name);
        for (const Form form : kForms)
            for (const uint32_t wave : {64u, 32u})
                for (const uint32_t rsrc1 : {kFlush, kPreserve})
                    for (uint32_t seed = 1; seed <= 2; ++seed)
                        check_run(def, form, wave, pair_inputs(seed * 0xf64), rsrc1);
    }
}

TEST_F(IrVectorCompareF64Execution, HandWorkedIeeeBinary64RelationsInBothDenormalModes) {
    ASSERT_EQ(std::size(kRows), vector_compare_f64_defs().size());
    for (const auto& row : kRows) {
        const auto* def = find_vector_compare(row.opcode);
        ASSERT_NE(def, nullptr);
        auto in = pair_inputs(row.opcode);
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            set_pair(in, lane, kPairs[lane % std::size(kPairs)][0],
                     kPairs[lane % std::size(kPairs)][1]);
        for (const uint32_t rsrc1 : {kFlush, kPreserve}) {
            const char* truth = rsrc1 == kFlush ? row.flush : row.preserve;
            for (size_t p = 0; p < std::size(kPairs); ++p)
                ASSERT_EQ(lane_truth(*def, in, p, rsrc1 == kFlush), Known(truth[p] == '1'))
                    << "op " << std::hex << row.opcode << " pair " << p;
            for (const Form form : kForms)
                for (const uint32_t wave : {64u, 32u}) {
                    const auto run = run_vector_op_per_lane(
                        compare_program(row.opcode, form, 0, 0, kVsrc2), in, wave, false, rsrc1);
                    ASSERT_TRUE(run.error.empty()) << run.error;
                    ASSERT_EQ(run.d.size(), in.lanes());
                    for (size_t lane = 0; lane < in.lanes(); ++lane)
                        ASSERT_EQ(run.d[lane],
                                  expected_mask_read(form, in.active[lane],
                                                     truth[lane % std::size(kPairs)] == '1',
                                                     in.old_d[lane]))
                            << "op " << std::hex << row.opcode << " form " << int(form)
                            << " rsrc1 " << rsrc1 << " lane " << std::dec << lane;
                }
        }
    }
}

TEST_F(IrVectorCompareF64Execution, InactiveLanesReadZeroAndActiveLanesDiffer) {
    for (const uint32_t opcode : {0x21u, 0x29u}) {   // lt_f64 and its unordered negation nge
        auto in = pair_inputs(0xf64);
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            set_pair(in, lane, kPairs[lane % std::size(kPairs)][0],
                     kPairs[lane % std::size(kPairs)][1]);
            in.active[lane] = lane % 5 != 0;
        }
        const char* truth = kRows[opcode - 0x20].preserve;
        for (const uint32_t wave : {64u, 32u}) {
            const auto run = run_vector_op_per_lane(
                compare_program(opcode, Form::VopcVcc, 0, 0, kVsrc2), in, wave, false, kPreserve);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), in.lanes());
            std::set<uint32_t> active;
            size_t inactive_true = 0;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                if (in.active[lane]) {
                    active.insert(run.d[lane]);
                } else {
                    inactive_true += truth[lane % std::size(kPairs)] == '1';
                    ASSERT_EQ(run.d[lane], 0u);
                }
            }
            EXPECT_EQ(active, (std::set<uint32_t>{0u, 1u}));
            EXPECT_GT(inactive_true, 0u) << "the control must observe EXEC independently of truth";
        }
    }
}

TEST_F(IrVectorCompareF64Execution, ConstantSourcesAreBinary64Values) {
    for (const auto& def : vector_compare_f64_defs())
        for (const auto& source : kConstants) {
            auto in = pair_inputs(def.opcode ^ source.operand);
            // Put S0's own value in some lanes of S1, so EQ and the orders all see both answers.
            for (size_t lane = 0; lane < in.lanes(); lane += 4) {
                in.s2[lane] = uint32_t(source.value);
                in.s3[lane] = uint32_t(source.value >> 32);
            }
            const auto run = run_vector_op_per_lane(
                constant_program(def.opcode, source.operand, source.literal), in, 64, false,
                kPreserve);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), in.lanes());
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                const Known test =
                    eval_vector_compare_pairs(def, uint32_t(source.value),
                                              uint32_t(source.value >> 32), in.s2[lane],
                                              in.s3[lane]);
                ASSERT_TRUE(test.has_value());
                ASSERT_EQ(run.d[lane], in.active[lane] && *test ? 1u : 0u)
                    << "op " << std::hex << def.opcode << " src0 " << source.operand << " lane "
                    << std::dec << lane;
            }
        }
}

// VOP3 ABS/NEG act on the sign bit of the high word: |S0| < -S1 and friends.
TEST_F(IrVectorCompareF64Execution, AbsAndNegModifySignBits) {
    for (const uint32_t opcode : {0x21u, 0x22u, 0x26u, 0x2du})
        for (const uint32_t abs : {0u, 1u, 2u, 3u})
            for (const uint32_t neg : {0u, 1u, 2u, 3u}) {
                const auto* def = find_vector_compare(opcode);
                ASSERT_NE(def, nullptr);
                const auto in = pair_inputs(opcode * 16 + abs * 4 + neg);
                const auto run = run_vector_op_per_lane(
                    compare_program(opcode, Form::Vop3Vcc, abs, neg, kVsrc2), in, 64, false,
                    kPreserve);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const uint64_t a = modified(uint64_t(in.s1[lane]) << 32 | in.s0[lane],
                                                abs & 1u, neg & 1u);
                    const uint64_t c = modified(uint64_t(in.s3[lane]) << 32 | in.s2[lane],
                                                abs & 2u, neg & 2u);
                    const Known test = eval_vector_compare_pairs(
                        *def, uint32_t(a), uint32_t(a >> 32), uint32_t(c), uint32_t(c >> 32));
                    ASSERT_EQ(run.d[lane], in.active[lane] && *test ? 1u : 0u)
                        << "op " << std::hex << opcode << " abs " << abs << " neg " << neg
                        << " lane " << std::dec << lane;
                }
            }
}

// Every case the lowering cannot settle refuses by name, and the device gate is fail-visible.
TEST_F(IrVectorCompareF64Execution, RefusalsAreNamed) {
    const auto in = pair_inputs(0xf64);
    const auto reason = [&](const std::vector<uint32_t>& words, uint32_t rsrc1) {
        return run_vector_op_per_lane(words, in, 64, false, rsrc1).error;
    };
    const auto lt = compare_program(0x21, Form::VopcVcc, 0, 0, kVsrc2);
    // A nonzero inline integer in a binary64 operand: bit pattern or value, the ISA does not say.
    EXPECT_NE(reason(constant_program(0x21, 0xc1, 0), kPreserve).find(
                  "f64-inline-integer-unspecified"),
              std::string::npos);
    EXPECT_NE(reason(constant_program(0x21, 0x81, 0), kPreserve).find(
                  "f64-inline-integer-unspecified"),
              std::string::npos);
    // No launch FLOAT_MODE: the denormal mode is unknown.
    EXPECT_NE(reason(lt, UINT32_MAX).find("f64-float-mode-unavailable"), std::string::npos);
    // No device has published, then a device without the feature has.
    reset_float64_support_for_test();
    EXPECT_NE(reason(lt, kPreserve).find("f64-device-unavailable"), std::string::npos);
    publish_float64_support(true, true, true);
    publish_float64_support(true, false, true);   // ANDed: the weaker device decides
    EXPECT_NE(reason(lt, kPreserve).find("f64-device-unavailable"), std::string::npos);
    // F and TRU read no value: they need neither the device feature nor the mode.
    for (const uint32_t opcode : {0x20u, 0x2fu}) {
        const auto run = run_vector_op_per_lane(compare_program(opcode, Form::VopcVcc, 0, 0, kVsrc2),
                                                in, 64, false, UINT32_MAX);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane], in.active[lane] && opcode == 0x2fu ? 1u : 0u);
    }
}

TEST_F(IrVectorCompareF64Execution, DefinitionsAgreeWithEffectsAndUseEveryBinary64Relation) {
    std::set<Opcode> seen;
    for (const auto& def : vector_compare_f64_defs()) {
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
        EXPECT_TRUE(name.ends_with("_f64")) << name;
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
    for (uint32_t op = uint32_t(Opcode::F64Lt); op <= uint32_t(Opcode::F64UnordGe); ++op)
        EXPECT_TRUE(seen.contains(Opcode(op))) << opcode_info(Opcode(op)).name;
}

// The witness corpus: every definition in every form and wave under the default (flushing)
// FLOAT_MODE, again under a preserving one, and the constant SRC0 operands.
TEST_F(IrVectorCompareF64Execution, ExportsEveryFormModeAndConstantSourceForTheCorpusGate) {
    const char* name = "vopc-f64-witness";
    const auto defs = vector_compare_f64_defs();
    const size_t plain = export_compare_witness(defs, name, 0xf64000000ull);
    ASSERT_EQ(plain, defs.size() * std::size(kForms) * 2u);
    size_t count = 0;
    for (const auto& def : defs)
        for (const Form form : kForms)
            for (const uint32_t wave : {64u, 32u}) {
                auto lp = lane_program(compare_program(def.opcode, form, 0, 0, kVsrc2), 64, wave,
                                       true, true);
                lp.config.compute_pgm_rsrc1 = kPreserve;
                write_compare_witness_program(compare_witness_dump(name),
                                              0xf64100000ull + count * 0x100, plain + count, lp);
                ++count;
            }
    for (const auto& def : defs)
        for (const auto& source : kConstants) {
            auto lp = lane_program(constant_program(def.opcode, source.operand, source.literal), 64,
                                   64, true, true);
            lp.config.compute_pgm_rsrc1 = kPreserve;
            write_compare_witness_program(compare_witness_dump(name),
                                          0xf64100000ull + count * 0x100, plain + count, lp);
            ++count;
        }
    EXPECT_EQ(count, defs.size() * (std::size(kForms) * 2u + std::size(kConstants)));
}
