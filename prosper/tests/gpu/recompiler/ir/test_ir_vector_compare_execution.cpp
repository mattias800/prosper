// VOPC F32 compares (ir/ops/vector_compare_f32.cpp) run through the real recompiler and the SPIR-V
// interpreter, per lane, in every encoding and both wave sizes, and checked at the MASK they write
// (vector_compare_harness.hpp: inactive lanes must read 0; a V_CMPX narrows EXEC). Expected values
// come from the definition's evaluation; the hand-worked table (NaN, +-0, +-inf) comes from
// AMD 70648 12.9, Table 61, not from eval.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_compare.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

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

constexpr uint32_t kQNan = 0x7fc00000u, kSNan = 0x7f800001u, kNegNan = 0xffc00000u;
constexpr uint32_t kInf = 0x7f800000u, kNegInf = 0xff800000u, kNegZero = 0x80000000u;

LaneInputs compare_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    const uint32_t edges[] = {0u,     kNegZero, 0x3f800000u, 0xbf800000u, kInf,       kNegInf,
                              kQNan,  kSNan,    kNegNan,     0x00000001u, 0x7f7fffffu, 0x40000000u};
    const size_t e = std::size(edges);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        if (lane < 4 * e) {   // the first lanes pair the edges with each other
            in.s0[lane] = edges[lane % e];
            in.s1[lane] = edges[(lane / e + lane) % e];
        } else if (lane % 3 == 0) {   // equal words, so EQ/LE/GE also see true outside the edges
            in.s1[lane] = in.s0[lane];
        }
    }
    return in;
}

}   // namespace

TEST(IrVectorCompareExecution, EveryF32CompareWritesTheExecMaskedLaneMask) {
    for (const auto& def : vector_compare_f32_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode).name);
        for (Form form : kForms)
            for (uint32_t wave : {64u, 32u})
                for (uint32_t seed = 1; seed <= 3; ++seed)
                    check_compare_run(def, form, wave, compare_inputs(seed * 0x5325u));
    }
}

TEST(IrVectorCompareExecution, InactiveLanesReadZeroAndActiveLanesDiffer) {
    // The positive control for the harness itself: v_cmp_lt_f32 sees per-lane operands (both
    // answers occur among the active lanes) and EXEC (every inactive lane reads 0, not its test).
    const auto in = compare_inputs(11);
    const auto run = run_vector_op_per_lane(compare_program(0x01, Form::VopcVcc), in, 64);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> active;
    size_t inactive_true = 0, inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const bool lt = *eval_vector_compare(*find_vector_compare(0x01), in.s0[lane], in.s1[lane]);
        if (in.active[lane]) {
            active.insert(run.d[lane]);
        } else {
            ++inactive;
            inactive_true += lt;
            EXPECT_EQ(run.d[lane], 0u) << "lane " << lane;
        }
    }
    EXPECT_EQ(active, (std::set<uint32_t>{0u, 1u}));
    EXPECT_GE(inactive, 2u);
    EXPECT_GE(inactive_true, 1u) << "no inactive lane whose own test was true: EXEC unobserved";
}

TEST(IrVectorCompareExecution, HandWorkedIsaRelationsWithNanSignedZeroAndInfinity) {
    // AMD 70648 12.9 Table 61. Columns: (1,2) (2,1) (1,1) (+0,-0) (NaN,1) (1,NaN) (+inf,+inf)
    // (-inf,+inf) (NaN,NaN). The ordered six are false on a NaN; each N-form is the negation of
    // its ordered relation, so it is true on a NaN; +0 == -0; O/U test for a NaN operand.
    const uint32_t pairs[][2] = {{0x3f800000u, 0x40000000u}, {0x40000000u, 0x3f800000u},
                                 {0x3f800000u, 0x3f800000u}, {0u, kNegZero},
                                 {kQNan, 0x3f800000u},       {0x3f800000u, kQNan},
                                 {kInf, kInf},               {kNegInf, kInf},
                                 {kQNan, kQNan}};
    struct Row {
        uint32_t op;
        const char* truth;
    };
    const Row rows[] = {
        {0x00, "000000000"},   // F
        {0x01, "100000010"},   // LT
        {0x02, "001100100"},   // EQ
        {0x03, "101100110"},   // LE
        {0x04, "010000000"},   // GT
        {0x05, "110000010"},   // LG: ordered not-equal
        {0x06, "011100100"},   // GE
        {0x07, "111100110"},   // O
        {0x08, "000011001"},   // U
        {0x09, "100011011"},   // NGE = !GE
        {0x0a, "001111101"},   // NLG = !LG
        {0x0b, "101111111"},   // NGT = !GT
        {0x0c, "010011001"},   // NLE = !LE
        {0x0d, "110011011"},   // NEQ = !EQ: unordered not-equal, unlike LG
        {0x0e, "011111101"},   // NLT = !LT
        {0x0f, "111111111"},   // TRU
    };
    ASSERT_EQ(std::size(rows), vector_compare_f32_defs().size());
    for (const Row& row : rows) {
        const VectorCompareDef* def = find_vector_compare(row.op);
        ASSERT_NE(def, nullptr) << row.op;
        LaneInputs in = lane_inputs(row.op + 1);
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            in.s0[lane] = pairs[lane % std::size(pairs)][0];
            in.s1[lane] = pairs[lane % std::size(pairs)][1];
        }
        for (size_t p = 0; p < std::size(pairs); ++p)
            EXPECT_EQ(eval_vector_compare(*def, pairs[p][0], pairs[p][1]),
                      Known(row.truth[p] == '1'))
                << "op " << row.op << " pair " << p;
        for (Form form : {Form::VopcVcc, Form::CmpxVopc}) {
            const auto run = run_vector_op_per_lane(compare_program(row.op, form), in, 64);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(run.d[lane], expected_mask_read(form, in.active[lane],
                                                row.truth[lane % std::size(pairs)] == '1',
                                                in.old_d[lane]))
                    << "op " << row.op << " form " << int(form) << " lane " << lane;
        }
    }
}

TEST(IrVectorCompareExecution, Vop3SourceModifiersApplyBeforeTheRelation) {
    // v_cmp_lt_f32_e64 vcc, |v2|, -v3: |S0| < -S1. Worked by hand: (-3, -4) -> 3 < 4 true;
    // (-3, 2) -> 3 < -2 false; (1, -1) -> 1 < 1 false; (-0.5, -1) -> 0.5 < 1 true.
    const uint32_t cases[][3] = {{0xc0400000u, 0xc0800000u, 1}, {0xc0400000u, 0x40000000u, 0},
                                 {0x3f800000u, 0xbf800000u, 0}, {0xbf000000u, 0xbf800000u, 1}};
    LaneInputs in = lane_inputs(77);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = cases[lane % 4][0];
        in.s1[lane] = cases[lane % 4][1];
    }
    for (uint32_t wave : {64u, 32u}) {
        const auto run = run_vector_op_per_lane(compare_program(0x01, Form::Vop3Vcc, 1, 2), in, wave);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane],
                      expected_mask_read(Form::Vop3Vcc, in.active[lane], cases[lane % 4][2] != 0, 0))
                << "wave " << wave << " lane " << lane;
    }
}

TEST(IrVectorCompareExecution, DefinitionsAgreeWithEffectsAndUseEveryF32Relation) {
    std::set<Opcode> seen;
    for (const auto& def : vector_compare_f32_defs()) {
        const auto& cmp = rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode);
        const auto& cmpx = rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode + 0x10u);
        ASSERT_EQ(cmp.lane_flow, Rdna2LaneFlow::LaneWise) << def.opcode;
        EXPECT_EQ(cmp.vdst_dwords, 0u) << "a compare writes a lane mask, not a VGPR";
        ASSERT_NE(cmp.name, nullptr);
        ASSERT_NE(cmpx.name, nullptr);
        const std::string name = cmp.name, x_name = cmpx.name;
        EXPECT_EQ(name.rfind("v_cmp_", 0), 0u) << name;
        EXPECT_EQ(x_name, "v_cmpx_" + name.substr(6)) << "the V_CMPX twin sits at +0x10";
        EXPECT_EQ(name.substr(name.size() - 4), "_f32") << name;
        EXPECT_TRUE(vopc_is_cmpx(def.opcode + 0x10u));
        EXPECT_FALSE(vopc_is_cmpx(def.opcode));
        Block block;
        Builder ir(block);
        const Value result = def.build(ir, {ir.input(Type::U32), ir.input(Type::U32)});
        EXPECT_EQ(block.at(result).type, Type::Bool) << cmp.name;
        for (const auto& inst : block.insts()) seen.insert(inst.op);
    }
    for (Opcode op : {Opcode::F32Lt, Opcode::F32Eq, Opcode::F32Ge, Opcode::F32UnordNe,
                      Opcode::BoolAnd, Opcode::BoolOr, Opcode::ConstTrue, Opcode::ConstFalse})
        EXPECT_TRUE(seen.contains(op)) << opcode_info(op).name;
    for (uint32_t op = uint32_t(Opcode::F32Le); op <= uint32_t(Opcode::F32UnordGe); ++op)
        EXPECT_TRUE(seen.contains(Opcode(op))) << opcode_info(Opcode(op)).name;
    EXPECT_EQ(find_vector_compare(0x10), nullptr) << "lookup takes the V_CMP_* opcode";
    EXPECT_EQ(find_vector_compare(0x81), nullptr) << "v_cmp_lt_i32 has no definition yet";
}

// The recorded corpus never runs v_cmp_f_f32 / v_cmp_tru_f32 (their positive control reports
// `changes: none`), so export every definition in every form and wave as a witness corpus.
TEST(IrVectorCompareExecution, ExportsEveryFormForTheCorpusGate) {
    EXPECT_EQ(export_compare_witness(vector_compare_f32_defs(), "vopc-f32-witness", 0x5325000000ull),
              vector_compare_f32_defs().size() * std::size(kForms) * 2u);
}
