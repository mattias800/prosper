// v_cndmask_b32 (ir/ops/vector_cndmask.cpp) per lane, through the real recompiler and the integer
// SPIR-V interpreter: the selector mask from VCC (VOP2, VOP3) and from an SGPR pair (VOP3), with
// VOP3 float source modifiers and VOP2 SDWA word selects, in Wave64 and Wave32.
#include "gpu/recompiler/ir/ops/vector_cndmask.hpp"
#include "gpu/recompiler/ir/vector_mask_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {

const VectorCndmaskDef& vop2_def() {
    return *find_vector_cndmask(Rdna2Format::VOP2, 0x01);
}
const VectorCndmaskDef& vop3_def() {
    return *find_vector_cndmask(Rdna2Format::VOP3, 0x101);
}

// What a lane's S0 and S1 become before the select, and how D is assembled from the selection.
struct CndmaskForm {
    const char* name;
    const VectorCndmaskDef* (*def)();
    uint32_t mask;   // the register the selector mask is written to and read from
    std::vector<uint32_t> op;
    uint32_t (*source0)(uint32_t), (*source1)(uint32_t);
    uint32_t (*merge)(uint32_t selected, uint32_t old_d);
};

uint32_t same(uint32_t v) {
    return v;
}
uint32_t negate(uint32_t v) {
    return v ^ 0x80000000u;
}
uint32_t absolute(uint32_t v) {
    return v & 0x7fffffffu;
}
uint32_t low_word(uint32_t v) {
    return v & 0xffffu;
}
uint32_t high_word(uint32_t v) {
    return v >> 16;
}
uint32_t whole(uint32_t selected, uint32_t) {
    return selected;
}
uint32_t into_high(uint32_t selected, uint32_t old) {
    return (old & 0xffffu) | (selected << 16);
}
uint32_t into_low(uint32_t selected, uint32_t old) {
    return (old & 0xffff0000u) | selected;
}

const VectorCndmaskDef* vop2_ptr() {
    return &vop2_def();
}
const VectorCndmaskDef* vop3_ptr() {
    return &vop3_def();
}

std::vector<CndmaskForm> forms() {
    return {
        {"vop2 vcc", vop2_ptr, kVcc, {vop2(0x01)}, same, same, whole},
        {"vop3 vcc", vop3_ptr, kVcc, vop3_words(0x101, 0, kVcc), same, same, whole},
        {"vop3 sgpr", vop3_ptr, kCarryOut, vop3_words(0x101, 0, kCarryOut), same, same, whole},
        // v_cndmask_b32_e64 v1, -v2, |v3|, s20 (llvm-mc gfx1030: d5010201 20520702).
        {"vop3 sgpr -s0 |s1|", vop3_ptr, kCarryOut, vop3_words(0x101, 2, kCarryOut, 1), negate,
         absolute, whole},
        // v_cndmask_b32_sdwa v1, v2, v3, vcc dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
        // src0_sel:WORD_0 src1_sel:WORD_1 (llvm-mc gfx1030: 020206f9 05041502).
        {"vop2 sdwa word1",
         vop2_ptr,
         kVcc,
         {0x020206f9u, 0x05041502u},
         low_word,
         high_word,
         into_high},
        // ... dst_sel:WORD_0 src0_sel:WORD_1 src1_sel:WORD_0 (020206f9 04051402).
        {"vop2 sdwa word0",
         vop2_ptr,
         kVcc,
         {0x020206f9u, 0x04051402u},
         high_word,
         low_word,
         into_low},
    };
}

std::vector<uint32_t> cndmask_program(const CndmaskForm& form) {
    return concat({mask_from_s2(form.mask), form.op});
}

// Finite normal binary32 words, so the float modifiers act on the sign bit alone.
LaneInputs cndmask_inputs(uint32_t seed) {
    LaneInputs in = mask_lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = (in.s0[lane] & 0x807fffffu) | 0x3f800000u;
        in.s1[lane] = (in.s1[lane] & 0x807fffffu) | 0x40000000u;
    }
    return in;
}

uint32_t expected_lane(const CndmaskForm& form, const LaneInputs& in, size_t lane) {
    if (!in.active[lane]) return in.old_d[lane];
    const Known selected =
        eval_vector_cndmask(*form.def(), in.s2[lane] != 0 ? 1u : 0u, form.source0(in.s0[lane]),
                            form.source1(in.s1[lane]));
    EXPECT_TRUE(selected.has_value());
    return form.merge(selected.value_or(0), in.old_d[lane]);
}

}   // namespace

TEST(IrVectorCndmaskExecution, EveryFormBothWaves) {
    for (const CndmaskForm& form : forms())
        for (const uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 3; ++seed) {
                const LaneInputs in = cndmask_inputs(seed * 0xc4d1u);
                const auto run = run_vector_op_per_lane(cndmask_program(form), in, wave);
                ASSERT_TRUE(run.error.empty()) << form.name << ": " << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    ASSERT_EQ(run.d[lane], expected_lane(form, in, lane))
                        << form.name << " wave " << wave << " lane " << lane << std::hex << " s0 "
                        << in.s0[lane] << " s1 " << in.s1[lane] << " s2 " << in.s2[lane]
                        << " active " << in.active[lane];
            }
}

TEST(IrVectorCndmaskExecution, HandWorkedIsaSelect) {
    // AMD 70648 6.5: D = mask[lane] ? S1 : S0.
    for (const VectorCndmaskDef* def : {&vop2_def(), &vop3_def()}) {
        EXPECT_EQ(eval_vector_cndmask(*def, 1u, 0x11u, 0x22u), Known(0x22u));
        EXPECT_EQ(eval_vector_cndmask(*def, 0u, 0x11u, 0x22u), Known(0x11u));
        EXPECT_EQ(eval_vector_cndmask(*def, 1u, 0xffffffffu, 0u), Known(0u));
        EXPECT_EQ(eval_vector_cndmask(*def, std::nullopt, 1u, 1u), std::nullopt);
    }
}

// The run can only tell the selected operand from the other one if the active lanes see both mask
// values and S0 != S1 there, and EXEC only if an inactive lane's selection differs from its old D.
TEST(IrVectorCndmaskExecution, HarnessSeesBothMaskValuesAndExec) {
    const LaneInputs in = cndmask_inputs(0xc4d1u);
    std::set<bool> active_masks;
    bool distinct_sources = false;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane]) {
            active_masks.insert(in.s2[lane] != 0);
            distinct_sources |= in.s0[lane] != in.s1[lane];
        }
    EXPECT_EQ(active_masks, (std::set<bool>{false, true}));
    EXPECT_TRUE(distinct_sources);
    EXPECT_LT(std::count(in.active.begin(), in.active.end(), true), 64);
}

TEST(IrVectorCndmaskExecution, DefinitionsAgreeWithTheEffectsTable) {
    ASSERT_EQ(vector_cndmask_defs().size(), 2u);
    for (const auto& def : vector_cndmask_defs())
        EXPECT_STREQ(rdna2_instruction_effects(def.fmt, def.opcode).name, "v_cndmask_b32");
    EXPECT_EQ(vop2_def().build, vop3_def().build);
}

TEST(IrVectorCndmaskExecution, ExportsReplayableWitnessPrograms) {
    std::vector<MaskWitness> programs;
    for (const CndmaskForm& form : forms()) programs.push_back({cndmask_program(form)});
    EXPECT_EQ(export_mask_witness(programs, "vector-cndmask-witness", 0xc4d1000000ull), 12u);
}
