// The carry ops (ir/ops/vector_carry.cpp) per lane, through the real recompiler and the integer
// SPIR-V interpreter: D, and the carry mask they write, in every encoding and destination the
// lowering admits, in Wave64 and Wave32. The truth table is hand-worked from AMD 70648 6.5.
#include "gpu/recompiler/ir/ops/vector_carry.hpp"
#include "gpu/recompiler/ir/vector_mask_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {

// Where the carry-in comes from and where the carry-out goes.
struct CarryForm {
    const char* name;
    uint32_t cin, cout;
};
constexpr CarryForm kVop2Form{"vop2 vcc->vcc", kVcc, kVcc};
constexpr CarryForm kCarryInForms[] = {{"vop3b vcc->vcc", kVcc, kVcc},
                                       {"vop3b sgpr->sgpr", kCarryIn, kCarryOut},
                                       {"vop3b vcc->sgpr", kVcc, kCarryOut},
                                       {"vop3b sgpr->vcc", kCarryIn, kVcc}};
constexpr CarryForm kCarryOutForms[] = {{"vop3b ->vcc", 0, kVcc}, {"vop3b ->sgpr", 0, kCarryOut}};

std::vector<CarryForm> forms_of(const VectorCarryDef& def) {
    if (def.fmt == Rdna2Format::VOP2) return {kVop2Form};
    if (def.reads_carry_in) return {std::begin(kCarryInForms), std::end(kCarryInForms)};
    return {std::begin(kCarryOutForms), std::end(kCarryOutForms)};
}

std::vector<uint32_t> carry_program(const VectorCarryDef& def, const CarryForm& form,
                                    bool read_carry) {
    const std::vector<uint32_t> op =
        def.fmt == Rdna2Format::VOP2
            ? std::vector<uint32_t>{vop2(def.opcode)}
            : vop3_words(def.opcode, form.cout, def.reads_carry_in ? form.cin : 0);
    return concat({def.reads_carry_in ? carry_from_s2(form.cin) : std::vector<uint32_t>{}, op,
                   read_carry ? read_mask(form.cout) : std::vector<uint32_t>{}});
}

// In Wave32 the VOP3B carry-in forms are refused before they lower: the fresh-mask proof
// (fresh_mask_sources_defined) charges SRC2 the Wave64 register pair, and the second word is never
// written. That is the pre-IR verdict, kept; Wave32RefusalsStay pins it.
bool admitted(const VectorCarryDef& def, uint32_t wave) {
    return wave == 64 || def.fmt == Rdna2Format::VOP2 || !def.reads_carry_in;
}

KnownCarry lane_eval(const VectorCarryDef& def, const LaneInputs& in, size_t lane) {
    return eval_vector_carry(def, in.s0[lane], in.s1[lane], in.s2[lane] != 0 ? 1u : 0u);
}

void check_carry_run(const VectorCarryDef& def, const CarryForm& form, uint32_t wave,
                     const LaneInputs& in) {
    for (const bool read_carry : {false, true}) {
        const auto run = run_vector_op_per_lane(carry_program(def, form, read_carry), in, wave);
        ASSERT_TRUE(run.error.empty()) << form.name << " wave " << wave << ": " << run.error;
        ASSERT_EQ(run.d.size(), in.lanes());
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            const KnownCarry out = lane_eval(def, in, lane);
            ASSERT_TRUE(out.d && out.carry);
            const uint32_t expected = read_carry ? (in.active[lane] ? *out.carry : 0u)
                                                 : (in.active[lane] ? *out.d : in.old_d[lane]);
            ASSERT_EQ(run.d[lane], expected)
                << "op " << std::hex << def.opcode << ' ' << form.name << " wave " << std::dec
                << wave << (read_carry ? " carry" : " d") << " lane " << lane << std::hex << " s0 "
                << in.s0[lane] << " s1 " << in.s1[lane] << " s2 " << in.s2[lane] << " active "
                << in.active[lane];
        }
    }
}

const char* effects_name(const VectorCarryDef& def) {
    return rdna2_instruction_effects(def.fmt, def.opcode).name;
}

// AMD 70648 6.5, worked by hand: {s0, s1, cin} -> {d, carry/borrow}.
struct Row {
    uint32_t s0, s1, cin, d, carry;
};
struct Table {
    const char* name;
    std::vector<Row> rows;
};
const Table kTruth[] = {
    {"v_add_co_u32",
     {{0xffffffffu, 1, 0, 0, 1},
      {1, 2, 0, 3, 0},
      {0x80000000u, 0x80000000u, 0, 0, 1},
      {0x7fffffffu, 1, 0, 0x80000000u, 0},
      {5, 0, 0, 5, 0},
      {0, 0, 0, 0, 0}}},
    {"v_sub_co_u32",
     {{0, 1, 0, 0xffffffffu, 1},
      {5, 3, 0, 2, 0},
      {3, 3, 0, 0, 0},
      {0x80000000u, 1, 0, 0x7fffffffu, 0}}},
    {"v_subrev_co_u32",
     {{1, 0, 0, 0xffffffffu, 1},
      {3, 5, 0, 2, 0},
      {3, 3, 0, 0, 0},
      {1, 0x80000000u, 0, 0x7fffffffu, 0}}},
    {"v_add_co_ci_u32",
     {{0xffffffffu, 0, 1, 0, 1},
      {0xffffffffu, 0, 0, 0xffffffffu, 0},
      {0xffffffffu, 0xffffffffu, 1, 0xffffffffu, 1},
      {1, 2, 1, 4, 0},
      {0x80000000u, 0x80000000u, 0, 0, 1},
      {5, 0, 0, 5, 0}}},
    {"v_sub_co_ci_u32",
     {{0, 0, 1, 0xffffffffu, 1},
      {5, 3, 1, 1, 0},
      {3, 3, 0, 0, 0},
      {0, 0xffffffffu, 1, 0, 1},
      {3, 3, 1, 0xffffffffu, 1}}},
    {"v_subrev_co_ci_u32",
     {{0, 0, 1, 0xffffffffu, 1},
      {3, 5, 1, 1, 0},
      {3, 3, 0, 0, 0},
      {0xffffffffu, 0, 1, 0, 1},
      {3, 3, 1, 0xffffffffu, 1}}},
};

}   // namespace

TEST(IrVectorCarryExecution, EveryDefinitionEveryFormBothWaves) {
    for (const auto& def : vector_carry_defs()) {
        SCOPED_TRACE(effects_name(def));
        for (const CarryForm& form : forms_of(def))
            for (const uint32_t wave : {64u, 32u})
                for (uint32_t seed = 1; seed <= 2 && admitted(def, wave); ++seed)
                    check_carry_run(def, form, wave, mask_lane_inputs(seed * 0x5c41u + def.opcode));
    }
}

TEST(IrVectorCarryExecution, HandWorkedIsaTruthTable) {
    size_t checked = 0;
    for (const auto& def : vector_carry_defs()) {
        const Table* table = nullptr;
        for (const auto& t : kTruth)
            if (std::strcmp(t.name, effects_name(def)) == 0) table = &t;
        ASSERT_NE(table, nullptr) << effects_name(def);
        for (const Row& row : table->rows) {
            const KnownCarry out = eval_vector_carry(def, row.s0, row.s1, row.cin);
            EXPECT_EQ(out.d, Known(row.d))
                << table->name << ' ' << std::hex << row.s0 << ' ' << row.s1;
            EXPECT_EQ(out.carry, Known(row.carry)) << table->name << ' ' << std::hex << row.s0;
            ++checked;
        }
        // The same rows per lane, through the recompiler: every lane active.
        LaneInputs in = mask_lane_inputs(def.opcode);
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            const Row& row = table->rows[lane % table->rows.size()];
            in.s0[lane] = row.s0;
            in.s1[lane] = row.s1;
            in.s2[lane] = row.cin ? 0x10u : 0u;
            in.active[lane] = true;
        }
        for (const uint32_t wave : {64u, 32u})
            for (const bool read_carry : {false, true}) {
                if (!admitted(def, wave)) continue;
                const auto run = run_vector_op_per_lane(
                    carry_program(def, forms_of(def).front(), read_carry), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const Row& row = table->rows[lane % table->rows.size()];
                    ASSERT_EQ(run.d[lane], read_carry ? row.carry : row.d)
                        << table->name << " wave " << wave << " lane " << lane;
                }
            }
    }
    EXPECT_EQ(checked, 46u);   // the _co_ci tables run for VOP2 and VOP3B
}

// The harness can only show EXEC masking if an inactive lane has a raw carry of 1 and the active
// lanes see both carry values; and only the carry-in read if the active lanes see both S2 values.
TEST(IrVectorCarryExecution, HarnessSeesExecAndBothCarries) {
    for (const auto& def : vector_carry_defs()) {
        const LaneInputs in = mask_lane_inputs(1 * 0x5c41u + def.opcode);
        std::set<uint32_t> active_carries, active_cins;
        bool inactive_carry = false;
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            // An inactive lane's carry-in bit was written 0 by the masked v_cmp.
            const uint32_t cin = in.active[lane] && in.s2[lane] != 0 ? 1u : 0u;
            const uint32_t carry =
                eval_vector_carry(def, in.s0[lane], in.s1[lane], cin).carry.value_or(2);
            if (in.active[lane]) {
                active_carries.insert(carry);
                active_cins.insert(in.s2[lane] != 0);
            } else {
                inactive_carry |= carry == 1;
            }
        }
        EXPECT_TRUE(inactive_carry) << effects_name(def);
        EXPECT_EQ(active_carries, (std::set<uint32_t>{0, 1})) << effects_name(def);
        EXPECT_EQ(active_cins, (std::set<uint32_t>{0, 1})) << effects_name(def);
    }
}

TEST(IrVectorCarryExecution, Wave32RefusalsStay) {
    size_t refused = 0;
    for (const auto& def : vector_carry_defs()) {
        if (admitted(def, 32)) continue;
        for (const CarryForm& form : forms_of(def)) {
            const auto run =
                run_vector_op_per_lane(carry_program(def, form, false), mask_lane_inputs(7), 32);
            EXPECT_NE(run.error.find("fabricated-fresh-mask-source"), std::string::npos)
                << effects_name(def) << ' ' << form.name << ": " << run.error;
            ++refused;
        }
    }
    EXPECT_EQ(refused, 12u);
}

TEST(IrVectorCarryExecution, DefinitionsAgreeWithTheEffectsTable) {
    ASSERT_EQ(vector_carry_defs().size(), 9u);
    for (const auto& def : vector_carry_defs()) {
        const char* name = effects_name(def);
        ASSERT_NE(name, nullptr) << std::hex << def.opcode;
        EXPECT_EQ(def.reads_carry_in, std::strstr(name, "_co_ci_") != nullptr) << name;
        // The VOP2 rows and their VOP3B twins at +0x100 are the same operation.
        if (def.fmt == Rdna2Format::VOP2) {
            const VectorCarryDef* twin = find_vector_carry(Rdna2Format::VOP3, def.opcode + 0x100);
            ASSERT_NE(twin, nullptr) << name;
            EXPECT_EQ(twin->build, def.build) << name;
            EXPECT_STREQ(effects_name(*twin), name);
        }
    }
}

// #5348: an e32 carry op replaces VCC, and in a Wave32 B32-mask stage VCC_LO is also a B32 mask
// lifetime (sreg_bool[106]) that the e64 readers naming vcc_lo prefer. The previous VCC here is the
// carry_from_s2 mask (S2 != 0), so a reader that still sees it disagrees with the new carry on the
// lanes the positive control counts.
namespace {

// v_cndmask_b32_e64 v1, 0, 1, vcc_lo after s_mov_b64 exec, -1: VCC read through the e64 operand,
// the form whose route prefers the B32 lifetime (read_mask(kVcc) uses the e32 form instead).
std::vector<uint32_t> read_vcc_e64() {
    return {kSMovExecAll, 0xd5010001u, 0x80u | (0x81u << 9) | (kVcc << 18)};
}
// s_mov_b32 vcc_hi, 0: defines the high word, which the fresh-mask proof charges a Wave32 VOP3B
// carry-in from vcc_lo (Wave32RefusalsStay); without it the carry-in program below is refused.
constexpr uint32_t kSMovVccHiZero = 0xbeeb0380u;

std::vector<const VectorCarryDef*> vop2_carry_defs() {
    std::vector<const VectorCarryDef*> out;
    for (const auto& def : vector_carry_defs())
        if (def.fmt == Rdna2Format::VOP2) out.push_back(&def);
    return out;
}

// The new carry of the e32 op for one lane (the carry-in is the masked S2 != 0 mask).
uint32_t vop2_carry(const VectorCarryDef& def, const LaneInputs& in, size_t lane) {
    const uint32_t cin = in.active[lane] && in.s2[lane] != 0 ? 1u : 0u;
    return in.active[lane] ? *eval_vector_carry(def, in.s0[lane], in.s1[lane], cin).carry : 0u;
}

// Lanes where a stale VCC read would differ from the new one: the case the tests can see.
size_t stale_visible_lanes(const VectorCarryDef& def, const LaneInputs& in) {
    size_t lanes = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const uint32_t previous = in.active[lane] && in.s2[lane] != 0 ? 1u : 0u;
        lanes += vop2_carry(def, in, lane) != previous;
    }
    return lanes;
}

// The three #5348 programs. The e32 carry op replaces the VCC that carry_from_s2 wrote, and then:
// v_cndmask_b32_e64 reads it back through vcc_lo; or its VOP3B twin takes its carry-in from vcc_lo
// (carry-out to s[20:21], D over the e32 op's D); or v_mad_u64_u32 / v_mad_i64_i32 replaces VCC
// through `sdst = vcc_lo` (64-bit result to v[7:8], so the sources v2, v3, v[5:6] stay intact).
std::vector<uint32_t> e64_reader_program(const VectorCarryDef& def) {
    return concat({carry_from_s2(kVcc), {vop2(def.opcode)}, read_vcc_e64()});
}
std::vector<uint32_t> carry_in_program(const VectorCarryDef& twin) {
    return concat({{kSMovVccHiZero},
                   carry_from_s2(kVcc),
                   {vop2(twin.opcode - 0x100)},
                   vop3_words(twin.opcode, kCarryOut, kVcc)});
}
std::vector<uint32_t> mad_program(uint32_t opcode) {
    return concat({carry_from_s2(kVcc),
                   {0xd4000000u | (opcode << 16) | (kVcc << 8) | 7u,
                    (256u + kVsrc0) | ((256u + kVsrc1) << 9) | ((256u + kVsrc2) << 18)},
                   read_vcc_e64()});
}

}   // namespace

TEST(IrVectorCarryExecution, E64CndmaskReadsTheVop2CarryNotTheStaleVccLo) {
    ASSERT_EQ(vop2_carry_defs().size(), 3u);
    for (const VectorCarryDef* def : vop2_carry_defs()) {
        SCOPED_TRACE(effects_name(*def));
        const auto program = e64_reader_program(*def);
        for (uint32_t seed = 1; seed <= 2; ++seed) {
            const LaneInputs in = mask_lane_inputs(seed * 0x5348u + def->opcode);
            ASSERT_GT(stale_visible_lanes(*def, in), 0u);   // positive control
            for (const uint32_t wave : {64u, 32u}) {   // Wave64 has no B32 lifetime: the control
                const auto run = run_vector_op_per_lane(program, in, wave);
                ASSERT_TRUE(run.error.empty()) << "wave " << wave << ": " << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    ASSERT_EQ(run.d[lane], vop2_carry(*def, in, lane))
                        << "wave " << wave << " lane " << lane << " active " << in.active[lane]
                        << std::hex << " s0 " << in.s0[lane] << " s1 " << in.s1[lane] << " s2 "
                        << in.s2[lane];
            }
        }
    }
}

TEST(IrVectorCarryExecution, Vop3bCarryInFromVccLoReadsTheVop2Carry) {
    for (const VectorCarryDef* def : vop2_carry_defs()) {
        SCOPED_TRACE(effects_name(*def));
        // The VOP3B twin of the same op takes its carry-in from vcc_lo and writes D over the e32
        // op's D; its carry-out goes to s[20:21] so VCC is not written again.
        const VectorCarryDef* twin = find_vector_carry(Rdna2Format::VOP3, def->opcode + 0x100);
        ASSERT_NE(twin, nullptr);
        const auto program = carry_in_program(*twin);
        for (uint32_t seed = 1; seed <= 2; ++seed) {
            const LaneInputs in = mask_lane_inputs(seed * 0x5348u + def->opcode);
            // Positive control: the carry-in seen through a stale VCC changes the active D.
            size_t visible = 0;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                if (!in.active[lane]) continue;
                const uint32_t stale = in.s2[lane] != 0 ? 1u : 0u;
                visible +=
                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): concrete carry inputs give a known D.
                    *eval_vector_carry(*twin, in.s0[lane], in.s1[lane], stale).d !=
                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): concrete carry inputs give a known D.
                    *eval_vector_carry(*twin, in.s0[lane], in.s1[lane], vop2_carry(*def, in, lane))
                         .d;
            }
            ASSERT_GT(visible, 0u);
            for (const uint32_t wave : {64u, 32u}) {
                const auto run = run_vector_op_per_lane(program, in, wave);
                ASSERT_TRUE(run.error.empty()) << "wave " << wave << ": " << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const uint32_t expected =
                        in.active[lane] ? *eval_vector_carry(*twin, in.s0[lane], in.s1[lane],
                                                             vop2_carry(*def, in, lane))
                                               .d
                                        : in.old_d[lane];
                    ASSERT_EQ(run.d[lane], expected)
                        << "wave " << wave << " lane " << lane << " active " << in.active[lane];
                }
            }
        }
    }
}

// v_mad_u64_u32 / v_mad_i64_i32 write their carry to SDST the same way (#5349 noted the missing
// Wave32 word write): with `sdst = vcc_lo`, the e64 reader must see the mad's carry.
TEST(IrVectorCarryExecution, Wave32MadCarryToVccLoReplacesTheLifetime) {
    for (const uint32_t opcode : {0x176u, 0x177u}) {
        SCOPED_TRACE(opcode == 0x176u ? "v_mad_u64_u32" : "v_mad_i64_i32");
        const auto program = mad_program(opcode);
        for (uint32_t seed = 1; seed <= 2; ++seed) {
            LaneInputs in = mask_lane_inputs(seed * 0x5349u + opcode);
            uint32_t state = seed * 0x2545f491u;
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                in.s3.push_back(state = state * 1664525u + 1013904223u);
            // AMD 70648: {carry, D} = S0 * S1 + S2 as a 65-bit sum, unsigned or signed.
            auto carry = [&](size_t lane) -> uint32_t {
                if (!in.active[lane]) return 0u;
                const uint64_t add = (uint64_t{in.s3[lane]} << 32) | in.s2[lane];
                if (opcode == 0x176u) {
                    const uint64_t product = uint64_t{in.s0[lane]} * in.s1[lane];
                    return product + add < product ? 1u : 0u;
                }
                const __int128 sum = __int128{int64_t{static_cast<int32_t>(in.s0[lane])} *
                                              static_cast<int32_t>(in.s1[lane])} +
                                     static_cast<int64_t>(add);
                // Bit 64 of the sign-extended 65-bit sum, i.e. its sign (ops/vector_mad64.cpp).
                return sum < 0 ? 1u : 0u;
            };
            size_t visible = 0;   // positive control: lanes where the stale mask differs
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                visible += carry(lane) != (in.active[lane] && in.s2[lane] != 0 ? 1u : 0u);
            ASSERT_GT(visible, 0u);
            for (const uint32_t wave : {64u, 32u}) {
                const auto run = run_vector_op_per_lane(program, in, wave);
                ASSERT_TRUE(run.error.empty()) << "wave " << wave << ": " << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    ASSERT_EQ(run.d[lane], carry(lane))
                        << "wave " << wave << " lane " << lane << " active " << in.active[lane]
                        << std::hex << " s0 " << in.s0[lane] << " s1 " << in.s1[lane] << " s2 "
                        << in.s2[lane] << " s3 " << in.s3[lane];
            }
        }
    }
}

// Synthetic witness corpus for the corpus gate (HOW_TO_ADD_A_FAMILY.md): every definition in every
// form, reading D and reading the carry mask, in both wave sizes.
TEST(IrVectorCarryExecution, ExportsReplayableWitnessPrograms) {
    std::vector<MaskWitness> programs;
    for (const auto& def : vector_carry_defs())
        for (const CarryForm& form : forms_of(def))
            for (const bool read_carry : {false, true}) {
                MaskWitness program{carry_program(def, form, read_carry)};
                if (!admitted(def, 32)) program.waves = {64u};
                programs.push_back(program);
            }
    EXPECT_EQ(export_mask_witness(programs, "vector-carry-witness", 0x5c41000000ull), 60u);
}

// The #5348 programs as a witness corpus of their own (Wave64 is the unchanged control): the
// recorded corpus has no program that reaches them.
TEST(IrVectorCarryExecution, ExportsVccLoLifetimeWitnessPrograms) {
    std::vector<MaskWitness> programs;
    for (const VectorCarryDef* def : vop2_carry_defs()) {
        programs.push_back({e64_reader_program(*def)});
        programs.push_back(
            {carry_in_program(*find_vector_carry(Rdna2Format::VOP3, def->opcode + 0x100))});
    }
    for (const uint32_t opcode : {0x176u, 0x177u}) {
        MaskWitness program{mad_program(opcode)};
        program.fourth_source = true;
        programs.push_back(program);
    }
    EXPECT_EQ(export_mask_witness(programs, "vcc-lo-lifetime-witness", 0x5348000000ull), 16u);
}
