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
