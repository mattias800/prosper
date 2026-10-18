// CLASS must observe the raw IEEE category, including both NaN kinds and signed zero, before
// the shared mask writer applies EXEC. AMD 70648, 12.9 Table 61 supplies these hand class bits.
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <set>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {
struct ClassRow {
    uint32_t raw, bit;
};
constexpr ClassRow kClasses[] = {
    {0x7f800001u, 1},   {0x7fc00000u, 2},   {0xff800000u, 4},   {0xbf800000u, 8},
    {0x80000001u, 16},  {0x80000000u, 32},  {0u, 64},           {1u, 128},
    {0x3f800000u, 256}, {0x7f800000u, 512}, {0xffbfffffu, 1},   {0xffffffffu, 2},
    {0xff7fffffu, 8},   {0x807fffffu, 16},  {0x007fffffu, 128}, {0x00800000u, 256},
    {0x80800000u, 8},   {0x7f7fffffu, 256},
};
constexpr Form kWideForms[] = {Form::Vop3Sgpr, Form::Vop3Vcc, Form::CmpxVop3};
constexpr Form kSdwaForms[] = {Form::VopcVcc, Form::Vop3Sgpr, Form::CmpxVopc};

LaneInputs class_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto& row = kClasses[lane % std::size(kClasses)];
        if (lane < 2 * std::size(kClasses)) in.s0[lane] = row.raw;
        in.s1[lane] = lane % 3 == 0 ? row.bit : lane % 3 == 1 ? 0x3ffu : 0x80000000u;
    }
    return in;
}

void check_class_run(const LaneInputs& in, Form form, uint32_t wave,
                     const std::vector<uint32_t>& words, uint32_t abs = 0, uint32_t neg = 0) {
    const auto* def = find_vector_compare(0x88);
    ASSERT_NE(def, nullptr);
    const auto run = run_vector_op_per_lane(words, in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        uint32_t raw = in.s0[lane];
        if (abs) raw &= 0x7fffffffu;
        if (neg) raw ^= 0x80000000u;
        const auto truth = eval_vector_compare(*def, raw, in.s1[lane]);
        ASSERT_TRUE(truth.has_value());
        ASSERT_EQ(run.d[lane],
                  expected_mask_read(form, in.active[lane], truth.value_or(0) != 0, in.old_d[lane]))
            << "form " << int(form) << " wave " << wave << " lane " << lane << " raw " << std::hex
            << raw << " mask " << in.s1[lane];
    }
}

std::vector<uint32_t> sdwa_program(Form form) {
    const bool scalar = form == Form::Vop3Sgpr;
    const uint32_t op = is_cmpx(form) ? 0x98u : 0x88u;
    auto words = compare_program(0x88, form);
    const uint32_t control =
        kVsrc0 | (6u << 16) | (6u << 24) | (scalar ? (1u << 15) | (kSgprPair << 8) : 0u);
    const auto tail = compare_program(0x88, is_cmpx(form) ? Form::CmpxVopc : Form::VopcVcc);
    words = {0x7c000000u | (op << 17) | (kVsrc1 << 9) | 0xf9u, control};
    words.insert(words.end(), tail.begin() + 1, tail.end());
    if (scalar) words.back() = 0x80u | (0x81u << 9) | (kSgprPair << 18);
    return words;
}

size_t append_wrapper_witnesses() {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this test never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "vopc-class-f32-witness";
    const fs::path dump = root / "refused_shaders_20000101-000000Z_synthetic";
    if (!fs::is_directory(dump)) return 0;
    size_t count = 0;
    const auto save = [&](const std::vector<uint32_t>& words, uint32_t wave) {
        const auto lp = lane_program(words, 64, wave);
        const uint64_t address = 0x5338100000ull + count * 0x100;
        const fs::path stem =
            dump / ("cs_" + std::to_string(address) + "_" + std::to_string(count));
        std::ofstream bin(stem.string() + ".bin", std::ios::binary);
        bin.write(reinterpret_cast<const char*>(lp.code.data()),
                  static_cast<std::streamsize>(lp.code.size() * sizeof(uint32_t)));
        bin.close();
        EXPECT_TRUE(bin.good());
        const auto inputs = stem.string() + ".inputs";
        EXPECT_TRUE(write_refused_compute_inputs(inputs, address, &lp.table, lp.config,
                                                 "synthetic-witness (no live refusal)"));
        const auto read = read_refused_shader_inputs(inputs);
        EXPECT_TRUE(read.incomplete.empty()) << read.incomplete;
        const auto replay = replay_refused_shader_inputs(lp.code, read);
        EXPECT_TRUE(replay.produced) << replay.reject;
        ++count;
    };
    for (const Form form : kWideForms)
        for (const uint32_t modifiers : {1u, 2u, 3u})
            for (const uint32_t wave : {64u, 32u})
                save(compare_program(0x88, form, modifiers & 1u, modifiers >> 1u), wave);
    for (const Form form : kSdwaForms)
        for (const uint32_t wave : {64u, 32u}) save(sdwa_program(form), wave);
    return count;
}
} // namespace

TEST(IrVectorCompareClassF32Execution, EveryFormWritesTheExecMaskedClassResult) {
    ASSERT_EQ(vector_compare_class_f32_defs().size(), 1u);
    for (const Form form : kForms)
        for (const uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 4; ++seed)
                check_class_run(class_inputs(seed * 0x5338), form, wave,
                                compare_program(0x88, form));
}

TEST(IrVectorCompareClassF32Execution, HandWorkedIsaClassBitsIncludeNanKindsSignsAndBoundaries) {
    const auto* def = find_vector_compare(0x88);
    ASSERT_NE(def, nullptr);
    for (const auto& row : kClasses) {
        for (uint32_t bit = 0; bit < 10; ++bit)
            ASSERT_EQ(eval_vector_compare(*def, row.raw, 1u << bit), Known(row.bit == (1u << bit)));
        EXPECT_EQ(eval_vector_compare(*def, row.raw, 0x3ffu), Known(1));
        EXPECT_EQ(eval_vector_compare(*def, row.raw, 0xfffffc00u), Known(0));
        EXPECT_EQ(eval_vector_compare(*def, row.raw, row.bit | 0xfffffc00u), Known(1));
    }
    auto in = class_inputs(0x5338);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto& row = kClasses[lane % std::size(kClasses)];
        in.s0[lane] = row.raw;
        in.s1[lane] = lane / std::size(kClasses) % 2 ? row.bit : 0x80000000u;
    }
    for (const Form form : kForms)
        for (const uint32_t wave : {64u, 32u}) {
            const auto run = run_vector_op_per_lane(compare_program(0x88, form), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), in.lanes());
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                ASSERT_EQ(run.d[lane],
                          expected_mask_read(form, in.active[lane],
                                             lane / std::size(kClasses) % 2 != 0, in.old_d[lane]));
        }
}

TEST(IrVectorCompareClassF32Execution, InactiveLanesReadZeroAndActiveLanesDiffer) {
    auto in = class_inputs(0x5338);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = 0xbf800000u;
        in.s1[lane] = lane % 3 == 0 ? 4u : 8u;
        in.active[lane] = lane % 5 != 0;
    }
    for (const uint32_t wave : {64u, 32u}) {
        const auto run = run_vector_op_per_lane(compare_program(0x88, Form::VopcVcc), in, wave);
        ASSERT_TRUE(run.error.empty()) << run.error;
        ASSERT_EQ(run.d.size(), in.lanes());
        std::set<uint32_t> active;
        size_t inactive_true = 0;
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            if (in.active[lane])
                active.insert(run.d[lane]);
            else {
                inactive_true += in.s1[lane] == 8u;
                ASSERT_EQ(run.d[lane], 0u);
            }
        }
        EXPECT_EQ(active, (std::set<uint32_t>{0u, 1u}));
        EXPECT_GT(inactive_true, 0u);
    }
}

TEST(IrVectorCompareClassF32Execution, RawAbsNegPreparationKeepsFiniteClassesAndMaskBits) {
    const auto in = class_inputs(0x5338);
    auto finite = in;
    for (size_t lane = 0; lane < finite.lanes(); ++lane) {
        const auto& row = kClasses[2 + lane % 8];   // infinities, finite signs, zeros and denormals
        finite.s0[lane] = row.raw;
        finite.s1[lane] = lane % 2 ? row.bit : (1023u ^ row.bit);
    }
    for (const Form form : kWideForms)
        for (const uint32_t wave : {64u, 32u})
            for (const uint32_t modifiers : {1u, 2u, 3u})
                check_class_run(finite, form, wave,
                                compare_program(0x88, form, modifiers & 1u, modifiers >> 1u),
                                modifiers & 1u, modifiers >> 1u);
}

TEST(IrVectorCompareClassF32Execution, NegatedNanPayloadsKeepTheirRawClass) {
    auto in = class_inputs(0x5338);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = lane % 2 ? 0xff800001u : 0x7fc00000u;
        in.s1[lane] = lane % 3 ? (lane % 2 ? 1u : 2u) : 0u;
    }
    for (const Form form : kWideForms)
        for (const uint32_t wave : {64u, 32u})
            check_class_run(in, form, wave, compare_program(0x88, form, 0, 1), 0, 1);
}

TEST(IrVectorCompareClassF32Execution, UnusedGenericAbsRetainsTheExistingNanOracleBoundary) {
    auto in = class_inputs(0x5338);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = 0xff800001u;
        in.s1[lane] = 1u;
    }
    check_class_run(in, Form::Vop3Vcc, 64, compare_program(0x88, Form::Vop3Vcc));
    // The old generic modifier emission is deliberately kept even though CLASS uses raw bits.
    // The eager VM cannot define FAbs(NaN); this is an oracle limit, not a driver result.
    const auto run = run_vector_op_per_lane(compare_program(0x88, Form::Vop3Vcc, 1), in, 64);
    EXPECT_EQ(run.error, "unspecified float bound NaN");
}

TEST(IrVectorCompareClassF32Execution, SdwaDwordSourcesKeepClassAndScalarDestination) {
    for (const Form form : kSdwaForms)
        for (const uint32_t wave : {64u, 32u}) {
            const auto words = sdwa_program(form);
            const auto decoded = rdna2_decode_one(words.data(), words.size());
            ASSERT_TRUE(decoded.has_sdwa);
            ASSERT_FALSE(decoded.has_modifier);
            ASSERT_EQ(decoded.sdwa_src0_sel, 6u);
            ASSERT_EQ(decoded.sdwa_src1_sel, 6u);
            check_class_run(class_inputs(0x5338), form, wave, words);
        }
}

TEST(IrVectorCompareClassF32Execution, EffectsDescribeOneClassDefinitionAndItsCmpxTwin) {
    ASSERT_EQ(vector_compare_class_f32_defs().size(), 1u);
    const auto& def = vector_compare_class_f32_defs().front();
    EXPECT_EQ(def.opcode, 0x88u);
    const auto& cmp = rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode);
    const auto& cmpx = rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode + 0x10);
    ASSERT_NE(cmp.name, nullptr);
    ASSERT_NE(cmpx.name, nullptr);
    EXPECT_EQ(std::string(cmp.name), "v_cmp_class_f32");
    EXPECT_EQ(std::string(cmpx.name), "v_cmpx_class_f32");
    EXPECT_EQ(cmp.lane_flow, Rdna2LaneFlow::LaneWise);
    EXPECT_EQ(cmp.vdst_dwords, 0u);
    EXPECT_EQ(cmp.ssrc_dwords[0], 1u);
    EXPECT_EQ(cmp.ssrc_dwords[1], 1u);
    EXPECT_TRUE(vopc_is_cmpx(0x98));
    EXPECT_EQ(find_vector_compare(0x98), nullptr);
    Block block;
    Builder ir(block);
    const auto value = def.build(ir, {ir.input(Type::U32), ir.input(Type::U32)});
    EXPECT_EQ(block.at(value).type, Type::Bool);
}

TEST(IrVectorCompareClassF32Execution, HalfClassRemainsUnsupportedBesideSingleSuccess) {
    check_class_run(class_inputs(0x5338), Form::VopcVcc, 64, compare_program(0x88, Form::VopcVcc));
    for (const uint32_t op : {0x8fu}) {
        EXPECT_EQ(find_vector_compare(op), nullptr);
        for (const Form form : kForms) {
            const auto lp = lane_program(compare_program(op, form), 64, 64);
            EXPECT_TRUE(
                recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config).empty());
        }
    }
}

TEST(IrVectorCompareClassF32Execution, EmptyExecKeepsEveryInactiveMaskAndDestination) {
    auto in = class_inputs(0x5338);
    for (size_t lane = 0; lane < in.lanes(); ++lane) in.active[lane] = false;
    for (const Form form : kForms)
        for (const uint32_t wave : {64u, 32u})
            check_class_run(in, form, wave, compare_program(0x88, form));
}

TEST(IrVectorCompareClassF32Execution, ExportsEveryPlainAndPreparedClassWitness) {
    ASSERT_EQ(export_compare_witness(vector_compare_class_f32_defs(), "vopc-class-f32-witness",
                                     0x5338000000ull),
              10u);
    EXPECT_EQ(append_wrapper_witnesses(), 24u);
}
