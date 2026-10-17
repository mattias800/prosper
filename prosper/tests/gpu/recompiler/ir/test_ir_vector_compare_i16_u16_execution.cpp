// #5333: sixteen-bit compares discard high garbage, extend each source in order and write
// the EXEC-masked lane result. The hand table is from AMD 70648 section 12.9, Table 61;
// it is independent of eval. Shared source/provenance and mask/proof tests stay unchanged.
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_compare.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <gtest/gtest.h>

#include <cstddef>
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
struct Pair {
    uint32_t a, b;
};
constexpr Pair kPairs[] = {
    {0x00000000, 0x00000000}, {0x7777ffff, 0x00000000}, {0xffff8000, 0x00007fff},
    {0x00007fff, 0xffff8000}, {0x00000001, 0x1234ffff}, {0x12340001, 0xffff0001},
    {0x123400ff, 0xabcdff00}, {0x00008000, 0xffff8000},
};
struct HandRow {
    uint32_t opcode;
    const char* truth;
};
// Columns are kPairs in order. Low 0xffff is -1 for I16 and 65535 for U16;
// 0x8000 is the signed minimum. Equal low halves ignore every upper-half bit.
constexpr HandRow kHand[] = {
    {0x89, "01100000"}, {0x8a, "10000101"}, {0x8b, "11100101"}, {0x8c, "00011010"},
    {0x8d, "01111010"}, {0x8e, "10011111"}, {0xa9, "00011010"}, {0xaa, "10000101"},
    {0xab, "10011111"}, {0xac, "01100000"}, {0xad, "01111010"}, {0xae, "11100101"},
};

LaneInputs compare_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        if (lane < 48) {
            in.s0[lane] = kPairs[lane % std::size(kPairs)].a;
            in.s1[lane] = kPairs[lane % std::size(kPairs)].b;
        } else if (lane % 3 == 0) {
            in.s1[lane] = (in.s0[lane] & 0xffffu) | (~in.s0[lane] & 0xffff0000u);
        }
    }
    return in;
}

std::vector<uint32_t> sdwa_program(uint32_t opcode, Form form, uint32_t sel0, uint32_t sel1) {
    auto words = compare_program(opcode, form);
    words[0] = (words[0] & ~0x1ffu) | 0xf9u;
    words.insert(words.begin() + 1, (sel0 << 16) | (sel1 << 24) | kVsrc0);
    return words;
}

void check_hand(const HandRow& row, Form form, uint32_t wave) {
    auto in = lane_inputs(row.opcode + 1);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = kPairs[lane % std::size(kPairs)].a;
        in.s1[lane] = kPairs[lane % std::size(kPairs)].b;
    }
    const auto run = run_vector_op_per_lane(compare_program(row.opcode, form), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        EXPECT_EQ(run.d[lane],
                  expected_mask_read(form, in.active[lane],
                                     row.truth[lane % std::size(kPairs)] == '1', in.old_d[lane]))
            << "op " << row.opcode << " form " << int(form) << " wave " << wave << " lane " << lane;
}
} // namespace

TEST(IrVectorCompareI16U16Execution, EveryDefinitionEveryFormAndWaveWritesTheExecMaskedMask) {
    for (const auto& def : vector_compare_i16_u16_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode).name);
        for (Form form : kForms)
            for (uint32_t wave : {64u, 32u})
                for (uint32_t seed = 1; seed <= 3; ++seed)
                    check_compare_run(def, form, wave, compare_inputs(seed * 0x5333u));
    }
}

TEST(IrVectorCompareI16U16Execution, InactiveLanesReadZeroAndActiveLanesDiffer) {
    for (uint32_t op : {0x89u, 0xa9u}) {
        auto in = compare_inputs(11);
        in.s0[0] = in.s0[1] = 0;
        in.s1[0] = in.s1[1] = 1;
        in.active[0] = false;
        in.active[1] = true;
        in.s0[2] = 2;
        in.s1[2] = 1;
        in.active[2] = true;
        for (uint32_t wave : {64u, 32u}) {
            const auto run = run_vector_op_per_lane(compare_program(op, Form::VopcVcc), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), in.lanes());
            EXPECT_EQ(run.d[0], 0u) << "inactive lane's 0 < 1 must be masked";
            EXPECT_EQ(run.d[1], 1u) << "active true control";
            EXPECT_EQ(run.d[2], 0u) << "active false control";
            std::set<uint32_t> active;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                if (in.active[lane])
                    active.insert(run.d[lane]);
                else
                    EXPECT_EQ(run.d[lane], 0u) << "lane " << lane;
            }
            EXPECT_EQ(active, (std::set<uint32_t>{0u, 1u}));
        }
    }
}

TEST(IrVectorCompareI16U16Execution, HandWorkedIsaTruthIgnoresHighHalfGarbage) {
    ASSERT_EQ(std::size(kHand), vector_compare_i16_u16_defs().size());
    for (const auto& row : kHand) {
        const auto* def = find_vector_compare(row.opcode);
        ASSERT_NE(def, nullptr);
        for (size_t pair = 0; pair < std::size(kPairs); ++pair)
            EXPECT_EQ(eval_vector_compare(*def, kPairs[pair].a, kPairs[pair].b),
                      Known(row.truth[pair] == '1'))
                << "op " << row.opcode << " pair " << pair;
        for (Form form : kForms)
            for (uint32_t wave : {64u, 32u}) check_hand(row, form, wave);
    }
}

TEST(IrVectorCompareI16U16Execution, SdwaWordSelectionPrecedesTheSixteenBitRelation) {
    const auto in = compare_inputs(17);
    for (const auto& def : vector_compare_i16_u16_defs())
        for (Form form : {Form::VopcVcc, Form::CmpxVopc})
            for (uint32_t wave : {64u, 32u})
                for (uint32_t sel0 : {4u, 5u}) {
                    const uint32_t sel1 = 9u - sel0;
                    const auto run = run_vector_op_per_lane(
                        sdwa_program(def.opcode, form, sel0, sel1), in, wave);
                    ASSERT_TRUE(run.error.empty()) << run.error;
                    ASSERT_EQ(run.d.size(), in.lanes());
                    for (size_t lane = 0; lane < in.lanes(); ++lane) {
                        const Known test =
                            eval_vector_compare(def, in.s0[lane] >> (16u * (sel0 - 4u)),
                                                in.s1[lane] >> (16u * (sel1 - 4u)));
                        ASSERT_TRUE(test.has_value());
                        if (!test.has_value()) return;
                        EXPECT_EQ(run.d[lane], expected_mask_read(form, in.active[lane], *test != 0,
                                                                  in.old_d[lane]))
                            << "op " << def.opcode << " lane " << lane;
                    }
                }
}

TEST(IrVectorCompareI16U16Execution, ExistingSdwaSextRefusalRemainsVisible) {
    const auto in = compare_inputs(19);
    for (const auto& def : vector_compare_i16_u16_defs()) {
        auto words = sdwa_program(def.opcode, Form::VopcVcc, 5, 4);
        ASSERT_FALSE(rdna2_decode_one(words.data(), words.size()).has_modifier);
        const auto positive = run_vector_op_per_lane(words, in, 64);
        ASSERT_TRUE(positive.error.empty()) << positive.error;
        words[1] |= 1u << 19; // existing no-SEXT admission, not a new guard
        ASSERT_TRUE(rdna2_decode_one(words.data(), words.size()).has_modifier);
        const auto refused = run_vector_op_per_lane(words, in, 64);
        EXPECT_TRUE(refused.d.empty());
        EXPECT_FALSE(refused.error.empty());
    }
}

TEST(IrVectorCompareI16U16Execution, MissingNumericInputsNeverBecomeAComparisonAnswer) {
    for (const auto& def : vector_compare_i16_u16_defs()) {
        EXPECT_TRUE(eval_vector_compare(def, 1u, 2u).has_value());
        EXPECT_FALSE(eval_vector_compare(def, Known{}, 2u).has_value());
        EXPECT_FALSE(eval_vector_compare(def, 1u, Known{}).has_value());
        EXPECT_FALSE(eval_vector_compare(def, Known{}, Known{}).has_value());
        EXPECT_EQ(find_vector_compare(def.opcode + 0x10u), nullptr)
            << "CMPX is normalized by lowering";
    }
    EXPECT_EQ(find_vector_compare(0x8f), nullptr);
    EXPECT_EQ(find_vector_compare(0xaf), nullptr);
}

TEST(IrVectorCompareI16U16Execution, EffectsAndOrderedExtensionsAgreeForEveryDefinition) {
    std::set<Opcode> seen;
    for (const auto& def : vector_compare_i16_u16_defs()) {
        const auto& cmp = rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode);
        const auto& cmpx = rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode + 0x10u);
        ASSERT_NE(cmp.name, nullptr);
        ASSERT_NE(cmpx.name, nullptr);
        const std::string name = cmp.name;
        EXPECT_EQ(std::string(cmpx.name), "v_cmpx_" + name.substr(6));
        EXPECT_EQ(cmp.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(cmp.vdst_dwords, 0u);
        EXPECT_EQ(cmp.ssrc_dwords[0], 1u);
        EXPECT_EQ(cmp.ssrc_dwords[1], 1u);
        EXPECT_EQ(cmp.ssrc_dwords[2], 0u);
        EXPECT_TRUE(vopc_is_cmpx(def.opcode + 0x10u));
        EXPECT_FALSE(vopc_is_cmpx(def.opcode));
        Block block;
        Builder ir(block);
        const Value s0 = ir.input(Type::U32), s1 = ir.input(Type::U32);
        const Value result = def.build(ir, {s0, s1});
        EXPECT_EQ(block.at(result).type, Type::Bool);
        std::vector<const Inst*> extracts;
        for (const auto& inst : block.insts()) {
            seen.insert(inst.op);
            if (inst.op == Opcode::BfeS || inst.op == Opcode::BfeU) extracts.push_back(&inst);
        }
        ASSERT_EQ(extracts.size(), 2u);
        EXPECT_EQ(extracts[0]->args[0], s0);
        EXPECT_EQ(extracts[1]->args[0], s1);
        const Opcode extension = name.ends_with("_i16") ? Opcode::BfeS : Opcode::BfeU;
        for (const auto* extract : extracts) {
            EXPECT_EQ(extract->op, extension);
            EXPECT_EQ(block.at(extract->args[1]).imm, 0u);
            EXPECT_EQ(block.at(extract->args[2]).imm, 16u);
        }
    }
    for (Opcode op :
         {Opcode::BfeS, Opcode::BfeU, Opcode::IEq, Opcode::INe, Opcode::SLt, Opcode::SLe,
          Opcode::SGt, Opcode::SGe, Opcode::ULt, Opcode::ULe, Opcode::UGt, Opcode::UGe})
        EXPECT_TRUE(seen.contains(op));
}

TEST(IrVectorCompareI16U16Execution, ExportsEveryFormForTheCorpusGate) {
    constexpr uint64_t base = 0x5333000000ull;
    ASSERT_EQ(export_compare_witness(vector_compare_i16_u16_defs(), "vopc-i16-u16-witness", base),
              vector_compare_i16_u16_defs().size() * std::size(kForms) * 2u);
    // Read and execute each exact exported module, rather than only regenerating a similar program.
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch setting.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const auto root = configured ? std::filesystem::path(configured)
                                 : prosper_test::test_scratch_dir() / "vopc-i16-u16-witness";
    size_t index = 0;
    for (const auto& def : vector_compare_i16_u16_defs())
        for (Form form : kForms)
            for (uint32_t wave : {64u, 32u}) {
                const auto stem =
                    root / "refused_shaders_20000101-000000Z_synthetic" /
                    ("cs_" + std::to_string(base + index * 0x100) + "_" + std::to_string(index));
                std::ifstream raw(stem.string() + ".bin", std::ios::binary | std::ios::ate);
                ASSERT_TRUE(raw.good());
                const std::streamoff bytes = raw.tellg();
                ASSERT_GT(bytes, 0);
                ASSERT_EQ(bytes % 4, 0);
                std::vector<uint32_t> code(static_cast<size_t>(bytes) / 4);
                raw.seekg(0);
                raw.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(bytes));
                ASSERT_TRUE(raw.good());
                const auto inputs = read_refused_shader_inputs(stem.string() + ".inputs");
                ASSERT_TRUE(inputs.incomplete.empty()) << inputs.incomplete;
                const auto replay = replay_refused_shader_inputs(code, inputs);
                ASSERT_TRUE(replay.produced) << replay.reject;
                std::ofstream spv(stem.string() + ".spv", std::ios::binary);
                spv.write(reinterpret_cast<const char*>(replay.spirv.data()),
                          static_cast<std::streamsize>(replay.spirv.size() * sizeof(uint32_t)));
                spv.close();
                ASSERT_TRUE(spv.good());
                const auto in = compare_inputs(def.opcode + wave);
                std::vector<uint32_t> buffer;
                for (const auto* a : {&in.s0, &in.s1, &in.old_d})
                    buffer.insert(buffer.end(), a->begin(), a->end());
                for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
                bpermute_oracle::Interpreter vm(replay.spirv);
                const auto output = vm.run_buffers(64, {{3, buffer}}, 3);
                ASSERT_TRUE(vm.error.empty()) << vm.error;
                ASSERT_EQ(output.size(), buffer.size());
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const Known test = eval_vector_compare(def, in.s0[lane], in.s1[lane]);
                    ASSERT_TRUE(test.has_value());
                    if (!test.has_value()) return;
                    EXPECT_EQ(output[lane],
                              expected_mask_read(form, in.active[lane], *test != 0, in.old_d[lane]))
                        << "witness " << index << " lane " << lane;
                }
                ++index;
            }
    EXPECT_EQ(index, 120u);
}
