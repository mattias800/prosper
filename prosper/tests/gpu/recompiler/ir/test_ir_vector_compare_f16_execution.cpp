// #5336: F16 relation definitions consume prepared F32 bits, never raw half words.
// The guest tests select the VM's nominal preserve-half profile, not a device fact.
// Prepared NaNs pin the relation; raw half NaNs retain the existing eager-unpack refusal.
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_compare.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
struct Row {
    uint32_t opcode;
    const char* prepared;
    const char* raw;
};
// AMD 70648 section 12.9, Table 61. Prepared columns: 1:2, 2:1, 1:1, +0:-0,
// NaN:1, 1:NaN, +inf:+inf, -inf:+inf, sNaN:-NaN, -1:1. Ordered relations reject NaNs.
// Raw columns: -1:1, 1:-1, 1:1, +0:-0, +inf:+inf, -inf:+inf, minsub:+0, -minsub:-0.
constexpr Row kRows[] = {
    {0xc9, "1000000101", "10000101"}, {0xca, "0011001000", "00111000"},
    {0xcb, "1011001101", "10111101"}, {0xcc, "0100000000", "01000010"},
    {0xcd, "1100000101", "11000111"}, {0xce, "0111001000", "01111010"},
};
constexpr uint32_t kPrepared[][2] = {
    {0x3f800000, 0x40000000}, {0x40000000, 0x3f800000}, {0x3f800000, 0x3f800000},
    {0, 0x80000000},          {0x7fc00001, 0x3f800000}, {0x3f800000, 0x7fc00001},
    {0x7f800000, 0x7f800000}, {0xff800000, 0x7f800000}, {0x7f800001, 0xffc00001},
    {0xbf800000, 0x3f800000},
};
constexpr uint32_t kRaw[][2] = {
    {0xbc00, 0x3c00}, {0x3c00, 0xbc00}, {0x3c00, 0x3c00}, {0, 0x8000},
    {0x7c00, 0x7c00}, {0xfc00, 0x7c00}, {1, 0},           {0x8001, 0x8000},
};

// Independent binary16 field formula: normal (1024+fraction)*2^(exponent-25),
// subnormal fraction*2^-24. Every finite half is exactly representable as F32.
Known prepared_half(uint32_t raw) {
    const uint32_t half = raw & 0xffffu, exponent = (half >> 10) & 31u;
    const uint32_t fraction = half & 1023u, sign = (half & 0x8000u) << 16;
    if (exponent == 31) return fraction ? Known{} : Known(sign | 0x7f800000u);
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24)
                                          : std::ldexp(static_cast<float>(1024u + fraction),
                                                       static_cast<int>(exponent) - 25);
    return std::bit_cast<uint32_t>(magnitude) | sign;
}
uint32_t finite_word(uint32_t word) {
    for (uint32_t shift : {0u, 16u})
        if (((word >> shift) & 0x7c00u) == 0x7c00u && ((word >> shift) & 1023u))
            word ^= 0x0400u << shift;
    return word;
}
LaneInputs half_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = finite_word(in.s0[lane]);
        in.s1[lane] = finite_word(in.s1[lane]);
        if (lane < 48) {
            in.s0[lane] = kRaw[lane % std::size(kRaw)][0] | (0x4000u << 16);
            in.s1[lane] = kRaw[lane % std::size(kRaw)][1] | (0xbc00u << 16);
        }
    }
    return in;
}

struct Shape {
    uint32_t sel0 = 6, sel1 = 6, abs = 0, neg = 0, source = 0;
    bool sdwa = false;
};
std::vector<uint32_t> words(const VectorCompareDef& def, Form form, Shape shape = {}) {
    auto code = compare_program(def.opcode, form, shape.abs, shape.neg);
    if (shape.sdwa) {
        code[0] = (code[0] & ~0x1ffu) | 0xf9u;
        code.insert(code.begin() + 1, (shape.sel0 << 16) | (shape.sel1 << 24) | kVsrc0);
    } else if (shape.source) {
        const bool e32 = form == Form::VopcVcc || form == Form::CmpxVopc;
        const uint32_t operand = shape.source == 1   ? 242u
                                 : shape.source == 2 ? 129u
                                 : shape.source == 3 ? 255u
                                                     : 0u;
        const size_t slot = e32 ? 0 : 1;
        code[slot] = (code[slot] & ~0x1ffu) | operand;
        if (shape.source == 3) code.insert(code.begin() + (e32 ? 1 : 2), 0x4000bc00u);
        if (shape.source == 4) code.insert(code.begin(), {0xbe8003ffu, 0x4000bc00u});
    }
    return code;
}
Known source_bits(const LaneInputs& in, size_t lane, Shape shape, uint32_t slot) {
    uint32_t raw = slot ? in.s1[lane] : in.s0[lane];
    if (slot == 0 && shape.source)
        raw = shape.source == 1 ? 0x3c00u : shape.source == 2 ? 1u : 0x4000bc00u;
    const uint32_t select = slot ? shape.sel1 : shape.sel0;
    const auto value = prepared_half(raw >> (shape.sdwa && select == 5 ? 16 : 0));
    if (!value) return {};
    uint32_t result = *value;
    if (shape.abs & (1u << slot)) result &= 0x7fffffffu;
    if (shape.neg & (1u << slot)) result ^= 0x80000000u;
    return result;
}
void check_output(const VectorCompareDef& def, Form form, Shape shape, const LaneInputs& in,
                  const std::vector<uint32_t>& output) {
    ASSERT_GE(output.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto a = source_bits(in, lane, shape, 0), b = source_bits(in, lane, shape, 1);
        if (!a || !b) {
            ADD_FAILURE() << "finite fixture contains a half NaN";
            return;
        }
        const Known result = eval_vector_compare(def, *a, *b);
        if (!result) {
            ADD_FAILURE() << "prepared relation is unknown";
            return;
        }
        EXPECT_EQ(output[lane],
                  expected_mask_read(form, in.active[lane], *result != 0, in.old_d[lane]))
            << "lane " << lane;
    }
}
void check_guest(const VectorCompareDef& def, Form form, uint32_t wave, const LaneInputs& in,
                 Shape shape = {}) {
    const auto run = run_vector_op_per_lane(words(def, form, shape), in, wave, true);
    ASSERT_TRUE(run.error.empty()) << run.error;
    check_output(def, form, shape, in, run.d);
}
struct Witness {
    const VectorCompareDef* def;
    Form form;
    uint32_t wave;
    Shape shape;
};
std::vector<Witness> witnesses() {
    std::vector<Witness> result;
    for (const auto& def : vector_compare_f16_defs())
        for (Form form : kForms)
            for (uint32_t wave : {32u, 64u}) {
                result.push_back({&def, form, wave, {}});
                for (uint32_t source = 1; source <= 4; ++source)
                    result.push_back({&def, form, wave, {.source = source}});
                if (form == Form::VopcVcc || form == Form::CmpxVopc) {
                    for (uint32_t a : {4u, 5u})
                        for (uint32_t b : {4u, 5u})
                            result.push_back(
                                {&def, form, wave, {.sel0 = a, .sel1 = b, .sdwa = true}});
                } else {
                    for (const auto& modifier : {Shape{.abs = 1}, Shape{.abs = 2}, Shape{.neg = 1},
                                                 Shape{.neg = 2}, Shape{.abs = 3, .neg = 3}})
                        result.push_back({&def, form, wave, modifier});
                }
            }
    return result;
}
std::vector<uint32_t> prepared_module(const VectorCompareDef& def, uint32_t a, uint32_t c) {
    SpirvCompute b;
    b.begin(1, nullptr, 1, 1, 1, 64, 0, true, true);
    const uint32_t lhs = b.uconst(a), rhs = b.uconst(c);
    const uint32_t test = emit_vector_compare(b, def, lhs, rhs);
    b.store_output_word(b.sel(test, b.uconst(1), b.uconst(0)), 1, 0);
    return b.finish();
}
} // namespace

TEST(IrVectorCompareF16Execution, EveryDefinitionEveryFormAndWaveUsesPreparedSources) {
    for (const auto& def : vector_compare_f16_defs())
        for (Form form : kForms)
            for (uint32_t wave : {64u, 32u})
                for (uint32_t seed = 1; seed <= 3; ++seed)
                    check_guest(def, form, wave, half_inputs(seed * 0x5336u));
}
TEST(IrVectorCompareF16Execution, PreparedHandTableIncludesNanSignedZeroAndInfinity) {
    ASSERT_EQ(std::size(kRows), vector_compare_f16_defs().size());
    for (const auto& row : kRows) {
        const auto* def = find_vector_compare(row.opcode);
        ASSERT_NE(def, nullptr);
        for (size_t pair = 0; pair < std::size(kPrepared); ++pair) {
            const auto a = kPrepared[pair][0], c = kPrepared[pair][1];
            const uint32_t expected = row.prepared[pair] == '1';
            EXPECT_EQ(eval_vector_compare(*def, a, c), Known(expected));
            bpermute_oracle::Interpreter vm(prepared_module(*def, a, c));
            const auto output = vm.run_buffers(1, {{0, {0}}, {1, {99}}}, 1);
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            ASSERT_EQ(output.size(), 1u);
            EXPECT_EQ(output[0], expected) << "op " << row.opcode << " pair " << pair;
        }
    }
}
TEST(IrVectorCompareF16Execution, RawHandTableExercisesTheAdapterAndNominalSubnormals) {
    for (const auto& row : kRows) {
        const auto* def = find_vector_compare(row.opcode);
        ASSERT_NE(def, nullptr);
        auto in = half_inputs(17);
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            in.s0[lane] = kRaw[lane % std::size(kRaw)][0];
            in.s1[lane] = kRaw[lane % std::size(kRaw)][1];
        }
        for (Form form : kForms)
            for (uint32_t wave : {32u, 64u}) {
                const auto run = run_vector_op_per_lane(words(*def, form), in, wave, true);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    EXPECT_EQ(run.d[lane],
                              expected_mask_read(form, in.active[lane],
                                                 row.raw[lane % std::size(kRaw)] == '1',
                                                 in.old_d[lane]));
            }
    }
}
TEST(IrVectorCompareF16Execution, SourceModifiersSdwaAndInlineFormsKeepTheirPreparation) {
    for (const auto& w : witnesses()) {
        SCOPED_TRACE(rdna2_instruction_effects(Rdna2Format::VOPC, w.def->opcode).name);
        check_guest(*w.def, w.form, w.wave, half_inputs(w.def->opcode + w.wave), w.shape);
    }
}
TEST(IrVectorCompareF16Execution, InactiveLanesReadZeroAndActiveLanesDiffer) {
    const auto* def = find_vector_compare(0xc9);
    ASSERT_NE(def, nullptr);
    auto in = half_inputs(11);
    in.s0[0] = in.s0[1] = 0xbc00;
    in.s1[0] = in.s1[1] = 0x3c00;
    in.active[0] = false;
    in.active[1] = in.active[2] = true;
    in.s0[2] = 0x4000;
    in.s1[2] = 0x3c00;
    for (uint32_t wave : {32u, 64u}) {
        const auto run = run_vector_op_per_lane(words(*def, Form::VopcVcc), in, wave, true);
        ASSERT_TRUE(run.error.empty()) << run.error;
        ASSERT_EQ(run.d.size(), in.lanes());
        EXPECT_EQ(run.d[0], 0u);
        EXPECT_EQ(run.d[1], 1u);
        EXPECT_EQ(run.d[2], 0u);
        check_output(*def, Form::VopcVcc, {}, in, run.d);
    }
}
TEST(IrVectorCompareF16Execution, ExistingHalfNanAndNominalProfileBoundaryRemainExplicit) {
    const auto in = half_inputs(23);
    for (const auto& def : vector_compare_f16_defs()) {
        const auto code = words(def, Form::VopcVcc);
        EXPECT_TRUE(run_vector_op_per_lane(code, in, 64, true).error.empty());
        EXPECT_FALSE(run_vector_op_per_lane(code, in, 64, false).error.empty());
        for (uint32_t nan : {0x00007e01u, 0x7e013c00u}) {
            auto bad = in;
            bad.s0[1] = nan; // selected or discarded half NaN; the old VM decodes both.
            const auto run = run_vector_op_per_lane(code, bad, 64, true);
            EXPECT_TRUE(run.d.empty());
            EXPECT_NE(run.error.find("unspecified half NaN payload"), std::string::npos);
        }
    }
}
TEST(IrVectorCompareF16Execution, MissingInputsAndUnadmittedRelationsStayUnknown) {
    for (const auto& def : vector_compare_f16_defs()) {
        EXPECT_FALSE(eval_vector_compare(def, Known{}, 0).has_value());
        EXPECT_FALSE(eval_vector_compare(def, 0, Known{}).has_value());
        EXPECT_EQ(find_vector_compare(def.opcode + 0x10u), nullptr);
    }
    EXPECT_EQ(find_vector_compare(0xc8), nullptr);
    EXPECT_EQ(find_vector_compare(0xcf), nullptr);
}
TEST(IrVectorCompareF16Execution, EffectsAndDefinitionOperationsAgree) {
    std::set<Opcode> seen;
    for (const auto& def : vector_compare_f16_defs()) {
        const auto& effect = rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode);
        const auto& twin = rdna2_instruction_effects(Rdna2Format::VOPC, def.opcode + 0x10u);
        ASSERT_NE(effect.name, nullptr);
        ASSERT_NE(twin.name, nullptr);
        const std::string name = effect.name;
        EXPECT_TRUE(name.ends_with("_f16"));
        EXPECT_EQ(std::string(twin.name), "v_cmpx_" + name.substr(6));
        EXPECT_EQ(effect.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(effect.vdst_dwords, 0u);
        EXPECT_EQ(effect.ssrc_dwords[0], 1u);
        EXPECT_EQ(effect.ssrc_dwords[1], 1u);
        Block block;
        Builder ir(block);
        const auto a = ir.input(Type::U32), b = ir.input(Type::U32);
        const auto result = def.build(ir, {a, b});
        EXPECT_EQ(block.at(result).type, Type::Bool);
        for (const auto& inst : block.insts()) seen.insert(inst.op);
    }
    for (Opcode op :
         {Opcode::F32Lt, Opcode::F32Eq, Opcode::F32Le, Opcode::F32Gt, Opcode::F32Lg, Opcode::F32Ge})
        EXPECT_TRUE(seen.contains(op));
}
TEST(IrVectorCompareF16Execution, ExportsAndExecutesEverySavedWitnessModule) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch setting.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "vopc-f16-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error);
    }
    ASSERT_FALSE(fs::exists(root)) << root << " must not exist; export refuses to overwrite";
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directories(dump));
    size_t count = 0;
    for (const auto& w : witnesses()) {
        const auto lp = lane_program(words(*w.def, w.form, w.shape), 64, w.wave);
        const uint64_t address = 0x5336000000ull + count * 0x100;
        const auto stem = dump / ("cs_" + std::to_string(address) + "_" + std::to_string(count));
        std::ofstream raw(stem.string() + ".bin", std::ios::binary);
        raw.write(reinterpret_cast<const char*>(lp.code.data()),
                  static_cast<std::streamsize>(lp.code.size() * sizeof(uint32_t)));
        raw.close();
        ASSERT_TRUE(raw.good());
        ASSERT_TRUE(write_refused_compute_inputs(stem.string() + ".inputs", address, &lp.table,
                                                 lp.config, "synthetic F16 witness"));
        const auto captured = read_refused_shader_inputs(stem.string() + ".inputs");
        ASSERT_TRUE(captured.incomplete.empty()) << captured.incomplete;
        const auto replay = replay_refused_shader_inputs(lp.code, captured);
        ASSERT_TRUE(replay.produced) << replay.reject;
        std::ofstream spv(stem.string() + ".spv", std::ios::binary);
        spv.write(reinterpret_cast<const char*>(replay.spirv.data()),
                  static_cast<std::streamsize>(replay.spirv.size() * sizeof(uint32_t)));
        spv.close();
        ASSERT_TRUE(spv.good());
        std::ifstream saved(stem.string() + ".spv", std::ios::binary | std::ios::ate);
        ASSERT_EQ(saved.tellg(),
                  static_cast<std::streamoff>(replay.spirv.size() * sizeof(uint32_t)));
        std::vector<uint32_t> module(replay.spirv.size());
        saved.seekg(0);
        saved.read(reinterpret_cast<char*>(module.data()),
                   static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
        ASSERT_TRUE(saved.good());
        ASSERT_EQ(module, replay.spirv);
        const auto in = half_inputs(w.def->opcode + w.wave);
        std::vector<uint32_t> buffer;
        for (const auto* a : {&in.s0, &in.s1, &in.old_d})
            buffer.insert(buffer.end(), a->begin(), a->end());
        for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
        bpermute_oracle::Interpreter vm(module);
        vm.half_rte_preserve_denorm = true;
        const auto output = vm.run_buffers(64, {{3, buffer}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        check_output(*w.def, w.form, w.shape, in, output);
        ++count;
    }
    EXPECT_EQ(count, 576u);
}
