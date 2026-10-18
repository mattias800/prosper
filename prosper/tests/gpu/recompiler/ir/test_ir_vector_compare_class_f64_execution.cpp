// AMD 70648 Table61: CLASS_F64 observes both raw words, including NaN payload and signed zero.
// It is an integer classification, independent of floating-point rounding/denormal arithmetic.
#include "fixtures/test_scratch.h"
#include "gpu/recompiler/float64_support.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ops/vector_compare.hpp"
#include "gpu/recompiler/ir/lower/vector_alu_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace {
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
struct ClassRow {
    uint64_t raw;
    uint32_t bit;
};
// One hand-worked representative per class, then low-word, normal-boundary and payload traps.
constexpr ClassRow kClasses[] = {
    {0x7ff0000000000001ull, 1},
    {0x7ff8000000000000ull, 2},
    {0xfff0000000000000ull, 4},
    {0xbff0000000000000ull, 8},
    {0x8000000000000001ull, 16},
    {0x8000000000000000ull, 32},
    {0, 64},
    {1, 128},
    {0x3ff0000000000000ull, 256},
    {0x7ff0000000000000ull, 512},
    {0xfff0000080000000ull, 1},
    {0xffffffffffffffffull, 2},
    {0x000fffffffffffffull, 128},
    {0x0010000000000000ull, 256},
    {0x800fffffffffffffull, 16},
    {0x8010000000000000ull, 8},
    {0x0000000080000000ull, 128},
    {0x7fefffffffffffffull, 256},
};
constexpr uint32_t kPreserve = kDefaultComputePgmRsrc1 | (0xc0u << 12);
uint32_t positive_class(uint32_t bit) {
    switch (bit) {
        case 4: return 512;
        case 8: return 256;
        case 16: return 128;
        case 32: return 64;
        default: return bit;
    }
}
uint32_t negated_class(uint32_t bit) {
    switch (bit) {
        case 4: return 512;
        case 512: return 4;
        case 8: return 256;
        case 256: return 8;
        case 16: return 128;
        case 128: return 16;
        case 32: return 64;
        case 64: return 32;
        default: return bit;
    }
}
struct Data {
    LaneInputs in;
    std::vector<uint32_t> bits;
};
Data data(uint32_t seed, uint32_t mask = 1) {
    Data out{lane_inputs(seed), {}};
    out.in.s2.resize(out.in.lanes());
    out.bits.reserve(out.in.lanes());
    for (std::size_t lane = 0; lane < out.in.lanes(); ++lane) {
        const auto row = kClasses[(lane + seed) % std::size(kClasses)];
        out.in.s0[lane] = static_cast<uint32_t>(row.raw);
        out.in.s1[lane] = static_cast<uint32_t>(row.raw >> 32);
        const uint32_t wrong = row.bit == 512u ? 1u : row.bit << 1u;
        const std::array masks{row.bit, wrong, 0x3ffu, 0xfffffc00u, 0u, row.bit | 0xfffffc00u};
        out.in.s2[lane] = masks[lane % masks.size()];
        out.in.active[lane] = mask == 0 || (mask == 1 && lane % 4 != 0);
        out.bits.push_back(row.bit);
    }
    return out;
}
struct Program {
    Form form;
    uint32_t wave;
    uint32_t mode = kDefaultComputePgmRsrc1;
    uint32_t abs = 0, neg = 0;
    std::vector<uint32_t> words;
    std::optional<ClassRow> source;
    std::optional<uint32_t> mask;
    bool full_exec = false;
};
Program program(Form form, uint32_t wave, uint32_t abs = 0, uint32_t neg = 0) {
    return {form, wave, kDefaultComputePgmRsrc1,
            abs,  neg,  compare_program(0xa8, form, abs, neg, kVsrc2),
            {},   {},   false};
}
void source_field(Program& p, uint32_t source) {
    if (p.form == Form::VopcVcc || p.form == Form::CmpxVopc)
        p.words[0] = (p.words[0] & ~0x1ffu) | source;
    else
        p.words[1] = (p.words[1] & ~0x1ffu) | source;
}
void mask_field(Program& p, uint32_t source) {
    p.words[1] = (p.words[1] & ~(0x1ffu << 9)) | (source << 9);
}
std::vector<uint32_t> scalar_pair(ClassRow row) {
    return {0xbe9e03ffu, uint32_t(row.raw), 0xbe9f03ffu, uint32_t(row.raw >> 32)};
}
void prepend(Program& p, const std::vector<uint32_t>& prefix) {
    p.words.insert(p.words.begin(), prefix.begin(), prefix.end());
}
void check(const Program& p, const Data& input) {
    const auto run = run_vector_op_per_lane(p.words, input.in, p.wave, false, p.mode);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), input.in.lanes());
    const auto* def = find_vector_compare(0xa8);
    ASSERT_NE(def, nullptr);
    EXPECT_TRUE(def->pair_sources);
    for (std::size_t lane = 0; lane < input.in.lanes(); ++lane) {
        uint64_t raw = p.source ? p.source->raw
                                : uint64_t(input.in.s0[lane]) | (uint64_t(input.in.s1[lane]) << 32);
        uint32_t bit = p.source ? p.source->bit : input.bits[lane];
        if (p.abs & 1u) {
            raw &= 0x7fffffffffffffffull;
            bit = positive_class(bit);
        }
        if (p.neg & 1u) {
            raw ^= 0x8000000000000000ull;
            bit = negated_class(bit);
        }
        const uint32_t mask = p.mask.value_or(input.in.s2[lane]);
        const bool truth = (bit & mask) != 0;
        EXPECT_EQ(
            eval_vector_compare_pairs(*def, uint32_t(raw), uint32_t(raw >> 32), mask, std::nullopt),
            Known(truth));
        EXPECT_EQ(run.d[lane], expected_mask_read(p.form, p.full_exec || input.in.active[lane],
                                                  truth, input.in.old_d[lane]))
            << "form=" << int(p.form) << " wave=" << p.wave << " lane=" << lane;
    }
}
std::vector<Program> witnesses() {
    std::vector<Program> out;
    for (const Form form : kForms)
        for (uint32_t wave : {32u, 64u})
            for (uint32_t mode : {kDefaultComputePgmRsrc1, kPreserve, UINT32_MAX}) {
                auto p = program(form, wave);
                p.mode = mode;
                out.push_back(p);
            }
    for (const Form form : {Form::Vop3Sgpr, Form::Vop3Vcc, Form::CmpxVop3})
        for (uint32_t wave : {32u, 64u})
            for (uint32_t abs = 0; abs < 4; ++abs)
                for (uint32_t neg = 0; neg < 4; ++neg)
                    if (abs || neg) out.push_back(program(form, wave, abs, neg));
    struct Constant {
        uint32_t code, literal;
        ClassRow value;
    };
    constexpr Constant constants[] = {
        {0xf0, 0, {0x3fe0000000000000ull, 256}},
        {0xf1, 0, {0xbfe0000000000000ull, 8}},
        {0xf2, 0, {0x3ff0000000000000ull, 256}},
        {0xf3, 0, {0xbff0000000000000ull, 8}},
        {0xf4, 0, {0x4000000000000000ull, 256}},
        {0xf5, 0, {0xc000000000000000ull, 8}},
        {0xf6, 0, {0x4010000000000000ull, 256}},
        {0xf7, 0, {0xc010000000000000ull, 8}},
        {0xf8, 0, {0x3fc45f306dc9c882ull, 256}},
        {0x80, 0, {0, 64}},
        {0x7d, 0, {0, 64}},
        {0xff, 0x3ff00000, {0x3ff0000000000000ull, 256}},
        {0xff, 0x80000000, {0x8000000000000000ull, 32}},
        {0xff, 0x7ff00000, {0x7ff0000000000000ull, 512}},
        {0xff, 0x7ff80000, {0x7ff8000000000000ull, 2}},
        {0xff, 1, {0x0000000100000000ull, 128}},
    };
    for (const auto& c : constants)
        for (uint32_t wave : {32u, 64u}) {
            auto p = program(Form::VopcVcc, wave);
            source_field(p, c.code);
            if (c.code == 0xff) p.words.insert(p.words.begin() + 1, c.literal);
            p.source = c.value;
            out.push_back(p);
        }
    for (const Form form : kForms)
        for (uint32_t wave : {32u, 64u})
            for (const auto row : {kClasses[6], kClasses[10], kClasses[16]}) {
                auto p = program(form, wave);
                source_field(p, 30);
                prepend(p, scalar_pair(row));
                p.source = row;
                out.push_back(p);
            }
    for (const Form form : {Form::Vop3Sgpr, Form::Vop3Vcc, Form::CmpxVop3})
        for (uint32_t wave : {32u, 64u}) {
            auto p = program(form, wave);
            mask_field(p, 34);
            prepend(p, {0xbea203ffu, 0x3ffu});
            p.mask = 0x3ff;
            out.push_back(p); // s35 is intentionally absent.
            p = program(form, wave);
            mask_field(p, 0xff);
            p.words.insert(p.words.begin() + 2, 0x3ff);
            p.mask = 0x3ff;
            out.push_back(p);
            p = program(form, wave);
            source_field(p, 30);
            mask_field(p, 34);
            auto prefix = scalar_pair(kClasses[10]);
            prefix.insert(prefix.end(), {0xbea203ffu, 0x3ffu, kSMovExecAll});
            prepend(p, prefix);
            p.source = kClasses[10];
            p.mask = 0x3ff;
            p.full_exec = true;
            out.push_back(p);
        }
    return out;
}
TEST(IrVectorCompareClassF64Execution, EveryFormAndWaveMatchesIndependentClassBits) {
    for (const Form form : kForms)
        for (uint32_t wave : {32u, 64u})
            for (uint32_t seed : {0u, 1u, 7u, 17u}) check(program(form, wave), data(seed));
}
TEST(IrVectorCompareClassF64Execution, RawClassesIgnoreArithmeticDenormalAndRoundingModes) {
    reset_float64_support_for_test();
    for (uint32_t mode : {kDefaultComputePgmRsrc1, kPreserve, UINT32_MAX}) {
        auto p = program(Form::VopcVcc, 64);
        p.mode = mode;
        check(p, data(0, 0));
    }
}
TEST(IrVectorCompareClassF64Execution, EveryPreparedAndConstantSourceHasAnExecutionWitness) {
    for (const auto& p : witnesses())
        for (uint32_t mask : {0u, 1u, 2u}) check(p, data(p.wave, mask));
}
TEST(IrVectorCompareClassF64Execution, ClassMaskIgnoresTheUnusedHighWord) {
    const auto* def = find_vector_compare(0xa8);
    ASSERT_NE(def, nullptr);
    for (const auto row : kClasses)
        for (uint32_t bit = 0; bit < 10; ++bit)
            EXPECT_EQ(eval_vector_compare_pairs(*def, uint32_t(row.raw), uint32_t(row.raw >> 32),
                                                1u << bit, std::nullopt),
                      Known(row.bit == (1u << bit)));
}
TEST(IrVectorCompareClassF64Execution, MissingScalarHighAndNonzeroInlineIntegerStayRefused) {
    auto in = data(3);
    auto p = program(Form::VopcVcc, 64);
    source_field(p, 30);
    prepend(p, {0xbe9e0380u});
    EXPECT_FALSE(run_vector_op_per_lane(p.words, in.in, 64).error.empty());
    p = program(Form::VopcVcc, 64);
    source_field(p, 0x81);
    EXPECT_FALSE(run_vector_op_per_lane(p.words, in.in, 64).error.empty());
    // Paired positive: a real zero pair is supported and has its IEEE zero category.
    p = program(Form::VopcVcc, 64);
    source_field(p, 30);
    prepend(p, scalar_pair(kClasses[6]));
    p.source = kClasses[6];
    check(p, in);
}
TEST(IrVectorCompareClassF64Execution, HalfRemainsRefusedBesideDoubleClass) {
    check(program(Form::VopcVcc, 64), data(8));
    for (const Form form : kForms) {
        const auto p = lane_program(compare_program(0x8f, form), 64, 64);
        EXPECT_TRUE(recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
    }
}
TEST(IrVectorCompareClassF64Execution, ScalarPairsNeedBothGenuineUniformWords) {
    for (int local : {-1, 30, 31, 34}) {
        LoweringFixture t;
        t.set_sgpr(30, 1);
        t.set_sgpr(31, 0x7ff00000u);
        t.set_sgpr(34, 0x3ff);
        if (local >= 0) t.rs.lane_local_scalars.insert(t.rs.sreg.at(local));
        const auto in = LoweringFixture::decode({0xd4a8006au, 30u | (34u << 9)});
        ASSERT_TRUE(lower_vopc(t.b, t.rs, in, t.ok, false, true));
        if (local < 0) {
            EXPECT_TRUE(t.ok);
            EXPECT_NE(t.rs.vcc, 0u);
            EXPECT_EQ(t.rs.vcc_wave_uniform, t.rs.vcc);
        } else {
            EXPECT_FALSE(t.ok);
            EXPECT_EQ(t.b.stage_reject_reason, local == 34 ? "class-f64-unproved-class-mask"
                                                           : "class-f64-unproved-scalar-pair");
            EXPECT_EQ(t.rs.vcc_wave_uniform, 0u);
        }
    }
}
TEST(IrVectorCompareClassF64Execution, SubdwordAndAmbiguousMaskFormsRefuseByName) {
    for (int shape : {0, 1, 2, 3}) {
        LoweringFixture t;
        t.rs.vreg[kVsrc0] = t.b.uconst(0);
        t.rs.vreg[kVsrc1] = t.b.uconst(0x3ff00000u);
        t.rs.vreg[kVsrc2] = t.b.uconst(256);
        auto in = LoweringFixture::decode(compare_program(0xa8, Form::Vop3Vcc, 0, 0, kVsrc2));
        if (shape == 0) in.has_sdwa = true;
        if (shape == 1) in.has_dpp = true;
        if (shape == 2) in.src[1] = {OperandKind::InlineFloat, 242};
        if (shape == 3) {
            in.src[1] = {OperandKind::Special, 253};
            t.rs.scc = t.b.btrue();
        }
        ASSERT_TRUE(lower_vopc(t.b, t.rs, in, t.ok, false, true));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.b.stage_reject_reason,
                  shape < 2 ? "class-f64-sdwa-dpp" : "class-f64-class-mask-unsupported");
    }
    check(program(Form::Vop3Vcc, 64), data(1));
}
TEST(IrVectorCompareClassF64Execution, EffectsWidthsAgreeAndNoFloat64InstructionsAreDeclared) {
    for (uint32_t op : {0xa8u, 0xb8u}) {
        const auto& effect = rdna2_instruction_effects(Rdna2Format::VOPC, op);
        EXPECT_EQ(effect.ssrc_dwords[0], 2u);
        EXPECT_EQ(effect.ssrc_dwords[1], 1u);
    }
    reset_float64_support_for_test();
    const auto lp = lane_program(compare_program(0xa8, Form::VopcVcc, 0, 0, kVsrc2), 64, 64, true);
    const auto spv = recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config);
    ASSERT_GT(spv.size(), 5u);
    for (size_t at = 5; at < spv.size();) {
        const uint32_t opcode = spv[at] & 0xffffu, count = spv[at] >> 16;
        ASSERT_GT(count, 0u);
        ASSERT_LE(at + count, spv.size());
        if (opcode == 17 && count == 2) EXPECT_NE(spv[at + 1], 10u); // Capability Float64
        if (opcode == 22 && count == 3) EXPECT_NE(spv[at + 2], 64u); // OpTypeFloat
        at += count;
    }
    check(program(Form::VopcVcc, 64), data(1));
}
TEST(IrVectorCompareClassF64Execution, ExportsEveryAdmittedClassForm) {
    namespace fs = std::filesystem;
    const char* name = "vopc-class-f64-witness";
    // NOLINTNEXTLINE(concurrency-mt-unsafe): the launch environment is immutable.
    if (!std::getenv("PROSPER_IR_WITNESS_DIR")) {
        std::error_code error;
        fs::remove_all(compare_witness_root(name), error);
        ASSERT_FALSE(error);
    }
    ASSERT_FALSE(fs::exists(compare_witness_root(name)));
    const auto dump = compare_witness_dump(name);
    ASSERT_TRUE(fs::create_directories(dump));
    std::size_t index = 0;
    for (const auto& p : witnesses()) {
        const auto lp = lane_program(p.words, 64, p.wave, true);
        auto config = lp.config;
        config.compute_pgm_rsrc1 = p.mode;
        auto saved = lp;
        saved.config = config;
        write_compare_witness_program(dump, 0x5d54050000ull + uint64_t(index) * 0x100u, index,
                                      saved);
        check(p, data(0));
        ++index;
    }
    EXPECT_EQ(index, 200u);
}
} // namespace
