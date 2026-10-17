// Ordinary scalar shift sources are uniform within a guest wave. Numeric core tests separately
// exercise differing raw words per invocation; these programs test genuine S_MOV producers.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/lower/scalar_alu_lowering.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/integer64_shift_witness.hpp"
#include "gpu/recompiler/ir/scalar_encoding.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdlib>
#include <system_error>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
constexpr uint64_t kWord = 0x8000000100000003ull;
std::vector<uint32_t> words(uint32_t op, uint64_t word, uint32_t count, uint32_t form,
                            uint32_t observation, uint32_t destination = 20) {
    std::vector<uint32_t> out = {0xbeb203ff, uint32_t(word), 0xbeb303ff, uint32_t(word >> 32)};
    const uint32_t count_source = form == 0 ? 52 : form == 1 ? 128 + count : 255;
    if (form == 0) out.insert(out.end(), {0xbeb403ff, count});
    out.push_back(sop2(op, destination, 50, count_source));
    if (form == 2) out.push_back(count);
    if (observation == 2) {
        out.push_back(sop2(0x0a, 54, 129, 128)); // s_cselect_b32 s54,1,0 observes SCC
        out.push_back(vop1(1, 1, 54));
    } else
        out.push_back(vop1(1, 1, destination + observation));
    return out;
}
std::vector<uint32_t> constant_source_words(uint32_t op, uint32_t source_form,
                                            uint32_t observation) {
    auto out = words(op, kWord, 33, 1, observation);
    const uint32_t selector = source_form == 0 ? 129 : source_form == 1 ? 193 : 255;
    out[4] = (out[4] & ~255u) | selector;
    if (source_form == 2) out.insert(out.begin() + 5, 0x89abcdef);
    return out;
}
uint64_t constant_source(uint32_t form) {
    return form == 0 ? uint64_t{1} : form == 1 ? UINT64_MAX : uint64_t{0x89abcdef};
}
uint64_t expected(uint32_t op, uint64_t word, uint32_t count) {
    const uint32_t n = count & 63;
    if (op == 0x1f) return word << n;
    if (op == 0x23 && n && (word >> 63)) return (word >> n) | (UINT64_MAX << (64 - n));
    return word >> n;
}
void check(uint32_t op, uint64_t word, uint32_t count, uint32_t form, uint32_t wave,
           uint32_t observation, uint32_t destination = 20) {
    const auto in = lane_inputs(173);
    const auto run =
        run_vector_op_per_lane(words(op, word, count, form, observation, destination), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    const uint64_t result = expected(op, word, count);
    const uint32_t want = observation == 2 ? result != 0 : uint32_t(result >> (observation * 32));
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane])
            << "op " << op << " lane " << lane;
}
Rdna2Inst decoded(uint32_t op, uint32_t destination = 20, uint32_t source = 50,
                  uint32_t count = 52) {
    return LoweringFixture::decode({sop2(op, destination, source, count)});
}
std::vector<uint32_t> mask_program(const std::vector<uint32_t>& producer, uint32_t source = 4,
                                   uint32_t count = 129) {
    std::vector<uint32_t> out = {
        0x7e0a0280,
        0x7e000505,   // actual v_readfirstlane keeps a uniform scalar branch unknown to the folder
        0xd4c20002, 0x00010080};
    out.insert(out.end(), producer.begin(), producer.end());
    out.push_back(sop2(0x23, 20, source, count));
    out.push_back(0xbeea0414);   // s_mov_b64 vcc,s[20:21], unchanged mask consumer/guard
    out.insert(out.end(), {0xd5010001, 0x01a90280});   // read VCC through CNDMASK into v1
    return out;
}
}   // namespace
TEST(IrScalarInteger64Shift, UniformProducersObserveBothHalvesAndSccUnderPartialExec) {
    for (uint32_t op : {0x1fu, 0x21u, 0x23u})
        for (uint64_t word :
             {uint64_t{0}, uint64_t{1}, kWord, UINT64_MAX, uint64_t{0x7fffffffffffffffull}})
            for (uint32_t count : {0u, 1u, 31u, 32u, 33u, 63u, 64u, UINT32_MAX})
                for (uint32_t wave : {32u, 64u})
                    for (uint32_t observation : {0u, 1u, 2u})
                        check(op, word, count, 0, wave, observation);
}
TEST(IrScalarInteger64Shift, InlineLiteralCountsAndAliasingDestinationsReadOldWords) {
    for (uint32_t op : {0x1fu, 0x21u, 0x23u})
        for (uint32_t form : {1u, 2u})
            for (uint32_t wave : {32u, 64u})
                for (uint32_t observation : {0u, 1u, 2u})
                    check(op, kWord, 33, form, wave, observation, 50);
}
TEST(IrScalarInteger64Shift, ExistingSourceConstantsKeepSignAndLiteralZeroExtension) {
    for (uint32_t op : {0x1fu, 0x21u})
        for (uint32_t form : {0u, 1u, 2u})
            for (uint32_t wave : {32u, 64u})
                for (uint32_t observation : {0u, 1u, 2u}) {
                    const auto in = lane_inputs(173);
                    const auto run = run_vector_op_per_lane(
                        constant_source_words(op, form, observation), in, wave);
                    ASSERT_TRUE(run.error.empty()) << run.error;
                    ASSERT_EQ(run.d.size(), in.lanes());
                    const uint64_t result = expected(op, constant_source(form), 33);
                    const uint32_t want =
                        observation == 2 ? result != 0 : uint32_t(result >> (observation * 32));
                    for (size_t lane = 0; lane < in.lanes(); ++lane)
                        EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
                }
}
TEST(IrScalarInteger64Shift, ExistingPairServiceRetainsSpecialWordsAndInputFallback) {
    for (uint32_t op : {0x1fu, 0x21u})
        for (int source : {50, 106, 108, 122})
            for (bool input : {false, true}) {
                LoweringFixture t;
                auto& map = input ? t.rs.sreg_input : t.rs.sreg;
                map[source] = t.b.uconst(3);
                map[source + 1] = t.b.uconst(0x80000001);
                t.set_sgpr(52, 33);
                const auto in = decoded(op, 20, source);
                const auto old_scc = t.rs.scc;
                ASSERT_TRUE(ir::lower_sop2(t.b, t.rs, in, t.ok, false));
                ASSERT_TRUE(t.ok) << source << " input " << input;
                EXPECT_TRUE(t.rs.sreg.contains(20));
                EXPECT_TRUE(t.rs.sreg.contains(21));
                EXPECT_NE(t.rs.scc, old_scc);
            }
}
TEST(IrScalarInteger64Shift, ArithmeticRightKeepsSourceConstantsOutsideItsNewAdmission) {
    for (const auto source :
         {Operand{OperandKind::InlineInt, 1}, Operand{OperandKind::InlineInt, -1},
          Operand{OperandKind::Literal, 0}}) {
        LoweringFixture t;
        t.set_sgpr(52, 33);
        auto in = decoded(0x23);
        in.src[0] = source;
        in.has_literal = true;
        in.literal = 0x89abcdef;
        const auto old = t.rs.sreg;
        const auto before = t.code_words();
        EXPECT_TRUE(ir::lower_sop2(t.b, t.rs, in, t.ok, false));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.rs.sreg, old);
        EXPECT_EQ(t.code_words(), before);
    }
}
TEST(IrScalarInteger64Shift, CountDestinationAliasAndEarlierRefusalKeepTheirBoundaries) {
    for (uint32_t op : {0x1fu, 0x21u, 0x23u}) {
        for (uint32_t wave : {32u, 64u})
            for (uint32_t observation : {0u, 1u, 2u})
                check(op, kWord, 33, 0, wave, observation, 52);
        LoweringFixture t;
        t.set_sgpr(50, 3);
        t.set_sgpr(51, 0x80000001);
        t.set_sgpr(52, 33);
        t.set_sgpr(20, 0xaabbccdd);
        t.set_sgpr(21, 0x11223344);
        const auto old = t.rs.sreg;
        const auto scc = t.rs.scc;
        t.ok = false;
        ASSERT_TRUE(ir::lower_sop2(t.b, t.rs, decoded(op), t.ok, false));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.rs.sreg, old);
        EXPECT_EQ(t.rs.scc, scc);
    }
}
TEST(IrScalarInteger64Shift, ArithmeticRightRefusesNonordinaryAndMisalignedShapesBeforeReading) {
    for (uint32_t arm = 0; arm < 9; ++arm) {
        LoweringFixture t;
        t.set_sgpr(50, 1);
        t.set_sgpr(51, 0x80000000);
        t.set_sgpr(52, 33);
        auto in = decoded(0x23);
        if (arm == 0) in.dst = {OperandKind::Special, 126};
        if (arm == 1) in.dst = {OperandKind::Special, 106};
        if (arm == 2) in.dst.value = 21;
        if (arm == 3) in.dst.value = 105;
        if (arm == 4) in.src[0].value = 51;
        if (arm == 5) in.src[0] = {OperandKind::Special, 106};
        if (arm == 6) in.src[1] = {OperandKind::Special, 124};
        if (arm == 7) in.src[1] = {OperandKind::InlineFloat, 0};
        if (arm == 8) in.src[1] = {OperandKind::Literal, 0};
        const auto old = t.rs.sreg;
        const auto before = t.code_words();
        EXPECT_TRUE(ir::lower_sop2(t.b, t.rs, in, t.ok, false));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.rs.sreg, old);
        EXPECT_EQ(t.code_words(), before);
    }
}
TEST(IrScalarInteger64Shift, ExistingCentralMarksKeepEveryFabricatedSourceWordHard) {
    for (uint32_t source : {50u, 51u, 52u}) {
        LoweringFixture t;
        t.set_sgpr(50, 1);
        t.set_sgpr(51, 0x80000000);
        t.set_sgpr(52, 33);
        t.rs.scalar_presence_has_no_placeholders = false;
        t.rs.sreg_merge_placeholder.insert(source);
        const auto in = decoded(0x23);
        const auto before = snapshot_saved_b64_masks(t.b, t.rs, in);
        ASSERT_TRUE(ir::lower_sop2(t.b, t.rs, in, t.ok, false));
        ASSERT_TRUE(t.ok);
        record_scalar_write(t.rs, in, false, before);
        EXPECT_TRUE(t.rs.sreg_merge_placeholder.contains(20));
        EXPECT_TRUE(t.rs.sreg_merge_placeholder.contains(21));
        EXPECT_TRUE(t.rs.scc_merge_placeholder);
    }
}
TEST(IrScalarInteger64Shift, ActualOneArmAndAbsentSourcesCannotAuthenticateAMask) {
    const std::vector<uint32_t> defined = {0xbe8403c1, 0xbe8503c1};
    const std::vector<uint32_t> one_arm = {0xbf068000, 0xbf850002, 0xbe8403c1, 0xbe8503c1};
    const std::vector<uint32_t> both_arms = {0xbf068000, 0xbf850003, 0xbe8403c1, 0xbe8503c1,
                                             0xbf820002, 0xbe840380, 0xbe8503c1};
    const auto compile = [&](const std::vector<uint32_t>& producer) {
        auto lp = lane_program(mask_program(producer), 64, 64);
        // This is a whole-wave scalar provenance control. Remove the vector harness's
        // initial CMPX: making its prefix run under partial EXEC would mark v5/s0 too.
        lp.code.erase(lp.code.begin() + 9);
        lp.config.native_subgroup_size = 64;
        return recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config);
    };
    for (const auto& producer : {defined, both_arms}) EXPECT_FALSE(compile(producer).empty());
    for (const auto& producer : {std::vector<uint32_t>{}, one_arm})
        EXPECT_TRUE(compile(producer).empty());
}
TEST(IrScalarInteger64Shift, ExportsUniformScalarFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    const auto root = compare_witness_root("scalar-integer64-shift-witness");
    // NOLINTNEXTLINE(concurrency-mt-unsafe): immutable launch environment.
    if (!std::getenv("PROSPER_IR_WITNESS_DIR")) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error);
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directories(dump));
    size_t count = 0;
    for (uint32_t op : {0x1fu, 0x21u, 0x23u})
        for (uint32_t wave : {32u, 64u})
            for (uint32_t form : {0u, 1u, 2u})
                for (uint32_t observation : {0u, 1u, 2u}) {
                    const auto lp = lane_program(words(op, kWord, 33, form, observation), 64, wave);
                    const auto module = saved_integer64_shift_witness(
                        dump, 0x6460000000ull + count * 0x100, count, lp);
                    const auto in = lane_inputs(173);
                    const auto run = execute_integer64_shift_module(module, in);
                    ASSERT_TRUE(run.error.empty()) << run.error;
                    ASSERT_EQ(run.d.size(), in.lanes());
                    const uint64_t result = expected(op, kWord, 33);
                    const uint32_t want =
                        observation == 2 ? result != 0 : uint32_t(result >> (observation * 32));
                    for (size_t lane = 0; lane < in.lanes(); ++lane)
                        EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
                    ++count;
                }
    for (uint32_t op : {0x1fu, 0x21u})
        for (uint32_t wave : {32u, 64u})
            for (uint32_t form : {0u, 1u, 2u})
                for (uint32_t observation : {0u, 1u, 2u}) {
                    const auto lp =
                        lane_program(constant_source_words(op, form, observation), 64, wave);
                    const auto module = saved_integer64_shift_witness(
                        dump, 0x6460000000ull + count * 0x100, count, lp);
                    const auto in = lane_inputs(173);
                    const auto run = execute_integer64_shift_module(module, in);
                    ASSERT_TRUE(run.error.empty()) << run.error;
                    ASSERT_EQ(run.d.size(), in.lanes());
                    const uint64_t result = expected(op, constant_source(form), 33);
                    const uint32_t want =
                        observation == 2 ? result != 0 : uint32_t(result >> (observation * 32));
                    for (size_t lane = 0; lane < in.lanes(); ++lane)
                        EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
                    ++count;
                }
    EXPECT_EQ(count, 90u);
}
