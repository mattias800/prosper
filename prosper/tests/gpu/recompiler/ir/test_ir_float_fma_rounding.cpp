// #5225: GLSL Fma expresses fused intent. The interpreter chooses an exact finite RTE
// reference; an executing-driver test is separate because Vulkan permits weaker precision.
#include "fixtures/f32_fma_rounding.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/ir/ops/vector_fma_float.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::fma_rounding;
namespace {
std::vector<uint32_t> compile(const Encoding& e, Shape s = {}, uint32_t wave = 64,
                              uint32_t rsrc1 = 0,
                              FloatTransportProfile profile = FloatTransportProfile::Unknown) {
    auto p = program(e, s, wave, rsrc1);
    p.config.float_transport = {profile};
    return recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
}
void check(const Encoding& e, Shape s, const LaneInputs& in, uint32_t rsrc1,
           const std::vector<uint32_t>& module) {
    ASSERT_FALSE(module.empty());
    std::string error;
    const auto out = execute(module, in, error);
    ASSERT_TRUE(error.empty()) << error;
    ASSERT_EQ(out.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        ASSERT_EQ(out[lane], fused_expected(e, s, in, lane, rsrc1)) << e.name << " lane " << lane;
}
std::vector<Shape> shapes(const Encoding& e) {
    std::vector<Shape> out;
    for (uint32_t form = 0; form <= 5; ++form) {
        Shape s{form};
        if (supports(e, s)) out.push_back(s);
    }
    if (e.format == Rdna2Format::VOP3) {
        for (uint32_t slot = 0; slot < 3; ++slot) {
            out.push_back({0, 1u << slot});
            out.push_back({0, 1u << slot, 1u << slot});
            out.push_back({0, 0, 1u << slot});
        }
        for (uint32_t omod = 0; omod < 4; ++omod)
            for (bool clamp : {false, true})
                if (omod || clamp) out.push_back({0, 0, 0, omod, clamp});
    }
    return out;
}
} // namespace
TEST(IrFloatFmaRounding, HandDerivedCancellationAndOverflowAreOneRoundingReferenceBoundaries) {
    // (1+2^-23)*(1-2^-23)-1 = -2^-46 exactly, unlike the separately rounded zero.
    EXPECT_EQ(eval_fused_mad_float({0x3f800001, 0x3f7ffffe, 0xbf800000}), Known(0xa8800000));
    EXPECT_EQ(evaluate_op(Opcode::F32Fma, 0, 0x3f800001, 0x3f7ffffe, 0xbf800000),
              Known(0xa8800000));
    // max*2-max = max exactly. The old evaluator knows +Inf after separate overflow;
    // this is a graph distinction, not evidence that the old folding domain was Unknown.
    EXPECT_EQ(eval_fused_mad_float({0x7f7fffff, 0x40000000, 0xff7fffff}), Known(0x7f7fffff));
    EXPECT_EQ(evaluate_op(Opcode::F32Mul, 0, 0x7f7fffff, 0x40000000, 0), Known(0x7f800000));
}
TEST(IrFloatFmaRounding, MissingNonfiniteAndSubnormalValuesStayOutsideTheReferenceDomain) {
    for (size_t missing = 0; missing < 3; ++missing) {
        std::array<Known, 3> values = {0x3f800001, 0x3f7ffffe, 0xbf800000};
        values[missing] = std::nullopt;
        EXPECT_FALSE(eval_fused_mad_float(values));
    }
    for (uint32_t outside : {0x7fc00001u, 0x7f800000u, 1u})
        EXPECT_FALSE(eval_fused_mad_float({outside, 0x3f800000, 0}));
    EXPECT_FALSE(eval_fused_mad_float({0x00800000, 0x3f000000, 0}));
    EXPECT_FALSE(eval_fused_mad_float({0x7f7fffff, 0x40000000, 0}));
}
TEST(IrFloatFmaRounding, MultiplicandsPrecedeLazyThirdSourceWithoutInventingTheirRelativeOrder) {
    prosper::test::ir::LoweringFixture fixture;
    std::vector<uint32_t> reads;
    const auto result = emit_fused_mad_float(fixture.b, [&](uint32_t slot) {
        reads.push_back(slot);
        return fixture.b.uconst(slot == 0 ? 0x3f800001 : slot == 1 ? 0x3f7ffffe : 0xbf800000);
    });
    EXPECT_NE(result, 0u);
    ASSERT_EQ(reads.size(), 3u);
    EXPECT_EQ((std::set<uint32_t>(reads.begin(), reads.begin() + 2)), (std::set<uint32_t>{0, 1}));
    EXPECT_EQ(reads.back(), 2u);
}
TEST(IrFloatFmaRounding, FourExistingGuestEncodingsUseScalarGlslFmaInEveryTransportProfile) {
    for (const auto profile : {FloatTransportProfile::Unknown, FloatTransportProfile::Implicit,
                               FloatTransportProfile::ExplicitNonFinite32})
        for (const auto& e : kEncodings) {
            const auto module = compile(e, {}, 64, 0, profile);
            ASSERT_FALSE(module.empty());
            std::map<uint32_t, uint32_t> float_widths;
            size_t fma = 0, separate = 0;
            for (size_t pc = 5; pc < module.size(); pc += module[pc] >> 16) {
                const uint32_t count = module[pc] >> 16, op = module[pc] & 0xffffu;
                ASSERT_GT(count, 0u);
                ASSERT_LE(pc + count, module.size());
                const auto* a = module.data() + pc + 1;
                if (op == Op_TypeFloat) float_widths[a[0]] = a[1];
                if (op == Op_ExtInst && count >= 5 && a[3] == Glsl_Fma) {
                    ASSERT_EQ(count, 8u);
                    EXPECT_EQ(float_widths.at(a[0]), 32u);
                    ++fma;
                }
                separate += op == Op_FMul || op == Op_FAdd;
            }
            EXPECT_EQ(fma, 1u) << e.name;
            EXPECT_EQ(separate, 0u) << e.name;
        }
}
TEST(IrFloatFmaRounding, FourFusedEncodingsKeepHandAnswersAndInactiveDestinationsInBothWaves) {
    for (const auto& e : kEncodings) {
        const auto in = inputs(&e);
        EXPECT_FALSE(in.active[0]);
        EXPECT_TRUE(in.active[4]);
        EXPECT_EQ(fused_expected(e, {}, in, 4, 0), 0xa8800000u);
        for (uint32_t wave : {32u, 64u}) check(e, {}, in, 0, compile(e, {}, wave));
    }
}
TEST(IrFloatFmaRounding, OperandFormsAndCallerModifiersSurviveTheSelectedRteLaunchProfiles) {
    for (const auto& e : kEncodings)
        for (const auto s : shapes(e))
            for (uint32_t rsrc1 : {0u, 0x20000u, 0x30000u, 0x800000u})
                for (uint32_t wave : {32u, 64u}) {
                    SCOPED_TRACE(std::string(e.name) + " form " + std::to_string(s.form) + " abs " +
                                 std::to_string(s.abs) + " neg " + std::to_string(s.neg) +
                                 " omod " + std::to_string(s.omod) + " mode " +
                                 std::to_string(rsrc1));
                    check(e, s, inputs(e, s), rsrc1, compile(e, s, wave, rsrc1));
                }
}
TEST(IrFloatFmaRounding, UnsupportedLiveReferenceInputsFailVisiblyInsteadOfGuessing) {
    const auto module = compile(kEncodings.back());
    for (uint32_t outside : {0x7fc00001u, 0x7f800000u, 1u}) {
        auto in = inputs(&kEncodings.back());
        in.s0[4] = outside;
        std::string error;
        EXPECT_TRUE(execute(module, in, error).empty());
        EXPECT_EQ(error, "outside finite normal-or-zero Fma reference");
    }
}
TEST(IrFloatFmaRounding, ExportsAndExecutesTheExactReplayedWitnessModules) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "f32-fma-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root));
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream manifest(root / "manifest.tsv");
    manifest << "stem\tencoding\twave\tform\tabs\tneg\tomod\tclamp\trsrc1\tnonzero_clamp_input\n";
    size_t count = 0;
    for (const auto& e : kEncodings)
        for (const auto s : shapes(e))
            for (uint32_t rsrc1 : {0u, 0x20000u, 0x30000u, 0x800000u})
                for (uint32_t wave : {32u, 64u}) {
                    auto p = program(e, s, wave, rsrc1);
                    const uint64_t address = 0x5225f00000ull + count * 0x100;
                    const auto stem = "cs_" + std::to_string(address) + "_" + std::to_string(count);
                    std::ofstream bin(dump / (stem + ".bin"), std::ios::binary);
                    bin.write(reinterpret_cast<const char*>(p.code.data()),
                              static_cast<std::streamsize>(p.code.size() * sizeof(uint32_t)));
                    bin.close();
                    ASSERT_TRUE(bin.good());
                    const auto file = dump / (stem + ".inputs");
                    ASSERT_TRUE(write_refused_compute_inputs(file, address, &p.table, p.config,
                                                             "synthetic-f32-fma-witness"));
                    const auto read = read_refused_shader_inputs(file);
                    ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
                    const auto replay = replay_refused_shader_inputs(p.code, read);
                    ASSERT_TRUE(replay.produced) << replay.reject;
                    ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
                    std::ofstream spv(root / (stem + ".spv"), std::ios::binary);
                    spv.write(reinterpret_cast<const char*>(replay.spirv.data()),
                              static_cast<std::streamsize>(replay.spirv.size() * sizeof(uint32_t)));
                    spv.close();
                    ASSERT_TRUE(spv.good());
                    std::ifstream saved(root / (stem + ".spv"), std::ios::binary);
                    std::vector<uint32_t> loaded(replay.spirv.size());
                    saved.read(reinterpret_cast<char*>(loaded.data()),
                               static_cast<std::streamsize>(loaded.size() * sizeof(uint32_t)));
                    ASSERT_TRUE(saved.good());
                    ASSERT_EQ(loaded, replay.spirv);
                    check(e, s, inputs(e, s), rsrc1, loaded);
                    manifest << stem << '\t' << e.name << '\t' << wave << '\t' << s.form << '\t'
                             << s.abs << '\t' << s.neg << '\t' << s.omod << '\t' << s.clamp << '\t'
                             << rsrc1 << '\t' << s.clamp << '\n';
                    ++count;
                }
    manifest.close();
    ASSERT_TRUE(manifest.good());
    EXPECT_EQ(count, 280u);
}
