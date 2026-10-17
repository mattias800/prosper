// #5225: non-fused F32 MAD/MAC
// must keep a rounded product. Check the actual arithmetic decoration targets and guest lanes.
#include "fixtures/f32_mad_rounding.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/ir/ops/vector_mad_float.hpp"
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
using namespace prosper::test::mad_rounding;
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
        ASSERT_EQ(out[lane], expected(e, s, in, lane, rsrc1)) << e.name << " lane " << lane;
}
std::vector<Shape> shapes(const Encoding& e) {
    std::vector<Shape> out;
    for (uint32_t form = 0; form <= 5; ++form) {
        Shape s{form};
        if (supports(e, s)) out.push_back(s);
    }
    if (e.format == Rdna2Format::VOP3) {
        for (uint32_t slot = 0; slot < (e.addend == Addend::Destination ? 2u : 3u); ++slot) {
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
TEST(IrFloatMadRounding, HandDerivedCancellationRequiresTheSeparatelyRoundedProduct) {
    // (1+2^-23)*(1-2^-23)-1 = -2^-46 exactly; the separately rounded product is 1.
    const auto product = evaluate_op(Opcode::F32MulNoContraction, 0, 0x3f800001, 0x3f7ffffe, 0);
    if (!product) {
        ADD_FAILURE() << "the finite product must be known";
        return;
    }
    ASSERT_EQ(product, Known(0x3f800000));
    EXPECT_EQ(evaluate_op(Opcode::F32AddNoContraction, 0, *product, 0xbf800000, 0), Known(0));
}
TEST(IrFloatMadRounding, NoContractionTargetsTheFloatingOperationsBeforeBitcasts) {
    // Exercise both implicit profiles and explicit bit-preserving transport; NoContraction
    // must not migrate onto transport Bitcasts or acquire conflicting fast-math metadata.
    for (const auto profile : {FloatTransportProfile::Unknown, FloatTransportProfile::Implicit,
                               FloatTransportProfile::ExplicitNonFinite32})
        for (const auto& e : kEncodings) {
            SCOPED_TRACE(e.name);
            const auto module = compile(e, {}, 64, 0, profile);
            ASSERT_FALSE(module.empty());
        // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
            if (const char* configured = std::getenv("PROSPER_F32_MAD_PROFILE_ARTIFACTS")) {
                const std::filesystem::path root(configured);
                std::filesystem::create_directories(root);
                std::ofstream output(root /
                                         (std::string(e.name) + "_profile" +
                                          std::to_string(static_cast<uint32_t>(profile)) + ".spv"),
                                     std::ios::binary);
                output.write(reinterpret_cast<const char*>(module.data()),
                             static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
                output.close();
                ASSERT_TRUE(output.good());
            }
            struct Def {
                uint32_t op, type, ext;
            };
            std::map<uint32_t, Def> definitions;
            std::set<uint32_t> constrained, fast_math;
            size_t mul = 0, add = 0, fma = 0;
            for (size_t pc = 5; pc < module.size(); pc += module[pc] >> 16) {
                const uint32_t count = module[pc] >> 16, op = module[pc] & 0xffffu;
                ASSERT_GT(count, 0u);
                ASSERT_LE(pc + count, module.size());
                const auto* a = module.data() + pc + 1;
                if (op == Op_Decorate && count >= 3) {
                    if (a[1] == Dec_NoContraction) constrained.insert(a[0]);
                    if (a[1] == Dec_FPFastMathMode) fast_math.insert(a[0]);
                }
                // FPFastMathDefault cannot coexist with NoContraction in this entry point.
                if (op == Op_ExecutionMode || op == 331u) EXPECT_NE(a[1], 6028u);
                if (op == Op_FMul || op == Op_FAdd || op == Op_Bitcast || op == Op_ExtInst) {
                    definitions[a[1]] = {op, a[0], op == Op_ExtInst ? a[3] : 0u};
                    mul += op == Op_FMul;
                    add += op == Op_FAdd;
                    fma += op == Op_ExtInst && a[3] == 50u;
                }
            }
            EXPECT_EQ(mul, 1u);
            EXPECT_EQ(add, 1u);
            EXPECT_EQ(fma, 0u);
            ASSERT_EQ(constrained.size(), 2u);
            EXPECT_EQ(fast_math.empty(), profile != FloatTransportProfile::ExplicitNonFinite32);
            for (uint32_t target : constrained) {
                ASSERT_TRUE(definitions.contains(target));
                const auto& def = definitions.at(target);
                EXPECT_TRUE(def.op == Op_FMul || def.op == Op_FAdd);
                EXPECT_FALSE(fast_math.contains(target));
                bool f32 = false;
                for (size_t pc = 5; pc < module.size(); pc += module[pc] >> 16)
                    if ((module[pc] & 0xffffu) == Op_TypeFloat && module[pc + 1] == def.type)
                        f32 = module[pc + 2] == 32;
                EXPECT_TRUE(f32) << "NoContraction must decorate the arithmetic F32 result";
            }
        }
}
TEST(IrFloatMadRounding, FiveExistingEncodingsKeepTheFiniteWitnessAndInactiveDestinations) {
    for (const auto& e : kEncodings) {
        const auto in = inputs(&e);
        EXPECT_FALSE(in.active[0]);
        EXPECT_TRUE(in.active[4]);
        EXPECT_EQ(expected(e, {}, in, 4, 0), 0u);
        for (uint32_t wave : {32u, 64u}) check(e, {}, in, 0, compile(e, {}, wave));
    }
}
TEST(IrFloatMadRounding, OperandFormsAndCallerModifiersSurviveExplicitRteLaunchModes) {
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
TEST(IrFloatMadRounding, ExistingZeroTieOracleBoundaryRemainsVisibleBesideNonzeroClampControl) {
    for (const auto& e : kEncodings) {
        if (e.format != Rdna2Format::VOP3) continue;
        Shape shape;
        shape.clamp = true;
        const auto module = compile(e, shape);
        ASSERT_FALSE(module.empty());
        std::string error;
        EXPECT_TRUE(execute(module, inputs(&e), error).empty());
        EXPECT_EQ(error, "unspecified float ALU result");
        check(e, shape, inputs(e, shape), 0, module);
    }
}
TEST(IrFloatMadRounding, ProductPrecedesTheThirdSourceAndMissingInputsStayUnknown) {
    prosper::test::ir::LoweringFixture fixture;
    std::vector<uint32_t> reads;
    std::array<size_t, 3> read_at{};
    const MadFloatDef nonfused{Rdna2Format::VOP3, 0x141, build_nonfused_mad_float};
    const auto result = emit_mad_float_op(fixture.b, nonfused, [&](uint32_t slot) {
        reads.push_back(slot);
        read_at[slot] = fixture.code_words();
        return fixture.b.uconst(slot == 0 ? 0x3f800001 : slot == 1 ? 0x3f7ffffe : 0xbf800000);
    });
    ASSERT_EQ(reads.size(), 3u);
    EXPECT_EQ((std::set<uint32_t>(reads.begin(), reads.end())), (std::set<uint32_t>{0, 1, 2}));
    EXPECT_GT(read_at[2], read_at[0]);
    EXPECT_GT(read_at[2], read_at[1]);
    EXPECT_NE(result, 0u);
    for (size_t missing = 0; missing < 3; ++missing) {
        std::array<Known, 3> values = {0x3f800001, 0x3f7ffffe, 0xbf800000};
        values[missing] = std::nullopt;
        EXPECT_FALSE(eval_mad_float_op(nonfused, values));
    }
    EXPECT_FALSE(eval_mad_float_op(nonfused, {0x7fc00001u, 0x3f800000u, 0xbf800000u}));
    EXPECT_EQ(find_vector_mad_float_op(Rdna2Format::VOP3, 0x14b), nullptr);
    EXPECT_EQ(find_vector_mad_float_op(Rdna2Format::VOP3, 0x34b), nullptr);
}
TEST(IrFloatMadRounding, ExportsAndExecutesTheExactReplayedWitnessModules) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "f32-mad-witness";
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
                    const uint64_t address = 0x5225000000ull + count * 0x100;
                    const auto stem = "cs_" + std::to_string(address) + "_" + std::to_string(count);
                    std::ofstream bin(dump / (stem + ".bin"), std::ios::binary);
                    bin.write(reinterpret_cast<const char*>(p.code.data()),
                              static_cast<std::streamsize>(p.code.size() * sizeof(uint32_t)));
                    bin.close();
                    ASSERT_TRUE(bin.good());
                    const auto file = dump / (stem + ".inputs");
                    ASSERT_TRUE(write_refused_compute_inputs(file, address, &p.table, p.config,
                                                             "synthetic-f32-mad-witness"));
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
    EXPECT_EQ(count, 424u);
}
