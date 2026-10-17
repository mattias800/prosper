// #5225: actual guest-module half FMA results, source selectors and EXEC publication.
// Runtime ineligible arithmetic keeps the original graph; statically ineligible MODEs
// are also exported for an immutable-BASE byte comparison outside this test.
#include <vector>
#include "fixtures/half_fma_reference.hpp"
#include <ios>
#include "fixtures/half_fma_finite.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
using namespace prosper::test::half_fma;
using namespace prosper::gpu;
namespace {
void check(Kind kind, Shape shape, uint32_t wave, uint32_t offset = 0, uint32_t rsrc1 = 0xc0000) {
    SCOPED_TRACE(std::string(name(kind)) + " wave=" + std::to_string(wave) +
                 " selectors=" + std::to_string(shape.low) + "/" + std::to_string(shape.high) +
                 " form=" + std::to_string(shape.form));
    const auto in = inputs(kind, shape, offset);
    const auto old = compile(kind, shape, wave, legacy_mode(rsrc1));
    const auto module = compile(kind, shape, wave, rsrc1);
    ASSERT_FALSE(old.empty());
    ASSERT_FALSE(module.empty());
    std::string error;
    const auto fallback = execute(old, in, error);
    ASSERT_TRUE(error.empty()) << error;
    ASSERT_EQ(fallback.size(), 64u);
    const auto out = execute(module, in, error);
    ASSERT_TRUE(error.empty()) << error;
    ASSERT_EQ(out.size(), 64u);
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        EXPECT_EQ(out[lane], profile(rsrc1, shape) ? expected(kind, shape, in, lane, fallback[lane])
                                                   : fallback[lane])
            << "lane=" << lane;
}
std::vector<Shape> shapes(Kind kind) {
    std::vector<Shape> out = {{}};
    for (uint32_t form = 1; form <= (kind == Kind::PackedMac ? 5u : 6u); ++form) {
        Shape shape;
        shape.form = form;
        out.push_back(shape);
    }
    if (kind != Kind::PackedMac)
        for (uint32_t k = 1; k <= 7; ++k) {
            Shape shape;
            shape.low = k | (kind == Kind::Native ? (k & 1) << 3 : 0);
            shape.high = 7 - k;
            shape.neg_low = k;
            shape.neg_high = 7 - k;
            if (kind == Kind::Native) shape.absolute = (k + 1) & 7;
            out.push_back(shape);
        }
    if (kind == Kind::Native)
        for (uint32_t omod : {1u, 2u, 3u}) {
            Shape s;
            s.omod = omod;
            out.push_back(s);
        }
    if (kind != Kind::PackedMac) {
        Shape s;
        s.clamp = true;
        out.push_back(s);
    }
    return out;
}
}   // namespace
TEST(IrHalfFmaFiniteExecution, BothKnownWitnessesChangeTheActualGuestResult) {
    EXPECT_EQ(reference(0xb838, 0x0b96, 0x36dc).bits, 0x36db);
    EXPECT_EQ(reference(0x3c01, 0x3e00, 0x8001).bits, 0x3e01);
    for (uint32_t wave : {32u, 64u}) {
        for (Kind kind : {Kind::Packed, Kind::PackedMac}) {
            std::string error;
            const auto old = execute(compile(kind, {}, wave, 0), inputs(kind), error);
            ASSERT_TRUE(error.empty()) << error;
            const auto out = execute(compile(kind, {}, wave), inputs(kind), error);
            ASSERT_TRUE(error.empty()) << error;
            ASSERT_EQ(old.size(), 64u);
            ASSERT_EQ(out.size(), 64u);
            EXPECT_EQ(out[4], 0x3e0136dbu);
            EXPECT_EQ(old[4], 0x3e0236dcu);
            EXPECT_EQ(out[0], inputs(kind).old_d[0]);
        }
        Shape s;
        s.low = 7;
        std::string error;
        const auto out = execute(compile(Kind::Native, s, wave), inputs(Kind::Native), error);
        ASSERT_TRUE(error.empty()) << error;
        ASSERT_EQ(out.size(), 64u);
        EXPECT_EQ(out[4] & 0xffffu, 0x3e01u);
        EXPECT_EQ(out[4] & 0xffff0000u, inputs(Kind::Native).old_d[4] & 0xffff0000u);
    }
}
TEST(IrHalfFmaFiniteExecution, VaryingLanesBothWavesAndInactiveDestinationsUseIndependentAnswers) {
    for (Kind kind : kKinds)
        for (uint32_t wave : {32u, 64u})
            for (uint32_t offset : {0u, 64u}) check(kind, {}, wave, offset);
}
TEST(IrHalfFmaFiniteExecution, NativeSourceSelectorsAndBothDestinationHalvesStayIndependent) {
    for (uint32_t sel = 0; sel < 16; ++sel)
        for (uint32_t wave : {32u, 64u}) {
            Shape s;
            s.low = sel;
            check(Kind::Native, s, wave);
        }
}
TEST(IrHalfFmaFiniteExecution, PackedLowAndHighSelectorsNegationAndSourceAliasesKeepTheirOwners) {
    for (Kind kind : kKinds)
        for (const auto shape : shapes(kind))
            for (uint32_t wave : {32u, 64u}) check(kind, shape, wave);
}
TEST(IrHalfFmaFiniteExecution, F32ModeBitsAreIndependentAndOtherHalfProfilesKeepTheLegacyGraph) {
    for (Kind kind : kKinds)
        for (uint32_t mode :
             {0u, 0x30u, 0x40u, 0x80u, 0xc0u, 0xc3u, 0xf0u, 0xf3u, 0xc4u, 0xc8u, 0xccu})
            check(kind, {}, 64, 0, mode << 12);
    for (Kind kind : kKinds) check(kind, {}, 64, 0, UINT32_MAX);
}
TEST(IrHalfFmaFiniteExecution, PackedRuntimeEligibilityIsSelectedSeparatelyForEachHalf) {
    for (Kind kind : {Kind::Packed, Kind::PackedMac})
        for (bool high_fallback : {false, true}) {
            auto in = inputs(kind);
            const uint32_t shift = high_fallback ? 16 : 0;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                // Exact cancellation is outside the new contract. The other half
                // retains the known double-rounding witness, in the same word.
                in.s0[lane] = (in.s0[lane] & ~(0xffffu << shift)) | (0x3c00u << shift);
                in.s1[lane] = (in.s1[lane] & ~(0xffffu << shift)) | (0x3c00u << shift);
                auto& c = kind == Kind::PackedMac ? in.old_d[lane] : in.s2[lane];
                c = (c & ~(0xffffu << shift)) | (0xbc00u << shift);
            }
            std::string error;
            const auto old = execute(compile(kind, {}, 64, 0), in, error);
            ASSERT_TRUE(error.empty()) << error;
            const auto out = execute(compile(kind, {}, 64), in, error);
            ASSERT_TRUE(error.empty()) << error;
            ASSERT_EQ(out.size(), 64u);
            ASSERT_EQ(old.size(), 64u);
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(out[lane], expected(kind, {}, in, lane, old[lane]));
        }
}
TEST(IrHalfFmaFiniteExecution, ExportsReplayedWitnessesAndExecutesThoseExactModules) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed launch input.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "half-fma-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error);
    }
    ASSERT_FALSE(fs::exists(root));
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream manifest(root / "manifest.tsv");
    manifest << "stem\tkind\twave\tlow\thigh\tabs\tneg_low\tneg_high\tform\tomod\tclamp\trsrc1\n";
    size_t count = 0;
    for (Kind kind : kKinds)
        for (const auto shape : shapes(kind))
            for (uint32_t rsrc1 : {0u, 0xc0000u, 0xf3000u, UINT32_MAX})
                for (uint32_t wave : {32u, 64u}) {
                    if ((shape.clamp || shape.omod) && rsrc1 != 0xc0000u) continue;
                    const auto p = program(kind, shape, wave, rsrc1);
                    const uint64_t address = 0x5225b00000ull + count * 0x100;
                    const auto stem = "cs_" + std::to_string(address) + "_" + std::to_string(count);
                    std::ofstream bin(dump / (stem + ".bin"), std::ios::binary);
                    bin.write(reinterpret_cast<const char*>(p.code.data()), p.code.size() * 4);
                    bin.close();
                    ASSERT_TRUE(bin.good());
                    ASSERT_TRUE(write_refused_compute_inputs(dump / (stem + ".inputs"), address,
                                                             &p.table, p.config,
                                                             "synthetic-half-fma-finite-witness"));
                    const auto read = read_refused_shader_inputs(dump / (stem + ".inputs"));
                    const auto replay = replay_refused_shader_inputs(p.code, read);
                    ASSERT_TRUE(read.incomplete.empty());
                    ASSERT_TRUE(replay.produced) << replay.reject;
                    ASSERT_TRUE(replay.incomplete.empty());
                    std::ofstream spv(root / (stem + ".spv"), std::ios::binary);
                    spv.write(reinterpret_cast<const char*>(replay.spirv.data()),
                              replay.spirv.size() * 4);
                    spv.close();
                    ASSERT_TRUE(spv.good());
                    std::string error;
                    const auto in = inputs(kind, shape);
                    const auto out = execute(replay.spirv, in, error);
                    ASSERT_TRUE(error.empty()) << error;
                    const auto old =
                        execute(compile(kind, shape, wave, legacy_mode(rsrc1)), in, error);
                    ASSERT_TRUE(error.empty()) << error;
                    ASSERT_EQ(out.size(), 64u);
                    ASSERT_EQ(old.size(), 64u);
                    for (size_t lane = 0; lane < 64; ++lane)
                        EXPECT_EQ(out[lane], !profile(rsrc1, shape)
                                                 ? old[lane]
                                                 : expected(kind, shape, in, lane, old[lane]));
                    manifest << stem << '\t' << name(kind) << '\t' << wave << '\t' << shape.low
                             << '\t' << shape.high << '\t' << shape.absolute << '\t'
                             << shape.neg_low << '\t' << shape.neg_high << '\t' << shape.form
                             << '\t' << shape.omod << '\t' << shape.clamp << '\t' << rsrc1 << '\n';
                    ++count;
                }
    manifest.close();
    ASSERT_TRUE(manifest.good());
    EXPECT_EQ(count, 282u);
}
