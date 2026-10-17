// Canonical e64 FLR reuses corrected e32 semantics; finite independent answers,
// both permitted Floor input profiles, and source ownership must survive admission.
#include "gpu/recompiler/ir/cvt_flr_e64_cases.hpp"
#include "gpu/recompiler/ir/ops/vector_float_integer_e64.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <string>
#include <system_error>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::cvt_flr_e64;
namespace {
void check(const std::vector<uint32_t>& module, const LaneInputs& in, cases::Form form = {}) {
    ASSERT_FALSE(module.empty());
    const auto* def = find_vector_cvt_flr_e64_op(Rdna2Format::VOP3, 0x18d);
    ASSERT_NE(def, nullptr);
    for (bool flush : {false, true}) {
        const auto run = cases::execute(module, in, flush);
        ASSERT_TRUE(run.error.empty()) << run.error;
        ASSERT_EQ(run.d.size(), in.lanes());
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            const auto source = cases::source(form, in, lane);
            const auto want = cases::finite_floor(source);
            ASSERT_TRUE(want.has_value()) << source;
            ASSERT_EQ(eval_conversion_op(*def, source), want);
            ASSERT_EQ(run.d[lane], in.active[lane] ? want.value_or(0) : in.old_d[lane])
                << "lane " << lane << " flush " << flush;
        }
    }
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& words, uint32_t wave = 64,
                              uint32_t mode = kDefaultComputePgmRsrc1) {
    auto p = lane_program(words, 64, wave);
    p.config.compute_pgm_rsrc1 = mode;
    const auto address = lane_harness_address();
    auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                    {RecompileDiagnosticStage::Compute, address});
    if (module.empty()) ADD_FAILURE() << last_terminal_reject_reason(address);
    return module;
}
void export_witness(const std::filesystem::path& root, std::ofstream& manifest, size_t index,
                    uint32_t wave, uint32_t form, uint32_t mode, uint32_t encoding) {
    auto p = lane_program(cases::words(cases::form(form), encoding), 64, wave);
    p.config.compute_pgm_rsrc1 = mode;
    const uint64_t address = 0x5355000000ull + index * 0x100;
    const auto stem = "cs_" + std::to_string(address) + "_" + std::to_string(index);
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    const auto binfile = dump / (stem + ".bin");
    std::ofstream bin(binfile, std::ios::binary);
    bin.write(reinterpret_cast<const char*>(p.code.data()),
              static_cast<std::streamsize>(p.code.size() * sizeof(uint32_t)));
    bin.close();
    ASSERT_TRUE(bin.good());
    ASSERT_TRUE(write_refused_compute_inputs(dump / (stem + ".inputs"), address, &p.table, p.config,
                                             "synthetic-cvt-flr-e64"));
    const auto read = read_refused_shader_inputs(dump / (stem + ".inputs"));
    ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
    std::vector<uint32_t> code(p.code.size());
    std::ifstream raw(binfile, std::ios::binary);
    raw.read(reinterpret_cast<char*>(code.data()),
             static_cast<std::streamsize>(code.size() * sizeof(uint32_t)));
    ASSERT_TRUE(raw.good());
    ASSERT_EQ(code, p.code);
    const auto replay = replay_refused_shader_inputs(code, read);
    ASSERT_TRUE(replay.produced) << replay.reject;
    ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
    const auto file = root / "modules" / (stem + ".spv");
    std::ofstream saved(file, std::ios::binary);
    saved.write(reinterpret_cast<const char*>(replay.spirv.data()),
                static_cast<std::streamsize>(replay.spirv.size() * sizeof(uint32_t)));
    saved.close();
    ASSERT_TRUE(saved.good());
    std::vector<uint32_t> loaded(replay.spirv.size());
    std::ifstream input(file, std::ios::binary);
    input.read(reinterpret_cast<char*>(loaded.data()),
               static_cast<std::streamsize>(loaded.size() * sizeof(uint32_t)));
    ASSERT_TRUE(input.good());
    ASSERT_EQ(loaded, replay.spirv);
    check(loaded, cases::inputs(17), cases::form(form));
    manifest << stem << '\t' << wave << '\t' << form << '\t' << mode << '\t' << encoding << '\n';
}
} // namespace
TEST(IrVectorCvtFlrE64Execution, EncodingsAndLookupKeepCanonicalArityAndOldRegistries) {
    EXPECT_EQ(cases::words(), (std::vector<uint32_t>{0xd58d0001, 0x102}));
    EXPECT_EQ(cases::words({}, 1), (std::vector<uint32_t>{0xd18d0001, 0x102}));
    EXPECT_EQ(cases::words({}, 2), (std::vector<uint32_t>{0x7e021b02}));
    EXPECT_EQ(cases::words(cases::form(3)), (std::vector<uint32_t>{0xd58d0001, 0xff, 0x3fe00000}));
    const auto* old = find_vector_conversion_op(Rdna2Format::VOP1, 0x0d);
    ASSERT_NE(old, nullptr);
    const auto* alias = find_vector_cvt_flr_e64_op(Rdna2Format::VOP3, 0x18d);
    ASSERT_NE(alias, nullptr);
    EXPECT_EQ(alias->build, old->build);
    EXPECT_EQ(alias->fmt, Rdna2Format::VOP3);
    EXPECT_EQ(alias->opcode, 0x18du);
    EXPECT_EQ(vector_float_integer_e64_defs().size(), 2u);
    EXPECT_EQ(vector_conversion_defs().size(), 13u);
}
TEST(IrVectorCvtFlrE64Execution, HandDerivedDownwardAnswersPinNegativeFractionsAndSubnormals) {
    struct Example {
        uint32_t bits, want;
    };
    constexpr std::array examples{Example{1, 0},
                                  Example{0x80000001, UINT32_MAX},
                                  Example{0x007fffff, 0},
                                  Example{0x807fffff, UINT32_MAX},
                                  Example{0x00800000, 0},
                                  Example{0x80800000, UINT32_MAX},
                                  Example{0, 0},
                                  Example{0x80000000, 0},
                                  Example{0x3fc00000, 1},
                                  Example{0xbfc00000, 0xfffffffe},
                                  Example{0xbf800001, 0xfffffffe},
                                  Example{0xc0100000, 0xfffffffd},
                                  Example{0x4effffff, 0x7fffff80},
                                  Example{0xcf000000, 0x80000000}};
    const auto* def = find_vector_cvt_flr_e64_op(Rdna2Format::VOP3, 0x18d);
    ASSERT_NE(def, nullptr);
    const auto module = compile(cases::words());
    for (const auto x : examples) {
        EXPECT_EQ(cases::finite_floor(x.bits), Known(x.want));
        EXPECT_EQ(eval_conversion_op(*def, x.bits), Known(x.want));
        auto in = cases::inputs(11);
        std::fill(in.s0.begin(), in.s0.end(), x.bits);
        check(module, in);
    }
}
TEST(IrVectorCvtFlrE64Execution, DifferingLanesAndAllExecShapesMatchBothPermittedFloorProfiles) {
    for (uint32_t wave : {32u, 64u}) {
        const auto module = compile(cases::words(), wave);
        for (uint32_t seed = 1; seed <= 24; ++seed) {
            auto in = cases::inputs(seed * 5355);
            if (seed == 1) std::fill(in.active.begin(), in.active.end(), false);
            if (seed == 2) std::fill(in.active.begin(), in.active.end(), true);
            check(module, in);
        }
    }
}
TEST(IrVectorCvtFlrE64Execution, EverySourceAndPrefixExecutesAndMatchesOldE32Module) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t form = 0; form < 8; ++form)
            for (uint32_t mode : {0u, 0x30000u, 1u << 23u, UINT32_MAX}) {
                const auto in = cases::inputs(13);
                const auto old = compile(cases::words(cases::form(form), 2), wave, mode);
                check(old, in, cases::form(form));
                for (uint32_t prefix : {0u, 1u}) {
                    const auto current =
                        compile(cases::words(cases::form(form), prefix), wave, mode);
                    EXPECT_EQ(current, old) << form << ':' << mode << ':' << prefix;
                    check(current, in, cases::form(form));
                }
            }
}
TEST(IrVectorCvtFlrE64Execution, AllModifiersAndNoncanonicalFieldsRemainRefusedBesidePlainControl) {
    const auto in = cases::inputs(19);
    for (uint32_t word = 0; word < 2; ++word)
        for (uint32_t bit = word ? 9u : 8u; bit < (word ? 32u : 16u); ++bit) {
            auto code = cases::words();
            code[word] |= 1u << bit;
            ASSERT_FALSE(run_vector_op_per_lane(code, in).error.empty()) << word << ':' << bit;
        }
    check(compile(cases::words()), in);
}
TEST(IrVectorCvtFlrE64Execution, MissingAndRelayedScalarSourcesCannotCreateFreshMasks) {
    for (uint32_t wave : {32u, 64u})
        for (bool relay : {false, true})
            for (bool initialize : {false, true}) {
                auto form = cases::form(2);
                form.initialize = initialize;
                std::vector<uint32_t> words;
                if (relay) {
                    if (initialize) words.insert(words.end(), {0xbeb203ff, form.scalar});
                    words.push_back(vop1(1, 254, 50));
                    form.initialize = false;
                    form.source = 510;
                }
                const auto op = cases::words(form);
                words.insert(words.end(), op.begin(), op.end());
                words.insert(words.end(), {0xbefe04c1, 0x7d8a0280});
                const auto run = run_vector_op_per_lane(words, cases::inputs(11), wave);
                if (initialize)
                    ASSERT_TRUE(run.error.empty()) << run.error;
                else
                    EXPECT_NE(run.error.find("fabricated-fresh-mask-source"), std::string::npos)
                        << run.error;
            }
}
TEST(IrVectorCvtFlrE64Execution, OneArmScalarSourcesKeepTheirObligationWithBothEdgeControl) {
    for (uint32_t wave : {32u, 64u})
        for (bool initialize : {false, true}) {
            std::vector<uint32_t> code;
            if (initialize) code.insert(code.end(), {0xbeb203ff, 0x3fe00000});
            code.insert(code.end(), {0xbf068004, 0xbf840002, 0xbeb203ff, 0x40000000});
            auto form = cases::form(2);
            form.initialize = false;
            const auto op = cases::words(form);
            code.insert(code.end(), op.begin(), op.end());
            code.insert(code.end(), {0xbefe04c1, 0x7d8a0280});
            auto p = lane_program(code, 64, wave);
            p.config.user_sgprs.resize(8);
            const auto address = lane_harness_address();
            const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                                  {RecompileDiagnosticStage::Compute, address});
            const auto reason = last_terminal_reject_reason(address);
            if (initialize)
                ASSERT_FALSE(module.empty()) << reason;
            else {
                EXPECT_TRUE(module.empty());
                EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
            }
        }
}
TEST(IrVectorCvtFlrE64Execution, NonfiniteAndUnknownPolicyMatchesOldModelBoundaryExactly) {
    const auto* def = find_vector_cvt_flr_e64_op(Rdna2Format::VOP3, 0x18d);
    ASSERT_NE(def, nullptr);
    EXPECT_FALSE(eval_conversion_op(*def, std::nullopt));
    for (uint32_t source : {258u, 255u, 193u})
        for (uint32_t bits : {0x7f800000u, 0xff800000u, 0x7fcabcdeu, 0xff800001u}) {
            cases::Form form;
            form.source = source;
            form.literal = bits;
            auto in = cases::inputs(11);
            in.s0[1] = bits;
            in.active[1] = true;
            const auto old = compile(cases::words(form, 2));
            const auto current = compile(cases::words(form));
            EXPECT_EQ(current, old);
            const auto raw = source == 193 ? UINT32_MAX : bits;
            EXPECT_FALSE(eval_conversion_op(*def, raw));
            for (bool flush : {false, true})
                EXPECT_EQ(cases::execute(current, in, flush).error,
                          "unsupported nonfinite unary float input");
        }
}
TEST(IrVectorCvtFlrE64Execution, ExportsAndExecutesEverySavedAliasAndMatchedE32Control) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this test does not change the environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "cvt-flr-e64-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root));
    ASSERT_TRUE(fs::create_directories(root / "refused_shaders_20000101-000000Z_synthetic"));
    ASSERT_TRUE(fs::create_directories(root / "modules"));
    std::ofstream manifest(root / "manifest.tsv");
    manifest << "stem\twave\tform\trsrc1\tencoding\n";
    size_t count = 0;
    for (uint32_t wave : {32u, 64u})
        for (uint32_t form = 0; form < 8; ++form)
            for (uint32_t mode : {0u, 0x30000u, 1u << 23u, UINT32_MAX})
                for (uint32_t encoding = 0; encoding < 3; ++encoding)
                    export_witness(root, manifest, count++, wave, form, mode, encoding);
    manifest.close();
    ASSERT_TRUE(manifest.good());
    EXPECT_EQ(count, 192u);
}
