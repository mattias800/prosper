// Canonical e64 FMAC must consume the old lane destination as the fused addend, and preserve
// inactive lanes. The CPU model selects finite normal-or-zero RTE; GLSL Fma remains MED
// portability (#5225), with device accuracy qualified separately by an independent control.
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ops/vector_mac_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
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
#include <set>
#include <string>
#include <system_error>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
constexpr uint32_t kForms = 18;
std::vector<uint32_t> words(uint32_t form = 0, bool legacy_prefix = false) {
    uint32_t a = 258, c = 259;
    if (form == 1 || form == 3) a = 242; // Inline F32 1.
    if (form == 2 || form == 3) c = 244; // Inline F32 2.
    if (form == 4) a = 255;
    if (form == 5) c = 255;
    if (form == 6) a = 50;
    if (form == 7) c = 50;
    if (form == 8 || form == 10) a = 257;
    if (form == 9 || form == 10) c = 257;
    if (form == 11) a = 128; // Inline integer zero is a zero raw F32 word.
    if (form == 12) c = 258;
    if (form == 13) a = 259;
    if (form == 14) c = 128;
    if (form == 15) {
        a = 50;
        c = 51;
    }
    if (form == 16) {
        a = 50;
        c = 255;
    }
    if (form == 17) {
        a = 255;
        c = 50;
    }
    std::vector<uint32_t> out;
    if (form == 6 || form == 7 || form >= 15) out.insert(out.end(), {0xbeb203ffu, 0x3fc00000u});
    if (form == 15) out.insert(out.end(), {0xbeb303ffu, 0xbf400000u});
    out.insert(out.end(), {legacy_prefix ? 0xd12b0001u : 0xd52b0001u, a | (c << 9)});
    if (form == 4 || form == 5 || form == 16 || form == 17)
        out.push_back(form == 4 || form == 17 ? 0x3f800001u : 0x3f7ffffeu);
    return out;
}
LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    const uint32_t a[] = {0x3f800001, 0x3fc00000, 0xbf800000, 0x40000000};
    const uint32_t c[] = {0x3f7ffffe, 0x40000000, 0x40000000, 0x3f000000};
    const uint32_t d[] = {0xbf800000, 0x3f000000, 0x40400000, 0xbf800000};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] =
            lane < 16 ? a[lane % 4] : (in.s0[lane] & 0x807fffffu) | ((112u + lane % 20) << 23);
        in.s1[lane] =
            lane < 16 ? c[lane % 4] : (in.s1[lane] & 0x807fffffu) | ((114u + lane % 18) << 23);
        seed = seed * 1664525u + 1013904223u;
        in.old_d[lane] =
            lane < 16 ? d[lane % 4] : (seed & 0x807fffffu) | ((116u + lane % 16) << 23);
    }
    in.active[4] = true;
    return in;
}
std::array<Known, 3> sources(const LaneInputs& in, size_t lane, uint32_t form) {
    uint32_t a = in.s0[lane], c = in.s1[lane];
    if (form == 1 || form == 3) a = 0x3f800000;
    if (form == 2 || form == 3) c = 0x40000000;
    if (form == 4) a = 0x3f800001;
    if (form == 5) c = 0x3f7ffffe;
    if (form == 6) a = 0x3fc00000;
    if (form == 7) c = 0x3fc00000;
    if (form == 8 || form == 10) a = in.old_d[lane];
    if (form == 9 || form == 10) c = in.old_d[lane];
    if (form == 11) a = 0;
    if (form == 12) c = in.s0[lane];
    if (form == 13) a = in.s1[lane];
    if (form == 14) c = 0;
    if (form == 15) {
        a = 0x3fc00000;
        c = 0xbf400000;
    }
    if (form == 16) {
        a = 0x3fc00000;
        c = 0x3f7ffffe;
    }
    if (form == 17) {
        a = 0x3f800001;
        c = 0x3fc00000;
    }
    return {a, c, in.old_d[lane]};
}
LaneRun execute(const std::vector<uint32_t>& module, const LaneInputs& in) {
    std::vector<uint32_t> buffer;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
    bpermute_oracle::Interpreter vm(module);
    auto out = vm.run_buffers(static_cast<uint32_t>(in.lanes()), {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    out.resize(in.lanes());
    return {out, {}};
}
void expect_lanes(const LaneRun& run, const LaneInputs& in, uint32_t form = 0, bool fused = true) {
    const auto* def = fused ? find_vector_fmac_float_op(Rdna2Format::VOP3, 0x12b)
                            : find_vector_mac_float_op(Rdna2Format::VOP3, 0x11f);
    ASSERT_NE(def, nullptr);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = eval_ternary_integer_op(*def, sources(in, lane, form));
        if (!want) {
            ADD_FAILURE() << "unknown reference lane " << lane;
            return;
        }
        ASSERT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << "form " << form << " lane " << lane;
    }
}
} // namespace

TEST(IrVectorFmacFloatE64Execution, LlvmEncodingAndCentralEffectsDescribeTwoSourcesAndOldD) {
    EXPECT_EQ(words(), (std::vector<uint32_t>{0xd52b0001u, 0x00020702u}));
    EXPECT_EQ(words(0, true), (std::vector<uint32_t>{0xd12b0001u, 0x00020702u}));
    EXPECT_EQ(words(1), (std::vector<uint32_t>{0xd52b0001u, 0x000206f2u}));
    EXPECT_EQ(words(2), (std::vector<uint32_t>{0xd52b0001u, 0x0001e902u}));
    EXPECT_EQ(words(4), (std::vector<uint32_t>{0xd52b0001u, 0x000206ffu, 0x3f800001u}));
    EXPECT_EQ(words(5), (std::vector<uint32_t>{0xd52b0001u, 0x0001ff02u, 0x3f7ffffeu}));
    const auto code = words();
    const auto in = rdna2_decode_one(code.data(), code.size());
    EXPECT_EQ(in.fmt, Rdna2Format::VOP3);
    EXPECT_EQ(in.opcode, 0x12bu);
    EXPECT_EQ(in.sdst.kind, OperandKind::None);
    EXPECT_EQ(code[1] & (0x1ffu << 18), 0u);
    const auto& e = rdna2_instruction_effects(in.fmt, in.opcode);
    EXPECT_EQ(std::string(e.name), "v_fmac_f32");
    EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
    EXPECT_TRUE(e.accumulates);
    EXPECT_EQ(e.vdst_dwords, 1u);
    EXPECT_FALSE(e.vdst_half);
    EXPECT_EQ(rdna2_valu_source_count(in.fmt, in.opcode), 2u);
    EXPECT_EQ(vector_mac_float_defs().size(), 1u) << "old nonfused inventory stays unchanged";
    for (uint32_t op : {0x11fu, 0x12cu, 0x12du, 0x14bu, 0x136u})
        EXPECT_EQ(find_vector_fmac_float_op(Rdna2Format::VOP3, op), nullptr);
}
TEST(IrVectorFmacFloatE64Execution, EveryOperandAndAliasFormMatchesDifferingLanesInBothWaves) {
    for (bool legacy_prefix : {false, true})
        for (uint32_t wave : {32u, 64u})
            for (uint32_t form = 0; form < kForms; ++form)
                for (uint32_t seed = 1; seed <= 4; ++seed)
                    expect_lanes(run_vector_op_per_lane(words(form, legacy_prefix),
                                                        inputs(seed * 5342), wave),
                                 inputs(seed * 5342), form);
}
TEST(IrVectorFmacFloatE64Execution, ExistingDecoderPrefixTwinsKeepModulesAndResultsIdentical) {
    // Both prefixes were already decoded as the same operation. Admit their verified operand
    // shape without making a claim about other ISA generations or changing the decoder.
    for (uint32_t wave : {32u, 64u})
        for (uint32_t form = 0; form < kForms; ++form) {
            std::array<std::vector<uint32_t>, 2> modules;
            for (size_t legacy = 0; legacy < 2; ++legacy) {
                const auto code = words(form, legacy != 0);
                const size_t pc = form == 15 ? 4u : form == 6 || form == 7 || form >= 16 ? 2u : 0u;
                const auto decoded = rdna2_decode_one(code.data() + pc, code.size() - pc);
                EXPECT_EQ(decoded.fmt, Rdna2Format::VOP3);
                EXPECT_EQ(decoded.opcode, 0x12bu);
                const auto p = lane_program(code, 64, wave);
                modules[legacy] =
                    recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
                ASSERT_FALSE(modules[legacy].empty());
                expect_lanes(execute(modules[legacy], inputs(5342)), inputs(5342), form);
            }
            EXPECT_EQ(modules[0], modules[1]) << "wave " << wave << " form " << form;
        }
}
TEST(IrVectorFmacFloatE64Execution, IndependentFusedRoundingAndOverflowCancellationExamples) {
    struct Example {
        uint32_t a, c, old, want;
    };
    // (1+2^-23)*(1-2^-23)-1 = -2^-46. A separate product loses the residual.
    // maxFinite*2-maxFinite = maxFinite exactly, despite an overflowing separate product.
    for (const auto x : {Example{0x3f800001, 0x3f7ffffe, 0xbf800000, 0xa8800000},
                         Example{0x7f7fffff, 0x40000000, 0xff7fffff, 0x7f7fffff},
                         Example{0x3fc00000, 0x40000000, 0x3f000000, 0x40600000},
                         Example{0xbf800000, 0x40000000, 0x40400000, 0x3f800000},
                         Example{0x80000000, 0x40000000, 0x80000000, 0x80000000}}) {
        const auto* def = find_vector_fmac_float_op(Rdna2Format::VOP3, 0x12b);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_ternary_integer_op(*def, {x.a, x.c, x.old}), Known(x.want));
        auto in = inputs(9);
        std::fill(in.s0.begin(), in.s0.end(), x.a);
        std::fill(in.s1.begin(), in.s1.end(), x.c);
        std::fill(in.old_d.begin(), in.old_d.end(), x.old);
        for (uint32_t wave : {32u, 64u}) {
            const auto run = run_vector_op_per_lane(words(), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : x.old) << lane;
        }
    }
}
TEST(IrVectorFmacFloatE64Execution, HarnessSeparatesAccumulatorAndInactiveDestinations) {
    const auto in = inputs(1);
    const auto run = run_vector_op_per_lane(words(), in);
    expect_lanes(run, in);
    ASSERT_EQ(run.d.size(), in.lanes());
    std::set<uint32_t> active;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane])
            active.insert(run.d[lane]);
        else {
            ++inactive;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    EXPECT_GT(active.size(), 16u);
    EXPECT_GT(inactive, 4u);
    EXPECT_EQ(run.d[4], 0xa8800000u);
}
TEST(IrVectorFmacFloatE64Execution, NoncanonicalModifiersAndUnusedFieldsRefuseBesideSuccess) {
    for (uint32_t word = 0; word < 2; ++word)
        for (uint32_t bit = word ? 18u : 8u; bit < (word ? 32u : 16u); ++bit) {
            auto bad = words();
            bad[word] |= 1u << bit;
            EXPECT_FALSE(run_vector_op_per_lane(bad, inputs(1)).error.empty())
                << "word " << word << " bit " << bit;
        }
    auto raw_literal = words();
    raw_literal[1] |= 255u << 18;
    raw_literal.push_back(0x3f800000u);
    EXPECT_FALSE(run_vector_op_per_lane(raw_literal, inputs(1)).error.empty());
    for (uint32_t op : {0x12cu, 0x12du, 0x136u, 0x13cu}) {
        auto absent = words();
        absent[0] = 0xd4000001u | (op << 16);
        EXPECT_FALSE(run_vector_op_per_lane(absent, inputs(1)).error.empty());
    }
    expect_lanes(run_vector_op_per_lane(words(), inputs(1)), inputs(1));
}
TEST(IrVectorFmacFloatE64Execution, ModelKeepsUnknownNonfiniteAndSubnormalBoundariesVisible) {
    const auto* def = find_vector_fmac_float_op(Rdna2Format::VOP3, 0x12b);
    ASSERT_NE(def, nullptr);
    for (size_t slot = 0; slot < 3; ++slot)
        for (const Known missing : {Known{}, Known(1u), Known(0x7f800000u), Known(0x7fc00001u)}) {
            std::array<Known, 3> a = {0x3f800000u, 0x40000000u, 0x40400000u};
            a[slot] = missing;
            EXPECT_FALSE(eval_ternary_integer_op(*def, a));
        }
    // Denorm MODE fields only: the selected finite-normal RTE witnesses do not establish
    // non-RTE or denormal device semantics, even when launch metadata is absent.
    for (uint32_t mode : {0u, 0x20000u, 0x30000u, UINT32_MAX})
        expect_lanes(run_vector_op_per_lane(words(), inputs(7), 64, false, mode), inputs(7));
}
TEST(IrVectorFmacFloatE64Execution, HardScalarInputsAndOldDStillCannotSupplyAFreshMask) {
    for (uint32_t form : {6u, 7u})
        for (uint32_t wave : {32u, 64u})
            for (bool genuine : {false, true}) {
                auto op = words(form);
                if (!genuine) op.erase(op.begin(), op.begin() + 2);
                op.insert(op.end(), {0xbefe04c1u, 0x7d8a0280u});
                const auto run = run_vector_op_per_lane(op, inputs(1), wave);
                if (genuine)
                    ASSERT_TRUE(run.error.empty()) << run.error;
                else
                    EXPECT_NE(run.error.find("fabricated-fresh-mask-source"), std::string::npos)
                        << run.error;
            }
    for (bool genuine : {false, true}) {
        auto op = words();
        if (genuine) op.insert(op.begin(), {0xbeb203ffu, 0x3f800000u});
        op.insert(op.begin() + (genuine ? 2 : 0), 0x7e020232u); // v_mov_b32 v1,s50
        op.insert(op.end(), {0xbefe04c1u, 0x7d8a0280u});
        const auto run = run_vector_op_per_lane(op, inputs(1));
        if (genuine)
            ASSERT_TRUE(run.error.empty()) << run.error;
        else
            EXPECT_NE(run.error.find("fabricated-fresh-mask-source"), std::string::npos)
                << run.error;
    }
}
TEST(IrVectorFmacFloatE64Execution, OneArmScalarSourceKeepsItsExistingMustObligation) {
    for (uint32_t form : {6u, 7u})
        for (uint32_t wave : {32u, 64u})
            for (bool genuine : {false, true}) {
                std::vector<uint32_t> op;
                if (genuine) op.insert(op.end(), {0xbeb203ffu, 0x3fc00000u});
                op.insert(op.end(), {0xbf068004u, 0xbf840002u, 0xbeb203ffu, 0x40000000u});
                const auto fmac = words(form);
                op.insert(op.end(), fmac.begin() + 2, fmac.end());
                op.insert(op.end(), {0xbefe04c1u, 0x7d8a0280u});
                auto p = lane_program(op, 64, wave);
                p.config.user_sgprs.resize(8);
                const auto address = lane_harness_address();
                const auto module =
                    recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                      {RecompileDiagnosticStage::Compute, address});
                const auto reason = last_terminal_reject_reason(address);
                if (genuine)
                    ASSERT_FALSE(module.empty()) << reason;
                else {
                    EXPECT_TRUE(module.empty());
                    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos)
                        << reason;
                }
            }
}
TEST(IrVectorFmacFloatE64Execution, ExportsExecutesAndSavesEveryAdmittedFormAndOldControl) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "e64-fmac-f32-witness";
    if (!configured) {
        std::error_code ec;
        fs::remove_all(root, ec);
        ASSERT_FALSE(ec) << ec.message();
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory: " << root;
    ASSERT_TRUE(fs::create_directories(root / "modules"));
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directory(dump));
    std::ofstream manifest(root / "forms.tsv");
    size_t count = 0;
    for (uint32_t encoding = 0; encoding < 4; ++encoding)
        for (uint32_t wave : {32u, 64u})
            for (uint32_t form = 0; form < (encoding == 0 || encoding == 3 ? kForms : 1u); ++form)
                for (uint32_t mode : {0u, 0x20000u, 0x30000u, UINT32_MAX}) {
                    if (encoding != 0 && encoding != 3 && mode) continue;
                    auto code = words(form, encoding == 3);
                    if (encoding == 1) code = {vop2(0x2b)};
                    if (encoding == 2) code[0] = 0xd51f0001u;
                    auto p = lane_program(code, 64, wave);
                    p.config.compute_pgm_rsrc1 = mode;
                    const auto address = 0x5342000000ull + count * 0x100;
                    const auto stem =
                        dump / ("cs_" + std::to_string(address) + "_" + std::to_string(count));
                    std::ofstream bin(stem.string() + ".bin", std::ios::binary);
                    bin.write(reinterpret_cast<const char*>(p.code.data()),
                              static_cast<std::streamsize>(p.code.size() * sizeof(uint32_t)));
                    bin.close();
                    ASSERT_TRUE(bin.good());
                    const auto file = stem.string() + ".inputs";
                    ASSERT_TRUE(write_refused_compute_inputs(
                        file, address, &p.table, p.config, "synthetic-witness (no live refusal)"));
                    const auto read = read_refused_shader_inputs(file);
                    ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
                    const auto replay = replay_refused_shader_inputs(p.code, read);
                    ASSERT_TRUE(replay.produced) << replay.reject;
                    expect_lanes(execute(replay.spirv, inputs(5342)), inputs(5342), form,
                                 encoding != 2);
                    std::ofstream spv(root / "modules" / (std::to_string(count) + ".spv"),
                                      std::ios::binary);
                    spv.write(reinterpret_cast<const char*>(replay.spirv.data()),
                              static_cast<std::streamsize>(replay.spirv.size() * sizeof(uint32_t)));
                    spv.close();
                    ASSERT_TRUE(spv.good());
                    manifest << stem.filename().string() << '\t' << wave << '\t' << form << '\t'
                             << mode << '\t' << encoding << '\n';
                    ++count;
                }
    EXPECT_EQ(count, 292u);
    ASSERT_TRUE(manifest.good());
}
