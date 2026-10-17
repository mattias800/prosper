// Execute actual guest modules against independent normal-half bit answers, including inactive D.
#include "gpu/recompiler/ir/half_float_exact_e64_cases.hpp"
#include "gpu/recompiler/ir/ops/vector_half_float_exact_e64.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <string>
#include <system_error>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::half_float_exact;
namespace {
std::vector<uint32_t> compile(const LaneProgram& p) {
    return recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                             {RecompileDiagnosticStage::Compute, lane_harness_address()});
}
LaneRun execute(const std::vector<uint32_t>& module, const LaneInputs& in) {
    std::vector<uint32_t> buffer;
    buffer.reserve(in.lanes() * 4);
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
    bpermute_oracle::Interpreter vm(module);
    // The VM's explicit half profile is needed to execute its unpack opcode. Every selected input
    // is normal and every output normal/nonzero, so neither half RTE nor denormal choices apply.
    vm.half_rte_preserve_denorm = true;
    auto result = vm.run_buffers(static_cast<uint32_t>(in.lanes()), {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    if (result.size() < in.lanes()) return {{}, "missing stored lanes"};
    result.resize(in.lanes());
    return {result, {}};
}
void expect_lanes(const std::vector<uint32_t>& module, const LaneInputs& in, uint32_t source,
                  uint32_t shape = 0, bool varying = false) {
    ASSERT_FALSE(module.empty());
    const auto* def = find_vector_half_float_exact_e64_op(Rdna2Format::VOP3, 0x18b);
    ASSERT_NE(def, nullptr);
    const auto run = execute(module, in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto raw = varying ? in.s0[lane] : source;
        const auto answer = cases::reference(raw);
        ASSERT_TRUE(answer.has_value()) << raw;
        ASSERT_EQ(eval_conversion_op(*def, raw), answer);
        const auto want = shape == 3 ? ~answer.value_or(0) : answer.value_or(0);
        ASSERT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]) << "lane " << lane;
    }
}
std::vector<uint32_t> load_words(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    if (!in) {
        ADD_FAILURE() << p;
        return {};
    }
    const auto size = in.tellg();
    if (size <= 0 || size % 4 != 0) {
        ADD_FAILURE() << p;
        return {};
    }
    std::vector<uint32_t> words(static_cast<size_t>(size) / sizeof(uint32_t));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(size));
    EXPECT_TRUE(in.good());
    return words;
}
void save_words(const std::filesystem::path& p, const std::vector<uint32_t>& words) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    out.close();
    ASSERT_TRUE(out.good()) << p;
}
void export_form(const std::filesystem::path& root, std::ofstream& manifest, size_t index,
                 uint32_t selector, uint32_t literal, uint32_t encoding, uint32_t wave,
                 uint32_t launch) {
    const auto program = cases::program(selector, literal, encoding, wave, 0, launch);
    const uint64_t address = 0x5362000000ull + index * 0x100;
    const auto stem = root / "refused_shaders_20000101-000000Z_synthetic" /
                      ("cs_" + std::to_string(address) + "_" + std::to_string(index));
    const auto bin = std::filesystem::path(stem.string() + ".bin");
    const auto sidecar = std::filesystem::path(stem.string() + ".inputs");
    save_words(bin, program.code);
    ASSERT_TRUE(write_refused_compute_inputs(sidecar, address, &program.table, program.config,
                                             "synthetic exact normal-half widening witness"));
    const auto saved_code = load_words(bin);
    ASSERT_EQ(saved_code, program.code);
    const auto read = read_refused_shader_inputs(sidecar);
    ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
    const auto replay = replay_refused_shader_inputs(saved_code, read);
    ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
    ASSERT_TRUE(replay.produced) << replay.reject;
    const auto spv_file = root / "modules" / (std::to_string(index) + ".spv");
    save_words(spv_file, replay.spirv);
    const auto saved_module = load_words(spv_file);
    ASSERT_EQ(saved_module, replay.spirv);
    const uint32_t half = selector == 255 ? literal : cases::kInlineHalf.at(selector - 240);
    expect_lanes(saved_module, cases::inputs(), half);
    manifest << stem.filename().string() << '\t' << selector << '\t' << half << '\t' << encoding
             << '\t' << wave << '\t' << launch << '\n';
}
}   // namespace

TEST(IrVectorHalfFloatExactE64Execution, EncodingsAndHandNormalWordsMatchIndependentBitAnswers) {
    EXPECT_EQ(cases::words(242), (std::vector<uint32_t>{0xd58b0001, 242}));
    EXPECT_EQ(cases::words(255, 0xfbff, 1), (std::vector<uint32_t>{0xd18b0001, 255, 0xfbff}));
    EXPECT_EQ(cases::words(255, 0xfbff, 2), (std::vector<uint32_t>{0x7e0216ff, 0xfbff}));
    for (const auto& h : cases::kHand) {
        ASSERT_EQ(cases::reference(h.half), Known(h.answer));
        for (uint32_t wave : {32u, 64u})
            for (uint32_t encoding : {0u, 1u})
                expect_lanes(compile(cases::program(255, h.half, encoding, wave)), cases::inputs(),
                             h.half);
    }
}
TEST(IrVectorHalfFloatExactE64Execution, AllInlineWidthsAndNormalLiteralSignsExecuteBothPrefixes) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t encoding : {0u, 1u}) {
            for (uint32_t selector = 240; selector <= 248; ++selector)
                expect_lanes(compile(cases::program(selector, 0, encoding, wave)), cases::inputs(),
                             cases::kInlineHalf.at(selector - 240));
            uint32_t seed = 5362;
            for (uint32_t index = 0; index < 64; ++index) {
                seed = seed * 1664525u + 1013904223u;
                const auto half = (index & 1 ? 0x8000u : 0u) | (0x0400u + seed % 0x7800u);
                expect_lanes(compile(cases::program(255, half, encoding, wave)), cases::inputs(),
                             half);
            }
        }
}
TEST(IrVectorHalfFloatExactE64Execution,
     FullEmptyPartialExecAndFollowingInstructionKeepTheirWords) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t encoding : {0u, 1u})
            for (uint32_t shape = 0; shape < 4; ++shape)
                for (uint32_t half : {0x0400u, 0xbc01u, 0x7bffu, 0xfbffu})
                    expect_lanes(compile(cases::program(255, half, encoding, wave, shape)),
                                 cases::inputs(shape), half, shape);
}
TEST(IrVectorHalfFloatExactE64Execution, RemovingExecMaskOverwritesEveryInactiveCanary) {
    const auto in = cases::inputs();
    for (uint32_t wave : {32u, 64u})
        for (uint32_t encoding : {0u, 1u}) {
            expect_lanes(compile(cases::program(242, 0, encoding, wave)), in, 0x3c00);
            const auto unmasked = compile(cases::program(242, 0, encoding, wave, 1));
            ASSERT_FALSE(unmasked.empty());
            const auto run = execute(unmasked, in);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), in.lanes());
            size_t inactive = 0, overwritten = 0;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                ASSERT_EQ(run.d[lane], 0x3f800000u);   // Exact half1.0 -> F32 1.0.
                ASSERT_NE(in.old_d[lane], 0x3f800000u);
                if (!in.active[lane]) {
                    ++inactive;
                    overwritten += run.d[lane] != in.old_d[lane];
                }
            }
            ASSERT_GT(inactive, 0u);
            EXPECT_EQ(overwritten, inactive);
        }
}
TEST(IrVectorHalfFloatExactE64Execution, AllModeBytesAndUnknownLeaveExactNormalWideningUnchanged) {
    for (uint32_t mode = 0; mode <= 256; ++mode)
        for (uint32_t ieee : {0u, 1u})
            for (uint32_t overflow : {0u, 1u}) {
                const auto launch =
                    mode == 256 ? UINT32_MAX : (mode << 12) | (ieee << 23) | (overflow << 26);
                const auto half = mode & 1 ? 0xbc01u : 0x3c01u;
                expect_lanes(
                    compile(cases::program(255, half, mode & 1, mode & 2 ? 32u : 64u, 0, launch)),
                    cases::inputs(), half);
            }
}
TEST(IrVectorHalfFloatExactE64Execution, OldE32RunsDistinctNormalInputsPerLane) {
    auto in = cases::inputs();
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        in.s0[lane] = (lane & 1 ? 0x8000u : 0u) | (0x0400u + static_cast<uint32_t>(lane) * 433u);
    for (uint32_t wave : {32u, 64u})
        expect_lanes(compile(cases::program(258, 0, 2, wave)), in, 0, 0, true);
}
TEST(IrVectorHalfFloatExactE64Execution, UnsupportedValuesSourcesAndRawFlagsRefuseBesideSuccess) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t encoding : {0u, 1u}) {
            for (uint32_t raw : {0u, 0x8000u, 1u, 0x8001u, 0x03ffu, 0x83ffu, 0x7c00u, 0xfc00u,
                                 0x7c01u, 0x7e00u, 0xfc01u, 0xffffu, 0x3c003c00u, 0x3f800000u})
                EXPECT_TRUE(compile(cases::program(255, raw, encoding, wave)).empty()) << raw;
            for (uint32_t selector : {50u, 125u, 128u, 129u, 257u, 258u}) {
                auto p = cases::program(selector, 0, encoding, wave);
                if (selector == 50) p.code.insert(p.code.begin(), {0xbeb203ff, 0x3c00});
                EXPECT_TRUE(compile(p).empty()) << selector;
            }
            for (uint32_t word : {0u, 1u})
                for (uint32_t bit = word ? 9u : 8u; bit < (word ? 32u : 16u); ++bit) {
                    auto raw = cases::words(242, 0, encoding);
                    raw[word] |= 1u << bit;
                    auto p = cases::program(242, 0, encoding, wave);
                    // Replace the two instruction words without changing the harness shape.
                    for (size_t i = 0; i + 1 < p.code.size(); ++i)
                        if (p.code[i] == (encoding ? 0xd18b0001u : 0xd58b0001u)) {
                            p.code[i] = raw[0];
                            p.code[i + 1] = raw[1];
                            break;
                        }
                    EXPECT_TRUE(compile(p).empty()) << word << ':' << bit;
                }
            expect_lanes(compile(cases::program(242, 0, encoding, wave)), cases::inputs(), 0x3c00);
        }
}
TEST(IrVectorHalfFloatExactE64Execution, SavedReplaysReloadAndExecuteEverySourceWaveAndEncoding) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "half-float-exact-e64";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root)) << "export requires a new witness directory";
    ASSERT_TRUE(fs::create_directories(root / "modules"));
    ASSERT_TRUE(fs::create_directory(root / "refused_shaders_20000101-000000Z_synthetic"));
    std::ofstream manifest(root / "manifest.tsv");
    size_t index = 0;
    for (uint32_t encoding : {0u, 1u, 2u})
        for (uint32_t wave : {32u, 64u})
            for (uint32_t launch : {UINT32_MAX, 0u, 0x33000u, 0x48ff000u}) {
                for (uint32_t selector = 240; selector <= 248; ++selector)
                    export_form(root, manifest, index++, selector, 0, encoding, wave, launch);
                for (const auto& h : cases::kHand)
                    export_form(root, manifest, index++, 255, h.half, encoding, wave, launch);
            }
    manifest.close();
    EXPECT_TRUE(manifest.good());
    EXPECT_EQ(index, 600u);   // 400new e64 attributes and200old e32 controls.
}
