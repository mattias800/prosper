// The newly admitted source is uniform by encoding; observe each actual lane's EXEC/oldD.
// Hand half-integer answers and integer normalization are independent of the shared builder/eval.
#include "gpu/recompiler/ir/cvt_rpi_exact_e64_cases.hpp"
#include "gpu/recompiler/ir/ops/vector_cvt_rpi_exact_e64.hpp"
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
namespace cases = prosper::test::ir::cvt_rpi_exact;
namespace {
std::vector<uint32_t> compile(const LaneProgram& p) {
    return recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                             {RecompileDiagnosticStage::Compute, lane_harness_address()});
}
LaneRun execute(const std::vector<uint32_t>& module, const LaneInputs& in, bool flush) {
    std::vector<uint32_t> buffer;
    buffer.reserve(in.lanes() * 4);
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
    bpermute_oracle::Interpreter vm(module);
    vm.floor_flush_denorm_input = flush;
    auto result = vm.run_buffers(static_cast<uint32_t>(in.lanes()), {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    if (result.size() < in.lanes()) return {{}, "missing stored lanes"};
    result.resize(in.lanes());
    return {result, {}};
}
void expect_lanes(const std::vector<uint32_t>& module, const LaneInputs& in, uint32_t source,
                  uint32_t prefix = 0, bool varying = false) {
    ASSERT_FALSE(module.empty());
    const auto* def = find_vector_cvt_rpi_exact_e64_op(Rdna2Format::VOP3, 0x18c);
    ASSERT_NE(def, nullptr);
    for (bool flush : {false, true}) {
        const auto run = execute(module, in, flush);
        ASSERT_TRUE(run.error.empty()) << run.error;
        ASSERT_EQ(run.d.size(), in.lanes());
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            const auto raw = varying ? in.s0[lane] : source;
            const auto answer = cases::reference(raw);
            ASSERT_TRUE(answer.has_value()) << raw;
            ASSERT_EQ(eval_conversion_op(*def, raw), answer);
            const auto want = prefix == 3 ? ~answer.value_or(0) : answer.value_or(0);
            ASSERT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]) << "lane " << lane;
        }
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
    const uint64_t address = 0x5360000000ull + index * 0x100;
    const auto stem = root / "refused_shaders_20000101-000000Z_synthetic" /
                      ("cs_" + std::to_string(address) + "_" + std::to_string(index));
    const auto bin = std::filesystem::path(stem.string() + ".bin");
    const auto sidecar = std::filesystem::path(stem.string() + ".inputs");
    save_words(bin, program.code);
    ASSERT_TRUE(write_refused_compute_inputs(sidecar, address, &program.table, program.config,
                                             "synthetic exact-half-step RPI witness"));
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
    const uint32_t bits = selector == 255 ? literal : cases::inline_bits(selector);
    expect_lanes(saved_module, cases::inputs(), bits);
    manifest << stem.filename().string() << '\t' << selector << '\t' << bits << '\t' << encoding
             << '\t' << wave << '\t' << launch << '\n';
}
} // namespace

TEST(IrVectorCvtRpiExactE64Execution, EncodingsAndHandTiesAreIndependentOfTheSharedDefinition) {
    EXPECT_EQ(cases::words(240), (std::vector<uint32_t>{0xd58c0001, 240}));
    EXPECT_EQ(cases::words(255, 0xbfc00000, 1),
              (std::vector<uint32_t>{0xd18c0001, 255, 0xbfc00000}));
    EXPECT_EQ(cases::words(255, 0xbfc00000, 2), (std::vector<uint32_t>{0x7e0218ff, 0xbfc00000}));
    for (const auto& h : cases::kHand) {
        ASSERT_EQ(cases::reference(h.bits), Known(h.answer));
        for (uint32_t encoding : {0u, 1u}) {
            const auto module = compile(cases::program(255, h.bits, encoding));
            ASSERT_FALSE(module.empty());
            expect_lanes(module, cases::inputs(), h.bits);
        }
    }
}
TEST(IrVectorCvtRpiExactE64Execution,
     EightInlineFloatsAndLiteralHalfStepsExecuteBothWavesAndPrefixes) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t encoding : {0u, 1u}) {
            for (uint32_t selector = 240; selector <= 247; ++selector)
                expect_lanes(compile(cases::program(selector, 0, encoding, wave)), cases::inputs(),
                             cases::inline_bits(selector));
            uint32_t seed = 5360;
            for (uint32_t index = 0; index < 64; ++index) {
                seed = seed * 1664525u + 1013904223u;
                const int32_t magnitude = static_cast<int32_t>(seed & 0xffffffu);
                const auto bits = cases::encode_units(index & 1 ? -magnitude : magnitude);
                expect_lanes(compile(cases::program(255, bits, encoding, wave)), cases::inputs(),
                             bits);
            }
        }
}
TEST(IrVectorCvtRpiExactE64Execution,
     FullEmptyPartialExecAndFollowingInstructionObserveTheirOwnWords) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t encoding : {0u, 1u})
            for (uint32_t prefix = 0; prefix < 4; ++prefix)
                for (const uint32_t bits : {0u, 0x80000000u, 0x3fc00000u, 0xbfc00000u})
                    expect_lanes(compile(cases::program(255, bits, encoding, wave, prefix)),
                                 cases::inputs(prefix), bits, prefix);
}
TEST(IrVectorCvtRpiExactE64Execution, RemovingExecMaskOverwritesInactiveCanaries) {
    // The paired full-EXEC program deliberately omits CMPX. A checker that ignored inactive
    // destinations would miss this corruption, so count its wrong canaries independently.
    const auto in = cases::inputs();
    for (uint32_t wave : {32u, 64u})
        for (uint32_t encoding : {0u, 1u}) {
            expect_lanes(compile(cases::program(255, 0x3fc00000, encoding, wave)), in, 0x3fc00000);
            const auto unmasked = compile(cases::program(255, 0x3fc00000, encoding, wave, 1));
            ASSERT_FALSE(unmasked.empty());
            for (bool flush : {false, true}) {
                const auto run = execute(unmasked, in, flush);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                size_t inactive = 0, overwritten = 0;
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    ASSERT_EQ(run.d[lane], 2u); // ISA: floor(1.5 + 0.5), independent of eval.
                    ASSERT_NE(in.old_d[lane], 2u);
                    if (!in.active[lane]) {
                        ++inactive;
                        overwritten += run.d[lane] != in.old_d[lane];
                    }
                }
                ASSERT_GT(inactive, 0u);
                EXPECT_EQ(overwritten, inactive);
            }
        }
}
TEST(IrVectorCvtRpiExactE64Execution, AllProducingModeBytesAndUnknownKeepTheExactIntegerAnswer) {
    for (uint32_t mode = 0; mode <= 256; ++mode)
        for (uint32_t ieee : {0u, 1u})
            for (uint32_t wave : {32u, 64u}) {
                const auto launch = mode == 256 ? UINT32_MAX : (mode << 12) | (ieee << 23);
                const auto bits = mode & 1 ? 0xbfc00000u : 0x3fc00000u;
                const uint32_t encoding = mode & 1;
                expect_lanes(compile(cases::program(255, bits, encoding, wave, 0, launch)),
                             cases::inputs(), bits);
            }
}
TEST(IrVectorCvtRpiExactE64Execution, OldE32SameDomainAlsoRunsDifferingLaneValues) {
    auto in = cases::inputs();
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        in.s0[lane] = cases::encode_units(static_cast<int32_t>(lane * 71) - 2000);
    for (uint32_t wave : {32u, 64u})
        expect_lanes(compile(cases::program(258, 0, 2, wave)), in, 0, 0, true);
}
TEST(IrVectorCvtRpiExactE64Execution,
     NonLatticeSpecialAndRegisterInputsStayRefusedBesideTiedSuccess) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t encoding : {0u, 1u}) {
            for (uint32_t bits :
                 {1u, 0x80000001u, 0x3e800000u, 0x3effffffu, 0x3f000001u, 0x4b000000u, 0xcb000000u,
                  0x7f800000u, 0xff800000u, 0x7fc00001u, 0xff800001u})
                EXPECT_TRUE(compile(cases::program(255, bits, encoding, wave)).empty()) << bits;
            for (uint32_t selector : {50u, 125u, 128u, 129u, 248u, 257u, 258u}) {
                auto p = cases::program(selector, 0, encoding, wave);
                if (selector == 50) p.code.insert(p.code.begin(), {0xbeb203ff, 0x3fc00000});
                EXPECT_TRUE(compile(p).empty()) << selector;
            }
            expect_lanes(compile(cases::program(255, 0x3fc00000, encoding, wave)), cases::inputs(),
                         0x3fc00000);
        }
}
TEST(IrVectorCvtRpiExactE64Execution,
     SavedReplaysAreReloadedAndExecutedForEverySourceWaveAndEncoding) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "cvt-rpi-exact-e64";
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
            for (uint32_t launch : {UINT32_MAX, 0u, 0x33000u, 0x8ff000u}) {
                for (uint32_t selector = 240; selector <= 247; ++selector)
                    export_form(root, manifest, index++, selector, 0, encoding, wave, launch);
                for (const auto& h : cases::kHand)
                    export_form(root, manifest, index++, 255, h.bits, encoding, wave, launch);
            }
    manifest.close();
    EXPECT_TRUE(manifest.good());
    EXPECT_EQ(index, 576u);
}
