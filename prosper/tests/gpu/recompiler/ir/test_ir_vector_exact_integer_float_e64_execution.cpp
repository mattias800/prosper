// Execute the actual constant-source e64 module and the old e32 value path independently of eval.
// Uniform encoded sources still require differing per-lane EXEC and inactive destination canaries.
#include "gpu/recompiler/ir/exact_integer_float_e64_cases.hpp"
#include "gpu/recompiler/ir/ops/vector_exact_integer_float_e64.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::exact_integer_float;
namespace {
LaneInputs inputs(uint32_t prefix = 0) {
    auto in = lane_inputs(5356);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.old_d[lane] = 0x3e000000u | (static_cast<uint32_t>(lane) * 1009u);
        if (prefix == 1) in.active[lane] = true;
        if (prefix == 2) in.active[lane] = false;
    }
    return in;
}
LaneProgram program_for(uint32_t opcode, uint32_t selector, uint32_t literal, uint32_t wave,
                        uint32_t mode = UINT32_MAX, uint32_t prefix = 0, bool old = false) {
    auto words = cases::words(opcode, selector, literal, old);
    // A live following instruction makes the literal's stream length observable in the result.
    if (prefix == 3) words.push_back(vop1(0x37, kVdst, 256 + kVdst));
    auto p = lane_program(words, 64, wave);
    if (prefix == 1) {
        const auto cmpx = std::find(p.code.begin(), p.code.end(), 0x7daa0880u);
        EXPECT_NE(cmpx, p.code.end());
        if (cmpx != p.code.end()) p.code.erase(cmpx);
    }
    p.config.compute_pgm_rsrc1 = mode;
    return p;
}
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
    auto result = vm.run_buffers(static_cast<uint32_t>(in.lanes()), {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    if (result.size() < in.lanes()) return {{}, "missing stored lanes"};
    result.resize(in.lanes());
    return {std::move(result), {}};
}
void expect_lanes(const LaneRun& run, const LaneInputs& in, uint32_t raw, bool is_signed,
                  uint32_t prefix = 0, bool varying = false) {
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto bits = cases::reference_bits(varying ? in.s0[lane] : raw, is_signed);
        ASSERT_TRUE(bits.has_value()) << raw << " lane " << lane;
        const uint32_t value = prefix == 3 ? ~bits.value_or(0) : bits.value_or(0);
        ASSERT_EQ(run.d[lane], in.active[lane] ? value : in.old_d[lane]) << "lane " << lane;
    }
}
std::vector<uint32_t> load_words(const std::filesystem::path& file) {
    std::ifstream input(file, std::ios::binary | std::ios::ate);
    if (!input) {
        ADD_FAILURE() << file;
        return {};
    }
    const auto size = input.tellg();
    if (size <= 0 || size % 4 != 0) {
        ADD_FAILURE() << "unaligned or empty words " << file;
        return {};
    }
    std::vector<uint32_t> words(static_cast<size_t>(size) / sizeof(uint32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(size));
    EXPECT_TRUE(input.good());
    return words;
}
void save_words(const std::filesystem::path& file, const std::vector<uint32_t>& words) {
    std::ofstream out(file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    out.close();
    ASSERT_TRUE(out.good()) << file;
}
void export_form(const std::filesystem::path& root, std::ofstream& manifest, size_t index,
                 const ConversionDef& def, uint32_t selector, uint32_t literal, uint32_t wave,
                 uint32_t prefix, uint32_t mode, bool old = false) {
    const auto program = program_for(def.opcode, selector, literal, wave, mode, prefix, old);
    const uint64_t address = 0x5356000000ull + index * 0x100;
    const auto stem = root / "refused_shaders_20000101-000000Z_synthetic" /
                      ("cs_" + std::to_string(address) + "_" + std::to_string(index));
    const auto bin = std::filesystem::path(stem.string() + ".bin");
    const auto sidecar = std::filesystem::path(stem.string() + ".inputs");
    save_words(bin, program.code);
    ASSERT_TRUE(write_refused_compute_inputs(sidecar, address, &program.table, program.config,
                                             "synthetic exact-constant witness"));
    const auto saved_code = load_words(bin);
    ASSERT_EQ(saved_code, program.code);
    const auto read = read_refused_shader_inputs(sidecar);
    ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
    const auto replay = replay_refused_shader_inputs(saved_code, read);
    ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
    const auto raw = selector == 255 ? literal : cases::inline_word(selector);
    const bool accepted = cases::reference_bits(raw, def.opcode == 0x185).has_value();
    ASSERT_EQ(replay.produced, accepted) << selector << ' ' << raw << ' ' << replay.reject;
    if (accepted) {
        const auto file = root / "modules" / (std::to_string(index) + ".spv");
        save_words(file, replay.spirv);
        const auto saved_module = load_words(file);
        ASSERT_EQ(saved_module, replay.spirv);
        expect_lanes(execute(saved_module, inputs(prefix)), inputs(prefix), raw,
                     def.opcode == 0x185, prefix);
    }
    manifest << stem.filename().string() << '\t' << def.opcode << '\t' << selector << '\t' << raw
             << '\t' << wave << '\t' << prefix << '\t' << mode << '\t' << old << '\t'
             << (accepted ? "accepted" : "refused") << '\n';
}
} // namespace

TEST(IrVectorExactIntegerFloatE64Execution, EveryInlineIntegerAndLiteralBoundaryHasACausalControl) {
    for (const auto& def : vector_exact_integer_float_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            for (uint32_t selector = 128; selector <= 208; ++selector) {
                const uint32_t raw = cases::inline_word(selector);
                const auto want = cases::reference_bits(raw, def.opcode == 0x185);
                const auto module = compile(program_for(def.opcode, selector, 0, wave));
                ASSERT_EQ(!module.empty(), want.has_value()) << selector << ' ' << def.opcode;
                if (want)
                    expect_lanes(execute(module, inputs()), inputs(), raw, def.opcode == 0x185);
            }
            for (uint32_t raw : cases::literals()) {
                const auto want = cases::reference_bits(raw, def.opcode == 0x185);
                const auto module = compile(program_for(def.opcode, 255, raw, wave));
                ASSERT_EQ(!module.empty(), want.has_value()) << raw << ' ' << def.opcode;
                if (want)
                    expect_lanes(execute(module, inputs()), inputs(), raw, def.opcode == 0x185);
            }
        }
}
TEST(IrVectorExactIntegerFloatE64Execution, FullZeroAndPartialExecKeepTheOldLaneWords) {
    for (const auto& def : vector_exact_integer_float_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t prefix = 0; prefix < 4; ++prefix) {
                const auto module =
                    compile(program_for(def.opcode, 255, 0x80000000, wave, UINT32_MAX, prefix));
                ASSERT_FALSE(module.empty());
                const auto in = inputs(prefix);
                expect_lanes(execute(module, in), in, 0x80000000, def.opcode == 0x185, prefix);
            }
}
TEST(IrVectorExactIntegerFloatE64Execution,
     AllProducingModeBytesAndIeeeArmsExecuteTheSameExactBits) {
    for (const auto& def : vector_exact_integer_float_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t mode = 0; mode < 256; ++mode)
                for (uint32_t ieee : {0u, 1u}) {
                    const auto module = compile(program_for(def.opcode, 255, 0xffffff00, wave,
                                                            (mode << 12) | (ieee << 23)));
                    ASSERT_FALSE(module.empty()) << mode << ' ' << ieee;
                    expect_lanes(execute(module, inputs()), inputs(), 0xffffff00,
                                 def.opcode == 0x185);
                }
}
TEST(IrVectorExactIntegerFloatE64Execution, OldE32PathExecutesVaryingExactLaneInputsHonestly) {
    for (const auto& def : vector_exact_integer_float_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t seed : {1u, 17u, 5356u}) {
                auto in = inputs();
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    seed = seed * 1664525u + 1013904223u;
                    const auto shift =
                        static_cast<uint32_t>(lane % (def.opcode == 0x185 ? 8u : 9u));
                    const auto magnitude = (0x800000u | (seed & 0x7fffffu)) << shift;
                    in.s0[lane] = def.opcode == 0x185 && (lane & 1u) ? 0u - magnitude : magnitude;
                }
                const auto module =
                    compile(program_for(def.opcode, 258, 0, wave, UINT32_MAX, 0, true));
                ASSERT_FALSE(module.empty());
                expect_lanes(execute(module, in), in, 0, def.opcode == 0x185, 0, true);
            }
}
TEST(IrVectorExactIntegerFloatE64Execution, InitializedRegisterAndInlineFloatSourcesStayRefused) {
    for (const auto& def : vector_exact_integer_float_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t selector : {50u, 258u, 242u}) {
                auto p = program_for(def.opcode, selector, 0, wave);
                if (selector == 50) p.code.insert(p.code.begin(), {0xbeb203ff, 0x3f800000});
                EXPECT_TRUE(compile(p).empty()) << selector;
                const auto good = compile(program_for(def.opcode, 255, 0x3f800000, wave));
                ASSERT_FALSE(good.empty());
                expect_lanes(execute(good, inputs()), inputs(), 0x3f800000, def.opcode == 0x185);
            }
}
TEST(IrVectorExactIntegerFloatE64Execution, EncodingsAndLiteralConsumptionArePinned) {
    EXPECT_EQ(cases::words(0x185, 129), (std::vector<uint32_t>{0xd5850001, 129}));
    EXPECT_EQ(cases::words(0x186, 255, 0xffffff00),
              (std::vector<uint32_t>{0xd5860001, 255, 0xffffff00}));
    for (const auto& def : vector_exact_integer_float_e64_defs()) {
        const auto words = cases::words(def.opcode, 255, 0x80000000);
        auto code = words;
        code.push_back(0xbf810000);
        std::vector<Rdna2Inst> decoded;
        EXPECT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 2u);
        EXPECT_EQ(decoded.front().len_dwords, 3u);
        EXPECT_TRUE(decoded.front().has_literal);
        EXPECT_EQ(decoded.back().pc, 3u);
        EXPECT_TRUE(decoded.back().is_end);
    }
}
TEST(IrVectorExactIntegerFloatE64Execution, ExportsActualSavedReplaysAndRefusedDomainNeighbors) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "exact-integer-float-e64";
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
    const auto literal_vector = cases::literals();
    const std::set<uint32_t> literals(literal_vector.begin(), literal_vector.end());
    for (const auto& def : vector_exact_integer_float_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            for (uint32_t selector = 128; selector <= 208; ++selector)
                export_form(root, manifest, index++, def, selector, 0, wave, 0, UINT32_MAX);
            for (uint32_t raw : literals)
                export_form(root, manifest, index++, def, 255, raw, wave, 0, UINT32_MAX);
            for (uint32_t prefix = 0; prefix < 4; ++prefix)
                for (bool old : {false, true})
                    export_form(root, manifest, index++, def, 255, 0x80000000, wave, prefix,
                                UINT32_MAX, old);
            for (uint32_t mode = 0; mode < 256; ++mode)
                for (uint32_t ieee : {0u, 1u})
                    export_form(root, manifest, index++, def, 255, 0xffffff00, wave, 0,
                                (mode << 12) | (ieee << 23));
        }
    manifest.close();
    EXPECT_TRUE(manifest.good());
    EXPECT_GT(index, 0u);
}
