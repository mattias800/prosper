// Native dynamic WRITELANE writes one selected lane even under empty guest EXEC.
#include "gpu/recompiler/ir/native_writelane_cases.hpp"
#include "gpu/recompiler/ir/eval_lanes.hpp"
#include "gpu/recompiler/ir/ops/vector_cross_lane.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::native_writelane;
namespace {
std::vector<uint32_t> compile(const LaneProgram& p) {
    return recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                             {RecompileDiagnosticStage::Compute, lane_harness_address()});
}
LaneRun execute(const std::vector<uint32_t>& module, const LaneInputs& in, uint32_t wave) {
    std::vector<uint32_t> buffer;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = wave;
    auto out = vm.run_buffers(wave, {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    if (out.size() < wave) return {{}, "missing stored lanes"};
    out.resize(wave);
    return {out, {}};
}
void expect_lanes(const std::vector<uint32_t>& module, const LaneInputs& in, uint32_t selector,
                  cases::Source source, uint32_t wave, bool following = false) {
    ASSERT_FALSE(module.empty());
    const auto run = execute(module, in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), wave);
    Block block;
    Builder ir(block);
    const Value s0 = ir.input(Type::U32), s1 = ir.input(Type::U32);
    const Value d = writelane_b32(ir, s0, s1, wave, [&] { return ir.input(Type::U32); });
    std::vector<InvocationInputs> inputs;
    for (uint32_t lane = 0; lane < wave; ++lane)
        inputs.push_back({cases::source_word(source), selector, in.old_d[lane]});
    const auto evaluated = evaluate_lanes(block, inputs);
    for (uint32_t lane = 0; lane < wave; ++lane) {
        const auto raw = cases::answer(in, lane, selector, source);
        EXPECT_EQ(evaluated[lane][d.index], raw) << lane;
        EXPECT_EQ(run.d[lane], cases::answer(in, lane, selector, source, following)) << lane;
    }
}
std::vector<uint32_t> load_words(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        ADD_FAILURE() << path;
        return {};
    }
    const auto bytes = in.tellg();
    if (bytes <= 0 || bytes % 4 != 0) {
        ADD_FAILURE() << path;
        return {};
    }
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(bytes));
    EXPECT_TRUE(in.good());
    return words;
}
void save_words(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    out.close();
    ASSERT_TRUE(out.good()) << path;
}
void export_form(const std::filesystem::path& root, std::ofstream& manifest, size_t index,
                 uint32_t wave, uint32_t native, uint32_t selector, cases::Source source,
                 uint32_t mask) {
    const auto p = cases::program(wave, selector, source, native, mask);
    const uint64_t address = 0x5370000000ull + index * 0x100;
    const auto stem = root / "refused_shaders_20000101-000000Z_synthetic" /
                      ("cs_" + std::to_string(address) + "_" + std::to_string(index));
    const auto bin = std::filesystem::path(stem.string() + ".bin");
    const auto sidecar = std::filesystem::path(stem.string() + ".inputs");
    save_words(bin, p.code);
    ASSERT_TRUE(write_refused_compute_inputs(sidecar, address, &p.table, p.config,
                                             "synthetic native dynamic WRITELANE witness"));
    const auto code = load_words(bin);
    ASSERT_EQ(code, p.code);
    const auto read = read_refused_shader_inputs(sidecar);
    ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
    const auto replay = replay_refused_shader_inputs(code, read);
    ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
    ASSERT_EQ(replay.produced, native == wave) << replay.reject;
    if (replay.produced) {
        const auto path = root / "modules" / (std::to_string(index) + ".spv");
        save_words(path, replay.spirv);
        const auto saved = load_words(path);
        ASSERT_EQ(saved, replay.spirv);
        expect_lanes(saved, cases::inputs(wave, mask), selector, source, wave);
    }
    manifest << stem.filename().string() << '\t' << wave << '\t' << native << '\t' << selector
             << '\t' << static_cast<uint32_t>(source) << '\t' << mask << '\n';
}
} // namespace
TEST(IrVectorWritelaneExecution, ScalarInlineAndLiteralDataSelectExactlyOneLane) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t selector : cases::selectors(wave))
            for (auto source :
                 {cases::Source::Scalar, cases::Source::InlineMinusOne, cases::Source::Literal})
                expect_lanes(compile(cases::program(wave, selector, source, wave)),
                             cases::inputs(wave), selector, source, wave);
}
TEST(IrVectorWritelaneExecution, InactiveSelectedLaneIsWrittenEvenWhenExecIsEmpty) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t mask : {0u, 1u, 2u})
            for (uint32_t selector : {0u, wave - 1u}) {
                const auto in = cases::inputs(wave, mask);
                if (mask != 1) ASSERT_FALSE(in.active[selector]);
                const auto p = cases::program(wave, selector, cases::Source::Scalar, wave, mask);
                expect_lanes(compile(p), in, selector, cases::Source::Scalar, wave);
            }
}
TEST(IrVectorWritelaneExecution, FollowingOrdinaryValuStillObservesExec) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t mask : {0u, 1u, 2u}) {
            const auto p = cases::program(wave, 0, cases::Source::Scalar, wave, mask, true);
            expect_lanes(compile(p), cases::inputs(wave, mask), 0, cases::Source::Scalar, wave,
                         true);
        }
}
TEST(IrVectorWritelaneExecution, HarnessDetectsAnUnmaskedFollowingInstruction) {
    for (uint32_t wave : {32u, 64u}) {
        const auto in = cases::inputs(wave);
        const auto p = cases::program(wave, 0, cases::Source::Scalar, wave, 1, true);
        const auto run = execute(compile(p), in, wave);
        ASSERT_TRUE(run.error.empty()) << run.error;
        ASSERT_EQ(run.d.size(), wave);
        size_t inactive = 0, corrupted = 0;
        for (uint32_t lane = 0; lane < wave; ++lane) {
            const auto raw = cases::answer(in, lane, 0, cases::Source::Scalar);
            EXPECT_EQ(run.d[lane], ~raw);
            if (!in.active[lane]) {
                ++inactive;
                corrupted += run.d[lane] != raw;
            }
        }
        ASSERT_GT(inactive, 0u);
        EXPECT_EQ(corrupted, inactive);
    }
}
TEST(IrVectorWritelaneExecution, PortableDynamicAndVectorSelectorsStillRefuseBesideNativeSuccess) {
    for (uint32_t wave : {32u, 64u}) {
        EXPECT_TRUE(compile(cases::program(wave, 5, cases::Source::Scalar, 0)).empty());
        auto p = cases::program(wave, 5, cases::Source::Scalar, wave);
        const auto at = std::find(p.code.begin(), p.code.end(), 0xd7610001u);
        ASSERT_NE(at, p.code.end());
        *(at + 1) = 20u | (258u << 9);
        EXPECT_TRUE(compile(p).empty());
        expect_lanes(compile(cases::program(wave, 5, cases::Source::Scalar, wave)),
                     cases::inputs(wave), 5, cases::Source::Scalar, wave);
    }
}
TEST(IrVectorWritelaneExecution, ExportsSavedReplayedAndExecutedWitnessPrograms) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never changes its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "native-writelane";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error);
    }
    ASSERT_FALSE(fs::exists(root)) << "export requires a new witness directory";
    ASSERT_TRUE(fs::create_directories(root / "modules"));
    ASSERT_TRUE(fs::create_directory(root / "refused_shaders_20000101-000000Z_synthetic"));
    std::ofstream manifest(root / "manifest.tsv");
    size_t count = 0;
    for (uint32_t wave : {32u, 64u})
        for (uint32_t native : {wave, 0u})
            for (uint32_t selector : cases::selectors(wave))
                for (auto source :
                     {cases::Source::Scalar, cases::Source::InlineMinusOne, cases::Source::Literal})
                    for (uint32_t mask : {0u, 1u, 2u})
                        export_form(root, manifest, count++, wave, native, selector, source, mask);
    manifest.close();
    EXPECT_TRUE(manifest.good());
    EXPECT_EQ(count, 288u);
}
