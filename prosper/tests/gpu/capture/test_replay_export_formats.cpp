// test_replay_export_formats -- #4703 review B2: gpu_replay's fragment recompiles must use the
// captured SPI_SHADER_COL_FORMAT and colour-target classes, exactly as the live draw does.
//
// gpu_replay regenerates fragment modules from captured raw ISA on two routes: `--recompile-raw`
// (the F9 -> replay iteration loop) and `--retry-failed-stage`. Both called recompile_fragment with
// the historical defaults, so a frozen Kena frame replayed the very bug #4703 fixes: f16-decoded
// UINT16 exports into a float-typed output. This test writes a one-draw capture whose pipeline is
// Kena's lighting-channel writer (R16_UINT target, COL_FORMAT 0x7), runs the real gpu_replay on
// both routes, and requires the regenerated module to be byte-identical to the one the live
// compiler builds with the derived formats -- and different from the {} module.
//
// Mutation: drop `fragment_export_formats(...)` from either recompile site in gpu_replay.cpp and
// that route's arm goes red. No GPU is used: --inspect-only and the stage retry never submit.
#include <gtest/gtest.h>

#include "fixtures/test_scratch.h"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/state/fragment_export_state.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace gpu = prosper::gpu;

namespace {

constexpr uint32_t kVertex[] = {
    0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02f6u,
    0x7e0c02f2u, 0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};
// v0 = (1, 0xBEEF), v1 = (3, 2); exp mrt0 v0, v1 compr done vm.
constexpr uint32_t kWriter[] = {
    0x7e0002ffu, 0xbeef0001u, 0x7e0202ffu, 0x00020003u, 0xf8001c0fu, 0x00000100u, 0xbf810000u,
};
constexpr uint32_t kR16Uint = 74u;   // VK_FORMAT_R16_UINT

gpu::GpuCaptureRawShaderVersion raw_shader(const uint32_t* words, size_t count) {
    gpu::GpuCaptureRawShaderVersion result;
    result.words.assign(words, words + count);
    result.has_endpgm = true;
    result.content_hash =
        gpu::gpu_capture_hash(reinterpret_cast<const uint8_t*>(words), count * sizeof(uint32_t));
    return result;
}

gpu::ResolvedPipelineState kena_writer_pipeline() {
    gpu::ResolvedPipelineState ps{};
    ps.topology = 3;
    ps.color0_format = kR16Uint;
    ps.color_write_mask = 0x1;
    ps.color_targets[0].format = kR16Uint;
    ps.color_targets[0].write_mask = 0x1;
    ps.spi_shader_col_format = 0x7u;
    return ps;
}

std::vector<uint32_t> expected_module(const gpu::FragmentExportFormats& formats) {
    return gpu::recompile_fragment(kWriter, std::size(kWriter), nullptr, nullptr, UINT32_MAX,
                                   nullptr, false, {gpu::RecompileDiagnosticStage::Fragment, 0}, {},
                                   nullptr, {}, {}, formats);
}

std::vector<uint32_t> read_words(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    std::vector<uint32_t> words(bytes.size() / 4);
    if (!words.empty()) std::memcpy(words.data(), bytes.data(), words.size() * 4);
    return words;
}

// The gpu_replay binary this build produced (set by CMake), or null when it was not built.
const char* replay_binary() {
    return std::getenv("PROSPER_GPU_REPLAY_BIN");   // NOLINT(concurrency-mt-unsafe): no setenv here
}

// Runs the real gpu_replay, single-threaded, on paths this test wrote.
int run_replay(const std::string& arguments, const std::filesystem::path& log) {
    const char* replay = replay_binary();
    if (!replay) return -1;
    const std::string command =
        std::string("\"") + replay + "\" " + arguments + " > \"" + log.string() + "\" 2>&1";
    // NOLINTNEXTLINE(bugprone-command-processor,concurrency-mt-unsafe)
    return std::system(command.c_str());
}

}   // namespace

class ReplayExportFormats : public ::testing::Test {
protected:
    void SetUp() override {
        if (!replay_binary()) GTEST_SKIP() << "gpu_replay not built";
        directory = prosper_test::test_scratch_dir() / "replay-export-formats";
        std::filesystem::create_directories(directory);
        live = expected_module(gpu::fragment_export_formats(kena_writer_pipeline()));
        legacy = expected_module({});
        ASSERT_FALSE(live.empty());
        ASSERT_NE(live, legacy) << "the fixture must be one whose module the formats change";
    }
    std::filesystem::path directory;
    std::vector<uint32_t> live, legacy;
};

TEST_F(ReplayExportFormats, RecompileRawUsesCapturedFormats) {
    gpu::GpuCaptureFile capture;
    capture.metadata.width = capture.metadata.height = 1;
    capture.raw_shader_versions.push_back(raw_shader(kVertex, std::size(kVertex)));
    capture.raw_shader_versions.push_back(raw_shader(kWriter, std::size(kWriter)));
    gpu::GpuCapturedDraw draw;
    draw.vs = gpu::recompile_vertex(kVertex, std::size(kVertex));
    draw.fs = legacy;   // a stale stored module: --recompile-raw must replace it
    draw.ps = kena_writer_pipeline();
    draw.draw_index = 0;
    draw.command_order = 1;
    draw.vertex_count = draw.raw_draw_count = 3;
    draw.color0_width = draw.color0_height = 1;
    draw.has_pixel_inputs = draw.has_system_inputs = false;
    draw.vs_raw_shader_index = 0;
    draw.fs_raw_shader_index = 1;
    capture.draws.push_back(draw);
    capture.operations.push_back({gpu::SubmitOperationKind::Draw, 0, 1, true});
    const auto path = directory / "recompile-raw.prgcap";
    std::string error;
    ASSERT_TRUE(gpu::write_gpu_capture(path.string(), capture, error)) << error;

    const auto module = directory / "recompile-raw-fs.spv";
    std::filesystem::remove(module);
    const int rc = run_replay("--inspect-only --recompile-raw --dump-shader 0:fs \"" +
                                  module.string() + "\" \"" + path.string() + "\"",
                              directory / "recompile-raw.log");
    ASSERT_EQ(rc, 0) << "see " << (directory / "recompile-raw.log").string();
    const auto words = read_words(module);
    EXPECT_EQ(words, live) << "--recompile-raw built the module without the captured formats";
    EXPECT_NE(words, legacy);
}

TEST_F(ReplayExportFormats, RetryFailedStageUsesCapturedFormats) {
    gpu::GpuCaptureFile failed;
    failed.metadata.width = failed.metadata.height = 1;
    failed.failure_diagnostics_available = true;
    failed.raw_shader_versions.push_back(raw_shader(kWriter, std::size(kWriter)));
    failed.operations.push_back({gpu::SubmitOperationKind::Draw, 7, 1, false});
    gpu::GpuCapturedOperationFailure failure;
    failure.source_index = 7;
    failure.command_order = 1;
    failure.reason = gpu::RealizationFailureReason::ShaderRecompile;
    failure.pipeline_present = true;
    failure.pipeline = kena_writer_pipeline();
    failure.vertex_retry_config_available = true;
    failure.fragment_retry_config_available = true;
    gpu::GpuCapturedStageDiagnostic stage;
    stage.stage = gpu::ShaderProgramStage::Fragment;
    stage.program_addr = 0x7000u;
    stage.raw_shader_index = 0;
    failure.stages.push_back(stage);
    failed.failure_diagnostics.push_back(failure);
    const auto path = directory / "retry.prgcap";
    std::string error;
    ASSERT_TRUE(gpu::write_gpu_capture(path.string(), failed, error)) << error;

    const auto module = directory / "retry-fs.spv";
    std::filesystem::remove(module);
    const int rc = run_replay("--retry-failed-stage 0:0 --retry-failed-stage-spv \"" +
                                  module.string() + "\" \"" + path.string() + "\"",
                              directory / "retry.log");
    ASSERT_EQ(rc, 0) << "see " << (directory / "retry.log").string();
    const auto words = read_words(module);
    // The retry passes the interpolation layout derived from the captured ABI, as live does.
    const auto interpolation = gpu::fragment_interpolation_layout(kWriter, std::size(kWriter));
    const auto expected = gpu::recompile_fragment(
        kWriter, std::size(kWriter), nullptr, nullptr, UINT32_MAX, &interpolation, false,
        {gpu::RecompileDiagnosticStage::Fragment, 0x7000u}, {}, nullptr, {}, {},
        gpu::fragment_export_formats(kena_writer_pipeline()));
    EXPECT_EQ(words, expected) << "--retry-failed-stage built the module without the formats";
    EXPECT_NE(words, legacy);
}
