// Allocation ownership must survive caches and replay; raw RSRC1 alone is not that contract.
#include "fixtures/allocated_movrels.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/compiler_choices.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::allocated_movrels;
GpuCaptureFile capture() {
    GpuCaptureFile c;
    GpuCapturedCompute compute;
    compute.recompile_config_available = true;
    compute.required_subgroup_size = 64;
    compute.recompile_config = fixture::config(23);
    compute.launch = {64, 1, 1, 64, 1, 1, 1, 1, 1};
    compute.spirv = fixture::compile(23);
    c.computes.push_back(compute);
    GpuCapturedOperationFailure failure;
    failure.kind = SubmitOperationKind::Dispatch;
    failure.source_index = 1;
    failure.command_order = 2;
    failure.reason = RealizationFailureReason::ShaderRecompile;
    failure.compute_launch = {32, 1, 1, 32, 1, 1, 1, 1, 1};
    GpuCapturedStageDiagnostic stage;
    stage.stage = ShaderProgramStage::Compute;
    stage.recompile_config_available = true;
    stage.recompile_config = fixture::config(0, 32);
    failure.stages.push_back(stage);
    c.failure_diagnostics.push_back(failure);
    c.operations.push_back({SubmitOperationKind::Dispatch, 1, 2, false});
    return c;
}
std::vector<uint8_t> encode(const GpuCaptureFile& c) {
    std::vector<uint8_t> bytes;
    std::string error;
    EXPECT_TRUE(serialize_gpu_capture(c, bytes, error)) << error;
    return bytes;
}
void put32(std::vector<uint8_t>& bytes, size_t at, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes[at + i] = static_cast<uint8_t>(value >> (8 * i));
}
} // namespace
TEST(ComputeVgprAllocation, SuccessfulAndFailedCaptureInputsRoundTrip) {
    const auto bytes = encode(capture());
    GpuCaptureFile back;
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, back, error)) << error;
    ASSERT_EQ(back.computes.size(), 1u);
    ASSERT_EQ(back.failure_diagnostics.size(), 1u);
    ASSERT_EQ(back.failure_diagnostics[0].stages.size(), 1u);
    EXPECT_EQ(back.format_version, 80u);
    EXPECT_EQ(back.computes[0].recompile_config.allocated_vgprs, 24u);
    EXPECT_EQ(back.failure_diagnostics[0].stages[0].recompile_config.allocated_vgprs, 24u);
    EXPECT_EQ(encode(back), bytes);
}
TEST(ComputeVgprAllocation, Version79CannotInventAllocationFromIdenticalRawRsrc1) {
    auto bytes = encode(capture());
    ASSERT_GT(bytes.size(), 20u);
    bytes.resize(bytes.size() - 20); // one compute and one diagnostic containing one stage
    put32(bytes, 8, 79);
    GpuCaptureFile back;
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, back, error)) << error;
    ASSERT_EQ(back.computes.size(), 1u);
    EXPECT_EQ(back.computes[0].recompile_config.allocated_vgprs, 0u);
    EXPECT_EQ(back.failure_diagnostics[0].stages[0].recompile_config.allocated_vgprs, 0u);
    EXPECT_EQ(back.computes[0].recompile_config.compute_pgm_rsrc1,
              fixture::config(23).compute_pgm_rsrc1);
}
TEST(ComputeVgprAllocation, InvalidCountsTruncationAndMismatchedGranularityRefuse) {
    const auto good = encode(capture());
    ASSERT_GT(good.size(), 20u);
    for (size_t offset : {20u, 16u, 12u, 8u, 4u}) {
        auto bad = good;
        put32(bad, bad.size() - offset, 999);
        GpuCaptureFile back;
        std::string error;
        EXPECT_FALSE(deserialize_gpu_capture(bad, back, error)) << offset;
    }
    auto truncated = good;
    truncated.pop_back();
    GpuCaptureFile back;
    std::string error;
    EXPECT_FALSE(deserialize_gpu_capture(truncated, back, error));
    for (bool config_missing : {false, true}) {
        auto bad = capture();
        if (config_missing)
            bad.computes[0].recompile_config_available = false;
        else
            bad.computes[0].recompile_config.allocated_vgprs = 20;
        std::vector<uint8_t> output;
        EXPECT_FALSE(serialize_gpu_capture(bad, output, error));
    }
}
TEST(ComputeVgprAllocation, UnknownContractsRetainTheirOlderWriterFormat) {
    auto c = capture();
    c.computes[0].recompile_config.allocated_vgprs = 0;
    c.failure_diagnostics[0].stages[0].recompile_config.allocated_vgprs = 0;
    GpuCaptureFile back;
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(encode(c), back, error)) << error;
    EXPECT_LT(back.format_version, 80u);
}
TEST(ComputeVgprAllocation, CacheKeysAndSidecarsDistinguishTheAllocationContract) {
    auto known = fixture::config(23);
    auto unknown = known;
    unknown.allocated_vgprs = 0;
    ShaderCompileKey a, b;
    a.stage = b.stage = ShaderProgramStage::Compute;
    a.has_compute_config = b.has_compute_config = true;
    a.code = b.code = std::make_shared<const std::vector<uint32_t>>(fixture::program());
    a.code_hash = b.code_hash = hash_shader_code(*a.code);
    populate_compute_shader_key(a, known);
    populate_compute_shader_key(b, unknown);
    EXPECT_FALSE(a == b);
    EXPECT_NE(ShaderCompileKeyHash::compute_impl<true>(a),
              ShaderCompileKeyHash::compute_impl<true>(b));
    EXPECT_NE(ShaderCompileKeyHash::compute_impl<false>(a),
              ShaderCompileKeyHash::compute_impl<false>(b));
    EXPECT_NE(refused_compute_inputs_identity(nullptr, known),
              refused_compute_inputs_identity(nullptr, unknown));
    const auto path = prosper_test::test_scratch_dir() / "compute-vgpr-allocation.inputs";
    for (const auto* c : {&known, &unknown}) {
        ASSERT_TRUE(write_refused_compute_inputs(path, 0x503900u, nullptr, *c, "fixture"));
        EXPECT_EQ(read_refused_shader_inputs(path).compute.allocated_vgprs, c->allocated_vgprs);
        std::filesystem::remove(path);
    }
    known.choices.set(CompilerChoice::Float32Rte, 0);
    ASSERT_TRUE(write_refused_compute_inputs(path, 0x503901u, nullptr, known, "fixture"));
    const auto back = read_refused_shader_inputs(path);
    EXPECT_EQ(back.compute.allocated_vgprs, 24u);
    EXPECT_EQ(back.compute.choices.get(CompilerChoice::Float32Rte), 0);
    std::filesystem::remove(path);
}
