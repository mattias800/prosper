// v75 retains raw launch words without upgrading legacy normalized dimensions to original entry
// authority. Codec success is metadata ownership only, never permission to render a capture.
#include "fixtures/portable_entry_exec.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/recompiler/rdna2_entry_exec_constants.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace prosper::gpu;
namespace fixture = prosper::test::portable_entry_exec;
namespace {
uint32_t u32(const std::vector<uint8_t>& bytes, size_t offset) {
    return uint32_t{bytes[offset]} | uint32_t{bytes[offset + 1]} << 8u |
           uint32_t{bytes[offset + 2]} << 16u | uint32_t{bytes[offset + 3]} << 24u;
}
void put32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    for (uint32_t n = 0; n < 4; ++n) bytes[offset + n] = static_cast<uint8_t>(value >> (8u * n));
}
GpuCaptureFile capture(bool known = true) {
    GpuCaptureFile c;
    GpuCapturedCompute compute;
    compute.recompile_config_available = true;
    compute.recompile_config = fixture::config(64);
    compute.recompile_config.lds_bytes = 512;
    if (!known) compute.recompile_config.dispatch_launch = {};
    compute.launch = {64, 1, 1, 64, 1, 1, 1, 1, 1};
    compute.spirv = fixture::compile(fixture::program(64), compute.recompile_config);
    c.computes.push_back(compute);
    return c;
}
std::vector<uint8_t> encode(const GpuCaptureFile& c) {
    std::string error;
    std::vector<uint8_t> bytes;
    EXPECT_TRUE(serialize_gpu_capture(c, bytes, error)) << error;
    return bytes;
}
}   // namespace

TEST(ComputeDispatchLaunchCodec, RealizedComputeRetainsAllRawWordsAndUnknownValues) {
    auto c = capture();
    c.computes[0].recompile_config.dispatch_launch = {ComputeDispatchLaunchState::kAll,
                                                      uint64_t{1} << 32u | 3u,
                                                      {31u << 16u | 64u, 0x12345678u, 0x87654321u},
                                                      {7, 8, 9}};
    const auto bytes = encode(c);
    ASSERT_FALSE(bytes.empty());
    EXPECT_EQ(u32(bytes, 8), 75u);
    GpuCaptureFile back;
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, back, error)) << error;
    ASSERT_EQ(back.computes.size(), 1u);
    EXPECT_EQ(back.computes[0].recompile_config.dispatch_launch,
              c.computes[0].recompile_config.dispatch_launch);
    EXPECT_FALSE(compute_dispatch_has_complete_groups(back.computes[0].recompile_config));
}
TEST(ComputeDispatchLaunchCodec, FailureConfigKeepsIndependentPartialKnownness) {
    auto c = capture(false);
    GpuCapturedOperationFailure failure;
    failure.kind = SubmitOperationKind::Dispatch;
    failure.source_index = 1;
    failure.command_order = 2;
    failure.reason = RealizationFailureReason::ShaderRecompile;
    failure.compute_launch = c.computes[0].launch;
    GpuCapturedStageDiagnostic stage;
    stage.stage = ShaderProgramStage::Compute;
    stage.recompile_config_available = true;
    stage.recompile_config = c.computes[0].recompile_config;
    stage.recompile_config.dispatch_launch = {ComputeDispatchLaunchState::kInitiator |
                                                  ComputeDispatchLaunchState::kNumThreadX,
                                              3u,
                                              {31u << 16u | 64u, 0u, 0u},
                                              {}};
    failure.stages.push_back(stage);
    c.failure_diagnostics.push_back(failure);
    c.operations.push_back({SubmitOperationKind::Dispatch, 1, 2, false});
    GpuCaptureFile back;
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(encode(c), back, error)) << error;
    ASSERT_EQ(back.failure_diagnostics.size(), 1u);
    ASSERT_EQ(back.failure_diagnostics[0].stages.size(), 1u);
    EXPECT_EQ(back.failure_diagnostics[0].stages[0].recompile_config.dispatch_launch,
              stage.recompile_config.dispatch_launch);
    EXPECT_EQ(back.computes[0].recompile_config.dispatch_launch, ComputeDispatchLaunchState{});
}
TEST(ComputeDispatchLaunchCodec, LegacyVersion74LoadsUnknownAndDoesNotAdmitEntryRead) {
    auto bytes = encode(capture());
    ASSERT_GT(bytes.size(), 40u);
    bytes.resize(bytes.size() - 40u); // one v75 record (36 bytes) and its count
    put32(bytes, 8, 74u); // preceding v72/v73/v74 empty tails remain intact
    GpuCaptureFile back;
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, back, error)) << error;
    ASSERT_EQ(back.computes.size(), 1u);
    EXPECT_EQ(back.computes[0].recompile_config.dispatch_launch, ComputeDispatchLaunchState{});
    EXPECT_TRUE(fixture::compile(fixture::program(64), back.computes[0].recompile_config).empty());
}
TEST(ComputeDispatchLaunchCodec, AbsentLaunchKeepsHistoricalWriterVersionAndBytes) {
    const auto bytes = encode(capture(false));
    ASSERT_FALSE(bytes.empty());
    EXPECT_EQ(u32(bytes, 8), 71u);
    GpuCaptureFile back;
    std::string error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, back, error)) << error;
    EXPECT_EQ(encode(back), bytes);
    EXPECT_EQ(back.computes[0].recompile_config.dispatch_launch, ComputeDispatchLaunchState{});
}
TEST(ComputeDispatchLaunchCodec, InvalidPresenceAndTailCountAreRefused) {
    auto c = capture();
    c.computes[0].recompile_config.dispatch_launch.known |= 128u;
    std::vector<uint8_t> bytes;
    std::string error;
    EXPECT_FALSE(serialize_gpu_capture(c, bytes, error));
    const auto good = encode(capture());
    for (bool bad_count : {false, true}) {
        auto bad = good;
        put32(bad, bad.size() - (bad_count ? 40u : 36u), bad_count ? 2u : 255u);
        GpuCaptureFile back;
        EXPECT_FALSE(deserialize_gpu_capture(bad, back, error));
    }
}
TEST(ComputeDispatchLaunchCodec, TruncatedRawLaunchTailCannotBecomeKnown) {
    const auto good = encode(capture());
    for (size_t missing : {size_t{1}, size_t{8}, size_t{35}, size_t{39}}) {
        auto bytes = good;
        bytes.resize(bytes.size() - missing);
        GpuCaptureFile back;
        std::string error;
        EXPECT_FALSE(deserialize_gpu_capture(bytes, back, error)) << missing;
    }
}
