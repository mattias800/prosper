// A conditionally selected legacy writer version does not bound the reader's supported formats.
// Exercise the actual public codec so a supported version cannot masquerade as a future-version
// fixture merely because its required tail was omitted.
#include "fixtures/capture_future_version.hpp"
#include "gpu/capture/gpu_capture.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;

GpuCaptureFile base_capture() {
    GpuCaptureFile capture;
    capture.metadata.title_id = "PPSA00000";
    capture.metadata.width = 8;
    capture.metadata.height = 4;
    capture.metadata.submit_index = 17;
    return capture;
}
} // namespace

TEST(CaptureFutureVersion, UntouchedConditionalWriterHeaderRoundTrips) {
    const auto capture = base_capture();
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    EXPECT_EQ(loaded.metadata.title_id, capture.metadata.title_id);
    EXPECT_EQ(loaded.metadata.width, capture.metadata.width);
    EXPECT_EQ(loaded.metadata.height, capture.metadata.height);
    EXPECT_EQ(loaded.metadata.submit_index, capture.metadata.submit_index);
}

TEST(CaptureFutureVersion, SyntheticFutureHeaderUsesThePublicVersionDiagnostic) {
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(base_capture(), bytes, error)) << error;
    const uint32_t version = prosper::test::install_unsupported_capture_version(bytes);
    GpuCaptureFile rejected;
    EXPECT_FALSE(deserialize_gpu_capture(bytes, rejected, error));
    EXPECT_EQ(error, "unsupported capture version " + std::to_string(version))
        << "a missing supported-format tail is a different reader contract";
}

TEST(CaptureFutureVersion, SupportedFrontFaceFormatRetainsItsRealCompilerInput) {
    auto capture = base_capture();
    GpuCapturedOperationFailure failed;
    failed.source_index = 9;
    failed.command_order = 12;
    failed.reason = RealizationFailureReason::ShaderRecompile;
    failed.vertex_retry_config_available = true;
    failed.fragment_retry_config_available = true;
    failed.has_system_inputs = true;
    failed.system_inputs = {0x1000u, 0x1000u};
    failed.system_inputs.retain_front_face_encoding(true, 1u << 24);
    capture.failure_diagnostics.push_back(failed);
    capture.operations.push_back({SubmitOperationKind::Draw, 9, 12, false});
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    ASSERT_EQ(loaded.failure_diagnostics.size(), 1u);
    EXPECT_EQ(loaded.format_version, 76u) << "the supported front-face tail stays readable";
    EXPECT_EQ(loaded.failure_diagnostics[0].system_inputs, failed.system_inputs);
    EXPECT_EQ(loaded.failure_diagnostics[0].system_inputs.front_face_encoding,
              FragmentFrontFaceEncoding::AllBits);
}
