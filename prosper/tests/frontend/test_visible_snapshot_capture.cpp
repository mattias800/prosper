// #5078: a visible snapshot must retain the exact successfully presented CPU publication,
// including origin, and sample it again on cadence when no newer publication arrives.
#include "snapshot_capture.hpp"
#include "build_revision.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>

#ifdef PROSPER_SNAPSHOT_PNG
// Decode with a separate implementation: dimensions and RGBA must describe the shown publication,
// rather than merely agree with the manifest emitted by the same capture code.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "stb_image.h"
#endif

namespace {
using namespace prosper;
using namespace prosper::frontend;
gpu::PresentFrameLease frame(gpu::PresentSource source, uint64_t identity = 7) {
    gpu::PresentFrameLease out;
    out.source = source;
    out.frame_seq = identity;
    out.guest_present_count = 3;
    out.width = 2;
    out.height = 1;
    out.rgba =
        std::make_shared<const std::vector<uint8_t>>(std::vector<uint8_t>{1, 2, 3, 0, 4, 5, 6, 0});
    return out;
}
}   // namespace

TEST(VisibleSnapshotCapture, RetainsTheCompletedPublicationWithItsOrigin) {
    PresentedSnapshotFrame saved;
    auto original = frame(gpu::PresentSource::GuestScanout);
    ASSERT_TRUE(saved.retain(original, true));
    original.rgba.reset();
    EXPECT_EQ(saved.get().source, gpu::PresentSource::GuestScanout);
    EXPECT_EQ(saved.get().frame_seq, 7u);
    EXPECT_EQ(saved.get().guest_present_count, 3u);
    EXPECT_EQ(saved.get().width, 2u);
    EXPECT_EQ((*saved.get().rgba)[0], 1u);
}

TEST(VisibleSnapshotCapture, ActualPublisherCarriesOriginWithTheImmutableLease) {
    for (const auto origin :
         {gpu::PresentFrameOrigin::Composited, gpu::PresentFrameOrigin::GuestScanout}) {
        gpu::present_reset();
        const std::vector<uint8_t> pixels{1, 2, 3, 0, 4, 5, 6, 0};
        gpu::present_write_frame(pixels.data(), 2, 1, origin);
        gpu::PresentFrameLease published;
        ASSERT_TRUE(gpu::present_acquire_rendered_frame(published));
        PresentedSnapshotFrame saved;
        ASSERT_TRUE(saved.retain(published, true));
        gpu::present_reset();
        EXPECT_EQ(saved.get().source, origin == gpu::PresentFrameOrigin::Composited
                                          ? gpu::PresentSource::Rendered
                                          : gpu::PresentSource::GuestScanout);
        EXPECT_EQ(*saved.get().rgba, pixels);
    }
}

TEST(VisibleSnapshotCapture, FailedPresentationCannotReplaceTheLastShownPixels) {
    PresentedSnapshotFrame saved;
    ASSERT_TRUE(saved.retain(frame(gpu::PresentSource::Rendered), true));
    EXPECT_FALSE(saved.retain(frame(gpu::PresentSource::Rendered, 8), false));
    EXPECT_EQ(saved.get().frame_seq, 7u);
}

TEST(VisibleSnapshotCapture, SuccessfulUnownedPresentationCannotKeepAnOlderGoodFrame) {
    for (const auto source : {gpu::PresentSource::None, gpu::PresentSource::RawScanout}) {
        PresentedSnapshotFrame saved;
        ASSERT_TRUE(saved.retain(frame(gpu::PresentSource::Rendered), true));
        EXPECT_FALSE(saved.retain(frame(source, 8), true));
        EXPECT_FALSE(saved.available());
    }
}

TEST(VisibleSnapshotCapture, IncompletePublicationNeverBecomesAZeroFilledImage) {
    PresentedSnapshotFrame saved;
    auto missing = frame(gpu::PresentSource::Rendered);
    missing.rgba = std::make_shared<const std::vector<uint8_t>>(4);
    EXPECT_FALSE(saved.retain(missing, true));
    EXPECT_FALSE(saved.available());
}

TEST(VisibleSnapshotCapture, StalePixelsRemainDueOnTheOriginalWallClockCadence) {
    SnapshotCaptureClock clock(5, 40, 3);
    EXPECT_FALSE(clock.due(4.9));
    ASSERT_TRUE(clock.due(5));
    clock.sampled(5);
    EXPECT_FALSE(clock.due(9.9));
    EXPECT_TRUE(clock.due(10)); // no new frame/flip is required
    clock.sampled(10);
    EXPECT_TRUE(clock.due(15));
    clock.sampled(15);
    EXPECT_TRUE(clock.complete());
    EXPECT_FALSE(clock.timed_out(40));
}

TEST(VisibleSnapshotCapture, DelayedPublicationDoesNotCreateCatchupSamples) {
    SnapshotCaptureClock clock(2, 30, 3);
    EXPECT_TRUE(clock.due(8));
    clock.sampled(8); // screenshot's last_cap is the actual attempt time
    EXPECT_FALSE(clock.due(8.1));
    EXPECT_TRUE(clock.due(10));
    EXPECT_FALSE(clock.complete());
    EXPECT_FALSE(clock.timed_out(30)); // legacy screenshot tests elapsed > timeout
    EXPECT_TRUE(clock.timed_out(30.1));
}

TEST(VisibleSnapshotCapture, MissingRequiredCaptureOptionsRefuse) {
    SnapshotCaptureRequest request;
    request.enabled = true;
    std::string error;
    EXPECT_FALSE(request.valid(error));
    request.seconds = 1;
    request.timeout = 40;
    request.count = 30;
    request.output = "captures";
    request.manifest = "captures/capture.jsonl";
    error.clear();
    EXPECT_TRUE(request.valid(error));
}

#ifdef PROSPER_SNAPSHOT_PNG
namespace {
class VisibleSnapshotFiles : public testing::Test {
protected:
    void SetUp() override {
        directory = prosper_test::test_scratch_path(
            testing::UnitTest::GetInstance()->current_test_info()->name());
        capture.request.enabled = true;
        capture.request.require_composited = true;
        capture.request.seconds = 5;
        capture.request.timeout = 40;
        capture.request.count = 3;
        capture.request.output = directory.string();
        capture.request.manifest = (directory / "capture.jsonl").string();
        screenshot::publish_guest_outcome(0, 0, 0, "synthetic returned guest");
    }
    void TearDown() override {
        capture.finish();
        screenshot::publish_guest_outcome(0, 0, 0, "synthetic returned guest");
    }
    bool start() {
        std::string error;
        const bool started = capture.start("synthetic", error);
        EXPECT_TRUE(started) << error;
        return started;
    }
    std::vector<std::string> records(const std::string& type) {
        std::ifstream input(capture.request.manifest);
        std::vector<std::string> result;
        std::string line;
        while (std::getline(input, line)) {
            if (line.find("\"type\":\"" + type + "\"") != std::string::npos) result.push_back(line);
        }
        return result;
    }
    static void contains(const std::string& text, const std::string& token) {
        EXPECT_NE(text.find(token), std::string::npos) << token << " absent from " << text;
    }
    SnapshotCapture capture;
    std::filesystem::path directory;
};
} // namespace

TEST_F(VisibleSnapshotFiles, CompletedShownPixelsRemainStaleSamplesAndDecodeByteExactly) {
    ASSERT_TRUE(start());
    capture.retain(frame(gpu::PresentSource::Rendered), true);
    EXPECT_FALSE(capture.poll_elapsed(4.9, true));
    EXPECT_FALSE(capture.poll_elapsed(5, true));
    EXPECT_FALSE(capture.poll_elapsed(10, true));
    EXPECT_TRUE(capture.poll_elapsed(15, true));
    ASSERT_EQ(capture.finish(), 0);
    EXPECT_EQ(capture.finish(), 0); // finishing twice must not append a second summary
    const auto samples = records("sample");
    ASSERT_EQ(samples.size(), 3u);
    contains(samples[0], "\"source\":\"composited\"");
    contains(samples[0], "\"frame_seq\":7");
    contains(samples[0], "\"present_count\":3");
    contains(samples[0], "\"front_index\":-1");
    contains(samples[1], "\"source_advanced\":false");
    contains(samples[1], "\"pixel_identical\":true");
    contains(samples[2], "\"stale_seconds\":10.000000");
    contains(samples[2], "\"pixel_stale_seconds\":10.000000");
    const auto summary = records("summary");
    ASSERT_EQ(summary.size(), 1u);
    contains(summary[0], "\"saved\":3");
    contains(summary[0], "\"rendered_samples\":3");
    contains(summary[0], "\"stop_reason\":\"request-satisfied\"");
    contains(summary[0], "\"status\":\"ok\"");
    const auto frontend = records("frontend");
    ASSERT_EQ(frontend.size(), 1u);
    contains(frontend[0], "\"name\":\"prosper-app\"");
    contains(frontend[0], "\"present_path\":\"cpu\"");
    contains(frontend[0],
             std::string("\"source_identity\":\"") + embedded_build_source_identity() + "\"");
    const auto run = records("run");
    ASSERT_EQ(run.size(), 1u);
    contains(run[0], std::string("\"build_revision\":\"") + embedded_build_revision() + "\"");
    contains(run[0], "\"stop_after_guest_fault\":false");
    const auto conditions = records("conditions");
    ASSERT_EQ(conditions.size(), 1u);
    contains(conditions[0], "\"harness\":\"prosper-app\"");
    contains(conditions[0], "\"present_path\":\"cpu_publication\"");
    contains(conditions[0], "\"gpu_known\":false");

    for (int sample = 0; sample < 3; ++sample) {
        std::ifstream file(directory / ("sample_" + std::to_string(sample) + ".png"),
                           std::ios::binary);
        const std::vector<uint8_t> png{std::istreambuf_iterator<char>(file), {}};
        int width = 0, height = 0, channels = 0;
        stbi_uc* decoded = stbi_load_from_memory(png.data(), static_cast<int>(png.size()), &width,
                                                 &height, &channels, 4);
        ASSERT_NE(decoded, nullptr) << stbi_failure_reason();
        EXPECT_EQ(width, 2);
        EXPECT_EQ(height, 1);
        const auto bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4;
        const std::vector<uint8_t> pixels(decoded, decoded + bytes);
        stbi_image_free(decoded);
        EXPECT_EQ(pixels, (std::vector<uint8_t>{1, 2, 3, 255, 4, 5, 6, 255}));
    }
}

TEST_F(VisibleSnapshotFiles, NeverPublishedOrNeverShownCannotSatisfyTheRequest) {
    ASSERT_TRUE(start());
    capture.retain(frame(gpu::PresentSource::Rendered), false);
    EXPECT_FALSE(capture.poll_elapsed(39, true));
    EXPECT_TRUE(capture.poll_elapsed(40.1, true));
    EXPECT_EQ(capture.finish(), 1);
    EXPECT_TRUE(records("sample").empty());
    const auto summary = records("summary");
    ASSERT_EQ(summary.size(), 1u);
    contains(summary[0], "\"saved\":0");
    contains(summary[0], "\"timed_out\":true");
    contains(summary[0], "\"status\":\"FAILED\"");
    EXPECT_FALSE(std::filesystem::exists(directory / "sample_0.png"));
}

TEST_F(VisibleSnapshotFiles, PublicationBeyondTheDeadlineCannotBecomeASuccess) {
    capture.request.count = 1;
    capture.request.timeout = 5;
    ASSERT_TRUE(start());
    capture.retain(frame(gpu::PresentSource::Rendered), true);
    EXPECT_TRUE(capture.poll_elapsed(5.1, true));
    EXPECT_EQ(capture.finish(), 1);
    EXPECT_TRUE(records("sample").empty());
    EXPECT_FALSE(std::filesystem::exists(directory / "sample_0.png"));
}

TEST_F(VisibleSnapshotFiles, HiddenWindowCannotYieldSuccessfulCaptureEvidence) {
    ASSERT_TRUE(start());
    capture.retain(frame(gpu::PresentSource::Rendered), true);
    EXPECT_TRUE(capture.poll_elapsed(5, false));
    EXPECT_EQ(capture.finish(), 1);
    EXPECT_TRUE(records("sample").empty());
    const auto summary = records("summary");
    ASSERT_EQ(summary.size(), 1u);
    contains(summary[0], "\"stop_reason\":\"frontend-stopped\"");
}

TEST_F(VisibleSnapshotFiles, GuestScanoutRemainsGuestScanoutAndFailsCompositedRequirement) {
    capture.request.count = 1;
    ASSERT_TRUE(start());
    capture.retain(frame(gpu::PresentSource::GuestScanout), true);
    EXPECT_TRUE(capture.poll_elapsed(5, true));
    EXPECT_EQ(capture.finish(), 1);
    const auto samples = records("sample");
    ASSERT_EQ(samples.size(), 1u);
    contains(samples[0], "\"source\":\"guest_scanout\"");
    const auto summary = records("summary");
    ASSERT_EQ(summary.size(), 1u);
    contains(summary[0], "\"rendered_samples\":0");
    contains(summary[0], "\"guest_scanout_samples\":1");
}

TEST_F(VisibleSnapshotFiles, FailedPngWriteNeverCountsAnUnsavedSample) {
    capture.request.count = 1;
    ASSERT_TRUE(start());
    std::filesystem::create_directory(directory / "sample_0.png");
    capture.retain(frame(gpu::PresentSource::Rendered), true);
    EXPECT_TRUE(capture.poll_elapsed(5, true));
    EXPECT_EQ(capture.finish(), 1);
    EXPECT_TRUE(records("sample").empty());
    const auto summary = records("summary");
    ASSERT_EQ(summary.size(), 1u);
    contains(summary[0], "\"saved\":0");
    contains(summary[0], "\"status\":\"FAILED\"");
}

TEST_F(VisibleSnapshotFiles, PrimaryGuestFaultRemainsAFailureEvenAfterTheLastSample) {
    capture.request.count = 1;
    ASSERT_TRUE(start());
    capture.retain(frame(gpu::PresentSource::Rendered), true);
    EXPECT_TRUE(capture.poll_elapsed(5, true));
    screenshot::publish_guest_outcome(2, 0x1234, 0x5678, "synthetic fault");
    EXPECT_EQ(capture.finish(), 1);
    const auto summary = records("summary");
    ASSERT_EQ(summary.size(), 1u);
    contains(summary[0], "\"guest_state\":\"faulted\"");
    contains(summary[0], "\"status\":\"GUEST-FAULT\"");
}

TEST_F(VisibleSnapshotFiles, PrimaryGuestFaultDoesNotShortenTheRequestedWallClockWindow) {
    ASSERT_TRUE(start());
    capture.retain(frame(gpu::PresentSource::Rendered), true);
    screenshot::publish_guest_outcome(2, 0x1234, 0x5678, "synthetic fault");
    EXPECT_FALSE(capture.poll_elapsed(5, true));
    EXPECT_TRUE(capture.poll_elapsed(40.1, true));
    EXPECT_EQ(capture.finish(), 1);
    const auto summary = records("summary");
    ASSERT_EQ(summary.size(), 1u);
    contains(summary[0], "\"saved\":1");
    contains(summary[0], "\"stop_reason\":\"timeout\"");
    contains(summary[0], "\"status\":\"GUEST-FAULT\"");
}
#endif
