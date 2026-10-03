// The macOS AvPlayer backend against a committed H.264 Main + AAC stereo clip (1 s, 128x96, 30 fps,
// 48 kHz) -- the same codec pair as Hollow Knight: Silksong's cinematics.
//
// The pixel check is the discriminating one: H.264 decoding is bit-exact in every conformant
// decoder, so VideoToolbox's NV12 output must hash identically to an independent decode. The
// expected value is FNV-1a 64 over all 30 packed NV12 frames from the host `ffmpeg -pix_fmt nv12`
// decode of the same file. A backend that returned the wrong plane order, a cropped or padded
// stride, full-range chroma, or a dropped/duplicated frame cannot produce it.
#include "videotoolbox_backend.hpp"

#include <VideoToolbox/VideoToolbox.h>
#include <gtest/gtest.h>

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

using namespace prosper::video;

namespace {

constexpr uint64_t kExpectedNv12Fnv = 0xb31d950966c27413ull;
constexpr unsigned kExpectedFrames = 30;
const char* const kAsset = PROSPER_TEST_H264_AAC_ASSET;
// The crop path: coded 128x96, bitstream crop right/bottom by 2 -> displayed 126x94, 15 pictures,
// no audio. Same independent-decode hash method (ffmpeg applies the crop).
constexpr uint64_t kCropNv12Fnv = 0x9297c79ac13bb167ull;
const char* const kCropAsset = PROSPER_TEST_H264_CROP_ASSET;
// MPEG-4 Part 2: VideoToolbox has NO hardware decoder for it (measured: a session that requires
// hardware fails with kVTCouldNotFindVideoDecoderErr), only a software one -- so it discriminates
// the hardware policy on every host, hardware-capable or not.
const char* const kSoftwareOnlyAsset = PROSPER_TEST_MPEG4_ASSET;

// Whether THIS host can actually create a hardware H.264 session for the clip's own parameter
// sets (SPS/PPS copied from h264_aac_testpattern.mp4's avcC), observed independently of the
// backend. VTIsHardwareDecodeSupported is a capability query and is not enough: a hosted
// macos-26-arm64 runner answers yes for H.264 and then cannot open a hardware session.
bool hardware_h264_session_available() {
    static const uint8_t kSps[] = {0x67, 0x4d, 0x40, 0x0a, 0xec, 0xa1, 0x06, 0xd8, 0x08, 0x80, 0x00,
                                   0x00, 0x03, 0x00, 0x80, 0x00, 0x00, 0x1e, 0x07, 0x89, 0x12, 0xcb};
    static const uint8_t kPps[] = {0x68, 0xeb, 0xe3, 0xcb, 0x20};
    const uint8_t* sets[] = {kSps, kPps};
    const size_t sizes[] = {sizeof(kSps), sizeof(kPps)};
    CMFormatDescriptionRef format = nullptr;
    if (CMVideoFormatDescriptionCreateFromH264ParameterSets(nullptr, 2, sets, sizes, 4, &format) != noErr)
        return false;
    const void* keys[] = {kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder};
    const void* values[] = {kCFBooleanTrue};
    CFDictionaryRef spec = CFDictionaryCreate(nullptr, keys, values, 1, &kCFTypeDictionaryKeyCallBacks,
                                              &kCFTypeDictionaryValueCallBacks);
    VTDecompressionSessionRef session = nullptr;
    const OSStatus created = VTDecompressionSessionCreate(nullptr, format, spec, nullptr, nullptr, &session);
    bool hardware = false;
    if (created == noErr && session) {
        CFBooleanRef using_hardware = nullptr;
        if (VTSessionCopyProperty(session, kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
                                  nullptr, &using_hardware) == noErr && using_hardware) {
            hardware = CFBooleanGetValue(using_hardware);
            CFRelease(using_hardware);
        }
        VTDecompressionSessionInvalidate(session);
        CFRelease(session);
    }
    CFRelease(spec);
    CFRelease(format);
    std::fprintf(stderr, "[test] hardware H.264: capability query %s; a hardware session %s (status %d)\n",
                 VTIsHardwareDecodeSupported(kCMVideoCodecType_H264) ? "yes" : "no",
                 hardware ? "yes" : "no", static_cast<int>(created));
    return hardware;
}

// This process's open_memory temporary copies (the backend's own prefix, which carries our pid,
// so a concurrent run of this test in another process cannot change the count).
size_t temp_copies() {
    const std::filesystem::path prefix(videotoolbox_temp_prefix());
    const std::string stem = prefix.filename().string();
    size_t n = 0;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(prefix.parent_path(), ec))
        if (e.path().filename().string().rfind(stem, 0) == 0) ++n;
    return n;
}

// Poll `done` until it holds or `limit` passes; never an unbounded loop in a test.
template <typename Pred>
bool eventually(Pred done, std::chrono::milliseconds limit = std::chrono::milliseconds(10000)) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

uint64_t fnv(uint64_t h, const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

struct Drained {
    unsigned video = 0, audio = 0;
    uint64_t first_pts = UINT64_MAX, nv12_fnv = 0xcbf29ce484222325ull;
    bool monotonic = true, planes_ok = true, audio_ok = true, reached_eof = false;
};

// Pull everything the session produces, as the AvPlayer HLE does (peek, then take).
Drained drain(VideoBackend& vb, int id, const StreamInfo& stream) {
    Drained d;
    uint64_t previous = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (std::chrono::steady_clock::now() < deadline) {
        bool progressed = false;
        VideoFrame peeked{}, video{};
        if (vb.peek_video(id, peeked)) {
            const bool got = vb.next_video(id, video);
            d.planes_ok &= got && video.pts_us == peeked.pts_us;
        }
        if (video.y) {
            progressed = true;
            d.planes_ok &= video.uv && video.width == stream.width && video.height == stream.height &&
                           video.y_stride >= video.width && video.uv_stride >= video.width;
            if (d.planes_ok) {
                for (uint32_t r = 0; r < video.height; ++r)
                    d.nv12_fnv = fnv(d.nv12_fnv, video.y + size_t(r) * video.y_stride, video.width);
                for (uint32_t r = 0; r < (video.height + 1) / 2; ++r)
                    d.nv12_fnv = fnv(d.nv12_fnv, video.uv + size_t(r) * video.uv_stride, video.width);
            }
            d.monotonic &= d.video == 0 || video.pts_us > previous;
            if (d.video == 0) d.first_pts = video.pts_us;
            previous = video.pts_us;
            ++d.video;
        }
        AudioFrame audio{};
        if (vb.next_audio(id, audio)) {
            progressed = true;
            d.audio_ok &= audio.pcm && audio.channels == 2 && audio.sample_rate == 48000 && audio.samples > 0;
            ++d.audio;
        }
        if (!progressed) {
            if (vb.eof(id)) { d.reached_eof = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return d;
}

class VideoToolboxBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        set_backend(nullptr);
        ASSERT_TRUE(install_videotoolbox_backend());
        vb_ = backend();
        ASSERT_NE(vb_, nullptr);
        // The hardware gate, exercised in whichever direction this host allows. A host that can
        // open a hardware H.264 session (every Apple Silicon and recent Intel Mac) runs every case
        // through it. A host that cannot -- a virtualised CI runner -- must REFUSE without
        // the explicit override, which is asserted here, and only then runs the same checks in
        // software.
        hardware_ = hardware_h264_session_available();
        // This test binary owns PROSPER_AVP_ALLOW_SOFTWARE: every case starts with it unset and
        // TearDown unsets it again, so no case inherits another's software arm.
        unsetenv("PROSPER_AVP_ALLOW_SOFTWARE");
        if (!hardware_) {
            const int refused = vb_->open(kAsset);
            EXPECT_LT(refused, 0) << "without a hardware decoder, open must refuse unless software is "
                                     "explicitly allowed";
            if (refused >= 0) vb_->close(refused);
            setenv("PROSPER_AVP_ALLOW_SOFTWARE", "1", 1);
        }
    }
    void TearDown() override {
        unsetenv("PROSPER_AVP_ALLOW_SOFTWARE");
        uninstall_videotoolbox_backend();
    }
    VideoBackend* vb_ = nullptr;
    bool hardware_ = false;
};

TEST_F(VideoToolboxBackendTest, MissingSourceFailsInsteadOfFabricatingASession) {
    EXPECT_LT(vb_->open("/prosper-missing/no-such-video.mp4"), 0);
}

TEST_F(VideoToolboxBackendTest, DecodesEveryPictureBitExactWithAudio) {
    RecordProperty("hardware_h264_decoder", hardware_ ? "yes" : "no");
    const int id = vb_->open(kAsset);
    ASSERT_GE(id, 0) << (hardware_ ? "hardware" : "explicitly allowed software") << " decode";
    StreamInfo stream{};
    ASSERT_TRUE(vb_->info(id, stream));
    EXPECT_EQ(stream.width, 128u);
    EXPECT_EQ(stream.height, 96u);
    EXPECT_GT(stream.fps, 29.0f);
    EXPECT_LT(stream.fps, 31.0f);
    EXPECT_TRUE(stream.has_audio);
    EXPECT_EQ(stream.audio_channels, 2u);
    EXPECT_EQ(stream.audio_rate, 48000u);
    EXPECT_GE(stream.duration_us, 900000u);
    EXPECT_LE(stream.duration_us, 1100000u);

    // The OBSERVED decoder (the session's own report), not a capability query: hardware wherever
    // the host has an H.264 decoder, software only under the explicit override.
    EXPECT_EQ(static_cast<VideoToolboxBackend*>(vb_)->decoder_uses_hardware(id), hardware_ ? 1 : 0);
    const Drained d = drain(*vb_, id, stream);
    EXPECT_TRUE(d.reached_eof);
    EXPECT_EQ(d.video, kExpectedFrames) << "every picture is delivered exactly once";
    EXPECT_TRUE(d.planes_ok) << "NV12 planes at the stream size; peek matches next";
    EXPECT_TRUE(d.monotonic) << "video timestamps strictly increase";
    EXPECT_EQ(d.nv12_fnv, kExpectedNv12Fnv) << "NV12 pixels must match an independent bit-exact decode";
    EXPECT_GT(d.audio, 0u);
    EXPECT_TRUE(d.audio_ok) << "audio is interleaved 16-bit stereo at 48 kHz";
    vb_->close(id);
    EXPECT_FALSE(vb_->info(id, stream)) << "a closed session is gone";
}

TEST_F(VideoToolboxBackendTest, SeekAfterEndOfStreamRestartsAtTheTarget) {
    const int id = vb_->open(kAsset);
    ASSERT_GE(id, 0);
    StreamInfo stream{};
    ASSERT_TRUE(vb_->info(id, stream));
    ASSERT_TRUE(drain(*vb_, id, stream).reached_eof);

    ASSERT_TRUE(vb_->seek(id, 500000));
    EXPECT_FALSE(vb_->eof(id)) << "seek clears end of stream";
    const Drained s = drain(*vb_, id, stream);
    EXPECT_GE(s.first_pts, 499000u) << "the first picture after the seek is the one at 0.5 s";
    EXPECT_LT(s.first_pts, 540000u);
    EXPECT_EQ(s.video, kExpectedFrames / 2) << "the remaining 15 pictures play";
    EXPECT_TRUE(s.reached_eof);
    vb_->close(id);
}

TEST_F(VideoToolboxBackendTest, OpenMemoryDecodesTheSamePictures) {
    std::ifstream file(kAsset, std::ios::binary);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                     std::istreambuf_iterator<char>());
    ASSERT_FALSE(bytes.empty());
    const int id = vb_->open_memory("memory:h264_aac_testpattern.mp4", bytes.data(), bytes.size());
    ASSERT_GE(id, 0);
    StreamInfo stream{};
    ASSERT_TRUE(vb_->info(id, stream));
    const Drained d = drain(*vb_, id, stream);
    EXPECT_EQ(d.video, kExpectedFrames);
    EXPECT_EQ(d.nv12_fnv, kExpectedNv12Fnv);
    vb_->close(id);
}

TEST_F(VideoToolboxBackendTest, CropsToTheDisplayedRectangleBitExact) {
    const int id = vb_->open(kCropAsset);
    ASSERT_GE(id, 0);
    StreamInfo stream{};
    ASSERT_TRUE(vb_->info(id, stream));
    EXPECT_EQ(stream.width, 126u) << "the decoded picture is 128 wide; the displayed one is 126";
    EXPECT_EQ(stream.height, 94u);
    EXPECT_FALSE(stream.has_audio);
    const Drained d = drain(*vb_, id, stream);
    EXPECT_TRUE(d.reached_eof);
    EXPECT_EQ(d.video, 15u);
    EXPECT_TRUE(d.planes_ok);
    EXPECT_EQ(d.nv12_fnv, kCropNv12Fnv) << "cropped NV12 must match an independent bit-exact decode";
    vb_->close(id);
}

TEST_F(VideoToolboxBackendTest, ACodecWithoutAHardwareDecoderIsRefusedUnlessSoftwareIsAllowed) {
    unsetenv("PROSPER_AVP_ALLOW_SOFTWARE");
    const int refused = vb_->open(kSoftwareOnlyAsset);
    EXPECT_LT(refused, 0) << "hardware is required for the actual session; MPEG-4 Part 2 has none";
    if (refused >= 0) vb_->close(refused);

    setenv("PROSPER_AVP_ALLOW_SOFTWARE", "1", 1);
    const int id = vb_->open(kSoftwareOnlyAsset);
    ASSERT_GE(id, 0) << "the explicit override admits the software decoder";
    EXPECT_EQ(static_cast<VideoToolboxBackend*>(vb_)->decoder_uses_hardware(id), 0)
        << "and the backend reports what the session actually used";
    StreamInfo stream{};
    ASSERT_TRUE(vb_->info(id, stream));
    EXPECT_GT(drain(*vb_, id, stream).video, 0u);
    vb_->close(id);
}

// A 128x96 420v buffer whose displayed rectangle is NOT centred vertically: top-left (2, 4),
// 120 x 92 -- 4 rows cut from the top and none from the bottom, 2 columns cut from the left and
// 6 from the right. Expressed as a clean aperture the way containers carry it (ISO/IEC 14496-12
// 'clap', and CoreVideo's keys follow it): the aperture CENTRE relative to the picture centre,
// with y growing downward -- horizontal (2 + 60) - 64 = -2, vertical (4 + 46) - 48 = +2.
// CVImageBufferGetCleanRect reports that rectangle with a LOWER-LEFT origin (y = 0 here); a crop
// that took that y as a top row would start at row 0 and keep the 4 rows that are not displayed.
TEST(VideoToolboxCrop, AnAsymmetricVerticalCleanApertureIsCopiedFromTheRightRows) {
    constexpr int kW = 128, kH = 96;
    CVPixelBufferRef image = nullptr;
    ASSERT_EQ(CVPixelBufferCreate(nullptr, kW, kH, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                  nullptr, &image), kCVReturnSuccess);
    auto luma = [](int r, int c) { return static_cast<uint8_t>((r * 7 + c * 3) & 0xff); };
    auto chroma = [](int r, int c) { return static_cast<uint8_t>((r * 11 + c + 128) & 0xff); };
    ASSERT_EQ(CVPixelBufferLockBaseAddress(image, 0), kCVReturnSuccess);
    auto* y = static_cast<uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(image, 0));
    auto* uv = static_cast<uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(image, 1));
    const size_t y_pitch = CVPixelBufferGetBytesPerRowOfPlane(image, 0);
    const size_t uv_pitch = CVPixelBufferGetBytesPerRowOfPlane(image, 1);
    for (int r = 0; r < kH; ++r) for (int c = 0; c < kW; ++c) y[r * y_pitch + c] = luma(r, c);
    for (int r = 0; r < kH / 2; ++r) for (int c = 0; c < kW; ++c) uv[r * uv_pitch + c] = chroma(r, c);
    CVPixelBufferUnlockBaseAddress(image, 0);

    auto number = [](double v) {
        return CFNumberCreate(nullptr, kCFNumberDoubleType, &v);
    };
    const void* keys[] = {kCVImageBufferCleanApertureWidthKey, kCVImageBufferCleanApertureHeightKey,
                          kCVImageBufferCleanApertureHorizontalOffsetKey,
                          kCVImageBufferCleanApertureVerticalOffsetKey};
    CFNumberRef values[] = {number(120), number(92), number(-2), number(2)};
    CFDictionaryRef aperture = CFDictionaryCreate(nullptr, keys, reinterpret_cast<const void**>(values), 4,
                                                  &kCFTypeDictionaryKeyCallBacks,
                                                  &kCFTypeDictionaryValueCallBacks);
    CVBufferSetAttachment(image, kCVImageBufferCleanApertureKey, aperture, kCVAttachmentMode_ShouldPropagate);
    CFRelease(aperture);
    for (CFNumberRef v : values) CFRelease(v);

    uint32_t rect[4] = {};
    ASSERT_TRUE(videotoolbox_display_rect_for_test(image, rect));
    EXPECT_EQ(rect[0], 2u);
    EXPECT_EQ(rect[1], 4u) << "4 rows are cut from the TOP: the lower-left clean rect must be flipped";
    EXPECT_EQ(rect[2], 120u);
    EXPECT_EQ(rect[3], 92u);

    std::vector<uint8_t> nv12;
    uint32_t stride = 0;
    ASSERT_TRUE(videotoolbox_pack_for_test(image, rect, &nv12, &stride));
    ASSERT_EQ(stride, 120u);
    ASSERT_EQ(nv12.size(), size_t(120) * (92 + 46));
    bool luma_ok = true, chroma_ok = true;
    for (int r = 0; r < 92; ++r)
        for (int c = 0; c < 120; ++c) luma_ok &= nv12[r * stride + c] == luma(r + 4, c + 2);
    for (int r = 0; r < 46; ++r)
        for (int c = 0; c < 120; ++c) chroma_ok &= nv12[(92 + r) * stride + c] == chroma(r + 2, c + 2);
    EXPECT_TRUE(luma_ok) << "every displayed luma sample comes from (row + 4, column + 2)";
    EXPECT_TRUE(chroma_ok) << "every chroma pair comes from (row + 2, byte + 2)";
    CVPixelBufferRelease(image);
}

TEST_F(VideoToolboxBackendTest, MidStreamSeekDiscardsQueuedPicturesAndLandsOnTarget) {
    const int id = vb_->open(kAsset);
    ASSERT_GE(id, 0);
    StreamInfo stream{};
    ASSERT_TRUE(vb_->info(id, stream));
    // Let the worker fill its queue (it parks on the full queue), take a few, then seek back.
    ASSERT_TRUE(eventually([&] { return vb_->video_queue_depth(id) == 8; }))
        << "precondition: the worker parks on a full queue";
    int taken = 0;
    ASSERT_TRUE(eventually([&] {
        VideoFrame f{};
        if (vb_->next_video(id, f)) ++taken;
        return taken == 3;
    }));
    ASSERT_TRUE(vb_->seek(id, 200000));
    const Drained s = drain(*vb_, id, stream);
    EXPECT_GE(s.first_pts, 199000u) << "no picture queued before the seek may survive it";
    EXPECT_LT(s.first_pts, 240000u);
    EXPECT_EQ(s.video, 24u) << "pictures 6..29 play after a seek to 0.2 s";
    // Decoding restarts at the sync sample before the target and drops the pictures in between;
    // starting at the target itself would decode P/B pictures without their references.
    EXPECT_EQ(s.nv12_fnv, 0x754be0e6e17a86cdull) << "pictures 6..29, bit-exact against ffmpeg";
    EXPECT_TRUE(s.reached_eof);
    vb_->close(id);
}

TEST_F(VideoToolboxBackendTest, SeekBeyondTheStreamFailsInsteadOfEndingSilently) {
    const int id = vb_->open(kAsset);
    ASSERT_GE(id, 0);
    EXPECT_FALSE(vb_->seek(id, 10'000'000)) << "the guest must see a failed seek (#1949)";
    EXPECT_TRUE(vb_->seek(id, 0)) << "and a later valid seek still works";
    StreamInfo stream{};
    ASSERT_TRUE(vb_->info(id, stream));
    EXPECT_EQ(drain(*vb_, id, stream).video, kExpectedFrames);
    vb_->close(id);
}

TEST_F(VideoToolboxBackendTest, CloseWhileTheWorkerIsBlockedOnAFullQueueReturns) {
    const int id = vb_->open(kAsset);
    ASSERT_GE(id, 0);
    // Nobody pulls, so the queue fills and the worker parks on it.
    ASSERT_TRUE(eventually([&] { return vb_->video_queue_depth(id) == 8; }))
        << "precondition: the worker is parked on a full queue";
    const auto t0 = std::chrono::steady_clock::now();
    vb_->close(id);
    EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(2));
    StreamInfo stream{};
    EXPECT_FALSE(vb_->info(id, stream));
}

TEST_F(VideoToolboxBackendTest, OpenMemoryRefusesGarbageAndLeavesNoTemporaryFile) {
    const size_t before = temp_copies();
    const std::vector<uint8_t> garbage(4096, 0x5a);
    EXPECT_LT(vb_->open_memory("memory:garbage", garbage.data(), garbage.size()), 0);
    EXPECT_EQ(temp_copies(), before) << "a refused open must unlink its temporary copy";
    std::ifstream file(kAsset, std::ios::binary);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                     std::istreambuf_iterator<char>());
    const int id = vb_->open_memory("memory:clip", bytes.data(), bytes.size());
    ASSERT_GE(id, 0);
    EXPECT_EQ(temp_copies(), before + 1);
    vb_->close(id);
    EXPECT_EQ(temp_copies(), before) << "close unlinks the temporary copy";
}

// A minimal stand-in for another frontend's backend, to prove uninstall touches only its own.
class OtherBackend final : public VideoBackend {
public:
    int open(const std::string&) override { return -1; }
    bool info(int, StreamInfo&) override { return false; }
    bool next_video(int, VideoFrame&) override { return false; }
    bool next_audio(int, AudioFrame&) override { return false; }
    bool eof(int) override { return true; }
    void close(int) override {}
};

TEST(VideoToolboxBackendInstall, UninstallClearsOnlyItsOwnRegistration) {
    set_backend(nullptr);
    ASSERT_TRUE(install_videotoolbox_backend());
    EXPECT_NE(backend(), nullptr);
    uninstall_videotoolbox_backend();
    EXPECT_EQ(backend(), nullptr);

    OtherBackend other;
    set_backend(&other);
    uninstall_videotoolbox_backend();
    EXPECT_EQ(backend(), &other) << "uninstall must not clear a different backend's registration";
    set_backend(nullptr);
}

} // namespace
