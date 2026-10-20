// The AU backend must report the pitch it actually copied: an odd NV12 width needs a complete
// final U/V pair. A width-sized UV stride makes every row after the first read the wrong bytes.
#include "vaapi_backend.hpp"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace prosper::video {
namespace {

uint8_t luma(int row, int col) {
    return static_cast<uint8_t>(1 + row * 7 + col);
}
uint8_t chroma_u(int row, int col) {
    return static_cast<uint8_t>(31 + row * 17 + col);
}
uint8_t chroma_v(int row, int col) {
    return static_cast<uint8_t>(101 + row * 19 + col);
}

struct SourceFrame {
    AVFrame frame{};
    std::vector<uint8_t> y, uv, u, v;

    SourceFrame(AVPixelFormat format, int width, int height) {
        frame.format = format;
        frame.width = width;
        frame.height = height;
        // Source padding differs from the packed destination, so copying source stride as payload
        // cannot accidentally satisfy the pixel assertions.
        frame.linesize[0] = width + 7;
        y.assign(frame.linesize[0] * height, 0xEE);
        frame.data[0] = y.data();
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col) y[row * frame.linesize[0] + col] = luma(row, col);

        const int cw = (width + 1) / 2, ch = (height + 1) / 2;
        if (format == AV_PIX_FMT_NV12) {
            frame.linesize[1] = 2 * cw + 9;
            uv.assign(frame.linesize[1] * ch, 0xEE);
            frame.data[1] = uv.data();
            for (int row = 0; row < ch; ++row) {
                for (int col = 0; col < cw; ++col) {
                    uv[row * frame.linesize[1] + 2 * col] = chroma_u(row, col);
                    uv[row * frame.linesize[1] + 2 * col + 1] = chroma_v(row, col);
                }
            }
        } else {
            frame.linesize[1] = cw + 5;
            frame.linesize[2] = cw + 11;
            u.assign(frame.linesize[1] * ch, 0xEE);
            v.assign(frame.linesize[2] * ch, 0xEE);
            frame.data[1] = u.data();
            frame.data[2] = v.data();
            for (int row = 0; row < ch; ++row) {
                for (int col = 0; col < cw; ++col) {
                    u[row * frame.linesize[1] + col] = chroma_u(row, col);
                    v[row * frame.linesize[2] + col] = chroma_v(row, col);
                }
            }
        }
    }
};

class VaapiAuPictureCopy : public testing::TestWithParam<AVPixelFormat> {
protected:
    void expect_picture(int width, int height, uint32_t expected_uv_stride,
                        uint64_t expected_bytes) {
        SourceFrame source(GetParam(), width, height);
        constexpr size_t guard = 16;
        std::vector<uint8_t> storage(expected_bytes + 2 * guard, 0xA5);
        uint8_t* dst = storage.data() + guard;
        VideoBackend::AuPicture out{};
        ASSERT_EQ(detail::copy_au_picture(&source.frame, false, dst, expected_bytes, out),
                  VideoBackend::AuResult::Decoded);
        ASSERT_EQ(out.width, width);
        ASSERT_EQ(out.height, height);
        ASSERT_EQ(out.y_stride, width);
        ASSERT_EQ(out.uv_stride, expected_uv_stride);
        ASSERT_EQ(out.nv12_bytes, expected_bytes);
        EXPECT_FALSE(out.hardware);
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
                EXPECT_EQ(dst[row * out.y_stride + col], luma(row, col));
        const uint8_t* uv = dst + out.y_stride * out.height;
        for (int row = 0; row < (height + 1) / 2; ++row) {
            for (int col = 0; col < (width + 1) / 2; ++col) {
                EXPECT_EQ(uv[row * out.uv_stride + 2 * col], chroma_u(row, col));
                EXPECT_EQ(uv[row * out.uv_stride + 2 * col + 1], chroma_v(row, col));
            }
        }
        EXPECT_TRUE(std::all_of(storage.begin(), storage.begin() + guard,
                                [](uint8_t byte) { return byte == 0xA5; }));
        EXPECT_TRUE(std::all_of(storage.end() - guard, storage.end(),
                                [](uint8_t byte) { return byte == 0xA5; }));
    }
};

TEST_P(VaapiAuPictureCopy, OddWidthReportsActualChromaPitch) {
    // 65*5 luma bytes plus three rows of 33 U/V pairs: 325 + 198 = 523.
    expect_picture(65, 5, 66, 523);
}

TEST_P(VaapiAuPictureCopy, EvenWidthKeepsPackedPitch) {
    expect_picture(64, 4, 64, 384);
}

TEST_P(VaapiAuPictureCopy, SinglePixelRetainsCompleteChromaPair) {
    expect_picture(1, 1, 2, 3);
}

TEST_P(VaapiAuPictureCopy, TooSmallReportsGeometryWithoutWriting) {
    SourceFrame source(GetParam(), 65, 5);
    std::vector<uint8_t> dst(522, 0xA5);
    VideoBackend::AuPicture out{};
    ASSERT_EQ(detail::copy_au_picture(&source.frame, true, dst.data(), dst.size(), out),
              VideoBackend::AuResult::FrameTooSmall);
    EXPECT_EQ(out.width, 65u);
    EXPECT_EQ(out.height, 5u);
    EXPECT_EQ(out.y_stride, 65u);
    EXPECT_EQ(out.uv_stride, 66u);
    EXPECT_EQ(out.nv12_bytes, 523u);
    EXPECT_TRUE(out.hardware);
    EXPECT_TRUE(std::all_of(dst.begin(), dst.end(), [](uint8_t byte) { return byte == 0xA5; }));
}

INSTANTIATE_TEST_SUITE_P(DecodedFormats, VaapiAuPictureCopy,
                         testing::Values(AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P));

}   // namespace
}   // namespace prosper::video
