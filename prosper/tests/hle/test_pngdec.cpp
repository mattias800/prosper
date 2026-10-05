// test_pngdec — libScePngDec must decode PNGs to caller buffers, not report success over
// untouched memory.
//
// All five exports were unregistered, so the dispatcher answered `0`. For Create that 0 is a
// null handle and for Decode/ParseHeader it is success over unwritten pixels and image
// structs — a title blits whatever its output buffer already held. Every arm drives the real
// NIDs: a hand-built 1x1 RGBA PNG (bytes below, zlib-compressed once with Python) decodes to
// exact pixels, and every refusal arm preserves sentinels.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper;

static uint64_t addr(const void* p) {
    return (uint64_t)(uintptr_t)p;
}
static uint64_t sx(uint32_t v) {
    return (uint64_t)(int64_t)(int32_t)v;
}

static constexpr uint64_t kAddr = 0x80690001ull;
static constexpr uint64_t kSize = 0x80690002ull;
static constexpr uint64_t kParam = 0x80690003ull;
static constexpr uint64_t kHandle = 0x80690004ull;
static constexpr uint64_t kWorkMemory = 0x80690005ull;
static constexpr uint64_t kData = 0x80690010ull;

namespace {

// 1x1 RGBA opaque red. IHDR 1x1 8-bit color-type 6, single stored-block IDAT, IEND.
const uint8_t kPng[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
    0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
    0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78,
    0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xf0, 0x1f, 0x00, 0x05, 0x00, 0x01, 0xff, 0x89, 0x99,
    0x3d, 0x1d, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

// CreateParam { u32 this_size @0; u32 attribute @4; u32 max_image_width @8 } — 0xC.
struct CreateParam {
    uint32_t this_size = 0xC;
    uint32_t attribute = 0;
    uint32_t max_image_width = 1024;
};
// ParseParam { const u8* png @0; u32 size @8; u32 reserved @12 } — 0x10.
struct ParseParam {
    uint64_t png_addr = 0;
    uint32_t png_size = 0;
    uint32_t reserved = 0;
};
// DecodeParam { const u8* png @0; u8* image @8; u32 png_size @16; u32 image_size @20;
// u16 pixel_format @24; u16 alpha_value @26; u32 pitch @28 } — 0x20.
struct DecodeParam {
    uint64_t png_addr = 0;
    uint64_t image_addr = 0;
    uint32_t png_size = 0;
    uint32_t image_size = 0;
    uint16_t pixel_format = 0;
    uint16_t alpha_value = 0;
    uint32_t pitch = 0;
};
// ImageInfo { u32 w @0; u32 h @4; u16 color_space @8; u16 bit_depth @10; u32 flags @12 }.
struct ImageInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    uint16_t color_space = 0;
    uint16_t bit_depth = 0;
    uint32_t flags = 0;
};

struct Api {
    HleFn create = nullptr, decode = nullptr, destroy = nullptr, parse = nullptr, query = nullptr;
};

Api registered() {
    register_builtin_hle();
    Api api;
    api.create = Hle::lookup("m0uW+8pFyaw");
    api.decode = Hle::lookup("WC216DD3El4");
    api.destroy = Hle::lookup("QbD+eENEwo8");
    api.parse = Hle::lookup("U6h4e5JRPaQ");
    api.query = Hle::lookup("-6srIGbLTIU");
    return api;
}

bool api_ok(const Api& api) {
    return api.create && api.decode && api.destroy && api.parse && api.query;
}

uint64_t make_handle(const Api& api, uint8_t* work, size_t work_size) {
    CreateParam param{};
    uint64_t handle = 0;
    if (api.create(addr(&param), addr(work), (uint64_t)work_size, addr(&handle), 0, 0) != 0) {
        return 0;
    }
    return handle;
}

}  // namespace

TEST(PngDec, NidsBound) {
    register_builtin_hle();
    EXPECT_NE(Hle::lookup(nid_hash("scePngDecCreate")), nullptr);
    EXPECT_NE(Hle::lookup(nid_hash("scePngDecDecode")), nullptr);
    EXPECT_NE(Hle::lookup(nid_hash("scePngDecDelete")), nullptr);
    EXPECT_NE(Hle::lookup(nid_hash("scePngDecParseHeader")), nullptr);
    EXPECT_NE(Hle::lookup(nid_hash("scePngDecQueryMemorySize")), nullptr);
}

TEST(PngDec, NidsResolveToKnownValues) {
    EXPECT_EQ(nid_hash("scePngDecCreate"), "m0uW+8pFyaw");
    EXPECT_EQ(nid_hash("scePngDecDecode"), "WC216DD3El4");
    EXPECT_EQ(nid_hash("scePngDecDelete"), "QbD+eENEwo8");
    EXPECT_EQ(nid_hash("scePngDecParseHeader"), "U6h4e5JRPaQ");
    EXPECT_EQ(nid_hash("scePngDecQueryMemorySize"), "-6srIGbLTIU");
    EXPECT_NE(nid_hash("scePngDecCreate"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

TEST(PngDec, QueryMemorySizeAndCreate) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    CreateParam param{};
    EXPECT_EQ(api.query(addr(&param), 0, 0, 0, 0, 0), 0x20u);
    EXPECT_EQ(api.query(0, 0, 0, 0, 0, 0), kParam) << "null param";
    CreateParam bad_attr = param;
    bad_attr.attribute = 7;
    EXPECT_EQ(api.query(addr(&bad_attr), 0, 0, 0, 0, 0), kParam) << "bad attribute";
    CreateParam bad_width = param;
    bad_width.max_image_width = 0;
    EXPECT_EQ(api.query(addr(&bad_width), 0, 0, 0, 0, 0), kSize) << "zero width";

    uint8_t work[0x20]{};
    uint64_t handle = 0;
    ASSERT_EQ(api.create(addr(&param), addr(work), sizeof work, addr(&handle), 0, 0), 0u);
    EXPECT_NE(handle, 0u) << "Create writes a handle, never the dispatcher 0";
    EXPECT_EQ(handle % 8u, 0u) << "handle is 8-aligned";
    EXPECT_EQ(api.create(addr(&param), addr(work), 0x10, addr(&handle), 0, 0), kWorkMemory)
        << "short work memory";
    EXPECT_EQ(api.create(addr(&param), 0, sizeof work, addr(&handle), 0, 0), kAddr)
        << "null work memory";
}

TEST(PngDec, DeleteInvalidatesExactlyOnce) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    uint8_t work[0x20]{};
    const uint64_t handle = make_handle(api, work, sizeof work);
    ASSERT_NE(handle, 0u);
    EXPECT_EQ(api.destroy(handle, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(api.destroy(handle, 0, 0, 0, 0, 0), kHandle) << "second delete fails";
    EXPECT_EQ(api.destroy(0xDEADu, 0, 0, 0, 0, 0), kHandle) << "forged handle fails";
    EXPECT_EQ(api.destroy(0, 0, 0, 0, 0, 0), kHandle) << "null handle fails";
}

TEST(PngDec, ParseHeaderReadsIhdr) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    ParseParam param{};
    param.png_addr = addr(kPng);
    param.png_size = sizeof kPng;
    ImageInfo info{};
    std::memset(&info, 0xAA, sizeof info);
    ASSERT_EQ(api.parse(addr(&param), addr(&info), 0, 0, 0, 0), 0u);
    EXPECT_EQ(info.width, 1u);
    EXPECT_EQ(info.height, 1u);
    EXPECT_EQ(info.color_space, 19u) << "6 = RGBA";
    EXPECT_EQ(info.bit_depth, 8u);
    EXPECT_EQ(api.parse(0, addr(&info), 0, 0, 0, 0), kParam) << "null param";
    EXPECT_EQ(api.parse(addr(&param), 0, 0, 0, 0, 0), kAddr) << "null info";
    ParseParam empty = param;
    empty.png_size = 0;
    EXPECT_EQ(api.parse(addr(&empty), addr(&info), 0, 0, 0, 0), kSize) << "empty input";
    const uint8_t garbage[16]{1, 2, 3};
    ParseParam bad{};
    bad.png_addr = addr(garbage);
    bad.png_size = sizeof garbage;
    std::memset(&info, 0xBB, sizeof info);
    EXPECT_EQ(api.parse(addr(&bad), addr(&info), 0, 0, 0, 0), kData) << "not a PNG";
    EXPECT_EQ(info.width, 0xBBBBBBBBu) << "failed parse writes no info";
}

TEST(PngDec, DecodeProducesPixels) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    uint8_t work[0x20]{};
    const uint64_t handle = make_handle(api, work, sizeof work);
    ASSERT_NE(handle, 0u);

    DecodeParam param{};
    param.png_addr = addr(kPng);
    param.png_size = sizeof kPng;
    std::vector<uint8_t> image(64, 0xAA);
    param.image_addr = addr(image.data());
    param.image_size = (uint32_t)image.size();
    param.pixel_format = 0;  // R8G8B8A8
    ImageInfo info{};
    const uint64_t r = api.decode(handle, addr(&param), addr(&info), 0, 0, 0);
    ASSERT_EQ(r, (1u << 16) | 1u) << "packed width<<16|height";
    EXPECT_EQ(image[0], 0xFF);
    EXPECT_EQ(image[1], 0x00);
    EXPECT_EQ(image[2], 0x00);
    EXPECT_EQ(image[3], 0xFF) << "opaque red pixel";
    EXPECT_EQ(info.width, 1u);
    EXPECT_EQ(info.color_space, 19u);

    // BGR order swaps red and blue.
    std::vector<uint8_t> image2(64, 0xAA);
    param.image_addr = addr(image2.data());
    param.pixel_format = 1;  // B8G8R8A8
    ASSERT_EQ(api.decode(handle, addr(&param), 0, 0, 0, 0), (1u << 16) | 1u);
    EXPECT_EQ(image2[0], 0x00);
    EXPECT_EQ(image2[2], 0xFF);

    // Refusals: bad handle, undersized output, bad pitch, garbage PNG.
    DecodeParam small = param;
    small.image_size = 3;
    EXPECT_EQ(api.decode(handle, addr(&small), 0, 0, 0, 0), kSize) << "3 bytes hold no pixel";
    DecodeParam bad_pitch = param;
    bad_pitch.pitch = 2;
    EXPECT_EQ(api.decode(handle, addr(&bad_pitch), 0, 0, 0, 0), kParam) << "pitch < row";
    const uint8_t garbage[16]{1, 2, 3};
    DecodeParam bad_png = param;
    bad_png.png_addr = addr(garbage);
    bad_png.png_size = sizeof garbage;
    std::vector<uint8_t> image3(64, 0xCC);
    bad_png.image_addr = addr(image3.data());
    bad_png.image_size = (uint32_t)image3.size();
    EXPECT_EQ(api.decode(handle, addr(&bad_png), 0, 0, 0, 0), kData);
    EXPECT_EQ(image3[0], 0xCC) << "failed decode writes no pixels";
    EXPECT_EQ(api.decode(0xDEADu, addr(&param), 0, 0, 0, 0), kHandle) << "forged handle";
    EXPECT_EQ(api.destroy(handle, 0, 0, 0, 0, 0), 0u);
}
