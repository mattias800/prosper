// test_pngdec — libScePngDec must decode PNGs to caller buffers the way the shipped module does,
// not report success over untouched memory.
//
// All five exports were unregistered, so the dispatcher answered `0`. For Create that 0 is a
// null handle and for Decode/ParseHeader it is success over unwritten pixels and image
// structs — a title blits whatever its output buffer already held. Every arm drives the real
// NIDs. The PNGs are built in-test by a small encoder (stored-deflate zlib, filter 0, optional
// Adam7), so each colour type / bit depth / tRNS / interlace path has an input whose decoded
// pixels are known exactly. Expected error codes and validation order come from the shipped
// module's disassembly; the offsets are cited in src/hle/util/hle_pngdec.cpp.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "host/memory/guest_memory_copy.hpp"
#include "host/memory/guest_write_watch.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace prosper;

static uint64_t addr(const void* p) {
    return (uint64_t)(uintptr_t)p;
}

static constexpr uint64_t kAddr = 0x80690001ull;
static constexpr uint64_t kSize = 0x80690002ull;
static constexpr uint64_t kParam = 0x80690003ull;
static constexpr uint64_t kHandle = 0x80690004ull;
static constexpr uint64_t kWorkMemory = 0x80690005ull;
static constexpr uint64_t kData = 0x80690010ull;
static constexpr uint64_t kUnsupport = 0x80690011ull;
static constexpr uint64_t kDecode = 0x80690012ull;

namespace {

// ---- a minimal PNG encoder ------------------------------------------------------------------

uint32_t crc32(const uint8_t* p, size_t n, uint32_t crc = 0xFFFFFFFFu) {
    for (size_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return crc;
}
void put_be32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 24));
    v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8));
    v.push_back((uint8_t)x);
}
std::vector<uint8_t> chunk(const char* type, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> c;
    put_be32(c, (uint32_t)data.size());
    c.insert(c.end(), type, type + 4);
    c.insert(c.end(), data.begin(), data.end());
    const uint32_t crc = ~crc32(c.data() + 4, c.size() - 4);
    put_be32(c, crc);
    return c;
}
std::vector<uint8_t> zlib_stored(const std::vector<uint8_t>& raw) {
    std::vector<uint8_t> z = {0x78, 0x01};
    size_t off = 0;
    do {
        const size_t n = std::min<size_t>(raw.size() - off, 65535);
        const bool last = off + n == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back((uint8_t)n);
        z.push_back((uint8_t)(n >> 8));
        z.push_back((uint8_t)~n);
        z.push_back((uint8_t)(~n >> 8));
        z.insert(z.end(), raw.begin() + (long)off, raw.begin() + (long)(off + n));
        off += n;
    } while (off < raw.size());
    uint32_t a = 1, b = 0;
    for (uint8_t x : raw) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    put_be32(z, (b << 16) | a);
    return z;
}
std::vector<uint8_t> ihdr(uint32_t w, uint32_t h, uint8_t depth, uint8_t ctype, uint8_t interlace) {
    std::vector<uint8_t> d;
    put_be32(d, w);
    put_be32(d, h);
    d.push_back(depth);
    d.push_back(ctype);
    d.push_back(0);
    d.push_back(0);
    d.push_back(interlace);
    return d;
}
const uint8_t kSig[8] = {137, 80, 78, 71, 13, 10, 26, 10};

struct Spec {
    uint32_t w = 1, h = 1;
    uint8_t depth = 8, ctype = 6, interlace = 0;
    std::vector<uint8_t> plte, trns;
    // Channel samples of pixel (x, y): 1 (gray/index), 2 (gray+alpha), 3 (RGB) or 4 (RGBA).
    std::function<std::vector<uint16_t>(uint32_t, uint32_t)> px;
};
int channels(uint8_t ctype) {
    switch (ctype) {
        case 0: case 3: return 1;
        case 4: return 2;
        case 2: return 3;
        default: return 4;
    }
}
// One filter-0 scanline of the pixels at x0, x0+dx, ... on row y.
void append_scanline(std::vector<uint8_t>& raw, const Spec& s, uint32_t y, uint32_t x0, uint32_t dx) {
    raw.push_back(0);
    const int ch = channels(s.ctype);
    uint32_t acc = 0;
    int bits = 0;
    for (uint32_t x = x0; x < s.w; x += dx) {
        const std::vector<uint16_t> v = s.px(x, y);
        for (int c = 0; c < ch; c++) {
            if (s.depth == 16) {
                raw.push_back((uint8_t)(v[c] >> 8));
                raw.push_back((uint8_t)v[c]);
            } else if (s.depth == 8) {
                raw.push_back((uint8_t)v[c]);
            } else {
                acc = (acc << s.depth) | (v[c] & ((1u << s.depth) - 1));
                bits += s.depth;
                if (bits == 8) {
                    raw.push_back((uint8_t)acc);
                    acc = 0;
                    bits = 0;
                }
            }
        }
    }
    if (bits) raw.push_back((uint8_t)(acc << (8 - bits)));
}
std::vector<uint8_t> encode(const Spec& s) {
    std::vector<uint8_t> raw;
    if (!s.interlace) {
        for (uint32_t y = 0; y < s.h; y++) append_scanline(raw, s, y, 0, 1);
    } else {
        static const uint32_t x0[7] = {0, 4, 0, 2, 0, 1, 0}, y0[7] = {0, 0, 4, 0, 2, 0, 1};
        static const uint32_t dx[7] = {8, 8, 4, 4, 2, 2, 1}, dy[7] = {8, 8, 8, 4, 4, 2, 2};
        for (int p = 0; p < 7; p++) {
            if (s.w <= x0[p] || s.h <= y0[p]) continue;  // an empty pass has no scanlines
            for (uint32_t y = y0[p]; y < s.h; y += dy[p]) append_scanline(raw, s, y, x0[p], dx[p]);
        }
    }
    std::vector<uint8_t> png(kSig, kSig + 8);
    auto add = [&](const std::vector<uint8_t>& c) { png.insert(png.end(), c.begin(), c.end()); };
    add(chunk("IHDR", ihdr(s.w, s.h, s.depth, s.ctype, s.interlace)));
    if (!s.plte.empty()) add(chunk("PLTE", s.plte));
    if (!s.trns.empty()) add(chunk("tRNS", s.trns));
    add(chunk("IDAT", zlib_stored(raw)));
    add(chunk("IEND", {}));
    return png;
}
// A PNG made of the signature and exactly these chunks.
std::vector<uint8_t> assemble(const std::vector<std::vector<uint8_t>>& chunks) {
    std::vector<uint8_t> png(kSig, kSig + 8);
    for (const auto& c : chunks) png.insert(png.end(), c.begin(), c.end());
    return png;
}

// RGBA test pattern: distinct value in every channel of every pixel.
std::vector<uint16_t> rgba_pattern(uint32_t x, uint32_t y) {
    return {(uint16_t)(0x10 + x), (uint16_t)(0x40 + y), (uint16_t)(0x80 + x + y), (uint16_t)(0xF0 - x)};
}
Spec rgba_spec(uint32_t w, uint32_t h) {
    Spec s;
    s.w = w;
    s.h = h;
    s.px = rgba_pattern;
    return s;
}

// ---- the API ---------------------------------------------------------------------------------

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
// What Create writes at the 16-aligned start of the work memory.
struct Context {
    uint64_t self;
    uint32_t this_size, attribute, max_width_round8, work_size;
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

uint32_t module_memory_size(uint32_t max_w, uint32_t attribute) {
    return ((max_w + 0x47u) & ~7u) * (8u << attribute) + 0xD090u;
}

// A decoder instance over its own work memory.
struct Decoder {
    std::vector<uint8_t> work;
    uint64_t handle = 0;
    Decoder(const Api& api, uint32_t max_w = 1024, uint32_t attribute = 0) {
        CreateParam param{};
        param.attribute = attribute;
        param.max_image_width = max_w;
        work.assign(module_memory_size(max_w, attribute) + 16, 0);
        const uint64_t r = api.create(addr(&param), addr(work.data()), work.size(), addr(&handle), 0, 0);
        if (r != 0) handle = 0;
    }
};

DecodeParam decode_param(const std::vector<uint8_t>& png, std::vector<uint8_t>& image) {
    DecodeParam p{};
    p.png_addr = addr(png.data());
    p.png_size = (uint32_t)png.size();
    p.image_addr = addr(image.data());
    p.image_size = (uint32_t)image.size();
    return p;
}

std::vector<uint8_t> pixel(const std::vector<uint8_t>& img, size_t stride, uint32_t x, uint32_t y) {
    const uint8_t* p = img.data() + y * stride + x * 4;
    return {p[0], p[1], p[2], p[3]};
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

TEST(PngDec, QueryMemorySizeIsTheModuleFormula) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    CreateParam p{};
    EXPECT_EQ(api.query(addr(&p), 0, 0, 0, 0, 0), 0xF290u) << "1024 wide, attribute 0";
    p.attribute = 1;
    EXPECT_EQ(api.query(addr(&p), 0, 0, 0, 0, 0), 0x11490u) << "attribute 1 doubles the row term";
    p.attribute = 0;
    p.max_image_width = 1;
    EXPECT_EQ(api.query(addr(&p), 0, 0, 0, 0, 0), 0xD2D0u);
    p.max_image_width = 1000000;
    EXPECT_EQ(api.query(addr(&p), 0, 0, 0, 0, 0), module_memory_size(1000000, 0)) << "upper bound";

    EXPECT_EQ(api.query(0, 0, 0, 0, 0, 0), kAddr) << "null param";
    CreateParam bad = CreateParam{};
    bad.this_size = 0x10;
    EXPECT_EQ(api.query(addr(&bad), 0, 0, 0, 0, 0), kSize) << "this_size != 0xC";
    bad = CreateParam{};
    bad.attribute = 2;
    EXPECT_EQ(api.query(addr(&bad), 0, 0, 0, 0, 0), kParam) << "attribute > 1";
    bad = CreateParam{};
    bad.max_image_width = 0;
    EXPECT_EQ(api.query(addr(&bad), 0, 0, 0, 0, 0), kParam) << "zero width";
    bad.max_image_width = 1000001;
    EXPECT_EQ(api.query(addr(&bad), 0, 0, 0, 0, 0), kParam) << "max_w - 1 > 999999";
}

TEST(PngDec, CreatePlacesTheContextLikeTheModule) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    CreateParam param{};
    param.attribute = 1;
    param.max_image_width = 1001;
    const uint32_t need = module_memory_size(1001, 1);
    std::vector<uint8_t> work(need + 64, 0xCD);
    // Deliberately misaligned memory: the context goes to the next 16-byte boundary.
    uint8_t* mem = work.data() + ((16 - (addr(work.data()) & 15)) & 15) + 3;
    const uint64_t expect_handle = (addr(mem) + 15) & ~15ull;
    uint64_t handle = 0;
    EXPECT_EQ(api.create(addr(&param), addr(mem), need - 1, addr(&handle), 0, 0), kSize)
        << "one byte short of QueryMemorySize";
    EXPECT_EQ(handle, 0u);
    ASSERT_EQ(api.create(addr(&param), addr(mem), need, addr(&handle), 0, 0), 0u);
    EXPECT_EQ(handle, expect_handle);
    Context ctx{};
    std::memcpy(&ctx, (const void*)(uintptr_t)handle, sizeof ctx);
    EXPECT_EQ(ctx.self, handle);
    EXPECT_EQ(ctx.this_size, 0xCu);
    EXPECT_EQ(ctx.attribute, 1u);
    EXPECT_EQ(ctx.max_width_round8, 1008u);
    EXPECT_EQ(ctx.work_size, need - 0x90u);
    EXPECT_EQ(api.destroy(handle, 0, 0, 0, 0, 0), 0u);

    // memory_size is a u32: the upper half of the register is not part of it.
    EXPECT_EQ(api.create(addr(&param), addr(mem), (1ull << 32) | need, addr(&handle), 0, 0), 0u);
    EXPECT_EQ(api.create(addr(&param), addr(mem), 1ull << 32, addr(&handle), 0, 0), kSize);

    EXPECT_EQ(api.create(0, addr(mem), need, addr(&handle), 0, 0), kAddr) << "null param";
    EXPECT_EQ(api.create(addr(&param), 0, need, addr(&handle), 0, 0), kAddr) << "null memory";
    EXPECT_EQ(api.create(addr(&param), addr(mem), need, 0, 0, 0), kAddr) << "null handle out";
    CreateParam bad = param;
    bad.this_size = 0xB;
    EXPECT_EQ(api.create(addr(&bad), addr(mem), need, addr(&handle), 0, 0), kSize);
    bad = param;
    bad.max_image_width = 1000001;
    EXPECT_EQ(api.create(addr(&bad), addr(mem), need, addr(&handle), 0, 0), kParam);
}

TEST(PngDec, DeleteRequiresA16ByteAlignedLiveHandle) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    Decoder d(api);
    ASSERT_NE(d.handle, 0u);
    EXPECT_EQ(d.handle % 16u, 0u);
    EXPECT_EQ(api.destroy(d.handle, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(api.destroy(d.handle, 0, 0, 0, 0, 0), kHandle) << "second delete fails";
    EXPECT_EQ(api.destroy(0, 0, 0, 0, 0, 0), kHandle) << "null handle fails";

    // A self-tagged block passes only at a 16-byte boundary: 8 alignment is not enough.
    alignas(16) uint64_t block[4] = {};
    block[1] = addr(&block[1]);
    EXPECT_EQ(api.destroy(addr(&block[1]), 0, 0, 0, 0, 0), kHandle) << "8- but not 16-aligned";
    EXPECT_EQ(block[1], addr(&block[1])) << "a refused delete does not clear the tag";
    block[0] = addr(&block[0]);
    EXPECT_EQ(api.destroy(addr(&block[0]), 0, 0, 0, 0, 0), 0u) << "16-aligned self tag";
    EXPECT_EQ(block[0], 0u);
}

TEST(PngDec, ParseHeaderReportsTheHeader) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    auto parse = [&](const std::vector<uint8_t>& png, ImageInfo* info) {
        ParseParam p{};
        p.png_addr = addr(png.data());
        p.png_size = (uint32_t)png.size();
        return api.parse(addr(&p), addr(info), 0, 0, 0, 0);
    };
    ImageInfo info{};
    ASSERT_EQ(parse(encode(rgba_spec(3, 2)), &info), 0u);
    EXPECT_EQ(info.width, 3u);
    EXPECT_EQ(info.height, 2u);
    EXPECT_EQ(info.color_space, 19u) << "colour type 6";
    EXPECT_EQ(info.bit_depth, 8u);
    EXPECT_EQ(info.flags, 0u);

    Spec gray;
    gray.ctype = 0;
    gray.depth = 2;
    gray.interlace = 1;
    gray.w = 5;
    gray.h = 3;
    gray.trns = {0, 1};
    gray.px = [](uint32_t x, uint32_t) { return std::vector<uint16_t>{(uint16_t)(x & 3)}; };
    ASSERT_EQ(parse(encode(gray), &info), 0u);
    EXPECT_EQ(info.color_space, 2u);
    EXPECT_EQ(info.bit_depth, 2u);
    EXPECT_EQ(info.flags, 3u) << "interlaced (1) | tRNS (2)";

    Spec pal;
    pal.ctype = 3;
    pal.depth = 4;
    pal.plte = {1, 2, 3};
    pal.px = [](uint32_t, uint32_t) { return std::vector<uint16_t>{0}; };
    ASSERT_EQ(parse(encode(pal), &info), 0u);
    EXPECT_EQ(info.color_space, 4u);
    EXPECT_EQ(info.flags, 0u);
}

TEST(PngDec, ParseHeaderValidatesInTheModulesOrder) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    const std::vector<uint8_t> png = encode(rgba_spec(2, 2));
    ParseParam p{};
    p.png_addr = addr(png.data());
    p.png_size = (uint32_t)png.size();
    ImageInfo info{};
    EXPECT_EQ(api.parse(0, addr(&info), 0, 0, 0, 0), kAddr) << "null param";
    ParseParam q = p;
    q.png_addr = 0;
    EXPECT_EQ(api.parse(addr(&q), addr(&info), 0, 0, 0, 0), kAddr) << "null png";
    q = p;
    q.png_size = 0;
    EXPECT_EQ(api.parse(addr(&q), addr(&info), 0, 0, 0, 0), kSize) << "empty png";
    q = p;
    q.reserved = 1;
    EXPECT_EQ(api.parse(addr(&q), addr(&info), 0, 0, 0, 0), kParam) << "reserved != 0";
    EXPECT_EQ(api.parse(addr(&p), 0, 0, 0, 0, 0), kAddr) << "null info";
    q = p;
    q.png_size = 7;
    EXPECT_EQ(api.parse(addr(&q), addr(&info), 0, 0, 0, 0), kData) << "shorter than a signature";

    auto parse = [&](const std::vector<uint8_t>& bytes) {
        ParseParam r{};
        r.png_addr = addr(bytes.data());
        r.png_size = (uint32_t)bytes.size();
        std::memset(&info, 0xBB, sizeof info);
        return api.parse(addr(&r), addr(&info), 0, 0, 0, 0);
    };
    std::vector<uint8_t> garbage(32, 0x11);
    EXPECT_EQ(parse(garbage), kData) << "signature mismatch";
    EXPECT_EQ(info.width, 0xBBBBBBBBu) << "failed parse writes no info";
    EXPECT_EQ(parse(assemble({chunk("IHDR", ihdr(4, 4, 8, 6, 0))})), kDecode)
        << "stream ends before IDAT";
    EXPECT_EQ(parse(assemble({chunk("IHDR", ihdr(1000001, 4, 8, 6, 0)), chunk("IDAT", {})})),
              kUnsupport)
        << "wider than libpng's 1,000,000 limit";
    EXPECT_EQ(parse(assemble({chunk("IHDR", ihdr(4, 1000001, 8, 6, 0)), chunk("IDAT", {})})),
              kUnsupport)
        << "taller than libpng's 1,000,000 limit";
    EXPECT_EQ(parse(assemble({chunk("IHDR", ihdr(4, 4, 4, 2, 0)), chunk("IDAT", {})})), kData)
        << "4-bit RGB is not a PNG";
    EXPECT_EQ(parse(assemble({chunk("IHDR", ihdr(0, 4, 8, 6, 0)), chunk("IDAT", {})})), kData)
        << "zero width";
    EXPECT_EQ(parse(assemble({chunk("IHDR", ihdr(4, 4, 8, 6, 2)), chunk("IDAT", {})})), kData)
        << "interlace method 2";
    EXPECT_EQ(parse(assemble({chunk("IDAT", {}), chunk("IHDR", ihdr(4, 4, 8, 6, 0))})), kDecode)
        << "IHDR is not first";
    EXPECT_EQ(parse(assemble({chunk("IHDR", ihdr(4, 4, 8, 3, 0)), chunk("IDAT", {})})), kDecode)
        << "palette image without PLTE";
    EXPECT_EQ(info.width, 0xBBBBBBBBu);
}

// P6: the header walk stops at the first IDAT, as png_read_info does. Nothing after it is read
// and nothing is allocated from the guest's size — so a size far larger than the mapped buffer
// still parses (the old copy-everything path read 4 GiB and failed with ADDR).
TEST(PngDec, ParseHeaderReadsOnlyUpToTheFirstIdat) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    const std::vector<uint8_t> png = encode(rgba_spec(7, 5));
    ParseParam p{};
    p.png_addr = addr(png.data());
    p.png_size = 0xFFFFFFF0u;
    ImageInfo info{};
    ASSERT_EQ(api.parse(addr(&p), addr(&info), 0, 0, 0, 0), 0u);
    EXPECT_EQ(info.width, 7u);
    EXPECT_EQ(info.height, 5u);
}

// P1: rows land at image + y*pitch, row_size bytes each; the caller's padding is never touched.
TEST(PngDec, DecodeWritesRowsAtPitchAndLeavesPaddingAlone) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    Decoder d(api);
    ASSERT_NE(d.handle, 0u);
    const uint32_t w = 3, h = 4, row = w * 4, pitch = row + 8;
    const std::vector<uint8_t> png = encode(rgba_spec(w, h));
    std::vector<uint8_t> image(pitch * h + 16, 0xEE);
    DecodeParam p = decode_param(png, image);
    p.pitch = pitch;
    ImageInfo info{};
    ASSERT_EQ(api.decode(d.handle, addr(&p), addr(&info), 0, 0, 0), (uint64_t)((w << 16) | h));
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            const std::vector<uint16_t> e = rgba_pattern(x, y);
            EXPECT_EQ(pixel(image, pitch, x, y),
                      (std::vector<uint8_t>{(uint8_t)e[0], (uint8_t)e[1], (uint8_t)e[2], (uint8_t)e[3]}))
                << "pixel " << x << "," << y;
        }
        for (uint32_t b = row; b < pitch; b++) {
            EXPECT_EQ(image[y * pitch + b], 0xEE) << "padding byte " << b << " of row " << y;
        }
    }
    for (size_t b = pitch * h; b < image.size(); b++) EXPECT_EQ(image[b], 0xEE) << "tail " << b;

    // pitch 0 means tightly packed rows.
    std::vector<uint8_t> tight(row * h, 0xEE);
    DecodeParam t = decode_param(png, tight);
    ASSERT_EQ(api.decode(d.handle, addr(&t), 0, 0, 0, 0), (uint64_t)((w << 16) | h));
    EXPECT_EQ(pixel(tight, row, 2, 3), pixel(image, pitch, 2, 3));
}

TEST(PngDec, DecodeFillsAlphaAndSwapsForBgra) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    Decoder d(api);
    Spec rgb;
    rgb.ctype = 2;
    rgb.w = 2;
    rgb.h = 2;
    rgb.px = [](uint32_t x, uint32_t y) {
        return std::vector<uint16_t>{(uint16_t)(0x11 * (x + 1)), (uint16_t)(0x22 + y), 0x33};
    };
    const std::vector<uint8_t> png = encode(rgb);
    std::vector<uint8_t> image(16, 0);
    DecodeParam p = decode_param(png, image);
    p.alpha_value = 0x40;
    ASSERT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), (2u << 16) | 2u);
    EXPECT_EQ(pixel(image, 8, 1, 1), (std::vector<uint8_t>{0x22, 0x23, 0x33, 0x40}))
        << "no alpha channel and no tRNS: alpha_value fills";
    p.pixel_format = 1;
    ASSERT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), (2u << 16) | 2u);
    EXPECT_EQ(pixel(image, 8, 1, 1), (std::vector<uint8_t>{0x33, 0x23, 0x22, 0x40})) << "BGRA";

    // An image that carries alpha keeps it; alpha_value is not applied.
    const std::vector<uint8_t> rgba = encode(rgba_spec(2, 2));
    DecodeParam q = decode_param(rgba, image);
    q.alpha_value = 0x40;
    ASSERT_EQ(api.decode(d.handle, addr(&q), 0, 0, 0, 0), (2u << 16) | 2u);
    EXPECT_EQ(image[4 * 3 + 3], (uint8_t)rgba_pattern(1, 1)[3]);
}

TEST(PngDec, DecodesPaletteGrayLowBitDepthAndTrns) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    Decoder d(api);
    std::vector<uint8_t> image(256, 0);

    // 4-bit palette with a partial tRNS: entries without a tRNS value are opaque, and alpha_value
    // is NOT applied because the image has tRNS.
    Spec pal;
    pal.ctype = 3;
    pal.depth = 4;
    pal.w = 3;
    pal.h = 1;
    pal.plte = {0xFF, 0, 0, 0, 0xFF, 0, 0, 0, 0xFF};
    pal.trns = {0x00, 0x80};
    pal.px = [](uint32_t x, uint32_t) { return std::vector<uint16_t>{(uint16_t)x}; };
    std::vector<uint8_t> png = encode(pal);
    DecodeParam p = decode_param(png, image);
    p.alpha_value = 0x11;
    ImageInfo info{};
    ASSERT_EQ(api.decode(d.handle, addr(&p), addr(&info), 0, 0, 0), (3u << 16) | 1u);
    EXPECT_EQ(pixel(image, 12, 0, 0), (std::vector<uint8_t>{0xFF, 0, 0, 0x00}));
    EXPECT_EQ(pixel(image, 12, 1, 0), (std::vector<uint8_t>{0, 0xFF, 0, 0x80}));
    EXPECT_EQ(pixel(image, 12, 2, 0), (std::vector<uint8_t>{0, 0, 0xFF, 0xFF}));
    EXPECT_EQ(info.flags, 2u);
    EXPECT_EQ(info.color_space, 4u);

    // The same palette without tRNS: alpha_value fills.
    pal.trns.clear();
    png = encode(pal);
    p = decode_param(png, image);
    p.alpha_value = 0x11;
    ASSERT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), (3u << 16) | 1u);
    EXPECT_EQ(pixel(image, 12, 1, 0), (std::vector<uint8_t>{0, 0xFF, 0, 0x11}));

    // 1-bit gray, 9 wide (not a whole byte per row): 0 -> black, 1 -> white.
    Spec g1;
    g1.ctype = 0;
    g1.depth = 1;
    g1.w = 9;
    g1.h = 2;
    g1.px = [](uint32_t x, uint32_t y) { return std::vector<uint16_t>{(uint16_t)((x + y) & 1)}; };
    png = encode(g1);
    p = decode_param(png, image);
    p.alpha_value = 0x7F;
    ASSERT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), (9u << 16) | 2u);
    EXPECT_EQ(pixel(image, 36, 8, 0), (std::vector<uint8_t>{0, 0, 0, 0x7F}));
    EXPECT_EQ(pixel(image, 36, 8, 1), (std::vector<uint8_t>{0xFF, 0xFF, 0xFF, 0x7F}));

    // 2-bit gray scales 0..3 to 0, 85, 170, 255.
    Spec g2;
    g2.ctype = 0;
    g2.depth = 2;
    g2.w = 4;
    g2.h = 1;
    g2.px = [](uint32_t x, uint32_t) { return std::vector<uint16_t>{(uint16_t)x}; };
    png = encode(g2);
    p = decode_param(png, image);
    ASSERT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), (4u << 16) | 1u);
    EXPECT_EQ(image[4 * 1], 85);
    EXPECT_EQ(image[4 * 2 + 1], 170);
    EXPECT_EQ(image[4 * 3 + 2], 255);

    // 8-bit gray with a tRNS key: the keyed value becomes transparent, the rest opaque.
    Spec gk;
    gk.ctype = 0;
    gk.w = 2;
    gk.h = 1;
    gk.trns = {0x00, 0x10};
    gk.px = [](uint32_t x, uint32_t) { return std::vector<uint16_t>{(uint16_t)(x ? 0x20 : 0x10)}; };
    png = encode(gk);
    p = decode_param(png, image);
    p.alpha_value = 0x55;
    ASSERT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), (2u << 16) | 1u);
    EXPECT_EQ(pixel(image, 8, 0, 0), (std::vector<uint8_t>{0x10, 0x10, 0x10, 0x00}));
    EXPECT_EQ(pixel(image, 8, 1, 0), (std::vector<uint8_t>{0x20, 0x20, 0x20, 0xFF}));

    // Gray + alpha keeps its own alpha.
    Spec ga;
    ga.ctype = 4;
    ga.w = 1;
    ga.h = 1;
    ga.px = [](uint32_t, uint32_t) { return std::vector<uint16_t>{0x90, 0x30}; };
    png = encode(ga);
    p = decode_param(png, image);
    p.alpha_value = 0x55;
    ASSERT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), (1u << 16) | 1u);
    EXPECT_EQ(pixel(image, 4, 0, 0), (std::vector<uint8_t>{0x90, 0x90, 0x90, 0x30}));
}

TEST(PngDec, InterlacedDecodesToTheSamePixels) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    Decoder d(api);
    Spec s = rgba_spec(5, 5);
    const std::vector<uint8_t> flat = encode(s);
    s.interlace = 1;
    const std::vector<uint8_t> adam7 = encode(s);
    std::vector<uint8_t> a(100, 0), b(100, 0);
    DecodeParam pa = decode_param(flat, a), pb = decode_param(adam7, b);
    ImageInfo info{};
    ASSERT_EQ(api.decode(d.handle, addr(&pa), 0, 0, 0, 0), (5u << 16) | 5u);
    ASSERT_EQ(api.decode(d.handle, addr(&pb), addr(&info), 0, 0, 0), (5u << 16) | 5u);
    EXPECT_EQ(a, b);
    EXPECT_EQ(info.flags, 1u);
}

// 16-bit input decodes only with attribute 1, and is stripped to its high byte.
TEST(PngDec, SixteenBitNeedsAttributeOne) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    Spec s;
    s.depth = 16;
    s.w = 2;
    s.h = 1;
    s.px = [](uint32_t x, uint32_t) {
        return std::vector<uint16_t>{(uint16_t)(0x1234 + x), 0xABCD, 0x00FF, 0xFF00};
    };
    const std::vector<uint8_t> png = encode(s);
    std::vector<uint8_t> image(8, 0xEE);
    DecodeParam p = decode_param(png, image);
    ImageInfo info{};
    Decoder eight(api, 1024, 0);
    EXPECT_EQ(api.decode(eight.handle, addr(&p), addr(&info), 0, 0, 0), kWorkMemory);
    EXPECT_EQ(info.bit_depth, 16u) << "info is written before the bit-depth gate";
    EXPECT_EQ(image[0], 0xEE);
    Decoder sixteen(api, 1024, 1);
    ASSERT_NE(sixteen.handle, 0u);
    ASSERT_EQ(api.decode(sixteen.handle, addr(&p), 0, 0, 0, 0), (2u << 16) | 1u);
    EXPECT_EQ(pixel(image, 8, 1, 0), (std::vector<uint8_t>{0x12, 0xAB, 0x00, 0xFF}));
}

TEST(PngDec, DecodeValidatesInTheModulesOrder) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    Decoder d(api);
    ASSERT_NE(d.handle, 0u);
    const std::vector<uint8_t> png = encode(rgba_spec(2, 2));
    std::vector<uint8_t> image(64, 0xCC);
    const DecodeParam good = decode_param(png, image);
    auto run = [&](const DecodeParam& p, ImageInfo* info = nullptr) {
        return api.decode(d.handle, addr(&p), addr(info), 0, 0, 0);
    };

    EXPECT_EQ(api.decode(0xDEAD0u, addr(&good), 0, 0, 0, 0), kHandle) << "forged handle";
    EXPECT_EQ(api.decode(d.handle + 8, addr(&good), 0, 0, 0, 0), kHandle) << "misaligned handle";
    EXPECT_EQ(api.decode(d.handle, 0, 0, 0, 0, 0), kAddr) << "null param";
    DecodeParam p = good;
    p.png_addr = 0;
    EXPECT_EQ(run(p), kAddr) << "null png";
    p = good;
    p.image_addr = 0;
    EXPECT_EQ(run(p), kAddr) << "null image";
    p = good;
    p.image_addr += 2;
    EXPECT_EQ(run(p), kAddr) << "image not 4-byte aligned";
    p = good;
    p.png_size = 0;
    EXPECT_EQ(run(p), kSize) << "empty png";
    p = good;
    p.image_size = 0;
    EXPECT_EQ(run(p), kSize) << "empty image";
    p = good;
    p.pixel_format = 2;
    EXPECT_EQ(run(p), kParam) << "unknown pixel format";
    p = good;
    p.alpha_value = 0x100;
    EXPECT_EQ(run(p), kParam) << "alpha_value > 0xFF";
    std::vector<uint8_t> garbage(32, 1);
    p = good;
    p.png_addr = addr(garbage.data());
    p.png_size = (uint32_t)garbage.size();
    EXPECT_EQ(run(p), kData) << "not a PNG";

    // Pitch rules (pitch != 0): all PARAM.
    p = good;
    p.pitch = 4;
    EXPECT_EQ(run(p), kParam) << "pitch < row";
    p.pitch = 10;
    EXPECT_EQ(run(p), kParam) << "pitch not a multiple of 4";
    p.pitch = 36;
    EXPECT_EQ(run(p), kParam) << "pitch * height > image_size (72 > 64)";
    p.pitch = 32;
    EXPECT_EQ(run(p), (2u << 16) | 2u) << "pitch * height == image_size";

    // pitch == 0 and too small an image is SIZE, and the info is still written.
    p = good;
    p.image_size = 15;
    ImageInfo info{};
    std::fill(image.begin(), image.end(), 0xCC);
    EXPECT_EQ(run(p, &info), kSize);
    EXPECT_EQ(info.width, 2u) << "info is written before the size checks";
    EXPECT_EQ(info.color_space, 19u);
    EXPECT_EQ(image[0], 0xCC) << "a refused decode writes no pixels";
}

// max_image_width is enforced after rounding both sides up to 8.
TEST(PngDec, DecodeEnforcesMaxImageWidth) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    Decoder d(api, 9);
    ASSERT_NE(d.handle, 0u);
    std::vector<uint8_t> image(17 * 4, 0);
    const std::vector<uint8_t> w16 = encode(rgba_spec(16, 1));
    DecodeParam p = decode_param(w16, image);
    EXPECT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), (16u << 16) | 1u)
        << "round8(16) <= round8(9)";
    const std::vector<uint8_t> w17 = encode(rgba_spec(17, 1));
    p = decode_param(w17, image);
    EXPECT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), kWorkMemory) << "round8(17) > round8(9)";
}

// The return packs width<<16|height only when both fit in 15 bits. Larger: 0 with an info
// struct, ADDR without one (nothing would tell the caller the size) — and then no decode.
TEST(PngDec, LargeDimensionsNeedAnInfoStruct) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    Decoder d(api, 32768);
    ASSERT_NE(d.handle, 0u);
    Spec s;
    s.ctype = 0;
    s.depth = 1;
    s.w = 32768;
    s.h = 1;
    s.px = [](uint32_t x, uint32_t) { return std::vector<uint16_t>{(uint16_t)(x == 32767)}; };
    const std::vector<uint8_t> png = encode(s);
    std::vector<uint8_t> image(32768 * 4, 0xEE);
    DecodeParam p = decode_param(png, image);
    EXPECT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), kAddr);
    EXPECT_EQ(image[0], 0xEE) << "no decode without an info struct";
    ImageInfo info{};
    EXPECT_EQ(api.decode(d.handle, addr(&p), addr(&info), 0, 0, 0), 0u);
    EXPECT_EQ(info.width, 32768u);
    EXPECT_EQ(image[0], 0x00) << "decoded";
    EXPECT_EQ(image[32767 * 4], 0xFF);
}

// pixel_format 0x84 (CLUT) is a valid format; its validation is the module's, but the index
// output is not implemented and is refused (loudly, on stderr) rather than decoded wrongly.
TEST(PngDec, ClutFormatIsValidatedThenRefused) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    Decoder d(api);
    std::vector<uint8_t> image(0x400 + 64, 0xEE);
    const std::vector<uint8_t> rgba = encode(rgba_spec(2, 2));
    DecodeParam p = decode_param(rgba, image);
    p.pixel_format = 0x84;
    EXPECT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), kParam) << "CLUT needs a palette PNG";
    Spec pal;
    pal.ctype = 3;
    pal.depth = 8;
    pal.w = 2;
    pal.h = 2;
    pal.plte = {1, 2, 3};
    pal.px = [](uint32_t, uint32_t) { return std::vector<uint16_t>{0}; };
    const std::vector<uint8_t> png = encode(pal);
    p = decode_param(png, image);
    p.pixel_format = 0x84;
    p.image_size = 0x400 + 3;
    EXPECT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), kSize) << "palette + 4 indices > size";
    p.image_size = (uint32_t)image.size();
    EXPECT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), kUnsupport);
    EXPECT_EQ(image[0], 0xEE);
}

#if defined(__linux__)
// P3: decoding into a page the renderer's write-watch has armed read-only must succeed and mark
// the watch dirty. Without the notify bracket the store goes through process_vm_writev, which
// returns EFAULT on the read-only page, and the decode fails with ADDR.
TEST(PngDec, DecodeIntoAWatchedTextureDisarmsAndDirtiesIt) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    Decoder d(api);
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const int fd = memfd_create("prosper-pngdec-ww", 0);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(ftruncate(fd, (off_t)page), 0);
    auto* tex = static_cast<uint8_t*>(mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    ASSERT_NE(tex, MAP_FAILED);
    std::memset(tex, 0xEE, page);
    host::guest_write_watch_set_fault_onstack(true);
    host::guest_write_watch_notify_direct_mapping_added(addr(tex), page, 0x7000000, 0x3);
    {
        host::GuestWriteWatch watch = host::GuestWriteWatch::create(addr(tex), page);
        ASSERT_TRUE(static_cast<bool>(watch)) << "the watch arms";
        ASSERT_EQ(watch.query(), host::GuestWriteWatchQuery::Unchanged);
        // Positive control: the page really is armed, so an unbracketed store is refused.
        const uint8_t probe = 1;
        EXPECT_FALSE(host::guest_write_exact(addr(tex) + page - 1, &probe, 1))
            << "control: a plain host store to the armed page fails";

        const std::vector<uint8_t> png = encode(rgba_spec(4, 4));
        DecodeParam p{};
        p.png_addr = addr(png.data());
        p.png_size = (uint32_t)png.size();
        p.image_addr = addr(tex);
        p.image_size = 64;
        EXPECT_EQ(api.decode(d.handle, addr(&p), 0, 0, 0, 0), (4u << 16) | 4u);
        EXPECT_EQ(tex[0], (uint8_t)rgba_pattern(0, 0)[0]) << "pixels landed";
        EXPECT_EQ(watch.query(), host::GuestWriteWatchQuery::Dirty) << "the renderer is told";
        watch.reset();
    }
    host::guest_write_watch_notify_direct_mapping_removed(addr(tex), page);
    host::guest_write_watch_set_fault_onstack(false);
    munmap(tex, page);
    close(fd);
}
#endif
