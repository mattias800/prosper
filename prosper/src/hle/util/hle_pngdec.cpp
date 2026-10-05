// libScePngDec — PNG decode to 8-bit RGBA/BGRA (scePngDecCreate/Decode/Delete/
// ParseHeader/QueryMemorySize).
//
// All five were unregistered, so the dispatcher answered `0`. For Create that 0 is a null
// handle and for Decode/ParseHeader it is success over unwritten pixels and image structs —
// a title blits whatever its output buffer already held.
//
// The decode itself is real, against the vendored stb_image (STB_IMAGE_STATIC: prosper_core
// links statically into frontends that already instantiate stb_image once, so this TU keeps
// its own copy — same reason hle_font.cpp owns its stb_truetype instantiation). Header
// metadata (dimensions, color space, bit depth, interlace, tRNS) comes from a minimal
// bounds-checked IHDR/chunk walk, because stb reports pixels but not the header fields the
// ImageInfo struct carries. Context handles live in the guest's own work memory with a
// self-tag (invalidated on Delete), following the reference model.
//
// Evidence: struct layouts, error facility 0x80690001-05/10/12 and validation order are
// identical in two independent reimplementations (cross-checked field for field).
// CONFIDENCE: HIGH on layouts, errors and validation; MED on 16-bit behavior (refused with
// UNSUPPORT_DATA — a real facility code — rather than decoded) and on max_image_width being
// advisory (stored, never enforced, as in the reference).
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "host/memory/guest_memory_copy.hpp"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "stb_image.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace prosper {

using host::guest_read_exact;
using host::guest_write_exact;

#define HLE(name)                                                                                  \
    static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,      \
                                          uint64_t a4, uint64_t a5)

namespace {

// Zero-extended 32-bit form: these contracts return int, and on hardware writing eax
// zeroes upper rax — so 0x0000000080690003 is what a real failure leaves in rax, not the
// sign-extended form. Guests read eax either way; the u64 equality below pins the exact word.
constexpr uint64_t kPngErrAddr = 0x80690001ull;
constexpr uint64_t kPngErrSize = 0x80690002ull;
constexpr uint64_t kPngErrParam = 0x80690003ull;
constexpr uint64_t kPngErrHandle = 0x80690004ull;
constexpr uint64_t kPngErrWorkMemory = 0x80690005ull;
constexpr uint64_t kPngErrData = 0x80690010ull;
constexpr uint64_t kPngErrUnsupport = 0x80690011ull;
constexpr uint64_t kPngErrDecode = 0x80690012ull;

constexpr uint32_t kPngMemorySize = 0x20;
constexpr uint64_t kPngHandleAlign = 8;
constexpr uint32_t kPngMaxImageWidth = 1000000;
constexpr uint32_t kPngMaxPackedDim = 32767;

// CreateParam { u32 this_size @0; u32 attribute @4; u32 max_image_width @8 } — 0xC.
struct PngCreateParam {
    uint32_t this_size;
    uint32_t attribute;
    uint32_t max_image_width;
};
// ParseParam { const u8* png @0; u32 size @8; u32 reserved @12 } — 0x10.
struct PngParseParam {
    uint64_t png_addr;
    uint32_t png_size;
    uint32_t reserved;
};
// DecodeParam { const u8* png @0; u8* image @8; u32 png_size @16; u32 image_size @20;
// u16 pixel_format @24; u16 alpha_value @26; u32 pitch @28 } — 0x20.
struct PngDecodeParam {
    uint64_t png_addr;
    uint64_t image_addr;
    uint32_t png_size;
    uint32_t image_size;
    uint16_t pixel_format;
    uint16_t alpha_value;
    uint32_t pitch;
};
// ImageInfo { u32 w @0; u32 h @4; u16 color_space @8; u16 bit_depth @10; u32 flag @12 } — 0x10.
struct PngImageInfo {
    uint32_t width;
    uint32_t height;
    uint16_t color_space;
    uint16_t bit_depth;
    uint32_t flags;
};
// Context header in guest work memory; `self` is the live-handle tag.
struct PngContext {
    uint64_t self;
    uint32_t attribute;
    uint32_t max_image_width;
};

bool png_read(uint64_t addr, void* dst, size_t n) {
    return addr && guest_read_exact(addr, dst, n);
}
bool png_write(uint64_t addr, const void* src, size_t n) {
    return addr && guest_write_exact(addr, src, n);
}
uint64_t png_validate_handle(uint64_t handle) {
    if (handle == 0 || handle % kPngHandleAlign != 0) return kPngErrHandle;
    PngContext ctx{};
    if (!png_read(handle, &ctx, sizeof ctx)) return kPngErrHandle;
    if (ctx.self != handle) return kPngErrHandle;  // forged or already deleted
    return 0;
}

// Minimal IHDR/chunk walk over a host-side copy. Fills info; false when the bytes are not a
// structurally valid PNG (signature, IHDR first, sane lengths, no overruns).
bool png_parse_header(const uint8_t* p, size_t n, PngImageInfo* info, bool* has_alpha) {
    static const uint8_t kSig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (n < 33 || memcmp(p, kSig, 8) != 0) return false;
    auto u32be = [&](size_t o) -> uint32_t {
        return ((uint32_t)p[o] << 24) | ((uint32_t)p[o + 1] << 16) | ((uint32_t)p[o + 2] << 8) |
               (uint32_t)p[o + 3];
    };
    if (u32be(8) != 13 || memcmp(p + 12, "IHDR", 4) != 0) return false;
    const uint32_t w = u32be(16), h = u32be(20);
    const uint8_t bit_depth = p[24], color_type = p[25];
    if (w == 0 || h == 0) return false;
    if (bit_depth != 1 && bit_depth != 2 && bit_depth != 4 && bit_depth != 8 && bit_depth != 16) {
        return false;
    }
    uint16_t color_space = 0;
    bool alpha = false;
    switch (color_type) {
        case 0: color_space = 2; break;                                        // grayscale
        case 2: color_space = 3; break;                                        // RGB
        case 3: color_space = 4; break;                                        // palette
        case 4:
            color_space = 18;
            alpha = true;
            break;                         // gray+alpha
        case 6:
            color_space = 19;
            alpha = true;
            break;                         // RGBA
        default: return false;
    }
    const uint8_t interlace = p[28];
    if (interlace > 1) return false;
    // Walk chunks for tRNS (bounds-checked; CRCs skipped — integrity is the decoder's job).
    bool trns = false;
    size_t off = 33;
    while (off + 12 <= n) {
        const uint32_t len = u32be(off);
        if (len > n || off + 12 + (size_t)len > n) break;
        if (memcmp(p + off + 4, "tRNS", 4) == 0) trns = true;
        if (memcmp(p + off + 4, "IEND", 4) == 0) break;
        off += 12 + (size_t)len;
        if (off <= 33) break;  // wrapped; corrupt
    }
    info->width = w;
    info->height = h;
    info->color_space = color_space;
    info->bit_depth = bit_depth;
    info->flags = (interlace ? 1u : 0u) | (trns ? 2u : 0u);
    if (has_alpha) *has_alpha = alpha || trns;
    return true;
}

}  // namespace

// scePngDecQueryMemorySize(param*) -> 0x20, or the error Create would give for a bad param.
HLE(png_query_memory_size) {
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    if (!a0) return kPngErrParam;
    PngCreateParam param{};
    if (!png_read(a0, &param, sizeof param)) return kPngErrParam;
    if (param.attribute != 0 && param.attribute != 1) return kPngErrParam;
    if (param.max_image_width == 0 || param.max_image_width - 1 > kPngMaxImageWidth) {
        return kPngErrSize;
    }
    return kPngMemorySize;
}

// scePngDecCreate(param*, memory*, memory_size, handle**) -> 0; context placed in guest memory.
HLE(png_create) {
    (void)a4;
    (void)a5;
    if (!a0) return kPngErrParam;
    PngCreateParam param{};
    if (!png_read(a0, &param, sizeof param)) return kPngErrParam;
    if (param.attribute != 0 && param.attribute != 1) return kPngErrParam;
    if (param.max_image_width == 0 || param.max_image_width - 1 > kPngMaxImageWidth) {
        return kPngErrSize;
    }
    if (!a1 || !a3) return kPngErrAddr;
    if (a2 < kPngMemorySize) return kPngErrWorkMemory;
    // Aligned slot is at most 7 bytes in, so the 24-byte context always lands inside the
    // 0x20-byte work area the caller sized. The placement write itself is the writability
    // check: a torn write can only strand caller scratch, never anyone else's memory.
    const uint64_t aligned = (a1 + kPngHandleAlign - 1) & ~(kPngHandleAlign - 1);
    PngContext ctx{};
    ctx.self = aligned;
    ctx.attribute = param.attribute;
    ctx.max_image_width = param.max_image_width;
    if (!png_write(aligned, &ctx, sizeof ctx)) return kPngErrWorkMemory;
    if (!png_write(a3, &aligned, sizeof aligned)) {
        uint64_t dead = 0;
        png_write(aligned, &dead, sizeof dead);  // unpublish the tag; nothing handed out
        return kPngErrAddr;
    }
    return 0;
}

// scePngDecDelete(handle) -> 0; invalidates the tag so a second Delete fails.
HLE(png_delete) {
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    const uint64_t bad = png_validate_handle(a0);
    if (bad) return bad;
    const uint64_t dead = 0;
    if (!png_write(a0, &dead, sizeof dead)) return kPngErrHandle;
    return 0;
}

// scePngDecParseHeader(param*, info*) -> 0; IHDR/tRNS into info without decoding pixels.
HLE(png_parse_header) {
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    if (!a0) return kPngErrParam;
    PngParseParam param{};
    if (!png_read(a0, &param, sizeof param)) return kPngErrParam;
    if (!param.png_addr || !a1) return kPngErrAddr;
    if (param.png_size == 0) return kPngErrSize;
    std::vector<uint8_t> png(param.png_size);
    if (!png_read(param.png_addr, png.data(), png.size())) return kPngErrAddr;
    PngImageInfo info{};
    bool has_alpha = false;
    if (!png_parse_header(png.data(), png.size(), &info, &has_alpha)) return kPngErrData;
    if (!png_write(a1, &info, sizeof info)) return kPngErrAddr;
    return 0;
}

// scePngDecDecode(handle, param*, info*) -> width<<16|height (both <= 32767), else 0.
HLE(png_decode) {
    (void)a3;
    (void)a4;
    (void)a5;
    const uint64_t bad_handle = png_validate_handle(a0);
    if (bad_handle) return bad_handle;
    if (!a1) return kPngErrParam;
    PngDecodeParam param{};
    if (!png_read(a1, &param, sizeof param)) return kPngErrParam;
    if (!param.png_addr || !param.image_addr) return kPngErrAddr;
    if (param.png_size == 0 || param.image_size == 0) return kPngErrSize;
    if (param.pixel_format != 0 && param.pixel_format != 1) return kPngErrParam;
    if (param.png_size > 0x7FFFFFFFu) return kPngErrSize;  // stb takes an int size
    std::vector<uint8_t> png(param.png_size);
    if (!png_read(param.png_addr, png.data(), png.size())) return kPngErrAddr;
    PngImageInfo info{};
    bool has_alpha = false;
    if (!png_parse_header(png.data(), png.size(), &info, &has_alpha)) return kPngErrData;
    if (info.bit_depth == 16) return kPngErrUnsupport;  // 16-bit output not decoded
    // Bound the output BEFORE decoding so a forged header cannot make us allocate or write
    // past the guest's buffer. All in uint64: nothing here can wrap.
    const uint64_t row_size = (uint64_t)info.width * 4ull;
    const uint64_t pitch = param.pitch == 0 ? row_size : (uint64_t)param.pitch;
    if (pitch < row_size) return kPngErrParam;
    const uint64_t extent = ((uint64_t)info.height - 1ull) * pitch + row_size;
    if (extent > param.image_size) return kPngErrSize;
    // Writability of the full extent is checked by the fault-safe write below; first verify
    // the range head is mapped so a garbage pointer fails before decoding.
    uint8_t head = 0;
    if (!guest_read_exact(param.image_addr, &head, 1)) return kPngErrAddr;
    int w = 0, h = 0;
    stbi_uc* pixels = stbi_load_from_memory(png.data(), (int)png.size(), &w, &h, nullptr, 4);
    if (!pixels) return kPngErrDecode;
    bool ok = (uint32_t)w == info.width && (uint32_t)h == info.height;
    if (ok) {
        std::vector<uint8_t> out((size_t)extent, 0);
        const bool swap_rb = param.pixel_format == 1;
        const bool fill_alpha = !has_alpha;
        const uint8_t alpha = (uint8_t)param.alpha_value;
        for (uint32_t y = 0; y < info.height && ok; y++) {
            uint8_t* row = out.data() + (size_t)y * (size_t)pitch;
            memcpy(row, pixels + (size_t)y * (size_t)row_size, (size_t)row_size);
            for (uint32_t x = 0; x < info.width; x++) {
                uint8_t* px = row + (size_t)x * 4u;
                if (swap_rb) {
                    const uint8_t t = px[0];
                    px[0] = px[2];
                    px[2] = t;
                }
                if (fill_alpha) px[3] = alpha;
            }
        }
        ok = png_write(param.image_addr, out.data(), out.size());
        if (ok && a2) {
            PngImageInfo out_info = info;
            ok = png_write(a2, &out_info, sizeof out_info);
        }
    }
    stbi_image_free(pixels);
    if (!ok) return kPngErrAddr;
    if (info.width > kPngMaxPackedDim || info.height > kPngMaxPackedDim) return 0;
    return (uint64_t)(((uint32_t)info.width << 16) | info.height);
}

void register_pngdec_hle() {
    Hle::register_fn("m0uW+8pFyaw", (HleFn)png_create, "scePngDecCreate");
    Hle::register_fn("WC216DD3El4", (HleFn)png_decode, "scePngDecDecode");
    Hle::register_fn("QbD+eENEwo8", (HleFn)png_delete, "scePngDecDelete");
    Hle::register_fn("U6h4e5JRPaQ", (HleFn)png_parse_header, "scePngDecParseHeader");
    Hle::register_fn("-6srIGbLTIU", (HleFn)png_query_memory_size, "scePngDecQueryMemorySize");
}

}  // namespace prosper
