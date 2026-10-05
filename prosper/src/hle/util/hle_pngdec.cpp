// libScePngDec — PNG decode to 8-bit RGBA/BGRA (scePngDecCreate/Decode/Delete/
// ParseHeader/QueryMemorySize).
//
// All five were unregistered, so the dispatcher answered `0`. For Create that 0 is a null
// handle and for Decode/ParseHeader it is success over unwritten pixels and image structs —
// a title blits whatever its output buffer already held.
//
// The pixel decode is real, through the vendored stb_image (STB_IMAGE_STATIC: prosper_core links
// statically into frontends that already instantiate stb_image once, so this TU keeps its own
// copy — the same reason hle_font.cpp owns its stb_truetype instantiation). stb_image parses
// bytes the title supplies; third_party/stb/README.md records why that is acceptable and the
// bounds this file applies before it is reached. Header metadata (dimensions, colour space, bit
// depth, interlace, tRNS) comes from a bounded chunk walk of our own that stops at the first
// IDAT, because stb reports pixels but not the header fields the ImageInfo struct carries, and
// because the module itself reads no further than that before it validates.
//
// Evidence: the shipped 3.20 module (libScePngDec.sprx, plaintext; offsets below are image RVAs).
// It is libpng behind a thin validation layer, and every validation, error code and the order
// they are applied in here was read out of its disassembly:
//   QueryMemorySize 0x0d0, Create 0x120, ParseHeader 0x1f0, Decode 0x4c0, RGBA row worker 0x7a0,
//   CLUT worker 0x990, Delete 0xbc0, libpng IHDR check 0x12b0 (the error-state word it sets
//   picks DATA / UNSUPPORT / DECODE in the setjmp handlers at 0x342 and 0x656).
// CONFIDENCE: HIGH on layouts, validation order and error codes for the paths named above.
// MED on which libpng failure maps to which code for malformed chunk streams (DECODE for
// structural/truncation errors is the module's default error state; DATA for the signature and
// IHDR field violations, UNSUPPORT for a dimension above libpng's 1,000,000 limit). Known
// differences from libpng, all on malformed input only: chunk CRCs are not verified, and stb_image
// refuses a tRNS chunk that libpng would ignore as a benign error (wrong length, or on a colour
// type that already has alpha), so such a PNG fails to DECODE here where the console decodes it.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "host/memory/guest_memory_copy.hpp"
#include "host/memory/guest_write_watch.hpp"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
// libpng's user width/height limit, which the module keeps at its default: anything larger is
// refused before stb_image is reached (png_read_header below), and this makes stb agree.
#define STBI_MAX_DIMENSIONS 1000000
#include "stb_image.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
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
// What the module returns when its work-memory allocator or libpng's own structs cannot be
// created (Decode 0x5b1/0x61f/0x635). Used here for the host-side equivalent: an allocation of
// the decode buffers failing.
constexpr uint64_t kPngErrFatal = 0x80690020ull;

constexpr uint32_t kPngCreateParamSize = 0xC;
constexpr uint32_t kPngMaxImageWidthMinus1 = 999999;  // `max_w - 1 > 999999` -> PARAM
constexpr uint32_t kPngLibpngDimLimit = 1000000;      // libpng user width/height maximum
constexpr uint32_t kPngMaxPackedDim = 0x7FFF;
constexpr uint16_t kPngFormatRgba = 0;
constexpr uint16_t kPngFormatBgra = 1;
constexpr uint16_t kPngFormatClut = 0x84;

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
// The context Create writes at round_up(memory, 16) (Create 0x191-0x1d7). Decode reads
// attribute and the rounded width back from here, so a guest that tampers with its own work
// memory sees the module's behaviour, not ours.
struct PngContext {
    uint64_t self;              // @0x00: live-handle tag; Delete zeroes it
    uint32_t this_size;         // @0x08: always 0xC
    uint32_t attribute;         // @0x0C: 0 = 8-bit output only, 1 = 16-bit input accepted
    uint32_t max_width_round8;  // @0x10: round_up(max_image_width, 8)
    uint32_t work_size;         // @0x14: the libpng arena size inside the work memory
};
static_assert(sizeof(PngContext) == 0x18, "context header Create writes is 0x18 bytes");

bool png_read(uint64_t addr, void* dst, size_t n) {
    return addr && guest_read_exact(addr, dst, n);
}
bool png_write(uint64_t addr, const void* src, size_t n) {
    return addr && guest_write_exact(addr, src, n);
}

// ((max_w + 0x47) & ~7) * (8 << attribute) + base, in u32 as the module computes it
// (QueryMemorySize 0x0fa-0x10d; Create stores the same product + 0xD000 as the arena size).
uint32_t png_work_product(uint32_t max_w, uint32_t attribute) {
    return ((max_w + 0x47u) & ~7u) * (8u << (attribute & 31u));
}

// Shared Query/Create validation (0x0d0-0x0f0, 0x120-0x158). 0 when the param is acceptable.
uint64_t png_check_create_param(uint64_t addr, PngCreateParam* param) {
    if (!png_read(addr, param, sizeof *param)) return kPngErrAddr;
    if (param->this_size != kPngCreateParamSize) return kPngErrSize;
    if (param->attribute > 1) return kPngErrParam;
    if (param->max_image_width - 1u > kPngMaxImageWidthMinus1) return kPngErrParam;
    return 0;
}

// Decode/Delete handle test (0x4e5-0x501, 0xbc5-0xbd3): non-null, 16-byte aligned, and the
// first quadword still holds its own address.
uint64_t png_validate_handle(uint64_t handle, PngContext* ctx) {
    if (handle == 0 || (handle & 0xF) != 0) return kPngErrHandle;
    if (!png_read(handle, ctx, sizeof *ctx)) return kPngErrHandle;
    if (ctx->self != handle) return kPngErrHandle;  // forged or already deleted
    return 0;
}

uint32_t be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

// What png_read_info leaves behind, as far as the ImageInfo struct and the decode path need it.
struct PngHeader {
    uint32_t width = 0;
    uint32_t height = 0;
    uint8_t bit_depth = 0;
    uint8_t color_type = 0;
    uint8_t interlace = 0;
    bool trns = false;  // libpng's num_trans != 0: a tRNS chunk it accepted
};

// The bounded equivalent of the module's png_sig_cmp + png_read_info: reads the signature and
// then only the 8-byte chunk headers (plus the 13-byte IHDR body) straight out of guest memory,
// stopping at the first IDAT. Nothing here allocates from a guest-supplied size, so a png_size of
// 4 GiB over a 2 KiB file costs a few small reads, as it does on the console.
uint64_t png_read_header(uint64_t png_addr, uint32_t png_size, PngHeader* out) {
    static const uint8_t kSig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (png_size < 8) return kPngErrData;
    uint8_t sig[8];
    if (!png_read(png_addr, sig, sizeof sig)) return kPngErrAddr;
    if (memcmp(sig, kSig, sizeof sig) != 0) return kPngErrData;

    PngHeader h;
    bool have_ihdr = false, have_plte = false;
    uint32_t plte_entries = 0;
    uint64_t off = 8;
    for (;;) {
        // libpng's read callback png_errors on a short read; its default error state is DECODE.
        if (off + 8 > png_size) return kPngErrDecode;
        uint8_t ch[8];
        if (!png_read(png_addr + off, ch, sizeof ch)) return kPngErrAddr;
        const uint32_t len = be32(ch);
        const uint8_t* type = ch + 4;
        if (len > 0x7FFFFFFFu) return kPngErrDecode;  // png_get_uint_31
        const bool is_ihdr = memcmp(type, "IHDR", 4) == 0;
        const bool is_idat = memcmp(type, "IDAT", 4) == 0;
        const bool is_plte = memcmp(type, "PLTE", 4) == 0;
        if (!have_ihdr && !is_ihdr) return kPngErrDecode;  // "Missing IHDR before ..."
        if (is_idat) {
            if (h.color_type == 3 && !have_plte) return kPngErrDecode;  // "Missing PLTE before IDAT"
            break;
        }
        if (off + 12 + (uint64_t)len > png_size) return kPngErrDecode;
        if (is_ihdr) {
            if (have_ihdr || len != 13) return kPngErrDecode;
            uint8_t b[13];
            if (!png_read(png_addr + off + 8, b, sizeof b)) return kPngErrAddr;
            h.width = be32(b);
            h.height = be32(b + 4);
            h.bit_depth = b[8];
            h.color_type = b[9];
            h.interlace = b[12];
            if (h.width > 0x7FFFFFFFu || h.height > 0x7FFFFFFFu) return kPngErrDecode;
            // png_check_IHDR: the dimension limit sets error state 2 and errors first.
            if (h.width > kPngLibpngDimLimit || h.height > kPngLibpngDimLimit) {
                return kPngErrUnsupport;
            }
            // Every other IHDR violation sets error state 1.
            bool bad = h.width == 0 || h.height == 0 || b[10] != 0 || b[11] != 0 || h.interlace > 1;
            switch (h.color_type) {
                case 0: bad |= !(h.bit_depth == 1 || h.bit_depth == 2 || h.bit_depth == 4 ||
                                 h.bit_depth == 8 || h.bit_depth == 16); break;
                case 3: bad |= !(h.bit_depth == 1 || h.bit_depth == 2 || h.bit_depth == 4 ||
                                 h.bit_depth == 8); break;
                case 2: case 4: case 6: bad |= !(h.bit_depth == 8 || h.bit_depth == 16); break;
                default: bad = true; break;
            }
            if (bad) return kPngErrData;
            have_ihdr = true;
        } else if (is_plte) {
            if (have_plte) return kPngErrDecode;
            if (h.color_type == 3 && (len == 0 || len % 3 != 0 || len / 3 > 256)) {
                return kPngErrDecode;
            }
            have_plte = true;
            plte_entries = len / 3;
        } else if (memcmp(type, "tRNS", 4) == 0) {
            // libpng accepts tRNS only where it means something and ignores it (benign error)
            // otherwise; the ImageInfo flag and the alpha fill both follow num_trans.
            if (h.color_type == 0) h.trns = len == 2;
            else if (h.color_type == 2) h.trns = len == 6;
            else if (h.color_type == 3) h.trns = have_plte && len > 0 && len <= plte_entries;
        } else if ((type[0] & 0x20) == 0) {
            return kPngErrDecode;  // an unknown critical chunk (IEND before IDAT included)
        }
        off += 12 + (uint64_t)len;
    }
    *out = h;
    return 0;
}

PngImageInfo png_image_info(const PngHeader& h) {
    // png_color_type -> SCE colour space, the table at .rodata 0xd688: 0->2, 2->3, 3->4, 4->18, 6->19.
    static const uint16_t kColorSpace[7] = {2, 2, 3, 4, 18, 2, 19};
    PngImageInfo info{};
    info.width = h.width;
    info.height = h.height;
    info.color_space = kColorSpace[h.color_type <= 6 ? h.color_type : 0];
    info.bit_depth = h.bit_depth;
    info.flags = (h.interlace ? 1u : 0u) | (h.trns ? 2u : 0u);
    return info;
}

// pixel_format 0x84 is the CLUT output: a 0x400-byte RGBA palette followed by one 8-bit index
// per pixel (worker 0x990). Its validation is reproduced so a guest is refused for the same
// reasons as on the console, but the index output itself is not implemented — stb_image only
// returns expanded colour. Fail-visible: a loud line names the format instead of a quiet code.
uint64_t png_decode_clut(const PngDecodeParam& param, const PngContext& ctx, const PngHeader& h) {
    const uint64_t w = h.width, rows = h.height;
    if (param.pitch != 0) {
        if (param.pitch < w) return kPngErrParam;
        const uint64_t total = rows * (uint64_t)param.pitch;
        if (total > 0x7FFFFBFFull || total + 0x400 > param.image_size) return kPngErrParam;
    } else {
        const uint64_t total = rows * w;
        if (total + 0x400 > param.image_size) return kPngErrSize;
        if (total >= 0x7FFFFC00ull) return kPngErrUnsupport;
    }
    if (((w + 7) & ~7ull) > ctx.max_width_round8) return kPngErrWorkMemory;
    if (h.bit_depth > (8u << (ctx.attribute & 31u))) return kPngErrWorkMemory;
    if (h.color_type != 3 || h.bit_depth > 8) return kPngErrParam;
    static std::atomic<bool> said{false};
    if (!said.exchange(true)) {
        std::fprintf(stderr,
                     "[prosper] libScePngDec: scePngDecDecode pixel_format 0x84 (8-bit CLUT: "
                     "RGBA palette + indices) is NOT IMPLEMENTED; returning 0x80690011 for a "
                     "%ux%u %u-bit palette PNG. The console decodes this.\n",
                     h.width, h.height, h.bit_depth);
    }
    return kPngErrUnsupport;
}

}  // namespace

// scePngDecQueryMemorySize(param*) -> work-memory bytes Create needs, or an error.
HLE(png_query_memory_size) {
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    PngCreateParam param{};
    if (const uint64_t err = png_check_create_param(a0, &param)) return err;
    return png_work_product(param.max_image_width, param.attribute) + 0xD090u;
}

// scePngDecCreate(param*, memory*, memory_size, handle**) -> 0; context placed in guest memory.
HLE(png_create) {
    (void)a4;
    (void)a5;
    PngCreateParam param{};
    if (const uint64_t err = png_check_create_param(a0, &param)) return err;
    if (!a1) return kPngErrAddr;
    const uint32_t product = png_work_product(param.max_image_width, param.attribute);
    if ((uint32_t)a2 < product + 0xD090u) return kPngErrSize;  // memory_size is a u32 (edx)
    if (!a3) return kPngErrAddr;
    const uint64_t aligned = (a1 + 15) & ~15ull;
    PngContext ctx{};
    ctx.self = aligned;
    ctx.this_size = kPngCreateParamSize;
    ctx.attribute = param.attribute;
    ctx.max_width_round8 = (param.max_image_width + 7u) & ~7u;
    ctx.work_size = product + 0xD000u;
    if (!png_write(aligned, &ctx, sizeof ctx)) return kPngErrAddr;
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
    PngContext ctx{};
    if (const uint64_t bad = png_validate_handle(a0, &ctx)) return bad;
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
    PngParseParam param{};
    if (!png_read(a0, &param, sizeof param)) return kPngErrAddr;
    if (!param.png_addr) return kPngErrAddr;
    if (param.png_size == 0) return kPngErrSize;
    if (param.reserved != 0) return kPngErrParam;
    if (!a1) return kPngErrAddr;
    PngHeader h;
    if (const uint64_t err = png_read_header(param.png_addr, param.png_size, &h)) return err;
    const PngImageInfo info = png_image_info(h);
    if (!png_write(a1, &info, sizeof info)) return kPngErrAddr;
    return 0;
}

// scePngDecDecode(handle, param*, info*) -> width<<16|height when both fit in 15 bits, else 0.
HLE(png_decode) {
    (void)a3;
    (void)a4;
    (void)a5;
    PngContext ctx{};
    if (const uint64_t bad = png_validate_handle(a0, &ctx)) return bad;
    PngDecodeParam param{};
    if (!png_read(a1, &param, sizeof param)) return kPngErrAddr;
    if (!param.png_addr || !param.image_addr || (param.image_addr & 3) != 0) return kPngErrAddr;
    if (param.png_size == 0 || param.image_size == 0) return kPngErrSize;
    if (param.pixel_format > kPngFormatBgra && param.pixel_format != kPngFormatClut) {
        return kPngErrParam;
    }
    if (param.alpha_value > 0xFF) return kPngErrParam;

    PngHeader h;
    if (const uint64_t err = png_read_header(param.png_addr, param.png_size, &h)) return err;
    // The info struct is written as soon as the header is read (0x6c8-0x728) — before any size,
    // pitch or bit-depth check — so a refused decode still reports what the image is.
    if (a2) {
        const PngImageInfo info = png_image_info(h);
        if (!png_write(a2, &info, sizeof info)) return kPngErrAddr;
    }
    const bool packable = (h.width | h.height) <= kPngMaxPackedDim;
    if (!a2 && !packable) return kPngErrAddr;  // 0x741-0x754: nowhere to report the size
    const uint64_t result = packable ? (((uint64_t)h.width << 16) + h.height) : 0;

    if (param.pixel_format == kPngFormatClut) return png_decode_clut(param, ctx, h);

    // RGBA/BGRA worker validation (0x7b6-0x870), in its order.
    const uint64_t row_size = (uint64_t)h.width * 4;
    uint64_t stride = row_size;
    if (param.pitch != 0) {
        if (param.pitch < row_size || (param.pitch & 3) != 0) return kPngErrParam;
        const uint64_t total = (uint64_t)h.height * param.pitch;
        if (total >= 0x80000000ull || total > param.image_size) return kPngErrParam;
        stride = param.pitch;
    } else {
        const uint64_t total = (uint64_t)h.height * row_size;
        if (total > param.image_size) return kPngErrSize;
        if (total >= 0x80000000ull) return kPngErrUnsupport;
    }
    if ((((uint64_t)h.width + 7) & ~7ull) > ctx.max_width_round8) return kPngErrWorkMemory;
    // 16-bit input needs attribute 1; the module then strips to 8 bits (0x8a0), which is what
    // stb_image's 16->8 conversion (high byte) does too.
    if (h.bit_depth > (8u << (ctx.attribute & 31u))) return kPngErrWorkMemory;

    // stb_image takes an int length. Unreachable for any real asset; refused loudly, not silently.
    if (param.png_size > 0x7FFFFFFFu) {
        std::fprintf(stderr,
                     "[prosper] libScePngDec: png_size 0x%x exceeds the 2 GiB the decoder "
                     "accepts; returning 0x80690011\n",
                     param.png_size);
        return kPngErrUnsupport;
    }
    std::vector<uint8_t> png;
    try {
        png.resize(param.png_size);
    } catch (const std::bad_alloc&) {
        return kPngErrFatal;
    }
    if (!png_read(param.png_addr, png.data(), png.size())) return kPngErrAddr;
    int w = 0, hgt = 0;
    stbi_uc* pixels = stbi_load_from_memory(png.data(), (int)png.size(), &w, &hgt, nullptr, 4);
    if (!pixels) return kPngErrDecode;
    if ((uint32_t)w != h.width || (uint32_t)hgt != h.height) {
        stbi_image_free(pixels);
        return kPngErrDecode;
    }

    // Convert in place: R/B swap for BGRA, and the alpha fill libpng's png_set_filler applies
    // when the image has neither an alpha channel nor an accepted tRNS (0x8b2-0x8dc).
    const bool swap_rb = param.pixel_format == kPngFormatBgra;
    const bool fill_alpha = (h.color_type & 4) == 0 && !h.trns;
    const uint8_t alpha = (uint8_t)param.alpha_value;
    const size_t pixel_count = (size_t)h.width * h.height;
    for (size_t i = 0; i < pixel_count; i++) {
        uint8_t* px = pixels + i * 4;
        if (swap_rb) {
            const uint8_t t = px[0];
            px[0] = px[2];
            px[2] = t;
        }
        if (fill_alpha) px[3] = alpha;
    }

    // One row of row_size bytes at image + y*stride, as the module's png_read_row loop does
    // (0x941-0x96c): the bytes between row_size and pitch belong to the caller and are never
    // touched. The range is announced to the renderer's write-watch first — a reused texture
    // backing may be armed read-only, and process_vm_writev would EFAULT on it (#1144 B5; the
    // same bracket hle_file.cpp's read_full uses).
    const uint64_t extent = ((uint64_t)h.height - 1) * stride + row_size;
    host::guest_write_watch_notify_host_write(param.image_addr, extent);
    bool ok = true;
    if (stride == row_size) {
        ok = png_write(param.image_addr, pixels, (size_t)extent);
    } else {
        for (uint32_t y = 0; y < h.height && ok; y++) {
            ok = png_write(param.image_addr + (uint64_t)y * stride, pixels + (size_t)y * row_size,
                           (size_t)row_size);
        }
    }
    host::guest_write_watch_notify_host_write_done(param.image_addr, extent);
    stbi_image_free(pixels);
    if (!ok) return kPngErrAddr;
    return result;
}

void register_pngdec_hle() {
    Hle::register_fn("m0uW+8pFyaw", (HleFn)png_create, "scePngDecCreate");
    Hle::register_fn("WC216DD3El4", (HleFn)png_decode, "scePngDecDecode");
    Hle::register_fn("QbD+eENEwo8", (HleFn)png_delete, "scePngDecDelete");
    Hle::register_fn("U6h4e5JRPaQ", (HleFn)png_parse_header, "scePngDecParseHeader");
    Hle::register_fn("-6srIGbLTIU", (HleFn)png_query_memory_size, "scePngDecQueryMemorySize");
}

}  // namespace prosper
