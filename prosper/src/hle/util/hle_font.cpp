// hle_font.cpp - libSceFont / libSceFontFt: lifecycle, metrics and REAL glyph rasterization.
//
// Two very different kinds of face reach this file, and the difference is the whole design:
//
//  * A MEMORY face (`sceFontOpenFontMemory`) is a font FILE the title hands us out of its own
//    dump. There is nothing to guess: the outlines, the advance widths and the vertical metrics
//    are all in those bytes. Such a face is rasterized for real, through the vendored
//    stb_truetype (third_party/stb -- read its README, this is the header that IS on the guest
//    path), and every metric reported for it is read out of the title's own font.
//  * A SYSTEM face (`sceFontOpenFontSet`) names a Sony system font that a dump does not contain
//    and prosper does not ship. There is no honest way to rasterize one, so those faces keep the
//    deterministic placeholder metrics this file has always produced. They are marked as such.
//
// Why the memory path had to become real (#2951): answering a value-returning contract with 0 and
// leaving the OUT-parameter untouched is not a harmless stub. *Metaphor: ReFantazio* asks
// `sceFontRenderCharGlyphImage` how big the glyph it just drew was; prosper wrote nothing, so the
// title read uninitialised stack (285,196,807), took it as the glyph cell width, asked its own
// renderer for a 285196807x70 texture, got one whose Gen5 T# was never built, looked up
// bytes-per-pixel from that all-zero T#'s FORMAT field, got 0, and divided by it. SIGFPE, five
// seconds into the boot, three subsystems away from the actual defect.
//
// The ABI facts below were derived from that title's own disassembly plus live register probes,
// not assumed; each one carries the evidence that fixed it.
#include "hle/dispatch/dispatch.hpp"

#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_STATIC
// stb_truetype pulls in malloc/free/assert/math through these; keep them explicit so the header
// never reaches for a global allocator we have not chosen.
#define STBTT_malloc(x, u)  ((void)(u), std::malloc(x))
#define STBTT_free(x, u)    ((void)(u), std::free(x))
#define STBTT_assert(x)     ((void)0)
#include "stb_truetype.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace prosper {
namespace {

constexpr uint64_t kLibraryMagic = 0x5052464f4e544c49ull; // "PRFONTLI"
constexpr uint64_t kFontMagic    = 0x5052464f4e54464eull; // "PRFONTFN"
constexpr uint64_t kRendererMagic= 0x5052464f4e545244ull; // "PRFONTRD"
constexpr uint64_t kStringMagic  = 0x5052464f4e545354ull; // "PRFONTST"

struct FontMemory {
    uint16_t kind;
    uint16_t attrs;
    uint32_t region_size;
    void* region;
    void* mspace;
    const void* callbacks;
    void* destroy;
    void* destroy_ctx;
    void* context1;
    void* context2;
};
static_assert(sizeof(FontMemory) == 64);

struct FontLibrary { uint64_t magic = kLibraryMagic; };
struct FontRenderer { uint64_t magic = kRendererMagic; };

// A parsed memory font. Shared, because `sceFontOpenFontInstance` clones a face and both clones
// must keep the same parsed outlines alive.
//
// The bytes are COPIED out of guest memory rather than referenced. stb_truetype keeps a pointer
// into whatever buffer it was initialised over, and the guest owns its buffer: it may unmap,
// reuse or overwrite that range at any point after the open call, and a rasterizer holding a
// dangling pointer would fault far away from the cause. A font file is ~200 KiB (Metaphor's is
// 180,424 bytes) and a title opens a handful, so the copy is not worth avoiding.
struct FontData {
    std::vector<uint8_t> bytes;
    stbtt_fontinfo info{};
    int ascent = 0, descent = 0, line_gap = 0;   // font units, from the font's own hhea/OS2
};

struct FontFace {
    uint64_t magic = kFontMagic;
    float width = 16.0f;
    float height = 16.0f;
    float slant = 0.0f;
    float weight_x = 1.0f;
    float weight_y = 1.0f;
    void* renderer = nullptr;
    // Non-null only for a face opened from the title's own font FILE. A system-font-set face
    // leaves this null and keeps the placeholder metrics -- see the file header.
    std::shared_ptr<FontData> data;
};

// The largest pixel scale and glyph bitmap prosper will act on.
//
// Not a style preference -- a bound. Every quantity that reaches the rasterizer comes from the
// guest: the pixel scale through `sceFontSetScalePixel`, the code point through the render call.
// On Windows those float arguments are additionally NOT reliably delivered at all -- the import
// trampoline remaps integer registers only, which is stated at
// src/host/image/exec_image_win.cpp's emit_sysv_to_ms_prologue (dispatch.hpp says the conversion
// happens in the trampoline, but not that it is integer-only; #2955) -- so
// a face there can legitimately be carrying an unreviewed scale. A clamp turns every one of those
// cases into a missing glyph instead of a multi-gigabyte allocation.
constexpr float kMaxPixelScale = 4096.0f;
constexpr int   kMaxGlyphDim   = 4096;

// The largest font FILE prosper will copy out of guest memory.
//
// The length is a guest-supplied uint32 and prosper cannot see how big the guest's buffer really
// is, so an oversized one is not a big allocation -- it is a read of that many bytes FROM guest
// memory, i.e. a host SIGSEGV, which no `catch (...)` can turn back into an error return. 64 MiB
// is far above any real font (Metaphor's is 180,424 bytes; a full CJK face is a few megabytes)
// and far below anything that can walk off a mapping.
constexpr uint32_t kMaxFontBytes = 64u * 1024u * 1024u;

FontLibrary g_library;
FontRenderer g_renderer;
uint8_t g_ft_selection[64]{};
// `sceFontSelectRendererFt` is the renderer-side twin of `sceFontSelectLibraryFt`: Metaphor calls
// it with 0 and hands the result straight to `sceFontCreateRendererWithEdition`
// (eboot+0x1000d2f..0x1000d4d), exactly as it hands SelectLibraryFt's result to
// CreateLibraryWithEdition. Same shape, so the same kind of opaque edition token -- a distinct
// blob, because the two are distinct editions and a title is entitled to tell them apart.
uint8_t g_ft_renderer_selection[64]{};
struct FontString { uint64_t magic = kStringMagic; uint32_t terminate_code = 0; };
struct FontGlyph { uint64_t magic = kFontMagic; };

struct GlyphMetrics {
    float width, height;
    float h_bearing_x, h_bearing_y, h_advance;
    float v_bearing_x, v_bearing_y, v_advance;
};
static_assert(sizeof(GlyphMetrics) == 32);
struct HorizontalLayout { float baseline, advance, decoration; };
struct VerticalLayout { float baseline, advance, decoration; };
struct WritingMetrics { float advance_x, advance_y, top, bottom, left, right; };

struct TextSource {
    uint64_t system;
    const void* start;
    const void* end;
    const void* current;
    void* parser;
    void* object;
    void* default_font;
    void* reserved[5];
};
static_assert(sizeof(TextSource) == 0x60);

struct TextCharacter {
    TextCharacter* prev;
    TextCharacter* next;
    void* order;
    void* font;
    void* shape;
    uint32_t code;
};

struct RenderSurface {
    void* buffer;
    int32_t width_bytes;
    int8_t pixel_size;
    uint8_t pad0;
    uint8_t style;
    uint8_t pad1;
    int32_t width;
    int32_t height;
    uint32_t scissor[4];
    uint64_t reserved[11];
};
static_assert(sizeof(RenderSurface) == 128);

FontFace* face(void* handle) {
    auto* f = static_cast<FontFace*>(handle);
    return f && f->magic == kFontMagic ? f : nullptr;
}

// The out-struct `sceFontRenderCharGlyphImage` fills in. 0x40 bytes.
//
// Derived from Metaphor, not assumed. Its caller at eboot+0x10013d0 builds both out-parameters as
// stack locals: `lea rcx,[rbp-0x80]` and `lea r8,[rbp-0x60]`, with the frame's stack cookie at
// [rbp-0x20]. That pins the two sizes exactly -- the metrics block is [rbp-0x80,rbp-0x60) = 0x20
// bytes (which is sizeof(GlyphMetrics), independently confirming that layout), and this block is
// [rbp-0x60,rbp-0x20) = 0x40 bytes.
//
// 0x40 is EXACT, not merely an upper bound from the cookie: the caller's first read back is
// `mov rax,QWORD PTR [rbp-0x28]` (eboot+0x1001412), eight bytes ending at the cookie, so it
// consumes this block right up to its last byte. Nothing live sits inside the range this file
// memsets.
//
// SIX fields are read back, across FOUR read instructions, at eboot+0x1001412..0x100143f:
//   1001412  mov    rax,[rbp-0x28]   -> +0x38 and +0x3c
//   100141a  mov    eax,[rbp-0x50]   -> +0x10
//   1001420  vmovsd xmm0,[rbp-0x38]  -> +0x28 and +0x2c
//   1001432  vmovss xmm1,[rbp-0x30]  -> +0x30
// The instruction listing is here because the count kept coming out wrong: this comment first said
// "four", review corrected it to "five" by incrementing that number, and re-review found that
// neither counts anything in the range. Two reads are qwords covering two fields each. The number
// mattered because the wrong one told the next reader there was nothing to look at at +0x30.
//   +0x10 -> the glyph cache entry's row STRIDE. It has to be the surface's pitch: the engine
//            later blits with `src + col + stride*row` straight out of the surface buffer
//            (eboot+0x1001a5b..0x1001a63), so any other value walks the wrong rows.
//   +0x28/+0x2c -> floats, truncated to int by the caller, recording WHERE in the surface the
//            glyph landed.
//   +0x30 -> a float, truncated to int exactly as the placement pair above is, then stored into
//            the engine's own glyph record at [glyph+0x34] (eboot+0x1001432..0x100143f). It is
//            CONSUMED -- what it MEANS is not derivable from anything in this title's code, which
//            only copies it. prosper writes 0.
//            **CONFIDENCE: LOW that 0 is the right answer.** It is the right *default* -- before
//            this file existed the field was uninitialised stack, so 0 is strictly an improvement
//            and at least deterministic -- but it is a guess about a value the engine keeps. If a
//            glyph-placement defect ever shows up on this title, start here.
//   +0x38/+0x3c -> the drawn size in pixels, as signed ints. These are the two the title maxes
//            against its configured cell size to size its glyph atlas texture -- the pair that
//            were uninitialised stack in #2951.
// The genuinely unread remainder is zeroed: prosper does not know what it means, and zero is the
// only answer that does not invent one.
struct RenderResult {
    uint32_t reserved0[4];   // +0x00
    int32_t  pitch;          // +0x10
    uint32_t reserved1[5];   // +0x14
    float    x;              // +0x28
    float    y;              // +0x2c
    float    consumed_0x30;  // +0x30  read by the caller; meaning underived. See above.
    uint32_t reserved3;      // +0x34
    int32_t  width;          // +0x38
    int32_t  height;         // +0x3c
};
static_assert(sizeof(RenderResult) == 0x40);
static_assert(offsetof(RenderResult, pitch)  == 0x10);
static_assert(offsetof(RenderResult, x)      == 0x28);
static_assert(offsetof(RenderResult, consumed_0x30) == 0x30);
static_assert(offsetof(RenderResult, width)  == 0x38);
static_assert(offsetof(RenderResult, height) == 0x3c);

bool finite_positive(float v) { return std::isfinite(v) && v > 0.0f; }

// Pixel scale for a face, in stb_truetype's units.
//
// CONFIDENCE: MED on the exact meaning of "scale pixel". Sony's name says pixels and gives two
// axes; stb offers both `ScaleForPixelHeight` (ascent-descent spans h) and
// `ScaleForMappingEmToPixels` (the em square spans h). We take the first, because that is the
// ordinary reading of "the glyph is h pixels tall" and it is the conservative one -- it makes
// glyphs no larger than the requested box, so a title that sizes an atlas cell from the requested
// height never needs to grow it. Nothing in the guest evidence distinguishes them: Metaphor sizes
// its cell as max(configured, measured) and so is correct under either.
bool face_scale(const FontFace* f, float* sx, float* sy) {
    if (!f || !f->data) return false;
    const float h = f->height;
    if (!finite_positive(h) || h > kMaxPixelScale) return false;
    const float y = stbtt_ScaleForPixelHeight(&f->data->info, h);
    if (!finite_positive(y)) return false;
    float x = y;
    if (finite_positive(f->width) && f->width <= kMaxPixelScale) x = y * (f->width / h);
    if (!finite_positive(x)) return false;
    *sx = x; *sy = y;
    return true;
}

// Distance from the top of a line box to the baseline, in pixels. This is the SAME number
// `sceFontGetHorizontalLayout` reports, and it must stay that way: a caller positions a glyph by
// subtracting the two, so a disagreement between the layout call and the renderer shows up as
// text sliding off its own line.
float face_baseline(const FontFace* f, float sy) {
    return f->data ? (float)f->data->ascent * sy : f->height * 0.8f;
}

// Real metrics for one code point, out of the face's own font file. Returns false for a face that
// has no font behind it (a system-font-set face).
bool real_metrics(const FontFace* f, uint32_t code, GlyphMetrics* out) {
    float sx = 0.0f, sy = 0.0f;
    if (!face_scale(f, &sx, &sy)) return false;
    const stbtt_fontinfo* info = &f->data->info;
    int advance = 0, lsb = 0;
    stbtt_GetCodepointHMetrics(info, (int)code, &advance, &lsb);
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    stbtt_GetCodepointBitmapBox(info, (int)code, sx, sy, &x0, &y0, &x1, &y1);
    const float w = (float)(x1 - x0), h = (float)(y1 - y0);
    // stb reports the box with y growing DOWN from the baseline, so y0 is negative for anything
    // that rises above it; the bearing Sony asks for is the height above the baseline, hence -y0.
    const float line = (float)(f->data->ascent - f->data->descent + f->data->line_gap) * sy;
    out->width = w;
    out->height = h;
    out->h_bearing_x = (float)x0;
    out->h_bearing_y = (float)(-y0);
    out->h_advance = (float)advance * sx;
    // Vertical writing. The font carries no vhea/vmtx here and stb exposes none, so these are
    // derived from the horizontal box the way a layout engine derives them when a font has no
    // vertical table: the glyph is centred on the vertical baseline and advances by one line.
    // CONFIDENCE: MED -- derived from this font's own numbers, but the convention is a
    // convention. No title in the corpus reads them yet.
    out->v_bearing_x = -w * 0.5f;
    out->v_bearing_y = (float)(-y0);
    out->v_advance = line;
    return true;
}

// Blit one rasterized glyph into a render surface, clipped to both the surface and its scissor.
// Returns the number of pixels actually covered in `dw`/`dh` (0 if entirely clipped away).
void blit_glyph(const RenderSurface* surf, const uint8_t* glyph, int gw, int gh,
                int dest_x, int dest_y, int* dw, int* dh) {
    *dw = 0; *dh = 0;
    auto* dst = static_cast<uint8_t*>(surf->buffer);
    const int pitch = surf->width_bytes;
    if (!dst || pitch <= 0 || surf->width <= 0 || surf->height <= 0) return;
    // One coverage byte per pixel is the only surface format this writes. Metaphor declares
    // exactly that (`sceFontRenderSurfaceInit(..., pixel_size=1, ...)` at eboot+0x100152a, and the
    // live probe agrees), and nothing else in the corpus reaches here -- but a wider surface would
    // otherwise get 8-bit coverage smeared across it with no diagnostic, which is a silent skip
    // where the charter asks for a loud one.
    //
    // Refusing is fail-visible IN THE GAME as well as on stderr, which is the real justification
    // and is better than "it stays in bounds": reporting a zero drawn size makes this title
    // substitute U+005F. Its wrapper re-reads `result+0x38`/`+0x3c` at eboot+0x100168f/0x100169c
    // and, if either is zero, re-renders `_` (eboot+0x10016a9, `mov esi,0x5f`) -- exempting only
    // U+0020 and U+3000, which is also why the whitespace `return 0` below is correct rather than
    // lossy. So an unsupported surface format shows up as a line of underscores, not as nothing.
    if (surf->pixel_size != 1) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                         "[font] RenderSurface pixel_size=%d is unsupported; only 1 (8-bit "
                         "coverage) is implemented. Glyphs will not be drawn. CONFIDENCE: LOW -- "
                         "no title in the corpus exercises this, so the layout is underived.\n",
                         (int)surf->pixel_size);
        }
        return;
    }

    // Clip box: the surface, intersected with its scissor. `font_surface_init` seeds the scissor to
    // the whole surface, so an untouched one clips nothing extra, and an EMPTY one clips
    // everything -- it is honoured unconditionally rather than being read as "no scissor".
    //
    // That matters because the two failure directions are not symmetric: treating an empty scissor
    // as absent draws MORE of the guest's buffer than it asked for, over glyph cells it may be
    // holding. Treating an absent one as empty only loses a glyph. And the choice needs no
    // interpretation of the field, because {0,0,0,0} is empty under BOTH readings of the trailing
    // pair -- extents or a second corner.
    //
    // CONFIDENCE: MED on that pair being {w, h} rather than {x1, y1}. It matches the seeding
    // `font_surface_init` has always done, so prosper is at least self-consistent, and the only
    // guest call in the corpus cannot tell them apart: Metaphor asks for (0, 0, 0x400, 0x400) on a
    // 0x400x0x400 surface (eboot+0x100156c), which is the whole surface either way.
    int64_t cx0 = 0, cy0 = 0, cx1 = surf->width, cy1 = surf->height;
    const int64_t sx0 = (int64_t)surf->scissor[0], sy0 = (int64_t)surf->scissor[1];
    const int64_t sw = (int64_t)surf->scissor[2], sh = (int64_t)surf->scissor[3];
    cx0 = std::max(cx0, sx0); cy0 = std::max(cy0, sy0);
    cx1 = std::min(cx1, sx0 + sw); cy1 = std::min(cy1, sy0 + sh);
    // A row must also fit the declared pitch -- a title is free to give a pitch narrower than the
    // width it also declared, and the pitch is what indexes memory.
    cx1 = std::min<int64_t>(cx1, pitch);
    if (cx1 <= cx0 || cy1 <= cy0) return;

    int min_x = INT32_MAX, min_y = INT32_MAX, max_x = INT32_MIN, max_y = INT32_MIN;
    for (int gy = 0; gy < gh; ++gy) {
        const int64_t dy = (int64_t)dest_y + gy;
        if (dy < cy0 || dy >= cy1) continue;
        const uint8_t* src_row = glyph + (size_t)gy * (size_t)gw;
        uint8_t* dst_row = dst + (size_t)dy * (size_t)pitch;
        for (int gx = 0; gx < gw; ++gx) {
            const int64_t dx = (int64_t)dest_x + gx;
            if (dx < cx0 || dx >= cx1) continue;
            const uint8_t cov = src_row[gx];
            if (!cov) continue;
            // Coverage combines by max, so two glyphs sharing a pixel do not darken past opaque
            // and a caller may compose a run into one surface.
            uint8_t& d = dst_row[dx];
            if (cov > d) d = cov;
            min_x = std::min<int>(min_x, (int)dx); max_x = std::max<int>(max_x, (int)dx);
            min_y = std::min<int>(min_y, (int)dy); max_y = std::max<int>(max_y, (int)dy);
        }
    }
    if (max_x >= min_x) { *dw = max_x - min_x + 1; *dh = max_y - min_y + 1; }
}

int32_t font_memory_init(FontMemory* mem, void* region, uint32_t region_size,
                         const void* callbacks, void* mspace, void* destroy,
                         void* destroy_ctx) {
    if (!mem) return static_cast<int32_t>(0x80540002u);
    std::memset(mem, 0, sizeof(*mem));
    mem->kind = region ? 1 : 0;
    mem->region_size = region_size;
    mem->region = region;
    mem->mspace = mspace;
    mem->callbacks = callbacks;
    mem->destroy = destroy;
    mem->destroy_ctx = destroy_ctx;
    return 0;
}

const void* font_select_library_ft(int) { return g_ft_selection; }
// sceFontSelectRendererFt -- the renderer-side twin, see g_ft_renderer_selection.
const void* font_select_renderer_ft(int) { return g_ft_renderer_selection; }

int32_t font_create_library(const FontMemory*, const void*, uint64_t, void** out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    *out = &g_library;
    return 0;
}
int32_t font_create_renderer(const FontMemory*, const void*, uint64_t, void** out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    *out = &g_renderer;
    return 0;
}
int32_t font_destroy_handle(void** handle) {
    if (handle) *handle = nullptr;
    return 0;
}
// sceFontMemoryTerm(FontMemory*). `font_memory_init` allocates nothing -- it only fills in the
// caller's own descriptor -- so releasing it IS clearing that descriptor, and there is no hidden
// state left behind. Named rather than pointed at the generic ok-stub so the fact that this is
// the complete behaviour, and not an unimplemented one, survives in the registry dump.
int32_t font_memory_term(FontMemory* mem) {
    if (!mem) return static_cast<int32_t>(0x80540002u);
    std::memset(mem, 0, sizeof(*mem));
    return 0;
}

int32_t font_open(void*, uint64_t, uint64_t, const void*, void** out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    auto* f = new (std::nothrow) FontFace;
    if (!f) return static_cast<int32_t>(0x80540001u);
    *out = f;
    return 0;
}
// sceFontOpenFontMemory(library, bytes, size, params, out)
//
// The title's own font FILE. Parse it: from here on this face reports real outlines and real
// metrics. Argument order confirmed live on Metaphor -- rdi=library, rsi=bytes, edx=size, ecx=0,
// r8=out, with the bytes reading `00 01 00 00` (sfnt 1.0) and 17 tables at 180,424 bytes.
//
// A blob we cannot parse is REFUSED, not accepted quietly. Reporting success over a font we did
// not read is how a caller ends up positioning text with numbers nobody produced -- which is the
// defect class this whole path was rewritten for (#2951).
int32_t font_open_memory(void* library, const void* bytes, uint32_t size,
                         const void* params, void** out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    *out = nullptr;
    if (!bytes || size < 12 || size > kMaxFontBytes)
        return static_cast<int32_t>(0x80540002u);

    auto data = std::make_shared<FontData>();
    try {
        data->bytes.assign(static_cast<const uint8_t*>(bytes),
                           static_cast<const uint8_t*>(bytes) + size);
    } catch (...) {
        return static_cast<int32_t>(0x80540001u);
    }
    const int offset = stbtt_GetFontOffsetForIndex(data->bytes.data(), 0);
    if (offset < 0 || !stbtt_InitFont(&data->info, data->bytes.data(), offset))
        return static_cast<int32_t>(0x80540002u);
    stbtt_GetFontVMetrics(&data->info, &data->ascent, &data->descent, &data->line_gap);
    if (data->ascent <= data->descent) return static_cast<int32_t>(0x80540002u);

    int32_t rc = font_open(library, 0, 0, params, out);
    if (rc != 0) return rc;
    if (auto* f = face(*out)) f->data = std::move(data);
    return 0;
}
int32_t font_open_instance(void*, void* source, void** out) {
    int32_t rc = font_open(nullptr, 0, 0, nullptr, out);
    if (!rc && face(source)) **reinterpret_cast<FontFace**>(out) = *face(source);
    return rc;
}
int32_t font_close(void* handle) {
    // The magic check is not cosmetic any more. FontFace gained a shared_ptr, so it is no longer
    // trivially destructible, and `delete` on a bogus handle would run a refcount decrement
    // through whatever the pointer happens to address -- an arbitrary write, where before this
    // change the same mistake was only a bad free. Validate first.
    if (!handle) return 0;
    auto* f = face(handle);
    if (!f) return static_cast<int32_t>(0x80540002u);
    // Scrub the magic before freeing so a double close is caught by the check above deterministically
    // rather than by luck. It usually would be anyway -- glibc's tcache overwrites the first 16 bytes
    // of a freed chunk -- but "usually, on this allocator" is not a property to rely on.
    f->magic = 0;
    delete f;
    return 0;
}

int32_t font_bind_renderer(void* handle, void* renderer) {
    if (auto* f = face(handle)) f->renderer = renderer;
    return 0;
}
int32_t font_unbind_renderer(void* handle) {
    if (auto* f = face(handle)) f->renderer = nullptr;
    return 0;
}
int32_t font_set_scale(void* handle, float width, float height) {
    // Clamped at both ends. The floor was always here; the ceiling and the NaN rejection are new,
    // because these two floats now size a real rasterization. See kMaxPixelScale for why a guest
    // value reaching here cannot simply be trusted -- especially on Windows, where the import
    // trampoline does not deliver float arguments at all.
    auto sane = [](float v, float fallback) {
        if (!std::isfinite(v) || v < 1.0f) return fallback;
        return std::min(v, kMaxPixelScale);
    };
    if (auto* f = face(handle)) {
        f->width = sane(width, f->width);
        f->height = sane(height, f->height);
    }
    return 0;
}
int32_t font_set_slant(void* handle, float slant) {
    if (auto* f = face(handle)) f->slant = slant;
    return 0;
}
int32_t font_set_weight(void* handle, float x, float y, uint32_t) {
    if (auto* f = face(handle)) { f->weight_x = x; f->weight_y = y; }
    return 0;
}

// Placeholder metrics for a face with no font behind it (a system font set prosper does not
// have). Deliberately unchanged from what this file has always produced: a title that only wants
// somewhere to put a cursor keeps working, and nothing here claims to describe a Sony font.
void fill_metrics(void* handle, GlyphMetrics* out) {
    if (!out) return;
    const auto* f = face(handle);
    const float h = f ? f->height : 16.0f;
    const float w = f ? f->width * 0.6f : 9.6f;
    *out = {w, h, 0.0f, h * 0.8f, w, w * 0.5f, h * 0.5f, h};
}

// sceFontGetCharGlyphMetrics / sceFontGetRenderCharGlyphMetrics.
//
// One implementation for both, and the reason is a property of prosper rather than of Sony: the
// "Render" variant differs by the render effects in force, and prosper applies none -- no slant,
// no emboldening -- so the two necessarily agree here. If effects ever become real, they split.
int32_t font_get_metrics(void* handle, uint32_t code, GlyphMetrics* out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    const auto* f = face(handle);
    if (f && f->data && real_metrics(f, code, out)) return 0;
    fill_metrics(handle, out);
    return 0;
}
int32_t font_get_horizontal(void* handle, HorizontalLayout* out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    const auto* f = face(handle);
    float sx = 0.0f, sy = 0.0f;
    if (f && f->data && face_scale(f, &sx, &sy)) {
        const auto& d = *f->data;
        // baseline and line advance out of the font's own hhea. The third field is the decoration
        // (underline) position, which neither this font's exposed tables nor any guest evidence
        // pins down, so it keeps the placeholder relationship rather than acquiring an invented
        // one. CONFIDENCE: HIGH on the derivation of baseline/advance from the font's tables --
        // but it INHERITS the MED above on what "scale pixel" means, and a label cannot exceed its
        // weakest input, so read the absolute values as MED. LOW on decoration.
        *out = {(float)d.ascent * sy,
                (float)(d.ascent - d.descent + d.line_gap) * sy,
                f->height};
        return 0;
    }
    const float h = f ? f->height : 16.0f;
    *out = {h * 0.8f, h * 1.2f, h};
    return 0;
}
int32_t font_get_vertical(void* handle, VerticalLayout* out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    const auto* f = face(handle);
    float sx = 0.0f, sy = 0.0f;
    if (f && f->data && face_scale(f, &sx, &sy)) {
        const auto& d = *f->data;
        const float line = (float)(d.ascent - d.descent + d.line_gap) * sy;
        // The line advance is the font's own. The other two are NOT: this font carries no
        // vhea/vmtx and stb exposes none, so the vertical baseline is placed at the centre of the
        // line box and the decoration falls back to the requested cell width.
        // **CONFIDENCE: LOW** on both -- they are conventions, not measurements, and *Astro Bot*
        // imports `sceFontGetVerticalLayout`, so unlike most of this file they do reach a title.
        // Nothing observed so far depends on them; if vertical text ever looks wrong, start here.
        *out = {line * 0.5f, line, f->width};
        return 0;
    }
    const float w = f ? f->width : 16.0f;
    *out = {w * 0.5f, w * 1.2f, w};
    return 0;
}

int32_t font_generate_glyph(void*, uint32_t, const void*, void** out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    *out = new (std::nothrow) FontGlyph;
    return *out ? 0 : static_cast<int32_t>(0x80540001u);
}
int32_t font_delete_glyph(const FontMemory*, void** glyph) {
    if (glyph) { delete static_cast<FontGlyph*>(*glyph); *glyph = nullptr; }
    return 0;
}

void font_surface_init(RenderSurface* out, void* buffer, int width_bytes, int pixel_size,
                       int width, int height) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));
    out->buffer = buffer; out->width_bytes = width_bytes; out->pixel_size = (int8_t)pixel_size;
    out->width = width; out->height = height;
    out->scissor[2] = width > 0 ? (uint32_t)width : 0;
    out->scissor[3] = height > 0 ? (uint32_t)height : 0;
}

// sceFontRenderSurfaceSetScissor(surface, x, y, w, h)
//
// Argument order from Metaphor's call at eboot+0x100156c: rdi=surface, esi=0, edx=0, ecx=width,
// r8d=height -- i.e. the same {x, y, w, h} quadruple `font_surface_init` already seeds into
// `scissor[]`, which is what makes the two agree without a conversion.
int32_t font_surface_set_scissor(RenderSurface* surface, int32_t x, int32_t y,
                                 int32_t w, int32_t h) {
    if (!surface) return static_cast<int32_t>(0x80540002u);
    // Clamping a negative origin to 0 has to take the same amount off the extent, or the rectangle
    // slides instead of being cropped: (-10, 0, 20, 20) is x in [-10,10), and clamping the origin
    // alone would turn it into [0,20) -- twice the requested area, half of it never asked for.
    // Same fail-open asymmetry as the empty-scissor case in blit_glyph -- and it inherits that
    // function's `CONFIDENCE: MED` on the trailing pair being {w, h}: under {x1, y1} this
    // over-clips, which is the safe direction and consistent with `font_surface_init`'s seeding.
    auto crop = [](int origin, int extent, uint32_t* out_origin, uint32_t* out_extent) {
        int64_t o = origin, e = std::max(extent, 0);
        if (o < 0) { e = std::max<int64_t>(0, e + o); o = 0; }
        *out_origin = (uint32_t)o;
        *out_extent = (uint32_t)e;
    };
    crop(x, w, &surface->scissor[0], &surface->scissor[2]);
    crop(y, h, &surface->scissor[1], &surface->scissor[3]);
    return 0;
}

// The real work behind sceFontRenderCharGlyphImage: rasterize `code` into `surface` with the pen
// box at (pen_x, pen_y), and report what was drawn through both out-parameters.
//
// Placement. The pen argument is the top-left of the LINE BOX, not the baseline origin -- derived
// from Metaphor rather than assumed. At eboot+0x10015de..0x10015f8 it computes
// `x = -metrics.h_bearing_x` and `y = metrics.h_bearing_y - layout.baseline`, then blits the
// result out of the surface starting at (0,0) (eboot+0x1001a38..0x1001a63). Those two facts only
// reconcile under one convention:
//     glyph_left = pen_x + h_bearing_x
//     glyph_top  = pen_y + layout.baseline - h_bearing_y
// which puts Metaphor's glyph at exactly (0,0), where its blit expects it. Note this stays
// self-consistent whatever Sony's absolute convention is, because prosper supplies BOTH sides of
// it -- the metrics and the baseline the caller subtracts are ours.
int32_t render_glyph(void* handle, uint32_t code, RenderSurface* surface,
                     float pen_x, float pen_y, bool have_pen,
                     GlyphMetrics* metrics, RenderResult* result) {
    // Both out-parameters are cleared FIRST, before any early return can take us out of here.
    // This is the actual regression surface of #2951: every path out of this function must leave
    // the caller's buffers describing something, because the caller reads them unconditionally
    // and cannot tell an untouched buffer from a reported zero.
    if (result) std::memset(result, 0, sizeof(*result));
    if (metrics) std::memset(metrics, 0, sizeof(*metrics));

    auto* f = face(handle);
    if (!f || !surface) return static_cast<int32_t>(0x80540002u);
    if (result) result->pitch = surface->width_bytes;

    if (metrics) {
        if (!(f->data && real_metrics(f, code, metrics))) fill_metrics(handle, metrics);
    }
    // No font behind this face: there is nothing to rasterize and nothing to pretend. The report
    // says zero pixels were drawn, which is true, and the caller's own fallback handles it.
    if (!f->data) return 0;

    GlyphMetrics m{};
    float sx = 0.0f, sy = 0.0f;
    if (!real_metrics(f, code, &m) || !face_scale(f, &sx, &sy)) return 0;

    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    stbtt_GetCodepointBitmapBox(&f->data->info, (int)code, sx, sy, &x0, &y0, &x1, &y1);
    const int gw = x1 - x0, gh = y1 - y0;
    if (gw <= 0 || gh <= 0) return 0;                       // whitespace: legitimately empty
    if (gw > kMaxGlyphDim || gh > kMaxGlyphDim) return 0;   // see kMaxGlyphDim

    std::vector<uint8_t> bitmap;
    try { bitmap.assign((size_t)gw * (size_t)gh, 0); }
    catch (...) { return static_cast<int32_t>(0x80540001u); }
    stbtt_MakeCodepointBitmap(&f->data->info, bitmap.data(), gw, gh, gw, sx, sy, (int)code);

    const float baseline = face_baseline(f, sy);
    // Windows cannot deliver the pen (see the registration below); there it is the line-box
    // origin, which is what a single-glyph-per-surface caller like Metaphor asks for anyway.
    const float ox = have_pen ? pen_x : -m.h_bearing_x;
    const float oy = have_pen ? pen_y : (m.h_bearing_y - baseline);
    if (!std::isfinite(ox) || !std::isfinite(oy)) return static_cast<int32_t>(0x80540002u);
    const double left = (double)ox + (double)m.h_bearing_x;
    const double top  = (double)oy + (double)baseline - (double)m.h_bearing_y;
    if (std::fabs(left) > 1e7 || std::fabs(top) > 1e7) return 0;   // pen nowhere near the surface

    int dw = 0, dh = 0;
    blit_glyph(surface, bitmap.data(), gw, gh, (int)std::lround(left), (int)std::lround(top),
               &dw, &dh);
    if (result) {
        result->x = (float)left;
        result->y = (float)top;
        result->width = dw;
        result->height = dh;
    }
    return 0;
}

#if defined(_WIN32)
// Windows: the import-stub trampoline converts the guest's SysV call to the MS ABI by remapping
// INTEGER registers only (exec_image_win.cpp emit_sysv_to_ms_prologue) -- it never touches xmm.
// That is #2955, a general defect of the trampoline rather than of this library; delete this #if
// when it is fixed.
// A declaration carrying `float x, float y` in their guest positions would therefore read the two
// pointer arguments after them out of the wrong slots and write through whatever happened to be
// in guest r9 and the guest stack. That is an arbitrary memory write, not a wrong glyph, so the
// Windows arm takes the five integer arguments only -- which the trampoline DOES place correctly
// -- and renders at the line-box origin.
int32_t font_render_char_glyph_image(void* handle, uint32_t code, RenderSurface* surface,
                                     GlyphMetrics* metrics, RenderResult* result) {
    return render_glyph(handle, code, surface, 0.0f, 0.0f, /*have_pen=*/false, metrics, result);
}
#else
int32_t font_render_char_glyph_image(void* handle, uint32_t code, RenderSurface* surface,
                                     float x, float y, GlyphMetrics* metrics,
                                     RenderResult* result) {
    return render_glyph(handle, code, surface, x, y, /*have_pen=*/true, metrics, result);
}
#endif

// sceFontRenderCharGlyphImageHorizontal (kAenWy1Zw5o). Astro Bot is the corpus's only importer,
// and its call sites (eboot+0xed9b46, +0xed9ba5, +0xedcb3a, +0xedcb70 of the flattened eboot,
// read the way Metaphor's were for #2951) pin the SAME signature as the non-Horizontal variant:
// rdi=font handle, esi=char code, rdx=surface, xmm0/xmm1=pen x/y, rcx=out metrics (0x20 bytes,
// bounded by the caller's own -0x30 local), r8=out result (0x40, bounded by the metrics buffer).
// One site loads xmm2 before the call; it is dead pen bookkeeping, not a third float parameter --
// the integer out-pointers sit in rcx/r8 across all four call sites in both functions. With no
// extra argument and both out-structures matching what render_glyph already fills, the honest
// implementation is the same honest renderer. CONFIDENCE: HIGH on the signature (four call sites
// in two independent functions); MED that Astro never exercises more than it (one glyph per
// surface, like Metaphor; vertical-writing layouts are unobserved here).
#if defined(_WIN32)
int32_t font_render_char_glyph_image_horizontal(void* handle, uint32_t code,
                                                RenderSurface* surface, GlyphMetrics* metrics,
                                                RenderResult* result) {
    return render_glyph(handle, code, surface, 0.0f, 0.0f, /*have_pen=*/false, metrics, result);
}
#else
int32_t font_render_char_glyph_image_horizontal(void* handle, uint32_t code,
                                                RenderSurface* surface, float x, float y,
                                                GlyphMetrics* metrics, RenderResult* result) {
    return render_glyph(handle, code, surface, x, y, /*have_pen=*/true, metrics, result);
}
#endif

int32_t font_text_source_init(TextSource* out, const void* text, uint32_t size,
                              void* parser, void* object) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    std::memset(out, 0, sizeof(*out));
    out->start = text; out->current = text;
    out->end = text ? static_cast<const uint8_t*>(text) + size : nullptr;
    out->parser = parser; out->object = object;
    return 0;
}
int32_t font_text_default(TextSource* source, void* handle) {
    if (source) source->default_font = handle;
    return 0;
}
int32_t font_text_writing_form(TextSource* source, int32_t form) {
    if (source) source->system = (uint32_t)form;
    return 0;
}

int32_t font_create_string(const FontMemory*, TextSource*, const void*, void** out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    *out = new (std::nothrow) FontString;
    return *out ? 0 : static_cast<int32_t>(0x80540001u);
}
int32_t font_destroy_string(void** string) {
    if (string) {
        delete static_cast<FontString*>(*string);
        *string = nullptr;
    }
    return 0;
}
uint32_t font_string_terminate(void* string) {
    auto* s = static_cast<FontString*>(string);
    return s && s->magic == kStringMagic ? s->terminate_code : 0;
}
TextCharacter* font_string_characters(void*, uint32_t* count) {
    if (count) *count = 0;
    return nullptr;
}
const TextCharacter* font_string_render_characters(void*, TextCharacter*, TextCharacter*,
                                                    uint32_t* count) {
    if (count) *count = 0;
    return nullptr;
}
int32_t font_string_writing_form(void*) { return 0; }

int32_t font_writing_init(void* writing, void*, const TextCharacter*) {
    if (writing) std::memset(writing, 0, 0x100);
    return 0;
}
int32_t font_writing_metrics(void*, WritingMetrics* out) {
    if (out) std::memset(out, 0, sizeof(*out));
    return 0;
}
const void* font_writing_step(void*) { return nullptr; }
TextCharacter* font_writing_character(void*, void* letter_step) {
    if (letter_step) std::memset(letter_step, 0, 0x40);
    return nullptr;
}

int32_t font_character_bidi(TextCharacter*, int* out) { if (out) *out = 0; return 0; }
int32_t font_character_code(TextCharacter* ch, void** handle, uint32_t* code) {
    if (handle) *handle = ch ? ch->font : nullptr;
    if (code) *code = ch ? ch->code : 0;
    return 0;
}
int32_t font_character_order(TextCharacter* ch, void** order) {
    if (order) *order = ch ? ch->order : nullptr;
    return 0;
}
uint32_t font_character_whitespace(TextCharacter* ch) {
    if (!ch) return 0;
    return ch->code == ' ' || ch->code == '\t' || ch->code == '\n' || ch->code == '\r';
}
TextCharacter* font_character_next(TextCharacter* ch) { return ch ? ch->next : nullptr; }

uint64_t font_ok(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) { return 0; }

} // namespace

void register_font_hle() {
    auto R = [](const char* nid, HleFn fn, const char* name) { Hle::register_fn(nid, fn, name); };
    R("whrS4oksXc4", (HleFn)font_memory_init, "sceFontMemoryInit");
    R("oM+XCzVG3oM", (HleFn)font_select_library_ft, "sceFontSelectLibraryFt");
    R("Xx974EW-QFY", (HleFn)font_select_renderer_ft, "sceFontSelectRendererFt");
    R("n590hj5Oe-k", (HleFn)font_create_library, "sceFontCreateLibraryWithEdition");
    R("WaSFJoRWXaI", (HleFn)font_create_renderer, "sceFontCreateRendererWithEdition");
    R("exAxkyVLt0s", (HleFn)font_destroy_handle, "sceFontDestroyRenderer");
    R("FXP359ygujs", (HleFn)font_destroy_handle, "sceFontDestroyLibrary");
    R("h6hIgxXEiEc", (HleFn)font_memory_term, "sceFontMemoryTerm");
    R("SSCaczu2aMQ", (HleFn)font_destroy_string, "sceFontDestroyString");
    R("PEjv7CVDRYs", (HleFn)font_ok, "sceFontDestroyWritingLine");
    R("cKYtVmeSTcw", (HleFn)font_open, "sceFontOpenFontSet");
    R("KXUpebrFk1U", (HleFn)font_open_memory, "sceFontOpenFontMemory");
    R("JzCH3SCFnAU", (HleFn)font_open_instance, "sceFontOpenFontInstance");
    R("vzHs3C8lWJk", (HleFn)font_close, "sceFontCloseFont");
    R("3OdRkSjOcog", (HleFn)font_bind_renderer, "sceFontBindRenderer");
    R("1QjhKxrsOB8", (HleFn)font_unbind_renderer, "sceFontUnbindRenderer");
    R("N1EBMeGhf7E", (HleFn)font_set_scale, "sceFontSetScalePixel");
    R("6vGCkkQJOcI", (HleFn)font_set_scale, "sceFontSetupRenderScalePixel");
    R("TMtqoFQjjbA", (HleFn)font_set_slant, "sceFontSetEffectSlant");
    R("lz9y9UFO2UU", (HleFn)font_set_slant, "sceFontSetupRenderEffectSlant");
    R("v0phZwa4R5o", (HleFn)font_set_weight, "sceFontSetEffectWeight");
    R("XIGorvLusDQ", (HleFn)font_set_weight, "sceFontSetupRenderEffectWeight");
    R("IQtleGLL5pQ", (HleFn)font_get_metrics, "sceFontGetRenderCharGlyphMetrics");
    R("imxVx8lm+KM", (HleFn)font_get_horizontal, "sceFontGetHorizontalLayout");
    R("3BrWWFU+4ts", (HleFn)font_get_vertical, "sceFontGetVerticalLayout");
    R("C-4Qw5Srlyw", (HleFn)font_generate_glyph, "sceFontGenerateCharGlyph");
    R("LHDoRWVFGqk", (HleFn)font_delete_glyph, "sceFontDeleteGlyph");
    R("gdUCnU0gHdI", (HleFn)font_surface_init, "sceFontRenderSurfaceInit");
    R("vRxf4d0ulPs", (HleFn)font_surface_set_scissor, "sceFontRenderSurfaceSetScissor");
    R("L97d+3OgMlE", (HleFn)font_get_metrics, "sceFontGetCharGlyphMetrics");
    R("3G4zhgKuxE8", (HleFn)font_render_char_glyph_image, "sceFontRenderCharGlyphImage");
    R("kAenWy1Zw5o", (HleFn)font_render_char_glyph_image_horizontal,
      "sceFontRenderCharGlyphImageHorizontal");
    R("oaJ1BpN2FQk", (HleFn)font_text_source_init, "sceFontTextSourceInit");
    R("eCRMCSk96NU", (HleFn)font_text_default, "sceFontTextSourceSetDefaultFont");
    R("OqQKX0h5COw", (HleFn)font_text_writing_form, "sceFontTextSourceSetWritingForm");
    R("MO24vDhmS4E", (HleFn)font_create_string, "sceFontCreateString");
    R("ObkDGDBsVtw", (HleFn)font_string_terminate, "sceFontStringGetTerminateCode");
    R("Avv7OApgCJk", (HleFn)font_string_characters, "sceFontStringRefersTextCharacters");
    R("hq5LffQjz-s", (HleFn)font_string_render_characters, "sceFontStringRefersRenderCharacters");
    R("o1vIEHeb6tw", (HleFn)font_string_writing_form, "sceFontStringGetWritingForm");
    R("fD5rqhEXKYQ", (HleFn)font_writing_init, "sceFontWritingInit");
    R("fljdejMcG1c", (HleFn)font_writing_metrics, "sceFontWritingGetRenderMetrics");
    R("W-2WOXEHGck", (HleFn)font_writing_step, "sceFontWritingRefersRenderStep");
    R("f4Onl7efPEY", (HleFn)font_writing_character, "sceFontWritingRefersRenderStepCharacter");
    R("6DFUkCwQLa8", (HleFn)font_character_bidi, "sceFontCharacterGetBidiLevel");
    R("zN3+nuA0SFQ", (HleFn)font_character_code, "sceFontCharacterGetTextFontCode");
    R("mxgmMj-Mq-o", (HleFn)font_character_order, "sceFontCharacterGetTextOrder");
    R("SaRlqtqaCew", (HleFn)font_character_whitespace, "sceFontCharacterLooksWhiteSpace");
    R("BkjBP+YC19w", (HleFn)font_character_next, "sceFontCharacterRefersTextNext");
    R("nlU2VnfpqTM", (HleFn)font_ok, "sceFontWritingLineGetRenderMetrics");
    R("+FYcYefsVX0", (HleFn)font_ok, "sceFontWritingLineRefersRenderStep");
    R("wyKFUOWdu3Q", (HleFn)font_ok, "sceFontWritingLineWritesOrder");
    R("8-zmgsxkBek", (HleFn)font_ok, "sceFontGlyphDefineAttribute");
}

} // namespace prosper
