// hle_font.cpp - minimal libSceFont/libSceFontFt lifecycle for native-engine bring-up.
//
// Astro Bot initializes Sony Font, opens system and memory-backed faces, then uses its own GPU
// text path. Returning success without writing opaque handles made FontSystem assert during boot.
// This implementation supplies real opaque lifetimes and deterministic empty-text/metric results;
// it deliberately does not claim to rasterize Sony system fonts.
#include "dispatch.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>

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
struct FontFace {
    uint64_t magic = kFontMagic;
    float width = 16.0f;
    float height = 16.0f;
    float slant = 0.0f;
    float weight_x = 1.0f;
    float weight_y = 1.0f;
    void* renderer = nullptr;
};
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

FontLibrary g_library;
FontRenderer g_renderer;
uint8_t g_ft_selection[64]{};

FontFace* face(void* handle) {
    auto* f = static_cast<FontFace*>(handle);
    return f && f->magic == kFontMagic ? f : nullptr;
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

int32_t font_open(void*, uint64_t, uint64_t, const void*, void** out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    auto* f = new (std::nothrow) FontFace;
    if (!f) return static_cast<int32_t>(0x80540001u);
    *out = f;
    return 0;
}
int32_t font_open_memory(void* library, const void* bytes, uint32_t size,
                         const void* params, void** out) {
    (void)bytes; (void)size;
    return font_open(library, 0, 0, params, out);
}
int32_t font_open_instance(void*, void* source, void** out) {
    int32_t rc = font_open(nullptr, 0, 0, nullptr, out);
    if (!rc && face(source)) **reinterpret_cast<FontFace**>(out) = *face(source);
    return rc;
}
int32_t font_close(void* handle) {
    delete static_cast<FontFace*>(handle);
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
    if (auto* f = face(handle)) {
        f->width = std::max(width, 1.0f);
        f->height = std::max(height, 1.0f);
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

void fill_metrics(void* handle, GlyphMetrics* out) {
    if (!out) return;
    const auto* f = face(handle);
    const float h = f ? f->height : 16.0f;
    const float w = f ? f->width * 0.6f : 9.6f;
    *out = {w, h, 0.0f, h * 0.8f, w, w * 0.5f, h * 0.5f, h};
}
int32_t font_get_metrics(void* handle, uint32_t, GlyphMetrics* out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    fill_metrics(handle, out);
    return 0;
}
int32_t font_get_horizontal(void* handle, HorizontalLayout* out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    const auto* f = face(handle); const float h = f ? f->height : 16.0f;
    *out = {h * 0.8f, h * 1.2f, h};
    return 0;
}
int32_t font_get_vertical(void* handle, VerticalLayout* out) {
    if (!out) return static_cast<int32_t>(0x80540002u);
    const auto* f = face(handle); const float w = f ? f->width : 16.0f;
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
    R("n590hj5Oe-k", (HleFn)font_create_library, "sceFontCreateLibraryWithEdition");
    R("WaSFJoRWXaI", (HleFn)font_create_renderer, "sceFontCreateRendererWithEdition");
    R("exAxkyVLt0s", (HleFn)font_destroy_handle, "sceFontDestroyRenderer");
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
    R("kAenWy1Zw5o", (HleFn)font_ok, "sceFontRenderCharGlyphImageHorizontal");
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

    // Intentional no-op lifecycle/capability surface used during Astro's initialization.
    R("SsRbbCiWoGw", (HleFn)font_ok, "sceFontSupportSystemFonts");
    R("mz2iTY0MK4A", (HleFn)font_ok, "sceFontSupportExternalFonts");
    R("CUKn5pX-NVY", (HleFn)font_ok, "sceFontAttachDeviceCacheBuffer");
    R("7rogx92EEyc", (HleFn)font_ok, "sceFontCreateWritingLine");
    R("1+DgKL0haWQ", (HleFn)font_ok, "sceFontWritingLineClear");
    R("JQKWIsS9joE", (HleFn)font_ok, "sceFontWritingLineGetOrderingSpace");
    R("nlU2VnfpqTM", (HleFn)font_ok, "sceFontWritingLineGetRenderMetrics");
    R("+FYcYefsVX0", (HleFn)font_ok, "sceFontWritingLineRefersRenderStep");
    R("wyKFUOWdu3Q", (HleFn)font_ok, "sceFontWritingLineWritesOrder");
    R("8-zmgsxkBek", (HleFn)font_ok, "sceFontGlyphDefineAttribute");
}

} // namespace prosper
