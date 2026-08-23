// test_font_render.cpp — libSceFont's memory-font path: real rasterization, and the out-parameter
// contract that #2951 was.
//
// The defect this guards is NOT "text looked wrong". It was that
// `sceFontRenderCharGlyphImage` answered 0 — success — and wrote nothing to either of its two
// out-parameters. *Metaphor: ReFantazio* read the untouched stack back as the glyph's size
// (285,196,807 x 4), used it to size a texture, and divided by the bytes-per-pixel of the
// all-zero descriptor that texture ended up with. So the assertion that matters here is that
// every field the guest reads is WRITTEN, on every path out — the poison arms below exist for
// exactly that, because a test that only checks "the numbers look sane" passes just as happily
// against a buffer nobody touched if the residue happens to be plausible.
//
// The font is synthesized in-process rather than shipped as an asset: it keeps the test hermetic,
// and it makes the expected metrics arithmetic rather than measured.

#include "hle/dispatch/dispatch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace prosper;

static int fails;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

// ---------------------------------------------------------------------------------------------
// A minimal, valid TrueType font: two glyphs (.notdef and a square mapped to 'A').
//
// unitsPerEm 1000, ascent 800, descent -200, lineGap 0. The square spans x 100..700 and y 0..700
// in font units, so at a 64 px scale (64 / (800 + 200) = 0.064) it is 38.4 x 44.8 px and sits on
// the baseline, and the baseline is 800 * 0.064 = 51.2 px below the top of the line box.
// ---------------------------------------------------------------------------------------------
namespace {

void u8(std::vector<uint8_t>& v, uint32_t x) { v.push_back((uint8_t)x); }
void u16(std::vector<uint8_t>& v, uint32_t x) { u8(v, x >> 8); u8(v, x & 0xff); }
void u32v(std::vector<uint8_t>& v, uint32_t x) { u16(v, x >> 16); u16(v, x & 0xffff); }
void i16(std::vector<uint8_t>& v, int32_t x) { u16(v, (uint32_t)(x & 0xffff)); }
void pad4(std::vector<uint8_t>& v) { while (v.size() % 4) v.push_back(0); }

struct Table { const char* tag; std::vector<uint8_t> data; };

std::vector<uint8_t> build_test_font() {
    std::vector<Table> t;

    {   // head — 54 bytes
        std::vector<uint8_t> d;
        u32v(d, 0x00010000);            // version
        u32v(d, 0x00010000);            // fontRevision
        u32v(d, 0);                     // checkSumAdjustment
        u32v(d, 0x5F0F3CF5);            // magicNumber
        u16(d, 0);                      // flags
        u16(d, 1000);                   // unitsPerEm
        for (int i = 0; i < 16; ++i) u8(d, 0);   // created + modified
        i16(d, 100); i16(d, 0); i16(d, 700); i16(d, 700);  // xMin yMin xMax yMax
        u16(d, 0);                      // macStyle
        u16(d, 8);                      // lowestRecPPEM
        i16(d, 2);                      // fontDirectionHint
        i16(d, 0);                      // indexToLocFormat = short
        i16(d, 0);                      // glyphDataFormat
        t.push_back({"head", d});
    }
    {   // maxp v1.0 — 32 bytes
        std::vector<uint8_t> d;
        u32v(d, 0x00010000);
        u16(d, 2);                      // numGlyphs
        for (int i = 0; i < 13; ++i) u16(d, 0);
        t.push_back({"maxp", d});
    }
    {   // hhea — 36 bytes
        std::vector<uint8_t> d;
        u32v(d, 0x00010000);
        i16(d, 800); i16(d, -200); i16(d, 0);    // ascender descender lineGap
        u16(d, 800);                             // advanceWidthMax
        i16(d, 100); i16(d, 100); i16(d, 700);   // min lsb, min rsb, xMaxExtent
        i16(d, 1); i16(d, 0); i16(d, 0);         // caret slope rise/run, caret offset
        for (int i = 0; i < 4; ++i) i16(d, 0);   // reserved
        i16(d, 0);                               // metricDataFormat
        u16(d, 2);                               // numberOfHMetrics
        t.push_back({"hhea", d});
    }
    {   // hmtx — 2 long metrics
        std::vector<uint8_t> d;
        u16(d, 800); i16(d, 0);
        u16(d, 800); i16(d, 100);
        t.push_back({"hmtx", d});
    }
    {   // glyf — glyph 0 empty, glyph 1 a 4-point square contour (34 bytes)
        std::vector<uint8_t> d;
        i16(d, 1);                                        // numberOfContours
        i16(d, 100); i16(d, 0); i16(d, 700); i16(d, 700); // bbox
        u16(d, 3);                                        // endPtsOfContours[0]
        u16(d, 0);                                        // instructionLength
        u8(d, 0x01); u8(d, 0x01); u8(d, 0x01); u8(d, 0x01);  // all on-curve, 16-bit deltas
        i16(d, 100); i16(d, 600); i16(d, 0); i16(d, -600);   // x deltas
        i16(d, 0); i16(d, 0); i16(d, 700); i16(d, 0);        // y deltas
        t.push_back({"glyf", d});
    }
    {   // loca (short) — glyph 0 empty at 0, glyph 1 spans [0, 34)
        std::vector<uint8_t> d;
        u16(d, 0); u16(d, 0); u16(d, 34 / 2);
        t.push_back({"loca", d});
    }
    {   // cmap — one (3,1) subtable, format 6, mapping 'A' to glyph 1
        std::vector<uint8_t> d;
        u16(d, 0);                       // version
        u16(d, 1);                       // numTables
        u16(d, 3); u16(d, 1);            // platform 3, encoding 1
        u32v(d, 12);                     // offset to the subtable
        u16(d, 6);                       // format 6
        u16(d, 12);                      // length: 10 header bytes + entryCount*2
        u16(d, 0);                       // language
        u16(d, 'A');                     // firstCode
        u16(d, 1);                       // entryCount
        u16(d, 1);                       // glyphIdArray[0] = glyph 1
        t.push_back({"cmap", d});
    }

    std::sort(t.begin(), t.end(), [](const Table& a, const Table& b) {
        return std::strncmp(a.tag, b.tag, 4) < 0;
    });

    const uint32_t n = (uint32_t)t.size();
    uint32_t sr = 16, es = 0;
    while (sr * 2 <= n * 16) { sr *= 2; ++es; }

    std::vector<uint8_t> out;
    u32v(out, 0x00010000);
    u16(out, n); u16(out, sr); u16(out, es); u16(out, n * 16 - sr);
    const size_t records = out.size();
    for (uint32_t i = 0; i < n; ++i) for (int k = 0; k < 16; ++k) out.push_back(0);
    for (uint32_t i = 0; i < n; ++i) {
        pad4(out);
        const uint32_t off = (uint32_t)out.size();
        std::memcpy(&out[records + i * 16], t[i].tag, 4);
        out[records + i * 16 + 8]  = (uint8_t)(off >> 24);
        out[records + i * 16 + 9]  = (uint8_t)(off >> 16);
        out[records + i * 16 + 10] = (uint8_t)(off >> 8);
        out[records + i * 16 + 11] = (uint8_t)off;
        const uint32_t len = (uint32_t)t[i].data.size();
        out[records + i * 16 + 12] = (uint8_t)(len >> 24);
        out[records + i * 16 + 13] = (uint8_t)(len >> 16);
        out[records + i * 16 + 14] = (uint8_t)(len >> 8);
        out[records + i * 16 + 15] = (uint8_t)len;
        out.insert(out.end(), t[i].data.begin(), t[i].data.end());
    }
    return out;
}

// The exact byte the guest heap was poisoned with in #2951, so a buffer this test leaves untouched
// reads back the way the real one did.
constexpr uint8_t kPoison = 0xaf;

}  // namespace

int main() {
    register_builtin_hle();
    auto create_lib   = Hle::lookup("n590hj5Oe-k");   // sceFontCreateLibraryWithEdition
    auto open_memory  = Hle::lookup("KXUpebrFk1U");   // sceFontOpenFontMemory
    auto set_scale    = Hle::lookup("N1EBMeGhf7E");   // sceFontSetScalePixel
    auto get_metrics  = Hle::lookup("L97d+3OgMlE");   // sceFontGetCharGlyphMetrics
    auto get_horiz    = Hle::lookup("imxVx8lm+KM");   // sceFontGetHorizontalLayout
    auto surface_init = Hle::lookup("gdUCnU0gHdI");   // sceFontRenderSurfaceInit
    auto set_scissor  = Hle::lookup("vRxf4d0ulPs");   // sceFontRenderSurfaceSetScissor
    auto render       = Hle::lookup("3G4zhgKuxE8");   // sceFontRenderCharGlyphImage
    auto sel_lib_ft   = Hle::lookup("oM+XCzVG3oM");   // sceFontSelectLibraryFt
    auto sel_rend_ft  = Hle::lookup("Xx974EW-QFY");   // sceFontSelectRendererFt
    auto destroy_lib  = Hle::lookup("FXP359ygujs");   // sceFontDestroyLibrary
    auto memory_term  = Hle::lookup("h6hIgxXEiEc");   // sceFontMemoryTerm

    CHECK(create_lib && open_memory && set_scale && get_metrics && get_horiz && surface_init &&
          set_scissor && render && sel_lib_ft && sel_rend_ft && destroy_lib && memory_term,
          "the six NIDs Metaphor left unregistered are registered (#2951)");
    if (!render || !open_memory || !get_metrics || !surface_init || !set_scissor) {
        std::printf("== FAIL: core surface missing ==\n");
        return 1;
    }

    auto P = [](const void* p) { return (uint64_t)(uintptr_t)p; };

    // Call sceFontRenderCharGlyphImage through the SAME shape the platform registers.
    //
    // This is not portability boilerplate -- it is the finding itself, and it cost a red Windows
    // CI run to learn. The Windows registration takes only the five INTEGER arguments, because the
    // import trampoline remaps integer registers and never touches xmm (#2955). A test that calls
    // the SysV seven-argument shape against it hands the handler `metrics` and `result` out of r9
    // and the shadow stack -- garbage pointers it then writes through. That is a **SegFault**, and
    // it is precisely the arbitrary write the Windows arm of hle_font.cpp exists to avoid; the
    // first version of this test demonstrated it on CI rather than in the emulator.
    //
    // The pen is ignored on Windows, and every assertion below still holds there, because the pen
    // Metaphor passes (`-h_bearing_x`, `h_bearing_y - baseline`) puts the glyph at the surface
    // origin -- which is exactly where the no-pen path draws it.
    auto call_render = [](HleFn fn, void* f, uint32_t code, void* surf, float x, float y,
                          void* metrics, void* result) -> int32_t {
#if defined(_WIN32)
        (void)x; (void)y;
        return ((int32_t (*)(void*, uint32_t, void*, void*, void*))fn)(f, code, surf, metrics,
                                                                       result);
#else
        return ((int32_t (*)(void*, uint32_t, void*, float, float, void*, void*))fn)(
            f, code, surf, x, y, metrics, result);
#endif
    };

    const std::vector<uint8_t> ttf = build_test_font();
    CHECK(ttf.size() > 200 && ttf[0] == 0 && ttf[1] == 1 && ttf[2] == 0 && ttf[3] == 0,
          "synthesized font carries the sfnt 1.0 signature");

    uint8_t mem[64]{};
    void* library = nullptr;
    CHECK(create_lib(P(mem), 0, 0, P(&library), 0, 0) == 0 && library,
          "CreateLibraryWithEdition writes a library handle");

    // --- a blob that is not a font must be REFUSED, not silently accepted -------------------
    std::vector<uint8_t> junk(512, 0x5a);
    void* junk_face = (void*)(uintptr_t)0xdeadbeefull;
    const uint64_t junk_rc = open_memory(P(library), P(junk.data()), junk.size(), 0, P(&junk_face), 0);
    CHECK((uint32_t)junk_rc != 0 && junk_face == nullptr,
          "OpenFontMemory refuses a non-font blob and clears the out handle");

    // --- open the real thing ----------------------------------------------------------------
    void* face = nullptr;
    CHECK(open_memory(P(library), P(ttf.data()), ttf.size(), 0, P(&face), 0) == 0 && face,
          "OpenFontMemory parses a real TrueType font");
    if (!face) { std::printf("== FAIL: no face ==\n"); return 1; }

    // sceFontSetScalePixel takes its two floats in xmm0/xmm1; call it through the real signature.
    auto set_scale_f = (int32_t(*)(void*, float, float))set_scale;
    set_scale_f(face, 64.0f, 64.0f);

    // A sentinel, deliberately NOT `kPoison`: writing that byte as a float VALUE gives 175.0, not
    // the 0xafafafaf bit pattern the memset arms rely on. Two different sentinels spelled the same
    // way, next to arms whose entire point is that the bit pattern reads as a plausible float, is
    // the confusion this file's header warns about. This one only has to differ from 51.2.
    constexpr float kUnset = -1.0f;
    float layout[3] = {kUnset, kUnset, kUnset};
    CHECK(get_horiz(P(face), P(layout), 0, 0, 0, 0) == 0,
          "GetHorizontalLayout succeeds on a memory face");
    // ascent 800 / unitsPerEm-derived scale 0.064 -> 51.2 px.
    CHECK(std::fabs(layout[0] - 51.2f) < 0.5f,
          "baseline comes from the font's own ascent, not a placeholder");

    float m[8];
    std::memset(m, kPoison, sizeof(m));
    CHECK(get_metrics(P(face), 'A', P(m), 0, 0, 0) == 0, "GetCharGlyphMetrics succeeds");
    // 600 and 700 font units at 0.064 -> 38.4 x 44.8 px, so 38-39 x 44-45 after grid fitting.
    CHECK(m[0] >= 38.0f && m[0] <= 40.0f, "glyph width comes from the font outline");
    CHECK(m[1] >= 44.0f && m[1] <= 46.0f, "glyph height comes from the font outline");
    CHECK(std::fabs(m[4] - 51.2f) < 0.5f, "advance comes from the font's hmtx (800 units)");
    const float bearing_x = m[2], bearing_y = m[3];

    // --- render, the way Metaphor renders ---------------------------------------------------
    // It places the line box so the glyph lands at the surface origin: x = -bearingX,
    // y = bearingY - baseline (eboot+0x10015de..0x10015f8).
    constexpr int kPitch = 128, kDim = 96;
    std::vector<uint8_t> surf_bytes((size_t)kPitch * kDim, 0);
    uint8_t surface[128];
    std::memset(surface, kPoison, sizeof(surface));
    auto surface_init_f = (void(*)(void*, void*, int, int, int, int))surface_init;
    surface_init_f(surface, surf_bytes.data(), kPitch, 1, kDim, kDim);

    uint8_t result[0x40];
    std::memset(result, kPoison, sizeof(result));
    const int32_t rrc = call_render(render, face, 'A', surface, -bearing_x,
                                    bearing_y - layout[0], m, result);
    CHECK(rrc == 0, "RenderCharGlyphImage reports success on a glyph it can draw");

    // THE #2951 REGRESSION. Both fields are read unconditionally by the caller, so both must have
    // been written — and the poison check is what separates "written" from "happened to look ok".
    auto rd_i32 = [&](int off) { int32_t v; std::memcpy(&v, result + off, 4); return v; };
    auto rd_f32 = [&](int off) { float v; std::memcpy(&v, result + off, 4); return v; };
    const int32_t rw = rd_i32(0x38), rh = rd_i32(0x3c);
    CHECK((uint32_t)rw != 0xafafafafu && (uint32_t)rh != 0xafafafafu,
          "the render result's width/height are WRITTEN, not left as caller residue (#2951)");
    CHECK(rw >= 38 && rw <= 40 && rh >= 44 && rh <= 46,
          "the reported drawn size matches the glyph actually rasterized");
    CHECK(rd_i32(0x10) == kPitch, "the result reports the surface pitch the blit used");
    // +0x30 is READ by the caller (eboot+0x1001432 truncates it and stores it into the engine's
    // own glyph record), so it is not "reserved" and it must not be left as residue. What it means
    // is underived, so the assertion is deliberately only that it was written -- see the
    // CONFIDENCE: LOW note on RenderResult. Review found this field; the first version of this
    // test asserted nothing here and would not have caught a regression.
    CHECK((uint32_t)rd_i32(0x30) != 0xafafafafu,
          "the field at +0x30 the caller also reads is written, not left as residue");
    // The poison compare is not redundant with the magnitude compare, and mutation testing is how
    // that was found: 0xafafafaf reinterpreted as a float is -3.2e-13, which sails through any
    // "close to zero" test. An untouched buffer therefore PASSES a placement assertion written the
    // obvious way -- the exact shape of assertion-true-mechanism-never-ran. Check the bits too.
    CHECK((uint32_t)rd_i32(0x28) != 0xafafafafu && (uint32_t)rd_i32(0x2c) != 0xafafafafu &&
              std::fabs(rd_f32(0x28)) < 0.6f && std::fabs(rd_f32(0x2c)) < 0.6f,
          "Metaphor's pen arithmetic lands the glyph at the surface origin");

    // ...and it must have actually drawn something, in the right place.
    size_t covered = 0, first_row = SIZE_MAX, first_col = SIZE_MAX;
    for (int y = 0; y < kDim; ++y)
        for (int x = 0; x < kPitch; ++x)
            if (surf_bytes[(size_t)y * kPitch + x]) {
                ++covered;
                if (first_row == SIZE_MAX) first_row = (size_t)y;
                first_col = std::min(first_col, (size_t)x);
            }
    CHECK(covered > 500, "the surface actually received glyph coverage");
    CHECK(first_row <= 1 && first_col <= 1, "coverage starts at the surface origin");
    CHECK(covered <= (size_t)(rw + 2) * (size_t)(rh + 2),
          "coverage stays inside the size the call reported");

    // --- the scissor must be honoured, and the result must say so ----------------------------
    std::fill(surf_bytes.begin(), surf_bytes.end(), 0);
    CHECK(set_scissor(P(surface), 80, 80, 8, 8, 0) == 0, "SetScissor succeeds");
    uint8_t result2[0x40];
    std::memset(result2, kPoison, sizeof(result2));
    call_render(render, face, 'A', surface, -bearing_x, bearing_y - layout[0], m, result2);
    size_t covered2 = 0;
    for (uint8_t b : surf_bytes) if (b) ++covered2;
    int32_t rw2; std::memcpy(&rw2, result2 + 0x38, 4);
    CHECK(covered2 == 0, "a scissor away from the glyph clips every pixel");
    CHECK(rw2 == 0, "a fully clipped glyph reports zero drawn width, not poison");

    // --- an EMPTY scissor clips everything; it is not read as "no scissor" -------------------
    // {0,0,0,0} is empty under both readings of the trailing pair (extents or a second corner), so
    // this arm does not depend on resolving that ambiguity. The failure it guards is the
    // asymmetric one: treating empty as absent would draw over the whole of the guest's buffer.
    std::fill(surf_bytes.begin(), surf_bytes.end(), 0);
    CHECK(set_scissor(P(surface), 0, 0, 0, 0, 0) == 0, "SetScissor accepts an empty rectangle");
    uint8_t result_empty[0x40];
    std::memset(result_empty, kPoison, sizeof(result_empty));
    call_render(render, face, 'A', surface, -bearing_x, bearing_y - layout[0], m, result_empty);
    size_t covered_empty = 0;
    for (uint8_t b : surf_bytes) if (b) ++covered_empty;
    int32_t rw_empty; std::memcpy(&rw_empty, result_empty + 0x38, 4);
    CHECK(covered_empty == 0, "an empty scissor draws nothing at all");
    CHECK(rw_empty == 0, "an empty scissor reports zero drawn width, not poison");

    // ...and restoring the scissor to the whole surface draws again, which is what proves the two
    // arms above measured the scissor rather than something that had simply stopped working.
    std::fill(surf_bytes.begin(), surf_bytes.end(), 0);
    set_scissor(P(surface), 0, 0, kDim, kDim, 0);
    uint8_t result_back[0x40];
    std::memset(result_back, kPoison, sizeof(result_back));
    call_render(render, face, 'A', surface, -bearing_x, bearing_y - layout[0], m, result_back);
    size_t covered_back = 0;
    for (uint8_t b : surf_bytes) if (b) ++covered_back;
    CHECK(covered_back > 500, "restoring a full scissor draws the glyph again");

    // --- a null surface must not be answered with success over an unwritten result -----------
    uint8_t result3[0x40];
    std::memset(result3, kPoison, sizeof(result3));
    const int32_t nrc = call_render(render, face, 'A', nullptr, 0.0f, 0.0f, nullptr, result3);
    int32_t rw3; std::memcpy(&rw3, result3 + 0x38, 4);
    CHECK(nrc != 0, "a null surface is an error, not a success");
    CHECK((uint32_t)rw3 != 0xafafafafu,
          "even the error path clears the result the caller will read anyway");

    // --- the two Ft selectors are distinct editions ------------------------------------------
    const uint64_t lib_ft = sel_lib_ft(0, 0, 0, 0, 0, 0);
    const uint64_t rend_ft = sel_rend_ft(0, 0, 0, 0, 0, 0);
    CHECK(lib_ft && rend_ft && lib_ft != rend_ft,
          "SelectLibraryFt and SelectRendererFt return distinct non-null editions");

    // --- lifecycle ---------------------------------------------------------------------------
    CHECK(memory_term(P(mem), 0, 0, 0, 0, 0) == 0, "MemoryTerm releases the memory descriptor");
    void* lib2 = library;
    CHECK(destroy_lib(P(&lib2), 0, 0, 0, 0, 0) == 0 && !lib2,
          "DestroyLibrary clears the library handle");

    std::printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
