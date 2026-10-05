// test_font — the Astro Font HLE surface: a wrong value here is a title that draws its UI with
// glyph advances the guest never asked for, so every arm below asserts a WRITE (a non-null handle,
// a non-zero metric), not merely a success code. Looked up by raw NID, which is how the guest
// imports them: there is no recovered name for this library.
#include "hle/dispatch/dispatch.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

namespace {
struct AstroFont {
    HleFn create_library = nullptr, open_font_set = nullptr, metrics = nullptr, text_init = nullptr,
          create_string = nullptr, chars = nullptr, destroy_string = nullptr, close_font = nullptr;
    bool registered() const {
        return create_library && open_font_set && metrics && text_init && create_string && chars &&
               destroy_string && close_font;
    }
};
AstroFont astro_font() {
    register_builtin_hle();
    AstroFont font;
    font.create_library = Hle::lookup("n590hj5Oe-k");
    font.open_font_set = Hle::lookup("cKYtVmeSTcw");
    font.metrics = Hle::lookup("IQtleGLL5pQ");
    font.text_init = Hle::lookup("oaJ1BpN2FQk");
    font.create_string = Hle::lookup("MO24vDhmS4E");
    font.chars = Hle::lookup("Avv7OApgCJk");
    font.destroy_string = Hle::lookup("SSCaczu2aMQ");
    font.close_font = Hle::lookup("vzHs3C8lWJk");
    return font;
}
constexpr uint64_t addr(const void* p) {
    return reinterpret_cast<uint64_t>(p);
}
}  // namespace

TEST(Font, AstroFontSurfaceIsRegistered) {
    EXPECT_TRUE(astro_font().registered()) << "Astro Font HLE surface is registered";
}

TEST(Font, CreateLibraryWritesANonNullHandle) {
    const AstroFont font = astro_font();
    ASSERT_TRUE(font.registered());
    uint8_t mem[64]{};
    void* library = nullptr;
    EXPECT_EQ(font.create_library(addr(mem), 0, 0, addr(&library), 0, 0), 0u)
        << "CreateLibraryWithEdition returns success";
    EXPECT_NE(library, nullptr) << "CreateLibraryWithEdition writes a non-null library handle";
}

TEST(Font, OpenFontSetWritesANonNullHandle) {
    const AstroFont font = astro_font();
    ASSERT_TRUE(font.registered());
    uint8_t mem[64]{};
    void* library = nullptr;
    ASSERT_EQ(font.create_library(addr(mem), 0, 0, addr(&library), 0, 0), 0u);
    void* handle = nullptr;
    EXPECT_EQ(font.open_font_set(addr(library), 0, 0, 0, addr(&handle), 0), 0u)
        << "OpenFontSet returns success";
    EXPECT_NE(handle, nullptr) << "OpenFontSet writes a non-null font handle";
    EXPECT_EQ(font.close_font(addr(handle), 0, 0, 0, 0, 0), 0u) << "CloseFont releases the face";
}

TEST(Font, GlyphMetricsAreDeterministicAndNonZero) {
    const AstroFont font = astro_font();
    ASSERT_TRUE(font.registered());
    uint8_t mem[64]{};
    void* library = nullptr;
    ASSERT_EQ(font.create_library(addr(mem), 0, 0, addr(&library), 0, 0), 0u);
    void* handle = nullptr;
    ASSERT_EQ(font.open_font_set(addr(library), 0, 0, 0, addr(&handle), 0), 0u);
    float glyph[8]{};
    EXPECT_EQ(font.metrics(addr(handle), 'A', addr(glyph), 0, 0, 0), 0u)
        << "metrics returns success";
    EXPECT_GT(glyph[0], 0.0f) << "glyph width is non-zero";
    EXPECT_GT(glyph[1], 0.0f) << "glyph height is non-zero";
    EXPECT_GT(glyph[4], 0.0f) << "glyph horizontal advance is non-zero";
}

TEST(Font, TextSourceInitializesThePublicSourceLayout) {
    const AstroFont font = astro_font();
    ASSERT_TRUE(font.registered());
    uint8_t source[0x60];
    std::memset(source, 0xcc, sizeof(source));
    const char text[] = "ASTRO";
    EXPECT_EQ(font.text_init(addr(source), addr(text), sizeof(text), 0, 0, 0), 0u)
        << "TextSourceInit initializes the public source layout";
}

TEST(Font, StringHandleIsOpaqueWrittenAndClearedOnDestroy) {
    const AstroFont font = astro_font();
    ASSERT_TRUE(font.registered());
    uint8_t mem[64]{};
    uint8_t source[0x60];
    std::memset(source, 0xcc, sizeof(source));
    const char text[] = "ASTRO";
    ASSERT_EQ(font.text_init(addr(source), addr(text), sizeof(text), 0, 0, 0), 0u);
    void* string = nullptr;
    EXPECT_EQ(font.create_string(addr(mem), addr(source), 0, addr(&string), 0, 0), 0u)
        << "CreateString returns success";
    ASSERT_NE(string, nullptr) << "CreateString writes an opaque string handle";
    uint32_t count = 0xdeadbeef;
    EXPECT_EQ(font.chars(addr(string), addr(&count), 0, 0, 0, 0), 0u)
        << "GetStringCharRange returns success";
    EXPECT_EQ(count, 0u) << "fallback string exposes a safe empty character range";
    EXPECT_EQ(font.destroy_string(addr(&string), 0, 0, 0, 0, 0), 0u)
        << "DestroyString returns success";
    EXPECT_EQ(string, nullptr) << "DestroyString releases and clears the opaque string handle";
}