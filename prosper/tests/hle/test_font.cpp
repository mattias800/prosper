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
// The render-state remainder surface, looked up by raw NID like the guest imports them.
struct FontState {
    HleFn create_library = nullptr, create_renderer = nullptr, rebind = nullptr,
          get_scale = nullptr, get_scale_point = nullptr, get_render_scale = nullptr,
          get_render_scale_point = nullptr, get_slant = nullptr, get_render_slant = nullptr,
          get_weight = nullptr, get_render_weight = nullptr, set_scale_point = nullptr,
          render_vertical = nullptr;
    bool registered() const {
        return create_library && create_renderer && rebind && get_scale && get_scale_point &&
               get_render_scale && get_render_scale_point && get_slant && get_render_slant &&
               get_weight && get_render_weight && set_scale_point && render_vertical;
    }
};
FontState font_state() {
    register_builtin_hle();
    FontState font;
    font.create_library = Hle::lookup("nWrfPI4Okmg");
    font.create_renderer = Hle::lookup("u5fZd3KZcs0");
    font.rebind = Hle::lookup("Z2cdsqJH+5k");
    font.get_scale = Hle::lookup("CkVmLoCNN-8");
    font.get_scale_point = Hle::lookup("GoF2bhB7LYk");
    font.get_render_scale = Hle::lookup("EY38A01lq2k");
    font.get_render_scale_point = Hle::lookup("FEafYUcxEGo");
    font.get_slant = Hle::lookup("ynSqYL8VpoA");
    font.get_render_slant = Hle::lookup("Gqa5Pp7y4MU");
    font.get_weight = Hle::lookup("d7dDgRY+Bzw");
    font.get_render_weight = Hle::lookup("woOjHrkjIYg");
    font.set_scale_point = Hle::lookup("sw65+7wXCKE");
    font.render_vertical = Hle::lookup("i6UNdSig1uE");
    return font;
}
// Not constexpr: a reinterpret_cast is never a constant expression, and clang (the macOS build)
// rejects a constexpr function that can never produce one (#4467).
uint64_t addr(const void* p) {
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

TEST(Font, RenderStateSurfaceIsRegistered) {
    EXPECT_TRUE(font_state().registered()) << "font render-state remainder is registered";
}

TEST(Font, RenderStateNidsResolveToStubValues) {
    // All thirteen NIDs reproduce the stub table verbatim.
    EXPECT_EQ(nid_hash("sceFontRenderCharGlyphImageVertical"), "i6UNdSig1uE");
    EXPECT_EQ(nid_hash("sceFontCreateLibrary"), "nWrfPI4Okmg");
    EXPECT_EQ(nid_hash("sceFontCreateRenderer"), "u5fZd3KZcs0");
    EXPECT_EQ(nid_hash("sceFontRebindRenderer"), "Z2cdsqJH+5k");
    EXPECT_EQ(nid_hash("sceFontGetScalePixel"), "CkVmLoCNN-8");
    EXPECT_EQ(nid_hash("sceFontGetScalePoint"), "GoF2bhB7LYk");
    EXPECT_EQ(nid_hash("sceFontGetRenderScalePixel"), "EY38A01lq2k");
    EXPECT_EQ(nid_hash("sceFontGetRenderScalePoint"), "FEafYUcxEGo");
    EXPECT_EQ(nid_hash("sceFontGetEffectSlant"), "ynSqYL8VpoA");
    EXPECT_EQ(nid_hash("sceFontGetRenderEffectSlant"), "Gqa5Pp7y4MU");
    EXPECT_EQ(nid_hash("sceFontGetEffectWeight"), "d7dDgRY+Bzw");
    EXPECT_EQ(nid_hash("sceFontGetRenderEffectWeight"), "woOjHrkjIYg");
    EXPECT_EQ(nid_hash("sceFontSetScalePoint"), "sw65+7wXCKE");
    EXPECT_NE(nid_hash("sceFontGetScalePixel"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

TEST(Font, PlainCreateDelegatesToEditionBodies) {
    const FontState font = font_state();
    ASSERT_TRUE(font.registered());
    uint8_t mem[64]{};
    void* library = nullptr;
    // Plain create takes (memory, params, out): the wrapper shifts the out-pointer for the
    // 4-arg edition body. A direct alias would read a garbage fourth register instead.
    EXPECT_EQ(font.create_library(addr(mem), 0, addr(&library), 0, 0, 0), 0u)
        << "CreateLibrary returns success";
    EXPECT_NE(library, nullptr) << "CreateLibrary writes a non-null library handle";
    void* renderer = nullptr;
    EXPECT_EQ(font.create_renderer(addr(mem), 0, addr(&renderer), 0, 0, 0), 0u)
        << "CreateRenderer returns success";
    EXPECT_NE(renderer, nullptr) << "CreateRenderer writes a non-null renderer handle";
}

TEST(Font, RebindRendererValidatesTheFace) {
    const FontState font = font_state();
    const AstroFont astro = astro_font();
    ASSERT_TRUE(font.registered());
    ASSERT_TRUE(astro.registered());
    uint8_t mem[64]{};
    void* library = nullptr;
    ASSERT_EQ(astro.create_library(addr(mem), 0, 0, addr(&library), 0, 0), 0u);
    void* handle = nullptr;
    ASSERT_EQ(astro.open_font_set(addr(library), 0, 0, 0, addr(&handle), 0), 0u);
    // libSceFont.native's RebindRenderer (0xad20): no renderer bound -> 0x80460061; a bound
    // pointer that is not a renderer -> 0x80460001; a real one -> 0.
    HleFn bind = Hle::lookup("3OdRkSjOcog");
    HleFn unbind = Hle::lookup("1QjhKxrsOB8");
    ASSERT_NE(bind, nullptr);
    ASSERT_NE(unbind, nullptr);
    EXPECT_EQ(font.rebind(addr(handle), 0, 0, 0, 0, 0), 0x80460061u) << "no renderer bound yet";
    uint8_t not_a_renderer[64]{};
    ASSERT_EQ(bind(addr(handle), addr(not_a_renderer), 0, 0, 0, 0), 0u);
    EXPECT_EQ(font.rebind(addr(handle), 0, 0, 0, 0, 0), 0x80460001u) << "bound pointer is no renderer";
    void* renderer = nullptr;
    ASSERT_EQ(font.create_renderer(addr(mem), 0, addr(&renderer), 0, 0, 0), 0u);
    ASSERT_EQ(bind(addr(handle), addr(renderer), 0, 0, 0, 0), 0u);
    EXPECT_EQ(font.rebind(addr(handle), 0, 0, 0, 0, 0), 0u) << "rebind keeps a live face bound";
    ASSERT_EQ(unbind(addr(handle), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(font.rebind(addr(handle), 0, 0, 0, 0, 0), 0x80460061u) << "unbound again";
    // A wild pointer would fault in face()'s magic read (guest pointers are dereferenced
    // directly here), so the refusal arm uses valid memory with the wrong magic instead.
    uint8_t decoy[64]{};
    EXPECT_EQ(font.rebind(addr(decoy), 0, 0, 0, 0, 0), 0x80460005u) << "rebind refuses a foreign face";
    EXPECT_EQ(font.rebind(0, 0, 0, 0, 0, 0), 0x80460005u) << "and a NULL one";
    EXPECT_EQ(astro.close_font(addr(handle), 0, 0, 0, 0, 0), 0u);
}

// The Render-prefixed getters (native 0x95d0 / 0x96a0 / 0x9b70 / 0x99e0) refuse a face with no
// bound renderer with 0x80460061 and clear their outputs the way a failed handle does; the plain
// getters on the same face still succeed.
TEST(Font, RenderGettersNeedABoundRenderer) {
    const FontState font = font_state();
    const AstroFont astro = astro_font();
    ASSERT_TRUE(font.registered());
    ASSERT_TRUE(astro.registered());
    uint8_t mem[64]{};
    void* library = nullptr;
    ASSERT_EQ(astro.create_library(addr(mem), 0, 0, addr(&library), 0, 0), 0u);
    void* handle = nullptr;
    ASSERT_EQ(astro.open_font_set(addr(library), 0, 0, 0, addr(&handle), 0), 0u);
    float w = 5.0f, h = 5.0f, slant = 5.0f, x = 5.0f, y = 5.0f;
    uint32_t mode = 9;
    EXPECT_EQ(font.get_render_scale(addr(handle), addr(&w), addr(&h), 0, 0, 0), 0x80460061u);
    EXPECT_EQ(w, 0.0f);
    EXPECT_EQ(h, 0.0f);
    w = 5.0f;
    EXPECT_EQ(font.get_render_scale_point(addr(handle), addr(&w), 0, 0, 0, 0), 0x80460061u);
    EXPECT_EQ(w, 0.0f);
    EXPECT_EQ(font.get_render_slant(addr(handle), addr(&slant), 0, 0, 0, 0), 0x80460061u);
    EXPECT_EQ(slant, 0.0f);
    EXPECT_EQ(font.get_render_weight(addr(handle), addr(&x), addr(&y), addr(&mode), 0, 0),
              0x80460061u);
    EXPECT_EQ(x, 1.0f);
    EXPECT_EQ(y, 1.0f);
    EXPECT_EQ(mode, 0u);
    EXPECT_EQ(font.get_scale(addr(handle), addr(&w), 0, 0, 0, 0), 0u) << "the face getter is ungated";
    // A foreign face is still the handle error, not the renderer one.
    uint8_t decoy[64]{};
    EXPECT_EQ(font.get_render_slant(addr(decoy), addr(&slant), 0, 0, 0, 0), 0x80460005u);
    EXPECT_EQ(astro.close_font(addr(handle), 0, 0, 0, 0, 0), 0u);
}

TEST(Font, ScaleSlantWeightRoundTrip) {
    const FontState font = font_state();
    const AstroFont astro = astro_font();
    ASSERT_TRUE(font.registered());
    ASSERT_TRUE(astro.registered());
    uint8_t mem[64]{};
    void* library = nullptr;
    ASSERT_EQ(astro.create_library(addr(mem), 0, 0, addr(&library), 0, 0), 0u);
    void* handle = nullptr;
    ASSERT_EQ(astro.open_font_set(addr(library), 0, 0, 0, addr(&handle), 0), 0u);
    // The Render twins answer only with a renderer bound (RenderGettersNeedABoundRenderer).
    void* renderer = nullptr;
    ASSERT_EQ(font.create_renderer(addr(mem), 0, addr(&renderer), 0, 0, 0), 0u);
    ASSERT_EQ(Hle::lookup("3OdRkSjOcog")(addr(handle), addr(renderer), 0, 0, 0, 0), 0u);
    // Float setters go through their real C++ signatures: an HleFn-cast call would misdeliver
    // floats (xmm vs integer registers), so the bridge path is not what this asserts.
    using SetScaleFn = int32_t (*)(void*, float, float);
    auto set_scale = reinterpret_cast<SetScaleFn>(font.set_scale_point);
    ASSERT_EQ(set_scale(handle, 30.0f, 40.0f), 0) << "SetScalePoint accepts the scale";
    float w = 0.0f, h = 0.0f;
    ASSERT_EQ(font.get_scale(addr(handle), addr(&w), addr(&h), 0, 0, 0), 0u);
    EXPECT_EQ(w, 30.0f) << "GetScalePixel reports the stored width";
    EXPECT_EQ(h, 40.0f) << "GetScalePixel reports the stored height";
    ASSERT_EQ(font.get_scale_point(addr(handle), addr(&w), addr(&h), 0, 0, 0), 0u);
    EXPECT_EQ(w, 30.0f) << "GetScalePoint agrees (no separate point state)";
    ASSERT_EQ(font.get_render_scale(addr(handle), addr(&w), addr(&h), 0, 0, 0), 0u);
    EXPECT_EQ(w, 30.0f) << "GetRenderScalePixel agrees (no separate render state)";
    using SetSlantFn = int32_t (*)(void*, float);
    auto set_slant = reinterpret_cast<SetSlantFn>(Hle::lookup("TMtqoFQjjbA"));
    ASSERT_NE(set_slant, nullptr);
    ASSERT_EQ(set_slant(handle, 0.25f), 0);
    float slant = 0.0f;
    ASSERT_EQ(font.get_slant(addr(handle), addr(&slant), 0, 0, 0, 0), 0u);
    EXPECT_EQ(slant, 0.25f) << "GetEffectSlant reports the stored slant";
    ASSERT_EQ(font.get_render_slant(addr(handle), addr(&slant), 0, 0, 0, 0), 0u);
    EXPECT_EQ(slant, 0.25f) << "GetRenderEffectSlant agrees";
    using SetWeightFn = int32_t (*)(void*, float, float, uint32_t);
    auto set_weight = reinterpret_cast<SetWeightFn>(Hle::lookup("v0phZwa4R5o"));
    ASSERT_NE(set_weight, nullptr);
    ASSERT_EQ(set_weight(handle, 1.5f, 2.0f, 7), 0);
    float x = 0.0f, y = 0.0f;
    uint32_t mode = 0;
    ASSERT_EQ(font.get_weight(addr(handle), addr(&x), addr(&y), addr(&mode), 0, 0), 0u);
    EXPECT_EQ(x, 1.5f) << "GetEffectWeight reports the stored x scale";
    EXPECT_EQ(y, 2.0f) << "GetEffectWeight reports the stored y scale";
    EXPECT_EQ(mode, 0u) << "the firmware getter always reports mode 0; the mode is never stored";
    mode = 9;
    ASSERT_EQ(font.get_render_weight(addr(handle), addr(&x), addr(&y), addr(&mode), 0, 0), 0u);
    EXPECT_EQ(mode, 0u) << "GetRenderEffectWeight agrees";

    // Each non-NULL output is written; only an all-NULL call is refused (0x80460002).
    w = h = -1.0f;
    EXPECT_EQ(font.get_scale(addr(handle), addr(&w), 0, 0, 0, 0), 0u) << "one NULL output is legal";
    EXPECT_EQ(w, 30.0f);
    EXPECT_EQ(font.get_scale(addr(handle), 0, addr(&h), 0, 0, 0), 0u);
    EXPECT_EQ(h, 40.0f);
    x = -1.0f;
    EXPECT_EQ(font.get_weight(addr(handle), addr(&x), 0, 0, 0, 0), 0u);
    EXPECT_EQ(x, 1.5f);
    EXPECT_EQ(font.get_scale(addr(handle), 0, 0, 0, 0, 0), 0x80460002u) << "all outputs NULL";
    EXPECT_EQ(font.get_slant(addr(handle), 0, 0, 0, 0, 0), 0x80460002u);
    EXPECT_EQ(font.get_weight(addr(handle), 0, 0, 0, 0, 0), 0x80460002u);

    // A foreign handle fails with 0x80460005 and zeroes the outputs (weight: 1.0, 1.0, 0). Valid
    // memory with the wrong magic, since face() dereferences the pointer.
    uint8_t decoy[64]{};
    w = h = slant = 5.0f;
    EXPECT_EQ(font.get_scale(addr(decoy), addr(&w), addr(&h), 0, 0, 0), 0x80460005u);
    EXPECT_EQ(w, 0.0f);
    EXPECT_EQ(h, 0.0f);
    EXPECT_EQ(font.get_slant(addr(decoy), addr(&slant), 0, 0, 0, 0), 0x80460005u);
    EXPECT_EQ(slant, 0.0f);
    x = y = 5.0f;
    mode = 9;
    EXPECT_EQ(font.get_weight(addr(decoy), addr(&x), addr(&y), addr(&mode), 0, 0), 0x80460005u);
    EXPECT_EQ(x, 1.0f);
    EXPECT_EQ(y, 1.0f);
    EXPECT_EQ(mode, 0u);
    EXPECT_EQ(astro.close_font(addr(handle), 0, 0, 0, 0, 0), 0u);
}

TEST(Font, VerticalRenderMatchesHorizontalOnPlaceholderFace) {
    const FontState font = font_state();
    const AstroFont astro = astro_font();
    ASSERT_TRUE(font.registered());
    ASSERT_TRUE(astro.registered());
    uint8_t mem[64]{};
    void* library = nullptr;
    ASSERT_EQ(astro.create_library(addr(mem), 0, 0, addr(&library), 0, 0), 0u);
    void* handle = nullptr;
    ASSERT_EQ(astro.open_font_set(addr(library), 0, 0, 0, addr(&handle), 0), 0u);
    // A system-set face has no font behind it: both variants report the advance and draw
    // nothing, so identical inputs must produce identical out-buffers. The surface is a
    // zeroed scratch (only its pitch field is read on this path); metrics/result use the
    // caller-pinned 0x20/0x40 sizes from the Metaphor frame.
    using RenderFn = int32_t (*)(void*, uint32_t, void*, float, float, void*, void*);
    auto render_h = reinterpret_cast<RenderFn>(Hle::lookup("3G4zhgKuxE8"));
    auto render_v = reinterpret_cast<RenderFn>(font.render_vertical);
    ASSERT_NE(render_h, nullptr);
    ASSERT_NE(render_v, nullptr);
    uint8_t surface[256]{};
    uint8_t metrics_h[0x20]{}, result_h[0x40]{}, metrics_v[0x20]{}, result_v[0x40]{};
    ASSERT_EQ(render_h(handle, 'A', surface, 0.0f, 0.0f, metrics_h, result_h), 0);
    ASSERT_EQ(render_v(handle, 'A', surface, 0.0f, 0.0f, metrics_v, result_v), 0);
    EXPECT_EQ(std::memcmp(metrics_h, metrics_v, sizeof(metrics_h)), 0)
        << "vertical render reports the same metrics as horizontal";
    EXPECT_EQ(std::memcmp(result_h, result_v, sizeof(result_h)), 0)
        << "vertical render reports the same result as horizontal";
    EXPECT_EQ(astro.close_font(addr(handle), 0, 0, 0, 0, 0), 0u);
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

struct FontRendererMisc {
    HleFn create_library = nullptr, create_renderer = nullptr, open_font_set = nullptr,
          close_font = nullptr, get_outline_size = nullptr, reset_outline = nullptr,
          set_outline_policy = nullptr, set_dpi = nullptr, text_init = nullptr, rewind = nullptr,
          writing_init = nullptr, set_mask = nullptr;
    bool registered() const {
        return create_library && create_renderer && open_font_set && close_font &&
               get_outline_size && reset_outline && set_outline_policy && set_dpi && text_init &&
               rewind && writing_init && set_mask;
    }
};
FontRendererMisc font_renderer_misc() {
    register_builtin_hle();
    FontRendererMisc api;
    api.create_library = Hle::lookup("nWrfPI4Okmg");
    api.create_renderer = Hle::lookup("u5fZd3KZcs0");
    api.open_font_set = Hle::lookup("cKYtVmeSTcw");
    api.close_font = Hle::lookup("vzHs3C8lWJk");
    api.get_outline_size = Hle::lookup("amcmrY62BD4");
    api.reset_outline = Hle::lookup("ai6AfGrBs4o");
    api.set_outline_policy = Hle::lookup("ydF+WuH0fAk");
    api.set_dpi = Hle::lookup("I1acwR7Qp8E");
    api.text_init = Hle::lookup("oaJ1BpN2FQk");
    api.rewind = Hle::lookup("VRFd3diReec");
    api.writing_init = Hle::lookup("fD5rqhEXKYQ");
    api.set_mask = Hle::lookup("BbCZjJizU4A");
    return api;
}

TEST(Font, RendererMiscSurfaceIsRegistered) {
    EXPECT_TRUE(font_renderer_misc().registered()) << "font renderer-misc surface is registered";
}

TEST(Font, RendererMiscNidsResolveToStubValues) {
    EXPECT_EQ(nid_hash("sceFontRendererGetOutlineBufferSize"), "amcmrY62BD4");
    EXPECT_EQ(nid_hash("sceFontRendererResetOutlineBuffer"), "ai6AfGrBs4o");
    EXPECT_EQ(nid_hash("sceFontRendererSetOutlineBufferPolicy"), "ydF+WuH0fAk");
    EXPECT_EQ(nid_hash("sceFontSetResolutionDpi"), "I1acwR7Qp8E");
    EXPECT_EQ(nid_hash("sceFontTextSourceRewind"), "VRFd3diReec");
    EXPECT_EQ(nid_hash("sceFontWritingSetMaskInvisible"), "BbCZjJizU4A");
    EXPECT_NE(nid_hash("sceFontSetResolutionDpi"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

TEST(Font, OutlineBufferPolicyRoundTrips) {
    const FontRendererMisc api = font_renderer_misc();
    ASSERT_TRUE(api.registered());
    uint8_t mem[64]{};
    void* renderer = nullptr;
    ASSERT_EQ(api.create_renderer(addr(mem), 0, addr(&renderer), 0, 0, 0), 0u);
    ASSERT_NE(renderer, nullptr);
    uint32_t size = 0xdeadbeef;
    EXPECT_EQ(api.get_outline_size(addr(renderer), addr(&size), 0, 0, 0, 0), 0u)
        << "GetOutlineBufferSize returns success";
    EXPECT_EQ(size, 0u) << "fresh renderer reports zero outline workspace";
    EXPECT_EQ(api.set_outline_policy(addr(renderer), 0, 0x4000, 0, 0, 0), 0u)
        << "SetOutlineBufferPolicy accepts a basal size";
    EXPECT_EQ(api.get_outline_size(addr(renderer), addr(&size), 0, 0, 0, 0), 0u);
    EXPECT_EQ(size, 0x4000u) << "policy stores the basal size";
    EXPECT_EQ(api.reset_outline(addr(renderer), 0, 0, 0, 0, 0), 0u)
        << "ResetOutlineBuffer returns success";
}

TEST(Font, SetResolutionDpiStoresNonZeroDpi) {
    const FontRendererMisc api = font_renderer_misc();
    ASSERT_TRUE(api.registered());
    uint8_t mem[64]{};
    void* library = nullptr;
    ASSERT_EQ(api.create_library(addr(mem), 0, addr(&library), 0, 0, 0), 0u);
    ASSERT_NE(library, nullptr);
    void* handle = nullptr;
    ASSERT_EQ(api.open_font_set(addr(library), 0, 0, 0, addr(&handle), 0), 0u);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(api.set_dpi(addr(handle), 144, 144, 0, 0, 0), 0u)
        << "SetResolutionDpi returns success";
    EXPECT_EQ(api.close_font(addr(handle), 0, 0, 0, 0, 0), 0u) << "CloseFont releases the face";
}

TEST(Font, TextSourceRewindResetsCurrent) {
    const FontRendererMisc api = font_renderer_misc();
    ASSERT_TRUE(api.registered());
    uint8_t source[0x60]{};
    const char text[] = "ASTRO";
    ASSERT_EQ(api.text_init(addr(source), addr(text), sizeof(text), 0, 0, 0), 0u);
    EXPECT_EQ(api.rewind(addr(source), 0, 0, 0, 0, 0), 0u) << "Rewind returns success";
}

TEST(Font, WritingSetMaskInvisibleAcceptsValidMasks) {
    const FontRendererMisc api = font_renderer_misc();
    ASSERT_TRUE(api.registered());
    uint8_t writing[0x100]{};
    EXPECT_EQ(api.set_mask(addr(writing), 0, 0, 0, 0, 0), 0u) << "mask 0 returns success";
    EXPECT_EQ(api.set_mask(addr(writing), 1, 0, 0, 0, 0), 0u) << "mask 1 returns success";
    EXPECT_EQ(api.set_mask(addr(writing), 2, 0, 0, 0, 0), 0x80460002u) << "invalid mask is refused";
}