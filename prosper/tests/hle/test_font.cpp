// test_font — the Astro Font HLE surface: a wrong value here is a title that draws its UI with
// glyph advances the guest never asked for, so every arm below asserts a WRITE (a non-null handle,
// a non-zero metric), not merely a success code. Looked up by raw NID, which is how the guest
// imports them: there is no recovered name for this library.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

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

struct FontStyleFrameApi {
    HleFn init = nullptr, set_scale_pixel = nullptr, set_scale_point = nullptr, set_dpi = nullptr,
          set_slant = nullptr, set_weight = nullptr, unset_scale = nullptr, unset_slant = nullptr,
          unset_weight = nullptr, get_scale_pixel = nullptr, get_scale_point = nullptr,
          get_dpi = nullptr, get_slant = nullptr, get_weight = nullptr, set_surface = nullptr;
    bool registered() const {
        return init && set_scale_pixel && set_scale_point && set_dpi && set_slant && set_weight &&
               unset_scale && unset_slant && unset_weight && get_scale_pixel && get_scale_point &&
               get_dpi && get_slant && get_weight && set_surface;
    }
};
FontStyleFrameApi font_style_frame_api() {
    register_builtin_hle();
    FontStyleFrameApi api;
    api.init = Hle::lookup("la2AOWnHEAc");
    api.set_scale_pixel = Hle::lookup("da4rQ4-+p-4");
    api.set_scale_point = Hle::lookup("O997laxY-Ys");
    api.set_dpi = Hle::lookup("dB4-3Wdwls8");
    api.set_slant = Hle::lookup("394sckksiCU");
    api.set_weight = Hle::lookup("faw77-pEBmU");
    api.unset_scale = Hle::lookup("bePC0L0vQWY");
    api.unset_slant = Hle::lookup("dUmABkAnVgk");
    api.unset_weight = Hle::lookup("hwsuXgmKdaw");
    api.get_scale_pixel = Hle::lookup("2QfqfeLblbg");
    api.get_scale_point = Hle::lookup("7x2xKiiB7MA");
    api.get_dpi = Hle::lookup("VSw18Aqzl0U");
    api.get_slant = Hle::lookup("lOfduYnjgbo");
    api.get_weight = Hle::lookup("HIUdjR-+Wl8");
    api.set_surface = Hle::lookup("0hr-w30SjiI");
    return api;
}
TEST(Font, StyleFrameSurfaceIsRegistered) {
    EXPECT_TRUE(font_style_frame_api().registered()) << "font style-frame surface is registered";
}

TEST(Font, StyleFrameNidsResolveToStubValues) {
    EXPECT_EQ(nid_hash("sceFontStyleFrameInit"), "la2AOWnHEAc");
    EXPECT_EQ(nid_hash("sceFontStyleFrameSetScalePixel"), "da4rQ4-+p-4");
    EXPECT_EQ(nid_hash("sceFontStyleFrameSetScalePoint"), "O997laxY-Ys");
    EXPECT_EQ(nid_hash("sceFontStyleFrameSetResolutionDpi"), "dB4-3Wdwls8");
    EXPECT_EQ(nid_hash("sceFontStyleFrameSetEffectSlant"), "394sckksiCU");
    EXPECT_EQ(nid_hash("sceFontStyleFrameSetEffectWeight"), "faw77-pEBmU");
    EXPECT_EQ(nid_hash("sceFontStyleFrameUnsetScale"), "bePC0L0vQWY");
    EXPECT_EQ(nid_hash("sceFontStyleFrameUnsetEffectSlant"), "dUmABkAnVgk");
    EXPECT_EQ(nid_hash("sceFontStyleFrameUnsetEffectWeight"), "hwsuXgmKdaw");
    EXPECT_EQ(nid_hash("sceFontStyleFrameGetScalePixel"), "2QfqfeLblbg");
    EXPECT_EQ(nid_hash("sceFontStyleFrameGetScalePoint"), "7x2xKiiB7MA");
    EXPECT_EQ(nid_hash("sceFontStyleFrameGetResolutionDpi"), "VSw18Aqzl0U");
    EXPECT_EQ(nid_hash("sceFontStyleFrameGetEffectSlant"), "lOfduYnjgbo");
    EXPECT_EQ(nid_hash("sceFontStyleFrameGetEffectWeight"), "HIUdjR-+Wl8");
    EXPECT_EQ(nid_hash("sceFontRenderSurfaceSetStyleFrame"), "0hr-w30SjiI");
}

TEST(Font, StyleFrameInitSeedsSeventyTwoDpi) {
    const FontStyleFrameApi api = font_style_frame_api();
    ASSERT_TRUE(api.registered());
    uint8_t frame[0x60]{};
    std::memset(frame, 0xcc, sizeof(frame));
    EXPECT_EQ(api.init(addr(frame), 0, 0, 0, 0, 0), 0u) << "StyleFrameInit returns success";
    EXPECT_EQ(frame[0], 0x09u) << "StyleFrameInit writes the magic low byte";
    EXPECT_EQ(frame[1], 0x0Fu) << "StyleFrameInit writes the magic high byte";
    uint32_t h_dpi = 0, v_dpi = 0;
    EXPECT_EQ(api.get_dpi(addr(frame), addr(&h_dpi), addr(&v_dpi), 0, 0, 0), 0u)
        << "GetResolutionDpi reads the seeded frame";
    EXPECT_EQ(h_dpi, 0x48u) << "StyleFrameInit seeds 72 horizontal DPI";
    EXPECT_EQ(v_dpi, 0x48u) << "StyleFrameInit seeds 72 vertical DPI";
}

TEST(Font, StyleFrameScalePixelRoundTrips) {
    const FontStyleFrameApi api = font_style_frame_api();
    ASSERT_TRUE(api.registered());
    uint8_t frame[0x60]{};
    ASSERT_EQ(api.init(addr(frame), 0, 0, 0, 0, 0), 0u);
    using SetFn = int32_t (*)(void*, float, float);
    auto set_pixel = reinterpret_cast<SetFn>(api.set_scale_pixel);
    ASSERT_NE(set_pixel, nullptr);
    EXPECT_EQ(set_pixel((void*)addr(frame), 16.0f, 20.0f), 0) << "SetScalePixel returns success";
    float w = 0.0f, h = 0.0f;
    EXPECT_EQ(api.get_scale_pixel(addr(frame), addr(&w), addr(&h), 0, 0, 0), 0u)
        << "GetScalePixel reads the set frame";
    EXPECT_EQ(w, 16.0f) << "GetScalePixel returns the set width";
    EXPECT_EQ(h, 20.0f) << "GetScalePixel returns the set height";
}

TEST(Font, StyleFrameUnsetScaleRefusesReads) {
    const FontStyleFrameApi api = font_style_frame_api();
    ASSERT_TRUE(api.registered());
    uint8_t frame[0x60]{};
    ASSERT_EQ(api.init(addr(frame), 0, 0, 0, 0, 0), 0u);
    using SetFn = int32_t (*)(void*, float, float);
    auto set_pixel = reinterpret_cast<SetFn>(api.set_scale_pixel);
    ASSERT_NE(set_pixel, nullptr);
    ASSERT_EQ(set_pixel((void*)addr(frame), 16.0f, 20.0f), 0);
    EXPECT_EQ(api.unset_scale(addr(frame), 0, 0, 0, 0, 0), 0u) << "UnsetScale returns success";
    float w = 0.0f, h = 0.0f;
    EXPECT_EQ(api.get_scale_pixel(addr(frame), addr(&w), addr(&h), 0, 0, 0), 0x80460058u)
        << "GetScalePixel refuses after UnsetScale";
}

TEST(Font, StyleFrameSlantClampsAndRoundTrips) {
    const FontStyleFrameApi api = font_style_frame_api();
    ASSERT_TRUE(api.registered());
    uint8_t frame[0x60]{};
    ASSERT_EQ(api.init(addr(frame), 0, 0, 0, 0, 0), 0u);
    using SlantFn = int32_t (*)(void*, float);
    auto set_slant = reinterpret_cast<SlantFn>(api.set_slant);
    ASSERT_NE(set_slant, nullptr);
    EXPECT_EQ(set_slant((void*)addr(frame), 2.0f), 0) << "SetEffectSlant clamps out-of-range";
    float slant = 0.0f;
    EXPECT_EQ(api.get_slant(addr(frame), addr(&slant), 0, 0, 0, 0), 0u)
        << "GetEffectSlant reads the clamped frame";
    EXPECT_EQ(slant, 1.0f) << "SetEffectSlant clamps to +1.0";
}

// The weight is stored as an offset from the neutral 1.0, clamped to +/-0.04, and the getter adds
// the 1.0 back (libSceFont.native.sprx +0xc3a0 / +0xc600). Storing the raw weight instead reads the
// neutral 1.0 back as 2.0.
TEST(Font, StyleFrameEffectWeightIsStoredAsAClampedOffset) {
    const FontStyleFrameApi api = font_style_frame_api();
    ASSERT_TRUE(api.registered());
    using WeightFn = int32_t (*)(void*, float, float, uint32_t);
    auto set_weight = reinterpret_cast<WeightFn>(api.set_weight);
    ASSERT_NE(set_weight, nullptr);
    struct Case {
        float in_x, in_y, want_x, want_y;
    };
    const Case cases[] = {
        {1.0f, 1.0f, 1.0f, 1.0f},   // neutral round-trips
        {1.5f, 0.5f, 1.04f, 0.96f},   // clamped to the band on both sides
        {1.02f, 0.99f, 1.02f, 0.99f},   // inside the band, kept
    };
    for (const Case& c : cases) {
        uint8_t frame[0x60]{};
        ASSERT_EQ(api.init(addr(frame), 0, 0, 0, 0, 0), 0u);
        EXPECT_EQ(set_weight(frame, c.in_x, c.in_y, 0), 0) << "SetEffectWeight returns success";
        float stored_x = 0.0f, stored_y = 0.0f;
        std::memcpy(&stored_x, frame + 0x1c, sizeof(stored_x));
        std::memcpy(&stored_y, frame + 0x20, sizeof(stored_y));
        EXPECT_NEAR(stored_x, c.want_x - 1.0f, 1e-6f) << "frame+0x1c holds the offset from 1.0";
        EXPECT_NEAR(stored_y, c.want_y - 1.0f, 1e-6f) << "frame+0x20 holds the offset from 1.0";
        float x = 0.0f, y = 0.0f;
        uint32_t mode = 0xdeadbeef;
        EXPECT_EQ(api.get_weight(addr(frame), addr(&x), addr(&y), addr(&mode), 0, 0), 0u);
        EXPECT_NEAR(x, c.want_x, 1e-6f) << "GetEffectWeight x for input " << c.in_x;
        EXPECT_NEAR(y, c.want_y, 1e-6f) << "GetEffectWeight y for input " << c.in_y;
        EXPECT_EQ(mode, 0u) << "GetEffectWeight reports mode 0";
    }
    uint8_t frame[0x60]{};
    ASSERT_EQ(api.init(addr(frame), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(static_cast<uint32_t>(set_weight(frame, 1.0f, 1.0f, 1)), 0x80460002u)
        << "a non-zero mode is rejected";
    float x = 0.0f;
    EXPECT_EQ(api.get_weight(addr(frame), addr(&x), 0, 0, 0, 0), 0x80460058u)
        << "a rejected set leaves the weight unset";
}

// libSceFont.native.sprx +0xabb0: the frame pointer lands at surface+0x28 whether or not it is
// null, the dword at surface+0x30 is zeroed, and surface+0xe bit 0 says whether a frame is set. A
// non-null frame without the magic is refused with nothing written.
TEST(Font, RenderSurfaceSetStyleFrameRecordsTheFrame) {
    const FontStyleFrameApi api = font_style_frame_api();
    ASSERT_TRUE(api.registered());
    uint8_t frame[0x60]{};
    ASSERT_EQ(api.init(addr(frame), 0, 0, 0, 0, 0), 0u);
    auto read_ptr = [](const uint8_t* p) {
        uint64_t v = 0;
        std::memcpy(&v, p, sizeof(v));
        return v;
    };
    auto read_u32 = [](const uint8_t* p) {
        uint32_t v = 0;
        std::memcpy(&v, p, sizeof(v));
        return v;
    };

    uint8_t surface[0x80];
    std::memset(surface, 0xcc, sizeof(surface));
    surface[0xe] = 0xfe;
    EXPECT_EQ(api.set_surface(addr(surface), addr(frame), 0, 0, 0, 0), 0u);
    EXPECT_EQ(surface[0xe], 0xffu) << "a frame sets bit 0 of surface+0xe and keeps the rest";
    EXPECT_EQ(read_ptr(surface + 0x28), addr(frame)) << "the frame pointer lands at surface+0x28";
    EXPECT_EQ(read_u32(surface + 0x30), 0u) << "the dword at surface+0x30 is zeroed";
    EXPECT_EQ(read_u32(surface + 0x34), 0xccccccccu) << "the write stops at the dword";

    std::memset(surface, 0xcc, sizeof(surface));
    EXPECT_EQ(api.set_surface(addr(surface), 0, 0, 0, 0, 0), 0u) << "a null frame detaches";
    EXPECT_EQ(surface[0xe], 0xccu & 0xfeu) << "a null frame clears bit 0 of surface+0xe";
    EXPECT_EQ(read_ptr(surface + 0x28), 0u) << "a null frame is still stored at surface+0x28";
    EXPECT_EQ(read_u32(surface + 0x30), 0u) << "the dword at surface+0x30 is zeroed";

    uint8_t bogus[0x60]{};
    std::memset(surface, 0xcc, sizeof(surface));
    EXPECT_EQ(api.set_surface(addr(surface), addr(bogus), 0, 0, 0, 0), 0x80460002u)
        << "a frame without the magic is refused";
    for (size_t i = 0; i < sizeof(surface); ++i)
        ASSERT_EQ(surface[i], 0xccu) << "a refused frame writes nothing, byte " << i;
    EXPECT_EQ(api.set_surface(0, addr(frame), 0, 0, 0, 0), 0x80460002u)
        << "a null surface is refused";
}

// ---------------------------------------------------------------------------------------------
// Kerning, library back-link, text-source rewind and character back-link (libSceFont.native.sprx).
// ---------------------------------------------------------------------------------------------
namespace {
constexpr uint64_t kErrInvalidParam = 0x80460002u;
constexpr uint64_t kErrInvalidHandle = 0x80460005u;
constexpr uint64_t kErrInvalidTextSource = 0x80460008u;
constexpr uint64_t kErrNoGlyph = 0x80460041u;
constexpr uint64_t kErrNoRenderer = 0x80460061u;

void be16(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 8));
    v.push_back((uint8_t)x);
}
void be32(std::vector<uint8_t>& v, uint32_t x) {
    be16(v, x >> 16);
    be16(v, x & 0xffff);
}

// A minimal TrueType font: .notdef, 'A' (glyph 1) and 'V' (glyph 2), unitsPerEm 1000, ascent 800,
// descent -200, so at the default 16 px face scale one font unit is 16/1000 px. Optionally a format-0
// `kern` table and/or a GPOS pair-adjustment lookup, each kerning the pair (A, V) by its own amount, so
// a test can tell which table answered.
std::vector<uint8_t> kerning_font(int kern_units, bool with_kern, int gpos_units, bool with_gpos) {
    struct Table {
        uint32_t tag;
        std::vector<uint8_t> data;
    };
    std::vector<Table> t;
    auto tag = [](const char* s) {
        return (uint32_t)s[0] << 24 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 8 | (uint32_t)s[3];
    };
    {   // cmap: one (3,1) format-6 subtable over 'A'..'V'; A -> 1, V -> 2, the rest -> 0
        std::vector<uint8_t> d;
        be16(d, 0);
        be16(d, 1);
        be16(d, 3);
        be16(d, 1);
        be32(d, 12);
        const uint32_t count = 'V' - 'A' + 1;
        be16(d, 6);
        be16(d, 10 + 2 * count);
        be16(d, 0);
        be16(d, 'A');
        be16(d, count);
        for (uint32_t c = 'A'; c <= 'V'; ++c) be16(d, c == 'A' ? 1 : c == 'V' ? 2 : 0);
        t.push_back({tag("cmap"), d});
    }
    {   // glyf: three empty glyphs are enough -- kerning never reads an outline
        t.push_back({tag("glyf"), {}});
    }
    {   // head
        std::vector<uint8_t> d;
        be32(d, 0x00010000);
        be32(d, 0x00010000);
        be32(d, 0);
        be32(d, 0x5F0F3CF5);
        be16(d, 0);
        be16(d, 1000);
        for (int i = 0; i < 16; ++i) d.push_back(0);
        for (int i = 0; i < 4; ++i) be16(d, 0);
        be16(d, 0);
        be16(d, 8);
        be16(d, 2);
        be16(d, 0);
        be16(d, 0);
        t.push_back({tag("head"), d});
    }
    {   // hhea
        std::vector<uint8_t> d;
        be32(d, 0x00010000);
        be16(d, 800);
        be16(d, (uint16_t)-200);
        be16(d, 0);
        be16(d, 800);
        for (int i = 0; i < 11; ++i) be16(d, 0);
        be16(d, 3);
        t.push_back({tag("hhea"), d});
    }
    {   // hmtx
        std::vector<uint8_t> d;
        for (int i = 0; i < 3; ++i) {
            be16(d, 800);
            be16(d, 0);
        }
        t.push_back({tag("hmtx"), d});
    }
    {   // loca (short): every glyph empty
        std::vector<uint8_t> d;
        for (int i = 0; i < 4; ++i) be16(d, 0);
        t.push_back({tag("loca"), d});
    }
    {   // maxp
        std::vector<uint8_t> d;
        be32(d, 0x00010000);
        be16(d, 3);
        for (int i = 0; i < 13; ++i) be16(d, 0);
        t.push_back({tag("maxp"), d});
    }
    if (with_kern) {   // kern v0, one horizontal format-0 subtable, one pair (1, 2)
        std::vector<uint8_t> d;
        be16(d, 0);
        be16(d, 1);
        be16(d, 0);
        be16(d, 6 + 8 + 6);
        be16(d, 0x0001);
        be16(d, 1);
        be16(d, 6);
        be16(d, 0);
        be16(d, 0);
        be16(d, 1);
        be16(d, 2);
        be16(d, (uint16_t)kern_units);
        t.push_back({tag("kern"), d});
    }
    if (with_gpos) {   // GPOS 1.0: one type-2 lookup, PairPos format 1, XAdvance on the first glyph
        std::vector<uint8_t> d;
        be16(d, 1);
        be16(d, 0);
        be16(d, 10);
        be16(d, 12);
        be16(d, 14);
        be16(d, 0);   // ScriptList @10: no scripts
        be16(d, 0);   // FeatureList @12: no features
        be16(d, 1);
        be16(d, 4);   // LookupList @14: one lookup at +4
        be16(d, 2);
        be16(d, 0);
        be16(d, 1);
        be16(d, 8);   // Lookup @18: type 2, one subtable at +8
        // PairPosFormat1 @26: coverage at +12, valueFormat1 = XAdvance, one PairSet at +18
        be16(d, 1);
        be16(d, 12);
        be16(d, 4);
        be16(d, 0);
        be16(d, 1);
        be16(d, 18);
        be16(d, 1);
        be16(d, 1);
        be16(d, 1);   // Coverage format 1: glyph 1
        be16(d, 1);
        be16(d, 2);
        be16(d, (uint16_t)gpos_units);   // PairSet: (2, xAdvance)
        t.push_back({tag("GPOS"), d});
    }
    std::sort(t.begin(), t.end(), [](const Table& a, const Table& b) { return a.tag < b.tag; });
    std::vector<uint8_t> out;
    const uint32_t n = (uint32_t)t.size();
    be32(out, 0x00010000);
    be16(out, n);
    be16(out, 0);
    be16(out, 0);
    be16(out, 0);
    uint32_t offset = 12 + 16 * n;
    for (const Table& tb : t) {
        be32(out, tb.tag);
        be32(out, 0);
        be32(out, offset);
        be32(out, (uint32_t)tb.data.size());
        offset += ((uint32_t)tb.data.size() + 3) & ~3u;
    }
    for (const Table& tb : t) {
        out.insert(out.end(), tb.data.begin(), tb.data.end());
        while (out.size() % 4) out.push_back(0);
    }
    return out;
}

struct KerningApi {
    HleFn create_library = nullptr, create_renderer = nullptr, open_set = nullptr,
          open_memory = nullptr, bind = nullptr, close = nullptr, kerning = nullptr,
          render_kerning = nullptr, get_library = nullptr;
    void* library = nullptr;
    bool ready() const {
        return create_library && create_renderer && open_set && open_memory && bind && close &&
               kerning && render_kerning && get_library && library;
    }
};
KerningApi kerning_api() {
    register_builtin_hle();
    KerningApi api;
    api.create_library = Hle::lookup("n590hj5Oe-k");
    api.create_renderer = Hle::lookup("u5fZd3KZcs0");
    api.open_set = Hle::lookup("cKYtVmeSTcw");
    api.open_memory = Hle::lookup("KXUpebrFk1U");
    api.bind = Hle::lookup("3OdRkSjOcog");
    api.close = Hle::lookup("vzHs3C8lWJk");
    api.kerning = Hle::lookup("sDuhHGNhHvE");
    api.render_kerning = Hle::lookup("ryPlnDDI3rU");
    api.get_library = Hle::lookup("LzmHDnlcwfQ");
    static uint8_t mem[64]{};
    if (api.create_library) api.create_library(addr(mem), 0, 0, addr(&api.library), 0, 0);
    return api;
}
// The face opened over `font`, or nullptr.
void* open_memory_face(const KerningApi& api, const std::vector<uint8_t>& font) {
    void* handle = nullptr;
    if (api.open_memory(addr(api.library), addr(font.data()), font.size(), 0, addr(&handle), 0))
        return nullptr;
    return handle;
}
}   // namespace

TEST(Font, KerningNidsMatchTheFirmwareDump) {
    EXPECT_EQ(nid_hash("sceFontGetKerning"), "sDuhHGNhHvE");
    EXPECT_EQ(nid_hash("sceFontGetRenderScaledKerning"), "ryPlnDDI3rU");
    EXPECT_EQ(nid_hash("sceFontGetLibrary"), "LzmHDnlcwfQ");
    EXPECT_EQ(nid_hash("sceFontTextSourceRewind"), "VRFd3diReec");
    EXPECT_EQ(nid_hash("sceFontCharacterRefersTextBack"), "6Gqlv5KdTbU");
}

// The native core's checks, in its order, and the 12-byte zeroing of every error path: out[3] is a
// poison the error paths must NOT touch, because the core writes a qword and a dword and no more.
TEST(Font, KerningFollowsTheNativeCheckOrder) {
    const KerningApi api = kerning_api();
    ASSERT_TRUE(api.ready());
    void* face = nullptr;
    ASSERT_EQ(api.open_set(addr(api.library), 0, 0, 0, addr(&face), 0), 0u);
    float out[4];
    auto poison = [&] { out[0] = out[1] = out[2] = out[3] = 7.0f; };
    auto zeroed12 = [&] {
        return out[0] == 0.0f && out[1] == 0.0f && out[2] == 0.0f && out[3] == 7.0f;
    };

    uint8_t decoy[64]{};
    poison();
    EXPECT_EQ(api.kerning(addr(decoy), 'A', 'V', addr(out), 0, 0), kErrInvalidHandle);
    EXPECT_TRUE(zeroed12()) << "a foreign face is refused and the first 12 bytes cleared";
    EXPECT_EQ(api.kerning(0, 'A', 'V', 0, 0, 0), kErrInvalidHandle) << "the face is checked first";
    poison();
    EXPECT_EQ(api.kerning(addr(face), 'A', 0, addr(out), 0, 0), kErrNoGlyph) << "code == 0";
    EXPECT_TRUE(zeroed12());
    EXPECT_EQ(api.kerning(addr(face), 'A', 0, 0, 0, 0), kErrNoGlyph)
        << "code == 0 is checked before the NULL out";
    EXPECT_EQ(api.kerning(addr(face), 'A', 'V', 0, 0, 0), kErrInvalidParam) << "NULL out";
    poison();
    EXPECT_EQ(api.kerning(addr(face), 0, 'V', addr(out), 0, 0), 0u) << "preCode == 0 is success";
    EXPECT_TRUE(zeroed12()) << "...with the first 12 bytes cleared";
    poison();
    EXPECT_EQ(api.kerning(addr(face), 'A', 'V', addr(out), 0, 0), 0u);
    EXPECT_EQ(out[0], 0.0f) << "a system-font face has no table to kern from";
    EXPECT_EQ(out[3], 0.0f) << "a success writes all sixteen bytes";

    void* orphan = nullptr;
    ASSERT_EQ(api.open_set(0, 0, 0, 0, addr(&orphan), 0), 0u);
    poison();
    EXPECT_EQ(api.kerning(addr(orphan), 'A', 'V', addr(out), 0, 0), kErrInvalidHandle)
        << "a face with no library (face+0x28 == 0)";
    EXPECT_TRUE(zeroed12());
    EXPECT_EQ(api.close(addr(orphan), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(api.close(addr(face), 0, 0, 0, 0, 0), 0u);
}

TEST(Font, RenderScaledKerningNeedsABoundRenderer) {
    const KerningApi api = kerning_api();
    ASSERT_TRUE(api.ready());
    void* face = nullptr;
    ASSERT_EQ(api.open_set(addr(api.library), 0, 0, 0, addr(&face), 0), 0u);
    float out[4] = {7.0f, 7.0f, 7.0f, 7.0f};
    EXPECT_EQ(api.render_kerning(addr(face), 0, 'V', addr(out), 0, 0), kErrNoRenderer)
        << "no renderer answers 0x80460061 even when preCode is 0";
    EXPECT_EQ(out[0], 0.0f);
    EXPECT_EQ(out[3], 7.0f);
    EXPECT_EQ(api.kerning(addr(face), 'A', 'V', addr(out), 0, 0), 0u)
        << "the plain variant does not need one";
    uint8_t mem[64]{};
    void* renderer = nullptr;
    ASSERT_EQ(api.create_renderer(addr(mem), 0, addr(&renderer), 0, 0, 0), 0u);
    ASSERT_EQ(api.bind(addr(face), addr(renderer), 0, 0, 0, 0), 0u);
    EXPECT_EQ(api.render_kerning(addr(face), 0, 'V', addr(out), 0, 0), 0u);
    EXPECT_EQ(api.render_kerning(addr(face), 'A', 'V', addr(out), 0, 0), 0u);
    EXPECT_EQ(api.close(addr(face), 0, 0, 0, 0, 0), 0u);
}

// The value itself, out of the title's own font file. The FreeType driver prefers the classic kern
// table and only falls back to GPOS when there is none; stb's public helper does the opposite, so the
// "both tables" arm is the one that separates the two precedences.
TEST(Font, KerningReadsTheKernTableBeforeGpos) {
    const KerningApi api = kerning_api();
    ASSERT_TRUE(api.ready());
    const float unit = 16.0f / 1000.0f;   // default 16 px face, ascent - descent = 1000 units

    const std::vector<uint8_t> kern_only = kerning_font(-100, true, 0, false);
    const std::vector<uint8_t> both = kerning_font(-100, true, -40, true);
    const std::vector<uint8_t> gpos_only = kerning_font(0, false, -40, true);
    void* faces[3] = {open_memory_face(api, kern_only), open_memory_face(api, both),
                      open_memory_face(api, gpos_only)};
    for (void* f : faces) ASSERT_NE(f, nullptr) << "the synthesized font parses";
    const float expected[3] = {-100.0f * unit, -100.0f * unit, -40.0f * unit};
    const char* what[3] = {"kern only", "kern and GPOS: kern wins", "GPOS only: the fallback"};
    for (int i = 0; i < 3; ++i) {
        float out[4] = {7.0f, 7.0f, 7.0f, 7.0f};
        ASSERT_EQ(api.kerning(addr(faces[i]), 'A', 'V', addr(out), 0, 0), 0u) << what[i];
        EXPECT_FLOAT_EQ(out[0], expected[i]) << what[i];
        EXPECT_EQ(out[1], 0.0f) << what[i];
        EXPECT_EQ(out[2], 0.0f) << what[i];
        EXPECT_EQ(out[3], 0.0f) << what[i];
        ASSERT_EQ(api.kerning(addr(faces[i]), 'V', 'A', addr(out), 0, 0), 0u);
        EXPECT_EQ(out[0], 0.0f) << what[i] << ": a pair the table does not list";
    }
    for (void* f : faces) EXPECT_EQ(api.close(addr(f), 0, 0, 0, 0, 0), 0u);
}

TEST(Font, GetLibraryValidatesBeforeAnswering) {
    const KerningApi api = kerning_api();
    ASSERT_TRUE(api.ready());
    void* face = nullptr;
    ASSERT_EQ(api.open_set(addr(api.library), 0, 0, 0, addr(&face), 0), 0u);
    void* back = nullptr;
    EXPECT_EQ(api.get_library(addr(face), addr(&back), 0, 0, 0, 0), 0u);
    EXPECT_EQ(back, api.library) << "GetLibrary reports the opening library";
    EXPECT_EQ(api.get_library(addr(face), 0, 0, 0, 0, 0), kErrInvalidParam) << "NULL out";

    uint8_t decoy[64]{};
    back = &back;
    EXPECT_EQ(api.get_library(addr(decoy), addr(&back), 0, 0, 0, 0), kErrInvalidHandle);
    EXPECT_EQ(back, nullptr) << "a refused call clears the out";
    EXPECT_EQ(api.get_library(addr(decoy), 0, 0, 0, 0, 0), kErrInvalidHandle)
        << "the face is checked before the out";

    void* orphan = nullptr;
    ASSERT_EQ(api.open_set(0, 0, 0, 0, addr(&orphan), 0), 0u);
    back = &back;
    EXPECT_EQ(api.get_library(addr(orphan), addr(&back), 0, 0, 0, 0), kErrInvalidHandle)
        << "a face with no library is refused, not answered with NULL";
    EXPECT_EQ(back, nullptr);
    uint8_t not_a_library[64]{};
    void* stranger = nullptr;
    ASSERT_EQ(api.open_set(addr(not_a_library), 0, 0, 0, addr(&stranger), 0), 0u);
    EXPECT_EQ(api.get_library(addr(stranger), addr(&back), 0, 0, 0, 0), kErrInvalidHandle)
        << "a library pointer without the library magic";
    EXPECT_EQ(api.get_library(addr(stranger), 0, 0, 0, 0, 0), kErrInvalidHandle)
        << "the library is checked before the out";
    for (void* f : {face, orphan, stranger}) EXPECT_EQ(api.close(addr(f), 0, 0, 0, 0, 0), 0u);
}

// TextSourceInit takes a snapshot that Rewind restores in full (native +0xf640 / +0xf7f0), and
// SetDefaultFont writes both copies. Offsets are the native layout: magic +0, form +4, start +8,
// end +0x10, current +0x18, parser +0x20, object +0x28, default font +0x30.
TEST(Font, TextSourceRewindRestoresTheInitSnapshot) {
    const AstroFont font = astro_font();
    ASSERT_TRUE(font.registered());
    HleFn rewind = Hle::lookup("VRFd3diReec");
    HleFn set_default = Hle::lookup("eCRMCSk96NU");
    HleFn set_form = Hle::lookup("OqQKX0h5COw");
    ASSERT_NE(rewind, nullptr);
    ASSERT_NE(set_default, nullptr);
    ASSERT_NE(set_form, nullptr);
    uint64_t src[12];
    std::memset(src, 0xcc, sizeof(src));
    const char text[] = "ASTRO";
    int parser = 0, object = 0, font_a = 0, font_b = 0;
    ASSERT_EQ(font.text_init(addr(src), addr(text), sizeof(text), addr(&parser), addr(&object), 0),
              0u);
    EXPECT_EQ(src[0] & 0xffff, 0x0f04u) << "Init stamps the text-source magic";
    EXPECT_EQ(src[0] >> 32, 0x10u) << "...and the default writing form";
    ASSERT_EQ(set_default(addr(src), addr(&font_a), 0, 0, 0, 0), 0u);
    EXPECT_EQ(src[6], addr(&font_a));

    // Move every live field, the way a consumer and a later SetDefaultFont-less edit would.
    src[1] = 0x1111;
    src[2] = 0x2222;
    src[3] = 0x3333;
    src[4] = 0x4444;
    src[5] = 0x5555;
    src[6] = addr(&font_b);
    ASSERT_EQ(rewind(addr(src), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(src[1], addr(text)) << "start restored";
    EXPECT_EQ(src[2], addr(text) + sizeof(text)) << "end restored";
    EXPECT_EQ(src[3], addr(text)) << "the cursor is back at the start";
    EXPECT_EQ(src[4], addr(&parser)) << "parser restored";
    EXPECT_EQ(src[5], addr(&object)) << "object restored";
    EXPECT_EQ(src[6], addr(&font_a)) << "the default font SetDefaultFont recorded is restored";

    EXPECT_EQ(set_form(addr(src), 0x11, 0, 0, 0, 0), 0u);
    EXPECT_EQ(src[0] & 0xffff, 0x0f04u) << "SetWritingForm leaves the magic alone";
    EXPECT_EQ(src[0] >> 32, 0x11u) << "...and writes the form at +4";
    EXPECT_EQ(set_form(addr(src), 0x13, 0, 0, 0, 0), kErrInvalidParam) << "forms are 0x10..0x12";
    EXPECT_EQ(set_form(addr(src), 0x0f, 0, 0, 0, 0), kErrInvalidParam);

    EXPECT_EQ(rewind(0, 0, 0, 0, 0, 0), kErrInvalidParam) << "NULL source";
    uint64_t foreign[12]{};
    EXPECT_EQ(rewind(addr(foreign), 0, 0, 0, 0, 0), kErrInvalidTextSource) << "no magic";
    EXPECT_EQ(set_default(addr(foreign), addr(&font_a), 0, 0, 0, 0), kErrInvalidTextSource);
    EXPECT_EQ(foreign[6], 0u) << "a refused SetDefaultFont writes nothing";
    EXPECT_EQ(set_default(0, addr(&font_a), 0, 0, 0, 0), kErrInvalidParam);
    EXPECT_EQ(set_form(addr(foreign), 0x11, 0, 0, 0, 0), kErrInvalidTextSource);
    EXPECT_EQ(set_form(0, 0x11, 0, 0, 0, 0), kErrInvalidParam);
}

// RefersTextNext / RefersTextBack follow their own link and step over characters flagged at +0x31 or
// +0x33. The chain is built in test memory with the native TextCharacter layout.
TEST(Font, CharacterLinksWalkTheirOwnDirectionAndSkipFlagged) {
    register_builtin_hle();
    HleFn next = Hle::lookup("BkjBP+YC19w");
    HleFn back = Hle::lookup("6Gqlv5KdTbU");
    ASSERT_NE(next, nullptr);
    ASSERT_NE(back, nullptr);
    struct Character {
        Character* prev;
        Character* next;
        void* order;
        void* font;
        void* shape;
        uint32_t code;
        uint8_t reserved[5];
        uint8_t flag_0x31;
        uint8_t reserved_0x32;
        uint8_t flag_0x33;
        uint8_t tail[12];
    };
    static_assert(offsetof(Character, flag_0x31) == 0x31 && offsetof(Character, flag_0x33) == 0x33);
    Character c[4]{};
    for (int i = 0; i < 4; ++i) {
        c[i].prev = i > 0 ? &c[i - 1] : nullptr;
        c[i].next = i < 3 ? &c[i + 1] : nullptr;
    }
    EXPECT_EQ(next(addr(&c[0]), 0, 0, 0, 0, 0), addr(&c[1]));
    EXPECT_EQ(back(addr(&c[2]), 0, 0, 0, 0, 0), addr(&c[1])) << "back follows prev, not next";
    EXPECT_EQ(back(addr(&c[0]), 0, 0, 0, 0, 0), 0u) << "the head has nothing before it";
    EXPECT_EQ(next(addr(&c[3]), 0, 0, 0, 0, 0), 0u);
    c[1].flag_0x31 = 1;
    c[2].flag_0x33 = 1;
    EXPECT_EQ(next(addr(&c[0]), 0, 0, 0, 0, 0), addr(&c[3]))
        << "flagged characters are stepped over";
    EXPECT_EQ(back(addr(&c[3]), 0, 0, 0, 0, 0), addr(&c[0]));
    EXPECT_EQ(next(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(back(0, 0, 0, 0, 0, 0), 0u);
}
