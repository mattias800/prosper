// test_overlay_text.cpp — the burned-in annotation must actually be there, and must not damage the
// frame it annotates.
//
// The failure this guards against is a silent one: an overlay that draws nothing (bad glyph table,
// clipped box, wrong stride) leaves a capture that looks exactly like a capture taken without the
// flag, so the caption says "fps overlay on" and the image says nothing. Every arm here therefore
// asserts on PIXELS rather than on the call succeeding.
//
// Pure: no Vulkan, no dump, no image codec.
#include "overlay_text.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace prosper::screenshot;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

namespace {

constexpr uint32_t kW = 640, kH = 360;

std::vector<uint8_t> flat_image(uint8_t value) {
    std::vector<uint8_t> px(static_cast<size_t>(kW) * kH * 4, value);
    for (size_t i = 3; i < px.size(); i += 4) px[i] = 255;
    return px;
}

size_t changed_pixels(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    size_t n = 0;
    for (size_t i = 0; i + 3 < a.size(); i += 4)
        if (a[i] != b[i] || a[i + 1] != b[i + 1] || a[i + 2] != b[i + 2]) n++;
    return n;
}

void font_table_is_well_formed() {
    // A row of the wrong length would walk off the end of a string literal during rasterization.
    // Checked through the public surface: every character the overlay can be asked to draw must
    // resolve to a glyph or to the deliberate fallback, and neither may crash or draw nothing.
    const std::string alphabet =
        " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.,:-+/()[]%@=?!_";
    bool all_present = true;
    for (char c : alphabet) all_present &= overlay_font_has_glyph(c);
    CHECK(all_present, "every documented character has a glyph");

    CHECK(overlay_font_has_glyph('a') && overlay_font_has_glyph('z'),
          "lowercase resolves through upper-casing");
    CHECK(!overlay_font_has_glyph('~'), "an undocumented character reports no glyph");

    // The fallback must DRAW. An unsupported character rendering as blank is the silent-loss case.
    std::vector<uint8_t> image = flat_image(0x40);
    const std::vector<uint8_t> before = image;
    OverlayStyle style;
    style.scale = 2;
    style.background[3] = 0;   // isolate the glyph pixels from the backing box
    draw_overlay_text(image, kW, kH, {"~"}, style);
    CHECK(changed_pixels(before, image) > 0, "an unsupported character renders as a visible box");
}

void the_overlay_is_actually_drawn() {
    std::vector<uint8_t> image = flat_image(0);
    const std::vector<uint8_t> before = image;
    OverlayStyle style;
    style.scale = 2;

    const OverlayBox box = draw_overlay_text(image, kW, kH, {"3.4 FPS", "3840X2160"}, style);
    CHECK(box.width > 0 && box.height > 0, "the overlay reports the box it drew");
    CHECK(changed_pixels(before, image) > 200, "the overlay changed a substantial number of pixels");

    // Every changed pixel must be inside the reported box: an annotation that claims one footprint
    // and paints another cannot be kept clear of what the capture is evidence for.
    bool all_inside = true;
    for (uint32_t y = 0; y < kH && all_inside; y++)
        for (uint32_t x = 0; x < kW; x++) {
            const size_t i = (static_cast<size_t>(y) * kW + x) * 4;
            const bool changed = image[i] != before[i] || image[i + 1] != before[i + 1] ||
                                 image[i + 2] != before[i + 2];
            const bool inside = static_cast<int>(x) >= box.x && static_cast<int>(x) < box.x + box.width &&
                                static_cast<int>(y) >= box.y && static_cast<int>(y) < box.y + box.height;
            if (changed && !inside) { all_inside = false; break; }
        }
    CHECK(all_inside, "nothing outside the reported box was touched");

    // ...and the rest of the frame is byte-identical, so a metric computed on an overlaid image
    // would still describe the same picture everywhere the annotation is not.
    CHECK(box.x + box.width < static_cast<int>(kW) / 2 &&
              box.y + box.height < static_cast<int>(kH) / 2,
          "the annotation occupies a corner, not the middle of the frame");
}

void different_text_draws_differently() {
    OverlayStyle style;
    style.scale = 2;
    std::vector<uint8_t> a = flat_image(0), b = flat_image(0);
    draw_overlay_text(a, kW, kH, {"1.0 FPS"}, style);
    draw_overlay_text(b, kW, kH, {"60.0 FPS"}, style);
    CHECK(changed_pixels(a, b) > 0, "two different strings produce two different images");

    std::vector<uint8_t> c = flat_image(0);
    draw_overlay_text(c, kW, kH, {"1.0 FPS"}, style);
    CHECK(changed_pixels(a, c) == 0, "the same string is rasterized deterministically");
}

void it_refuses_rather_than_corrupts() {
    OverlayStyle style;
    style.scale = 8;   // far larger than the image below can hold
    std::vector<uint8_t> tiny(4 * 4 * 4, 0x11);
    const std::vector<uint8_t> before = tiny;
    const OverlayBox box = draw_overlay_text(tiny, 4, 4, {"3.4 FPS"}, style);
    CHECK(box.width == 0 && box.height == 0, "an image too small for the text reports no box");
    CHECK(tiny == before, "...and is left byte-for-byte unchanged rather than half-drawn");

    std::vector<uint8_t> mismatched(16, 0);
    CHECK(draw_overlay_text(mismatched, kW, kH, {"X"}, style).width == 0,
          "a buffer whose size disagrees with its dimensions is refused");

    std::vector<uint8_t> image = flat_image(0);
    const std::vector<uint8_t> unchanged = image;
    draw_overlay_text(image, kW, kH, {}, style);
    CHECK(image == unchanged, "no lines means no drawing");
}

void scale_tracks_resolution() {
    CHECK(overlay_scale_for_width(3840) == 4, "4K gets a 4x cell (28 px tall)");
    CHECK(overlay_scale_for_width(1920) == 2, "1080p gets a 2x cell (14 px tall)");
    CHECK(overlay_scale_for_width(1280) == 1, "720p gets a 1x cell");
    CHECK(overlay_scale_for_width(64) == 1, "a tiny frame never gets a zero or negative scale");
}

} // namespace

int main() {
    std::printf("== font table ==\n");        font_table_is_well_formed();
    std::printf("== it draws ==\n");          the_overlay_is_actually_drawn();
    std::printf("== it discriminates ==\n");  different_text_draws_differently();
    std::printf("== it refuses ==\n");        it_refuses_rather_than_corrupts();
    std::printf("== scale ==\n");             scale_tracks_resolution();
    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
