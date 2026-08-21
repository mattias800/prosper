#pragma once
// overlay_text.hpp — burn a short, legible string into an RGBA8 image prosper produced.
//
// WHY THIS IS PROJECT CODE AND NOT A VENDORED FONT
// ------------------------------------------------
// third_party/ carries stb_image.h (a DECODER) and imgui; neither gives a CPU text rasterizer that
// works on a bare pixel buffer, and vendoring a new library is an explicit owner decision rather
// than something a lane takes on its own initiative (CLAUDE.md). What the annotation needs is small
// enough that a hand-authored 5x7 bitmap font is the proportionate answer: it is dependency-free,
// deterministic across platforms, and unit-testable without a device.
//
// The font is UPPERCASE-ONLY, and callers get upper-casing for free. That is a deliberate limit on
// how much glyph authoring this feature justifies, not an oversight: `3.4 FPS @ 3840X2160` is
// perfectly legible, and an unsupported character renders as a hollow box rather than as nothing,
// so a missing glyph is visible instead of silently dropping information.
//
// WHAT THIS IS ALLOWED TO BE USED FOR
// -----------------------------------
// Annotation that prosper draws over its own output — an fps counter, a frame ordinal, a route
// name, a date. Per CLAUDE.md, that is admissible progression evidence: it adds measured facts
// about the run and does not change what the run rendered. What it must never be used for is making
// a capture show more than the emulator achieved. Keep it legible and clear of whatever the capture
// is evidence FOR, and say in the caption that it is on.

#include <cstdint>
#include <string>
#include <vector>

namespace prosper::screenshot {

// Glyph cell, before scaling. Advance includes the one-column gap between characters.
constexpr int kGlyphWidth = 5;
constexpr int kGlyphHeight = 7;
constexpr int kGlyphAdvance = kGlyphWidth + 1;

struct OverlayStyle {
    int scale = 1;                              // integer magnification of the 5x7 cell
    int margin = 8;                             // distance from the image edge, in final pixels
    int padding = 6;                            // backing-box inset around the text, in final pixels
    int line_spacing = 2;                       // extra rows between lines, in glyph cells
    uint8_t foreground[4] = {255, 255, 255, 255};
    uint8_t background[4] = {0, 0, 0, 150};     // translucent so the frame stays readable underneath
};

// Legibility is a function of the image, not of the font: a 5x7 cell is unreadable at 4K unscaled.
// 1x below 1920 wide, 2x at 1080p, 4x at 2160p.
int overlay_scale_for_width(uint32_t width);

// Where `draw_overlay_text` will put the backing box, in image pixels. Exposed so a caller (or a
// test) can reason about what the annotation covers without rendering it.
struct OverlayBox {
    int x = 0, y = 0, width = 0, height = 0;
};
OverlayBox overlay_box(const std::vector<std::string>& lines, const OverlayStyle& style);

// Draw `lines` into `rgba` (w*h, 4 bytes/pixel) at the top-left, in place. Alpha-blends, so the
// backing box dims rather than erases. Clips to the image; an image too small for the text is left
// untouched rather than corrupted. Returns the box actually drawn (zero-sized if nothing was).
OverlayBox draw_overlay_text(std::vector<uint8_t>& rgba, uint32_t width, uint32_t height,
                             const std::vector<std::string>& lines, const OverlayStyle& style);

// True when the font has a glyph for `c` (after upper-casing). Everything else renders as the
// hollow-box fallback.
bool overlay_font_has_glyph(char c);

// Every glyph row must be exactly kGlyphWidth characters: the rasterizer indexes rows[gy][gx] for
// gx in [0, kGlyphWidth), so a short row reads past the end of a string literal. The table is
// hand-authored, which is precisely why this is checked rather than trusted -- a typo'd row is
// invisible on inspection and out of bounds at runtime. Returns the offending glyph's character in
// `bad_glyph` when it fails, so the test can name it.
bool overlay_font_rows_are_well_formed(char* bad_glyph = nullptr);

} // namespace prosper::screenshot
