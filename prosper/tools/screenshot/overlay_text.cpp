// overlay_text.cpp — see overlay_text.hpp.
//
// The font is written as rows of '#' and ' ' rather than as packed hex, so a reviewer can see the
// letter. It is converted to a bitmask once, on first use.
#include "overlay_text.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>

namespace prosper::screenshot {
namespace {

struct Glyph {
    char code;
    const char* rows[kGlyphHeight];
};

// A hollow box: an unsupported character must be VISIBLE, never silently absent.
constexpr const char* kFallbackRows[kGlyphHeight] = {
    "#####",
    "#   #",
    "#   #",
    "#   #",
    "#   #",
    "#   #",
    "#####",
};

const Glyph kFont[] = {
    {' ', {"     ", "     ", "     ", "     ", "     ", "     ", "     "}},

    {'0', {" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "}},
    {'1', {"  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "}},
    {'2', {" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"}},
    {'3', {"#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "}},
    {'4', {"   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "}},
    {'5', {"#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "}},
    {'6', {"  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "}},
    {'7', {"#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "}},
    {'8', {" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "}},
    {'9', {" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "}},

    {'A', {" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"}},
    {'B', {"#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "}},
    {'C', {" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "}},
    {'D', {"###  ", "#  # ", "#   #", "#   #", "#   #", "#  # ", "###  "}},
    {'E', {"#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"}},
    {'F', {"#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "}},
    {'G', {" ### ", "#   #", "#    ", "#  ##", "#   #", "#   #", " ### "}},
    {'H', {"#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"}},
    {'I', {" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "}},
    {'J', {"  ###", "    #", "    #", "    #", "    #", "#   #", " ### "}},
    {'K', {"#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"}},
    {'L', {"#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"}},
    {'M', {"#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"}},
    {'N', {"#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"}},
    {'O', {" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "}},
    {'P', {"#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "}},
    {'Q', {" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"}},
    {'R', {"#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"}},
    {'S', {" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "}},
    {'T', {"#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "}},
    {'U', {"#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "}},
    {'V', {"#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "}},
    {'W', {"#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"}},
    {'X', {"#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"}},
    {'Y', {"#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "}},
    {'Z', {"#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"}},

    {'.', {"     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "}},
    {',', {"     ", "     ", "     ", "     ", " ##  ", " ##  ", " #   "}},
    {':', {"     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     "}},
    {'-', {"     ", "     ", "     ", "#####", "     ", "     ", "     "}},
    {'+', {"     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     "}},
    {'/', {"    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "}},
    {'(', {"  ## ", " #   ", " #   ", " #   ", " #   ", " #   ", "  ## "}},
    {')', {" ##  ", "   # ", "   # ", "   # ", "   # ", "   # ", " ##  "}},
    {'[', {" ### ", " #   ", " #   ", " #   ", " #   ", " #   ", " ### "}},
    {']', {" ### ", "   # ", "   # ", "   # ", "   # ", "   # ", " ### "}},
    {'%', {"##   ", "##  #", "   # ", "  #  ", " #   ", "#  ##", "   ##"}},
    {'@', {" ### ", "#   #", "# ###", "# # #", "# ## ", "#    ", " ### "}},
    {'=', {"     ", "     ", "#####", "     ", "#####", "     ", "     "}},
    {'?', {" ### ", "#   #", "    #", "   # ", "  #  ", "     ", "  #  "}},
    {'!', {"  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "     ", "  #  "}},
    {'_', {"     ", "     ", "     ", "     ", "     ", "     ", "#####"}},
};

char normalize(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

const char* const* rows_for(char c) {
    const char wanted = normalize(c);
    for (const Glyph& g : kFont)
        if (g.code == wanted) return g.rows;
    return kFallbackRows;
}

size_t longest_line(const std::vector<std::string>& lines) {
    size_t longest = 0;
    for (const std::string& line : lines) longest = std::max(longest, line.size());
    return longest;
}

// Alpha-blend `src` (straight alpha) over one RGBA8 pixel. The presented image is opaque, so the
// destination alpha is left at whatever it was rather than being composited into.
void blend(uint8_t* dst, const uint8_t src[4]) {
    const unsigned a = src[3];
    if (a == 0) return;
    if (a == 255) { dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; return; }
    for (int c = 0; c < 3; c++)
        dst[c] = static_cast<uint8_t>((src[c] * a + dst[c] * (255u - a) + 127u) / 255u);
}

} // namespace

bool overlay_font_rows_are_well_formed(char* bad_glyph) {
    auto rows_ok = [](const char* const* rows) {
        for (int y = 0; y < kGlyphHeight; y++)
            if (!rows[y] || std::strlen(rows[y]) != static_cast<size_t>(kGlyphWidth)) return false;
        return true;
    };
    for (const Glyph& g : kFont)
        if (!rows_ok(g.rows)) {
            if (bad_glyph) *bad_glyph = g.code;
            return false;
        }
    if (!rows_ok(kFallbackRows)) {
        if (bad_glyph) *bad_glyph = '?';
        return false;
    }
    return true;
}

bool overlay_font_has_glyph(char c) {
    const char wanted = normalize(c);
    for (const Glyph& g : kFont)
        if (g.code == wanted) return true;
    return false;
}

int overlay_scale_for_width(uint32_t width) {
    return std::max(1, static_cast<int>(width / 960u));
}

OverlayBox overlay_box(const std::vector<std::string>& lines, const OverlayStyle& style) {
    OverlayBox box;
    if (lines.empty() || style.scale <= 0) return box;
    const int scale = style.scale;
    const int text_w = static_cast<int>(longest_line(lines)) * kGlyphAdvance * scale;
    const int line_h = (kGlyphHeight + style.line_spacing) * scale;
    const int text_h = static_cast<int>(lines.size()) * line_h - style.line_spacing * scale;
    box.x = style.margin;
    box.y = style.margin;
    box.width = text_w + style.padding * 2;
    box.height = text_h + style.padding * 2;
    return box;
}

OverlayBox draw_overlay_text(std::vector<uint8_t>& rgba, uint32_t width, uint32_t height,
                             const std::vector<std::string>& lines, const OverlayStyle& style) {
    const OverlayBox box = overlay_box(lines, style);
    if (box.width == 0 || box.height == 0) return {};
    if (rgba.size() != static_cast<size_t>(width) * height * 4) return {};
    // An image too small to hold the annotation is left exactly as it was. Half an overlay is
    // neither legible nor honest, and a caller that gets an unchanged frame can say so.
    //
    // Both ends are checked. The glyph loop below clamps in both directions, but the background
    // fill does not -- it indexes `rgba[(y * width + x) * 4]` directly -- so a negative origin (a
    // negative `style.margin`) would write before the buffer. No caller passes one today; the
    // function's contract says it clips, and a one-sided guard does not.
    if (box.x < 0 || box.y < 0 ||
        box.x + box.width > static_cast<int>(width) ||
        box.y + box.height > static_cast<int>(height))
        return {};

    for (int y = box.y; y < box.y + box.height; y++)
        for (int x = box.x; x < box.x + box.width; x++)
            blend(&rgba[(static_cast<size_t>(y) * width + x) * 4], style.background);

    const int scale = style.scale;
    const int line_h = (kGlyphHeight + style.line_spacing) * scale;
    for (size_t li = 0; li < lines.size(); li++) {
        const int origin_y = box.y + style.padding + static_cast<int>(li) * line_h;
        for (size_t ci = 0; ci < lines[li].size(); ci++) {
            const char* const* rows = rows_for(lines[li][ci]);
            const int origin_x = box.x + style.padding + static_cast<int>(ci) * kGlyphAdvance * scale;
            for (int gy = 0; gy < kGlyphHeight; gy++) {
                for (int gx = 0; gx < kGlyphWidth; gx++) {
                    if (rows[gy][gx] != '#') continue;
                    for (int sy = 0; sy < scale; sy++) {
                        const int py = origin_y + gy * scale + sy;
                        if (py < 0 || py >= static_cast<int>(height)) continue;
                        for (int sx = 0; sx < scale; sx++) {
                            const int px = origin_x + gx * scale + sx;
                            if (px < 0 || px >= static_cast<int>(width)) continue;
                            blend(&rgba[(static_cast<size_t>(py) * width + px) * 4], style.foreground);
                        }
                    }
                }
            }
        }
    }
    return box;
}

} // namespace prosper::screenshot
