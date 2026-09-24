#pragma once

#include <cstddef>
#include <cstdint>
#include <charconv>
#include <limits>
#include <span>
#include <system_error>

namespace prosper::gpu {

// A diagnostic predicate over one immutable RGBA8 publication. The rectangles and thresholds are
// supplied by a capture request; neither a game's identity nor a target submit ordinal is inferred
// from these pixels. This only says whether the same publication shows the requested UI phase.
struct MenuFrameRoi {
    uint32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;

    constexpr bool inside(uint32_t width, uint32_t height) const {
        return x0 < x1 && y0 < y1 && x1 <= width && y1 <= height;
    }
};

struct MenuFrameGateSpec {
    uint32_t width = 0, height = 0;
    MenuFrameRoi menu;
    MenuFrameRoi logo;
    uint32_t center_x = 0, center_y = 0;
    uint8_t bright_above = 170;
    uint8_t center_at_most = 48;
    uint32_t menu_bright_min = 1000;
    uint32_t logo_bright_min = 1000;

    constexpr bool valid() const {
        return width && height && menu.inside(width, height) && logo.inside(width, height) &&
               center_x < width && center_y < height && menu_bright_min && logo_bright_min;
    }
};

// WxH:menu-x0,y0,x1,y1:logo-x0,y0,x1,y1:center-x,y:bright,dark,menu-min,logo-min.
// The complete numeric form keeps the policy reusable and makes each ROI assumption visible in the
// run command. A malformed request refuses instead of turning the gate off.
inline bool parse_menu_frame_gate_spec(const char* text, MenuFrameGateSpec& out) {
    out = {};
    if (!text || !*text) return false;
    const char* cursor = text;
    // Sixteen numeric values, including two uint8 thresholds, and their separators fit below this
    // bound. Refuse an oversized environment value before parsing or scanning it further.
    constexpr size_t kMaxSpecLength = 256;
    size_t length = 0;
    while (length <= kMaxSpecLength && text[length]) ++length;
    if (length > kMaxSpecLength) return false;
    const char* end = text + length;
    auto read = [&](uint32_t& value, char delimiter) {
        const auto parsed = std::from_chars(cursor, end, value);
        if (parsed.ec != std::errc{} || parsed.ptr == cursor ||
            parsed.ptr == end || *parsed.ptr != delimiter)
            return false;
        cursor = parsed.ptr + 1;
        return true;
    };
    auto read_last = [&](uint32_t& value) {
        const auto parsed = std::from_chars(cursor, end, value);
        return parsed.ec == std::errc{} && parsed.ptr != cursor && parsed.ptr == end;
    };
    uint32_t width = 0, height = 0, mx0 = 0, my0 = 0, mx1 = 0, my1 = 0;
    uint32_t lx0 = 0, ly0 = 0, lx1 = 0, ly1 = 0, cx = 0, cy = 0;
    uint32_t bright = 0, dark = 0, menu_min = 0, logo_min = 0;
    if (!read(width, 'x') || !read(height, ':') ||
        !read(mx0, ',') || !read(my0, ',') ||
        !read(mx1, ',') || !read(my1, ':') ||
        !read(lx0, ',') || !read(ly0, ',') ||
        !read(lx1, ',') || !read(ly1, ':') ||
        !read(cx, ',') || !read(cy, ':') ||
        !read(bright, ',') || !read(dark, ',') ||
        !read(menu_min, ',') || !read_last(logo_min) ||
        bright > 255 || dark > 255)
        return false;
    MenuFrameGateSpec parsed{.width = width, .height = height,
                             .menu = {mx0, my0, mx1, my1},
                             .logo = {lx0, ly0, lx1, ly1},
                             .center_x = cx, .center_y = cy,
                             .bright_above = static_cast<uint8_t>(bright),
                             .center_at_most = static_cast<uint8_t>(dark),
                             .menu_bright_min = menu_min, .logo_bright_min = logo_min};
    if (!parsed.valid()) return false;
    out = parsed;
    return true;
}

enum class MenuFrameGateVerdict : uint8_t {
    InvalidImage,
    NotMenu,
    MenuWithLogo,
    MenuWithoutLogo,
};

struct MenuFrameGateResult {
    MenuFrameGateVerdict verdict = MenuFrameGateVerdict::InvalidImage;
    uint64_t menu_bright = 0;
    uint64_t logo_bright = 0;
    bool center_dark = false;
};

inline MenuFrameGateResult classify_menu_frame(std::span<const uint8_t> rgba,
                                                uint32_t width, uint32_t height,
                                                const MenuFrameGateSpec& spec) {
    MenuFrameGateResult result;
    if (!spec.valid() || width != spec.width || height != spec.height ||
        width > std::numeric_limits<size_t>::max() / height / 4 ||
        rgba.size() != static_cast<size_t>(width) * height * 4)
        return result;

    auto pixel = [&](uint32_t x, uint32_t y) {
        return rgba.data() + (static_cast<size_t>(y) * width + x) * 4;
    };
    const uint8_t* center = pixel(spec.center_x, spec.center_y);
    result.center_dark = center[0] <= spec.center_at_most &&
                         center[1] <= spec.center_at_most &&
                         center[2] <= spec.center_at_most;
    if (!result.center_dark) {
        result.verdict = MenuFrameGateVerdict::NotMenu;
        return result;
    }

    auto count_bright = [&](MenuFrameRoi roi) {
        uint64_t count = 0;
        for (uint32_t y = roi.y0; y < roi.y1; ++y)
            for (uint32_t x = roi.x0; x < roi.x1; ++x) {
                const uint8_t* p = pixel(x, y);
                count += p[0] > spec.bright_above && p[1] > spec.bright_above &&
                         p[2] > spec.bright_above;
            }
        return count;
    };
    result.menu_bright = count_bright(spec.menu);
    if (result.menu_bright <= spec.menu_bright_min) {
        result.verdict = MenuFrameGateVerdict::NotMenu;
        return result;
    }
    result.logo_bright = count_bright(spec.logo);
    result.verdict = result.logo_bright < spec.logo_bright_min
        ? MenuFrameGateVerdict::MenuWithoutLogo
        : MenuFrameGateVerdict::MenuWithLogo;
    return result;
}

} // namespace prosper::gpu
