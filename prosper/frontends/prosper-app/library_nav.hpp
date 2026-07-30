#pragma once
// library_nav.hpp — moving the selection around the library grid (#1471).
//
// Pure and free of ImGui, SDL and Vulkan, so the navigation rules are unit-tested in the default core
// build rather than only by someone pressing arrow keys — the same regression seam as game_path.hpp and
// window_controls.hpp. The drawing code owns pixels; this owns "which entry is selected".

#include <algorithm>

namespace prosper::frontend {

enum class LibraryNavKey {
    none,
    left, right, up, down,
    home, end,
    page_up, page_down,
};

// Where the selection lands after `key`.
//
// `count` is the number of titles, `columns` how many fit per row, `rows_per_page` how many rows are
// visible. Everything is clamped into [0, count) — the selection can never leave the list, so callers
// never have to bounds-check what they index with.
//
// Left/right walk the FLAT list rather than stopping at a row edge, so holding one of them sweeps the
// whole library the way a list would; up/down move by a row. Neither wraps around the ends: wrapping
// makes it easy to overshoot past the last title and land back at the first without noticing.
//
// Down from a partial last row clamps to the final entry instead of refusing to move, which is what a
// person means by "down" when the row below is ragged.
constexpr int library_nav_apply(LibraryNavKey key, int selected, int count, int columns,
                                int rows_per_page) {
    if (count <= 0) return 0;
    if (columns < 1) columns = 1;
    if (rows_per_page < 1) rows_per_page = 1;
    const int last = count - 1;
    int at = std::clamp(selected, 0, last);   // tolerate a stale selection after the list changed

    switch (key) {
    case LibraryNavKey::left:      return std::max(0, at - 1);
    case LibraryNavKey::right:     return std::min(last, at + 1);
    case LibraryNavKey::up:        return at - columns < 0 ? at : at - columns;
    case LibraryNavKey::down:      return at + columns > last ? (at == last ? at : last)
                                                             : at + columns;
    case LibraryNavKey::home:      return 0;
    case LibraryNavKey::end:       return last;
    case LibraryNavKey::page_up:   return std::max(0, at - columns * rows_per_page);
    case LibraryNavKey::page_down: return std::min(last, at + columns * rows_per_page);
    case LibraryNavKey::none:      break;
    }
    return at;
}

// How many columns fit, given the space available and what one cell costs. At least one, so a window
// narrower than a single cell still shows the library one entry at a time rather than dividing by zero
// or displaying nothing.
constexpr int library_columns_for_width(float available_width, float cell_width) {
    if (!(cell_width > 0.0f)) return 1;
    const int fits = static_cast<int>(available_width / cell_width);
    return fits < 1 ? 1 : fits;
}

// Keep the selected row on screen: the first row to draw so `selected` is visible, given the current
// scroll. Returns a row index, never negative.
constexpr int library_scroll_row_for(int selected, int columns, int rows_per_page, int first_row) {
    if (columns < 1) columns = 1;
    if (rows_per_page < 1) rows_per_page = 1;
    if (selected < 0) selected = 0;
    if (first_row < 0) first_row = 0;
    const int row = selected / columns;
    if (row < first_row) return row;                          // scrolled above the view
    if (row >= first_row + rows_per_page) return row - rows_per_page + 1;  // below it
    return first_row;                                          // already visible: do not move
}

} // namespace prosper::frontend
