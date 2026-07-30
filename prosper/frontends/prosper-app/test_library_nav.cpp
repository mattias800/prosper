// test_library_nav — grid selection movement for the library view (#1471). No ImGui, no SDL, no
// window: the rules are arithmetic, so they run in the default core build.
#include "library_nav.hpp"

#include <cstdio>

using prosper::frontend::LibraryNavKey;
using prosper::frontend::library_columns_for_width;
using prosper::frontend::library_nav_apply;
using prosper::frontend::library_scroll_row_for;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

// 7 titles in a 3-wide grid, 2 rows visible:
//   0 1 2
//   3 4 5
//   6
static int nav(LibraryNavKey k, int at) { return library_nav_apply(k, at, 7, 3, 2); }

int main() {
    std::printf("== test_prosper_app_library_nav ==\n");

    // --- left/right walk the flat list ----------------------------------------------------------
    CHECK(nav(LibraryNavKey::right, 0) == 1, "right advances one");
    CHECK(nav(LibraryNavKey::left, 1) == 0, "left goes back one");
    CHECK(nav(LibraryNavKey::right, 2) == 3, "right crosses a row boundary rather than stopping");
    CHECK(nav(LibraryNavKey::left, 3) == 2, "left crosses back");
    // Neither end wraps: overshooting past the last title back to the first is easy to miss.
    CHECK(nav(LibraryNavKey::left, 0) == 0, "left at the start does not wrap to the end");
    CHECK(nav(LibraryNavKey::right, 6) == 6, "right at the end does not wrap to the start");

    // --- up/down move by a row ------------------------------------------------------------------
    CHECK(nav(LibraryNavKey::down, 0) == 3, "down moves a whole row");
    CHECK(nav(LibraryNavKey::up, 3) == 0, "up moves a whole row back");
    CHECK(nav(LibraryNavKey::up, 1) == 1, "up on the first row stays put");
    // The last row is ragged (just entry 6). Down from 4 has no entry directly below, and refusing to
    // move would feel broken, so it clamps to the last title.
    CHECK(nav(LibraryNavKey::down, 4) == 6, "down onto a ragged last row clamps to the final entry");
    CHECK(nav(LibraryNavKey::down, 6) == 6, "down on the last entry stays put");
    CHECK(nav(LibraryNavKey::down, 3) == 6, "down from directly above the last entry reaches it");

    // --- home/end/page --------------------------------------------------------------------------
    CHECK(nav(LibraryNavKey::home, 5) == 0, "home selects the first title");
    CHECK(nav(LibraryNavKey::end, 0) == 6, "end selects the last title");
    CHECK(nav(LibraryNavKey::page_down, 0) == 6, "page down moves a page and clamps");
    CHECK(nav(LibraryNavKey::page_up, 6) == 0, "page up moves a page and clamps");
    CHECK(nav(LibraryNavKey::none, 4) == 4, "no key does not move the selection");

    // --- degenerate inputs must not produce an out-of-range index --------------------------------
    CHECK(library_nav_apply(LibraryNavKey::down, 0, 0, 3, 2) == 0, "an empty library selects 0");
    CHECK(library_nav_apply(LibraryNavKey::right, 0, 0, 3, 2) == 0, "no movement in an empty library");
    CHECK(library_nav_apply(LibraryNavKey::down, 2, 5, 0, 2) == 3,
          "zero columns is treated as one rather than dividing by zero");
    CHECK(library_nav_apply(LibraryNavKey::page_down, 0, 5, 3, 0) == 3,
          "zero rows per page is treated as one");
    // A selection left over from a longer list must be pulled back in range, not indexed with.
    CHECK(library_nav_apply(LibraryNavKey::none, 99, 7, 3, 2) == 6,
          "a stale out-of-range selection clamps to the last entry");
    CHECK(library_nav_apply(LibraryNavKey::none, -5, 7, 3, 2) == 0,
          "a negative selection clamps to the first entry");

    // --- column fitting --------------------------------------------------------------------------
    CHECK(library_columns_for_width(640.0f, 200.0f) == 3, "columns divide the available width");
    CHECK(library_columns_for_width(199.0f, 200.0f) == 1,
          "a window narrower than one cell still shows one column");
    CHECK(library_columns_for_width(640.0f, 0.0f) == 1, "a zero cell width does not divide by zero");
    CHECK(library_columns_for_width(0.0f, 200.0f) == 1, "a zero-width window still reports one column");

    // --- scrolling keeps the selection visible ----------------------------------------------------
    CHECK(library_scroll_row_for(0, 3, 2, 0) == 0, "already visible: do not scroll");
    CHECK(library_scroll_row_for(4, 3, 2, 0) == 0, "row 1 is visible with 2 rows shown");
    CHECK(library_scroll_row_for(6, 3, 2, 0) == 1, "selecting below the view scrolls down by one row");
    CHECK(library_scroll_row_for(0, 3, 2, 1) == 0, "selecting above the view scrolls back up");
    CHECK(library_scroll_row_for(0, 3, 2, -4) == 0, "a negative scroll is treated as the top");
    CHECK(library_scroll_row_for(-1, 3, 2, 0) == 0, "a negative selection does not scroll negative");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
