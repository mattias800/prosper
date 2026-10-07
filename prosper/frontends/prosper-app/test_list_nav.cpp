// test_list_nav — the list keyboard gate: typing in the search box must never move the
// selection or boot a game. No SDL, no ImGui, no window: the gate is pure, so the regression
// arm types a space with text active and asserts the list takes nothing.
#include "list_nav.hpp"

#include <gtest/gtest.h>

using prosper::frontend::ListNavMove;
using prosper::frontend::list_nav_move;
using prosper::frontend::list_nav_open;

TEST(ListNavMove, MapsEachKey) {
    EXPECT_EQ(list_nav_move(true, false, false, false, false), ListNavMove::up);
    EXPECT_EQ(list_nav_move(false, true, false, false, false), ListNavMove::down);
    EXPECT_EQ(list_nav_move(false, false, true, false, false), ListNavMove::home);
    EXPECT_EQ(list_nav_move(false, false, false, true, false), ListNavMove::end);
    EXPECT_EQ(list_nav_move(false, false, false, false, false), ListNavMove::none);
}

TEST(ListNavMove, FirstKeyWins) {
    EXPECT_EQ(list_nav_move(true, true, false, false, false), ListNavMove::up);
}

TEST(ListNavMove, TextActiveSuppressesMovement) {
    // The regression: a space typed in the search box booted the first matching game.
    EXPECT_EQ(list_nav_move(false, false, false, false, true), ListNavMove::none);
    EXPECT_EQ(list_nav_move(true, true, true, true, true), ListNavMove::none);
}

TEST(ListNavOpen, EnterOpensWithoutText) {
    EXPECT_TRUE(list_nav_open(true, false));
    EXPECT_FALSE(list_nav_open(false, false));
}

TEST(ListNavOpen, TextActiveSuppressesOpen) {
    EXPECT_FALSE(list_nav_open(true, true));
}
