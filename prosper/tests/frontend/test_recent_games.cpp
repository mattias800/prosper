// Successful library launches must refresh File > Recent games while the library stays open
// (#5117). Persistence alone left the live menu stale until restart. These CPU controls exercise
// the same update boundary as the frontend, including a failed save, without SDL or a game dump.
#include "app_config.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using prosper::frontend::AppConfig;
using prosper::frontend::record_recent_game;

TEST(RecentGames, SuccessfulLaunchPublishesTheSavedHistory) {
    AppConfig cfg;
    cfg.recent_games = {"/games/old-app0"};
    cfg.games_dirs = {"/games"};
    std::vector<std::string> saved, live;
    EXPECT_TRUE(record_recent_game(
        cfg, "/games/new-app0/",
        [&](const AppConfig& updated) {
            saved = updated.recent_games;
            EXPECT_EQ(updated.games_dirs, cfg.games_dirs);
            return true;
        },
        [&](const AppConfig& updated) { live = updated.recent_games; }));
    EXPECT_EQ(saved, (std::vector<std::string>{"/games/new-app0", "/games/old-app0"}));
    EXPECT_EQ(live, saved) << "saving a launch must also update the running menu";
}

TEST(RecentGames, FailedSaveStillPublishesTheActualLaunch) {
    AppConfig cfg;
    cfg.recent_games = {"/games/old-app0"};
    std::vector<std::string> live;
    EXPECT_FALSE(record_recent_game(
        cfg, "/games/new-app0", [](const AppConfig&) { return false; },
        [&](const AppConfig& updated) { live = updated.recent_games; }));
    EXPECT_EQ(live, (std::vector<std::string>{"/games/new-app0", "/games/old-app0"}))
        << "a persistence error must not hide a successfully started game from this session";
}

TEST(RecentGames, ReopenedTitleMovesToTheFrontWithoutDuplicates) {
    AppConfig cfg;
    cfg.recent_games = {"/games/new-app0", "/games/old-app0/"};
    std::vector<std::string> live;
    EXPECT_TRUE(record_recent_game(
        cfg, "/games/old-app0", [](const AppConfig&) { return true; },
        [&](const AppConfig& updated) { live = updated.recent_games; }));
    EXPECT_EQ(live, (std::vector<std::string>{"/games/old-app0", "/games/new-app0"}));
}
