// test_game_library — the library scan, param.json metadata, and settings precedence (#1471). No SDL,
// no window, no real dump: the filesystem is a fixed set of names and contents, so every branch runs
// in ordinary CI.
#include "app_config.hpp"
#include "game_library.hpp"

#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

using prosper::frontend::AppConfig;
using prosper::frontend::GameEntry;
using prosper::frontend::GameLibraryIo;
using prosper::frontend::GamePathProbe;
using prosper::frontend::parse_app_config;
using prosper::frontend::parse_param_title_id;
using prosper::frontend::parse_param_title_name;
using prosper::frontend::path_basename;
using prosper::frontend::resolve_games_dir;
using prosper::frontend::scan_game_library;
using prosper::frontend::serialize_app_config;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

// A fake games directory:
//   /games/PPSA24651-app0   full metadata + icon        -> "The Messenger"
//   /games/PPSA13579-app0   metadata, NO icon           -> "Blasphemous 2"
//   /games/PPSA00001-app0   eboot only, no param.json   -> falls back to the directory name
//   /games/PPSA00002-app0   param.json with no titleName-> falls back to the directory name
//   /games/notes            not a title                 -> skipped
//   /games/loose.txt        a file                      -> skipped
static const std::set<std::string> kDirs = {
    "/games",
    "/games/PPSA24651-app0", "/games/PPSA24651-app0/sce_sys",
    "/games/PPSA13579-app0", "/games/PPSA13579-app0/sce_sys",
    "/games/PPSA00001-app0",
    "/games/PPSA00002-app0", "/games/PPSA00002-app0/sce_sys",
    "/games/notes",
    "/empty",
};
static const std::set<std::string> kFiles = {
    "/games/PPSA24651-app0/eboot.bin",
    "/games/PPSA24651-app0/sce_sys/param.json",
    "/games/PPSA24651-app0/sce_sys/icon0.png",
    "/games/PPSA13579-app0/sce_sys/param.json",
    "/games/PPSA00001-app0/eboot.bin",
    "/games/PPSA00002-app0/sce_sys/param.json",
    "/games/notes/readme.txt",
    "/games/loose.txt",
};
static const std::map<std::string, std::vector<std::string>> kChildren = {
    {"/games", {"PPSA13579-app0", "notes", "PPSA24651-app0", "loose.txt", "PPSA00001-app0",
                "PPSA00002-app0"}},   // deliberately unsorted
    {"/empty", {}},
};
// Real shape: defaultLanguage plus several localized entries, the wanted one NOT first.
static const std::string kMessengerJson = R"({
    "titleId": "PPSA24651",
    "localizedParameters": {
        "de-DE": { "titleName": "Der Bote" },
        "en-US": { "titleName": "The Messenger" },
        "defaultLanguage": "en-US"
    }
})";
static const std::string kBlasphemousJson = R"({
    "titleId": "PPSA13579",
    "localizedParameters": {
        "defaultLanguage": "en-US",
        "en-US": { "titleName": "Blasphemous 2" }
    }
})";
// The shape that exposed the language-selection bug (#1471), copied from PPSA15319's real param.json:
// a language object BEFORE defaultLanguage, defaultLanguage naming a DIFFERENT language, and that
// language's object AFTER it. The fallback-to-first-titleName path returns the wrong name here, so this
// case cannot pass by accident — it fails unless defaultLanguage's own object is genuinely resolved.
static const std::string kPluckyJson = R"({
    "titleId": "PPSA15319",
    "localizedParameters": {
        "de-DE": { "titleName": "DER KUEHNE KNAPPE" },
        "defaultLanguage": "en-US",
        "en-US": { "titleName": "The Plucky Squire" },
        "fr-FR": { "titleName": "LE VAILLANT PETIT PAGE" }
    }
})";
static const std::string kNoNameJson = R"({ "titleId": "PPSA00002" })";

static GamePathProbe fake_probe() {
    GamePathProbe p;
    p.is_dir  = [](const std::string& s) { return kDirs.count(s) != 0; };
    p.is_file = [](const std::string& s) { return kFiles.count(s) != 0; };
    return p;
}
static GameLibraryIo fake_io() {
    GameLibraryIo io;
    io.list_dir = [](const std::string& d) -> std::vector<std::string> {
        auto it = kChildren.find(d);
        return it == kChildren.end() ? std::vector<std::string>{} : it->second;
    };
    io.read_file = [](const std::string& p) -> std::string {
        if (p == "/games/PPSA24651-app0/sce_sys/param.json") return kMessengerJson;
        if (p == "/games/PPSA13579-app0/sce_sys/param.json") return kBlasphemousJson;
        if (p == "/games/PPSA00002-app0/sce_sys/param.json") return kNoNameJson;
        return "";
    };
    return io;
}

int main() {
    std::printf("== test_prosper_app_game_library ==\n");

    // --- param.json readers (previously untested: this logic drove the window title) --------------
    CHECK(parse_param_title_name(kMessengerJson) == "The Messenger",
          "defaultLanguage's titleName wins over an earlier language entry");
    CHECK(parse_param_title_id(kMessengerJson) == "PPSA24651", "titleId is read");
    // NOTE: kBlasphemousJson has only one language, so it would also pass via the
    // fallback-to-first-titleName path. kPluckyJson below is the non-vacuous version of this case.
    CHECK(parse_param_title_name(kBlasphemousJson) == "Blasphemous 2",
          "a single-language dump with defaultLanguage first resolves");
    CHECK(parse_param_title_name(kPluckyJson) == "The Plucky Squire",
          "#1471: defaultLanguage's own object wins even when declared before it and after another "
          "language (the fallback would return the de-DE name here)");
    CHECK(parse_param_title_id(kPluckyJson) == "PPSA15319", "titleId read from the same dump shape");
    // The language name also appears as defaultLanguage's VALUE; matching that instead of the object
    // key is exactly how the old logic went wrong, so pin that the key is what gets found.
    CHECK(prosper::frontend::find_language_object(kPluckyJson, "en-US") >
              kPluckyJson.find("\"defaultLanguage\""),
          "the language OBJECT is found after defaultLanguage, not its value");
    CHECK(prosper::frontend::find_language_object("{\"defaultLanguage\":\"en-US\"}", "en-US") ==
              std::string::npos,
          "a language named only as a value is not mistaken for an object");
    CHECK(parse_param_title_name(kNoNameJson).empty(), "no titleName yields empty, not garbage");
    CHECK(parse_param_title_id(kNoNameJson) == "PPSA00002", "titleId is read without a titleName");
    CHECK(parse_param_title_name("").empty(), "empty json yields no name");
    CHECK(parse_param_title_id("").empty(), "empty json yields no id");
    CHECK(parse_param_title_name("{\"localizedParameters\":{\"en-US\":{\"titleName\":\"Solo\"}}}")
              == "Solo",
          "a single language with no defaultLanguage falls back to the first titleName");
    // A defaultLanguage naming a language with no object must not silently pick another language's
    // name via the fallback... it does fall back, which is the documented behaviour; pin it so the
    // choice is deliberate rather than accidental.
    CHECK(parse_param_title_name(
              "{\"localizedParameters\":{\"fr-FR\":{\"titleName\":\"Seul\"},"
              "\"defaultLanguage\":\"ja-JP\"}}") == "Seul",
          "a defaultLanguage with no matching object falls back to the first titleName");
    CHECK(parse_param_title_name("{\"titleName\": \"Quote \\\" and backslash \\\\ here\"}")
              == "Quote \" and backslash \\ here",
          "escaped quotes and backslashes are unescaped");
    CHECK(parse_param_title_name("{\"titleName\":\"\"}").empty(), "an empty titleName reads as empty");

    // --- path_basename --------------------------------------------------------------------------
    CHECK(path_basename("/games/PPSA24651-app0") == "PPSA24651-app0", "basename of a path");
    CHECK(path_basename("/games/PPSA24651-app0/") == "PPSA24651-app0", "trailing separator ignored");
    CHECK(path_basename("C:\\games\\PPSA24651-app0") == "PPSA24651-app0", "Windows separators");
    CHECK(path_basename("bare") == "bare", "a bare name is its own basename");

    // --- scan_game_library ----------------------------------------------------------------------
    const GamePathProbe probe = fake_probe();
    const GameLibraryIo io = fake_io();
    const std::vector<GameEntry> games = scan_game_library("/games", probe, io);
    CHECK(games.size() == 4, "exactly the four titles are found (non-titles and files skipped)");
    if (games.size() == 4) {
        // Display order is case-insensitive by name: Blasphemous 2, PPSA00001-app0, PPSA00002-app0,
        // The Messenger. Note this proves list_dir's order does not leak through.
        CHECK(games[0].title_name == "Blasphemous 2", "sorted by display name (1st)");
        CHECK(games[1].title_name == "PPSA00001-app0", "a title with no param.json falls back to its directory name");
        CHECK(games[2].title_name == "PPSA00002-app0", "a title with no titleName falls back to its directory name");
        CHECK(games[3].title_name == "The Messenger", "sorted by display name (last)");

        CHECK(games[3].app0_root == "/games/PPSA24651-app0", "app0 root is what boot_program wants");
        CHECK(games[3].title_id == "PPSA24651", "content id captured");
        CHECK(games[3].icon_path == "/games/PPSA24651-app0/sce_sys/icon0.png", "icon located");
        CHECK(games[0].icon_path.empty(), "a title with no icon0.png reports no icon, not a bad path");
        CHECK(games[1].title_id.empty(), "no param.json means no content id");
        CHECK(games[2].title_id == "PPSA00002", "content id survives a missing titleName");
    }

    // --- scan_game_library: what must NOT happen ------------------------------------------------
    CHECK(scan_game_library("/empty", probe, io).empty(), "a directory with no titles yields nothing");
    CHECK(scan_game_library("/does/not/exist", probe, io).empty(), "a missing directory yields nothing");
    CHECK(scan_game_library("", probe, io).empty(), "an empty path yields nothing");
    // The games dir itself is never treated as a title, even when it is one — documented behaviour, so
    // pointing this at a single app0 folder yields nothing rather than one oddly-named entry.
    CHECK(scan_game_library("/games/PPSA24651-app0", probe, io).empty(),
          "a title root used as the games dir yields nothing (children only)");
    CHECK(scan_game_library("/games/", probe, io).size() == 4, "a trailing separator is tolerated");
    const GameLibraryIo empty_io;
    CHECK(scan_game_library("/games", probe, empty_io).empty(), "an unpopulated io scans nothing");

    // --- config parsing -------------------------------------------------------------------------
    CHECK(parse_app_config("games_dir = /games").games_dir == "/games", "a setting is read");
    CHECK(parse_app_config("games_dir=/games").games_dir == "/games", "spaces around = are optional");
    CHECK(parse_app_config("  games_dir  =  /games  ").games_dir == "/games", "surrounding space trimmed");
    CHECK(parse_app_config("# comment\n\ngames_dir = /games\n").games_dir == "/games",
          "comments and blank lines are ignored");
    CHECK(parse_app_config("games_dir = /a\ngames_dir = /b").games_dir == "/b", "a later duplicate wins");
    CHECK(parse_app_config("future_key = 1\ngames_dir = /games").games_dir == "/games",
          "an unknown key does not break parsing");
    CHECK(parse_app_config("games_dir = /my games/ps5 = final # 1").games_dir == "/my games/ps5 = final # 1",
          "the value is literal after the first = (spaces, =, and # all survive)");
    CHECK(parse_app_config("").games_dir.empty(), "an empty file sets nothing");
    CHECK(parse_app_config("nonsense").games_dir.empty(), "a line with no = sets nothing");
    CHECK(parse_app_config("games_dir = /games\r\n").games_dir == "/games", "CRLF is tolerated");

    // --- config round-trip ----------------------------------------------------------------------
    AppConfig cfg;
    cfg.games_dir = "/my games/ps5";
    CHECK(parse_app_config(serialize_app_config(cfg)).games_dir == "/my games/ps5",
          "serialize -> parse round-trips a path with spaces");
    CHECK(parse_app_config(serialize_app_config(AppConfig{})).games_dir.empty(),
          "an unset config round-trips as unset");

    // --- precedence -----------------------------------------------------------------------------
    AppConfig from_file; from_file.games_dir = "/from-file";
    CHECK(resolve_games_dir("/from-flag", "/from-env", from_file) == "/from-flag",
          "--games-dir wins over everything");
    CHECK(resolve_games_dir("", "/from-env", from_file) == "/from-env",
          "the environment wins over the persisted setting");
    CHECK(resolve_games_dir("", "", from_file) == "/from-file",
          "the persisted setting applies when nothing else does");
    CHECK(resolve_games_dir("", "", AppConfig{}).empty(), "with no source there is no games dir");
    CHECK(resolve_games_dir("/from-flag", "", AppConfig{}) == "/from-flag",
          "the flag alone is enough");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
