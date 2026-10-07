// test_game_library — the library scan, param.json metadata, and settings precedence (#1471). No SDL,
// no window, no real dump: the filesystem is a fixed set of names and contents, so every branch runs
// in ordinary CI.
#include "app_config.hpp"
#include "game_library.hpp"

#include <algorithm>
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
using prosper::frontend::game_entry_matches_filter;
using prosper::frontend::parse_param_content_version;
using prosper::frontend::parse_param_region;
using prosper::frontend::parse_param_title_id;
using prosper::frontend::parse_param_title_name;
using prosper::frontend::path_basename;
using prosper::frontend::HostPolicyInputs;
using prosper::frontend::forget_games_dir;
using prosper::frontend::note_games_dir;
using prosper::frontend::note_recent_game;
using prosper::frontend::parse_volume_percent;
using prosper::frontend::resolve_host_policy;
using prosper::frontend::resolve_games_dirs;
using prosper::frontend::scan_game_libraries;
using prosper::frontend::scan_game_library;
using prosper::frontend::serialize_app_config;
using prosper::frontend::guest_args_for;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

// A fake games directory:
//   /games/PPSA24651-app0   full metadata + icon        -> "The Messenger"
//   /games/PPSA13579-app0   metadata, NO icon           -> "Blasphemous 2"
//   /games/PPSA00001-app0   eboot only, no param.json   -> falls back to the directory name
//   /games/PPSA00002-app0   param.json with no titleName-> falls back to the directory name
//   /games/PPSA00003-app0   titleName "astro bot"       -> pins the case-insensitive sort
//   /games/PPSA00004-app0   titleName "Same Name"       -> pins the content-id tie-break
//   /games/PPSA00005-app0   titleName "Same Name"       -> ...its partner
//   /games/notes            not a title                 -> skipped
//   /games/loose.txt        a file                      -> skipped
//   /drive2/middle-app0  titleName "Middle Title"    -> a SECOND games folder's only title; its
//                                                          name sorts between /games' titles
static const std::set<std::string> kDirs = {
    "/games",
    "/games/PPSA24651-app0",
    "/games/PPSA24651-app0/sce_sys",
    "/games/PPSA13579-app0",
    "/games/PPSA13579-app0/sce_sys",
    "/games/PPSA00001-app0",
    "/games/PPSA00002-app0",
    "/games/PPSA00002-app0/sce_sys",
    "/games/PPSA00003-app0",
    "/games/PPSA00003-app0/sce_sys",
    "/games/PPSA00004-app0",
    "/games/PPSA00004-app0/sce_sys",
    "/games/PPSA00005-app0",
    "/games/PPSA00005-app0/sce_sys",
    "/games/notes",
    "/empty",
    "/drive2",
    "/drive2/middle-app0",
    "/drive2/middle-app0/sce_sys",
};
static const std::set<std::string> kFiles = {
    "/games/PPSA24651-app0/eboot.bin",
    "/games/PPSA24651-app0/sce_sys/param.json",
    "/games/PPSA24651-app0/sce_sys/icon0.png",
    "/games/PPSA13579-app0/sce_sys/param.json",
    "/games/PPSA00001-app0/eboot.bin",
    "/games/PPSA00002-app0/sce_sys/param.json",
    "/games/PPSA00003-app0/sce_sys/param.json",
    "/games/PPSA00004-app0/sce_sys/param.json",
    "/games/PPSA00005-app0/sce_sys/param.json",
    "/games/notes/readme.txt",
    "/games/loose.txt",
    "/drive2/middle-app0/sce_sys/param.json",
};
static const std::map<std::string, std::vector<std::string>> kChildren = {
    {"/games",
     {"PPSA13579-app0", "notes", "PPSA24651-app0", "loose.txt", "PPSA00001-app0", "PPSA00002-app0",
      "PPSA00005-app0", "PPSA00003-app0", "PPSA00004-app0"}},
    // ^ deliberately unsorted, and PPSA00005 deliberately before the two it ties with
    {"/empty", {}},
    {"/drive2", {"middle-app0"}},
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
// Lowercase display name: sorts FIRST case-insensitively but LAST by raw bytes ('a' = 0x61 > 'T'),
// so this is what pins the fold in game_entry_display_less.
static const std::string kLowerJson = R"({ "titleId": "PPSA00003", "titleName": "astro bot" })";
// Two titles sharing a display name, to pin the content-id tie-break.
static const std::string kTieAJson = R"({ "titleId": "PPSA00004", "titleName": "Same Name" })";
static const std::string kTieBJson = R"({ "titleId": "PPSA00005", "titleName": "Same Name" })";
// The second games folder's title. Its name sorts after "Blasphemous 2" and before the
// directory-named fallbacks, so a merge that only concatenates folders (no cross-folder sort) puts
// it in the wrong place. No titleId, so the fixture names no real title.
static const char kDrive2Json[] = R"({ "titleName": "Middle Title" })";
// defaultLanguage's own object carries NO titleName. This distinguishes a BOUNDED lookup from an
// unbounded one: searching forward from the en-US key with no limit finds fr-FR's name ("After"),
// whereas bounding the lookup to en-US's own object finds nothing there and falls back to the first
// name in localizedParameters ("Before"). The bounded result is the deterministic one — document order
// rather than "whichever language happens to follow the default key".
static const std::string kMissingNameJson = R"({
    "titleId": "PPSA00006",
    "localizedParameters": {
        "de-DE": { "titleName": "Before" },
        "defaultLanguage": "en-US",
        "en-US": { "someOtherField": 1 },
        "fr-FR": { "titleName": "After" }
    }
})";

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
        if (p == "/games/PPSA00003-app0/sce_sys/param.json") return kLowerJson;
        if (p == "/games/PPSA00004-app0/sce_sys/param.json") return kTieAJson;
        if (p == "/games/PPSA00005-app0/sce_sys/param.json") return kTieBJson;
        if (p == "/drive2/middle-app0/sce_sys/param.json") return kDrive2Json;
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
    // --- version + region: the list columns ------------------------------------------------
    CHECK(parse_param_content_version(
              "{\"contentVersion\":\"01.000.006\",\"titleId\":\"TEST19990\"}") == "01.000.006",
          "contentVersion is read");
    CHECK(parse_param_content_version("").empty(), "empty json yields no version");
    CHECK(parse_param_content_version("{\"titleId\":\"TEST19990\"}").empty(),
          "a dump without contentVersion yields empty, not garbage");
    CHECK(parse_param_region(
              "{\"contentId\":\"EP0700-TEST19990_00-SOME-GAME-00\"}") == "EP",
          "the region is the two letters heading the contentId prefix");
    CHECK(parse_param_region("{\"contentId\":\"nodashes\"}").empty(),
          "a dashless contentId yields no region rather than the whole string");
    CHECK(parse_param_region("").empty(), "empty json yields no region");
    // --- search filter ---------------------------------------------------------------------
    {
        GameEntry e;
        e.app0_root = "/games/TEST24651-app0";
        e.title_id = "TEST24651";
        e.title_name = "The Messenger";
        CHECK(game_entry_matches_filter(e, ""), "an empty box matches everything");
        CHECK(game_entry_matches_filter(e, "messenger"), "name matches case-insensitively");
        CHECK(game_entry_matches_filter(e, "TEST24651"), "title id matches");
        CHECK(game_entry_matches_filter(e, "test24651"), "title id matches case-insensitively");
        CHECK(game_entry_matches_filter(e, "24651-app0"), "the path matches too");
        CHECK(!game_entry_matches_filter(e, "zelda"), "an unrelated needle matches nothing");
    }
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
          "escaped quotes and backslashes are unescaped (\\n etc. are NOT interpreted)");
    CHECK(parse_param_title_name("{\"titleName\":\"\"}").empty(), "an empty titleName reads as empty");
    // Bounded to defaultLanguage's OWN object. A dump whose default language carries no titleName is
    // malformed; showing some real name still beats showing a content-id directory, so the documented
    // fallback applies — but it must be the deterministic document-order one, not whatever language
    // happens to sit after the default key. An unbounded search would answer "After" here.
    CHECK(parse_param_title_name(kMissingNameJson) == "Before",
          "#1471: an empty defaultLanguage object falls back in document order, not to the next "
          "language after it");
    // A language-keyed object OUTSIDE localizedParameters must not be mistaken for a title entry.
    CHECK(parse_param_title_name(
              "{\"share\":{\"en-US\":{\"titleName\":\"Share Blurb\"}},"
              "\"localizedParameters\":{\"defaultLanguage\":\"en-US\","
              "\"en-US\":{\"titleName\":\"Real Title\"}}}") == "Real Title",
          "#1471: lookups are anchored inside localizedParameters");

    // --- path_basename --------------------------------------------------------------------------
    CHECK(path_basename("/games/PPSA24651-app0") == "PPSA24651-app0", "basename of a path");
    CHECK(path_basename("/games/PPSA24651-app0/") == "PPSA24651-app0", "trailing separator ignored");
    CHECK(path_basename("C:\\games\\PPSA24651-app0") == "PPSA24651-app0", "Windows separators");
    CHECK(path_basename("bare") == "bare", "a bare name is its own basename");

    // --- scan_game_library ----------------------------------------------------------------------
    const GamePathProbe probe = fake_probe();
    const GameLibraryIo io = fake_io();
    const std::vector<GameEntry> games = scan_game_library("/games", probe, io);
    CHECK(games.size() == 7, "exactly the seven titles are found (non-titles and files skipped)");
    if (games.size() == 7) {
        // Case-insensitive display order: astro bot, Blasphemous 2, PPSA00001, PPSA00002, Same Name
        // (PPSA00004), Same Name (PPSA00005), The Messenger. Two things are pinned here that a raw
        // byte sort would get wrong: "astro bot" leads despite 'a' > 'T', and the tie between the two
        // "Same Name" entries breaks on content id even though list_dir returned PPSA00005 first.
        CHECK(games[0].title_name == "astro bot", "the sort folds case ('astro bot' leads, not trails)");
        CHECK(games[1].title_name == "Blasphemous 2", "sorted by display name");
        CHECK(games[2].title_name == "PPSA00001-app0", "a title with no param.json falls back to its directory name");
        CHECK(games[3].title_name == "PPSA00002-app0", "a title with no titleName falls back to its directory name");
        CHECK(games[4].title_id == "PPSA00004" && games[5].title_id == "PPSA00005",
              "equal display names break the tie on content id, not on list_dir order");
        CHECK(games[6].title_name == "The Messenger", "sorted by display name (last)");

        CHECK(games[6].app0_root == "/games/PPSA24651-app0", "app0 root is what boot_program wants");
        CHECK(games[6].title_id == "PPSA24651", "content id captured");
        CHECK(games[6].icon_path == "/games/PPSA24651-app0/sce_sys/icon0.png", "icon located");
        CHECK(games[1].icon_path.empty(), "a title with no icon0.png reports no icon, not a bad path");
        CHECK(games[2].title_id.empty(), "no param.json means no content id");
        CHECK(games[3].title_id == "PPSA00002", "content id survives a missing titleName");
    }

    // --- scan_game_library: what must NOT happen ------------------------------------------------
    CHECK(scan_game_library("/empty", probe, io).empty(), "a directory with no titles yields nothing");
    CHECK(scan_game_library("/does/not/exist", probe, io).empty(), "a missing directory yields nothing");
    CHECK(scan_game_library("", probe, io).empty(), "an empty path yields nothing");
    // The games dir itself IS listed when it is a title root — a dump can sit at a drive
    // root, where the folder is the game rather than a folder of games. A dedicated probe (not
    // a PPSA-named fixture) so this test adds no title-id occurrences of its own.
    {
        GamePathProbe solo_probe;
        solo_probe.is_dir = [](const std::string& s) { return s == "/solo"; };
        solo_probe.is_file = [](const std::string& s) { return s == "/solo/eboot.bin"; };
        GameLibraryIo solo_io;
        solo_io.list_dir = [](const std::string&) { return std::vector<std::string>{}; };
        const std::vector<GameEntry> self = scan_game_library("/solo", solo_probe, solo_io);
        CHECK(self.size() == 1 && self[0].title_name == "solo" &&
                  self[0].app0_root == "/solo",
              "a title root used as the games dir lists itself");
    }
    CHECK(scan_game_library("/games/", probe, io).size() == 7, "a trailing separator is tolerated");
    const GameLibraryIo empty_io;
    CHECK(scan_game_library("/games", probe, empty_io).empty(), "an unpopulated io scans nothing");

    // --- describe_game: the recorded icon path must be one that opens -----------------------------
    {
        GamePathProbe casey;                      // accepts the icon only via a case correction
        casey.is_dir  = [](const std::string& d) { return d == "/g/T-app0"; };
        casey.is_file = [](const std::string& f) {
            return f == "/g/T-app0/SCE_SYS/ICON0.PNG" || f == "/g/T-app0/sce_sys/icon0.png";
        };
        GameLibraryIo casey_io;
        casey_io.read_file = [](const std::string&) { return std::string(); };
        casey_io.resolve_case = [](const std::string& want) {
            return want == "/g/T-app0/sce_sys/icon0.png" ? std::string("/g/T-app0/SCE_SYS/ICON0.PNG")
                                                         : want;
        };
        CHECK(prosper::frontend::describe_game("/g/T-app0", casey, casey_io).icon_path ==
                  "/g/T-app0/SCE_SYS/ICON0.PNG",
              "icon_path records the real on-disk spelling, not the requested one");
        GameLibraryIo plain_io;
        plain_io.read_file = [](const std::string&) { return std::string(); };
        CHECK(prosper::frontend::describe_game("/g/T-app0", casey, plain_io).icon_path ==
                  "/g/T-app0/sce_sys/icon0.png",
              "with no resolver the requested path is recorded unchanged");
    }

    // --- config parsing -------------------------------------------------------------------------
    CHECK(parse_app_config("games_dir = /games").games_dirs ==
              std::vector<std::string>{"/games"},
          "a setting is read");
    CHECK(parse_app_config("games_dir=/games").games_dirs ==
              std::vector<std::string>{"/games"},
          "spaces around = are optional");
    CHECK(parse_app_config("  games_dir  =  /games  ").games_dirs ==
              std::vector<std::string>{"/games"},
          "surrounding space trimmed");
    CHECK(parse_app_config("# comment\n\ngames_dir = /games\n").games_dirs ==
              std::vector<std::string>{"/games"},
          "comments and blank lines are ignored");
    CHECK(parse_app_config("games_dir = /a\ngames_dir = /b").games_dirs ==
              (std::vector<std::string>{"/a", "/b"}),
          "repeated games_dir lines accumulate in order; adding never drops");
    CHECK(parse_app_config("games_dir = /a\ngames_dir = /a\n").games_dirs ==
              std::vector<std::string>{"/a"},
          "a hand-duplicated folder line still lists once");
    CHECK(parse_app_config("future_key = 1\ngames_dir = /games").games_dirs ==
              std::vector<std::string>{"/games"},
          "an unknown key does not break parsing");
    CHECK(parse_app_config("guest_args = -force-gfx-direct").guest_args_default == "-force-gfx-direct",
          "a default guest_args value is read");
    AppConfig pt = parse_app_config("guest_args.PPSA02664 = -a\nguest_args.PPSA02664 = -b");
    CHECK(pt.guest_args_by_title.size() == 1 && pt.guest_args_by_title.at("PPSA02664") == "-b",
          "per-title guest_args: later duplicate wins, key splits on the dot");
    CHECK(guest_args_for(pt, "PPSA02664") == "-b" && guest_args_for(pt, "PPSA99999") == "",
          "per-title resolution falls back to empty without a default");
    AppConfig both = parse_app_config(
        "guest_args = -force-gfx-direct\nguest_args.PPSA02664 = -title-specific");
    CHECK(guest_args_for(both, "PPSA02664") == "-title-specific" &&
              guest_args_for(both, "PPSA24651") == "-force-gfx-direct",
          "per-title entry wins over the default; other titles get the default");
    AppConfig rt = parse_app_config("guest_args = d\n");
    rt.guest_args_by_title["PPSA02664"] = "x";
    const std::string round = serialize_app_config(rt);
    AppConfig back = parse_app_config(round);
    CHECK(back.guest_args_default == "d" && back.guest_args_by_title.at("PPSA02664") == "x",
          "guest args survive a serialize round trip");
    CHECK(round.find("guest_args.PPSA02664 = x") != std::string::npos &&
              round.find("guest_args = d") != std::string::npos,
          "serialized keys use the documented spellings");
    CHECK(parse_app_config("future_key = 1").unknown_lines.size() == 1,
          "an unknown key is retained rather than discarded");
    CHECK(parse_app_config("games_dir = /g").unknown_lines.empty(),
          "a known key is not also retained as unknown");
    CHECK(parse_app_config("games_dir = /my games/ps5 = final # 1").games_dirs ==
              std::vector<std::string>{"/my games/ps5 = final # 1"},
          "the value is literal after the first = (spaces, =, and # all survive)");
    CHECK(parse_app_config("").games_dirs.empty(), "an empty file sets nothing");
    CHECK(parse_app_config("nonsense").games_dirs.empty(), "a line with no = sets nothing");
    CHECK(parse_app_config("games_dir = /games\r\n").games_dirs ==
              std::vector<std::string>{"/games"},
          "CRLF is tolerated");

    // --- config round-trip ----------------------------------------------------------------------
    AppConfig cfg;
    cfg.games_dirs = {"/my games/ps5", "/more"};
    CHECK(parse_app_config(serialize_app_config(cfg)).games_dirs ==
              (std::vector<std::string>{"/my games/ps5", "/more"}),
          "serialize -> parse round-trips several folders with spaces, in order");
    CHECK(parse_app_config(serialize_app_config(AppConfig{})).games_dirs.empty(),
          "an unset config round-trips as unset");
    // --set-games-dir rewrites the whole file, so a key a NEWER build stored (stage 2 will add some)
    // must survive being read and written by an older one. Without unknown_lines this silently
    // deletes the user's other settings.
    {
        const AppConfig newer = parse_app_config("games_dir = /old\nui_scale = 1.5\n");
        AppConfig rewritten = newer;
        rewritten.games_dirs = {"/new"};
        const AppConfig after = parse_app_config(serialize_app_config(rewritten));
        CHECK(after.games_dirs == std::vector<std::string>{"/new"},
              "a rewrite updates the setting it owns");
        CHECK(after.unknown_lines.size() == 1 && after.unknown_lines[0] == "ui_scale = 1.5",
              "a rewrite preserves a setting this build does not understand");
    }

    // --- recent games --------------------------------------------------------------------------
    {
        AppConfig rc;
        note_recent_game(rc, "/games/B-app0");
        note_recent_game(rc, "/games/A-app0");
        CHECK(rc.recent_games.size() == 2 && rc.recent_games[0] == "/games/A-app0" &&
                  rc.recent_games[1] == "/games/B-app0",
              "recent games are most-recent-first");
        note_recent_game(rc, "/games/A-app0/");
        CHECK(rc.recent_games.size() == 2 && rc.recent_games[0] == "/games/A-app0",
              "reopening moves to the head without duplicating, separators canonicalized");
        note_recent_game(rc, "");
        CHECK(rc.recent_games.size() == 2, "an empty path records nothing");
        for (int i = 0; i < 12; i++)
            note_recent_game(rc, "/games/G" + std::to_string(i) + "-app0");
        CHECK(rc.recent_games.size() == AppConfig::kRecentGamesMax &&
                  rc.recent_games[0] == "/games/G11-app0",
              "the recent list is capped, newest head");
        const AppConfig back = parse_app_config(serialize_app_config(rc));
        CHECK(back.recent_games == rc.recent_games, "recent games survive a round trip in order");
        CHECK(parse_app_config("recent = \nrecent = /a").recent_games.size() == 1,
              "an empty recent line is not kept");
    }

    // --- volume --------------------------------------------------------------------------------
    CHECK(parse_volume_percent("") == -1, "an empty value is unset, not muted");
    CHECK(parse_volume_percent("0") == 0, "zero is a real choice (muted)");
    CHECK(parse_volume_percent("75") == 75, "a plain percent is read");
    CHECK(parse_volume_percent("100") == 100, "a hundred is read");
    CHECK(parse_volume_percent("120") == 100, "overflow clamps like the --volume flag");
    CHECK(parse_volume_percent("12x") == -1, "trailing junk is unset, not partial");
    CHECK(parse_volume_percent("-5") == -1, "a sign is unset, not negative");
    CHECK(parse_volume_percent(" 80") == -1, "surrounding space is unset (values arrive trimmed)");
    {
        AppConfig vc;
        vc.volume_percent = 0;
        CHECK(parse_app_config(serialize_app_config(vc)).volume_percent == 0,
              "muted survives a round trip rather than reading as unset");
        CHECK(parse_app_config(serialize_app_config(AppConfig{})).volume_percent == -1,
              "an unset volume stays unset (no phantom key is written)");
    }

    // --- host policy: the precedence main.cpp ships ------------------------------------------------
    // Every arm below feeds the environment THROUGH the inputs — including the env-wins arms the
    // old resolve_* tests could not reach because main read the environment inline.
    {
        AppConfig file;
        file.savedata_dir = "/saves";
        file.present_mode = "mailbox";
        file.display_mode = "host";
        file.volume_percent = 75;
        HostPolicyInputs in;
        in.file = file;
        const auto quiet = resolve_host_policy(in);
        CHECK(quiet.savedata_dir == "/saves" && quiet.present_mode == "mailbox" &&
                  quiet.display_mode == "host" && quiet.volume_percent == 75,
              "a silent run takes everything from the file");
        HostPolicyInputs loud = in;
        loud.flag_present_mode = true;
        loud.flag_display_mode = true;
        loud.flag_volume = true;
        loud.env_savedata_dir = "/env-saves";
        loud.env_display_mode = "legacy";
        const auto overridden = resolve_host_policy(loud);
        CHECK(overridden.savedata_dir.empty() && overridden.present_mode.empty() &&
                  overridden.display_mode.empty() && overridden.volume_percent == -1,
              "a flag or environment on every source leaves nothing for the file");
        HostPolicyInputs mixed = in;
        mixed.env_savedata_dir = "/env-saves";
        const auto mixed_out = resolve_host_policy(mixed);
        CHECK(mixed_out.savedata_dir.empty() && mixed_out.present_mode == "mailbox",
              "each source suppresses only its own file value");
        CHECK(resolve_host_policy(HostPolicyInputs{}).savedata_dir.empty() &&
                  resolve_host_policy(HostPolicyInputs{}).volume_percent == -1,
              "an empty everything resolves to leave-everything-alone");
    }

    // --- display mode: the flag and the environment each suppress the file ON THEIR OWN ----------
    // The `loud` arm above sets both, so each condition masks the other; these arms take one each.
    {
        HostPolicyInputs env_only;
        env_only.file.display_mode = "host";
        env_only.env_display_mode = "legacy";
        CHECK(resolve_host_policy(env_only).display_mode.empty(),
              "PROSPER_DISPLAY_MODE alone (no --display-mode) suppresses the file's display_mode");
        HostPolicyInputs flag_only;
        flag_only.file.display_mode = "host";
        flag_only.flag_display_mode = true;
        CHECK(resolve_host_policy(flag_only).display_mode.empty(),
              "--display-mode alone (no PROSPER_DISPLAY_MODE) suppresses the file's display_mode");
    }

    // --- which runs take the persisted host settings at all ------------------------------------
    // has_dump covers both `prosper-app <dump>` and `--dump <x>`: main.cpp passes !dump.empty().
    {
        namespace pf = prosper::frontend;
        CHECK(!pf::host_policy_applies(/*has_dump=*/true, /*test_pattern=*/false,
                                       /*from_library=*/false),
              "a scripted dump run never takes the saved settings");
        CHECK(!pf::host_policy_applies(/*has_dump=*/false, /*test_pattern=*/true,
                                       /*from_library=*/false),
              "a --test-pattern run never takes the saved settings");
        CHECK(pf::host_policy_applies(/*has_dump=*/false, /*test_pattern=*/false,
                                      /*from_library=*/false),
              "a bare launch (the library) takes the saved settings");
        CHECK(pf::host_policy_applies(/*has_dump=*/true, /*test_pattern=*/false,
                                      /*from_library=*/true),
              "a dump relaunched with --from-library takes the saved settings");
    }

    // --- volume: main.cpp applies the RESOLVER's answer, through volume_after_policy -------------
    {
        namespace pf = prosper::frontend;
        HostPolicyInputs in;
        in.file.volume_percent = 75;
        CHECK(pf::volume_after_policy(resolve_host_policy(in), 100) == 75,
              "with no --volume, the saved volume replaces the default");
        HostPolicyInputs flagged = in;
        flagged.flag_volume = true;
        CHECK(pf::volume_after_policy(resolve_host_policy(flagged), 40) == 40,
              "--volume wins: the flag's value survives a saved volume");
        CHECK(pf::volume_after_policy(resolve_host_policy(HostPolicyInputs{}), 100) == 100,
              "no saved volume leaves the current volume alone");
        pf::HostPolicy muted;
        muted.volume_percent = 0;
        CHECK(pf::volume_after_policy(muted, 100) == 0,
              "a saved volume of 0 (mute) is a real answer, not 'unset'");
    }

    // --- games folders: add, forget, merge ---------------------------------------------------------
    {
        AppConfig fc;
        note_games_dir(fc, "/games/B");
        note_games_dir(fc, "/games/A");
        CHECK(fc.games_dirs == (std::vector<std::string>{"/games/B", "/games/A"}),
              "added folders accumulate in order");
        note_games_dir(fc, "/games/A/");
        CHECK(fc.games_dirs.size() == 2 && fc.games_dirs[1] == "/games/A",
              "re-adding a folder is a no-op, separators canonicalized");
        note_games_dir(fc, "");
        CHECK(fc.games_dirs.size() == 2, "an empty path adds nothing");
        forget_games_dir(fc, "/games/B");
        CHECK(fc.games_dirs == std::vector<std::string>{"/games/A"},
              "forgetting removes that folder and keeps the rest");
        forget_games_dir(fc, "/games/missing");
        CHECK(fc.games_dirs == std::vector<std::string>{"/games/A"},
              "forgetting a folder that was never listed changes nothing");
        forget_games_dir(fc, "/games/A/");
        CHECK(fc.games_dirs.empty(), "forgetting canonicalizes too, so the last folder goes");
        const AppConfig back = parse_app_config(serialize_app_config(fc));
        CHECK(back.games_dirs.empty(), "an emptied list round-trips as unset");
    }
    {
        const GamePathProbe probe = fake_probe();
        const GameLibraryIo io = fake_io();
        const std::vector<GameEntry> one = scan_game_library("/games", probe, io);
        const std::vector<GameEntry> merged =
            scan_game_libraries({"/games", "/empty", "/missing"}, probe, io);
        CHECK(merged.size() == one.size(), "a merge is the single scan plus empty and gone folders");
        bool same = merged.size() == one.size();
        for (size_t i = 0; same && i < one.size(); i++)
            same = merged[i].app0_root == one[i].app0_root;
        CHECK(same, "merging adds nothing and reorders nothing when the extras hold no titles");
        CHECK(scan_game_libraries({"/games", "/games"}, probe, io).size() == one.size(),
              "listing one folder twice still lists its titles once");
        CHECK(scan_game_libraries({}, probe, io).empty(), "no folders is no titles");
        CHECK(scan_game_libraries({"/missing"}, probe, io).empty(),
              "a gone folder contributes nothing instead of hiding the rest");

        // Two folders holding DIFFERENT titles: both folders' titles are listed, in one display
        // order across the folders. Fails if only the first folder is scanned (Middle Title is
        // absent), and fails if the folders are concatenated without the cross-folder sort (Middle
        // Title would come last instead of straight after Blasphemous 2).
        const std::vector<GameEntry> two = scan_game_libraries({"/games", "/drive2"}, probe, io);
        std::vector<GameEntry> expected = one;
        expected.push_back(scan_game_library("/drive2", probe, io).at(0));
        std::sort(expected.begin(), expected.end(), prosper::frontend::game_entry_display_less);
        bool has_first = false, has_second = false;
        for (const GameEntry& g : two) {
            if (g.title_name == "The Messenger") has_first = true;
            if (g.app0_root == "/drive2/middle-app0") has_second = true;
        }
        CHECK(two.size() == one.size() + 1 && has_first && has_second,
              "two folders with different titles list the titles of BOTH folders");
        bool ordered = two.size() == expected.size();
        for (size_t i = 0; ordered && i < expected.size(); i++)
            ordered = two[i].app0_root == expected[i].app0_root;
        CHECK(ordered, "titles from two folders share one display order (game_entry_display_less)");
        bool middle_placed = false;
        for (size_t i = 1; i + 1 < two.size(); i++)
            if (two[i].app0_root == "/drive2/middle-app0" &&
                two[i - 1].title_name == "Blasphemous 2")
                middle_placed = true;
        CHECK(middle_placed, "the second folder's title sorts between the first folder's titles");
        const std::vector<GameEntry> swapped =
            scan_game_libraries({"/drive2", "/games"}, probe, io);
        bool same_order = swapped.size() == two.size();
        for (size_t i = 0; same_order && i < two.size(); i++)
            same_order = swapped[i].app0_root == two[i].app0_root;
        CHECK(same_order, "folder order does not change the listed order");
    }

    // --- --list-games over several folders: the library's rule for a missing folder ---------------
    {
        namespace pf = prosper::frontend;
        const GamePathProbe probe = fake_probe();
        const GameLibraryIo io = fake_io();

        // One of two folders missing (an unplugged drive): warn about it, list the other, exit 0.
        const pf::GamesDirsAvailability one_gone =
            pf::split_available_games_dirs({"/games", "/missing"}, probe);
        CHECK(one_gone.available == std::vector<std::string>{"/games"} &&
                  one_gone.missing == std::vector<std::string>{"/missing"},
              "one of two folders missing: the present one is scanned, the gone one warned about");
        const size_t listed = scan_game_libraries(one_gone.available, probe, io).size();
        CHECK(listed == scan_game_library("/games", probe, io).size() && listed != 0,
              "one of two folders missing: the present folder's titles are still listed");
        CHECK(pf::list_games_exit_code(one_gone, listed) == 0,
              "one of two folders missing: exit 0, not 2 -- a gone drive does not hide the rest");

        // Every folder missing: nothing to list, exit 2 (a wrong path, not an empty library).
        const pf::GamesDirsAvailability all_gone =
            pf::split_available_games_dirs({"/missing", "/also-missing"}, probe);
        CHECK(all_gone.available.empty() && all_gone.missing.size() == 2,
              "every folder missing: both are warned about, none is scanned");
        CHECK(pf::list_games_exit_code(all_gone, 0) == 2, "every folder missing: exit 2");

        // The single-folder (--games-dir / PROSPER_GAMES_DIR) contract is unchanged.
        CHECK(pf::list_games_exit_code(pf::split_available_games_dirs({"/missing"}, probe), 0) == 2,
              "a single missing folder still exits 2");
        CHECK(pf::list_games_exit_code(pf::split_available_games_dirs({"/empty"}, probe), 0) == 1,
              "a present folder holding no titles exits 1");
        CHECK(pf::list_games_exit_code(pf::split_available_games_dirs({}, probe), 0) == 2,
              "no folder set exits 2");
    }

    // --- newline safety ----------------------------------------------------------------------------
    {
        AppConfig evil;
        evil.games_dirs = {"/games\nplanted = 1"};
        evil.savedata_dir = "/saves\nplanted = 1";
        evil.recent_games = {"/ok", "/bad\nplanted = 1"};
        const std::string text = serialize_app_config(evil);
        CHECK(text.find("planted") == std::string::npos,
              "a newline in a path value cannot inject a key on rewrite");
        CHECK(parse_app_config(text).games_dirs.empty(),
              "a dropped games_dir reads as unset, not as half a path");
    }

    // --- precedence -----------------------------------------------------------------------------
    AppConfig from_file;
    from_file.games_dirs = {"/from-file", "/also-file"};
    CHECK(resolve_games_dirs("/from-flag", "/from-env", from_file) ==
              std::vector<std::string>{"/from-flag"},
          "--games-dir wins over everything, naming the whole library for the run");
    CHECK(resolve_games_dirs("", "/from-env", from_file) ==
              std::vector<std::string>{"/from-env"},
          "the environment wins over the persisted setting");
    CHECK(resolve_games_dirs("", "", from_file) ==
              (std::vector<std::string>{"/from-file", "/also-file"}),
          "the persisted list applies whole when nothing else does");
    CHECK(resolve_games_dirs("", "", AppConfig{}).empty(), "with no source there is no games dir");
    CHECK(resolve_games_dirs("/from-flag", "", AppConfig{}) ==
              std::vector<std::string>{"/from-flag"},
          "the flag alone is enough");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
