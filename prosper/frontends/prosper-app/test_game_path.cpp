// test_game_path.cpp — the interactive open path's pure decision seam (#1469). No SDL, no window,
// no real dump: the filesystem is a fixed set of names, so every branch is exercised in CI.
#include "game_path.hpp"

#include <cstdio>
#include <set>
#include <string>

using prosper::frontend::GameOpenAction;
using prosper::frontend::GamePathProbe;
using prosper::frontend::StartupPickInputs;
using prosper::frontend::decide_open_action;
using prosper::frontend::is_app0_root;
using prosper::frontend::parent_directory;
using prosper::frontend::quote_windows_argument;
using prosper::frontend::resolve_app0_root;
using prosper::frontend::should_pick_at_startup;
using prosper::frontend::strip_trailing_separators;
using prosper::frontend::windows_command_line;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

// A fake tree holding one real title, one title that ships only metadata, and one ordinary folder:
//
//   /games/PPSA24651-app0/{eboot.bin, sce_sys/param.json, Media/movie.bin}
//   /games/PPSA13579-app0/sce_sys/param.json          (no eboot.bin)
//   /games/notes/readme.txt
static const std::set<std::string> kDirs = {
    "/games",
    "/games/PPSA24651-app0", "/games/PPSA24651-app0/sce_sys", "/games/PPSA24651-app0/Media",
    "/games/PPSA24651-app0/Media/sub", "/games/PPSA24651-app0/Media/sub/deep",
    "/games/PPSA13579-app0", "/games/PPSA13579-app0/sce_sys",
    "/games/notes",
};
static const std::set<std::string> kFiles = {
    "/games/PPSA24651-app0/eboot.bin",
    "/games/PPSA24651-app0/sce_sys/param.json",
    "/games/PPSA24651-app0/Media/movie.bin",
    "/games/PPSA24651-app0/Media/sub/deep/asset.bin",
    "/games/PPSA13579-app0/sce_sys/param.json",
    "/games/notes/readme.txt",
};
static GamePathProbe fake_probe() {
    GamePathProbe p;
    p.is_dir  = [](const std::string& s) { return kDirs.count(s) != 0; };
    p.is_file = [](const std::string& s) { return kFiles.count(s) != 0; };
    return p;
}

int main() {
    std::printf("== test_prosper_app_game_path ==\n");

    // --- strip_trailing_separators ------------------------------------------------------------
    CHECK(strip_trailing_separators("/games/x/") == "/games/x", "a trailing separator is dropped");
    CHECK(strip_trailing_separators("/games/x///") == "/games/x", "repeated separators are dropped");
    CHECK(strip_trailing_separators("/games/x") == "/games/x", "a clean path is unchanged");
    CHECK(strip_trailing_separators("/") == "/", "the POSIX root keeps its separator");
    CHECK(strip_trailing_separators("C:\\") == "C:\\", "a Windows drive root keeps its separator");
    CHECK(strip_trailing_separators("C:\\games\\x\\") == "C:\\games\\x", "Windows separators strip too");
    CHECK(strip_trailing_separators("") == "", "an empty path stays empty");

    // --- parent_directory ---------------------------------------------------------------------
    CHECK(parent_directory("/games/x") == "/games", "parent of a nested path");
    CHECK(parent_directory("/games/x/") == "/games", "trailing separators do not shift the parent");
    CHECK(parent_directory("/games") == "/", "parent of a top-level entry is the root");
    CHECK(parent_directory("/") == "", "the root has no parent");
    CHECK(parent_directory("game") == "", "a bare name has no parent");
    CHECK(parent_directory("C:\\games\\x") == "C:\\games", "parent of a nested Windows path");
    CHECK(parent_directory("C:\\games") == "C:\\", "parent of a top-level Windows entry is the drive root");
    CHECK(parent_directory("C:\\") == "", "a Windows drive root has no parent");

    // --- is_app0_root -------------------------------------------------------------------------
    const GamePathProbe probe = fake_probe();
    CHECK(is_app0_root("/games/PPSA24651-app0", probe), "a directory holding eboot.bin is a title root");
    CHECK(is_app0_root("/games/PPSA13579-app0", probe),
          "sce_sys/param.json alone also identifies a title root");
    CHECK(!is_app0_root("/games", probe), "the folder CONTAINING titles is not itself a title root");
    CHECK(!is_app0_root("/games/notes", probe), "an ordinary folder is not a title root");
    CHECK(!is_app0_root("/games/PPSA24651-app0/eboot.bin", probe), "a file is not a title root");
    CHECK(!is_app0_root("/nope", probe), "a path that does not exist is not a title root");
    CHECK(!is_app0_root("", probe), "an empty path is not a title root");

    // --- resolve_app0_root: everything a drop or a picker can plausibly deliver ----------------
    const std::string kRoot = "/games/PPSA24651-app0";
    CHECK(resolve_app0_root(kRoot, probe) == kRoot, "the app0 root resolves to itself");
    CHECK(resolve_app0_root(kRoot + "/", probe) == kRoot, "a picker's trailing separator is tolerated");
    CHECK(resolve_app0_root(kRoot + "/eboot.bin", probe) == kRoot, "a dropped eboot.bin resolves up");
    CHECK(resolve_app0_root(kRoot + "/sce_sys", probe) == kRoot, "a dropped sce_sys resolves up");
    CHECK(resolve_app0_root(kRoot + "/sce_sys/param.json", probe) == kRoot,
          "a file one level below the root resolves up");
    CHECK(resolve_app0_root(kRoot + "/Media/movie.bin", probe) == kRoot,
          "an asset one level below the root resolves up");
    CHECK(resolve_app0_root("/games/PPSA13579-app0/sce_sys/param.json", probe) == "/games/PPSA13579-app0",
          "a metadata-only title resolves from inside sce_sys");

    // --- resolve_app0_root: what must NOT resolve ----------------------------------------------
    CHECK(resolve_app0_root("/games", probe).empty(),
          "the folder containing titles does not resolve (stage 1 has no library scan)");
    CHECK(resolve_app0_root("/games/notes", probe).empty(), "an ordinary folder does not resolve");
    CHECK(resolve_app0_root("/games/notes/readme.txt", probe).empty(),
          "a file in an ordinary folder does not resolve");
    CHECK(resolve_app0_root("/does/not/exist", probe).empty(), "a missing path does not resolve");
    CHECK(resolve_app0_root("", probe).empty(), "an empty path does not resolve");
    CHECK(resolve_app0_root("/games/PPSA24651-app0/Media", probe) == kRoot,
          "a directory one level below the root still resolves to that root");
    // The walk is BOUNDED, and these are the assertions that pin the bound: widening it would let a
    // drop from deep inside one title's assets climb out and boot whatever sits above.
    CHECK(resolve_app0_root("/games/PPSA24651-app0/Media/sub/deep", probe).empty(),
          "a directory three levels below the root does NOT resolve (walk is bounded)");
    CHECK(resolve_app0_root("/games/PPSA24651-app0/Media/sub/deep/asset.bin", probe).empty(),
          "a file four levels below the root does NOT resolve (walk is bounded)");
    // ...and this pins the far edge that still must work: a picked file starts at its containing
    // directory, so it reaches exactly one level deeper than a picked directory.
    CHECK(resolve_app0_root("/games/PPSA24651-app0/Media/sub", probe) == kRoot,
          "a directory two levels below the root is the deepest that still resolves");

    // A probe with no callbacks must be inert rather than crash.
    const GamePathProbe empty_probe;
    CHECK(resolve_app0_root(kRoot, empty_probe).empty(), "an unpopulated probe resolves nothing");
    CHECK(!is_app0_root(kRoot, empty_probe), "an unpopulated probe identifies no root");

    // --- decide_open_action --------------------------------------------------------------------
    CHECK(decide_open_action(kRoot, /*boot_attempted=*/false) == GameOpenAction::boot_in_process,
          "with no boot attempted yet, a title boots on this process");
    // The attempt is what is spent, not the success: boot_program() appends into its Program and
    // re-runs one-shot global setup, so a second in-process call would link behind the leftovers of
    // the first and enter the wrong image. A FAILED boot must therefore relaunch, exactly like a
    // successful one — this is the assertion that pins that.
    CHECK(decide_open_action(kRoot, /*boot_attempted=*/true) == GameOpenAction::relaunch,
          "once a boot has been attempted — successful OR failed — a title needs a fresh process");
    CHECK(decide_open_action("", /*boot_attempted=*/false) == GameOpenAction::ignore,
          "an unresolved path is ignored, not booted");
    CHECK(decide_open_action("", /*boot_attempted=*/true) == GameOpenAction::ignore,
          "an unresolved path never relaunches a running game");

    // --- quote_windows_argument / windows_command_line ------------------------------------------
    // The relaunch's Windows command line. A trailing backslash is the dangerous case: a
    // tab-completed directory has one, and so does a title at a drive root.
    CHECK(quote_windows_argument("plain") == "\"plain\"", "an ordinary argument is just quoted");
    CHECK(quote_windows_argument("with space") == "\"with space\"", "a space needs no escaping inside quotes");
    CHECK(quote_windows_argument("C:\\games\\PPSA24651-app0") == "\"C:\\games\\PPSA24651-app0\"",
          "interior backslashes stay literal");
    CHECK(quote_windows_argument("C:\\games\\PPSA24651-app0\\") == "\"C:\\games\\PPSA24651-app0\\\\\"",
          "a trailing backslash is doubled so it cannot escape the closing quote");
    CHECK(quote_windows_argument("D:\\") == "\"D:\\\\\"", "a drive root's backslash is doubled");
    CHECK(quote_windows_argument("ends\\\\") == "\"ends\\\\\\\\\"", "a trailing RUN of backslashes is fully doubled");
    CHECK(quote_windows_argument("a\"b") == "\"a\\\"b\"", "an embedded quote is escaped");
    CHECK(quote_windows_argument("a\\\"b") == "\"a\\\\\\\"b\"",
          "backslashes before an embedded quote are doubled, then the quote is escaped");
    CHECK(quote_windows_argument("") == "\"\"", "an empty argument survives as an empty quoted token");
    CHECK(windows_command_line({"exe", "--dump", "C:\\g\\t\\"}) == "\"exe\" \"--dump\" \"C:\\g\\t\\\\\"",
          "the joined command line quotes every argument and preserves the trailing separator");
    CHECK(windows_command_line({}).empty(), "no arguments produce an empty command line");

    // --- should_pick_at_startup ------------------------------------------------------------------
    StartupPickInputs in{};
    in.bare_launch = true;
    CHECK(should_pick_at_startup(in), "a no-argument launch offers the picker");
    in.bare_launch = false;
    CHECK(!should_pick_at_startup(in), "an argument-bearing launch does not pop a dialog on its own");
    in.forced = true;
    CHECK(should_pick_at_startup(in), "--pick opens the picker without a bare launch");
    in.suppressed = true;
    CHECK(!should_pick_at_startup(in), "--no-pick overrides --pick");
    in = {}; in.bare_launch = true; in.suppressed = true;
    CHECK(!should_pick_at_startup(in), "--no-pick suppresses the bare-launch picker");
    in = {}; in.bare_launch = true; in.has_dump = true;
    CHECK(!should_pick_at_startup(in), "a run given a dump is never interrupted by a dialog");
    in = {}; in.forced = true; in.has_dump = true;
    CHECK(!should_pick_at_startup(in), "an explicit dump wins over --pick");
    in = {}; in.bare_launch = true; in.test_pattern = true;
    CHECK(!should_pick_at_startup(in), "--test-pattern already knows what to show");
    in = {}; in.forced = true; in.test_pattern = true;
    CHECK(!should_pick_at_startup(in), "--test-pattern wins over --pick");
    in = {};
    CHECK(!should_pick_at_startup(in), "nothing implicit happens without a bare launch");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
