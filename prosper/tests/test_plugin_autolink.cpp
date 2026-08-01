// test_plugin_autolink — guards discover_extra_plugin_modules (#1609): the boot preloader must link
// the Unity native plugins a title ships that nobody hard-coded into the fixed preload list.
//
// Why this matters: prosper has no runtime PRX loading (#639), so a plugin that is not linked at boot
// makes the guest's first P/Invoke into it raise DllNotFoundException. IL2CPP raises that inside an
// async state machine, so it is captured into a Task and never printed — the title just stops making
// progress with no diagnostic. Tales of Graces f Remastered (PPSA19991) ships cri_ware_unity.prx,
// GameNative.prx, PlayGo.prx and lib_burst_generated.prx; three of the thirty singletons its boot
// state machine awaits fault that way, so `NobleTask.WhenAll` never completes and the engine renders
// an empty scene forever at full frame rate.
//
// The contract under test: everything in <dump>/Media/Plugins/*.prx that the caller did not already
// name (case-insensitively) is returned, in a deterministic order, and nothing else is.
#include "../src/host/boot_program.hpp"
#include "test_scratch.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using prosper::discover_extra_plugin_modules;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

static std::vector<std::string> basenames(const std::vector<std::string>& paths) {
    std::vector<std::string> out;
    for (const auto& p : paths) out.push_back(fs::path(p).filename().string());
    return out;
}

static bool has(const std::vector<std::string>& v, const std::string& s) {
    for (const auto& e : v) if (e == s) return true;
    return false;
}

int main() {
    std::error_code ec;
    const fs::path root = prosper_test::test_scratch_dir() / "prosper_test_plugin_autolink";
    fs::remove_all(root, ec);
    fs::create_directories(root / "Media" / "Plugins", ec);
    auto touch = [&](const char* name) { std::ofstream(root / "Media" / "Plugins" / name) << "x"; };
    // A PPSA19991-shaped plugin directory: two names the fixed list knows, four it does not.
    touch("PSN.prx");
    touch("SaveData.prx");
    touch("cri_ware_unity.prx");
    touch("GameNative.prx");
    touch("PlayGo.prx");
    touch("lib_burst_generated.prx");
    touch("lib_burst_generated.txt");   // sibling non-module file must be ignored
    touch("notaplugin.so");

    const std::string dump = root.string();
    const std::vector<std::string> listed = { "eboot.bin", "PSN.prx", "SaveData.prx", "libc.prx" };

    const std::vector<std::string> got = discover_extra_plugin_modules(dump, listed);
    const std::vector<std::string> names = basenames(got);

    CHECK(names.size() == 4, "returns exactly the four unlisted .prx entries");
    CHECK(has(names, "cri_ware_unity.prx"), "cri_ware_unity.prx discovered");
    CHECK(has(names, "GameNative.prx"), "GameNative.prx discovered");
    CHECK(has(names, "PlayGo.prx"), "PlayGo.prx discovered");
    CHECK(has(names, "lib_burst_generated.prx"), "lib_burst_generated.prx discovered");
    CHECK(!has(names, "PSN.prx") && !has(names, "SaveData.prx"),
          "already-listed plugins are not returned twice");
    CHECK(!has(names, "lib_burst_generated.txt") && !has(names, "notaplugin.so"),
          "non-.prx files are ignored");

    // Every returned entry must be a real, openable path (the caller links it directly).
    bool all_exist = true;
    for (const auto& p : got) if (!fs::exists(fs::path(p))) all_exist = false;
    CHECK(all_exist, "returned entries are existing absolute paths");

    // Deterministic order: descending case-insensitive basename, so the caller's reverse-order init
    // runs them ascending. Without this a filesystem-defined order would make boots irreproducible.
    bool ordered = true;
    auto lower = [](std::string s) { for (auto& c : s) c = (char)std::tolower((unsigned char)c); return s; };
    for (size_t i = 1; i < names.size(); i++)
        if (!(lower(names[i - 1]) > lower(names[i]))) ordered = false;
    CHECK(ordered, "result is sorted descending by lowercased basename (reproducible boot)");

    // Case-insensitive listed-name match: a dump spelling the plugin differently must still be
    // treated as already handled, never linked twice at two different bases.
    const std::vector<std::string> listed_caps = { "psn.PRX", "savedata.prx" };
    const std::vector<std::string> ci = basenames(discover_extra_plugin_modules(dump, listed_caps));
    CHECK(!has(ci, "PSN.prx") && !has(ci, "SaveData.prx"),
          "listed-name comparison is case-insensitive");

    // A dump with no Plugins directory at all must yield nothing (and not throw).
    const fs::path bare = root / "bare";
    fs::create_directories(bare / "Media", ec);
    CHECK(discover_extra_plugin_modules(bare.string(), listed).empty(),
          "absent Media/Plugins directory yields no entries");
    CHECK(discover_extra_plugin_modules("", listed).empty(), "empty dump root yields no entries");

    // Case-only mismatch on the directory itself (case-sensitive host filesystems, cf. #1006).
    const fs::path lc = root / "lowerdir";
    fs::create_directories(lc / "media" / "plugins", ec);
    { std::ofstream(lc / "media" / "plugins" / "Extra.prx") << "x"; }
    const std::vector<std::string> lcnames = basenames(discover_extra_plugin_modules(lc.string(), listed));
    CHECK(has(lcnames, "Extra.prx"), "lower-cased Media/Plugins directory is still discovered");

    fs::remove_all(root, ec);
    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
