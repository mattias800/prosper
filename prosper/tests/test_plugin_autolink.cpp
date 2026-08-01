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
#include "../src/loader/linker.hpp"
#include "test_scratch.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
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

int main(int argc, char** argv) {
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

    // ---- export-collision predicate ------------------------------------------------------------
    // Discovery alone is not enough: a title can ship two builds of the SAME library under different
    // filenames. Evergate (PPSA01885) ships `libfmod.prx` AND `libfmodL.prx` — FMOD's release and
    // logging builds, exporting identical NIDs. The global export table is first-definition-wins and
    // silent, so linking both would run two init_arrays and make sceKernelDlsym answer differently
    // depending on which module handle it is asked through. Basename dedup cannot see that; the
    // linker therefore refuses an optional module whose exports collide.
    //
    // Module is a plain struct, so the predicate is exercised here on synthetic symbol tables with no
    // ELF fixture and no game dump — this part runs in CI.
    auto sym = [](const char* nid, bool is_import, uint64_t value) {
        prosper::Symbol s; s.nid = nid; s.is_import = is_import; s.value = value; return s;
    };
    prosper::Module release;                        // stands in for libfmod.prx
    release.path = "libfmod.prx";
    release.symbols = { sym("AAAAAAAAAAA", false, 0x1000), sym("BBBBBBBBBBB", false, 0x2000) };
    prosper::Module logging;                        // stands in for libfmodL.prx: same NIDs
    logging.path = "libfmodL.prx";
    logging.symbols = { sym("AAAAAAAAAAA", false, 0x8000), sym("BBBBBBBBBBB", false, 0x9000) };
    prosper::Module distinct;                       // stands in for libresonanceaudio.prx
    distinct.path = "libresonanceaudio.prx";
    distinct.symbols = { sym("CCCCCCCCCCC", false, 0x1000) };

    CHECK(prosper::module_export_nids(release).size() == 2, "module_export_nids counts real exports");

    std::unordered_map<std::string, std::string> claimed;
    for (const auto& n : prosper::module_export_nids(release)) claimed.emplace(n, release.path);

    const prosper::ExportCollision dup = prosper::find_export_collision(logging, claimed);
    CHECK(dup.nid == "AAAAAAAAAAA", "duplicate-export module reports the first colliding NID");
    CHECK(dup.owner_path == "libfmod.prx", "collision names the module that already owns the NID");
    CHECK(prosper::find_export_collision(distinct, claimed).nid.empty(),
          "a module with disjoint exports does not collide");
    CHECK(prosper::find_export_collision(logging, {}).nid.empty(),
          "the debug build alone does not collide (a title shipping only it still links it)");

    // The collision predicate must match the export-table predicate exactly, or a module could be
    // accepted as collision-free and then still alias someone else's NID: imports, empty NIDs and
    // zero-valued symbols contribute nothing and must never trigger a skip.
    prosper::Module noise;
    noise.path = "noise.prx";
    noise.symbols = { sym("AAAAAAAAAAA", true, 0x1000),    // import of the same NID
                      sym("BBBBBBBBBBB", false, 0),        // defined but value 0
                      sym("", false, 0x4000) };            // no NID
    CHECK(prosper::module_export_nids(noise).empty(),
          "imports, zero-valued and NID-less symbols are not exports");
    CHECK(prosper::find_export_collision(noise, claimed).nid.empty(),
          "importing a claimed NID is not a collision");

    // ---- dump-backed wiring (only when a game dump is configured) ------------------------------
    // Proves the flag is actually honoured by link_program: the same PRX listed twice is a guaranteed
    // self-collision, so the flagged second copy must be dropped, and the unflagged control must not.
    if (argc >= 2) {
        const std::string dump_root = argv[1];
        const std::string plugin = dump_root + "/Media/Plugins/PSN.prx";
        if (fs::exists(fs::path(plugin))) {
            const std::vector<prosper::LinkInput> control = {
                { dump_root + "/eboot.bin", 0x410000000ull },
                { plugin, 0x5d0000000ull },
                { plugin, 0x5d4000000ull },                 // no flag: linked twice, as before
            };
            std::vector<prosper::LinkInput> guarded = control;
            guarded[2].skip_on_export_collision = true;

            prosper::Program pc, pg;
            std::string lerr;
            const bool okc = prosper::link_program(control, 0x600000000ull, pc, &lerr);
            const bool okg = prosper::link_program(guarded, 0x600000000ull, pg, &lerr);
            CHECK(okc && okg, "both link runs succeed");
            CHECK(pc.skipped_modules.empty(), "control: an unflagged duplicate is still linked");
            CHECK(pc.mods.size() == 3, "control: three modules linked");
            CHECK(pg.skipped_modules.size() == 1, "guarded: the colliding duplicate is skipped");
            CHECK(pg.mods.size() == 2, "guarded: only two modules linked");
            if (pg.skipped_modules.size() == 1) {
                CHECK(pg.skipped_modules[0].path == plugin, "skip record names the dropped module");
                CHECK(pg.skipped_modules[0].owner_path == plugin, "skip record names the owner");
                CHECK(!pg.skipped_modules[0].nid.empty(), "skip record carries the colliding NID");
            }

            // #1635: the CONTROL case is the real-world shape — two modules exporting the same NIDs,
            // neither flagged, both linked. exports.emplace() keeps the first and discards the second
            // with no warning, no counter and nothing in the summary. That silence is the defect;
            // first-wins itself is the intended policy and is unchanged.
            //
            // Measured, not hypothetical: a census over 30 local dumps found 41 aliased NIDs across 7
            // titles, ALL among modules linked by name (tools/re/dup_exports.py).
            CHECK(!pc.aliased_exports.empty(),
                  "#1635: linking a duplicate records the aliased exports instead of silently dropping them");
            CHECK(pg.aliased_exports.empty(),
                  "#1635: a module the guard skipped contributes no aliases");
            if (!pc.aliased_exports.empty()) {
                const auto& a = pc.aliased_exports.front();
                CHECK(!a.nid.empty(), "#1635: the alias record carries the NID");
                CHECK(a.winner_path == plugin && a.loser_path == plugin,
                      "#1635: the alias record names both the winner and the discarded loser");
                // The two copies are mapped at different bases, so the discarded definition really is
                // a different address — proof this is an alias, not a duplicate of one address.
                CHECK(a.winner != a.loser,
                      "#1635: the discarded definition is a genuinely different address");
                // first-wins means the winner lies inside the image of the EARLIER-linked module.
                // Comparing the two addresses directly would only assert this fixture's base ordering
                // (0x5d0000000 < 0x5d4000000) — both records share one RVA, so that comparison is a
                // statement about the test's own setup, not about the loader's policy.
                bool winner_in_first = false, loser_in_second = false;
                if (pc.imgs.size() >= 3) {
                    const uint64_t b1 = pc.imgs[1].base, b2 = pc.imgs[2].base;
                    winner_in_first  = a.winner >= b1 && a.winner < b2;
                    loser_in_second  = a.loser  >= b2;
                }
                CHECK(winner_in_first && loser_in_second,
                      "#1635: first-wins — the winner is in the earlier image, the loser in the later");
            }
        } else {
            printf("  [skip] %s has no Media/Plugins/PSN.prx — link wiring not exercised\n",
                   dump_root.c_str());
        }
    } else {
        printf("  [skip] no game dump argument — link wiring not exercised\n");
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
