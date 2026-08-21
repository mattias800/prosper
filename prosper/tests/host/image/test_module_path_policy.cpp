// test_module_path_policy — prosper links guest modules only from the dump locations it names, and
// refuses everything else by default.
//
// Two layers, and they are NOT the same strength, so the difference is stated rather than left for a
// reader to assume:
//
//   1. `classify_module_path` unit arms. Pure string policy; every arm reddens under a targeted
//      mutation of the function.
//   2. ONE end-to-end arm through `boot_link_inputs`. It is built specifically so that it FAILS
//      without the guard: `Media/Plugins/libSceNpEntitlementAccess.prx` sits in a directory that
//      #1609 auto-links wholesale, so before this policy existed the loader linked it. That makes it
//      a real regression arm rather than a restatement of an invariant that already held.
//
//      By contrast a `fakelib/*.sprx` fixture would pass with OR without the guard — wrong
//      directory, not recursive, wrong extension — so it is included only as an invariant arm and
//      labelled as one. It cannot fail today; its job is to fail the day module discovery widens.
#include "fixtures/test_scratch.h"
#include "host/image/boot_program.hpp"
#include "host/image/module_path_policy.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace prosper;
namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static void permit(const char* root, const char* path, const char* what) {
    const ModulePathDecision d = classify_module_path(root, path);
    if (!d.permitted()) printf("         (refused: %s)\n", d.reason.c_str());
    CHECK(d.permitted(), what);
}

static void refuse(const char* root, const char* path, ModulePathVerdict want, const char* what) {
    const ModulePathDecision d = classify_module_path(root, path);
    const bool ok = !d.permitted() && d.verdict == want && !d.reason.empty();
    if (!ok && d.permitted()) printf("         (permitted, but should not be)\n");
    CHECK(ok, what);
}

static void write_file(const fs::path& p, const char* bytes) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f.write(bytes, (std::streamsize)std::char_traits<char>::length(bytes));
}

static bool linked(const std::vector<LinkInput>& in, const std::string& needle) {
    for (const auto& e : in)
        if (e.path.find(needle) != std::string::npos) return true;
    return false;
}

int main() {
    printf("== test_module_path_policy ==\n");
    const char* R = "/dumps/PPSA00000-app0";

    printf("-- permitted locations --\n");
    permit(R, "/dumps/PPSA00000-app0/eboot.bin", "eboot.bin in the dump root");
    permit(R, "/dumps/PPSA00000-app0/sce_module/libc.prx", "sce_module/libc.prx");
    permit(R, "/dumps/PPSA00000-app0/sce_module/libSceNpCppWebApi.prx",
           "sce_module MAY hold a Sony library -- that is the whole point of the directory");
    permit(R, "/dumps/PPSA00000-app0/Media/Plugins/PSN.prx", "Media/Plugins/PSN.prx");
    permit(R, "/dumps/PPSA00000-app0/Media/Modules/Il2cppUserAssemblies.prx", "Media/Modules/...");
    // Casing: PS5 module paths are effectively case-insensitive and #1006 case-corrects them, so a
    // policy that were case-SENSITIVE would refuse modules the loader legitimately resolved.
    permit(R, "/dumps/PPSA00000-app0/media/plugins/PSN.prx", "directory match is case-insensitive");
    permit(R, "/dumps/PPSA00000-app0/Media/Modules/Il2CppUserAssemblies.prx",
           "the #1006 alternate casing of a real module is still permitted");
    // Separator normalization, so a Windows-shaped path is judged the same way.
    permit("C:\\dumps\\PPSA00000-app0", "C:\\dumps\\PPSA00000-app0\\sce_module\\libc.prx",
           "backslash-separated Windows path");
    permit("/dumps/PPSA00000-app0/", "/dumps/PPSA00000-app0//sce_module/libc.prx",
           "trailing and doubled separators normalize away");

    printf("-- refused: outside the dump --\n");
    refuse(R, "/etc/ld.so.preload", ModulePathVerdict::OutsideDumpRoot, "an absolute host path");
    refuse(R, "/dumps/PPSA00000-app0-evil/sce_module/libc.prx", ModulePathVerdict::OutsideDumpRoot,
           "a sibling directory sharing the root's prefix is NOT inside it");
    refuse(R, "/dumps/PPSA00000-app0/sce_module/../../other/x.prx",
           ModulePathVerdict::OutsideDumpRoot, "`..` is refused, not resolved");
    refuse(R, "/dumps/PPSA00000-app0/./sce_module/libc.prx", ModulePathVerdict::OutsideDumpRoot,
           "`.` is refused so one location has exactly one spelling");

    printf("-- refused: wrong directory --\n");
    refuse(R, "/dumps/PPSA00000-app0/fakelib/libSceNpEntitlementAccess.sprx",
           ModulePathVerdict::DirectoryNotPermitted, "fakelib/libSceNpEntitlementAccess.sprx");
    refuse(R, "/dumps/PPSA00000-app0/fakelib/libSceAmpr.sprx",
           ModulePathVerdict::DirectoryNotPermitted, "fakelib/libSceAmpr.sprx");
    refuse(R, "/dumps/PPSA00000-app0/FAKELIB/libSceAppContent.sprx",
           ModulePathVerdict::DirectoryNotPermitted, "...and renaming the directory's case does not help");
    refuse(R, "/dumps/PPSA00000-app0/prx/anything.prx", ModulePathVerdict::DirectoryNotPermitted,
           "a `prx/` directory some dumps ship is still not on the allowlist");
    refuse(R, "/dumps/PPSA00000-app0/libc.prx", ModulePathVerdict::DirectoryNotPermitted,
           "the dump root holds no linkable module but eboot.bin");
    refuse(R, "/dumps/PPSA00000-app0/Media/Plugins/sub/deep.prx",
           ModulePathVerdict::DirectoryNotPermitted,
           "a SUBdirectory of a permitted directory is not itself permitted");

    printf("-- refused: a Sony library outside sce_module --\n");
    refuse(R, "/dumps/PPSA00000-app0/Media/Plugins/libSceNpEntitlementAccess.prx",
           ModulePathVerdict::SonyLibraryOutsideSceModule,
           "moving the bypass into the auto-linked plugin directory does not smuggle it in");
    refuse(R, "/dumps/PPSA00000-app0/Media/Modules/libSceAppContent.prx",
           ModulePathVerdict::SonyLibraryOutsideSceModule, "same for Media/Modules");
    refuse(R, "/dumps/PPSA00000-app0/Media/Plugins/LIBSCEampr.prx",
           ModulePathVerdict::SonyLibraryOutsideSceModule, "the libSce test is case-insensitive");

    printf("-- the rejection reason names what the file is --\n");
    {
        const auto d = classify_module_path(R, "/dumps/PPSA00000-app0/fakelib/libSceAppContent.sprx");
        CHECK(d.reason.find("entitlement") != std::string::npos ||
              d.reason.find("local inventory") != std::string::npos,
              "a fakelib Sony library is described as an entitlement matter, not just a path miss");
    }

    printf("-- enforce_module_path_policy filters and reports --\n");
    {
        std::vector<LinkInput> in = {
            { std::string(R) + "/eboot.bin", 0x1000 },
            { std::string(R) + "/fakelib/libSceAppContent.sprx", 0x2000 },
            { std::string(R) + "/sce_module/libc.prx", 0x3000 },
            { std::string(R) + "/Media/Plugins/libSceGameUpdate.prx", 0x4000 },
        };
        const auto rejected = enforce_module_path_policy(R, in);
        CHECK(in.size() == 2, "two of four inputs survive");
        CHECK(linked(in, "/eboot.bin") && linked(in, "/sce_module/libc.prx"),
              "the two survivors are the permitted ones");
        CHECK(rejected.size() == 2, "both refusals are reported back to the caller");
        CHECK(rejected.size() == 2 && rejected[0].path.find("fakelib") != std::string::npos,
              "reported in list order, so the log reads like the link list");
        bool all_explained = true;
        for (const auto& r : rejected) if (r.reason.empty()) all_explained = false;
        CHECK(all_explained, "every refusal carries a reason -- a silent drop is the failure mode");
    }

    printf("-- end-to-end through boot_link_inputs --\n");
    {
        const fs::path root = prosper_test::test_scratch_dir() / "prosper_test_module_path_policy";
        std::error_code ec;
        fs::remove_all(root, ec);

        // Deliberately NOT valid SELF images: boot_link_inputs only stats these paths, and using
        // non-images keeps the arm about path policy alone.
        write_file(root / "eboot.bin", "not-a-real-image");
        write_file(root / "Media" / "Plugins" / "RealPlugin.prx", "not-a-real-image");
        write_file(root / "Media" / "Plugins" / "libSceNpEntitlementAccess.prx", "not-a-real-image");
        write_file(root / "fakelib" / "libSceAppContent.sprx", "not-a-real-image");

        const std::vector<LinkInput> in = boot_link_inputs(root.string(), /*verbose=*/false);

        // THE regression arm. Without the policy, #1609's auto-link discovers every .prx in
        // Media/Plugins and links this one -- so this assertion fails on a tree with the guard
        // removed. Verified by doing exactly that before trusting it.
        CHECK(!linked(in, "libSceNpEntitlementAccess"),
              "a Sony library dropped into the auto-linked plugin directory is NOT linked");

        // Positive control for the arm above: the guard must not have simply disabled auto-linking.
        // Without this, deleting the whole #1609 block would also make the arm pass.
        CHECK(linked(in, "RealPlugin.prx"),
              "...while a genuine title plugin in the same directory still IS linked");

        CHECK(linked(in, "eboot.bin"), "the eboot is linked");

        // INVARIANT arm, not a regression arm: this passes with or without the guard, because
        // discovery never reaches fakelib/ today. It exists to fail if discovery ever widens.
        CHECK(!linked(in, "fakelib"), "[invariant] nothing under fakelib/ is linked");

        // Whole-list invariant: whatever the loader decided to link, all of it is permitted.
        bool all_permitted = true;
        for (const auto& e : in)
            if (!classify_module_path(root.string(), e.path).permitted()) {
                printf("         (unpermitted survivor: %s)\n", e.path.c_str());
                all_permitted = false;
            }
        CHECK(all_permitted, "[invariant] every linked path satisfies the policy");

        fs::remove_all(root, ec);
    }

    printf(fails ? "== FAILED (%d) ==\n" : "== passed ==\n", fails);
    return fails ? 1 : 0;
}
