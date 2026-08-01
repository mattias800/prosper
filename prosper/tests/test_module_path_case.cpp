// test_module_path_case — guards resolve_host_path_case (#1006): the boot preloader must find a
// module whose on-disk casing differs from the hard-coded preload path. The Messenger ships
// "Il2cppUserAssemblies.prx" while Blasphemous 2 / Evergate ship "Il2CppUserAssemblies.prx"; on a
// case-sensitive host filesystem the old exact-case fopen probe silently dropped the present module,
// and the guest's runtime sceKernelLoadStartModule ENOENT then null-jumped (SIGSEGV at rip=0).
// On a case-INSENSITIVE filesystem (NTFS / default APFS) the wrong-case probes below open the real
// file directly, so resolution legitimately returns the input unchanged — both outcomes are accepted
// where they are equivalent, and the "present either way" checks are the cross-platform contract.
#include "../src/host/boot_program.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include "test_scratch.h"

namespace fs = std::filesystem;
using prosper::resolve_host_path_case;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    std::error_code ec;
    const fs::path root = prosper_test::test_scratch_dir() / "prosper_test_module_path_case";
    fs::remove_all(root, ec);                       // stale run debris
    fs::create_directories(root / "Media" / "Modules");
    { std::ofstream(root / "Media" / "Modules" / "Il2CppUserAssemblies.prx") << "x"; }  // uppercase C on disk

    const std::string base = root.string();

    // Exact-case path: exists, returned unchanged.
    const std::string exact = base + "/Media/Modules/Il2CppUserAssemblies.prx";
    CHECK(resolve_host_path_case(exact) == exact, "exact-case path returned unchanged");

    // The #1006 scenario: preload list says lowercase "Il2cpp", disk has uppercase "Il2Cpp".
    // On a case-sensitive FS the result is the corrected real name; on a case-insensitive FS the
    // wrong-case path already opens, so it comes back unchanged. Either way it must reference the
    // real file (fs::exists true) — that is the property boot_program's absence check relies on.
    const std::string wrongBase = base + "/Media/Modules/Il2cppUserAssemblies.prx";
    const std::string r1 = resolve_host_path_case(wrongBase);
    CHECK(fs::exists(fs::path(r1)), "wrong-case basename resolves to a present file");
    CHECK(fs::path(r1).parent_path() == fs::path(wrongBase).parent_path(),
          "resolution stays inside the requested directory");

    // Case mismatch in an intermediate DIRECTORY component as well ("media/modules/...").
    const std::string wrongDirs = base + "/media/modules/IL2CPPUSERASSEMBLIES.PRX";
    const std::string r2 = resolve_host_path_case(wrongDirs);
    CHECK(fs::exists(fs::path(r2)), "wrong-case directory components resolve to the present file");

    // Genuinely absent file: input returned unchanged so the caller's skip path still triggers.
    const std::string missing = base + "/Media/Modules/DoesNotExist.prx";
    CHECK(resolve_host_path_case(missing) == missing, "absent file returned unchanged");

    // Whole directory chain absent: unchanged, no throw.
    const std::string noDir = base + "/NoSuchDir/Nested/Nope.prx";
    CHECK(resolve_host_path_case(noDir) == noDir, "absent directory chain returned unchanged");

    // Name that differs by more than case must NOT match ("Il2cppUserAssemblies2.prx").
    const std::string lenDiff = base + "/Media/Modules/Il2cppUserAssemblies2.prx";
    CHECK(resolve_host_path_case(lenDiff) == lenDiff, "length-differing name is not case-matched");

    // #1236: an absent LEAF under a case-CORRECTED parent chain must keep the corrected ancestors
    // (only the unmatched leaf retains its requested spelling), so an O_CREAT lands in the real
    // directory. Query with wrong-case dirs ("media/MODULES") and an absent leaf against the on-disk
    // "Media/Modules" (created above). On a case-SENSITIVE host the result is the corrected
    // ".../Media/Modules/BrandNew.prx"; on a case-INSENSITIVE host resolve short-circuits (the
    // wrong-case parent already "exists") and returns the input unchanged. The cross-platform
    // contract in BOTH cases: the returned PARENT directory references a real directory, while the
    // leaf stays absent. Before #1236 the case-sensitive path dropped back to the wrong-case parent,
    // whose directory does NOT exist — so this fails without the fix on a case-sensitive FS.
    const std::string corrParent = base + "/media/MODULES/BrandNew.prx";
    const std::string rc = resolve_host_path_case(corrParent);
    CHECK(fs::exists(fs::path(rc).parent_path()),
          "absent leaf: returned parent references a real directory (corrected on case-sensitive hosts)");
    CHECK(!fs::exists(fs::path(rc)),
          "the leaf is still absent, so a read probe still fails ENOENT");

    // Empty input: unchanged, no throw.
    CHECK(resolve_host_path_case("").empty(), "empty path returned unchanged");

    fs::remove_all(root, ec);
    printf(fails ? "test_module_path_case: %d FAILURE(S)\n" : "test_module_path_case: all ok\n", fails);
    return fails ? 1 : 0;
}
