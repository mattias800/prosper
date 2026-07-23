// test_translate_sandbox — the guest-path -> host-path mapping (translate(), exercised via the
// public resolve_guest_path):
//  - #1205: a guest path must not climb OUT of its virtual root (/app0, /temp0, /savedata0). A
//    sub-path containing ".." is lexically normalized; an ABSOLUTE path that escapes above the root
//    is denied. Normal paths and benign in-sandbox ".." are byte-for-byte unaffected.
//  - #1226: sceAvPlayer sources arrive as file:// URLs (UE4:
//    "file://../../../<project>/Content/Movies/x.mp4"). The scheme is stripped; a RELATIVE path's
//    leading ".." clamps at the /app0 root (prosper models no UE4 BaseDir/CWD) instead of being
//    denied as an escape.
//  - #1226: the PS5 filesystem namespace is case-insensitive, so a guest spelling whose case
//    differs from the on-disk dump entry must resolve to the real file on a case-sensitive host
//    (ArcRunner asks for "Content/Movies/...", its dump ships "content/movies/...").
// The app0 root is a real temp tree: translate() probes the host FS for the case-corrected
// fallback. It returns composed paths unchanged whenever the exact-case entry exists (or nothing
// matches), which keeps the byte-exact checks below valid on case-insensitive filesystems too.
#include "../src/hle/dispatch.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

static bool denied(const std::string& r) { return r.rfind("/prosper-denied", 0) == 0; }

int main() {
    std::printf("== test_translate_sandbox ==\n");
    std::error_code ec;
    const fs::path root = fs::temp_directory_path() / "prosper_test_translate_sandbox";
    fs::remove_all(root, ec);                       // stale run debris
    fs::create_directories(root / "data");
    fs::create_directories(root / "arcrunner" / "content" / "movies");   // dump-style lowercase
    { std::ofstream(root / "data" / "config.bin") << "x"; }
    { std::ofstream(root / "arcrunner" / "content" / "movies" / "intro_ps5.mp4") << "x"; }
    const std::string base = root.string();
    set_app0_root(base);

    // ---- #1205: sandbox containment (unchanged behavior) ----

    // Normal path: composed directly under the host root (identical to pre-#1205 behavior).
    CHECK(resolve_guest_path("/app0/data/config.bin") == base + "/data/config.bin",
          "normal /app0 path maps directly under the root");

    // Benign in-sandbox '..': normalizes to the same host file the OS would have resolved anyway.
    CHECK(resolve_guest_path("/app0/data/../data/config.bin") == base + "/data/config.bin",
          "in-sandbox '..' normalizes to the same host file");

    // A filename that merely CONTAINS '..' as a substring is a normal component, not traversal.
    CHECK(resolve_guest_path("/app0/foo..bar") == base + "/foo..bar",
          "'foo..bar' is a normal filename, not a traversal");

    // Escaping traversal: denied, and never composes the escaping host path.
    {
        std::string r = resolve_guest_path("/app0/../escape");
        CHECK(denied(r), "'/app0/../escape' escaping the root is denied");
        CHECK(r.find(base + "/../") == std::string::npos,
              "denied result does not compose the escaping host path");
    }
    CHECK(denied(resolve_guest_path("/app0/../../etc/passwd")), "double-'..' escape denied");
    CHECK(denied(resolve_guest_path("/app0/a/../../b")),        "'..' popping past the root denied");

    // A genuinely absent file keeps its composed exact-case path (ENOENT downstream, as before).
    CHECK(resolve_guest_path("/app0/nope/missing.bin") == base + "/nope/missing.bin",
          "absent file composes unchanged (still fails with ENOENT)");

    // ---- #1226: file:// scheme stripping ----

    // Absolute file:// URL: scheme stripped, then the ordinary /app0 mount translation.
    CHECK(resolve_guest_path("file:///app0/data/config.bin") == base + "/data/config.bin",
          "'file:///app0/...' strips the scheme and maps under the root");

    // The sandbox guard applies to the stripped ABSOLUTE path exactly as without the scheme.
    CHECK(denied(resolve_guest_path("file:///app0/../escape")),
          "'file:///app0/../escape' is still a denied escape");

    // ---- #1226: relative media URLs root at /app0 with leading-'..' clamping ----

    // Plain relative path (pre-existing contract): rooted at /app0.
    CHECK(resolve_guest_path("data/config.bin") == base + "/data/config.bin",
          "bare relative path roots at /app0");

    // The observed UE4 shape: BaseDir-relative '../../..' clamps at the /app0 root instead of
    // being denied (prosper has no BaseDir/CWD model; the ".." count matches the base-dir depth).
    CHECK(resolve_guest_path("file://../../../arcrunner/content/movies/intro_ps5.mp4")
              == base + "/arcrunner/content/movies/intro_ps5.mp4",
          "UE4 'file://../../../<project>/...' clamps to the /app0 package root");

    // Interior '..' still pops normally before clamping applies.
    CHECK(resolve_guest_path("file://a/../../../data/config.bin") == base + "/data/config.bin",
          "interior '..' pops, leading overflow clamps");

    // ---- #1226: case-insensitive fallback against the real on-disk entries ----

    // The ArcRunner scenario end-to-end: mixed-case request, lowercase dump. On a case-sensitive
    // host the result is the corrected real path; on a case-insensitive host the composed spelling
    // already opens the same file — the cross-platform contract is that the result exists.
    {
        std::string r = resolve_guest_path("file://../../../ArcRunner/Content/Movies/INTRO_PS5.MP4");
        CHECK(fs::exists(fs::path(r)), "mixed-case UE4 movie URL resolves to the on-disk file");
    }
    {
        std::string r = resolve_guest_path("/app0/Data/Config.BIN");
        CHECK(fs::exists(fs::path(r)), "mixed-case absolute /app0 path resolves to the on-disk file");
    }

    // Case correction must not invent matches: absent names stay the composed exact-case path.
    CHECK(resolve_guest_path("/app0/Data/Config2.BIN") == base + "/Data/Config2.BIN",
          "no case-insensitive match leaves the composed path unchanged");

    fs::remove_all(root, ec);
    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
