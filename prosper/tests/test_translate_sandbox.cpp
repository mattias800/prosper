// test_translate_sandbox — #1205: the guest-path -> host-path mapping (translate(), exercised via the
// public resolve_guest_path) must not let a guest path climb OUT of its virtual root (/app0, /temp0,
// /savedata0). A sub-path containing ".." is lexically normalized; one that escapes above the root is
// denied. Normal paths and benign in-sandbox ".." are byte-for-byte unaffected (a benign ".." resolves
// to the same host file it always would). No real filesystem access is needed — translate() is pure.
#include "../src/hle/dispatch.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

static bool denied(const std::string& r) { return r.rfind("/prosper-denied", 0) == 0; }

int main() {
    // g_app0 is cached on first use, so set the root before any translate() call.
    // setenv is POSIX-only (MinGW-w64 lacks it); use _putenv_s on Windows.
#ifdef _WIN32
    _putenv_s("PROSPER_APP0", "/hostroot/app0");
#else
    setenv("PROSPER_APP0", "/hostroot/app0", 1);
#endif
    std::printf("== test_translate_sandbox ==\n");

    // Normal path: composed directly under the host root (identical to pre-#1205 behavior).
    CHECK(resolve_guest_path("/app0/foo/bar") == "/hostroot/app0/foo/bar",
          "normal /app0 path maps directly under the root");

    // Benign in-sandbox '..': normalizes to the same host file the OS would have resolved anyway.
    CHECK(resolve_guest_path("/app0/data/../config") == "/hostroot/app0/config",
          "in-sandbox '..' normalizes to <root>/config");

    // A filename that merely CONTAINS '..' as a substring is a normal component, not traversal.
    CHECK(resolve_guest_path("/app0/foo..bar") == "/hostroot/app0/foo..bar",
          "'foo..bar' is a normal filename, not a traversal");

    // Escaping traversal: denied, and never composes the escaping host path.
    {
        std::string r = resolve_guest_path("/app0/../escape");
        CHECK(denied(r), "'/app0/../escape' escaping the root is denied");
        CHECK(r.find("/hostroot/app0/../") == std::string::npos,
              "denied result does not compose the escaping host path");
    }
    CHECK(denied(resolve_guest_path("/app0/../../etc/passwd")), "double-'..' escape denied");
    CHECK(denied(resolve_guest_path("/app0/a/../../b")),        "'..' popping past the root denied");

    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
