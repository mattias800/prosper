// test_savedata_dirname — a guest save dirName must be a single directory component under the host
// save root. savedata_dirname_ok is the shared guard for savedata0_mount and savedata0_dir_mtime, so
// a crafted/garbled name (empty, "." / "..", or one with an embedded separator) can never traverse
// out of the sandbox to stat/create a directory elsewhere. Pure predicate — no filesystem.
#include "../src/hle/dispatch.hpp"
#include <cstdio>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_savedata_dirname ==\n");

    // Accepted: a normal single-component save directory name.
    CHECK(savedata_dirname_ok("SAVEDATA00"), "a normal save dir name is accepted");
    CHECK(savedata_dirname_ok("save_slot_1"), "underscore/digit name accepted");

    // Rejected: empty, current/parent dir, and any embedded path separator (traversal).
    CHECK(!savedata_dirname_ok(""),           "empty name rejected");
    CHECK(!savedata_dirname_ok("."),          "'.' rejected");
    CHECK(!savedata_dirname_ok(".."),         "'..' rejected");
    CHECK(!savedata_dirname_ok("../foo"),     "'../foo' parent traversal rejected");
    CHECK(!savedata_dirname_ok("a/b"),        "forward-slash component rejected");
    CHECK(!savedata_dirname_ok("a\\b"),       "backslash component rejected");
    CHECK(!savedata_dirname_ok("foo/../bar"), "embedded '..' via slash rejected");
    CHECK(!savedata_dirname_ok("/etc/passwd"),"absolute path rejected");

    // Call-site integration: savedata0_mount must apply the guard BEFORE composing save0_base()+name,
    // so a traversal name is rejected without ever stat()/mkdir()ing a host directory outside the
    // sandbox. Create policy is discriminating: without the guard, "../escape" would be mkdir'd and
    // return Created; with it, the name is rejected up front -> NotFound and nothing is created.
    CHECK(savedata0_mount("../escape", SaveDataMountPolicy::Create) == SaveDataMountOutcome::NotFound,
          "savedata0_mount('../escape', Create) -> NotFound (no host dir created outside the sandbox)");
    CHECK(savedata0_mount("a/b", SaveDataMountPolicy::OpenOrCreate) == SaveDataMountOutcome::NotFound,
          "savedata0_mount('a/b', OpenOrCreate) -> NotFound (embedded separator rejected)");

    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
