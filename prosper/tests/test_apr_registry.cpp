// test_apr_registry — guards the APR container registry and its size-keyed read matching
// (issue #62). The APR read-submit handler identifies WHICH file to read by total byte size,
// because the captured read-request object carries no legible file id. That key is only sound
// when unambiguous, so this locks in: stable 1-based id assignment, path-dedup on re-resolve
// (a duplicate entry would make every read of that file look ambiguous), and the match counts
// the read path uses to refuse ambiguous / unplaceable (chunk) reads instead of guessing.
#include <cstdio>
#include <cstdint>
#include <string>

namespace prosper {
    uint32_t    prosper_apr_register(const std::string& path, uint64_t size);
    void        prosper_apr_reset_for_test();
    int         prosper_apr_match_by_size(uint64_t size, std::string* out_path);
    std::string prosper_apr_path_for_id(uint32_t id);
}
using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    printf("== test_apr_registry ==\n");
    prosper_apr_reset_for_test();

    // Stable 1-based ids in resolve order.
    uint32_t a = prosper_apr_register("/dump/global.utoc", 645);
    uint32_t b = prosper_apr_register("/dump/global.ucas", 100000);
    CHECK(a == 1 && b == 2, "ids assigned 1-based in resolve order");
    CHECK(prosper_apr_path_for_id(1) == "/dump/global.utoc", "path_for_id(1)");
    CHECK(prosper_apr_path_for_id(2) == "/dump/global.ucas", "path_for_id(2)");
    CHECK(prosper_apr_path_for_id(0).empty() && prosper_apr_path_for_id(3).empty(),
          "out-of-range ids resolve to empty");

    // Unambiguous size keying (the only case the read path may serve).
    std::string p;
    CHECK(prosper_apr_match_by_size(645, &p) == 1 && p == "/dump/global.utoc",
          "unique size matches exactly one container");

    // A count that equals no container's TOTAL size (i.e. a partial/chunk read) must not match.
    CHECK(prosper_apr_match_by_size(644, nullptr) == 0, "chunk-sized count matches nothing");

    // Two containers sharing a size: the read path must see the ambiguity (count > 1).
    uint32_t c = prosper_apr_register("/dump/pakchunk0-ps5.utoc", 645);
    CHECK(c == 3, "distinct path gets a new id even at a colliding size");
    CHECK(prosper_apr_match_by_size(645, &p) == 2, "size collision reported as 2 matches");

    // Re-resolving the same host path must return the SAME id (no duplicate entry), so a game
    // that resolves a container twice doesn't turn every read of it ambiguous.
    uint32_t a2 = prosper_apr_register("/dump/global.utoc", 645);
    CHECK(a2 == a, "re-resolving a path returns its existing id");
    CHECK(prosper_apr_match_by_size(645, &p) == 2, "re-resolve does not add a duplicate entry");

    // Re-resolve with a changed size updates the record (single source of truth per path).
    prosper_apr_register("/dump/global.ucas", 200000);
    CHECK(prosper_apr_match_by_size(100000, nullptr) == 0 &&
          prosper_apr_match_by_size(200000, &p) == 1 && p == "/dump/global.ucas",
          "re-resolve updates the stored size");

    prosper_apr_reset_for_test();
    CHECK(prosper_apr_match_by_size(645, nullptr) == 0 && prosper_apr_path_for_id(1).empty(),
          "reset clears the registry");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
