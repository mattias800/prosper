// test_import_data_seed — the initial contents of an unresolved Sony DATA import (#3529).
//
// Pure: no dump, no mmap, no guest. It pins the ONE variable prosper answers today
// (`__stack_chk_guard`) and, just as importantly, that it answers nothing else — a seeder that
// wrote into every slot would satisfy the first assertion and destroy the honest zero every other
// unresolved variable depends on.
#include "hle/dispatch/import_data_seed.hpp"
#include "hle/dispatch/nid.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    printf("== test_import_data_seed ==\n");

    // The two documented properties of the canary, checked on the constant itself so a future edit
    // to the value cannot quietly drop either.
    CHECK(kStackGuardValue != 0, "the seeded stack guard is non-zero");
    CHECK((kStackGuardValue & 0xffu) == 0,
          "the seeded stack guard's low byte is the terminator NUL");

    // A slot is a page of zeroes when install_import_data hands it over; the sentinel below stands
    // in for "not zero", so a seeder that writes nothing is distinguishable from one that writes 0.
    constexpr uint8_t kSentinel = 0xa5;
    uint8_t slot[64];

    memset(slot, kSentinel, sizeof slot);
    const std::string guard = nid_hash("__stack_chk_guard");
    CHECK(seed_import_data(guard, slot, sizeof slot), "__stack_chk_guard is seeded");
    uint64_t got = 0; memcpy(&got, slot, sizeof got);
    CHECK(got == kStackGuardValue, "__stack_chk_guard's slot holds exactly the seeded value");
    // Only the object itself is written -- the rest of the slot's page belongs to nothing.
    bool tail_intact = true;
    for (size_t i = sizeof(uint64_t); i < sizeof slot; i++)
        if (slot[i] != kSentinel) tail_intact = false;
    CHECK(tail_intact, "seeding writes 8 bytes and does not clobber the rest of the slot");

    // The negative arm. Without it the case above would pass against a seeder that ignores the NID
    // entirely and stamps the canary into every data slot in the program.
    memset(slot, kSentinel, sizeof slot);
    const std::string other = nid_hash("__progname");
    CHECK(!seed_import_data(other, slot, sizeof slot), "an unknown DATA import is NOT seeded");
    bool untouched = true;
    for (size_t i = 0; i < sizeof slot; i++) if (slot[i] != kSentinel) untouched = false;
    CHECK(untouched, "an unknown DATA import's slot is left exactly as it was");

    // A slot smaller than the object must be refused outright rather than half-written: a truncated
    // canary is a value the guest would compare against a full one.
    memset(slot, kSentinel, sizeof slot);
    CHECK(!seed_import_data(guard, slot, 4), "a slot too small for the value is refused");
    untouched = true;
    for (size_t i = 0; i < sizeof slot; i++) if (slot[i] != kSentinel) untouched = false;
    CHECK(untouched, "a refused seed writes nothing at all");

    CHECK(!seed_import_data(guard, nullptr, 8), "a null slot is refused rather than dereferenced");

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
