// test_savedata_txres — sceSaveDataCreateTransactionResource returns the new resource's ID, and a
// valid ID is POSITIVE. This is not a style preference: Sonic Origins' (PPSA05325) save handler at
// eboot+0x93fdb0 is literally
//     xor edi,edi; call sceSaveDataCreateTransactionResource; test eax,eax; jle <fail>
// so a zero return is read as failure. The handler then records error 3, every later save operation
// short-circuits on that sticky error, and the title's boot coroutine polls the failed job forever —
// which is why PPSA05325 never left its first boot step (#1905).
//
// The assertions below are chosen so each one FAILS against the old `return 0` implementation or
// against a fake that hands out a constant: the positivity check kills `return 0`, the distinctness
// check kills a constant, and the destroy-twice check proves the registry actually records ids
// rather than answering true unconditionally.
#include "../src/hle/dispatch.hpp"
#include <cstdio>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_savedata_txres ==\n");

    const size_t live0 = savedata_tx_resource_live_count();

    // The property the guest's `test eax,eax; jle` demands.
    int32_t a = savedata_tx_resource_create();
    CHECK(a > 0, "create returns a POSITIVE resource id (a zero return reads as failure)");

    // Two live resources must be distinguishable — a constant id would let Delete free the wrong one.
    int32_t b = savedata_tx_resource_create();
    CHECK(b > 0, "a second create is also positive");
    CHECK(a != b, "two live resources get distinct ids");
    CHECK(savedata_tx_resource_live_count() == live0 + 2, "both ids are recorded as live");

    // Delete frees exactly the named id, once.
    CHECK(savedata_tx_resource_destroy(a), "destroy of a created id succeeds");
    CHECK(savedata_tx_resource_live_count() == live0 + 1, "only the named id was freed");
    CHECK(!savedata_tx_resource_destroy(a), "destroying the same id twice fails (it is gone)");
    CHECK(savedata_tx_resource_destroy(b), "the other id is still live and destroys cleanly");
    CHECK(savedata_tx_resource_live_count() == live0, "the registry is back to its starting size");

    // Ids that were never handed out are rejected — including the value the old stub returned.
    CHECK(!savedata_tx_resource_destroy(0), "id 0 was never created, so destroy fails");
    CHECK(!savedata_tx_resource_destroy(-1), "a negative id is rejected");
    CHECK(!savedata_tx_resource_destroy(0x7fffffff), "an unallocated id is rejected");

    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
