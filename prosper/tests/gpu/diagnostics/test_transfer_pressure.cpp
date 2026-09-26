// Transfer pressure: host bytes moved per second, reported every run.
//
// The defect class this exists for was found four times in one session and cost a hunt each time,
// because the cost is DIFFUSE -- no call is slow and no profile leaf is damning. What makes it
// obvious is the aggregate: 2,059 MiB/s of host copying on a static splash screen against 4 MiB/s
// on a title that runs at 59 fps. So the properties worth pinning are the ones that decide whether
// anyone ever reads that number.

#include "diagnostics/transfer_pressure.hpp"

#include <cstdio>
#include <cstring>

using namespace prosper::diagnostics;

static int failures = 0;
static void check(bool ok, const char* name) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) failures++;
}

int main() {
    bool names_ok = true;
    for (int i = 0; i < static_cast<int>(Transfer::Count); i++) {
        const char* n = transfer_name(static_cast<Transfer>(i));
        if (!n || !*n || std::strcmp(n, "unknown") == 0) names_ok = false;
    }
    check(names_ok, "every Transfer category has a stable non-placeholder name");

    // Categories must stay separable. Pooling them would report one total, and the live run's
    // value was that it split 100 GiB three ways -- showing the pattern was systemic rather than
    // one bad call site. A single number would have pointed at whichever site happened to be
    // largest and hidden that the other two were the same defect.
    note_transfer(Transfer::StorageMaterialize, 1000);
    note_transfer(Transfer::Detile, 30);
    note_transfer(Transfer::StorageMaterialize, 500);
    check(transfer_bytes(Transfer::StorageMaterialize) == 1500,
          "a category accumulates its own bytes across calls");
    check(transfer_bytes(Transfer::Detile) == 30,
          "a quiet category is not absorbed into a loud one");
    check(transfer_bytes(Transfer::BufferUpload) == 0,
          "a category that never fired reports zero, not a share of the total");

    // A zero-byte call must not count as an event. Sites are called on every binding, including
    // empty ones, and a call count inflated by no-ops would make MiB/call meaningless.
    const uint64_t before = transfer_bytes(Transfer::BufferCompare);
    note_transfer(Transfer::BufferCompare, 0);
    check(transfer_bytes(Transfer::BufferCompare) == before,
          "a zero-byte transfer is not recorded");

    // Out-of-range must be inert rather than corrupting a neighbouring category, since the enum
    // is cast from callers in other layers.
    note_transfer(static_cast<Transfer>(99), 4096);
    check(transfer_bytes(Transfer::StorageMaterialize) == 1500,
          "an out-of-range category does not land in another category's total");

    std::printf("%s\n", failures ? "FAILURES" : "ALL PASS");
    return failures ? 1 : 0;
}
