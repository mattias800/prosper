// Transfer pressure: host bytes moved per second, reported every run.
//
// The defect class this exists for was found four times in one session and cost a hunt each time,
// because the cost is DIFFUSE -- no call is slow and no profile leaf is damning. What makes it
// obvious is the aggregate: 2,059 MiB/s of host copying on a static splash screen against 4 MiB/s
// on a title that runs at 59 fps. So the properties worth pinning are the ones that decide whether
// anyone ever reads that number.

#include "diagnostics/transfer_pressure.hpp"

#include <cstdio>
#include <gtest/gtest.h>
#include <cstring>

using namespace prosper::diagnostics;

static void check(bool ok, const char* name) { EXPECT_TRUE(ok) << name; }

TEST(TransferPressure, CategoriesStaySeparable) {
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

    // Calls are counted beside bytes (bytes per call separates "bigger copies" from "more copies",
    // the difference between GTA V's two host-copy regimes, #3926).
    const uint64_t calls_before = transfer_calls(Transfer::RenderTargetSnapshot);
    note_transfer(Transfer::RenderTargetSnapshot, 64);
    note_transfer(Transfer::RenderTargetSnapshot, 0);
    note_transfer(Transfer::RenderTargetSnapshot, 64);
    check(transfer_calls(Transfer::RenderTargetSnapshot) == calls_before + 2,
          "calls count the transfers that carried bytes, not the zero-byte ones");

    // A shared helper reports under its own generic category; a caller whose copy belongs to
    // another reader re-attributes it for the scope, on this thread only, innermost scope winning.
    const uint64_t detile0 = transfer_bytes(Transfer::Detile);
    const uint64_t scanout0 = transfer_bytes(Transfer::GuestScanout);
    const uint64_t upload0 = transfer_bytes(Transfer::BufferUpload);
    {
        const TransferAttributionScope outer(Transfer::GuestScanout);
        note_transfer(Transfer::Detile, 100);
        {
            const TransferAttributionScope inner(Transfer::BufferUpload);
            note_transfer(Transfer::Detile, 7);
        }
        note_transfer(Transfer::Detile, 1);
    }
    note_transfer(Transfer::Detile, 5);
    check(transfer_bytes(Transfer::GuestScanout) == scanout0 + 101 &&
              transfer_bytes(Transfer::BufferUpload) == upload0 + 7,
          "an attribution scope charges the helper's bytes to the scope's category, innermost first");
    check(transfer_bytes(Transfer::Detile) == detile0 + 5,
          "...and the helper's own category sees only what ran outside every scope");

}
