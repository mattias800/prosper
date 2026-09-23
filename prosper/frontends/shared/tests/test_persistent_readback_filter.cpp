#include "shared/diagnostics/persistent_readback_filter.hpp"

#include <cstdio>
#include <initializer_list>

using namespace prosper::frontend;

static int failures = 0;
#define CHECK(condition, message) do { if (!(condition)) { std::printf("FAIL: %s\n", message); ++failures; } } while (0)

int main() {
    const auto broad = parse_persistent_readback_filter(nullptr);
    CHECK(broad.state == PersistentReadbackFilterState::All &&
          broad.allows(0x3080030000ull) && broad.allows(0x9fc2000000ull),
          "no selector retains the existing broad census");

    const auto selected = parse_persistent_readback_filter(
        "0x3080030000,0x9fc2000000");
    CHECK(selected.state == PersistentReadbackFilterState::Selected &&
          selected.addresses.size() == 2 &&
          selected.allows(0x3080030000ull) && selected.allows(0x9fc2000000ull) &&
          !selected.allows(0x3084070000ull),
          "the two requested targets pass and an unrequested target is excluded");

    for (const char* malformed : {"", "3080030000", "0x3080030000,0x", "0x0"}) {
        const auto refused = parse_persistent_readback_filter(malformed);
        CHECK(refused.state == PersistentReadbackFilterState::Malformed &&
              refused.addresses.empty() && !refused.allows(0x3080030000ull),
              "a malformed selector refuses every readback");
    }

    const auto oversized = parse_persistent_readback_filter(
        "0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9");
    CHECK(oversized.state == PersistentReadbackFilterState::TooMany &&
          oversized.addresses.empty() && !oversized.allows(0x3080030000ull),
          "more than eight addresses refuse every readback");

    uint64_t bytes = 123;
    PersistentReadbackBudget exact_budget;
    CHECK(exact_budget.admit(8192, 8192, 4, bytes) == PersistentReadbackCharge::Admitted &&
          bytes == PersistentReadbackBudget::kMaxBytes &&
          exact_budget.charged_bytes == PersistentReadbackBudget::kMaxBytes,
          "an exactly 256 MiB selected readback is admitted and charged");
    CHECK(exact_budget.admit(1, 1, 1, bytes) == PersistentReadbackCharge::OverBudget &&
          bytes == 0 && exact_budget.charged_bytes == PersistentReadbackBudget::kMaxBytes,
          "one byte after the exact boundary is refused without changing the charge");

    PersistentReadbackBudget cumulative_budget;
    CHECK(cumulative_budget.admit(4096, 8192, 4, bytes) == PersistentReadbackCharge::Admitted &&
          bytes == PersistentReadbackBudget::kMaxBytes / 2,
          "the first half-budget target is admitted");
    CHECK(cumulative_budget.admit(4096, 8192, 4, bytes) == PersistentReadbackCharge::Admitted &&
          bytes == PersistentReadbackBudget::kMaxBytes / 2 &&
          cumulative_budget.charged_bytes == PersistentReadbackBudget::kMaxBytes,
          "the second half-budget target reaches the exact cumulative limit");
    CHECK(cumulative_budget.admit(1, 1, 4, bytes) == PersistentReadbackCharge::OverBudget &&
          bytes == 0,
          "a later target beyond the cumulative limit is refused");

    PersistentReadbackBudget overflow_budget;
    CHECK(overflow_budget.admit(UINT32_MAX, UINT32_MAX, 2, bytes) ==
              PersistentReadbackCharge::SizeOverflow &&
          bytes == 0 && overflow_budget.charged_bytes == 0,
          "width times height times pixel stride overflow refuses without charging");
    CHECK(overflow_budget.admit(1, 1, 0, bytes) == PersistentReadbackCharge::InvalidSize &&
          bytes == 0 && overflow_budget.charged_bytes == 0,
          "unknown zero-stride format refuses without charging");

    if (failures) return 1;
    std::printf("== PASS ==\n");
    return 0;
}
