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

    if (failures) return 1;
    std::printf("== PASS ==\n");
    return 0;
}
