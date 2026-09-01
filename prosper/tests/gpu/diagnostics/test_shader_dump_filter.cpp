// PROSPER_SHADER_DUMP_PROGRAM (#3196) — the parse/match half, tested without any GPU, any
// environment, or any filesystem. The end-to-end behaviour (filename, withholding, failing open) is
// pinned separately in tests/gpu/execute/test_shader_recompile_cache.cpp; this file exists so a
// regression in the SELECTOR is not diagnosed as a regression in the dump.

#include "gpu/diagnostics/shader_dump_filter.hpp"

#include <cstdio>

using prosper::gpu::ShaderDumpProgramFilter;
using Result = prosper::gpu::ShaderDumpProgramFilter::ConfigureResult;

static int failures = 0;
#define CHECK(condition, message) do { \
    if (condition) std::printf("  [ok]   %s\n", message); \
    else { std::printf("  [FAIL] %s\n", message); ++failures; } \
} while (0)

int main() {
    std::printf("== test_shader_dump_filter ==\n");

    ShaderDumpProgramFilter filter;
    CHECK(filter.configure(nullptr) == Result::Unset && !filter.armed() &&
              filter.allows(0x5008efd00ull) && filter.allows(0),
          "unset arms nothing and allows every program, including an unknown address");
    CHECK(filter.configure("") == Result::Unset && !filter.armed(),
          "an empty value is unset, not malformed");

    CHECK(filter.configure("0x5008efd00") == Result::Armed && filter.armed() &&
              filter.size() == 1,
          "a single 0x address arms the filter");
    CHECK(filter.allows(0x5008efd00ull) && !filter.allows(0x5008f1400ull),
          "an armed filter admits only the address it names");
    CHECK(!filter.allows(0), "an armed filter never admits a program with no address");
    CHECK(filter.allows(0x5008f1400ull, 0x5008efd00ull),
          "naming either half of a vertex chain selects the pair");
    CHECK(!filter.allows(0x5008f1400ull, 0x5008f1500ull),
          "a chain whose halves are both unnamed stays withheld");

    CHECK(filter.configure(" 0x10 , 0x20,0x10 ") == Result::Armed && filter.size() == 2 &&
              filter.allows(0x10) && filter.allows(0x20) && !filter.allows(0x30),
          "a comma list tolerates spaces and collapses duplicates");

    // The strict parser is `watch_list.hpp`'s, and this is why: `strtoull(spec, &end, 0)` reads an
    // unprefixed token as DECIMAL, so `5008efd00` would arm on 5,008 and the run would report itself
    // armed on an address nobody asked about. Every one of these must arm NOTHING.
    const char* malformed[] = {
        "5008efd00",            // no 0x prefix -- the trap the strict parser exists for
        "0x",                   // prefix with no digits
        "0xzz",                 // not hex
        "0x10,",                // trailing comma: an empty final token
        "0x10,,0x20",           // doubled comma
        "0x10 junk",            // trailing junk after a valid token
        "0x0",                  // zero is never a guest program address
        "0x10000000000000000",  // overflows 64 bits
    };
    bool all_rejected = true;
    for (const char* spec : malformed) {
        filter.configure("0x1234");  // armed first, so a no-op configure cannot pass by accident
        if (filter.configure(spec) != Result::Malformed || filter.armed() ||
            !filter.allows(0x5008efd00ull))
            all_rejected = false;
    }
    CHECK(all_rejected,
          "every malformed spec reports Malformed, arms nothing, and fails OPEN (dumps everything)");

    // Failing open is the deliberate asymmetry with the SKIP selectors, and it is worth its own
    // assertion: an empty dump directory reads as "that program never compiled", which is a false
    // negative wearing the clothes of evidence.
    filter.configure("nonsense");
    CHECK(filter.allows(0x1) && filter.allows(0x2) && !filter.armed(),
          "a disarmed filter is not a filter that blocks everything");

    filter.configure("0x1234");
    const auto first = filter.note_withheld();
    const auto second = filter.note_withheld();
    const auto third = filter.note_withheld();
    CHECK(first.ordinal == 1 && first.print && second.ordinal == 2 && second.print &&
              third.ordinal == 3 && !third.print && filter.withheld_total() == 3,
          "withholdings carry a 1-based ordinal and print first, then powers of two");
    filter.configure("0x1234");
    CHECK(filter.withheld_total() == 0, "re-arming resets the withheld count");

    if (failures) {
        std::printf("== FAIL: %d ==\n", failures);
        return 1;
    }
    std::printf("== PASS ==\n");
    return 0;
}
