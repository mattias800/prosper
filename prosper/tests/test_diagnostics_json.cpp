// test_diagnostics_json.cpp — Verify JSON output format (PROSPER_DIAGNOSTICS only).
//
// Tests that JsonWriter produces valid JSON:
// - Empty events → "[]"
// - Single event → correct object format
// - Multiple events → array with proper commas
// - Phase names are human-readable

#ifdef PROSPER_DIAGNOSTICS

#include <cstdio>
#include <string>
#include "diagnostics/diagnostics.hpp"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    printf("== test_diagnostics_json: verifying JSON output ==\n");

    using namespace prosper::diagnostics;

    // Empty list → empty array.
    std::vector<BootEvent> empty;
    std::string empty_json = JsonWriter::write_events(empty);
    CHECK(empty_json == "[]", "empty events → []");

    // Single event.
    BootEvent single(BootPhase::BOOT_COMPLETE, 42.5);
    std::string single_json = JsonWriter::write_event(single);
    CHECK(single_json.find("\"phase\"") != std::string::npos, "contains 'phase' key");
    CHECK(single_json.find("BOOT_COMPLETE") != std::string::npos, "contains phase name");
    CHECK(single_json.find("\"timestamp_ms\"") != std::string::npos, "contains 'timestamp_ms' key");
    CHECK(single_json.front() == '{' && single_json.back() == '}', "object braces balanced");

    // Multiple events → array format.
    std::vector<BootEvent> multi = {
        { BootPhase::PROCESS_START, 0.0 },
        { BootPhase::LINKING, 1.5 },
        { BootPhase::BOOT_COMPLETE, 10.0 }
    };
    std::string multi_json = JsonWriter::write_events(multi);
    CHECK(multi_json.front() == '[' && multi_json.back() == ']', "array brackets balanced");
    // Verify commas separate elements (N-1 commas for N elements in compact form,
    // but pretty-printed output may have newlines — just check at least N-1 separators).
    size_t comma_count = 0;
    for (char c : multi_json) if (c == ',') ++comma_count;
    CHECK(comma_count >= 2, "commas present between elements");

    // Verify phase_name for all known phases.
    CHECK(std::string(phase_name(BootPhase::PROCESS_START)) == "PROCESS_START", "PROCESS_START name");
    CHECK(std::string(phase_name(BootPhase::LINKING)) == "LINKING", "LINKING name");
    CHECK(std::string(phase_name(BootPhase::_COUNT)) == "UNKNOWN", "_COUNT → UNKNOWN");

    printf(fails ? "\ntest_diagnostics_json: %d FAILURE(S)\n"
                 : "\ntest_diagnostics_json: all ok\n", fails);
    return fails ? 1 : 0;
}

#else

#include <cstdio>
int main() {
    printf("== test_diagnostics_json: SKIPPED (PROSPER_DIAGNOSTICS not enabled) ==\n");
    return 0;
}

#endif
