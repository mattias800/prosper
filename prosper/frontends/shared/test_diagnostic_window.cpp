// test_diagnostic_window — the census window `PROSPER_DUMP_PERSISTENT` / `PROSPER_PASS_LOG` open.
//
// What this guards: those two switches take a renderer-callback ORDINAL, and the states worth
// censusing are named in seconds. On Sonic Frontiers (#1968) the ordinal a 360 s route reaches
// (6,560) and the submit number a capture from the same route reports (26,209) differ by 4x, so an
// ordinal estimated from a capture missed the window entirely and the run produced no census at all
// — a silent zero, indistinguishable from "every target was empty". The `ms:` form removes the
// guess; these arms pin both its behaviour and the exact preservation of the ordinal form.
//
// Pure predicate, so no renderer, no device and no clock: `contains()` takes the elapsed time as an
// argument precisely so the latch can be driven backwards, forwards and out of order under test.
#include "diagnostic_window.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); ++fails; } \
                              else         { std::printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    using prosper::frontend::DiagnosticWindow;
    using prosper::frontend::parse_diagnostic_window;

    // --- The ordinal form must be unchanged for every reachable value ---------------------------
    {
        DiagnosticWindow w{parse_diagnostic_window("26000")};
        CHECK(!w.contains(25999, 0), "an ordinal window is closed before its first ordinal");
        CHECK(w.contains(26000, 0) && w.contains(26001, 0) && w.contains(26002, 0),
              "an ordinal window admits exactly its ordinal and the next two");
        CHECK(!w.contains(26003, 0), "an ordinal window closes after three callbacks");
        CHECK(w.contains(26001, 0),
              "an ordinal window is stateless: revisiting an in-window ordinal still answers yes");
    }
    {
        DiagnosticWindow w{parse_diagnostic_window("0x10")};
        CHECK(w.contains(16, 0) && !w.contains(15, 0),
              "the ordinal form keeps strtoull base 0, so 0x10 means 16");
    }
    {
        // Historical behaviour, deliberately preserved: junk parses as ordinal 0 and fires at once.
        DiagnosticWindow w{parse_diagnostic_window("yes")};
        CHECK(w.contains(0, 0) && w.contains(2, 0) && !w.contains(3, 0),
              "unparseable text keeps its historical meaning of ordinal 0, not 'never'");
    }
    {
        DiagnosticWindow w{parse_diagnostic_window(nullptr)};
        CHECK(w.contains(0, 0), "a null value parses as ordinal 0 rather than crashing");
    }

    // --- The time form: latch on the first callback at or after the deadline --------------------
    {
        DiagnosticWindow w{parse_diagnostic_window("ms:240000")};
        CHECK(w.spec().by_time, "the ms: prefix selects the time form");
        CHECK(w.spec().value == 240000, "the ms: prefix parses the milliseconds after it");
        CHECK(!w.contains(100, 0) && !w.contains(4000, 239999),
              "a time window stays closed until the deadline, whatever the ordinal");
        CHECK(w.contains(5000, 240000), "the first callback at or after the deadline opens it");
        CHECK(w.contains(5001, 240016) && w.contains(5002, 240033),
              "the two callbacks after the opening one are in the window");
        CHECK(!w.contains(5003, 240050),
              "the time window is three callbacks wide, not 'the rest of the run'");
        CHECK(!w.contains(9000, 330000),
              "the latch does not re-open later — this is the runaway-census guard");
    }
    {
        // The two PROSPER_PASS_LOG call sites ask about the SAME ordinal (one loads the counter, the
        // other fetch_adds it), so a repeated query must not consume or shift the window.
        DiagnosticWindow w{parse_diagnostic_window("ms:1000")};
        CHECK(w.contains(700, 1000) && w.contains(700, 1000) && w.contains(700, 1001),
              "asking twice about the opening ordinal is idempotent");
        CHECK(w.contains(702, 1002) && !w.contains(703, 1003),
              "the span is counted from the latched ordinal, not from the number of queries");
    }
    {
        DiagnosticWindow w{parse_diagnostic_window("ms:0")};
        CHECK(w.contains(0, 0), "ms:0 opens on the first callback");
    }
    {
        // Span is a parameter so a future call site can ask for a different width; the default is
        // the three callbacks both existing censuses have always used.
        DiagnosticWindow w{parse_diagnostic_window("ms:100", 1)};
        CHECK(w.contains(10, 100) && !w.contains(11, 101), "an explicit span of 1 admits one callback");
    }
    {
        // A very large ordinal must not wrap when the span is added — the reason `contains()`
        // subtracts rather than adding. This is the ONE place the ordinal form's answer differs
        // from the `at >= min && at < min + 3u` test it replaced: that addition wrapped, so the top
        // three uint64 windows could never fire. Unreachable in practice; pinned so the difference
        // is a decision rather than a surprise.
        DiagnosticWindow w{parse_diagnostic_window("18446744073709551615")};
        CHECK(w.contains(UINT64_MAX, 0) && !w.contains(0, 0),
              "an ordinal at the top of the range does not wrap the window open");
    }

    std::printf(fails ? "test_diagnostic_window: %d FAILURE(S)\n" : "test_diagnostic_window: all ok\n",
                fails);
    return fails ? 1 : 0;
}
