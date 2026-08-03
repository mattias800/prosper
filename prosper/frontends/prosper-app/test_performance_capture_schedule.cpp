#include "performance_capture_schedule.hpp"

#include <cstdio>
#include <limits>

using prosper::frontend::ElapsedPerformanceCaptureTrigger;
using prosper::frontend::parse_performance_capture_delay_ns;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

int main() {
    std::printf("== test_prosper_app_performance_capture_schedule ==\n");
    CHECK(parse_performance_capture_delay_ns(nullptr) == 0,
          "unset delay disables automatic performance capture");
    CHECK(parse_performance_capture_delay_ns("") == 0,
          "empty delay is rejected");
    CHECK(parse_performance_capture_delay_ns("0") == 0,
          "zero delay is rejected");
    CHECK(parse_performance_capture_delay_ns("+1") == 0,
          "signed positive delay is rejected");
    CHECK(parse_performance_capture_delay_ns("-1") == 0,
          "negative delay is rejected");
    CHECK(parse_performance_capture_delay_ns("5000x") == 0,
          "trailing text is rejected");
    CHECK(parse_performance_capture_delay_ns("5000") == 5'000'000'000ull,
          "positive decimal milliseconds convert exactly to nanoseconds");
    CHECK(parse_performance_capture_delay_ns("18446744073709") ==
              18'446'744'073'709'000'000ull,
          "largest whole-millisecond delay is accepted without truncation");
    CHECK(parse_performance_capture_delay_ns("18446744073710") == 0,
          "nanosecond conversion overflow is rejected");
    CHECK(parse_performance_capture_delay_ns("18446744073709551616") == 0,
          "decimal parse overflow is rejected");

    ElapsedPerformanceCaptureTrigger disabled(0);
    CHECK(!disabled.take_if_due(100, std::numeric_limits<uint64_t>::max()),
          "disabled schedule never fires");

    ElapsedPerformanceCaptureTrigger scheduled(500);
    CHECK(!scheduled.take_if_due(1'000, 999),
          "clock reversal cannot fire the automatic trigger");
    CHECK(!scheduled.take_if_due(1'000, 1'499),
          "automatic trigger waits before its elapsed deadline");
    CHECK(scheduled.take_if_due(1'000, 1'500),
          "automatic trigger fires exactly at its elapsed deadline");
    CHECK(!scheduled.take_if_due(1'000, 2'000),
          "automatic trigger remains one-shot after the arm attempt");

    ElapsedPerformanceCaptureTrigger skipped_deadline(500);
    CHECK(skipped_deadline.take_if_due(1'000, 1'900),
          "a delayed app loop consumes the first iteration after the deadline");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
