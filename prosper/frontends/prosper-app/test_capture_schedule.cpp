#include "capture_schedule.hpp"

#include <cstdio>
#include <limits>

using prosper::frontend::capture_frame_due;
using prosper::frontend::capture_pad_flip_due;
using prosper::frontend::parse_capture_frame;
using prosper::frontend::parse_capture_pad_flip;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

int main() {
    std::printf("== test_prosper_app_capture_schedule ==\n");
    CHECK(parse_capture_frame(nullptr) == 0, "unset frame disables the trigger");
    CHECK(parse_capture_frame("") == 0, "empty frame disables the trigger");
    CHECK(parse_capture_frame("0") == 0, "frame zero is rejected");
    CHECK(parse_capture_frame("700") == 700, "decimal frame is accepted");
    CHECK(parse_capture_frame("700x") == 0, "trailing text is rejected");
    CHECK(parse_capture_frame("-1") == 0, "negative frame is rejected");
    CHECK(parse_capture_frame("18446744073709551615") ==
              std::numeric_limits<uint64_t>::max(),
          "maximum frame is accepted without truncation");
    CHECK(parse_capture_frame("18446744073709551616") == 0,
          "overflowing frame is rejected");

    CHECK(parse_capture_pad_flip(nullptr) == 0, "unset pad flip disables the trigger");
    CHECK(parse_capture_pad_flip("") == 0, "empty pad flip disables the trigger");
    CHECK(parse_capture_pad_flip("0") == 0, "pad flip zero is rejected");
    CHECK(parse_capture_pad_flip("2828") == 2828, "positive pad flip is accepted");
    CHECK(parse_capture_pad_flip("-1") == 0, "negative pad flip is rejected");
    CHECK(parse_capture_pad_flip("2828x") == 0, "pad flip trailing text is rejected");
    CHECK(parse_capture_pad_flip("9223372036854775807") == std::numeric_limits<int64_t>::max(),
          "maximum signed pad flip is accepted");
    CHECK(parse_capture_pad_flip("9223372036854775808") == 0,
          "pad flip beyond INT64_MAX is rejected");

    CHECK(!capture_frame_due(700, 698, false), "capture waits before its requested frame");
    CHECK(capture_frame_due(700, 699, false), "capture arms at its requested frame");
    CHECK(capture_frame_due(700, 705, false), "a skipped present captures the next real frame");
    CHECK(!capture_frame_due(700, 699, true), "capture remains one-shot after arming");
    CHECK(!capture_frame_due(0, 999, false), "disabled capture never arms");
    CHECK(capture_frame_due(std::numeric_limits<uint64_t>::max(),
                            std::numeric_limits<uint64_t>::max(), false),
          "maximum frame does not overflow the due check");

    CHECK(!capture_pad_flip_due(2828, -1, false),
          "unavailable pad ordinal does not arm a capture");
    CHECK(!capture_pad_flip_due(2828, 2827, false), "pad capture waits before its target");
    CHECK(capture_pad_flip_due(2828, 2828, false), "pad capture arms at its target");
    CHECK(capture_pad_flip_due(2828, 2833, false), "late pad observation arms once");
    CHECK(!capture_pad_flip_due(2828, 2833, true),
          "late pad observation remains consumed after arming");
    CHECK(!capture_pad_flip_due(0, std::numeric_limits<int64_t>::max(), false),
          "disabled pad capture never arms at a large ordinal");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
