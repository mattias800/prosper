// Strict hex parsing for PROSPER_*_WATCH address lists.
//
// The defect this pins: `strtoull(spec, &end, 0)` accepts a bare DECIMAL token, so
// `PROSPER_RTT_INVALIDATE_WATCH=2063380000` armed a watch on 2,063,380,000 — reported success, matched
// nothing, and produced a confident zero-hit result about an address nobody asked about. It was already
// an instrument trap for PROSPER_TARGET_WATCH, then reproduced by a diagnostic whose own comment claimed
// `0x` was required. Nothing checked the prefix. So the rule is tested, not restated.
#include "gpu/watch_list.hpp"

#include <cstdio>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {
bool rejects(const char* spec) {
    std::vector<uint64_t> out{0xdeadu};   // seeded, so a rejection must also CLEAR it
    const bool ok = parse_hex_watch_list(spec, out);
    return !ok && out.empty();
}
}  // namespace

int main() {
    printf("== test_watch_list ==\n");

    // Accepted: 0x-prefixed, single and comma-separated, either case, with tolerated spaces.
    {
        std::vector<uint64_t> out;
        CHECK(parse_hex_watch_list("0x2063380000", out) && out.size() == 1 &&
                  out[0] == 0x2063380000ull,
              "a single 0x-prefixed address parses to its hex value");

        CHECK(parse_hex_watch_list("0x2063380000,0x20471e0000", out) && out.size() == 2 &&
                  out[0] == 0x2063380000ull && out[1] == 0x20471e0000ull,
              "a comma list parses in order");

        CHECK(parse_hex_watch_list("0X2063380000", out) && out.size() == 1 &&
                  out[0] == 0x2063380000ull,
              "an uppercase 0X prefix is accepted");

        CHECK(parse_hex_watch_list(" 0xAbCd , 0x10 ", out) && out.size() == 2 &&
                  out[0] == 0xabcdull && out[1] == 0x10ull,
              "surrounding spaces are tolerated and hex digits are case-insensitive");

        CHECK(parse_hex_watch_list("0xffffffffffffffff", out) && out.size() == 1 &&
                  out[0] == UINT64_MAX,
              "the largest 64-bit address parses exactly");
    }

    // THE DEFECT ITSELF: a bare decimal token must be REJECTED, not silently reinterpreted. This is the
    // arm whose absence let a watch arm on the wrong address and report success.
    CHECK(rejects("2063380000"),
          "a bare decimal token is rejected, never parsed as decimal");
    CHECK(rejects("0x2063380000,2063380000"),
          "one bad token in a list rejects the WHOLE list rather than arming a partial watch");

    // Malformed shapes.
    CHECK(rejects("0x"),          "a prefix with no digits is rejected");
    CHECK(rejects("0xzz"),        "non-hex digits after the prefix are rejected");
    CHECK(rejects("0x10junk"),    "trailing text after a valid token is rejected");
    CHECK(rejects("0x10 0x20"),   "a space-separated list without commas is rejected");
    CHECK(rejects("0x10,"),       "a trailing comma is rejected");
    CHECK(rejects("0x10,,0x20"),  "a doubled comma is rejected");
    CHECK(rejects(","),           "a lone comma is rejected");
    CHECK(rejects("x10"),         "a prefix without its leading zero is rejected");
    CHECK(rejects("10"),          "a bare hex-looking token without 0x is rejected");
    CHECK(rejects("0x1ffffffffffffffff"), "a value wider than 64 bits is rejected, not truncated");

    // A zero address is never a guest surface: accepting it arms a watch that matches nothing while
    // reporting itself armed, which is the same class of lie as the decimal parse.
    CHECK(rejects("0x0"),        "a zero address is rejected");
    CHECK(rejects("0x00000000"), "a long-form zero address is rejected");
    CHECK(rejects("0x10,0x0"),   "a zero anywhere in the list rejects the list");

    // Empty and absent specs are rejections, not empty successes.
    {
        std::vector<uint64_t> out{0xdeadu};
        CHECK(!parse_hex_watch_list(nullptr, out) && out.empty(), "a null spec is rejected");
        CHECK(!parse_hex_watch_list("", out) && out.empty(), "an empty spec is rejected");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
