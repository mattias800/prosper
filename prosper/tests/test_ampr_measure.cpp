// Regression guard for the libSceAmpr command-size NIDs used by Sonic Origins. These functions
// return byte counts, not status codes: treating MeasureReadFile as a real read returned EINVAL for
// a sizing-only call, and the guest used 0x80020016 as an allocation size.
#include "../src/hle/dispatch.hpp"
#include <cstdint>
#include <cstdio>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

int main() {
    std::printf("== test_ampr_measure ==\n");
    register_builtin_hle();

    HleFn write_address = Hle::lookup("4fgtGfXDrFc");
    HleFn write_equeue  = Hle::lookup("sSAUCCU1dv4");
    HleFn read_file     = Hle::lookup("vWU-odnS+fU");
    CHECK(write_address && write_equeue && read_file, "AMPR measure NIDs registered");

    if (write_address) {
        CHECK(write_address(8, 0, 0, 0, 0, 0) == 8, "compact zero immediate measures 8 bytes");
        CHECK(write_address(8, 0x1234567, 0, 0, 0, 0) == 12,
              "compact nonzero immediate measures 12 bytes");
        CHECK(write_address(8, 1ull << 34, 0, 0, 0, 0) == 16,
              "wide immediate measures 16 bytes");
    }
    if (write_equeue)
        CHECK(write_equeue(0, 0, 0, 0, 0, 0) == 20, "kernel-equeue command measures 20 bytes");
    if (read_file) {
        CHECK(read_file(0, 0, UINT32_MAX, 0xffffffffffull, 0, 0) == 24,
              "wide-offset read command measures 24 bytes (never EINVAL)");
        CHECK(read_file(0, 0, 0, 0xffffffffull, 0, 0) == 20,
              "32-bit-offset read command measures 20 bytes");
    }

    std::printf("== %s ==\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
