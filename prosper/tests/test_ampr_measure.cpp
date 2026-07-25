// Regression guard for the libSceAmpr command-size NIDs used by Sonic Origins. These functions
// return byte counts, not status codes. A sizing-only call must remain pure, while DOLL's older SDK
// wrapper passes a registered APR id and relies on the same entry point to fill its destination.
#include "../src/hle/dispatch.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

namespace prosper {
uint32_t prosper_apr_register(const std::string& path, uint64_t size);
void prosper_apr_reset_for_test();
}

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
        CHECK(write_address(8, 0, 0, 0, 0, 0) == 32,
              "write-address reserves the conservative 32-byte record");
        CHECK(write_address(8, 1ull << 34, 0, 0, 0, 0) == 32,
              "write-address size does not under-allocate for a wide value");
    }
    if (write_equeue)
        CHECK(write_equeue(0, 0, 0, 0, 0, 0) == 20,
              "kernel-equeue command measures 20 bytes");
    if (read_file) {
        CHECK(read_file(0, 0, UINT32_MAX, 0xffffffffffull, 0, 0) == 24,
              "wide-offset read command measures 24 bytes (never EINVAL)");
        CHECK(read_file(0, 0, 0, 0xffffffffull, 0, 0) == 20,
              "32-bit-offset read command measures 20 bytes");

        const char* fixture_path = "prosper-test-ampr-measure.tmp";
        const std::array<uint8_t, 4> fixture = {0x10, 0x20, 0x30, 0x40};
        bool fixture_written = false;
        if (FILE* file = std::fopen(fixture_path, "wb")) {
            const size_t written = std::fwrite(fixture.data(), 1, fixture.size(), file);
            const bool closed = std::fclose(file) == 0;
            fixture_written = written == fixture.size() && closed;
        }
        prosper_apr_reset_for_test();
        std::array<uint8_t, 4> destination = {0xA1, 0xB2, 0xC3, 0xD4};
        const auto before = destination;
        CHECK(read_file(0, (uint64_t)(uintptr_t)destination.data(), destination.size(),
                        0, 0, 0) == 20 &&
                  destination == before,
              "unregistered read-file measurement is a pure query");
        const uint32_t id = prosper_apr_register(fixture_path, fixture.size());
        CHECK(fixture_written &&
                  read_file(id, (uint64_t)(uintptr_t)destination.data(), destination.size(),
                            0, 0, 0) == 20 &&
                  destination == fixture,
              "registered read-file measurement preserves the DOLL compatibility read");
        prosper_apr_reset_for_test();
        std::remove(fixture_path);
    }

    std::printf("== %s ==\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
