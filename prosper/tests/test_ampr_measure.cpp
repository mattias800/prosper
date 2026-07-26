// Regression guard for the libSceAmpr command-size and completion NIDs used by Sonic Origins and
// Pathless. Size queries return byte counts, not status codes. A sizing-only call must remain pure,
// while DOLL's older SDK wrapper passes a registered APR id and relies on its legacy entry point to
// fill the destination.
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
    HleFn write_equeue_320 = Hle::lookup("Zi3dBUjgyXI");
    HleFn append_equeue = Hle::lookup("H896Pt-yB4I");
    HleFn append_equeue_320 = Hle::lookup("o67gODLFpls");
    HleFn add_ampr_event = Hle::lookup("bBfz7kMF2Ho");
    HleFn read_file     = Hle::lookup("vWU-odnS+fU");
    CHECK(write_address && write_equeue && write_equeue_320 && append_equeue && append_equeue_320 &&
              add_ampr_event && read_file,
          "AMPR measure and completion-event NIDs registered");
    CHECK(append_equeue_320 == append_equeue,
          "PS5 3.20 kernel-equeue writer aliases the modeled completion path");
    if (add_ampr_event)
        CHECK(add_ampr_event(0, 0, 0, 0, 0, 0) == 0,
              "AMPR event registration accepts a null queue as a no-op");

    if (write_address) {
        CHECK(write_address(8, 0, 0, 0, 0, 0) == 32,
              "write-address reserves the conservative 32-byte record");
        CHECK(write_address(8, 1ull << 34, 0, 0, 0, 0) == 32,
              "write-address size does not under-allocate for a wide value");
    }
    if (write_equeue)
        CHECK(write_equeue(0, 0, 0, 0, 0, 0) == 20,
              "kernel-equeue command measures 20 bytes");
    if (write_equeue_320)
        CHECK(write_equeue_320(0, 0, 0, 0, 0, 0) == 0x20,
              "PS5 3.20 kernel-equeue command measures a 0x20-byte record");
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
