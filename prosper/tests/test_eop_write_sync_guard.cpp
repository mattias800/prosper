// test_eop_write_sync_guard — #729: with PROSPER_EOP_WRITE_SYNC=1 the honor_* label writes run
// synchronously, WITHOUT apply_deferred_effect's #449 guest_readable guard. A guest-PM4-supplied
// label address that is unmapped (mis-decoded packet, or freed+decommitted between decode and
// apply) must be skipped, not dereferenced: before the fix each case below SIGSEGVs.
// The mapped-target cases assert the guard introduces no false skips (liveness).
#include "../src/gpu/command_processor.hpp"
#include "../src/gpu/pm4_decode.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static uint32_t PM4(uint32_t len, uint32_t op, uint32_t r) {
    return 0xC0000000u | (((len - 2u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) | ((r & (R_NUM - 1u)) << 2u);
}

// A canonical guest-space address that is certainly unmapped in this test process.
static constexpr uint64_t kUnmapped = 0x5F0000010000ull;

int main() {
    // Must be set before the first honor_* call: eop_write_sync() latches on first use.
    // (MinGW has no setenv; _putenv is the Windows equivalent.)
#ifdef _WIN32
    _putenv((char*)"PROSPER_EOP_WRITE_SYNC=1");
#else
    setenv("PROSPER_EOP_WRITE_SYNC", "1", 1);
#endif
    printf("== test_eop_write_sync_guard ==\n");

    // RELEASE_MEM (data_sel=1) to an unmapped label: the #312 pre-read alone used to fault.
    {
        uint32_t buf[7];
        buf[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
        buf[1] = (uint32_t)(kUnmapped & 0xffffffffu); buf[2] = (uint32_t)(kUnmapped >> 32);
        buf[3] = 1; buf[4] = 1; buf[5] = 0; buf[6] = 0x04;
        GpuState st;
        size_t n = run_command_buffer(buf, 7, st);
        CHECK(n == 1, "sync RELEASE_MEM to an unmapped label is skipped, not dereferenced");
    }
    // EVENT_WRITE (address-carrying) to an unmapped label.
    {
        uint32_t buf[4];
        buf[0] = PM4(4, IT_EVENT_WRITE, 0);
        buf[1] = 0x14;
        buf[2] = (uint32_t)(kUnmapped & 0xffffffffu); buf[3] = (uint32_t)(kUnmapped >> 32);
        GpuState st;
        size_t n = run_command_buffer(buf, 4, st);
        CHECK(n == 1, "sync EVENT_WRITE to an unmapped label is skipped, not dereferenced");
    }
    // WRITE_DATA to an unmapped destination (also exercises the 8-byte pre-read guard).
    {
        uint32_t buf[8];
        buf[0] = PM4(8, IT_NOP, R_WRITE_DATA);
        buf[1] = 0;
        buf[2] = (uint32_t)(kUnmapped & 0xffffffffu); buf[3] = (uint32_t)(kUnmapped >> 32);
        buf[4] = 3; buf[5] = 0x11u; buf[6] = 0x22u; buf[7] = 0x33u;
        GpuState st;
        size_t n = run_command_buffer(buf, 8, st);
        CHECK(n == 1, "sync WRITE_DATA to an unmapped target is skipped, not dereferenced");
    }

    // Liveness: mapped targets must still be written (the guard must not false-skip).
    {
        uint64_t label = 0;
        uint32_t buf[7];
        uint64_t addr = (uint64_t)(uintptr_t)&label;
        buf[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
        buf[1] = (uint32_t)(addr & 0xffffffffu); buf[2] = (uint32_t)(addr >> 32);
        buf[3] = 2; buf[4] = 0xF00DBEEFu; buf[5] = 0x12345678u; buf[6] = 0x04;
        GpuState st; run_command_buffer(buf, 7, st);
        CHECK(label == 0x12345678F00DBEEFull, "sync RELEASE_MEM to a mapped label still writes");
    }
    {
        uint32_t target[3] = {0, 0, 0};
        uint32_t buf[8];
        uint64_t addr = (uint64_t)(uintptr_t)target;
        buf[0] = PM4(8, IT_NOP, R_WRITE_DATA);
        buf[1] = 0;
        buf[2] = (uint32_t)(addr & 0xffffffffu); buf[3] = (uint32_t)(addr >> 32);
        buf[4] = 3; buf[5] = 0x11111111u; buf[6] = 0x22222222u; buf[7] = 0x33333333u;
        GpuState st; run_command_buffer(buf, 8, st);
        CHECK(target[0] == 0x11111111u && target[2] == 0x33333333u,
              "sync WRITE_DATA to a mapped target still writes all dwords");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
