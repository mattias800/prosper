// test_eop_write — Stage B of docs/GPU_EXECUTOR_DESIGN.md: the CommandProcessor honors the Dcb's
// memory-side effects. Because our fold is synchronous (submit == pipe drain), a RELEASE_MEM /
// EVENT_WRITE_EOP label write and a WRITE_DATA are performed at apply time — correct end-of-pipe
// semantics, not a shim. This hand-builds the exact packet layout hle_agc.cpp's builders emit (the
// builders themselves read SysV stack args via __builtin_frame_address, only valid under the loaded
// game) and asserts run_command_buffer writes the right bytes to the target address.
#include "../src/gpu/command_processor.hpp"
#include "../src/gpu/pm4_decode.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// PM4 type-3 NOP-wrapped header, identical to hle_agc.cpp PM4(): len = total dwords incl. header.
static uint32_t PM4(uint32_t len, uint32_t op, uint32_t r) {
    return 0xC0000000u | (((len - 2u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) | ((r & (R_NUM - 1u)) << 2u);
}

int main() {
    printf("== test_eop_write ==\n");

    // A RELEASE_MEM writing a 64-bit fence value (data_sel==2) to a label, exactly as agc_cb_release_mem
    // lays it out: [0]=hdr [1..2]=addr [3]=data_sel [4..5]=value lo/hi [6]=action.
    {
        uint64_t label = 0;
        uint32_t buf[7];
        uint64_t addr = (uint64_t)(uintptr_t)&label;
        buf[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
        buf[1] = (uint32_t)(addr & 0xffffffffu); buf[2] = (uint32_t)(addr >> 32);
        buf[3] = 2;                                   // data_sel = Data64
        buf[4] = 0xF00DBEEFu; buf[5] = 0x12345678u;   // value = 0x12345678F00DBEEF
        buf[6] = 0x04;                                // event action
        GpuState st;
        size_t n = run_command_buffer(buf, 7, st);
        CHECK(n == 1, "RELEASE_MEM decoded as one packet");
        CHECK(label == 0x12345678F00DBEEFull, "RELEASE_MEM (data_sel=2) wrote the 64-bit fence value to the label");
    }

    // data_sel==1 -> 32-bit write; the upper 32 bits of the label must be untouched.
    {
        uint64_t label = 0xAAAAAAAAAAAAAAAAull;
        uint32_t buf[7];
        uint64_t addr = (uint64_t)(uintptr_t)&label;
        buf[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
        buf[1] = (uint32_t)(addr & 0xffffffffu); buf[2] = (uint32_t)(addr >> 32);
        buf[3] = 1;                                   // data_sel = Data32Low
        buf[4] = 0x0000CAFEu; buf[5] = 0;
        buf[6] = 0x04;
        GpuState st; run_command_buffer(buf, 7, st);
        CHECK((label & 0xffffffffu) == 0x0000CAFEu, "RELEASE_MEM (data_sel=1) wrote the low 32 bits");
        CHECK((label >> 32) == 0xAAAAAAAAu, "RELEASE_MEM (data_sel=1) left the upper 32 bits untouched");
    }

    // data_sel==3 -> monotonic GPU clock: two successive writes must not decrease.
    {
        uint64_t l1 = 0, l2 = 0;
        auto build = [](uint32_t* buf, uint64_t addr) {
            buf[0] = PM4(7, IT_NOP, R_RELEASE_MEM);
            buf[1] = (uint32_t)(addr & 0xffffffffu); buf[2] = (uint32_t)(addr >> 32);
            buf[3] = 3; buf[4] = 0; buf[5] = 0; buf[6] = 0x04;
        };
        uint32_t b1[7], b2[7];
        build(b1, (uint64_t)(uintptr_t)&l1); build(b2, (uint64_t)(uintptr_t)&l2);
        GpuState st; run_command_buffer(b1, 7, st); run_command_buffer(b2, 7, st);
        CHECK(l1 != 0 && l2 >= l1, "RELEASE_MEM (data_sel=3) wrote a monotonic non-zero GPU clock");
    }

    // WRITE_DATA: [0]=hdr [1]=dst [2..3]=addr [4]=num_dwords [5..]=data.
    {
        uint32_t target[3] = {0, 0, 0};
        uint32_t buf[8];
        uint64_t addr = (uint64_t)(uintptr_t)target;
        buf[0] = PM4(8, IT_NOP, R_WRITE_DATA);
        buf[1] = 0;                                   // dst
        buf[2] = (uint32_t)(addr & 0xffffffffu); buf[3] = (uint32_t)(addr >> 32);
        buf[4] = 3;                                   // num_dwords
        buf[5] = 0x11111111u; buf[6] = 0x22222222u; buf[7] = 0x33333333u;
        GpuState st; run_command_buffer(buf, 8, st);
        CHECK(target[0] == 0x11111111u && target[1] == 0x22222222u && target[2] == 0x33333333u,
              "WRITE_DATA copied all 3 inline dwords to the destination");
    }

    // A WRITE_DATA claiming more dwords than the packet holds must clamp (never read past the packet).
    {
        uint32_t target[4] = {0, 0, 0, 0};
        uint32_t buf[7];
        uint64_t addr = (uint64_t)(uintptr_t)target;
        buf[0] = PM4(7, IT_NOP, R_WRITE_DATA);
        buf[1] = 0; buf[2] = (uint32_t)(addr & 0xffffffffu); buf[3] = (uint32_t)(addr >> 32);
        buf[4] = 99;                                  // lies: claims 99 dwords, only 2 present (buf[5],buf[6])
        buf[5] = 0xDEADu; buf[6] = 0xBEEFu;
        GpuState st; run_command_buffer(buf, 7, st);
        CHECK(target[0] == 0xDEADu && target[1] == 0xBEEFu && target[2] == 0 && target[3] == 0,
              "WRITE_DATA clamped num_dwords to what the packet actually holds");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
