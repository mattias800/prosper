// test_agc_dcb — guards the ported AGC "Gen5" Draw Command Buffer HLE (hle_agc.cpp). Sets up a
// synthetic Dcb over a dword buffer, drives the Dcb functions through the NID registry, and asserts
// they build the correct PM4 packets, advance the write cursor, return the packet pointer, and that
// the indirect-register patch helpers modify a previously-returned packet. This validates the port
// independently of the (locale-blocked) boot.
#include "../src/hle/dispatch.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Mirror of hle_agc.cpp's AgcDcb layout (game's own struct; must match byte-for-byte).
struct Dcb {
    uint32_t* bottom; uint32_t* top; uint32_t* cursor_up; uint32_t* cursor_down;
    void* callback; void* user_data; uint32_t reserved_dw; uint32_t pad;
};

// PM4 header the functions should emit (matches hle_agc.cpp's PM4()).
static uint32_t PM4(uint32_t len, uint32_t op, uint32_t r) {
    return 0xC0000000u | (((len - 2u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) | ((r & 0x3fu) << 2u);
}
constexpr uint32_t IT_NOP = 0x10, IT_SET_SH_REG = 0x76, R_DRAW_RESET = 0x05, R_CX_REGS_INDIRECT = 0x12;

struct ShaderRegister { uint32_t offset, value; };

int main() {
    printf("== test_agc_dcb ==\n");
    register_builtin_hle();

    auto reset  = Hle::lookup("TRO721eVt4g");   // GraphicsDcbResetQueue
    auto setcx  = Hle::lookup("ZvwO9euwYzc");   // GraphicsDcbSetCxRegistersIndirect
    auto p_add  = Hle::lookup("d-6uF9sZDIU");   // SetCxRegIndirectPatchAddRegisters
    auto p_addr = Hle::lookup("vcmNN+AAXnY");   // SetCxRegIndirectPatchSetAddress
    CHECK(reset && setcx && p_add && p_addr, "AGC Dcb functions registered (override the glog stubs)");
    if (!(reset && setcx && p_add && p_addr)) { printf("== FAIL ==\n"); return 1; }

    uint32_t buffer[256];
    memset(buffer, 0xEE, sizeof buffer);
    Dcb dcb{};
    dcb.bottom = buffer; dcb.top = buffer + 256; dcb.cursor_up = buffer; dcb.cursor_down = buffer + 256;
    dcb.callback = nullptr; dcb.user_data = nullptr; dcb.reserved_dw = 0;
    auto D = (uint64_t)(uintptr_t)&dcb;

    // ResetQueue(dcb, op=0x3ff, state=0) -> 2-dw R_DRAW_RESET, returns the packet, cursor += 2.
    uint64_t r0 = reset(D, 0x3ff, 0, 0, 0, 0);
    CHECK(r0 == (uint64_t)(uintptr_t)buffer, "ResetQueue returns the allocated packet ptr (== buffer start)");
    CHECK(buffer[0] == PM4(2, IT_NOP, R_DRAW_RESET), "ResetQueue wrote R_DRAW_RESET PM4 header");
    CHECK(buffer[1] == 0, "ResetQueue wrote cmd[1]=0");
    CHECK(dcb.cursor_up == buffer + 2, "ResetQueue advanced the cursor by 2 dwords");

    // SetCxRegistersIndirect(dcb, regs=0x1122334455667788, num_regs=3) -> 4-dw packet.
    uint64_t regs = 0x1122334455667788ull;
    uint64_t rc = setcx(D, regs, 3, 0, 0, 0);
    auto* cmd = (uint32_t*)(uintptr_t)rc;
    CHECK(cmd == buffer + 2, "SetCx returns the next packet ptr");
    CHECK(cmd[0] == PM4(4, IT_NOP, R_CX_REGS_INDIRECT), "SetCx wrote R_CX_REGS_INDIRECT PM4 header");
    CHECK(cmd[1] == 3, "SetCx wrote num_regs=3");
    CHECK(cmd[2] == 0x55667788u && cmd[3] == 0x11223344u, "SetCx wrote regs vaddr lo/hi");
    CHECK(dcb.cursor_up == buffer + 6, "SetCx advanced the cursor by 4 dwords");

    // Patch helpers modify the returned packet (the old stub returned 0 -> these wrote through null).
    p_add(rc, 5, 0, 0, 0, 0);
    CHECK(cmd[1] == 8, "PatchAddRegisters did cmd[1] += 5 (3 -> 8)");
    p_addr(rc, 0xAABBCCDD00112233ull, 0, 0, 0, 0);
    CHECK(cmd[2] == 0x00112233u && cmd[3] == 0xAABBCCDDu, "PatchSetAddress rewrote cmd[2]/cmd[3]");

    // sceAgcGetDataPacketPayloadAddress (#140): resolves a data packet to its register-bank payload.
    // Type 0 is NOP header + metadata; type 1 is register header + offset. Both payloads start +2.
    // Null pointers and unknown packet types are invalid. Verify the exact *addr contract.
    auto getpayload = Hle::lookup("V++UgBtQhn0");
    CHECK(getpayload != nullptr, "GetDataPacketPayloadAddress registered");
    if (getpayload) {
        uint32_t pkt[8]; uint32_t* out = nullptr;
        CHECK(getpayload((uint64_t)(uintptr_t)&out, (uint64_t)(uintptr_t)pkt, /*type*/1, 0, 0, 0) == 0 &&
              out == pkt + 2, "type 1: *addr = cmd+2, rc 0");
        out = nullptr;
        CHECK(getpayload((uint64_t)(uintptr_t)&out, (uint64_t)(uintptr_t)pkt, /*type*/0, 0, 0, 0) == 0 &&
              out == pkt + 2, "type 0: *addr = cmd+2, rc 0");
        out = nullptr;
        CHECK(getpayload((uint64_t)(uintptr_t)&out, (uint64_t)(uintptr_t)pkt, /*type*/2, 0, 0, 0) != 0 &&
              out == nullptr, "unknown type: invalid-arg and out pointer remains untouched");
        CHECK(getpayload(0, (uint64_t)(uintptr_t)pkt, 1, 0, 0, 0) != 0, "null out-ptr -> invalid-arg");
        CHECK(getpayload((uint64_t)(uintptr_t)&out, 0, 1, 0, 0, 0) != 0, "null cmd -> invalid-arg");
    }

    // sceAgcCbSetShRegistersDirect (#574): group adjacent offsets, but preserve caller order and
    // split gaps into separate packets so every {offset,value} pair reaches the SH register file.
    auto setshdirect = Hle::lookup("UZbQjYAwwXM");
    CHECK(setshdirect != nullptr, "SetShRegistersDirect registered");
    if (setshdirect) {
        ShaderRegister sh[] = {
            {0x20c, 0x11111111u}, {0x20d, 0x22222222u},
            {0x212, 0x33333333u}, {0x100, 0x44444444u}, {0x101, 0x55555555u},
        };
        uint32_t* before = dcb.cursor_up;
        uint64_t first_addr = setshdirect(D, (uint64_t)(uintptr_t)sh, 5, 0, 0, 0);
        auto* first = (uint32_t*)(uintptr_t)first_addr;
        CHECK(first == before, "SetShRegistersDirect returns the first emitted packet");
        CHECK(first[0] == PM4(4, IT_SET_SH_REG, 0) && first[1] == 0x20c &&
              first[2] == 0x11111111u && first[3] == 0x22222222u,
              "first consecutive SH-register run emitted as one packet");
        auto* second = first + 4;
        CHECK(second[0] == PM4(3, IT_SET_SH_REG, 0) && second[1] == 0x212 &&
              second[2] == 0x33333333u, "register gap starts a new SH packet");
        auto* third = second + 3;
        CHECK(third[0] == PM4(4, IT_SET_SH_REG, 0) && third[1] == 0x100 &&
              third[2] == 0x44444444u && third[3] == 0x55555555u,
              "out-of-order consecutive run preserves caller order");
        CHECK(dcb.cursor_up == before + 11, "SetShRegistersDirect advances by all packet dwords");

        before = dcb.cursor_up;
        CHECK(setshdirect(D, 0, 5, 0, 0, 0) == 0 && dcb.cursor_up == before,
              "null register array rejected without allocating");
        CHECK(setshdirect(D, (uint64_t)(uintptr_t)sh, 0, 0, 0, 0) == 0 && dcb.cursor_up == before,
              "zero register count rejected without allocating");
        CHECK(setshdirect(D, (uint64_t)(uintptr_t)sh, 4097, 0, 0, 0) == 0 && dcb.cursor_up == before,
              "oversized register count rejected without allocating");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
