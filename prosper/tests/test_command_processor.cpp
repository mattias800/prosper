// test_command_processor — validates the CommandProcessor's apply stage (src/gpu/command_processor).
// Builds a Draw Command Buffer via the real AGC Dcb functions, with a Set*RegistersIndirect packet
// pointing at a real {offset,value} array, then replays the stream into a GpuState and asserts the
// register files, index type, and draw list. Exercises decode+apply end-to-end against the real
// emitters; independent of the (SDK-gated) boot.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/gpu/command_processor.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper;
using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

struct Dcb {   // mirror of hle_agc.cpp's AgcDcb
    uint32_t* bottom; uint32_t* top; uint32_t* cursor_up; uint32_t* cursor_down;
    void* callback; void* user_data; uint32_t reserved_dw; uint32_t pad;
};

int main() {
    printf("== test_command_processor ==\n");
    register_builtin_hle();

    auto reset = Hle::lookup("TRO721eVt4g");   // ResetQueue
    auto idx   = Hle::lookup("GIIW2J37e70");   // SetIndexSize
    auto setcx = Hle::lookup("ZvwO9euwYzc");   // SetCxRegistersIndirect
    auto setsh = Hle::lookup("-HOOCn0JY48");   // SetShRegistersIndirect
    auto draw  = Hle::lookup("Yw0jKSqop+E");   // DrawIndexAuto
    CHECK(reset && idx && setcx && setsh && draw, "AGC Dcb builders registered");
    if (!(reset && idx && setcx && setsh && draw)) { printf("== FAIL ==\n"); return 1; }

    uint32_t buffer[256];
    memset(buffer, 0, sizeof buffer);
    Dcb dcb{};
    dcb.bottom = buffer; dcb.top = buffer + 256; dcb.cursor_up = buffer; dcb.cursor_down = buffer + 256;
    auto D = (uint64_t)(uintptr_t)&dcb;

    // Real register arrays the guest would build (offset, value pairs). Their addresses go into the
    // indirect packets; the CommandProcessor reads them 1:1 (guest VA == host addr in prosper).
    ShaderReg cx_regs[3] = { {0xA318u, 0x11111111u}, {0x00C0u, 0x22222222u}, {0x02DEu, 0x33333333u} };
    ShaderReg sh_regs[2] = { {0x2C0Cu, 0xAAAAAAAAu}, {0x2C0Du, 0xBBBBBBBBu} };

    reset(D, 0x3ff, 0, 0, 0, 0);
    idx(D, 2, 0, 0, 0, 0);
    setcx(D, (uint64_t)(uintptr_t)cx_regs, 3, 0, 0, 0);
    setsh(D, (uint64_t)(uintptr_t)sh_regs, 2, 0, 0, 0);
    draw(D, 0x0300, 0, 0, 0, 0);
    draw(D, 0x0006, 0, 0, 0, 0);

    GpuState st;
    size_t n = run_command_buffer(buffer, 256, st);
    printf("  applied %zu packets\n", n);

    // Cx register file: the three offsets set to their values.
    CHECK(st.cx.size() == 3, "cx file has 3 registers");
    CHECK(st.cx.count(0xA318u) && st.cx[0xA318u] == 0x11111111u, "cx[0xA318] = 0x11111111");
    CHECK(st.cx.count(0x00C0u) && st.cx[0x00C0u] == 0x22222222u, "cx[0x00C0] = 0x22222222");
    CHECK(st.cx.count(0x02DEu) && st.cx[0x02DEu] == 0x33333333u, "cx[0x02DE] = 0x33333333");

    // Sh register file: two registers; must NOT leak into cx/uc.
    CHECK(st.sh.size() == 2, "sh file has 2 registers");
    CHECK(st.sh.count(0x2C0Cu) && st.sh[0x2C0Cu] == 0xAAAAAAAAu, "sh[0x2C0C] = 0xAAAAAAAA");
    CHECK(st.sh.count(0x2C0Du) && st.sh[0x2C0Du] == 0xBBBBBBBBu, "sh[0x2C0D] = 0xBBBBBBBB");
    CHECK(st.uc.empty(), "uc file untouched");

    CHECK(st.index_type == 2, "index_type = 2 (from SetIndexSize)");

    CHECK(st.draws.size() == 2, "2 draws recorded");
    CHECK(st.draws.size() == 2 && st.draws[0].index_count == 0x0300, "draw0 index_count = 0x300");
    CHECK(st.draws.size() == 2 && st.draws[1].index_count == 0x0006, "draw1 index_count = 0x6");

    // SET_SH_REG (IT_SET_SH_REG=0x76) sets a RANGE: payload[0]=start offset, payload[1..]=consecutive
    // values. The driver uploads the whole user-data descriptor block this way, so the decode+apply MUST
    // write every value — regressing to "first register only" silently drops the shaders' V#/T# SGPRs.
    {
        // Build a type-3 SET_SH_REG packet: header + [offset, v0,v1,v2,v3,v4] (5 registers @ 0x100).
        const uint32_t M = 6;                                   // payload dwords (offset + 5 values)
        uint32_t pkt[1 + M];
        pkt[0] = 0xC0000000u | ((M - 1u) << 16) | (0x76u << 8); // type3, len, op=IT_SET_SH_REG
        pkt[1] = 0x100u;                                        // start register offset
        for (uint32_t k = 0; k < 5; k++) pkt[2 + k] = 0xD0000000u + k;
        GpuState s2;
        run_command_buffer(pkt, 1 + M, s2);
        CHECK(s2.sh.size() == 5, "SET_SH_REG range set all 5 registers (not just the first)");
        bool all = true;
        for (uint32_t k = 0; k < 5; k++) all &= (s2.sh.count(0x100u + k) && s2.sh[0x100u + k] == 0xD0000000u + k);
        CHECK(all, "SET_SH_REG range values sh[0x100..0x104] are the consecutive payload values");
    }

    // In-stream flip (sceAgcDcbSetFlip -> R_FLIP): the game flips via the Dcb, not the sceVideoOut
    // API, so the CommandProcessor MUST perform the videoout flip when it reaches the packet —
    // GetFlipStatus's count/flipArg/currentBuffer are what Unity's frame pacer polls before building
    // the next frame; a dropped in-stream flip stalls the game at one rendered frame.
    {
        auto setflip    = Hle::lookup("YUeqkyT7mEQ");                            // sceAgcDcbSetFlip
        auto flipstatus = Hle::lookup(nid_hash("sceVideoOutGetFlipStatus"));
        CHECK(setflip && flipstatus, "SetFlip builder + GetFlipStatus registered");
        if (setflip && flipstatus) {
            uint8_t st_before[0x40], st_after[0x40];
            flipstatus(0x1001, (uint64_t)(uintptr_t)st_before, 0, 0, 0, 0);
            uint32_t fbuf[64]; memset(fbuf, 0, sizeof fbuf);
            Dcb fd{}; fd.bottom = fbuf; fd.top = fbuf + 64; fd.cursor_up = fbuf; fd.cursor_down = fbuf + 64;
            setflip((uint64_t)(uintptr_t)&fd, 0x1001, 1, 2, 0x1234567890abcdefull, 0);
            GpuState s3;
            run_command_buffer(fbuf, 64, s3);
            flipstatus(0x1001, (uint64_t)(uintptr_t)st_after, 0, 0, 0, 0);
            uint64_t cnt_b = *(uint64_t*)(st_before + 0x00), cnt_a = *(uint64_t*)(st_after + 0x00);
            int64_t  arg_a = *(int64_t*)(st_after + 0x18);
            int32_t  buf_a = *(int32_t*)(st_after + 0x38);
            CHECK(cnt_a == cnt_b + 1, "in-stream Flip advances GetFlipStatus.count");
            CHECK(arg_a == (int64_t)0x1234567890abcdefll, "in-stream Flip publishes its 64-bit flipArg");
            CHECK(buf_a == 1, "in-stream Flip publishes its buffer index");
        }
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
