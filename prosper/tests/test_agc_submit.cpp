// test_agc_submit — end-to-end guard for the AGC submit path: build a Draw Command Buffer through
// the real Dcb HLE functions (as the game does), submit it via sceAgcDriverSubmitDcb (UglJIZjGssM),
// and assert the CommandProcessor folded the stream into a GpuState — registers written, draw
// recorded, submit/draw stats updated. This locks in the "first real command buffer executes"
// milestone programmatically (previously only observed in a boot log line, which VERIFICATION.md
// says not to rely on). It exercises the exact front-half -> back-half handoff: Dcb build -> PM4 ->
// decode -> GpuState.
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/command_processor.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper;

// Front-half stats exposed by hle_agc.cpp.
extern "C" void prosper_agc_submit_stats(uint64_t* submits, uint64_t* draws);

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Game's Dcb struct (must match hle_agc.cpp's AgcDcb byte-for-byte).
struct Dcb {
    uint32_t* bottom; uint32_t* top; uint32_t* cursor_up; uint32_t* cursor_down;
    void* callback; void* user_data; uint32_t reserved_dw; uint32_t pad;
};
// The AgcDriver submit descriptor (Kyty Gen5Driver::Packet).
struct Packet { uint32_t* addr; uint32_t dw_num; uint8_t pad[4]; };

int main() {
    printf("== test_agc_submit ==\n");
    register_builtin_hle();

    auto reset  = Hle::lookup("TRO721eVt4g");   // GraphicsDcbResetQueue
    auto setcx  = Hle::lookup("ZvwO9euwYzc");   // GraphicsDcbSetCxRegistersIndirect
    auto setidx = Hle::lookup("GIIW2J37e70");   // GraphicsDcbSetIndexSize
    auto draw   = Hle::lookup("Yw0jKSqop+E");   // GraphicsDcbDrawIndexAuto
    auto submit = Hle::lookup("UglJIZjGssM");   // GraphicsDriverSubmitDcb
    auto regmem = Hle::lookup("AOLcoIkQDgM");   // QueryResourceRegistrationUserMemoryRequirements
    auto maxname = Hle::lookup("uJziRsODk1c");   // sceAgcDriverGetResourceRegistrationMaxNameLength
    CHECK(reset && setcx && setidx && draw && submit && regmem && maxname,
          "AGC Dcb, submit, and resource-registration functions registered");
    if (!(reset && setcx && setidx && draw && submit && regmem && maxname)) { printf("== FAIL ==\n"); return 1; }

    uint64_t registration_bytes = 0x7a2a67fe6900ull;
    CHECK(regmem((uint64_t)(uintptr_t)&registration_bytes, 0x2000, 0x38, 0, 0, 0) == 0,
          "resource-registration memory query returned OK");
    CHECK(registration_bytes == 0x80800,
          "resource-registration memory query replaced the poisoned size deterministically");
    CHECK((registration_bytes & 0x3f) == 0 && registration_bytes < 0x100000,
          "resource-registration memory requirement is aligned and bounded");
    CHECK((int64_t)regmem(0, 0x2000, 0x38, 0, 0, 0) < 0,
          "resource-registration memory query rejects a null output");

    uint32_t max_name_length = 0xdeadbeefu;
    CHECK(maxname((uint64_t)(uintptr_t)&max_name_length, 0, 0, 0, 0, 0) == 0,
          "resource-registration max-name query returned OK");
    CHECK(max_name_length == 0xfc,
          "resource-registration max-name query initialized the poisoned output");
    CHECK((int64_t)maxname(0, 0, 0, 0, 0, 0) < 0,
          "resource-registration max-name query rejects a null output");

    // Baseline stats (other tests in-process may have submitted; measure deltas).
    uint64_t s0 = 0, d0 = 0; prosper_agc_submit_stats(&s0, &d0);

    // Build a command buffer exactly as the game would: the register-indirect packet carries a
    // pointer to a real {offset,value} array, which the CommandProcessor reads back at submit time.
    static uint32_t buffer[256];
    memset(buffer, 0, sizeof buffer);
    Dcb dcb{};
    dcb.bottom = buffer; dcb.top = buffer + 256; dcb.cursor_up = buffer; dcb.cursor_down = buffer + 256;
    auto D = (uint64_t)(uintptr_t)&dcb;

    struct Reg { uint32_t offset, value; };
    static Reg cx_regs[] = { {0x10, 0xAAAA}, {0x11, 0xBBBB}, {0x12, 0xCCCC} };

    reset(D, 0x3ff, 0, 0, 0, 0);
    setcx(D, (uint64_t)(uintptr_t)cx_regs, 3, 0, 0, 0);
    setidx(D, 1 /*index_size*/, 0, 0, 0, 0);
    draw(D, 42 /*index_count*/, 0, 0, 0, 0);

    uint32_t dw = (uint32_t)(dcb.cursor_up - buffer);
    CHECK(dw > 0, "Dcb build advanced the write cursor");

    // Submit it through the real driver entrypoint.
    Packet pkt{ buffer, dw, {0,0,0,0} };
    uint64_t rc = submit((uint64_t)(uintptr_t)&pkt, 0, 0, 0, 0, 0);
    CHECK(rc == 0, "SubmitDcb returned OK (0)");

    uint64_t s1 = 0, d1 = 0; prosper_agc_submit_stats(&s1, &d1);
    CHECK(s1 == s0 + 1, "submit count incremented by 1");
    CHECK(d1 == d0 + 1, "draw count incremented by 1 (one DrawIndexAuto)");

    // Replay the same buffer through the CommandProcessor directly to inspect the folded GpuState
    // (SubmitDcb uses a private state internally; this asserts the same decode produces the right
    // register file + draw — the exact transformation the submit performed).
    {
        gpu::GpuState st;
        size_t packets = gpu::run_command_buffer(buffer, dw, st);
        CHECK(packets == 4, "CommandProcessor decoded 4 packets (reset/setcx/setidx/draw)");
        CHECK(st.cx[0x10] == 0xAAAA && st.cx[0x11] == 0xBBBB && st.cx[0x12] == 0xCCCC,
              "folded the indirect Cx registers (0x10/0x11/0x12) from the pointed-to array");
        CHECK(st.index_type == 1, "folded the index type");
        CHECK(st.draws.size() == 1 && st.draws[0].index_count == 42,
              "recorded one DrawIndexAuto with index_count=42");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
