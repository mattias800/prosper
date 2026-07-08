// test_pm4_decode — validates the PM4 command-stream decoder (src/gpu/pm4_decode.cpp), the front of
// the CommandProcessor. It builds a real Draw Command Buffer by driving the AGC Dcb functions (via
// the NID registry, exactly as the guest would), then decodes the resulting dword stream and asserts
// every packet is recognized with the correct operands. This exercises the decoder end-to-end against
// the real emitters (hle_agc.cpp) with no fabricated data, independent of the (SDK-gated) boot.
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/pm4_decode.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper;
using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Mirror of hle_agc.cpp's AgcDcb layout (must match byte-for-byte).
struct Dcb {
    uint32_t* bottom; uint32_t* top; uint32_t* cursor_up; uint32_t* cursor_down;
    void* callback; void* user_data; uint32_t reserved_dw; uint32_t pad;
};

int main() {
    printf("== test_pm4_decode ==\n");
    register_builtin_hle();

    auto reset = Hle::lookup("TRO721eVt4g");   // ResetQueue
    auto idx   = Hle::lookup("GIIW2J37e70");   // SetIndexSize
    auto setcx = Hle::lookup("ZvwO9euwYzc");   // SetCxRegistersIndirect
    auto p_add = Hle::lookup("d-6uF9sZDIU");   // patch: AddRegisters
    auto p_adr = Hle::lookup("vcmNN+AAXnY");   // patch: SetAddress
    auto draw  = Hle::lookup("Yw0jKSqop+E");   // DrawIndexAuto
    auto drawi = Hle::lookup("q88lQ+GP5Yk");   // DrawIndex (indexed sibling)
    auto evt   = Hle::lookup("aJf+j5yntiU");   // EventWrite
    CHECK(reset && idx && setcx && p_add && p_adr && draw && drawi && evt, "AGC Dcb builders registered");
    if (!(reset && idx && setcx && p_add && p_adr && draw && drawi && evt)) { printf("== FAIL ==\n"); return 1; }

    uint32_t buffer[256];
    memset(buffer, 0, sizeof buffer);          // zero => any tail dword is a non-type3 stream end
    Dcb dcb{};
    dcb.bottom = buffer; dcb.top = buffer + 256; dcb.cursor_up = buffer; dcb.cursor_down = buffer + 256;
    auto D = (uint64_t)(uintptr_t)&dcb;

    // Build a representative frame: reset -> index type -> set Cx regs (patched) -> draw -> event.
    reset(D, 0x3ff, 0, 0, 0, 0);
    idx(D, /*index_size*/ 2, 0, 0, 0, 0);
    uint64_t rc = setcx(D, /*regs vaddr*/ 0x1122334455667788ull, /*num*/ 3, 0, 0, 0);
    p_add(rc, 5, 0, 0, 0, 0);                                     // num_regs: 3 -> 8
    p_adr(rc, 0xCAFEF00DDEADBEEFull, 0, 0, 0, 0);                 // rewrite the regs vaddr
    draw(D, /*index_count*/ 0x1234, 0, 0, 0, 0);
    drawi(D, /*index_count*/ 0x0600, /*indices*/ 0xDEAD0000BEEF0040ull, /*modifier*/ 0x40000000ull, 0, 0);
    evt(D, /*event_type*/ 0x42, 0, 0, 0, 0);

    size_t used_dw = (size_t)(dcb.cursor_up - dcb.bottom);
    printf("  built %zu dwords\n", used_dw);

    std::vector<Pm4Command> ops;
    size_t consumed = decode_pm4(buffer, 256, ops);   // pass full buffer; decoder stops at the zero tail
    CHECK(consumed == used_dw, "decoder consumed exactly the built dwords (stops at zero pad)");
    CHECK(ops.size() == 6, "decoded 6 packets");
    if (ops.size() != 6) { printf("== FAIL: got %zu packets ==\n", ops.size()); return 1; }

    using K = Pm4Command::Kind;
    CHECK(ops[0].kind == K::DrawReset, "op0 = DrawReset");

    CHECK(ops[1].kind == K::SetIndexType && ops[1].index_size == 2, "op1 = SetIndexType(size=2)");

    CHECK(ops[2].kind == K::SetRegsIndirect, "op2 = SetRegsIndirect");
    CHECK(ops[2].reg_class == RegClass::Cx, "op2 reg_class = Cx");
    CHECK(ops[2].num_regs == 8, "op2 num_regs = 8 (3 + patched 5)");
    CHECK(ops[2].regs_vaddr == 0xCAFEF00DDEADBEEFull, "op2 regs_vaddr = patched address");

    CHECK(ops[3].kind == K::DrawIndexAuto && ops[3].index_count == 0x1234, "op3 = DrawIndexAuto(0x1234)");

    // DrawIndex round-trip (issue #63): the builder wrote [1]=count, [2..3]=index addr, [4..5]=modifier;
    // the decoder must hand every field back (indexed draws were previously dropped as unknown NOPs).
    CHECK(ops[4].kind == K::DrawIndex && ops[4].index_count == 0x0600, "op4 = DrawIndex(count=0x600)");
    CHECK(ops[4].di_index_addr == 0xDEAD0000BEEF0040ull, "op4 index-buffer address round-trips");
    CHECK(ops[4].di_modifier == 0x40000000ull && ops[4].di_valid, "op4 modifier round-trips (valid)");

    CHECK(ops[5].kind == K::EventWrite && ops[5].event_type == 0x42, "op5 = EventWrite(0x42)");

    // Every decoded packet's len must match its header, and the payload pointer must be in-buffer.
    bool spans_ok = true;
    for (auto& c : ops) {
        if (c.len < 1 || c.len > 16) spans_ok = false;
        if (c.len > 1 && (c.payload < buffer || c.payload + (c.len - 1) > buffer + 256)) spans_ok = false;
    }
    CHECK(spans_ok, "all packet lengths/payload spans are in-bounds");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
