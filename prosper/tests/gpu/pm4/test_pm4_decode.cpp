// test_pm4_decode — validates the PM4 command-stream decoder (src/gpu/pm4/pm4_decode.cpp), the front of
// the CommandProcessor. It builds a real Draw Command Buffer by driving the AGC Dcb functions (via
// the NID registry, exactly as the guest would), then decodes the resulting dword stream and asserts
// every packet is recognized with the correct operands. This exercises the decoder end-to-end against
// the real emitters (hle_agc.cpp) with no fabricated data, independent of the (SDK-gated) boot.
#include "hle/dispatch/dispatch.hpp"
#include <gtest/gtest.h>
#include "gpu/pm4/pm4_decode.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper;
using namespace prosper::gpu;

#define CHECK(c, m) EXPECT_TRUE(c) << (m)

// Mirror of hle_agc.cpp's AgcDcb layout (must match byte-for-byte).
struct Dcb {
    uint32_t* bottom; uint32_t* top; uint32_t* cursor_up; uint32_t* cursor_down;
    void* callback; void* user_data; uint32_t reserved_dw; uint32_t pad;
};

static uint32_t PM4(uint32_t len, uint32_t op, uint32_t r) {
    return 0xC0000000u | (((len - 2u) & 0x3fffu) << 16u) |
           ((op & 0xffu) << 8u) | ((r & (R_NUM - 1u)) << 2u);
}

TEST(Pm4Decode, Contract) {
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
    auto push  = Hle::lookup("+kSrjIVxKFE");   // PushMarker (#641)
    auto pop   = Hle::lookup("H7uZqCoNuWk");   // PopMarker (#641)
    auto instances = Hle::lookup("tSBxhAPyytQ"); // SetNumInstances
    auto setcx_direct = Hle::lookup("LHFXRrlTPD8"); // SetCxRegisterDirect (#395 F5)
    auto setsh_direct = Hle::lookup("pFLArOT53+w"); // SetShRegisterDirect (#395 F5)
    auto setuc_direct = Hle::lookup("w4-d0n60hdo"); // SetUcRegisterDirect (#395 F5)
    auto set_indirect_base = Hle::lookup("RmaJwLtc8rY"); // SetBaseIndirectArgs
    auto stall_parser = Hle::lookup("u2T2DiA5hRI");      // StallCommandBufferParser
    auto draw_indirect = Hle::lookup("t1vNu082-jM");     // DrawIndexIndirect
    auto draw_nonindexed_indirect = Hle::lookup("1q1titRBL6o"); // DrawIndirect (#2929)
    auto dispatch_indirect = Hle::lookup("CtB+A9-VxO0"); // DispatchIndirect
    CHECK(reset && idx && setcx && p_add && p_adr && draw && drawi && evt && push && pop && instances &&
          setcx_direct && setsh_direct && setuc_direct && set_indirect_base && stall_parser &&
          draw_indirect && dispatch_indirect && draw_nonindexed_indirect,
          "AGC Dcb builders registered");
    if (!(reset && idx && setcx && p_add && p_adr && draw && drawi && evt && push && pop && instances &&
          setcx_direct && setsh_direct && setuc_direct && set_indirect_base && stall_parser &&
          draw_indirect && dispatch_indirect && draw_nonindexed_indirect)) {
        printf("== FAIL ==\n"); FAIL() << "legacy early exit";
    }

    uint32_t buffer[256];
    memset(buffer, 0, sizeof buffer);          // zero => any tail dword is a non-type3 stream end
    Dcb dcb{};
    dcb.bottom = buffer; dcb.top = buffer + 256; dcb.cursor_up = buffer; dcb.cursor_down = buffer + 256;
    auto D = (uint64_t)(uintptr_t)&dcb;

    // Build a representative frame: marker scope -> reset -> index type -> set Cx regs -> draws -> event.
    const char marker[] = "test frame";
    push(D, (uint64_t)(uintptr_t)marker, 0, 0, 0, 0);
    reset(D, 0x3ff, 0, 0, 0, 0);
    idx(D, /*index_size*/ 2, 0, 0, 0, 0);
    uint64_t rc = setcx(D, /*regs vaddr*/ 0x1122334455667788ull, /*num*/ 3, 0, 0, 0);
    p_add(rc, 5, 0, 0, 0, 0);                                     // num_regs: 3 -> 8
    p_adr(rc, 0xCAFEF00DDEADBEEFull, 0, 0, 0, 0);                 // rewrite the regs vaddr
    auto pack_reg = [](uint32_t offset, uint32_t value) {
        return (uint64_t)offset | ((uint64_t)value << 32u);
    };
    setcx_direct(D, pack_reg(0x100, 0x11111111), 0, 0, 0, 0);
    setsh_direct(D, pack_reg(0x200, 0x22222222), 0, 0, 0, 0);
    setuc_direct(D, pack_reg(0x300, 0x33333333), 0, 0, 0, 0);
    instances(D, 7, 0, 0, 0, 0);
    draw(D, /*index_count*/ 0x1234, /*modifier*/ 0x1122334455667788ull, 0, 0, 0);
    drawi(D, /*index_count*/ 0x0600, /*indices*/ 0xDEAD0000BEEF0040ull, /*modifier*/ 0x40000000ull, 0, 0);
    evt(D, /*event_type*/ 0x42, /*address*/ 0x1400ABCD00ull, 0, 0, 0);
    set_indirect_base(D, /*graphics*/ 0, 0x123456789ABC0000ull, 0, 0, 0);
    stall_parser(D, 0, 0, 0, 0, 0);
    draw_indirect(D, /*byte_offset*/ 0x40, /*modifier*/ 0x80000000ull, 0, 0, 0);
    // #2929: the non-indexed sibling shares the graphics argument base selected above and
    // carries the same (offset, modifier) operands; only the sub-op and the four-dword argument
    // layout at the far end differ, so it must decode to its OWN kind.
    draw_nonindexed_indirect(D, /*byte_offset*/ 0x60, /*modifier*/ 0x90000000ull, 0, 0, 0);
    set_indirect_base(D, /*compute*/ 1, 0xFEDCBA9876540000ull, 0, 0, 0);
    dispatch_indirect(D, /*byte_offset*/ 0x80, /*modifier*/ 1, 0, 0, 0);
    pop(D, 0, 0, 0, 0, 0);

    size_t used_dw = (size_t)(dcb.cursor_up - dcb.bottom);
    printf("  built %zu dwords\n", used_dw);

    std::vector<Pm4Command> ops;
    size_t consumed = decode_pm4(buffer, 256, ops);   // pass full buffer; decoder stops at the zero tail
    CHECK(consumed == used_dw, "decoder consumed exactly the built dwords (stops at zero pad)");
    CHECK(ops.size() == 18, "decoded 18 packets");
    if (ops.size() != 18) { printf("== FAIL: got %zu packets ==\n", ops.size()); FAIL() << "legacy early exit"; }

    using K = Pm4Command::Kind;
    CHECK(ops[0].kind == K::PushMarker && ops[0].marker_label &&
          strcmp(ops[0].marker_label, marker) == 0, "op0 = PushMarker(label)");

    CHECK(ops[1].kind == K::DrawReset, "op1 = DrawReset");

    CHECK(ops[2].kind == K::SetIndexType && ops[2].index_size == 2, "op2 = SetIndexType(size=2)");

    CHECK(ops[3].kind == K::SetRegsIndirect, "op3 = SetRegsIndirect");
    CHECK(ops[3].reg_class == RegClass::Cx, "op3 reg_class = Cx");
    CHECK(ops[3].num_regs == 8, "op3 num_regs = 8 (3 + patched 5)");
    CHECK(ops[3].regs_vaddr == 0xCAFEF00DDEADBEEFull, "op3 regs_vaddr = patched address");

    CHECK(ops[4].kind == K::SetRegDirect && ops[4].reg_class == RegClass::Cx &&
          ops[4].reg_offset == 0x100 && ops[4].reg_count == 1 &&
          ops[4].reg_data && ops[4].reg_data[0] == 0x11111111,
          "op4 = SetCxRegisterDirect(offset/value)");
    CHECK(ops[5].kind == K::SetRegDirect && ops[5].reg_class == RegClass::Sh &&
          ops[5].reg_offset == 0x200 && ops[5].reg_count == 1 &&
          ops[5].reg_data && ops[5].reg_data[0] == 0x22222222,
          "op5 = SetShRegisterDirect(offset/value)");
    CHECK(ops[6].kind == K::SetRegDirect && ops[6].reg_class == RegClass::Uc &&
          ops[6].reg_offset == 0x300 && ops[6].reg_count == 1 &&
          ops[6].reg_data && ops[6].reg_data[0] == 0x33333333,
          "op6 = SetUcRegisterDirect(offset/value)");

    CHECK(ops[7].kind == K::SetNumInstances && ops[7].instance_count == 7,
          "op7 = SetNumInstances(7)");
    CHECK(ops[8].kind == K::DrawIndexAuto && ops[8].index_count == 0x1234, "op8 = DrawIndexAuto(0x1234)");
    CHECK(ops[8].di_modifier == 0x1122334455667788ull,
          "op8 DrawIndexAuto modifier round-trips");

    // DrawIndex round-trip (issue #63): the builder wrote [1]=count, [2..3]=index addr, [4..5]=modifier;
    // the decoder must hand every field back (indexed draws were previously dropped as unknown NOPs).
    CHECK(ops[9].kind == K::DrawIndex && ops[9].index_count == 0x0600, "op9 = DrawIndex(count=0x600)");
    CHECK(ops[9].di_index_addr == 0xDEAD0000BEEF0040ull, "op9 index-buffer address round-trips");
    CHECK(ops[9].di_modifier == 0x40000000ull && ops[9].di_valid, "op9 modifier round-trips (valid)");

    CHECK(ops[10].kind == K::EventWrite && ops[10].event_type == 0x42, "op10 = EventWrite(0x42)");
    // Address-carrying EVENT_WRITE (#132): the widened packet now round-trips its destination
    // address (was discarded, so an address-carrying event lost its write target).
    CHECK(ops[10].event_addr == 0x1400ABCD00ull, "op10 EventWrite address round-trips (was dropped)");
    CHECK(ops[11].kind == K::SetBaseIndirectArgs && ops[11].indirect_shader_type == 0 &&
          ops[11].indirect_base == 0x123456789ABC0000ull,
          "op11 = graphics indirect argument base");
    CHECK(ops[12].kind == K::StallCommandBufferParser, "op12 = parser visibility barrier");
    CHECK(ops[13].kind == K::DrawIndexIndirect && ops[13].indirect_offset == 0x40 &&
          ops[13].di_modifier == 0x80000000ull,
          "op13 = indexed indirect draw offset/modifier");
    // #2929: a distinct kind, not DrawIndexIndirect with a flag. Before the fix this packet did
    // not exist -- sceAgcDcbDrawIndirect had no handler, the builder call returned 0 having
    // written nothing, and the stream simply had one fewer draw in it.
    CHECK(ops[14].kind == K::DrawIndirect && ops[14].indirect_offset == 0x60 &&
          ops[14].di_modifier == 0x90000000ull,
          "op14 = non-indexed indirect draw offset/modifier");
    CHECK(ops[15].kind == K::SetBaseIndirectArgs && ops[15].indirect_shader_type == 1 &&
          ops[15].indirect_base == 0xFEDCBA9876540000ull,
          "op15 = compute indirect argument base");
    CHECK(ops[16].kind == K::DispatchIndirect && ops[16].indirect_offset == 0x80 &&
          ops[16].dispatch_modifier == 1,
          "op16 = indirect dispatch offset/modifier");
    CHECK(ops[17].kind == K::PopMarker, "op17 = PopMarker");

    // Every decoded packet's len must match its header, and the payload pointer must be in-buffer.
    bool spans_ok = true;
    for (auto& c : ops) {
        if (c.len < 1 || c.len > 16) spans_ok = false;
        if (c.len > 1 && (c.payload < buffer || c.payload + (c.len - 1) > buffer + 256)) spans_ok = false;
    }
    CHECK(spans_ok, "all packet lengths/payload spans are in-bounds");

    // #401: sceAgcCbNop(dcb, 1) must NOT underflow the type-3 length field (num-2 == 0x3fff) and
    // mis-frame the whole submit. A 1-dword pad is emitted as a PM4 TYPE-2 filler (0x80000000) that
    // the decoder skips, so a real packet placed AFTER a 1-dword NOP still decodes — before the fix
    // the malformed 0x4001-dword header swallowed or truncated everything after it.
    {
        auto nop = Hle::lookup("LtTouSCZjHM");   // sceAgcCbNop
        CHECK(nop, "sceAgcCbNop registered");
        uint32_t buf2[64]; memset(buf2, 0, sizeof buf2);
        Dcb d2{}; d2.bottom = buf2; d2.top = buf2 + 64; d2.cursor_up = buf2; d2.cursor_down = buf2 + 64;
        auto D2 = (uint64_t)(uintptr_t)&d2;
        uint64_t np = nop(D2, /*num_dwords*/ 1, 0, 0, 0, 0);
        CHECK(np && *(const uint32_t*)(uintptr_t)np == 0x80000000u,
              "cb_nop(1) emits a 1-dword PM4 type-2 filler (0x80000000), not an underflowed type-3 header");
        CHECK((size_t)(d2.cursor_up - d2.bottom) == 1, "cb_nop(1) advanced the cursor by exactly 1 dword");
        draw(D2, /*index_count*/ 0x0055, 0, 0, 0, 0);   // a real packet AFTER the 1-dword NOP
        std::vector<Pm4Command> ops2;
        decode_pm4(buf2, 64, ops2);
        CHECK(ops2.size() == 1 && ops2[0].kind == K::DrawIndexAuto && ops2[0].index_count == 0x0055,
              "the type-2 filler is skipped and the draw after a 1-dword NOP still decodes (fold intact)");
        // A normal multi-dword NOP still frames as one type-3 packet (regression guard).
        uint32_t buf3[64]; memset(buf3, 0, sizeof buf3);
        Dcb d3{}; d3.bottom = buf3; d3.top = buf3 + 64; d3.cursor_up = buf3; d3.cursor_down = buf3 + 64;
        nop((uint64_t)(uintptr_t)&d3, /*num_dwords*/ 4, 0, 0, 0, 0);
        CHECK((size_t)(d3.cursor_up - d3.bottom) == 4, "cb_nop(4) reserves 4 dwords (type-3 NOP unchanged)");

        // #450: begin_packet rejects an OVER-LENGTH packet (n > 0x4001) at the shared choke point rather
        // than wrapping the 14-bit length field into a header claiming a tiny packet. cb_nop(0x4002) must
        // emit nothing (return 0) and not advance the cursor — the overflow sibling of #401's underflow.
        uint32_t buf4[64]; memset(buf4, 0, sizeof buf4);
        Dcb d4{}; d4.bottom = buf4; d4.top = buf4 + 64; d4.cursor_up = buf4; d4.cursor_down = buf4 + 64;
        uint64_t big = nop((uint64_t)(uintptr_t)&d4, /*num_dwords*/ 0x4002, 0, 0, 0, 0);
        CHECK(big == 0 && d4.cursor_up == d4.bottom, "cb_nop(0x4002) rejected (packet > 0x4001, no cursor advance)");

        // #450: sceAgcDcbWriteData bounds num against the 5-dword packet overhead — num=0x3FFF (5+num =
        // 0x4004 > 0x4001) must be REJECTED, not emit a wrapped header that truncates the submit.
        auto wd = Hle::lookup("i1jyy49AjXU");   // sceAgcDcbWriteData
        CHECK(wd, "sceAgcDcbWriteData registered");
        uint32_t buf5[64]; memset(buf5, 0, sizeof buf5);
        Dcb d5{}; d5.bottom = buf5; d5.top = buf5 + 64; d5.cursor_up = buf5; d5.cursor_down = buf5 + 64;
        uint64_t wr = wd((uint64_t)(uintptr_t)&d5, /*dst*/0, /*policy*/0, /*addr*/0, /*data*/0, /*num*/0x3FFF);
        CHECK(wr == 0 && d5.cursor_up == d5.bottom, "WriteData(num=0x3FFF) rejected (5+num overflows the length field)");
    }

    // Retain both WRITE_DATA's declared count and the bounded number of inline words present in
    // the packet. A short packet may expose one safe-to-read word, but it is not a complete
    // one-word write: downstream ordering must be able to keep the declared two-word effect
    // fail-closed without ever indexing beyond this six-dword buffer.
    {
        const uint64_t address = 0x123456780ull;
        uint32_t short_write[6] = {};
        short_write[0] = PM4(6, IT_NOP, R_WRITE_DATA);
        short_write[1] = 0;
        short_write[2] = (uint32_t)address;
        short_write[3] = (uint32_t)(address >> 32);
        short_write[4] = 2;
        short_write[5] = 0x11223344u;
        std::vector<Pm4Command> short_ops;
        const size_t short_consumed = decode_pm4(short_write, 6, short_ops);
        CHECK(short_consumed == 6 && short_ops.size() == 1 &&
                  short_ops[0].kind == K::WriteData &&
                  short_ops[0].wd_addr == address &&
                  short_ops[0].wd_declared_num == 2 && short_ops[0].wd_num == 1 &&
                  short_ops[0].wd_data && short_ops[0].wd_data[0] == 0x11223344u &&
                  !short_ops[0].wd_valid,
              "WRITE_DATA decoder preserves a truncated declared-two/available-one payload");

        uint32_t complete_write[7] = {};
        memcpy(complete_write, short_write, sizeof short_write);
        complete_write[0] = PM4(7, IT_NOP, R_WRITE_DATA);
        complete_write[6] = 0x55667788u;
        std::vector<Pm4Command> complete_ops;
        const size_t complete_consumed = decode_pm4(complete_write, 7, complete_ops);
        CHECK(complete_consumed == 7 && complete_ops.size() == 1 &&
                  complete_ops[0].kind == K::WriteData &&
                  complete_ops[0].wd_declared_num == 2 && complete_ops[0].wd_num == 2 &&
                  complete_ops[0].wd_data && complete_ops[0].wd_data[1] == 0x55667788u &&
                  complete_ops[0].wd_valid,
              "WRITE_DATA decoder marks a complete declared-two payload valid");
    }

    // Hardware IT_LOAD_* packets (Task H01 Black Flag console capture submit0.bin):
    {
        uint32_t load_stream[] = {
            // IT_LOAD_SH_REG: 5 dwords, op 0x63, addr 0x4074ec2fbc, flags 0x80000000, num_regs 10
            PM4(5, IT_LOAD_SH_REG, 0),
            0x74ec2fbcu, 0x00000040u, 0x80000000u, 10u,
            // IT_LOAD_CONTEXT_REG: 5 dwords, op 0x64, addr 0x406616e000, flags 0x80000000, num_regs 6
            PM4(5, IT_LOAD_CONTEXT_REG, 0),
            0x6616e000u, 0x00000040u, 0x80000000u, 6u,
            // IT_LOAD_CONTEXT_REG_INDEX: 5 dwords, op 0x9F, addr 0x4075172000, flags 0x80000000, num_regs 143
            PM4(5, IT_LOAD_CONTEXT_REG_INDEX, 0),
            0x75172000u, 0x00000040u, 0x80000000u, 143u,
        };
        std::vector<Pm4Command> load_ops;
        const size_t c = decode_pm4(load_stream, std::size(load_stream), load_ops);
        CHECK(c == 15, "consumed 15 dwords of IT_LOAD_* packets");
        CHECK(load_ops.size() == 3, "decoded 3 IT_LOAD_* packets");

        CHECK(load_ops[0].kind == K::SetRegsIndirect, "op0 is SetRegsIndirect");
        CHECK(load_ops[0].reg_class == RegClass::Sh, "op0 class is Sh");
        CHECK(load_ops[0].regs_vaddr == 0x4074ec2fbcu, "op0 vaddr matches");
        CHECK(load_ops[0].num_regs == 10, "op0 num_regs = 10");

        CHECK(load_ops[1].kind == K::SetRegsIndirect, "op1 is SetRegsIndirect");
        CHECK(load_ops[1].reg_class == RegClass::Cx, "op1 class is Cx");
        CHECK(load_ops[1].regs_vaddr == 0x406616e000u, "op1 vaddr matches");
        CHECK(load_ops[1].num_regs == 6, "op1 num_regs = 6");

        CHECK(load_ops[2].kind == K::SetRegsIndirect, "op2 is SetRegsIndirect");
        CHECK(load_ops[2].reg_class == RegClass::Cx, "op2 class is Cx");
        CHECK(load_ops[2].regs_vaddr == 0x4075172000u, "op2 vaddr matches");
        CHECK(load_ops[2].num_regs == 143, "op2 num_regs = 143");
    }

    // Hardware IT_ACQUIRE_MEM (0x58) and IT_RELEASE_MEM (0x49) (Black Flag submit0.bin):
    {
        uint32_t sync_stream[] = {
            // IT_ACQUIRE_MEM: 8 dwords, op 0x58
            PM4(8, IT_ACQUIRE_MEM, 0),
            0x80000000u, 0x00000001u, 0x00000000u, 0x40056300u, 0x00000000u, 0x00000019u, 0x00009000u,
            // IT_RELEASE_MEM: 8 dwords, op 0x49, data_sel=1 (32-bit write), addr=0x40749a2f00, value=0x408c
            PM4(8, IT_RELEASE_MEM, 0),
            0x06000528u, (1u << 29) | 0x00010000u, 0x749a2f00u, 0x00000040u, 0x0000408cu, 0x00000000u, 0x00000000u,
        };
        std::vector<Pm4Command> sync_ops;
        const size_t c = decode_pm4(sync_stream, std::size(sync_stream), sync_ops);
        CHECK(c == 16, "consumed 16 dwords of hardware sync packets");
        CHECK(sync_ops.size() == 2, "decoded 2 sync packets");

        CHECK(sync_ops[0].kind == K::AcquireMem, "op0 is AcquireMem");

        CHECK(sync_ops[1].kind == K::ReleaseMem, "op1 is ReleaseMem");
        CHECK(sync_ops[1].rel_data_sel == 1, "op1 data_sel == 1");
        CHECK(sync_ops[1].rel_addr == 0x40749a2f00ull, "op1 rel_addr == 0x40749a2f00");
        CHECK(sync_ops[1].rel_value_valid && sync_ops[1].rel_value == 0x408cu, "op1 rel_value == 0x408c");
    }

    // Hardware IT_DISPATCH_DIRECT (0x15) and IT_DISPATCH_INDIRECT (0x16) (Black Flag submit0.bin):
    {
        uint32_t dispatch_stream[] = {
            // IT_DISPATCH_DIRECT: 5 dwords, op 0x15, dims=(0x168, 0xcb, 1), modifier=0x41
            PM4(5, IT_DISPATCH_DIRECT, 0),
            0x168u, 0xcbu, 1u, 0x41u,
            // IT_DISPATCH_INDIRECT: 3 dwords, op 0x16, offset=0x2000, modifier=0x41
            PM4(3, IT_DISPATCH_INDIRECT, 0),
            0x2000u, 0x41u,
        };
        std::vector<Pm4Command> disp_ops;
        const size_t c = decode_pm4(dispatch_stream, std::size(dispatch_stream), disp_ops);
        CHECK(c == 8, "consumed 8 dwords of hardware dispatch packets");
        CHECK(disp_ops.size() == 2, "decoded 2 dispatch packets");

        CHECK(disp_ops[0].kind == K::DispatchDirect, "op0 is DispatchDirect");
        CHECK(disp_ops[0].threads_x == 0x168u, "op0 threads_x == 0x168");
        CHECK(disp_ops[0].threads_y == 0xcbu, "op0 threads_y == 0xcb");
        CHECK(disp_ops[0].threads_z == 1u, "op0 threads_z == 1");
        CHECK(disp_ops[0].dispatch_modifier == 0x41u, "op0 modifier == 0x41");

        CHECK(disp_ops[1].kind == K::DispatchIndirect, "op1 is DispatchIndirect");
        CHECK(disp_ops[1].indirect_offset == 0x2000u, "op1 offset == 0x2000");
        CHECK(disp_ops[1].dispatch_modifier == 0x41u, "op1 modifier == 0x41");
    }

    // Hardware draw and write packets (Black Flag submit0.bin):
    {
        uint32_t work_stream[] = {
            // IT_SET_BASE: 4 dwords, op 0x11, type=1 (compute), addr=0x40756412e0
            PM4(4, IT_SET_BASE, 0),
            1u, 0x756412e0u, 0x00000040u,
            // IT_DRAW_INDEX_AUTO: 3 dwords, op 0x2D, count=0x300, modifier=2
            PM4(3, IT_DRAW_INDEX_AUTO, 0),
            0x300u, 2u,
            // IT_DRAW_INDIRECT: 3 dwords, op 0x24, offset=0x40, modifier=2
            PM4(3, IT_DRAW_INDIRECT, 0),
            0x40u, 2u,
            // IT_DRAW_INDEX_INDIRECT: 3 dwords, op 0x25, offset=0x60, modifier=2
            PM4(3, IT_DRAW_INDEX_INDIRECT, 0),
            0x60u, 2u,
            // IT_INDEX_BASE: 3 dwords, op 0x26, addr=0x403fde86dc
            PM4(3, IT_INDEX_BASE, 0),
            0x3fde86dcu, 0x00000040u,
            // IT_WRITE_DATA: 5 dwords, op 0x37, dst=0x100200, addr=0x406616c058, data=0
            PM4(5, IT_WRITE_DATA, 0),
            0x100200u, 0x6616c058u, 0x00000040u, 0u,
        };
        std::vector<Pm4Command> work_ops;
        const size_t c = decode_pm4(work_stream, std::size(work_stream), work_ops);
        CHECK(c == 21, "consumed 21 dwords of hardware work packets");
        CHECK(work_ops.size() == 6, "decoded 6 hardware work packets");

        CHECK(work_ops[0].kind == K::SetBaseIndirectArgs, "op0 is SetBaseIndirectArgs");
        CHECK(work_ops[0].indirect_shader_type == 1u, "op0 type == compute");
        CHECK(work_ops[0].indirect_base == 0x40756412e0ull, "op0 base == 0x40756412e0");

        CHECK(work_ops[1].kind == K::DrawIndexAuto, "op1 is DrawIndexAuto");
        CHECK(work_ops[1].index_count == 0x300u, "op1 count == 0x300");
        CHECK(work_ops[1].di_modifier == 2u, "op1 modifier == 2");

        CHECK(work_ops[2].kind == K::DrawIndirect, "op2 is DrawIndirect");
        CHECK(work_ops[2].indirect_offset == 0x40u, "op2 offset == 0x40");

        CHECK(work_ops[3].kind == K::DrawIndexIndirect, "op3 is DrawIndexIndirect");
        CHECK(work_ops[3].indirect_offset == 0x60u, "op3 offset == 0x60");

        CHECK(work_ops[4].kind == K::SetIndexBase, "op4 is SetIndexBase");
        CHECK(work_ops[4].ib_addr == 0x403fde86dcull, "op4 ib_addr == 0x403fde86dc");

        CHECK(work_ops[5].kind == K::WriteData, "op5 is WriteData");
        CHECK(work_ops[5].wd_addr == 0x406616c058ull, "op5 wd_addr == 0x406616c058");
        CHECK(work_ops[5].wd_num == 1u && work_ops[5].wd_valid, "op5 wd_num == 1");
    }

    // Hardware IT_WAIT_REG_MEM (0x3C) (Black Flag submit0.bin):
    {
        uint32_t wait_stream[] = {
            // IT_WAIT_REG_MEM: 7 dwords, op 0x3C, func=3, addr=0x406616c058, ref=1, mask=0xffffffff
            PM4(7, IT_WAIT_REG_MEM, 0),
            0x13u, 0x6616c058u, 0x00000040u, 1u, 0xffffffffu, 0x19u,
        };
        std::vector<Pm4Command> wait_ops;
        const size_t c = decode_pm4(wait_stream, std::size(wait_stream), wait_ops);
        CHECK(c == 7, "consumed 7 dwords of hardware wait packet");
        CHECK(wait_ops.size() == 1, "decoded 1 wait packet");

        CHECK(wait_ops[0].kind == K::WaitRegMem, "op0 is WaitRegMem");
        CHECK(wait_ops[0].wm_func == 3u, "op0 func == 3");
        CHECK(wait_ops[0].wm_addr == 0x406616c058ull, "op0 addr == 0x406616c058");
        CHECK(wait_ops[0].wm_ref == 1u, "op0 ref == 1");
        CHECK(wait_ops[0].wm_mask == 0xffffffffu && wait_ops[0].wm_valid, "op0 mask == 0xffffffff and valid");
    }

    // Hardware control and barrier packets (Black Flag submit0.bin):
    {
        uint32_t ctl_stream[] = {
            // IT_CLEAR_STATE: 2 dwords, op 0x12
            PM4(2, IT_CLEAR_STATE, 0), 0u,
            // IT_INDEX_BUFFER_SIZE: 2 dwords, op 0x13, count=0x38a
            PM4(2, IT_INDEX_BUFFER_SIZE, 0), 0x38au,
            // IT_ATOMIC_MEM: 2 dwords, op 0x1E
            PM4(2, IT_ATOMIC_MEM, 0), 0u,
            // IT_CONTEXT_CONTROL: 2 dwords, op 0x28
            PM4(2, IT_CONTEXT_CONTROL, 0), 0u,
            // IT_PFP_SYNC_ME: 2 dwords, op 0x42
            PM4(2, IT_PFP_SYNC_ME, 0), 0u,
        };
        std::vector<Pm4Command> ctl_ops;
        const size_t c = decode_pm4(ctl_stream, std::size(ctl_stream), ctl_ops);
        CHECK(c == 10, "consumed 10 dwords of control/barrier packets");
        CHECK(ctl_ops.size() == 5, "decoded 5 control/barrier packets");

        CHECK(ctl_ops[0].kind == K::DrawReset, "op0 is DrawReset");
        CHECK(ctl_ops[1].kind == K::SetIndexCount && ctl_ops[1].index_count == 0x38au, "op1 is SetIndexCount");
        CHECK(ctl_ops[2].kind == K::AcquireMem, "op2 is AcquireMem");
        CHECK(ctl_ops[3].kind == K::DrawReset, "op3 is DrawReset");
        CHECK(ctl_ops[4].kind == K::StallCommandBufferParser, "op4 is StallCommandBufferParser");
    }

    // Hardware conditional execution, multi-draw indirect, and 64-bit wait packets (Black Flag submit0.bin):
    {
        uint32_t cond_stream[] = {
            // IT_COND_EXEC: 5 dwords, op 0x22: [0..1]=addr, [2]=control, [3]=count
            PM4(5, IT_COND_EXEC, 0), 0xe0040060u, 0x0000000fu, 0x0u, 0x8u,
            // IT_DRAW_INDEX_INDIRECT_MULTI: 10 dwords, op 0x38: [0]=offset 0x19000, [1]=count 0x98...
            PM4(10, IT_DRAW_INDEX_INDIRECT_MULTI, 0), 0x19000u, 0x98u, 0x99u, 0x280u, 0x13u, 0x0u, 0x0u, 0x14u, 0x0u,
            // IT_WAIT_REG_MEM64: 9 dwords, op 0x93: [0]=func/flags, [1..2]=addr, [3]=ref, [4]=mask...
            PM4(9, IT_WAIT_REG_MEM64, 0), 0x6000113u, 0x800040a8u, 0xcu, 0x0u, 0x0u, 0xffffffffu, 0xffffffffu, 0x40u,
        };
        std::vector<Pm4Command> cond_ops;
        const size_t c = decode_pm4(cond_stream, std::size(cond_stream), cond_ops);
        CHECK(c == 24, "consumed 24 dwords of cond/multi-draw/wait64 packets");
        CHECK(cond_ops.size() == 3, "decoded 3 cond/multi-draw/wait64 packets");

        CHECK(cond_ops[0].kind == K::SetPredication, "op0 is SetPredication");
        CHECK(cond_ops[0].pred_addr == 0x0000000fe0040060ull, "op0 pred_addr matches");
        CHECK(cond_ops[0].pred_valid, "op0 pred_valid");

        CHECK(cond_ops[1].kind == K::DrawIndexIndirect, "op1 is DrawIndexIndirect");
        CHECK(cond_ops[1].indirect_offset == 0x19000u, "op1 indirect_offset == 0x19000");
        CHECK(cond_ops[1].di_modifier == 0x98u, "op1 di_modifier matches count word");

        CHECK(cond_ops[2].kind == K::WaitRegMem, "op2 is WaitRegMem");
        CHECK(cond_ops[2].wm_func == (0x6000113u & 7u), "op2 wm_func matches");
        CHECK(cond_ops[2].wm_addr == 0x0000000c800040a8ull, "op2 wm_addr matches");
        CHECK(cond_ops[2].wm_ref == 0u, "op2 wm_ref == 0");
    }
}

