// test_agc_dcb — guards the ported AGC "Gen5" Draw Command Buffer HLE (hle_agc.cpp). Sets up a
// synthetic Dcb over a dword buffer, drives the Dcb functions through the NID registry, and asserts
// they build the correct PM4 packets, advance the write cursor, return the packet pointer, and that
// the indirect-register patch helpers modify a previously-returned packet. This validates the port
// independently of the (locale-blocked) boot.
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/pm4_decode.hpp"
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
constexpr uint32_t IT_NOP = 0x10, IT_NUM_INSTANCES = 0x2F, IT_SET_CONTEXT_REG = 0x69,
                   IT_SET_SH_REG = 0x76, IT_SET_UCONFIG_REG = 0x79,
                   R_DRAW_RESET = 0x05, R_PUSH_MARKER = 0x0b, R_POP_MARKER = 0x0c,
                   R_SH_REGS_INDIRECT = 0x11, R_CX_REGS_INDIRECT = 0x12,
                   R_UC_REGS_INDIRECT = 0x13, R_DMA_DATA = 0x19;

struct ShaderRegister { uint32_t offset, value; };

struct RegisterDefaults {
    ShaderRegister** tbl0;
    ShaderRegister** tbl1;
    ShaderRegister** tbl2;
    ShaderRegister** tbl3;
    uint64_t unknown[2];
    uint32_t* types;
    uint32_t count;
};
static_assert(offsetof(RegisterDefaults, count) == 0x38);
extern "C" void* prosper_agc_reg_defaults(unsigned int version);

int main() {
    printf("== test_agc_dcb ==\n");
    register_builtin_hle();

    auto reset  = Hle::lookup("TRO721eVt4g");   // GraphicsDcbResetQueue
    auto setcx  = Hle::lookup("ZvwO9euwYzc");   // GraphicsDcbSetCxRegistersIndirect
    auto p_add  = Hle::lookup("d-6uF9sZDIU");   // SetCxRegIndirectPatchAddRegisters
    auto p_addr = Hle::lookup("vcmNN+AAXnY");   // SetCxRegIndirectPatchSetAddress
    auto p_dma_src = Hle::lookup("cdDRpqcFGbU"); // DmaDataPatchSetSrcAddressOrOffsetOrImmediate
    auto setcx_direct = Hle::lookup("LHFXRrlTPD8"); // sceAgcDcbSetCxRegisterDirect
    auto setsh_direct = Hle::lookup("pFLArOT53+w"); // sceAgcDcbSetShRegisterDirect
    auto setuc_direct = Hle::lookup("w4-d0n60hdo"); // sceAgcDcbSetUcRegisterDirect
    auto type2 = Hle::lookup("qj7QZpgr9Uw"); // one-dword TYPE-2 filler
    auto interpolants = Hle::lookup("dbOlWdppb4o"); // Gen5 interpolant mapping
    CHECK(reset && setcx && p_add && p_addr && p_dma_src &&
          setcx_direct && setsh_direct && setuc_direct && type2 && interpolants,
          "AGC Dcb functions registered (override the glog stubs)");
    if (!(reset && setcx && p_add && p_addr && p_dma_src &&
          setcx_direct && setsh_direct && setuc_direct && type2 && interpolants)) {
        printf("== FAIL ==\n"); return 1;
    }

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

    // This NID is the one legal one-dword PM4 builder: a TYPE-2 filler cannot use a TYPE-3 header.
    uint32_t* type2_before = dcb.cursor_up;
    uint64_t type2_result = type2(D, 0, 0, 0, 0, 0);
    CHECK(type2_result == (uint64_t)(uintptr_t)type2_before && *type2_before == 0x80000000u,
          "TYPE-2 filler writes its one-dword native PM4 packet");
    CHECK(dcb.cursor_up == type2_before + 1, "TYPE-2 filler advances the cursor by one dword");

    // SetCxRegistersIndirect(dcb, regs=0x1122334455667788, num_regs=3) -> 4-dw packet.
    uint64_t regs = 0x1122334455667788ull;
    uint64_t rc = setcx(D, regs, 3, 0, 0, 0);
    auto* cmd = (uint32_t*)(uintptr_t)rc;
    CHECK(cmd == buffer + 3, "SetCx returns the next packet ptr");
    CHECK(cmd[0] == PM4(4, IT_NOP, R_CX_REGS_INDIRECT), "SetCx wrote R_CX_REGS_INDIRECT PM4 header");
    CHECK(cmd[1] == 3, "SetCx wrote num_regs=3");
    CHECK(cmd[2] == 0x55667788u && cmd[3] == 0x11223344u, "SetCx wrote regs vaddr lo/hi");
    CHECK(dcb.cursor_up == buffer + 7, "SetCx advanced the cursor by 4 dwords");

    // Patch helpers modify the returned packet (the old stub returned 0 -> these wrote through null).
    p_add(rc, 5, 0, 0, 0, 0);
    CHECK(cmd[1] == 8, "PatchAddRegisters did cmd[1] += 5 (3 -> 8)");
    p_addr(rc, 0xAABBCCDD00112233ull, 0, 0, 0, 0);
    CHECK(cmd[2] == 0x00112233u && cmd[3] == 0xAABBCCDDu, "PatchSetAddress rewrote cmd[2]/cmd[3]");

    // PatchSetNumRegisters SETS the count outright — the only member of this family that can
    // LOWER one. The AGC record-then-patch idiom reserves the packet before the register array is
    // known, so an ignored patch leaves the RECORD-TIME placeholder in cmd[1]: a packet reserved
    // with 0 applies no registers at all (GpuState::apply returns early on num_regs == 0) and the
    // whole bind disappears from the decoded stream. All three class variants must be registered,
    // must write the count slot, and must refuse a packet of the wrong class.
    auto p_num_cx = Hle::lookup("whb1RL7K4Ss");   // sceAgcSetCxRegIndirectPatchSetNumRegisters
    auto p_num_sh = Hle::lookup("nCUgItdN2ms");   // sceAgcSetShRegIndirectPatchSetNumRegisters
    auto p_num_uc = Hle::lookup("fRG-JOH5+sI");   // sceAgcSetUcRegIndirectPatchSetNumRegisters
    auto setsh_ind = Hle::lookup("-HOOCn0JY48");  // sceAgcDcbSetShRegistersIndirect
    auto setuc_ind = Hle::lookup("hvUfkUIQcOE");  // sceAgcDcbSetUcRegistersIndirect
    CHECK(p_num_cx && p_num_sh && p_num_uc && setsh_ind && setuc_ind,
          "all three PatchSetNumRegisters NIDs and the Sh/Uc indirect builders are registered");
    if (p_num_cx && p_num_sh && p_num_uc && setsh_ind && setuc_ind) {
        p_num_cx(rc, 2, 0, 0, 0, 0);
        CHECK(cmd[1] == 2, "Cx PatchSetNumRegisters SETS the count (8 -> 2), it does not accumulate");
        CHECK(cmd[2] == 0x00112233u && cmd[3] == 0xAABBCCDDu,
              "Cx PatchSetNumRegisters leaves the array address untouched");

        // The record-with-a-placeholder-count sequence the defect turned into a dropped bind.
        uint64_t sh_rc = setsh_ind(D, 0, 0, 0, 0, 0);
        auto* sh_cmd = (uint32_t*)(uintptr_t)sh_rc;
        CHECK(sh_cmd && sh_cmd[0] == PM4(4, IT_NOP, R_SH_REGS_INDIRECT) && sh_cmd[1] == 0,
              "SetShRegistersIndirect reserves a packet with the placeholder count 0");
        p_addr(sh_rc, 0x0000000340000000ull, 0, 0, 0, 0);
        p_num_sh(sh_rc, 12, 0, 0, 0, 0);
        CHECK(sh_cmd[1] == 12, "Sh PatchSetNumRegisters turns the reserved 0-count packet into 12");
        CHECK(sh_cmd[2] == 0x40000000u && sh_cmd[3] == 0x00000003u,
              "Sh reserved packet keeps its patched array address");

        uint64_t uc_rc = setuc_ind(D, 0, 7, 0, 0, 0);
        auto* uc_cmd = (uint32_t*)(uintptr_t)uc_rc;
        p_num_uc(uc_rc, 1, 0, 0, 0, 0);
        CHECK(uc_cmd && uc_cmd[1] == 1, "Uc PatchSetNumRegisters lowers a recorded count (7 -> 1)");

        // Class check: the Sh patcher must refuse the Cx packet rather than corrupt its count.
        const uint32_t cx_num_before = cmd[1];
        p_num_sh(rc, 999, 0, 0, 0, 0);
        CHECK(cmd[1] == cx_num_before,
              "PatchSetNumRegisters refuses a packet of a different register class");
    }

    // #395 F5: all three single-register direct NIDs append the native packet opcode and preserve
    // the by-value ShaderRegister's offset/value halves. Previously SH was dead code and Cx/Uc fell
    // through to unimplemented-success without advancing the DCB at all.
    uint32_t* direct_before = dcb.cursor_up;
    auto pack_reg = [](uint32_t offset, uint32_t value) {
        return (uint64_t)offset | ((uint64_t)value << 32u);
    };
    auto* cx_direct = (uint32_t*)(uintptr_t)setcx_direct(D, pack_reg(0x123, 0x11111111), 0, 0, 0, 0);
    auto* sh_direct = (uint32_t*)(uintptr_t)setsh_direct(D, pack_reg(0x234, 0x22222222), 0, 0, 0, 0);
    auto* uc_direct = (uint32_t*)(uintptr_t)setuc_direct(D, pack_reg(0x345, 0x33333333), 0, 0, 0, 0);
    CHECK(cx_direct == direct_before &&
          cx_direct[0] == PM4(3, IT_SET_CONTEXT_REG, 0) &&
          cx_direct[1] == 0x123 && cx_direct[2] == 0x11111111,
          "SetCxRegisterDirect appends one context-register packet");
    CHECK(sh_direct == direct_before + 3 &&
          sh_direct[0] == PM4(3, IT_SET_SH_REG, 0) &&
          sh_direct[1] == 0x234 && sh_direct[2] == 0x22222222,
          "SetShRegisterDirect appends one shader-register packet");
    CHECK(uc_direct == direct_before + 6 &&
          uc_direct[0] == PM4(3, IT_SET_UCONFIG_REG, 0) &&
          uc_direct[1] == 0x345 && uc_direct[2] == 0x33333333,
          "SetUcRegisterDirect appends one user-config-register packet");
    CHECK(dcb.cursor_up == direct_before + 9,
          "single-register direct writers advance the DCB by all three packets");

    // sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate (#189): patch only the source qword of
    // a verified DMA_DATA packet. A wrong packet kind must remain untouched.
    // BOTH packet shapes: the live 7-dword one (#1756, the hardware DMA_DATA size, what the builder
    // emits today) and the historical 9-dword one still present in pre-#1756 captures. The patcher
    // must be shape-independent — it writes payload [2..3] either way — and testing only the shape
    // prosper no longer emits would leave the live path unasserted.
    uint32_t dma7[7] = {};
    dma7[0] = PM4(7, IT_NOP, R_DMA_DATA);
    dma7[1] = 0x11111111u; dma7[2] = 0x22222222u;  // destination must survive the source patch
    p_dma_src((uint64_t)(uintptr_t)dma7, 0xAABBCCDD00112233ull, 0, 0, 0, 0);
    CHECK(dma7[1] == 0x11111111u && dma7[2] == 0x22222222u,
          "DmaData source patch leaves the destination untouched (live 7-dword packet)");
    CHECK(dma7[3] == 0x00112233u && dma7[4] == 0xAABBCCDDu,
          "DmaData source patch writes the full 64-bit source (live 7-dword packet)");

    uint32_t dma[9] = {};
    dma[0] = PM4(9, IT_NOP, R_DMA_DATA);
    dma[1] = 0x11111111u; dma[2] = 0x22222222u;
    p_dma_src((uint64_t)(uintptr_t)dma, 0xAABBCCDD00112233ull, 0, 0, 0, 0);
    CHECK(dma[1] == 0x11111111u && dma[2] == 0x22222222u,
          "DmaData source patch leaves the destination untouched (historical 9-dword packet)");
    CHECK(dma[3] == 0x00112233u && dma[4] == 0xAABBCCDDu,
          "DmaData source patch writes the full 64-bit source (historical 9-dword packet)");

    // Tactics Ogre's movie upload reaches the same builder through its generic copyData helper.
    // That call shape keeps a1 at zero and carries sourceKind=2 plus the real source pointer in a7.
    // Encoding a1 makes a valid address copy become an immediate-zero fill, which leaves both movie
    // planes byte-for-byte zero even though AvPlayer returned decoded pixels. Exercise the exact
    // nine-argument ABI and require the packet—not merely a helper—to retain the source address.
    {
        auto dma_build = Hle::lookup("WmAc2MEj6Io");
        CHECK(dma_build != nullptr, "sceAgcDcbDmaData registered for address-source ABI coverage");
        if (dma_build) {
            alignas(16) uint8_t source[64] = {};
            uint32_t buf[16] = {};
            Dcb d{}; d.bottom = buf; d.top = buf + 16; d.cursor_up = buf; d.cursor_down = buf + 16;
            constexpr uint64_t dst = 0x2012340000ull;
            const uint64_t src = (uint64_t)(uintptr_t)source;
            auto* packet = (uint32_t*)(uintptr_t)
                ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
                              uint64_t,uint64_t,uint64_t))dma_build)(
                    (uint64_t)(uintptr_t)&d,
                    /*srcImmediateOrOffset*/0, /*dstSel*/3, /*srcSel*/3, dst,
                    /*policy*/3, /*sourceKind*/2, src, sizeof source);
            CHECK(packet == buf && packet[0] == PM4(7, IT_NOP, R_DMA_DATA),
                  "address-source DmaData call emits the ordinary seven-dword packet");
            CHECK(packet[1] == (uint32_t)dst && packet[2] == (uint32_t)(dst >> 32u),
                  "address-source DmaData preserves the destination address");
            CHECK(packet[3] == (uint32_t)src && packet[4] == (uint32_t)(src >> 32u) &&
                      packet[5] == sizeof source,
                  "sourceKind=2 selects the stack source instead of immediate zero");
            CHECK((packet[6] & gpu::kDmaDataAddressSource) != 0,
                  "address-source DmaData preserves the asserted source form in packet metadata");

            // The other ABI arm must remain independent of a7. Existing immediate/offset calls use
            // sourceKind=0; give this one a deliberately tempting address in a7 and prove that the
            // packet still contains a1. This keeps the historical fill path intact.
            constexpr uint64_t immediate = 0x12345678ull;
            d.cursor_up = buf;
            packet = (uint32_t*)(uintptr_t)
                ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
                              uint64_t,uint64_t,uint64_t))dma_build)(
                    (uint64_t)(uintptr_t)&d,
                    immediate, /*dstSel*/3, /*srcSel*/0, dst,
                    /*policy*/2, /*sourceKind*/0, src, sizeof source);
            CHECK(packet == buf && packet[0] == PM4(7, IT_NOP, R_DMA_DATA),
                  "immediate-source DmaData call emits the ordinary seven-dword packet");
            CHECK(packet[3] == (uint32_t)immediate && packet[4] == 0 &&
                      packet[5] == sizeof source,
                  "sourceKind=0 preserves a1 even when stack source looks addressable");
            CHECK((packet[6] & gpu::kDmaDataAddressSource) == 0,
                  "immediate-source DmaData does not acquire address-form metadata");

            // An asserted address form must not degrade into an immediate fill merely because the
            // address is malformed or currently unmapped. The executor owns validation and will
            // reject this address visibly; the builder's only job is to retain the selected form.
            // Deliberately keep this below UINT32_MAX. Numeric width used to be the executor's only
            // discriminator, so this exact asserted address degraded into a 0x4000 fill.
            constexpr uint64_t invalid_source = 0x4000ull;
            d.cursor_up = buf;
            packet = (uint32_t*)(uintptr_t)
                ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
                              uint64_t,uint64_t,uint64_t))dma_build)(
                    (uint64_t)(uintptr_t)&d,
                    immediate, /*dstSel*/3, /*srcSel*/3, dst,
                    /*policy*/3, /*sourceKind*/2, invalid_source, sizeof source);
            CHECK(packet[3] == (uint32_t)invalid_source &&
                      packet[4] == 0 && (packet[6] & gpu::kDmaDataAddressSource) != 0,
                  "sourceKind=2 retains an invalid address instead of silently filling from a1");
        }
    }

    // #1124's clobbered-header recovery, and the ONLY guard on it (#1756). Alex Kidd (PPSA02664,
    // rung 6) zeroes a DMA_DATA header before patching it; dma_patch_recover_header rebuilds the
    // header from scratch, which means it STAMPS A LENGTH. That length must equal what the builder
    // emits, and nothing enforced it: a full ctest is green with the recovery's literal desynced from
    // the builder, while a live boot shows 32 SHORT FOLDs from the first submit because the command
    // processor then walks two dwords into the following packet.
    //
    // Both sides are MEASURED here, so this needs no hardware number and cannot go stale: build a
    // real packet through the real builder to learn its size, clobber the header the way the guest
    // does, patch it, and require the restored header to declare exactly that size.
    {
        auto dma_build = Hle::lookup("WmAc2MEj6Io");   // sceAgcDcbDmaData
        CHECK(dma_build != nullptr, "sceAgcDcbDmaData registered");
        if (dma_build) {
            uint32_t buf[32] = {};
            Dcb d{}; d.bottom = buf; d.top = buf + 32; d.cursor_up = buf; d.cursor_down = buf + 32;
            uint64_t dst = 0x2010000040ull;   // plausible to the recovery's own body check
            ((uint64_t(*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t))
             dma_build)((uint64_t)(uintptr_t)&d, 0, 0, 0, dst, 0, 0, 0, 64);
            const uint32_t built_dw = (uint32_t)(d.cursor_up - buf);
            CHECK(built_dw >= 2, "DmaData builder emitted a packet");

            buf[0] = 0;                        // the guest's clobber: header zeroed, body intact
            p_dma_src((uint64_t)(uintptr_t)buf, 0xCAFEBABE0BADF00Dull, 0, 0, 0, 0);
            const uint32_t restored_dw = ((buf[0] >> 16) & 0x3fffu) + 2u;
            CHECK(buf[0] != 0, "recovery restored a clobbered DMA_DATA header");
            CHECK(restored_dw == built_dw,
                  "the recovered header declares exactly the length the builder emits "
                  "(desyncing them walks the command processor into the next packet)");
            CHECK(buf[3] == 0x0BADF00Du && buf[4] == 0xCAFEBABEu,
                  "recovery is followed by the source patch it was performed for");
        }
    }

    uint32_t wrong_packet[5] = {PM4(5, IT_NOP, R_CX_REGS_INDIRECT), 1, 2, 3, 4};
    p_dma_src((uint64_t)(uintptr_t)wrong_packet, 0xDEADBEEFCAFEBABEull, 0, 0, 0, 0);
    CHECK(wrong_packet[1] == 1 && wrong_packet[2] == 2 && wrong_packet[3] == 3 && wrong_packet[4] == 4,
          "DmaData source patch rejects a non-DMA packet without modifying it");

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

    // The authoritative PS5 symbol map identifies these NIDs as DCB marker/instance builders.
    // PushMarker used to clear live state through the obsolete g_agc_ctx_init workaround (#641).
    auto pushmarker = Hle::lookup("+kSrjIVxKFE");
    auto setmarker = Hle::lookup("QhCbS4X9Rl8");
    auto popmarker = Hle::lookup("H7uZqCoNuWk");
    auto setinstances = Hle::lookup("tSBxhAPyytQ");
    CHECK(pushmarker && setmarker && popmarker && setinstances,
          "marker and instance Dcb functions registered");
    if (pushmarker && setmarker && popmarker && setinstances) {
        uint32_t* before = dcb.cursor_up;
        const char marker[] = "Blasphemous 2 landing";
        uint32_t marker_payload_dw = (uint32_t)((sizeof marker + 3u) / 4u);
        uint64_t pushed = pushmarker(D, (uint64_t)(uintptr_t)marker, 0, 0, 0, 0);
        auto* push = (uint32_t*)(uintptr_t)pushed;
        CHECK(push == before, "PushMarker returns the allocated packet");
        CHECK(push[0] == PM4(1 + marker_payload_dw, IT_NOP, R_PUSH_MARKER),
              "PushMarker emits a correctly sized R_PUSH_MARKER packet");
        CHECK(strcmp((const char*)(push + 1), marker) == 0, "PushMarker copies the NUL-terminated label");

        const char set_label[] = "Sonic SetMarker";
        uint32_t set_payload_dw = (uint32_t)((sizeof set_label + 3u) / 4u);
        auto* set = (uint32_t*)(uintptr_t)setmarker(
            D, (uint64_t)(uintptr_t)set_label, 0xff00ff00u, 0, 0, 0);
        CHECK(set == push + 1 + marker_payload_dw &&
              set[0] == PM4(1 + set_payload_dw, IT_NOP, R_PUSH_MARKER) &&
              strcmp((const char*)(set + 1), set_label) == 0,
              "SetMarker appends the same native marker packet as PushMarker");

        auto* pop = (uint32_t*)(uintptr_t)popmarker(D, 0, 0, 0, 0, 0);
        CHECK(pop == set + 1 + set_payload_dw && pop[0] == PM4(2, IT_NOP, R_POP_MARKER),
              "PopMarker appends R_POP_MARKER after the label packet");

        auto* instances = (uint32_t*)(uintptr_t)setinstances(D, 3, 0, 0, 0, 0);
        CHECK(instances == pop + 2 && instances[0] == PM4(2, IT_NUM_INSTANCES, 0) && instances[1] == 3,
              "SetNumInstances appends the native IT_NUM_INSTANCES packet");
    }

    // Resource registration returns a 32-bit opaque ID through out*. Sonic places the owner and
    // resource outputs four bytes apart on its stack, so an accidental 64-bit store corrupts one.
    auto reg_owner = Hle::lookup("X-Nm5KLREeg");
    auto reg_resource = Hle::lookup("W5z4eZrjEas");
    CHECK(reg_owner && reg_resource, "AGC owner/resource registration handlers registered");
    if (reg_owner && reg_resource) {
        struct { uint32_t resource, owner, canary; } outputs{0, 0, 0xA5A55A5Au};
        const char owner_name[] = "Sonic Origins";
        const char resource_name[] = "Needle shader";
        CHECK(reg_owner((uint64_t)(uintptr_t)&outputs.owner,
                        (uint64_t)(uintptr_t)owner_name, 1, 0, 0, 0) == 0 && outputs.owner != 0,
              "RegisterOwner publishes a non-zero opaque owner handle");
        CHECK(reg_resource((uint64_t)(uintptr_t)&outputs.resource, 6, 0x411e70400ull, 0x190,
                           (uint64_t)(uintptr_t)resource_name, outputs.owner) == 0 &&
                  outputs.resource != 0,
              "RegisterResource publishes a non-zero opaque resource handle");
        uint32_t second = 0;
        reg_resource((uint64_t)(uintptr_t)&second, 6, 0x411e70600ull, 0xd0,
                     (uint64_t)(uintptr_t)resource_name, outputs.owner);
        CHECK(second != 0 && second != outputs.resource,
              "successive resource registrations receive distinct handles");
        CHECK(outputs.canary == 0xA5A55A5Au,
              "32-bit registration outputs preserve adjacent caller stack fields");
        CHECK(reg_owner(0, 0, 0, 0, 0, 0) != 0 && reg_resource(0, 0, 0, 0, 0, 0) != 0,
              "registration rejects null output pointers");
    }

    auto lookup_default = [](RegisterDefaults* d, uint32_t hash) -> ShaderRegister* {
        if (!d || !d->types) return nullptr;
        for (uint32_t i = 0; i < d->count; ++i) {
            const uint32_t* type = d->types + i * 3;
            if (type[0] != hash) continue;
            const uint32_t packed = type[1];
            ShaderRegister** banks[] = {d->tbl0, d->tbl1, d->tbl2, d->tbl3};
            const uint32_t bank = packed & 3u;
            const uint32_t id = (packed & 0x3fcu) / 4u;
            return banks[bank] ? banks[bank][id] : nullptr;
        }
        return nullptr;
    };

    // DQ requests version 13, searches the 12-byte records, and resolves the pointer-bank/id word.
    // Its render-target-zero key is absent from the older corpus and must produce the observed blend
    // default rather than the guest's all-ones not-found sentinel.
    auto* defaults = static_cast<RegisterDefaults*>(prosper_agc_reg_defaults(13));
    ShaderRegister* blend = lookup_default(defaults, 0xa6d12629u);
    CHECK(defaults && defaults->count == 128 && defaults->unknown[0] == 0 &&
          defaults->unknown[1] == 0 && blend && blend[0].offset == 0x1e0u &&
          blend[0].value == 0x20010001u,
          "SDK 13 resolves the render-target-zero blend default");

    // Guard a legacy multi-register run: the pointer must begin at the first register pair, not at
    // the preceding type hash. DQ copies these exact 16 bytes into its FOV defaults block.
    ShaderRegister* fov = lookup_default(defaults, 0x88f5e915u);
    CHECK(fov && fov[0].offset == 0xebu && fov[0].value == 0xff00ff00u &&
          fov[1].offset == 0xecu && fov[1].value == 0,
          "SDK 13 preserves legacy multi-register run pointers");

    auto* legacy_defaults = static_cast<RegisterDefaults*>(prosper_agc_reg_defaults(8));
    CHECK(legacy_defaults && legacy_defaults != defaults && legacy_defaults->count == 127 &&
          legacy_defaults->unknown[0] == 0 && legacy_defaults->unknown[1] == 0 &&
          !lookup_default(legacy_defaults, 0xa6d12629u),
          "SDK 8 callers retain the legacy table layout and defaults");

    uint64_t interpolant_regs[32];
    memset(interpolant_regs, 0, sizeof interpolant_regs);
    CHECK(interpolants((uint64_t)(uintptr_t)interpolant_regs, 0, 0, 0, 0, 0) == 0,
          "Gen5 interpolant helper accepts the identity/default mapping");
    bool identity_mapping = true;
    for (uint32_t i = 0; i < 32; ++i) {
        identity_mapping &= static_cast<uint32_t>(interpolant_regs[i]) == 0x10000000u + i;
        identity_mapping &= static_cast<uint32_t>(interpolant_regs[i] >> 32u) == i;
    }
    CHECK(identity_mapping,
          "Gen5 interpolant helper initializes all 32 virtual-offset/value pairs");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
