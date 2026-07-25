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
constexpr uint32_t IT_NOP = 0x10, IT_NUM_INSTANCES = 0x2F, IT_SET_CONTEXT_REG = 0x69,
                   IT_SET_SH_REG = 0x76, IT_SET_UCONFIG_REG = 0x79,
                   R_DRAW_RESET = 0x05, R_PUSH_MARKER = 0x0b, R_POP_MARKER = 0x0c,
                   R_CX_REGS_INDIRECT = 0x12, R_DMA_DATA = 0x19;

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
    uint32_t dma[9] = {};
    dma[0] = PM4(9, IT_NOP, R_DMA_DATA);
    dma[1] = 0x11111111u; dma[2] = 0x22222222u; // destination must survive the source patch
    p_dma_src((uint64_t)(uintptr_t)dma, 0xAABBCCDD00112233ull, 0, 0, 0, 0);
    CHECK(dma[1] == 0x11111111u && dma[2] == 0x22222222u,
          "DmaData source patch leaves the destination address untouched");
    CHECK(dma[3] == 0x00112233u && dma[4] == 0xAABBCCDDu,
          "DmaData source patch writes the complete 64-bit source address");
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
    auto popmarker = Hle::lookup("H7uZqCoNuWk");
    auto setinstances = Hle::lookup("tSBxhAPyytQ");
    CHECK(pushmarker && popmarker && setinstances, "marker and instance Dcb functions registered");
    if (pushmarker && popmarker && setinstances) {
        uint32_t* before = dcb.cursor_up;
        const char marker[] = "Blasphemous 2 landing";
        uint32_t marker_payload_dw = (uint32_t)((sizeof marker + 3u) / 4u);
        uint64_t pushed = pushmarker(D, (uint64_t)(uintptr_t)marker, 0, 0, 0, 0);
        auto* push = (uint32_t*)(uintptr_t)pushed;
        CHECK(push == before, "PushMarker returns the allocated packet");
        CHECK(push[0] == PM4(1 + marker_payload_dw, IT_NOP, R_PUSH_MARKER),
              "PushMarker emits a correctly sized R_PUSH_MARKER packet");
        CHECK(strcmp((const char*)(push + 1), marker) == 0, "PushMarker copies the NUL-terminated label");

        auto* pop = (uint32_t*)(uintptr_t)popmarker(D, 0, 0, 0, 0, 0);
        CHECK(pop == push + 1 + marker_payload_dw && pop[0] == PM4(2, IT_NOP, R_POP_MARKER),
              "PopMarker appends R_POP_MARKER after the label packet");

        auto* instances = (uint32_t*)(uintptr_t)setinstances(D, 3, 0, 0, 0, 0);
        CHECK(instances == pop + 2 && instances[0] == PM4(2, IT_NUM_INSTANCES, 0) && instances[1] == 3,
              "SetNumInstances appends the native IT_NUM_INSTANCES packet");
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
