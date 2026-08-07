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
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

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
// #1650: how many packet-patcher calls missed the DCB ring registry and paid for an OS write probe.
// The registry exists so the per-draw path never does; without a way to observe it, "no syscall on
// the hot path" would be an unfalsifiable claim in a comment.
extern "C" uint64_t prosper_agc_patch_probe_count();

// Capture everything written to stderr while `body` runs. A refused guest write that says nothing
// is the failure mode #1650 was filed for, so "was it reported" is part of the contract under test,
// not decoration. Same fd-swap shape as test_service_logging.cpp.
template <typename F>
static bool capture_stderr(F&& body, char* out, size_t out_size) {
    out[0] = '\0';
    FILE* capture = tmpfile();
    if (!capture) return false;
    if (fflush(stderr) != 0) { fclose(capture); return false; }
#ifdef _WIN32
    const int fd = _fileno(stderr), saved = _dup(fd);
    const bool redirected = saved >= 0 && _dup2(_fileno(capture), fd) == 0;
#else
    const int fd = fileno(stderr), saved = dup(fd);
    const bool redirected = saved >= 0 && dup2(fileno(capture), fd) >= 0;
#endif
    if (!redirected) {
#ifdef _WIN32
        if (saved >= 0) _close(saved);
#else
        if (saved >= 0) close(saved);
#endif
        fclose(capture);
        return false;
    }
    body();
    fflush(stderr);
#ifdef _WIN32
    const bool restored = _dup2(saved, fd) == 0; _close(saved);
#else
    const bool restored = dup2(saved, fd) >= 0; close(saved);
#endif
    rewind(capture);
    const size_t n = fread(out, 1, out_size - 1, capture);
    out[n] = '\0';
    fclose(capture);
    return restored;
}

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
    // The Sh members of the address/count family. Since #1650 each register class has its own
    // handler and verifies the packet's sub-op, so a test — like a guest — must patch an Sh packet
    // through the Sh NIDs. Reaching for the Cx one here used to "work" only because a single shared
    // handler served all three classes, which is the defect.
    auto p_addr_sh = Hle::lookup("Qrj4c+61z4A");  // sceAgcSetShRegIndirectPatchSetAddress
    auto p_add_sh  = Hle::lookup("z2duB-hHQSM");  // sceAgcSetShRegIndirectPatchAddRegisters
    CHECK(p_num_cx && p_num_sh && p_num_uc && setsh_ind && setuc_ind && p_addr_sh && p_add_sh,
          "all three PatchSetNumRegisters NIDs, the Sh/Uc indirect builders and the per-class Sh "
          "address/count patchers are registered");
    if (p_num_cx && p_num_sh && p_num_uc && setsh_ind && setuc_ind && p_addr_sh && p_add_sh) {
        p_num_cx(rc, 2, 0, 0, 0, 0);
        CHECK(cmd[1] == 2, "Cx PatchSetNumRegisters SETS the count (8 -> 2), it does not accumulate");
        CHECK(cmd[2] == 0x00112233u && cmd[3] == 0xAABBCCDDu,
              "Cx PatchSetNumRegisters leaves the array address untouched");

        // The record-with-a-placeholder-count sequence the defect turned into a dropped bind.
        uint64_t sh_rc = setsh_ind(D, 0, 0, 0, 0, 0);
        auto* sh_cmd = (uint32_t*)(uintptr_t)sh_rc;
        CHECK(sh_cmd && sh_cmd[0] == PM4(4, IT_NOP, R_SH_REGS_INDIRECT) && sh_cmd[1] == 0,
              "SetShRegistersIndirect reserves a packet with the placeholder count 0");
        p_addr_sh(sh_rc, 0x0000000340000000ull, 0, 0, 0, 0);
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

    // ---- #1650: PatchSetAddress / PatchAddRegisters must refuse a pointer that is not one of
    // their own packets, and must SAY SO.
    //
    // Both took a caller-supplied pointer behind nothing but a null check and stored through it.
    // A wrapped, aliased or stale builder pointer therefore got a 64-bit register-array address
    // written over two dwords of unrelated command-buffer or heap memory, with no log and no
    // reject — corruption at a distance, attributed to whatever broke next. These are not cold
    // paths: Sonic CrossWorlds logs 128 AddRegisters calls in a single 400-tick `hle_calls` window
    // (docs/SONIC_CROSSWORLDS_STATUS.md), and Blue Prince builds its register arrays this way per
    // draw (command_processor.hpp, #1264).
    //
    // Four arms, each naming the mutation it kills. Arm 1 is the positive control, because a fix
    // that refuses EVERYTHING would satisfy a refusal-only test while silently dropping every
    // register bind in those titles.
    if (p_addr_sh && p_add_sh) {
        // Arm 1 — POSITIVE CONTROL: a real packet, patched through its OWN class, still works.
        // Kills a "refuse everything" fix. It also pins the performance contract: an in-ring
        // pointer must be validated by range compare, never by an OS probe. guest_writable is
        // uncached and fopen()s /proc/self/maps on Linux on every call, so a probe here would be a
        // per-draw syscall on the hottest patch path — the reason this issue could not simply copy
        // the PatchSetNumRegisters guard (#1654).
        uint32_t ring[16];
        memset(ring, 0xEE, sizeof ring);
        Dcb rd{};
        rd.bottom = ring; rd.top = ring + 16; rd.cursor_up = ring; rd.cursor_down = ring + 16;
        const uint64_t RD = (uint64_t)(uintptr_t)&rd;
        const uint64_t pkt_addr = setcx(RD, 0, 4, 0, 0, 0);
        auto* pkt = (uint32_t*)(uintptr_t)pkt_addr;
        CHECK(pkt == ring && pkt[0] == PM4(4, IT_NOP, R_CX_REGS_INDIRECT) && pkt[1] == 4,
              "#1650 arm1: the builder reserved a Cx packet in a fresh ring");

        const uint64_t probes_before_ok = prosper_agc_patch_probe_count();
        p_addr(pkt_addr, 0x0000001200003400ull, 0, 0, 0, 0);
        p_add(pkt_addr, 3, 0, 0, 0, 0);
        CHECK(pkt[2] == 0x00003400u && pkt[3] == 0x00000012u,
              "#1650 arm1: matching-class PatchSetAddress still writes the array address");
        CHECK(pkt[1] == 7,
              "#1650 arm1: matching-class PatchAddRegisters still accumulates (4 -> 7)");
        CHECK(prosper_agc_patch_probe_count() == probes_before_ok,
              "#1650 arm1: an in-ring packet is accepted by range compare, with no OS write probe");

        // Arms 2-4 all REFUSE, and the refusal has to be visible in a run log. Capture stderr so
        // "refused" and "refused silently" cannot pass the same assertions.
        char log[4096];
        uint32_t victim[8];
        for (auto& v : victim) v = 0xA5A5A5A5u;      // ordinary writable memory, NOT one of our packets
        const uint32_t pkt1 = pkt[1], pkt2 = pkt[2], pkt3 = pkt[3];
        const uint64_t probes_before_refusals = prosper_agc_patch_probe_count();
        const uint64_t victim_addr = (uint64_t)(uintptr_t)victim;
        // Below the 0x1000 floor every guest-memory predicate rejects, and far below anything the
        // process maps: dereferencing it faults. On master both handlers stored straight through a
        // pointer like this, so without the fix this arm does not merely fail, it SIGSEGVs.
        constexpr uint64_t unmapped_addr = 0x800ull;

        const bool captured = capture_stderr([&] {
            // Arm 2 — the silent-arbitrary-write case itself: a mapped, writable pointer that is
            // not our packet. Kills "no header sub-op check".
            p_addr(victim_addr, 0xDEADBEEFCAFEBABEull, 0, 0, 0, 0);
            p_add(victim_addr, 99, 0, 0, 0, 0);
            // Arm 3 — a REAL packet of the WRONG register class. A single shared handler for
            // Cx/Sh/Uc cannot catch this, because the class is carried by the NID alone; kills
            // "keep one handler registered for all three NIDs" (exactly what master did).
            p_addr_sh(pkt_addr, 0x1111111122222222ull, 0, 0, 0, 0);
            p_add_sh(pkt_addr, 55, 0, 0, 0, 0);
            // Arm 4 — a pointer that cannot be dereferenced at all. Kills "no mappedness probe".
            p_addr(unmapped_addr, 0x3333333344444444ull, 0, 0, 0, 0);
            p_add(unmapped_addr, 77, 0, 0, 0, 0);
        }, log, sizeof log);
        CHECK(captured, "#1650: stderr capture around the refusal arms worked");

        bool victim_intact = true;
        for (const auto& v : victim) victim_intact &= (v == 0xA5A5A5A5u);
        CHECK(victim_intact,
              "#1650 arm2: a mapped non-packet pointer is left byte-for-byte alone");
        CHECK(pkt[1] == pkt1 && pkt[2] == pkt2 && pkt[3] == pkt3,
              "#1650 arm3: a wrong-class patcher does not touch a packet of another class");

        // Which guard fired matters as much as that one did. Arm 2's target is writable, so it must
        // be the HEADER check that refuses it — if guest_writable had wrongly rejected the stack,
        // the message would say "not writable" and this assertion fails loudly instead of passing
        // for the wrong reason.
        CHECK(strstr(log, "SetCxRegIndirectPatchSetAddress: header") != nullptr &&
              strstr(log, "SetCxRegIndirectPatchAddRegisters: header") != nullptr,
              "#1650 arm2: the refusal names the call site and the unexpected header, on stderr");
        CHECK(strstr(log, "SetShRegIndirectPatchSetAddress: header") != nullptr &&
              strstr(log, "SetShRegIndirectPatchAddRegisters: header") != nullptr,
              "#1650 arm3: the wrong-class refusal names the Sh call site, on stderr");
        CHECK(strstr(log, "is not writable guest memory") != nullptr,
              "#1650 arm4: an undereferenceable packet is reported, not silently skipped");
        CHECK(prosper_agc_patch_probe_count() > probes_before_refusals,
              "#1650: out-of-ring pointers DO reach the write probe — so arm1's zero-probe "
              "assertion measures the registry rather than a counter stuck at zero");

        // Arm 4b — the SAME undereferenceable-pointer case, for the six patchers #1650 left
        // behind (#2157): predication, the DMA dst/src pair, and the WriteData / WaitRegMem /
        // ReleaseMem sync patchers. Each read cmd[0] inside its header check before anything
        // validated the pointer, so on master this arm SIGSEGVs rather than failing.
        //
        // Two of them are the sharper ones and are why the spans are per-handler rather than one
        // constant: SetPacketPredication validates cmd[0] and then writes **cmd[4]**, and
        // ReleaseMem writes cmd[7]/cmd[8] on its long arm. A packet at the very end of a mapping
        // passes a header-sized probe and faults on the store, so those re-probe before the tail.
        {
            const struct { const char* nid; const char* name; } late[] = {
                { "w6Dj1VJt5qY", "SetPacketPredication"    },
                { "IxYiarKlXxM", "DmaDataPatchSetDst"      },
                { "cdDRpqcFGbU", "DmaDataPatchSetSrc"      },
                { "fPSCdQxgpSw", "WriteDataPatchAddress"   },
                { "3KDcnM3lrcU", "WaitRegMemPatchAddress"  },
                { "0fWWK5uG9rQ", "ReleaseMemPatchAddress"  },
            };
            char log2[8192] = {0};
            const bool captured2 = capture_stderr([&] {
                for (const auto& l : late) {
                    HleFn fn = Hle::lookup(l.nid);
                    if (fn) fn(unmapped_addr, 0x5555555566666666ull, 0, 0, 0, 0);
                }
            }, log2, sizeof log2);
            CHECK(captured2, "#2157: stderr capture around the late-patcher refusals worked");

            // Reaching this line at all is the primary assertion: on master the cmd[0] read faults.
            for (const auto& l : late) {
                char msg[200];
                snprintf(msg, sizeof msg, "#2157: %s is registered", l.name);
                CHECK(Hle::lookup(l.nid) != nullptr, msg);
                // Naming the call site matters as much as refusing: a shared message could not tell
                // a reader WHICH patcher was handed the bad pointer, and these six are reached from
                // different guest paths.
                snprintf(msg, sizeof msg,
                         "#2157: %s refuses an undereferenceable packet and names itself", l.name);
                CHECK(strstr(log2, l.name) != nullptr, msg);
            }
            CHECK(strstr(log2, "is not writable guest memory") != nullptr,
                  "#2157: the refusal is the writability probe, not an incidental header mismatch");
        }

        // Arm 5 — the diagnostic budget is PER CALL SITE.
        //
        // It used to be a single 8-message counter shared by every patcher family in the
        // translation unit, so eight refusals anywhere spent it for everyone and the first refusal
        // of a family that started misbehaving later never printed. That is the same defect class
        // this issue is about — prosper declining a guest write and saying nothing.
        //
        // This arm has to FLOOD one call site past the budget to see it, and that is the whole
        // point: the arms above produce only eight refusals in total across the entire run, which
        // is exactly the old shared budget, so every one of them still prints under the old code
        // and none of them can fail if the budget change is reverted. Counting refusals is not
        // optional here — a cheaper assertion that merely re-greps two names already checked above
        // cannot fail independently of the header check.
        char log2[16384];
        const bool captured2 = capture_stderr([&] {
            for (int i = 0; i < 16; ++i)                  // spend this one site's budget outright
                p_addr(victim_addr, 0xDEADBEEFCAFEBABEull, 0, 0, 0, 0);
            p_add_sh(victim_addr, 1, 0, 0, 0, 0);         // a DIFFERENT site, still owed a voice
        }, log2, sizeof log2);
        CHECK(captured2, "#1650 arm5: stderr capture around the budget flood worked");

        auto occurrences = [](const char* hay, const char* needle) {
            size_t n = 0;
            for (const char* p = strstr(hay, needle); p; p = strstr(p + 1, needle)) ++n;
            return n;
        };
        const size_t flooded = occurrences(log2, "SetCxRegIndirectPatchSetAddress");
        CHECK(flooded > 0 && flooded < 16,
              "#1650 arm5: a flooded call site is rate-limited — neither silent nor unbounded");
        CHECK(strstr(log2, "SetShRegIndirectPatchAddRegisters") != nullptr,
              "#1650 arm5: a DIFFERENT call site still reports after that flood — budgets are per "
              "call site, so one patcher family can no longer silence another");

        bool victim_still_intact = true;
        for (const auto& v : victim) victim_still_intact &= (v == 0xA5A5A5A5u);
        CHECK(victim_still_intact,
              "#1650 arm5: seventeen refused patches still wrote nothing");
        if (fails) printf("  captured stderr:\n%s\n  ---- flood ----\n%s", log, log2);
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

    // The sibling key. 0xef550356 is the record the same pipeline builder reads for render targets
    // 1..7, deriving each one's register offset as `record.offset + rtIndex`; 0xa6d12629 is the one
    // it reads for render target 0. They must therefore describe the SAME register and the SAME
    // default, or RT0 would start from a different blend state than RT1..7 in a title that programs
    // both. Pin them together so an edit to one that misses the other fails here rather than
    // showing up as an inter-target blend mismatch on some future title.
    ShaderRegister* blend_siblings = lookup_default(defaults, 0xef550356u);
    CHECK(blend_siblings && blend && blend_siblings[0].offset == 0x1e0u &&
          blend_siblings[0].value == 0x20010001u &&
          blend_siblings[0].offset == blend[0].offset &&
          blend_siblings[0].value == blend[0].value,
          "the RT1..7 blend key agrees with the RT0 key on register and default");

    // Guard a legacy multi-register run: the pointer must begin at the first register pair, not at
    // the preceding type hash. DQ copies these exact 16 bytes into its FOV defaults block.
    ShaderRegister* fov = lookup_default(defaults, 0x88f5e915u);
    CHECK(fov && fov[0].offset == 0xebu && fov[0].value == 0xff00ff00u &&
          fov[1].offset == 0xecu && fov[1].value == 0,
          "SDK 13 preserves legacy multi-register run pointers");

    // A pre-13 caller searches for the SAME render-target-zero key. This used to assert the
    // opposite — that an older caller must NOT find 0xa6d12629 — which pinned a defect rather
    // than a contract: the key was added while fixing DQ (which asks for 13), and the version
    // gate around it was incidental. The Oregon Trail (PPSA19244) asks for version 12, searches
    // this table for exactly this key at eboot+0x4ab2e00, and on a miss caches its all-ones
    // sentinel and emits its RT0 blend write with register offset 0xffffffff — which the command
    // processor drops, leaving CB_BLEND0_CONTROL at its blend-disabled default for every draw in
    // the title (#1946: Slate glyphs as solid blocks, logo on an opaque black panel).
    for (unsigned version : {0u, 8u, 12u}) {
        auto* older = static_cast<RegisterDefaults*>(prosper_agc_reg_defaults(version));
        ShaderRegister* older_blend = lookup_default(older, 0xa6d12629u);
        CHECK(older && older->count == 128 && older->unknown[0] == 0 && older->unknown[1] == 0 &&
              older_blend && older_blend[0].offset == 0x1e0u &&
              older_blend[0].value == 0x20010001u,
              "a pre-13 caller resolves the render-target-zero blend default too");
        // …and it agrees with the RT1..7 key on every version too, for the reason above.
        ShaderRegister* older_siblings = lookup_default(older, 0xef550356u);
        CHECK(older_siblings && older_blend && older_siblings[0].offset == 0x1e0u &&
              older_siblings[0].value == 0x20010001u &&
              older_siblings[0].offset == older_blend[0].offset &&
              older_siblings[0].value == older_blend[0].value,
              "a pre-13 caller's RT1..7 and RT0 blend keys agree");
        // The rest of the table must be untouched by that addition: same multi-register run,
        // same pointer arithmetic, for every version.
        ShaderRegister* older_fov = lookup_default(older, 0x88f5e915u);
        CHECK(older_fov && older_fov[0].offset == 0xebu && older_fov[0].value == 0xff00ff00u &&
              older_fov[1].offset == 0xecu && older_fov[1].value == 0,
              "a pre-13 caller keeps the legacy multi-register run pointers");
    }

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
