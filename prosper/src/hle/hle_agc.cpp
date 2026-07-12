// hle_agc.cpp — libSceAgc "Gen5" graphics-driver HLE: the Draw Command Buffer (Dcb) frontend.
//
// The AGC Dcb functions ARE the driver: each appends a PM4 packet into the game's command buffer and
// RETURNS the allocated dword pointer, which the game (and the indirect-patch helpers) then use. Our
// old stubs returned 0 -> the patch helpers wrote through a null pointer (the eboot+0x3b5ea6 fault).
//
// Ported/adapted from Kyty (../Kyty, MIT-licensed; source/emulator/src/Graphics/Graphics.cpp +
// Pm4.h) — the Dcb struct layout, PM4 encoding, and per-function packet contents are Kyty's, which
// reverse-engineered the real libSceAgc ABI. A later CommandProcessor will decode this PM4 stream to
// Vulkan (see docs/AGC_IMPL_PLAN.md). Registered by raw NID (AGC lib is undocumented).
#include "dispatch.hpp"
#include "gpu/pm4_registers.hpp"
#include "gpu/command_processor.hpp"
#include "gpu/pm4_decode.hpp"
#include "gpu/gpu_execute.hpp"
#include "gpu/gpu_timeline.hpp"
#include "gpu/videoout_present.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>

#ifndef _WIN32
// #312 label-slot write watch (exec_image_linux.cpp). Weak: tools that link the AGC HLE without
// the Linux host exec image simply skip the arm.
extern "C" void prosper_arm_label_watch(uint64_t addr) __attribute__((weak));
extern "C" void prosper_arm_label_watch_force(uint64_t addr) __attribute__((weak));
#endif
// #312 fence BUILD journal (command_processor.cpp): records, per fence packet, the guest-passed
// target + the target's content at build time + a timestamp — the timing-vs-wrong-target
// discriminator consulted by the stomp catcher when a fence write lands over freed heap.
extern "C" void prosper_fence_journal_record(uint64_t pkt, uint64_t addr);
// #312 per-label protocol history (command_processor.cpp): build-side notes for the consumed-marker
// legs (DmaData init / ReleaseMem fence / WaitRegMem poll), keyed by label address.
extern "C" void prosper_label_hist_dma_built(uint64_t addr, uint64_t cb, uint32_t src, uint8_t builder);
extern "C" void prosper_label_hist_rel_built(uint64_t addr, uint64_t cb);
extern "C" void prosper_label_hist_wait_built(uint64_t addr, uint64_t cb);

namespace prosper {

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)

namespace {

// --- PM4 encoding (Kyty Pm4.h) ---------------------------------------------------------------
constexpr uint32_t IT_NOP = 0x10, IT_INDEX_TYPE = 0x2A, IT_EVENT_WRITE = 0x46, IT_SET_SH_REG = 0x76;
// Custom sub-opcodes carried inside IT_NOP:
constexpr uint32_t R_DRAW_INDEX = 0x03, R_DRAW_INDEX_AUTO = 0x04, R_DRAW_RESET = 0x05, R_WAIT_FLIP_DONE = 0x06,
                   R_SH_REGS_INDIRECT = 0x11, R_CX_REGS_INDIRECT = 0x12, R_UC_REGS_INDIRECT = 0x13,
                   R_ACQUIRE_MEM = 0x14, R_WRITE_DATA = 0x15, R_WAIT_MEM_64 = 0x16, R_FLIP = 0x17,
                   R_RELEASE_MEM = 0x18, R_DMA_DATA = 0x19, R_DISPATCH_DIRECT = 0x1a,
                   R_INDEX_BASE = 0x1b, R_INDEX_COUNT = 0x1c, R_DRAW_INDEX_OFFSET = 0x1d,
                   R_JUMP = 0x1e, R_SET_PRED = 0x1f;
constexpr uint32_t R_NUM = 0x40;
inline uint32_t PM4(uint32_t len, uint32_t op, uint32_t r) {
    return 0xC0000000u | (((len - 2u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) | ((r & (R_NUM - 1u)) << 2u);
}

// --- The Draw Command Buffer (the game's own struct; we cast the guest pointer). Kyty layout
// (Graphics.cpp:777): a dword ring [bottom,top) with a write cursor and a "buffer full" callback. ---
struct AgcDcb {
    uint32_t* bottom;       // +0x00 buffer base
    uint32_t* top;          // +0x08 buffer end
    uint32_t* cursor_up;    // +0x10 current write ptr
    uint32_t* cursor_down;  // +0x18 reserve limit
    bool (*callback)(AgcDcb*, uint32_t, void*);  // +0x20 buffer-full callback (guest fn)
    void*     user_data;    // +0x28
    uint32_t  reserved_dw;  // +0x30

    uint32_t available_dw() const {
        auto avail = (uint32_t)(cursor_down - cursor_up);
        return avail > reserved_dw ? avail - reserved_dw : 0;
    }
    uint32_t* allocate_dw(uint32_t n) {
        if (n == 0) return nullptr;
        if (n > available_dw()) {
            if (!callback || !callback(this, n + reserved_dw, user_data)) return nullptr;
            if (available_dw() < n) return nullptr;
        }
        uint32_t* r = cursor_up;
        cursor_up += n;
        return r;
    }
};

// Helper: allocate `n` dwords and set the header; returns cmd (or nullptr).
inline uint32_t* begin_packet(uint64_t buf, uint32_t n, uint32_t op, uint32_t r, uint32_t** out) {
    auto* dcb = (AgcDcb*)(uintptr_t)buf;
    if (!dcb) { *out = nullptr; return nullptr; }
    // A type-3 PM4 packet's length is a 14-bit field encoded as (n-2), so a valid packet is 2..0x4001
    // dwords. An out-of-range n wraps the length field and emits a header claiming a tiny packet for a
    // physically huge one — mis-framing (truncating) the rest of the submit fold. This is the shared
    // choke point for every builder, so guarding it here covers WriteData, CbNop and the SetShReg range
    // at once (#450, the overflow sibling of #401's num==1 underflow). Reject rather than corrupt.
    if (n < 2u || n > 0x4001u) { *out = nullptr; return nullptr; }
    uint32_t* cmd = dcb->allocate_dw(n);
    if (cmd) cmd[0] = PM4(n, op, r);
    *out = cmd;
    return cmd;
}

// Verify a to-be-patched packet's own header sub-op before writing — a mismatch means the RE
// drifted, so log loudly (once) and refuse to corrupt the stream. Shared by the sync-packet
// address patchers (#213/#222) and SetPacketPredication (#319).
inline bool patch_check(uint32_t* cmd, uint32_t want_r, const char* who) {
    uint32_t r = (cmd[0] >> 2) & (R_NUM - 1u), op = (cmd[0] >> 8) & 0xffu;
    if (op == IT_NOP && r == want_r) return true;
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1) < 8)
        fprintf(stderr, "[agc] %s: header 0x%08x is not the expected packet (op=0x%x r=0x%x want r=0x%x) — patch refused\n",
                who, cmd[0], op, r, want_r);
    return false;
}

}  // namespace

// --- Dcb append functions (a0 = Dcb*, return = the allocated packet pointer). ------------------
HLE(agc_dcb_reset_queue) {  // (buf, op=0x3ff, state=0)
    uint32_t* cmd; if (!begin_packet(a0, 2, IT_NOP, R_DRAW_RESET, &cmd)) return 0;
    cmd[1] = 0; return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_wait_safe_for_rendering) {  // (buf, video_out_handle, display_buffer_index)
    uint32_t* cmd; if (!begin_packet(a0, 7, IT_NOP, R_WAIT_FLIP_DONE, &cmd)) return 0;
    cmd[1] = (uint32_t)a1; cmd[2] = (uint32_t)a2; cmd[3] = cmd[4] = cmd[5] = cmd[6] = 0;
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_set_sh_register_direct) {  // (buf, ShaderRegister reg)  reg packed in a1: offset|value<<32
    uint32_t* cmd; if (!begin_packet(a0, 3, IT_SET_SH_REG, 0, &cmd)) return 0;
    cmd[1] = (uint32_t)(a1 & 0xffffffffu); cmd[2] = (uint32_t)(a1 >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}
static uint64_t set_regs_indirect(uint64_t buf, uint64_t regs, uint64_t num_regs, uint32_t r) {
    uint32_t* cmd; if (!begin_packet(buf, 4, IT_NOP, r, &cmd)) return 0;
    cmd[1] = (uint32_t)num_regs; cmd[2] = (uint32_t)(regs & 0xffffffffu); cmd[3] = (uint32_t)(regs >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_set_cx_regs_indirect) { return set_regs_indirect(a0, a1, a2, R_CX_REGS_INDIRECT); }
HLE(agc_dcb_set_sh_regs_indirect) { return set_regs_indirect(a0, a1, a2, R_SH_REGS_INDIRECT); }
HLE(agc_dcb_set_uc_regs_indirect) { return set_regs_indirect(a0, a1, a2, R_UC_REGS_INDIRECT); }
HLE(agc_dcb_set_index_size) {  // (buf, index_size, cache_policy)
    uint32_t* cmd; if (!begin_packet(a0, 2, IT_INDEX_TYPE, 0, &cmd)) return 0;
    cmd[1] = (uint32_t)a1; return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_draw_index_auto) {  // (buf, index_count, modifier)
    uint32_t* cmd; if (!begin_packet(a0, 7, IT_NOP, R_DRAW_INDEX_AUTO, &cmd)) return 0;
    cmd[1] = (uint32_t)a1;
    // Store the draw modifier (ShaderDrawModifier bits) instead of dropping it, and zero the
    // packet tail — cmd[3..6] previously carried stale ring-buffer memory inside a packet whose
    // header says len=7 (poisoned GFXLOG dumps; live bug the moment a consumer reads past pl[0]).
    cmd[2] = (uint32_t)(a2 & 0xffffffffu); cmd[3] = (uint32_t)(a2 >> 32u);
    cmd[4] = cmd[5] = cmd[6] = 0;
    return (uint64_t)(uintptr_t)cmd;
}
// sceAgcDcbDrawIndex (NID q88lQ+GP5Yk, name via shadPS4 aerolib.inl) — the indexed-draw sibling of
// DrawIndexAuto: (Dcb* buf, uint32_t index_count, const void* indices, uint64_t modifier).
// Kyty has no Gen5 reference, but the a1/a2 roles are pinned by its Gen4 pair: GraphicsDrawIndex
// (Graphics.cpp:313) emits this same R_DRAW_INDEX custom packet as cmd[1]=index_count,
// cmd[2..3]=index_addr lo/hi, and its CommandProcessor consumer cp_op_draw_index
// (GraphicsRun.cpp:2757) reads buffer[0]=index_count, buffer[1..2]=index_addr — CONFIDENCE: HIGH.
// a3 follows the Gen5 Dcb convention of a trailing 64-bit ShaderDrawModifier (cf.
// GraphicsDcbDrawIndexAuto (buf, index_count, modifier), Graphics.cpp:1971), where Gen4 carried
// 32-bit flags+type instead — CONFIDENCE: MED, stored raw. pm4_decode decodes this packet as
// Kind::DrawIndex (issue #63); the index ELEMENT SIZE comes from the preceding SetIndexSize packet.
HLE(agc_dcb_draw_index) {
    uint32_t* cmd; if (!begin_packet(a0, 7, IT_NOP, R_DRAW_INDEX, &cmd)) return 0;
    cmd[1] = (uint32_t)a1;
    cmd[2] = (uint32_t)(a2 & 0xffffffffu); cmd[3] = (uint32_t)(a2 >> 32u);
    cmd[4] = (uint32_t)(a3 & 0xffffffffu); cmd[5] = (uint32_t)(a3 >> 32u);
    cmd[6] = 0;
    return (uint64_t)(uintptr_t)cmd;
}
// --- Gen5 indexed-draw builders (issue #232, DOLL/UE4 geometry path). ---------------------------
// DOLL's UE4 RHI submits ALL scene geometry through THREE builders that were unimplemented->0 (no
// packet appended), so every draw was silently dropped before the PM4 stream (0 DrawIndex in the
// histogram despite 5.4M calls/run each). The Messenger uses DrawIndex/DrawIndexAuto instead, so it
// never exercised this path. NIDs verified against the PS5 3.20 stub tables (../PS5-3.20_Libs):
//   l4fM9K-Lyks = sceAgcDcbSetIndexBuffer, 8N2tmT3jmC8 = sceAgcDcbSetIndexCount,
//   B+aG9DUnTKA = sceAgcDcbDrawIndexOffset.
// ABI: Kyty has no Gen5 reference for these, so args are pinned by the sce::Agc DrawCommandBuffer
// draw model + live capture (PROSPER_GFXLOG dumps a0..a3). setIndexBuffer(dcb, indexAddr);
// setIndexCount(dcb, count); drawIndexOffset(dcb, startIndex[, count]). The bound base+count thread
// through GpuState to the emitted indexed Draw (decoder R_INDEX_*; command_processor DrawIndexOffset).
// CONFIDENCE: MED (arg roles from the AGC draw model + live values; refined by the GFXLOG capture).
static std::atomic<uint64_t> g_ib_log{0};
HLE(agc_dcb_set_index_buffer) {  // (dcb, indexBufferAddr)
    if (getenv("PROSPER_GFXLOG") && g_ib_log.fetch_add(1) < 24)
        fprintf(stderr, "[agc] SetIndexBuffer dcb=0x%llx addr=0x%llx a2=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
    uint32_t* cmd; if (!begin_packet(a0, 3, IT_NOP, R_INDEX_BASE, &cmd)) return 0;
    cmd[1] = (uint32_t)(a1 & 0xffffffffu); cmd[2] = (uint32_t)(a1 >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}
static std::atomic<uint64_t> g_ic_log{0};
HLE(agc_dcb_set_index_count) {  // (dcb, indexCount)
    if (getenv("PROSPER_GFXLOG") && g_ic_log.fetch_add(1) < 24)
        fprintf(stderr, "[agc] SetIndexCount dcb=0x%llx count=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1);
    uint32_t* cmd; if (!begin_packet(a0, 2, IT_NOP, R_INDEX_COUNT, &cmd)) return 0;
    cmd[1] = (uint32_t)a1;
    return (uint64_t)(uintptr_t)cmd;
}
static std::atomic<uint64_t> g_do_log{0};
HLE(agc_dcb_draw_index_offset) {  // (dcb, startIndex[, indexCount])
    if (getenv("PROSPER_GFXLOG") && g_do_log.fetch_add(1) < 24)
        fprintf(stderr, "[agc] DrawIndexOffset dcb=0x%llx off=0x%llx a2=0x%llx a3=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1,
                (unsigned long long)a2, (unsigned long long)a3);
    uint32_t* cmd; if (!begin_packet(a0, 3, IT_NOP, R_DRAW_INDEX_OFFSET, &cmd)) return 0;
    cmd[1] = (uint32_t)a1;
    cmd[2] = (uint32_t)a2;   // explicit draw count if the builder passes one; 0 => use SetIndexCount
    return (uint64_t)(uintptr_t)cmd;
}

// --- Command-buffer call (Jump) + GPU predication (issue #319, the DOLL untextured-title cause). --
//
// DOLL's UE4 RHI records its per-frame BACKBUFFER COMPOSITE (the fullscreen-triangle pass that
// writes the textured scene into the VideoOut scanout buffer) NOT inline in the main Dcb, but in
// small side segments invoked via sceAgcDcbJump under sceAgcDcbSetPredication windows:
//
//   sceAgcDcbSetPredication(dcb, 1, 3, 1, condAddr)   <- begin window, 64-bit condition at condAddr
//   pkt = sceAgcDcbJump(dcb, 0, 0, target, ndw)       <- call `ndw` dwords at `target`
//   sceAgcSetPacketPredication(pkt, ...)              <- mark that jump packet as predicated
//   sceAgcDcbSetPredication(dcb, 1, 0, 1, 0)          <- end window
//
// Live capture (PROSPER_PREDLOG, title steady state): 8 such clusters per frame — the SAME two
// composite segments (one per scanout buffer 0x8fc0000000/0x8fc2000000) each invoked at 4 points
// with 4 distinct conditions; every segment is exactly `ndw` dwords of Set*RegsIndirect (full
// render state: CB_COLOR0_BASE = the scanout buffer, 3840x2160 viewport, ES PGM) + DrawIndexAuto(3)
// (a fullscreen triangle), with NO jump-back packet, and the parent stream continues after the
// cluster — so Jump is a CALL-WITH-LENGTH, not a goto. With these three calls unimplemented
// (return 0, no packet), the composite never executed and the presented backbuffer only ever held
// the direct-to-scanout UI rectangle draws — the #319 "untextured title".
//
// No Kyty/shadPS4 reference exists for these Gen5 calls; the ABI below is pinned by the live
// capture (arg roles consistent over 100k+ calls/run). CONFIDENCE: HIGH on (a3=target, a4=dword
// count) and (a4=condition address begin / 0 end) — the dword count equals the decoded segment
// length exactly, and the condition addresses are re-recorded per frame from a descending
// per-frame allocation. CONFIDENCE: MED on the remaining flag args (recorded raw in the packet).
HLE(agc_dcb_jump) {  // sceAgcDcbJump(dcb, ?, ?, target_addr, num_dw)
    if (getenv("PROSPER_PREDLOG")) {
        static std::atomic<uint64_t> n{0};
        uint64_t k = n.fetch_add(1);
        if (k < 32 || (k % 4096) == 0)
            fprintf(stderr, "[pred] DcbJump #%llu dcb=0x%llx target=0x%llx ndw=0x%llx a1=0x%llx a2=0x%llx\n",
                    (unsigned long long)k, (unsigned long long)a0, (unsigned long long)a3,
                    (unsigned long long)a4, (unsigned long long)a1, (unsigned long long)a2);
    }
    uint32_t* cmd; if (!begin_packet(a0, 5, IT_NOP, R_JUMP, &cmd)) return 0;
    cmd[1] = (uint32_t)(a3 & 0xffffffffu); cmd[2] = (uint32_t)(a3 >> 32u);
    cmd[3] = (uint32_t)a4;
    cmd[4] = 0;   // predicated flag — set by sceAgcSetPacketPredication on this returned packet
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_set_predication) {  // sceAgcDcbSetPredication(dcb, 1, op, 1, cond_addr) / (dcb, 1, 0, 1, 0)=end
    if (getenv("PROSPER_PREDLOG")) {
        static std::atomic<uint64_t> n{0};
        uint64_t k = n.fetch_add(1);
        if (k < 32 || (k % 4096) == 0)
            fprintf(stderr, "[pred] DcbSetPredication #%llu dcb=0x%llx cond=0x%llx op=0x%llx a1=0x%llx a3=0x%llx\n",
                    (unsigned long long)k, (unsigned long long)a0, (unsigned long long)a4,
                    (unsigned long long)a2, (unsigned long long)a1, (unsigned long long)a3);
    }
    uint32_t* cmd; if (!begin_packet(a0, 4, IT_NOP, R_SET_PRED, &cmd)) return 0;
    cmd[1] = (uint32_t)(a4 & 0xffffffffu); cmd[2] = (uint32_t)(a4 >> 32u);
    cmd[3] = (uint32_t)a2;
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_set_packet_predication) {  // sceAgcSetPacketPredication(packet, ...)
    // Marks an already-built packet as participating in the enclosing predication window. For our
    // R_JUMP encoding that is payload[3] (cmd[4]). Header-verified like the other packet patchers —
    // a different packet kind means the RE drifted, so refuse loudly rather than corrupt the stream.
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    if (!patch_check(cmd, R_JUMP, "SetPacketPredication")) return 0;
    cmd[4] = 1;
    return 0;
}

// sceAgcSuspendPoint (NID h9z6+0hEydk, name via shadPS4) — the TRC R5089 GPU suspend point the
// game forces in a loop ("Forcing call to sce::Agc::suspendPoint" spam). It has no out-params and
// no game-visible state; on a devkit it just gives the system a safe suspend opportunity.
// Returning 0 (OK) is the full contract for us — registered so it stops logging as unimplemented.
HLE(agc_suspend_point) { return 0; }

// --- #117 experiment: real appenders for the CreateWorkload Dcb builders. -----------------------
// sceAgcCbNop (LtTouSCZjHM) — append a1 padding dwords as a single NOP packet; return its pointer.
HLE(agc_cb_nop) {  // (dcb, num_dwords, ...)
    uint32_t num = (uint32_t)a1; if (!a0 || !num) return 0;
    // A type-3 NOP packet is at minimum 2 dwords (header + >=1 body). PM4() encodes the length as
    // (num-2), so num==1 would UNDERFLOW the 14-bit length field to 0x3fff — a header claiming ~0x4001
    // dwords for a physically 1-dword packet, which mis-frames (drops or truncates) the entire rest of
    // the submit fold (#401). A genuine 1-dword pad is a PM4 TYPE-2 filler NOP (0x80000000), which the
    // CP / our decoder now skips as a single dword — emit that instead of a malformed type-3 header.
    if (num == 1) {
        auto* dcb = (AgcDcb*)(uintptr_t)a0;
        uint32_t* cmd = dcb->allocate_dw(1);
        if (!cmd) return 0;
        cmd[0] = 0x80000000u;   // PM4 type-2 single-dword NOP filler
        return (uint64_t)(uintptr_t)cmd;
    }
    uint32_t* cmd; if (!begin_packet(a0, num, IT_NOP, 0, &cmd)) return 0;
    for (uint32_t i = 1; i < num; i++) cmd[i] = 0;
    return (uint64_t)(uintptr_t)cmd;
}
// sceAgcDcbDmaData (WmAc2MEj6Io) — append a CP-DMA packet (issue #312, the MallocBinned3 heap-
// corruption root cause). ABI RE-pinned from three eboot callsites (disassembly, this branch):
//   sceAgcDcbDmaData(dcb, srcAddressOrOffsetOrImmediate, srcSel?, dstSel?, dstAddressOrOffset,
//                    <sel/policy>, stack7, stack8, stack9=numBytes, ..., stack12=isBlocking)
// Evidence for the roles:
//  - eboot+0x221ad2d..6c: GMalloc->Malloc(size) result (+ running offset) is passed in A4 and the
//    packet fills it — a freshly-allocated buffer can only be a DESTINATION, so a4 = dst.
//  - the patcher family names are sceAgcDmaDataPatchSetDstAddressOrOffset (patches [1..2] below)
//    and sceAgcDmaDataPatchSetSrcAddressOrOffsetOrIMMEDIATE — src can be an immediate value.
//  - eboot+0x2212384 / 0x220d31e (the RHIThread translate loop's per-segment consumed-marker
//    emitters): DmaData(src=0, sels=3/3, dst=<chunk fence label>, stack9=4 bytes, blocking) right
//    before ReleaseMem(label <- 1, data_sel=1). This is the GPU-side LABEL INIT (label := 0): the
//    guest never CPU-initializes these labels (live page-watch capture), and the labels are LIFO-
//    recycled heap blocks — without the init, the guest's consumption poll reads the STALE 1 of the
//    previous generation at the same address, "retires" the segment instantly, frees the label while
//    this generation's fence packets are still in flight, and our (faithful) value-1 fence writes
//    then land in freed MallocBinned3 memory — the #312 "Canary was 0x3, should be 0x1" /
//    "free an unrecognized block 0x1000000001" / 0x20015f00 freelist-pop fatals.
// numBytes is stack arg9 — beyond the swap stub's 2-arg re-push window, recovered from the intact
// original guest frame exactly like agc_driver_submit_dcb_variant's arg9 (validated: the re-pushed
// args must match the originals and the return address must be guest text; on mismatch numBytes
// stays 0 and the CommandProcessor skips the packet with a loud log — the pre-fix behavior).
// Custom R_DMA_DATA payload: [1..2]=dst lo/hi, [3..4]=srcOrImm lo/hi, [5]=numBytes, [6]=sels.
// CONFIDENCE: HIGH on a4=dst/a1=srcOrImm (malloc-destination callsite + patcher names + protocol);
// MED on the selector args (recorded raw; the executor only honors the small-immediate form).
HLE(agc_dcb_dma_data) {  // (dcb, srcOrImm, srcSel?, dstSel?, dst, sel/policy, ..., stack9=numBytes)
    volatile uint64_t* fp = (uint64_t*)__builtin_frame_address(0);
    uint64_t num_bytes = 0;
    if (fp[6] == fp[2] && fp[7] == fp[3] &&
        fp[5] >= 0x400000000ull && fp[5] < 0x500000000ull) {
        uint64_t n = fp[8];                       // original stack arg9 = byte count
        if (n && n <= 0x10000000ull) num_bytes = n;
    }
    static std::atomic<uint64_t> g_dma_n{0};
    if (getenv("PROSPER_PREDLOG") || getenv("PROSPER_GFXLOG")) {
        uint64_t k = g_dma_n.fetch_add(1);
        if (k < 48 || (k % 4096) == 0)
            fprintf(stderr, "[pred] DcbDmaData #%llu dst=0x%llx srcOrImm=0x%llx bytes=0x%llx sels=0x%llx/0x%llx\n",
                    (unsigned long long)k, (unsigned long long)a4, (unsigned long long)a1,
                    (unsigned long long)num_bytes, (unsigned long long)a2, (unsigned long long)a3);
    }
    uint32_t* cmd; if (!begin_packet(a0, 7, IT_NOP, R_DMA_DATA, &cmd)) return 0;
    cmd[1] = (uint32_t)(a4 & 0xffffffffu); cmd[2] = (uint32_t)(a4 >> 32u);
    cmd[3] = (uint32_t)(a1 & 0xffffffffu); cmd[4] = (uint32_t)(a1 >> 32u);
    cmd[5] = (uint32_t)num_bytes;
    cmd[6] = (uint32_t)((a2 & 0xffu) | ((a3 & 0xffu) << 8));
    if (num_bytes <= 8) prosper_label_hist_dma_built(a4, a0, (uint32_t)a1, 1);   // #312 (label-init form)
    return (uint64_t)(uintptr_t)cmd;
}
// sceAgcAcbDmaData (-RnpfpxIhec) — the async-compute-queue sibling. Same operation, shifted ABI
// (RE'd from the Acb consumed-marker emitter at eboot+0x220d31e, which pairs it with the same
// sceAgcCbReleaseMem(label<-1): (acb, srcSel?=3, dstSel?=3, dst=label, 2, 3, stack7=srcOrImm=0,
// stack8=numBytes=4). srcOrImm/numBytes are stack args 7/8 = fp[2]/fp[3], INSIDE the swap stub's
// forwarded window. CONFIDENCE: MED (single callsite; the executor gate limits the blast radius).
HLE(agc_acb_dma_data) {  // (acb, srcSel?, dstSel?, dst, ?, ?, stack7=srcOrImm, stack8=numBytes)
    volatile uint64_t* fp = (uint64_t*)__builtin_frame_address(0);
    uint64_t src_imm = fp[2], num_bytes = fp[3];
    if (num_bytes > 0x10000000ull) num_bytes = 0;
    uint32_t* cmd; if (!begin_packet(a0, 7, IT_NOP, R_DMA_DATA, &cmd)) return 0;
    cmd[1] = (uint32_t)(a3 & 0xffffffffu); cmd[2] = (uint32_t)(a3 >> 32u);
    cmd[3] = (uint32_t)(src_imm & 0xffffffffu); cmd[4] = (uint32_t)(src_imm >> 32u);
    cmd[5] = (uint32_t)num_bytes;
    cmd[6] = (uint32_t)((a1 & 0xffu) | ((a2 & 0xffu) << 8));
    if (num_bytes <= 8) prosper_label_hist_dma_built(a3, a0, (uint32_t)src_imm, 2);   // #312
    return (uint64_t)(uintptr_t)cmd;
}
// sceAgcDmaDataPatchSetDstAddressOrOffset (IxYiarKlXxM) / ...SetSrcAddressOrOffsetOrImmediate
// (cdDRpqcFGbU): patch a built DMA_DATA packet's dst / src field (same patch-placeholder pattern
// as the #213 sync-packet address patchers; header-verified before writing).
HLE(agc_patch_dma_data_dst) {  // (cmd, address)
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    if (!patch_check(cmd, R_DMA_DATA, "DmaDataPatchSetDst")) return 0;
    cmd[1] = (uint32_t)(a1 & 0xffffffffu); cmd[2] = (uint32_t)(a1 >> 32u);
    return 0;
}
HLE(agc_patch_dma_data_src) {  // (cmd, addressOrImmediate)
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    if (!patch_check(cmd, R_DMA_DATA, "DmaDataPatchSetSrc")) return 0;
    cmd[3] = (uint32_t)(a1 & 0xffffffffu); cmd[4] = (uint32_t)(a1 >> 32u);
    return 0;
}
HLE(agc_dcb_event_write) {  // (buf, event_type, address)
    // Widen the packet to carry the address (a2) like ReleaseMem (#132): an address-carrying
    // EVENT_WRITE (the timestamp/label variant) writes a completion value there, and discarding it
    // left a guest waiting on that label blocked forever. cmd[1]=event_type, cmd[2..3]=address lo/hi.
    // An address-less event (a2=0, the pipeline-sync variants: partial-flush, cache-inval) stays a
    // no-op in the CommandProcessor (event_addr==0), exactly as before.
    uint32_t* cmd; if (!begin_packet(a0, 4, IT_EVENT_WRITE, 0, &cmd)) return 0;
    cmd[1] = (uint32_t)(a1 & 0xffu);
    cmd[2] = (uint32_t)(a2 & 0xffffffffu); cmd[3] = (uint32_t)(a2 >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_acquire_mem) {  // (buf, engine, cb_db_op, gcr_cntl, ...) — 8 dw
    uint32_t* cmd; if (!begin_packet(a0, 8, IT_NOP, R_ACQUIRE_MEM, &cmd)) return 0;
    cmd[1] = (uint32_t)a2; cmd[2] = cmd[3] = cmd[4] = cmd[5] = cmd[6] = 0; cmd[7] = (uint32_t)a3;
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_cb_release_mem) {  // sceAgcCbReleaseMem(buf, action, gcr_cntl, dst, cache_policy, address, data_sel, data, …)
    // ABI pinned to Kyty GraphicsCbReleaseMem (Graphics.cpp:1763): a5=label address, then the two stack
    // args are data_sel (7th) and the 64-bit fence value (8th). SysV: fp[1]=ret, fp[2]=data_sel, fp[3]=data.
    // Valid under BOTH stub shapes: the guest-%fs swap stub forwards the first two stack args
    // (exec_image_linux.cpp emit_swap_stub) — previously fp[2]/fp[3] were the saved guest-fs base and
    // guest return address there, so every fence under PROSPER_GUEST_FS was written with garbage.
    volatile uint64_t* fp = (uint64_t*)__builtin_frame_address(0);
    uint64_t data_sel = fp[2], data = fp[3];
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc] ReleaseMem action=0x%llx dst=0x%llx addr=0x%llx data_sel=0x%llx data=0x%llx\n",
            (unsigned long long)a1,(unsigned long long)a3,(unsigned long long)a5,
            (unsigned long long)data_sel,(unsigned long long)data);
    // #312 attribution probe: the heap-corrupting writes are 32-bit value-1 fences into 0x20-
    // stride label arrays. Log each DISTINCT guest callsite that builds one (first eboot-range
    // return address on the stack), so the owning guest subsystem can be disassembled. Bounded.
    if (data_sel == 1 && data == 1) {
#ifndef _WIN32
        // #312 label watch (PROSPER_WATCH_LABEL=1): watch the first heap-resident fence label's
        // block for the game's allocator free-path writing over it while cbs still signal it.
        // Under PROSPER_WATCH_ABS the passed address is ignored (fixed slot) — call unconditionally
        // so the arm retries until the fixed slot's page is mapped.
        if (prosper_arm_label_watch &&
            (getenv("PROSPER_WATCH_ABS") || (a5 >= 0x1000000000ull && a5 < 0x1200000000ull)))
            prosper_arm_label_watch(a5);
#ifndef _WIN32
        // PROSPER_WATCH_HOT=N (#312): arm the page watch on the Nth fence BUILD targeting the
        // same heap-resident label — a long-lived hot label is exactly the population that later
        // gets freed by the guest while still being fenced (the corruption class). Catches the
        // allocator's free-path writes over it WITH call stacks (see lwatch).
        static const long hot_n = [] { const char* e = getenv("PROSPER_WATCH_HOT");
                                       return e ? atol(e) : 0; }();
        if (hot_n > 0 && prosper_arm_label_watch_force && a5 >= 0x1000000000ull && a5 < 0x1200000000ull) {
            constexpr uint32_t kHotSize = 8192;                     // direct-mapped addr -> count
            static uint64_t hot_addr[kHotSize];
            static uint32_t hot_cnt[kHotSize];
            uint32_t s = (uint32_t)((a5 >> 4) * 2654435761u) & (kHotSize - 1);
            if (hot_addr[s] != a5) { hot_addr[s] = a5; hot_cnt[s] = 0; }
            if (++hot_cnt[s] == (uint32_t)hot_n) prosper_arm_label_watch_force(a5);
        }
#endif
#endif
        static std::atomic<int> seen_n{0};
        static uint64_t seen_ra[16];
        uint64_t ra = 0, ra2 = 0, ra3 = 0;
        for (int i = 1; i < 96; i++) {
            uint64_t v = fp[i];
            if (v >= 0x400000000ull && v < 0x4c0000000ull) {
                if (!ra) ra = v; else if (!ra2) ra2 = v; else { ra3 = v; break; }
            }
        }
        uint64_t key = ra2 ? ra2 : ra;
        if (key) {
            bool logged = false;
            int n = seen_n.load();
            for (int i = 0; i < n && i < 16; i++) if (seen_ra[i] == key) { logged = true; break; }
            if (!logged && n < 16) {
                seen_ra[seen_n.fetch_add(1) & 15] = key;
                fprintf(stderr, "[agc] fence1-builder ra=eboot+0x%llx ra2=eboot+0x%llx ra3=eboot+0x%llx addr=0x%llx action=0x%llx\n",
                        (unsigned long long)(ra - 0x400000000ull),
                        (unsigned long long)(ra2 ? ra2 - 0x400000000ull : 0),
                        (unsigned long long)(ra3 ? ra3 - 0x400000000ull : 0),
                        (unsigned long long)a5, (unsigned long long)a1);
            }
        }
    }
    // EOP fence: stash the label address (a5), the data_sel, the full 64-bit value, and the event action
    // into the packet so the CommandProcessor performs the completion write at SUBMIT time (correct
    // end-of-pipe timing — our GPU folds synchronously, so submit == pipe drain). CONFIDENCE: HIGH — the
    // arg positions are fixed by the AGC ABI; a5-as-label was independently confirmed (WaitRegMem polls it).
    uint32_t* cmd; if (!begin_packet(a0, 7, IT_NOP, R_RELEASE_MEM, &cmd)) return 0;
    cmd[1] = (uint32_t)(a5 & 0xffffffffu); cmd[2] = (uint32_t)(a5 >> 32u);
    cmd[3] = (uint32_t)data_sel;
    cmd[4] = (uint32_t)(data & 0xffffffffu); cmd[5] = (uint32_t)(data >> 32u);
    cmd[6] = (uint32_t)a1;
    if (data_sel == 1) {
        prosper_fence_journal_record((uint64_t)(uintptr_t)cmd, a5);   // #312 discriminator
        prosper_label_hist_rel_built(a5, a0);                         // #312 protocol history
    }
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_write_data) {  // sceAgcDcbWriteData(buf, dst, cache_policy, address_or_offset, data*, num_dwords, …)
    // ABI per Kyty GraphicsDcbWriteData (Graphics.cpp:2061): a1=dst, a3=addr, a4=data ptr, a5=num_dwords.
    const uint32_t* src = (const uint32_t*)(uintptr_t)a4;
    uint32_t num = (uint32_t)a5;
    // Real PM4 WRITE_DATA carries a 14-bit dword count — accept up to that, and REJECT (loudly)
    // anything larger as a mis-decoded arg. The previous silent clamp to 60 returned success while
    // dropping the tail of any larger write (e.g. a 256-byte descriptor table), leaving the
    // destination stale beyond dword 60 with no diagnostic. An accepted-but-large num cannot
    // overrun the ring: begin_packet -> allocate_dw() checks 5+num against available_dw() (with
    // the grow callback) and returns null on failure, which takes the `return 0` path below.
    // The emitted packet is 5 + num dwords, and a type-3 PM4 packet is at most 0x4001 dwords (14-bit
    // length field). Bound `num` against the PACKET limit, not the raw data count: num > 0x3FFC would
    // make 5+num overflow the length field and mis-frame the submit (#450). begin_packet re-checks the
    // same 0x4001 ceiling defensively, but reject early here with a clear diagnostic.
    if (num > 0x3FFCu) {
        fprintf(stderr, "[agc] WriteData REJECTED num_dwords=%u (5+num > PM4 max 0x4001 — mis-decoded arg?)\n", num);
        return 0;
    }
    if (getenv("PROSPER_GFXLOG")) fprintf(stderr, "[agc] WriteData dst=0x%llx addr=0x%llx num=%u src=%p\n",
        (unsigned long long)a1,(unsigned long long)a3, num, (const void*)src);
    // Copy the inline data into the packet so the CommandProcessor can perform the write at submit time.
    uint32_t* cmd; if (!begin_packet(a0, 5 + num, IT_NOP, R_WRITE_DATA, &cmd)) return 0;
    cmd[1] = (uint32_t)a1;
    cmd[2] = (uint32_t)(a3 & 0xffffffffu); cmd[3] = (uint32_t)(a3 >> 32u);
    cmd[4] = num;
    if (src) for (uint32_t i = 0; i < num; i++) cmd[5 + i] = src[i];
    if (num <= 4) prosper_fence_journal_record((uint64_t)(uintptr_t)cmd, a3);   // #312 discriminator
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_wait_reg_mem) {  // sceAgcDcbWaitRegMem(buf, size, compare_func, op, cache_policy, address, reference, mask, poll_cycles)
    // ABI pinned to Kyty GraphicsDcbWaitRegMem (Graphics.cpp:2096): a5 = label address; the 7th/8th
    // args (reference, mask) are STACK args, readable at fp[2]/fp[3] under both stub shapes (the
    // guest-%fs swap stub forwards two stack args). poll_cycles (9th) is beyond the forwarded
    // window and is only a poll-interval hint — encode 0. The payload previously ZEROED every
    // wait parameter at build time (address/ref/mask/function destroyed — the wait could never be
    // honored downstream); now it mirrors Kyty's layout exactly:
    // [1..2]=addr lo/hi, [3..4]=mask lo/hi, [5..6]=reference lo/hi, [7]=compare_function, [8]=interval.
    volatile uint64_t* fp = (uint64_t*)__builtin_frame_address(0);
    uint64_t reference = fp[2], mask = fp[3];
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc] WaitRegMem size=%llu func=%llu op=%llu addr=0x%llx ref=0x%llx mask=0x%llx\n",
            (unsigned long long)a1,(unsigned long long)a2,(unsigned long long)a3,
            (unsigned long long)a5,(unsigned long long)reference,(unsigned long long)mask);
    uint32_t* cmd; if (!begin_packet(a0, 9, IT_NOP, R_WAIT_MEM_64, &cmd)) return 0;
    cmd[1] = (uint32_t)(a5 & 0xffffffffu);        cmd[2] = (uint32_t)(a5 >> 32u);
    cmd[3] = (uint32_t)(mask & 0xffffffffu);      cmd[4] = (uint32_t)(mask >> 32u);
    cmd[5] = (uint32_t)(reference & 0xffffffffu); cmd[6] = (uint32_t)(reference >> 32u);
    cmd[7] = (uint32_t)a2;                        // compare_function
    cmd[8] = 0;                                   // poll interval hint (arg9 not forwarded; unused by our fold)
    prosper_fence_journal_record((uint64_t)(uintptr_t)cmd, a5);   // #312 discriminator (wait leg)
    prosper_label_hist_wait_built(a5, a0);                        // #312 protocol history
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_set_flip) {  // (buf, video_out_handle, display_buffer_index, flip_mode, flip_arg) — 6 dw
    uint32_t* cmd; if (!begin_packet(a0, 6, IT_NOP, R_FLIP, &cmd)) return 0;
    cmd[1] = (uint32_t)a1; cmd[2] = (uint32_t)a2; cmd[3] = (uint32_t)a3;
    cmd[4] = (uint32_t)(a4 & 0xffffffffu); cmd[5] = (uint32_t)(a4 >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}

// --- sceAgcCreateShader (NID f3dg2CSgRKY) — the shader-object constructor. ---------------------
//
// THE boot blocker's root (docs/GRAPHICS.md "register source"): Unity registers its ~36 built-in
// shaders at graphics init via eboot+0x14e74c0, which parses shader ELFs embedded in eboot rodata
// (e_machine=0xe0 EM_AMDGPU; sections .shader_header / .shader_text) and calls
// sceAgcCreateShader(&slot->shader, header, code) per shader — through the arg-validating wrapper
// eboot+0x3ae120 (rejects nulls + non-256-aligned code) that tail-jumps to this import. Our old
// stub returned 0 WITHOUT writing *dst, so every registry slot's shader stayed null — including
// [eboot+0x2048c60] (= slot base 0x2048c50 + 0x10), the "source" object whose null pointer flows
// through SetSource (eboot+0x3af400) into the register-context sub-objects and null-derefs at
// eboot+0x3b1562/0x3b5ea6. The Shader IS the register source: SetSource reads [src+0x08]
// (user_data), [src+0x28] (specials), [src+0x5a] (type) — the Shader field offsets below.
//
// Semantics per Kyty GraphicsCreateShader (Graphics.cpp:1432) — layout-verified against the real
// SDK blobs (file_header '1234', version 0x18, matching our captured gfx1030 shaders). The header's
// pointer fields are stored SELF-RELATIVE in the blob; the constructor relocates them in place
// (ptr += &ptr), binds the code pointer, and patches the shader-program base address into the
// leading SPI_SHADER_PGM_LO/HI sh-register pair. Beyond Kyty (which supports only ES/PS pairs and
// hard-aborts otherwise): we accept all five PGM pairs (PS/GS/ES/HS/CS), guard against double
// relocation, and never abort — unexpected shapes are logged and survive.
namespace {

struct AgcShaderRegister { uint32_t offset, value; };   // guest-visible (Kyty Shader.h:941)
struct AgcShaderUserData {           // guest-visible (Kyty Shader.h:911)
    uint16_t* direct_resource_offset;   // +0x00  self-relative until relocated
    void*     sharp_resource_offset[4]; // +0x08  (ShaderSharp*[4])
    uint16_t  eud_size_dw, srt_size_dw, direct_resource_count, sharp_resource_count[4];
};
struct AgcShader {                   // guest-visible (Kyty Shader.h:974)
    uint32_t           file_header;             // +0x00  '1234' = 0x34333231
    uint32_t           version;                 // +0x04  0x18
    AgcShaderUserData* user_data;               // +0x08  \.
    const void*        code;                    // +0x10   | SetSource consumes +0x08/+0x28/+0x5a
    AgcShaderRegister* cx_registers;            // +0x18   |
    AgcShaderRegister* sh_registers;            // +0x20   |
    void*              specials;                // +0x28  /
    void*              input_semantics;         // +0x30
    void*              output_semantics;        // +0x38
    uint32_t           header_size;             // +0x40
    uint32_t           shader_size;             // +0x44
    uint32_t           embedded_constant_buffer_size_dqw; // +0x48
    uint32_t           target;                  // +0x4c
    uint32_t           num_input_semantics;     // +0x50
    uint16_t           scratch_size_dw_per_thread; // +0x54
    uint16_t           num_output_semantics;    // +0x56
    uint16_t           special_sizes_bytes;     // +0x58
    uint8_t            type;                    // +0x5a
    uint8_t            num_cx_registers;        // +0x5b
    uint8_t            num_sh_registers;        // +0x5c
};
static_assert(offsetof(AgcShader, user_data) == 0x08 && offsetof(AgcShader, code) == 0x10 &&
              offsetof(AgcShader, specials) == 0x28 && offsetof(AgcShader, type) == 0x5a &&
              offsetof(AgcShader, num_sh_registers) == 0x5c, "AgcShader must match the SDK blob layout");

// AGC error codes (observed in the eboot wrapper at 0x3ae120).
constexpr uint64_t kAgcErrInvalidArg = 0x8a6c000aull;

// Shaders whose headers we already relocated (the fixup is not idempotent), and the registry of all
// created shaders — the AGC->Vulkan pipeline consumes this to find shader code by PGM base.
// The registry mutex covers CreateShader's push_back (guest loader threads) against the submit
// thread's lookup iteration: an unsynchronized push_back REALLOCATION frees the vector's backing
// store mid-iteration — a use-after-free read, not the "benign stale read" it was assumed to be.
std::unordered_set<const void*>& agc_relocated() { static std::unordered_set<const void*> s; return s; }
std::mutex&                      agc_shaders_mx() { static std::mutex m; return m; }
std::vector<const AgcShader*>&   agc_shaders()   { static std::vector<const AgcShader*> v; return v; }

// Relocate one self-relative pointer field in place (no-op for null fields).
template <class T> inline void agc_fix_ptr(T*& m) {
    if (m) m = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(m) + reinterpret_cast<uintptr_t>(&m));
}

}  // namespace

// Exposed for tests: how many shaders the guest successfully registered.
extern "C" size_t prosper_agc_shader_count() {
    std::lock_guard<std::mutex> lk(agc_shaders_mx());
    return agc_shaders().size();
}

// Look up a registered shader header by its bound code address (the SHADER_PGM base the executor
// recovers from the sh registers). Returns the AgcShader* (layout-compatible with gpu::AgcShaderHeader
// — file_header@0, user_data@0x08, code@0x10, type@0x5a) so the GPU executor can build the shader's
// resource table from its user_data descriptors. Null if no registered shader binds that code. The
// registry is written by guest loader threads (CreateShader) and read on the submit thread; both
// sides take the registry mutex (an unlocked read raced push_back's reallocation — UAF).
extern "C" const void* prosper_agc_shader_header_for_code(uint64_t code_addr) {
    if (!code_addr) return nullptr;
    std::lock_guard<std::mutex> lk(agc_shaders_mx());
    for (const AgcShader* h : agc_shaders())
        if (h && (uint64_t)(uintptr_t)h->code == code_addr) return h;
    return nullptr;
}

// Bounded RE probe (PROSPER_PIPETRACE): log the raw pointer args of the shader/pipeline construction
// calls, so their argument/return pointers can be diffed against a later fault object (e.g. the
// eboot+0xba6e08 pipeline object's [+0x140] companion). Purely observational.
static void pipetrace(const char* fn, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    if (getenv("PROSPER_PIPETRACE"))
        fprintf(stderr, "[pipe] %s a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx a4=0x%llx a5=0x%llx\n", fn,
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
}

static void pipetrace_shader_user_data(const char* tag, const AgcShader* h) {
    if (!getenv("PROSPER_PIPETRACE")) return;
    if (!h) { fprintf(stderr, "  [ud] %s shader=(null)\n", tag); return; }
    const AgcShaderUserData* ud = h->user_data;
    if (!ud) { fprintf(stderr, "  [ud] %s shader=%p user_data=(null)\n", tag, (const void*)h); return; }

    fprintf(stderr,
            "  [ud] %s shader=%p user_data=%p eud=%u srt=%u direct_count=%u sharp_counts={%u,%u,%u,%u}\n",
            tag, (const void*)h, (const void*)ud, ud->eud_size_dw, ud->srt_size_dw,
            ud->direct_resource_count, ud->sharp_resource_count[0], ud->sharp_resource_count[1],
            ud->sharp_resource_count[2], ud->sharp_resource_count[3]);

    if (ud->direct_resource_offset && ud->direct_resource_count && ud->direct_resource_count <= 64) {
        fprintf(stderr, "    direct offsets:");
        for (uint16_t i = 0; i < ud->direct_resource_count && i < 16; i++)
            fprintf(stderr, " type%u=%04x", i, ud->direct_resource_offset[i]);
        if (ud->direct_resource_count > 16) fprintf(stderr, " ...");
        fprintf(stderr, "\n");
    }

    for (int imm = 0; imm < 4; imm++) {
        const uint16_t count = ud->sharp_resource_count[imm];
        auto* raw = (const uint16_t*)ud->sharp_resource_offset[imm];
        if (!raw || count == 0 || count > 64) continue;
        const char* kind = (imm == 0) ? "texture" : (imm == 2) ? "sampler" :
                           (imm == 3) ? "storage" : "sharp1";
        fprintf(stderr, "    sharp[%d] %s:", imm, kind);
        for (uint16_t slot = 0; slot < count && slot < 16; slot++) {
            uint16_t v = raw[slot];
            fprintf(stderr, " slot%u={off=%04x,size=%u}", slot, (unsigned)(v & 0x7fffu), (unsigned)(v >> 15));
        }
        if (count > 16) fprintf(stderr, " ...");
        fprintf(stderr, "\n");
    }
}

HLE(agc_create_shader) {  // (Shader** dst, void* header, const void* code)
    pipetrace("CreateShader", a0, a1, a2, a3, a4, a5);
    auto** dst   = (AgcShader**)(uintptr_t)a0;
    auto*  h     = (AgcShader*)(uintptr_t)a1;
    auto*  code  = (const void*)(uintptr_t)a2;
    if (!dst || !h || !code) return kAgcErrInvalidArg;

    if (h->file_header != 0x34333231u || h->version != 0x18u) {
        fprintf(stderr, "[agc] CreateShader: unexpected header magic=0x%08x version=0x%x (want '1234'/0x18)\n",
                h->file_header, h->version);
        return kAgcErrInvalidArg;   // wrong layout model -- fail loudly rather than corrupt the blob
    }

    if (agc_relocated().insert(h).second) {
        agc_fix_ptr(h->cx_registers);
        agc_fix_ptr(h->sh_registers);
        agc_fix_ptr(h->user_data);
        agc_fix_ptr(h->specials);
        agc_fix_ptr(h->input_semantics);
        agc_fix_ptr(h->output_semantics);
        if (h->user_data) {
            agc_fix_ptr(h->user_data->direct_resource_offset);
            for (auto& sharp : h->user_data->sharp_resource_offset) agc_fix_ptr(sharp);
        }
    }
    h->code = code;

    // Patch the shader-program base into the leading SPI/COMPUTE PGM_LO/HI register pair. The blob
    // pre-seeds the pair's offsets, so match any known stage rather than gating on `type` (Kyty
    // handles only ES/PS and aborts otherwise; the game also registers other stages).
    using namespace prosper::agc::Pm4;
    static constexpr uint32_t kPgmPairs[][2] = {
        {SPI_SHADER_PGM_LO_PS, SPI_SHADER_PGM_HI_PS}, {SPI_SHADER_PGM_LO_ES, SPI_SHADER_PGM_HI_ES},
        {SPI_SHADER_PGM_LO_GS, SPI_SHADER_PGM_HI_GS}, {SPI_SHADER_PGM_LO_HS, SPI_SHADER_PGM_HI_HS},
        {COMPUTE_PGM_LO,       COMPUTE_PGM_HI},
    };
    uint64_t base = (uint64_t)(uintptr_t)code;
    bool patched = false;
    if (h->sh_registers && h->num_sh_registers >= 2) {
        for (const auto& pair : kPgmPairs) {
            if (h->sh_registers[0].offset == pair[0] && h->sh_registers[1].offset == pair[1]) {
                h->sh_registers[0].value = (uint32_t)((base >> 8) & 0xffffffffu);
                h->sh_registers[1].value = (uint32_t)((base >> 40) & 0xffu);
                patched = true;
                break;
            }
        }
    }
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc] CreateShader #%zu: header=%p code=%p type=%u target=0x%x size=%u "
                        "cx=%u sh=%u pgm_patched=%d\n",
                agc_shaders().size(), (void*)h, (void*)code, h->type, h->target, h->shader_size,
                h->num_cx_registers, h->num_sh_registers, (int)patched);

    // Semantic surfacing probe (PROSPER_PIPETRACE): logs the shader's input/output vertex semantics.
    // CORRECTION (2026-07-05, fresh 3-agent RE, ground-truth via eboot x86 disasm): an earlier theory
    // blamed the eboot+0xba6e08 GfxDevice fault on "empty semantics leaving the reflection table empty
    // via a VS->PS linkage predicate at eboot+0xd58710." That was WRONG. eboot+0xd58710 is Unity's
    // generic SafeBinaryRead::Transfer (asset deserialization; ~4955 call sites; references the
    // "SafeBinaryRead::BeginTransfer name mismatch" string) — unrelated to the crash. The 0xba6e08 fault
    // is a GfxDevice GC / deferred-release DRAIN pass (fn eboot+0xba6720) that, for a category-{5,9,15}
    // pipeline whose processed-flag [+0x1a0]==0, consults its GPU companion [+0x140] to decide keep-vs-
    // release — and ours is null. The companion is built eagerly on a normal PS5 path (writers in the
    // eboot 0xb3xxxx GfxDevice module); the fix is to make that companion get built (an AGC/GPU-resource
    // gap), NOT to surface semantics. See docs/GFXDEVICE_BRINGUP_PROBLEM.md §"2026-07-05 correction".
    if (getenv("PROSPER_PIPETRACE")) {
        fprintf(stderr, "  [sem] type=%u num_in=%u num_out=%u in_sem=%p out_sem=%p\n",
                h->type, h->num_input_semantics, h->num_output_semantics,
                (void*)h->input_semantics, (void*)h->output_semantics);
        auto dump = [](const char* tag, const void* arr, uint32_t n) {
            if (!arr || n == 0 || n > 64) { fprintf(stderr, "    %s: (none)\n", tag); return; }
            const uint32_t* s = (const uint32_t*)arr;
            fprintf(stderr, "    %s[%u]:", tag, n);
            for (uint32_t i = 0; i < n && i < 8; i++) fprintf(stderr, " %08x", s[i]);
            fprintf(stderr, "\n");
        };
        dump("in ", h->input_semantics,  h->num_input_semantics);
        dump("out", h->output_semantics, (uint32_t)h->num_output_semantics);
        // Raw header count region (0x50..0x5f) to confirm counts are read from the real blob, not a
        // relocation artifact: num_input_semantics u32@0x50, num_output_semantics u16@0x56.
        const uint8_t* hb = (const uint8_t*)h;
        fprintf(stderr, "    raw[0x50..0x5f]:");
        for (int i = 0x50; i < 0x60; i++) fprintf(stderr, " %02x", hb[i]);
        fprintf(stderr, "\n");
    }
    pipetrace_shader_user_data("CreateShader", h);

    { std::lock_guard<std::mutex> lk(agc_shaders_mx()); agc_shaders().push_back(h); }
    *dst = h;               // <- the write our old stub omitted; populates the shader-registry slots
    return 0;
}

// --- Shader-derived pipeline-state constructors + register-bank plumbing. ----------------------
//
// With CreateShader real, the eboot's register-context machinery runs its true path: it allocates
// "data packets" in the game's own Dcb and points each register-set sub-object's banks
// ([sub+0x10]/[sub+0x18]) at the packet payload. The leaf imports it needs are these four
// (semantics per Kyty Graphics.cpp:1606-1762, generalized — see each note).
namespace {

struct AgcShaderSemantic {           // guest-visible 32-bit bitfield (Kyty Shader.h:958)
    uint32_t semantic         : 8;
    uint32_t hardware_mapping : 8;
    uint32_t size_in_elements : 4;
    uint32_t is_f16           : 2;
    uint32_t is_flat_shaded   : 1;
    uint32_t is_linear        : 1;
    uint32_t is_custom        : 1;
    uint32_t static_vb_index  : 1;
    uint32_t static_attribute : 1;
    uint32_t reserved         : 1;
    uint32_t default_value    : 2;
    uint32_t default_value_hi : 2;
};
static_assert(sizeof(AgcShaderSemantic) == 4, "semantic must be one dword");

struct AgcShaderSpecialRegs {        // guest-visible (Kyty Shader.h:947); pointed to by Shader+0x28
    AgcShaderRegister ge_cntl;                  // +0x00
    AgcShaderRegister vgt_shader_stages_en;     // +0x08
    uint32_t          dispatch_modifier;        // +0x10
    uint16_t          user_data_range_start;    // +0x14  (consumed by SetSource @ eboot+0x3af400)
    uint16_t          user_data_range_end;      // +0x16
    uint64_t          draw_modifier;            // +0x18  (packed ShaderDrawModifier bitfields)
    AgcShaderRegister vgt_gs_out_prim_type;     // +0x20
    AgcShaderRegister ge_user_vgpr_en;          // +0x28
};
static_assert(offsetof(AgcShaderSpecialRegs, vgt_gs_out_prim_type) == 0x20, "specials layout");

}  // namespace

// sceAgcGetDataPacketPayloadAddress (V++UgBtQhn0) — called from INSIDE eboot's static AGC code
// (register-bank prepare, eboot+0x3af040 path, callsite +0x3af170): resolve a data packet built in
// the Dcb to its payload address, which becomes the register bank the guest reads/writes through.
// Both exercised data-packet types have two non-payload dwords (#140). Type 1 is SET_UCONFIG_REG:
// cmd[0] is its PM4 header and cmd[1] the register offset. Type 0 is a NOP data packet: cmd[0] is
// its PM4 header and cmd[1] the data-packet metadata/tag. In both cases the caller-visible payload
// begins at cmd+2. A live Dead Cells A/B confirmed the type-0 metadata role: returning cmd+1 shifts
// the caller's writes over that slot and immediately triggers its texture-memory fatal; cmd+2 boots.
HLE(agc_get_data_packet_payload) {  // (uint32_t** addr, uint32_t* cmd, int type)
    auto** addr = (uint32_t**)(uintptr_t)a0;
    auto*  cmd  = (uint32_t*)(uintptr_t)a1;
    if (!addr || !cmd) return kAgcErrInvalidArg;
    if (a2 == 0 || a2 == 1) { *addr = cmd + 2; return 0; }
    {
        static std::atomic<uint32_t> logged_mask{0};
        uint32_t bit = ((int)a2 >= 0 && (int)a2 < 31) ? (1u << (uint32_t)a2) : (1u << 31);
        if (!(logged_mask.fetch_or(bit, std::memory_order_relaxed) & bit))
            fprintf(stderr, "[agc] GetDataPacketPayloadAddress: unsupported packet type=%d\n", (int)a2);
    }
    return kAgcErrInvalidArg;
}

// sceAgcCbSetShRegisterRangeDirect (n2fD4A+pb+g) — append an IT_SET_SH_REG packet covering
// [offset, offset+num) with the given values (zeros if values==null); returns the packet pointer
// (the game patches/fills it afterwards). Kyty also emits a 2-dword NOP marker (payload 0x6875000d)
// first, mirroring the real library's output; we keep it for stream fidelity (NOP = inert).
HLE(agc_cb_set_sh_register_range_direct) {  // (buf, offset, values, num_values)
    uint32_t num = (uint32_t)a3;
    if (!a0 || !num) return 0;
    uint32_t* marker; if (begin_packet(a0, 2, IT_NOP, 0, &marker)) marker[1] = 0x6875000du;
    uint32_t* cmd;    if (!begin_packet(a0, num + 2, IT_SET_SH_REG, 0, &cmd)) return 0;
    cmd[1] = (uint32_t)a1;
    auto* values = (const uint32_t*)(uintptr_t)a2;
    if (values) memcpy(cmd + 2, values, (size_t)num * 4);
    else        memset(cmd + 2, 0, (size_t)num * 4);
    return (uint64_t)(uintptr_t)cmd;
}

// sceAgcCbSetShRegistersDirect (UZbQjYAwwXM) -- append a set of non-contiguous SH registers.
// Live Dead Cells imports this immediately before its compute dispatch stream; dropping the call
// leaves COMPUTE_PGM_* and compute user-data at zero. The ABI is (Dcb*, ShaderRegister*, count),
// where each entry is {offset,value}. Preserve caller order and combine only adjacent consecutive
// offsets, so duplicate/out-of-order writes retain their original semantics.
HLE(agc_cb_set_sh_registers_direct) {
    auto* regs = (const AgcShaderRegister*)(uintptr_t)a1;
    const uint32_t count = (uint32_t)a2;
    if (!a0 || !regs || !count || count > 4096) return 0;

    if (getenv("PROSPER_GFXLOG")) {
        static std::atomic<uint32_t> logged{0};
        const uint32_t n = logged.fetch_add(1, std::memory_order_relaxed);
        if (n < 16) {
            fprintf(stderr, "[agc] SetShRegistersDirect dcb=0x%llx regs=%p count=%u:",
                    (unsigned long long)a0, (const void*)regs, count);
            for (uint32_t i = 0; i < count && i < 24; ++i)
                fprintf(stderr, " (off=0x%x val=0x%x)", regs[i].offset, regs[i].value);
            if (count > 24) fprintf(stderr, " ...");
            fprintf(stderr, "\n");
        }
    }

    uint32_t* first = nullptr;
    uint32_t start = 0;
    while (start < count) {
        uint32_t end = start + 1;
        while (end < count && regs[end].offset == regs[end - 1].offset + 1) ++end;

        const uint32_t run_count = end - start;
        uint32_t* cmd = nullptr;
        if (!begin_packet(a0, run_count + 2, IT_SET_SH_REG, 0, &cmd)) return 0;
        if (!first) first = cmd;
        cmd[1] = regs[start].offset;
        for (uint32_t i = 0; i < run_count; ++i) cmd[2 + i] = regs[start + i].value;
        start = end;
    }
    return (uint64_t)(uintptr_t)first;
}

// sceAgcCreatePrimState (D9sr1xGUriE) — derive the primitive-pipeline registers from the bound
// geometry shader's "specials" block. Kyty hard-asserts hs==null and exact register offsets; we
// copy the specials through and log deviations instead (tessellation would arrive via hs).
HLE(agc_create_prim_state) {  // (cx_regs, uc_regs, hs, gs, prim_type)
    pipetrace("CreatePrimState", a0, a1, a2, a3, a4, a5);
    auto* cx = (AgcShaderRegister*)(uintptr_t)a0;
    auto* uc = (AgcShaderRegister*)(uintptr_t)a1;
    auto* gs = (const AgcShader*)(uintptr_t)a3;
    if (!cx || !uc || !gs || !gs->specials) return kAgcErrInvalidArg;
    if (a2 && getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc] CreatePrimState: hs shader present (tessellation) — not yet modeled\n");
    const auto* sp = (const AgcShaderSpecialRegs*)gs->specials;
    using namespace prosper::agc::Pm4;
    cx[0] = sp->vgt_shader_stages_en;
    cx[1] = sp->vgt_gs_out_prim_type;
    uc[0] = sp->ge_cntl;
    uc[1] = sp->ge_user_vgpr_en;
    uc[2] = {VGT_PRIMITIVE_TYPE, (uint32_t)a4};
    return 0;
}

// sceAgcCreateInterpolantMapping (HV4j+E0MBHE) — build the 32 SPI_PS_INPUT_CNTL_* registers wiring
// PS inputs to GS/VS output parameter slots. Generalized beyond Kyty (which asserts Unity's exact
// identity layout): match each PS input to the GS output with the same semantic id and use that
// output's parameter slot, with the flat-shade bit (0x400) from the PS side. Falls back to the
// identity mapping (slot i) when there is nothing to match — which reproduces Kyty's behaviour on
// the layouts Kyty supports.
HLE(agc_create_interpolant_mapping) {  // (ShaderRegister regs[32], const Shader* gs, const Shader* ps)
    pipetrace("CreateInterpolantMapping", a0, a1, a2, a3, a4, a5);
    auto* regs = (AgcShaderRegister*)(uintptr_t)a0;
    auto* gs   = (const AgcShader*)(uintptr_t)a1;
    auto* ps   = (const AgcShader*)(uintptr_t)a2;
    if (!regs || !gs) return kAgcErrInvalidArg;
    using namespace prosper::agc::Pm4;
    const auto* gs_out = (const AgcShaderSemantic*)gs->output_semantics;
    const auto* ps_in  = ps ? (const AgcShaderSemantic*)ps->input_semantics : nullptr;
    uint32_t n = ps ? ps->num_input_semantics : (uint32_t)gs->num_output_semantics;
    if (getenv("PROSPER_PIPETRACE"))
        fprintf(stderr, "  [interp] gs.num_out=%u ps=%p ps.num_in=%u -> mapping %u interpolants\n",
                (uint32_t)gs->num_output_semantics, (void*)ps,
                ps ? ps->num_input_semantics : 0u, n);
    pipetrace_shader_user_data("CreateInterpolant.gs", gs);
    pipetrace_shader_user_data("CreateInterpolant.ps", ps);
    for (uint32_t i = 0; i < 32; i++) {
        regs[i].offset = SPI_PS_INPUT_CNTL_0 + i;
        uint32_t value = 0;
        if (i < n) {
            uint32_t slot = i; bool flat = false;
            if (ps_in && gs_out) {
                flat = ps_in[i].is_flat_shaded != 0;
                for (uint32_t j = 0; j < gs->num_output_semantics; j++)
                    if (gs_out[j].semantic == ps_in[i].semantic) { slot = gs_out[j].hardware_mapping; break; }
            } else if (gs_out) {
                slot = gs_out[i].hardware_mapping;
            }
            value = slot | (flat ? 0x400u : 0u);
        }
        regs[i].value = value;
    }
    return 0;
}

// --- sceAgcDriverSubmitDcb (AgcDriver NID UglJIZjGssM) — the GPU submission point. -------------
//
// The game hands us a Packet {dcb_addr, dw_num} (Kyty Gen5Driver::Packet, Graphics.cpp:2168). We
// replay the PM4 stream through the CommandProcessor into a persistent GpuState — register files +
// draw list — which the AGC->Vulkan translation (render_state/vk_translate) consumes. No rendering
// happens here yet; this is the semantic execution of the submit, observable via PROSPER_GFXLOG and
// prosper_agc_submit_stats(). The Dcb address is a guest pointer, valid 1:1 in-process.
namespace {
gpu::GpuState& agc_gpu_state() { static gpu::GpuState st; return st; }
uint64_t g_submit_count = 0;
// Serializes every access to the shared GpuState (fold + render + stats). DOLL's UE4 RHI submits
// from TWO guest threads concurrently (core-proven, issue #278: one thread inside
// agc_driver_submit_dcb -> execute_and_present -> extract_render_state reading st.cx/sh/uc while
// another is inside agc_driver_submit_dcb_variant -> run_command_buffer writing them). An
// unordered_map rehash under a lock-free reader tears bucket pointers -> #GP with si_addr=0 (the
// "rip=0 / writer 0x0" steady-state crash that ended runs as the scene began). The real driver
// serializes ring submission internally — concurrent guest submits are legal and processed one at
// a time — so one mutex over the whole submit path IS the real contract, not a workaround.
// CONFIDENCE: HIGH (root cause proven by a two-thread core dump; contract matches driver behavior).
std::mutex g_agc_state_mu;
}

// EOP completion signaling (hle_kernel_time.cpp): fire the GPU end-of-pipe events the game registered
// via sceGnmAddEqEvent. Our fold is synchronous, so a completed SubmitDcb == the GPU pipe having drained.
void prosper_eq_trigger_eop();

// Total flips so far (hle_graphics.cpp) — part of the PROSPER_PROGRESS heartbeat below.
extern "C" uint64_t prosper_vo_flip_count();

namespace {
// PROSPER_PROGRESS=<secs>: emit a one-line timestamped forward-progress heartbeat at most every
// <secs> seconds, from the GPU submit path (both entry points). Purpose: long diagnostic runs need
// a cheap progression signal (is the frame loop advancing? are draws/dispatches still growing?)
// without PROSPER_GFXLOG's per-packet firehose (GBs over a 10-minute run). Called under
// g_agc_state_mu. Additive, default-off. CONFIDENCE: HIGH (diagnostic only, no behavior change).
uint64_t g_presents_cum = 0;   // frames handed to the present path (bumped at each render call site)
void progress_heartbeat(size_t draws_last, uint64_t submits, uint64_t dispatches) {
    static const long interval = [] {
        const char* e = getenv("PROSPER_PROGRESS"); return e ? atol(e) : 0; }();
    if (interval <= 0) return;
    static uint64_t draws_cum = 0;
    draws_cum += draws_last;
    static const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    static std::chrono::steady_clock::time_point next = t0;
    auto now = std::chrono::steady_clock::now();
    if (now < next) return;
    next = now + std::chrono::seconds(interval);
    double t = std::chrono::duration<double>(now - t0).count();
    fprintf(stderr, "[progress] t=%.1fs submits=%llu draws_last=%zu draws_cum=%llu dispatches=%llu flips=%llu presents=%llu\n",
            t, (unsigned long long)submits, draws_last, (unsigned long long)draws_cum,
            (unsigned long long)dispatches, (unsigned long long)prosper_vo_flip_count(),
            (unsigned long long)g_presents_cum);
    // PROSPER_PROGRESS_UNIMPL=1: with the heartbeat on, also dump the unimplemented-NID call-count
    // table every ~12 heartbeats. Purpose: the first-seen unimpl log is deduped, so a guest POLLING
    // an unimplemented call (the classic "waits on a wrong stub answer" stall) is invisible without
    // the per-call counts — and a `timeout`-killed diagnostic run never reaches the exit dump.
    static const bool dump_unimpl = getenv("PROSPER_PROGRESS_UNIMPL") != nullptr;
    static unsigned hb = 0;
    if (dump_unimpl && (hb++ % 12) == 0) dump_call_log(stderr);
}
}

extern "C" void prosper_agc_submit_stats(uint64_t* submits, uint64_t* draws) {
    std::lock_guard<std::mutex> lk(g_agc_state_mu);
    if (submits) *submits = g_submit_count;
    if (draws)   *draws   = (uint64_t)agc_gpu_state().draws.size();
}

namespace {
// Barrier-model release watchdog (#312): deferred streams were only re-checked at SUBSEQUENT
// submits, but a guest that CPU-polls a deferred stream's label before submitting again deadlocks
// (observed: boot wedged at submit #1 under the first WAIT_DEFER experiment). A 2 ms ticker
// re-runs the flush (same submit mutex), releasing a paused stream's gated writes the moment its
// condition satisfies (typically via the pend-queue completion writes) or its timeout expires.
// Started lazily the first time a deferred stream exists; inert while no stream is paused.
void start_defer_watchdog() {
    static std::atomic<bool> started{false};
    bool expect = false;
    if (!started.compare_exchange_strong(expect, true)) return;
    std::thread([] {
        for (;;) {
            struct timespec ts{ 0, 2000000 };   // 2 ms cadence
            nanosleep(&ts, nullptr);
            // EOP pulses fire at submit time (liveness); the flush only applies deferred writes.
            std::lock_guard<std::mutex> lk(g_agc_state_mu);
            gpu::flush_deferred_streams();
        }
    }).detach();
}
}

static bool execute_submit_work(gpu::GpuState& st, uint64_t submit_no, unsigned& draw_submits) {
    // Native-speed semantic capture happens before renderer sampling. PROSPER_RENDER_EVERY and
    // warmup may skip Vulkan work, but must not erase submit/present history from the timeline.
    gpu::record_gpu_timeline_submit(st, submit_no);
    static const unsigned render_every = [] {
        const char* e = getenv("PROSPER_RENDER_EVERY");
        long v = e ? atol(e) : 1;
        return v > 0 ? (unsigned)v : 1u;
    }();
    const bool render = gpu::have_submit_renderer() && !st.draws.empty() &&
                        (draw_submits++ % render_every) == 0;
    if (!render) {
        if (gpu::have_submit_compute() && !st.dispatches.empty() &&
            !gpu::execute_compute_dispatches(st, submit_no) && getenv("PROSPER_COMPUTELOG"))
            fprintf(stderr, "[compute] submit #%llu produced no executable work\n",
                    (unsigned long long)submit_no);
        return false;
    }

    uint32_t w = gpu::present_width(), h = gpu::present_height();
    static const uint32_t scale = [] {
        const char* e = getenv("PROSPER_RENDER_SCALE");
        long v = e ? atol(e) : 1;
        return v > 0 ? (uint32_t)v : 1u;
    }();
    if (scale > 1) {
        w = (w / scale) & ~1u;
        h = (h / scale) & ~1u;
        if (!w) w = 2;
        if (!h) h = 2;
    }
    // Draw spans and dispatches are synchronous. Drain packet writes before the renderer reads guest
    // resources, then retire completion only after the ordered mixed timeline has finished.
    gpu::flush_deferred_streams();
    prosper_gpu_drain_completion_writes();
    const bool presented = gpu::execute_ordered_and_present(st, w, h, submit_no);
    if (presented) g_presents_cum++;
    if (presented && getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc] SubmitDcb #%llu: executed %zu draws -> presented %ux%u frame\n",
                (unsigned long long)submit_no, st.draws.size(), w, h);
    return presented;
}

// Fold one raw Dcb dword stream through the CommandProcessor, fire the GPU EOP completion events, and
// (if a live renderer is wired and the submit produced draws) execute+present. Shared by the primary
// submit path (sceAgcDriverSubmitDcb) and the DOLL warmup-submit variant (w1KFAHVqpaU) below.
// `dw_num == 0` means "length unknown" — decode_pm4 self-terminates at the first non-type-3 dword, and
// we cap the walk at kUnknownCap dwords as a runaway guard (a bounded ring can't legitimately exceed it).
static uint64_t submit_dcb_stream(const uint32_t* addr, uint32_t dw_num, const char* who) {
    if (!addr) return 0;
    constexpr uint32_t kUnknownCap = 0x80000;   // 512 KiB of dwords — well past any real Dcb
    uint32_t walk = dw_num ? dw_num : kUnknownCap;
    std::lock_guard<std::mutex> lk(g_agc_state_mu);   // serialize with the other submit entry (#278)
    // #312: flush any earlier stream paused on a WAIT_REG_MEM — this submit may be its producer.
    gpu::flush_deferred_streams();
    agc_gpu_state().draws.clear();
    agc_gpu_state().dispatches.clear();
    gpu::begin_gpu_timeline_submit(g_submit_count + 1);
    size_t applied = gpu::run_command_buffer(addr, walk, agc_gpu_state());
    g_submit_count++;
    gpu::diagnose_compute_dispatches(agc_gpu_state(), g_submit_count);
    static unsigned draw_submits2 = 0;
    execute_submit_work(agc_gpu_state(), g_submit_count, draw_submits2);
    gpu::diagnose_resource_provenance(agc_gpu_state(), g_submit_count);
    // #312 EOP visibility contract: the pulse fires immediately only when no gated writes are
    // pending; otherwise it is OWED and delivered when the tail drains (the guest's completion
    // scan must never observe a half-retired frame — see command_processor.cpp). The flip and the
    // pipe-drain writes still flow, and the watchdog + timeout bound the delay, so the pacer
    // stays alive (the naive always-pulse variant let the scan free live labels; the naive
    // never-pulse variant starved the pacer — both measured).
    gpu::flush_deferred_streams();
    gpu::submit_completion_pulse();
    // Watchdog keys off PENDING streams, not just this fold: streams can outlive their fold (and
    // the Jump-recursion flag reset once hid a deferring fold entirely — the wedge class).
    if (gpu::deferred_pending()) start_defer_watchdog();
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc] %s #%llu: %u dwords (walk=%u) -> %zu packets applied (draws: %zu, dispatches: %llu)\n",
                who, (unsigned long long)g_submit_count, dw_num, walk, applied,
                agc_gpu_state().draws.size(), (unsigned long long)agc_gpu_state().dispatch_count);
    progress_heartbeat(agc_gpu_state().draws.size(), g_submit_count,
                       (uint64_t)agc_gpu_state().dispatch_count);
    return 0;
}

// sceAgc submit variant (NID w1KFAHVqpaU) — DOLL's UE4 RHI submits its per-frame command buffers
// through TWO driver entry points: intermediate buffers via sceAgcDriverSubmitDcb (UglJIZjGssM,
// folded above) and the FINAL buffer of a SubmitCommandBuffers batch through THIS one (the guest's
// SubmitCommandBuffers impl at eboot+0x220a9a0 loops the buffer array, calling the indirect submit for
// buffers [0..n-2] and w1KFAHVqpaU for buffer n-1). Left unimplemented, the final buffer's GPU EOP
// fences (ReleaseMem writes) never executed, so the label the RenderThread polls (eboot+0x221d6c2,
// FRenderCommandFence/RHI frame-sync) stayed 0 forever and the GameThread timed out — the #232 wall.
//
// ABI (RE'd from the compiler-generated adapter thunk at eboot+0x58df3f0, which marshals DOLL's
// internal call at eboot+0x220aace into the Sony import): the raw Dcb dword-stream address arrives as
// stack arg8, which the guest-%fs swap stub forwards to fp[3] (the same slot ReleaseMem/WaitRegMem read
// their 8th arg from). The dword COUNT is stack arg9: the callsite loads the buffer-array entry
// {addr @+0x00, dw_count32 @+0x10} (`mov 0x10(%rax,%r12,1),%r10d`) and the adapter pushes it as the
// import's 9th arg (32-bit, matching Packet.dw_num on the primary UglJIZjGssM path).
//
// arg9 is beyond the swap stub's 2-arg re-push window, but the ORIGINAL guest stack frame is still
// intact above the stub's pushes. Guest-path stub frame at handler entry ([rsp] up):
//   ret-to-stub, arg7-copy, arg8-copy, saved-r11(guest fs), ret-to-guest, orig-arg7, orig-arg8, orig-arg9
// i.e. with the handler's rbp frame: fp[2]=arg7, fp[3]=arg8, fp[4]=saved r11, fp[5]=ret-to-guest,
// fp[6]=orig arg7, fp[7]=orig arg8, fp[8]=orig arg9. We take arg9 from fp[8] ONLY after validating the
// frame shape (fp[7]==fp[3] — the re-pushed arg8 must equal the original — and fp[5] must return into
// guest code): a mismatch means a different stub layout (host path / future stub change), and we fall
// back to the self-terminating fold rather than trust a garbage count.
//
// WHY the exact count matters (#241 and the boot crash-zoo): the Dcb is carved from a live ring whose
// STALE tail is still valid type-3 PM4 from previous frames. An unknown-length fold does not stop at
// the logical end — it re-executes the stale packets, including stale WriteData/ReleaseMem fence writes
// whose destination heap blocks the guest has since freed and reused. That scribbles 8-byte GPU values
// into live allocator state (the deterministic 0x20015f00 free-list node of issue #241, the libc.prx
// logger stack-smash) — intermittent because it depends on what got reallocated under the stale fences.
// CONFIDENCE: HIGH on the arg9=count ABI (callsite + adapter disassembly agree, and the count matches
// the primary path's Packet.dw_num field offset); the validated-frame walk falls back safely.
HLE(agc_driver_submit_dcb_variant) {
    auto is_pm4 = [](uint64_t p) -> bool {
        if (p < 0x10000 || (p & 3)) return false;
        uint32_t h = *(const volatile uint32_t*)(uintptr_t)p;
        return (h & 0xC0000000u) == 0xC0000000u;
    };
    volatile uint64_t* fp = (uint64_t*)__builtin_frame_address(0);
    uint64_t cand = fp[3];                                  // arg8 = the Dcb stream address (adapter ABI)
    if (!is_pm4(cand)) {                                    // fallback: scan the register args
        const uint64_t regs[6] = { a0, a1, a2, a3, a4, a5 };
        cand = 0;
        for (uint64_t r : regs) if (is_pm4(r)) { cand = r; break; }
    }
    if (!cand) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 4)
            fprintf(stderr, "[agc] w1KFAHVqpaU: no PM4 stream found (fp[3]=0x%llx a0=0x%llx a1=0x%llx) — submit refused\n",
                    (unsigned long long)fp[3], (unsigned long long)a0, (unsigned long long)a1);
        return 0;
    }
    // Recover the true dword count (arg9) from the original guest frame, validated as described above.
    // Guest code range for the return-address check: main image at 0x400000000 (see classify_addr).
    uint32_t dw_num = 0;
    if (cand == fp[3] && fp[7] == fp[3] &&
        fp[5] >= 0x400000000ull && fp[5] < 0x500000000ull) {
        uint64_t n = fp[8];
        if (n && n <= 0x400000ull)                          // plausible: bounded by any real ring size
            dw_num = (uint32_t)n;
    }
    if (!dw_num) {
        static std::atomic<int> logged9{0};
        if (logged9.fetch_add(1) < 4)
            fprintf(stderr, "[agc] w1KFAHVqpaU: arg9 count unavailable (fp[5]=0x%llx fp[7]=0x%llx fp[8]=0x%llx) — self-terminating fold\n",
                    (unsigned long long)fp[5], (unsigned long long)fp[7], (unsigned long long)fp[8]);
    }
    return submit_dcb_stream((const uint32_t*)(uintptr_t)cand, dw_num, "SubmitDcbFinal");
}

HLE(agc_driver_submit_dcb) {  // (const Packet* packet)
    struct Packet { uint32_t* addr; uint32_t dw_num; uint8_t pad[4]; };
    const auto* p = (const Packet*)(uintptr_t)a0;
    if (!p || !p->addr || !p->dw_num) return kAgcErrInvalidArg;
    std::lock_guard<std::mutex> lk(g_agc_state_mu);   // serialize with submit_dcb_stream/w1KFAHVqpaU (#278)
    // Reset the per-submit draw list BEFORE folding this Dcb. The folded GpuState is process-lifetime and
    // its register files (cx/sh/uc) persist across submits (as on real hardware), but its `draws` vector
    // must NOT accumulate — otherwise it grows unbounded and every later frame re-renders stale geometry.
    // Clearing here (not after) keeps this submit's draws inspectable once the handler returns. (Ported
    // from PRs #31/#32.)
    // #312: flush any earlier stream paused on a WAIT_REG_MEM — this submit may be its producer.
    gpu::flush_deferred_streams();
    agc_gpu_state().draws.clear();
    agc_gpu_state().dispatches.clear();
    gpu::begin_gpu_timeline_submit(g_submit_count + 1);
    size_t applied = gpu::run_command_buffer(p->addr, p->dw_num, agc_gpu_state());
    g_submit_count++;
    gpu::diagnose_compute_dispatches(agc_gpu_state(), g_submit_count);
    static unsigned draw_submits = 0;
    execute_submit_work(agc_gpu_state(), g_submit_count, draw_submits);
    gpu::diagnose_resource_provenance(agc_gpu_state(), g_submit_count);
    // The submit has "completed" (synchronous fold): fire any registered GPU EOP events. Inert unless the
    // game called sceGnmAddEqEvent (b0xyllnVY-I); the RELEASE_MEM label write already happened in apply().
    // #312 EOP visibility contract: pulse only when no gated writes are pending, else owed until
    // the tail drains (see submit_dcb_stream / command_processor.cpp).
    gpu::flush_deferred_streams();
    gpu::submit_completion_pulse();
    if (gpu::deferred_pending()) start_defer_watchdog();
    if (getenv("PROSPER_GFXLOG")) {
        fprintf(stderr, "[agc] SubmitDcb #%llu: %u dwords -> %zu packets applied (draws so far: %zu, dispatches total: %llu)\n",
                (unsigned long long)g_submit_count, p->dw_num, applied, agc_gpu_state().draws.size(),
                (unsigned long long)agc_gpu_state().dispatch_count);
        // Dump each packet's kind + first payload dwords so we can see whether the game embeds a
        // completion-label write (WriteData/ReleaseMem/EventWrite) or a GPU-side wait in the stream.
        std::vector<gpu::Pm4Command> ops; gpu::decode_pm4(p->addr, p->dw_num, ops);
        static const char* kKindName[] = {"DrawReset","WaitFlipDone","SetShRegDirect","SetRegsIndirect",
            "SetIndexType","DrawIndex","DrawIndexAuto","EventWrite","AcquireMem","WriteData","WaitRegMem",
            "Flip","ReleaseMem","DispatchDirect","SetIndexBase","SetIndexCount","DrawIndexOffset",
            "Jump","SetPredication","DmaData","Unknown"};
        for (auto& c : ops) {
            uint32_t k = (uint32_t)c.kind; const char* nm = k < 21 ? kKindName[k] : "?";
            fprintf(stderr, "[agc]   pkt op=0x%02x r=0x%02x len=%u kind=%s pl0=0x%08x pl1=0x%08x pl2=0x%08x\n",
                    c.op, c.r, c.len, nm,
                    c.len > 1 ? c.payload[0] : 0, c.len > 2 ? c.payload[1] : 0, c.len > 3 ? c.payload[2] : 0);
        }
    }
    progress_heartbeat(agc_gpu_state().draws.size(), g_submit_count,
                       (uint64_t)agc_gpu_state().dispatch_count);
    return 0;
}

// --- Indirect-register patch helpers: modify a packet returned by a Set*RegsIndirect call.
// cmd = a0 (that returned pointer). Old stub returned 0 for Set*RegsIndirect -> these wrote to null. -
HLE(agc_patch_set_address) {  // (cmd, regs): cmd[2..3] = regs vaddr
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    cmd[2] = (uint32_t)(a1 & 0xffffffffu); cmd[3] = (uint32_t)(a1 >> 32u); return 0;
}
HLE(agc_patch_add_registers) {  // (cmd, num_regs): cmd[1] += num_regs
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    cmd[1] += (uint32_t)a1; return 0;
}

// --- Sync-packet address patch family (issue #213/#222 — the DOLL per-draw fence cluster). ------
//
// RE'd from DOLL's statically-linked AGC fence builder (eboot+0x59a1780, gdb-verified live): the
// guest allocates a LABEL inside a Dcb NOP data packet (sceAgcCbNop + sceAgcGetDataPacketPayload-
// Address), pre-builds WriteData(label:=0) / ReleaseMem(label<-1) / WaitRegMem(label==1) packets
// with NULL placeholder addresses, then patches the label address into each packet through these
// three imports (call order fPS->3KD->0fW; the per-draw burst passes each packet at exactly our
// builders' emitted offsets: WriteData at +0x1c(6dw), ReleaseMem at +0x34(7dw), WaitRegMem at
// +0x50(9dw) after the 7-dword NOP label packet — the offset arithmetic pins which patch targets
// which packet kind). Signature is (cmd, address) — the fence-builder disassembly passes exactly
// two args; the varying a2..a4 seen under GFXLOG are deterministic register residue from the PLT
// thunks. Leaving these as no-ops was the RenderThread stall: the GPU-side label writes all
// targeted address 0 (skipped), so the engine's CPU-side poll of the label never completed and
// "GameThread timed out waiting for RenderThread after 120.00 secs" fired. Each patcher verifies
// the packet's own header sub-op before writing — a mismatch means the RE drifted, so it logs
// loudly (once) and refuses to corrupt the stream. CONFIDENCE: HIGH (disassembly + live capture
// + offset arithmetic all agree).
HLE(agc_patch_write_data_addr) {  // fPSCdQxgpSw (cmd, address): WriteData payload [1..2] = address
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    if (!patch_check(cmd, R_WRITE_DATA, "WriteDataPatchAddress")) return 0;
    cmd[2] = (uint32_t)(a1 & 0xffffffffu); cmd[3] = (uint32_t)(a1 >> 32u);
    prosper_fence_journal_record((uint64_t)(uintptr_t)cmd, a1);   // #312: target set post-build
    return 0;
}
HLE(agc_patch_wait_reg_mem_addr) {  // 3KDcnM3lrcU (cmd, address): WaitRegMem payload [0..1] = address
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    if (!patch_check(cmd, R_WAIT_MEM_64, "WaitRegMemPatchAddress")) return 0;
    cmd[1] = (uint32_t)(a1 & 0xffffffffu); cmd[2] = (uint32_t)(a1 >> 32u);
    prosper_fence_journal_record((uint64_t)(uintptr_t)cmd, a1);   // #312: target set post-build
    return 0;
}
HLE(agc_patch_release_mem_addr) {  // 0fWWK5uG9rQ (cmd, address): ReleaseMem payload [0..1] = address
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    if (!patch_check(cmd, R_RELEASE_MEM, "ReleaseMemPatchAddress")) return 0;
    cmd[1] = (uint32_t)(a1 & 0xffffffffu); cmd[2] = (uint32_t)(a1 >> 32u);
    prosper_fence_journal_record((uint64_t)(uintptr_t)cmd, a1);   // #312: target set post-build
    return 0;
}

// sceAgcDriverGetResourceRegistrationMaxNameLength. Dead Cells asks for this before formatting
// every resource name into a dynamic stack buffer. The generic success stub left *out untouched;
// stale stack data became a multi-gigabyte alloca and faulted at eboot+0x172cebd (#539). Its
// registration configuration passes 0xfc as the maximum, matching the driver-visible limit here.
// The error value is inherited from the PS4 resource-registration ABI; PS5 retains the API family.
// CONFIDENCE: HIGH on the pointer/output contract and 0xfc value (live caller + core), MED on error.
HLE(agc_driver_get_resource_registration_max_name_length) {
    if (!a0)
        return (uint64_t)(int64_t)(int32_t)0x80D19013u;
    *(uint32_t*)(uintptr_t)a0 = 0xfcu;
    return 0;
}

// sceAgcCbDispatch (NID k3GhuSNmBLU) — compute dispatch.
// The authoritative PS5 3.20 export name is sceAgcCbDispatch, and its ABI is
// (dcb, thread_count_x, thread_count_y, thread_count_z, dispatch_modifier). Dead Cells' bound
// fill kernel is decisive: `index = (TGID_X << 6) + local_id_x`, local size is 64x1x1, and a1
// exactly equals its V# record count (0x1fe000 / 0xa000 / 0x87000), while a4 is the small 0x21
// modifier. Preserve thread counts here; the executor derives ceil(threads/local_size) groups
// from the retained register snapshot (#580). Kyty's Gen4 GraphicsDispatchDirect is a differently
// named API and is not an oracle for this Gen5 builder. DOLL's first real frame issues ~535 of
// these (UE4's compute prologue: GPUScene build, clears, culling) before any graphics draw. Appending
// the packet keeps the stream faithful; the CommandProcessor retains it for ordered submit execution.
// CONFIDENCE: HIGH (PS5 symbol table + live kernel disassembly, register state, and descriptor sizes).
HLE(agc_cb_dispatch) {  // (buf, thread_count_x, thread_count_y, thread_count_z, dispatch_modifier)
    uint32_t* cmd; if (!begin_packet(a0, 6, IT_NOP, R_DISPATCH_DIRECT, &cmd)) return 0;
    cmd[1] = (uint32_t)a1; cmd[2] = (uint32_t)a2; cmd[3] = (uint32_t)a3;
    cmd[4] = (uint32_t)(a4 & 0xffffffffu); cmd[5] = (uint32_t)(a4 >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}

void register_agc_hle() {
    #define RN(nid, fn) Hle::register_fn(nid, (HleFn)(fn), nid)
    RN("f3dg2CSgRKY", agc_create_shader);   // sceAgcCreateShader — populates the shader registry
    RN("uJziRsODk1c", agc_driver_get_resource_registration_max_name_length);
    RN("V++UgBtQhn0", agc_get_data_packet_payload);          // data packet -> register-bank payload
    RN("n2fD4A+pb+g", agc_cb_set_sh_register_range_direct);  // SET_SH_REG range packet
    RN("UZbQjYAwwXM", agc_cb_set_sh_registers_direct);       // non-contiguous SET_SH_REG packets
    RN("D9sr1xGUriE", agc_create_prim_state);                // prim registers from gs specials
    RN("HV4j+E0MBHE", agc_create_interpolant_mapping);       // SPI_PS_INPUT_CNTL_* wiring
    RN("TRO721eVt4g", agc_dcb_reset_queue);
    RN("MWiElSNE8j8", agc_dcb_wait_safe_for_rendering);
    RN("ZvwO9euwYzc", agc_dcb_set_cx_regs_indirect);
    RN("-HOOCn0JY48", agc_dcb_set_sh_regs_indirect);
    RN("hvUfkUIQcOE", agc_dcb_set_uc_regs_indirect);
    RN("GIIW2J37e70", agc_dcb_set_index_size);
    RN("Yw0jKSqop+E", agc_dcb_draw_index_auto);
    RN("q88lQ+GP5Yk", agc_dcb_draw_index);      // sceAgcDcbDrawIndex (see handler comment)
    // Gen5 indexed-draw path (issue #232, DOLL/UE4 geometry — was unimplemented->0, all draws dropped).
    RN("l4fM9K-Lyks", agc_dcb_set_index_buffer); // sceAgcDcbSetIndexBuffer
    RN("8N2tmT3jmC8", agc_dcb_set_index_count);  // sceAgcDcbSetIndexCount
    RN("B+aG9DUnTKA", agc_dcb_draw_index_offset);// sceAgcDcbDrawIndexOffset
    RN("h9z6+0hEydk", agc_suspend_point);       // sceAgcSuspendPoint — no-op OK
    // Command-buffer call + GPU predication (#319 — the DOLL backbuffer-composite path).
    RN("xSAR0LTcRKM", agc_dcb_jump);               // sceAgcDcbJump (call-with-length)
    RN("bbFueFP+J4k", agc_dcb_set_predication);    // sceAgcDcbSetPredication (window begin/end)
    RN("w6Dj1VJt5qY", agc_set_packet_predication); // sceAgcSetPacketPredication (mark jump packet)
    RN("aJf+j5yntiU", agc_dcb_event_write);
    RN("57labkp+rSQ", agc_dcb_acquire_mem);
    RN("wr23dPKyWc0", agc_cb_release_mem);
    RN("i1jyy49AjXU", agc_dcb_write_data);
    RN("VmW0Tdpy420", agc_dcb_wait_reg_mem);
    // Indirect-register patch helpers (SetAddress patches cmd[2..3]; AddRegisters patches cmd[1]).
    RN("vcmNN+AAXnY", agc_patch_set_address);   RN("d-6uF9sZDIU", agc_patch_add_registers);   // Cx
    RN("Qrj4c+61z4A", agc_patch_set_address);   RN("z2duB-hHQSM", agc_patch_add_registers);   // Sh
    RN("6lNcCp+fxi4", agc_patch_set_address);   RN("vRoArM9zaIk", agc_patch_add_registers);   // Uc
    RN("LtTouSCZjHM", agc_cb_nop);              // sceAgcCbNop — append padding dwords (#117)
    // Sync-packet address patchers + compute dispatch (issue #213 — the DOLL per-draw fence
    // cluster; RE write-up at each handler). These OVERRIDE the glog no-op thunks (map insert
    // order: register_agc_hle runs after the glog registration, same as WmAc2MEj6Io).
    RN("fPSCdQxgpSw", agc_patch_write_data_addr);    // WriteData packet: set destination address
    RN("3KDcnM3lrcU", agc_patch_wait_reg_mem_addr);  // WaitRegMem packet: set label address
    RN("0fWWK5uG9rQ", agc_patch_release_mem_addr);   // ReleaseMem packet: set label address
    RN("k3GhuSNmBLU", agc_cb_dispatch);              // sceAgcCbDispatch (threads x/y/z, modifier)
    RN("WmAc2MEj6Io", agc_dcb_dma_data);        // sceAgcDcbDmaData — append DMA_DATA packet (#117/#312)
    RN("-RnpfpxIhec", agc_acb_dma_data);        // sceAgcAcbDmaData — async-compute sibling (#312)
    RN("IxYiarKlXxM", agc_patch_dma_data_dst);  // sceAgcDmaDataPatchSetDstAddressOrOffset
    RN("cdDRpqcFGbU", agc_patch_dma_data_src);  // sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate
    RN("YUeqkyT7mEQ", agc_dcb_set_flip);        // sceAgcDcbSetFlip (Kyty LibGraphicsDriver.cpp:98)
    RN("UglJIZjGssM", agc_driver_submit_dcb);   // sceAgcDriverSubmitDcb -> CommandProcessor replay
    RN("w1KFAHVqpaU", agc_driver_submit_dcb_variant);  // final-buffer submit variant (#232, DOLL RHI batch)
    #undef RN
}

}  // namespace prosper
