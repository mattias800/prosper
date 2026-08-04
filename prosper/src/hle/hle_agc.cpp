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
#include "../host/boot_program.hpp"   // #1659: shared guest-module labelling
#include "hle_kernel_time.hpp"
#include "gpu/pm4_registers.hpp"
#include "gpu/command_processor.hpp"
#include "gpu/pm4_decode.hpp"
#include "gpu/gpu_execute.hpp"
#include "gpu/gpu_timeline.hpp"
#include "gpu/videoout_present.hpp"
#include "gpu/mb3_freelist.hpp"   // #1226: per-submit POOLSHIFT window probe
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>
#include <unordered_map>

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

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define HLE9(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6, \
                                       uint64_t a7, uint64_t a8)

namespace {

// --- PM4 encoding (Kyty Pm4.h) ---------------------------------------------------------------
constexpr uint32_t IT_NOP = 0x10, IT_INDEX_TYPE = 0x2A, IT_NUM_INSTANCES = 0x2F,
                   IT_EVENT_WRITE = 0x46, IT_SET_CONTEXT_REG = 0x69,
                   IT_SET_SH_REG = 0x76, IT_SET_UCONFIG_REG = 0x79;
// Custom sub-opcodes carried inside IT_NOP:
constexpr uint32_t R_DRAW_INDEX = 0x03, R_DRAW_INDEX_AUTO = 0x04, R_DRAW_RESET = 0x05, R_WAIT_FLIP_DONE = 0x06,
                   R_PUSH_MARKER = 0x0b, R_POP_MARKER = 0x0c,
                   R_SH_REGS_INDIRECT = 0x11, R_CX_REGS_INDIRECT = 0x12, R_UC_REGS_INDIRECT = 0x13,
                   R_ACQUIRE_MEM = 0x14, R_WRITE_DATA = 0x15, R_WAIT_MEM_64 = 0x16, R_FLIP = 0x17,
                   R_RELEASE_MEM = 0x18, R_DMA_DATA = 0x19, R_DISPATCH_DIRECT = 0x1a,
                   R_INDEX_BASE = 0x1b, R_INDEX_COUNT = 0x1c, R_DRAW_INDEX_OFFSET = 0x1d,
                   R_JUMP = 0x1e, R_SET_PRED = 0x1f,
                   R_SET_BASE_INDIRECT_ARGS = 0x20, R_STALL_COMMAND_BUFFER_PARSER = 0x21,
                   R_DRAW_INDEX_INDIRECT = 0x22, R_DISPATCH_INDIRECT = 0x23;
constexpr uint32_t R_NUM = 0x40;

// --- Packet sizes, in DWORDS, shared by each builder and its sceAgc*GetSize (#1143, #1748, #1756) ---
//
// These are ONE definition used twice on purpose. A guest reserves command-buffer space by calling
// GetSize and then calls the builder, so the two must agree exactly; a GetSize that under-reports
// makes the builder overrun the reservation, which does not look like a size bug when it fails (on
// #1748 it looked like a 210 MiB/s heap leak). test_agc_getsize MEASURES every builder's real cursor
// advance and asserts its GetSize equals it, so a literal typed here twice could not drift silently
// — but a single constant cannot drift at all.
//
// Where prosper's encoding differs from the hardware packet (the register-indirect forms carry a
// pointer + count instead of the register values; ReleaseMem carries a #312 build snapshot in its
// spare slot), the constant is still what makes the guest self-consistent: it reserves exactly what
// prosper writes. Emitting FEWER dwords than hardware is safe — it only wastes reserved space.
// Emitting MORE is the fatal direction, and is what #1748 was.
constexpr uint32_t kDwDrawIndex          = 7;
constexpr uint32_t kDwDrawIndexAuto      = 7;
constexpr uint32_t kDwDrawIndexOffset    = 3;
constexpr uint32_t kDwDrawIndexIndirect  = 4;
constexpr uint32_t kDwDispatch           = 6;
constexpr uint32_t kDwDispatchIndirect   = 4;
// DMA_DATA is 7: header + dst lo/hi, srcOrImm lo/hi, numBytes, selectors — exactly the RDNA2
// `DMA_DATA` field list, with no slot to spare. It was 9 because it also carried the #312 build-time
// snapshot of the destination qword: prosper's own invention, not a decoded guest field, so removing
// it needs no title evidence about the size and 9 -> 7 is the safe direction for the guest (a
// reservation can only gain room). All of the risk is internal, and it is a REMOVAL, not a
// relocation — the #312 label-history event survives via prosper_label_hist_dma_built, but the
// pre-image value dma_build_pre_changed compares does not.
//
// Anything that stamps or validates a DMA_DATA header MUST use this constant. #1124's
// dma_patch_recover_header rebuilds a clobbered header from scratch, and a hard-coded 9 there would
// write a 9-dword header over a 7-dword allocation — the CP walk would step two dwords into the next
// packet and desync the rest of the submit, on Alex Kidd (PPSA02664, rung 6) specifically, and no
// draw-count A/B can see that.
constexpr uint32_t kDwDmaData            = 7;
constexpr uint32_t kDwEventWrite         = 4;
constexpr uint32_t kDwSetIndexBuffer     = 3;
constexpr uint32_t kDwSetIndexCount      = 2;
constexpr uint32_t kDwSetIndexSize       = 2;
constexpr uint32_t kDwSetNumInstances    = 2;
constexpr uint32_t kDwStallCbParser      = 2;
constexpr uint32_t kDwSetRegisterDirect  = 3;
constexpr uint32_t kDwSetRegsIndirect    = 4;
constexpr uint32_t kDwAcquireMem         = 8;
constexpr uint32_t kDwReleaseMem         = 8;
constexpr uint32_t kDwJump               = 5;
constexpr uint64_t kAgcErrInvalidArg = 0x8a6c000aull;
constexpr uint64_t kAgcErrInvalidShaderHalves = 0x8a6c0008ull;
inline uint32_t PM4(uint32_t len, uint32_t op, uint32_t r) {
    return 0xC0000000u | (((len - 2u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) | ((r & (R_NUM - 1u)) << 2u);
}

// --- #1756: is prosper about to overrun a reservation the guest sized from the REAL AGC packet? ---
//
// prosper's builders append into the GUEST's command buffer, and a guest reserves that buffer from
// the packet sizes it was compiled against. A builder that emits more dwords than the hardware
// packet it stands for therefore overruns the reservation — and the failure does not look like a
// packet-size problem. On Asterix & Obelix - Babylon Mission (#1748) it looked like a 210 MiB/s heap
// leak: the append did not fit, the Dcb's "buffer full" callback fired, and the guest's chunk
// allocator handed over a fresh ~600 KiB command-buffer chunk for every submit until direct memory
// was gone.
//
// The defect is invisible in most titles, and that asymmetry is why it needs a probe rather than a
// crash to find it: a guest that calls `sceAgc*GetSize` is TOLD prosper's size and reserves exactly
// what prosper will write, so it stays self-consistent at any size. Only a guest with the sizes
// inlined from the AGC headers — which cannot be told — is exposed. The bug is therefore silent in
// every title that asks and fatal in the one that does not, and the next affected title may not
// crash at all: it may simply churn command-buffer chunks and run slowly.
//
// Two gated reports, both off by default and neither able to gate or alter a single write:
//
//   PROSPER_DCBFULL=1  every buffer-full callback, with the sub-op that triggered it and the dwords
//                      needed vs available. A SMALL window is the signature: a multi-hundred-KiB
//                      buffer filling up is ordinary growth, while a window of a few dozen dwords is
//                      a guest reservation prosper has just overrun.
//   PROSPER_DCBWIN=N   every packet appended into a window of N dwords or fewer (N=1 means the
//                      default 64). A tightly sized window states the guest's own per-packet budget
//                      in hardware sizes, so its contents ARE the reference for the packets in it.
struct AgcDcb;
const char* dcb_subop_name(uint32_t r);
void dcb_report_full(const AgcDcb* dcb, uint32_t need);
void dcb_report_window(const AgcDcb* dcb, uint32_t n, uint32_t op, uint32_t r);
void dcb_diag_tick(bool was_full);
inline bool dcb_diag_full() {
    static const bool v = getenv("PROSPER_DCBFULL") != nullptr;
    return v;
}
inline uint32_t dcb_diag_win() {
    static const uint32_t v = [] {
        const char* e = getenv("PROSPER_DCBWIN");
        if (!e) return 0u;
        const long n = strtol(e, nullptr, 0);
        if (n <= 0) return 0u;                 // explicit 0 (or junk) means OFF, not "the default"
        return n > 1 ? (uint32_t)n : 64u;      // PROSPER_DCBWIN=1 selects the default 64
    }();
    return v;
}
// Which builder is inside allocate_dw right now, so the buffer-full report can name it. Per-thread:
// several guest threads build into their own Dcbs concurrently, and a shared global would attribute
// one thread's overrun to another's packet. 0xffffffff = a raw allocate_dw with no sub-op (CbNop,
// the type-2 filler).
//
// `thread_local` is safe here even though guest threads run with the guest's `%fs` (see the aliasing
// warning in hle_kernel.cpp): an HLE only ever runs behind an import stub, and the stub swaps back to
// the HOST `%fs` for the duration of the call. Both writer and reader are inside that window, in the
// same call, on the same thread.
constexpr uint32_t kDcbDiagRaw = 0xffffffffu;
inline thread_local uint32_t g_dcb_diag_r = kDcbDiagRaw;
inline thread_local uint32_t g_dcb_diag_op = 0;

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
            dcb_report_full(this, n);          // #1756 probe; no-op unless PROSPER_DCBFULL
            if (!callback || !callback(this, n + reserved_dw, user_data)) return nullptr;
            if (available_dw() < n) return nullptr;
        }
        uint32_t* r = cursor_up;
        cursor_up += n;
        return r;
    }
};

const char* dcb_subop_name(uint32_t r) {
    switch (r) {
        case R_DRAW_INDEX:                 return "DrawIndex";
        case R_DRAW_INDEX_AUTO:            return "DrawIndexAuto";
        case R_DRAW_RESET:                 return "ResetQueue";
        case R_WAIT_FLIP_DONE:             return "WaitUntilSafeForRendering";
        case R_PUSH_MARKER:                return "PushMarker";
        case R_POP_MARKER:                 return "PopMarker";
        case R_SH_REGS_INDIRECT:           return "SetShRegsIndirect";
        case R_CX_REGS_INDIRECT:           return "SetCxRegsIndirect";
        case R_UC_REGS_INDIRECT:           return "SetUcRegsIndirect";
        case R_ACQUIRE_MEM:                return "AcquireMem";
        case R_WRITE_DATA:                 return "WriteData";
        case R_WAIT_MEM_64:                return "WaitRegMem";
        case R_FLIP:                       return "SetFlip";
        case R_RELEASE_MEM:                return "ReleaseMem/EopAction";
        case R_DMA_DATA:                   return "DmaData";
        case R_DISPATCH_DIRECT:            return "Dispatch";
        case R_INDEX_BASE:                 return "SetIndexBuffer";
        case R_INDEX_COUNT:                return "SetIndexCount";
        case R_DRAW_INDEX_OFFSET:          return "DrawIndexOffset";
        case R_JUMP:                       return "Jump";
        case R_SET_PRED:                   return "SetPredication";
        case R_SET_BASE_INDIRECT_ARGS:     return "SetBaseIndirectArgs";
        case R_STALL_COMMAND_BUFFER_PARSER: return "StallCommandBufferParser";
        case R_DRAW_INDEX_INDIRECT:        return "DrawIndexIndirect";
        case R_DISPATCH_INDIRECT:          return "DispatchIndirect";
        // A raw allocate_dw with no sub-op (CbNop's 1-dword form, the type-2 filler). Without its own
        // case it would print as "(op-carried)" and be indistinguishable from a real op-carried packet.
        case kDcbDiagRaw:                  return "raw-allocate";
        // Everything else is named by its PM4 opcode, not a custom sub-op; both reports print `op=0x..`
        // alongside, which is what identifies it (0x76 SET_SH_REG, 0x46 EVENT_WRITE, 0x2f NUM_INSTANCES…).
        default:                           return "(op-carried)";
    }
}
// A ZERO FROM A GATED PROBE MUST PROVE IT WAS ARMED. #1756's corpus census concluded that Syberia
// issues no buffer-full callbacks at all, and that negative went into a permanent `## Ruled out`
// row — but an unarmed run produces exactly the same empty log as a clean one, so the conclusion
// rested on the operator having set the variable correctly. It now rests on the log: when either
// probe is armed the first packet emits a banner, and every 2^20 packets a tally reports how many
// were observed and how many buffer-full events fired. A run whose log contains
// `[dcbdiag] armed` and `full=0` measured zero; a run containing neither measured nothing, and the
// difference is visible without re-running anything.
std::atomic<uint64_t> g_dcb_diag_packets{0}, g_dcb_diag_fulls{0};
// Totals at exit as well, for a clean shutdown. Do not rely on it: a `timeout`-bounded diagnostic
// run dies on SIGTERM and never runs static destructors, which is why the periodic tally above is
// the primary evidence and its interval is small.
struct DcbDiagSummary {
    ~DcbDiagSummary() {
        const uint64_t n = g_dcb_diag_packets.load(std::memory_order_relaxed);
        if (!n) return;
        fprintf(stderr, "[dcbdiag] TOTAL seen=%llu packets, full=%llu buffer-full events\n",
                (unsigned long long)n,
                (unsigned long long)g_dcb_diag_fulls.load(std::memory_order_relaxed));
    }
};
DcbDiagSummary g_dcb_diag_summary;
void dcb_diag_tick(bool was_full) {
    if (was_full) { g_dcb_diag_fulls.fetch_add(1, std::memory_order_relaxed); return; }
    const uint64_t n = g_dcb_diag_packets.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1)
        fprintf(stderr, "[dcbdiag] armed: DCBFULL=%d DCBWIN=%u — this line proves the probe is live; "
                        "a run without it measured nothing\n",
                dcb_diag_full() ? 1 : 0, dcb_diag_win());
    // Every 2^14 packets, not 2^20: these runs are `timeout`-bounded, so SIGTERM skips the exit
    // summary below, and a low-volume title (Syberia emits well under 2^20 in 45 s) would otherwise
    // produce no tally at all — leaving "measured zero" and "never armed" looking identical, which
    // is the whole failure this reporting exists to prevent.
    if ((n & 0x3fffu) == 0)
        fprintf(stderr, "[dcbdiag] seen=%llu packets, full=%llu buffer-full events\n",
                (unsigned long long)n, (unsigned long long)g_dcb_diag_fulls.load(std::memory_order_relaxed));
}
void dcb_report_full(const AgcDcb* dcb, uint32_t need) {
    if (!dcb_diag_full() || !dcb) return;
    dcb_diag_tick(true);
    const uint32_t window = dcb->bottom && dcb->cursor_down
                          ? (uint32_t)(dcb->cursor_down - dcb->bottom) : 0;
    // Bound the report. A title whose command buffer simply grows produces one of these per chunk
    // for its whole run; the interesting population is the first few hundred, and the per-window-size
    // census the reader wants survives truncation because tight windows repeat.
    static std::atomic<uint32_t> n{0};
    const uint32_t i = n.fetch_add(1, std::memory_order_relaxed);
    if (i >= 512 && (i & 8191u) != 0) return;
    fprintf(stderr, "[dcbfull] #%u %s op=0x%x need=%u avail=%u reserved=%u window=%u dw at +%u cb=%p\n",
            i, dcb_subop_name(g_dcb_diag_r), g_dcb_diag_op, need, dcb->available_dw(), dcb->reserved_dw, window,
            dcb->bottom && dcb->cursor_up ? (unsigned)(dcb->cursor_up - dcb->bottom) : 0,
            (void*)(uintptr_t)dcb->callback);
}
void dcb_report_window(const AgcDcb* dcb, uint32_t n, uint32_t op, uint32_t r) {
    // begin_packet is the hottest path in the HLE (millions of packets per minute), so do nothing at
    // all — not even the two thread-local stores that name the builder for the buffer-full report —
    // unless one of the probes is armed. One relaxed load of a function-local static guard is the
    // entire cost of an unarmed run.
    static const bool armed = dcb_diag_full() || dcb_diag_win() != 0;
    if (!armed) return;
    dcb_diag_tick(false);                   // banner + periodic tally, so a zero is self-evidencing
    g_dcb_diag_r = r; g_dcb_diag_op = op;   // remembered for the buffer-full report just below
    const uint32_t limit = dcb_diag_win();
    if (!limit || !dcb || !dcb->bottom || !dcb->cursor_down) return;
    const uint32_t window = (uint32_t)(dcb->cursor_down - dcb->bottom);
    if (!window || window > limit) return;
    static std::atomic<uint32_t> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) >= 4096) return;
    fprintf(stderr, "[dcbwin] window=%u dw : %s n=%u op=0x%x at +%u (avail %u)\n",
            window, dcb_subop_name(r), n, op,
            (unsigned)(dcb->cursor_up - dcb->bottom), dcb->available_dw());
}

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
    dcb_report_window(dcb, n, op, r);          // #1756 probe; no-op unless PROSPER_DCBWIN
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
// Debug markers are regular DCB packets. The old +kSrjIVxKFE handler mistook PushMarker for a
// context constructor and zeroed live DCB state on every call (#641). Preserve the NUL-terminated
// marker text in the packet, as Kyty's Gen4 implementation does; the command processor may ignore
// the semantic marker, but the stream remains correctly framed and tools can recover its label.
HLE(agc_dcb_push_marker) {  // (buf, const char* marker)
    const char* marker = (const char*)(uintptr_t)a1;
    if (!marker) return 0;
    constexpr size_t kMaxMarkerBytes = 0x4000u * sizeof(uint32_t);
    size_t bytes = strnlen(marker, kMaxMarkerBytes);
    if (bytes == kMaxMarkerBytes) return 0;
    bytes++;  // include NUL
    uint32_t dwords = 1u + (uint32_t)((bytes + sizeof(uint32_t) - 1u) / sizeof(uint32_t));
    uint32_t* cmd; if (!begin_packet(a0, dwords, IT_NOP, R_PUSH_MARKER, &cmd)) return 0;
    memset(cmd + 1, 0, (dwords - 1u) * sizeof(uint32_t));
    memcpy(cmd + 1, marker, bytes);
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_pop_marker) {  // (buf)
    uint32_t* cmd; if (!begin_packet(a0, 2, IT_NOP, R_POP_MARKER, &cmd)) return 0;
    cmd[1] = 0;
    return (uint64_t)(uintptr_t)cmd;
}
// A TYPE-2 filler is the one valid single-dword PM4 packet. Gen5 callers use this when only one
// alignment dword is available; it cannot go through begin_packet because TYPE-3 lengths start at 2.
HLE(agc_dcb_type2_nop) {  // (buf)
    auto* dcb = reinterpret_cast<AgcDcb*>(static_cast<uintptr_t>(a0));
    if (!dcb) return 0;
    g_dcb_diag_r = kDcbDiagRaw; g_dcb_diag_op = 0;       // #1756: not routed through begin_packet
    uint32_t* cmd = dcb->allocate_dw(1);
    if (!cmd) return 0;
    *cmd = 0x80000000u;
    return reinterpret_cast<uint64_t>(cmd);
}

// Initialise the 32-entry pixel-interpolant descriptor table used by the Gen5 driver. Dragon Quest
// VII calls this for every graphics pipeline and later submits the low dwords as virtual offsets;
// returning success without touching `out` left those offsets uninitialised. Descriptor-specific
// translation is not implemented yet, so keep the bounded identity fallback explicit.
// CONFIDENCE: HIGH on output/count and the default encoding (live caller and command stream), LOW
// on the unused descriptor inputs; they remain deliberately uninterpreted.
static void agc_write_interpolant_defaults(uint64_t* out) {
    constexpr uint64_t kDefaultEntry = 0x10000000ull;
    for (uint32_t i = 0; i < 32; ++i) {
        const uint32_t low = static_cast<uint32_t>(kDefaultEntry) + i;
        const uint32_t high = static_cast<uint32_t>(kDefaultEntry >> 37u) * 32u + i;
        out[i] = (static_cast<uint64_t>(high) << 32u) | low;
    }
}

HLE(agc_build_interpolant_mapping) {  // (uint64_t out[32], opaque descriptor, opaque source)
    if (!a0 || !gpu::guest_writable(a0, 32u * sizeof(uint64_t))) return kAgcErrInvalidArg;
    auto* out = reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(a0));
    agc_write_interpolant_defaults(out);
    return 0;
}
HLE(agc_dcb_set_num_instances) {  // (buf, num_instances)
    uint32_t* cmd; if (!begin_packet(a0, kDwSetNumInstances, IT_NUM_INSTANCES, 0, &cmd)) return 0;
    cmd[1] = (uint32_t)a1;
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_wait_safe_for_rendering) {  // (buf, video_out_handle, display_buffer_index)
    uint32_t* cmd; if (!begin_packet(a0, 7, IT_NOP, R_WAIT_FLIP_DONE, &cmd)) return 0;
    cmd[1] = (uint32_t)a1; cmd[2] = (uint32_t)a2; cmd[3] = cmd[4] = cmd[5] = cmd[6] = 0;
    return (uint64_t)(uintptr_t)cmd;
}
static uint64_t set_register_direct(uint64_t buf, uint64_t packed_reg, uint32_t opcode) {
    uint32_t* cmd; if (!begin_packet(buf, kDwSetRegisterDirect, opcode, 0, &cmd)) return 0;
    cmd[1] = (uint32_t)(packed_reg & 0xffffffffu);
    cmd[2] = (uint32_t)(packed_reg >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}
// ShaderRegister is passed by value as {uint32_t offset, uint32_t value}, packed into a1 by the
// SysV ABI. These are real hardware SET_*_REG packets; keeping the register class in the opcode lets
// the command processor update the correct persistent register file before the next draw (#395 F5).
HLE(agc_dcb_set_cx_register_direct) { return set_register_direct(a0, a1, IT_SET_CONTEXT_REG); }
HLE(agc_dcb_set_sh_register_direct) { return set_register_direct(a0, a1, IT_SET_SH_REG); }
HLE(agc_dcb_set_uc_register_direct) { return set_register_direct(a0, a1, IT_SET_UCONFIG_REG); }
static uint64_t set_regs_indirect(uint64_t buf, uint64_t regs, uint64_t num_regs, uint32_t r) {
    uint32_t* cmd; if (!begin_packet(buf, kDwSetRegsIndirect, IT_NOP, r, &cmd)) return 0;
    cmd[1] = (uint32_t)num_regs; cmd[2] = (uint32_t)(regs & 0xffffffffu); cmd[3] = (uint32_t)(regs >> 32u);
    // PROSPER_REGBLOAT (#1264): record-time provenance for the register-array bloat investigation —
    // correlate each packet's ORIGINAL count with later PatchAddRegisters growth and the fold-time
    // garbage seen in command_processor's [regbloat] log.
    static const bool regbloat = getenv("PROSPER_REGBLOAT") != nullptr;
    if (regbloat) {
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 96)
            fprintf(stderr, "[regbloat-rec] r=0x%x cmd=%p regs=0x%llx num=%llu\n", r, (void*)cmd,
                    (unsigned long long)regs, (unsigned long long)num_regs);
    }
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_set_cx_regs_indirect) { return set_regs_indirect(a0, a1, a2, R_CX_REGS_INDIRECT); }
HLE(agc_dcb_set_sh_regs_indirect) { return set_regs_indirect(a0, a1, a2, R_SH_REGS_INDIRECT); }
HLE(agc_dcb_set_uc_regs_indirect) { return set_regs_indirect(a0, a1, a2, R_UC_REGS_INDIRECT); }
HLE(agc_dcb_set_index_size) {  // (buf, index_size, cache_policy)
    uint32_t* cmd; if (!begin_packet(a0, kDwSetIndexSize, IT_INDEX_TYPE, 0, &cmd)) return 0;
    cmd[1] = (uint32_t)a1; return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_draw_index_auto) {  // (buf, index_count, modifier)
    uint32_t* cmd; if (!begin_packet(a0, kDwDrawIndexAuto, IT_NOP, R_DRAW_INDEX_AUTO, &cmd)) return 0;
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
    uint32_t* cmd; if (!begin_packet(a0, kDwDrawIndex, IT_NOP, R_DRAW_INDEX, &cmd)) return 0;
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
    uint32_t* cmd; if (!begin_packet(a0, kDwSetIndexBuffer, IT_NOP, R_INDEX_BASE, &cmd)) return 0;
    cmd[1] = (uint32_t)(a1 & 0xffffffffu); cmd[2] = (uint32_t)(a1 >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}
static std::atomic<uint64_t> g_ic_log{0};
HLE(agc_dcb_set_index_count) {  // (dcb, indexCount)
    if (getenv("PROSPER_GFXLOG") && g_ic_log.fetch_add(1) < 24)
        fprintf(stderr, "[agc] SetIndexCount dcb=0x%llx count=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1);
    uint32_t* cmd; if (!begin_packet(a0, kDwSetIndexCount, IT_NOP, R_INDEX_COUNT, &cmd)) return 0;
    cmd[1] = (uint32_t)a1;
    return (uint64_t)(uintptr_t)cmd;
}
static std::atomic<uint64_t> g_do_log{0};
HLE(agc_dcb_draw_index_offset) {  // (dcb, startIndex[, indexCount])
    if (getenv("PROSPER_GFXLOG") && g_do_log.fetch_add(1) < 24)
        fprintf(stderr, "[agc] DrawIndexOffset dcb=0x%llx off=0x%llx a2=0x%llx a3=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1,
                (unsigned long long)a2, (unsigned long long)a3);
    uint32_t* cmd; if (!begin_packet(a0, kDwDrawIndexOffset, IT_NOP, R_DRAW_INDEX_OFFSET, &cmd)) return 0;
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
    uint32_t* cmd; if (!begin_packet(a0, kDwJump, IT_NOP, R_JUMP, &cmd)) return 0;
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
        g_dcb_diag_r = kDcbDiagRaw; g_dcb_diag_op = 0;   // #1756: not routed through begin_packet
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
// corruption root cause). ABI RE-pinned from four eboot callsites (disassembly, this branch):
//   sceAgcDcbDmaData(dcb, srcImmediateOrOffset, dstSel?, srcSel?, dstAddressOrOffset,
//                    <sel/policy>, sourceKind, sourceAddress, stack9=numBytes, ...,
//                    stack12=isBlocking)
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
//  - Tactics Ogre eboot+0x5a0ac0 emits the address-backed copy form used by its movie upload. It
//    passes sourceKind=2 and the decoded row address in stack arg8 (a7), while a1 remains zero. The
//    old builder encoded a1 unconditionally, turning every movie row copy into an immediate-zero
//    fill. The two call shapes are self-distinguishing: the established immediate/placeholder forms
//    pass sourceKind=0 and sourceAddress=0; the address form passes sourceKind=2 and a 64-bit source.
//    Select solely from that ABI discriminator: an invalid address must reach the executor and fail
//    visibly rather than silently changing into an immediate fill.
// numBytes is stack arg9. The import ABI bridge forwards args 7-9 as normal fixed parameters on
// Windows and on Linux's guest-FS path; no compiler-frame decoding is involved (#672).
// Custom R_DMA_DATA payload: [1..2]=dst lo/hi, [3..4]=srcOrImm lo/hi, [5]=numBytes, [6]=sels,
// [7..8]=the destination qword when this exact packet was built (#312 generation identity).
// CONFIDENCE: HIGH on a4=dst/a1=srcOrImm (malloc-destination callsite + patcher names + protocol);
// MED on the selector args (recorded raw; the executor distinguishes the captured 32-bit immediate
// domain from mapped 64-bit address sources and keeps GDS/unmapped forms fail-closed).
static uint64_t label_build_pre(uint64_t dst, uint64_t num_bytes) {
    uint64_t pre = 0;
    if (num_bytes <= 8 && dst >= 0x10000 && !(dst & 3) && gpu::guest_readable(dst, sizeof pre))
        memcpy(&pre, (const void*)(uintptr_t)dst, sizeof pre);
    return pre;
}

HLE9(agc_dcb_dma_data) {  // (..., srcImmediate, dstSel?, srcSel?, dst, policy, sourceKind, sourceAddress, bytes)
    uint64_t num_bytes = a8 <= 0x10000000ull ? a8 : 0;
    const bool address_source = a6 == 2;
    const uint64_t src_or_imm = address_source ? a7 : a1;
    static std::atomic<uint64_t> g_dma_n{0};
    uint32_t* cmd; if (!begin_packet(a0, kDwDmaData, IT_NOP, R_DMA_DATA, &cmd)) return 0;
    if (getenv("PROSPER_PREDLOG") || getenv("PROSPER_GFXLOG")) {
        uint64_t k = g_dma_n.fetch_add(1);
        if (k < 48 || (k % 4096) == 0)
            fprintf(stderr, "[pred] DcbDmaData #%llu cmd=%p dst=0x%llx srcOrImm=0x%llx bytes=0x%llx sels=0x%llx/0x%llx source-kind=0x%llx source-address=0x%llx selected=%s\n",
                    (unsigned long long)k, (void*)cmd, (unsigned long long)a4,
                    (unsigned long long)src_or_imm, (unsigned long long)num_bytes,
                    (unsigned long long)a2, (unsigned long long)a3,
                    (unsigned long long)a6, (unsigned long long)a7,
                    address_source ? "address" : "immediate/offset");
    }
    cmd[1] = (uint32_t)(a4 & 0xffffffffu); cmd[2] = (uint32_t)(a4 >> 32u);
    cmd[3] = (uint32_t)(src_or_imm & 0xffffffffu); cmd[4] = (uint32_t)(src_or_imm >> 32u);
    cmd[5] = (uint32_t)num_bytes;
    cmd[6] = (uint32_t)((a2 & 0xffu) | ((a3 & 0xffu) << 8));
    if (num_bytes <= 8) {
        prosper_label_hist_dma_built(a4, a0, (uint32_t)a1, 1);                  // #312 label-init form
    }
    return (uint64_t)(uintptr_t)cmd;
}
// sceAgcAcbDmaData (-RnpfpxIhec) — the async-compute-queue sibling. Same operation, shifted ABI
// (RE'd from the Acb consumed-marker emitter at eboot+0x220d31e, which pairs it with the same
// sceAgcCbReleaseMem(label<-1): (acb, srcSel?=3, dstSel?=3, dst=label, 2, 3, stack7=srcOrImm=0,
// stack8=numBytes=4). srcOrImm/numBytes are explicit stack args 7/8 after the import bridge.
// CONFIDENCE: MED (single callsite; the executor gate limits the blast radius).
HLE9(agc_acb_dma_data) {  // (acb, srcSel?, dstSel?, dst, ?, ?, stack7=srcOrImm, stack8=numBytes)
    uint64_t src_imm = a6, num_bytes = a7;
    if (num_bytes > 0x10000000ull) num_bytes = 0;
    uint32_t* cmd; if (!begin_packet(a0, kDwDmaData, IT_NOP, R_DMA_DATA, &cmd)) return 0;
    cmd[1] = (uint32_t)(a3 & 0xffffffffu); cmd[2] = (uint32_t)(a3 >> 32u);
    cmd[3] = (uint32_t)(src_imm & 0xffffffffu); cmd[4] = (uint32_t)(src_imm >> 32u);
    cmd[5] = (uint32_t)num_bytes;
    cmd[6] = (uint32_t)((a1 & 0xffu) | ((a2 & 0xffu) << 8));
    if (num_bytes <= 8) {
        prosper_label_hist_dma_built(a3, a0, (uint32_t)src_imm, 2);              // #312
    }
    return (uint64_t)(uintptr_t)cmd;
}
// sceAgcDmaDataPatchSetDstAddressOrOffset (IxYiarKlXxM) / ...SetSrcAddressOrOffsetOrImmediate
// (cdDRpqcFGbU): patch a built DMA_DATA packet's dst / src field (same patch-placeholder pattern
// as the #213 sync-packet address patchers; header-verified before writing).
//
// #1124: PPSA02664's GfxDevice "workload" flow builds a DMA_DATA packet into a temp DCB via
// sceAgcDcbDmaData, memcpy's it into the real command buffer, then DIRECTLY writes the copy's first
// qword (using the real-AGC field offset) before calling these patchers. prosper's custom 9-dword
// packet places the R_DMA_DATA header in that first dword, so the direct write clobbers the header
// to 0x00000000 — which is a PM4 type-0 word that TRUNCATES the fold walk, dropping the DMA entirely
// (verified live: the packet body survives in prosper's layout — dst at [1..2], bytes at [5] — only
// the header is lost; restoring it takes the captured gameplay submit from "0 DMA copies" to "1").
// The observed instances are the workload's 8-byte fence-label inits (#312 consumed-marker lifecycle,
// previously silently dropped here); the mechanism is general to any DMA the game post-patches.
// Since the guest is explicitly declaring "this is my DMA packet" by calling the patcher, and a real
// DMA packet can never have a zero header, restore prosper's header before patching so the copy
// executes. Guarded to the exact observed clobber (header == 0 AND a plausible DMA body: mapped
// dst, sane byte count), so a genuine non-DMA target still refuses loudly. CONFIDENCE: MED-HIGH.
static bool dma_patch_recover_header(uint32_t* cmd, const char* who) {
    if (!gpu::guest_readable((uint64_t)(uintptr_t)cmd, kDwDmaData * sizeof(uint32_t))) return false;
    if (cmd[0] != 0) return false;                                   // not the zero-clobber shape
    const uint64_t dst   = (uint64_t)cmd[1] | ((uint64_t)cmd[2] << 32);
    const uint32_t bytes = cmd[5];
    if (dst < 0x10000 || bytes == 0 || bytes > 0x10000000u) return false;  // implausible DMA body
    cmd[0] = PM4(kDwDmaData, IT_NOP, R_DMA_DATA);   // MUST match the builder: a stale literal here
                                                    // stamps a longer header over a shorter
                                                    // allocation and desyncs the submit (#1756)
    static std::atomic<int> n{0};
    if (n.fetch_add(1) < 8)
        fprintf(stderr, "[agc] %s: restored clobbered DMA header (dst=0x%llx bytes=%u) — #1124\n",
                who, (unsigned long long)dst, bytes);
    return true;
}
// Header check for the DMA patchers: accept a valid R_DMA_DATA header, or recover the #1124
// zero-clobber; only a genuinely unexpected header reaches patch_check's loud refusal.
static bool dma_patch_ready(uint32_t* cmd, const char* who) {
    const uint32_t r = (cmd[0] >> 2) & (R_NUM - 1u), op = (cmd[0] >> 8) & 0xffu;
    if (op == IT_NOP && r == R_DMA_DATA) return true;
    if (dma_patch_recover_header(cmd, who)) return true;
    return patch_check(cmd, R_DMA_DATA, who);   // refuse loudly for any other unexpected header
}
HLE(agc_patch_dma_data_dst) {  // (cmd, address)
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    if (!dma_patch_ready(cmd, "DmaDataPatchSetDst")) return 0;
    cmd[1] = (uint32_t)(a1 & 0xffffffffu); cmd[2] = (uint32_t)(a1 >> 32u);
    // Defensive tolerance for the historical 9-dword shape, not a live path: with #1756 every packet
    // reachable here at runtime is a 7-dword one this file just built.
    uint32_t len = ((cmd[0] >> 16) & 0x3fffu) + 2;
    if (len >= 9) {
        uint64_t build_pre = label_build_pre(a1, cmd[5]);
        cmd[7] = (uint32_t)build_pre; cmd[8] = (uint32_t)(build_pre >> 32);
    }
    return 0;
}
HLE(agc_patch_dma_data_src) {  // (cmd, addressOrImmediate)
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    if (!dma_patch_ready(cmd, "DmaDataPatchSetSrc")) return 0;
    cmd[3] = (uint32_t)(a1 & 0xffffffffu); cmd[4] = (uint32_t)(a1 >> 32u);
    return 0;
}
HLE(agc_dcb_event_write) {  // (buf, event_type, address)
    // Widen the packet to carry the address (a2) like ReleaseMem (#132): an address-carrying
    // EVENT_WRITE (the timestamp/label variant) writes a completion value there, and discarding it
    // left a guest waiting on that label blocked forever. cmd[1]=event_type, cmd[2..3]=address lo/hi.
    // An address-less event (a2=0, the pipeline-sync variants: partial-flush, cache-inval) stays a
    // no-op in the CommandProcessor (event_addr==0), exactly as before.
    uint32_t* cmd; if (!begin_packet(a0, kDwEventWrite, IT_EVENT_WRITE, 0, &cmd)) return 0;
    cmd[1] = (uint32_t)(a1 & 0xffu);
    cmd[2] = (uint32_t)(a2 & 0xffffffffu); cmd[3] = (uint32_t)(a2 >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_acquire_mem) {  // (buf, engine, cb_db_op, gcr_cntl, ...) — 8 dw
    uint32_t* cmd; if (!begin_packet(a0, kDwAcquireMem, IT_NOP, R_ACQUIRE_MEM, &cmd)) return 0;
    cmd[1] = (uint32_t)a2; cmd[2] = cmd[3] = cmd[4] = cmd[5] = cmd[6] = 0; cmd[7] = (uint32_t)a3;
    return (uint64_t)(uintptr_t)cmd;
}
HLE9(agc_cb_release_mem) {  // sceAgcCbReleaseMem(buf, action, gcr_cntl, dst, cache_policy, address, data_sel, data, …)
    // ABI pinned to Kyty GraphicsCbReleaseMem (Graphics.cpp:1763): a5=label address, then the two stack
    // args are data_sel (7th) and the 64-bit fence value (8th), forwarded as a6/a7 by the import bridge.
    uint64_t data_sel = a6, data = a7;
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
        // Diagnostic-only stack scan for guest callsite attribution. ABI arguments never come from it.
        volatile uint64_t* stack_scan = (uint64_t*)__builtin_frame_address(0);
        uint64_t ra = 0, ra2 = 0, ra3 = 0;
        for (int i = 1; i < 96; i++) {
            uint64_t v = stack_scan[i];
            if (prosper::guest_va_in_module(v)) {   // #1659: was a stale literal base
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
                fprintf(stderr, "[agc] fence1-builder ra=%s+0x%llx ra2=%s+0x%llx ra3=%s+0x%llx addr=0x%llx action=0x%llx\n",
                        prosper::guest_module_name(ra),
                        (unsigned long long)prosper::guest_module_offset(ra),
                        ra2 ? prosper::guest_module_name(ra2) : "",
                        (unsigned long long)(ra2 ? prosper::guest_module_offset(ra2) : 0),
                        ra3 ? prosper::guest_module_name(ra3) : "",
                        (unsigned long long)(ra3 ? prosper::guest_module_offset(ra3) : 0),
                        (unsigned long long)a5, (unsigned long long)a1);
            }
        }
    }
    // EOP fence: stash the label address (a5), the data_sel, the full 64-bit value, and the event action
    // into the packet so the CommandProcessor performs the completion write at SUBMIT time (correct
    // end-of-pipe timing — our GPU folds synchronously, so submit == pipe drain). CONFIDENCE: HIGH — the
    // arg positions are fixed by the AGC ABI; a5-as-label was independently confirmed (WaitRegMem polls it).
    //
    // The packet is EXACTLY 8 DWORDS, the size of the RDNA2 PM4 RELEASE_MEM it stands for (header +
    // EVENT_CNTL, DATA_CNTL, ADDRESS_LO/HI, DATA_LO/HI, INT_CTXID). This is a hard constraint, not a
    // preference: prosper's builders write into the GUEST's command buffer, and the guest reserves
    // that buffer from the REAL AGC packet sizes it was compiled against. Emitting one dword more
    // than hardware overruns the reservation. Asterix & Obelix - Babylon Mission (PPSA30490, #1748)
    // is the case that proved it: its submit epilogue is a 16-dword window holding exactly
    // sceAgcDcbAcquireMem (8) + this end-of-pipe action (8). At 9 dwords the append did not fit, the
    // DCB "buffer full" callback fired, and the guest's chunk allocator handed over a fresh ~600 KiB
    // command-buffer chunk for EVERY submit — ~210 MiB/s of direct memory, never reclaimed (the
    // chunk's own completion fence ends up in the next chunk, which is never submitted), emptying
    // the direct-memory pool in ~125 s.
    //
    // That leaves ONE spare payload dword for the #312 build-time snapshot of the destination qword,
    // so only its HIGH half is retained — which is the only half `stale_release_generation` reads
    // (it masks with 0xffffffff00000000). Keep this at 8; if a future field is needed, drop the
    // snapshot rather than growing the packet.
    uint32_t* cmd; if (!begin_packet(a0, kDwReleaseMem, IT_NOP, R_RELEASE_MEM, &cmd)) return 0;
    cmd[1] = (uint32_t)(a5 & 0xffffffffu); cmd[2] = (uint32_t)(a5 >> 32u);
    cmd[3] = (uint32_t)data_sel;
    cmd[4] = (uint32_t)(data & 0xffffffffu); cmd[5] = (uint32_t)(data >> 32u);
    cmd[6] = (uint32_t)a1;
    uint64_t build_pre = label_build_pre(a5, data_sel == 1 ? 4 : 8);
    cmd[7] = (uint32_t)(build_pre >> 32);
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
    // The packet header declares 5+num dwords and cmd[4]=num tells the CommandProcessor how many inline
    // data dwords follow; the CP writes all `num` of them to the destination. If a caller passes a null
    // data pointer with num>0, the tail cmd[5..] would otherwise carry STALE ring-buffer memory that the
    // CP then writes into guest memory (the exact stale-tail class fixed for draw_index_auto; the sibling
    // set_sh_register_range_direct builder already zero-fills its tail). Zero-fill instead of leaking it.
    if (src) for (uint32_t i = 0; i < num; i++) cmd[5 + i] = src[i];
    else     memset(cmd + 5, 0, (size_t)num * 4u);
    if (num <= 4) prosper_fence_journal_record((uint64_t)(uintptr_t)cmd, a3);   // #312 discriminator
    return (uint64_t)(uintptr_t)cmd;
}
HLE9(agc_dcb_wait_reg_mem) {  // sceAgcDcbWaitRegMem(buf, size, compare_func, op, cache_policy, address, reference, mask, poll_cycles)
    // ABI pinned to Kyty GraphicsDcbWaitRegMem (Graphics.cpp:2096): a5 = label address; the 7th/8th
    // stack args reference, mask, and poll_cycles are forwarded as a6/a7/a8 by the import bridge.
    // The payload previously ZEROED every
    // wait parameter at build time (address/ref/mask/function destroyed — the wait could never be
    // honored downstream); now it mirrors Kyty's layout exactly:
    // [1..2]=addr lo/hi, [3..4]=mask lo/hi, [5..6]=reference lo/hi, [7]=compare_function, [8]=interval.
    uint64_t reference = a6, mask = a7;
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc] WaitRegMem size=%llu func=%llu op=%llu addr=0x%llx ref=0x%llx mask=0x%llx\n",
            (unsigned long long)a1,(unsigned long long)a2,(unsigned long long)a3,
            (unsigned long long)a5,(unsigned long long)reference,(unsigned long long)mask);
    uint32_t* cmd; if (!begin_packet(a0, 9, IT_NOP, R_WAIT_MEM_64, &cmd)) return 0;
    cmd[1] = (uint32_t)(a5 & 0xffffffffu);        cmd[2] = (uint32_t)(a5 >> 32u);
    cmd[3] = (uint32_t)(mask & 0xffffffffu);      cmd[4] = (uint32_t)(mask >> 32u);
    cmd[5] = (uint32_t)(reference & 0xffffffffu); cmd[6] = (uint32_t)(reference >> 32u);
    cmd[7] = (uint32_t)a2;                        // compare_function
    cmd[8] = (uint32_t)a8;                        // poll interval hint
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

// Registry of all created shaders — the AGC->Vulkan pipeline consumes this to find shader code by
// PGM base.
// The registry mutex covers CreateShader's push_back (guest loader threads) against the submit
// thread's lookup iteration: an unsynchronized push_back REALLOCATION frees the vector's backing
// store mid-iteration — a use-after-free read, not the "benign stale read" it was assumed to be.
std::mutex&                      agc_shaders_mx() { static std::mutex m; return m; }
std::vector<const AgcShader*>&   agc_shaders()   { static std::vector<const AgcShader*> v; return v; }
struct AgcFusedShaderPair {
    uint64_t front_code = 0;
    const AgcShader* back = nullptr;
};
std::vector<AgcFusedShaderPair>& agc_fused_shader_pairs() {
    static std::vector<AgcFusedShaderPair> v;
    return v;
}
std::unordered_map<uint64_t, uint64_t>& agc_shader_continuations() {
    static std::unordered_map<uint64_t, uint64_t> continuations;
    return continuations;
}

// Relocate one self-relative pointer field in place. SDK shader blobs use small forward offsets from
// the pointer field; linked guest pointers live above 4 GiB. Inspect the field representation rather
// than caching the header address: Sonic Origins frees a shader header and later constructs another
// at the same allocation, so an address-only "already relocated" set misclassifies the new blob and
// dereferences its raw 0x70/0x78 register offsets as host pointers.
template <class T> inline bool agc_fix_ptr(T*& m) {
    const uintptr_t raw = reinterpret_cast<uintptr_t>(m);
    if (!raw || raw >= 0x100000000ull) return false;
    m = reinterpret_cast<T*>(raw + reinterpret_cast<uintptr_t>(&m));
    return true;
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
    // PROSPER_SHADER_HEADER_NEWEST=1 (#305 A/B): resolve to the MOST RECENT registration bound to
    // this code address instead of the first. Diagnostic switch for the recycled-allocation
    // hypothesis; see prosper_agc_shader_headers_for_code below.
    static const bool newest = std::getenv("PROSPER_SHADER_HEADER_NEWEST") != nullptr;
    std::lock_guard<std::mutex> lk(agc_shaders_mx());
    if (newest) {
        const auto& all = agc_shaders();
        for (size_t i = all.size(); i-- > 0;)
            if (all[i] && (uint64_t)(uintptr_t)all[i]->code == code_addr) return all[i];
        return nullptr;
    }
    for (const AgcShader* h : agc_shaders())
        if (h && (uint64_t)(uintptr_t)h->code == code_addr) return h;
    return nullptr;
}

// #305 instrument: enumerate the registry by index, so a diagnostic can ask "which registered
// shader's declared user-data layout matches the block this draw actually programmed?".
extern "C" const void* prosper_agc_shader_at(size_t index) {
    std::lock_guard<std::mutex> lk(agc_shaders_mx());
    const auto& all = agc_shaders();
    return index < all.size() ? all[index] : nullptr;
}

// #305 instrument: how many registered shaders bind the SAME code address, and which. The registry
// is append-only (CreateShader never removes), and the lookup above returns the FIRST match, so a
// guest that re-registers a different shader over a recycled code allocation keeps resolving to the
// oldest header — a correct register block read through a stale user-data layout. Fills `out` in
// registration order (oldest first) and returns the TOTAL match count, which may exceed `max`.
extern "C" size_t prosper_agc_shader_headers_for_code(uint64_t code_addr, const void** out,
                                                      size_t max) {
    if (!code_addr) return 0;
    std::lock_guard<std::mutex> lk(agc_shaders_mx());
    size_t n = 0;
    for (const AgcShader* h : agc_shaders()) {
        if (!h || (uint64_t)(uintptr_t)h->code != code_addr) continue;
        if (out && n < max) out[n] = h;
        ++n;
    }
    return n;
}

// A fused GS/HS programs the front-half address into the stage register while retaining the actual
// shader body in the back-half header. Preserve that association for the Vulkan executor, which sees
// only the bound front address after PM4 decoding.
extern "C" const void* prosper_agc_fused_back_header_for_front(uint64_t front_code_addr) {
    if (!front_code_addr) return nullptr;
    std::lock_guard<std::mutex> lk(agc_shaders_mx());
    const AgcShader* match = nullptr;
    for (const auto& pair : agc_fused_shader_pairs()) {
        if (pair.front_code != front_code_addr) continue;
        if (match && match->code != pair.back->code) return nullptr; // ambiguous reuse: fail closed
        match = pair.back;
    }
    return match;
}

// Fusion installs the front-half program in the hardware stage's PGM registers while its terminal
// s_setpc_b64 transfers to the separately allocated back half. Preserve that exact SDK relationship
// at construction time: the PM4 state intentionally exposes only the installed front address, and
// trying to infer the continuation later from another PGM register loses valid NGG vertex programs.
extern "C" uint64_t prosper_agc_shader_continuation_for_code(uint64_t code_addr) {
    if (!code_addr) return 0;
    std::lock_guard<std::mutex> lk(agc_shaders_mx());
    const auto found = agc_shader_continuations().find(code_addr);
    return found == agc_shader_continuations().end() ? 0 : found->second;
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

    agc_fix_ptr(h->cx_registers);
    agc_fix_ptr(h->sh_registers);
    agc_fix_ptr(h->user_data);
    agc_fix_ptr(h->specials);
    agc_fix_ptr(h->input_semantics);
    agc_fix_ptr(h->output_semantics);
    if (h->user_data) {
        agc_fix_ptr(h->user_data->direct_resource_offset);
        for (auto& sharp : h->user_data->sharp_resource_offset)
            agc_fix_ptr(sharp);
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

struct AgcSizeAlign {
    uint64_t size;
    uint64_t align;
};

constexpr uint8_t kAgcShaderGs = 2;
constexpr uint8_t kAgcShaderHs = 3;
constexpr uint8_t kAgcShaderGsFront = 4;
constexpr uint8_t kAgcShaderHsFront = 5;
constexpr uint8_t kAgcShaderGsBack = 6;
constexpr uint8_t kAgcShaderHsBack = 7;

bool agc_shader_halves_match(const AgcShader* front, const AgcShader* back) {
    return front && back &&
           ((front->type == kAgcShaderGsFront && back->type == kAgcShaderGsBack) ||
            (front->type == kAgcShaderHsFront && back->type == kAgcShaderHsBack));
}

AgcShaderRegister* agc_find_shader_register(AgcShaderRegister* regs, uint32_t count,
                                             uint32_t offset, uint32_t occurrence = 0) {
    if (!regs) return nullptr;
    for (uint32_t i = 0; i < count; ++i) {
        if (regs[i].offset != offset) continue;
        if (occurrence == 0) return &regs[i];
        --occurrence;
    }
    return nullptr;
}

const AgcShaderRegister* agc_find_shader_register(const AgcShaderRegister* regs, uint32_t count,
                                                   uint32_t offset, uint32_t occurrence = 0) {
    if (!regs) return nullptr;
    for (uint32_t i = 0; i < count; ++i) {
        if (regs[i].offset != offset) continue;
        if (occurrence == 0) return &regs[i];
        --occurrence;
    }
    return nullptr;
}

void agc_patch_shader_address(AgcShaderRegister* regs, uint32_t count,
                              uint32_t lo_offset, uint64_t address) {
    AgcShaderRegister* lo = agc_find_shader_register(regs, count, lo_offset);
    if (!lo || lo + 1 >= regs + count || (lo + 1)->offset != lo_offset + 1u) return;
    lo->value = (uint32_t)((address >> 8u) & 0xffffffffu);
    (lo + 1)->value = ((lo + 1)->value & 0xffffff00u) | (uint32_t)((address >> 40u) & 0xffu);
}

}  // namespace

// sceAgcGetFusedShaderSize / sceAgcFuseShaderHalves. UE builds geometry and hull shaders from
// separately compiled front/back halves. The size query reserves scratch space for a private copy
// of the back half's SH-register array; fusion then patches that copy with the front program address
// (and GS checksums), producing an ordinary type-2 GS or type-3 HS shader. Returning success without
// these writes left UE's stack-local fused Shader uninitialized, so a later TaskGraph worker treated
// stack bytes as num_cx_registers and called memcpy from its null cx_registers pointer (#1213).
HLE(agc_get_fused_shader_size) {  // (SizeAlign* dst, const Shader* front, const Shader* back)
    auto* dst = (AgcSizeAlign*)(uintptr_t)a0;
    auto* front = (const AgcShader*)(uintptr_t)a1;
    auto* back = (const AgcShader*)(uintptr_t)a2;
    if (!dst || !front || !back) return kAgcErrInvalidArg;
    if (!agc_shader_halves_match(front, back)) return kAgcErrInvalidShaderHalves;
    dst->size = (uint64_t)back->num_sh_registers * sizeof(AgcShaderRegister);
    dst->align = alignof(uint32_t);
    return 0;
}

HLE(agc_fuse_shader_halves) {  // (Shader* dst, const Shader* front, const Shader* back, void* scratch)
    auto* dst = (AgcShader*)(uintptr_t)a0;
    auto* front = (const AgcShader*)(uintptr_t)a1;
    auto* back = (const AgcShader*)(uintptr_t)a2;
    void* scratch = (void*)(uintptr_t)a3;
    if (!dst || !front || !back) return kAgcErrInvalidArg;
    if (!agc_shader_halves_match(front, back)) return kAgcErrInvalidShaderHalves;

    *dst = *back;
    dst->type = front->type == kAgcShaderGsFront ? kAgcShaderGs : kAgcShaderHs;

    auto* front_specials = (const AgcShaderSpecialRegs*)front->specials;
    auto* back_specials = (const AgcShaderSpecialRegs*)back->specials;
    if (front_specials && back_specials) {
        const uint32_t mismatch_bit = front->type == kAgcShaderGsFront ? (1u << 22u) : (1u << 21u);
        if (((front_specials->vgt_shader_stages_en.value ^
              back_specials->vgt_shader_stages_en.value) & mismatch_bit) != 0)
            return kAgcErrInvalidShaderHalves;
    }

    if (scratch && back->sh_registers && back->num_sh_registers) {
        auto* copied = (AgcShaderRegister*)scratch;
        memcpy(copied, back->sh_registers,
               (size_t)back->num_sh_registers * sizeof(AgcShaderRegister));
        dst->sh_registers = copied;
    }

    using namespace prosper::agc::Pm4;
    if (front->type == kAgcShaderGsFront) {
        for (uint32_t occurrence = 0; occurrence < 2; ++occurrence) {
            AgcShaderRegister* out = agc_find_shader_register(
                dst->sh_registers, dst->num_sh_registers, SPI_SHADER_PGM_CHKSUM_GS, occurrence);
            const AgcShaderRegister* in = agc_find_shader_register(
                front->sh_registers, front->num_sh_registers, SPI_SHADER_PGM_CHKSUM_GS, occurrence);
            if (out && in) out->value = in->value;
        }
        agc_patch_shader_address(dst->sh_registers, dst->num_sh_registers,
                                 SPI_SHADER_PGM_LO_ES, (uint64_t)(uintptr_t)front->code);
    } else {
        agc_patch_shader_address(dst->sh_registers, dst->num_sh_registers,
                                 SPI_SHADER_PGM_LO_LS, (uint64_t)(uintptr_t)front->code);
    }
    dst->user_data = nullptr;
    {
        std::lock_guard<std::mutex> lk(agc_shaders_mx());
        const uint64_t front_code = (uint64_t)(uintptr_t)front->code;
        const uint64_t back_code = (uint64_t)(uintptr_t)back->code;
        auto& pairs = agc_fused_shader_pairs();
        auto found = std::find_if(pairs.begin(), pairs.end(), [front_code, back](const auto& pair) {
            return pair.front_code == front_code && pair.back->code == back->code;
        });
        if (found == pairs.end()) pairs.push_back({front_code, back});
        else found->back = back;
        if (front_code && back_code) agc_shader_continuations()[front_code] = back_code;
    }
    return 0;
}

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

// sceAgcCreateInterpolantMapping — build the 32 SPI_PS_INPUT_CNTL_* registers wiring
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
    uint32_t n = ps ? ps->num_input_semantics : (uint32_t)gs->num_output_semantics;
    if (getenv("PROSPER_PIPETRACE"))
        fprintf(stderr, "  [interp] gs.num_out=%u ps=%p ps.num_in=%u -> mapping %u interpolants\n",
                (uint32_t)gs->num_output_semantics, (void*)ps,
                ps ? ps->num_input_semantics : 0u, n);
    pipetrace_shader_user_data("CreateInterpolant.gs", gs);
    pipetrace_shader_user_data("CreateInterpolant.ps", ps);
    const auto mapping = prosper::gpu::derive_agc_pixel_input_controls(
        reinterpret_cast<const prosper::gpu::AgcShaderHeader*>(gs),
        reinterpret_cast<const prosper::gpu::AgcShaderHeader*>(ps));
    for (uint32_t i = 0; i < 32; i++) {
        regs[i].offset = SPI_PS_INPUT_CNTL_0 + i;
        regs[i].value = (mapping.valid_mask & (1u << i)) ? mapping.controls[i] : 0u;
    }
    return 0;
}

// The SDK-revision export dbOlWdppb4o has two live title-visible argument shapes. Cobra passes
// complete producer/pixel shader blobs (types 2 and 1) and consumes 32 SPI_PS_INPUT_CNTL register
// pairs. Dragon Quest VII passes opaque/default inputs and consumes the 32-entry virtual-offset
// table modeled by agc_build_interpolant_mapping. Discriminate on the exact shader types validated
// by Cobra's eboot wrapper; all other shapes retain DQ's bounded default contract.
// CONFIDENCE: HIGH (both live callers plus independent focused regressions).
HLE(agc_create_interpolant_mapping_sdk_alias) {
    if (a1 && a2 && gpu::guest_readable(a1, sizeof(AgcShader)) &&
        gpu::guest_readable(a2, sizeof(AgcShader))) {
        const auto* producer = reinterpret_cast<const AgcShader*>(static_cast<uintptr_t>(a1));
        const auto* pixel = reinterpret_cast<const AgcShader*>(static_cast<uintptr_t>(a2));
        if (producer->type == 2 && pixel->type == 1)
            return agc_create_interpolant_mapping(a0, a1, a2, a3, a4, a5);
    }
    return agc_build_interpolant_mapping(a0, a1, a2, a3, a4, a5);
}

// --- sceAgcDriverSubmitDcb (AgcDriver NID UglJIZjGssM) — the GPU submission point. -------------
//
// The game hands us a Packet {dcb_addr, dw_num} (Kyty Gen5Driver::Packet, Graphics.cpp:2168). We
// replay the PM4 stream through the CommandProcessor into a persistent GpuState — register files +
// draw list — which the AGC->Vulkan translation (render_state/vk_translate) consumes. No rendering
// happens here yet; this is the semantic execution of the submit, observable via PROSPER_GFXLOG and
// prosper_agc_submit_stats(). The Dcb address is a guest pointer, valid 1:1 in-process.
namespace {
gpu::GpuState& agc_graphics_state() { static gpu::GpuState st; return st; }
gpu::GpuState& agc_compute_state(uint64_t queue) {
    // Every hardware async-compute queue owns its register context. Queue ids are driver handles,
    // so retain state lazily for exactly the ids a title submits instead of assuming one global ACB.
    static std::unordered_map<uint64_t, gpu::GpuState> states;
    return states.try_emplace(queue).first->second;
}
gpu::GpuState& agc_gpu_state() { return agc_graphics_state(); }
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
extern "C" int prosper_vo_flip_rate();

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

// Focused state probe used by the submit regression. Keeping the graphics and async-compute
// register files distinct is architectural state, not a title workaround: the two hardware queues
// may use the same numeric SH-register offsets concurrently without changing each other's bindings.
extern "C" bool prosper_agc_submit_sh_reg(uint64_t queue, uint32_t offset, uint32_t* value) {
    if (!value) return false;
    std::lock_guard<std::mutex> lk(g_agc_state_mu);
    const auto& sh = queue ? agc_compute_state(queue).sh : agc_graphics_state().sh;
    const auto it = sh.find(offset);
    if (it == sh.end()) return false;
    *value = it->second;
    return true;
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
    // Account at most one requested refresh interval per flip while the host performs synchronous
    // GPU work. A real console returns from submission without charging shader compilation,
    // resource conversion, execution, or readback latency to the guest's next frame delta.
    static uint64_t accounted_flips = 0;
    const uint64_t flips = prosper_vo_flip_count();
    const uint64_t flip_delta = flips >= accounted_flips ? flips - accounted_flips : flips;
    accounted_flips = flips;
    const int flip_rate = prosper_vo_flip_rate();
    const uint64_t refresh_divisor = flip_rate >= 0 && flip_rate <= 2
        ? (uint64_t)flip_rate + 1 : 1;
    const uint64_t period_ns = (1000000000ull * refresh_divisor) / 60ull;
    const uint64_t clock_budget_ns = flip_delta > UINT64_MAX / period_ns
        ? UINT64_MAX : flip_delta * period_ns;
    // A bounded sparse phase accelerates long intros, then cadence=1 rebuilds any temporal render
    // targets before a visual checkpoint. Screenshot exposes these as --render-every/--render-every-for-seconds.
    static const unsigned render_every = [] {
        const char* e = getenv("PROSPER_RENDER_EVERY");
        long v = e ? atol(e) : 1;
        return v > 0 ? (unsigned)v : 1u;
    }();
    static const uint64_t render_every_for_ms = [] {
        const char* e = getenv("PROSPER_RENDER_EVERY_FOR_MS");
        long long v = e ? atoll(e) : 0;
        return v > 0 ? (uint64_t)v : 0;
    }();
    static const auto render_sampling_t0 = std::chrono::steady_clock::now();
    const uint64_t sampling_elapsed_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - render_sampling_t0).count();
    const unsigned cadence = render_every_for_ms > 0 && sampling_elapsed_ms >= render_every_for_ms
        ? 1u : render_every;
    // A retained copy may read pixels produced by an earlier draw in this submit. Sampling cadence
    // cannot discard that producer and then let DMA read the previous cached target version.
    const bool ordered_dma_requires_render = !st.dma_copies.empty() && !st.draws.empty();
    bool render = false;
    if (gpu::have_submit_renderer() && !st.draws.empty()) {
        const bool cadence_render = (draw_submits++ % cadence) == 0;
        render = ordered_dma_requires_render || cadence_render;
    }
    if (!render) {
        if (!st.dispatches.empty() || !st.dma_copies.empty()) {
            HostGpuClockScope host_gpu_clock(clock_budget_ns);
            if (!gpu::execute_nonrender_submit_work(st, submit_no) && getenv("PROSPER_COMPUTELOG"))
                fprintf(stderr, "[compute] submit #%llu produced no executable work\n",
                        (unsigned long long)submit_no);
        }
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
    // Draw spans and dispatches are synchronous. Apply bulk resource uploads needed by the renderer,
    // but leave fence/label completions on their post-submit FIFO: exposing a ReleaseMem fence before
    // this call returns lets another guest thread recycle its label while the submitter is still
    // updating the corresponding allocation lists (#312).
    HostGpuClockScope host_gpu_clock(clock_budget_ns);
    gpu::flush_deferred_streams();
    prosper_gpu_drain_renderer_writes();
    // PROSPER_PRESENT_EVERY=N suppresses publication only. Unlike PROSPER_RENDER_EVERY, every
    // graphics/compute operation still executes and persistent targets advance on every submit.
    // Publish the first eligible frame, then one in every N so a consumer gets an image promptly.
    static const unsigned present_every = [] {
        const char* e = getenv("PROSPER_PRESENT_EVERY");
        long v = e ? atol(e) : 1;
        return v > 0 ? (unsigned)v : 1u;
    }();
    static uint64_t publication_candidates = 0;
    const bool publish = (publication_candidates++ % present_every) == 0;
    const bool presented = gpu::execute_ordered_and_present(st, w, h, submit_no, publish);
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
extern "C" uint64_t prosper_guest_tsc_ns();                    // shared guest clock (hle_kernel_time)
// #1226 POOLSHIFT window probe (PROSPER_POOLSHIFT_WINDOW=1): after each submit, scan the learned
// TLS pool bins + global recycler slots for the byte-shifted-head poison signature and log on STATE
// CHANGE. A change gives only a WEAK bound — the poison was at a scanned head between these two
// submits — NOT the free()-to-fault lifetime: the scan samples head residence, and a poison buried
// in a chain interior or surfacing in a sub-submit sliver is missed (see mb3_poolshift_window_scan).
// The decisive instrument is PROSPER_MB3WATCH (traps the store). Serialized by the callers'
// g_agc_state_mu, so the plain statics are single-threaded.
static void poolshift_window_probe(uint64_t submit_no) {
    static const bool on = getenv("PROSPER_POOLSHIFT_WINDOW") != nullptr;
    if (!on) return;
    static char buf[2048];
    static int last_found = -1;
    int found = gpu::mb3_poolshift_window_scan(buf, sizeof buf);
    if (found != last_found) {
        fprintf(stderr, "[agc] POOLSHIFT-WINDOW submit=%llu found=%d (was %d) t=%llums\n%s",
                (unsigned long long)submit_no, found, last_found,
                (unsigned long long)prosper_guest_tsc_ns() / 1000000ull, found ? buf : "");
        last_found = found;
    }
}
// #305/#1662: a fold that decodes fewer dwords than the guest declared has silently dropped the tail
// of that submit — and if a shader bind was in the dropped tail, the NEXT submit inherits a register
// file hardware never had, which is exactly the "user-data block larger than the bound pipeline's
// window" shape #305 is chasing. decode_pm4 stops early at a non-type-3 dword or a packet whose
// declared length overruns the buffer; neither was visible before. Reported unconditionally (not
// behind a switch) and rate-limited, because a truncated command stream is never expected.
static void report_short_fold(const char* who, uint64_t submit_no, const uint32_t* addr,
                              uint32_t declared, size_t consumed) {
    if (!declared || consumed >= declared) return;   // 0 = unknown length: self-termination is normal
    static std::atomic<int> logged{0};
    const int n = logged.fetch_add(1);
    if (n >= 32) return;
    const uint32_t stop = addr && consumed < declared ? addr[consumed] : 0;
    fprintf(stderr,
            "[agc] SHORT FOLD %s #%llu: guest declared %u dwords, decode stopped after %zu "
            "(%.1f%%) at stream dword 0x%08x — the remaining %zu dwords, INCLUDING ANY BIND, were "
            "never applied%s\n",
            who, (unsigned long long)submit_no, declared, consumed,
            100.0 * (double)consumed / (double)declared, stop,
            (size_t)declared - consumed, n == 31 ? " [further reports suppressed]" : "");
}
// #305 submit ORDER vs FOLD order (PROSPER_SUBMITORDER=1).
//
// prosper folds every submit synchronously inside the caller's HLE handler, serialised by
// g_agc_state_mu. That makes the FOLD order a property of lock acquisition, not of the guest's
// program order — and #305 turns on exactly that distinction: a q3 (SubmitDcbFinal) fold appears
// immediately after a q1 (SubmitDcb) bind, and the question is whether hardware would have run them
// in that order at all. Nikoderiko additionally issues its two submits from two DIFFERENT guest
// functions (eboot+0xfb0c10 and eboot+0xfb04b0), not one loop over a buffer array as the DOLL
// derivation assumed, so the ordering prosper inherits has never been checked against this title.
//
// The call stamp is therefore taken at HLE ENTRY, BEFORE the mutex. Taking it after would record the
// order the lock happened to grant and answer the question with its own assumption.
struct SubmitCallStamp { uint64_t call_seq; uint64_t ns; uint64_t thread; };
static std::atomic<uint64_t> g_submit_call_seq{0};
static std::atomic<uint64_t> g_submit_thread_next{1};
static thread_local uint64_t t_submit_thread_token = 0;
static bool submitorder_on() {
    static const bool on = getenv("PROSPER_SUBMITORDER") != nullptr;
    return on;
}
// Identity only, never a handle: pthread_t is opaque and must not be compared as an integer.
static SubmitCallStamp stamp_submit_call() {
    if (!t_submit_thread_token)
        t_submit_thread_token = g_submit_thread_next.fetch_add(1, std::memory_order_relaxed);
    SubmitCallStamp st{};
    st.thread = t_submit_thread_token;
    st.ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
    st.call_seq = g_submit_call_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
    return st;
}
// Emitted AFTER the fold, carrying the pre-lock stamp. `call` and `fold` diverging is the signal: a
// submit the guest issued earlier was folded later, so prosper's register state at a draw is not the
// state the guest's own submission order would produce.
//
// Counted always, logged SPARSELY. A line per submit would be ~70 MB over a 440 s route, and the I/O
// would perturb the very ordering this is measuring — the instrument would become part of the result.
// So: the opening submits (to see the steady-state shape), every out-of-order event (the exception
// that matters), and a periodic census.
static void report_submit_order(const char* who, const SubmitCallStamp& st, uint64_t fold_seq,
                                const uint32_t* addr, uint32_t dw_num, size_t draws) {
    if (!submitorder_on()) return;
    struct Census { std::atomic<uint64_t> n{0}, threads{0}; };
    static std::atomic<uint64_t> last_call{0}, out_of_order{0}, total{0};
    static std::atomic<uint64_t> per_thread[8]{};        // count by thread token (1..7 observed)
    static std::atomic<uint64_t> final_per_thread[8]{};  // ...for SubmitDcbFinal specifically
    const bool is_final = strcmp(who, "SubmitDcbFinal") == 0;
    const uint64_t n = total.fetch_add(1, std::memory_order_relaxed) + 1;
    if (st.thread < 8) {
        per_thread[st.thread].fetch_add(1, std::memory_order_relaxed);
        if (is_final) final_per_thread[st.thread].fetch_add(1, std::memory_order_relaxed);
    }
    const uint64_t prev = last_call.exchange(st.call_seq, std::memory_order_acq_rel);
    const bool inverted = st.call_seq < prev;
    if (inverted) {
        const uint64_t k = out_of_order.fetch_add(1, std::memory_order_relaxed);
        if (k < 32)
            fprintf(stderr, "[submitorder] *** FOLDED OUT OF CALL ORDER *** %s call=%llu (previous "
                            "fold's call=%llu) fold=%llu thread=%llu t=%lluus dw=%u draws=%zu\n",
                    who, (unsigned long long)st.call_seq, (unsigned long long)prev,
                    (unsigned long long)fold_seq, (unsigned long long)st.thread,
                    (unsigned long long)(st.ns / 1000ull), dw_num, draws);
    }
    if (n <= 20 || (n % 20000) == 0) {
        char tb[128]; int o = 0;
        for (int i = 1; i < 8 && o < 100; ++i) {
            const uint64_t c = per_thread[i].load(std::memory_order_relaxed);
            if (c) o += snprintf(tb + o, sizeof(tb) - o, " t%d=%llu/%llu", i,
                                 (unsigned long long)c,
                                 (unsigned long long)final_per_thread[i].load(std::memory_order_relaxed));
        }
        fprintf(stderr, "[submitorder] %-14s call=%llu fold=%llu thread=%llu t=%lluus addr=%p dw=%u "
                        "draws=%zu | total=%llu inverted=%llu by-thread(all/final):%s\n",
                who, (unsigned long long)st.call_seq, (unsigned long long)fold_seq,
                (unsigned long long)st.thread, (unsigned long long)(st.ns / 1000ull),
                (const void*)addr, dw_num, draws, (unsigned long long)n,
                (unsigned long long)out_of_order.load(std::memory_order_relaxed), tb);
    }
}

// #1669 A/B ONLY — NOT A FIX, AND MUST NOT BE MADE DEFAULT-ON ON THE STRENGTH OF A NUMBER.
//
// #305 established that SubmitDcb (q1) and SubmitDcbFinal (q3) fold into the SAME graphics register
// file, and that q3 is issued from threads that never issue q1. Whether sharing is correct depends on
// an unestablished fact: is SubmitDcbFinal the same hardware ring as SubmitDcb — in which case it MUST
// inherit q1's registers — or a distinct queue with its own register context, in which case merging
// them is the defect? #1226 found exactly that symptom one queue over and fixed it by splitting the
// register file, which raises the prior but does not settle it.
//
// This switch gives q3 its own GpuState so the question can be MEASURED: if the title's
// recompile/vertex-reject counts collapse AND its world renders, that is evidence for the distinct-
// queue answer; if they are unchanged, or draws fail in new ways, that is evidence for inheritance and
// rules the direction out. Either result advances #1669 for one route.
//
// MEASURED (2026-08-01, Nikoderiko PPSA23760, two back-to-back 440 s routes, same binary):
//
//     metric                    control    split
//     recompile-reject              524      529
//     vertex-recompile-reject       138      135
//     mubuf-unresolved              126      108
//
// No collapse — every delta is far inside the run-to-run variance of this route (an earlier control
// on the same build measured 310/64/61). The frames are the decisive half and they are IDENTICAL:
// the title screen renders and the 3D world is black in BOTH arms. The split is not a no-op — it
// shifts the failure mix ([lazy-commit] 1118 -> 3476, [mimg-unresolved] appearing) — it simply does
// not fix anything. So the direction is ruled out as a fix, and the switch is retained OFF as a
// documented negative result, exactly like PROSPER_UD_TAIL_ALIGN.
//
// LIMIT OF THIS EXPERIMENT, stated because the pre-registered rule ("unchanged => inheritance") was
// slightly too strong: a q3 buffer opens with user-data writes and a draw and has NO bind of its own,
// so a freshly split state has no program bound under EITHER hypothesis. The result therefore rules
// out the FIX cleanly, but is weaker evidence about ring semantics than the rule assumed. #1669's
// section 1 still needs the guest-use / contract / disassembly evidence.
//
// It stays default-off until the SEMANTICS are established, however good the number looks. A numeric
// improvement is not evidence of a correct model — that is the failure mode recorded as instrument
// trap 15, and separating a queue that must inherit would create a new defect shaped exactly like the
// one it appears to fix.
static bool split_final_state() {
    static const bool on = getenv("PROSPER_AGC_SPLIT_FINAL_STATE") != nullptr;
    return on;
}
// Distinct from the Acb per-queue map: keyed by nothing, because there is exactly one Dcb-final
// stream. Retained process-lifetime like the graphics state it shadows.
static gpu::GpuState& agc_final_state() { static gpu::GpuState st; return st; }
static uint64_t submit_dcb_stream(const uint32_t* addr, uint32_t dw_num, const char* who,
                                  uint64_t queue_id = 0) {
    if (!addr) return 0;
    constexpr uint32_t kUnknownCap = 0x80000;   // 512 KiB of dwords — well past any real Dcb
    uint32_t walk = dw_num ? dw_num : kUnknownCap;
    const SubmitCallStamp call_stamp = stamp_submit_call();   // BEFORE the lock: see the note above
    std::lock_guard<std::mutex> lk(g_agc_state_mu);   // serialize with the other submit entry (#278)
    const bool async_compute = strcmp(who, "SubmitAcb") == 0;
    const bool dcb_final = strcmp(who, "SubmitDcbFinal") == 0;
    gpu::GpuState& state = async_compute            ? agc_compute_state(queue_id)
                         : (dcb_final && split_final_state()) ? agc_final_state()   // #1669 A/B
                                                             : agc_graphics_state();
    if (dcb_final && split_final_state()) {
        static std::atomic<bool> announced{false};
        if (!announced.exchange(true))
            fprintf(stderr, "[agc] PROSPER_AGC_SPLIT_FINAL_STATE=1: SubmitDcbFinal is folding into its "
                            "OWN register file (#1669 MEASUREMENT, not a fix — this configuration is "
                            "not validated and must not be treated as correct behaviour)\n");
    }
    // #1226: stamp this fold's submit entry point so fence-protocol history can distinguish the
    // graphics Dcb stream from the async-compute Acb stream. The queues share ordered memory
    // effects, but not register files: folding Acb SH writes into graphics state overwrote live
    // vertex user-data bindings before the next Dcb draw (Plucky's first gameplay scene).
    gpu::prosper_gpu_set_fold_origin(async_compute                         ? 2
                                     : strcmp(who, "SubmitDcbFinal") == 0 ? 3 : 1);
    // #312: flush any earlier stream paused on a WAIT_REG_MEM — this submit may be its producer.
    gpu::flush_deferred_streams();
    state.draws.clear();
    state.dispatches.clear();
    state.dma_copies.clear();
    state.dma_data_records.clear();
    state.dma_data_record_count = 0;
    state.dma_data_records_truncated = false;
    state.dma_execution_rejected = false;
    state.ordered_memory_effects.clear();
    state.capture_dma_data_records = gpu::begin_gpu_timeline_submit(g_submit_count + 1);
    size_t consumed = 0;
    size_t applied = gpu::run_command_buffer(addr, walk, state, &consumed);
    g_submit_count++;
    report_short_fold(who, g_submit_count, addr, dw_num, consumed);
    report_submit_order(who, call_stamp, g_submit_count, addr, dw_num, state.draws.size());
    gpu::diagnose_compute_dispatches(state, g_submit_count);
    static unsigned draw_submits2 = 0;
    execute_submit_work(state, g_submit_count, draw_submits2);
    gpu::diagnose_resource_provenance(state, g_submit_count);
    // #312 EOP visibility contract: the pulse fires immediately only when no gated writes are
    // pending; otherwise it is OWED and delivered when the tail drains (the guest's completion
    // scan must never observe a half-retired frame — see command_processor.cpp). The flip and the
    // pipe-drain writes still flow, and the watchdog + timeout bound the delay, so the pacer
    // stays alive (the naive always-pulse variant let the scan free live labels; the naive
    // never-pulse variant starved the pacer — both measured).
    gpu::flush_deferred_streams();
    gpu::submit_completion_pulse(state.dma_execution_rejected);
    // Watchdog keys off PENDING streams, not just this fold: streams can outlive their fold (and
    // the Jump-recursion flag reset once hid a deferring fold entirely — the wedge class).
    if (gpu::deferred_pending()) start_defer_watchdog();
    poolshift_window_probe(g_submit_count);
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc] %s #%llu: %u dwords (walk=%u) -> %zu packets applied (draws: %zu, dispatches: %llu)\n",
                who, (unsigned long long)g_submit_count, dw_num, walk, applied,
                state.draws.size(), (unsigned long long)state.dispatch_count);
    progress_heartbeat(state.draws.size(), g_submit_count,
                       (uint64_t)state.dispatch_count);
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
// stack arg8, forwarded as a7 by the import ABI bridge. The dword COUNT is stack arg9: the callsite loads the buffer-array entry
// {addr @+0x00, dw_count32 @+0x10} (`mov 0x10(%rax,%r12,1),%r10d`) and the adapter pushes it as the
// import's 9th arg (32-bit, matching Packet.dw_num on the primary UglJIZjGssM path).
//
// WHY the exact count matters (#241 and the boot crash-zoo): the Dcb is carved from a live ring whose
// STALE tail is still valid type-3 PM4 from previous frames. An unknown-length fold does not stop at
// the logical end — it re-executes the stale packets, including stale WriteData/ReleaseMem fence writes
// whose destination heap blocks the guest has since freed and reused. That scribbles 8-byte GPU values
// into live allocator state (the deterministic 0x20015f00 free-list node of issue #241, the libc.prx
// logger stack-smash) — intermittent because it depends on what got reallocated under the stale fences.
// CONFIDENCE: HIGH on the arg9=count ABI (callsite + adapter disassembly agree, and the count matches
// the primary path's Packet.dw_num field offset).
HLE9(agc_driver_submit_dcb_variant) {
    // The generated return hook is attached to this import invocation even when validation rejects
    // it. Begin before every early return so nested/re-entrant tagged calls remain exactly paired.
    prosper_gpu_submit_scope_begin();
    auto is_pm4 = [](uint64_t p) -> bool {
        if (p < 0x10000 || (p & 3)) return false;
        uint32_t h = *(const volatile uint32_t*)(uintptr_t)p;
        return (h & 0xC0000000u) == 0xC0000000u;
    };
    auto first_dword = [](uint64_t p) -> uint32_t {
        return (p >= 0x10000 && !(p & 3)) ? *(const volatile uint32_t*)(uintptr_t)p : 0;
    };
    uint64_t cand = a7;                                     // arg8 = the Dcb stream address (adapter ABI)
    if (!is_pm4(cand)) {                                    // fallback: scan the register args
        // #1662: this substitutes a GUESSED stream address for the one the guest passed, on the
        // strength of a ONE-DWORD type-3 test. It used to do so silently — only the found-nothing
        // case below logged — so a submit folding a scavenged pointer was indistinguishable in the
        // log from one folding the guest's. That is unproved provenance that is not fail-visible.
        //
        // The guess is unusually dangerous here because the adapter thunks that reach this import do
        // NOT set the register arguments (verified by disassembly on two UE4 titles): a0..a5 hold
        // whatever the caller left there, which at Nikoderiko's call site includes a STACK address.
        // An arbitrary stack dword passes a bits-31:30 test about one time in four, and a fold does
        // not merely read — WRITE_DATA/RELEASE_MEM packets write into guest memory (#241).
        //
        // Reported unconditionally and rate-limited. The policy question (refuse rather than guess)
        // is deliberately separate from making it visible; see #1662.
        const uint64_t regs[6] = { a0, a1, a2, a3, a4, a5 };
        const uint64_t rejected = cand;
        cand = 0;
        int winner = -1;
        for (int i = 0; i < 6; ++i) if (is_pm4(regs[i])) { cand = regs[i]; winner = i; break; }
        static std::atomic<int> logged_sub{0};
        const int n = logged_sub.fetch_add(1);
        if (n < 16)
            fprintf(stderr,
                    "[agc] w1KFAHVqpaU SUBSTITUTED STREAM ADDRESS (#1662): arg8=0x%llx rejected "
                    "(first dword 0x%08x is not a type-3 header); %s a%d=0x%llx (first dword 0x%08x); "
                    "arg9=0x%llx. The fold below is from a GUESSED pointer, not the guest's%s\n",
                    (unsigned long long)rejected, first_dword(rejected),
                    winner < 0 ? "no register qualified —" : "using",
                    winner, (unsigned long long)cand, first_dword(cand),
                    (unsigned long long)a8, n == 15 ? " [further reports suppressed]" : "");
    }
    if (!cand) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 4)
            fprintf(stderr, "[agc] w1KFAHVqpaU: no PM4 stream found (arg8=0x%llx a0=0x%llx a1=0x%llx) — submit refused\n",
                    (unsigned long long)a7, (unsigned long long)a0, (unsigned long long)a1);
        return 0;
    }
    uint32_t dw_num = a8 && a8 <= 0x400000ull ? (uint32_t)a8 : 0;
    if (!dw_num) {
        static std::atomic<int> logged9{0};
        if (logged9.fetch_add(1) < 4)
            fprintf(stderr, "[agc] w1KFAHVqpaU: invalid arg9 count 0x%llx — self-terminating fold\n",
                    (unsigned long long)a8);
    }
    return submit_dcb_stream((const uint32_t*)(uintptr_t)cand, dw_num, "SubmitDcbFinal");
}

HLE(agc_driver_submit_dcb) {  // (const Packet* packet)
    // Paired with this NID's generated post-handler return hook, including invalid calls.
    prosper_gpu_submit_scope_begin();
    struct Packet { uint32_t* addr; uint32_t dw_num; uint8_t pad[4]; };
    const auto* p = (const Packet*)(uintptr_t)a0;
    if (!p || !p->addr || !p->dw_num) return kAgcErrInvalidArg;
    const SubmitCallStamp call_stamp = stamp_submit_call();   // BEFORE the lock: see the note above
    std::lock_guard<std::mutex> lk(g_agc_state_mu);   // serialize with submit_dcb_stream/w1KFAHVqpaU (#278)
    gpu::prosper_gpu_set_fold_origin(1);   // #1226: this direct entry is the graphics SubmitDcb path
    // Reset the per-submit draw list BEFORE folding this Dcb. The folded GpuState is process-lifetime and
    // its register files (cx/sh/uc) persist across submits (as on real hardware), but its `draws` vector
    // must NOT accumulate — otherwise it grows unbounded and every later frame re-renders stale geometry.
    // Clearing here (not after) keeps this submit's draws inspectable once the handler returns. (Ported
    // from PRs #31/#32.)
    // #312: flush any earlier stream paused on a WAIT_REG_MEM — this submit may be its producer.
    gpu::flush_deferred_streams();
    agc_gpu_state().draws.clear();
    agc_gpu_state().dispatches.clear();
    agc_gpu_state().dma_copies.clear();
    agc_gpu_state().dma_data_records.clear();
    agc_gpu_state().dma_data_record_count = 0;
    agc_gpu_state().dma_data_records_truncated = false;
    agc_gpu_state().dma_execution_rejected = false;
    agc_gpu_state().ordered_memory_effects.clear();
    agc_gpu_state().capture_dma_data_records =
        gpu::begin_gpu_timeline_submit(g_submit_count + 1);
    size_t consumed = 0;
    size_t applied = gpu::run_command_buffer(p->addr, p->dw_num, agc_gpu_state(), &consumed);
    g_submit_count++;
    report_short_fold("SubmitDcb", g_submit_count, p->addr, p->dw_num, consumed);
    report_submit_order("SubmitDcb", call_stamp, g_submit_count, p->addr, p->dw_num,
                        agc_gpu_state().draws.size());
    gpu::diagnose_compute_dispatches(agc_gpu_state(), g_submit_count);
    static unsigned draw_submits = 0;
    execute_submit_work(agc_gpu_state(), g_submit_count, draw_submits);
    gpu::diagnose_resource_provenance(agc_gpu_state(), g_submit_count);
    // The submit has "completed" (synchronous fold): fire any registered GPU EOP events. Inert unless the
    // game called sceGnmAddEqEvent (b0xyllnVY-I); the RELEASE_MEM label write already happened in apply().
    // #312 EOP visibility contract: pulse only when no gated writes are pending, else owed until
    // the tail drains (see submit_dcb_stream / command_processor.cpp).
    gpu::flush_deferred_streams();
    gpu::submit_completion_pulse(agc_gpu_state().dma_execution_rejected);
    if (gpu::deferred_pending()) start_defer_watchdog();
    poolshift_window_probe(g_submit_count);
    if (getenv("PROSPER_GFXLOG")) {
        fprintf(stderr, "[agc] SubmitDcb #%llu: %u dwords -> %zu packets applied (draws so far: %zu, dispatches total: %llu)\n",
                (unsigned long long)g_submit_count, p->dw_num, applied, agc_gpu_state().draws.size(),
                (unsigned long long)agc_gpu_state().dispatch_count);
        // Dump each packet's kind + first payload dwords so we can see whether the game embeds a
        // completion-label write (WriteData/ReleaseMem/EventWrite) or a GPU-side wait in the stream.
        std::vector<gpu::Pm4Command> ops; gpu::decode_pm4(p->addr, p->dw_num, ops);
        auto kind_name = [](gpu::Pm4Command::Kind kind) {
            using K = gpu::Pm4Command::Kind;
            switch (kind) {
                case K::DrawReset:       return "DrawReset";
                case K::WaitFlipDone:    return "WaitFlipDone";
                case K::PushMarker:      return "PushMarker";
                case K::PopMarker:       return "PopMarker";
                case K::SetRegDirect:    return "SetRegDirect";
                case K::SetRegsIndirect: return "SetRegsIndirect";
                case K::SetIndexType:    return "SetIndexType";
                case K::SetNumInstances: return "SetNumInstances";
                case K::DrawIndex:       return "DrawIndex";
                case K::DrawIndexAuto:   return "DrawIndexAuto";
                case K::EventWrite:      return "EventWrite";
                case K::AcquireMem:      return "AcquireMem";
                case K::WriteData:       return "WriteData";
                case K::WaitRegMem:      return "WaitRegMem";
                case K::Flip:            return "Flip";
                case K::ReleaseMem:      return "ReleaseMem";
                case K::DispatchDirect:  return "DispatchDirect";
                case K::SetIndexBase:    return "SetIndexBase";
                case K::SetIndexCount:   return "SetIndexCount";
                case K::DrawIndexOffset: return "DrawIndexOffset";
                case K::Jump:            return "Jump";
                case K::SetPredication:  return "SetPredication";
                case K::SetBaseIndirectArgs: return "SetBaseIndirectArgs";
                case K::StallCommandBufferParser: return "StallCommandBufferParser";
                case K::DrawIndexIndirect: return "DrawIndexIndirect";
                case K::DispatchIndirect: return "DispatchIndirect";
                case K::DmaData:         return "DmaData";
                case K::Unknown:         return "Unknown";
            }
            return "?";
        };
        for (auto& c : ops) {
            fprintf(stderr, "[agc]   pkt op=0x%02x r=0x%02x len=%u kind=%s pl0=0x%08x pl1=0x%08x pl2=0x%08x\n",
                    c.op, c.r, c.len, kind_name(c.kind),
                    c.len > 1 ? c.payload[0] : 0, c.len > 2 ? c.payload[1] : 0, c.len > 3 ? c.payload[2] : 0);
        }
    }
    progress_heartbeat(agc_gpu_state().draws.size(), g_submit_count,
                       (uint64_t)agc_gpu_state().dispatch_count);
    return 0;
}

// Async-compute submissions use the same AgcDriver::Packet layout as DCB submissions. The command
// processor already retains dispatches and orders their ReleaseMem/WaitRegMem side effects, so fold
// ACB streams through the same serialized queue model. Keeping a named wrapper makes the queue type
// visible in diagnostics and leaves room for independent hardware queue state later.
HLE(agc_driver_submit_acb) {  // sceAgcDriverSubmitAcb(queue, const AcbPacket*, ...)
    // Paired with this NID's generated post-handler return hook, including invalid calls.
    prosper_gpu_submit_scope_begin();
    auto reject = [&](const char* reason, uint64_t stream = 0, uint64_t count = 0,
                      uint32_t header = 0) -> uint64_t {
        static std::atomic<unsigned> logged{0};
        if (logged.fetch_add(1) < 16)
            fprintf(stderr,
                    "[agc] SubmitAcb reject=%s args=[0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx] "
                    "stream=0x%llx count=0x%llx header=0x%08x\n",
                    reason, (unsigned long long)a0, (unsigned long long)a1,
                    (unsigned long long)a2, (unsigned long long)a3,
                    (unsigned long long)a4, (unsigned long long)a5,
                    (unsigned long long)stream, (unsigned long long)count, header);
        return kAgcErrInvalidArg;
    };
    if (getenv("PROSPER_GFXLOG")) {
        static std::atomic<unsigned> logged{0};
        if (logged.fetch_add(1) < 16) {
            uint64_t words[8]{};
            bool readable = a1 >= 0x10000 && gpu::guest_readable(a1, sizeof words);
            if (readable) memcpy(words, (const void*)(uintptr_t)a1, sizeof words);
            fprintf(stderr, "[agc] SubmitAcb args=[0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx] a1mem=%s "
                    "[%llx %llx %llx %llx %llx %llx %llx %llx]\n",
                    (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                    (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5,
                    readable ? "yes" : "no", (unsigned long long)words[0],
                    (unsigned long long)words[1], (unsigned long long)words[2],
                    (unsigned long long)words[3], (unsigned long long)words[4],
                    (unsigned long long)words[5], (unsigned long long)words[6],
                    (unsigned long long)words[7]);
        }
    }
    // Live submissions share a 16-byte record: {stream address, uint32 dword_count, uint32 flags}.
    // Astro Bot left FLAGS zero, which made the last two fields look like one qword count; Plucky sets
    // FLAGS=1 and exposed that accidental interpretation as count 0x10000000088 instead of 0x88.
    // Read the fields at their architectural widths, then validate the stream and optional count mirror.
    if (!a1 || !gpu::guest_readable(a1, 16)) return reject("record-unreadable");
    uint64_t stream = 0;
    uint32_t count32 = 0, submit_flags = 0;
    memcpy(&stream, (const void*)(uintptr_t)a1, 8);
    memcpy(&count32, (const void*)(uintptr_t)(a1 + 8), 4);
    memcpy(&submit_flags, (const void*)(uintptr_t)(a1 + 12), 4);
    (void)submit_flags; // retained for the validated ABI layout; no observed flag changes fold semantics
    const uint64_t count64 = count32;
    if (!stream || count64 == 0 || count64 > 0x400000ull)
        return reject("record-fields", stream, count64);
    // The count mirror slot differs by adapter: Astro Bot uses a3; ArcRunner and Plucky use a2 while
    // a3 mirrors the queue id. Both non-zero register values disagreeing with the record is invalid.
    if (a2 && a3 && a2 != count64 && a3 != count64)
        return reject("count-mirror", stream, count64);
    // The decoder consumes every advertised dword.  Checking only the header allowed a stream at
    // the end of a readable page to fault as soon as decode crossed into the following guard page.
    const uint64_t stream_bytes = count64 * sizeof(uint32_t);  // bounded above, cannot overflow
    if (!gpu::guest_readable(stream, (size_t)stream_bytes))
        return reject("stream-unreadable", stream, count64);
    uint32_t header = 0; memcpy(&header, (const void*)(uintptr_t)stream, sizeof header);
    if ((header & 0xc0000000u) != 0xc0000000u && header != 0x80000000u)
        return reject("stream-header", stream, count64, header);
    return submit_dcb_stream((const uint32_t*)(uintptr_t)stream, count32, "SubmitAcb", a0);
}

// --- Indirect-register patch helpers: modify a packet returned by a Set*RegsIndirect call.
// cmd = a0 (that returned pointer). Old stub returned 0 for Set*RegsIndirect -> these wrote to null. -
HLE(agc_patch_set_address) {  // (cmd, regs): cmd[2..3] = regs vaddr
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    static const bool regbloat = getenv("PROSPER_REGBLOAT") != nullptr;
    if (regbloat) {
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 96)
            fprintf(stderr, "[regbloat-patch] set_address cmd=%p old=0x%x%08x new=0x%llx num=%u\n",
                    (void*)cmd, cmd[3], cmd[2], (unsigned long long)a1, cmd[1]);
    }
    cmd[2] = (uint32_t)(a1 & 0xffffffffu); cmd[3] = (uint32_t)(a1 >> 32u); return 0;
}
HLE(agc_patch_add_registers) {  // (cmd, num_regs): cmd[1] += num_regs
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    static const bool regbloat = getenv("PROSPER_REGBLOAT") != nullptr;
    if (regbloat) {
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 96)
            fprintf(stderr, "[regbloat-patch] add_registers cmd=%p old_num=%u add=%u\n",
                    (void*)cmd, cmd[1], (uint32_t)a1);
    }
    cmd[1] += (uint32_t)a1; return 0;
}

// sceAgcSet{Cx,Sh,Uc}RegIndirectPatchSetNumRegisters — SET, not accumulate, the register count of a
// packet returned by the matching Set*RegistersIndirect call.
//
// The record-then-patch idiom reserves the packet BEFORE the register array is known: the builder
// emits Set*RegistersIndirect(dcb, <placeholder>, <placeholder count>), fills the array as the
// draw's state is assembled, then patches in the final address and count. `PatchAddRegisters`
// accumulates while the array grows; `PatchSetNumRegisters` writes the final count outright and is
// the only member of the family that can LOWER one — so it is the only way an over-counted packet
// can be corrected, and no amount of AddRegisters can substitute for it.
//
// All three were unregistered, so they took the unimplemented-import path (dispatch.cpp
// prosper_on_unimpl: log once, return 0) and the packet kept its RECORD-TIME count. The failure that
// leaves is silent in both directions: a packet reserved with count 0 applies NOTHING (the
// SetRegsIndirect arm of GpuState::apply returns early on num_regs == 0), so a whole register write
// disappears from the decoded stream; a packet whose count the guest lowered keeps its high-water
// count and folds whatever follows the real registers in as further register writes.
//
// Every UE4 title dumped here imports these NIDs, but no title measured so far CALLS them (a
// 460 s Nikoderiko/PPSA23760 boot through gameplay logs no unimplemented hit for nCUgItdN2ms), so
// this is a gap fill against a real ABI, not a fix for any diagnosed defect. In particular it is NOT
// the cause of the stale user-data block in #305 — that was measured to a different mechanism — and
// the register-file bloat of #1264 was separately attributed to stale arena slots. Do not infer a
// causal link to either from this handler's presence.
//
// The packet layout is the one the Set*RegsIndirect builder emits (cmd[1] = count,
// cmd[2..3] = array vaddr), and the header sub-op is verified first, so a pointer that is not the
// expected packet is refused loudly rather than silently corrupting the stream.
// CONFIDENCE: HIGH on the count-slot semantics (the packet layout is prosper's own, the SET-vs-ADD
// distinction is the name, and the sibling patchers in this family already write these same slots).
// CONFIDENCE: MED that no live title needs it yet — absence of a call in the routes measured is not
// proof of absence across titles, which is exactly why it should be implemented rather than left to
// return 0.
static uint64_t patch_set_num_registers(uint64_t cmd_addr, uint64_t num, uint32_t want_r,
                                        const char* who) {
    auto* cmd = (uint32_t*)(uintptr_t)cmd_addr; if (!cmd) return 0;
    // These NIDs previously reached prosper_on_unimpl and touched nothing, so this is a NEW guest
    // dereference AND a new guest STORE on a path that used to be inert. The probe must therefore be
    // the WRITE predicate: guest_readable would pass a read-only mapping (on Linux/macOS it probes
    // by having the kernel read the byte, and on Windows it explicitly accepts PAGE_READONLY /
    // PAGE_WRITECOPY / PAGE_EXECUTE_READ), and the store below would then fault on a page the probe
    // just called fine — the module's own .text/.rodata are large read-only mappings in this same
    // address space. guest_writable covers the freed/unmapped case as well.
    //
    // COST CAVEAT, do not copy this call verbatim onto a hot path: guest_writable is UNCACHED, and
    // on Linux it fopen()s and parses /proc/self/maps on every call — unlike guest_readable, which
    // has a thread-local range cache. That is harmless here because no measured title calls these
    // NIDs at all, but the sibling patchers (#1650) run per draw on Blue Prince, and tightening
    // those needs a cached predicate rather than this one.
    if (!gpu::guest_writable(cmd_addr, 2 * sizeof(uint32_t))) {
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 8)
            fprintf(stderr, "[agc] %s: packet 0x%llx not writable — patch skipped\n", who,
                    (unsigned long long)cmd_addr);
        return 0;
    }
    // Report before the early return, and report the refusal too: an investigation into a wrong
    // register count most wants the case where the patch did not land.
    static const bool regbloat = getenv("PROSPER_REGBLOAT") != nullptr;
    const bool ok = patch_check(cmd, want_r, who);
    if (regbloat) {
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 96)
            fprintf(stderr, "[regbloat-patch] set_num_registers cmd=%p hdr=0x%08x old_num=%u new=%u%s\n",
                    (void*)cmd, cmd[0], cmd[1], (uint32_t)num, ok ? "" : " REFUSED");
    }
    if (!ok) return 0;
    cmd[1] = (uint32_t)num;
    return 0;
}
HLE(agc_patch_set_num_registers_cx) {
    return patch_set_num_registers(a0, a1, R_CX_REGS_INDIRECT, "SetCxRegIndirectPatchSetNumRegisters");
}
HLE(agc_patch_set_num_registers_sh) {
    return patch_set_num_registers(a0, a1, R_SH_REGS_INDIRECT, "SetShRegIndirectPatchSetNumRegisters");
}
HLE(agc_patch_set_num_registers_uc) {
    return patch_set_num_registers(a0, a1, R_UC_REGS_INDIRECT, "SetUcRegIndirectPatchSetNumRegisters");
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
// which packet kind). Those offsets are a RECORD OF prosper's builder sizes at the time of the RE,
// not constants the guest computes: the guest patches the pointers the builders returned, so they
// move with the builders. ReleaseMem has been 9 dwords (#750) and is now 8 (#1748) with no effect
// here — measured live on DOLL (PPSA17942) after #1748: 1333/1334/1333 calls to the three patchers
// and ZERO `patch refused`, with the patched header reading 0xc0061060 (IT_NOP, R_RELEASE_MEM,
// len 8). `patch_check` is what makes that safe; keep it. Signature is (cmd, address) — the
// fence-builder disassembly passes exactly
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
    uint32_t len = ((cmd[0] >> 16) & 0x3fffu) + 2;
    // Re-snapshot the destination for the #312 generation identity. Every packet this can be handed
    // at runtime is one agc_cb_release_mem just built, i.e. 8 dwords, keeping only the high half;
    // the >= 9 arm is defensive tolerance for the historical shape, not a live path (a replayed
    // capture never re-enters the HLE — gpu_replay executes recorded dwords directly).
    if (len >= 9) {
        uint64_t build_pre = label_build_pre(a1, cmd[3] == 1 ? 4 : 8);
        cmd[7] = (uint32_t)build_pre; cmd[8] = (uint32_t)(build_pre >> 32);
    } else if (len >= 8) {
        cmd[7] = (uint32_t)(label_build_pre(a1, cmd[3] == 1 ? 4 : 8) >> 32);
    }
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

// sceAgcDriverQueryResourceRegistrationUserMemoryRequirements. The API returns the opaque backing
// store required by ResourceRegistration::init for the requested resource and owner capacities.
// Prosper keeps the registry on the host and does not consume this guest block, so use a stable,
// conservative layout estimate rather than exposing host implementation details. Most importantly,
// always initialize the size_t output: the former success stub left Dead Cells' stack poison in it,
// intermittently turning this allocation into a multi-gigabyte texture-pool request (#660).
// CONFIDENCE: HIGH on ABI/output/error; MED on opaque record sizes (irrelevant until init consumes it).
HLE(agc_driver_query_resource_registration_user_memory_requirements) {
    if (!a0)
        return (uint64_t)(int64_t)(int32_t)0x80D19005u;

    constexpr uint64_t kHeaderBytes = 0x100;
    constexpr uint64_t kResourceBytes = 0x40;
    constexpr uint64_t kOwnerBytes = 0x20;
    constexpr uint64_t kAlignment = 0x40;
    const uint64_t required = kHeaderBytes + (uint32_t)a1 * kResourceBytes
                            + (uint32_t)a2 * kOwnerBytes;
    *(uint64_t*)(uintptr_t)a0 = (required + kAlignment - 1) & ~(kAlignment - 1);
    return 0;
}

// sceAgcDriverRegisterOwner / RegisterResource publish 32-bit opaque handles through their first
// argument. A Sonic Origins startup trace pins both ABIs: RegisterOwner(out, name, userData, ...),
// followed by RegisterResource(out, type, base, bytes, name, owner). The former generic-success
// tracers left the two adjacent stack outputs untouched. Prosper does not expose Razor resource
// inspection yet, so metadata remains guest-owned; stable non-zero IDs are the observable runtime
// contract needed by callers and by later query APIs. CONFIDENCE: HIGH on args/output width from the
// live PS5 title callsites; MED on invalid-pointer error until the PS5 error enum is published.
namespace {
std::atomic<uint32_t> g_agc_owner_handle{1};
std::atomic<uint32_t> g_agc_resource_handle{1};

uint32_t next_agc_registration_handle(std::atomic<uint32_t>& counter) {
    uint32_t handle = counter.fetch_add(1, std::memory_order_relaxed);
    if (handle == 0) handle = counter.fetch_add(1, std::memory_order_relaxed);
    return handle;
}
}
HLE(agc_driver_register_owner) {
    if (!a0) return (uint64_t)(int64_t)(int32_t)0x80D19005u;
    *(uint32_t*)(uintptr_t)a0 = next_agc_registration_handle(g_agc_owner_handle);
    return 0;
}
HLE(agc_driver_register_resource) {
    if (!a0) return (uint64_t)(int64_t)(int32_t)0x80D19005u;
    *(uint32_t*)(uintptr_t)a0 = next_agc_registration_handle(g_agc_resource_handle);
    return 0;
}

// sceAgcCbDispatch (NID k3GhuSNmBLU) — compute dispatch.
// The authoritative PS5 3.20 export name is sceAgcCbDispatch. Arguments a1-a3 are dispatch
// dimensions and a4 is ShaderDispatchModifier. Modifier.USE_THREAD_DIMENSIONS (bit 5) defines the
// units: Dead Cells passes 0x21 and total thread counts matching its V# records, while Blasphemous 2
// passes 0x1 and UE4 workgroup counts (240x135 groups of 8x8 for a 1920x1080 copy). Preserve the raw
// dimensions here; the executor applies the modifier using the retained local-size registers.
// Kyty's Gen4 GraphicsDispatchDirect is a differently named API and is not an oracle for this Gen5
// builder. DOLL's first real frame issues ~535 of
// these (UE4's compute prologue: GPUScene build, clears, culling) before any graphics draw. Appending
// the packet keeps the stream faithful; the CommandProcessor retains it for ordered submit execution.
// CONFIDENCE: HIGH (PS5 symbol table + live kernels, register state, modifier, and descriptor sizes).
HLE(agc_cb_dispatch) {  // (buf, dimension_x, dimension_y, dimension_z, dispatch_modifier)
    uint32_t* cmd; if (!begin_packet(a0, kDwDispatch, IT_NOP, R_DISPATCH_DIRECT, &cmd)) return 0;
    cmd[1] = (uint32_t)a1; cmd[2] = (uint32_t)a2; cmd[3] = (uint32_t)a3;
    cmd[4] = (uint32_t)(a4 & 0xffffffffu); cmd[5] = (uint32_t)(a4 >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}

// Indirect work is built as three ordered commands: select a graphics/compute argument base,
// synchronize the command-buffer parser with prior shader writes, then consume an offset from that
// base. Astro Bot's live callsites pin the ABI as (dcb, shaderType, baseAddress) followed by
// (dcb, byteOffset, modifier); shaderType is 0 for indexed graphics and 1 for compute. Preserve the
// address and offset in semantic packets so the executor can read the arguments only after earlier
// compute dispatches have completed and written them back to guest memory.
HLE(agc_dcb_set_base_indirect_args) {
    if (a1 > 1) return 0;
    uint32_t* cmd; if (!begin_packet(a0, 4, IT_NOP, R_SET_BASE_INDIRECT_ARGS, &cmd)) return 0;
    cmd[1] = static_cast<uint32_t>(a1);
    cmd[2] = static_cast<uint32_t>(a2);
    cmd[3] = static_cast<uint32_t>(a2 >> 32u);
    return reinterpret_cast<uint64_t>(cmd);
}
HLE(agc_dcb_stall_command_buffer_parser) {
    uint32_t* cmd; if (!begin_packet(a0, kDwStallCbParser, IT_NOP, R_STALL_COMMAND_BUFFER_PARSER, &cmd)) return 0;
    cmd[1] = 0;
    return reinterpret_cast<uint64_t>(cmd);
}
HLE(agc_dcb_draw_index_indirect) {
    uint32_t* cmd; if (!begin_packet(a0, kDwDrawIndexIndirect, IT_NOP, R_DRAW_INDEX_INDIRECT, &cmd)) return 0;
    cmd[1] = static_cast<uint32_t>(a1);
    cmd[2] = static_cast<uint32_t>(a2);
    cmd[3] = static_cast<uint32_t>(a2 >> 32u);
    return reinterpret_cast<uint64_t>(cmd);
}
HLE(agc_cb_dispatch_indirect) {
    uint32_t* cmd; if (!begin_packet(a0, kDwDispatchIndirect, IT_NOP, R_DISPATCH_INDIRECT, &cmd)) return 0;
    cmd[1] = static_cast<uint32_t>(a1);
    cmd[2] = static_cast<uint32_t>(a2);
    cmd[3] = static_cast<uint32_t>(a2 >> 32u);
    return reinterpret_cast<uint64_t>(cmd);
}

// AGC command-buffer size queries (GTA V / PPSA04263, RAGE, issue #1137). A GetSize returns how many
// BYTES a command will occupy so the guest can size its DrawCommandBuffer before building. RAGE's
// render thread (eboot+0x2bfef.. / +0x2bffd..) sums these across a workload, some used directly as a
// byte accumulator and some converted to dwords via (n+3)>>2, then rounds up to a 128-byte boundary
// and allocates the guest DCB (AgcDcb.bottom..top). Stubbed to 0 the guest allocates a 128-byte buffer
// and the builders overflow it -> corrupt submit -> RAGE fatal, before any GPU frame. Each returns the
// matching builder's dword count * 4 (see the begin_packet(a0, N, ...) sizes below); the invariant is
// only that reserved >= written, so a builder prosper hasn't implemented (Rewind) still reserves safely.
// CONFIDENCE: MED — units (bytes) and the sum/allocate contract are pinned from live render-thread
// disassembly; exact per-command dword counts mirror prosper's own builders.
HLE(agc_dcb_jump_get_size)        { (void)a0; return kDwJump * 4u; }
HLE(agc_dcb_acquire_mem_get_size) { (void)a0; return kDwAcquireMem * 4u; }
HLE(agc_dcb_rewind_get_size)      { (void)a0; return 2u * 4u; }   // PM4 REWIND, 2 dwords (reserve-only)
HLE(agc_cb_eop_action_get_size)   { (void)a0; return kDwReleaseMem * 4u; }

// #1756: the rest of the GetSize family whose builder prosper HAS. libSceAgc 3.20 exports 65
// GetSize functions; prosper implemented 4 of them (plus Rewind and Nop) and left the other 59 to
// the generic unimplemented path, which returns **0**. Zero is the worst possible answer here: a
// guest that sizes its command buffer from GetSize reserves NOTHING and the matching builder then
// overruns it. That is the same failure #1748 hit with an off-by-one, and #1137 hit before that
// ("Stubbed to 0 the guest allocates a 128-byte buffer and the builders overflow it -> corrupt
// submit -> RAGE fatal") — each time found by a crash in one title and fixed one NID at a time.
//
// Each of these returns the DWORD COUNT PROSPER'S OWN BUILDER EMITS, which is correct by
// construction and needs no hardware size: the guest reserves exactly what prosper will write, so
// the pair is self-consistent even where prosper's encoding differs from the real packet. That is
// also why this is the safe half of the audit — the unsafe half (a guest with the sizes INLINED,
// which cannot be told anything) is what PROSPER_DCBFULL exists to detect.
//
// Deliberately NOT here: the size-carrying builders whose GetSize argument position is unknown
// (WriteData, SetShRegisterRangeDirect, SetShRegistersDirect — see the issue), and the 33 GetSize
// functions whose builder prosper does not implement either, where a size would be a guess and the
// builder call would fail anyway.
// CONFIDENCE: HIGH on the contract (GetSize == builder emission) and on each NID (3.20 firmware
// export table); the dword counts are prosper's own and are asserted against the builders by
// test_agc_getsize.
HLE(agc_draw_index_get_size)          { (void)a0; return kDwDrawIndex * 4u; }
HLE(agc_draw_index_auto_get_size)     { (void)a0; return kDwDrawIndexAuto * 4u; }
HLE(agc_draw_index_offset_get_size)   { (void)a0; return kDwDrawIndexOffset * 4u; }
HLE(agc_draw_index_indirect_get_size) { (void)a0; return kDwDrawIndexIndirect * 4u; }
HLE(agc_dispatch_get_size)            { (void)a0; return kDwDispatch * 4u; }
HLE(agc_dispatch_indirect_get_size)   { (void)a0; return kDwDispatchIndirect * 4u; }
HLE(agc_dma_data_get_size)            { (void)a0; return kDwDmaData * 4u; }
HLE(agc_event_write_get_size)         { (void)a0; return kDwEventWrite * 4u; }
HLE(agc_set_index_buffer_get_size)    { (void)a0; return kDwSetIndexBuffer * 4u; }
HLE(agc_set_index_count_get_size)     { (void)a0; return kDwSetIndexCount * 4u; }
HLE(agc_set_index_size_get_size)      { (void)a0; return kDwSetIndexSize * 4u; }
HLE(agc_set_num_instances_get_size)   { (void)a0; return kDwSetNumInstances * 4u; }
HLE(agc_stall_cb_parser_get_size)     { (void)a0; return kDwStallCbParser * 4u; }
HLE(agc_set_register_direct_get_size) { (void)a0; return kDwSetRegisterDirect * 4u; }
HLE(agc_set_regs_indirect_get_size)   { (void)a0; return kDwSetRegsIndirect * 4u; }
// sceAgcCbNopGetSize(numDwords): the NOP is exactly the requested dword count (min 1). arg in a0.
HLE(agc_cb_nop_get_size)          { uint32_t n = (uint32_t)a0; if (n == 0) n = 1; return (uint64_t)n * 4u; }

void register_agc_hle() {
    #define RN(nid, fn) Hle::register_fn(nid, (HleFn)(fn), nid)
    #define RN_SUBMIT(nid, fn) Hle::register_fn(nid, (HleFn)(fn), nid, &prosper_gpu_submit_scope_end)
    RN("VEGu4dixjUg", agc_dcb_jump_get_size);        // sceAgcDcbJumpGetSize (#1137)
    RN("-vnlTPPXPrw", agc_dcb_acquire_mem_get_size); // sceAgcDcbAcquireMemGetSize
    RN("ewobAQeMo5k", agc_dcb_acquire_mem_get_size); // sceAgcAcbAcquireMemGetSize (same size)
    RN("QIXCsbipds0", agc_dcb_rewind_get_size);      // sceAgcDcbRewindGetSize
    RN("hL7C0IRpWZI", agc_cb_eop_action_get_size);   // sceAgcCbQueueEndOfPipeActionGetSize
    // #1756: GetSize for every builder prosper has. NIDs from the PS5 3.20 libSceAgc export table.
    RN("6ee9Hd3EWXQ", agc_draw_index_get_size);          // sceAgcDcbDrawIndexGetSize
    RN("WrdP9Zxx3lQ", agc_draw_index_auto_get_size);     // sceAgcDcbDrawIndexAutoGetSize
    RN("qMlfB1ZhMDc", agc_draw_index_offset_get_size);   // sceAgcDcbDrawIndexOffsetGetSize
    RN("mStuvI0zOtc", agc_draw_index_indirect_get_size); // sceAgcDcbDrawIndexIndirectGetSize
    RN("Abendgtz+3o", agc_dispatch_get_size);            // sceAgcCbDispatchGetSize
    RN("w8HVkEeXPv8", agc_dispatch_indirect_get_size);   // sceAgcDcbDispatchIndirectGetSize
    RN("PxKWV2fVAps", agc_dispatch_indirect_get_size);   // sceAgcAcbDispatchIndirectGetSize
    RN("2ccJz9LQI+w", agc_dma_data_get_size);            // sceAgcDcbDmaDataGetSize
    RN("M0ttm8h7SKA", agc_dma_data_get_size);            // sceAgcAcbDmaDataGetSize
    RN("C4l9fB17t8w", agc_event_write_get_size);         // sceAgcDcbEventWriteGetSize
    RN("Y-5vneiBtzk", agc_event_write_get_size);         // sceAgcAcbEventWriteGetSize
    RN("j4emHHndCPY", agc_set_index_buffer_get_size);    // sceAgcDcbSetIndexBufferGetSize
    RN("mljzuGDZRQ4", agc_set_index_count_get_size);     // sceAgcDcbSetIndexCountGetSize
    RN("ca4KPvp0qLQ", agc_set_index_size_get_size);      // sceAgcDcbSetIndexSizeGetSize
    RN("6DFuRKT4C9w", agc_set_num_instances_get_size);   // sceAgcDcbSetNumInstancesGetSize
    RN("+u6dKSLWM2o", agc_stall_cb_parser_get_size);     // sceAgcDcbStallCommandBufferParserGetSize
    RN("QhPDD513V0w", agc_set_register_direct_get_size); // sceAgcDcbSetShRegisterDirectGetSize
    RN("1DeUNpRIDDA", agc_set_register_direct_get_size); // sceAgcDcbSetCxRegisterDirectGetSize
    RN("aP1Ki9G3++4", agc_set_register_direct_get_size); // sceAgcDcbSetUcRegisterDirectGetSize
    RN("nNlUtdDDvZ0", agc_set_regs_indirect_get_size);   // sceAgcDcbSetShRegistersIndirectGetSize
    RN("GBCh3zCihoU", agc_set_regs_indirect_get_size);   // sceAgcDcbSetCxRegistersIndirectGetSize
    RN("UQGTw4xRlcM", agc_set_regs_indirect_get_size);   // sceAgcDcbSetUcRegistersIndirectGetSize
    RN("t7PlZ9nt5Lc", agc_cb_nop_get_size);          // sceAgcCbNopGetSize
    RN("f3dg2CSgRKY", agc_create_shader);   // sceAgcCreateShader — populates the shader registry
    RN("dolOmWH+huQ", agc_get_fused_shader_size); // sceAgcGetFusedShaderSize (Pathless SDK)
    RN("nQT5kYLv0cg", agc_get_fused_shader_size); // sceAgcGetFusedShaderSize (PS5 3.20)
    RN("nApJjpKNBl4", agc_fuse_shader_halves);    // sceAgcFuseShaderHalves (PS5 3.20)
    RN("fd5Bp5tGTgo", agc_fuse_shader_halves);    // older observed SDK alias
    RN("AOLcoIkQDgM", agc_driver_query_resource_registration_user_memory_requirements);
    RN("uJziRsODk1c", agc_driver_get_resource_registration_max_name_length);
    RN("X-Nm5KLREeg", agc_driver_register_owner);
    RN("W5z4eZrjEas", agc_driver_register_resource);
    RN("V++UgBtQhn0", agc_get_data_packet_payload);          // data packet -> register-bank payload
    RN("n2fD4A+pb+g", agc_cb_set_sh_register_range_direct);  // SET_SH_REG range packet
    RN("UZbQjYAwwXM", agc_cb_set_sh_registers_direct);       // non-contiguous SET_SH_REG packets
    RN("D9sr1xGUriE", agc_create_prim_state);                // prim registers from gs specials
    // CreateInterpolantMapping changed NID across SDK revisions. PS5 3.20 names pdEV7bI6COI;
    // Cobra imports dbOlWdppb4o, and its eboot wrapper at +0x3b2780 proves the same
    // (regs[32], producer type 2, pixel type 1) ABI before copying all 32 records into the Cx bank.
    // Leaving that alias on the generic success stub advertised 32 unwritten stack records and
    // made coincidental junk offsets program real registers.
    RN("pdEV7bI6COI", agc_create_interpolant_mapping);       // PS5 3.20
    RN("dbOlWdppb4o", agc_create_interpolant_mapping_sdk_alias); // Cobra/DQ SDK revision
    RN("HV4j+E0MBHE", agc_create_interpolant_mapping);       // older observed revision
    RN("TRO721eVt4g", agc_dcb_reset_queue);
    RN("+kSrjIVxKFE", agc_dcb_push_marker);
    RN("QhCbS4X9Rl8", agc_dcb_push_marker);  // sceAgcDcbSetMarker (same native marker packet)
    RN("H7uZqCoNuWk", agc_dcb_pop_marker);
    RN("qj7QZpgr9Uw", agc_dcb_type2_nop);
    RN("tSBxhAPyytQ", agc_dcb_set_num_instances);
    RN("MWiElSNE8j8", agc_dcb_wait_safe_for_rendering);
    RN("LHFXRrlTPD8", agc_dcb_set_cx_register_direct);
    RN("pFLArOT53+w", agc_dcb_set_sh_register_direct);
    RN("w4-d0n60hdo", agc_dcb_set_uc_register_direct);
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
    // Async-compute command-buffer builders share the packet encodings with their DCB siblings.
    // Astro Bot builds producer fences on ACBs and waits for them from its graphics DCB; dropping
    // these calls leaves the consumer queue permanently parked on labels that can never be written.
    RN("JrtiDtKeS38", agc_dcb_reset_queue);       // sceAgcAcbResetQueue
    RN("cpCILPya5Zk", agc_dcb_push_marker);       // sceAgcAcbPushMarker
    RN("6mFxkVqdmbQ", agc_dcb_pop_marker);        // sceAgcAcbPopMarker
    RN("KT-hTp-Ch14", agc_dcb_acquire_mem);       // sceAgcAcbAcquireMem
    RN("cFazmnXpJOE", agc_dcb_event_write);       // sceAgcAcbEventWrite
    RN("htn36gPnBk4", agc_dcb_wait_reg_mem);      // sceAgcAcbWaitRegMem
    // sceAgcAcbWriteData — the WriteData member of this same ACB set, omitted when the others landed.
    // Syberia: Remastered (PPSA30140, Unity/IL2CPP) builds its ACB fence cluster with it; dropping the
    // call left the label permanently unwritten and parked Unity's main thread in the guest's own
    // poll loop (`mov (%rdx),%rsi; cmp %rcx,%rsi; jne .-3` at eboot+0x1152b7e3) after 1 flip. The
    // guest never reached a second present. Sharing the DCB builder unblocks the boot to its profile
    // menu. ABI verified against the DCB sibling by the num_dwords guard below staying silent across
    // a full boot: a shifted ABI would land garbage in a5 and trip the loud REJECT.
    // CONFIDENCE: MED (matches the five ACB builders above; sceAgcAcbDmaData is the known exception
    // that needed its own shifted-ABI handler, so re-check if a title ever trips the reject here).
    RN("eZ4+17OQz4Q", agc_dcb_write_data);        // sceAgcAcbWriteData
    // Indirect-register patch helpers (SetAddress patches cmd[2..3]; AddRegisters patches cmd[1]).
    RN("vcmNN+AAXnY", agc_patch_set_address);   RN("d-6uF9sZDIU", agc_patch_add_registers);   // Cx
    RN("Qrj4c+61z4A", agc_patch_set_address);   RN("z2duB-hHQSM", agc_patch_add_registers);   // Sh
    RN("6lNcCp+fxi4", agc_patch_set_address);   RN("vRoArM9zaIk", agc_patch_add_registers);   // Uc
    // ...PatchSetNumRegisters: the SET (non-accumulating) count patcher of the same family (#305).
    RN("whb1RL7K4Ss", agc_patch_set_num_registers_cx);   // sceAgcSetCxRegIndirectPatchSetNumRegisters
    RN("nCUgItdN2ms", agc_patch_set_num_registers_sh);   // sceAgcSetShRegIndirectPatchSetNumRegisters
    RN("fRG-JOH5+sI", agc_patch_set_num_registers_uc);   // sceAgcSetUcRegIndirectPatchSetNumRegisters
    RN("LtTouSCZjHM", agc_cb_nop);              // sceAgcCbNop — append padding dwords (#117)
    // Sync-packet address patchers + compute dispatch (issue #213 — the DOLL per-draw fence
    // cluster; RE write-up at each handler). These OVERRIDE the glog no-op thunks (map insert
    // order: register_agc_hle runs after the glog registration, same as WmAc2MEj6Io).
    RN("fPSCdQxgpSw", agc_patch_write_data_addr);    // WriteData packet: set destination address
    RN("3KDcnM3lrcU", agc_patch_wait_reg_mem_addr);  // WaitRegMem packet: set label address
    RN("0fWWK5uG9rQ", agc_patch_release_mem_addr);   // ReleaseMem packet: set label address
    RN("k3GhuSNmBLU", agc_cb_dispatch);              // sceAgcCbDispatch (dimensions x/y/z, modifier)
    RN("RmaJwLtc8rY", agc_dcb_set_base_indirect_args);       // graphics/compute indirect base
    RN("u2T2DiA5hRI", agc_dcb_stall_command_buffer_parser); // parser visibility barrier
    RN("t1vNu082-jM", agc_dcb_draw_index_indirect);         // indexed indirect draw
    RN("CtB+A9-VxO0", agc_cb_dispatch_indirect);            // DCB indirect compute dispatch
    RN("j3EtxFkSIhQ", agc_cb_dispatch_indirect);            // ACB sibling
    RN("WmAc2MEj6Io", agc_dcb_dma_data);        // sceAgcDcbDmaData — append DMA_DATA packet (#117/#312)
    RN("-RnpfpxIhec", agc_acb_dma_data);        // sceAgcAcbDmaData — async-compute sibling (#312)
    RN("IxYiarKlXxM", agc_patch_dma_data_dst);  // sceAgcDmaDataPatchSetDstAddressOrOffset
    RN("cdDRpqcFGbU", agc_patch_dma_data_src);  // sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate
    RN("YUeqkyT7mEQ", agc_dcb_set_flip);        // sceAgcDcbSetFlip (Kyty LibGraphicsDriver.cpp:98)
    // Completion labels stay private until the handler has returned into its generated import
    // trampoline. This per-NID checkpoint ends the submit scope immediately before guest return.
    RN_SUBMIT("UglJIZjGssM", agc_driver_submit_dcb);   // sceAgcDriverSubmitDcb -> CommandProcessor replay
    RN_SUBMIT("gSRnr79F8tQ", agc_driver_submit_acb);   // sceAgcDriverSubmitAcb -> ordered compute replay
    RN_SUBMIT("w1KFAHVqpaU", agc_driver_submit_dcb_variant);  // final-buffer submit variant (#232, DOLL RHI batch)
    #undef RN_SUBMIT
    #undef RN
}

}  // namespace prosper
