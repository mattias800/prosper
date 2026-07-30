// command_processor.cpp — see command_processor.hpp.
#include "command_processor.hpp"
#include "mb3_freelist.hpp"
#include "pm4_registers.hpp"
#include "writer_provenance.hpp"
#include "hle/sync_futex.hpp"   // wake_label_waiters (shared with sceKernelWaitOnAddress's futex)

// hle_graphics.cpp: perform the videoout flip for an in-stream SetFlip packet — advances the flip
// status (count/flipArg/currentBuffer) that sceVideoOutGetFlipStatus reports, exactly like the API
// flip does. The game's frame pacer polls that status for its submitted flipArg; a dropped in-stream
// flip stalls the frame loop at one rendered frame.
extern "C" void prosper_vo_flip_from_gpu(uint32_t handle, int32_t bufidx, uint32_t flip_mode, int64_t flip_arg);
// hle_kernel_time.cpp: fire the GPU EOP equeue events the game registered via sceGnmAddEqEvent.
// The submit paths pulse at submit time; flush_deferred_streams pulses AGAIN when a deferred
// stream's gated writes finally land (#312 barrier model) — an equeue waiter that consumed the
// submit-time pulse, checked its still-gated label and went back to sleep would otherwise never
// be woken (wake_on_label only wakes sync_on_address futex waiters, not equeue waiters; observed
// live as DOLL's "GameThread timed out waiting for RenderThread after 120.00 secs" wedge).
namespace prosper { void prosper_eq_trigger_eop(); }
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <array>
#include <deque>
#include <unordered_set>
#include <condition_variable>
#include <thread>

namespace prosper::gpu {

// Readability probe (gpu_executor.cpp, declared in gpu_execute.hpp): page-granular check that a
// guest range is mapped, so the Jump fold below never walks an unmapped segment address.
bool guest_readable(uint64_t addr, uint32_t bytes);
bool guest_writable(uint64_t addr, uint32_t bytes);
// Guest GPU writes invalidate renderer-owned copies of overlapping resources.
void notify_guest_gpu_write(uint64_t addr, uint64_t size);

// Wake any thread blocked in sync_on_address (a futex) on `addr`. A GPU completion label write only
// changes memory; a futex waiter does NOT wake on a value change — it needs an explicit FUTEX_WAKE. The
// game's render/producer threads sync_on_address on the very labels the GPU writes via RELEASE_MEM /
// WRITE_DATA, so without this wake they block forever on already-satisfied semaphores (the documented
// 3-thread render deadlock — see hle_kernel_mem.cpp). This provides that missing GPU-completion wake.
// wake_label_waiters shares the sync HLE's futex implementation and skips the syscalls (this runs per
// RELEASE_MEM/WRITE_DATA packet) when no thread is blocked.
// CONFIDENCE: HIGH (matches the futex model of sceKernelWaitOnAddress; guest+host share the address space).
static void wake_on_label(uint64_t addr) { wake_label_waiters(addr); }

// Disabled only for bring-up bisection. Honoring the Dcb's memory writes is correct default behavior:
// because our CommandProcessor folds each submit synchronously, the pipe has "drained" by the time we
// apply a packet, so this IS the end-of-pipe moment. Set PROSPER_NO_EOP_WRITE=1 to suppress the writes.
static bool eop_writes_disabled() {
    const char* off = getenv("PROSPER_NO_EOP_WRITE");
    return off && off[0] == '1';
}

// A monotonic 64-bit "GPU clock" for RELEASE_MEM data_sel==3 (GpuClock64). On real hardware the GPU
// EOP timestamp is the SAME counter the guest reads via sceKernelReadTsc (Kyty: GraphicsRender writes
// KernelReadTsc() for the EOP timestamp; GetGpuCoreClockFrequency == GetTscFrequency), so we share
// the guest TSC clock rather than a separate steady_clock (#156). It reports monotonic nanoseconds at
// the 1 GHz that sceKernelGetTscFrequency advertises, so a guest that reads two fence timestamps and
// divides the delta by the queried frequency gets real seconds — AND a GPU fence timestamp lies on the
// same timeline as a CPU sceKernelReadTsc value (the old steady_clock had a disjoint epoch/period).
extern "C" uint64_t prosper_guest_tsc_ns();   // hle_kernel_time.cpp — same source as sceKernelReadTsc
static uint64_t gpu_clock64() { return prosper_guest_tsc_ns(); }

// --- GPU-write attribution ring (diagnostic for issue #312 heap-corruption hunt). ---------------
// Records every guest-memory write this command processor performs (EOP label / WRITE_DATA /
// EVENT_WRITE) in a fixed lock-free ring so the fault handler can answer, async-signal-safely,
// "did the GPU recently write near address X?" — the attribution question for a stomped
// MallocBinned3 free-block canary. Always-on: 3 relaxed atomics per honored write, no allocation.
namespace {
struct GpuWriteRec { uint64_t addr; uint64_t value; uint64_t pkt; uint32_t seq; uint8_t size; uint8_t kind; };
constexpr uint32_t kWriteRingSize = 16384;              // power of two
GpuWriteRec g_write_ring[kWriteRingSize];
std::atomic<uint32_t> g_write_seq{0};
// Coarse monotonic ms since process start (diagnostic timestamps for the #312 fence journal).
uint64_t now_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
}
// `pkt` = the guest address of the PM4 packet that requested the write (c.payload-1). The stale-
// vs-live discriminator: a STALE ring-tail packet re-executed across frames has the SAME pkt
// address in every record; a legitimately re-recorded per-frame write moves with the ring.
void ring_record(uint64_t addr, uint64_t value, uint8_t size, uint8_t kind, uint64_t pkt) {
    uint32_t s = g_write_seq.fetch_add(1, std::memory_order_relaxed);
    GpuWriteRec& r = g_write_ring[s & (kWriteRingSize - 1)];
    r.addr = addr; r.value = value; r.pkt = pkt; r.seq = s; r.size = size; r.kind = kind;
}
uint64_t pkt_addr(const Pm4Command& c) { return c.payload ? (uint64_t)(uintptr_t)(c.payload - 1) : 0; }
}
// Scan the ring for writes intersecting [lo, hi); format up to `max` matches into out (NUL-
// terminated). Async-signal-safe: no locks, no allocation, tolerates racy ring slots. kind:
// 1=RELEASE_MEM 2=EVENT_WRITE 3=WRITE_DATA 4=DMA_DATA.
extern "C" int prosper_gpu_write_ring_scan(uint64_t lo, uint64_t hi, char* out, size_t cap) {
    size_t off = 0; int found = 0;
    uint32_t seq_now = g_write_seq.load(std::memory_order_relaxed);
    uint32_t n = seq_now < kWriteRingSize ? seq_now : kWriteRingSize;
    for (uint32_t i = 0; i < n && off + 96 < cap; i++) {
        const GpuWriteRec& r = g_write_ring[i & (kWriteRingSize - 1)];
        if (!r.addr || r.addr + r.size <= lo || r.addr >= hi) continue;
        int m = snprintf(out + off, cap - off,
                         "[gpuring] seq=%u kind=%u addr=0x%llx size=%u value=0x%llx pkt=0x%llx (age=%u)\n",
                         r.seq, r.kind, (unsigned long long)r.addr, r.size,
                         (unsigned long long)r.value, (unsigned long long)r.pkt, seq_now - r.seq);
        if (m > 0) off += (size_t)m;
        found++;
    }
    if (off < cap) out[off] = 0;
    return found;
}

// --- #312 fence BUILD journal: the timing-vs-wrong-target discriminator. -------------------------
// The AGC builders/patchers record, per fence packet (keyed by the packet's guest address):
// the target label address the GUEST passed, the 8 bytes the target held AT BUILD TIME, and a
// build timestamp. When honor_eop_write later catches a fence write landing over pointer-like
// (freed-heap-header-shaped) memory, the journal answers: (a) did the packet's target change
// between build and fold (packet mutated/mis-decoded => wrong-target), (b) was the target ALREADY
// freed-looking when the guest built the fence (stale guest structure), or (c) was it clean at
// build and freed in the build->write window (ordering: our write lands after the guest's free)?
// Direct-mapped by packet address — collisions just replace (diagnostic-grade).
namespace {
struct FenceBuildRec { uint64_t pkt, addr, pre; uint64_t t_ms; };
constexpr uint32_t kJournalSize = 65536;                // power of two
FenceBuildRec g_fence_journal[kJournalSize];
inline uint32_t journal_slot(uint64_t pkt) { return (uint32_t)((pkt >> 2) * 2654435761u) & (kJournalSize - 1); }
}
extern "C" void prosper_fence_journal_record(uint64_t pkt, uint64_t addr) {
    if (!pkt) return;
    uint64_t pre = 0;
    if (addr >= 0x10000 && !(addr & 3)) memcpy(&pre, (const void*)(uintptr_t)addr, sizeof pre);
    FenceBuildRec& r = g_fence_journal[journal_slot(pkt)];
    r.pkt = pkt; r.addr = addr; r.pre = pre; r.t_ms = now_ms();
}
extern "C" int prosper_fence_journal_lookup(uint64_t pkt, uint64_t* addr, uint64_t* pre, uint64_t* t_ms) {
    if (!pkt) return 0;
    const FenceBuildRec& r = g_fence_journal[journal_slot(pkt)];
    if (r.pkt != pkt) return 0;
    *addr = r.addr; *pre = r.pre; *t_ms = r.t_ms;
    return 1;
}

// --- #312 per-label protocol history: the "missing init leg" discriminator. ----------------------
// The consumed-marker protocol per 64 KiB command chunk is (RE'd, eboot+0x22122bd / 0x220d2d2):
//   DmaData(label := 0, 4B)  ...  ReleaseMem(label <- 1, dsel=1)   — same cb, stream-ordered,
// where label = an UNINITIALIZED Malloc(0x20) block (malloc residue = an MB3 freelist pointer is
// EXPECTED at fence-build time; the GPU DmaData is the only initializer). So a pointer-valued label
// at WRITE time means exactly one of: (a) the init DmaData never built, (b) built but not executed
// (skipped/deferred/other-queue), or (c) executed but the guest re-pointered the block afterwards
// (freed/re-linked => our fence is late or unexpected). This table answers which, per address.
namespace {
// Event types for the per-label ring. Build events fire in the AGC builder HLEs (guest thread);
// exec events fire in the honor_* paths (fold thread / pend worker).
enum : uint8_t { LE_DMA_BUILT = 1, LE_REL_BUILT, LE_WAIT_BUILT, LE_DMA_EXEC, LE_REL_EXEC, LE_DMA_SKIP,
                 LE_DMA_FREE, LE_REL_FREE };
struct LabelEvent {
    uint8_t  type;
    uint8_t  origin;      // #1226: exec events — queue_origin of the folding submit (0=?, 1=Dcb, 2=Acb, 3=Final)
    uint32_t fold;        // g_fold_seq at event time (exec events; builds carry it too for context)
    uint64_t t_ms;
    uint64_t aux;         // builds: packet addr; execs: pre-content qword at the label
};
struct LabelHist {
    uint64_t addr;
    std::atomic<uint32_t> n;              // total events (ring index)
    std::atomic<uint32_t> dma_built_n, dma_exec_n, rel_built_n, rel_exec_n;   // overlap counters
    std::atomic<uint32_t> mb3_dma_suppressed_n, mb3_rel_debt_consumed_n;
    LabelEvent ev[16];                    // last 16 events
};
constexpr uint32_t kLabelHistSize = 16384;            // power of two
LabelHist g_label_hist[kLabelHistSize];
std::atomic<uint32_t> g_fold_seq{0};                  // top-level fold counter (submit streams)
inline LabelHist& label_hist_slot(uint64_t addr) {
    LabelHist& h = g_label_hist[(uint32_t)((addr >> 2) * 2654435761u) & (kLabelHistSize - 1)];
    if (h.addr != addr) {                              // collision/new: reset (diagnostic-grade)
        h.addr = addr; h.n.store(0, std::memory_order_relaxed);
        h.dma_built_n = h.dma_exec_n = h.rel_built_n = h.rel_exec_n = 0;
        h.mb3_dma_suppressed_n = h.mb3_rel_debt_consumed_n = 0;
        memset(h.ev, 0, sizeof h.ev);
    }
    return h;
}
void label_hist_event(uint64_t addr, uint8_t type, uint64_t aux, uint8_t origin = 0) {
    if (!addr) return;
    LabelHist& h = label_hist_slot(addr);
    uint32_t i = h.n.fetch_add(1, std::memory_order_relaxed);
    LabelEvent& e = h.ev[i & 15];
    e.type = type; e.origin = origin; e.t_ms = now_ms(); e.aux = aux;
    e.fold = g_fold_seq.load(std::memory_order_relaxed);
}
}
extern "C" void prosper_label_hist_dma_built(uint64_t addr, uint64_t cb, uint32_t /*src*/, uint8_t builder) {
    if (!addr) return;
    label_hist_slot(addr).dma_built_n.fetch_add(1, std::memory_order_relaxed);
    label_hist_event(addr, LE_DMA_BUILT, cb | builder);   // cb object ptr | 1=Dcb 2=Acb (cb is 8-aligned)
}
extern "C" void prosper_label_hist_rel_built(uint64_t addr, uint64_t cb) {
    if (!addr) return;
    label_hist_slot(addr).rel_built_n.fetch_add(1, std::memory_order_relaxed);
    label_hist_event(addr, LE_REL_BUILT, cb);
}
extern "C" void prosper_label_hist_wait_built(uint64_t addr, uint64_t cb) {
    label_hist_event(addr, LE_WAIT_BUILT, cb);
}
// Dump a label's event ring outside the suspect path (WaitRegMem violations — #312).
extern "C" void prosper_label_hist_dump(uint64_t addr, char* out, unsigned cap);
namespace {
uint64_t peek_qword(uint64_t addr) {
    uint64_t v = 0;
    if (addr >= 0x10000 && !(addr & 3)) memcpy(&v, (const void*)(uintptr_t)addr, sizeof v);
    return v;
}
void label_hist_dma_exec(uint64_t addr, uint64_t pre, uint8_t origin = 0) {
    label_hist_slot(addr).dma_exec_n.fetch_add(1, std::memory_order_relaxed);
    label_hist_event(addr, LE_DMA_EXEC, pre, origin);
}
void label_hist_dma_skip(uint64_t addr)               { label_hist_event(addr, LE_DMA_SKIP, 0); }
void label_hist_rel_exec(uint64_t addr, uint64_t pre, uint8_t origin = 0) {
    label_hist_slot(addr).rel_exec_n.fetch_add(1, std::memory_order_relaxed);
    label_hist_event(addr, LE_REL_EXEC, pre, origin);
}
void label_hist_dma_free(uint64_t addr, uint64_t pool_base) {
    label_hist_slot(addr).mb3_dma_suppressed_n.fetch_add(1, std::memory_order_relaxed);
    label_hist_event(addr, LE_DMA_FREE, pool_base);
}
bool label_hist_take_dma_free_debt(uint64_t addr) {
    LabelHist& h = label_hist_slot(addr);
    uint32_t used = h.mb3_rel_debt_consumed_n.load(std::memory_order_relaxed);
    for (;;) {
        uint32_t made = h.mb3_dma_suppressed_n.load(std::memory_order_acquire);
        if (used >= made) return false;
        if (h.mb3_rel_debt_consumed_n.compare_exchange_weak(
                used, used + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) return true;
    }
}
void label_hist_rel_free(uint64_t addr, uint64_t pool_base) {
    label_hist_event(addr, LE_REL_FREE, pool_base);
}
// #312 in-flight-overlap probe: at REL1 exec time, a SECOND init/fence pair already built for the
// same label (built-execed >= 2 pending, counting this one) means two fence generations were in
// flight together — the guest may free the label on the FIRST 1 while the second pair is still
// queued, and the late pair then stomps the freed block (bundle-next := 1 -> the exact
// free(0x1000000001) / NextFreeBlock==0x1 / canary fatal family). Returns pending inits.
int label_rel_overlap(uint64_t addr) {
    LabelHist& h = label_hist_slot(addr);
    uint32_t db = h.dma_built_n.load(std::memory_order_relaxed);
    uint32_t dx = h.dma_exec_n.load(std::memory_order_relaxed);
    return (int)(db - dx);
}
// #312: is this address part of DOLL's DmaData(:=0)+ReleaseMem(<-1) consumed-marker protocol? A
// plain fence label (Messenger's ReleaseMem/WriteData targets) has NO DmaData-init history, so this
// gates the REL1-stomp guard to the exact population that carries the corruption — never a title
// whose fences are simple EOP labels. (label_hist_slot resets on hash collision, so a false 0 only
// forgoes the guard on one write; the protocol re-arms it on the address's next DmaData build.)
bool label_is_consumed_marker(uint64_t addr) {
    return label_hist_slot(addr).dma_built_n.load(std::memory_order_relaxed) > 0;
}
// #312 CLOSE — label free-state model (sessions 3-10 established the residual is a write-after-free,
// NOT resolvable from write CONTENT alone). A LIVE consumed-marker label, at ReleaseMem(<-value)
// time, holds exactly one of: its DmaData-init'd 0, or the already-signaled fence `value` (idempotent
// re-fence). The guest polls it, and the instant it reads the fence value it FREES the 0x20 block and
// LIFO-recycles it. So any deviation identifies a FREED/reused block:
//   Case A (content): pre is neither 0 nor `value` -> the block was freed and its memory reused for an
//     FFreeBlock header (a 64 KiB-aligned next-pointer 0x1000000000, a canary, or a size/count field).
//     Writing `value` over it forges the exact #312 corruption — the pointer-forge (#510), the
//     0x20015f00 misaligned-read artifact, AND the "Canary was 0x3, should be 0x1" fatal are all this
//     one sub-qword write landing on different FFreeBlock fields. Subsumes #510's forges_freelist_ptr
//     and #505's REL1-LIVE (both are pre!=0 && pre_low != value shapes).
//   Case B (protocol free-state): pre == 0 is ambiguous — a live init'd label AND a freed block whose
//     NextFreeBlock reads NULL both read 0, and writing 1 over a NULL next-pointer forges
//     NextFreeBlock==0x1 (the deref-0x1 worker fault). Disambiguate via TRACKED lifecycle counters:
//     each real DmaData(:=0) exec bumps dma_exec_n, each honored ReleaseMem bumps rel_exec_n (a
//     SUPPRESSED write bumps NEITHER). A live fence therefore has a pending un-signaled init
//     (dma_exec_n > rel_exec_n) here; when signals have caught up to inits (rel_exec_n >= dma_exec_n,
//     dma_exec_n>0) this ReleaseMem is a stale/duplicate fence to an already-consumed+freed block.
// Gated by the caller to consumed-marker labels only, so plain fence labels (Messenger) never qualify.
// Strictly correct: a freed block satisfies no live WaitRegMem==value consumer (its content is not
// `value`, or the guest already consumed the value and freed it). CONFIDENCE: MED-HIGH.
// Returns the freed-manifestation category (for verification logging), or 0 if the label looks live.
//   2 = the fence's LOW DWORD (the 4 bytes the guest polls) is neither the DmaData-init'd 0 nor the
//       fence value — a freed/reused FFreeBlock header field (canary / size / count) sits there. This
//       is the "Canary 0x3" residual that value-SHAPE (ptr_like) guards miss. A live consumed-marker
//       label's low dword is ONLY ever 0 or the fence value, so this cannot be a live label. NOTE: we
//       key on the LOW DWORD, not the qword — the high dword of a live 4-byte-fence label is malloc
//       residue (e.g. pre=0x1_00000000, low dword 0 = a LIVE init'd label; suppressing on the qword
//       wrongly killed those and crashed boot). The ptr_like next-pointer cases (#505 REL1-LIVE, #510
//       forge, which have low dword 0 or a real pointer half) are handled by the guards below.
//   3 = LOW DWORD is 0 (looks like a freshly init'd LIVE label) BUT the tracked protocol counters
//       show NO pending un-signaled DmaData init (rel_exec_n >= dma_exec_n): this is a stale/
//       duplicate fence to a block the guest already consumed + freed + LIFO-recycled, whose
//       FFreeBlock next-pointer reads NULL — writing the fence value would forge NextFreeBlock==0x1
//       (the eboot+0x231012b deref-0x1 fatal / "Canary 0x3"). Content is INDISTINGUISHABLE from a
//       live label here (both read low dword 0), so ONLY the lifecycle counters resolve it — this is
//       the guest-free-state track the residual needs. A live fence always has its paired init
//       pending (dma_exec_n > rel_exec_n) at this point, so it is never suppressed.
int label_freed_marker_kind(uint64_t addr, uint64_t pre, uint64_t value) {
    uint32_t lo = (uint32_t)pre;
    uint32_t vlo = (uint32_t)value;
    if (lo != 0 && lo != vlo) return 2;                                  // Case A'
    if (lo == 0 && vlo != 0) {                                           // Case B (free-state track)
        LabelHist& h = label_hist_slot(addr);
        uint32_t dx = h.dma_exec_n.load(std::memory_order_relaxed);
        uint32_t rx = h.rel_exec_n.load(std::memory_order_relaxed);
        if (dx > 0 && rx >= dx) return 3;
    }
    return 0;
}
// Format the event ring for a suspect report, oldest first. b=built x=exec, aux in hex.
void label_hist_report(uint64_t addr, char* out, size_t cap) {
    static const char* nm[] = {"?", "dmaB", "relB", "waitB", "dmaX", "relX", "dmaSKIP",
                               "dmaFREE", "relFREE"};
    const LabelHist& h = label_hist_slot(addr);
    uint32_t n = h.n.load(std::memory_order_relaxed);
    uint32_t first = n > 16 ? n - 16 : 0;
    size_t off = (size_t)snprintf(out, cap, "events(total=%u):", n);
    // Exec events carry the folding submit entry point (#1226): D=SubmitDcb A=SubmitAcb
    // F=SubmitDcbFinal, absent when unknown/build-side — the cross-queue discriminator.
    static const char* qn[] = {"", "(D)", "(A)", "(F)"};
    for (uint32_t i = first; i < n && off + 52 < cap; i++) {
        const LabelEvent& e = h.ev[i & 15];
        int m = snprintf(out + off, cap - off, " %s%s@%llu/f%u:0x%llx",
                         e.type <= 8 ? nm[e.type] : "?", e.origin <= 3 ? qn[e.origin] : "",
                         (unsigned long long)e.t_ms, e.fold, (unsigned long long)e.aux);
        if (m > 0) off += (size_t)m;
    }
}
}
extern "C" void prosper_label_hist_dump(uint64_t addr, char* out, unsigned cap) {
    label_hist_report(addr, out, cap);
}

// #312: "pointer-like" pre-content — a freed MallocBinned3 block header holds heap pointers into
// the 512 GiB arena (0x1000000000) or the allocator-metadata pools (0x2000000000 region, where the
// FPoolInfo tables live — e.g. the 0x20015f0000 pool-info table of #161/#241).
static bool ptr_like(uint64_t v) {
    return (v >= 0x1000000000ull && v < 0x1200000000ull) ||
           (v >= 0x2000000000ull && v < 0x2100000000ull);
}
// #312 POOLSHIFT tripwire: the dominant residual crash reads a BYTE-SHIFTED pool-info pointer
// (historically `0x20015f00` == `0x20015f0000 >> 8` at eboot+0x2316c91). A qword is "byte-shifted
// pool ptr" when its top 32 bits are 0 and (v<<8) lands in the allocator-metadata region — i.e. an
// 8-byte pool pointer stored one byte too LOW (an aligned read then drops the low byte). This
// catches ANY GPU-path write that CREATES that shape in the act, with the packet builder callsite.
// #1226: the window is [0x2000000000, 0x4000000000) — the old DOLL-era [0x20..0x21) window went
// stale when prosper's dmem layout moved: post-#1249 faults on BOTH DOLL and ArcRunner dereference
// `0x30015f00` (<<8 = 0x30015f0000, the same FPoolInfo +0x15f0000 arena offset), which the old
// window could not see (a silent false-negative for the exact hunt this tripwire exists for).
static inline bool is_byteshift_poolptr(uint64_t v) {
    if (v >> 32) return false;                 // must fit in low 32 bits (top byte(s) were dropped)
    if (v < 0x100000) return false;            // ignore tiny ints
    uint64_t s = v << 8;
    return s >= 0x2000000000ull && s < 0x4000000000ull;
}
// A full (unshifted) pool-info pointer as a WRITE PAYLOAD is itself anomalous for a GPU fence/label
// write (fence values are small ints / timestamps), so flag that too. (#1226: same widened window.)
static inline bool is_poolinfo_ptr(uint64_t v) {
    return v >= 0x2000000000ull && v < 0x4000000000ull;
}
// Scan the just-written span [dst-8, dst+bytes+8) for a byte-shifted pool pointer and log the GPU
// writer (kind + dst + packet builder addr) if found. Bounded; small writes only (the label/pointer
// writes — large fills are memset-0). `payload` is the value the packet wrote (for payload flagging).
static void poolshift_check(const char* kind, uint64_t dst, uint64_t bytes, uint64_t payload, uint64_t pkt) {
    // Gated OFF by default (PROSPER_POOLSHIFT=1 to arm): the per-write span scan adds cost to every
    // fence/label write, so keep it out of the Messenger/default path. It found ZERO hits across
    // many DOLL crashing runs — decisive evidence the GPU write path never creates the byte-shifted
    // 0x20015f00 pool pointer — so it stays as diagnostic-only instrumentation for the next session.
    static const bool on = getenv("PROSPER_POOLSHIFT") != nullptr;
    if (!on) return;
    // #1252 review: PAYLOAD and STOMP get SEPARATE caps — with the widened windows, a shared cap
    // let ordinary address-source DMA payloads (full arena VAs now match is_poolinfo_ptr) burn the
    // budget and silence the stomp scan, the exact silent-false-negative this tripwire hunts.
    // The payload flag also requires the same mapped-target proof as the scan (byteshift form) and
    // skips DMA (whose `payload` is a SOURCE ADDRESS, legitimately arena-valued, not written data).
    static std::atomic<int> n_payload{0};
    static std::atomic<int> n{0};   // STOMP-scan reports only
    const bool payload_suspicious =
        (is_byteshift_poolptr(payload) && guest_readable(payload << 8, 8)) ||
        (is_poolinfo_ptr(payload) && kind[0] != 'D');
    if (payload_suspicious) {
        if (n_payload.fetch_add(1) < 128)
            fprintf(stderr, "[agc] POOLSHIFT-PAYLOAD kind=%s dst=0x%llx bytes=%llu payload=0x%llx pkt=eboot+0x%llx t=%llums\n",
                    kind, (unsigned long long)dst, (unsigned long long)bytes,
                    (unsigned long long)payload, (unsigned long long)pkt, (unsigned long long)now_ms());
    }
    if (n.load(std::memory_order_relaxed) >= 128) return;
    if (bytes > 256) return;                    // cap the scan (label/pointer writes are small)
    uint64_t lo = (dst >= 8 ? dst - 8 : dst) & ~7ull;
    uint64_t hi = dst + bytes + 8;
    for (uint64_t a = lo; a < hi; a += 8) {
        uint64_t q = peek_qword(a);
        if (is_byteshift_poolptr(q)) {
            // #1226 precision: the widened window admits any dword in [0x20000000,0x3fffffff]
            // (floats, sizes) — a REAL shifted pool pointer's <<8 must at least be MAPPED guest
            // memory (0x3d787f0000-style unmapped values are guest data, not pointers). Also
            // dedup consecutive repeats: one stale value next to a hot per-frame label otherwise
            // saturates the report cap in seconds (measured: 128/128 on one address).
            if (!guest_readable(q << 8, 8)) continue;
            static std::atomic<uint64_t> last_a{0}, last_q{0};
            if (a == last_a.load(std::memory_order_relaxed) && q == last_q.load(std::memory_order_relaxed)) continue;
            last_a.store(a, std::memory_order_relaxed); last_q.store(q, std::memory_order_relaxed);
            bool inspan = (a + 7 >= dst) && (a < dst + bytes);   // did THIS write touch these bytes?
            if (n.fetch_add(1) < 128)
                fprintf(stderr, "[agc] POOLSHIFT-STOMP kind=%s @0x%llx=0x%llx (<<8=0x%llx) dst=0x%llx bytes=%llu inspan=%d payload=0x%llx pkt=eboot+0x%llx t=%llums\n",
                        kind, (unsigned long long)a, (unsigned long long)q, (unsigned long long)(q << 8),
                        (unsigned long long)dst, (unsigned long long)bytes, inspan,
                        (unsigned long long)payload, (unsigned long long)pkt, (unsigned long long)now_ms());
        }
    }
}
// #312 (session-10 capture): the ROOT corruptor forges a freelist next-pointer. A per-thread MB3
// pool cache head was caught being written 0x20015f00 by the guest allocator's own freelist POP
// (eboot+0x2316ad2) — propagating an already-corrupt chain whose old head was 0x1000000001. That
// 0x1000000001 = 0x1000000000 | 1, i.e. a live freelist next-pointer 0x1000000000 (a 64 KiB-aligned
// block, low dword ZERO) whose low dword was overwritten with our 4-byte fence value 1. The pop then
// misaligned-reads *(0x1000000001) = the adjacent block's pool pointer 0x20015f0000 shifted one byte
// = 0x20015f00 (the "byte-shift" is a READ ARTIFACT, not an off-by-one store — this overturns the
// session-8 theory and unifies all three fatal signatures into ONE root write). The existing
// REL1-LIVE guard (case 1) MISSES this because it requires pre_low != 0; a 0x1000000000-shaped
// pointer has pre_low == 0. forges_freelist_ptr() detects that missed case.
static inline bool forges_freelist_ptr(uint64_t pre, uint64_t width, uint64_t value) {
    if (!ptr_like(pre)) return false;        // dst must currently hold a heap/pool pointer
    if ((uint32_t)pre != 0) return false;    // low dword already nonzero -> the REL1-LIVE guard covers it
    if (width >= 8) return false;            // a full-width write replaces the whole qword (no forge)
    return (uint32_t)value != 0;             // our sub-qword write makes the low dword nonzero -> pre|value
}
// Log-only tripwire (PROSPER_FORGE_TRIP=1): report a GPU write that forges a freelist pointer, with
// the packet builder callsite — deciding host(GPU)-vs-guest for the root write. Default OFF (no-op).
static void forge_trip(const char* kind, uint64_t dst, uint64_t pre, uint64_t value, uint64_t width, uint64_t pkt) {
    static const bool on = getenv("PROSPER_FORGE_TRIP") != nullptr;
    if (!on || !forges_freelist_ptr(pre, width, value)) return;
    static std::atomic<int> n{0};
    if (n.fetch_add(1) < 64)
        fprintf(stderr, "[agc] FORGE-STOMP kind=%s dst=0x%llx pre=0x%llx val=0x%llx -> 0x%llx pkt=eboot+0x%llx t=%llums\n",
                kind, (unsigned long long)dst, (unsigned long long)pre, (unsigned long long)value,
                (unsigned long long)(pre | (value & 0xffffffffull)), (unsigned long long)pkt,
                (unsigned long long)now_ms());
}
// #312 ROOT fix gate (default ON; PROSPER_REL1_FORGE_GUARD=0 for A/B baseline). Suppress a fence
// write that would forge a freelist next-pointer (see forges_freelist_ptr), gated to the consumed-
// marker label population so plain fence labels (Messenger) are untouched.
static bool forge_guard() {
    static const bool v = [] { const char* e = getenv("PROSPER_REL1_FORGE_GUARD");
                               return !e || strtol(e, nullptr, 0) != 0; }();   // default ON
    return v;
}
// #312 label free-state write-after-free guard — see label_freed_marker_kind. Default OFF
// (PROSPER_REL1_WAF_GUARD=1 to arm). RATIONALE (session-11 A/B, ~30 menu-drive runs): this guard is
// STRICTLY-CORRECT — every write it suppresses is a genuine write-after-free into an MB3
// consumed-marker label the guest already freed — but it does NOT close #312, because the DOMINANT
// residual is undetectable from the GPU side: the guest frees the 0x20 label block while BOTH its
// DmaData(:=0) init AND its ReleaseMem(<-1) fence are still in flight, so our init writes 0 to the
// freed block's next-pointer field (making it read exactly like a freshly-init'd LIVE label, low
// dword 0) AND bumps dma_exec_n (making the protocol counters read as a live pending fence). Neither
// content nor tracked counters can separate that freed block from a live label — only the guest's
// actual FREELIST MEMBERSHIP can, and cheaply hooking the MB3 free-push is infeasible (it is far too
// hot to trap without destroying the repro's timing). So the guard ships OFF: the default boot stays
// at the proven #505/#510 state, and this remains armed A/B instrumentation for the freelist-
// membership approach the close ultimately needs. CONFIDENCE: HIGH on the mechanism + why it can't
// be resolved GPU-side (definitive per-write capture, session 10-11).
static bool waf_guard() {
    static const bool v = [] { const char* e = getenv("PROSPER_REL1_WAF_GUARD");
                               return e && strtol(e, nullptr, 0) != 0; }();   // default OFF
    return v;
}
// #312 CLOSE: bounded log of a suppressed write-after-free. PER-CATEGORY caps so a rare non-pointer
// (canary-residual) or Case-B suppression is always visible even after the common pointer case fills.
static void waf_report(const char* kind, uint64_t addr, uint64_t pre, uint64_t value, int cat) {
    static const char* catname[] = {"?", "A-ptr", "A-canary", "B-stale0"};
    static std::atomic<int> nc[4] = {};
    int cap = (cat == 1) ? 32 : 512;   // pointer case is common (#510 covered it); log the rest fully
    if (nc[cat & 3].fetch_add(1) < cap)
        fprintf(stderr, "[agc] REL-WAF-SUPPRESS kind=%s cat=%s [0x%llx] pre=0x%llx value=0x%llx t=%llums\n",
                kind, catname[cat & 3], (unsigned long long)addr, (unsigned long long)pre,
                (unsigned long long)value, (unsigned long long)now_ms());
}
// #312 close: direct, non-trapping membership in the guest allocator's Malloc(0x20) freelists.
// Unlike content/counter inference, this remains truthful before our DmaData init has overwritten a
// freed node's NextFreeBlock with 0.
// #1226: default OFF; PROSPER_MB3_FREELIST_GUARD=1 re-arms for investigation (and is also
// required, together with PROSPER_GENERATION_GUARD=1, to re-arm the DMA-init generation check —
// see generation_guard below). Membership is not proof of staleness: a label the guest freed and
// IMMEDIATELY re-malloc'd sits at the same address, and this walk races the allocator's own
// updates — observed on ArcRunner as 64 suppressed live-protocol writes per run. Partial
// suppression is the worst state (generation off + this on faulted 3/3 EARLY: inits landed while
// their paired fences were membership-suppressed); with the whole family off, pooled figures are
// ArcRunner 2/7 faulting vs 4/7 on, all survivors the pre-existing POOLSHIFT class, and DOLL 0
// fence fatals in 4/4 (full A/B at generation_guard below). The #241 stale-ring source is closed
// by exact submit counts. CONFIDENCE: MED-HIGH.
static bool mb3_freelist_guard() {
    static const bool v = [] { const char* e = getenv("PROSPER_MB3_FREELIST_GUARD");
                               return e && strtol(e, nullptr, 0) != 0; }();   // default OFF (#1226)
    return v;
}
// A consumed-marker label is intentionally uninitialized when its DmaData packet is built. The
// packet-exact snapshot distinguishes that harmless residue from a target whose 0x20-byte block was
// freed and reused before this old packet executed. Content alone cannot do that: a reused block may
// also begin with 0/1, while a live uninitialized label may contain an arbitrary pointer residue.
static bool dma_build_pre_changed(const Pm4Command& c, uint64_t pre, uint64_t* build_pre) {
    if (!c.dd_build_pre_valid || c.dd_build_pre == pre) return false;
    if (build_pre) *build_pre = c.dd_build_pre;
    return true;
}
static void stale_dma_change_report(uint64_t addr, uint64_t build_pre, uint64_t pre, uint64_t pkt) {
    static std::atomic<uint32_t> n{0};
    if (n.fetch_add(1, std::memory_order_relaxed) < 256)
        fprintf(stderr, "[agc] DMA-GENERATION-CHANGED-STALE-SUPPRESS [0x%llx] build-pre=0x%llx "
                        "exec-pre=0x%llx pkt=0x%llx t=%llums\n",
                (unsigned long long)addr, (unsigned long long)build_pre,
                (unsigned long long)pre, (unsigned long long)pkt,
                (unsigned long long)now_ms());
}
// #1226: gate for the build-pre generation checks. Default OFF. Re-arm contract: this env alone
// re-arms only the REL form (stale_release_generation below); the DMA-init form at honor_dma_data
// lives INSIDE the MB3-gated block and additionally requires PROSPER_MB3_FREELIST_GUARD=1 —
// deliberately, because a generation-suppressed init creates dma-free debt that only the MB3
// release leg consumes, and re-arming it alone would recreate the partial-suppression state
// (init suppressed, paired fence lands) that measured WORST in the A/B. The content-stability
// premise ("an owned label's residue cannot change between build and exec, so a change means the
// packet is stale") is FALSE for titles that build command buffers ahead of submit over a
// churning label pool: ArcRunner's deferred/late-flushed inits routinely observe drifted content,
// the suppressed init's debt also kills the paired fence, and every consumer wait on that label
// then times out (a 1 Hz cascade under PROSPER_WAIT_DEFER).
// Live A/B evidence (2026-07-23, both titles on the #1245 build, pooled): ArcRunner 120 s worker
// faults 2/7 with this and the MB3 membership guard off vs 4/7 with them on — and EVERY surviving
// fault in both titles is the pre-existing POOLSHIFT byte-shifted-pool-pointer class, so an
// observed POOLSHIFT fault is NOT a regression of this flip. Partial removal (generation off,
// membership on) faulted 3/3 early. DOLL: 0 free-unrecognized fatals in 4/4 runs off vs fatals
// recurring on (257 generation suppressions in the fatal run). The stale-ring re-execution these
// checks guarded against (#241) is closed at the source by exact submit dword counts on every
// entry point. CONFIDENCE: MED-HIGH.
static bool generation_guard() {
    static const bool v = [] { const char* e = getenv("PROSPER_GENERATION_GUARD");
                               return e && strtol(e, nullptr, 0) != 0; }();   // default OFF (#1226)
    return v;
}
static bool stale_release_generation(const Pm4Command& c, uint64_t pre) {
    if (!generation_guard()) return false;
    if (!c.rel_build_pre_valid || c.rel_data_sel != 1 || c.rel_value != 1 ||
        !label_is_consumed_marker(c.rel_addr)) return false;
    uint64_t initialized = c.rel_build_pre & 0xffffffff00000000ull;
    uint64_t signaled = initialized | 1ull;
    if (pre == initialized || pre == signaled) return false;
    static std::atomic<uint32_t> n{0};
    if (n.fetch_add(1, std::memory_order_relaxed) < 256)
        fprintf(stderr, "[agc] REL-GENERATION-CHANGED-STALE-SUPPRESS [0x%llx] "
                        "build-pre=0x%llx expected-init=0x%llx exec-pre=0x%llx "
                        "pkt=0x%llx t=%llums\n",
                (unsigned long long)c.rel_addr, (unsigned long long)c.rel_build_pre,
                (unsigned long long)initialized, (unsigned long long)pre,
                (unsigned long long)pkt_addr(c), (unsigned long long)now_ms());
    label_hist_rel_free(c.rel_addr, 0);
    return true;
}
static void mb3_freelist_report(const char* kind, uint64_t addr, uint64_t pre,
                                const Mb3FreelistMatch* match, bool debt) {
    static std::atomic<uint32_t> n{0};
    uint32_t i = n.fetch_add(1, std::memory_order_relaxed);
    if (i < 64 || (i & 4095u) == 0)
        fprintf(stderr, "[agc] MB3-FREE-SUPPRESS kind=%s [0x%llx] pre=0x%llx via=%s "
                        "pool=0x%llx list=%u head=0x%llx hops=%u t=%llums\n",
                kind, (unsigned long long)addr, (unsigned long long)pre,
                debt ? "dma-debt" : "membership",
                (unsigned long long)(match ? match->pool_base : 0), match ? match->list : 0,
                (unsigned long long)(match ? match->head : 0), match ? match->hops : 0,
                (unsigned long long)now_ms());
}
// If an init was suppressed while its target was free, its paired ReleaseMem must stay suppressed
// even when the allocator pops/reuses the block between the two packets. Otherwise direct membership
// would correctly turn false but the old generation's fence would corrupt the block's new owner.
static bool mb3_suppress_release(uint64_t addr, uint64_t value, const char* kind) {
    if (!mb3_freelist_guard() || value != 1 || !label_is_consumed_marker(addr)) return false;
    uint64_t pre = peek_qword(addr);
    if (label_hist_take_dma_free_debt(addr)) {
        label_hist_rel_free(addr, 0);
        mb3_freelist_report(kind, addr, pre, nullptr, true);
        return true;
    }
    Mb3FreelistMatch match{};
    if (!mb3_freelist_contains_stable(addr, &match)) return false;
    label_hist_rel_free(addr, match.pool_base);
    mb3_freelist_report(kind, addr, pre, &match, false);
    return true;
}
// #312: report one suspicious fence write with its build-journal verdict. kindtag: "REL1" etc.
//
// RUN-2026-07-10 FINDING (this instrumentation's own history): the historical "~96 pointer-valued
// labels in a burst at t~10 s" was a FALSE-POSITIVE artifact. Per the consumed-marker protocol the
// label legitimately holds a DEAD chain pointer (0x10xxxxxxxx) at build time (the guest flattens
// its pending-label list into an array BEFORE emitting, so the intrusive next pointers are dead);
// our in-stream DmaData(label := 0, 4B) then zeroes the LOW dword, leaving the qword reading
// exactly 0x1000000000 (stale high half) at ReleaseMem time — ptr_like() matched that benign
// composite, and the 10 s gate + 96-line budget made it look like a t~10 s burst (label-history
// proof: dma built==exec, rel built==exec, same fold, every time, in CLEAN runs too). The REAL
// anomaly class is a fence write finding a NONZERO LOW DWORD pointer at write time: the init leg
// did not land before us (missing/misordered) or the block was reused/re-linked (we are late).
static void report_suspect_write(const char* kindtag, uint64_t addr, uint64_t value, uint64_t pre,
                                 uint64_t pkt) {
    // Skip the first 2 s (module-load churn); the protocol is steady-state by then.
    if (now_ms() < 2000) return;
    static std::atomic<int> n{0};
    if (n.fetch_add(1) >= 192) return;
    uint64_t baddr = 0, bpre = 0, bt = 0;
    int have = prosper_fence_journal_lookup(pkt, &baddr, &bpre, &bt);
    char hist[512]; label_hist_report(addr, hist, sizeof hist);
    fprintf(stderr, "[agc] SUSPECT-%s [0x%llx] pre=0x%llx value=0x%llx pkt=0x%llx t=%llums | "
                    "journal:%s built@%llums(age=%lldms) built-addr=0x%llx%s pre@build=0x%llx%s | %s\n",
            kindtag, (unsigned long long)addr, (unsigned long long)pre, (unsigned long long)value,
            (unsigned long long)pkt, (unsigned long long)now_ms(),
            have ? "" : " MISS", (unsigned long long)bt,
            have ? (long long)(now_ms() - bt) : -1,
            (unsigned long long)baddr, (have && baddr != addr) ? " TARGET-CHANGED" : "",
            (unsigned long long)bpre, (have && ptr_like(bpre)) ? " STALE-AT-BUILD" : "", hist);
}

// #312 the WHERE fix (default ON; PROSPER_REL1_STOMP_GUARD=0 restores the old barrel-through for
// A/B). When a data_sel==1 fence write would land on a live MallocBinned3 freelist/pool block —
// captured live as REL1-LIVE: the destination qword is a heap pointer (ptr_like) whose low dword is
// a real pointer half, not 0 (init'd) or 1 (signaled) — the paired DmaData init never ran and the
// guest already freed the recycled label back to the allocator. Writing our 4-byte 1 over +0 forges
// 0x10000000_00000001 (or a 0x2001... pool pointer) — the exact #312 fatal family. Suppress it.
static bool rel1_stomp_guard() {
    static const bool v = [] { const char* e = getenv("PROSPER_REL1_STOMP_GUARD");
                               return !e || strtol(e, nullptr, 0) != 0; }();   // default ON
    return v;
}
// --- #1226 clock-fence provenance: persistent per-address record of 64-bit GPU-clock writes. -----
// ArcRunner's intermittent RHIThread/RenderThread free-list crash dereferences a bin head holding
// {high = GPU-clock-like counter, low = constant 0x00024001} — the shape of a RELEASE_MEM
// data_sel==3 / EVENT_WRITE timestamp write whose low dword the guest later re-initialized. The 16K
// attribution ring above wraps in well under a second on a busy title, which is why the fault-time
// ring scan found no writer (#1226). This table instead RETAINS the last clock write per target
// address for the whole run (direct-mapped by address; collisions replace — diagnostic-grade), so
// the PROSPER_FAULTOBJ worker-fault dump can answer "was the corrupted address EVER a clock-fence
// target, and what did it hold before the write?" long after the ring has wrapped. Always-on: one
// hashed store per clock write, no allocation. This is provenance only — it never gates a write.
namespace {
struct ClockFenceRec {
    uint64_t addr;                     // 0 = empty slot
    uint64_t value, pkt, pre;          // last write: clock value, builder packet, pre-content
    uint64_t first_ms, last_ms;
    uint32_t count;
    uint8_t kind;                      // 1=RELEASE_MEM data_sel==3, 2=EVENT_WRITE timestamp
    uint8_t heapish_pre;               // some write to this address saw pointer-like pre-content
    // Recent-value history: run-2 evidence shows the poison qword is {orig_low32, clock_low32} —
    // an 8-byte clock write at A read back at A-4 — but the stomping write is usually NOT the
    // LAST write to its target (fence labels re-fence every frame), so "last value" alone cannot
    // answer "which target once held clock X?". Keep a tiny per-target ring of {value, t_ms}.
    struct { uint64_t value, t_ms; } hist[4];
    uint32_t hist_next;
};
constexpr uint32_t kClockFenceSlots = 8192;            // power of two
ClockFenceRec g_clock_fences[kClockFenceSlots];
// Diagnostic-only "looks like a heap pointer" — deliberately wider than ptr_like(): ArcRunner's
// MallocBinned3 arena and RHI objects live in the 0x2400000000..0x3200000000 region (#1226 FAULTOBJ
// dumps: objects at 0x2420e48000 / 0x3152b50000 / 0x316366c154), which the DOLL-era ptr_like()
// windows predate. Flags records for the reader; never used to gate or suppress a write.
bool clockfence_heapish(uint64_t v) {
    return ptr_like(v) || (v >= 0x2100000000ull && v < 0x4000000000ull);
}
bool clockfence_log() {
    static const bool v = getenv("PROSPER_CLOCKFENCE_LOG") != nullptr;
    return v;
}
// Slot hash: use the TOP bits of a 64-bit golden-ratio product. Masking the low bits of an odd
// multiply (the label_hist idiom) makes two addresses collide exactly when equal mod 64 KiB —
// and 64 KiB-aligned recycled chunks are precisely the fence-label population under suspicion
// (review of #1239), so that stride would systematically evict the interesting records.
inline uint32_t clockfence_slot(uint64_t addr) {
    return (uint32_t)(((addr >> 3) * 0x9E3779B97F4A7C15ull) >> 51) & (kClockFenceSlots - 1);
}
void clockfence_record(uint64_t addr, uint64_t pre, uint64_t value, uint64_t pkt, uint8_t kind) {
    ClockFenceRec& r = g_clock_fences[clockfence_slot(addr)];
    const uint64_t t = now_ms();
    if (r.addr != addr) {
        r.addr = addr; r.count = 0; r.first_ms = t; r.heapish_pre = 0;
        r.hist_next = 0; memset(r.hist, 0, sizeof r.hist);
    }
    r.value = value; r.pkt = pkt; r.pre = pre; r.last_ms = t; r.kind = kind; r.count++;
    r.hist[r.hist_next & 3] = { value, t }; r.hist_next++;
    if (clockfence_heapish(pre)) {
        r.heapish_pre = 1;
        // The durable form of #1226's one-off REL3 diagnostic: a clock fence overwriting memory
        // that currently holds a pointer — a recycled label, or a live allocator/RHI structure
        // (the crash class). Bounded and off by default (PROSPER_CLOCKFENCE_LOG=1).
        static std::atomic<int> n{0};
        if (clockfence_log() && n.fetch_add(1) < 64)
            fprintf(stderr,
                    "[agc] CLOCKFENCE-OVER-PTR kind=%u [0x%llx] pre=0x%llx clock=0x%llx pkt=0x%llx t=%llums\n",
                    kind, (unsigned long long)addr, (unsigned long long)pre,
                    (unsigned long long)value, (unsigned long long)pkt, (unsigned long long)t);
    }
}
}
// Scan the persistent clock-fence table for writes overlapping [lo, hi); format matches into `out`
// (NUL-terminated). Async-signal-safe: no locks, no allocation, tolerates torn racy slots
// (diagnostic-grade). Called from the PROSPER_FAULTOBJ worker-fault dump (exec_image_linux.cpp) to
// attribute a stomped allocator/RHI field to the exact fence target + builder packet.
extern "C" int prosper_gpu_clockfence_scan(uint64_t lo, uint64_t hi, char* out, size_t cap) {
    size_t off = 0; int found = 0;
    for (uint32_t i = 0; i < kClockFenceSlots && off + 208 < cap; i++) {
        const ClockFenceRec& r = g_clock_fences[i];
        if (!r.addr || r.addr + 8 <= lo || r.addr >= hi) continue;
        int m = snprintf(out + off, cap - off,
                         "[clockfence] addr=0x%llx kind=%u count=%u last=0x%llx pre=0x%llx "
                         "heapish_pre=%u pkt=0x%llx t=%llu..%llums\n",
                         (unsigned long long)r.addr, r.kind, r.count, (unsigned long long)r.value,
                         (unsigned long long)r.pre, r.heapish_pre, (unsigned long long)r.pkt,
                         (unsigned long long)r.first_ms, (unsigned long long)r.last_ms);
        if (m > 0) off += (size_t)m;
        found++;
    }
    if (off < cap) out[off] = 0;
    return found;
}

// #1226 run-2 evidence: the corrupted bin head reads {high=0x0B782E3D, low=0x00024001} while NO
// clock fence ever targeted its page — but 0x0B782E3D is exactly the LOW 32 bits of the GPU clock
// at ~69-73s (16-17 2^32-ns wraps), minutes into the run and just before the ~76s fault. That is
// the signature of an 8-byte clock write at some OTHER address A whose bytes, read as the qword at
// A-4, yield {orig_low32, clock_low32} — poison the guest then COPIES into the bin head via a
// free-list pop. This finder answers, at fault time: which fence target ever wrote a clock whose
// low32 matches the corrupted value's high dword? EXACT hits match a retained value's low 32 bits bit-for-bit;
// NEAR hits (within ~134ms of clock) catch a target re-fenced shortly after the stomp. A freed
// label stops being re-fenced, so the poison clock tends to survive as its last/history value.
extern "C" int prosper_gpu_clockfence_find_low32(uint32_t low32, char* out, size_t cap) {
    size_t off = 0; int found = 0;
    for (uint32_t i = 0; i < kClockFenceSlots && off + 208 < cap; i++) {
        const ClockFenceRec& r = g_clock_fences[i];
        if (!r.addr) continue;
        const char* how = nullptr; uint64_t match_v = 0, match_t = 0;
        uint64_t cand[5]; uint64_t cand_t[5]; int nc = 0;
        cand[nc] = r.value; cand_t[nc++] = r.last_ms;
        for (int h = 0; h < 4; h++) if (r.hist[h].value) { cand[nc] = r.hist[h].value; cand_t[nc++] = r.hist[h].t_ms; }
        // EXACT anywhere beats NEAR anywhere: EXACT vs NEAR is the evidence grade this tool
        // exists to produce, so a NEAR-matching last value must not shadow an EXACT history hit.
        for (int k = 0; k < nc && !how; k++)
            if ((uint32_t)cand[k] == low32) { how = "EXACT"; match_v = cand[k]; match_t = cand_t[k]; }
        for (int k = 0; k < nc && !how; k++) {
            uint32_t vl = (uint32_t)cand[k];
            if ((uint32_t)(vl - low32) < 0x08000000u || (uint32_t)(low32 - vl) < 0x08000000u) {
                how = "NEAR"; match_v = cand[k]; match_t = cand_t[k];
            }
        }
        if (!how) continue;
        int m = snprintf(out + off, cap - off,
                         "[clockfence-find] %s addr=0x%llx kind=%u count=%u v=0x%llx@%llums "
                         "last=0x%llx pre=0x%llx heapish_pre=%u pkt=0x%llx\n",
                         how, (unsigned long long)r.addr, r.kind, r.count,
                         (unsigned long long)match_v, (unsigned long long)match_t,
                         (unsigned long long)r.value, (unsigned long long)r.pre, r.heapish_pre,
                         (unsigned long long)r.pkt);
        if (m > 0) off += (size_t)m;
        found++;
    }
    if (off < cap) out[off] = 0;
    return found;
}

// Honor a RELEASE_MEM / EVENT_WRITE_EOP completion write. data_sel (Kyty GraphicsCbReleaseMem allows {2,3};
// shadPS4 DataSelect enum): 1=write 32-bit value, 2=write 64-bit value, 3=write 64-bit GPU clock. The write
// uses memcpy so an only-4-byte-aligned 64-bit label is handled portably. CONFIDENCE: HIGH — address,
// data_sel and value are decoded directly from the packet the game's ReleaseMem call built.
static void honor_eop_write(const Pm4Command& c) {
    if (eop_writes_disabled() || !c.rel_addr || (c.rel_addr & 3)) return;
    // #729: the synchronous path (PROSPER_EOP_WRITE_SYNC=1) reaches here without
    // apply_deferred_effect's #449 guard, and the #312 pre-reads below read 8 bytes regardless of
    // the write size — probe mappedness before any dereference of the guest-PM4-supplied address.
    // Skipping matches what the deferred path does for an unmapped label.
    if (!guest_readable(c.rel_addr, 8)) {
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 24)
            fprintf(stderr, "[agc] RELEASE_MEM label unmapped — write SKIPPED: addr=0x%llx\n",
                    (unsigned long long)c.rel_addr);
        return;
    }
    void* dst = (void*)(uintptr_t)c.rel_addr;
    switch (c.rel_data_sel) {
        // data_sel==0 is "interrupt only, NO data write" (PM4 spec) — writing anyway clobbers 8 bytes
        // at a live label address (and a mis-extraction that yields 0 has a garbage value dword too,
        // so skipping is right in both readings). CONFIDENCE: MED.
        case 0: return;
        // Cases 1/2 need the same rel_value_valid guard the default case got: a short-decoded
        // packet's rel_value is a fabricated 0 that could move a satisfied fence label BACKWARDS
        // (re-blocking a `*label >= expected` poll).
        case 1: { if (!c.rel_value_valid) return;
                  if (mb3_suppress_release(c.rel_addr, c.rel_value, "REL1")) return;
                  // #312 stomp-catcher: a live fence label holds small ints/timestamps; a freed
                  // MallocBinned3 FFreeBlock header holds heap POINTERS. Pointer-like pre-content
                  // means this fence write is landing in freed (or reused) memory — log it in the
                  // act, with the packet address + the build-journal verdict (timing vs wrong-
                  // target). A fence TARGET inside the allocator-metadata region (0x20xxxxxxxx,
                  // the FPoolInfo tables) is suspect regardless of content. Diagnostic, bounded.
                  // (Run-1 finding: legit fence labels DO live in the 0x20xxxxxxxx region — a
                  // target-region trigger caught only benign fences and burned the report cap.
                  // Trigger on pointer-like PRE-CONTENT only.)
                  uint64_t pre = 0; memcpy(&pre, dst, sizeof pre);
                  if (stale_release_generation(c, pre)) return;
                  // #312 — label free-state write-after-free guard (default OFF; A/B instrumentation,
                  // see waf_guard). Keys on tracked lifecycle state; suppresses genuine WAFs but does
                  // NOT close the gate (the dominant residual is GPU-side-undetectable — see waf_guard).
                  // Records NO rel_exec (a suppressed fence must not advance the signal counter, or
                  // Case B would mis-count the next generation).
                  if (int cat; waf_guard() && label_is_consumed_marker(c.rel_addr) &&
                      (cat = label_freed_marker_kind(c.rel_addr, pre, c.rel_value)) != 0) {
                      waf_report("REL1", c.rel_addr, pre, c.rel_value, cat);
                      return;
                  }
                  // Post-init state (low dword 0 or an already-signaled 1 from OUR OWN just-landed
                  // write... no: low==1 means the previous generation's fence value survived with
                  // NO re-init between — the init leg missed. Classify (see the finding above):
                  //   low==0            -> benign post-init composite; not reported.
                  //   ptr_like, low!=0  -> REL1-LIVE: a real pointer at write time (stomp-in-the-act).
                  //   low==1 && high!=0 -> REL1-NOINIT: recycled label re-fenced without its DmaData.
                  uint32_t pre_low = (uint32_t)pre;
                  if (pre_low != 0) {
                      if (ptr_like(pre) && pre_low != 1) {
                          report_suspect_write("REL1-LIVE", c.rel_addr, c.rel_value, pre, pkt_addr(c));
                          // #312 FIX (the WHERE leg): dst holds a live MallocBinned3 freelist/pool
                          // next-pointer, not an initialized consumed-marker label — its paired
                          // DmaData(:=0) init never landed (a valid label reads low-dword 0 here).
                          // The guest already freed this recycled 0x20 block, so no WaitRegMem==1
                          // consumer can be satisfied by it anyway (the pointer's low dword != 1);
                          // performing the 4-byte value-1 write would forge 0x10000000_00000001 and
                          // corrupt the freelist — the exact #312 fatal. Suppress it. Gated to the
                          // consumed-marker population so plain fence labels (Messenger) are never
                          // affected. CONFIDENCE: HIGH (mechanism captured live, sessions 2-5).
                          if (rel1_stomp_guard() && label_is_consumed_marker(c.rel_addr)) {
                              // Keep the event visible without consuming an init generation. A
                              // skipped fence did not signal anything; advancing rel_exec_n here
                              // makes the next real init/fence pair look stale and suppresses it.
                              label_hist_event(c.rel_addr, LE_REL_EXEC, pre, c.queue_origin);
                              return;
                          }
                      }
                      else if (pre_low == 1 && (pre >> 32))
                          report_suspect_write("REL1-NOINIT", c.rel_addr, c.rel_value, pre, pkt_addr(c));
                  }
                  // #312 ROOT (session-10): pre is a freelist next-pointer with a ZERO low dword
                  // (0x1000000000-shaped). The pre_low!=0 branch above never sees it, so the value-1
                  // write below would forge pre|1 (0x1000000001) and seed the crash.
                  //
                  // #1226 CORRECTION to session-10's "this is always a write-after-free" claim: it is
                  // NOT. The guest's DmaData init writes only 4 BYTES, so a LIVE, correctly init'd
                  // label whose 0x20-byte malloc residue had nonzero HIGH bits reads exactly
                  // {stale_high, low=0} — byte-identical to the freed-block shape. Observed live on
                  // ArcRunner (PPSA21406): residue 0x2020e31680, 4-byte init -> qword 0x2000000000,
                  // and this guard then suppressed the guest's own paired fence, leaving every
                  // consumer WaitRegMem dependency-violated (thousands/min) and desynchronizing the
                  // fence-gated free protocol — the very corruption the guard exists to prevent.
                  // Content cannot discriminate the two; PROTOCOL STATE can (same model as
                  // label_freed_marker_kind Case B): an EXECUTED init not yet consumed by a fence
                  // (dma_exec_n > rel_exec_n) means this rel is the paired fence of a live
                  // generation — real hardware performs the 32-bit write (the consumer polls only
                  // the low dword; the stale high half is invisible to it). Suppress only when no
                  // executed init is outstanding (the true stale/WAF fence). CONFIDENCE: MED-HIGH
                  // (live ArcRunner protocol trace + the WAF model's documented lifecycle).
                  else if (forges_freelist_ptr(pre, 4, c.rel_value)) {
                      forge_trip("REL1", c.rel_addr, pre, c.rel_value, 4, pkt_addr(c));
                      LabelHist& fh = label_hist_slot(c.rel_addr);
                      const bool live_pair =
                          fh.dma_exec_n.load(std::memory_order_relaxed) >
                          fh.rel_exec_n.load(std::memory_order_relaxed);
                      if (forge_guard() && label_is_consumed_marker(c.rel_addr) && !live_pair) {
                          report_suspect_write("REL1-FORGE", c.rel_addr, c.rel_value, pre, pkt_addr(c));
                          // Ring visibility WITHOUT the rel_exec_n bump: a suppressed fence must not
                          // advance the consumed-count, or the NEXT generation's live check above
                          // would read dma_exec_n == rel_exec_n and wrongly suppress a real fence.
                          label_hist_event(c.rel_addr, LE_REL_EXEC, pre, c.queue_origin);
                          return;
                      }
                  }
                  // In-flight overlap: a second init+fence pair to this label is already built but
                  // not executed — two fence generations in the pipe together (see label_rel_overlap).
                  // Content-independent: catches the late-pair stomp class even when pre reads 0.
                  if (int ov = label_rel_overlap(c.rel_addr); ov >= 1) {
                      static std::atomic<int> novl{0};
                      if (now_ms() >= 2000 && novl.fetch_add(1) < 64) {
                          char hist[512]; label_hist_report(c.rel_addr, hist, sizeof hist);
                          fprintf(stderr, "[agc] SUSPECT-REL1-OVERLAP [0x%llx] pending-inits=%d pre=0x%llx t=%llums | %s\n",
                                  (unsigned long long)c.rel_addr, ov, (unsigned long long)pre,
                                  (unsigned long long)now_ms(), hist);
                      }
                  }
                  uint32_t v = (uint32_t)c.rel_value; memcpy(dst, &v, sizeof v);
                  ring_record(c.rel_addr, v, 4, 1, pkt_addr(c));
                  poolshift_check("REL1", c.rel_addr, 4, c.rel_value, pkt_addr(c));
                  label_hist_rel_exec(c.rel_addr, pre, c.queue_origin); break; }
        case 2: { if (!c.rel_value_valid) return;
                  if (mb3_suppress_release(c.rel_addr, c.rel_value, "REL2")) return;
                  // #312: the same freelist-stomp guard as the 32-bit path. An 8-byte value-1 fence
                  // over a live consumed-marker freelist node overwrites BOTH the next-pointer dwords
                  // (observed live: 8-byte kind=1 writes to a hot recycled label preceding a canary
                  // fatal). Skip when the target still holds a live heap pointer (its DmaData init
                  // never landed) — see honor case 1. CONFIDENCE: HIGH.
                  uint64_t pre = 0; memcpy(&pre, dst, sizeof pre);
                  // #312 CLOSE — same label free-state WAF guard as case 1 (8-byte fence leg).
                  if (int cat; waf_guard() && label_is_consumed_marker(c.rel_addr) &&
                      (cat = label_freed_marker_kind(c.rel_addr, pre, c.rel_value)) != 0) {
                      waf_report("REL2", c.rel_addr, pre, c.rel_value, cat);
                      return;
                  }
                  if (rel1_stomp_guard() && ptr_like(pre) && (uint32_t)pre != 0 &&
                      pre != c.rel_value && label_is_consumed_marker(c.rel_addr)) {
                      report_suspect_write("REL2-LIVE", c.rel_addr, c.rel_value, pre, pkt_addr(c));
                      // A suppressed write belongs in the diagnostic ring, but it must not advance
                      // the completed-fence counter (same protocol rule as REL1-LIVE above).
                      label_hist_event(c.rel_addr, LE_REL_EXEC, pre, c.queue_origin);
                      return;
                  }
                  uint64_t v = c.rel_value;           memcpy(dst, &v, sizeof v);
                  ring_record(c.rel_addr, v, 8, 1, pkt_addr(c));
                  poolshift_check("REL2", c.rel_addr, 8, c.rel_value, pkt_addr(c)); break; }
        case 3: { uint64_t pre = 0; memcpy(&pre, dst, sizeof pre);   // #1226 provenance pre-read
                  uint64_t v = gpu_clock64();         memcpy(dst, &v, sizeof v);
                  ring_record(c.rel_addr, v, 8, 1, pkt_addr(c));
                  clockfence_record(c.rel_addr, pre, v, pkt_addr(c), 1); break; }
        // Unknown selector: LOG AND SKIP. The old default wrote the 64-bit value for ANY
        // unrecognized data_sel — a band-aid for the swap-stub stack-arg mis-extraction that made
        // data_sel arrive as a pointer (fixed in exec_image_linux.cpp emit_swap_stub: handlers now
        // see real stack args, verified live with data_sel=0x2/0x3). With the root cause gone, an
        // unknown selector means a genuinely unexpected packet: writing 8 bytes on a guess could
        // clobber the dword after a 32-bit label. Log so the gap is visible, never write.
        default:
            fprintf(stderr, "[agc] RELEASE_MEM: unknown data_sel=%u addr=0x%llx value=0x%llx — write SKIPPED\n",
                    c.rel_data_sel, (unsigned long long)c.rel_addr, (unsigned long long)c.rel_value);
            return;
    }
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   EOP write [0x%llx] data_sel=%u value=0x%llx\n",
                (unsigned long long)c.rel_addr, c.rel_data_sel, (unsigned long long)c.rel_value);
    notify_guest_gpu_write(c.rel_addr, c.rel_data_sel == 1 ? 4 : 8);
    wake_on_label(c.rel_addr);   // wake any sync_on_address futex waiter on this completion label
}

// Honor an address-carrying EVENT_WRITE (#132): the timestamp/label variant writes a completion
// value to its address. Our GPU folds synchronously (submit == pipe drain), so by the time we
// process this packet the event has "happened" — write a monotonic GPU clock (the value a timestamp
// event carries) and wake any waiter, resolving the "a guest waiting on the label blocks forever"
// case. Address-less events (event_addr == 0, the pipeline-sync variants: partial-flush, cache
// inval) stay no-ops. CONFIDENCE: LOW on the value for the counter-sample event types
// (ZPASS_DONE / streamout stats read a counter, not a timestamp) — but a defined monotonic write is
// strictly better than the old discard (no write at all, which is what blocked the waiter). No title
// currently exercises this (the Messenger fences via ReleaseMem/WriteData), so it's latent.
static void honor_event_write(const Pm4Command& c) {
    if (eop_writes_disabled() || !c.event_addr || (c.event_addr & 3)) return;
    if (!guest_readable(c.event_addr, 8)) {   // #729: sync path lacks the #449 deferred guard
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 24)
            fprintf(stderr, "[agc] EVENT_WRITE label unmapped — write SKIPPED: addr=0x%llx\n",
                    (unsigned long long)c.event_addr);
        return;
    }
    uint64_t pre = 0; memcpy(&pre, (void*)(uintptr_t)c.event_addr, sizeof pre);   // #1226 provenance
    uint64_t v = gpu_clock64();
    memcpy((void*)(uintptr_t)c.event_addr, &v, sizeof v);
    notify_guest_gpu_write(c.event_addr, sizeof v);
    ring_record(c.event_addr, v, 8, 2, pkt_addr(c));
    clockfence_record(c.event_addr, pre, v, pkt_addr(c), 2);
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   EventWrite [0x%llx] event_type=%u -> clock 0x%llx\n",
                (unsigned long long)c.event_addr, c.event_type, (unsigned long long)v);
    wake_on_label(c.event_addr);   // wake any sync_on_address futex waiter on this completion label
}

// Honor a DMA_DATA packet's memory effect (issue #312 — the MallocBinned3 heap-corruption root
// cause). DOLL's RHIThread emits DmaData(srcOrImm=0, dst=<per-chunk fence label>, 4 bytes) per
// translated segment: the GPU-side INIT (label := 0) of the consumed-marker protocol whose
// completion leg is ReleaseMem(label <- 1). We previously DROPPED every DMA_DATA packet, so the
// LIFO-recycled label kept the previous generation's 1 and the guest's consumption poll freed the
// label while fences to it were still in flight (full evidence chain: hle_agc agc_dcb_dma_data).
// DOLL uses two immediate-fill instances: the 4-byte per-segment label init above, and 64 KiB
// zero-fills of freshly allocated 64 KiB-aligned command-stream chunks. Issue #189 completes the
// address-backed sibling: a source above the 32-bit immediate domain is copied only when the whole
// source and destination spans are mapped. Raw selector arguments do not distinguish the forms:
// the title's captured immediate-zero call passes 3/3, values which overlap the memory/L2 selector
// vocabulary. Source mappedness plus the established 32-bit immediate ABI is the safe discriminator;
// GDS offsets and malformed/unmapped endpoints remain fail-closed.
// CONFIDENCE: HIGH on immediate fill and mapped address-copy behavior; MED on the large-fill form.
enum class DmaDataForm { Invalid, Immediate, Copy };

static DmaDataForm dma_data_form(const Pm4Command& c, bool source_materialized = false) {
    constexpr uint32_t kMaxImmediateBytes = 0x1000000;   // existing 16 MiB fill safety bound
    constexpr uint32_t kMaxCopyBytes = 0x10000000;      // HLE builder's 256 MiB API bound
    if (!c.dd_valid || !c.dd_bytes || c.dd_bytes > kMaxCopyBytes || c.dd_dst < 0x10000)
        return DmaDataForm::Invalid;
    // Preserve the established ABI discriminator: every <=32-bit source is immediate data. Guest
    // image/heap addresses in prosper are 64-bit, so address copies occupy the other domain.
    if (c.dd_src <= UINT32_MAX) {
        // The immediate path peeks one qword for the consumed-marker safety journal.
        const uint32_t probe_bytes = c.dd_bytes < 8 ? 8 : c.dd_bytes;
        return c.dd_bytes <= kMaxImmediateBytes && !(c.dd_dst & 3) &&
                       guest_readable(c.dd_dst, probe_bytes)
                   ? DmaDataForm::Immediate
                   : DmaDataForm::Invalid;
    }
    return (source_materialized || guest_readable(c.dd_src, c.dd_bytes)) &&
                   guest_writable(c.dd_dst, c.dd_bytes)
               ? DmaDataForm::Copy
               : DmaDataForm::Invalid;
}

static void report_invalid_dma_data(const Pm4Command& c) {
    // A skipped immediate init leaves the label pointer-valued. Address-copy failures have no
    // consumed-marker protocol leg and must not alter those lifecycle counters.
    if (c.dd_src <= UINT32_MAX) label_hist_dma_skip(c.dd_dst);
    static std::atomic<int> n{0};
    if (n.fetch_add(1) < 24)
        fprintf(stderr, "[agc] DMA_DATA not executed (invalid/unmapped form): dst=0x%llx src=0x%llx bytes=%u sels=0x%x\n",
                (unsigned long long)c.dd_dst, (unsigned long long)c.dd_src, c.dd_bytes, c.dd_sels);
}

static void honor_dma_data(const Pm4Command& c, uint64_t retained_packet_addr = 0,
                           const uint8_t* authoritative_source = nullptr) {
    if (eop_writes_disabled() || !c.dd_valid) return;
    const uint64_t packet_addr = retained_packet_addr ? retained_packet_addr : pkt_addr(c);
    const DmaDataForm form = dma_data_form(c, authoritative_source != nullptr);
    // #1124 (gated): trace every DMA that targets a specific guest region — used to find whether a
    // texture's backing is (or should be) written by a CP-DMA the fold might drop/mis-form.
    if (const char* w = getenv("PROSPER_DMA_WATCH_DST")) {
        uint64_t lo = strtoull(w, nullptr, 0), hi = lo + 0x1000;
        if (c.dd_dst >= lo - 0x1000 && c.dd_dst < hi)
            fprintf(stderr, "[dma-watch] dst=0x%llx src=0x%llx bytes=%u sels=0x%x form=%d\n",
                    (unsigned long long)c.dd_dst, (unsigned long long)c.dd_src, c.dd_bytes,
                    c.dd_sels, (int)form);
    }
    if (form == DmaDataForm::Invalid) {
        report_invalid_dma_data(c);
        return;
    }
    uint8_t* dst = (uint8_t*)(uintptr_t)c.dd_dst;
    if (form == DmaDataForm::Immediate) {
        const uint32_t v32 = (uint32_t)c.dd_src;
        uint64_t pre_dma = peek_qword(c.dd_dst);   // #312: generation/membership pre-content
        // The exact consumed-marker initializer is a 4-byte immediate zero. Test allocator
        // membership BEFORE touching it: memset(0) would erase a free node's NextFreeBlock.
        if (mb3_freelist_guard() && c.dd_bytes == 4 && v32 == 0 &&
            label_is_consumed_marker(c.dd_dst)) {
            uint64_t build_pre = 0;
            bool generation_changed = generation_guard() && dma_build_pre_changed(c, pre_dma, &build_pre);
            if (generation_changed) {
                label_hist_dma_free(c.dd_dst, 0);
                stale_dma_change_report(c.dd_dst, build_pre, pre_dma, packet_addr);
                return;
            }
            Mb3FreelistMatch match{};
            if (mb3_freelist_contains_stable(c.dd_dst, &match)) {
                label_hist_dma_free(c.dd_dst, match.pool_base);
                mb3_freelist_report("DMA", c.dd_dst, pre_dma, &match, false);
                return;
            }
        }
        forge_trip("DMA", c.dd_dst, pre_dma, v32, 4, packet_addr);
        if (v32 == 0) {
            memset(dst, 0, c.dd_bytes);
        } else {
            uint32_t i = 0;
            for (; i + 4 <= c.dd_bytes; i += 4) memcpy(dst + i, &v32, 4);
            if (i < c.dd_bytes) memcpy(dst + i, &v32, c.dd_bytes - i);
        }
        poolshift_check("DMA", c.dd_dst, c.dd_bytes, c.dd_src, packet_addr);
        if (c.dd_bytes <= 8) label_hist_dma_exec(c.dd_dst, pre_dma, c.queue_origin);
        if (getenv("PROSPER_GFXLOG"))
            fprintf(stderr, "[agc]   DmaData fill [0x%llx] := 0x%x (%u bytes)\n",
                    (unsigned long long)c.dd_dst, v32, c.dd_bytes);
    } else {
        // memmove is byte-for-byte memcpy behavior for the normal non-overlapping GPU-buffer case,
        // while remaining deterministic if an unusual packet overlaps its endpoints.
        memmove(dst, authoritative_source ? authoritative_source
                                          : (const uint8_t*)(uintptr_t)c.dd_src,
                c.dd_bytes);
        if (getenv("PROSPER_GFXLOG"))
            fprintf(stderr, "[agc]   DmaData copy [0x%llx] <- [0x%llx] (%u bytes)\n",
                    (unsigned long long)c.dd_dst, (unsigned long long)c.dd_src, c.dd_bytes);
    }
    notify_guest_gpu_write(c.dd_dst, c.dd_bytes);
    ring_record(c.dd_dst, c.dd_src, (uint8_t)(c.dd_bytes > 255 ? 255 : c.dd_bytes), 4, packet_addr);
    if (writer_provenance_enabled() && c.dd_bytes >= 256)
        record_guest_write(GuestWriterKind::DmaData, c.dd_dst, c.dd_bytes,
                           0, 0, c.stream_order, packet_addr);
    wake_on_label(c.dd_dst);
}

bool execute_ordered_dma_copy(const GpuState::DmaCopy& copy,
                              const uint8_t* authoritative_source) {
    Pm4Command c{};
    c.kind = Pm4Command::Kind::DmaData;
    c.dd_dst = copy.dst;
    c.dd_src = copy.src;
    c.dd_bytes = copy.bytes;
    c.dd_sels = copy.sels;
    c.dd_valid = true;
    c.stream_order = copy.command_order;
    const bool executable = !eop_writes_disabled() &&
        dma_data_form(c, authoritative_source != nullptr) != DmaDataForm::Invalid;
    honor_dma_data(c, copy.packet_addr, authoritative_source);
    return executable;
}

// Honor a WRITE_DATA packet: copy the inline dwords to the destination address (same synchronous timing).
static void honor_write_data(const Pm4Command& c) {
    if (eop_writes_disabled() || !c.wd_addr || (c.wd_addr & 3) || !c.wd_data || !c.wd_num) return;
    // #729: guard both the payload span and the 8-byte #312 pre-read below (a 4-byte label write
    // still pre-reads one qword). The deferred path's #449 guard only covers the payload size.
    uint32_t wbytes = c.wd_num * 4;
    if (!guest_readable(c.wd_addr, wbytes > 8 ? wbytes : 8)) {
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 24)
            fprintf(stderr, "[agc] WRITE_DATA target unmapped — write SKIPPED: addr=0x%llx dwords=%u\n",
                    (unsigned long long)c.wd_addr, c.wd_num);
        return;
    }
    // #312 stomp-catcher (same as the ReleaseMem one): a small label-init WriteData landing over
    // pointer-like memory, or targeting the allocator-metadata region, is a suspect stomp.
    if (c.wd_num <= 4) {
        uint64_t pre = 0; memcpy(&pre, (const void*)(uintptr_t)c.wd_addr, sizeof pre);
        if (ptr_like(pre))
            report_suspect_write("WDATA", c.wd_addr, c.wd_data[0], pre, pkt_addr(c));
        forge_trip("WDATA", c.wd_addr, pre, c.wd_data[0], 4, pkt_addr(c));   // #312 session-10 tripwire
    }
    memcpy((void*)(uintptr_t)c.wd_addr, c.wd_data, (size_t)c.wd_num * 4);
    notify_guest_gpu_write(c.wd_addr, static_cast<uint64_t>(c.wd_num) * 4);
    ring_record(c.wd_addr, c.wd_data[0], (uint8_t)(c.wd_num * 4 > 255 ? 255 : c.wd_num * 4), 3, pkt_addr(c));
    poolshift_check("WDATA", c.wd_addr, (uint64_t)c.wd_num * 4, c.wd_num ? c.wd_data[0] : 0, pkt_addr(c));
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   WriteData [0x%llx] %u dwords (first=0x%08x)\n",
                (unsigned long long)c.wd_addr, c.wd_num, c.wd_data[0]);
    if (writer_provenance_enabled() && c.wd_num >= 64)
        record_guest_write(GuestWriterKind::WriteData, c.wd_addr,
                           static_cast<uint64_t>(c.wd_num) * 4, 0, 0,
                           c.stream_order, pkt_addr(c));
    wake_on_label(c.wd_addr);   // wake any sync_on_address futex waiter on this written label
}

// --- Deferred completion writes: the pipe-drain model for fence/label writes (issue #312). ------
//
// On real hardware NO completion side-effect of a submit — fence label writes, EVENT_WRITE
// timestamps, WRITE_DATA fence values, the flip — is observable until after the submit call has
// returned (the GPU only sees the Dcb when the driver rings the doorbell at the end of the
// submit) plus the pipe-drain latency. prosper's synchronous fold performed these writes INSIDE
// the submit call. The EOP *event* was already deferred for exactly this reason (#232/#241:
// hle_kernel_time.cpp's 1 ms FIFO worker — DOLL's AgcInterrupt->AgcCleanup chain observed frame N
// complete before the AgcSubmissionThread finished its own post-submit bookkeeping and the
// cleanup raced the submitter's retired-allocation list: the SAME "MallocBinned3 Corruption
// Canary was 0x3, should be 0x1" fatal). But DOLL's cleanup ALSO polls the fence LABELS, which
// still became visible mid-submit — under the menu-driven content-load burst (#312) that lets the
// game retire+free GPU-tracking heap blocks while cbs referencing them are still being submitted,
// and our later label writes stomp MallocBinned3 free-block headers (live-attributed: the GPU
// write-ring shows our RELEASE_MEM value-1 writes at exactly the corrupted qword).
//
// Model: honor_* enqueue the write; a FIFO worker applies them in submission order after a 1 ms
// modeled pipe-drain latency (same constant as the EOP-event worker). Synchronous drain points
// keep every intra-model data dependency exact:
//   - WaitRegMem fold checks drain first (a prior submit's fence must be visible to its consumer),
//   - execute_and_present's callers drain first (the renderer reads WRITE_DATA-uploaded memory),
//   - the EOP-event worker drains before posting (an event must never overtake its data writes).
// PROSPER_EOP_WRITE_SYNC=1 restores the old synchronous writes (A/B lever + fallback).
// CONFIDENCE: HIGH on the invariant (completion is post-submit by construction on real HW; Kyty
// writes fences from its GPU thread, never inside the submit call). The cross-queue wait ordering
// is handled by the WAIT_REG_MEM barrier model below (opt-in, PROSPER_WAIT_DEFER=1).
namespace {
std::atomic<bool> g_post_submit_visibility{false};

bool post_submit_visibility_enabled() {
    return g_post_submit_visibility.load(std::memory_order_acquire);
}

bool eop_write_sync() {
    static const bool v = [] { const char* e = getenv("PROSPER_EOP_WRITE_SYNC");
                               return e && strtol(e, nullptr, 0) != 0; }();
    return v;
}
struct PendWrite {
    Pm4Command cmd;
    std::vector<uint32_t> wd_copy;     // owns a WriteData payload (cmd.wd_data repointed here)
};
// IMMORTAL (leaked) worker state — same pattern as the EOP-event worker (hle_kernel_time.cpp):
// the worker is detached and outlives main; static destructors must never run for it.
struct PendQueue {
    std::mutex mx;
    std::condition_variable cv;
    std::deque<PendWrite> q;
    bool worker_started = false;
    int  inflight = 0;    // items popped but whose write hasn't landed yet (see drain)
    int  active_submits = 0; // fence writes stay private until the import return checkpoint
    std::chrono::steady_clock::time_point release_after{}; // modeled GPU latency after that checkpoint
};
PendQueue& pend_q() { static PendQueue* p = new PendQueue; return *p; }
// A return hook is attached to the import NID, so it also runs when that handler rejects its
// arguments before opening a submit scope. Track scopes on the calling thread as well as globally:
// only the synchronous import call that began a scope may retire it at its return checkpoint.
thread_local uint32_t t_submit_scope_depth = 0;
void apply_effect(const Pm4Command& c);   // fwd (defined with the WAIT_DEFER machinery below)
void apply_deferred_effect(const Pm4Command& c);   // fwd: guarded apply (#449)
// Drain returns only when every pending write has LANDED, and writes land STRICTLY IN QUEUE ORDER.
//
// #312 ROOT CAUSE (2026-07-10, label-event-ring attribution): the previous loop popped the NEXT
// item while another drainer was still mid-apply on the PREVIOUS one (it only waited when the
// queue was already empty). The pend queue is drained concurrently by the pend worker, the EOP-
// event worker, and the submit thread's WaitRegMem/renderer drains — so the guest's paired
//   DmaData(label := 0)  ->  ReleaseMem(label <- 1)
// consumed-marker writes (adjacent, same address) were routinely applied by TWO threads in
// PARALLEL with no ordering: the fence 1 became guest-visible before/interleaved-with its own
// init (captured live: SUSPECT-REL1-LIVE with the paired dmaX in the same millisecond reading the
// SAME pre-content, or the dma still queued). The guest's completion poll then freed + reused the
// 0x20-byte label block while our other write was still in flight, and the late 4-byte write
// landed in the block's NEXT OWNER — the MallocBinned3 "Canary was 0x3" / "free an unrecognized
// block 0x1000000001" / pool-metadata corruption family. Fix: never begin item k+1 until item k
// has LANDED — one write in flight globally, FIFO order == landing order. The writes are 4/8-byte
// memcpys; serializing them costs nothing measurable. CONFIDENCE: HIGH (mechanism captured live;
// the suspect class vanishes with this fix).
void pend_drain_locked(PendQueue& p, std::unique_lock<std::mutex>& lk) {
    for (;;) {
        if (p.inflight > 0) {            // another drainer is mid-apply: WAIT — never overtake it
            p.cv.wait(lk);
            continue;
        }
        if (p.q.empty()) break;
        PendWrite w = std::move(p.q.front());
        p.q.pop_front();
        p.inflight++;
        lk.unlock();                     // the write itself never needs the queue lock
        // Guard the target's mappedness (#449/#483): the pend queue applies completion writes ~1 ms
        // after enqueue (the pipe-drain window), during which the guest may have freed+decommitted
        // the label page (MallocBinned3, #312). apply_deferred_effect probes guest_readable before
        // the raw memcpy — without it an unmapped label SIGSEGVs here, exactly the case the deferred-
        // stream path already survives (this pend path releases asynchronously too, so it needs it).
        apply_deferred_effect(w.cmd);
        lk.lock();
        p.inflight--;
        p.cv.notify_all();               // wake both drain waiters and the pend worker
    }
}
// The HLE import trampoline owns scope_end(), so active_submits cannot reach zero until the submit
// handler has returned and the trampoline is at its guest-return checkpoint. The deadline below is
// only modeled GPU latency after that real boundary; correctness no longer depends on a timer being
// long enough to cover guest-side bookkeeping. A new submit may begin during the latency window; in
// that case wait for its later checkpoint/deadline instead.
void pend_wait_post_submit(PendQueue& p, std::unique_lock<std::mutex>& lk) {
    for (;;) {
        p.cv.wait(lk, [&] { return p.active_submits == 0; });
        const auto deadline = p.release_after;
        if (std::chrono::steady_clock::now() >= deadline) return;
        p.cv.wait_until(lk, deadline);
    }
}
void pend_worker() {
    PendQueue& p = pend_q();
    std::unique_lock<std::mutex> lk(p.mx);
    for (;;) {
        p.cv.wait(lk, [&] { return !p.q.empty(); });
        if (post_submit_visibility_enabled()) {
            pend_wait_post_submit(p, lk);
        } else {
            // Preserve the established compatibility path for older SDK callers.
            lk.unlock();
            struct timespec ts{0, 1000000};
            nanosleep(&ts, nullptr);
            lk.lock();
        }
        pend_drain_locked(p, lk);
    }
}
void pend_enqueue(const Pm4Command& c) {
    PendQueue& p = pend_q();
    PendWrite w; w.cmd = c;
    if (c.kind == Pm4Command::Kind::WriteData && c.wd_data && c.wd_num) {
        w.wd_copy.assign(c.wd_data, c.wd_data + c.wd_num);   // cb memory may be recycled before drain
        w.cmd.wd_data = w.wd_copy.data();
    }
    {
        std::lock_guard<std::mutex> lk(p.mx);
        if (!p.worker_started) { p.worker_started = true; std::thread(pend_worker).detach(); }
        p.q.push_back(std::move(w));
    }
    p.cv.notify_all();   // notify_one could wake a drain waiter instead of the pend worker
}
} // namespace

// Synchronous drain: apply every pending completion write NOW, in order. Called from the fold's
// WaitRegMem check, before the renderer executes, and by the EOP-event worker before it posts.
extern "C" void prosper_gpu_drain_completion_writes() {
    PendQueue& p = pend_q();
    std::unique_lock<std::mutex> lk(p.mx);
    pend_wait_post_submit(p, lk);
    pend_drain_locked(p, lk);
}

extern "C" void prosper_gpu_enable_post_submit_visibility() {
    g_post_submit_visibility.store(true, std::memory_order_release);
}

extern "C" void prosper_gpu_submit_scope_begin() {
    if (!post_submit_visibility_enabled()) return;
    PendQueue& p = pend_q();
    std::unique_lock<std::mutex> lk(p.mx);
    p.cv.wait(lk, [&] { return p.inflight == 0; });
    t_submit_scope_depth++;
    p.active_submits++;
    p.cv.notify_all();
}

extern "C" void prosper_gpu_submit_scope_end() {
    if (!post_submit_visibility_enabled()) return;
    // Invalid/rejected calls to a submit NID still pass through its generated return hook. Such a
    // call has no local token and must not retire a valid submit executing on another thread.
    if (t_submit_scope_depth == 0) return;
    PendQueue& p = pend_q();
    {
        std::lock_guard<std::mutex> lk(p.mx);
        if (p.active_submits == 0) return; // defensive: preserve both counters if the invariant broke
        t_submit_scope_depth--;
        if (--p.active_submits == 0)
            p.release_after = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
    }
    p.cv.notify_all();
}

extern "C" bool prosper_gpu_submit_scope_active() {
    if (!post_submit_visibility_enabled()) return false;
    PendQueue& p = pend_q();
    std::lock_guard<std::mutex> lk(p.mx);
    return p.active_submits > 0;
}

// The renderer runs synchronously inside the submit import and needs resource initialization before
// it samples guest memory. A full drain here is unsafe: it also exposes ReleaseMem and EVENT_WRITE
// completion fences before the submitter has returned and finished its guest-side bookkeeping,
// allowing another thread to recycle their 0x20-byte labels (#312). Extract every queued
// WriteData/DmaData resource update except one whose destination overlaps a queued completion
// fence/event. Selected resource writes may pass unrelated completion records, but never a write to
// the same label. This boundary retains small descriptor/constant uploads used by older titles;
// size/content heuristics left those writes one frame behind.
extern "C" void prosper_gpu_drain_renderer_writes() {
    if (!post_submit_visibility_enabled()) {
        prosper_gpu_drain_completion_writes();
        return;
    }
    PendQueue& p = pend_q();
    for (;;) {
        std::unique_lock<std::mutex> lk(p.mx);
        if (p.inflight > 0) {
            p.cv.wait(lk);
            continue;
        }
        auto overlaps_completion = [&](uint64_t addr, uint64_t bytes) {
            if (!addr || !bytes) return true;
            const uint64_t end = addr + bytes;
            if (end < addr) return true;
            for (const PendWrite& queued : p.q) {
                uint64_t target = 0, target_bytes = 0;
                if (queued.cmd.kind == Pm4Command::Kind::ReleaseMem) {
                    target = queued.cmd.rel_addr;
                    target_bytes = queued.cmd.rel_data_sel == 1 ? 4 : 8;
                } else if (queued.cmd.kind == Pm4Command::Kind::EventWrite) {
                    target = queued.cmd.event_addr;
                    target_bytes = 8;
                }
                if (target && target < end && addr < target + target_bytes) return true;
            }
            return false;
        };
        auto it = std::find_if(p.q.begin(), p.q.end(), [&](const PendWrite& w) {
            using K = Pm4Command::Kind;
            if (w.cmd.kind == K::DmaData)
                return !overlaps_completion(w.cmd.dd_dst, w.cmd.dd_bytes);
            if (w.cmd.kind != K::WriteData || !w.cmd.wd_data || !w.cmd.wd_num) return false;
            const uint64_t bytes = (uint64_t)w.cmd.wd_num * 4;
            return !overlaps_completion(w.cmd.wd_addr, bytes);
        });
        if (it == p.q.end()) return;
        PendWrite w = std::move(*it);
        p.q.erase(it);
        p.inflight++;
        lk.unlock();
        apply_deferred_effect(w.cmd);
        lk.lock();
        p.inflight--;
        p.cv.notify_all();
    }
}

// Resolve a scalar label against the writes queued by the currently executing submit without
// publishing those writes to guest memory. This gives WAIT_REG_MEM its in-queue ordering semantics
// while other guest threads continue to see the pre-submit value until the import returns.
bool pend_overlay_qword(uint64_t addr, uint64_t* value) {
    PendQueue& p = pend_q();
    std::lock_guard<std::mutex> lk(p.mx);
    if (p.q.empty()) return false;
    uint64_t v = *value;
    bool touched = false;
    for (const PendWrite& w : p.q) {
        const Pm4Command& c = w.cmd;
        using K = Pm4Command::Kind;
        if (c.kind == K::ReleaseMem && c.rel_addr == addr && c.rel_value_valid) {
            if (c.rel_data_sel == 1) {
                uint32_t lo = (uint32_t)c.rel_value;
                memcpy(&v, &lo, sizeof lo);
                touched = true;
            } else if (c.rel_data_sel == 2) {
                v = c.rel_value;
                touched = true;
            }
        } else if (c.kind == K::WriteData && c.wd_addr == addr && c.wd_data && c.wd_num) {
            const size_t n = std::min<size_t>((size_t)c.wd_num * 4, sizeof v);
            memcpy(&v, c.wd_data, n);
            touched = true;
        } else if (c.kind == K::DmaData && c.dd_dst == addr && c.dd_valid && c.dd_src <= UINT32_MAX) {
            const uint32_t word = (uint32_t)c.dd_src;
            const size_t n = std::min<size_t>(c.dd_bytes, sizeof v);
            for (size_t off = 0; off < n; off += sizeof word)
                memcpy((uint8_t*)&v + off, &word, std::min(sizeof word, n - off));
            touched = true;
        }
    }
    if (touched) *value = v;
    return touched;
}

// --- WAIT_REG_MEM per-queue barrier model (issue #312 heap-corruption root cause). ---------------
//
// On real hardware WAIT_REG_MEM BLOCKS its queue until its condition holds; packets after it —
// including the RELEASE_MEM fence the game polls — execute only once the dependency is satisfied.
// Other queues keep running while one is paused. Our fold is synchronous at submit time, and
// DOLL's UE4 RHI submits from two guest threads, so a consumer Dcb (wait label==1, then EOP fence
// write) routinely arrives BEFORE its producer (the Dcb whose RELEASE_MEM writes that label). The
// old fold logged "dependency violated" and barreled on: the consumer's fence completed early, the
// game freed the heap block holding the transient semaphore label, and the producer's later
// RELEASE_MEM value-1 write landed on the freed MallocBinned3 FFreeBlock header — the "Canary was
// 0x3, should be 0x1" / "free an unrecognized block 0x1000000001" fatals (a 4-byte 0x1 stomp over
// NextFreeBlock; attributed live by the GPU write-ring: 13 ReleaseMem value-1 writes at exactly
// the corrupted qword's address). A/B evidence (issue #312): every run that honored the barrier
// ordering (PROSPER_WAIT_DEFER=1, 5/5) had ZERO corruption fatals; default runs stomped at
// t=40-150 s.
//
// VERDICT AFTER LIVE MEASUREMENT (2026-07-10, ~20 instrumented DOLL menu-drive runs, same build
// A/B): the model is OPT-IN (PROSPER_WAIT_DEFER=1), not default. What the A/B showed:
//   - Model ON eliminates the headline "MallocBinned3 Corruption Canary was 0x3" (Line 152)
//     fatal: 0/6 model runs vs 2/2 barrel-on runs at t~70 s — the class the wait-ordering
//     violation genuinely seeds.
//   - BUT a SECOND, wait-order-INDEPENDENT corruption leg remains and dominates: a burst of ~96
//     ReleaseMem(value=1) writes at t~10 s landing on labels whose qword reads 0x1000000000
//     (producing the exact "free an unrecognized block 0x1000000001" fatal + the 0x20015f00
//     pool-metadata-read crashes). It appears in EVERY config — barrel-on, full deferral with
//     ZERO timeouts, PROSPER_NO_JUMP=1, and PROSPER_EOP_WRITE_SYNC=1 (labels are ptr-valued
//     already AT FENCE BUILD TIME) — so no ordering/timing model of honest writes can remove it.
//   - Under the burst, model-ON runs die EARLIER (t=15-45 s, worker-fault/free-unrecognized
//     family) than barrel-on (t~70 s, canary) — deferral latency shifts guest timing into the
//     already-injected corruption sooner. Net: defaulting ON buys no gameplay and costs run
//     length, so the default stays barrel-on until the injection leg is found.
//   - The historical "5/5 PROSPER_WAIT_DEFER=1 runs have zero corruption" evidence (issue #312)
//     is CONFOUNDED: every one of those runs lost liveness within a minute (semi-frozen guest =
//     no content-load burst = no corruption window). This session's liveness-correct model
//     dissolves that isolation.
//
// The model (opt-in via PROSPER_WAIT_DEFER=1; default = the old barrel-on behavior):
//   - Dcb and DcbFinal feed the same guest graphics ring — the guest's own SubmitCommandBuffers
//     batch splits its buffer array across BOTH (RE'd at eboot+0x220a9a0), so they MUST keep mutual
//     order. SubmitAcb is a distinct async-compute queue: ArcRunner submits real ACB producers, and
//     queue provenance showed their DMA/Release effects must flow while a graphics wait is paused.
//   - When a fold hits an UNSATISFIED wait, ITS stream pauses: the wait and this stream's
//     remaining guest-visible memory effects (ReleaseMem / EVENT_WRITE / WRITE_DATA / DMA_DATA,
//     and further waits) defer IN ORDER behind the barrier.
//   - PER-ADDRESS ORDERING DOMAINS for later submits: while gated items are pending, a NEW fold's
//     memory effect defers (in ring order, behind everything already gated) IFF it touches an
//     address that already has a gated pending write — per-address write order is exactly what
//     the guest's consumed-marker protocol observes (DmaData label:=0 / ReleaseMem label<-1 /
//     poll==1 / free / realloc at the same recycled address), and letting a later write overtake
//     a gated one at the same label swaps fence generations and re-seeds the #312 stomp (measured:
//     run with full independence stomped with ZERO timeouts). Effects to untouched addresses FLOW
//     — gating them recreated a CPU<->GPU circular stall (guest polls fence F gated behind barrier
//     W; the guest write that would satisfy W happens after F's poll -> 1 s timeout cascade,
//     measured as an early-boot wedge + violation stomps).
//   - Each gated tail releases in STRICT FIFO order WITHIN ITS HARDWARE QUEUE. A blocked graphics
//     stream cannot hold an async-compute producer (or vice versa); after another queue makes
//     progress the blocked fronts are re-checked in the same flush. The watchdog also re-checks
//     every 2 ms. Later writes to an awaited address are not gated by the wait alone: WAIT_REG_MEM
//     polls memory, so a producer from another queue/engine/CPU must be able to satisfy it.
//   - NEVER gated (liveness, measured: gating these wedged every naive-pause run): draws/register
//     state (touch no guest memory) and the in-stream Flip (the frame pacer; on real HW the flip
//     engine signals independently). The submit's EOP equeue PULSE follows the hardware
//     visibility contract instead: immediate when no gated writes are pending, otherwise OWED and
//     delivered when the tail drains — see submit_completion_pulse below (pulsing while gated
//     writes were pending let the guest's completion scan free live label blocks: stomps with
//     ZERO ordering violations, measured; withholding without ever re-firing wedged the
//     RenderThread, also measured).
//   - Liveness backstop: a barrier pending longer than defer_timeout_ms() (PROSPER_WAIT_TIMEOUT_MS,
//     default 1000) falls back to the old proceed-with-loud-log behavior, so a genuinely
//     never-written label (or a producer we fail to model) degrades to the pre-#312 state instead
//     of wedging the queue. Every timeout is a dependency violation — the corruption mechanism —
//     so the default is generous (50 ms measurably re-seeded the fatal via real-but-slow
//     producers).
// All of this runs under the caller's submit mutex (hle_agc g_agc_state_mu).
// CONFIDENCE: HIGH on the wait semantics (Kyty/shadPS4 both block the queue's execution thread
// until the condition holds — deferral-in-order is the synchronous-fold equivalent) and on the
// queue-local strict-FIFO ordering (it is what each ring guarantees on hardware); MED on the
// never-gated flip/EOP-pulse liveness exceptions (hardware would order them too, but gating them
// deadlocks our synchronous fold — the re-pulse keeps the guest-observable protocol sound).
namespace {
bool wait_regmem_satisfied(const Pm4Command& c) {
    // The label page can be unmapped/freed — a recycled command buffer referencing a prior generation's
    // fence label, or a producer/consumer that freed the tracking block (the #312 freed-label class). A
    // raw 8-byte read of a stale guest address is a host SEGV, so probe first and treat an unmapped label
    // as NOT satisfied (the barrel-on default) — matching flush_deferred_streams(), which already guards
    // its WAIT_REG_MEM re-check this way. The wm_addr&3 alignment gate at the call site does not catch an
    // unmapped-but-aligned address. #380.
    if (!guest_readable(c.wm_addr, sizeof(uint64_t))) return false;
    uint64_t mem = 0; memcpy(&mem, (const void*)(uintptr_t)c.wm_addr, sizeof mem);
    pend_overlay_qword(c.wm_addr, &mem);
    uint64_t v = mem & c.wm_mask, r = c.wm_ref;
    switch (c.wm_func) {           // PM4 WAIT_REG_MEM compare functions
        case 0: return true;
        case 1: return v <  r;
        case 2: return v <= r;
        case 3: return v == r;
        case 4: return v != r;
        case 5: return v >= r;
        case 6: return v >  r;
        default: return false;
    }
}
uint64_t defer_now_ms() {
    // The submit path executes shader translation, pipeline creation, dispatches, and rendering
    // synchronously while holding the queue mutex. Real hardware performs that work after returning
    // from submit, so another guest submit cannot deliver a WAIT_REG_MEM producer during this host-
    // only interval. The shared guest/GPU clock excludes excess HostGpuClockScope time; using raw
    // steady_clock here aged a healthy barrier past the liveness timeout while its producer was
    // prevented from entering the queue, then deliberately violated ordering as soon as rendering
    // returned (Plucky's first gameplay scene needed 1-2 s of first-use pipeline work). Keep the
    // timeout on the emulated timeline where the guest and producer can actually make progress.
    return prosper_guest_tsc_ns() / 1000000ull;
}
struct DeferItem {
    Pm4Command cmd;                    // barrier (WaitRegMem) or effect (ReleaseMem/EventWrite/WriteData/Flip)
    std::vector<uint32_t> wd_copy;     // owns a WriteData payload (cmd.wd_data repointed into this)
    uint64_t first_blocked_ms = 0;     // barrier only: when it was first found unsatisfied
    bool first_blocked_recorded = false; // guest clock legitimately starts at zero after compensation
};
struct DeferredStream {
    std::vector<DeferItem> items;
    size_t next = 0;
    uint8_t queue = 0;                    // 0=graphics (Dcb/DcbFinal/unknown), 1=async compute (Acb)
    // A retained address DMA cannot execute through this legacy queue. Preserve effects before the
    // rejected copy, but discard every same-stream completion effect appended after it so stale work
    // cannot later signal success when the wait releases.
    size_t discard_from = static_cast<size_t>(-1);
    bool suppress_completion = false;
};
std::vector<DeferredStream> g_deferred;   // paused queue tails; guarded by the caller's submit mutex
bool     g_fold_deferring = false;        // current fold hit an unsatisfied wait
bool     g_fold_discard_deferred_suffix = false;
uint8_t  g_fold_origin = 0;               // current submit entry point; set under the submit mutex
uint64_t g_defer_streams = 0, g_defer_timeouts = 0;
size_t   g_defer_items = 0;               // total queued items across streams (memory guard)
// 1000 ms default. The timeout is the CORRUPTION knob, not a perf knob: every timeout is a
// dependency violation (the pre-#312 corruption mechanism), so it must be rare — and under
// per-stream independence a blocked stream stalls nothing but itself (EOP pulses, flips and all
// other streams flow), so a generous value costs only latency on a genuinely-unmodeled
// dependency. Measured on DOLL's menu content-load burst: at 50 ms the burst produced 8 timeouts
// and the MallocBinned3 fatal returned (late producers — the guest CPU writes some of these
// labels itself under heavy load); satisfied barriers otherwise cluster near ~1 ms via the pend
// queue. Tunable via PROSPER_WAIT_TIMEOUT_MS.
uint64_t defer_timeout_ms() {
    static const uint64_t v = [] {
        const char* e = getenv("PROSPER_WAIT_TIMEOUT_MS");
        long n = e ? strtol(e, nullptr, 0) : 0;
        return n > 0 ? (uint64_t)n : 1000ull; }();
    return v;
}
constexpr size_t   kDeferMaxStreams = 256;   // runaway guard: force-flush oldest beyond this
constexpr size_t   kDeferMaxItems = 200000;  // memory guard (#312 WAIT_DEFER OOM): force-flush
constexpr size_t   kDeferredQueueCount = 2;

// SubmitDcbFinal is the final packet buffer of the ordinary graphics queue, not a third queue.
// Unknown origins retain the historical graphics behavior for direct/test callers.
size_t deferred_queue(uint8_t origin) { return origin == 2 ? 1u : 0u; }
bool queue_pending(size_t queue) {
    return std::any_of(g_deferred.begin(), g_deferred.end(),
                       [queue](const DeferredStream& s) { return s.queue == queue; });
}

// Opt-in gate for the whole model (PROSPER_WAIT_DEFER=1). Default OFF = the pre-#312 barrel-on
// behavior — see the measured verdict in the model block above for why this is not (yet) the
// default: the model is semantically right and kills the canary-152 class, but the remaining
// wait-order-independent injection makes deferral a net loss for DOLL run length today.
bool defer_enabled() {
    static const bool v = [] {
        const char* e = getenv("PROSPER_WAIT_DEFER");
        return e && strtol(e, nullptr, 0) != 0; }();
    return v;
}

// --- Per-address ordering domains: which guest addresses currently have a GATED pending write.
// Small writes (fence labels — the protocol-critical case) index by 8-byte granule in a hash set;
// large fills (the 64 KiB DmaData chunk zero-fills) go in a range list. Each queue has its own
// domain: entries accumulate while that queue's tail is non-empty and clear when it drains. A
// partial drain can leave STALE entries, which only OVER-gate later work on the same queue (safe;
// the tail drains within the watchdog cadence anyway).
std::array<std::unordered_set<uint64_t>, kDeferredQueueCount> g_gated_granules;             // addr >> 3
std::array<std::vector<std::pair<uint64_t, uint64_t>>, kDeferredQueueCount> g_gated_ranges; // [lo, hi)
// The guest-memory span a command writes (0 bytes = writes nothing we track).
void effect_span(const Pm4Command& c, uint64_t* addr, uint64_t* bytes) {
    using K = Pm4Command::Kind;
    switch (c.kind) {
        case K::ReleaseMem: *addr = c.rel_addr;   *bytes = 8; break;
        case K::EventWrite: *addr = c.event_addr; *bytes = 8; break;
        case K::WriteData:  *addr = c.wd_addr;    *bytes = (uint64_t)c.wd_num * 4; break;
        case K::DmaData:    *addr = c.dd_dst;     *bytes = c.dd_bytes; break;
        default:            *addr = 0;            *bytes = 0; break;
    }
}
void gated_register(const Pm4Command& c) {
    uint64_t a = 0, n = 0;
    effect_span(c, &a, &n);
    if (!a || !n) return;
    const size_t queue = deferred_queue(c.queue_origin);
    if (n <= 32) {
        for (uint64_t g = a >> 3; g <= ((a + n - 1) >> 3); g++) g_gated_granules[queue].insert(g);
    } else {
        g_gated_ranges[queue].emplace_back(a, a + n);
    }
}
bool addr_gated(uint64_t a, uint64_t n, uint8_t origin) {
    if (!a || !n) return false;
    const size_t queue = deferred_queue(origin);
    if (!g_gated_granules[queue].empty())
        for (uint64_t g = a >> 3; g <= ((a + n - 1) >> 3); g++)
            if (g_gated_granules[queue].count(g)) return true;
    for (const auto& r : g_gated_ranges[queue])
        if (a < r.second && r.first < a + n) return true;
    return false;
}
bool effect_gated(const Pm4Command& c) {
    uint64_t a = 0, n = 0;
    effect_span(c, &a, &n);
    return addr_gated(a, n, c.queue_origin);
}

// Should this command join the gated tail? Yes when its own fold is paused (everything downstream
// of the fold's unsatisfied wait keeps stream order), or when it writes into an address domain
// that already has a gated pending write from the SAME queue (per-address ring order across folds).
bool defer_gate(const Pm4Command& c) {
    if (g_fold_deferring) return true;
    if (g_deferred.empty()) return false;
    return effect_gated(c);
}

bool g_fold_stream_open = false;   // current top-level fold already opened its deferred stream
void defer_push(const Pm4Command& c) {
    if (!g_fold_stream_open) {     // one deferred stream per fold that defers anything
        g_deferred.emplace_back();
        g_deferred.back().queue = static_cast<uint8_t>(deferred_queue(c.queue_origin));
        if (g_fold_discard_deferred_suffix) {
            g_deferred.back().discard_from = 0;
            g_deferred.back().suppress_completion = true;
        }
        g_fold_stream_open = true;
        g_defer_streams++;
    }
    DeferItem it; it.cmd = c;
    if (c.kind == Pm4Command::Kind::WriteData && c.wd_data && c.wd_num) {
        it.wd_copy.assign(c.wd_data, c.wd_data + c.wd_num);   // cb memory may be recycled before flush
        it.cmd.wd_data = it.wd_copy.data();
    }
    gated_register(it.cmd);
    g_deferred.back().items.push_back(std::move(it));
    g_defer_items++;
}
void apply_effect(const Pm4Command& c) {
    using K = Pm4Command::Kind;
    switch (c.kind) {
        case K::ReleaseMem: honor_eop_write(c); break;
        case K::EventWrite: honor_event_write(c); break;
        case K::WriteData:  honor_write_data(c); break;
        case K::DmaData:    honor_dma_data(c); break;
        case K::Flip:       if (c.flip_valid) prosper_vo_flip_from_gpu(c.flip_handle, c.flip_bufidx,
                                                                       c.flip_mode, c.flip_arg); break;
        default: break;
    }
}
// The guest-memory span a deferred effect writes (0 = none / self-guarded). A deferred effect can
// be released tens of ms after its fold; guard against the guest having unmapped the target in the
// window (MallocBinned3 decommits pool pages) — the fold-time path writes "immediately" and never
// needed this. honor_dma_data() range-checks itself.
uint64_t effect_target(const Pm4Command& c, uint32_t* bytes) {
    using K = Pm4Command::Kind;
    switch (c.kind) {
        case K::ReleaseMem: *bytes = 8; return c.rel_addr;
        case K::EventWrite: *bytes = 8; return c.event_addr;
        case K::WriteData:  *bytes = c.wd_num * 4; return c.wd_addr;
        default: *bytes = 0; return 0;
    }
}
void apply_deferred_effect(const Pm4Command& c) {
    uint32_t bytes = 0;
    uint64_t t = effect_target(c, &bytes);
    if (t && bytes && !guest_readable(t, bytes)) {
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 24)
            fprintf(stderr, "[agc] deferred effect target unmapped — write SKIPPED: kind=%u addr=0x%llx bytes=%u\n",
                    (unsigned)c.kind, (unsigned long long)t, bytes);
        return;
    }
    apply_effect(c);
}
} // namespace

void execute_ordered_memory_effect(const GpuState::MemoryEffect& effect) {
    apply_deferred_effect(effect.cmd);
}

bool last_fold_deferred() { return g_fold_deferring; }
bool deferred_pending()   { return !g_deferred.empty(); }

// --- EOP pulse visibility contract (#312, the decisive leg). -------------------------------------
// On real hardware the EOP interrupt for a submission fires only after EVERYTHING before it in
// the ring has executed — the guest's completion scan (AgcInterrupt -> cleanup, label sweeps,
// frame-end label batch-free at eboot+0x220bd50) relies on it: when the interrupt arrives, every
// fence write of that frame is visible. Firing the pulse while gated tail writes are still
// pending lets the scan observe a half-retired frame, free live label blocks, and take the stomp
// when the gated write lands — corruption WITHOUT any ordering violation among the writes
// themselves (observed live: stomps with zero DEFER timeouts). So: a submit's pulse fires
// immediately only when its queue's gated tail is empty; otherwise it is OWED and fires when that
// tail fully drains (flush below). The other queue's pulse remains independent. Liveness: flips
// flow regardless, the 2 ms watchdog drains a tail the moment its front barrier satisfies, and the
// timeout bounds a genuinely-stuck barrier.
// CONFIDENCE: HIGH on the hardware contract (ring-ordered interrupt), MED that pulse-on-full-
// drain (vs per-stream) is the right granularity — it only errs later, the safe direction.
namespace { std::array<uint64_t, kDeferredQueueCount> g_owed_pulses{}; }
void submit_completion_pulse(bool submit_rejected) {
    const size_t queue = deferred_queue(g_fold_origin);
    const bool pending = queue_pending(queue);
    if (getenv("PROSPER_EOPLOG")) {
        static uint64_t call = 0;
        fprintf(stderr, "[eop-pulse] #%llu queue=%s rejected=%d queue_pending=%d owed=%llu -> %s\n",
                (unsigned long long)++call, queue ? "Acb" : "Dcb", submit_rejected ? 1 : 0,
                pending ? 1 : 0, (unsigned long long)g_owed_pulses[queue],
                submit_rejected ? "SKIP(rejected)" : (!pending ? "FIRE" : "OWE"));
    }
    if (submit_rejected) return;
    if (pending) { g_owed_pulses[queue]++; return; }
    prosper_eq_trigger_eop();
}

// Release deferred streams (call at every submit and from hle_agc's watchdog ticker, under the
// submit mutex). STRICT PER-QUEUE FIFO (#312/#1226): items apply in submission order within Dcb or
// Acb, and a front barrier holds later streams from that queue. The other queue keeps running, as
// hardware requires; the submit boundaries and watchdog re-check every blocked front. A barrier
// pending > defer_timeout_ms() proceeds with the pre-#312 "dependency violated" loud log (liveness
// backstop). Returns how many streams fully completed across both queues.
int flush_deferred_streams() {
    if (g_deferred.empty()) return 0;
    // Legacy SDK callers retain the original eager visibility. Modern callers consult the pending
    // scalar overlay instead, so their completion labels remain post-submit.
    if (!post_submit_visibility_enabled()) prosper_gpu_drain_completion_writes();
    int completed = 0;
    std::array<int, kDeferredQueueCount> signalable_completed{};
    std::array<bool, kDeferredQueueCount> blocked_queue{};
    for (size_t si = 0; si < g_deferred.size(); ) {
        DeferredStream& s = g_deferred[si];
        const size_t queue = s.queue;
        if (blocked_queue[queue]) { si++; continue; }
        bool blocked = false;
        bool force = g_deferred.size() > kDeferMaxStreams || g_defer_items > kDeferMaxItems;
        while (s.next < s.items.size()) {
            if (s.next >= s.discard_from) {
                s.next = s.items.size();
                break;
            }
            DeferItem& it = s.items[s.next];
            if (it.cmd.kind == Pm4Command::Kind::WaitRegMem) {
                // The label page can be unmapped by the time we re-check (freed mid-defer):
                // treat as never-satisfiable and take the timeout path immediately.
                bool readable = guest_readable(it.cmd.wm_addr, 8);
                if (readable && it.first_blocked_recorded && wait_regmem_satisfied(it.cmd)) {
                    // Barrier-latency telemetry (bounded): how long do REAL producers take? This
                    // is the data the timeout default is tuned against.
                    uint64_t waited = defer_now_ms() - it.first_blocked_ms;
                    if (waited > 20) {
                        static std::atomic<int> n{0};
                        int ln = n.fetch_add(1);
                        if (ln < 32 || (ln & 511) == 0)
                            fprintf(stderr, "[agc] WaitRegMem satisfied after %llums blocked: [0x%llx] ref=0x%llx\n",
                                    (unsigned long long)waited, (unsigned long long)it.cmd.wm_addr,
                                    (unsigned long long)it.cmd.wm_ref);
                    }
                }
                if (!readable || !wait_regmem_satisfied(it.cmd)) {
                    uint64_t now = defer_now_ms();
                    if (!it.first_blocked_recorded) {
                        it.first_blocked_ms = now;
                        it.first_blocked_recorded = true;
                    }
                    if (readable && !force && now - it.first_blocked_ms < defer_timeout_ms()) {
                        blocked = true; break;
                    }
                    g_defer_timeouts++;
                    static std::atomic<int> logged{0};
                    int ln = logged.fetch_add(1);
                    if (ln < 64 || (ln & 255) == 0)
                        fprintf(stderr, "[agc] WaitRegMem DEFER TIMEOUT #%llu after %llums: [0x%llx]&0x%llx func=%u ref=0x%llx%s — dependency violated (proceeding)\n",
                                (unsigned long long)g_defer_timeouts,
                                (unsigned long long)(now - it.first_blocked_ms),
                                (unsigned long long)it.cmd.wm_addr, (unsigned long long)it.cmd.wm_mask,
                                it.cmd.wm_func, (unsigned long long)it.cmd.wm_ref,
                                readable ? "" : " (label UNMAPPED)");
                }
                s.next++;
                continue;
            }
            apply_deferred_effect(it.cmd);
            s.next++;
        }
        if (blocked) {
            blocked_queue[queue] = true;   // later streams in THIS queue cannot overtake
            si++;
            continue;
        }
        g_defer_items -= s.items.size() < g_defer_items ? s.items.size() : g_defer_items;
        signalable_completed[queue] += !s.suppress_completion;
        g_deferred.erase(g_deferred.begin() + si);
        completed++;
    }

    for (size_t queue = 0; queue < kDeferredQueueCount; queue++) {
        if (queue_pending(queue)) continue;
        // Address-domain entries are queue-local. Clear them as soon as that queue drains so a
        // blocked peer queue cannot over-gate future work from this one.
        g_gated_granules[queue].clear();
        g_gated_ranges[queue].clear();
        // Deliver every pulse owed by submissions from this queue. A blocked peer queue retains its
        // own pulses; an Acb completion must not wait for an unrelated Dcb barrier (and vice versa).
        uint64_t owed = g_owed_pulses[queue];
        g_owed_pulses[queue] = 0;
        if (!owed && signalable_completed[queue] > 0) owed = 1;
        while (owed--) prosper_eq_trigger_eop();
    }
    return completed;
}

void GpuState::apply(const Pm4Command& c) {
    using K = Pm4Command::Kind;
    command_order = c.stream_order ? c.stream_order : command_order + 1;
    switch (c.kind) {
        case K::SetRegsIndirect: {
            if (!dma_copies.empty()) {
                dma_execution_rejected = true;
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    fprintf(stderr,
                            "[agc] ordered DMA submit rejected: SetRegsIndirect reads guest memory "
                            "after retained DMA (order=%llu)\n",
                            (unsigned long long)command_order);
                break;
            }
            if (c.regs_vaddr == 0 || c.num_regs == 0 || c.num_regs > kMaxRegsPerPacket) return;
            // #312/#448: the indirect-register array lives in GUEST memory (regs_vaddr from the packet),
            // and a freed/decommitted or recycled command buffer can leave it unmapped. Every other
            // guest read in this file is guest_readable-guarded; this reader (up to 4096*8 = 32 KiB) was
            // the one gap — an unmapped regs_vaddr would host-SIGSEGV in the loop below. Skip if unmapped.
            if (!guest_readable(c.regs_vaddr, c.num_regs * (uint32_t)sizeof(ShaderReg))) {
                static std::atomic<int> n{0};
                if (n.fetch_add(1) < 24)
                    fprintf(stderr, "[agc] SetRegsIndirect array unmapped — packet SKIPPED: vaddr=0x%llx num=%u\n",
                            (unsigned long long)c.regs_vaddr, c.num_regs);
                return;
            }
            auto* regs = reinterpret_cast<const ShaderReg*>(static_cast<uintptr_t>(c.regs_vaddr));
            auto& file = (c.reg_class == RegClass::Cx) ? cx
                       : (c.reg_class == RegClass::Sh) ? sh : uc;
            if (getenv("PROSPER_RESDUMP")) {
                const char* cn = c.reg_class == RegClass::Cx ? "Cx" : c.reg_class == RegClass::Sh ? "Sh" : "Uc";
                fprintf(stderr, "[regindir] class=%s num=%u vaddr=0x%llx pairs:", cn, c.num_regs,
                        (unsigned long long)c.regs_vaddr);
                for (uint32_t i = 0; i < c.num_regs && i < 40; i++)
                    fprintf(stderr, " (off=0x%x val=0x%x)", regs[i].offset, regs[i].value);
                fprintf(stderr, "\n");
            }
            // PROSPER_REGBLOAT (#1264 investigation): Blue Prince's cx register file was observed live
            // with ~94,000 entries whose keys span the full 32-bit space (real register offsets are a
            // few hundred), making the per-draw snapshot copy in the Draw case below take seconds per
            // submit. Attribute the garbage: log any indirect array whose offsets exceed the sane
            // register window, with the packet's provenance, so the corrupt producer is identifiable.
            static const bool regbloat = getenv("PROSPER_REGBLOAT") != nullptr;
            uint32_t bad = 0, first_bad = 0;
            // #1364: within-array clobber detector — a DB base offset written nonzero and then
            // ZERO by a LATER slot of the SAME array is the stale-slot signature (#1353's
            // (DB_Z_WRITE_BASE, 0)). Dump the whole array for the first few occurrences so the
            // arena's real record format is decodable offline. Bits 0..3 = LO 0x12..0x15,
            // bits 4..7 = HI 0x1A..0x1D. Gated with the trace.
            uint32_t dbbase_nonzero_mask = 0;
            bool dbbase_clobbered = false;
            for (uint32_t i = 0; i < c.num_regs; i++) {
                // Gen5's BuildInterpolantMapping helper returns a reserved virtual Cx-register
                // bank instead of raw hardware offsets. The Dcb consumes those pairs unchanged;
                // the real driver resolves 0x10000000+n to SPI_PS_INPUT_CNTL_n while building the
                // command stream. Our HLE packet deliberately retains the guest array until fold
                // time (it may be patched after packet creation), so perform that one documented
                // resolution here before applying the ordinary bounded-register-file guard.
                uint32_t offset = regs[i].offset;
                constexpr uint32_t kVirtualPsInputCntl0 = 0x10000000u;
                if (c.reg_class == RegClass::Cx &&
                    offset >= kVirtualPsInputCntl0 &&
                    offset < kVirtualPsInputCntl0 + 32u) {
                    offset = prosper::agc::Pm4::SPI_PS_INPUT_CNTL_0 +
                             (offset - kVirtualPsInputCntl0);
                }
                // Hardware drops writes to nonexistent register offsets; mirror that instead of
                // folding placeholder/stale array slots into the register file (see kRegOffsetLimit).
                if (offset >= kRegOffsetLimit) {
                    if (bad++ == 0) first_bad = i;
                    static std::atomic<int> dropped_note{0};
                    if (dropped_note.fetch_add(1) < 4)
                        fprintf(stderr,
                                "[agc] out-of-range indirect reg write dropped: class=%d off=0x%x "
                                "val=0x%x (#1264; PROSPER_REGBLOAT=1 for provenance)\n",
                                (int)c.reg_class, regs[i].offset, regs[i].value);
                    continue;
                }
                file[offset] = regs[i].value;
                // PROSPER_DBBASETRACE (#1353): log every cx write to the DB Z/STENCIL base
                // registers (LO 0x12..0x15, HI 0x1A..0x1D) with its source path, to attribute
                // which packet family programs (or clobbers) a base half — this trace found the
                // stale arena slot writing (DB_Z_WRITE_BASE, 0) after the real pair write.
                static const bool dbbase_trace = getenv("PROSPER_DBBASETRACE") != nullptr;
                if (dbbase_trace && c.reg_class == RegClass::Cx &&
                    ((offset >= 0x12u && offset <= 0x15u) ||
                     (offset >= 0x1Au && offset <= 0x1Du))) {
                    static std::atomic<int> n{0};
                    if (n.fetch_add(1) < 400000)
                        fprintf(stderr, "[dbbase] indirect off=0x%x val=0x%x order=%llu\n",
                                regs[i].offset, regs[i].value,
                                (unsigned long long)command_order);
                    const uint32_t bit = offset <= 0x15u
                        ? (offset - 0x12u) : (4u + offset - 0x1Au);
                    if (regs[i].value != 0u) dbbase_nonzero_mask |= 1u << bit;
                    else if (dbbase_nonzero_mask & (1u << bit)) dbbase_clobbered = true;
                }
                if (c.reg_class == RegClass::Cx &&
                    offset == prosper::agc::Pm4::DB_RENDER_CONTROL &&
                    (regs[i].value & 0x3u) &&
                    getenv("PROSPER_DS_CLEARLOG"))
                    fprintf(stderr,
                            "[ds-clear-reg] order=%llu value=%08x depth=%u stencil=%u\n",
                            (unsigned long long)command_order, regs[i].value,
                            regs[i].value & 1u, (regs[i].value >> 1) & 1u);
            }
            if (dbbase_clobbered) {
                static std::atomic<int> dumps{0};
                const int seq = dumps.fetch_add(1);
                if (seq < 6) {
                    fprintf(stderr,
                            "[dbbase-clobber] seq=%d order=%llu vaddr=0x%llx num=%u full array:",
                            seq, (unsigned long long)command_order,
                            (unsigned long long)c.regs_vaddr, c.num_regs);
                    for (uint32_t k = 0; k < c.num_regs && k < 512; k++)
                        fprintf(stderr, " %x:%x", regs[k].offset, regs[k].value);
                    fprintf(stderr, "\n");
                }
            }
            if (regbloat) {
                if (bad) {
                    static std::atomic<int> n{0};
                    const int seq = n.fetch_add(1);
                    if (seq < 48) {
                        const char* cn = c.reg_class == RegClass::Cx ? "Cx"
                                       : c.reg_class == RegClass::Sh ? "Sh" : "Uc";
                        fprintf(stderr,
                                "[regbloat] indirect class=%s vaddr=0x%llx num=%u bad=%u first_bad_i=%u "
                                "order=%llu pairs@first_bad:",
                                cn, (unsigned long long)c.regs_vaddr, c.num_regs, bad, first_bad,
                                (unsigned long long)command_order);
                        for (uint32_t k = first_bad; k < c.num_regs && k < first_bad + 4; k++)
                            fprintf(stderr, " (0x%x,0x%x)", regs[k].offset, regs[k].value);
                        fprintf(stderr, "\n");
                    }
                    // Level 2: dump the ENTIRE array for the first few bad packets so the real
                    // entry format (tags/blocks vs flat pairs) is decodable offline.
                    static const bool full = []{ const char* v = getenv("PROSPER_REGBLOAT");
                                                 return v && v[0] == '2'; }();
                    if (full && seq < 6) {
                        fprintf(stderr, "[regbloat-full] seq=%d vaddr=0x%llx num=%u:",
                                seq, (unsigned long long)c.regs_vaddr, c.num_regs);
                        for (uint32_t k = 0; k < c.num_regs && k < 256; k++)
                            fprintf(stderr, " %x:%x", regs[k].offset, regs[k].value);
                        fprintf(stderr, "\n");
                    }
                }
                static std::atomic<uint32_t> watermark{4096};
                uint32_t wm = watermark.load(std::memory_order_relaxed);
                if (file.size() >= wm &&
                    watermark.compare_exchange_strong(wm, wm * 4, std::memory_order_relaxed))
                    fprintf(stderr, "[regbloat] register file size crossed %u (class=%d order=%llu)\n",
                            wm, (int)c.reg_class, (unsigned long long)command_order);
            }
            state_dirty_ = true;   // register state changed -> the next draw needs a fresh snapshot
            break;
        }
        case K::SetRegDirect: {
            // SET_*_REG writes a consecutive range into the file named by its opcode. SH commonly
            // uploads a whole user-data SGPR block, while the direct APIs emit one Cx/Sh/Uc pair.
            auto& file = (c.reg_class == RegClass::Cx) ? cx
                       : (c.reg_class == RegClass::Sh) ? sh : uc;
            if (getenv("PROSPER_RESDUMP")) {
                const char* cn = c.reg_class == RegClass::Cx ? "Cx" : c.reg_class == RegClass::Sh ? "Sh" : "Uc";
                fprintf(stderr, "[regdirect] class=%s off=0x%x count=%u vals:", cn, c.reg_offset,
                        c.reg_data ? c.reg_count : 1u);
                if (c.reg_data && c.reg_count)
                    for (uint32_t k = 0; k < c.reg_count && k < 40; k++)
                        fprintf(stderr, " 0x%x", c.reg_data[k]);
                else fprintf(stderr, " 0x%x", c.reg_value);
                fprintf(stderr, "\n");
            }
            if (!c.reg_data || c.reg_count == 0 || c.reg_count > kMaxRegsPerPacket) break;
            // Same bounded-register-file contract as the indirect path (see kRegOffsetLimit).
            if (c.reg_offset >= kRegOffsetLimit || c.reg_offset + c.reg_count > kRegOffsetLimit) {
                static std::atomic<int> n{0};
                if (n.fetch_add(1) < 8)
                    fprintf(stderr,
                            "[agc] out-of-range direct reg write dropped: class=%d off=0x%x count=%u "
                            "val0=0x%x order=%llu (#1264)\n",
                            (int)c.reg_class, c.reg_offset, c.reg_count, c.reg_data[0],
                            (unsigned long long)command_order);
                if (c.reg_offset >= kRegOffsetLimit) break;
            }
            for (uint32_t k = 0; k < c.reg_count && c.reg_offset + k < kRegOffsetLimit; k++)
                file[c.reg_offset + k] = c.reg_data[k];
            // PROSPER_DBBASETRACE (#1353): direct-span sibling of the indirect-path trace above.
            {
                static const bool dbbase_trace = getenv("PROSPER_DBBASETRACE") != nullptr;
                if (dbbase_trace && c.reg_class == RegClass::Cx && c.reg_offset <= 0x1Du &&
                    c.reg_offset + c.reg_count > 0x12u) {
                    static std::atomic<int> n{0};
                    if (n.fetch_add(1) < 400000) {
                        fprintf(stderr, "[dbbase] direct off=0x%x count=%u order=%llu vals:",
                                c.reg_offset, c.reg_count, (unsigned long long)command_order);
                        for (uint32_t k = 0; k < c.reg_count && k < 16; k++)
                            fprintf(stderr, " 0x%x", c.reg_data[k]);
                        fprintf(stderr, "\n");
                    }
                }
            }
            state_dirty_ = true;
            break;
        }
        case K::SetIndexType:
            index_type = c.index_size;
            state_dirty_ = true;
            break;
        case K::SetNumInstances:
            num_instances = c.instance_count;
            state_dirty_ = true;
            break;
        case K::SetIndexBase:
            index_base = c.ib_addr;   // bind index-buffer base (issue #232)
            break;
        case K::SetIndexCount:
            index_num = c.index_count;   // bind index count (issue #232)
            break;
        case K::SetBaseIndirectArgs:
            // Live SDK-13 streams use both full GPU virtual addresses and 32-bit updates within the
            // already-selected aperture (Astro alternates 0x5074063c0 and 0x074063c0 for the same
            // compute argument allocation). Low guest GPU VAs are not valid mappings, so retain the
            // last explicit upper half for a low-only update; without prior aperture state the low
            // value remains fail-closed and will be rejected as unreadable by the executor.
            {
                if (c.indirect_shader_type > 1) break;
                uint64_t& current = c.indirect_shader_type == 0
                    ? indirect_graphics_base : indirect_compute_base;
                uint64_t base = c.indirect_base;
                if (base <= UINT32_MAX && current > UINT32_MAX)
                    base |= current & ~static_cast<uint64_t>(UINT32_MAX);
                current = base;
            }
            break;
        case K::StallCommandBufferParser:
            parser_stalls.push_back({command_order});
            break;
        case K::DrawIndexOffset: {
            // Gen5 indexed draw (issue #232). Uses the bound index base + count; DrawIndexOffset's own
            // count (c.index_count) overrides the SetIndexCount state when non-zero. The element size is
            // the current SetIndexType (0=16-bit, 1=32-bit), captured in the per-draw snapshot.
            if (getenv("PROSPER_RESDUMP")) {   // draw-vs-bind association diagnostic (#273)
                auto rd = [&](uint32_t off) { auto it = sh.find(off); return it == sh.end() ? 0u : it->second; };
                fprintf(stderr, "[drawpkt] idx#%zu es=0x%08x count=%u dirty=%d ud=[%08x %08x %08x %08x | %08x %08x %08x %08x]\n",
                        draws.size(), rd(0xc8), c.index_count ? c.index_count : index_num, (int)state_dirty_,
                        rd(0x8c), rd(0x8d), rd(0x8e), rd(0x8f), rd(0x90), rd(0x91), rd(0x92), rd(0x93));
            }
            if (state_dirty_ || !last_snapshot_) {
                auto snap = std::make_shared<GpuState>();
                snap->cx = cx; snap->sh = sh; snap->uc = uc; snap->index_type = index_type;
                snap->num_instances = num_instances;
                last_snapshot_ = std::move(snap);
                state_dirty_ = false;
            }
            uint32_t elem = index_type ? 4u : 2u;
            Draw d;
            d.index_count = c.index_count ? c.index_count : index_num;
            d.instance_count = num_instances;
            d.state = last_snapshot_;
            if (index_base && d.index_count) {
                d.indexed = true;
                d.index_addr = index_base + (uint64_t)c.index_offset * elem;
                // Preserve the raw base + element offset so the executor can recompute the address if
                // it auto-detects a different element size (#304 — DOLL's 32-bit Slate index buffers).
                d.index_base = index_base;
                d.index_offset = c.index_offset;
                d.from_offset = true;
            }
            draws.push_back(std::move(d));
            draws.back().command_order = command_order;
            break;
        }
        case K::DrawIndexAuto:
        case K::DrawIndex: {
            // Snapshot the register state AT THE DRAW (shared with consecutive draws until a register
            // write dirties it), so a future per-draw executor can render each draw under its own
            // shaders/mask/blend instead of the end-of-submit fold. Inert for the current renderer.
            // The snapshot also carries index_type — the index element size a DrawIndex needs (#64).
            if (state_dirty_ || !last_snapshot_) {
                auto snap = std::make_shared<GpuState>();
                snap->cx = cx; snap->sh = sh; snap->uc = uc; snap->index_type = index_type;
                snap->num_instances = num_instances;
                last_snapshot_ = std::move(snap);
                state_dirty_ = false;
            }
            Draw d;
            d.index_count = c.index_count;
            d.instance_count = num_instances;
            d.state = last_snapshot_;
            d.modifier = c.di_modifier;
            if (c.kind == K::DrawIndex) {
                // Mark as indexed only when the packet was fully decoded — a short packet's addr/
                // modifier would be fabricated zeros, and `indexed` promises index_addr is real.
                d.indexed = c.di_valid;
                d.index_addr = c.di_index_addr;
            }
            draws.push_back(std::move(d));
            draws.back().command_order = command_order;
            break;
        }
        case K::DrawIndexIndirect: {
            if (state_dirty_ || !last_snapshot_) {
                auto snap = std::make_shared<GpuState>();
                snap->cx = cx; snap->sh = sh; snap->uc = uc; snap->index_type = index_type;
                snap->num_instances = num_instances;
                last_snapshot_ = std::move(snap);
                state_dirty_ = false;
            }
            Draw d;
            d.state = last_snapshot_;
            d.indexed = true;
            d.modifier = c.di_modifier;
            d.index_base = index_base;
            d.indirect = true;
            if (indirect_graphics_base <= UINT64_MAX - c.indirect_offset)
                d.indirect_args_addr = indirect_graphics_base + c.indirect_offset;
            d.command_order = command_order;
            draws.push_back(std::move(d));
            break;
        }
        case K::ReleaseMem:
            // EOP completion label write. While the queue is paused (this fold hit an unsatisfied
            // wait, or an earlier submit's gated tail is still pending), the write queues behind
            // the barrier IN RING ORDER (see the #312 block above) — completing a fence early is
            // what let the game free live label memory. Otherwise it goes through the pipe-drain
            // queue: completion becomes guest-visible only after the submit returns.
            if (defer_gate(c)) { defer_push(c); break; }
            if (!dma_copies.empty()) {
                ordered_memory_effects.emplace_back(c, command_order);
                break;
            }
            if (eop_write_sync()) honor_eop_write(c); else pend_enqueue(c);
            break;
        case K::WriteData:
            if (defer_gate(c)) { defer_push(c); break; }
            if (!dma_copies.empty()) {
                ordered_memory_effects.emplace_back(c, command_order);
                break;
            }
            if (eop_write_sync()) honor_write_data(c); else pend_enqueue(c);
            break;
        case K::EventWrite:
            if (defer_gate(c)) { defer_push(c); break; }
            if (!dma_copies.empty()) {
                ordered_memory_effects.emplace_back(c, command_order);
                break;
            }
            if (eop_write_sync()) honor_event_write(c); else pend_enqueue(c);
            break;
        case K::DmaData:
            // Address-backed copies are ordinary in-stream producers: retain them beside draws and
            // dispatches so the ordered executor exposes old bytes to earlier consumers and copied
            // bytes to later consumers (#189). Immediate fills keep their established completion-
            // FIFO behavior until a general copy appears; its suffix joins the ordered timeline.
            if (c.dd_src > UINT32_MAX) {
                if (!c.dd_valid) { report_invalid_dma_data(c); break; }
                const bool gated_destination = defer_gate(c);
                const bool gated_source = !g_deferred.empty() &&
                                          addr_gated(c.dd_src, c.dd_bytes, c.queue_origin);
                if (gated_destination || gated_source) {
                    dma_copies.push_back({c.dd_dst, c.dd_src, c.dd_bytes, c.dd_sels,
                                          command_order, pkt_addr(c)});
                    dma_execution_rejected = true;
                    g_fold_discard_deferred_suffix = true;
                    if (g_fold_stream_open && !g_deferred.empty()) {
                        DeferredStream& stream = g_deferred.back();
                        stream.discard_from = std::min(stream.discard_from, stream.items.size());
                        stream.suppress_completion = true;
                    }
                    static std::atomic<int> warned{0};
                    if (warned.fetch_add(1) < 24)
                        fprintf(stderr,
                                "[agc] ordered DMA submit rejected: WAIT_DEFER owns %s dependency "
                                "(src=0x%llx dst=0x%llx bytes=%u order=%llu)\n",
                                gated_source ? "source" : "destination/stream",
                                (unsigned long long)c.dd_src, (unsigned long long)c.dd_dst,
                                c.dd_bytes, (unsigned long long)command_order);
                    break;
                }
                // Everything queued before the first general copy is its ordered prefix. Land that
                // prefix now; subsequent effects are retained below and cannot overtake the copy.
                if (dma_copies.empty()) prosper_gpu_drain_renderer_writes();
                dma_copies.push_back({c.dd_dst, c.dd_src, c.dd_bytes, c.dd_sels,
                                      command_order, pkt_addr(c)});
                break;
            }
            if (defer_gate(c)) { defer_push(c); break; }
            if (!dma_copies.empty()) {
                ordered_memory_effects.emplace_back(c, command_order);
                break;
            }
            if (eop_write_sync()) honor_dma_data(c); else pend_enqueue(c);
            break;
        case K::WaitRegMem:
            if (!dma_copies.empty()) {
                dma_execution_rejected = true;
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    fprintf(stderr,
                            "[agc] ordered DMA submit rejected: WAIT_REG_MEM reads guest memory "
                            "after retained DMA (order=%llu)\n",
                            (unsigned long long)command_order);
                break;
            }
            // Real WAIT_REG_MEM semantics (#312): an unsatisfied wait PAUSES this queue — the
            // stream's remaining memory effects are deferred until the condition holds (flushed
            // at subsequent submits by flush_deferred_streams; loud timeout fallback preserves
            // liveness). A satisfied wait is a no-op, as before.
            if (c.wm_valid && c.wm_addr && !(c.wm_addr & 3)) {
                // Own fold already paused — the wait keeps stream order. OR: the awaited address
                // has a GATED PENDING WRITE (an earlier-in-ring write it must observe): evaluating
                // now against the stale value could wrong-satisfy — join the tail in ring order.
                // (The rest of this fold's effects then gate too: stream order.)
                if (g_fold_deferring ||
                    (!g_deferred.empty() && addr_gated(c.wm_addr, 8, c.queue_origin))) {
                    g_fold_deferring = true;
                    defer_push(c);
                    break;
                }
                // Legacy callers consume the concrete value. Modern callers keep it private to the
                // submit and let the evaluator overlay the queued scalar value.
                if (!post_submit_visibility_enabled()) prosper_gpu_drain_completion_writes();
                if (!wait_regmem_satisfied(c)) {
                    // Barrier model (#312), opt-in via PROSPER_WAIT_DEFER=1: pause the queue —
                    // everything downstream defers until the condition holds (see the model
                    // block above, including the measured verdict on why this is not default).
                    // Default: the old barrel-on ("dependency violated") behavior.
                    // Bounded diagnostic (an unsatisfied wait is now NORMAL, handled state — and
                    // under the content-load burst it fires thousands of times a minute).
                    static std::atomic<int> logged{0};
                    int ln = logged.fetch_add(1);
                    if (ln < 40 || (ln & 1023) == 0) {
                        // wait_regmem_satisfied() returns false for an UNMAPPED label too (#380), so this
                        // "not satisfied" diagnostic is reached with a stale/freed label whose page may be
                        // gone — a raw 8-byte read then SEGVs, the exact crash #380 fixed but via the log
                        // path #380 did not cover (#448). Read only when mapped; report UNMAPPED otherwise.
                        bool label_readable = guest_readable(c.wm_addr, 8);
                        uint64_t mem = 0; if (label_readable) memcpy(&mem, (const void*)(uintptr_t)c.wm_addr, sizeof mem);
                        // #312 discriminator: build-journal age + freed-heap-shaped label content.
                        uint64_t baddr = 0, bpre = 0, bt = 0;
                        int have = prosper_fence_journal_lookup(pkt_addr(c), &baddr, &bpre, &bt);
                        char hist[512]; label_hist_report(c.wm_addr, hist, sizeof hist);
                        static const char* qn2[] = {"?", "D", "A", "F"};
                        fprintf(stderr, "[agc] WaitRegMem #%d q=%s NOT satisfied at fold time: [0x%llx]&0x%llx = 0x%llx, func=%u ref=0x%llx — %s | built@%llums(age=%lldms)%s pre@build=0x%llx%s%s | %s\n",
                                ln, c.queue_origin <= 3 ? qn2[c.queue_origin] : "?",
                                (unsigned long long)c.wm_addr, (unsigned long long)c.wm_mask,
                                (unsigned long long)(mem & c.wm_mask), c.wm_func, (unsigned long long)c.wm_ref,
                                defer_enabled() ? "pausing queue (deferred effects)" : "dependency violated",
                                (unsigned long long)bt, have ? (long long)(now_ms() - bt) : -1,
                                (have && baddr != c.wm_addr) ? " TARGET-CHANGED" : "",
                                (unsigned long long)bpre, ptr_like(mem) ? " CONTENT-PTR-LIKE(freed?)" : "",
                                label_readable ? "" : " LABEL-UNMAPPED", hist);
                    }
                    if (defer_enabled()) {
                        g_fold_deferring = true;
                        defer_push(c);
                    }
                }
            }
            break;
        case K::DispatchDirect:
            // Retain the dispatch and its exact register snapshot. The submit executor recompiles
            // supported compute programs and runs them in this vector's stream order before exposing
            // completion; unsupported programs remain visible in diagnostics (#576).
            if (state_dirty_ || !last_snapshot_) {
                auto snap = std::make_shared<GpuState>();
                snap->cx = cx; snap->sh = sh; snap->uc = uc; snap->index_type = index_type;
                snap->num_instances = num_instances;
                last_snapshot_ = std::move(snap);
                state_dirty_ = false;
            }
            dispatches.push_back({c.threads_x, c.threads_y, c.threads_z,
                                  c.dispatch_modifier, last_snapshot_, command_order});
            dispatch_count++;
            break;
        case K::DispatchIndirect: {
            if (state_dirty_ || !last_snapshot_) {
                auto snap = std::make_shared<GpuState>();
                snap->cx = cx; snap->sh = sh; snap->uc = uc; snap->index_type = index_type;
                snap->num_instances = num_instances;
                last_snapshot_ = std::move(snap);
                state_dirty_ = false;
            }
            Dispatch d;
            d.modifier = c.dispatch_modifier;
            d.state = last_snapshot_;
            d.command_order = command_order;
            d.indirect = true;
            if (indirect_compute_base <= UINT64_MAX - c.indirect_offset)
                d.indirect_args_addr = indirect_compute_base + c.indirect_offset;
            dispatches.push_back(std::move(d));
            dispatch_count++;
            break;
        }
        case K::SetPredication:
            // Begin/end a GPU predication window (#319). The begin form carries the condition
            // address; the end form carries 0. A short-decoded packet conservatively ENDS the
            // window (never leaves a stale condition gating later jumps).
            pred_cond_addr = c.pred_valid ? c.pred_addr : 0;
            break;
        case K::Jump: {
            // sceAgcDcbJump (#319): execute `jump_dwords` dwords at `jump_addr`, then resume
            // (call-with-length — live capture shows the target segment is exactly the passed
            // size, no jump-back packet, and the parent stream continues after the jump).
            //
            // Predication: a packet-predicated jump (jump_pred, set by sceAgcSetPacketPredication)
            // inside an open SetPredication window executes only when the 64-bit condition reads 0
            // at fold time. POLARITY (CONFIDENCE: MED, empirically pinned on DOLL's title): the
            // predicated jump segments are the game's per-frame backbuffer composite draws — a
            // title frame REQUIRES the composite every frame on real hardware, and the condition
            // memory reads 0 throughout the title steady state, so 0 must mean "execute" here
            // ("skip when non-zero", matching PM4 SET_PREDICATION's draw-discard-on-set model).
            if (!dma_copies.empty()) {
                dma_execution_rejected = true;
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    fprintf(stderr,
                            "[agc] ordered DMA submit rejected: Jump reads target/predication "
                            "memory after retained DMA (order=%llu)\n",
                            (unsigned long long)command_order);
                break;
            }
            if (!c.jump_valid || !c.jump_addr || (c.jump_addr & 3) || !c.jump_dwords) break;
            // PROSPER_NO_JUMP=1: diagnostic A/B — reproduce the pre-#319 behavior (jump ignored).
            static const bool no_jump = [] { const char* e = getenv("PROSPER_NO_JUMP"); return e && e[0] == '1'; }();
            if (no_jump) break;
            constexpr uint32_t kMaxJumpDwords = 0x40000;   // 1 MiB of dwords — far past any real segment
            constexpr uint32_t kMaxJumpDepth  = 8;
            if (c.jump_dwords > kMaxJumpDwords || jump_depth >= kMaxJumpDepth) break;
            bool skip = false;
            uint64_t cond = 0;
            if (c.jump_pred && pred_cond_addr && !(pred_cond_addr & 7) &&
                guest_readable(pred_cond_addr, 8)) {
                memcpy(&cond, (const void*)(uintptr_t)pred_cond_addr, sizeof cond);
                skip = (cond != 0);
            }
            if (getenv("PROSPER_PREDLOG")) {
                static int logged = 0;
                if (logged++ < 96)
                    fprintf(stderr, "[pred] fold Jump target=0x%llx ndw=%u pred=%u cond@0x%llx=0x%llx -> %s\n",
                            (unsigned long long)c.jump_addr, c.jump_dwords, c.jump_pred,
                            (unsigned long long)pred_cond_addr, (unsigned long long)cond,
                            skip ? "SKIP" : "EXEC");
            }
            if (skip) break;
            if (!guest_readable(c.jump_addr, c.jump_dwords * 4)) break;   // whole segment must be mapped
            jump_depth++;
            run_command_buffer((const uint32_t*)(uintptr_t)c.jump_addr, c.jump_dwords, *this);
            jump_depth--;
            break;
        }
        case K::Flip:
            // The GPU reaching the SetFlip packet IS the flip moment: perform the videoout flip so
            // GetFlipStatus advances and the game's frame pacer sees its flipArg complete. Only for
            // a fully-decoded payload — a short packet must not fabricate a flip.
            //
            // #312 WAIT_DEFER liveness: the flip is NOT held behind an unsatisfied barrier. Every
            // WAIT_DEFER run that paused flips wedged the frame loop within seconds (the pacer
            // starves and the guest stops submitting — including the producer that would satisfy
            // the barrier). Withholding only the MEMORY writes (ReleaseMem/WriteData/DmaData) keeps
            // the corruption-relevant ordering (never show a fence value ahead of its barrier)
            // while the pacing signals flow; the failure direction becomes "label still reads 0 a
            // little longer" — the safe side of the guest's consumption poll. CONFIDENCE: MED.
            if (!c.flip_valid) break;
            if (eop_write_sync()) prosper_vo_flip_from_gpu(c.flip_handle, c.flip_bufidx, c.flip_mode, c.flip_arg);
            else pend_enqueue(c);
            break;
        default:
            break;   // events / waits / unknown: no register-state effect (handled later)
    }
}

// #1226: the submit entry point currently folding (1=SubmitDcb, 2=SubmitAcb, 3=SubmitDcbFinal).
// Set by hle_agc under g_agc_state_mu (all folds are serialized), stamped onto every decoded
// command so deferred/pended effects retain their queue of origin and barrier scheduling can keep
// Dcb/DcbFinal ordered independently from Acb.
extern "C" void prosper_gpu_set_fold_origin(uint8_t origin) { g_fold_origin = origin; }

size_t run_command_buffer(const uint32_t* buf, size_t dwords, GpuState& st) {
    // Each TOP-LEVEL stream starts a fresh fold state (#312: the pause flag and the lazily-opened
    // deferred stream are per-fold). A Jump recursion must NOT reset them: the jump target
    // executes INSIDE the paused stream, and resetting mid-stream both un-gated the parent's
    // remaining effects (ordering break) and made last_fold_deferred() read false at submit end,
    // so hle_agc never started the release watchdog — the observed WAIT_DEFER wedge with zero
    // DEFER-TIMEOUT logs (a deferred stream nobody ever re-checked while the guest CPU-polled one
    // of its labels).
    if (st.jump_depth == 0) {
        g_fold_deferring = false; g_fold_stream_open = false;
        g_fold_discard_deferred_suffix = false;
        g_fold_seq.fetch_add(1, std::memory_order_relaxed);   // #312 label-history fold id
    }
    std::vector<Pm4Command> ops;
    decode_pm4(buf, dwords, ops);
    for (auto& c : ops) {
        c.stream_order = st.command_order + 1;
        c.queue_origin = g_fold_origin;   // #1226: retained by deferred/pended effects
        st.apply(c);
    }
    return ops.size();
}

} // namespace prosper::gpu
