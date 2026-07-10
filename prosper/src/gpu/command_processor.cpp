// command_processor.cpp — see command_processor.hpp.
#include "command_processor.hpp"
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
#include <chrono>
#include <atomic>
#include <deque>
#include <unordered_set>
#include <condition_variable>
#include <thread>

namespace prosper::gpu {

// Readability probe (gpu_executor.cpp, declared in gpu_execute.hpp): page-granular check that a
// guest range is mapped, so the Jump fold below never walks an unmapped segment address.
bool guest_readable(uint64_t addr, uint32_t bytes);

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

// #312: "pointer-like" pre-content — a freed MallocBinned3 block header holds heap pointers into
// the 512 GiB arena (0x1000000000) or the allocator-metadata pools (0x2000000000 region, where the
// FPoolInfo tables live — e.g. the 0x20015f0000 pool-info table of #161/#241).
static bool ptr_like(uint64_t v) {
    return (v >= 0x1000000000ull && v < 0x1200000000ull) ||
           (v >= 0x2000000000ull && v < 0x2100000000ull);
}
// #312: report one suspicious fence write with its build-journal verdict. kindtag: "REL1" etc.
static void report_suspect_write(const char* kindtag, uint64_t addr, uint64_t value, uint64_t pre,
                                 uint64_t pkt) {
    // Skip the first 10 s: boot-time label-array allocations carry benign freelist residue that
    // burned the whole report budget at t=3.4 s in every run; the corruption class is mid-burst.
    if (now_ms() < 10000) return;
    static std::atomic<int> n{0};
    if (n.fetch_add(1) >= 96) return;
    uint64_t baddr = 0, bpre = 0, bt = 0;
    int have = prosper_fence_journal_lookup(pkt, &baddr, &bpre, &bt);
    fprintf(stderr, "[agc] SUSPECT-%s [0x%llx] pre=0x%llx value=0x%llx pkt=0x%llx t=%llums | "
                    "journal:%s built@%llums(age=%lldms) built-addr=0x%llx%s pre@build=0x%llx%s\n",
            kindtag, (unsigned long long)addr, (unsigned long long)pre, (unsigned long long)value,
            (unsigned long long)pkt, (unsigned long long)now_ms(),
            have ? "" : " MISS", (unsigned long long)bt,
            have ? (long long)(now_ms() - bt) : -1,
            (unsigned long long)baddr, (have && baddr != addr) ? " TARGET-CHANGED" : "",
            (unsigned long long)bpre, (have && ptr_like(bpre)) ? " STALE-AT-BUILD" : "");
}

// Honor a RELEASE_MEM / EVENT_WRITE_EOP completion write. data_sel (Kyty GraphicsCbReleaseMem allows {2,3};
// shadPS4 DataSelect enum): 1=write 32-bit value, 2=write 64-bit value, 3=write 64-bit GPU clock. The write
// uses memcpy so an only-4-byte-aligned 64-bit label is handled portably. CONFIDENCE: HIGH — address,
// data_sel and value are decoded directly from the packet the game's ReleaseMem call built.
static void honor_eop_write(const Pm4Command& c) {
    if (eop_writes_disabled() || !c.rel_addr || (c.rel_addr & 3)) return;
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
                  if (ptr_like(pre))
                      report_suspect_write("REL1", c.rel_addr, c.rel_value, pre, pkt_addr(c));
                  uint32_t v = (uint32_t)c.rel_value; memcpy(dst, &v, sizeof v);
                  ring_record(c.rel_addr, v, 4, 1, pkt_addr(c)); break; }
        case 2: { if (!c.rel_value_valid) return;
                  uint64_t v = c.rel_value;           memcpy(dst, &v, sizeof v);
                  ring_record(c.rel_addr, v, 8, 1, pkt_addr(c)); break; }
        case 3: { uint64_t v = gpu_clock64();         memcpy(dst, &v, sizeof v);
                  ring_record(c.rel_addr, v, 8, 1, pkt_addr(c)); break; }
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
    uint64_t v = gpu_clock64();
    memcpy((void*)(uintptr_t)c.event_addr, &v, sizeof v);
    ring_record(c.event_addr, v, 8, 2, pkt_addr(c));
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
// Execution is deliberately GATED to the immediate-fill form (srcOrImm is a 32-bit value, PM4
// CP-DMA src_sel=DATA semantics: the value is replicated across the destination). DOLL uses two
// instances of it: the 4-byte per-segment label init above, and 64 KiB zero-fills of freshly
// allocated 64 KiB-aligned command-stream CHUNKS (whose boundary headers hold the label pointers
// the consumed-marker emitter reads — a recycled chunk with a stale header is the same stomp).
// General CP-DMA copies (real src addresses / GDS) stay unexecuted with a bounded log, as before.
// CONFIDENCE: HIGH on the init form (three-callsite ABI + live page-watch protocol capture);
// MED on the large-fill form (same selectors + src=0 + fresh-buffer destinations).
static void honor_dma_data(const Pm4Command& c) {
    if (eop_writes_disabled() || !c.dd_valid) return;
    constexpr uint32_t kMaxFill = 0x1000000;   // 16 MiB — far past any observed fill
    // dst >= 0x10000: a small dst is a GDS offset (unmodeled), and the PROSPER_NULL_PAGE
    // read-only zero page would otherwise pass guest_readable() and SEGV the memset.
    bool imm_form = c.dd_bytes >= 1 && c.dd_bytes <= kMaxFill &&
                    c.dd_dst >= 0x10000 && !(c.dd_dst & 3) && c.dd_src <= 0xffffffffull;
    if (!imm_form || !guest_readable(c.dd_dst, c.dd_bytes)) {
        // Not the immediate-fill form we model (or unmapped dst) — log so the gap stays visible.
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 24)
            fprintf(stderr, "[agc] DMA_DATA not executed (unmodeled form): dst=0x%llx src=0x%llx bytes=%u sels=0x%x\n",
                    (unsigned long long)c.dd_dst, (unsigned long long)c.dd_src, c.dd_bytes, c.dd_sels);
        return;
    }
    uint32_t v32 = (uint32_t)c.dd_src;
    uint8_t* dst = (uint8_t*)(uintptr_t)c.dd_dst;
    if (v32 == 0) {
        memset(dst, 0, c.dd_bytes);
    } else {
        uint32_t i = 0;
        for (; i + 4 <= c.dd_bytes; i += 4) memcpy(dst + i, &v32, 4);
        if (i < c.dd_bytes) memcpy(dst + i, &v32, c.dd_bytes - i);
    }
    ring_record(c.dd_dst, c.dd_src, (uint8_t)(c.dd_bytes > 255 ? 255 : c.dd_bytes), 4, pkt_addr(c));
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   DmaData [0x%llx] := 0x%x (%u bytes)\n",
                (unsigned long long)c.dd_dst, v32, c.dd_bytes);
    wake_on_label(c.dd_dst);
}

// Honor a WRITE_DATA packet: copy the inline dwords to the destination address (same synchronous timing).
static void honor_write_data(const Pm4Command& c) {
    if (eop_writes_disabled() || !c.wd_addr || (c.wd_addr & 3) || !c.wd_data || !c.wd_num) return;
    // #312 stomp-catcher (same as the ReleaseMem one): a small label-init WriteData landing over
    // pointer-like memory, or targeting the allocator-metadata region, is a suspect stomp.
    if (c.wd_num <= 4) {
        uint64_t pre = 0; memcpy(&pre, (const void*)(uintptr_t)c.wd_addr, sizeof pre);
        if (ptr_like(pre))
            report_suspect_write("WDATA", c.wd_addr, c.wd_data[0], pre, pkt_addr(c));
    }
    memcpy((void*)(uintptr_t)c.wd_addr, c.wd_data, (size_t)c.wd_num * 4);
    ring_record(c.wd_addr, c.wd_data[0], (uint8_t)(c.wd_num * 4 > 255 ? 255 : c.wd_num * 4), 3, pkt_addr(c));
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   WriteData [0x%llx] %u dwords (first=0x%08x)\n",
                (unsigned long long)c.wd_addr, c.wd_num, c.wd_data[0]);
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
};
PendQueue& pend_q() { static PendQueue* p = new PendQueue; return *p; }
void apply_effect(const Pm4Command& c);   // fwd (defined with the WAIT_DEFER machinery below)
// Drain returns only when every pending write has LANDED — including one a concurrent drainer
// popped and is mid-applying (the in-flight window). Without that wait, a caller could observe an
// empty queue and proceed to apply a LATER same-address write (a released #312 gated item) that
// overtakes the in-flight one — the exact per-address inversion the barrier model exists to stop.
void pend_drain_locked(PendQueue& p, std::unique_lock<std::mutex>& lk) {
    for (;;) {
        if (!p.q.empty()) {
            PendWrite w = std::move(p.q.front());
            p.q.pop_front();
            p.inflight++;
            lk.unlock();                 // the write itself never needs the queue lock
            apply_effect(w.cmd);
            lk.lock();
            p.inflight--;
            if (p.inflight == 0) p.cv.notify_all();
            continue;
        }
        if (p.inflight == 0) break;
        p.cv.wait(lk);                   // another drainer is mid-apply — wait for it to land
    }
}
void pend_worker() {
    PendQueue& p = pend_q();
    std::unique_lock<std::mutex> lk(p.mx);
    for (;;) {
        p.cv.wait(lk, [&] { return !p.q.empty(); });
        lk.unlock();
        struct timespec ts{ 0, 1000000 };   // 1 ms modeled pipe-drain latency (matches the EOP worker)
        nanosleep(&ts, nullptr);
        lk.lock();
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
    p.cv.notify_one();
}
} // namespace

// Synchronous drain: apply every pending completion write NOW, in order. Called from the fold's
// WaitRegMem check, before the renderer executes, and by the EOP-event worker before it posts.
extern "C" void prosper_gpu_drain_completion_writes() {
    PendQueue& p = pend_q();
    std::unique_lock<std::mutex> lk(p.mx);
    pend_drain_locked(p, lk);
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
//   - prosper models ONE submission queue today: both driver entry points (sceAgcDriverSubmitDcb
//     and the w1KFAHVqpaU final-submit variant) feed the same guest gfx ring — the guest's own
//     SubmitCommandBuffers batch splits its buffer array across BOTH (RE'd at eboot+0x220a9a0),
//     so they are one stream and MUST keep mutual order. Compute rides the same Dcb
//     (DISPATCH_DIRECT inline). The only cross-queue producers are guest CPU label writes, which
//     are never gated by us.
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
//   - The gated tail releases in STRICT submission order (one FIFO), so every gated write lands
//     in ring order; it is re-checked at every subsequent submit and by hle_agc's 2 ms watchdog
//     ticker; the front barrier releases the moment it reads satisfied (a pend-queue fence write
//     from an earlier submit, a flowing later fence, or a CPU-written label — real producers
//     measured up to ~216 ms under load). Later same-queue writes to an awaited address are NOT
//     gated by the wait itself: WAIT_REG_MEM polls memory, so a later write satisfying an earlier
//     blocked wait matches hardware (the producer is another engine/queue/CPU from the ring's
//     point of view).
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
// single-queue strict-FIFO ordering (it is what the ring guarantees on hardware); MED on the
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
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct DeferItem {
    Pm4Command cmd;                    // barrier (WaitRegMem) or effect (ReleaseMem/EventWrite/WriteData/Flip)
    std::vector<uint32_t> wd_copy;     // owns a WriteData payload (cmd.wd_data repointed into this)
    uint64_t first_blocked_ms = 0;     // barrier only: when it was first found unsatisfied
};
struct DeferredStream { std::vector<DeferItem> items; size_t next = 0; };
std::vector<DeferredStream> g_deferred;   // paused queue tails; guarded by the caller's submit mutex
bool     g_fold_deferring = false;        // current fold hit an unsatisfied wait
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
// large fills (the 64 KiB DmaData chunk zero-fills) go in a range list. Entries accumulate while
// the queue tail is non-empty and are cleared when it fully drains: a partial drain can leave
// STALE entries, which only OVER-gates (an extra write joins the ordered tail — safe, order is
// still exact; the tail drains within the watchdog cadence anyway).
std::unordered_set<uint64_t> g_gated_granules;                 // addr >> 3
std::vector<std::pair<uint64_t, uint64_t>> g_gated_ranges;     // [lo, hi)
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
    if (n <= 32) {
        for (uint64_t g = a >> 3; g <= ((a + n - 1) >> 3); g++) g_gated_granules.insert(g);
    } else {
        g_gated_ranges.emplace_back(a, a + n);
    }
}
bool addr_gated(uint64_t a, uint64_t n) {
    if (!a || !n) return false;
    if (!g_gated_granules.empty())
        for (uint64_t g = a >> 3; g <= ((a + n - 1) >> 3); g++)
            if (g_gated_granules.count(g)) return true;
    for (const auto& r : g_gated_ranges)
        if (a < r.second && r.first < a + n) return true;
    return false;
}
bool effect_gated(const Pm4Command& c) {
    uint64_t a = 0, n = 0;
    effect_span(c, &a, &n);
    return addr_gated(a, n);
}

// Should this command join the gated tail? Yes when its own fold is paused (everything downstream
// of the fold's unsatisfied wait keeps stream order), or when it writes into an address domain
// that already has a gated pending write (per-address ring order across folds).
bool defer_gate(const Pm4Command& c) {
    if (g_fold_deferring) return true;
    if (g_deferred.empty()) return false;
    return effect_gated(c);
}

bool g_fold_stream_open = false;   // current top-level fold already opened its deferred stream
void defer_push(const Pm4Command& c) {
    if (!g_fold_stream_open) {     // one deferred stream per fold that defers anything
        g_deferred.emplace_back();
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
// immediately only when the gated tail is empty; otherwise it is OWED and fires when the tail
// fully drains (flush below). Liveness: flips flow regardless, the 2 ms watchdog drains the tail
// the moment its front barrier satisfies, and the timeout bounds a genuinely-stuck barrier.
// CONFIDENCE: HIGH on the hardware contract (ring-ordered interrupt), MED that pulse-on-full-
// drain (vs per-stream) is the right granularity — it only errs later, the safe direction.
namespace { uint64_t g_owed_pulses = 0; }
void submit_completion_pulse() {
    if (!g_deferred.empty()) { g_owed_pulses++; return; }
    prosper_eq_trigger_eop();
}

// Release deferred streams (call at every submit and from hle_agc's watchdog ticker, under the
// submit mutex). STRICT QUEUE FIFO (#312): the deferred streams are the paused tail of prosper's
// single submission queue — items apply strictly in submission order, and the front stream's
// blocked barrier holds everything behind it (exactly the ring order hardware guarantees;
// releasing later streams around a blocked one was measured to re-seed the corruption). A barrier
// pending > defer_timeout_ms() proceeds with the pre-#312 "dependency violated" loud log (liveness
// backstop). Returns how many streams fully completed.
int flush_deferred_streams() {
    if (g_deferred.empty()) return 0;
    // Intra-queue order: PRE-pause writes ride the 1 ms pipe-drain queue while gated writes are
    // applied here directly. Drain first so a released tail never overtakes its own head — and so
    // a producer fence still sitting in the pend queue is visible to the barrier checks below.
    prosper_gpu_drain_completion_writes();
    int completed = 0;
    for (size_t si = 0; si < g_deferred.size(); ) {
        DeferredStream& s = g_deferred[si];
        bool blocked = false;
        bool force = g_deferred.size() > kDeferMaxStreams || g_defer_items > kDeferMaxItems;
        while (s.next < s.items.size()) {
            DeferItem& it = s.items[s.next];
            if (it.cmd.kind == Pm4Command::Kind::WaitRegMem) {
                // The label page can be unmapped by the time we re-check (freed mid-defer):
                // treat as never-satisfiable and take the timeout path immediately.
                bool readable = guest_readable(it.cmd.wm_addr, 8);
                if (readable && it.first_blocked_ms && wait_regmem_satisfied(it.cmd)) {
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
                    if (!it.first_blocked_ms) it.first_blocked_ms = now;
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
        if (blocked) break;   // strict FIFO: the front barrier holds the whole gated tail
        g_defer_items -= s.items.size() < g_defer_items ? s.items.size() : g_defer_items;
        g_deferred.erase(g_deferred.begin() + si);
        completed++;
    }
    if (g_deferred.empty()) {
        // Address-domain index entries are added per gated item and only cleared here, on full
        // drain (a partial drain leaves stale entries that merely over-gate — see the index note).
        g_gated_granules.clear();
        g_gated_ranges.clear();
        // The tail is fully visible again: deliver every owed EOP pulse (one per submit whose
        // pulse was withheld while gated writes were pending — see submit_completion_pulse).
        // Distinct pulses, not coalesced (#236: a lost completion is a lost semaphore). This is
        // also the re-pulse that wakes an equeue waiter which consumed an earlier pulse and found
        // its label still gated (DOLL's "GameThread timed out waiting for RenderThread" wedge).
        uint64_t owed = g_owed_pulses;
        g_owed_pulses = 0;
        if (!owed && completed > 0) owed = 1;   // deferral began before any pulse was owed
        while (owed--) prosper_eq_trigger_eop();
    }
    return completed;
}

void GpuState::apply(const Pm4Command& c) {
    using K = Pm4Command::Kind;
    switch (c.kind) {
        case K::SetRegsIndirect: {
            if (c.regs_vaddr == 0 || c.num_regs == 0 || c.num_regs > kMaxRegsPerPacket) return;
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
            for (uint32_t i = 0; i < c.num_regs; i++) file[regs[i].offset] = regs[i].value;
            state_dirty_ = true;   // register state changed -> the next draw needs a fresh snapshot
            break;
        }
        case K::SetShRegDirect:
            // SET_SH_REG writes a consecutive RANGE (the driver uploads the whole user-data SGPR block
            // this way). Write every value, not just the first — else all descriptors past the range's
            // first register are silently dropped (which left the shaders' V#/T# SGPRs empty).
            if (getenv("PROSPER_RESDUMP")) {
                fprintf(stderr, "[shdirect] off=0x%x count=%u vals:", c.sh_reg_offset,
                        c.sh_reg_data ? c.sh_reg_count : 1u);
                if (c.sh_reg_data && c.sh_reg_count)
                    for (uint32_t k = 0; k < c.sh_reg_count && k < 40; k++)
                        fprintf(stderr, " 0x%x", c.sh_reg_data[k]);
                else fprintf(stderr, " 0x%x", c.sh_reg_value);
                fprintf(stderr, "\n");
            }
            if (c.sh_reg_data && c.sh_reg_count) {
                for (uint32_t k = 0; k < c.sh_reg_count && c.sh_reg_count <= kMaxRegsPerPacket; k++)
                    sh[c.sh_reg_offset + k] = c.sh_reg_data[k];
            } else {
                sh[c.sh_reg_offset] = c.sh_reg_value;   // fallback (single-register / legacy path)
            }
            state_dirty_ = true;
            break;
        case K::SetIndexType:
            index_type = c.index_size;
            state_dirty_ = true;
            break;
        case K::SetIndexBase:
            index_base = c.ib_addr;   // bind index-buffer base (issue #232)
            break;
        case K::SetIndexCount:
            index_num = c.index_count;   // bind index count (issue #232)
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
                last_snapshot_ = std::move(snap);
                state_dirty_ = false;
            }
            uint32_t elem = index_type ? 4u : 2u;
            Draw d;
            d.index_count = c.index_count ? c.index_count : index_num;
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
                last_snapshot_ = std::move(snap);
                state_dirty_ = false;
            }
            Draw d;
            d.index_count = c.index_count;
            d.state = last_snapshot_;
            if (c.kind == K::DrawIndex) {
                // Mark as indexed only when the packet was fully decoded — a short packet's addr/
                // modifier would be fabricated zeros, and `indexed` promises index_addr is real.
                d.indexed = c.di_valid;
                d.index_addr = c.di_index_addr;
                d.modifier = c.di_modifier;
            }
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
            if (eop_write_sync()) honor_eop_write(c); else pend_enqueue(c);
            break;
        case K::WriteData:
            if (defer_gate(c)) { defer_push(c); break; }
            if (eop_write_sync()) honor_write_data(c); else pend_enqueue(c);
            break;
        case K::EventWrite:
            if (defer_gate(c)) { defer_push(c); break; }
            if (eop_write_sync()) honor_event_write(c); else pend_enqueue(c);
            break;
        case K::DmaData:
            // CP-DMA memory effect (#312: the per-segment fence-label INIT). Rides the same FIFO
            // as the other guest-visible writes: stream order between a generation's ReleaseMem
            // (label <- 1) and the NEXT generation's DmaData (label := 0) at the same recycled
            // address is exactly what the guest's consumption poll depends on.
            if (defer_gate(c)) { defer_push(c); break; }
            if (eop_write_sync()) honor_dma_data(c); else pend_enqueue(c);
            break;
        case K::WaitRegMem:
            // Real WAIT_REG_MEM semantics (#312): an unsatisfied wait PAUSES this queue — the
            // stream's remaining memory effects are deferred until the condition holds (flushed
            // at subsequent submits by flush_deferred_streams; loud timeout fallback preserves
            // liveness). A satisfied wait is a no-op, as before.
            if (c.wm_valid && c.wm_addr && !(c.wm_addr & 3)) {
                // Own fold already paused — the wait keeps stream order. OR: the awaited address
                // has a GATED PENDING WRITE (an earlier-in-ring write it must observe): evaluating
                // now against the stale value could wrong-satisfy — join the tail in ring order.
                // (The rest of this fold's effects then gate too: stream order.)
                if (g_fold_deferring || (!g_deferred.empty() && addr_gated(c.wm_addr, 8))) {
                    g_fold_deferring = true;
                    defer_push(c);
                    break;
                }
                // A prior submit's fence write may still sit in the pipe-drain queue — a consumer's
                // wait must see it (data dependency): drain before evaluating.
                prosper_gpu_drain_completion_writes();
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
                        fprintf(stderr, "[agc] WaitRegMem #%d NOT satisfied at fold time: [0x%llx]&0x%llx = 0x%llx, func=%u ref=0x%llx — %s | built@%llums(age=%lldms)%s pre@build=0x%llx%s%s\n",
                                ln, (unsigned long long)c.wm_addr, (unsigned long long)c.wm_mask,
                                (unsigned long long)(mem & c.wm_mask), c.wm_func, (unsigned long long)c.wm_ref,
                                defer_enabled() ? "pausing queue (deferred effects)" : "dependency violated",
                                (unsigned long long)bt, have ? (long long)(now_ms() - bt) : -1,
                                (have && baddr != c.wm_addr) ? " TARGET-CHANGED" : "",
                                (unsigned long long)bpre, ptr_like(mem) ? " CONTENT-PTR-LIKE(freed?)" : "",
                                label_readable ? "" : " LABEL-UNMAPPED");
                    }
                    if (defer_enabled()) {
                        g_fold_deferring = true;
                        defer_push(c);
                    }
                }
            }
            break;
        case K::DispatchDirect:
            // Compute dispatch (issue #213 — DOLL's UE4 compute prologue). Recorded for stats/
            // diagnostics only: prosper has no compute execution path yet. The fence cluster the
            // guest builds around each dispatch (label init + EOP write + wait) completes at fold
            // time independently of the dispatch itself, so skipping the shader work cannot hang
            // the stream — it only leaves compute-written buffers stale (surfaced by the counter).
            dispatch_count++;
            break;
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

size_t run_command_buffer(const uint32_t* buf, size_t dwords, GpuState& st) {
    // Each TOP-LEVEL stream starts a fresh fold state (#312: the pause flag and the lazily-opened
    // deferred stream are per-fold). A Jump recursion must NOT reset them: the jump target
    // executes INSIDE the paused stream, and resetting mid-stream both un-gated the parent's
    // remaining effects (ordering break) and made last_fold_deferred() read false at submit end,
    // so hle_agc never started the release watchdog — the observed WAIT_DEFER wedge with zero
    // DEFER-TIMEOUT logs (a deferred stream nobody ever re-checked while the guest CPU-polled one
    // of its labels).
    if (st.jump_depth == 0) { g_fold_deferring = false; g_fold_stream_open = false; }
    std::vector<Pm4Command> ops;
    decode_pm4(buf, dwords, ops);
    for (const auto& c : ops) st.apply(c);
    return ops.size();
}

} // namespace prosper::gpu
