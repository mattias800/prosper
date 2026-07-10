// command_processor.cpp — see command_processor.hpp.
#include "command_processor.hpp"
#include "hle/sync_futex.hpp"   // wake_label_waiters (shared with sceKernelWaitOnAddress's futex)

// hle_graphics.cpp: perform the videoout flip for an in-stream SetFlip packet — advances the flip
// status (count/flipArg/currentBuffer) that sceVideoOutGetFlipStatus reports, exactly like the API
// flip does. The game's frame pacer polls that status for its submitted flipArg; a dropped in-stream
// flip stalls the frame loop at one rendered frame.
extern "C" void prosper_vo_flip_from_gpu(uint32_t handle, int32_t bufidx, uint32_t flip_mode, int64_t flip_arg);
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <atomic>
#include <deque>
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
// 1=RELEASE_MEM 2=EVENT_WRITE 3=WRITE_DATA.
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
                  // act, with the packet address for attribution. Diagnostic only, always-on
                  // (one 8-byte read per fence write), bounded to 64 reports.
                  uint64_t pre = 0; memcpy(&pre, dst, sizeof pre);
                  if (pre >= 0x1000000000ull && pre < 0x1200000000ull) {
                      static std::atomic<int> n{0};
                      if (n.fetch_add(1) < 64)
                          fprintf(stderr, "[agc] EOP-WRITE-OVER-POINTER [0x%llx] pre=0x%llx value=0x%llx pkt=0x%llx\n",
                                  (unsigned long long)c.rel_addr, (unsigned long long)pre,
                                  (unsigned long long)c.rel_value, (unsigned long long)pkt_addr(c));
                  }
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

// Honor a WRITE_DATA packet: copy the inline dwords to the destination address (same synchronous timing).
static void honor_write_data(const Pm4Command& c) {
    if (eop_writes_disabled() || !c.wd_addr || (c.wd_addr & 3) || !c.wd_data || !c.wd_num) return;
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
// writes fences from its GPU thread, never inside the submit call); MED that this closes ALL of
// #312 (the cross-queue wait-ordering approximation remains, gated separately behind
// PROSPER_WAIT_DEFER).
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
};
PendQueue& pend_q() { static PendQueue* p = new PendQueue; return *p; }
void apply_effect(const Pm4Command& c);   // fwd (defined with the WAIT_DEFER machinery below)
void pend_drain_locked(PendQueue& p, std::unique_lock<std::mutex>& lk) {
    while (!p.q.empty()) {
        PendWrite w = std::move(p.q.front());
        p.q.pop_front();
        lk.unlock();                     // the write itself never needs the queue lock
        apply_effect(w.cmd);
        lk.lock();
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

// --- WAIT_REG_MEM queue-pause semantics (issue #312 heap-corruption root cause). -----------------
//
// On real hardware WAIT_REG_MEM BLOCKS the queue until its condition holds; packets after it —
// including the RELEASE_MEM fence the game polls — execute only once the dependency is satisfied.
// Our fold is synchronous at submit time, and DOLL's UE4 RHI submits from two guest threads, so a
// consumer Dcb (wait label==1, then EOP fence write) routinely arrives BEFORE its producer (the
// Dcb whose RELEASE_MEM writes that label). The old fold logged "dependency violated" and barreled
// on: the consumer's fence completed early, the game freed the heap block holding the transient
// semaphore label, and the producer's later RELEASE_MEM value-1 write landed on the freed
// MallocBinned3 FFreeBlock header — the "Canary was 0x3, should be 0x1" / "free an unrecognized
// block 0x1000000001" fatals (a 4-byte 0x1 stomp over NextFreeBlock; attributed live by the GPU
// write-ring: 13 ReleaseMem value-1 writes at exactly the corrupted qword's address).
//
// Honest semantics with a synchronous fold: when a fold hits an UNSATISFIED wait, the rest of that
// stream's guest-visible memory effects (ReleaseMem / EVENT_WRITE / WRITE_DATA / Flip and any
// further waits) are DEFERRED in order. Deferred streams are flushed FIFO at every subsequent
// submit (the producer's own fold is what satisfies the barrier; a CPU-written label is re-checked
// at the same points), and the submit's EOP equeue pulse fires only when its stream fully flushes.
// Draws/register state still fold immediately — they touch no guest memory, so they cannot corrupt
// (worst case is the same rendering-order approximation as before). Liveness: a barrier pending
// longer than kDeferTimeoutMs falls back to the old proceed-with-loud-log behavior, so a genuinely
// never-written label (or a producer we fail to model) degrades to today's state instead of
// wedging the queue. All of this runs under the caller's submit mutex (hle_agc g_agc_state_mu).
// CONFIDENCE: HIGH on the mechanism + the wait semantics; MED on cross-stream write ordering while
// a stream is deferred (independent queues on real HW run concurrently, so a later submit's
// effects landing before a paused stream's tail matches hardware; same-queue back-to-back submits
// would be reordered, which hardware forbids — accepted, logged, and bounded by the timeout).
namespace {
bool wait_regmem_satisfied(const Pm4Command& c) {
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
std::vector<DeferredStream> g_deferred;   // FIFO; guarded by the caller's submit mutex
bool     g_fold_deferring = false;        // current fold hit an unsatisfied wait
uint64_t g_defer_streams = 0, g_defer_timeouts = 0;
constexpr uint64_t kDeferTimeoutMs = 500;
constexpr size_t   kDeferMaxStreams = 256;   // runaway guard: force-flush oldest beyond this

void defer_push(const Pm4Command& c) {
    DeferItem it; it.cmd = c;
    if (c.kind == Pm4Command::Kind::WriteData && c.wd_data && c.wd_num) {
        it.wd_copy.assign(c.wd_data, c.wd_data + c.wd_num);   // cb memory may be recycled before flush
        it.cmd.wd_data = it.wd_copy.data();
    }
    g_deferred.back().items.push_back(std::move(it));
}
void apply_effect(const Pm4Command& c) {
    using K = Pm4Command::Kind;
    switch (c.kind) {
        case K::ReleaseMem: honor_eop_write(c); break;
        case K::EventWrite: honor_event_write(c); break;
        case K::WriteData:  honor_write_data(c); break;
        case K::Flip:       if (c.flip_valid) prosper_vo_flip_from_gpu(c.flip_handle, c.flip_bufidx,
                                                                       c.flip_mode, c.flip_arg); break;
        default: break;
    }
}
} // namespace

bool last_fold_deferred() { return g_fold_deferring; }

// Flush deferred streams FIFO. Returns how many streams COMPLETED (the caller owes one EOP equeue
// pulse per completed stream). A stream whose next barrier is still unsatisfied blocks the flush
// (strict FIFO among deferred effects) unless it has been pending > kDeferTimeoutMs, in which case
// it proceeds with the pre-#312 "dependency violated" loud log.
int flush_deferred_streams() {
    int completed = 0;
    while (!g_deferred.empty()) {
        DeferredStream& s = g_deferred.front();
        bool blocked = false;
        bool force = g_deferred.size() > kDeferMaxStreams;
        while (s.next < s.items.size()) {
            DeferItem& it = s.items[s.next];
            if (it.cmd.kind == Pm4Command::Kind::WaitRegMem) {
                if (!wait_regmem_satisfied(it.cmd)) {
                    uint64_t now = defer_now_ms();
                    if (!it.first_blocked_ms) it.first_blocked_ms = now;
                    if (!force && now - it.first_blocked_ms < kDeferTimeoutMs) { blocked = true; break; }
                    g_defer_timeouts++;
                    fprintf(stderr, "[agc] WaitRegMem DEFER TIMEOUT after %llums: [0x%llx]&0x%llx func=%u ref=0x%llx — dependency violated (proceeding)\n",
                            (unsigned long long)(now - it.first_blocked_ms),
                            (unsigned long long)it.cmd.wm_addr, (unsigned long long)it.cmd.wm_mask,
                            it.cmd.wm_func, (unsigned long long)it.cmd.wm_ref);
                }
                s.next++;
                continue;
            }
            apply_effect(it.cmd);
            s.next++;
        }
        if (blocked) break;
        g_deferred.erase(g_deferred.begin());
        completed++;
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
            // EOP completion label write. Once this stream hit an unsatisfied wait, the write is
            // deferred behind that barrier (see the #312 block above) — completing the fence
            // early is what let the game free live label memory. Otherwise it goes through the
            // pipe-drain queue: completion becomes guest-visible only after the submit returns.
            if (g_fold_deferring) { defer_push(c); break; }
            if (eop_write_sync()) honor_eop_write(c); else pend_enqueue(c);
            break;
        case K::WriteData:
            if (g_fold_deferring) { defer_push(c); break; }
            if (eop_write_sync()) honor_write_data(c); else pend_enqueue(c);
            break;
        case K::EventWrite:
            if (g_fold_deferring) { defer_push(c); break; }
            if (eop_write_sync()) honor_event_write(c); else pend_enqueue(c);
            break;
        case K::WaitRegMem:
            // Real WAIT_REG_MEM semantics (#312): an unsatisfied wait PAUSES this queue — the
            // stream's remaining memory effects are deferred until the condition holds (flushed
            // at subsequent submits by flush_deferred_streams; loud timeout fallback preserves
            // liveness). A satisfied wait is a no-op, as before.
            if (c.wm_valid && c.wm_addr && !(c.wm_addr & 3)) {
                if (g_fold_deferring) { defer_push(c); break; }   // ordered behind the first barrier
                // A prior submit's fence write may still sit in the pipe-drain queue — a consumer's
                // wait must see it (data dependency): drain before evaluating.
                prosper_gpu_drain_completion_writes();
                if (!wait_regmem_satisfied(c)) {
                    uint64_t mem = 0; memcpy(&mem, (const void*)(uintptr_t)c.wm_addr, sizeof mem);
                    // PROSPER_WAIT_DEFER=1 (experimental): pause the queue — defer the stream's
                    // remaining memory effects until the condition holds. Default OFF: label
                    // slots are recycled with residue frequently equal to the awaited value, so
                    // value-based gating cannot distinguish a satisfied wait from stale residue
                    // (measured live: enabling it moved the #312 fatal EARLIER). Kept for
                    // experiments while the real per-queue model is built.
                    static const bool defer_on = [] {
                        const char* e = getenv("PROSPER_WAIT_DEFER");
                        return e && strtol(e, nullptr, 0) != 0; }();
                    fprintf(stderr, "[agc] WaitRegMem NOT satisfied at fold time: [0x%llx]&0x%llx = 0x%llx, func=%u ref=0x%llx — %s\n",
                            (unsigned long long)c.wm_addr, (unsigned long long)c.wm_mask,
                            (unsigned long long)(mem & c.wm_mask), c.wm_func, (unsigned long long)c.wm_ref,
                            defer_on ? "pausing queue (deferred effects)" : "dependency violated");
                    if (defer_on) {
                        g_fold_deferring = true;
                        g_defer_streams++;
                        g_deferred.emplace_back();
                        DeferItem it; it.cmd = c; it.first_blocked_ms = defer_now_ms();
                        g_deferred.back().items.push_back(std::move(it));
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
            // a fully-decoded payload — a short packet must not fabricate a flip. Behind an
            // unsatisfied wait the flip defers with the other effects (#312); otherwise it rides
            // the pipe-drain queue like every completion signal (a flip is a lifecycle signal —
            // the game recycles display buffers on it — and must not be observable mid-submit).
            if (!c.flip_valid) break;
            if (g_fold_deferring) { defer_push(c); break; }
            if (eop_write_sync()) prosper_vo_flip_from_gpu(c.flip_handle, c.flip_bufidx, c.flip_mode, c.flip_arg);
            else pend_enqueue(c);
            break;
        default:
            break;   // events / waits / unknown: no register-state effect (handled later)
    }
}

size_t run_command_buffer(const uint32_t* buf, size_t dwords, GpuState& st) {
    g_fold_deferring = false;   // each stream starts unpaused (#312 queue-pause state is per-fold)
    std::vector<Pm4Command> ops;
    decode_pm4(buf, dwords, ops);
    for (const auto& c : ops) st.apply(c);
    return ops.size();
}

} // namespace prosper::gpu
