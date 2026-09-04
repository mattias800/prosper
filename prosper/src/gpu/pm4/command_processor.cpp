// command_processor.cpp — see command_processor.hpp.
#include "gpu/pm4/command_processor.hpp"
#include "hle/memory/guest_memory_topology.hpp"
#include "gpu/diagnostics/diag_ratelimit.hpp"   // #1761: single-sourced ordinal + sparse-tail rule for capped logs
#include "gpu/execute/mb3_freelist.hpp"
#include "diagnostics/env_numeric.hpp"   // #3267: a typo must not switch a default-ON guard off
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/capture/writer_provenance.hpp"
#include "hle/sync/sync_futex.hpp"   // wake_label_waiters (shared with sceKernelWaitOnAddress's futex)

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
#include <mutex>
#include <unordered_set>
#include <condition_variable>
#include <thread>

namespace prosper::gpu {

// High 32 bits of the guest GPU VA aperture, learned from any full indirect-args base. One address
// space per process, so this is process state rather than per-fold state.
std::atomic<uint32_t> g_indirect_va_aperture{0};

// Opt-in gate for the indirect-argument aperture recovery below. Read once: this sits on the packet
// decode path, and #2214 is the standing lesson about per-operation getenv in a hot loop.
bool aperture_recovery_enabled() {
    static const bool enabled = [] {
        const char* spec = getenv("PROSPER_INDIRECT_APERTURE_RECOVERY");
        return spec && *spec && *spec != '0';
    }();
    return enabled;
}


std::vector<RegWatchEntry> parse_reg_watch(const char* setting) {
    std::vector<RegWatchEntry> entries;
    if (!setting || !*setting) return entries;
    const std::string text(setting);
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        std::string item = text.substr(start, comma == std::string::npos ? std::string::npos
                                                                         : comma - start);
        start = comma == std::string::npos ? text.size() + 1 : comma + 1;
        // Trim surrounding spaces so a human-written list stays forgiving.
        while (!item.empty() && std::isspace(static_cast<unsigned char>(item.front())))
            item.erase(item.begin());
        while (!item.empty() && std::isspace(static_cast<unsigned char>(item.back())))
            item.pop_back();
        if (item.empty()) continue;
        RegWatchEntry entry;
        const size_t colon = item.find(':');
        if (colon != std::string::npos) {
            const std::string cls = item.substr(0, colon);
            if (cls == "Cx" || cls == "cx") entry.reg_class = RegClass::Cx;
            else if (cls == "Sh" || cls == "sh") entry.reg_class = RegClass::Sh;
            else if (cls == "Uc" || cls == "uc") entry.reg_class = RegClass::Uc;
            else continue;   // unknown class: skip this entry, keep the rest of the list
            item = item.substr(colon + 1);
            if (item.empty()) continue;
        }
        if (item == "*") {
            // Whole-class watch; only meaningful with an explicit class prefix (a bare "*" would
            // not say which file to watch, so it is skipped like any other unparsable entry).
            if (colon == std::string::npos) continue;
            entry.all_offsets = true;
            if (std::find(entries.begin(), entries.end(), entry) == entries.end())
                entries.push_back(entry);
            continue;
        }
        char* end = nullptr;
        errno = 0;
        const unsigned long parsed = std::strtoul(item.c_str(), &end, 0);
        if (errno || !end || *end || parsed > 0xFFFFFFFFul) continue;
        entry.offset = static_cast<uint32_t>(parsed);
        if (std::find(entries.begin(), entries.end(), entry) == entries.end())
            entries.push_back(entry);
    }
    return entries;
}

namespace {
const std::vector<RegWatchEntry>& reg_watch_entries() {
    static const std::vector<RegWatchEntry> entries =
        parse_reg_watch(std::getenv("PROSPER_REGWATCH"));
    return entries;
}

// One line per watched write. `values`/`count` describe a direct multi-register span so a watched
// offset inside a larger SET_*_REG range still reports the dword that actually landed on it.
void reg_watch_report(RegClass reg_class, uint32_t offset, uint32_t value, uint32_t count,
                      const uint32_t* values, const char* path, uint64_t command_order) {
    const auto& entries = reg_watch_entries();
    if (entries.empty()) return;
    for (const auto& entry : entries) {
        if (entry.reg_class != reg_class) continue;
        // A whole-class watch reports every offset of the reported span; a single-offset watch
        // reports only its own dword within that span.
        const uint32_t span = count ? count : 1u;
        const uint32_t first = entry.all_offsets ? 0u : entry.offset - offset;
        if (!entry.all_offsets &&
            (entry.offset < offset || entry.offset >= offset + span)) continue;
        for (uint32_t index = first; index < (entry.all_offsets ? span : first + 1u); ++index) {
            const uint32_t written = values && index < count ? values[index] : value;
            static std::atomic<int> emitted{0};
            if (emitted.fetch_add(1) >= 400000) return;
            const char* cn = reg_class == RegClass::Cx ? "Cx"
                           : reg_class == RegClass::Sh ? "Sh" : "Uc";
            std::fprintf(stderr,
                         "[regwatch] class=%s off=0x%x val=0x%08x path=%s span=0x%x+%u order=%llu\n",
                         cn, offset + index, written, path, offset, count,
                         static_cast<unsigned long long>(command_order));
        }
    }
}
}  // namespace

// Enable per-SH-register write provenance recording for #305 or selected capture #1853.
bool udprov_enabled() {
    // #1853: a selected resource-input witness needs this state from the first SH write, before the
    // capture candidate is reached. The selector is process-start configuration just like UDPROV;
    // enabling it records the maps without enabling RESDUMP/UDPROV's high-volume live output.
    static const bool on = std::getenv("PROSPER_UDPROV") != nullptr ||
        std::getenv("PROSPER_GPU_CAPTURE_RESOURCE_PROVENANCE") != nullptr;
    return on;
}

namespace {
// Test-only fault injection for the #1853 provenance collector.  Keep the public activation
// predicate true so the mutation exercises the selected-witness consumer, but suppress both the
// write maps and their draw snapshot.  This is intentionally queried dynamically: the focused
// regression arms it only for its safe cross-thread fixture after the older .at()-based provenance
// checks have completed.
bool udprov_collection_enabled() {
    return udprov_enabled() &&
        std::getenv("PROSPER_TEST_DISABLE_UDPROV_COLLECTION") == nullptr;
}
} // namespace

// Readability probe (gpu_executor.cpp, declared in gpu_execute.hpp): page-granular check that a
// guest range is mapped, so the Jump fold below never walks an unmapped segment address.
bool guest_readable(uint64_t addr, uint32_t bytes);
bool guest_writable(uint64_t addr, uint32_t bytes);
// Guest GPU writes invalidate renderer-owned copies of overlapping resources.
void notify_guest_gpu_write(uint64_t addr, uint64_t size);
// Names the PM4 packet responsible for the next notify, for PROSPER_GUEST_WRITE_WATCH.
void set_guest_gpu_write_origin(const char* origin);
// The 64 KiB Global Data Share (gpu_executor.cpp, declared in gpu_execute.hpp). DMA_DATA can name a
// GDS offset rather than a guest address as its destination, and shaders reach the same backing.
uint8_t* compute_gds_backing();
size_t compute_gds_size();

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

// --- #1226 GPU-write VALUE trap (PROSPER_WRITE_TRAP=0xV[,0xV...]; default OFF). ------------------
// The address-keyed instruments (PROSPER_PROVENANCE_ADDR, PROSPER_HWWATCH, the poolshift span scan)
// all need the destination up front. A corruption whose POISON VALUE is invariant across runs while
// its address is not — ArcRunner's free-list head reads a constant 0x30016000 at three different
// per-thread pool pages — has no such handle, so the question "does a prosper PM4 write path store
// this dword into guest memory?" was unanswerable. This traps on the payload instead: every
// completion/label/DMA/WRITE_DATA store THIS COMMAND PROCESSOR performs is checked against the
// listed dwords and reported with its destination, the destination's pre-content and the packet
// builder address.
//
// SCOPE, because a null result is the point: this sees the PM4 write paths only. Compute-dispatch
// writeback, the Vulkan backend's target readback, and every HLE that writes guest memory are
// INVISIBLE to it, so zero hits means "no PM4 write path created that value", never "prosper did
// not". The scan is also 4-byte-strided from each payload's base — correct for REL/EVENT/WDATA/
// DMA-immediate payloads, which are dword-aligned by construction, but a DMA COPY is a raw byte
// move, so a poison lying at a source offset that is not a multiple of 4 is not seen. That is
// exactly the shape of an off-by-one store, so a null DMA-copy result does not refute one.
constexpr unsigned kWriteTrapMax = 8;
struct WriteTrap { uint32_t v[kWriteTrapMax]; unsigned n = 0; };
const WriteTrap& write_trap() {
    static const WriteTrap t = [] {
        WriteTrap w;
        const char* e = getenv("PROSPER_WRITE_TRAP");
        if (!e) return w;                              // unset: the trap is simply not in use
        // Parse loudly. This instrument's headline result is "zero hits", so a silently disarmed
        // trap reads exactly like a hard negative — the worst possible failure mode for it. An
        // EMPTY value is the likeliest way to get there by accident (PROSPER_WRITE_TRAP=$UNSET),
        // so it is reported rather than treated like "unset".
        if (!*e) {
            fprintf(stderr, "[write-trap] NOT ARMED — PROSPER_WRITE_TRAP is set but empty\n");
            return w;
        }
        const char* p = e;
        bool bad = false;
        while (*p) {
            char* end = nullptr;
            // Reject a bare sign or a stray radix prefix before strtoull's lenient reading of them
            // turns `0x` (a truncated paste) into the value 0 — the noisiest possible trap value.
            if (!(*p >= '0' && *p <= '9')) { bad = true; break; }
            const unsigned long long v = strtoull(p, &end, 0);
            if (end == p || (end[0] && end[0] != ',')) { bad = true; break; }
            if (v > 0xffffffffull) {
                fprintf(stderr, "[write-trap] REFUSED 0x%llx: values are 32-bit dwords\n",
                        (unsigned long long)v);
                bad = true;
            } else if (w.n == kWriteTrapMax) {
                fprintf(stderr, "[write-trap] REFUSED 0x%llx: at most %u values\n",
                        (unsigned long long)v, kWriteTrapMax);
                bad = true;
            } else {
                // A duplicate would silently never advance its own counter (write_trap_index
                // returns the first match), so drop it visibly rather than arming a dead slot.
                bool dup = false;
                for (unsigned i = 0; i < w.n; i++) dup = dup || w.v[i] == (uint32_t)v;
                if (dup) fprintf(stderr, "[write-trap] IGNORED duplicate 0x%llx\n",
                                 (unsigned long long)v);
                else     w.v[w.n++] = (uint32_t)v;
            }
            if (*end != ',') { p = end; break; }
            p = end + 1;
        }
        if (*p) { fprintf(stderr, "[write-trap] REFUSED trailing '%s'\n", p); bad = true; }
        if (!w.n) {
            fprintf(stderr, "[write-trap] NOT ARMED — PROSPER_WRITE_TRAP='%s' parsed to no values\n", e);
            return w;
        }
        fprintf(stderr, "[write-trap] armed on %u value(s):", w.n);
        for (unsigned i = 0; i < w.n; i++) fprintf(stderr, " 0x%x", w.v[i]);
        fprintf(stderr, "%s\n", bad ? "  (SOME INPUT REFUSED — see above)" : "");
        return w;
    }();
    return t;
}
inline bool write_trap_armed() { return write_trap().n != 0; }
// Index of `v` in the armed set, or -1. Per-VALUE identity, because the counters and the print
// budget below are per value.
inline int write_trap_index(uint32_t v) {
    const WriteTrap& t = write_trap();
    for (unsigned i = 0; i < t.n; i++) if (t.v[i] == v) return (int)i;
    return -1;
}
// Per-value match totals, and one grand total for the test hook. UNCAPPED — the printing is not.
std::atomic<uint64_t> g_write_trap_matches{0};
std::atomic<uint64_t> g_write_trap_per_value[kWriteTrapMax];
void write_trap_report(const char* kind, uint64_t dst, uint64_t pre, uint32_t v, int idx,
                       uint32_t byte_off, uint64_t pkt) {
    g_write_trap_matches.fetch_add(1, std::memory_order_relaxed);
    const uint64_t ord = g_write_trap_per_value[idx & (kWriteTrapMax - 1)]
                             .fetch_add(1, std::memory_order_relaxed) + 1;
    // Budget PER VALUE, ordinal on every line, and keep printing at powers of two past the cap.
    // A shared budget lets a common positive-control value (a fence `0x1`) exhaust the log before
    // the value actually under investigation ever gets a line — which would turn this instrument's
    // own recommended workflow into a silent false negative. And a bare cap makes "64 hits" or
    // "256 hits" read as a count: that is the #1226 trap this file exists to stop repeating.
    if (diag_should_print(ord))
        fprintf(stderr,
                "[agc] WRITE-TRAP #%llu[val=0x%x] kind=%s dst=0x%llx+%u pre=0x%llx pkt=0x%llx t=%llums\n",
                (unsigned long long)ord, v, kind, (unsigned long long)dst, byte_off,
                (unsigned long long)pre, (unsigned long long)pkt, (unsigned long long)now_ms());
}
// Scan a payload the command processor is about to store. `pre` is the destination's current qword
// (0 when the caller has not read it); `bytes` is bounded by the caller.
void write_trap_scan(const char* kind, uint64_t dst, uint64_t pre, const void* payload,
                     uint64_t bytes, uint64_t pkt) {
    if (!write_trap_armed() || !payload) return;
    const uint8_t* p = (const uint8_t*)payload;
    for (uint64_t off = 0; off + 4 <= bytes; off += 4) {
        uint32_t v; memcpy(&v, p + off, 4);
        if (const int idx = write_trap_index(v); idx >= 0)
            write_trap_report(kind, dst, pre, v, idx, (uint32_t)off, pkt);
    }
}

// Unit-test fault injection for the only fallible store after dma_data_form has accepted a direct
// destination. This proves provenance is attributed after a successful device write, not merely
// after validation. Production never arms it.
std::atomic<bool> g_dma_backing_write_fail_once_for_test{false};
bool dma_backing_write(uint64_t destination, const void* source, size_t bytes) {
    if (g_dma_backing_write_fail_once_for_test.exchange(false, std::memory_order_acq_rel))
        return false;
    return prosper::guest_memory_gpu_write(destination, source, bytes);
}
}
extern "C" void prosper_dma_backing_write_fail_once_for_test() {
    g_dma_backing_write_fail_once_for_test.store(true, std::memory_order_release);
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
// #1226 (arc7): the fold counter lives HERE rather than beside the label ring below, because the
// build journal now records the fold a packet was built in. Everything that reads it is further
// down the file; the move is a declaration order change and nothing else.
std::atomic<uint32_t> g_fold_seq{0};                  // top-level fold counter (submit streams)
// `fold` is the value of g_fold_seq when the GUEST built this packet. It is what makes the
// build->exec distance expressible in FOLDS rather than only in milliseconds: a wall-clock age is
// not comparable between a route that adds a fixed delay per submit and one that does not, because
// the delay changes how many submits fit in a millisecond. See fold_margin_* below.
struct FenceBuildRec { uint64_t pkt, addr, pre; uint64_t t_ms; uint32_t fold; };
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
    r.fold = g_fold_seq.load(std::memory_order_relaxed);
}
extern "C" int prosper_fence_journal_lookup(uint64_t pkt, uint64_t* addr, uint64_t* pre, uint64_t* t_ms,
                                            uint32_t* fold) {
    if (!pkt) return 0;
    const FenceBuildRec& r = g_fence_journal[journal_slot(pkt)];
    if (r.pkt != pkt) return 0;
    *addr = r.addr; *pre = r.pre; *t_ms = r.t_ms;
    if (fold) *fold = r.fold;
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
    // #1226 (arc7): the last build/exec fold for this label, kept alongside the ring so the
    // fold_margin census never has to SCAN the ring (a scan would read entries the 16-slot ring may
    // already have overwritten, which is precisely the population a busy label falls into).
    // kNoFold = "not yet seen"; both are reset with the counters on a slot collision.
    uint32_t last_build_fold, last_exec_fold;
    LabelEvent ev[16];                    // last 16 events
};
constexpr uint32_t kNoFold = 0xFFFFFFFFu;
constexpr uint32_t kLabelHistSize = 16384;            // power of two
LabelHist g_label_hist[kLabelHistSize];
// g_fold_seq is defined above, next to the build journal that now records it.
inline uint32_t label_hist_index(uint64_t addr) {
    return (uint32_t)((addr >> 2) * 2654435761u) & (kLabelHistSize - 1);
}
// #1226: read-only lookup — nullptr when this address does not currently own its slot. REPORTING
// paths must use this, never label_hist_slot(): the slot getter RESETS a colliding entry, so a
// diagnostic that merely prints a label's history would evict a DIFFERENT label's protocol state,
// silently turning label_is_consumed_marker() false for it and making every #312 guard inert on
// that label's next write. A tripwire must not be able to change what the guards decide.
inline const LabelHist* label_hist_find(uint64_t addr) {
    const LabelHist& h = g_label_hist[label_hist_index(addr)];
    return h.addr == addr ? &h : nullptr;
}
inline LabelHist& label_hist_slot(uint64_t addr) {
    LabelHist& h = g_label_hist[label_hist_index(addr)];
    if (h.addr != addr) {                              // collision/new: reset (diagnostic-grade)
        h.addr = addr; h.n.store(0, std::memory_order_relaxed);
        h.dma_built_n = h.dma_exec_n = h.rel_built_n = h.rel_exec_n = 0;
        h.mb3_dma_suppressed_n = h.mb3_rel_debt_consumed_n = 0;
        h.last_build_fold = h.last_exec_fold = kNoFold;
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

// --- #1226 (arc7) PROSPER_FOLD_MARGIN: the per-fold account of the recycle race. ------------------
//
// Everything this title's investigation has measured about the submit throttle is a WHOLE-RUN
// aggregate, and two such rows have already been retracted (`ARCRUNNER_STATUS.md` § Ruled out):
// "the throttle shortens the guest's build-to-submit interval" and "the throttle changes how fast
// the guest runs". Both failed the same way — a rate over a denominator the two arms do not share.
// The mechanism, though, is stated in FOLDS: *build at fold f, exec at f+5 or f+6, rebuild at f+6;
// when the exec lands on f+6 the two collide.* A count of folds is immune to the denominator
// problem, because it is the same unit in which the collision is defined.
//
// So this census reports, in folds, the two distances that bracket the race, one measured at the
// GUEST's action and one at PROSPER's:
//
//   BUILD side, at `prosper_label_hist_dma_built` (the guest's own AGC builder call):
//     * margin  = fold_now - last_exec_fold  — how many folds prosper's execution of the PREVIOUS
//       generation preceded this rebuild by. This is the safety margin, in the unit the mechanism
//       is stated in. It is undefined for a label's first build (reported separately as no-exec).
//     * REBUILD-BEFORE-EXEC: dma_built_n > dma_exec_n at build time, i.e. the guest is building a
//       new generation while prosper has not executed the previous one at all. This is the exact
//       precondition for a late write landing in a block the guest has since recycled, and it is
//       the build-side twin of `label_rel_overlap()` (fence side) and of DMA-INIT-GEN's `depth>=2`
//       (exec side). Having all three lets a claim be cross-checked from three independent points.
//
//   EXEC side, at the 4-byte label init prosper executes:
//     * age_folds = fold_now - build_fold, from the build journal — the build->exec distance, in
//       folds. Its milliseconds twin is already reported by DMA-INIT-GEN and is exactly the figure
//       that could not discriminate: ~345 ms in BOTH arms. If the throttle changes this distance in
//       folds while leaving it unchanged in milliseconds, that is the per-fold account, and it is
//       also arithmetically forced — a fixed delay per submit means fewer submits fit in 345 ms.
//
// Log-only. It gates nothing, allocates nothing, and adds one relaxed load plus a few adds per
// label event. Default OFF, and its arms must be scored against the same endpoint as any other:
// instrument trap 104 records that a read-only probe's DURATION can decide whether this failure
// happens at all, so a FOLD-MARGIN arm that stops faulting is a confound, not a result.
struct FoldMarginTotals {
    uint64_t builds = 0, builds_with_exec = 0, rebuild_before_exec = 0, first_builds = 0;
    uint64_t margin_hist[13] = {};        // folds between prosper's last exec and this rebuild
    uint64_t period_hist[13] = {};        // folds between this label's own consecutive builds
    uint64_t execs = 0, execs_journal = 0;
    uint64_t age_hist[13] = {};           // folds between the guest's build and prosper's exec
    uint64_t age_fold_sum = 0, age_fold_max = 0;
    uint64_t deep_execs = 0, deep_age_fold_sum = 0;
};
std::mutex g_fold_margin_mu;
FoldMarginTotals g_fold_margin;
// Per-FOLD counters, drained by the submit-side ledger in hle_agc.cpp so one line can carry both
// halves: how much label protocol this individual fold carried, next to what it cost in time.
std::atomic<uint32_t> g_fm_fold_builds{0}, g_fm_fold_execs{0}, g_fm_fold_deep{0}, g_fm_fold_rbe{0};
bool fold_margin_on() {
    static const bool v = [] {
        const char* e = getenv("PROSPER_FOLD_MARGIN");
        const bool on = e && strtol(e, nullptr, 0) != 0;
        if (on)
            fprintf(stderr, "[agc] FOLD-MARGIN ARMED: per-fold recycle-margin census for the #1226 "
                            "label protocol (log-only, gates nothing)\n");
        return on;
    }();
    return v;
}
// 0..9 exact, then 10-15, 16-31, 32+. Exact small buckets matter because the whole mechanism lives
// between f+5 and f+6 — a log-scale histogram would put the entire population in one bin.
inline unsigned fold_bucket(uint64_t d) {
    if (d < 10) return (unsigned)d;
    if (d < 16) return 10;
    if (d < 32) return 11;
    return 12;
}
void fold_hist_str(const uint64_t (&h)[13], char* out, size_t cap) {
    static const char* kName[13] = {"0","1","2","3","4","5","6","7","8","9","10-15","16-31","32+"};
    size_t off = 0;
    for (unsigned i = 0; i < 13 && off + 24 < cap; i++) {
        if (!h[i]) continue;
        int m = snprintf(out + off, cap - off, "%s%s:%llu", off ? "," : "", kName[i],
                         (unsigned long long)h[i]);
        if (m > 0) off += (size_t)m;
    }
    if (!off && cap) { snprintf(out, cap, "(empty)"); return; }
    if (off < cap) out[off] = 0;
}
void fold_margin_report(const char* why) {
    char mh[256], ph[256], ah[256];
    uint64_t builds, bwe, rbe, fb, execs, ej, afs, afm, de, dafs;
    {
        std::lock_guard<std::mutex> lk(g_fold_margin_mu);
        const FoldMarginTotals& t = g_fold_margin;
        fold_hist_str(t.margin_hist, mh, sizeof mh);
        fold_hist_str(t.period_hist, ph, sizeof ph);
        fold_hist_str(t.age_hist, ah, sizeof ah);
        builds = t.builds; bwe = t.builds_with_exec; rbe = t.rebuild_before_exec; fb = t.first_builds;
        execs = t.execs; ej = t.execs_journal; afs = t.age_fold_sum; afm = t.age_fold_max;
        de = t.deep_execs; dafs = t.deep_age_fold_sum;
    }
    fprintf(stderr,
            "[agc] FOLD-MARGIN-TOTALS (%s) fold=%u builds=%llu(first=%llu with-exec=%llu "
            "REBUILD-BEFORE-EXEC=%llu) margin-folds{%s} rebuild-period-folds{%s} "
            "execs=%llu(journal=%llu) age-folds{%s} age-fold(max=%llu mean=%.2f) "
            "deep(execs=%llu mean-age-folds=%.2f) t=%llums\n",
            why, g_fold_seq.load(std::memory_order_relaxed),
            (unsigned long long)builds, (unsigned long long)fb, (unsigned long long)bwe,
            (unsigned long long)rbe, mh, ph,
            (unsigned long long)execs, (unsigned long long)ej, ah,
            (unsigned long long)afm, ej ? (double)afs / (double)ej : 0.0,
            (unsigned long long)de, de ? (double)dafs / (double)de : 0.0,
            (unsigned long long)now_ms());
}
// Called from the guest's own DMA-label builder, BEFORE the built counter is bumped, so
// `built - exec` still describes the state the guest is rebuilding into.
void fold_margin_build(LabelHist& h) {
    if (!fold_margin_on()) return;
    const uint32_t fold_now = g_fold_seq.load(std::memory_order_relaxed);
    const uint32_t db = h.dma_built_n.load(std::memory_order_relaxed);
    const uint32_t dx = h.dma_exec_n.load(std::memory_order_relaxed);
    const bool rebuild_before_exec = db > dx;
    bool report;
    uint64_t ord;
    {
        std::lock_guard<std::mutex> lk(g_fold_margin_mu);
        FoldMarginTotals& t = g_fold_margin;
        ord = ++t.builds;
        if (h.last_exec_fold == kNoFold) t.first_builds++;
        else {
            t.builds_with_exec++;
            t.margin_hist[fold_bucket(fold_now >= h.last_exec_fold ? fold_now - h.last_exec_fold : 0)]++;
        }
        if (h.last_build_fold != kNoFold)
            t.period_hist[fold_bucket(fold_now >= h.last_build_fold ? fold_now - h.last_build_fold : 0)]++;
        if (rebuild_before_exec) t.rebuild_before_exec++;
        report = (ord == 1) || (ord & 255) == 0 || rebuild_before_exec;
    }
    g_fm_fold_builds.fetch_add(1, std::memory_order_relaxed);
    if (rebuild_before_exec) g_fm_fold_rbe.fetch_add(1, std::memory_order_relaxed);
    // The detail line is the event itself, not a sample of it: the guest is reusing a label block
    // whose previous generation prosper has not executed. Budgeted separately from the totals so a
    // rare event can never be starved by the common one.
    if (rebuild_before_exec) {
        static std::atomic<uint64_t> n_rbe{0};
        if (diag_should_print(n_rbe.fetch_add(1) + 1, 64))
            fprintf(stderr, "[agc] FOLD-MARGIN-REBUILD-BEFORE-EXEC #%llu [0x%llx] fold=%u built=%u "
                            "exec=%u last-build-fold=%d last-exec-fold=%d t=%llums\n",
                    (unsigned long long)n_rbe.load(), (unsigned long long)h.addr, fold_now, db, dx,
                    h.last_build_fold == kNoFold ? -1 : (int)h.last_build_fold,
                    h.last_exec_fold == kNoFold ? -1 : (int)h.last_exec_fold,
                    (unsigned long long)now_ms());
    }
    h.last_build_fold = fold_now;
    if (report) fold_margin_report(rebuild_before_exec ? "rebuild-before-exec" : "cadence");
}
}   // namespace
// #1226 (arc7): drain this fold's label-protocol counters and report the fold ordinal. Called once
// per submit from the ledger in hle_agc.cpp, so the counts belong to exactly one fold.
extern "C" void prosper_fold_margin_take(uint32_t* builds, uint32_t* execs, uint32_t* deep,
                                         uint32_t* rebuild_before_exec, uint32_t* fold) {
    if (builds) *builds = g_fm_fold_builds.exchange(0, std::memory_order_relaxed);
    if (execs)  *execs  = g_fm_fold_execs.exchange(0, std::memory_order_relaxed);
    if (deep)   *deep   = g_fm_fold_deep.exchange(0, std::memory_order_relaxed);
    if (rebuild_before_exec)
        *rebuild_before_exec = g_fm_fold_rbe.exchange(0, std::memory_order_relaxed);
    if (fold)   *fold   = g_fold_seq.load(std::memory_order_relaxed);
}
extern "C" void prosper_label_hist_dma_built(uint64_t addr, uint64_t cb, uint32_t /*src*/, uint8_t builder) {
    if (!addr) return;
    LabelHist& h = label_hist_slot(addr);
    fold_margin_build(h);                                  // #1226 (arc7): before the bump
    h.dma_built_n.fetch_add(1, std::memory_order_relaxed);
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
    LabelHist& h = label_hist_slot(addr);
    h.dma_exec_n.fetch_add(1, std::memory_order_relaxed);
    h.last_exec_fold = g_fold_seq.load(std::memory_order_relaxed);   // #1226 (arc7) fold_margin
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
// Read-only (label_hist_find, not label_hist_slot): printing a label's history must never evict a
// colliding label's protocol state — see label_hist_find.
void label_hist_report(uint64_t addr, char* out, size_t cap) {
    static const char* nm[] = {"?", "dmaB", "relB", "waitB", "dmaX", "relX", "dmaSKIP",
                               "dmaFREE", "relFREE"};
    const LabelHist* hp = label_hist_find(addr);
    if (!hp) { snprintf(out, cap, "events(total=0, no-history)"); return; }
    const LabelHist& h = *hp;
    uint32_t n = h.n.load(std::memory_order_relaxed);
    uint32_t first = n > 16 ? n - 16 : 0;
    size_t off = (size_t)snprintf(out, cap, "events(total=%u):", n);
    // Exec events carry the folding submit entry point (#1226): D=SubmitDcb A=SubmitAcb
    // F=SubmitDcbFinal, absent when unknown/build-side — the cross-queue discriminator.
    static const char* qn[] = {"", "(D)", "(A)", "(F)"};
    // #2192: the reserve was 52 while one entry can reach 63 bytes
    // (" %s%s@%llu/f%u:0x%llx" = 1 + 7 + 3 + 1 + 20 + 2 + 10 + 3 + 16), so the last admitted
    // iteration truncated MID-ENTRY -- a half-written field with nothing to say the remaining
    // events were dropped rather than absent. A truncated label-event ring is exactly the
    // evidence this instrument exists to print, and it is read from the fault handler, so
    // "looks complete but is not" is the worst outcome available here.
    //
    // Fixed by construction rather than by correcting the constant: append FIRST, and if the
    // entry did not fit, roll the cursor back so no partial entry survives, then say how many
    // were lost. There is no reserve left to drift out of step with the format string.
    // Room the drop marker will need, held back from the entry loop. A RESERVE is what this change
    // set out to remove -- but the two are not the same thing and the difference is the whole point:
    // a reserve for a VARIABLE entry drifts the moment the format changes, while this one is for a
    // FIXED string whose maximum is fully determined by the format. " +4294967295 more" is 17 bytes
    // plus a terminator; 24 is that with room to spare, and it cannot rot because `dropped` is a
    // uint32_t.
    //
    // Held back rather than written over the tail, which is how the first version of this fix was
    // wrong -- and it was wrong in exactly the way it was fixing. Placing the marker at
    // min(off, cap - 24) let it land INSIDE the last complete entry when the buffer was nearly
    // full, clipping a good entry to write the notice that entries were clipped. macOS CI caught it
    // (the platform neither this author nor the reviewer runs); the arm that fired was the
    // partial-entry one, which had been kept only as a regression guard.
    constexpr size_t kDropMark = 24;
    const size_t entry_limit = cap > kDropMark ? cap - kDropMark : 0;
    uint32_t dropped = 0;
    for (uint32_t i = first; i < n; i++) {
        const LabelEvent& e = h.ev[i & 15];
        const size_t before = off;
        const int m = off < entry_limit
            ? snprintf(out + off, entry_limit - off, " %s%s@%llu/f%u:0x%llx",
                       e.type <= 8 ? nm[e.type] : "?",
                       e.origin <= 3 ? qn[e.origin] : "",
                       (unsigned long long)e.t_ms, e.fold,
                       (unsigned long long)e.aux)
            : -1;
        if (m > 0 && before + (size_t)m < entry_limit) { off = before + (size_t)m; continue; }
        out[before] = '\0';                  // drop the partial entry snprintf just wrote
        dropped = n - i;
        break;
    }
    // `off <= entry_limit == cap - kDropMark`, so the marker always fits at `off` and can never
    // reach an entry that was admitted.
    if (dropped && cap > kDropMark)
        snprintf(out + off, cap - off, " +%u more", dropped);
}
}
extern "C" void prosper_label_hist_dump(uint64_t addr, char* out, unsigned cap) {
    label_hist_report(addr, out, cap);
}
// #1226: total PROSPER_WRITE_TRAP matches, uncapped (the printed lines are capped). Exposed so a
// test can assert both arms of the trap without parsing stderr — a value the guest's stream writes
// must be counted, an adjacent value must not.
extern "C" uint64_t prosper_gpu_write_trap_matches() {
    return g_write_trap_matches.load(std::memory_order_relaxed);
}

// #312: "pointer-like" pre-content — a freed MallocBinned3 block header holds heap pointers into
// the 512 GiB arena (0x1000000000) or the allocator-metadata pools (0x2000000000 region, where the
// FPoolInfo tables live — e.g. the 0x20015f0000 pool-info table of #161/#241).
//
// #1226 — THIS WINDOW IS DOLL-ERA AND IS THE REASON ARCRUNNER'S OWN TERMINAL FORGE WAS INVISIBLE.
// ArcRunner (PPSA21406) reserves its MallocBinned3 arena at [0x2000000000, 0xa000000000) (live:
// `reserve ENTRY hint=0x1000000000 len=0x8000000000` -> `reserve -> 0x2000000000`, and the guest's
// own "Memory va range 2000000000 - 9fc0000000"), so the upper bound below covers 4 GiB of a
// 512 GiB arena. The terminal `addr=(nil)` fault that ends most ArcRunner runs dereferences
// `rdi = 0x2100000001` — a `0x21xxxxxxxx` heap pointer whose low dword was zeroed and then set to 1,
// i.e. exactly the forge shape this predicate exists to catch, sitting EXACTLY ONE BYTE above the
// bound. Every `forges_freelist_ptr()` census taken on this title is therefore a lower bound over
// 1/128th of the arena, while the INIT-side census (init_trip) already used the wide predicate — the
// two sides were never comparable.
//
// The split below is deliberate:
//   * heap_ptr_like()  — the honest "this qword is plausibly a live guest pointer" predicate, over
//     prosper's whole guest-VA window. Used by every REPORTING path (forge_trip, init_trip,
//     clock-fence records) so a census stops lying. It gates NO default-boot decision — but it is
//     not gate-free: `init_suppress()` reads it through `clockfence_heapish()`, so widening it does
//     change which writes the **default-OFF** `PROSPER_INIT_SUPPRESS=ptr` A/B arm drops. Stated
//     rather than glossed, because "report-only" was the previous claim and it was not true.
//   * ptr_like()       — what the default GUARDS still use. It is NOT silently widened here because
//     rel1_stomp_guard() is default-ON: widening it arms a suppression over 500 GiB of previously
//     unreachable addresses on every title at once, and over-suppressing fences is the documented
//     #1245 regression (thousands/min of dependency-violated WaitRegMem). `PROSPER_PTRLIKE_WIDE=1`
//     is the A/B lever that arms the guards over the wide window, so the flip is measured before it
//     is made. CONFIDENCE: HIGH that the narrow window is stale; MED on the right guard policy.
static inline bool ptr_like_narrow(uint64_t v) {
    return (v >= 0x1000000000ull && v < 0x1200000000ull) ||
           (v >= 0x2000000000ull && v < 0x2100000000ull);
}
// prosper's guest heap/dmem VA window. The question a report asks is "was a plausible live pointer
// here", not "which allocator owns it", so a broad window is the right shape for it. Note the lower
// bound is exactly GPU_VA_HI (exec_image_linux.cpp): the lazily-backed GPU-VA window
// [0x100000000, 0x1000000000) is BELOW this and deliberately excluded — a GPU VA is not a heap
// pointer, and admitting it would make every render-target address read as a stomped link.
// CONFIDENCE: MED on the exact bounds — they follow map_guest_auto's placement and the two arenas
// observed live (DOLL at 0x1000000000, ArcRunner at [0x2000000000, 0xa000000000)), so a title that
// places its arena elsewhere will under-report until this is derived from the memory HLE instead.
static inline bool heap_ptr_like(uint64_t v) {
    return v >= 0x1000000000ull && v < 0xa000000000ull;
}
static bool ptrlike_wide() {
    // Self-witnessing: an A/B lever that cannot show it moved turns a hard negative into a void
    // result, because "the guard saw nothing" and "the guard was never widened" print identically.
    // Announce the arm once, with the exact window, and announce a malformed value as NOT ARMED
    // rather than silently falling back to the narrow default.
    static const bool v = [] {
        const char* e = getenv("PROSPER_PTRLIKE_WIDE");
        if (!e) return false;
        char* end = nullptr;
        const long parsed = strtol(e, &end, 0);
        if (end == e || (end && *end)) {
            fprintf(stderr, "[agc] PTRLIKE-WIDE NOT ARMED: PROSPER_PTRLIKE_WIDE='%s' is not a number "
                            "— guards keep the narrow window\n", e);
            return false;
        }
        if (parsed == 0) return false;
        fprintf(stderr, "[agc] PTRLIKE-WIDE ARMED: guard window [0x1000000000,0xa000000000) "
                        "(narrow default was [0x1000000000,0x1200000000)+[0x2000000000,0x2100000000))\n");
        return true;
    }();
    return v;
}
static bool ptr_like(uint64_t v) {
    return ptrlike_wide() ? heap_ptr_like(v) : ptr_like_narrow(v);
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
            fprintf(stderr, "[agc] POOLSHIFT-PAYLOAD kind=%s dst=0x%llx bytes=%llu payload=0x%llx pkt=0x%llx t=%llums\n",
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
                fprintf(stderr, "[agc] POOLSHIFT-STOMP kind=%s @0x%llx=0x%llx (<<8=0x%llx) dst=0x%llx bytes=%llu inspan=%d payload=0x%llx pkt=0x%llx t=%llums\n",
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
//
// #1226 — what this shape is on ArcRunner (PPSA21406), and why the guard below cannot fix it.
// SAMPLING NOTE, because the numbers below are the point: forge_trip prints the first 64 hits and
// then powers of two, so "every hit" here means every REPORTED hit — ~68 lines out of a population
// exceeding 1,024 per run, which is a biased sample (it is front-loaded). Read the figures as
// "unanimous across every hit that was reported", never as a census.
//
// Every reported hit carries an identical `pre=0x2000000000 val=0x1 -> 0x2000000001` and an
// executed-but-unsignalled init outstanding (`dma_exec_n > rel_exec_n`), so `!live_pair` declines
// them. Read `live_pair` carefully: it is prosper's OWN bookkeeping. It is NECESSARY for the
// live-paired-fence story (a label with no outstanding init is definitely not mid-protocol) but not
// SUFFICIENT, and in particular it cannot see the case waf_guard() documents — the guest freeing
// the 0x20 label while BOTH our writes are in flight produces exactly these counters on an
// already-freed block.
//
// The forged qword is a POINTER WITH ITS LOW BIT SET, so a guest free-list walk that dereferences
// it reads from ONE BYTE HIGHER and gets the target qword shifted DOWN by 8 (little-endian; the
// top byte comes from the following byte, which is 0 here). On every reported hit
// `*(0x2000000001) = 0x30016000`, byte-for-byte the value the terminal free-list fault at
// eboot+0x127e751 dereferences. Note what the probe does and does not show: it measures the
// CONTENT of the target, which reads the same whether or not any forge happened. What establishes
// that the guest actually dereferenced `0x2000000001` is the disassembled pop plus the guest's own
// `FMallocBinned3 Attempt to free/realloc an unrecognized block 2000000001` fatal — one pop both
// RETURNS that node (the fatal) and stores `[node]` as the new head (the next pop's SIGSEGV).
// Together: **the "POOLSHIFT byte-shifted pool pointer" signature this issue family has chased
// since #1249 is the READ ARTIFACT of this forge, not an independent defect** — the session-10
// read-artifact note below, now with a direct measurement behind it. This supersedes the #1252
// reading that the poison "provably enters via the guest's own free()" with the push path's count
// ADVANCING; the pop at eboot+0x127e751 DECREMENTS its count, so that observation needs re-checking
// before it is relied on again.
//
// A one-run-per-arm A/B that suppresses every forging fence removes `0x30016000` and moves the
// fault to `0x2400100024001`: with no fence value the qword stays `0x2000000000`, and the walk then
// reads it ALIGNED as `0x3001600000` and steps into that buffer's `0x00024001` fill (two pops
// further, not one). The load-bearing half of that result is not the changed value — the fault site
// is the free-list pop, where ANY heap corruption lands — but that the head is still
// `0x2000000000`, which only the 4-byte DMA_DATA immediate-zero init can produce by zeroing the low
// dword of a live `0x20xxxxxxxx` link. So the fence is the SECOND half of the damage and the init
// is the first; guarding the fence alone cannot help, and the discriminator has to be the guest's
// actual freelist MEMBERSHIP at write time (mb3_freelist_guard) rather than either write's content.
// Do not "fix" this by widening a value-shape predicate; tests/gpu/test_eop_write.cpp pins both arms.
//
// #1226 UPDATE — the init experiment was run. It does NOT support the paragraph above's second
// half, and it does not support the opposite either. Whole-run totals (PROSPER_INIT_TRIP=1, one
// 90 s run, n = every 4-byte zero init to a consumed-marker label):
//
//     n=1024  overptr=926  member=10  both=1
//
// So 90% of these inits do overwrite pointer-shaped content, but only ~1% of the destinations are
// on an idx=1 free chain the walk models, and exactly ONE init in the run is both. The init is
// therefore NOT systematically destroying live FFreeBlock links — the pointer it overwrites is
// almost always the label pool's own stale link in a block the guest holds. `overptr` alone never
// could have shown this, which is why #1754 left it at MED.
//
// READ THE TOTALS, NOT A LINE. The `member` population appears LATE: every sample before ordinal
// ~512 reads member=0, and quoting one of those as the run's answer is how the first version of
// this comment claimed "member=0 every time". That is the trap-51 cap-as-count error in a second
// costume, which is why init_trip now prints INIT-TRIP-TOTALS every 256 as well as on the sparse
// schedule. Scope: "not on the idx=1 chains this walk models" (per-thread bundles, the eight
// recycler slots, central runs) — NOT "on no list anywhere". The label pool may keep its own.
//
// The suppression arm does NOT show what an earlier draft of this comment claimed. Recorded so it
// is not re-derived: PROSPER_INIT_SUPPRESS=ptr drops 2,048+ inits, collapses the forge population
// from ~4,096 to 21, additionally costs 192 fences through the REL1-LIVE guard — and the same
// 0x30016000 still appears. That is NOT "neither write is necessary": one poisoned node is all the
// pop needs, so a 200x rate reduction is not a necessity test, and the arm one paragraph above
// (fence fully suppressed, landed forges = 0) DID remove 0x30016000. Across the three arms the
// series reads 4096 -> poison, 21 -> poison, 0 -> no poison, which points at the fence being
// SUFFICIENT, the opposite of the withdrawn claim.
//
// CONFIDENCE: HIGH on the read-artifact mechanism (arithmetic + guest fatal + disassembly).
// LOW on any causal statement about the init: measured as a ~1-in-1000 event, mechanism unproven.
// The decisive arm still unrun is landed-forges-zero AND init-suppressed together; if 0x30016000
// survives that, prosper's label writes are excluded, and if it does not, the fence is the author.
static inline bool forges_freelist_ptr(uint64_t pre, uint64_t width, uint64_t value) {
    if (!ptr_like(pre)) return false;        // dst must currently hold a heap/pool pointer
    if ((uint32_t)pre != 0) return false;    // low dword already nonzero -> the REL1-LIVE guard covers it
    if (width >= 8) return false;            // a full-width write replaces the whole qword (no forge)
    return (uint32_t)value != 0;             // our sub-qword write makes the low dword nonzero -> pre|value
}
// #1226: the same shape test over the honest heap window. `forges_freelist_ptr()` above is the
// DECISION predicate and stays on the (default-narrow) guard window; this one is REPORT-ONLY, so a
// forge no guard can currently see is still counted, and is labelled `window=wide-only` in the line
// and in the terminal totals. Without this split a census silently measures the predicate instead of
// the title: on ArcRunner the forge that composes the terminal fault's `0x2100000001` is wide-only.
static inline bool forges_freelist_ptr_wide(uint64_t pre, uint64_t width, uint64_t value) {
    if (!heap_ptr_like(pre)) return false;
    if ((uint32_t)pre != 0) return false;
    if (width >= 8) return false;
    return (uint32_t)value != 0;
}
namespace {
struct ForgeTripTotals { uint64_t seen = 0, narrow = 0, wide_only = 0; };
std::mutex g_forge_trip_totals_mu;
ForgeTripTotals g_forge_trip_totals;
}
// The two live predicates, exported so a test can pin the exact bound that caused the #1226 blind
// spot rather than re-deriving it from a log. The narrow verdict MUST come from
// `forges_freelist_ptr()` itself — the decision function the guards actually call. An earlier form
// re-derived it from `ptr_like_narrow()` plus a private copy of the shape rules, which meant the
// one regression these cases exist to prevent (collapsing the two predicates by widening
// `ptr_like()`'s body) left every case passing while every guard silently widened. A test that
// cannot fail for its own stated regression is not a test.
extern "C" int prosper_forge_predicate_for_test(uint64_t pre, uint64_t width, uint64_t value,
                                                int* wide) {
    if (wide) *wide = forges_freelist_ptr_wide(pre, width, value) ? 1 : 0;
    return forges_freelist_ptr(pre, width, value) ? 1 : 0;
}
// Log-only tripwire (PROSPER_FORGE_TRIP=1): report a GPU write that forges a freelist pointer, with
// the packet builder callsite — deciding host(GPU)-vs-guest for the root write. Default OFF (no-op).
static void forge_trip(const char* kind, uint64_t dst, uint64_t pre, uint64_t value, uint64_t width, uint64_t pkt) {
    static const bool on = getenv("PROSPER_FORGE_TRIP") != nullptr;
    if (!on || !forges_freelist_ptr_wide(pre, width, value)) return;
    // Which window matched. `narrow` means the default guards can see (and possibly decline) this
    // write; `wide-only` means no guard is reachable for it at all on this build's settings.
    const bool narrow_window = forges_freelist_ptr(pre, width, value);
    uint64_t tot_seen = 0, tot_narrow = 0, tot_wide_only = 0;
    {
        std::lock_guard<std::mutex> lock(g_forge_trip_totals_mu);
        g_forge_trip_totals.seen++;
        if (narrow_window) g_forge_trip_totals.narrow++; else g_forge_trip_totals.wide_only++;
        tot_seen = g_forge_trip_totals.seen;
        tot_narrow = g_forge_trip_totals.narrow;
        tot_wide_only = g_forge_trip_totals.wide_only;
    }
    // TOTALS ride their own DENSE schedule (every 256, plus the first), exactly as `init_trip` does
    // 250 lines below and for the same reason. The detail schedule is sparse — first 64, then powers
    // of two — so a totals figure carried only on detail lines is stale by up to 255 events, and a
    // `wide_only=0` read off the last detail line would be a claim about event 256 dressed up as a
    // claim about the run. The worker-fault path calls `_exit()`, so no atexit summary can close
    // the gap either.
    if (tot_seen == 1 || (tot_seen & 255) == 0)
        fprintf(stderr, "[agc] FORGE-TRIP-TOTALS #%llu seen=%llu narrow=%llu wide_only=%llu t=%llums\n",
                (unsigned long long)tot_seen, (unsigned long long)tot_seen,
                (unsigned long long)tot_narrow, (unsigned long long)tot_wide_only,
                (unsigned long long)now_ms());
    // #1226: print the first 64, then keep printing at every power of two, and carry the ordinal on
    // every line. The old form printed exactly 64 lines and nothing else, so "64 hits" — quoted as a
    // count on this issue — was the CAP, indistinguishable from a run with 64,000. A tripwire that
    // reports a population must never let its own limit masquerade as the measurement.
    static std::atomic<uint64_t> n{0};
    const uint64_t ord = n.fetch_add(1) + 1;
    if (diag_should_print(ord)) {
        // #1226: report WHY the paired guard did or did not decline, in the same line. The guard is
        // `forge_guard() && label_is_consumed_marker(dst) && !live_pair`, and a bare hit count could
        // not distinguish "the label has no tracked init protocol at all" (cm=0 — every #312 guard is
        // then inert on it) from "a live init is outstanding, so the write is the guest's own paired
        // fence" (live=1 — suppressing would repeat the #1245 regression). The event ring carries the
        // per-generation PRE-CONTENT at each exec, which is what shows the two-write composition of
        // the forged qword (init zeroes the low dword of a pointer, the fence then sets it to 1).
        // label_hist_FIND, not label_hist_slot: the slot getter evicts a colliding label's protocol
        // state, which would make the #312 guards inert on it — a tripwire must never change what
        // the guards decide.
        const LabelHist* h = label_hist_find(dst);
        uint32_t db = h ? h->dma_built_n.load(std::memory_order_relaxed) : 0;
        uint32_t dx = h ? h->dma_exec_n.load(std::memory_order_relaxed) : 0;
        uint32_t rb = h ? h->rel_built_n.load(std::memory_order_relaxed) : 0;
        uint32_t rx = h ? h->rel_exec_n.load(std::memory_order_relaxed) : 0;
        char hist[640]; label_hist_report(dst, hist, sizeof hist);
        // #1226: the same membership question init_trip asks, at the OTHER write. `live_pair` is our
        // own bookkeeping and cannot arbitrate whether the guest still owns this block; the guest's
        // free chains can. Only on printed lines, so the walk's cost stays bounded, and always with
        // the self-test beside it so a `member=0` is never read from an unarmed walk.
        Mb3FreelistMatch fmatch{};
        const bool fmember = mb3_freelist_contains_stable(dst, &fmatch);
        char fself[192]; mb3_freelist_selftest(fself, sizeof fself);
        // The forged qword is a POINTER WITH ITS LOW BIT SET. Should the guest allocator ever
        // dereference it as a free-block next-pointer, that read is MISALIGNED by one byte and
        // returns the target qword shifted right 8 bits — which is exactly the shape of the
        // constant `0x30016000` / `0x30015f00` / `0x20015f00` poisons this issue family chases.
        // Report what the misread would yield so "forged pointer -> misaligned read -> byte-shifted
        // pool address" can be confirmed or refuted from one line rather than inferred.
        const uint64_t forged = pre | (value & 0xffffffffull);
        uint64_t misread = 0;
        const bool mis_ok = guest_readable(forged, 8);
        if (mis_ok) memcpy(&misread, (const void*)(uintptr_t)forged, sizeof misread);
        char mis[40];
        if (mis_ok) snprintf(mis, sizeof mis, "0x%llx", (unsigned long long)misread);
        else        snprintf(mis, sizeof mis, "unmapped");
        fprintf(stderr,
                "[agc] FORGE-STOMP #%llu kind=%s window=%s dst=0x%llx pre=0x%llx val=0x%llx -> 0x%llx "
                "pkt=0x%llx t=%llums cm=%d live=%d member=%d(list=%u) dmaB=%u dmaX=%u relB=%u relX=%u "
                "misread(*forged)=%s | FORGE-TRIP-TOTALS seen=%llu narrow=%llu wide_only=%llu "
                "| selftest %s | %s\n",
                (unsigned long long)ord,
                kind, narrow_window ? "narrow" : "wide-only",
                (unsigned long long)dst, (unsigned long long)pre, (unsigned long long)value,
                (unsigned long long)forged, (unsigned long long)pkt,
                (unsigned long long)now_ms(), db > 0 ? 1 : 0, dx > rx ? 1 : 0,
                (int)fmember, fmatch.list, db, dx, rb, rx,
                mis,
                (unsigned long long)tot_seen, (unsigned long long)tot_narrow,
                (unsigned long long)tot_wide_only,
                fself, hist);
    }
}
// #1226 — the census every predicate above is STRUCTURALLY blind to: a landed sub-qword label write
// that destroys the low dword of a live pointer whose TARGET lies outside the heap windows.
//
// `forges_freelist_ptr()` and `report_suspect_write("REL1-LIVE")` both filter `pre` through
// `ptr_like()`/`heap_ptr_like()`, whose widest window is `[0x1000000000,0xa000000000)` — the guest
// ARENA. A C++ object's vtable pointer does not live there: it points into a loaded module IMAGE
// (ArcRunner's eboot base is 0x410000000, BELOW that bound -- by a factor of about four, not the
// "three orders of magnitude" an earlier draft of this comment claimed). So no guard,
// tripwire or census in this file can see a 4-byte write that overwrites a vptr's low dword, and the
// `wide_only=0` / `SUSPECT-REL1-LIVE=0` negatives already recorded on this title say nothing about
// that population. That is not a tuning gap — an arena bound is exactly what a heap predicate is for.
//
// This trip therefore asks the shape question with NO address window at all. `pre` counts as a live
// pointer when its high dword is nonzero and its VALUE addresses mapped memory in this process
// (`guest_readable`, which on Linux probes with a real `write(2)` EFAULT test and so covers module
// images as well as the arena). Every reported hit is bucketed by `pre >> 32`, so the region is read
// OFF the census instead of assumed.
//
// Reading a zero: it is only a negative next to its own population. `examined` counts every
// sub-qword write offered to the trip and is printed on its own dense schedule even when `live` is
// 0, so "armed and nothing matched" is distinguishable from "never armed" — the failure mode that
// made three earlier arms on this title void. Log-only, default OFF, no behaviour change.
//
// CONFIDENCE: HIGH that the blind spot is real (the window bounds are literal, above).
// Width note (`CONFIDENCE: HIGH` for the measured population, which is entirely 4-byte writes;
// the modelling is loose either side of it and the reported `after` below is only exact at 4).
// `dma_data_form` constrains `dd_bytes` to [1, 0x1000000] and 4-aligns `dd_dst` but NOT `dd_bytes`,
// so widths 1-3 and 5-7 do reach here: at 1-3 the write replaces only part of the low dword, so the
// `pre_low != value` test can call a no-op write a stomp; at 5-7 it also clobbers part of the HIGH
// dword, which `after` does not model. Diagnostic accuracy only — the guard arm is default OFF.
static inline bool stomps_live_ptr_shape(uint64_t pre, uint64_t width, uint64_t value) {
    if (width >= 8) return false;              // a full-width write replaces the whole qword
    if ((pre >> 32) == 0) return false;        // no high half -> not a 64-bit pointer-shaped qword
    const uint32_t pre_low = (uint32_t)pre;
    if (pre_low == 0) return false;            // forges_freelist_ptr() owns the zero-low-dword case
    return pre_low != (uint32_t)value;         // the write must actually change the low dword
}
// Exported so a test pins the exact boundary this exists to cover: the vptr case that both heap
// predicates reject. Keeping the test on the real decision function (not a private copy of the
// rules) is the same lesson `prosper_forge_predicate_for_test` records above.
extern "C" int prosper_liveptr_shape_for_test(uint64_t pre, uint64_t width, uint64_t value) {
    return stomps_live_ptr_shape(pre, width, value) ? 1 : 0;
}
// The GUARD's content decision. `!heap_ptr_like(pre)` is the whole reason the arm cannot repeat
// #1245: without it the arm would decline the ~1,280 free-list-residue writes an ArcRunner run makes.
//
// `declines_nonheap_ptr()` calls THIS function rather than re-deriving the conjunction, and that is
// the point. A first version exported a private copy of the same expression, so deleting the guard's
// own `!heap_ptr_like` line left every case green — a test that could not fail for the deletion it
// claimed to cover, which is the trap recorded beside `prosper_forge_predicate_for_test` above.
// One definition, two callers: now the deletion fails `nonheap guard content` in test_eop_write.
// The liveness probe (`guest_readable`) stays at the call site — it needs a live process — so this
// covers exactly the pure part.
static inline bool nonheap_guard_content(uint64_t pre, uint64_t width, uint64_t value) {
    return stomps_live_ptr_shape(pre, width, value) && !heap_ptr_like(pre);
}
extern "C" int prosper_nonheap_guard_content_for_test(uint64_t pre, uint64_t width, uint64_t value) {
    return nonheap_guard_content(pre, width, value) ? 1 : 0;
}
namespace {
struct LivePtrTotals {
    uint64_t examined = 0, shape = 0, live = 0;
    static constexpr int kBuckets = 12;
    uint64_t hi[kBuckets]{};      // pre >> 32 of a live hit
    uint64_t hi_n[kBuckets]{};
    int hi_used = 0;
    uint64_t hi_other = 0;        // live hits past the bucket table
};
// The detail schedule is PER BUCKET, and that is load-bearing rather than tidy. A single shared
// ordinal makes the rare class invisible: ArcRunner produces hundreds of free-list-residue hits
// (`pre >> 32 == 0x20`) for every one module-image vptr hit, so on a shared `diag_should_print` the
// vptr lands at some ordinal like 300 — past the first-64 window, not a power of two — and is never
// printed even though it was counted. That happened here on the first arm and cost a run. Keyed per
// bucket, the FIRST hit of every new `pre >> 32` class always prints.
//
// The same failure returns past `kBuckets` (12) distinct classes: everything beyond shares the
// `hi_other` ordinal, so only the first of THEM gets an `ord == 1` print. That is the same trap at a
// higher threshold, and it is why `hi=other:N` is printed on the totals line — a nonzero `hi_other`
// means classes exist that the detail lines did not individually announce, so read it as "raise
// kBuckets and re-run", never as "no further classes".
std::mutex g_liveptr_mu;
LivePtrTotals g_liveptr;
// Formats into `out` rather than printing: the caller holds `g_liveptr_mu` and prints after
// releasing it, the same shape `forge_trip` uses.
void liveptr_totals_line(const char* why, char* out, size_t cap) {
    char buckets[320]; int off = 0; buckets[0] = 0;
    for (int i = 0; i < g_liveptr.hi_used && off < (int)sizeof buckets - 40; ++i)
        off += snprintf(buckets + off, sizeof buckets - off, " hi=0x%llx:%llu",
                        (unsigned long long)g_liveptr.hi[i], (unsigned long long)g_liveptr.hi_n[i]);
    // `off` accumulates snprintf's WOULD-BE length, so it can run past the array: the loop admits an
    // iteration at off <= 279 and one entry's would-be length reaches 43, so off can hit 322. The
    // loop then exits safely, but the append below would compute `buckets + 322` with
    // `sizeof buckets - off` — an int-to-size_t conversion that underflows to SIZE_MAX-1, i.e. an
    // out-of-bounds write with an effectively unbounded size. Unreachable at kBuckets == 12 (twelve
    // realistic entries total ~180 chars), but the comment above tells the next agent to RAISE
    // kBuckets, which is exactly what arms it. Clamp rather than leave that trap behind the advice.
    // (Same snprintf-return-value class as the [labelhist] defect this PR fixed; reported in review.)
    if (off > (int)sizeof buckets - 1) off = (int)sizeof buckets - 1;
    if (g_liveptr.hi_other)
        snprintf(buckets + off, sizeof buckets - off, " hi=other:%llu",
                 (unsigned long long)g_liveptr.hi_other);
    snprintf(out, cap,
             "[agc] LIVEPTR-TOTALS(%s) examined=%llu shape=%llu live=%llu t=%llums |%s\n",
             why, (unsigned long long)g_liveptr.examined, (unsigned long long)g_liveptr.shape,
             (unsigned long long)g_liveptr.live, (unsigned long long)now_ms(),
             buckets[0] ? buckets : " (no live hits)");
}
} // namespace
static void liveptr_trip(const char* kind, uint64_t dst, uint64_t pre, uint64_t value,
                         uint64_t width, uint64_t pkt) {
    static const bool on = [] {
        const bool armed = getenv("PROSPER_LIVEPTR_TRIP") != nullptr;
        if (armed)
            fprintf(stderr, "[agc] LIVEPTR-TRIP ARMED: reporting landed sub-qword writes over any "
                            "mapped-target pointer, with no address window\n");
        return armed;
    }();
    if (!on) return;
    const bool shape = stomps_live_ptr_shape(pre, width, value);
    // `guest_readable` costs a cached syscall, so it runs only for the shape-matching minority.
    const bool live = shape && guest_readable(pre, 8);
    uint64_t ord = 0; bool print_totals = false; char totals_line[512] = {0};
    {
        std::lock_guard<std::mutex> lock(g_liveptr_mu);
        g_liveptr.examined++;
        if (shape) g_liveptr.shape++;
        if (live) {
            g_liveptr.live++;
            const uint64_t hi = pre >> 32;
            int slot = -1;
            for (int i = 0; i < g_liveptr.hi_used; ++i) if (g_liveptr.hi[i] == hi) { slot = i; break; }
            if (slot < 0 && g_liveptr.hi_used < LivePtrTotals::kBuckets) {
                slot = g_liveptr.hi_used++;
                g_liveptr.hi[slot] = hi;
            }
            if (slot >= 0) ord = ++g_liveptr.hi_n[slot]; else ord = ++g_liveptr.hi_other;
        }
        // The population line rides `examined`, not `live`: a run whose `live` is 0 must still
        // publish that it inspected thousands of writes, or the zero is unreadable.
        print_totals = (g_liveptr.examined == 1) || (g_liveptr.examined & 4095) == 0 ||
                       (live && (g_liveptr.live == 1 || (g_liveptr.live & 255) == 0));
        // Formatted into a local under the lock, printed outside it — matching `forge_trip`, which
        // copies its totals out at the same point and for the same reason: the invariant worth
        // keeping is that stderr is never written while a totals mutex is held. Nothing here can
        // deadlock either way (the order is always mutex -> stderr, never the reverse), but the two
        // tripwires should not model it differently. (Reported in review of #2077.)
        if (print_totals) liveptr_totals_line(live ? "hit" : "population", totals_line,
                                              sizeof totals_line);
    }
    if (print_totals) fputs(totals_line, stderr);
    if (!live || !diag_should_print(ord)) return;
    // What the write turns the pointer into, and whether the corrupted value still addresses
    // anything — the guest dereferences it next, so an unmapped result names the coming fault.
    const uint64_t after = (pre & 0xffffffff00000000ull) | (value & 0xffffffffull);
    char hist[512]; label_hist_report(dst, hist, sizeof hist);
    fprintf(stderr,
            "[agc] LIVEPTR-STOMP hi=0x%llx#%llu kind=%s dst=0x%llx pre=0x%llx (mapped, %s) "
            "val=0x%llx -> 0x%llx after-mapped=%d width=%llu pkt=0x%llx t=%llums | %s\n",
            (unsigned long long)(pre >> 32), (unsigned long long)ord, kind,
            (unsigned long long)dst, (unsigned long long)pre,
            heap_ptr_like(pre) ? "heap" : "NON-HEAP(module image?)",
            (unsigned long long)value, (unsigned long long)after,
            guest_readable(after, 8) ? 1 : 0, (unsigned long long)width,
            (unsigned long long)pkt, (unsigned long long)now_ms(), hist);
}
// #1226 A/B arm (default OFF): decline a sub-qword label write whose destination currently holds a
// MAPPED pointer that is NOT in the guest heap window — in practice a C++ vtable pointer into a
// loaded module image.
//
// Why this content test is sound where the heap one is not. `rel1_stomp_guard()` already declines a
// fence over `ptr_like(pre)` content, and widening THAT caused the #1245 regression, because a
// legitimately initialised label is byte-identical to a freed free-list node: the guest's 4-byte init
// leaves `{stale malloc residue, 0}`, and stale residue is a heap pointer. Content genuinely cannot
// separate those two.
//
// A module-image pointer is a NARROWER twin, but it is not twin-free, and the difference matters.
// Nothing a label protocol or an allocator free list *writes* is a code-image pointer: an
// `FFreeBlock::NextFreeBlock` points at another heap block, and a fence label holds a small integer.
// So `pre` in that range means the destination most recently held a C++ object.
//
// It does NOT prove that object is still LIVE, and an earlier draft of this comment claimed it did.
// A block that held a C++ object, was freed, and was handed back out for a label still carries the
// dead object's vptr at offset 0 until something overwrites it — residue, not a live object, and
// declining the init in that case is #1245 transposed to the module-image half. This project's own
// static attribution makes that twin MORE plausible rather than less: the stomped object is a
// 24-byte `new` allocation (`eboot+0x1250720`, `eboot+0x128e4b0`), which lands in the same 32-byte
// `FMallocBinned3` bin the 0x20-byte labels come from, so object→label recycling in that bin is
// exactly the expected traffic. The residue window is narrower than the heap case's (a freed block's
// first qword is heap-pointer-shaped for the whole time it sits on a free list, whereas a vptr
// survives only until the allocator's own bookkeeping lands on it) — but "narrower" is not "absent".
// Discriminating the two needs block ownership at write time, not content. CONFIDENCE: MED.
//
// Measured on ArcRunner (`PPSA21406`), the terminal `AudioMixerRende` fault, with the label ring at
// the faulting register: `dmaX(D)@7895/f37:0x41700f1e8` then `relX(D)@7895/f37:0x400000000` — our
// 4-byte init zeroed the low dword of the vtable pointer `eboot+0x700f1e8`, our 4-byte fence then
// set it to 1, and the guest's next `mov rax,[rdi]; call [rax+0x20]` at `eboot+0x32b61be` read
// `0x400000001` and jumped to 0. See `docs/ARCRUNNER_STATUS.md`.
//
// Default OFF because it is an A/B arm, not yet a shipped contract: declining leaves that label
// unwritten, and although no consumer of a reallocated block can legitimately be waiting on it, that
// argument is reasoning rather than measurement. `PROSPER_NONHEAP_PTR_GUARD=1` announces itself and
// counts, so an arm can never be confused with a no-op. CONFIDENCE: MED.
static bool nonheap_ptr_guard() {
    static const bool v = [] {
        const char* e = getenv("PROSPER_NONHEAP_PTR_GUARD");
        const bool on = e && strtol(e, nullptr, 0) != 0;
        if (on)
            fprintf(stderr, "[agc] NONHEAP-PTR-GUARD ARMED: declining sub-qword label writes whose "
                            "destination holds a mapped non-heap pointer\n");
        return on;
    }();
    return v;
}
// True when this sub-qword write would destroy the low dword of a mapped pointer that lies outside
// the heap window. The content decision comes from `nonheap_guard_content()` — the SAME function the
// test exports — rather than a second copy of its conjunction, so removing the `!heap_ptr_like` term
// (the one line standing between this arm and #1245) fails a test instead of passing silently.
static bool declines_nonheap_ptr(const char* kind, uint64_t dst, uint64_t pre, uint64_t value,
                                 uint64_t width) {
    if (!nonheap_ptr_guard()) return false;
    if (!nonheap_guard_content(pre, width, value)) return false;
    if (!guest_readable(pre, 8)) return false;     // not a live pointer at all
    static std::atomic<uint64_t> n{0};
    const uint64_t ord = n.fetch_add(1, std::memory_order_relaxed) + 1;
    if (diag_should_print(ord)) {
        char hist[512]; label_hist_report(dst, hist, sizeof hist);
        fprintf(stderr, "[agc] NONHEAP-PTR-DECLINE #%llu kind=%s dst=0x%llx pre=0x%llx val=0x%llx "
                        "t=%llums | %s\n",
                (unsigned long long)ord, kind, (unsigned long long)dst, (unsigned long long)pre,
                (unsigned long long)value, (unsigned long long)now_ms(), hist);
    }
    return true;
}
// #312 ROOT fix gate (default ON; PROSPER_REL1_FORGE_GUARD=0 for A/B baseline). Suppress a fence
// write that would forge a freelist next-pointer (see forges_freelist_ptr), gated to the consumed-
// marker label population so plain fence labels (Messenger) are untouched.
static bool forge_guard() {
    // DEFAULT ON, and it is the #312 ROOT fix -- so `strtol` answering 0 for `=yes`/`=true`/`=on`
    // silently REMOVED the fix while the operator believed they had pinned it on (#3267).
    static const bool v = [] { const char* e = getenv("PROSPER_REL1_FORGE_GUARD");
                               return prosper::diag::env_u64_or_default_auto(
                                   "PROSPER_REL1_FORGE_GUARD", e, 1ull) != 0; }();   // default ON
    return v;
}
// #1226 decisive A/B arm: suppress EVERY REL1 write matching forges_freelist_ptr(), including the
// live-paired population the correctness guard above intentionally lets through. This is diagnostic
// only, default OFF, and deliberately unsafe for normal play: those live consumers will never see
// their fence. Combine it with PROSPER_INIT_SUPPRESS=ptr to remove both prosper-authored halves of
// ArcRunner's observed 0x2000000001 composition.
//
// The counters are part of the experiment, not decoration. A run is a valid all-forge-suppressed arm
// only when FORGE-DECISION-TOTALS says candidates == suppressed and landed == 0. Every one of the
// first 256 candidates prints, then every 256th. ArcRunner's worker-fault path calls `_exit(90)`, so
// a normal atexit summary cannot run; the dense prefix guarantees a terminal census for the combined
// arm's measured sub-128 population without making the diagnostic unbounded on other workloads.
namespace {
struct Rel1ForgeDecisionCounters {
    uint64_t candidates = 0;
    uint64_t suppressed = 0;
    uint64_t landed = 0;
};
Rel1ForgeDecisionCounters g_rel1_forge_decisions;
std::mutex g_rel1_forge_decisions_mu;
// -1 selects the real environment. Tests use 0=off, 1=suppress-all, and 2=observe-only so each
// counter is positive-controlled through the real store site in the same process.
std::atomic<int> g_rel1_forge_suppress_all_test_override{-1};

int rel1_forge_decision_mode() {
    const int test_override = g_rel1_forge_suppress_all_test_override.load(std::memory_order_relaxed);
    if (test_override >= 0) return test_override;
    static const int mode = [] {
        const char* e = getenv("PROSPER_REL1_FORGE_SUPPRESS_ALL");
        if (!e || !*e || !strcmp(e, "0") || !strcmp(e, "off")) return 0;
        if (!strcmp(e, "1") || !strcmp(e, "on") || !strcmp(e, "all")) {
            fprintf(stderr,
                    "[agc] PROSPER_REL1_FORGE_SUPPRESS_ALL=1 ARMED — diagnostic only; "
                    "valid arm requires candidates=suppressed and landed=0\n");
            return 1;
        }
        fprintf(stderr,
                "[agc] PROSPER_REL1_FORGE_SUPPRESS_ALL='%s' NOT ARMED "
                "(want 0|off|1|on|all)\n", e);
        return 0;
    }();
    return mode;
}

void rel1_forge_report_totals_locked(const char* suffix = "") {
    fprintf(stderr,
            "[agc] FORGE-DECISION-TOTALS candidates=%llu suppressed=%llu landed=%llu mode=all%s\n",
            (unsigned long long)g_rel1_forge_decisions.candidates,
            (unsigned long long)g_rel1_forge_decisions.suppressed,
            (unsigned long long)g_rel1_forge_decisions.landed, suffix);
}

bool rel1_forge_report_due(uint64_t candidates) {
    return candidates > 0 && (candidates <= 256 || (candidates % 256) == 0);
}

bool rel1_forge_suppress_candidate() {
    const int mode = rel1_forge_decision_mode();
    if (!mode) return false;
    std::lock_guard<std::mutex> lock(g_rel1_forge_decisions_mu);
    ++g_rel1_forge_decisions.candidates;
    if (mode != 1) return false;   // test-only observe mode positive-controls the landed counter
    ++g_rel1_forge_decisions.suppressed;
    if (rel1_forge_report_due(g_rel1_forge_decisions.candidates))
        rel1_forge_report_totals_locked();
    return true;
}

void rel1_forge_note_landed() {
    const int mode = rel1_forge_decision_mode();
    if (!mode) return;
    std::lock_guard<std::mutex> lock(g_rel1_forge_decisions_mu);
    ++g_rel1_forge_decisions.landed;
    // Any hit is a broken discriminator: fail visibly even when it is not on the periodic schedule.
    if (mode == 1) rel1_forge_report_totals_locked(" VIOLATION=FORGE_LANDED");
}
}  // namespace

extern "C" void prosper_rel1_forge_suppress_all_override_for_test(int value) {
    g_rel1_forge_suppress_all_test_override.store(value, std::memory_order_relaxed);
}
extern "C" void prosper_rel1_forge_decision_reset_for_test() {
    std::lock_guard<std::mutex> lock(g_rel1_forge_decisions_mu);
    g_rel1_forge_decisions = {};
}
extern "C" void prosper_rel1_forge_decision_totals(uint64_t* candidates, uint64_t* suppressed,
                                                     uint64_t* landed) {
    std::lock_guard<std::mutex> lock(g_rel1_forge_decisions_mu);
    if (candidates) *candidates = g_rel1_forge_decisions.candidates;
    if (suppressed) *suppressed = g_rel1_forge_decisions.suppressed;
    if (landed) *landed = g_rel1_forge_decisions.landed;
}
extern "C" bool prosper_rel1_forge_report_due_for_test(uint64_t candidates) {
    return rel1_forge_report_due(candidates);
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
// --- #1226 INIT-side measurement and A/B arm. ----------------------------------------------------
// Reuses the existing wide "looks like a heap pointer" predicate (defined below with the clock-fence
// records) rather than adding a fourth window: ArcRunner's arena and RHI objects sit in the
// 0x2400000000..0x3200000000 range that the DOLL-era ptr_like() cannot see.
namespace { bool clockfence_heapish(uint64_t v); }
static bool mb3_freelist_guard();   // defined below; init_trip refuses to be read as a control under it
// #1754 established the fence write is the SECOND half of ArcRunner's free-list corruption and left
// the first half at CONFIDENCE: MED — the paired 4-byte DMA_DATA immediate-zero init is inferred to
// be destroying the low dword of a live `FFreeBlock::NextFreeBlock`, but that was never measured
// directly. This is the measurement. On every 4-byte immediate-zero init to a consumed-marker label
// it reports, at the instant before the write lands:
//   overptr=1  the destination currently holds a heap-pointer-shaped qword with a NONZERO low dword,
//              i.e. this write is about to destroy a pointer rather than initialise a scratch label;
//   member=1   the destination is, right now, reachable from the guest allocator's own idx=1 free
//              chains (mb3_freelist_contains_stable).
// `overptr` alone cannot decide anything — a label the guest legitimately POPPED from its pool free
// list still carries the stale link, so the content is identical either way; that ambiguity is what
// left the claim at MED. `member` is the discriminator, because a popped block is off the chain.
// Log-only, default OFF, and it never gates a write.
//
// Printing follows trap 51 — ordinal on every line, first 64 then powers of two — but the TOTALS
// ride a separate, denser schedule (every 256) on their own `INIT-TRIP-TOTALS` line. That split is
// deliberate and was learned the hard way: the sparse schedule front-loads its samples, `member`
// positives here appear only after ordinal ~512, and quoting a detail line as the run's answer is
// what produced this comment's first, wrong version. The fractions are the result; a line is not.
// There is no flush at exit (the repro is killed by a timeout or a SIGSEGV, so atexit would not
// run) — up to 255 inits after the last multiple of 256 go unreported. The counters are cumulative,
// so that bounds the last figure's staleness rather than corrupting it.
static std::atomic<uint64_t> g_init_n{0}, g_init_overptr{0}, g_init_member{0}, g_init_both{0};
static void init_trip(const Pm4Command& c, uint64_t pre, uint32_t v32) {
    static const bool on = getenv("PROSPER_INIT_TRIP") != nullptr;
    if (!on || c.dd_bytes != 4 || v32 != 0) return;
    if (!label_is_consumed_marker(c.dd_dst)) return;
    // A membership null is only evidence if the CALL SITE can observe a positive. With
    // PROSPER_MB3_FREELIST_GUARD=1, honor_dma_data returns for exactly the member==true inits BEFORE
    // reaching here, so this would report member=0 for 100% of what it sees — with a green self-test
    // printed beside it. Refuse to be read that way.
    static const bool not_a_control = mb3_freelist_guard();
    if (not_a_control) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            fprintf(stderr, "[agc] INIT-TRIP NOT A CONTROL — PROSPER_MB3_FREELIST_GUARD is on. It "
                            "returns before this probe for exactly the member=1 population (and, "
                            "with PROSPER_GENERATION_GUARD, for the drifted-generation one too), "
                            "so member=0 here is structural, not a measurement.\n");
    }
    const bool overptr = clockfence_heapish(pre) && (uint32_t)pre != 0;
    // Membership can only ever be TRUE for a 0x20-aligned block base, so a misaligned label is a
    // STRUCTURAL member=0, indistinguishable from walked-and-not-found unless it is reported. Same
    // class of trap as the one above, one level down. This duplicates mb3_freelist.cpp's
    // plausible_node(), which is in that file's anonymous namespace and so cannot be called here;
    // if that gate changes, this must follow or the field silently starts lying.
    const bool aligned = c.dd_dst >= 0x10000 && !(c.dd_dst & 0x1full);
    Mb3FreelistMatch match{};
    const bool member = mb3_freelist_contains_stable(c.dd_dst, &match);
    const uint64_t ord = g_init_n.fetch_add(1, std::memory_order_relaxed) + 1;
    if (overptr) g_init_overptr.fetch_add(1, std::memory_order_relaxed);
    if (member)  g_init_member.fetch_add(1, std::memory_order_relaxed);
    if (overptr && member) g_init_both.fetch_add(1, std::memory_order_relaxed);
    // Detail lines are sparse (trap 51). TOTALS are printed every 256 as well, because the sparse
    // schedule front-loads its samples: a population that only appears late — which is exactly what
    // `member` does here — is absent from every early line, and quoting one of those as the run's
    // answer is the cap-as-count trap wearing a different hat.
    const bool detail = diag_should_print(ord);
    if (detail) {
        char self[256]; mb3_freelist_selftest(self, sizeof self);
        fprintf(stderr, "[agc] INIT-TRIP-SELFTEST #%llu %s\n", (unsigned long long)ord, self);
        fprintf(stderr,
                "[agc] INIT-TRIP #%llu dst=0x%llx pre=0x%llx overptr=%d aligned=%d "
                "member=%d(list=%u pool=0x%llx hops=%u) t=%llums\n",
                (unsigned long long)ord, (unsigned long long)c.dd_dst, (unsigned long long)pre,
                (int)overptr, (int)aligned,
                (int)member, match.list, (unsigned long long)match.pool_base, match.hops,
                (unsigned long long)now_ms());
    }
    if (detail || (ord % 256) == 0)
        fprintf(stderr, "[agc] INIT-TRIP-TOTALS n=%llu overptr=%llu member=%llu both=%llu t=%llums\n",
                (unsigned long long)g_init_n.load(std::memory_order_relaxed),
                (unsigned long long)g_init_overptr.load(std::memory_order_relaxed),
                (unsigned long long)g_init_member.load(std::memory_order_relaxed),
                (unsigned long long)g_init_both.load(std::memory_order_relaxed),
                (unsigned long long)now_ms());
}
// A/B arm for the same claim (PROSPER_INIT_SUPPRESS=off|ptr|member, default off). NOT a candidate
// fix — `ptr` deliberately suppresses on CONTENT, which #1245 proved cannot distinguish a live
// label from a freed block, so arming it will also drop legitimate inits. It exists so the
// counter-arm can be run: if prosper's own init is the first half of the damage, removing it must
// change the corruption, and if it does not, the first half is somewhere else entirely.
//   ptr    — suppress whenever this write would destroy a pointer-shaped qword (upper bound: every
//            init that could possibly be the first half, including the legitimate ones).
//   member — suppress only when the block is on the guest's free list right now (the honest form,
//            equivalent to the existing PROSPER_MB3_FREELIST_GUARD init leg without its other legs).
static int init_suppress_mode() {
    static const int v = [] {
        const char* e = getenv("PROSPER_INIT_SUPPRESS");
        if (!e || !*e) return 0;
        if (!strcmp(e, "ptr")) return 1;
        if (!strcmp(e, "member")) return 2;
        if (!strcmp(e, "off") || !strcmp(e, "0")) return 0;
        fprintf(stderr, "[agc] PROSPER_INIT_SUPPRESS='%s' unrecognized (want off|ptr|member) — OFF\n", e);
        return 0;
    }();
    return v;
}
static bool init_suppress(const Pm4Command& c, uint64_t pre, uint32_t v32) {
    const int mode = init_suppress_mode();
    if (!mode || c.dd_bytes != 4 || v32 != 0) return false;
    if (!label_is_consumed_marker(c.dd_dst)) return false;
    if (!(clockfence_heapish(pre) && (uint32_t)pre != 0)) return false;
    if (mode == 2 && !mb3_freelist_contains_stable(c.dd_dst)) return false;
    static std::atomic<uint64_t> n{0};
    const uint64_t ord = n.fetch_add(1, std::memory_order_relaxed) + 1;
    if (diag_should_print(ord))
        fprintf(stderr, "[agc] INIT-SUPPRESS #%llu mode=%s dst=0x%llx pre=0x%llx t=%llums\n",
                (unsigned long long)ord, mode == 1 ? "ptr" : "member",
                (unsigned long long)c.dd_dst, (unsigned long long)pre,
                (unsigned long long)now_ms());
    return true;
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
    // #1756: DMA_DATA is now 7 dwords — the hardware size, with no slot for the build snapshot this
    // check reads. It therefore only ever fires on a pre-#1756 capture. Say so ONCE when the guard is
    // enabled and the snapshot is absent, rather than returning a quiet false: an opt-in check that
    // has silently become inert is indistinguishable from one that ran and found nothing, and that
    // mistake has been made in this repository before. `prosper_label_hist_dma_built` still records
    // the same init event out-of-band, so the label-history legs of #312 are unaffected.
    // (Reached only with the guard enabled — its sole caller tests generation_guard() first.)
    if (!c.dd_build_pre_valid) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            fprintf(stderr, "[agc] PROSPER_GENERATION_GUARD: the DMA-init leg needs the pre-#1756 "
                            "9-dword DMA_DATA packet and is INERT on this build (#1756). The "
                            "RELEASE_MEM leg is unaffected.\n");
        return false;
    }
    if (!c.dd_build_pre_valid || c.dd_build_pre == pre) return false;
    if (build_pre) *build_pre = c.dd_build_pre;
    return true;
}
static void stale_dma_change_report(uint64_t addr, uint64_t build_pre, uint64_t pre, uint64_t pkt) {
    // #1761: same class as the tripwires above — a suppression reporter whose line count is read as
    // "how many writes were suppressed". Ordinal on every line, sparse tail past the cap.
    static std::atomic<uint64_t> n{0};
    const uint64_t ord = n.fetch_add(1, std::memory_order_relaxed) + 1;
    if (diag_should_print(ord, 256))
        fprintf(stderr, "[agc] DMA-GENERATION-CHANGED-STALE-SUPPRESS #%llu [0x%llx] build-pre=0x%llx "
                        "exec-pre=0x%llx pkt=0x%llx t=%llums\n",
                (unsigned long long)ord, (unsigned long long)addr, (unsigned long long)build_pre,
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
    // Only the HIGH half of the build snapshot is used, which is also the only half the 8-dword
    // ReleaseMem packet retains (#1748) — so print that half rather than `rel_build_pre` itself,
    // whose zero low dword would read as a measured value.
    uint64_t initialized = c.rel_build_pre & 0xffffffff00000000ull;
    uint64_t signaled = initialized | 1ull;
    if (pre == initialized || pre == signaled) return false;
    // #1761: as above — the count of this line is read as the suppressed-write population.
    static std::atomic<uint64_t> n{0};
    const uint64_t ord = n.fetch_add(1, std::memory_order_relaxed) + 1;
    if (diag_should_print(ord, 256))
        fprintf(stderr, "[agc] REL-GENERATION-CHANGED-STALE-SUPPRESS #%llu [0x%llx] "
                        "build-pre-hi=0x%08x expected-init=0x%llx exec-pre=0x%llx "
                        "pkt=0x%llx t=%llums\n",
                (unsigned long long)ord, (unsigned long long)c.rel_addr,
                (unsigned)(c.rel_build_pre >> 32),
                (unsigned long long)initialized, (unsigned long long)pre,
                (unsigned long long)pkt_addr(c), (unsigned long long)now_ms());
    label_hist_rel_free(c.rel_addr, 0);
    return true;
}
static void mb3_freelist_report(const char* kind, uint64_t addr, uint64_t pre,
                                const Mb3FreelistMatch* match, bool debt) {
    // #1761: the old rule (i < 64 || (i & 4095) == 0) put the 65th line on the 4,097th event and
    // carried no ordinal, so "64 suppressed live-protocol writes per ArcRunner run" — quoted on
    // #1226 — bounded the population only below 4,096 rather than measuring it.
    static std::atomic<uint64_t> n{0};
    const uint64_t ord = n.fetch_add(1, std::memory_order_relaxed) + 1;
    if (diag_should_print(ord))
        fprintf(stderr, "[agc] MB3-FREE-SUPPRESS #%llu kind=%s [0x%llx] pre=0x%llx via=%s "
                        "pool=0x%llx list=%u head=0x%llx hops=%u t=%llums\n",
                (unsigned long long)ord, kind, (unsigned long long)addr, (unsigned long long)pre,
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
    // #1761: ordinal on every line, budget PER KIND, sparse tail past the cap. The old form printed
    // at most 192 lines shared across ALL kinds and carried no ordinal, so (a) "192 suspect writes"
    // read as a census when it was the ceiling, and (b) a noisy REL1-LIVE could exhaust the budget
    // before REL1-FORGE — the kind under investigation — ever got a line.
    static const char* const kKinds[] = {"REL1-LIVE", "REL1-NOINIT", "REL1-FORGE", "REL2-LIVE",
                                         "WDATA"};
    static constexpr size_t kNKinds = sizeof(kKinds) / sizeof(kKinds[0]);
    static std::atomic<uint64_t> per_kind[kNKinds + 1];
    const uint64_t ord =
        per_kind[diag_key_slot(kindtag, kKinds, kNKinds)].fetch_add(1, std::memory_order_relaxed) + 1;
    if (!diag_should_print(ord, 192)) return;
    uint64_t baddr = 0, bpre = 0, bt = 0;
    int have = prosper_fence_journal_lookup(pkt, &baddr, &bpre, &bt, nullptr);
    char hist[512]; label_hist_report(addr, hist, sizeof hist);
    fprintf(stderr, "[agc] SUSPECT-%s #%llu [0x%llx] pre=0x%llx value=0x%llx pkt=0x%llx t=%llums | "
                    "journal:%s built@%llums(age=%lldms) built-addr=0x%llx%s pre@build=0x%llx%s | %s\n",
            kindtag, (unsigned long long)ord, (unsigned long long)addr, (unsigned long long)pre,
            (unsigned long long)value,
            (unsigned long long)pkt, (unsigned long long)now_ms(),
            have ? "" : " MISS", (unsigned long long)bt,
            have ? (long long)(now_ms() - bt) : -1,
            (unsigned long long)baddr, (have && baddr != addr) ? " TARGET-CHANGED" : "",
            (unsigned long long)bpre, (have && ptr_like(bpre)) ? " STALE-AT-BUILD" : "", hist);
}

// #1226 (arc5): the DMA-init staleness census — PROSPER_DMA_INIT_GEN=1, default OFF, log-only,
// never gates a write.
//
// This asks, at the moment prosper EXECUTES a 4-byte label init, the two questions the
// generation guard used to ask at the same point and can no longer answer. #1756 shrank DMA_DATA
// to its hardware 7 dwords and removed the slot `dma_build_pre_changed()` read, so its DMA leg is
// inert on every current build; both questions are still answerable out of band:
//
//   1. GENERATION DEPTH — `dma_built_n - dma_exec_n`, read BEFORE this exec's own bump (which
//      happens at the end of the immediate branch, after every guard). >= 2 means the guest has
//      already BUILT a later generation of this label's protocol while this one is still
//      unexecuted, i.e. two generations are in flight over one 0x20-byte block. That is the
//      init-side twin of `label_rel_overlap()`, which only sees the fence side.
//   2. CONTENT DRIFT — the fence build journal, keyed by the packet's guest address, records the
//      qword the target held when the GUEST built the packet. A consumed-marker label is
//      deliberately uninitialised at build time (malloc residue is EXPECTED), so residue alone
//      says nothing; what does say something is the residue CHANGING before we execute. Nothing
//      may legitimately write a label between its own build and its own init: the previous
//      generation's fence is older than this build, and the guest does not initialise the block
//      itself. A changed qword therefore means the block left the guest's label ownership in the
//      build->exec window — freed, recycled, and handed to a new owner whose first field we are
//      about to overwrite.
//
// The journal record for the DMA leg is added by this change (`agc_dcb_dma_data` /
// `agc_acb_dma_data` in hle_agc.cpp); the RELEASE_MEM, WRITE_DATA and WAIT_REG_MEM legs already
// recorded theirs. A journal MISS is reported as its own count rather than folded into either
// answer, so an unpopulated journal can never read as "no drift".
//
// The totals line rides `examined`, not any hit count, and prints on its own schedule: a run that
// ends in `_exit(90)` (the worker-fault path) never reaches an atexit handler, so a census that
// only printed at exit would be empty exactly on the runs that matter. CONFIDENCE: HIGH that the
// instrument reports what it says; it gates nothing.
namespace {
struct DmaInitGenTotals {
    uint64_t examined = 0, with_history = 0, journal_hit = 0, journal_miss = 0, target_changed = 0;
    uint64_t depth_ge2 = 0, drift_any = 0, drift_lo = 0;
    uint64_t age_max_ms = 0, age_sum_ms = 0, depth_max = 0;
};
std::mutex g_dma_init_gen_mu;
DmaInitGenTotals g_dma_init_gen;
bool dma_init_gen_trip_on() {
    static const bool v = [] {
        const char* e = getenv("PROSPER_DMA_INIT_GEN");
        const bool on = e && strtol(e, nullptr, 0) != 0;
        if (on)
            fprintf(stderr, "[agc] DMA-INIT-GEN ARMED: per-label generation depth and build->exec "
                            "content drift at DMA-init exec time (log-only, gates nothing)\n");
        return on;
    }();
    return v;
}
// One derivation, read by BOTH the census and the A/B guard below, so the guard can never decline
// on a condition the census does not report (and vice versa).
struct DmaInitProvenance {
    bool     has_history = false;   // this address owns a label-protocol history slot
    bool     have_journal = false;  // the guest's build of THIS packet is on record
    bool     same_target = false;   // ...and it named this same label address
    bool     drift = false;         // the qword changed between build and exec
    bool     drift_lo = false;      // ...in the low dword, the 4 bytes the init writes
    uint64_t bpre = 0, age_ms = 0;
    uint32_t depth = 0, db = 0, dx = 0;
    uint32_t build_fold = 0;        // #1226 (arc7): g_fold_seq when the guest built this packet
    uint32_t age_folds = 0;         // ...and how many folds ago that was, at this exec
};
DmaInitProvenance dma_init_provenance(uint64_t dst, uint64_t pre, uint64_t pkt) {
    DmaInitProvenance p;
    // Read-only lookup: this path must never evict a colliding label's protocol state (see
    // label_hist_find) — a probe that changes what the guards decide is not a probe.
    const LabelHist* h = label_hist_find(dst);
    p.has_history = h != nullptr;
    p.db = h ? h->dma_built_n.load(std::memory_order_relaxed) : 0;
    p.dx = h ? h->dma_exec_n.load(std::memory_order_relaxed) : 0;
    p.depth = p.db > p.dx ? p.db - p.dx : 0;   // this exec's own bump is downstream of here
    uint64_t baddr = 0, bt = 0;
    p.have_journal = prosper_fence_journal_lookup(pkt, &baddr, &p.bpre, &bt, &p.build_fold) != 0;
    p.same_target = p.have_journal && baddr == dst;
    p.age_ms = (p.have_journal && now_ms() > bt) ? now_ms() - bt : 0;
    const uint32_t fold_now = g_fold_seq.load(std::memory_order_relaxed);
    p.age_folds = (p.have_journal && fold_now > p.build_fold) ? fold_now - p.build_fold : 0;
    p.drift = p.same_target && p.bpre != pre;
    p.drift_lo = p.same_target && (uint32_t)p.bpre != (uint32_t)pre;
    return p;
}
// #1226 (arc7): the exec half of the fold-margin census. Called at the same point as
// dma_init_gen_trip — before every guard and before this exec's own dma_exec_n bump — so the two
// censuses count the same population and can be cross-read line for line.
void fold_margin_exec(uint64_t dst, uint64_t pre, uint32_t value, uint32_t width, uint64_t pkt) {
    if (!fold_margin_on()) return;
    if (width != 4 || value != 0) return;             // the exact consumed-marker initializer
    const DmaInitProvenance p = dma_init_provenance(dst, pre, pkt);
    bool report;
    {
        std::lock_guard<std::mutex> lk(g_fold_margin_mu);
        FoldMarginTotals& t = g_fold_margin;
        t.execs++;
        if (p.have_journal) {
            t.execs_journal++;
            t.age_hist[fold_bucket(p.age_folds)]++;
            t.age_fold_sum += p.age_folds;
            if (p.age_folds > t.age_fold_max) t.age_fold_max = p.age_folds;
            if (p.depth >= 2) { t.deep_execs++; t.deep_age_fold_sum += p.age_folds; }
        }
        report = (t.execs & 255) == 0 || p.depth >= 2;
    }
    g_fm_fold_execs.fetch_add(1, std::memory_order_relaxed);
    if (p.depth >= 2) g_fm_fold_deep.fetch_add(1, std::memory_order_relaxed);
    if (report) fold_margin_report(p.depth >= 2 ? "gen-overlap-exec" : "exec-cadence");
}
void dma_init_gen_trip(uint64_t dst, uint64_t pre, uint32_t value, uint32_t width, uint64_t pkt) {
    if (!dma_init_gen_trip_on()) return;
    if (width != 4 || value != 0) return;             // the exact consumed-marker initializer
    const DmaInitProvenance p = dma_init_provenance(dst, pre, pkt);
    const LabelHist* h = p.has_history ? label_hist_find(dst) : nullptr;
    const uint32_t db = p.db, dx = p.dx, depth = p.depth;
    const uint64_t bpre = p.bpre, age = p.age_ms;
    const bool have = p.have_journal, same_target = p.same_target;
    const bool drift = p.drift, drift_lo = p.drift_lo;
    uint64_t ord;
    char totals[256];
    bool print_totals;
    {
        std::lock_guard<std::mutex> lk(g_dma_init_gen_mu);
        DmaInitGenTotals& t = g_dma_init_gen;
        ord = ++t.examined;
        if (h) t.with_history++;
        if (have) t.journal_hit++; else t.journal_miss++;
        if (have && !same_target) t.target_changed++;
        if (depth >= 2) t.depth_ge2++;
        if (depth > t.depth_max) t.depth_max = depth;
        if (drift) t.drift_any++;
        if (drift_lo) t.drift_lo++;
        if (same_target) { t.age_sum_ms += age; if (age > t.age_max_ms) t.age_max_ms = age; }
        // The schedule must cover the population of interest, not merely the run. The first
        // version of this census printed every 256th examined write and the whole
        // generation-overlap population of the run fell into the unprinted tail after the last
        // multiple of 256 — the run died three tenths of a second later. So: every drift and every
        // generation overlap forces a totals line of its own, and the periodic cadence is 64.
        print_totals = (ord == 1) || (ord & 63) == 0 || drift || depth >= 2;
        if (print_totals)
            snprintf(totals, sizeof totals,
                     "[agc] DMA-INIT-GEN-TOTALS examined=%llu with-history=%llu journal(hit=%llu "
                     "miss=%llu target-changed=%llu) depth>=2=%llu depth-max=%llu drift(qword=%llu "
                     "lodword=%llu) age(max=%llums mean=%llums) t=%llums\n",
                     (unsigned long long)t.examined, (unsigned long long)t.with_history,
                     (unsigned long long)t.journal_hit,
                     (unsigned long long)t.journal_miss, (unsigned long long)t.target_changed,
                     (unsigned long long)t.depth_ge2, (unsigned long long)t.depth_max,
                     (unsigned long long)t.drift_any, (unsigned long long)t.drift_lo,
                     (unsigned long long)t.age_max_ms,
                     (unsigned long long)(t.journal_hit ? t.age_sum_ms / t.journal_hit : 0),
                     (unsigned long long)now_ms());
    }
    if (print_totals) fputs(totals, stderr);
    // Detail lines are budgeted PER CLASS, so the rare drift/deep-generation case can never be
    // starved by the common clean one sharing a single ordinal (the trap that cost #1226 a run).
    static std::atomic<uint64_t> n_clean{0}, n_drift{0}, n_deep{0};
    const bool deep = depth >= 2;
    if (!drift && !deep) { if (!diag_should_print(n_clean.fetch_add(1) + 1, 8)) return; }
    else if (drift)      { if (!diag_should_print(n_drift.fetch_add(1) + 1, 64)) return; }
    else                 { if (!diag_should_print(n_deep.fetch_add(1) + 1, 64)) return; }
    char hist[512]; label_hist_report(dst, hist, sizeof hist);
    fprintf(stderr, "[agc] DMA-INIT-GEN #%llu [0x%llx] depth=%u(built=%u exec=%u) "
                    "pre@build=0x%llx pre@exec=0x%llx%s%s journal=%s%s age=%llums pkt=0x%llx "
                    "t=%llums | %s\n",
            (unsigned long long)ord, (unsigned long long)dst, depth, db, dx,
            (unsigned long long)bpre, (unsigned long long)pre,
            drift ? " DRIFTED" : "", deep ? " GEN-OVERLAP" : "",
            have ? "hit" : "MISS", (have && !same_target) ? " TARGET-CHANGED" : "",
            (unsigned long long)age, (unsigned long long)pkt, (unsigned long long)now_ms(), hist);
}

// #1226 (arc5) A/B arm, PROSPER_DMA_INIT_DRIFT_GUARD=1, default OFF: decline a 4-byte label init
// whose target's content CHANGED between the guest building the packet and prosper executing it.
//
// This is the check `dma_build_pre_changed()` performs, re-armed from the build journal instead of
// the in-packet snapshot #1756 removed — and gated on the DRIFT alone, with none of the
// `mb3_freelist_guard()` membership walk the old DMA leg additionally required. That coupling is
// why the previous A/B could not separate the two: the recorded worst state was "generation off,
// membership on", and the arms that measured worse ran both.
//
// Why drift is the right predicate and residue is not. A consumed-marker label is DELIBERATELY
// uninitialised when its packet is built — stale `FFreeBlock` residue is expected there, and every
// content-shape guard that tried to read staleness out of it repeated #1245. What residue cannot
// be is UNSTABLE: between the guest building this init and prosper executing it, nothing may
// legitimately write the block. The paired fence is younger than the build, the guest does not
// initialise the label itself, and no other generation of the protocol owns the address while this
// one is outstanding. So a changed qword means the block left the guest's label ownership inside
// that window — freed, recycled, and handed to a new owner whose first field this init is about to
// zero, which is the first half of the `{stale high dword, 1}` forge.
//
// Declining the init alone is deliberately sufficient for the pair on the population it was
// measured against: with the low dword left holding a real pointer half, the default-ON
// `rel1_stomp_guard()` declines the paired fence on its own predicate (it cannot fire today only
// because our init zeroes that dword first). The dma-free debt is still recorded so the MB3 release
// leg also declines when THAT guard is armed.
//
// Measured on PPSA21406 with `PROSPER_DMA_INIT_GEN=1`, three default runs against two
// `PROSPER_SUBMIT_STALL_US=1500` runs of one build: **14 / 26 / 25** drifts in ~524 journal-matched
// inits per faulting run, against **0** in 10,079 on a surviving one. The predicate fires only in
// the arm that faults, which is the specificity control a content-shape guard has never had here.
//
// That specificity cuts BOTH ways, and the honest reading is the unflattering one: nothing about
// the title differs between those arms — only prosper's own schedule does — so drift is a measure
// of prosper's timing rather than of the guest. Acting on it declines two writes the guest asked
// for and hardware performs, instead of performing them at the right time. This is a LEVER for
// isolating the race, not a candidate fix, and it must not be made default-ON on the strength of
// the progression it buys (see `docs/ARCRUNNER_STATUS.md`). CONFIDENCE: HIGH that it is a lever;
// MED on the drift predicate's own soundness (see the journal-rebuild caveat in that document).
// Levels, so the init decline and the paired-fence decline are SEPARABLE arms. They are not the
// same experiment and the first head of this change conflated them: level 1 alone took a default
// run from 34 to 271 delivered video frames, and level 2 on the same build produced a fault class
// neither arm had shown before. One env var with ordinal levels rather than two booleans, because
// "1" and "2" cannot be set in a combination that means nothing.
//   1 = decline the drifted init only (the paired fence still runs its own default guards)
//   2 = also retire that generation's paired fence through the dma-free debt
int dma_init_drift_level() {
    static const int v = [] {
        const char* e = getenv("PROSPER_DMA_INIT_DRIFT_GUARD");
        if (!e) return 0;
        char* end = nullptr;
        const long parsed = strtol(e, &end, 0);
        if (end == e || (end && *end) || parsed < 0 || parsed > 2) {
            fprintf(stderr, "[agc] DMA-INIT-DRIFT-GUARD NOT ARMED: "
                            "PROSPER_DMA_INIT_DRIFT_GUARD='%s' is not 0, 1 or 2\n", e);
            return 0;
        }
        if (parsed >= 1)
            fprintf(stderr, "[agc] DMA-INIT-DRIFT-GUARD ARMED level=%ld: declining a 4-byte label "
                            "init whose target changed content between the guest's build and this "
                            "exec%s\n", parsed,
                    parsed >= 2 ? ", and retiring that generation's paired fence" : "");
        return (int)parsed;
    }();
    return v;
}
bool dma_init_drift_guard() { return dma_init_drift_level() >= 1; }
bool declines_drifted_init(uint64_t dst, uint64_t pre, uint32_t value, uint32_t width,
                           uint64_t pkt) {
    if (!dma_init_drift_guard()) return false;
    if (width != 4 || value != 0) return false;        // the exact consumed-marker initializer
    const DmaInitProvenance p = dma_init_provenance(dst, pre, pkt);
    if (!p.has_history || !p.same_target || !p.drift) return false;
    // Counted uncapped, printed sparsely: the LINE count of a suppression reporter is read as the
    // suppressed population often enough that #1761 had to fix it twice.
    static std::atomic<uint64_t> n{0};
    const uint64_t ord = n.fetch_add(1, std::memory_order_relaxed) + 1;
    if (diag_should_print(ord, 256)) {
        char hist[512]; label_hist_report(dst, hist, sizeof hist);
        fprintf(stderr, "[agc] DMA-INIT-DRIFT-DECLINE #%llu [0x%llx] pre@build=0x%llx "
                        "pre@exec=0x%llx depth=%u(built=%u exec=%u) age=%llums pkt=0x%llx "
                        "t=%llums | %s\n",
                (unsigned long long)ord, (unsigned long long)dst, (unsigned long long)p.bpre,
                (unsigned long long)pre, p.depth, p.db, p.dx, (unsigned long long)p.age_ms,
                (unsigned long long)pkt, (unsigned long long)now_ms(), hist);
    }
    return true;
}
// The declined init's PAIRED FENCE must be declined too, and an earlier head of this change
// assumed `rel1_stomp_guard()` would do it for free. It does not, and the terminal fault of the
// arm that assumed so named the reason exactly: the surviving pre-content was `0x21c0f88b70`,
// which is above `ptr_like_narrow()`'s `[0x2000000000, 0x2100000000)` ceiling, so the fence guard
// could not see it and wrote `1` over the pointer's low dword anyway — producing `0x2100000001`,
// the value in `rdi` at the fault. Declining half a pair is worse than declining neither: the
// fence alone forges `{stale high dword, 1}` without any help from the init.
//
// So the decline is carried across the pair explicitly, by the dma-free debt the init records.
// `label_hist_take_dma_free_debt` is a CAS, so each declined init retires exactly one fence and a
// later live generation at the same address is untouched. No `rel_exec_n` bump: a suppressed fence
// signalled nothing, and advancing the counter would make the NEXT real pair read as stale.
bool declines_drifted_pair_release(uint64_t addr, uint64_t value, const char* kind) {
    if (dma_init_drift_level() < 2 || value != 1) return false;
    if (!label_hist_take_dma_free_debt(addr)) return false;
    static std::atomic<uint64_t> n{0};
    const uint64_t ord = n.fetch_add(1, std::memory_order_relaxed) + 1;
    if (diag_should_print(ord, 256))
        fprintf(stderr, "[agc] DMA-INIT-DRIFT-PAIR-DECLINE #%llu kind=%s [0x%llx] pre=0x%llx "
                        "t=%llums\n",
                (unsigned long long)ord, kind, (unsigned long long)addr,
                (unsigned long long)peek_qword(addr), (unsigned long long)now_ms());
    label_hist_rel_free(addr, 0);
    return true;
}
}   // namespace

// #312 the WHERE fix (default ON; PROSPER_REL1_STOMP_GUARD=0 restores the old barrel-through for
// A/B). When a data_sel==1 fence write would land on a live MallocBinned3 freelist/pool block —
// captured live as REL1-LIVE: the destination qword is a heap pointer (ptr_like) whose low dword is
// a real pointer half, not 0 (init'd) or 1 (signaled) — the paired DmaData init never ran and the
// guest already freed the recycled label back to the allocator. Writing our 4-byte 1 over +0 forges
// 0x10000000_00000001 (or a 0x2001... pool pointer) — the exact #312 fatal family. Suppress it.
static bool rel1_stomp_guard() {
    // DEFAULT ON -- same inversion as PROSPER_REL1_FORGE_GUARD, same #312 fatal family (#3267).
    static const bool v = [] { const char* e = getenv("PROSPER_REL1_STOMP_GUARD");
                               return prosper::diag::env_u64_or_default_auto(
                                   "PROSPER_REL1_STOMP_GUARD", e, 1ull) != 0; }();   // default ON
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
// Diagnostic-only "looks like a heap pointer" — deliberately wider than the default ptr_like():
// ArcRunner's MallocBinned3 arena and RHI objects live above 0x2100000000 (#1226 FAULTOBJ dumps:
// objects at 0x2420e48000 / 0x3152b50000 / 0x316366c154), which the DOLL-era windows predate.
// Flags records for the reader, and is read by ONE suppression path — `init_suppress()`, the
// default-OFF `PROSPER_INIT_SUPPRESS=ptr` A/B arm. It gates no default-boot decision.
//
// #1226: this is now a thin alias of heap_ptr_like() rather than its own fourth window. The old
// body stopped at 0x4000000000 and so still missed the top half of ArcRunner's
// [0x2000000000, 0xa000000000) arena. Two consequences, both deliberate:
//   * `overptr` figures recorded before this change (e.g. the "n=1024 overptr=926" init census
//     quoted above) were taken through the narrower window. They are LOWER BOUNDS and must not be
//     compared numerically against figures taken after it.
//   * `PROSPER_INIT_SUPPRESS=ptr` now drops a superset of the inits it used to. That arm is an
//     upper-bound counter-arm by design ("suppress every init that could possibly be the first
//     half"), so a wider superset is consistent with its stated purpose — but a before/after
//     comparison of its results is not valid either.
bool clockfence_heapish(uint64_t v) {
    return heap_ptr_like(v);
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
        // #1761: its count is quoted the same way the forge tripwire's was, so it carries an
        // ordinal on every line and keeps printing sparsely past the cap.
        static std::atomic<uint64_t> n{0};
        if (clockfence_log()) {
            const uint64_t ord = n.fetch_add(1, std::memory_order_relaxed) + 1;
            if (diag_should_print(ord))
                fprintf(stderr,
                    "[agc] CLOCKFENCE-OVER-PTR #%llu kind=%u [0x%llx] pre=0x%llx clock=0x%llx pkt=0x%llx t=%llums\n",
                    (unsigned long long)ord, kind, (unsigned long long)addr, (unsigned long long)pre,
                    (unsigned long long)value, (unsigned long long)pkt, (unsigned long long)t);
        }
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
                  // #1226 A/B: retire the fence of a generation whose init this run declined for
                  // build->exec content drift. Inert unless PROSPER_DMA_INIT_DRIFT_GUARD=1.
                  if (declines_drifted_pair_release(c.rel_addr, c.rel_value, "REL1")) return;
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
                  bool rel1_forge_candidate = false;
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
                      rel1_forge_candidate = true;
                      forge_trip("REL1", c.rel_addr, pre, c.rel_value, 4, pkt_addr(c));
                      if (rel1_forge_suppress_candidate()) {
                          // Preserve diagnostic history without claiming a completed generation.
                          label_hist_event(c.rel_addr, LE_REL_EXEC, pre, c.queue_origin);
                          return;
                      }
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
                      static std::atomic<uint64_t> novl{0};
                      if (now_ms() >= 2000) {
                          // #1761: ordinal on every line and a sparse tail — a bare 64-line cap made
                          // this tripwire's ceiling indistinguishable from its population.
                          const uint64_t ord = novl.fetch_add(1, std::memory_order_relaxed) + 1;
                          if (diag_should_print(ord)) {
                              char hist[512]; label_hist_report(c.rel_addr, hist, sizeof hist);
                              fprintf(stderr, "[agc] SUSPECT-REL1-OVERLAP #%llu [0x%llx] pending-inits=%d pre=0x%llx t=%llums | %s\n",
                                      (unsigned long long)ord, (unsigned long long)c.rel_addr, ov,
                                      (unsigned long long)pre, (unsigned long long)now_ms(), hist);
                          }
                      }
                  }
                  uint32_t v = (uint32_t)c.rel_value;
                  write_trap_scan("REL1", c.rel_addr, pre, &v, sizeof v, pkt_addr(c));
                  liveptr_trip("REL1", c.rel_addr, pre, v, 4, pkt_addr(c));   // #1226 windowless census
                  if (declines_nonheap_ptr("REL1", c.rel_addr, pre, v, 4)) {   // #1226 A/B
                      // A declined fence must not advance the consumed count — same protocol rule
                      // the REL1-LIVE and REL1-FORGE suppressions follow above.
                      label_hist_event(c.rel_addr, LE_REL_EXEC, pre, c.queue_origin);
                      return;
                  }
                  if (rel1_forge_candidate) rel1_forge_note_landed();
                  memcpy(dst, &v, sizeof v);
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
                  uint64_t v = c.rel_value;
                  write_trap_scan("REL2", c.rel_addr, pre, &v, sizeof v, pkt_addr(c));
                  memcpy(dst, &v, sizeof v);
                  ring_record(c.rel_addr, v, 8, 1, pkt_addr(c));
                  poolshift_check("REL2", c.rel_addr, 8, c.rel_value, pkt_addr(c)); break; }
        case 3: { uint64_t pre = 0; memcpy(&pre, dst, sizeof pre);   // #1226 provenance pre-read
                  uint64_t v = gpu_clock64();
                  write_trap_scan("REL3", c.rel_addr, pre, &v, sizeof v, pkt_addr(c));
                  memcpy(dst, &v, sizeof v);
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
    set_guest_gpu_write_origin("RELEASE_MEM");
    notify_guest_gpu_write(c.rel_addr, c.rel_data_sel == 1 ? 4 : 8);
    set_guest_gpu_write_origin(nullptr);
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
    write_trap_scan("EVENT", c.event_addr, pre, &v, sizeof v, pkt_addr(c));
    memcpy((void*)(uintptr_t)c.event_addr, &v, sizeof v);
    set_guest_gpu_write_origin("EVENT_WRITE");
    notify_guest_gpu_write(c.event_addr, sizeof v);
    set_guest_gpu_write_origin(nullptr);
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
// address-backed sibling: a selected address source is copied only when the whole source and
// destination spans are mapped. Raw selector arguments do not distinguish the forms: the title's
// captured immediate-zero call passes 3/3, values which overlap the memory/L2 vocabulary. New HLE
// packets therefore preserve sourceKind in kDmaDataAddressSource; historical packets retain the
// established >32-bit address fallback. GDS offsets and malformed/unmapped endpoints fail closed.
// CONFIDENCE: HIGH on immediate fill and mapped address-copy behavior; MED on the large-fill form.
enum class DmaDataForm { Invalid, Immediate, Copy, GdsImmediate, MemoryToGds };

// The destination selector occupies the LOW byte of the packed `dd_sels` word: the HLE builder writes
// `(a2 & 0xff) | ((a3 & 0xff) << 8)`, and a2 is the argument that tracks the destination DOMAIN. This
// resolves the `srcSel?`/`dstSel?` question the HLE prototype left open at CONFIDENCE: MED — the two
// names are transposed. `agc_acb_dma_data` packs the FIRST of its two selector arguments into the low
// byte as well, so one rule covers both builders; `Pm4Command::queue_origin` can split them if a title
// ever forces it.
//
// Evidence, all of it title evidence rather than a hardware table:
//  - Across a routed Astro Bot run the low byte partitions destinations with no overlap: 1 appears
//    only with GDS-sized offsets (0x4, 0x24, 0xc64, 0xc68, 0xc6c, 0xc70, 0xc74, 0xc78, 0xc7c) and 3
//    only with full 64-bit guest addresses.
//  - Sony's parameter is `dstAddressOrOffset` (export `sceAgcDmaDataPatchSetDstAddressOrOffset`), so
//    an offset domain exists; GDS is the only offset domain DMA_DATA has, and `sceAgcDcbAtomicGds` /
//    `sceAgcAcbAtomicGds` / `sceAgcDriverRegisterGdsResource` are published exports.
//  - DOLL's #312 packet (immediate zero into a memory label) passes 3 in the low byte, consistent
//    with 3 meaning a memory destination.
//
// Do NOT read these values as PM4's: that same DOLL packet passes 3 in the HIGH byte for a source
// that is demonstrably an immediate, where PM4 encodes DATA as 2. The AGC enum is its own vocabulary,
// so the specific value 1 is pinned by prosper's title evidence above and by nothing else.
// CONFIDENCE: MED on dst_sel==1 meaning GDS — the partition is clean and the API naming corroborates
// it, but the sample is a handful of call sites, and no shader has yet been shown to READ these
// offsets. The guards below keep every unrecognised form fail-closed.
static constexpr uint32_t kDmaSelGds = 1;
static constexpr uint32_t kDmaSelMemory = 3;
static uint32_t dma_data_dst_sel(const Pm4Command& c) { return c.dd_sels & 0xffu; }
static uint32_t dma_data_src_sel(const Pm4Command& c) { return (c.dd_sels >> 8) & 0xffu; }
static bool dma_data_address_source(const Pm4Command& c) {
    // New HLE packets preserve sourceKind explicitly. Keep the numeric fallback for historical
    // packets/captures, which predate the metadata bit but used prosper's 64-bit address domain.
    return (c.dd_sels & kDmaDataAddressSource) != 0 || c.dd_src > UINT32_MAX;
}
static bool dma_data_immediate_source(const Pm4Command& c) {
    return !dma_data_address_source(c) && c.dd_src <= UINT32_MAX;
}

static DmaDataForm dma_data_form(const Pm4Command& c, bool source_materialized = false) {
    constexpr uint32_t kMaxImmediateBytes = 0x1000000;   // existing 16 MiB fill safety bound
    constexpr uint32_t kMaxCopyBytes = 0x10000000;      // HLE builder's 256 MiB API bound
    if (!c.dd_valid || !c.dd_bytes || c.dd_bytes > kMaxCopyBytes) return DmaDataForm::Invalid;
    // A GDS destination is an OFFSET into the 64 KiB share, not a guest address, so it is legitimately
    // below the 0x10000 floor that rejects malformed guest pointers — which is exactly why every one
    // of these was being discarded. Astro Bot now provides exact title evidence for the address
    // sibling too: dstSel=1/srcSel=3/sourceKind=2 copies four bytes from guest memory into GDS+0x24.
    // Keep that contract narrow; GDS-to-memory and every unrecognised selector direction stay
    // fail-closed.
    // A GDS SOURCE would make `dd_src` a small offset, which the immediate path below cannot tell from
    // a 32-bit fill value — it would write the offset itself into guest memory. There is no title
    // evidence for that form, so reject it rather than mis-execute it. (This is a behaviour change
    // only for a packet whose high selector byte is 1, which no observed title emits.)
    if (dma_data_src_sel(c) == kDmaSelGds) return DmaDataForm::Invalid;
    if (dma_data_dst_sel(c) == kDmaSelGds) {
        const size_t gds_size = compute_gds_size();
        const bool in_range = c.dd_dst < gds_size && c.dd_bytes <= gds_size - c.dd_dst;
        if (!in_range || (c.dd_dst & 3) || (c.dd_bytes & 3)) return DmaDataForm::Invalid;
        if (dma_data_immediate_source(c)) return DmaDataForm::GdsImmediate;
        return dma_data_src_sel(c) == kDmaSelMemory &&
                       (source_materialized || guest_readable(c.dd_src, c.dd_bytes))
                   ? DmaDataForm::MemoryToGds
                   : DmaDataForm::Invalid;
    }
    if (c.dd_dst < 0x10000) return DmaDataForm::Invalid;
    // sourceKind=2 is authoritative even for a low or currently-unmapped source address. Legacy
    // packets without that metadata retain the established >32-bit address discriminator.
    if (dma_data_immediate_source(c)) {
        // The immediate path peeks one qword for the consumed-marker safety journal.
        const uint32_t probe_bytes = c.dd_bytes < 8 ? 8 : c.dd_bytes;
        return c.dd_bytes <= kMaxImmediateBytes && !(c.dd_dst & 3) &&
                       guest_readable(c.dd_dst, probe_bytes) &&
                       (guest_writable(c.dd_dst, c.dd_bytes) ||
                        prosper::guest_memory_gpu_write_supported(c.dd_dst, c.dd_bytes))
                   ? DmaDataForm::Immediate
                   : DmaDataForm::Invalid;
    }
    return (source_materialized || guest_readable(c.dd_src, c.dd_bytes)) &&
                   (guest_writable(c.dd_dst, c.dd_bytes) ||
                    prosper::guest_memory_gpu_write_supported(c.dd_dst, c.dd_bytes))
               ? DmaDataForm::Copy
               : DmaDataForm::Invalid;
}

static void report_invalid_dma_data(const Pm4Command& c) {
    // A skipped immediate init leaves the label pointer-valued. Address-copy failures have no
    // consumed-marker protocol leg and must not alter those lifecycle counters.
    if (dma_data_immediate_source(c)) label_hist_dma_skip(c.dd_dst);
    static std::atomic<int> n{0};
    if (n.fetch_add(1) < 24)
        fprintf(stderr, "[agc] DMA_DATA not executed (invalid/unmapped form): dst=0x%llx src=0x%llx bytes=%u sels=0x%x\n",
                (unsigned long long)c.dd_dst, (unsigned long long)c.dd_src, c.dd_bytes, c.dd_sels);
}

static bool honor_dma_data(const Pm4Command& c, uint64_t retained_packet_addr = 0,
                           const uint8_t* authoritative_source = nullptr) {
    if (eop_writes_disabled() || !c.dd_valid) return false;
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
        return false;
    }
    if (form == DmaDataForm::GdsImmediate || form == DmaDataForm::MemoryToGds) {
        uint8_t* gds = compute_gds_backing();
        if (form == DmaDataForm::GdsImmediate) {
            // A GDS counter reset: the guest zeroes these offsets every frame, and shaders reach the
            // same 64 KiB backing through the internal binding. NOT the cause of #1742: with these
            // immediate resets restored, Astro's indirect dispatch still grew without bound. Do not
            // re-derive that falsified reset hypothesis.
            const uint32_t v32 = (uint32_t)c.dd_src;
            for (uint32_t i = 0; i + 4 <= c.dd_bytes; i += 4)
                memcpy(gds + c.dd_dst + i, &v32, 4);
            if (getenv("PROSPER_GFXLOG") || getenv("PROSPER_GDSLOG"))
                fprintf(stderr, "[agc]   DmaData GDS fill [gds+0x%llx] := 0x%x (%u bytes)\n",
                        (unsigned long long)c.dd_dst, v32, c.dd_bytes);
        } else {
            const uint8_t* source = authoritative_source
                ? authoritative_source
                : reinterpret_cast<const uint8_t*>(uintptr_t(c.dd_src));
            memcpy(gds + c.dd_dst, source, c.dd_bytes);
            if (getenv("PROSPER_GFXLOG") || getenv("PROSPER_GDSLOG")) {
                static std::atomic<uint64_t> ordinal{0};
                const uint64_t n = ordinal.fetch_add(1, std::memory_order_relaxed) + 1;
                if (prosper::diag_should_print(n, 16))
                    fprintf(stderr,
                            "[agc] DmaData memory-to-GDS #%llu [gds+0x%llx] <- "
                            "[0x%llx] (%u bytes order=%llu sels=0x%x)\n",
                            (unsigned long long)n, (unsigned long long)c.dd_dst,
                            (unsigned long long)c.dd_src, c.dd_bytes,
                            (unsigned long long)c.stream_order, c.dd_sels);
            }
        }
        return true;
    }
    uint8_t* dst = (uint8_t*)(uintptr_t)c.dd_dst;
    bool used_direct_backing = false;
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
                return false;
            }
            Mb3FreelistMatch match{};
            if (mb3_freelist_contains_stable(c.dd_dst, &match)) {
                label_hist_dma_free(c.dd_dst, match.pool_base);
                mb3_freelist_report("DMA", c.dd_dst, pre_dma, &match, false);
                return false;
            }
        }
        forge_trip("DMA", c.dd_dst, pre_dma, v32, 4, packet_addr);
        // #1226 (arc5): generation depth + build->exec content drift, before any guard and before
        // this exec's own dma_exec_n bump (which is at the end of this branch).
        dma_init_gen_trip(c.dd_dst, pre_dma, v32, c.dd_bytes, packet_addr);
        fold_margin_exec(c.dd_dst, pre_dma, v32, c.dd_bytes, packet_addr);   // #1226 (arc7)
        if (declines_drifted_init(c.dd_dst, pre_dma, v32, c.dd_bytes, packet_addr)) {  // #1226 A/B
            label_hist_dma_skip(c.dd_dst);
            label_hist_dma_free(c.dd_dst, 0);   // debt: the MB3 release leg declines the pair too
            return false;
        }
        init_trip(c, pre_dma, v32);   // #1226: is THIS write the first half of the corruption?
        if (init_suppress(c, pre_dma, v32)) {   // #1226 A/B arm; default OFF
            label_hist_dma_skip(c.dd_dst);
            return false;
        }
        liveptr_trip("DMA-imm", c.dd_dst, pre_dma, v32, c.dd_bytes, packet_addr);   // #1226
        if (declines_nonheap_ptr("DMA-imm", c.dd_dst, pre_dma, v32, c.dd_bytes)) {   // #1226 A/B
            label_hist_dma_skip(c.dd_dst);
            return false;
        }
        if (guest_writable(c.dd_dst, c.dd_bytes)) {
            if (v32 == 0) {
                memset(dst, 0, c.dd_bytes);
            } else {
                uint32_t i = 0;
                for (; i + 4 <= c.dd_bytes; i += 4) memcpy(dst + i, &v32, 4);
                if (i < c.dd_bytes) memcpy(dst + i, &v32, c.dd_bytes - i);
            }
        } else {
            // Device fills are allowed to target CPU-read-only direct memory too. Materialize a
            // bounded repeated chunk and write the physical backing without changing VA protection.
            constexpr size_t kFillChunk = 64u * 1024u;
            std::vector<uint8_t> fill(std::min<size_t>(c.dd_bytes, kFillChunk));
            for (size_t i = 0; i < fill.size(); ++i)
                fill[i] = reinterpret_cast<const uint8_t*>(&v32)[i & 3u];
            size_t done = 0;
            while (done < c.dd_bytes) {
                const size_t take = std::min<size_t>(fill.size(), c.dd_bytes - done);
                if (!dma_backing_write(c.dd_dst + done, fill.data(), take)) {
                    report_invalid_dma_data(c);
                    return false;
                }
                done += take;
            }
            used_direct_backing = true;
        }
        // The value trap answers "which prosper write STORED this dword". Attribute only after the
        // whole fallible backing operation succeeds; a partial/failing write is conservatively
        // dirtied by guest_memory_gpu_write but is not a proven full payload store.
        write_trap_scan("DMA-imm", c.dd_dst, pre_dma, &v32, sizeof v32, packet_addr);
        poolshift_check("DMA", c.dd_dst, c.dd_bytes, c.dd_src, packet_addr);
        if (c.dd_bytes <= 8) label_hist_dma_exec(c.dd_dst, pre_dma, c.queue_origin);
        if (getenv("PROSPER_GFXLOG"))
            fprintf(stderr, "[agc]   DmaData fill [0x%llx] := 0x%x (%u bytes)\n",
                    (unsigned long long)c.dd_dst, v32, c.dd_bytes);
    } else {
        // memmove is byte-for-byte memcpy behavior for the normal non-overlapping GPU-buffer case,
        // while remaining deterministic if an unusual packet overlaps its endpoints.
        const uint8_t* copy_src = authoritative_source ? authoritative_source
                                                       : (const uint8_t*)(uintptr_t)c.dd_src;
        // The value trap is provenance for the payload this command stored.  Snapshot its bounded
        // prefix before an overlap-capable copy: after memmove (including hidden physical-alias
        // overlap), copy_src may already name bytes overwritten by the destination.  Publication
        // still happens only after the complete store succeeds below.
        std::vector<uint8_t> write_trap_payload;
        if (write_trap_armed()) {
            const size_t trap_bytes = std::min<size_t>(c.dd_bytes, 4096);
            write_trap_payload.assign(copy_src, copy_src + trap_bytes);
        }
        // Different writable guest VAs can still overlap in the same physical direct backing.
        // std::memmove sees only the virtual ranges and cannot choose the physical copy direction,
        // so route that exact topology through the backing-aware helper as well.
        const bool hidden_direct_overlap = !authoritative_source &&
            guest_writable(c.dd_dst, c.dd_bytes) &&
            prosper::guest_memory_gpu_write_supported(c.dd_dst, c.dd_bytes) &&
            prosper::guest_memory_topology_relation(c.dd_dst, c.dd_bytes, c.dd_src, c.dd_bytes) ==
                prosper::GuestMemoryTopologyRelation::Overlap;
        if (guest_writable(c.dd_dst, c.dd_bytes) && !hidden_direct_overlap) {
            memmove(dst, copy_src, c.dd_bytes);
        } else if (!dma_backing_write(c.dd_dst, copy_src, c.dd_bytes)) {
            report_invalid_dma_data(c);
            return false;
        } else {
            used_direct_backing = true;
        }
        // #1226 value trap: bound the scan — a copy DMA can move megabytes and this is diagnostic.
        // `pre` is 0 because bulk destination pre-content is not meaningful. As above, scan only
        // after the complete virtual or physical write has succeeded.
        if (!write_trap_payload.empty())
            write_trap_scan("DMA-copy", c.dd_dst, 0, write_trap_payload.data(),
                            write_trap_payload.size(), packet_addr);
        if (getenv("PROSPER_GFXLOG"))
            fprintf(stderr, "[agc]   DmaData copy [0x%llx] <- [0x%llx] (%u bytes)\n",
                    (unsigned long long)c.dd_dst, (unsigned long long)c.dd_src, c.dd_bytes);
    }
    if (used_direct_backing && getenv("PROSPER_DMA_BACKING_LOG")) {
        // Exact positive control for device writes that bypass a CPU-read-only guest VA. Keep a
        // separate ordinal per form and a sparse tail so neither kind can consume the other's log
        // budget and the print count can never be mistaken for the event population.
        static std::atomic<uint64_t> ordinals[2]{};
        const size_t slot = form == DmaDataForm::Immediate ? 0 : 1;
        const uint64_t ordinal = ordinals[slot].fetch_add(1, std::memory_order_relaxed) + 1;
        if (prosper::diag_should_print(ordinal, 8))
            fprintf(stderr,
                    "[agc] DMA_DATA direct-backing %s #%llu dst=0x%llx bytes=%u\n",
                    slot == 0 ? "fill" : "copy", (unsigned long long)ordinal,
                    (unsigned long long)c.dd_dst, c.dd_bytes);
    }
    set_guest_gpu_write_origin("DMA_DATA");
    notify_guest_gpu_write(c.dd_dst, c.dd_bytes);
    set_guest_gpu_write_origin(nullptr);
    ring_record(c.dd_dst, c.dd_src, (uint8_t)(c.dd_bytes > 255 ? 255 : c.dd_bytes), 4, packet_addr);
    if (writer_provenance_enabled() &&
        (c.dd_bytes >= 256 || writer_provenance_full_enabled()))
        record_guest_write(GuestWriterKind::DmaData, c.dd_dst, c.dd_bytes,
                           0, 0, c.stream_order, packet_addr);
    wake_on_label(c.dd_dst);
    return true;
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
    return honor_dma_data(c, copy.packet_addr, authoritative_source);
}

// Honor a WRITE_DATA packet: copy the inline dwords to the destination address (same synchronous timing).
static void honor_write_data(const Pm4Command& c) {
    if (eop_writes_disabled()) return;
    if (!c.wd_valid) {
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 24)
            fprintf(stderr,
                    "[agc] WRITE_DATA payload incomplete — write SKIPPED: addr=0x%llx "
                    "declared=%u available=%u\n",
                    (unsigned long long)c.wd_addr, c.wd_declared_num, c.wd_num);
        return;
    }
    if (!c.wd_addr || (c.wd_addr & 3) || !c.wd_data || !c.wd_num) return;
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
        // #1226. Inside this branch, reusing `pre`, deliberately: as a separate statement below, the
        // `peek_qword(c.wd_addr)` argument would be evaluated BEFORE the gated function is entered,
        // so every WRITE_DATA packet on an UNARMED run would pay an extra 8-byte guest read. A
        // default-off diagnostic must cost nothing when off. (Reported in review of #2077.) Only
        // `wd_num == 1` can pass the sub-qword shape test anyway, and it is inside this branch.
        liveptr_trip("WDATA", c.wd_addr, pre, c.wd_data[0], (uint64_t)c.wd_num * 4, pkt_addr(c));
    }
    if (write_trap_armed())
        write_trap_scan("WDATA", c.wd_addr, peek_qword(c.wd_addr), c.wd_data,
                        (uint64_t)c.wd_num * 4, pkt_addr(c));
    memcpy((void*)(uintptr_t)c.wd_addr, c.wd_data, (size_t)c.wd_num * 4);
    set_guest_gpu_write_origin("WRITE_DATA");
    notify_guest_gpu_write(c.wd_addr, static_cast<uint64_t>(c.wd_num) * 4);
    set_guest_gpu_write_origin(nullptr);
    ring_record(c.wd_addr, c.wd_data[0], (uint8_t)(c.wd_num * 4 > 255 ? 255 : c.wd_num * 4), 3, pkt_addr(c));
    poolshift_check("WDATA", c.wd_addr, (uint64_t)c.wd_num * 4, c.wd_num ? c.wd_data[0] : 0, pkt_addr(c));
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc]   WriteData [0x%llx] %u dwords (first=0x%08x)\n",
                (unsigned long long)c.wd_addr, c.wd_num, c.wd_data[0]);
    if (writer_provenance_enabled() &&
        (c.wd_num >= 64 || writer_provenance_full_enabled()))
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

// #1226 (arc7) A/B lever, default OFF and log-only in the sense that it changes nothing unless
// set: `PROSPER_POST_SUBMIT_VISIBILITY=1` forces this model on regardless of the SDK version the
// guest asked for, `=0` forces it off. It exists because the per-fold census (see
// `ARCRUNNER_STATUS.md` § arc7) localised ArcRunner's corruption to the guest's builder thread
// being released MID-FOLD by completion writes prosper applies while it is still executing the rest
// of the same command buffer — and ArcRunner requests SDK version 10, so the post-submit contract
// that exists precisely to prevent that is not armed for it. Whether the contract is correct for a
// pre-13 title is a separate question this lever does not answer; it makes the experiment runnable.
bool post_submit_visibility_enabled() {
    static const int forced = [] {
        const int v = prosper::diag::env_tristate_or_default(
            "PROSPER_POST_SUBMIT_VISIBILITY", getenv("PROSPER_POST_SUBMIT_VISIBILITY"), -1);
        if (v == 1)
            fprintf(stderr, "[agc] POST-SUBMIT-VISIBILITY FORCED ON (#1226 A/B) — completion writes "
                            "stay private until the submit scope closes, regardless of SDK version\n");
        else if (v == 0)
            fprintf(stderr, "[agc] POST-SUBMIT-VISIBILITY FORCED OFF (#1226 A/B)\n");
        return v;
    }();
    if (forced >= 0) return forced != 0;
    return g_post_submit_visibility.load(std::memory_order_acquire);
}

bool eop_write_sync() {
    // #1226: announce the arm. This is an A/B lever whose whole purpose is to be compared against the
    // default, and a result from it was already recorded as "non-discriminating, not negative" partly
    // because nothing in the log distinguished an armed run from an unarmed one. A lever nobody can
    // witness cannot carry a null. Printed once, only when on, so the default path is untouched.
    static const bool v = [] {
        const char* e = getenv("PROSPER_EOP_WRITE_SYNC");
        const bool on = e && strtol(e, nullptr, 0) != 0;
        if (on)
            fprintf(stderr, "[agc] EOP-WRITE-SYNC ARMED: completion writes land inside the submit "
                            "call, not through the post-submit worker\n");
        return on;
    }();
    return v;
}
struct PendWrite {
    Pm4Command cmd;
    std::vector<uint32_t> wd_copy;     // owns a WriteData payload (cmd.wd_data repointed here)
    std::chrono::steady_clock::time_point queued{};   // #1945: enqueue instant (see pend_age_note)
};
// #1945: how long a completion write actually sat in this queue before it landed in guest memory.
// The model promises "post-submit plus ~1 ms of modeled pipe drain"; the WAF family this queue was
// built to prevent only happens when a write lands after the guest has recycled its 0x20-byte
// label, so the queue's real residency IS the exposure window and nothing measured it. Reports the
// running max and a loud line for any write older than PROSPER_PEND_AGE_WARN_MS (default 20).
// Default OFF, log-only, never gates a write.
//
// Called from EVERY site that applies a queued write — the FIFO drain, the renderer's selective
// extraction, and the gated-span release. A residency figure taken from only one of the three would
// bound only that path, and the conclusion drawn from this instrument ("prosper's completion writes
// do NOT land late on PPSA07809") is exactly the kind that a partially-instrumented population
// would falsely support.
static void pend_age_note(std::chrono::steady_clock::time_point queued) {
    static const int on = [] { const char* e = getenv("PROSPER_PEND_AGE");
                               return e && strtol(e, nullptr, 0) != 0 ? 1 : 0; }();
    if (!on || queued.time_since_epoch().count() == 0) return;
    static const long warn_ms = [] { const char* e = getenv("PROSPER_PEND_AGE_WARN_MS");
                                     return e ? strtol(e, nullptr, 0) : 20L; }();
    const long age = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - queued).count();
    static std::atomic<uint64_t> n{0};
    static std::atomic<long> peak{0};
    const uint64_t ord = n.fetch_add(1, std::memory_order_relaxed) + 1;
    long prev = peak.load(std::memory_order_relaxed);
    while (age > prev && !peak.compare_exchange_weak(prev, age)) {}
    const bool loud = age >= warn_ms;
    if (loud || (ord % 512) == 0)
        fprintf(stderr, "[agc] PEND-AGE #%llu age=%ldms peak=%ldms%s\n",
                (unsigned long long)ord, age, peak.load(std::memory_order_relaxed),
                loud ? "  <<<<< completion write landed long after its submit" : "");
}
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
        pend_age_note(w.queued);
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
    PendWrite w; w.cmd = c; w.queued = std::chrono::steady_clock::now();
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
    static const bool batch_enabled =
        std::getenv("PROSPER_NO_BATCH_RENDERER_WRITE_DRAIN") == nullptr;
    if (!batch_enabled) {
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
                if (w.cmd.kind != K::WriteData || !w.cmd.wd_data || !w.cmd.wd_num)
                    return false;
                const uint64_t bytes = (uint64_t)w.cmd.wd_num * 4;
                return !overlaps_completion(w.cmd.wd_addr, bytes);
            });
            if (it == p.q.end()) return;
            PendWrite w = std::move(*it);
            p.q.erase(it);
            p.inflight++;
            lk.unlock();
            pend_age_note(w.queued);
            apply_deferred_effect(w.cmd);
            lk.lock();
            p.inflight--;
            p.cv.notify_all();
        }
    }

    struct CompletionSpan {
        uint64_t begin = 0;
        uint64_t end = 0;
    };
    for (;;) {
        std::unique_lock<std::mutex> lk(p.mx);
        if (p.inflight > 0) {
            p.cv.wait(lk);
            continue;
        }
        std::vector<CompletionSpan> completion_spans;
        completion_spans.reserve(p.q.size());
        for (const PendWrite& queued : p.q) {
            uint64_t target = 0, target_bytes = 0;
            if (queued.cmd.kind == Pm4Command::Kind::ReleaseMem) {
                target = queued.cmd.rel_addr;
                target_bytes = queued.cmd.rel_data_sel == 1 ? 4 : 8;
            } else if (queued.cmd.kind == Pm4Command::Kind::EventWrite) {
                target = queued.cmd.event_addr;
                target_bytes = 8;
            }
            if (!target || !target_bytes) continue;
            completion_spans.push_back({
                target,
                target > UINT64_MAX - target_bytes ? UINT64_MAX : target + target_bytes});
        }
        std::sort(completion_spans.begin(), completion_spans.end(),
                  [](const CompletionSpan& a, const CompletionSpan& b) {
                      return a.begin < b.begin || (a.begin == b.begin && a.end < b.end);
                  });
        size_t merged_count = 0;
        for (const CompletionSpan& span : completion_spans) {
            if (merged_count && span.begin <= completion_spans[merged_count - 1].end) {
                completion_spans[merged_count - 1].end =
                    std::max(completion_spans[merged_count - 1].end, span.end);
            } else {
                completion_spans[merged_count++] = span;
            }
        }
        completion_spans.resize(merged_count);
        auto overlaps_completion = [&](uint64_t addr, uint64_t bytes) {
            if (!addr || !bytes || addr > UINT64_MAX - bytes) return true;
            const uint64_t end = addr + bytes;
            const auto found = std::lower_bound(
                completion_spans.begin(), completion_spans.end(), addr,
                [](const CompletionSpan& span, uint64_t value) {
                    return span.end <= value;
                });
            return found != completion_spans.end() && found->begin < end;
        };
        auto renderer_write = [&](const PendWrite& w) {
            using K = Pm4Command::Kind;
            if (w.cmd.kind == K::DmaData)
                return !overlaps_completion(w.cmd.dd_dst, w.cmd.dd_bytes);
            if (w.cmd.kind != K::WriteData || !w.cmd.wd_data || !w.cmd.wd_num) return false;
            const uint64_t bytes = (uint64_t)w.cmd.wd_num * 4;
            return !overlaps_completion(w.cmd.wd_addr, bytes);
        };
        const size_t ready_count = static_cast<size_t>(std::count_if(
            p.q.begin(), p.q.end(), renderer_write));
        if (!ready_count) return;

        // Extract every currently eligible resource write in one stable partition. The previous
        // loop searched the complete queue for every candidate and erased one deque element at a
        // time. A submit with thousands of private completion labels therefore became quadratic
        // before Vulkan saw any work. The lock makes this snapshot atomic with enqueue; writes
        // appended after the partition are later in queue order and cannot be overtaken.
        std::vector<PendWrite> ready;
        ready.reserve(ready_count);
        std::deque<PendWrite> blocked;
        while (!p.q.empty()) {
            PendWrite write = std::move(p.q.front());
            p.q.pop_front();
            if (renderer_write(write))
                ready.push_back(std::move(write));
            else
                blocked.push_back(std::move(write));
        }
        p.q.swap(blocked);
        p.inflight++;
        lk.unlock();
        for (const PendWrite& write : ready) {
            pend_age_note(write.queued);
            apply_deferred_effect(write.cmd);
        }
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
        } else if (c.kind == K::WriteData && c.wd_valid && c.wd_addr == addr &&
                   c.wd_data && c.wd_num) {
            const size_t n = std::min<size_t>((size_t)c.wd_num * 4, sizeof v);
            memcpy(&v, c.wd_data, n);
            touched = true;
        } else if (c.kind == K::DmaData && c.dd_dst == addr && c.dd_valid &&
                   dma_data_immediate_source(c)) {
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
bool wait_regmem_value_satisfied(const Pm4Command& c, uint64_t memory_value) {
    const uint64_t v = memory_value & c.wm_mask, r = c.wm_ref;
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
    return wait_regmem_value_satisfied(c, mem);
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
// GDS has its own 64 KiB address space.  Keeping these spans separate is not merely an authority
// concern: an immediate GDS fill deferred in one fold must order a later memory-to-GDS copy in a
// different fold, while the numerically equal guest address remains unrelated.
std::array<std::vector<std::pair<uint64_t, uint64_t>>, kDeferredQueueCount>
    g_gated_gds_ranges; // [offset, offset + bytes)
// The guest-memory span a command writes (0 bytes = writes nothing we track).
void effect_span(const Pm4Command& c, uint64_t* addr, uint64_t* bytes) {
    using K = Pm4Command::Kind;
    switch (c.kind) {
        case K::ReleaseMem: *addr = c.rel_addr;   *bytes = 8; break;
        case K::EventWrite: *addr = c.event_addr; *bytes = 8; break;
        case K::WriteData:  *addr = c.wd_addr;
                            *bytes = (uint64_t)c.wd_declared_num * 4; break;
        case K::DmaData:    *addr = c.dd_dst;     *bytes = c.dd_bytes; break;
        default:            *addr = 0;            *bytes = 0; break;
    }
}
void gated_register(const Pm4Command& c) {
    if (c.kind == Pm4Command::Kind::DmaData &&
        dma_data_dst_sel(c) == kDmaSelGds) {
        const uint64_t gds_size = compute_gds_size();
        if (!c.dd_valid || !c.dd_bytes || c.dd_dst >= gds_size ||
            c.dd_bytes > gds_size - c.dd_dst || (c.dd_dst & 3u) ||
            (c.dd_bytes & 3u))
            return;
        g_gated_gds_ranges[deferred_queue(c.queue_origin)].emplace_back(
            c.dd_dst, c.dd_dst + c.dd_bytes);
        return;
    }
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
bool gds_gated(uint64_t offset, uint64_t bytes, uint8_t origin) {
    const uint64_t gds_size = compute_gds_size();
    if (!bytes || offset >= gds_size || bytes > gds_size - offset) return false;
    for (const auto& r : g_gated_gds_ranges[deferred_queue(origin)])
        if (offset < r.second && r.first < offset + bytes) return true;
    return false;
}
bool effect_gated(const Pm4Command& c) {
    if (c.kind == Pm4Command::Kind::DmaData &&
        dma_data_dst_sel(c) == kDmaSelGds)
        return gds_gated(c.dd_dst, c.dd_bytes, c.queue_origin);
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
        g_gated_gds_ranges[queue].clear();
        // Deliver every pulse owed by submissions from this queue. A blocked peer queue retains its
        // own pulses; an Acb completion must not wait for an unrelated Dcb barrier (and vice versa).
        uint64_t owed = g_owed_pulses[queue];
        g_owed_pulses[queue] = 0;
        if (!owed && signalable_completed[queue] > 0) owed = 1;
        while (owed--) prosper_eq_trigger_eop();
    }
    return completed;
}

// An eager guest-memory reader after a retained address DMA is unsafe only when it can observe that
// DMA's destination. Direct-memory aliases make VA-only interval comparison insufficient, so ask the
// authoritative kernel mapping table about every VA-disjoint pair. Unknown/malformed/untracked
// topology remains fail-closed; only a proven physical disjoint result relaxes the whole-submit
// rejection. This is deliberately correctness-first despite the bounded mapping lookups.
// Tactics Ogre's movie upload is the decisive disjoint case: thousands of row copies target its
// Y/UV staging textures while later waits and indirect-register packets read separate allocations.
static bool retained_dma_destination_overlaps(
        const std::vector<GpuState::DmaCopy>& copies, uint64_t address, uint64_t bytes) {
    for (const GpuState::DmaCopy& copy : copies) {
        // GDS destinations are offsets into the private 64 KiB share, not guest VAs. Treating the
        // live Astro Bot offset 0x24 as an unknown guest address made it overlap every wait.
        if ((copy.sels & 0xffu) == kDmaSelGds) continue;
        if (guest_memory_topology_relation(address, bytes, copy.dst, copy.bytes) !=
            GuestMemoryTopologyRelation::Disjoint)
            return true;
    }
    return false;
}

// Memory effects retained after the first address DMA are earlier producers too. An eager reader
// must not be folded from stale guest bytes merely because the original DMA destination is disjoint:
// `DMA(A) -> WRITE_DATA(B) -> SET_REGS_INDIRECT(B)` and the immediate-DMA/wait sibling both depend
// on B. Inspect every earlier retained destination and require the same authoritative physical-
// topology proof as above. GDS destinations are offsets into the private share, not guest ranges.
static bool retained_ordered_effect_destination_overlaps(
        const std::vector<GpuState::MemoryEffect>& effects, uint64_t address, uint64_t bytes) {
    for (const GpuState::MemoryEffect& effect : effects) {
        const Pm4Command& command = effect.cmd;
        if (command.kind == Pm4Command::Kind::DmaData &&
            dma_data_dst_sel(command) == kDmaSelGds)
            continue;
        uint64_t destination = 0, effect_bytes = 0;
        effect_span(command, &destination, &effect_bytes);
        if (!destination || !effect_bytes) continue;
        if (guest_memory_topology_relation(address, bytes, destination, effect_bytes) !=
            GuestMemoryTopologyRelation::Disjoint)
            return true;
    }
    return false;
}

enum class OrderedQwordOverlay { Untouched, Touched, Ambiguous };

const char* ordered_wait_effect_class_name(OrderedWaitEffectClass effect_class) {
    switch (effect_class) {
        case OrderedWaitEffectClass::Disjoint: return "disjoint";
        case OrderedWaitEffectClass::ImmediateDma: return "immediate-dma";
        case OrderedWaitEffectClass::AddressDma: return "address-dma";
        case OrderedWaitEffectClass::FixedRelease32: return "fixed-release32";
        case OrderedWaitEffectClass::FixedRelease64: return "fixed-release64";
        case OrderedWaitEffectClass::InterruptOnlyRelease: return "interrupt-only-release";
        case OrderedWaitEffectClass::DynamicRelease: return "dynamic-release";
        case OrderedWaitEffectClass::WriteData: return "write-data";
        case OrderedWaitEffectClass::WriteDataPartial: return "write-data-partial";
        case OrderedWaitEffectClass::WriteDataOversized: return "write-data-oversized";
        case OrderedWaitEffectClass::WriteDataUnowned: return "write-data-unowned";
        case OrderedWaitEffectClass::EventTimestamp: return "event-timestamp";
        case OrderedWaitEffectClass::OffsetOverlap: return "offset-overlap";
        case OrderedWaitEffectClass::AliasedOverlap: return "aliased-overlap";
        case OrderedWaitEffectClass::Unsupported: return "unsupported";
    }
    return "unsupported";
}

const char* ordered_wait_effect_overlay_name(OrderedWaitEffectOverlay overlay) {
    switch (overlay) {
        case OrderedWaitEffectOverlay::None: return "none";
        case OrderedWaitEffectOverlay::Applied: return "applied";
        case OrderedWaitEffectOverlay::Ambiguous: return "ambiguous";
    }
    return "ambiguous";
}

OrderedWaitEffectDiagnostic diagnose_ordered_wait_effect(
        const GpuState::MemoryEffect& effect, uint64_t wait_address, uint64_t value_before) {
    OrderedWaitEffectDiagnostic diagnostic;
    diagnostic.command_order = effect.cmd.stream_order;
    diagnostic.value_before = value_before;
    diagnostic.value_after = value_before;

    const Pm4Command& command = effect.cmd;
    effect_span(command, &diagnostic.address, &diagnostic.bytes);
    if (!diagnostic.address || !diagnostic.bytes ||
        guest_memory_topology_relation(
            wait_address, sizeof(value_before), diagnostic.address, diagnostic.bytes) ==
            GuestMemoryTopologyRelation::Disjoint) {
        diagnostic.effect_class = OrderedWaitEffectClass::Disjoint;
        diagnostic.overlay = OrderedWaitEffectOverlay::None;
        return diagnostic;
    }

    if (command.kind == Pm4Command::Kind::DmaData) {
        if (command.dd_dst != wait_address) {
            diagnostic.effect_class = OrderedWaitEffectClass::OffsetOverlap;
            return diagnostic;
        }
        if (dma_data_form(command) != DmaDataForm::Immediate) {
            diagnostic.effect_class = dma_data_address_source(command)
                ? OrderedWaitEffectClass::AddressDma
                : OrderedWaitEffectClass::Unsupported;
            return diagnostic;
        }
        const uint32_t word = static_cast<uint32_t>(command.dd_src);
        const size_t bytes = std::min<size_t>(command.dd_bytes, sizeof(value_before));
        for (size_t offset = 0; offset < bytes; offset += sizeof(word))
            memcpy(reinterpret_cast<uint8_t*>(&diagnostic.value_after) + offset, &word,
                   std::min(sizeof(word), bytes - offset));
        diagnostic.effect_class = OrderedWaitEffectClass::ImmediateDma;
        diagnostic.overlay = OrderedWaitEffectOverlay::Applied;
        return diagnostic;
    }

    if (command.kind == Pm4Command::Kind::ReleaseMem) {
        if (command.rel_addr != wait_address) {
            diagnostic.effect_class = OrderedWaitEffectClass::OffsetOverlap;
            return diagnostic;
        }
        if (command.rel_data_sel == 0) {
            diagnostic.effect_class = OrderedWaitEffectClass::InterruptOnlyRelease;
            diagnostic.overlay = OrderedWaitEffectOverlay::None;
            return diagnostic;
        }
        if (!command.rel_value_valid) {
            diagnostic.effect_class = OrderedWaitEffectClass::DynamicRelease;
            return diagnostic;
        }
        if (command.rel_data_sel == 1) {
            const uint32_t low = static_cast<uint32_t>(command.rel_value);
            memcpy(&diagnostic.value_after, &low, sizeof(low));
            diagnostic.effect_class = OrderedWaitEffectClass::FixedRelease32;
            diagnostic.overlay = OrderedWaitEffectOverlay::Applied;
            return diagnostic;
        }
        if (command.rel_data_sel == 2) {
            diagnostic.value_after = command.rel_value;
            diagnostic.effect_class = OrderedWaitEffectClass::FixedRelease64;
            diagnostic.overlay = OrderedWaitEffectOverlay::Applied;
            return diagnostic;
        }
        diagnostic.effect_class = OrderedWaitEffectClass::DynamicRelease;
        return diagnostic;
    }

    if (command.kind == Pm4Command::Kind::WriteData) {
        if (command.wd_addr != wait_address) {
            const bool virtual_overlap =
                wait_address <= UINT64_MAX - sizeof(value_before) &&
                command.wd_addr <= UINT64_MAX - diagnostic.bytes &&
                wait_address < command.wd_addr + diagnostic.bytes &&
                command.wd_addr < wait_address + sizeof(value_before);
            diagnostic.effect_class = virtual_overlap
                ? OrderedWaitEffectClass::OffsetOverlap
                : OrderedWaitEffectClass::AliasedOverlap;
            return diagnostic;
        }
        // MemoryEffect deep-copies WRITE_DATA's inline dwords and rebinds cmd.wd_data after every
        // copy/move. Prove that exact ownership before reading: a pointer into the guest command
        // buffer may be recycled as soon as folding returns. A short vector is an incomplete
        // snapshot, and more than the waited qword is intentionally outside this scalar model.
        if (!command.wd_valid || effect.write_data.size() != command.wd_num) {
            diagnostic.effect_class = OrderedWaitEffectClass::WriteDataPartial;
            return diagnostic;
        }
        if (!command.wd_data || effect.write_data.empty() ||
            command.wd_data != effect.write_data.data()) {
            diagnostic.effect_class = OrderedWaitEffectClass::WriteDataUnowned;
            return diagnostic;
        }
        if (command.wd_num > sizeof(value_before) / sizeof(uint32_t)) {
            diagnostic.effect_class = OrderedWaitEffectClass::WriteDataOversized;
            return diagnostic;
        }
        diagnostic.value_after =
            (value_before & 0xffffffff00000000ull) | effect.write_data[0];
        if (command.wd_num == 2)
            diagnostic.value_after =
                static_cast<uint64_t>(effect.write_data[0]) |
                (static_cast<uint64_t>(effect.write_data[1]) << 32);
        diagnostic.effect_class = OrderedWaitEffectClass::WriteData;
        diagnostic.overlay = OrderedWaitEffectOverlay::Applied;
        return diagnostic;
    }
    if (command.kind == Pm4Command::Kind::EventWrite) {
        diagnostic.effect_class = OrderedWaitEffectClass::EventTimestamp;
        return diagnostic;
    }
    diagnostic.effect_class = OrderedWaitEffectClass::Unsupported;
    return diagnostic;
}

struct OrderedQwordOverlayResult {
    OrderedQwordOverlay overlay = OrderedQwordOverlay::Untouched;
    OrderedWaitEffectClass reason = OrderedWaitEffectClass::Disjoint;
    uint64_t value = 0;
};

// WAIT_REG_MEM is an eager scalar reader, but the overwhelmingly common suffix dependency is an
// immediate DMA or owned, qword-bounded WRITE_DATA label initialization followed by a fixed-value
// release. Evaluate those exact producers from the ordered journal rather than rejecting a valid
// `DMA(A), write(B), release(B), wait(B)` submit or reading stale B. Keep every other overlapping
// effect fail-closed: timestamps, offset/aliased/incompletely-owned writes, oversized writes, and
// address copies need execution-time semantics, not a guess.
static OrderedQwordOverlayResult overlay_ordered_qword(
        const std::vector<GpuState::MemoryEffect>& effects, uint64_t address, uint64_t* value) {
    bool touched = false;
    uint64_t result = *value;
    OrderedWaitEffectClass last_applied = OrderedWaitEffectClass::Disjoint;
    for (const GpuState::MemoryEffect& effect : effects) {
        if (effect.cmd.kind == Pm4Command::Kind::DmaData &&
            dma_data_dst_sel(effect.cmd) == kDmaSelGds)
            continue;
        const OrderedWaitEffectDiagnostic diagnostic =
            diagnose_ordered_wait_effect(effect, address, result);
        if (diagnostic.overlay == OrderedWaitEffectOverlay::None) {
            if (diagnostic.effect_class != OrderedWaitEffectClass::Disjoint)
                last_applied = diagnostic.effect_class;
            continue;
        }
        if (diagnostic.overlay == OrderedWaitEffectOverlay::Ambiguous)
            return {OrderedQwordOverlay::Ambiguous, diagnostic.effect_class, result};
        result = diagnostic.value_after;
        last_applied = diagnostic.effect_class;
        touched = true;
    }
    if (touched) *value = result;
    return {touched ? OrderedQwordOverlay::Touched : OrderedQwordOverlay::Untouched,
            last_applied, result};
}

static const char* ordered_qword_overlay_name(OrderedQwordOverlay overlay) {
    switch (overlay) {
        case OrderedQwordOverlay::Untouched: return "untouched";
        case OrderedQwordOverlay::Touched: return "touched";
        case OrderedQwordOverlay::Ambiguous: return "ambiguous";
    }
    return "ambiguous";
}

// This instrument deliberately lives behind the existing high-volume GFXLOG switch. A rejection
// alone says only that the fail-closed guard ran; it does not say which retained operation made the
// eager wait unknowable. Report the first eight occurrences and a power-of-two tail, with a stable
// 1-based ordinal on every line. Each reported occurrence is additionally bounded to sixteen
// overlapping operations, while the summary retains the complete overlap counts.
static void diagnose_ordered_wait_rejection(
        const Pm4Command& wait, const std::vector<GpuState::DmaCopy>& copies,
        const std::vector<GpuState::MemoryEffect>& effects, bool readable,
        uint64_t base_value, const OrderedQwordOverlayResult& overlay,
        bool comparison_satisfied) {
    if (!std::getenv("PROSPER_GFXLOG")) return;
    static std::atomic<uint64_t> ordinal_counter{0};
    const uint64_t ordinal = ordinal_counter.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!prosper::diag_should_print(ordinal, 8)) return;

    size_t overlapping_copies = 0, overlapping_effects = 0;
    for (const GpuState::DmaCopy& copy : copies)
        if (guest_memory_topology_relation(
                wait.wm_addr, sizeof(uint64_t), copy.dst, copy.bytes) !=
            GuestMemoryTopologyRelation::Disjoint)
            ++overlapping_copies;
    for (const GpuState::MemoryEffect& effect : effects) {
        const OrderedWaitEffectDiagnostic diagnostic =
            diagnose_ordered_wait_effect(effect, wait.wm_addr, base_value);
        if (diagnostic.effect_class != OrderedWaitEffectClass::Disjoint)
            ++overlapping_effects;
    }

    std::fprintf(stderr,
                 "[agc] ordered-wait rejection #%llu addr=0x%llx readable=%u "
                 "base=0x%016llx overlaid=0x%016llx overlay=%s reason=%s "
                 "compare=%s mask=0x%llx ref=0x%llx func=%u "
                 "dma-overlaps=%zu effect-overlaps=%zu\n",
                 static_cast<unsigned long long>(ordinal),
                 static_cast<unsigned long long>(wait.wm_addr), readable ? 1u : 0u,
                 static_cast<unsigned long long>(base_value),
                 static_cast<unsigned long long>(overlay.value),
                 ordered_qword_overlay_name(overlay.overlay),
                 ordered_wait_effect_class_name(overlay.reason),
                 overlay.overlay == OrderedQwordOverlay::Touched
                     ? (comparison_satisfied ? "satisfied" : "unsatisfied")
                     : "not-evaluated",
                 static_cast<unsigned long long>(wait.wm_mask),
                 static_cast<unsigned long long>(wait.wm_ref), wait.wm_func,
                 overlapping_copies, overlapping_effects);

    constexpr size_t kMaxDetailEffects = 16;
    size_t detail = 0;
    for (const GpuState::DmaCopy& copy : copies) {
        if (guest_memory_topology_relation(
                wait.wm_addr, sizeof(uint64_t), copy.dst, copy.bytes) ==
            GuestMemoryTopologyRelation::Disjoint)
            continue;
        if (detail++ >= kMaxDetailEffects) continue;
        std::fprintf(stderr,
                     "[agc] ordered-wait rejection #%llu effect[%zu] kind=address-dma "
                     "overlay=ambiguous span=0x%llx+%u order=%llu\n",
                     static_cast<unsigned long long>(ordinal), detail,
                     static_cast<unsigned long long>(copy.dst), copy.bytes,
                     static_cast<unsigned long long>(copy.command_order));
    }
    uint64_t diagnostic_value = base_value;
    for (const GpuState::MemoryEffect& effect : effects) {
        const OrderedWaitEffectDiagnostic diagnostic =
            diagnose_ordered_wait_effect(effect, wait.wm_addr, diagnostic_value);
        if (diagnostic.effect_class == OrderedWaitEffectClass::Disjoint) continue;
        if (detail++ < kMaxDetailEffects) {
            std::fprintf(stderr,
                         "[agc] ordered-wait rejection #%llu effect[%zu] kind=%s overlay=%s "
                         "span=0x%llx+%llu order=%llu before=0x%016llx after=0x%016llx\n",
                         static_cast<unsigned long long>(ordinal), detail,
                         ordered_wait_effect_class_name(diagnostic.effect_class),
                         ordered_wait_effect_overlay_name(diagnostic.overlay),
                         static_cast<unsigned long long>(diagnostic.address),
                         static_cast<unsigned long long>(diagnostic.bytes),
                         static_cast<unsigned long long>(diagnostic.command_order),
                         static_cast<unsigned long long>(diagnostic.value_before),
                         static_cast<unsigned long long>(diagnostic.value_after));
        }
        if (diagnostic.overlay == OrderedWaitEffectOverlay::Applied)
            diagnostic_value = diagnostic.value_after;
    }
    if (overlapping_copies + overlapping_effects > kMaxDetailEffects)
        std::fprintf(stderr,
                     "[agc] ordered-wait rejection #%llu effects-truncated retained=%zu shown=%zu\n",
                     static_cast<unsigned long long>(ordinal),
                     overlapping_copies + overlapping_effects, kMaxDetailEffects);
}

// A route progressing after a semantic change is not proof that the new lever ran. When GFXLOG is
// armed, positively identify satisfied eager waits whose ordered scalar evaluation actually
// consumed at least one owned WRITE_DATA overlay. This is bounded independently of rejections.
static void diagnose_ordered_wait_acceptance(
        const Pm4Command& wait, const std::vector<GpuState::MemoryEffect>& effects,
        uint64_t base_value, const OrderedQwordOverlayResult& overlay) {
    if (!std::getenv("PROSPER_GFXLOG")) return;
    size_t write_data_overlays = 0;
    uint64_t diagnostic_value = base_value;
    for (const GpuState::MemoryEffect& effect : effects) {
        const OrderedWaitEffectDiagnostic diagnostic =
            diagnose_ordered_wait_effect(effect, wait.wm_addr, diagnostic_value);
        if (diagnostic.overlay != OrderedWaitEffectOverlay::Applied) continue;
        if (diagnostic.effect_class == OrderedWaitEffectClass::WriteData)
            ++write_data_overlays;
        diagnostic_value = diagnostic.value_after;
    }
    if (!write_data_overlays) return;

    static std::atomic<uint64_t> ordinal_counter{0};
    const uint64_t ordinal = ordinal_counter.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!prosper::diag_should_print(ordinal, 8)) return;
    std::fprintf(stderr,
                 "[agc] ordered-wait acceptance #%llu addr=0x%llx order=%llu "
                 "base=0x%016llx overlaid=0x%016llx final=%s "
                 "write-data-overlays=%zu effect-count=%zu\n",
                 static_cast<unsigned long long>(ordinal),
                 static_cast<unsigned long long>(wait.wm_addr),
                 static_cast<unsigned long long>(wait.stream_order),
                 static_cast<unsigned long long>(base_value),
                 static_cast<unsigned long long>(overlay.value),
                 ordered_wait_effect_class_name(overlay.reason),
                 write_data_overlays, effects.size());
}


// Apply all retained address-backed DMA copies to guest memory, in stream order, and clear the
// retention. Called at a DMA-overlap dependency point (WAIT_REG_MEM / SetRegsIndirect / Jump
// that reads a retained DMA's destination) so the dependent command sees the DMA-written data —
// in-order execution, matching real hardware where there is no "rejection", only ordering.
void apply_retained_dma_copies(std::vector<GpuState::DmaCopy>& copies) {
    for (const auto& copy : copies) {
        if (!copy.dst || !copy.src || !copy.bytes) continue;
        if (!guest_readable(copy.src, copy.bytes) || !guest_readable(copy.dst, copy.bytes)) {
            static std::atomic<int> n{0};
            if (n.fetch_add(1) < 8)
                fprintf(stderr, "[agc] apply_retained_dma_copies: skipping unmapped DMA "
                        "dst=0x%llx src=0x%llx bytes=%llu\n",
                        (unsigned long long)copy.dst, (unsigned long long)copy.src,
                        (unsigned long long)copy.bytes);
            continue;
        }
        memmove(reinterpret_cast<void*>(uintptr_t(copy.dst)),
                reinterpret_cast<const void*>(uintptr_t(copy.src)),
                copy.bytes);
    }
    copies.clear();
}

// State-at-draw snapshot (see the declaration). One snapshot is shared by consecutive work items
// with no register write between them, so this is a pointer copy in the common case.
const std::shared_ptr<const GpuState>& GpuState::refresh_state_snapshot() {
    if (state_dirty_ || !last_snapshot_) {
        auto snap = std::make_shared<GpuState>();
        snap->cx = cx; snap->sh = sh; snap->uc = uc; snap->index_type = index_type;
        snap->index_type_announced = index_type_announced;
        snap->num_instances = num_instances;
        snap->command_order = command_order;   // #305 instrument: order at snapshot
        if (udprov_collection_enabled()) { snap->sh_prov = sh_prov; snap->sh_prov_src = sh_prov_src; }
        last_snapshot_ = std::move(snap);
        state_dirty_ = false;
    }
    return last_snapshot_;
}

void GpuState::apply(const Pm4Command& c) {
    using K = Pm4Command::Kind;
    command_order = c.stream_order ? c.stream_order : command_order + 1;
    switch (c.kind) {
        case K::SetRegsIndirect: {
            if (c.regs_vaddr == 0 || c.num_regs == 0 || c.num_regs > kMaxRegsPerPacket) return;
            const uint64_t regs_bytes =
                static_cast<uint64_t>(c.num_regs) * sizeof(ShaderReg);
            const bool overlaps_dma = !dma_copies.empty() &&
                retained_dma_destination_overlaps(dma_copies, c.regs_vaddr, regs_bytes);
            const bool overlaps_effect = !ordered_memory_effects.empty() &&
                retained_ordered_effect_destination_overlaps(
                    ordered_memory_effects, c.regs_vaddr, regs_bytes);
            if (overlaps_dma) {
                // In-order execution (#2975): apply the retained DMAs so the reader sees the
                // copied bytes, then proceed. Real hardware has no "rejection" — only ordering.
                apply_retained_dma_copies(dma_copies);
            }
            if (overlaps_effect) {
                // Effect overlaps are a different mechanism (retained WRITE_DATA, not DMA) and
                // keep their fail-closed rejection — the correct fix for those is separate.
                dma_execution_rejected = true;
                break;
            }
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
            // PROSPER_BINDTRACE=1 (#305 instrument): the pipeline's shader-program registers arrive
            // through this path as a POINTER to a guest array that prosper reads at FOLD time, while
            // the stage's user data arrives inline through SET_SH_REG. If a title recycles the array
            // between recording the packet and submitting the buffer, several binds fold to the same
            // final contents and a draw runs with another pipeline's program while holding its own
            // user data. Report each Sh bind's array identity plus the program/RSRC2 values it
            // carries, so repeated `vaddr` with differing draws is directly observable.
            static const bool bindtrace = getenv("PROSPER_BINDTRACE") != nullptr;
            if (bindtrace && c.reg_class == RegClass::Sh) {
                uint32_t es_lo = 0, rsrc2 = 0, pgm_ps = 0;
                bool has = false;
                for (uint32_t i = 0; i < c.num_regs; i++) {
                    if (regs[i].offset == prosper::agc::Pm4::SPI_SHADER_PGM_LO_ES) { es_lo = regs[i].value; has = true; }
                    else if (regs[i].offset == prosper::agc::Pm4::SPI_SHADER_PGM_RSRC2_GS) rsrc2 = regs[i].value;
                    else if (regs[i].offset == prosper::agc::Pm4::SPI_SHADER_PGM_LO_PS) pgm_ps = regs[i].value;
                }
                if (has) {
                    static std::atomic<int> n{0};
                    if (n.fetch_add(1) < 2000000)
                        fprintf(stderr,
                                "[bind] order=%llu q%u f%u j%u vaddr=0x%llx num=%u es_lo=0x%x "
                                "rsrc2=0x%x ps_lo=0x%x\n",
                                (unsigned long long)command_order, (unsigned)c.queue_origin,
                                g_fold_seq.load(std::memory_order_relaxed), jump_depth,
                                (unsigned long long)c.regs_vaddr, c.num_regs, es_lo, rsrc2, pgm_ps);
                }
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
                // PROSPER_REGWATCH: indirect-path half. Reported before the bounds drop for the same
                // reason as the direct path — a watched register whose only writes are dropped here
                // is a different finding from one that is never written.
                reg_watch_report(c.reg_class, offset, regs[i].value, 1u, nullptr,
                                 "indirect", command_order);
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
                if (udprov_collection_enabled() && c.reg_class == RegClass::Sh) {
                    sh_prov[offset] = command_order | kProvIndirect;
                    sh_prov_src[offset] = pack_prov_src(
                        c.queue_origin, jump_depth,
                        g_fold_seq.load(std::memory_order_relaxed));
                }
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
            // PROSPER_REGWATCH: direct-path half of the register-write watch (see below for the
            // indirect half). Emitted before the bounds check so a dropped out-of-range write is
            // still observable — "the write happened but landed nowhere" is a distinct diagnosis
            // from "the write never happened".
            reg_watch_report(c.reg_class, c.reg_offset,
                             c.reg_data && c.reg_count ? c.reg_data[0] : c.reg_value,
                             c.reg_data ? c.reg_count : 1u, c.reg_data, "direct", command_order);
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
            if (udprov_collection_enabled() && c.reg_class == RegClass::Sh) {
                const uint64_t src = pack_prov_src(c.queue_origin, jump_depth,
                                                   g_fold_seq.load(std::memory_order_relaxed));
                for (uint32_t k = 0; k < c.reg_count && c.reg_offset + k < kRegOffsetLimit; k++) {
                    sh_prov[c.reg_offset + k] = command_order;
                    sh_prov_src[c.reg_offset + k] = src;
                }
            }
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
            // #3009: record the ANNOUNCEMENT, not just the value. A type-3 packet is at least two
            // dwords (pm4_decode.cpp hdr_len adds 2), so a decoded SetIndexType always carried its
            // payload dword and this is unconditional -- the `npl >= 1` guard on c.index_size is
            // defensive, not a reachable "announced nothing" case.
            index_type_announced = true;
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
                const uint64_t previous = current;
                const bool low_only = base <= UINT32_MAX;
                const bool inherited = low_only && current > UINT32_MAX;
                if (inherited)
                    base |= current & ~static_cast<uint64_t>(UINT32_MAX);
                current = base;
                // Remember the VA aperture. A PS5 process has ONE GPU virtual address space, but
                // `indirect_*_base` is per-fold state, so a queue whose stream carries no SetBase
                // starts at zero and its low-only DispatchIndirect offsets resolve to unmapped
                // addresses. Retaining the aperture across folds is what lets those be recovered —
                // see the recovery at the DispatchIndirect site, which is gated on readability.
                if (base > UINT32_MAX)
                    g_indirect_va_aperture.store(base >> 32, std::memory_order_relaxed);
                // Name a truncated base WITHOUT PROSPER_INDIRECTLOG, bounded. A low-only update with
                // no upper half to inherit silently disables every indirect operation that consumes
                // it: the executor later reports only "unreadable arguments at 0x<sum>", which says
                // nothing about which half was lost or which packet lost it. Same reasoning as the
                // ungated `[compute] skip … reason=` line — a diagnostic reachable only through a
                // switch that perturbs the route is not reachable.
                if (low_only && !inherited) {
                    static std::atomic<int> truncated{0};
                    if (truncated.fetch_add(1) < 24)
                        fprintf(stderr,
                                "[agc] indirect-args base TRUNCATED: %s packet=0x%llx has no upper "
                                "half and the current base (0x%llx) has none to inherit\n",
                                c.indirect_shader_type == 0 ? "graphics" : "compute",
                                (unsigned long long)c.indirect_base,
                                (unsigned long long)previous);
                }
                // PROSPER_INDIRECTLOG: what the guest actually sends, because the fail-closed branch
                // below it is silent. A low-only SetBase with no prior aperture leaves the base
                // truncated, the executor then rejects the argument buffer as unreadable, and the
                // indirect dispatch is skipped — with nothing in any log connecting the skip to the
                // packet that caused it.
                if (getenv("PROSPER_INDIRECTLOG")) {
                    static std::atomic<int> logged{0};
                    if (logged.fetch_add(1) < 64)
                        fprintf(stderr,
                                "[agc-setbase] %s packet=0x%llx low_only=%d prior=0x%llx "
                                "inherited=%d -> base=0x%llx%s\n",
                                c.indirect_shader_type == 0 ? "graphics" : "compute",
                                (unsigned long long)c.indirect_base, (int)low_only,
                                (unsigned long long)previous, (int)inherited,
                                (unsigned long long)current,
                                (low_only && !inherited) ? "  TRUNCATED-NO-APERTURE" : "");
                }
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
            // #305: bind-vs-work sequence in stream order. Cached — this runs per draw/dispatch,
            // and a per-item environ scan is measurable at this title's ~876k items per route.
            // The three graphics arms emit "DRAW"; the two compute arms below emit "DISPATCH", so
            // an analysis over these lines is never silently counting dispatches as draws (a
            // compute item does not consume SPI_SHADER_PGM_LO_ES at all, so mixing them corrupts
            // any bind/draw agreement statistic).
            static const bool bindtrace_draw = getenv("PROSPER_BINDTRACE") != nullptr;
            if (bindtrace_draw) {
                auto rd = [&](uint32_t off) { auto it = sh.find(off); return it == sh.end() ? 0u : it->second; };
                static std::atomic<int> nd{0};
                if (nd.fetch_add(1) < 2000000)
                    fprintf(stderr,
                            "[bind] DRAW order=%llu q%u f%u j%u es_lo=0x%x rsrc2=0x%x "
                            "ud0..3=%08x %08x %08x %08x\n",
                            (unsigned long long)command_order, (unsigned)c.queue_origin,
                            g_fold_seq.load(std::memory_order_relaxed), jump_depth,
                            rd(0xc8), rd(0x8b), rd(0x8c), rd(0x8d), rd(0x8e), rd(0x8f));
            }
            refresh_state_snapshot();
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
            // #305: bind-vs-work sequence in stream order. Cached — this runs per draw/dispatch,
            // and a per-item environ scan is measurable at this title's ~876k items per route.
            // The three graphics arms emit "DRAW"; the two compute arms below emit "DISPATCH", so
            // an analysis over these lines is never silently counting dispatches as draws (a
            // compute item does not consume SPI_SHADER_PGM_LO_ES at all, so mixing them corrupts
            // any bind/draw agreement statistic).
            static const bool bindtrace_draw = getenv("PROSPER_BINDTRACE") != nullptr;
            if (bindtrace_draw) {
                auto rd = [&](uint32_t off) { auto it = sh.find(off); return it == sh.end() ? 0u : it->second; };
                static std::atomic<int> nd{0};
                if (nd.fetch_add(1) < 2000000)
                    fprintf(stderr,
                            "[bind] DRAW order=%llu q%u f%u j%u es_lo=0x%x rsrc2=0x%x "
                            "ud0..3=%08x %08x %08x %08x\n",
                            (unsigned long long)command_order, (unsigned)c.queue_origin,
                            g_fold_seq.load(std::memory_order_relaxed), jump_depth,
                            rd(0xc8), rd(0x8b), rd(0x8c), rd(0x8d), rd(0x8e), rd(0x8f));
            }
            refresh_state_snapshot();
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
            // #305: bind-vs-work sequence in stream order. Cached — this runs per draw/dispatch,
            // and a per-item environ scan is measurable at this title's ~876k items per route.
            // The three graphics arms emit "DRAW"; the two compute arms below emit "DISPATCH", so
            // an analysis over these lines is never silently counting dispatches as draws (a
            // compute item does not consume SPI_SHADER_PGM_LO_ES at all, so mixing them corrupts
            // any bind/draw agreement statistic).
            static const bool bindtrace_draw = getenv("PROSPER_BINDTRACE") != nullptr;
            if (bindtrace_draw) {
                auto rd = [&](uint32_t off) { auto it = sh.find(off); return it == sh.end() ? 0u : it->second; };
                static std::atomic<int> nd{0};
                if (nd.fetch_add(1) < 2000000)
                    fprintf(stderr,
                            "[bind] DRAW order=%llu q%u f%u j%u es_lo=0x%x rsrc2=0x%x "
                            "ud0..3=%08x %08x %08x %08x\n",
                            (unsigned long long)command_order, (unsigned)c.queue_origin,
                            g_fold_seq.load(std::memory_order_relaxed), jump_depth,
                            rd(0xc8), rd(0x8b), rd(0x8c), rd(0x8d), rd(0x8e), rd(0x8f));
            }
            refresh_state_snapshot();
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
            // Journal every raw DMA_DATA form before execution classification. The ordered executor
            // retains only address-backed copies, so deriving a DMA census from `dma_copies` made
            // immediate fills (including Astro Bot's GDS resets) disappear from timelines entirely.
            // This record is observation-only: the existing completion/ordered execution paths below
            // are deliberately unchanged.
            if (capture_dma_data_records) {
                if (dma_data_record_count != UINT64_MAX) {
                    ++dma_data_record_count;
                } else {
                    dma_data_records_truncated = true;
                }
                if (dma_data_records.size() < kMaxDmaDataRecords) {
                    dma_data_records.push_back({c.dd_dst, c.dd_src, c.dd_bytes, c.dd_sels,
                                                command_order, pkt_addr(c)});
                } else {
                    dma_data_records_truncated = true;
                }
            }
            // Address-backed copies are ordinary in-stream producers: retain them beside draws and
            // dispatches so the ordered executor exposes old bytes to earlier consumers and copied
            // bytes to later consumers (#189). Immediate fills keep their established completion-
            // FIFO behavior until a general copy appears; its suffix joins the ordered timeline.
            if (dma_data_address_source(c)) {
                if (!c.dd_valid) { report_invalid_dma_data(c); break; }
                // Destination ordering is domain-aware: guest VAs and GDS offsets never alias one
                // another, but either kind can carry a pending same-domain write across folds.
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
        case K::WaitRegMem: {
            const bool wait_overlaps_dma = !dma_copies.empty() && c.wm_valid && c.wm_addr &&
                !(c.wm_addr & 3) &&
                retained_dma_destination_overlaps(dma_copies, c.wm_addr, 8);
            const bool wait_overlaps_effect = !ordered_memory_effects.empty() && c.wm_valid &&
                c.wm_addr && !(c.wm_addr & 3) &&
                retained_ordered_effect_destination_overlaps(
                    ordered_memory_effects, c.wm_addr, 8);
            bool ordered_wait_satisfied = false;
            const bool wait_value_needed =
                (wait_overlaps_effect && !wait_overlaps_dma) ||
                (wait_overlaps_dma && std::getenv("PROSPER_GFXLOG"));
            const bool wait_readable = wait_value_needed &&
                c.wm_valid && c.wm_addr && !(c.wm_addr & 3) &&
                guest_readable(c.wm_addr, sizeof(uint64_t));
            uint64_t ordered_base_value = 0;
            OrderedQwordOverlayResult ordered_overlay;
            if (wait_readable) {
                memcpy(&ordered_base_value,
                       reinterpret_cast<const void*>(uintptr_t(c.wm_addr)),
                       sizeof(ordered_base_value));
                pend_overlay_qword(c.wm_addr, &ordered_base_value);
                ordered_overlay.value = ordered_base_value;
            } else if (wait_overlaps_dma || wait_overlaps_effect) {
                ordered_overlay.overlay = OrderedQwordOverlay::Ambiguous;
                ordered_overlay.reason = OrderedWaitEffectClass::Unsupported;
            }
            if (wait_overlaps_effect && !wait_overlaps_dma && wait_readable) {
                uint64_t ordered_value = 0;
                ordered_value = ordered_base_value;
                ordered_overlay = overlay_ordered_qword(
                    ordered_memory_effects, c.wm_addr, &ordered_value);
                ordered_wait_satisfied =
                    ordered_overlay.overlay == OrderedQwordOverlay::Touched &&
                    wait_regmem_value_satisfied(c, ordered_value);
            }
            if (ordered_wait_satisfied)
                diagnose_ordered_wait_acceptance(
                    c, ordered_memory_effects, ordered_base_value, ordered_overlay);
            if (wait_overlaps_dma) {
                // In-order execution (#2975): apply the retained DMAs so the wait observes the
                // copied bytes. Real hardware has no "rejection" — the DMA engine writes, then
                // the wait polls the written value and proceeds.
                apply_retained_dma_copies(dma_copies);
            }
            if (wait_overlaps_effect && !ordered_wait_satisfied) {
                dma_execution_rejected = true;
                diagnose_ordered_wait_rejection(
                    c, dma_copies, ordered_memory_effects, wait_readable,
                    ordered_base_value, ordered_overlay, ordered_wait_satisfied);
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    fprintf(stderr,
                            "[agc] ordered DMA submit rejected: WAIT_REG_MEM has unresolved earlier "
                            "retained memory-effect dependency (addr=0x%llx order=%llu)\n",
                            (unsigned long long)c.wm_addr,
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
                if (!ordered_wait_satisfied && !wait_regmem_satisfied(c)) {
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
                        int have = prosper_fence_journal_lookup(pkt_addr(c), &baddr, &bpre, &bt, nullptr);
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
        }
        case K::DispatchDirect:
            // Retain the dispatch and its exact register snapshot. The submit executor recompiles
            // supported compute programs and runs them in this vector's stream order before exposing
            // completion; unsupported programs remain visible in diagnostics (#576).
            // #305: bind-vs-work sequence in stream order. Cached — this runs per draw/dispatch,
            // and a per-item environ scan is measurable at this title's ~876k items per route.
            // The three graphics arms emit "DRAW"; the two compute arms below emit "DISPATCH", so
            // an analysis over these lines is never silently counting dispatches as draws (a
            // compute item does not consume SPI_SHADER_PGM_LO_ES at all, so mixing them corrupts
            // any bind/draw agreement statistic).
            static const bool bindtrace_draw = getenv("PROSPER_BINDTRACE") != nullptr;
            if (bindtrace_draw) {
                auto rd = [&](uint32_t off) { auto it = sh.find(off); return it == sh.end() ? 0u : it->second; };
                static std::atomic<int> nd{0};
                if (nd.fetch_add(1) < 2000000)
                    fprintf(stderr,
                            "[bind] DISPATCH order=%llu q%u f%u j%u es_lo=0x%x rsrc2=0x%x "
                            "ud0..3=%08x %08x %08x %08x\n",
                            (unsigned long long)command_order, (unsigned)c.queue_origin,
                            g_fold_seq.load(std::memory_order_relaxed), jump_depth,
                            rd(0xc8), rd(0x8b), rd(0x8c), rd(0x8d), rd(0x8e), rd(0x8f));
            }
            refresh_state_snapshot();
            dispatches.push_back({c.threads_x, c.threads_y, c.threads_z,
                                  c.dispatch_modifier, last_snapshot_, command_order});
            dispatch_count++;
            break;
        case K::DispatchIndirect: {
            // #305: bind-vs-work sequence in stream order. Cached — this runs per draw/dispatch,
            // and a per-item environ scan is measurable at this title's ~876k items per route.
            // The three graphics arms emit "DRAW"; the two compute arms below emit "DISPATCH", so
            // an analysis over these lines is never silently counting dispatches as draws (a
            // compute item does not consume SPI_SHADER_PGM_LO_ES at all, so mixing them corrupts
            // any bind/draw agreement statistic).
            static const bool bindtrace_draw = getenv("PROSPER_BINDTRACE") != nullptr;
            if (bindtrace_draw) {
                auto rd = [&](uint32_t off) { auto it = sh.find(off); return it == sh.end() ? 0u : it->second; };
                static std::atomic<int> nd{0};
                if (nd.fetch_add(1) < 2000000)
                    fprintf(stderr,
                            "[bind] DISPATCH order=%llu q%u f%u j%u es_lo=0x%x rsrc2=0x%x "
                            "ud0..3=%08x %08x %08x %08x\n",
                            (unsigned long long)command_order, (unsigned)c.queue_origin,
                            g_fold_seq.load(std::memory_order_relaxed), jump_depth,
                            rd(0xc8), rd(0x8b), rd(0x8c), rd(0x8d), rd(0x8e), rd(0x8f));
            }
            refresh_state_snapshot();
            Dispatch d;
            d.modifier = c.dispatch_modifier;
            d.state = last_snapshot_;
            d.command_order = command_order;
            d.indirect = true;
            // The ACB form carries the whole address (see R_DISPATCH_INDIRECT_ADDR): the
            // async-compute ring has no base register, so adding `indirect_compute_base` here would
            // be adding the DCB's unrelated base to a complete address.
            if (c.indirect_address_absolute)
                d.indirect_args_addr = c.indirect_address;
            else if (indirect_compute_base <= UINT64_MAX - c.indirect_offset)
                d.indirect_args_addr = indirect_compute_base + c.indirect_offset;
            // Recover a low-only argument address against the learned VA aperture.
            //
            // `indirect_compute_base` is per-fold state, but a PS5 process has ONE GPU address
            // space. GTA V's async-compute queue carries DispatchIndirect packets whose 32-bit
            // payload is a full address within the already-selected aperture, and no SetBase of its
            // own — so its base is zero and the args resolve to an unmapped low address. Measured on
            // a routed boot: 50 of 64 indirect compute dispatches, all on that queue, every one
            // skipped as "unreadable arguments".
            //
            // OPT-IN (PROSPER_INDIRECT_APERTURE_RECOVERY=1), and it must stay that way until the
            // packet's real provenance is established.
            //
            // "Fail-closed" was claimed for this on the grounds that recovery is attempted only when
            // the current address is unreadable and accepted only when the recovered one is readable.
            // That argument does not hold: `g_indirect_va_aperture` learns the high 32 bits from ANY
            // SetBase, on any queue, in any fold, and one process VA space can hold mapped
            // allocations under several high-32 prefixes. Mapped is not the same as "this is the
            // argument buffer" -- the failure it admits is reading group counts out of an unrelated
            // live allocation and dispatching them, which is worse than the skip it replaces and is
            // invisible at the point it happens.
            //
            // What it buys, measured on a routed boot, so the trade is recorded rather than implied:
            // with it off, 50 of 64 indirect compute dispatches (all on the async-compute queue) are
            // skipped as "unreadable arguments"; with it on, 0 are, and the probe found the raw low
            // address unmapped and `aperture | low` mapped on 49 of 49. That is a real signal about
            // where those arguments live, and it is still not provenance.
            if (aperture_recovery_enabled() && !c.indirect_address_absolute &&
                !indirect_compute_base && d.indirect_args_addr &&
                !guest_readable(d.indirect_args_addr, 3u * sizeof(uint32_t))) {
                const uint64_t aperture = g_indirect_va_aperture.load(std::memory_order_relaxed);
                const uint64_t recovered =
                    (aperture << 32) | (d.indirect_args_addr & 0xffffffffull);
                if (aperture && guest_readable(recovered, 3u * sizeof(uint32_t))) {
                    static std::atomic<int> recovered_count{0};
                    if (recovered_count.fetch_add(1) < 24)
                        fprintf(stderr,
                                "[agc] indirect dispatch args recovered into the guest VA aperture: "
                                "0x%llx -> 0x%llx (queue %u)\n",
                                (unsigned long long)d.indirect_args_addr,
                                (unsigned long long)recovered, (unsigned)c.queue_origin);
                    d.indirect_args_addr = recovered;
                }
            }
            // Split the sum WITHOUT PROSPER_INDIRECTLOG, bounded, and only for the packets that are
            // already lost. The executor's "unreadable arguments at 0x…" prints the sum alone, which
            // cannot separate "the base was never set on this queue" (base 0, offset carries a whole
            // low address) from "the base arrived truncated" (base low-only, offset small) — and
            // those two have different fixes. The readability probe runs only while the bound is
            // unspent, so a title with thousands of indirect dispatches pays for 24 of them.
            {
                static std::atomic<int> unreadable{0};
                if (unreadable.load(std::memory_order_relaxed) < 24 && d.indirect_args_addr &&
                    !guest_readable(d.indirect_args_addr, 3u * sizeof(uint32_t)) &&
                    unreadable.fetch_add(1) < 24)
                    fprintf(stderr,
                            "[agc] indirect dispatch args UNREADABLE: q=%u base=0x%llx offset=0x%x "
                            "-> args=0x%llx%s\n",
                            (unsigned)c.queue_origin,
                            (unsigned long long)indirect_compute_base, c.indirect_offset,
                            (unsigned long long)d.indirect_args_addr,
                            indirect_compute_base ? "" : "  BASE-UNSET");
            }
            // PROSPER_INDIRECTLOG: base and offset SEPARATELY, plus the queue. The executor's
            // "unreadable arguments at 0x…" message prints only the sum, which cannot distinguish a
            // bad offset from a base that was never set on this queue — and `indirect_compute_base`
            // is one field shared by every queue the processor folds.
            if (getenv("PROSPER_INDIRECTLOG")) {
                static std::atomic<int> logged{0};
                if (logged.fetch_add(1) < 64)
                    fprintf(stderr,
                            "[agc-dispatchindirect] q=%u base=0x%llx offset=0x%x modifier=0x%llx "
                            "(lo=0x%x hi=0x%x) -> args=0x%llx%s\n",
                            (unsigned)c.queue_origin,
                            (unsigned long long)indirect_compute_base, c.indirect_offset,
                            (unsigned long long)c.dispatch_modifier,
                            (unsigned)(c.dispatch_modifier & 0xffffffffu),
                            (unsigned)(c.dispatch_modifier >> 32),
                            (unsigned long long)d.indirect_args_addr,
                            indirect_compute_base ? "" : "  BASE-UNSET");
                // Which interpretation of a base-less packet is actually MAPPED? Probing costs
                // nothing and settles what the payload means: a raw low address, the same low bits
                // in the 0x20_ aperture every other resource in this title uses, or a genuine
                // 64-bit address whose high dword we are currently folding into the modifier.
                if (!indirect_compute_base && logged.load() < 64) {
                    const uint64_t lo = c.indirect_offset;
                    const uint64_t ap20 = lo | 0x2000000000ull;
                    const uint64_t hi64 =
                        lo | ((c.dispatch_modifier & 0xffffffffull) << 32);
                    fprintf(stderr,
                            "[agc-dispatchindirect]   readable? low=%d aperture20=%d hi-dword=%d "
                            "(low=0x%llx ap20=0x%llx hi64=0x%llx)\n",
                            (int)guest_readable(lo, 12), (int)guest_readable(ap20, 12),
                            (int)guest_readable(hi64, 12),
                            (unsigned long long)lo, (unsigned long long)ap20,
                            (unsigned long long)hi64);
                }
                // Readable is not the question; PLAUSIBLE is. A recovered address can be mapped and
                // still be the wrong bytes, and then the dispatch is declined much later for a
                // workgroup-count limit with no way to tell a genuine huge launch from a misread
                // one. Printing the three group-count dwords makes that distinction local to the
                // packet: real counts are small, and a misread reports whatever happens to live
                // there. Live GTA V evidence for why this matters — 0xff000000 in all three fields.
                if (d.indirect_args_addr && guest_readable(d.indirect_args_addr, 12) &&
                    logged.load() < 64) {
                    const uint32_t* args =
                        reinterpret_cast<const uint32_t*>(uintptr_t(d.indirect_args_addr));
                    fprintf(stderr,
                            "[agc-dispatchindirect]   args=[%u, %u, %u] (0x%08x 0x%08x 0x%08x)%s\n",
                            args[0], args[1], args[2], args[0], args[1], args[2],
                            (args[0] > 0xffffffu || args[1] > 0xffffffu || args[2] > 0xffffffu)
                                ? "  IMPLAUSIBLE" : "");
                }
            }
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
            // #1982: decline only on a REAL overlap, the way the two sibling readers already do.
            //
            // This used to reject the whole submit whenever ANY DMA was retained, without asking
            // whether that DMA touched the memory the jump reads. `SetRegsIndirect` (above) tests
            // its register block against `retained_dma_destination_overlaps`, and `WaitRegMem`
            // tests its label; only this site painted with the full brush. And because
            // `dma_execution_rejected` discards the submit's ORDERED MEMORY EFFECTS along with the
            // jump, the label the guest polls was never written at all — so the guest waited
            // forever on a write prosper had decided to throw away.
            //
            // Two titles proved it, both stalling on the last graphics event of the run with every
            // thread parked and the missing side ours: The Oregon Trail (PPSA19244, #1982) right
            // after its health-warning screen, and Crisis Core (PPSA07809) with 90 threads parked
            // through AgcRHI down to an `AgcInterruptThr` waiting on a GPU event that never came.
            // Lifting the blanket decline took Crisis Core from guest frame ~25 to f124, mounted
            // all nine pak containers, and doubled `present_count` within a second — and BOTH the
            // skip-only and execute arms cleared it, which is what localises the defect to the
            // submit-wide decline rather than to the jump.
            //
            // The jump reads exactly two things, so those are exactly what to test: the target
            // segment it is about to run, and the predication condition that decides whether to run
            // it. A retained DMA landing on either really would make the jump read stale bytes;
            // anything else is unrelated traffic and must not cost the guest its label.
            // CONFIDENCE: HIGH — the narrow contract is the one the sibling readers already use,
            // and the blanket form is falsified by both titles clearing under either arm.
            const uint64_t jump_bytes =
                c.jump_valid ? static_cast<uint64_t>(c.jump_dwords) * 4u : 0u;
            const bool jump_reads_dma =
                !dma_copies.empty() &&
                ((c.jump_addr && jump_bytes &&
                  retained_dma_destination_overlaps(dma_copies, c.jump_addr, jump_bytes)) ||
                 (c.jump_pred && pred_cond_addr &&
                  retained_dma_destination_overlaps(dma_copies, pred_cond_addr, 8)));
            const bool jump_reads_effect =
                !ordered_memory_effects.empty() &&
                ((c.jump_addr && jump_bytes &&
                  retained_ordered_effect_destination_overlaps(
                      ordered_memory_effects, c.jump_addr, jump_bytes)) ||
                 (c.jump_pred && pred_cond_addr &&
                  retained_ordered_effect_destination_overlaps(
                      ordered_memory_effects, pred_cond_addr, 8)));
            if (jump_reads_dma) {
                // In-order execution (#2975): apply the retained DMAs so the jump target and
                // predication reads see the copied bytes.
                apply_retained_dma_copies(dma_copies);
            }
            if (jump_reads_effect) {
                dma_execution_rejected = true;
                static std::atomic<int> warned{0};
                const uint64_t ord = warned.fetch_add(1) + 1;
                if (ord <= 24 || (ord & (ord - 1)) == 0)
                    fprintf(stderr,
                            "[agc] ordered DMA submit rejected #%llu: Jump reads target/predication "
                            "memory overlapping a retained memory-effect (target=0x%llx bytes=%llu "
                            "cond=0x%llx order=%llu)\n",
                            (unsigned long long)ord,
                            (unsigned long long)c.jump_addr, (unsigned long long)jump_bytes,
                            (unsigned long long)pred_cond_addr,
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
                // A flat first-N cap answers only about start-up. On a routed GTA V boot the 3D
                // chain does not begin until roughly 87% of the run, so a 96-line cap expired
                // thousands of frames before the phase under investigation and reported the loading
                // screen's jumps as though they were gameplay's. Log on a power-of-two schedule so
                // the whole run is sampled at bounded volume, and carry running EXEC/SKIP totals so
                // the ratio is readable without counting lines.
                static std::atomic<uint64_t> seen{0}, executed{0}, skipped{0};
                const uint64_t n = seen.fetch_add(1) + 1;
                (skip ? skipped : executed).fetch_add(1, std::memory_order_relaxed);
                if (n <= 32 || (n & (n - 1)) == 0)
                    fprintf(stderr,
                            "[pred] fold Jump #%llu target=0x%llx ndw=%u pred=%u cond@0x%llx=0x%llx "
                            "-> %s (exec=%llu skip=%llu)\n",
                            (unsigned long long)n, (unsigned long long)c.jump_addr, c.jump_dwords,
                            c.jump_pred, (unsigned long long)pred_cond_addr,
                            (unsigned long long)cond, skip ? "SKIP" : "EXEC",
                            (unsigned long long)executed.load(),
                            (unsigned long long)skipped.load());
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

size_t run_command_buffer(const uint32_t* buf, size_t dwords, GpuState& st,
                          size_t* consumed_dwords) {
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
    const size_t consumed = decode_pm4(buf, dwords, ops);
    if (consumed_dwords) *consumed_dwords = consumed;
    for (auto& c : ops) {
        c.stream_order = st.command_order + 1;
        c.queue_origin = g_fold_origin;   // #1226: retained by deferred/pended effects
        st.apply(c);
    }
    // PROSPER_BINDTRACE (#305): per-top-level-fold census. A stream that issues draws but contains
    // no shader-program bind of its own is running on the register state a PREVIOUS submit left —
    // legitimate on a shared ring, and the exact shape to check when a draw's user-data block does
    // not match the pipeline it appears to be bound to. A stream whose leading packets were never
    // decoded shows up here as draws-without-binds.
    //
    // Caveat this census cannot avoid: it walks the DECODED packet list, and the indirect arrays it
    // dereferences are re-read from guest memory AFTER the fold, not sampled at each packet's own
    // apply() moment. If the guest mutates an array between the two, the counts describe the later
    // contents. Every array measured on the reference title applied exactly one distinct
    // (es_lo, rsrc2) across thousands of folds, so that has not bitten here — but a count from this
    // census is a census, not a replay.
    static const bool bindtrace_fold = getenv("PROSPER_BINDTRACE") != nullptr;
    if (bindtrace_fold && st.jump_depth == 0) {
        using K = Pm4Command::Kind;
        uint32_t sh_indirect = 0, sh_direct = 0, pgm_writes = 0, draws = 0;
        for (const auto& c : ops) {
            if (c.kind == K::SetRegsIndirect && c.reg_class == RegClass::Sh) {
                ++sh_indirect;
                // Same bound apply() enforces before reading this guest array. Without it the
                // byte count below overflows uint32 at num_regs >= 0x20000000, guest_readable is
                // asked for 0 bytes, passes, and the loop walks 4 GiB — re-opening #312/#448.
                if (c.regs_vaddr && c.num_regs && c.num_regs <= GpuState::kMaxRegsPerPacket &&
                    guest_readable(c.regs_vaddr,
                                   c.num_regs * (uint32_t)sizeof(ShaderReg))) {
                    const auto* r = reinterpret_cast<const ShaderReg*>(
                        static_cast<uintptr_t>(c.regs_vaddr));
                    for (uint32_t i = 0; i < c.num_regs; i++)
                        if (r[i].offset == prosper::agc::Pm4::SPI_SHADER_PGM_LO_ES) ++pgm_writes;
                }
            } else if (c.kind == K::SetRegDirect && c.reg_class == RegClass::Sh) {
                ++sh_direct;
                if (c.reg_offset <= prosper::agc::Pm4::SPI_SHADER_PGM_LO_ES &&
                    c.reg_offset + c.reg_count > prosper::agc::Pm4::SPI_SHADER_PGM_LO_ES)
                    ++pgm_writes;
            } else if (c.kind == K::DrawIndex || c.kind == K::DrawIndexAuto ||
                       c.kind == K::DrawIndexOffset || c.kind == K::DrawIndexIndirect) {
                ++draws;
            }
        }
        // Which SH registers a stream touches matters as much as how many: #1226 gave the async
        // compute (Acb) queue its own register file because its SH writes were clobbering live
        // graphics user data. If an Acb stream writes GRAPHICS user-data offsets rather than
        // COMPUTE_USER_DATA, that split now DROPS writes hardware would apply, and the next
        // graphics submit inherits a register file the hardware would never have had. Report the
        // touched offset range so the two cases are distinguishable.
        uint32_t sh_lo = UINT32_MAX, sh_hi = 0, sh_gfx_ud = 0, sh_compute_ud = 0;
        for (const auto& c : ops) {
            if (c.reg_class != RegClass::Sh) continue;
            auto note = [&](uint32_t off) {
                sh_lo = std::min(sh_lo, off); sh_hi = std::max(sh_hi, off);
                namespace P = prosper::agc::Pm4;
                if ((off >= P::SPI_SHADER_USER_DATA_PS_0 && off < P::SPI_SHADER_USER_DATA_PS_0 + 32) ||
                    (off >= P::SPI_SHADER_USER_DATA_VS_0 && off < P::SPI_SHADER_USER_DATA_VS_0 + 32) ||
                    (off >= P::SPI_SHADER_USER_DATA_GS_0 && off < P::SPI_SHADER_USER_DATA_GS_0 + 32) ||
                    (off >= P::SPI_SHADER_USER_DATA_HS_0 && off < P::SPI_SHADER_USER_DATA_HS_0 + 32))
                    ++sh_gfx_ud;   // all four graphics stages, not just the two this title uses
                else if (off >= P::COMPUTE_USER_DATA_0 && off < P::COMPUTE_USER_DATA_0 + 16)
                    ++sh_compute_ud;
            };
            if (c.kind == K::SetRegDirect) {
                for (uint32_t k = 0; k < c.reg_count; k++) note(c.reg_offset + k);
            } else if (c.kind == K::SetRegsIndirect && c.regs_vaddr && c.num_regs &&
                       c.num_regs <= GpuState::kMaxRegsPerPacket &&
                       guest_readable(c.regs_vaddr,
                                      c.num_regs * (uint32_t)sizeof(ShaderReg))) {
                const auto* r = reinterpret_cast<const ShaderReg*>(
                    static_cast<uintptr_t>(c.regs_vaddr));
                for (uint32_t i = 0; i < c.num_regs; i++) note(r[i].offset);
            }
        }
        if (draws || sh_direct || sh_indirect) {
            static std::atomic<int> n{0};
            if (n.fetch_add(1) < 200000)
                fprintf(stderr,
                        "[bind] FOLD f=%u q=%u dwords=%zu packets=%zu sh_indirect=%u sh_direct=%u "
                        "es_pgm_writes=%u draws=%u sh_off=[0x%x,0x%x] gfx_ud=%u compute_ud=%u\n",
                        g_fold_seq.load(std::memory_order_relaxed), (unsigned)g_fold_origin,
                        dwords, ops.size(), sh_indirect, sh_direct, pgm_writes, draws,
                        sh_lo == UINT32_MAX ? 0u : sh_lo, sh_hi, sh_gfx_ud, sh_compute_ud);
        }
    }
    return ops.size();
}

} // namespace prosper::gpu
