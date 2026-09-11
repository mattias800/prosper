#include "gpu/diagnostics/gpu_memory_budget.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace prosper::gpu {
namespace {

// Vulkan caps heaps at VK_MAX_MEMORY_HEAPS (16). Fixed storage keeps the per-heap counters usable
// from an allocation path that must not itself allocate.
constexpr uint32_t kMaxHeaps = 16;

std::atomic<uint64_t> g_held[kMaxHeaps];      // bytes prosper currently holds, per heap
std::atomic<uint64_t> g_peak[kMaxHeaps];      // high-water mark, so a freed spike still shows
std::atomic<uint64_t> g_next_step[kMaxHeaps]; // next held-bytes threshold that prints

// handle -> (bytes, heap), so a free can give back exactly what its allocation took. The free sites
// hold only a VkDeviceMemory; making each one carry the size and heap would be several places to
// get wrong instead of one. A map and a mutex are affordable here because these sit on
// vkAllocateMemory, not on a draw.
struct LiveAlloc { uint64_t bytes; uint32_t heap; };
std::mutex g_live_mx;
std::unordered_map<uint64_t, LiveAlloc> g_live;

// The device's heap layout, recorded once. Held under its own mutex and read on every allocation,
// which is why it is copied into fixed arrays rather than pointed at: the caller's
// VkPhysicalDeviceMemoryProperties may be a stack temporary.
constexpr uint32_t kMaxTypes = 32;               // VK_MAX_MEMORY_TYPES
std::mutex g_layout_mx;
uint32_t g_type_heap[kMaxTypes];
uint64_t g_heap_size[kMaxHeaps];
uint32_t g_type_count = 0;
uint32_t g_heap_count = 0;

bool logging_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("PROSPER_GPU_MEM_LOG");
        return !(v && v[0] == '0' && v[1] == '\0');   // default ON; only "0" silences it
    }();
    return on;
}

uint64_t step_bytes() {
    static const uint64_t step = [] () -> uint64_t {
        const char* v = std::getenv("PROSPER_GPU_MEM_LOG_MIB");
        const long long mib = v ? std::atoll(v) : 0;
        // Clamp before multiplying. A value near 2^44 would overflow the shift to exactly zero, and
        // `held / step` below would then divide by zero on the first allocation -- a crash in a
        // diagnostic, triggered by a typo in its own environment variable.
        const uint64_t clamped = (mib > 0 && mib < (1ll << 32)) ? (uint64_t)mib : 256ull;
        return clamped * 1024ull * 1024ull;
    }();
    return step;
}

const char* pressure_label(uint64_t held, uint64_t heap_size) {
    if (!heap_size) return "";
    const uint64_t pct = held * 100ull / heap_size;
    if (pct >= 90) return "  *** OVER 90% OF THE HEAP ***";
    if (pct >= 70) return "  (over 70% of the heap)";
    return "";
}

}  // namespace

void set_device_heaps(const uint32_t* type_heap_index, uint32_t type_count,
                      const uint64_t* heap_sizes, uint32_t heap_count) {
    if (!type_heap_index || !heap_sizes) return;
    const uint32_t types = type_count < kMaxTypes ? type_count : kMaxTypes;
    const uint32_t heaps = heap_count < kMaxHeaps ? heap_count : kMaxHeaps;
    std::lock_guard<std::mutex> lk(g_layout_mx);
    if (g_type_count) {
        // Already set. Keep the first layout rather than renumbering the heaps under a count that
        // is already running -- but say so, because silently mixing two devices' heaps would make
        // every number printed afterwards wrong in a way nothing in the output could reveal.
        bool same = g_type_count == types && g_heap_count == heaps;
        for (uint32_t i = 0; same && i < types; i++) same = g_type_heap[i] == type_heap_index[i];
        for (uint32_t i = 0; same && i < heaps; i++) same = g_heap_size[i] == heap_sizes[i];
        if (!same && logging_enabled()) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr, "[gpu-mem] a SECOND device with a different heap layout was "
                                     "registered; keeping the first, so these totals cover one "
                                     "device only\n");
            }
        }
        return;
    }
    for (uint32_t i = 0; i < types; i++) g_type_heap[i] = type_heap_index[i];
    for (uint32_t i = 0; i < heaps; i++) g_heap_size[i] = heap_sizes[i];
    g_type_count = types;
    g_heap_count = heaps;
}

void note_device_alloc(uint64_t handle, uint32_t memory_type, uint64_t bytes) {
    if (!bytes || !handle) return;
    uint32_t heap = 0;
    uint64_t heap_size = 0;
    {
        std::lock_guard<std::mutex> lk(g_layout_mx);
        // No layout yet means set_device_heaps was never called -- a caller wired an allocation site
        // without wiring its device creation. Counting it against heap 0 would attribute bytes to a
        // heap they may not be on, so drop it rather than print a confident wrong attribution.
        if (memory_type >= g_type_count) return;
        heap = g_type_heap[memory_type];
        if (heap >= g_heap_count) return;
        heap_size = g_heap_size[heap];
    }
    if (heap >= kMaxHeaps) return;
    {
        std::lock_guard<std::mutex> lk(g_live_mx);
        // Overwrite rather than insert: a driver is free to hand back a handle value it has already
        // freed, and a stale entry would make the matching free subtract the wrong size from the
        // wrong heap.
        g_live[handle] = LiveAlloc{bytes, heap};
    }
    const uint64_t held = g_held[heap].fetch_add(bytes, std::memory_order_relaxed) + bytes;

    // Peak is a max, not a store: a run that allocates 3 GiB and frees it has still demanded 3 GiB
    // of the heap, and that is the number that explains a submission failure after the fact.
    uint64_t peak = g_peak[heap].load(std::memory_order_relaxed);
    while (held > peak &&
           !g_peak[heap].compare_exchange_weak(peak, held, std::memory_order_relaxed)) {}

    if (!logging_enabled()) return;
    const uint64_t step = step_bytes();
    // `observed` is what the counter really holds and is what the CAS must compare against;
    // `threshold` is what it MEANS. They differ only in the initial state, where a stored 0 stands
    // for "one whole step". Folding the two into one variable is a defect that hides completely:
    // the substituted threshold never matches the stored 0, so the exchange fails, the counter is
    // never advanced, and every later crossing fails the same way -- the growth line then never
    // prints for the life of the process while every counter stays perfectly correct. It shipped
    // that way through a green 18-assertion test and was caught only by looking at a real run.
    uint64_t observed = g_next_step[heap].load(std::memory_order_relaxed);
    const uint64_t threshold = observed ? observed : step;
    if (held < threshold) return;
    // Advance past every step this allocation jumped, so one huge allocation prints once rather
    // than arming a burst of lines on the allocations that follow it.
    const uint64_t advanced = ((held / step) + 1ull) * step;
    if (!g_next_step[heap].compare_exchange_strong(observed, advanced, std::memory_order_relaxed))
        return;   // another thread is printing this crossing

    std::fprintf(stderr,
                 "[gpu-mem] prosper holds %llu MiB on heap %u of %llu MiB (peak %llu MiB)%s\n",
                 (unsigned long long)(held >> 20), heap,
                 (unsigned long long)(heap_size >> 20),
                 (unsigned long long)(g_peak[heap].load(std::memory_order_relaxed) >> 20),
                 pressure_label(held, heap_size));
}

void note_device_free(uint64_t handle) {
    if (!handle) return;
    LiveAlloc rec{};
    {
        std::lock_guard<std::mutex> lk(g_live_mx);
        const auto found = g_live.find(handle);
        if (found == g_live.end()) return;   // never counted, so there is nothing to give back
        rec = found->second;
        g_live.erase(found);
    }
    if (rec.heap >= kMaxHeaps) return;
    uint64_t held = g_held[rec.heap].load(std::memory_order_relaxed);
    // Never wrap. Clamping to zero keeps the number readable instead of printing 16 exabytes if the
    // bookkeeping is ever wrong; the peak above still records what was really demanded.
    while (true) {
        const uint64_t next = rec.bytes > held ? 0 : held - rec.bytes;
        if (g_held[rec.heap].compare_exchange_weak(held, next, std::memory_order_relaxed)) break;
    }
}

uint64_t device_bytes_held(uint32_t heap) {
    return heap < kMaxHeaps ? g_held[heap].load(std::memory_order_relaxed) : 0;
}

uint64_t device_peak_bytes(uint32_t heap) {
    return heap < kMaxHeaps ? g_peak[heap].load(std::memory_order_relaxed) : 0;
}

void report_allocation_failure(int result, uint64_t bytes, uint32_t memory_type) {
    if (!logging_enabled()) return;
    static std::atomic<int> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) >= 4) return;
    std::fprintf(stderr,
                 "[gpu-mem] a device allocation FAILED (%d): asked for %llu MiB of memory type %u\n",
                 result, (unsigned long long)(bytes >> 20), memory_type);
    report_device_memory("allocation failed");
}

void report_device_memory(const char* why) {
    if (!logging_enabled()) return;
    uint64_t sizes[kMaxHeaps];
    uint32_t heaps = 0;
    {
        std::lock_guard<std::mutex> lk(g_layout_mx);
        heaps = g_heap_count;
        for (uint32_t i = 0; i < heaps; i++) sizes[i] = g_heap_size[i];
    }
    if (!heaps) {
        std::fprintf(stderr, "[gpu-mem] %s: no device heap layout recorded yet\n",
                     why ? why : "report");
        return;
    }
    for (uint32_t i = 0; i < heaps; i++) {
        const uint64_t held = g_held[i].load(std::memory_order_relaxed);
        std::fprintf(stderr,
                     "[gpu-mem] %s: heap %u is %llu MiB; prosper holds %llu MiB (peak %llu MiB)%s\n",
                     why ? why : "report", i,
                     (unsigned long long)(sizes[i] >> 20),
                     (unsigned long long)(held >> 20),
                     (unsigned long long)(g_peak[i].load(std::memory_order_relaxed) >> 20),
                     pressure_label(held, sizes[i]));
    }
    // Say what the numbers do NOT cover, every time they are printed. On an integrated GPU these
    // heaps are shared with the compositor, so "prosper holds far less than the heap" is not on its
    // own evidence that an allocation will succeed -- and reading it that way is the mistake this
    // instrument exists because of (#3533).
    std::fprintf(stderr,
                 "[gpu-mem] these count PROSPER's own vkAllocateMemory only -- not other processes, "
                 "the compositor, or driver overhead\n");
}

}  // namespace prosper::gpu
