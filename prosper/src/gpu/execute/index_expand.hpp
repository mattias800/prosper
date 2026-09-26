#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define PROSPER_INDEX_EXPAND_AVX2 1
#else
#define PROSPER_INDEX_EXPAND_AVX2 0
#endif

namespace prosper::gpu {

// dst is a separately allocated host vector; src is the guest's validated 16-bit index range.
// Read exactly count indices so a valid range ending at a guest mapping boundary stays valid.
//
// This is the ONLY path the emulator takes, and it is not scalar despite looking like it: the
// compiler already vectorizes this loop at the project's optimization level, because the widening
// u16->u32 store and the unsigned-max reduction are both idioms GCC recognizes. At plain `-O2` it
// emits `movdqu` -> `punpcklwd`/`punpckhwd` -> two `movups`, i.e. 8 indices per iteration; given
// AVX2 it goes to 16. Check the emitted code before assuming a hand-written kernel has anything
// to beat -- see `docs/OUTER_WILDS_STATUS.md` § Ruled out, where one already lost.
inline uint32_t copy_indices_u16_max(uint32_t* dst, const uint16_t* src, size_t count) {
    uint32_t maximum = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t index = src[i];
        dst[i] = index;
        maximum = std::max(maximum, index);
    }
    return maximum;
}

#if PROSPER_INDEX_EXPAND_AVX2
// Only `tests/gpu/execute/test_index_expand.cpp` calls this.
inline bool index_expand_avx2_available() {
    static const bool available = __builtin_cpu_supports("avx2");
    return available;
}

// NOT CALLED BY THE EMULATOR, and that is deliberate rather than an oversight.
//
// It is retained as the executable half of the falsification in `docs/OUTER_WILDS_STATUS.md`
// § Ruled out: `test_index_expand --bench` interleaves it against the loop above in one binary
// and prints each kernel's per-index cost, which is the measurement that row's third reason
// rests on. Anyone re-opening "should index expansion be hand-vectorized?" can re-run that in
// seconds instead of re-deriving it.
//
// There is no environment variable and no dispatcher branch. An earlier revision of this file
// had both; they were removed on review (#3866) because a runtime switch could only ever select
// the option the measurement rules out, and because index expansion is a pure function of
// (dst, src, count) -- fully A/B-able in a unit test, unlike a gate such as
// `PROSPER_UD_TAIL_ALIGN` whose effect is only observable end-to-end in a live title.
//
// Each iteration loads exactly 8 indices (16 bytes) and stores exactly 8 (32 bytes), so the
// read never passes `src + count` even when that address is a guest mapping edge. The maximum
// is reduced from the SAME widened register that is stored, so a concurrent guest rewrite can
// never yield a stored index above the returned maximum -- the caller sizes the vertex buffer
// from that maximum.
//
// CONFIDENCE: HIGH that this agrees with the loop above. The same test compares them value by
// value across 19 lengths x 3 source offsets x 3 output offsets x 5 peak positions, and again
// against a `PROT_NONE` page so the wide load is checked at the mapping edge it must not cross.
__attribute__((target("avx2")))
inline uint32_t copy_indices_u16_max_avx2(uint32_t* dst, const uint16_t* src, size_t count) {
    __m256i vector_max = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i packed;
        std::memcpy(&packed, src + i, sizeof(packed));
        const __m256i widened = _mm256_cvtepu16_epi32(packed);
        std::memcpy(dst + i, &widened, sizeof(widened));
        vector_max = _mm256_max_epu32(vector_max, widened);
    }
    alignas(32) uint32_t lanes[8];
    std::memcpy(lanes, &vector_max, sizeof(lanes));
    uint32_t maximum = 0;
    for (uint32_t lane : lanes) maximum = std::max(maximum, lane);
    for (; i < count; ++i) {
        const uint32_t index = src[i];
        dst[i] = index;
        maximum = std::max(maximum, index);
    }
    return maximum;
}
#else
inline bool index_expand_avx2_available() { return false; }
#endif

// `PROSPER_INDEX_EXPAND_STATS=1`: how many indices this emulator actually expands per second.
//
// This exists because the question "is index expansion worth optimizing?" has exactly one
// unknown, and it is not the kernel's speed. `test_index_expand --bench` measures ns/index;
// multiply that by what this reports to get the loop's share of a frame. Without the volume,
// a kernel speedup of any size converts to an unknown end-to-end effect.
//
// The 16- and 32-bit element widths are counted separately because only the 16-bit volume is
// a candidate for the kernel above, and a title can be entirely one or the other. Both widths
// are printed on every line, so a title that expands only 32-bit indices still reports an
// explicit 16-bit zero instead of looking unarmed.
//
// The blind spot that leaves, stated rather than papered over: this is called from the draw
// path, so a run with NO indexed draws at all prints nothing, and nothing is exactly what an
// unarmed census prints. Confirm the run rendered before reading silence as a measured zero.
//
// Volume only, deliberately: timing each call would need two clock reads per indexed draw,
// which on a small draw is comparable to the work being timed. Counters are relaxed atomics
// because indexed draws are realized on more than one thread, and the report is emitted by
// whichever thread crosses the interval, so its window boundaries are approximate by design.
struct IndexExpandCensus {
    std::atomic<uint64_t> calls[2]{};        // [0] 16-bit elements, [1] every other width
    std::atomic<uint64_t> indices[2]{};
    std::mutex report_mutex;
    // Guarded by report_mutex; only the reporting thread reads or writes them.
    uint64_t reported_calls[2]{};
    uint64_t reported_indices[2]{};
    std::chrono::steady_clock::time_point started{std::chrono::steady_clock::now()};
    std::chrono::steady_clock::time_point reported{started};
    // Mirrors `reported` for the lock-free interval check in index_expand_record.
    std::atomic<int64_t> reported_ns{std::chrono::duration_cast<std::chrono::nanoseconds>(
        started.time_since_epoch()).count()};
};

inline IndexExpandCensus& index_expand_census() {
    static IndexExpandCensus census;
    return census;
}

inline bool index_expand_census_enabled() {
    static const bool enabled = [] {
        const char* setting = std::getenv("PROSPER_INDEX_EXPAND_STATS");
        return setting && setting[0] == '1' && setting[1] == '\0';
    }();
    return enabled;
}

inline void index_expand_record(size_t count, unsigned element_bytes) {
    if (!index_expand_census_enabled()) return;
    IndexExpandCensus& census = index_expand_census();
    const int bucket = element_bytes == 2 ? 0 : 1;
    census.calls[bucket].fetch_add(1, std::memory_order_relaxed);
    census.indices[bucket].fetch_add(count, std::memory_order_relaxed);

    // Check the interval lock-free first. Taking the mutex on every indexed draw would put an
    // atomic CAS on a path several threads run concurrently, to answer a question that is false
    // almost every time.
    constexpr auto interval = std::chrono::seconds(2);
    const auto now = std::chrono::steady_clock::now();
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               now.time_since_epoch()).count();
    if (now_ns - census.reported_ns.load(std::memory_order_relaxed) <
        std::chrono::duration_cast<std::chrono::nanoseconds>(interval).count())
        return;
    std::unique_lock lock(census.report_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return;
    if (now - census.reported < interval) return; // another thread reported while we waited

    const double window = std::chrono::duration<double>(now - census.reported).count();
    const double lifetime = std::chrono::duration<double>(now - census.started).count();
    uint64_t calls[2], indices[2], window_indices[2];
    for (int width = 0; width < 2; ++width) {
        calls[width] = census.calls[width].load(std::memory_order_relaxed);
        indices[width] = census.indices[width].load(std::memory_order_relaxed);
        window_indices[width] = indices[width] - census.reported_indices[width];
    }
    std::fprintf(stderr,
                 "[index-expand] %.2fs window: 16-bit %llu calls / %llu indices (%.3f M/s), "
                 "other %llu calls / %llu indices (%.3f M/s) | since start %.1fs: "
                 "16-bit %llu calls / %llu indices (%.3f M/s), other %llu calls / %llu indices\n",
                 window,
                 (unsigned long long)(calls[0] - census.reported_calls[0]),
                 (unsigned long long)window_indices[0],
                 window > 0 ? double(window_indices[0]) / window / 1e6 : 0.0,
                 (unsigned long long)(calls[1] - census.reported_calls[1]),
                 (unsigned long long)window_indices[1],
                 window > 0 ? double(window_indices[1]) / window / 1e6 : 0.0,
                 lifetime,
                 (unsigned long long)calls[0], (unsigned long long)indices[0],
                 lifetime > 0 ? double(indices[0]) / lifetime / 1e6 : 0.0,
                 (unsigned long long)calls[1], (unsigned long long)indices[1]);
    for (int width = 0; width < 2; ++width) {
        census.reported_calls[width] = calls[width];
        census.reported_indices[width] = indices[width];
    }
    census.reported = now;
    census.reported_ns.store(now_ns, std::memory_order_relaxed);
}

} // namespace prosper::gpu

#undef PROSPER_INDEX_EXPAND_AVX2
