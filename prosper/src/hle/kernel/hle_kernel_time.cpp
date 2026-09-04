// hle_kernel_time.cpp — time/clock sources, C11 thread primitives, and assorted
// libkernel stubs the engine needs during init. Cross-platform (chrono + pthread).
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/hle_kernel_time.hpp"
#include "host/image/boot_program.hpp"   // #1659: shared guest-module labelling
#include "host/platform/posix_shim.hpp"     // PROSPER_ASM_TRAMPOLINE (pass entry %rsp as 7th arg)
#include "host/platform/precise_sleep.hpp"   // guest sleeps must not inherit the winpthreads tick (#3013)
#include "hle/kernel/timedwait_census.hpp"   // PROSPER_TIMEDWAIT_CENSUS: which primitive a title paces on
#include "host/image/runtime_module_load.hpp"   // #639: real runtime PRX loading
#include "host/memory/guest_write_watch.hpp"     // flush dmem writer diagnostic before guest _Exit
#include "hle/dispatch/callback_fs.hpp"            // recover the caller's guest %fs from the import-stub frame
#include "hle/kernel/sce_errno.hpp"    // #1612: the guest reads FreeBSD errnos, not this host's
#include "hle/memory/heap_mutex.hpp"   // #707: keep hot equeue/APR mutexes off macOS __DATA
#include "hle/sync/pthread_slot.hpp"   // #2596: resolve a guest sync slot the way libkernel does
#include "hle/sync/sync_futex.hpp"
#include "hle/sync/sync_retire.hpp"   // #2042: a destroyed guest sync object's storage is retired, not freed
#include "diagnostics/env_numeric.hpp"  // #3304: strict numeric env parsing
#include <pthread.h>
#include <chrono>
#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/uio.h>   // process_vm_readv (slot-echo scan, issue #180)
#elif defined(_WIN32)
#include <windows.h>
#endif
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <cerrno>       // select/pselect fail-visible path sets errno for __error() (#1660)
#include <ctime>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define P(x) ((void*)(uintptr_t)(x))

extern "C" uint64_t prosper_vo_flip_count();

namespace {
    using clk = std::chrono::steady_clock;
    clk::time_point g_start = clk::now();
    uint64_t real_ns() {
        return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - g_start).count();
    }

    // The live GPU backend is synchronous today: the guest submitter performs shader realization,
    // Vulkan execution, and readback before its HLE call returns. On hardware that work runs on the
    // GPU after a cheap CPU submission. Without compensation, a one-off host pipeline/resource
    // warmup is incorrectly exposed as a giant guest frame delta and can skip entire animations.
    //
    // During a host-GPU scope, monotonic time advances only through the caller-supplied display
    // budget, then holds. At scope exit the excess is permanently removed from process-time/TSC.
    // Ordinary guest execution and all realtime/RTC surfaces keep advancing from the host clocks.
    // This is deliberately narrower than PROSPER_DET_CLOCK: time-gated media and wait loops can
    // continue between flips, so a title cannot deadlock waiting to produce its next frame.
    struct HostGpuClockState {
        std::mutex writer_mutex;
        std::atomic<uint64_t> sequence{0};
        std::atomic<uint64_t> total_excess_ns{0};
        std::atomic<uint64_t> active_start_ns{0};
        std::atomic<uint64_t> active_budget_ns{0};
        std::atomic<uint64_t> active_token{0};
        uint64_t next_token = 1;
        std::atomic<uint64_t> last_ns{0};
    };
    HostGpuClockState g_host_gpu_clock;

    bool host_gpu_clock_enabled() {
        static const bool enabled = getenv("PROSPER_NO_GPU_TIME_COMPENSATION") == nullptr;
        return enabled;
    }

    uint64_t host_gpu_compensated_ns(uint64_t mono) {
        // Process-time reads are hot and may come from many guest threads. Writers publish their
        // handful of atomic fields under a seqlock, giving readers one consistent snapshot without
        // serializing every clock query on the scope-management mutex.
        uint64_t total_excess;
        uint64_t active_start;
        uint64_t active_budget;
        uint64_t active_token;
        for (;;) {
            const uint64_t before = g_host_gpu_clock.sequence.load(std::memory_order_acquire);
            if (before & 1) continue;
            total_excess = g_host_gpu_clock.total_excess_ns.load(std::memory_order_relaxed);
            active_start = g_host_gpu_clock.active_start_ns.load(std::memory_order_relaxed);
            active_budget = g_host_gpu_clock.active_budget_ns.load(std::memory_order_relaxed);
            active_token = g_host_gpu_clock.active_token.load(std::memory_order_relaxed);
            const uint64_t after = g_host_gpu_clock.sequence.load(std::memory_order_acquire);
            if (before == after) break;
        }

        uint64_t excess = total_excess;
        if (active_token && mono > active_start) {
            const uint64_t elapsed = mono - active_start;
            if (elapsed > active_budget) {
                const uint64_t active_excess = elapsed - active_budget;
                excess = active_excess > UINT64_MAX - excess ? UINT64_MAX : excess + active_excess;
            }
        }
        const uint64_t current = mono >= excess ? mono - excess : 0;
        uint64_t last = g_host_gpu_clock.last_ns.load(std::memory_order_relaxed);
        while (last < current && !g_host_gpu_clock.last_ns.compare_exchange_weak(
                   last, current, std::memory_order_relaxed, std::memory_order_relaxed)) {}
        return std::max(last, current);
    }

    // PROSPER_DET_CLOCK: derive the guest MONOTONIC clock from the flip count instead of host elapsed
    // time, so per-frame deltaTime is fixed regardless of host render cost. Wall-clock/RTC surfaces keep
    // using real_ns(): deterministic gameplay time must not freeze the calendar clock between flips.
    // Before the first flip, real time keeps time-gated initialization moving. After it, monotonic time
    // intentionally pauses between flips; callers using it for timeouts therefore opt into that behavior.
    // CONFIDENCE: MED.
    struct DetClockState {
        std::mutex mutex;
        bool anchored = false;
        uint64_t anchor_ns = 0;
        uint64_t anchor_flip = 0;
        uint64_t last_ns = 0;
    };
    DetClockState g_det_clock;

    uint64_t det_clock_fps() {
        static const uint64_t fps = [] {
            const char* value = getenv("PROSPER_DET_FPS");
            if (!value || !*value) return 60ull;
            char* end = nullptr;
            uint64_t parsed = std::strtoull(value, &end, 10);
            return end != value && *end == '\0' && parsed > 0 && parsed <= 1000000ull ? parsed : 60ull;
        }();
        return fps;
    }

    uint64_t ns_now() {
        static const bool det = getenv("PROSPER_DET_CLOCK") != nullptr;
        uint64_t mono = real_ns();
        if (!det) return host_gpu_compensated_ns(mono);
        uint64_t flip = prosper_vo_flip_count();
        std::lock_guard<std::mutex> lock(g_det_clock.mutex);
        if (!g_det_clock.anchored) {
            if (flip == 0) return mono;
            g_det_clock.anchored = true;
            g_det_clock.anchor_ns = mono;
            g_det_clock.anchor_flip = flip;
            g_det_clock.last_ns = mono;
        }
        const uint64_t delta_flips = flip >= g_det_clock.anchor_flip ? flip - g_det_clock.anchor_flip : 0;
        const uint64_t current = g_det_clock.anchor_ns + delta_flips * (1000000000ull / det_clock_fps());
        g_det_clock.last_ns = std::max(g_det_clock.last_ns, current);
        return g_det_clock.last_ns;
    }

    // --- Wall-clock anchor (#92). The host real-time clock is sampled ONCE, paired with the
    // monotonic ns_now() at the same instant; every wall-clock surface (CLOCK_REALTIME,
    // gettimeofday, time(), sceRtc*) derives from base + monotonic-elapsed. This makes all of
    // them (a) agree on one "now", (b) show the true current date instead of the old synthetic
    // bases (uptime-since-1970 for clock_gettime/gettimeofday vs frozen 1700000000 ≈ Nov 2023
    // for time()/sceRtc — three notions of now, ~54 years apart), and (c) advance strictly
    // monotonically — a host NTP step after boot cannot make guest wall time jump backwards.
    struct WallAnchor { uint64_t base_us; uint64_t mono_ns; };
    const WallAnchor& wall_anchor() {
        static const WallAnchor a = [] {
            uint64_t mono = real_ns();
            uint64_t us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::system_clock::now().time_since_epoch()).count();
            return WallAnchor{ us, mono };
        }();
        return a;
    }
    uint64_t wall_now_us() {   // microseconds since the unix epoch, monotonically advancing
        const WallAnchor& a = wall_anchor();
        return a.base_us + (real_ns() - a.mono_ns) / 1000ull;
    }
    // RTC epoch: SceRtcTick counts microseconds since 0001-01-01 00:00:00 UTC; the unix epoch is
    // 62135596800 s after it (the documented Orbis RTC convention, also shadPS4 UNIX_EPOCH_TICKS).
    constexpr uint64_t kRtcUnixEpochOffsetUs = 62135596800ull * 1000000ull;
}

uint64_t guest_clock_host_gpu_begin(uint64_t budget_ns) {
    if (!host_gpu_clock_enabled()) return 0;
    std::lock_guard<std::mutex> lock(g_host_gpu_clock.writer_mutex);
    // AGC submissions are serialized, so overlap indicates a future caller violated the scope
    // contract. Fail open rather than letting an inner end truncate or double-count the outer wait.
    if (g_host_gpu_clock.active_token.load(std::memory_order_relaxed)) return 0;
    // Make the snapshot unavailable before sampling: if this writer is preempted, readers must not
    // advance against the old state and then have that already-observed interval discounted.
    g_host_gpu_clock.sequence.fetch_add(1, std::memory_order_acq_rel);
    const uint64_t mono = real_ns();
    uint64_t token = g_host_gpu_clock.next_token++;
    if (!token) token = g_host_gpu_clock.next_token++;
    g_host_gpu_clock.active_start_ns.store(mono, std::memory_order_relaxed);
    g_host_gpu_clock.active_budget_ns.store(budget_ns, std::memory_order_relaxed);
    g_host_gpu_clock.active_token.store(token, std::memory_order_relaxed);
    g_host_gpu_clock.sequence.fetch_add(1, std::memory_order_release);
    return token;
}

void guest_clock_host_gpu_end(uint64_t token) {
    if (!token) return;
    std::lock_guard<std::mutex> lock(g_host_gpu_clock.writer_mutex);
    if (g_host_gpu_clock.active_token.load(std::memory_order_relaxed) != token) return;
    // Block reader snapshots across the active-to-inactive boundary, then sample as late as possible
    // so a writer preemption cannot expose a stale end time as a completed transition.
    g_host_gpu_clock.sequence.fetch_add(1, std::memory_order_acq_rel);
    const uint64_t active_start = g_host_gpu_clock.active_start_ns.load(std::memory_order_relaxed);
    const uint64_t active_budget = g_host_gpu_clock.active_budget_ns.load(std::memory_order_relaxed);
    uint64_t total_excess = g_host_gpu_clock.total_excess_ns.load(std::memory_order_relaxed);
    const uint64_t mono = real_ns();
    const uint64_t elapsed = mono > active_start ? mono - active_start : 0;
    if (elapsed > active_budget) {
        const uint64_t excess = elapsed - active_budget;
        total_excess = excess > UINT64_MAX - total_excess ? UINT64_MAX : total_excess + excess;
    }
    g_host_gpu_clock.total_excess_ns.store(total_excess, std::memory_order_relaxed);
    g_host_gpu_clock.active_start_ns.store(0, std::memory_order_relaxed);
    g_host_gpu_clock.active_budget_ns.store(0, std::memory_order_relaxed);
    g_host_gpu_clock.active_token.store(0, std::memory_order_relaxed);
    g_host_gpu_clock.sequence.fetch_add(1, std::memory_order_release);
}

// --- time / clock (return real, advancing time so wait-for-time loops progress) ---
HLE(k_get_ptc)        { return ns_now(); }                 // sceKernelGetProcessTimeCounter
HLE(k_get_ptc_freq)   { return 1000000000ull; }            // counter is in ns -> 1 GHz
HLE(k_get_proc_time)  { return ns_now() / 1000; }          // microseconds
HLE(k_read_tsc)       { return ns_now(); }
HLE(k_tsc_freq)       { return 1000000000ull; }

// The GPU EOP timestamp (RELEASE_MEM data_sel=3 / an address-carrying EVENT_WRITE) is, on real
// hardware, the SAME counter the guest reads via sceKernelReadTsc (Kyty: GraphicsRender writes
// KernelReadTsc() for the EOP timestamp; GetGpuCoreClockFrequency == GetTscFrequency). prosper
// models that counter as this monotonic-ns clock at 1 GHz (k_tsc_freq). Exposed so the
// CommandProcessor's gpu_clock64 shares the EXACT clock + epoch (#156) — previously it used a
// separate steady_clock with a different epoch and an unspecified period, so a guest correlating a
// GPU fence timestamp with a CPU sceKernelReadTsc value saw two disjoint timelines.
extern "C" uint64_t prosper_guest_tsc_ns() { return ns_now(); }
// sceKernelClockGettime / clock_gettime(clockid, struct timespec*). The PS5 inherits FreeBSD's
// clockid numbering (shadPS4 time.h ORBIS_CLOCK_* == FreeBSD sys/time.h CLOCK_*; Kyty Pthread.cpp
// KernelClockGettime agrees on 0=REALTIME, 4=MONOTONIC). Previously the id was IGNORED — every
// clock, including CLOCK_REALTIME, returned uptime-since-process-start, i.e. wall time = Jan 1
// 1970 + uptime (#92). Realtime family -> the anchored wall clock; monotonic/uptime family -> the
// steady clock (FreeBSD's UPTIME clocks ARE its MONOTONIC clocks — both count since boot).
// CONFIDENCE: HIGH on ids {0,4,5,7,8,9,10,11,12,13} (FreeBSD + both references); MED on the
// default branch (cpu-time ids 1/2/14 fall back to the steady clock rather than real cpu time —
// still monotonic, never backwards).
HLE(k_clock_gettime) {                                     // (clockid, struct timespec*)
    if (!a1) return 0x8002000eull;                          // SCE_KERNEL_ERROR_EFAULT (Kyty/shadPS4)
    int64_t sec, nsec;
    switch ((int)a0) {
    case 0: case 9: case 10: {                              // REALTIME / _PRECISE / _FAST
        uint64_t us = wall_now_us();
        sec = (int64_t)(us / 1000000ull); nsec = (int64_t)(us % 1000000ull) * 1000;
        break;
    }
    case 13: {                                              // CLOCK_SECOND: realtime, 1 s resolution
        sec = (int64_t)(wall_now_us() / 1000000ull); nsec = 0;
        break;
    }
    case 4: case 5: case 7: case 8: case 11: case 12:       // MONOTONIC/UPTIME family
    case 15:                                                // PROCTIME (uptime == our process time)
    default: {                                              // incl. cpu-time ids — steady fallback
        uint64_t ns = ns_now();
        sec = (int64_t)(ns / 1000000000ull); nsec = (int64_t)(ns % 1000000000ull);
        break;
    }
    }
    ((int64_t*)P(a1))[0] = sec;                             // tv_sec
    ((int64_t*)P(a1))[1] = nsec;                            // tv_nsec
    return 0;
}
HLE(k_gettimeofday) {                                      // (struct timeval*, tz*)
    if (!a0) return 0;
    uint64_t us = wall_now_us();
    ((int64_t*)P(a0))[0] = (int64_t)(us / 1000000ull);      // tv_sec
    ((int64_t*)P(a0))[1] = (int64_t)(us % 1000000ull);      // tv_usec
    return 0;
}
HLE(k_time) { uint64_t s = wall_now_us() / 1000000ull; if (a0) *(int64_t*)P(a0) = (int64_t)s; return s; }
HLE(k_clock) { return ns_now() / 1000; }   // clock(): CLOCKS_PER_SEC=1e6 -> microseconds
// sceRtcGetCurrentTick(SceRtcTick* tick): a SceRtcTick is a single u64 = microseconds since the Sony RTC
// epoch 0001-01-01 00:00:00 UTC. Built from the anchored wall clock (#92 — was the frozen synthetic base
// 1700000000 ≈ Nov 2023) + the RTC↔unix epoch offset. CONFIDENCE: HIGH — SceRtcTick = bare u64 µs and the
// 62135596800 s offset are the documented Orbis RTC convention (shadPS4 rtc.cpp UNIX_EPOCH_TICKS agrees).
HLE(k_rtc_get_current_tick) {
    if (!a0) return 0;
    *(uint64_t*)P(a0) = wall_now_us() + kRtcUnixEpochOffsetUs;
    return 0;
}
// Fill the 16-byte SceRtcDateTime {u16 year,month,day,hour,minute,second; u32 microsecond} from a
// broken-down tm + microsecond remainder. A real calendar conversion is load-bearing for the UE4
// boot: zeroed output (month=0, day=0) trips UE4's FDateTime "Invalid Date values" assert, whose
// failed-assert handler then calls a null crash-handler pointer.
static void fill_rtc_datetime(void* out, const struct tm& tmv, uint32_t usec) {
    uint16_t* d = (uint16_t*)out;
    d[0] = (uint16_t)(tmv.tm_year + 1900);
    d[1] = (uint16_t)(tmv.tm_mon + 1);
    d[2] = (uint16_t)tmv.tm_mday;
    d[3] = (uint16_t)tmv.tm_hour;
    d[4] = (uint16_t)tmv.tm_min;
    d[5] = (uint16_t)tmv.tm_sec;
    *(uint32_t*)(d + 6) = usec;
}
// sceRtcGetCurrentClockLocalTime(SceRtcDateTime* dt) — the current wall clock in the HOST's local
// timezone (previously gmtime on the frozen synthetic base: "local" was UTC and the date was stuck
// at Nov 2023, #92). Reference: shadPS4 rtc.cpp sceRtcGetCurrentClockLocalTime = current tick +
// the sceKernelGettimezone offset (minuteswest/dst) — i.e. the host tz including DST, which is
// exactly what localtime_r/localtime_s compute. Kyty has no LibRtc to cross-check; single-arg
// signature and host-tz semantics per shadPS4. CONFIDENCE: MED.
HLE(k_rtc_get_clock_localtime) {
    if (!a0) return 0;
    uint64_t us = wall_now_us();
    time_t secs = (time_t)(us / 1000000ull);
    struct tm tmv {};
#ifdef _WIN32
    localtime_s(&tmv, &secs);
#else
    localtime_r(&secs, &tmv);
#endif
    fill_rtc_datetime(P(a0), tmv, (uint32_t)(us % 1000000ull));
    return 0;
}
// sceRtcGetCurrentClock(SceRtcDateTime* dt, int tz_minutes) — the current wall clock shifted by an
// EXPLICIT caller-supplied timezone offset in minutes (tz was previously ignored). Reference:
// shadPS4 rtc.cpp sceRtcGetCurrentClock does tick + sceRtcTickAddMinutes(timeZone). Signed: west
// of UTC is negative. CONFIDENCE: MED (shadPS4 only; Kyty has no LibRtc).
HLE(k_rtc_get_current_clock) {
    if (!a0) return 0;
    int64_t us = (int64_t)wall_now_us() + (int64_t)(int32_t)a1 * 60000000ll;
    if (us < 0) us = 0;                                     // absurd tz on a near-epoch clock: clamp
    time_t secs = (time_t)(us / 1000000ll);
    struct tm tmv {};
#ifdef _WIN32
    gmtime_s(&tmv, &secs);
#else
    gmtime_r(&secs, &tmv);
#endif
    fill_rtc_datetime(P(a0), tmv, (uint32_t)(us % 1000000ll));
    return 0;
}
// sceRtcGetCurrentDateTimeUtc(SceRtcDateTime* dt) — plain UTC.
HLE(k_rtc_get_clock_utc) {
    if (!a0) return 0;
    uint64_t us = wall_now_us();
    time_t secs = (time_t)(us / 1000000ull);
    struct tm tmv {};
#ifdef _WIN32
    gmtime_s(&tmv, &secs);
#else
    gmtime_r(&secs, &tmv);
#endif
    fill_rtc_datetime(P(a0), tmv, (uint32_t)(us % 1000000ull));
    return 0;
}

// sceRtcSetTick(SceRtcDateTime* dt, const SceRtcTick* tick): broken-down UTC datetime from a tick
// (u64 µs since 0001-01-01, the Orbis RTC convention above). Reference: shadPS4 rtc.cpp
// sceRtcSetTick (tick - UNIX_EPOCH_TICKS -> gmtime). Unimplemented-0 left the out struct at the
// caller's zero-init, and UE4's FDateTime(Y:0,M:0,D:0,...) "Invalid Date values" fatal spammed
// thousands of times during the post-shader-map load (issue #115 follow-on wall). glibc gmtime_r
// handles pre-1970 (negative time_t) fine, so ticks below the unix epoch still convert.
// CONFIDENCE: MED-HIGH (shadPS4 reference; struct layout shared with the GetCurrentClock family).
HLE(k_rtc_set_tick) {   // (SceRtcDateTime* dt, const SceRtcTick* tick)
    if (!a0 || !a1) return 0x80250001ull;   // SCE_RTC_ERROR_INVALID_POINTER
    int64_t us = (int64_t)*(const uint64_t*)P(a1) - (int64_t)kRtcUnixEpochOffsetUs;
    time_t secs = (time_t)(us >= 0 ? us / 1000000ll : (us - 999999ll) / 1000000ll);   // floor
    int64_t rem = us - (int64_t)secs * 1000000ll;
    struct tm tmv {};
#ifdef _WIN32
    gmtime_s(&tmv, &secs);
#else
    gmtime_r(&secs, &tmv);
#endif
    fill_rtc_datetime(P(a0), tmv, (uint32_t)rem);
    return 0;
}
// sceRtcGetTick(const SceRtcDateTime* dt, SceRtcTick* tick): the inverse (datetime assumed UTC).
HLE(k_rtc_get_tick) {   // (const SceRtcDateTime* dt, SceRtcTick* tick)
    if (!a0 || !a1) return 0x80250001ull;
    const uint16_t* d = (const uint16_t*)P(a0);
    struct tm tmv {};
    tmv.tm_year = (int)d[0] - 1900; tmv.tm_mon = (int)d[1] - 1; tmv.tm_mday = (int)d[2];
    tmv.tm_hour = (int)d[3]; tmv.tm_min = (int)d[4]; tmv.tm_sec = (int)d[5];
#ifdef _WIN32
    int64_t secs = _mkgmtime64(&tmv);
#else
    int64_t secs = (int64_t)timegm(&tmv);
#endif
    *(uint64_t*)P(a1) = (uint64_t)(secs * 1000000ll + (int64_t)*(const uint32_t*)(d + 6)
                                   + (int64_t)kRtcUnixEpochOffsetUs);
    return 0;
}

// Real sleeps so timed wait loops actually yield the CPU (and advance real time).
//
// NOT nanosleep, and on Windows that is the whole point (#3013). MinGW's nanosleep is winpthreads',
// whose timed waits resolve on the winpthreads master tick REGARDLESS of timeBeginPeriod -- measured
// on this toolchain at 15.67 ms mean for a 5.33 ms request, unchanged (15.59 ms) with the process
// timer resolution raised to 1 ms. A guest audio mixer pacing 256-frame grains at 48 kHz asks for
// 5.33 ms and got 15.6 ms: it delivered one grain per tick instead of per grain, ~2.9x too slowly,
// and every title underran continuously on Windows while Linux was clean.
//
// sleep_until_steady_ns uses CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, which is independent of the
// process timer period: 5.64 ms mean / 5.99 ms worst for the same request on the same box. It also
// takes an ABSOLUTE deadline, so the per-call overhead cannot compound across a pacing loop the way
// a relative sleep's does.
//
// The deadline is taken as early as the body allows, because it is the guest's schedule and any work
// done first is silently added to the interval it asked for. Not an absolute: the census scope's
// constructor and, in k_nanosleep, the guest read and range guard necessarily precede it. Those are
// a few hundred nanoseconds against a millisecond-scale request, which is why the ordering is worth
// stating as an intent rather than asserting as an invariant.
static inline void guest_sleep_ns(uint64_t ns) {
    // PROSPER_GUEST_SLEEP_LEGACY=1 restores the pre-#3013 nanosleep path. It exists ONLY so the A/B
    // that established the fix stays reproducible -- the same reason PROSPER_UD_TAIL_ALIGN survives
    // (CLAUDE.md). Do not set it to fix anything.
    //
    // What the A/B does and does NOT show, because an earlier version of this comment cited the
    // wrong half: audio DELIVERY RATE does not discriminate. The Messenger measures ~100% of the
    // 384,000 B/s that f32/2ch/48 kHz needs in BOTH arms (385,024-389,120 B/s legacy), which is
    // exactly why a delivery-rate check cannot find this defect -- quoting the 100.8% as evidence
    // for the fix, as that comment did, points a reader at a number that separates nothing. What the
    // lever does separate is sleep ACCURACY: requested 18.17 -> actual 23.74 ms (x1.31) legacy
    // against 15.41 -> 15.69 (x1.02) fixed, on the same route.
    static const bool legacy = getenv("PROSPER_GUEST_SLEEP_LEGACY") != nullptr;
    if (legacy) {
        struct timespec ts{ (time_t)(ns / 1000000000ull), (long)(ns % 1000000000ull) };
        nanosleep(&ts, nullptr);
        return;
    }
    const uint64_t deadline = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch()).count() + ns;
    prosper::host::sleep_until_steady_ns(deadline);
}

// Saturating, for the same reason k_nanosleep saturates: the argument is guest-controlled and a wrap
// turns a long sleep into a short one. Having one of these three saturate and the others wrap would
// be an inconsistency the next reader has to re-derive, so they all do.
//
// This comment used to add that saturating was "benign in consequence here -- a wrapped deadline
// lands in the past and precise_sleep returns at once". That was TRUE and was not benign: it meant
// the clamp handed its own failure mode straight back, since an instant return for a 584-year
// request is exactly the short sleep being prevented. #3038 measured it (0 ms for both INT64_MAX+1
// and UINT64_MAX) and fixed it in precise_sleep's POSIX backend, which now clamps the deadline to
// the largest its signed chrono rep can hold rather than wrapping it. A saturated interval now
// blocks, on Linux as it already did on Windows.
static inline uint64_t guest_ns_from(uint64_t value, uint64_t ns_per_unit) {
    const uint64_t kNsMax = ~0ull;
    return value > kNsMax / ns_per_unit ? kNsMax : value * ns_per_unit;
}

HLE(k_usleep)   { const uint64_t ns = guest_ns_from(a0, 1000ull);
                  hle::WaitCensusScope c(hle::WaitKind::Usleep, ns); guest_sleep_ns(ns); return 0; }

// POSIX sleep() returns the number of seconds LEFT unslept (0 on full completion), not the input.
// Returning the input breaks the canonical resume idiom `while ((left = sleep(left))) ;` into an
// infinite busy-sleep. We always sleep the full duration, so return 0 (Kyty KernelSleep returns OK/0).
HLE(k_sleep_s)  { const uint64_t ns = guest_ns_from(a0, 1000000000ull);
                  hle::WaitCensusScope c(hle::WaitKind::SleepSeconds, ns); guest_sleep_ns(ns); return 0; }
// The remainder out-parameter is written ZERO rather than left untouched, on BOTH exits: a served
// request sleeps in full so nothing remains, and a refused one slept nothing but must still not hand
// back whatever was in that memory. Either way a guest reading an uninitialised remainder could
// resume a wait it should not. (This paragraph previously said "we always sleep the full duration",
// which the refusal path made false.)
//
// Both fields are indexed as int64 rather than reached through a HOST struct timespec, because the
// two layouts differ on the platform this targets: the guest is FreeBSD x86-64, where tv_nsec is
// 64-bit, while MinGW-w64 declares long tv_nsec (sys/types.h) -- 32-bit on Windows x64. A
// host-struct write therefore covers 12 of the 16 guest bytes and leaves the HIGH half of tv_nsec
// holding whatever was there, so the remainder a guest reads back is not zero and it may resume a
// wait already served. The READ survived the same cast only by little-endian accident, since every
// legal nanosecond value fits in the low half. Line 278 already indexes explicitly for this reason.
//
// A negative tv_sec, or a negative or out-of-range tv_nsec, is REFUSED -- the body returns 0 without sleeping. The comment
// on the guard itself says why that is the behaviour to preserve. (An earlier revision of this block
// described the OPPOSITE, carrying the value into the total. That text survived the commit that added
// the guard and read as a rationale for removing it, which is how a comment gets a safety check
// deleted.)
HLE(k_nanosleep){
    if (!a0) return 0;
    const int64_t* req = (const int64_t*)P(a0);
    // The range rule and the saturating conversion both live in guest_interval_from_timespec
    // (precise_sleep.hpp) since #3038, where select()/pselect() call the same function rather than
    // carrying a second, narrower copy of this arithmetic. The behaviour is unchanged; what moved
    // is where it can be tested from.
    //
    // A malformed request returns WITHOUT sleeping, which is what this entry point already did: the
    // old body handed the guest struct straight to the host nanosleep, and POSIX makes a tv_nsec
    // outside [0, 1e9) EINVAL -- so the host refused instantly and the HLE returned 0 having slept
    // nothing. Preserving that matters in BOTH directions. An earlier version of this fix carried an
    // out-of-range tv_nsec into the total, which turns a garbage 0x7fff... into a near-infinite
    // sleep: a hang where there used to be an immediate return. Returning an error instead would be
    // the opposite new failure mode, for guests that currently see success. Same range rule as
    // hle_kernel.cpp:1313, which validates the guest timespec it is handed.
    const prosper::host::GuestInterval want =
        prosper::host::guest_interval_from_timespec(req[0], req[1]);
    if (!want.valid) {
        // Zeroed rather than left untouched (the host would have left it) so a guest reading the
        // remainder after a refused request cannot resume on uninitialised memory.
        if (a1) { int64_t* rem = (int64_t*)P(a1); rem[0] = 0; rem[1] = 0; }
        return 0;
    }
    { hle::WaitCensusScope c(hle::WaitKind::Nanosleep, want.ns); guest_sleep_ns(want.ns); }
    if (a1) { int64_t* rem = (int64_t*)P(a1); rem[0] = 0; rem[1] = 0; }   // slept in full
    return 0;
}



// --- select / pselect: the PURE-SLEEP shape only (#1660) --------------------------------------
//
// `select(0, NULL, NULL, NULL, &tv)` — nfds <= 0 with all three descriptor sets NULL — is the
// canonical portable sleep idiom. No descriptor is involved, so it needs no socket backing at all:
// wait out the timeout and return 0. **0 is the CORRECT return here** ("timed out, nothing became
// ready"); returning it *immediately* is the defect. What has to be honoured is the timeout.
//
// Evidence: on PPSA19244 (The Oregon Trail) the Gameloft SDK's `OLD_HTTPClient` service thread uses
// exactly this call as its 3-second inter-pass sleep — arguments captured live at the import stub
// (`PROSPER_STUBDUMP` -> 0x600000000+off -> breakpoint at stub entry), six consecutive hits, all
// `select(nfds=0, NULL, NULL, NULL, {tv_sec=3, tv_usec=0})`. Falling through to the generic
// unimplemented stub returned 0 in ~111 ns, collapsing a 3 s sleep into a no-op: the loop ran
// **9.0 million times per second — 1,300,185,430 calls in 144 s**, saturating a core through the
// dispatch path. That is both a performance cost and a measurement hazard, because the burn
// attributes itself to prosper's dispatch rather than to the guest and distorts any CPU profile of
// a title in this state. `select`-as-sleep is a portable idiom, so this is not title-specific.
//
// A query with an actual descriptor set is NOT implemented and deliberately stays fail-visible:
// prosper has no socket backing, so replying "0 = nothing ready" would be precisely the
// success-shaped answer-we-cannot-honour that caused this bug. Log it loudly and fail.
// CONFIDENCE: HIGH on the sleep shape (arguments observed live); the descriptor shape is
// intentionally unimplemented, not unknown.
//
// #3038: honouring the timeout means honouring the length of it, on both platforms. The timeout is
// now converted in 64 bits and waited out on an absolute steady-clock deadline like the rest of the
// guest sleep family (#3022), not on a host `struct timespec` handed to nanosleep. The Oregon Trail
// evidence below is a 3 s sleep, where a ~15.6 ms tick is noise; an inter-pass sleep of a few
// milliseconds through the same call is where it is not.
namespace {
// True for the pure-sleep shape: no descriptor may be examined.
bool select_is_pure_sleep(uint64_t nfds, uint64_t readfds, uint64_t writefds, uint64_t exceptfds) {
    return (int64_t)nfds <= 0 && readfds == 0 && writefds == 0 && exceptfds == 0;
}

// Sleep the FULL interval the guest asked for, on an ABSOLUTE deadline.
//
// This was a raw `nanosleep` retry loop until #3038, and the loop's own purpose survives its
// removal. It existed because a bare nanosleep returns early on a signal and prosper delivers
// signals of its own (the SIGSEGV fault handler), so a short sleep would reintroduce the spin #1660
// removed. An absolute deadline subsumes that: on POSIX guest_sleep_ns ends in libstdc++'s own
// EINTR-restarting nanosleep loop, and on Windows sleep_until_steady_ns re-reads the clock after
// every wait and goes again. What the loop could NOT do is get off the winpthreads ~15.6 ms master
// tick that #3013 measured and #3022 routed the other three sleep entry points off -- an inter-pass
// sleep of a few milliseconds through select() was served in ~15.6 ms, and this path was the last
// one still on it.
//
// One caveat, because it is a real difference rather than a wash: PROSPER_GUEST_SLEEP_LEGACY=1
// restores guest_sleep_ns's single pre-#3013 `nanosleep`, which does NOT restart on EINTR -- so
// under that lever a signal can truncate this wait where the loop above resumed it. The lever
// exists only to keep #3013's A/B reproducible and is not a supported runtime configuration
// (CLAUDE.md's PROSPER_UD_TAIL_ALIGN precedent).
void sleep_full_ns(uint64_t ns) { guest_sleep_ns(ns); }

// A NULL timeout in the pure-sleep shape means "block forever". Sleep in bounded chunks rather than
// one unbounded call so the process still responds to termination. Not censused: the requested
// interval here is prosper's own chunk size, not anything the guest asked for, so counting it would
// report a 1.00 ratio for a wait that has no guest-supplied duration at all.
void sleep_forever() { for (;;) sleep_full_ns(1000000000ull); }

uint64_t select_unsupported(const char* fn, uint64_t nfds,
                            uint64_t readfds, uint64_t writefds, uint64_t exceptfds) {
    static std::atomic<int> logged{0};
    const int n = logged.fetch_add(1);
    // Distinguish the two ways a call can miss the implemented shape, because they mean very
    // different things. A real descriptor set genuinely cannot be answered without socket backing.
    // `nfds > 0` with every set NULL examines no descriptor either, so it is *probably* also a
    // sleep — but no title has been observed doing it, so it is refused rather than assumed, and
    // called out here so the first title that does hit it is immediately visible instead of
    // silently inheriting a widened contract. See #1660 before extending the shape.
    const bool no_sets = (readfds == 0 && writefds == 0 && exceptfds == 0);
    if (n < 8)
        fprintf(stderr, "[posix] %s: %s is UNIMPLEMENTED (no socket backing) "
                        "nfds=%lld readfds=%#llx writefds=%#llx exceptfds=%#llx -> -1 ENOSYS "
                        "(#%d; only the nfds<=0 + all-NULL pure-sleep shape is supported, see #1660)%s\n",
                fn, no_sets ? "descriptor-free call with nfds>0" : "descriptor-set query",
                (long long)(int64_t)nfds, (unsigned long long)readfds,
                (unsigned long long)writefds, (unsigned long long)exceptfds, n,
                no_sets ? "  <- examines no descriptor; likely a sleep, but unobserved: report it"
                        : "");
    // Follows the existing libScePosix wrapper convention in hle_file.cpp: set the errno that
    // __error()/h_errno_location hands back and return -1. The number must be FREEBSD's, because
    // that is what the guest compares against: host ENOSYS is 38 on Linux and 40 on MinGW, while
    // the guest tests for 78, so an untranslated write sends it down a generic-error path instead
    // of the fallback branch its author wrote (#2296). Nothing re-converts this slot -- k_select
    // and k_pselect return this value straight through -- so publishing FreeBSD's here is the
    // whole fix for this call. (This comment used to cite #1612; that issue is closed and was
    // about the `0x80020000|errno` RETURN encoding, a different slot with the same trap.)
    hle::set_guest_errno(ENOSYS);
    return (uint64_t)(int64_t)-1;
}
} // namespace

// select(int nfds, fd_set* r, fd_set* w, fd_set* e, struct timeval* timeout)
HLE(k_select) {
    if (!select_is_pure_sleep(a0, a1, a2, a3)) return select_unsupported("select", a0, a1, a2, a3);
    if (!a4) sleep_forever();
    else {
        // Indexed as two int64s and converted in 64 bits, never assembled into a HOST timespec:
        // `(long)(tv[1] * 1000)` narrowed the product to 32 bits on Windows x64 and multiplied
        // before it narrowed, so an out-of-range tv_usec could land back INSIDE the legal range as
        // a far shorter sleep (#3038). An out-of-range field is now REFUSED -- no sleep, and still
        // return 0 -- which is what this call already did on a host wide enough to see the value,
        // because the host nanosleep saw the out-of-range tv_nsec and returned EINVAL instantly.
        // Returning -1/EINVAL instead would be the more POSIX answer and is deliberately not done
        // here: it is a guest-visible contract change for titles that currently see success, with
        // no way to verify it from this side.
        // struct timeval { int64 tv_sec; int64 tv_usec; }
        const int64_t* tv = (const int64_t*)P(a4);
        const prosper::host::GuestInterval want =
            prosper::host::guest_interval_from_timeval(tv[0], tv[1]);
        if (want.valid) {
            hle::WaitCensusScope c(hle::WaitKind::Select, want.ns);
            sleep_full_ns(want.ns);
        }
    }
    return 0;   // timed out with nothing ready — the correct answer for a descriptor-free wait
}

// pselect(int nfds, fd_set* r, fd_set* w, fd_set* e, const struct timespec* timeout,
//         const sigset_t* sigmask). Same rule; the timeout is a timespec, not a timeval.
HLE(k_pselect) {
    if (!select_is_pure_sleep(a0, a1, a2, a3)) return select_unsupported("pselect", a0, a1, a2, a3);
    if (!a4) sleep_forever();
    else {
        // struct timespec { int64 tv_sec; int64 tv_nsec; }
        const int64_t* ts = (const int64_t*)P(a4);
        const prosper::host::GuestInterval want =
            prosper::host::guest_interval_from_timespec(ts[0], ts[1]);
        if (want.valid) {
            hle::WaitCensusScope c(hle::WaitKind::Select, want.ns);
            sleep_full_ns(want.ns);
        }
    }
    return 0;
}
// Guest-visible process id. Keep it stable across host runs and distinct from the kernel's
// special pid 0; shadPS4 uses the same 0xBAD1 compatibility pid. Returning generic-stub success
// (zero) violates POSIX and can collapse per-process paths/ownership keys. CONFIDENCE: HIGH.
HLE(k_getpid)   { return 0xbad1; }

// sceKernelGettimezone(struct timezone* tz) = { int tz_minuteswest; int tz_dsttime }. Was MISSING -> the
// generic stub left the out-struct uninitialized (the #82/#190 uninit-out class). We present a UTC clock
// (the RTC path uses timegm), so report {0, 0} deterministically.
HLE(k_gettimezone) { if (a0) { int* tz = (int*)P(a0); tz[0] = 0; tz[1] = 0; } return 0; }
// sceKernelClockGetres(clockid, struct timespec* res): our time source is nanosecond-resolution. Was
// MISSING -> left *res uninitialized. Report 1 ns for every clock.
HLE(k_clock_getres) { if (a1) { int64_t* r = (int64_t*)P(a1); r[0] = 0; r[1] = 1; } return 0; }

// --- assorted libkernel stubs ---
HLE(k_ok)              { return 0; }                       // generic success no-op

// --- libSceSysmodule: per-process load state (#2002) ---------------------------------------------
//
// sceSysmoduleIsLoaded is a *state query*, and it used to share k_ok with Load/UnloadModule: every
// module id answered 0 == "loaded". That is not a harmlessly optimistic answer. The idiom titles
// actually write is
//     if (sceSysmoduleIsLoaded(id) == SCE_SYSMODULE_ERROR_UNLOADED) { LoadModule(id); Initialize(); }
// so a wrong "loaded" makes the guest conclude somebody already did the setup and **skip its own
// initialization**. The defect then surfaces as missing content minutes later with nothing tying it
// back to the query. Grand Theft Auto V (PPSA04263) does exactly this at eboot+0x197a22a and again
// at eboot+0x4b978f (`cmp eax,0x805a1001`), losing sceAppContentInitialize and its
// userDefinedParam1 read; the guest's own compare is the primary evidence for the errno value.
// Measured statically over the 44 local dumps with tools/re/nid_gate_scan.py: 39 import the NID, and
// at all 441 call sites the returned value is READ before it dies — 237 compare it against this
// exact errno, 204 test it for zero, none discard it. Read that bound precisely: the scan classifies
// the COMPARE, not the branch target, so it says which titles *can* change behaviour under a
// different answer, not which ones do, and "441 sites" is a lower bound on reach (it follows one
// stub level and one call deep). The `test eax,eax` half matters as much as the errno half — it is
// the plain defensive form of the same gate. It was recorded here as one that "never mentions
// 0x805A1001"; that is FALSE for 70 of those sites, which name the constant inside their error arm
// where the scan of the day could not see it. The claim is quoted rather than deleted so this stays
// a dated record of what was believed, and the re-measure directly below is what corrects it.
// RE-MEASURED 2026-08-17 (PR #2637), and it moved in the direction that strengthens this: the scan
// used to stop at the first branch, so a site that tested generically and compared the errno inside
// its ERROR ARM was counted in the `test eax,eax` half. Reading the arms, the same 445 sites across
// the same 66 modules split const=307 / nonzero=78 / gate-open=47 / undecodable=13, against
// const=237 / nonzero=208 with `--no-follow-arms`. So **70 more sites name 0x805A1001 than this
// paragraph could see**, and the 47 `gate-open` + 13 `undecodable` are honestly unresolved rather
// than counted as insensitive.
// The 441 -> 445 drift against the older reading is resolved, and it is apparatus rather than
// corpus: PPSA08804-app0 carries a second copy of its eboot at `_DUPLEX_/Original eboot/eboot.bin`
// (a release-group leftover, and NOT byte-identical to the shipped one -- a different build). The
// scanner walks it as its own module, and it contributes exactly 4 sites and 1 module: 445 - 4 =
// 441 and 66 - 1 = 65, reproducing the historical figures. Any sweep quoting a site total over the
// local dumps double-counts that one title's eboot.
//
// What prosper can honestly answer is which ids the guest asked it to load. The sysmodule id space
// is the set of *optional* modules an application must request; the always-resident libraries
// (libkernel, libc, libSceGnmDriver, libSceVideoOut) are bound through DT_NEEDED and never travel
// through this API — so per-process "did this process load it?" is the contract, and prosper's own
// load history is exactly the state that answers it. Supporting evidence that Sony models
// residency-without-a-request as a DIFFERENT question: libSceSysmodule exports a separate
// sceSysmoduleIsCameraPreloaded (3.20 export list), which would be redundant if IsLoaded already
// reported system-preloaded modules.
//
// Answering UNLOADED cannot cost a capability here: prosper's HLE surface for a module is resident
// whether or not the guest ever calls LoadModule, so the worst case is that the guest performs the
// load+initialize sequence it performs on hardware, and the load succeeds. The opposite answer is
// the only one that can silently remove behaviour. Sony's loader reference-counts the per-process
// load; prosper does not model the count, because an unload that drops a nested load only produces
// an extra UNLOADED, which prompts a redundant (succeeding) load — the error in the safe direction.
// CONFIDENCE: HIGH that answering "loaded" for a module this process never loaded is wrong (the
// guest's own compare proves the contract). CONFIDENCE: MED that no sysmodule id is legitimately
// resident before the app asks — that half rests on the API's shape, not on primary evidence — and
// MED for the unmodelled ref-count nesting.
//
// The id is masked to 16 bits because that is genuinely its width in Sony's prototype
// (`sceSysmoduleLoadModule(uint16_t)`), so a caller may leave anything in the upper half; both the
// load and the query must therefore agree on the same truncation, or they name different entries
// for one module. Note which direction the risk runs, because it is the opposite of this change's
// general direction: everywhere else an over-eager UNLOADED is the safe error, but truncation is
// the ONE place here that can manufacture a false *loaded* — two distinct ids congruent mod 2^16
// would share an entry. The tripwire is therefore an observed id >= 0x10000; the corpus's widest is
// 0x130, so nothing is close, and if one ever appears this mask is the first thing to revisit.
//
// The eight *Internal entry points were the same defect on a parallel path, and are fixed in #2128:
// unregistered, they fell to prosper_on_unimpl, which returns 0 — SUCCESS — without recording
// anything, so a title that loaded through one of them and later called IsLoaded was told UNLOADED
// about a load prosper itself had just reported as succeeding. Still no dump in the local corpus
// imports any of them (re-measured on five, with the plain LoadModule NID as a positive control),
// so this closes a class rather than fixing an observed symptom.
constexpr uint64_t k_sysmodule_error_unloaded = 0x805A1001;   // SCE_SYSMODULE_ERROR_UNLOADED

namespace {
// Heap-backed, not a namespace-scope std::mutex: on macOS the latter is constant-initialized
// non-zero and lands in __DATA, where the #707 corruptor zeroes its pthread signature during
// Blasphemous 2's asset load — and Blasphemous 2 is one of the titles that reaches these handlers.
// See heap_mutex.hpp. Call sites must use the deduced `std::lock_guard lk(...)` form.
PROSPER_HEAP_MUTEX(g_sysmodule_mx);
std::unordered_map<uint16_t, bool> g_sysmodule_state;   // id -> loaded; absent == never asked for

// One line per id the FIRST time it answers UNLOADED, so a title that reacts badly to an honest
// answer is diagnosable instead of silent (the charter's fail-visible rule). Deduped per id — the
// line says nothing about how often the id is queried.
void sysmodule_report_unloaded(uint16_t id) {
    static std::unordered_map<uint16_t, bool> seen;      // guarded by g_sysmodule_mx
    if (seen.size() >= 64 || !seen.emplace(id, true).second) return;
    fprintf(stderr, "[sysmodule] IsLoaded(0x%x) -> UNLOADED: this process has not loaded it "
                    "(first query for this id; repeats are not logged)\n", (unsigned)id);
}
} // namespace

// sceSysmoduleLoadModule(uint16_t id). prosper's PRX and HLE for the module are already resident,
// so the load itself is a no-op that genuinely succeeds — the only new behaviour is that the load
// is now recorded, so the state query stops contradicting the loader's own history.
HLE(k_sysmodule_load) {
    const uint16_t id = (uint16_t)a0;
    std::lock_guard lk(g_sysmodule_mx);
    g_sysmodule_state[id] = true;
    return 0;
}

// sceSysmoduleUnloadModule(uint16_t id)
HLE(k_sysmodule_unload) {
    const uint16_t id = (uint16_t)a0;
    std::lock_guard lk(g_sysmodule_mx);
    g_sysmodule_state[id] = false;
    return 0;
}

// sceSysmoduleLoadModuleByNameInternal (CU8m+Qs+HN4) / ...UnloadModuleByNameInternal (vpTHmA6Knvg).
//
// These take a NAME, not an id, so they cannot write the id-keyed map above — and prosper has no
// name→id table to build one from. Registered anyway, to a handler that REFUSES, because leaving
// them unregistered is not neutral: prosper_on_unimpl returns 0, which for this contract is success,
// and the caller then believes a module is loaded that nothing recorded (#2128).
//
// An error is the honest answer of the two available. A refused load is a path the guest already
// has; a success it cannot see the consequences of is not.
HLE(k_sysmodule_by_name_unsupported) {
    static std::atomic<unsigned> seen{0};
    if (seen.fetch_add(1) < 8)
        fprintf(stderr, "[sysmodule] LoadModuleByName/UnloadModuleByName refused: prosper's load "
                        "state is keyed by module ID and there is no name->id table, so this call "
                        "cannot be recorded. Reporting failure rather than an unrecorded "
                        "success (#2128)\n");
    return k_sysmodule_error_unloaded;
}

// sceSysmoduleGetModuleHandleInternal (D8cuU4d72xM). Never 0: for THIS contract 0 is a valid handle
// shape and the caller will dereference it, which is strictly worse than the missing implementation
// it stands in for. prosper has no per-sysmodule-id handles to hand out at all, so the answer is the
// same whether or not the id was loaded — and saying so is the point.
HLE(k_sysmodule_get_handle_internal) {
    static std::atomic<unsigned> seen{0};
    if (seen.fetch_add(1) < 8)
        fprintf(stderr, "[sysmodule] GetModuleHandleInternal(0x%x) refused: prosper keeps loaded "
                        "STATE per id but no module handle, and returning 0 here would hand the "
                        "guest a dereferenceable handle it never got (#2128)\n", (unsigned)a0);
    return k_sysmodule_error_unloaded;
}

// sceSysmoduleIsLoaded(uint16_t id) -> SCE_OK when this process loaded it, else UNLOADED.
HLE(k_sysmodule_is_loaded) {
    const uint16_t id = (uint16_t)a0;
    std::lock_guard lk(g_sysmodule_mx);
    const auto it = g_sysmodule_state.find(id);
    if (it != g_sysmodule_state.end() && it->second) return 0;
    sysmodule_report_unloaded(id);
    return k_sysmodule_error_unloaded;
}
// sceKernelLoadStartModule: a PRX already in the linked set resolves to its REAL handle, and dlsym
// then consults that module's own exports first (#147). A path NOT in the linked set used to get a
// fake monotonically-increasing success handle while loading nothing (#146) — the guest believed
// the load succeeded and called exports that resolved to ESRCH fallbacks — and then, once that was
// removed, an honest ENOENT for a capability prosper did not have. Since #639 it is a real load:
// only a genuinely absent or unloadable file is an error. See host/runtime_module_load.hpp.
// sceKernelLoadStartModule(const char* name, size_t args, const void* argp, uint32_t flags,
//                          const SceKernelLoadModuleOpt* opt, int* pRes)
static uint64_t load_start_mod_run(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a5,
                                   uint64_t guest_fs) {
    const char* path = a0 ? (const char*)P(a0) : nullptr;
    // A pre-linked module (the fixed set plus the Media/Plugins auto-link, #1609) already has a
    // handle; hand back the same one rather than mapping a second copy. This is also what makes a
    // repeat load of a runtime-loaded module idempotent — it is registered here too, by basename,
    // which is the PS5's own module identity.
    if (uint64_t h = module_handle_for_path(path)) {
        if (a5) *(int32_t*)P(a5) = 0;
        return h;
    }
    // Not pre-linked: load it for real (#639). Only an absent/unloadable file is an error now.
    uint64_t handle = 0; int32_t res = 0;
    const uint64_t rc = runtime_load_start_module(path, a1, a2, guest_fs, &res, &handle);
    if (rc == 0) {
        if (a5) *(int32_t*)P(a5) = res;
        return handle;
    }
    if (getenv("PROSPER_MODLOG"))
        fprintf(stderr, "[loadmod] '%s' could not be loaded -> 0x%llx\n",
                path ? path : "(null)", (unsigned long long)rc);
    return rc;
}
#ifndef _WIN32
// The module's own module_start / init_array is GUEST code. Under the guest-%fs gate this handler
// runs on the host %fs (the import stub swapped), so recover the caller's guest thread pointer from
// the import-stub frame and start the module on it — the same entry-rsp trampoline the AvPlayer and
// IME callbacks use (#1286). callback_guest_fs_from_entry_stack returns 0 for a non-guest frame, so
// a plain tail-jump boot is an unchanged direct call.
extern "C" uint64_t k_load_start_mod_c(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                       uint64_t a4, uint64_t a5, uint64_t entry_rsp) {
    (void)a3; (void)a4;
    return load_start_mod_run(a0, a1, a2, a5, callback_guest_fs_from_entry_stack(entry_rsp));
}
PROSPER_ASM_TRAMPOLINE(k_load_start_mod_entry, k_load_start_mod_c)
extern "C" void k_load_start_mod_entry();
#else
HLE(k_load_start_mod) {   // Windows/MinGW: no import-boundary %fs swap -> guest_fs = 0
    (void)a3; (void)a4;
    return load_start_mod_run(a0, a1, a2, a5, 0);
}
#endif

// _exit(status): terminate the process. Previously an unimplemented stub RETURNED 0, so libc's
// exit path fell through into its deliberate ud2 (SIGILL) — terminate for real, loudly.
HLE(k_exit) {
    fprintf(stderr, "[prosper] guest _exit(%d) -- terminating\n", (int)a0);
    host::guest_dmem_write_trace_report();
    fflush(nullptr);
    _Exit((int)a0);
}
// sceKernelDebugRaiseExceptionOnReleaseMode(code, arg) — the guest's FATAL-error raise (UE4 calls
// it from failed check()s in shipping builds). Report and terminate rather than "return 0" and
// let the guest run on in an undefined state.
HLE(k_debug_raise_release) {
    fprintf(stderr, "[prosper] guest sceKernelDebugRaiseExceptionOnReleaseMode(code=0x%llx, arg=0x%llx) -- terminating\n",
            (unsigned long long)a0, (unsigned long long)a1);
    host::guest_dmem_write_trace_report();
    fflush(nullptr);
    _Exit(0x66);
}
// Console pthread IDs are small, process-local identities, not native kernel TIDs.  Exposing Linux
// gettid/GetCurrentThreadId happened to satisfy mutex ownership checks, but it also leaked a different
// identity domain into guest code which stores the value in its TCB and uses it as a Havok context-map
// key.  Allocate a monotonically increasing guest ID lazily on the host thread's first guest call.  HLE
// handlers run with the host TLS base restored, so host thread_local storage is safe even while the
// surrounding program uses a guest FS base.  IDs are deliberately never recycled during the process.
namespace {
std::atomic<uint32_t> g_next_guest_thread_id{1};

uint32_t current_guest_thread_id() {
    static thread_local const uint32_t id =
        g_next_guest_thread_id.fetch_add(1, std::memory_order_relaxed);
    return id;
}
} // namespace

HLE(k_getthreadid) { return current_guest_thread_id(); } // scePthreadGetthreadid
// sceKernelAprResolveFilepathsToIdsAndFileSizes — PS5 APR (async page read) IO path. Signature
// unconfirmed; a garbage-out "success" poisons the engine's file table, so fail cleanly and let
// the engine take its non-APR file path. CONFIDENCE: LOW on ABI, MED that failing is safer.
HLE(k_apr_unavailable) { return 0x80020016ull; }   // EINVAL
HLE(k_uuid_create) {                                       // fill 16 non-zero bytes
    if (a0) { uint8_t* u = (uint8_t*)P(a0); uint64_t t = ns_now(); for (int i = 0; i < 16; i++) u[i] = (uint8_t)(t >> (i * 4)) ^ (0xA5 + i); }
    return 0;
}

// --- C11 threads (used by MSVC STL std::mutex/std::condition_variable) ---
HLE(m_mtx_init)   { if (a0) { auto* m = (pthread_mutex_t*)calloc(1, sizeof(pthread_mutex_t)); pthread_mutexattr_t at; pthread_mutexattr_init(&at); pthread_mutexattr_settype(&at, PTHREAD_MUTEX_RECURSIVE); pthread_mutex_init(m, &at); pthread_mutexattr_destroy(&at); *(void**)P(a0) = m; } return 0; }
// #2596: every one of these used a PRIVATE `if (a0 && *(void**)P(a0))` guard, and it was wrong in
// TWO different ways depending on which sentinel the slot held -- a distinction worth stating,
// because the first revision of this comment claimed the whole class was merely "skipped" and a
// reviewer falsified that in one grep. The guard tests the slot's VALUE for non-zero, while
// `pt_static_sentinel` (hle_kernel.cpp) treats EVERYTHING below 0x1000 as a static initialiser:
//
//   * NULL (PTHREAD_MUTEX_INITIALIZER) -- the guard is false, so the operation was SKIPPED
//     ENTIRELY. `_Mtx_lock` reported success without taking the lock, which is precisely the bdwgc
//     GC_allocate_ml static-mutex shape that was the ROOT CAUSE of the level-1 heap corruption
//     (#793), reached through the other spelling; `_Cnd_wait` reported success for a wait that
//     never happened.
//   * ANY OTHER SENTINEL -- 1 (PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP) and the destroyed poison
//     kPtDestroyed (0xDEA) -- the guard is TRUE, so the value was DEREFERENCED AS AN OBJECT
//     POINTER: `interruptible_mutex_lock((pthread_mutex_t*)0xDEA)`. Not a silent skip at all, a
//     segfault. Routing through the resolver fixes a use-after-destroy crash on this spelling as
//     well as the missing operation.
//
// The guard is gone. #2596 replaced it with guest_mutex_from_slot / guest_cond_from_slot, which ARE
// ensure_mutex / ensure_cond -- the same resolution scePthreadMutexLock and scePthreadCondWait use.
// (#2619/#2623 then went further and took the whole libkernel BODY, so no handler below names those
// two resolvers any more; the resolution they describe is still exactly what happens, one level in.)
// That is where the guest itself answers the question: its C11 wrappers pass the slot pointer
// STRAIGHT THROUGH to those entry points and inspect nothing (_Cnd_wait @0x5670 and _Mtx_lock
// @0x5e80 in the shipped libc.prx, both re-derived through their JMPREL slots for #2596 -- see
// pthread_slot.hpp), so a private guard here answers a different question from the one libkernel
// answers. A null slot ADDRESS, and a slot holding the destroyed sentinel, still resolve to nullptr
// and still skip -- there genuinely is no object.
//
// The family moves TOGETHER on purpose. Fixing only the wait would leave `_Mtx_lock` no-opping on a
// static mutex while `_Cnd_wait` self-initialised and waited on it, i.e. a wait on a mutex nothing
// had locked -- one spelling of the #1873 shape, where the answer depends on which entry point the
// guest happened to use.
//
// SHARING THE RESOLUTION WAS ONLY HALF (#2619, #2623). Each handler below still carried a PRIVATE
// BODY around the shared resolver, and four pieces of per-object bookkeeping the libkernel bodies
// keep did not travel with it: the destroy pair never claimed or poisoned the slot (#2619, so a
// double `_Mtx_destroy` was a double free and a slot already destroyed through the Sony spelling had
// its 0xDEA poison dereferenced as a pointer); the C11 signal/broadcast never bumped the
// missed-wakeup generation; the C11 wait was invisible to the cond-destroy busy check (#2168); and
// the C11 lock/unlock bypassed the Windows guest-mutex ownership map. Each is the same #1873 shape
// again, one level down. So every handler below is now the libkernel BODY -- `pthread_slot.hpp`'s
// whole-operation family, which the scePthread* handlers call too -- and there is no private body
// left to drift.
//
// DESTROY IS IN THE LIST NOW, and #2596's note that it is not is superseded rather than contradicted:
// what that note ruled out was retiring an object the guest never initialised, and
// `guest_mutex_destroy_slot` does not do that either. `pt_claim_slot` returns nullptr for every
// sub-page value, so an untouched static slot is still a no-op.
//
// WHAT EACH HANDLER RETURNS is the one thing that stays per-spelling, and it is READ OUT of the
// guest's own wrappers rather than chosen -- see the table in pthread_slot.hpp. `_Mtx_unlock`,
// `_Cnd_signal`, `_Cnd_broadcast` and `_Cnd_wait` are 13 bytes ending `xor eax,eax; pop rbp; ret`,
// so 0 is their CONTRACT. The destroy pair is 11 bytes and never touches eax after its call, which
// does NOT establish that it forwards libkernel's answer -- eleven bytes with no `xor eax,eax` is
// precisely what a VOID wrapper compiles to, `_Thrd_yield` is the byte-identical control, and 0 of
// the pair's 30 call sites read the value (the reading is worked through in pthread_slot.hpp,
// #2636). So both destroys answer 0, and #2168's refusal is expressed where it is actually
// observable: in the object NOT being retired. `_Mtx_lock` maps its result onto a `_Thrd_*` code
// (`0 -> 0, 0x8002000b -> 3, else 4`), and #2626 is where prosper stopped answering 0 unconditionally
// there. It was the widest false success left in the family: `std::mutex::lock()` was told it holds a
// lock it does not, on the two paths `guest_mutex_lock_slot` can refuse — a slot carrying the
// destroyed poison, and a same-thread relock of an ERRORCHECK mutex.
//
// WHY THIS IS A DIVERGENCE AND NOT A STYLE QUESTION, in this repository's own terms: the linker
// resolves an import against sibling guest modules FIRST ("Cross-module export beats a stub slot",
// linker.cpp pass 2). So a title that ships `sce_module/libc.prx` runs the wrapper disassembled in
// pthread_slot.hpp and gets the `_Thrd_*` mapping; a title that does not ships that import to THIS
// handler instead. Before #2626 the two answered a failing lock differently — the #1873 shape, with
// the door chosen by the title's packaging rather than by anything it does.
//
// The blast radius is bounded on the side that matters: the SUCCESS path is untouched, and a
// `_Thrd_error` can only reach Dinkumware (where it terminates the guest — see the note above
// SCE_PTHREAD_ALIAS in hle_kernel.cpp) on a lock prosper REFUSED, which is a lock the guest does not
// hold. Terminating there is the shipped wrapper's own contract; returning 0 was a corruption.
HLE(m_mtx_lock)   { return thrd_rc_from_mutex_lock(guest_mutex_lock_slot(a0)); }
HLE(m_mtx_unlock) { (void)guest_mutex_unlock_slot(a0); return 0; }
// _Mtx_destroy / _Cnd_destroy are the guest STL's own spelling of the same objects, so they carry
// the same lifetime rule as scePthreadMutexDestroy / scePthreadCondDestroy: quarantine the storage
// instead of freeing it here, and defer the host destroy to reclaim, because a thread may be parked
// inside the object. See the contract block above k_mutex_destroy in hle_kernel.cpp, and
// sync_retire.hpp (#2042). The `SyncObjectKind` argument is the ONLY thing that differs from the
// Sony spelling: it keeps the `_Mtx`/`_Cnd` census buckets #2619 used as its reachability evidence.
HLE(m_mtx_destroy){ return guest_mutex_destroy_slot(a0, SyncObjectKind::StlMutex); }
HLE(m_cnd_init)   { if (a0) { auto* c = (pthread_cond_t*)calloc(1, sizeof(pthread_cond_t)); pthread_cond_init(c, nullptr); *(void**)P(a0) = c; } return 0; }
HLE(m_cnd_signal) { guest_cond_signal_slot(a0); return 0; }
HLE(m_cnd_broadcast){ guest_cond_broadcast_slot(a0); return 0; }
// The `return 0` here is NOT the defect and must stay. The shipped guest libc.prx's own `_Cnd_wait`
// is `push rbp; mov rbp,rsp; call <plt scePthreadCondWait>; xor eax,eax; pop rbp; ret` -- it
// discards the result on every path, so an unconditional 0 is the CONTRACT rather than an oversight
// (#1983, recorded above k_cond_wait, which is where the reading was made). #2596 is only about the
// wait not happening. Do not "fix" this line. What DID change is the body whose result it discards:
// `guest_cond_wait_slot` takes the #2168 waiter scope, so a thread parked here is visible to every
// cond-destroy busy check instead of having its condvar retired out from under it (#2623).
HLE(m_cnd_wait)   { (void)guest_cond_wait_slot(a0, a1); return 0; }
// Answers 0, exactly like its `_Mtx_destroy` sibling above. The guest's eleven-byte wrapper never
// touches eax after its call to `scePthreadCondDestroy`, and that shape does NOT establish a
// forwarded result -- it is what a void wrapper compiles to (pthread_slot.hpp works the reading
// through against `_Thrd_yield`, its byte-identical void control, and against a call-site census in
// which 0 of 30 destroy sites read the value; #2636).
// What #2168 needs from this handler is a REFUSAL, not a number, and the refusal is in the shared
// body: `guest_cond_destroy_slot` declines to claim or retire a condvar a thread is parked on, so
// the guest keeps a handle that still works. Publishing `kSceKernelErrorEBUSY` here instead would
// put a LIBKERNEL-encoded value through a C11-spelled door -- a third convention, neither the bare
// errno the POSIX spelling answers nor a `_Thrd_*` code, and the one answer no reading of the bytes
// supports. The `(void)` is where the shared body's result is dropped, deliberately and visibly.
HLE(m_cnd_destroy){ (void)guest_cond_destroy_slot(a0, SyncObjectKind::StlCond); return 0; }

// --- event queue (sceKernelEqueue): kqueue-like event mechanism the engine uses for vsync/flip
// and async I/O completion. Headless: give a valid queue object; WaitEqueue yields briefly and
// reports no events (callers time out and retry) so nothing busy-spins and no null queue is used. ---
namespace { bool evlog() { static int v = getenv("PROSPER_EVLOG") ? 1 : 0; return v; } }

// --- Real event-queue backend (kqueue/kevent model). Registered flip/vblank sources post events into
// their equeue; WaitEqueue blocks until an event is ready (or timeout) and returns it. A single ~60 Hz
// pump thread drives vblank + flip-completion events so Unity's render/timing threads pace frames. ---
// The shared vblank grid, defined in hle/graphics/hle_graphics.cpp. Declared here rather than
// in a header because that file deliberately keeps the grid internal, and these two accessors
// are the whole of what it publishes (#3024). The PERIOD is not a constant: since #3017 it comes
// from the display mode prosper advertises to the guest, so this pump paces at whatever rate the
// title was told it is running at -- which is the point, and why it is read rather than assumed.
extern "C" uint64_t prosper_vo_vblank_grid_origin_ns();
extern "C" uint64_t prosper_vo_vblank_period_ns();

namespace {
    // SceKernelEvent (FreeBSD kevent layout, 0x20 bytes): ident@0, filter@8(i16), flags@0xA(u16),
    // fflags@0xC(u32), data@0x10, udata@0x18. The game reads udata (its flip context) + data.
    struct SceKEvent { int64_t ident; int16_t filter; uint16_t flags; uint32_t fflags; int64_t data; uint64_t udata; };
    // PS5 filter ids (negative, FreeBSD-style). VideoOut flip/vblank use the DISPLAY filter family.
    // SCE VideoOut event filter (flip + vblank): -13 per BOTH references (Kyty EventQueue.h:19
    // KERNEL_EVFILT_VIDEO_OUT = -13; shadPS4 equeue.h Filter::VideoOut = -13). The previous -10
    // is FreeBSD's EVFILT_LIO and matched neither — guest code switching on event.filter (the
    // standard way to recognize a flip kevent, cf. Kyty's assert) would never match our events.
    constexpr int16_t EVFILT_VIDEO_OUT = -13;
    // VideoOut event idents (Kyty VideoOut.cpp:34): the kevent's ident names the EVENT KIND, not the
    // videoout handle. A flip event's data is the completed flip's flipArg (Kyty
    // flip_event_trigger_func) — the game compares it against the arg it submitted.
    constexpr int64_t VIDEO_OUT_EVENT_FLIP = 0, VIDEO_OUT_EVENT_VBLANK = 1;

    // Equeue lifetime (#67): states are SHARED-ptr owned. eq_find hands out a reference that keeps
    // the object alive across the caller's wait, so sceKernelDeleteEqueue can never destroy a mutex/
    // condvar a waiter is blocked on (the old raw-pointer scheme was a use-after-free: delete freed
    // the state while k_eq_wait sat in cv.wait on it). Delete marks `deleted` and wakes waiters; the
    // object dies when the last reference drops.
    struct PendingEvent { SceKEvent event; bool coalescible; };
    struct EqState {
        std::mutex m;
        std::condition_variable cv;
        std::deque<PendingEvent> ready;
        size_t coalescible_ready = 0;
        uint64_t identity = 0;
        bool deleted = false;
    };
    PROSPER_HEAP_MUTEX(g_eq_mx);   // #707: heap-backed on macOS (std::mutex would land in the corrupted __DATA cluster)
    std::unordered_map<uint64_t, std::shared_ptr<EqState>> g_eqs;   // guest eq handle -> state
    std::atomic<uint64_t> g_eq_next_identity{1};
    struct FlipReg { uint64_t eq; int64_t ident; uint64_t udata; };
    std::vector<FlipReg> g_flip_regs, g_vblank_regs;
    // GPU end-of-pipe (EOP) event sources registered via sceGnmAddEqEvent / GraphicsAddEqEvent
    // (NID b0xyllnVY-I). Mirrors shadPS4 sceGnmAddEqEvent: on submit completion the GPU interrupt
    // triggers TriggerEvent(ident=id, filter=GraphicsCore, data=id, udata). id is GfxEop=0x40 (or a
    // ComputeN ring id). Filter GraphicsCore=-14 (shadPS4 equeue.h).
    constexpr int16_t EVFILT_GRAPHICS_CORE = -14;
    std::vector<FlipReg> g_eop_regs;   // same (eq,id,udata) shape
    // #987 diagnostic: a completion snapshot taken before any EOP source is registered is currently
    // delivered to zero queues. Count those observations under the same mutex as registration so the
    // PROSPER_EOPLOG ordering is exact. This is diagnostic only: the count is never replayed.
    uint64_t g_eop_zero_consumer_count = 0;
    bool eoplog() { static const bool enabled = getenv("PROSPER_EOPLOG") != nullptr; return enabled; }
    // Registered user-event sources: (eq,id,udata) the game added via sceKernelAddUserEvent and may later
    // trigger. Declared here (before the pump) so the diagnostic PROSPER_PUMP_USEREV heartbeat can fire them.
    struct UserReg { uint64_t eq; int64_t id; uint64_t udata; };
    std::vector<UserReg> g_user_regs;   // guarded by g_eq_mx
    // Pending one-shot timers (#67): (eq,id) -> cancellation token. The detached timer thread
    // checks the token before posting, so sceKernelDelete(HR)TimerEvent (previously a no-op — a
    // cancelled timer still fired, delivering a phantom event with possibly-freed udata) and
    // sceKernelDeleteEqueue really cancel. Guarded by g_eq_mx.
    struct TimerTok { std::shared_ptr<std::atomic<bool>> cancelled; };
    std::map<std::pair<uint64_t, int64_t>, TimerTok> g_timers;
    std::atomic<bool> g_pump_started{false};

    std::shared_ptr<EqState> eq_find(uint64_t eq, uint64_t expected_identity = 0) {
        std::lock_guard lk(g_eq_mx);
        auto it = g_eqs.find(eq);
        if (it == g_eqs.end() ||
            (expected_identity && it->second->identity != expected_identity))
            return nullptr;
        return it->second;
    }
    void eq_post(uint64_t eq, const SceKEvent& e, bool coalesce = true,
                 uint64_t expected_identity = 0) {
        auto s = eq_find(eq, expected_identity); if (!s) return;
        std::lock_guard<std::mutex> lk(s->m);
        if (s->deleted) return;
        // kqueue semantics: one knote per (ident, filter) — a re-trigger UPDATES the pending event
        // in place (fresh data/fflags/udata) instead of queuing a duplicate. This is what makes the
        // 60 Hz vblank pump safe: previously a fixed 4-entry cap dropped the NEWEST event once the
        // queue filled with pumped vblanks — and the dropped one could be the flip-completion event
        // carrying the exact flipArg the game's pacer was waiting to observe (frame-pacing stall).
        // coalesce=false queues a distinct entry instead: the APR pointer-tag channel (#210) needs
        // EVERY completion delivered — two in-flight completions share (ident, filter) but carry
        // different request tags in data, and replacing one loses a completion forever.
        if (coalesce)
            for (auto& pending : s->ready) {
                auto& q = pending.event;
                if (pending.coalescible && q.ident == e.ident && q.filter == e.filter) {
                    // Timer / HR-timer (filter -7 / -15): ACCUMULATE expirations across coalesced fires so
                    // the delivered event carries expirations-since-last-read (kqueue EVFILT_TIMER). Other
                    // filters keep replace-in-place (vblank/flip: the newest event wins).
                    int64_t data = (e.filter == -7 || e.filter == -15) ? q.data + e.data : e.data;
                    // DIAGNOSTIC (PROSPER_EVLOG=1): a replace-in-place is a DELIVERY THAT WILL NOT HAPPEN.
                    // For a level source (vblank/flip) that is the intended semantics and the line is
                    // noise; for a COUNTED completion source it is a lost wakeup, and this is the only
                    // place that can see it — by the time a consumer notices, the evidence is a thread
                    // blocked in WaitEqueue with nothing to say why. Printed with ident/filter so the
                    // APR completion filter (-24) is separable from the 60 Hz pump. Same gate as every
                    // other producer here, so the count can never be an unarmed zero.
                    if (evlog() && e.filter != -7 && e.filter != -15)
                        fprintf(stderr, "[ev] coalesce-replace eq=0x%llx ident=%lld filter=%d "
                                        "(dropped data=%lld, new data=%lld)\n",
                                (unsigned long long)eq, (long long)e.ident, (int)e.filter,
                                (long long)q.data, (long long)data);
                    q = e; q.data = data; s->cv.notify_all(); return;
                }
            }
        // Distinct coalesced (ident, filter) pairs are few; cap that level-style queue as a leak
        // guard and shed the oldest COALESCIBLE event if it ever fires. Count-sensitive EOP/APR
        // entries may share this deque and must never be selected as overflow victims.
        if (coalesce && s->coalescible_ready >= 64) {
            auto oldest = std::find_if(s->ready.begin(), s->ready.end(),
                                       [](const PendingEvent& p) { return p.coalescible; });
            if (oldest != s->ready.end()) {
                s->ready.erase(oldest);
                s->coalescible_ready--;
            }
        }
        s->ready.push_back({ e, coalesce });
        s->coalescible_ready += coalesce;
        s->cv.notify_all();
    }
    void vblank_pump() {
        uint64_t frame = 0;
        // Absolute schedule on the SHARED 59.94 Hz grid that hle_graphics.cpp uses for the
        // vblank status and wait paths, waited with host::sleep_until_steady_ns. Two defects
        // this replaces (#3024):
        //
        //   * A relative nanosleep(16.67 ms) resolves on the winpthreads tick on Windows. A
        //     standalone microbenchmark of that nanosleep on this toolchain (MinGW-w64/UCRT,
        //     GCC 16.1) returns every 29.95 ms, so the pump was posting vblank kevents at
        //     roughly 33 Hz rather than 60, and every consumer pacing on this event stream
        //     inherited it. Provenance stated because it matters: that is the sleep primitive
        //     measured in isolation, NOT an instrumented measurement of this pump, and the
        //     figure is quoted here only to size the defect. Same microbenchmark under
        //     timeBeginPeriod(1) gives 24.78 ms -- still not 16.67 -- and that arm is
        //     hypothetical for prosper, which calls timeBeginPeriod nowhere (see
        //     precise_sleep.cpp's own note). So the unrescued 29.95 ms is what ran.
        //     It is the same defect #1765 fixed for the status/wait grid one file away, which
        //     is why that file already says "NOT std::this_thread::sleep_until" over its wait.
        //   * A relative sleep also drifts against the status grid. hle_graphics.cpp records
        //     that as an open follow-up in as many words -- the two clocks "drift by roughly one
        //     tick per second and their boundaries do not coincide". Deriving both from one
        //     origin and one period fixes the BOUNDARIES: a kevent now lands on an instant the
        //     status grid also calls a boundary.
        //
        //     It does NOT make the two COUNTS equal, and the earlier wording here claimed it
        //     did. `e.data` below is `frame`, which counts posts; GetVblankStatus's `count` is
        //     the grid INDEX. They agree only while no boundary is skipped -- and skipping is
        //     what this design does deliberately when a tick is missed, rather than bursting.
        //     A title cross-checking the two would still see a divergence after any stall.
        //     Making the payload the grid ordinal would close that too, but `data = frame`
        //     predates this change and nothing here establishes what real hardware puts in
        //     that field, so it is left alone and named instead of quietly overclaimed.
        //
        // One call carries the whole schedule: host::next_grid_deadline_ns returns the next
        // boundary strictly after now, so the ordinary case and a stall of any length are the
        // same expression and the phase cannot drift. It is a separate tested function because
        // the obvious hand-rolled spelling -- advance by a period, re-anchor to now+period when
        // behind -- silently loses the phase, and an earlier draft of this loop also halved the
        // rate. See its comment in precise_sleep.hpp.
        //
        // The cost of that property, named because it is real: phase stays exact however
        // many boundaries were skipped, so a pump running at HALF rate has perfect phase
        // and no symptom. The relative sleep this replaces at least drifted visibly, which
        // is how #3024 was noticed. #3075 restores the symptom with host::grid_boundaries_missed()
        // below, WITHOUT giving back the drift: that helper is deliberately a second, separate
        // computation over next_grid_deadline_ns()'s own return values rather than a change to what
        // the scheduler returns or how it schedules -- see its comment in precise_sleep.hpp for why
        // folding the two together would just recreate the drift #3024 removed.
        //
        // One composition caveat, recorded because the helper alone does not cover it (#3074):
        // this loop assumes the wait cannot return EARLY. If it does, the next `now` is still
        // below the same boundary, the helper correctly returns that boundary again, and this
        // thread posts two kevents for one vblank. Reachable only on the Win32SleepFallback
        // path, which does not re-check the clock the way the high-resolution timer path does;
        // bounded (one extra event, and vblank posts coalesce) so it is filed rather than
        // worked around here. The fix belongs in precise_sleep, which already documents the
        // post-condition it fails to enforce on that one path.
        // Reading the origin here is what usually ANCHORS it -- see the note in
        // hle_graphics.cpp. Two qualifications, because the obvious statement of that is not
        // quite true: ensure_pump() detaches this thread, so a guest status or wait call
        // arriving inside the thread-start window anchors the grid instead, which makes the
        // epoch's wall-clock phase nondeterministic by thread-start latency; and the pump is
        // started by prosper_eq_add_flip as well as prosper_eq_add_vblank, so a title that
        // registers only a FLIP event still anchors it here.
        const uint64_t origin_ns = prosper_vo_vblank_grid_origin_ns();
        const uint64_t period_ns = prosper_vo_vblank_period_ns();

        // --- #3075: dropped-tick visibility ------------------------------------------------------
        // `deadline_ns` below is read back and compared against the PREVIOUS iteration's own
        // deadline via host::grid_boundaries_missed(), which is exactly the history
        // next_grid_deadline_ns() is required to discard for its phase-exactness guarantee. This is
        // a SEPARATE computation, not a change to the schedule: `deadline_ns` itself is still fed to
        // sleep_until_steady_ns() unmodified, so scheduling behaviour (and #3024's fix) is untouched.
        bool have_prev_deadline = false;
        uint64_t prev_deadline_ns = 0;
        // Windowed, not per-boundary: a stall of N periods is ONE iteration reporting `missed=N`,
        // never N reports, so this cannot flood on a long stall. It IS one report per iteration on a
        // sustained degraded rate (e.g. every iteration of a half-rate run), which is the case that
        // must stay loud -- bounded above by the pump's own (halved, in that case) iteration rate.
        uint64_t win_ticks = 0, win_missed = 0, win_worst_gap = 0;
        uint64_t lifetime_missed = 0;
        clk::time_point win_start = clk::now();
        // "More than a few percent of boundaries over a multi-second window" -- the design question
        // #3075 posed rather than answered. Answered here: yes, unconditionally (never gated behind
        // PROSPER_EVLOG, so it is visible on a default run), because a pump silently running at a
        // fraction of its nominal rate is exactly what this project's fail-visible policy exists for.
        // A DETAILED per-window line (below, gated) is still useful for a below-threshold trickle of
        // drops that this warning deliberately does not raise.
        constexpr double kDropWarnThreshold = 0.05;      // >5% of boundaries dropped in the window
        constexpr auto   kDropWindow = std::chrono::seconds(2);

        for (;;) {
            const uint64_t now_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                clk::now().time_since_epoch()).count();
            const uint64_t deadline_ns =
                prosper::host::next_grid_deadline_ns(origin_ns, now_ns, period_ns);
            prosper::host::sleep_until_steady_ns(deadline_ns);
            frame++;

            if (have_prev_deadline) {
                const uint64_t missed =
                    prosper::host::grid_boundaries_missed(prev_deadline_ns, deadline_ns, period_ns);
                win_ticks++;
                if (missed) {
                    win_missed += missed;
                    lifetime_missed += missed;
                    if (missed > win_worst_gap) win_worst_gap = missed;
                }
            }
            have_prev_deadline = true;
            prev_deadline_ns = deadline_ns;

            const clk::time_point report_now = clk::now();
            if (report_now - win_start >= kDropWindow) {
                const uint64_t win_boundaries = win_ticks + win_missed;   // ticks observed + skipped
                const double ratio =
                    win_boundaries ? (double)win_missed / (double)win_boundaries : 0.0;
                if (ratio > kDropWarnThreshold) {
                    fprintf(stderr,
                            "[vblank] WARNING: pump dropped %llu of %llu expected boundaries in the "
                            "last %llds (%.1f%%, worst single wait skipped %llu) -- a consumer "
                            "pacing on the vblank event stream is running slow.\n",
                            (unsigned long long)win_missed, (unsigned long long)win_boundaries,
                            (long long)kDropWindow.count(), ratio * 100.0,
                            (unsigned long long)win_worst_gap);
                }
                if (evlog())
                    fprintf(stderr,
                            "[ev] vblank-pump interval=%llds ticks=%llu dropped=%llu/%llu (%.2f%%, "
                            "worst=%llu) | lifetime dropped=%llu\n",
                            (long long)kDropWindow.count(), (unsigned long long)win_ticks,
                            (unsigned long long)win_missed, (unsigned long long)win_boundaries,
                            ratio * 100.0, (unsigned long long)win_worst_gap,
                            (unsigned long long)lifetime_missed);
                win_ticks = win_missed = win_worst_gap = 0;
                win_start = report_now;
            }

            std::vector<FlipReg> vr;
            { std::lock_guard lk(g_eq_mx); vr = g_vblank_regs; }
            // Vblank ticks are periodic by nature — pump them. FLIP events are NOT pumped: a flip
            // event fires when a submitted flip completes (prosper_eq_trigger_flip below), carrying
            // that flip's flipArg in `data`. The old timer-driven flip event carried a frame counter
            // — the game compared it against its submitted flipArg (top-bit-set values like
            // 0x8000000000000001), never saw its flip complete, and re-waited forever.
            for (auto& r : vr) { SceKEvent e{}; e.ident = VIDEO_OUT_EVENT_VBLANK; e.filter = EVFILT_VIDEO_OUT;
                                 e.fflags = 1; e.data = (int64_t)frame; e.udata = r.udata; eq_post(r.eq, e); }
            // PROSPER_PUMP_USEREV: heartbeat-fire registered user events. Some engines run a worker thread
            // that blocks on a user event (Unity's FTM queue: user event id=999) waiting for the producer
            // to signal work; if the producer path isn't reached, that thread starves and the game idles.
            // Firing it each vblank tests whether waking that consumer lets the game progress to real draws.
            if (getenv("PROSPER_PUMP_USEREV")) {
                std::vector<UserReg> ur; { std::lock_guard lk(g_eq_mx); ur = g_user_regs; }
                for (auto& r : ur) { SceKEvent e{}; e.ident = r.id; e.filter = -11 /*EVFILT_USER*/; e.udata = r.udata; eq_post(r.eq, e); }
            }
        }
    }
    void ensure_pump() { if (!g_pump_started.exchange(true)) std::thread(vblank_pump).detach(); }
}

// Exposed to hle_graphics.cpp (sceVideoOut* flip/vblank event registration).
void prosper_eq_add_flip(uint64_t eq, int64_t ident, uint64_t udata) {
    { std::lock_guard lk(g_eq_mx); g_flip_regs.push_back({ eq, ident, udata }); }
    ensure_pump();
}
// Fire the flip-completion event on every registered flip equeue — called by BOTH flip paths
// (sceVideoOutSubmitFlip and the in-stream Dcb SetFlip) at the flip moment. Kevent shape per Kyty
// flip_event_trigger_func: ident=VIDEO_OUT_EVENT_FLIP, data=the completed flip's flipArg.
void prosper_eq_trigger_flip(int64_t flip_arg) {
    std::vector<FlipReg> regs;
    { std::lock_guard lk(g_eq_mx); regs = g_flip_regs; }
    for (auto& r : regs) {
        SceKEvent e{}; e.ident = VIDEO_OUT_EVENT_FLIP; e.filter = EVFILT_VIDEO_OUT;
        e.fflags = 1; e.data = flip_arg; e.udata = r.udata;
        eq_post(r.eq, e);
    }
}
void prosper_eq_add_vblank(uint64_t eq, int64_t ident, uint64_t udata) {
    { std::lock_guard lk(g_eq_mx); g_vblank_regs.push_back({ eq, ident, udata }); }
    ensure_pump();
}
void prosper_eq_start_eop_watchdog();   // TEMPORARY diagnostic, defined below

// Exposed to the AGC submit path (hle_agc.cpp). Register a GPU EOP event source (sceGnmAddEqEvent).
void prosper_eq_add_eop(uint64_t eq, int64_t id, uint64_t udata) {
    size_t registration_count = 0;
    uint64_t zero_consumer_count = 0;
    uint64_t registration_ns = 0;
    {
        std::lock_guard lk(g_eq_mx);
        g_eop_regs.push_back({ eq, id, udata });
        registration_count = g_eop_regs.size();
        zero_consumer_count = g_eop_zero_consumer_count;
        registration_ns = real_ns();
    }
    prosper_eq_start_eop_watchdog();
    if (eoplog())
        fprintf(stderr, "[eoprace] t_ns=%llu EOP source registered eq=0x%llx id=%lld "
                        "(regs=%zu zero-consumer-completions=%llu)\n",
                (unsigned long long)registration_ns, (unsigned long long)eq, (long long)id,
                registration_count, (unsigned long long)zero_consumer_count);
}
// Fire the registered EOP events — called when a submit completes. Posts TriggerEvent(ident=id,
// filter=GraphicsCore, data=id, udata) to each registered equeue, matching shadPS4's IRQ handler.
// Inert if none registered.
// GPU pipe-drain completion-write drain (src/gpu/pm4/command_processor.cpp, #312): an EOP EVENT must
// never overtake its submit's fence/label WRITES — drain the write queue before posting. Weak so
// binaries that link the kernel HLE without the gpu lib still link.
extern "C" void prosper_gpu_drain_completion_writes() __attribute__((weak));
extern "C" bool prosper_gpu_submit_scope_active() __attribute__((weak));
namespace {
    // Actually post one EOP completion to every registered equeue (the worker below calls this).
    void eop_post_now() {
        if (prosper_gpu_drain_completion_writes) prosper_gpu_drain_completion_writes();
        // Deliver EVERY GPU-completion event as a DISTINCT equeue entry (coalesce=false). All EOP events
        // share the same (ident, EVFILT_GRAPHICS_CORE), so coalescing collapses N submit-completions into
        // ONE pending event — and the game's EOP handler posts a work-queue semaphore once per delivered
        // completion, so a coalesced completion UNDER-posts the semaphore and its consumer deadlocks
        // (GfxDevice work-queue eboot+0xb06ad2 / PreloadManager 0x18a83b5 / a worker). That was The
        // Messenger's post-SaveData scene-activation stall (#234): with coalesce=false the scene activates
        // and the intro cutscene plays. Same reason the APR channel (#210) is coalesce=false — a completion
        // is a discrete count, never a level. PROSPER_EOP_COALESCE restores the old behavior.
        static const bool coalesce = getenv("PROSPER_EOP_COALESCE") != nullptr;
        std::vector<FlipReg> regs;
        uint64_t zero_consumer_ordinal = 0;
        uint64_t snapshot_ns = 0;
        {
            std::lock_guard lk(g_eq_mx);
            regs = g_eop_regs;
            if (regs.empty()) {
                zero_consumer_ordinal = ++g_eop_zero_consumer_count;
                snapshot_ns = real_ns();
            }
        }
        if (zero_consumer_ordinal && eoplog())
            fprintf(stderr, "[eoprace] t_ns=%llu EOP completion snapshot had 0 registered "
                            "queues (zero-consumer ordinal=%llu)\n",
                    (unsigned long long)snapshot_ns, (unsigned long long)zero_consumer_ordinal);
        // TEMPORARY: post N events per completion, to test whether the guest needs more than one.
        // Counted so the lever can PROVE it moved -- a null from an unverified lever is void.
        static const int repeat = [] {
            const char* v = getenv("PROSPER_EOP_REPEAT");
            const int n = v ? atoi(v) : 1;
            return n > 0 ? n : 1;
        }();
        static std::atomic<uint64_t> posted{0};
        for (auto& r : regs) {
            for (int i = 0; i < repeat; ++i) {
                SceKEvent e{}; e.ident = r.ident; e.filter = EVFILT_GRAPHICS_CORE;
                e.data = r.ident; e.udata = r.udata;
                eq_post(r.eq, e, coalesce);
                const uint64_t n = ++posted;
                if (getenv("PROSPER_EOP_TRACE") && (n % 2000) == 0)
                    fprintf(stderr, "[eop-post] posted=%llu (repeat=%d)\n",
                            (unsigned long long)n, repeat);
            }
        }
    }
    // Deferred, ORDERED EOP delivery worker. On real hardware the EOP interrupt can only fire AFTER
    // the submit call has returned — the GPU sees the command buffer when the driver rings the doorbell
    // at the END of the submit — and the interrupt arrives micro/milliseconds later. prosper's fold is
    // synchronous, so firing the kevent INSIDE the submit call violated that invariant: the guest's
    // interrupt->cleanup chain (AgcInterruptThread -> AgcCleanupThread) could observe frame N complete
    // BEFORE the AgcSubmissionThread finished frame N's own post-submit bookkeeping, and the cleanup
    // raced the submitter's retired-allocation list. DOLL (UE4 PPSA17942) hit exactly that: an
    // intermittent MallocBinned3 "Corruption Canary was 0x3, should be 0x1" LowLevelFatalError in the
    // RHI's free-list loop (eboot+0x220bd50, GMalloc->Free over a retired-buffer array) -> libc abort
    // (int $0x45, stop code 0xa002000b), killing ~half of boots between flips 200-2500 (issues
    // #232/#241). Deferring only the EVENT (label/fence WRITES stay synchronous — they are data
    // dependencies, not lifecycle signals) restores the real ordering: the submit returns first, then
    // the completion interrupt fires ~1 ms later. Completion writes now share the same post-submit
    // visibility gate, and eop_post_now drains them before publishing the event. A single FIFO
    // worker preserves inter-submit order
    // (#236 needs every completion delivered distinctly and in order). PROSPER_EOP_SYNC=1 restores
    // synchronous delivery only outside a live submit scope; an in-submit request still queues to
    // avoid deadlocking on its own completion-write drain. CONFIDENCE: HIGH on the invariant (real
    // EOP is post-submit by
    // construction; Kyty's GraphicsRunDone/EOP also fires from the GPU thread after CommandProcessor
    // execution, never inside the submit call).
    // IMMORTAL (leaked) worker state: the worker thread is detached and outlives main, so this state
    // must never run static destructors — a worker blocked in wait() on a destroyed condvar/mutex
    // hangs process exit (test_equeue_events under ctest caught exactly that).
    struct EopQueue { std::mutex mx; std::condition_variable cv; uint64_t pending = 0; };
    EopQueue& eop_q() { static EopQueue* q = new EopQueue; return *q; }
    std::atomic<bool> g_eop_worker_started{false};
    void eop_worker() {
        EopQueue& q = eop_q();
        std::unique_lock<std::mutex> lk(q.mx);
        for (;;) {
            q.cv.wait(lk, [&] { return q.pending > 0; });
            uint64_t n = q.pending;
            q.pending = 0;
            lk.unlock();
            // Modeled GPU pipe-drain latency: long enough for the submitting guest thread to return
            // from the submit import and finish its bookkeeping, short enough to not throttle the
            // frame loop (same order as the APR channel's 2 ms modeled DMA latency). One sleep covers
            // the whole burst; each queued completion is still posted DISTINCTLY and in order (#236).
            struct timespec ts{ 0, 1000000 };   // 1 ms
            nanosleep(&ts, nullptr);
            while (n--) eop_post_now();
            lk.lock();
        }
    }
}
// TEMPORARY: an independent EOP pulse, started at REGISTRATION rather than at the first trigger, so
// it keeps posting after the guest stops submitting. That is the only way to break a circular wait:
// events driven by submits cannot arrive when the guest is blocked and therefore not submitting.
void prosper_eq_start_eop_watchdog() {
    static std::atomic<bool> started{false};
    const char* ms_env = getenv("PROSPER_EOP_WATCHDOG_MS");
    if (!ms_env || !*ms_env) return;
    const long ms = atol(ms_env);
    if (ms <= 0) return;
    if (started.exchange(true)) return;
    fprintf(stderr, "[eop-wd] watchdog armed at %ld ms\n", ms);
    fflush(stderr);
    std::thread([ms] {
        uint64_t n = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            eop_post_now();
            if ((++n % 500) == 0) {
                fprintf(stderr, "[eop-wd] fired=%llu\n", (unsigned long long)n);
                fflush(stderr);
            }
        }
    }).detach();
}

void prosper_eq_trigger_eop() {
    static const bool sync = getenv("PROSPER_EOP_SYNC") != nullptr;
    const bool submit_active = prosper_gpu_submit_scope_active && prosper_gpu_submit_scope_active();
    if (sync && !submit_active) { eop_post_now(); return; }
    if (!g_eop_worker_started.exchange(true)) std::thread(eop_worker).detach();
    EopQueue& q = eop_q();
    { std::lock_guard<std::mutex> lk(q.mx); q.pending++; }
    q.cv.notify_one();
}

// Stable identity for the current lifetime of an opaque equeue handle. Allocator address reuse can
// give a later queue the same handle, so delayed producers must retain and validate this value.
uint64_t prosper_eq_identity(uint64_t eq) {
    auto s = eq_find(eq);
    return s ? s->identity : 0;
}

HLE(k_eq_create) {
    auto s = std::make_shared<EqState>();
    s->identity = g_eq_next_identity.fetch_add(1, std::memory_order_relaxed);
    if (a0) *(void**)P(a0) = (void*)s.get();   // the guest's opaque SceKernelEqueue handle IS our state ptr
    { std::lock_guard lk(g_eq_mx); g_eqs[(uint64_t)(uintptr_t)s.get()] = s; }
    if (evlog()) fprintf(stderr, "[ev] CreateEqueue -> eq=%p name=%s\n", (void*)s.get(), a1 ? (const char*)P(a1) : "");
    return 0;
}
HLE(k_eq_delete) {
    if (!a0) return 0;
    std::shared_ptr<EqState> s;
    {
        std::lock_guard lk(g_eq_mx);
        auto it = g_eqs.find(a0);
        if (it != g_eqs.end()) { s = std::move(it->second); g_eqs.erase(it); }
        // Purge every registration pointing at this queue (#67). Without this, a later heap
        // allocation reusing the same address RESURRECTED the dead registrations: pump/flip/EOP
        // events were delivered onto an unrelated new queue.
        auto drop_eq = [&](std::vector<FlipReg>& v) {
            v.erase(std::remove_if(v.begin(), v.end(), [&](const FlipReg& r){ return r.eq == a0; }), v.end());
        };
        drop_eq(g_flip_regs); drop_eq(g_vblank_regs); drop_eq(g_eop_regs);
        g_user_regs.erase(std::remove_if(g_user_regs.begin(), g_user_regs.end(),
                                         [&](const UserReg& r){ return r.eq == a0; }), g_user_regs.end());
        // Cancel this queue's pending one-shot timers.
        for (auto it2 = g_timers.begin(); it2 != g_timers.end(); ) {
            if (it2->first.first == a0) { it2->second.cancelled->store(true); it2 = g_timers.erase(it2); }
            else ++it2;
        }
    }
    if (s) {   // wake waiters; the shared_ptr they hold keeps the state alive until they exit
        std::lock_guard<std::mutex> lk(s->m);
        s->deleted = true;
        s->cv.notify_all();
    }
    if (evlog()) fprintf(stderr, "[ev] DeleteEqueue eq=0x%llx\n", (unsigned long long)a0);
    return 0;
}
HLE(k_eq_wait)   {   // (eq, SceKernelEvent* ev, int num, int* out, SceKernelUseconds* timeout)
    if (evlog()) fprintf(stderr, "[ev] WaitEqueue eq=0x%llx num=%llu timeout=%s ra=%s+0x%llx\n",
        (unsigned long long)a0, (unsigned long long)a2, a4 ? "yes" : "inf",
        prosper::guest_module_name((uint64_t)__builtin_return_address(0)),
        (unsigned long long)prosper::guest_module_offset((uint64_t)__builtin_return_address(0)));
    const int num = static_cast<int32_t>(a2);
    auto s = eq_find(a0);
    // Match the API's validation order: queue handle, event array, then count. EFAULT does not
    // touch *out; EBADF and EINVAL report zero delivered events when the caller supplied it.
    if (!s) { if (a3) *(int32_t*)P(a3) = 0; return 0x80020009ull; }   // SCE_KERNEL_ERROR_EBADF
    if (!a1) return 0x8002000eull;                                    // SCE_KERNEL_ERROR_EFAULT
    if (num < 1) { if (a3) *(int32_t*)P(a3) = 0; return 0x80020016ull; } // SCE_KERNEL_ERROR_EINVAL
    // PROSPER_WAITCALLER: scan the stack for the GAME's wait-loop return address (eboot code range,
    // NOT the stub region at 0x6..) so we can disassemble the loop's exit condition. Log once per eq.
    // PROSPER_DUMPCODE=<hex eboot offset>[,<off2>...]: dump 224 code bytes at each (guest memory is mapped),
    // so a game function reached from the wait loop can be disassembled offline. Runs once.
    // Env probes are cached in statics: k_eq_wait is the guest frame loop's wait (called every frame by
    // several threads), so per-call getenv() environ scans are pure hot-path waste (cf. evlog()).
    static const char* const dumpcode = getenv("PROSPER_DUMPCODE");
    if (const char* dc = dumpcode) {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            const char* p = dc;
            while (*p) {
                uint64_t off = strtoull(p, nullptr, 16);
                // #1659: this DEREFERENCES the computed address, so the stale base did not merely mislabel —
                // PROSPER_DUMPCODE read 0x10000000 below the intended instruction.
                const uint8_t* code = (const uint8_t*)(uintptr_t)(prosper::BOOT_EBOOT + off);
                fprintf(stderr, "[dumpcode] eboot+0x%llx:", (unsigned long long)off);
                for (int b = 0; b < 224; b++) fprintf(stderr, "%02x", code[b]);
                fprintf(stderr, "\n");
                const char* c = strchr(p, ','); if (!c) break; p = c + 1;
            }
        }
    }
    static const bool waitcaller = getenv("PROSPER_WAITCALLER") != nullptr;
    if (waitcaller) {
        static std::atomic<int> shown{0};
        if (shown.load() < 8 && shown.fetch_add(1) < 8) {
            uint64_t* sp = (uint64_t*)__builtin_frame_address(0);
            for (int i = 0; i < 160; i++) {
                uint64_t v = sp[i];
                // _code: the weak form accepts BOOT_STUB, and the call-shape check below is
                // explicitly written for an in-eboot thunk rather than the stub region (#2045).
                if (!prosper::guest_va_in_module_code(v)) continue;   // any fixed guest module (#1659)
                // A genuine caller is a return address preceded by a call: `call rel32` (0xe8 at v-5,
                // any target — the import call goes through an in-eboot thunk, NOT straight to the
                // stub region, so no target filter) or an indirect `call` (0xff at v-6/-3/-2).
                const uint8_t* pre = (const uint8_t*)(uintptr_t)(v - 8);
                bool is_call = pre[3] == 0xe8 ||                            // call rel32 at v-5
                               pre[2] == 0xff || pre[5] == 0xff || pre[6] == 0xff;  // call r/m64 forms
                if (!is_call) continue;
                fprintf(stderr, "[waitcaller] eq=0x%llx num=%llu stack[%d] %s+0x%llx | code@ra-0x18:",
                        (unsigned long long)a0, (unsigned long long)a2, i,
                        prosper::guest_module_name(v), (unsigned long long)prosper::guest_module_offset(v));
                const uint8_t* code = (const uint8_t*)(uintptr_t)(v - 0x18);
                for (int b = 0; b < 0x40; b++) fprintf(stderr, "%02x", code[b]);
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "[waitcaller] ---\n");
        }
    }
    std::unique_lock<std::mutex> lk(s->m);
    if (s->ready.empty() && !s->deleted) {
        // timeout arg is a pointer to micro-seconds; NULL = wait forever (real semantics — the old
        // 100 ms cap-and-return-success invented a state the API never produces, and callers
        // following the documented `if (Wait(...) == 0) consume(ev[0])` pattern read stale stack).
        // PROSPER_WAITCAP (diagnostic, default off) still bounds a TIMED wait for bisection runs.
        auto pred = [&]{ return !s->ready.empty() || s->deleted; };
        if (a4) {
            uint64_t us = *(uint32_t*)P(a4);
            static const uint64_t cap = prosper::diag::env_u64_or_default_capped(
                "PROSPER_WAITCAP", getenv("PROSPER_WAITCAP"), 0ull, UINT64_MAX, "us", "0 = no cap");
            if (cap && us > cap) us = cap;
            if (evlog()) fprintf(stderr, "[ev]   WAIT.empty req=%lluus\n", (unsigned long long)us);
            s->cv.wait_for(lk, std::chrono::microseconds(us), pred);
        } else {
            if (evlog()) fprintf(stderr, "[ev]   WAIT.empty (infinite)\n");
            s->cv.wait(lk, pred);
        }
    }
    if (s->deleted && s->ready.empty()) { if (a3) *(int32_t*)P(a3) = 0; return 0x80020009ull; }  // deleted under us
    int n = 0; auto* ev = (SceKEvent*)P(a1);
    while (n < num && !s->ready.empty()) {
        if (ev) ev[n] = s->ready.front().event;
        s->coalescible_ready -= s->ready.front().coalescible;
        s->ready.pop_front();
        n++;
    }
    if (a3) *(int32_t*)P(a3) = n;
    if (evlog() && n > 0) fprintf(stderr, "[ev]   -> delivered %d ev(s) eq=0x%llx ra=%s+0x%llx (ident=%lld filter=%d)\n",
        n, (unsigned long long)a0, prosper::guest_module_name((uint64_t)__builtin_return_address(0)),
        (unsigned long long)prosper::guest_module_offset((uint64_t)__builtin_return_address(0)),
        (long long)(ev ? ev[0].ident : 0), (int)(ev ? ev[0].filter : 0));
    // Timed wait that expired with nothing: the real API distinguishes this from success (Kyty
    // EventQueue.cpp:310 KERNEL_ERROR_ETIMEDOUT). Only reachable with a timeout arg — the infinite
    // wait can only exit with events or a delete.
    if (n == 0) { return 0x8002003Cull; }   // KERNEL_ERROR_ETIMEDOUT
    return 0;
}
HLE(k_eq_getcount){
    auto s = eq_find(a0); int n = 0;
    if (s) { std::lock_guard<std::mutex> lk(s->m); n = (int)s->ready.size(); }
    if (evlog()) fprintf(stderr, "[ev] GetEventCount eq=0x%llx -> %d\n", (unsigned long long)a0, n);
    return (uint64_t)n;
}

// --- APR (Ampr file-read engine) completion events — issues #115/#180/#208. -------------------
// UE4's PS5 platform file layer (FAPRFileHandle / FAPREventQueueListener) drives batched APR reads
// through a listener context that is an eboot GLOBAL (PPSA17942 eboot+0x95aebd8). The FULL contract
// was recovered by static disassembly of the guest (issue #208; all offsets eboot-relative):
//   - ctx CONSTRUCTOR (+0x22a0670, run once at first batch submit): creates the APREventQueue
//     ([ctx+0x18]), registers ids 0x74fe+ring for rings 0..5 on it (the sSAUCCU1dv4 calls), and
//     SEEDS the per-ring counters: token counter [ctx+0xc0+ring*0x28] = 0x3e8 (1000) and listener
//     last-processed [ctx+0xc8+ring*0x28] = 0x3e7 (999). THE GUEST SEEDS ITS OWN RANGE WALK.
//   - batch SUBMIT (+0x22a02b0): token = (ring<<58) | [ctx+0xc0+ring*0x28]++ — the guest-chosen
//     completion token. It is passed to H896Pt-yB4I as the binding tag, stored in the per-ring
//     tracked slot ([ctx+0xa8+ring*0x28] -> [slot+0x10]), and inserted into a hash map at ctx+0x58
//     ({token -> completion callback}, 128-byte entries, chain sentinel -1).
//   - LISTENER loop (+0x22740b0): data = (ring<<58)|cnt via sceKernelGetEventData; walks
//     seq = last+1 ..= cnt calling the completion handler (+0x229dcb0) per seq, then stores
//     last := cnt UNCONDITIONALLY (even when cnt < last+1 — a low counter REGRESSES the seed).
//   - HANDLER (+0x229dcb0): matches token against [slot+0x10] (completes the cb + resubmits the
//     next queued batch via ASoW5WE-UPo with ring_1b=ring+1), then looks the token up in the hash;
//     a found entry is erased and its callback INVOKED (this is what fires the FEvent the blocked
//     CreateGlobalShaderMap precache waits on). A walked seq with NO slot match and NO hash entry
//     takes the null-entry path: a 64-byte ymm swap against address 0x10 (+0x229df3e) — FATAL on
//     real HW too (address 0x10 is never mapped). The guest therefore GUARANTEES every walked seq
//     has a tracked entry: counters are dense from 1000 and the walk starts there (seed 999).
// CONSEQUENCES for prosper (the #180 tag-echo experiment's residual +0x229df3e fault was caused by
// prosper itself):
//   - The ONLY events we may post are exact guest-chosen H896 binding tags, one per bound submit.
//   - NEVER post invented counters (pre-#208 catchup replays / vWU direct-read wakeups / unbound
//     submit counters): cnt < 1000 does not walk but REGRESSES last-processed via the
//     unconditional store, and the next real tag event then walks the gap seqs into the fatal
//     miss path. Record-polled flows (mount-era submits, vWU direct reads, the ring-6/ring_1b=6
//     sync channel) complete WITHOUT events — live-verified: the gdb-unwedged engine streamed the
//     whole remaining load through vWU reads with no matching events in flight.
// Event shape: guest code reads ONLY data (GetEventData) — ident/filter are never inspected — but
// eq_post coalesces on (ident,filter), so ident carries the ring to keep concurrent rings'
// completions from replacing each other; the per-ring coalescing to the HIGHEST counter is exactly
// the kqueue "completed up to" semantics the listener's range walk implements.
// SCOPE (#1673): everything in this block describes the COUNTER dialect, and the coalescing above
// is conditional on it. A third consumer — CRI ADX2 — binds through the same id != 0 branch with a
// constant ZERO tag and no range walk, so its completions are counted, not levelled, and are posted
// individually. See the discrimination comment in prosper_eq_post_apr_token below.
// CONFIDENCE: HIGH — ctor seeding, walk bounds, unconditional last:=cnt store, handler match/hash/
// null-entry paths all from static disassembly; tag-echo event consumption live-verified (#180).
namespace {
    constexpr int16_t EVFILT_AMPR_MODELED = -24;   // guest never reads filter; distinct on purpose
    struct AprEqReg { uint64_t eq; int64_t id; };
    // Own mutex (NOT g_eq_mx): the post path calls eq_post/eq_find, which lock g_eq_mx themselves.
    // Detached APR delivery threads can outlive main, so the mutex and every container they touch
    // are one intentionally immortal heap object. Heap placement also keeps the hot mutex out of
    // the macOS __DATA cluster affected by #707.
    struct AprTokenState {
        std::mutex mx;
        std::vector<AprEqReg> eq_regs;
        uint64_t ring_seq[64] = {};
        std::unordered_map<uint64_t, uint64_t> tag_hwm;
    };
    AprTokenState& apr_token_state() {
        static AprTokenState* state = new AprTokenState;
        return *state;
    }
    uint64_t apr_hwm_key(uint64_t eq_identity, unsigned ring) {
        return (eq_identity << 6) | (ring & 0x3f);
    }
    void apr_post(uint64_t eq, uint64_t eq_identity, int64_t id,
                  unsigned ring, uint64_t token, bool coalesce) {   // no APR lock held
        SceKEvent e{}; e.ident = id + (int64_t)ring; e.filter = EVFILT_AMPR_MODELED;
        e.data = (int64_t)token; e.udata = 0;
        eq_post(eq, e, coalesce, eq_identity);
    }
}
// Post an EXACT guest-chosen completion token (the H896Pt-yB4I binding tag) to the binding's own
// equeue — the completion signal for the bound/batched APR channel. Deferred ~2 ms so the guest
// finishes installing its tracking slot/hash entry first (real DMA latency the submitter's few
// bookkeeping instructions never race). The deferred thread posts the ring's HIGHEST tag counter
// recorded at post time, not the captured token: two deferred posts can run out of order, and the
// coalesced knote must never regress the "completed up to" counter (the listener walks
// last+1..cnt, so the highest counter covers every pending batch on the ring).
void prosper_eq_post_apr_token(uint64_t eq, uint64_t eq_identity,
                               int64_t id, uint64_t token) {
    if (!eq_identity || prosper_eq_identity(eq) != eq_identity) return;
    // TWO tag dialects share the H896 binding call, discriminated by the binding's id (a2):
    //   id = 0x74fe+ring (the FAPREventQueueListener channel, #208): tag = (ring<<58)|counter with
    //     a ctor-seeded dense per-ring counter. The listener range-walks last+1..cnt, so the knote
    //     coalesces per ring to the HIGHEST counter ("completed up to" — the kqueue semantics the
    //     walk implements). Handled below with the per-ring HWM.
    //   id = 0 (the IoDispatcher direct channel, #210 — live capture: H896(cb, ioDispatcherEq,
    //     id=0, tag=REQUEST POINTER, 0, 7|0xf)): the tag is an opaque per-request pointer, NOT a
    //     counter. Ring/HWM math on a pointer is nonsense (every post regressed to the max pointer
    //     ever seen, and eq_post's (ident,filter) coalescing replaced still-pending completions —
    //     two of the last three in-flight IoDispatcher reads never completed, which is exactly the
    //     post-first-flip flush-async-loading hang: the packages behind those reads never advanced).
    //     Post the EXACT tag, one distinct queued event per submit, no coalescing.
    // CONFIDENCE: HIGH on the discrimination (both dialects live-captured; id 0x7501 tags are
    // counters, id 0 tags are request pointers in every capture); MED on the id-0 event shape
    // (ident=id=0 kept, guest consumes via GetEventData like the #208 listener).
    if (id == 0) {
        // ORDERED delivery is load-bearing here (issue #232, the DOLL FlushAsyncLoading wall). The
        // ptr-tag is the guest's BATCH object pointer, and the consumer (eboot+0x22aa7d0 ->
        // batch-retire 0x227e8e0) retires its in-flight list FROM THE HEAD *up to* the tagged batch,
        // decrementing the in-flight batch counter [disp+0x30] once per retired node. Submission
        // order == list order, so an event delivered OUT of submission order over-retires: the walk
        // for the earlier batch's late event no longer finds it and marches through batches that are
        // still in flight, driving [disp+0x30] past zero. The tail-flush gate
        // (`if ([disp+0x30] <= 1) flush()` at eboot+0x227e7d3, UNSIGNED compare) then never passes
        // again, the final partial batch never submits, and the GameThread spins in
        // FlushAsyncLoading forever (~16k gettid/s) while every IO thread idles — 0 scene draws.
        // The old per-post detached threads (independent 2 ms sleeps) made cross-post ordering a
        // scheduler coin toss exactly under IO bursts. One FIFO worker + one modeled-latency sleep
        // per batch preserves submission order by construction. CONFIDENCE: HIGH (retire-walk
        // semantics from static disassembly; the stall's log signature — final ReadFile appended,
        // no H896/ASoW after it, all delivered events balanced — matches exactly).
        if (evlog()) fprintf(stderr, "[ev] AprPtrTagComplete tag=0x%llx -> eq=0x%llx scheduled\n",
                             (unsigned long long)token, (unsigned long long)eq);
        struct PtrPost { uint64_t eq, eq_identity, token; };
        struct PtrQueue { std::mutex mx; std::condition_variable cv; std::deque<PtrPost> q; };
        static PtrQueue* pq = new PtrQueue;   // immortal: the worker is detached and outlives exit
        static std::atomic<bool> started{false};
        if (!started.exchange(true)) std::thread([] {
            std::unique_lock<std::mutex> lk(pq->mx);
            for (;;) {
                pq->cv.wait(lk, [] { return !pq->q.empty(); });
                std::deque<PtrPost> batch;
                batch.swap(pq->q);
                lk.unlock();
                struct timespec ts{ 0, 2000000 };   // 2 ms modeled DMA latency (as the #208 path)
                nanosleep(&ts, nullptr);
                for (auto& p : batch) {
                    SceKEvent e{}; e.ident = 0; e.filter = EVFILT_AMPR_MODELED; e.data = (int64_t)p.token;
                    eq_post(p.eq, e, /*coalesce=*/false, p.eq_identity);
                }
                lk.lock();
            }
        }).detach();
        { std::lock_guard<std::mutex> lk(pq->mx); pq->q.push_back({ eq, eq_identity, token }); }
        pq->cv.notify_one();
        return;
    }
    unsigned ring = (unsigned)(token >> 58) & 0x3f;
    uint64_t cnt = token & ((1ull << 58) - 1);
    // A THIRD consumer shares this id != 0 branch, and level coalescing is wrong for it (#1673).
    //
    // The branch's coalescing is justified by the #208 listener's range walk: it consumes
    // `last+1 ..= cnt` and stores `last := cnt`, so one knote carrying the HIGHEST counter retires
    // every batch below it — genuine kqueue "completed up to" level semantics. That justification
    // rests entirely on the tag BEING a counter.
    //
    // CRI ADX2 (cri_ware_unity.prx; Tales of Graces f Remastered PPSA19991, Sonic Origins
    // PPSA05325) binds through the same NID with a tag that is a literal ZERO (`xor ecx,ecx` at
    // cri+0x11b88f) and waits with sceKernelWaitEqueue(..., NULL timeout), retrying until
    // sceKernelGetEventId matches its id (cri+0x11ba65). Its "counter" therefore never advances:
    // the high-water mark stays 0, every completion posts an IDENTICAL (ident, filter, data=0)
    // event, and level coalescing collapses N discrete completions into ONE delivery. A CRI waiter
    // that does not receive one blocks forever — there is no timeout to rescue it and no range walk
    // to make the lost event redundant.
    //
    // So coalesce only when the tag really is a counter, and use the counter itself as the test.
    // cnt == 0 cannot be a legitimate value in the #208 dialect: the guest's listener ctor seeds the
    // per-ring counters at 0x3e8 (1000) with last-processed 0x3e7, counters are dense upward from
    // there, and #180 already forbids posting cnt < 1000 on that channel because the listener's
    // unconditional `last := cnt` store would REGRESS the seed. Every token with a nonzero counter
    // therefore takes the pre-existing path with bit-identical behaviour — this cannot perturb the
    // UE4/IoStore channel, which is the one #180 and #208 constrain.
    //
    // Live shape (PPSA19991, headless boot_trace, PROSPER_AMPRLOG=1 PROSPER_EVLOG=1): one shared
    // equeue, 471 bound submits — 437 on the id == 0 pointer dialect (already non-coalescing since
    // #210) and 34 on this branch, every one of them `token=0x0 (ring=0)`, with 22 sharing id=1 and
    // so sharing one coalescing key.
    //
    // This restores the rule the other two completion sources already follow: a completion is a
    // discrete COUNT, never a level (EOP #234, APR pointer-tag #210). The counter dialect is the
    // single genuine exception, because there the counter value itself carries "completed up to".
    // CONFIDENCE: HIGH on the CRI half (guest disassembly for the zero tag and the ident-only
    // waiter, plus the live token census above); HIGH that the counter dialect is untouched (the
    // predicate is false for every token it can produce).
    const bool counter_dialect = cnt != 0;
    const uint64_t hwm_key = apr_hwm_key(eq_identity, ring);
    {
        AprTokenState& state = apr_token_state();
        std::lock_guard lk(state.mx);
        if (cnt > state.tag_hwm[hwm_key]) state.tag_hwm[hwm_key] = cnt;
    }
    // `id` is on this line because it is the COALESCING KEY (apr_post posts ident = id + ring) and
    // was previously unprintable: joining a completion back to its binding meant guessing from the
    // interleaving of a multi-threaded log, which silently mis-attributes every post.
    if (evlog()) fprintf(stderr, "[ev] AprTagComplete token=0x%llx (ring=%u id=%lld) -> eq=0x%llx scheduled\n",
                         (unsigned long long)token, ring, (long long)id, (unsigned long long)eq);
    std::thread([eq, eq_identity, id, ring, hwm_key, counter_dialect] {
        struct timespec ts{ 0, 2000000 };   // 2 ms
        nanosleep(&ts, nullptr);
        uint64_t hwm;
        {
            AprTokenState& state = apr_token_state();
            std::lock_guard lk(state.mx);
            hwm = state.tag_hwm[hwm_key];
        }
        apr_post(eq, eq_identity, id, ring, ((uint64_t)ring << 58) | hwm, counter_dialect);
    }).detach();
}
// Assign the next completion token for `ring` (0-based, 6 bits) — for UNBOUND submits only, whose
// token the engine consumes through the ASoW out slots / completion record (record-polled; no
// event is ever posted for these — see the block comment above).
uint64_t prosper_apr_next_token(unsigned ring) {
    ring &= 0x3f;
    AprTokenState& state = apr_token_state();
    std::lock_guard lk(state.mx);
    uint64_t seq = ++state.ring_seq[ring];
    return ((uint64_t)ring << 58) | (seq & ((1ull << 58) - 1));
}
// Record an APR completion registration (sSAUCCU1dv4 / H896Pt-yB4I target queue). Registration is
// bookkeeping only: NO catch-up replay, NO ring resets — pre-registration completions are consumed
// by the guest via record polling, and replaying invented counters would regress the listener's
// ctor-seeded per-ring last-processed (see the block comment above; the pre-#208 replay was the
// root cause of the #180 range-walk fault).
void prosper_eq_add_apr(uint64_t eq, int64_t id) {
    AprTokenState& state = apr_token_state();
    std::lock_guard lk(state.mx);
    for (auto& r : state.eq_regs)
        if (r.eq == eq && r.id == id) return;   // idempotent
    state.eq_regs.push_back({ eq, id });
}
HLE(k_add_ampr_event) {   // sceKernelAddAmprEvent(eq, id, udata)
    if (a0) prosper_eq_add_apr(a0, (int64_t)a1);
    if (evlog()) fprintf(stderr, "[ev] AddAmprEvent eq=0x%llx id=%lld udata=0x%llx\n",
        (unsigned long long)a0, (long long)a1, (unsigned long long)a2);
    return 0;
}

// --- SceKernelEvent field accessors (Kyty EventQueue.cpp:318-378: plain field reads). The APR
// listener consumes its events EXCLUSIVELY through sceKernelGetEventData; unimplemented-0 here made
// every event decode as ring 0 / counter 0 (a no-op for the range loop). ---
HLE(k_get_event_data)    { return a0 ? (uint64_t)((const SceKEvent*)P(a0))->data   : 0; }
HLE(k_get_event_id)      { return a0 ? (uint64_t)((const SceKEvent*)P(a0))->ident  : 0; }
HLE(k_get_event_filter)  { return a0 ? (uint64_t)(int64_t)((const SceKEvent*)P(a0))->filter : 0; }
HLE(k_get_event_fflags)  { return a0 ? (uint64_t)((const SceKEvent*)P(a0))->fflags : 0; }
HLE(k_get_event_udata)   { return a0 ? ((const SceKEvent*)P(a0))->udata : 0; }
HLE(k_get_event_error)   { return 0; }

// --- User + timer event sources (sceKernelAddUserEvent / TriggerUserEvent / AddHRTimerEvent /
// AddTimerEvent). Previously no-ops, which starved any equeue the game feeds via these (e.g. Unity's
// FTM queue registers user event id=999 and blocks on it). Now real: registration records the source;
// a trigger (or timer expiry) posts a matching SceKernelEvent so WaitEqueue returns it. FreeBSD-style
// negative filter ids: EVFILT_USER=-11, EVFILT_TIMER=-7 (PS5 SCE_KERNEL_EVFILT_* match FreeBSD). ---
namespace {
    constexpr int16_t EVFILT_USER    = -11;
    constexpr int16_t EVFILT_TIMER   = -7;
    constexpr int16_t EVFILT_HRTIMER = -15;   // Sony-specific: HR timers deliver a distinct filter
    // (UserReg / g_user_regs are declared above, before the vblank pump.)
}
HLE(k_add_user_event) {   // (eq, id, udata?) — register a user event source on the equeue
    { std::lock_guard lk(g_eq_mx); g_user_regs.push_back({ a0, (int64_t)a1, a2 }); }
    if (evlog()) fprintf(stderr, "[ev] AddUserEvent eq=0x%llx id=%lld udata=0x%llx\n",
        (unsigned long long)a0, (long long)a1, (unsigned long long)a2);
    return 0;
}
HLE(k_trigger_user_event) {   // (eq, id, udata) — fire the user event: post it to the equeue
    uint64_t udata = a2;
    { std::lock_guard lk(g_eq_mx);
      for (auto& r : g_user_regs) if (r.eq == a0 && r.id == (int64_t)a1) { if (!udata) udata = r.udata; break; } }
    SceKEvent e{}; e.ident = (int64_t)a1; e.filter = EVFILT_USER; e.udata = udata;
    eq_post(a0, e);
    if (evlog()) fprintf(stderr, "[ev] TriggerUserEvent eq=0x%llx id=%lld udata=0x%llx\n",
        (unsigned long long)a0, (long long)a1, (unsigned long long)udata);
    return 0;
}
// One-shot timer: post an EVFILT_TIMER event to the equeue after `usec` microseconds. The timer is
// CANCELLABLE (#67): registration stores a token in g_timers; the detached thread re-checks it after
// the sleep, so sceKernelDelete(HR)TimerEvent / DeleteEqueue really stop a pending timer (previously
// Delete* were no-ops — a "cancelled" timer still fired, delivering a phantom event whose udata the
// guest may have freed). Posting through eq_post's shared_ptr lookup is safe even if the queue died.
// Post an EVFILT_(HR)TIMER event to the equeue after `usec` microseconds. `periodic` timers re-arm and
// keep firing every `usec` until cancelled (sceKernelAddTimerEvent — matches FreeBSD kqueue EVFILT_TIMER,
// which repeats by default); one-shot timers fire once (sceKernelAddHRTimerEvent). The delivered event's
// `data` carries the running expiration count (kqueue semantics), NOT the interval. Timers are CANCELLABLE
// (#67): registration stores a token in g_timers; the thread re-checks it, so Delete*/DeleteEqueue stop it.
static void post_after(uint64_t eq, int64_t id, uint64_t udata, uint64_t usec, int16_t filter, bool periodic) {
    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    {
        std::lock_guard lk(g_eq_mx);
        auto key = std::make_pair(eq, id);
        auto it = g_timers.find(key);
        if (it != g_timers.end()) it->second.cancelled->store(true);   // re-arm replaces the pending shot
        g_timers[key] = TimerTok{ cancelled };
    }
    std::thread([eq, id, udata, usec, filter, periodic, cancelled]{
        struct timespec ts{ (time_t)(usec / 1000000), (long)((usec % 1000000) * 1000) };
        do {
            nanosleep(&ts, nullptr);
            if (cancelled->load()) break;
            // One expiration per fire. eq_post ACCUMULATES coalesced timer expirations, so the delivered
            // event carries expirations-since-last-read (kqueue EVFILT_TIMER semantics) and delivery clears
            // it -- previously data was a cumulative running total (++count), so a guest that accumulates
            // timer.data (a common fixed-timestep pattern) over-counted without bound after any missed tick.
            SceKEvent e{}; e.ident = id; e.filter = filter; e.data = 1; e.udata = udata;
            eq_post(eq, e);
        } while (periodic && !cancelled->load());
        // forget the registration once we stop firing (only if it is still OUR token, not a re-arm's)
        std::lock_guard lk(g_eq_mx);
        auto it = g_timers.find(std::make_pair(eq, id));
        if (it != g_timers.end() && it->second.cancelled == cancelled) g_timers.erase(it);
    }).detach();
}
// Cancel a pending one-shot timer registered for (eq, id). Shared by both Delete*TimerEvent names.
HLE(k_del_timer_event) {   // (eq, id)
    bool cancelled = false;
    {
        std::lock_guard lk(g_eq_mx);
        auto it = g_timers.find(std::make_pair(a0, (int64_t)a1));
        if (it != g_timers.end()) { it->second.cancelled->store(true); g_timers.erase(it); cancelled = true; }
    }
    if (evlog()) fprintf(stderr, "[ev] DeleteTimerEvent eq=0x%llx id=%lld -> %s\n",
        (unsigned long long)a0, (long long)a1, cancelled ? "cancelled" : "not-pending");
    return 0;
}
// Remove a registered user-event source (previously a no-op: a deleted source kept receiving
// TriggerUserEvent posts and diagnostic-pump heartbeats).
HLE(k_del_user_event) {   // (eq, id)
    std::lock_guard lk(g_eq_mx);
    g_user_regs.erase(std::remove_if(g_user_regs.begin(), g_user_regs.end(),
                      [&](const UserReg& r){ return r.eq == a0 && r.id == (int64_t)a1; }), g_user_regs.end());
    if (evlog()) fprintf(stderr, "[ev] DeleteUserEvent eq=0x%llx id=%lld\n", (unsigned long long)a0, (long long)a1);
    return 0;
}
HLE(k_add_hrtimer_event) {   // (eq, id, SceKernelTimespec* ts, udata) — orbis: 3rd arg is a timespec*
    uint64_t usec = 1000;
    if (a2) { const int64_t* ts = (const int64_t*)P(a2);   // { tv_sec, tv_nsec }
              usec = (uint64_t)ts[0] * 1000000ull + (uint64_t)ts[1] / 1000ull; if (!usec) usec = 1000; }
    post_after(a0, (int64_t)a1, a3, usec, EVFILT_HRTIMER, /*periodic=*/false);   // HR timer is one-shot
    if (evlog()) fprintf(stderr, "[ev] AddHRTimerEvent eq=0x%llx id=%lld usec=%llu\n",
        (unsigned long long)a0, (long long)a1, (unsigned long long)usec);
    return 0;
}
HLE(k_add_timer_event) {   // (eq, id, usec, udata) — coarse timer, same one-shot post
    uint64_t usec = a2 ? a2 : 1000;
    post_after(a0, (int64_t)a1, a3, usec, EVFILT_TIMER, /*periodic=*/true);   // coarse timer repeats
    if (evlog()) fprintf(stderr, "[ev] AddTimerEvent eq=0x%llx id=%lld usec=%llu\n",
        (unsigned long long)a0, (long long)a1, (unsigned long long)usec);
    return 0;
}

// --- libkernel/libScePosix signal + time-conversion surface (#190) ---
// PPSA02664 calls these; previously all fell to the generic unimplemented stub (return 0), which for
// the time-convert pair left the caller's out-param uninitialized (cf. #82 — garbage time_t out).

// Host UTC offset (seconds EAST of UTC) and DST flag for a given unix time, portably. The convert
// functions below approximate the offset by running the HOST timezone over the input instant — the
// same approach shadPS4 takes; it is exact except within the ~1 h wall-clock ambiguity of a DST
// transition, which no title depends on. CONFIDENCE: HIGH for the offset itself.
static void host_local_offset(time_t t, long& gmtoff_sec, int& isdst) {
    struct tm lt{};
#ifdef _WIN32
    localtime_s(&lt, &t);
    // Windows struct tm has no tm_gmtoff: reinterpret the local wall values as UTC and diff. The
    // isdst flag from localtime_s drives the DST-seconds field below.
    time_t as_utc = _mkgmtime64(&lt);
    gmtoff_sec = (long)(as_utc - t);
    isdst = lt.tm_isdst;
#else
    localtime_r(&t, &lt);
    gmtoff_sec = lt.tm_gmtoff;
    isdst = lt.tm_isdst;
#endif
}

// OrbisTimesec out-struct written by both convert functions: { s64 t; u32 west_sec; u32 dst_sec }
// (shadPS4 time_management OrbisTimesec). `west_sec` is seconds WEST of UTC (= -gmtoff); `dst_sec` is
// the DST correction in seconds. CONFIDENCE: MED on the struct's field semantics; the primary
// converted-time out-param (filled first, and what callers actually read) is HIGH.
static void fill_timesec(void* p, int64_t converted, long gmtoff_sec, int isdst) {
    uint8_t* b = (uint8_t*)p;
    int64_t  t = converted;             memcpy(b + 0, &t, 8);
    uint32_t west = (uint32_t)(-gmtoff_sec); memcpy(b + 8, &west, 4);
    uint32_t dst  = (uint32_t)(isdst > 0 ? 3600 : 0); memcpy(b + 12, &dst, 4);
}

// sceKernelConvertUtcToLocaltime(time_t utc, time_t* local_out, OrbisTimesec* st, u64* dst_sec).
// local = utc + gmtoff. CONFIDENCE: MED-HIGH (shadPS4 prototype; primary out-param HIGH).
HLE(k_convert_utc_to_local) {
    time_t utc = (time_t)(int64_t)a0;
    long gmtoff; int isdst; host_local_offset(utc, gmtoff, isdst);
    int64_t local = (int64_t)utc + (int64_t)gmtoff;
    if (a1) *(int64_t*)P(a1) = local;
    if (a2) fill_timesec(P(a2), local, gmtoff, isdst);
    if (a3) *(uint64_t*)P(a3) = (uint64_t)(isdst > 0 ? 3600 : 0);
    return 0;
}

// sceKernelConvertLocaltimeToUtc(time_t local, u64 unk, time_t* utc_out, OrbisTimesec* st, u64* dst).
// utc = local - gmtoff. The 2nd arg is an unused/opaque u64 in the shadPS4 prototype, so the UTC
// result is at a2. CONFIDENCE: MED (prototype incl. the a1 gap; primary out-param HIGH).
HLE(k_convert_local_to_utc) {
    time_t local = (time_t)(int64_t)a0;
    long gmtoff; int isdst; host_local_offset(local, gmtoff, isdst);
    int64_t utc = (int64_t)local - (int64_t)gmtoff;
    if (a2) *(int64_t*)P(a2) = utc;
    if (a3) fill_timesec(P(a3), utc, gmtoff, isdst);
    if (a4) *(uint64_t*)P(a4) = (uint64_t)(isdst > 0 ? 3600 : 0);
    return 0;
}

// pthread_setcancelstate(int state, int* old_state) — Orbis enum 0=ENABLE, 1=DISABLE (Kyty
// PthreadSetcancelstate). prosper never cancels guest threads, so cancellation is a no-op; we validate
// the state, report the POSIX default ENABLE(0) as the previous state, and return OK. (No per-thread
// tracking: a host thread_local here pulls TLS-init machinery into the runtime that perturbs the
// guest-%fs handling enough to break the Messenger boot — and the tracked value is unobservable since
// nothing is ever cancelled.) CONFIDENCE: HIGH.
HLE(k_pthread_setcancelstate) {
    int state = (int)a0;
    // This is registered ONLY as the POSIX `pthread_setcancelstate`, whose contract returns a plain
    // errno — not the 0x80020000-encoded libkernel form its sceKernel siblings use. It returned
    // 0x80020016, so a guest comparing the result against EINVAL never matched (#1612).
    if (state != 0 && state != 1)
        return (uint64_t)static_cast<uint32_t>(prosper::hle::FreeBsdErrno::EInval);   // 22
    if (a1) *(int*)P(a1) = 0;                              // previous state = ENABLE (default)
    return 0;
}

// Signal machinery we do not model (guest runs natively; no guest-directed POSIX signals). Explicit
// no-ops so they resolve here instead of the unimplemented logger. _sigprocmask's callers pass a null
// oldset in the boot path; returning 0 (success, mask unchanged) is the correct no-op. CONFIDENCE: HIGH.
HLE(k_sigprocmask_noop)    { return 0; }
HLE(k_is_signal_return)    { return 0; }   // "is this frame a signal return?" — never, for us

void register_kernel_time_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceKernelCreateEqueue", k_eq_create);   R("sceKernelDeleteEqueue", k_eq_delete);
    R("sceKernelWaitEqueue", k_eq_wait);        R("sceKernelGetEventCount", k_eq_getcount);
    // SceKernelEvent accessors (the APR listener reads its events only through GetEventData)
    R("sceKernelGetEventData", k_get_event_data);     R("sceKernelGetEventId", k_get_event_id);
    R("sceKernelGetEventFilter", k_get_event_filter); R("sceKernelGetEventFflags", k_get_event_fflags);
    R("sceKernelGetEventUserData", k_get_event_udata);R("sceKernelGetEventError", k_get_event_error);
    R("sceKernelAddHRTimerEvent", k_add_hrtimer_event); R("sceKernelAddUserEvent", k_add_user_event);
    R("sceKernelAddAmprEvent", k_add_ampr_event);
    R("sceKernelAddUserEventEdge", k_add_user_event);   R("sceKernelTriggerUserEvent", k_trigger_user_event);
    R("sceKernelDeleteUserEvent", k_del_user_event);   R("sceKernelDeleteHRTimerEvent", k_del_timer_event);
    R("sceKernelDeleteTimerEvent", k_del_timer_event); R("sceKernelAddTimerEvent", k_add_timer_event);
    R("sceKernelGetProcessTimeCounter", k_get_ptc);
    R("sceKernelGetProcessTimeCounterFrequency", k_get_ptc_freq);
    R("sceKernelGetProcessTime", k_get_proc_time);
    R("sceKernelReadTsc", k_read_tsc);
    R("sceKernelGetTscFrequency", k_tsc_freq);
    R("sceKernelClockGettime", k_clock_gettime);
    R("sceKernelUsleep", k_usleep);   R("usleep", k_usleep);
    R("sceKernelSleep", k_sleep_s);   R("sleep", k_sleep_s);
    R("sceKernelNanosleep", k_nanosleep);  R("nanosleep", k_nanosleep);  R("_nanosleep", k_nanosleep);
    R("getpid", k_getpid);
    R("sceKernelGettimezone", k_gettimezone);   // was MISSING -> uninitialized tz out-struct
    R("sceKernelClockGetres", k_clock_getres);  R("clock_getres", k_clock_getres);
    R("clock_settime", k_ok);   R("sceKernelClockSettime", k_ok);   // guest clock is read-only here
    R("clock_gettime", k_clock_gettime);
    R("sceKernelGettimeofday", k_gettimeofday);
    R("gettimeofday", k_gettimeofday);
    R("time", k_time);
    R("clock", k_clock);
    R("sceRtcGetCurrentTick", k_rtc_get_current_tick);
    R("sceRtcGetCurrentNetworkTick", k_rtc_get_current_tick);
    R("sceRtcGetCurrentClockLocalTime", k_rtc_get_clock_localtime); // (dt) — host-local tz
    R("sceRtcGetCurrentClock", k_rtc_get_current_clock);            // (dt, tz_minutes)
    R("sceRtcGetCurrentDateTimeUtc", k_rtc_get_clock_utc);
    R("sceRtcSetTick", k_rtc_set_tick);   // tick -> UTC datetime (issue #115 follow-on: FDateTime spam)
    R("sceRtcGetTick", k_rtc_get_tick);   // UTC datetime -> tick
    // module loading (report success; real PRX are already resident in our address space).
    // IsLoaded is a state QUERY and must not share that stub — see the handlers above (#2002).
    R("sceSysmoduleLoadModule", k_sysmodule_load);
    R("sceSysmoduleUnloadModule", k_sysmodule_unload);
    R("sceSysmoduleIsLoaded", k_sysmodule_is_loaded);
    // The *Internal variants, onto the SAME recording handlers (#2128). The id-taking forms share
    // them directly: LoadModuleInternalWithArg's extra (args, argp, pRes) follow the id in a0, so
    // the recording path is identical and only the ignored tail differs.
    R("sceSysmoduleLoadModuleInternal", k_sysmodule_load);
    R("sceSysmoduleLoadModuleInternalWithArg", k_sysmodule_load);
    R("sceSysmoduleUnloadModuleInternal", k_sysmodule_unload);
    R("sceSysmoduleUnloadModuleInternalWithArg", k_sysmodule_unload);
    R("sceSysmoduleIsLoadedInternal", k_sysmodule_is_loaded);
    // The two the id-keyed map cannot represent, and the one that must never answer 0.
    R("sceSysmoduleLoadModuleByNameInternal", k_sysmodule_by_name_unsupported);
    R("sceSysmoduleUnloadModuleByNameInternal", k_sysmodule_by_name_unsupported);
    R("sceSysmoduleGetModuleHandleInternal", k_sysmodule_get_handle_internal);
#ifndef _WIN32
    R("sceKernelLoadStartModule", k_load_start_mod_entry);   // entry-rsp trampoline (#639)
#else
    R("sceKernelLoadStartModule", k_load_start_mod);
#endif
    R("sceKernelStopUnloadModule", k_ok);
    // Thread scheduling: Set*/Get* are registered in hle_kernel.cpp, where the Get* handlers FILL
    // their out-params (affinity mask 0xff, priority 700) — a Get* that returns success without
    // writing hands the caller uninitialized stack memory. register_kernel_time_hle() runs AFTER
    // register_kernel_hle(), so re-registering scePthreadGet{affinity,prio} to a bare k_ok here
    // (last-write-wins) SILENTLY re-broke exactly that fix. Thread naming also has its real,
    // guest-visible implementation in hle_kernel.cpp; leave the entire family there.
    R("sceKernelUuidCreate", k_uuid_create);
    R("_exit", k_exit);
    R("sceKernelDebugRaiseExceptionOnReleaseMode", k_debug_raise_release);
    R("scePthreadGetthreadid", k_getthreadid);
    R("pthread_getthreadid_np", k_getthreadid);
    // sceKernelAprResolveFilepathsToIdsAndFileSizes is now implemented for real in hle_file.cpp
    // (f_apr_resolve: stat each path, assign an id, record id->host-path). Registered there.
    // C11 threads
    R("_Mtx_init", m_mtx_init);   R("_Mtx_lock", m_mtx_lock);   R("_Mtx_unlock", m_mtx_unlock);
    R("_Mtx_destroy", m_mtx_destroy);
    R("_Cnd_init", m_cnd_init);   R("_Cnd_signal", m_cnd_signal); R("_Cnd_broadcast", m_cnd_broadcast);
    R("_Cnd_wait", m_cnd_wait);   R("_Cnd_destroy", m_cnd_destroy);
    // libkernel/libScePosix signal + time-conversion surface (#190)
    R("sceKernelConvertUtcToLocaltime", k_convert_utc_to_local);
    R("sceKernelConvertLocaltimeToUtc", k_convert_local_to_utc);
    R("pthread_setcancelstate", k_pthread_setcancelstate);
    R("_sigprocmask", k_sigprocmask_noop);
    R("_is_signal_return", k_is_signal_return);
    // select/pselect — pure-sleep shape only (#1660). nid_hash("select") == "T8fER+tIGgk" and
    // nid_hash("pselect") == "ZO2nWoTAv60", matching the PS5 3.20 libkernel/libkernel_web/
    // libkernel_sys export tables and the import PPSA19244 actually binds.
    R("select", k_select);
    R("pselect", k_pselect);
    #undef R
}

} // namespace prosper
