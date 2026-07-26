// hle_kernel_time.cpp — time/clock sources, C11 thread primitives, and assorted
// libkernel stubs the engine needs during init. Cross-platform (chrono + pthread).
#include "dispatch.hpp"
#include "nid.hpp"
#include "hle_kernel_time.hpp"
#include "heap_mutex.hpp"   // #707: keep hot equeue/APR mutexes off macOS __DATA
#include "sync_futex.hpp"
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

// real sleeps so timed wait loops actually yield the CPU (and advance real time)
HLE(k_usleep)   { uint64_t us = a0; struct timespec ts{ (time_t)(us / 1000000), (long)((us % 1000000) * 1000) }; nanosleep(&ts, nullptr); return 0; }
// POSIX sleep() returns the number of seconds LEFT unslept (0 on full completion), not the input.
// Returning the input breaks the canonical resume idiom `while ((left = sleep(left))) ;` into an
// infinite busy-sleep. We always sleep the full duration, so return 0 (Kyty KernelSleep returns OK/0).
HLE(k_sleep_s)  { struct timespec ts{ (time_t)a0, 0 }; nanosleep(&ts, nullptr); return 0; }
HLE(k_nanosleep){ if (a0) nanosleep((const struct timespec*)P(a0), a1 ? (struct timespec*)P(a1) : nullptr); return 0; }
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
// sceKernelLoadStartModule(path, ...): the PRX are pre-linked into our address space, so "loading"
// resolves the path to its linked module and returns a REAL handle — dlsym then consults that
// module's own exports first (#147). A path NOT in the linked set previously got a fake
// monotonically-increasing success handle while loading nothing (#146): the guest then believed the
// load succeeded and called exports that resolved to ESRCH fallbacks / the wrong module instead of
// getting the honest ENOENT. Return SCE_KERNEL_ERROR_ENOENT for a miss so the guest takes its
// module-not-found path. Both current titles preload every PRX they use, so no real load misses.
HLE(k_load_start_mod)  {
    if (uint64_t h = module_handle_for_path(a0 ? (const char*)P(a0) : nullptr)) return h;
    if (getenv("PROSPER_MODLOG"))
        fprintf(stderr, "[loadmod] '%s' not in the linked module set -> ENOENT\n",
                a0 ? (const char*)P(a0) : "(null)");
    return 0x80020002ull;   // SCE_KERNEL_ERROR_ENOENT (a non-preloaded PRX isn't present)
}

// _exit(status): terminate the process. Previously an unimplemented stub RETURNED 0, so libc's
// exit path fell through into its deliberate ud2 (SIGILL) — terminate for real, loudly.
HLE(k_exit) {
    fprintf(stderr, "[prosper] guest _exit(%d) — terminating\n", (int)a0);
    fflush(nullptr);
    _Exit((int)a0);
}
// sceKernelDebugRaiseExceptionOnReleaseMode(code, arg) — the guest's FATAL-error raise (UE4 calls
// it from failed check()s in shipping builds). Report and terminate rather than "return 0" and
// let the guest run on in an undefined state.
HLE(k_debug_raise_release) {
    fprintf(stderr, "[prosper] guest sceKernelDebugRaiseExceptionOnReleaseMode(code=0x%llx, arg=0x%llx) — terminating\n",
            (unsigned long long)a0, (unsigned long long)a1);
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
HLE(m_mtx_lock)   { if (a0 && *(void**)P(a0)) interruptible_mutex_lock((pthread_mutex_t*)*(void**)P(a0)); return 0; }
HLE(m_mtx_unlock) { if (a0 && *(void**)P(a0)) pthread_mutex_unlock((pthread_mutex_t*)*(void**)P(a0)); return 0; }
HLE(m_mtx_destroy){ if (a0 && *(void**)P(a0)) { pthread_mutex_destroy((pthread_mutex_t*)*(void**)P(a0)); free(*(void**)P(a0)); } return 0; }
HLE(m_cnd_init)   { if (a0) { auto* c = (pthread_cond_t*)calloc(1, sizeof(pthread_cond_t)); pthread_cond_init(c, nullptr); *(void**)P(a0) = c; } return 0; }
HLE(m_cnd_signal) { if (a0 && *(void**)P(a0)) interruptible_cond_signal((pthread_cond_t*)*(void**)P(a0)); return 0; }
HLE(m_cnd_broadcast){ if (a0 && *(void**)P(a0)) interruptible_cond_broadcast((pthread_cond_t*)*(void**)P(a0)); return 0; }
HLE(m_cnd_wait)   { if (a0 && *(void**)P(a0) && a1 && *(void**)P(a1)) interruptible_cond_wait((pthread_cond_t*)*(void**)P(a0), (pthread_mutex_t*)*(void**)P(a1)); return 0; }
HLE(m_cnd_destroy){ if (a0 && *(void**)P(a0)) { auto* c = (pthread_cond_t*)*(void**)P(a0); pthread_cond_destroy(c); interruptible_cond_forget(c); free(c); } return 0; }

// --- event queue (sceKernelEqueue): kqueue-like event mechanism the engine uses for vsync/flip
// and async I/O completion. Headless: give a valid queue object; WaitEqueue yields briefly and
// reports no events (callers time out and retry) so nothing busy-spins and no null queue is used. ---
namespace { bool evlog() { static int v = getenv("PROSPER_EVLOG") ? 1 : 0; return v; } }

// --- Real event-queue backend (kqueue/kevent model). Registered flip/vblank sources post events into
// their equeue; WaitEqueue blocks until an event is ready (or timeout) and returns it. A single ~60 Hz
// pump thread drives vblank + flip-completion events so Unity's render/timing threads pace frames. ---
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
        for (;;) {
            struct timespec ts{ 0, 16666667 }; nanosleep(&ts, nullptr);   // ~60 Hz
            frame++;
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
    if (eoplog())
        fprintf(stderr, "[eoprace] t_ns=%llu EOP source registered eq=0x%llx id=%lld "
                        "(regs=%zu zero-consumer-completions=%llu)\n",
                (unsigned long long)registration_ns, (unsigned long long)eq, (long long)id,
                registration_count, (unsigned long long)zero_consumer_count);
}
// Fire the registered EOP events — called when a submit completes. Posts TriggerEvent(ident=id,
// filter=GraphicsCore, data=id, udata) to each registered equeue, matching shadPS4's IRQ handler.
// Inert if none registered.
// GPU pipe-drain completion-write drain (src/gpu/command_processor.cpp, #312): an EOP EVENT must
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
        for (auto& r : regs) {
            SceKEvent e{}; e.ident = r.ident; e.filter = EVFILT_GRAPHICS_CORE; e.data = r.ident; e.udata = r.udata;
            eq_post(r.eq, e, coalesce);
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
    if (evlog()) fprintf(stderr, "[ev] WaitEqueue eq=0x%llx num=%llu timeout=%s ra=eboot+0x%llx\n",
        (unsigned long long)a0, (unsigned long long)a2, a4 ? "yes" : "inf",
        (unsigned long long)((uint64_t)__builtin_return_address(0) - 0x400000000ull));
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
                const uint8_t* code = (const uint8_t*)(uintptr_t)(0x400000000ull + off);
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
                if (v < 0x400000000ull || v >= 0x4c0000000ull) continue;   // eboot / IL2CPP executable range
                // A genuine caller is a return address preceded by a call: `call rel32` (0xe8 at v-5,
                // any target — the import call goes through an in-eboot thunk, NOT straight to the
                // stub region, so no target filter) or an indirect `call` (0xff at v-6/-3/-2).
                const uint8_t* pre = (const uint8_t*)(uintptr_t)(v - 8);
                bool is_call = pre[3] == 0xe8 ||                            // call rel32 at v-5
                               pre[2] == 0xff || pre[5] == 0xff || pre[6] == 0xff;  // call r/m64 forms
                if (!is_call) continue;
                fprintf(stderr, "[waitcaller] eq=0x%llx num=%llu stack[%d] eboot+0x%llx | code@ra-0x18:",
                        (unsigned long long)a0, (unsigned long long)a2, i,
                        (unsigned long long)(v - 0x400000000ull));
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
            static const uint64_t cap = [] {   // parsed once (cf. punch_secs in hle_kernel_mem.cpp)
                const char* e = getenv("PROSPER_WAITCAP"); return e ? (uint64_t)atoll(e) : 0; }();
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
    if (evlog() && n > 0) fprintf(stderr, "[ev]   -> delivered %d ev(s) eq=0x%llx ra=eboot+0x%llx (ident=%lld filter=%d)\n",
        n, (unsigned long long)a0, (unsigned long long)((uint64_t)__builtin_return_address(0) - 0x400000000ull),
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
// CONFIDENCE: HIGH — ctor seeding, walk bounds, unconditional last:=cnt store, handler match/hash/
// null-entry paths all from static disassembly; tag-echo event consumption live-verified (#180).
namespace {
    constexpr int16_t EVFILT_AMPR_MODELED = -24;   // guest never reads filter; distinct on purpose
    struct AprEqReg { uint64_t eq; int64_t id; };
    // Own mutex (NOT g_eq_mx): the post path calls eq_post/eq_find, which lock g_eq_mx themselves.
    PROSPER_HEAP_MUTEX(g_apr_mx);   // #707: heap-backed on macOS (hot APR mutex in the corrupted __DATA cluster)
    std::vector<AprEqReg> g_apr_eq_regs;               // guarded by g_apr_mx (registration log)
    uint64_t g_apr_ring_seq[64] = {};                  // per-ring counters for prosper-issued tokens
    uint64_t g_apr_tag_hwm[64] = {};                   // per-ring highest tag counter posted (guarded)
    void apr_post(uint64_t eq, uint64_t eq_identity, int64_t id,
                  unsigned ring, uint64_t token) {   // no APR lock held
        SceKEvent e{}; e.ident = id + (int64_t)ring; e.filter = EVFILT_AMPR_MODELED;
        e.data = (int64_t)token; e.udata = 0;
        eq_post(eq, e, /*coalesce=*/true, eq_identity);
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
    if (!eq_identity) return;
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
    {
        std::lock_guard lk(g_apr_mx);
        if (cnt > g_apr_tag_hwm[ring]) g_apr_tag_hwm[ring] = cnt;
    }
    if (evlog()) fprintf(stderr, "[ev] AprTagComplete token=0x%llx (ring=%u) -> eq=0x%llx scheduled\n",
                         (unsigned long long)token, ring, (unsigned long long)eq);
    std::thread([eq, eq_identity, id, ring] {
        struct timespec ts{ 0, 2000000 };   // 2 ms
        nanosleep(&ts, nullptr);
        uint64_t hwm;
        { std::lock_guard lk(g_apr_mx); hwm = g_apr_tag_hwm[ring]; }
        apr_post(eq, eq_identity, id, ring, ((uint64_t)ring << 58) | hwm);
    }).detach();
}
// Assign the next completion token for `ring` (0-based, 6 bits) — for UNBOUND submits only, whose
// token the engine consumes through the ASoW out slots / completion record (record-polled; no
// event is ever posted for these — see the block comment above).
uint64_t prosper_apr_next_token(unsigned ring) {
    ring &= 0x3f;
    std::lock_guard lk(g_apr_mx);
    uint64_t seq = ++g_apr_ring_seq[ring];
    return ((uint64_t)ring << 58) | (seq & ((1ull << 58) - 1));
}
// Record an APR completion registration (sSAUCCU1dv4 / H896Pt-yB4I target queue). Registration is
// bookkeeping only: NO catch-up replay, NO ring resets — pre-registration completions are consumed
// by the guest via record polling, and replaying invented counters would regress the listener's
// ctor-seeded per-ring last-processed (see the block comment above; the pre-#208 replay was the
// root cause of the #180 range-walk fault).
void prosper_eq_add_apr(uint64_t eq, int64_t id) {
    std::lock_guard lk(g_apr_mx);
    for (auto& r : g_apr_eq_regs)
        if (r.eq == eq && r.id == id) return;   // idempotent
    g_apr_eq_regs.push_back({ eq, id });
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
    if (state != 0 && state != 1) return 0x80020016ull;   // EINVAL
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
    // module loading (report success; real PRX are already resident in our address space)
    R("sceSysmoduleLoadModule", k_ok);
    R("sceSysmoduleUnloadModule", k_ok);
    R("sceSysmoduleIsLoaded", k_ok);
    R("sceKernelLoadStartModule", k_load_start_mod);
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
    #undef R
}

} // namespace prosper
