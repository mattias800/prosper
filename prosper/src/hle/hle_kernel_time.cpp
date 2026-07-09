// hle_kernel_time.cpp — time/clock sources, C11 thread primitives, and assorted
// libkernel stubs the engine needs during init. Cross-platform (chrono + pthread).
#include "dispatch.hpp"
#include "nid.hpp"
#include <pthread.h>
#include <chrono>
#ifndef _WIN32
#include <unistd.h>
#include <sys/syscall.h>
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

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define P(x) ((void*)(uintptr_t)(x))

namespace {
    using clk = std::chrono::steady_clock;
    clk::time_point g_start = clk::now();
    uint64_t ns_now() { return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - g_start).count(); }
    std::atomic<uint64_t> g_module_handle{1};

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
            uint64_t mono = ns_now();
            uint64_t us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::system_clock::now().time_since_epoch()).count();
            return WallAnchor{ us, mono };
        }();
        return a;
    }
    uint64_t wall_now_us() {   // microseconds since the unix epoch, monotonically advancing
        const WallAnchor& a = wall_anchor();
        return a.base_us + (ns_now() - a.mono_ns) / 1000ull;
    }
    // RTC epoch: SceRtcTick counts microseconds since 0001-01-01 00:00:00 UTC; the unix epoch is
    // 62135596800 s after it (the documented Orbis RTC convention, also shadPS4 UNIX_EPOCH_TICKS).
    constexpr uint64_t kRtcUnixEpochOffsetUs = 62135596800ull * 1000000ull;
}

// --- time / clock (return real, advancing time so wait-for-time loops progress) ---
HLE(k_get_ptc)        { return ns_now(); }                 // sceKernelGetProcessTimeCounter
HLE(k_get_ptc_freq)   { return 1000000000ull; }            // counter is in ns -> 1 GHz
HLE(k_get_proc_time)  { return ns_now() / 1000; }          // microseconds
HLE(k_read_tsc)       { return ns_now(); }
HLE(k_tsc_freq)       { return 1000000000ull; }
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
HLE(k_sleep_s)  { struct timespec ts{ (time_t)a0, 0 }; nanosleep(&ts, nullptr); return (uint64_t)a0; }
HLE(k_nanosleep){ if (a0) nanosleep((const struct timespec*)P(a0), a1 ? (struct timespec*)P(a1) : nullptr); return 0; }

// --- assorted libkernel stubs ---
HLE(k_ok)              { return 0; }                       // generic success no-op
// sceKernelLoadStartModule(path, ...): the PRX are pre-linked into our address space, so
// "loading" resolves the path to its linked module and returns a REAL handle — dlsym then
// consults that module's own exports first (#147). An unknown path still gets the synthetic
// success counter (that behavior is #146's scope) so optional plugins don't fail the boot.
HLE(k_load_start_mod)  {
    if (uint64_t h = module_handle_for_path(a0 ? (const char*)P(a0) : nullptr)) return h;
    return g_module_handle++;
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
#ifndef _WIN32
HLE(k_getthreadid) { return (uint64_t)syscall(SYS_gettid); }   // scePthreadGetthreadid
#else
HLE(k_getthreadid) { return 1; }
#endif
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
HLE(m_mtx_lock)   { if (a0 && *(void**)P(a0)) pthread_mutex_lock((pthread_mutex_t*)*(void**)P(a0)); return 0; }
HLE(m_mtx_unlock) { if (a0 && *(void**)P(a0)) pthread_mutex_unlock((pthread_mutex_t*)*(void**)P(a0)); return 0; }
HLE(m_mtx_destroy){ if (a0 && *(void**)P(a0)) { pthread_mutex_destroy((pthread_mutex_t*)*(void**)P(a0)); free(*(void**)P(a0)); } return 0; }
HLE(m_cnd_init)   { if (a0) { auto* c = (pthread_cond_t*)calloc(1, sizeof(pthread_cond_t)); pthread_cond_init(c, nullptr); *(void**)P(a0) = c; } return 0; }
HLE(m_cnd_signal) { if (a0 && *(void**)P(a0)) pthread_cond_signal((pthread_cond_t*)*(void**)P(a0)); return 0; }
HLE(m_cnd_broadcast){ if (a0 && *(void**)P(a0)) pthread_cond_broadcast((pthread_cond_t*)*(void**)P(a0)); return 0; }
HLE(m_cnd_wait)   { if (a0 && *(void**)P(a0) && a1 && *(void**)P(a1)) pthread_cond_wait((pthread_cond_t*)*(void**)P(a0), (pthread_mutex_t*)*(void**)P(a1)); return 0; }
HLE(m_cnd_destroy){ if (a0 && *(void**)P(a0)) { pthread_cond_destroy((pthread_cond_t*)*(void**)P(a0)); free(*(void**)P(a0)); } return 0; }

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
    struct EqState { std::mutex m; std::condition_variable cv; std::deque<SceKEvent> ready; bool deleted = false; };
    std::mutex g_eq_mx;
    std::unordered_map<uint64_t, std::shared_ptr<EqState>> g_eqs;   // guest eq handle -> state
    struct FlipReg { uint64_t eq; int64_t ident; uint64_t udata; };
    std::vector<FlipReg> g_flip_regs, g_vblank_regs;
    // GPU end-of-pipe (EOP) event sources registered via sceGnmAddEqEvent / GraphicsAddEqEvent
    // (NID b0xyllnVY-I). Mirrors shadPS4 sceGnmAddEqEvent: on submit completion the GPU interrupt
    // triggers TriggerEvent(ident=id, filter=GraphicsCore, data=id, udata). id is GfxEop=0x40 (or a
    // ComputeN ring id). Filter GraphicsCore=-14 (shadPS4 equeue.h).
    constexpr int16_t EVFILT_GRAPHICS_CORE = -14;
    std::vector<FlipReg> g_eop_regs;   // same (eq,id,udata) shape
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

    std::shared_ptr<EqState> eq_find(uint64_t eq) {
        std::lock_guard<std::mutex> lk(g_eq_mx);
        auto it = g_eqs.find(eq);
        return it == g_eqs.end() ? nullptr : it->second;
    }
    void eq_post(uint64_t eq, const SceKEvent& e) {
        auto s = eq_find(eq); if (!s) return;
        std::lock_guard<std::mutex> lk(s->m);
        if (s->deleted) return;
        // kqueue semantics: one knote per (ident, filter) — a re-trigger UPDATES the pending event
        // in place (fresh data/fflags/udata) instead of queuing a duplicate. This is what makes the
        // 60 Hz vblank pump safe: previously a fixed 4-entry cap dropped the NEWEST event once the
        // queue filled with pumped vblanks — and the dropped one could be the flip-completion event
        // carrying the exact flipArg the game's pacer was waiting to observe (frame-pacing stall).
        for (auto& q : s->ready)
            if (q.ident == e.ident && q.filter == e.filter) { q = e; s->cv.notify_all(); return; }
        // Distinct (ident, filter) pairs are few; the cap is a leak guard only. If it ever fires,
        // shed the OLDEST event — never the just-posted one.
        if (s->ready.size() >= 64) s->ready.pop_front();
        s->ready.push_back(e);
        s->cv.notify_all();
    }
    void vblank_pump() {
        uint64_t frame = 0;
        for (;;) {
            struct timespec ts{ 0, 16666667 }; nanosleep(&ts, nullptr);   // ~60 Hz
            frame++;
            std::vector<FlipReg> vr;
            { std::lock_guard<std::mutex> lk(g_eq_mx); vr = g_vblank_regs; }
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
                std::vector<UserReg> ur; { std::lock_guard<std::mutex> lk(g_eq_mx); ur = g_user_regs; }
                for (auto& r : ur) { SceKEvent e{}; e.ident = r.id; e.filter = -11 /*EVFILT_USER*/; e.udata = r.udata; eq_post(r.eq, e); }
            }
        }
    }
    void ensure_pump() { if (!g_pump_started.exchange(true)) std::thread(vblank_pump).detach(); }
}

// Exposed to hle_graphics.cpp (sceVideoOut* flip/vblank event registration).
void prosper_eq_add_flip(uint64_t eq, int64_t ident, uint64_t udata) {
    { std::lock_guard<std::mutex> lk(g_eq_mx); g_flip_regs.push_back({ eq, ident, udata }); }
    ensure_pump();
}
// Fire the flip-completion event on every registered flip equeue — called by BOTH flip paths
// (sceVideoOutSubmitFlip and the in-stream Dcb SetFlip) at the flip moment. Kevent shape per Kyty
// flip_event_trigger_func: ident=VIDEO_OUT_EVENT_FLIP, data=the completed flip's flipArg.
void prosper_eq_trigger_flip(int64_t flip_arg) {
    std::vector<FlipReg> regs;
    { std::lock_guard<std::mutex> lk(g_eq_mx); regs = g_flip_regs; }
    for (auto& r : regs) {
        SceKEvent e{}; e.ident = VIDEO_OUT_EVENT_FLIP; e.filter = EVFILT_VIDEO_OUT;
        e.fflags = 1; e.data = flip_arg; e.udata = r.udata;
        eq_post(r.eq, e);
    }
}
void prosper_eq_add_vblank(uint64_t eq, int64_t ident, uint64_t udata) {
    { std::lock_guard<std::mutex> lk(g_eq_mx); g_vblank_regs.push_back({ eq, ident, udata }); }
    ensure_pump();
}
// Exposed to the AGC submit path (hle_agc.cpp). Register a GPU EOP event source (sceGnmAddEqEvent).
void prosper_eq_add_eop(uint64_t eq, int64_t id, uint64_t udata) {
    std::lock_guard<std::mutex> lk(g_eq_mx); g_eop_regs.push_back({ eq, id, udata });
}
// Fire the registered EOP events — called when a submit completes (our fold is synchronous, so submit
// == GPU pipe drain == the EOP interrupt moment). Posts TriggerEvent(ident=id, filter=GraphicsCore,
// data=id, udata) to each registered equeue, matching shadPS4's IRQ handler. Inert if none registered.
void prosper_eq_trigger_eop() {
    std::vector<FlipReg> regs;
    { std::lock_guard<std::mutex> lk(g_eq_mx); regs = g_eop_regs; }
    for (auto& r : regs) {
        SceKEvent e{}; e.ident = r.ident; e.filter = EVFILT_GRAPHICS_CORE; e.data = r.ident; e.udata = r.udata;
        eq_post(r.eq, e);
    }
}

HLE(k_eq_create) {
    auto s = std::make_shared<EqState>();
    if (a0) *(void**)P(a0) = (void*)s.get();   // the guest's opaque SceKernelEqueue handle IS our state ptr
    { std::lock_guard<std::mutex> lk(g_eq_mx); g_eqs[(uint64_t)(uintptr_t)s.get()] = s; }
    if (evlog()) fprintf(stderr, "[ev] CreateEqueue -> eq=%p name=%s\n", (void*)s.get(), a1 ? (const char*)P(a1) : "");
    return 0;
}
HLE(k_eq_delete) {
    if (!a0) return 0;
    std::shared_ptr<EqState> s;
    {
        std::lock_guard<std::mutex> lk(g_eq_mx);
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
    auto s = eq_find(a0);
    int num = (int)a2; if (num < 1) num = 1;
    // Unknown/deleted queue: the real API errors (Kyty EventQueue.cpp returns KERNEL_ERROR_EBADF);
    // the old success-with-0-events reply made the caller consume a never-written event struct.
    if (!s) { if (a3) *(int32_t*)P(a3) = 0; return 0x80020009ull; }   // KERNEL_ERROR_EBADF
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
    while (n < num && !s->ready.empty()) { if (ev) ev[n] = s->ready.front(); s->ready.pop_front(); n++; }
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

// --- APR (Ampr file-read engine) completion events — issue #115. ------------------------------
// UE4's PS5 platform file layer runs an "FAPREventQueueListener" thread that blocks in
// WaitEqueue(APREventQueue, evs, 15, ...) and decodes each event with sceKernelGetEventData():
//   data = (ring_index << 58) | completion_counter      (live-disassembled at eboot+0x4022740b0:
//   r15 = data >> 0x3a; cnt = bzhi(data, 0x3a); per-ring "last processed" at listener+0xc8+ring*40;
//   for seq in last+1..cnt: handler(ctx, (ring<<58)|seq) — eboot+0x40229dcb0 matches the token
//   against the tracking object's +0x10 slot, else a hash-map keyed by the full token.)
// The token the engine stores at submit time comes OUT of the submit call (ASoW5WE-UPo writes it
// through its two out-pointers; the vWU-odnS+fU direct read returns it). prosper's APR reads are
// synchronous, so "submit" == "complete": we assign the token and immediately post the event.
// Event shape: guest code reads ONLY data (GetEventData) — ident/filter are never inspected — but
// eq_post coalesces on (ident,filter), so ident carries the ring index to keep concurrent rings'
// completions from replacing each other (per-ring coalescing to the HIGHEST counter is exactly the
// kqueue/"completed up to" semantics the listener's range loop implements).
// CONFIDENCE: HIGH on the data encoding + listener decode (disassembly); MED on registration arg
// order (live capture: sSAUCCU1dv4(eq, id=0x7502, 0, 0, 0x43, 0), H896Pt-yB4I(ctx, eq, id, ...)).
namespace {
    constexpr int16_t EVFILT_AMPR_MODELED = -24;   // guest never reads filter; distinct on purpose
    struct AprEqReg { uint64_t eq; int64_t id; };
    // Own mutex (NOT g_eq_mx): the post path calls eq_post/eq_find, which lock g_eq_mx themselves.
    std::mutex g_apr_mx;
    std::vector<AprEqReg> g_apr_eq_regs;               // guarded by g_apr_mx
    uint64_t g_apr_ring_seq[64]     = {};              // per-ring completion counters (guarded)
    bool     g_apr_ring_catchup[64] = {};              // ring had a TRACKED pre-registration read
    void apr_post(const AprEqReg& r, unsigned ring, uint64_t token) {   // no APR lock held
        SceKEvent e{}; e.ident = r.id + (int64_t)ring; e.filter = EVFILT_AMPR_MODELED;
        e.data = (int64_t)token; e.udata = 0;
        eq_post(r.eq, e);
    }
}
// Assign the next completion token for `ring` (0-based, 6 bits). Called by the APR submit paths.
// `tracked_catchup`: the caller's completion must survive a registration that happens AFTER the
// submit (the vWU direct read fires 2 calls before sSAUCCU1dv4 registers the queue). Untracked
// submissions (the mount-era synchronous ASoW flow, consumed by polling the completion record)
// must NOT replay at registration: the listener's range loop would hand their seqs to the
// completion handler, whose hash-miss path is not null-tolerant (live crash: phantom ring-5 seqs
// 1..13 -> vmovups from 0x10 at eboot+0x229df3e). Their rings reset to 0 instead, so the first
// post-registration submission is seq 1 — matching the listener's zero-initialized per-ring
// "last processed" counter. CONFIDENCE: MED-HIGH (crash-verified both ways).
uint64_t prosper_apr_next_token(unsigned ring, bool tracked_catchup) {
    ring &= 0x3f;
    std::lock_guard<std::mutex> lk(g_apr_mx);
    uint64_t seq = ++g_apr_ring_seq[ring];
    if (tracked_catchup && g_apr_eq_regs.empty()) g_apr_ring_catchup[ring] = true;
    return ((uint64_t)ring << 58) | (seq & ((1ull << 58) - 1));
}
// Post the completion event for `token`'s ring to every registered APR event queue — DEFERRED by
// ~2 ms on a detached thread. Posting synchronously inside the submit loses the race the real
// hardware never runs: the engine inserts its {token -> request} hash entry right AFTER the
// submit call returns, and the listener's hash-MISS path is not tolerant (it copies from
// entry 0 + 0x10 — the live eboot+0x229df3e fault). A real DMA read takes far longer than the
// submitter's few bookkeeping instructions; the delay models that latency. The deferred post
// reads the ring's counter AT POST TIME (not the captured token), so out-of-order wakeups can
// never regress the coalesced event's "completed up to" counter. CONFIDENCE: MED (latency model;
// the guest-visible contract — event data = (ring<<58)|counter after the tracking insert — is
// crash-verified in both failure modes).
namespace {
void apr_schedule_post(unsigned ring) {
    std::thread([ring] {
        struct timespec ts{ 0, 2000000 };   // 2 ms
        nanosleep(&ts, nullptr);
        std::vector<AprEqReg> regs; uint64_t cur;
        {
            std::lock_guard<std::mutex> lk(g_apr_mx);
            regs = g_apr_eq_regs;
            cur  = g_apr_ring_seq[ring];
        }
        if (!cur) return;
        uint64_t tok = ((uint64_t)ring << 58) | (cur & ((1ull << 58) - 1));
        for (auto& r : regs) apr_post(r, ring, tok);
        if (evlog()) fprintf(stderr, "[ev] AprComplete posted ring=%u upto=%llu -> %zu eq(s)\n",
            ring, (unsigned long long)cur, regs.size());
    }).detach();
}
}
void prosper_eq_trigger_apr(uint64_t token) {
    unsigned ring = (unsigned)(token >> 58) & 0x3f;
    if (evlog()) fprintf(stderr, "[ev] AprComplete token=0x%llx (ring=%u seq=%llu) scheduled\n",
        (unsigned long long)token, ring, (unsigned long long)(token & ((1ull << 58) - 1)));
    apr_schedule_post(ring);
}
// Register an APR completion target. Tracked pre-registration completions (vWU direct reads) are
// re-delivered so their listener wake-up isn't lost; untracked rings reset (see above).
void prosper_eq_add_apr(uint64_t eq, int64_t id) {
    bool pending[64] = {};
    {
        std::lock_guard<std::mutex> lk(g_apr_mx);
        for (auto& r : g_apr_eq_regs) if (r.eq == eq && r.id == id) return;   // idempotent
        bool first = g_apr_eq_regs.empty();
        g_apr_eq_regs.push_back({ eq, id });
        for (unsigned ring = 0; ring < 64; ring++) {
            if (!g_apr_ring_seq[ring]) continue;
            if (g_apr_ring_catchup[ring]) pending[ring] = true;
            else if (first)               g_apr_ring_seq[ring] = 0;   // drop untracked history
        }
    }
    // Deferred like every completion post (the engine may still be inserting its tracking entry
    // for the pre-registration read when this registration call runs).
    for (unsigned ring = 0; ring < 64; ring++)
        if (pending[ring]) apr_schedule_post(ring);
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
    constexpr int16_t EVFILT_USER  = -11;
    constexpr int16_t EVFILT_TIMER = -7;
    // (UserReg / g_user_regs are declared above, before the vblank pump.)
}
HLE(k_add_user_event) {   // (eq, id, udata?) — register a user event source on the equeue
    { std::lock_guard<std::mutex> lk(g_eq_mx); g_user_regs.push_back({ a0, (int64_t)a1, a2 }); }
    if (evlog()) fprintf(stderr, "[ev] AddUserEvent eq=0x%llx id=%lld udata=0x%llx\n",
        (unsigned long long)a0, (long long)a1, (unsigned long long)a2);
    return 0;
}
HLE(k_trigger_user_event) {   // (eq, id, udata) — fire the user event: post it to the equeue
    uint64_t udata = a2;
    { std::lock_guard<std::mutex> lk(g_eq_mx);
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
static void post_after(uint64_t eq, int64_t id, uint64_t udata, uint64_t usec, int16_t filter) {
    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    {
        std::lock_guard<std::mutex> lk(g_eq_mx);
        auto key = std::make_pair(eq, id);
        auto it = g_timers.find(key);
        if (it != g_timers.end()) it->second.cancelled->store(true);   // re-arm replaces the pending shot
        g_timers[key] = TimerTok{ cancelled };
    }
    std::thread([eq, id, udata, usec, filter, cancelled]{
        struct timespec ts{ (time_t)(usec / 1000000), (long)((usec % 1000000) * 1000) };
        nanosleep(&ts, nullptr);
        if (cancelled->load()) return;
        {   // one-shot: forget the registration (only if it is still OUR token, not a re-arm's)
            std::lock_guard<std::mutex> lk(g_eq_mx);
            auto it = g_timers.find(std::make_pair(eq, id));
            if (it != g_timers.end() && it->second.cancelled == cancelled) g_timers.erase(it);
        }
        SceKEvent e{}; e.ident = id; e.filter = filter; e.data = (int64_t)usec; e.udata = udata;
        eq_post(eq, e);
    }).detach();
}
// Cancel a pending one-shot timer registered for (eq, id). Shared by both Delete*TimerEvent names.
HLE(k_del_timer_event) {   // (eq, id)
    bool cancelled = false;
    {
        std::lock_guard<std::mutex> lk(g_eq_mx);
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
    std::lock_guard<std::mutex> lk(g_eq_mx);
    g_user_regs.erase(std::remove_if(g_user_regs.begin(), g_user_regs.end(),
                      [&](const UserReg& r){ return r.eq == a0 && r.id == (int64_t)a1; }), g_user_regs.end());
    if (evlog()) fprintf(stderr, "[ev] DeleteUserEvent eq=0x%llx id=%lld\n", (unsigned long long)a0, (long long)a1);
    return 0;
}
HLE(k_add_hrtimer_event) {   // (eq, id, SceKernelTimespec* ts, udata) — orbis: 3rd arg is a timespec*
    uint64_t usec = 1000;
    if (a2) { const int64_t* ts = (const int64_t*)P(a2);   // { tv_sec, tv_nsec }
              usec = (uint64_t)ts[0] * 1000000ull + (uint64_t)ts[1] / 1000ull; if (!usec) usec = 1000; }
    post_after(a0, (int64_t)a1, a3, usec, EVFILT_TIMER);
    if (evlog()) fprintf(stderr, "[ev] AddHRTimerEvent eq=0x%llx id=%lld usec=%llu\n",
        (unsigned long long)a0, (long long)a1, (unsigned long long)usec);
    return 0;
}
HLE(k_add_timer_event) {   // (eq, id, usec, udata) — coarse timer, same one-shot post
    uint64_t usec = a2 ? a2 : 1000;
    post_after(a0, (int64_t)a1, a3, usec, EVFILT_TIMER);
    if (evlog()) fprintf(stderr, "[ev] AddTimerEvent eq=0x%llx id=%lld usec=%llu\n",
        (unsigned long long)a0, (long long)a1, (unsigned long long)usec);
    return 0;
}

void register_kernel_time_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceKernelCreateEqueue", k_eq_create);   R("sceKernelDeleteEqueue", k_eq_delete);
    R("sceKernelWaitEqueue", k_eq_wait);        R("sceKernelGetEventCount", k_eq_getcount);
    // SceKernelEvent accessors (the APR listener reads its events only through GetEventData)
    R("sceKernelGetEventData", k_get_event_data);     R("sceKernelGetEventId", k_get_event_id);
    R("sceKernelGetEventFilter", k_get_event_filter); R("sceKernelGetEventFflags", k_get_event_fflags);
    R("sceKernelGetEventUserData", k_get_event_udata);R("sceKernelGetEventError", k_get_event_error);
    R("sceKernelAddHRTimerEvent", k_add_hrtimer_event); R("sceKernelAddUserEvent", k_add_user_event);
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
    R("sceKernelNanosleep", k_nanosleep);  R("nanosleep", k_nanosleep);
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
    // thread scheduling hints — safe no-ops
    R("scePthreadSetaffinity", k_ok);
    R("scePthreadGetaffinity", k_ok);
    R("scePthreadSetprio", k_ok);
    R("scePthreadGetprio", k_ok);
    R("scePthreadSetschedparam", k_ok);
    R("scePthreadRename", k_ok);
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
    #undef R
}

} // namespace prosper
