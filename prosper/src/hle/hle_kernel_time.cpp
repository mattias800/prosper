// hle_kernel_time.cpp — time/clock sources, C11 thread primitives, and assorted
// libkernel stubs the engine needs during init. Cross-platform (chrono + pthread).
#include "dispatch.hpp"
#include "nid.hpp"
#include <pthread.h>
#include <chrono>
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

namespace prosper {

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define P(x) ((void*)(uintptr_t)(x))

namespace {
    using clk = std::chrono::steady_clock;
    clk::time_point g_start = clk::now();
    uint64_t ns_now() { return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - g_start).count(); }
    std::atomic<uint64_t> g_module_handle{1};
}

// --- time / clock (return real, advancing time so wait-for-time loops progress) ---
HLE(k_get_ptc)        { return ns_now(); }                 // sceKernelGetProcessTimeCounter
HLE(k_get_ptc_freq)   { return 1000000000ull; }            // counter is in ns -> 1 GHz
HLE(k_get_proc_time)  { return ns_now() / 1000; }          // microseconds
HLE(k_read_tsc)       { return ns_now(); }
HLE(k_tsc_freq)       { return 1000000000ull; }
HLE(k_clock_gettime) {                                     // (clockid, struct timespec*)
    if (!a1) return 0;
    uint64_t ns = ns_now();
    ((int64_t*)P(a1))[0] = (int64_t)(ns / 1000000000ull);   // tv_sec
    ((int64_t*)P(a1))[1] = (int64_t)(ns % 1000000000ull);   // tv_nsec
    return 0;
}
HLE(k_gettimeofday) {                                      // (struct timeval*, tz*)
    if (!a0) return 0;
    uint64_t us = ns_now() / 1000;
    ((int64_t*)P(a0))[0] = (int64_t)(us / 1000000ull);      // tv_sec
    ((int64_t*)P(a0))[1] = (int64_t)(us % 1000000ull);      // tv_usec
    return 0;
}
HLE(k_time) { uint64_t s = ns_now() / 1000000000ull + 1700000000ull; if (a0) *(int64_t*)P(a0) = (int64_t)s; return s; }
HLE(k_clock) { return ns_now() / 1000; }   // clock(): CLOCKS_PER_SEC=1e6 -> microseconds
// sceRtcGetCurrentTick(SceRtcTick* tick): a SceRtcTick is a single u64 = microseconds since the Sony RTC
// epoch 0001-01-01 00:00:00 UTC. Was unimplemented (→0), which zeroes the game's wall-clock and skews any
// timestamp/interval math derived from it. Return a real, monotonically-advancing tick built from our
// synthetic unix time (base 1700000000 ≈ 2023-11) + the RTC↔unix epoch offset (62135596800 s).
// CONFIDENCE: MED — SceRtcTick = bare u64 µs and the 62135596800 s offset are the documented Orbis RTC
// convention; the wall-clock base is synthetic (we don't read the host RTC) but advances correctly.
HLE(k_rtc_get_current_tick) {
    if (!a0) return 0;
    uint64_t unix_us = (ns_now() / 1000ull) + 1700000000ull * 1000000ull;
    uint64_t tick = unix_us + 62135596800ull * 1000000ull;   // shift to the 0001-01-01 RTC epoch
    *(uint64_t*)P(a0) = tick;
    return 0;
}
// real sleeps so timed wait loops actually yield the CPU (and advance real time)
HLE(k_usleep)   { uint64_t us = a0; struct timespec ts{ (time_t)(us / 1000000), (long)((us % 1000000) * 1000) }; nanosleep(&ts, nullptr); return 0; }
HLE(k_sleep_s)  { struct timespec ts{ (time_t)a0, 0 }; nanosleep(&ts, nullptr); return (uint64_t)a0; }
HLE(k_nanosleep){ if (a0) nanosleep((const struct timespec*)P(a0), a1 ? (struct timespec*)P(a1) : nullptr); return 0; }

// --- assorted libkernel stubs ---
HLE(k_ok)              { return 0; }                       // generic success no-op
HLE(k_load_start_mod)  { return g_module_handle++; }       // return a positive module id
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
    constexpr int16_t EVFILT_VIDEO_OUT = -0x0a;   // SCE VideoOut event filter (flip + vblank)

    struct EqState { std::mutex m; std::condition_variable cv; std::deque<SceKEvent> ready; };
    std::mutex g_eq_mx;
    std::unordered_map<uint64_t, EqState*> g_eqs;              // guest eq ptr -> state
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
    std::atomic<bool> g_pump_started{false};

    EqState* eq_find(uint64_t eq) { std::lock_guard<std::mutex> lk(g_eq_mx); auto it = g_eqs.find(eq); return it == g_eqs.end() ? nullptr : it->second; }
    void eq_post(uint64_t eq, const SceKEvent& e) {
        EqState* s = eq_find(eq); if (!s) return;
        std::lock_guard<std::mutex> lk(s->m);
        if (s->ready.size() < 4) s->ready.push_back(e);   // cap so an un-drained queue can't grow unbounded
        s->cv.notify_all();
    }
    void vblank_pump() {
        uint64_t frame = 0;
        for (;;) {
            struct timespec ts{ 0, 16666667 }; nanosleep(&ts, nullptr);   // ~60 Hz
            frame++;
            std::vector<FlipReg> fr, vr;
            { std::lock_guard<std::mutex> lk(g_eq_mx); fr = g_flip_regs; vr = g_vblank_regs; }
            for (auto& r : vr) { SceKEvent e{}; e.ident = r.ident; e.filter = EVFILT_VIDEO_OUT; e.data = (int64_t)frame; e.udata = r.udata; eq_post(r.eq, e); }
            for (auto& r : fr) { SceKEvent e{}; e.ident = r.ident; e.filter = EVFILT_VIDEO_OUT; e.data = (int64_t)frame; e.udata = r.udata; eq_post(r.eq, e); }
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
    EqState* s = new EqState();
    if (a0) *(void**)P(a0) = (void*)s;   // the guest's opaque SceKernelEqueue handle IS our state ptr
    { std::lock_guard<std::mutex> lk(g_eq_mx); g_eqs[(uint64_t)(uintptr_t)s] = s; }
    if (evlog()) fprintf(stderr, "[ev] CreateEqueue -> eq=%p name=%s\n", (void*)s, a1 ? (const char*)P(a1) : "");
    return 0;
}
HLE(k_eq_delete) {
    if (a0) { std::lock_guard<std::mutex> lk(g_eq_mx); auto it = g_eqs.find(a0); if (it != g_eqs.end()) { delete it->second; g_eqs.erase(it); } }
    return 0;
}
HLE(k_eq_wait)   {   // (eq, SceKernelEvent* ev, int num, int* out, SceKernelUseconds* timeout)
    if (evlog()) fprintf(stderr, "[ev] WaitEqueue eq=0x%llx num=%llu timeout=%s ra=eboot+0x%llx\n",
        (unsigned long long)a0, (unsigned long long)a2, a4 ? "yes" : "inf",
        (unsigned long long)((uint64_t)__builtin_return_address(0) - 0x400000000ull));
    // PROSPER_WAITCALLER: scan the stack for the GAME's wait-loop return address (eboot code range,
    // NOT the stub region at 0x6..) so we can disassemble the loop's exit condition. Log once per eq.
    if (getenv("PROSPER_WAITCALLER")) {
        static std::atomic<int> shown{0};
        if (shown.fetch_add(1) < 4) {
            uint64_t* sp = (uint64_t*)__builtin_frame_address(0);
            for (int i = 0; i < 80; i++) {
                uint64_t v = sp[i];
                if (v >= 0x400000000ull && v < 0x4c0000000ull) {   // eboot / IL2CPP executable range
                    fprintf(stderr, "[waitcaller] stack[%d] = eboot+0x%llx | code:", i, (unsigned long long)(v - 0x400000000ull));
                    // Dump the bytes of the call + the loop's post-call check/branch (from a bit before the
                    // return addr) so the wait-loop's exit condition can be disassembled offline.
                    const uint8_t* code = (const uint8_t*)(uintptr_t)(v - 0x10);
                    for (int b = 0; b < 0x40; b++) fprintf(stderr, "%02x", code[b]);
                    fprintf(stderr, "\n");
                }
            }
            fprintf(stderr, "[waitcaller] ---\n");
        }
    }
    EqState* s = eq_find(a0);
    int num = (int)a2; if (num < 1) num = 1;
    if (!s) { struct timespec ts{ 0, 1000000 }; nanosleep(&ts, nullptr); if (a3) *(int32_t*)P(a3) = 0; return 0; }
    std::unique_lock<std::mutex> lk(s->m);
    if (s->ready.empty()) {
        // timeout arg is a pointer to micro-seconds (null = wait forever). Cap the wait so a headless
        // run never hard-blocks the guest thread even if no source is registered yet.
        uint64_t req = a4 ? *(uint32_t*)P(a4) : 100000;
        // PROSPER_WAITCAP: cap the empty-queue wait (µs). Our 60 Hz pump posts a flip/vblank event every
        // ~16 ms, so a >16 ms cap makes an event-DRAIN loop (while WaitEqueue keeps returning events) never
        // reach the empty state → the guest's frame loop stalls draining pumped events. A sub-vblank cap
        // (e.g. 8 ms) lets an empty drain time out (0 events) so the guest exits the drain and renders.
        uint64_t cap = getenv("PROSPER_WAITCAP") ? (uint64_t)atoll(getenv("PROSPER_WAITCAP")) : 100000;
        uint64_t us = req; if (us > cap) us = cap;
        if (evlog()) fprintf(stderr, "[ev]   WAIT.empty req=%lluus cap=%lluus\n",
                             (unsigned long long)req, (unsigned long long)cap);
        s->cv.wait_for(lk, std::chrono::microseconds(us), [&]{ return !s->ready.empty(); });
    }
    int n = 0; auto* ev = (SceKEvent*)P(a1);
    while (n < num && !s->ready.empty()) { if (ev) ev[n] = s->ready.front(); s->ready.pop_front(); n++; }
    if (a3) *(int32_t*)P(a3) = n;
    if (evlog() && n > 0) fprintf(stderr, "[ev]   -> delivered %d ev(s) eq=0x%llx ra=eboot+0x%llx (ident=%lld filter=%d)\n",
        n, (unsigned long long)a0, (unsigned long long)((uint64_t)__builtin_return_address(0) - 0x400000000ull),
        (long long)(ev ? ev[0].ident : 0), (int)(ev ? ev[0].filter : 0));
    return 0;   // 0 = success (n events returned; n may be 0 on timeout)
}
HLE(k_eq_getcount){
    EqState* s = eq_find(a0); int n = 0;
    if (s) { std::lock_guard<std::mutex> lk(s->m); n = (int)s->ready.size(); }
    if (evlog()) fprintf(stderr, "[ev] GetEventCount eq=0x%llx -> %d\n", (unsigned long long)a0, n);
    return (uint64_t)n;
}

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
// One-shot timer: post an EVFILT_TIMER event to the equeue after `usec` microseconds.
static void post_after(uint64_t eq, int64_t id, uint64_t udata, uint64_t usec, int16_t filter) {
    std::thread([eq, id, udata, usec, filter]{
        struct timespec ts{ (time_t)(usec / 1000000), (long)((usec % 1000000) * 1000) };
        nanosleep(&ts, nullptr);
        SceKEvent e{}; e.ident = id; e.filter = filter; e.data = (int64_t)usec; e.udata = udata;
        eq_post(eq, e);
    }).detach();
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
    R("sceKernelAddHRTimerEvent", k_add_hrtimer_event); R("sceKernelAddUserEvent", k_add_user_event);
    R("sceKernelAddUserEventEdge", k_add_user_event);   R("sceKernelTriggerUserEvent", k_trigger_user_event);
    R("sceKernelDeleteUserEvent", k_ok);        R("sceKernelDeleteHRTimerEvent", k_ok);
    R("sceKernelDeleteTimerEvent", k_ok);       R("sceKernelAddTimerEvent", k_add_timer_event);
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
    // C11 threads
    R("_Mtx_init", m_mtx_init);   R("_Mtx_lock", m_mtx_lock);   R("_Mtx_unlock", m_mtx_unlock);
    R("_Mtx_destroy", m_mtx_destroy);
    R("_Cnd_init", m_cnd_init);   R("_Cnd_signal", m_cnd_signal); R("_Cnd_broadcast", m_cnd_broadcast);
    R("_Cnd_wait", m_cnd_wait);   R("_Cnd_destroy", m_cnd_destroy);
    #undef R
}

} // namespace prosper
