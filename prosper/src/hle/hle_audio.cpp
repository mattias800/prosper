// hle_audio.cpp — libSceAudioOut HLE, backed by a pluggable AudioSink (see audio.hpp).
//
// Decodes the PS5 sceAudioOut* calls into port lifecycle + interleaved PCM grains and forwards
// them to the installed backend. prosper_core ships only the headless default (silent, real-time
// paced) so it stays dependency-free; a concrete frontend (SDL3, ...) installs itself via
// audio_set_sink() from outside the core.
#include "dispatch.hpp"
#include "nid.hpp"
#include "audio.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <sys/uio.h>
#include <unistd.h>
#endif
#include "../host/posix_shim.hpp"   // Darwin: process_vm_readv/writev

namespace prosper {

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define P(x) ((void*)(uintptr_t)(x))

namespace {

constexpr int kMaxPorts  = 16;
constexpr int kVolume0dB = 32768;   // SCE_AUDIO_VOLUME_0DB

struct Port {
    bool          in_use = false;
    int           type = 0;   // SceAudioOutPortType (0=MAIN, 1=BGM, 2=VOICE, 3=PERSONAL, 4=PADSPK, 127=AUX)
    AudioPortInfo info;
    int           vol[8] = { kVolume0dB, kVolume0dB, kVolume0dB, kVolume0dB,
                             kVolume0dB, kVolume0dB, kVolume0dB, kVolume0dB };
};

// SCE audio error codes the guest actually tests for (Kyty Errno.h:342/344). A generic -1 is
// unrecognizable to retry-vs-abort logic (a single wrong errno already caused a full render
// stall once — see hle_service.cpp's GetEvent note). Sign-extended: the guest ABI returns
// int32 in eax (negative), and host-side callers/tests compare the full u64 as int64.
constexpr uint64_t kAudioErrInvalidPort = (uint64_t)(int64_t)(int32_t)0x80260003;
constexpr uint64_t kAudioErrPortFull    = (uint64_t)(int64_t)(int32_t)0x80260005;

std::mutex g_mx;                 // guards the port table
Port       g_ports[kMaxPorts];

// --- default backend: silent, real-time paced (headless) --------------------------------
// sceAudioOutOutput on real hardware blocks until the audio ring has room, which paces the
// game's audio thread at real time. With no device attached we reproduce that pacing by
// sleeping each grain's wall-clock duration, so the guest advances at the correct speed.
struct RealtimeSilentSink : AudioSink {
    struct Pace { std::chrono::steady_clock::time_point next{}; long long ns_per_grain = 0; };
    Pace p_[kMaxPorts];
    bool open(int port, const AudioPortInfo& info) override {
        if (port < 1 || port > kMaxPorts) return false;
        int freq = info.freq > 0 ? info.freq : 48000;
        int grain = info.grain > 0 ? info.grain : 256;
        p_[port - 1].ns_per_grain = (long long)grain * 1000000000LL / freq;
        p_[port - 1].next = {};
        return true;
    }
    void output(int port, const void*, int frames) override {
        if (port < 1 || port > kMaxPorts) return;
        auto& s = p_[port - 1];
        long long ns = s.ns_per_grain > 0 ? s.ns_per_grain : ((long long)frames * 1000000000LL / 48000);
        auto now = std::chrono::steady_clock::now();
        auto dur = std::chrono::nanoseconds(ns);
        // (Re)sync if unset or we fell far behind (e.g. after a stall) to avoid burst catch-up.
        if (s.next.time_since_epoch().count() == 0 || s.next < now - dur * 4) s.next = now;
        s.next += dur;
        if (s.next > now) std::this_thread::sleep_until(s.next);
    }
    void close(int port) override { if (port >= 1 && port <= kMaxPorts) p_[port - 1] = {}; }
};

RealtimeSilentSink        g_default_sink;
std::atomic<AudioSink*>   g_sink{ &g_default_sink };

// Caller must hold g_mx.
Port* port_of(int handle) {
    if (handle < 1 || handle > kMaxPorts) return nullptr;
    Port& p = g_ports[handle - 1];
    return p.in_use ? &p : nullptr;
}

} // namespace

// --- public backend hooks (audio.hpp) ---------------------------------------------------
void audio_set_sink(AudioSink* sink) { g_sink.store(sink ? sink : &g_default_sink); }
AudioSink* audio_sink() { return g_sink.load(); }

void audio_decode_format(uint32_t param, int& channels, AudioFmt& fmt) {
    switch (param & 0xff) {                                   // SceAudioOutParamFormat (low byte)
        case 0: channels = 1; fmt = AudioFmt::S16; break;     // S16_MONO
        case 1: channels = 2; fmt = AudioFmt::S16; break;     // S16_STEREO
        case 2: channels = 8; fmt = AudioFmt::S16; break;     // S16_8CH
        case 3: channels = 1; fmt = AudioFmt::F32; break;     // FLOAT_MONO
        case 4: channels = 2; fmt = AudioFmt::F32; break;     // FLOAT_STEREO
        case 5: channels = 8; fmt = AudioFmt::F32; break;     // FLOAT_8CH
        case 6: channels = 8; fmt = AudioFmt::S16; break;     // S16_8CH_STD
        case 7: channels = 8; fmt = AudioFmt::F32; break;     // FLOAT_8CH_STD
        default: channels = 2; fmt = AudioFmt::S16; break;    // unknown -> S16 stereo
    }
}

void audio_reset() {
    AudioSink* s = audio_sink();
    std::lock_guard<std::mutex> lk(g_mx);
    for (int i = 0; i < kMaxPorts; i++) {
        if (g_ports[i].in_use && s) s->close(i + 1);
        g_ports[i] = Port{};
    }
    g_sink.store(&g_default_sink);
}

// --- sceAudioOut HLE --------------------------------------------------------------------
HLE(audio_init) { (void)a0; return 0; }   // sceAudioOutInit: idempotent success

// sceAudioOutOpen(userId, type, index, len, freq, param) -> handle (>=1) or negative error.
HLE(audio_open) {
    (void)a0; (void)a2;
    AudioPortInfo info;
    info.grain = (int)(a3 ? a3 : 256);
    info.freq  = (int)(a4 ? a4 : 48000);
    audio_decode_format((uint32_t)a5, info.channels, info.fmt);
    int type = (int)a1;   // kept for GetPortState's type-dependent output/channel report

    int handle = 0;
    { std::lock_guard<std::mutex> lk(g_mx);
      for (int i = 0; i < kMaxPorts; i++) {
          if (g_ports[i].in_use) continue;
          g_ports[i].in_use = true;
          g_ports[i].type = type;
          g_ports[i].info = info;
          for (int c = 0; c < 8; c++) g_ports[i].vol[c] = kVolume0dB;
          handle = i + 1;
          break;
      } }
    if (!handle) return kAudioErrPortFull;
    if (auto* s = audio_sink()) s->open(handle, info);
    return (uint64_t)handle;
}

// sceAudioOutOutput(handle, ptr) -> frames written (>=0) or negative error. ptr==0 => drain.
HLE(audio_output) {
    AudioPortInfo info;
    { std::lock_guard<std::mutex> lk(g_mx); Port* p = port_of((int)a0); if (!p) return kAudioErrInvalidPort; info = p->info; }
    if (a1 == 0) return 0;   // drain/flush: nothing buffered in the headless model
    if (auto* s = audio_sink()) s->output((int)a0, P(a1), info.grain);
    return (uint64_t)info.grain;
}

// sceAudioOutOutputs(SceAudioOutOutputParam param[], int num) -> total frames or negative error.
// SceAudioOutOutputParam = { int32 handle; int32 reserved; void* ptr } (16 bytes).
HLE(audio_outputs) {
    struct OutParam { int32_t handle; int32_t reserved; uint64_t ptr; };
    const auto* arr = (const OutParam*)P(a0);
    int num = (int)a1;
    if (!arr || num <= 0) return 0;
    // sceAudioOutOutputs writes the SAME time-slice to N ports in parallel; the return is samples-per-
    // channel of that slice (one grain), NOT the additive sum over ports. Returning the sum made a guest
    // using the count as a sample-clock over-count by N x (Kyty/shadPS4 both return a single port's grain).
    uint64_t grain = 0; bool have = false;
    for (int i = 0; i < num; i++) {
        AudioPortInfo info; bool ok;
        { std::lock_guard<std::mutex> lk(g_mx); Port* p = port_of(arr[i].handle); ok = (p != nullptr); if (ok) info = p->info; }
        if (!ok) continue;
        if (arr[i].ptr) { if (auto* s = audio_sink()) s->output(arr[i].handle, P(arr[i].ptr), info.grain); }
        if (!have) { grain = info.grain; have = true; }
    }
    return grain;
}

// sceAudioOutSetVolume(handle, flag(channel mask), int vol[]) -> 0 or negative error.
HLE(audio_set_volume) {
    uint32_t mask = (uint32_t)a1;
    const int* vols = (const int*)P(a2);
    { std::lock_guard<std::mutex> lk(g_mx);
      Port* p = port_of((int)a0); if (!p) return kAudioErrInvalidPort;
      if (vols) { int vi = 0; for (int c = 0; c < 8; c++) if (mask & (1u << c)) p->vol[c] = vols[vi++]; } }
    if (auto* s = audio_sink()) s->set_volume((int)a0, mask, vols);
    return 0;
}

// sceAudioOutClose(handle) -> 0 or negative error.
HLE(audio_close) {
    int handle = (int)a0;
    { std::lock_guard<std::mutex> lk(g_mx);
      Port* p = port_of(handle); if (!p) return kAudioErrInvalidPort; p->in_use = false; }
    if (auto* s = audio_sink()) s->close(handle);
    return 0;
}

// sceAudioOutGetPortState(handle, SceAudioOutPortState* state) -> 0 or negative error.
// Layout per Kyty Audio.cpp:340 (the previous fill invented its own: channel as u16 @2
// clobbering reserved1, volume as u32 @8 — which is the FLAG field — and left the real
// volume @4 zero, i.e. "muted"): uint16 output @0; uint8 channel @2; uint8 reserved @3;
// int16 volume @4; uint16 reroute_counter @6; uint64 flag @8; uint64 reserved2[2] @0x10.
HLE(audio_get_port_state) {
    int channels, type;
    { std::lock_guard<std::mutex> lk(g_mx); Port* p = port_of((int)a0); if (!p) return kAudioErrInvalidPort;
      channels = p->info.channels; type = p->type; }
    if (a1) {
        auto* st = (uint8_t*)P(a1);
        memset(st, 0, 0x20);
        *(int16_t*)(st + 4) = 127;   // volume (Kyty AudioOutGetPortState reports 127)
        switch (type) {              // output/channel are port-type dependent (Kyty :432-448)
            case 3: case 127: /* output=0, channel=0 */ break;
            case 4:  *(uint16_t*)(st + 0) = 4; st[2] = 1; break;                          // pad speaker
            default: *(uint16_t*)(st + 0) = 1; st[2] = (uint8_t)(channels > 2 ? 2 : channels); break;
        }
    }
    return 0;
}

// ---- libSceAudioOut2 (PS5-only; no Kyty/shadPS4 reference exists) -----------------------
// DOLL's CRI Atom (ADX) middleware drives audio through AudioOut2. The generic unimplemented
// stub (return 0, out-params untouched) made CRI read an UNINITIALIZED context-memory size and
// malloc/memset it: when the stack garbage happened to be unallocatable the main thread died in
// libc memset(NULL) (RUN ENDED at libc.prx+0x10556, backtrace through the CRI region
// eboot+0x5ff..0x605M) — the intermittent "1 flip then crash" of issue #213. NID identities
// recovered by nid_hash brute force over the sce_stubs corpus: g2tViFIohHE=sceAudioOut2Initialize,
// t5YrizufpQc=sceAudioOut2ContextResetParam, pDmme7Bgm6E=sceAudioOut2ContextQueryMemory.
//
// Contracts recovered by LIVE CAPTURE (PROSPER_AUDIO2LOG probe run, /tmp/draws_a2.log,
// 2026-07-09) — this is a null-device backend in the sense of Wine's null audio driver: real
// handle lifecycle + real-time pacing, no host audio device.
//   sceAudioOut2ContextResetParam(param*)              param is 0x40 bytes (guest zero-fills
//     0x00..0x3f then sets {+0:queue=8, +4:0x40, +8:0, +0xc:2, +0x10:grain=0x100, +0x14:1}).
//   sceAudioOut2ContextQueryMemory(param*, size_t* out) out is the work-memory byte size the
//     guest allocates and hands to ContextCreate (a1 = a0-8 on the create path, live).
//   sceAudioOut2ContextCreate(param*, mem, memSize, Handle* out)
//   sceAudioOut2UserCreate(userId, Handle* out)         (userId=0xff live)
//   sceAudioOut2PortCreate(ctx, portParam*, Handle* out, ...)
//   pump loop (dedicated CRI server thread, live): PortGetState(port, state*) ->
//     PortSetAttributes(port, attr*, n) -> ContextAdvance(ctx) -> ContextPush(ctx, flag).
// ContextPush paces one grain of wall-clock time (blocking-when-full HW semantics, same model
// as RealtimeSilentSink) so the pump thread advances at real time instead of spinning.
// CONFIDENCE: MED (arg positions + struct sizes live-verified; field meanings partly inferred;
// PortGetState layout unknown -> zero-filled 0x20, marked LOW below).
namespace {

bool audio2log() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PROSPER_AUDIO2LOG"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v == 1;
}

// Fault-safe guest-memory hexdump for live ABI capture (unmapped args must not crash the HLE).
void a2_dump(const char* tag, uint64_t p, size_t n) {
#ifndef _WIN32
    if (!p) return;
    std::vector<uint8_t> buf(n);
    struct iovec l { buf.data(), n }, r { (void*)(uintptr_t)p, n };
    ssize_t got = process_vm_readv(getpid(), &l, 1, &r, 1, 0);
    if (got <= 0) { fprintf(stderr, "[audio2]   %s @0x%llx: <unreadable>\n", tag, (unsigned long long)p); return; }
    fprintf(stderr, "[audio2]   %s @0x%llx:", tag, (unsigned long long)p);
    for (ssize_t i = 0; i < got; i++) {
        if ((i & 15) == 0) fprintf(stderr, "\n[audio2]     +%02zx ", (size_t)i);
        fprintf(stderr, "%02x ", buf[i]);
    }
    fprintf(stderr, "\n");
#else
    (void)tag; (void)p; (void)n;
#endif
}

void a2_log(const char* name, uint64_t a0, uint64_t a1, uint64_t a2,
            uint64_t a3, uint64_t a4, uint64_t a5, void* ra) {
    if (!audio2log()) return;
    fprintf(stderr, "[audio2] %s(0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx) ra=eboot+0x%llx\n",
            name, (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
            (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5,
            (unsigned long long)((uint64_t)ra - 0x400000000ull));
}

} // namespace

#define A2LOG(name) a2_log(name, a0, a1, a2, a3, a4, a5, __builtin_return_address(0))

// AudioOut2 handle space. Handles are opaque u64s the guest stores and passes back; tag them so
// stray guest values are distinguishable in logs. One context + a few ports is all CRI uses.
constexpr uint64_t kA2CtxTag  = 0xA2C0000000000000ull;
constexpr uint64_t kA2UserTag = 0xA2D0000000000000ull;
constexpr uint64_t kA2PortTag = 0xA2E0000000000000ull;

struct A2Context {
    bool     used = false;
    uint32_t grain = 256;         // samples per Advance/Push cycle (param +0x10, live: 0x100)
    // Real-time pacing state for ContextPush (blocking-when-full HW semantics).
    std::chrono::steady_clock::time_point next{};
};
std::mutex g_a2_mx;
A2Context  g_a2_ctx[4];
uint32_t   g_a2_users = 0, g_a2_ports = 0;

// Fault-safe u64 store to a guest out-pointer (same rationale as apr_write_guest_dst: a bad
// pointer must fail the call, not SIGSEGV inside the HLE).
bool a2_store_u64(uint64_t dst, uint64_t v) {
#ifndef _WIN32
    struct iovec l { &v, sizeof v }, r { (void*)(uintptr_t)dst, sizeof v };
    return process_vm_writev(getpid(), &l, 1, &r, 1, 0) == (ssize_t)sizeof v;
#else
    if (!dst) return false;
    *(uint64_t*)(uintptr_t)dst = v; return true;
#endif
}
bool a2_store_zeros(uint64_t dst, size_t n) {
#ifndef _WIN32
    std::vector<uint8_t> z(n, 0);
    struct iovec l { z.data(), n }, r { (void*)(uintptr_t)dst, n };
    return process_vm_writev(getpid(), &l, 1, &r, 1, 0) == (ssize_t)n;
#else
    if (!dst) return false;
    memset((void*)(uintptr_t)dst, 0, n); return true;
#endif
}

constexpr uint64_t kA2ErrInvalid = (uint64_t)(int64_t)(int32_t)0x80260003;   // AudioOut error space

// sceAudioOut2Initialize(void) -> 0. Idempotent success (same as sceAudioOutInit).
HLE(audio2_initialize) { A2LOG("sceAudioOut2Initialize"); return 0; }

// sceAudioOut2ContextResetParam(SceAudioOut2ContextParam* param) -> 0.
// The param is 0x40 bytes (live: the create-path caller zero-fills exactly 0x00..0x3f before
// setting fields). Reset = zero-fill; the guest overwrites every field it uses afterwards.
HLE(audio2_ctx_reset_param) {
    A2LOG("sceAudioOut2ContextResetParam");
    if (!a2_store_zeros(a0, 0x40)) return kA2ErrInvalid;
    return 0;
}

// sceAudioOut2ContextQueryMemory(const param*, size_t* outSize) -> 0.
// Live: a1 is the out size the guest allocates and passes straight to ContextCreate as
// (mem, memSize). The null backend needs no guest work memory; report a fixed 1 MiB so the
// allocation is real and cheap (the value's only observable effect is that malloc succeeds —
// the garbage value 0x244811c was allocated and accepted in the capture run).
HLE(audio2_ctx_query_memory) {
    A2LOG("sceAudioOut2ContextQueryMemory");
    if (audio2log()) a2_dump("param", a0, 0x40);
    if (!a2_store_u64(a1, 0x100000)) return kA2ErrInvalid;
    return 0;
}

// sceAudioOut2ContextCreate(param*, mem, memSize, Handle* outCtx) -> 0.
HLE(audio2_ctx_create) {
    A2LOG("sceAudioOut2ContextCreate");
    uint32_t grain = 256;
#ifndef _WIN32
    if (a0) {   // param +0x10 = samples per grain (live: 0x100)
        uint32_t g = 0;
        struct iovec l { &g, 4 }, r { (void*)(uintptr_t)(a0 + 0x10), 4 };
        if (process_vm_readv(getpid(), &l, 1, &r, 1, 0) == 4 && g >= 64 && g <= 4096) grain = g;
    }
#endif
    std::lock_guard<std::mutex> lk(g_a2_mx);
    for (int i = 0; i < 4; i++) {
        if (g_a2_ctx[i].used) continue;
        g_a2_ctx[i] = A2Context{};
        g_a2_ctx[i].used = true;
        g_a2_ctx[i].grain = grain;
        if (!a2_store_u64(a3, kA2CtxTag | (uint64_t)(i + 1))) { g_a2_ctx[i].used = false; return kA2ErrInvalid; }
        return 0;
    }
    return kA2ErrInvalid;
}
HLE(audio2_ctx_destroy) {
    A2LOG("sceAudioOut2ContextDestroy");
    std::lock_guard<std::mutex> lk(g_a2_mx);
    uint64_t idx = a0 & 0xff;
    if ((a0 & ~0xffull) == kA2CtxTag && idx >= 1 && idx <= 4) g_a2_ctx[idx - 1] = A2Context{};
    return 0;
}

// sceAudioOut2UserCreate(userId, Handle* out) -> 0. (live: userId=0xff)
HLE(audio2_user_create) {
    A2LOG("sceAudioOut2UserCreate");
    std::lock_guard<std::mutex> lk(g_a2_mx);
    if (!a2_store_u64(a1, kA2UserTag | (uint64_t)++g_a2_users)) return kA2ErrInvalid;
    return 0;
}
HLE(audio2_user_destroy) { A2LOG("sceAudioOut2UserDestroy"); return 0; }

// sceAudioOut2PortCreate(ctx, const portParam*, Handle* outPort, ...) -> 0.
HLE(audio2_port_create) {
    A2LOG("sceAudioOut2PortCreate");
    std::lock_guard<std::mutex> lk(g_a2_mx);
    if (!a2_store_u64(a2, kA2PortTag | (uint64_t)++g_a2_ports)) return kA2ErrInvalid;
    return 0;
}
HLE(audio2_port_destroy) { A2LOG("sceAudioOut2PortDestroy"); return 0; }

// sceAudioOut2PortGetState(port, State* out) -> 0. Layout unknown; the pump loop consumed a
// zero-filled 0x20 in the capture run and kept cycling. CONFIDENCE: LOW (zero state = silent
// port, no observed guest field reads yet — re-probe if CRI branches on it).
HLE(audio2_port_get_state) {
    A2LOG("sceAudioOut2PortGetState");
    if (!a2_store_zeros(a1, 0x20)) return kA2ErrInvalid;
    return 0;
}
HLE(audio2_port_set_attr) { A2LOG("sceAudioOut2PortSetAttributes"); return 0; }

// sceAudioOut2ContextAdvance(ctx) -> 0. State advance only; the pacing lives in Push.
HLE(audio2_ctx_advance) { A2LOG("sceAudioOut2ContextAdvance"); return 0; }

// sceAudioOut2ContextPush(ctx, flag) -> 0. On hardware Push blocks while the output queue is
// full; the null backend reproduces that as one grain of wall-clock pacing per call (same
// model as RealtimeSilentSink) so CRI's server thread runs at real time, not a hot spin.
HLE(audio2_ctx_push) {
    A2LOG("sceAudioOut2ContextPush");
    uint64_t idx = a0 & 0xff;
    std::chrono::steady_clock::time_point target;
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        A2Context* c = ((a0 & ~0xffull) == kA2CtxTag && idx >= 1 && idx <= 4 && g_a2_ctx[idx - 1].used)
                       ? &g_a2_ctx[idx - 1] : &g_a2_ctx[0];
        uint32_t grain = c->grain ? c->grain : 256;
        auto dur = std::chrono::nanoseconds((long long)grain * 1000000000LL / 48000);
        auto now = std::chrono::steady_clock::now();
        // (Re)sync if unset or far behind (post-stall) to avoid burst catch-up — same policy
        // as RealtimeSilentSink::output.
        if (c->next.time_since_epoch().count() == 0 || c->next < now - dur * 4) c->next = now;
        c->next += dur;
        target = c->next;
    }
    std::this_thread::sleep_until(target);   // pace outside the lock
    return 0;
}

// Generic logging probes for the not-yet-exercised remainder of the surface.
#define A2_PROBE(fn, str) HLE(fn) { A2LOG(str); return 0; }
A2_PROBE(audio2_ctx_get_queue_level, "sceAudioOut2ContextGetQueueLevel")
A2_PROBE(audio2_ctx_set_attr,    "sceAudioOut2ContextSetAttributes")
A2_PROBE(audio2_ctx_bed_write,   "sceAudioOut2ContextBedWrite")
A2_PROBE(audio2_port_register,   "sceAudioOut2PortRegister")
A2_PROBE(audio2_port_unregister, "sceAudioOut2PortUnregister")
A2_PROBE(audio2_get_system_state, "sceAudioOut2GetSystemState")
A2_PROBE(audio2_get_speaker_info, "sceAudioOut2GetSpeakerInfo")
A2_PROBE(audio2_mastering_init,  "sceAudioOut2MasteringInit")
A2_PROBE(audio2_mastering_term,  "sceAudioOut2MasteringTerm")
A2_PROBE(audio2_mastering_set_param, "sceAudioOut2MasteringSetParam")
A2_PROBE(audio2_mastering_get_state, "sceAudioOut2MasteringGetState")
#undef A2_PROBE

// --- libSceAjm (Audio Job Manager — compressed-audio decode: ATRAC9/MP3/AAC) (#187) ---
// PPSA02664 initializes AJM at startup. prosper does not decode compressed audio (the AudioOut path
// already discards/plays raw PCM headlessly), so this is a faithful HEADLESS lifecycle: every handle
// out-param (context / instance / batch id) is filled with a REAL opaque handle — never left as the
// caller's garbage, the bug the generic stub caused — inputs are validated, and a submitted decode
// batch reports "started" then "complete" so the guest's audio pipeline proceeds (producing silence,
// exactly like AudioSink discarding). It does NOT run the decode jobs — real ATRAC9/MP3/AAC decoding
// is a separate, large piece and isn't needed to boot. Signatures verified vs shadPS4
// src/core/libraries/ajm/ajm.h; error codes from ajm_error.h. CONFIDENCE: HIGH (handle lifecycle);
// the no-decode behavior is intentional, not a guess.
namespace {
    std::atomic<uint32_t> g_ajm_next{1};   // one non-zero counter for context/instance/batch handles
    constexpr uint64_t AJM_ERR_INVALID_CONTEXT   = 0x80930002ull;
    constexpr uint64_t AJM_ERR_INVALID_INSTANCE  = 0x80930003ull;
    constexpr uint64_t AJM_ERR_INVALID_PARAMETER = 0x80930005ull;
}
// sceAjmInitialize(s64 reserved, u32* out_context): create a context. Filling out_context is the point.
HLE(ajm_initialize) {
    if (!a1) return AJM_ERR_INVALID_PARAMETER;
    *(uint32_t*)P(a1) = g_ajm_next.fetch_add(1);
    return 0;
}
HLE(ajm_finalize)         { return 0; }
// sceAjmModuleRegister(u32 context, AjmCodecType codec, s64 reserved): register a codec on the context.
HLE(ajm_module_register)  { return a0 ? 0 : AJM_ERR_INVALID_CONTEXT; }
HLE(ajm_module_unregister){ return 0; }
// sceAjmInstanceCreate(u32 context, codec, flags, u32* instance): a decoder instance handle.
HLE(ajm_instance_create) {
    if (!a0) return AJM_ERR_INVALID_CONTEXT;
    if (!a3) return AJM_ERR_INVALID_PARAMETER;
    *(uint32_t*)P(a3) = g_ajm_next.fetch_add(1);
    return 0;
}
// sceAjmInstanceDestroy(u32 context, u32 instance).
HLE(ajm_instance_destroy) {
    if (!a0) return AJM_ERR_INVALID_CONTEXT;
    if (!a1) return AJM_ERR_INVALID_INSTANCE;
    return 0;
}
// sceAjmBatchStartBuffer(context, batch, size, prio, AjmBatchError* err, u32* out_batch_id): accept a
// decode batch and report it started (out_batch_id filled). We don't run the jobs; BatchWait completes
// it. The AjmBatchError out (a4) is left as the caller's value (its layout isn't needed for no-error).
HLE(ajm_batch_start) {
    if (!a0) return AJM_ERR_INVALID_CONTEXT;
    if (!a5) return AJM_ERR_INVALID_PARAMETER;
    *(uint32_t*)P(a5) = g_ajm_next.fetch_add(1);
    return 0;
}
HLE(ajm_batch_wait)       { return a0 ? 0 : AJM_ERR_INVALID_CONTEXT; }   // batch completed
HLE(ajm_batch_errordump)  { return 0; }

// --- libSceNgs2 silent lifecycle ---------------------------------------------------------------
// Dead Cells is the first title to exercise NGS2. PROSPER_NGS2_TRACE preserves and logs all six
// guest arguments for this PS5 surface; normal execution uses the same handlers without logging.
namespace {

bool ngs2_trace_enabled() {
    const char* e = getenv("PROSPER_NGS2_TRACE");
    return e && *e && *e != '0';
}

void ngs2_trace_call(const char* name, std::atomic<uint64_t>& calls,
                     uint64_t a0, uint64_t a1, uint64_t a2,
                     uint64_t a3, uint64_t a4, uint64_t a5) {
    const uint64_t n = calls.fetch_add(1) + 1;
    if (ngs2_trace_enabled() && (n <= 8 || (n & (n - 1)) == 0)) {
        fprintf(stderr, "[ngs2] %s #%llu (0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx)\n",
                name, (unsigned long long)n,
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
    }
}

constexpr uint64_t kNgs2SystemTag = 0x4e47533253000000ull; // "NGS2S"
constexpr uint64_t kNgs2RackTag   = 0x4e47533252000000ull; // "NGS2R"
constexpr uint64_t kNgs2VoiceTag  = 0x4e47533256000000ull; // "NGS2V"
constexpr uint64_t kNgs2TagMask   = 0xffffffffffffff00ull;
constexpr uint64_t kNgs2VoiceMask = 0xffffffffff000000ull;
constexpr uint64_t kNgs2ErrInvalidOut    = (uint64_t)(int64_t)(int32_t)0x804a0053;
constexpr uint64_t kNgs2ErrInvalidSystem = (uint64_t)(int64_t)(int32_t)0x804a0230;
constexpr uint64_t kNgs2ErrInvalidRack   = (uint64_t)(int64_t)(int32_t)0x804a0261;
constexpr uint64_t kNgs2ErrInvalidVoice  = (uint64_t)(int64_t)(int32_t)0x804a0302;

struct Ngs2RackState {
    bool used = false;
    uint64_t system = 0;
    uint32_t rack_id = 0;
    uint32_t max_voices = 64;
};

std::mutex g_ngs2_mx;
bool g_ngs2_systems[4]{};
Ngs2RackState g_ngs2_racks[32];
std::mutex g_ngs2_zero_mx;
std::vector<uint8_t> g_ngs2_zeros;

bool ngs2_read_u32(uint64_t src, uint32_t& value) {
#ifndef _WIN32
    struct iovec l { &value, sizeof value }, r { (void*)(uintptr_t)src, sizeof value };
    return src && process_vm_readv(getpid(), &l, 1, &r, 1, 0) == (ssize_t)sizeof value;
#else
    if (!src) return false;
    value = *(const uint32_t*)(uintptr_t)src;
    return true;
#endif
}

bool ngs2_read_bytes(uint64_t src, void* dst, size_t size) {
#ifndef _WIN32
    struct iovec l { dst, size }, r { (void*)(uintptr_t)src, size };
    return src && process_vm_readv(getpid(), &l, 1, &r, 1, 0) == (ssize_t)size;
#else
    if (!src) return false;
    memcpy(dst, (const void*)(uintptr_t)src, size);
    return true;
#endif
}

bool ngs2_zero_bytes(uint64_t dst, size_t size) {
#ifndef _WIN32
    // Do not use thread_local here: guest execution swaps %fs, and adding host TLS to prosper_core
    // perturbs Messenger before its first syscall. Process-global zero storage is sufficient because
    // NGS2 rendering is serialized by its audio thread; the lock also makes that contract explicit.
    std::lock_guard<std::mutex> lock(g_ngs2_zero_mx);
    if (g_ngs2_zeros.size() < size) g_ngs2_zeros.resize(size, 0);
    struct iovec l { g_ngs2_zeros.data(), size }, r { (void*)(uintptr_t)dst, size };
    return dst && process_vm_writev(getpid(), &l, 1, &r, 1, 0) == (ssize_t)size;
#else
    if (!dst) return false;
    memset((void*)(uintptr_t)dst, 0, size);
    return true;
#endif
}

uint32_t ngs2_max_voices(uint64_t option) {
    uint32_t voices = 0;
    // SceNgs2RackOption: size[8], name[16], flags, maxGrainSamples, maxVoices.
    if (option) ngs2_read_u32(option + 0x20, voices);
    return voices ? voices : 64;
}

bool ngs2_valid_system(uint64_t handle) {
    const uint64_t slot = handle & 0xff;
    return (handle & kNgs2TagMask) == kNgs2SystemTag && slot >= 1 && slot <= 4 &&
           g_ngs2_systems[slot - 1];
}

Ngs2RackState* ngs2_rack(uint64_t handle) {
    const uint64_t slot = handle & 0xff;
    if ((handle & kNgs2TagMask) != kNgs2RackTag || slot < 1 || slot > 32) return nullptr;
    Ngs2RackState& rack = g_ngs2_racks[slot - 1];
    return rack.used ? &rack : nullptr;
}

} // namespace

#define NGS2_LOG(label) do { static std::atomic<uint64_t> calls{0}; \
    ngs2_trace_call(label, calls, a0, a1, a2, a3, a4, a5); } while (0)

// NGS2 is implemented as a silent backend: the guest still receives real buffer requirements and
// opaque lifecycle handles, while SystemRender produces silence. Layouts/prototypes agree between
// the 3.20 export table, Kyty, shadPS4, and the Dead Cells live trace above. CONFIDENCE: HIGH on
// argument/output positions and handle flow; MED on the deliberately private work-buffer sizes.
HLE(ngs2_system_query_buffer) {
    NGS2_LOG("sceNgs2SystemQueryBufferSize");
    if (!a1 || !a2_store_zeros(a1, 0x40) || !a2_store_u64(a1 + 8, 0x1000))
        return kNgs2ErrInvalidOut;
    return 0;
}

HLE(ngs2_system_create) {
    NGS2_LOG("sceNgs2SystemCreate");
    if (!a1 || !a2) return kNgs2ErrInvalidOut;
    std::lock_guard<std::mutex> lock(g_ngs2_mx);
    for (uint64_t i = 0; i < 4; ++i) {
        if (g_ngs2_systems[i]) continue;
        g_ngs2_systems[i] = true;
        if (!a2_store_u64(a2, kNgs2SystemTag | (i + 1))) {
            g_ngs2_systems[i] = false;
            return kNgs2ErrInvalidOut;
        }
        return 0;
    }
    return kNgs2ErrInvalidSystem;
}

HLE(ngs2_rack_query_buffer) {
    NGS2_LOG("sceNgs2RackQueryBufferSize");
    const uint64_t size = 0x1000ull + (uint64_t)ngs2_max_voices(a1) * 0x40ull;
    if (!a2 || !a2_store_zeros(a2, 0x40) || !a2_store_u64(a2 + 8, size))
        return kNgs2ErrInvalidOut;
    return 0;
}

HLE(ngs2_rack_create) {
    NGS2_LOG("sceNgs2RackCreate");
    if (!a3 || !a4) return kNgs2ErrInvalidOut;
    std::lock_guard<std::mutex> lock(g_ngs2_mx);
    if (!ngs2_valid_system(a0)) return kNgs2ErrInvalidSystem;
    for (uint64_t i = 0; i < 32; ++i) {
        if (g_ngs2_racks[i].used) continue;
        g_ngs2_racks[i] = {true, a0, (uint32_t)a1, ngs2_max_voices(a2)};
        if (!a2_store_u64(a4, kNgs2RackTag | (i + 1))) {
            g_ngs2_racks[i] = {};
            return kNgs2ErrInvalidOut;
        }
        return 0;
    }
    return kNgs2ErrInvalidRack;
}

HLE(ngs2_rack_get_voice) {
    NGS2_LOG("sceNgs2RackGetVoiceHandle");
    if (!a2) return kNgs2ErrInvalidOut;
    std::lock_guard<std::mutex> lock(g_ngs2_mx);
    Ngs2RackState* rack = ngs2_rack(a0);
    if (!rack) return kNgs2ErrInvalidRack;
    if (a1 >= rack->max_voices || a1 > 0xff) return kNgs2ErrInvalidVoice;
    const uint64_t rack_slot = a0 & 0xff;
    return a2_store_u64(a2, kNgs2VoiceTag | (rack_slot << 8) | a1) ? 0 : kNgs2ErrInvalidOut;
}

HLE(ngs2_voice_control) {
    NGS2_LOG("sceNgs2VoiceControl");
    return (a0 & kNgs2VoiceMask) == kNgs2VoiceTag ? 0 : kNgs2ErrInvalidVoice;
}

HLE(ngs2_voice_run_commands) {
    NGS2_LOG("sceNgs2VoiceRunCommands");
    return (a0 & kNgs2VoiceMask) == kNgs2VoiceTag ? 0 : kNgs2ErrInvalidVoice;
}

HLE(ngs2_voice_get_state) {
    NGS2_LOG("sceNgs2VoiceGetState");
    if ((a0 & kNgs2VoiceMask) != kNgs2VoiceTag) return kNgs2ErrInvalidVoice;
    // An inert backend has no active decoder: zero stateFlags means Empty. Dead Cells requests the
    // sampler extension, so clear the caller-declared payload as Kyty does before filling it.
    if (!a1 || !a2 || a2 > 0x1000 || !a2_store_zeros(a1, (size_t)a2)) return kNgs2ErrInvalidOut;
    return 0;
}

HLE(ngs2_geom_reset_source) {
    NGS2_LOG("sceNgs2GeomResetSourceParam");
    return a0 && a2_store_zeros(a0, 0xa8) ? 0 : kNgs2ErrInvalidOut;
}

HLE(ngs2_system_render) {
    NGS2_LOG("sceNgs2SystemRender");
    {
        std::lock_guard<std::mutex> lock(g_ngs2_mx);
        if (!ngs2_valid_system(a0)) return kNgs2ErrInvalidSystem;
    }
    struct RenderBufferInfo { uint64_t buffer, size; uint32_t waveform_type, channels; };
    if (!a1 || a2 == 0 || a2 > 16) return kNgs2ErrInvalidOut;
    for (uint64_t i = 0; i < a2; ++i) {
        RenderBufferInfo info{};
        if (!ngs2_read_bytes(a1 + i * sizeof(info), &info, sizeof(info)) ||
            !info.buffer || info.size > 64ull * 1024 * 1024 || !ngs2_zero_bytes(info.buffer, (size_t)info.size))
            return kNgs2ErrInvalidOut;
    }
    return 0;
}

HLE(ngs2_geom_reset_listener) {
    NGS2_LOG("sceNgs2GeomResetListenerParam");
    return a0 && a2_store_zeros(a0, 0xa0) ? 0 : kNgs2ErrInvalidOut;
}

HLE(ngs2_geom_calc_listener) {
    NGS2_LOG("sceNgs2GeomCalcListener");
    return a0 && a1 && a2_store_zeros(a1, 0x60) ? 0 : kNgs2ErrInvalidOut;
}

#undef NGS2_LOG

void register_audio_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceAudioOutInit", audio_init);
    R("sceAudioOutOpen", audio_open);
    R("sceAudioOutOutput", audio_output);
    R("sceAudioOutOutputs", audio_outputs);
    R("sceAudioOutSetVolume", audio_set_volume);
    R("sceAudioOutClose", audio_close);
    R("sceAudioOutGetPortState", audio_get_port_state);
    // libSceAudioOut2 (PS5) — see the probe block above.
    R("sceAudioOut2Initialize", audio2_initialize);
    R("sceAudioOut2ContextResetParam", audio2_ctx_reset_param);
    R("sceAudioOut2ContextQueryMemory", audio2_ctx_query_memory);
    R("sceAudioOut2ContextCreate", audio2_ctx_create);
    R("sceAudioOut2ContextDestroy", audio2_ctx_destroy);
    R("sceAudioOut2ContextAdvance", audio2_ctx_advance);
    R("sceAudioOut2ContextPush", audio2_ctx_push);
    R("sceAudioOut2ContextGetQueueLevel", audio2_ctx_get_queue_level);
    R("sceAudioOut2ContextSetAttributes", audio2_ctx_set_attr);
    R("sceAudioOut2ContextBedWrite", audio2_ctx_bed_write);
    R("sceAudioOut2UserCreate", audio2_user_create);
    R("sceAudioOut2UserDestroy", audio2_user_destroy);
    R("sceAudioOut2PortCreate", audio2_port_create);
    R("sceAudioOut2PortDestroy", audio2_port_destroy);
    R("sceAudioOut2PortGetState", audio2_port_get_state);
    R("sceAudioOut2PortSetAttributes", audio2_port_set_attr);
    R("sceAudioOut2PortRegister", audio2_port_register);
    R("sceAudioOut2PortUnregister", audio2_port_unregister);
    R("sceAudioOut2GetSystemState", audio2_get_system_state);
    R("sceAudioOut2GetSpeakerInfo", audio2_get_speaker_info);
    R("sceAudioOut2MasteringInit", audio2_mastering_init);
    R("sceAudioOut2MasteringTerm", audio2_mastering_term);
    R("sceAudioOut2MasteringSetParam", audio2_mastering_set_param);
    R("sceAudioOut2MasteringGetState", audio2_mastering_get_state);
    // libSceAjm (#187): headless decode-lifecycle (valid handles, no actual decode -> silence).
    R("sceAjmInitialize", ajm_initialize);            R("sceAjmFinalize", ajm_finalize);
    R("sceAjmModuleRegister", ajm_module_register);   R("sceAjmModuleUnregister", ajm_module_unregister);
    R("sceAjmInstanceCreate", ajm_instance_create);   R("sceAjmInstanceDestroy", ajm_instance_destroy);
    R("sceAjmBatchStartBuffer", ajm_batch_start);     R("sceAjmBatchWait", ajm_batch_wait);
    R("sceAjmBatchErrorDump", ajm_batch_errordump);
    Hle::register_fn("pgFAiLR5qT4", ngs2_system_query_buffer, "sceNgs2SystemQueryBufferSize");
    Hle::register_fn("koBbCMvOKWw", ngs2_system_create, "sceNgs2SystemCreate");
    Hle::register_fn("0eFLVCfWVds", ngs2_rack_query_buffer, "sceNgs2RackQueryBufferSize");
    Hle::register_fn("cLV4aiT9JpA", ngs2_rack_create, "sceNgs2RackCreate");
    Hle::register_fn("MwmHz8pAdAo", ngs2_rack_get_voice, "sceNgs2RackGetVoiceHandle");
    Hle::register_fn("uu94irFOGpA", ngs2_voice_control, "sceNgs2VoiceControl");
    Hle::register_fn("AbYvTOZ8Pts", ngs2_voice_run_commands, "sceNgs2VoiceRunCommands");
    Hle::register_fn("-TOuuAQ-buE", ngs2_voice_get_state, "sceNgs2VoiceGetState");
    Hle::register_fn("0lbbayqDNoE", ngs2_geom_reset_source, "sceNgs2GeomResetSourceParam");
    Hle::register_fn("i0VnXM-C9fc", ngs2_system_render, "sceNgs2SystemRender");
    Hle::register_fn("7Lcfo8SmpsU", ngs2_geom_reset_listener, "sceNgs2GeomResetListenerParam");
    Hle::register_fn("1WsleK-MTkE", ngs2_geom_calc_listener, "sceNgs2GeomCalcListener");
    #undef R
}

} // namespace prosper
