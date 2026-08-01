// hle_audio.cpp — libSceAudioOut/AudioIn HLE, with pluggable output (see audio.hpp).
//
// Decodes the PS5 sceAudioOut* calls into port lifecycle + interleaved PCM grains and forwards
// them to the installed backend. The inherited AudioIn core provides deterministic paced silence.
// prosper_core stays dependency-free; a concrete output frontend (SDL3, ...) installs itself via
// audio_set_sink() from outside the core.
#include "dispatch.hpp"
#include "../host/boot_program.hpp"   // #1659: shared guest-module labelling
#include "nid.hpp"
#include "audio.hpp"
#include "ajm_decoder.hpp"     // optional host codecs (MP3); core retains AJM ABI + guest copies
#include "atrac9_decode.hpp"    // vendored LibAtrac9 glue — AJM ATRAC9 batch decode (Blasphemous 2)
#include "callback_fs.hpp"      // recover the caller's guest %fs for firing guest callbacks
#include <memory>
#include "../host/posix_shim.hpp" // PROSPER_ASM_TRAMPOLINE (pass entry %rsp as 7th arg)
#include <algorithm>
#include <atomic>
#include <deque>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <sys/uio.h>
#include <unistd.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include "../host/posix_shim.hpp"   // Darwin: process_vm_readv/writev

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define HLE8(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6, \
                                       uint64_t a7)
#define HLE10(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6, \
                                       uint64_t a7, uint64_t a8, uint64_t a9)
#define P(x) ((void*)(uintptr_t)(x))

namespace {

constexpr int kMaxPorts  = 16;
// Host-only slots above the 16 public sceAudioOut handles keep each of the four AudioOut2
// contexts on an independent device stream. The host device mixes those streams concurrently;
// serializing multiple contexts through one paced stream would insert one context's grains into
// another context's timeline.
constexpr int kMaxSinkPorts = kMaxPorts + 4;
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
    Pace p_[kMaxSinkPorts];
    bool open(int port, const AudioPortInfo& info) override {
        if (port < 1 || port > kMaxSinkPorts) return false;
        int freq = info.freq > 0 ? info.freq : 48000;
        int grain = info.grain > 0 ? info.grain : 256;
        p_[port - 1].ns_per_grain = (long long)grain * 1000000000LL / freq;
        p_[port - 1].next = {};
        return true;
    }
    void output(int port, const void*, int frames) override {
        if (port < 1 || port > kMaxSinkPorts) return;
        auto& s = p_[port - 1];
        long long ns = s.ns_per_grain > 0 ? s.ns_per_grain : ((long long)frames * 1000000000LL / 48000);
        auto now = std::chrono::steady_clock::now();
        auto dur = std::chrono::nanoseconds(ns);
        // (Re)sync if unset or we fell far behind (e.g. after a stall) to avoid burst catch-up.
        if (s.next.time_since_epoch().count() == 0 || s.next < now - dur * 4) s.next = now;
        s.next += dur;
        if (s.next > now) std::this_thread::sleep_until(s.next);
    }
    void close(int port) override { if (port >= 1 && port <= kMaxSinkPorts) p_[port - 1] = {}; }
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
static void audio_in_reset_ports();
static void audio2_reset();

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

void audio_apply_channel_volumes(int dst[8], uint32_t mask, const int* vols) {
    if (!dst || !vols) return;
    for (int c = 0; c < 8; c++)
        if (mask & (1u << c)) dst[c] = vols[c];
}

int audio_peak_channel_volume(uint32_t mask, const int* vols) {
    if (!vols) return 0;
    int peak = 0;
    for (int c = 0; c < 8; c++)
        if ((mask & (1u << c)) && vols[c] > peak) peak = vols[c];
    return peak;
}

bool audio2_reserve_queue_slot(uint32_t& queued, uint32_t queue_depth) {
    if (queued >= queue_depth) return false;
    ++queued;
    return true;
}

void audio_reset() {
    AudioSink* s = audio_sink();
    {
        std::lock_guard<std::mutex> lk(g_mx);
        for (int i = 0; i < kMaxPorts; i++) {
            if (g_ports[i].in_use && s) s->close(i + 1);
            g_ports[i] = Port{};
        }
    }
    // Close AudioOut2's host-only context streams while the currently installed sink is still
    // reachable, then restore the headless sink for the next test/application lifecycle.
    audio2_reset();
    g_sink.store(&g_default_sink);
    audio_in_reset_ports();
}

// --- legacy-path diagnostics --------------------------------------------------------------
// PROSPER_AUDIOLOG=1: log port opens (raw args + decoded format), SetVolume args, and a
// once-per-second PCM peak per port, so a silent/quiet/garbled title shows WHERE the signal
// degrades (no output calls vs silent PCM vs wrong format vs volume mapping).
// PROSPER_AUDIO_DUMP=PATH: append each port's raw output() grains to PATH.portN.raw for
// offline analysis, mirroring PROSPER_SHADER_DUMP's capture-first workflow.
namespace {

int audiolog_level() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PROSPER_AUDIOLOG"); v = (e && *e) ? atoi(e) : 0; if (v < 0) v = 0; }
    return v;
}
bool audiolog() { return audiolog_level() >= 1; }

const char* audio_dump_path() {
    static const char* p = getenv("PROSPER_AUDIO_DUMP");
    return (p && *p) ? p : nullptr;
}

// Peak |sample| of one grain, normalized to [0,1] for either sample format.
double audio_pcm_peak(const void* pcm, int frames, const AudioPortInfo& info) {
    if (!pcm || frames <= 0) return 0.0;
    const int n = frames * info.channels;
    double peak = 0.0;
    if (info.fmt == AudioFmt::F32) {
        const float* s = (const float*)pcm;
        for (int i = 0; i < n; i++) { double v = s[i] < 0 ? -(double)s[i] : (double)s[i]; if (v > peak) peak = v; }
    } else {
        const int16_t* s = (const int16_t*)pcm;
        for (int i = 0; i < n; i++) { double v = (s[i] < 0 ? -(double)s[i] : (double)s[i]) / 32768.0; if (v > peak) peak = v; }
    }
    return peak;
}

// Once-per-second peak report + optional raw dump. Called from the output paths (guest audio
// thread) with the port's decoded info; per-port state, no cross-port locking needed beyond
// the distinct slots.
void audio_observe_output(int handle, const void* pcm, int frames, const AudioPortInfo& info) {
    if (!audiolog() && !audio_dump_path()) return;
    if (handle < 1 || handle > kMaxSinkPorts) return;
    static struct Obs { uint64_t calls = 0; double peak = 0.0; FILE* dump = nullptr;
                        std::chrono::steady_clock::time_point last{}; } st[kMaxSinkPorts];
    Obs& s = st[handle - 1];
    s.calls++;
    double p = audio_pcm_peak(pcm, frames, info);
    if (p > s.peak) s.peak = p;
    if (audiolog()) {
        auto now = std::chrono::steady_clock::now();
        if (s.last.time_since_epoch().count() == 0) s.last = now;
        if (now - s.last >= std::chrono::seconds(1)) {
            fprintf(stderr, "[audio] port %d: %llu output calls, 1s-peak=%.4f (fmt=%s ch=%d freq=%d grain=%d)\n",
                    handle, (unsigned long long)s.calls, s.peak,
                    info.fmt == AudioFmt::F32 ? "f32" : "s16", info.channels, info.freq, info.grain);
            s.last = now; s.peak = 0.0;
        }
    }
    if (const char* base = audio_dump_path()) {
        if (!s.dump) {
            char path[1024];
            snprintf(path, sizeof path, "%s.port%d.raw", base, handle);
            s.dump = fopen(path, "ab");
            fprintf(stderr, "[audio] port %d: dumping raw PCM to %s (fmt=%s ch=%d freq=%d grain=%d)\n",
                    handle, path, info.fmt == AudioFmt::F32 ? "f32" : "s16",
                    info.channels, info.freq, info.grain);
        }
        if (s.dump && pcm && frames > 0) fwrite(pcm, 1, (size_t)frames * audio_frame_bytes(info), s.dump);
    }
}

} // namespace

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
    if (audiolog())
        fprintf(stderr, "[audio] open: handle=%d userId=%d type=%d index=%d len=%llu freq=%llu param=0x%llx"
                        " -> fmt=%s ch=%d grain=%d\n",
                handle, (int)a0, type, (int)a2, (unsigned long long)a3, (unsigned long long)a4,
                (unsigned long long)a5, info.fmt == AudioFmt::F32 ? "f32" : "s16",
                info.channels, info.grain);
    if (auto* s = audio_sink()) s->open(handle, info);
    return (uint64_t)handle;
}

// sceAudioOutOutput(handle, ptr) -> frames written (>=0) or negative error. ptr==0 => drain.
HLE(audio_output) {
    AudioPortInfo info;
    { std::lock_guard<std::mutex> lk(g_mx); Port* p = port_of((int)a0); if (!p) return kAudioErrInvalidPort; info = p->info; }
    if (a1 == 0) return 0;   // drain/flush: nothing buffered in the headless model
    audio_observe_output((int)a0, P(a1), info.grain, info);
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
        if (arr[i].ptr) { audio_observe_output(arr[i].handle, P(arr[i].ptr), info.grain, info);
                          if (auto* s = audio_sink()) s->output(arr[i].handle, P(arr[i].ptr), info.grain); }
        if (!have) { grain = info.grain; have = true; }
    }
    return grain;
}

// sceAudioOutSetVolume(handle, flag(channel mask), int vol[]) -> 0 or negative error.
HLE(audio_set_volume) {
    uint32_t mask = (uint32_t)a1;
    const int* vols = (const int*)P(a2);
    if (audiolog() && vols)
        fprintf(stderr, "[audio] set_volume: handle=%d mask=0x%x vols=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                (int)a0, mask, vols[0], vols[1], vols[2], vols[3], vols[4], vols[5], vols[6], vols[7]);
    { std::lock_guard<std::mutex> lk(g_mx);
      Port* p = port_of((int)a0); if (!p) return kAudioErrInvalidPort;
      audio_apply_channel_volumes(p->vol, mask, vols); }
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
            case 2: case 3: *(uint16_t*)(st + 0) = 0x40; st[2] = 1; break;                // voice/personal -> headphone
            case 4:         *(uint16_t*)(st + 0) = 0x04; st[2] = 1; break;                // pad speaker
            case 127:       *(uint16_t*)(st + 0) = 0x80; break;                           // aux -> external
            default:        *(uint16_t*)(st + 0) = 0x01;
                            st[2] = (uint8_t)(channels > 2 ? 2 : channels); break;          // main/bgm -> primary
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
            prosper::guest_module_name((uint64_t)ra),
            (unsigned long long)prosper::guest_module_offset((uint64_t)ra));
}

} // namespace

#define A2LOG(name) a2_log(name, a0, a1, a2, a3, a4, a5, __builtin_return_address(0))

// AudioOut2 handle space. Handles are opaque u64s the guest stores and passes back; tag them so
// stray guest values are distinguishable in logs. One context + a few ports is all CRI uses.
constexpr uint64_t kA2CtxTag  = 0xA2C0000000000000ull;
constexpr uint64_t kA2UserTag = 0xA2D0000000000000ull;
constexpr uint64_t kA2PortTag = 0xA2E0000000000000ull;
constexpr uint64_t kA2SpeakerArrayTag = 0xA2F0000000000000ull;
constexpr uint64_t kA2HandleTagMask = 0xffff000000000000ull;
constexpr uint64_t kA2HandleIndexMask = 0xffffull;
constexpr uint32_t kA2MaxPorts = 256;
// Main-bed port flag bit 1 carries 20 dB of intentional digital headroom for AudioOut2's platform
// mastering stage.  Restore that reference level at the host boundary; flag-zero ports are already
// consumer-level PCM and must remain unchanged (notably video playback and ordinary stereo titles).
constexpr uint32_t kA2PortFlag20DbHeadroom = 1u << 1;
constexpr float kA2HeadroomGain = 10.0f;

struct A2Context {
    bool     used = false;
    uint32_t generation = 0;
    bool     sink_opened = false;
    bool     sink_open_ok = false;
    uint32_t queue_depth = 4;
    uint32_t grain = 256;         // samples per Advance/Push cycle (param +0x10, live: 0x100)
    uint32_t queued = 0;          // submitted grains not yet consumed by the 48 kHz device clock
    std::chrono::steady_clock::time_point last_queue_update{};
    // Real-time pacing state for ContextPush (blocking-when-full HW semantics).
    std::chrono::steady_clock::time_point next{};
};
std::mutex g_a2_mx;
// Serializes each context slot's host stream across open/output/close. Always acquire a slot mutex
// before g_a2_mx when both are needed, so Destroy cannot close a stream reopened by a recycled
// context and an in-flight Push cannot output after that slot has been torn down.
std::mutex g_a2_sink_mx[4];
A2Context  g_a2_ctx[4];
uint32_t   g_a2_users = 0;

// AudioOut2 data path (derived live from Evergate/GTA V and cross-checked against Kyty's public
// reverse-engineered structs):
// each tick the guest calls PortSetAttributes(port, attrs, count) where an attribute triple
// {u32 id=0, u32 reserved, value_ptr, value_size=8} carries a guest pointer to the port's current grain of
// interleaved PCM (grain frames from the context param, 256 live). ContextAdvance
// advances engine state and ContextPush submits the grain to the device. We store each port's PCM
// pointer here and mix + forward all ports' grains to the host AudioSink at Push.
struct A2PortState {
    bool     used = false;
    uint32_t generation = 0;
    uint64_t context = 0;      // owning context; Push must not submit another context's ports
    uint64_t pcm_ptr = 0;      // guest address of this port's current PCM grain (attr id 0)
    uint16_t type = 0;         // portParam +0x00: 0 = MAIN (speaker output); others aux (personal/...)
    uint32_t data_format = 0;  // portParam +0x04: bits 8..15=channels, bits 0..6=0 F32 / 1 S16,
                               // bit 7 selects the standard 8-channel order
    uint32_t flags = 0;        // portParam +0x0c: device/mastering mode bits
};
A2PortState g_a2_port_state[kA2MaxPorts];
bool g_a2_speaker_arrays[32]{};
constexpr int kA2SinkPortBase = kMaxPorts + 1; // host-only ports 17..20, one per AudioOut2 context

uint32_t audio2_next_generation(uint32_t generation) {
    ++generation;
    return generation ? generation : 1;
}

uint64_t audio2_make_handle(uint64_t tag, uint32_t generation, uint32_t one_based_index) {
    return tag | ((uint64_t)generation << 16) | one_based_index;
}

template <typename Slot>
void audio2_clear_slot(Slot& slot) {
    const uint32_t generation = slot.generation;
    slot = Slot{};
    slot.generation = generation;
}

// Callers hold g_a2_mx. Centralizing the full tag/index/generation/live checks prevents a stale
// handle from aliasing a recycled slot and keeps every implemented AudioOut2 operation consistent.
A2Context* audio2_context_locked(uint64_t handle, uint32_t* slot_out = nullptr) {
    const uint32_t one_based_index = (uint32_t)(handle & kA2HandleIndexMask);
    const uint32_t generation = (uint32_t)(handle >> 16);
    if ((handle & kA2HandleTagMask) != kA2CtxTag || one_based_index < 1 ||
        one_based_index > std::size(g_a2_ctx) || !generation) return nullptr;
    A2Context& context = g_a2_ctx[one_based_index - 1];
    if (!context.used || context.generation != generation) return nullptr;
    if (slot_out) *slot_out = one_based_index - 1;
    return &context;
}

A2PortState* audio2_port_locked(uint64_t handle, uint32_t* slot_out = nullptr) {
    const uint32_t one_based_index = (uint32_t)(handle & kA2HandleIndexMask);
    const uint32_t generation = (uint32_t)(handle >> 16);
    if ((handle & kA2HandleTagMask) != kA2PortTag || one_based_index < 1 ||
        one_based_index > kA2MaxPorts || !generation) return nullptr;
    A2PortState& port = g_a2_port_state[one_based_index - 1];
    if (!port.used || port.generation != generation) return nullptr;
    if (slot_out) *slot_out = one_based_index - 1;
    return &port;
}

uint32_t audio2_format_channels(uint32_t data_format) {
    const uint32_t encoded = (data_format >> 8u) & 0xffu;
    return encoded ? std::min(encoded, 16u) : 2u;
}

static void audio2_reset() {
    std::scoped_lock sink_locks(g_a2_sink_mx[0], g_a2_sink_mx[1],
                                g_a2_sink_mx[2], g_a2_sink_mx[3]);
    bool close_sink[4]{};
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        for (size_t i = 0; i < 4; ++i) {
            close_sink[i] = g_a2_ctx[i].sink_opened;
            audio2_clear_slot(g_a2_ctx[i]);
        }
        for (auto& port : g_a2_port_state) audio2_clear_slot(port);
        std::fill(std::begin(g_a2_speaker_arrays), std::end(g_a2_speaker_arrays), false);
        g_a2_users = 0;
    }
    if (AudioSink* sink = audio_sink())
        for (size_t i = 0; i < 4; ++i)
            if (close_sink[i]) sink->close(kA2SinkPortBase + (int)i);
}

// Fault-safe store to a guest out-pointer (same rationale as apr_write_guest_dst: a bad pointer
// must fail the call, not SIGSEGV inside the HLE). WriteProcessMemory validates the complete
// destination range before copying, matching the all-or-fail process_vm_writev contract here.
bool audio_store_bytes(uint64_t dst, const void* src, size_t n) {
    if (!dst || (!src && n)) return false;
    if (!n) return true;
#ifndef _WIN32
    struct iovec l { const_cast<void*>(src), n }, r { (void*)(uintptr_t)dst, n };
    return process_vm_writev(getpid(), &l, 1, &r, 1, 0) == (ssize_t)n;
#else
    SIZE_T written = 0;
    return WriteProcessMemory(GetCurrentProcess(), (void*)(uintptr_t)dst, src, n, &written) &&
           written == n;
#endif
}

// Fault-safe read from guest memory. Both platform paths require the complete range so callers
// never consume a partially copied guest structure after an inaccessible-range failure.
bool audio_read_bytes(uint64_t src, void* dst, size_t n) {
    if (!src || (!dst && n)) return false;
    if (!n) return true;
#ifndef _WIN32
    struct iovec l { dst, n }, r { (void*)(uintptr_t)src, n };
    return process_vm_readv(getpid(), &l, 1, &r, 1, 0) == (ssize_t)n;
#else
    SIZE_T read = 0;
    return ReadProcessMemory(GetCurrentProcess(), (const void*)(uintptr_t)src, dst, n, &read) &&
           read == n;
#endif
}

// Best-effort read: copy as many leading bytes of [src, src+n) as are accessible, returning the
// count actually read (0 on total failure). Unlike audio_read_bytes this tolerates a short/partial
// guest range — the NGS2 mixer needs to consume whatever valid PCM a voice still has this render.
size_t audio_read_bytes_partial(uint64_t src, void* dst, size_t n) {
    if (!src || !dst || !n) return 0;
#ifndef _WIN32
    struct iovec l { dst, n }, r { (void*)(uintptr_t)src, n };
    ssize_t got = process_vm_readv(getpid(), &l, 1, &r, 1, 0);
    if (got > 0) return (size_t)got;
    // process_vm_readv is all-or-nothing per iovec; on failure, binary-search the readable prefix so a
    // block that ends partway through the requested span still yields its valid head.
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo + 1) / 2;
        struct iovec l2 { dst, mid }, r2 { (void*)(uintptr_t)src, mid };
        if (process_vm_readv(getpid(), &l2, 1, &r2, 1, 0) == (ssize_t)mid) lo = mid; else hi = mid - 1;
    }
    return lo;
#else
    SIZE_T read = 0;
    ReadProcessMemory(GetCurrentProcess(), (const void*)(uintptr_t)src, dst, n, &read);
    return (size_t)read;
#endif
}

bool a2_store_u32(uint64_t dst, uint32_t v) { return audio_store_bytes(dst, &v, sizeof v); }
bool a2_store_u64(uint64_t dst, uint64_t v) { return audio_store_bytes(dst, &v, sizeof v); }
bool a2_store_zeros(uint64_t dst, size_t n) {
    std::vector<uint8_t> z(n, 0);
    return audio_store_bytes(dst, z.data(), n);
}

// ---- libSceAudioIn: headless, silent, real-time paced ----------------------------------
//
// The inherited PS4/PS5 core ABI has seven ports. Handles carry the public 0x30000000 tag,
// the port type in bits 16..23, and a zero-based port id in bits 0..7. We deliberately do not
// open a host microphone: returning exact-size silence gives privacy-preserving deterministic
// behavior, while grain/frequency pacing preserves the blocking capture-thread contract.
namespace {

constexpr int      kAudioInMaxPorts  = 7;
constexpr uint32_t kAudioInHandleTag = 0x30000000u;
constexpr uint64_t kAudioInErrInvalidHandle = (uint64_t)(int64_t)(int32_t)0x80260101;
constexpr uint64_t kAudioInErrInvalidSize   = (uint64_t)(int64_t)(int32_t)0x80260102;
constexpr uint64_t kAudioInErrInvalidFreq   = (uint64_t)(int64_t)(int32_t)0x80260103;
constexpr uint64_t kAudioInErrInvalidPtr    = (uint64_t)(int64_t)(int32_t)0x80260105;
constexpr uint64_t kAudioInErrInvalidParam  = (uint64_t)(int64_t)(int32_t)0x80260106;
constexpr uint64_t kAudioInErrPortFull      = (uint64_t)(int64_t)(int32_t)0x80260107;
constexpr uint64_t kAudioInErrNotOpened     = (uint64_t)(int64_t)(int32_t)0x80260109;

struct AudioInPort {
    bool in_use = false;
    uint32_t grain = 0;
    uint32_t freq = 0;
    uint32_t channels = 0;
    std::chrono::steady_clock::time_point next{};
};

std::mutex  g_audio_in_mx;
AudioInPort g_audio_in_ports[kAudioInMaxPorts];

int audio_in_port_id(uint64_t raw_handle) {
    uint32_t handle = (uint32_t)raw_handle;
    if ((handle & 0x7f000000u) != kAudioInHandleTag) return -1;
    uint32_t id = handle & 0xffu;
    return id < kAudioInMaxPorts ? (int)id : -1;
}

} // namespace

static void audio_in_reset_ports() {
    std::lock_guard<std::mutex> lk(g_audio_in_mx);
    for (auto& port : g_audio_in_ports) port = {};
}

HLE(audio_in_init) { return 0; }

// sceAudioInOpen(userId, type, index, len, freq, param) -> tagged handle or AudioIn error.
// Public formats are S16 mono (0) and S16 stereo (2); supported rates are 16/48 kHz and the
// hardware grain limit is 1..2048 frames. userId/type/index policy is left to the guest-facing
// service just as in the reference implementation; type is retained in the opaque handle.
HLE(audio_in_open) {
    (void)a0; (void)a2;
    uint32_t grain = (uint32_t)a3;
    uint32_t freq = (uint32_t)a4;
    uint32_t format = (uint32_t)a5;
    if (!grain || grain > 2048) return kAudioInErrInvalidSize;
    if (freq != 16000 && freq != 48000) return kAudioInErrInvalidFreq;
    uint32_t channels;
    if (format == 0) channels = 1;
    else if (format == 2) channels = 2;
    else return kAudioInErrInvalidParam;

    std::lock_guard<std::mutex> lk(g_audio_in_mx);
    for (int id = 0; id < kAudioInMaxPorts; id++) {
        if (g_audio_in_ports[id].in_use) continue;
        g_audio_in_ports[id] = {true, grain, freq, channels, {}};
        return (uint32_t)(kAudioInHandleTag | (((uint32_t)a1 & 0xffu) << 16) | (uint32_t)id);
    }
    return kAudioInErrPortFull;
}

// sceAudioInInput(handle, dst) blocks for one capture grain and returns frames captured. The
// null backend writes exactly grain*channels S16 samples. Fault-safe output preserves the HLE
// process when a guest supplies an inaccessible range.
HLE(audio_in_input) {
    int id = audio_in_port_id(a0);
    if (id < 0) return kAudioInErrInvalidHandle;
    if (!a1) return kAudioInErrInvalidPtr;

    std::chrono::steady_clock::time_point deadline;
    uint32_t grain;
    {
        std::lock_guard<std::mutex> lk(g_audio_in_mx);
        AudioInPort& port = g_audio_in_ports[id];
        if (!port.in_use) return kAudioInErrNotOpened;
        size_t bytes = (size_t)port.grain * port.channels * sizeof(int16_t);
        if (!a2_store_zeros(a1, bytes)) return kAudioInErrInvalidPtr;

        grain = port.grain;
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::nanoseconds((uint64_t)port.grain * 1000000000ull / port.freq);
        if (port.next.time_since_epoch().count() == 0 || port.next < now - duration * 4)
            port.next = now;
        port.next += duration;
        deadline = port.next;
    }
    if (deadline > std::chrono::steady_clock::now()) std::this_thread::sleep_until(deadline);
    return grain;
}

HLE(audio_in_close) {
    int id = audio_in_port_id(a0);
    if (id < 0) return kAudioInErrInvalidHandle;
    std::lock_guard<std::mutex> lk(g_audio_in_mx);
    if (!g_audio_in_ports[id].in_use) return kAudioInErrNotOpened;
    g_audio_in_ports[id] = {};
    return 0;
}

constexpr uint64_t kA2ErrInvalidParam = (uint64_t)(int64_t)(int32_t)0x80268001;
constexpr uint64_t kA2ErrNotReady = (uint64_t)(int64_t)(int32_t)0x80268008;
constexpr uint64_t kA2ErrPortFull = (uint64_t)(int64_t)(int32_t)0x80268012;

// Advance the emulated hardware queue by elapsed 48 kHz grains. Caller holds g_a2_mx. Keeping this
// state independent of the host sink matters because GetQueueLevel is an observable device clock:
// middleware uses it to pace state changes even when the backend happens to consume synchronously.
void audio2_update_queue_locked(A2Context& context,
                                std::chrono::steady_clock::time_point now) {
    if (!context.queued) {
        context.last_queue_update = now;
        return;
    }
    const uint32_t grain = context.grain ? context.grain : 256;
    const auto grain_duration = std::chrono::nanoseconds(
        (long long)grain * 1000000000LL / 48000);
    if (grain_duration.count() <= 0 || context.last_queue_update.time_since_epoch().count() == 0) {
        context.last_queue_update = now;
        return;
    }
    if (now <= context.last_queue_update) return;
    const auto elapsed = now - context.last_queue_update;
    const uint64_t drained = std::min<uint64_t>(
        context.queued, (uint64_t)(elapsed / grain_duration));
    if (!drained) return;
    context.queued -= (uint32_t)drained;
    context.last_queue_update += grain_duration * (long long)drained;
    if (!context.queued) context.last_queue_update = now;
}

// sceAudioOut2Initialize(void) -> 0. Idempotent success (same as sceAudioOutInit).
HLE(audio2_initialize) { A2LOG("sceAudioOut2Initialize"); return 0; }

// sceAudioOut2ContextResetParam(SceAudioOut2ContextParam* param) -> 0.  A reset is not merely a
// memset: titles may retain any default they do not override.  In particular queue_depth=4,
// num_grains=512, and flags=1 are part of the observable ABI.
HLE(audio2_ctx_reset_param) {
    A2LOG("sceAudioOut2ContextResetParam");
    struct ContextParam {
        uint32_t max_ports, max_object_ports, guarantee_object_ports;
        uint32_t queue_depth, num_grains, flags, reserved[10];
    } param{};
    static_assert(sizeof(ContextParam) == 0x40);
    param.max_ports = 256;
    param.max_object_ports = 256;
    param.queue_depth = 4;
    param.num_grains = 512;
    param.flags = 1;
    if (!audio_store_bytes(a0, &param, sizeof param)) return kA2ErrInvalidParam;
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
    if (!a2_store_u64(a1, 0x100000)) return kA2ErrInvalidParam;
    return 0;
}

// sceAudioOut2GetSpeakerArrayMemorySize(...) -> size (GTA V / PPSA04263, RAGE, issue #1134).
// RETURNS the speaker-array work-memory byte size directly (rax); the guest uses it verbatim as an
// allocation size: `r15 = ret; ptr = allocator->alloc(r15, 0x10)`. Live [RAGE] disassembly at
// eboot+0x2adf25e: arg rdi=8 (speaker/channel config). Stubbed to 0 the guest allocated a ZERO-byte
// speaker-array buffer and then overran it, aborting RAGE audio/streaming init with its int 0x41
// fatal — the deterministic pre-render crash on this title.  The SDK sizing contract is 0x400 bytes
// of base state plus 0x40 per VBAP speaker or 0x100 per ambisonics speaker, with 0x200 extra for 3D.
HLE(audio2_get_speaker_array_memory_size) {
    A2LOG("sceAudioOut2GetSpeakerArrayMemorySize");
    const uint64_t speakers = std::clamp<uint64_t>(a0, 1, 32);
    return 0x400 + speakers * (a2 ? 0x100 : 0x40) + (a1 ? 0x200 : 0);
}

// sceAudioOut2ContextCreate(param*, mem, memSize, Handle* outCtx) -> 0.
HLE(audio2_ctx_create) {
    A2LOG("sceAudioOut2ContextCreate");
    uint32_t grain = 256;
    uint32_t queue_depth = 4;
    if (a0) {
        uint32_t param[5]{};
        if (audio_read_bytes(a0, param, sizeof param)) {
            if (param[3] >= 1 && param[3] <= 64) queue_depth = param[3];
            if (param[4] >= 64 && param[4] <= 4096) grain = param[4];
        }
    }
    std::lock_guard<std::mutex> lk(g_a2_mx);
    for (int i = 0; i < 4; i++) {
        if (g_a2_ctx[i].used) continue;
        const uint32_t generation = audio2_next_generation(g_a2_ctx[i].generation);
        g_a2_ctx[i] = A2Context{};
        g_a2_ctx[i].used = true;
        g_a2_ctx[i].generation = generation;
        g_a2_ctx[i].queue_depth = queue_depth;
        g_a2_ctx[i].grain = grain;
        if (!a2_store_u64(a3, audio2_make_handle(kA2CtxTag, generation, i + 1))) {
            audio2_clear_slot(g_a2_ctx[i]);
            return kA2ErrInvalidParam;
        }
        return 0;
    }
    return kA2ErrInvalidParam;
}
HLE(audio2_ctx_destroy) {
    A2LOG("sceAudioOut2ContextDestroy");
    const uint32_t one_based_index = (uint32_t)(a0 & kA2HandleIndexMask);
    if ((a0 & kA2HandleTagMask) != kA2CtxTag || one_based_index < 1 ||
        one_based_index > std::size(g_a2_ctx)) return kA2ErrInvalidParam;
    const uint32_t context_slot = one_based_index - 1;
    std::lock_guard<std::mutex> sink_lk(g_a2_sink_mx[context_slot]);
    bool close_sink = false;
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        A2Context* context = audio2_context_locked(a0);
        if (!context) return kA2ErrInvalidParam;
        close_sink = context->sink_opened;
        audio2_clear_slot(*context);
        for (auto& port : g_a2_port_state)
            if (port.used && port.context == a0) audio2_clear_slot(port);
    }
    if (close_sink)
        if (AudioSink* sink = audio_sink()) sink->close(kA2SinkPortBase + (int)context_slot);
    return 0;
}

// sceAudioOut2UserCreate(userId, Handle* out) -> 0. (live: userId=0xff)
HLE(audio2_user_create) {
    A2LOG("sceAudioOut2UserCreate");
    std::lock_guard<std::mutex> lk(g_a2_mx);
    if (!a2_store_u64(a1, kA2UserTag | (uint64_t)++g_a2_users)) return kA2ErrInvalidParam;
    return 0;
}
HLE(audio2_user_destroy) { A2LOG("sceAudioOut2UserDestroy"); return 0; }

// sceAudioOut2PortCreate(ctx, const portParam*, Handle* outPort, ...) -> 0.
HLE(audio2_port_create) {
    A2LOG("sceAudioOut2PortCreate");
    if (audio2log()) a2_dump("portParam", a1, 0x40);
    // AudioOut2PortParam: u16 port_type @0, u32 data_format @4, u32 sampling_freq @8.  The format is
    // NOT a block length: 0x800 means eight-channel float PCM, while 0x100/0x200 mean mono/stereo.
    // Confusing it for `channels * grain` happens to work at grain=256, but fails for every other
    // grain and leaves PortGetState unable to report the format that the guest mixer consumes.
    uint16_t ptype = 0;
    uint32_t data_format = 0;
    uint32_t flags = 0;
    if (a1) {
        struct { uint16_t type, pad; uint32_t format; uint32_t rate, flags; } hdr{};
        if (audio_read_bytes(a1, &hdr, sizeof hdr)) {
            ptype = hdr.type;
            data_format = hdr.format;
            flags = hdr.flags;
        }
    }
    std::lock_guard<std::mutex> lk(g_a2_mx);
    if (!audio2_context_locked(a0)) return kA2ErrInvalidParam;
    // First-free-slot allocation (not a monotonic counter): slots are recycled on PortDestroy, so a
    // title that churns ports over a long session can't exhaust the table. The generation in each
    // returned handle keeps an old identity from aliasing the recycled slot.
    for (uint32_t id = 1; id <= kA2MaxPorts; ++id) {
        if (g_a2_port_state[id - 1].used) continue;
        A2PortState& port = g_a2_port_state[id - 1];
        const uint32_t generation = audio2_next_generation(port.generation);
        port.generation = generation; // failed out-pointer publication still consumes this identity
        const uint64_t handle = audio2_make_handle(kA2PortTag, generation, id);
        if (!a2_store_u64(a2, handle)) return kA2ErrInvalidParam;
        port.used = true;
        port.context = a0;
        port.pcm_ptr = 0;
        port.type = ptype;
        port.data_format = data_format;
        port.flags = flags;
        return 0;
    }
    return kA2ErrPortFull;
}
HLE(audio2_port_destroy) {
    A2LOG("sceAudioOut2PortDestroy");
    // Clear the slot so a destroyed port stops being mixed (its guest PCM buffer may be freed and
    // reused) and the slot is reusable by the next PortCreate.
    std::lock_guard<std::mutex> lk(g_a2_mx);
    A2PortState* port = audio2_port_locked(a0);
    if (!port) return kA2ErrInvalidParam;
    audio2_clear_slot(*port);
    return 0;
}

// sceAudioOut2PortGetState(port, State* out) -> 0.  Titles read this state before rebuilding their
// speaker mixer, so the active output, channel count, and device volume are observable ABI state.
HLE(audio2_port_get_state) {
    A2LOG("sceAudioOut2PortGetState");
    struct PortState {
        uint16_t output;
        uint8_t num_channels, pad1;
        int16_t volume;
        uint16_t reroute_counter;
        uint32_t flags, pad2;
        uint64_t reserved[6];
    } state{};
    static_assert(sizeof(PortState) == 0x40);
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        A2PortState* port = audio2_port_locked(a0);
        if (!port) return kA2ErrInvalidParam;
        state.num_channels = (uint8_t)audio2_format_channels(port->data_format);
    }
    state.output = 1;
    state.volume = 127;
    if (!audio_store_bytes(a1, &state, sizeof state)) return kA2ErrInvalidParam;
    return 0;
}
// sceAudioOut2PortSetAttributes(port, const SceAudioOut2Attribute* attrs, size_t count).
// Attribute = {u32 id, u32 reserved, const void* value, size_t valueSize} (0x18 stride).
// `reserved` is ABI padding and may be indeterminate (GTA V leaves stack-address bits there), so
// it must never be folded into the id. Live-decoded ids:
//   0 = per-grain PCM data pointer (value: a guest pointer qword to interleaved stereo f32 samples).
// Unknown ids are skipped fail-visibly under PROSPER_AUDIO2LOG. CONFIDENCE: HIGH (Evergate + GTA V).
HLE(audio2_port_set_attr) {
    A2LOG("sceAudioOut2PortSetAttributes");
    const uint64_t count = a2 <= 32 ? a2 : 32;    // defensive cap; log the clamp fail-visibly
    if (count != a2 && audio2log())
        fprintf(stderr, "[audio2] PortSetAttributes: count %llu clamped to 32\n", (unsigned long long)a2);
    std::lock_guard<std::mutex> lk(g_a2_mx);      // covers port state AND the diagnostic set below
    uint32_t port_slot = 0;
    A2PortState* port = audio2_port_locked(a0, &port_slot);
    if (!port) return kA2ErrInvalidParam;
    if (a2 == 0) return 0;                        // no attributes: a legal no-op on a live port
    for (uint64_t i = 0; i < count; i++) {
        struct { uint32_t id, reserved; uint64_t vptr, vsize; } at{};
        static_assert(sizeof(at) == 0x18);
        if (!audio_read_bytes(a1 + i * 0x18, &at, sizeof at)) break;
        if (at.id == 0 && at.vsize == 8) {
            uint64_t pcm = 0;
            if (audio_read_bytes(at.vptr, &pcm, 8)) port->pcm_ptr = pcm;
            if (getenv("PROSPER_AUDIO2_PROBE")) {
                // Keep probe history per AudioOut2 port. Object-heavy titles routinely create
                // more than 16 ports; aliasing this table made their alternating grain buffers
                // look like a state change on every SetAttributes call and flooded the log.
                static uint64_t seen_store[kA2MaxPorts] = {0};
                static uint8_t logged_changes[kA2MaxPorts] = {0};
                if (seen_store[port_slot] != pcm) {
                    seen_store[port_slot] = pcm;
                    if (logged_changes[port_slot] < 4) {
                        ++logged_changes[port_slot];
                        fprintf(stderr, "[audio2-attr] port%llu id=0x%x vptr=0x%llx vsize=%llu -> pcm=0x%llx\n",
                                (unsigned long long)(port_slot + 1), at.id,
                                (unsigned long long)at.vptr, (unsigned long long)at.vsize,
                                (unsigned long long)pcm);
                    }
                }
            }
        } else if (audio2log() || getenv("PROSPER_AUDIO2_PROBE")) {
            static std::set<uint64_t> seen;
            if (seen.insert(at.id).second) {
                uint8_t value[16]{};
                const size_t got = audio_read_bytes_partial(
                    at.vptr, value, std::min<size_t>((size_t)at.vsize, sizeof(value)));
                fprintf(stderr, "[audio2] PortSetAttributes: port%llu unhandled attr id=%u "
                        "size=%llu value:", (unsigned long long)(port_slot + 1), at.id,
                        (unsigned long long)at.vsize);
                for (size_t j = 0; j < got; ++j) fprintf(stderr, " %02x", value[j]);
                fprintf(stderr, "\n");
            }
        }
    }
    return 0;
}

// sceAudioOut2ContextAdvance(ctx) -> 0. Update the public queue clock; PCM submission lives in Push.
HLE(audio2_ctx_advance) {
    A2LOG("sceAudioOut2ContextAdvance");
    std::lock_guard<std::mutex> lk(g_a2_mx);
    A2Context* context = audio2_context_locked(a0);
    if (!context) return kA2ErrInvalidParam;
    audio2_update_queue_locked(*context, std::chrono::steady_clock::now());
    return 0;
}

// sceAudioOut2ContextPush(ctx, flag) -> 0. On hardware Push blocks while the output queue is
// full; the null backend reproduces that as one grain of wall-clock pacing per call (same
// model as RealtimeSilentSink) so CRI's server thread runs at real time, not a hot spin.
HLE(audio2_ctx_push) {
    A2LOG("sceAudioOut2ContextPush");
    uint32_t grain = 256;
    uint32_t context_slot = 0;
    // Reserve one hardware queue slot before reading the guest grain. A nonblocking push on a full
    // queue returns NOT_READY; a blocking push waits until the 48 kHz device clock drains a slot.
    for (;;) {
        std::chrono::nanoseconds wait_duration{};
        {
            std::lock_guard<std::mutex> lk(g_a2_mx);
            A2Context* context = audio2_context_locked(a0, &context_slot);
            if (!context) return kA2ErrInvalidParam;
            const auto now = std::chrono::steady_clock::now();
            audio2_update_queue_locked(*context, now);
            const bool was_empty = !context->queued;
            if (audio2_reserve_queue_slot(context->queued, context->queue_depth)) {
                if (was_empty) context->last_queue_update = now;
                grain = context->grain;
                break;
            }
            wait_duration = std::chrono::nanoseconds(
                (long long)(context->grain ? context->grain : 256) * 1000000000LL / 48000);
        }
        if (!a1) return kA2ErrNotReady;
        std::this_thread::sleep_for(wait_duration);
    }
    // Mix each active port's current grain into a stereo bed. AudioOut2 data_format encodes the
    // sample type in bits 0..6 (0 = F32, 1 = S16) and channel count in bits 8..15. Read exactly one
    // grain at that width, convert it to the float mix bed, then fold a surround MAIN bed into the
    // stereo host sink. Both sample types are public AudioOut2 formats: for example, 0x200 is F32
    // stereo and 0x201 is S16 stereo.
    static thread_local std::vector<float> bed(4096 * 2);
    static thread_local std::vector<float> tmp;
    static thread_local std::vector<int16_t> tmp_s16;
    bool have_pcm = false;
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        A2Context* c = audio2_context_locked(a0);
        if (!c) return kA2ErrInvalidParam;
        grain = (c->grain >= 64 && c->grain <= 4096) ? c->grain : 256;
        std::memset(bed.data(), 0, sizeof(float) * grain * 2);
        const bool probe = getenv("PROSPER_AUDIO2_PROBE") != nullptr;
        uint32_t object_ports_with_pcm = 0;
        float object_peak = 0.0f;
        int object_peak_port = 0;
        int pidx_probe = 0;
        for (const auto& ps : g_a2_port_state) {
            const int this_pidx = pidx_probe++;
            if (!ps.used || ps.context != a0 || !ps.pcm_ptr) continue;
            const uint32_t data_type = ps.data_format & 0x7fu;
            const uint32_t channels = audio2_format_channels(ps.data_format);
            if ((data_type != 0 && data_type != 1) || channels < 1 || channels > 8) {
                if (probe) fprintf(stderr, "[audio2-probe] port%d skipped: format=0x%x "
                                           "type=%u channels=%u unsupported\n",
                                           this_pidx + 1, ps.data_format, data_type, channels);
                continue;
            }
            const uint32_t read_samples = channels * grain;
            size_t frames_got = 0;
            if (data_type == 0) {
                if (tmp.size() < read_samples) tmp.resize(read_samples);
                const size_t got = audio_read_bytes_partial(
                    ps.pcm_ptr, tmp.data(), sizeof(float) * read_samples);
                frames_got = got / (sizeof(float) * channels);
            } else {
                if (tmp_s16.size() < read_samples) tmp_s16.resize(read_samples);
                const size_t got = audio_read_bytes_partial(
                    ps.pcm_ptr, tmp_s16.data(), sizeof(int16_t) * read_samples);
                frames_got = got / (sizeof(int16_t) * channels);
            }
            auto sample_at = [&](size_t frame, uint32_t channel) {
                const size_t sample = frame * channels + channel;
                return data_type == 0 ? tmp[sample]
                                      : (float)tmp_s16[sample] * (1.0f / 32768.0f);
            };
            if (probe) {
                static uint64_t call_ct[kA2MaxPorts] = {0};
                const size_t nf = frames_got * channels;
                size_t nan_ct = 0; float amax = 0.0f;
                for (size_t k = 0; k < nf; k++) {
                    float v = data_type == 0 ? tmp[k] : (float)tmp_s16[k] * (1.0f / 32768.0f);
                    if (v != v) nan_ct++; else { float a = v < 0 ? -v : v; if (a > amax) amax = a; } }
                // Object-heavy engines keep hundreds of valid but silent ports. Reporting every
                // silent buffer once per 64 grains both hides the active route and perturbs its
                // real-time pump. Sample only ports that actually contain a signal.
                if ((call_ct[this_pidx]++ % 64) == 0 && amax > 0.0f)
                    fprintf(stderr, "[audio2-probe] port%d type=%u fmt=%s ch=%u pcm=0x%llx "
                            "frames=%zu nan=%zu/%zu |max|=%.4g\n", this_pidx + 1, ps.type,
                            data_type == 0 ? "f32" : "s16", channels,
                            (unsigned long long)ps.pcm_ptr, frames_got, nan_ct, nf, amax);
                if ((ps.type & 0xff00u) == 0x0100u) {
                    object_ports_with_pcm++;
                    if (amax > object_peak) {
                        object_peak = amax;
                        object_peak_port = this_pidx + 1;
                    }
                }
            }
            // Only the MAIN port (type 0) drives the host speaker output. Ordinary non-main types
            // route to separate hardware devices (personal/pad speaker, chat, vibration). Object
            // ports also remain unmixed here until their attributes and speaker routing are
            // implemented, but probe mode still measures them above so discarded object audio is
            // visible rather than mistaken for decoder attenuation.
            if (ps.type != 0) {
                continue;
            }
            const float port_gain = (ps.flags & kA2PortFlag20DbHeadroom) ? kA2HeadroomGain : 1.0f;
            for (size_t fno = 0; fno < frames_got; fno++) {
                float left = sample_at(fno, 0);
                float right = channels >= 2 ? sample_at(fno, 1) : left;  // mono duplicates ch0
                constexpr float kCenter = 0.70710678f;        // -3 dB
                constexpr float kLfe = 0.5f;                  // -6 dB
                constexpr float kSurround = 0.70710678f;      // -3 dB

                // Sony's MAIN speaker beds use the conventional order FL, FR, FC, LFE, followed
                // by rear/side pairs. Keep the smaller layouts useful too: 3ch adds FC, 4ch is
                // quad, 5ch is FL/FR/FC/SL/SR, and 6ch is standard 5.1.
                if (channels == 3) {
                    left += sample_at(fno, 2) * kCenter;
                    right += sample_at(fno, 2) * kCenter;
                } else if (channels == 4) {
                    left += sample_at(fno, 2) * kSurround;
                    right += sample_at(fno, 3) * kSurround;
                } else if (channels == 5) {
                    left += sample_at(fno, 2) * kCenter + sample_at(fno, 3) * kSurround;
                    right += sample_at(fno, 2) * kCenter + sample_at(fno, 4) * kSurround;
                } else if (channels >= 6) {
                    left += sample_at(fno, 2) * kCenter + sample_at(fno, 3) * kLfe
                          + sample_at(fno, 4) * kSurround;
                    right += sample_at(fno, 2) * kCenter + sample_at(fno, 3) * kLfe
                           + sample_at(fno, 5) * kSurround;
                    if (channels == 7) {
                        left += sample_at(fno, 6) * (kSurround * 0.5f);
                        right += sample_at(fno, 6) * (kSurround * 0.5f);
                    } else if (channels == 8) {
                        left += sample_at(fno, 6) * kSurround;
                        right += sample_at(fno, 7) * kSurround;
                    }
                }
                bed[fno * 2 + 0] += left * port_gain;
                bed[fno * 2 + 1] += right * port_gain;
            }
            if (frames_got) have_pcm = true;
        }
        if (probe) {
            static uint64_t object_probe_calls = 0;
            if ((object_probe_calls++ % 64) == 0 && object_peak > 0.0f)
                fprintf(stderr, "[audio2-probe] objects: pcm_ports=%u peak_port=%d |max|=%.4g\n",
                        object_ports_with_pcm, object_peak_port, object_peak);
        }
    }
    // From this point through host output or silent pacing, serialize with Destroy for this exact
    // sink slot. Revalidate the full generation under g_a2_mx after acquiring the slot mutex.
    std::unique_lock<std::mutex> sink_lk(g_a2_sink_mx[context_slot]);
    auto pace_silently = [&]() -> uint64_t {
        std::chrono::steady_clock::time_point target;
        {
            std::lock_guard<std::mutex> lk(g_a2_mx);
            A2Context* c = audio2_context_locked(a0);
            if (!c) return kA2ErrInvalidParam;
            uint32_t g2 = c->grain ? c->grain : 256;
            auto dur = std::chrono::nanoseconds((long long)g2 * 1000000000LL / 48000);
            auto now = std::chrono::steady_clock::now();
            // (Re)sync if unset or far behind (post-stall) to avoid burst catch-up — same policy
            // as RealtimeSilentSink::output.
            if (c->next.time_since_epoch().count() == 0 || c->next < now - dur * 4) c->next = now;
            c->next += dur;
            target = c->next;
        }
        std::this_thread::sleep_until(target);
        return 0;
    };

    AudioSink* sink = audio_sink();
    if (sink && have_pcm) {
        // Forward the mixed grain to the host device; the sink paces one grain per call in real
        // time (same contract as the v1 sceAudioOutOutput path), so no extra sleep on this path.
        // A FAILED device open must fall through to the silent wall-clock pacing below instead:
        // the SDL3 sink's output() is a no-op on a null stream (no sleep), and returning early on
        // every push would hot-spin the guest's pump thread on hosts with no audio device.
        AudioPortInfo info; info.freq = 48000; info.channels = 2; info.fmt = AudioFmt::F32;
        info.grain = (int)grain;
        bool open_ok = false;
        {
            std::lock_guard<std::mutex> lk(g_a2_mx);
            A2Context* context = audio2_context_locked(a0);
            if (!context) return kA2ErrInvalidParam;
            if (!context->sink_opened) {
                context->sink_opened = true;
                context->sink_open_ok = sink->open(kA2SinkPortBase + (int)context_slot, info);
            }
            open_ok = context->sink_open_ok;
        }
        const int sink_port = kA2SinkPortBase + (int)context_slot;
        if (open_ok) {
            for (uint32_t i = 0; i < grain * 2; i++) {
                float v = bed[i];
                if (v != v) v = 0.0f;                        // NaN -> 0; Inf is caught by the clamp below
                bed[i] = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
            }
            audio_observe_output(sink_port, bed.data(), (int)grain, info);
            sink->output(sink_port, bed.data(), (int)grain);
            return 0;
        }
        return pace_silently();
    }
    // No sink / no data yet: keep the silent real-time pacing so the guest's pump thread does not
    // hot-spin (on hardware Push blocks while the output queue is full).
    return pace_silently();
}

// Targeted control-surface tracing. AudioOut2 ports feed a pre-mastering speaker bed, so these
// calls are relevant whenever valid decoded PCM reaches the bed but the audible master is wrong.
void audio2_control_probe(const char* name, uint64_t a0, uint64_t a1, uint64_t a2,
                          uint64_t a3, uint64_t a4, uint64_t a5) {
    if (!getenv("PROSPER_AUDIO2_CONTROL_PROBE")) return;
    static std::atomic<uint32_t> count{0};
    const uint32_t n = count.fetch_add(1);
    if (n >= 128) return;
    fprintf(stderr, "[audio2-control] %s(0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx)\n",
            name, (unsigned long long)a0, (unsigned long long)a1,
            (unsigned long long)a2, (unsigned long long)a3,
            (unsigned long long)a4, (unsigned long long)a5);
    const uint64_t args[6] = {a0, a1, a2, a3, a4, a5};
    for (uint32_t i = 0; i < 6; ++i) {
        if (args[i] < 0x10000 || (args[i] & 0xffff000000000000ull)) continue;
        uint8_t bytes[48]{};
        const size_t got = audio_read_bytes_partial(args[i], bytes, sizeof(bytes));
        if (!got) continue;
        fprintf(stderr, "[audio2-control]   a%u[0..%zu]:", i, got);
        for (size_t k = 0; k < got; ++k) fprintf(stderr, " %02x", bytes[k]);
        fprintf(stderr, "\n");
    }
}

HLE(audio2_ctx_set_attr) {
    audio2_control_probe("sceAudioOut2ContextSetAttributes", a0, a1, a2, a3, a4, a5);
    std::lock_guard<std::mutex> lk(g_a2_mx);
    if (!audio2_context_locked(a0)) return kA2ErrInvalidParam;
    return 0;
}
HLE(audio2_get_speaker_info) {
    static std::atomic<uint32_t> probes{0};
    if (probes.fetch_add(1) < 8)
        audio2_control_probe("sceAudioOut2GetSpeakerInfo", a0, a1, a2, a3, a4, a5);
    // The host sink exposed by this HLE is stereo. Report its real speaker mask and positions so
    // guest mixers can construct their final matrix; leaving this output untouched reports zero
    // available speakers and causes engines to attenuate/misroute an otherwise healthy mix.
    struct SpeakerPosition { int16_t azimuth, elevation; };
    struct SpeakerInfo {
        uint8_t type;
        uint8_t reserved0;
        int16_t reserved1;
        uint32_t available_bits;
        uint32_t flags;
        uint32_t reserved2;
        SpeakerPosition positions[16];
    } info{};
    static_assert(sizeof(SpeakerInfo) == 0x50);
    info.type = 0;                 // conventional speaker array
    info.available_bits = 0x3;    // front-left and front-right
    info.positions[0] = {-30, 0};
    info.positions[1] = { 30, 0};
    if (!audio_store_bytes(a0, &info, sizeof(info))) return kA2ErrInvalidParam;
    return 0;
}

bool audio2_speaker_array_valid(uint64_t handle) {
    const uint64_t index = handle & 0xff;
    return (handle & ~0xffull) == kA2SpeakerArrayTag && index >= 1 && index <= 32 &&
           g_a2_speaker_arrays[index - 1];
}

HLE(audio2_speaker_array_create) {
    // Signature: (SpeakerArrayHandle* out, const void* vbap_params, const void* ambi_params).
    // The work-memory and speaker geometry live in those parameter objects; this backend only needs
    // an opaque identity because it computes deterministic stereo coefficients below.
    if (getenv("PROSPER_AUDIO2_PROBE")) {
        fprintf(stderr, "[audio2-speaker] create out=0x%llx vbap=0x%llx ambi=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
        const uint64_t params[2] = {a1, a2};
        for (uint32_t p = 0; p < 2; ++p) {
            uint8_t bytes[64]{};
            const size_t got = audio_read_bytes_partial(params[p], bytes, sizeof(bytes));
            if (!got) continue;
            fprintf(stderr, "[audio2-speaker]   param%u[0..%zu]:", p, got);
            for (size_t i = 0; i < got; ++i) fprintf(stderr, " %02x", bytes[i]);
            fprintf(stderr, "\n");
        }
    }
    std::lock_guard<std::mutex> lk(g_a2_mx);
    for (uint64_t i = 0; i < 32; ++i) {
        if (g_a2_speaker_arrays[i]) continue;
        const uint64_t handle = kA2SpeakerArrayTag | (i + 1);
        if (!a2_store_u64(a0, handle)) return kA2ErrInvalidParam;
        g_a2_speaker_arrays[i] = true;
        return 0;
    }
    return kA2ErrPortFull;
}

HLE(audio2_speaker_array_destroy) {
    std::lock_guard<std::mutex> lk(g_a2_mx);
    if (!audio2_speaker_array_valid(a0)) return kA2ErrInvalidParam;
    g_a2_speaker_arrays[(a0 & 0xff) - 1] = false;
    return 0;
}

// The position/spread/downmix arguments use the SysV guest ABI's independent XMM argument sequence;
// handle/output/count/height therefore arrive in RDI/RSI/RDX/RCX. The current deterministic stereo
// fallback does not consume the float inputs, so use the normal integer HLE bridge. This is also
// required on Windows, whose import stubs translate the guest's integer registers but do not yet
// marshal XMM arguments into a typed Microsoft-ABI call.
HLE(audio2_get_speaker_array_coefficients) {
    const uint64_t handle = a0;
    const uint64_t coefficients = a1;
    const uint32_t count = (uint32_t)a2;
    const uint8_t height_aware = (uint8_t)a3;
    if (getenv("PROSPER_AUDIO2_PROBE")) {
        static std::atomic<uint32_t> calls{0};
        const uint32_t call = calls.fetch_add(1);
        if (call < 64)
            fprintf(stderr, "[audio2-speaker] coeff #%u handle=0x%llx out=0x%llx "
                    "count=%u height=%u\n", call + 1, (unsigned long long)handle,
                    (unsigned long long)coefficients, count, height_aware);
    }
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        if (!audio2_speaker_array_valid(handle)) return kA2ErrInvalidParam;
    }
    if ((!coefficients && count) || count > 256) return kA2ErrInvalidParam;
    std::vector<float> values(count, 0.0f);
    if (count > 0) values[0] = 1.0f;
    if (count > 1) values[1] = 1.0f;
    if (values.empty()) return 0;
    return audio_store_bytes(coefficients, values.data(), values.size() * sizeof(float))
        ? 0 : kA2ErrInvalidParam;
}

HLE(audio2_get_speaker_array_ambisonics_coefficients) {
    if (getenv("PROSPER_AUDIO2_PROBE")) {
        static std::atomic<uint32_t> calls{0};
        const uint32_t call = calls.fetch_add(1);
        if (call < 64)
            fprintf(stderr, "[audio2-speaker] ambi #%u handle=0x%llx channel=%llu "
                    "out=0x%llx count=%llu\n", call + 1, (unsigned long long)a0,
                    (unsigned long long)a1, (unsigned long long)a2,
                    (unsigned long long)a3);
    }
    std::lock_guard<std::mutex> lk(g_a2_mx);
    if (!audio2_speaker_array_valid(a0) || (!a2 && a3) || a3 > 256) return kA2ErrInvalidParam;
    std::vector<float> values((size_t)a3, 0.0f);
    if (!values.empty()) values[0] = (a1 == 0 || a1 == 64) ? 0.70710677f : 1.0f;
    if (values.empty()) return 0;
    return audio_store_bytes(a2, values.data(), values.size() * sizeof(float))
        ? 0 : kA2ErrInvalidParam;
}

HLE(audio2_mastering_init) {
    audio2_control_probe("sceAudioOut2MasteringInit", a0, a1, a2, a3, a4, a5);
    return 0;
}
HLE(audio2_mastering_term) {
    audio2_control_probe("sceAudioOut2MasteringTerm", a0, a1, a2, a3, a4, a5);
    return 0;
}
HLE(audio2_mastering_set_param) {
    audio2_control_probe("sceAudioOut2MasteringSetParam", a0, a1, a2, a3, a4, a5);
    return 0;
}
HLE(audio2_mastering_get_state) {
    audio2_control_probe("sceAudioOut2MasteringGetState", a0, a1, a2, a3, a4, a5);
    return 0;
}

// Return the device queue state after draining whole grains according to the 48 kHz hardware clock.
// Both outputs are optional in the SDK contract.
HLE(audio2_ctx_get_queue_level) {
    A2LOG("sceAudioOut2ContextGetQueueLevel");
    static std::atomic<uint32_t> control_probes{0};
    if (control_probes.fetch_add(1) < 8)
        audio2_control_probe("sceAudioOut2ContextGetQueueLevel", a0, a1, a2, a3, a4, a5);
    uint32_t queued = 0, available = 4;
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        A2Context* context = audio2_context_locked(a0);
        if (!context) return kA2ErrInvalidParam;
        audio2_update_queue_locked(*context, std::chrono::steady_clock::now());
        queued = context->queued;
        available = queued < context->queue_depth ? context->queue_depth - queued : 0;
    }
    if (a1 && !a2_store_u32(a1, queued)) return kA2ErrInvalidParam;
    if (a2 && !a2_store_u32(a2, available)) return kA2ErrInvalidParam;
    return 0;
}

HLE(audio2_get_system_state) {
    A2LOG("sceAudioOut2GetSystemState");
    static std::atomic<uint32_t> control_probes{0};
    if (control_probes.fetch_add(1) < 8)
        audio2_control_probe("sceAudioOut2GetSystemState", a0, a1, a2, a3, a4, a5);
    // { float loudness; u32 pad; u64 reserved[7]; }
    if (!a2_store_zeros(a0, 0x40)) return kA2ErrInvalidParam;
    return 0;
}

// Generic logging probes for the not-yet-exercised remainder of the surface. These functions can
// be called once per grain, so sample each export independently. Their detailed behavior remains a
// stub, but handle and ownership validation is observable and must match the implemented paths.
#define A2_PROBE_LOG(str)                                                \
    A2LOG(str);                                                          \
    static std::atomic<uint32_t> control_probes{0};                      \
    if (control_probes.fetch_add(1) < 8)                                 \
        audio2_control_probe(str, a0, a1, a2, a3, a4, a5)
HLE(audio2_ctx_bed_write) {
    A2_PROBE_LOG("sceAudioOut2ContextBedWrite");
    std::lock_guard<std::mutex> lk(g_a2_mx);
    return audio2_context_locked(a0) ? 0 : kA2ErrInvalidParam;
}
HLE(audio2_port_register) {
    A2_PROBE_LOG("sceAudioOut2PortRegister");
    std::lock_guard<std::mutex> lk(g_a2_mx);
    A2Context* context = audio2_context_locked(a0);
    A2PortState* port = audio2_port_locked(a1);
    return context && port && port->context == a0 ? 0 : kA2ErrInvalidParam;
}
HLE(audio2_port_unregister) {
    A2_PROBE_LOG("sceAudioOut2PortUnregister");
    std::lock_guard<std::mutex> lk(g_a2_mx);
    A2Context* context = audio2_context_locked(a0);
    A2PortState* port = audio2_port_locked(a1);
    return context && port && port->context == a0 ? 0 : kA2ErrInvalidParam;
}
#undef A2_PROBE_LOG

// --- libSceAjm (Audio Job Manager — compressed-audio decode: ATRAC9/MP3/AAC) --------------------
// The core owns AJM handles, builders, synchronous batch execution, guest-memory copies, and result
// sidebands. ATRAC9 is decoded by vendored LibAtrac9; MP3 is delegated through ajm_decoder.hpp to an
// optional general-purpose host codec frontend. Unsupported codecs fail truthfully in the sideband.
// CONFIDENCE: HIGH for the exercised ATRAC9 (Blasphemous 2) and MP3 (GTA V) batch-2 paths.
namespace {
    std::atomic<uint32_t> g_ajm_next{1};   // one non-zero counter for context/instance/batch handles
    // AJM returns signed 32-bit SCE errors. Preserve that ABI in the full HLE return register,
    // matching the AudioOut error constants above rather than leaving the upper half zeroed.
    constexpr uint64_t AJM_ERR_INVALID_CONTEXT   = (uint64_t)(int64_t)(int32_t)0x80930002u;
    constexpr uint64_t AJM_ERR_INVALID_INSTANCE  = (uint64_t)(int64_t)(int32_t)0x80930003u;
    constexpr uint64_t AJM_ERR_INVALID_BATCH     = (uint64_t)(int64_t)(int32_t)0x80930004u;
    constexpr uint64_t AJM_ERR_INVALID_PARAMETER = (uint64_t)(int64_t)(int32_t)0x80930005u;
    constexpr uint64_t AJM_INITIALIZE_PS5_CONFIG = 0x200000000ull;

    // The four pointer-returning builder exports serialize a batch from 8/16-byte chunks. These
    // layout values are shared by Sony's PS4-inherited AJM ABI and the PS5 3.20 export surface.
    enum : uint32_t {
        AJM_IDENT_JOB = 0,
        AJM_IDENT_INPUT_RUN = 1,
        AJM_IDENT_INPUT_CONTROL = 2,
        AJM_IDENT_CONTROL_FLAGS = 3,
        AJM_IDENT_RUN_FLAGS = 4,
        AJM_IDENT_RETURN_ADDRESS = 6,
        AJM_IDENT_INLINE = 7,
        AJM_IDENT_OUTPUT_RUN = 17,
        AJM_IDENT_OUTPUT_CONTROL = 18,
    };
    constexpr uint32_t AJM_INSTANCE_STATISTICS = 0x80000;
    constexpr uint64_t AJM_CONTROL_FLAGS_MASK = 0x000060000000e7ffull;
    constexpr uint64_t AJM_STATISTICS_FLAGS_MASK = 0x00000000c0018007ull;
    constexpr uint64_t AJM_RUN_FLAGS_MASK = 0x0000e00000001fffull;
    constexpr size_t AJM_MAX_BUILDER_BYTES = 64u * 1024u * 1024u;

    struct AjmJobChunk {
        uint32_t header;
        uint32_t size;
    };
    struct AjmFlagsChunk {
        uint32_t header;
        uint32_t flags_low;
    };
    struct AjmBufferChunk {
        uint32_t header;
        uint32_t size;
        uint64_t address;
    };
    struct AjmGuestBuffer {
        uint64_t address;
        uint64_t size;
    };
    static_assert(sizeof(AjmJobChunk) == 8);
    static_assert(sizeof(AjmFlagsChunk) == 8);
    static_assert(sizeof(AjmBufferChunk) == 16);
    static_assert(sizeof(AjmGuestBuffer) == 16);

    uint32_t ajm_chunk_header(uint32_t ident, uint32_t payload = 0) {
        return (ident & 0x3fu) | ((payload & 0xfffffu) << 6u);
    }

    template <typename T>
    void ajm_append(std::vector<uint8_t>& bytes, const T& value) {
        const size_t offset = bytes.size();
        bytes.resize(offset + sizeof(value));
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    void ajm_append_buffer(std::vector<uint8_t>& bytes, uint32_t ident,
                           uint64_t address, uint32_t size) {
        ajm_append(bytes, AjmBufferChunk{ajm_chunk_header(ident), size, address});
    }

    void ajm_append_flags(std::vector<uint8_t>& bytes, uint32_t ident, uint64_t flags) {
        ajm_append(bytes, AjmFlagsChunk{ajm_chunk_header(ident, (uint32_t)(flags >> 32)),
                                        (uint32_t)flags});
    }

    uint64_t ajm_write_job(uint64_t destination, uint32_t instance,
                           const std::vector<uint8_t>& payload) {
        if (!destination || payload.size() > UINT32_MAX ||
            payload.size() + sizeof(AjmJobChunk) > AJM_MAX_BUILDER_BYTES) return 0;
        std::vector<uint8_t> job;
        job.reserve(sizeof(AjmJobChunk) + payload.size());
        ajm_append(job, AjmJobChunk{ajm_chunk_header(AJM_IDENT_JOB, instance),
                                    (uint32_t)payload.size()});
        job.insert(job.end(), payload.begin(), payload.end());
        return audio_store_bytes(destination, job.data(), job.size())
            ? destination + job.size() : 0;
    }
}
// --- AJM decode state (batch-2.0 path; see ajm2_decode_batch below) ----------------------------
namespace {
struct AjmDecodeInst {
    uint32_t codec = UINT32_MAX;          // AjmCodecType supplied to InstanceCreate
    uint64_t flags = 0;
    std::unique_ptr<ajm::StreamDecoder> host_dec; // optional MP3/AAC frontend codec
    uint8_t  config[4] = {0};
    bool     have_config = false;
    std::unique_ptr<Atrac9Decoder> at9_dec; // persistent per instance: preserves MDCT overlap across blocks
    uint32_t skip_samples = 0;            // gapless program: priming frames to drop at program start
    uint64_t total_samples = 0;           // gapless program: trimmed frames to deliver (0 = no program)
    uint32_t skip_remaining = 0;          // frames still to drop (counts down from skip_samples)
    uint64_t gapless_delivered = 0;       // trimmed frames delivered for the CURRENT program
    uint64_t decoded_samples = 0;         // cumulative delivered sample-frames (no-program sideband)
    // PCM decoded but not yet delivered to a guest output buffer. Lives with the decoder (NOT per
    // batch): the decoder state already spans batches, so a spill must too — dropping it at a batch
    // boundary would lose audio the guest was told (via iSizeConsumed) it had received.
    std::vector<int16_t> carry;
};
struct AjmDecJob {
    uint32_t instance = 0;
    uint64_t in_addr = 0, out_addr = 0, result_addr = 0;
    uint32_t in_size = 0, out_size = 0;
};
// SCE_AJM_ERROR_INVALID_PARAMETER — the AJM error space (see the constants above); -1 is not a value
// the guest's error mapping recognizes.
constexpr int32_t kAjm2ErrDecode = (int32_t)0x80930005;
// Monotonic milliseconds for [ajm2] diagnostics: pad-script presses are scheduled in wall seconds,
// so timestamped lifecycle logs let a press at a known time be matched to its AJM traffic (#1097).
uint64_t ajm2_log_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

// Optional live codec capture. Keeping compressed input and the exact returned PCM in separate,
// per-instance files lets the same stream be decoded independently when diagnosing a codec or ABI
// mismatch. Disabled unless explicitly requested; AJM execution is serialized by g_ajm2_mx.
void ajm2_dump_decode(uint32_t instance, std::span<const uint8_t> input,
                      const int16_t* pcm, uint32_t pcm_bytes) {
    const char* base = getenv("PROSPER_AJM_DUMP");
    if (!base || !*base) return;
    char path[1024];
    auto append = [&](const char* suffix, const void* data, size_t size) {
        if (!data || !size) return;
        const int n = std::snprintf(path, sizeof(path), "%s.inst%u.%s", base, instance, suffix);
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(path)) return;
        if (FILE* f = std::fopen(path, "ab")) {
            std::fwrite(data, 1, size, f);
            std::fclose(f);
        }
    };
    append("mp3", input.data(), input.size());
    append("s16le", pcm, pcm_bytes);
}
std::mutex g_ajm2_mx;
std::map<uint32_t, AjmDecodeInst> g_ajm2_inst;             // instance id -> codec + decode state
std::map<uint64_t, std::vector<AjmDecJob>> g_ajm2_jobs;   // batchInfo -> queued decode jobs
} // namespace

// sceAjmInitialize(u64 config, u32* out_context): create a context. PS4-style callers pass zero,
// while PS5 titles also use the observed 0x200000000 configuration value.
HLE(ajm_initialize) {
    if ((a0 != 0 && a0 != AJM_INITIALIZE_PS5_CONFIG) || !a1) return AJM_ERR_INVALID_PARAMETER;
    if (!a2_store_u32(a1, g_ajm_next.fetch_add(1))) return AJM_ERR_INVALID_PARAMETER;
    return 0;
}
HLE(ajm_finalize)         { return 0; }
// sceAjmModuleRegister(u32 context, AjmCodecType codec, s64 reserved): register a codec on the context.
HLE(ajm_module_register)  {
    if (getenv("PROSPER_AUDIOLOG")) fprintf(stderr, "[ajm] ModuleRegister ctx=%llu codec=%llu\n",
                                            (unsigned long long)a0, (unsigned long long)a1);
    return a0 ? 0 : AJM_ERR_INVALID_CONTEXT;
}
HLE(ajm_module_unregister){ return 0; }
// sceAjmInstanceCreate(u32 context, codec, flags, u32* instance): a decoder instance handle.
HLE(ajm_instance_create) {
    if (getenv("PROSPER_AUDIOLOG")) fprintf(stderr, "[ajm] InstanceCreate ctx=%llu codec=%llu flags=0x%llx\n",
                                            (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
    if (!a0) return AJM_ERR_INVALID_CONTEXT;
    if (!a3) return AJM_ERR_INVALID_PARAMETER;
    const uint32_t id = g_ajm_next.fetch_add(1);
    if (!a2_store_u32(a3, id)) return AJM_ERR_INVALID_PARAMETER;
    AjmDecodeInst instance{};
    instance.codec = (uint32_t)a1;
    instance.flags = a2;
    if (ajm::DecoderBackend* backend = ajm::decoder_backend())
        instance.host_dec = backend->create((ajm::Codec)instance.codec, instance.flags);
    {
        std::lock_guard<std::mutex> lk(g_ajm2_mx);
        g_ajm2_inst.emplace(id, std::move(instance));
    }
    return 0;
}
// sceAjmInstanceDestroy(u32 context, u32 instance).
HLE(ajm_instance_destroy) {
    if (!a0) return AJM_ERR_INVALID_CONTEXT;
    if (!a1) return AJM_ERR_INVALID_INSTANCE;
    {   // Release this instance's persistent codec state. Instance ids are monotonic, so without
        // this both LibAtrac9 and host decoder state would grow for the process lifetime.
        std::lock_guard<std::mutex> lk(g_ajm2_mx);
        g_ajm2_inst.erase((uint32_t)a1);
    }
    return 0;
}
// sceAjmBatchInitialize(void* pBatchBuffer, size_t szBatchBuffer, SceAjmBatchInfo* pBatchInfo): prepare
// the caller's batch info so its job builders write into pBatchBuffer starting at offset 0. Was
// UNIMPLEMENTED (Evergate calls it first, NID MmpF1XsQiHw): the generic stub returned 0 without touching
// pBatchInfo, so the guest built its audio-decode batch on a garbage buffer/cursor, never ran a decode
// job, and never opened an AudioOut port -> total silence. GTA V confirms the full 40-byte layout:
// { void* pBuffer; size_t offset; size_t size; void* lastGoodJob; void* lastGoodJobReturnAddress; }.
// CONFIDENCE: HIGH — live Evergate construction plus GTA V's initialized 40-byte object and builders.
HLE(ajm_batch_initialize) {
    if (getenv("PROSPER_AUDIOLOG")) { static int n; if (n++ < 8)
        fprintf(stderr, "[ajm] BatchInitialize buf=0x%llx size=%llu info=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2); }
    if (!a0 || !a1 || !a2) return AJM_ERR_INVALID_PARAMETER;
    uint64_t info[5] = { a0, 0, a1, 0, 0 };
    if (!audio_store_bytes(a2, info, sizeof info)) return AJM_ERR_INVALID_PARAMETER;
    {   // Re-initializing a batch discards anything queued on it: a guest that abandons a batch and
        // rebuilds it must not have the old jobs decoded into output buffers it may since have freed.
        std::lock_guard<std::mutex> lk(g_ajm2_mx);
        g_ajm2_jobs.erase(a2);
    }
    return 0;
}
// --- AJM batch walker (diagnostic) -------------------------------------------------------------------
// prosper's AJM builders write jobs into the guest batch buffer with a known chunk format (see
// ajm_write_job). Under PROSPER_AUDIOLOG, walk the submitted batch and log each job's chunk shape
// (instance, flags, input/output buffers) — the wire-format documentation the future REAL decode
// needs. NOTE: no decode runs yet; jobs are parsed and logged only. ATRAC9 batch decode via the
// vendored LibAtrac9 is tracked in #1065.
struct AjmChunkRef { uint32_t ident; uint32_t size; uint64_t address; };

void ajm_execute_batch(uint64_t batch_addr, uint64_t batch_size) {
    const bool log = getenv("PROSPER_AUDIOLOG") != nullptr;
    if (!log) return;   // diagnostic-only until #1065 wires real decode
    if (!batch_addr || !batch_size || batch_size > AJM_MAX_BUILDER_BYTES) return;
    std::vector<uint8_t> buf(batch_size, 0);
    const size_t got = audio_read_bytes_partial(batch_addr, buf.data(), buf.size());
    if (got < 8) return;
    int logged = 0;
    size_t cur = 0;
    while (cur + 8 <= got) {
        uint32_t jhdr = 0, jsize = 0;
        std::memcpy(&jhdr, buf.data() + cur, 4);
        std::memcpy(&jsize, buf.data() + cur + 4, 4);
        if ((jhdr & 0x3fu) != AJM_IDENT_JOB) break;         // not a job header -> end of batch
        const uint32_t instance = (jhdr >> 6) & 0xfffffu;
        const size_t payload_end = cur + 8 + jsize;
        if (payload_end > got) break;
        std::vector<AjmChunkRef> chunks;
        uint64_t run_flags = 0, ctrl_flags = 0;
        size_t pc = cur + 8;
        while (pc + 4 <= payload_end) {
            uint32_t chdr = 0; std::memcpy(&chdr, buf.data() + pc, 4);
            const uint32_t cident = chdr & 0x3fu;
            if (cident == AJM_IDENT_CONTROL_FLAGS || cident == AJM_IDENT_RUN_FLAGS) {
                if (pc + 8 > payload_end) break;
                uint32_t flo = 0; std::memcpy(&flo, buf.data() + pc + 4, 4);
                const uint64_t f = ((uint64_t)((chdr >> 6) & 0xfffffu) << 32) | flo;
                if (cident == AJM_IDENT_RUN_FLAGS) run_flags = f; else ctrl_flags = f;
                pc += 8;
            } else if (cident == AJM_IDENT_INLINE) {
                if (pc + 8 > payload_end) break;   // header straddles the payload end
                uint32_t isz = 0; std::memcpy(&isz, buf.data() + pc + 4, 4);
                pc += 8 + (((size_t)isz + 7u) & ~size_t{7u});
            } else {                                        // buffer chunk (16 bytes)
                if (pc + 16 > payload_end) break;
                uint32_t csize = 0; uint64_t caddr = 0;
                std::memcpy(&csize, buf.data() + pc + 4, 4);
                std::memcpy(&caddr, buf.data() + pc + 8, 8);
                chunks.push_back({cident, csize, caddr});
                pc += 16;
            }
        }
        auto find = [&](uint32_t id) -> const AjmChunkRef* {
            for (const auto& c : chunks) if (c.ident == id) return &c; return nullptr; };
        const AjmChunkRef *in_run = find(AJM_IDENT_INPUT_RUN),  *out_run = find(AJM_IDENT_OUTPUT_RUN);
        const AjmChunkRef *in_ctl = find(AJM_IDENT_INPUT_CONTROL), *out_ctl = find(AJM_IDENT_OUTPUT_CONTROL);
        if (log && logged++ < 32) {
            fprintf(stderr, "[ajm] JOB inst=%u run_flags=0x%llx ctrl_flags=0x%llx chunks=%zu"
                    " in_run=%s out_run=%s in_ctl=%s out_ctl=%s\n", instance,
                    (unsigned long long)run_flags, (unsigned long long)ctrl_flags, chunks.size(),
                    in_run?"y":"-", out_run?"y":"-", in_ctl?"y":"-", out_ctl?"y":"-");
            if (in_ctl && in_ctl->size && in_ctl->size <= 64) {
                uint8_t cb[64] = {}; audio_read_bytes_partial(in_ctl->address, cb, in_ctl->size);
                fprintf(stderr, "[ajm]   in_ctl[%u]:", in_ctl->size);
                for (uint32_t i = 0; i < in_ctl->size; i++) fprintf(stderr, " %02x", cb[i]);
                fprintf(stderr, "\n");
            }
            if (in_run) fprintf(stderr, "[ajm]   in_run addr=0x%llx size=%u  out_run addr=0x%llx size=%u  out_ctl size=%u\n",
                    (unsigned long long)in_run->address, in_run->size,
                    (unsigned long long)(out_run?out_run->address:0), out_run?out_run->size:0, out_ctl?out_ctl->size:0);
        }
        (void)in_run; (void)out_run; (void)out_ctl; (void)instance;
        cur = payload_end;
    }
}

// sceAjmBatchStartBuffer(context, batch, size, prio, AjmBatchError* err, u32* out_batch_id): accept a
// decode batch and report it started (out_batch_id filled). We don't run the jobs; BatchWait completes
// it. The AjmBatchError out (a4) is left as the caller's value (its layout isn't needed for no-error).
HLE(ajm_batch_start) {
    if (getenv("PROSPER_AUDIOLOG")) fprintf(stderr, "[ajm] BatchStart ctx=%llu batch=0x%llx size=%llu prio=%llu\n",
            (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)a3);
    if (!a0) return AJM_ERR_INVALID_CONTEXT;
    if (!a5) return AJM_ERR_INVALID_PARAMETER;
    ajm_execute_batch(a1, a2);   // diagnostic walk/log of the submitted jobs (no decode yet: #1065)
    if (!a2_store_u32(a5, g_ajm_next.fetch_add(1))) return AJM_ERR_INVALID_PARAMETER;
    return 0;
}
HLE(ajm_batch_wait)       {
    if (getenv("PROSPER_AUDIOLOG")) { static int n; if (n++ < 40)
        fprintf(stderr, "[ajm] BatchWait ctx=%llu batch=%llu\n", (unsigned long long)a0, (unsigned long long)a1); }
    return a0 ? 0 : AJM_ERR_INVALID_CONTEXT; }   // batch completed
HLE(ajm_batch_cancel) {
    if (!a0) return AJM_ERR_INVALID_CONTEXT;
    if (!a1) return AJM_ERR_INVALID_BATCH;
    return 0;
}
HLE(ajm_batch_errordump)  { return 0; }

// Build one control job and return the next free byte in the caller's batch buffer. Returning zero
// here is not a harmless stub result: the guest chains the returned cursor into its next builder.
HLE8(ajm_batch_job_control_buffer_ra) {
    if (getenv("PROSPER_AUDIOLOG")) { static int n; if (n++ < 40)
        fprintf(stderr, "[ajm] JobControlRa batch=0x%llx inst=%llu flags=0x%llx in=0x%llx/%llu out=0x%llx/%llu ra=0x%llx\n",
            (unsigned long long)a0,(unsigned long long)a1,(unsigned long long)a2,(unsigned long long)a3,
            (unsigned long long)a4,(unsigned long long)a5,(unsigned long long)a6,(unsigned long long)a7); }
    if (a4 > UINT32_MAX || a6 > UINT32_MAX) return 0;
    std::vector<uint8_t> payload;
    payload.reserve(a7 ? 56 : 40);
    if (a7) ajm_append_buffer(payload, AJM_IDENT_RETURN_ADDRESS, a7, 0);
    ajm_append_buffer(payload, AJM_IDENT_INPUT_CONTROL, a3, (uint32_t)a4);
    const uint64_t mask = (uint32_t)a1 == AJM_INSTANCE_STATISTICS
        ? AJM_STATISTICS_FLAGS_MASK : AJM_CONTROL_FLAGS_MASK;
    ajm_append_flags(payload, AJM_IDENT_CONTROL_FLAGS, a2 & mask);
    ajm_append_buffer(payload, AJM_IDENT_OUTPUT_CONTROL, a5, (uint32_t)a6);
    return ajm_write_job(a0, (uint32_t)a1, payload);
}

// Store caller data inline after an AJM inline header. The reported batch address points at the
// copied payload; the next cursor is rounded up to AJM's required 8-byte boundary.
HLE(ajm_batch_job_inline_buffer) {
    if (!a0 || !a3 || a2 > UINT32_MAX || a2 > AJM_MAX_BUILDER_BYTES - sizeof(AjmJobChunk))
        return 0;
    const size_t data_size = (size_t)a2;
    const size_t aligned_size = (data_size + 7u) & ~size_t{7u};
    std::vector<uint8_t> bytes(sizeof(AjmJobChunk) + aligned_size, 0);
    const AjmJobChunk header{ajm_chunk_header(AJM_IDENT_INLINE), (uint32_t)aligned_size};
    std::memcpy(bytes.data(), &header, sizeof(header));
    if (data_size && !audio_read_bytes(a1, bytes.data() + sizeof(header), data_size)) return 0;
    if (!audio_store_bytes(a0, bytes.data(), bytes.size()) ||
        !a2_store_u64(a3, a0 + sizeof(AjmJobChunk))) return 0;
    return a0 + bytes.size();
}

HLE10(ajm_batch_job_run_buffer_ra) {
    if (getenv("PROSPER_AUDIOLOG")) { static int n; if (n++ < 40)
        fprintf(stderr, "[ajm] JobRunRa batch=0x%llx inst=%llu flags=0x%llx in=0x%llx/%llu out=0x%llx/%llu sb=0x%llx/%llu ra=0x%llx\n",
            (unsigned long long)a0,(unsigned long long)a1,(unsigned long long)a2,(unsigned long long)a3,(unsigned long long)a4,
            (unsigned long long)a5,(unsigned long long)a6,(unsigned long long)a7,(unsigned long long)a8,(unsigned long long)a9); }
    if (a4 > UINT32_MAX || a6 > UINT32_MAX || a8 > UINT32_MAX) return 0;
    std::vector<uint8_t> payload;
    payload.reserve(a9 ? 72 : 56);
    if (a9) ajm_append_buffer(payload, AJM_IDENT_RETURN_ADDRESS, a9, 0);
    ajm_append_buffer(payload, AJM_IDENT_INPUT_RUN, a3, (uint32_t)a4);
    ajm_append_flags(payload, AJM_IDENT_RUN_FLAGS, a2 & AJM_RUN_FLAGS_MASK);
    ajm_append_buffer(payload, AJM_IDENT_OUTPUT_RUN, a5, (uint32_t)a6);
    ajm_append_buffer(payload, AJM_IDENT_OUTPUT_CONTROL, a7, (uint32_t)a8);
    return ajm_write_job(a0, (uint32_t)a1, payload);
}

HLE10(ajm_batch_job_run_split_buffer_ra) {
    if (getenv("PROSPER_AUDIOLOG")) { static int n; if (n++ < 40)
        fprintf(stderr, "[ajm] JobRunSplitRa batch=0x%llx inst=%llu\n", (unsigned long long)a0,(unsigned long long)a1); }
    if (a4 > UINT32_MAX || a6 > UINT32_MAX || a8 > UINT32_MAX) return 0;
    const uint64_t chunk_count = a4 + a6 + 2u + (a9 ? 1u : 0u);
    if (chunk_count > (AJM_MAX_BUILDER_BYTES - sizeof(AjmJobChunk)) /
                      sizeof(AjmBufferChunk)) return 0;
    if ((a4 && !a3) || (a6 && !a5)) return 0;

    std::vector<uint8_t> payload;
    payload.reserve((size_t)chunk_count * sizeof(AjmBufferChunk));
    if (a9) ajm_append_buffer(payload, AJM_IDENT_RETURN_ADDRESS, a9, 0);
    for (uint64_t i = 0; i < a4; ++i) {
        AjmGuestBuffer buffer{};
        if (!audio_read_bytes(a3 + i * sizeof(buffer), &buffer, sizeof(buffer)) ||
            buffer.size > UINT32_MAX) return 0;
        ajm_append_buffer(payload, AJM_IDENT_INPUT_RUN, buffer.address, (uint32_t)buffer.size);
    }
    ajm_append_flags(payload, AJM_IDENT_RUN_FLAGS, a2 & AJM_RUN_FLAGS_MASK);
    for (uint64_t i = 0; i < a6; ++i) {
        AjmGuestBuffer buffer{};
        if (!audio_read_bytes(a5 + i * sizeof(buffer), &buffer, sizeof(buffer)) ||
            buffer.size > UINT32_MAX) return 0;
        ajm_append_buffer(payload, AJM_IDENT_OUTPUT_RUN, buffer.address, (uint32_t)buffer.size);
    }
    ajm_append_buffer(payload, AJM_IDENT_OUTPUT_CONTROL, a7, (uint32_t)a8);
    return ajm_write_job(a0, (uint32_t)a1, payload);
}

// --- AJM batch-2.0 compressed-audio decode (Blasphemous 2 / GTA V) -----------------------------
// B2's FMOD decodes ATRAC9 and GTA V decodes MP3 through the newer BatchJob* builders (distinct from
// the RunBufferRa model above). ABI recovered from live PROSPER_AUDIOLOG captures:
//   sceAjmBatchJobInitialize(batchInfo, instance, cfgPtr, cfgSize=8, resultPtr, ...)
//        cfgPtr[0..3] = the 4-byte ATRAC9 ConfigData (starts 0xFE; e.g. FE 72 09 F0 = 48k stereo).
//   sceAjmBatchJobSetGaplessDecode(batchInfo, instance, {u32 total, u32 skip}, 1, resultPtr, ...)
//   sceAjmBatchJobDecode(batchInfo, instance, in, inSize, out, outSize, resultPtr, retAddr, 0, resultPtr)
//        decode inSize compressed bytes at `in` into interleaved S16 PCM at `out` (<= outSize).
//   sceAjmBatchStart(context, batchInfo, priority, ...) -> run all jobs queued on batchInfo.
// prosper accepts the batches, decodes each job through vendored LibAtrac9 or an optional host codec,
// writes PCM into the guest output buffer, and fills the result sideband. The guest then mixes it into
// its AudioOut2 MAIN port. CONFIDENCE: HIGH — live B2 and GTA V evidence plus focused codec tests.
namespace {

// Write the AJM decode result sideband (published contract), into the 32-byte decode buffer:
//   SceAjmSidebandResult { s32 iResult; s32 iCodecResult; }          (0 / 0 = success)
//   SceAjmSidebandStream { u32 iSizeConsumed; u32 iSizeProduced;     (input bytes used / PCM bytes out)
//                          u64 uiTotalDecodedSamples; }
//   SceAjmSidebandMFrame { u32 numFrames; u32 reserved; }            (codec frames decoded by this job)
// uiTotalDecodedSamples is load-bearing, not padding: with it left zero the guest's mixer (FMOD) stops
// after a single batch, so it carries the instance's running sample-frame total.
bool ajm2_write_result(uint64_t result_addr, int32_t err, uint32_t consumed, uint32_t produced,
                       uint64_t total_samples = 0, uint32_t decoded_frames = 0) {
    if (!result_addr) return true;
    struct Sideband { int32_t iResult; int32_t iCodecResult; uint32_t iSizeConsumed;
                      uint32_t iSizeProduced; uint64_t uiTotalDecodedSamples;
                      uint32_t numFrames; uint32_t reserved; };
    static_assert(sizeof(Sideband) == 32, "AJM decode sideband must include the MFrame result");
    Sideband sb{ err, 0, consumed, produced, total_samples, decoded_frames, 0 };
    return audio_store_bytes(result_addr, &sb, sizeof sb);
}

// Decode a batch's queued jobs. Jobs sharing an instance are consecutive stream blocks whose input
// buffers are contiguous in guest memory, so we treat each instance's jobs as ONE continuous ATRAC9
// stream: decode whole superframes in order (frames within a superframe aren't independently
// decodable, so partial-superframe decoding would desync the decoder), and route the decoded PCM
// into the jobs' output buffers in sequence, filling each completely. Decoding is bounded by the
// TOTAL output capacity across the instance's jobs; the compressed bytes actually consumed are
// reported per job via iSizeConsumed, so the guest re-submits whatever it provided beyond capacity.
void ajm2_decode_batch(std::vector<AjmDecJob>& jobs) {
    for (size_t ji = 0; ji < jobs.size();) {
        const uint32_t inst_id = jobs[ji].instance;
        size_t je = ji; while (je < jobs.size() && jobs[je].instance == inst_id) ++je;  // [ji,je) same instance
        auto it = g_ajm2_inst.find(inst_id);
        // InstanceCreate normally constructs optional codecs. Also retry lazily in case a test or
        // embedding frontend installed its backend after creating the guest instance.
        if (it != g_ajm2_inst.end() && !it->second.host_dec &&
            it->second.codec != UINT32_MAX) {
            if (ajm::DecoderBackend* backend = ajm::decoder_backend())
                it->second.host_dec = backend->create((ajm::Codec)it->second.codec, it->second.flags);
        }
        if (it != g_ajm2_inst.end() && it->second.host_dec && it->second.host_dec->valid()) {
            AjmDecodeInst& instance = it->second;
            const bool log = getenv("PROSPER_AUDIOLOG") != nullptr;
            for (size_t k = ji; k < je; ++k) {
                AjmDecJob& job = jobs[k];
                if (!instance.host_dec->valid()) {
                    ajm2_write_result(job.result_addr, kAjm2ErrDecode, 0, 0,
                                      instance.decoded_samples);
                    continue;
                }
                std::vector<uint8_t> input(job.in_size);
                const size_t got = audio_read_bytes_partial(job.in_addr, input.data(), input.size());
                std::vector<int16_t> pcm(job.out_size / sizeof(int16_t));
                const ajm::DecodeResult decoded = instance.host_dec->decode(
                    std::span<const uint8_t>(input.data(), got), std::span<int16_t>(pcm));
                const uint32_t frame_bytes = decoded.channels * sizeof(int16_t);
                // A streaming parser may consume a compressed prefix before it has one complete
                // frame and therefore before it can report the stream's channel count. That is a
                // successful zero-output job: requiring frame_bytes here would report an error
                // after advancing persistent parser state, prompting the guest to resend bytes we
                // already consumed. Channel alignment becomes meaningful only once PCM is emitted.
                const bool pcm_shape_valid = decoded.produced_bytes == 0 ||
                    (frame_bytes != 0 && decoded.produced_bytes % frame_bytes == 0);
                const bool valid = decoded.ok && decoded.consumed_bytes <= got &&
                    decoded.produced_bytes <= job.out_size &&
                    decoded.produced_bytes <= pcm.size() * sizeof(int16_t) &&
                    pcm_shape_valid;
                // A failed result publishes zero consumption. The backend may already have
                // advanced parser/codec state, so it must become terminal before the guest retries
                // the same compressed prefix. This also covers a malformed backend result rejected
                // by the guest-facing validation above.
                if (!valid) instance.host_dec->invalidate();
                int32_t err = valid ? 0 : kAjm2ErrDecode;
                uint32_t consumed = valid ? decoded.consumed_bytes : 0;
                uint32_t produced = valid ? decoded.produced_bytes : 0;
                if (produced && !audio_store_bytes(job.out_addr, pcm.data(), produced)) {
                    instance.host_dec->invalidate();
                    err = kAjm2ErrDecode;
                    consumed = produced = 0;
                }
                if (!err) ajm2_dump_decode(inst_id,
                    std::span<const uint8_t>(input.data(), consumed), pcm.data(), produced);
                if (!err && produced) instance.decoded_samples += produced / frame_bytes;
                const bool result_published = ajm2_write_result(
                    job.result_addr, err, consumed, produced, instance.decoded_samples,
                    err ? 0 : decoded.decoded_frames);
                if (!result_published) {
                    // The codec and guest PCM may already have advanced, but the guest did not
                    // receive the progress sideband. Preserve the same terminal invariant as every
                    // other unpublishable result before another queued job reaches the decoder.
                    instance.host_dec->invalidate();
                }
                if (log)
                    fprintf(stderr, "[ajm2] t=%llums decode inst=%u codec=%u in=%zu/%u "
                            "out=%u -> %u consumed, %u PCM, %uch/%uHz total=%llu%s\n",
                            (unsigned long long)ajm2_log_ms(), inst_id, instance.codec,
                            got, job.in_size, job.out_size, consumed, produced, decoded.channels,
                            decoded.sample_rate, (unsigned long long)instance.decoded_samples,
                            !result_published ? " RESULT_STORE_ERR" : (err ? " ERR" : ""));
            }
            ji = je;
            continue;
        }
        // A non-null invalid host decoder is deliberately terminal. Do not fall through to the
        // ATRAC9 path or erase the cumulative sample count on a later batch's error sideband.
        if (it != g_ajm2_inst.end() && it->second.host_dec) {
            for (size_t k = ji; k < je; ++k)
                ajm2_write_result(jobs[k].result_addr, kAjm2ErrDecode, 0, 0,
                                  it->second.decoded_samples);
            ji = je;
            continue;
        }
        Atrac9Decoder* dec = (it != g_ajm2_inst.end() && it->second.at9_dec && it->second.at9_dec->valid())
                             ? it->second.at9_dec.get() : nullptr;
        if (!dec) {
            const uint64_t total = it != g_ajm2_inst.end()
                ? (it->second.total_samples ? it->second.gapless_delivered
                                            : it->second.decoded_samples)
                : 0;
            for (size_t k = ji; k < je; ++k)
                ajm2_write_result(jobs[k].result_addr, kAjm2ErrDecode, 0, 0, total);
            ji = je;
            continue;
        }
        const int ch = dec->channels();
        const int sfb = dec->superframe_bytes();
        const int sfs = dec->superframe_samples();
        if (ch <= 0 || sfb <= 0 || sfs <= 0) {
            for (size_t k = ji; k < je; ++k) ajm2_write_result(jobs[k].result_addr, kAjm2ErrDecode, 0, 0); ji = je; continue; }
        const uint32_t frame_bytes = (uint32_t)ch * sizeof(int16_t);   // one interleaved sample-frame
        const uint32_t sf_out_bytes = (uint32_t)sfs * frame_bytes;
        std::vector<int16_t> pcm((size_t)sfs * ch);
        std::vector<int16_t>& carry = it->second.carry;   // spans batches: the decoder state does too
        std::vector<uint8_t> in;
        const bool log = getenv("PROSPER_AUDIOLOG") != nullptr;

        for (size_t k = ji; k < je; ++k) {
            AjmDecJob& job = jobs[k];
            AjmDecodeInst& I = it->second;
            // A prior job may have advanced ATRAC9 state and then failed to publish its result.
            // Such an instance is terminal: later jobs must report zero progress instead of
            // decoding against state the guest could not observe.
            if (!dec) {
                ajm2_write_result(job.result_addr, kAjm2ErrDecode, 0, 0,
                                  I.total_samples ? I.gapless_delivered : I.decoded_samples);
                continue;
            }
            // A programmed gapless decode (#1097): drop `skip` priming frames at the program start,
            // deliver EXACTLY `total` trimmed frames, then pin the reported total there with no
            // further PCM. FMOD's channel-end and codec-slot recycling key off that exact landing;
            // overshooting it (whole raw superframes, cumulative counter) left every one-shot SFX
            // "still playing" forever, so codec slots were never recycled and the pool exhausted
            // after ~32 sounds -- every later sound silent.
            const bool prog = I.total_samples > 0;
            // Over-allocate by a full superframe: LibAtrac9's bit reader is unbounded, so a truncated
            // or corrupt block's parse may step past the nominal superframe. decode_superframe still
            // rejects such a parse, but the read must land in memory we own.
            in.assign((size_t)job.in_size + (size_t)sfb, 0);
            const size_t got = audio_read_bytes_partial(job.in_addr, in.data(), job.in_size);
            // Only ever hand the guest whole sample-frames: a byte-granular partial write would shift
            // the interleave and swap L/R for the rest of the stream.
            const uint32_t out_cap = job.out_size - (job.out_size % frame_bytes);
            uint32_t in_cur = 0, produced = 0, decoded_codec_frames = 0;
            int32_t err = 0;
            // Deliver `nframes` at `src`: end-trim against the program total, then write what the
            // output has room for. Returns the number of frames written; `spill` gets the within-
            // total remainder (room starvation), while beyond-total frames are discarded.
            auto deliver = [&](const int16_t* src, uint32_t nframes,
                               const int16_t** spill, uint32_t* spill_frames) -> uint32_t {
                *spill = nullptr; *spill_frames = 0;
                if (prog) {
                    const uint64_t left = I.total_samples > I.gapless_delivered
                                              ? I.total_samples - I.gapless_delivered : 0;
                    if (nframes > left) nframes = (uint32_t)left;
                }
                const uint32_t room_frames = (out_cap - produced) / frame_bytes;
                const uint32_t w = std::min(nframes, room_frames);
                if (w) {
                    if (!audio_store_bytes(job.out_addr + produced, src,
                                           w * frame_bytes)) { err = kAjm2ErrDecode; return 0; }
                    produced += w * frame_bytes;
                    if (prog) I.gapless_delivered += w;   // only meaningful for a gapless program
                }
                *spill = src + (size_t)w * ch;
                *spill_frames = nframes - w;
                return w;
            };
            // 1. Drain carry-over PCM (decoded earlier, not yet delivered) into this output first.
            //    Carry holds post-skip frames, so only the total/room trims apply here.
            if (!carry.empty() && produced < out_cap) {
                const uint32_t cframes = (uint32_t)(carry.size() / ch);
                const int16_t* spill = nullptr; uint32_t spill_frames = 0;
                deliver(carry.data(), cframes, &spill, &spill_frames);
                if (!err) {
                    if (spill_frames)
                        carry.erase(carry.begin(),
                                    carry.end() - (size_t)spill_frames * ch);
                    else
                        carry.clear();
                }
            }
            // 2. Decode this block's superframes into the remaining space. Stop BEFORE decoding once
            //    the output is full (so `iSizeConsumed` never covers PCM the guest did not receive)
            //    or the gapless program is complete (post-EOS input produces nothing).
            while (!err && produced < out_cap && in_cur + (uint32_t)sfb <= got &&
                   !(prog && I.gapless_delivered >= I.total_samples)) {
                if (dec->decode_superframe(in.data() + in_cur, pcm.data(),
                                           (int)(got - in_cur)) < 0) { err = kAjm2ErrDecode; break; }
                in_cur += (uint32_t)sfb;
                ++decoded_codec_frames;
                const int16_t* block = pcm.data();
                uint32_t nframes = (uint32_t)sfs;
                if (prog && I.skip_remaining) {     // priming skip: stream-order, decode-time drop
                    const uint32_t drop = std::min(I.skip_remaining, nframes);
                    I.skip_remaining -= drop;
                    block += (size_t)drop * ch;
                    nframes -= drop;
                }
                const int16_t* spill = nullptr; uint32_t spill_frames = 0;
                deliver(block, nframes, &spill, &spill_frames);
                if (err) break;
                if (spill_frames)                                       // spilled: carry forward
                    carry.insert(carry.end(), spill, spill + (size_t)spill_frames * ch);
            }
            I.decoded_samples += produced / frame_bytes;
            const bool result_published = ajm2_write_result(
                job.result_addr, err, in_cur, produced,
                prog ? I.gapless_delivered : I.decoded_samples, decoded_codec_frames);
            if (!result_published) {
                // Decoder/trim/carry state and guest PCM may already have advanced, but the guest
                // did not receive the progress sideband. Keep cumulative totals for diagnostics
                // and terminal error sidebands, while preventing any retry from advancing further.
                I.at9_dec.reset();
                I.carry.clear();
                dec = nullptr;
            }
            if (log)
                fprintf(stderr, "[ajm2] t=%llums decode inst=%u in=%zu/%u out=%u -> %u consumed, %u PCM, total=%llu carry=%zu%s\n",
                        (unsigned long long)ajm2_log_ms(), inst_id, got, job.in_size, job.out_size,
                        in_cur, produced, (unsigned long long)it->second.decoded_samples,
                        carry.size() * sizeof(int16_t),
                        !result_published ? " RESULT_STORE_ERR" : (err ? " ERR" : ""));
        }
        ji = je;
    }
}
} // namespace

HLE10(ajm_batch_job_initialize) {
    // Read the 4-byte ATRAC9 config and (re)create this instance's decoder on a config change. Same
    // config -> keep the existing decoder so streaming MDCT overlap is preserved across blocks/batches.
    uint8_t cfg[4] = {0};
    const bool have = a2 && audio_read_bytes(a2, cfg, 4) && cfg[0] == 0xFE;
    std::lock_guard<std::mutex> lk(g_ajm2_mx);
    AjmDecodeInst& inst = g_ajm2_inst[(uint32_t)a1];
    const bool config_changed =
        have && (!inst.have_config || std::memcmp(inst.config, cfg, 4) != 0);
    if (getenv("PROSPER_AUDIOLOG")) {   // #1097: every re-init, with the state it inherits
        static std::atomic<uint64_t> n{0};
        const uint64_t k = n.fetch_add(1);
        if (k < 400 || (k % 100) == 0)
            fprintf(stderr, "[ajm2] t=%llums JobInitialize inst=%llu config=%02x%02x%02x%02x "
                    "changed=%d inherited_decoded=%llu\n",
                    (unsigned long long)ajm2_log_ms(), (unsigned long long)a1,
                    cfg[0], cfg[1], cfg[2], cfg[3], config_changed ? 1 : 0,
                    (unsigned long long)inst.decoded_samples);
    }
    if (config_changed) {
        std::memcpy(inst.config, cfg, 4);
        inst.have_config = true;
        inst.codec = (uint32_t)ajm::Codec::Atrac9;
        inst.at9_dec = std::make_unique<Atrac9Decoder>();
        if (!inst.at9_dec->init(cfg)) inst.at9_dec.reset();
        // A different stream format on a reused slot: drop the old stream's pending PCM and
        // program state; the guest programs a fresh gapless spec for the new sound (#1097).
        inst.carry.clear();
        inst.skip_remaining = 0;
        inst.gapless_delivered = 0;
        inst.total_samples = 0;
        if (getenv("PROSPER_AUDIOLOG"))
            fprintf(stderr, "[ajm2] JobInitialize inst=%llu config=%02x%02x%02x%02x decoder=%s\n",
                    (unsigned long long)a1, cfg[0], cfg[1], cfg[2], cfg[3],
                    inst.at9_dec ? "ok" : "init-failed");
    }
    return 0;
}
HLE10(ajm_batch_job_gapless) {
    // gaplessPtr = { u32 totalSamples, u32 skipSamples }. Recorded for future front/end trim; not yet
    // applied (skip is ~256 encoder-priming samples, inaudible for playback bring-up).
    uint32_t g[2] = {0, 0};
    if (a2) audio_read_bytes(a2, g, sizeof g);
    std::lock_guard<std::mutex> lk(g_ajm2_mx);
    AjmDecodeInst& inst = g_ajm2_inst[(uint32_t)a1];
    inst.total_samples = g[0];
    inst.skip_samples = g[1];
    // A gapless spec (re)starts the decode PROGRAM (#1097): the skip applies from here, the
    // delivered counter restarts (FMOD re-arms at a loop boundary and expects the next pass to
    // count 0..total again), and pending spill PCM belongs to the previous program -- a reused
    // codec slot must never emit the prior sound's tail into the new one.
    inst.skip_remaining = g[1];
    inst.gapless_delivered = 0;
    inst.carry.clear();
    if (getenv("PROSPER_AUDIOLOG")) {   // #1097 lifecycle evidence
        static std::atomic<uint64_t> n{0};
        const uint64_t k = n.fetch_add(1);
        if (k < 400 || (k % 100) == 0)
            fprintf(stderr, "[ajm2] t=%llums SetGapless inst=%llu total=%u skip=%u\n",
                    (unsigned long long)ajm2_log_ms(), (unsigned long long)a1, g[0], g[1]);
    }
    return 0;
}
HLE10(ajm_batch_job_decode) {
    // Queue a decode op on this batch; executed in order at BatchStart. a2/a3 = in/inSize,
    // a4/a5 = out/outSize, a6 = result sideband.
    if (getenv("PROSPER_AUDIOLOG")) { static std::atomic<uint32_t> n{0}; if (n.fetch_add(1) < 16)
        fprintf(stderr, "[ajm2] JobDecode batch=0x%llx inst=%llu in=0x%llx/%llu "
                        "out=0x%llx/%llu result=0x%llx ra=0x%llx tail=0x%llx,0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1,
                (unsigned long long)a2, (unsigned long long)a3,
                (unsigned long long)a4, (unsigned long long)a5,
                (unsigned long long)a6, (unsigned long long)a7,
                (unsigned long long)a8, (unsigned long long)a9); }
    if (!a0 || !a2 || !a4) return AJM_ERR_INVALID_PARAMETER;
    // Bound the guest-supplied sizes with the same cap the older AJM builders use. Unbounded values
    // would request a multi-gigabyte zero-filled staging buffer per job (a bad_alloc thrown through a
    // guest frame), and are never legitimate for a decode block.
    if (a3 > AJM_MAX_BUILDER_BYTES || a5 > AJM_MAX_BUILDER_BYTES) return AJM_ERR_INVALID_PARAMETER;
    std::lock_guard<std::mutex> lk(g_ajm2_mx);
    g_ajm2_jobs[a0].push_back({(uint32_t)a1, a2, a4, a6, (uint32_t)a3, (uint32_t)a5});
    return 0;
}
HLE10(ajm_batch_start2) {
    // Run every decode job queued on this batchInfo (a1), in order, then clear it. Synchronous:
    // BatchWait then just returns success. GTA V passes its u32 batch-id output in a4 and immediately
    // waits on it; leaving the initialized sentinel (0xffffffff) there breaks the lifecycle even when
    // decode itself succeeded. a3 is the optional batch-error object, unused on a successful start.
    if (getenv("PROSPER_AUDIOLOG")) { static std::atomic<uint32_t> n{0}; if (n.fetch_add(1) < 16)
        fprintf(stderr, "[ajm2] BatchStart ctx=%llu batch=0x%llx priority=%llu "
                        "a3=0x%llx a4=0x%llx a5=0x%llx a6=0x%llx a7=0x%llx a8=0x%llx a9=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5,
                (unsigned long long)a6, (unsigned long long)a7, (unsigned long long)a8,
                (unsigned long long)a9); }
    std::lock_guard<std::mutex> lk(g_ajm2_mx);
    // The batch id is the caller-visible proof that BatchStart accepted this submission. Validate
    // and publish it before any persistent decoder, PCM output, or result sideband advances. On an
    // inaccessible non-null pointer, leave the queued jobs intact so the guest can retry safely.
    if (a4 && !a2_store_u32(a4, g_ajm_next.fetch_add(1))) return AJM_ERR_INVALID_PARAMETER;
    auto it = g_ajm2_jobs.find(a1);
    if (it != g_ajm2_jobs.end()) {
        ajm2_decode_batch(it->second);
        g_ajm2_jobs.erase(it);      // erase, not clear: the map is keyed by a guest pointer
    }
    return 0;
}

// --- libSceNgs2 silent lifecycle ---------------------------------------------------------------
// Dead Cells is the first title to exercise NGS2. PROSPER_NGS2_TRACE preserves and logs all six
// guest arguments for this PS5 surface; normal execution uses the same handlers without logging.
namespace {

int ngs2_trace_level() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PROSPER_NGS2_TRACE"); v = (e && *e) ? atoi(e) : 0; if (v < 0) v = 0; }
    return v;
}
bool ngs2_trace_enabled() { return ngs2_trace_level() >= 1; }

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
// NGS2 produces num_grain_samples frames per SystemRender; the guest hands us a buffer that can be far
// larger (Dead Cells passes 37888 frames) and only consumes one grain, streaming one grain-sized block
// per render. We fill only this many frames of an oversized buffer — filling the whole capacity makes a
// streaming voice's read cursor race ~9x ahead of the ~8-block queue the guest keeps topped up, i.e. the
// rhythmic underrun. Real NGS2 grains are small (SDK default 256, max 512), so this 4096 default is a
// safe upper bound that only trims oversized buffers. sceNgs2SystemCreate reads the real value from the
// SceNgs2SystemOption (num_grain_samples @ +0x70) when the title supplies one; Dead Cells passes null.
uint32_t g_ngs2_grain = 4096;
Ngs2RackState g_ngs2_racks[32];
std::mutex g_ngs2_zero_mx;
std::vector<uint8_t> g_ngs2_zeros;

// --- NGS2 sampler PCM mixer (Dead Cells) -------------------------------------------------------
// Dead Cells decodes its audio to interleaved 16-bit PCM in-engine (Heaps/Haxe) and streams it into
// NGS2 sampler voices; NGS2 mixes those voices in SystemRender and the guest hands the result to
// sceAudioOut. The previous headless backend zero-filled SystemRender, so the game played silence.
// This mixes the real voices. Protocol observed live (PROSPER_NGS2_TRACE=2), voice param IDs:
//   0x40010000  waveform format   { u32 waveform_type@0, u32 channels@4, u32 sample_rate@8 }
//   0x40010001  waveform block    { u64 data_ptr@0, ... }   (interleaved PCM at data_ptr)
//   0x40001300  matrix/gain       (per-render float matrix; unity for v1)
// waveform_type 0x12 == signed-16 LE PCM (the only type Dead Cells uses); ATRAC9 is deferred until a
// title that needs it exists. CONFIDENCE: MED — format/data/render layouts are live-verified; block
// length and one-shot/loop semantics are played from the current pointer and re-based when the guest
// re-points (fault-safe reads bound every access), pending an ngs2.h WaveformBlock cross-check.
constexpr uint32_t kNgs2ParamWaveformFormat = 0x40010000;
constexpr uint32_t kNgs2ParamWaveformBlock  = 0x40010001;
constexpr uint32_t kNgs2ParamMatrixLevels   = 0x40001300;   // per-render output level matrix (gains)
constexpr uint32_t kNgs2ParamVoiceCallback  = 0x00000007;   // Ngs2VoiceCallbackParam (block-done handler)
constexpr uint32_t kNgs2WaveformTypePcmS16  = 0x12;
// A voice stops when it has been mixed this many consecutive renders without a fresh waveform block.
// Sampler SFX are re-fed continuously while audible (observed: a play is followed by a stream of block
// params); one that stops being fed has ended, so expiring it bounds voice accumulation (voice-pool
// handles otherwise read a persistent/reused buffer forever and pile up into clipping over time).
constexpr int kNgs2VoiceIdleRendersMax = 3;

struct Ngs2Voice {
    bool     playing      = false;
    bool     played_audio = false; // has produced any non-silent sample since the last (re)trigger
    int      idle         = 0;     // renders mixed since the last waveform-block feed
    uint64_t data_ptr  = 0;        // guest base of the current interleaved-PCM block
    uint32_t channels  = 2;        // source channel count
    uint32_t rate      = 48000;    // source sample rate (Hz)
    // Per-output-channel gain from the voice's level matrix (kNgs2ParamMatrixLevels). Sized for the
    // full PS5 surround layout (up to 7.1); prosper currently mixes only the front L/R pair (see
    // ngs2_mix_voices), so gain[2..7] are parsed and stored but not yet summed into a surround bus.
    // SURROUND-TODO: when a multichannel output bus is added, mix all channels through these gains.
    float    gain[8]   = {1,1,1,1,1,1,1,1};
    uint32_t wtype     = kNgs2WaveformTypePcmS16;
    double   cursor    = 0.0;      // fractional source-frame position within blocks.front()
    // Fed waveform blocks awaiting playback, in feed order (front = currently playing). A streaming
    // guest feeds a fixed-size block per grain and reuses a small ring of non-contiguous pool buffers,
    // so the mixer must play them one block at a time and never read past a block boundary (doing so
    // samples the next, unrelated pool buffer -> the garbled "digital noise").
    std::deque<uint64_t> blocks;
    // Waveform-block completion callback (voice param 0x0007, Ngs2VoiceCallbackParam). A streaming guest
    // registers it and expects NGS2 to invoke it as each fed block finishes, so it can queue the next.
    uint64_t cb_fn = 0, cb_data = 0;   // guest handler function pointer + user data
    uint32_t cb_flags = 0;
    uint64_t played_samples = 0;       // wall-clock-paced total source frames played; reported as
                                       // VoiceGetState's num_decoded_samples so the guest's decode/feed
                                       // loop paces to real time instead of to our (faster) render rate.
    uint64_t t0_ns          = 0;       // steady-clock ns at stream start (0 = not started / resync).
};
std::map<uint32_t, Ngs2Voice> g_ngs2_voices;   // key: voice handle & 0xffffff (rack_slot<<8 | voice_id)

bool ngs2_mix_disabled() {
    static const bool off = [] { const char* e = getenv("PROSPER_NGS2_SILENT"); return e && *e && *e != '0'; }();
    return off;
}

// Parse a voice-param chain and fold format/data updates into the voice state. `handle` is the full
// guest voice handle; the low 24 bits key g_ngs2_voices. Runs under g_ngs2_mx.
void ngs2_apply_voice_params(uint64_t handle, uint64_t list) {
    if ((handle & kNgs2VoiceMask) != kNgs2VoiceTag || !list) return;
    Ngs2Voice& v = g_ngs2_voices[(uint32_t)(handle & 0xffffff)];
    uint64_t p = list;
    for (int i = 0; i < 32 && p; ++i) {
        struct { uint16_t size; int16_t next; uint32_t id; } head{};
        if (!audio_read_bytes(p, &head, sizeof(head)) || head.size < sizeof(head) || head.size > 0x400) break;
        if (head.id == kNgs2ParamWaveformFormat) {
            uint32_t fmt[3] = {};   // { waveform_type, channels, sample_rate }
            if (audio_read_bytes(p + 8, fmt, sizeof(fmt))) {
                // A waveform-format param starts a NEW sound on this voice: the guest reuses a small pool
                // of voices, reformatting one each time it (re)triggers a click/SFX, and streams a fresh
                // track's blocks after re-formatting the music voice. Reset the stream state so the new
                // sound plays from the start instead of queueing behind the previous sound's leftover
                // blocks and stale clock (that pile-up is the "buffer mess" on the second click).
                v.blocks.clear();
                v.cursor = 0.0;
                v.played_samples = 0;
                v.t0_ns = 0;
                v.wtype = fmt[0];
                if (fmt[1] >= 1 && fmt[1] <= 8)          v.channels = fmt[1];
                if (fmt[2] >= 8000 && fmt[2] <= 192000)  v.rate     = fmt[2];
                if (getenv("PROSPER_NGS2_TRACE"))
                    fprintf(stderr, "[ngs2] FORMAT voice=0x%llx waveform_type=0x%x ch=%u rate=%u\n",
                            (unsigned long long)handle, fmt[0], fmt[1], fmt[2]);
            }
        } else if (head.id == kNgs2ParamMatrixLevels) {
            // Per-render output level matrix. Observed layout (VoiceControl capture): a leading count
            // word (+0x10) and a ramp/config float (+0x14, ~1e4), then the front L/R output levels as
            // two floats at +0x18 and +0x1c (e.g. 0.707, 1.0). Apply them as this voice's front L/R
            // gains so the game's own mix balance holds (this is what keeps the summed bus in range —
            // prosper was mixing every voice at unity before). Values outside a sane [0,4] range are
            // ignored (the matrix also carries non-gain fields like ramp lengths).
            // SURROUND-TODO: the full matrix maps source channels to ALL output ports (up to 7.1); we
            // read only the front pair. Parse the rest into gain[2..7] once a surround bus exists.
            float lv[2] = {};
            if (audio_read_bytes(p + 0x18, lv, sizeof(lv))) {
                if (lv[0] >= 0.0f && lv[0] <= 4.0f) v.gain[0] = lv[0];
                if (lv[1] >= 0.0f && lv[1] <= 4.0f) v.gain[1] = lv[1];
            }
        } else if (head.id == kNgs2ParamWaveformBlock) {
            uint64_t ptr = 0;
            if (audio_read_bytes(p + 8, &ptr, 8) && ptr >= 0x200000000ull) {
                // Enqueue the block for in-order playback (bounded so a runaway feed can't grow it
                // without limit). The mixer plays through the queue one block at a time.
                if (v.blocks.size() < 64) v.blocks.push_back(ptr);
                v.data_ptr     = ptr;  // last fed block (FEED trace / diagnostics)
                v.playing      = true; // streaming
                v.idle         = 0;    // fed this render -> not idle
                if (getenv("PROSPER_NGS2_TRACE")) {
                    int16_t probe[64] = {}; size_t g = audio_read_bytes_partial(ptr, probe, sizeof probe);
                    int pk = 0; for (size_t j = 0; j < g / 2; j++) { int a = probe[j] < 0 ? -probe[j] : probe[j]; if (a > pk) pk = a; }
                    fprintf(stderr, "[ngs2] FEED voice=0x%llx ptr=0x%llx got=%zu probe_peak=%d ch=%u rate=%u\n",
                            (unsigned long long)handle, (unsigned long long)ptr, g, pk, v.channels, v.rate);
                }
            }
        } else if (head.id == kNgs2ParamVoiceCallback) {
            // Ngs2VoiceCallbackParam: header(8), callback fn(8), callback_data(8), flags(4), reserved(4).
            // The guest registers a per-voice waveform-block completion handler here; NGS2 invokes it as
            // each fed block finishes so the guest queues the next (see ngs2_fire_pending_callbacks).
            uint64_t fn = 0, data = 0; uint32_t fl = 0;
            audio_read_bytes(p + 8,  &fn,   8);
            audio_read_bytes(p + 16, &data, 8);
            audio_read_bytes(p + 24, &fl,   4);
            v.cb_fn = fn; v.cb_data = data; v.cb_flags = fl;
            if (getenv("PROSPER_NGS2_TRACE"))
                fprintf(stderr, "[ngs2] CALLBACK voice=0x%llx fn=0x%llx data=0x%llx flags=0x%x\n",
                        (unsigned long long)handle, (unsigned long long)fn, (unsigned long long)data, fl);
        }
        if (head.next == 0) break;
        p += (uint64_t)head.next;
    }
}

// --- NGS2 waveform-block completion callbacks --------------------------------------------------
// A streaming guest (Dead Cells) registers a per-voice callback (voice param 0x0007) and expects
// NGS2 to invoke it as each fed waveform block finishes, so it can queue the next block. Without it
// the stream stalls after the initial ring fill and no music is ever produced. The registered handler
// is GUEST code; recovered from Dead Cells' handler at eboot+0x1740f50 it reads a
// SceNgs2VoiceCallbackInfo*:
//     +0x00 : callbackData  (the game's per-stream context pointer we were handed at registration)
//     +0x10 : event flags   (bit0 = "waveform block consumed"; the handler no-ops if it is clear)
//     +0x18 : a pointer whose byte +0x48 must be non-zero for the handler to run its state update
// The handler then re-enters sceNgs2VoiceGetState and reads num_decoded_samples (state+0x10) to
// advance its stream clock. So the callback must run (a) with the caller's guest %fs restored (its
// nested imports resolve through the guest TCB) and (b) OUTSIDE g_ngs2_mx (it re-enters our HLE).
namespace {
struct Ngs2PendingCallback { uint64_t fn, data; uint32_t flags; };
// Populated (under g_ngs2_mx) while mixing, drained after the lock drops on the same SystemRender
// (guest) thread. Thread-local: each guest audio thread renders and fires its own system's voices.
thread_local std::vector<Ngs2PendingCallback> t_ngs2_pending_cb;
thread_local uint64_t t_ngs2_render_guest_fs = 0;  // caller's guest %fs, captured at SystemRender entry

// Firing a guest callback needs the FSGSBASE swap and the SystemRender entry trampoline, both of which
// only exist on the Linux guest-execution path (the trampoline is registered under the same guard). On
// other platforms the callbacks are simply not fired (music streaming is a Linux capability for now), so
// nothing here — including the queuing in ngs2_mix_voices — is compiled elsewhere.
#if defined(__linux__)
inline uint64_t ngs2_rd_fsbase() { uint64_t v; __asm__ volatile("rdfsbase %0" : "=r"(v)); return v; }
inline void     ngs2_wr_fsbase(uint64_t v) { __asm__ volatile("wrfsbase %0" : : "r"(v)); }

void ngs2_fire_pending_callbacks() {
    if (t_ngs2_pending_cb.empty()) return;
    std::vector<Ngs2PendingCallback> pend;
    pend.swap(t_ngs2_pending_cb);
    const uint64_t guest_fs = t_ngs2_render_guest_fs;
    uint64_t host_fs = 0; bool swapped = false;
    if (guest_fs) { host_fs = ngs2_rd_fsbase(); ngs2_wr_fsbase(guest_fs); swapped = true; }
    for (const auto& cb : pend) {
        if (!cb.fn) continue;
        // Minimal, valid SceNgs2VoiceCallbackInfo on the host stack (guest code reads it directly; the
        // guest runs in-process so a host address is a valid guest pointer). obj carries the +0x48
        // "active" byte the handler checks via eboot+0x173f490.
        alignas(16) uint8_t obj[0x50] = {}; obj[0x48] = 1;
        alignas(16) uint8_t info[0x40] = {};
        *(uint64_t*)(info + 0x00) = cb.data;          // callbackData -> game stream context
        *(uint32_t*)(info + 0x10) = cb.flags | 1u;    // event flags, bit0 (block-consumed) set
        *(uint64_t*)(info + 0x18) = (uint64_t)(uintptr_t)obj;
        ((void (*)(uint64_t))(uintptr_t)cb.fn)((uint64_t)(uintptr_t)info);
    }
    if (swapped) ngs2_wr_fsbase(host_fs);
}
#endif
} // namespace

// Mix all playing voices of `system` into one render buffer of `frames` interleaved frames at
// `out_rate`/`out_channels`, accumulating float samples into `mix` (size frames*out_channels).
// Reads source PCM fault-safely in one bulk read per voice; a short/unmapped source just stops that
// voice. Linear resample from each voice's source rate; source stereo→out stereo direct, mono→dup.
void ngs2_mix_voices(uint64_t system, float* mix, uint32_t frames, uint32_t out_channels, uint32_t out_rate) {
    // De-duplicate the 16-voice sampler pool: Dead Cells cycles a fixed pool and reuses a small set of
    // shared PCM buffers, so after the pool wraps, several stale voice handles still reference the SAME
    // buffer. NGS2 stops those on hardware; prosper lacks the (undocumented) one-shot-end/stop state, so
    // without this the same click renders on top of itself a few ms apart (audible doubling once >16
    // clicks have fired). Render each distinct buffer only once, choosing the FRESHEST voice (smallest
    // cursor = most recently (re)triggered); older duplicates at that buffer are skipped this pass.
    // CONFIDENCE: MED — a pragmatic proxy for real voice lifecycle; identical-buffer voices are truly
    // redundant into a mono/stereo bus, so this cannot drop a distinct sound.
    // A streamed waveform block is one grain of frames (the guest feeds one block per grain), so the
    // retire boundary and the per-block read length both track num_grain_samples rather than a constant.
    // For Dead Cells (grain 4096) this is the validated value; a title with a different grain retires and
    // fires its completion callback at the correct rate instead of ~grain/4096x off.
    const uint64_t kBlockFrames = g_ngs2_grain ? g_ngs2_grain : 4096;
    const uint64_t now_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::set<uint64_t> rendered_front;         // render each distinct front block once (pool reuse across voices)
    for (auto& [key, v] : g_ngs2_voices) {
        if (!v.playing) continue;
        if (v.blocks.empty()) {
            // Not fed this render. A callback (streaming) voice is re-fed by its block-completion callback,
            // so it may sit briefly between feeds; give it a much longer idle window than a one-shot before
            // expiring (a genuinely dead stream still stops, just later). HOLD the playback clock across the
            // gap by rebasing t0 so (now - t0)*rate == played_samples: the pause neither advances nor
            // rewinds the stream position, and playback resumes seamlessly. (Zeroing t0 here instead would,
            // on resume, make target=0 << played_samples and clamp advance to 0 for played_samples/rate
            // seconds — a multi-second freeze whenever the queue momentarily drains.)
            if (v.rate) v.t0_ns = now_ns - (uint64_t)((double)v.played_samples / (double)v.rate * 1e9);
            const int idle_max = v.cb_fn ? 240 : kNgs2VoiceIdleRendersMax;
            if (++v.idle > idle_max) v.playing = false;
            continue;
        }
        v.idle = 0;
        if (v.wtype != kNgs2WaveformTypePcmS16) {           // only S16 PCM implemented (ATRAC9 deferred)
            if (getenv("PROSPER_NGS2_TRACE")) {
                static uint32_t skipped = 0;
                if ((skipped++ & 0xff) == 0)
                    fprintf(stderr, "[ngs2] SKIP non-PCM voice=0x%x waveform_type=0x%x (music/codec?)\n", key, v.wtype);
            }
            continue;
        }
        if (!rendered_front.insert(v.blocks.front()).second) continue;  // duplicate front block this pass

        const double step = (double)v.rate / (double)(out_rate ? out_rate : 48000);
        const uint32_t src_ch = v.channels ? v.channels : 2;
        // Gather the source frames this render needs by reading whole blocks in queue order. Reading one
        // block at a time (never a single span across two pool buffers) is what stops the mixer sampling
        // an unrelated neighbouring block. cursor is the fractional read position within blocks.front().
        const uint64_t need_frames = (uint64_t)(v.cursor + (double)frames * step) + 2;
        std::vector<int16_t> src;
        src.reserve((size_t)std::min<uint64_t>(need_frames, 1u << 20) * src_ch);
        for (size_t bi = 0; bi < v.blocks.size() && src.size() < need_frames * src_ch; ++bi) {
            std::vector<int16_t> blk((size_t)kBlockFrames * src_ch, 0);
            const size_t got = audio_read_bytes_partial(v.blocks[bi], blk.data(), blk.size() * 2);
            const size_t bf  = got / ((size_t)src_ch * 2);   // valid frames in this block
            src.insert(src.end(), blk.begin(), blk.begin() + bf * src_ch);
            if (bf < kBlockFrames) break;                    // short/unmapped block -> stream truncated
        }
        const uint64_t have_frames = src.size() / src_ch;
        if (have_frames == 0) { v.blocks.clear(); v.playing = false; continue; }
        for (uint32_t f = 0; f < frames; ++f) {
            const double sp = v.cursor + (double)f * step;
            const uint64_t i0 = (uint64_t)sp;
            const double frac = sp - (double)i0;
            bool ran_out = false;
            for (uint32_t oc = 0; oc < out_channels; ++oc) {
                // Channel routing. SURROUND-LIMITATION: prosper targets a 2-channel (front L/R) bus,
                // so we map source channel oc→oc for stereo and fold mono to both. A true PS5 surround
                // (up to 7.1) output would route every source channel to its speaker via the full level
                // matrix; here only gain[0]/gain[1] (front L/R) are applied. Extend when a surround bus
                // and the full matrix parse (gain[2..7]) land.
                const uint32_t sc = (src_ch == 1) ? 0 : (oc < src_ch ? oc : src_ch - 1);
                const uint64_t a = (i0)     * src_ch + sc;
                const uint64_t b = (i0 + 1) * src_ch + sc;
                if (b >= have_frames * src_ch) { ran_out = true; break; }
                const double s = (1.0 - frac) * src[a] + frac * src[b];
                mix[f * out_channels + oc] += (float)(s / 32768.0) * (oc < 8 ? v.gain[oc] : 1.0f);
            }
            if (ran_out) break;   // consumed all valid source this render
        }
        // Advance the read cursor by WALL-CLOCK real time, not by frames*step. The guest over-calls
        // SystemRender (measured ~4x real time) while its decoder feeds blocks at real time; advancing by
        // frames*step every call would drain the queue ~4x faster than it is filled (the rhythmic under-
        // run). Pacing to real time makes consumption match the real-time feed, so the queue holds its
        // depth. The full frames*step window is still MIXED from the cursor each call — the guest double-
        // buffers and outputs one render per real-time grain — but the cursor only moves by elapsed time.
        if (v.t0_ns == 0) { v.t0_ns = now_ns; }
        const double target = (double)(now_ns - v.t0_ns) * 1e-9 * (double)v.rate;
        double advance = target - (double)v.played_samples;
        if (advance < 0.0) advance = 0.0;
        if (advance > (double)frames * step) advance = (double)frames * step;  // at most one grain/call
        v.cursor += advance;
        v.played_samples += (uint64_t)advance;
        // Retire fully-consumed blocks; each completed block fires the guest's block-completion callback
        // so it queues the next one. Fired after the lock drops (see ngs2_fire_pending_callbacks).
        while (v.cursor >= (double)kBlockFrames && !v.blocks.empty()) {
            v.blocks.pop_front();
            v.cursor -= (double)kBlockFrames;
#if defined(__linux__)
            if (v.cb_fn) t_ngs2_pending_cb.push_back({v.cb_fn, v.cb_data, v.cb_flags});
#endif
        }
    }
}

bool ngs2_read_u32(uint64_t src, uint32_t& value) {
    return audio_read_bytes(src, &value, sizeof value);
}

bool ngs2_read_bytes(uint64_t src, void* dst, size_t size) {
    return audio_read_bytes(src, dst, size);
}

bool ngs2_zero_bytes(uint64_t dst, size_t size) {
    // Do not use thread_local here: guest execution swaps %fs, and adding host TLS to prosper_core
    // perturbs Messenger before its first syscall. Process-global zero storage is sufficient because
    // NGS2 rendering is serialized by its audio thread; the lock also makes that contract explicit.
    std::lock_guard<std::mutex> lock(g_ngs2_zero_mx);
    if (g_ngs2_zeros.size() < size) g_ngs2_zeros.resize(size, 0);
    return audio_store_bytes(dst, g_ngs2_zeros.data(), size);
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
    // a0 = const SceNgs2SystemOption* (optional). num_grain_samples is at +0x70 (size@0, name[64]@8,
    // job_scheduler_options[4]@0x48, flags@0x68, max_grain_samples@0x6c, num_grain_samples@0x70). Honour
    // it when present and sane, else keep the streaming-safe default. Dead Cells passes a0 == null.
    // CONFIDENCE: MED — offset from the published SceNgs2SystemOption layout; not yet seen non-null live.
    if (a0) { uint32_t g = 0; if (audio_read_bytes(a0 + 0x70, &g, 4) && g >= 64 && g <= 8192) g_ngs2_grain = g; }
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

// PROSPER_NGS2_TRACE=2 additionally dumps the voice-command param chain: each entry is a
// Ngs2VoiceParamHead { uint16 size; int16 next; uint32 id; payload... } (Sony's documented
// voice-param list shape). This is capture-first RE for the real sampler implementation —
// the play/setup commands carry the waveform format (codec/rate/channels) and data pointers
// we need before mixing can be implemented faithfully.
void ngs2_dump_param_chain(const char* tag, uint64_t list) {
    if (ngs2_trace_level() < 2 || !list) return;
    uint64_t p = list;
    for (int i = 0; i < 16 && p; ++i) {
        struct { uint16_t size; int16_t next; uint32_t id; } head{};
        if (!ngs2_read_bytes(p, &head, sizeof(head))) break;
        fprintf(stderr, "[ngs2]   %s param[%d] @0x%llx size=0x%x next=%d id=0x%08x\n",
                tag, i, (unsigned long long)p, head.size, head.next, head.id);
        if (head.size == 0 || head.size > 0x400) break;
        a2_dump("payload", p, head.size < 0x60 ? head.size : 0x60);
        // Fingerprint any guest pointer the payload carries (waveform data address): dump its first
        // bytes so PCM (sample data) vs a codec container (RIFF/'RIFF', 'AT9 ', ATRAC9 config) is
        // decidable from the magic. Guest heap/direct pointers live in the 0x2_00000000+ aperture.
        for (uint16_t off = 8; off + 8 <= head.size; off += 8) {
            uint64_t val = 0;
            if (!ngs2_read_bytes(p + off, &val, 8)) break;
            if (val >= 0x200000000ull && val < 0x8000000000ull) {
                fprintf(stderr, "[ngs2]   %s param[%d] +0x%x -> guest ptr 0x%llx:\n",
                        tag, i, off, (unsigned long long)val);
                a2_dump("wavedata", val, 0x40);
            }
        }
        if (head.next == 0) break;
        p += (uint64_t)head.next;
    }
}

HLE(ngs2_voice_control) {
    NGS2_LOG("sceNgs2VoiceControl");
    ngs2_dump_param_chain("VoiceControl", a1);
    if ((a0 & kNgs2VoiceMask) != kNgs2VoiceTag) return kNgs2ErrInvalidVoice;
    if (!ngs2_mix_disabled()) { std::lock_guard<std::mutex> lock(g_ngs2_mx); ngs2_apply_voice_params(a0, a1); }
    return 0;
}

HLE(ngs2_voice_run_commands) {
    NGS2_LOG("sceNgs2VoiceRunCommands");
    ngs2_dump_param_chain("RunCommands", a1);
    if ((a0 & kNgs2VoiceMask) != kNgs2VoiceTag) return kNgs2ErrInvalidVoice;
    if (!ngs2_mix_disabled()) { std::lock_guard<std::mutex> lock(g_ngs2_mx); ngs2_apply_voice_params(a0, a1); }
    return 0;
}

HLE(ngs2_voice_get_state) {
    NGS2_LOG("sceNgs2VoiceGetState");
    if ((a0 & kNgs2VoiceMask) != kNgs2VoiceTag) return kNgs2ErrInvalidVoice;
    if (!a1 || !a2 || a2 > 0x1000 || !a2_store_zeros(a1, (size_t)a2)) return kNgs2ErrInvalidOut;
    // SceNgs2SamplerVoiceState (0x30): state_flags@0x00, envelope_height@0x04(f), peak_height@0x08(f),
    // reserved@0x0c, num_decoded_samples@0x10(u64), decoded_data_size@0x18(u64), user_data@0x20,
    // waveform_data@0x28. The block-completion callback re-enters here and reads num_decoded_samples to
    // advance its stream clock, so it must reflect real playback progress or the stream never advances.
    std::lock_guard<std::mutex> lock(g_ngs2_mx);
    auto it = g_ngs2_voices.find((uint32_t)(a0 & 0xffffff));
    const bool playing = it != g_ngs2_voices.end() && it->second.playing && it->second.data_ptr;
    a2_store_u32(a1, playing ? 0x3u : 0u);          // state_flags: Playing / Empty
    if (a2 >= 0x30) {
        a2_store_u32(a1 + 0x04, 0x3f800000u);       // envelope_height = 1.0f
        if (it != g_ngs2_voices.end()) {
            const uint64_t decoded = it->second.played_samples;
            const uint32_t ch = it->second.channels ? it->second.channels : 2;
            a2_store_u64(a1 + 0x10, decoded);        // num_decoded_samples (per-channel frames)
            a2_store_u64(a1 + 0x18, decoded * ch * 2); // decoded_data_size (bytes, s16)
        }
    }
    return 0;
}

// State-flag bits the guest tests (Kyty Ngs2GetStateFlags): Empty=0, Playing=0x3, Paused=0x5,
// Stopped=0xb. Report Playing for a voice we are actively mixing so the game's audio state machine
// treats the voice as live; else Empty.
uint32_t ngs2_voice_state_flags(uint64_t handle) {
    if ((handle & kNgs2VoiceMask) != kNgs2VoiceTag) return 0;
    auto it = g_ngs2_voices.find((uint32_t)(handle & 0xffffff));
    return (it != g_ngs2_voices.end() && it->second.playing && it->second.data_ptr) ? 0x3u : 0u;
}

// sceNgs2VoiceGetStateFlags(voice, uint32_t* outFlags) -> 0. Was UNREGISTERED (#TBD): the generic
// stub returned 0 and left *outFlags uninitialized, so the guest read garbage voice state and its
// audio scheduler misbehaved (the uninit-out class). Return real Playing/Empty flags.
HLE(ngs2_voice_get_state_flags) {
    NGS2_LOG("sceNgs2VoiceGetStateFlags");
    if ((a0 & kNgs2VoiceMask) != kNgs2VoiceTag) return kNgs2ErrInvalidVoice;
    if (!a1) return kNgs2ErrInvalidOut;
    uint32_t flags; { std::lock_guard<std::mutex> lock(g_ngs2_mx); flags = ngs2_voice_state_flags(a0); }
    return a2_store_u32(a1, flags) ? 0 : kNgs2ErrInvalidOut;
}

// sceNgs2RackDestroy(rack, Ngs2ContextBufferInfo* out) -> 0. Was UNREGISTERED. Release the rack and
// drop its voices so their handles do not leak into later mixes; the out context-buffer info, if
// present, is zeroed (we own no returned host buffer).
HLE(ngs2_rack_destroy) {
    NGS2_LOG("sceNgs2RackDestroy");
    std::lock_guard<std::mutex> lock(g_ngs2_mx);
    const uint64_t slot = a0 & 0xff;
    if ((a0 & kNgs2TagMask) != kNgs2RackTag || slot < 1 || slot > 32) return kNgs2ErrInvalidRack;
    g_ngs2_racks[slot - 1] = {};
    for (auto it = g_ngs2_voices.begin(); it != g_ngs2_voices.end(); )
        it = ((it->first >> 8) == slot) ? g_ngs2_voices.erase(it) : std::next(it);
    if (a1 && !a2_store_zeros(a1, 0x40)) return kNgs2ErrInvalidOut;
    return 0;
}

// sceNgs2GeomApply(...) -> 0. Was UNREGISTERED (NID eF8yRCC6W64, seen unimplemented in boot logs):
// applies 3D listener/source geometry to a voice's pan matrix. prosper mixes without 3D panning, so
// this is a faithful no-op success (the guest keeps its pre-initialized pan) — never leave it to the
// generic stub, which the guest cannot distinguish from a real error path.
HLE(ngs2_geom_apply) {
    NGS2_LOG("sceNgs2GeomApply");
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
            !info.buffer || info.size > 64ull * 1024 * 1024)
            return kNgs2ErrInvalidOut;

        // The render buffer is the mix destination. Output format: waveform_type 0x12 == S16, anything
        // else (the NGS2 render default) == F32. Frame count derives from size and the output stride.
        const bool     out_s16      = info.waveform_type == kNgs2WaveformTypePcmS16;
        const uint32_t out_channels = (info.channels >= 1 && info.channels <= 8) ? info.channels : 2;
        const uint32_t out_bps      = out_s16 ? 2 : 4;
        uint32_t       frames       = (uint32_t)(info.size / ((uint64_t)out_channels * out_bps));
        // Fill only num_grain_samples frames of an oversized buffer (see g_ngs2_grain); the rest stays
        // zeroed. Filling the whole capacity makes a streaming voice's read cursor race ahead of the
        // guest's block queue and produce the rhythmic underrun.
        if (g_ngs2_grain && frames > g_ngs2_grain) frames = g_ngs2_grain;

        if (ngs2_mix_disabled() || frames == 0) {
            if (!ngs2_zero_bytes(info.buffer, (size_t)info.size)) return kNgs2ErrInvalidOut;
            continue;
        }

        std::vector<float> mix((size_t)frames * out_channels, 0.0f);
        int playing = 0;
        {
            std::lock_guard<std::mutex> lock(g_ngs2_mx);
            for (auto& [k, vc] : g_ngs2_voices) if (vc.playing && vc.data_ptr) playing++;
            ngs2_mix_voices(a0, mix.data(), frames, out_channels, 48000 /* NGS2 system rate */);
        }
        if (ngs2_trace_enabled()) {
            float pk = 0; for (float f : mix) { float a = f < 0 ? -f : f; if (a > pk) pk = a; }
            static uint64_t rc = 0; static float maxpk = 0; static int maxpv = 0;
            if (pk > maxpk) maxpk = pk; if (playing > maxpv) maxpv = playing;
            if ((++rc & 0x3f) == 1 || (pk > 0 && playing > 0))
                fprintf(stderr, "[ngs2] render buf#%llu out=%s ch=%u frames=%u size=%llu playing_voices=%d(max%d) mix_peak=%.4f(max%.4f)\n",
                        (unsigned long long)i, out_s16 ? "s16" : "f32", out_channels, frames,
                        (unsigned long long)info.size, playing, maxpv, pk, maxpk);
        }
        // Serialize the float mix into the guest buffer's native format (clamped), then commit it.
        std::vector<uint8_t> out((size_t)info.size, 0);
        for (size_t s = 0; s < mix.size(); ++s) {
            float f = mix[s];
            f = f > 1.0f ? 1.0f : (f < -1.0f ? -1.0f : f);
            if (out_s16) { int16_t v = (int16_t)(f * 32767.0f); memcpy(&out[s * 2], &v, 2); }
            else         { memcpy(&out[s * 4], &f, 4); }
        }
        if (!audio_store_bytes(info.buffer, out.data(), out.size())) return kNgs2ErrInvalidOut;
    }
    return 0;
}

// SystemRender fires guest waveform-block callbacks (queued during mixing). Guest code must run with
// the caller's guest %fs, recovered from the import-stub frame via the shared entry trampoline (it
// forwards entry %rsp as a 7th argument), and after mixing has released g_ngs2_mx. Non-Linux platforms
// register the plain handler; their guest-callback dispatch would need prosper_call_guest_sysv instead.
#if defined(__linux__)
extern "C" uint64_t ngs2_system_render_c(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                         uint64_t a4, uint64_t a5, uint64_t entry_rsp);
PROSPER_ASM_TRAMPOLINE(ngs2_system_render_entry, ngs2_system_render_c)
extern "C" void ngs2_system_render_entry();
extern "C" uint64_t ngs2_system_render_c(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                         uint64_t a4, uint64_t a5, uint64_t entry_rsp) {
    t_ngs2_render_guest_fs = prosper::callback_guest_fs_from_entry_stack(entry_rsp);
    const uint64_t r = ngs2_system_render(a0, a1, a2, a3, a4, a5);
    ngs2_fire_pending_callbacks();
    t_ngs2_render_guest_fs = 0;
    return r;
}
#endif

HLE(ngs2_geom_reset_listener) {
    NGS2_LOG("sceNgs2GeomResetListenerParam");
    return a0 && a2_store_zeros(a0, 0xa0) ? 0 : kNgs2ErrInvalidOut;
}

HLE(ngs2_geom_calc_listener) {
    NGS2_LOG("sceNgs2GeomCalcListener");
    return a0 && a1 && a2_store_zeros(a1, 0x60) ? 0 : kNgs2ErrInvalidOut;
}

#undef NGS2_LOG

// --- libSceVoice (voice chat, #1158). prosper has no voice-capture backend (no microphone / network
// voice path). GTA V (PPSA04263, RAGE) initialises voice at boot and, when it believes voice is up,
// builds a voice context and sizes its ring buffers by DIVIDING by voice-parameter fields (bitrate,
// samples-per-frame, ...) it obtains from the voice API — e.g. `x * 8000 / bitrate` at eboot+0x26043b0
// and `x / ctx[+0x2dc0]` at eboot+0x2905... . prosper's generic unimplemented stubs returned 0 WITHOUT
// writing those outputs, so several divisors were 0 → a cascade of guest integer #DE → SIGFPE.
//
// Report the subsystem as unavailable from sceVoiceInit: GTA treats a failed voice init as "voice
// chat off" (a real state — e.g. voice disabled in system settings / no headset region) and skips its
// entire voice-buffer setup, so none of the divide-by-zero math runs and boot continues to the title.
// This is the honest model for a host with no voice hardware, and voice is not needed to reach the
// title screen. CONFIDENCE: MED — verified live to clear the SIGFPE cascade and reach the GTA V title;
// a faithful null-backend voice implementation (init succeeds, ports return valid silent parameters)
// is the alternative if a title ever needs working voice (tracked separately).
HLE(voice_init_unavailable) {   // sceVoiceInit / sceVoiceInitHQ -> report voice unavailable
    // Negative SCE_VOICE-class error (facility 0x8041; the guest only sign-checks the return via `js`).
    // Chosen as a HARD, non-retryable init failure: it is NOT the ALREADY_INITIALIZED code (0x80410004,
    // which some engines treat as success), so the guest takes its "voice off" branch and does not retry.
    return (uint64_t)(int64_t)(int32_t)0x80410002;
}

void register_audio_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceAudioOutInit", audio_init);
    R("sceAudioOutOpen", audio_open);
    // libSceVoice: report voice unavailable so titles with no host voice backend skip voice-chat
    // setup instead of #DE'ing on never-written voice-parameter divisors (GTA V, #1158). Both the
    // standard and HQ init entry points get the same "unavailable" answer.
    R("sceVoiceInit", voice_init_unavailable);
    R("sceVoiceInitHQ", voice_init_unavailable);
    R("sceAudioOutOutput", audio_output);
    R("sceAudioOutOutputs", audio_outputs);
    R("sceAudioOutSetVolume", audio_set_volume);
    R("sceAudioOutClose", audio_close);
    R("sceAudioOutGetPortState", audio_get_port_state);
    // libSceAudioIn inherited core: deterministic null microphone with real-time capture pacing.
    R("sceAudioInInit", audio_in_init);
    R("sceAudioInOpen", audio_in_open);
    R("sceAudioInInput", audio_in_input);
    R("sceAudioInClose", audio_in_close);
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
    R("sceAudioOut2GetSpeakerArrayMemorySize", audio2_get_speaker_array_memory_size);  // GTA V (#1134)
    R("sceAudioOut2SpeakerArrayCreate", audio2_speaker_array_create);
    R("sceAudioOut2SpeakerArrayDestroy", audio2_speaker_array_destroy);
    R("sceAudioOut2GetSpeakerArrayCoefficients", audio2_get_speaker_array_coefficients);
    R("sceAudioOut2GetSpeakerArrayAmbisonicsCoefficients",
      audio2_get_speaker_array_ambisonics_coefficients);
    R("sceAudioOut2MasteringInit", audio2_mastering_init);
    R("sceAudioOut2MasteringTerm", audio2_mastering_term);
    R("sceAudioOut2MasteringSetParam", audio2_mastering_set_param);
    R("sceAudioOut2MasteringGetState", audio2_mastering_get_state);
    // libSceAjm (#187): headless decode-lifecycle (valid handles, no actual decode -> silence).
    R("sceAjmInitialize", ajm_initialize);            R("sceAjmFinalize", ajm_finalize);
    R("sceAjmModuleRegister", ajm_module_register);   R("sceAjmModuleUnregister", ajm_module_unregister);
    R("sceAjmInstanceCreate", ajm_instance_create);   R("sceAjmInstanceDestroy", ajm_instance_destroy);
    R("sceAjmBatchInitialize", ajm_batch_initialize);
    R("sceAjmBatchStartBuffer", ajm_batch_start);     R("sceAjmBatchWait", ajm_batch_wait);
    R("sceAjmBatchCancel", ajm_batch_cancel);
    R("sceAjmBatchJobControlBufferRa", ajm_batch_job_control_buffer_ra);
    R("sceAjmBatchJobInlineBuffer", ajm_batch_job_inline_buffer);
    R("sceAjmBatchJobRunBufferRa", ajm_batch_job_run_buffer_ra);
    R("sceAjmBatchJobRunSplitBufferRa", ajm_batch_job_run_split_buffer_ra);
    R("sceAjmBatchErrorDump", ajm_batch_errordump);
    R("sceAjmBatchJobInitialize", ajm_batch_job_initialize);
    R("sceAjmBatchJobDecode", ajm_batch_job_decode);
    R("sceAjmBatchJobSetGaplessDecode", ajm_batch_job_gapless);
    R("sceAjmBatchStart", ajm_batch_start2);
    Hle::register_fn("pgFAiLR5qT4", ngs2_system_query_buffer, "sceNgs2SystemQueryBufferSize");
    Hle::register_fn("koBbCMvOKWw", ngs2_system_create, "sceNgs2SystemCreate");
    Hle::register_fn("0eFLVCfWVds", ngs2_rack_query_buffer, "sceNgs2RackQueryBufferSize");
    Hle::register_fn("cLV4aiT9JpA", ngs2_rack_create, "sceNgs2RackCreate");
    Hle::register_fn("MwmHz8pAdAo", ngs2_rack_get_voice, "sceNgs2RackGetVoiceHandle");
    Hle::register_fn("uu94irFOGpA", ngs2_voice_control, "sceNgs2VoiceControl");
    Hle::register_fn("AbYvTOZ8Pts", ngs2_voice_run_commands, "sceNgs2VoiceRunCommands");
    Hle::register_fn("-TOuuAQ-buE", ngs2_voice_get_state, "sceNgs2VoiceGetState");
    Hle::register_fn("rEh728kXk3w", ngs2_voice_get_state_flags, "sceNgs2VoiceGetStateFlags");
    Hle::register_fn("lCqD7oycmIM", ngs2_rack_destroy, "sceNgs2RackDestroy");
    Hle::register_fn("eF8yRCC6W64", ngs2_geom_apply, "sceNgs2GeomApply");
    Hle::register_fn("0lbbayqDNoE", ngs2_geom_reset_source, "sceNgs2GeomResetSourceParam");
#if defined(__linux__)
    // The trampoline forwards entry %rsp so the handler can recover the caller's guest %fs and fire
    // waveform-block completion callbacks (guest code) after mixing. See ngs2_system_render_c.
    Hle::register_fn("i0VnXM-C9fc", (HleFn)ngs2_system_render_entry, "sceNgs2SystemRender");
#else
    Hle::register_fn("i0VnXM-C9fc", ngs2_system_render, "sceNgs2SystemRender");
#endif
    Hle::register_fn("7Lcfo8SmpsU", ngs2_geom_reset_listener, "sceNgs2GeomResetListenerParam");
    Hle::register_fn("1WsleK-MTkE", ngs2_geom_calc_listener, "sceNgs2GeomCalcListener");
    #undef R
}

} // namespace prosper
