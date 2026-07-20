// hle_audio.cpp — libSceAudioOut/AudioIn HLE, with pluggable output (see audio.hpp).
//
// Decodes the PS5 sceAudioOut* calls into port lifecycle + interleaved PCM grains and forwards
// them to the installed backend. The inherited AudioIn core provides deterministic paced silence.
// prosper_core stays dependency-free; a concrete output frontend (SDL3, ...) installs itself via
// audio_set_sink() from outside the core.
#include "dispatch.hpp"
#include "nid.hpp"
#include "audio.hpp"
#include "callback_fs.hpp"      // recover the caller's guest %fs for firing guest callbacks
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
static void audio_in_reset_ports();

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

void audio_reset() {
    AudioSink* s = audio_sink();
    {
        std::lock_guard<std::mutex> lk(g_mx);
        for (int i = 0; i < kMaxPorts; i++) {
            if (g_ports[i].in_use && s) s->close(i + 1);
            g_ports[i] = Port{};
        }
        g_sink.store(&g_default_sink);
    }
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
    if (handle < 1 || handle > kMaxPorts) return;
    static struct Obs { uint64_t calls = 0; double peak = 0.0; FILE* dump = nullptr;
                        std::chrono::steady_clock::time_point last{}; } st[kMaxPorts];
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
uint32_t   g_a2_users = 0;

// AudioOut2 data path (derived live from Evergate, the first title to PLAY through this surface):
// each tick the guest calls PortSetAttributes(port, attrs, count) where an attribute triple
// {id=0, value_ptr, value_size=8} carries a guest pointer to the port's current grain of
// interleaved stereo float32 PCM (grain frames from the context param, 256 live). ContextAdvance
// advances engine state and ContextPush submits the grain to the device. We store each port's PCM
// pointer here and mix + forward all ports' grains to the host AudioSink at Push.
struct A2PortState {
    bool     used = false;
    uint64_t pcm_ptr = 0;      // guest address of this port's current PCM grain (attr id 0)
    uint32_t type = 0;         // portParam +0x00: 0 = MAIN (speaker output); others aux (personal/aux/...)
    uint32_t block_floats = 0; // portParam +0x04: interleaved floats per grain = channels * grain.
                               // Read EXACTLY this many; the channel count is block_floats/grain (no
                               // hardcoded channel assumption -> no over-read on non-8ch ports).
};
A2PortState g_a2_port_state[16];
constexpr int kA2SinkPort = 16;     // host sink port id for the AudioOut2 device (sink accepts 1..16;
                                    // v1 sceAudioOut handles stay in the low single digits)

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
    if (audio2log()) a2_dump("portParam", a1, 0x40);
    // portParam layout (live-decoded, both Evergate and Blasphemous 2): +0x00 u32 port type
    // (0 = MAIN speaker output), +0x04 u32 interleaved floats per grain block (= channels * grain;
    // MAIN is 0x800 = 256*8 -> the 8-channel 7.1 bed Evergate uses), +0x08 u32 sample rate (48000).
    // Reading block_floats here lets ContextPush read each port at its REAL width and derive the
    // channel count as block_floats/grain, instead of assuming every port is an 8-channel main bed.
    uint32_t ptype = 0, block_floats = 0;
    if (a1) {
        uint32_t hdr[2] = {0, 0};
        if (audio_read_bytes(a1, hdr, sizeof hdr)) { ptype = hdr[0]; block_floats = hdr[1]; }
    }
    std::lock_guard<std::mutex> lk(g_a2_mx);
    // First-free-slot allocation (not a monotonic counter): slots are recycled on PortDestroy, so a
    // title that churns ports over a long session can't exhaust the table or alias slot state.
    for (uint32_t id = 1; id <= 16; ++id) {
        if (g_a2_port_state[id - 1].used) continue;
        if (!a2_store_u64(a2, kA2PortTag | (uint64_t)id)) return kA2ErrInvalid;
        g_a2_port_state[id - 1] = A2PortState{true, 0, ptype, block_floats};
        return 0;
    }
    return kA2ErrInvalid;
}
HLE(audio2_port_destroy) {
    A2LOG("sceAudioOut2PortDestroy");
    // Clear the slot so a destroyed port stops being mixed (its guest PCM buffer may be freed and
    // reused) and the slot is reusable by the next PortCreate.
    const uint64_t pidx = a0 & 0xff;
    if ((a0 & ~0xffull) == kA2PortTag && pidx >= 1 && pidx <= 16) {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        g_a2_port_state[pidx - 1] = A2PortState{};
    }
    return 0;
}

// sceAudioOut2PortGetState(port, State* out) -> 0. Layout unknown; the pump loop consumed a
// zero-filled 0x20 in the capture run and kept cycling. CONFIDENCE: LOW (zero state = silent
// port, no observed guest field reads yet — re-probe if CRI branches on it).
HLE(audio2_port_get_state) {
    A2LOG("sceAudioOut2PortGetState");
    if (!a2_store_zeros(a1, 0x20)) return kA2ErrInvalid;
    return 0;
}
// sceAudioOut2PortSetAttributes(port, const SceAudioOut2Attribute* attrs, size_t count).
// Attribute = {u64 id, const void* value, size_t valueSize} (0x18 stride). Live-decoded ids:
//   0 = per-grain PCM data pointer (value: a guest pointer qword to interleaved stereo f32 samples).
// Unknown ids are skipped fail-visibly under PROSPER_AUDIO2LOG. CONFIDENCE: MED (Evergate-derived).
HLE(audio2_port_set_attr) {
    A2LOG("sceAudioOut2PortSetAttributes");
    const uint64_t pidx = (a0 & 0xff);
    if ((a0 & ~0xffull) != kA2PortTag || pidx < 1 || pidx > 16) return kA2ErrInvalid;
    if (a2 == 0) return 0;                        // no attributes: a legal no-op
    const uint64_t count = a2 <= 32 ? a2 : 32;    // defensive cap; log the clamp fail-visibly
    if (count != a2 && audio2log())
        fprintf(stderr, "[audio2] PortSetAttributes: count %llu clamped to 32\n", (unsigned long long)a2);
    std::lock_guard<std::mutex> lk(g_a2_mx);      // covers port state AND the diagnostic set below
    for (uint64_t i = 0; i < count; i++) {
        struct { uint64_t id, vptr, vsize; } at{};
        if (!audio_read_bytes(a1 + i * 0x18, &at, sizeof at)) break;
        if (at.id == 0 && at.vsize == 8) {
            uint64_t pcm = 0;
            if (audio_read_bytes(at.vptr, &pcm, 8)) g_a2_port_state[pidx - 1].pcm_ptr = pcm;
            if (getenv("PROSPER_AUDIO2_PROBE")) {
                static uint64_t seen_store[16] = {0};
                if (seen_store[(pidx - 1) & 15] != pcm) {
                    seen_store[(pidx - 1) & 15] = pcm;
                    fprintf(stderr, "[audio2-attr] port%llu id=0x%llx vptr=0x%llx vsize=%llu -> pcm=0x%llx\n",
                            (unsigned long long)pidx, (unsigned long long)at.id,
                            (unsigned long long)at.vptr, (unsigned long long)at.vsize,
                            (unsigned long long)pcm);
                }
            }
        } else if (audio2log() || getenv("PROSPER_AUDIO2_PROBE")) {
            static std::set<uint64_t> seen;
            if (seen.insert(at.id).second)
                fprintf(stderr, "[audio2] PortSetAttributes: unhandled attr id=%llu size=%llu\n",
                        (unsigned long long)at.id, (unsigned long long)at.vsize);
        }
    }
    return 0;
}

// sceAudioOut2ContextAdvance(ctx) -> 0. State advance only; the pacing lives in Push.
HLE(audio2_ctx_advance) { A2LOG("sceAudioOut2ContextAdvance"); return 0; }

// sceAudioOut2ContextPush(ctx, flag) -> 0. On hardware Push blocks while the output queue is
// full; the null backend reproduces that as one grain of wall-clock pacing per call (same
// model as RealtimeSilentSink) so CRI's server thread runs at real time, not a hot spin.
HLE(audio2_ctx_push) {
    A2LOG("sceAudioOut2ContextPush");
    uint64_t idx = a0 & 0xff;
    uint32_t grain = 256;
    // Mix each active port's current grain into a stereo bed. A port's grain buffer holds
    // `block_floats` interleaved float32 samples = (channels * grain), with the channel count taken
    // from the port's own param (portParam +0x04) — NOT a fixed assumption. The 7.1 MAIN port
    // (Evergate + Blasphemous 2) is 8 channels (block 0x800 = 256*8); a stereo/mono aux port is 2/1.
    // Reading each port at its real width is what prevents an over-read from splattering adjacent
    // heap (NaN/garbage) into the mix. We forward the front L/R pair (mono duplicates ch0).
    // SURROUND-TODO: channels 2..N (C/LFE/surrounds) are dropped until the host sink grows a
    // surround output; fold them in with proper downmix gains when it does.
    static thread_local std::vector<float> bed(4096 * 2);
    static thread_local std::vector<float> tmp;
    bool have_pcm = false;
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        A2Context* c = ((a0 & ~0xffull) == kA2CtxTag && idx >= 1 && idx <= 4 && g_a2_ctx[idx - 1].used)
                       ? &g_a2_ctx[idx - 1] : &g_a2_ctx[0];
        grain = (c->grain >= 64 && c->grain <= 4096) ? c->grain : 256;
        std::memset(bed.data(), 0, sizeof(float) * grain * 2);
        const bool probe = getenv("PROSPER_AUDIO2_PROBE") != nullptr;
        int pidx_probe = 0;
        for (const auto& ps : g_a2_port_state) {
            const int this_pidx = pidx_probe++;
            if (!ps.used || !ps.pcm_ptr) continue;
            // Only the MAIN port (type 0) drives the host speaker output. Other types route to
            // separate hardware devices (type 3/4 personal/pad-speaker, chat, etc.); summing them
            // into the main TV mix would leak that audio. They stay unmixed until per-type device
            // routing exists. CONFIDENCE: MED (Evergate + Blasphemous 2: MAIN is type 0; aux types
            // are silent here, so this changes nothing audible today but prevents a latent leak).
            if (ps.type != 0) {
                if (probe) { static uint64_t n[16] = {0};
                    if ((n[this_pidx & 15]++ % 256) == 0)
                        fprintf(stderr, "[audio2-probe] port%d type=%u NOT mixed (non-MAIN)\n",
                                this_pidx + 1, ps.type); }
                continue;
            }
            // Channel count from this port's own block size (block_floats = channels*grain), so we
            // never assume 8. Read EXACTLY channels*grain floats (<= block_floats, no over-read).
            uint32_t channels;
            if (ps.block_floats == 0) {
                channels = 8;                       // param was unreadable: legacy 7.1 MAIN assumption
            } else {
                channels = ps.block_floats / grain; // block is channels*grain for a well-formed param
                if (channels < 1 || channels > 8) {
                    if (probe) fprintf(stderr, "[audio2-probe] port%d skipped: block=%u grain=%u "
                                       "-> channels=%u (implausible)\n",
                                       this_pidx + 1, ps.block_floats, grain, channels);
                    continue;                        // fail-visible: don't over-read or mis-stride
                }
            }
            const uint32_t read_floats = channels * grain;   // <= block_floats: capped by construction
            if (tmp.size() < read_floats) tmp.resize(read_floats);
            const size_t got = audio_read_bytes_partial(ps.pcm_ptr, tmp.data(), sizeof(float) * read_floats);
            const size_t frames_got = got / (sizeof(float) * channels);
            if (probe) {
                static uint64_t call_ct[16] = {0};
                const size_t nf = frames_got * channels;
                size_t nan_ct = 0; float amax = 0.0f;
                for (size_t k = 0; k < nf; k++) { float v = tmp[k];
                    if (v != v) nan_ct++; else { float a = v < 0 ? -v : v; if (a > amax) amax = a; } }
                if ((call_ct[this_pidx & 15]++ % 64) == 0)
                    fprintf(stderr, "[audio2-probe] port%d type=%u ch=%u pcm=0x%llx frames=%zu "
                            "nan=%zu/%zu |max|=%.4g\n", this_pidx + 1, ps.type, channels,
                            (unsigned long long)ps.pcm_ptr, frames_got, nan_ct, nf, amax);
            }
            const uint32_t r = channels >= 2 ? 1u : 0u;     // mono duplicates ch0 to both outputs
            for (size_t fno = 0; fno < frames_got; fno++) {
                bed[fno * 2 + 0] += tmp[fno * channels + 0];   // front left
                bed[fno * 2 + 1] += tmp[fno * channels + r];   // front right
            }
            if (frames_got) have_pcm = true;
        }
    }
    AudioSink* sink = audio_sink();
    if (sink && have_pcm) {
        // Forward the mixed grain to the host device; the sink paces one grain per call in real
        // time (same contract as the v1 sceAudioOutOutput path), so no extra sleep on this path.
        // A FAILED device open must fall through to the silent wall-clock pacing below instead:
        // the SDL3 sink's output() is a no-op on a null stream (no sleep), and returning early on
        // every push would hot-spin the guest's pump thread on hosts with no audio device.
        static std::atomic<bool> opened{false};
        static std::atomic<bool> open_ok{false};
        AudioPortInfo info; info.freq = 48000; info.channels = 2; info.fmt = AudioFmt::F32;
        info.grain = (int)grain;
        if (!opened.exchange(true)) open_ok.store(sink->open(kA2SinkPort, info));
        if (open_ok.load()) {
            for (uint32_t i = 0; i < grain * 2; i++) {
                float v = bed[i];
                if (v != v) v = 0.0f;                        // NaN -> 0; Inf is caught by the clamp below
                bed[i] = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
            }
            audio_observe_output(kA2SinkPort, bed.data(), (int)grain, info);
            sink->output(kA2SinkPort, bed.data(), (int)grain);
            return 0;
        }
    }
    // No sink / no data yet: keep the silent real-time pacing so the guest's pump thread does not
    // hot-spin (on hardware Push blocks while the output queue is full).
    std::chrono::steady_clock::time_point target;
    {
        std::lock_guard<std::mutex> lk(g_a2_mx);
        A2Context* c = ((a0 & ~0xffull) == kA2CtxTag && idx >= 1 && idx <= 4 && g_a2_ctx[idx - 1].used)
                       ? &g_a2_ctx[idx - 1] : &g_a2_ctx[0];
        uint32_t g2 = c->grain ? c->grain : 256;
        auto dur = std::chrono::nanoseconds((long long)g2 * 1000000000LL / 48000);
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
    // AJM returns signed 32-bit SCE errors. Preserve that ABI in the full HLE return register,
    // matching the AudioOut error constants above rather than leaving the upper half zeroed.
    constexpr uint64_t AJM_ERR_INVALID_CONTEXT   = (uint64_t)(int64_t)(int32_t)0x80930002u;
    constexpr uint64_t AJM_ERR_INVALID_INSTANCE  = (uint64_t)(int64_t)(int32_t)0x80930003u;
    constexpr uint64_t AJM_ERR_INVALID_BATCH     = (uint64_t)(int64_t)(int32_t)0x80930004u;
    constexpr uint64_t AJM_ERR_INVALID_PARAMETER = (uint64_t)(int64_t)(int32_t)0x80930005u;

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
// sceAjmInitialize(s64 reserved, u32* out_context): create a context. Filling out_context is the point.
HLE(ajm_initialize) {
    if (a0 != 0 || !a1) return AJM_ERR_INVALID_PARAMETER;
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
    if (!a2_store_u32(a3, g_ajm_next.fetch_add(1))) return AJM_ERR_INVALID_PARAMETER;
    return 0;
}
// sceAjmInstanceDestroy(u32 context, u32 instance).
HLE(ajm_instance_destroy) {
    if (!a0) return AJM_ERR_INVALID_CONTEXT;
    if (!a1) return AJM_ERR_INVALID_INSTANCE;
    return 0;
}
// sceAjmBatchInitialize(void* pBatchBuffer, size_t szBatchBuffer, SceAjmBatchInfo* pBatchInfo): prepare
// the caller's batch info so its job builders write into pBatchBuffer starting at offset 0. Was
// UNIMPLEMENTED (Evergate calls it first, NID MmpF1XsQiHw): the generic stub returned 0 without touching
// pBatchInfo, so the guest built its audio-decode batch on a garbage buffer/cursor, never ran a decode
// job, and never opened an AudioOut port -> total silence. Layout: { void* pBuffer; size_t offset(cursor
// bytes used, starts 0); size_t size; }. CONFIDENCE: MED — 24-byte layout derived from Evergate's live
// call pattern (buffer/size/info args + a clean validated run); not yet cross-checked against a second
// title or the caller's stack-frame disassembly.
HLE(ajm_batch_initialize) {
    if (getenv("PROSPER_AUDIOLOG")) { static int n; if (n++ < 8)
        fprintf(stderr, "[ajm] BatchInitialize buf=0x%llx size=%llu info=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2); }
    if (!a0 || !a1 || !a2) return AJM_ERR_INVALID_PARAMETER;
    uint64_t info[3] = { a0, 0, a1 };   // pBuffer, offset=0, size
    if (!audio_store_bytes(a2, info, sizeof info)) return AJM_ERR_INVALID_PARAMETER;
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

void register_audio_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceAudioOutInit", audio_init);
    R("sceAudioOutOpen", audio_open);
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
