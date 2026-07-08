// test_audio — behavioral test for the sceAudioOut HLE (hle_audio.cpp) via a capturing sink.
//
// Agentic-first: no device, no PATH/DLL setup. Installs a fake AudioSink, drives the real HLE
// entrypoints through the dispatch table (so registration + arg decoding + forwarding are all
// exercised), and asserts the port lifecycle, format decoding, PCM forwarding, volume and error
// paths. Exit code is truth.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/audio.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(cond) do { if (!(cond)) { printf("  [FAIL] %s:%d  %s\n", __FILE__, __LINE__, #cond); fails++; } } while (0)

// Records every backend event so the test can assert on them.
struct CapturingSink : AudioSink {
    struct OpenEv  { int port; AudioPortInfo info; };
    struct OutEv   { int port; int frames; std::vector<uint8_t> pcm; };
    struct VolEv   { int port; uint32_t mask; std::vector<int> vols; };
    std::vector<OpenEv> opens;
    std::vector<OutEv>  outs;
    std::vector<VolEv>  vols_;
    std::vector<int>    closes;

    bool open(int port, const AudioPortInfo& info) override { opens.push_back({port, info}); return true; }
    void output(int port, const void* pcm, int frames) override {
        OutEv e; e.port = port; e.frames = frames;
        // Copy the delivered grain so we can verify byte-for-byte (uses the port's most-recent open).
        int fb = frames * (opens.empty() ? 4 : audio_frame_bytes(opens.back().info));
        e.pcm.assign((const uint8_t*)pcm, (const uint8_t*)pcm + fb);
        outs.push_back(std::move(e));
    }
    void set_volume(int port, uint32_t mask, const int* v) override {
        VolEv e; e.port = port; e.mask = mask;
        for (int c = 0; c < 8; c++) if (mask & (1u << c)) e.vols.push_back(v[c]);
        vols_.push_back(std::move(e));
    }
    void close(int port) override { closes.push_back(port); }
};

// Typed shims over the dispatch table.
static HleFn FN(const char* n) { return Hle::lookup(nid_hash(n)); }
static int64_t call(const char* n, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0,
                    uint64_t a3 = 0, uint64_t a4 = 0, uint64_t a5 = 0) {
    HleFn f = FN(n);
    if (!f) { printf("  [FAIL] not registered: %s\n", n); fails++; return -999; }
    return (int64_t)f(a0, a1, a2, a3, a4, a5);
}
static uint64_t PTR(const void* p) { return (uint64_t)(uintptr_t)p; }

int main() {
    printf("== test_audio ==\n");
    register_builtin_hle();

    // --- 1. format decoding (all 8 SceAudioOutParamFormat values + an unknown) ---------------
    struct { uint32_t param; int ch; AudioFmt fmt; } fmts[] = {
        {0, 1, AudioFmt::S16}, {1, 2, AudioFmt::S16}, {2, 8, AudioFmt::S16}, {3, 1, AudioFmt::F32},
        {4, 2, AudioFmt::F32}, {5, 8, AudioFmt::F32}, {6, 8, AudioFmt::S16}, {7, 8, AudioFmt::F32},
        {0x1234, 2, AudioFmt::S16},                     // unknown low byte 0x34 -> default stereo S16
    };
    for (auto& t : fmts) {
        int ch; AudioFmt f; audio_decode_format(t.param, ch, f);
        CHECK(ch == t.ch); CHECK(f == t.fmt);
    }

    // --- 2. open -> handle + backend.open with decoded params --------------------------------
    audio_reset();
    CapturingSink sink; audio_set_sink(&sink);

    // sceAudioOutOpen(userId=1, type=0(MAIN), index=0, len=256, freq=48000, param=1(S16_STEREO))
    int64_t h = call("sceAudioOutOpen", 1, 0, 0, 256, 48000, 1);
    CHECK(h >= 1);
    CHECK(sink.opens.size() == 1);
    if (!sink.opens.empty()) {
        auto& o = sink.opens[0];
        CHECK(o.port == (int)h);
        CHECK(o.info.freq == 48000); CHECK(o.info.channels == 2);
        CHECK(o.info.fmt == AudioFmt::S16); CHECK(o.info.grain == 256);
        CHECK(audio_grain_bytes(o.info) == 256 * 2 * 2);   // 256 frames * 2ch * 2B
    }

    // --- 3. output forwards the exact grain (frames + bytes) ---------------------------------
    std::vector<uint8_t> pcm(256 * 2 * 2);
    for (size_t i = 0; i < pcm.size(); i++) pcm[i] = (uint8_t)(i * 7 + 3);
    int64_t n = call("sceAudioOutOutput", (uint64_t)h, PTR(pcm.data()));
    CHECK(n == 256);                                      // returns frames written
    CHECK(sink.outs.size() == 1);
    if (!sink.outs.empty()) {
        CHECK(sink.outs[0].port == (int)h);
        CHECK(sink.outs[0].frames == 256);
        CHECK(sink.outs[0].pcm == pcm);                  // byte-for-byte forwarded
    }
    // ptr == 0 is a drain request: returns 0, does NOT forward a grain.
    CHECK(call("sceAudioOutOutput", (uint64_t)h, 0) == 0);
    CHECK(sink.outs.size() == 1);

    // --- 4. set volume forwards mask + values -----------------------------------------------
    int vols[2] = { 20000, 15000 };
    CHECK(call("sceAudioOutSetVolume", (uint64_t)h, 0x3 /*ch0|ch1*/, PTR(vols)) == 0);
    CHECK(sink.vols_.size() == 1);
    if (!sink.vols_.empty()) {
        CHECK(sink.vols_[0].mask == 0x3);
        CHECK(sink.vols_[0].vols.size() == 2);
        CHECK(sink.vols_[0].vols[0] == 20000); CHECK(sink.vols_[0].vols[1] == 15000);
    }

    // --- 5. get port state fills the struct --------------------------------------------------
    uint8_t state[0x20]; memset(state, 0xEE, sizeof state);
    CHECK(call("sceAudioOutGetPortState", (uint64_t)h, PTR(state)) == 0);
    // Layout per Kyty Audio.cpp:340: u16 output @0; u8 channel @2; i16 volume @4; u16 reroute @6; u64 flag @8.
    CHECK(*(uint16_t*)(state + 0) == 1);                  // output enabled
    CHECK(*(uint8_t*)(state + 2) == 2);                   // channel (u8, capped at 2 for main port)
    CHECK(*(int16_t*)(state + 4) == 127);                 // volume (Kyty reports 127)
    CHECK(*(uint64_t*)(state + 8) == 0);                  // flag must NOT carry a bogus volume

    // --- 6. error paths: the real SCE codes (Kyty Errno.h), not a generic -1 -----------------
    CHECK((int32_t)call("sceAudioOutOutput", 999, PTR(pcm.data())) == (int32_t)0x80260003);  // INVALID_PORT
    CHECK((int32_t)call("sceAudioOutSetVolume", 999, 0x1, PTR(vols)) == (int32_t)0x80260003);
    CHECK((int32_t)call("sceAudioOutClose", 999) == (int32_t)0x80260003);
    CHECK(call("sceAudioOutInit", 0) == 0);

    // --- 7. close, then output-after-close fails --------------------------------------------
    CHECK(call("sceAudioOutClose", (uint64_t)h) == 0);
    CHECK(!sink.closes.empty() && sink.closes.back() == (int)h);
    CHECK(call("sceAudioOutOutput", (uint64_t)h, PTR(pcm.data())) < 0);

    // --- 8. exhaustion: 16 ports open, 17th fails; float format decoded on open --------------
    audio_reset(); sink = CapturingSink{}; audio_set_sink(&sink);
    int opened = 0;
    for (int i = 0; i < 16; i++) if (call("sceAudioOutOpen", 1, 0, 0, 512, 44100, 4 /*F32 stereo*/) >= 1) opened++;
    CHECK(opened == 16);
    CHECK((int32_t)call("sceAudioOutOpen", 1, 0, 0, 512, 44100, 4) == (int32_t)0x80260005);  // PORT_FULL
    CHECK(sink.opens.size() == 16);
    if (!sink.opens.empty()) {
        CHECK(sink.opens[0].info.fmt == AudioFmt::F32);
        CHECK(sink.opens[0].info.freq == 44100);
        CHECK(audio_grain_bytes(sink.opens[0].info) == 512 * 2 * 4);
    }

    // --- 9. sceAudioOutOutputs (batch) forwards each valid entry -----------------------------
    audio_reset(); sink = CapturingSink{}; audio_set_sink(&sink);
    int64_t ha = call("sceAudioOutOpen", 1, 0, 0, 128, 48000, 1);
    int64_t hb = call("sceAudioOutOpen", 1, 0, 0, 128, 48000, 1);
    CHECK(ha >= 1 && hb >= 1);
    struct OutParam { int32_t handle; int32_t reserved; uint64_t ptr; } batch[3];
    std::vector<uint8_t> pa(128 * 2 * 2, 0x11), pb(128 * 2 * 2, 0x22);
    batch[0] = { (int32_t)ha, 0, PTR(pa.data()) };
    batch[1] = { 999,          0, PTR(pb.data()) };        // invalid handle -> skipped
    batch[2] = { (int32_t)hb, 0, PTR(pb.data()) };
    int64_t tot = call("sceAudioOutOutputs", PTR(batch), 3);
    CHECK(tot == 128 + 128);                              // two valid entries
    CHECK(sink.outs.size() == 2);

    // --- restore the default sink so we don't dangle a stack pointer -------------------------
    audio_reset();

    if (fails) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }
    printf("== PASS: sceAudioOut HLE (format/open/output/volume/state/close/batch/errors) ==\n");
    return 0;
}
