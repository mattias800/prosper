// test_audio — behavioral tests for the sceAudioOut and NGS2 HLEs in hle_audio.cpp.
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
static int64_t call_raw(const char* nid, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0,
                        uint64_t a3 = 0, uint64_t a4 = 0, uint64_t a5 = 0) {
    HleFn f = Hle::lookup(nid);
    if (!f) { printf("  [FAIL] raw NID not registered: %s\n", nid); fails++; return -999; }
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

    // AudioOut2 shares the guest-store primitive with AJM. Inaccessible outputs must report an
    // error instead of faulting the host; these cover both fixed-size zero-fill and u64 stores.
    CHECK((int32_t)call("sceAudioOut2ContextResetParam", 1) == (int32_t)0x80260003);
    CHECK((int32_t)call("sceAudioOut2ContextQueryMemory", 0, 1) == (int32_t)0x80260003);

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

    // --- 4. sparse volume masks use Sony's channel-indexed array, not compacted values -------
    int vols[8] = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000};
    constexpr uint32_t sparse_mask = (1u << 4) | (1u << 7);
    int cached[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    audio_apply_channel_volumes(cached, sparse_mask, vols);
    CHECK(cached[4] == 5000); CHECK(cached[7] == 8000);
    CHECK(cached[0] == -1); CHECK(cached[6] == -1);
    CHECK(audio_peak_channel_volume(sparse_mask, vols) == 8000);
    CHECK(call("sceAudioOutSetVolume", (uint64_t)h, sparse_mask, PTR(vols)) == 0);
    CHECK(sink.vols_.size() == 1);
    if (!sink.vols_.empty()) {
        CHECK(sink.vols_[0].mask == sparse_mask);
        CHECK(sink.vols_[0].vols.size() == 2);
        CHECK(sink.vols_[0].vols[0] == 5000); CHECK(sink.vols_[0].vols[1] == 8000);
    }

    // --- 5. get port state fills the struct --------------------------------------------------
    uint8_t state[0x20]; memset(state, 0xEE, sizeof state);
    CHECK(call("sceAudioOutGetPortState", (uint64_t)h, PTR(state)) == 0);
    // Layout per Kyty Audio.cpp:340: u16 output @0; u8 channel @2; i16 volume @4; u16 reroute @6; u64 flag @8.
    CHECK(*(uint16_t*)(state + 0) == 1);                  // output enabled
    CHECK(*(uint8_t*)(state + 2) == 2);                   // channel (u8, capped at 2 for main port)
    CHECK(*(int16_t*)(state + 4) == 127);                 // volume (Kyty reports 127)
    CHECK(*(uint64_t*)(state + 8) == 0);                  // flag must NOT carry a bogus volume

    // Port routing is independent of the requested PCM channel count. Voice and Personal report
    // a mono headphone route, Aux reports the external route, and PadSpk reports its mono route.
    struct PortRoute { uint32_t type; uint16_t output; uint8_t channel; } routes[] = {
        {1, 0x01, 2}, {2, 0x40, 1}, {3, 0x40, 1}, {4, 0x04, 1}, {127, 0x80, 0},
    };
    for (const auto& route : routes) {
        int64_t route_h = call("sceAudioOutOpen", 1, route.type, 0, 256, 48000, 2 /* S16 8ch */);
        CHECK(route_h >= 1);
        uint8_t route_state[0x20]; memset(route_state, 0xEE, sizeof route_state);
        CHECK(call("sceAudioOutGetPortState", (uint64_t)route_h, PTR(route_state)) == 0);
        CHECK(*(uint16_t*)(route_state + 0) == route.output);
        CHECK(route_state[2] == route.channel);
        CHECK(call("sceAudioOutClose", (uint64_t)route_h) == 0);
    }

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
    CHECK(tot == 128);                                    // ONE grain (the shared slice), not the sum over
                                                          // ports (Kyty/shadPS4 both return a single grain)
    CHECK(sink.outs.size() == 2);                         // ...but both valid entries are still forwarded

    // --- 10. libSceNgs2 silent lifecycle: sizes, handles, state, and render output -------------
    struct BufferInfo { uint64_t host_buffer, host_buffer_size, reserved[5], user_data; };
    struct RackOption {
        uint64_t size; char name[16]; uint32_t flags, max_grain, max_voices,
            max_delay, max_matrices, max_ports, reserved[20];
    };
    struct RenderInfo { uint64_t buffer, size; uint32_t waveform_type, channels; };

    BufferInfo sys_info; memset(&sys_info, 0xEE, sizeof sys_info);
    CHECK(call_raw("pgFAiLR5qT4", 0, PTR(&sys_info)) == 0); // SystemQueryBufferSize
    CHECK(sys_info.host_buffer == 0);
    CHECK(sys_info.host_buffer_size == 0x1000);
    CHECK(sys_info.reserved[0] == 0 && sys_info.user_data == 0);
    std::vector<uint8_t> sys_work(sys_info.host_buffer_size, 0);
    sys_info.host_buffer = PTR(sys_work.data());
    uint64_t system = 0xDEADBEEFDEADBEEFull;
    CHECK(call_raw("koBbCMvOKWw", 0, PTR(&sys_info), PTR(&system)) == 0); // SystemCreate
    CHECK(system != 0 && system != 0xDEADBEEFDEADBEEFull);

    RackOption rack_opt{}; rack_opt.size = sizeof rack_opt; rack_opt.max_voices = 2;
    memcpy(rack_opt.name, "test", 5);
    BufferInfo rack_info; memset(&rack_info, 0xDD, sizeof rack_info);
    CHECK(call_raw("0eFLVCfWVds", 0x2001, PTR(&rack_opt), PTR(&rack_info)) == 0);
    CHECK(rack_info.host_buffer == 0 && rack_info.host_buffer_size >= 0x1000);
    std::vector<uint8_t> rack_work(rack_info.host_buffer_size, 0);
    rack_info.host_buffer = PTR(rack_work.data());
    uint64_t rack = 0xDEADBEEFDEADBEEFull;
    CHECK(call_raw("cLV4aiT9JpA", system, 0x2001, PTR(&rack_opt), PTR(&rack_info), PTR(&rack)) == 0);
    CHECK(rack != 0 && rack != 0xDEADBEEFDEADBEEFull);

    uint64_t voice = 0xDEADBEEFDEADBEEFull;
    CHECK(call_raw("MwmHz8pAdAo", rack, 1, PTR(&voice)) == 0);
    CHECK(voice != 0 && voice != 0xDEADBEEFDEADBEEFull);
    uint64_t invalid_voice = 0xDEADBEEFDEADBEEFull;
    CHECK((int32_t)call_raw("MwmHz8pAdAo", rack, 2, PTR(&invalid_voice)) == (int32_t)0x804A0302);
    CHECK(invalid_voice == 0xDEADBEEFDEADBEEFull);

    uint8_t voice_state[0x30]; memset(voice_state, 0xCC, sizeof voice_state);
    CHECK(call_raw("-TOuuAQ-buE", voice, PTR(voice_state), sizeof voice_state) == 0);
    for (uint8_t b : voice_state) CHECK(b == 0);           // inert voice = Empty

    uint8_t ngs_pcm[256]; memset(ngs_pcm, 0xA5, sizeof ngs_pcm);
    RenderInfo render{PTR(ngs_pcm), sizeof ngs_pcm, 0, 2};
    CHECK(call_raw("i0VnXM-C9fc", system, PTR(&render), 1) == 0);
    for (uint8_t b : ngs_pcm) CHECK(b == 0);               // silent backend produces silence
    CHECK((int32_t)call_raw("i0VnXM-C9fc", 0, PTR(&render), 1) == (int32_t)0x804A0230);

    uint8_t source[0xA8], listener[0xA0], listener_work[0x60];
    memset(source, 0xBB, sizeof source); memset(listener, 0xBB, sizeof listener);
    memset(listener_work, 0xBB, sizeof listener_work);
    CHECK(call_raw("0lbbayqDNoE", PTR(source)) == 0);
    CHECK(call_raw("7Lcfo8SmpsU", PTR(listener)) == 0);
    CHECK(call_raw("1WsleK-MTkE", PTR(listener), PTR(listener_work), 0) == 0);
    for (uint8_t b : source) CHECK(b == 0);
    for (uint8_t b : listener) CHECK(b == 0);
    for (uint8_t b : listener_work) CHECK(b == 0);

    // --- restore the default sink so we don't dangle a stack pointer -------------------------
    audio_reset();

    if (fails) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }
    printf("== PASS: sceAudioOut + NGS2 HLE lifecycle/output/error contracts ==\n");
    return 0;
}
