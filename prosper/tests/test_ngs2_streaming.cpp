// test_ngs2_streaming — #1060 part 3. Guards the #1059 NGS2 waveform-block streaming that gives Dead
// Cells its music: a fed waveform block must make the voice report Playing and advance
// num_decoded_samples across SystemRender calls. num_decoded_samples is the stream clock the guest's
// block-completion callback reads to pace its feed loop; if it never advances, the stream stalls after
// the first ring fill and no music is produced (the exact pre-#1059 defect).
//
// The existing test_audio only pins the STATIC un-fed VoiceGetState fields; this feeds a real block and
// drives playback. The block queuing/mixing is cross-platform (only the guest-callback firing is
// Linux-only), but the block pointer must clear the `>= 0x200000000` guest-address guard in
// ngs2_apply_voice_params, so the streaming section maps a PCM buffer there with mmap and soft-skips if
// that fixed mapping isn't available (e.g. Windows/MinGW) — the lifecycle assertions still run there.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <thread>
#include <chrono>
#include <vector>
#if !defined(_WIN32)
#include <sys/mman.h>
#endif

#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); fails++; } \
                              else          { std::printf("  [ok]   %s\n", msg); } } while (0)

static int64_t call_raw(const char* nid, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0,
                        uint64_t a3 = 0, uint64_t a4 = 0, uint64_t a5 = 0) {
    HleFn f = Hle::lookup(nid);
    return f ? (int64_t)(int32_t)f(a0, a1, a2, a3, a4, a5) : (int64_t)0x7fffffff;
}
static uint64_t PTR(const void* p) { return (uint64_t)(uintptr_t)p; }

struct BufferInfo { uint64_t host_buffer, host_buffer_size, reserved[5], user_data; };
struct RackOption { uint64_t size; char name[16]; uint32_t flags, max_grain, max_voices,
                    max_delay, max_matrices, max_ports, reserved[20]; };
struct RenderInfo { uint64_t buffer, size; uint32_t waveform_type, channels; };

int main() {
    std::printf("== test_ngs2_streaming (#1060: fed block -> Playing + advancing num_decoded_samples) ==\n");
    register_builtin_hle();

    // ---- NGS2 lifecycle: system -> rack -> voice (mirrors test_audio section 11) ----------------
    BufferInfo sys_info{}; std::memset(&sys_info, 0xEE, sizeof sys_info);
    CHECK(call_raw("pgFAiLR5qT4", 0, PTR(&sys_info)) == 0, "SystemQueryBufferSize");
    std::vector<uint8_t> sys_work(sys_info.host_buffer_size ? sys_info.host_buffer_size : 0x1000, 0);
    sys_info.host_buffer = PTR(sys_work.data());
    uint64_t system = 0;
    CHECK(call_raw("koBbCMvOKWw", 0, PTR(&sys_info), PTR(&system)) == 0 && system != 0, "SystemCreate");

    RackOption rack_opt{}; rack_opt.size = sizeof rack_opt; rack_opt.max_voices = 2;
    std::memcpy(rack_opt.name, "strm", 5);
    BufferInfo rack_info{}; std::memset(&rack_info, 0xDD, sizeof rack_info);
    CHECK(call_raw("0eFLVCfWVds", 0x2001, PTR(&rack_opt), PTR(&rack_info)) == 0, "RackQueryBufferSize");
    std::vector<uint8_t> rack_work(rack_info.host_buffer_size ? rack_info.host_buffer_size : 0x1000, 0);
    rack_info.host_buffer = PTR(rack_work.data());
    uint64_t rack = 0;
    CHECK(call_raw("cLV4aiT9JpA", system, 0x2001, PTR(&rack_opt), PTR(&rack_info), PTR(&rack)) == 0 && rack,
          "RackCreate");
    uint64_t voice = 0;
    CHECK(call_raw("MwmHz8pAdAo", rack, 1, PTR(&voice)) == 0 && voice, "RackGetVoiceHandle");

    auto voice_num_decoded = [&](uint32_t* out_flags) -> uint64_t {
        uint8_t st[0x30]; std::memset(st, 0xCC, sizeof st);
        if (call_raw("-TOuuAQ-buE", voice, PTR(st), sizeof st) != 0) return ~0ull;
        if (out_flags) *out_flags = *(uint32_t*)(st + 0x00);
        return *(uint64_t*)(st + 0x10);
    };

    // Un-fed voice: Empty, zero decoded (baseline; also covered by test_audio).
    uint32_t flags0 = 99;
    CHECK(voice_num_decoded(&flags0) == 0 && flags0 == 0, "un-fed voice: Empty, num_decoded=0");

    // ---- Streaming: feed a waveform block, render, and watch num_decoded_samples advance ---------
#if !defined(_WIN32)
    const uint64_t kBlockVA = 0x200000000ull;
    const size_t   kBlockBytes = 0x10000;   // 64 KiB PCM: ample headroom past the render/wall-clock span
    void* pcm = mmap((void*)kBlockVA, kBlockBytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (pcm != MAP_FAILED && (uint64_t)(uintptr_t)pcm == kBlockVA) {
        for (size_t i = 0; i < kBlockBytes / 2; i++) ((int16_t*)pcm)[i] = (int16_t)((i * 37) & 0x3fff);

        // Param chain: {u16 size; i16 next; u32 id; payload...}. WaveformFormat sets rate/channels and
        // resets the stream; WaveformBlock enqueues the block pointer (read at param+8).
        uint8_t chain[64] = {};
        *(uint16_t*)(chain + 0)  = 20; *(int16_t*)(chain + 2) = 20; *(uint32_t*)(chain + 4) = 0x40010000u;
        *(uint32_t*)(chain + 8)  = 0x12;    // waveform_type = PcmS16
        *(uint32_t*)(chain + 12) = 2;       // channels
        *(uint32_t*)(chain + 16) = 48000;   // sample rate
        *(uint16_t*)(chain + 20) = 16; *(int16_t*)(chain + 22) = 0; *(uint32_t*)(chain + 24) = 0x40010001u;
        *(uint64_t*)(chain + 28) = kBlockVA;   // WaveformBlock pointer (>= 0x200000000)

        CHECK(call_raw("uu94irFOGpA", voice, PTR(chain)) == 0, "VoiceControl feed (format + block)");

        uint8_t out[512];
        RenderInfo render{PTR(out), sizeof out, 0, 2};
        // First render starts the stream clock; then advance wall-clock and re-render.
        CHECK(call_raw("i0VnXM-C9fc", system, PTR(&render), 1) == 0, "SystemRender after feed");
        uint32_t flags1 = 0;
        voice_num_decoded(&flags1);
        CHECK(flags1 == 0x3u, "fed voice reports Playing (state_flags 0x3)");

        uint64_t prev = 0; bool monotonic = true, advanced = false;
        for (int i = 0; i < 12 && !advanced; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            call_raw("i0VnXM-C9fc", system, PTR(&render), 1);   // mixer paces played_samples to wall-clock
            uint64_t n = voice_num_decoded(nullptr);
            if (n < prev) monotonic = false;
            prev = n;
            if (n > 0) advanced = true;
        }
        CHECK(monotonic, "num_decoded_samples is monotonically non-decreasing across renders");
        CHECK(advanced, "num_decoded_samples advances past 0 once a block is fed (stream is consumed)");

        munmap(pcm, kBlockBytes);
    } else {
        std::printf("  [skip] MAP_FIXED @0x%llx unavailable — streaming assertions skipped here\n",
                    (unsigned long long)kBlockVA);
        if (pcm != MAP_FAILED) munmap(pcm, kBlockBytes);
    }
#else
    std::printf("  [skip] Windows: fixed high-address mapping not set up; streaming validated on POSIX CI\n");
#endif

    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
