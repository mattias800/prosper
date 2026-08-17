// test_videodec2_decode (#2270) — sceVideodec2Decode must deliver a decoded picture to the guest.
//
// The defect this pins is the FALSE SUCCESS family (#2081): before the fix, sceVideodec2Decode
// validated its arguments thoroughly, then reported SCE_OK with `pictures = 0` on EVERY call, for
// every title, forever. That return is the CORRECT answer for "the decoder needs more input", which
// is exactly what makes it catastrophic as the only answer — a title feeding access units cannot
// distinguish it from a decoder that is about to produce a frame, so it waits for one that cannot
// arrive, with nothing in the log.
//
// Three arms, selected by argv so each runs in its own process (each arm's environment variable is
// read once into a function-local static, which is deliberate: this is a per-frame path):
//
//   (default)        a backend that offers the access-unit path is USED. The guest's frame buffer
//                    receives the NV12 picture and VdecOutput reports valid/pictures/dimensions.
//   --no-decode      PROSPER_VDEC2_NO_DECODE=1 restores the old no-picture behaviour, so the A/B
//                    that justified turning decoding on by default stays runnable.
//   --format-probe   PROSPER_VDEC2_FORMAT reaches the guest's struct — see kFormatProbe for why a
//                    sweep instrument needs its own positive control.
//
// WHAT THIS TEST DOES AND DOES NOT PROVE. The backend here is a fake that returns a synthetic NV12
// picture, so this is a test of the HLE PLUMBING — that the handler opens a decoder, submits the
// guest's access unit, copies the returned planes into the guest's own buffer, and fills the output
// struct. It deliberately says nothing about whether a real bitstream decodes correctly; that is
// test_video_vaapi's AU-decode section, which checks a real H.264 elementary stream against frame
// hashes taken from an independent decoder. Splitting them that way is the point: a fake picture
// cannot accidentally pass the decode test, and a broken decoder cannot accidentally pass this one.

#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/video_backend.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "test_scratch.h"
#if !defined(_WIN32)
#include <unistd.h>     // dup/dup2/close — the stderr capture in the banner-count arm
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

// The guest-facing structs, mirrored from the handler's own definitions. static_assert on each size
// so a silent divergence makes this file fail to compile rather than exercise a different ABI.
struct VdecConfig {
    uint64_t size; uint32_t resource, codec, profile, max_level;
    int32_t max_width, max_height, max_dpb; uint32_t input_depth;
    uint64_t compute_queue, affinity; int32_t priority;
    uint8_t optimize, check_memory, reserved0, reserved1; uint64_t extra;
};
struct VdecInput { uint64_t size, data, data_size, pts, dts, attached; };
struct VdecFrame { uint64_t size, data, data_size; uint8_t accepted, pad[7]; };
struct VdecOutput {
    uint64_t size; uint8_t valid, error, pictures, discarded;
    uint32_t codec, width, pitch, height; uint64_t frame, frame_size;
    uint32_t format, pitch_bytes;
};
static_assert(sizeof(VdecConfig) == 72, "test VdecConfig must mirror the HLE layout");
static_assert(sizeof(VdecInput) == 48, "test VdecInput must mirror the HLE layout");
static_assert(sizeof(VdecFrame) == 32, "test VdecFrame must mirror the HLE layout");
static_assert(sizeof(VdecOutput) == 56, "test VdecOutput must mirror the HLE layout");

// A backend that implements ONLY the access-unit entry points. Every stream method aborts the test
// if reached: routing Videodec2 through the AvPlayer stream path would be a different bug wearing
// this one's clothes, and it must not pass quietly.
class AuOnlyBackend final : public video::VideoBackend {
public:
    static constexpr uint32_t kWidth = 64;
    static constexpr uint32_t kHeight = 32;

    int  open(const std::string&) override { unreachable("open"); return -1; }
    bool info(int, video::StreamInfo&) override { unreachable("info"); return false; }
    bool next_video(int, video::VideoFrame&) override { unreachable("next_video"); return false; }
    bool next_audio(int, video::AudioFrame&) override { unreachable("next_audio"); return false; }
    bool eof(int) override { unreachable("eof"); return true; }
    void close(int) override { unreachable("close"); }

    int open_decoder(uint32_t codec) override {
        ++opens; last_codec = codec;
        if (refuse_open) return -1;
        // A DISTINCT id per open, so "the decoder was re-opened" is observable rather than assumed.
        return next_id++;
    }
    AuResult decode_au(int id, const uint8_t* au, size_t bytes,
                       uint8_t* dst, uint64_t dst_bytes, AuPicture& out) override {
        ++decodes; last_id = id; last_au_bytes = bytes;
        last_au_first = (au && bytes) ? au[0] : 0;
        if (starve) return AuResult::NoPicture;   // "no picture yet" — legitimate, not an error
        // ZERO-SIZE arm (#2571 N8). A backend that claims a decode over a 0x0 picture must not be
        // able to mint a reported picture out of it. Only a fake backend can produce this, which is
        // exactly why the guard needs a fake backend to test it.
        if (zero_size) { out = AuPicture{}; return AuResult::Decoded; }
        out.width = kWidth; out.height = kHeight;
        out.y_stride = kWidth; out.uv_stride = kWidth;
        const uint64_t y_bytes = static_cast<uint64_t>(kWidth) * kHeight;
        out.nv12_bytes = y_bytes + y_bytes / 2;
        if (!dst || dst_bytes < out.nv12_bytes) return AuResult::FrameTooSmall;
        // A synthetic picture whose bytes are a function of the SUBMITTED access unit, not a
        // constant: a handler that copied the wrong buffer, or copied nothing, still produces a
        // plausible-looking frame if the pattern is fixed, and this test would not notice.
        for (uint64_t i = 0; i < y_bytes; ++i)
            dst[i] = static_cast<uint8_t>((last_au_first + i) & 0xFF);
        for (uint64_t i = 0; i < y_bytes / 2; ++i)
            dst[y_bytes + i] = static_cast<uint8_t>((last_au_first * 3u + i * 7u) & 0xFF);
        return AuResult::Decoded;
    }
    void close_decoder(int id) override { ++closes; last_closed = id; }

    bool refuse_open = false;
    bool starve = false;
    bool zero_size = false;
    int  next_id = 7;
    int  opens = 0, decodes = 0, closes = 0;
    int  last_id = -1, last_closed = -1;
    uint32_t last_codec = 0;
    size_t last_au_bytes = 0;
    uint8_t last_au_first = 0;

private:
    static void unreachable(const char* what) {
        printf("  [FAIL] Videodec2 must not use the AvPlayer stream path (%s)\n", what);
        ++fails;
    }
};

}  // namespace

// PROSPER_VDEC2_FORMAT's positive control, and it exists because of what it is used FOR. `format` is
// the one VdecOutput field prosper cannot derive, so the variable is how a candidate value gets
// swept against a live title without a rebuild. A sweep that reports "no title behaviour changed"
// is worthless unless the lever provably moved — otherwise "the guest ignores this field" and "the
// override never reached the guest" are the same observation. This arm establishes only the second
// half: the value the variable names is what lands in the guest's struct. It says nothing about
// whether a guest reads it, which is the open question.
constexpr uint32_t kFormatProbe = 0x2a;

// _putenv rather than _putenv_s on Windows: MinGW-w64 declares the _s form only for a new enough
// CRT, and this test builds on every platform in CI.
void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    const std::string assignment = std::string(name) + "=" + value;
    _putenv(assignment.c_str());
#else
    setenv(name, value, 1);
#endif
}

// THE ODD-DIMENSION ARM (#2571 review D1). The exact NV12 size is the sharpest correction in this
// change and, until this arm existed, **nothing in the suite could see it**: every dimension anywhere
// near this code is even (128x96, 1920x1088, 3840x2160), so reinstating the `w*h*3/2` shorthand
// passed green. A correct change with no test defending it is one a future tidy-up reverts.
//
// Only ONE dimension needs to be odd. That is the half of the rule that is easy to state wrongly —
// "even product" is the seductive wrong version, and 1920x1081 has an even product while the
// shorthand is 960 bytes short there.
void check_nv12_sizes() {
    struct Case { uint32_t w, h; uint64_t exact; const char* what; };
    // Hand-computed, not produced by the function under test — a table generated from the
    // implementation would agree with any implementation, including a wrong one.
    static const Case cases[] = {
        // w    h     exact                                    shorthand   short by
        {  64,  32,  64ull * 32 + 2 * 32 * 16,  "even x even — the shorthand agrees here" },
        {  65,  32,  65ull * 32 + 2 * 33 * 16,  "ODD width  (3136 vs 3120, short by 16)"  },
        {  64,  33,  64ull * 33 + 2 * 32 * 17,  "ODD height (3200 vs 3168, short by 32)"  },
        {  65,  33,  65ull * 33 + 2 * 33 * 17,  "BOTH odd   (3267 vs 3217, short by 50)"  },
        { 1920, 1081, 1920ull * 1081 + 2 * 960 * 541,
          "1920x1081 — EVEN PRODUCT, and the shorthand is still 960 short" },
        { 1920, 1088, 1920ull * 1088 + 2 * 960 * 544, "the live PPSA19991 geometry" },
        { 3840, 2160, 3840ull * 2160 + 2 * 1920 * 1080, "the live PPSA05325 geometry" },
    };
    for (const Case& c : cases) {
        const uint64_t got = video::nv12_bytes(c.w, c.h);
        char msg[192];
        std::snprintf(msg, sizeof msg, "nv12_bytes(%u,%u) == %llu — %s", c.w, c.h,
                      (unsigned long long)c.exact, c.what);
        CHECK(got == c.exact, msg);
        // The discriminator, stated as its own assertion so a reader can see WHICH cases the
        // shorthand would have passed: it must be short exactly when a dimension is odd.
        const uint64_t shorthand = (uint64_t)c.w * c.h * 3 / 2;
        const bool any_odd = (c.w & 1u) || (c.h & 1u);
        std::snprintf(msg, sizeof msg,
                      "%ux%u: the w*h*3/2 shorthand is %s here (%llu vs %llu)", c.w, c.h,
                      any_odd ? "SHORT, which is why it is not used" : "equal",
                      (unsigned long long)shorthand, (unsigned long long)c.exact);
        CHECK(any_odd ? (shorthand < c.exact) : (shorthand == c.exact), msg);
    }
    // Widening: `2*((w+1)/2)` on a 32-bit `w` at UINT32_MAX would wrap to 0 before the multiply.
    CHECK(video::nv12_bytes(0xFFFFFFFFu, 2) == 0xFFFFFFFFull * 2 + 2 * 0x80000000ull * 1,
          "nv12_bytes widens before arithmetic — no 32-bit wrap at the top of the range");
}

int main(int argc, char** argv) {
    const std::string arg = argc > 1 ? argv[1] : "";
    const bool no_decode = arg == "--no-decode";
    const bool format_probe = arg == "--format-probe";
    if (no_decode) set_env("PROSPER_VDEC2_NO_DECODE", "1");
    if (format_probe) set_env("PROSPER_VDEC2_FORMAT", "42");   // decimal 42 == kFormatProbe
    printf("== test_videodec2_decode (%s) ==\n",
           no_decode ? "opt-out arm" : format_probe ? "format-override arm" : "decode arm");
    check_nv12_sizes();
    register_builtin_hle();

    auto alloc_queue    = Hle::lookup("eD+X2SmxUt4");
    auto query_compute  = Hle::lookup("RnDibcGCPKw");
    auto query_decoder  = Hle::lookup("qqMCwlULR+E");
    auto create         = Hle::lookup("CNNRoRYd8XI");
    auto decode         = Hle::lookup("852F5+q6+iM");
    auto reset          = Hle::lookup("wJXikG6QFN8");
    auto destroy        = Hle::lookup("jwImxXRGSKA");
    CHECK(alloc_queue && query_compute && query_decoder && create && decode && reset && destroy,
          "Videodec2 lifecycle NIDs resolve");
    if (!(alloc_queue && query_compute && query_decoder && create && decode && reset && destroy))
        return 1;

    AuOnlyBackend fake;
    video::set_backend(&fake);

    // ---- the guest's own lifecycle, in the order a title performs it ---------------------------
    alignas(256) static uint8_t workspace[4][64u << 10];
    uint64_t compute_info[3] = {24, 0, 0};
    CHECK(query_compute((uint64_t)(uintptr_t)compute_info, 0, 0, 0, 0, 0) == 0,
          "QueryComputeMemoryInfo succeeds");
    compute_info[2] = (uint64_t)(uintptr_t)workspace[3];
    struct { uint64_t size; uint16_t pipe, queue; uint8_t check, r0; uint16_t r1; } cq{16, 0, 0, 0, 0, 0};
    uint64_t compute_queue = 0;
    CHECK(alloc_queue((uint64_t)(uintptr_t)&cq, (uint64_t)(uintptr_t)compute_info,
                      (uint64_t)(uintptr_t)&compute_queue, 0, 0, 0) == 0 && compute_queue,
          "AllocateComputeQueue publishes a queue");

    // codec=1 / profile=100 / 1920x1088 is the live contract captured from Tales of Graces f
    // (PPSA19991) in #2281. The dimensions here are the fake picture's, so the guest frame buffer
    // sized from them is the one a real decode would need.
    VdecConfig config{};
    config.size = sizeof config;
    config.resource = 1; config.codec = 1; config.profile = 100; config.max_level = 41;
    config.max_width = AuOnlyBackend::kWidth; config.max_height = AuOnlyBackend::kHeight;
    config.max_dpb = -1; config.input_depth = 4; config.compute_queue = compute_queue;

    uint64_t decoder_memory[9] = {72};
    CHECK(query_decoder((uint64_t)(uintptr_t)&config, (uint64_t)(uintptr_t)decoder_memory,
                        0, 0, 0, 0) == 0,
          "QueryDecoderMemoryInfo succeeds");
    decoder_memory[2] = (uint64_t)(uintptr_t)workspace[0];
    decoder_memory[4] = (uint64_t)(uintptr_t)workspace[1];
    decoder_memory[6] = (uint64_t)(uintptr_t)workspace[2];
    uint64_t handle = 0;
    CHECK(create((uint64_t)(uintptr_t)&config, (uint64_t)(uintptr_t)decoder_memory,
                 (uint64_t)(uintptr_t)&handle, 0, 0, 0) == 0 && handle,
          "CreateDecoder publishes a handle");

    // A guest access unit with a recognisable first byte, and a guest frame buffer big enough for
    // the NV12 picture the config describes.
    const size_t y_bytes = static_cast<size_t>(AuOnlyBackend::kWidth) * AuOnlyBackend::kHeight;
    const size_t nv12_bytes = y_bytes + y_bytes / 2;
    std::vector<uint8_t> au(573);          // a per-frame NAL size in the live range (565..3984)
    au[0] = 0xA7;
    for (size_t i = 1; i < au.size(); ++i) au[i] = static_cast<uint8_t>(i);
    std::vector<uint8_t> guest_frame(nv12_bytes + 4096, 0xCD);

    VdecInput input{sizeof(VdecInput), (uint64_t)(uintptr_t)au.data(), au.size(), 0, 0, 0};
    VdecFrame frame{sizeof(VdecFrame), (uint64_t)(uintptr_t)guest_frame.data(), nv12_bytes, 0, {}};
    VdecOutput out{}; out.size = sizeof out;

    const uint64_t rc = decode(handle, (uint64_t)(uintptr_t)&input, (uint64_t)(uintptr_t)&frame,
                               (uint64_t)(uintptr_t)&out, 0, 0);
    CHECK(rc == 0, "Decode returns SCE_OK");

    if (no_decode) {
        // The opt-out must reach the backend NOT AT ALL — "decoded but discarded" would satisfy a
        // check on `out` alone while still paying for the decode, and would not be the old
        // behaviour this arm exists to preserve.
        CHECK(fake.opens == 0 && fake.decodes == 0,
              "PROSPER_VDEC2_NO_DECODE bypasses the backend entirely");
        CHECK(out.valid == 0 && out.pictures == 0 && frame.accepted == 0,
              "PROSPER_VDEC2_NO_DECODE restores the no-picture answer");
        CHECK(out.codec == config.codec && out.frame == frame.data && out.frame_size == nv12_bytes,
              "the no-picture answer still fills the output struct it always filled");
    } else {
        CHECK(fake.opens == 1 && fake.last_codec == config.codec,
              "Decode opens exactly one backend decoder, for the codec the guest configured");
        CHECK(fake.decodes == 1 && fake.last_au_bytes == au.size() && fake.last_au_first == 0xA7,
              "Decode submits the guest's own access unit, unmodified");
        // THE REGRESSION. Before #2270's fix these four were 0/0/0/0 on every call, forever.
        CHECK(frame.accepted == 1, "the access unit is reported ACCEPTED");
        CHECK(out.valid == 1 && out.pictures == 1 && out.error == 0 && out.discarded == 0,
              "Decode reports ONE VALID PICTURE, not SCE_OK with nothing (#2270)");
        CHECK(out.width == AuOnlyBackend::kWidth && out.height == AuOnlyBackend::kHeight &&
                  out.pitch == AuOnlyBackend::kWidth && out.pitch_bytes == AuOnlyBackend::kWidth,
              "Decode publishes the decoded picture's dimensions and pitch");
        CHECK(out.codec == config.codec && out.frame == frame.data && out.frame_size == nv12_bytes,
              "Decode publishes the guest's own frame buffer as the picture's location");
        // `format` is unestablished (#2270) and defaults to 0 rather than to a plausible constant.
        // The override is the sweep instrument; both halves are pinned so a sweep's result can be
        // read at all — see kFormatProbe.
        CHECK(out.format == (format_probe ? kFormatProbe : 0u),
              format_probe ? "PROSPER_VDEC2_FORMAT reaches the guest's output struct"
                           : "the unestablished output format defaults to 0, not to a guess");
        // The pixels, not just the flags. A handler that set the flags and copied nothing would
        // pass every check above — that is precisely the false-success shape in a new costume.
        bool luma_ok = true, chroma_ok = true;
        for (size_t i = 0; i < y_bytes; ++i)
            if (guest_frame[i] != static_cast<uint8_t>((0xA7 + i) & 0xFF)) { luma_ok = false; break; }
        for (size_t i = 0; i < y_bytes / 2; ++i)
            if (guest_frame[y_bytes + i] != static_cast<uint8_t>((0xA7 * 3u + i * 7u) & 0xFF)) {
                chroma_ok = false; break;
            }
        CHECK(luma_ok, "the decoded LUMA plane is written into the guest's frame buffer");
        CHECK(chroma_ok, "the decoded interleaved CHROMA plane follows it, at the NV12 offset");
        bool tail_clean = true;
        for (size_t i = nv12_bytes; i < guest_frame.size(); ++i)
            if (guest_frame[i] != 0xCD) { tail_clean = false; break; }
        CHECK(tail_clean, "nothing is written past the NV12 picture the guest sized for");

        // A frame buffer smaller than the picture must NOT be overrun. This is the hazard the
        // sizing query's own comment predicted when a real decoder landed.
        std::vector<uint8_t> tiny(nv12_bytes - 1, 0xEE);
        VdecFrame small{sizeof(VdecFrame), (uint64_t)(uintptr_t)tiny.data(), tiny.size(), 0, {}};
        VdecOutput small_out{}; small_out.size = sizeof small_out;
        CHECK(decode(handle, (uint64_t)(uintptr_t)&input, (uint64_t)(uintptr_t)&small,
                     (uint64_t)(uintptr_t)&small_out, 0, 0) == 0 &&
                  small_out.pictures == 0 && small.accepted == 0,
              "a too-small guest frame buffer reports NO PICTURE instead of overrunning it");
        bool untouched = true;
        for (uint8_t b : tiny) if (b != 0xEE) { untouched = false; break; }
        CHECK(untouched, "the too-small buffer is left entirely unwritten");

        // A ZERO-SIZE picture must not become a reported one (#2571 review N8). `w*h*3/2` is 0 for a
        // 0x0 frame, so a size check alone passes it, copies nothing, and answers valid=1 —
        // a confident "here is your picture" over nothing, which is #2270's own shape re-created
        // inside the success path this PR adds. Only a backend bug can produce it, so only a fake
        // backend can test it.
        {
            fake.zero_size = true;
            std::vector<uint8_t> buf(nv12_bytes, 0x11);
            VdecFrame zf{sizeof(VdecFrame), (uint64_t)(uintptr_t)buf.data(), nv12_bytes, 1, {}};
            VdecOutput zo{}; zo.size = sizeof zo;
            const uint64_t zrc = decode(handle, (uint64_t)(uintptr_t)&input,
                                        (uint64_t)(uintptr_t)&zf, (uint64_t)(uintptr_t)&zo, 0, 0);
            CHECK(zrc == 0 && zo.valid == 0 && zo.pictures == 0 && zf.accepted == 0,
                  "a 0x0 'decoded' picture is REFUSED, not reported as a picture (#2571 N8)");
            CHECK(zo.width == 0 && zo.height == 0,
                  "the refused zero-size picture publishes no dimensions");
            bool zero_untouched = true;
            for (uint8_t b : buf) if (b != 0x11) { zero_untouched = false; break; }
            CHECK(zero_untouched, "the guest frame buffer is untouched by a refused zero-size picture");
            fake.zero_size = false;
        }

        // Reset means "forget every decoded reference". Continuing to feed the same libavcodec
        // context would decode the guest's next access units against references it just discarded,
        // which yields a corrupt picture rather than an error.
        const int before = fake.closes;
        CHECK(reset(handle, 0, 0, 0, 0, 0) == 0, "Reset succeeds");
        CHECK(fake.closes == before + 1, "Reset drops the backend decoder rather than reusing it");
        VdecFrame again{sizeof(VdecFrame), (uint64_t)(uintptr_t)guest_frame.data(), nv12_bytes, 0, {}};
        VdecOutput again_out{}; again_out.size = sizeof again_out;
        CHECK(decode(handle, (uint64_t)(uintptr_t)&input, (uint64_t)(uintptr_t)&again,
                     (uint64_t)(uintptr_t)&again_out, 0, 0) == 0 && again_out.pictures == 1,
              "decoding resumes after Reset");
        CHECK(fake.opens == 2, "the access unit after a Reset goes to a FRESH backend decoder");
    }

    // A backend that refuses the codec must not be mistaken for one that is warming up. The
    // handler's own announcement covers the log; what is asserted here is that it does not retry
    // the open on every access unit (which would hide the refusal in a flood) and does not report
    // a picture it does not have.
    uint64_t refused = 0;
    fake.refuse_open = true;
    VdecConfig other = config; other.codec = 0xBADC0DE;
    uint64_t other_memory[9] = {72};
    CHECK(query_decoder((uint64_t)(uintptr_t)&other, (uint64_t)(uintptr_t)other_memory,
                        0, 0, 0, 0) == 0, "QueryDecoderMemoryInfo succeeds for the refused codec");
    other_memory[2] = (uint64_t)(uintptr_t)workspace[0];
    other_memory[4] = (uint64_t)(uintptr_t)workspace[1];
    other_memory[6] = (uint64_t)(uintptr_t)workspace[2];
    if (create((uint64_t)(uintptr_t)&other, (uint64_t)(uintptr_t)other_memory,
               (uint64_t)(uintptr_t)&refused, 0, 0, 0) == 0 && refused) {
        const int opens_before = fake.opens;
        VdecFrame f2{sizeof(VdecFrame), (uint64_t)(uintptr_t)guest_frame.data(), nv12_bytes, 0, {}};
        VdecOutput o2{}; o2.size = sizeof o2;
        for (int i = 0; i < 3; ++i) {
            f2.accepted = 1;
            CHECK(decode(refused, (uint64_t)(uintptr_t)&input, (uint64_t)(uintptr_t)&f2,
                         (uint64_t)(uintptr_t)&o2, 0, 0) == 0 && o2.pictures == 0 && !f2.accepted,
                  "a refused codec reports no picture rather than a fabricated one");
        }
        CHECK(fake.opens == opens_before + (no_decode ? 0 : 1),
              "a refused codec is opened ONCE, so the refusal is announced once and not per unit");

        // THE BANNER ITSELF, counted (#2571 review N3). Everything above measures `open_decoder`
        // calls, and Reset deliberately re-arms those — so a refused codec in a guest retry loop
        // re-enters the open and could re-print the two-line NO DECODER banner on every cycle,
        // turning the fail-visible path into a flood. A flooded diagnostic is a muted one. Nothing
        // observable through the HLE distinguishes "announced once" from "announced every cycle", so
        // this arm reads the log: it captures stderr across three reset cycles and counts banners.
        //
        // POSIX only — it needs fd redirection, and the assertion is about a log line rather than
        // about behaviour a Windows build could differ on.
#if !defined(_WIN32)
        if (!no_decode) {
            // Scratch path, NOT /tmp. On this box /tmp is a RAM-backed tmpfs with a per-user quota
            // shared by every concurrent agent, and exhausting it takes the tooling down for
            // everyone — test_scratch.h exists to make that unrepeatable, and it also gives a
            // per-case, per-pid path so parallel ctest cases cannot collide (#1621).
            const std::string log_path =
                prosper_test::test_scratch_file("vdec2-banner-capture.log");
            std::fflush(stderr);
            const int saved = dup(fileno(stderr));
            const bool redirected = saved >= 0 && std::freopen(log_path.c_str(), "w", stderr);
            if (redirected) {
                for (int cycle = 0; cycle < 3; ++cycle) {
                    reset(refused, 0, 0, 0, 0, 0);
                    f2.accepted = 1;
                    decode(refused, (uint64_t)(uintptr_t)&input, (uint64_t)(uintptr_t)&f2,
                           (uint64_t)(uintptr_t)&o2, 0, 0);
                }
                std::fflush(stderr);
                dup2(saved, fileno(stderr));
                clearerr(stderr);
            }
            if (saved >= 0) close(saved);
            // A FAILED CAPTURE FAILS THE TEST. This block used to be wrapped in `if (captured)`,
            // so a failed freopen dropped both assertions below and the case still reported
            // success — a test whose failure mode is indistinguishable from passing, which is the
            // whole subject of #2149 and the shape #2566 hit. If the instrument cannot run, that is
            // a result, not a skip. (#2571 review D2.)
            CHECK(redirected, "the banner-count arm could capture stderr (a failed capture is a "
                              "FAILURE, never a silent skip)");
            if (redirected) {
                int banners = 0;
                if (std::FILE* f = std::fopen(log_path.c_str(), "rb")) {
                    char line[512];
                    while (std::fgets(line, sizeof line, f))
                        if (std::strstr(line, "NO DECODER")) ++banners;
                    std::fclose(f);
                } else {
                    CHECK(false, "the captured banner log could be re-opened for reading");
                }
                std::remove(log_path.c_str());
                // Three resets re-open three times; the banner must still have been printed zero
                // more times, because it already fired before this block.
                CHECK(banners == 0,
                      "Reset re-arms the open but does NOT re-print the NO DECODER banner (#2571 N3)");
                CHECK(fake.opens == opens_before + 4,
                      "each Reset does re-open, so the banner suppression is not just a dead path");
            }
        }
#endif
        CHECK(destroy(refused, 0, 0, 0, 0, 0) == 0, "the refused decoder tears down cleanly");
    }

    CHECK(destroy(handle, 0, 0, 0, 0, 0) == 0, "DeleteDecoder tears down cleanly");
    if (!no_decode)
        CHECK(fake.closes >= 1 && fake.last_closed >= 0,
              "DeleteDecoder closes the backend decoder rather than stranding it");

    video::set_backend(nullptr);
    printf(fails ? "FAILED (%d)\n" : "PASSED (%d failures)\n", fails);
    return fails ? 1 : 0;
}
