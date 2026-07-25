// test_ajm_mp3 (#1347) — real AJM batch-2 MP3 decode through the optional FFmpeg backend.
//
// The project-owned input is four constant-bitrate MPEG-1 Layer III frames generated from a 1 kHz
// sine wave (FFmpeg lavfi + libmp3lame, 48 kHz mono, 80 kbps, no container/tag). Its dimensions
// mirror GTA V's live jobs exactly: 960 compressed bytes -> 4 * 1152 mono S16 samples = 9216 bytes.
#include "hle/dispatch.hpp"
#include "hle/ajm_decoder.hpp"
#include "hle/nid.hpp"
#include "ajm_ffmpeg.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

using HleFn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
using Hle10Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                             uint64_t, uint64_t, uint64_t, uint64_t);

struct Sideband {
    int32_t iResult;
    int32_t iCodecResult;
    uint32_t iSizeConsumed;
    uint32_t iSizeProduced;
    uint64_t uiTotalDecodedSamples;
    uint32_t numFrames;
    uint32_t reserved;
};
static_assert(sizeof(Sideband) == 32);

struct BatchInfo {
    uint64_t buffer;
    uint64_t offset;
    uint64_t size;
    uint64_t last_good_job;
    uint64_t last_good_job_return_address;
};
static_assert(sizeof(BatchInfo) == 40);

static uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }

static std::vector<uint8_t> mp3_input() {
    constexpr std::string_view hex =
        "fffb64c400000a5c552e15a78001bc99e6e73940003060cc18330a14c28530a0"
        "c0c0d7bb706588f6614a9a3626ee19c5866f5a9a93a644396fc580d02082682e"
        "0420b81d0ac676357a7d5eaf677f7a51e3c78f0109707c1fd4efcff29e1fe047"
        "70ff28ee7fa3dfd0000060821041040040050818543db99fa13249580bc34622a"
        "f8502658f19a8cb8643131594f45653f0314a40d21d5703041c3920b9efc2d08"
        "59a251fc3008c70a0842c393ff8b9456a43873873899fff22a404a44588b18977"
        "fff264c8bc5e312e974f7f2c15054b1e3bff2a223c1aa8a54b59971802002a98"
        "0d4028986200939858c1b5984a4044fffb64c40c83cd859b021dfa0000bcb3604"
        "1701ec8981ec1f8990c057a99129d0e9a5ba5a71909a1bf180b00669814c0929"
        "8248027980b202601aa5a0645e00c361eb7ffcb3fffffcb2dfffffcb5fffffaca"
        "adfffff3adfffff9c3dfffffacf7febffd4471abbff4fc22015830312f08f4ee0"
        "c32c7ffbffe59fc4e5bfffff8c7fffff18fffffc4edfffff942dfffffab7febff"
        "c405bffa55f84402b060625e11e9dc18658fff7ffcb3f89cb7fffff18fffffe31"
        "fffff89dbfffff285bfffff56ffd7ff880b7ff4fc22015830312f08f4ee0c32c"
        "7ffbffe59fc4e5bfffff8c7fffff18fffffc4edfffff942dfffffab7ffffb64c4"
        "2c03c5e59b020b80f640bcb36041701ec8ebffc405bffa55f84402b060625e11"
        "e9dc18658fff7ffcb3f89cb7fffff18fffffe31fffff89dbfffff285bfffff56f"
        "fd7ff880b7ff4fc22015830312f08f4ee0c32c7ffbffe59fc4e5bfffff8c7fff"
        "ff18fffffc4edfffff942dfffffab7febffc405bffa55761aaa61975c4401c180"
        "b4000985f40511843c182185840ea1858a24f98f0245898ab7c6b988be51518b"
        "e8043184120b6981fa09c9822a03898000003037800d0b900676273dffeb2cfff"
        "ffeb2c9a7fffff2d7fffff2d7ffffe593dfffff9c3dfffffadbff5fff2511ffe9"
        "f7801123d1f0b85e3717fffb64c46a03c5e59b020b80f640bcb36041701ec889"
        "c0c04f7c632ed9b9266c287502ef9615fad16afe965c6215f0344201bdc06188"
        "0317e185c0503890000060145400463950821d06e580d0d00e17f2b93e70d09c"
        "00008028a80c80200d0203077f2c93e41cb04e1100022e0dc802448018605ff0"
        "2c37facdcdd481a2c0283863404418028007a60a0106cfffba6a642ee2380bb81"
        "b682930bb41bbc63034afffff12814c608a0ca0314500591882b7ffffffd47cb8"
        "b3c5f51f2e2278beea6726800025104c4f45a1d19190944931104c4c5d4c210"
        "120928044b4d22448ecccb51c483822782a0aac34581a3c2278";
    auto nibble = [](char c) -> uint8_t {
        return c >= 'a' ? (uint8_t)(c - 'a' + 10) : (uint8_t)(c - '0');
    };
    std::vector<uint8_t> bytes(hex.size() / 2);
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = (uint8_t)((nibble(hex[i * 2]) << 4) | nibble(hex[i * 2 + 1]));
    return bytes;
}

int main() {
    std::puts("== test_ajm_mp3 ==");
    register_builtin_hle();
    CHECK(ajm::install_ffmpeg_decoder_backend(), "FFmpeg AJM decoder backend installs");
    CHECK(ajm::decoder_backend()->create(ajm::Codec::Mp3, 0) != nullptr,
          "zero AJM max-channel field retains its default stereo S16 contract");

    auto initialize = reinterpret_cast<HleFn>(Hle::lookup(nid_hash("sceAjmInitialize")));
    auto module_register = reinterpret_cast<HleFn>(Hle::lookup(nid_hash("sceAjmModuleRegister")));
    auto instance_create = reinterpret_cast<HleFn>(Hle::lookup(nid_hash("sceAjmInstanceCreate")));
    auto batch_initialize = reinterpret_cast<HleFn>(Hle::lookup(nid_hash("sceAjmBatchInitialize")));
    auto instance_destroy = reinterpret_cast<HleFn>(Hle::lookup(nid_hash("sceAjmInstanceDestroy")));
    auto batch_wait = reinterpret_cast<HleFn>(Hle::lookup(nid_hash("sceAjmBatchWait")));
    auto job_decode = reinterpret_cast<Hle10Fn>(Hle::lookup(nid_hash("sceAjmBatchJobDecode")));
    auto batch_start = reinterpret_cast<Hle10Fn>(Hle::lookup(nid_hash("sceAjmBatchStart")));
    CHECK(initialize && module_register && instance_create && batch_initialize &&
          instance_destroy && batch_wait && job_decode && batch_start,
          "AJM lifecycle and batch-2 handlers are registered");
    if (fails) return 1;

    uint32_t context = 0;
    CHECK(initialize(0, addr(&context), 0, 0, 0, 0) == 0 && context != 0,
          "AJM context initializes");
    CHECK(module_register(context, 0 /* MP3 */, 0, 0, 0, 0) == 0,
          "AJM MP3 module registers");
    uint32_t instance = 0;
    CHECK(instance_create(context, 0 /* MP3 */, 1 /* signed-16 output */, addr(&instance), 0, 0) == 0 &&
          instance != 0, "AJM MP3 instance owns a persistent decoder");

    uint8_t batch_buffer[64]{};
    struct { BatchInfo info; uint64_t canary; } batch{};
    std::memset(&batch.info, 0xCC, sizeof(batch.info));
    batch.canary = 0x1122334455667788ull;
    CHECK(batch_initialize(addr(batch_buffer), sizeof(batch_buffer), addr(&batch.info), 0, 0, 0) == 0,
          "BatchInitialize accepts caller storage");
    CHECK(batch.info.buffer == addr(batch_buffer) && batch.info.offset == 0 &&
          batch.info.size == sizeof(batch_buffer) && batch.info.last_good_job == 0 &&
          batch.info.last_good_job_return_address == 0,
          "BatchInitialize writes the complete 40-byte batch-info contract");
    CHECK(batch.canary == 0x1122334455667788ull,
          "BatchInitialize does not overrun the 40-byte batch info");

    const std::vector<uint8_t> mp3 = mp3_input();
    CHECK(mp3.size() == 960, "test stream is four 240-byte MP3 frames");

    // The host seam promises to retain decoded PCM that did not fit. Exercise it directly with a
    // one-frame output followed by an input-free drain: no compressed data or decoded samples may be
    // lost merely because a guest submits a smaller output buffer.
    std::unique_ptr<ajm::StreamDecoder> streaming =
        ajm::decoder_backend()->create(ajm::Codec::Mp3, 1);
    std::vector<int16_t> first_frame(1152), remaining_frames(3 * 1152);
    const ajm::DecodeResult first = streaming->decode(mp3, first_frame);
    const ajm::DecodeResult remaining = streaming->decode({}, remaining_frames);
    CHECK(first.ok && first.consumed_bytes == 960 && first.produced_bytes == 2304,
          "MP3 decoder retains PCM that exceeds one output frame");
    CHECK(remaining.ok && remaining.consumed_bytes == 0 && remaining.produced_bytes == 6912,
          "MP3 decoder drains all retained PCM without re-consuming input");

    // Fill the decoder's retained-PCM high-water mark with no output capacity. Once capped, another
    // call must consume no new compressed bytes; after draining, the PCM count must match exactly the
    // codec frames accepted before backpressure. The old post-packet-only check grew carry by one
    // frame on every zero-output retry.
    std::unique_ptr<ajm::StreamDecoder> bounded =
        ajm::decoder_backend()->create(ajm::Codec::Mp3, 1);
    std::vector<uint8_t> repeated_mp3;
    repeated_mp3.reserve(mp3.size() * 240);
    for (int i = 0; i < 240; ++i)
        repeated_mp3.insert(repeated_mp3.end(), mp3.begin(), mp3.end());
    const ajm::DecodeResult filled = bounded->decode(repeated_mp3, {});
    CHECK(filled.ok && filled.consumed_bytes > 0 && filled.consumed_bytes < repeated_mp3.size() &&
          filled.produced_bytes == 0 && filled.decoded_frames > 0,
          "MP3 decoder stops accepting input when retained PCM reaches its bound");
    const ajm::DecodeResult capped = bounded->decode(
        std::span<const uint8_t>(repeated_mp3).subspan(filled.consumed_bytes), {});
    CHECK(capped.ok && capped.consumed_bytes == 0 && capped.produced_bytes == 0 &&
          capped.decoded_frames == 0,
          "capped MP3 decoder applies backpressure before parsing another packet");
    std::vector<int16_t> bounded_drain((size_t)filled.decoded_frames * 1152 + 1152);
    const ajm::DecodeResult drained = bounded->decode({}, bounded_drain);
    CHECK(drained.ok && drained.consumed_bytes == 0 &&
          drained.produced_bytes / sizeof(int16_t) == (size_t)filled.decoded_frames * 1152,
          "bounded MP3 carry drains exactly the frames accepted before backpressure");

    // AJM jobs need not end on an MPEG frame boundary. The first half-frame is a successful
    // parser-only job: it consumes bytes into persistent state while channel/rate metadata and PCM
    // remain unavailable. A later batch supplies the suffix and must complete the original frame
    // without the guest resending its prefix.
    constexpr size_t split = 120;
    Sideband prefix_sideband{};
    std::vector<int16_t> split_pcm(4 * 1152, (int16_t)0x5555);
    CHECK(job_decode(addr(&batch.info), instance, addr(mp3.data()), split,
                     addr(split_pcm.data()), split_pcm.size() * sizeof(int16_t),
                     addr(&prefix_sideband), 0, 0, 0) == 0,
          "partial-frame MP3 prefix job queues");
    uint32_t prefix_batch_id = 0xffffffffu;
    CHECK(batch_start(context, addr(&batch.info), 40, 0, addr(&prefix_batch_id),
                      0, 0, 0, 0, 0) == 0,
          "partial-frame MP3 prefix batch executes");
    CHECK(prefix_sideband.iResult == 0 && prefix_sideband.iCodecResult == 0 &&
          prefix_sideband.iSizeConsumed == split && prefix_sideband.iSizeProduced == 0 &&
          prefix_sideband.uiTotalDecodedSamples == 0 && prefix_sideband.numFrames == 0,
          "parser-only MP3 job succeeds with consumed input and no invented PCM");

    Sideband suffix_sideband{};
    CHECK(job_decode(addr(&batch.info), instance, addr(mp3.data() + split), mp3.size() - split,
                     addr(split_pcm.data()), split_pcm.size() * sizeof(int16_t),
                     addr(&suffix_sideband), 0, 0, 0) == 0,
          "remaining MP3 suffix job queues on the persistent instance");
    uint32_t suffix_batch_id = 0xffffffffu;
    CHECK(batch_start(context, addr(&batch.info), 40, 0, addr(&suffix_batch_id),
                      0, 0, 0, 0, 0) == 0,
          "remaining MP3 suffix completes the split frame");
    CHECK(suffix_sideband.iResult == 0 && suffix_sideband.iSizeConsumed == mp3.size() - split &&
          suffix_sideband.iSizeProduced == 9216 &&
          suffix_sideband.uiTotalDecodedSamples == 4608 && suffix_sideband.numFrames == 4,
          "split MP3 stream decodes once without duplicating its consumed prefix");
    CHECK(std::any_of(split_pcm.begin(), split_pcm.end(),
                      [](int16_t sample) { return sample != 0; }),
          "split MP3 stream produces the generated tone");

    CHECK(instance_destroy(context, instance, 0, 0, 0, 0) == 0,
          "destroying the split-stream MP3 instance releases decoder state");
    instance = 0;
    CHECK(instance_create(context, 0 /* MP3 */, 1 /* signed-16 output */, addr(&instance), 0, 0) == 0 &&
          instance != 0, "fresh MP3 instance starts an independent stream");

    std::vector<int16_t> pcm(4 * 1152, (int16_t)0x5555);
    Sideband sideband{};
    CHECK(job_decode(addr(&batch.info), instance, addr(mp3.data()), mp3.size(),
                     addr(pcm.data()), pcm.size() * sizeof(int16_t), addr(&sideband),
                     0, 0, 0) == 0, "four-frame MP3 decode job queues");
    uint32_t batch_id = 0xffffffffu;
    CHECK(batch_start(context, addr(&batch.info), 40, 0, addr(&batch_id), 0, 0, 0, 0, 0) == 0,
          "BatchStart executes the queued MP3 job");
    CHECK(batch_id != 0 && batch_id != 0xffffffffu,
          "BatchStart writes a valid batch id for BatchWait");
    CHECK(batch_wait(context, batch_id, 0, 0, 0, 0) == 0,
          "BatchWait accepts the returned batch id");
    CHECK(sideband.iResult == 0 && sideband.iCodecResult == 0,
          "MP3 result sideband reports success");
    CHECK(sideband.iSizeConsumed == 960,
          "MP3 sideband reports all four compressed frames consumed");
    CHECK(sideband.iSizeProduced == 9216 && sideband.uiTotalDecodedSamples == 4608,
          "MP3 sideband reports four decoded 1152-sample mono frames");
    CHECK(sideband.numFrames == 4 && sideband.reserved == 0,
          "MP3 MFrame sideband reports all four codec frames");
    CHECK(std::any_of(pcm.begin(), pcm.end(), [](int16_t sample) { return sample != 0; }),
          "decoded MP3 PCM contains the generated tone rather than silence");

    CHECK(instance_destroy(context, instance, 0, 0, 0, 0) == 0,
          "destroying an MP3 instance releases decoder state");

    // The FFmpeg seam currently implements only AJM's signed-16 output encoding. An S32 instance
    // must fail truthfully in the sideband and leave the guest output untouched; reporting success
    // with S16 bytes would silently corrupt the guest mixer's sample interpretation.
    uint32_t unsupported_instance = 0;
    constexpr uint64_t s32_mono_flags = 1u | (1u << 7u);
    CHECK(instance_create(context, 0 /* MP3 */, s32_mono_flags,
                          addr(&unsupported_instance), 0, 0) == 0 && unsupported_instance != 0,
          "unsupported MP3 output encoding retains normal AJM instance lifecycle");
    std::vector<int16_t> rejected_pcm(4 * 1152, (int16_t)0x5555);
    Sideband rejected_sideband{};
    CHECK(job_decode(addr(&batch.info), unsupported_instance, addr(mp3.data()), mp3.size(),
                     addr(rejected_pcm.data()), rejected_pcm.size() * sizeof(int16_t),
                     addr(&rejected_sideband), 0, 0, 0) == 0,
          "unsupported-encoding MP3 job queues for truthful sideband failure");
    uint32_t rejected_batch_id = 0xffffffffu;
    CHECK(batch_start(context, addr(&batch.info), 40, 0, addr(&rejected_batch_id),
                      0, 0, 0, 0, 0) == 0,
          "unsupported-encoding MP3 batch executes");
    CHECK(rejected_sideband.iResult != 0 && rejected_sideband.iSizeConsumed == 0 &&
          rejected_sideband.iSizeProduced == 0 && rejected_sideband.uiTotalDecodedSamples == 0,
          "unsupported MP3 output encoding reports decode failure without consuming input");
    CHECK(std::all_of(rejected_pcm.begin(), rejected_pcm.end(),
                      [](int16_t sample) { return sample == (int16_t)0x5555; }),
          "unsupported MP3 output encoding leaves guest PCM untouched");
    CHECK(instance_destroy(context, unsupported_instance, 0, 0, 0, 0) == 0,
          "unsupported MP3 instance destroys normally");

    ajm::uninstall_ffmpeg_decoder_backend();
    std::printf(fails ? "test_ajm_mp3: %d FAILURE(S)\n" : "test_ajm_mp3: all ok\n", fails);
    return fails ? 1 : 0;
}
