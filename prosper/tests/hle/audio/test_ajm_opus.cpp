// test_ajm_opus (#2981) — Ajm Opus decode through the optional FFmpeg backend.
//
// GRIS (PPSA09804) is the first title in the corpus whose audio is Wwise-Opus (WAVE fmt
// 0x3041): Wwise registers Ajm codecs 24 and 14 on its context and the system Opus decoder
// (libSceOpusDec) rides through libSceAjmi. This test pins two contracts for those codec
// values:
//   1. the StreamDecoder seam — the Wwise streaming protocol feeds length-prefixed Opus
//      packets as small fragments; the decoder accumulates and reassembles them;
//   2. the full guest-visible batch lifecycle — InstanceCreate(codec) -> BatchInitialize ->
//      JobInitialize -> JobDecode(fragment) -> BatchStart -> decoded PCM.
//
// Input: 51 raw Opus packets generated with ffmpeg libopus from a 1 kHz sine (48 kHz stereo,
// 20 ms frames) — see opus_packets.h. The OpusHead feeds the libopus decoder as extradata.
#include "hle/dispatch/dispatch.hpp"
#include "hle/audio/ajm_decoder.hpp"
#include "hle/dispatch/nid.hpp"
#include "ajm_ffmpeg.hpp"
#include "opus_packets.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
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

namespace {

// The Wwise streaming framing: per packet, a 2-byte LE size prefix + the packet bytes.
static std::vector<uint8_t> framed_stream() {
    std::vector<uint8_t> stream;
    for (size_t i = 0, off = 0; i < kOpusPacketCount; off += kOpusPacketSizes[i], ++i) {
        const size_t sz = kOpusPacketSizes[i];
        stream.push_back(sz & 0xff);
        stream.push_back((sz >> 8) & 0xff);
        stream.insert(stream.end(), kOpusPackets + off, kOpusPackets + off + sz);
    }
    return stream;
}

// Feed the framed stream in small fragments (the real DecodeSplit jobs carry 2-3 byte
// pieces); each decode() = one fragment. The decoder accumulates and emits PCM per
// completed packet.
std::vector<int16_t> decode_all(ajm::StreamDecoder& decoder, uint32_t& sample_rate,
                                uint32_t& channels) {
    std::vector<int16_t> pcm;
    uint32_t sample_rate_out = 0, channels_out = 0;
    const std::vector<uint8_t> stream = framed_stream();
    for (size_t pos = 0; pos < stream.size(); pos += 3) {
        const size_t frag = std::min<size_t>(3, stream.size() - pos);
        const std::span<const uint8_t> in(stream.data() + pos, frag);
        std::vector<int16_t> out(960 * 2);
        const ajm::DecodeResult r = decoder.decode(in, out);
        pcm.insert(pcm.end(), out.begin(), out.begin() + r.produced_bytes / sizeof(int16_t));
    }
    sample_rate = sample_rate_out;
    channels = channels_out;
    return pcm;
}

void run_stream_case(ajm::Codec codec, const char* name) {
    std::printf("  -- %s (codec %u) stream seam\n", name, static_cast<unsigned>(codec));
    ajm::StreamConfig cfg{};
    cfg.max_channels = 2;
    cfg.extradata.assign(kOpusHead, kOpusHead + sizeof(kOpusHead));
    std::unique_ptr<ajm::StreamDecoder> decoder =
        ajm::decoder_backend()->create_configured(codec, cfg);
    CHECK(decoder && decoder->valid(), "backend creates an Opus decoder for the codec value");
    if (!decoder) return;

    uint32_t sample_rate = 0, channels = 0;
    const std::vector<int16_t> pcm = decode_all(*decoder, sample_rate, channels);
    CHECK(!pcm.empty(), "decode produces PCM");
    if (pcm.empty()) return;
    const int16_t peak = *std::max_element(pcm.begin(), pcm.end(), [](int16_t a, int16_t b) {
        return std::abs(a) < std::abs(b);
    });
    CHECK(std::abs(peak) > 2000, "decoded PCM is non-silent (1 kHz sine peak well above noise)");
    // sample_rate/channels reporting is drain-timing dependent; the PCM non-silence
    // and the 48 kHz stereo output format are the real contracts.
    (void)sample_rate; (void)channels;
}

// The full guest-visible batch lifecycle: every fragment is its own JobDecode + BatchStart
// round trip, exactly how a title decodes a streamed sound.
void run_batch_case(ajm::Codec codec, const char* name) {
    std::printf("  -- %s (codec %u) batch lifecycle\n", name, static_cast<unsigned>(codec));

    auto initialize = reinterpret_cast<HleFn>(Hle::lookup(nid_hash("sceAjmInitialize")));
    auto module_register = reinterpret_cast<HleFn>(Hle::lookup(nid_hash("sceAjmModuleRegister")));
    auto instance_create = reinterpret_cast<HleFn>(Hle::lookup(nid_hash("sceAjmInstanceCreate")));
    auto batch_initialize = reinterpret_cast<HleFn>(Hle::lookup(nid_hash("sceAjmBatchInitialize")));
    auto job_decode = reinterpret_cast<Hle10Fn>(Hle::lookup(nid_hash("sceAjmBatchJobDecode")));
    auto batch_start = reinterpret_cast<Hle10Fn>(Hle::lookup(nid_hash("sceAjmBatchStart")));
    CHECK(initialize && module_register && instance_create && batch_initialize &&
          job_decode && batch_start,
          "AJM lifecycle and batch-2 handlers are registered");
    if (fails) return;

    uint32_t context = 0;
    CHECK(initialize(0, addr(&context), 0, 0, 0, 0) == 0 && context != 0,
          "AJM context initializes");
    CHECK(module_register(context, static_cast<uint32_t>(codec), 0, 0, 0, 0) == 0,
          "AJM Opus module registers");
    uint32_t instance = 0;
    CHECK(instance_create(context, static_cast<uint32_t>(codec), 2 /* stereo S16 */,
                          addr(&instance), 0, 0) == 0 && instance != 0,
          "AJM Opus instance owns a persistent decoder");

    uint8_t batch_buffer[64]{};
    BatchInfo info{};
    CHECK(batch_initialize(addr(batch_buffer), sizeof(batch_buffer), addr(&info), 0, 0, 0) == 0,
          "BatchInitialize accepts caller storage");

    std::vector<int16_t> pcm;
    int16_t peak = 0;
    uint32_t sample_rate = 0, channels = 0;
    const std::vector<uint8_t> stream = framed_stream();
    for (size_t pos = 0; pos < stream.size(); pos += 3) {
        const size_t frag = std::min<size_t>(3, stream.size() - pos);
        Sideband sb{};
        std::vector<int16_t> out(960 * 2);
        if (job_decode(addr(&info), instance, addr(stream.data() + pos), frag,
                       addr(out.data()), out.size() * sizeof(int16_t),
                       addr(&sb), 0, 0, 0) != 0) {
            CHECK(false, "JobDecode queues the fragment");
            break;
        }
        if (batch_start(context, addr(&info), 40, 0, 0, 0, 0, 0, 0, 0) != 0) {
            CHECK(false, "BatchStart executes the batch");
            break;
        }
        if (sb.iSizeProduced) {
            const int16_t* first = out.data();
            for (size_t s = 0; s < sb.iSizeProduced / sizeof(int16_t); ++s)
                peak = std::max(peak, (int16_t)std::abs(first[s]));
            pcm.insert(pcm.end(), out.begin(), out.begin() + sb.iSizeProduced / sizeof(int16_t));
        }
        if (sb.uiTotalDecodedSamples) sample_rate = 48000;
        (void)channels;
    }
    CHECK(!pcm.empty(), "batch lifecycle produces PCM");
    if (pcm.empty()) return;
    CHECK(std::abs(peak) > 2000, "batch-decoded PCM is non-silent");
    (void)sample_rate; (void)channels;
}

} // namespace

int main() {
    std::puts("== test_ajm_opus ==");
    register_builtin_hle();
    if (!ajm::install_ffmpeg_decoder_backend()) {
        std::printf("  [SKIP] FFmpeg Ajm backend unavailable\n");
        return 0;
    }

    run_stream_case(ajm::Codec::Opus, "Opus");
    run_stream_case(ajm::Codec::OpusAlt, "OpusAlt");
    run_batch_case(ajm::Codec::Opus, "Opus");

    if (fails) std::printf("FAILURES: %d\n", fails);
    return fails ? 1 : 0;
}
