// ajm_decoder.hpp — optional host codec seam for compressed AJM streams.
//
// prosper_core owns the guest-facing AJM ABI, instance/batch lifecycle, and guest-memory copies.
// General-purpose host codecs live in frontends/ and install a backend here, keeping the core free
// of FFmpeg/OS media dependencies. ATRAC9 remains the core's vendored LibAtrac9 implementation.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace prosper::ajm {

enum class Codec : uint32_t {
    Mp3 = 0,
    Atrac9 = 1,
    Aac = 2,
    // Observed in the wild but absent from the local SDK headers: GRIS's Wwise (PPSA09804,
    // #2981) registers codecs 14 and 24 on its Ajm context, and libSceOpusDec.prx depends on
    // libSceAjmi — the system Opus decoder rides through Ajm. Values verified empirically;
    // rename from the SDK header when one is available.
    Opus = 24,
    OpusAlt = 14,
};

struct DecodeResult {
    bool ok = false;
    uint32_t consumed_bytes = 0;
    uint32_t produced_bytes = 0;
    uint32_t decoded_frames = 0;  // codec frames represented by this decode call's output sideband
    uint32_t channels = 0;
    uint32_t sample_rate = 0;
};

// Stateful decoder for one AJM instance. decode() emits interleaved signed-16 PCM. Implementations
// retain parser/codec state and any decoded PCM that did not fit in `output` across calls. A caller
// that cannot publish a successful result must invalidate the decoder: retrying input after hidden
// parser/codec state advanced can duplicate or corrupt the stream.
class StreamDecoder {
public:
    virtual ~StreamDecoder() = default;
    virtual bool valid() const = 0;
    // Terminal and idempotent. Implementations may call this before returning an error; HLE calls it
    // again whenever the result cannot be published safely.
    virtual void invalidate() = 0;
    virtual DecodeResult decode(std::span<const uint8_t> input,
                                std::span<int16_t> output) = 0;
    // Compressed bytes the decoder is holding from earlier calls (a partial packet awaiting its
    // tail). The batch executor allows decode() to consume up to `input.size() + pending_bytes()`:
    // retiring buffered bytes is legitimate consumption even when they were fed by an earlier job.
    virtual size_t pending_bytes() const { return 0; }
};

// Out-of-band stream configuration, for callers that know more than AJM's instance flags can say.
//
// AJM packs only a channel bound and a sample encoding into `instance_flags`, which is everything
// an AJM instance is told. libSceAudiodec is configured differently: its guest hands a codec
// parameter block carrying the stream's sample rate and channel count, and a raw (MP4/`esds`) AAC
// elementary stream CANNOT be decoded without them — the AudioSpecificConfig lives in the
// container, not in the access units, so a decoder that only sees AUs has no framing to recover it
// from. Passing them through this struct keeps that knowledge in the caller that actually has it.
struct StreamConfig {
    uint32_t max_channels = 0;  // 0 -> the backend's own default bound
    uint32_t sample_rate = 0;   // 0 -> unknown; a hint, never a substitute for what decodes
    uint32_t channels = 0;      // 0 -> unknown; ditto

    // Raw MPEG-4 AudioSpecificConfig fields, for a caller that genuinely has them. libSceAudiodec's
    // AAC parameter block IS an unpacked ASC -- measured live on PPSA06367, whose 48 kHz stereo
    // AAC-LC stream arrives as `{objectType=2, samplingFrequencyIndex=3, channelConfiguration=2}`.
    // Supplying them means the config is RECONSTRUCTED from the guest's own bytes rather than
    // guessed from a rate, which is the difference between a derivation and an assumption.
    uint32_t aac_object_type = 0;         // 0 -> not supplied
    uint32_t aac_sample_rate_index = 0;   // only meaningful when aac_object_type != 0
    uint32_t aac_channel_config = 0;      // ditto

    // Raw codec header bytes (e.g. the 19-byte OpusHead), fed to the host decoder as extradata.
    // Without the OpusHead the FFmpeg libopus decoder's state breaks on the second sequential
    // packet (#2981). Empty when the caller has no header.
    std::vector<uint8_t> extradata;
};

// MPEG-4 AudioSpecificConfig sampling-frequency index table (ISO/IEC 14496-3, Table 1.18). Shared
// because the HLE needs the rate in Hz to publish to its guest and the codec backend needs the
// index to rebuild the config; two copies of a published constant table is one copy too many.
inline constexpr int kAacSampleRates[] = {96000, 88200, 64000, 48000, 44100, 32000,
                                          24000, 22050, 16000, 12000, 11025, 8000, 7350};

// Hz for an ASC sampling-frequency index, or 0 for an index MPEG-4 does not define (15 means the
// rate is written out explicitly in the config, which an unpacked parameter block cannot express).
inline uint32_t aac_sample_rate_from_index(uint32_t index) {
    if (index >= sizeof(kAacSampleRates) / sizeof(kAacSampleRates[0])) return 0;
    return static_cast<uint32_t>(kAacSampleRates[index]);
}

// Compressed input handed to a StreamDecoder must be zero-padded to this many trailing bytes:
// FFmpeg's optimized parsers read AV_INPUT_BUFFER_PADDING_SIZE past the packet. The seam owns the
// number so core HLE never encodes an FFmpeg constant whose value changed across versions.
inline constexpr size_t kStreamInputPadding = 64;

class DecoderBackend {
public:
    virtual ~DecoderBackend() = default;
    virtual std::unique_ptr<StreamDecoder> create(Codec codec, uint64_t instance_flags) = 0;

    // Default: ignore the out-of-band configuration and fall back to the flags form. A backend
    // written before this seam existed therefore keeps working unchanged, and simply cannot serve
    // the codecs that need it (it returns nullptr for them exactly as it did before).
    virtual std::unique_ptr<StreamDecoder> create_configured(Codec codec, const StreamConfig& cfg) {
        return create(codec, static_cast<uint64_t>(cfg.max_channels & 0x7fu));
    }
};

// Non-owning backend registration. Installation happens before the guest starts.
void set_decoder_backend(DecoderBackend* backend);
DecoderBackend* decoder_backend();

} // namespace prosper::ajm
