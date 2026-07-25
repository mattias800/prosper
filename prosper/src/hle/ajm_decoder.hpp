// ajm_decoder.hpp — optional host codec seam for compressed AJM streams.
//
// prosper_core owns the guest-facing AJM ABI, instance/batch lifecycle, and guest-memory copies.
// General-purpose host codecs live in frontends/ and install a backend here, keeping the core free
// of FFmpeg/OS media dependencies. ATRAC9 remains the core's vendored LibAtrac9 implementation.
#pragma once

#include <cstdint>
#include <memory>
#include <span>

namespace prosper::ajm {

enum class Codec : uint32_t {
    Mp3 = 0,
    Atrac9 = 1,
    Aac = 2,
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
    virtual void invalidate() = 0;
    virtual DecodeResult decode(std::span<const uint8_t> input,
                                std::span<int16_t> output) = 0;
};

class DecoderBackend {
public:
    virtual ~DecoderBackend() = default;
    virtual std::unique_ptr<StreamDecoder> create(Codec codec, uint64_t instance_flags) = 0;
};

// Non-owning backend registration. Installation happens before the guest starts.
void set_decoder_backend(DecoderBackend* backend);
DecoderBackend* decoder_backend();

} // namespace prosper::ajm
