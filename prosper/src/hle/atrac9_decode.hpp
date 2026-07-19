// atrac9_decode.hpp — prosper's C++ glue over the vendored LibAtrac9 decoder (third_party/libatrac9).
//
// ATRAC9 is Sony's proprietary MDCT audio codec; PS5 titles (Dead Cells' music) stream it into NGS2
// sampler voices. This wraps LibAtrac9's C handle API into an RAII decoder that prosper's NGS2 sampler
// path drives: init once from the 4-byte config, then decode compressed superframes to interleaved
// S16 PCM which the existing voice mixer consumes. The decoder itself is upstream MIT code; this glue
// (lifetime, config, superframe/frame iteration) is prosper's own.
#pragma once
#include <cstdint>
#include <vector>

namespace prosper {

class Atrac9Decoder {
public:
    Atrac9Decoder() = default;
    ~Atrac9Decoder();
    Atrac9Decoder(const Atrac9Decoder&) = delete;
    Atrac9Decoder& operator=(const Atrac9Decoder&) = delete;

    // Initialize from the title's 4-byte ATRAC9 config word. Returns false on a bad/unsupported config.
    bool init(const uint8_t config[4]);
    bool valid() const { return handle_ != nullptr; }

    // Codec geometry (valid after init): interleaved output is `channels` samples per frame position.
    int channels() const { return channels_; }
    int sample_rate() const { return sample_rate_; }
    int superframe_bytes() const { return superframe_bytes_; }        // compressed bytes per superframe
    int superframe_samples() const { return frame_samples_ * frames_per_superframe_; }  // per channel

    // Decode ONE superframe (`superframe_bytes()` compressed bytes at `in`) into `out`, which must hold
    // superframe_samples()*channels() interleaved S16 samples. Returns the number of sample-frames
    // written per channel (== superframe_samples()) on success, or -1 on a decode error.
    int decode_superframe(const uint8_t* in, int16_t* out);

private:
    void* handle_ = nullptr;
    int channels_ = 0, sample_rate_ = 0, superframe_bytes_ = 0;
    int frame_samples_ = 0, frames_per_superframe_ = 0;
};

} // namespace prosper
