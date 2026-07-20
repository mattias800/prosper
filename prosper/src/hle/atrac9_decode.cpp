// atrac9_decode.cpp — see atrac9_decode.hpp. Thin RAII glue over vendored LibAtrac9.
#include "atrac9_decode.hpp"

extern "C" {
#include "libatrac9.h"   // third_party/libatrac9/src (added to prosper_core's include path via CMake)
}

namespace prosper {

Atrac9Decoder::~Atrac9Decoder() {
    if (handle_) Atrac9ReleaseHandle(handle_);
}

bool Atrac9Decoder::init(const uint8_t config[4]) {
    if (handle_) { Atrac9ReleaseHandle(handle_); handle_ = nullptr; }
    void* h = Atrac9GetHandle();
    if (!h) return false;
    // Atrac9InitDecoder takes a non-const pointer but only reads the 4 config bytes.
    unsigned char cfg[ATRAC9_CONFIG_DATA_SIZE];
    for (int i = 0; i < ATRAC9_CONFIG_DATA_SIZE; i++) cfg[i] = config[i];
    if (Atrac9InitDecoder(h, cfg) != 0) { Atrac9ReleaseHandle(h); return false; }
    Atrac9CodecInfo ci{};
    if (Atrac9GetCodecInfo(h, &ci) != 0) { Atrac9ReleaseHandle(h); return false; }
    handle_ = h;
    channels_ = ci.channels;
    sample_rate_ = ci.samplingRate;
    superframe_bytes_ = ci.superframeSize;
    frame_samples_ = ci.frameSamples;
    frames_per_superframe_ = ci.framesInSuperframe;
    return true;
}

int Atrac9Decoder::decode_superframe(const uint8_t* in, int16_t* out, int avail) {
    if (!handle_ || !in || !out) return -1;
    // LibAtrac9 decodes one FRAME per call (frame_samples * channels interleaved shorts) and reports
    // the compressed bytes it consumed; iterate the frames_per_superframe frames that make up a
    // superframe, advancing the compressed cursor by the consumed byte count each time.
    //
    // Atrac9Decode takes no input length and its bit reader is unbounded, so a truncated or corrupt
    // block can parse past the superframe. Bound the running total against the superframe size (and
    // the caller's readable extent when supplied) and reject rather than trust an overrun parse.
    const int limit = (avail >= 0 && avail < superframe_bytes_) ? avail : superframe_bytes_;
    const int per_frame = frame_samples_ * channels_;
    int consumed = 0;
    int16_t* o = out;
    for (int f = 0; f < frames_per_superframe_; ++f) {
        int used = 0;
        if (Atrac9Decode(handle_, in + consumed, o, &used) != 0 || used <= 0) return -1;
        consumed += used;
        if (consumed > limit) return -1;      // frame ran past the superframe / readable input
        o += per_frame;
    }
    return superframe_samples();
}

} // namespace prosper
