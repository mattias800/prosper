#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

// Host video-decode backend for the core sceAvPlayer HLE (#705).
//
// The core owns the guest-facing sceAvPlayer contract, per-player lifecycle, the guest event
// callback, and marshalling decoded frames into the guest SceAvPlayerFrameInfo buffers. The actual
// hardware decode is delegated to a backend that the APP (frontend) registers via set_backend() —
// Media Foundation (Windows) / VideoToolbox (macOS) / VA-API (Linux). Native OS media APIs do demux
// + hardware decode together, so a backend owns the whole open->demux->decode pipeline and just hands
// the core the next decoded frame.
//
// Synthetic playback is an explicit diagnostic selected with PROSPER_AVP_SYNTH_FRAMES. Merely
// running without a backend must fail source-open truthfully instead of fabricating playback.
// Requirement: real backends MUST use hardware decoding.
namespace prosper::video {

struct StreamInfo {
    uint32_t width = 0, height = 0;
    float    fps = 0.0f;
    bool     has_audio = false;
    uint32_t audio_channels = 0, audio_rate = 0;
    uint64_t duration_us = 0;
};

// One decoded video frame, NV12 (Y plane + interleaved UV plane) — the format PS5 AvPlayer delivers.
struct VideoFrame {
    const uint8_t* y = nullptr;
    const uint8_t* uv = nullptr;
    uint32_t width = 0, height = 0, y_stride = 0, uv_stride = 0;
    uint64_t pts_us = 0;
};

struct AudioFrame {
    const int16_t* pcm = nullptr;
    uint32_t channels = 0, samples = 0, sample_rate = 0;
    uint64_t pts_us = 0;
};

class VideoBackend {
public:
    virtual ~VideoBackend() = default;
    virtual int  open(const std::string& host_path) = 0;  // returns stream id (>=0), <0 on error; starts demux+HW decode
    // Open a source whose bytes the CALLER supplies rather than a host path (#1955). sceAvPlayerInit's
    // guest file-replacement callbacks are the only way a title can express "the media is a byte range
    // inside this container file", and prosper cannot call them from a decode worker (they are guest
    // code and need the calling guest thread's TCB), so the HLE reads the media through them up front
    // and hands the bytes over here. The backend TAKES A COPY; `data` need not outlive the call.
    // `debug_name` is for logs only. Defaulted to unsupported so other backends keep compiling.
    virtual int  open_memory(const std::string& /*debug_name*/, const uint8_t* /*data*/,
                             size_t /*bytes*/) { return -1; }
    virtual bool info(int id, StreamInfo& out) = 0;        // true once stream headers are parsed
    // Successful pull pointers remain valid until the next pull of that media type for this id.
    virtual bool next_video(int id, VideoFrame& out) = 0;  // false if no frame ready yet
    virtual bool next_audio(int id, AudioFrame& out) = 0;  // false if no audio ready yet
    // AvPlayer video completion is driven by the video stream. Queued audio must not keep a
    // video-only consumer active after decode has ended and the final video frame was delivered.
    virtual bool eof(int id) = 0;
    // Reposition playback so the NEXT pulled frame is the one at or after `position_us`, discarding
    // everything already queued and clearing any end-of-stream state (#1949, sceAvPlayerJumpToTime).
    //
    // Defaulted to "unsupported" so a backend that cannot seek keeps compiling AND answers honestly:
    // the HLE turns `false` into a real sceAvPlayer error. That distinction is the whole point of the
    // issue — the unregistered NID's implicit SCE_OK was read by PPSA30490 as a completed seek, so it
    // waited forever for a position that never moved instead of taking its own failure branch.
    virtual bool seek(int /*id*/, uint64_t /*position_us*/) { return false; }
    virtual void close(int id) = 0;

    // ---- Access-unit decoding, for sceVideodec2 (#2270) ----------------------------------------
    //
    // A DIFFERENT shape from open()/next_video() above, and the difference is the guest's, not ours:
    // AvPlayer hands prosper a whole stream to demux, while Videodec2's caller demuxes ITSELF and
    // submits one compressed access unit at a time, expecting at most one picture back. Measured on
    // Tales of Graces f (PPSA19991): access units of 565-3,984 bytes, i.e. per-frame NALs.
    //
    // That is precisely libavcodec's send_packet/receive_frame contract, so this is a second entry
    // point onto the same decoder rather than a second decoder.
    //
    // Defaulted to unsupported so a backend without it keeps compiling AND answers honestly -- the
    // same reason seek() and open_memory() are defaulted. An HLE that cannot decode must be able to
    // say so; #2270 exists because sceVideodec2Decode instead reported SCE_OK with no picture,
    // forever, which a title cannot distinguish from "no frame ready yet".
    virtual int  open_decoder(uint32_t /*codec*/) { return -1; }

    // What one access-unit submission produced. THREE outcomes, not a bool, because the caller must
    // answer each of them differently and two of them are indistinguishable once collapsed:
    // "the decoder needs more input" is correct and benign, while "a picture exists and your buffer
    // is too small" means the size derivation everything rests on is wrong. #2270 is the issue that
    // exists because those two were reported identically.
    enum class AuResult {
        NoPicture,      // needs more input -- correct, and NOT an error
        FrameTooSmall,  // a picture exists; `dst_bytes` cannot hold it. `out` says how much it needs
        Decoded,        // `dst` now holds the packed NV12 picture
    };

    // Geometry of the picture decode_au produced or refused. Deliberately carries NO plane pointers.
    //
    // The earlier shape returned a VideoFrame whose `y`/`uv` pointed into the decoder's own staging
    // buffer and stayed valid only "until the next decode_au on this id" -- which made the caller's
    // copy race a concurrent close_decoder that frees exactly that buffer. Handing the DESTINATION
    // to the decoder instead of handing decoder-owned pointers to the caller removes the lifetime
    // question rather than documenting it: the copy happens inside the backend's own lock, so there
    // is nothing a caller can hold that another thread can free. (#2571 review N5.)
    struct AuPicture {
        uint32_t width = 0, height = 0, y_stride = 0, uv_stride = 0;
        // Exact packed NV12 size: w*h + 2*ceil(w/2)*ceil(h/2). NOT the `w*h*3/2` shorthand, which is
        // short whenever a dimension is odd -- see s_videodec2_query_decoder_memory's note on why
        // this project writes the exact form. The guest sizes its frame buffer from that same
        // expression, so any other one here disagrees with the buffer it is copying into.
        uint64_t nv12_bytes = 0;
    };

    // Submit one access unit and, if it completes a picture, write the packed NV12 into `dst`.
    // Defaulted to NoPicture so a backend without the access-unit path keeps compiling AND answers
    // honestly -- the same reason seek() and open_memory() are defaulted.
    virtual AuResult decode_au(int /*id*/, const uint8_t* /*au*/, size_t /*bytes*/,
                               uint8_t* /*dst*/, uint64_t /*dst_bytes*/, AuPicture& /*out*/) {
        return AuResult::NoPicture;
    }
    virtual void close_decoder(int /*id*/) {}
};

// Registered by the app frontend. nullptr means no native decoder is available.
void         set_backend(VideoBackend* b);
VideoBackend* backend();

} // namespace prosper::video
