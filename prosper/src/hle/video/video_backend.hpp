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

// The exact packed NV12 size: a full-resolution luma plane plus a half-resolution interleaved chroma
// plane, each chroma dimension rounded UP.
//
// ONE function with three callers -- the Videodec2 sizing query that tells the guest how big a frame
// buffer to allocate, the backend's own "does the caller's buffer fit" check, and the copy that
// fills it. They must agree by construction; a divergence would be a guest buffer sized by one rule
// and written by another, which is a heap overflow presenting as a decoder bug.
//
// `w*h*3/2` is NOT used, and neither is `wh + (wh+1)/2`. Both are exact only when BOTH dimensions
// are even, and that rule is unusually easy to state wrongly: "even product" is the seductive wrong
// version, and 1920x1081 has an even product while both shorthands are 960 bytes short there. It
// never bit because H.264 codes in 16x16 macroblocks so its dimensions are always even -- but VP9
// frame dimensions are arbitrary, and VP9 is the branch Sonic Racing: CrossWorlds takes. Only ONE
// dimension needs to be odd for the shorthand to be short (65x32 is short by 16).
//
// WIDEN BEFORE ARITHMETIC. Written as `2*((w+1)/2)*...` on `int32_t` inputs the `+1` overflows at
// INT32_MAX before any cast; taking uint32_t and widening here makes every operation 64-bit.
inline uint64_t nv12_bytes(uint32_t w, uint32_t h) {
    const uint64_t uw = w, uh = h;
    return uw * uh + 2 * ((uw + 1) / 2) * ((uh + 1) / 2);
}

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
    // Non-destructive front-of-queue inspection, for PTS-gated delivery: the caller decides
    // whether the frame is due BEFORE taking it, because next_video pops. Default false means
    // "this backend cannot peek", which leaves delivery ungated rather than blocking playback.
    virtual bool peek_video(int id, VideoFrame& out) { return false; }
    // Frames dropped inside the backend (queue-full eviction). Diagnostic: the HLE layer reports
    // it at EOF so a short-delivery measurement can name the stage that lost them.
    virtual uint64_t video_frames_dropped(int id) { return 0; }
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

    // What one access-unit submission produced. THREE outcomes, not a bool, because collapsing them
    // destroys information the caller needs even where its guest-facing answer is the same.
    //
    // Stated as what the HLE actually does with them, rather than as a rule it does not follow:
    // `sceVideodec2Decode` answers the GUEST identically for NoPicture and FrameTooSmall — SCE_OK
    // with no picture, which is the honest answer in both cases — and differs only in the LOG. That
    // difference is the whole point: "the decoder needs more input" is benign and expected, while
    // "a picture exists and your buffer cannot hold it" means the size derivation the guest
    // allocated from is wrong, and the two are indistinguishable to a reader once merged. #2270 is
    // the issue that exists because outcomes were reported identically. (#2571 review D6.)
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
        // The OUTCOME of the hardware request, not the request (#2586). True only when the frame
        // came back in the backend's hardware pixel format, which is the only moment hardware decode
        // is actually knowable -- the backend's own open log says "requested" for exactly this
        // reason. Without it a two-pass test that asks for hardware and then forces software cannot
        // tell a host that gave it both paths from a host that quietly gave it the same path twice,
        // which is the coverage claim the test exists to make.
        bool hardware = false;
    };

    // Submit one access unit and, if it completes a picture, write the packed NV12 into `dst`.
    // Defaulted to NoPicture so a backend without the access-unit path keeps compiling AND answers
    // honestly -- the same reason seek() and open_memory() are defaulted.
    virtual AuResult decode_au(int /*id*/, const uint8_t* /*au*/, size_t /*bytes*/,
                               uint8_t* /*dst*/, uint64_t /*dst_bytes*/, AuPicture& /*out*/) {
        return AuResult::NoPicture;
    }

    // Discard the decoder's buffered state IN PLACE, keeping the decoder itself (#2585,
    // sceVideodec2Reset). Returns false if this backend has no such operation.
    //
    // Three lifecycle verbs, three different contracts, and the whole of #2585 was implementing one
    // as another: `sceVideodec2Flush` DRAINS (it carries a VdecFrame/VdecOutput, so it hands
    // pictures back), `sceVideodec2Reset` DISCARDS (it carries neither), and
    // `sceVideodec2DeleteDecoder` DESTROYS. They are three separate exports of libSceVideodec2, and
    // the library has no re-initialise entry point between Create and Delete -- so a Reset that
    // destroyed the decoder would leave the guest no way to rebuild it except Delete + Create with
    // the full VdecConfig, which is what DeleteDecoder is already for. Reset therefore has to leave
    // the decoder usable. CONFIDENCE: HIGH on that, from the export list alone.
    //
    // Defaulted to `false` rather than to an empty body, because an empty body is the one answer
    // that is wrong under EVERY reading of the contract: it would leave the pre-reset DPB live and
    // decode the guest's next access units against references it just asked us to forget -- a
    // corrupt picture rather than an error, which is silent. `false` lets the caller fall back to
    // close-and-reopen and SAY it did, which forgets too much rather than too little.
    virtual bool reset_decoder(int /*id*/) { return false; }

    virtual void close_decoder(int /*id*/) {}
};

// Registered by the app frontend. nullptr means no native decoder is available.
void         set_backend(VideoBackend* b);
VideoBackend* backend();

} // namespace prosper::video
