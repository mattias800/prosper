// hle_service.cpp — HLE of PS5 system services (user, NP/online, mouse, app content,
// dialogs). Bring-up policy: openers return a valid positive handle; queries zero their
// output struct and report a sane "not signed in / no device" state and success, so the
// game gets consistent values instead of uninitialized memory.
// (Game-controller input — libScePad — moved to hle_pad.cpp with a real host backend.)
#include "hle/dispatch/dispatch.hpp"
#include "hle/service/hle_addcontent.hpp"
#include "hle/fs/save_paths.hpp"   // per-title save roots (#2734)
#include "hle/fs/save_param.hpp"   // the save's parameter block (#2786)
#include "hle/util/hle_json2.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"   // libkernel error encoding (libSceRandom reject arms)
#include "diagnostics/env_numeric.hpp"   // #3267: a typo must not unregister a default-ON NID family
#include "hle/dispatch/callback_fs.hpp"
#include "hle/input/ime_input.hpp"
#include "hle/service/platform_ui.hpp"
#include "hle/video/video_backend.hpp"   // sceAvPlayer -> host hardware-decode backend (#705)
#include "hle/video/h264_sps.hpp"        // SPS/VUI extraction for GetPictureInfo (#2898)
#include "gpu/texture/guest_texture_layout.hpp" // exact HLE-produced sampled-linear layouts
#include "host/platform/posix_shim.hpp"   // Darwin process_vm_readv shim + asm portability
#include "host/image/boot_program.hpp"  // guest_module_name: is a callback target guest code?
#include "host/platform/lifecycle.hpp"  // cooperative stop when the guest reports its own crash (#3119)
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <thread>
#include <deque>
#include <filesystem>
#include <mutex>
#include <new>          // std::bad_alloc — the guest file-replacement buffer is guest-sized (#1955)
#include <set>
#include <unordered_map>
#include <vector>
#include <string>
#ifdef _WIN32
#include <direct.h>     // _mkdir (SaveDataMemory persistence dir)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>   // mkdir
#include <sys/uio.h>    // process_vm_readv: fault-contained diagnostic snapshots
#include <sys/random.h> // getentropy: the host CSPRNG behind sceRandomGetRandomNumber
#include <unistd.h>
#endif
#include "hle/service/service_trace.hpp"
#include "hle/service/hle_handles.hpp"
#ifdef _WIN32
#include <bcrypt.h>     // BCryptGenRandom (prosper_core already links bcrypt on Windows)
#endif

namespace prosper {

#ifdef _WIN32
extern "C" uint64_t prosper_call_guest_sysv4(uint64_t fn, uint64_t a0, uint64_t a1,
                                               uint64_t a2, uint64_t a3);
#endif

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define PW(x) ((void*)(uintptr_t)(x))


// Diagnostic logging for the Sony service families in this file, gated on PROSPER_SVCLOG=1 (same
// pattern as PROSPER_FILELOG/[file]). Dumps call args and a bounded hexdump of pointer-shaped args
// so PS5-only ABIs with no Kyty/shadPS4 reference can be pinned from live captures instead of
// guessed.

// PS5 Game Intent is how the shell starts a title at a selected activity instead of its default
// entry point. A host frontend can model that user action by setting an activity id before boot;
// without it the console truthfully has no pending intent. Keep the payload opaque to the guest:
// titles consume it through sceNpGameIntentGetPropertyValueString, as on the console.
// CONFIDENCE: HIGH — ABI/error contracts match Kyty; event id, type, key, and offsets are also
// independently exercised by Sonic's eboot and its launchActivity declaration in param.json.
namespace {
}
// --- libSceVideodec2 ---------------------------------------------------------------------------
// These ABI layouts/NIDs are the PS5 3.20 interfaces.  Decoding is intentionally a no-picture
// implementation for now: it consumes each access unit, reports a valid lifecycle, and lets the
// movie reach EOF instead of faking video pixels.  Critically, every successful query/open writes
// all output fields.  The former generic success stubs left a null compute queue and crashed libc
// at +0xfe66 on its first queue use.
//
// WHO ACTUALLY EXERCISES THIS, as recorded rather than as assumed.  A header line here used to read
// "Sonic's CRI Mana movie backend uses the native VideoDec2 compute queue and decoder lifecycle."
// It came from #1368 (8b37be95) — the same commit that added the unexplained compute_queue check
// below and an equally unsupported "Sonic Origins' CRI Mana backend passes it", which was already
// withdrawn.  No Videodec2 evidence for Sonic exists anywhere in docs/ or COMPATIBILITY.md, so it is
// withdrawn on the same basis rather than left to be inherited a third time.  What IS recorded:
//   - QueryComputeMemoryInfo only — Dragon Quest VII Reimagined and the UE4 bring-up.
//   - The full decoder lifecycle — Tales of Graces f Remastered (PPSA19991) via criMvPly, and only
//     after #1687; before that no title in this repo's history had ever reached CreateDecoder.
namespace {
// The current no-picture backend has no hardware decoder workspace: it only needs a stable,
// caller-owned address for the opaque compute-queue identity.  Keep one guest page so clients that
// require a non-null allocation can follow the real query/allocate lifecycle without reserving the
// 16 MiB placeholder used by software-decoding implementations.  That placeholder exhausted
// Sonic Origins' 80 MiB CRI CPU/GPU pool before its movie workers could allocate their own buffers.
}
// ===== libSceAvPlayer (#324/#705): real playback lifecycle over a host video-decode backend =====
// The core owns the guest sceAvPlayer contract + per-player state + the guest event callback and
// pulls decoded frames from a registered VideoBackend (app-side hardware decode). With NO backend
// (headless/tests) it runs a synthetic black-frame lifecycle so the title's video state machine still
// completes and the boot advances. ABI structs mirror the published sceAvPlayer layout (guest is
// SysV LP64 == host on Linux, so guest pointers are read directly).
namespace {
// Guest event callback: void(void* object_ptr, uint32_t event, int32_t source_id, void* data).
using AvpEventCb = void (PROSPER_SYSV_ABI *)(void*, uint32_t, int32_t, void*);
// Guest memory callbacks use the PS5/SysV ABI even when the emulator host is Windows.  Keep their
// types explicit here; calling them as native Windows functions corrupts the argument registers.
using AvpAllocateCb = void* (PROSPER_SYSV_ABI *)(void*, uint32_t, uint32_t);
using AvpDeallocateCb = void (PROSPER_SYSV_ABI *)(void*, void*);
enum : uint32_t { AVP_STOP = 0x01, AVP_READY = 0x02, AVP_PLAY = 0x03, AVP_PAUSE = 0x04, AVP_BUFFERING = 0x05 };
// The WARNING channel of the same callback. Unlike the state events above it carries a payload: the
// fourth callback argument points at a uint32_t notification code, and a guest that is waiting for a
// particular one compares against THAT, not against the event id.
//
// AVP_WARNING_SEEK_COMPLETE is the code sceAvPlayerJumpToTime must publish once the source has
// actually been repositioned. #1974 implemented the seek itself and left this half out, so every
// guest that waits for it burns its own timeout on a seek that already succeeded.
//
// EVIDENCE, in the project's own order of preference:
//   (1) The guest's code. Space Adventure Cobra (PPSA17337) writes 0x806a00a3 into its
//       wait-slot at eboot+0x15400b8, immediately BEFORE calling sceAvPlayerJumpToTime at
//       eboot+0x15400c2, and afterwards blocks in its 15,000 ms event wait at eboot+0x153ce5d
//       (`esi=0x3a98`). Its event callback (eboot+0x153ef30) posts that wait only when the event id
//       is 0x20 AND the payload word equals the awaited code -- for a warning it compares the
//       PAYLOAD, having replaced the id with `*(uint32_t*)eventData` at eboot+0x153ef5c. The
//       matching failure code the same callback special-cases is 0x806a0004.
//   (2) Directly measured: with nothing published, that wait times out at exactly its 15,000 ms
//       budget (15,003 ms between the post-seek frame and sceAvPlayerResume, PROSPER_AVPLOG).
//   (3) Cross-title: of the 42 dumps in the local corpus that import sceAvPlayerJumpToTime
//       (`XC9wM+xULz8`), 40 contain 0x806a00a3 as an immediate and 33 also contain 0x806a0004 --
//       against roughly 0.008 expected occurrences per 32 MB eboot by chance. It is an SDK
//       constant shared across engines, not one title's private token.
// CONFIDENCE: HIGH on the behaviour (read off the guest and measured in both directions).
// CONFIDENCE: MED on the name -- Sony's own symbol for 0x806a00a3 is not in any source available
// here, so the name describes the contract rather than quoting a header.
enum : uint32_t { AVP_WARNING_ID = 0x20 };
enum : uint32_t { AVP_WARNING_SEEK_COMPLETE = 0x806a00a3u };

struct AvpMemAllocator {
    void* obj;
    AvpAllocateCb allocate;
    AvpDeallocateCb deallocate;
    AvpAllocateCb allocate_texture;
    AvpDeallocateCb deallocate_texture;
};
struct AvpFileReplace   { void* obj; void* open; void* close; void* read_offset; void* size; };
struct AvpEventReplace  { void* obj; void* event_callback; };
struct AvpInitData {                 // sceAvPlayerInit
    AvpMemAllocator memory; AvpFileReplace file; AvpEventReplace event;
    uint32_t debug_level; uint32_t base_priority; int32_t num_fb; uint8_t auto_start; uint8_t rsv[3]; const char* default_language;
};
struct AvpThreadInfo {
    uint32_t priority, stack_size; uint64_t affinity; uint8_t reserved[32];
};
struct AvpInitDataEx {               // sceAvPlayerInitEx (note: this_size first)
    uint64_t this_size; AvpMemAllocator memory; AvpFileReplace file; AvpEventReplace event;
    const char* default_language; uint32_t debug_level; uint8_t auto_start; uint8_t rsv[3];
    AvpThreadInfo audio_decoder, video_decoder, demuxer, event_thread, call_queue;
    AvpThreadInfo http_command_processor, http_segment_manager, http_streamlist, file_streaming;
    int32_t num_fb; uint8_t rsv2[4];
};
static_assert(sizeof(AvpThreadInfo) == 48 && sizeof(AvpInitDataEx) == 560 &&
              offsetof(AvpInitDataEx, auto_start) == 116 &&
              offsetof(AvpInitDataEx, num_fb) == 552, "AvPlayerInitDataEx ABI");
static_assert(sizeof(AvpMemAllocator) == 40 && offsetof(AvpInitData, num_fb) == 104 &&
              offsetof(AvpInitData, default_language) == 112 && sizeof(AvpInitData) == 120,
              "AvPlayerInitData ABI");
struct AvpUri           { const char* name; uint32_t length; };
struct AvpSourceDetails { AvpUri uri; uint8_t rsv1[64]; uint32_t source_type; uint8_t rsv2[44]; };
struct AvpFrameInfo {                // sceAvPlayerGetVideoData / GetAudioData out-param
    uint8_t* p_data; uint8_t reserved[4]; uint64_t timestamp;
    uint32_t d0, d1, d2, d3;         // AvPlayerStreamDetails union (16B): video={width,height,aspect,lang}
};
struct AvpVideoEx {                  // AvPlayerVideoEx (in the 80B details union)
    uint32_t width, height; float aspect; uint8_t lang[4]; uint8_t reserved0[4];
    uint32_t crop_left_offset, crop_right_offset, crop_top_offset, crop_bottom_offset, pitch;
    uint8_t luma_bd, chroma_bd, full_range, reserved1[5];
    double framerate; uint32_t colour_primaries, transfer_characteristics; uint8_t reserved2[16];
};
static_assert(sizeof(AvpVideoEx) == 80 &&
              offsetof(AvpVideoEx, crop_left_offset) == 20 &&
              offsetof(AvpVideoEx, crop_right_offset) == 24 &&
              offsetof(AvpVideoEx, crop_top_offset) == 28 &&
              offsetof(AvpVideoEx, crop_bottom_offset) == 32 &&
              offsetof(AvpVideoEx, pitch) == 36 &&
              offsetof(AvpVideoEx, framerate) == 48, "AvPlayerVideoEx ABI");
struct AvpFrameInfoEx {              // sceAvPlayerGetVideoDataEx out-param (larger details union)
    void* p_data; uint8_t reserved[4]; uint64_t timestamp;
    union { AvpVideoEx video; uint8_t raw[80]; } details;
};
struct AvpVideo { uint32_t width, height; float aspect; uint8_t lang[4]; };
struct AvpAudio { uint16_t channels; uint8_t reserved[2]; uint32_t sample_rate, size; uint8_t lang[4]; };
struct AvpAudioEx {
    uint16_t channels; uint8_t reserved[2]; uint32_t sample_rate, size; uint8_t lang[4];
    uint8_t reserved1[64];
};
static_assert(sizeof(AvpAudio) == 16 && sizeof(AvpAudioEx) == 80, "AvPlayerAudio ABI");
struct AvpStreamInfo {
    uint32_t type; uint8_t reserved[4];
    union { AvpVideo video; AvpAudio audio; uint8_t raw[16]; } details; uint64_t duration;
};
struct AvpStreamInfoEx {
    uint64_t this_size; uint32_t type; uint8_t reserved[4];
    union { AvpVideoEx video; AvpAudioEx audio; uint8_t raw[80]; } details; uint64_t duration;
};
static_assert(sizeof(AvpStreamInfo) == 32 && offsetof(AvpStreamInfo, duration) == 24,
              "AvPlayerStreamInfo ABI");
static_assert(sizeof(AvpStreamInfoEx) == 104 && offsetof(AvpStreamInfoEx, duration) == 96,
              "AvPlayerStreamInfoEx ABI");

struct AvpPlayer {
    void* ev_obj = nullptr; AvpEventCb ev_cb = nullptr;
    AvpMemAllocator memory{};
    // The guest's file-replacement table, retained verbatim from sceAvPlayerInit(Ex). When all four
    // entries are present, sceAvPlayerAddSource(Ex) reads the media THROUGH it rather than opening
    // the URI as a host path (#1955) — a title that stores a clip inside a container file has no
    // other way to express the byte range. Reported under PROSPER_AVPLOG; every path that declines
    // to use it reports unconditionally. See avp_add_source.
    AvpFileReplace file{};
    int32_t num_fb = 0;
    bool auto_start = false, have_source = false, synthetic = false;
    bool playing = false, paused = false, stop_fired = false;
    // Set by a successful sceAvPlayerJumpToTime, cleared by the first frame delivered after it.
    // A seek repositions the player and must publish the new position even while the guest holds
    // the player PAUSED — see the delivery gate in s_avp_getvideodata[ex] for the guest evidence.
    bool seek_deliver = false;
    std::string guest_path;
    prosper::video::VideoBackend* backend = nullptr;
    int backend_id = -1;             // >=0 when a host backend is decoding
    uint32_t width = 1920, height = 1080;
    float fps = 30.0f;
    bool has_audio = true;
    uint32_t audio_channels = 2, audio_rate = 48000;
    uint64_t duration_ms = 0;
    uint64_t poll = 0;               // synthetic frames delivered through GetVideoData[Ex]
    uint64_t audio_poll = 0;
    uint64_t active_poll = 0;        // independent progress for IsActive-only playback loops
    uint64_t last_ts_ms = 0;         // newest delivered frame timestamp (sceAvPlayerCurrentTime)
    // When something last ASKED this player for a video frame — the media clock's only input
    // (#1973, see avp_media_played_out). Refreshed by every GetVideoData[Ex] call whatever that call
    // returns, and at every point presentation (re)starts: AddSource, Start, Resume, JumpToTime.
    // Those are also every place `paused` is cleared, which is what keeps paused time — during which
    // nothing is presented and nothing should be pulled — from counting as an absent consumer.
    std::chrono::steady_clock::time_point video_request_at = std::chrono::steady_clock::now();
    // The media-clock anchor for BLOCKING video delivery: the frame with pts P is due at
    // play_start_wall + (P - play_start_pts). The caller is held until due — the real
    // AvPlayer's contract, and the only pacing this guest respects: it sleeps its whole
    // poll budget when a non-blocking gate refuses a frame, which collapsed playback to
    // ~1 frame per sleep (measured 0.4-1.2 fps, #2981 FMV).
    std::chrono::steady_clock::time_point play_start_wall = std::chrono::steady_clock::now();
    uint64_t play_start_pts = 0;
    bool gate_started = false;
    // When the title supplies its AvPlayer texture allocator, decoded NV12 must be written into those
    // guest-visible buffers.  A host vector is invisible to the title's pre-wired texture descriptors.
    std::vector<uint8_t*> texture_frames;
    size_t texture_frame_bytes = 0;
    size_t next_texture_frame = 0;
    std::vector<uint8_t> frame;      // fallback for tests/titles with no texture callbacks
    std::vector<int16_t> audio;      // synthetic silent stereo PCM
};
std::mutex g_avp_mx;
std::unordered_map<uint64_t, AvpPlayer> g_avp;

// PS5 AvPlayer exposes decoded NV12 as a sampled-linear surface. Its physical row pitch is aligned
// to 256 bytes; the valid image width remains separate and the padded pixels are reported through
// AvPlayerVideoEx::crop_right_offset. Keeping those two extents distinct is required by consumers
// that validate an AGC texture footprint before binding its second plane.
bool avp_video_pitch(uint32_t width, uint32_t& pitch) {
    if (!width || width > UINT32_MAX - 255u) return false;
    pitch = (width + 255u) & ~255u;
    return true;
}

bool avp_nv12_bytes(uint32_t pitch, uint32_t height, size_t& bytes) {
    if (!pitch || !height || pitch > SIZE_MAX / height) return false;
    const size_t y_bytes = static_cast<size_t>(pitch) * height;
    const size_t uv_rows = (static_cast<size_t>(height) + 1) / 2;
    if (pitch > SIZE_MAX / uv_rows) return false;
    const size_t uv_bytes = static_cast<size_t>(pitch) * uv_rows;
    if (uv_bytes > SIZE_MAX - y_bytes) return false;
    bytes = y_bytes + uv_bytes;
    return true;
}

void avp_register_frame_layout(uint8_t* base, size_t bytes, uint32_t pitch) {
    gpu::register_guest_linear_texture_layout(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)), bytes, pitch);
}

// AvPlayerVideoEx publishes `width` as the extent the crop offsets TRIM, not as the visible picture.
//
// The load-bearing evidence is that three shipping titles run on retail PS5 hardware while spelling
// "how wide is the visible picture" two different ways, so Sony's own publication must satisfy both
// at once. Two of the three fields are already pinned by prior live evidence:
//
//   pitch = 2048   forced by R-Type Delta, which builds its luma T# with `pitch` AS the descriptor
//                  width and requires agc_footprint == pitch*height (#1814, R_TYPE_DELTA_STATUS.md).
//   crop_right     forced by GRIS: #1393 padded the pitch with a zero crop and exposed the padded
//     = 128        columns as a right-edge strip, so the crop must describe that padding.
//
// With those pinned, ArcRunner's spelling leaves exactly one free value:
//
//   GRIS       visible = pitch - crop_left - crop_right  ->  2048 - 0 - 128 = 1920   (correct)
//   ArcRunner  visible = width - crop_left - crop_right  ->  1920 - 0 - 128 = 1792   (WRONG)
//
// so `width` must be 2048. Measured on ArcRunner (PPSA21406, #2011): the guest sizes its movie luma
// T# from the published pitch (2048x1080, chroma 1024x540 — half of it) and, while width published
// the visible 1920, sized every converted movie surface 128 columns short at 1792x1080. Publishing
// width == pitch satisfies both spellings at once and leaves GRIS and R-Type unchanged.
//
// The four crop field names mirror H.264's frame_crop_{left,right,top,bottom}_offset, and that is an
// ANALOGY, not the evidence: H.264's coded extent is macroblock-aligned (a 1920x1080 stream codes
// 1920x1088 with crop_bottom, crop_right=0), never memory-pitch-aligned, so the SPS convention alone
// would argue for width=1920/crop_right=0. The real statement is that the crop fields are the only
// ABI channel able to describe buffer padding, so that is what this contract uses them for.
//
// `aspect` stays the VISIBLE display ratio: it is not part of the crop arithmetic, and a consumer
// reads it to letterbox, never to size a surface.
//
// `width == pitch` is only meaningful because every path here is 8-bit (luma_bd/chroma_bd below):
// `pitch` is a BYTE stride and `width` a PIXEL count, and they coincide only at 1 byte per luma
// sample. A 10-bit/P010 source would need the pixel pitch, not the byte pitch, computed here.
//
// CONFIDENCE: MED — derived from three independent shipping consumers that all work on retail
// hardware. No Sony header was consulted. The counter-model this cannot exclude is that
// AvPlayerVideoEx merely EXTENDS AvPlayerVideo (whose `width` is unambiguously visible) and Sony's
// crop fields describe bitstream cropping rather than buffer padding.
bool avp_fill_video_ex(AvpVideoEx& out, uint32_t visible_w, uint32_t visible_h, uint32_t pitch,
                       double framerate) {
    out = {};
    // Every current caller derives `pitch` from the same width it passes, so this cannot trip today.
    // It is a guard rather than an assumption because the underflow would be silent and guest-visible:
    // crop_right_offset is unsigned, so pitch < visible_w publishes ~4 G and every consumer that
    // subtracts it gets a nonsense extent.
    //
    // FAIL-VISIBLE rather than clamping. Clamping to `visible_w` looks like the safe repair and is the
    // more dangerous of the two: the buffer was staged at the REAL pitch, so publishing a larger row
    // stride with crop_right = 0 tells a guest that trusts it to read past the end of every row, and
    // the extents it gets are plausible enough that nobody notices. The ~4 G crop is at least absurd
    // on sight. Since the branch is unreachable from any current caller, refusing costs nothing and
    // keeps a future caller's mistake loud instead of silently corrupting its sampling.
    if (pitch < visible_w) {
        fprintf(stderr,
                "[avp] REFUSING AvPlayerVideoEx: pitch %u < visible width %u -- the staged surface and "
                "the published extent disagree; frame not published\n",
                pitch, visible_w);
        return false;
    }
    out.width = pitch;                 // the extent the crop offsets trim
    out.height = visible_h;            // prosper stages no padded rows
    out.aspect = visible_h ? static_cast<float>(visible_w) / static_cast<float>(visible_h) : 0.0f;
    out.crop_right_offset = pitch - visible_w;
    out.pitch = pitch;
    out.luma_bd = 8;
    out.chroma_bd = 8;
    out.framerate = framerate;
    return true;
}

// The byte values the pitch padding is filled with, and why they are not 0.
//
// `Y=0, U=V=0` is NOT black in either BT.601 or BT.709, limited or full range: it converts to roughly
// (0, 136, 0) — mid green. Since #2011 publishes `width` as the CODED extent, a consumer that reads
// `width` and ignores the crop offsets samples the padded columns, and zero-filled padding would draw
// a bright green stripe down the right edge of every movie frame. Limited-range black is `Y=0x10`
// with `U=V=0x80`, which is what the decoders themselves emit for a black frame (measured on
// ArcRunner's intro movie with PROSPER_DUMP_RAWTEX), so padding filled this way is indistinguishable
// from the picture's own black and costs exactly the same memset.
constexpr uint8_t AVP_PAD_LUMA = 0x10;
constexpr uint8_t AVP_PAD_CHROMA = 0x80;

void avp_release_frame_storage(AvpPlayer& player) {
    if (!player.frame.empty())
        gpu::unregister_guest_linear_texture_layout(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(player.frame.data())));
    player.frame.clear();
}

uint8_t* avp_frame_storage(AvpPlayer& player, size_t bytes, uint32_t pitch) {
    if (player.frame.size() != bytes) {
        avp_release_frame_storage(player);
        player.frame.resize(bytes);
    }
    if (player.frame.empty()) return nullptr;
    avp_register_frame_layout(player.frame.data(), player.frame.size(), pitch);
    return player.frame.data();
}

// Native decoders own their dequeue storage and may move or recycle it on the next pull. Copy the
// visible NV12 rows into a guest texture buffer when one exists.  The host-vector fallback preserves
// the old lifetime guarantee for native tests and titles that omit the replacement callbacks.
bool avp_stage_video(AvpPlayer& player, const prosper::video::VideoFrame& frame,
                     bool extended_layout, uint8_t*& data, uint32_t& pitch) {
    if (!frame.y || !frame.uv || frame.width == 0 || frame.height == 0) return false;
    const size_t y_stride = frame.y_stride ? frame.y_stride : frame.width;
    const size_t uv_stride = frame.uv_stride ? frame.uv_stride : frame.width;
    if (y_stride < frame.width || uv_stride < frame.width) return false;
    const size_t uv_rows = (static_cast<size_t>(frame.height) + 1) / 2;
    if (y_stride > SIZE_MAX / frame.height || uv_stride > SIZE_MAX / uv_rows) return false;
    pitch = frame.width;
    if (extended_layout && !avp_video_pitch(frame.width, pitch)) return false;
    size_t bytes = 0;
    if (!avp_nv12_bytes(pitch, frame.height, bytes)) return false;
    uint8_t* dst = nullptr;
    if (!player.texture_frames.empty()) {
        if (bytes > player.texture_frame_bytes) return false;
        dst = player.texture_frames[player.next_texture_frame++ % player.texture_frames.size()];
    } else {
        dst = avp_frame_storage(player, bytes, pitch);
    }
    if (!dst) return false;
    // Copy visible pixels and fill only the padded tail. Filling the entire frame before copying
    // it would double memory traffic on every decoded frame; the crop still must not expose stale
    // decoder bytes if a title deliberately samples beyond the valid image extent. The fill is
    // limited-range BLACK, not zero — see AVP_PAD_LUMA/AVP_PAD_CHROMA above.
    const size_t padding = static_cast<size_t>(pitch) - frame.width;
    for (uint32_t row = 0; row < frame.height; ++row) {
        memcpy(dst + static_cast<size_t>(row) * pitch,
               frame.y + static_cast<size_t>(row) * y_stride, frame.width);
        if (padding)
            memset(dst + static_cast<size_t>(row) * pitch + frame.width, AVP_PAD_LUMA, padding);
    }
    uint8_t* dst_uv = dst + static_cast<size_t>(pitch) * frame.height;
    for (size_t row = 0; row < uv_rows; ++row) {
        memcpy(dst_uv + row * pitch, frame.uv + row * uv_stride, frame.width);
        if (padding) memset(dst_uv + row * pitch + frame.width, AVP_PAD_CHROMA, padding);
    }
    // Callback storage is registered before the first pull. Refresh it here because the basic API
    // retains its historical tight layout while GetVideoDataEx publishes the padded physical pitch.
    avp_register_frame_layout(dst, bytes, pitch);
    data = dst;
    return true;
}

bool avp_stage_synthetic(AvpPlayer& player, bool extended_layout,
                         uint8_t*& data, uint32_t& pitch) {
    pitch = player.width;
    if (extended_layout && !avp_video_pitch(player.width, pitch)) return false;
    size_t need = 0;
    if (!avp_nv12_bytes(pitch, player.height, need)) return false;
    if (!player.texture_frames.empty()) {
        if (need > player.texture_frame_bytes) return false;
        data = player.texture_frames[player.next_texture_frame++ % player.texture_frames.size()];
        if (!data) return false;
    } else {
        data = avp_frame_storage(player, need, pitch);
        if (!data) return false;
    }
    // Same reason as the padding fill: a zero-filled NV12 frame is mid green, not black. The
    // synthetic path fabricates a blank frame, so it should fabricate a legitimately black one.
    const size_t synth_y_bytes = static_cast<size_t>(pitch) * player.height;
    memset(data, AVP_PAD_LUMA, synth_y_bytes);
    memset(data + synth_y_bytes, AVP_PAD_CHROMA, need - synth_y_bytes);
    avp_register_frame_layout(data, need, pitch);
    return true;
}

bool avp_synth_enabled() { return getenv("PROSPER_AVP_SYNTH_FRAMES") != nullptr; }
uint64_t avp_synth_frames() {
    const char* value = getenv("PROSPER_AVP_SYNTH_FRAMES");
    return value ? strtoull(value, nullptr, 0) : 0;
}
uint64_t avp_synth_duration_ms() {
    const char* value = getenv("PROSPER_AVP_DURATION_MS");
    return value ? strtoull(value, nullptr, 0) : avp_synth_frames() * 33;
}
bool avp_log() { static const bool enabled = getenv("PROSPER_AVPLOG") != nullptr; return enabled; }
// One shared monotonic timeline for every [avp] log line — per-site static t0s made cross-
// referencing deliveries against each other guesswork (#2981 FMV measurement).
uint64_t avp_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
}
// PROSPER_AVP_PAUSED_DELIVER — a PRESERVED FALSIFICATION LEVER, NOT A FEATURE, AND NOT A FIX.
// It MUST stay off by default; do not "enable" it and do not promote it into the paused contract.
//
// The shipped contract is the one below: while the guest holds a player paused, GetVideoData/Ex
// deliver nothing. Turning this lever on makes them keep handing out already-decoded frames
// instead. It exists only so the A/B that KILLED a hypothesis stays reproducible, exactly as
// PROSPER_UD_TAIL_ALIGN does for the user-data-tail hypothesis (see CLAUDE.md).
//
// The dead hypothesis (#1599, Asterix & Obelix: Babylon Mission PPSA30490): "the title is stuck in
// Unity's video splash because pause gates frame delivery." Measured, the lever is real but is not
// the cause — it collapses the futile pull census from 14,546 calls to 14 and delivers two further
// frames after the guest's seek, yet sceAvPlayerResume is still never called, all AvPlayer traffic
// then stops, and the output stays uniformly black (max_rgb=0). The actual blocker was that
// sceAvPlayerJumpToTime (NID XC9wM+xULz8) was unimplemented, so the dispatcher's default 0 was read
// as a successful seek: #1949, and prosper/docs/ASTERIX_BABYLON_STATUS.md § Ruled out.
//
// #1949 has since landed and the title now reaches its title screen. Note carefully what that did
// and did not vindicate: a *seek* legitimately publishes its landing frame while paused, and
// s_avp_getvideodata[ex] opens a one-frame window for exactly that (AvpPlayer::seek_deliver). This
// blanket lever is still not that, is still not a fix, and must stay off.
bool avp_paused_deliver() {
    static const bool enabled = getenv("PROSPER_AVP_PAUSED_DELIVER") != nullptr;  // default: false
    return enabled;
}

// ---- the media clock (#1973) ------------------------------------------------------------------
//
// prosper's host decode backends are CONSUMER-DRIVEN: the decode worker fills a small bounded frame
// queue and blocks while it is full, so the decoder only advances when someone pulls. That makes
// "the decoder finished" — until now the player's ONLY end-of-playback signal — unreachable for a
// guest that starts a movie and then never collects a frame. The queue nobody drains never empties,
// VideoBackend::eof() never turns true, and sceAvPlayerIsActive answers 1 forever.
//
// PPSA27624 (Bendy and the Dark Revival) does exactly that: measured on ff72e77c, 14,843
// sceAvPlayerIsActive polls from one call site, ZERO sceAvPlayerGetVideoData[Ex] calls, and no STOP
// event, for a 15.1 s source. The title waits on a movie that cannot end.
//
// A real player is CLOCKED, not consumer-driven. The console's playback timeline runs whether or not
// the application collects frames, so a movie nobody watches still finishes when its media does.
// Model exactly that and nothing more: when NOTHING has asked this player for a video frame for as
// long as the remainder of the media would have taken to play, the media has played out. That is a
// statement about the source's own timeline, not a guess — which is why it may only be made when
// there is a real host decode session with a duration the container actually reported. A synthetic
// session has its own frame budget, and an unknown duration is precisely the case where prosper must
// NOT claim to know when the movie ends.
//
// The "nothing has asked" input is what keeps this off every title that works today. A consumer
// resets the window on EVERY GetVideoData[Ex] call, including calls that return no frame, so a
// starved consumer, or one running far below real time because prosper renders slowly, still ends
// its movie the way it does now: by the queue draining after the decoder finished. This clock can
// only fire where no consumer exists at all — the one case the drain path cannot resolve.
//
// REJECTED ALTERNATIVE, so nobody re-derives it: making video enqueue non-blocking the way audio
// already is (drop the oldest frame when the queue is full) — #1973's own suggested fix. It does let
// decode reach EOF, but it also removes the ONLY thing pacing a movie for every title that works
// today. The decoder would run the whole clip at CPU speed, dropping all but the last few frames, so
// a consuming title's 15-second cutscene becomes six frames and an immediate STOP. The queue's
// backpressure is correct for a decoder; what was missing was a player that knows when its media is
// over. Nothing in the decode backends is touched by this fix, and the same reasoning rules out
// running decode ahead to EOF "bounded by the queue cap" — the bound IS the frame dropping.
//
// CONFIDENCE: HIGH that an unconsumed source must reach the end of its own media. CONFIDENCE: MED on
// the guard band below, which is slop for scheduling rather than a modelled console property.
constexpr uint64_t kAvpMediaClockGraceMs = 250;

bool avp_media_played_out(const AvpPlayer& p, std::chrono::steady_clock::time_point now,
                          uint64_t& idle_ms, uint64_t& remaining_ms) {
    if (!p.backend || p.backend_id < 0 || p.duration_ms == 0) return false;
    remaining_ms = p.duration_ms > p.last_ts_ms ? p.duration_ms - p.last_ts_ms : 0;
    const auto idle =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - p.video_request_at).count();
    idle_ms = idle > 0 ? static_cast<uint64_t>(idle) : 0;
    return idle_ms >= remaining_ms + kAvpMediaClockGraceMs;
}

#if defined(__linux__)
// Linux import stubs swap the guest %fs to the host TCB before entering an HLE handler.  AvPlayer
// event callbacks go the other way and are guest code, so they must run with the caller's guest TCB.
// The entry shims below recover that TCB from the import-stub frame and publish it only for the
// dynamic extent of the HLE call.  Nested AvPlayer calls from the callback preserve the outer value.
thread_local uint64_t t_avp_callback_guest_fs = 0;
struct AvpEntryGuestFsScope {
    uint64_t previous;
    explicit AvpEntryGuestFsScope(uint64_t guest_fs)
        : previous(t_avp_callback_guest_fs) { t_avp_callback_guest_fs = guest_fs; }
    ~AvpEntryGuestFsScope() { t_avp_callback_guest_fs = previous; }
};
struct AvpCallbackGuestFsScope {
    uint64_t saved = 0;
    bool active = false;
    explicit AvpCallbackGuestFsScope(uint64_t guest_fs) {
        if (guest_fs) {
            __asm__ volatile("rdfsbase %0" : "=r"(saved));
            __asm__ volatile("wrfsbase %0" : : "r"(guest_fs));
            active = true;
        }
    }
    ~AvpCallbackGuestFsScope() {
        if (active) __asm__ volatile("wrfsbase %0" : : "r"(saved));
    }
};
#endif

// ---- guest file-replacement reader (#1955) ---------------------------------------------------
// SceAvPlayerFileReplacement is {objectPointer, open, close, readOffset, size} — the order this file
// already declares. Signatures follow the published sceAvPlayer contract: open(obj, path) and
// readOffset(obj, buffer, position, length) return a signed count/status, size(obj) returns the
// media length. CONFIDENCE: MED-HIGH — struct order is confirmed by the ABI static_asserts above and
// by PPSA27624 supplying all four entries, but no live title has yet exercised the call semantics.
using AvpOpenFileCb = int32_t (PROSPER_SYSV_ABI *)(void*, const char*);
using AvpCloseFileCb = int32_t (PROSPER_SYSV_ABI *)(void*);
using AvpReadOffsetFileCb = int32_t (PROSPER_SYSV_ABI *)(void*, uint8_t*, uint64_t, uint32_t);
using AvpSizeFileCb = uint64_t (PROSPER_SYSV_ABI *)(void*);

bool avp_has_file_replacement(const AvpFileReplace& file) {
    return file.open && file.close && file.read_offset && file.size;
}

// A table with SOME entries set but not all four. prosper cannot use it — calling through the null
// member would fault — but it also must not treat it as absence, because absence and "the guest DID
// ask us to read through it and we declined" have opposite meanings for the bytes that get demuxed.
// Split out so the caller can say which one it saw. CONFIDENCE: HIGH (this is a property of the
// pointers prosper was handed, not an inference about the published contract).
bool avp_has_partial_file_replacement(const AvpFileReplace& file) {
    const bool any = file.open || file.close || file.read_offset || file.size;
    return any && !avp_has_file_replacement(file);
}

int32_t avp_file_open(const AvpFileReplace& file, const char* path) {
#if defined(_WIN32)
    return (int32_t)prosper_call_guest_sysv4(reinterpret_cast<uint64_t>(file.open),
                                             reinterpret_cast<uint64_t>(file.obj),
                                             reinterpret_cast<uint64_t>(path), 0, 0);
#else
#if defined(__linux__)
    AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
#endif
    return ((AvpOpenFileCb)file.open)(file.obj, path);
#endif
}

int32_t avp_file_close(const AvpFileReplace& file) {
#if defined(_WIN32)
    return (int32_t)prosper_call_guest_sysv4(reinterpret_cast<uint64_t>(file.close),
                                             reinterpret_cast<uint64_t>(file.obj), 0, 0, 0);
#else
#if defined(__linux__)
    AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
#endif
    return ((AvpCloseFileCb)file.close)(file.obj);
#endif
}

uint64_t avp_file_size(const AvpFileReplace& file) {
#if defined(_WIN32)
    return prosper_call_guest_sysv4(reinterpret_cast<uint64_t>(file.size),
                                    reinterpret_cast<uint64_t>(file.obj), 0, 0, 0);
#else
#if defined(__linux__)
    AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
#endif
    return ((AvpSizeFileCb)file.size)(file.obj);
#endif
}

int32_t avp_file_read(const AvpFileReplace& file, uint8_t* buffer, uint64_t position,
                      uint32_t length) {
#if defined(_WIN32)
    return (int32_t)prosper_call_guest_sysv4(reinterpret_cast<uint64_t>(file.read_offset),
                                             reinterpret_cast<uint64_t>(file.obj),
                                             reinterpret_cast<uint64_t>(buffer), position, length);
#else
#if defined(__linux__)
    AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
#endif
    return ((AvpReadOffsetFileCb)file.read_offset)(file.obj, buffer, position, length);
#endif
}

void* avp_allocate_texture(const AvpMemAllocator& memory, uint32_t align, uint32_t size) {
    if (!memory.allocate_texture) return nullptr;
#if defined(_WIN32)
    return reinterpret_cast<void*>(static_cast<uintptr_t>(prosper_call_guest_sysv4(
        reinterpret_cast<uint64_t>(memory.allocate_texture),
        reinterpret_cast<uint64_t>(memory.obj), align, size, 0)));
#elif defined(__linux__)
    AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
    return memory.allocate_texture(memory.obj, align, size);
#else
    return memory.allocate_texture(memory.obj, align, size);
#endif
}

void avp_deallocate_texture(const AvpMemAllocator& memory, void* allocation) {
    if (!memory.deallocate_texture || !allocation) return;
#if defined(_WIN32)
    prosper_call_guest_sysv4(reinterpret_cast<uint64_t>(memory.deallocate_texture),
                             reinterpret_cast<uint64_t>(memory.obj),
                             reinterpret_cast<uint64_t>(allocation), 0, 0);
#elif defined(__linux__)
    AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
    memory.deallocate_texture(memory.obj, allocation);
#else
    memory.deallocate_texture(memory.obj, allocation);
#endif
}

void avp_release_textures(const AvpMemAllocator& memory,
                          const std::vector<uint8_t*>& textures) {
    for (auto* texture : textures) {
        gpu::unregister_guest_linear_texture_layout(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(texture)));
        avp_deallocate_texture(memory, texture);
    }
}

bool avp_build_textures(const AvpMemAllocator& memory, int32_t requested,
                        uint32_t width, uint32_t height,
                        std::vector<uint8_t*>& textures, size_t& texture_bytes) {
    const bool have_allocate = memory.allocate_texture != nullptr;
    const bool have_deallocate = memory.deallocate_texture != nullptr;
    if (!have_allocate && !have_deallocate) return true; // legacy/test fallback
    uint32_t pitch = 0;
    if (!have_allocate || !have_deallocate || !avp_video_pitch(width, pitch) ||
        !avp_nv12_bytes(pitch, height, texture_bytes) || texture_bytes > UINT32_MAX)
        return false;

    // The public API permits a caller-selected framebuffer count.  Zero requests the normal double
    // buffer; bound hostile values so a malformed guest structure cannot fan out allocations.
    const int count = std::clamp(requested > 0 ? requested : 2, 2, 16);
    constexpr uint32_t alignment = 0x100;
    textures.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        void* allocation = avp_allocate_texture(memory, alignment,
                                                static_cast<uint32_t>(texture_bytes));
        if (!allocation) {
            avp_release_textures(memory, textures);
            textures.clear();
            texture_bytes = 0;
            return false;
        }
        auto* texture = static_cast<uint8_t*>(allocation);
        textures.push_back(texture);
        avp_register_frame_layout(texture, texture_bytes, pitch);
    }
    if (avp_log()) {
        fprintf(stderr,
                "[avp] guest texture buffers requested=%d allocated=%d align=%u bytes=%zu first=%p\n",
                requested, count, alignment, texture_bytes,
                textures.empty() ? nullptr : textures.front());
    }
    return true;
}

// #1955 discriminator: report the guest's file-replacement table verbatim. A title that stores media
// inside a container file can only express the byte range through these callbacks, so which titles
// supply the table — and whether all four entries are there — is a measurement, not an assumption.
// Print all four entries (and "absent" when the table is null); avp_add_source reports what it then
// did with them.
void avp_log_file_replacement(const char* entry, const AvpFileReplace& file) {
    const bool present = file.open || file.close || file.read_offset || file.size;
    fprintf(stderr,
            "[avp] %s file-replacement=%s obj=%p open=%p close=%p read-offset=%p size=%p\n",
            entry, present ? "present" : "absent", file.obj, file.open, file.close,
            file.read_offset, file.size);
}

// Fire the guest event callback. Caller MUST NOT hold g_avp_mx (the callback re-enters AvPlayer HLE).
// `data` is the callback's fourth argument: null for the state events, and a pointer to the
// notification code for the warning channel (see AVP_WARNING_ID). The guest dereferences it during
// the call, so it may live on this frame -- but it must not outlive the call, and nothing here keeps
// it.
void avp_fire_data(void* obj, AvpEventCb cb, uint32_t ev, int32_t source_id, const void* data) {
    if (!cb) return;
    if (svclog() || avp_log()) fprintf(stderr, "[avp] -> event 0x%02x\n", ev);
#if defined(_WIN32)
    prosper_call_guest_sysv4((uint64_t)(uintptr_t)cb, (uint64_t)(uintptr_t)obj, ev,
                             (uint64_t)(uint32_t)source_id, (uint64_t)(uintptr_t)data);
#elif defined(__linux__)
    {
        AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
        cb(obj, ev, source_id, const_cast<void*>(data));
    }
#else
    cb(obj, ev, source_id, const_cast<void*>(data));
#endif
    if (avp_log()) fprintf(stderr, "[avp] <- event 0x%02x\n", ev);
}
void avp_fire(void* obj, AvpEventCb cb, uint32_t ev) { avp_fire_data(obj, cb, ev, 0, nullptr); }
} // namespace

HLE(s_avplayer_init) {   // AvPlayerHandle sceAvPlayerInit(AvPlayerInitData*)
    svc_log("sceAvPlayerInit", a0,a1,a2,a3,a4,a5);
    uint64_t h = g_handle.fetch_add(1);
    AvpPlayer p;
    if (auto* d = (const AvpInitData*)PW(a0)) {
        p.memory = d->memory; p.file = d->file; p.num_fb = d->num_fb;
        p.ev_obj = d->event.obj; p.ev_cb = (AvpEventCb)d->event.event_callback;
        p.auto_start = d->auto_start != 0;
        if (avp_log()) {
            fprintf(stderr,
                "[avp] init memory=%p alloc=%p free=%p texture=%p texture-free=%p num-fb=%d auto=%d\n",
                p.memory.obj, (void*)p.memory.allocate, (void*)p.memory.deallocate,
                (void*)p.memory.allocate_texture, (void*)p.memory.deallocate_texture,
                p.num_fb, (int)p.auto_start);
            avp_log_file_replacement("init", p.file);
        }
    }
    { std::lock_guard<std::mutex> lk(g_avp_mx); g_avp[h] = std::move(p); }
    return h;   // non-NULL SceAvPlayerHandle
}
// sceAvPlayerInitEx returns an int32 error code (0 = success) and writes the handle to *out — a
// DIFFERENT ABI from sceAvPlayerInit (which returns the handle). Returning the handle here made the
// game read it as an error and abort the intro (live-captured, PPSA02664). CONFIDENCE: HIGH.
HLE(s_avplayer_initex) {   // s32 sceAvPlayerInitEx(const AvPlayerInitDataEx*, AvPlayerHandle* out)
    svc_log("sceAvPlayerInitEx", a0,a1,a2,a3,a4,a5);
    uint64_t h = g_handle.fetch_add(1);
    AvpPlayer p;
    if (auto* d = (const AvpInitDataEx*)PW(a0)) {
        p.memory = d->memory; p.file = d->file; p.num_fb = d->num_fb;
        p.ev_obj = d->event.obj; p.ev_cb = (AvpEventCb)d->event.event_callback;
        p.auto_start = d->auto_start != 0;
        if (avp_log()) {
            fprintf(stderr,
                "[avp] init-ex memory=%p alloc=%p free=%p texture=%p texture-free=%p num-fb=%d auto=%d\n",
                p.memory.obj, (void*)p.memory.allocate, (void*)p.memory.deallocate,
                (void*)p.memory.allocate_texture, (void*)p.memory.deallocate_texture,
                p.num_fb, (int)p.auto_start);
            avp_log_file_replacement("init-ex", p.file);
        }
    }
    { std::lock_guard<std::mutex> lk(g_avp_mx); g_avp[h] = std::move(p); }
    if (a1) *(uint64_t*)PW(a1) = h;
    return 0;
}
HLE(s_avplayer_isactive) {   // bool sceAvPlayerIsActive(AvPlayerHandle)
    svc_log("sceAvPlayerIsActive", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool fire_stop = false; uint64_t active = 0;
    bool played_out = false; uint64_t clock_idle_ms = 0, clock_remaining_ms = 0, clock_duration_ms = 0;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0;
        AvpPlayer& p = it->second;
        if (p.playing) {
            const auto now = std::chrono::steady_clock::now();
            // A paused player is active and presents nothing; its media clock is restarted by
            // sceAvPlayerResume, the only exit from this state, so no time is accumulated here.
            if (p.paused) return 1;
            p.active_poll++;
            bool still = p.backend && p.backend_id >= 0
                             ? !p.backend->eof(p.backend_id)
                             : p.synthetic && p.active_poll < avp_synth_frames();
            // #1973: a consumer-driven decoder cannot report the end of a source nobody consumes.
            if (still && avp_media_played_out(p, now, clock_idle_ms, clock_remaining_ms)) {
                still = false;
                played_out = true;
                clock_duration_ms = p.duration_ms;
            }
            if (still) { active = 1; }
            else if (!p.stop_fired) { p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb; }
        }
    }
    if (played_out) {
        // Loud by design, like the JumpToTime failure above: playback that ends on the clock rather
        // than by delivering its frames is a title consuming nothing, and must never read as an
        // ordinary drain in a default run.
        fprintf(stderr,
                "[avp] handle=0x%llx: the %llu ms source played out on the media clock -- nothing "
                "requested a video frame for %llu ms and %llu ms of the media remained (#1973)\n",
                (unsigned long long)a0, (unsigned long long)clock_duration_ms,
                (unsigned long long)clock_idle_ms, (unsigned long long)clock_remaining_ms);
    }
    if (fire_stop) avp_fire(obj, cb, AVP_STOP);   // playback complete -> game advances past the video
    return active;
}
HLE(s_avp_postinit)   { svc_log("sceAvPlayerPostInit", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_avp_setlogcb)   { svc_log("sceAvPlayerSetLogCallback", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_avp_streamcount) {   // s32 sceAvPlayerStreamCount(handle)
    svc_log("sceAvPlayerStreamCount", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0);
    int count = it != g_avp.end() && it->second.have_source
                    ? (it->second.has_audio ? 2 : 1) : 0;
    if (avp_log()) fprintf(stderr, "[avp] stream-count handle=0x%llx -> %d\n",
                           (unsigned long long)a0, count);
    return count;
}
HLE(s_avp_getstreaminfo) { // s32 sceAvPlayerGetStreamInfo(handle, stream_id, out)
    svc_log("sceAvPlayerGetStreamInfo", a0,a1,a2,a3,a4,a5);
    auto* out = (AvpStreamInfo*)PW(a2); if (!out) return 0x806a0001ull;
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0); if (it == g_avp.end() || !it->second.have_source) return 0x806a0001ull;
    const AvpPlayer& p = it->second;
    if (a1 >= (p.has_audio ? 2u : 1u)) return 0x806a0001ull;
    *out = {}; out->type = a1 == 0 ? 1u : 2u;
    if (a1 == 0) {
        // The basic AvPlayerVideo has NO pitch and NO crop fields, so its `width` must be the VISIBLE
        // extent — there is no channel here through which a consumer could subtract padding. That is
        // why it deliberately differs from the Ex struct's coded `width` (see avp_fill_video_ex): the
        // two structs answer the same question through different field sets, and the basic path also
        // stages tight rows to match.
        out->details.video.width = p.width; out->details.video.height = p.height;
        out->details.video.aspect = (float)p.width / (float)p.height;
    } else {
        out->details.audio.channels = (uint16_t)p.audio_channels;
        out->details.audio.sample_rate = p.audio_rate;
        out->details.audio.size = 1024 * p.audio_channels * sizeof(int16_t);
    }
    out->duration = p.duration_ms;
    return 0;
}
HLE(s_avp_getstreaminfoex) { // s32 sceAvPlayerGetStreamInfoEx(handle, stream_id, out)
    svc_log("sceAvPlayerGetStreamInfoEx", a0,a1,a2,a3,a4,a5);
    auto* out = (AvpStreamInfoEx*)PW(a2); if (!out) return 0x806a0001ull;
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0); if (it == g_avp.end() || !it->second.have_source) return 0x806a0001ull;
    const AvpPlayer& p = it->second; const uint64_t caller_size = out->this_size;
    if (a1 >= (p.has_audio ? 2u : 1u)) return 0x806a0001ull;
    *out = {}; out->this_size = caller_size; out->type = a1 == 0 ? 1u : 2u;
    if (a1 == 0) {
        uint32_t pitch = 0;
        if (!avp_video_pitch(p.width, pitch)) return 0x806a0001ull;
        if (!avp_fill_video_ex(out->details.video, p.width, p.height, pitch, p.fps))
            return 0x806a0001ull;
    } else {
        out->details.audio.channels = (uint16_t)p.audio_channels;
        out->details.audio.sample_rate = p.audio_rate;
        out->details.audio.size = 1024 * p.audio_channels * sizeof(int16_t);
    }
    out->duration = p.duration_ms;
    return 0;
}
HLE(s_avp_stream_ok) { svc_log("sceAvPlayerStreamControl", a0,a1,a2,a3,a4,a5); return 0; }
// Begin a source: resolve/open it, fire READY, and (auto_start) begin playback with PLAY.
static uint64_t avp_add_source(uint64_t handle, const char* guest_path) {
    if (!guest_path || !*guest_path) return 0x806a0001ull;

    if (avp_log()) {
        fprintf(stderr, "[avp] add source request handle=0x%llx guest='%s'\n",
                (unsigned long long)handle, guest_path);
        fflush(stderr);
    }

    prosper::video::VideoBackend* old_backend = nullptr;
    int old_backend_id = -1;
    AvpMemAllocator memory{};
    AvpFileReplace file{};
    int32_t requested_textures = 0;
    std::vector<uint8_t*> old_textures;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(handle); if (it == g_avp.end()) return 0x80000000ull;
        AvpPlayer& p = it->second;
        file = p.file;
        old_backend = p.backend;
        old_backend_id = p.backend_id;
        memory = p.memory;
        requested_textures = p.num_fb;
        old_textures = std::move(p.texture_frames);
        avp_release_frame_storage(p);
        p.texture_frame_bytes = 0;
        p.next_texture_frame = 0;
        p.have_source = false; p.synthetic = false; p.playing = false; p.paused = false;
        p.backend = nullptr; p.backend_id = -1;
    }
    if (old_backend && old_backend_id >= 0) old_backend->close(old_backend_id);
    avp_release_textures(memory, old_textures);

    // #1955: when the guest supplies a file-replacement table, IT decides which bytes of the named
    // file are the media. A title that stores a clip at an offset inside a container has no other way
    // to express that — SceAvPlayerSourceDetails carries no byte range — so opening the host path at
    // offset 0 demuxes the wrong bytes. Read the media through the guest's own reader instead.
    //
    // The read happens HERE, on the guest thread that called sceAvPlayerAddSource(Ex), because these
    // callbacks are guest code and need that thread's TCB; the decode worker is a prosper-owned host
    // thread and must never call them. The whole media is therefore buffered up front.
    //
    // EVERY branch below that declines the guest's reader falls back to opening the host path at
    // offset 0 — which is exactly the #1955 defect when the guest asked us to read a byte range. So
    // none of them may be silent: the messages here are deliberately NOT gated on PROSPER_AVPLOG,
    // because a default run has to be able to say which branch it took. (The success line stays
    // gated; it is the only one that is not a degradation.)
    std::vector<uint8_t> guest_media;
    if (avp_has_partial_file_replacement(file)) {
        // Not the same thing as no table. The guest asked to be read through, and prosper cannot
        // oblige a table it would have to call a null member of, so it is about to demux the host
        // file from offset 0 — the wrong bytes whenever the media is embedded.
        fprintf(stderr,
                "[avp] the guest file-replacement table for '%s' is INCOMPLETE "
                "(open=%p close=%p read-offset=%p size=%p); ignoring it and falling back to the "
                "host path (#1955)\n",
                guest_path, file.open, file.close, file.read_offset, file.size);
    }
    if (avp_has_file_replacement(file)) {
        const int32_t opened = avp_file_open(file, guest_path);
        const uint64_t media_bytes = opened >= 0 ? avp_file_size(file) : 0;
        if (opened < 0) {
            fprintf(stderr,
                    "[avp] guest file-replacement open('%s') failed: %d -> falling back to the host "
                    "path (#1955)\n", guest_path, (int)opened);
        } else if (avp_log()) {
            fprintf(stderr, "[avp] guest file-replacement reader: open('%s') -> %d size=%llu bytes\n",
                    guest_path, (int)opened, (unsigned long long)media_bytes);
        }
        if (opened >= 0) {
            // Bound the buffer: a guest that answers with a whole multi-hundred-megabyte container
            // is not describing a clip, and silently allocating that is worse than saying so.
            constexpr uint64_t kMaxGuestMediaBytes = 256ull * 1024 * 1024;
            if (media_bytes == 0 || media_bytes > kMaxGuestMediaBytes) {
                fprintf(stderr,
                        "[avp] guest file-replacement reader for '%s' reports %llu bytes; refusing to "
                        "buffer it and falling back to the host path (#1955)\n",
                        guest_path, (unsigned long long)media_bytes);
            } else {
                // The cap bounds the request, not the machine: a host that cannot spare the buffer
                // still throws here, and an uncaught bad_alloc inside an HLE handler ends the
                // process rather than the movie. Degrade to the same loud fallback as every other
                // branch. (Noted in #1975's review of #1974; it is a failure mode of exactly this
                // block, so it is fixed here rather than left to the buffering optimisation.)
                uint64_t position = 0;
                bool read_ok = true;
                try {
                    guest_media.resize(static_cast<size_t>(media_bytes));
                } catch (const std::bad_alloc&) {
                    fprintf(stderr,
                            "[avp] could not allocate %llu bytes for the guest file-replacement "
                            "media of '%s'; falling back to the host path (#1955)\n",
                            (unsigned long long)media_bytes, guest_path);
                    read_ok = false;
                }
                // `read_ok &&` is load-bearing, not defensive: without it a failed resize leaves an
                // EMPTY vector and the loop hands `data() + 0` to the guest's reader to write into.
                while (read_ok && position < media_bytes) {
                    const uint64_t remaining = media_bytes - position;
                    const uint32_t chunk = static_cast<uint32_t>(
                        std::min<uint64_t>(remaining, 1u << 20));
                    const int32_t got = avp_file_read(file, guest_media.data() + position, position,
                                                      chunk);
                    if (got <= 0) {
                        fprintf(stderr,
                                "[avp] guest file-replacement read('%s') failed at offset %llu: %d "
                                "-> falling back to the host path (#1955)\n",
                                guest_path, (unsigned long long)position, (int)got);
                        read_ok = false;
                        break;
                    }
                    // A count LARGER than the buffer space offered is the one answer that must never
                    // be believed. Taken at face value it advances `position` past `media_bytes`, the
                    // loop exits with `read_ok` still true, and the zero-filled tail of the buffer is
                    // handed to the demuxer AS IF it were the complete media — a silent truncation
                    // that surfaces as a corrupt stream far from here. It also means the callee may
                    // already have written past the space it was given, which prosper cannot undo but
                    // must not build on. Refuse the media and say so.
                    if (static_cast<uint32_t>(got) > chunk) {
                        fprintf(stderr,
                                "[avp] guest file-replacement read('%s') at offset %llu returned %d "
                                "for a %u-byte request; refusing the media and falling back to the "
                                "host path (#1955)\n",
                                guest_path, (unsigned long long)position, (int)got, chunk);
                        read_ok = false;
                        break;
                    }
                    position += static_cast<uint64_t>(got);
                }
                // With the over-count refused above, a loop that ran to completion can only have
                // landed exactly on media_bytes. Assert it rather than assume it: a future edit to
                // the arithmetic that stops short would otherwise deliver a short buffer as a
                // complete one. (`read_ok` is already false on the allocation and read failures, so
                // this reports only the case those two did not.)
                if (read_ok && position != media_bytes) {
                    fprintf(stderr,
                            "[avp] guest file-replacement read('%s') assembled %llu of %llu bytes; "
                            "refusing the media and falling back to the host path (#1955)\n",
                            guest_path, (unsigned long long)position,
                            (unsigned long long)media_bytes);
                    read_ok = false;
                }
                if (!read_ok) guest_media.clear();
            }
            avp_file_close(file);
        }
    }

    const std::string host_path = resolve_guest_path(guest_path);
    auto* selected_backend = prosper::video::backend();
    if (avp_log()) {
        fprintf(stderr, "[avp] opening source host='%s' backend=%d guest-media=%zu bytes\n",
                host_path.c_str(), selected_backend != nullptr, guest_media.size());
        fflush(stderr);
    }
    prosper::video::StreamInfo stream{};
    int backend_id = -1;
    if (selected_backend && !guest_media.empty()) {
        backend_id = selected_backend->open_memory(host_path, guest_media.data(),
                                                   guest_media.size());
        // A backend with no in-memory path, or media the demuxer rejects, must not silently lose the
        // titles that work today: fall back to the host file exactly as before.
        if (backend_id < 0)
            fprintf(stderr,
                    "[avp] the guest-supplied media for '%s' could not be demuxed; falling back to "
                    "the host path (#1955)\n", guest_path);
    }
    if (backend_id < 0 && selected_backend) backend_id = selected_backend->open(host_path);
    if (backend_id >= 0 && !selected_backend->info(backend_id, stream)) {
        selected_backend->close(backend_id);
        backend_id = -1;
    }
    // #1105 test lever: force a total source-open failure to exercise the graceful immediate-EOF path
    // (an undecodable source) without needing a corrupt media file. Off by default.
    if (backend_id >= 0 && getenv("PROSPER_AVP_FORCE_FAIL")) {
        selected_backend->close(backend_id);
        backend_id = -1;
    }
    const bool synthetic = backend_id < 0 && avp_synth_enabled();
    if (backend_id < 0 && !synthetic) {
        // #1105: a source that cannot be opened (corrupt file, missing codec, demux error) must not hang
        // the title. Returning an error with no lifecycle event leaves a title whose intro state machine
        // waits for the movie to finish deadlocked in its black scene (the #320 shape). Instead set up an
        // EMPTY source and drive it to a graceful immediate EOF: fire AVP_READY (so the title starts it),
        // and the first IsActive/GetVideoData poll finds no backend and no synthetic frames (:506 / :760)
        // and fires AVP_STOP — the game skips the unplayable movie and proceeds.
        if (avp_log()) fprintf(stderr, "[avp] source open failed guest='%s' host='%s' backend=%d -> "
                               "graceful skip (empty source, immediate EOF)\n",
                               guest_path, host_path.c_str(), selected_backend != nullptr);
        void* obj = nullptr; AvpEventCb cb = nullptr; bool play = false, player_missing = false;
        {
            std::lock_guard<std::mutex> lk(g_avp_mx);
            auto it = g_avp.find(handle);
            if (it == g_avp.end()) {
                player_missing = true;
            } else {
                AvpPlayer& p = it->second;
                p.guest_path = guest_path;
                p.have_source = true; p.synthetic = false;
                p.poll = 0; p.audio_poll = 0; p.active_poll = 0; p.last_ts_ms = 0;
                p.paused = false; p.stop_fired = false; p.seek_deliver = false;
                p.video_request_at = std::chrono::steady_clock::now();
                p.play_start_wall = p.video_request_at; p.play_start_pts = 0; p.gate_started = false;
                p.texture_frames.clear(); p.texture_frame_bytes = 0; p.next_texture_frame = 0;
                p.backend = nullptr; p.backend_id = -1;
                // Non-zero placeholder dims/fps: a title that queries GetStreamInfo up-front (before it
                // discovers the immediate EOF) then gets a benign aspect (w/h) and never a 0x0 surface or
                // a divide-by-zero. The source still delivers zero frames and EOFs on the first poll.
                p.width = 1920; p.height = 1080; p.fps = 30.0f;
                p.has_audio = false; p.audio_channels = 0; p.audio_rate = 0; p.duration_ms = 0;
                obj = p.ev_obj; cb = p.ev_cb;
                if (p.auto_start) { p.playing = true; play = true; }
            }
        }
        if (player_missing) return 0x80000000ull;
        avp_fire(obj, cb, AVP_READY);
        if (play) avp_fire(obj, cb, AVP_PLAY);
        return 0;
    }

    const uint32_t video_width = synthetic ? 1920u : stream.width;
    const uint32_t video_height = synthetic ? 1080u : stream.height;
    std::vector<uint8_t*> textures;
    size_t texture_bytes = 0;
    if (!avp_build_textures(memory, requested_textures, video_width, video_height,
                            textures, texture_bytes)) {
        if (backend_id >= 0) selected_backend->close(backend_id);
        if (avp_log()) fprintf(stderr,
            "[avp] source rejected: guest texture allocation failed (%ux%u num-fb=%d)\n",
            video_width, video_height, requested_textures);
        return 0x806a0001ull;
    }

    void* obj = nullptr; AvpEventCb cb = nullptr; bool play = false, player_missing = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(handle);
        if (it == g_avp.end()) {
            player_missing = true;
        } else {
            AvpPlayer& p = it->second;
            p.guest_path = guest_path;
            p.have_source = true; p.synthetic = synthetic;
            p.poll = 0; p.audio_poll = 0; p.active_poll = 0; p.last_ts_ms = 0; p.paused = false;
            p.stop_fired = false; p.seek_deliver = false;
            p.video_request_at = std::chrono::steady_clock::now();
            p.play_start_wall = p.video_request_at; p.play_start_pts = 0; p.gate_started = false;
            p.texture_frames = std::move(textures);
            p.texture_frame_bytes = texture_bytes;
            p.next_texture_frame = 0;
            p.backend = backend_id >= 0 ? selected_backend : nullptr;
            p.backend_id = backend_id;
            if (synthetic) {
                p.width = 1920; p.height = 1080; p.fps = 30.0f;
                p.has_audio = true; p.audio_channels = 2; p.audio_rate = 48000;
                p.duration_ms = avp_synth_duration_ms();
            } else {
                p.width = stream.width; p.height = stream.height; p.fps = stream.fps;
                p.has_audio = stream.has_audio; p.audio_channels = stream.audio_channels;
                p.audio_rate = stream.audio_rate; p.duration_ms = stream.duration_us / 1000;
            }
            obj = p.ev_obj; cb = p.ev_cb;
            if (p.auto_start) { p.playing = true; play = true; }
        }
    }
    if (player_missing) {
        if (backend_id >= 0) selected_backend->close(backend_id);
        avp_release_textures(memory, textures);
        return 0x80000000ull;
    }
    avp_fire(obj, cb, AVP_READY);
    if (play)  avp_fire(obj, cb, AVP_PLAY);
    // The duration is the media clock's only input besides delivered frames (#1973); log it so a run
    // shows whether this source has a timeline prosper can reason about at all.
    if (avp_log()) fprintf(stderr,
                           "[avp] add source handle=0x%llx guest='%s' host='%s' mode=%s auto_start=%d "
                           "duration=%llums\n",
                           (unsigned long long)handle, guest_path, host_path.c_str(),
                           synthetic ? "synthetic-explicit" : "native", (int)play,
                           (unsigned long long)(synthetic ? avp_synth_duration_ms()
                                                          : stream.duration_us / 1000));
    return 0;
}
HLE(s_avp_addsource)   {   // s32 sceAvPlayerAddSource(handle, const char* filename)
    svc_log("sceAvPlayerAddSource", a0,a1,a2,a3,a4,a5);
    return avp_add_source(a0, (const char*)PW(a1));
}
HLE(s_avp_addsourceex) {   // s32 sceAvPlayerAddSourceEx(handle, AvPlayerUriType, AvPlayerSourceDetails*)
    svc_log("sceAvPlayerAddSourceEx", a0,a1,a2,a3,a4,a5);
    const char* path = nullptr;
    if (auto* d = (const AvpSourceDetails*)PW(a2)) path = d->uri.name;
    return avp_add_source(a0, path);
}
HLE(s_avp_start) {   // s32 sceAvPlayerStart(handle) — used when auto_start is false
    svc_log("sceAvPlayerStart", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool play = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0x80000000ull;
        AvpPlayer& p = it->second;
        if (p.have_source && !p.playing && !p.stop_fired) {
            p.playing = true; p.paused = false; play = true; obj = p.ev_obj; cb = p.ev_cb;
            p.video_request_at = std::chrono::steady_clock::now();
        }
    }
    if (play) avp_fire(obj, cb, AVP_PLAY);
    return 0;
}
HLE(s_avp_pause) {   // s32 sceAvPlayerPause(handle)
    svc_log("sceAvPlayerPause", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool notify = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0x806a0001ull;
        if (it->second.playing && !it->second.paused) {
            it->second.paused = true;
            obj = it->second.ev_obj; cb = it->second.ev_cb; notify = true;
        }
    }
    if (notify) avp_fire(obj, cb, AVP_PAUSE);
    return 0;
}
HLE(s_avp_resume) {   // s32 sceAvPlayerResume(handle)
    svc_log("sceAvPlayerResume", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool notify = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0x806a0001ull;
        it->second.seek_deliver = false;   // ordinary delivery resumes; the seek window is over
        if (it->second.playing && it->second.paused) {
            it->second.paused = false;
            obj = it->second.ev_obj; cb = it->second.ev_cb; notify = true;
            // Presentation restarts now; the media clock must measure from here, not from whenever
            // the player was paused.
            it->second.video_request_at = std::chrono::steady_clock::now();
        }
    }
    if (notify) avp_fire(obj, cb, AVP_PLAY);
    return 0;
}
// s32 sceAvPlayerJumpToTime(SceAvPlayerHandle, uint64_t offset_ms) — reposition playback (#1949).
//
// UNITS: milliseconds. PPSA30490 builds the argument as a seconds double scaled by 1000
// (eboot+0x15dc197 .. 0x15dc1ec) and then compares the resulting value directly against the
// millisecond SceAvPlayerFrameInfoEx::timeStamp that sceAvPlayerGetVideoDataEx returns, and against
// sceAvPlayerCurrentTime's millisecond position. CONFIDENCE: HIGH.
//
// WHY EVERY FAILURE PATH RETURNS NON-ZERO: this NID used to be unregistered, so the dispatcher's
// default 0 reached the guest as SCE_OK. PPSA30490 pre-sets its own failure state, tests only
// `eax == 0`, and on zero enters a wait for the player's position to move before it calls
// sceAvPlayerResume. A "successful" seek that moved nothing therefore deadlocked the title in its
// video splash forever, while the guest had a perfectly good `failed to jump while seeking` branch
// it could never reach. A truthful error is strictly better than a silent no-op here.
HLE(s_avp_jumptotime) {
    svc_log("sceAvPlayerJumpToTime", a0,a1,a2,a3,a4,a5);
    const uint64_t target_ms = a1;
    const char* reason = nullptr;
    void* ev_obj = nullptr; AvpEventCb ev_cb = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0);
        if (it == g_avp.end()) {
            reason = "unknown player handle";
        } else {
            AvpPlayer& p = it->second;
            // The backend seek runs under g_avp_mx on purpose: it tears down and recreates the
            // decode worker, and a concurrent GetVideoData/IsActive that observed the intermediate
            // end-of-decode state would fire a spurious AVP_STOP at the guest.
            if (!p.have_source) {
                reason = "no source is attached to this player";
            } else if (!p.backend || p.backend_id < 0) {
                // Includes the #1105 graceful-skip empty source and any headless/synthetic session:
                // there is no decoder to move, so say so instead of reporting a seek that cannot exist.
                reason = "the source has no host decoder to reposition";
            } else if (target_ms > UINT64_MAX / 1000ull) {
                reason = "the requested position overflows a microsecond position";
            } else if (!p.backend->seek(p.backend_id, target_ms * 1000ull)) {
                reason = "the host video backend could not seek this source";
            } else {
                // The player is now at the requested time even though no frame has been pulled yet,
                // so sceAvPlayerCurrentTime must already report it. A seek deliberately does NOT
                // change play/pause state: the observed guest sequence is pause -> jump -> resume.
                // CONFIDENCE: LOW on seeking a player that has already reported AVP_STOP — no title
                // does it, so nothing here revives one; it stays stopped with a repositioned source.
                p.last_ts_ms = target_ms;
                p.seek_deliver = true;
                // Re-anchor the delivery gate at the new position: without this, post-seek
                // frames deliver un-paced against the stale pre-seek anchor until the media
                // clock climbs back over it (#2989 re-review).
                p.play_start_wall = std::chrono::steady_clock::now();
                p.play_start_pts = target_ms;
                p.gate_started = true;
                // The source was just repositioned: the media clock measures the remainder from the
                // new position, starting now.
                p.video_request_at = std::chrono::steady_clock::now();
                ev_obj = p.ev_obj; ev_cb = p.ev_cb;
            }
        }
    }
    if (reason) {
        // A FAILED seek publishes no warning at all, and the reason is NOT that `0x806a0004` would
        // be a lie -- it is this surface's truthful failure signal, and the guests that listen for
        // it are entitled to hear it.
        //
        // The reason is that prosper's failure set above is WIDER than "the seek was refused". It
        // includes "the source has no host decoder to reposition" -- the #1105 graceful-skip empty
        // source and any headless or synthetic session -- which are benign states, not errors the
        // title should latch. And a listener does latch: PPSA25009's event callback
        // (eboot+0x13e46a0) dispatches three warning codes at +0x13e4a53, and its `0x806a0004` arm
        // at +0x13e4a77 stores a pointer into the player object and sets a byte beside it, i.e. a
        // PERSISTENT error state rather than a transient notice. Publishing that code because
        // prosper had no decoder would permanently mark a player the title was merely skipping an
        // unplayable movie on.
        //
        // Narrowing the failure set so the genuinely-refused case could announce itself is a real
        // improvement and is deliberately not attempted here; it needs a title that waits on the
        // failure arm to verify against. (Analysis from review of #3348.)
        //
        // Loud by design (not gated on PROSPER_AVPLOG): a guest-visible seek failure is exactly the
        // kind of gap that must not read as "handled" in a default run.
        fprintf(stderr, "[avp] sceAvPlayerJumpToTime handle=0x%llx target=%llu ms FAILED: %s\n",
                (unsigned long long)a0, (unsigned long long)target_ms, reason);
        return 0x806a0001ull;   // SCE_AVPLAYER_ERROR_INVALID_PARAMS, as used across this surface
    }
    // Repositioning is only half of the contract: the guest is told the seek LANDED through the
    // event callback's warning channel, and until it hears that it is entitled to sit in its own
    // wait. Published outside g_avp_mx, like every other event -- the callback re-enters AvPlayer HLE.
    //
    // Deliberately synchronous rather than deferred to a worker. Two arguments for that are TRUE
    // but cover only part of the population, and neither is what this rests on:
    //
    //   * "the guests arm their wait slot before calling" -- PPSA17337 writes its awaited code at
    //     eboot+0x15400b8, one instruction group before the call at +0x15400c2. Two titles observed.
    //   * "the join is a COUNTING semaphore, so an early post is stored rather than lost" -- true
    //     where it applies: that guest's callback posts with `lock xadd` at eboot+0x153f041 and its
    //     waiter sleeps only when the pre-decrement value was <= 0. But that is ONE binary's
    //     encoding, and it generalises no further than the first argument did. Measured across the
    //     corpus: of the 42 dumps importing this NID, the exact arm-and-join encoding appears in
    //     **11** -- one byte-identical Unity plugin. PPSA25009 is one of the 31 without it.
    //
    // What covers all 42 is weaker in form and stronger in reach, and it is a DEGRADATION bound
    // rather than a safety proof: if a guest's join is an EDGE, an early post is simply lost, and
    // that guest then waits exactly the timeout it already pays today with no notification at all.
    // So publishing synchronously is never worse than the status quo for any importer, and is
    // strictly better for the ones whose join can store it. That is the claim this code makes.
    //
    // SCOPE THE BOUND PRECISELY, because it is narrower than it first reads: it is about the WAIT.
    // An edge-triggered join loses the wake, and the guest then waits out the timeout it already
    // pays today -- but losing the wake does NOT stop the callback BODY from running, and that body
    // is demonstrably not inert. PPSA25009's handler dispatches three warning codes at
    // eboot+0x13e4a53 and its 0x806a00a3 arm at +0x13e4b5e walks a per-stream table doing real work.
    // So side effects inside an edge-triggered guest's callback are OUTSIDE this bound and remain
    // the open residual, which is the same one the PR body records: nobody can say from prosper's
    // code what any of the other importers' handlers do with a notification they have never had.
    //
    // Re-entrancy is outside the bound too, and separately covered rather than overlooked: prosper
    // has always fired guest callbacks synchronously from HLE entry points (see :2360 and :2378,
    // sceAvPlayerPause and sceAvPlayerResume), and PPSA17337's own pause -> jump -> resume sequence
    // already re-enters twice per seek on the pre-existing paths.
    //
    // Publishing here also keeps the notification ordered strictly after the reposition it reports,
    // which a deferred worker could not guarantee. (Population measured in review of #3348.)
    const uint32_t seek_complete = AVP_WARNING_SEEK_COMPLETE;
    avp_fire_data(ev_obj, ev_cb, AVP_WARNING_ID, 0, &seek_complete);
    if (avp_log()) fprintf(stderr, "[avp] jump-to-time handle=0x%llx target=%llu ms -> ok\n",
                           (unsigned long long)a0, (unsigned long long)target_ms);
    return 0;
}
// bool sceAvPlayerGetVideoData(handle, AvPlayerFrameInfo*) — deliver the next decoded frame (NV12).
// BLOCKING media-clock gate: hold the caller until the frame's PTS is due. The real AvPlayer
// delivers a frame when its PTS reaches the media clock; this guest sleeps its whole poll budget
// when a call returns "no frame yet", so a non-blocking refusal collapsed playback to ~1 frame
// per sleep (measured 0.4-1.2 fps, #2981 FMV). g_avp_mx is HELD by the caller for the whole
// wait — deliberate: no player state (stop/pause/seek/destroy) can change mid-wait, so the
// only exit is due time; the 5 s cap keeps a broken anchor from hanging the video thread.
void avp_await_due(AvpPlayer& p, uint64_t pts_ms) {
    if (!p.gate_started) return;   // no anchor yet: the first delivery arms the clock
    const auto due_wall = p.play_start_wall + std::chrono::milliseconds(
        pts_ms > p.play_start_pts ? pts_ms - p.play_start_pts : 0);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= due_wall || now >= deadline) return;
        // Sleep the remaining wait, clamped: a long first sleep keeps the loop cheap, and the
        // final steps shrink so the wakeup lands within ~1 ms of due (an 8 ms flat quantum
        // overshot every frame by ~4 ms and cost ~10% of the playback rate).
        const auto remaining = due_wall - now;
        std::this_thread::sleep_for(
            remaining > std::chrono::milliseconds(8) ? std::chrono::milliseconds(8) : remaining);
    }
}


HLE(s_avp_getvideodata) {
    svc_log("sceAvPlayerGetVideoData", a0,a1,a2,a3,a4,a5);
    auto* fi = (AvpFrameInfo*)PW(a1); if (!fi) return 0;
    void* obj = nullptr; AvpEventCb cb = nullptr; bool fire_stop = false; uint64_t result = 0;
    prosper::video::VideoBackend* log_backend = nullptr; int log_backend_id = -1;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0);
        // Paused gates delivery, EXCEPT in the window opened by a successful sceAvPlayerJumpToTime.
        // A seek repositions the player and must publish the new position while it is still paused:
        // PPSA30490 pauses, seeks, then pulls frames until the player's timestamp moves and only
        // then calls sceAvPlayerResume (eboot+0x15dc45c loop -> +0x15dc0c1). A player that delivered
        // nothing while paused could never let that guest leave its splash. The window closes on the
        // first delivered frame. avp_paused_deliver() is the separate default-off #1599 falsification
        // lever documented above, not a feature; it never changes a default run.
        if (it == g_avp.end()) return 0;
        // A consumer exists (#1973). Record that BEFORE any refusal below: a pull that returns
        // nothing because the player is paused or because decode has not caught up is still a title
        // asking for frames, and the media clock must never mistake it for one that never asks.
        it->second.video_request_at = std::chrono::steady_clock::now();
        if (!it->second.playing ||
            (it->second.paused && !it->second.seek_deliver && !avp_paused_deliver())) return 0;
        AvpPlayer& p = it->second;
        log_backend = p.backend; log_backend_id = p.backend_id;
        if (auto* b = p.backend; b && p.backend_id >= 0) {
            prosper::video::VideoFrame vf;
            // Media-clock gate, same as GetVideoDataEx above (#2981 FMV speed).
            const bool gated = p.gate_started && b->can_peek_video();
            bool have = gated ? b->peek_video(p.backend_id, vf) : true;
            if (have && gated && !p.seek_deliver) {
                // BLOCKING media-clock gate: hold until the frame's PTS is due (see the
                // Ex handler; measured 0.4-1.2 fps with a non-blocking gate on this guest).
                avp_await_due(p, vf.pts_us / 1000);
            }
            if (have && b->next_video(p.backend_id, vf)) {
                p.play_start_wall = std::chrono::steady_clock::now();
                p.play_start_pts = vf.pts_us / 1000;
                p.gate_started = true;
                uint8_t* staged = nullptr;
                uint32_t pitch = 0;
                if (avp_stage_video(p, vf, false, staged, pitch)) {
                    fi->p_data = staged;
                    fi->timestamp = vf.pts_us / 1000;
                    p.last_ts_ms = fi->timestamp;
                    p.seek_deliver = false;
                    fi->d0 = vf.width; fi->d1 = vf.height; fi->d2 = 0; fi->d3 = 0;
                    result = 1;
                }
            } else if (b->eof(p.backend_id) && !p.stop_fired) {
                p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
            }
        } else if (p.synthetic && p.poll < avp_synth_frames()) {
            // Synthetic (headless): a black NV12 frame at the default dimensions.  Advance on
            // frame delivery: Astro Bot never polls IsActive, so an IsActive-only counter made EOF
            // and the STOP event unreachable even though it continuously pulled video frames.
            uint8_t* staged = nullptr;
            uint32_t pitch = 0;
            if (avp_stage_synthetic(p, false, staged, pitch)) {
                fi->p_data = staged; fi->timestamp = p.poll * 33ull; // ~30fps PTS (ms)
                p.last_ts_ms = fi->timestamp;
                fi->d0 = p.width; fi->d1 = p.height; fi->d2 = 0; fi->d3 = 0;
                p.poll++; result = 1;
            }
        } else if (!p.stop_fired) {
            p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
        }
    }
    if (fire_stop) avp_fire(obj, cb, AVP_STOP);
    if (avp_log()) fprintf(stderr, "[avp] video handle=0x%llx result=%llu data=%p stop=%d\n",
                           (unsigned long long)a0, (unsigned long long)result,
                           result ? fi->p_data : nullptr, (int)fire_stop);
    return result;
}
// bool sceAvPlayerGetVideoDataEx(handle, AvPlayerFrameInfoEx*) — the game uses this Ex variant.
HLE(s_avp_getvideodataex) {
    svc_log("sceAvPlayerGetVideoDataEx", a0,a1,a2,a3,a4,a5);
    auto* fi = (AvpFrameInfoEx*)PW(a1); if (!fi) return 0;
    void* obj = nullptr; AvpEventCb cb = nullptr; bool fire_stop = false; uint64_t result = 0;
    uint64_t media_snapshot = UINT64_MAX;
    prosper::video::VideoBackend* log_backend = nullptr; int log_backend_id = -1;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0);
        // See s_avp_getvideodata: paused gates delivery except inside the post-seek window, and any
        // request — answered or refused — proves a consumer exists for the media clock (#1973).
        if (it == g_avp.end()) return 0;
        it->second.video_request_at = std::chrono::steady_clock::now();
        if (!it->second.playing ||
            (it->second.paused && !it->second.seek_deliver && !avp_paused_deliver())) return 0;
        AvpPlayer& p = it->second;
        log_backend = p.backend; log_backend_id = p.backend_id;
        if (auto* b = p.backend; b && p.backend_id >= 0) {
            prosper::video::VideoFrame vf;
            // Media-clock gate: the real AvPlayer owns the clock and refuses frames whose PTS is
            // not due yet. Measured without a gate, GRIS's 73.4 s intro delivered all 1761 frames
            // in 19.0 s of wall (3.86x) — this title presents every frame it receives, so the
            // pacing must live HERE, not in the guest. Peek first: next_video pops, and a frame
            // refused after the pop would be lost.
            // The PTS gate needs a non-destructive peek. A backend without one keeps ungated
            // delivery (the pre-gate behavior) — refusing frames it cannot re-inspect would
            // deliver nothing at all (#2989 review, blocking 1).
            const bool gated = p.gate_started && b->can_peek_video();
            bool have = gated ? b->peek_video(p.backend_id, vf) : true;
            if (have && gated && !p.seek_deliver) {
                // BLOCKING media-clock gate: hold until the frame's PTS is due (see the
                // non-Ex handler; measured 0.4-1.2 fps with a non-blocking gate on this guest).
                avp_await_due(p, vf.pts_us / 1000);
            }
            if (have && b->next_video(p.backend_id, vf)) {
                p.play_start_wall = std::chrono::steady_clock::now();
                p.play_start_pts = vf.pts_us / 1000;
                p.gate_started = true;
                media_snapshot = p.play_start_pts;
                uint8_t* staged = nullptr;
                uint32_t pitch = 0;
                // Publish the frame only if its metadata is publishable: a refused fill would
                // otherwise hand the guest a staged pointer described by a zeroed AvPlayerVideoEx.
                if (avp_stage_video(p, vf, true, staged, pitch) &&
                    avp_fill_video_ex(fi->details.video, vf.width, vf.height, pitch, p.fps)) {
                    fi->p_data = staged;
                    fi->timestamp = vf.pts_us / 1000;
                    p.last_ts_ms = fi->timestamp;
                    p.seek_deliver = false;
                    result = 1;
                }
            } else if (b->eof(p.backend_id) && !p.stop_fired) {
                p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
                if (avp_log())
                    fprintf(stderr, "[avp] video-ex EOF handle=0x%llx decoder_dropped=%llu\n",
                            (unsigned long long)a0,
                            (unsigned long long)b->video_frames_dropped(p.backend_id));
            }
        } else if (p.synthetic && p.poll < avp_synth_frames()) {
            uint8_t* staged = nullptr;
            uint32_t pitch = 0;
            if (avp_stage_synthetic(p, true, staged, pitch) &&
                avp_fill_video_ex(fi->details.video, p.width, p.height, pitch, p.fps)) {
                fi->p_data = staged; fi->timestamp = p.poll * 33ull;
                p.last_ts_ms = fi->timestamp;
                p.poll++; result = 1;
            }
        } else if (!p.stop_fired) {
            p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
        }
    }
    if (fire_stop) avp_fire(obj, cb, AVP_STOP);
    if (avp_log()) {
        static std::atomic<uint64_t> delivered{0};
        if (result)
            fprintf(stderr, "[avp] video-ex handle=0x%llx DELIVER #%llu t=%llums pts=%llums media=%llums\n",
                    (unsigned long long)a0,
                    (unsigned long long)delivered.fetch_add(1) + 1,
                    (unsigned long long)avp_ms(), (unsigned long long)fi->timestamp,
                    (unsigned long long)media_snapshot);
        else
            fprintf(stderr, "[avp] video-ex handle=0x%llx empty t=%llums media=%llums q=%d\n",
                    (unsigned long long)a0, (unsigned long long)avp_ms(),
                    (unsigned long long)media_snapshot,
                    log_backend ? log_backend->video_queue_depth(log_backend_id) : -2);
    }
    return result;
}
HLE(s_avp_getaudiodata)   {   // bool sceAvPlayerGetAudioData(handle, AvPlayerFrameInfo*)
    svc_log("sceAvPlayerGetAudioData", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0);
    if (it == g_avp.end() || !it->second.playing || it->second.paused) return 0;
    auto* fi = (AvpFrameInfo*)PW(a1); if (!fi) return 0;
    AvpPlayer& p = it->second;
    if (auto* b = p.backend; b && p.backend_id >= 0) {
        prosper::video::AudioFrame af;
        if (!b->next_audio(p.backend_id, af)) return 0;
        if (!af.pcm || af.channels == 0 || af.samples == 0 ||
            af.samples > SIZE_MAX / af.channels) return 0;
        const size_t sample_count = static_cast<size_t>(af.samples) * af.channels;
        if (sample_count > SIZE_MAX / sizeof(int16_t)) return 0;
        p.audio.resize(sample_count);
        memcpy(p.audio.data(), af.pcm, sample_count * sizeof(int16_t));
        fi->p_data = reinterpret_cast<uint8_t*>(p.audio.data()); fi->timestamp = af.pts_us / 1000;
        if (fi->timestamp > p.last_ts_ms) p.last_ts_ms = fi->timestamp;
        AvpAudio details{(uint16_t)af.channels, {}, af.sample_rate,
                         (uint32_t)(af.samples * af.channels * sizeof(int16_t)), {}};
        memcpy(&fi->d0, &details, sizeof(details));
        if (avp_log()) {
            // A real counter: this field was a literal 0, so every line read "#0" and the one thing
            // the field exists to answer -- how much audio has actually been handed over -- could
            // only be recovered by counting log lines.
            static std::atomic<uint64_t> audio_delivered{0};
            fprintf(stderr, "[avp] audio-deliver handle=0x%llx #%llu t=%llums pts=%llums samples=%u\n",
                    (unsigned long long)a0,
                    (unsigned long long)audio_delivered.fetch_add(1) + 1,
                    (unsigned long long)avp_ms(),
                    (unsigned long long)fi->timestamp, af.samples);
        }
        return 1;
    }
    if (!p.synthetic || p.audio_poll >= avp_synth_frames()) return 0;
    constexpr uint32_t samples = 1024, channels = 2;
    if (p.audio.size() != samples * channels) p.audio.assign(samples * channels, 0);
    fi->p_data = (uint8_t*)p.audio.data();
    fi->timestamp = p.audio_poll * 21ull; // 1024/48 kHz, milliseconds
    if (fi->timestamp > p.last_ts_ms) p.last_ts_ms = fi->timestamp;
    AvpAudio details{2, {}, 48000, samples * channels * sizeof(int16_t), {}};
    memcpy(&fi->d0, &details, sizeof(details));
    p.audio_poll++;
    if (avp_log()) fprintf(stderr, "[avp] audio handle=0x%llx result=1 poll=%llu\n",
                           (unsigned long long)a0, (unsigned long long)p.audio_poll);
    return 1;
}
// u64 sceAvPlayerCurrentTime(handle) — current playback position in MILLISECONDS. Returns the
// newest delivered frame timestamp (video, or audio when audio leads), 0 before the first frame or
// for an unknown handle. Alex Kidd DX's pre-level cutscene runner polls this to pace/finish the
// sequence; the old unresolved-import 0 held its fade-to-black over an already-running level
// forever (#320). Fires no guest callbacks, so no AVP_CALLBACK_ENTRY shim is needed.
HLE(s_avp_current_time) {
    svc_log("sceAvPlayerCurrentTime", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0);
    const uint64_t now_ms = it == g_avp.end() ? 0 : it->second.last_ts_ms;
    if (avp_log()) fprintf(stderr, "[avp] current-time handle=0x%llx -> %llu ms\n",
                           (unsigned long long)a0, (unsigned long long)now_ms);
    return now_ms;
}
HLE(s_avp_stop) {   // s32 sceAvPlayerStop(handle)
    svc_log("sceAvPlayerStop", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool fire_stop = false;
    prosper::video::VideoBackend* backend = nullptr; int backend_id = -1;
    AvpMemAllocator memory{};
    std::vector<uint8_t*> textures;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0;
        AvpPlayer& p = it->second;
        backend = p.backend; backend_id = p.backend_id;
        memory = p.memory;
        textures = std::move(p.texture_frames);
        avp_release_frame_storage(p);
        p.texture_frame_bytes = 0; p.next_texture_frame = 0;
        p.backend = nullptr; p.backend_id = -1; p.have_source = false; p.synthetic = false;
        p.seek_deliver = false;
        if (p.playing && !p.stop_fired) {
            p.playing = false; p.paused = false; p.stop_fired = true;
            fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
        }
    }
    if (backend && backend_id >= 0) backend->close(backend_id);
    avp_release_textures(memory, textures);
    if (fire_stop) avp_fire(obj, cb, AVP_STOP);
    return 0;
}
HLE(s_avp_close) {   // s32 sceAvPlayerClose(handle)
    svc_log("sceAvPlayerClose", a0,a1,a2,a3,a4,a5);
    prosper::video::VideoBackend* backend = nullptr; int backend_id = -1;
    AvpMemAllocator memory{};
    std::vector<uint8_t*> textures;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0);
        if (it != g_avp.end()) {
            backend = it->second.backend; backend_id = it->second.backend_id;
            memory = it->second.memory;
            textures = std::move(it->second.texture_frames);
            avp_release_frame_storage(it->second);
            g_avp.erase(it);
        }
    }
    if (backend && backend_id >= 0) backend->close(backend_id);
    avp_release_textures(memory, textures);
    return 0;
}

#ifndef _WIN32
// Pass the HLE-entry stack pointer to callback-producing AvPlayer handlers.  On Linux it identifies
// the guest TCB saved by the import stub; on macOS callback_guest_fs_from_entry_stack() fails closed
// to zero because guest TLS uses trap emulation rather than a hardware %fs switch.
#define AVP_CALLBACK_ENTRY(entry, target, handler) \
    extern "C" uint64_t target(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t); \
    PROSPER_ASM_TRAMPOLINE(entry, target) \
    extern "C" void entry(); \
    extern "C" uint64_t target(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, \
                                uint64_t a4, uint64_t a5, uint64_t entry_rsp) { \
        const uint64_t guest_fs = callback_guest_fs_from_entry_stack(entry_rsp); \
        if (avp_log() && guest_fs) { \
            const uint64_t guest_ra = *(const uint64_t*)(uintptr_t)(entry_rsp + 0x30); \
            /* The title-facing AvPlayer adapters use a frame pointer plus two saved registers; */ \
            /* retain their caller as a diagnostic without changing normal logging or behavior. */ \
            const uint64_t guest_caller = *(const uint64_t*)(uintptr_t)(entry_rsp + 0x50); \
            fprintf(stderr, "[avp] call %s guest_ra=0x%llx guest_caller=0x%llx guest_fs=1\n", \
                    #handler, (unsigned long long)guest_ra, (unsigned long long)guest_caller); \
        } \
        (void)guest_fs; \
        /* Only Linux swaps hardware %fs at the import boundary. */ \
        PROSPER_AVP_ENTRY_SCOPE(guest_fs) \
        return handler(a0, a1, a2, a3, a4, a5); \
    }

#if defined(__linux__)
#define PROSPER_AVP_ENTRY_SCOPE(guest_fs) AvpEntryGuestFsScope avp_entry_guest_fs_scope(guest_fs);
#else
#define PROSPER_AVP_ENTRY_SCOPE(guest_fs)
#endif

AVP_CALLBACK_ENTRY(s_avp_addsource_entry, s_avp_addsource_entry_c, s_avp_addsource)
AVP_CALLBACK_ENTRY(s_avp_addsourceex_entry, s_avp_addsourceex_entry_c, s_avp_addsourceex)
AVP_CALLBACK_ENTRY(s_avp_start_entry, s_avp_start_entry_c, s_avp_start)
AVP_CALLBACK_ENTRY(s_avplayer_isactive_entry, s_avplayer_isactive_entry_c, s_avplayer_isactive)
AVP_CALLBACK_ENTRY(s_avp_getvideodata_entry, s_avp_getvideodata_entry_c, s_avp_getvideodata)
AVP_CALLBACK_ENTRY(s_avp_getvideodataex_entry, s_avp_getvideodataex_entry_c, s_avp_getvideodataex)
AVP_CALLBACK_ENTRY(s_avp_stop_entry, s_avp_stop_entry_c, s_avp_stop)
AVP_CALLBACK_ENTRY(s_avp_close_entry, s_avp_close_entry_c, s_avp_close)
AVP_CALLBACK_ENTRY(s_avp_pause_entry, s_avp_pause_entry_c, s_avp_pause)
AVP_CALLBACK_ENTRY(s_avp_resume_entry, s_avp_resume_entry_c, s_avp_resume)

#undef PROSPER_AVP_ENTRY_SCOPE
#undef AVP_CALLBACK_ENTRY
#endif

// Registration for this library, called by register_builtin_hle(). One entry point per
// Sony library keeps every handler `static` to the file that implements it, which is what
// lets these files be separate translation units at all (#3735).
void register_avplayer_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    // libSceAvPlayer (#324): let a post-credits / intro video complete so the game reaches its scene.
    // Init/InitEx must return a non-NULL handle; IsActive must report finished; the rest succeed as no-ops.
    R("sceAvPlayerInit",           s_avplayer_init);
    R("sceAvPlayerInitEx",         s_avplayer_initex);
    R("sceAvPlayerPostInit",       s_avp_postinit);
    R("sceAvPlayerSetLogCallback", s_avp_setlogcb);
#ifdef _WIN32
    R("sceAvPlayerAddSource",      s_avp_addsource);
    R("sceAvPlayerAddSourceEx",    s_avp_addsourceex);
    R("sceAvPlayerStart",          s_avp_start);
    R("sceAvPlayerIsActive",       s_avplayer_isactive);
    R("sceAvPlayerGetVideoData",   s_avp_getvideodata);
    R("sceAvPlayerGetVideoDataEx", s_avp_getvideodataex);
#else
    R("sceAvPlayerAddSource",      s_avp_addsource_entry);
    R("sceAvPlayerAddSourceEx",    s_avp_addsourceex_entry);
    R("sceAvPlayerStart",          s_avp_start_entry);
    R("sceAvPlayerIsActive",       s_avplayer_isactive_entry);
    R("sceAvPlayerGetVideoData",   s_avp_getvideodata_entry);
    R("sceAvPlayerGetVideoDataEx", s_avp_getvideodataex_entry);
#endif
    R("sceAvPlayerGetAudioData",   s_avp_getaudiodata);
#ifdef _WIN32
    R("sceAvPlayerStop",           s_avp_stop);
    R("sceAvPlayerClose",          s_avp_close);
#else
    R("sceAvPlayerStop",           s_avp_stop_entry);
    R("sceAvPlayerClose",          s_avp_close_entry);
#endif
    // PS5 raw NIDs verified against the 3.20 import stubs. Astro Bot enumerates streams before it
    // consumes GetVideoDataEx; returning the generic unresolved-import zero meant "no streams" and
    // left its logo movie state machine permanently resident even after synthetic EOF.
    Hle::register_fn("hdTyRzCXQeQ", (HleFn)s_avp_streamcount,   "sceAvPlayerStreamCount");
    Hle::register_fn("wwM99gjFf1Y", (HleFn)s_avp_current_time,  "sceAvPlayerCurrentTime");
    Hle::register_fn("d8FcbzfAdQw", (HleFn)s_avp_getstreaminfo, "sceAvPlayerGetStreamInfo");
    Hle::register_fn("ctTAcF5DiKQ", (HleFn)s_avp_getstreaminfoex, "sceAvPlayerGetStreamInfoEx");
    Hle::register_fn("ODJK2sn9w4A", (HleFn)s_avp_stream_ok, "sceAvPlayerEnableStream");
    Hle::register_fn("BOVKAzRmuTQ", (HleFn)s_avp_stream_ok, "sceAvPlayerDisableStream");
    Hle::register_fn("buMCiJftcfw", (HleFn)s_avp_stream_ok, "sceAvPlayerChangeStream");
#ifdef _WIN32
    Hle::register_fn("9y5v+fGN4Wk", (HleFn)s_avp_pause, "sceAvPlayerPause");
    Hle::register_fn("w5moABNwnRY", (HleFn)s_avp_resume, "sceAvPlayerResume");
#else
    Hle::register_fn("9y5v+fGN4Wk", (HleFn)s_avp_pause_entry, "sceAvPlayerPause");
    Hle::register_fn("w5moABNwnRY", (HleFn)s_avp_resume_entry, "sceAvPlayerResume");
#endif
    // sceAvPlayerJumpToTime (#1949). Unregistered, its dispatcher default of 0 was read as a
    // completed seek and deadlocked PPSA30490's Unity video splash; the handler now performs a real
    // backend seek and returns a real error when it cannot. NID verified in the PS5 3.20
    // libSceAvPlayer stub table. Calls no guest code, so it needs no entry-rsp trampoline.
    Hle::register_fn("XC9wM+xULz8", (HleFn)s_avp_jumptotime, "sceAvPlayerJumpToTime");
    Hle::register_fn("k-q+xOxdc3E", (HleFn)s_avp_stream_ok, "sceAvPlayerSetAvSyncMode");
    Hle::register_fn("OVths0xGfho", (HleFn)s_avp_stream_ok, "sceAvPlayerSetLooping");
    #undef R
}
// The local network libraries allocate opaque contexts even on a disconnected console; connection
// state is reported separately through NetCtl/NP. Returning generic success (0) from these ID-returning
// constructors instead creates an invalid context and makes their owner's initialization fail. The
// signatures and positive-return contract agree with the PS5 3.20 symbol table and the independently
// implemented SDK surface; no online identity or connectivity is fabricated here.
namespace {
}

// --- libSceIme keyboard API (#186) ---
// PPSA02664 polls this every frame in its input loop. We have no physical PS5 keyboard, so we report a
// consistent "no keyboard connected" state: Open/Update succeed, GetResourceId returns an empty
// resource array (no keyboards), GetInfo reports a disconnected device — the game then uses controller
// / on-screen input instead of waiting on a keyboard. Struct layouts + field offsets verified against
// shadPS4 src/core/libraries/ime/ime_common.h; error codes from ime_error.h. Note sceImeKeyboardOpen
// returns an Error (0 = SCE_OK), NOT a handle — so success is 0. CONFIDENCE: HIGH on the layouts; MED
// on the device/status enum "disconnected" == 0.
//   OrbisImeKeyboardResourceIdArray: user_id@0(s32) resource_id[5]@4(u32)                     size 24
//   OrbisImeKeyboardInfo: user_id@0 device@4 type@8 repeat_delay@12 repeat_rate@16 status@20 rsv[12]@24  size 36
// --- IME keyboard event injection (issue #1093) -------------------------------------------------
// PPSA02664 (Alex Kidd) reads its "press any button" / menu input ONLY through the IME keyboard
// path: it opens a keyboard and calls sceImeUpdate(handler) every frame, and never touches
// libScePad or sceUserService. sceImeUpdate must invoke handler(arg, &event) for each queued
// keyboard event; the old stub returned without calling, delivered no input, and the title never
// advanced (injected pad input was inert — the game wasn't reading the pad).
//
// Event ABI derived directly from the guest handler at eboot+0xf2c540 (PPSA02664 disassembly):
//   handler(rdi=arg, rsi=SceImeEvent*)
//   SceImeEvent:  +0x00 u32 id   (0x101 = KEY_DOWN, 0x102 = KEY_UP; 0x103..0x106 = other kbd events)
//                 +0x08 u16 keycode  (USB HID usage id; indexes the guest's keycode->bit table)
//                 +0x0a u16 (modifier/char)   +0x0c u32 (status)
//   KEY_DOWN sets the guest's current + newly-pressed keyboard bitmasks; KEY_UP clears them.
// HID keycodes verified against the guest's table: Enter=0x28, Space=0x2c, Z=0x1d, A=0x04.
// CONFIDENCE: HIGH on the id/keycode/offset layout (read from the guest handler + its HID table).
namespace {
} // namespace

// --- app content ---
namespace {
} // namespace

// sceSystemServiceGetStatus(SceSystemServiceStatus* status) — the out-struct is ARG 0 (single-arg
// call). This was aliased to s_appcontent_int, which returned success while writing 4 bytes through
// a1 — whatever stale value the caller left in RSI — and left the real status struct uninitialized.
// Layout cross-checked against Kyty LibSystemService.cpp:83: int32 eventNum @0; bool
// isSystemUiOverlaid @4, isInBackgroundExecution @5, isCpuMode7CpuNormal @6 (defaults TRUE),
// isGameLiveStreamingOnAir @7, isOutOfVrPlayArea @8; 12 bytes with tail padding.
namespace {
// ErrorDialog is system-owned UI. Platform glue uses SystemService status' background -> foreground
// transition to resume requests after it closes, independently of the CommonDialog FINISHED value.
// A real backend may stay active across many status samples; the headless backend dismisses during
// Open and must still expose one observable enter/return pair instead of collapsing both edges into
// a permanent foreground value. Oregon Trail #1606 positive-controlled this exact contract twice:
// the account-completion writer ran only after both SystemService samples. No title address or
// title identity participates in this state machine.
}

// Message-dialog LIFECYCLE (#144). Status enum: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3.
// The old handler returned FINISHED(3) UNCONDITIONALLY — including before any Open, where the real
// API reports NONE/INITIALIZED — so a guest polling GetStatus as a guard saw "dialog already done"
// at the wrong stage and skipped/duplicated its dialog logic. We track the real transitions:
// Initialize -> INITIALIZED, Open -> auto-dismiss to FINISHED (headless: no interactive UI, so the
// game's "wait until dismissed" loop still exits immediately), Close/Terminate -> back to NONE.
// GetResult -> zeroed struct = OK/no button pressed.
// A registered PlatformUi gets first refusal at Open; if it takes the dialog, status/result/close
// route there (real message box). Otherwise the core auto-dismisses headlessly. `g_msgdialog_backed`
// records which path Open chose. See platform_ui.hpp (#347).

// --- libSceSaveDataDialog ---------------------------------------------------------------
// Uses the common-dialog status enum: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3. A generic
// success stub is not safe for UpdateStatus: Dead Cells polled it 1,135 times in 16 seconds because
// zero means NONE, leaving its level transition in a permanent wait (#768). The headless policy
// matches MsgDialog above: Open auto-dismisses to FINISHED so the guest can consume a neutral result.
//
// PS4/PS5 SDK layout used below (also used by shadPS4): param.mode is u32 at +0x34 and param.userData
// is a pointer at +0x70. Result begins with mode/result/buttonId/pad followed by three pointers. The
// dirName/param pointer slots are caller-owned output destinations, not fields for the service to
// replace. Write only mode/result/buttonId/userData and preserve every padding, pointer, and reserved
// byte. A PlatformUi that accepts Open remains the intended owner, but every callback obtains a
// registry lease first: unregistering the backend safely abandons the dialog instead of dereferencing
// a stale non-owning pointer or rerouting the dialog to a replacement backend.
namespace {
}

// --- libSceImeDialog (on-screen text-entry dialog) (#191). We have no keyboard UI, so the dialog
// auto-completes: Init -> FINISHED immediately, so the game's "poll GetStatus until Finished" loop
// exits at once instead of hanging on a dialog that never appears; GetResult reports endStatus =
// OK/ENTER with the (unchanged/empty) input buffer; Term/Abort return to NONE.
// IMPORTANT: sceImeDialogGetStatus returns the IME-dialog's OWN enum (OrbisImeDialogStatus: NONE=0,
// RUNNING=1, FINISHED=2) — NOT the 4-value SceCommonDialogStatus that MsgDialog/ErrorDialog use
// (…RUNNING=2, FINISHED=3). Verified in shadPS4 ime_dialog.cpp. Returning 3 here made a game's poll
// loop never see Finished(2). We write only the 4-byte endStatus at GetResult offset 0 (the field the
// game branches on), never more, so a wrong tail-field guess can't corrupt the caller's struct.
// A registered PlatformUi (the app frontend) gets first refusal on the dialog: if it takes it
// (imeDialogOpen -> true), status/result/close route there so a real text field is shown; otherwise
// the core auto-completes headlessly (below). `g_imedialog_backed` records which path Init chose so a
// later poll/result/term goes to the same place. See platform_ui.hpp (#347).

// --- Transaction resources (#1905). sceSaveDataCreateTransactionResource returns the NEW RESOURCE'S
// ID, not 0-on-success. Returning 0 is the shape of the call that succeeds, so this looked correct
// and is not: Sonic Origins' (PPSA05325) save handler at eboot+0x93fdb0 is exactly
//     xor edi,edi; call sceSaveDataCreateTransactionResource; test eax,eax; jle <fail>
// — a zero return is read as failure, the handler records error 3, every later save operation
// returns that sticky error without running, and the title's boot coroutine polls the failed job
// forever. That single wrong return value is why PPSA05325 never leaves its first boot step.
// It really is a HANDLE and not merely "a positive number" — Sonic's full lifecycle proves it:
//   0x93fdc7  mov [r14+0xc0],eax        Create's result is retained in a member
//   0x9402cd  mov eax,[r14+0xc0]        ...loaded again to build the Mount3 descriptor,
//   0x9402d4  mov [rsp+0x68],eax        ...at descriptor base (rsp+0x40) + 0x28,
//   0x93f24e  mov edi,[rdi+0xc0]        ...and handed back as Delete's sole argument,
//   0x93f254  cmp edi,0xffffffff        guarded by the guest's own -1 "no resource" sentinel.
// That also independently reproduces the Mount3 +0x28 field this file documents from DQ7. The id is
// opaque to the guest, so any positive value satisfies it; we hand out a monotonic counter and keep
// the live set so Delete can reject one that was never created.
// CONFIDENCE: HIGH on the polarity, on "returns the resource id", and on Delete's argument being a
// scalar int32 in edi (the mov above precedes all four of Sonic's Delete sites and Oregon Trail's).
// Delete's error CODE for an unknown id is the residual unknown: PARAMETER is the facility's generic
// bad-argument code, and no title is observed passing an id Create did not return.
namespace {
}   // namespace

// --- libSceSaveData "save-data memory" API (#191). A per-(user,slot) fixed-size memory block the
// managed SaveData layer reads/writes by offset and syncs to storage; on PS5 this is how a title
// keeps a small always-resident save (settings/progress). We back each slot with a host-memory
// block: Setup allocates it, Set copies guest->block, Get copies block->guest, Sync commits. Struct
// layouts + field offsets verified against shadPS4 save_data/savedata.cpp; error codes from
// savedata_error.h. CONFIDENCE: HIGH on the happy-path round-trip (the contract a title depends on).
//   OrbisSaveDataMemoryData:  buf@0(ptr) bufSize@8(u64) offset@16(s64)               size 64
//   OrbisSaveDataMemorySetup2: option@0 userId@4 memorySize@8 iconMemorySize@16
//                              initParam@24 initIcon@32 slotId@40                    size 64
//   OrbisSaveDataMemorySet2:  userId@0 [pad@4] data@8(ptr) param@16 icon@24 dataNum@32 slotId@36
//   OrbisSaveDataMemoryGet2:  userId@0 [pad@4] data@8(ptr) param@16 icon@24 slotId@32
namespace {
}

// ===== Issue #306: honest OFFLINE console for the online/update/entitlement boot chain. =========
// DOLL's UE4 front-end runs a patch/entitlement check before the title screen; every subsystem in
// that chain answered success-with-garbage, so the check could neither succeed nor FAIL — the flow
// waited forever (docs/DOLL_LOADING_PROGRESSION.md). The blocks below give the chain the answers a
// real, network-disconnected, signed-out console gives.

// --- Guest-callback delivery discipline (shared by NetCtl + Np state callbacks). ----------------
// A registered callback is guest code: under PROSPER_GUEST_FS the HLE runs on the HOST %fs (the
// import swap-stub switched), so the guest callback must run with the GUEST %fs restored or its
// TLS accesses (UE MallocBinned caches!) read host TLS garbage. The swap-stub saves the guest fs
// base in its frame (push r11), so an asm entry shim (the f_apr_read_submit_entry pattern) hands
// the handler its entry %rsp. The guest swap path re-pushes args7/8/9, an alignment pad, then the
// saved r11: [rsp]=ret-to-stub, args at +8/+0x10/+0x18, pad at +0x20, guest fs at +0x28, and guest
// RA at +0x30. A [rsp] outside the stub region [0x6_0000_0000,0x7_0000_0000) means the
// host-context tail-jmp path (no swap happened) — call the callback on the current fs.
// Mechanism proven live by the PROSPER_NETCTL_CB experiment (run 7/9: delivered + consumed
// cleanly, no crash). CONFIDENCE: HIGH.
namespace {
#ifndef _WIN32
#endif
}

// --- libSceNetCtl: a network-DISCONNECTED console (default ON since #306). ----------------------
// DOLL registers a NetCtl state callback once at boot (sceNetCtlRegisterCallback) and then pumps
// sceNetCtlCheckCallback EXACTLY once per frame forever (14,191 calls in a 240 s run). On real
// hardware CheckCallback invokes the registered callback on the calling thread with the current
// state — an offline console still delivers an immediate DISCONNECTED. Register records {func,arg}
// and writes the callback id (Kyty Network.cpp NetCtlRegisterCallback); CheckCallback invokes the
// callback ONCE with SCE_NET_CTL_EVENT_TYPE_DISCONNECTED (PS4-inherited constant = 1; identical
// export names+NIDs on PS5 3.20 — CONFIDENCE MED on the PS5 value). Was the gated experiment
// PROSPER_NETCTL_CB=1; proven correct+consumed live (DOLL run 7/9), now default ON.
// PROSPER_NETCTL_CB=0 restores the old unimplemented behavior.
namespace {
#ifndef _WIN32
#endif
}

// --- libSceNpManager state callback: deliver SIGNED_OUT once (#306). ----------------------------
// DOLL registers its Np sign-in state callback via sceNpRegisterStateCallbackA and pumps
// sceNpCheckCallback. shadPS4 (offline mode) queues exactly one SIGNED_OUT event for the initial
// user and delivers it inside sceNpCheckCallback on the pumping thread; we mirror that. Callback-A
// prototype (shadPS4 np_manager.h): void cb(s32 userId, s32 state, void* userdata); state
// SIGNED_OUT = 1 (Unknown=0, SignedOut=1, SignedIn=2 — Kyty + shadPS4 agree). Register returns the
// positive callback id (shadPS4 RegisterStateCallbackA returns slot+1). CONFIDENCE: HIGH on the
// contract (two agreeing PS4 references, PS4-inherited surface; PS5 3.20 exports the same names).
namespace {
#ifndef _WIN32
#endif
}

// --- libSceErrorDialog: the real Initialize/Open/Close lifecycle (auto-dismiss, headless). ------
// Status enum shared with CommonDialog: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3 (shadPS4
// commondialog.h + error_dialog.cpp; PS4-inherited, identical export NIDs on PS5 3.20).
// GetStatus/UpdateStatus RETURN the status value directly (not an out-param). DOLL pumps
// sceErrorDialogUpdateStatus once per frame from boot (housekeeping); if the honest offline chain
// makes it Open the "could not download patch data" dialog, auto-dismiss to FINISHED so the flow's
// "wait until dismissed" loop exits — the MsgDialog precedent (#144). Open's param is
// { s32 size; s32 errorCode; s32 userId; s32 reserved } (shadPS4 error_dialog.cpp Param); the
// errorCode is printed unconditionally — a one-shot, high-value diagnostic of WHAT the game
// thinks failed. A PlatformUi that accepts Open is pinned as the non-owning expected backend; every
// later callback obtains a registry lease for that exact pointer. Unregister/replacement abandons
// safely to headless FINISHED and can never reroute to a backend that did not accept the request.
// CONFIDENCE: HIGH (shadPS4 implements this exact lifecycle).
namespace {
}

// ===== libSceRandom ============================================================================
//
// `sceRandomGetRandomNumber(void* buf, size_t size)` is the library's ONLY export (PS5 3.20
// `libSceRandom.c` lists exactly one `sprx_dlsym` line), and 27 of the 44 local dumps import it.
//
// Unregistered, it reached the dispatcher's `return 0` — and for this contract 0 is SCE_OK. The
// guest was told its buffer had been filled while nothing wrote to it, so it read back whatever
// its own stack or heap already held and used that as entropy. The worst property of that failure
// is that it is STABLE: a fresh stack allocation at the same call site tends to hold the same
// residue on every call, so a "random" session id or nonce can be identical all run. Nothing
// crashes and no diagnostic fires; only code that inspected the buffer could notice. (#2065, swept
// under #2081.)
//
// The fix is a real implementation rather than a fail-visible stub, because the honest answer here
// is cheap: the host has a CSPRNG. Returning an error instead would be strictly worse — this is a
// *value-producing* contract, and guests do gate on it.
//
// How the guests actually consume it, measured with `tools/re/nid_gate_scan.py --nid PI7jIZj4pcE`
// over the local dumps (call sites classified by what happens to eax):
//
//     PPSA08804  nonzero=2 ignored=1        PPSA04263  nonzero=2
//     PPSA24651  nonzero=1                  PPSA13579  nonzero=1
//     PPSA17942  ignored=1                  PPSA19244  forward=1 ignored=1
//
// So most call sites branch on "did this fail", and — the load-bearing part — NOT ONE compares the
// result against a specific constant (no `const` / `other-cmp` site anywhere). That is what makes
// the reject arms safe to add: the guest reads only the SIGN of the answer, never its value.
// CONFIDENCE: HIGH that success must fill the buffer and that failure must be non-zero;
// LOW on the exact error constant — the SCE_RANDOM error space is not in the 3.20 dump (which
// carries names and NIDs only) and no call site discriminates it, so the libkernel encoding is
// used per the project's default rather than an invented SCE_RANDOM_* value.
namespace {

// The published single-request cap. A guest asking for more than this gets an error on hardware, so
// prosper must not quietly serve it: a partial fill reported as success would recreate the exact
// bug this handler exists to remove, and an over-long fill would paper over a guest bug that real
// hardware rejects.
//
// CONFIDENCE: MED — the cap is from the API documentation, not from a live capture. This is also
// the ONLY arm of this handler that can newly FAIL a call the previous stub "succeeded", so it is
// the one place a wrong constant costs something rather than merely being imprecise.
//
// What is NOT established: the request sizes local titles actually pass. `nid_gate_scan` classifies
// what the guest does with `eax` and cannot recover an argument, so no instrument here has measured
// the `size` operand at any call site — deliberately stated rather than left as an implied "we
// checked". Settling it needs the argument registers read at the call, e.g. PROSPER_SVCLOG-style
// logging on this NID or a hardware breakpoint at the import. If a title is ever found requesting
// more, this constant is where to look first.

} // namespace

} // namespace prosper
