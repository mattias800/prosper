#ifndef __APPLE__
#error "The VideoToolbox backend is macOS-only"
#endif

#if !__has_feature(objc_arc)
#error "videotoolbox_backend.mm must be compiled with -fobjc-arc"
#endif

#include "videotoolbox_backend.hpp"

#include "diagnostics/env_cache.hpp"
#include "hle/video/h264_sps.hpp"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <VideoToolbox/VideoToolbox.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// The synchronous AVAsset/AVAssetTrack property getters (tracks, duration, formatDescriptions,
// nominalFrameRate) are deprecated in favour of the async `load...` family, but they still work and
// simply block until the value is loaded. Every call here runs on the session's own worker thread,
// where blocking is exactly what we want, so the async form would only add a semaphore per property.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace prosper::video {
namespace {

constexpr size_t kVideoQueueCapacity = 8;
constexpr size_t kAudioQueueCapacity = 32;
// How far audio decode may run ahead of video. AVAssetReader interleaves its outputs internally;
// reading one far ahead of the other makes it buffer the other's samples, so the worker keeps the
// two read positions within this window and alternates between them.
constexpr uint64_t kAudioLeadUs = 500000;

bool avp_log() { return PROSPER_ENV_ON("PROSPER_AVPLOG"); }
// Deliberately a LIVE read, not PROSPER_ENV_ON: test_video_vt arms it at runtime to exercise the
// software arm on a host without a hardware decoder, and a cached read would make that arm vacuous.
bool software_decode_allowed() { return std::getenv("PROSPER_AVP_ALLOW_SOFTWARE") != nullptr; }

uint64_t cmtime_us(CMTime time) {
    if (!CMTIME_IS_NUMERIC(time)) return 0;
    const double seconds = CMTimeGetSeconds(time);
    return seconds > 0.0 ? static_cast<uint64_t>(seconds * 1e6 + 0.5) : 0;
}

struct VideoPacket {
    std::vector<uint8_t> bytes;
    uint32_t width = 0, height = 0, stride = 0;
    uint64_t pts_us = 0;
};

struct AudioPacket {
    std::vector<int16_t> pcm;
    uint32_t channels = 0, sample_rate = 0;
    uint64_t pts_us = 0;
};

// The displayed rectangle inside the decoder's buffer, fixed by the first picture.
struct Crop { uint32_t x = 0, y = 0, width = 0, height = 0; };

struct Session {
    std::string path;          // what AVFoundation opens
    std::string display_name;  // what the log says (the guest path for open_memory)
    bool owns_path = false;    // open_memory's temporary copy, unlinked when the worker exits
    bool allow_software = false;

    std::mutex mutex;
    std::condition_variable cv;
    bool initialized = false;
    bool init_ok = false;
    bool stopping = false;
    bool worker_exited = false;
    // Set once the VIDEO stream has ended (or failed): eof() follows video, per the interface.
    bool decode_done = false;
    bool hardware_video = false;
    // seek(): the worker rebuilds its reader at seek_target_us when seek_generation moves, then
    // publishes the outcome for that generation; seek() blocks until it does (as VA-API's does).
    uint64_t seek_generation = 0;
    uint64_t seek_target_us = 0;
    uint64_t seek_settled_generation = 0;
    bool seek_ok = false;

    StreamInfo stream_info;

    std::deque<VideoPacket> video_queue;
    std::deque<AudioPacket> audio_queue;
    VideoPacket last_video;
    AudioPacket last_audio;
    std::thread thread;
};

void finish_initialization(Session& session, bool ok) {
    std::lock_guard<std::mutex> lock(session.mutex);
    session.initialized = true;
    session.init_ok = ok;
    if (!ok) session.decode_done = true;
    session.cv.notify_all();
}

void mark_video_done(Session& session, const char* why) {
    if (why && avp_log())
        std::fprintf(stderr, "[avp-vt] %s '%s'\n", why, session.display_name.c_str());
    std::lock_guard<std::mutex> lock(session.mutex);
    session.decode_done = true;
    session.cv.notify_all();
}

// The reader and its outputs for one contiguous playback range. AVAssetReader cannot be
// repositioned once it has started, so a seek builds a new one.
struct Reader {
    AVAssetReader* reader = nil;
    AVAssetReaderTrackOutput* video = nil;
    AVAssetReaderTrackOutput* audio = nil;
};

bool make_reader_unguarded(AVAsset* asset, AVAssetTrack* video_track, AVAssetTrack* audio_track,
                           const StreamInfo& info, NSData* channel_layout, CMTime start,
                           Reader& out) {
    NSError* error = nil;
    AVAssetReader* reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
    if (!reader) return false;
    if (CMTIME_IS_NUMERIC(start) && CMTimeGetSeconds(start) > 0)
        reader.timeRange = CMTimeRangeMake(start, kCMTimePositiveInfinity);

    // Video is read COMPRESSED (nil output settings = passthrough) and decoded by this backend's own
    // VTDecompressionSession, because AVAssetReader neither lets a caller require hardware decoding
    // nor reports which decoder it chose. See Decoder below.
    AVAssetReaderTrackOutput* video =
        [[AVAssetReaderTrackOutput alloc] initWithTrack:video_track outputSettings:nil];
    video.alwaysCopiesSampleData = NO;
    if (![reader canAddOutput:video]) return false;
    [reader addOutput:video];

    AVAssetReaderTrackOutput* audio = nil;
    if (audio_track) {
        NSMutableDictionary* audio_settings = [@{
            AVFormatIDKey : @(kAudioFormatLinearPCM),
            AVLinearPCMBitDepthKey : @16,
            AVLinearPCMIsFloatKey : @NO,
            AVLinearPCMIsBigEndianKey : @NO,
            AVLinearPCMIsNonInterleaved : @NO,
            AVNumberOfChannelsKey : @(info.audio_channels),
            AVSampleRateKey : @(info.audio_rate),
        } mutableCopy];
        // More than two channels needs an explicit layout; the source's own keeps its channel order.
        if (channel_layout) audio_settings[AVChannelLayoutKey] = channel_layout;
        audio = [[AVAssetReaderTrackOutput alloc] initWithTrack:audio_track
                                                 outputSettings:audio_settings];
        audio.alwaysCopiesSampleData = NO;
        if (![reader canAddOutput:audio]) return false;
        [reader addOutput:audio];
    }
    if (![reader startReading]) {
        if (avp_log())
            std::fprintf(stderr, "[avp-vt] startReading failed: %s\n",
                         reader.error ? reader.error.localizedDescription.UTF8String : "(no error)");
        return false;
    }
    out.reader = reader;
    out.video = video;
    out.audio = audio;
    return true;
}

// AVFoundation reports invalid output settings by RAISING (NSInvalidArgumentException), not by a
// return value. Unwinding that through this worker would terminate the whole emulator, so a reader
// that cannot be built is a failed open/seek instead.
bool make_reader(AVAsset* asset, AVAssetTrack* video_track, AVAssetTrack* audio_track,
                 const StreamInfo& info, NSData* channel_layout, CMTime start, Reader& out) {
    @try {
        return make_reader_unguarded(asset, video_track, audio_track, info, channel_layout,
                                     start, out);
    } @catch (NSException* e) {
        if (avp_log())
            std::fprintf(stderr, "[avp-vt] reader construction raised %s: %s\n",
                         e.name.UTF8String, e.reason ? e.reason.UTF8String : "");
        return false;
    }
}

// A clean-aperture rectangle (top-left origin, in buffer pixels) as a usable crop of the plane, or
// nothing. The origin is rounded down to even so chroma rows start on a CbCr pair.
std::optional<Crop> crop_from_rect(CGRect r, uint32_t plane_w, uint32_t plane_h) {
    if (!(r.size.width >= 1.0 && r.size.height >= 1.0) || r.origin.x < 0 || r.origin.y < 0)
        return std::nullopt;
    const uint32_t x = static_cast<uint32_t>(std::floor(r.origin.x)) & ~1u;
    const uint32_t y = static_cast<uint32_t>(std::floor(r.origin.y)) & ~1u;
    const uint32_t w = static_cast<uint32_t>(std::lround(r.size.width));
    const uint32_t h = static_cast<uint32_t>(std::lround(r.size.height));
    if (w == 0 || h == 0 || x + w > plane_w || y + h > plane_h) return std::nullopt;
    return Crop{x, y, w, h};
}

// The H.264 SPS frame crop (h264::frame_crop_rect, 7.4.2.1.1), origin rounded down to even so
// chroma rows start on a CbCr pair -- only reachable with an odd offset in 4:2:2/4:4:4, which the
// NV12 output resamples to 4:2:0 anyway.
std::optional<Crop> sps_crop(const h264::SpsPictureMeta& m) {
    h264::CropRect r;
    if (!h264::frame_crop_rect(m, &r)) return std::nullopt;
    const uint32_t x = r.x & ~1u, y = r.y & ~1u;
    return Crop{x, y, r.width + (r.x - x), r.height + (r.y - y)};
}

// The displayed rectangle of a decoded picture -- what the guest sees, and what FFmpeg (the Linux
// backend) delivers. In pixels, never the track's naturalSize, which is a DISPLAY size scaled by the
// pixel aspect ratio on anamorphic content. In order:
//   1. the decoded buffer's own clean aperture (converted from its lower-left origin), when it is
//      attached and actually crops;
//   2. the SPS frame crop, when the decoder handed back the full CODED picture. Measured on this
//      host: for an MP4 whose crop lives only in the SPS (container headers at the coded size),
//      AVFoundation reports and decodes the coded size everywhere -- naturalSize, the format
//      description's dimensions and clean aperture, and the buffer (128x96 for a 126x94 picture).
//      Silksong's cinematics are not this case: their container already says 1916x1080 and the
//      buffers arrive 1916x1080, so step 2's coded-size guard leaves them alone;
//   3. the format description's clean aperture (container 'clap'), top-left origin;
//   4. the whole luma plane.
Crop picture_crop(CVImageBufferRef image, CMFormatDescriptionRef format,
                  const std::optional<h264::SpsPictureMeta>& sps) {
    const uint32_t plane_w = static_cast<uint32_t>(CVPixelBufferGetWidthOfPlane(image, 0));
    const uint32_t plane_h = static_cast<uint32_t>(CVPixelBufferGetHeightOfPlane(image, 0));
    // CVImageBufferGetCleanRect is documented with a LOWER-LEFT origin; crop_from_rect (and the
    // copy, which starts at the top row) wants top-left. Convert: top = height - (y + h).
    CGRect clean = CVImageBufferGetCleanRect(image);
    clean.origin.y = static_cast<CGFloat>(plane_h) - (clean.origin.y + clean.size.height);
    if (auto c = crop_from_rect(clean, plane_w, plane_h);
        c && (c->width < plane_w || c->height < plane_h))
        return *c;
    if (sps && sps->coded_width == plane_w && sps->coded_height == plane_h)
        if (auto c = sps_crop(*sps)) return *c;
    if (format) {
        if (auto c = crop_from_rect(CMVideoFormatDescriptionGetCleanAperture(format, /*originIsAtTopLeft=*/true),
                                    plane_w, plane_h))
            return *c;
    }
    return Crop{0, 0, plane_w, plane_h};
}

// The stream's first SPS, from the avcC the format description carries, through the same parser
// sceVideodec2GetPictureInfo uses (#2898). Nothing for non-H.264 tracks or an unparseable SPS.
std::optional<h264::SpsPictureMeta> track_sps(CMFormatDescriptionRef format) {
    if (!format || CMFormatDescriptionGetMediaSubType(format) != kCMVideoCodecType_H264) return std::nullopt;
    const uint8_t* sps = nullptr;
    size_t sps_size = 0, count = 0;
    int nal_header_length = 0;
    if (CMVideoFormatDescriptionGetH264ParameterSetAtIndex(format, 0, &sps, &sps_size, &count,
                                                           &nal_header_length) != noErr ||
        !sps || !sps_size)
        return std::nullopt;
    std::vector<uint8_t> annexb = {0, 0, 0, 1};
    annexb.insert(annexb.end(), sps, sps + sps_size);
    h264::SpsPictureMeta meta;
    if (!h264::parse_first_sps(annexb.data(), annexb.size(), &meta)) return std::nullopt;
    return meta;
}

// Copy the decoded picture's crop into a packed NV12 packet: `height` luma rows of `width` bytes,
// then ceil(height/2) chroma rows of 2*ceil(width/2) bytes, all at stride 2*ceil(width/2). That is
// nv12_bytes(width, height) exactly when width is even; an odd width carries one pad byte per luma
// row (the stride), which consumers address through y_stride.
bool copy_picture(CVImageBufferRef image, CMTime pts, const Crop& crop, VideoPacket& packet) {
    if (!image || CVPixelBufferGetPixelFormatType(image) != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
        CVPixelBufferGetPlaneCount(image) != 2) return false;
    const uint32_t width = crop.width, height = crop.height;
    const uint32_t uv_rows = (height + 1u) / 2u;
    const uint32_t uv_row_bytes = 2u * ((width + 1u) / 2u);
    if (CVPixelBufferGetWidthOfPlane(image, 0) < crop.x + width ||
        CVPixelBufferGetHeightOfPlane(image, 0) < crop.y + height ||
        CVPixelBufferGetHeightOfPlane(image, 1) < crop.y / 2u + uv_rows) return false;
    if (CVPixelBufferLockBaseAddress(image, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) return false;
    const auto* y = static_cast<const uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(image, 0));
    const auto* uv = static_cast<const uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(image, 1));
    const size_t y_pitch = CVPixelBufferGetBytesPerRowOfPlane(image, 0);
    const size_t uv_pitch = CVPixelBufferGetBytesPerRowOfPlane(image, 1);
    const bool ok = y && uv && y_pitch >= crop.x + width && uv_pitch >= crop.x + uv_row_bytes;
    if (ok) {
        const uint32_t stride = uv_row_bytes;   // >= width; equal unless width is odd
        packet.bytes.assign(static_cast<size_t>(stride) * (height + uv_rows), 0);
        for (uint32_t row = 0; row < height; ++row)
            std::memcpy(packet.bytes.data() + static_cast<size_t>(row) * stride,
                        y + static_cast<size_t>(crop.y + row) * y_pitch + crop.x, width);
        uint8_t* dst_uv = packet.bytes.data() + static_cast<size_t>(stride) * height;
        for (uint32_t row = 0; row < uv_rows; ++row)
            std::memcpy(dst_uv + static_cast<size_t>(row) * stride,
                        uv + static_cast<size_t>(crop.y / 2u + row) * uv_pitch + crop.x, uv_row_bytes);
        packet.width = width;
        packet.height = height;
        packet.stride = stride;
        packet.pts_us = cmtime_us(pts);
    }
    CVPixelBufferUnlockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
    return ok;
}

// The video decoder: a VTDecompressionSession this backend owns, so the hardware policy is ENFORCED
// on the actual session and the actual selection is OBSERVED, rather than inferred from a codec
// capability query (VTIsHardwareDecodeSupported says hardware exists for a codec; a given format,
// configuration or a busy engine can still get software).
//   - Without PROSPER_AVP_ALLOW_SOFTWARE the session is created with
//     kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder: creation fails rather
//     than falling back, and the open is refused truthfully.
//   - With it, hardware is still preferred (EnableHardwareAcceleratedVideoDecoder) but not required.
//   - Either way kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder is read back and
//     is what the backend reports; a session that says software without the override is refused.
// Decoding is synchronous and pictures come back in DECODE order -- measured: neither
// kVTDecodeFrame_EnableTemporalProcessing nor adding asynchronous decompression reorders a B-frame
// stream here. Display order is restored by release_ready() below.
// `pts` is what the decoder callback reports and what the guest is given; `order` is the source
// sample's own presentation time on the MEDIA timeline -- the timeline its decode timestamps live
// on. They differ by the container's edit list (measured: the test clip's I-frame is sample PTS
// 0.0667 s, delivered at 0), so ordering must compare `order` with DTS, never `pts`.
struct DecodedPicture { CVImageBufferRef image = nullptr; CMTime pts = kCMTimeInvalid; CMTime order = kCMTimeInvalid; };

struct Decoder {
    VTDecompressionSessionRef session = nullptr;
    CMFormatDescriptionRef format = nullptr;   // retained
    bool hardware = false;
    std::mutex out_mutex;                      // the output callback may run on a VT thread
    std::vector<DecodedPicture> out;
    std::vector<DecodedPicture> pending;       // decoded, not yet released; ordered by PTS
    // Each DecodeFrame call's source-sample PTS, keyed by a serial passed as sourceFrameRefCon.
    // A map rather than a heap pointer the callback frees: whether VT invokes the callback for a
    // frame whose DecodeFrame call FAILED is not something this code has to know -- an unmatched
    // entry is simply erased, and nothing can be freed twice.
    uint64_t next_serial = 1;
    std::unordered_map<uint64_t, CMTime> order_by_serial;   // guarded by out_mutex

    ~Decoder() { reset(); }
    void reset() {
        if (session) {
            VTDecompressionSessionWaitForAsynchronousFrames(session);
            VTDecompressionSessionInvalidate(session);
            CFRelease(session);
            session = nullptr;
        }
        for (auto& p : take()) CVBufferRelease(p.image);
        for (auto& p : pending) CVBufferRelease(p.image);
        pending.clear();
        {
            std::lock_guard<std::mutex> lock(out_mutex);
            order_by_serial.clear();
        }
        if (format) { CFRelease(format); format = nullptr; }
        hardware = false;
    }
    std::vector<DecodedPicture> take() {
        std::lock_guard<std::mutex> lock(out_mutex);
        std::vector<DecodedPicture> pictures;
        pictures.swap(out);
        return pictures;
    }
};

// Release, in presentation order, every decoded picture whose PTS is <= `limit`; all of them when
// `limit` is invalid (end of stream). Correct with `limit` = the DTS of the next sample to be
// decoded: decode timestamps are monotonic and every frame has DTS <= PTS, so any frame with
// PTS <= limit was submitted before that sample -- and, decoding synchronously, has been output.
std::vector<DecodedPicture> release_ready(Decoder& d, CMTime limit) {
    std::vector<DecodedPicture> fresh = d.take();
    d.pending.insert(d.pending.end(), fresh.begin(), fresh.end());
    std::stable_sort(d.pending.begin(), d.pending.end(), [](const DecodedPicture& a, const DecodedPicture& b) {
        return CMTimeCompare(a.order, b.order) < 0;
    });
    size_t n = d.pending.size();
    if (CMTIME_IS_NUMERIC(limit)) {
        n = 0;
        while (n < d.pending.size() && CMTimeCompare(d.pending[n].order, limit) <= 0) ++n;
    }
    std::vector<DecodedPicture> ready(d.pending.begin(), d.pending.begin() + static_cast<ptrdiff_t>(n));
    d.pending.erase(d.pending.begin(), d.pending.begin() + static_cast<ptrdiff_t>(n));
    return ready;
}

void decoder_output(void* refcon, void* frame_refcon, OSStatus status, VTDecodeInfoFlags flags,
                    CVImageBufferRef image, CMTime pts, CMTime) {
    auto* d = static_cast<Decoder*>(refcon);
    // The source sample's media-timeline PTS, recorded per DecodeFrame call (decode_sample).
    std::optional<CMTime> order;
    {
        std::lock_guard<std::mutex> lock(d->out_mutex);
        auto it = d->order_by_serial.find(reinterpret_cast<uintptr_t>(frame_refcon));
        if (it != d->order_by_serial.end()) { order = it->second; d->order_by_serial.erase(it); }
    }
    if (status != noErr || !image || (flags & kVTDecodeInfo_FrameDropped)) {
        if (status != noErr && avp_log())
            std::fprintf(stderr, "[avp-vt] decode error %d\n", static_cast<int>(status));
        return;
    }
    CVBufferRetain(image);
    std::lock_guard<std::mutex> lock(d->out_mutex);
    d->out.push_back({image, pts, order ? *order : pts});
}

bool make_decoder(Decoder& d, CMFormatDescriptionRef format, bool allow_software, const std::string& name) {
    d.reset();
    if (!format) return false;
    NSDictionary* spec = allow_software
        ? @{(id)kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder : @YES}
        : @{(id)kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder : @YES};
    // 420v: bi-planar 4:2:0, luma plane + interleaved CbCr plane, video range -- NV12.
    NSDictionary* attrs = @{
        (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
    };
    VTDecompressionOutputCallbackRecord callback{decoder_output, &d};
    const OSStatus created = VTDecompressionSessionCreate(nullptr, format, (__bridge CFDictionaryRef)spec,
                                                          (__bridge CFDictionaryRef)attrs, &callback,
                                                          &d.session);
    if (created != noErr || !d.session) {
        // Unconditional: the title's video will not play, and this is the reason.
        std::fprintf(stderr, "[avp-vt] '%s': could not create a %s video decoder session (status %d)\n",
                     name.c_str(), allow_software ? "hardware-or-software" : "HARDWARE",
                     static_cast<int>(created));
        d.session = nullptr;
        return false;
    }
    CFBooleanRef using_hardware = nullptr;
    if (VTSessionCopyProperty(d.session, kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
                              nullptr, &using_hardware) == noErr && using_hardware) {
        d.hardware = CFBooleanGetValue(using_hardware);
        CFRelease(using_hardware);
    }
    if (!d.hardware && !allow_software) {
        std::fprintf(stderr, "[avp-vt] '%s': session reports a software decoder; refused "
                             "(PROSPER_AVP_ALLOW_SOFTWARE=1 to allow)\n", name.c_str());
        d.reset();
        return false;
    }
    CFRetain(format);
    d.format = format;
    return true;
}

// Feed one compressed sample. A mid-stream format change the session cannot take recreates it
// under the same hardware policy.
bool decode_sample(Decoder& d, CMSampleBufferRef sample, bool allow_software, const std::string& name) {
    CMFormatDescriptionRef format = CMSampleBufferGetFormatDescription(sample);
    if (format && d.format && !CMFormatDescriptionEqual(format, d.format) &&
        !VTDecompressionSessionCanAcceptFormatDescription(d.session, format)) {
        VTDecompressionSessionFinishDelayedFrames(d.session);
        VTDecompressionSessionWaitForAsynchronousFrames(d.session);
        std::vector<DecodedPicture> pending = d.take();
        if (!make_decoder(d, format, allow_software, name)) {
            for (auto& p : pending) CVBufferRelease(p.image);
            return false;
        }
        std::lock_guard<std::mutex> lock(d.out_mutex);
        d.out.insert(d.out.begin(), pending.begin(), pending.end());
    }
    VTDecodeInfoFlags info = 0;
    const uint64_t serial = d.next_serial++;
    {
        std::lock_guard<std::mutex> lock(d.out_mutex);
        d.order_by_serial[serial] = CMSampleBufferGetPresentationTimeStamp(sample);
    }
    const OSStatus st = VTDecompressionSessionDecodeFrame(d.session, sample,
                                                          0 /* synchronous, no temporal delay */,
                                                          reinterpret_cast<void*>(static_cast<uintptr_t>(serial)),
                                                          &info);
    if (st != noErr) {   // whether or not VT called back for it, its key must not linger
        std::lock_guard<std::mutex> lock(d.out_mutex);
        d.order_by_serial.erase(serial);
    }
    return st == noErr;
}

void flush_decoder(Decoder& d) {
    if (!d.session) return;
    VTDecompressionSessionFinishDelayedFrames(d.session);
    VTDecompressionSessionWaitForAsynchronousFrames(d.session);
}

bool copy_audio_sample(CMSampleBufferRef sample, const StreamInfo& info, AudioPacket& packet) {
    CMBlockBufferRef block = CMSampleBufferGetDataBuffer(sample);
    if (!block || info.audio_channels == 0) return false;
    const size_t bytes = CMBlockBufferGetDataLength(block);
    const size_t frame_bytes = static_cast<size_t>(info.audio_channels) * sizeof(int16_t);
    if (bytes < frame_bytes || bytes % frame_bytes != 0) return false;
    packet.pcm.resize(bytes / sizeof(int16_t));
    // CopyDataBytes, not GetDataPointer: the block buffer is not guaranteed to be contiguous.
    if (CMBlockBufferCopyDataBytes(block, 0, bytes, packet.pcm.data()) != kCMBlockBufferNoErr)
        return false;
    packet.channels = info.audio_channels;
    packet.sample_rate = info.audio_rate;
    packet.pts_us = cmtime_us(CMSampleBufferGetPresentationTimeStamp(sample));
    return true;
}

void decode_loop(Session& session, AVAsset* asset, AVAssetTrack* video_track,
                 AVAssetTrack* audio_track, StreamInfo info, NSData* channel_layout,
                 CMFormatDescriptionRef video_format) {
    const std::optional<h264::SpsPictureMeta> sps = track_sps(video_format);
    auto decoder = std::make_unique<Decoder>();   // stable address: it is the callback's refcon
    if (!make_decoder(*decoder, video_format, session.allow_software, session.display_name)) {
        finish_initialization(session, false);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(session.mutex);
        session.hardware_video = decoder->hardware;
    }
    Reader reader;
    uint64_t generation = 0;
    uint64_t skip_before_us = 0;   // after a seek: drop samples presented before the target
    if (!make_reader(asset, video_track, audio_track, info, channel_layout, kCMTimeZero, reader)) {
        finish_initialization(session, false);
        return;
    }
    Crop crop;
    bool video_eof = false, audio_eof = !audio_track;
    uint64_t video_pos_us = 0, audio_pos_us = 0;

    // Turn every picture the decoder has handed back into a packet. False only before the first
    // picture was accepted (the open fails) -- afterwards a bad picture ends the video stream.
    auto deliver = [&](std::vector<DecodedPicture> pictures) -> bool {
        bool ok = true;
        for (auto& picture : pictures) {
            if (!ok) { CVBufferRelease(picture.image); continue; }
            if (!session.initialized) crop = picture_crop(picture.image, video_format, sps);
            VideoPacket packet;
            const bool copied = copy_picture(picture.image, picture.pts, crop, packet);
            CVBufferRelease(picture.image);
            if (!copied) {
                if (!session.initialized) { ok = false; continue; }
                video_eof = audio_eof = true;
                mark_video_done(session, "unexpected decoded picture layout");
                ok = false;
                continue;
            }
            video_pos_us = std::max(video_pos_us, packet.pts_us);
            if (packet.pts_us + 1 < skip_before_us) continue;
            if (!session.initialized) {
                info.width = crop.width;
                info.height = crop.height;
                {
                    std::lock_guard<std::mutex> lock(session.mutex);
                    session.stream_info = info;
                }
                if (avp_log())
                    std::fprintf(stderr,
                                 "[avp-vt] opened '%s': %ux%u (crop at %u,%u) %.3f fps, "
                                 "audio=%u/%u, duration=%.3fs, decoder=%s (observed)\n",
                                 session.display_name.c_str(), info.width, info.height, crop.x,
                                 crop.y, info.fps, info.audio_channels, info.audio_rate,
                                 info.duration_us / 1e6,
                                 decoder->hardware ? "hardware VideoToolbox" : "software (explicit)");
                finish_initialization(session, true);
            }
            std::lock_guard<std::mutex> lock(session.mutex);
            if (session.seek_generation != generation) continue;   // stale: a seek raced it
            session.video_queue.push_back(std::move(packet));
            session.cv.notify_all();
        }
        return ok || session.initialized;
    };

    while (true) {
        @autoreleasepool {
            bool read_video = false, read_audio = false;
            {
                std::unique_lock<std::mutex> lock(session.mutex);
                for (;;) {
                    if (session.stopping) break;
                    if (session.seek_generation != generation) break;
                    const bool video_room = session.video_queue.size() < kVideoQueueCapacity;
                    const bool audio_room = session.audio_queue.size() < kAudioQueueCapacity;
                    read_video = !video_eof && video_room;
                    // While video runs, audio keeps pace with it (and a full audio queue drops its
                    // oldest, so an unconsumed audio stream cannot stall video). After video has
                    // ended, the tail is read only into free room: nothing can be dropped any more.
                    read_audio = !audio_eof &&
                                 (video_eof ? audio_room : audio_pos_us <= video_pos_us + kAudioLeadUs);
                    if (read_audio && read_video) {
                        // Alternate by position so neither output starves the other.
                        if (audio_pos_us <= video_pos_us) read_video = false;
                        else read_audio = false;
                    }
                    if (read_audio || read_video) break;
                    // Queue full, or finished: wait for a pull, a seek, or close.
                    session.cv.wait(lock);
                }
                if (session.stopping) break;
                if (session.seek_generation != generation) {
                    generation = session.seek_generation;
                    skip_before_us = session.seek_target_us;
                    session.video_queue.clear();
                    session.audio_queue.clear();
                    session.decode_done = false;
                    lock.unlock();
                    [reader.reader cancelReading];
                    reader = Reader{};
                    // A seek beyond the stream lands nowhere: report it rather than "succeed" into
                    // an immediate end of stream. CONFIDENCE: MED (VA-API's equivalent is the
                    // container seek's own result).
                    const bool in_range = info.duration_us == 0 || skip_before_us <= info.duration_us;
                    // The pictures between the decodable start and the target are decoded and dropped
                    // (skip_before_us), so the first one delivered is the requested one. Measured: a
                    // passthrough AVAssetReader whose timeRange starts at 0.2 s delivers from the
                    // preceding sync sample (the keyframe at DTS 0), so the reader itself supplies a
                    // decodable start; the mid-stream seek test pins the pixels this produces. The
                    // decoder is rebuilt under the same hardware policy.
                    const CMTime target = CMTimeMake(static_cast<int64_t>(skip_before_us), 1000000);
                    const bool rebuilt = in_range &&
                        make_decoder(*decoder, video_format, session.allow_software, session.display_name) &&
                        make_reader(asset, video_track, audio_track, info, channel_layout, target, reader);
                    video_eof = !rebuilt;
                    audio_eof = !rebuilt || !audio_track;
                    video_pos_us = audio_pos_us = skip_before_us;
                    std::lock_guard<std::mutex> settle(session.mutex);
                    if (session.seek_generation == generation) {
                        session.seek_settled_generation = generation;
                        session.seek_ok = rebuilt;
                        if (!rebuilt) session.decode_done = true;
                    }
                    session.cv.notify_all();
                    continue;
                }
            }

            if (read_video) {
                CMSampleBufferRef sample = [reader.video copyNextSampleBuffer];
                if (!sample) {
                    const bool failed = reader.reader.status == AVAssetReaderStatusFailed;
                    if (failed && avp_log())
                        std::fprintf(stderr, "[avp-vt] video read failed '%s': %s\n",
                                     session.display_name.c_str(),
                                     reader.reader.error
                                         ? reader.reader.error.localizedDescription.UTF8String
                                         : "(no error object)");
                    flush_decoder(*decoder);
                    if (!deliver(release_ready(*decoder, kCMTimeInvalid)) || !session.initialized) {
                        finish_initialization(session, false);
                        break;
                    }
                    video_eof = true;
                    mark_video_done(session, failed ? "video stream failed" : nullptr);
                    continue;
                }
                const bool has_data = CMSampleBufferGetNumSamples(sample) > 0 &&
                                      CMSampleBufferGetDataBuffer(sample);
                // Everything presented no later than this sample's decode time is final now.
                CMTime dts = CMSampleBufferGetDecodeTimeStamp(sample);
                if (!CMTIME_IS_NUMERIC(dts)) dts = CMSampleBufferGetPresentationTimeStamp(sample);
                if (CMTIME_IS_NUMERIC(dts) && !deliver(release_ready(*decoder, dts))) {
                    CFRelease(sample);
                    finish_initialization(session, false);
                    break;
                }
                const bool decoded = !has_data ||
                    decode_sample(*decoder, sample, session.allow_software, session.display_name);
                CFRelease(sample);
                if (!decoded) {
                    if (!session.initialized) { finish_initialization(session, false); break; }
                    video_eof = audio_eof = true;
                    mark_video_done(session, "decoder rejected a sample");
                    continue;
                }
                (void)0;   // decoded pictures wait in pending until the next sample's DTS releases them
            } else if (read_audio) {
                CMSampleBufferRef sample = [reader.audio copyNextSampleBuffer];
                if (!sample) {
                    audio_eof = true;
                    continue;
                }
                AudioPacket packet;
                const bool copied = CMSampleBufferGetNumSamples(sample) > 0 &&
                                    copy_audio_sample(sample, info, packet);
                CFRelease(sample);
                if (!copied) continue;   // an empty or malformed audio sample loses audio only
                audio_pos_us = packet.pts_us;
                if (packet.pts_us + 1 < skip_before_us) continue;
                std::lock_guard<std::mutex> lock(session.mutex);
                if (session.seek_generation != generation) continue;
                // A title may render video while routing or disabling audio elsewhere. While video
                // runs, an unconsumed audio queue must not stall the reader before the next frame.
                if (session.audio_queue.size() == kAudioQueueCapacity) session.audio_queue.pop_front();
                session.audio_queue.push_back(std::move(packet));
                session.cv.notify_all();
            }
        }
    }
    [reader.reader cancelReading];
}

void decode_session(Session& session) {
    // Every exit -- early refusals included -- releases a seek() waiting on this worker.
    struct ExitMark {
        Session& s;
        ~ExitMark() {
            std::lock_guard<std::mutex> lock(s.mutex);
            s.worker_exited = true;
            s.cv.notify_all();
        }
    } exit_mark{session};
    @autoreleasepool {
        @try {
            NSString* path = [NSString stringWithUTF8String:session.path.c_str()];
            NSURL* url = path ? [NSURL fileURLWithPath:path] : nil;
            AVURLAsset* asset = url ? [AVURLAsset URLAssetWithURL:url options:nil] : nil;
            NSArray<AVAssetTrack*>* video_tracks = asset ? [asset tracksWithMediaType:AVMediaTypeVideo] : nil;
            AVAssetTrack* video_track = video_tracks.count ? video_tracks.firstObject : nil;
            if (!video_track) {
                if (avp_log())
                    std::fprintf(stderr, "[avp-vt] open '%s': no video track\n", session.display_name.c_str());
                finish_initialization(session, false);
                return;
            }
            NSArray<AVAssetTrack*>* audio_tracks = [asset tracksWithMediaType:AVMediaTypeAudio];
            AVAssetTrack* audio_track = audio_tracks.count ? audio_tracks.firstObject : nil;

            // Width and height are taken from the first decoded picture (picture_crop); these are
            // only the provisional display size until then.
            StreamInfo info;
            const CGSize natural = video_track.naturalSize;
            info.width = static_cast<uint32_t>(natural.width + 0.5);
            info.height = static_cast<uint32_t>(natural.height + 0.5);
            info.fps = video_track.nominalFrameRate;
            info.duration_us = cmtime_us(asset.duration);
            CMFormatDescriptionRef video_format =
                video_track.formatDescriptions.count
                    ? (__bridge CMFormatDescriptionRef)video_track.formatDescriptions.firstObject : nullptr;
            const FourCharCode codec = video_format ? CMFormatDescriptionGetMediaSubType(video_format) : 0;
            NSData* channel_layout = nil;
            if (audio_track && audio_track.formatDescriptions.count) {
                const auto audio_format =
                    (__bridge CMAudioFormatDescriptionRef)audio_track.formatDescriptions.firstObject;
                const AudioStreamBasicDescription* asbd =
                    CMAudioFormatDescriptionGetStreamBasicDescription(audio_format);
                if (asbd && asbd->mChannelsPerFrame > 0 && asbd->mSampleRate > 0) {
                    info.has_audio = true;
                    info.audio_channels = asbd->mChannelsPerFrame;
                    info.audio_rate = static_cast<uint32_t>(asbd->mSampleRate + 0.5);
                    if (info.audio_channels > 2) {
                        size_t layout_size = 0;
                        const AudioChannelLayout* layout =
                            CMAudioFormatDescriptionGetChannelLayout(audio_format, &layout_size);
                        if (layout && layout_size)
                            channel_layout = [NSData dataWithBytes:layout length:layout_size];
                        else
                            info.audio_channels = 2;   // no layout to keep: AVFoundation downmixes
                    }
                }
            }
            if (!info.has_audio) audio_track = nil;

            // The hardware policy is enforced and observed on the decoder session itself
            // (make_decoder), not inferred here from a codec capability query.
            (void)codec;
            {
                std::lock_guard<std::mutex> lock(session.mutex);
                session.stream_info = info;
            }
            decode_loop(session, asset, video_track, audio_track, info, channel_layout, video_format);
        } @catch (NSException* e) {
            if (avp_log())
                std::fprintf(stderr, "[avp-vt] '%s' raised %s: %s\n", session.display_name.c_str(),
                             e.name.UTF8String, e.reason ? e.reason.UTF8String : "");
            if (!session.initialized) finish_initialization(session, false);
            else mark_video_done(session, "decode worker stopped by an exception");
        }
        if (avp_log()) std::fprintf(stderr, "[avp-vt] decode worker exiting '%s'\n", session.display_name.c_str());
    }
}

void stop_session(const std::shared_ptr<Session>& session) {
    if (!session) return;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->stopping = true;
        session->cv.notify_all();
    }
    if (session->thread.joinable()) session->thread.join();
    if (session->owns_path) ::unlink(session->path.c_str());
}

} // namespace

bool videotoolbox_display_rect_for_test(CVPixelBufferRef image, uint32_t rect[4]) {
    if (!image || !rect) return false;
    const Crop c = picture_crop(image, nullptr, std::nullopt);
    rect[0] = c.x; rect[1] = c.y; rect[2] = c.width; rect[3] = c.height;
    return true;
}

bool videotoolbox_pack_for_test(CVPixelBufferRef image, const uint32_t rect[4],
                                std::vector<uint8_t>* nv12, uint32_t* stride) {
    if (!image || !rect || !nv12 || !stride) return false;
    VideoPacket packet;
    if (!copy_picture(image, kCMTimeZero, Crop{rect[0], rect[1], rect[2], rect[3]}, packet)) return false;
    *nv12 = std::move(packet.bytes);
    *stride = packet.stride;
    return true;
}

// open_memory's private copies: `<NSTemporaryDirectory>/prosper-avp-<pid>-XXXXXX.mp4`. The pid
// keeps one process's copies distinguishable from another's in the shared per-user directory.
std::string videotoolbox_temp_prefix() {
    @autoreleasepool {
        return std::string(NSTemporaryDirectory().UTF8String) + "prosper-avp-" +
               std::to_string(static_cast<long>(::getpid())) + "-";
    }
}

struct VideoToolboxBackend::Impl {
    std::atomic<int> next_id{1};
    std::mutex mutex;
    std::unordered_map<int, std::shared_ptr<Session>> sessions;

    ~Impl() {
        std::vector<std::shared_ptr<Session>> pending;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto& [id, session] : sessions) pending.push_back(std::move(session));
            sessions.clear();
        }
        for (const auto& session : pending) stop_session(session);
    }

    std::shared_ptr<Session> get(int id) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = sessions.find(id);
        return it == sessions.end() ? nullptr : it->second;
    }

    int start(std::shared_ptr<Session> session) {
        session->allow_software = software_decode_allowed();
        Session* raw = session.get();
        session->thread = std::thread([raw] { decode_session(*raw); });
        {
            std::unique_lock<std::mutex> lock(session->mutex);
            session->cv.wait(lock, [&] { return session->initialized; });
            if (!session->init_ok) {
                lock.unlock();
                stop_session(session);
                return -1;
            }
        }
        const int id = next_id.fetch_add(1);
        std::lock_guard<std::mutex> lock(mutex);
        sessions.emplace(id, std::move(session));
        return id;
    }
};

VideoToolboxBackend::VideoToolboxBackend() : impl_(std::make_unique<Impl>()) {}
VideoToolboxBackend::~VideoToolboxBackend() = default;

bool VideoToolboxBackend::available() const { return impl_ != nullptr; }

int VideoToolboxBackend::open(const std::string& host_path) {
    if (host_path.empty() || ::access(host_path.c_str(), R_OK) != 0) {
        if (avp_log()) std::fprintf(stderr, "[avp-vt] open '%s': not readable\n", host_path.c_str());
        return -1;
    }
    auto session = std::make_shared<Session>();
    session->path = host_path;
    session->display_name = host_path;
    return impl_->start(std::move(session));
}

int VideoToolboxBackend::open_memory(const std::string& debug_name, const uint8_t* data, size_t bytes) {
    if (!data || bytes == 0) return -1;
    // AVURLAsset reads from a URL. A private temporary file is the plain way to hand it caller bytes;
    // the backend takes its copy here, as the contract requires, and unlinks it when the session ends.
    std::string pattern;
    @autoreleasepool {
        pattern = videotoolbox_temp_prefix() + "XXXXXX.mp4";
    }
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const int fd = ::mkstemps(name.data(), 4);
    if (fd < 0) return -1;
    size_t written = 0;
    while (written < bytes) {
        const ssize_t n = ::write(fd, data + written, bytes - written);
        if (n <= 0) break;
        written += static_cast<size_t>(n);
    }
    ::close(fd);
    if (written != bytes) { ::unlink(name.data()); return -1; }
    auto session = std::make_shared<Session>();
    session->path = name.data();
    session->display_name = debug_name;
    session->owns_path = true;
    return impl_->start(session);   // on failure start() already stopped the session and unlinked
}

bool VideoToolboxBackend::info(int id, StreamInfo& out) {
    auto session = impl_->get(id);
    if (!session) return false;
    std::lock_guard<std::mutex> lock(session->mutex);
    out = session->stream_info;
    return session->init_ok;
}

static void expose(const VideoPacket& packet, VideoFrame& out) {
    const size_t y_bytes = static_cast<size_t>(packet.stride) * packet.height;
    out.y = packet.bytes.data();
    out.uv = packet.bytes.data() + y_bytes;
    out.width = packet.width;
    out.height = packet.height;
    out.y_stride = packet.stride;
    out.uv_stride = packet.stride;
    out.pts_us = packet.pts_us;
}

bool VideoToolboxBackend::peek_video(int id, VideoFrame& out) {
    auto session = impl_->get(id);
    if (!session) return false;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->video_queue.empty()) return false;
    // Pointers into the queue front. A deque push_back never moves existing elements, and the
    // caller's next_video moves the front's vector (same heap block), so the peeked bytes stay put.
    expose(session->video_queue.front(), out);
    return true;
}

bool VideoToolboxBackend::next_video(int id, VideoFrame& out) {
    auto session = impl_->get(id);
    if (!session) return false;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->video_queue.empty()) return false;
    session->last_video = std::move(session->video_queue.front());
    session->video_queue.pop_front();
    session->cv.notify_all();
    expose(session->last_video, out);
    return true;
}

bool VideoToolboxBackend::next_audio(int id, AudioFrame& out) {
    auto session = impl_->get(id);
    if (!session) return false;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->audio_queue.empty()) return false;
    session->last_audio = std::move(session->audio_queue.front());
    session->audio_queue.pop_front();
    session->cv.notify_all();
    out.pcm = session->last_audio.pcm.data();
    out.channels = session->last_audio.channels;
    out.samples = static_cast<uint32_t>(session->last_audio.pcm.size() / session->last_audio.channels);
    out.sample_rate = session->last_audio.sample_rate;
    out.pts_us = session->last_audio.pts_us;
    return true;
}

bool VideoToolboxBackend::eof(int id) {
    auto session = impl_->get(id);
    if (!session) return true;
    std::lock_guard<std::mutex> lock(session->mutex);
    // Completion follows the video stream: queued audio must not keep a video consumer active once
    // the last picture is delivered (the VideoBackend::eof contract).
    return session->decode_done && session->video_queue.empty();
}

bool VideoToolboxBackend::seek(int id, uint64_t position_us) {
    auto session = impl_->get(id);
    if (!session) return false;
    std::unique_lock<std::mutex> lock(session->mutex);
    if (!session->init_ok || session->worker_exited) return false;
    // Cleared here, not only in the worker, so a pull racing the seek cannot return a pre-seek frame.
    session->video_queue.clear();
    session->audio_queue.clear();
    session->decode_done = false;
    session->seek_target_us = position_us;
    const uint64_t generation = ++session->seek_generation;
    session->cv.notify_all();
    // Block until the worker has actually repositioned (or failed to), as the VA-API backend does:
    // the HLE turns `false` into a guest-visible sceAvPlayer error (#1949).
    session->cv.wait(lock, [&] {
        return session->seek_settled_generation >= generation || session->worker_exited ||
               session->stopping;
    });
    return session->seek_settled_generation == generation && session->seek_ok;
}

void VideoToolboxBackend::close(int id) {
    std::shared_ptr<Session> session;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto it = impl_->sessions.find(id);
        if (it == impl_->sessions.end()) return;
        session = std::move(it->second);
        impl_->sessions.erase(it);
    }
    stop_session(session);
}

int VideoToolboxBackend::decoder_uses_hardware(int id) {
    auto session = impl_->get(id);
    if (!session) return -1;
    std::lock_guard<std::mutex> lock(session->mutex);
    return session->hardware_video ? 1 : 0;
}

int VideoToolboxBackend::video_queue_depth(int id) {
    auto session = impl_->get(id);
    if (!session) return -1;
    std::lock_guard<std::mutex> lock(session->mutex);
    return static_cast<int>(session->video_queue.size());
}

static VideoToolboxBackend& shared_videotoolbox_backend() {
    static VideoToolboxBackend backend;
    return backend;
}

bool install_videotoolbox_backend() {
    auto& vt = shared_videotoolbox_backend();
    if (!vt.available()) return false;
    set_backend(&vt);
    return true;
}

void uninstall_videotoolbox_backend() {
    if (backend() == &shared_videotoolbox_backend()) set_backend(nullptr);
}

} // namespace prosper::video
