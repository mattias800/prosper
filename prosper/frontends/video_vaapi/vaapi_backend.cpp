#include "vaapi_backend.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libavutil/rational.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace prosper::video {
namespace {

constexpr size_t kVideoQueueCapacity = 6;
constexpr size_t kAudioQueueCapacity = 32;

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && *value && std::strcmp(value, "0") != 0;
}

bool avp_log() { return env_enabled("PROSPER_AVP_LOG"); }
// Software decode is a REAL FFmpeg libavcodec fallback — it produces the genuine movie pixels (NOT
// the synthetic black-frame path, which stays explicit/opt-in per #841). It is HARDWARE-PREFERRED:
// av_hwdevice_ctx_create is tried first and used whenever VA-API succeeds; software only engages if
// hardware device creation fails (headless systems, no DRI render node, or no working VA driver).
// Default ON so a title's movie decodes on ANY Linux host rather than failing to open and deadlocking
// the game — PPSA02664 (Alex Kidd) hung forever in its black intro scene on hosts without VA-API
// because sceAvPlayer's source open failed and no completion event ever fired (#320). Set
// PROSPER_AVP_ALLOW_SOFTWARE=0 to force hardware-only (e.g. to detect a VA-API regression).
bool software_decode_allowed() {
    const char* value = std::getenv("PROSPER_AVP_ALLOW_SOFTWARE");
    if (value && *value) return std::strcmp(value, "0") != 0;   // honor an explicit override
    return true;                                                 // default: hw-preferred, sw fallback
}

std::string ff_error(int error) {
    char text[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(error, text, sizeof(text));
    return text;
}

struct VideoPacket {
    std::vector<uint8_t> nv12;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t stride = 0;
    uint64_t pts_us = 0;
};

struct AudioPacket {
    std::vector<int16_t> pcm;
    uint32_t channels = 0;
    uint32_t sample_rate = 0;
    uint64_t pts_us = 0;
};

struct Session {
    std::string path;
    bool allow_software = false;

    std::mutex mutex;
    std::condition_variable cv;
    bool initialized = false;
    bool init_ok = false;
    bool stopping = false;
    bool decode_done = false;
    std::string failure;
    StreamInfo stream_info;

    std::deque<VideoPacket> video_queue;
    std::deque<AudioPacket> audio_queue;
    VideoPacket last_video;
    AudioPacket last_audio;
    std::jthread thread;
};

struct HardwareSelection {
    AVPixelFormat format = AV_PIX_FMT_NONE;
    bool require_hardware = true;
};

AVPixelFormat select_video_format(AVCodecContext* context, const AVPixelFormat* formats) {
    auto* selection = static_cast<HardwareSelection*>(context->opaque);
    if (!selection || !formats) return AV_PIX_FMT_NONE;
    if (selection->format != AV_PIX_FMT_NONE) {
        for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
            if (*format == selection->format) {
                return *format;
            }
        }
    }
    if (selection->require_hardware) return AV_PIX_FMT_NONE;
    for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        const AVPixFmtDescriptor* descriptor = av_pix_fmt_desc_get(*format);
        if (descriptor && !(descriptor->flags & AV_PIX_FMT_FLAG_HWACCEL)) return *format;
    }
    return AV_PIX_FMT_NONE;
}

struct Pipeline {
    AVFormatContext* format = nullptr;
    AVCodecContext* video = nullptr;
    AVCodecContext* audio = nullptr;
    AVBufferRef* hardware_device = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* transferred = nullptr;
    SwsContext* scaler = nullptr;
    SwrContext* resampler = nullptr;
    AVSampleFormat resampler_format = AV_SAMPLE_FMT_NONE;
    int resampler_rate = 0;
    int resampler_channels = 0;

    ~Pipeline() {
        swr_free(&resampler);
        sws_freeContext(scaler);
        av_frame_free(&transferred);
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&audio);
        avcodec_free_context(&video);
        av_buffer_unref(&hardware_device);
        avformat_close_input(&format);
    }
};

bool session_stopping(Session& session, std::stop_token stop) {
    if (stop.stop_requested()) return true;
    std::lock_guard<std::mutex> lock(session.mutex);
    return session.stopping;
}

uint64_t frame_timestamp_us(const AVFrame* frame, AVRational time_base) {
    const int64_t timestamp = frame->best_effort_timestamp != AV_NOPTS_VALUE
                                  ? frame->best_effort_timestamp
                                  : frame->pts;
    if (timestamp == AV_NOPTS_VALUE) return 0;
    const int64_t value = av_rescale_q(timestamp, time_base, AVRational{1, 1'000'000});
    return value > 0 ? static_cast<uint64_t>(value) : 0;
}

bool enqueue_video(Session& session, VideoPacket packet, std::stop_token stop) {
    std::unique_lock<std::mutex> lock(session.mutex);
    session.cv.wait(lock, [&] {
        return session.stopping || stop.stop_requested() ||
               session.video_queue.size() < kVideoQueueCapacity;
    });
    if (session.stopping || stop.stop_requested()) return false;
    if (!session.initialized) {
        session.stream_info.width = packet.width;
        session.stream_info.height = packet.height;
        session.init_ok = true;
        session.initialized = true;
    }
    session.video_queue.push_back(std::move(packet));
    session.cv.notify_all();
    return true;
}

bool enqueue_audio(Session& session, AudioPacket packet, std::stop_token stop) {
    std::lock_guard<std::mutex> lock(session.mutex);
    if (session.stopping || stop.stop_requested()) return false;
    // A title can render video while routing audio elsewhere. Keep decoding rather than allowing
    // an unconsumed audio queue to block the demux worker before the next video packet.
    if (session.audio_queue.size() == kAudioQueueCapacity) session.audio_queue.pop_front();
    session.audio_queue.push_back(std::move(packet));
    session.cv.notify_all();
    return true;
}

bool convert_video_frame(Pipeline& pipeline, HardwareSelection& selection, Session& session,
                         AVFrame* decoded, AVRational time_base, std::stop_token stop,
                         std::string& failure) {
    const bool hardware = selection.format != AV_PIX_FMT_NONE &&
                          decoded->format == selection.format;
    if (!hardware && selection.require_hardware) {
        failure = "decoder returned a software frame while VA-API hardware decode is required";
        return false;
    }

    AVFrame* source = decoded;
    if (hardware) {
        av_frame_unref(pipeline.transferred);
        const int transfer = av_hwframe_transfer_data(pipeline.transferred, decoded, 0);
        if (transfer < 0) {
            failure = "VA-API surface transfer failed: " + ff_error(transfer);
            return false;
        }
        source = pipeline.transferred;
    }

    if (source->width <= 0 || source->height <= 0 ||
        (source->width & 1) || (source->height & 1)) {
        failure = "decoded video dimensions are not valid for NV12";
        return false;
    }

    VideoPacket packet;
    packet.width = static_cast<uint32_t>(source->width);
    packet.height = static_cast<uint32_t>(source->height);
    packet.stride = packet.width;
    packet.pts_us = frame_timestamp_us(decoded, time_base);
    packet.nv12.resize(static_cast<size_t>(packet.stride) * packet.height * 3 / 2);
    uint8_t* destination[4] = {
        packet.nv12.data(),
        packet.nv12.data() + static_cast<size_t>(packet.stride) * packet.height,
        nullptr,
        nullptr,
    };
    int destination_stride[4] = {
        static_cast<int>(packet.stride),
        static_cast<int>(packet.stride),
        0,
        0,
    };
    pipeline.scaler = sws_getCachedContext(
        pipeline.scaler, source->width, source->height,
        static_cast<AVPixelFormat>(source->format), source->width, source->height, AV_PIX_FMT_NV12,
        SWS_POINT, nullptr, nullptr, nullptr);
    if (!pipeline.scaler) {
        failure = "could not create NV12 conversion context";
        return false;
    }
    const int rows = sws_scale(pipeline.scaler, source->data, source->linesize, 0, source->height,
                               destination, destination_stride);
    if (rows != source->height) {
        failure = "decoded frame could not be converted to NV12";
        return false;
    }

    if (avp_log() && !session.initialized) {
        std::fprintf(stderr, "[avp-vaapi] %s decode selected for '%s' (%s -> NV12)\n",
                     hardware ? "hardware VA-API" : "software (libavcodec)",
                     session.path.c_str(), av_get_pix_fmt_name(static_cast<AVPixelFormat>(source->format)));
    }
    return enqueue_video(session, std::move(packet), stop);
}

bool convert_audio_frame(Pipeline& pipeline, Session& session, AVFrame* frame,
                         AVRational time_base, std::stop_token stop, std::string& failure) {
    const int channels = frame->ch_layout.nb_channels;
    const int rate = frame->sample_rate;
    const auto format = static_cast<AVSampleFormat>(frame->format);
    if (channels <= 0 || rate <= 0 || format == AV_SAMPLE_FMT_NONE) {
        failure = "decoded audio frame has invalid format metadata";
        return false;
    }

    if (!pipeline.resampler || pipeline.resampler_format != format ||
        pipeline.resampler_rate != rate || pipeline.resampler_channels != channels) {
        swr_free(&pipeline.resampler);
        AVChannelLayout input_layout{};
        if (frame->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC)
            av_channel_layout_default(&input_layout, channels);
        else if (av_channel_layout_copy(&input_layout, &frame->ch_layout) < 0) {
            failure = "could not copy decoded audio channel layout";
            return false;
        }
        AVChannelLayout output_layout{};
        av_channel_layout_default(&output_layout, channels);
        const int allocate = swr_alloc_set_opts2(
            &pipeline.resampler, &output_layout, AV_SAMPLE_FMT_S16, rate,
            &input_layout, format, rate, 0, nullptr);
        av_channel_layout_uninit(&output_layout);
        av_channel_layout_uninit(&input_layout);
        if (allocate < 0 || !pipeline.resampler) {
            failure = "could not create PCM conversion context: " + ff_error(allocate);
            return false;
        }
        const int initialize = swr_init(pipeline.resampler);
        if (initialize < 0) {
            failure = "could not initialize PCM conversion context: " + ff_error(initialize);
            return false;
        }
        pipeline.resampler_format = format;
        pipeline.resampler_rate = rate;
        pipeline.resampler_channels = channels;
    }

    const int output_capacity = static_cast<int>(av_rescale_rnd(
        swr_get_delay(pipeline.resampler, rate) + frame->nb_samples, rate, rate, AV_ROUND_UP));
    if (output_capacity <= 0) return true;
    AudioPacket packet;
    packet.channels = static_cast<uint32_t>(channels);
    packet.sample_rate = static_cast<uint32_t>(rate);
    packet.pts_us = frame_timestamp_us(frame, time_base);
    packet.pcm.resize(static_cast<size_t>(output_capacity) * channels);
    uint8_t* output = reinterpret_cast<uint8_t*>(packet.pcm.data());
    const int converted = swr_convert(pipeline.resampler, &output, output_capacity,
                                      const_cast<const uint8_t**>(frame->extended_data),
                                      frame->nb_samples);
    if (converted < 0) {
        failure = "audio conversion failed: " + ff_error(converted);
        return false;
    }
    packet.pcm.resize(static_cast<size_t>(converted) * channels);
    return converted == 0 || enqueue_audio(session, std::move(packet), stop);
}

void decode_session(Session& session, std::stop_token stop) {
    HardwareSelection selection;
    selection.require_hardware = !session.allow_software;
    Pipeline pipeline;
    std::string failure;
    int video_stream = -1;
    int audio_stream = -1;
    AVRational video_time_base{1, 1};
    AVRational audio_time_base{1, 1};

    int result = avformat_open_input(&pipeline.format, session.path.c_str(), nullptr, nullptr);
    if (result < 0) {
        failure = "could not open source: " + ff_error(result);
        goto finished;
    }
    result = avformat_find_stream_info(pipeline.format, nullptr);
    if (result < 0) {
        failure = "could not read stream headers: " + ff_error(result);
        goto finished;
    }

    video_stream = av_find_best_stream(pipeline.format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream < 0) {
        failure = "source has no supported video stream: " + ff_error(video_stream);
        goto finished;
    }
    video_time_base = pipeline.format->streams[video_stream]->time_base;
    {
        const AVCodecParameters* parameters = pipeline.format->streams[video_stream]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(parameters->codec_id);
        if (!codec) {
            failure = "no decoder exists for the source video codec";
            goto finished;
        }
        for (int index = 0;; ++index) {
            const AVCodecHWConfig* config = avcodec_get_hw_config(codec, index);
            if (!config) break;
            if ((config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) &&
                config->device_type == AV_HWDEVICE_TYPE_VAAPI) {
                selection.format = config->pix_fmt;
                break;
            }
        }
        if (selection.format == AV_PIX_FMT_NONE && selection.require_hardware) {
            failure = "source video codec has no FFmpeg VA-API hardware configuration";
            goto finished;
        }
        if (selection.format != AV_PIX_FMT_NONE) {
            const char* configured_device = std::getenv("PROSPER_AVP_VAAPI_DEVICE");
            if (configured_device && !*configured_device) configured_device = nullptr;
            result = av_hwdevice_ctx_create(&pipeline.hardware_device, AV_HWDEVICE_TYPE_VAAPI,
                                            configured_device, nullptr, 0);
            if (result < 0) {
                if (selection.require_hardware) {
                    failure = "could not create VA-API device: " + ff_error(result);
                    goto finished;
                }
                selection.format = AV_PIX_FMT_NONE;
                if (avp_log()) {
                    std::fprintf(stderr,
                                 "[avp-vaapi] VA-API unavailable; falling back to software decode: %s\n",
                                 ff_error(result).c_str());
                }
            }
        }
        pipeline.video = avcodec_alloc_context3(codec);
        if (!pipeline.video) {
            failure = "could not allocate video decoder";
            goto finished;
        }
        result = avcodec_parameters_to_context(pipeline.video, parameters);
        if (result < 0) {
            failure = "could not configure video decoder: " + ff_error(result);
            goto finished;
        }
        pipeline.video->pkt_timebase = video_time_base;
        pipeline.video->opaque = &selection;
        pipeline.video->get_format = select_video_format;
        if (pipeline.hardware_device)
            pipeline.video->hw_device_ctx = av_buffer_ref(pipeline.hardware_device);
        result = avcodec_open2(pipeline.video, codec, nullptr);
        if (result < 0) {
            failure = "could not start video decoder: " + ff_error(result);
            goto finished;
        }

        StreamInfo stream;
        stream.width = parameters->width > 0 ? static_cast<uint32_t>(parameters->width) : 0;
        stream.height = parameters->height > 0 ? static_cast<uint32_t>(parameters->height) : 0;
        AVRational frame_rate = av_guess_frame_rate(
            pipeline.format, pipeline.format->streams[video_stream], nullptr);
        if (frame_rate.num > 0 && frame_rate.den > 0)
            stream.fps = static_cast<float>(av_q2d(frame_rate));
        if (pipeline.format->duration != AV_NOPTS_VALUE && pipeline.format->duration > 0)
            stream.duration_us = static_cast<uint64_t>(pipeline.format->duration);
        else if (pipeline.format->streams[video_stream]->duration != AV_NOPTS_VALUE)
            stream.duration_us = static_cast<uint64_t>(std::max<int64_t>(
                0, av_rescale_q(pipeline.format->streams[video_stream]->duration,
                                video_time_base, AVRational{1, 1'000'000})));
        std::lock_guard<std::mutex> lock(session.mutex);
        session.stream_info = stream;
    }

    audio_stream = av_find_best_stream(pipeline.format, AVMEDIA_TYPE_AUDIO, -1, video_stream,
                                       nullptr, 0);
    if (audio_stream >= 0) {
        audio_time_base = pipeline.format->streams[audio_stream]->time_base;
        const AVCodecParameters* parameters = pipeline.format->streams[audio_stream]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(parameters->codec_id);
        if (!codec) {
            failure = "no decoder exists for the source audio codec";
            goto finished;
        }
        pipeline.audio = avcodec_alloc_context3(codec);
        if (!pipeline.audio) {
            failure = "could not allocate audio decoder";
            goto finished;
        }
        result = avcodec_parameters_to_context(pipeline.audio, parameters);
        if (result >= 0) pipeline.audio->pkt_timebase = audio_time_base;
        if (result < 0 || (result = avcodec_open2(pipeline.audio, codec, nullptr)) < 0) {
            failure = "could not start audio decoder: " + ff_error(result);
            goto finished;
        }
        std::lock_guard<std::mutex> lock(session.mutex);
        session.stream_info.has_audio = true;
        session.stream_info.audio_channels =
            static_cast<uint32_t>(pipeline.audio->ch_layout.nb_channels);
        session.stream_info.audio_rate = static_cast<uint32_t>(pipeline.audio->sample_rate);
    }

    pipeline.packet = av_packet_alloc();
    pipeline.frame = av_frame_alloc();
    pipeline.transferred = av_frame_alloc();
    if (!pipeline.packet || !pipeline.frame || !pipeline.transferred) {
        failure = "could not allocate decoder packets/frames";
        goto finished;
    }

    {
        auto receive_frames = [&](AVCodecContext* decoder, bool video) -> bool {
            for (;;) {
                const int receive = avcodec_receive_frame(decoder, pipeline.frame);
                if (receive == AVERROR(EAGAIN) || receive == AVERROR_EOF) return true;
                if (receive < 0) {
                    failure = std::string(video ? "video" : "audio") +
                              " decode failed: " + ff_error(receive);
                    return false;
                }
                const bool converted = video
                    ? convert_video_frame(pipeline, selection, session, pipeline.frame,
                                          video_time_base, stop, failure)
                    : convert_audio_frame(pipeline, session, pipeline.frame,
                                          audio_time_base, stop, failure);
                av_frame_unref(pipeline.frame);
                if (!converted) return false;
            }
        };
        auto submit_packet = [&](AVCodecContext* decoder, AVPacket* packet, bool video) -> bool {
            int send = avcodec_send_packet(decoder, packet);
            if (send == AVERROR(EAGAIN)) {
                if (!receive_frames(decoder, video)) return false;
                send = avcodec_send_packet(decoder, packet);
            }
            if (send == AVERROR_EOF) return true;
            if (send < 0) {
                failure = std::string(video ? "video" : "audio") +
                          " packet submission failed: " + ff_error(send);
                return false;
            }
            return receive_frames(decoder, video);
        };

        bool decode_ok = true;
        while (!session_stopping(session, stop)) {
            result = av_read_frame(pipeline.format, pipeline.packet);
            if (result == AVERROR_EOF) break;
            if (result < 0) {
                failure = "MP4 demux failed: " + ff_error(result);
                decode_ok = false;
                break;
            }
            if (pipeline.packet->stream_index == video_stream)
                decode_ok = submit_packet(pipeline.video, pipeline.packet, true);
            else if (pipeline.audio && pipeline.packet->stream_index == audio_stream)
                decode_ok = submit_packet(pipeline.audio, pipeline.packet, false);
            av_packet_unref(pipeline.packet);
            if (!decode_ok) break;
        }
        if (decode_ok && !session_stopping(session, stop)) {
            decode_ok = submit_packet(pipeline.video, nullptr, true);
            if (decode_ok && pipeline.audio)
                decode_ok = submit_packet(pipeline.audio, nullptr, false);
        }
    }

finished:
    {
        std::lock_guard<std::mutex> lock(session.mutex);
        if (!session.initialized) {
            session.initialized = true;
            session.init_ok = false;
        }
        session.failure = failure;
        session.decode_done = true;
        session.cv.notify_all();
    }
    if (!failure.empty())
        std::fprintf(stderr, "[avp-vaapi] '%s': %s\n", session.path.c_str(), failure.c_str());
    else if (avp_log())
        std::fprintf(stderr, "[avp-vaapi] decode loop stopped '%s'\n", session.path.c_str());
}

void stop_session(const std::shared_ptr<Session>& session) {
    if (!session) return;
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->stopping = true;
        session->cv.notify_all();
    }
    session->thread.request_stop();
    if (session->thread.joinable()) session->thread.join();
}

} // namespace

struct VaapiBackend::Impl {
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
        const auto found = sessions.find(id);
        return found == sessions.end() ? nullptr : found->second;
    }
};

VaapiBackend::VaapiBackend() : impl_(std::make_unique<Impl>()) {}
VaapiBackend::~VaapiBackend() = default;

bool VaapiBackend::available() const { return impl_ != nullptr; }

int VaapiBackend::open(const std::string& host_path) {
    if (!available() || host_path.empty()) return -1;
    auto session = std::make_shared<Session>();
    session->path = host_path;
    session->allow_software = software_decode_allowed();
    session->thread = std::jthread(
        [session](std::stop_token stop) { decode_session(*session, stop); });
    {
        std::unique_lock<std::mutex> lock(session->mutex);
        session->cv.wait(lock, [&] { return session->initialized; });
        if (!session->init_ok) {
            lock.unlock();
            stop_session(session);
            return -1;
        }
    }
    const int id = impl_->next_id.fetch_add(1);
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->sessions.emplace(id, std::move(session));
    return id;
}

bool VaapiBackend::info(int id, StreamInfo& out) {
    const auto session = impl_->get(id);
    if (!session) return false;
    std::lock_guard<std::mutex> lock(session->mutex);
    out = session->stream_info;
    return session->init_ok;
}

bool VaapiBackend::next_video(int id, VideoFrame& out) {
    const auto session = impl_->get(id);
    if (!session) return false;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->video_queue.empty()) return false;
    session->last_video = std::move(session->video_queue.front());
    session->video_queue.pop_front();
    session->cv.notify_all();
    const size_t y_bytes = static_cast<size_t>(session->last_video.stride) *
                           session->last_video.height;
    out.y = session->last_video.nv12.data();
    out.uv = session->last_video.nv12.data() + y_bytes;
    out.width = session->last_video.width;
    out.height = session->last_video.height;
    out.y_stride = session->last_video.stride;
    out.uv_stride = session->last_video.stride;
    out.pts_us = session->last_video.pts_us;
    return true;
}

bool VaapiBackend::next_audio(int id, AudioFrame& out) {
    const auto session = impl_->get(id);
    if (!session) return false;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->audio_queue.empty()) return false;
    session->last_audio = std::move(session->audio_queue.front());
    session->audio_queue.pop_front();
    session->cv.notify_all();
    out.pcm = session->last_audio.pcm.data();
    out.channels = session->last_audio.channels;
    out.samples = session->last_audio.channels
                      ? static_cast<uint32_t>(session->last_audio.pcm.size() /
                                              session->last_audio.channels)
                      : 0;
    out.sample_rate = session->last_audio.sample_rate;
    out.pts_us = session->last_audio.pts_us;
    return true;
}

bool VaapiBackend::eof(int id) {
    const auto session = impl_->get(id);
    if (!session) return true;
    std::lock_guard<std::mutex> lock(session->mutex);
    return session->decode_done && session->video_queue.empty();
}

void VaapiBackend::close(int id) {
    std::shared_ptr<Session> session;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto found = impl_->sessions.find(id);
        if (found == impl_->sessions.end()) return;
        session = std::move(found->second);
        impl_->sessions.erase(found);
    }
    stop_session(session);
}

VaapiBackend& shared_vaapi_backend() {
    static VaapiBackend backend_instance;
    return backend_instance;
}

bool install_vaapi_backend() {
    auto& vaapi = shared_vaapi_backend();
    if (!vaapi.available()) return false;
    set_backend(&vaapi);
    return true;
}

void uninstall_vaapi_backend() {
    auto& vaapi = shared_vaapi_backend();
    if (backend() == &vaapi) set_backend(nullptr);
}

} // namespace prosper::video
