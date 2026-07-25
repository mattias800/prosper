#include "ajm_ffmpeg.hpp"

#include "hle/ajm_decoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace prosper::ajm {
namespace {

class FfmpegMp3Decoder final : public StreamDecoder {
public:
    explicit FfmpegMp3Decoder(uint32_t max_channels): max_channels_(max_channels) {
        const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_MP3);
        if (!codec) return;
        parser_ = av_parser_init(AV_CODEC_ID_MP3);
        context_ = avcodec_alloc_context3(codec);
        frame_ = av_frame_alloc();
        packet_ = av_packet_alloc();
        if (!parser_ || !context_ || !frame_ || !packet_ ||
            avcodec_open2(context_, codec, nullptr) < 0) {
            release();
            return;
        }
        valid_ = true;
    }

    ~FfmpegMp3Decoder() override { release(); }

    bool valid() const override { return valid_; }

    void invalidate() override {
        valid_ = false;
        carry_.clear();
        carry_offset_ = 0;
    }

    DecodeResult decode(std::span<const uint8_t> input,
                        std::span<int16_t> output) override {
        DecodeResult result{};
        if (!valid_) return result;

        size_t written_samples = drain(output);
        size_t input_offset = 0;
        uint32_t decoded_frames = 0;

        // Once retained PCM reaches the high-water mark, apply backpressure before parsing or
        // sending another packet. Otherwise a caller that repeatedly supplies input with a zero-size
        // output would grow carry_ by one decoded frame per call despite the in-loop bound below.
        // Report the compressed bytes as unconsumed so the guest can retry them after draining PCM.
        if (carry_.size() - carry_offset_ >= kMaxCarrySamples) {
            result.ok = true;
            result.produced_bytes = static_cast<uint32_t>(written_samples * sizeof(int16_t));
            result.channels = channels_;
            result.sample_rate = sample_rate_;
            return result;
        }

        // FFmpeg permits optimized parsers/decoders to read AV_INPUT_BUFFER_PADDING_SIZE bytes past
        // the packet. Guest buffers provide no such promise, so stage and zero-pad them here.
        if (input.size() > std::numeric_limits<size_t>::max() - AV_INPUT_BUFFER_PADDING_SIZE)
            return result;
        std::vector<uint8_t> padded(input.size() + AV_INPUT_BUFFER_PADDING_SIZE, 0);
        if (!input.empty()) std::memcpy(padded.data(), input.data(), input.size());

        while (input_offset < input.size()) {
            uint8_t* packet_data = nullptr;
            int packet_size = 0;
            const size_t left = input.size() - input_offset;
            const int parse_size = static_cast<int>(std::min<size_t>(
                left, static_cast<size_t>(std::numeric_limits<int>::max())));
            const int used = av_parser_parse2(parser_, context_, &packet_data, &packet_size,
                                              padded.data() + input_offset, parse_size,
                                              AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (used < 0 || (used == 0 && packet_size == 0)) return fail();
            input_offset += static_cast<size_t>(used);

            if (packet_size > 0) {
                av_packet_unref(packet_);
                packet_->data = packet_data;
                packet_->size = packet_size;
                int send = avcodec_send_packet(context_, packet_);
                if (send == AVERROR(EAGAIN)) {
                    if (!receive_frames(&decoded_frames)) return fail();
                    send = avcodec_send_packet(context_, packet_);
                }
                if (send < 0 || !receive_frames(&decoded_frames)) return fail();
                written_samples += drain(output.subspan(written_samples));
            }

            // Bound retained PCM under an adversarially large input with a tiny output. Stop at a
            // frame boundary and report only the compressed prefix the persistent decoder accepted.
            if (carry_.size() - carry_offset_ >= kMaxCarrySamples) break;
        }

        result.ok = true;
        result.consumed_bytes = static_cast<uint32_t>(input_offset);
        result.produced_bytes = static_cast<uint32_t>(written_samples * sizeof(int16_t));
        result.decoded_frames = decoded_frames;
        result.channels = channels_;
        result.sample_rate = sample_rate_;
        return result;
    }

private:
    static constexpr size_t kMaxCarrySamples = 1024u * 1024u;

    DecodeResult fail() {
        // FFmpeg's parser, codec, resampler, and retained PCM all advance independently. Once any
        // one of them fails there is no reliable way to roll the others back to the byte offset AJM
        // will publish (zero on error), so make the instance terminal instead of accepting a retry
        // against partially advanced state.
        invalidate();
        return {};
    }

    void release() {
        if (swr_) swr_free(&swr_);
        if (parser_) av_parser_close(parser_);
        if (context_) avcodec_free_context(&context_);
        if (frame_) av_frame_free(&frame_);
        if (packet_) av_packet_free(&packet_);
        av_channel_layout_uninit(&input_layout_);
        valid_ = false;
    }

    bool configure_resampler(const AVFrame* frame) {
        const AVSampleFormat format = static_cast<AVSampleFormat>(frame->format);
        const bool same = swr_ && input_format_ == format && input_rate_ == frame->sample_rate &&
                          av_channel_layout_compare(&input_layout_, &frame->ch_layout) == 0;
        if (same) return true;

        const uint32_t next_channels = static_cast<uint32_t>(frame->ch_layout.nb_channels);
        // AJM instance flags bound the guest output contract. This backend currently preserves the
        // compressed stream's channel layout rather than downmixing, so fail truthfully instead of
        // returning more interleaved channels than the guest requested.
        if (next_channels == 0 || next_channels > max_channels_) return false;
        if (channels_ && next_channels != channels_ && carry_offset_ != carry_.size()) return false;

        if (swr_) swr_free(&swr_);
        av_channel_layout_uninit(&input_layout_);
        if (av_channel_layout_copy(&input_layout_, &frame->ch_layout) < 0) return false;
        AVChannelLayout output_layout{};
        if (av_channel_layout_copy(&output_layout, &frame->ch_layout) < 0) return false;
        const int rc = swr_alloc_set_opts2(&swr_, &output_layout, AV_SAMPLE_FMT_S16,
                                           frame->sample_rate, &frame->ch_layout, format,
                                           frame->sample_rate, 0, nullptr);
        av_channel_layout_uninit(&output_layout);
        if (rc < 0 || !swr_ || swr_init(swr_) < 0) {
            if (swr_) swr_free(&swr_);
            return false;
        }
        input_format_ = format;
        input_rate_ = frame->sample_rate;
        channels_ = next_channels;
        sample_rate_ = static_cast<uint32_t>(frame->sample_rate);
        return channels_ > 0 && sample_rate_ > 0;
    }

    bool receive_frames(uint32_t* decoded_frames) {
        for (;;) {
            const int rc = avcodec_receive_frame(context_, frame_);
            if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) return true;
            if (rc < 0 || !configure_resampler(frame_)) return false;

            const int capacity = swr_get_out_samples(swr_, frame_->nb_samples);
            if (capacity < 0) return false;
            const size_t old_size = carry_.size();
            if (channels_ && static_cast<size_t>(capacity) >
                std::numeric_limits<size_t>::max() / channels_) return false;
            const size_t capacity_samples = static_cast<size_t>(capacity) * channels_;
            if (capacity_samples > carry_.max_size() - old_size) return false;
            carry_.resize(old_size + capacity_samples);
            uint8_t* destination = reinterpret_cast<uint8_t*>(carry_.data() + old_size);
            const int converted = swr_convert(swr_, &destination, capacity,
                                              const_cast<const uint8_t**>(frame_->extended_data),
                                              frame_->nb_samples);
            if (converted < 0) return false;
            carry_.resize(old_size + static_cast<size_t>(converted) * channels_);
            if (decoded_frames) ++*decoded_frames;
            av_frame_unref(frame_);
        }
    }

    size_t drain(std::span<int16_t> output) {
        const size_t available = carry_.size() - carry_offset_;
        size_t count = std::min(available, output.size());
        if (channels_ > 0) count -= count % channels_; // never expose a partial interleaved frame
        if (count) {
            std::memcpy(output.data(), carry_.data() + carry_offset_, count * sizeof(int16_t));
            carry_offset_ += count;
        }
        if (carry_offset_ == carry_.size()) {
            carry_.clear();
            carry_offset_ = 0;
        } else if (carry_offset_ > kMaxCarrySamples / 2) {
            carry_.erase(carry_.begin(), carry_.begin() + static_cast<ptrdiff_t>(carry_offset_));
            carry_offset_ = 0;
        }
        return count;
    }

    bool valid_ = false;
    AVCodecParserContext* parser_ = nullptr;
    AVCodecContext* context_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* packet_ = nullptr;
    SwrContext* swr_ = nullptr;
    AVChannelLayout input_layout_{};
    AVSampleFormat input_format_ = AV_SAMPLE_FMT_NONE;
    int input_rate_ = 0;
    uint32_t max_channels_ = 0;
    uint32_t channels_ = 0;
    uint32_t sample_rate_ = 0;
    std::vector<int16_t> carry_;
    size_t carry_offset_ = 0;
};

class FfmpegDecoderBackend final : public DecoderBackend {
public:
    std::unique_ptr<StreamDecoder> create(Codec codec, uint64_t instance_flags) override {
        if (codec != Codec::Mp3) return nullptr;
        // AJM packs the maximum output channel count in bits 0..6 and sample encoding in bits
        // 7..9. The seam emits signed-16 PCM today; reject S32/float instances rather than silently
        // handing them S16 bytes with a successful sideband.
        const uint32_t encoded_max_channels = static_cast<uint32_t>(instance_flags & 0x7fu);
        const uint32_t max_channels = encoded_max_channels ? encoded_max_channels : 2u;
        const uint32_t sample_encoding = static_cast<uint32_t>((instance_flags >> 7u) & 0x7u);
        if (max_channels > 8 || sample_encoding != 0) return nullptr;
        auto decoder = std::make_unique<FfmpegMp3Decoder>(max_channels);
        return decoder->valid() ? std::move(decoder) : nullptr;
    }
};

FfmpegDecoderBackend g_ffmpeg_backend;

} // namespace

bool install_ffmpeg_decoder_backend() {
    auto probe = g_ffmpeg_backend.create(Codec::Mp3, 0); // default stereo signed-16 contract
    if (!probe) return false;
    set_decoder_backend(&g_ffmpeg_backend);
    return true;
}

void uninstall_ffmpeg_decoder_backend() {
    if (decoder_backend() == &g_ffmpeg_backend) set_decoder_backend(nullptr);
}

} // namespace prosper::ajm
