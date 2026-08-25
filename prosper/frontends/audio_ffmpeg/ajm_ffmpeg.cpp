#include "ajm_ffmpeg.hpp"

#include "hle/audio/ajm_decoder.hpp"

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
#include <atomic>
#include <iterator>
#include <span>
#include <vector>

namespace prosper::ajm {
namespace {
// Hot path (per decode call): getenv walks environ linearly; sample once.
inline bool audiolog() { static const bool on = getenv("PROSPER_AUDIOLOG") != nullptr; return on; }

// How a codec's byte stream is cut into packets.
//
//   Parser  every call's bytes are a slice of one continuous elementary stream and FFmpeg's parser
//           finds the frame boundaries (MP3, and ADTS-framed AAC).
//   Unit    every call's bytes are exactly one complete access unit, already de-framed by the
//           guest's own container demuxer, with the codec configuration supplied out of band
//           (raw AAC out of an MP4 `esds`). Running a parser over these finds no syncword and
//           silently yields nothing, so the parser must be bypassed rather than merely tolerated.
enum class Framing { Parser, Unit, Sniff,
    // Wwise streaming (#2981): each decode() input is a fragment of a length-prefixed Opus
    // packet stream — 2-byte little-endian size prefix, then that many payload bytes. The
    // decoder accumulates fragments and emits one complete Opus packet per decode.
    OpusStream };

// Pack a two-byte MPEG-4 AudioSpecificConfig -- the bytes an MP4 `esds` box would have carried,
// and the only thing that can configure a de-framed (raw) AAC elementary stream.
//
//   5 bits audioObjectType | 4 bits samplingFrequencyIndex | 4 bits channelConfiguration | 3 pad
//
// Every field comes from the caller, which for libSceAudiodec means it comes from the guest's own
// parameter block. Nothing here is inferred. Returns empty for a combination MPEG-4 cannot express
// in two bytes, because a wrong config decodes to noise rather than to an error and there is
// nothing downstream that could catch it: object types 31+ use the escape encoding, index 15 means
// the rate is spelled out explicitly, and channel configuration 0 means the layout is described by
// a program config element inside the stream.
inline std::vector<uint8_t> aac_audio_specific_config(uint32_t object_type, uint32_t freq_index,
                                                      uint32_t channel_config) {
    if (object_type == 0 || object_type > 30) return {};
    if (freq_index > 14) return {};
    if (channel_config == 0 || channel_config > 7) return {};
    std::vector<uint8_t> asc(2);
    asc[0] = static_cast<uint8_t>((object_type << 3) | (freq_index >> 1));
    asc[1] = static_cast<uint8_t>(((freq_index & 1u) << 7) | (channel_config << 3));
    return asc;
}

// An ADTS access unit begins with a 12-bit syncword; the four bits after it are layer (always 00
// for ADTS) and the protection-absent flag, so 0xF6 masks off the one bit that legitimately varies.
inline bool looks_like_adts(std::span<const uint8_t> data) {
    return data.size() >= 2 && data[0] == 0xFF && (data[1] & 0xF6) == 0xF0;
}

class FfmpegStreamDecoder final : public StreamDecoder {
public:
    FfmpegStreamDecoder(AVCodecID codec_id, uint32_t max_channels, Framing framing,
                        std::vector<uint8_t> extradata)
        : codec_id_(codec_id), max_channels_(max_channels), framing_(framing),
          extradata_(std::move(extradata)) {
        const AVCodec* codec = avcodec_find_decoder(codec_id_);
        if (!codec) return;
        context_ = avcodec_alloc_context3(codec);
        frame_ = av_frame_alloc();
        packet_ = av_packet_alloc();
        if (!context_ || !frame_ || !packet_) { release(); return; }
        // Sniff framing decides between Parser and Unit from the first access unit, so its codec
        // open is deferred; the other two know their answer now and fail construction if it fails.
        if (framing_ != Framing::Sniff && !open_codec(framing_)) { release(); return; }
        valid_ = true;
    }

    ~FfmpegStreamDecoder() override { release(); }

    bool valid() const override { return valid_; }

    void invalidate() override {
        if (!valid_) return;
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

        // A Sniff decoder has not opened its codec yet: the first access unit decides whether this
        // is a self-framed (ADTS) elementary stream or de-framed units needing out-of-band config.
        if (framing_ == Framing::Sniff) {
            if (input.empty()) {          // nothing to decide on yet, and nothing to decode either
                result.ok = true;
                result.produced_bytes = static_cast<uint32_t>(written_samples * sizeof(int16_t));
                return result;
            }
            const Framing resolved = looks_like_adts(input) ? Framing::Parser : Framing::Unit;
            if (!open_codec(resolved)) return fail();
            framing_ = resolved;
        }

        // FFmpeg permits optimized parsers/decoders to read AV_INPUT_BUFFER_PADDING_SIZE bytes past
        // the packet. Guest buffers provide no such promise, so stage and zero-pad them here.
        if (input.size() > std::numeric_limits<size_t>::max() - AV_INPUT_BUFFER_PADDING_SIZE)
            return result;
        std::vector<uint8_t> padded(input.size() + AV_INPUT_BUFFER_PADDING_SIZE, 0);
        if (!input.empty()) std::memcpy(padded.data(), input.data(), input.size());

        // Unit framing: the span IS one access unit. Send it whole and report it fully consumed --
        // there is no parser to tell us otherwise, and a partial consume would desynchronise the
        // guest's own demuxer cursor.
        // Wwise streaming (#2981): accumulate fragments; each decode() emits one complete
        // Opus packet (2-byte LE size prefix + payload), reassembled from the fragment stream.
        if (framing_ == Framing::OpusStream) {
            // Ring-model framing (#2981 review 2c): retire every complete packet the output span
            // can hold — 2-byte LE size prefix + payload per packet (prefix endianness and
            // framing verified against live GRIS traffic) — and report exactly the bytes retired
            // as consumed. Anything unconsumed (a trailing partial packet, or packets left for
            // lack of output space) is DROPPED here, because the guest's ring re-submits its
            // unconsumed buffer: a live trace shows the same in=524 job submitted at t=11 ms and
            // t=22 ms with iSizeConsumed=0, then consumed on the third submit once the packet
            // completed. Retaining unconsumed bytes therefore duplicates them on the retry (the
            // accumulator version corrupted 5 decode streams per session that way). No
            // accumulator: memory is bounded by one input chunk.
            size_t off = 0;   // consumed within this input
            static std::atomic<int> w{0};
            while (input.size() - off >= 2) {
                const unsigned pkt_len = input[off] | (input[off + 1] << 8);
                // Opus caps a packet at 1275 bytes per frame; a desynchronised stream reads two
                // payload bytes as a length and would otherwise wait forever for a packet that
                // will never complete. Fail the decode visibly instead — deliberately a hard
                // stop for the instance: the guest re-feeds unconsumed bytes, so nothing here
                // can resync; recovery is ClearContext (Wwise calls it on restart/seek).
                if (pkt_len > 1275u * 8) return fail();
                if (input.size() - off < 2 + pkt_len) {
                    if (audiolog() && w.fetch_add(1) < 4)
                        fprintf(stderr, "[opus-stream] waiting: avail=%zu need=%u\n",
                                input.size() - off, 2 + pkt_len);
                    break;   // trailing partial packet: dropped, the guest re-feeds it
                }
                // Backpressure mirrors the carry_ high-water check above: with no output space
                // (or retained PCM at the cap), stop retiring so the guest retries.
                if (written_samples >= output.size() ||
                    carry_.size() - carry_offset_ >= kMaxCarrySamples)
                    break;
                av_packet_unref(packet_);
                if (int ap = av_new_packet(packet_, static_cast<int>(pkt_len)); ap < 0)
                    return fail();
                std::memcpy(packet_->data, input.data() + off + 2, pkt_len);
                int send = avcodec_send_packet(context_, packet_);
                if (send == AVERROR(EAGAIN)) {
                    if (!receive_frames(&decoded_frames)) return fail();
                    send = avcodec_send_packet(context_, packet_);
                }
                if (send < 0) return fail();
                if (!receive_frames(&decoded_frames)) return fail();
                written_samples += drain(output.subspan(written_samples));
                off += 2 + pkt_len;
            }
            if (audiolog() && off)
                fprintf(stderr, "[opus-stream] decoded %zu bytes produced=%zu\n",
                        off, written_samples);
            result.ok = true;
            result.consumed_bytes = static_cast<uint32_t>(off);
            result.produced_bytes = static_cast<uint32_t>(written_samples * sizeof(int16_t));
            result.decoded_frames = decoded_frames;
            result.channels = channels_;
            result.sample_rate = sample_rate_;
            return result;
        }
        if (framing_ == Framing::Unit) {
            if (!input.empty()) {
                // Refcounted packet: av_new_packet allocates with FFmpeg's own padding and
                // proper avbuffer ownership. The old spelling (av_packet_unref + manual
                // data/size pointing at a stack-local padded buffer) left the decoder
                // referencing freed staging when the packet size SHRANK between calls —
                // Wwise-Opus's variable packet sizes hit exactly that (#2981).
                av_packet_unref(packet_);
                if (int ap = av_new_packet(packet_, static_cast<int>(input.size())); ap < 0)
                    return fail();
                std::memcpy(packet_->data, input.data(), input.size());
                if (audiolog()) {
                    fprintf(stderr, "[ajm-opus] send pkt size=%d bytes=%02x %02x %02x %02x %02x %02x %02x %02x\n",
                            packet_->size, packet_->data[0], packet_->data[1], packet_->data[2],
                            packet_->data[3], packet_->data[4], packet_->data[5], packet_->data[6],
                            packet_->data[7]);
                }
                int send = avcodec_send_packet(context_, packet_);
                if (send == AVERROR(EAGAIN)) {
                    if (!receive_frames(&decoded_frames)) return fail();
                    send = avcodec_send_packet(context_, packet_);
                }
                if (send < 0 || !receive_frames(&decoded_frames)) return fail();
                written_samples += drain(output.subspan(written_samples));
            }
            result.ok = true;
            result.consumed_bytes = static_cast<uint32_t>(input.size());
            result.produced_bytes = static_cast<uint32_t>(written_samples * sizeof(int16_t));
            result.decoded_frames = decoded_frames;
            result.channels = channels_;
            result.sample_rate = sample_rate_;
            return result;
        }

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

    // Open the codec for a concrete framing. Parser framing allocates FFmpeg's parser; Unit framing
    // installs the out-of-band extradata instead, which is the only thing that can configure a
    // de-framed stream.
    bool open_codec(Framing framing) {
        // Same decoder selection as the constructor: the Sniff path re-opens after the first
        // access unit and must not swap the codec out from under the allocated context
        // (libopus-vs-native mismatch, #2981).
        const AVCodec* codec = avcodec_find_decoder(codec_id_);
        if (!codec || !context_) return false;
        if (framing == Framing::Parser) {
            if (!parser_) parser_ = av_parser_init(codec_id_);
            if (!parser_) return false;
        } else if (!extradata_.empty()) {
            // FFmpeg takes ownership of extradata and frees it with av_free, so it must come from
            // FFmpeg's allocator and carry the padding its bitstream readers over-read into.
            uint8_t* buffer = static_cast<uint8_t*>(
                av_mallocz(extradata_.size() + AV_INPUT_BUFFER_PADDING_SIZE));
            if (!buffer) return false;
            std::memcpy(buffer, extradata_.data(), extradata_.size());
            av_freep(&context_->extradata);
            context_->extradata = buffer;
            context_->extradata_size = static_cast<int>(extradata_.size());
        }
        return avcodec_open2(context_, codec, nullptr) >= 0;
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
    AVCodecID codec_id_ = AV_CODEC_ID_NONE;
    Framing framing_ = Framing::Parser;
    std::vector<uint8_t> extradata_;
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
        // AJM packs the maximum output channel count in bits 0..6 and sample encoding in bits
        // 7..9. The seam emits signed-16 PCM today; reject S32/float instances rather than silently
        // handing them S16 bytes with a successful sideband.
        const uint32_t encoded_max_channels = static_cast<uint32_t>(instance_flags & 0x7fu);
        const uint32_t sample_encoding = static_cast<uint32_t>((instance_flags >> 7u) & 0x7u);
        if (sample_encoding != 0) return nullptr;
        StreamConfig cfg{};
        cfg.max_channels = encoded_max_channels;
        return create_configured(codec, cfg);
    }

    std::unique_ptr<StreamDecoder> create_configured(Codec codec, const StreamConfig& cfg) override {
        const uint32_t max_channels = cfg.max_channels ? cfg.max_channels : 2u;
        if (max_channels > 8) return nullptr;

        std::unique_ptr<FfmpegStreamDecoder> decoder;
        switch (codec) {
        case Codec::Mp3:
            decoder = std::make_unique<FfmpegStreamDecoder>(AV_CODEC_ID_MP3, max_channels,
                                                            Framing::Parser, std::vector<uint8_t>{});
            break;
        case Codec::Aac: {
            // Raw AAC out of an MP4 needs the container's AudioSpecificConfig, which the caller's
            // rate/channel hints reconstruct. Without them only self-framed (ADTS) input can be
            // served, so leave the extradata empty and let the sniff resolve to Parser framing --
            // a de-framed stream then fails visibly at its first access unit rather than decoding
            // silence forever.
            std::vector<uint8_t> asc;
            if (cfg.aac_object_type)
                asc = aac_audio_specific_config(cfg.aac_object_type, cfg.aac_sample_rate_index,
                                                cfg.aac_channel_config);
            decoder = std::make_unique<FfmpegStreamDecoder>(AV_CODEC_ID_AAC, max_channels,
                                                            Framing::Sniff, std::move(asc));
            break;
        }
        case Codec::Opus:
        case Codec::OpusAlt:
            // Wwise-Opus (WAVE fmt 0x3041). The live shape is streamed media:
            // sceAjmBatchJobDecodeSplit fragments of the length-prefixed Opus packet stream —
            // OpusStream framing accumulates and reassembles them. NOTE: this framing is chosen
            // unconditionally; a title submitting whole raw Opus packets per job would need
            // Unit framing (its first two bytes would be read as a length prefix here). No
            // such title is in the corpus; revisit if one appears.
            // The guest path never supplies extradata (nothing in the HLE layer populates
            // StreamConfig::extradata), and live GRIS decodes 12k+ sequential packets without
            // it — libopus via FFmpeg initialises fine with defaults for a plain stereo
            // stream. The earlier "breaks on the second packet" claim here was wrong; keep
            // cfg.extradata plumbed for callers (tests) that do supply an OpusHead.
            decoder = std::make_unique<FfmpegStreamDecoder>(AV_CODEC_ID_OPUS, max_channels,
                                                            Framing::OpusStream, cfg.extradata);
            break;
        default:
            return nullptr;   // ATRAC9 stays with the core's vendored LibAtrac9 implementation
        }
        return decoder && decoder->valid() ? std::move(decoder) : nullptr;
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
