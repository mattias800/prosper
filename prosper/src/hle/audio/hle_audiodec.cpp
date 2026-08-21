// hle_audiodec.cpp — libSceAudiodec HLE: the guest-facing compressed-audio decoder library.
//
// This is the library Unreal Engine's **Electra** media player uses to decode a movie's audio
// track. It is a different surface from libSceAjm (`hle_audio.cpp`), which the same title may also
// use for its sound middleware: AJM is a batched job queue, libSceAudiodec is a direct
// create/decode/delete handle API. Both funnel their host codec work through the same seam
// (`ajm_decoder.hpp`), so a codec implemented for one is available to the other.
//
// ## Why this exists (PPSA06367, *The Lord of the Rings: Gollum*)
//
// Every entry point here used to be unregistered, so the dispatcher's default answered `0` to all
// of them. `0` is `SCE_OK` for a status-returning contract, so Electra believed it had created an
// AAC decoder, ran a decode that wrote nothing, and then read the stream's channel count out of a
// bitstream-info block that nothing had ever filled:
//
//     eboot+0xec15e1  mov  esi, DWORD PTR [r13+0x240]   ; bsiInfo.uiChannels   == 0
//     eboot+0xec15e8  mov  eax, DWORD PTR [r13+0x260]   ; pcmItem.uiPcmSize    == 0x4000
//     eboot+0xec15f6  add  rsi, rsi                     ; 2 bytes per sample * 0 channels
//     eboot+0xec15f9  div  rsi                          ; #DE -> SIGFPE, kills the process
//
// The faulting function is identified by the string it logs, `AudioAACConvertOutput`
// (eboot+0x7e81031). See `docs/GOLLUM_STATUS.md`.
//
// ## The ABI, and where it comes from
//
// Recovered from this title's own call sites — no invented layout. `CONFIDENCE: HIGH` on every
// offset below; each is cited to the instruction that establishes it.
//
//   sceAudiodecInitLibrary(u32 codecType)                -> 0 on success   (`test eax,eax`  @ebf044)
//   sceAudiodecCreateDecoder(SceAudiodecCtrl*, u32 type) -> handle >= 0, negative error
//                                                                          (`test/jns`     @ec2985)
//   sceAudiodecDecode(s32 handle, SceAudiodecCtrl*)      -> 0 on success   (`test/jne`     @ec152f)
//
//   struct SceAudiodecCtrl {          //  32 B, built at eboot+0xec2964/0xec296b/0xec0fe4/0xec0feb
//       SceAudiodecParam*   pParam;   // +0x00
//       SceAudiodecBsiInfo* pBsiInfo; // +0x08  OUT: what the stream actually turned out to be
//       SceAudiodecAuInfo*  pAuInfo;  // +0x10
//       SceAudiodecPcmItem* pPcmItem; // +0x18
//   };
//   struct SceAudiodecAuInfo  {       //  24 B, uiSize = 0x18 written at eboot+0xec0fd9
//       u32 uiSize; u32 _reserved;
//       const void* pAuAddr;          // +0x08  eboot+0xec13e4: demux buffer base + consumed offset
//       u32 uiAuSize;                 // +0x10  IN: bytes available. OUT: bytes CONSUMED --
//                                     //        eboot+0xec153b adds it to the guest's own cursor,
//                                     //        so it is a delta and not a new absolute offset.
//       u32 _reserved2;
//   };
//   struct SceAudiodecPcmItem {       //  24 B, uiSize = 0x18 written at eboot+0xec0fce
//       u32 uiSize; u32 _reserved;
//       void* pPcmAddr;               // +0x08  eboot+0xec1460
//       u32 uiPcmSize;                // +0x10  IN: capacity in bytes (eboot+0xec14e0 writes
//                                     //        0x4000). OUT: bytes produced; eboot+0xec1542 tests
//                                     //        it against zero to detect "nothing came out".
//       u32 _reserved2;
//   };
//   struct SceAudiodecBsiInfoAac {    //  20 B, uiSize = 0x14 written at eboot+0xec28de
//       u32 uiSize;
//       u32 uiSampleRate;             // +0x04  read at eboot+0xec15ef
//       u32 uiChannels;               // +0x08  read at eboot+0xec15e1 -- the divisor above
//       u32 _reserved[2];
//   };
//   struct SceAudiodecParamAac {      //  32 B (uiSize = 0x20), built at eboot+0xec28be..0xec2959
//       u32 uiSize;                   // +0x00  0x20
//       u32 uiVersion;                // +0x04  constant 1        CONFIDENCE: LOW on the meaning
//       u32 uiObjectType;             // +0x08  MPEG-4 audioObjectType
//       u32 uiSampleRateIndex;        // +0x0c  MPEG-4 samplingFrequencyIndex -- an INDEX, NOT Hz
//       u32 uiChannelConfig;          // +0x10  MPEG-4 channelConfiguration, floored at 2 by the
//                                     //        guest (eboot+0xec292e `cmovae`)
//       u32 uiFlag;                   // +0x14  (config[0] == 1)  CONFIDENCE: LOW on the meaning
//       u32 uiEnable;                 // +0x18  constant 1        CONFIDENCE: LOW on the meaning
//       u32 _reserved;                // +0x1c
//   };
//
// **The parameter block is an unpacked MPEG-4 AudioSpecificConfig**, and that is not a guess: a
// live boot prints `{+0x08=2, +0x0c=3, +0x10=2}` for a movie `ffprobe` independently reports as
// `profile=LC, sample_rate=48000, channels=2` with a two-byte `esds` config -- object type 2 is
// AAC-LC, index 3 is 48000 Hz, channel configuration 2 is stereo. That matters because a raw AAC
// elementary stream out of an MP4 carries no in-band configuration, so these three fields are the
// ONLY thing that can configure the decoder, and they are RECONSTRUCTED rather than assumed.
//
// The trap this closes: `+0x0c` was first read as a sample rate in Hz, which is what the field's
// position suggests and what a `SceAudiodecParam` for any other codec would hold. A decoder
// configured with "3 Hz" fails at its first access unit. The live log line is what corrected it --
// see `PROSPER_AUDIODEC_LOG`, which prints the whole block for exactly this reason.
//
// ## Error values
//
// libSceAudiodec's own error space is NOT recovered — the observed call sites test only sign and
// zero, so nothing in this title's bytes distinguishes one negative value from another, and the
// charter forbids inventing platform constants. This uses prosper's existing libkernel encodings,
// which are real constants with the right sign, and says so rather than dressing a guess up as an
// SCE_AUDIODEC_ERROR_*. `CONFIDENCE: HIGH` that the sign is right and the guest branches correctly;
// `CONFIDENCE: LOW` that the numeric value matches hardware.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/audio/audio.hpp"
#include "hle/audio/ajm_decoder.hpp"
#include "hle/kernel/sce_errno.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                                        uint64_t a3, uint64_t a4, uint64_t a5)

namespace {

// --- guest structure offsets (see the header comment for the evidence) -------------------------
constexpr uint32_t kCtrlSize      = 0x20;
constexpr uint32_t kAuInfoSize    = 0x18;
constexpr uint32_t kPcmItemSize   = 0x18;
constexpr uint32_t kBsiInfoSize   = 0x14;
constexpr uint32_t kParamAacSize  = 0x20;

constexpr size_t kCtrlParam    = 0x00;
constexpr size_t kCtrlBsiInfo  = 0x08;
constexpr size_t kCtrlAuInfo   = 0x10;
constexpr size_t kCtrlPcmItem  = 0x18;

constexpr size_t kAuAddr       = 0x08;
constexpr size_t kAuSize       = 0x10;
constexpr size_t kPcmAddr      = 0x08;
constexpr size_t kPcmSize      = 0x10;
constexpr size_t kBsiSampleRate = 0x04;
constexpr size_t kBsiChannels   = 0x08;
constexpr size_t kParamObjectType     = 0x08;
constexpr size_t kParamSampleRateIndex = 0x0c;
constexpr size_t kParamChannelConfig  = 0x10;

// The one codec type observed in a live boot. `sceAudiodecInitLibrary(3)` and
// `sceAudiodecCreateDecoder(ctrl, 3)` are both called with it (nid_gate_scan reports the constant
// directly), and the streams it is created for are `codec_name=aac`.
constexpr uint32_t kCodecTypeAac = 3;

// A decode call's PCM capacity. Bounded so a malformed guest size cannot ask for an unbounded host
// allocation; the observed capacity is 0x4000 bytes.
constexpr uint32_t kMaxPcmBytes = 4u * 1024u * 1024u;
constexpr uint32_t kMaxAuBytes  = 4u * 1024u * 1024u;

// Errors. See the header comment: real constants, honest sign, unverified numeric value.
constexpr uint64_t kErrInvalid  = hle::kSceKernelErrorEINVAL;
constexpr uint64_t kErrNoCodec  = hle::kSceKernelErrorENOSYS;

bool audiodec_log() {
    static const bool on = getenv("PROSPER_AUDIODEC_LOG") != nullptr;
    return on;
}

struct Instance {
    uint32_t codec_type = 0;
    uint32_t param_sample_rate = 0;
    uint32_t param_channels = 0;
    uint32_t sample_rate = 0;      // what actually decoded; 0 until the first frame comes out
    uint32_t channels = 0;
    ajm::StreamConfig config{};    // retained so ClearContext rebuilds an identical decoder
    std::unique_ptr<ajm::StreamDecoder> decoder;
};

std::mutex g_mx;
std::map<int32_t, Instance> g_instances;
std::atomic<int32_t> g_next_handle{1};
std::atomic<bool> g_library_ready{false};

bool load_u32(uint64_t addr, uint32_t* out) {
    return addr && audio_read_bytes(addr, out, sizeof *out);
}
bool load_u64(uint64_t addr, uint64_t* out) {
    return addr && audio_read_bytes(addr, out, sizeof *out);
}
bool store_u32(uint64_t addr, uint32_t v) {
    return addr && audio_store_bytes(addr, &v, sizeof v);
}

// Read the four pointers of a SceAudiodecCtrl. Returns false if the control block itself is
// unreadable — the one condition under which nothing about the call can be trusted.
struct Ctrl {
    uint64_t param = 0, bsi_info = 0, au_info = 0, pcm_item = 0;
};
bool read_ctrl(uint64_t ctrl_addr, Ctrl* out) {
    uint8_t raw[kCtrlSize];
    if (!ctrl_addr || !audio_read_bytes(ctrl_addr, raw, sizeof raw)) return false;
    std::memcpy(&out->param,    raw + kCtrlParam,   sizeof out->param);
    std::memcpy(&out->bsi_info, raw + kCtrlBsiInfo, sizeof out->bsi_info);
    std::memcpy(&out->au_info,  raw + kCtrlAuInfo,  sizeof out->au_info);
    std::memcpy(&out->pcm_item, raw + kCtrlPcmItem, sizeof out->pcm_item);
    return true;
}

// Publish what the stream turned out to be. The guest divides its produced PCM byte count by
// `2 * uiChannels`, so a zero here is a guest-side #DE: never write one, and never claim a decode
// succeeded without having a real channel count to publish.
void publish_bsi_info(uint64_t bsi_addr, uint32_t sample_rate, uint32_t channels) {
    if (!bsi_addr || !sample_rate || !channels) return;
    store_u32(bsi_addr + kBsiSampleRate, sample_rate);
    store_u32(bsi_addr + kBsiChannels, channels);
}

} // namespace

// sceAudiodecInitLibrary(u32 codecType): make the codec available process-wide.
//
// Answering success for a codec that cannot then be created would move the failure to
// CreateDecoder, where this title's guest handles it correctly anyway — but a library that reports
// itself initialised and then refuses every decoder is a worse contract than one that refuses up
// front, so this checks the same precondition CreateDecoder does.
HLE(audiodec_init_library) {
    const uint32_t codec_type = (uint32_t)a0;
    if (audiodec_log())
        fprintf(stderr, "[audiodec] InitLibrary codec=%u\n", codec_type);
    if (codec_type != kCodecTypeAac) {
        fprintf(stderr, "[audiodec] sceAudiodecInitLibrary: codec type %u is not implemented "
                        "(only %u = AAC is). Audio for this stream will not decode.\n",
                codec_type, kCodecTypeAac);
        return kErrNoCodec;
    }
    if (!ajm::decoder_backend()) {
        fprintf(stderr, "[audiodec] sceAudiodecInitLibrary: no host codec backend is installed "
                        "(build without PROSPER_AUDIO_FFMPEG?); AAC cannot decode.\n");
        return kErrNoCodec;
    }
    g_library_ready.store(true, std::memory_order_release);
    return 0;
}

// sceAudiodecTermLibrary(u32 codecType).
HLE(audiodec_term_library) {
    (void)a0;
    std::lock_guard<std::mutex> lk(g_mx);
    g_instances.clear();
    g_library_ready.store(false, std::memory_order_release);
    return 0;
}

// sceAudiodecCreateDecoder(SceAudiodecCtrl* ctrl, u32 codecType) -> handle >= 0 | negative error.
HLE(audiodec_create_decoder) {
    const uint64_t ctrl_addr = a0;
    const uint32_t codec_type = (uint32_t)a1;

    Ctrl ctrl{};
    if (!read_ctrl(ctrl_addr, &ctrl)) return kErrInvalid;
    if (codec_type != kCodecTypeAac) {
        fprintf(stderr, "[audiodec] sceAudiodecCreateDecoder: codec type %u is not implemented\n",
                codec_type);
        return kErrNoCodec;
    }

    // The parameter block carries the stream's sample rate and channel count. A raw (MP4 `esds`)
    // AAC elementary stream cannot be decoded without them, so a param block we cannot read is a
    // hard failure rather than something to guess around.
    uint32_t param_size = 0, object_type = 0, rate_index = 0, channel_config = 0;
    if (!load_u32(ctrl.param, &param_size) ||
        !load_u32(ctrl.param + kParamObjectType, &object_type) ||
        !load_u32(ctrl.param + kParamSampleRateIndex, &rate_index) ||
        !load_u32(ctrl.param + kParamChannelConfig, &channel_config))
        return kErrInvalid;
    if (param_size < kParamAacSize) {
        fprintf(stderr, "[audiodec] sceAudiodecCreateDecoder: parameter block declares uiSize=%u, "
                        "expected at least %u (AAC). Refusing rather than reading past it.\n",
                param_size, kParamAacSize);
        return kErrInvalid;
    }
    if (audiodec_log()) {
        uint8_t raw[kParamAacSize] = {};
        audio_read_bytes(ctrl.param, raw, sizeof raw);
        fprintf(stderr, "[audiodec] CreateDecoder codec=%u param={", codec_type);
        for (size_t i = 0; i + 4 <= sizeof raw; i += 4) {
            uint32_t w = 0; std::memcpy(&w, raw + i, sizeof w);
            fprintf(stderr, " +0x%02zx=%u", i, w);
        }
        fprintf(stderr, " } ctrl={param=0x%llx bsi=0x%llx au=0x%llx pcm=0x%llx}\n",
                (unsigned long long)ctrl.param, (unsigned long long)ctrl.bsi_info,
                (unsigned long long)ctrl.au_info, (unsigned long long)ctrl.pcm_item);
    }

    ajm::DecoderBackend* backend = ajm::decoder_backend();
    if (!backend) {
        fprintf(stderr, "[audiodec] sceAudiodecCreateDecoder: no host codec backend installed\n");
        return kErrNoCodec;
    }
    // Real hardware very likely requires sceAudiodecInitLibrary first, but nothing in the observed
    // guest establishes that, and refusing on an unevidenced precondition would be inventing a
    // contract. Report it once instead: if a title ever reaches here without initialising, that log
    // line is the evidence needed to decide, and until then this stays a diagnostic.
    if (!g_library_ready.load(std::memory_order_acquire)) {
        static std::once_flag once;
        std::call_once(once, [] {
            fprintf(stderr, "[audiodec] sceAudiodecCreateDecoder called before a successful "
                            "sceAudiodecInitLibrary. Proceeding -- prosper does not enforce an "
                            "ordering it has no evidence for.\n");
        });
    }
    // The index -> Hz mapping is the published MPEG-4 table. An index the table cannot name means
    // the parameter block is not the layout recovered here, so refuse instead of decoding at a rate
    // nobody chose.
    const uint32_t sample_rate_hz = ajm::aac_sample_rate_from_index(rate_index);
    if (!sample_rate_hz || !channel_config) {
        fprintf(stderr, "[audiodec] sceAudiodecCreateDecoder: parameter block has no usable AAC "
                        "configuration (objectType=%u sampleRateIndex=%u channelConfig=%u)\n",
                object_type, rate_index, channel_config);
        return kErrInvalid;
    }

    ajm::StreamConfig cfg{};
    cfg.sample_rate = sample_rate_hz;
    cfg.channels = channel_config;
    cfg.max_channels = channel_config;
    cfg.aac_object_type = object_type;
    cfg.aac_sample_rate_index = rate_index;
    cfg.aac_channel_config = channel_config;
    std::unique_ptr<ajm::StreamDecoder> decoder =
        backend->create_configured(ajm::Codec::Aac, cfg);
    if (!decoder || !decoder->valid()) {
        fprintf(stderr, "[audiodec] sceAudiodecCreateDecoder: the host backend cannot serve an AAC "
                        "stream at %u Hz / %u channels (objectType=%u)\n",
                sample_rate_hz, channel_config, object_type);
        return kErrNoCodec;
    }

    Instance instance{};
    instance.codec_type = codec_type;
    instance.param_sample_rate = sample_rate_hz;
    instance.param_channels = channel_config;
    instance.config = cfg;
    instance.decoder = std::move(decoder);

    const int32_t handle = g_next_handle.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(g_mx);
        g_instances.emplace(handle, std::move(instance));
    }

    // Publish the configured stream shape immediately. The guest reads these after a decode, but a
    // decoder that has been told its rate and channel count knows them now, and a decode that
    // produces no PCM (a priming access unit) must not leave the block at zero.
    publish_bsi_info(ctrl.bsi_info, sample_rate_hz, channel_config);
    return (uint64_t)(int64_t)handle;
}

// sceAudiodecDeleteDecoder(s32 handle).
HLE(audiodec_delete_decoder) {
    const int32_t handle = (int32_t)a0;
    std::lock_guard<std::mutex> lk(g_mx);
    if (g_instances.erase(handle) == 0) return kErrInvalid;
    return 0;
}

// sceAudiodecClearContext(s32 handle): drop retained codec state without destroying the handle.
// Recreating the decoder is the only way this seam can express a flush, and it is the correct one:
// the alternative -- keeping the old decoder -- would emit PCM from before the seek.
HLE(audiodec_clear_context) {
    const int32_t handle = (int32_t)a0;
    ajm::DecoderBackend* backend = ajm::decoder_backend();
    std::lock_guard<std::mutex> lk(g_mx);
    auto it = g_instances.find(handle);
    if (it == g_instances.end()) return kErrInvalid;
    if (!backend) return kErrNoCodec;
    auto fresh = backend->create_configured(ajm::Codec::Aac, it->second.config);
    if (!fresh || !fresh->valid()) return kErrNoCodec;
    it->second.decoder = std::move(fresh);
    it->second.sample_rate = 0;
    it->second.channels = 0;
    return 0;
}

// sceAudiodecDecode(s32 handle, SceAudiodecCtrl* ctrl) -> 0 | negative error.
//
// In:  pAuInfo->{pAuAddr, uiAuSize}   one access unit (or a run of them)
//      pPcmItem->{pPcmAddr, uiPcmSize} the output buffer and its capacity in bytes
// Out: pAuInfo->uiAuSize   bytes consumed (a DELTA — eboot+0xec153b adds it to the guest's cursor)
//      pPcmItem->uiPcmSize bytes produced
//      pBsiInfo->{uiSampleRate, uiChannels} what the stream actually is
HLE(audiodec_decode) {
    const int32_t handle = (int32_t)a0;
    const uint64_t ctrl_addr = a1;

    Ctrl ctrl{};
    if (!read_ctrl(ctrl_addr, &ctrl)) return kErrInvalid;

    uint64_t au_addr = 0, pcm_addr = 0;
    uint32_t au_size = 0, pcm_capacity = 0;
    if (!load_u64(ctrl.au_info + kAuAddr, &au_addr) ||
        !load_u32(ctrl.au_info + kAuSize, &au_size) ||
        !load_u64(ctrl.pcm_item + kPcmAddr, &pcm_addr) ||
        !load_u32(ctrl.pcm_item + kPcmSize, &pcm_capacity))
        return kErrInvalid;
    if (au_size > kMaxAuBytes || pcm_capacity > kMaxPcmBytes) return kErrInvalid;

    std::vector<uint8_t> au(au_size);
    if (au_size && (!au_addr || !audio_read_bytes(au_addr, au.data(), au_size))) return kErrInvalid;

    // Decode under the registry lock. A StreamDecoder holds parser and codec state that no two
    // threads may advance concurrently, and holding the registry lock also keeps the instance alive
    // against a concurrent sceAudiodecDeleteDecoder. This serialises decoders against each other,
    // which is coarser than it needs to be; the observed titles run one movie stream at a time, so
    // splitting it into a per-instance lock is a change that should wait for a title that shows the
    // contention rather than being made on principle.
    uint32_t produced_bytes = 0, consumed_bytes = 0, sample_rate = 0, channels = 0;
    std::vector<int16_t> pcm(pcm_capacity / sizeof(int16_t));
    {
        std::lock_guard<std::mutex> lk(g_mx);
        auto it = g_instances.find(handle);
        if (it == g_instances.end()) return kErrInvalid;
        Instance& instance = it->second;
        if (!instance.decoder || !instance.decoder->valid()) return kErrNoCodec;

        const ajm::DecodeResult r = instance.decoder->decode(
            std::span<const uint8_t>(au.data(), au.size()),
            std::span<int16_t>(pcm.data(), pcm.size()));
        if (!r.ok) {
            fprintf(stderr, "[audiodec] sceAudiodecDecode: the host codec rejected a %u-byte "
                            "access unit on handle %d\n", au_size, handle);
            return kErrInvalid;
        }
        if (r.sample_rate) instance.sample_rate = r.sample_rate;
        if (r.channels) instance.channels = r.channels;
        produced_bytes = r.produced_bytes;
        consumed_bytes = r.consumed_bytes;
        // Fall back to what the stream was configured as until a frame has actually come out. The
        // guest divides by this, so it must never be zero on a call that reports success.
        sample_rate = instance.sample_rate ? instance.sample_rate : instance.param_sample_rate;
        channels = instance.channels ? instance.channels : instance.param_channels;
    }

    if (produced_bytes > pcm_capacity) produced_bytes = pcm_capacity;
    if (produced_bytes && (!pcm_addr || !audio_store_bytes(pcm_addr, pcm.data(), produced_bytes)))
        return kErrInvalid;

    if (!channels || !sample_rate) {
        // Refusing here is the whole point of this file: a success return with a zero channel count
        // is the divide-by-zero this library was written to stop.
        fprintf(stderr, "[audiodec] sceAudiodecDecode: refusing to report success with an unknown "
                        "stream shape (rate=%u channels=%u)\n", sample_rate, channels);
        return kErrInvalid;
    }
    publish_bsi_info(ctrl.bsi_info, sample_rate, channels);
    store_u32(ctrl.au_info + kAuSize, consumed_bytes);
    store_u32(ctrl.pcm_item + kPcmSize, produced_bytes);
    if (audiodec_log())
        fprintf(stderr, "[audiodec] Decode h=%d au=%u consumed=%u pcm=%u/%u %uHz %uch\n",
                handle, au_size, consumed_bytes, produced_bytes, pcm_capacity, sample_rate, channels);
    return 0;
}

// The decode entry points that differ only in priority or in taking a decoder-owned output buffer
// are deliberately NOT registered. Their control-block layouts have not been recovered from a live
// title, and registering a status-returning contract that writes no out-parameters is exactly the
// defect this file exists to remove. Leaving them unregistered keeps them visible in the
// `[prosper] unimplemented:` log, which is where the next title that needs them will surface.

void register_audiodec_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceAudiodecInitLibrary",   audiodec_init_library);
    R("sceAudiodecTermLibrary",   audiodec_term_library);
    R("sceAudiodecCreateDecoder", audiodec_create_decoder);
    R("sceAudiodecDeleteDecoder", audiodec_delete_decoder);
    R("sceAudiodecClearContext",  audiodec_clear_context);
    R("sceAudiodecDecode",        audiodec_decode);
    #undef R
}

// Test seam: drop every decoder and reset the library so one process can run independent cases.
void audiodec_reset_for_test() {
    std::lock_guard<std::mutex> lk(g_mx);
    g_instances.clear();
    g_library_ready.store(false, std::memory_order_release);
}

} // namespace prosper
