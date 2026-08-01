#pragma once
// launcher_media.hpp — parsing and policy for the library view's per-title background art and focus
// music (#1630). Pure: no SDL, no Vulkan, no filesystem. Everything here is decided from bytes already
// read and from clock values already sampled, so the whole surface is unit-tested without a window, a
// device, or a real dump — the same regression seam game_library.hpp uses.
//
// The two container formats are read strictly rather than optimistically. Both ship inside the game
// dump, so a file that does not match what we expect is a file we do not understand, and the right
// answer is to skip it: a missing background or a silent title is normal (17 of 30 local dumps have no
// snd0.at9 at all), whereas guessing at a layout produces plausible-looking garbage. Every rejection
// carries a `reason` so the log says which file was skipped and why.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace prosper::frontend {

// --- little-endian scalar reads -------------------------------------------------------------------
// Byte assembly rather than a cast: the offsets below are not guaranteed to be aligned, and the files
// are little-endian regardless of the host.

inline uint16_t le16_at(const std::string& b, size_t off) {
    if (off + 2 > b.size()) return 0;
    return static_cast<uint16_t>(static_cast<uint8_t>(b[off]) |
                                 (static_cast<uint16_t>(static_cast<uint8_t>(b[off + 1])) << 8));
}

inline uint32_t le32_at(const std::string& b, size_t off) {
    if (off + 4 > b.size()) return 0;
    return static_cast<uint32_t>(static_cast<uint8_t>(b[off])) |
           (static_cast<uint32_t>(static_cast<uint8_t>(b[off + 1])) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(b[off + 2])) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(b[off + 3])) << 24);
}

// The four bytes at `off` equal `tag` (a 4-character literal). Bounds-checked.
inline bool fourcc_at(const std::string& b, size_t off, const char (&tag)[5]) {
    if (off + 4 > b.size()) return false;
    return b[off] == tag[0] && b[off + 1] == tag[1] && b[off + 2] == tag[2] && b[off + 3] == tag[3];
}

// --- DDS (sce_sys/pic1.dds) ------------------------------------------------------------------------
//
// Layout, confirmed byte-for-byte on all 30 local dumps: "DDS " magic, a 124-byte DDS_HEADER, and —
// because ddspf.dwFourCC is "DX10" — a further 20-byte DDS_HEADER_DXT10, so pixel data begins at 148.
// dxgiFormat is 98 (BC7_UNORM), the image is 3840x2160 with one mip level, and the file is exactly
// 148 + 3840*2160 bytes (BC7 stores one byte per texel: 16 bytes per 4x4 block).
//
// Nothing above is assumed at runtime — each field is checked, because these numbers describe the
// dumps that happen to be on hand, not a guarantee about every PS5 title.

inline constexpr size_t kDdsMagicSize    = 4;    // "DDS "
inline constexpr size_t kDdsHeaderSize   = 124;  // DDS_HEADER.dwSize, and its actual length
inline constexpr size_t kDdsDx10Size     = 20;   // DDS_HEADER_DXT10
inline constexpr size_t kDdsPixelOffset  = kDdsMagicSize + kDdsHeaderSize + kDdsDx10Size;  // 148

// DXGI_FORMAT values we can hand to Vulkan directly.
inline constexpr uint32_t kDxgiBc7Unorm     = 98;
inline constexpr uint32_t kDxgiBc7UnormSrgb = 99;

// A sane ceiling. 4K key art is 3840x2160; this only exists so a corrupt header cannot drive an
// allocation or an overflow, and it is far above any plausible background.
inline constexpr uint32_t kDdsMaxDimension = 16384;

struct DdsBc7Image {
    bool     ok           = false;
    uint32_t width        = 0;
    uint32_t height       = 0;
    bool     srgb         = false;  // dxgiFormat 99 rather than 98
    size_t   data_offset  = 0;      // where the block data starts (148 for every observed file)
    size_t   data_size    = 0;      // exactly the bytes mip 0 needs, never "rest of file"
    const char* reason    = "";     // why it was rejected; "" when ok
};

// Bytes one BC7 mip level occupies: 16 per 4x4 block, partial blocks rounded up. 64-bit throughout so
// a hostile width/height cannot wrap the product.
inline uint64_t bc7_level_bytes(uint32_t width, uint32_t height) {
    const uint64_t bx = (static_cast<uint64_t>(width) + 3) / 4;
    const uint64_t by = (static_cast<uint64_t>(height) + 3) / 4;
    return bx * by * 16ull;
}

// Validate a .dds and locate its BC7 mip-0 block data. Rejects (never throws, never reads out of
// bounds) anything that is not a single-image, non-array, non-cube 2D BC7 texture that the file
// actually contains in full.
inline DdsBc7Image parse_dds_bc7(const std::string& bytes) {
    DdsBc7Image img;
    const auto fail = [&](const char* why) { img.ok = false; img.reason = why; return img; };

    if (bytes.size() < kDdsPixelOffset) return fail("shorter than a DDS+DX10 header");
    if (!fourcc_at(bytes, 0, "DDS ")) return fail("not a DDS file");
    if (le32_at(bytes, 4) != kDdsHeaderSize) return fail("unexpected DDS header size");

    // ddspf sits at 76: dwSize@76, dwFlags@80, dwFourCC@84.
    constexpr uint32_t kDdpfFourCC = 0x4;
    if ((le32_at(bytes, 80) & kDdpfFourCC) == 0) return fail("pixel format is not FourCC");
    if (!fourcc_at(bytes, 84, "DX10")) return fail("FourCC is not DX10");

    // DDS_HEADER_DXT10 at 128: dxgiFormat, resourceDimension, miscFlag, arraySize, miscFlags2.
    const uint32_t dxgi     = le32_at(bytes, 128);
    const uint32_t dimension = le32_at(bytes, 132);
    const uint32_t miscFlag  = le32_at(bytes, 136);
    const uint32_t arraySize = le32_at(bytes, 140);

    // CONFIDENCE: HIGH for 98 — every local dump uses it. CONFIDENCE: MED for 99, which is the same
    // block encoding tagged sRGB; it maps to a real Vulkan format and costs two lines, but no dump on
    // hand exercises it.
    if (dxgi != kDxgiBc7Unorm && dxgi != kDxgiBc7UnormSrgb) return fail("not BC7_UNORM");
    constexpr uint32_t kD3d10ResourceDimensionTexture2D = 3;
    if (dimension != kD3d10ResourceDimensionTexture2D) return fail("not a 2D texture");
    constexpr uint32_t kDdsResourceMiscTextureCube = 0x4;
    if (miscFlag & kDdsResourceMiscTextureCube) return fail("cube maps are not supported");
    // arraySize is 0 in some writers' output for a plain 2D texture, so treat 0 and 1 alike and reject
    // only a genuine array, whose extra slices we would silently ignore.
    if (arraySize > 1) return fail("texture arrays are not supported");

    const uint32_t height = le32_at(bytes, 12);
    const uint32_t width  = le32_at(bytes, 16);
    if (width == 0 || height == 0) return fail("zero-sized image");
    if (width > kDdsMaxDimension || height > kDdsMaxDimension) return fail("implausible dimensions");

    const uint64_t need = bc7_level_bytes(width, height);
    if (bytes.size() - kDdsPixelOffset < need) return fail("truncated: mip 0 is incomplete");

    img.ok          = true;
    img.width       = width;
    img.height      = height;
    img.srgb        = (dxgi == kDxgiBc7UnormSrgb);
    img.data_offset = kDdsPixelOffset;
    img.data_size   = static_cast<size_t>(need);
    img.reason      = "";
    return img;
}

// --- ATRAC9 in RIFF/WAVE (sce_sys/snd0.at9) --------------------------------------------------------
//
// A normal WAVE file whose format tag is WAVE_FORMAT_EXTENSIBLE (0xFFFE) and whose SubFormat GUID is
// ATRAC9's. The chunk sequence is walked rather than indexed: the local files are split between
// fmt/fact/data and fmt/fact/smpl/data, so any hardcoded data offset would work on some dumps and
// silently read the wrong bytes on others.
//
// The `fmt ` extension is 34 bytes:
//   0  samplesPerBlock (2)   6  SubFormat GUID (16)   26  configData (4)
//   2  channelMask     (4)  22  versionInfo    (4)    30  reserved   (4)
// The decoder wants configData at 26. Offset 22 is versionInfo — constant 0x00000001 in every local
// file, and close enough in shape to be accepted as a config without complaint, which is exactly the
// kind of quiet mis-parse this comment exists to prevent.

inline constexpr uint16_t kWaveFormatExtensible = 0xFFFE;
inline constexpr size_t   kAt9FmtExtensionSize  = 34;
inline constexpr size_t   kAt9ConfigOffset      = 26;   // within the extension

// ATRAC9's SubFormat GUID, {47E142D2-36BA-4D8D-88FC-61654F8C836C}, in the little-endian byte order a
// GUID is serialized with.
inline const uint8_t* atrac9_subformat_guid() {
    static const uint8_t kGuid[16] = {0xD2, 0x42, 0xE1, 0x47, 0xBA, 0x36, 0x8D, 0x4D,
                                      0x88, 0xFC, 0x61, 0x65, 0x4F, 0x8C, 0x83, 0x6C};
    return kGuid;
}

// A forward loop from the RIFF `smpl` chunk, in decoded sample-frames (per channel). `end` is the last
// frame played, per the RIFF spec, so playback covers [start, end] and then jumps back to start.
//
// Worth honouring rather than restarting the file: of the 9 local tracks that carry smpl, 5 declare a
// loop that begins well into the track (up to 1,234,542 frames — 25 s of intro), and one ends 607,162
// frames before the data does, so it has a tail that must NOT be heard when looping. Ignoring these
// would play the intro on every repeat and let a track run into its own ending.
struct At9Loop {
    bool     ok    = false;
    uint32_t start = 0;
    uint32_t end   = 0;   // inclusive
};

inline constexpr uint32_t kSmplLoopForward = 0;

struct At9Clip {
    bool     ok                = false;
    uint16_t channels          = 0;
    uint32_t sample_rate       = 0;
    uint32_t block_align       = 0;   // compressed bytes per superframe
    uint32_t samples_per_block = 0;   // decoded sample-frames per superframe, per channel
    uint8_t  config[4]         = {0, 0, 0, 0};
    size_t   data_offset       = 0;
    size_t   data_size         = 0;   // truncated to whole superframes
    At9Loop  loop;                    // absent (ok == false) on the 4 local tracks with no smpl chunk
    const char* reason         = "";

    // Decoded sample-frames per channel in the whole clip.
    uint64_t total_frames() const {
        if (block_align == 0) return 0;
        return static_cast<uint64_t>(data_size / block_align) * samples_per_block;
    }
};

// Where playback resumes after `frame`, given the clip's loop. Returns the next frame to play.
//
// Whole-file looping is the fallback, used when the track has no smpl chunk or its loop did not
// validate — a seam at the join, but a track that keeps playing, which is what a launcher needs.
inline uint64_t at9_next_frame(const At9Clip& clip, uint64_t frame) {
    const uint64_t total = clip.total_frames();
    if (total == 0) return 0;
    const uint64_t next = frame + 1;
    if (clip.loop.ok) {
        if (next > clip.loop.end) return clip.loop.start;
        return next;
    }
    return next >= total ? 0 : next;
}

// Validate a .at9 and locate its config word and compressed data.
//
// `data_size` is rounded DOWN to a whole multiple of `block_align`. The superframe is the only safe
// advance unit for Atrac9Decoder (its per-frame consumed counts drift), so a trailing partial block is
// not decodable and keeping it would only invite an over-read at the end of the track.
inline At9Clip parse_at9_riff(const std::string& bytes) {
    At9Clip clip;
    const auto fail = [&](const char* why) { clip.ok = false; clip.reason = why; return clip; };

    if (bytes.size() < 12) return fail("shorter than a RIFF header");
    if (!fourcc_at(bytes, 0, "RIFF")) return fail("not a RIFF file");
    if (!fourcc_at(bytes, 8, "WAVE")) return fail("not a WAVE file");

    // Bound the walk by the RIFF size as well as the file: a declared size larger than the file must
    // not let the walk run past the end, and one smaller must not let it read trailing junk.
    const uint64_t declared = static_cast<uint64_t>(le32_at(bytes, 4)) + 8;
    const size_t   end = static_cast<size_t>(declared < bytes.size() ? declared : bytes.size());

    bool   haveFmt = false, haveData = false;
    size_t dataOff = 0;
    uint64_t dataSize = 0;
    // Held aside until the walk finishes: smpl precedes data in every local file, so the total frame
    // count needed to range-check the loop is not known yet at the point the chunk is read.
    bool     sawLoop = false;
    uint32_t loopStart = 0, loopEnd = 0;

    for (size_t off = 12; off + 8 <= end;) {
        const uint32_t size = le32_at(bytes, off + 4);
        const size_t   body = off + 8;
        // Stop rather than wrap: a size that runs past the chunk area means the file is malformed from
        // here on, and every later offset derived from it would be fiction.
        if (static_cast<uint64_t>(body) + size > static_cast<uint64_t>(end)) break;

        if (fourcc_at(bytes, off, "fmt ")) {
            if (size < 18) return fail("fmt chunk too small");
            if (le16_at(bytes, body) != kWaveFormatExtensible) return fail("not WAVE_FORMAT_EXTENSIBLE");
            clip.channels    = le16_at(bytes, body + 2);
            clip.sample_rate = le32_at(bytes, body + 4);
            clip.block_align = le16_at(bytes, body + 12);
            const uint16_t cbSize = le16_at(bytes, body + 16);
            if (cbSize < kAt9FmtExtensionSize || size < 18 + kAt9FmtExtensionSize)
                return fail("fmt extension too small for ATRAC9");
            const size_t ext = body + 18;
            for (size_t i = 0; i < 16; i++)
                if (static_cast<uint8_t>(bytes[ext + 6 + i]) != atrac9_subformat_guid()[i])
                    return fail("SubFormat GUID is not ATRAC9");
            clip.samples_per_block = le16_at(bytes, ext + 0);
            for (size_t i = 0; i < 4; i++)
                clip.config[i] = static_cast<uint8_t>(bytes[ext + kAt9ConfigOffset + i]);
            haveFmt = true;
        } else if (fourcc_at(bytes, off, "data")) {
            dataOff  = body;
            dataSize = size;
            haveData = true;
        } else if (fourcc_at(bytes, off, "smpl")) {
            // 36-byte header then 24 bytes per loop; numSampleLoops at 28. Only the first forward loop
            // is used — every local file declares exactly one, and a second would have no meaning for
            // a single looping bed.
            if (size >= 36) {
                const uint32_t loops = le32_at(bytes, body + 28);
                if (loops >= 1 && size >= 36 + 24) {
                    const size_t l = body + 36;
                    if (le32_at(bytes, l + 4) == kSmplLoopForward) {
                        loopStart = le32_at(bytes, l + 8);
                        loopEnd   = le32_at(bytes, l + 12);
                        sawLoop   = true;
                    }
                }
            }
        }

        // Chunks are word-aligned: an odd size is followed by a pad byte that is not part of it.
        const uint64_t next = static_cast<uint64_t>(off) + 8 + size + (size & 1u);
        if (next <= off) break;               // defensive: never move backwards or stall
        off = static_cast<size_t>(next);
    }

    if (!haveFmt)  return fail("no fmt chunk");
    if (!haveData) return fail("no data chunk");
    if (clip.channels == 0)    return fail("zero channels");
    if (clip.sample_rate == 0) return fail("zero sample rate");
    if (clip.block_align == 0) return fail("zero block align");
    // A zero superframe would make total_frames() zero and every cursor computation meaningless. The
    // runtime cross-checks this against the decoder anyway, but a parsed clip must never escape this
    // function in a state where its own arithmetic is undefined.
    if (clip.samples_per_block == 0) return fail("zero samples per superframe");

    dataSize -= dataSize % clip.block_align;          // whole superframes only
    if (dataSize == 0) return fail("no complete superframe in data");

    clip.ok          = true;
    clip.data_offset = dataOff;
    clip.data_size   = static_cast<size_t>(dataSize);
    clip.reason      = "";

    // Range-check the loop against what the file actually decodes to. A loop that points past the end
    // is dropped rather than clamped: a clamp invents an edit the author did not ask for, whereas
    // falling back to whole-file looping is a behaviour we already support and can describe.
    if (sawLoop) {
        const uint64_t total = clip.total_frames();
        if (loopStart < loopEnd && static_cast<uint64_t>(loopEnd) < total) {
            clip.loop.ok    = true;
            clip.loop.start = loopStart;
            clip.loop.end   = loopEnd;
        }
    }
    return clip;
}

// --- asset locations -------------------------------------------------------------------------------

// Backgrounds to try for a title, best first. pic1 is the clean plate — the same artwork as pic0 with
// the title logo and characters left out — which is why the console puts it behind UI and why it is
// preferred here. pic0.png is the degraded fallback for a device without BC support: it is real 4K art
// of the same scene, just with the logo composited in. There is deliberately no synthetic last resort;
// a title with no usable art shows no background rather than something invented.
struct BackgroundCandidate {
    std::string path;
    bool        block_compressed = false;   // .dds (needs textureCompressionBC) vs .png (stb_image)
};

inline std::vector<BackgroundCandidate> background_candidates(const std::string& app0_root,
                                                              bool bc_supported) {
    std::vector<BackgroundCandidate> out;
    if (app0_root.empty()) return out;
    if (bc_supported) out.push_back({app0_root + "/sce_sys/pic1.dds", true});
    out.push_back({app0_root + "/sce_sys/pic0.png", false});
    return out;
}

// Where a title's focus music would be. Present in 13 of 30 local dumps; absent is normal, and the
// caller treats a missing file as silence rather than as an error.
inline std::string music_path_for(const std::string& app0_root) {
    if (app0_root.empty()) return "";
    return app0_root + "/sce_sys/snd0.at9";
}

// --- focus policy ----------------------------------------------------------------------------------

// Milliseconds focus must rest on a title before its 8 MB background and its music are loaded. Holding
// an arrow key walks the grid far faster than this, so a sweep across the library starts exactly one
// load — the one the user stopped on — instead of one per cell.
inline constexpr uint64_t kFocusDebounceMs = 180;

// Transition lengths, applied to the background alpha and the audio gain alike. The two halves run in
// sequence, never together: the outgoing asset fades to nothing, and only then does the incoming one
// fade up. See fade_state_at().
inline constexpr uint64_t kFadeOutMs = 200;
inline constexpr uint64_t kFadeInMs  = 250;

// Backgrounds kept resident: the incoming one, the outgoing one still visible in a cross-fade, and one
// slack so stepping back to the previous title is instant. At ~8 MB of BC7 apiece this is ~24 MB, where
// preloading all 30 local titles would be ~249 MB of VRAM.
inline constexpr size_t kBackgroundCacheCapacity = 3;

// Has focus rested long enough to start loading? `changed_ms` is when the selection last moved.
inline bool focus_load_due(uint64_t changed_ms, uint64_t now_ms, uint64_t debounce_ms = kFocusDebounceMs) {
    return now_ms >= changed_ms && (now_ms - changed_ms) >= debounce_ms;
}

// --- transition ------------------------------------------------------------------------------------
//
// A focus change plays out as two consecutive fades, NOT a cross-fade: the outgoing background and
// track fade to nothing, and only when that finishes does the incoming pair fade up. Blending two 4K
// backgrounds would read as a muddy double exposure, and summing two tracks doubles the loudness for
// the length of the overlap.
//
// The fade-out begins as soon as the debounce expires, in parallel with the asynchronous load, so the
// load latency hides behind it instead of adding to it. When the outgoing fade finishes before the new
// asset has arrived, the transition simply holds at nothing — a brief empty background — and the
// fade-in starts from the moment the asset becomes ready. Nothing waits on the UI thread at any point.

enum class FadeStage {
    out,   // the previous asset is fading down, or has, and the next is not showing yet
    in,    // the next asset is fading up
    idle,  // settled on the next asset
};

struct FadeState {
    FadeStage stage     = FadeStage::idle;
    float     old_alpha = 0.0f;   // opacity/gain for the OUTGOING asset
    float     new_alpha = 1.0f;   // opacity/gain for the INCOMING asset
    bool      done      = false;  // the transition is over; the caller may release the outgoing asset
};

// Where a transition that began at `started_ms` stands at `now_ms`.
//
// `ready_ms` is when the incoming asset finished loading, and is only read when `new_ready` is true —
// a load that has not landed yet keeps the transition held at nothing rather than fading up an image
// that does not exist. The fade-in is anchored at whichever came later, the end of the fade-out or the
// asset's arrival, so a slow load produces a full-length fade-in rather than a partial one that jumps.
//
// INVARIANT, and the reason this is a pure function with its own test: old_alpha and new_alpha are
// never both non-zero. There is no instant at which two backgrounds are blended or two tracks audible.
inline FadeState fade_state_at(uint64_t started_ms, uint64_t now_ms, bool new_ready, uint64_t ready_ms,
                               uint64_t out_ms = kFadeOutMs, uint64_t in_ms = kFadeInMs) {
    FadeState s;
    const uint64_t elapsed = now_ms > started_ms ? now_ms - started_ms : 0;

    if (elapsed < out_ms) {                       // still fading the old asset down
        s.stage     = FadeStage::out;
        s.old_alpha = 1.0f - static_cast<float>(elapsed) / static_cast<float>(out_ms);
        s.new_alpha = 0.0f;
        s.done      = false;
        return s;
    }

    s.old_alpha = 0.0f;                           // the outgoing asset is gone from here on
    if (!new_ready) {                             // holding at nothing until the load lands
        s.stage     = FadeStage::out;
        s.new_alpha = 0.0f;
        s.done      = false;
        return s;
    }

    const uint64_t outEnd  = started_ms + out_ms;
    const uint64_t inStart = ready_ms > outEnd ? ready_ms : outEnd;
    if (now_ms <= inStart) {                      // arrived exactly at the boundary
        s.stage     = FadeStage::in;
        s.new_alpha = 0.0f;
        s.done      = false;
        return s;
    }
    const uint64_t t = now_ms - inStart;
    if (in_ms == 0 || t >= in_ms) {
        s.stage     = FadeStage::idle;
        s.new_alpha = 1.0f;
        s.done      = true;
        return s;
    }
    s.stage     = FadeStage::in;
    s.new_alpha = static_cast<float>(t) / static_cast<float>(in_ms);
    s.done      = false;
    return s;
}

// --- LRU ---------------------------------------------------------------------------------------------

// Record `key` as most recently used and report what no longer fits, least recently used first. `order`
// is oldest-to-newest. Pure so the eviction order is tested without allocating a single image; the
// caller destroys whatever comes back.
//
// Capacity 0 is treated as "keep nothing": the key is still reported as evicted, so a caller that
// cannot cache still gets a consistent answer instead of an entry that is silently retained.
inline std::vector<std::string> lru_touch(std::vector<std::string>& order, const std::string& key,
                                          size_t capacity = kBackgroundCacheCapacity) {
    for (size_t i = 0; i < order.size(); i++)
        if (order[i] == key) { order.erase(order.begin() + static_cast<ptrdiff_t>(i)); break; }
    order.push_back(key);
    std::vector<std::string> evicted;
    while (order.size() > capacity) {
        evicted.push_back(order.front());
        order.erase(order.begin());
    }
    return evicted;
}

// --- settings ----------------------------------------------------------------------------------------

// Whether launcher music plays. Precedence matches the project's existing convention for host settings
// (see app_config.hpp): an explicit environment variable beats the persisted file, which beats the
// default. `env` is the raw PROSPER_LAUNCHER_MUSIC value, or nullptr when unset.
//
// Default on for a person browsing their library: the console-like presentation is the point of the
// feature, and it is one click to turn off and stays off once persisted.
//
// Default OFF for a measurement run, which is what `automated` selects. A screenshot capture or a
// frame-time sweep changes focus far faster than a person does, so the only thing it can produce is a
// burst of half-second fragments of thirty different themes — noise on the developer's desktop that
// demonstrates nothing. Silence is therefore the default there and sound is opt-in
// (PROSPER_LAUNCHER_MUSIC=1) for the one run that actually demonstrates the feature.
//
// Unit tests never reach this code at all: the test binary includes only this pure header, so no test
// can open an audio device however it is configured.
inline bool resolve_launcher_music(const char* env, bool from_config, bool automated = false) {
    if (env && *env) {
        // An explicit setting always wins, so a deliberate demonstration run can still have sound.
        const std::string v(env);
        if (v == "0" || v == "false" || v == "off" || v == "no") return false;
        if (v == "1" || v == "true" || v == "on" || v == "yes") return true;
    }
    if (automated) return false;
    return from_config;
}

// Playback gain for launcher music, as a linear multiplier.
//
// Not full scale: these are mastered game themes, and a launcher that opens at 100% is startling and
// louder than the desktop around it. 0.55 (about -5 dB) sits under a typical desktop mix while staying
// clearly audible. It is a presentation default, not a correctness claim. CONFIDENCE: MED — chosen by
// ear against several of the local tracks rather than derived from anything.
inline constexpr float kDefaultMusicGain = 0.55f;

// PROSPER_LAUNCHER_MUSIC_VOLUME overrides it, clamped to [0,1]. Anything unparseable leaves the default
// alone rather than silencing the music, so a typo is not mistaken for a missing asset.
inline float resolve_launcher_music_gain(const char* env, float fallback = kDefaultMusicGain) {
    if (!env || !*env) return fallback;
    char* end = nullptr;
    const double v = std::strtod(env, &end);
    if (end == env || !end) return fallback;
    while (*end == ' ' || *end == '\t') ++end;
    if (*end != '\0') return fallback;          // trailing junk: not a number we understand
    if (v != v) return fallback;                // NaN
    if (v < 0.0) return 0.0f;
    if (v > 1.0) return 1.0f;
    return static_cast<float>(v);
}

} // namespace prosper::frontend
