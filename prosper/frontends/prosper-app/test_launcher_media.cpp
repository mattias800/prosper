// test_launcher_media — the library view's background/music parsing and focus policy (#1630). Pure:
// no SDL, no Vulkan, no real dump. Every container is built byte-by-byte here, so the rejection paths
// (truncation, wrong format, missing chunks) are exercised in ordinary CI, where the real 4K art and
// ATRAC9 tracks are not available and could not be committed anyway.
#include "launcher_media.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace prosper::frontend;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

// --- builders --------------------------------------------------------------------------------------

static void put32(std::string& s, size_t off, uint32_t v) {
    s[off + 0] = static_cast<char>(v & 0xFF);
    s[off + 1] = static_cast<char>((v >> 8) & 0xFF);
    s[off + 2] = static_cast<char>((v >> 16) & 0xFF);
    s[off + 3] = static_cast<char>((v >> 24) & 0xFF);
}
static void put16(std::string& s, size_t off, uint16_t v) {
    s[off + 0] = static_cast<char>(v & 0xFF);
    s[off + 1] = static_cast<char>((v >> 8) & 0xFF);
}

struct DdsOpts {
    uint32_t width = 3840, height = 2160;
    uint32_t dxgi = kDxgiBc7Unorm;
    uint32_t dimension = 3;         // TEXTURE2D
    uint32_t misc = 0;
    uint32_t array_size = 1;
    uint32_t header_size = 124;
    uint32_t pf_flags = 0x4;        // DDPF_FOURCC
    const char* magic = "DDS ";
    const char* fourcc = "DX10";
    bool full_payload = true;       // false -> truncate mip 0 by one byte
};

// A DDS whose header matches the shape every local pic1.dds has, with each field overridable so a
// single test can flip exactly one thing and prove the parser rejects on that field alone.
static std::string make_dds(const DdsOpts& o = DdsOpts{}) {
    std::string s(kDdsPixelOffset, '\0');
    for (int i = 0; i < 4; i++) s[i] = o.magic[i];
    put32(s, 4, o.header_size);
    put32(s, 12, o.height);
    put32(s, 16, o.width);
    put32(s, 80, o.pf_flags);
    for (int i = 0; i < 4; i++) s[84 + i] = o.fourcc[i];
    put32(s, 128, o.dxgi);
    put32(s, 132, o.dimension);
    put32(s, 136, o.misc);
    put32(s, 140, o.array_size);
    size_t payload = static_cast<size_t>(bc7_level_bytes(o.width, o.height));
    if (!o.full_payload && payload > 0) payload -= 1;
    s.append(payload, '\xAB');
    return s;
}

struct At9Opts {
    uint16_t tag = kWaveFormatExtensible;
    uint16_t channels = 2;
    uint32_t sample_rate = 48000;
    uint16_t block_align = 384;
    uint16_t samples_per_block = 1024;
    uint16_t cb_size = 34;
    uint8_t  config[4] = {0xFE, 0x74, 0x0B, 0xF0};
    bool     smpl_chunk = false;    // the fmt/fact/smpl/data ordering some dumps use
    uint32_t loop_start = 256;
    uint32_t loop_end = 4000;
    uint32_t loop_type = 0;         // 0 = forward
    uint32_t loop_count = 1;
    bool     with_data = true;
    bool     good_guid = true;
    uint32_t data_blocks = 4;
    uint32_t data_extra = 0;        // trailing bytes that do not complete a superframe
    const char* riff = "RIFF";
    const char* wave = "WAVE";
};

static void append_chunk(std::string& s, const char* id, const std::string& body) {
    s.append(id, 4);
    std::string sz(4, '\0');
    put32(sz, 0, static_cast<uint32_t>(body.size()));
    s += sz;
    s += body;
    if (body.size() & 1) s.push_back('\0');   // word alignment
}

// A RIFF/WAVE ATRAC9 file. The extension layout is the one the real files use:
// samplesPerBlock(2) channelMask(4) GUID(16) versionInfo(4) configData(4) reserved(4).
static std::string make_at9(const At9Opts& o = At9Opts{}) {
    std::string fmt(18 + 34, '\0');
    put16(fmt, 0, o.tag);
    put16(fmt, 2, o.channels);
    put32(fmt, 4, o.sample_rate);
    put32(fmt, 8, o.sample_rate * 2);          // nAvgBytesPerSec, unread
    put16(fmt, 12, o.block_align);
    put16(fmt, 14, 0);                          // wBitsPerSample, 0 for ATRAC9
    put16(fmt, 16, o.cb_size);
    put16(fmt, 18 + 0, o.samples_per_block);
    put32(fmt, 18 + 2, 0x3);                    // channel mask
    for (int i = 0; i < 16; i++)
        fmt[18 + 6 + i] = static_cast<char>(o.good_guid ? atrac9_subformat_guid()[i] : 0x11);
    put32(fmt, 18 + 22, 1);                     // versionInfo — deliberately NOT the config
    for (int i = 0; i < 4; i++) fmt[18 + 26 + i] = static_cast<char>(o.config[i]);

    std::string body;
    append_chunk(body, "fmt ", fmt);
    append_chunk(body, "fact", std::string(12, '\0'));
    if (o.smpl_chunk) {
        // 36-byte header + one 24-byte loop, the shape all 9 local files with an smpl chunk have.
        std::string smpl(60, '\0');
        put32(smpl, 28, o.loop_count);          // numSampleLoops
        put32(smpl, 36 + 4, o.loop_type);
        put32(smpl, 36 + 8, o.loop_start);
        put32(smpl, 36 + 12, o.loop_end);
        append_chunk(body, "smpl", smpl);
    }
    if (o.with_data)
        append_chunk(body, "data",
                     std::string(static_cast<size_t>(o.block_align) * o.data_blocks + o.data_extra, '\x5A'));

    std::string s;
    s.append(o.riff, 4);
    std::string sz(4, '\0');
    put32(sz, 0, static_cast<uint32_t>(4 + body.size()));
    s += sz;
    s.append(o.wave, 4);
    s += body;
    return s;
}

int main() {
    // --- DDS: the shape every local pic1.dds has --------------------------------------------------
    {
        const std::string dds = make_dds();
        const DdsBc7Image img = parse_dds_bc7(dds);
        CHECK(img.ok, "a 3840x2160 BC7_UNORM DX10 dds parses");
        CHECK(img.width == 3840 && img.height == 2160, "dimensions are read from the header");
        CHECK(img.data_offset == 148, "pixel data starts after the DDS and DX10 headers");
        CHECK(img.data_size == 3840u * 2160u, "BC7 mip 0 is one byte per texel");
        CHECK(!img.srgb, "dxgiFormat 98 is the linear BC7 format");
        CHECK(dds.size() == 8294548, "the synthetic file matches the real files' size exactly");
    }

    // --- DDS: every rejection is on its own field --------------------------------------------------
    {
        DdsOpts o; o.magic = "DDX ";
        CHECK(!parse_dds_bc7(make_dds(o)).ok, "a wrong magic is rejected");
    }
    {
        DdsOpts o; o.dxgi = 71;   // BC1_UNORM
        const DdsBc7Image img = parse_dds_bc7(make_dds(o));
        CHECK(!img.ok, "a non-BC7 dxgiFormat is rejected");
        CHECK(std::string(img.reason) == "not BC7_UNORM", "and says so");
    }
    {
        DdsOpts o; o.dxgi = kDxgiBc7UnormSrgb;
        const DdsBc7Image img = parse_dds_bc7(make_dds(o));
        CHECK(img.ok && img.srgb, "dxgiFormat 99 is accepted and flagged sRGB");
    }
    {
        DdsOpts o; o.full_payload = false;
        const DdsBc7Image img = parse_dds_bc7(make_dds(o));
        CHECK(!img.ok, "a file one byte short of a complete mip 0 is rejected");
        CHECK(std::string(img.reason) == "truncated: mip 0 is incomplete", "and says so");
    }
    {
        DdsOpts o; o.fourcc = "DXT5";
        CHECK(!parse_dds_bc7(make_dds(o)).ok, "a non-DX10 FourCC is rejected");
        DdsOpts p; p.pf_flags = 0x40;   // DDPF_RGB
        CHECK(!parse_dds_bc7(make_dds(p)).ok, "an uncompressed pixel format is rejected");
    }
    {
        DdsOpts o; o.header_size = 100;
        CHECK(!parse_dds_bc7(make_dds(o)).ok, "a wrong DDS header size is rejected");
        DdsOpts a; a.array_size = 6;
        CHECK(!parse_dds_bc7(make_dds(a)).ok, "a texture array is rejected");
        DdsOpts c; c.misc = 0x4;
        CHECK(!parse_dds_bc7(make_dds(c)).ok, "a cube map is rejected");
        DdsOpts d; d.dimension = 4;      // TEXTURE3D
        CHECK(!parse_dds_bc7(make_dds(d)).ok, "a 3D texture is rejected");
    }
    {
        DdsOpts z; z.width = 0;
        CHECK(!parse_dds_bc7(make_dds(z)).ok, "a zero width is rejected");
        // A header claiming an enormous image must be rejected on the dimension, before any size
        // arithmetic could overflow or any allocation could be attempted.
        // Assert the REASON, not merely that it failed. With kDdsMaxDimension removed this input is
        // still rejected by the later payload-size check, so `!ok` alone proves nothing about the
        // dimension guard that is supposed to run first.
        std::string huge = make_dds();
        put32(huge, 16, 0xFFFFFFFFu);
        const DdsBc7Image hugeImg = parse_dds_bc7(huge);
        CHECK(!hugeImg.ok, "an implausible width is rejected without overflowing");
        CHECK(std::string(hugeImg.reason) == "implausible dimensions",
              "and is rejected on the dimension itself, before any size arithmetic");
    }
    CHECK(!parse_dds_bc7("").ok, "an empty file is rejected");
    CHECK(!parse_dds_bc7(std::string(147, '\0')).ok, "a file shorter than the headers is rejected");
    CHECK(bc7_level_bytes(1, 1) == 16, "a 1x1 BC7 image still occupies one whole 4x4 block");
    CHECK(bc7_level_bytes(3840, 2160) == 3840ull * 2160ull, "BC7 is one byte per texel at 4K");

    // --- AT9: the shape the local snd0.at9 files have ----------------------------------------------
    {
        const At9Clip c = parse_at9_riff(make_at9());
        CHECK(c.ok, "a RIFF/WAVE ATRAC9 file parses");
        CHECK(c.channels == 2 && c.sample_rate == 48000, "stereo 48 kHz is read from fmt");
        CHECK(c.block_align == 384, "the superframe size is read from nBlockAlign");
        CHECK(c.samples_per_block == 1024, "samplesPerBlock is read from the extension");
        // The whole point of the offset-26 comment in the header: offset 22 holds versionInfo
        // (0x00000001) and would be silently accepted as a config word if it were read instead.
        CHECK(c.config[0] == 0xFE && c.config[1] == 0x74 && c.config[2] == 0x0B && c.config[3] == 0xF0,
              "the config word comes from extension offset 26, not versionInfo at 22");
    }
    {
        // Chunk order varies between dumps: some carry an smpl chunk before data, some do not. A
        // hardcoded data offset would work on one and read the wrong bytes on the other.
        const At9Clip without = parse_at9_riff(make_at9());
        At9Opts o; o.smpl_chunk = true;
        const At9Clip with = parse_at9_riff(make_at9(o));
        CHECK(with.ok && without.ok, "both chunk orderings parse");
        CHECK(with.data_offset == without.data_offset + 68,
              "the data offset follows the walk, shifted by the whole smpl chunk");
        CHECK(with.data_size == without.data_size, "and the payload is identified identically");
    }
    {
        // nBlockAlign is 320, 384 or 512 across the local files, tracking the config word. Nothing may
        // assume one particular value.
        for (uint16_t ba : {uint16_t(320), uint16_t(384), uint16_t(512)}) {
            At9Opts o; o.block_align = ba;
            const At9Clip c = parse_at9_riff(make_at9(o));
            CHECK(c.ok && c.block_align == ba, "a non-512 block align parses");
        }
    }
    {
        At9Opts o; o.data_extra = 7;    // a partial superframe at the end
        const At9Clip c = parse_at9_riff(make_at9(o));
        CHECK(c.ok, "a trailing partial superframe does not reject the file");
        CHECK(c.data_size == 384u * 4, "it is truncated to whole superframes, never over-read");
    }
    {
        At9Opts o; o.with_data = false;
        const At9Clip c = parse_at9_riff(make_at9(o));
        CHECK(!c.ok && std::string(c.reason) == "no data chunk", "a file with no data chunk is rejected");
    }
    {
        At9Opts o; o.good_guid = false;
        CHECK(!parse_at9_riff(make_at9(o)).ok, "a non-ATRAC9 SubFormat GUID is rejected");
        At9Opts t; t.tag = 0x0001;       // PCM
        CHECK(!parse_at9_riff(make_at9(t)).ok, "a non-extensible format tag is rejected");
        At9Opts s; s.cb_size = 2;
        CHECK(!parse_at9_riff(make_at9(s)).ok, "a fmt extension too small for ATRAC9 is rejected");
        At9Opts b; b.block_align = 0;
        CHECK(!parse_at9_riff(make_at9(b)).ok, "a zero block align is rejected");
        At9Opts r; r.riff = "RIFX";
        CHECK(!parse_at9_riff(make_at9(r)).ok, "a non-RIFF file is rejected");
        At9Opts w; w.wave = "AVI ";
        CHECK(!parse_at9_riff(make_at9(w)).ok, "a RIFF that is not WAVE is rejected");
    }
    {
        CHECK(!parse_at9_riff("").ok, "an empty file is rejected");
        // A chunk size that runs past the end must stop the walk rather than wrap an offset.
        std::string bad = make_at9();
        put32(bad, 16, 0xFFFFFFF0u);
        CHECK(!parse_at9_riff(bad).ok, "a chunk size past the end of the file is rejected");
        // A truncated file: the data chunk header says more than the file holds.
        std::string cut = make_at9();
        cut.resize(cut.size() - 100);
        CHECK(!parse_at9_riff(cut).ok, "a truncated data chunk is rejected");
        // The case that actually exercises the chunk-bounds guard: a file cut off INSIDE the fmt chunk,
        // where the declared size is honest but the bytes are absent. Without `body + size > end` the
        // extension reads at ext+6 and ext+26 would run past the buffer. The cases above fail earlier,
        // on a missing or short data chunk, so neither demonstrates this one.
        //
        // The REASON is what discriminates, and `!ok` alone does not: without the guard these are still
        // rejected, but by later checks reading memory past the buffer — at 20 the format tag is read
        // from beyond the end and fails as "not WAVE_FORMAT_EXTENSIBLE", at 40 and 60 the GUID check
        // fails on whatever bytes happened to follow. Only the guard makes all three stop at the walk,
        // and there is no sanitizer in CI to catch the over-read otherwise.
        {
            const std::string full = make_at9();
            for (size_t at : {size_t(20), size_t(40), size_t(60)}) {
                const At9Clip c = parse_at9_riff(full.substr(0, at));
                CHECK(!c.ok, "a file truncated inside the fmt chunk is rejected");
                CHECK(std::string(c.reason) == "no fmt chunk",
                      "and is stopped by the chunk-bounds walk, never read past the buffer");
            }
        }
    }

    // --- AT9: smpl loop points ----------------------------------------------------------------------
    {
        const At9Clip none = parse_at9_riff(make_at9());
        CHECK(!none.loop.ok, "a track with no smpl chunk declares no loop");
        CHECK(none.total_frames() == 4u * 1024, "the frame count comes from whole superframes");
        At9Opts o; o.smpl_chunk = true;                 // 4 blocks * 1024 = 4096 frames
        const At9Clip c = parse_at9_riff(make_at9(o));
        CHECK(c.loop.ok, "a forward smpl loop is honoured");
        CHECK(c.loop.start == 256 && c.loop.end == 4000, "with the declared start and end");
        // Playback walks forward and jumps back to loop_start after the inclusive end, so the intro
        // before loop_start plays once and the tail after loop_end is never heard.
        CHECK(at9_next_frame(c, 3999) == 4000, "the loop end itself is played");
        CHECK(at9_next_frame(c, 4000) == 256, "and playback then returns to the loop start");
        CHECK(at9_next_frame(c, 100) == 101, "the intro before the loop plays normally the first time");
    }
    {
        // Whole-file looping is the fallback: a seam, but the music keeps going.
        const At9Clip c = parse_at9_riff(make_at9());
        CHECK(at9_next_frame(c, 4094) == 4095, "without a loop, playback runs to the end");
        CHECK(at9_next_frame(c, 4095) == 0, "and wraps to the beginning");
    }
    {
        // A loop that does not fit the decoded audio is dropped rather than clamped: clamping would
        // invent an edit, whereas whole-file looping is a behaviour we can describe.
        At9Opts past; past.smpl_chunk = true; past.loop_end = 999999;
        CHECK(!parse_at9_riff(make_at9(past)).loop.ok, "a loop ending past the audio is ignored");
        At9Opts inv; inv.smpl_chunk = true; inv.loop_start = 3000; inv.loop_end = 1000;
        CHECK(!parse_at9_riff(make_at9(inv)).loop.ok, "an inverted loop is ignored");
        At9Opts eq; eq.smpl_chunk = true; eq.loop_start = 1000; eq.loop_end = 1000;
        CHECK(!parse_at9_riff(make_at9(eq)).loop.ok, "a zero-length loop is ignored");
        At9Opts none; none.smpl_chunk = true; none.loop_count = 0;
        CHECK(!parse_at9_riff(make_at9(none)).loop.ok, "an smpl chunk declaring no loops is ignored");
        At9Opts back; back.smpl_chunk = true; back.loop_type = 1;   // alternating
        CHECK(!parse_at9_riff(make_at9(back)).loop.ok, "a non-forward loop type is ignored");
        // Dropping the loop must never drop the track.
        CHECK(parse_at9_riff(make_at9(past)).ok, "a bad loop still leaves a playable clip");
    }

    // --- asset locations ---------------------------------------------------------------------------
    {
        const auto with_bc = background_candidates("/games/PPSA24651-app0", true);
        CHECK(with_bc.size() == 2, "with BC support there is a preferred and a fallback background");
        CHECK(with_bc[0].path == "/games/PPSA24651-app0/sce_sys/pic1.dds" && with_bc[0].block_compressed,
              "pic1.dds is preferred: it is the clean plate, without the logo pic0 composites in");
        CHECK(with_bc[1].path == "/games/PPSA24651-app0/sce_sys/pic0.png" && !with_bc[1].block_compressed,
              "pic0.png is the non-BC fallback");
        const auto without_bc = background_candidates("/games/PPSA24651-app0", false);
        CHECK(without_bc.size() == 1 && !without_bc[0].block_compressed,
              "without textureCompressionBC the dds is not even attempted");
        CHECK(background_candidates("", true).empty(), "no root means no candidates");
        CHECK(music_path_for("/games/PPSA24651-app0") == "/games/PPSA24651-app0/sce_sys/snd0.at9",
              "the music path is sce_sys/snd0.at9");
        CHECK(music_path_for("").empty(), "no root means no music path");
    }

    // --- debounce ----------------------------------------------------------------------------------
    {
        CHECK(!focus_load_due(1000, 1000), "a load does not start the instant focus moves");
        CHECK(!focus_load_due(1000, 1000 + kFocusDebounceMs - 1), "nor just before the debounce expires");
        CHECK(focus_load_due(1000, 1000 + kFocusDebounceMs), "it starts once focus has rested");
        CHECK(focus_load_due(1000, 5000), "and stays due afterwards");
        // Holding an arrow key steps the grid far faster than the debounce, so a sweep must never
        // reach the load point until the user stops.
        bool startedDuringSweep = false;
        for (uint64_t t = 0; t < 900; t += 30)      // a keypress every 30 ms
            if (focus_load_due(t, t + 29)) startedDuringSweep = true;
        CHECK(!startedDuringSweep, "sweeping the grid at 30 ms per step starts no load at all");
    }

    // --- the two-stage fade ------------------------------------------------------------------------
    {
        const uint64_t t0 = 10000;
        // Fading the old asset out, with the new one not yet loaded.
        const FadeState a = fade_state_at(t0, t0, false, 0);
        CHECK(a.stage == FadeStage::out && a.old_alpha == 1.0f && a.new_alpha == 0.0f,
              "the transition opens with the outgoing asset at full and nothing incoming");
        const FadeState b = fade_state_at(t0, t0 + kFadeOutMs / 2, false, 0);
        CHECK(b.stage == FadeStage::out && b.old_alpha > 0.4f && b.old_alpha < 0.6f,
              "the outgoing asset is half faded at the midpoint");
        CHECK(b.new_alpha == 0.0f, "and the incoming asset is still not showing");

        // The load has not landed by the time the fade-out ends: hold at nothing rather than
        // fading up an asset that does not exist yet.
        const FadeState held = fade_state_at(t0, t0 + kFadeOutMs + 500, false, 0);
        CHECK(held.old_alpha == 0.0f && held.new_alpha == 0.0f && !held.done,
              "a slow load holds the background empty instead of stalling on the old art");

        // It lands late; the fade-in is anchored to the arrival, so it plays in full.
        const uint64_t ready = t0 + kFadeOutMs + 500;
        const FadeState atReady = fade_state_at(t0, ready, true, ready);
        CHECK(atReady.new_alpha == 0.0f, "the fade-in starts from zero at the moment of arrival");
        const FadeState mid = fade_state_at(t0, ready + kFadeInMs / 2, true, ready);
        CHECK(mid.stage == FadeStage::in && mid.new_alpha > 0.4f && mid.new_alpha < 0.6f,
              "and is half done a half-duration later, not skipped because the load was late");
        const FadeState end = fade_state_at(t0, ready + kFadeInMs, true, ready);
        CHECK(end.stage == FadeStage::idle && end.new_alpha == 1.0f && end.done,
              "the transition finishes with the incoming asset fully up");
        CHECK(fade_state_at(t0, ready + 100000, true, ready).done, "and stays finished");
    }
    {
        // THE invariant that makes this a sequential fade rather than a cross-fade: at no instant are
        // two backgrounds blended or two tracks audible. Swept at 1 ms over a whole transition, for a
        // load that arrives immediately and one that arrives late.
        const uint64_t t0 = 500;
        bool overlapped = false, sawOldVisible = false, sawNewVisible = false;
        for (uint64_t ready : {t0, t0 + kFadeOutMs + 300}) {
            for (uint64_t t = t0; t <= ready + kFadeInMs + 50; t++) {
                const FadeState s = fade_state_at(t0, t, t >= ready, ready);
                if (s.old_alpha > 0.0f && s.new_alpha > 0.0f) overlapped = true;
                if (s.old_alpha > 0.0f) sawOldVisible = true;
                if (s.new_alpha > 0.0f) sawNewVisible = true;
                if (s.old_alpha < 0.0f || s.old_alpha > 1.0f) overlapped = true;
                if (s.new_alpha < 0.0f || s.new_alpha > 1.0f) overlapped = true;
            }
        }
        CHECK(!overlapped, "the outgoing and incoming assets are never both visible, at any millisecond");
        CHECK(sawOldVisible && sawNewVisible, "and the sweep really did cover both halves of the fade");
    }
    {
        // A title with no snd0.at9 still fades the previous track out: the transition is driven by the
        // focus change, not by the existence of an incoming asset. With nothing to fade in, "ready"
        // is immediate and new_alpha stays 0 because the caller has no track to apply it to.
        const uint64_t t0 = 0;
        const FadeState s = fade_state_at(t0, kFadeOutMs, true, t0);
        CHECK(s.old_alpha == 0.0f, "moving to a title with no music still silences the previous track");
    }

    // --- LRU ---------------------------------------------------------------------------------------
    {
        std::vector<std::string> order;
        CHECK(lru_touch(order, "a").empty(), "the first three titles fit");
        CHECK(lru_touch(order, "b").empty(), "...");
        CHECK(lru_touch(order, "c").empty(), "...");
        CHECK(order.size() == 3, "three backgrounds stay resident");
        const auto ev = lru_touch(order, "d");
        CHECK(ev.size() == 1 && ev[0] == "a", "the fourth evicts the least recently focused");
        CHECK(order.size() == 3, "and the cache stays bounded");
        // Re-focusing a resident title must renew it, not duplicate it.
        CHECK(lru_touch(order, "b").empty(), "re-focusing a cached title evicts nothing");
        CHECK(order.size() == 3 && order.back() == "b", "and moves it to most-recently-used");
        const auto ev2 = lru_touch(order, "e");
        CHECK(ev2.size() == 1 && ev2[0] == "c", "so the next eviction is the one not re-focused");
        // Stepping back and forth between two titles must never evict either of them.
        std::vector<std::string> pair;
        bool evictedInPair = false;
        for (int i = 0; i < 20; i++)
            if (!lru_touch(pair, (i % 2) ? "x" : "y").empty()) evictedInPair = true;
        CHECK(!evictedInPair, "stepping between two titles reloads neither");
        // 30 titles must never leave more than the capacity resident — the whole memory argument.
        std::vector<std::string> many;
        size_t peak = 0;
        for (int i = 0; i < 30; i++) {
            lru_touch(many, "PPSA" + std::to_string(i));
            if (many.size() > peak) peak = many.size();
        }
        CHECK(peak == kBackgroundCacheCapacity, "walking all 30 titles never exceeds the capacity");
    }
    {
        std::vector<std::string> order;
        const auto ev = lru_touch(order, "a", 0);
        CHECK(ev.size() == 1 && ev[0] == "a" && order.empty(),
              "a zero capacity reports the entry as evicted rather than silently keeping it");
    }

    // --- the music switch ---------------------------------------------------------------------------
    {
        CHECK(resolve_launcher_music(nullptr, true), "with nothing set, the persisted setting applies");
        CHECK(!resolve_launcher_music(nullptr, false), "including when it is off");
        CHECK(!resolve_launcher_music("0", true), "the environment can turn music off");
        CHECK(resolve_launcher_music("1", false), "and back on, over a persisted off");
        CHECK(!resolve_launcher_music("false", true) && !resolve_launcher_music("off", true) &&
              !resolve_launcher_music("no", true), "the usual false spellings are accepted");
        CHECK(resolve_launcher_music("true", false) && resolve_launcher_music("on", false) &&
              resolve_launcher_music("yes", false), "and the usual true ones");
        CHECK(resolve_launcher_music("", true), "an empty value is not a setting; the file still applies");
    }
    {
        // An automated run — a screenshot capture, a timing sweep, a scripted route — must not play
        // audio on the developer's desktop. Silence is the DEFAULT there and sound is opt-in, so a
        // harness cannot make noise by forgetting to mute; it can only make noise by asking.
        CHECK(!resolve_launcher_music(nullptr, true, /*automated=*/true),
              "an automated run is silent even when the persisted setting says music is on");
        CHECK(!resolve_launcher_music("", true, true), "and an empty value does not re-enable it");
        CHECK(resolve_launcher_music("1", false, true),
              "an explicit PROSPER_LAUNCHER_MUSIC=1 still enables sound for a demonstration run");
        CHECK(!resolve_launcher_music("0", true, true), "and an explicit off stays off");
        CHECK(resolve_launcher_music(nullptr, true, /*automated=*/false),
              "while an ordinary interactive run still plays music");
    }
    {
        CHECK(resolve_launcher_music_gain(nullptr) == kDefaultMusicGain,
              "launcher music plays below full scale by default");
        CHECK(kDefaultMusicGain > 0.0f && kDefaultMusicGain < 1.0f, "and that default really is reduced");
        CHECK(resolve_launcher_music_gain("0.25") == 0.25f, "the volume can be set explicitly");
        CHECK(resolve_launcher_music_gain("0") == 0.0f, "including to silence");
        CHECK(resolve_launcher_music_gain("5") == 1.0f, "an over-range value clamps rather than clipping");
        CHECK(resolve_launcher_music_gain("-1") == 0.0f, "and a negative one clamps to silence");
        // A typo must not be mistaken for "off" — that would look exactly like a missing asset.
        CHECK(resolve_launcher_music_gain("loud") == kDefaultMusicGain, "unparseable input keeps the default");
        CHECK(resolve_launcher_music_gain("0.5x") == kDefaultMusicGain, "so does trailing junk");
        CHECK(resolve_launcher_music_gain("") == kDefaultMusicGain, "so does an empty value");
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
