// test_ngs2_waveform — the NGS2 waveform surface: container parsing, block calculation,
// and the system-option reset.
//
// ParseWaveformData and CalcWaveformBlock were unregistered, so the dispatcher answered `0`:
// a parse that described nothing, a block nobody computed. Every TEST drives the real NIDs
// against hand-built container images (a minimal PCM WAV, an ATRAC9 RIFF, a VAGp blob), so a
// broken chunk walk misplaces data_offset and a wrong type ladder mistypes — both fail loudly
// instead of reading as music that never plays.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper;

static uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }
static constexpr uint64_t sx(uint32_t v) { return (uint64_t)(int64_t)(int32_t)v; }

static constexpr uint64_t kInvalidOut = sx(0x804a0053u);
static constexpr uint64_t kInvalidFormat = sx(0x804a0400u);
static constexpr uint64_t kUnknownFormat = sx(0x804a0401u);
static constexpr uint64_t kInvalidData = sx(0x804a0408u);

static uint32_t rd32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const uint8_t* p) {
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

TEST(Ngs2Waveform, AllNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        "sceNgs2ParseWaveformData", "sceNgs2CalcWaveformBlock", "sceNgs2SystemResetOption",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 3, "the 3 waveform exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
}

TEST(Ngs2Waveform, NidsResolveToStubValues) {
    EXPECT_EQ(nid_hash("sceNgs2ParseWaveformData"), "hyVLT2VlOYk");
    EXPECT_EQ(nid_hash("sceNgs2CalcWaveformBlock"), "3pCNbVM11UA");
    EXPECT_EQ(nid_hash("sceNgs2SystemResetOption"), "AQkj7C0f3PY");
    EXPECT_NE(nid_hash("sceNgs2ParseWaveformData"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

TEST(Ngs2Waveform, ResetOptionRestoresDefaults) {
    register_builtin_hle();
    HleFn reset = Hle::lookup(nid_hash("sceNgs2SystemResetOption"));
    ASSERT_NE(reset, nullptr);
    EXPECT_EQ(reset(0, 0, 0, 0, 0, 0), kInvalidOut) << "null option is refused";
    uint8_t option[0x80];
    std::memset(option, 0xAB, sizeof(option));
    ASSERT_EQ(reset(addr(option), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(rd32(option + 0x00), 0x78u) << "size echoes the struct";
    EXPECT_EQ(rd32(option + 0x6c), 512u) << "default max grain samples";
    EXPECT_EQ(rd32(option + 0x70), 256u) << "default grain samples";
    EXPECT_EQ(rd32(option + 0x74), 48000u) << "default sample rate";
    for (size_t i = 0x78; i < sizeof(option); ++i) {
        EXPECT_EQ(option[i], 0xABu) << "nothing written past the struct at byte " << i;
    }
}

// Minimal stereo S16 WAV: 8 frames. Data payload starts at file offset 44.
static std::vector<uint8_t> pcm_wav() {
    std::vector<uint8_t> f;
    auto put = [&](const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*)p;
        f.insert(f.end(), b, b + n);
    };
    auto put32 = [&](uint32_t v) {
        uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
        put(b, 4);
    };
    auto put16 = [&](uint32_t v) {
        uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)};
        put(b, 2);
    };
    put("RIFF", 4);
    put32(36 + 32);
    put("WAVE", 4);
    put("fmt ", 4);
    put32(16);
    put16(1);
    put16(2);
    put32(48000);
    put32(48000 * 4);
    put16(4);
    put16(16);
    put("data", 4);
    put32(32);
    for (int i = 0; i < 32; ++i) f.push_back((uint8_t)i);
    return f;
}

TEST(Ngs2Waveform, ParsePcmWave) {
    register_builtin_hle();
    HleFn parse = Hle::lookup(nid_hash("sceNgs2ParseWaveformData"));
    ASSERT_NE(parse, nullptr);
    const std::vector<uint8_t> wav = pcm_wav();
    uint8_t info[232];
    std::memset(info, 0xAB, sizeof(info));
    ASSERT_EQ(parse(addr(wav.data()), wav.size(), addr(info), 0, 0, 0), 0u);
    EXPECT_EQ(rd32(info + 0x00), 0x12u) << "S16 LE PCM type";
    EXPECT_EQ(rd32(info + 0x04), 2u) << "two channels";
    EXPECT_EQ(rd32(info + 0x08), 48000u) << "sample rate from the fmt chunk";
    EXPECT_EQ(rd32(info + 0x18), 44u) << "data payload file offset";
    EXPECT_EQ(rd32(info + 0x1c), 32u) << "data payload size";
    EXPECT_EQ(rd32(info + 0x28), 8u) << "8 stereo frames parsed";
    EXPECT_EQ(rd32(info + 0x2c), 4u) << "audio unit is one interleaved frame";
    EXPECT_EQ(rd32(info + 0x44), 1u) << "single block";
    EXPECT_EQ(rd64(info + 0x48), 44u) << "block mirrors the data offset";
    EXPECT_EQ(rd64(info + 0x50), 32u) << "block mirrors the data size";
    EXPECT_EQ(rd32(info + 0x60), 8u) << "block covers all samples";

    EXPECT_EQ(parse(addr(wav.data()), wav.size(), 0, 0, 0, 0), kInvalidOut)
        << "null info is refused";
    const uint8_t garbage[11] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT_EQ(parse(addr(garbage), sizeof(garbage), addr(info), 0, 0, 0), kUnknownFormat)
        << "a non-container is unknown, not parsed";
    uint8_t riff_only[12];
    std::memcpy(riff_only, "RIFF\x04\x00\x00\x00WAVE", 12);
    EXPECT_EQ(parse(addr(riff_only), sizeof(riff_only), addr(info), 0, 0, 0), kInvalidData)
        << "a container without format and data chunks is invalid data";
}

// Minimal ATRAC9 RIFF: extensible fmt with the Sony GUID, config FE 72 09 F0 (2ch/48kHz
// per the AJM config contract), a fact chunk declaring 900 samples, one superframe of data.
static std::vector<uint8_t> atrac9_file(uint32_t rate, size_t superframes) {
    static const uint8_t guid[16] = {0xd2, 0x42, 0xe1, 0x47, 0xba, 0x36, 0x8d,
                                     0x4d, 0x88, 0xfc, 0x61, 0x65, 0x4f, 0x8c,
                                     0x83, 0x6c};
    std::vector<uint8_t> f;
    auto put = [&](const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*)p;
        f.insert(f.end(), b, b + n);
    };
    auto put32 = [&](uint32_t v) {
        uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
        put(b, 4);
    };
    auto put16 = [&](uint32_t v) {
        uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)};
        put(b, 2);
    };
    put("RIFF", 4);
    const size_t size_at = f.size();
    put32(0);
    put("WAVE", 4);
    put("fmt ", 4);
    put32(52);
    put16(0xfffe);
    put16(2);
    put32(rate);
    put32(12000);
    put16(256);
    put16(0);
    put16(34);
    put16(256);
    put32(4);
    put(guid, 16);
    put32(1);
    const uint8_t cfg[4] = {0xfe, 0x72, 0x09, 0xf0};
    put(cfg, 4);
    put32(0);
    put("fact", 4);
    put32(4);
    put32(900);
    const size_t data_at = f.size();
    put("data", 4);
    put32(0);
    const size_t payload_at = f.size();
    for (size_t i = 0; i < superframes * 256; ++i) f.push_back((uint8_t)(i & 0xff));
    auto rewrite32 = [&](size_t at, uint32_t v) {
        f[at] = (uint8_t)v;
        f[at + 1] = (uint8_t)(v >> 8);
        f[at + 2] = (uint8_t)(v >> 16);
        f[at + 3] = (uint8_t)(v >> 24);
    };
    rewrite32(data_at + 4, (uint32_t)(f.size() - payload_at));
    rewrite32(size_at, (uint32_t)(f.size() - 8));
    return f;
}

TEST(Ngs2Waveform, ParseAtrac9Container) {
    register_builtin_hle();
    HleFn parse = Hle::lookup(nid_hash("sceNgs2ParseWaveformData"));
    ASSERT_NE(parse, nullptr);
    const std::vector<uint8_t> file = atrac9_file(48000, 1);
    uint8_t info[232]{};
    ASSERT_EQ(parse(addr(file.data()), file.size(), addr(info), 0, 0, 0), 0u);
    EXPECT_EQ(rd32(info + 0x00), 0x40u) << "ATRAC9 type from the GUID";
    EXPECT_EQ(rd32(info + 0x04), 2u) << "channels from the config, not the fmt header";
    EXPECT_EQ(rd32(info + 0x08), 48000u) << "rate from the config";
    EXPECT_EQ(rd32(info + 0x0c), 0xf00972feu) << "config word echoed";
    EXPECT_EQ(rd32(info + 0x28), 900u) << "sample count from the fact chunk";
    const uint32_t unit = rd32(info + 0x2c);
    const uint32_t unit_samples = rd32(info + 0x30);
    const uint32_t per_frame = rd32(info + 0x34);
    EXPECT_GT(unit, 0u) << "nonzero audio unit";
    EXPECT_GT(unit_samples, 0u) << "nonzero unit samples";
    EXPECT_GT(per_frame, 0u) << "nonzero units per frame";
    EXPECT_EQ(rd32(info + 0x38), unit * per_frame) << "frame size is unit x per-frame";
    EXPECT_EQ(rd32(info + 0x40), unit_samples) << "delay is one unit";
    EXPECT_EQ(rd32(info + 0x44), 1u) << "single block";
    EXPECT_EQ(rd64(info + 0x50), (uint64_t)(file.size() - rd32(info + 0x18)))
        << "block size covers the trailing data payload";
    EXPECT_EQ(rd32(info + 0x60), 900u) << "block spans the declared samples";

    const std::vector<uint8_t> wrong_rate = atrac9_file(44100, 1);
    // The container rate disagrees with the config; the config wins like the channels above.
    ASSERT_EQ(parse(addr(wrong_rate.data()), wrong_rate.size(), addr(info), 0, 0, 0), 0u);
    EXPECT_EQ(rd32(info + 0x08), 48000u) << "config rate wins over the fmt header";
}

TEST(Ngs2Waveform, ParseVagContainer) {
    register_builtin_hle();
    HleFn parse = Hle::lookup(nid_hash("sceNgs2ParseWaveformData"));
    ASSERT_NE(parse, nullptr);
    // VAGp: 0x30 big-endian header (size@0x0c, rate@0x10, channels@0x1e) + 16-byte units.
    std::vector<uint8_t> vag(0x30 + 32, 0);
    vag[0] = 'V';
    vag[1] = 'A';
    vag[2] = 'G';
    vag[3] = 'p';
    vag[0x0c + 3] = 32;
    vag[0x12] = 0xbb;
    vag[0x10 + 3] = 0x80;
    vag[0x1e] = 1;
    uint8_t info[232]{};
    ASSERT_EQ(parse(addr(vag.data()), vag.size(), addr(info), 0, 0, 0), 0u);
    EXPECT_EQ(rd32(info + 0x00), 0x1cu) << "VAG type from the magic";
    EXPECT_EQ(rd32(info + 0x04), 1u) << "channel count from the header";
    EXPECT_EQ(rd32(info + 0x08), 48000u) << "big-endian rate 0xbb80";
    EXPECT_EQ(rd32(info + 0x18), 0x30u) << "data starts past the header";
    EXPECT_EQ(rd32(info + 0x1c), 32u) << "two 16-byte units of data";
    EXPECT_EQ(rd32(info + 0x28), 56u) << "2 units x 28 samples";
    EXPECT_EQ(rd32(info + 0x2c), 16u) << "unit is one 16-byte frame";
    EXPECT_EQ(rd32(info + 0x30), 28u) << "28 samples per unit";
}

TEST(Ngs2Waveform, CalcBlockMapsLinearPcm) {
    register_builtin_hle();
    HleFn calc = Hle::lookup(nid_hash("sceNgs2CalcWaveformBlock"));
    ASSERT_NE(calc, nullptr);
    // Format: {type, channels, rate, config, frame_offset, frame_margin}.
    uint32_t format[6] = {0x12, 2, 48000, 0, 0, 0};
    uint8_t block[40];
    std::memset(block, 0xAB, sizeof(block));
    ASSERT_EQ(calc(addr(format), 10, 20, addr(block), 0, 0), 0u);
    EXPECT_EQ(rd64(block + 0x00), 40u) << "sample 10 of stereo S16 starts at byte 40";
    EXPECT_EQ(rd64(block + 0x08), 80u) << "20 frames span 80 bytes";
    EXPECT_EQ(rd32(block + 0x10), 0u) << "no repeats";
    EXPECT_EQ(rd32(block + 0x14), 0u) << "no skip";
    EXPECT_EQ(rd32(block + 0x18), 20u) << "sample count echoes";
    ASSERT_EQ(calc(addr(format), 10, 0, addr(block), 0, 0), 0u);
    EXPECT_EQ(rd64(block + 0x08), 0u) << "zero samples describe a zero block";

    EXPECT_EQ(calc(addr(format), 10, 20, 0, 0, 0), kInvalidOut) << "null block is refused";
    uint32_t atrac9[6] = {0x40, 2, 48000, 0, 0, 0};
    EXPECT_EQ(calc(addr(atrac9), 0, 100, addr(block), 0, 0), kInvalidFormat)
        << "non-PCM has no linear mapping (mixer boundary)";
    uint32_t no_channels[6] = {0x12, 0, 48000, 0, 0, 0};
    EXPECT_EQ(calc(addr(no_channels), 0, 100, addr(block), 0, 0), kInvalidFormat)
        << "zero channels fail instead of dividing by zero";
}
