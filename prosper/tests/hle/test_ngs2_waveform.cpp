// test_ngs2_waveform — sceNgs2ParseWaveformData / sceNgs2CalcWaveformBlock /
// sceNgs2SystemResetOption against libSceNgs2.native.sprx, the module PS5 titles bind.
//
// Every expected value here is derived from that module's disassembly (offsets are cited in
// src/hle/audio/ngs2_waveform.cpp), not from a secondary implementation. Each TEST drives the real
// NIDs through the dispatcher against container images built in-test (WAV, extensible WAV, ATRAC9
// RIFF, VAGp), so a wrong chunk walk, type ladder, loop rule or block mapping fails loudly.
#include "hle/audio/ngs2_waveform.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper;

namespace {

uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }
constexpr uint64_t sx(uint32_t v) { return (uint64_t)(int64_t)(int32_t)v; }

constexpr uint64_t kNullOut = sx(0x804a8010u);
constexpr uint64_t kInvalidData = sx(0x804a8430u);
constexpr uint64_t kNullFormat = sx(0x804a8431u);
constexpr uint64_t kUnsupported = sx(0x804a8432u);

uint32_t rd32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
uint64_t rd64(const uint8_t* p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }

HleFn fn(const char* name) {
    register_builtin_hle();
    HleFn f = Hle::lookup(nid_hash(name));
    EXPECT_NE(f, nullptr) << name;
    return f;
}

struct Bytes {
    std::vector<uint8_t> v;
    void raw(const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*)p;
        v.insert(v.end(), b, b + n);
    }
    void u8(uint32_t x) { v.push_back((uint8_t)x); }
    void u16(uint32_t x) { u8(x); u8(x >> 8); }
    void u32(uint32_t x) { u16(x); u16(x >> 16); }
    void be32(uint32_t x) { u8(x >> 24); u8(x >> 16); u8(x >> 8); u8(x); }
    void zeros(size_t n) { v.insert(v.end(), n, 0); }
};

// RIFF image from (fourcc, body) chunks, each padded to even length, RIFF size = file - 8.
std::vector<uint8_t> riff(const std::vector<std::pair<std::string, std::vector<uint8_t>>>& chunks,
                          const char* form = "WAVE") {
    Bytes f;
    f.raw("RIFF", 4);
    f.u32(0);
    f.raw(form, 4);
    for (const auto& [id, body] : chunks) {
        f.raw(id.data(), 4);
        f.u32((uint32_t)body.size());
        f.raw(body.data(), body.size());
        if (body.size() & 1) f.u8(0);
    }
    const uint32_t size = (uint32_t)f.v.size() - 8;
    std::memcpy(f.v.data() + 4, &size, 4);
    return f.v;
}

std::vector<uint8_t> fmt_pcm(uint32_t tag, uint32_t ch, uint32_t rate, uint32_t bits) {
    Bytes b;
    b.u16(tag);
    b.u16(ch);
    b.u32(rate);
    b.u32(rate * ch * bits / 8);
    b.u16(ch * bits / 8);
    b.u16(bits);
    return b.v;
}

constexpr uint8_t kGuidPcm[16] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
                                  0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71};
constexpr uint8_t kGuidAtrac9[16] = {0xd2, 0x42, 0xe1, 0x47, 0xba, 0x36, 0x8d, 0x4d,
                                     0x88, 0xfc, 0x61, 0x65, 0x4f, 0x8c, 0x83, 0x6c};
constexpr uint8_t kGuidExt43[16] = {0x1c, 0xe0, 0x10, 0x9e, 0xcc, 0x8b, 0xd9, 0x47,
                                    0x8f, 0x3b, 0xe3, 0x0b, 0x8b, 0xdc, 0x35, 0x29};

// WAVE_FORMAT_EXTENSIBLE fmt body: 0x28 bytes, or 0x34 with an ATRAC9 config word at +0x2c.
std::vector<uint8_t> fmt_ext(uint32_t ch, uint32_t rate, uint32_t bits, const uint8_t* guid,
                             const uint8_t* at9_config = nullptr) {
    Bytes b;
    b.u16(0xfffe);
    b.u16(ch);
    b.u32(rate);
    b.u32(rate * ch * bits / 8);
    b.u16(ch * bits / 8);
    b.u16(bits);
    b.u16(at9_config ? 34 : 22);
    b.u16(bits);
    b.u32(3);
    b.raw(guid, 16);
    if (at9_config) {
        b.u32(1);
        b.raw(at9_config, 4);
        b.u32(0);
    }
    return b.v;
}

std::vector<uint8_t> u32s(std::initializer_list<uint32_t> xs) {
    Bytes b;
    for (uint32_t x : xs) b.u32(x);
    return b.v;
}

// Sampler chunk with one loop [start, end] (inclusive end, as the WAV smpl chunk stores it).
std::vector<uint8_t> smpl(uint32_t loops, uint32_t start, uint32_t end) {
    Bytes b;
    b.zeros(0x1c);
    b.u32(loops);
    b.u32(0);
    b.zeros(8);   // cue id, type
    b.u32(start);
    b.u32(end);
    b.u32(0);
    b.u32(0);
    return b.v;   // 0x3c bytes
}

std::vector<uint8_t> payload(size_t n) {
    std::vector<uint8_t> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)i;
    return p;
}

struct Info {
    uint8_t b[0xe8];
    Info() { std::memset(b, 0xAB, sizeof b); }
    uint32_t u(size_t off) const { return rd32(b + off); }
    uint64_t blk_off(int i) const { return rd64(b + 0x48 + 0x28 * i); }
    uint64_t blk_size(int i) const { return rd64(b + 0x48 + 0x28 * i + 8); }
    uint32_t blk_rep(int i) const { return rd32(b + 0x48 + 0x28 * i + 0x10); }
    uint32_t blk_skip(int i) const { return rd32(b + 0x48 + 0x28 * i + 0x14); }
    uint32_t blk_num(int i) const { return rd32(b + 0x48 + 0x28 * i + 0x18); }
};

uint64_t parse(const std::vector<uint8_t>& f, Info& info) {
    return fn("sceNgs2ParseWaveformData")(addr(f.data()), f.size(), addr(info.b), 0, 0, 0);
}

}   // namespace

TEST(Ngs2Waveform, AllNidsBound) {
    register_builtin_hle();
    for (const char* name : {"sceNgs2ParseWaveformData", "sceNgs2CalcWaveformBlock",
                             "sceNgs2SystemResetOption"}) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
    EXPECT_EQ(nid_hash("sceNgs2ParseWaveformData"), "hyVLT2VlOYk");
    EXPECT_EQ(nid_hash("sceNgs2CalcWaveformBlock"), "3pCNbVM11UA");
    EXPECT_EQ(nid_hash("sceNgs2SystemResetOption"), "AQkj7C0f3PY");
}

// Native 0x21090: 0x90 bytes from the rodata template; size 0x90, 512/256/48000/8, rest zero.
TEST(Ngs2Waveform, ResetOptionWritesTheNativeTemplate) {
    HleFn reset = fn("sceNgs2SystemResetOption");
    EXPECT_EQ(reset(0, 0, 0, 0, 0, 0), kNullOut) << "NULL option";
    uint8_t option[0xa0];
    std::memset(option, 0xAB, sizeof(option));
    ASSERT_EQ(reset(addr(option), 0, 0, 0, 0, 0), 0u);
    uint8_t expected[0x90]{};
    const uint32_t fields[][2] = {{0x00, 0x90}, {0x6c, 512}, {0x70, 256}, {0x74, 48000}, {0x78, 8}};
    for (const auto& f : fields) std::memcpy(expected + f[0], &f[1], 4);
    for (size_t i = 0; i < sizeof(expected); ++i)
        EXPECT_EQ(option[i], expected[i]) << "template byte 0x" << std::hex << i;
    for (size_t i = 0x90; i < sizeof(option); ++i) EXPECT_EQ(option[i], 0xABu) << "past the struct";
}

// Error family and pre-zeroing (native 0x10ef7..0x11169, 0x11037/0x11053).
TEST(Ngs2Waveform, ParseErrorsUseTheNativeFamily) {
    HleFn p = fn("sceNgs2ParseWaveformData");
    const auto wav = riff({{"fmt ", fmt_pcm(1, 2, 48000, 16)}, {"data", payload(32)}});
    Info info;
    EXPECT_EQ(p(addr(wav.data()), wav.size(), 0, 0, 0, 0), kNullOut) << "NULL info is checked first";
    EXPECT_EQ(p(0, 0, addr(info.b), 0, 0, 0), kInvalidData) << "NULL data";
    EXPECT_EQ(p(addr(wav.data()), 3, addr(info.b), 0, 0, 0), kInvalidData) << "size < 4";

    const uint8_t junk[16] = {'R', 'I', 'F', 'X', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Info pre;
    EXPECT_EQ(p(addr(junk), sizeof junk, addr(pre.b), 0, 0, 0), kInvalidData)
        << "a magic that is neither VAGp nor RIFF";
    for (int i = 0; i < 8; ++i) EXPECT_EQ(pre.b[i], 0u) << "info[0..8) zeroed even on failure";
    EXPECT_EQ(pre.b[8], 0xABu) << "only 8 bytes are pre-zeroed";

    auto missing_fmt = riff({{"data", payload(32)}});
    EXPECT_EQ(parse(missing_fmt, info), kInvalidData) << "no fmt chunk";
    auto missing_data = riff({{"fmt ", fmt_pcm(1, 2, 48000, 16)}});
    EXPECT_EQ(parse(missing_data, info), kInvalidData) << "no data chunk";
    auto adpcm = riff({{"fmt ", fmt_pcm(2, 2, 48000, 4)}, {"data", payload(32)}});
    EXPECT_EQ(parse(adpcm, info), kUnsupported) << "fmt tag 2";
    auto twelve = riff({{"fmt ", fmt_pcm(1, 2, 48000, 12)}, {"data", payload(32)}});
    EXPECT_EQ(parse(twelve, info), kUnsupported) << "12-bit PCM";
    auto ext_short = riff({{"fmt ", fmt_pcm(0xfffe, 2, 48000, 16)}, {"data", payload(32)}});
    EXPECT_EQ(parse(ext_short, info), kUnsupported) << "extensible fmt shorter than 0x28";
}

TEST(Ngs2Waveform, ParsePcmWave) {
    const auto wav = riff({{"fmt ", fmt_pcm(1, 2, 44100, 16)}, {"data", payload(32)}});
    Info info;
    ASSERT_EQ(parse(wav, info), 0u);
    EXPECT_EQ(info.u(0x00), 0x12u) << "S16";
    EXPECT_EQ(info.u(0x04), 2u);
    EXPECT_EQ(info.u(0x08), 44100u);
    EXPECT_EQ(info.u(0x0c), 0u) << "no config for PCM";
    EXPECT_EQ(info.u(0x10), 0u);
    EXPECT_EQ(info.u(0x14), 0u);
    EXPECT_EQ(info.u(0x18), 44u) << "data body offset";
    EXPECT_EQ(info.u(0x1c), 32u);
    EXPECT_EQ(info.u(0x20), 0u) << "no smpl: loop begin 0";
    EXPECT_EQ(info.u(0x24), 0u) << "no smpl: loop end 0, not samples-1";
    EXPECT_EQ(info.u(0x28), 8u);
    EXPECT_EQ(info.u(0x2c), 4u) << "unit = ch * bytes per sample";
    EXPECT_EQ(info.u(0x30), 1u);
    EXPECT_EQ(info.u(0x34), 1u);
    EXPECT_EQ(info.u(0x38), 4u);
    EXPECT_EQ(info.u(0x3c), 1u);
    EXPECT_EQ(info.u(0x40), 0u);
    EXPECT_EQ(info.u(0x44), 1u);
    EXPECT_EQ(info.blk_off(0), 44u);
    EXPECT_EQ(info.blk_size(0), 32u);
    EXPECT_EQ(info.blk_rep(0), 0u);
    EXPECT_EQ(info.blk_num(0), 8u);
    EXPECT_EQ(info.b[0x48 + 0x28], 0xABu) << "blocks past the count are not written";
}

// "WAVE" is never checked, and a fact chunk supplies the PCM sample count when present.
TEST(Ngs2Waveform, ParsePcmIgnoresFormTypeAndHonoursFact) {
    const auto wav = riff({{"fmt ", fmt_pcm(1, 1, 48000, 16)}, {"fact", u32s({5})},
                           {"data", payload(32)}},
                          "JUNK");
    Info info;
    ASSERT_EQ(parse(wav, info), 0u);
    EXPECT_EQ(info.u(0x28), 5u) << "fact samples win over data/unit";
    EXPECT_EQ(info.blk_num(0), 5u);
    EXPECT_EQ(info.blk_size(0), 10u);
}

TEST(Ngs2Waveform, ParsePcmTypeLadder) {
    struct Case { uint32_t tag, bits, type, unit; } cases[] = {
        {1, 8, 0x11, 3}, {1, 16, 0x12, 6}, {1, 24, 0x14, 9}, {1, 32, 0x16, 12},
        {3, 32, 0x18, 12}, {3, 64, 0x1a, 24},
    };
    for (const Case& c : cases) {
        const auto wav = riff({{"fmt ", fmt_pcm(c.tag, 3, 48000, c.bits)}, {"data", payload(48)}});
        Info info;
        ASSERT_EQ(parse(wav, info), 0u) << c.tag << "/" << c.bits;
        EXPECT_EQ(info.u(0x00), c.type) << c.tag << "/" << c.bits;
        EXPECT_EQ(info.u(0x2c), c.unit) << c.tag << "/" << c.bits;
        EXPECT_EQ(info.u(0x28), 48u / c.unit) << c.tag << "/" << c.bits;
    }
    const auto f16 = riff({{"fmt ", fmt_pcm(3, 2, 48000, 16)}, {"data", payload(32)}});
    Info info;
    EXPECT_EQ(parse(f16, info), kUnsupported) << "16-bit float";
}

// Loop rule (0x116cd..0x11729): begin = smpl start, end = smpl end + 1; 3 blocks with the loop
// block repeating (0x11a7d..0x11cdd).
TEST(Ngs2Waveform, ParsePcmSmplLoopBuildsThreeBlocks) {
    const auto wav = riff({{"fmt ", fmt_pcm(1, 2, 48000, 16)}, {"smpl", smpl(1, 2, 5)},
                           {"data", payload(32)}});
    Info info;
    ASSERT_EQ(parse(wav, info), 0u);
    const uint32_t data = info.u(0x18);
    EXPECT_EQ(info.u(0x20), 2u);
    EXPECT_EQ(info.u(0x24), 6u) << "exclusive end = smpl end + 1";
    ASSERT_EQ(info.u(0x44), 3u);
    EXPECT_EQ(info.blk_off(0), data + 0u);
    EXPECT_EQ(info.blk_size(0), 8u);
    EXPECT_EQ(info.blk_num(0), 2u);
    EXPECT_EQ(info.blk_rep(0), 0u);
    EXPECT_EQ(info.blk_off(1), data + 8u);
    EXPECT_EQ(info.blk_size(1), 16u);
    EXPECT_EQ(info.blk_num(1), 4u);
    EXPECT_EQ(info.blk_rep(1), 0xffffffffu) << "the loop block repeats";
    EXPECT_EQ(info.blk_off(2), data + 24u);
    EXPECT_EQ(info.blk_size(2), 8u);
    EXPECT_EQ(info.blk_num(2), 2u);

    // smpl with zero loops is ignored; a loop from 0 to the end is a single repeating block.
    const auto no_loops = riff({{"fmt ", fmt_pcm(1, 2, 48000, 16)}, {"smpl", smpl(0, 2, 5)},
                                {"data", payload(32)}});
    ASSERT_EQ(parse(no_loops, info), 0u);
    EXPECT_EQ(info.u(0x20), 0u);
    EXPECT_EQ(info.u(0x24), 0u);
    EXPECT_EQ(info.u(0x44), 1u);
    const auto whole = riff({{"fmt ", fmt_pcm(1, 2, 48000, 16)}, {"smpl", smpl(1, 0, 7)},
                             {"data", payload(32)}});
    ASSERT_EQ(parse(whole, info), 0u);
    ASSERT_EQ(info.u(0x44), 1u);
    EXPECT_EQ(info.blk_size(0), 32u);
    EXPECT_EQ(info.blk_rep(0), 0xffffffffu);
}

// Extensible PCM (KSDATAFORMAT_SUBTYPE_PCM, rodata 0x70820) routes to the integer PCM ladder.
TEST(Ngs2Waveform, ParseExtensiblePcm) {
    const auto wav = riff({{"fmt ", fmt_ext(6, 48000, 16, kGuidPcm)}, {"data", payload(120)}});
    Info info;
    ASSERT_EQ(parse(wav, info), 0u);
    EXPECT_EQ(info.u(0x00), 0x12u);
    EXPECT_EQ(info.u(0x04), 6u);
    EXPECT_EQ(info.u(0x2c), 12u);
    EXPECT_EQ(info.u(0x28), 10u);
    uint8_t other[16];
    std::memcpy(other, kGuidPcm, 16);
    other[0] = 3;   // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT is not accepted
    const auto flt = riff({{"fmt ", fmt_ext(2, 48000, 32, other)}, {"data", payload(32)}});
    EXPECT_EQ(parse(flt, info), kUnsupported);
}

// ATRAC9 (0x11ce2): config big-endian from fmt+0x2c; channels and rate from the fmt header;
// fact required; loop points shifted by the decoder delay.
TEST(Ngs2Waveform, ParseAtrac9) {
    const uint8_t cfg[4] = {0xfe, 0x72, 0x09, 0xf0};   // rate idx 7, 80-byte frames, 4/superframe
    const auto file = riff({{"fmt ", fmt_ext(1, 44100, 0, kGuidAtrac9, cfg)}, {"fact", u32s({900, 0})},
                            {"data", payload(640)}});
    Info info;
    ASSERT_EQ(parse(file, info), 0u);
    EXPECT_EQ(info.u(0x00), 0x40u);
    EXPECT_EQ(info.u(0x04), 1u) << "channels from fmt, not the config";
    EXPECT_EQ(info.u(0x08), 44100u) << "rate from fmt, not the config";
    EXPECT_EQ(info.u(0x0c), 0xfe7209f0u) << "config word is big-endian";
    EXPECT_EQ(info.u(0x28), 900u);
    EXPECT_EQ(info.u(0x2c), 80u) << "frame bytes";
    EXPECT_EQ(info.u(0x30), 256u) << "frame samples for rate index 7";
    EXPECT_EQ(info.u(0x34), 4u);
    EXPECT_EQ(info.u(0x38), 320u);
    EXPECT_EQ(info.u(0x3c), 1024u);
    EXPECT_EQ(info.u(0x40), 256u) << "delay = one frame";
    EXPECT_EQ(info.u(0x20), 0u);
    EXPECT_EQ(info.u(0x24), 0u);
    ASSERT_EQ(info.u(0x44), 1u);
    EXPECT_EQ(info.blk_off(0), (uint64_t)info.u(0x18));
    EXPECT_EQ(info.blk_size(0), 640u) << "ceil((900+256)/1024) superframes of 320 bytes";
    EXPECT_EQ(info.blk_skip(0), 256u);
    EXPECT_EQ(info.blk_num(0), 900u);

    const auto looped = riff({{"fmt ", fmt_ext(2, 48000, 0, kGuidAtrac9, cfg)},
                              {"fact", u32s({4000, 0})}, {"smpl", smpl(1, 1300, 3299)},
                              {"data", payload(1280)}});
    ASSERT_EQ(parse(looped, info), 0u);
    EXPECT_EQ(info.u(0x20), 1300u - 256u) << "begin minus delay";
    EXPECT_EQ(info.u(0x24), 3299u - 256u + 1u) << "end minus delay, plus one";
    EXPECT_EQ(info.u(0x44), 3u);
    EXPECT_EQ(info.blk_rep(1), 0xffffffffu);

    const auto no_fact = riff({{"fmt ", fmt_ext(1, 48000, 0, kGuidAtrac9, cfg)}, {"data", payload(320)}});
    EXPECT_EQ(parse(no_fact, info), kInvalidData) << "ATRAC9 needs a fact chunk";

    // A 0x34-byte extensible fmt is ATRAC9 whatever its GUID says.
    uint8_t other[16]{};
    other[0] = 0x55;
    const auto by_size = riff({{"fmt ", fmt_ext(1, 48000, 0, other, cfg)}, {"fact", u32s({10, 0})},
                               {"data", payload(320)}});
    ASSERT_EQ(parse(by_size, info), 0u);
    EXPECT_EQ(info.u(0x00), 0x40u);
}

// VAGp (0x11174..0x11f04): samples = frames*28, minus one frame unless the second frame's flag
// byte has bit 1; loop points from the ADPCM flags (6 = start, 3 = end).
TEST(Ngs2Waveform, ParseVag) {
    Bytes v;
    v.raw("VAGp", 4);
    v.zeros(8);
    v.be32(32);      // data size
    v.be32(48000);   // rate
    v.zeros(0x1e - 0x14);
    v.u8(1);         // channels
    v.zeros(0x30 - 0x1f);
    v.zeros(32);
    Info info;
    ASSERT_EQ(parse(v.v, info), 0u);
    EXPECT_EQ(info.u(0x00), 0x1cu);
    EXPECT_EQ(info.u(0x04), 1u);
    EXPECT_EQ(info.u(0x08), 48000u);
    EXPECT_EQ(info.u(0x18), 0x30u);
    EXPECT_EQ(info.u(0x1c), 32u);
    EXPECT_EQ(info.u(0x28), 28u) << "two frames, minus one without the flag";
    EXPECT_EQ(info.u(0x2c), 16u);
    EXPECT_EQ(info.u(0x30), 28u);
    EXPECT_EQ(info.u(0x24), 0u);
    ASSERT_EQ(info.u(0x44), 1u);
    EXPECT_EQ(info.blk_off(0), 0x30u);
    EXPECT_EQ(info.blk_size(0), 16u);
    EXPECT_EQ(info.blk_num(0), 28u);

    // Stereo, 4 frames of 32 bytes: frame 1 flag 6 (loop start), frame 3 flag 3 (loop end).
    Bytes s;
    s.raw("VAGp", 4);
    s.zeros(8);
    s.be32(128);
    s.be32(22050);
    s.zeros(0x1e - 0x14);
    s.u8(2);
    s.zeros(0x30 - 0x1f);
    s.zeros(128);
    s.v[0x30 + 32 + 1] = 6;
    s.v[0x30 + 64 + 1] = 2;
    s.v[0x30 + 96 + 1] = 3;
    ASSERT_EQ(parse(s.v, info), 0u);
    EXPECT_EQ(info.u(0x28), 112u) << "flag bit 1 set: no frame subtracted";
    EXPECT_EQ(info.u(0x20), 28u);
    EXPECT_EQ(info.u(0x24), 112u);
    ASSERT_EQ(info.u(0x44), 2u);
    EXPECT_EQ(info.blk_off(0), 0x30u);
    EXPECT_EQ(info.blk_size(0), 32u);
    EXPECT_EQ(info.blk_num(0), 28u);
    EXPECT_EQ(info.blk_off(1), 0x30u + 32u);
    EXPECT_EQ(info.blk_size(1), 96u);
    EXPECT_EQ(info.blk_rep(1), 0xffffffffu);

    // A declared size past the image fails the flag scan.
    std::vector<uint8_t> big = s.v;
    big[0x0c + 3] = 0;
    big[0x0c + 2] = 1;   // 256 bytes declared, 128 present
    EXPECT_EQ(parse(big, info), kInvalidData);
}

// CalcWaveformBlock (0x12060): u32 arguments, the bytes-per-sample table for 0x10..0x1b, VAG and
// ATRAC9 mappings, no channel cap, block zeroed before the format is examined.
TEST(Ngs2Waveform, CalcBlock) {
    HleFn calc = fn("sceNgs2CalcWaveformBlock");
    uint32_t pcm[6] = {0x12, 2, 48000, 0, 0, 0};
    uint8_t b[40], c[40];
    std::memset(b, 0xAB, sizeof b);
    ASSERT_EQ(calc(addr(pcm), 10, 20, addr(b), 0, 0), 0u);
    EXPECT_EQ(rd64(b + 0x00), 40u);
    EXPECT_EQ(rd64(b + 0x08), 80u);
    EXPECT_EQ(rd32(b + 0x10), 0u);
    EXPECT_EQ(rd32(b + 0x14), 0u);
    EXPECT_EQ(rd32(b + 0x18), 20u);
    EXPECT_EQ(rd64(b + 0x20), 0u);
    ASSERT_EQ(calc(addr(pcm), 0xdead00000000000aull, 0xbeef000000000014ull, addr(c), 0, 0), 0u);
    EXPECT_EQ(std::memcmp(b, c, sizeof b), 0) << "upper argument halves are ignored";

    const uint32_t bps[12] = {1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 8, 8};
    for (uint32_t t = 0x10; t <= 0x1b; ++t) {
        uint32_t f[6] = {t, 100, 48000, 0, 0, 0};
        ASSERT_EQ(calc(addr(f), 3, 5, addr(b), 0, 0), 0u) << t;
        EXPECT_EQ(rd64(b), 3u * 100u * bps[t - 0x10]) << t;
        EXPECT_EQ(rd64(b + 8), 5u * 100u * bps[t - 0x10]) << t;
    }
    uint32_t wide[6] = {0x1a, 2, 48000, 0, 0, 0};
    ASSERT_EQ(calc(addr(wide), 0x20000000u, 1, addr(b), 0, 0), 0u);
    EXPECT_EQ(rd64(b), 0u) << "offset is computed modulo 2^32, not rejected";

    uint32_t vag[6] = {0x1c, 2, 48000, 0, 0, 0};
    ASSERT_EQ(calc(addr(vag), 30, 60, addr(b), 0, 0), 0u);
    EXPECT_EQ(rd64(b), 32u) << "frame 1 of 32-byte stereo frames";
    EXPECT_EQ(rd64(b + 8), 96u) << "frames 1..3";
    EXPECT_EQ(rd32(b + 0x14), 0u);
    EXPECT_EQ(rd32(b + 0x18), 84u) << "whole frames of 28";

    uint32_t at9[6] = {0x40, 2, 48000, 0xfe7209f0u, 0, 0};
    ASSERT_EQ(calc(addr(at9), 1500, 100, addr(b), 0, 0), 0u);
    EXPECT_EQ(rd64(b), 320u) << "superframe 1";
    EXPECT_EQ(rd64(b + 8), 320u);
    EXPECT_EQ(rd32(b + 0x14), 732u) << "delay + position inside the superframe";
    EXPECT_EQ(rd32(b + 0x18), 100u);

    std::memset(b, 0xAB, sizeof b);
    EXPECT_EQ(calc(addr(pcm), 0, 1, 0, 0, 0), kNullOut);
    EXPECT_EQ(calc(0, 0, 1, addr(b), 0, 0), kNullFormat);
    for (uint8_t x : b) EXPECT_EQ(x, 0u) << "zeroed before the format check";
    std::memset(b, 0xAB, sizeof b);
    uint32_t bad[6] = {0x20, 2, 48000, 0, 0, 0};
    EXPECT_EQ(calc(addr(bad), 0, 1, addr(b), 0, 0), kUnsupported);
    for (uint8_t x : b) EXPECT_EQ(x, 0u);
}

// Type 0x43 (third SubFormat GUID, rodata 0x70840): fact gives samples and delay, cmap[0] the
// config, and a "loop" chunk places the blocks directly (0x11a25..0x11c13).
TEST(Ngs2Waveform, ParseExt43) {
    std::vector<uint8_t> loop = u32s({1, 0x100, 0x80, 7, 100, 0x300, 0x280, 9, 400});
    const auto file = riff({{"fmt ", fmt_ext(2, 48000, 16, kGuidExt43)}, {"fact", u32s({1000, 64})},
                            {"cmap", {5, 0, 0, 0}}, {"loop", loop}, {"data", payload(0x400)}});
    Info info;
    ASSERT_EQ(parse(file, info), 0u);
    const uint64_t data = info.u(0x18);
    EXPECT_EQ(info.u(0x00), 0x43u);
    EXPECT_EQ(info.u(0x0c), 5u);
    EXPECT_EQ(info.u(0x20), 100u);
    EXPECT_EQ(info.u(0x24), 400u);
    EXPECT_EQ(info.u(0x28), 1000u);
    EXPECT_EQ(info.u(0x2c), 0u);
    EXPECT_EQ(info.u(0x30), 1u);
    EXPECT_EQ(info.u(0x40), 64u);
    ASSERT_EQ(info.u(0x44), 3u);
    EXPECT_EQ(info.blk_off(0), data);
    EXPECT_EQ(info.blk_size(0), 0x80u);
    EXPECT_EQ(info.blk_skip(0), 64u);
    EXPECT_EQ(info.blk_num(0), 36u);
    EXPECT_EQ(info.blk_off(1), data + 0x100);
    EXPECT_EQ(info.blk_size(1), 0x180u);
    EXPECT_EQ(info.blk_rep(1), 0xffffffffu);
    EXPECT_EQ(info.blk_skip(1), 7u);
    EXPECT_EQ(info.blk_num(1), 300u);
    EXPECT_EQ(info.blk_off(2), data + 0x300);
    EXPECT_EQ(info.blk_size(2), 0x100u);
    EXPECT_EQ(info.blk_skip(2), 9u);
    EXPECT_EQ(info.blk_num(2), 600u);
}

// The pure mapping is reachable without the dispatcher too (same function the parse uses).
TEST(Ngs2Waveform, CalcBlockDirectRefusesZeroFrameAtrac9) {
    uint8_t b[ngs2_waveform::kNgs2NativeWaveformBlockBytes];
    std::memset(b, 0xAB, sizeof b);
    // Rate index 0 has no frame-sample count; the module would divide by zero.
    EXPECT_EQ(ngs2_waveform::calc_block(0x40, 2, 0xfe0209f0u, 0, 10, b),
              ngs2_waveform::kNgs2NativeErrUnsupportedFormat);
    for (uint8_t x : b) EXPECT_EQ(x, 0u);
}
