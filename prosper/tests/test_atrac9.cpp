// test_atrac9 — the vendored LibAtrac9 decoder + prosper's Atrac9Decoder glue reproduce a known-good
// decode bit-for-bit. The embedded vector is the first 16 superframes of The Messenger's snd0.at9 and
// the CRC32 of its reference PCM (produced by LibAtrac9 itself, the canonical PS ATRAC9 decoder). This
// guards the glue wiring (config parse, per-frame superframe iteration, output layout) and the vendored
// build integration; pure and self-contained (no game dump), so it runs in CI.
#include "../src/hle/atrac9_decode.hpp"
#include "at9_testvec.h"
#include <cstdio>
#include <cstdint>
#include <vector>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

static uint32_t crc32(const uint8_t* p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(c & 1)));
    }
    return ~c;
}

int main() {
    prosper::Atrac9Decoder dec;
    CHECK(dec.init(kAt9Config), "init from 4-byte config");
    if (!dec.valid()) { printf("test_atrac9: init failed, aborting\n"); return 1; }

    CHECK(dec.channels() == 2, "channels == 2 (stereo)");
    CHECK(dec.sample_rate() == 48000, "sample rate == 48000");
    CHECK(dec.superframe_bytes() == 384, "superframe == 384 bytes");

    const int sf_bytes = dec.superframe_bytes();
    const int sf_samples = dec.superframe_samples();      // per channel
    const int per_sf_interleaved = sf_samples * dec.channels();
    const uint32_t nsf = kAt9DataLen / sf_bytes;

    std::vector<int16_t> pcm;
    std::vector<int16_t> sf(per_sf_interleaved);
    bool decode_ok = true;
    for (uint32_t i = 0; i < nsf; i++) {
        int got = dec.decode_superframe(kAt9Data + i * sf_bytes, sf.data());
        if (got != sf_samples) { decode_ok = false; break; }
        pcm.insert(pcm.end(), sf.begin(), sf.end());
    }
    CHECK(decode_ok, "all 16 superframes decode without error");
    CHECK(pcm.size() == kAt9ExpectedSamples, "decoded sample count matches reference");

    uint32_t crc = crc32(reinterpret_cast<const uint8_t*>(pcm.data()), pcm.size() * sizeof(int16_t));
    printf("  decoded %zu samples, crc32=0x%08x (expected 0x%08x)\n",
           pcm.size(), crc, kAt9ExpectedCrc32);
    CHECK(crc == kAt9ExpectedCrc32, "decoded PCM is bit-exact vs LibAtrac9 reference (CRC32)");

    printf(fails ? "test_atrac9: %d FAILURE(S)\n" : "test_atrac9: all ok\n", fails);
    return fails ? 1 : 0;
}
