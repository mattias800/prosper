// test_bc_decode — verify BC1/BC2/BC3 block decompression against hand-built blocks with known output.
// Exit code 0 = pass. Pure, no Vulkan/game dump.
#include "../src/gpu/bc_decode.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } else { printf("  [ok]   %s\n", m); } } while (0)

static void put16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)(v & 0xFF); p[1] = (uint8_t)(v >> 8); }

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool parse_hex(const char* text, uint8_t* out, size_t out_size) {
    if (std::strlen(text) != out_size * 2) return false;
    for (size_t i = 0; i < out_size; i++) {
        const int hi = hex_nibble(text[i * 2]), lo = hex_nibble(text[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

// RGB565 constants
static const uint16_t RED565 = 0xF800, BLUE565 = 0x001F, GREEN565 = 0x07E0, BLACK565 = 0x0000;

int main() {
    printf("== test_bc_decode ==\n");

    CHECK(bc_block_bytes(DataFormat::Bc1) == 8, "BC1 block = 8 bytes");
    CHECK(bc_block_bytes(DataFormat::Bc3) == 16, "BC3 block = 16 bytes");
    CHECK(bc_block_bytes(DataFormat::Float32) == 0, "non-BC format -> 0 block bytes");

    // Complete 4x4 known-block outputs for the S3TC/RGTC families. These exercise every decoded
    // texel, byte order, direct-alpha nibble, and both independent BC5 channel streams.
    struct KnownBlock {
        DataFormat format;
        const char* name;
        const char* block;
        const char* rgba;
    };
    static const KnownBlock kKnownBlocks[] = {
        {DataFormat::Bc1, "BC1",
         "00f81f0000000000",
         "ff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ff"},
        {DataFormat::Bc2, "BC2",
         "1032547698badcfe00f81f0000000000",
         "ff000000ff000011ff000022ff000033ff000044ff000055ff000066ff000077ff000088ff000099ff0000aaff0000bbff0000ccff0000ddff0000eeff0000ff"},
        {DataFormat::Bc3, "BC3",
         "ff0088c6fa88c6fa00f81f0000000000",
         "ff0000ffff000000ff0000daff0000b6ff000091ff00006dff000048ff000024ff0000ffff000000ff0000daff0000b6ff000091ff00006dff000048ff000024"},
        {DataFormat::Bc4, "BC4",
         "ff0088c6fa88c6fa",
         "ff0000ff000000ffda0000ffb60000ff910000ff6d0000ff480000ff240000ffff0000ff000000ffda0000ffb60000ff910000ff6d0000ff480000ff240000ff"},
        {DataFormat::Bc5, "BC5",
         "ff0088c6fa88c6fa00ff773905773905",
         "ffff00ff000000ffdacc00ffb69900ff916600ff6d3300ff48ff00ff240000ffffff00ff000000ffdacc00ffb69900ff916600ff6d3300ff48ff00ff240000ff"},
    };
    for (const KnownBlock& v : kKnownBlocks) {
        const size_t block_size = bc_block_bytes(v.format);
        uint8_t block[16] = {}, expected[64] = {}, out[64] = {};
        const bool parsed = parse_hex(v.block, block, block_size) && parse_hex(v.rgba, expected, sizeof expected);
        const bool decoded = parsed && bc_decode_surface(out, block, block_size, 4, 4, v.format);
        char label[64];
        std::snprintf(label, sizeof label, "%s known block is bit-exact", v.name);
        CHECK(decoded && std::memcmp(out, expected, sizeof out) == 0, label);
    }

    // --- BC1: solid red 4x4 (c0>c1, all indices 0) ---
    {
        uint8_t blk[8] = {0};
        put16(blk + 0, RED565);   // c0
        put16(blk + 2, BLUE565);  // c1 (c0>c1 -> 4-color mode)
        // indices all 0 -> every texel = color0 = red
        uint8_t out[4 * 4 * 4];
        bool ok = bc_decode_surface(out, blk, sizeof blk, 4, 4, DataFormat::Bc1);
        CHECK(ok, "BC1 decode returns true");
        bool all_red = true;
        for (int t = 0; t < 16; t++) {
            const uint8_t* px = out + t * 4;
            if (px[0] != 255 || px[1] != 0 || px[2] != 0 || px[3] != 255) all_red = false;
        }
        CHECK(all_red, "BC1 solid: all 16 texels = opaque red (255,0,0,255)");
    }

    // --- BC1: interpolated color at index 2 = (2*c0 + c1)/3 ---
    {
        uint8_t blk[8] = {0};
        put16(blk + 0, RED565);    // c0 = (255,0,0)
        put16(blk + 2, GREEN565);  // c1 = (0,255,0);  c0>c1
        // set texel 0 to index 2 (2 bits = 0b10 in the low bits)
        blk[4] = 0x02;
        uint8_t out[4 * 4 * 4];
        bc_decode_surface(out, blk, sizeof blk, 4, 4, DataFormat::Bc1);
        // color2 = (2*255+0)/3=170 R, (2*0+255)/3=85 G
        CHECK(out[0] == 170 && out[1] == 85 && out[2] == 0, "BC1 index2 = (2*c0+c1)/3 (170,85,0)");
    }

    // --- BC1: punch-through (c0<=c1) index 3 = transparent black ---
    {
        uint8_t blk[8] = {0};
        put16(blk + 0, BLUE565);   // c0
        put16(blk + 2, RED565);    // c1 -> c0<c1 -> 3-color+alpha mode
        blk[4] = 0x03;             // texel 0 index = 3 -> transparent black
        uint8_t out[4 * 4 * 4];
        bc_decode_surface(out, blk, sizeof blk, 4, 4, DataFormat::Bc1);
        CHECK(out[0] == 0 && out[1] == 0 && out[2] == 0 && out[3] == 0, "BC1 punch-through index3 = transparent black");
    }

    // --- BC3: color = red (4-color, ignores c0<=c1 alpha rule), alpha ramp a0=255/a1=0 ---
    {
        uint8_t blk[16] = {0};
        blk[0] = 255; blk[1] = 0;  // alpha endpoints a0>a1 -> a[0]=255, a[1]=0
        // alpha indices (bytes 2..7): texel0 -> index 1 (=a1=0), texel1 -> index 0 (=a0=255), rest 0
        // 3 bits each: texel0=1 (0b001), texel1=0. bits = 0b...000 001 -> byte2 low bits = 0x01
        blk[2] = 0x01;
        put16(blk + 8, RED565);    // color c0
        put16(blk + 10, BLUE565);  // color c1
        // color indices 0 -> all red
        uint8_t out[4 * 4 * 4];
        bool ok = bc_decode_surface(out, blk, sizeof blk, 4, 4, DataFormat::Bc3);
        CHECK(ok, "BC3 decode returns true");
        CHECK(out[0] == 255 && out[1] == 0 && out[2] == 0, "BC3 color sub-block = red");
        CHECK(out[3] == 0, "BC3 texel0 alpha index1 -> a1 = 0");
        CHECK(out[7] == 255, "BC3 texel1 alpha index0 -> a0 = 255");
    }

    // --- BC4: single-channel ramp (a0>a1 -> 6 interpolants); hardware channel rule (R,0,0,255) ---
    {
        uint8_t blk[8] = {0};
        blk[0] = 255; blk[1] = 0;   // e0=255, e1=0 (e0>e1 -> 6-interpolant ramp)
        // 3-bit indices: t0=0 (=e0), t1=1 (=e1), t2=2 (=(6*e0+e1)/7), rest 0
        uint64_t bits = (1ull << 3) | (2ull << 6);
        for (int i = 0; i < 6; i++) blk[2 + i] = (uint8_t)(bits >> (i * 8));
        uint8_t out[4 * 4 * 4];
        bool ok = bc_decode_surface(out, blk, sizeof blk, 4, 4, DataFormat::Bc4);
        CHECK(ok, "BC4 decode returns true");
        CHECK(out[0] == 255 && out[1] == 0 && out[2] == 0 && out[3] == 255, "BC4 t0 idx0 -> (255,0,0,255)");
        CHECK(out[4] == 0, "BC4 t1 idx1 -> e1 = 0");
        CHECK(out[8] == (uint8_t)((6 * 255 + 1 * 0) / 7), "BC4 t2 idx2 -> (6*e0+e1)/7 = 218");
    }

    // --- BC5: two independent channel blocks -> (R,G,0,255) ---
    {
        uint8_t blk[16] = {0};
        blk[0] = 200; blk[1] = 100;  // R channel: e0=200 (idx0 everywhere)
        blk[8] = 40;  blk[9] = 10;   // G channel: e0=40
        uint8_t out[4 * 4 * 4];
        bool ok = bc_decode_surface(out, blk, sizeof blk, 4, 4, DataFormat::Bc5);
        CHECK(ok, "BC5 decode returns true");
        CHECK(out[0] == 200 && out[1] == 40 && out[2] == 0 && out[3] == 255, "BC5 -> (R,G,0,255) = (200,40,0,255)");
    }

    // --- BC7 ---
    // Bit-writer mirroring the decoder's little-endian stream, to build spec-exact blocks.
    struct BitW {
        uint8_t b[16]; uint32_t pos;
        BitW() : pos(0) { std::memset(b, 0, sizeof b); }
        void put(uint32_t v, uint32_t n) {
            for (uint32_t i = 0; i < n; i++, pos++)
                if ((v >> i) & 1u) b[pos >> 3] |= (uint8_t)(1u << (pos & 7));
        }
    };

    // Mode 5 solid color: rotation=0, both endpoints equal, all indices 0.
    // 7-bit color 0x40 expands to (0x40<<1) | (0x80>>8) = 0x81; alpha is 8-bit literal.
    {
        BitW w;
        w.put(0x20, 6);            // mode 5: five 0 bits then the 1 (LSB-first -> value 0b100000)
        w.put(0, 2);               // rotation
        for (int i = 0; i < 2; i++) w.put(0x40, 7);   // R endpoints
        for (int i = 0; i < 2; i++) w.put(0x40, 7);   // G
        for (int i = 0; i < 2; i++) w.put(0x40, 7);   // B
        for (int i = 0; i < 2; i++) w.put(0xC3, 8);   // A endpoints (literal 8-bit)
        // color indices: anchor texel0 = 1 bit, texels 1..15 = 2 bits, all zero
        // alpha indices: same widths, all zero -> stream is already zero; nothing to put.
        uint8_t out[4 * 4 * 4];
        bool ok = bc_decode_surface(out, w.b, sizeof w.b, 4, 4, DataFormat::Bc7);
        CHECK(ok, "BC7 decode returns true");
        bool solid = true;
        for (int t = 0; t < 16; t++) {
            const uint8_t* px = out + t * 4;
            if (px[0] != 0x81 || px[1] != 0x81 || px[2] != 0x81 || px[3] != 0xC3) solid = false;
        }
        CHECK(solid, "BC7 mode5 solid: every texel = (0x81,0x81,0x81,0xC3)");
    }

    // Mode 6 interpolation: ep0 = 0 (p=0), ep1 = 0xFF (raw 0x7F, p=1); texel1 index 8 -> weight 34.
    {
        BitW w;
        w.put(0x40, 7);            // mode 6: six 0 bits then the 1
        for (int c = 0; c < 4; c++) { w.put(0x00, 7); w.put(0x7F, 7); }  // RGBA endpoint pairs? NO:
        // ^ WRONG order — endpoints are grouped per channel: e0.r, e1.r, then e0.g, e1.g, ... For
        //   this block e0.* = 0 and e1.* = 0x7F for every channel, so the grouped order writes the
        //   same bits either way (0, 0x7F repeated 4x). Kept simple on purpose.
        w.put(0, 1); w.put(1, 1);  // p-bits: e0 -> 0, e1 -> 1  => e1 = (0x7F<<1)|1 = 0xFF
        w.put(0, 3);               // texel0 (anchor) 3-bit index = 0
        w.put(8, 4);               // texel1 4-bit index = 8 -> aWeight4[8] = 34
        uint8_t out[4 * 4 * 4];
        bc_decode_surface(out, w.b, sizeof w.b, 4, 4, DataFormat::Bc7);
        CHECK(out[0] == 0 && out[3] == 0, "BC7 mode6 t0 idx0 -> endpoint0 (0, alpha 0)");
        uint8_t expect = (uint8_t)((0 * (64 - 34) + 255 * 34 + 32) >> 6);   // = 135
        CHECK(out[4] == expect && out[5] == expect && out[6] == expect && out[7] == expect,
              "BC7 mode6 t1 idx8 -> (255*34+32)>>6 = 135 in all channels");
    }

    // Mode 1 two-subset, partition 0 (left 2 columns subset 0, right 2 subset 1): subset0 = red,
    // subset1 = blue, all indices 0. 6-bit endpoints + shared p-bit per subset.
    {
        BitW w;
        w.put(0x02, 2);            // mode 1: one 0 bit then the 1
        w.put(0, 6);               // partition 0
        // R endpoints for e0..e3 (subset0 e0,e1 then subset1 e0,e1): red -> 0x3F,0x3F,0,0
        w.put(0x3F, 6); w.put(0x3F, 6); w.put(0x00, 6); w.put(0x00, 6);
        // G: all 0
        for (int i = 0; i < 4; i++) w.put(0x00, 6);
        // B: 0,0,0x3F,0x3F
        w.put(0x00, 6); w.put(0x00, 6); w.put(0x3F, 6); w.put(0x3F, 6);
        w.put(0, 1); w.put(0, 1);  // shared p-bits 0: 0x3F -> 7-bit 0x7E -> expands to 0xFD; 0 stays 0
        // indices: 3-bit, anchors (texel0 subset0, texel15 subset1 for partition 0) 2-bit; all zero.
        uint8_t out[4 * 4 * 4];
        bc_decode_surface(out, w.b, sizeof w.b, 4, 4, DataFormat::Bc7);
        CHECK(out[0] == 0xFD && out[1] == 0 && out[2] == 0 && out[3] == 255, "BC7 mode1 texel(0,0) = red-ish 0xFD (subset 0)");
        const uint8_t* pr = out + 3 * 4;   // texel (0,3): right column -> subset 1
        CHECK(pr[0] == 0 && pr[1] == 0 && pr[2] == 0xFD && pr[3] == 255, "BC7 mode1 texel(0,3) = blue-ish 0xFD (subset 1)");
    }

    // Reserved mode (all 8 mode bits zero) -> transparent black, no crash.
    {
        uint8_t blk[16] = {0};
        uint8_t out[4 * 4 * 4];
        std::memset(out, 0xAA, sizeof out);
        bc_decode_surface(out, blk, sizeof blk, 4, 4, DataFormat::Bc7);
        CHECK(out[0] == 0 && out[3] == 0, "BC7 reserved mode -> transparent black");
    }

    // --- BC6H (UF16) -> clamped RGBA8 (#273) ---
    {
        // All-zero block: mode 0 (2-subset 10.555), every endpoint/index 0 -> RGB 0, alpha 255.
        uint8_t blk[16]; std::memset(blk, 0, sizeof blk);
        uint8_t out[4 * 4 * 4];
        CHECK(bc_decode_surface(out, blk, sizeof blk, 4, 4, DataFormat::Bc6), "BC6H surface decodes (UF16 wired)");
        CHECK(out[0] == 0 && out[1] == 0 && out[2] == 0 && out[3] == 255, "BC6H zero block -> opaque black");
    }
    {
        // Mode 11 (5-bit mode 00011: one subset, direct 10-bit endpoints, no delta). Constant-color
        // block: rw=511, gw=0, bw=100, endpoint1/indices all 0 -> every texel = endpoint 0.
        // Expected per the spec math: R unq = ((511<<16)+0x8000)>>10 = 32736, finish (x*31)>>6 =
        // 15856 = half 0x3DF0 ~ 1.48 -> clamps to 255; G = 0; B: ((100<<16)+0x8000)>>10 = 6432 ->
        // 3115 = half 0x0C2B ~ 0.0004 -> 0. So (255, 0, 0, 255).
        uint8_t blk[16]; std::memset(blk, 0, sizeof blk);
        const uint64_t low = 3ull | (511ull << 5) | (100ull << 25);   // mode(5b) rw(10b) gw(10b) bw(10b...)
        std::memcpy(blk, &low, 8);
        uint8_t out[4 * 4 * 4];
        bc_decode_surface(out, blk, sizeof blk, 4, 4, DataFormat::Bc6);
        const uint8_t* p5 = out + ((size_t)1 * 4 + 1) * 4;   // interior texel, same expectation
        CHECK(out[0] == 255 && out[1] == 0 && out[2] == 0 && out[3] == 255, "BC6H mode-11 endpoint0 -> (255,0,0,255)");
        CHECK(p5[0] == 255 && p5[1] == 0 && p5[2] == 0, "BC6H mode-11 constant across the block");
    }

    // One complete, non-trivial output vector for every valid BC6H and BC7 mode. Expected RGBA bytes
    // were derived from the Khronos equations and are stored directly in this dependency-free test.
    struct ModeVector { const char* block; const char* rgba; };
    static const ModeVector kBc6Modes[] = {
        {"14b4e30a3b7cc0c9f00f126eca237600", "33721dff34731cff377519ff36741aff2e761cff35741bff34731cff3a7816ff2d711eff2d741dff2f791aff387618ff2e761cff2c6e1fff2c6e1fff2c6e1fff"},
        {"f9e5d64cd79bbf08fb2710c0bd27716f", "181005ff150f08ff181005ff0e9501ff0a093bff0c0b29ff08085cffc20603ff0f0d11ff110e0cff110e0cff531102ff0f0d11ff08085cff0c0b29ff165a02ff"},
        {"a251afa5675eb8f29ad15ca9cf07b9c8", "0844ebff0845eeff0844eaff0844edff0845eeff0945f1ff0843f1ff084adbff0844ebff0942f4ff0844edff0849dfff0848e2ff0942f4ff0843f1ff0846eaff"},
        {"065ccca5372efb01cd9bcc93aac173fb", "1378e9ff1377eaff1474edff1377eaff1377eaff1379e8ff1376ebff148ddeff127be7ff1377eaff1474edff1480dcff1378e9ff1378e9ff1494dfff147cdbff"},
        {"0a6feb5246104585c02c9bccb46e706a", "5a0129ff5b0129ff5b0129ff59012aff570124ff58012bff56012cff5a0129ff5a0127ff57012bff5b0129ff5c0128ff560123ff580126ff5a0129ff59012aff"},
        {"2e58693dcdc4a586f9a4c5b932fd505e", "1a3b07ff174f08ff184608ff165e0aff165609ff193f07ff165e0aff174f08ff253a04ff283f04ff192105ff1f3104ff1b2605ff253a04ff1d2b05ff192105ff"},
        {"d24735e43a7cb4d9706bb31093105443", "023c77ff02434dff01ba40ff01cb35ff023e69ff0770eaff03838fff02473eff023c77ff02a857ff01cb35ff02405bff0579bdff0770eaff023c77ff02405bff"},
        {"d629b5b51abaa9e32b6ca4053d31c94e", "064711ff065513ff076315ff022431ff063c10ff078319ff030611ff030f1dff063c10ff076315ff031b29ff031b29ff079f1bff030815ff022431ff022431ff"},
        {"ba8eb5c1a26d2e3d3659e9b7ddb2c437", "743218ff3f1b11ff391710ff4b1e13ff4b1e13ff4b1e13ff672b16ff4a1c03ff823919ff4a1c03ff5e1d02ff5e1d02ff361b04ff311b04ff361b04ff671d02ff"},
        {"bec04793d9cd132fa933870e352fa50a", "010301ff000200ffff371eff6c0d04ff080fffffff371effff7056ffff1b0cff7356ffffff371effff1b0cff080601ff3031ffffffff0eff080601ff000200ff"},
        {"032c1e3302ee2032a078561c5fe9f3bf", "0dff03ff341100ff272b00ff213e00ff1e6201ff1aa101ff3f0700ff0fff02ff650200ff1aa101ff2d1c00ff590300ff14ff01ff650200ff650200ff3a0c00ff"},
        {"077be5bd7ccdb119fe39e25ec9768ba7", "a97806ff6b3f08ff966a07ffd3ab05ffddb905ff704408ff704408ffbf8f06ff966a07ff7b5407ffb37f06ffa97806ff805b07ff9f7106ffa97806ff896207ff"},
        {"ab5dfdc956b78a0ddf338f5fb8264501", "5602b5ff5203bfff5902aeff5902aeff5003c3ff5502b7ff5003c3ff5702b1ff5502b7ff5302bcff5602b3ff5902acff5702b1ff5802b0ff5a02aaff5b02a9ff"},
        {"0fb6bef14b8b1de0d69d6d0034c353d8", "397f01ff397e01ff397e01ff397e01ff397e01ff397e01ff397f01ff397f01ff397f01ff397f01ff397f01ff397e01ff397f01ff397f01ff397e01ff397e01ff"},
    };
    static const ModeVector kBc7Modes[] = {
        {"214880b50892e1613d4cec85fecac5a7", "1851d5ff4a4a08ff096ad9ff2100b5ff324d6dff424b2aff0094e7ff133ecaff3a4c4bffb85845ffbb7636ff1c15bcffb1235effad086bffb85845ffbf912aff"},
        {"8ae51b52a7e59533c9299ed40697d48e", "a680b4ff807e68ff9b93c5ff618f3eff599334ffb76299ff877a72ff959dcdffbd5891ff78825effa08abcff78825eff618f3effb26ba2ff718654ffac75aaff"},
        {"dc367f8a9c8a7203be67ee70ae43d41d", "e18ae4ffe465d9ffe465d9ffe742ceffdeadefff7bcee7ff6eaec1ffdeadefff5f8b99ff31a6cbff2100c6ff526b73ff6eaec1ff39f7ceff2951c9ff7bcee7ff"},
        {"f8215576b71ab586d9ea8669e42ebc15", "7da996ff91d56dffe7af3fffdc60a6ffe18773ff5450eaff687cc1ffecd60cffecd60cff5450eaff5450eaffe18773ffe7af3fffe7af3fff7da996ff91d56dff"},
        {"90f65ec7c601a9a55de310851b9b82b6", "b5bd631cb7a84e28ba923828b7a84e41b5bd6335bd73181cbc7d2328b5bd6341b89e4335b89e4341b7a84e35b6b35835b5bd6328bb882d1cbb882d41bb882d28"},
        {"e0a7c7f888c7b5fcec00857c4d548a13", "3eb52dc73eb53fc71e8f2d703eb533c74ec72df14ec733f14ec733f12ea1339a2ea1399a4ec739f14ec72df13eb539c72ea13f9a1e8f2d701e8f33702ea12d9a"},
        {"402a6c5be6d4a5eb50f89eba2327ec61", "a9b739a591bd73b582c197bf60caead665c9dfd37ec2a2c278c4b0c673c5bbc99abb5daf9fba52ac87c08cbc9fba52ac6fc6c6cc65c9dfd3a4b844a88bbf81b9"},
        {"80347cb328fcb019510e86d9e88bfc2c", "8655341c75918da328c392617dc71c6575918da375918da38655341c837a2c344dab8f817dc71c657dc71c6575918da380a2244d837a2c3475918da39a798ac3"},
    };
    auto check_modes = [&](DataFormat format, const char* family, const ModeVector* vectors, size_t count) {
        for (size_t mode = 0; mode < count; mode++) {
            uint8_t block[16] = {}, expected[64] = {}, out[64] = {};
            const bool parsed = parse_hex(vectors[mode].block, block, sizeof block) &&
                                parse_hex(vectors[mode].rgba, expected, sizeof expected);
            const bool decoded = parsed && bc_decode_surface(out, block, sizeof block, 4, 4, format);
            char label[64];
            std::snprintf(label, sizeof label, "%s mode %zu known block is bit-exact", family, mode);
            CHECK(decoded && std::memcmp(out, expected, sizeof out) == 0, label);
        }
    };
    check_modes(DataFormat::Bc6, "BC6H", kBc6Modes, sizeof kBc6Modes / sizeof kBc6Modes[0]);
    check_modes(DataFormat::Bc7, "BC7", kBc7Modes, sizeof kBc7Modes / sizeof kBc7Modes[0]);

    // The upload adapter accepts byte storage, which does not promise uint64_t alignment. Keep a
    // known BC6H block one byte off an explicitly aligned base to catch typed-load regressions.
    {
        alignas(uint64_t) uint8_t storage[17] = {}, expected[64] = {}, out[64] = {};
        const bool parsed = parse_hex(kBc6Modes[0].block, storage + 1, 16) &&
                            parse_hex(kBc6Modes[0].rgba, expected, sizeof expected);
        const bool decoded = parsed && bc_decode_surface(out, storage + 1, 16, 4, 4, DataFormat::Bc6);
        CHECK(decoded && std::memcmp(out, expected, sizeof out) == 0,
              "BC6H known block decodes from intentionally misaligned byte storage");
    }

    // --- larger surface: 8x8 BC1, second block a distinct solid color; verify block placement ---
    {
        const uint32_t W = 8, H = 8;                 // 2x2 blocks
        uint8_t blocks[4][8]; std::memset(blocks, 0, sizeof blocks);
        const uint16_t cols[4] = {RED565, GREEN565, BLUE565, 0xFFFF /*white*/};
        for (int i = 0; i < 4; i++) { put16(blocks[i] + 0, cols[i]); put16(blocks[i] + 2, BLACK565); } // c0>c1, idx0
        uint8_t out[W * H * 4];
        bc_decode_surface(out, (const uint8_t*)blocks, sizeof blocks, W, H, DataFormat::Bc1);
        // block (1,1) covers texels x=4..7,y=4..7 -> color index 3 = white
        const uint8_t* px = out + ((size_t)5 * W + 5) * 4;
        CHECK(px[0] == 255 && px[1] == 255 && px[2] == 255, "BC1 8x8: bottom-right block = white (block layout correct)");
        // block (0,0) top-left = red
        CHECK(out[0] == 255 && out[1] == 0 && out[2] == 0, "BC1 8x8: top-left block = red");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
