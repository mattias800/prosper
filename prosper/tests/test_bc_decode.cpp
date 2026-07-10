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

// RGB565 constants
static const uint16_t RED565 = 0xF800, BLUE565 = 0x001F, GREEN565 = 0x07E0, BLACK565 = 0x0000;

int main() {
    printf("== test_bc_decode ==\n");

    CHECK(bc_block_bytes(DataFormat::Bc1) == 8, "BC1 block = 8 bytes");
    CHECK(bc_block_bytes(DataFormat::Bc3) == 16, "BC3 block = 16 bytes");
    CHECK(bc_block_bytes(DataFormat::Float32) == 0, "non-BC format -> 0 block bytes");

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
