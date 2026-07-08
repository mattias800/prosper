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
