// bc_decode.cpp — see bc_decode.hpp. Reference S3TC/DXTn decode (Khronos data-format spec §18-19).
#include "bc_decode.hpp"
#include <cstring>

namespace prosper::gpu {

uint32_t bc_block_bytes(DataFormat f) {
    switch (f) {
        case DataFormat::Bc1: case DataFormat::Bc4:                       return 8;
        case DataFormat::Bc2: case DataFormat::Bc3:
        case DataFormat::Bc5: case DataFormat::Bc6: case DataFormat::Bc7: return 16;
        default: return 0;
    }
}

namespace {

inline uint16_t rd16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

// RGB565 -> RGB888 with the standard bit-replication expansion (so 0x1F/0x3F map to 0xFF).
inline void unpack565(uint16_t c, uint8_t& r, uint8_t& g, uint8_t& b) {
    uint8_t r5 = (uint8_t)((c >> 11) & 0x1F), g6 = (uint8_t)((c >> 5) & 0x3F), b5 = (uint8_t)(c & 0x1F);
    r = (uint8_t)((r5 << 3) | (r5 >> 2));
    g = (uint8_t)((g6 << 2) | (g6 >> 4));
    b = (uint8_t)((b5 << 3) | (b5 >> 2));
}

// Decode one BC1 color sub-block (8 bytes) into a 4x4 RGBA8 tile (row-major, 16 texels). `dxt1_alpha`
// enables BC1's 1-bit punch-through alpha (color index 3 in the c0<=c1 branch = transparent black);
// BC2/BC3 always use the 4-color branch and supply alpha separately, so pass false there.
void decode_color_block(const uint8_t* blk, uint8_t out[16][4], bool dxt1_alpha) {
    uint16_t c0 = rd16(blk), c1 = rd16(blk + 2);
    uint8_t r[4], g[4], b[4], a[4] = {255, 255, 255, 255};
    unpack565(c0, r[0], g[0], b[0]);
    unpack565(c1, r[1], g[1], b[1]);
    if (c0 > c1 || !dxt1_alpha) {                 // 4-color mode (BC2/BC3 always take this path)
        r[2] = (uint8_t)((2 * r[0] + r[1]) / 3); g[2] = (uint8_t)((2 * g[0] + g[1]) / 3); b[2] = (uint8_t)((2 * b[0] + b[1]) / 3);
        r[3] = (uint8_t)((r[0] + 2 * r[1]) / 3); g[3] = (uint8_t)((g[0] + 2 * g[1]) / 3); b[3] = (uint8_t)((b[0] + 2 * b[1]) / 3);
    } else {                                      // 3-color + transparent (BC1 punch-through)
        r[2] = (uint8_t)((r[0] + r[1]) / 2); g[2] = (uint8_t)((g[0] + g[1]) / 2); b[2] = (uint8_t)((b[0] + b[1]) / 2);
        r[3] = g[3] = b[3] = 0; a[3] = 0;
    }
    uint32_t idx = (uint32_t)blk[4] | ((uint32_t)blk[5] << 8) | ((uint32_t)blk[6] << 16) | ((uint32_t)blk[7] << 24);
    for (int t = 0; t < 16; t++) {
        uint32_t i = (idx >> (t * 2)) & 0x3u;
        out[t][0] = r[i]; out[t][1] = g[i]; out[t][2] = b[i]; out[t][3] = a[i];
    }
}

// BC3 alpha sub-block (8 bytes): two 8-bit endpoints + 16 x 3-bit indices into an 8-entry ramp.
void decode_bc3_alpha(const uint8_t* blk, uint8_t out[16][4]) {
    uint8_t a0 = blk[0], a1 = blk[1];
    uint8_t a[8];
    a[0] = a0; a[1] = a1;
    if (a0 > a1) for (int i = 1; i <= 6; i++) a[i + 1] = (uint8_t)(((7 - i) * a0 + i * a1) / 7);
    else { for (int i = 1; i <= 4; i++) a[i + 1] = (uint8_t)(((5 - i) * a0 + i * a1) / 5); a[6] = 0; a[7] = 255; }
    uint64_t bits = 0; for (int i = 0; i < 6; i++) bits |= (uint64_t)blk[2 + i] << (i * 8);
    for (int t = 0; t < 16; t++) out[t][3] = a[(bits >> (t * 3)) & 0x7u];
}

// BC2 alpha sub-block (8 bytes): 16 x 4-bit direct alpha (bit-replicated to 8-bit).
void decode_bc2_alpha(const uint8_t* blk, uint8_t out[16][4]) {
    for (int t = 0; t < 16; t++) {
        uint8_t nib = (blk[t / 2] >> ((t & 1) * 4)) & 0xF;
        out[t][3] = (uint8_t)((nib << 4) | nib);
    }
}

// BC4 single-channel sub-block (8 bytes): identical structure to the BC3 alpha sub-block — two 8-bit
// endpoints + 16 x 3-bit indices into an 8-entry ramp (6 interpolants when e0>e1, else 4 interpolants
// + literal 0 and 255). Emits the 16 channel VALUES; the caller places them in R (BC4) or R/G (BC5).
void decode_bc4_channel(const uint8_t* blk, uint8_t vals[16]) {
    uint8_t a0 = blk[0], a1 = blk[1];
    uint8_t a[8];
    a[0] = a0; a[1] = a1;
    if (a0 > a1) for (int i = 1; i <= 6; i++) a[i + 1] = (uint8_t)(((7 - i) * a0 + i * a1) / 7);
    else { for (int i = 1; i <= 4; i++) a[i + 1] = (uint8_t)(((5 - i) * a0 + i * a1) / 5); a[6] = 0; a[7] = 255; }
    uint64_t bits = 0; for (int i = 0; i < 6; i++) bits |= (uint64_t)blk[2 + i] << (i * 8);
    for (int t = 0; t < 16; t++) vals[t] = a[(bits >> (t * 3)) & 0x7u];
}

// ---------------------------------------------------------------------------------------------------
// BC7 (BPTC_UNORM) — Khronos Data Format spec §BPTC / Microsoft D3D11 functional spec §19.5.
// Constant tables (mode field widths, partition/anchor assignments, interpolation weights) are the
// published spec constants, taken verbatim from bcdec (MIT/Unlicense, Sergii Kudlai) — the partition
// table entries carry the anchor ("fix-up") flag in bit 7 (128+subset), exactly as bcdec encodes them.
// Decode logic written against the spec and fuzz-validated byte-exact vs texture2ddecoder (see
// tests/test_bc_decode.cpp for the hand vectors; the random-block cross-check ran pre-merge).

// Per-mode endpoint component precision: [0] = RGB bits, [1] = alpha bits (0 = opaque mode).
static const uint8_t kBc7ColorBits[8] = { 4, 6, 5, 7, 5, 7, 7, 5 };
static const uint8_t kBc7AlphaBits[8] = { 0, 0, 0, 0, 6, 8, 7, 5 };
// Modes with a per-endpoint p-bit (0,3,6,7) — mode 1 has per-SUBSET shared p-bits, handled separately.
static const uint8_t kBc7ModeHasPBits = 0xCB;   // 0b11001011

// Partition/anchor table (spec constants; anchor texels flagged with +128, subset id in low bits).
static const uint8_t kBc7PartitionSets[2][64][4][4] = {
        {   /* Partition table for 2-subset BPTC */
            { {128, 0,   1, 1}, {0, 0,   1, 1}, {  0, 0, 1, 1}, {0, 0, 1, 129} }, /*  0 */
            { {128, 0,   0, 1}, {0, 0,   0, 1}, {  0, 0, 0, 1}, {0, 0, 0, 129} }, /*  1 */
            { {128, 1,   1, 1}, {0, 1,   1, 1}, {  0, 1, 1, 1}, {0, 1, 1, 129} }, /*  2 */
            { {128, 0,   0, 1}, {0, 0,   1, 1}, {  0, 0, 1, 1}, {0, 1, 1, 129} }, /*  3 */
            { {128, 0,   0, 0}, {0, 0,   0, 1}, {  0, 0, 0, 1}, {0, 0, 1, 129} }, /*  4 */
            { {128, 0,   1, 1}, {0, 1,   1, 1}, {  0, 1, 1, 1}, {1, 1, 1, 129} }, /*  5 */
            { {128, 0,   0, 1}, {0, 0,   1, 1}, {  0, 1, 1, 1}, {1, 1, 1, 129} }, /*  6 */
            { {128, 0,   0, 0}, {0, 0,   0, 1}, {  0, 0, 1, 1}, {0, 1, 1, 129} }, /*  7 */
            { {128, 0,   0, 0}, {0, 0,   0, 0}, {  0, 0, 0, 1}, {0, 0, 1, 129} }, /*  8 */
            { {128, 0,   1, 1}, {0, 1,   1, 1}, {  1, 1, 1, 1}, {1, 1, 1, 129} }, /*  9 */
            { {128, 0,   0, 0}, {0, 0,   0, 1}, {  0, 1, 1, 1}, {1, 1, 1, 129} }, /* 10 */
            { {128, 0,   0, 0}, {0, 0,   0, 0}, {  0, 0, 0, 1}, {0, 1, 1, 129} }, /* 11 */
            { {128, 0,   0, 1}, {0, 1,   1, 1}, {  1, 1, 1, 1}, {1, 1, 1, 129} }, /* 12 */
            { {128, 0,   0, 0}, {0, 0,   0, 0}, {  1, 1, 1, 1}, {1, 1, 1, 129} }, /* 13 */
            { {128, 0,   0, 0}, {1, 1,   1, 1}, {  1, 1, 1, 1}, {1, 1, 1, 129} }, /* 14 */
            { {128, 0,   0, 0}, {0, 0,   0, 0}, {  0, 0, 0, 0}, {1, 1, 1, 129} }, /* 15 */
            { {128, 0,   0, 0}, {1, 0,   0, 0}, {  1, 1, 1, 0}, {1, 1, 1, 129} }, /* 16 */
            { {128, 1, 129, 1}, {0, 0,   0, 1}, {  0, 0, 0, 0}, {0, 0, 0,   0} }, /* 17 */
            { {128, 0,   0, 0}, {0, 0,   0, 0}, {129, 0, 0, 0}, {1, 1, 1,   0} }, /* 18 */
            { {128, 1, 129, 1}, {0, 0,   1, 1}, {  0, 0, 0, 1}, {0, 0, 0,   0} }, /* 19 */
            { {128, 0, 129, 1}, {0, 0,   0, 1}, {  0, 0, 0, 0}, {0, 0, 0,   0} }, /* 20 */
            { {128, 0,   0, 0}, {1, 0,   0, 0}, {129, 1, 0, 0}, {1, 1, 1,   0} }, /* 21 */
            { {128, 0,   0, 0}, {0, 0,   0, 0}, {129, 0, 0, 0}, {1, 1, 0,   0} }, /* 22 */
            { {128, 1,   1, 1}, {0, 0,   1, 1}, {  0, 0, 1, 1}, {0, 0, 0, 129} }, /* 23 */
            { {128, 0, 129, 1}, {0, 0,   0, 1}, {  0, 0, 0, 1}, {0, 0, 0,   0} }, /* 24 */
            { {128, 0,   0, 0}, {1, 0,   0, 0}, {129, 0, 0, 0}, {1, 1, 0,   0} }, /* 25 */
            { {128, 1, 129, 0}, {0, 1,   1, 0}, {  0, 1, 1, 0}, {0, 1, 1,   0} }, /* 26 */
            { {128, 0, 129, 1}, {0, 1,   1, 0}, {  0, 1, 1, 0}, {1, 1, 0,   0} }, /* 27 */
            { {128, 0,   0, 1}, {0, 1,   1, 1}, {129, 1, 1, 0}, {1, 0, 0,   0} }, /* 28 */
            { {128, 0,   0, 0}, {1, 1,   1, 1}, {129, 1, 1, 1}, {0, 0, 0,   0} }, /* 29 */
            { {128, 1, 129, 1}, {0, 0,   0, 1}, {  1, 0, 0, 0}, {1, 1, 1,   0} }, /* 30 */
            { {128, 0, 129, 1}, {1, 0,   0, 1}, {  1, 0, 0, 1}, {1, 1, 0,   0} }, /* 31 */
            { {128, 1,   0, 1}, {0, 1,   0, 1}, {  0, 1, 0, 1}, {0, 1, 0, 129} }, /* 32 */
            { {128, 0,   0, 0}, {1, 1,   1, 1}, {  0, 0, 0, 0}, {1, 1, 1, 129} }, /* 33 */
            { {128, 1,   0, 1}, {1, 0, 129, 0}, {  0, 1, 0, 1}, {1, 0, 1,   0} }, /* 34 */
            { {128, 0,   1, 1}, {0, 0,   1, 1}, {129, 1, 0, 0}, {1, 1, 0,   0} }, /* 35 */
            { {128, 0, 129, 1}, {1, 1,   0, 0}, {  0, 0, 1, 1}, {1, 1, 0,   0} }, /* 36 */
            { {128, 1,   0, 1}, {0, 1,   0, 1}, {129, 0, 1, 0}, {1, 0, 1,   0} }, /* 37 */
            { {128, 1,   1, 0}, {1, 0,   0, 1}, {  0, 1, 1, 0}, {1, 0, 0, 129} }, /* 38 */
            { {128, 1,   0, 1}, {1, 0,   1, 0}, {  1, 0, 1, 0}, {0, 1, 0, 129} }, /* 39 */
            { {128, 1, 129, 1}, {0, 0,   1, 1}, {  1, 1, 0, 0}, {1, 1, 1,   0} }, /* 40 */
            { {128, 0,   0, 1}, {0, 0,   1, 1}, {129, 1, 0, 0}, {1, 0, 0,   0} }, /* 41 */
            { {128, 0, 129, 1}, {0, 0,   1, 0}, {  0, 1, 0, 0}, {1, 1, 0,   0} }, /* 42 */
            { {128, 0, 129, 1}, {1, 0,   1, 1}, {  1, 1, 0, 1}, {1, 1, 0,   0} }, /* 43 */
            { {128, 1, 129, 0}, {1, 0,   0, 1}, {  1, 0, 0, 1}, {0, 1, 1,   0} }, /* 44 */
            { {128, 0,   1, 1}, {1, 1,   0, 0}, {  1, 1, 0, 0}, {0, 0, 1, 129} }, /* 45 */
            { {128, 1,   1, 0}, {0, 1,   1, 0}, {  1, 0, 0, 1}, {1, 0, 0, 129} }, /* 46 */
            { {128, 0,   0, 0}, {0, 1, 129, 0}, {  0, 1, 1, 0}, {0, 0, 0,   0} }, /* 47 */
            { {128, 1,   0, 0}, {1, 1, 129, 0}, {  0, 1, 0, 0}, {0, 0, 0,   0} }, /* 48 */
            { {128, 0, 129, 0}, {0, 1,   1, 1}, {  0, 0, 1, 0}, {0, 0, 0,   0} }, /* 49 */
            { {128, 0,   0, 0}, {0, 0, 129, 0}, {  0, 1, 1, 1}, {0, 0, 1,   0} }, /* 50 */
            { {128, 0,   0, 0}, {0, 1,   0, 0}, {129, 1, 1, 0}, {0, 1, 0,   0} }, /* 51 */
            { {128, 1,   1, 0}, {1, 1,   0, 0}, {  1, 0, 0, 1}, {0, 0, 1, 129} }, /* 52 */
            { {128, 0,   1, 1}, {0, 1,   1, 0}, {  1, 1, 0, 0}, {1, 0, 0, 129} }, /* 53 */
            { {128, 1, 129, 0}, {0, 0,   1, 1}, {  1, 0, 0, 1}, {1, 1, 0,   0} }, /* 54 */
            { {128, 0, 129, 1}, {1, 0,   0, 1}, {  1, 1, 0, 0}, {0, 1, 1,   0} }, /* 55 */
            { {128, 1,   1, 0}, {1, 1,   0, 0}, {  1, 1, 0, 0}, {1, 0, 0, 129} }, /* 56 */
            { {128, 1,   1, 0}, {0, 0,   1, 1}, {  0, 0, 1, 1}, {1, 0, 0, 129} }, /* 57 */
            { {128, 1,   1, 1}, {1, 1,   1, 0}, {  1, 0, 0, 0}, {0, 0, 0, 129} }, /* 58 */
            { {128, 0,   0, 1}, {1, 0,   0, 0}, {  1, 1, 1, 0}, {0, 1, 1, 129} }, /* 59 */
            { {128, 0,   0, 0}, {1, 1,   1, 1}, {  0, 0, 1, 1}, {0, 0, 1, 129} }, /* 60 */
            { {128, 0, 129, 1}, {0, 0,   1, 1}, {  1, 1, 1, 1}, {0, 0, 0,   0} }, /* 61 */
            { {128, 0, 129, 0}, {0, 0,   1, 0}, {  1, 1, 1, 0}, {1, 1, 1,   0} }, /* 62 */
            { {128, 1,   0, 0}, {0, 1,   0, 0}, {  0, 1, 1, 1}, {0, 1, 1, 129} }  /* 63 */
        },
        {   /* Partition table for 3-subset BPTC */
            { {128, 0, 1, 129}, {0,   0,   1, 1}, {  0,   2,   2, 1}, {  2,   2, 2, 130} }, /*  0 */
            { {128, 0, 0, 129}, {0,   0,   1, 1}, {130,   2,   1, 1}, {  2,   2, 2,   1} }, /*  1 */
            { {128, 0, 0,   0}, {2,   0,   0, 1}, {130,   2,   1, 1}, {  2,   2, 1, 129} }, /*  2 */
            { {128, 2, 2, 130}, {0,   0,   2, 2}, {  0,   0,   1, 1}, {  0,   1, 1, 129} }, /*  3 */
            { {128, 0, 0,   0}, {0,   0,   0, 0}, {129,   1,   2, 2}, {  1,   1, 2, 130} }, /*  4 */
            { {128, 0, 1, 129}, {0,   0,   1, 1}, {  0,   0,   2, 2}, {  0,   0, 2, 130} }, /*  5 */
            { {128, 0, 2, 130}, {0,   0,   2, 2}, {  1,   1,   1, 1}, {  1,   1, 1, 129} }, /*  6 */
            { {128, 0, 1,   1}, {0,   0,   1, 1}, {130,   2,   1, 1}, {  2,   2, 1, 129} }, /*  7 */
            { {128, 0, 0,   0}, {0,   0,   0, 0}, {129,   1,   1, 1}, {  2,   2, 2, 130} }, /*  8 */
            { {128, 0, 0,   0}, {1,   1,   1, 1}, {129,   1,   1, 1}, {  2,   2, 2, 130} }, /*  9 */
            { {128, 0, 0,   0}, {1,   1, 129, 1}, {  2,   2,   2, 2}, {  2,   2, 2, 130} }, /* 10 */
            { {128, 0, 1,   2}, {0,   0, 129, 2}, {  0,   0,   1, 2}, {  0,   0, 1, 130} }, /* 11 */
            { {128, 1, 1,   2}, {0,   1, 129, 2}, {  0,   1,   1, 2}, {  0,   1, 1, 130} }, /* 12 */
            { {128, 1, 2,   2}, {0, 129,   2, 2}, {  0,   1,   2, 2}, {  0,   1, 2, 130} }, /* 13 */
            { {128, 0, 1, 129}, {0,   1,   1, 2}, {  1,   1,   2, 2}, {  1,   2, 2, 130} }, /* 14 */
            { {128, 0, 1, 129}, {2,   0,   0, 1}, {130,   2,   0, 0}, {  2,   2, 2,   0} }, /* 15 */
            { {128, 0, 0, 129}, {0,   0,   1, 1}, {  0,   1,   1, 2}, {  1,   1, 2, 130} }, /* 16 */
            { {128, 1, 1, 129}, {0,   0,   1, 1}, {130,   0,   0, 1}, {  2,   2, 0,   0} }, /* 17 */
            { {128, 0, 0,   0}, {1,   1,   2, 2}, {129,   1,   2, 2}, {  1,   1, 2, 130} }, /* 18 */
            { {128, 0, 2, 130}, {0,   0,   2, 2}, {  0,   0,   2, 2}, {  1,   1, 1, 129} }, /* 19 */
            { {128, 1, 1, 129}, {0,   1,   1, 1}, {  0,   2,   2, 2}, {  0,   2, 2, 130} }, /* 20 */
            { {128, 0, 0, 129}, {0,   0,   0, 1}, {130,   2,   2, 1}, {  2,   2, 2,   1} }, /* 21 */
            { {128, 0, 0,   0}, {0,   0, 129, 1}, {  0,   1,   2, 2}, {  0,   1, 2, 130} }, /* 22 */
            { {128, 0, 0,   0}, {1,   1,   0, 0}, {130,   2, 129, 0}, {  2,   2, 1,   0} }, /* 23 */
            { {128, 1, 2, 130}, {0, 129,   2, 2}, {  0,   0,   1, 1}, {  0,   0, 0,   0} }, /* 24 */
            { {128, 0, 1,   2}, {0,   0,   1, 2}, {129,   1,   2, 2}, {  2,   2, 2, 130} }, /* 25 */
            { {128, 1, 1,   0}, {1,   2, 130, 1}, {129,   2,   2, 1}, {  0,   1, 1,   0} }, /* 26 */
            { {128, 0, 0,   0}, {0,   1, 129, 0}, {  1,   2, 130, 1}, {  1,   2, 2,   1} }, /* 27 */
            { {128, 0, 2,   2}, {1,   1,   0, 2}, {129,   1,   0, 2}, {  0,   0, 2, 130} }, /* 28 */
            { {128, 1, 1,   0}, {0, 129,   1, 0}, {  2,   0,   0, 2}, {  2,   2, 2, 130} }, /* 29 */
            { {128, 0, 1,   1}, {0,   1,   2, 2}, {  0,   1, 130, 2}, {  0,   0, 1, 129} }, /* 30 */
            { {128, 0, 0,   0}, {2,   0,   0, 0}, {130,   2,   1, 1}, {  2,   2, 2, 129} }, /* 31 */
            { {128, 0, 0,   0}, {0,   0,   0, 2}, {129,   1,   2, 2}, {  1,   2, 2, 130} }, /* 32 */
            { {128, 2, 2, 130}, {0,   0,   2, 2}, {  0,   0,   1, 2}, {  0,   0, 1, 129} }, /* 33 */
            { {128, 0, 1, 129}, {0,   0,   1, 2}, {  0,   0,   2, 2}, {  0,   2, 2, 130} }, /* 34 */
            { {128, 1, 2,   0}, {0, 129,   2, 0}, {  0,   1, 130, 0}, {  0,   1, 2,   0} }, /* 35 */
            { {128, 0, 0,   0}, {1,   1, 129, 1}, {  2,   2, 130, 2}, {  0,   0, 0,   0} }, /* 36 */
            { {128, 1, 2,   0}, {1,   2,   0, 1}, {130,   0, 129, 2}, {  0,   1, 2,   0} }, /* 37 */
            { {128, 1, 2,   0}, {2,   0,   1, 2}, {129, 130,   0, 1}, {  0,   1, 2,   0} }, /* 38 */
            { {128, 0, 1,   1}, {2,   2,   0, 0}, {  1,   1, 130, 2}, {  0,   0, 1, 129} }, /* 39 */
            { {128, 0, 1,   1}, {1,   1, 130, 2}, {  2,   2,   0, 0}, {  0,   0, 1, 129} }, /* 40 */
            { {128, 1, 0, 129}, {0,   1,   0, 1}, {  2,   2,   2, 2}, {  2,   2, 2, 130} }, /* 41 */
            { {128, 0, 0,   0}, {0,   0,   0, 0}, {130,   1,   2, 1}, {  2,   1, 2, 129} }, /* 42 */
            { {128, 0, 2,   2}, {1, 129,   2, 2}, {  0,   0,   2, 2}, {  1,   1, 2, 130} }, /* 43 */
            { {128, 0, 2, 130}, {0,   0,   1, 1}, {  0,   0,   2, 2}, {  0,   0, 1, 129} }, /* 44 */
            { {128, 2, 2,   0}, {1,   2, 130, 1}, {  0,   2,   2, 0}, {  1,   2, 2, 129} }, /* 45 */
            { {128, 1, 0,   1}, {2,   2, 130, 2}, {  2,   2,   2, 2}, {  0,   1, 0, 129} }, /* 46 */
            { {128, 0, 0,   0}, {2,   1,   2, 1}, {130,   1,   2, 1}, {  2,   1, 2, 129} }, /* 47 */
            { {128, 1, 0, 129}, {0,   1,   0, 1}, {  0,   1,   0, 1}, {  2,   2, 2, 130} }, /* 48 */
            { {128, 2, 2, 130}, {0,   1,   1, 1}, {  0,   2,   2, 2}, {  0,   1, 1, 129} }, /* 49 */
            { {128, 0, 0,   2}, {1, 129,   1, 2}, {  0,   0,   0, 2}, {  1,   1, 1, 130} }, /* 50 */
            { {128, 0, 0,   0}, {2, 129,   1, 2}, {  2,   1,   1, 2}, {  2,   1, 1, 130} }, /* 51 */
            { {128, 2, 2,   2}, {0, 129,   1, 1}, {  0,   1,   1, 1}, {  0,   2, 2, 130} }, /* 52 */
            { {128, 0, 0,   2}, {1,   1,   1, 2}, {129,   1,   1, 2}, {  0,   0, 0, 130} }, /* 53 */
            { {128, 1, 1,   0}, {0, 129,   1, 0}, {  0,   1,   1, 0}, {  2,   2, 2, 130} }, /* 54 */
            { {128, 0, 0,   0}, {0,   0,   0, 0}, {  2,   1, 129, 2}, {  2,   1, 1, 130} }, /* 55 */
            { {128, 1, 1,   0}, {0, 129,   1, 0}, {  2,   2,   2, 2}, {  2,   2, 2, 130} }, /* 56 */
            { {128, 0, 2,   2}, {0,   0,   1, 1}, {  0,   0, 129, 1}, {  0,   0, 2, 130} }, /* 57 */
            { {128, 0, 2,   2}, {1,   1,   2, 2}, {129,   1,   2, 2}, {  0,   0, 2, 130} }, /* 58 */
            { {128, 0, 0,   0}, {0,   0,   0, 0}, {  0,   0,   0, 0}, {  2, 129, 1, 130} }, /* 59 */
            { {128, 0, 0, 130}, {0,   0,   0, 1}, {  0,   0,   0, 2}, {  0,   0, 0, 129} }, /* 60 */
            { {128, 2, 2,   2}, {1,   2,   2, 2}, {  0,   2,   2, 2}, {129,   2, 2, 130} }, /* 61 */
            { {128, 1, 0, 129}, {2,   2,   2, 2}, {  2,   2,   2, 2}, {  2,   2, 2, 130} }, /* 62 */
            { {128, 1, 1, 129}, {2,   0,   1, 1}, {130,   2,   0, 1}, {  2,   2, 2,   0} }  /* 63 */
        }
};

static const int kWeight2[4]  = { 0, 21, 43, 64 };
static const int kWeight3[8]  = { 0, 9, 18, 27, 37, 46, 55, 64 };
static const int kWeight4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

// Little-endian bit reader over the 128-bit BC7 block.
struct Bc7Bits {
    uint64_t lo, hi;
    uint32_t pos = 0;
    uint32_t get(uint32_t n) {           // n <= 8 for every BC7 field; fields never straddle bit 64...
        uint32_t v = 0;                  // ...actually they CAN straddle it, so read bit-by-bit (16 texel
        for (uint32_t i = 0; i < n; i++) //    fields x <=4 bits: negligible cost, zero edge cases).
            v |= get1() << i;
        return v;
    }
    uint32_t get1() {
        uint32_t b = (pos < 64) ? (uint32_t)((lo >> pos) & 1u) : (uint32_t)((hi >> (pos - 64)) & 1u);
        pos++;
        return b;
    }
};

inline uint8_t bc7_interp(uint32_t a, uint32_t b, int w) {
    return (uint8_t)((a * (64 - (uint32_t)w) + b * (uint32_t)w + 32) >> 6);
}

// Decode one 16-byte BC7 block into a 4x4 RGBA8 tile (row-major, 16 texels).
void decode_bc7_block(const uint8_t* blk, uint8_t out[16][4]) {
    Bc7Bits bs;
    std::memcpy(&bs.lo, blk, 8);
    std::memcpy(&bs.hi, blk + 8, 8);

    int mode = 0;
    while (mode < 8 && bs.get1() == 0) mode++;
    if (mode >= 8) {                           // reserved: all-zero mode byte -> transparent black
        std::memset(out, 0, 16 * 4);
        return;
    }

    int num_subsets = 1, partition = 0, rotation = 0, index_selection = 0;
    if (mode == 0 || mode == 2) num_subsets = 3;
    else if (mode == 1 || mode == 3 || mode == 7) num_subsets = 2;
    if (num_subsets > 1) partition = (int)bs.get(mode == 0 ? 4 : 6);
    if (mode == 4 || mode == 5) {
        rotation = (int)bs.get(2);
        if (mode == 4) index_selection = (int)bs.get1();
    }

    // Endpoints: all R, then all G, then all B, then all A (if the mode carries alpha).
    const int num_endpoints = num_subsets * 2;
    uint32_t ep[6][4] = {};
    for (int c = 0; c < 3; c++)
        for (int e = 0; e < num_endpoints; e++) ep[e][c] = bs.get(kBc7ColorBits[mode]);
    if (kBc7AlphaBits[mode])
        for (int e = 0; e < num_endpoints; e++) ep[e][3] = bs.get(kBc7AlphaBits[mode]);

    // P-bits: appended as the endpoint's LSB (all components share the bit).
    const bool per_ep_pbit = (kBc7ModeHasPBits >> mode) & 1;
    if (mode == 1) {                            // shared per-subset p-bits
        uint32_t p0 = bs.get1(), p1 = bs.get1();
        for (int e = 0; e < 4; e++)
            for (int c = 0; c < 3; c++) ep[e][c] = (ep[e][c] << 1) | (e < 2 ? p0 : p1);
    } else if (per_ep_pbit) {
        for (int e = 0; e < num_endpoints; e++) {
            uint32_t p = bs.get1();
            for (int c = 0; c < 4; c++) ep[e][c] = (ep[e][c] << 1) | p;
        }
    }

    // Expand each component to 8 bits (shift so the MSB lands at bit 7, replicate into the low bits).
    const uint32_t cb = (uint32_t)kBc7ColorBits[mode] + (per_ep_pbit || mode == 1 ? 1u : 0u);
    const uint32_t ab = kBc7AlphaBits[mode] ? (uint32_t)kBc7AlphaBits[mode] + (per_ep_pbit ? 1u : 0u) : 0u;
    for (int e = 0; e < num_endpoints; e++) {
        for (int c = 0; c < 3; c++) { ep[e][c] <<= (8 - cb); ep[e][c] |= ep[e][c] >> cb; }
        if (ab) { ep[e][3] <<= (8 - ab); ep[e][3] |= ep[e][3] >> ab; }
        else    ep[e][3] = 255;                 // opaque modes (0-3): alpha = 1.0
    }

    // Index widths: primary (color unless the mode-4 selection bit swaps) + optional secondary.
    const int ib  = (mode == 0 || mode == 1) ? 3 : (mode == 6 ? 4 : 2);
    const int ib2 = (mode == 4) ? 3 : (mode == 5 ? 2 : 0);
    const int* w1 = (ib == 2) ? kWeight2 : (ib == 3 ? kWeight3 : kWeight4);
    const int* w2 = (ib2 == 2) ? kWeight2 : kWeight3;

    // Subset id + anchor flag per texel (bit 7 marks the anchor texel, whose index has one less bit).
    uint8_t pset[16];
    for (int t = 0; t < 16; t++) {
        if (num_subsets == 1) pset[t] = (t == 0) ? 128 : 0;
        else pset[t] = kBc7PartitionSets[num_subsets - 2][partition][t >> 2][t & 3];
    }

    // Pass 1: the 16 primary indices (bitstream order; anchors read one less bit).
    uint8_t idx1[16];
    for (int t = 0; t < 16; t++) idx1[t] = (uint8_t)bs.get((uint32_t)ib - ((pset[t] & 0x80) ? 1 : 0));
    // Pass 2: secondary indices (modes 4/5), then interpolate + rotate.
    for (int t = 0; t < 16; t++) {
        const int s = (pset[t] & 3) * 2;
        uint32_t r, g, b, a;
        if (!ib2) {
            r = bc7_interp(ep[s][0], ep[s + 1][0], w1[idx1[t]]);
            g = bc7_interp(ep[s][1], ep[s + 1][1], w1[idx1[t]]);
            b = bc7_interp(ep[s][2], ep[s + 1][2], w1[idx1[t]]);
            a = bc7_interp(ep[s][3], ep[s + 1][3], w1[idx1[t]]);
        } else {
            uint32_t i2 = bs.get((uint32_t)ib2 - (t == 0 ? 1 : 0));   // single-subset: anchor = texel 0
            const int wc = index_selection ? w2[i2] : w1[idx1[t]];    // color weight
            const int wa = index_selection ? w1[idx1[t]] : w2[i2];    // alpha weight
            r = bc7_interp(ep[s][0], ep[s + 1][0], wc);
            g = bc7_interp(ep[s][1], ep[s + 1][1], wc);
            b = bc7_interp(ep[s][2], ep[s + 1][2], wc);
            a = bc7_interp(ep[s][3], ep[s + 1][3], wa);
        }
        switch (rotation) {                     // modes 4/5: swap alpha with the "scalar" channel
            case 1: { uint32_t x = a; a = r; r = x; } break;
            case 2: { uint32_t x = a; a = g; g = x; } break;
            case 3: { uint32_t x = a; a = b; b = x; } break;
        }
        out[t][0] = (uint8_t)r; out[t][1] = (uint8_t)g; out[t][2] = (uint8_t)b; out[t][3] = (uint8_t)a;
    }
}

} // namespace

bool bc_decode_surface(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                       uint32_t width, uint32_t height, DataFormat fmt) {
    const uint32_t bb = bc_block_bytes(fmt);
    if (fmt != DataFormat::Bc1 && fmt != DataFormat::Bc2 && fmt != DataFormat::Bc3 &&
        fmt != DataFormat::Bc4 && fmt != DataFormat::Bc5 && fmt != DataFormat::Bc7) return false;
    const uint32_t bw = (width + 3) / 4, bh = (height + 3) / 4;
    std::memset(dst, 0, (size_t)width * height * 4);
    for (uint32_t by = 0; by < bh; by++) {
        for (uint32_t bx = 0; bx < bw; bx++) {
            size_t boff = ((size_t)by * bw + bx) * bb;
            if (boff + bb > src_bytes) continue;             // short source -> leave block transparent
            const uint8_t* blk = src + boff;
            uint8_t texels[16][4];
            if (fmt == DataFormat::Bc1) {
                decode_color_block(blk, texels, /*dxt1_alpha*/true);
            } else if (fmt == DataFormat::Bc4) {
                // Single-channel UNORM: hardware returns (R, 0, 0, 1); the T# DST_SEL swizzle (applied
                // downstream, #261) routes it into whatever channels the material reads.
                uint8_t v[16]; decode_bc4_channel(blk, v);
                for (int t = 0; t < 16; t++) { texels[t][0] = v[t]; texels[t][1] = 0; texels[t][2] = 0; texels[t][3] = 255; }
            } else if (fmt == DataFormat::Bc5) {
                // Two-channel UNORM (two BC4 sub-blocks): hardware returns (R, G, 0, 1).
                uint8_t r[16], g[16]; decode_bc4_channel(blk, r); decode_bc4_channel(blk + 8, g);
                for (int t = 0; t < 16; t++) { texels[t][0] = r[t]; texels[t][1] = g[t]; texels[t][2] = 0; texels[t][3] = 255; }
            } else if (fmt == DataFormat::Bc7) {
                decode_bc7_block(blk, texels);
            } else {                                          // BC2/BC3: alpha sub-block then color sub-block
                decode_color_block(blk + 8, texels, /*dxt1_alpha*/false);
                if (fmt == DataFormat::Bc3) decode_bc3_alpha(blk, texels);
                else                        decode_bc2_alpha(blk, texels);
            }
            for (int ty = 0; ty < 4; ty++) {
                uint32_t y = by * 4 + ty; if (y >= height) break;
                for (int tx = 0; tx < 4; tx++) {
                    uint32_t x = bx * 4 + tx; if (x >= width) break;
                    std::memcpy(dst + ((size_t)y * width + x) * 4, texels[ty * 4 + tx], 4);
                }
            }
        }
    }
    return true;
}

} // namespace prosper::gpu
