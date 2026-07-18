// bc_decode.cpp — see bc_decode.hpp. Reference S3TC/DXTn decode (Khronos data-format spec §18-19).
#include "bc_decode.hpp"
#include <bit>
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
// Decode logic written against the spec; checked-in known blocks cover every mode.

// Per-mode endpoint component precision: [0] = RGB bits, [1] = alpha bits (0 = opaque mode).
static const uint8_t kBc7ColorBits[8] = { 4, 6, 5, 7, 5, 7, 7, 5 };
static const uint8_t kBc7AlphaBits[8] = { 0, 0, 0, 0, 6, 8, 7, 5 };
// Modes with a per-endpoint p-bit (0,3,6,7). Mode 1's per-subset shared p-bits are handled separately.
static const uint8_t kBc7ModeHasEndpointPBits = 0xC9;   // 0b11001001

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
    const bool per_ep_pbit = (kBc7ModeHasEndpointPBits >> mode) & 1;
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

// ---------------------------------------------------------------------------------------------------
// BC6H (BPTC_FLOAT) — Khronos Data Format spec / Microsoft D3D11 functional spec paragraph 19.6.
// Decode logic and constants adapted VERBATIM from bcdec (MIT/Unlicense, Sergii Kudlai) — the same
// reference the BC7 tables above credit — because the 14 mode bit-layouts are heavily scrambled and
// a hand transcription would be error-prone. Output here is a 4x4 tile of RGB half-floats which
// decode_bc6h_block converts to RGBA8 (clamp01 * 255): the upload path is 8-bit, so HDR energy
// above 1.0 clamps — same policy as the fp16-surface path (#290). UNSIGNED (UF16) only for now;
// the SF16 variant stays skipped upstream (no live example).
typedef struct bc6h__bitstream {
    unsigned long long low;
    unsigned long long high;
} bc6h__bitstream_t;

static int bc6h__bitstream_read_bits(bc6h__bitstream_t* bstream, int numBits) {
    unsigned int mask = (1 << numBits) - 1;
    /* Read the low N bits */
    unsigned int bits = (bstream->low & mask);

    bstream->low >>= numBits;
    /* Put the low N bits of "high" into the high 64-N bits of "low". */
    bstream->low |= (bstream->high & mask) << (sizeof(bstream->high) * 8 - numBits);
    bstream->high >>= numBits;
    
    return bits;
}

static int bc6h__bitstream_read_bit(bc6h__bitstream_t* bstream) {
    return bc6h__bitstream_read_bits(bstream, 1);
}

/*  reversed bits pulling, used in BC6H decoding
    why ?? just why ??? */
static int bc6h__bitstream_read_bits_r(bc6h__bitstream_t* bstream, int numBits) {
    int bits = bc6h__bitstream_read_bits(bstream, numBits);
    /* Reverse the bits. */
    int result = 0;
    while (numBits--) {
        result <<= 1;
        result |= (bits & 1);
        bits >>= 1;
    }
    return result;
}




static int bc6h__extend_sign(int val, int bits) {
    // `val` is an unsigned `bits`-wide field. The usual signed-shift idiom has undefined behavior
    // when the left shift overflows `int`; subtracting the modulus expresses the same two's-complement
    // sign extension without depending on compiler behavior.
    const int sign = 1 << (bits - 1);
    return (val & sign) ? val - (1 << bits) : val;
}

static int bc6h__transform_inverse(int val, int a0, int bits, int isSigned) {
    /* If the precision of A0 is "p" bits, then the transform algorithm is:
       B0 = (B0 + A0) & ((1 << p) - 1) */
    val = (val + a0) & ((1 << bits) - 1);
    if (isSigned) {
        val = bc6h__extend_sign(val, bits);
    }
    return val;
}

/* pretty much copy-paste from documentation */
static int bc6h__unquantize(int val, int bits, int isSigned) {
    int unq, s = 0;

    if (!isSigned) {
        if (bits >= 15) {
            unq = val;
        } else if (!val) {
            unq = 0;
        } else if (val == ((1 << bits) - 1)) {
            unq = 0xFFFF;
        } else {
            unq = ((val << 16) + 0x8000) >> bits;
        }
    } else {
        if (bits >= 16) {
            unq = val;
        } else {
            if (val < 0) {
                s = 1;
                val = -val;
            }

            if (val == 0) {
                unq = 0;
            } else if (val >= ((1 << (bits - 1)) - 1)) {
                unq = 0x7FFF;
            } else {
                unq = ((val << 15) + 0x4000) >> (bits - 1);
            }

            if (s) {
                unq = -unq;
            }
        }
    }
    return unq;
}

static int bc6h__interpolate(int a, int b, int* weights, int index) {
    return (a * (64 - weights[index]) + b * weights[index] + 32) >> 6;
}

static unsigned short bc6h__finish_unquantize(int val, int isSigned) {
    int s;

    if (!isSigned) {
        return (unsigned short)((val * 31) >> 6);                   /* scale the magnitude by 31 / 64 */
    } else {
        val = (val < 0) ? -(((-val) * 31) >> 5) : (val * 31) >> 5;  /* scale the magnitude by 31 / 32 */
        s = 0;
        if (val < 0) {
            s = 0x8000;
            val = -val;
        }
        return (unsigned short)(s | val);
    }
}

/* modified half_to_float_fast4 from https://gist.github.com/rygorous/2144712 */
static float bc6h__half_to_float_quick(unsigned short half) {
    constexpr uint32_t shifted_exp = 0x7c00u << 13;         /* exponent mask after shift */
    constexpr float magic = std::bit_cast<float>(113u << 23);
    uint32_t bits = (static_cast<uint32_t>(half) & 0x7fffu) << 13; /* exponent/mantissa bits */
    const uint32_t exp = shifted_exp & bits;                 /* just the exponent */
    bits += (127u - 15u) << 23;                              /* exponent adjust */

    /* handle exponent special cases */
    if (exp == shifted_exp) {                               /* Inf/NaN? */
        bits += (128u - 16u) << 23;                         /* extra exp adjust */
    } else if (exp == 0) {                                  /* Zero/Denormal? */
        bits += 1u << 23;                                   /* extra exp adjust */
        bits = std::bit_cast<uint32_t>(std::bit_cast<float>(bits) - magic); /* renormalize */
    }

    bits |= (static_cast<uint32_t>(half) & 0x8000u) << 16;  /* sign bit */
    return std::bit_cast<float>(bits);
}

static void bc6h__decode_block_half(const void* compressedBlock, void* decompressedBlock, int destinationPitch, int isSigned) {
    static char actual_bits_count[4][14] = {
        { 10, 7, 11, 11, 11, 9, 8, 8, 8, 6, 10, 11, 12, 16 },   /*  W */
        {  5, 6,  5,  4,  4, 5, 6, 5, 5, 6, 10,  9,  8,  4 },   /* dR */
        {  5, 6,  4,  5,  4, 5, 5, 6, 5, 6, 10,  9,  8,  4 },   /* dG */
        {  5, 6,  4,  4,  5, 5, 5, 5, 6, 6, 10,  9,  8,  4 }    /* dB */
    };

    /* There are 32 possible partition sets for a two-region tile.
       Each 4x4 block represents a single shape.
       Here also every fix-up index has MSB bit set. */
    static unsigned char partition_sets[32][4][4] = {
        { {128, 0,   1, 1}, {0, 0, 1, 1}, {  0, 0, 1, 1}, {0, 0, 1, 129} },   /*  0 */
        { {128, 0,   0, 1}, {0, 0, 0, 1}, {  0, 0, 0, 1}, {0, 0, 0, 129} },   /*  1 */
        { {128, 1,   1, 1}, {0, 1, 1, 1}, {  0, 1, 1, 1}, {0, 1, 1, 129} },   /*  2 */
        { {128, 0,   0, 1}, {0, 0, 1, 1}, {  0, 0, 1, 1}, {0, 1, 1, 129} },   /*  3 */
        { {128, 0,   0, 0}, {0, 0, 0, 1}, {  0, 0, 0, 1}, {0, 0, 1, 129} },   /*  4 */
        { {128, 0,   1, 1}, {0, 1, 1, 1}, {  0, 1, 1, 1}, {1, 1, 1, 129} },   /*  5 */
        { {128, 0,   0, 1}, {0, 0, 1, 1}, {  0, 1, 1, 1}, {1, 1, 1, 129} },   /*  6 */
        { {128, 0,   0, 0}, {0, 0, 0, 1}, {  0, 0, 1, 1}, {0, 1, 1, 129} },   /*  7 */
        { {128, 0,   0, 0}, {0, 0, 0, 0}, {  0, 0, 0, 1}, {0, 0, 1, 129} },   /*  8 */
        { {128, 0,   1, 1}, {0, 1, 1, 1}, {  1, 1, 1, 1}, {1, 1, 1, 129} },   /*  9 */
        { {128, 0,   0, 0}, {0, 0, 0, 1}, {  0, 1, 1, 1}, {1, 1, 1, 129} },   /* 10 */
        { {128, 0,   0, 0}, {0, 0, 0, 0}, {  0, 0, 0, 1}, {0, 1, 1, 129} },   /* 11 */
        { {128, 0,   0, 1}, {0, 1, 1, 1}, {  1, 1, 1, 1}, {1, 1, 1, 129} },   /* 12 */
        { {128, 0,   0, 0}, {0, 0, 0, 0}, {  1, 1, 1, 1}, {1, 1, 1, 129} },   /* 13 */
        { {128, 0,   0, 0}, {1, 1, 1, 1}, {  1, 1, 1, 1}, {1, 1, 1, 129} },   /* 14 */
        { {128, 0,   0, 0}, {0, 0, 0, 0}, {  0, 0, 0, 0}, {1, 1, 1, 129} },   /* 15 */
        { {128, 0,   0, 0}, {1, 0, 0, 0}, {  1, 1, 1, 0}, {1, 1, 1, 129} },   /* 16 */
        { {128, 1, 129, 1}, {0, 0, 0, 1}, {  0, 0, 0, 0}, {0, 0, 0,   0} },   /* 17 */
        { {128, 0,   0, 0}, {0, 0, 0, 0}, {129, 0, 0, 0}, {1, 1, 1,   0} },   /* 18 */
        { {128, 1, 129, 1}, {0, 0, 1, 1}, {  0, 0, 0, 1}, {0, 0, 0,   0} },   /* 19 */
        { {128, 0, 129, 1}, {0, 0, 0, 1}, {  0, 0, 0, 0}, {0, 0, 0,   0} },   /* 20 */
        { {128, 0,   0, 0}, {1, 0, 0, 0}, {129, 1, 0, 0}, {1, 1, 1,   0} },   /* 21 */
        { {128, 0,   0, 0}, {0, 0, 0, 0}, {129, 0, 0, 0}, {1, 1, 0,   0} },   /* 22 */
        { {128, 1,   1, 1}, {0, 0, 1, 1}, {  0, 0, 1, 1}, {0, 0, 0, 129} },   /* 23 */
        { {128, 0, 129, 1}, {0, 0, 0, 1}, {  0, 0, 0, 1}, {0, 0, 0,   0} },   /* 24 */
        { {128, 0,   0, 0}, {1, 0, 0, 0}, {129, 0, 0, 0}, {1, 1, 0,   0} },   /* 25 */
        { {128, 1, 129, 0}, {0, 1, 1, 0}, {  0, 1, 1, 0}, {0, 1, 1,   0} },   /* 26 */
        { {128, 0, 129, 1}, {0, 1, 1, 0}, {  0, 1, 1, 0}, {1, 1, 0,   0} },   /* 27 */
        { {128, 0,   0, 1}, {0, 1, 1, 1}, {129, 1, 1, 0}, {1, 0, 0,   0} },   /* 28 */
        { {128, 0,   0, 0}, {1, 1, 1, 1}, {129, 1, 1, 1}, {0, 0, 0,   0} },   /* 29 */
        { {128, 1, 129, 1}, {0, 0, 0, 1}, {  1, 0, 0, 0}, {1, 1, 1,   0} },   /* 30 */
        { {128, 0, 129, 1}, {1, 0, 0, 1}, {  1, 0, 0, 1}, {1, 1, 0,   0} }    /* 31 */
    };

    static int aWeight3[8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
    static int aWeight4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

    bc6h__bitstream_t bstream;
    int mode, partition, numPartitions, i, j, partitionSet, indexBits, index, ep_i, actualBits0Mode;
    int r[4], g[4], b[4];       /* wxyz */
    unsigned short* decompressed;
    int* weights;

    decompressed = (unsigned short*)decompressedBlock;

    const auto* compressed_bytes = static_cast<const uint8_t*>(compressedBlock);
    std::memcpy(&bstream.low, compressed_bytes, sizeof bstream.low);
    std::memcpy(&bstream.high, compressed_bytes + sizeof bstream.low, sizeof bstream.high);

    r[0] = r[1] = r[2] = r[3] = 0;
    g[0] = g[1] = g[2] = g[3] = 0;
    b[0] = b[1] = b[2] = b[3] = 0;

    mode = bc6h__bitstream_read_bits(&bstream, 2);
    if (mode > 1) {
        mode |= (bc6h__bitstream_read_bits(&bstream, 3) << 2);
    }

    /* modes >= 11 (10 in my code) are using 0 one, others will read it from the bitstream */
    partition = 0;

    switch (mode) {
        /* mode 1 */
        case 0b00: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 75 bits (10.555, 10.555, 10.555) */
            g[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gy[4]   */
            b[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* by[4]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* bz[4]   */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* rw[9:0] */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* gw[9:0] */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* bw[9:0] */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* rx[4:0] */
            g[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gz[4]   */
            g[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gy[3:0] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* gx[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream);            /* bz[0]   */
            g[3] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gz[3:0] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* bx[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 1;       /* bz[1]   */
            b[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* by[3:0] */
            r[2] |= bc6h__bitstream_read_bits(&bstream, 5);        /* ry[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 2;       /* bz[2]   */
            r[3] |= bc6h__bitstream_read_bits(&bstream, 5);        /* rz[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 3;       /* bz[3]   */
            partition = bc6h__bitstream_read_bits(&bstream, 5);    /* d[4:0]  */
            mode = 0;
        } break;

        /* mode 2 */
        case 0b01: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 75 bits (7666, 7666, 7666) */
            g[2] |= bc6h__bitstream_read_bit(&bstream) << 5;       /* gy[5]   */
            g[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gz[4]   */
            g[3] |= bc6h__bitstream_read_bit(&bstream) << 5;       /* gz[5]   */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 7);        /* rw[6:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream);            /* bz[0]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 1;       /* bz[1]   */
            b[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* by[4]   */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 7);        /* gw[6:0] */
            b[2] |= bc6h__bitstream_read_bit(&bstream) << 5;       /* by[5]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 2;       /* bz[2]   */
            g[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gy[4]   */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 7);        /* bw[6:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 3;       /* bz[3]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 5;       /* bz[5]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* bz[4]   */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 6);        /* rx[5:0] */
            g[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gy[3:0] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 6);        /* gx[5:0] */
            g[3] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gz[3:0] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 6);        /* bx[5:0] */
            b[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* by[3:0] */
            r[2] |= bc6h__bitstream_read_bits(&bstream, 6);        /* ry[5:0] */
            r[3] |= bc6h__bitstream_read_bits(&bstream, 6);        /* rz[5:0] */
            partition = bc6h__bitstream_read_bits(&bstream, 5);    /* d[4:0]  */
            mode = 1;
        } break;

        /* mode 3 */
        case 0b00010: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (11.555, 11.444, 11.444) */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* rw[9:0] */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* gw[9:0] */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* bw[9:0] */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* rx[4:0] */
            r[0] |= bc6h__bitstream_read_bit(&bstream) << 10;      /* rw[10]  */
            g[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gy[3:0] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gx[3:0] */
            g[0] |= bc6h__bitstream_read_bit(&bstream) << 10;      /* gw[10]  */
            b[3] |= bc6h__bitstream_read_bit(&bstream);            /* bz[0]   */
            g[3] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gz[3:0] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 4);        /* bx[3:0] */
            b[0] |= bc6h__bitstream_read_bit(&bstream) << 10;      /* bw[10]  */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 1;       /* bz[1]   */
            b[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* by[3:0] */
            r[2] |= bc6h__bitstream_read_bits(&bstream, 5);        /* ry[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 2;       /* bz[2]   */
            r[3] |= bc6h__bitstream_read_bits(&bstream, 5);        /* rz[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 3;       /* bz[3]   */
            partition = bc6h__bitstream_read_bits(&bstream, 5);    /* d[4:0]  */
            mode = 2;
        } break;

        /* mode 4 */
        case 0b00110: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (11.444, 11.555, 11.444) */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* rw[9:0] */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* gw[9:0] */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* bw[9:0] */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 4);        /* rx[3:0] */
            r[0] |= bc6h__bitstream_read_bit(&bstream) << 10;      /* rw[10]  */
            g[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gz[4]   */
            g[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gy[3:0] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* gx[4:0] */
            g[0] |= bc6h__bitstream_read_bit(&bstream) << 10;      /* gw[10]  */
            g[3] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gz[3:0] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 4);        /* bx[3:0] */
            b[0] |= bc6h__bitstream_read_bit(&bstream) << 10;      /* bw[10]  */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 1;       /* bz[1]   */
            b[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* by[3:0] */
            r[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* ry[3:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream);            /* bz[0]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 2;       /* bz[2]   */
            r[3] |= bc6h__bitstream_read_bits(&bstream, 4);        /* rz[3:0] */
            g[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gy[4]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 3;       /* bz[3]   */
            partition = bc6h__bitstream_read_bits(&bstream, 5);    /* d[4:0]  */
            mode = 3;
        } break;

        /* mode 5 */
        case 0b01010: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (11.444, 11.444, 11.555) */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* rw[9:0] */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* gw[9:0] */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* bw[9:0] */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 4);        /* rx[3:0] */
            r[0] |= bc6h__bitstream_read_bit(&bstream) << 10;      /* rw[10]  */
            b[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* by[4]   */
            g[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gy[3:0] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gx[3:0] */
            g[0] |= bc6h__bitstream_read_bit(&bstream) << 10;      /* gw[10]  */
            b[3] |= bc6h__bitstream_read_bit(&bstream);            /* bz[0]   */
            g[3] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gz[3:0] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* bx[4:0] */
            b[0] |= bc6h__bitstream_read_bit(&bstream) << 10;      /* bw[10]  */
            b[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* by[3:0] */
            r[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* ry[3:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 1;       /* bz[1]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 2;       /* bz[2]   */
            r[3] |= bc6h__bitstream_read_bits(&bstream, 4);        /* rz[3:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* bz[4]   */ 
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 3;       /* bz[3]   */
            partition = bc6h__bitstream_read_bits(&bstream, 5);    /* d[4:0]  */
            mode = 4;
        } break;

        /* mode 6 */
        case 0b01110: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (9555, 9555, 9555) */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 9);        /* rw[8:0] */
            b[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* by[4]   */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 9);        /* gw[8:0] */
            g[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gy[4]   */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 9);        /* bw[8:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* bz[4]   */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* rx[4:0] */
            g[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gz[4]   */
            g[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gy[3:0] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* gx[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream);            /* bz[0]   */
            g[3] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gx[3:0] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* bx[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 1;       /* bz[1]   */
            b[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* by[3:0] */
            r[2] |= bc6h__bitstream_read_bits(&bstream, 5);        /* ry[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 2;       /* bz[2]   */
            r[3] |= bc6h__bitstream_read_bits(&bstream, 5);        /* rz[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 3;       /* bz[3]   */
            partition = bc6h__bitstream_read_bits(&bstream, 5);    /* d[4:0]  */
            mode = 5;
        } break;

        /* mode 7 */
        case 0b10010: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (8666, 8555, 8555) */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 8);        /* rw[7:0] */
            g[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gz[4]   */
            b[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* by[4]   */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 8);        /* gw[7:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 2;       /* bz[2]   */
            g[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gy[4]   */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 8);        /* bw[7:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 3;       /* bz[3]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* bz[4]   */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 6);        /* rx[5:0] */
            g[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gy[3:0] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* gx[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream);            /* bz[0]   */
            g[3] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gz[3:0] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* bx[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 1;       /* bz[1]   */
            b[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* by[3:0] */
            r[2] |= bc6h__bitstream_read_bits(&bstream, 6);        /* ry[5:0] */
            r[3] |= bc6h__bitstream_read_bits(&bstream, 6);        /* rz[5:0] */
            partition = bc6h__bitstream_read_bits(&bstream, 5);    /* d[4:0]  */
            mode = 6;
        } break;

        /* mode 8 */
        case 0b10110: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (8555, 8666, 8555) */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 8);        /* rw[7:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream);            /* bz[0]   */
            b[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* by[4]   */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 8);        /* gw[7:0] */
            g[2] |= bc6h__bitstream_read_bit(&bstream) << 5;       /* gy[5]   */
            g[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gy[4]   */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 8);        /* bw[7:0] */
            g[3] |= bc6h__bitstream_read_bit(&bstream) << 5;       /* gz[5]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* bz[4]   */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* rx[4:0] */
            g[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gz[4]   */
            g[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gy[3:0] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 6);        /* gx[5:0] */
            g[3] |= bc6h__bitstream_read_bits(&bstream, 4);        /* zx[3:0] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* bx[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 1;       /* bz[1]   */
            b[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* by[3:0] */
            r[2] |= bc6h__bitstream_read_bits(&bstream, 5);        /* ry[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 2;       /* bz[2]   */
            r[3] |= bc6h__bitstream_read_bits(&bstream, 5);        /* rz[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 3;       /* bz[3]   */
            partition = bc6h__bitstream_read_bits(&bstream, 5);    /* d[4:0]  */
            mode = 7;
        } break;

        /* mode 9 */
        case 0b11010: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (8555, 8555, 8666) */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 8);        /* rw[7:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 1;       /* bz[1]   */
            b[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* by[4]   */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 8);        /* gw[7:0] */
            b[2] |= bc6h__bitstream_read_bit(&bstream) << 5;       /* by[5]   */
            g[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gy[4]   */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 8);        /* bw[7:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 5;       /* bz[5]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* bz[4]   */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* bw[4:0] */
            g[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gz[4]   */
            g[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gy[3:0] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 5);        /* gx[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream);            /* bz[0]   */
            g[3] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gz[3:0] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 6);        /* bx[5:0] */
            b[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* by[3:0] */
            r[2] |= bc6h__bitstream_read_bits(&bstream, 5);        /* ry[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 2;       /* bz[2]   */
            r[3] |= bc6h__bitstream_read_bits(&bstream, 5);        /* rz[4:0] */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 3;       /* bz[3]   */
            partition = bc6h__bitstream_read_bits(&bstream, 5);    /* d[4:0]  */
            mode = 8;
        } break;

        /* mode 10 */
        case 0b11110: {
            /* Partitition indices: 46 bits
               Partition: 5 bits
               Color Endpoints: 72 bits (6666, 6666, 6666) */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 6);        /* rw[5:0] */
            g[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gz[4]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream);            /* bz[0]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 1;       /* bz[1]   */
            b[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* by[4]   */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 6);        /* gw[5:0] */
            g[2] |= bc6h__bitstream_read_bit(&bstream) << 5;       /* gy[5]   */
            b[2] |= bc6h__bitstream_read_bit(&bstream) << 5;       /* by[5]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 2;       /* bz[2]   */
            g[2] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* gy[4]   */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 6);        /* bw[5:0] */
            g[3] |= bc6h__bitstream_read_bit(&bstream) << 5;       /* gz[5]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 3;       /* bz[3]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 5;       /* bz[5]   */
            b[3] |= bc6h__bitstream_read_bit(&bstream) << 4;       /* bz[4]   */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 6);        /* rx[5:0] */
            g[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gy[3:0] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 6);        /* gx[5:0] */
            g[3] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gz[3:0] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 6);        /* bx[5:0] */
            b[2] |= bc6h__bitstream_read_bits(&bstream, 4);        /* by[3:0] */
            r[2] |= bc6h__bitstream_read_bits(&bstream, 6);        /* ry[5:0] */
            r[3] |= bc6h__bitstream_read_bits(&bstream, 6);        /* rz[5:0] */
            partition = bc6h__bitstream_read_bits(&bstream, 5);    /* d[4:0]  */
            mode = 9;
        } break;

        /* mode 11 */
        case 0b00011: {
            /* Partitition indices: 63 bits
               Partition: 0 bits
               Color Endpoints: 60 bits (10.10, 10.10, 10.10) */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* rw[9:0] */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* gw[9:0] */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* bw[9:0] */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 10);       /* rx[9:0] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 10);       /* gx[9:0] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 10);       /* bx[9:0] */
            mode = 10;
        } break;

        /* mode 12 */
        case 0b00111: {
            /* Partitition indices: 63 bits
               Partition: 0 bits
               Color Endpoints: 60 bits (11.9, 11.9, 11.9) */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* rw[9:0] */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* gw[9:0] */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* bw[9:0] */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 9);        /* rx[8:0] */
            r[0] |= bc6h__bitstream_read_bit(&bstream) << 10;      /* rw[10]  */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 9);        /* gx[8:0] */
            g[0] |= bc6h__bitstream_read_bit(&bstream) << 10;      /* gw[10]  */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 9);        /* bx[8:0] */
            b[0] |= bc6h__bitstream_read_bit(&bstream) << 10;      /* bw[10]  */
            mode = 11;
        } break;

        /* mode 13 */
        case 0b01011: {
            /* Partitition indices: 63 bits
               Partition: 0 bits
               Color Endpoints: 60 bits (12.8, 12.8, 12.8) */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* rw[9:0] */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* gw[9:0] */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* bw[9:0] */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 8);        /* rx[7:0] */
            r[0] |= bc6h__bitstream_read_bits_r(&bstream, 2) << 10;/* rx[10:11] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 8);        /* gx[7:0] */
            g[0] |= bc6h__bitstream_read_bits_r(&bstream, 2) << 10;/* gx[10:11] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 8);        /* bx[7:0] */
            b[0] |= bc6h__bitstream_read_bits_r(&bstream, 2) << 10;/* bx[10:11] */
            mode = 12;
        } break;

        /* mode 14 */
        case 0b01111: {
            /* Partitition indices: 63 bits
               Partition: 0 bits
               Color Endpoints: 60 bits (16.4, 16.4, 16.4) */
            r[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* rw[9:0] */
            g[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* gw[9:0] */
            b[0] |= bc6h__bitstream_read_bits(&bstream, 10);       /* bw[9:0] */
            r[1] |= bc6h__bitstream_read_bits(&bstream, 4);        /* rx[3:0] */
            r[0] |= bc6h__bitstream_read_bits_r(&bstream, 6) << 10;/* rw[10:15] */
            g[1] |= bc6h__bitstream_read_bits(&bstream, 4);        /* gx[3:0] */
            g[0] |= bc6h__bitstream_read_bits_r(&bstream, 6) << 10;/* gw[10:15] */
            b[1] |= bc6h__bitstream_read_bits(&bstream, 4);        /* bx[3:0] */
            b[0] |= bc6h__bitstream_read_bits_r(&bstream, 6) << 10;/* bw[10:15] */
            mode = 13;
        } break;

        default: {
            /* Modes 10011, 10111, 11011, and 11111 (not shown) are reserved.
               Do not use these in your encoder. If the hardware is passed blocks
               with one of these modes specified, the resulting decompressed block
               must contain all zeroes in all channels except for the alpha channel. */
            for (i = 0; i < 4; ++i) {
                for (j = 0; j < 4; ++j) {
                    decompressed[j * 3 + 0] = 0;
                    decompressed[j * 3 + 1] = 0;
                    decompressed[j * 3 + 2] = 0;
                }
                decompressed += destinationPitch;
            }

            return;
        }
    }

    numPartitions = (mode >= 10) ? 0 : 1;

    actualBits0Mode = actual_bits_count[0][mode];
    if (isSigned) {
        r[0] = bc6h__extend_sign(r[0], actualBits0Mode);
        g[0] = bc6h__extend_sign(g[0], actualBits0Mode);
        b[0] = bc6h__extend_sign(b[0], actualBits0Mode);
    }

    /* Mode 11 (like Mode 10) does not use delta compression,
       and instead stores both color endpoints explicitly.  */
    if ((mode != 9 && mode != 10) || isSigned) {
        for (i = 1; i < (numPartitions + 1) * 2; ++i) {
            r[i] = bc6h__extend_sign(r[i], actual_bits_count[1][mode]);
            g[i] = bc6h__extend_sign(g[i], actual_bits_count[2][mode]);
            b[i] = bc6h__extend_sign(b[i], actual_bits_count[3][mode]);
        }
    }

    if (mode != 9 && mode != 10) {
        for (i = 1; i < (numPartitions + 1) * 2; ++i) {
            r[i] = bc6h__transform_inverse(r[i], r[0], actualBits0Mode, isSigned);
            g[i] = bc6h__transform_inverse(g[i], g[0], actualBits0Mode, isSigned);
            b[i] = bc6h__transform_inverse(b[i], b[0], actualBits0Mode, isSigned);
        }
    }

    for (i = 0; i < (numPartitions + 1) * 2; ++i) {
        r[i] = bc6h__unquantize(r[i], actualBits0Mode, isSigned);
        g[i] = bc6h__unquantize(g[i], actualBits0Mode, isSigned);
        b[i] = bc6h__unquantize(b[i], actualBits0Mode, isSigned);
    }

    weights = (mode >= 10) ? aWeight4 : aWeight3;
    for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) {
            partitionSet = (mode >= 10) ? ((i|j) ? 0 : 128) : partition_sets[partition][i][j];

            indexBits = (mode >= 10) ? 4 : 3;
            /* fix-up index is specified with one less bit */
            /* The fix-up index for subset 0 is always index 0 */
            if (partitionSet & 0x80) {
                indexBits--;
            }
            partitionSet &= 0x01;

            index = bc6h__bitstream_read_bits(&bstream, indexBits);

            ep_i = partitionSet * 2;
            decompressed[j * 3 + 0] = bc6h__finish_unquantize(
                                            bc6h__interpolate(r[ep_i], r[ep_i+1], weights, index), isSigned);
            decompressed[j * 3 + 1] = bc6h__finish_unquantize(
                                            bc6h__interpolate(g[ep_i], g[ep_i+1], weights, index), isSigned);
            decompressed[j * 3 + 2] = bc6h__finish_unquantize(
                                            bc6h__interpolate(b[ep_i], b[ep_i+1], weights, index), isSigned);
        }

        decompressed += destinationPitch;
    }
}


// Decode one 16-byte BC6H (UF16) block into a 4x4 RGBA8 tile (row-major).
void decode_bc6h_block(const uint8_t* blk, uint8_t out[16][4]) {
    unsigned short halfs[16 * 3];
    bc6h__decode_block_half(blk, halfs, 4 * 3, /*isSigned*/0);
    for (int t = 0; t < 16; t++) {
        for (int c = 0; c < 3; c++) {
            float f = bc6h__half_to_float_quick(halfs[t * 3 + c]);
            out[t][c] = (f != f || f <= 0.f) ? 0 : (f >= 1.f ? 255 : (uint8_t)(f * 255.f + 0.5f));
        }
        out[t][3] = 255;
    }
}

bool bc_decode_surface(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                       uint32_t width, uint32_t height, DataFormat fmt) {
    const uint32_t bb = bc_block_bytes(fmt);
    if (fmt != DataFormat::Bc1 && fmt != DataFormat::Bc2 && fmt != DataFormat::Bc3 &&
        fmt != DataFormat::Bc4 && fmt != DataFormat::Bc5 && fmt != DataFormat::Bc6 &&
        fmt != DataFormat::Bc7) return false;
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
            } else if (fmt == DataFormat::Bc6) {
                decode_bc6h_block(blk, texels);      // UF16 -> clamped RGBA8 (see the adapter above)
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
