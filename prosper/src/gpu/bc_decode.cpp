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

} // namespace

bool bc_decode_surface(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                       uint32_t width, uint32_t height, DataFormat fmt) {
    const uint32_t bb = bc_block_bytes(fmt);
    if (fmt != DataFormat::Bc1 && fmt != DataFormat::Bc2 && fmt != DataFormat::Bc3) return false;
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
