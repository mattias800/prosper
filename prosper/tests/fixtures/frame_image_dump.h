// frame_image_dump.h -- write an RGBA8 frame to a PPM or a 24-bit BMP. Moved verbatim out of
// render_runner.h; the live renderer and the render demos use them for on-disk screenshots.
#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

namespace prosper::test {

// Write an RGBA8 framebuffer to a binary PPM (P6) so a rendered frame is viewable as an image. PPM is
// trivially convertible to PNG (e.g. `magick in.ppm out.png`). Used by the render demos to leave
// screenshots on disk. Returns true on success.
inline bool dump_ppm(const char* path, const std::vector<uint8_t>& px, uint32_t W, uint32_t H) {
    if (px.size() != (size_t)W * H * 4) return false;
    FILE* f = fopen(path, "wb"); if (!f) return false;
    fprintf(f, "P6\n%u %u\n255\n", W, H);
    for (size_t i = 0; i < (size_t)W * H; i++) fwrite(&px[i * 4], 1, 3, f);   // RGB (drop alpha)
    fclose(f); return true;
}

// Write an RGBA8 framebuffer to a 24-bit BMP — natively viewable on Windows (double-click). BMP rows
// are bottom-up and BGR, padded to a 4-byte boundary.
inline bool dump_bmp(const char* path, const std::vector<uint8_t>& px, uint32_t W, uint32_t H) {
    if (px.size() != (size_t)W * H * 4) return false;
    FILE* f = fopen(path, "wb"); if (!f) return false;
    const uint32_t rowpad = (4 - (W * 3) % 4) % 4;
    const uint32_t dataSize = (W * 3 + rowpad) * H;
    const uint32_t fileSize = 54 + dataSize;
    auto u16 = [&](uint32_t v){ uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)}; fwrite(b, 1, 2, f); };
    auto u32 = [&](uint32_t v){ uint8_t b[4] = {(uint8_t)v, (uint8_t)(v>>8), (uint8_t)(v>>16), (uint8_t)(v>>24)}; fwrite(b, 1, 4, f); };
    fputc('B', f); fputc('M', f); u32(fileSize); u32(0); u32(54);                 // file header
    u32(40); u32(W); u32(H); u16(1); u16(24); u32(0); u32(dataSize); u32(2835); u32(2835); u32(0); u32(0);  // info header
    for (int y = (int)H - 1; y >= 0; y--) {                                       // bottom-up rows
        for (uint32_t x = 0; x < W; x++) {
            const uint8_t* p = &px[((size_t)y * W + x) * 4];
            fputc(p[2], f); fputc(p[1], f); fputc(p[0], f);                        // BGR
        }
        for (uint32_t k = 0; k < rowpad; k++) fputc(0, f);
    }
    fclose(f); return true;
}

} // namespace prosper::test
