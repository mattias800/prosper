// Compare selected mip writes with a separately tiled full-block placement oracle.
// The companion runner also compares every result byte with the scalar control.
#include "gpu/texture/tile.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace prosper::gpu;
static int failures = 0;
static bool emit_enabled = true;
static void check(bool value) { failures += !value; }
static void emit(const std::vector<uint8_t>& bytes) {
    if (emit_enabled) std::fwrite(bytes.data(), 1, bytes.size(), stdout);
}

int main(int argc, char** argv) {
    emit_enabled = !(argc > 1 && std::strcmp(argv[1], "--check") == 0);
#ifdef _WIN32
    if (emit_enabled) _setmode(_fileno(stdout), _O_BINARY);
#endif
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const bool corrupt = argc > 1 && std::strcmp(argv[1], "--corrupt-oracle") == 0;
    if (bench) {
        std::vector<uint8_t> linear(64 * 64 * 8, 0x39), tiled(65536, 0xa5);
        constexpr unsigned iterations = 3000;
        const auto start = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < iterations; ++i) {
            tile_surface_level(tiled.data(), tiled.size(), linear.data(), 64, 64, 27, 8, 0, 0);
            detile_surface_level(linear.data(), tiled.data(), tiled.size(), 64, 64, 27, 8, 0, 0);
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::printf("iterations=%u tile-plus-detile-ms=%.6f sample=%u\n", iterations, ms, linear[123]);
        return 0;
    }
    for (uint32_t mode : {9u, 24u, 27u}) for (uint32_t bpe : {1u, 2u, 4u, 8u, 16u}) {
        const uint32_t bits = 16 - std::countr_zero(bpe);
        const uint32_t bw = 1u << ((bits + 1) / 2), bh = 1u << (bits / 2);
        std::vector<uint8_t> full(65536), encoded(65536);
        for (size_t i = 0; i < full.size(); ++i)
            full[i] = uint8_t((i * 31 + (i >> 7) * 13) ^ (i >> 11));
        tile_surface(encoded.data(), full.data(), bw, bh, mode, 0, bpe, false);
        for (auto [ox, oy] : {std::pair{0u, 0u}, std::pair{bw / 2, bh / 2},
                              std::pair{bw - 1, bh - 1}}) {
            const uint32_t w = bw - ox, h = bh - oy;
            std::vector<uint8_t> selected(size_t(w) * h * bpe);
            for (uint32_t y = 0; y < h; ++y)
                std::memcpy(selected.data() + size_t(y) * w * bpe,
                    full.data() + (size_t(y + oy) * bw + ox) * bpe, size_t(w) * bpe);
            // Stamped coverage locates valid whole texels even for truncated backing.
            std::vector<uint8_t> coverage(65536, 0), linear_coverage(65536, 0);
            for (uint32_t y = 0; y < h; ++y)
                std::memset(linear_coverage.data() + (size_t(y + oy) * bw + ox) * bpe,
                            1, size_t(w) * bpe);
            tile_surface(coverage.data(), linear_coverage.data(), bw, bh, mode, 0, bpe, false);
            for (size_t available : {size_t{0}, size_t{1}, size_t{32767}, size_t{65535}, size_t{65536}}) {
                std::vector<uint8_t> output(65536 + 32, 0xa5), expected = output;
                for (size_t i = 0; i + bpe <= available; i += bpe)
                    if (coverage[i]) std::memcpy(expected.data() + 16 + i, encoded.data() + i, bpe);
                if (corrupt) expected[16] ^= 1;
                tile_surface_level(output.data() + 16, available, selected.data(), w, h,
                                   mode, bpe, ox, oy);
                check(output == expected); emit(output);
                std::vector<uint8_t> read(selected.size() + 32, 0xdb), reference(read.size(), 0xdb);
                // A full-surface bounded detile supplies the independent read oracle.
                std::vector<uint8_t> complete(65536);
                detile_elements(complete.data(), encoded.data(), available, bw, bh, bpe, mode);
                for (uint32_t y = 0; y < h; ++y)
                    std::memcpy(reference.data() + 16 + size_t(y) * w * bpe,
                        complete.data() + (size_t(y + oy) * bw + ox) * bpe, size_t(w) * bpe);
                detile_surface_level(read.data() + 16, encoded.data(), available, w, h,
                                     mode, bpe, ox, oy);
                check(read == reference); emit(read);
            }
        }
        // Inputs outside an ordinary one-block tail compare byte-for-byte with the
        // scalar process: stack admission boundary, global coordinates and wrap.
        for (uint32_t width : {1u, 255u, 256u, 257u})
        for (uint32_t origin : {0u, 129u, UINT32_MAX}) {
            std::vector<uint8_t> input(size_t(width) * 3 * bpe, 0x72), output(3 * 65536, 0xa5);
            tile_surface_level(output.data(), output.size() - 1, input.data(), width, 3,
                               mode, bpe, origin, origin);
            emit(output);
            detile_surface_level(input.data(), output.data(), output.size() - 1, width, 3,
                                 mode, bpe, origin, origin);
            emit(input);
        }
    }
    std::fprintf(stderr, "mip-tail oracle failures=%d\n", failures);
    return failures ? 1 : 0;
}
