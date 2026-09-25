#include "shared/live/depth_cube_quantize.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using prosper::frontend::depth_cube_avx2_available;
using prosper::frontend::depth_cube_simd_enabled;
using prosper::frontend::quantize_depth_cube_rgba8;
using prosper::frontend::quantize_depth_cube_rgba8_scalar;

static int failures = 0;
static void check(bool condition, const char* reason) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", reason);
        ++failures;
    }
}

int main(int argc, char** argv) {
    const bool control = argc == 2 && std::strcmp(argv[1], "--control") == 0;
    check(depth_cube_simd_enabled() == (depth_cube_avx2_available() && !control),
          "selected implementation follows the CPU capability and same-binary control");

    const std::array<float, 13> edges{
        -std::numeric_limits<float>::infinity(), -0.25f, -0.0f, 0.0f,
        0.49f / 255.0f, 0.5f / 255.0f, 0.51f / 255.0f,
        0.25f, 0.5f, 0.75f, 1.0f, 1.25f,
        std::numeric_limits<float>::infinity()};
    constexpr std::array<uint8_t, 13> expected{
        0, 0, 0, 0, 0, 1, 1, 64, 128, 191, 255, 255, 255};
    std::vector<float> values(edges.begin(), edges.end());
    // A multiply/add rounding boundary where x87 extended precision and SSE
    // can disagree. The fast path is deliberately unavailable on 32-bit x86.
    values.push_back(std::bit_cast<float>(0x3f010101u));
    // Bit-pattern coverage includes subnormals and out-of-range finite values, and
    // exercises many SIMD batches plus scalar tails without relying on a game image.
    uint32_t state = 0x7a92b16cu;
    for (size_t i = 0; i < 32768; ++i) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        const float value = std::bit_cast<float>(state);
        if (std::isfinite(value)) values.push_back(value);
    }
    std::vector<uint8_t> source(values.size() * sizeof(float));
    std::memcpy(source.data(), values.data(), source.size());
    for (size_t count : {size_t{0}, size_t{1}, size_t{7}, size_t{8}, size_t{9},
                         size_t{13}, size_t{31}, values.size()}) {
        std::vector<uint8_t> reference(count * 4 + 16, 0x5au);
        std::vector<uint8_t> result(count * 4 + 16, 0x5au);
        quantize_depth_cube_rgba8_scalar(reference.data(), source.data(), count);
        quantize_depth_cube_rgba8(result.data(), source.data(), count);
        check(result == reference, "selected conversion matches scalar bytes and preserves the tail");
        for (size_t i = 0; i < std::min(count, edges.size()); ++i)
            if (result[i * 4] != expected[i] || result[i * 4 + 1] != expected[i] ||
                result[i * 4 + 2] != expected[i] || result[i * 4 + 3] != 255) {
                std::fprintf(stderr, "edge %zu value %.9g expected %u got %u\n",
                             i, edges[i], expected[i], result[i * 4]);
                check(false, "known depth values retain the numeric RGBA8 contract");
            }
    }
    std::printf("depth cube quantize: %d failures; avx2=%d selected=%d\n",
                failures, depth_cube_avx2_available(), depth_cube_simd_enabled());
    return failures ? 1 : 0;
}
