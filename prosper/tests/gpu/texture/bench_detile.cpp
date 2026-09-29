// Explicit, unregistered benchmark: compare PROSPER_NO_PAIRED_AVX2_DETILE=1 with the default
// using the same binary and alternating process order. Each invocation first proves exact bytes.
#include "gpu/texture/tile.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace prosper::gpu;

int main() {
    constexpr uint32_t width = 3840;
    constexpr uint32_t height = 2160;
    constexpr uint32_t bpe = 8;
    constexpr uint32_t mode = 27;  // Sw64KbRX, the observed 4K storage-seed mode.
    constexpr int rounds = 100;
    std::vector<uint8_t> linear(size_t(width) * height * bpe);
    for (size_t i = 0; i < linear.size(); ++i)
        linear[i] = static_cast<uint8_t>((i * 13u + 7u) ^ (i >> 8));
    std::vector<uint8_t> tiled(tiled_surface_bytes(width, height, mode, 0, bpe));
    std::vector<uint8_t> result(linear.size());
    tile_surface(tiled.data(), linear.data(), width, height, mode, 0, bpe);
    detile_surface(result.data(), tiled.data(), width, height, mode, 0, bpe);
    if (result != linear) return 1;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; ++i)
        detile_surface(result.data(), tiled.data(), width, height, mode, 0, bpe);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (result != linear) return 1;
    std::cout << "detile mode=" << mode << " extent=" << width << 'x' << height
              << " bpe=" << bpe << " rounds=" << rounds << " ms_per_call="
              << std::fixed << std::setprecision(6) << elapsed / rounds << '\n';
    return 0;
}
