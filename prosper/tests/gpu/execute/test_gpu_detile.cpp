// The shipped detile module runs on Vulkan and must match the established CPU
// address walk and scalar half conversion, including padding and pipe XOR.
#include "fixtures/compute_runner.h"
#include "gpu/recompiler/spirv_builder.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
  using namespace prosper::gpu;
  int failures = 0;
  auto check = [&](bool good, const char *message) {
    std::printf("[%s] %s\n", good ? "ok" : "FAIL", message);
    failures += !good;
  };
  for (const auto [width, height] :
       {std::pair{257u, 131u}, std::pair{1031u, 67u}}) {
    constexpr uint32_t faces = 6;
    const uint32_t count = width * height * faces;
    const size_t face_bytes = tiled_surface_bytes(width, height, 27, 0, 8);
    // An extra page between faces proves that the shader uses the supplied
    // stride, rather than a tight width*height*8 or inferred face footprint.
    const size_t stride = face_bytes + 4096;
    std::vector<uint8_t> bytes(80 + stride * faces, 0xa5);
    const uint32_t parameters[]{width, height,
                                static_cast<uint32_t>(stride / 4), count};
    std::memcpy(bytes.data(), parameters, sizeof(parameters));
    const auto equation = tile27_rgba16f_equation();
    std::memcpy(bytes.data() + 16, equation.data(), sizeof(equation));
    std::vector<uint32_t> expected(count);
    std::array<std::array<bool, 65536>, 4> seen{};
    for (uint32_t face = 0; face < faces; ++face) {
      std::vector<uint8_t> linear(size_t(width) * height * 8);
      for (uint32_t pixel = 0; pixel < width * height; ++pixel) {
        uint32_t packed = 0;
        for (uint32_t channel = 0; channel < 4; ++channel) {
          const uint16_t half = static_cast<uint16_t>(face * width * height +
                                                      pixel + channel * 173u);
          seen[channel][half] = true;
          std::memcpy(linear.data() + size_t(pixel) * 8 + channel * 2, &half,
                      2);
          float value = half_to_float(half);
          if (!std::isfinite(value) || value <= 0.0f)
            value = 0.0f;
          else if (value >= 1.0f)
            value = 1.0f;
          packed |= uint32_t(std::lround(value * 255.0f)) << (channel * 8);
        }
        expected[face * width * height + pixel] = packed;
      }
      tile_surface(bytes.data() + 80 + face * stride, linear.data(), width,
                   height, 27, 0, 8);
    }
    bool exhaustive = true;
    for (const auto &channel : seen)
      for (bool value : channel)
        exhaustive &= value;
    check(exhaustive, "every half encoding occurs in every channel");
    if (width == 257 && std::getenv("PROSPER_RX_PIPES") == nullptr) {
      // Independent addresses for the default 16-pipe eight-byte equation:
      // x8 sets bits8/13; x8+y8 cancels bit8; x64+y32 cancels bit10;
      // y64 adds pipe bit11 on top of the next 3-block-wide macroblock row.
      struct Address {
        uint32_t x, y, offset;
      };
      for (auto a : {Address{8, 0, 0x2100}, Address{8, 8, 0x2000},
                     Address{64, 32, 0x8000}, Address{0, 64, 0x30800}}) {
        uint16_t actual = 0;
        std::memcpy(&actual, bytes.data() + 80 + a.offset, 2);
        check(actual == uint16_t(a.y * width + a.x),
              "known tile address vector matches the source");
      }
    }
    std::vector<float> input(bytes.size() / 4);
    std::memcpy(input.data(), bytes.data(), bytes.size());
    const auto output = prosper::test::run_compute(
        build_compute_detile_rgba16f(), input, count + 127, count + 127, {}, {},
        nullptr, 128);
    check(output.size() == count + 127, "detile shader executes on Vulkan");
    if (output.size() != count + 127)
      continue;
    size_t mismatches = 0;
    for (uint32_t i = 0; i < count; ++i)
      mismatches += std::bit_cast<uint32_t>(output[i]) != expected[i];
    for (size_t i = count; i < output.size(); ++i)
      mismatches += std::bit_cast<uint32_t>(output[i]) != 0;
    std::printf("%ux%ux6: %zu byte-exact output/tail mismatches\n", width,
                height, mismatches);
    check(mismatches == 0, "GPU detile/quantize matches CPU pixels and refuses "
                           "excess invocations");
  }
  if (std::getenv("PROSPER_RX_PIPES") == nullptr) {
    // Seed the GPU input directly at independently known byte addresses, without
    // tile_surface. This catches a mutually consistent CPU/shader table error.
    constexpr uint32_t width = 257, height = 65, count = width * height;
    constexpr uint32_t face_bytes = 6 * 65536;
    std::vector<uint32_t> words(20 + face_bytes / 4, 0);
    words[0] = width;
    words[1] = height;
    words[2] = face_bytes / 4;
    words[3] = count;
    const auto equation = tile27_rgba16f_equation();
    std::memcpy(words.data() + 4, equation.data(), sizeof(equation));
    struct Witness {
      uint32_t x, y, byte_offset, low_halves, high_halves, rgba;
    };
    constexpr Witness witnesses[]{
        {8, 0, 0x2100, 0x38003c00, 0x30003400, 0x204080ff},
        {8, 8, 0x2000, 0x3c003000, 0x34003800, 0x4080ff20},
        {64, 32, 0x8000, 0x30003400, 0x38003c00, 0x80ff2040},
        {0, 64, 0x30800, 0x34003800, 0x3c003000, 0xff204080},
    };
    std::vector<uint32_t> expected(count, 0);
    for (const auto &witness : witnesses) {
      words[20 + witness.byte_offset / 4] = witness.low_halves;
      words[21 + witness.byte_offset / 4] = witness.high_halves;
      expected[witness.y * width + witness.x] = witness.rgba;
    }
    std::vector<float> input(words.size());
    std::memcpy(input.data(), words.data(), words.size() * sizeof(uint32_t));
    const auto output = prosper::test::run_compute(
        build_compute_detile_rgba16f(), input, count + 127, count + 127, {}, {},
        nullptr, 128);
    bool correct = output.size() == count + 127;
    for (size_t i = 0; i < output.size(); ++i)
      correct &= std::bit_cast<uint32_t>(output[i]) ==
                 (i < expected.size() ? expected[i] : 0);
    check(correct, "GPU reads directly seeded independent address witnesses");
  }
  return failures ? 1 : 0;
}
