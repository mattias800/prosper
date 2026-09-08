// #3467: raw storage may retain an untouched floating-point texel without a per-store mask only
// when pack(unpack(guest)) preserves every bit. Unlike SNORM's redundant minimum encoding, binary16
// and unsigned F11/F10 must preserve their complete domains, including signaling-NaN payloads.
// CPU only: this calls the real public storage ABI without creating a Vulkan device. Register fresh
// processes with both PROSPER_NO_F16C_STORAGE_{UNPACK,PACK} controls to cover all four combinations.
#include "gpu/resources/shader_resources.hpp"
#include "shared/live/live_compute.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace prosper::frontend;
using namespace prosper::gpu;

namespace {
uint64_t failures = 0;
void check(const char* name, uint32_t input, uint32_t actual, uint32_t expected) {
    if (actual == expected) return;
    if (failures < 20)
        std::fprintf(stderr, "FAIL: %s input=%08x actual=%08x expected=%08x\n",
                     name, input, actual, expected);
    ++failures;
}

bool half_storage_roundtrip(uint32_t components) {
    // Each carried channel covers all 65,536 encodings; the extra texel also exercises the
    // accelerated path's scalar tail. Channel offsets expose channel/stride mixups.
    constexpr uint32_t texels = 65537;
    std::vector<uint16_t> input(texels * components), output(texels * components, 0xa55a);
    std::vector<uint32_t> channels(texels * 4);
    for (uint32_t texel = 0; texel < texels; ++texel)
        for (uint32_t channel = 0; channel < components; ++channel)
            input[texel * components + channel] = static_cast<uint16_t>(texel + channel * 7919);
    if (!storage_image_unpack_raw_uvec4(
            reinterpret_cast<const uint8_t*>(input.data()), input.size() * sizeof(uint16_t),
            DataFormat::Float16, components, texels, channels.data(), channels.size()))
        return false;
    for (uint32_t texel = 0; texel < texels; ++texel)
        for (uint32_t channel = 0; channel < components; ++channel) {
            const uint16_t bits = input[texel * components + channel];
            check("half ABI unpack", bits, channels[texel * 4 + channel],
                  storage_unpack_float16_bits(bits));
        }
    if (!storage_image_writeback_raw_uvec4(
            channels.data(), channels.size(), DataFormat::Float16, components,
            texels, 1, 1, 0, false, 0, 0, 0,
            reinterpret_cast<uint8_t*>(output.data()), output.size() * sizeof(uint16_t)))
        return false;
    for (size_t i = 0; i < input.size(); ++i)
        check("half ABI roundtrip", input[i], output[i], input[i]);

    if (components == 4) {
        storage_unpack_float16x4_range(
            reinterpret_cast<const uint8_t*>(input.data()), texels, channels.data());
        for (size_t i = 0; i < input.size(); ++i)
            check("half x4 range unpack", input[i], channels[i],
                  storage_unpack_float16_bits(input[i]));
        storage_pack_float16x4_range(
            channels.data(), texels, reinterpret_cast<uint8_t*>(output.data()));
        for (size_t i = 0; i < input.size(); ++i)
            check("half x4 range roundtrip", input[i], output[i], input[i]);
    }
    return true;
}

bool packed_storage_roundtrip(uint32_t lane) {
    // The production packed conversions operate independently on three disjoint fields. Sweep
    // each field's complete domain with NaNs in its neighbors, checking the full packed word.
    // This executes 5,120 words, not all 2^32 combinations.
    const uint32_t texels = lane == 2 ? 1024 : 2048;
    const uint32_t shift = lane == 0 ? 0 : lane == 1 ? 11 : 22;
    const uint32_t mask = lane == 2 ? 1023 : 2047;
    std::vector<uint32_t> input(texels), output(texels), channels(texels * 4);
    for (uint32_t i = 0; i < texels; ++i)
        input[i] = (0xffbff7ffu & ~(mask << shift)) | (i << shift);
    if (!storage_image_unpack_raw_uvec4(
            reinterpret_cast<const uint8_t*>(input.data()), input.size() * sizeof(uint32_t),
            DataFormat::Float10_11_11, 3, texels, channels.data(), channels.size()))
        return false;
    for (uint32_t i = 0; i < texels; ++i) {
        check("F11 lane 0 unpack", input[i], channels[i * 4],
              std::bit_cast<uint32_t>(f11_to_float(input[i] & 2047)));
        check("F11 lane 1 unpack", input[i], channels[i * 4 + 1],
              std::bit_cast<uint32_t>(f11_to_float((input[i] >> 11) & 2047)));
        check("F10 lane 2 unpack", input[i], channels[i * 4 + 2],
              std::bit_cast<uint32_t>(f10_to_float(input[i] >> 22)));
    }
    if (!storage_image_writeback_raw_uvec4(
            channels.data(), channels.size(), DataFormat::Float10_11_11, 3,
            texels, 1, 1, 0, false, 0, 0, 0,
            reinterpret_cast<uint8_t*>(output.data()), output.size() * sizeof(uint32_t)))
        return false;
    for (uint32_t i = 0; i < texels; ++i)
        check("F11/F10 ABI roundtrip", input[i], output[i], input[i]);
    return true;
}
} // namespace

int main() {
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const uint32_t unpacked = storage_unpack_float16_bits(static_cast<uint16_t>(bits));
        check("half scalar bits", bits, unpacked,
              std::bit_cast<uint32_t>(half_to_float(static_cast<uint16_t>(bits))));
        check("half scalar roundtrip", bits, float_to_half(std::bit_cast<float>(unpacked)), bits);
    }
    for (uint32_t bits = 0; bits < 2048; ++bits)
        check("F11 scalar roundtrip", bits, float_to_f11(f11_to_float(bits)), bits);
    for (uint32_t bits = 0; bits < 1024; ++bits)
        check("F10 scalar roundtrip", bits, float_to_f10(f10_to_float(bits)), bits);
    for (uint32_t components = 1; components <= 4; ++components)
        check("half ABI accepted", components, half_storage_roundtrip(components), true);
    for (uint32_t lane = 0; lane < 3; ++lane)
        check("packed ABI accepted", lane, packed_storage_roundtrip(lane), true);
    std::printf("storage float roundtrip: half=65536 F11=2048 F10=1024 encodings; "
                "half ABI=655370 + x4 range=262148 channel values; packed sweeps=5120 words; "
                "failures=%llu\n", static_cast<unsigned long long>(failures));
    return failures ? 1 : 0;
}
