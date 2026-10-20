// Test shader blobs encode pointer fields relative to the field, just like unrelocated guest
// bytes. Absolute fixture pointers below 4 GiB are otherwise mistaken for offsets (#5295).
#pragma once

#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/pm4/command_processor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace prosper::test {

template <std::size_t N>
struct AgcShaderBlob {
    static_assert(N > 0 && N <= 255);
    gpu::AgcShaderHeader header{};
    std::array<gpu::ShaderReg, N> registers{};

    // Registration relocates the header in place. Reset before registering it again; the raw
    // representation can also be copied to another address before its first registration.
    void reset(std::array<gpu::ShaderReg, N> initial, uint32_t shader_size, uint8_t type = 0) {
        header = {};
        registers = initial;
        header.file_header = 0x34333231u;
        header.version = 0x18;
        using Blob = AgcShaderBlob<N>;
        constexpr auto field =
            offsetof(Blob, header) + offsetof(gpu::AgcShaderHeader, sh_registers);
        constexpr auto target = offsetof(Blob, registers);
        static_assert(target > field && target - field < 0x100000000ull);
        header.sh_registers = reinterpret_cast<const void*>(target - field);
        header.shader_size = shader_size;
        header.type = type;
        header.num_sh_registers = N;
    }
};

} // namespace prosper::test
