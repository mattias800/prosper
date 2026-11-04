#pragma once
// The entry parameter word is followed by a real scalar-memory lifetime in s[16:23].
#include "fixtures/native_pixel_parameter.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <bit>
#include <cstdint>

namespace prosper::test::native_parameter_load {
using Words = native_pixel_parameter::Words;
inline Words program(uint32_t parameter = 20, bool smooth = false, bool buffer = false,
                     uint32_t observed = 20) {
    auto code = native_pixel_parameter::program(parameter, smooth);
    code[1] =
        (code[1] & ~(127u << 16)) | (30u << 16); // only the control overwrites its entry word
    const auto read = 9u + static_cast<uint32_t>(smooth);
    code[read] = (code[read] & ~511u) | observed;
    // LLVM gfx1030: S_LOAD/S_BUFFER_LOAD_DWORDX8 s[16:23], s[0:1]/s[4:7], 0.
    // Wait for the complete scalar result before its later numerical read (70648 section 7.3).
    const Words load{buffer ? 0xf42c0402u : 0xf40c0400u, 0xfa000000u, 0xbf8cc07fu};
    code.insert(code.begin() + 2, load.begin(), load.end());
    return code;
}
inline gpu::ShaderResourceTable resources(bool buffer = false) {
    gpu::ShaderResource source;
    source.cls = gpu::ResourceClass::ConstantBuffer;
    source.format = gpu::DataFormat::Uint32;
    source.num_components = 1;
    source.stride = 4;
    source.binding = 2;
    source.fetch_pc = 2;
    source.sgpr_base = buffer ? 4u : 0u;
    source.gpu_addr = 0x1000000;
    source.size = 8 * sizeof(uint32_t);
    source.scalar_buffer_dword_count = buffer ? 8u : 0u;
    gpu::ShaderResourceTable result;
    result.resources.push_back(source);
    return result;
}
inline Words payload(float green, uint32_t observed = 20) {
    Words result(8, 0x40000000u);
    result.at(observed - 16) = std::bit_cast<uint32_t>(green);
    return result;
}
} // namespace prosper::test::native_parameter_load
