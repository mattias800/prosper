// Synthetic packed UINT image stores shared by compiler, validation and live execution tests.
// Nonzero sign-bit halves distinguish unsigned numeric conversion from half-float unpacking.
#pragma once

#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::test::uint_d16_store {
using namespace prosper::gpu;
inline constexpr uint32_t width = 8, height = 4;
inline constexpr std::array<uint32_t, 4> components{0x8001u, 0xffffu, 0x1234u, 0xabcdu};
inline constexpr std::array<uint32_t, 4> packed{0xffff8001u, 0xabcd1234u, 0x76543210u, 0xfedcba98u};
struct Options {
    uint32_t dim = 1, mask = 15, opcode = 8;
    bool d16 = true, dynamic_layer = false, predicated = false;
    std::array<uint32_t, 4> data = packed;
};
struct Program {
    std::vector<uint32_t> words;
    size_t image_pc = 0;
};
inline Program program(Options options = {}) {
    Program result;
    result.words.reserve(18);
    result.words.insert(
        result.words.end(),
        {0x7e080300u, 0x7e0a0281u,
         options.dynamic_layer ? 0x7e0c0300u : 0x7e0c0281u}); // v4=x,v5=1,v6=layer
    for (uint32_t word = 0; word < options.data.size(); ++word)
        result.words.insert(result.words.end(), {0x7e0002ffu | (word << 17), options.data[word]});
    if (options.predicated) result.words.push_back(0x7da40b04u); // CMPX_EQ_U32 v4,v5
    result.image_pc = result.words.size();
    result.words.push_back(0xf0000000u | (options.opcode << 18) | (options.mask << 8) |
                           (options.dim << 3));
    result.words.push_back((options.d16 ? 0x80000000u : 0u) | 4u); // T#s0,VDATA0,VADDR4
    result.words.push_back(0xbf810000u);
    return result;
}
inline ShaderResourceTable table(DataFormat format = DataFormat::Uint32, uint32_t channels = 4,
                                 uint32_t dim = 1, uint32_t layers = 1) {
    ShaderResource image;
    image.cls = ResourceClass::StorageImage;
    image.binding = 4;
    image.sgpr_base = 0;
    image.format = format;
    image.num_components = channels;
    image.img_dim = dim;
    image.width = width;
    image.height = height;
    image.depth = layers;
    image.gpu_addr = 0x10000;   // compiler-only symbolic backing, replaced by owned execution bytes
    image.size = width * height * layers * channels * data_format_bytes(format);
    ShaderResourceTable result;
    result.resources.push_back(image);
    return result;
}
inline ComputeShaderConfig config(uint32_t capabilities = 0) {
    ComputeShaderConfig result;
    result.user_sgprs.resize(8);
    result.local_x = width;
    result.local_y = result.local_z = 1;
    result.native_storage_format_support = capabilities;
    return result;
}
inline uint64_t diagnostic_address() {
    static std::atomic<uint64_t> next{0x4900000000ull};
    return next.fetch_add(0x100, std::memory_order_relaxed);
}
inline std::vector<uint32_t> compile(const Program& code, const ShaderResourceTable& resources,
                                     uint32_t capabilities = 0,
                                     uint64_t address = diagnostic_address()) {
    return recompile_compute(code.words.data(), code.words.size(), &resources, config(capabilities),
                             {RecompileDiagnosticStage::Compute, address});
}
}   // namespace prosper::test::uint_d16_store
