// Project-owned DIM5 gather programs shared by compiler, SPIR-V validation and live-backend tests.
// Each invocation supplies a different floating-point array layer, not a constant base slice.
#pragma once

#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::test::compute_array_gather {
using namespace prosper::gpu;
inline constexpr uint32_t width = 8, height = 8, layers = 16;

struct Options {
    uint32_t opcode = 0x47, component = 0;
    bool nsa = false, unnormalized = false;
    float layer_bias = 0;
};
struct Program {
    std::vector<uint32_t> words;
    size_t image_pc = 0;
};

inline Program program(Options options = {}) {
    Program result;
    result.words.reserve(24);
    const bool offset = options.opcode == 0x57;
    const uint32_t address = options.nsa ? 5u : 0u;
    const uint32_t u = options.nsa ? (offset ? 36u : 5u) : (offset ? 1u : 0u);
    const uint32_t v = options.nsa ? (offset ? 37u : 36u) : (offset ? 2u : 1u);
    const uint32_t layer = options.nsa ? 34u : (offset ? 3u : 2u);
    auto mov = [&](uint32_t reg, uint32_t bits) {
        result.words.insert(result.words.end(), {0x7e0002ffu | (reg << 17), bits});
    };
    // Preserve LocalInvocationID.x for the output index before replacing spatial v0.
    result.words.push_back(0x7e080300u); // v_mov_b32 v4,v0
    result.words.push_back(0x7e000d00u | (layer << 17)); // v_cvt_f32_u32 layer,v0
    if (options.layer_bias != 0)
        result.words.insert(result.words.end(), {0x060000ffu | (layer << 17) | (layer << 9),
                                                 std::bit_cast<uint32_t>(options.layer_bias)});
    if (offset) mov(address, 0x3f01u); // signed spatial offset (+1,-1), never a layer offset
    mov(u, std::bit_cast<uint32_t>(options.unnormalized ? 3.0f : 3.0f / width));
    mov(v, std::bit_cast<uint32_t>(options.unnormalized ? 4.0f : 4.0f / height));
    result.image_pc = result.words.size();
    // T# s[8:15], S# s[16:19], VDATA v[8:11]. NSA deliberately separates every coordinate.
    result.words.push_back(0xf0000000u | (options.opcode << 18) | (1u << (8 + options.component)) |
                           (5u << 3) | (options.nsa ? 2u : 0u));
    result.words.push_back(0x00820800u | address);
    if (options.nsa) result.words.push_back(offset ? 0x00222524u : 0x00002224u);
    result.words.insert(result.words.end(), {0xbf8c3f70u, 0xe01c2000u, 0x80050804u, 0xbf810000u});
    return result; // buffer_store_dwordx4 v[8:11],v4,s[20:23],idxen
}

inline Program query() {
    // Query LOD0 and publish width,height,selected layer count,mip count for each invocation.
    return {
        {0x7e080300u, 0x7e000280u, 0xf0380f28u, 0x00820800u, 0xe01c2000u, 0x80050804u, 0xbf810000u},
        2};
}

inline ShaderResourceTable table(DataFormat format = DataFormat::Float16, uint32_t count = layers,
                                 uint32_t components = 1) {
    ShaderResourceTable result;
    result.resources.reserve(2);
    ShaderResource image;
    image.cls = ResourceClass::Texture;
    image.format = format;
    image.num_components = components;
    image.binding = 4;
    image.sgpr_base = 8;
    image.sampler_sgpr_base = 16;
    image.img_dim = 5;
    image.width = width;
    image.height = height;
    image.depth = count;
    image.gpu_addr = 0x10000;   // symbolic compiler-only backing; execution supplies owned storage
    image.size = width * height * count * components * (format == DataFormat::Float16 ? 2u : 4u);
    image.declared_mip_levels = 1;
    image.mag_filter = image.min_filter = 0;
    image.addr_uvw[0] = image.addr_uvw[1] = image.addr_uvw[2] = 2;   // clamp to edge
    result.resources.push_back(image);
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 4;
    output.binding = 0;
    output.sgpr_base = 20;
    output.stride = 16;
    output.size = layers * 16;
    output.gpu_addr = 0x20000;
    result.resources.push_back(output);
    return result;
}

inline ComputeShaderConfig config() {
    ComputeShaderConfig result;
    result.user_sgprs.resize(24);
    result.local_x = layers;
    result.local_y = result.local_z = 1;
    return result;
}
inline uint64_t diagnostic_address() {
    static std::atomic<uint64_t> next{0x4894000000ull};
    return next.fetch_add(0x100, std::memory_order_relaxed);
}
inline std::vector<uint32_t> compile(const Program& code, const ShaderResourceTable& resources,
                                     uint64_t address = diagnostic_address()) {
    return recompile_compute(code.words.data(), code.words.size(), &resources, config(),
                             {RecompileDiagnosticStage::Compute, address});
}
inline float texel(uint32_t layer, uint32_t x, uint32_t y, uint32_t component = 0) {
    return static_cast<float>(static_cast<int>(layer * 64 + y * 8 + x + component * 2) - 256);
}
}   // namespace prosper::test::compute_array_gather
