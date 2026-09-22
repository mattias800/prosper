// Float32 array gather preserves the layer, packed signed offset and four-texel result order.
#include "fixtures/render_runner.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <bit>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace prosper::gpu;
using namespace prosper::test;
static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}

constexpr uint32_t Width = 8, Height = 8, Layers = 3;
static float texel(uint32_t layer, uint32_t x, uint32_t y, uint32_t component) {
    return float(64 * layer + 8 * y + x + 3 * component) / 256.0f + 1.0f / 65536.0f;
}

struct Case {
    const char* name;
    float layer;
    uint32_t selected_layer;
    bool offset, nsa;
    int x_offset, y_offset;
    uint32_t component;
    bool unnormalized = false;
};

struct Program {
    std::vector<uint32_t> words;
    size_t gather_pc;
};

static Program program(const Case& c) {
    Program result;
    auto mov = [&](uint32_t reg, uint32_t bits) {
        result.words.insert(result.words.end(), {0x7e0002ffu | (reg << 17), bits});
    };
    const uint32_t address = c.nsa ? 5u : 0u;
    const uint32_t u = c.nsa ? (c.offset ? 36u : 5u) : (c.offset ? 1u : 0u);
    const uint32_t v = c.nsa ? (c.offset ? 37u : 36u) : (c.offset ? 2u : 1u);
    const uint32_t layer = c.nsa ? 34u : (c.offset ? 3u : 2u);
    if (c.offset)
        mov(address, (uint32_t(c.x_offset) & 63u) | ((uint32_t(c.y_offset) & 63u) << 8));
    mov(u, std::bit_cast<uint32_t>(c.unnormalized ? 3.0f : 3.0f / Width));
    mov(v, std::bit_cast<uint32_t>(c.unnormalized ? 4.0f : 4.0f / Height));
    mov(layer, std::bit_cast<uint32_t>(c.layer));
    result.gather_pc = result.words.size();
    // T# s[8:15], S# s[16:19], VDATA v[12:15]. NSA offset layout deliberately uses
    // nonconsecutive [v5,v36,v37,v34], so reading VADDR+3 cannot accidentally select the layer.
    result.words.push_back(0xf0000000u | ((c.offset ? 0x57u : 0x47u) << 18) |
                           (1u << (8 + c.component)) | (5u << 3) | (c.nsa ? 2u : 0u));
    result.words.push_back(0x00820c00u | address);
    if (c.nsa) result.words.push_back(c.offset ? 0x00222524u : 0x00002224u);
    // At (3/8,4/8), the linear footprint starts at (2,3). Gather order is
    // (i0,j1),(i1,j1),(i1,j0),(i0,j0), independently specified here from the texture contents.
    constexpr uint32_t dx[]{0, 1, 1, 0}, dy[]{1, 1, 0, 0};
    for (uint32_t channel = 0; channel < 4; ++channel) {
        const float expected = texel(c.selected_layer, 2 + c.x_offset + dx[channel],
                                      3 + c.y_offset + dy[channel], c.component);
        const uint32_t reg = 12 + channel;
        result.words.insert(result.words.end(), {
            0x080000ffu | (reg << 17) | (reg << 9), std::bit_cast<uint32_t>(expected),
            0x100000ffu | (reg << 17) | (reg << 9), std::bit_cast<uint32_t>(65536.0f),
            0x060000ffu | (reg << 17) | (reg << 9), 0x3f000000u});
    }
    result.words.insert(result.words.end(), {0xf800000fu, 0x0f0e0d0cu, 0xbf810000u});
    return result;
}

static uint32_t constant_bits(const std::vector<uint32_t>& module, uint32_t id, unsigned depth = 0) {
    if (depth > 4) return UINT32_MAX;
    for (size_t at = 5; at < module.size();) {
        const uint32_t count = module[at] >> 16, opcode = module[at] & 0xffffu;
        if (!count || at + count > module.size()) break;
        if (count == 4 && module[at + 2] == id) {
            if (opcode == 43u) return module[at + 3]; // OpConstant
            if (opcode == 124u) return constant_bits(module, module[at + 3], depth + 1);
        }
        at += count;
    }
    return UINT32_MAX;
}

static bool gather_shape(const std::vector<uint32_t>& module, const Case& c) {
    uint32_t coordinate = 0;
    bool array_type = false, correct_gather = false;
    for (size_t at = 5; at < module.size();) {
        const uint32_t count = module[at] >> 16, opcode = module[at] & 0xffffu;
        if (!count || at + count > module.size()) return false;
        if (opcode == 25u && count >= 9u) // OpTypeImage Dim2D, Arrayed=1
            array_type |= module[at + 3] == 1u && module[at + 5] == 1u;
        if (opcode == 96u) { // OpImageGather; no Lod operand for the _lz base-view-level contract.
            coordinate = module[at + 4];
            correct_gather = count == (c.offset ? 8u : 6u) &&
                constant_bits(module, module[at + 5]) == c.component &&
                (!c.offset || module[at + 6] == 16u); // dynamic spatial Offset
        }
        at += count;
    }
    if (!array_type || !correct_gather) return false;
    for (size_t at = 5; at < module.size();) {
        const uint32_t count = module[at] >> 16, opcode = module[at] & 0xffffu;
        if (opcode == 80u && count == 6u && module[at + 2] == coordinate)
            return constant_bits(module, module[at + 5]) == std::bit_cast<uint32_t>(c.layer);
        at += count;
    }
    return false;
}

int main(int argc, char** argv) {
    bool compile_only = false;
    std::filesystem::path output;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--compile-only") == 0) compile_only = true;
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output = argv[++i];
        else { std::fprintf(stderr, "usage: %s [--compile-only] [--output DIR]\n", argv[0]); return 2; }
    }
    if (!output.empty()) std::filesystem::create_directories(output);
    ShaderResourceTable table;
    ShaderResource texture{};
    texture.cls = ResourceClass::Texture; texture.format = DataFormat::Float32;
    texture.binding = 4; texture.sgpr_base = 8; texture.img_dim = 5;
    texture.width = Width; texture.height = Height; texture.depth = Layers;
    texture.num_components = 4;
    table.resources.push_back(texture);
    const Case cases[]{
        {"plain-layer0", 0.0f, 0, false, false, 0, 0, 0},
        {"plain-layer2", 2.0f, 2, false, false, 0, 0, 0},
        {"plain-nsa-layer1", 1.0f, 1, false, true, 0, 0, 2},
        {"offset-positive-negative", 2.0f, 2, true, false, 1, -2, 0},
        {"offset-nsa-negative-positive", 1.0f, 1, true, true, -2, 1, 2},
        {"layer-round-down", 1.49f, 1, true, false, 1, -2, 0},
        {"layer-round-up", 1.51f, 2, true, true, -2, 1, 0},
        {"layer-clamp-low", -3.0f, 0, false, false, 0, 0, 0},
        {"layer-clamp-high", 7.0f, 2, false, true, 0, 0, 0},
        {"unnormalized-spatial-only", 2.0f, 2, true, true, 1, -2, 0, true},
    };
    std::vector<std::vector<uint32_t>> modules;
    for (const auto& c : cases) {
        table.resources[0].unnormalized = c.unnormalized;
        const auto guest = program(c);
        modules.push_back(recompile_fragment(guest.words.data(), guest.words.size(), &table));
        std::printf("case %s\n", c.name);
        check(!modules.back().empty() && gather_shape(modules.back(), c),
              "array gather keeps layer SSA, channel, base-level operation and spatial offset shape");
        if (!output.empty() && !modules.back().empty()) {
            std::ofstream file(output / (std::string(c.name) + ".spv"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(modules.back().data()), modules.back().size() * 4);
            check(bool(file), "SPIR-V module saved for offline validation");
        }
    }
    table.resources[0].unnormalized = false;
    const auto positive = program(cases[3]);
    struct Refusal { const char* name; unsigned word; uint32_t mask; };
    const Refusal refusals[]{
        {"A16 address", 1, 1u << 30}, {"D16 result", 1, 1u << 31},
        {"R128 descriptor", 0, 1u << 15},
        {"TFE status", 0, 1u << 16}, {"LWE status", 0, 1u << 17},
        {"DLC cache", 0, 1u << 7}, {"GLC cache", 0, 1u << 13},
        {"SLC cache", 0, 1u << 25}, {"reserved packet bit", 0, 1u << 6},
        {"multiple gather channels", 0, 2u << 8},
    };
    for (const auto& refusal : refusals) {
        auto words = positive.words;
        words[positive.gather_pc + refusal.word] |= refusal.mask;
        std::printf("refusal %s\n", refusal.name);
        check(recompile_fragment(words.data(), words.size(), &table).empty(),
              "unmodeled array gather control remains fail-visible beside its positive control");
    }
    auto unsupported = positive.words;
    unsupported[positive.gather_pc] ^= (0x57u ^ 0x37u) << 18; // sample_lz_o still loses array layer
    check(recompile_fragment(unsupported.data(), unsupported.size(), &table).empty(),
          "array sample-offset remains rejected; gather admission does not widen unrelated forms");
    if (compile_only || failures) return failures ? 1 : 0;

    const uint32_t vs[]{0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u,
        0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u,
        0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u};
    std::vector<float> pixels(Width * Height * Layers * 4);
    for (uint32_t layer = 0; layer < Layers; ++layer)
        for (uint32_t y = 0; y < Height; ++y)
            for (uint32_t x = 0; x < Width; ++x)
                for (uint32_t c = 0; c < 4; ++c)
                    pixels[((layer * Height + y) * Width + x) * 4 + c] = texel(layer, x, y, c);
    FrameResource resource;
    resource.binding = 4; resource.set = 1; resource.img_dim = 5; resource.guest_array = true;
    resource.tw = Width; resource.th = Height; resource.sample_count = Layers;
    resource.texture_format = VK_FORMAT_R32G32B32A32_SFLOAT;
    resource.tex_rgba = reinterpret_cast<const uint8_t*>(pixels.data());
    resource.tex_byte_size = pixels.size() * sizeof(float);
    resource.min_filter = resource.mag_filter = 0; // gather still takes the four linear-footprint texels
    ResolvedPipelineState state;
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; state.color_write_mask = 15;
    BackendDraw draw;
    draw.vs = recompile_vertex(vs, std::size(vs)); draw.ps = &state; draw.vcount = 3;
    draw.R = {resource};
    for (size_t i = 0; i < modules.size(); ++i) {
        draw.fs = modules[i];
        const auto rgba = render_draws_rgba({draw}, Width, Height);
        const size_t center = (Width * (Height / 2) + Width / 2) * 4;
        bool exact = rgba.size() == Width * Height * 4;
        if (exact) for (size_t c = 0; c < 4; ++c)
            exact &= rgba[center + c] >= 127 && rgba[center + c] <= 128;
        std::printf("execute %s\n", cases[i].name);
        check(exact, "all four gathered Float32 texels match independent layer/offset/order expectations");
    }
    return failures ? 1 : 0;
}
