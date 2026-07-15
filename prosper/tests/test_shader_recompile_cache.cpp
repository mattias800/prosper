#include "../src/gpu/gpu_execute.hpp"
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

static int failures = 0;
#define CHECK(condition, message) do { \
    if (condition) std::printf("  [ok]   %s\n", message); \
    else { std::printf("  [FAIL] %s\n", message); ++failures; } \
} while (0)

// Fullscreen triangle and solid green shaders assembled for gfx1030. These are also used by the
// end-to-end GPU executor tests, but this test needs no Vulkan device.
static const uint32_t kVs[] = {
    0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u,
    0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u, 0x08040D02u,
    0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
};
static const uint32_t kPs[] = {
    0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u,
    0xF800180Fu, 0x03020100u, 0xBF810000u,
};

int main() {
    std::printf("== test_shader_recompile_cache ==\n");
    clear_shader_recompile_cache();

    ShaderResource resource;
    resource.cls = ResourceClass::VertexBuffer;
    resource.format = DataFormat::Float32;
    resource.num_components = 3;
    resource.binding = 7;
    resource.stride = 12;
    resource.srt_offset = 0x20;
    resource.sgpr_base = 8;
    resource.fetch_pc = 4;
    resource.gpu_addr = 0x100000;
    resource.size = 4096;
    ShaderResourceTable table;
    table.resources.push_back(resource);

    const auto direct_vs = recompile_vertex(kVs, std::size(kVs), &table);
    uint64_t first_identity = 0;
    const auto cached_vs = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &first_identity);
    CHECK(!direct_vs.empty() && cached_vs == direct_vs,
          "cache miss is byte-identical to the direct vertex recompiler");
    auto stats = shader_recompile_cache_stats();
    CHECK(stats.misses == 1 && stats.hits == 0 && stats.entries == 1,
          "first shader realization records one cache miss");

    uint64_t repeated_identity = 0;
    const auto repeated_vs = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &repeated_identity);
    stats = shader_recompile_cache_stats();
    CHECK(repeated_vs == direct_vs && stats.hits == 1 && stats.misses == 1,
          "identical code and descriptor semantics hit the cache");
    CHECK(first_identity != 0 && repeated_identity == first_identity,
          "shader cache hits preserve a non-zero compiled-shader identity");

    // These fields are consumed by the runtime backend, not by the recompiler. Changing them must
    // reuse SPIR-V while each DrawItem continues to carry the new table to descriptor upload.
    table.resources[0].gpu_addr = 0x900000;
    table.resources[0].size = 8192;
    table.resources[0].width = 1024;
    table.resources[0].height = 512;
    table.resources[0].mag_filter = 0;
    table.resources[0].host_data = reinterpret_cast<uint8_t*>(0x1234);
    table.resources[0].host_data_size = 16;
    const auto runtime_changed = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table);
    stats = shader_recompile_cache_stats();
    CHECK(runtime_changed == direct_vs && stats.hits == 2 && stats.misses == 1,
          "runtime-only resource changes preserve the compiled shader cache entry");

    // Binding participates in both descriptor declarations and memory-op lowering, so it must miss.
    table.resources[0].binding = 9;
    const auto direct_rebound = recompile_vertex(kVs, std::size(kVs), &table);
    const auto cached_rebound = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table);
    stats = shader_recompile_cache_stats();
    CHECK(cached_rebound == direct_rebound && stats.misses == 2 && stats.entries == 2,
          "compile-time resource changes miss and remain byte-identical to the oracle");

    const auto direct_ps = recompile_fragment(kPs, std::size(kPs), nullptr);
    const auto cached_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kPs, std::size(kPs), nullptr);
    CHECK(!direct_ps.empty() && cached_ps == direct_ps,
          "fragment cache output is byte-identical to the direct recompiler");

    // Interpolant wiring changes vertex PARAM export locations/defaults, so it is part of the
    // compile-time key even when the guest code and resource interface are otherwise identical.
    stats = shader_recompile_cache_stats();
    const uint64_t mapping_misses = stats.misses;
    const uint64_t mapping_hits = stats.hits;
    PixelInputMapping mapping;
    mapping.valid_mask = 1;
    mapping.controls[0] = 1;
    const auto mapped_once = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, &mapping);
    const auto mapped_again = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, &mapping);
    mapping.controls[0] = 2;
    const auto remapped = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, &mapping);
    stats = shader_recompile_cache_stats();
    CHECK(!mapped_once.empty() && mapped_again == mapped_once && !remapped.empty() &&
              stats.misses == mapping_misses + 2 && stats.hits == mapping_hits + 1,
          "pixel-input mappings participate in the vertex shader cache key");

    // Fragment system-input placement changes SPIR-V declarations and the initial VGPR values.
    // Both ENA and ADDR therefore belong to the cache key.
    const uint64_t system_misses = stats.misses;
    PixelSystemInputMapping system_inputs{0x00000303u, 0x00000303u};
    const auto system_once = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kPs, std::size(kPs), nullptr, nullptr, &system_inputs);
    const auto system_again = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kPs, std::size(kPs), nullptr, nullptr, &system_inputs);
    system_inputs.addr |= 1u << 2;
    const auto system_remapped = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kPs, std::size(kPs), nullptr, nullptr, &system_inputs);
    stats = shader_recompile_cache_stats();
    CHECK(!system_once.empty() && system_again == system_once && !system_remapped.empty() &&
              stats.misses == system_misses + 2,
          "pixel-system ENA/ADDR mappings participate in the fragment shader cache key");

    const uint64_t identity_before_clear = first_identity;
    clear_shader_recompile_cache();
    stats = shader_recompile_cache_stats();
    CHECK(stats.entries == 0 && stats.hits == 0 && stats.misses == 0 && stats.bytes == 0,
          "cache reset clears entries and instrumentation");
    uint64_t identity_after_clear = 0;
    (void)recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &identity_after_clear);
    CHECK(identity_after_clear > identity_before_clear,
          "cache reset never recycles compiled-shader identities");

    if (failures) {
        std::printf("== FAIL: %d ==\n", failures);
        return 1;
    }
    std::printf("== PASS ==\n");
    return 0;
}
