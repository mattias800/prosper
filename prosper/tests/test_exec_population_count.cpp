// test_exec_population_count -- #2481.
//
// `s_bcnt1_i32_b64 sdst, exec` counts the wave's active lanes. emit_alu materializes it directly
// from architectural EXEC, so a shader that consumes the result in the SAME basic block compiles
// without any dataflow fact -- which is why the existing structured-route coverage arm passed while
// this defect was live. The CFG dispatcher erases every scalar word missing from its wave64 MUST set
// at each block entry, so unless that dataflow agrees the destination is scalar, the value silently
// disappears across a branch and the failure surfaces at whatever finally reads it, several
// instructions later and in an unrelated domain.
//
// GTA V's `exec_cs_413d88400` is that shader: `s_bcnt1_i32_b64 s6, exec` at pc337, `s_lshl_b32
// vcc_lo, s6, 2` at pc351, a block boundary at pc353, and `v_mov_b32 v1, vcc_lo` at pc354 -- which
// reported `mode=unresolved-operand ... fmt=7 op=0x1`, three instructions and one register file away
// from the cause. Synthetic reproductions of the shape were tried first and every one of them
// compiled on BOTH sides of the fix, because the structured routes claim them before the dispatcher
// ever runs. So this pins the exact production kernel and its exact routed resource table.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {

int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; } \
} while (0)

uint8_t nibble(char value) {
    return value >= '0' && value <= '9' ? static_cast<uint8_t>(value - '0')
                                        : static_cast<uint8_t>(value - 'a' + 10);
}

std::vector<uint32_t> program() {
    const std::filesystem::path path = std::filesystem::path(__FILE__).parent_path() /
        "data/rdna2_exec_population_count.hex";
    std::ifstream input(path);
    std::string hex((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    while (!hex.empty() && (hex.back() == '\n' || hex.back() == '\r')) hex.pop_back();
    CHECK(input.good() || input.eof(), "population-count program fixture is readable");
    CHECK(hex.size() == 372u * 8u, "population-count program fixture has 372 dwords");
    std::vector<uint32_t> words(hex.size() / 8u);
    auto* bytes = reinterpret_cast<uint8_t*>(words.data());
    for (size_t index = 0; index < words.size() * sizeof(uint32_t); ++index)
        bytes[index] = static_cast<uint8_t>(
            nibble(hex[index * 2u]) << 4u | nibble(hex[index * 2u + 1u]));
    return words;
}

struct RoutedResource {
    uint32_t binding;
    uint64_t gpu_addr;
    uint32_t size;
    uint32_t stride;
    uint32_t format;
    uint32_t components;
    uint32_t srt_offset;
    uint32_t sgpr_base;
    uint32_t fetch_pc;
};

// The exact table the routed dispatch published, read back out of its v54 capsule.
constexpr RoutedResource kRoutedResources[] = {
    {2u, 0x209cc963c0ull, 120u, 0u, 2u, 1u, 0xffffffffu, 0xffffffffu, 0x00000003u},
    {3u, 0x209cc964e8ull, 16512u, 8u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x0000000eu},
    {4u, 0x20f8480000ull, 148u, 0u, 2u, 1u, 0xffffffffu, 0xffffffffu, 0x0000001fu},
    {5u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000021u},
    {6u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000023u},
    {7u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000025u},
    {8u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000027u},
    {9u, 0x209cc963c0ull, 136u, 0u, 2u, 1u, 0xffffffffu, 0xffffffffu, 0x00000033u},
    {10u, 0x20f8480000ull, 120u, 0u, 2u, 1u, 0xffffffffu, 0xffffffffu, 0x00000041u},
    {11u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000054u},
    {12u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000056u},
    {13u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x0000005au},
    {14u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000064u},
    {15u, 0x209cc964e8ull, 16512u, 8u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x0000006eu},
    {16u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000078u},
    {17u, 0x209cc964e8ull, 16512u, 8u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000083u},
    {18u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x0000008eu},
    {19u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000090u},
    {20u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000092u},
    {21u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000094u},
    {22u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000097u},
    {23u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x0000009au},
    {24u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x0000009du},
    {25u, 0x20f84d2b50ull, 8272u, 4u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000a3u},
    {26u, 0x20f84d2b50ull, 8272u, 4u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000acu},
    {27u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000b8u},
    {28u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000bau},
    {29u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000beu},
    {30u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000c7u},
    {31u, 0x209cc964e8ull, 16512u, 8u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000d1u},
    {32u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000dbu},
    {33u, 0x209cc964e8ull, 16512u, 8u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000e6u},
    {34u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000f1u},
    {35u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000f3u},
    {36u, 0x209cc76000ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000f5u},
    {37u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000f9u},
    {38u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000fbu},
    {39u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x000000feu},
    {40u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000101u},
    {41u, 0x20f84d2b50ull, 8272u, 4u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000109u},
    {42u, 0x20f84d2b50ull, 8272u, 4u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000113u},
    {43u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000123u},
    {44u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000136u},
    {45u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000138u},
    {46u, 0x20f8480100ull, 132032u, 64u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000145u},
    {47u, 0x20f84d2b50ull, 8272u, 4u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x0000014au},
    {48u, 0x20f8480080ull, 4u, 4u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000163u},
    {49u, 0x20f84d0b14ull, 8252u, 4u, 1u, 1u, 0xffffffffu, 0xffffffffu, 0x00000171u},
};

ShaderResourceTable routed_table() {
    ShaderResourceTable table;
    for (const RoutedResource& entry : kRoutedResources) {
        ShaderResource resource;
        resource.cls = ResourceClass::ConstantBuffer;
        resource.binding = entry.binding;
        resource.gpu_addr = entry.gpu_addr;
        resource.size = entry.size;
        resource.stride = entry.stride;
        resource.format = static_cast<DataFormat>(entry.format);
        resource.num_components = entry.components;
        resource.srt_offset = entry.srt_offset;
        resource.sgpr_base = entry.sgpr_base;
        resource.fetch_pc = entry.fetch_pc;
        table.resources.push_back(resource);
    }
    return table;
}

ComputeShaderConfig routed_config() {
    ComputeShaderConfig config;
    // The routed launch: 12 user SGPRs, 64-wide workgroups, Wave64, native 64-lane subgroup. The
    // subgroup size is load-bearing: the ballot equals the guest wave mask only when the native
    // subgroup IS the guest wave, which is exactly the precondition the reduction is admitted under.
    config.user_sgprs.assign(12u, 0u);
    config.local_x = 64u; config.local_y = 1u; config.local_z = 1u;
    config.threads_x = 2063u; config.threads_y = 1u; config.threads_z = 1u;
    config.wave_size = 64u;
    config.native_subgroup_size = 64u;
    config.lds_bytes = 0u;
    config.tgid_x_en = true;
    return config;
}

} // namespace

int main() {
    const std::vector<uint32_t> code = program();
    ShaderResourceTable table = routed_table();
    const ComputeShaderConfig config = routed_config();
    const std::vector<uint32_t> spirv =
        recompile_compute(code.data(), code.size(), &table, config);
    CHECK(!spirv.empty(),
          "GTA V's EXEC population count keeps its scalar result across a dispatcher block");

    // Same-site mutation: a 32-lane native subgroup cannot supply the Wave64 ballot, so the
    // reduction is no longer exact and the shader must go back to failing visibly rather than
    // reading half a mask as though it were whole.
    ComputeShaderConfig narrow = config;
    narrow.native_subgroup_size = 32u;
    ShaderResourceTable narrow_table = routed_table();
    const std::vector<uint32_t> narrow_spirv =
        recompile_compute(code.data(), code.size(), &narrow_table, narrow);
    CHECK(narrow_spirv.empty(),
          "a narrower native subgroup keeps the population count fail-visible");

    if (failures) { std::fprintf(stderr, "== FAIL: %d ==\n", failures); return 1; }
    std::fprintf(stderr, "== PASS ==\n");
    return 0;
}
