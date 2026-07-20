#include "../src/gpu/gpu_execute.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

static int failures = 0;
#define CHECK(condition, message) do { \
    if (condition) std::printf("  [ok]   %s\n", message); \
    else { std::printf("  [FAIL] %s\n", message); ++failures; } \
} while (0)

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

static size_t count_extension(const std::filesystem::path& directory, const char* extension) {
    std::error_code ec;
    size_t count = 0;
    for (std::filesystem::directory_iterator it(directory, ec), end; !ec && it != end;
         it.increment(ec)) {
        if (it->is_regular_file(ec) && it->path().extension() == extension) ++count;
    }
    return ec ? 0 : count;
}

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
    clear_shader_analysis_cache();

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
    const std::filesystem::path dump_directory =
        std::filesystem::temp_directory_path() / "prosper-shader-dump-test";
    std::error_code dump_ec;
    std::filesystem::remove_all(dump_directory, dump_ec);
    set_test_env("PROSPER_SHADER_DUMP_SUCCESS", dump_directory.string().c_str());
    uint64_t first_identity = 0;
    const auto cached_vs = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &first_identity);
    CHECK(!direct_vs.empty() && cached_vs == direct_vs,
          "cache miss is byte-identical to the direct vertex recompiler");
    auto stats = shader_recompile_cache_stats();
    CHECK(stats.misses == 1 && stats.hits == 0 && stats.entries == 1,
          "first shader realization records one cache miss");
    CHECK(count_extension(dump_directory, ".bin") == 1 &&
              count_extension(dump_directory, ".spv") == 1,
          "successful shader diagnostics create one raw/SPIR-V pair");

    uint64_t repeated_identity = 0;
    const auto repeated_vs = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &repeated_identity);
    stats = shader_recompile_cache_stats();
    CHECK(repeated_vs == direct_vs && stats.hits == 1 && stats.misses == 1,
          "identical code and descriptor semantics hit the cache");
    CHECK(first_identity != 0 && repeated_identity == first_identity,
          "shader cache hits preserve a non-zero compiled-shader identity");
    auto analysis_stats = shader_analysis_cache_stats();
    CHECK(analysis_stats.misses == 1 && analysis_stats.hits == 1 &&
              analysis_stats.entries == 1,
          "repeated recompilation reuses byte-validated immutable shader analysis");
    CHECK(count_extension(dump_directory, ".bin") == 1 &&
              count_extension(dump_directory, ".spv") == 1,
          "successful shader diagnostics deduplicate cache hits");
    set_test_env("PROSPER_SHADER_DUMP_SUCCESS", nullptr);
    std::filesystem::remove_all(dump_directory, dump_ec);

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

    // Guest programs with identical bytes may be mapped at different addresses. Shader analysis is
    // address-local, but the compiled shader cache must still compare its immutable code by value.
    std::vector<uint32_t> relocated_vs(kVs, kVs + std::size(kVs));
    const auto relocated_cached = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, relocated_vs.data(), relocated_vs.size(), &table);
    stats = shader_recompile_cache_stats();
    CHECK(relocated_cached == cached_rebound && stats.hits == 3 && stats.misses == 2 &&
              stats.entries == 2,
          "byte-identical shader code at a different address reuses the compiled cache entry");

    const auto direct_ps = recompile_fragment(kPs, std::size(kPs), nullptr);
    const auto cached_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kPs, std::size(kPs), nullptr);
    CHECK(!direct_ps.empty() && cached_ps == direct_ps,
          "fragment cache output is byte-identical to the direct recompiler");

    // Unreal fragment shaders place small constant tables after S_ENDPGM and address them through an
    // s_getpc_b64-built V#. The owning cache copy must include that proven tail; copying only the walked
    // instruction span makes the cached recompiler reject s_getpc_b64 even though the direct path works.
    std::vector<uint32_t> pcrel_ps = {
        0xbe841f00u,               // s_getpc_b64 s[4:5]
        0x800404b0u,               // s_add_u32 s4, 48, s4 (table begins at byte 52)
        0x82050580u,               // s_addc_u32 s5, 0, s5
        0xbe860390u,               // s_mov_b32 s6, 16 bytes
        0xbe8703ffu, 0x10005004u,  // s_mov_b32 s7, V# format/stride
        0x7e020280u,               // v_mov_b32 v1, 0 (table byte offset)
        0xe0301000u, 0x80010101u,  // buffer_load_dword v1, v1, s[4:7], 0 offen
        0xbf8c3f70u,               // s_waitcnt vmcnt(0)
        0xf800180fu, 0x01010101u,  // exp mrt0 v1,v1,v1,v1
        0xbf810000u,               // s_endpgm
        7u, 11u, 13u, 17u,        // embedded table
    };
    CHECK(rdna2_recompile_code_span(pcrel_ps.data(), pcrel_ps.size()) == pcrel_ps.size(),
          "PC-relative cache span includes the embedded table tail");
    ShaderResourceTable pcrel_table;
    const auto direct_pcrel_ps = recompile_fragment(
        pcrel_ps.data(), pcrel_ps.size(), &pcrel_table);
    const auto cached_pcrel_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, pcrel_ps.data(), pcrel_ps.size(), &pcrel_table);
    CHECK(!direct_pcrel_ps.empty() && cached_pcrel_ps == direct_pcrel_ps,
          "fragment cache retains a proven post-ENDPGM PC-relative table");

    const auto pcrel_stats = shader_recompile_cache_stats();
    pcrel_ps.back() = 19u;
    const auto changed_pcrel_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, pcrel_ps.data(), pcrel_ps.size(), &pcrel_table);
    stats = shader_recompile_cache_stats();
    CHECK(!changed_pcrel_ps.empty() && changed_pcrel_ps != cached_pcrel_ps &&
              stats.misses == pcrel_stats.misses + 1,
          "embedded table contents participate in the shader cache key");

    // Astro Bot's title PS uses SMEM rather than MUBUF to consume its getpc-built scalar table.
    // The post-ENDPGM payload must be owned and hashed by the graphics cache in this form as well.
    std::vector<uint32_t> pcrel_smem_ps = {
        0xbe801f00u,               // s_getpc_b64 s[0:1]
        0x800000ffu, 52u,          // table = next-PC byte 4 + 52 = byte 56
        0x82010180u,               // s_addc_u32 s1, 0, s1
        0xbe820394u,               // s_mov_b32 s2, 20 bytes
        0xbe8303ffu, 0x10005004u,  // s_mov_b32 s3, V# config
        0xbeea0384u,               // s_mov_b32 vcc_lo, 4
        0xf4280100u, 0xd4000000u,  // s_buffer_load_dwordx4 s[4:7], s[0:3], vcc_lo
        0x7e000204u,               // v_mov_b32 v0, s4
        0xf800180fu, 0x00000000u,  // exp mrt0 v0,v0,v0,v0
        0xbf810000u,               // s_endpgm
        7u, 11u, 13u, 17u, 19u,
    };
    CHECK(rdna2_recompile_code_span(pcrel_smem_ps.data(), pcrel_smem_ps.size()) ==
              pcrel_smem_ps.size(),
          "PC-relative SMEM cache span includes the embedded scalar table tail");
    const auto direct_pcrel_smem_ps = recompile_fragment(
        pcrel_smem_ps.data(), pcrel_smem_ps.size(), nullptr);
    const auto cached_pcrel_smem_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, pcrel_smem_ps.data(), pcrel_smem_ps.size(), nullptr);
    CHECK(!direct_pcrel_smem_ps.empty() && cached_pcrel_smem_ps == direct_pcrel_smem_ps,
          "fragment cache retains a proven post-ENDPGM PC-relative SMEM table");
    const auto pcrel_smem_stats = shader_recompile_cache_stats();
    pcrel_smem_ps[pcrel_smem_ps.size() - 4] = 23u; // selected table[1]
    const auto changed_pcrel_smem_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, pcrel_smem_ps.data(), pcrel_smem_ps.size(), nullptr);
    stats = shader_recompile_cache_stats();
    CHECK(!changed_pcrel_smem_ps.empty() && changed_pcrel_smem_ps != cached_pcrel_smem_ps &&
              stats.misses == pcrel_smem_stats.misses + 1,
          "PC-relative SMEM table contents participate in the shader cache key");

    // A terminal if/else may build a PC-relative V# only in the second arm, after the first
    // s_endpgm. Span analysis must use the same extended instruction stream as emission. Stopping at
    // span pc20 immediately after the second end would omit four table dwords that can change SPIR-V.
    const uint32_t terminal_pcrel[] = {
        0xbe800387u, 0xbe810388u, 0xbf060100u, 0xbf840003u,
        0x7e0002ffu, 0x42280000u, 0xbf810000u,
        0xbe841f00u,               // pc7: s_getpc_b64 s[4:5] (next PC byte = 32)
        0x800404b0u,               // pc8: s_add_u32 s4, 48, s4 (table byte = 80)
        0x82050580u,               // pc9: s_addc_u32 s5, 0, s5
        0xbe860390u,               // pc10: s_mov_b32 s6, 16 bytes
        0xbe8703ffu, 0x10005004u,  // pc11: s_mov_b32 s7, V# format/stride
        0x7e020f00u,               // pc13: v_cvt_u32_f32 v1, v0
        0x34020282u,               // pc14: v_lshlrev_b32 v1, 2, v1
        0xe0301000u, 0x80010101u,  // pc15: buffer_load_dword v1, v1, s[4:7], 0 offen
        0xbf8c3f70u,               // pc17: s_waitcnt vmcnt(0)
        0x7e000d01u,               // pc18: v_cvt_f32_u32 v0, v1
        0xbf810000u,               // pc19: second s_endpgm
        7u, 11u, 13u, 17u,        // pc20: embedded table used only by the else arm
    };
    CHECK(rdna2_recompile_code_span(terminal_pcrel, std::size(terminal_pcrel)) ==
              std::size(terminal_pcrel),
          "terminal else-arm PC-relative table participates in the owning shader span");

    // Unity emits a bounded uniform jump table for uber-shader variants. The selector is loaded from
    // a direct constant buffer, adjusted by -1, clamped, scaled by eight, then used by s_setpc_b64.
    // This compact synthetic stream has two paths which set v0 differently before a common export.
    std::vector<uint32_t> dispatch_ps = {
        0xf4201a8cu, 0xfa000010u, // pc0: s_buffer_load_dword s106, s[24:27], 0x10
        0x816ac16au,              // pc2: s_add_i32 s106, s106, -1
        0x83ea826au,              // pc3: s_min_u32 s106, s106, 2
        0x8f6a836au,              // pc4: s_lshl_b32 s106, s106, 3
        0xbea01f00u,              // pc5: s_getpc_b64 s[32:33]
        0x802020ffu, 64u,         // pc6: add table byte delta (table starts at aligned pc22)
        0x82212180u,              // pc8: s_addc_u32 s33, 0, s33
        0xf4040890u, 0xd4000000u, // pc9: s_load_dwordx2 s[34:35], s[32:33], s106
        0xbea81f00u,              // pc11: s_getpc_b64 s[40:41]
        0x80282228u,              // pc12: s_add_u32 s40, s40, s34
        0x82292329u,              // pc13: s_addc_u32 s41, s41, s35
        0xbe802028u,              // pc14: s_setpc_b64 s[40:41]
        0x7e000280u,              // pc15 target A: v_mov_b32 v0, 0
        0xbf820001u,              // pc16: s_branch common export at pc18
        0x7e0002f2u,              // pc17 target B: v_mov_b32 v0, 1.0
        0xf800180fu, 0x00000000u, // pc18: exp mrt0 v0,v0,v0,v0
        0xbf810000u,              // pc20: s_endpgm
        0u,                       // pc21: alignment padding before the qword table
    };
    // Entries are signed byte offsets relative to the instruction after the second s_getpc (pc12).
    for (uint32_t index = 0; index < 3; ++index) {
        dispatch_ps.push_back(index == 0 ? 12u : index == 1 ? 20u : 24u); // pc15 / pc17 / merge pc18
        dispatch_ps.push_back(0u);
    }
    const PcrelDispatchInfo dispatch = rdna2_pcrel_dispatch_info(
        dispatch_ps.data(), dispatch_ps.size());
    CHECK(dispatch.valid && dispatch.selector_sgpr_base == 24 &&
              dispatch.selector_byte_offset == 0x10 && dispatch.selector_addend == -1 &&
              dispatch.selector_max == 2 && dispatch.target_pcs.size() == 3 &&
              dispatch.target_pcs[0] == 15 && dispatch.target_pcs[1] == 17 &&
              dispatch.target_pcs[2] == 18,
          "bounded PC-relative scalar dispatch is recognized and every target is proven");
    CHECK(rdna2_recompile_code_span(dispatch_ps.data(), dispatch_ps.size()) == dispatch_ps.size(),
          "scalar-dispatch table tail participates in the owning shader span");

    std::vector<uint32_t> reversed_shift_dispatch = dispatch_ps;
    reversed_shift_dispatch[4] = 0x8f6a6a83u; // s_lshl_b32 s106, 3, s106
    CHECK(!rdna2_pcrel_dispatch_info(reversed_shift_dispatch.data(),
                                     reversed_shift_dispatch.size()).valid,
          "reversed non-commutative dispatch shift is rejected");

    std::array<uint32_t, 5> selector_words{};
    ShaderResourceTable dispatch_table;
    ShaderResource selector_resource;
    selector_resource.cls = ResourceClass::ConstantBuffer;
    selector_resource.binding = 0;
    selector_resource.sgpr_base = 24;
    selector_resource.size = sizeof(selector_words);
    selector_resource.host_data = reinterpret_cast<uint8_t*>(selector_words.data());
    selector_resource.host_data_size = sizeof(selector_words);
    dispatch_table.resources.push_back(selector_resource);
    const auto direct_dispatch_a = recompile_fragment(
        dispatch_ps.data(), dispatch_ps.size(), &dispatch_table, nullptr, 15);
    const auto direct_dispatch_b = recompile_fragment(
        dispatch_ps.data(), dispatch_ps.size(), &dispatch_table, nullptr, 17);
    CHECK(!direct_dispatch_a.empty() && !direct_dispatch_b.empty(),
          "direct fragment specialization accepts both proven dispatch paths");
    selector_words[4] = 1; // adjusted selector 0 -> target A
    const auto dispatch_a = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, dispatch_ps.data(), dispatch_ps.size(), &dispatch_table);
    const auto dispatch_stats = shader_recompile_cache_stats();
    selector_words[4] = 2; // adjusted selector 1 -> target B
    const auto dispatch_b = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, dispatch_ps.data(), dispatch_ps.size(), &dispatch_table);
    stats = shader_recompile_cache_stats();
    CHECK(!dispatch_a.empty() && !dispatch_b.empty(),
          "both proven uniform-dispatch targets recompile");
    CHECK(dispatch_a != dispatch_b,
          "uniform dispatch target specializes the emitted fragment code");
    CHECK(stats.misses == dispatch_stats.misses + 1,
          "uniform dispatch target participates in the cache key");

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

    clear_shader_recompile_cache();
    clear_shader_analysis_cache();
    std::vector<uint32_t> mutable_ps(kPs, kPs + std::size(kPs));
    (void)recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, mutable_ps.data(), mutable_ps.size(), nullptr);
    (void)recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, mutable_ps.data(), mutable_ps.size(), nullptr);
    mutable_ps[0] ^= 1u;  // same allocation, changed instruction bytes
    (void)recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, mutable_ps.data(), mutable_ps.size(), nullptr);
    analysis_stats = shader_analysis_cache_stats();
    stats = shader_recompile_cache_stats();
    CHECK(analysis_stats.hits == 1 && analysis_stats.misses == 2 &&
              analysis_stats.invalidations == 1 && stats.hits == 1 && stats.misses == 2,
          "same-address shader mutation invalidates analysis and misses the compiled cache");

    // Live realization retains the cache allocation directly. Repeated hits must share one immutable
    // word vector, and eviction/reset must not invalidate a DrawItem that still owns that version.
    clear_shader_recompile_cache();
    uint64_t shared_first_identity = 0, shared_second_identity = 0;
    const SharedShaderWords shared_first = recompile_graphics_shader_cached_shared(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &shared_first_identity);
    const SharedShaderWords shared_second = recompile_graphics_shader_cached_shared(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &shared_second_identity);
    stats = shader_recompile_cache_stats();
    CHECK(shared_first && shared_first == shared_second && *shared_first ==
              recompile_vertex(kVs, std::size(kVs), &table) &&
              shared_first_identity != 0 && shared_first_identity == shared_second_identity &&
              stats.misses == 1 && stats.hits == 1,
          "live shader-cache hits share one exact immutable SPIR-V allocation");
    clear_shader_recompile_cache();
    CHECK(shared_first && !shared_first->empty(),
          "shared shader words outlive cache reset while a realized draw retains them");

    if (failures) {
        std::printf("== FAIL: %d ==\n", failures);
        return 1;
    }
    std::printf("== PASS ==\n");
    return 0;
}
