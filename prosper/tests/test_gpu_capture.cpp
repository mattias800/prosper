#include "../src/gpu/gpu_capture.hpp"
#include "../src/gpu/agc_shader_layout.hpp"
#include "../src/gpu/guest_texture_layout.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include "../src/gpu/tile.hpp"
#include "gta5_nullable_output_fixture.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include "test_scratch.h"

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

alignas(256) static const uint32_t kDiagnosticVs[] = {
    0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u,
    0x10020B01u, 0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u,
    0xF80008CFu, 0x04030201u, 0xBF810000u,
};
alignas(256) static const uint32_t kDiagnosticMainVs[] = {
    0x7E000280u, 0x7E020281u, 0xF80008CFu, 0x01010000u, 0xBF810000u,
};
// An unconditional scalar branch remains deliberately unsupported and fail-visible.
alignas(256) static const uint32_t kDiagnosticBadPs[] = {
    0x06000300u, 0xbf820005u, 0xBF810000u,
};
alignas(256) static const uint32_t kDiagnosticCompute[] = {
    0xbf810000u, // s_endpgm
};

// Astro Bot's fullscreen VS addresses a 72-byte table after S_ENDPGM through an s_getpc_b64-built
// V#. Captures must retain that proven table span: raw replay otherwise sees only the first 45 words
// and rejects the shader even though live recompilation succeeded.
alignas(256) static const uint32_t kDiagnosticPcrelVs[] = {
    0xbfa00002u, 0x93eaff03u, 0x00080008u, 0x876bff03u,
    0x000000ffu, 0x8f6a8c6au, 0x887c6a6bu, 0xbf800000u,
    0xbf900009u, 0x906a8803u, 0x81ea6a80u, 0x90fe6ac1u,
    0xf8000941u, 0x00000000u, 0x81ea0380u, 0xbf8cff0fu,
    0x90fe6ac1u, 0xd56a0000u, 0x00020affu, 0xaaaaaaabu,
    0xbe8303ffu, 0x10005004u, 0xb0020048u, 0xbe801f00u,
    0x800000ffu, 0x00000060u, 0x82010180u, 0x2c000081u,
    0xd7460000u, 0x04010300u, 0x4c000105u, 0x34000083u,
    0xd7460004u, 0x04010300u, 0xe0381000u, 0x80000004u,
    0xe0341010u, 0x80000404u, 0xbf8c3f71u, 0xf80008cfu,
    0x03020100u, 0xbf8c3f70u, 0xf8000203u, 0x00000504u,
    0xbf810000u,
    // Alignment padding followed by the 18-dword table at byte offset 192.
    0u, 0u, 0u,
    0x3f800000u, 0x00000000u, 0x00000000u, 0x3f800000u,
    0x00000000u, 0x3f800000u, 0x00000000u, 0x3f800000u,
    0x3f800000u, 0x3f800000u, 0x00000000u, 0x3f800000u,
    0x00000000u, 0x00000000u, 0x3f800000u, 0x3f800000u,
    0x00000000u, 0x3f800000u,
};

static void set_pgm(GpuState& state, uint32_t lo_register, uint32_t hi_register,
                    const void* program) {
    const uint64_t addr = reinterpret_cast<uint64_t>(program);
    state.sh[lo_register] = static_cast<uint32_t>(addr >> 8);
    state.sh[hi_register] = static_cast<uint32_t>((addr >> 40) & 0xffu);
}

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

static std::string shell_quote(const std::filesystem::path& path) {
    return "\"" + path.string() + "\"";
}

static std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

int main(int argc, char** argv) {
    std::printf("== test_gpu_capture ==\n");
    const char* inspect_tool = argc == 3 &&
            std::strcmp(argv[1], "--inspect-resource-input") == 0
        ? argv[2] : nullptr;
    const char* original_save0_value = std::getenv(kGpuCaptureSave0Env);
    const bool had_original_save0 = original_save0_value != nullptr;
    const std::string original_save0 = original_save0_value ? original_save0_value : "";
    GpuCaptureMetadata save_roots;
    set_test_env(kGpuCaptureSave0Env, "/tmp/prosper-test-effective-save0");
    annotate_gpu_capture_save_roots(save_roots);
    CHECK(save_roots.renderer_env.size() == 1 &&
              save_roots.renderer_env[0].first == kGpuCaptureSave0Env &&
              save_roots.renderer_env[0].second == "/tmp/prosper-test-effective-save0",
          "capture diagnostics retain the effective /savedata0 host root");
    set_test_env(kGpuCaptureSave0Env, nullptr);
    annotate_gpu_capture_save_roots(save_roots);
    CHECK(save_roots.renderer_env.size() == 1 &&
              save_roots.renderer_env[0].second == kGpuCaptureDefaultSave0Root,
          "capture diagnostics make the default /savedata0 root explicit");
    set_test_env(kGpuCaptureSave0Env,
                 had_original_save0 ? original_save0.c_str() : nullptr);
    std::vector<uint8_t> memory(32);
    for (size_t i = 0; i < memory.size(); ++i) memory[i] = static_cast<uint8_t>(0x40 + i);
    auto reader = [&](uint64_t addr, uint8_t* dst, size_t n) -> size_t {
        const auto read_shader = [&](const uint32_t* words, size_t bytes) -> size_t {
            const uint64_t base = reinterpret_cast<uint64_t>(words);
            if (addr < base || addr >= base + bytes) return 0;
            const size_t offset = static_cast<size_t>(addr - base);
            const size_t take = std::min(n, bytes - offset);
            std::memcpy(dst, reinterpret_cast<const uint8_t*>(words) + offset, take);
            return take;
        };
        if (const size_t read = read_shader(kDiagnosticVs, sizeof(kDiagnosticVs))) return read;
        if (const size_t read = read_shader(kDiagnosticMainVs, sizeof(kDiagnosticMainVs))) return read;
        if (const size_t read = read_shader(kDiagnosticBadPs, sizeof(kDiagnosticBadPs))) return read;
        if (const size_t read = read_shader(kDiagnosticCompute, sizeof(kDiagnosticCompute))) return read;
        if (const size_t read = read_shader(kDiagnosticPcrelVs, sizeof(kDiagnosticPcrelVs))) return read;
        if (addr < 0x1000 || addr >= 0x1000 + memory.size()) return 0;
        size_t off = static_cast<size_t>(addr - 0x1000), take = std::min(n, memory.size() - off);
        std::memcpy(dst, memory.data() + off, take); return take;
    };

    auto table = std::make_shared<ShaderResourceTable>();
    ShaderResource a{}; a.cls = ResourceClass::VertexBuffer; a.binding = 9; a.gpu_addr = 0x1000;
    a.size = 16; a.stride = 4; a.format = DataFormat::Sint2_10_10_10; a.num_components = 4; a.fetch_pc = 12;
    a.fetch_index_mode = VertexFetchIndexMode::Instance;
    ShaderResource b{}; b.cls = ResourceClass::ConstantBuffer; b.binding = 2; b.gpu_addr = 0x1008;
    b.size = 16; b.format = DataFormat::Float32; b.num_components = 4; b.srt_offset = 0x20;
    b.bvh_box_grow = 6;
    b.bvh_sort_enabled = true;
    ShaderResource dcc{}; dcc.cls = ResourceClass::Texture; dcc.binding = 4; dcc.gpu_addr = 0x1000;
    dcc.size = 16; dcc.format = DataFormat::Unorm8; dcc.num_components = 4;
    dcc.width = dcc.height = 2; dcc.max_uncompressed_block_size = 2;
    dcc.max_compressed_block_size = 1; dcc.meta_pipe_aligned = true;
    dcc.compression_enabled = true; dcc.alpha_is_on_msb = true;
    dcc.metadata_addr = 0x206e33ab00ull;
    table->resources = {a, b, dcc};

    DrawItem draw; draw.vs = {0x07230203, 1, 2}; draw.fs = {0x07230203, 3}; draw.vrt = table;
    draw.vs_guest_addr = reinterpret_cast<uint64_t>(kDiagnosticVs);
    draw.vs_chain_guest_addr = reinterpret_cast<uint64_t>(kDiagnosticMainVs);
    draw.fs_guest_addr = reinterpret_cast<uint64_t>(kDiagnosticBadPs);
    draw.vertex_lds_dwords = 2176;
    draw.has_pixel_inputs = true;
    draw.pixel_inputs.valid_mask = 3;
    draw.pixel_inputs.passthrough_mask = 1;
    draw.pixel_inputs.controls[0] = 0x420;
    draw.pixel_inputs.controls[1] = 1;
    draw.has_system_inputs = true;
    draw.system_inputs = {0x1fu, 0x1fu};
    draw.vertex_count = 6; draw.instance_count = 3;
    draw.raw_draw_count = 6; draw.raw_indexed = false;   // #1256: raw draw-packet state (v23)
    draw.raw_draw_modifier = 0x1122334455667788ull;
    draw.vertex_offset = -37;
    draw.indices = {0, 1, 2, 2, 3, 0}; draw.color0_base = 0x2000;
    draw.color0_width = 1024; draw.color0_height = 32;
    draw.color1_base = 0x3000; draw.color1_width = 1024; draw.color1_height = 32;
    draw.draw_index = 7; draw.command_order = 123;
    draw.ps.topology = 3; draw.ps.color0_format = 37; draw.ps.blend_enable = true;
    draw.ps.has_viewport = true; draw.ps.viewport_w = 1920; draw.ps.viewport_h = -1080;
    draw.ps.db_render_control = 2; draw.ps.stencil_clear_enable = true;
    draw.ps.has_stencil_clear = true; draw.ps.stencil_clear_value = 3;
    draw.ps.stencil_read_base = 0x12345000; draw.ps.raw_stencil_op[0][1] = 4;
    draw.ps.db_shader_control = 2; draw.ps.stencil_test_val_export_enable = true;
    draw.ps.db_depth_view = 0x04002001; draw.ps.htile_data_base = 0x12389000;
    draw.ps.db_depth_size_xy = 0x01230234; draw.ps.db_depth_info = 0x44;
    draw.ps.db_z_info = 0x55; draw.ps.db_stencil_info = 0x66;
    draw.ps.db_depth_size = 0x77; draw.ps.db_depth_slice = 0x88;
    draw.ps.db_htile_surface = 0x99;
    draw.ps.color1_format = 37; draw.ps.has_clear_color1 = true;
    draw.ps.clear_color1[0] = 0.25f; draw.ps.clear_color1[3] = 1.0f;
    draw.ps.blend1_enable = true; draw.ps.src_color_blend_factor1 = 1;
    draw.ps.dst_color_blend_factor1 = 7; draw.ps.color_blend_op1 = 0;
    draw.ps.src_alpha_blend_factor1 = 1; draw.ps.dst_alpha_blend_factor1 = 7;
    draw.ps.alpha_blend_op1 = 0; draw.ps.color1_write_mask = 0xf;
    draw.ps.cb_resolve = true;
    draw.ps.depth_bias_enable = 1;            // #1349: v29 depth-bias tail
    draw.ps.depth_bias_constant = 4.0f;
    draw.ps.depth_bias_slope = 2.0f;
    draw.ps.depth_bias_clamp = 0.5f;
    draw.ps.spi_shader_col_format = 4;
    draw.ps.sx_ps_downconvert = 5;
    // #1459: v43 raw color-state provenance. A PRESENT zero target mask is deliberately chosen here
    // because it is exactly the state an absent register must NOT be confused with.
    draw.ps.has_cb_color_control = true; draw.ps.cb_color_control = 0x00CC0010u;
    draw.ps.has_cb_target_mask = true;   draw.ps.cb_target_mask = 0u;
    draw.ps.has_cb_shader_mask = true;   draw.ps.cb_shader_mask = 0x000000FFu;
    draw.ps.has_scissor = true; draw.ps.scissor_left = 12; draw.ps.scissor_top = 19;
    draw.ps.scissor_right = 70; draw.ps.scissor_bottom = 75;
    draw.ps.logic_op_enable = true; draw.ps.logic_op = 6;

    GpuCaptureMetadata meta; meta.width = 480; meta.height = 270; meta.submit_index = 42;
    meta.revision = "deadbeef"; meta.title_id = "PPSA24651"; meta.input_route = "@reach-level.pad";
    meta.savedata_dir = "/tmp/prosper-test-save";
    meta.renderer_env = {{"PROSPER_RTT_PERTARGET", "1"}, {"PROSPER_NO_CULL", "1"}};
    GpuCaptureFile captured; std::string error;
    const std::vector<uint8_t> temporal_rgba = {
        255, 0, 0, 255, 0, 255, 0, 255,
        0, 0, 255, 255, 255, 255, 255, 255,
    };
    auto rtt_reader = [&](uint64_t addr, GpuCaptureRttSeed& seed) {
        if (addr != draw.color0_base) return false;
        seed.guest_addr = addr; seed.width = 2; seed.height = 2; seed.rgba = temporal_rgba; return true;
    };
    CHECK(capture_draw_items({draw}, meta, reader, captured, error, rtt_reader), "capture realized draw succeeds");
    CHECK(captured.blobs.size() == 1 && captured.blobs[0].guest_addr == 0x1000 && captured.blobs[0].bytes.size() == 24,
          "overlapping resource ranges merge into one alias-preserving blob");
    CHECK(captured.blobs[0].bytes_read == 24 && captured.blobs[0].bytes[8] == memory[8],
          "capture records readable byte count and resource contents");
    CHECK(captured.blobs[0].content_hash == gpu_capture_hash(captured.blobs[0].bytes),
          "resource bytes receive an immutable content identity");
    CHECK(captured.draws[0].vrt.resources[0].blob_index == captured.draws[0].vrt.resources[1].blob_index &&
          captured.draws[0].vrt.resources[1].blob_offset == 8, "resource references preserve overlap offsets");
    CHECK(captured.rtt_seeds.size() == 1 && captured.rtt_seeds[0].guest_addr == draw.color0_base &&
          captured.rtt_seeds[0].rgba == temporal_rgba,
          "capture stores referenced temporal RTT pixels separately from guest memory");
    CHECK(captured.shader_versions.size() == 2 && captured.operations.size() == 1 &&
          captured.operations[0].source_index == 7 && captured.operations[0].realized,
          "capture indexes unique shaders and retains operation provenance");
    CHECK(captured.draws.size() == 1 && captured.draws[0].instance_count == 3,
          "capture retains the realized draw instance count");
    CHECK(captured.draws[0].raw_draw_count == 6 && !captured.draws[0].raw_indexed,
          "capture retains the raw draw-packet count and indexed flag (#1256)");
    CHECK(captured.draws[0].raw_draw_modifier == 0x1122334455667788ull &&
          captured.draws[0].vertex_offset == -37,
          "capture retains the draw modifier and GE_INDX_OFFSET draw parameter");
    CHECK(captured.raw_shader_versions.size() == 3 &&
          captured.draws[0].vs_raw_shader_index < captured.raw_shader_versions.size() &&
          captured.draws[0].fs_raw_shader_index < captured.raw_shader_versions.size() &&
          captured.draws[0].vs_chain_raw_shader_index < captured.raw_shader_versions.size() &&
          captured.raw_shader_versions[captured.draws[0].vs_raw_shader_index].words ==
              std::vector<uint32_t>(std::begin(kDiagnosticVs), std::end(kDiagnosticVs)) &&
          captured.raw_shader_versions[captured.draws[0].fs_raw_shader_index].words ==
              std::vector<uint32_t>(std::begin(kDiagnosticBadPs), std::end(kDiagnosticBadPs)) &&
          captured.raw_shader_versions[captured.draws[0].vs_chain_raw_shader_index].words ==
              std::vector<uint32_t>(std::begin(kDiagnosticMainVs), std::end(kDiagnosticMainVs)) &&
          captured.draws[0].vertex_lds_dwords == 2176,
          "realized draw captures exact bounded linked VS, FS, and graphics-LDS state");
    CHECK(captured.draws[0].has_pixel_inputs && captured.draws[0].has_system_inputs &&
          captured.draws[0].pixel_inputs == draw.pixel_inputs &&
          captured.draws[0].system_inputs == draw.system_inputs,
          "realized draw captures the exact fixed-function pixel-stage ABI");

    // #1849: one opt-in semantic resource snapshot must sample the bytes at draw realization,
    // independently of ordinary deferred materialization after the submit.
    GpuCaptureResourceSelector provenance_selector;
    CHECK(parse_gpu_capture_resource_selector("0x4d:ps:040", provenance_selector) &&
              provenance_selector.draw_index == 77 &&
              provenance_selector.stage == ShaderProgramStage::Fragment &&
              provenance_selector.binding == 32,
          "resource provenance parser shares strict base-0 semantic selector rules");
    CHECK(!parse_gpu_capture_resource_selector("77:fs:32", provenance_selector) &&
              !parse_gpu_capture_resource_selector("77:ps", provenance_selector) &&
              !parse_gpu_capture_resource_selector("-1:ps:32", provenance_selector) &&
              !parse_gpu_capture_resource_selector("77:ps:0x100000000", provenance_selector),
          "resource provenance parser rejects malformed, signed, and oversized selectors");

    std::array<uint8_t, 16> provenance_memory{};
    for (size_t i = 0; i < provenance_memory.size(); ++i)
        provenance_memory[i] = static_cast<uint8_t>(0x10 + i);
    const std::array<uint8_t, 16> realization_bytes = provenance_memory;
    auto provenance_table = std::make_shared<ShaderResourceTable>();
    ShaderResource provenance_resource;
    provenance_resource.cls = ResourceClass::ConstantBuffer;
    provenance_resource.binding = 32;
    provenance_resource.gpu_addr = reinterpret_cast<uint64_t>(provenance_memory.data());
    provenance_resource.size = provenance_memory.size();
    provenance_resource.stride = 16;
    provenance_resource.sgpr_base = 0;
    provenance_resource.direct_vsharp_sh_register_base =
        P::SPI_SHADER_USER_DATA_PS_0;
    provenance_table->resources = {provenance_resource};
    DrawItem provenance_draw = draw;
    provenance_draw.draw_index = 77;
    provenance_draw.command_order = 777;
    provenance_draw.vrt.reset();
    provenance_draw.prt = provenance_table;
    const std::vector<DrawItem> provenance_draws = {provenance_draw};
    const std::vector<SubmitOperation> provenance_operations = {
        {SubmitOperationKind::Draw, 77, 777},
    };
    auto make_provenance_pending = [&](const std::filesystem::path& path) {
        auto pending = std::make_unique<PendingGpuCapture>();
        pending->materialized = false;
        pending->path = path.string();
        pending->capture.metadata = meta;
        pending->resource_provenance_armed = true;
        pending->resource_provenance_selector = {77, ShaderProgramStage::Fragment, 32};
        return pending;
    };
    const auto provenance_path = prosper_test::test_scratch_dir() /
        "prosper_gpu_capture_resource_provenance.prgcap";
    auto provenance_pending = make_provenance_pending(provenance_path);
    snapshot_pending_gpu_capture_draw_resource(provenance_pending.get(), provenance_draw);
    auto& input_witness = provenance_pending->resource_provenance;
    input_witness.input_mode = GpuCaptureResourceInputMode::DirectVSharp;
    input_witness.input_unavailable_reason =
        GpuCaptureResourceInputUnavailableReason::None;
    input_witness.input_write_provenance_mask = 0x0f;
    input_witness.input_sh_register_base = P::SPI_SHADER_USER_DATA_PS_0;
    const uint64_t provenance_addr = reinterpret_cast<uint64_t>(provenance_memory.data());
    input_witness.input_dwords = {
        static_cast<uint32_t>(provenance_addr),
        static_cast<uint32_t>((provenance_addr >> 32u) & 0xffffu) | (16u << 16u),
        1u,
        (22u << 12u) | 0xFACu,
    };
    for (uint32_t i = 0; i < 4; ++i) {
        input_witness.input_last_writes[i] = 700u + i;
        input_witness.input_write_sources[i] = GpuState::pack_prov_src(1, 0, 91);
    }
    for (size_t i = 0; i < provenance_memory.size(); ++i)
        provenance_memory[i] = static_cast<uint8_t>(0xa0 + i);
    const std::array<uint8_t, 16> post_bytes = provenance_memory;
    GpuCaptureFile provenance_loaded;
    CHECK(finish_requested_gpu_capture(
              std::move(provenance_pending), {}, error, &provenance_draws, nullptr,
              &provenance_operations) &&
              read_gpu_capture(provenance_path.string(), provenance_loaded, error) &&
              provenance_loaded.resource_provenance.size() == 1,
          "selected resource provenance persists one exact temporal audit");
    const auto* provenance = provenance_loaded.resource_provenance.size() == 1
        ? &provenance_loaded.resource_provenance.front() : nullptr;
    const auto* realization_blob = provenance &&
            provenance->realization_blob_index < provenance_loaded.blobs.size()
        ? &provenance_loaded.blobs[provenance->realization_blob_index] : nullptr;
    const auto* post_blob = provenance &&
            provenance->post_blob_index < provenance_loaded.blobs.size()
        ? &provenance_loaded.blobs[provenance->post_blob_index] : nullptr;
    CHECK(provenance && realization_blob && post_blob &&
              realization_blob->bytes_read == provenance_memory.size() &&
              post_blob->bytes_read >= provenance->post_blob_offset + provenance_memory.size() &&
              std::equal(realization_bytes.begin(), realization_bytes.end(),
                         realization_blob->bytes.begin()) &&
              std::equal(post_bytes.begin(), post_bytes.end(),
                         post_blob->bytes.begin() + provenance->post_blob_offset) &&
              provenance->realization_content_hash ==
                  gpu_capture_hash(realization_bytes.data(), realization_bytes.size()) &&
              provenance->post_content_hash ==
                  gpu_capture_hash(post_bytes.data(), post_bytes.size()) &&
              provenance->realization_content_hash != provenance->post_content_hash,
          "selected snapshot retains draw-time A and independently reports post-submit B");
    CHECK(provenance && provenance->draw_index == 77 &&
              provenance->stage == ShaderProgramStage::Fragment &&
              provenance->binding == 32 &&
              provenance->resource_class == ResourceClass::ConstantBuffer &&
              provenance->guest_addr == reinterpret_cast<uint64_t>(provenance_memory.data()) &&
              provenance->requested_bytes == provenance_memory.size() &&
              provenance->srt_offset == 0xFFFFFFFFu && provenance->sgpr_base == 0 &&
              provenance->input_sh_register_base == P::SPI_SHADER_USER_DATA_PS_0 &&
              provenance->input_write_provenance_mask == 0x0f &&
              gpu_capture_resource_input_verdict(*provenance, provenance_resource) ==
                  GpuCaptureResourceInputVerdict::FullMatch,
          "resource provenance round-trips an independently decodable direct-V# input witness");

    GpuCaptureFile mutated_input = provenance_loaded;
    std::vector<uint8_t> mismatch_bytes;
    GpuCaptureFile mismatch_loaded;
    mutated_input.resource_provenance[0].input_dwords[0] ^= 0x100u;
    CHECK(serialize_gpu_capture(mutated_input, mismatch_bytes, error) &&
              deserialize_gpu_capture(mismatch_bytes, mismatch_loaded, error) &&
              gpu_capture_resource_input_verdict(
                  mismatch_loaded.resource_provenance[0], provenance_resource) ==
                  GpuCaptureResourceInputVerdict::Mismatch,
          "a raw address/output disagreement round-trips as an explicit mismatch");

    GpuCaptureResourceProvenance zero_record = *provenance;
    zero_record.input_dwords[2] = 0;
    zero_record.input_unavailable_reason =
        GpuCaptureResourceInputUnavailableReason::RawSizeUnavailable;
    ShaderResource compatibility_sized = provenance_resource;
    compatibility_sized.size = compatibility_sized.stride * 4u;
    CHECK(gpu_capture_resource_input_verdict(zero_record, compatibility_sized) ==
              GpuCaptureResourceInputVerdict::Unavailable,
          "zero-record V# does not manufacture the PS compatibility allocation as raw proof");
    ShaderResource zero_record_wrong_format = compatibility_sized;
    zero_record_wrong_format.format = DataFormat::Uint32;
    CHECK(gpu_capture_resource_input_verdict(zero_record, zero_record_wrong_format) ==
              GpuCaptureResourceInputVerdict::Mismatch,
          "zero-record V# still rejects a normalized format disagreement");

    GpuCaptureResourceProvenance unknown_format = *provenance;
    unknown_format.input_dwords[3] &= ~(0x7fu << 12u);
    unknown_format.input_unavailable_reason =
        GpuCaptureResourceInputUnavailableReason::RawFormatUnavailable;
    ShaderResource shader_formatted = provenance_resource;
    shader_formatted.format = DataFormat::Float32;
    shader_formatted.num_components = 4;
    CHECK(gpu_capture_resource_input_verdict(unknown_format, shader_formatted) ==
              GpuCaptureResourceInputVerdict::Unavailable,
          "FORMAT=INVALID V# does not manufacture shader-derived Float32x4 as raw proof");
    ShaderResource unknown_format_wrong_size = shader_formatted;
    unknown_format_wrong_size.size += unknown_format_wrong_size.stride;
    CHECK(gpu_capture_resource_input_verdict(unknown_format, unknown_format_wrong_size) ==
              GpuCaptureResourceInputVerdict::Mismatch,
          "FORMAT=INVALID V# still rejects a normalized byte-size disagreement");

    auto resource_bool_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t rc = 0;
        for (const auto& d : f.draws) rc += d.vrt.resources.size() + d.prt.resources.size();
        for (const auto& cm : f.computes) rc += cm.resources.resources.size();
        for (const auto& diagnostic : f.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                rc += stage.resource_table.resources.size();
        return 4 + rc;
    };
    // v47/v48 use one proof byte per resource; v49 stores one qword record-count dword.
    auto v47_tail = [&](const GpuCaptureFile& f) -> size_t { return resource_bool_tail(f); };
    auto v48_tail = [&](const GpuCaptureFile& f) -> size_t { return resource_bool_tail(f); };
    auto v49_tail = [&](const GpuCaptureFile& f) -> size_t {
        return 4u + (resource_bool_tail(f) - 4u) * sizeof(uint32_t);
    };
    auto v50_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t configs = 0;
        for (const auto& compute : f.computes)
            configs += compute.recompile_config_available;
        for (const auto& diagnostic : f.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                configs += stage.recompile_config_available;
        return 4u + configs * sizeof(uint32_t);
    };

    // The fixed v46 tail is 4 bytes of count plus 100 bytes per selected witness. Mutate the
    // serialized tail, rather than merely constructing invalid in-memory objects, so the reader's
    // hostile-input boundary proves the exact pack contract it accepts.
    std::vector<uint8_t> valid_v46_bytes;
    CHECK(serialize_gpu_capture(provenance_loaded, valid_v46_bytes, error) &&
              valid_v46_bytes.size() >= 104u + v47_tail(provenance_loaded) +
                  v48_tail(provenance_loaded) + v49_tail(provenance_loaded) +
                  v50_tail(provenance_loaded),
          "v46 resource input witness has a complete validation tail");
    const size_t v46_tail_base = valid_v46_bytes.size() >=
            104u + v47_tail(provenance_loaded) + v48_tail(provenance_loaded) +
                v49_tail(provenance_loaded) + v50_tail(provenance_loaded)
        ? valid_v46_bytes.size() - v50_tail(provenance_loaded) -
              v49_tail(provenance_loaded) -
              v48_tail(provenance_loaded) -
              v47_tail(provenance_loaded) - 104u : 0u;
    auto malformed_tail_rejected = [&](size_t offset,
                                       std::initializer_list<uint8_t> replacement,
                                       const char* wanted_error) {
        if (valid_v46_bytes.size() < 104u + v47_tail(provenance_loaded) +
                v48_tail(provenance_loaded) + v49_tail(provenance_loaded) +
                v50_tail(provenance_loaded) ||
            offset > valid_v46_bytes.size() - replacement.size())
            return false;
        std::vector<uint8_t> bytes = valid_v46_bytes;
        std::copy(replacement.begin(), replacement.end(), bytes.begin() + offset);
        GpuCaptureFile rejected;
        error.clear();
        return !deserialize_gpu_capture(bytes, rejected, error) && error == wanted_error;
    };
    constexpr size_t kV46BaseOffset = 20u;
    constexpr size_t kV46SourcesOffset = 72u;
    CHECK(malformed_tail_rejected(
              v46_tail_base + kV46SourcesOffset,
              {0, 0, 0, 0, 0, 0, 0, 0},
              "invalid resource input write source"),
          "v46 tail rejects a zero source/fold provenance word");
    CHECK(malformed_tail_rejected(
              v46_tail_base + kV46SourcesOffset,
              {0xff, 0, 91, 0, 0, 0, 0, 0},
              "invalid resource input write source"),
          "v46 tail rejects an out-of-contract queue origin");
    CHECK(malformed_tail_rejected(
              v46_tail_base + kV46SourcesOffset,
              {1, 9, 91, 0, 0, 0, 0, 0},
              "invalid resource input write source"),
          "v46 tail rejects a Jump depth beyond the fold recursion bound");
    CHECK(malformed_tail_rejected(
              v46_tail_base + kV46SourcesOffset,
              {1, 0, 91, 0, 0, 0, 0, 0x80},
              "invalid resource input write source"),
          "v46 tail rejects high bits outside pack_prov_src");
    CHECK(malformed_tail_rejected(
              v46_tail_base + kV46BaseOffset,
              {0xfd, 0xff, 0, 0},
              "invalid direct resource input witness"),
          "v46 tail rejects a four-dword SH origin outside the bounded register file");

    if (inspect_tool) {
        GpuCaptureFile inspect_mismatch = provenance_loaded;
        inspect_mismatch.draws[0].prt.resources[0].resource.format = DataFormat::Uint32;
        const auto inspect_mismatch_path = prosper_test::test_scratch_path(
            "resource-input-inspect-mismatch.prgcap");
        GpuCaptureFile inspect_unavailable = provenance_loaded;
        inspect_unavailable.resource_provenance[0] = zero_record;
        const auto inspect_unavailable_path = prosper_test::test_scratch_path(
            "resource-input-inspect-unavailable.prgcap");
        CHECK(write_gpu_capture(inspect_mismatch_path.string(), inspect_mismatch, error) &&
                  write_gpu_capture(inspect_unavailable_path.string(), inspect_unavailable, error),
              "gpu_replay resource-input inspect fixtures serialize");

        const auto run_inspect = [&](const std::filesystem::path& capture_path,
                                     const char* log_name) {
            const auto log_path = prosper_test::test_scratch_path(log_name);
            const std::string command = shell_quote(inspect_tool) + " --inspect-only " +
                shell_quote(capture_path) + " > " + shell_quote(log_path) + " 2>&1";
            const int status = std::system(command.c_str());
            return std::pair<int, std::string>{status, read_text_file(log_path)};
        };
        const auto full = run_inspect(provenance_path, "resource-input-inspect-full.log");
        const auto mismatch = run_inspect(
            inspect_mismatch_path, "resource-input-inspect-mismatch.log");
        const auto unavailable = run_inspect(
            inspect_unavailable_path, "resource-input-inspect-unavailable.log");
        CHECK(full.first == 0 &&
                  full.second.find(
                      "resource-input draw=77 stage=ps binding=32 raw-identity=full-match") !=
                      std::string::npos &&
                  full.second.find("writes=d700/q1,f91,j0") != std::string::npos,
              "gpu_replay --inspect-only prints the full-match raw-input witness");
        CHECK(mismatch.first == 0 &&
                  mismatch.second.find(
                      "resource-input draw=77 stage=ps binding=32 raw-identity=mismatch") !=
                      std::string::npos,
              "gpu_replay --inspect-only prints a raw-input mismatch");
        CHECK(unavailable.first == 0 &&
                  unavailable.second.find(
                      "resource-input draw=77 stage=ps binding=32 raw-identity=unavailable "
                      "reason=raw-size-not-architecturally-available") != std::string::npos,
              "gpu_replay --inspect-only prints partial raw-input unavailability");
        std::printf("== %s ==\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    }

    std::vector<uint8_t> v45_provenance_bytes;
    GpuCaptureFile v45_provenance_loaded;
    CHECK(serialize_gpu_capture(provenance_loaded, v45_provenance_bytes, error) &&
              v45_provenance_bytes.size() >=
                  v50_tail(provenance_loaded) + v49_tail(provenance_loaded) +
                      v48_tail(provenance_loaded) +
                      v47_tail(provenance_loaded) +
                      4u + 100u * provenance_loaded.resource_provenance.size(),
          "v46 resource input tail is removable for a v45 compatibility fixture");
    if (v45_provenance_bytes.size() >= v50_tail(provenance_loaded) +
            v49_tail(provenance_loaded) +
            v48_tail(provenance_loaded) + v47_tail(provenance_loaded) +
            4u + 100u * provenance_loaded.resource_provenance.size()) {
        v45_provenance_bytes.resize(
            v45_provenance_bytes.size() - v50_tail(provenance_loaded) -
                v49_tail(provenance_loaded) -
                v48_tail(provenance_loaded) -
                v47_tail(provenance_loaded));
        v45_provenance_bytes.resize(
            v45_provenance_bytes.size() -
            (4u + 100u * provenance_loaded.resource_provenance.size()));
        v45_provenance_bytes[8] = 45;
        CHECK(deserialize_gpu_capture(v45_provenance_bytes, v45_provenance_loaded, error) &&
                  v45_provenance_loaded.format_version == 45 &&
                  v45_provenance_loaded.resource_provenance.size() == 1 &&
                  v45_provenance_loaded.resource_provenance[0].input_mode ==
                      GpuCaptureResourceInputMode::Unavailable &&
                  v45_provenance_loaded.resource_provenance[0].input_unavailable_reason ==
                      GpuCaptureResourceInputUnavailableReason::CapturePredatesWitness,
              "v45 selected-resource capture reads with explicit raw-input unavailability");
    }

    size_t default_reader_calls = 0;
    PendingGpuCapture default_pending;
    snapshot_pending_gpu_capture_draw_resource(
        &default_pending, provenance_draw,
        [&](uint64_t, uint8_t*, size_t) -> size_t {
            ++default_reader_calls;
            return 0;
        });
    CHECK(default_reader_calls == 0 && default_pending.resource_provenance_matches == 0 &&
              default_pending.resource_provenance_realization_blob.bytes.empty(),
          "disabled provenance performs no resource read, allocation, or byte copy");

    const auto partial_path = prosper_test::test_scratch_dir() /
        "prosper_gpu_capture_resource_provenance_partial.prgcap";
    auto partial_pending = make_provenance_pending(partial_path);
    snapshot_pending_gpu_capture_draw_resource(
        partial_pending.get(), provenance_draw,
        [&](uint64_t, uint8_t* destination, size_t) -> size_t {
            std::memcpy(destination, provenance_memory.data(), 4);
            return 4;
        });
    GpuCaptureFile partial_loaded;
    CHECK(finish_requested_gpu_capture(
              std::move(partial_pending), {}, error, &provenance_draws, nullptr,
              &provenance_operations) &&
              read_gpu_capture(partial_path.string(), partial_loaded, error) &&
              partial_loaded.resource_provenance.size() == 1 &&
              partial_loaded.blobs[
                  partial_loaded.resource_provenance[0].realization_blob_index].bytes_read == 4 &&
              partial_loaded.blobs[
                  partial_loaded.resource_provenance[0].post_blob_index].bytes_read >=
                  partial_loaded.resource_provenance[0].post_blob_offset +
                      provenance_memory.size() &&
              partial_loaded.resource_provenance[0].input_mode ==
                  GpuCaptureResourceInputMode::Unavailable &&
              partial_loaded.resource_provenance[0].input_unavailable_reason ==
                  GpuCaptureResourceInputUnavailableReason::NoSemanticState,
          "partial read and missing semantic input state remain independently fail-visible");

    DrawItem srt_draw = provenance_draw;
    srt_draw.prt = std::make_shared<ShaderResourceTable>();
    ShaderResource srt_resource = provenance_resource;
    srt_resource.srt_offset = 0;
    srt_resource.sgpr_base = 0xFFFFFFFFu;
    srt_resource.direct_vsharp_sh_register_base =
        ShaderResource::kDirectVSharpOriginUnavailable;
    srt_draw.prt->resources = {srt_resource};
    PendingGpuCapture srt_pending;
    srt_pending.resource_provenance_armed = true;
    srt_pending.resource_provenance_selector = {
        77, ShaderProgramStage::Fragment, 32};
    snapshot_pending_gpu_capture_draw_resource(&srt_pending, srt_draw);
    CHECK(srt_pending.resource_provenance.input_mode ==
              GpuCaptureResourceInputMode::Unavailable &&
              srt_pending.resource_provenance.input_unavailable_reason ==
                  GpuCaptureResourceInputUnavailableReason::NonDirectResource,
          "SRT-loaded V# remains explicitly unavailable to the direct-SH witness");

    DrawItem ambiguous_origin_draw = provenance_draw;
    ambiguous_origin_draw.prt = std::make_shared<ShaderResourceTable>();
    ShaderResource ambiguous_origin_resource = provenance_resource;
    ambiguous_origin_resource.direct_vsharp_sh_register_base =
        ShaderResource::kDirectVSharpOriginAmbiguous;
    ambiguous_origin_draw.prt->resources = {ambiguous_origin_resource};
    PendingGpuCapture ambiguous_origin_pending;
    ambiguous_origin_pending.resource_provenance_armed = true;
    ambiguous_origin_pending.resource_provenance_selector = {
        77, ShaderProgramStage::Fragment, 32};
    snapshot_pending_gpu_capture_draw_resource(
        &ambiguous_origin_pending, ambiguous_origin_draw);
    CHECK(ambiguous_origin_pending.resource_provenance.input_mode ==
              GpuCaptureResourceInputMode::Unavailable &&
              ambiguous_origin_pending.resource_provenance.input_unavailable_reason ==
                  GpuCaptureResourceInputUnavailableReason::AmbiguousRawOrigin,
          "shader-derived or ambiguous V# origin remains explicitly unavailable");

    auto zero_match_pending = make_provenance_pending(
        prosper_test::test_scratch_dir() / "prosper_gpu_capture_resource_provenance_zero.prgcap");
    zero_match_pending->resource_provenance_selector.binding = 99;
    snapshot_pending_gpu_capture_draw_resource(zero_match_pending.get(), provenance_draw);
    CHECK(!finish_requested_gpu_capture(
              std::move(zero_match_pending), {}, error, &provenance_draws, nullptr,
              &provenance_operations) &&
              error.find("matched 0 resources") != std::string::npos,
          "zero-match resource provenance selector fails visibly before capture write");

    DrawItem ambiguous_draw = provenance_draw;
    ambiguous_draw.prt = std::make_shared<ShaderResourceTable>(*provenance_table);
    ambiguous_draw.prt->resources.push_back(provenance_resource);
    const std::vector<DrawItem> ambiguous_draws = {ambiguous_draw};
    auto ambiguous_pending = make_provenance_pending(
        prosper_test::test_scratch_dir() / "prosper_gpu_capture_resource_provenance_ambiguous.prgcap");
    snapshot_pending_gpu_capture_draw_resource(ambiguous_pending.get(), ambiguous_draw);
    CHECK(!finish_requested_gpu_capture(
              std::move(ambiguous_pending), {}, error, &ambiguous_draws, nullptr,
              &provenance_operations) &&
              error.find("ambiguous") != std::string::npos,
          "multiple-match resource provenance selector fails visibly before capture write");

    DrawItem pcrel_draw = draw;
    pcrel_draw.vs_guest_addr = reinterpret_cast<uint64_t>(kDiagnosticPcrelVs);
    pcrel_draw.vs_chain_guest_addr = 0;
    pcrel_draw.fs_guest_addr = 0;
    GpuCaptureFile pcrel_capture;
    CHECK(capture_draw_items({pcrel_draw}, meta, reader, pcrel_capture, error, rtt_reader) &&
              pcrel_capture.raw_shader_versions.size() == 1 &&
              pcrel_capture.raw_shader_versions[0].words ==
                  std::vector<uint32_t>(std::begin(kDiagnosticPcrelVs),
                                        std::end(kDiagnosticPcrelVs)),
          "Astro PC-relative VS capture retains its proven post-ENDPGM table");

    DrawItem shared_shader_draw = draw;
    shared_shader_draw.vs_shared =
        std::make_shared<const std::vector<uint32_t>>(shared_shader_draw.vs);
    shared_shader_draw.fs_shared =
        std::make_shared<const std::vector<uint32_t>>(shared_shader_draw.fs);
    shared_shader_draw.vs.clear();
    shared_shader_draw.fs.clear();
    GpuCaptureFile shared_shader_capture;
    CHECK(capture_draw_items({shared_shader_draw}, meta, reader, shared_shader_capture,
                             error, rtt_reader) &&
              shared_shader_capture.draws.size() == 1 &&
              shared_shader_capture.draws[0].vs == draw.vs &&
              shared_shader_capture.draws[0].fs == draw.fs,
          "live capture materializes shared shader words without dropping SPIR-V");

    // #636: a descriptor-looking scalar quartet with no matching MUBUF instruction is not a compute
    // resource. The capture path must therefore neither probe its guest address nor retain a blob.
    uint32_t scalar_sgprs[32] = {};
    scalar_sgprs[0] = 0x00100000u;
    scalar_sgprs[2] = 16u;
    scalar_sgprs[3] = (2u << 12) | 0xFACu;
    AgcShaderUserData scalar_user_data{};
    AgcShaderHeader scalar_header{};
    scalar_header.file_header = 0x34333231u;
    scalar_header.version = 0x18;
    scalar_header.type = 0;
    scalar_header.user_data = &scalar_user_data;
    auto scalar_table = std::make_shared<ShaderResourceTable>(
        build_shader_resources(scalar_header, scalar_sgprs, 32));
    CHECK(scalar_table->resources.empty(),
          "#636: scalar-only compute user data produces no capture-table resource");
    ComputeItem scalar_compute;
    scalar_compute.spirv = {0x07230203u};
    scalar_compute.resources = scalar_table;
    scalar_compute.dispatch_index = 636;
    scalar_compute.command_order = 636;
    const std::vector<SubmitOperation> scalar_operations = {
        {SubmitOperationKind::Dispatch, 636, 636},
    };
    size_t scalar_reader_calls = 0;
    auto scalar_reader = [&](uint64_t, uint8_t*, size_t) -> size_t {
        ++scalar_reader_calls;
        return 0;
    };
    GpuCaptureFile scalar_capture;
    CHECK(capture_submit_items({}, {scalar_compute}, scalar_operations, meta,
                               scalar_reader, scalar_capture, error) &&
              scalar_reader_calls == 0 && scalar_capture.blobs.empty() &&
              scalar_capture.computes.size() == 1 &&
              scalar_capture.computes[0].resources.resources.empty(),
          "#636: capture ignores unconsumed descriptor-looking scalar arguments");

    // v29 (#1349): byte length of the depth-bias tail (u8 enable + three u32 per pipeline).
    auto v29_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t present_failures = 0;
        for (const auto& diagnostic : f.failure_diagnostics)
            if (diagnostic.pipeline_present) ++present_failures;
        return 4 + 13 * f.draws.size() + 4 + 13 * present_failures;
    };

    // v27: byte length of the trailing draw-modifier/vertex-offset block.
    auto v27_tail = [](const GpuCaptureFile& f) -> size_t {
        return 4 + 12 * f.draws.size();
    };

    // v28: byte length of the linear-layout tail (u32 count + u32 pitch/u64 span per resource).
    auto v28_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t rc = 0;
        for (const auto& d : f.draws) rc += d.vrt.resources.size() + d.prt.resources.size();
        for (const auto& cm : f.computes) rc += cm.resources.resources.size();
        return 4 + 12 * rc;
    };

    // v26: byte length of the trailing color-export/downconversion block.
    auto v26_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t present_failures = 0;
        for (const auto& diagnostic : f.failure_diagnostics)
            if (diagnostic.pipeline_present) ++present_failures;
        return 4 + 8 * f.draws.size() + 4 + 8 * present_failures;
    };

    // v30: byte length of the per-resource fetch-index-mode tail.
    auto v30_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t rc = 0;
        for (const auto& d : f.draws) rc += d.vrt.resources.size() + d.prt.resources.size();
        for (const auto& cm : f.computes) rc += cm.resources.resources.size();
        return 4 + 4 * rc;
    };

    // v31: byte length of the linked-vertex raw index and graphics-LDS tail.
    auto v31_tail = [](const GpuCaptureFile& f) -> size_t {
        return 4 + 8 * f.draws.size();
    };

    // v32: byte length of the per-resource array-layer stride/selected-mip-offset tail.
    auto v32_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t rc = 0;
        for (const auto& d : f.draws) rc += d.vrt.resources.size() + d.prt.resources.size();
        for (const auto& cm : f.computes) rc += cm.resources.resources.size();
        return 4 + 8 * rc;
    };

    // v34: six additional color-target identities and complete fixed-function records per draw.
    auto v34_tail = [](const GpuCaptureFile& f) -> size_t {
        constexpr size_t color_target_pipeline_bytes = 51;
        size_t bytes = 4 + f.draws.size() * 6 * (16 + color_target_pipeline_bytes) + 4;
        for (const auto& diagnostic : f.failure_diagnostics) {
            bytes += 6 * 16;
            if (diagnostic.pipeline_present) bytes += 6 * color_target_pipeline_bytes;
        }
        return bytes;
    };

    // v35: failed-stage resource tables use the byte-exact base resource record.
    auto v35_tail = [](const GpuCaptureFile& f) -> size_t {
        constexpr size_t resource_bytes = 145;
        size_t bytes = 4;
        for (const auto& diagnostic : f.failure_diagnostics) {
            bytes += 4;
            for (const auto& stage : diagnostic.stages)
                bytes += 5 + resource_bytes * stage.resource_table.resources.size();
        }
        return bytes;
    };

    // v36: conditional fixed-function pixel-stage ABI per realized draw.
    auto v36_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t bytes = 4;
        for (const auto& draw : f.draws) {
            bytes += 1;
            if (draw.has_pixel_inputs) bytes += 8 + 32 * 4;
            if (draw.has_system_inputs) bytes += 8;
        }
        return bytes;
    };

    // v37: one exact required-subgroup size per realized compute.
    auto v37_tail = [](const GpuCaptureFile& f) -> size_t {
        return 4 + f.computes.size() * 4;
    };

    // v39: raw realized-compute index plus the optional semantic recompile state.
    auto v39_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t bytes = 4;
        for (const auto& compute : f.computes) {
            bytes += 5; // available + raw shader index
            if (compute.recompile_config_available)
                bytes += 50 + compute.recompile_config.user_sgprs.size() * 4;
        }
        return bytes;
    };

    // v40: one BOX_GROW value per realized and failed-stage resource.
    auto v40_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t rc = 0;
        for (const auto& d : f.draws) rc += d.vrt.resources.size() + d.prt.resources.size();
        for (const auto& cm : f.computes) rc += cm.resources.resources.size();
        for (const auto& diagnostic : f.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                rc += stage.resource_table.resources.size();
        return 4 + 4 * rc;
    };

    // v41: complete versioned descriptor/codegen state per failed-stage resource.
    auto v41_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t rc = 0;
        for (const auto& diagnostic : f.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                rc += stage.resource_table.resources.size();
        return 4 + 71 * rc;
    };

    // v42: optional exact compute specialization per failed stage.
    auto v42_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t bytes = 4;
        for (const auto& diagnostic : f.failure_diagnostics) {
            bytes += 4;
            for (const auto& stage : diagnostic.stages) {
                bytes += 1;
                if (stage.recompile_config_available)
                    bytes += 50 + stage.recompile_config.user_sgprs.size() * 4;
            }
        }
        return bytes;
    };

    // v43 (#1459): byte length of the trailing raw color-state block. One present-bitmask byte plus
    // three raw registers per realized draw and per pipeline-bearing failure diagnostic.
    auto v43_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t present_failures = 0;
        for (const auto& diagnostic : f.failure_diagnostics)
            if (diagnostic.pipeline_present) ++present_failures;
        return 4 + f.draws.size() * 13 + 4 + present_failures * 13;
    };

    // v44: one guest sample count per realized and failed-stage resource.
    auto v44_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t rc = 0;
        for (const auto& d : f.draws) rc += d.vrt.resources.size() + d.prt.resources.size();
        for (const auto& cm : f.computes) rc += cm.resources.resources.size();
        for (const auto& diagnostic : f.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                rc += stage.resource_table.resources.size();
        return 4 + 4 * rc;
    };

    // v45: count plus one fixed-size selected resource-provenance record. The sampled bytes live in
    // ordinary capture blobs, so the tail itself remains fixed-size per record.
    auto v45_tail = [](const GpuCaptureFile& f) -> size_t {
        return 4 + 73 * f.resource_provenance.size();
    };

    // v46: count plus one 100-byte raw input/provenance witness per selected v45 record.
    auto v46_tail = [](const GpuCaptureFile& f) -> size_t {
        return 4 + 100 * f.resource_provenance.size();
    };

    // v25 (#1240): byte length of the trailing resolve-state block.
    auto v25_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t present_failures = 0;
        for (const auto& diagnostic : f.failure_diagnostics)
            if (diagnostic.pipeline_present) ++present_failures;
        return 4 + f.draws.size() + 4 + present_failures;
    };

    // v24 (#1280): byte length of the declared-mip-levels tail (u32 count + one u32 per resource).
    auto v24_tail = [](const GpuCaptureFile& f) -> size_t {
        size_t rc = 0;
        for (const auto& d : f.draws) rc += d.vrt.resources.size() + d.prt.resources.size();
        for (const auto& cm : f.computes) rc += cm.resources.resources.size();
        return 4 + 4 * rc;
    };

    // v17 appends scissor state without changing the legacy pipeline prefix. Remove v24's declared-mip
    // tail, v22's internal resource state, v21's logic op,
    // v20's instance count, v19's realized raw-stage indices, v18's geometry index, then the v17 tail
    // to recover a byte-exact v16 capture.
    GpuCaptureFile v16_scissor_source = captured;
    v16_scissor_source.raw_shader_versions.clear();
    v16_scissor_source.draws[0].vs_raw_shader_index = 0xFFFFFFFFu;
    v16_scissor_source.draws[0].fs_raw_shader_index = 0xFFFFFFFFu;
    v16_scissor_source.draws[0].vs_chain_raw_shader_index = 0xFFFFFFFFu;
    std::vector<uint8_t> v16_scissor_bytes;
    GpuCaptureFile v16_scissor_loaded;
    CHECK(serialize_gpu_capture(v16_scissor_source, v16_scissor_bytes, error) &&
              v16_scissor_bytes.size() >= 25,
          "v17 capture serializes effective guest scissor state");
    if (v16_scissor_bytes.size() >= 25) {
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v50_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v49_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v48_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v47_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v46_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v45_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v44_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v43_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v42_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v41_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v40_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v39_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v37_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v36_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v35_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v34_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v32_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v31_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v30_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v29_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v28_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v27_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v26_tail(v16_scissor_source));
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v25_tail(v16_scissor_source)); // v25 resolve tail
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - v24_tail(v16_scissor_source)); // v24 declared-mip tail
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - 12); // v23 count + one raw-draw-state
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - 28); // v22 count + three empty vectors
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - 13); // v21 one draw + zero failures
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - 8); // v20 draw count + instance count
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - 12); // v19 count + two raw indices
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - 8); // v18 draw count + no-GS sentinel
        v16_scissor_bytes.resize(v16_scissor_bytes.size() - 25);
        v16_scissor_bytes[8] = 16;
        v16_scissor_bytes[9] = v16_scissor_bytes[10] = v16_scissor_bytes[11] = 0;
    }
    CHECK(deserialize_gpu_capture(v16_scissor_bytes, v16_scissor_loaded, error) &&
              !v16_scissor_loaded.draws[0].ps.has_scissor,
          "v16 capture reopens with the historical full-target scissor default");

    // v16: a packed mip-tail view captures the shared macroblock and retains both AddrLib origins.
    std::vector<uint8_t> tail_memory(65536, 0x6d);
    auto tail_reader = [&](uint64_t addr, uint8_t* dst, size_t n) -> size_t {
        if (addr < 0x900000 || addr >= 0x900000 + tail_memory.size()) return 0;
        const size_t offset = static_cast<size_t>(addr - 0x900000);
        const size_t take = std::min(n, tail_memory.size() - offset);
        std::memcpy(dst, tail_memory.data() + offset, take);
        return take;
    };
    ShaderResource tail_resource{};
    tail_resource.cls = ResourceClass::Texture; tail_resource.binding = 3;
    tail_resource.gpu_addr = 0x900000; tail_resource.size = 60 * 33 * 4;
    tail_resource.format = DataFormat::Uint32; tail_resource.num_components = 1;
    tail_resource.width = 60; tail_resource.height = 33;
    tail_resource.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbRX);
    tail_resource.in_mip_tail = true; tail_resource.mip_tail_offset = 32768;
    tail_resource.mip_tail_bytes = 65536; tail_resource.mip_tail_x = 64;
    tail_resource.mip_tail_y = 0;
    tail_resource.declared_mip_levels = 5;   // #1280: a non-default T#-declared mip-chain length
    auto tail_table = std::make_shared<ShaderResourceTable>();
    tail_table->resources = {tail_resource};
    DrawItem tail_draw = draw; tail_draw.vrt = tail_table;
    GpuCaptureFile tail_capture;
    CHECK(capture_draw_items({tail_draw}, meta, tail_reader, tail_capture, error) &&
              tail_capture.blobs.size() == 1 && tail_capture.blobs[0].bytes.size() == 65536,
          "packed mip-tail capture owns the complete shared 64 KiB macroblock");
    std::vector<uint8_t> tail_bytes;
    GpuCaptureFile tail_loaded;
    CHECK(serialize_gpu_capture(tail_capture, tail_bytes, error) &&
              deserialize_gpu_capture(tail_bytes, tail_loaded, error) &&
              tail_loaded.draws[0].vrt.resources[0].resource.in_mip_tail &&
              tail_loaded.draws[0].vrt.resources[0].resource.mip_tail_offset == 32768 &&
              tail_loaded.draws[0].vrt.resources[0].resource.mip_tail_bytes == 65536 &&
              tail_loaded.draws[0].vrt.resources[0].resource.mip_tail_x == 64 &&
              tail_loaded.draws[0].vrt.resources[0].resource.mip_tail_y == 0,
          "v16 capture round-trips packed-tail byte and coordinate placement exactly");
    CHECK(tail_loaded.draws[0].vrt.resources[0].resource.declared_mip_levels == 5,
          "#1280: v24 capture round-trips the T#-declared mip-chain length (was silently defaulting to 1)");

    ShaderResource msaa_resource{};
    msaa_resource.cls = ResourceClass::Texture; msaa_resource.binding = 4;
    msaa_resource.gpu_addr = 0x900000; msaa_resource.size = 65536;
    msaa_resource.format = DataFormat::Float32; msaa_resource.num_components = 1;
    msaa_resource.img_dim = 6; msaa_resource.width = 64; msaa_resource.height = 64;
    msaa_resource.sample_count = 4;
    msaa_resource.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbZX);
    msaa_resource.compression_enabled = true;
    msaa_resource.meta_pipe_aligned = true;
    msaa_resource.metadata_addr = 0xa00000;
    // Exact title-live depth-zero fast-clear metadata. Keep it in the ordinary metadata blob: an
    // offline replay must be able to materialize +0 sample planes without process guest memory.
    std::vector<uint32_t> msaa_htile(32768u / sizeof(uint32_t), 0u);
    auto msaa_reader = [&](uint64_t addr, uint8_t* dst, size_t n) -> size_t {
        if (addr >= msaa_resource.metadata_addr &&
            addr < msaa_resource.metadata_addr + 32768u) {
            const size_t offset = static_cast<size_t>(addr - msaa_resource.metadata_addr);
            const size_t take = std::min(n, 32768u - offset);
            std::memcpy(dst, reinterpret_cast<const uint8_t*>(msaa_htile.data()) + offset,
                        take);
            return take;
        }
        return tail_reader(addr, dst, n);
    };
    auto msaa_table = std::make_shared<ShaderResourceTable>();
    msaa_table->resources = {msaa_resource};
    DrawItem msaa_draw = draw; msaa_draw.vrt = msaa_table;
    GpuCaptureFile msaa_capture;
    CHECK(capture_draw_items({msaa_draw}, meta, msaa_reader, msaa_capture, error) &&
              msaa_capture.blobs.size() == 2 &&
              msaa_capture.blobs[0].bytes.size() == 65536 &&
              msaa_capture.draws[0].vrt.resources[0].metadata_size == 32768u &&
              msaa_capture.draws[0].vrt.resources[0].metadata_blob_index == 1u &&
              msaa_capture.blobs[1].guest_addr == msaa_resource.metadata_addr &&
              msaa_capture.blobs[1].bytes.size() == 32768u,
          "2D_MSAA capture owns the tiled samples and exact HTILE metadata span");
    std::vector<uint8_t> msaa_bytes;
    GpuCaptureFile msaa_loaded;
    const bool msaa_serialized = serialize_gpu_capture(msaa_capture, msaa_bytes, error);
    if (!msaa_serialized) std::printf("  [diag] MSAA capture serialization: %s\n", error.c_str());
    const bool msaa_deserialized = msaa_serialized &&
        deserialize_gpu_capture(msaa_bytes, msaa_loaded, error);
    if (!msaa_deserialized) std::printf("  [diag] MSAA capture deserialization: %s\n", error.c_str());
    CHECK(msaa_deserialized &&
              msaa_loaded.format_version == 50 &&
              msaa_loaded.draws[0].vrt.resources[0].resource.sample_count == 4 &&
              msaa_loaded.blobs.size() == 2 &&
              msaa_loaded.blobs[1].bytes.size() == 32768u &&
              std::all_of(msaa_loaded.blobs[1].bytes.begin(),
                          msaa_loaded.blobs[1].bytes.end(),
                          [](uint8_t value) { return value == 0u; }),
          "v44 capture round-trips guest 2D_MSAA sample identity and self-contained "
          "depth-zero HTILE");
    GpuReplayFrame msaa_replay;
    const bool msaa_materialized = msaa_deserialized &&
        materialize_gpu_replay(msaa_loaded, msaa_replay, error);
    const ShaderResource* replay_msaa_resource =
        msaa_materialized && !msaa_replay.items.empty() && msaa_replay.items[0].vrt &&
                !msaa_replay.items[0].vrt->resources.empty()
            ? &msaa_replay.items[0].vrt->resources[0]
            : nullptr;
    CHECK(replay_msaa_resource && replay_msaa_resource->dcc_metadata_host_data &&
              replay_msaa_resource->dcc_metadata_host_data_size == 32768u &&
              std::all_of(replay_msaa_resource->dcc_metadata_host_data,
                          replay_msaa_resource->dcc_metadata_host_data +
                              replay_msaa_resource->dcc_metadata_host_data_size,
                          [](uint8_t value) { return value == 0u; }),
          "offline replay owns the complete zero-HTILE plane needed for fast-clear materialization");
    std::vector<uint8_t> mismatched_msaa_metadata = msaa_bytes;
    const uint32_t two_samples = 2;
    std::memcpy(mismatched_msaa_metadata.data() + mismatched_msaa_metadata.size() -
                    v50_tail(msaa_capture) - v49_tail(msaa_capture) -
                    v48_tail(msaa_capture) -
                    v47_tail(msaa_capture) - v46_tail(msaa_capture) -
                    v45_tail(msaa_capture) - 4u,
                &two_samples, sizeof(two_samples));
    GpuCaptureFile mismatched_msaa_loaded;
    CHECK(!deserialize_gpu_capture(
                  mismatched_msaa_metadata, mismatched_msaa_loaded, error) &&
              error == "invalid resource DCC metadata reference",
          "v44 reader validates deferred HTILE size after reading append-only sample identity");
    // v43 predates sample-count identity and therefore cannot carry a 2D_MSAA HTILE footprint.
    // Build its compatibility fixture from the same resource with compression metadata explicitly
    // unavailable, then remove only the v44 sample tail as an actual old writer would have done.
    GpuCaptureFile v43_msaa_source = msaa_capture;
    auto& v43_msaa_captured = v43_msaa_source.draws[0].vrt.resources[0];
    v43_msaa_captured.resource.compression_enabled = false;
    v43_msaa_captured.resource.meta_pipe_aligned = false;
    v43_msaa_captured.resource.metadata_addr = 0;
    v43_msaa_captured.resource.dcc_metadata_size = 0;
    v43_msaa_captured.metadata_size = 0;
    v43_msaa_captured.metadata_blob_index = 0xffffffffu;
    v43_msaa_captured.metadata_blob_offset = 0;
    v43_msaa_source.blobs.resize(1);
    std::vector<uint8_t> v43_msaa_bytes;
    CHECK(serialize_gpu_capture(v43_msaa_source, v43_msaa_bytes, error),
          "current writer prepares an uncompressed v43 MSAA compatibility fixture");
    v43_msaa_bytes.resize(v43_msaa_bytes.size() - v50_tail(v43_msaa_source));
    v43_msaa_bytes.resize(v43_msaa_bytes.size() - v49_tail(v43_msaa_source));
    v43_msaa_bytes.resize(v43_msaa_bytes.size() - v48_tail(v43_msaa_source));
    v43_msaa_bytes.resize(v43_msaa_bytes.size() - v47_tail(v43_msaa_source));
    v43_msaa_bytes.resize(v43_msaa_bytes.size() - v46_tail(v43_msaa_source));
    v43_msaa_bytes.resize(v43_msaa_bytes.size() - v45_tail(v43_msaa_source));
    v43_msaa_bytes.resize(v43_msaa_bytes.size() - v44_tail(v43_msaa_source));
    v43_msaa_bytes[8] = 43;
    v43_msaa_bytes[9] = v43_msaa_bytes[10] = v43_msaa_bytes[11] = 0;
    GpuCaptureFile v43_msaa_loaded;
    CHECK(deserialize_gpu_capture(v43_msaa_bytes, v43_msaa_loaded, error) &&
              v43_msaa_loaded.draws[0].vrt.resources[0].resource.sample_count == 1,
          "v43 capture reopens with the historical single-sample default");
    // Keep this arm metadata-independent so the observed rejection is specifically the malformed
    // sample count, not the necessarily-unavailable HTILE span for an unsupported 3xaa shape.
    GpuCaptureFile invalid_msaa_capture = v43_msaa_source;
    invalid_msaa_capture.draws[0].vrt.resources[0].resource.sample_count = 3;
    CHECK(!serialize_gpu_capture(invalid_msaa_capture, msaa_bytes, error) &&
              error == "invalid resource sample count",
          "capture rejects a non-power-of-two guest sample count");

    auto large_table = std::make_shared<ShaderResourceTable>();
    ShaderResource large_resource = a;
    large_resource.size = 2u << 20;
    CHECK(gpu_capture_resource_footprint(large_resource) == (2u << 20),
          "public capture footprint reports the planner's declared buffer range");
    ShaderResource video_chroma;
    video_chroma.cls = ResourceClass::Texture;
    video_chroma.format = DataFormat::Unorm8;
    video_chroma.num_components = 1;
    video_chroma.img_dim = 1;
    video_chroma.width = 1920;
    video_chroma.height = 1080;
    video_chroma.tile_mode = static_cast<uint32_t>(TileMode::Linear);
    video_chroma.size = 1920u * 1080u;
    CHECK(gpu_capture_resource_footprint(video_chroma) == 2048u * 1080u,
          "capture includes 256-byte row padding for a guest-backed linear R8 sampled image");
    uint8_t host_video_texel = 0;
    video_chroma.host_data = &host_video_texel;
    CHECK(gpu_capture_resource_footprint(video_chroma) == 1920u * 1080u,
          "capture keeps tightly-packed host texture fixtures unpadded");
    video_chroma.host_data = nullptr;
    video_chroma.gpu_addr = 0xa00000;
    video_chroma.height = 2;
    video_chroma.size = 1920u * 2u;
    auto video_table = std::make_shared<ShaderResourceTable>();
    video_table->resources = {video_chroma};
    DrawItem video_draw = draw;
    video_draw.vrt.reset();
    video_draw.prt = video_table;
    std::vector<uint8_t> video_memory(2048u * 2u, 0x80);
    auto video_reader = [&](uint64_t addr, uint8_t* dst, size_t n) -> size_t {
        if (addr < video_chroma.gpu_addr ||
            addr >= video_chroma.gpu_addr + video_memory.size()) return 0;
        const size_t offset = static_cast<size_t>(addr - video_chroma.gpu_addr);
        const size_t take = std::min(n, video_memory.size() - offset);
        std::memcpy(dst, video_memory.data() + offset, take);
        return take;
    };
    GpuCaptureFile video_capture;
    CHECK(capture_draw_items({video_draw}, meta, video_reader, video_capture, error) &&
              video_capture.format_version == 50 && video_capture.blobs.size() == 1 &&
              video_capture.blobs[0].bytes.size() == video_memory.size() &&
              video_capture.draws[0].prt.resources[0].captured_size == video_memory.size() &&
              video_capture.draws[0].prt.resources[0].resource.linear_row_pitch_bytes == 2048,
          "v28 capture retains the complete padded rows and their resolved byte pitch");
    video_capture.draws[0].prt.resources[0].resource.proven_zero_mip = true;
    std::vector<uint8_t> video_capture_bytes;
    GpuCaptureFile video_loaded;
    GpuReplayFrame video_replay;
    CHECK(serialize_gpu_capture(video_capture, video_capture_bytes, error) &&
              deserialize_gpu_capture(video_capture_bytes, video_loaded, error) &&
              video_loaded.format_version == 50 &&
              video_loaded.draws[0].prt.resources[0].resource.proven_zero_mip &&
              video_loaded.draws[0].prt.resources[0].captured_size == video_memory.size() &&
              video_loaded.draws[0].prt.resources[0].resource.linear_row_pitch_bytes == 2048 &&
              materialize_gpu_replay(video_loaded, video_replay, error) &&
              video_replay.items[0].prt->resources[0].host_data_size == video_memory.size(),
          "v28 replay round-trips and binds the complete pitch-padded source span");
    std::vector<uint8_t> v46_zero_mip_bytes = video_capture_bytes;
    v46_zero_mip_bytes.resize(v46_zero_mip_bytes.size() - v50_tail(video_capture));
    v46_zero_mip_bytes.resize(v46_zero_mip_bytes.size() - v49_tail(video_capture));
    v46_zero_mip_bytes.resize(v46_zero_mip_bytes.size() - v48_tail(video_capture));
    v46_zero_mip_bytes.resize(v46_zero_mip_bytes.size() - v47_tail(video_capture));
    v46_zero_mip_bytes[8] = 46;
    GpuCaptureFile v46_zero_mip_loaded;
    CHECK(deserialize_gpu_capture(v46_zero_mip_bytes, v46_zero_mip_loaded, error) &&
              !v46_zero_mip_loaded.draws[0].prt.resources[0].resource.proven_zero_mip,
          "v46 capture reopens without manufacturing a zero-mip specialization proof");
    std::vector<uint8_t> malformed_zero_mip_bytes = video_capture_bytes;
    malformed_zero_mip_bytes[malformed_zero_mip_bytes.size() -
                             v50_tail(video_capture) - v49_tail(video_capture) -
                             v48_tail(video_capture) - 1u] = 2;
    GpuCaptureFile malformed_zero_mip_loaded;
    CHECK(!deserialize_gpu_capture(
              malformed_zero_mip_bytes, malformed_zero_mip_loaded, error) &&
              error == "invalid resource zero-mip state",
          "v47 capture rejects a non-boolean zero-mip specialization marker");
    GpuCaptureFile legacy_video = video_capture;
    legacy_video.format_version = 27;
    legacy_video.draws[0].prt.resources[0].captured_size = 0;
    legacy_video.draws[0].prt.resources[0].resource.linear_row_pitch_bytes = 0;
    legacy_video.blobs[0].bytes.resize(video_chroma.size);
    legacy_video.blobs[0].bytes_read = legacy_video.blobs[0].bytes.size();
    legacy_video.blobs[0].content_hash = gpu_capture_hash(legacy_video.blobs[0].bytes);
    GpuReplayFrame legacy_video_replay;
    CHECK(materialize_gpu_replay(legacy_video, legacy_video_replay, error) &&
              legacy_video_replay.items[0].prt->resources[0].host_data_size == video_chroma.size &&
              legacy_video_replay.items[0].prt->resources[0].linear_row_pitch_bytes == 2048,
          "v27 replay keeps its tight captured span readable while deriving the guest row pitch");
    std::vector<uint8_t> upgraded_video_bytes;
    GpuCaptureFile upgraded_video;
    GpuReplayFrame upgraded_video_replay;
    CHECK(serialize_gpu_capture(legacy_video, upgraded_video_bytes, error) &&
              deserialize_gpu_capture(upgraded_video_bytes, upgraded_video, error) &&
              upgraded_video.format_version == 50 &&
              upgraded_video.draws[0].prt.resources[0].captured_size == video_chroma.size &&
              upgraded_video.draws[0].prt.resources[0].resource.linear_row_pitch_bytes == 2048 &&
              materialize_gpu_replay(upgraded_video, upgraded_video_replay, error) &&
              upgraded_video_replay.items[0].prt->resources[0].host_data_size == video_chroma.size,
          "rewriting a v27 capture preserves its short span and best-effort derived pitch in v28");
    register_guest_linear_texture_layout(video_chroma.gpu_addr, video_chroma.size, 1920);
    GpuCaptureFile tight_video_capture;
    CHECK(gpu_capture_resource_footprint(video_chroma) == video_chroma.size &&
              capture_draw_items({video_draw}, meta, video_reader, tight_video_capture, error) &&
              tight_video_capture.blobs.size() == 1 &&
              tight_video_capture.blobs[0].bytes.size() == video_chroma.size &&
              tight_video_capture.draws[0].prt.resources[0].captured_size == video_chroma.size &&
              tight_video_capture.draws[0].prt.resources[0].resource.linear_row_pitch_bytes == 1920,
          "capture preserves an exact registered tight pitch for an HLE-produced guest texture");
    unregister_guest_linear_texture_layout(video_chroma.gpu_addr);
    CHECK(gpu_capture_resource_footprint(video_chroma) == 2048u * 2u,
          "unregistered guest textures return to the GFX10 aligned-pitch fallback");
    large_table->resources = {large_resource};
    DrawItem large_draw = draw;
    large_draw.vrt = large_table;
    GpuCaptureFile bounded_capture;
    set_test_env("PROSPER_GPU_CAPTURE_MAX_MB", "1");
    CHECK(!capture_draw_items({large_draw}, meta, reader, bounded_capture, error, rtt_reader) &&
          error.find("requires 2 MiB") != std::string::npos &&
          error.find("PROSPER_GPU_CAPTURE_METADATA_ONLY=1") != std::string::npos,
          "resource capture rejects an oversized plan before allocating and reports the thin fallback");
    set_test_env("PROSPER_GPU_CAPTURE_METADATA_ONLY", "1");
    CHECK(capture_draw_items({large_draw}, meta, reader, bounded_capture, error, rtt_reader) &&
          bounded_capture.blobs.empty() && bounded_capture.rtt_seeds.empty() &&
          bounded_capture.draws[0].vrt.resources[0].blob_index == 0xFFFFFFFFu &&
          bounded_capture.metadata.renderer_env.back().first == "PROSPER_GPU_CAPTURE_METADATA_ONLY" &&
          bounded_capture.metadata.renderer_env.back().second == "1",
          "metadata-only capture retains resource descriptors without guest or RTT byte blobs");
    GpuReplayFrame bounded_replay;
    CHECK(materialize_gpu_replay(bounded_capture, bounded_replay, error) &&
          bounded_replay.items[0].vrt->resources[0].gpu_addr == large_resource.gpu_addr &&
          bounded_replay.items[0].vrt->resources[0].host_data == nullptr,
          "metadata-only capture materializes inspectable descriptors with unavailable bytes explicit");
    GpuCaptureFile array_layout_capture = bounded_capture;
    ShaderResource& array_resource =
        array_layout_capture.draws[0].vrt.resources[0].resource;
    array_resource.cls = ResourceClass::StorageImage;
    array_resource.gpu_addr = 0x31465d0000ull;
    array_resource.size = 128u * 128u * 6u * 8u;
    array_resource.format = DataFormat::Float16;
    array_resource.num_components = 4;
    array_resource.img_dim = 5;
    array_resource.width = array_resource.height = 128;
    array_resource.depth = 6;
    array_resource.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbRX);
    array_resource.layer_stride_bytes = 720896;
    array_resource.layer_mip_offset_bytes = 65536;
    std::vector<uint8_t> array_layout_bytes;
    GpuCaptureFile array_layout_loaded;
    CHECK(gpu_capture_resource_footprint(array_resource) == 3801088u,
          "array-mip capture footprint spans the selected level in the final full-chain slice");
    ShaderResource linear_array_resource = array_resource;
    linear_array_resource.width = 65;
    linear_array_resource.height = 19;
    linear_array_resource.depth = 3;
    linear_array_resource.tile_mode = 0;
    linear_array_resource.format = DataFormat::Unorm8;
    linear_array_resource.num_components = 4;
    linear_array_resource.size = 65u * 19u * 3u * 4u;
    linear_array_resource.layer_stride_bytes = 62464;
    linear_array_resource.layer_mip_offset_bytes = 8704;
    CHECK(gpu_capture_resource_footprint(linear_array_resource) ==
              2u * 62464u + 8704u + 512u * 19u,
          "linear array capture includes selected-level row padding in the final slice");
    ShaderResource plucky_volume = array_resource;
    plucky_volume.gpu_addr = 0x3022c70000ull;
    plucky_volume.size = 32u * 32u * 32u * 8u;
    plucky_volume.img_dim = 2;
    plucky_volume.width = plucky_volume.height = plucky_volume.depth = 32;
    plucky_volume.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbS);
    plucky_volume.layer_stride_bytes = 0;
    plucky_volume.layer_mip_offset_bytes = 0;
    CHECK(gpu_capture_resource_footprint(plucky_volume) == 4u * 65536u,
          "Plucky RGBA16 32-cubed S3 capture uses its four true 3D macroblocks");
    CHECK(serialize_gpu_capture(array_layout_capture, array_layout_bytes, error) &&
              deserialize_gpu_capture(array_layout_bytes, array_layout_loaded, error) &&
              array_layout_loaded.format_version == 50 &&
              array_layout_loaded.draws[0].vrt.resources[0].resource.layer_stride_bytes == 720896u &&
              array_layout_loaded.draws[0].vrt.resources[0].resource.layer_mip_offset_bytes == 65536u,
          "v32 capture round-trips thin-array slice stride and selected-mip offset");
    GpuCaptureFile cube_layout_capture = array_layout_capture;
    ShaderResource& cube_resource = cube_layout_capture.draws[0].vrt.resources[0].resource;
    cube_resource.img_dim = 3;
    std::vector<uint8_t> cube_layout_bytes;
    GpuCaptureFile cube_layout_loaded;
    CHECK(gpu_capture_resource_footprint(cube_resource) == 3801088u &&
              serialize_gpu_capture(cube_layout_capture, cube_layout_bytes, error) &&
              deserialize_gpu_capture(cube_layout_bytes, cube_layout_loaded, error) &&
              cube_layout_loaded.draws[0].vrt.resources[0].resource.img_dim == 3u &&
              cube_layout_loaded.draws[0].vrt.resources[0].resource.layer_stride_bytes == 720896u &&
              cube_layout_loaded.draws[0].vrt.resources[0].resource.layer_mip_offset_bytes == 65536u,
          "v32 capture round-trips cube-face stride and selected-mip offset");
    ShaderResource small_tail_cube = cube_resource;
    small_tail_cube.width = small_tail_cube.height = 32;
    small_tail_cube.size = 6u * 65536u;
    small_tail_cube.in_mip_tail = true;
    small_tail_cube.mip_tail_offset = 32768u;
    small_tail_cube.mip_tail_bytes = 65536u;
    small_tail_cube.mip_tail_x = 64u;
    small_tail_cube.mip_tail_y = 0u;
    small_tail_cube.layer_stride_bytes = 65536u;
    small_tail_cube.layer_mip_offset_bytes = 0;
    CHECK(gpu_capture_resource_footprint(small_tail_cube) == 6u * 65536u,
          "cube chain wholly in one tail block captures exactly six complete face blocks");
    set_test_env("PROSPER_GPU_CAPTURE_METADATA_ONLY", nullptr);
    set_test_env("PROSPER_GPU_CAPTURE_MAX_MB", nullptr);

    // --- v12: capture the separate GFX10 DCC control surface for Unreal's R_X volume LUTs. ---
    ShaderResource volume_dcc{};
    volume_dcc.cls = ResourceClass::Texture; volume_dcc.binding = 7;
    volume_dcc.gpu_addr = 0x100000; volume_dcc.size = 32 * 32 * 32 * 4;
    volume_dcc.format = DataFormat::Sint32; volume_dcc.num_components = 1;
    volume_dcc.img_dim = 2; volume_dcc.width = 32; volume_dcc.height = 32;
    volume_dcc.depth = 32; volume_dcc.tile_mode = (uint32_t)TileMode::Sw64KbRX;
    volume_dcc.compression_enabled = true; volume_dcc.meta_pipe_aligned = true;
    volume_dcc.metadata_addr = 0x400000;
    CHECK(gpu_capture_dcc_metadata_footprint(volume_dcc) == 128 * 1024,
          "32x32x32 R_X volume descriptor plans the validated 128 KiB DCC span");
    ShaderResource packed_volume_dcc = volume_dcc;
    packed_volume_dcc.gpu_addr = 0x500000; packed_volume_dcc.size = 40 * 5 * 5 * 4;
    packed_volume_dcc.format = DataFormat::Float10_11_11; packed_volume_dcc.num_components = 3;
    packed_volume_dcc.width = 40; packed_volume_dcc.height = 5; packed_volume_dcc.depth = 5;
    packed_volume_dcc.metadata_addr = 0x600000;
    CHECK(gpu_capture_dcc_metadata_footprint(packed_volume_dcc) == 20 * 1024,
          "40x5x5 packed R11G11B10 volume descriptor plans its 20 KiB DCC span");
    auto volume_table = std::make_shared<ShaderResourceTable>();
    volume_table->resources = {volume_dcc};
    DrawItem volume_draw; volume_draw.vs = {0x07230203, 31};
    volume_draw.fs = {0x07230203, 32}; volume_draw.vrt = volume_table;
    auto volume_reader = [](uint64_t addr, uint8_t* dst, size_t n) -> size_t {
        for (size_t i = 0; i < n; ++i) {
            const uint64_t at = addr + i;
            dst[i] = static_cast<uint8_t>(at ^ (at >> 11) ^ (at >> 23));
        }
        return n;
    };
    GpuCaptureFile volume_capture;
    CHECK(capture_draw_items({volume_draw}, meta, volume_reader, volume_capture, error),
          "capture reads a compressed base allocation and its separate DCC control surface");
    const auto& captured_volume = volume_capture.draws[0].vrt.resources[0];
    CHECK(volume_capture.blobs.size() == 2 && captured_volume.metadata_size == 128 * 1024 &&
          captured_volume.metadata_blob_index != 0xFFFFFFFFu &&
          volume_capture.blobs[captured_volume.metadata_blob_index].guest_addr == 0x400000 &&
          volume_capture.blobs[captured_volume.metadata_blob_index].bytes.size() == 128 * 1024,
          "DCC metadata receives an exact independently-addressed content blob");
    std::vector<uint8_t> volume_bytes;
    CHECK(serialize_gpu_capture(volume_capture, volume_bytes, error),
          "v12 DCC metadata capture serializes");
    GpuCaptureFile volume_loaded;
    GpuReplayFrame volume_replay;
    CHECK(deserialize_gpu_capture(volume_bytes, volume_loaded, error) &&
          materialize_gpu_replay(volume_loaded, volume_replay, error) &&
          volume_replay.items[0].vrt->resources[0].dcc_metadata_host_data &&
          volume_replay.items[0].vrt->resources[0].dcc_metadata_host_data_size == 128 * 1024 &&
          volume_replay.resource_instances.size() == 2,
          "v12 replay owns base and DCC bytes as distinct mutable address instances");
    auto packed_volume_table = std::make_shared<ShaderResourceTable>();
    packed_volume_table->resources = {packed_volume_dcc};
    DrawItem packed_volume_draw = volume_draw; packed_volume_draw.vrt = packed_volume_table;
    GpuCaptureFile packed_volume_capture;
    CHECK(capture_draw_items({packed_volume_draw}, meta, volume_reader,
                             packed_volume_capture, error) &&
          packed_volume_capture.blobs.size() == 2 &&
          packed_volume_capture.draws[0].vrt.resources[0].metadata_size == 20 * 1024 &&
          packed_volume_capture.blobs[
              packed_volume_capture.draws[0].vrt.resources[0].metadata_blob_index].bytes.size() ==
              20 * 1024,
          "capture retains packed R11G11B10 volume DCC bytes as a separate content blob");
    GpuCaptureFile bad_metadata_ref = volume_capture;
    bad_metadata_ref.draws[0].vrt.resources[0].metadata_blob_offset = 128 * 1024;
    CHECK(!serialize_gpu_capture(bad_metadata_ref, volume_bytes, error) &&
          error == "resource DCC metadata exceeds its capture blob",
          "writer rejects an out-of-bounds DCC metadata reference");
    CHECK(serialize_gpu_capture(volume_capture, volume_bytes, error),
          "recreated valid v12 DCC bytes after malformed-reference check");
    volume_bytes.resize(volume_bytes.size() - v50_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v49_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v48_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v47_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v46_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v45_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v44_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v43_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v42_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v41_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v40_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v39_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v37_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v36_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v35_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v34_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v32_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v31_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v30_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v29_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v28_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v27_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v26_tail(volume_capture));
    volume_bytes.resize(volume_bytes.size() - v25_tail(volume_capture)); // v25 resolve tail
    volume_bytes.resize(volume_bytes.size() - v24_tail(volume_capture)); // v24 declared-mip tail
    volume_bytes.resize(volume_bytes.size() - 12); // v23 count + one raw-draw-state
    volume_bytes.resize(volume_bytes.size() - 12); // v22 count + one empty vector
    volume_bytes.resize(volume_bytes.size() - 13); // v21 one logic-op draw + zero failures
    volume_bytes.resize(volume_bytes.size() - 8); // v20 draw count + instance count
    volume_bytes.resize(volume_bytes.size() - 12); // v19 count + two raw indices
    volume_bytes.resize(volume_bytes.size() - 8); // v18 draw count + no-GS sentinel
    volume_bytes.resize(volume_bytes.size() - 25); // v17 one-draw scissor tail, zero failures
    volume_bytes.resize(volume_bytes.size() - 21); // v16 count plus one 17-byte mip-tail state
    volume_bytes.resize(volume_bytes.size() - 5); // v15 count plus one depth-compare flag
    volume_bytes.resize(volume_bytes.size() - 4); // v14 zero-DMA count
    volume_bytes.resize(volume_bytes.size() - 24); // v12 count plus one 20-byte reference
    volume_bytes[8] = 11; volume_bytes[9] = volume_bytes[10] = volume_bytes[11] = 0;
    CHECK(deserialize_gpu_capture(volume_bytes, volume_loaded, error) &&
          volume_loaded.draws[0].vrt.resources[0].metadata_size == 128 * 1024 &&
          volume_loaded.draws[0].vrt.resources[0].metadata_blob_index == 0xFFFFFFFFu,
          "v11 capsule reopens with the DCC span derivable but metadata bytes explicitly unavailable");
    set_test_env("PROSPER_GPU_CAPTURE_METADATA_ONLY", "1");
    CHECK(capture_draw_items({volume_draw}, meta, volume_reader, volume_capture, error) &&
          volume_capture.blobs.empty() &&
          volume_capture.draws[0].vrt.resources[0].metadata_size == 128 * 1024 &&
          volume_capture.draws[0].vrt.resources[0].metadata_blob_index == 0xFFFFFFFFu,
          "metadata-only capture retains the planned DCC span without pretending bytes exist");
    set_test_env("PROSPER_GPU_CAPTURE_METADATA_ONLY", nullptr);

    captured.expected_output_valid = true;
    captured.expected_output_hash = 0x1122334455667788ull; captured.expected_output_bytes = 480ull * 270 * 4;
    GpuCaptureDsSeed ds_seed;
    ds_seed.depth_read_base = 0x810000; ds_seed.depth_write_base = 0x810000;
    ds_seed.stencil_read_base = 0x820000; ds_seed.stencil_write_base = 0x820000;
    ds_seed.htile_data_base = 0x800000; ds_seed.width = 2; ds_seed.height = 2;
    ds_seed.format = GpuCaptureDsFormat::D32FloatS8;
    ds_seed.depth_valid = true; ds_seed.stencil_valid = true;
    ds_seed.depth.assign(16, 0x31); ds_seed.stencil = {1, 2, 3, 4};
    captured.ds_seeds.push_back(ds_seed);
    captured.draws[0].vrt.resources[0].resource.depth = 7;

    auto path = prosper_test::test_scratch_dir() / "prosper_gpu_capture_test.prgcap";
    GpuCaptureFile duplicate_seed = captured; duplicate_seed.rtt_seeds.push_back(captured.rtt_seeds[0]);
    CHECK(!write_gpu_capture(path.string(), duplicate_seed, error) && error == "duplicate RTT seed address",
          "writer rejects duplicate temporal RTT seed addresses");
    GpuCaptureFile duplicate_ds = captured; duplicate_ds.ds_seeds.push_back(ds_seed);
    CHECK(!write_gpu_capture(path.string(), duplicate_ds, error) && error == "duplicate DS seed identity",
          "writer rejects duplicate persistent DS seed identities");
    GpuCaptureFile stencil_only = captured;
    stencil_only.ds_seeds[0].depth_valid = false; stencil_only.ds_seeds[0].depth.clear();
    std::vector<uint8_t> stencil_only_bytes;
    GpuCaptureFile stencil_only_loaded;
    CHECK(serialize_gpu_capture(stencil_only, stencil_only_bytes, error) &&
          deserialize_gpu_capture(stencil_only_bytes, stencil_only_loaded, error) &&
          !stencil_only_loaded.ds_seeds[0].depth_valid &&
          stencil_only_loaded.ds_seeds[0].stencil_valid,
          "stencil-only validity survives capture v12 independently of depth");
    GpuCaptureFile invalid_stencil_format = stencil_only;
    invalid_stencil_format.ds_seeds[0].format = GpuCaptureDsFormat::D32Float;
    CHECK(!serialize_gpu_capture(invalid_stencil_format, stencil_only_bytes, error) &&
          error == "DS seed stencil byte count does not match its extent, format, and validity",
          "writer rejects stencil bytes for a depth-only format");
    GpuCaptureFile invalid_dcc = captured;
    invalid_dcc.draws[0].vrt.resources[2].resource.max_compressed_block_size = 4;
    CHECK(!serialize_gpu_capture(invalid_dcc, stencil_only_bytes, error) &&
          error == "invalid resource DCC state",
          "writer rejects DCC block-size values wider than their two-bit descriptor fields");
    CHECK(write_gpu_capture(path.string(), captured, error), "versioned capture writes atomically");
    GpuCaptureFile loaded;
    CHECK(read_gpu_capture(path.string(), loaded, error), "versioned capture reads back");
    // A byte-level pre-v31 fixture cannot retain the new continuation reference. Remove its raw
    // stream as well so the older payload remains a self-contained, valid two-stage capture.
    GpuCaptureFile pre_v31_source = captured;
    const uint32_t removed_chain_index = pre_v31_source.draws[0].vs_chain_raw_shader_index;
    pre_v31_source.draws[0].vs_chain_raw_shader_index = 0xFFFFFFFFu;
    pre_v31_source.draws[0].vertex_lds_dwords = 0;
    if (removed_chain_index < pre_v31_source.raw_shader_versions.size()) {
        pre_v31_source.raw_shader_versions.erase(
            pre_v31_source.raw_shader_versions.begin() + removed_chain_index);
        for (uint32_t* index : {&pre_v31_source.draws[0].vs_raw_shader_index,
                                &pre_v31_source.draws[0].fs_raw_shader_index})
            if (*index > removed_chain_index) --*index;
    }
    std::vector<uint8_t> v20_bytes;
    GpuCaptureFile v20_loaded;
    CHECK(serialize_gpu_capture(pre_v31_source, v20_bytes, error) && v20_bytes.size() >= 21,
          "v21 capture serializes the logic-op extension");
    if (v20_bytes.size() >= 21) {
        v20_bytes.resize(v20_bytes.size() - v50_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v49_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v48_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v47_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v46_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v45_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v44_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v43_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v42_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v41_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v40_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v39_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v37_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v36_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v35_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v34_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v32_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v31_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v30_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v29_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v28_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v27_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v26_tail(pre_v31_source));
        v20_bytes.resize(v20_bytes.size() - v25_tail(pre_v31_source)); // v25 resolve tail
        v20_bytes.resize(v20_bytes.size() - v24_tail(pre_v31_source)); // v24 declared-mip tail
        v20_bytes.resize(v20_bytes.size() - 12); // v23 count + one raw-draw-state
        v20_bytes.resize(v20_bytes.size() - 28); // v22 count + three empty vectors
        v20_bytes.resize(v20_bytes.size() - 13); // v21 one logic-op draw + zero failures
        v20_bytes[8] = 20; v20_bytes[9] = v20_bytes[10] = v20_bytes[11] = 0;
    }
    CHECK(deserialize_gpu_capture(v20_bytes, v20_loaded, error) &&
              !v20_loaded.draws[0].ps.logic_op_enable && v20_loaded.draws[0].ps.logic_op == 3 &&
              v20_loaded.draws[0].instance_count == 3,
          "v20 capture reopens with disabled/COPY logic-op defaults");
    std::vector<uint8_t> v19_bytes = v20_bytes;
    GpuCaptureFile v19_loaded;
    if (v19_bytes.size() >= 12) {
        v19_bytes.resize(v19_bytes.size() - 8); // v20 draw count + one instance count
        v19_bytes[8] = 19; v19_bytes[9] = v19_bytes[10] = v19_bytes[11] = 0;
    }
    CHECK(deserialize_gpu_capture(v19_bytes, v19_loaded, error) &&
              v19_loaded.draws[0].instance_count == 1,
          "v19 capture reopens with the historical one-instance default");
    CHECK(loaded.metadata.submit_index == 42 && loaded.metadata.title_id == "PPSA24651" &&
          loaded.metadata.renderer_env.size() == 2 && loaded.metadata.renderer_env[0].first == "PROSPER_RTT_PERTARGET" &&
          loaded.draws.size() == 1 && loaded.draws[0].indices.size() == 6, "metadata and draw data round-trip");
    CHECK(loaded.draws[0].ps.viewport_h == -1080 && loaded.draws[0].ps.blend_enable,
          "fixed-function pipeline state round-trips explicitly");
    CHECK(loaded.draws[0].ps.has_scissor && loaded.draws[0].ps.scissor_left == 12 &&
          loaded.draws[0].ps.scissor_top == 19 && loaded.draws[0].ps.scissor_right == 70 &&
          loaded.draws[0].ps.scissor_bottom == 75,
          "effective guest scissor round-trips through the v17 extension");
    CHECK(loaded.draws[0].ps.logic_op_enable && loaded.draws[0].ps.logic_op == 6,
          "v21 framebuffer logic op round-trips explicitly");
    CHECK(loaded.draws[0].ps.cb_resolve,
          "v25 MODE=3 resolve intent round-trips explicitly for gpu_replay");
    CHECK(loaded.draws[0].ps.depth_bias_enable == 1 &&
              loaded.draws[0].ps.depth_bias_constant == 4.0f &&
              loaded.draws[0].ps.depth_bias_slope == 2.0f &&
              loaded.draws[0].ps.depth_bias_clamp == 0.5f,
          "v29 depth-bias state round-trips explicitly (#1349)");
    CHECK(loaded.draws[0].ps.spi_shader_col_format == 4 &&
          loaded.draws[0].ps.sx_ps_downconvert == 5,
          "v26 color export/downconversion state round-trips explicitly");
    CHECK(loaded.draws[0].ps.has_cb_color_control &&
              loaded.draws[0].ps.cb_color_control == 0x00CC0010u &&
              loaded.draws[0].ps.has_cb_target_mask &&
              loaded.draws[0].ps.cb_target_mask == 0u &&
              loaded.draws[0].ps.has_cb_shader_mask &&
              loaded.draws[0].ps.cb_shader_mask == 0x000000FFu,
          "v43 raw color-state round-trips a PRESENT zero target mask distinctly (#1459)");
    // v43 is an append-only tail, so a v42 fixture is this capture minus exactly that block. It must
    // reopen reporting the provenance ABSENT rather than fabricating a write-all or zero default.
    {
        auto v43_tail = [](const GpuCaptureFile& f) -> size_t {
            size_t present_failures = 0;
            for (const auto& diagnostic : f.failure_diagnostics)
                if (diagnostic.pipeline_present) ++present_failures;
            return 4 + f.draws.size() * 13 + 4 + present_failures * 13;
        };
        std::vector<uint8_t> v42_bytes;
        CHECK(serialize_gpu_capture(captured, v42_bytes, error),
              "v43 capture serializes before the color-state downgrade");
        v42_bytes.resize(v42_bytes.size() - v50_tail(captured));
        v42_bytes.resize(v42_bytes.size() - v49_tail(captured));
        v42_bytes.resize(v42_bytes.size() - v48_tail(captured));
        v42_bytes.resize(v42_bytes.size() - v47_tail(captured));
        v42_bytes.resize(v42_bytes.size() - v46_tail(captured));
        v42_bytes.resize(v42_bytes.size() - v45_tail(captured));
        v42_bytes.resize(v42_bytes.size() - v44_tail(captured));
        v42_bytes.resize(v42_bytes.size() - v43_tail(captured));
        v42_bytes[8] = 42; v42_bytes[9] = v42_bytes[10] = v42_bytes[11] = 0;
        GpuCaptureFile v42_loaded;
        CHECK(deserialize_gpu_capture(v42_bytes, v42_loaded, error) &&
                  !v42_loaded.draws[0].ps.has_cb_color_control &&
                  !v42_loaded.draws[0].ps.has_cb_target_mask &&
                  !v42_loaded.draws[0].ps.has_cb_shader_mask,
              "v42 capture reopens with color-state provenance absent, not a fabricated default");
    }
    CHECK(loaded.draws[0].raw_draw_modifier == 0x1122334455667788ull &&
          loaded.draws[0].vertex_offset == -37,
          "v27 draw modifier and vertex offset round-trip explicitly");
    CHECK(loaded.draws[0].vrt.resources[0].resource.fetch_index_mode ==
              VertexFetchIndexMode::Instance,
          "v30 fetch-index provenance round-trips explicitly for gpu_replay");
    CHECK(loaded.draws[0].vrt.resources[1].resource.bvh_box_grow == 6 &&
              loaded.draws[0].vrt.resources[1].resource.bvh_sort_enabled,
          "v40/v48 BVH box-grow and sort state round-trip explicitly for raw replay");
    std::vector<uint8_t> v48_bvh_bytes;
    CHECK(serialize_gpu_capture(captured, v48_bvh_bytes, error) &&
              v48_bvh_bytes.size() >=
                  v50_tail(captured) + v49_tail(captured) + v48_tail(captured),
          "v48 BVH sort state has a complete append-only tail");
    v48_bvh_bytes.resize(v48_bvh_bytes.size() - v50_tail(captured));
    v48_bvh_bytes.resize(v48_bvh_bytes.size() - v49_tail(captured));
    v48_bvh_bytes[8] = 48;
    v48_bvh_bytes[9] = v48_bvh_bytes[10] = v48_bvh_bytes[11] = 0;
    std::vector<uint8_t> v47_bvh_bytes = v48_bvh_bytes;
    v47_bvh_bytes.resize(v47_bvh_bytes.size() - v48_tail(captured));
    v47_bvh_bytes[8] = 47;
    v47_bvh_bytes[9] = v47_bvh_bytes[10] = v47_bvh_bytes[11] = 0;
    GpuCaptureFile v47_bvh_loaded;
    CHECK(deserialize_gpu_capture(v47_bvh_bytes, v47_bvh_loaded, error) &&
              v47_bvh_loaded.draws[0].vrt.resources[1].resource.bvh_box_grow == 6 &&
              !v47_bvh_loaded.draws[0].vrt.resources[1].resource.bvh_sort_enabled,
          "v47 capture reopens without manufacturing BVH sorting");
    std::vector<uint8_t> malformed_bvh_sort_bytes = v48_bvh_bytes;
    malformed_bvh_sort_bytes.back() = 2u;
    GpuCaptureFile malformed_bvh_sort_loaded;
    CHECK(!deserialize_gpu_capture(malformed_bvh_sort_bytes,
                                   malformed_bvh_sort_loaded, error) &&
              error == "invalid BVH sort state",
          "v48 capture rejects a non-boolean BVH sort marker");

    GpuCaptureFile atomic_capture = captured;
    atomic_capture.draws[0].vrt.resources[0].resource
        .atomic_x2_record_count = 25u;
    std::vector<uint8_t> current_atomic_bytes;
    std::vector<uint8_t> v49_atomic_bytes;
    GpuCaptureFile v49_atomic_loaded;
    CHECK(serialize_gpu_capture(atomic_capture, current_atomic_bytes, error) &&
              current_atomic_bytes.size() >=
                  v50_tail(atomic_capture) + v49_tail(atomic_capture),
          "current capture has complete v49 and v50 append-only tails");
    v49_atomic_bytes = current_atomic_bytes;
    v49_atomic_bytes.resize(v49_atomic_bytes.size() - v50_tail(atomic_capture));
    v49_atomic_bytes[8] = 49;
    v49_atomic_bytes[9] = v49_atomic_bytes[10] = v49_atomic_bytes[11] = 0;
    CHECK(deserialize_gpu_capture(v49_atomic_bytes, v49_atomic_loaded, error) &&
              v49_atomic_loaded.format_version == 49 &&
              v49_atomic_loaded.draws[0].vrt.resources[0].resource
                  .atomic_x2_record_count == 25u,
          "v49 capture round-trips the exact qword-atomic record count");
    std::vector<uint8_t> v48_atomic_bytes = v49_atomic_bytes;
    v48_atomic_bytes.resize(v48_atomic_bytes.size() - v49_tail(atomic_capture));
    v48_atomic_bytes[8] = 48;
    v48_atomic_bytes[9] = v48_atomic_bytes[10] = v48_atomic_bytes[11] = 0;
    GpuCaptureFile v48_atomic_loaded;
    CHECK(deserialize_gpu_capture(v48_atomic_bytes, v48_atomic_loaded, error) &&
              !v48_atomic_loaded.draws[0].vrt.resources[0].resource
                   .atomic_x2_record_count,
          "v48 capture reopens without manufacturing a qword-atomic descriptor proof");
    std::vector<uint8_t> malformed_atomic_bytes = v49_atomic_bytes;
    malformed_atomic_bytes.pop_back();
    GpuCaptureFile malformed_atomic_loaded;
    CHECK(!deserialize_gpu_capture(malformed_atomic_bytes, malformed_atomic_loaded, error) &&
              error == "invalid atomic-x2 state",
          "v49 capture rejects a truncated qword-atomic record count");
    std::vector<uint8_t> v24_resolve_bytes;
    GpuCaptureFile v24_resolve_loaded;
    CHECK(serialize_gpu_capture(pre_v31_source, v24_resolve_bytes, error) &&
              v24_resolve_bytes.size() >= v50_tail(pre_v31_source) +
                  v49_tail(pre_v31_source) +
                  v48_tail(pre_v31_source) +
                  v47_tail(pre_v31_source) +
                  v46_tail(pre_v31_source) +
              v45_tail(pre_v31_source) +
              v44_tail(pre_v31_source) +
              v43_tail(pre_v31_source) + v42_tail(pre_v31_source) +
              v41_tail(pre_v31_source) +
              v40_tail(pre_v31_source) +
              v39_tail(pre_v31_source) +
              v37_tail(pre_v31_source) +
              v36_tail(pre_v31_source) +
              v35_tail(pre_v31_source) + v34_tail(pre_v31_source) +
              v32_tail(pre_v31_source) +
              v31_tail(pre_v31_source) + v30_tail(pre_v31_source) +
              v29_tail(pre_v31_source) + v28_tail(pre_v31_source) +
              v27_tail(pre_v31_source) + v26_tail(pre_v31_source) +
              v25_tail(pre_v31_source),
          "v25 capture exposes a removable trailing resolve-state block");
    if (v24_resolve_bytes.size() >= v50_tail(pre_v31_source) +
            v49_tail(pre_v31_source) +
            v48_tail(pre_v31_source) + v47_tail(pre_v31_source) +
            v46_tail(pre_v31_source) +
            v45_tail(pre_v31_source) +
            v44_tail(pre_v31_source) +
            v43_tail(pre_v31_source) + v42_tail(pre_v31_source) +
            v41_tail(pre_v31_source) +
            v40_tail(pre_v31_source) +
            v39_tail(pre_v31_source) +
            v37_tail(pre_v31_source) +
            v36_tail(pre_v31_source) +
            v35_tail(pre_v31_source) + v34_tail(pre_v31_source) +
            v32_tail(pre_v31_source) +
            v31_tail(pre_v31_source) + v30_tail(pre_v31_source) +
            v29_tail(pre_v31_source) + v28_tail(pre_v31_source) +
            v27_tail(pre_v31_source) + v26_tail(pre_v31_source) +
            v25_tail(pre_v31_source)) {
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v50_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v49_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v48_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v47_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v46_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v45_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v44_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v43_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v42_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v41_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v40_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v39_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v37_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v36_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v35_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v34_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v32_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v31_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v30_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v29_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v28_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v27_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v26_tail(pre_v31_source));
        v24_resolve_bytes.resize(v24_resolve_bytes.size() - v25_tail(pre_v31_source));
        v24_resolve_bytes[8] = 24;
        v24_resolve_bytes[9] = v24_resolve_bytes[10] = v24_resolve_bytes[11] = 0;
    }
    CHECK(deserialize_gpu_capture(v24_resolve_bytes, v24_resolve_loaded, error) &&
              !v24_resolve_loaded.draws[0].ps.cb_resolve,
          "v24 capture reopens with the historical no-resolve default");
    CHECK(loaded.draws[0].color0_width == 1024 && loaded.draws[0].color0_height == 32,
          "per-target extent round-trips");
    CHECK(loaded.draws[0].color1_base == 0x3000 && loaded.draws[0].color1_width == 1024 &&
          loaded.draws[0].color1_height == 32 && loaded.draws[0].ps.color1_format == 37 &&
          loaded.draws[0].ps.has_clear_color1 && loaded.draws[0].ps.clear_color1[0] == 0.25f &&
          loaded.draws[0].ps.blend1_enable && loaded.draws[0].ps.color1_write_mask == 0xf,
          "MRT1 target and fixed-function state round-trip through the v10 extension");
    CHECK(loaded.draws[0].vrt.resources[0].resource.format == DataFormat::Sint2_10_10_10,
          "newest packed vertex format enum round-trips");
    CHECK(loaded.draws[0].vrt.resources[0].resource.depth == 7,
          "v9 resource depth round-trips");
    const auto& loaded_dcc = loaded.draws[0].vrt.resources[2].resource;
    CHECK(loaded_dcc.compression_enabled && loaded_dcc.meta_pipe_aligned &&
          loaded_dcc.alpha_is_on_msb && !loaded_dcc.write_compress_enabled &&
          !loaded_dcc.color_transform && loaded_dcc.max_uncompressed_block_size == 2 &&
          loaded_dcc.max_compressed_block_size == 1 &&
          loaded_dcc.metadata_addr == 0x206e33ab00ull,
          "v11 resource DCC descriptor state remains intact in the v12 capsule");
    CHECK(loaded.shader_versions.size() == 2 && loaded.draws[0].draw_index == 7 &&
          loaded.draws[0].command_order == 123 && loaded.operations[0].source_index == 7,
          "content versions and draw operation identity round-trip");
    CHECK(loaded.draws[0].instance_count == 3,
          "v20 realized draw instance count round-trips");
    CHECK(loaded.draws[0].raw_draw_count == 6 && !loaded.draws[0].raw_indexed,
          "v23 raw draw-packet state round-trips through write/read (#1256)");
    CHECK(loaded.raw_shader_versions.size() == 3 &&
          loaded.draws[0].vs_raw_shader_index < loaded.raw_shader_versions.size() &&
          loaded.draws[0].fs_raw_shader_index < loaded.raw_shader_versions.size() &&
          loaded.draws[0].vs_chain_raw_shader_index < loaded.raw_shader_versions.size() &&
          loaded.draws[0].vertex_lds_dwords == 2176,
          "v31 linked raw-stage identities and graphics LDS round-trip with their content versions");
    CHECK(loaded.draws[0].has_pixel_inputs && loaded.draws[0].has_system_inputs &&
          loaded.draws[0].pixel_inputs == draw.pixel_inputs &&
          loaded.draws[0].system_inputs == draw.system_inputs,
          "v36 pixel linkage and system-input ABI round-trip exactly");
    CHECK(loaded.rtt_seeds.size() == 1 && loaded.rtt_seeds[0].width == 2 &&
          loaded.rtt_seeds[0].format == GpuCaptureColorFormat::Rgba8Unorm &&
          loaded.rtt_seeds[0].rgba == temporal_rgba, "temporal RTT seed round-trips");
    GpuCaptureFile fp16_capture = captured;
    fp16_capture.rtt_seeds[0].format = GpuCaptureColorFormat::Rgba16Float;
    fp16_capture.rtt_seeds[0].rgba.assign(2 * 2 * 8, 0x5a);
    std::vector<uint8_t> fp16_bytes;
    GpuCaptureFile fp16_loaded;
    CHECK(serialize_gpu_capture(fp16_capture, fp16_bytes, error) &&
          deserialize_gpu_capture(fp16_bytes, fp16_loaded, error) &&
          fp16_loaded.rtt_seeds.size() == 1 &&
          fp16_loaded.rtt_seeds[0].format == GpuCaptureColorFormat::Rgba16Float &&
          fp16_loaded.rtt_seeds[0].rgba == fp16_capture.rtt_seeds[0].rgba,
          "native FP16 RTT seed format and bytes round-trip");
    GpuCaptureFile scalar_seed_capture = captured;
    scalar_seed_capture.rtt_seeds[0].format = GpuCaptureColorFormat::R8Unorm;
    scalar_seed_capture.rtt_seeds[0].rgba.assign(2 * 2, 0x7f);
    std::vector<uint8_t> scalar_bytes;
    GpuCaptureFile scalar_loaded;
    CHECK(serialize_gpu_capture(scalar_seed_capture, scalar_bytes, error) &&
          deserialize_gpu_capture(scalar_bytes, scalar_loaded, error) &&
          scalar_loaded.rtt_seeds.size() == 1 &&
          scalar_loaded.rtt_seeds[0].format == GpuCaptureColorFormat::R8Unorm &&
          scalar_loaded.rtt_seeds[0].rgba == scalar_seed_capture.rtt_seeds[0].rgba,
          "native R8 RTT seed format and bytes round-trip");
    scalar_seed_capture.rtt_seeds[0].format = GpuCaptureColorFormat::R32Uint;
    scalar_seed_capture.rtt_seeds[0].rgba.assign(2 * 2 * 4, 0x42);
    CHECK(serialize_gpu_capture(scalar_seed_capture, scalar_bytes, error) &&
          deserialize_gpu_capture(scalar_bytes, scalar_loaded, error) &&
          scalar_loaded.rtt_seeds.size() == 1 &&
          scalar_loaded.rtt_seeds[0].format == GpuCaptureColorFormat::R32Uint &&
          scalar_loaded.rtt_seeds[0].rgba == scalar_seed_capture.rtt_seeds[0].rgba,
          "native R32_UINT RTT seed format and bytes round-trip");
    std::vector<uint8_t> v37_compatible_bytes;
    GpuCaptureFile v37_compatible_loaded;
    CHECK(serialize_gpu_capture(captured, v37_compatible_bytes, error) &&
              v37_compatible_bytes.size() >= 12,
          "current capture prepares the layout-compatible v37 fixture");
    if (v37_compatible_bytes.size() >=
        v50_tail(captured) + v49_tail(captured) + v48_tail(captured) +
        v47_tail(captured) +
        v46_tail(captured) + v45_tail(captured) +
        v44_tail(captured) + v43_tail(captured) +
            v42_tail(captured) + v41_tail(captured) +
            v40_tail(captured) + v39_tail(captured)) {
        v37_compatible_bytes.resize(v37_compatible_bytes.size() - v50_tail(captured));
        v37_compatible_bytes.resize(v37_compatible_bytes.size() - v49_tail(captured));
        v37_compatible_bytes.resize(v37_compatible_bytes.size() - v48_tail(captured));
        v37_compatible_bytes.resize(v37_compatible_bytes.size() - v47_tail(captured));
        v37_compatible_bytes.resize(v37_compatible_bytes.size() - v46_tail(captured));
        v37_compatible_bytes.resize(v37_compatible_bytes.size() - v45_tail(captured));
        v37_compatible_bytes.resize(v37_compatible_bytes.size() - v44_tail(captured));
        v37_compatible_bytes.resize(v37_compatible_bytes.size() - v43_tail(captured));
        v37_compatible_bytes.resize(v37_compatible_bytes.size() - v42_tail(captured));
        v37_compatible_bytes.resize(v37_compatible_bytes.size() - v41_tail(captured));
        v37_compatible_bytes.resize(v37_compatible_bytes.size() - v40_tail(captured));
        v37_compatible_bytes.resize(v37_compatible_bytes.size() - v39_tail(captured));
        v37_compatible_bytes[8] = 37;
        v37_compatible_bytes[9] = v37_compatible_bytes[10] =
            v37_compatible_bytes[11] = 0;
    }
    CHECK(deserialize_gpu_capture(v37_compatible_bytes, v37_compatible_loaded, error) &&
              v37_compatible_loaded.format_version == 37 &&
              v37_compatible_loaded.rtt_seeds.size() == 1 &&
              v37_compatible_loaded.rtt_seeds[0].format ==
                  GpuCaptureColorFormat::Rgba8Unorm,
          "v37 capsules remain readable after removing the v39 compute-recompile tail");
    CHECK(loaded.ds_seeds.size() == 1 && loaded.ds_seeds[0].depth == ds_seed.depth &&
          loaded.ds_seeds[0].stencil == ds_seed.stencil && loaded.ds_seeds[0].depth_valid &&
          loaded.ds_seeds[0].stencil_valid,
          "persistent depth and stencil planes round-trip independently");
    CHECK(loaded.draws[0].ps.db_render_control == 2 && loaded.draws[0].ps.stencil_clear_enable &&
          loaded.draws[0].ps.has_stencil_clear && loaded.draws[0].ps.stencil_clear_value == 3 &&
          loaded.draws[0].ps.stencil_read_base == 0x12345000 &&
          loaded.draws[0].ps.raw_stencil_op[0][1] == 4 && loaded.draws[0].ps.db_shader_control == 2 &&
          loaded.draws[0].ps.stencil_test_val_export_enable &&
          loaded.draws[0].ps.db_depth_view == 0x04002001 &&
          loaded.draws[0].ps.htile_data_base == 0x12389000 &&
          loaded.draws[0].ps.db_depth_size_xy == 0x01230234 &&
          loaded.draws[0].ps.db_htile_surface == 0x99,
          "clear intent, DS identity/programming, and raw stencil-op provenance round-trip");

    GpuReplayFrame replay;
    CHECK(materialize_gpu_replay(loaded, replay, error), "capture materializes owned replay draw items");
    CHECK(replay.items.size() == 1 && replay.items[0].ps.cb_resolve,
          "materialized gpu_replay draw retains MODE=3 resolve intent");
    // #1459: gpu_replay --inspect reads the color-state line from the MATERIALIZED item, so assert
    // the provenance survives that step too — a round-trip into GpuCaptureFile alone would not prove
    // the offline diagnostic can actually see a present-and-zero mask.
    CHECK(replay.items[0].ps.has_cb_color_control &&
              replay.items[0].ps.cb_color_control == 0x00CC0010u &&
              replay.items[0].ps.has_cb_target_mask &&
              replay.items[0].ps.cb_target_mask == 0u &&
              replay.items[0].ps.has_cb_shader_mask &&
              replay.items[0].ps.cb_shader_mask == 0x000000FFu,
          "materialized gpu_replay draw retains the v43 raw color-state provenance (#1459)");
    const auto& rr = replay.items[0].vrt->resources;
    CHECK(rr[0].gpu_addr == 0x1000 && rr[1].gpu_addr == 0x1008, "replay retains logical guest addresses");
    CHECK(rr[0].host_data && rr[1].host_data == rr[0].host_data + 8 && rr[1].host_data[0] == memory[8],
          "replay resources point into shared owned backing at captured offsets");
    CHECK(replay.expected_output_valid && replay.expected_output_hash == captured.expected_output_hash,
          "expected render oracle round-trips");
    const auto no_oracle_path = prosper_test::test_scratch_dir() /
        "prosper_gpu_capture_no_oracle_test.prgcap";
    auto no_oracle_pending = std::make_unique<PendingGpuCapture>();
    no_oracle_pending->path = no_oracle_path.string();
    no_oracle_pending->capture = captured;
    GpuCaptureFile no_oracle_loaded;
    CHECK(finish_requested_gpu_capture(std::move(no_oracle_pending), {}, error) &&
          read_gpu_capture(no_oracle_path.string(), no_oracle_loaded, error) &&
          !no_oracle_loaded.expected_output_valid && no_oracle_loaded.expected_output_bytes == 0 &&
          no_oracle_loaded.expected_output_hash == 0,
          "an empty renderer result does not become a valid replay oracle");
    CHECK(replay.items[0].draw_index == 7 && replay.items[0].command_order == 123,
          "materialized draw retains its source operation identity");
    CHECK(replay.items[0].instance_count == 3,
          "materialized replay retains the draw instance count");
    CHECK(replay.items[0].raw_draw_modifier == 0x1122334455667788ull &&
          replay.items[0].vertex_offset == -37,
          "materialized replay retains its draw modifier and Vulkan vertex offset");
    CHECK(replay.items[0].vs_raw_shader_index == loaded.draws[0].vs_raw_shader_index &&
          replay.items[0].fs_raw_shader_index == loaded.draws[0].fs_raw_shader_index &&
          replay.items[0].vs_chain_raw_shader_index ==
              loaded.draws[0].vs_chain_raw_shader_index &&
          replay.items[0].vertex_lds_dwords == 2176,
          "materialized replay exposes the linked raw-stage identities and graphics LDS");
    CHECK(replay.items[0].has_pixel_inputs && replay.items[0].has_system_inputs &&
          replay.items[0].pixel_inputs == draw.pixel_inputs &&
          replay.items[0].system_inputs == draw.system_inputs,
          "materialized replay exposes the exact captured pixel-stage ABI");
    CHECK(replay.rtt_seeds.size() == 1 && replay.rtt_seeds[0].guest_addr == draw.color0_base,
          "materialized replay owns temporal RTT seed pixels");
    CHECK(replay.ds_seeds.size() == 1 && replay.ds_seeds[0].htile_data_base == 0x800000,
          "materialized replay owns persistent DS checkpoint bytes");

    std::vector<uint8_t> repeated(48);
    for (size_t i = 0; i < 16; ++i) repeated[i] = repeated[i + 32] = static_cast<uint8_t>(i * 3 + 1);
    const uint32_t compute_endpgm = 0xbf810000u;
    std::memcpy(repeated.data(), &compute_endpgm, sizeof(compute_endpgm));
    std::memcpy(repeated.data() + 32, &compute_endpgm, sizeof(compute_endpgm));
    auto repeated_reader = [&](uint64_t addr, uint8_t* dst, size_t n) -> size_t {
        if (addr < 0x3000 || addr >= 0x3000 + repeated.size()) return 0;
        const size_t offset = static_cast<size_t>(addr - 0x3000);
        const size_t take = std::min(n, repeated.size() - offset);
        std::memcpy(dst, repeated.data() + offset, take);
        return take;
    };
    auto draw_table = std::make_shared<ShaderResourceTable>();
    ShaderResource draw_resource{}; draw_resource.gpu_addr = 0x3000; draw_resource.size = 16;
    draw_resource.binding = 2; draw_table->resources = {draw_resource};
    auto compute_table = std::make_shared<ShaderResourceTable>();
    ShaderResource compute_resource = draw_resource; compute_resource.gpu_addr = 0x3020;
    compute_table->resources = {compute_resource};
    DrawItem mixed_draw; mixed_draw.vs = {0x07230203, 11}; mixed_draw.fs = {0x07230203, 22};
    mixed_draw.vrt = draw_table; mixed_draw.draw_index = 4; mixed_draw.command_order = 100;
    ComputeItem mixed_compute; mixed_compute.spirv = mixed_draw.fs; mixed_compute.resources = compute_table;
    mixed_compute.dispatch_index = 9; mixed_compute.submit_no = 88; mixed_compute.command_order = 101;
    mixed_compute.launch.groups_x = 8; mixed_compute.launch.local_x = 64;
    mixed_compute.required_subgroup_size = 64;
    mixed_compute.code_addr = 0x3000;
    mixed_compute.recompile_config_available = true;
    mixed_compute.recompile_config.user_sgprs = {0x12345678u, 0x9abcdef0u};
    mixed_compute.recompile_config.local_x = 64;
    mixed_compute.recompile_config.local_y = 1;
    mixed_compute.recompile_config.local_z = 1;
    mixed_compute.recompile_config.wave_size = 64;
    mixed_compute.recompile_config.compute_pgm_rsrc1 = 0x402c00c3u;
    mixed_compute.recompile_config.tgid_x_en = true;
    mixed_compute.recompile_config.lds_bytes = 512;
    mixed_compute.recompile_config.native_subgroup_size = 64;
    mixed_compute.recompile_config.native_storage_format_support =
        native_storage_format_support_bit(DataFormat::Float16, 4);
    std::vector<SubmitOperation> mixed_operations = {
        {SubmitOperationKind::Draw, 4, 100},
        {SubmitOperationKind::Dispatch, 9, 101},
        {SubmitOperationKind::Draw, 5, 102},
    };
    GpuCaptureFile mixed;
    CHECK(capture_submit_items({mixed_draw}, {mixed_compute}, mixed_operations, meta,
                               repeated_reader, mixed, error),
          "mixed graphics/compute submit captures without a renderer");
    CHECK(mixed.blobs.size() == 1 && mixed.shader_versions.size() == 2 &&
              mixed.computes[0].required_subgroup_size == 64 &&
              mixed.computes[0].raw_shader_index < mixed.raw_shader_versions.size() &&
              mixed.computes[0].recompile_config_available,
          "identical address-distinct resources and shared shaders deduplicate by content");
    CHECK(mixed.operations.size() == 3 && mixed.operations[0].realized &&
          mixed.operations[1].realized && !mixed.operations[2].realized,
          "mixed operation plan explicitly retains unrealized work");
    CHECK(mixed.failure_diagnostics_available && mixed.failure_diagnostics.size() == 1 &&
          mixed.failure_diagnostics[0].reason == RealizationFailureReason::Unknown,
          "pre-realized capture inputs label an unexplained omission explicitly");
    const auto mixed_path = prosper_test::test_scratch_dir() / "prosper_gpu_capture_mixed_test.prgcap";
    CHECK(write_gpu_capture(mixed_path.string(), mixed, error), "mixed versioned capture writes");
    GpuCaptureFile mixed_loaded;
    CHECK(read_gpu_capture(mixed_path.string(), mixed_loaded, error) &&
          mixed_loaded.computes.size() == 1 && mixed_loaded.operations.size() == 3 &&
          mixed_loaded.computes[0].required_subgroup_size == 64 &&
          mixed_loaded.computes[0].recompile_config.user_sgprs ==
              mixed_compute.recompile_config.user_sgprs &&
          mixed_loaded.computes[0].recompile_config.compute_pgm_rsrc1 == 0x402c00c3u &&
          mixed_loaded.computes[0].raw_shader_index <
              mixed_loaded.raw_shader_versions.size() &&
          !mixed_loaded.expected_output_valid,
          "mixed items, operation order, compute source/ABI, and absent output oracle round-trip");
    GpuReplayFrame mixed_replay;
    CHECK(materialize_gpu_replay(mixed_loaded, mixed_replay, error) &&
          mixed_replay.computes.size() == 1 && mixed_replay.computes[0].dispatch_index == 9 &&
          mixed_replay.computes[0].required_subgroup_size == 64 &&
          mixed_replay.computes[0].recompile_config.compute_pgm_rsrc1 == 0x402c00c3u &&
          mixed_replay.computes[0].user_sgprs == mixed_compute.recompile_config.user_sgprs &&
          mixed_replay.computes[0].raw_shader_index <
              mixed_replay.raw_shader_versions.size() &&
          mixed_replay.computes[0].resources->resources[0].host_data[0] == repeated[32],
          "offline materialization owns compute resources and preserves raw-recompile state");

    // The guarded-null raw-store marker is conditional proof, not self-authenticating metadata.
    // An unrelated captured shader cannot acquire it merely by forging the impossible stride.
    GpuCaptureFile guarded_store_capture = mixed;
    GpuCapturedResource& guarded_store_captured =
        guarded_store_capture.computes[0].resources.resources[0];
    guarded_store_captured.resource.cls = ResourceClass::ConstantBuffer;
    guarded_store_captured.resource.format = DataFormat::Unknown;
    guarded_store_captured.resource.num_components = 0u;
    guarded_store_captured.resource.gpu_addr = 0u;
    guarded_store_captured.resource.size = 0u;
    guarded_store_captured.resource.stride = kProvenNullGuardedRawStoreStride;
    guarded_store_captured.resource.srt_offset = 0xFFFFFFFFu;
    guarded_store_captured.resource.sgpr_base = 0xFFFFFFFFu;
    guarded_store_captured.resource.fetch_pc = 74u;
    guarded_store_captured.captured_size = 0u;
    guarded_store_captured.blob_index = 0xFFFFFFFFu;
    guarded_store_captured.blob_offset = 0u;
    std::vector<uint8_t> guarded_store_bytes;
    GpuCaptureFile guarded_store_loaded;
    GpuReplayFrame guarded_store_replay;
    CHECK(!serialize_gpu_capture(guarded_store_capture, guarded_store_bytes, error),
          "a forged guarded-null marker is rejected at the capture boundary");

    std::vector<uint32_t> guarded_store_raw(81u, 0xbf800000u); // s_nop 0
    guarded_store_raw[22] = 0xbe92047eu; // s_mov_b64 s[18:19], exec
    guarded_store_raw[41] = 0xbefe0412u; // s_mov_b64 exec, s[18:19]
    guarded_store_raw[42] = 0xbf128002u;
    guarded_store_raw[43] = 0x7d8a00f9u;
    guarded_store_raw[44] = 0x06868080u;
    guarded_store_raw[45] = 0x85ea8012u;
    guarded_store_raw[46] = 0x8dea006au;
    guarded_store_raw[47] = 0x87fe126au;
    guarded_store_raw[48] = 0xbf88001fu;
    guarded_store_raw[74] = 0xe0740030u;
    guarded_store_raw[75] = 0x80000700u;
    guarded_store_raw[80] = 0xbf810000u;
    GpuCaptureFile validated_guarded_store_capture = guarded_store_capture;
    auto& validated_compute = validated_guarded_store_capture.computes[0];
    CHECK(validated_compute.raw_shader_index <
              validated_guarded_store_capture.raw_shader_versions.size(),
          "guarded-store capture fixture retains a raw compute slot");
    auto& validated_raw = validated_guarded_store_capture.raw_shader_versions[
        validated_compute.raw_shader_index];
    validated_raw.words = guarded_store_raw;
    validated_raw.has_endpgm = true;
    validated_raw.content_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(guarded_store_raw.data()),
        guarded_store_raw.size() * sizeof(uint32_t));
    validated_compute.recompile_config_available = true;
    validated_compute.recompile_config.user_sgprs.assign(4u, 0u);
    validated_compute.recompile_config.local_x = validated_compute.launch.local_x;
    validated_compute.recompile_config.local_y = validated_compute.launch.local_y;
    validated_compute.recompile_config.local_z = validated_compute.launch.local_z;
    validated_compute.recompile_config.threads_x = validated_compute.launch.threads_x;
    validated_compute.recompile_config.threads_y = validated_compute.launch.threads_y;
    validated_compute.recompile_config.threads_z = validated_compute.launch.threads_z;
    validated_compute.recompile_config.native_subgroup_size =
        validated_compute.required_subgroup_size;
    error.clear();
    CHECK(serialize_gpu_capture(validated_guarded_store_capture,
                                guarded_store_bytes, error) &&
              deserialize_gpu_capture(guarded_store_bytes, guarded_store_loaded, error) &&
              is_proven_null_guarded_raw_store(
                  guarded_store_loaded.computes[0].resources.resources[0].resource) &&
              materialize_gpu_replay(guarded_store_loaded, guarded_store_replay, error) &&
              is_proven_null_guarded_raw_store(
                  guarded_store_replay.computes[0].resources->resources[0]) &&
              guarded_store_replay.computes[0].null_guarded_raw_store_validated,
          "raw shader and dispatch proof mint the replay backend's internal marker token");

    for (const uint32_t wrong_fetch_pc : {73u, 79u}) {
        GpuCaptureFile wrong_site_guarded_store_capture = validated_guarded_store_capture;
        wrong_site_guarded_store_capture.computes[0]
            .resources.resources[0].resource.fetch_pc = wrong_fetch_pc;
        error.clear();
        CHECK(!serialize_gpu_capture(wrong_site_guarded_store_capture,
                                     guarded_store_bytes, error),
              "capture rejects a guarded-store marker whose fetch PC is not an exact store site");
    }

    GpuCaptureFile stale_guarded_store_capture = validated_guarded_store_capture;
    auto& stale_guarded_raw_version = stale_guarded_store_capture.raw_shader_versions[
        stale_guarded_store_capture.computes[0].raw_shader_index];
    stale_guarded_raw_version.words[40] =
        0xbe820381u; // s_mov_b32 s2, 1 before the pc42 compare
    stale_guarded_raw_version.content_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(stale_guarded_raw_version.words.data()),
        stale_guarded_raw_version.words.size() * sizeof(uint32_t));
    error.clear();
    CHECK(!serialize_gpu_capture(stale_guarded_store_capture,
                                 guarded_store_bytes, error),
          "capture rejects guarded-store provenance whose shader overwrites s2 before pc42");

    // The nullable-output marker is backed by a captured 40-byte dispatch-table witness. Its
    // impossible stride remains metadata only: replay mints the internal authority token after
    // independently checking the exact raw shader, launch, marker PC, and zero +0x20 qword.
    constexpr uint64_t nullable_code_addr = 0x730000u;
    constexpr uint64_t nullable_table_addr = 0x740000u;
    std::array<uint8_t, kGtaNullableOutputWitnessBytes> nullable_witness{};
    ShaderResource nullable_marker;
    nullable_marker.cls = ResourceClass::ConstantBuffer;
    nullable_marker.format = DataFormat::Unknown;
    nullable_marker.num_components = 0u;
    nullable_marker.gpu_addr = nullable_table_addr;
    nullable_marker.size = kGtaNullableOutputWitnessBytes;
    nullable_marker.stride = kProvenNullNullableRawBufferStride;
    nullable_marker.fetch_pc = 38u;
    nullable_marker.host_data = nullable_witness.data();
    nullable_marker.host_data_size = nullable_witness.size();

    ComputeShaderConfig nullable_config;
    nullable_config.user_sgprs.assign(9u, 0u);
    nullable_config.user_sgprs[0] = static_cast<uint32_t>(nullable_table_addr);
    nullable_config.user_sgprs[1] = static_cast<uint32_t>(nullable_table_addr >> 32u);
    nullable_config.user_sgprs[7] = kGtaNullableOutputFixtureRecordCount;
    nullable_config.user_sgprs[8] = kGtaNullableOutputUserSgpr8;
    nullable_config.local_x = kGtaNullableOutputLocalSize;
    nullable_config.local_y = nullable_config.local_z = 1u;
    nullable_config.threads_x = kGtaNullableOutputFixtureThreads;
    nullable_config.threads_y = nullable_config.threads_z = 1u;
    nullable_config.wave_size = 64u;
    nullable_config.tgid_x_en = true;

    ComputeItem nullable_compute;
    nullable_compute.spirv = {0x07230203u};
    nullable_compute.user_sgprs = nullable_config.user_sgprs;
    nullable_compute.resources = std::make_shared<ShaderResourceTable>();
    nullable_compute.resources->resources.push_back(nullable_marker);
    nullable_compute.launch.threads_x = kGtaNullableOutputFixtureThreads;
    nullable_compute.launch.threads_y = nullable_compute.launch.threads_z = 1u;
    nullable_compute.launch.local_x = kGtaNullableOutputLocalSize;
    nullable_compute.launch.local_y = nullable_compute.launch.local_z = 1u;
    nullable_compute.launch.groups_x = kGtaNullableOutputFixtureRecordCount;
    nullable_compute.launch.groups_y = nullable_compute.launch.groups_z = 1u;
    nullable_compute.code_addr = nullable_code_addr;
    nullable_compute.dispatch_index = 2481u;
    nullable_compute.command_order = 2481u;
    nullable_compute.recompile_config = nullable_config;
    nullable_compute.recompile_config_available = true;
    const std::vector<SubmitOperation> nullable_operations = {
        {SubmitOperationKind::Dispatch, 2481u, 2481u},
    };
    auto nullable_reader = [&](uint64_t addr, uint8_t* dst, size_t n) -> size_t {
        const auto read_span = [&](uint64_t base, const uint8_t* src,
                                   size_t bytes) -> size_t {
            if (addr < base || addr >= base + bytes) return 0u;
            const size_t offset = static_cast<size_t>(addr - base);
            const size_t take = std::min(n, bytes - offset);
            std::memcpy(dst, src + offset, take);
            return take;
        };
        if (const size_t read = read_span(
                nullable_code_addr,
                reinterpret_cast<const uint8_t*>(
                    prosper::test::kGta5WorkgroupStoreProgram.data()),
                sizeof(prosper::test::kGta5WorkgroupStoreProgram)))
            return read;
        return read_span(nullable_table_addr, nullable_witness.data(),
                         nullable_witness.size());
    };

    GpuCaptureFile nullable_capture;
    CHECK(capture_submit_items({}, {nullable_compute}, nullable_operations, meta,
                               nullable_reader, nullable_capture, error) &&
              nullable_capture.computes.size() == 1u &&
              nullable_capture.computes[0].raw_shader_index <
                  nullable_capture.raw_shader_versions.size() &&
              nullable_capture.computes[0].resources.resources.size() == 1u &&
              nullable_capture.computes[0].resources.resources[0].captured_size ==
                  kGtaNullableOutputWitnessBytes,
          "capture retains the exact nullable-output shader, launch, marker, and table witness");
    std::vector<uint8_t> nullable_capture_bytes;
    GpuCaptureFile nullable_loaded;
    GpuReplayFrame nullable_replay;
    CHECK(serialize_gpu_capture(nullable_capture, nullable_capture_bytes, error) &&
              deserialize_gpu_capture(nullable_capture_bytes, nullable_loaded, error) &&
              materialize_gpu_replay(nullable_loaded, nullable_replay, error) &&
              nullable_replay.computes.size() == 1u &&
              nullable_replay.computes[0].nullable_output_raw_buffer_validated &&
              is_proven_null_nullable_raw_buffer(
                  nullable_replay.computes[0].resources->resources[0]),
          "nullable-output capture round-trip revalidates provenance and mints the replay token");

    GpuCaptureFile nonnull_nullable_capture = nullable_loaded;
    const GpuCapturedResource& loaded_nullable_marker =
        nonnull_nullable_capture.computes[0].resources.resources[0];
    nonnull_nullable_capture.blobs[loaded_nullable_marker.blob_index]
        .bytes[loaded_nullable_marker.blob_offset + kGtaNullableOutputPointerOffset] = 1u;
    CHECK(!materialize_gpu_replay(nonnull_nullable_capture, nullable_replay, error),
          "replay rejects a nullable-output capture whose +0x20 witness became non-null");

    GpuCaptureFile wrong_pc_nullable_capture = nullable_loaded;
    wrong_pc_nullable_capture.computes[0].resources.resources[0].resource.fetch_pc =
        UINT32_MAX;
    CHECK(!materialize_gpu_replay(wrong_pc_nullable_capture, nullable_replay, error),
          "replay rejects a malformed nullable-output sentinel whose exact marker PC was lost");

    GpuCaptureFile stale_nullable_shader_capture = nullable_loaded;
    auto& stale_nullable_raw = stale_nullable_shader_capture.raw_shader_versions[
        stale_nullable_shader_capture.computes[0].raw_shader_index];
    stale_nullable_raw.words[38] ^= 1u;
    stale_nullable_raw.content_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(stale_nullable_raw.words.data()),
        stale_nullable_raw.words.size() * sizeof(uint32_t));
    CHECK(!materialize_gpu_replay(stale_nullable_shader_capture, nullable_replay, error),
          "replay rejects nullable-output provenance after mutating the exact raw-buffer site");

    GpuCaptureFile stale_nullable_launch_capture = nullable_loaded;
    stale_nullable_launch_capture.computes[0].launch.groups_x =
        kGtaNullableOutputFixtureRecordCount + 1u;
    CHECK(!materialize_gpu_replay(stale_nullable_launch_capture, nullable_replay, error),
          "replay rejects nullable-output provenance after the captured dispatch grid changes");

    GpuCaptureFile mixed_atomic = mixed;
    mixed_atomic.computes[0].recompile_config.storage_buffer_int64_atomics = true;
    // Capture serialization treats the marker as opaque provenance; the emitter separately
    // revalidates the live descriptor's size/stride/address before it can use this proof.
    mixed_atomic.computes[0].resources.resources[0].resource
        .atomic_x2_record_count = 25u;
    std::vector<uint8_t> mixed_atomic_bytes;
    GpuCaptureFile mixed_atomic_loaded;
    GpuReplayFrame mixed_atomic_replay;
    CHECK(serialize_gpu_capture(mixed_atomic, mixed_atomic_bytes, error) &&
              deserialize_gpu_capture(mixed_atomic_bytes, mixed_atomic_loaded, error) &&
              mixed_atomic_loaded.computes[0]
                  .recompile_config.storage_buffer_int64_atomics &&
              mixed_atomic_loaded.computes[0]
                  .recompile_config.compute_pgm_rsrc1 == 0x402c00c3u &&
              materialize_gpu_replay(mixed_atomic_loaded, mixed_atomic_replay, error) &&
              mixed_atomic_replay.computes[0]
                  .recompile_config.storage_buffer_int64_atomics &&
              mixed_atomic_replay.computes[0].resources->resources[0]
                  .atomic_x2_record_count == 25u,
          "v50 raw-compute replay preserves PGM_RSRC1, qword-atomic support, and record count");
    std::vector<uint8_t> v49_mixed_atomic = mixed_atomic_bytes;
    v49_mixed_atomic.resize(v49_mixed_atomic.size() - v50_tail(mixed_atomic));
    v49_mixed_atomic[8] = 49;
    v49_mixed_atomic[9] = v49_mixed_atomic[10] = v49_mixed_atomic[11] = 0;
    GpuCaptureFile v49_mixed_atomic_loaded;
    CHECK(deserialize_gpu_capture(v49_mixed_atomic, v49_mixed_atomic_loaded, error) &&
              v49_mixed_atomic_loaded.computes[0]
                  .recompile_config.compute_pgm_rsrc1 == UINT32_MAX,
          "v49 raw-compute replay leaves unrecorded PGM_RSRC1 explicitly unknown");
    std::vector<uint8_t> malformed_pgm_rsrc1_count = mixed_atomic_bytes;
    const uint32_t zero_count = 0;
    std::memcpy(malformed_pgm_rsrc1_count.data() + malformed_pgm_rsrc1_count.size() -
                    v50_tail(mixed_atomic),
                &zero_count, sizeof(zero_count));
    GpuCaptureFile malformed_pgm_rsrc1_loaded;
    CHECK(!deserialize_gpu_capture(malformed_pgm_rsrc1_count,
                                   malformed_pgm_rsrc1_loaded, error) &&
              error == "invalid compute PGM_RSRC1 state count",
          "v50 capture rejects a PGM_RSRC1 state count that omits a realized compute");
    std::vector<uint8_t> v48_reserved_atomic_flag = v49_mixed_atomic;
    v48_reserved_atomic_flag.resize(v48_reserved_atomic_flag.size() - v49_tail(mixed_atomic));
    v48_reserved_atomic_flag[8] = 48;
    v48_reserved_atomic_flag[9] = v48_reserved_atomic_flag[10] =
        v48_reserved_atomic_flag[11] = 0;
    GpuCaptureFile v48_reserved_atomic_loaded;
    CHECK(!deserialize_gpu_capture(v48_reserved_atomic_flag,
                                   v48_reserved_atomic_loaded, error),
          "v48 capture rejects the int64-atomic config bit that was reserved before v49");
    GpuCaptureFile invalid_subgroup = mixed;
    invalid_subgroup.computes[0].required_subgroup_size = 16;
    std::vector<uint8_t> invalid_subgroup_bytes;
    CHECK(!serialize_gpu_capture(invalid_subgroup, invalid_subgroup_bytes, error) &&
              error == "invalid compute required-subgroup size",
          "capture rejects an invented compute subgroup contract");
    GpuCaptureFile invalid_compute_raw = mixed;
    invalid_compute_raw.computes[0].raw_shader_index =
        static_cast<uint32_t>(invalid_compute_raw.raw_shader_versions.size());
    CHECK(!serialize_gpu_capture(invalid_compute_raw, invalid_subgroup_bytes, error) &&
              error == "realized compute references an invalid raw shader",
          "capture rejects an out-of-range realized compute source");
    GpuCaptureFile missing_compute_state = mixed;
    missing_compute_state.computes[0].recompile_config_available = false;
    CHECK(!serialize_gpu_capture(missing_compute_state, invalid_subgroup_bytes, error) &&
              error == "compute raw shader has no recompile state",
          "capture never pairs compute source bytes with invented launch state");
    GpuCaptureFile invalid_compute_state = mixed;
    invalid_compute_state.computes[0].recompile_config.local_x = 32;
    CHECK(!serialize_gpu_capture(invalid_compute_state, invalid_subgroup_bytes, error) &&
              error == "invalid compute recompile state",
          "capture rejects a raw-compute ABI that disagrees with the realized launch");
    std::vector<uint8_t> v36_compute_bytes;
    GpuCaptureFile v36_compute_loaded;
    GpuCaptureFile v36_compute_source = mixed;
    v36_compute_source.computes[0].raw_shader_index = 0xFFFFFFFFu;
    v36_compute_source.computes[0].recompile_config_available = false;
    v36_compute_source.raw_shader_versions.clear();
    CHECK(serialize_gpu_capture(v36_compute_source, v36_compute_bytes, error) &&
              v36_compute_bytes.size() >=
                  v50_tail(v36_compute_source) + v49_tail(v36_compute_source) +
                  v48_tail(v36_compute_source) +
                  v47_tail(v36_compute_source) +
                  v46_tail(v36_compute_source) +
                  v45_tail(v36_compute_source) +
                      v44_tail(v36_compute_source) +
                      v43_tail(v36_compute_source) +
                      v42_tail(v36_compute_source) + v41_tail(v36_compute_source) +
                      v40_tail(v36_compute_source) +
                      v39_tail(v36_compute_source) +
                      v37_tail(v36_compute_source),
          "v37 capture exposes a removable compute subgroup-contract tail");
    if (v36_compute_bytes.size() >=
        v50_tail(v36_compute_source) + v49_tail(v36_compute_source) +
        v48_tail(v36_compute_source) +
        v47_tail(v36_compute_source) +
        v46_tail(v36_compute_source) +
        v45_tail(v36_compute_source) +
            v44_tail(v36_compute_source) +
            v43_tail(v36_compute_source) +
            v42_tail(v36_compute_source) + v41_tail(v36_compute_source) +
            v40_tail(v36_compute_source) +
            v39_tail(v36_compute_source) +
            v37_tail(v36_compute_source)) {
        v36_compute_bytes.resize(v36_compute_bytes.size() - v50_tail(v36_compute_source));
        v36_compute_bytes.resize(v36_compute_bytes.size() - v49_tail(v36_compute_source));
        v36_compute_bytes.resize(v36_compute_bytes.size() - v48_tail(v36_compute_source));
        v36_compute_bytes.resize(v36_compute_bytes.size() - v47_tail(v36_compute_source));
        v36_compute_bytes.resize(v36_compute_bytes.size() - v46_tail(v36_compute_source));
        v36_compute_bytes.resize(v36_compute_bytes.size() - v45_tail(v36_compute_source));
        v36_compute_bytes.resize(v36_compute_bytes.size() - v44_tail(v36_compute_source));
        v36_compute_bytes.resize(v36_compute_bytes.size() - v43_tail(v36_compute_source));
        v36_compute_bytes.resize(v36_compute_bytes.size() - v42_tail(v36_compute_source));
        v36_compute_bytes.resize(v36_compute_bytes.size() - v41_tail(v36_compute_source));
        v36_compute_bytes.resize(v36_compute_bytes.size() - v40_tail(v36_compute_source));
        v36_compute_bytes.resize(v36_compute_bytes.size() - v39_tail(v36_compute_source));
        v36_compute_bytes.resize(v36_compute_bytes.size() - v37_tail(v36_compute_source));
        v36_compute_bytes[8] = 36;
        v36_compute_bytes[9] = v36_compute_bytes[10] = v36_compute_bytes[11] = 0;
    }
    CHECK(deserialize_gpu_capture(v36_compute_bytes, v36_compute_loaded, error) &&
              v36_compute_loaded.computes.size() == 1 &&
              v36_compute_loaded.computes[0].required_subgroup_size == 0,
          "v36 capture reopens with the historical portable compute contract");
    uint8_t* mixed_draw_bytes = mixed_replay.items[0].vrt->resources[0].host_data;
    uint8_t* mixed_compute_bytes = mixed_replay.computes[0].resources->resources[0].host_data;
    CHECK(mixed_draw_bytes != mixed_compute_bytes &&
          mixed_replay.resource_instances.size() == 2,
          "address-distinct users of one content version receive independent mutable instances");
    mixed_compute_bytes[0] ^= 0xff;
    CHECK(mixed_draw_bytes[0] == repeated[0],
          "compute copy-on-write cannot create a false alias between equal resource versions");

    // --- v14: ordered DMA mutates the shared replay instance between exact consumers. ---
    std::array<uint8_t, 32> ordered_memory{};
    ordered_memory[0] = 0x11; ordered_memory[16] = 0xa1;
    auto ordered_reader = [&](uint64_t addr, uint8_t* dst, size_t n) -> size_t {
        if (addr < 0x5000 || addr >= 0x5000 + ordered_memory.size()) return 0;
        const size_t offset = static_cast<size_t>(addr - 0x5000);
        const size_t take = std::min(n, ordered_memory.size() - offset);
        std::memcpy(dst, ordered_memory.data() + offset, take);
        return take;
    };
    auto ordered_table = std::make_shared<ShaderResourceTable>();
    ShaderResource ordered_resource{};
    ordered_resource.cls = ResourceClass::ConstantBuffer;
    ordered_resource.gpu_addr = 0x5000; ordered_resource.size = 4;
    ordered_table->resources = {ordered_resource};
    DrawItem before_dma, after_dma;
    before_dma.vs = after_dma.vs = {0x07230203, 71};
    before_dma.fs = after_dma.fs = {0x07230203, 72};
    before_dma.vrt = after_dma.vrt = ordered_table;
    before_dma.draw_index = 100; before_dma.command_order = 10;
    after_dma.draw_index = 101; after_dma.command_order = 30;
    ComputeItem after_dma_compute;
    after_dma_compute.spirv = {0x07230203, 73};
    after_dma_compute.resources = ordered_table;
    after_dma_compute.dispatch_index = 200; after_dma_compute.command_order = 40;
    GpuState::DmaCopy ordered_copy{0x5000, 0x5010, 4, 0, 20, 0xabc0};
    std::vector<SubmitOperation> ordered_operations = {
        {SubmitOperationKind::Draw, 100, 10},
        {SubmitOperationKind::DmaCopy, 0, 20},
        {SubmitOperationKind::Draw, 101, 30},
        {SubmitOperationKind::Dispatch, 200, 40},
    };
    GpuCaptureFile ordered_capture;
    CHECK(capture_submit_items({before_dma, after_dma}, {after_dma_compute},
                               ordered_operations, meta, ordered_reader, ordered_capture, error,
                               {}, {}, {ordered_copy}) &&
          ordered_capture.dma_copies.size() == 1 && ordered_capture.blobs.size() == 2 &&
          ordered_capture.operations[1].kind == SubmitOperationKind::DmaCopy,
          "v14 capture closes over ordered DMA source/destination versions");
    std::vector<uint8_t> ordered_bytes;
    GpuCaptureFile ordered_loaded;
    GpuReplayFrame ordered_replay;
    CHECK(serialize_gpu_capture(ordered_capture, ordered_bytes, error) &&
          deserialize_gpu_capture(ordered_bytes, ordered_loaded, error) &&
          materialize_gpu_replay(ordered_loaded, ordered_replay, error) &&
          ordered_replay.dma_copies.size() == 1 &&
          ordered_replay.dma_copies[0].source_data &&
          ordered_replay.dma_copies[0].destination_data,
          "v14 DMA records and endpoint bindings round-trip into owned replay storage");
    std::vector<SubmitOperation> replay_operations;
    for (const auto& operation : ordered_replay.operations)
        if (operation.realized)
            replay_operations.push_back({operation.kind,
                                         static_cast<size_t>(operation.source_index),
                                         operation.command_order});
    std::vector<uint8_t> draw_observations, compute_observations;
    std::vector<std::pair<uint64_t, uint64_t>> invalidations;
    bool producer_rendered = false;
    const std::vector<uint8_t> current_render_target = {0xe1, 0xe2, 0xe3, 0xe4};
    set_live_target_byte_range_reader(
        [&](uint64_t addr, uint32_t bytes, std::vector<uint8_t>& output) {
            if (addr != ordered_copy.src) return LiveTargetByteReadResult::NotFound;
            if (!producer_rendered || bytes != current_render_target.size())
                return LiveTargetByteReadResult::InvalidRange;
            output = current_render_target;
            return LiveTargetByteReadResult::Success;
        });
    set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
        invalidations.emplace_back(addr, size);
    });
    execute_ordered_items(
        replay_operations, ordered_replay.items, ordered_replay.computes,
        ordered_replay.dma_copies,
        [&](const std::vector<DrawItem>& items, uint32_t, uint32_t) {
            for (const auto& item : items)
                draw_observations.push_back(item.vrt->resources[0].host_data[0]);
            producer_rendered = true;
            return RenderedFrame{};
        },
        [&](const std::vector<ComputeItem>& items) {
            compute_observations.push_back(items[0].resources->resources[0].host_data[0]);
            return true;
        }, 1, 1);
    set_guest_gpu_write_observer({});
    set_live_target_byte_range_reader({});
    CHECK(draw_observations == std::vector<uint8_t>({0x11, 0xe1}),
          "draw -> DMA -> draw replay copies the current producer target, not its stale blob");
    CHECK(compute_observations == std::vector<uint8_t>({0xe1}),
          "DMA -> compute replay observes the current producer bytes at exact command order");
    CHECK((invalidations == std::vector<std::pair<uint64_t, uint64_t>>({{0x5000, 4}})),
          "replay DMA invalidates renderer caches by captured guest destination identity");

    // Selector-1 destinations belong to the capture's shared internal GDS instance. They have no
    // guest destination blob and must mutate the exact backing consumed by later compute work.
    std::vector<uint8_t> initial_gds(64u * 1024u, 0);
    initial_gds[0x24] = 0x19;
    auto gds_table = std::make_shared<ShaderResourceTable>();
    ShaderResource gds_resource{};
    gds_resource.cls = ResourceClass::ConstantBuffer;
    gds_resource.binding = kComputeInternalGdsBinding;
    gds_resource.gpu_addr = 0;
    gds_resource.size = initial_gds.size();
    gds_resource.stride = 4;
    gds_resource.host_data = initial_gds.data();
    gds_resource.host_data_size = initial_gds.size();
    gds_table->resources = {gds_resource};
    ComputeItem gds_consumer;
    gds_consumer.spirv = {0x07230203, 74};
    gds_consumer.resources = gds_table;
    gds_consumer.dispatch_index = 201;
    gds_consumer.command_order = 30;
    GpuState::DmaCopy gds_copy{
        0x24, 0x5010, 4, 1u | (3u << 8) | kDmaDataAddressSource, 20, 0xabc4};
    const std::vector<SubmitOperation> gds_operations = {
        {SubmitOperationKind::DmaCopy, 0, 20},
        {SubmitOperationKind::Dispatch, 201, 30},
    };
    GpuCaptureFile gds_capture;
    CHECK(capture_submit_items({}, {gds_consumer}, gds_operations, meta, ordered_reader,
                               gds_capture, error, {}, {}, {gds_copy}) &&
              gds_capture.dma_copies.size() == 1 && gds_capture.blobs.size() == 2 &&
              gds_capture.dma_copies[0].destination_blob_index <
                  gds_capture.blobs.size() &&
              gds_capture.dma_copies[0].destination_blob_offset == gds_copy.dst &&
              gds_capture.blobs[gds_capture.dma_copies[0].destination_blob_index].bytes ==
                  initial_gds,
          "memory-to-GDS capture retains the complete pre-submit GDS independently");
    std::vector<uint8_t> gds_capture_bytes;
    GpuCaptureFile gds_loaded;
    GpuReplayFrame gds_replay;
    CHECK(serialize_gpu_capture(gds_capture, gds_capture_bytes, error) &&
              deserialize_gpu_capture(gds_capture_bytes, gds_loaded, error) &&
              materialize_gpu_replay(gds_loaded, gds_replay, error) &&
              gds_replay.dma_copies.size() == 1 && gds_replay.computes.size() == 1 &&
              gds_replay.computes[0].resources &&
              gds_replay.dma_copies[0].destination_data ==
                  gds_replay.computes[0].resources->resources[0].host_data + 0x24,
          "selector-1 destination round-trips into the shared captured GDS instance");
    std::vector<SubmitOperation> gds_replay_operations;
    for (const auto& operation : gds_replay.operations)
        if (operation.realized)
            gds_replay_operations.push_back({operation.kind,
                                             static_cast<size_t>(operation.source_index),
                                             operation.command_order});
    std::array<uint8_t, 4> gds_observed{};
    invalidations.clear();
    set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
        invalidations.emplace_back(addr, size);
    });
    execute_ordered_items(
        gds_replay_operations, gds_replay.items, gds_replay.computes,
        gds_replay.dma_copies, {},
        [&](const std::vector<ComputeItem>& items) {
            std::memcpy(gds_observed.data(),
                        items[0].resources->resources[0].host_data + 0x24,
                        gds_observed.size());
            return true;
        }, 1, 1);
    set_guest_gpu_write_observer({});
    CHECK((gds_observed == std::array<uint8_t, 4>{0xa1, 0, 0, 0}),
          "later replay compute observes the ordered memory-to-GDS copy");
    CHECK(invalidations.empty(),
          "replay never publishes a GDS offset as a guest GPU write");

    // A valid submit may contain the retained DMA and no compute table at all. Its standalone
    // backing must therefore come from the pre-submit GDS snapshot, not from a coincidental
    // same-submit consumer descriptor.
    const std::vector<SubmitOperation> gds_dma_only_operations = {
        {SubmitOperationKind::DmaCopy, 0, 20},
    };
    GpuCaptureFile gds_dma_only_capture;
    CHECK(capture_submit_items({}, {}, gds_dma_only_operations, meta, ordered_reader,
                               gds_dma_only_capture, error, {}, {}, {gds_copy}, 0,
                               initial_gds.data(), initial_gds.size()) &&
              gds_dma_only_capture.computes.empty() &&
              gds_dma_only_capture.dma_copies.size() == 1 &&
              gds_dma_only_capture.dma_copies[0].destination_blob_index <
                  gds_dma_only_capture.blobs.size(),
          "full-data memory-to-GDS capture owns a snapshot without a compute table");
    std::vector<uint8_t> gds_dma_only_bytes;
    GpuCaptureFile gds_dma_only_loaded;
    GpuReplayFrame gds_dma_only_replay;
    const bool gds_dma_only_materialized =
        serialize_gpu_capture(gds_dma_only_capture, gds_dma_only_bytes, error) &&
        deserialize_gpu_capture(gds_dma_only_bytes, gds_dma_only_loaded, error) &&
        materialize_gpu_replay(gds_dma_only_loaded, gds_dma_only_replay, error);
    const bool gds_dma_only_backed = gds_dma_only_materialized &&
              gds_dma_only_replay.computes.empty() &&
              gds_dma_only_replay.dma_copies.size() == 1 &&
              gds_dma_only_replay.dma_copies[0].destination_data &&
              gds_dma_only_replay.dma_copies[0].source_data &&
              gds_dma_only_replay.dma_copies[0].destination_data[0] == 0x19;
    CHECK(gds_dma_only_backed,
          "DMA-only GDS state serializes and materializes with its exact pre-submit byte");
    std::vector<SubmitOperation> gds_dma_only_replay_operations;
    for (const auto& operation : gds_dma_only_replay.operations)
        if (operation.realized)
            gds_dma_only_replay_operations.push_back(
                {operation.kind, static_cast<size_t>(operation.source_index),
                 operation.command_order});
    if (gds_dma_only_backed)
        execute_ordered_items(gds_dma_only_replay_operations, {}, {},
                              gds_dma_only_replay.dma_copies, {}, {}, 1, 1);
    CHECK(gds_dma_only_backed &&
              gds_dma_only_replay.dma_copies[0].destination_data[0] == 0xa1,
          "DMA-only replay executes against its independently captured GDS backing");

    // Thin captures retain truthful operation/selector metadata while making both omitted payloads
    // explicitly unavailable. This is the contract used by inspect/validate/graph.
    set_test_env("PROSPER_GPU_CAPTURE_METADATA_ONLY", "1");
    GpuCaptureFile gds_metadata_capture;
    const bool gds_metadata_captured = capture_submit_items(
        {}, {}, gds_dma_only_operations, meta, ordered_reader, gds_metadata_capture, error,
        {}, {}, {gds_copy}, 0, initial_gds.data(), initial_gds.size());
    set_test_env("PROSPER_GPU_CAPTURE_METADATA_ONLY", nullptr);
    std::vector<uint8_t> gds_metadata_bytes;
    GpuCaptureFile gds_metadata_loaded;
    GpuReplayFrame gds_metadata_replay;
    CHECK(gds_metadata_captured && gds_metadata_capture.blobs.empty() &&
              gds_metadata_capture.dma_copies.size() == 1 &&
              gds_metadata_capture.dma_copies[0].source_blob_index == 0xFFFFFFFFu &&
              gds_metadata_capture.dma_copies[0].destination_blob_index == 0xFFFFFFFFu &&
              serialize_gpu_capture(gds_metadata_capture, gds_metadata_bytes, error) &&
              deserialize_gpu_capture(gds_metadata_bytes, gds_metadata_loaded, error) &&
              materialize_gpu_replay(gds_metadata_loaded, gds_metadata_replay, error) &&
              gds_metadata_replay.dma_copies.size() == 1 &&
              !gds_metadata_replay.dma_copies[0].source_data &&
              !gds_metadata_replay.dma_copies[0].destination_data,
          "metadata-only memory-to-GDS capture materializes with unavailable backings explicit");

    GpuCaptureFile unsupported_gds_source = gds_capture;
    unsupported_gds_source.dma_copies[0].sels = 3u | (1u << 8) |
        kDmaDataAddressSource;
    CHECK(!serialize_gpu_capture(unsupported_gds_source, gds_capture_bytes, error) &&
              error == "unsupported ordered DMA GDS source selector",
          "capture rejects the unsupported GDS-to-memory direction fail-visibly");

    GpuCaptureFile unsupported_gds_memory_selector = gds_capture;
    unsupported_gds_memory_selector.dma_copies[0].sels =
        1u | (2u << 8) | kDmaDataAddressSource;
    CHECK(!serialize_gpu_capture(unsupported_gds_memory_selector, gds_capture_bytes, error) &&
              error == "unsupported ordered DMA memory-to-GDS selector form",
          "serialized memory-to-GDS capture rejects source selector 2 fail-visibly");

    GpuCaptureFile unsupported_gds_source_kind = gds_capture;
    unsupported_gds_source_kind.dma_copies[0].sels = 1u | (3u << 8);
    unsupported_gds_source_kind.dma_copies[0].src = 0x5010;
    CHECK(!serialize_gpu_capture(unsupported_gds_source_kind, gds_capture_bytes, error) &&
              error == "unsupported ordered DMA memory-to-GDS selector form",
          "serialized memory-to-GDS capture requires address-source semantics");

    std::array<uint8_t, 4> rejected_gds_destination = {0x55, 0x66, 0x77, 0x88};
    const std::array<uint8_t, 4> rejected_gds_source = {0xaa, 0xbb, 0xcc, 0xdd};
    ReplayDmaCopy rejected_gds_replay;
    rejected_gds_replay.dst = 0x24;
    rejected_gds_replay.src = 0x5010;
    rejected_gds_replay.bytes = 4;
    rejected_gds_replay.sels = 1u | (2u << 8) | kDmaDataAddressSource;
    rejected_gds_replay.command_order = 20;
    rejected_gds_replay.destination_data = rejected_gds_destination.data();
    rejected_gds_replay.destination_size = rejected_gds_destination.size();
    rejected_gds_replay.source_data = rejected_gds_source.data();
    rejected_gds_replay.source_size = rejected_gds_source.size();
    execute_ordered_items(gds_dma_only_operations, {}, {}, {rejected_gds_replay}, {}, {}, 1, 1);
    CHECK((rejected_gds_destination == std::array<uint8_t, 4>{0x55, 0x66, 0x77, 0x88}),
          "replay executor refuses a retained GDS copy with source selector 2");

    GpuState failed_state;
    set_pgm(failed_state, P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, kDiagnosticVs);
    set_pgm(failed_state, P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, kDiagnosticBadPs);
    failed_state.uc[P::VGT_PRIMITIVE_TYPE] = 4;
    failed_state.cx[P::CB_TARGET_MASK] = 0xf;
    failed_state.cx[P::CB_COLOR_CONTROL] =
        (P::CB_COLOR_CONTROL_MODE_NORMAL << P::CB_COLOR_CONTROL_MODE_SHIFT) |
        (0x66u << P::CB_COLOR_CONTROL_ROP3_SHIFT);
    failed_state.cx[P::CB_COLOR0_BASE] = 0x1234;
    failed_state.cx[P::PA_SC_SCREEN_SCISSOR_TL] = 0x000A0009u;
    failed_state.cx[P::PA_SC_SCREEN_SCISSOR_BR] = 0x0028001Eu;
    failed_state.draws.push_back({3});
    failed_state.draws.back().command_order = 777;
    GpuCaptureFile failed_capture;
    CHECK(capture_gpustate_submit(failed_state, 99, 640, 360, meta, failed_capture, error),
          "actual realization path captures a deliberately failed synthetic draw");
    CHECK(failed_capture.draws.empty() && failed_capture.operations.size() == 1 &&
          !failed_capture.operations[0].realized && failed_capture.failure_diagnostics_available &&
          failed_capture.failure_diagnostics.size() == 1,
          "unrealized operation retains one explicit v7 failure diagnostic");
    const auto& failed = failed_capture.failure_diagnostics[0];
    const auto failed_stage = std::find_if(failed.stages.begin(), failed.stages.end(), [](const auto& stage) {
        return stage.stage == ShaderProgramStage::Fragment;
    });
    CHECK(failed.reason == RealizationFailureReason::ShaderRecompile && failed.pipeline_present &&
          failed.pipeline.color_write_mask == 0xf && failed.vertex_count == 3 &&
          failed.pipeline.logic_op_enable && failed.pipeline.logic_op == 6 &&
          failed.pipeline.has_scissor && failed.pipeline.scissor_left == 9 &&
          failed.pipeline.scissor_top == 10 && failed.pipeline.scissor_right == 30 &&
          failed.pipeline.scissor_bottom == 40 &&
          failed_stage != failed.stages.end(),
          "diagnostic distinguishes shader rejection and retains decoded draw state");
    CHECK(failed_stage != failed.stages.end() && !failed_stage->recompiled &&
          failed_stage->coverage.unsupported == 1 && failed_stage->coverage.first_bad_pc == 1 &&
          failed_stage->coverage.first_bad_op == 0x02,
          "failed fragment stage retains the exact rejection opcode and dword PC");
    const GpuCaptureRawShaderVersion* failed_raw =
        failed_stage != failed.stages.end() &&
        failed_stage->raw_shader_index < failed_capture.raw_shader_versions.size()
            ? &failed_capture.raw_shader_versions[failed_stage->raw_shader_index] : nullptr;
    CHECK(failed_raw && failed_raw->has_endpgm &&
          failed_raw->words == std::vector<uint32_t>(std::begin(kDiagnosticBadPs), std::end(kDiagnosticBadPs)) &&
          failed_raw->content_hash == gpu_capture_hash(
              reinterpret_cast<const uint8_t*>(kDiagnosticBadPs), sizeof(kDiagnosticBadPs)),
          "failed stage retains the exact content-addressed raw stream through s_endpgm");

    OperationRealizationFailure compute_failure;
    compute_failure.kind = SubmitOperationKind::Dispatch;
    compute_failure.index = 8;
    compute_failure.command_order = 1026;
    compute_failure.reason = RealizationFailureReason::ShaderRecompile;
    compute_failure.compute_launch = {37, 9, 1, 16, 8, 1, 3, 2, 1};
    ShaderRealizationDiagnostic compute_stage;
    compute_stage.stage = ShaderProgramStage::Compute;
    compute_stage.program_addr = reinterpret_cast<uint64_t>(kDiagnosticCompute);
    compute_stage.resources = std::make_shared<ShaderResourceTable>();
    compute_stage.recompile_config_available = true;
    compute_stage.recompile_config.user_sgprs = {
        0x12345678u, 0x9abcdef0u, 0x0badc0deu,
    };
    compute_stage.recompile_config.local_x = 16;
    compute_stage.recompile_config.local_y = 8;
    compute_stage.recompile_config.local_z = 1;
    compute_stage.recompile_config.exact_thread_extent = true;
    compute_stage.recompile_config.threads_x = 37;
    compute_stage.recompile_config.threads_y = 9;
    compute_stage.recompile_config.threads_z = 1;
    compute_stage.recompile_config.wave_size = 64;
    compute_stage.recompile_config.compute_pgm_rsrc1 = 0x402c008au;
    compute_stage.recompile_config.tidig_comp_cnt = 2;
    compute_stage.recompile_config.tgid_x_en = true;
    compute_stage.recompile_config.tgid_y_en = true;
    compute_stage.recompile_config.tg_size_en = true;
    compute_stage.recompile_config.lds_bytes = 512;
    compute_stage.recompile_config.native_subgroup_size = 64;
    compute_stage.recompile_config.native_storage_format_support =
        native_storage_format_support_bit(DataFormat::Float16, 4);
    compute_failure.stages.push_back(compute_stage);
    const std::vector<SubmitOperation> failed_compute_operations = {
        {SubmitOperationKind::Dispatch, 8, 1026},
    };
    GpuCaptureFile failed_compute_capture;
    CHECK(capture_submit_items({}, {}, failed_compute_operations, meta, reader,
                               failed_compute_capture, error, {}, {compute_failure}) &&
              failed_compute_capture.failure_diagnostics.size() == 1 &&
              failed_compute_capture.failure_diagnostics[0].stages.size() == 1 &&
              failed_compute_capture.failure_diagnostics[0].stages[0]
                  .recompile_config_available,
          "v42 capture retains exact specialization for a failed compute translation");
    std::vector<uint8_t> failed_compute_bytes;
    GpuCaptureFile failed_compute_loaded;
    CHECK(serialize_gpu_capture(failed_compute_capture, failed_compute_bytes, error) &&
              deserialize_gpu_capture(failed_compute_bytes, failed_compute_loaded, error) &&
              failed_compute_loaded.format_version == 50 &&
              failed_compute_loaded.failure_diagnostics[0].compute_launch.threads_x == 37 &&
              failed_compute_loaded.failure_diagnostics[0].stages[0]
                      .recompile_config.user_sgprs ==
                  compute_stage.recompile_config.user_sgprs &&
              failed_compute_loaded.failure_diagnostics[0].stages[0]
                      .recompile_config.local_y == 8 &&
              failed_compute_loaded.failure_diagnostics[0].stages[0]
                      .recompile_config.compute_pgm_rsrc1 == 0x402c008au &&
              failed_compute_loaded.failure_diagnostics[0].stages[0]
                      .recompile_config.native_subgroup_size == 64,
          "v42 failed-compute launch and user-SGPR ABI round-trips exactly");

    std::vector<uint8_t> v41_failed_compute_bytes = failed_compute_bytes;
    v41_failed_compute_bytes.resize(
        v41_failed_compute_bytes.size() - v50_tail(failed_compute_capture) -
            v49_tail(failed_compute_capture) -
            v48_tail(failed_compute_capture) -
            v47_tail(failed_compute_capture) -
        v46_tail(failed_compute_capture) -
            v45_tail(failed_compute_capture) -
            v44_tail(failed_compute_capture) -
            v43_tail(failed_compute_capture) -
            v42_tail(failed_compute_capture));
    v41_failed_compute_bytes[8] = 41;
    v41_failed_compute_bytes[9] = v41_failed_compute_bytes[10] =
        v41_failed_compute_bytes[11] = 0;
    GpuCaptureFile v41_failed_compute_loaded;
    CHECK(deserialize_gpu_capture(v41_failed_compute_bytes, v41_failed_compute_loaded, error) &&
              !v41_failed_compute_loaded.failure_diagnostics[0].stages[0]
                   .recompile_config_available,
          "v41 failed-compute captures remain readable with specialization unavailable");

    GpuCaptureFile invalid_failed_compute = failed_compute_capture;
    invalid_failed_compute.failure_diagnostics[0].stages[0]
        .recompile_config.user_sgprs.resize(33);
    CHECK(!serialize_gpu_capture(invalid_failed_compute, failed_compute_bytes, error) &&
              error == "invalid failed-compute recompile state",
          "capture rejects an oversized failed-compute user-SGPR ABI");
    invalid_failed_compute = failed_compute_capture;
    invalid_failed_compute.failure_diagnostics[0].stages[0].recompile_config.local_x = 8;
    CHECK(!serialize_gpu_capture(invalid_failed_compute, failed_compute_bytes, error) &&
              error == "invalid failed-compute recompile state",
          "capture rejects failed-compute specialization that disagrees with its launch");

    // Live one-shot capture receives the already-realized backend list as well as the original
    // semantic state. A failed operation is necessarily absent from that backend list, but must not
    // lose the exact reason/program/pipeline that the semantic realization can still diagnose.
    request_interactive_gpu_capture(prosper_test::test_scratch_file("prosper_semantic_failure_diagnostic.prgcap"));
    auto semantic_failure_pending = begin_requested_gpu_capture(
        {}, {}, plan_submit_operations(failed_state), 640, 360, &failed_state, 101,
        failed_state.draws.size());
    CHECK(semantic_failure_pending && !interactive_gpu_capture_armed() &&
          semantic_failure_pending->capture.operations.size() == 1 &&
          !semantic_failure_pending->capture.operations[0].realized &&
          semantic_failure_pending->capture.failure_diagnostics.size() == 1 &&
          semantic_failure_pending->capture.failure_diagnostics[0].reason ==
              RealizationFailureReason::ShaderRecompile &&
          semantic_failure_pending->capture.failure_diagnostics[0].pipeline_present &&
          semantic_failure_pending->capture.failure_diagnostics[0].stages.size() == 2,
          "one-shot semantic capture preserves exact diagnostics for unrealized draws");

    std::array<uint8_t, 32> dma_memory{};
    for (size_t i = 0; i < 16; ++i) dma_memory[16 + i] = static_cast<uint8_t>(0xa0 + i);
    GpuState dma_state;
    dma_state.dma_copies.push_back({reinterpret_cast<uint64_t>(dma_memory.data()),
                                    reinterpret_cast<uint64_t>(dma_memory.data() + 16),
                                    16, 0, 55, 0x12345678});
    GpuCaptureFile dma_capture;
    error.clear();
    CHECK(capture_gpustate_submit(dma_state, 100, 640, 360, meta, dma_capture, error) &&
          dma_capture.dma_copies.size() == 1 && dma_capture.operations.size() == 1 &&
          dma_capture.operations[0].kind == SubmitOperationKind::DmaCopy &&
          dma_capture.operations[0].realized && dma_capture.blobs.size() == 1 &&
          dma_capture.dma_copies[0].source_blob_offset == 16,
          "central GpuState capture retains ordered DMA endpoints and backing closure");
    GpuCaptureFile target_dma_capture;
    CHECK(capture_gpustate_target_submit(
              dma_state, 100, 640, 360, 640, 360, meta, target_dma_capture, error) &&
          target_dma_capture.dma_copies.size() == 1 &&
          target_dma_capture.operations[0].kind == SubmitOperationKind::DmaCopy,
          "target-selected capture preserves ordered DMA instead of bypassing it");

    // A selected live DMA capture consumes exactly one selector match and cannot silently retarget.
    set_test_env("PROSPER_GPU_CAPTURE", "ordered-dma-selector.prgcap");
    set_test_env("PROSPER_GPU_CAPTURE_AT", "0");
    set_test_env("PROSPER_GPU_CAPTURE_COMPUTE_ADDR", "not-an-address");
    set_test_env("PROSPER_GPU_CAPTURE_SHADER_ADDR", "not-an-address");
    set_test_env("PROSPER_GPU_CAPTURE_TARGET_DIM", "not-a-dimension");
    const auto invalid_selector_pending = begin_requested_gpu_capture({}, {}, {}, 1, 1,
                                                                      &dma_state, 98, 0);
    set_test_env("PROSPER_GPU_CAPTURE_COMPUTE_ADDR", "0x30130f0000");
    set_test_env("PROSPER_GPU_CAPTURE_SHADER_ADDR", "0x1");
    auto selected_dispatch_state = std::make_shared<GpuState>();
    selected_dispatch_state->sh[prosper::agc::Pm4::COMPUTE_PGM_LO] = 0x30130e00u;
    selected_dispatch_state->sh[prosper::agc::Pm4::COMPUTE_PGM_HI] = 0u;
    GpuState::Dispatch selected_dispatch;
    selected_dispatch.state = selected_dispatch_state;
    dma_state.dispatches.push_back(selected_dispatch);
    auto selected_draw_state = std::make_shared<GpuState>();
    set_pgm(*selected_draw_state, P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES,
            kDiagnosticVs);
    set_pgm(*selected_draw_state, P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS,
            kDiagnosticBadPs);
    selected_draw_state->cx[P::CB_TARGET_MASK] = 0xf;
    selected_draw_state->cx[P::CB_COLOR0_ATTRIB2] = 0; // dimension-minus-one fields: 1x1
    GpuState::Draw selected_draw{3};
    selected_draw.state = selected_draw_state;
    dma_state.draws.push_back(selected_draw);
    const auto skipped_dma_pending = begin_requested_gpu_capture({}, {}, {}, 1, 1,
                                                                 &dma_state, 99, 0);
    selected_dispatch_state->sh[prosper::agc::Pm4::COMPUTE_PGM_LO] = 0x30130f00u;
    char selected_shader_addr[32];
    std::snprintf(selected_shader_addr, sizeof(selected_shader_addr), "0x%llx",
                  static_cast<unsigned long long>(reinterpret_cast<uint64_t>(kDiagnosticBadPs)));
    set_test_env("PROSPER_GPU_CAPTURE_SHADER_ADDR", selected_shader_addr);
    set_test_env("PROSPER_GPU_CAPTURE_TARGET_DIM", "2x1");
    const auto wrong_dimension_pending = begin_requested_gpu_capture({}, {}, {}, 1, 1,
                                                                     &dma_state, 100, 1);
    set_test_env("PROSPER_GPU_CAPTURE_TARGET_DIM", "1x1");
    const auto dma_pending = begin_requested_gpu_capture({}, {}, {}, 1, 1,
                                                         &dma_state, 100, 1);
    const auto retargeted_pending = begin_requested_gpu_capture({}, {}, {}, 1, 1);
    CHECK(!invalid_selector_pending && !skipped_dma_pending && !wrong_dimension_pending &&
              dma_pending && !retargeted_pending && dma_pending->capture.dma_copies.size() == 1,
          "compute/shader/target-selected DMA capture rejects invalid/nonmatches and cannot retarget");
    set_test_env("PROSPER_GPU_CAPTURE_TARGET_DIM", nullptr);
    set_test_env("PROSPER_GPU_CAPTURE_SHADER_ADDR", nullptr);
    set_test_env("PROSPER_GPU_CAPTURE_COMPUTE_ADDR", nullptr);
    set_test_env("PROSPER_GPU_CAPTURE_AT", nullptr);
    set_test_env("PROSPER_GPU_CAPTURE", nullptr);

    const auto failed_path = prosper_test::test_scratch_dir() /
        "prosper_gpu_capture_failed_diagnostic_test.prgcap";
    failed_capture.failure_diagnostics[0].pipeline.cb_resolve = true;
    // NGG draws can have a separately allocated vertex prolog and vertex main.  A failed capture
    // must retain both same-kind programs so gpu_replay can dump either half of the linked stage.
    auto linked_vertex = failed_capture.failure_diagnostics[0].stages[0];
    linked_vertex.program_addr += 0x1000;
    failed_capture.failure_diagnostics[0].stages.push_back(linked_vertex);
    auto mutable_failed_stage = std::find_if(
        failed_capture.failure_diagnostics[0].stages.begin(),
        failed_capture.failure_diagnostics[0].stages.end(), [](const auto& stage) {
            return stage.stage == ShaderProgramStage::Fragment;
        });
    CHECK(mutable_failed_stage != failed_capture.failure_diagnostics[0].stages.end(),
          "failed-stage v41 fixture finds the fragment stage");
    GpuCapturedResource failed_shadow;
    failed_shadow.resource.cls = ResourceClass::Texture;
    failed_shadow.resource.format = DataFormat::Float16;
    failed_shadow.resource.num_components = 4;
    failed_shadow.resource.binding = 97;
    failed_shadow.resource.gpu_addr = 0x5005d7d000ull;
    failed_shadow.resource.size = 4u * 262144u;
    failed_shadow.resource.srt_offset = 64;
    failed_shadow.resource.sgpr_base = 8;
    failed_shadow.resource.fetch_pc = 1095;
    failed_shadow.resource.fetch_index_mode = VertexFetchIndexMode::Instance;
    failed_shadow.resource.bvh_box_grow = 9;
    failed_shadow.resource.flat_base_sgpr = 12;
    failed_shadow.resource.img_dim = 5;
    failed_shadow.resource.width = 128;
    failed_shadow.resource.height = 64;
    failed_shadow.resource.depth = 4;
    failed_shadow.resource.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbRX);
    failed_shadow.resource.linear_row_pitch_bytes = 1024;
    failed_shadow.resource.declared_mip_levels = 3;
    failed_shadow.resource.in_mip_tail = true;
    failed_shadow.resource.mip_tail_offset = 4096;
    failed_shadow.resource.mip_tail_bytes = 65536;
    failed_shadow.resource.mip_tail_x = 32;
    failed_shadow.resource.mip_tail_y = 16;
    failed_shadow.resource.layer_stride_bytes = 262144;
    failed_shadow.resource.depth_compare = true;
    failed_shadow.resource.depth_compare_func = 3;
    failed_shadow.resource.max_uncompressed_block_size = 2;
    failed_shadow.resource.max_compressed_block_size = 1;
    failed_shadow.resource.meta_pipe_aligned = true;
    failed_shadow.resource.write_compress_enabled = true;
    failed_shadow.resource.compression_enabled = true;
    failed_shadow.resource.alpha_is_on_msb = true;
    failed_shadow.resource.color_transform = true;
    failed_shadow.resource.metadata_addr = 0x5006d00000ull;
    failed_shadow.captured_size = 987654;
    failed_shadow.metadata_size = gpu_capture_dcc_metadata_footprint(failed_shadow.resource);
    GpuCapturedResource failed_msaa;
    failed_msaa.resource.cls = ResourceClass::Texture;
    failed_msaa.resource.format = DataFormat::Float32;
    failed_msaa.resource.num_components = 1;
    failed_msaa.resource.binding = 98;
    failed_msaa.resource.gpu_addr = 0x5007d00000ull;
    failed_msaa.resource.size = 65536;
    failed_msaa.resource.img_dim = 6;
    failed_msaa.resource.width = 64;
    failed_msaa.resource.height = 64;
    failed_msaa.resource.depth = 1;
    failed_msaa.resource.sample_count = 4;
    failed_msaa.resource.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbZX);
    failed_msaa.resource.declared_mip_levels = 1;
    failed_msaa.captured_size = 65536;
    if (mutable_failed_stage != failed_capture.failure_diagnostics[0].stages.end()) {
        mutable_failed_stage->resource_table.present = true;
        mutable_failed_stage->resource_table.resources = {failed_shadow, failed_msaa};
        mutable_failed_stage->resource_table_present = true;
        mutable_failed_stage->resource_count = 2;
    }
    CHECK(write_gpu_capture(failed_path.string(), failed_capture, error),
          "failed-operation diagnostic capture writes linked vertex programs");
    GpuCaptureFile failed_loaded;
    CHECK(read_gpu_capture(failed_path.string(), failed_loaded, error) &&
          failed_loaded.failure_diagnostics_available && failed_loaded.failure_diagnostics.size() == 1 &&
          failed_loaded.raw_shader_versions.size() == failed_capture.raw_shader_versions.size() &&
          failed_loaded.failure_diagnostics[0].pipeline.has_scissor &&
          failed_loaded.failure_diagnostics[0].pipeline.scissor_left == 9 &&
          failed_loaded.failure_diagnostics[0].pipeline.logic_op_enable &&
          failed_loaded.failure_diagnostics[0].pipeline.logic_op == 6 &&
          failed_loaded.failure_diagnostics[0].pipeline.cb_resolve &&
          failed_loaded.failure_diagnostics[0].stages.size() == 3 &&
          failed_loaded.failure_diagnostics[0].stages[0].stage == ShaderProgramStage::Vertex &&
          failed_loaded.failure_diagnostics[0].stages[2].stage == ShaderProgramStage::Vertex &&
          failed_loaded.failure_diagnostics[0].stages[0].program_addr !=
              failed_loaded.failure_diagnostics[0].stages[2].program_addr &&
          failed_loaded.failure_diagnostics[0].stages[1].coverage.first_bad_pc == 1,
          "linked stage state, coverage, and raw shader versions round-trip offline");
    const auto loaded_failed_stage = std::find_if(
        failed_loaded.failure_diagnostics[0].stages.begin(),
        failed_loaded.failure_diagnostics[0].stages.end(), [](const auto& stage) {
            return stage.stage == ShaderProgramStage::Fragment;
        });
    const GpuCapturedResource* loaded_shadow = nullptr;
    const GpuCapturedResource* loaded_failed_msaa = nullptr;
    if (loaded_failed_stage != failed_loaded.failure_diagnostics[0].stages.end()) {
        for (const auto& resource : loaded_failed_stage->resource_table.resources) {
            if (resource.resource.binding == failed_shadow.resource.binding)
                loaded_shadow = &resource;
            if (resource.resource.binding == failed_msaa.resource.binding)
                loaded_failed_msaa = &resource;
        }
    }
    CHECK(failed_loaded.format_version == 50 && loaded_shadow &&
          loaded_shadow->resource.depth == 4 &&
          loaded_shadow->resource.max_uncompressed_block_size == 2 &&
          loaded_shadow->resource.max_compressed_block_size == 1 &&
          loaded_shadow->resource.meta_pipe_aligned &&
          loaded_shadow->resource.write_compress_enabled &&
          loaded_shadow->resource.compression_enabled &&
          loaded_shadow->resource.alpha_is_on_msb &&
          loaded_shadow->resource.color_transform &&
          loaded_shadow->resource.metadata_addr == 0x5006d00000ull &&
          loaded_shadow->resource.depth_compare && loaded_shadow->resource.in_mip_tail &&
          loaded_shadow->resource.mip_tail_offset == 4096 &&
          loaded_shadow->resource.mip_tail_bytes == 65536 &&
          loaded_shadow->resource.mip_tail_x == 32 &&
          loaded_shadow->resource.mip_tail_y == 16 &&
          loaded_shadow->resource.declared_mip_levels == 3 &&
          loaded_shadow->resource.linear_row_pitch_bytes == 1024 &&
          loaded_shadow->captured_size == 987654 &&
          loaded_shadow->resource.fetch_index_mode == VertexFetchIndexMode::Instance &&
          loaded_shadow->resource.layer_stride_bytes == 262144 &&
          loaded_shadow->resource.layer_mip_offset_bytes == 0 &&
          loaded_shadow->resource.flat_base_sgpr == 12 &&
          loaded_shadow->resource.bvh_box_grow == 9 &&
          loaded_shadow->metadata_size == failed_shadow.metadata_size &&
          loaded_shadow->resource.dcc_metadata_size == failed_shadow.metadata_size &&
          loaded_failed_msaa && loaded_failed_msaa->resource.img_dim == 6 &&
          loaded_failed_msaa->resource.sample_count == 4 &&
          loaded_failed_msaa->captured_size == 65536,
          "v41 failed-stage resources round-trip complete descriptor and codegen state");

    std::vector<uint8_t> malformed_failed_sample_bytes;
    CHECK(serialize_gpu_capture(failed_capture, malformed_failed_sample_bytes, error),
          "v44 failed-stage sample identity prepares a corruption fixture");
    if (malformed_failed_sample_bytes.size() >= sizeof(uint32_t)) {
        const uint32_t invalid_samples = 3;
        // Fragment is the final stage with resources, and its MSAA descriptor is the final resource,
        // so this is exactly its v44 sample-count record rather than a preceding extension field.
        std::memcpy(malformed_failed_sample_bytes.data() +
                        malformed_failed_sample_bytes.size() - v50_tail(failed_capture) -
                        v49_tail(failed_capture) -
                        v48_tail(failed_capture) -
                        v47_tail(failed_capture) -
                        v46_tail(failed_capture) -
                            v45_tail(failed_capture) -
                            sizeof(uint32_t),
                    &invalid_samples, sizeof(invalid_samples));
    }
    GpuCaptureFile malformed_failed_sample_loaded;
    CHECK(!deserialize_gpu_capture(malformed_failed_sample_bytes,
                                   malformed_failed_sample_loaded, error) &&
              error == "invalid resource sample count",
          "v44 reader rejects a malformed failed-stage sample-count tail");

    std::vector<uint8_t> v40_failed_bytes;
    GpuCaptureFile v40_failed_loaded;
    CHECK(serialize_gpu_capture(failed_capture, v40_failed_bytes, error) &&
              v40_failed_bytes.size() >=
                  v50_tail(failed_capture) + v49_tail(failed_capture) +
                  v48_tail(failed_capture) +
                  v47_tail(failed_capture) +
                  v46_tail(failed_capture) +
                  v45_tail(failed_capture) +
                      v44_tail(failed_capture) +
                      v43_tail(failed_capture) +
                      v42_tail(failed_capture) + v41_tail(failed_capture),
          "v41 failed-stage resource tail is removable for a v40 compatibility fixture");
    if (v40_failed_bytes.size() >=
        v50_tail(failed_capture) + v49_tail(failed_capture) +
        v48_tail(failed_capture) +
        v47_tail(failed_capture) +
        v46_tail(failed_capture) +
        v45_tail(failed_capture) +
            v44_tail(failed_capture) +
            v43_tail(failed_capture) +
            v42_tail(failed_capture) + v41_tail(failed_capture)) {
        v40_failed_bytes.resize(v40_failed_bytes.size() - v50_tail(failed_capture));
        v40_failed_bytes.resize(v40_failed_bytes.size() - v49_tail(failed_capture));
        v40_failed_bytes.resize(v40_failed_bytes.size() - v48_tail(failed_capture));
        v40_failed_bytes.resize(v40_failed_bytes.size() - v47_tail(failed_capture));
        v40_failed_bytes.resize(v40_failed_bytes.size() - v46_tail(failed_capture));
        v40_failed_bytes.resize(v40_failed_bytes.size() - v45_tail(failed_capture));
        v40_failed_bytes.resize(v40_failed_bytes.size() - v44_tail(failed_capture));
        v40_failed_bytes.resize(v40_failed_bytes.size() - v43_tail(failed_capture));
        v40_failed_bytes.resize(v40_failed_bytes.size() - v42_tail(failed_capture));
        v40_failed_bytes.resize(v40_failed_bytes.size() - v41_tail(failed_capture));
        v40_failed_bytes[8] = 40;
        v40_failed_bytes[9] = v40_failed_bytes[10] = v40_failed_bytes[11] = 0;
    }
    CHECK(deserialize_gpu_capture(v40_failed_bytes, v40_failed_loaded, error),
          "v40 failed-stage capture remains readable without the v41 state tail");
    const auto v40_failed_stage = std::find_if(
        v40_failed_loaded.failure_diagnostics[0].stages.begin(),
        v40_failed_loaded.failure_diagnostics[0].stages.end(), [](const auto& stage) {
            return stage.stage == ShaderProgramStage::Fragment;
        });
    const GpuCapturedResource* v40_shadow = nullptr;
    if (v40_failed_stage != v40_failed_loaded.failure_diagnostics[0].stages.end())
        for (const auto& resource : v40_failed_stage->resource_table.resources)
            if (resource.resource.binding == failed_shadow.resource.binding)
                v40_shadow = &resource;
    CHECK(v40_failed_loaded.format_version == 40 && v40_shadow &&
          v40_shadow->resource.bvh_box_grow == 9 && v40_shadow->resource.depth == 1 &&
          !v40_shadow->resource.meta_pipe_aligned &&
          !v40_shadow->resource.write_compress_enabled &&
          !v40_shadow->resource.compression_enabled &&
          !v40_shadow->resource.alpha_is_on_msb &&
          !v40_shadow->resource.color_transform &&
          v40_shadow->resource.max_uncompressed_block_size == 0 &&
          v40_shadow->resource.max_compressed_block_size == 0 &&
          v40_shadow->resource.metadata_addr == 0 && !v40_shadow->resource.depth_compare &&
          !v40_shadow->resource.in_mip_tail && v40_shadow->resource.mip_tail_offset == 0 &&
          v40_shadow->resource.mip_tail_bytes == 0 && v40_shadow->resource.mip_tail_x == 0 &&
          v40_shadow->resource.mip_tail_y == 0 &&
          v40_shadow->resource.declared_mip_levels == 1 &&
          v40_shadow->resource.linear_row_pitch_bytes == 0 && v40_shadow->captured_size == 0 &&
          v40_shadow->resource.fetch_index_mode == VertexFetchIndexMode::Automatic &&
          v40_shadow->resource.layer_stride_bytes == 0 &&
          v40_shadow->resource.layer_mip_offset_bytes == 0 &&
          v40_shadow->resource.flat_base_sgpr == 0xFFFFFFFFu &&
          v40_shadow->metadata_size == 0 && v40_shadow->resource.dcc_metadata_size == 0,
          "v40 failed-stage resources keep explicit historical defaults for missing v41 state");

    GpuCaptureFile stale_raw = failed_capture;
    stale_raw.raw_shader_versions[0].content_hash ^= 1;
    CHECK(!serialize_gpu_capture(stale_raw, repeated, error) &&
          error == "raw shader content hash mismatch",
          "writer rejects stale failed-shader content identity");
    GpuCaptureFile bad_raw_index = failed_capture;
    bad_raw_index.failure_diagnostics[0].stages[0].raw_shader_index = 0xfffffffeu;
    CHECK(!serialize_gpu_capture(bad_raw_index, repeated, error) &&
          error == "invalid failed-stage diagnostic metadata",
          "writer rejects an out-of-range failed-shader reference");
    GpuCaptureFile bad_realized_raw_index = captured;
    bad_realized_raw_index.draws[0].vs_raw_shader_index = 0xfffffffeu;
    CHECK(!serialize_gpu_capture(bad_realized_raw_index, repeated, error) &&
          error == "realized draw references an invalid raw shader",
          "writer rejects an out-of-range realized-shader reference");
    GpuCaptureFile bad_pixel_abi = captured;
    bad_pixel_abi.draws[0].pixel_inputs.passthrough_mask = 4;
    CHECK(!serialize_gpu_capture(bad_pixel_abi, repeated, error) &&
          error == "invalid realized-draw pixel-stage ABI",
          "writer rejects pixel passthrough state outside the retained valid-input mask");
    GpuCaptureFile oversized_raw = failed_capture;
    oversized_raw.raw_shader_versions[0].words.resize(0x4001, 0xbf810000u);
    oversized_raw.raw_shader_versions[0].content_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(oversized_raw.raw_shader_versions[0].words.data()),
        oversized_raw.raw_shader_versions[0].words.size() * sizeof(uint32_t));
    CHECK(!serialize_gpu_capture(oversized_raw, repeated, error) &&
          error == "raw shader data exceeds its bounded limit",
          "writer enforces the documented 64 KiB per-stage raw bound");

    GpuCaptureFile legacy_source = captured; legacy_source.ds_seeds.clear();
    legacy_source.raw_shader_versions.clear();
    legacy_source.draws[0].vs_raw_shader_index = 0xFFFFFFFFu;
    legacy_source.draws[0].fs_raw_shader_index = 0xFFFFFFFFu;
    legacy_source.draws[0].vs_chain_raw_shader_index = 0xFFFFFFFFu;
    legacy_source.draws[0].vertex_lds_dwords = 0;
    std::vector<uint8_t> legacy_bytes;
    CHECK(serialize_gpu_capture(legacy_source, legacy_bytes, error) && legacy_bytes.size() >= 32,
          "created a diagnostic-free v22 payload for legacy-reader fixtures");
    std::vector<uint8_t> v13_bytes = legacy_bytes;
    if (v13_bytes.size() >= 32) {
        const size_t legacy_resource_count = legacy_source.draws[0].vrt.resources.size() +
                                             legacy_source.draws[0].prt.resources.size();
        v13_bytes.resize(v13_bytes.size() - v50_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v49_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v48_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v47_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v46_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v45_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v44_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v43_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v42_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v41_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v40_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v39_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v37_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v36_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v35_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v34_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v32_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v31_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v30_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v29_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v28_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v27_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v26_tail(legacy_source));
        v13_bytes.resize(v13_bytes.size() - v25_tail(legacy_source)); // v25 resolve tail
        v13_bytes.resize(v13_bytes.size() - v24_tail(legacy_source)); // v24 declared-mip tail
        v13_bytes.resize(v13_bytes.size() - 12); // v23 count + one raw-draw-state
        v13_bytes.resize(v13_bytes.size() - (4 + 8 * legacy_resource_count)); // v22
        v13_bytes.resize(v13_bytes.size() - 13); // v21 one logic-op draw + zero failures
        v13_bytes.resize(v13_bytes.size() - 8); // v20 draw count + instance count
        v13_bytes.resize(v13_bytes.size() - 12); // v19 count + two raw indices
        v13_bytes.resize(v13_bytes.size() - 8); // v18 draw count + no-GS sentinel
        v13_bytes.resize(v13_bytes.size() - 25); // v17 one-draw scissor tail, zero failures
        v13_bytes.resize(v13_bytes.size() - (4 + 17 * legacy_resource_count));
        v13_bytes.resize(v13_bytes.size() - (4 + legacy_resource_count));
        v13_bytes.resize(v13_bytes.size() - 4);
        v13_bytes[8] = 13; v13_bytes[9] = v13_bytes[10] = v13_bytes[11] = 0;
    }
    GpuCaptureFile v13_loaded;
    CHECK(deserialize_gpu_capture(v13_bytes, v13_loaded, error) &&
          v13_loaded.dma_copies.empty() &&
          std::none_of(v13_loaded.operations.begin(), v13_loaded.operations.end(),
                       [](const auto& operation) {
                           return operation.kind == SubmitOperationKind::DmaCopy;
                       }),
          "v13 readers remain backward-compatible without inventing DMA operations");
    if (legacy_bytes.size() >= 32) {
        const size_t legacy_resource_count = legacy_source.draws[0].vrt.resources.size() +
                                             legacy_source.draws[0].prt.resources.size();
        legacy_bytes.resize(legacy_bytes.size() - v50_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v49_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v48_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v47_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v46_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v45_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v44_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v43_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v42_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v41_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v40_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v39_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v37_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v36_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v35_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v34_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v32_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v31_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v30_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v29_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v28_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v27_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v26_tail(legacy_source));
        legacy_bytes.resize(legacy_bytes.size() - v25_tail(legacy_source)); // v25 resolve tail
        legacy_bytes.resize(legacy_bytes.size() - v24_tail(legacy_source)); // v24 declared-mip tail
        legacy_bytes.resize(legacy_bytes.size() - 12); // v23 count + one raw-draw-state
        legacy_bytes.resize(legacy_bytes.size() - (4 + 8 * legacy_resource_count)); // v22
        legacy_bytes.resize(legacy_bytes.size() - 13); // v21 one logic-op draw + zero failures
        legacy_bytes.resize(legacy_bytes.size() - 8); // v20 draw count + instance count
        legacy_bytes.resize(legacy_bytes.size() - 12); // v19 count + two raw indices
        legacy_bytes.resize(legacy_bytes.size() - 8); // v18 draw count + no-GS sentinel
        legacy_bytes.resize(legacy_bytes.size() - 25); // v17 one-draw scissor tail, zero failures
        legacy_bytes.resize(legacy_bytes.size() - (4 + 17 * legacy_resource_count));
        legacy_bytes.resize(legacy_bytes.size() - (4 + legacy_resource_count));
        legacy_bytes.resize(legacy_bytes.size() - 4); // remove v14 zero-DMA count
        // Remove the v12 DCC metadata-reference tail: count + 20-byte reference per resource.
        legacy_bytes.resize(legacy_bytes.size() - (4 + 20 * legacy_resource_count));
        // Remove the v11 DCC tail: count + 17-byte state record per resource.
        legacy_bytes.resize(legacy_bytes.size() - (4 + 17 * legacy_resource_count));
        // v13 inserted one format dword between every RTT seed extent and byte vector. Remove the
        // fixture's only format word to recover a byte-exact v10 prefix before trimming older tails.
        const auto& seed = legacy_source.rtt_seeds[0];
        std::vector<uint8_t> marker;
        auto append_u32 = [&](uint32_t value) {
            for (unsigned i = 0; i < 4; ++i) marker.push_back(uint8_t(value >> (8 * i)));
        };
        auto append_u64 = [&](uint64_t value) {
            for (unsigned i = 0; i < 8; ++i) marker.push_back(uint8_t(value >> (8 * i)));
        };
        append_u64(seed.guest_addr); append_u32(seed.width); append_u32(seed.height);
        append_u32(static_cast<uint32_t>(seed.format)); append_u64(seed.rgba.size());
        auto seed_header = std::search(legacy_bytes.begin(), legacy_bytes.end(),
                                       marker.begin(), marker.end());
        if (seed_header != legacy_bytes.end())
            legacy_bytes.erase(seed_header + 16, seed_header + 20);
        legacy_bytes[8] = 10;
        // Remove the v10 MRT tail: draw count + one (target identity + 50-byte pipeline state) +
        // zero failure count. The remaining payload is byte-exact v9.
        legacy_bytes.resize(legacy_bytes.size() - (4 + 16 + 50 + 4));
        legacy_bytes.resize(legacy_bytes.size() - 4 - 4 - 4 * legacy_resource_count);
        // remove v9 depth section (count + values) and the v8 DS-seed zero count
        legacy_bytes[8] = 7; legacy_bytes[9] = legacy_bytes[10] = legacy_bytes[11] = 0;
    }
    GpuCaptureFile legacy_loaded;
    CHECK(deserialize_gpu_capture(legacy_bytes, legacy_loaded, error) &&
          legacy_loaded.failure_diagnostics_available && legacy_loaded.ds_seeds.empty(),
          "v7 capture reopens without persistent DS checkpoint data");
    if (legacy_bytes.size() >= 20) {
        legacy_bytes.resize(legacy_bytes.size() - 8); // remove v7 raw-shader/diagnostic zero counts
        legacy_bytes[8] = 6;
    }
    CHECK(deserialize_gpu_capture(legacy_bytes, legacy_loaded, error) &&
          !legacy_loaded.failure_diagnostics_available && legacy_loaded.failure_diagnostics.empty(),
          "v6 capture reopens with failed-operation diagnostics reported unavailable");
    if (legacy_bytes.size() >= 12) legacy_bytes[8] = 51;   // kVersion + 1: a future version
    CHECK(!deserialize_gpu_capture(legacy_bytes, legacy_loaded, error) &&
          error == "unsupported capture version 51",
          "future capture versions fail with a concrete version error");

    GpuCaptureFile bad_hash = mixed;
    bad_hash.blobs[0].content_hash ^= 1;
    CHECK(!write_gpu_capture(mixed_path.string(), bad_hash, error) &&
          error == "capture blob content hash mismatch",
          "writer rejects a resource version whose content identity is stale");

    set_gpu_capture_rtt_seed_reader([](uint64_t addr, GpuCaptureRttSeed& seed) {
        if (addr != 0x9000) return false;
        seed.guest_addr = addr; seed.width = 2; seed.height = 2;
        seed.rgba.assign(16, 0x5a);
        return true;
    });
    GpuCaptureRttSeed exported_seed;
    CHECK(read_gpu_capture_rtt_seed(0x9000, exported_seed, error) &&
          exported_seed.guest_addr == 0x9000 && exported_seed.rgba.size() == 16 &&
          exported_seed.rgba[0] == 0x5a,
          "registered RTT reader exports a validated temporal seed");
    set_gpu_capture_rtt_seed_snapshot_reader([](std::vector<GpuCaptureRttSeed>& seeds, std::string&) {
        GpuCaptureRttSeed high, low;
        high.guest_addr = 0xa000; high.width = 1; high.height = 1; high.rgba.assign(4, 0xa0);
        low.guest_addr = 0x8000; low.width = 1; low.height = 1; low.rgba.assign(4, 0x80);
        seeds.push_back(std::move(high)); seeds.push_back(std::move(low));
        return true;
    });
    std::vector<GpuCaptureRttSeed> exported_seeds;
    CHECK(read_all_gpu_capture_rtt_seeds(exported_seeds, error) && exported_seeds.size() == 2 &&
          exported_seeds[0].guest_addr == 0x8000 && exported_seeds[1].guest_addr == 0xa000,
          "registered RTT snapshot reader exports and sorts the complete live cache");
    set_gpu_capture_rtt_seed_snapshot_reader({});
    CHECK(!read_all_gpu_capture_rtt_seeds(exported_seeds, error) &&
          error == "live renderer has no RTT seed snapshot reader",
          "RTT snapshot export fails explicitly without a registered renderer reader");
    set_gpu_capture_rtt_seed_reader({});
    CHECK(!read_gpu_capture_rtt_seed(0x9000, exported_seed, error) &&
          error == "live renderer has no RTT seed reader",
          "RTT export fails explicitly without a registered renderer reader");

    set_gpu_capture_ds_seed_snapshot_reader(
        [&](std::vector<GpuCaptureDsSeed>& seeds, std::string&) {
            GpuCaptureDsSeed high = ds_seed, low = ds_seed;
            high.depth_read_base = high.depth_write_base = 0xa10000;
            low.depth_read_base = low.depth_write_base = 0x710000;
            seeds.push_back(std::move(high)); seeds.push_back(std::move(low));
            return true;
        });
    std::vector<GpuCaptureDsSeed> exported_ds;
    CHECK(read_all_gpu_capture_ds_seeds(exported_ds, error) && exported_ds.size() == 2 &&
          exported_ds[0].depth_read_base == 0x710000 &&
          exported_ds[1].depth_read_base == 0xa10000,
          "registered DS snapshot reader validates and sorts the complete live cache");
    GpuCaptureFile referenced_ds_capture;
    referenced_ds_capture.draws.resize(1);
    referenced_ds_capture.draws[0].color0_width = 2;
    referenced_ds_capture.draws[0].color0_height = 2;
    referenced_ds_capture.draws[0].ps.depth_test_enable = true;
    referenced_ds_capture.draws[0].ps.depth_read_base = 0xa10000;
    referenced_ds_capture.draws[0].ps.depth_write_base = 0xa10000;
    referenced_ds_capture.draws[0].ps.stencil_read_base = ds_seed.stencil_read_base;
    referenced_ds_capture.draws[0].ps.stencil_write_base = ds_seed.stencil_write_base;
    referenced_ds_capture.draws[0].ps.htile_data_base = ds_seed.htile_data_base;
    CHECK(gpu_capture_ds_seed_snapshot_available() &&
          capture_referenced_gpu_ds_seeds(referenced_ds_capture, error) &&
          referenced_ds_capture.ds_seeds.size() == 1 &&
          referenced_ds_capture.ds_seeds[0].depth_read_base == 0xa10000,
          "standalone capture retains only the exact referenced live DS checkpoint");
    DrawItem one_shot_ds_draw = draw;
    one_shot_ds_draw.vrt = std::make_shared<ShaderResourceTable>();
    one_shot_ds_draw.color0_width = one_shot_ds_draw.color0_height = 2;
    one_shot_ds_draw.draw_index = 0;
    one_shot_ds_draw.command_order = 0;
    one_shot_ds_draw.ps.depth_test_enable = true;
    one_shot_ds_draw.ps.depth_read_base = one_shot_ds_draw.ps.depth_write_base = 0xa10000;
    one_shot_ds_draw.ps.stencil_read_base = ds_seed.stencil_read_base;
    one_shot_ds_draw.ps.stencil_write_base = ds_seed.stencil_write_base;
    one_shot_ds_draw.ps.htile_data_base = ds_seed.htile_data_base;
    request_interactive_gpu_capture(prosper_test::test_scratch_file("prosper_interactive_ds_checkpoint.prgcap"));
    auto one_shot_ds_pending = begin_requested_gpu_capture(
        {one_shot_ds_draw}, {}, {{SubmitOperationKind::Draw, 0, 0}}, 2, 2);
    CHECK(one_shot_ds_pending && one_shot_ds_pending->capture.ds_seeds.size() == 1 &&
          one_shot_ds_pending->capture.ds_seeds[0].depth_read_base == 0xa10000,
          "interactive one-shot capture retains its referenced pre-submit DS checkpoint");
    GpuCaptureDsSeed restored_ds;
    set_gpu_replay_ds_seed_writer([&](const GpuCaptureDsSeed& seed, std::string&) {
        restored_ds = seed; return true;
    });
    CHECK(restore_gpu_replay_ds_seeds({ds_seed}, error) && restored_ds.depth == ds_seed.depth &&
          restored_ds.stencil == ds_seed.stencil,
          "registered DS writer receives exact validated checkpoint planes");
    set_gpu_capture_ds_seed_snapshot_reader({});
    CHECK(!read_all_gpu_capture_ds_seeds(exported_ds, error) &&
          error == "live renderer has no DS seed snapshot reader",
          "DS snapshot export fails explicitly without a registered renderer reader");
    set_gpu_replay_ds_seed_writer({});
    CHECK(!restore_gpu_replay_ds_seeds({ds_seed}, error) &&
          error == "live renderer has no DS seed writer",
          "DS restore fails explicitly without a registered renderer writer");

    auto truncated = path; truncated += ".truncated";
    std::filesystem::copy_file(path, truncated, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 5);
    GpuCaptureFile bad;
    CHECK(!read_gpu_capture(truncated.string(), bad, error) && !error.empty(), "truncated capture fails loudly");
    std::filesystem::remove(path); std::filesystem::remove(truncated); std::filesystem::remove(mixed_path);
    std::filesystem::remove(failed_path); std::filesystem::remove(no_oracle_path);

    // Interactive one-shot capture trigger (frame_grab hotkey). No env capture is configured here, so
    // begin_requested_gpu_capture is a no-op until a grab is armed — this both proves the default-OFF
    // path and exercises the arm/consume state machine that the app's hotkey drives.
    set_test_env("PROSPER_GPU_CAPTURE", nullptr);   // ensure the env path is off for this block
    CHECK(!interactive_gpu_capture_armed(), "interactive capture is not armed by default");
    // Not armed + no env => begin_requested is inert, even with drawing work present.
    { DrawItem d; CHECK(begin_requested_gpu_capture({d}, {}, {}, 64, 64) == nullptr,
                        "no capture when neither env nor interactive is set"); }
    request_interactive_gpu_capture(prosper_test::test_scratch_file("prosper_interactive_grab.prgcap"));
    CHECK(interactive_gpu_capture_armed(), "request_interactive_gpu_capture arms a one-shot grab");
    // A ZERO-draw invocation must NOT consume the arm (the grab should land on real frame content).
    (void)begin_requested_gpu_capture({}, {}, {}, 64, 64);
    CHECK(interactive_gpu_capture_armed(), "a zero-draw invocation does not consume the armed grab");
    // The next DRAWING invocation consumes it (disarms), regardless of whether the synthetic capture
    // then succeeds — the trigger decision is what the hotkey relies on.
    { DrawItem d; (void)begin_requested_gpu_capture({d}, {}, {}, 64, 64); }
    CHECK(!interactive_gpu_capture_armed(), "a drawing invocation consumes the armed grab (one-shot)");
    // Re-arming and re-disarming works (a second press).
    request_interactive_gpu_capture(prosper_test::test_scratch_file("prosper_interactive_grab2.prgcap"));
    CHECK(interactive_gpu_capture_armed(), "the grab can be re-armed for a subsequent press");
    { DrawItem d; (void)begin_requested_gpu_capture({d}, {}, {}, 64, 64); }
    CHECK(!interactive_gpu_capture_armed(), "the re-armed grab is consumed by the next drawing invocation");

    // --- PROSPER_CAPTURE_BLOB_MAX_MB parse/clamp rules (#2440) ------------------------------------
    // The per-resource ceiling is a runtime value because GTA V gameplay binds a 1,105,723,396-byte
    // buffer -- 1.0298x over the 1 GiB default -- and one resource over the line aborts the whole
    // grab. Asserted through the pure parse function rather than the accessor, which caches in a
    // `static` and so exposes exactly one value per process.
    {
        constexpr uint64_t kDefault = 1ull << 30;
        constexpr uint64_t kTotal   = 3ull << 30;
        std::string note;

        note.clear();
        CHECK(capture_blob_max_bytes_from_env_for_test(nullptr, &note) == kDefault && note.empty(),
              "unset leaves the 1 GiB default, silently");
        note.clear();
        CHECK(capture_blob_max_bytes_from_env_for_test("", &note) == kDefault && note.empty(),
              "empty leaves the default, silently");

        // The case the issue is about: 1057 MiB clears a 1,105,723,396-byte resource.
        note.clear();
        CHECK(capture_blob_max_bytes_from_env_for_test("1057", &note) == 1057ull * (1ull << 20) &&
                  note.empty(),
              "a plain value is taken as given, in MiB");
        CHECK(capture_blob_max_bytes_from_env_for_test("1057", nullptr) > 1105723396ull,
              "and 1057 MiB really does clear GTA V's oversized buffer");

        // Malformed input must keep the default and SAY so -- a typo costs a capture, never a
        // silently widened bound.
        for (const char* bad : {"0", "abc", "12x", "-4", "1.5"}) {
            note.clear();
            CHECK(capture_blob_max_bytes_from_env_for_test(bad, &note) == kDefault && !note.empty(),
                  "a malformed value keeps the default and reports why");
        }

        // Above the total-blob budget is unsatisfiable, so clamp rather than accept a ceiling that
        // could only fail later and further from the cause.
        note.clear();
        CHECK(capture_blob_max_bytes_from_env_for_test("999999", &note) == kTotal && !note.empty(),
              "a value above the total-blob budget clamps to it, and reports the clamp");
        note.clear();
        CHECK(capture_blob_max_bytes_from_env_for_test("3072", &note) == kTotal && note.empty(),
              "exactly the total-blob budget is accepted as given (boundary, not clamped)");
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n"); return 0;
}
