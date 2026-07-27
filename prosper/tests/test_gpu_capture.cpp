#include "../src/gpu/gpu_capture.hpp"
#include "../src/gpu/agc_shader_layout.hpp"
#include "../src/gpu/guest_texture_layout.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include "../src/gpu/tile.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

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

int main() {
    std::printf("== test_gpu_capture ==\n");
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
              video_capture.format_version == 36 && video_capture.blobs.size() == 1 &&
              video_capture.blobs[0].bytes.size() == video_memory.size() &&
              video_capture.draws[0].prt.resources[0].captured_size == video_memory.size() &&
              video_capture.draws[0].prt.resources[0].resource.linear_row_pitch_bytes == 2048,
          "v28 capture retains the complete padded rows and their resolved byte pitch");
    std::vector<uint8_t> video_capture_bytes;
    GpuCaptureFile video_loaded;
    GpuReplayFrame video_replay;
    CHECK(serialize_gpu_capture(video_capture, video_capture_bytes, error) &&
              deserialize_gpu_capture(video_capture_bytes, video_loaded, error) &&
              video_loaded.format_version == 36 &&
              video_loaded.draws[0].prt.resources[0].captured_size == video_memory.size() &&
              video_loaded.draws[0].prt.resources[0].resource.linear_row_pitch_bytes == 2048 &&
              materialize_gpu_replay(video_loaded, video_replay, error) &&
              video_replay.items[0].prt->resources[0].host_data_size == video_memory.size(),
          "v28 replay round-trips and binds the complete pitch-padded source span");
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
              upgraded_video.format_version == 36 &&
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
              array_layout_loaded.format_version == 36 &&
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

    auto path = std::filesystem::temp_directory_path() / "prosper_gpu_capture_test.prgcap";
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
    CHECK(loaded.draws[0].raw_draw_modifier == 0x1122334455667788ull &&
          loaded.draws[0].vertex_offset == -37,
          "v27 draw modifier and vertex offset round-trip explicitly");
    CHECK(loaded.draws[0].vrt.resources[0].resource.fetch_index_mode ==
              VertexFetchIndexMode::Instance,
          "v30 fetch-index provenance round-trips explicitly for gpu_replay");
    std::vector<uint8_t> v24_resolve_bytes;
    GpuCaptureFile v24_resolve_loaded;
    CHECK(serialize_gpu_capture(pre_v31_source, v24_resolve_bytes, error) &&
          v24_resolve_bytes.size() >= v36_tail(pre_v31_source) +
              v35_tail(pre_v31_source) + v34_tail(pre_v31_source) +
              v32_tail(pre_v31_source) +
              v31_tail(pre_v31_source) + v30_tail(pre_v31_source) +
              v29_tail(pre_v31_source) + v28_tail(pre_v31_source) +
              v27_tail(pre_v31_source) + v26_tail(pre_v31_source) +
              v25_tail(pre_v31_source),
          "v25 capture exposes a removable trailing resolve-state block");
    if (v24_resolve_bytes.size() >= v36_tail(pre_v31_source) +
            v35_tail(pre_v31_source) + v34_tail(pre_v31_source) +
            v32_tail(pre_v31_source) +
            v31_tail(pre_v31_source) + v30_tail(pre_v31_source) +
            v29_tail(pre_v31_source) + v28_tail(pre_v31_source) +
            v27_tail(pre_v31_source) + v26_tail(pre_v31_source) +
            v25_tail(pre_v31_source)) {
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
    const auto& rr = replay.items[0].vrt->resources;
    CHECK(rr[0].gpu_addr == 0x1000 && rr[1].gpu_addr == 0x1008, "replay retains logical guest addresses");
    CHECK(rr[0].host_data && rr[1].host_data == rr[0].host_data + 8 && rr[1].host_data[0] == memory[8],
          "replay resources point into shared owned backing at captured offsets");
    CHECK(replay.expected_output_valid && replay.expected_output_hash == captured.expected_output_hash,
          "expected render oracle round-trips");
    const auto no_oracle_path = std::filesystem::temp_directory_path() /
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
    std::vector<SubmitOperation> mixed_operations = {
        {SubmitOperationKind::Draw, 4, 100},
        {SubmitOperationKind::Dispatch, 9, 101},
        {SubmitOperationKind::Draw, 5, 102},
    };
    GpuCaptureFile mixed;
    CHECK(capture_submit_items({mixed_draw}, {mixed_compute}, mixed_operations, meta,
                               repeated_reader, mixed, error),
          "mixed graphics/compute submit captures without a renderer");
    CHECK(mixed.blobs.size() == 1 && mixed.shader_versions.size() == 2,
          "identical address-distinct resources and shared shaders deduplicate by content");
    CHECK(mixed.operations.size() == 3 && mixed.operations[0].realized &&
          mixed.operations[1].realized && !mixed.operations[2].realized,
          "mixed operation plan explicitly retains unrealized work");
    CHECK(mixed.failure_diagnostics_available && mixed.failure_diagnostics.size() == 1 &&
          mixed.failure_diagnostics[0].reason == RealizationFailureReason::Unknown,
          "pre-realized capture inputs label an unexplained omission explicitly");
    const auto mixed_path = std::filesystem::temp_directory_path() / "prosper_gpu_capture_mixed_test.prgcap";
    CHECK(write_gpu_capture(mixed_path.string(), mixed, error), "mixed versioned capture writes");
    GpuCaptureFile mixed_loaded;
    CHECK(read_gpu_capture(mixed_path.string(), mixed_loaded, error) &&
          mixed_loaded.computes.size() == 1 && mixed_loaded.operations.size() == 3 &&
          !mixed_loaded.expected_output_valid,
          "mixed items, operation order, and absent output oracle round-trip");
    GpuReplayFrame mixed_replay;
    CHECK(materialize_gpu_replay(mixed_loaded, mixed_replay, error) &&
          mixed_replay.computes.size() == 1 && mixed_replay.computes[0].dispatch_index == 9 &&
          mixed_replay.computes[0].resources->resources[0].host_data[0] == repeated[32],
          "offline materialization owns compute resources without guest pointers");
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

    // Live one-shot capture receives the already-realized backend list as well as the original
    // semantic state. A failed operation is necessarily absent from that backend list, but must not
    // lose the exact reason/program/pipeline that the semantic realization can still diagnose.
    request_interactive_gpu_capture("/tmp/prosper_semantic_failure_diagnostic.prgcap");
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

    const auto failed_path = std::filesystem::temp_directory_path() /
        "prosper_gpu_capture_failed_diagnostic_test.prgcap";
    failed_capture.failure_diagnostics[0].pipeline.cb_resolve = true;
    // NGG draws can have a separately allocated vertex prolog and vertex main.  A failed capture
    // must retain both same-kind programs so gpu_replay can dump either half of the linked stage.
    auto linked_vertex = failed_capture.failure_diagnostics[0].stages[0];
    linked_vertex.program_addr += 0x1000;
    failed_capture.failure_diagnostics[0].stages.push_back(linked_vertex);
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
    if (legacy_bytes.size() >= 12) legacy_bytes[8] = 37;   // kVersion + 1: a future version
    CHECK(!deserialize_gpu_capture(legacy_bytes, legacy_loaded, error) &&
          error == "unsupported capture version 37",
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
    request_interactive_gpu_capture("/tmp/prosper_interactive_ds_checkpoint.prgcap");
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
    request_interactive_gpu_capture("/tmp/prosper_interactive_grab.prgcap");
    CHECK(interactive_gpu_capture_armed(), "request_interactive_gpu_capture arms a one-shot grab");
    // A ZERO-draw invocation must NOT consume the arm (the grab should land on real frame content).
    (void)begin_requested_gpu_capture({}, {}, {}, 64, 64);
    CHECK(interactive_gpu_capture_armed(), "a zero-draw invocation does not consume the armed grab");
    // The next DRAWING invocation consumes it (disarms), regardless of whether the synthetic capture
    // then succeeds — the trigger decision is what the hotkey relies on.
    { DrawItem d; (void)begin_requested_gpu_capture({d}, {}, {}, 64, 64); }
    CHECK(!interactive_gpu_capture_armed(), "a drawing invocation consumes the armed grab (one-shot)");
    // Re-arming and re-disarming works (a second press).
    request_interactive_gpu_capture("/tmp/prosper_interactive_grab2.prgcap");
    CHECK(interactive_gpu_capture_armed(), "the grab can be re-armed for a subsequent press");
    { DrawItem d; (void)begin_requested_gpu_capture({d}, {}, {}, 64, 64); }
    CHECK(!interactive_gpu_capture_armed(), "the re-armed grab is consumed by the next drawing invocation");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n"); return 0;
}
