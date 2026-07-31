#include "../tools/gpu_replay/realized_shader_dump.hpp"
#include "../tools/gpu_replay/compute_recompile.hpp"
#include "../src/gpu/diagnostic_selectors.hpp"

#include <cstdio>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_gpu_replay_shader_dump ==\n");

    tools::RealizedShaderSelector selector;
    CHECK(tools::parse_realized_shader_selector("0:vs", selector) &&
              selector.draw_index == 0 && selector.vertex,
          "realized VS selector parses exactly");
    CHECK(tools::parse_realized_shader_selector("0x1:fs", selector) &&
              selector.draw_index == 1 && !selector.vertex,
          "realized FS selector accepts an explicit numeric base");
    CHECK(tools::parse_realized_shader_selector("7:vs-main", selector) &&
              selector.draw_index == 7 && selector.vertex && selector.vertex_main,
          "linked vertex-main selector parses exactly");
    CHECK(!tools::parse_realized_shader_selector("-1:vs", selector) &&
              !tools::parse_realized_shader_selector("1junk:fs", selector) &&
              !tools::parse_realized_shader_selector("1:ps", selector) &&
              !tools::parse_realized_shader_selector("1:vs:extra", selector),
          "malformed realized-shader selectors are rejected without partial parsing");

    uint64_t diagnostic_draw = 0;
    CHECK(gpu::parse_diagnostic_draw_id("0x481", diagnostic_draw) && diagnostic_draw == 1153,
          "geometry probe accepts the same explicit-base semantic ID in every layer");
    CHECK(!gpu::parse_diagnostic_draw_id("1153junk", diagnostic_draw) &&
              !gpu::parse_diagnostic_draw_id("-1", diagnostic_draw) &&
              !gpu::parse_diagnostic_draw_id("18446744073709551616", diagnostic_draw),
          "geometry probe rejects partial, signed, and overflowing draw IDs");

    CHECK(tools::select_geometry_probe_stage(true, true, false) ==
              tools::GeometryProbeStage::GeneratedGeometry,
          "rebuilt generated geometry owns probe capture without a raw VS");
    CHECK(tools::select_geometry_probe_stage(false, false, true) ==
              tools::GeometryProbeStage::Vertex,
          "VS-only geometry probe instruments a captured raw vertex stream");
    CHECK(tools::select_geometry_probe_stage(false, false, false) ==
              tools::GeometryProbeStage::Unsupported &&
              tools::select_geometry_probe_stage(false, true, true) ==
                  tools::GeometryProbeStage::Unsupported,
          "geometry probe rejects a missing raw VS and an unrebuildable existing GS");

    uint64_t draw_first = 0, draw_last = 0;
    CHECK(gpu::parse_diagnostic_draw_range("0x7:013", draw_first, draw_last) &&
              draw_first == 7 && draw_last == 11 &&
              gpu::parse_diagnostic_draw_range("1153", draw_first, draw_last) &&
              draw_first == 1153 && draw_last == 1153,
          "draw selection parses complete semantic IDs and explicit numeric bases");
    CHECK(!gpu::parse_diagnostic_draw_range("7junk", draw_first, draw_last) &&
              !gpu::parse_diagnostic_draw_range("7:", draw_first, draw_last) &&
              !gpu::parse_diagnostic_draw_range(":11", draw_first, draw_last) &&
              !gpu::parse_diagnostic_draw_range("7:11:12", draw_first, draw_last),
          "draw selection rejects partial and incomplete ranges");

    uint32_t tap_pc = 0;
    CHECK(gpu::parse_fragment_tap_selector("0x481:0x18f", diagnostic_draw, tap_pc) &&
              diagnostic_draw == 1153 && tap_pc == 399,
          "fragment tap parses a complete semantic draw and 32-bit PC");
    CHECK(!gpu::parse_fragment_tap_selector("1153", diagnostic_draw, tap_pc) &&
              !gpu::parse_fragment_tap_selector("1153:399junk", diagnostic_draw, tap_pc) &&
              !gpu::parse_fragment_tap_selector("1153:0x100000000", diagnostic_draw, tap_pc) &&
              !gpu::parse_fragment_tap_selector("1153:399:1", diagnostic_draw, tap_pc),
          "fragment tap rejects missing, partial, overflowing, and extra components");

    gpu::GpuReplayFrame replay;
    replay.raw_shader_versions = {
        {11, true, {0x11111111u, 0xbf810000u}},
        {22, true, {0x22222222u, 0xbf810000u}},
        {33, true, {0x33333333u, 0xbf810000u}},
    };
    gpu::DrawItem draw;
    draw.draw_index = 7;
    draw.vs_raw_shader_index = 1;
    draw.fs_raw_shader_index = 0;
    draw.vs_chain_raw_shader_index = 2;
    replay.items.push_back(draw);

    gpu::DrawItem later = draw;
    later.draw_index = 11;
    later.vs_raw_shader_index = 0;
    later.fs_raw_shader_index = 1;
    replay.items.push_back(later);
    replay.operations = {
        {gpu::SubmitOperationKind::Draw, 7, 100, true},
        {gpu::SubmitOperationKind::Dispatch, 3, 101, true},
        {gpu::SubmitOperationKind::Draw, 11, 102, true},
    };

    std::string error;
    const auto* vs = tools::select_realized_raw_shader(replay, "7:vs", error);
    const auto* fs = tools::select_realized_raw_shader(replay, "7:fs", error);
    const auto* main = tools::select_realized_raw_shader(replay, "7:vs-main", error);
    CHECK(vs == &replay.raw_shader_versions[1] && fs == &replay.raw_shader_versions[0] &&
              main == &replay.raw_shader_versions[2],
          "selector resolves a semantic draw ID instead of a compact item offset");
    CHECK(tools::replay_item_index_for_draw(replay, 11) == 1 &&
              tools::replay_operation_index_for_draw(replay, 11) == 2 &&
              tools::replay_item_index_for_draw(replay, 1) == SIZE_MAX,
          "draw IDs map across compact-item holes and mixed operation indices");
    CHECK(!tools::select_realized_raw_shader(replay, "2:vs", error) &&
              error.find("realized draw 2 not found") != std::string::npos,
          "missing semantic draw reports a selector error");
    replay.items[0].fs_raw_shader_index = 0xFFFFFFFFu;
    CHECK(!tools::select_realized_raw_shader(replay, "7:fs", error) &&
              error.find("capture predates v19 or source was unreadable") != std::string::npos,
          "missing legacy raw source is explicit instead of selecting unrelated bytes");
    replay.items[0].vs_chain_raw_shader_index = 0xFFFFFFFFu;
    CHECK(!tools::select_realized_raw_shader(replay, "7:vs-main", error) &&
              error.find("capture predates v31 or draw is not linked") != std::string::npos,
          "missing linked main source reports its versioned capture requirement");

    gpu::GpuReplayFrame shared_replay;
    gpu::DrawItem shared_draw;
    shared_draw.draw_index = 19;
    shared_draw.vs = {0xaaaaaaaa};
    shared_draw.fs = {0xbbbbbbbb};
    shared_draw.vs_shared =
        std::make_shared<const std::vector<uint32_t>>(std::vector<uint32_t>{0x07230203, 0x11});
    shared_draw.fs_shared =
        std::make_shared<const std::vector<uint32_t>>(std::vector<uint32_t>{0x07230203, 0x22, 0x33});
    shared_replay.items.push_back(shared_draw);
    bool shared = false;
    const auto* stored_vs = tools::select_recompiled_shader(shared_replay, "19:vs", shared, error);
    CHECK(stored_vs == shared_draw.vs_shared.get() && shared,
          "recompiled VS selection reads the shared words that rendering consumes");
    const auto* stored_fs = tools::select_recompiled_shader(shared_replay, "19:fs", shared, error);
    CHECK(stored_fs == shared_draw.fs_shared.get() && shared,
          "recompiled FS selection reads the shared words that rendering consumes");
    shared_replay.items[0].vs_shared.reset();
    const auto* owned_vs = tools::select_recompiled_shader(shared_replay, "19:vs", shared, error);
    CHECK(owned_vs == &shared_replay.items[0].vs && !shared,
          "recompiled shader selection retains the ordinary owned-vector path");
    CHECK(!tools::select_recompiled_shader(shared_replay, "20:vs", shared, error) &&
              error.find("realized draw 20 not found") != std::string::npos,
          "recompiled shader selection rejects an absent semantic draw");

    gpu::ComputeItem raw_compute;
    raw_compute.spirv = {0x07230203u, 0u};
    raw_compute.launch.local_x = 64;
    raw_compute.recompile_config_available = true;
    raw_compute.recompile_config.local_x = 64;
    raw_compute.recompile_config.user_sgprs = {0x12345678u};
    raw_compute.raw_shader_index = 0;
    std::vector<gpu::GpuCaptureRawShaderVersion> compute_raw = {
        {0, true, {0xbf810000u}}, // s_endpgm
    };
    CHECK(tools::recompile_captured_compute(
              raw_compute, compute_raw, nullptr, false, false, true) &&
              raw_compute.spirv.size() > 5 &&
              raw_compute.spirv[0] == 0x07230203u &&
              raw_compute.user_sgprs == raw_compute.recompile_config.user_sgprs,
          "capture v39 raw compute replay substitutes current SPIR-V and push constants");
    raw_compute.raw_shader_index = 1;
    CHECK(!tools::recompile_captured_compute(
              raw_compute, compute_raw, nullptr, false, false, true),
          "raw compute replay keeps the stored module when source is unavailable");

    gpu::GpuCapturedStageDiagnostic failed_compute;
    failed_compute.stage = gpu::ShaderProgramStage::Compute;
    failed_compute.raw_shader_index = 0;
    failed_compute.resource_table_present = false;
    failed_compute.resource_count = 0;
    failed_compute.recompile_config_available = true;
    failed_compute.recompile_config.user_sgprs = {0x12345678u, 0x9abcdef0u};
    failed_compute.recompile_config.local_x = 8;
    failed_compute.recompile_config.local_y = 4;
    failed_compute.recompile_config.local_z = 1;
    failed_compute.recompile_config.exact_thread_extent = true;
    failed_compute.recompile_config.threads_x = 17;
    failed_compute.recompile_config.threads_y = 9;
    failed_compute.recompile_config.threads_z = 1;
    failed_compute.recompile_config.wave_size = 64;
    failed_compute.recompile_config.tidig_comp_cnt = 2;
    failed_compute.recompile_config.tgid_x_en = true;
    failed_compute.recompile_config.tgid_y_en = true;
    failed_compute.recompile_config.tg_size_en = true;
    failed_compute.recompile_config.lds_bytes = 512;
    failed_compute.recompile_config.native_subgroup_size = 0;
    std::vector<uint32_t> failed_compute_spirv;
    const auto direct_failed_compute_spirv = gpu::recompile_compute_shader_cached(
        compute_raw[0].words.data(), compute_raw[0].words.size(), nullptr,
        failed_compute.recompile_config);
    CHECK(tools::recompile_failed_compute_stage(
              failed_compute, compute_raw, failed_compute_spirv, error) &&
              !failed_compute_spirv.empty() &&
              failed_compute_spirv == direct_failed_compute_spirv,
          "failed compute retry reproduces the exact captured specialization byte-for-byte");
    const gpu::DescriptorValidationReport retry_contract =
        tools::validate_recompiled_failed_stage_contract(
            failed_compute.stage, failed_compute_spirv, nullptr);
    const gpu::DescriptorValidationReport live_contract =
        gpu::validate_spirv_descriptor_interface(
            failed_compute_spirv, nullptr, 0, gpu::SpirvShaderStage::Compute, false);
    CHECK(retry_contract.ok() == live_contract.ok() &&
          retry_contract.descriptors.size() == live_contract.descriptors.size() &&
          retry_contract.issues.size() == live_contract.issues.size(),
          "failed-stage retry reports the same accepted descriptor contract as live compute");
    const std::vector<uint32_t> rejected_retry_spirv;
    const gpu::DescriptorValidationReport rejected_retry_contract =
        tools::validate_recompiled_failed_stage_contract(
            failed_compute.stage, rejected_retry_spirv, nullptr);
    const gpu::DescriptorValidationReport rejected_live_contract =
        gpu::validate_spirv_descriptor_interface(
            rejected_retry_spirv, nullptr, 0, gpu::SpirvShaderStage::Compute, false);
    CHECK(!rejected_retry_contract.ok() &&
          rejected_retry_contract.ok() == rejected_live_contract.ok() &&
          rejected_retry_contract.issues.size() == rejected_live_contract.issues.size() &&
          !rejected_retry_contract.issues.empty() &&
          rejected_retry_contract.issues.front().code ==
              gpu::DescriptorIssueCode::MalformedSpirv,
          "failed-stage retry preserves live contract rejection instead of becoming a fallback");
    auto different_failed_compute = failed_compute;
    different_failed_compute.recompile_config.local_x = 4;
    std::vector<uint32_t> different_failed_compute_spirv;
    CHECK(tools::recompile_failed_compute_stage(
              different_failed_compute, compute_raw, different_failed_compute_spirv, error) &&
              different_failed_compute_spirv != failed_compute_spirv,
          "failed compute retry consumes the captured local-size specialization");
    failed_compute.recompile_config_available = false;
    CHECK(!tools::recompile_failed_compute_stage(
              failed_compute, compute_raw, failed_compute_spirv, error) &&
              error.find("predates v42") != std::string::npos,
          "failed compute retry refuses older captures instead of inventing ABI state");

    tools::ReplayRenderIntent render_intent;
    render_intent.positional_count = 1;
    CHECK(tools::replay_will_render(render_intent),
          "one-capsule replay without an output path still initializes its renderer");
    render_intent.inspect_only = true;
    CHECK(!tools::replay_will_render(render_intent),
          "inspect-only raw recompilation stays Vulkan-independent");
    render_intent.inspect_only = false;
    render_intent.graph_only = true;
    CHECK(!tools::replay_will_render(render_intent),
          "dependency-graph raw recompilation stays Vulkan-independent");
    render_intent.graph_only = false;
    render_intent.realized_shader_dump = true;
    CHECK(!tools::replay_will_render(render_intent),
          "terminal one-capsule raw-shader dump stays Vulkan-independent");
    render_intent.inspect = true;
    CHECK(tools::replay_will_render(render_intent),
          "inspecting after a raw-shader dump preserves the command's renderer execution");

    // Rendering must replace, not intersect with or trust, the capture host's optional format and
    // subgroup policy. Use opaque non-null handles: the selection helper only inspects the published
    // feature contract and does not call Vulkan.
    static const uint32_t storage_compute_code[] = {
        0x7e080300u, 0xf0000f00u, 0x00000004u, 0xbf8c3f70u,
        0xf0200f00u, 0x00020004u, 0xbf810000u,
    };
    auto storage_resources = std::make_shared<gpu::ShaderResourceTable>();
    for (uint32_t i = 0; i < 2; ++i) {
        gpu::ShaderResource image;
        image.cls = gpu::ResourceClass::StorageImage;
        image.format = gpu::DataFormat::Float16;
        image.num_components = 4;
        image.binding = 4 + i;
        image.sgpr_base = i * 8;
        storage_resources->resources.push_back(image);
    }
    gpu::ComputeItem device_compute;
    device_compute.spirv = {0x07230203u, 0u};
    device_compute.resources = storage_resources;
    device_compute.raw_shader_index = 0;
    device_compute.recompile_config_available = true;
    device_compute.recompile_config.user_sgprs.resize(16);
    device_compute.recompile_config.local_x = 8;
    device_compute.recompile_config.local_y = 8;
    device_compute.recompile_config.wave_size = 64;
    device_compute.recompile_config.native_storage_format_support =
        gpu::native_storage_format_support_bit(gpu::DataFormat::Float32, 4);
    std::vector<gpu::GpuCaptureRawShaderVersion> storage_raw = {
        {0, true, std::vector<uint32_t>(std::begin(storage_compute_code),
                                       std::end(storage_compute_code))},
    };
    gpu::SharedVulkanContext capable;
    capable.instance = capable.physical = capable.device = capable.queue =
        reinterpret_cast<void*>(1);
    capable.queue_family = 0;
    capable.compute_queue_supported = true;
    capable.storage_image_read_without_format = true;
    capable.storage_image_write_without_format = true;
    capable.native_storage_format_support =
        gpu::native_storage_format_support_bit(gpu::DataFormat::Float16, 4);
    capable.compute_subgroup_size_control = true;
    capable.compute_full_subgroups = true;
    capable.compute_subgroup_vote = true;
    capable.compute_subgroup_arithmetic = true;
    capable.min_compute_subgroup_size = 32;
    capable.max_compute_subgroup_size = 64;
    capable.max_compute_workgroup_subgroups = 4;
    capable.max_compute_workgroup_size_x = 256;
    capable.max_compute_workgroup_invocations = 256;
    auto device_native_compute = device_compute;
    CHECK(tools::recompile_captured_compute(
              device_native_compute, storage_raw, &capable, false, false, true) &&
              device_native_compute.required_subgroup_size == 64,
          "rendering raw replay selects the replay device's native format and subgroup policy");
    auto non_adoptable = capable;
    non_adoptable.storage_image_write_without_format = false;
    auto device_portable_compute = device_compute;
    CHECK(tools::recompile_captured_compute(
              device_portable_compute, storage_raw, &non_adoptable, false, false, true) &&
              device_portable_compute.required_subgroup_size == 0 &&
              device_portable_compute.spirv != device_native_compute.spirv,
          "non-adoptable replay device replaces capture-host policy with portable compute");

    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
