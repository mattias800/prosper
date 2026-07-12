#include "../src/gpu/gpu_capture.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_gpu_capture ==\n");
    std::vector<uint8_t> memory(32);
    for (size_t i = 0; i < memory.size(); ++i) memory[i] = static_cast<uint8_t>(0x40 + i);
    auto reader = [&](uint64_t addr, uint8_t* dst, size_t n) -> size_t {
        if (addr < 0x1000 || addr >= 0x1000 + memory.size()) return 0;
        size_t off = static_cast<size_t>(addr - 0x1000), take = std::min(n, memory.size() - off);
        std::memcpy(dst, memory.data() + off, take); return take;
    };

    auto table = std::make_shared<ShaderResourceTable>();
    ShaderResource a{}; a.cls = ResourceClass::VertexBuffer; a.binding = 9; a.gpu_addr = 0x1000;
    a.size = 16; a.stride = 4; a.format = DataFormat::Unorm8; a.num_components = 4; a.fetch_pc = 12;
    ShaderResource b{}; b.cls = ResourceClass::ConstantBuffer; b.binding = 2; b.gpu_addr = 0x1008;
    b.size = 16; b.format = DataFormat::Float32; b.num_components = 4; b.srt_offset = 0x20;
    table->resources = {a, b};

    DrawItem draw; draw.vs = {0x07230203, 1, 2}; draw.fs = {0x07230203, 3}; draw.vrt = table;
    draw.vertex_count = 6; draw.indices = {0, 1, 2, 2, 3, 0}; draw.color0_base = 0x2000;
    draw.color0_width = 1024; draw.color0_height = 32;
    draw.draw_index = 7; draw.command_order = 123;
    draw.ps.topology = 3; draw.ps.color0_format = 37; draw.ps.blend_enable = true;
    draw.ps.has_viewport = true; draw.ps.viewport_w = 1920; draw.ps.viewport_h = -1080;
    draw.ps.db_render_control = 2; draw.ps.stencil_clear_enable = true;
    draw.ps.has_stencil_clear = true; draw.ps.stencil_clear_value = 3;
    draw.ps.stencil_read_base = 0x12345000; draw.ps.raw_stencil_op[0][1] = 4;
    draw.ps.db_shader_control = 2; draw.ps.stencil_test_val_export_enable = true;

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
    captured.expected_output_valid = true;
    captured.expected_output_hash = 0x1122334455667788ull; captured.expected_output_bytes = 480ull * 270 * 4;

    auto path = std::filesystem::temp_directory_path() / "prosper_gpu_capture_test.prgcap";
    GpuCaptureFile duplicate_seed = captured; duplicate_seed.rtt_seeds.push_back(captured.rtt_seeds[0]);
    CHECK(!write_gpu_capture(path.string(), duplicate_seed, error) && error == "duplicate RTT seed address",
          "writer rejects duplicate temporal RTT seed addresses");
    CHECK(write_gpu_capture(path.string(), captured, error), "versioned capture writes atomically");
    GpuCaptureFile loaded;
    CHECK(read_gpu_capture(path.string(), loaded, error), "versioned capture reads back");
    CHECK(loaded.metadata.submit_index == 42 && loaded.metadata.title_id == "PPSA24651" &&
          loaded.metadata.renderer_env.size() == 2 && loaded.metadata.renderer_env[0].first == "PROSPER_RTT_PERTARGET" &&
          loaded.draws.size() == 1 && loaded.draws[0].indices.size() == 6, "metadata and draw data round-trip");
    CHECK(loaded.draws[0].ps.viewport_h == -1080 && loaded.draws[0].ps.blend_enable,
          "fixed-function pipeline state round-trips explicitly");
    CHECK(loaded.draws[0].color0_width == 1024 && loaded.draws[0].color0_height == 32,
          "per-target extent round-trips");
    CHECK(loaded.shader_versions.size() == 2 && loaded.draws[0].draw_index == 7 &&
          loaded.draws[0].command_order == 123 && loaded.operations[0].source_index == 7,
          "content versions and draw operation identity round-trip");
    CHECK(loaded.rtt_seeds.size() == 1 && loaded.rtt_seeds[0].width == 2 &&
          loaded.rtt_seeds[0].rgba == temporal_rgba, "temporal RTT seed round-trips");
    CHECK(loaded.draws[0].ps.db_render_control == 2 && loaded.draws[0].ps.stencil_clear_enable &&
          loaded.draws[0].ps.has_stencil_clear && loaded.draws[0].ps.stencil_clear_value == 3 &&
          loaded.draws[0].ps.stencil_read_base == 0x12345000 &&
          loaded.draws[0].ps.raw_stencil_op[0][1] == 4 && loaded.draws[0].ps.db_shader_control == 2 &&
          loaded.draws[0].ps.stencil_test_val_export_enable,
          "clear intent, DS identity, and raw stencil-op provenance round-trip");

    GpuReplayFrame replay;
    CHECK(materialize_gpu_replay(loaded, replay, error), "capture materializes owned replay draw items");
    const auto& rr = replay.items[0].vrt->resources;
    CHECK(rr[0].gpu_addr == 0x1000 && rr[1].gpu_addr == 0x1008, "replay retains logical guest addresses");
    CHECK(rr[0].host_data && rr[1].host_data == rr[0].host_data + 8 && rr[1].host_data[0] == memory[8],
          "replay resources point into shared owned backing at captured offsets");
    CHECK(replay.expected_output_valid && replay.expected_output_hash == captured.expected_output_hash,
          "expected render oracle round-trips");
    CHECK(replay.items[0].draw_index == 7 && replay.items[0].command_order == 123,
          "materialized draw retains its source operation identity");
    CHECK(replay.rtt_seeds.size() == 1 && replay.rtt_seeds[0].guest_addr == draw.color0_base,
          "materialized replay owns temporal RTT seed pixels");

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
    GpuCaptureFile bad_hash = mixed;
    bad_hash.blobs[0].content_hash ^= 1;
    CHECK(!write_gpu_capture(mixed_path.string(), bad_hash, error) &&
          error == "capture blob content hash mismatch",
          "writer rejects a resource version whose content identity is stale");

    auto truncated = path; truncated += ".truncated";
    std::filesystem::copy_file(path, truncated, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 5);
    GpuCaptureFile bad;
    CHECK(!read_gpu_capture(truncated.string(), bad, error) && !error.empty(), "truncated capture fails loudly");
    std::filesystem::remove(path); std::filesystem::remove(truncated); std::filesystem::remove(mixed_path);

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n"); return 0;
}
