#include "../src/gpu/gpu_capture_bundle.hpp"
#include "../src/gpu/tile.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_gpu_capture_bundle ==\n");
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
        ("prosper-gpu-bundle-" + std::to_string(nonce) + ".prgbundle");
    const auto corrupt_path = path.string() + ".corrupt";

    GpuCaptureFile first;
    first.metadata.width = 64; first.metadata.height = 64;
    first.metadata.submit_index = 41; first.metadata.revision = "bundle-test";
    GpuCaptureBlob blob;
    blob.guest_addr = 0x100000; blob.bytes_read = 256 * 1024;
    blob.bytes.resize(static_cast<size_t>(blob.bytes_read));
    uint32_t random = 0x12345678u;
    for (auto& byte : blob.bytes) {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        byte = static_cast<uint8_t>(random);
    }
    blob.content_hash = gpu_capture_hash(blob.bytes);
    first.blobs.push_back(blob);
    first.operations.push_back({SubmitOperationKind::Draw, 5, 900, false});
    GpuCaptureRawShaderVersion failed_shader;
    failed_shader.words = {0xbf860001u, 0xbf810000u};
    failed_shader.has_endpgm = true;
    failed_shader.content_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(failed_shader.words.data()), failed_shader.words.size() * 4);
    first.raw_shader_versions.push_back(failed_shader);
    GpuCapturedOperationFailure failed_operation;
    failed_operation.kind = SubmitOperationKind::Draw;
    failed_operation.source_index = 5;
    failed_operation.command_order = 900;
    failed_operation.reason = RealizationFailureReason::ShaderRecompile;
    GpuCapturedStageDiagnostic failed_stage;
    failed_stage.stage = ShaderProgramStage::Fragment;
    failed_stage.program_addr = 0x400000;
    failed_stage.raw_shader_index = 0;
    failed_stage.coverage.total = 1;
    failed_stage.coverage.unsupported = 1;
    failed_stage.coverage.first_bad_fmt = 2;
    failed_stage.coverage.first_bad_op = 6;
    failed_stage.coverage.first_bad_pc = 0;
    failed_operation.stages.push_back(failed_stage);
    first.failure_diagnostics.push_back(failed_operation);
    first.failure_diagnostics_available = true;
    GpuCaptureDsSeed ds_seed;
    ds_seed.depth_read_base = ds_seed.depth_write_base = 0x310000;
    ds_seed.htile_data_base = 0x300000; ds_seed.width = 2; ds_seed.height = 2;
    ds_seed.format = GpuCaptureDsFormat::D32Float;
    ds_seed.depth_valid = true; ds_seed.depth.assign(16, 0x5a);
    first.ds_seeds.push_back(ds_seed);
    GpuCaptureFile second = first;
    second.metadata.submit_index = 42;
    second.metadata.input_route = "shift-the-serialized-blob-boundary";
    second.blobs[0].guest_addr = 0x200000;

    GpuCaptureFile stale = second;
    stale.metadata.submit_index = 43;
    stale.blobs[0].bytes[17] ^= 0x80;

    GpuCaptureBundle bundle;
    std::string error;
    CHECK(append_gpu_capture_bundle(bundle, first, error), "first submit appends to bundle");
    CHECK(append_gpu_capture_bundle(bundle, second, error), "second submit appends in order");
    CHECK(bundle.version == 2 && bundle.resources.size() == 1 &&
          bundle.submits[0].blob_resource_indices == bundle.submits[1].blob_resource_indices,
          "same content at different addresses reuses one exact resource");
    CHECK(!append_gpu_capture_bundle(bundle, stale, error) &&
          error.find("content hash mismatch") != std::string::npos &&
          bundle.resources.size() == 1 && bundle.submits.size() == 2,
          "failed append rolls back and cannot reuse a stale same-address resource");
    stale.blobs[0].content_hash = gpu_capture_hash(stale.blobs[0].bytes);
    CHECK(append_gpu_capture_bundle(bundle, stale, error) && bundle.resources.size() == 2 &&
          bundle.submits[2].blob_resource_indices[0] != bundle.submits[1].blob_resource_indices[0],
          "changed content at the same address creates a distinct resource version");
    CHECK(bundle.submits.size() == 3 &&
          gpu_capture_bundle_unique_bytes(bundle) * 10 < bundle.logical_bytes * 7,
          "resource dictionary and manifest chunks reduce physical bytes");
    const GpuCaptureBundleStats stats = gpu_capture_bundle_stats(bundle);
    CHECK(stats.resource_reference_count == 3 && stats.exact_reuse_count == 1 &&
          stats.resource_logical_bytes == blob.bytes.size() * 3,
          "bundle statistics distinguish logical references from exact reuse");
    GpuCaptureBundle rolling = bundle;
    rolling.submits.erase(rolling.submits.begin(), rolling.submits.begin() + 2);
    rolling.logical_bytes = rolling.submits[0].logical_bytes;
    const uint64_t rolling_before = gpu_capture_bundle_unique_bytes(rolling);
    GpuCaptureFile rolling_restored;
    CHECK(compact_gpu_capture_bundle(rolling, error) && rolling.resources.size() == 1 &&
          gpu_capture_bundle_unique_bytes(rolling) < rolling_before &&
          materialize_gpu_capture_bundle_submit(rolling, 0, rolling_restored, error) &&
          rolling_restored.blobs[0].bytes == stale.blobs[0].bytes,
          "rolling-window compaction removes unreachable resources and chunks");
    CHECK(write_gpu_capture_bundle(path.string(), bundle, error), "bundle writes atomically");

    GpuCaptureBundle loaded;
    CHECK(read_gpu_capture_bundle(path.string(), loaded, error), "checksummed bundle reads");
    GpuCaptureFile restored_first, restored_second;
    GpuCaptureFile second_manifest;
    CHECK(materialize_gpu_capture_bundle_submit(loaded, 0, restored_first, error) &&
          materialize_gpu_capture_bundle_submit(loaded, 1, restored_second, error) &&
          restored_second.metadata.submit_index == 42 && restored_second.blobs.size() == 1 &&
          restored_second.blobs[0].guest_addr == 0x200000 && restored_second.blobs[0].bytes == blob.bytes &&
          restored_second.failure_diagnostics_available && restored_second.failure_diagnostics.size() == 1 &&
          restored_second.raw_shader_versions[0].words == failed_shader.words &&
          restored_second.ds_seeds.size() == 1 && restored_second.ds_seeds[0].depth == ds_seed.depth,
          "bundle reconstructs an exact validated capture");
    CHECK(materialize_gpu_capture_bundle_manifest(loaded, 1, second_manifest, error) &&
          second_manifest.metadata.submit_index == 42 && second_manifest.blobs.size() == 1 &&
          second_manifest.blobs[0].bytes.empty() && second_manifest.draws.size() == restored_second.draws.size() &&
          second_manifest.ds_seeds.size() == 1 && second_manifest.ds_seeds[0].depth == ds_seed.depth,
          "manifest-only materialization exposes submit state without resource payload reconstruction");
    restored_first.blobs[0].bytes[0] ^= 0xff;
    CHECK(restored_second.blobs[0].bytes == blob.bytes,
          "materialized submits own independent mutable resource bytes");

    GpuCaptureFile dcc_capture;
    dcc_capture.metadata.submit_index = 51;
    dcc_capture.metadata.revision = "bundle-dcc-test";
    ShaderResource dcc_resource{};
    dcc_resource.cls = ResourceClass::Texture; dcc_resource.binding = 7;
    dcc_resource.gpu_addr = 0x500000; dcc_resource.size = 32 * 32 * 32 * 4;
    dcc_resource.format = DataFormat::Sint32; dcc_resource.num_components = 1;
    dcc_resource.img_dim = 2; dcc_resource.width = 32; dcc_resource.height = 32;
    dcc_resource.depth = 32; dcc_resource.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbRX);
    dcc_resource.compression_enabled = true; dcc_resource.meta_pipe_aligned = true;
    dcc_resource.metadata_addr = 0x900000;
    GpuCaptureBlob dcc_base;
    dcc_base.guest_addr = dcc_resource.gpu_addr;
    dcc_base.bytes.resize(static_cast<size_t>(gpu_capture_resource_footprint(dcc_resource)), 0x41);
    dcc_base.bytes_read = dcc_base.bytes.size();
    dcc_base.content_hash = gpu_capture_hash(dcc_base.bytes);
    GpuCaptureBlob dcc_metadata;
    dcc_metadata.guest_addr = dcc_resource.metadata_addr;
    dcc_metadata.bytes.resize(static_cast<size_t>(gpu_capture_dcc_metadata_footprint(dcc_resource)), 0x82);
    dcc_metadata.bytes_read = dcc_metadata.bytes.size();
    dcc_metadata.content_hash = gpu_capture_hash(dcc_metadata.bytes);
    dcc_capture.blobs = {dcc_base, dcc_metadata};
    GpuCapturedDraw dcc_draw;
    dcc_draw.vs = {0x07230203, 51}; dcc_draw.fs = {0x07230203, 52};
    dcc_draw.vrt.present = true;
    GpuCapturedResource captured_dcc;
    captured_dcc.resource = dcc_resource; captured_dcc.blob_index = 0;
    captured_dcc.metadata_size = dcc_metadata.bytes.size(); captured_dcc.metadata_blob_index = 1;
    dcc_draw.vrt.resources.push_back(captured_dcc);
    dcc_capture.draws.push_back(dcc_draw);
    GpuCaptureBundle dcc_bundle;
    GpuCaptureFile dcc_manifest, dcc_restored;
    GpuReplayFrame dcc_replay;
    CHECK(append_gpu_capture_bundle(dcc_bundle, dcc_capture, error) &&
          materialize_gpu_capture_bundle_manifest(dcc_bundle, 0, dcc_manifest, error) &&
          dcc_manifest.blobs.size() == 2 && dcc_manifest.blobs[1].bytes.empty() &&
          dcc_manifest.draws[0].vrt.resources[0].metadata_blob_index == 1 &&
          materialize_gpu_capture_bundle_submit(dcc_bundle, 0, dcc_restored, error) &&
          materialize_gpu_replay(dcc_restored, dcc_replay, error) &&
          dcc_replay.items[0].vrt->resources[0].dcc_metadata_host_data_size == dcc_metadata.bytes.size(),
          "bundle manifest preserves separate DCC references while payloads stay deduplicated");

    GpuCaptureBundle malformed = loaded;
    malformed.submits[0].blob_resource_indices[0] =
        static_cast<uint32_t>(malformed.resources.size());
    GpuCaptureFile rejected_capture;
    CHECK(!materialize_gpu_capture_bundle_submit(malformed, 0, rejected_capture, error) &&
          error.find("invalid resource") != std::string::npos,
          "materialization rejects an out-of-range resource reference");

    const auto v1_path = path.string() + ".v1";
    std::vector<uint8_t> first_bytes;
    CHECK(serialize_gpu_capture(first, first_bytes, error), "created a legacy capture payload");
    GpuCaptureBundle legacy;
    legacy.version = 1; legacy.chunk_bytes = 16u << 20;
    legacy.logical_bytes = first_bytes.size();
    legacy.chunks.push_back(first_bytes); legacy.chunk_hashes.push_back(gpu_capture_hash(first_bytes));
    GpuCaptureBundleSubmit legacy_submit;
    legacy_submit.submit_index = first.metadata.submit_index;
    legacy_submit.logical_bytes = legacy_submit.manifest_bytes = first_bytes.size();
    legacy_submit.chunk_indices.push_back(0); legacy.submits.push_back(legacy_submit);
    CHECK(write_gpu_capture_bundle(v1_path, legacy, error), "writer preserves the legacy v1 layout");
    GpuCaptureBundle loaded_legacy;
    GpuCaptureFile restored_legacy;
    CHECK(read_gpu_capture_bundle(v1_path, loaded_legacy, error) && loaded_legacy.version == 1 &&
          materialize_gpu_capture_bundle_submit(loaded_legacy, 0, restored_legacy, error) &&
          restored_legacy.blobs[0].bytes == blob.bytes,
          "reader and materializer remain compatible with v1 bundles");

    std::ifstream input(path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    if (!bytes.empty()) bytes.back() ^= 0x80;
    std::ofstream corrupt(corrupt_path, std::ios::binary);
    corrupt.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    corrupt.close();
    GpuCaptureBundle rejected;
    CHECK(!read_gpu_capture_bundle(corrupt_path, rejected, error) &&
          error.find("checksum") != std::string::npos,
          "bundle rejects manifest corruption");

    std::error_code ec; std::filesystem::remove(path, ec); std::filesystem::remove(corrupt_path, ec);
    std::filesystem::remove(v1_path, ec);
    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n"); return 0;
}
