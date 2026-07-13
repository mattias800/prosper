#include "../src/gpu/gpu_capture_bundle.hpp"

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
          restored_second.blobs[0].guest_addr == 0x200000 && restored_second.blobs[0].bytes == blob.bytes,
          "bundle reconstructs an exact validated capture");
    CHECK(materialize_gpu_capture_bundle_manifest(loaded, 1, second_manifest, error) &&
          second_manifest.metadata.submit_index == 42 && second_manifest.blobs.size() == 1 &&
          second_manifest.blobs[0].bytes.empty() && second_manifest.draws.size() == restored_second.draws.size(),
          "manifest-only materialization exposes submit state without resource payload reconstruction");
    restored_first.blobs[0].bytes[0] ^= 0xff;
    CHECK(restored_second.blobs[0].bytes == blob.bytes,
          "materialized submits own independent mutable resource bytes");

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
