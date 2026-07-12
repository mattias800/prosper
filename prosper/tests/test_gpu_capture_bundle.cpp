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
    first.blobs.push_back(blob);
    GpuCaptureFile second = first;
    second.metadata.submit_index = 42;
    second.metadata.input_route = "shift-the-serialized-blob-boundary";

    GpuCaptureBundle bundle;
    std::string error;
    CHECK(append_gpu_capture_bundle(bundle, first, error), "first submit appends to bundle");
    CHECK(append_gpu_capture_bundle(bundle, second, error), "second submit appends in order");
    CHECK(bundle.submits.size() == 2 &&
          gpu_capture_bundle_unique_bytes(bundle) * 10 < bundle.logical_bytes * 7,
          "content-defined chunks resynchronize and deduplicate after shifted metadata");
    CHECK(write_gpu_capture_bundle(path.string(), bundle, error), "bundle writes atomically");

    GpuCaptureBundle loaded;
    CHECK(read_gpu_capture_bundle(path.string(), loaded, error), "checksummed bundle reads");
    GpuCaptureFile restored;
    CHECK(materialize_gpu_capture_bundle_submit(loaded, 1, restored, error) &&
          restored.metadata.submit_index == 42 && restored.blobs.size() == 1 &&
          restored.blobs[0].bytes == blob.bytes,
          "bundle reconstructs an exact validated capture");

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
    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n"); return 0;
}
