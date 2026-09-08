#include "gpu/capture/gpu_capture_bundle.hpp"
#include "fixtures/test_scratch.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <new>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

// Measure requested ordinary C++ allocations only during write_gpu_capture_bundle. Input
// collection, reader allocations, libc/aligned allocation and physical RSS are not bounded here.
namespace {
std::atomic<bool> measuring{false};
std::atomic<size_t> allocation_bytes{0}, largest_allocation{0};
void record_allocation(size_t bytes) {
    if (!measuring.load(std::memory_order_relaxed)) return;
    allocation_bytes.fetch_add(bytes, std::memory_order_relaxed);
    size_t previous = largest_allocation.load(std::memory_order_relaxed);
    while (previous < bytes && !largest_allocation.compare_exchange_weak(
        previous, bytes, std::memory_order_relaxed)) {}
}
}
void* operator new(size_t bytes) {
    void* p = std::malloc(bytes ? bytes : 1);
    if (!p) throw std::bad_alloc();
    record_allocation(bytes);
    return p;
}
void* operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }

using namespace prosper::gpu;
namespace {
int failures = 0;
#define CHECK(c, message) do { if (!(c)) { \
    std::fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); ++failures; } } while (0)

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void put_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    file.close();
    CHECK(file.good(), "fixture file closes successfully");
}
std::vector<uint8_t> hex_bytes(const char* text) {
    std::vector<uint8_t> bytes;
    for (; *text; text += 2) {
        const auto digit = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
        bytes.push_back(static_cast<uint8_t>(digit(text[0]) * 16 + digit(text[1])));
    }
    return bytes;
}
// Frozen wire bytes, independently encoded from the legacy little-endian format, including
// its historical nonstandard FNV basis and checksum trailer. Never derive expected bytes by
// invoking the writer under test. Tiny chunks intentionally test framing, not capture parsing.
const char* const wire_v1 =
    "505247424e444c0001000000040302010000040003000000000000000100000075e41c394f4e8da7"
    "030000000081ff01000000290000000000000003000000000000000100000000000000ddcbdec4af56e141";
const char* const wire_v2 =
    "505247424e444c0002000000040302010000040008000000000000000200000075e41c394f4e8da7"
    "030000000081ff9a2a8baa89d6af41050000000102030405010000009a2a8baa89d6af410500000000"
    "000000010000000100000001000000290000000000000008000000000000000300000000000000"
    "01000000000000000100000000000000050000000000000008f9c4afb1551a0e";

GpuCaptureBundle tiny(uint32_t version) {
    GpuCaptureBundle bundle;
    bundle.version = version;
    bundle.logical_bytes = version == 1 ? 3 : 8;
    bundle.chunks = {{0, 0x81, 0xff}};
    if (version == 2) bundle.chunks.push_back({1, 2, 3, 4, 5});
    for (const auto& chunk : bundle.chunks) bundle.chunk_hashes.push_back(gpu_capture_hash(chunk));
    GpuCaptureBundleSubmit submit;
    submit.submit_index = 41;
    submit.logical_bytes = bundle.logical_bytes;
    submit.manifest_bytes = 3;
    submit.chunk_indices = {0};
    if (version == 2) {
        bundle.resources.push_back({bundle.chunk_hashes[1], 5, {1}});
        submit.blob_resource_indices = {0};
        submit.blob_bytes_read = {5};
    }
    bundle.submits.push_back(std::move(submit));
    return bundle;
}

// Every test uses an otherwise empty private directory. No spelling of the implementation's
// temporary filename is assumed; all unexpected sibling files/directories are detected.
bool only_target(const std::filesystem::path& directory, const std::filesystem::path& target) {
    size_t entries = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path() != target) return false;
        ++entries;
    }
    return entries == 1;
}
void expect_failure(const std::filesystem::path& directory, const GpuCaptureBundle& bundle,
                    const std::vector<uint8_t>& original, const char* label) {
    const auto target = directory / "target.prgbundle";
    put_bytes(target, original);
    std::string error;
    CHECK(!write_gpu_capture_bundle(target.string(), bundle, error) && !error.empty(), label);
    CHECK(read_bytes(target) == original, "failed write preserves entire previous destination and tail");
    CHECK(only_target(directory, target), "failed write removes every temporary artifact");
}
}

int main() {
    const auto directory = prosper_test::test_scratch_dir() / "streamed-bundle";
    std::filesystem::create_directory(directory);
    const auto target = directory / "target.prgbundle";
    const auto target_string = target.string();
    const auto first = tiny(1), second = tiny(2);
    const auto expected_first = hex_bytes(wire_v1), expected_second = hex_bytes(wire_v2);
    std::string error;

    for (uint32_t version : {1u, 2u}) {
        const auto& expected = version == 1 ? expected_first : expected_second;
        const auto& bundle = version == 1 ? first : second;
        CHECK(write_gpu_capture_bundle(target_string, bundle, error), "frozen fixture writes");
        CHECK(read_bytes(target) == expected, "wire version matches every frozen byte including trailer");
        put_bytes(target, expected);
        GpuCaptureBundle loaded;
        CHECK(read_gpu_capture_bundle(target_string, loaded, error) &&
              loaded.version == version && loaded.chunks == bundle.chunks &&
              loaded.chunk_hashes == bundle.chunk_hashes && loaded.logical_bytes == bundle.logical_bytes &&
              loaded.submits.size() == 1 && loaded.submits[0].submit_index == 41 &&
              loaded.submits[0].blob_resource_indices == bundle.submits[0].blob_resource_indices,
              "reader accepts independent frozen legacy bytes and resource references");
    }

    // Prebuild both payloads and path/error storage. The growing output-vector implementation
    // exceeds both fixed ceilings even for the smaller input; streaming must not scale with bytes.
    std::vector<GpuCaptureBundle> payloads;
    for (size_t size : {size_t{1} << 20, size_t{16} << 20}) {
        GpuCaptureBundle bundle = tiny(1);
        bundle.chunk_bytes = 16u << 20;
        bundle.chunks[0].resize(size);
        for (size_t i = 0; i < size; ++i) bundle.chunks[0][i] = static_cast<uint8_t>((i * 29) ^ (i >> 9));
        bundle.chunk_hashes[0] = gpu_capture_hash(bundle.chunks[0]);
        bundle.logical_bytes = size;
        bundle.submits[0].logical_bytes = bundle.submits[0].manifest_bytes = size;
        payloads.push_back(std::move(bundle));
    }
    error.reserve(1024);
    allocation_bytes = 0; largest_allocation = 0; measuring = true;
    void* observed = ::operator new(12345);
    measuring = false;
    ::operator delete(observed);
    CHECK(allocation_bytes == 12345 && largest_allocation == 12345,
          "allocation observer detects a deliberate allocation");
    for (const auto& bundle : payloads) {
        allocation_bytes = 0; largest_allocation = 0; measuring = true;
        const bool written = write_gpu_capture_bundle(target_string, bundle, error);
        measuring = false;
        const auto total = allocation_bytes.load(), largest = largest_allocation.load();
        std::printf("payload=%zu write-allocation-total=%zu largest=%zu\n",
                    bundle.chunks[0].size(), total, largest);
        CHECK(written, "prebuilt payload writes within allocation measurement");
        CHECK(total < 512 * 1024 && largest < 128 * 1024,
              "write allocations stay bounded independently of prebuilt payload size");
        GpuCaptureBundle loaded;
        CHECK(read_gpu_capture_bundle(target_string, loaded, error) && loaded.chunks == bundle.chunks &&
              loaded.logical_bytes == bundle.logical_bytes, "large payload and tail survive checked readback");
    }

    auto invalid = second;
    invalid.submits[0].blob_resource_indices[0] = 1;
    expect_failure(directory, invalid, expected_second, "late invalid resource reference refuses installation");
    invalid = second; invalid.resources[0].content_hash ^= 1;
    expect_failure(directory, invalid, expected_second, "resource checksum mismatch refuses installation");
    invalid = second; invalid.chunks[1].back() ^= 1;
    expect_failure(directory, invalid, expected_second, "later chunk corruption refuses installation");
    invalid = second; invalid.submits[0].chunk_indices[0] = 2;
    expect_failure(directory, invalid, expected_second, "late invalid manifest chunk refuses installation");

#ifdef __linux__
    // A real OS short write/flush failure in a child, without a process memory limit or any
    // production injection hook. _exit prevents the child's scratch destructor deleting ours.
    for (bool buffered : {false, true}) {
        put_bytes(target, expected_second);
        const pid_t child = fork();
        if (child == 0) {
            std::signal(SIGXFSZ, SIG_IGN);
            // The 152-byte wire fixture fits the stream buffer but exceeds the 64-byte
            // file limit, exercising deferred flush failure on buffered implementations.
            const rlim_t maximum = buffered ? 64 : 4096;
            const rlimit limit{maximum, maximum};
            if (setrlimit(RLIMIT_FSIZE, &limit) != 0) _exit(20);
            std::string failure;
            const bool ok = write_gpu_capture_bundle(target_string,
                buffered ? second : payloads[0], failure);
            _exit(!ok && !failure.empty() ? 0 : 21);
        }
        int status = 0;
        CHECK(child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "OS file-size failure is reported without killing the writer child");
        CHECK(read_bytes(target) == expected_second, "real OS failure preserves previous complete file");
        CHECK(only_target(directory, target), "real OS failure cleans temporary file and directory");
    }
#elif defined(_WIN32)
    put_bytes(target, expected_second);
    HANDLE held = CreateFileW(target.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    CHECK(held != INVALID_HANDLE_VALUE, "hold destination without delete sharing");
    if (held != INVALID_HANDLE_VALUE) {
        CHECK(!write_gpu_capture_bundle(target_string, first, error) && !error.empty(),
              "OS sharing restriction refuses replacement");
        CHECK(read_bytes(target) == expected_second, "denied replacement preserves complete destination");
        CHECK(only_target(directory, target), "denied replacement cleans all temporary artifacts");
        CloseHandle(held);
    }
#endif
    CHECK(write_gpu_capture_bundle(target_string, first, error) && read_bytes(target) == expected_first,
          "retry replaces longer old destination without a stale tail");
    CHECK(only_target(directory, target), "successful replacement leaves only destination");

    // An obstruction exercises installation failure on every platform.
    std::filesystem::remove(target);
    std::filesystem::create_directory(target);
    put_bytes(target / "keep", expected_second);
    CHECK(!write_gpu_capture_bundle(target_string, first, error) && !error.empty(),
          "nonempty directory destination refuses installation");
    CHECK(read_bytes(target / "keep") == expected_second && only_target(directory, target),
          "installation failure preserves obstruction and cleans temporary artifacts");
    std::filesystem::remove_all(target);

    // Simultaneous writers must reserve separate temporaries and publish one complete file.
    std::atomic<unsigned> ready{0};
    std::atomic<bool> go{false};
    bool success[2] = {false, false};
    std::thread writers[2];
    for (unsigned i = 0; i < 2; ++i) writers[i] = std::thread([&, i] {
        ready.fetch_add(1);
        while (!go.load()) std::this_thread::yield();
        std::string local_error;
        success[i] = write_gpu_capture_bundle(target_string, i == 0 ? first : second, local_error);
    });
    while (ready.load() != 2) std::this_thread::yield();
    go = true;
    for (auto& writer : writers) writer.join();
    const auto concurrent = read_bytes(target);
    CHECK(success[0] && success[1], "both concurrent independent writes complete");
    CHECK(concurrent == expected_first || concurrent == expected_second,
          "concurrent installation leaves one entire wire file, never mixed bytes");
    CHECK(only_target(directory, target), "concurrent writers clean their private temporary paths");
    std::printf("gpu_capture_bundle_streaming: %d failures\n", failures);
    return failures ? 1 : 0;
}
