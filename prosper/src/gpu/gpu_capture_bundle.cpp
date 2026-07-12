#include "gpu_capture_bundle.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_map>

namespace prosper::gpu {
namespace {

constexpr uint8_t kMagic[8] = {'P', 'R', 'G', 'B', 'N', 'D', 'L', '\0'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kEndian = 0x01020304u;
constexpr uint32_t kMaxChunks = 1u << 20;
constexpr uint32_t kMaxSubmits = 64;
constexpr uint64_t kMaxFileBytes = 4ull << 30;

struct Writer {
    std::vector<uint8_t> data;
    void raw(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p); data.insert(data.end(), b, b + n);
    }
    void u32(uint32_t v) {
        uint8_t b[4]; for (int i = 0; i < 4; ++i) b[i] = static_cast<uint8_t>(v >> (i * 8)); raw(b, 4);
    }
    void u64(uint64_t v) {
        uint8_t b[8]; for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>(v >> (i * 8)); raw(b, 8);
    }
    void bytes(const std::vector<uint8_t>& v) { u32(static_cast<uint32_t>(v.size())); raw(v.data(), v.size()); }
};

struct Reader {
    const uint8_t* p = nullptr;
    size_t left = 0;
    bool take(void* dst, size_t n) {
        if (n > left) return false; std::memcpy(dst, p, n); p += n; left -= n; return true;
    }
    bool u32(uint32_t& v) {
        uint8_t b[4]; if (!take(b, 4)) return false; v = 0;
        for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(b[i]) << (i * 8); return true;
    }
    bool u64(uint64_t& v) {
        uint8_t b[8]; if (!take(b, 8)) return false; v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(b[i]) << (i * 8); return true;
    }
    bool bytes(std::vector<uint8_t>& v) {
        uint32_t n = 0; if (!u32(n) || n > left) return false; v.resize(n); return take(v.data(), n);
    }
};

bool install_file(const std::string& path, const std::vector<uint8_t>& bytes, std::string& error) {
    std::filesystem::path target(path), temp = target; temp += ".tmp";
    std::error_code ec;
    if (target.has_parent_path()) std::filesystem::create_directories(target.parent_path(), ec);
    std::ofstream file(temp, std::ios::binary | std::ios::trunc);
    if (!file || !file.write(reinterpret_cast<const char*>(bytes.data()),
                             static_cast<std::streamsize>(bytes.size()))) {
        error = "cannot write bundle temporary file"; return false;
    }
    file.close(); std::filesystem::rename(temp, target, ec);
    if (ec) { std::filesystem::remove(target, ec); ec.clear(); std::filesystem::rename(temp, target, ec); }
    if (ec) { error = "cannot install bundle: " + ec.message(); return false; }
    return true;
}

uint64_t gear_value(uint8_t byte) {
    uint64_t value = static_cast<uint64_t>(byte) + 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

size_t content_chunk_size(const std::vector<uint8_t>& bytes, size_t offset, size_t maximum) {
    constexpr size_t minimum = 16 * 1024;
    constexpr uint64_t average_mask = (64 * 1024) - 1;
    const size_t available = std::min(maximum, bytes.size() - offset);
    if (available <= minimum) return available;
    uint64_t rolling = 0;
    for (size_t size = 1; size <= available; ++size) {
        rolling = (rolling << 1) + gear_value(bytes[offset + size - 1]);
        if (size >= minimum && (!(rolling & average_mask) || size == available)) return size;
    }
    return available;
}

} // namespace

uint64_t gpu_capture_bundle_unique_bytes(const GpuCaptureBundle& bundle) {
    uint64_t total = 0;
    for (const auto& chunk : bundle.chunks) total += chunk.size();
    return total;
}

bool append_gpu_capture_bundle(GpuCaptureBundle& bundle, const GpuCaptureFile& capture,
                               std::string& error) {
    error.clear();
    if (!bundle.chunk_bytes || bundle.chunk_bytes > (16u << 20) || bundle.submits.size() >= kMaxSubmits) {
        error = "invalid bundle bounds"; return false;
    }
    if (!bundle.submits.empty() && capture.metadata.submit_index <= bundle.submits.back().submit_index) {
        error = "bundle submits must be appended in increasing order"; return false;
    }
    std::vector<uint8_t> bytes;
    if (!serialize_gpu_capture(capture, bytes, error)) return false;
    GpuCaptureBundleSubmit submit;
    submit.submit_index = capture.metadata.submit_index;
    submit.logical_bytes = bytes.size();
    std::unordered_map<uint64_t, std::vector<uint32_t>> by_hash;
    by_hash.reserve(bundle.chunks.size());
    for (uint32_t i = 0; i < bundle.chunks.size(); ++i) by_hash[bundle.chunk_hashes[i]].push_back(i);
    for (size_t offset = 0; offset < bytes.size();) {
        const size_t size = content_chunk_size(bytes, offset, bundle.chunk_bytes);
        std::vector<uint8_t> chunk(bytes.begin() + offset, bytes.begin() + offset + size);
        const uint64_t hash = gpu_capture_hash(chunk);
        uint32_t index = UINT32_MAX;
        auto found = by_hash.find(hash);
        if (found != by_hash.end())
            for (uint32_t candidate : found->second)
                if (bundle.chunks[candidate] == chunk) { index = candidate; break; }
        if (index == UINT32_MAX) {
            if (bundle.chunks.size() >= kMaxChunks) { error = "bundle chunk count exceeds limit"; return false; }
            index = static_cast<uint32_t>(bundle.chunks.size());
            bundle.chunk_hashes.push_back(hash); bundle.chunks.push_back(std::move(chunk));
            by_hash[hash].push_back(index);
        }
        submit.chunk_indices.push_back(index);
        offset += size;
    }
    bundle.logical_bytes += bytes.size();
    bundle.submits.push_back(std::move(submit));
    return true;
}

bool materialize_gpu_capture_bundle_submit(const GpuCaptureBundle& bundle, size_t submit_index,
                                           GpuCaptureFile& capture, std::string& error) {
    error.clear();
    if (submit_index >= bundle.submits.size()) { error = "bundle submit index is out of range"; return false; }
    const auto& submit = bundle.submits[submit_index];
    if (submit.logical_bytes > std::numeric_limits<size_t>::max()) {
        error = "bundle submit is too large"; return false;
    }
    std::vector<uint8_t> bytes; bytes.reserve(static_cast<size_t>(submit.logical_bytes));
    for (uint32_t index : submit.chunk_indices) {
        if (index >= bundle.chunks.size()) { error = "bundle references an invalid chunk"; return false; }
        if (bundle.chunks[index].size() > submit.logical_bytes - bytes.size()) {
            error = "bundle submit chunk sizes exceed the manifest"; return false;
        }
        bytes.insert(bytes.end(), bundle.chunks[index].begin(), bundle.chunks[index].end());
    }
    if (bytes.size() != submit.logical_bytes) { error = "bundle submit is truncated"; return false; }
    if (!deserialize_gpu_capture(bytes, capture, error)) return false;
    if (capture.metadata.submit_index != submit.submit_index) {
        error = "bundle submit identity mismatch"; return false;
    }
    return true;
}

bool write_gpu_capture_bundle(const std::string& path, const GpuCaptureBundle& bundle,
                              std::string& error) {
    error.clear();
    if (bundle.chunk_hashes.size() != bundle.chunks.size() || bundle.chunks.size() > kMaxChunks ||
        bundle.submits.empty() || bundle.submits.size() > kMaxSubmits) {
        error = "invalid bundle structure"; return false;
    }
    Writer writer; writer.raw(kMagic, sizeof(kMagic)); writer.u32(kVersion); writer.u32(kEndian);
    writer.u32(bundle.chunk_bytes); writer.u64(bundle.logical_bytes);
    writer.u32(static_cast<uint32_t>(bundle.chunks.size()));
    for (size_t i = 0; i < bundle.chunks.size(); ++i) {
        const uint64_t hash = gpu_capture_hash(bundle.chunks[i]);
        if (bundle.chunk_hashes[i] != hash) { error = "bundle chunk hash mismatch"; return false; }
        writer.u64(hash); writer.bytes(bundle.chunks[i]);
    }
    writer.u32(static_cast<uint32_t>(bundle.submits.size()));
    for (const auto& submit : bundle.submits) {
        writer.u64(submit.submit_index); writer.u64(submit.logical_bytes);
        writer.u32(static_cast<uint32_t>(submit.chunk_indices.size()));
        for (uint32_t index : submit.chunk_indices) {
            if (index >= bundle.chunks.size()) { error = "bundle references an invalid chunk"; return false; }
            writer.u32(index);
        }
    }
    const uint64_t manifest_hash = gpu_capture_hash(writer.data);
    writer.u64(manifest_hash);
    if (writer.data.size() > kMaxFileBytes) { error = "bundle exceeds 4 GiB"; return false; }
    return install_file(path, writer.data, error);
}

bool read_gpu_capture_bundle(const std::string& path, GpuCaptureBundle& bundle,
                             std::string& error) {
    error.clear(); bundle = {};
    std::error_code ec; const uint64_t file_size = std::filesystem::file_size(path, ec);
    if (ec || file_size < 32 || file_size > kMaxFileBytes || file_size > std::numeric_limits<size_t>::max()) {
        error = "invalid bundle file size"; return false;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
    std::ifstream file(path, std::ios::binary);
    if (!file || !file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        error = "cannot read bundle"; return false;
    }
    uint64_t stored_hash = 0;
    Reader tail{bytes.data() + bytes.size() - 8, 8}; tail.u64(stored_hash);
    if (gpu_capture_hash(bytes.data(), bytes.size() - 8) != stored_hash) {
        error = "bundle manifest checksum mismatch"; return false;
    }
    Reader reader{bytes.data(), bytes.size() - 8}; uint8_t magic[8]; uint32_t version = 0, endian = 0;
    if (!reader.take(magic, 8) || std::memcmp(magic, kMagic, 8) || !reader.u32(version) ||
        version != kVersion || !reader.u32(endian) || endian != kEndian ||
        !reader.u32(bundle.chunk_bytes) || !bundle.chunk_bytes || bundle.chunk_bytes > (16u << 20) ||
        !reader.u64(bundle.logical_bytes)) {
        error = "unsupported bundle header"; return false;
    }
    bundle.version = version;
    uint32_t chunk_count = 0;
    if (!reader.u32(chunk_count) || chunk_count > kMaxChunks) { error = "invalid bundle chunk count"; return false; }
    bundle.chunks.resize(chunk_count); bundle.chunk_hashes.resize(chunk_count);
    for (uint32_t i = 0; i < chunk_count; ++i) {
        if (!reader.u64(bundle.chunk_hashes[i]) || !reader.bytes(bundle.chunks[i]) ||
            bundle.chunks[i].size() > bundle.chunk_bytes ||
            gpu_capture_hash(bundle.chunks[i]) != bundle.chunk_hashes[i]) {
            error = "invalid bundle chunk"; return false;
        }
    }
    uint32_t submit_count = 0;
    if (!reader.u32(submit_count) || !submit_count || submit_count > kMaxSubmits) {
        error = "invalid bundle submit count"; return false;
    }
    uint64_t logical_total = 0, prior_submit = 0; bundle.submits.resize(submit_count);
    for (auto& submit : bundle.submits) {
        uint32_t refs = 0;
        if (!reader.u64(submit.submit_index) || !reader.u64(submit.logical_bytes) ||
            !reader.u32(refs) || refs > kMaxChunks) { error = "invalid bundle submit"; return false; }
        submit.chunk_indices.resize(refs);
        for (auto& index : submit.chunk_indices)
            if (!reader.u32(index) || index >= chunk_count) { error = "invalid bundle chunk reference"; return false; }
        if (submit.submit_index <= prior_submit) { error = "bundle submits are not ordered"; return false; }
        prior_submit = submit.submit_index;
        if (logical_total > UINT64_MAX - submit.logical_bytes) { error = "bundle logical size overflow"; return false; }
        logical_total += submit.logical_bytes;
    }
    if (reader.left || logical_total != bundle.logical_bytes) {
        error = "bundle manifest size mismatch"; return false;
    }
    return true;
}

} // namespace prosper::gpu
