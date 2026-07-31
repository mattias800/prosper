#include "gpu_capture_bundle.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace prosper::gpu {
namespace {

constexpr uint8_t kMagic[8] = {'P', 'R', 'G', 'B', 'N', 'D', 'L', '\0'};
constexpr uint32_t kVersion = 2;
constexpr uint32_t kEndian = 0x01020304u;
constexpr uint32_t kMaxChunks = 1u << 20;
constexpr uint32_t kMaxSubmits = 4096;
constexpr uint64_t kMaxFileBytes = 4ull << 30;

struct Writer {
    std::vector<uint8_t> data;
    void raw(const void* p, size_t n) {
        if (!n) return;
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

bool append_chunks(GpuCaptureBundle& bundle, const std::vector<uint8_t>& bytes,
                   std::vector<uint32_t>& indices, std::string& error) {
    auto& by_hash = bundle.chunk_indices_by_hash;
    if (by_hash.empty() && !bundle.chunks.empty()) {
        by_hash.reserve(bundle.chunks.size());
        for (uint32_t i = 0; i < bundle.chunks.size(); ++i)
            by_hash[bundle.chunk_hashes[i]].push_back(i);
    }
    for (size_t offset = 0; offset < bytes.size();) {
        const size_t size = content_chunk_size(bytes, offset, bundle.chunk_bytes);
        std::vector<uint8_t> chunk(bytes.begin() + offset, bytes.begin() + offset + size);
        const uint64_t hash = gpu_capture_hash(chunk);
        uint32_t index = UINT32_MAX;
        auto found = by_hash.find(hash);
        if (found != by_hash.end())
            for (uint32_t candidate : found->second)
                if (candidate < bundle.chunks.size() && bundle.chunks[candidate] == chunk) {
                    index = candidate;
                    break;
                }
        if (index == UINT32_MAX) {
            if (bundle.chunks.size() >= kMaxChunks) {
                error = "bundle chunk count exceeds limit";
                return false;
            }
            index = static_cast<uint32_t>(bundle.chunks.size());
            bundle.chunk_hashes.push_back(hash);
            bundle.chunks.push_back(std::move(chunk));
            by_hash[hash].push_back(index);
        }
        indices.push_back(index);
        offset += size;
    }
    return true;
}

bool resource_equals(const GpuCaptureBundle& bundle, const GpuCaptureBundleResource& resource,
                     const std::vector<uint8_t>& bytes) {
    if (resource.logical_bytes != bytes.size()) return false;
    size_t offset = 0;
    for (uint32_t index : resource.chunk_indices) {
        if (index >= bundle.chunks.size() || bundle.chunks[index].size() > bytes.size() - offset ||
            !std::equal(bundle.chunks[index].begin(), bundle.chunks[index].end(), bytes.begin() + offset))
            return false;
        offset += bundle.chunks[index].size();
    }
    return offset == bytes.size();
}

uint64_t resource_hash(const GpuCaptureBundle& bundle,
                       const GpuCaptureBundleResource& resource, bool& valid) {
    uint64_t hash = 1469598103934665603ull;
    uint64_t bytes = 0;
    valid = true;
    for (uint32_t index : resource.chunk_indices) {
        if (index >= bundle.chunks.size() || bytes > UINT64_MAX - bundle.chunks[index].size()) {
            valid = false;
            return 0;
        }
        for (uint8_t byte : bundle.chunks[index]) {
            hash ^= byte;
            hash *= 1099511628211ull;
        }
        bytes += bundle.chunks[index].size();
    }
    valid = bytes == resource.logical_bytes;
    return hash;
}

GpuCaptureFile make_capture_manifest(const GpuCaptureFile& capture) {
    GpuCaptureFile manifest;
    manifest.metadata = capture.metadata;
    manifest.expected_output_hash = capture.expected_output_hash;
    manifest.expected_output_bytes = capture.expected_output_bytes;
    manifest.expected_output_valid = capture.expected_output_valid;
    manifest.rtt_seeds = capture.rtt_seeds;
    manifest.ds_seeds = capture.ds_seeds;
    manifest.shader_versions = capture.shader_versions;
    manifest.raw_shader_versions = capture.raw_shader_versions;
    manifest.draws = capture.draws;
    manifest.computes = capture.computes;
    manifest.dma_copies = capture.dma_copies;
    manifest.operations = capture.operations;
    manifest.failure_diagnostics = capture.failure_diagnostics;
    manifest.blobs.resize(capture.blobs.size());
    const uint64_t empty_hash = gpu_capture_hash(nullptr, 0);
    for (size_t i = 0; i < capture.blobs.size(); ++i) {
        manifest.blobs[i].guest_addr = capture.blobs[i].guest_addr;
        manifest.blobs[i].content_hash = empty_hash;
    }
    return manifest;
}

} // namespace

uint64_t gpu_capture_bundle_unique_bytes(const GpuCaptureBundle& bundle) {
    uint64_t total = 0;
    for (const auto& chunk : bundle.chunks) {
        if (total > UINT64_MAX - chunk.size()) return UINT64_MAX;
        total += chunk.size();
    }
    return total;
}

GpuCaptureBundleStats gpu_capture_bundle_stats(const GpuCaptureBundle& bundle) {
    GpuCaptureBundleStats stats;
    std::unordered_set<uint32_t> manifest_chunks, referenced_resources, resource_chunks;
    for (const auto& submit : bundle.submits) {
        manifest_chunks.insert(submit.chunk_indices.begin(), submit.chunk_indices.end());
        referenced_resources.insert(submit.blob_resource_indices.begin(),
                                    submit.blob_resource_indices.end());
    }
    for (uint32_t index : referenced_resources)
        if (index < bundle.resources.size())
            resource_chunks.insert(bundle.resources[index].chunk_indices.begin(),
                                   bundle.resources[index].chunk_indices.end());
    for (uint32_t index : manifest_chunks)
        if (index < bundle.chunks.size()) stats.manifest_unique_bytes += bundle.chunks[index].size();
    for (uint32_t index : resource_chunks)
        if (index < bundle.chunks.size()) stats.resource_unique_bytes += bundle.chunks[index].size();
    for (const auto& submit : bundle.submits) {
        stats.resource_reference_count += submit.blob_resource_indices.size();
        for (uint32_t index : submit.blob_resource_indices)
            if (index < bundle.resources.size()) stats.resource_logical_bytes += bundle.resources[index].logical_bytes;
    }
    stats.exact_reuse_count = stats.resource_reference_count > referenced_resources.size()
        ? stats.resource_reference_count - referenced_resources.size() : 0;
    return stats;
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
    if (bundle.version != kVersion) {
        error = "cannot append to a legacy capture bundle";
        return false;
    }
    struct Rollback {
        GpuCaptureBundle& bundle;
        size_t chunks;
        size_t hashes;
        size_t resources;
        size_t submits;
        uint64_t logical_bytes;
        bool committed = false;
        ~Rollback() {
            if (committed) return;
            bundle.chunks.resize(chunks);
            bundle.chunk_hashes.resize(hashes);
            bundle.resources.resize(resources);
            bundle.submits.resize(submits);
            bundle.logical_bytes = logical_bytes;
            bundle.chunk_indices_by_hash.clear();
            for (uint32_t i = 0; i < bundle.chunks.size(); ++i)
                bundle.chunk_indices_by_hash[bundle.chunk_hashes[i]].push_back(i);
            bundle.resource_indices_by_hash.clear();
            for (uint32_t i = 0; i < bundle.resources.size(); ++i)
                bundle.resource_indices_by_hash[bundle.resources[i].content_hash].push_back(i);
        }
    } rollback{bundle, bundle.chunks.size(), bundle.chunk_hashes.size(), bundle.resources.size(),
               bundle.submits.size(), bundle.logical_bytes};
    GpuCaptureBundleSubmit submit;
    submit.submit_index = capture.metadata.submit_index;
    auto& resources_by_hash = bundle.resource_indices_by_hash;
    if (resources_by_hash.empty() && !bundle.resources.empty()) {
        resources_by_hash.reserve(bundle.resources.size());
        for (uint32_t i = 0; i < bundle.resources.size(); ++i)
            resources_by_hash[bundle.resources[i].content_hash].push_back(i);
    }
    uint64_t resource_bytes = 0;
    for (const auto& blob : capture.blobs) {
        if (blob.bytes_read > blob.bytes.size()) {
            error = "capture blob readable bytes exceed its content";
            return false;
        }
        uint64_t hash = blob.content_hash;
        uint32_t index = UINT32_MAX;
        auto found = resources_by_hash.find(hash);
        if (found != resources_by_hash.end())
            for (uint32_t candidate : found->second)
                if (resource_equals(bundle, bundle.resources[candidate], blob.bytes)) {
                    index = candidate;
                    break;
                }
        if (index == UINT32_MAX) {
            const uint64_t actual_hash = gpu_capture_hash(blob.bytes);
            if (hash && hash != actual_hash) {
                error = "capture blob content hash mismatch";
                return false;
            }
            hash = actual_hash;
            found = resources_by_hash.find(hash);
            if (found != resources_by_hash.end())
                for (uint32_t candidate : found->second)
                    if (resource_equals(bundle, bundle.resources[candidate], blob.bytes)) {
                        index = candidate;
                        break;
                    }
            if (index == UINT32_MAX) {
                if (bundle.resources.size() >= kMaxChunks) {
                    error = "bundle resource count exceeds limit";
                    return false;
                }
                index = static_cast<uint32_t>(bundle.resources.size());
                GpuCaptureBundleResource resource;
                resource.content_hash = hash;
                resource.logical_bytes = blob.bytes.size();
                if (!append_chunks(bundle, blob.bytes, resource.chunk_indices, error)) return false;
                bundle.resources.push_back(std::move(resource));
                resources_by_hash[hash].push_back(index);
            }
        }
        submit.blob_resource_indices.push_back(index);
        submit.blob_bytes_read.push_back(blob.bytes_read);
        if (resource_bytes > UINT64_MAX - blob.bytes.size()) {
            error = "bundle submit resource size overflow";
            return false;
        }
        resource_bytes += blob.bytes.size();
    }

    GpuCaptureFile manifest = make_capture_manifest(capture);
    std::vector<uint8_t> bytes;
    if (!serialize_gpu_capture(manifest, bytes, error)) return false;
    submit.manifest_bytes = bytes.size();
    if (submit.manifest_bytes > UINT64_MAX - resource_bytes) {
        error = "bundle submit logical size overflow";
        return false;
    }
    submit.logical_bytes = submit.manifest_bytes + resource_bytes;
    if (!append_chunks(bundle, bytes, submit.chunk_indices, error)) return false;
    if (bundle.logical_bytes > UINT64_MAX - submit.logical_bytes) {
        error = "bundle logical size overflow";
        return false;
    }
    bundle.logical_bytes += submit.logical_bytes;
    bundle.submits.push_back(std::move(submit));
    rollback.committed = true;
    return true;
}

bool materialize_gpu_capture_bundle_manifest(const GpuCaptureBundle& bundle, size_t submit_index,
                                             GpuCaptureFile& capture, std::string& error) {
    error.clear();
    if (submit_index >= bundle.submits.size()) { error = "bundle submit index is out of range"; return false; }
    const auto& submit = bundle.submits[submit_index];
    const uint64_t serialized_bytes = bundle.version >= 2 ? submit.manifest_bytes : submit.logical_bytes;
    if (serialized_bytes > std::numeric_limits<size_t>::max()) {
        error = "bundle submit is too large"; return false;
    }
    std::vector<uint8_t> bytes; bytes.reserve(static_cast<size_t>(serialized_bytes));
    for (uint32_t index : submit.chunk_indices) {
        if (index >= bundle.chunks.size()) { error = "bundle references an invalid chunk"; return false; }
        if (bundle.chunks[index].size() > serialized_bytes - bytes.size()) {
            error = "bundle submit chunk sizes exceed the manifest"; return false;
        }
        bytes.insert(bytes.end(), bundle.chunks[index].begin(), bundle.chunks[index].end());
    }
    if (bytes.size() != serialized_bytes) { error = "bundle submit is truncated"; return false; }
    if (!deserialize_gpu_capture(bytes, capture, error)) return false;
    if (capture.metadata.submit_index != submit.submit_index) {
        error = "bundle submit identity mismatch"; return false;
    }
    return true;
}

bool materialize_gpu_capture_bundle_submit(const GpuCaptureBundle& bundle, size_t submit_index,
                                           GpuCaptureFile& capture, std::string& error) {
    if (!materialize_gpu_capture_bundle_manifest(bundle, submit_index, capture, error)) return false;
    const auto& submit = bundle.submits[submit_index];
    if (bundle.version >= 2) {
        if (capture.blobs.size() != submit.blob_resource_indices.size() ||
            capture.blobs.size() != submit.blob_bytes_read.size()) {
            error = "bundle submit resource manifest mismatch";
            return false;
        }
        uint64_t logical_bytes = submit.manifest_bytes;
        for (size_t i = 0; i < capture.blobs.size(); ++i) {
            const uint32_t resource_index = submit.blob_resource_indices[i];
            if (resource_index >= bundle.resources.size()) {
                error = "bundle references an invalid resource";
                return false;
            }
            const auto& resource = bundle.resources[resource_index];
            if (submit.blob_bytes_read[i] > resource.logical_bytes ||
                resource.logical_bytes > std::numeric_limits<size_t>::max() ||
                logical_bytes > UINT64_MAX - resource.logical_bytes) {
                error = "bundle resource metadata is invalid";
                return false;
            }
            capture.blobs[i].bytes.clear();
            capture.blobs[i].bytes.reserve(static_cast<size_t>(resource.logical_bytes));
            for (uint32_t index : resource.chunk_indices) {
                if (index >= bundle.chunks.size() || bundle.chunks[index].size() >
                        resource.logical_bytes - capture.blobs[i].bytes.size()) {
                    error = "bundle resource chunks are invalid";
                    return false;
                }
                capture.blobs[i].bytes.insert(capture.blobs[i].bytes.end(),
                                              bundle.chunks[index].begin(),
                                              bundle.chunks[index].end());
            }
            if (capture.blobs[i].bytes.size() != resource.logical_bytes ||
                gpu_capture_hash(capture.blobs[i].bytes) != resource.content_hash) {
                error = "bundle resource content mismatch";
                return false;
            }
            capture.blobs[i].bytes_read = submit.blob_bytes_read[i];
            capture.blobs[i].content_hash = resource.content_hash;
            logical_bytes += resource.logical_bytes;
        }
        if (logical_bytes != submit.logical_bytes) {
            error = "bundle submit logical size mismatch";
            return false;
        }
    }
    return true;
}

bool replace_gpu_capture_bundle_submit_ds_seeds(
    GpuCaptureBundle& bundle, size_t submit_index,
    const std::vector<GpuCaptureDsSeed>& ds_seeds, std::string& error) {
    error.clear();
    if (bundle.version != kVersion || submit_index >= bundle.submits.size()) {
        error = "bundle DS replacement requires a current-version submit";
        return false;
    }
    GpuCaptureFile manifest;
    if (!materialize_gpu_capture_bundle_manifest(bundle, submit_index, manifest, error))
        return false;
    manifest.ds_seeds = ds_seeds;
    std::vector<uint8_t> bytes;
    if (!serialize_gpu_capture(manifest, bytes, error)) return false;

    GpuCaptureBundleSubmit& submit = bundle.submits[submit_index];
    const size_t old_chunk_count = bundle.chunks.size();
    const size_t old_hash_count = bundle.chunk_hashes.size();
    const uint64_t old_manifest_bytes = submit.manifest_bytes;
    const uint64_t old_logical_bytes = submit.logical_bytes;
    if (old_logical_bytes < old_manifest_bytes) {
        error = "bundle submit logical size is smaller than its manifest";
        return false;
    }
    const uint64_t resource_bytes = old_logical_bytes - old_manifest_bytes;
    if (bytes.size() > UINT64_MAX - resource_bytes ||
        bundle.logical_bytes < old_logical_bytes ||
        bundle.logical_bytes - old_logical_bytes >
            UINT64_MAX - (bytes.size() + resource_bytes)) {
        error = "bundle DS replacement size overflow";
        return false;
    }
    std::vector<uint32_t> indices;
    if (!append_chunks(bundle, bytes, indices, error)) {
        bundle.chunks.resize(old_chunk_count);
        bundle.chunk_hashes.resize(old_hash_count);
        bundle.chunk_indices_by_hash.clear();
        for (uint32_t i = 0; i < bundle.chunks.size(); ++i)
            bundle.chunk_indices_by_hash[bundle.chunk_hashes[i]].push_back(i);
        return false;
    }
    submit.chunk_indices = std::move(indices);
    submit.manifest_bytes = bytes.size();
    submit.logical_bytes = bytes.size() + resource_bytes;
    bundle.logical_bytes = bundle.logical_bytes - old_logical_bytes + submit.logical_bytes;
    return true;
}

bool compact_gpu_capture_bundle(GpuCaptureBundle& bundle, std::string& error) {
    error.clear();
    if (bundle.chunk_hashes.size() != bundle.chunks.size()) {
        error = "bundle chunk hash count mismatch";
        return false;
    }
    std::vector<uint8_t> resource_used(bundle.resources.size(), 0);
    for (const auto& submit : bundle.submits) {
        for (uint32_t index : submit.chunk_indices)
            if (index >= bundle.chunks.size()) {
                error = "bundle submit references an invalid chunk";
                return false;
            }
        for (uint32_t index : submit.blob_resource_indices) {
            if (index >= bundle.resources.size()) {
                error = "bundle submit references an invalid resource";
                return false;
            }
            resource_used[index] = 1;
        }
    }
    for (const auto& resource : bundle.resources)
        for (uint32_t index : resource.chunk_indices)
            if (index >= bundle.chunks.size()) {
                error = "bundle resource references an invalid chunk";
                return false;
            }

    std::vector<uint32_t> resource_map(bundle.resources.size(), UINT32_MAX);
    std::vector<GpuCaptureBundleResource> resources;
    resources.reserve(std::count(resource_used.begin(), resource_used.end(), uint8_t{1}));
    for (uint32_t i = 0; i < bundle.resources.size(); ++i) {
        if (!resource_used[i]) continue;
        resource_map[i] = static_cast<uint32_t>(resources.size());
        resources.push_back(std::move(bundle.resources[i]));
    }
    for (auto& submit : bundle.submits)
        for (auto& index : submit.blob_resource_indices) index = resource_map[index];

    std::vector<uint8_t> chunk_used(bundle.chunks.size(), 0);
    for (const auto& submit : bundle.submits)
        for (uint32_t index : submit.chunk_indices) chunk_used[index] = 1;
    for (const auto& resource : resources)
        for (uint32_t index : resource.chunk_indices) chunk_used[index] = 1;
    std::vector<uint32_t> chunk_map(bundle.chunks.size(), UINT32_MAX);
    std::vector<std::vector<uint8_t>> chunks;
    std::vector<uint64_t> hashes;
    chunks.reserve(std::count(chunk_used.begin(), chunk_used.end(), uint8_t{1}));
    hashes.reserve(chunks.capacity());
    for (uint32_t i = 0; i < bundle.chunks.size(); ++i) {
        if (!chunk_used[i]) continue;
        chunk_map[i] = static_cast<uint32_t>(chunks.size());
        chunks.push_back(std::move(bundle.chunks[i]));
        hashes.push_back(bundle.chunk_hashes[i]);
    }
    for (auto& submit : bundle.submits)
        for (auto& index : submit.chunk_indices) index = chunk_map[index];
    for (auto& resource : resources)
        for (auto& index : resource.chunk_indices) index = chunk_map[index];

    bundle.resources = std::move(resources);
    bundle.chunks = std::move(chunks);
    bundle.chunk_hashes = std::move(hashes);
    bundle.chunk_indices_by_hash.clear();
    for (uint32_t i = 0; i < bundle.chunks.size(); ++i)
        bundle.chunk_indices_by_hash[bundle.chunk_hashes[i]].push_back(i);
    bundle.resource_indices_by_hash.clear();
    for (uint32_t i = 0; i < bundle.resources.size(); ++i)
        bundle.resource_indices_by_hash[bundle.resources[i].content_hash].push_back(i);
    return true;
}

bool write_gpu_capture_bundle(const std::string& path, const GpuCaptureBundle& bundle,
                              std::string& error) {
    error.clear();
    if (bundle.chunk_hashes.size() != bundle.chunks.size() || bundle.chunks.size() > kMaxChunks ||
        bundle.resources.size() > kMaxChunks || bundle.submits.empty() ||
        bundle.submits.size() > kMaxSubmits || bundle.version < 1 || bundle.version > kVersion) {
        error = "invalid bundle structure"; return false;
    }
    Writer writer; writer.raw(kMagic, sizeof(kMagic)); writer.u32(bundle.version); writer.u32(kEndian);
    writer.u32(bundle.chunk_bytes); writer.u64(bundle.logical_bytes);
    writer.u32(static_cast<uint32_t>(bundle.chunks.size()));
    for (size_t i = 0; i < bundle.chunks.size(); ++i) {
        const uint64_t hash = gpu_capture_hash(bundle.chunks[i]);
        if (bundle.chunk_hashes[i] != hash) { error = "bundle chunk hash mismatch"; return false; }
        writer.u64(hash); writer.bytes(bundle.chunks[i]);
    }
    if (bundle.version >= 2) {
        writer.u32(static_cast<uint32_t>(bundle.resources.size()));
        for (const auto& resource : bundle.resources) {
            bool valid = false;
            const uint64_t hash = resource_hash(bundle, resource, valid);
            if (!valid || resource.content_hash != hash) {
                error = "bundle resource hash mismatch";
                return false;
            }
            writer.u64(hash);
            writer.u64(resource.logical_bytes);
            writer.u32(static_cast<uint32_t>(resource.chunk_indices.size()));
            for (uint32_t index : resource.chunk_indices) {
                if (index >= bundle.chunks.size()) {
                    error = "bundle resource references an invalid chunk";
                    return false;
                }
                writer.u32(index);
            }
        }
    }
    writer.u32(static_cast<uint32_t>(bundle.submits.size()));
    for (const auto& submit : bundle.submits) {
        uint64_t reconstructed_bytes = 0;
        for (uint32_t index : submit.chunk_indices) {
            if (index >= bundle.chunks.size() ||
                reconstructed_bytes > UINT64_MAX - bundle.chunks[index].size()) {
                error = "bundle references an invalid chunk";
                return false;
            }
            reconstructed_bytes += bundle.chunks[index].size();
        }
        const uint64_t expected_manifest_bytes = bundle.version >= 2
            ? submit.manifest_bytes : submit.logical_bytes;
        if (reconstructed_bytes != expected_manifest_bytes) {
            error = "bundle submit manifest size mismatch";
            return false;
        }
        writer.u64(submit.submit_index); writer.u64(submit.logical_bytes);
        if (bundle.version >= 2) writer.u64(submit.manifest_bytes);
        writer.u32(static_cast<uint32_t>(submit.chunk_indices.size()));
        for (uint32_t index : submit.chunk_indices) {
            if (index >= bundle.chunks.size()) { error = "bundle references an invalid chunk"; return false; }
            writer.u32(index);
        }
        if (bundle.version >= 2) {
            if (submit.blob_resource_indices.size() != submit.blob_bytes_read.size()) {
                error = "bundle resource reference count mismatch";
                return false;
            }
            writer.u32(static_cast<uint32_t>(submit.blob_resource_indices.size()));
            for (size_t i = 0; i < submit.blob_resource_indices.size(); ++i) {
                if (submit.blob_resource_indices[i] >= bundle.resources.size()) {
                    error = "bundle references an invalid resource";
                    return false;
                }
                writer.u32(submit.blob_resource_indices[i]);
                writer.u64(submit.blob_bytes_read[i]);
                if (reconstructed_bytes > UINT64_MAX -
                        bundle.resources[submit.blob_resource_indices[i]].logical_bytes) {
                    error = "bundle submit logical size overflow";
                    return false;
                }
                reconstructed_bytes +=
                    bundle.resources[submit.blob_resource_indices[i]].logical_bytes;
            }
        }
        if (reconstructed_bytes != submit.logical_bytes) {
            error = "bundle submit logical size mismatch";
            return false;
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
        version < 1 || version > kVersion || !reader.u32(endian) || endian != kEndian ||
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
        bundle.chunk_indices_by_hash[bundle.chunk_hashes[i]].push_back(i);
    }
    if (version >= 2) {
        uint32_t resource_count = 0;
        if (!reader.u32(resource_count) || resource_count > kMaxChunks) {
            error = "invalid bundle resource count";
            return false;
        }
        bundle.resources.resize(resource_count);
        for (uint32_t resource_index = 0; resource_index < bundle.resources.size(); ++resource_index) {
            auto& resource = bundle.resources[resource_index];
            uint32_t refs = 0;
            if (!reader.u64(resource.content_hash) || !reader.u64(resource.logical_bytes) ||
                !reader.u32(refs) || refs > kMaxChunks) {
                error = "invalid bundle resource";
                return false;
            }
            resource.chunk_indices.resize(refs);
            for (auto& index : resource.chunk_indices)
                if (!reader.u32(index) || index >= bundle.chunks.size()) {
                    error = "invalid bundle resource chunk reference";
                    return false;
                }
            bool valid = false;
            if (resource_hash(bundle, resource, valid) != resource.content_hash || !valid) {
                error = "invalid bundle resource content";
                return false;
            }
            bundle.resource_indices_by_hash[resource.content_hash].push_back(resource_index);
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
            (version >= 2 && !reader.u64(submit.manifest_bytes)) ||
            !reader.u32(refs) || refs > kMaxChunks) { error = "invalid bundle submit"; return false; }
        if (version < 2) submit.manifest_bytes = submit.logical_bytes;
        submit.chunk_indices.resize(refs);
        for (auto& index : submit.chunk_indices)
            if (!reader.u32(index) || index >= chunk_count) { error = "invalid bundle chunk reference"; return false; }
        if (version >= 2) {
            uint32_t resource_refs = 0;
            if (!reader.u32(resource_refs) || resource_refs > kMaxChunks) {
                error = "invalid bundle resource reference count";
                return false;
            }
            submit.blob_resource_indices.resize(resource_refs);
            submit.blob_bytes_read.resize(resource_refs);
            for (uint32_t i = 0; i < resource_refs; ++i)
                if (!reader.u32(submit.blob_resource_indices[i]) ||
                    submit.blob_resource_indices[i] >= bundle.resources.size() ||
                    !reader.u64(submit.blob_bytes_read[i]) ||
                    submit.blob_bytes_read[i] >
                        bundle.resources[submit.blob_resource_indices[i]].logical_bytes) {
                    error = "invalid bundle resource reference";
                    return false;
                }
        }
        uint64_t reconstructed_bytes = 0;
        for (uint32_t index : submit.chunk_indices) {
            if (reconstructed_bytes > UINT64_MAX - bundle.chunks[index].size()) {
                error = "bundle submit manifest size overflow";
                return false;
            }
            reconstructed_bytes += bundle.chunks[index].size();
        }
        if (reconstructed_bytes != submit.manifest_bytes) {
            error = "bundle submit manifest size mismatch";
            return false;
        }
        if (version >= 2) {
            for (uint32_t index : submit.blob_resource_indices) {
                if (reconstructed_bytes > UINT64_MAX - bundle.resources[index].logical_bytes) {
                    error = "bundle submit logical size overflow";
                    return false;
                }
                reconstructed_bytes += bundle.resources[index].logical_bytes;
            }
        }
        if (reconstructed_bytes != submit.logical_bytes) {
            error = "bundle submit logical size mismatch";
            return false;
        }
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
