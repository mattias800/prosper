// Execute a compile-only merged-NGG export module against one failed draw's captured buffer
// resources. This reports guest lane outputs, not a raster image or a validated draw ABI.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/compute_runner.h"

#include <array>
#include <bit>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

namespace {
bool eligible_buffer(const prosper::gpu::ShaderResource& resource, uint32_t slot) {
    using prosper::gpu::ResourceClass;
    constexpr std::array<ResourceClass, 5> kExpectedClasses = {
        ResourceClass::ConstantBuffer, ResourceClass::VertexBuffer,
        ResourceClass::VertexBuffer, ResourceClass::ConstantBuffer,
        ResourceClass::ConstantBuffer};
    return slot < kExpectedClasses.size() && resource.cls == kExpectedClasses[slot] &&
           resource.size && resource.size <= 4096u && resource.size % sizeof(uint32_t) == 0;
}

bool exact_probe_local_size(const std::vector<uint32_t>& words, uint32_t expected_x) {
    // The runner dispatches the single GLCompute entry named "main" as a 64-lane group.
    // LocalSizeId or BuiltIn WorkgroupSize can override a literal LocalSize, so reject both.
    // A compile-only four-wave module must never be dispatched as unrelated 64-lane groups.
    if (words.size() < 5 || words[0] != 0x07230203u) return false;
    uint32_t entry = 0;
    uint32_t mode_entry = 0;
    bool found_mode = false;
    for (size_t at = 5; at < words.size();) {
        const uint32_t count = words[at] >> 16u;
        if (!count || at + count > words.size()) return false;
        const uint32_t op = words[at] & 0xffffu;
        if (op == 15u) { // OpEntryPoint GLCompute %id "main"
            if (entry || count < 5u || words[at + 1u] != 5u || !words[at + 2u] ||
                words[at + 3u] != 0x6e69616du || words[at + 4u] != 0u)
                return false;
            entry = words[at + 2u];
        } else if (op == 16u && count >= 3u && words[at + 2u] == 17u) {
            if (found_mode || count != 6u || words[at + 3u] != expected_x ||
                words[at + 4u] != 1u || words[at + 5u] != 1u)
                return false;
            found_mode = true;
            mode_entry = words[at + 1u];
        } else if (op == 331u || op == 332u || (op >= 73u && op <= 75u)) {
            // OpExecutionModeId, OpDecorateId, and decoration groups are not emitted by this
            // probe. Refuse their alternate size/decorating paths rather than parsing a subset.
            return false;
        } else if ((op == 71u && count >= 4u && words[at + 2u] == 11u &&
                    words[at + 3u] == 25u) ||
                   (op == 72u && count >= 5u && words[at + 3u] == 11u &&
                    words[at + 4u] == 25u)) {
            // OpDecorate/OpMemberDecorate BuiltIn WorkgroupSize takes precedence over LocalSize.
            return false;
        }
        at += count;
    }
    return entry && found_mode && mode_entry == entry;
}

bool selftest() {
    using prosper::gpu::ResourceClass;
    prosper::gpu::ShaderResource resource;
    resource.size = 16;
    for (uint32_t slot = 0; slot < 5; ++slot) {
        resource.cls = slot == 1 || slot == 2 ? ResourceClass::VertexBuffer
                                              : ResourceClass::ConstantBuffer;
        if (!eligible_buffer(resource, slot)) return false;
        resource.cls = resource.cls == ResourceClass::VertexBuffer
                           ? ResourceClass::ConstantBuffer : ResourceClass::VertexBuffer;
        if (eligible_buffer(resource, slot)) return false;
        resource.cls = ResourceClass::Texture;
        if (eligible_buffer(resource, slot)) return false;
        resource.cls = ResourceClass::StorageImage;
        if (eligible_buffer(resource, slot)) return false;
    }
    resource.cls = ResourceClass::ConstantBuffer;
    resource.size = 15;
    if (eligible_buffer(resource, 0)) return false;
    resource.size = 4097;
    if (eligible_buffer(resource, 0)) return false;
    resource.size = 0;
    if (eligible_buffer(resource, 0) || eligible_buffer(resource, 5)) return false;
    std::vector<uint32_t> module = {
        0x07230203u, 0x00010300u, 0u, 8u, 0u,
        (5u << 16u) | 15u, 5u, 1u, 0x6e69616du, 0u,
        (6u << 16u) | 16u, 1u, 17u, 64u, 1u, 1u,
    };
    if (!exact_probe_local_size(module, 64u)) return false;
    auto wrong_size = module;
    wrong_size[13] = 256u;
    if (exact_probe_local_size(wrong_size, 64u) ||
        !exact_probe_local_size(wrong_size, 256u)) return false;
    auto wrong_entry = module;
    wrong_entry[11] = 2u;
    if (exact_probe_local_size(wrong_entry, 64u)) return false;
    auto override_size = module;
    override_size.insert(override_size.end(), {(4u << 16u) | 71u, 3u, 11u, 25u});
    if (exact_probe_local_size(override_size, 64u)) return false;
    auto id_size = module;
    id_size.insert(id_size.end(), {(6u << 16u) | 331u, 1u, 38u, 3u, 4u, 5u});
    return !exact_probe_local_size(id_size, 64u);
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--selftest") == 0)
        return selftest() ? 0 : 1;
    if (argc < 5 || argc > 7) {
        std::fprintf(stderr,
                     "usage: %s CAPTURE FAILURE COMPILE_ONLY_MODULE.spv OUTPUT.bin "
                     "[--zero-inputs|--zero-binding=2..6] [--packed-offsets=FILE]\n",
                     argv[0]);
        return 2;
    }
    bool zero_inputs = false;
    uint32_t zero_binding = 0;
    std::string packed_offsets_path;
    for (int i = 5; i < argc; ++i) {
        if (std::strcmp(argv[i], "--zero-inputs") == 0 && !zero_inputs && !zero_binding) {
            zero_inputs = true;
        } else if (std::strncmp(argv[i], "--zero-binding=", 15) == 0 &&
                   !zero_inputs && !zero_binding) {
            char* binding_end = nullptr;
            errno = 0;
            const unsigned long parsed = std::strtoul(argv[i] + 15, &binding_end, 10);
            if (errno || !binding_end || *binding_end || parsed < 2 || parsed > 6) {
                std::fprintf(stderr, "ngg_capture_probe: invalid zero binding\n");
                return 2;
            }
            zero_binding = static_cast<uint32_t>(parsed);
        } else if (std::strncmp(argv[i], "--packed-offsets=", 17) == 0 &&
                   argv[i][17] && packed_offsets_path.empty()) {
            packed_offsets_path = argv[i] + 17;
        } else {
            std::fprintf(stderr, "ngg_capture_probe: invalid or duplicate option %s\n", argv[i]);
            return 2;
        }
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long failure_index = std::strtoul(argv[2], &end, 0);
    if (errno || !end || *end || failure_index > UINT32_MAX) {
        std::fprintf(stderr, "ngg_capture_probe: invalid failure index\n");
        return 2;
    }
    prosper::gpu::GpuCaptureFile capture;
    std::string error;
    if (!prosper::gpu::read_gpu_capture(argv[1], capture, error)) {
        std::fprintf(stderr, "ngg_capture_probe: capture: %s\n", error.c_str());
        return 2;
    }
    if (!capture.failure_diagnostics_available ||
        failure_index >= capture.failure_diagnostics.size()) {
        std::fprintf(stderr, "ngg_capture_probe: failure index out of range\n");
        return 2;
    }
    const auto& failure = capture.failure_diagnostics[failure_index];
    if (failure.pipeline_present)
        std::fprintf(stderr, "[ngg-input] topology=%u draw=%llu vertices=%u instances=%u\n",
                     failure.pipeline.topology,
                     static_cast<unsigned long long>(failure.source_index),
                     failure.vertex_count, failure.instance_count);
    if (failure.stages.size() < 2 ||
        failure.stages[0].stage != prosper::gpu::ShaderProgramStage::Vertex ||
        failure.stages[1].stage != prosper::gpu::ShaderProgramStage::Vertex ||
        !failure.stages[0].resource_table.present ||
        !failure.vertex_count || !failure.instance_count ||
        failure.vertex_count > 4096u || failure.instance_count > 4096u ||
        static_cast<uint64_t>(failure.vertex_count) * failure.instance_count > 65536u) {
        std::fprintf(stderr, "ngg_capture_probe: unsupported or incomplete split draw\n");
        return 2;
    }
    std::array<std::vector<uint32_t>, 5> buffers;
    std::array<bool, 5> seen{};
    std::array<prosper::gpu::ResourceClass, 5> classes{};
    for (const auto& captured : failure.stages[0].resource_table.resources) {
        const auto& r = captured.resource;
        if (r.binding < 2u || r.binding > 6u) continue;
        const uint32_t slot = r.binding - 2u;
        const bool valid_blob = captured.blob_index < capture.blobs.size() &&
            captured.blob_offset <= capture.blobs[captured.blob_index].bytes.size() &&
            r.size <= capture.blobs[captured.blob_index].bytes.size() - captured.blob_offset;
        if (seen[slot] || !valid_blob || !eligible_buffer(r, slot)) {
            std::fprintf(stderr,
                         "ngg_capture_probe: incomplete/non-buffer binding %u class=%u "
                         "size=%u blob=%u duplicate=%d\n",
                         r.binding, static_cast<uint32_t>(r.cls), r.size,
                         captured.blob_index, seen[slot]);
            return 2;
        }
        seen[slot] = true;
        classes[slot] = r.cls;
        buffers[slot].resize(r.size / sizeof(uint32_t));
        const auto& blob = capture.blobs[captured.blob_index].bytes;
        std::memcpy(buffers[slot].data(), blob.data() + captured.blob_offset, r.size);
    }
    for (uint32_t slot = 0; slot < seen.size(); ++slot)
        if (!seen[slot]) {
            std::fprintf(stderr, "ngg_capture_probe: missing binding %u\n", slot + 2u);
            return 2;
        }
    for (uint32_t slot = 0; slot < buffers.size(); ++slot) {
        const auto& words = buffers[slot];
        std::fprintf(stderr, "[ngg-input] binding=%u class=%u words=%zu first=",
                     slot + 2u, static_cast<uint32_t>(classes[slot]), words.size());
        for (size_t index = 0; index < std::min<size_t>(words.size(), 16); ++index)
            std::fprintf(stderr, "%s%08x", index ? "," : "", words[index]);
        std::fprintf(stderr, "\n");
    }
    if (zero_inputs)
        for (auto& buffer : buffers) std::fill(buffer.begin(), buffer.end(), 0u);
    else if (zero_binding)
        std::fill(buffers[zero_binding - 2u].begin(),
                  buffers[zero_binding - 2u].end(), 0u);
    std::ifstream module_file(argv[3], std::ios::binary | std::ios::ate);
    const std::streampos module_bytes = module_file ? module_file.tellg() : std::streampos(-1);
    if (module_bytes < 20 || module_bytes > (1 << 20) ||
        static_cast<uint64_t>(module_bytes) % sizeof(uint32_t)) {
        std::fprintf(stderr, "ngg_capture_probe: invalid SPIR-V file\n");
        return 2;
    }
    std::vector<uint32_t> module(static_cast<size_t>(module_bytes) / sizeof(uint32_t));
    module_file.seekg(0);
    if (!module_file.read(reinterpret_cast<char*>(module.data()), module_bytes) ||
        module[0] != 0x07230203u) {
        std::fprintf(stderr, "ngg_capture_probe: cannot read SPIR-V file\n");
        return 2;
    }
    if (!exact_probe_local_size(module, 64u)) {
        std::fprintf(stderr,
                     "ngg_capture_probe: module must declare one 64x1x1 workgroup; "
                     "four-wave modules are compile-only here\n");
        return 2;
    }
    const uint32_t lanes = failure.vertex_count * failure.instance_count;
    constexpr uint32_t kWords = prosper::gpu::kNggExportProbeWords;
    const std::array<std::vector<uint32_t>, 3> extra{
        buffers[2], buffers[3], buffers[4]};
    std::vector<float> launch_inputs(lanes, 0.0f);
    if (!packed_offsets_path.empty()) {
        const size_t expected_bytes = static_cast<size_t>(lanes) * 2u * sizeof(uint32_t);
        std::ifstream offsets(packed_offsets_path, std::ios::binary | std::ios::ate);
        if (!offsets || offsets.tellg() != static_cast<std::streampos>(expected_bytes)) {
            std::fprintf(stderr,
                         "ngg_capture_probe: packed offsets require exactly %zu bytes\n",
                         expected_bytes);
            return 2;
        }
        std::vector<uint32_t> raw_offsets(static_cast<size_t>(lanes) * 2u);
        offsets.seekg(0);
        if (!offsets.read(reinterpret_cast<char*>(raw_offsets.data()), expected_bytes)) {
            std::fprintf(stderr, "ngg_capture_probe: cannot read packed offsets\n");
            return 2;
        }
        launch_inputs.resize(raw_offsets.size());
        for (size_t i = 0; i < raw_offsets.size(); ++i)
            launch_inputs[i] = std::bit_cast<float>(raw_offsets[i]);
    }
    const auto result = prosper::test::run_compute(
        module, launch_inputs, lanes, lanes * kWords,
        buffers[0], buffers[1], nullptr, 64, nullptr, nullptr, &extra);
    if (result.size() != static_cast<size_t>(lanes) * kWords) {
        std::fprintf(stderr, "ngg_capture_probe: Vulkan dispatch failed or timed out\n");
        return 1;
    }
    std::vector<uint32_t> raw(result.size());
    for (size_t i = 0; i < result.size(); ++i) raw[i] = std::bit_cast<uint32_t>(result[i]);
    FILE* output = std::fopen(argv[4], "wb");
    if (!output) {
        std::fprintf(stderr, "ngg_capture_probe: cannot open %s\n", argv[4]);
        return 2;
    }
    const bool wrote = std::fwrite(raw.data(), sizeof(uint32_t), raw.size(), output) == raw.size();
    const bool closed = std::fclose(output) == 0;
    if (!wrote || !closed) {
        std::fprintf(stderr, "ngg_capture_probe: cannot write %s\n", argv[4]);
        return 2;
    }
    std::set<uint32_t> layers;
    uint32_t emitted = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint32_t* words = raw.data() + static_cast<size_t>(lane) * kWords;
        emitted += words[0] != 0;
        layers.insert(words[7]); // POS1.z: captured guest layer route, still unvalidated
    }
    std::fprintf(stderr,
                 "[ngg-capture-probe] COMPILE-ONLY MODULE EXECUTION; lanes=%u words/lane=%u "
                 "nonzero-prim=%u distinct-pos1-z=%zu inputs=%s zero-binding=%u "
                 "packed-offsets=%s output=%s\n",
                 lanes, kWords, emitted, layers.size(),
                 zero_inputs ? "zeroed-control" : "captured", zero_binding,
                 packed_offsets_path.empty() ? "none" : packed_offsets_path.c_str(), argv[4]);
    return 0;
}
