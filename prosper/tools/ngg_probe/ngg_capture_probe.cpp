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
    return !eligible_buffer(resource, 0) && !eligible_buffer(resource, 5);
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--selftest") == 0)
        return selftest() ? 0 : 1;
    const bool zero_inputs = argc == 6 && std::strcmp(argv[5], "--zero-inputs") == 0;
    uint32_t zero_binding = 0;
    if (argc == 6 && !zero_inputs &&
        std::strncmp(argv[5], "--zero-binding=", 15) == 0) {
        char* binding_end = nullptr;
        const unsigned long parsed = std::strtoul(argv[5] + 15, &binding_end, 10);
        if (binding_end && !*binding_end && parsed >= 2 && parsed <= 6)
            zero_binding = static_cast<uint32_t>(parsed);
    }
    if (argc != 5 && !zero_inputs && !zero_binding) {
        std::fprintf(stderr,
                     "usage: %s CAPTURE FAILURE COMPILE_ONLY_MODULE.spv OUTPUT.bin "
                     "[--zero-inputs|--zero-binding=2..6]\n",
                     argv[0]);
        return 2;
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
    const uint32_t lanes = failure.vertex_count * failure.instance_count;
    constexpr uint32_t kWords = prosper::gpu::kNggExportProbeWords;
    const std::array<std::vector<uint32_t>, 3> extra{
        buffers[2], buffers[3], buffers[4]};
    const auto result = prosper::test::run_compute(
        module, std::vector<float>(lanes, 0.0f), lanes, lanes * kWords,
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
                 "nonzero-prim=%u distinct-pos1-z=%zu inputs=%s zero-binding=%u output=%s\n",
                 lanes, kWords, emitted, layers.size(),
                 zero_inputs ? "zeroed-control" : "captured", zero_binding, argv[4]);
    return 0;
}
