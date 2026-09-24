// Execute a compile-only merged-NGG export module against one failed draw's captured buffer
// resources. This reports guest lane outputs, not a raster image or a validated draw ABI.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/compute_runner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <string_view>
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
    // The runner dispatches the single GLCompute entry named "main" at the requested size.
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

int module_marker_count(const std::vector<uint32_t>& words, std::string_view marker) {
    if (words.size() < 5 || words[0] != 0x07230203u) return -1;
    int matches = 0;
    for (size_t at = 5; at < words.size();) {
        const uint32_t count = words[at] >> 16u;
        if (!count || at + count > words.size()) return -1;
        if ((words[at] & 0xffffu) == 330u && count >= 2u) {
            const char* value = reinterpret_cast<const char*>(words.data() + at + 1u);
            const size_t bytes = static_cast<size_t>(count - 1u) * sizeof(uint32_t);
            const void* end = std::memchr(value, '\0', bytes);
            if (end && static_cast<const char*>(end) - value == marker.size() &&
                std::memcmp(value, marker.data(), marker.size()) == 0)
                ++matches;
        }
        at += count;
    }
    return matches;
}

bool exact_native_wave64_marker(const std::vector<uint32_t>& words) {
    return module_marker_count(words, "Prosper.NggProbeExactSubgroup=64") == 1;
}

struct IndexedExportSummary {
    uint32_t indexed_primitives = 0;
    uint32_t consistent_layer_primitives = 0;
    uint32_t mixed_layer_primitives = 0;
    uint32_t out_of_range_primitives = 0;
    uint32_t null_primitives = 0;
    std::set<uint32_t> referenced_vertices;
    std::set<uint32_t> candidate_layers;
};

std::optional<IndexedExportSummary> summarize_indexed_exports(
    const std::vector<uint32_t>& raw, uint32_t lanes, uint32_t words_per_lane) {
    if (words_per_lane < 8u || raw.size() != static_cast<size_t>(lanes) * words_per_lane)
        return std::nullopt;
    IndexedExportSummary result;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint32_t primitive = raw[static_cast<size_t>(lane) * words_per_lane];
        if (!primitive) continue; // no PRIM export in this output slot
        if (primitive & 0x80000000u) {
            ++result.null_primitives;
            continue;
        }
        const std::array<uint32_t, 3> vertices = {
            primitive & 0x3ffu, (primitive >> 10u) & 0x3ffu,
            (primitive >> 20u) & 0x3ffu};
        if (std::any_of(vertices.begin(), vertices.end(),
                        [lanes](uint32_t index) { return index >= lanes; })) {
            ++result.out_of_range_primitives;
            continue;
        }
        ++result.indexed_primitives;
        for (uint32_t index : vertices) result.referenced_vertices.insert(index);
        const uint32_t layer = raw[static_cast<size_t>(vertices[0]) * words_per_lane + 7u];
        const bool same_layer = std::all_of(vertices.begin() + 1, vertices.end(),
            [&](uint32_t index) {
                return raw[static_cast<size_t>(index) * words_per_lane + 7u] == layer;
            });
        if (!same_layer) {
            ++result.mixed_layer_primitives;
            continue;
        }
        ++result.consistent_layer_primitives;
        result.candidate_layers.insert(layer);
    }
    return result;
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
    if (exact_probe_local_size(id_size, 64u) || exact_native_wave64_marker(module))
        return false;
    constexpr char kMarker[] = "Prosper.NggProbeExactSubgroup=64";
    std::vector<uint32_t> payload((sizeof(kMarker) + 3u) / 4u);
    std::memcpy(payload.data(), kMarker, sizeof(kMarker));
    const size_t marker_instruction = module.size();
    module.push_back((static_cast<uint32_t>(payload.size() + 1u) << 16u) | 330u);
    module.insert(module.end(), payload.begin(), payload.end());
    if (!exact_native_wave64_marker(module)) return false;
    constexpr char kTraceMarker[] = "Prosper.NggTraceWord13Hit14";
    std::vector<uint32_t> trace_payload((sizeof(kTraceMarker) + 3u) / 4u);
    std::memcpy(trace_payload.data(), kTraceMarker, sizeof(kTraceMarker));
    module.push_back((static_cast<uint32_t>(trace_payload.size() + 1u) << 16u) | 330u);
    module.insert(module.end(), trace_payload.begin(), trace_payload.end());
    if (module_marker_count(module, kTraceMarker) != 1) return false;
    auto duplicate_trace = module;
    duplicate_trace.push_back((static_cast<uint32_t>(trace_payload.size() + 1u) << 16u) | 330u);
    duplicate_trace.insert(duplicate_trace.end(), trace_payload.begin(), trace_payload.end());
    if (module_marker_count(duplicate_trace, kTraceMarker) != 2) return false;
    module[marker_instruction + 1u] ^= 1u; // mutate 'P', not trailing padding
    if (exact_native_wave64_marker(module)) return false;

    // PRIM lives in one lane, but its three vertex records can live in other lanes. A census
    // over PRIM lanes alone missed Kena's later output vertices and undercounted its layers.
    std::vector<uint32_t> exports(4u * 13u);
    exports[0] = 1u | (2u << 10u) | (3u << 20u);
    for (uint32_t vertex = 1; vertex <= 3; ++vertex)
        exports[static_cast<size_t>(vertex) * 13u + 7u] = 7u;
    const auto good = summarize_indexed_exports(exports, 4u, 13u);
    if (!good || good->indexed_primitives != 1u ||
        good->consistent_layer_primitives != 1u ||
        good->referenced_vertices.size() != 3u ||
        good->candidate_layers != std::set<uint32_t>{7u}) return false;
    exports[3u * 13u + 7u] = 8u;
    const auto mixed = summarize_indexed_exports(exports, 4u, 13u);
    if (!mixed || mixed->indexed_primitives != 1u ||
        mixed->mixed_layer_primitives != 1u || !mixed->candidate_layers.empty()) return false;
    exports[0] = 1u | (2u << 10u) | (5u << 20u);
    const auto outside = summarize_indexed_exports(exports, 4u, 13u);
    if (!outside || outside->out_of_range_primitives != 1u ||
        outside->indexed_primitives != 0u) return false;
    exports[0] |= 0x80000000u;
    const auto null_primitive = summarize_indexed_exports(exports, 4u, 13u);
    if (!null_primitive || null_primitive->null_primitives != 1u ||
        null_primitive->out_of_range_primitives != 0u) return false;
    return !summarize_indexed_exports(std::vector<uint32_t>(4u * 7u), 4u, 7u) &&
           !summarize_indexed_exports(exports, 5u, 13u);
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--selftest") == 0)
        return selftest() ? 0 : 1;
    if (argc < 5 || argc > 7) {
        std::fprintf(stderr,
                     "usage: %s CAPTURE FAILURE COMPILE_ONLY_MODULE.spv OUTPUT.bin "
                     "[--zero-inputs|--zero-binding=2..6] "
                     "[--packed-offsets=FILE|--full-inputs=FILE]\n",
                     argv[0]);
        return 2;
    }
    bool zero_inputs = false;
    uint32_t zero_binding = 0;
    std::string packed_offsets_path;
    std::string full_inputs_path;
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
        } else if (std::strncmp(argv[i], "--full-inputs=", 14) == 0 &&
                   argv[i][14] && full_inputs_path.empty()) {
            full_inputs_path = argv[i] + 14;
        } else {
            std::fprintf(stderr, "ngg_capture_probe: invalid or duplicate option %s\n", argv[i]);
            return 2;
        }
    }
    if (!packed_offsets_path.empty() && !full_inputs_path.empty()) {
        std::fprintf(stderr, "ngg_capture_probe: launch input modes are mutually exclusive\n");
        return 2;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long failure_index = std::strtoul(argv[2], &end, 0);
    if (errno || !end || *end || failure_index > UINT32_MAX) {
        std::fprintf(stderr, "ngg_capture_probe: invalid failure index\n");
        return 2;
    }
    // Captures can exceed a gigabyte. Reject a mismatched module or device before reading one.
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
    const bool full_launch = !full_inputs_path.empty();
    const int trace_markers = module_marker_count(module, "Prosper.NggTraceWord13Hit14");
    const int old_trace_markers = module_marker_count(module, "Prosper.NggTraceWord13");
    if (trace_markers < 0 || trace_markers > 1 || old_trace_markers != 0 ||
        (trace_markers && !full_launch)) {
        std::fprintf(stderr,
                     "ngg_capture_probe: trace marker requires one native full launch\n");
        return 2;
    }
    if (full_launch &&
        (!exact_probe_local_size(module, 256u) || !exact_native_wave64_marker(module))) {
        std::fprintf(stderr,
                     "ngg_capture_probe: full launch requires a 256x1x1 native-Wave64 probe module\n");
        return 2;
    }
    if (!full_launch &&
        (!exact_probe_local_size(module, 64u) || exact_native_wave64_marker(module))) {
        std::fprintf(stderr,
                     "ngg_capture_probe: module must declare one 64x1x1 workgroup; "
                     "four-wave modules require explicit full inputs and native Wave64\n");
        return 2;
    }
    std::vector<uint32_t> full_input_words;
    if (full_launch) {
        constexpr size_t kExpectedBytes = 256u * 10u * sizeof(uint32_t);
        std::ifstream inputs(full_inputs_path, std::ios::binary | std::ios::ate);
        if (!inputs || inputs.tellg() != static_cast<std::streampos>(kExpectedBytes)) {
            std::fprintf(stderr,
                         "ngg_capture_probe: full launch requires exactly %zu bytes\n",
                         kExpectedBytes);
            return 2;
        }
        full_input_words.resize(256u * 10u);
        inputs.seekg(0);
        if (!inputs.read(reinterpret_cast<char*>(full_input_words.data()), kExpectedBytes)) {
            std::fprintf(stderr, "ngg_capture_probe: cannot read full launch inputs\n");
            return 2;
        }
        for (uint32_t wave = 0; wave < 4u; ++wave) {
            const uint32_t s3 = full_input_words[static_cast<size_t>(wave) * 64u * 10u + 9u];
            for (uint32_t lane = 1; lane < 64u; ++lane)
                if (full_input_words[(static_cast<size_t>(wave) * 64u + lane) * 10u + 9u]
                    != s3) {
                    std::fprintf(stderr,
                                 "ngg_capture_probe: s3 must be uniform within guest wave %u\n",
                                 wave);
                    return 2;
                }
        }
    }
    if (full_launch &&
        !prosper::test::default_compute_required_subgroup_supported(64u, 256u)) {
        std::fprintf(stderr,
                     "ngg_capture_probe: full launch requires supported exact 64-lane full subgroups\n");
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
    const uint32_t lanes = full_launch ? 256u : failure.vertex_count * failure.instance_count;
    const uint32_t kWords = trace_markers ? prosper::gpu::kNggTraceProbeWords
                                          : prosper::gpu::kNggExportProbeWords;
    const std::array<std::vector<uint32_t>, 3> extra{
        buffers[2], buffers[3], buffers[4]};
    std::vector<float> launch_inputs(full_launch ? static_cast<size_t>(lanes) * 10u : lanes,
                                     0.0f);
    if (full_launch) {
        static_assert(sizeof(float) == sizeof(uint32_t));
        std::memcpy(launch_inputs.data(), full_input_words.data(),
                    full_input_words.size() * sizeof(uint32_t));
    } else if (!packed_offsets_path.empty()) {
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
        buffers[0], buffers[1], nullptr, full_launch ? 256u : 64u,
        nullptr, nullptr, &extra, full_launch ? 64u : 0u);
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
    const auto indexed = summarize_indexed_exports(raw, lanes, kWords);
    if (!indexed) {
        std::fprintf(stderr, "ngg_capture_probe: invalid export record shape\n");
        return 2;
    }
    const std::string layer_range = indexed->candidate_layers.empty()
        ? "none"
        : std::to_string(*indexed->candidate_layers.begin()) + ".." +
          std::to_string(*indexed->candidate_layers.rbegin());
    std::fprintf(stderr,
                 "[ngg-capture-probe] COMPILE-ONLY MODULE EXECUTION; lanes=%u words/lane=%u "
                 "nonzero-prim=%u distinct-pos1-z=%zu indexed-prims=%u "
                 "referenced-vertices=%zu consistent-layer-prims=%u candidate-layers=%zu "
                 "candidate-layer-range=%s "
                 "mixed-layer-prims=%u out-of-range-prims=%u null-prims=%u "
                 "inputs=%s zero-binding=%u "
                 "packed-offsets=%s full-inputs=%s output=%s\n",
                 lanes, kWords, emitted, layers.size(), indexed->indexed_primitives,
                 indexed->referenced_vertices.size(), indexed->consistent_layer_primitives,
                 indexed->candidate_layers.size(), layer_range.c_str(),
                 indexed->mixed_layer_primitives,
                 indexed->out_of_range_primitives, indexed->null_primitives,
                 zero_inputs ? "zeroed-control" : "captured", zero_binding,
                 packed_offsets_path.empty() ? "none" : packed_offsets_path.c_str(),
                 full_inputs_path.empty() ? "none" : full_inputs_path.c_str(), argv[4]);
    return 0;
}
