#include "gpu/gpu_capture.hpp"
#include "gpu/gpu_capture_bundle.hpp"
#include "gpu/gpu_dependency_graph.hpp"
#include "gpu/gpu_execute.hpp"
#include "live_renderer.hpp"
#include "render_runner.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

bool set_environment(const std::string& name, const std::string& value) {
#ifdef _WIN32
    return _putenv_s(name.c_str(), value.c_str()) == 0;
#else
    return setenv(name.c_str(), value.c_str(), 1) == 0;
#endif
}

void usage(const char* argv0) {
    std::fprintf(stderr, "usage: %s [--inspect|--inspect-only|--validate|--graph] "
                         "[--graph-json PATH] [--draw N[:M]] "
                         "[--bundle capture.prgbundle] "
                         "[--prepend producer.prgcap] "
                         "[--dump-resource DRAW:vs|ps:BINDING PATH] [--allow-mismatch] "
                         "[--dump-shader DRAW:vs|fs PATH] "
                         "<capture.prgcap> [output.bmp]\n", argv0);
}

std::vector<uint8_t> execute_frame(const prosper::gpu::GpuReplayFrame& replay,
                                   bool draws_only = false) {
    std::vector<prosper::gpu::SubmitOperation> operations;
    for (const auto& operation : replay.operations)
        if (operation.realized)
            operations.push_back({operation.kind, static_cast<size_t>(operation.source_index),
                                  operation.command_order});
    if (operations.empty() || draws_only)
        return prosper::gpu::render_submit_items(
            replay.items, replay.metadata.width, replay.metadata.height);
    auto result = prosper::gpu::execute_ordered_items(
        operations, replay.items, replay.computes,
        [](const auto& items, uint32_t width, uint32_t height) {
            return prosper::gpu::render_submit_items(items, width, height);
        },
        [](const auto& items) { return prosper::gpu::execute_compute_items(items); },
        replay.metadata.width, replay.metadata.height);
    return std::move(result.pixels);
}

const char* class_name(prosper::gpu::ResourceClass c);

void print_graph(const prosper::gpu::GpuDependencyGraph& graph) {
    std::printf("dependency-graph operations=%zu edges=%zu external-leaves=%zu\n",
                graph.nodes.size(), graph.edges.size(), graph.external_leaves.size());
    for (const auto& node : graph.nodes)
        if (!node.realized)
            std::printf("missing operation=%u kind=%s source=%llu order=%llu\n",
                        node.operation_index,
                        node.kind == prosper::gpu::SubmitOperationKind::Draw ? "draw" : "dispatch",
                        static_cast<unsigned long long>(node.source_index),
                        static_cast<unsigned long long>(node.command_order));
    for (const auto& edge : graph.edges)
        std::printf("edge producer=%u consumer=%u stage=%s binding=%u addr=%016llx bytes=%llu dims=%ux%u\n",
                    edge.producer_operation, edge.consumer_operation, edge.access.stage.c_str(),
                    edge.access.binding, static_cast<unsigned long long>(edge.access.addr),
                    static_cast<unsigned long long>(edge.access.size), edge.access.width, edge.access.height);
    for (const auto& leaf : graph.external_leaves)
        std::printf("external consumers=%zu first=%u future-writer=%lld stage=%s binding=%u class=%s "
                    "addr=%016llx bytes=%llu dims=%ux%u\n",
                    leaf.consumer_operations.size(), leaf.consumer_operations.front(),
                    leaf.first_future_writer == UINT32_MAX ? -1ll : static_cast<long long>(leaf.first_future_writer),
                    leaf.access.stage.c_str(), leaf.access.binding,
                    class_name(leaf.access.resource_class),
                    static_cast<unsigned long long>(leaf.access.addr),
                    static_cast<unsigned long long>(leaf.access.size), leaf.access.width, leaf.access.height);
}

bool write_graph_json(const std::string& path, const prosper::gpu::GpuDependencyGraph& graph) {
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    std::fprintf(file, "{\n  \"operation_count\": %zu,\n  \"nodes\": [\n", graph.nodes.size());
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        std::fprintf(file, "    {\"operation\":%u,\"kind\":\"%s\",\"source\":%llu,"
                           "\"order\":%llu,\"realized\":%s}%s\n",
                     node.operation_index,
                     node.kind == prosper::gpu::SubmitOperationKind::Draw ? "draw" : "dispatch",
                     static_cast<unsigned long long>(node.source_index),
                     static_cast<unsigned long long>(node.command_order),
                     node.realized ? "true" : "false", i + 1 == graph.nodes.size() ? "" : ",");
    }
    std::fprintf(file, "  ],\n  \"edges\": [\n");
    for (size_t i = 0; i < graph.edges.size(); ++i) {
        const auto& edge = graph.edges[i];
        std::fprintf(file, "    {\"producer\":%u,\"consumer\":%u,\"stage\":\"%s\","
                           "\"binding\":%u,\"addr\":%llu,\"bytes\":%llu,\"width\":%u,\"height\":%u}%s\n",
                     edge.producer_operation, edge.consumer_operation, edge.access.stage.c_str(),
                     edge.access.binding, static_cast<unsigned long long>(edge.access.addr),
                     static_cast<unsigned long long>(edge.access.size), edge.access.width,
                     edge.access.height, i + 1 == graph.edges.size() ? "" : ",");
    }
    std::fprintf(file, "  ],\n  \"external_leaves\": [\n");
    for (size_t i = 0; i < graph.external_leaves.size(); ++i) {
        const auto& leaf = graph.external_leaves[i];
        std::fprintf(file, "    {\"consumers\":[");
        for (size_t c = 0; c < leaf.consumer_operations.size(); ++c)
            std::fprintf(file, "%s%u", c ? "," : "", leaf.consumer_operations[c]);
        if (leaf.first_future_writer == UINT32_MAX) std::fprintf(file, "],\"future_writer\":null");
        else std::fprintf(file, "],\"future_writer\":%u", leaf.first_future_writer);
        std::fprintf(file, ",\"stage\":\"%s\",\"binding\":%u,"
                           "\"class\":\"%s\",\"addr\":%llu,\"bytes\":%llu,"
                           "\"width\":%u,\"height\":%u}%s\n",
                     leaf.access.stage.c_str(), leaf.access.binding,
                     class_name(leaf.access.resource_class),
                     static_cast<unsigned long long>(leaf.access.addr),
                     static_cast<unsigned long long>(leaf.access.size), leaf.access.width,
                     leaf.access.height, i + 1 == graph.external_leaves.size() ? "" : ",");
    }
    std::fprintf(file, "  ]\n}\n");
    return std::fclose(file) == 0;
}

const char* class_name(prosper::gpu::ResourceClass c) {
    using RC = prosper::gpu::ResourceClass;
    switch (c) {
    case RC::ConstantBuffer: return "CB"; case RC::VertexBuffer: return "VB";
    case RC::Texture: return "TEX"; case RC::Sampler: return "SAMP"; case RC::StorageImage: return "STORAGE";
    }
    return "?";
}

void inspect_table(const char* stage, const prosper::gpu::ShaderResourceTable* table,
                   const std::vector<prosper::gpu::GpuCaptureRttSeed>& seeds) {
    if (!table) { std::printf("  %s: none\n", stage); return; }
    for (const auto& r : table->resources) {
        size_t nz = 0; uint32_t first = 0;
        if (r.host_data) {
            for (uint64_t i = 0; i < r.host_data_size; ++i) nz += r.host_data[i] != 0;
            if (r.host_data_size >= 4) std::memcpy(&first, r.host_data, 4);
        }
        uint64_t hash = r.host_data ? prosper::gpu::gpu_capture_hash(r.host_data, static_cast<size_t>(r.host_data_size)) : 0;
        const bool temporal_seed = std::any_of(seeds.begin(), seeds.end(), [&](const auto& seed) {
            return seed.guest_addr == r.gpu_addr;
        });
        std::printf("  %s %-7s b=%u addr=%016llx bytes=%llu nz=%zu hash=%016llx first=%08x "
                    "fmt=%u nc=%u stride=%u %ux%u tile=%u addr=%u%u%u swz=%u%u%u%u filt=%u/%u/%u "
                    "srt=%08x sgpr=%08x pc=%08x%s\n",
                    stage, class_name(r.cls), r.binding, static_cast<unsigned long long>(r.gpu_addr),
                    static_cast<unsigned long long>(r.host_data_size), nz, static_cast<unsigned long long>(hash), first,
                    static_cast<unsigned>(r.format), r.num_components, r.stride, r.width, r.height, r.tile_mode,
                    r.addr_uvw[0], r.addr_uvw[1], r.addr_uvw[2],
                    r.swizzle[0], r.swizzle[1], r.swizzle[2], r.swizzle[3],
                    r.mag_filter, r.min_filter, r.mip_filter,
                    r.srt_offset, r.sgpr_base, r.fetch_pc, temporal_seed ? " temporal-RTT-seed" : "");
        if (r.host_data && r.host_data_size >= 16 &&
            (r.cls == prosper::gpu::ResourceClass::ConstantBuffer ||
             r.cls == prosper::gpu::ResourceClass::VertexBuffer)) {
            uint32_t u[4]; float f[4]; std::memcpy(u, r.host_data, sizeof(u)); std::memcpy(f, u, sizeof(f));
            std::printf("    first4 u32=%08x,%08x,%08x,%08x f32=%g,%g,%g,%g\n",
                        u[0], u[1], u[2], u[3], f[0], f[1], f[2], f[3]);
        }
    }
}

void inspect_frame(const prosper::gpu::GpuReplayFrame& replay) {
    for (const auto& seed : replay.rtt_seeds) {
        const uint64_t hash = prosper::gpu::gpu_capture_hash(seed.rgba);
        std::printf("rtt-seed addr=%016llx extent=%ux%u bytes=%zu hash=%016llx\n",
                    static_cast<unsigned long long>(seed.guest_addr), seed.width, seed.height,
                    seed.rgba.size(), static_cast<unsigned long long>(hash));
    }
    for (size_t i = 0; i < replay.items.size(); ++i) {
        const auto& d = replay.items[i];
        std::printf("draw[%zu] target=%016llx extent=%ux%u vcount=%u indices=%zu topo=%u fmt=%u cwm=%x "
                    "depth=%d/%d/%u stencil=%d blend=%d raster=%u/%u/%u viewport=%d %.1f,%.1f %.1fx%.1f "
                    "vs=%zu/%016llx fs=%zu/%016llx\n",
                    i, static_cast<unsigned long long>(d.color0_base), d.color0_width, d.color0_height,
                    d.vertex_count, d.indices.size(),
                    d.ps.topology, d.ps.color0_format, d.ps.color_write_mask, d.ps.depth_test_enable,
                    d.ps.depth_write_enable, d.ps.depth_compare_op, d.ps.stencil_enable, d.ps.blend_enable,
                    d.ps.cull_mode, d.ps.front_face, d.ps.polygon_mode,
                    d.ps.has_viewport, d.ps.viewport_x, d.ps.viewport_y, d.ps.viewport_w, d.ps.viewport_h,
                    d.vs.size(), static_cast<unsigned long long>(prosper::gpu::gpu_capture_hash(
                        reinterpret_cast<const uint8_t*>(d.vs.data()), d.vs.size() * 4)),
                    d.fs.size(), static_cast<unsigned long long>(prosper::gpu::gpu_capture_hash(
                        reinterpret_cast<const uint8_t*>(d.fs.data()), d.fs.size() * 4)));
        std::printf("  stencil clear=%u cmp=%u/%u fail=%u/%u pass=%u/%u zfail=%u/%u "
                    "ref=%u/%u opval=%u/%u cmask=%02x/%02x wmask=%02x/%02x\n",
                    d.ps.stencil_clear_value, d.ps.stencil_compare_op[0], d.ps.stencil_compare_op[1],
                    d.ps.stencil_fail_op[0], d.ps.stencil_fail_op[1], d.ps.stencil_pass_op[0],
                    d.ps.stencil_pass_op[1], d.ps.stencil_depth_fail_op[0], d.ps.stencil_depth_fail_op[1],
                    d.ps.stencil_ref[0], d.ps.stencil_ref[1], d.ps.stencil_op_val[0], d.ps.stencil_op_val[1],
                    d.ps.stencil_compare_mask[0], d.ps.stencil_compare_mask[1],
                    d.ps.stencil_write_mask[0], d.ps.stencil_write_mask[1]);
        std::printf("  ds rc=%08x clear=%d/%d programmed=%d/%d base z=%016llx/%016llx s=%016llx/%016llx "
                    "rawops=%u,%u,%u/%u,%u,%u shaderctl=%08x stexport=%d/%d\n",
                    d.ps.db_render_control, d.ps.depth_clear_enable, d.ps.stencil_clear_enable,
                    d.ps.has_depth_clear, d.ps.has_stencil_clear,
                    static_cast<unsigned long long>(d.ps.depth_read_base),
                    static_cast<unsigned long long>(d.ps.depth_write_base),
                    static_cast<unsigned long long>(d.ps.stencil_read_base),
                    static_cast<unsigned long long>(d.ps.stencil_write_base),
                    d.ps.raw_stencil_op[0][0], d.ps.raw_stencil_op[0][1], d.ps.raw_stencil_op[0][2],
                    d.ps.raw_stencil_op[1][0], d.ps.raw_stencil_op[1][1], d.ps.raw_stencil_op[1][2],
                    d.ps.db_shader_control, d.ps.stencil_test_val_export_enable,
                    d.ps.stencil_op_val_export_enable);
        inspect_table("VS", d.vrt.get(), replay.rtt_seeds);
        inspect_table("PS", d.prt.get(), replay.rtt_seeds);
    }
    for (size_t i = 0; i < replay.computes.size(); ++i) {
        const auto& c = replay.computes[i];
        std::printf("compute[%zu] source=%llu order=%llu code=%016llx groups=%ux%ux%u local=%ux%ux%u "
                    "shader=%zu/%016llx\n",
                    i, static_cast<unsigned long long>(c.dispatch_index),
                    static_cast<unsigned long long>(c.command_order),
                    static_cast<unsigned long long>(c.code_addr), c.launch.groups_x,
                    c.launch.groups_y, c.launch.groups_z, c.launch.local_x, c.launch.local_y,
                    c.launch.local_z, c.spirv.size(),
                    static_cast<unsigned long long>(prosper::gpu::gpu_capture_hash(
                        reinterpret_cast<const uint8_t*>(c.spirv.data()), c.spirv.size() * 4)));
        inspect_table("CS", c.resources.get(), replay.rtt_seeds);
    }
    for (size_t i = 0; i < replay.operations.size(); ++i) {
        const auto& operation = replay.operations[i];
        std::printf("operation[%zu] %s source=%llu order=%llu realized=%s\n", i,
                    operation.kind == prosper::gpu::SubmitOperationKind::Draw ? "draw" : "dispatch",
                    static_cast<unsigned long long>(operation.source_index),
                    static_cast<unsigned long long>(operation.command_order),
                    operation.realized ? "yes" : "no");
    }
}

bool validate_frame(const prosper::gpu::GpuReplayFrame& replay) {
    bool valid = true;
    for (size_t i = 0; i < replay.items.size(); ++i) {
        const auto& d = replay.items[i];
        struct StageInput {
            const char* name;
            const std::vector<uint32_t>* spirv;
            const prosper::gpu::ShaderResourceTable* table;
            uint32_t set;
            prosper::gpu::SpirvShaderStage stage;
        } stages[] = {
            {"VS", &d.vs, d.vrt.get(), 0, prosper::gpu::SpirvShaderStage::Vertex},
            {"PS", &d.fs, d.prt.get(), 1, prosper::gpu::SpirvShaderStage::Fragment},
        };
        for (const auto& s : stages) {
            auto report = prosper::gpu::validate_spirv_descriptor_interface(
                *s.spirv, s.table, s.set, s.stage, true);
            std::printf("draw[%zu] %s descriptors=%zu runtime=%zu result=%s\n", i, s.name,
                        report.descriptors.size(), s.table ? s.table->resources.size() : 0,
                        report.ok() ? "accept" : "reject");
            for (const auto& binding : report.descriptors)
                std::printf("  set=%u binding=%u type=%s required=%llu%s\n",
                            binding.set, binding.binding,
                            prosper::gpu::spirv_descriptor_kind_name(binding.kind),
                            static_cast<unsigned long long>(binding.required_bytes),
                            binding.dynamic_access ? "+dynamic" : "");
            for (const auto& issue : report.issues) {
                std::printf("  %s %s set=%u binding=%u expected=%s actual=%s required=%llu available=%llu\n",
                            issue.error ? "ERROR" : "warn",
                            prosper::gpu::descriptor_issue_name(issue.code), issue.set, issue.binding,
                            prosper::gpu::spirv_descriptor_kind_name(issue.expected),
                            prosper::gpu::spirv_descriptor_kind_name(issue.actual),
                            static_cast<unsigned long long>(issue.required_bytes),
                            static_cast<unsigned long long>(issue.available_bytes));
                valid &= !issue.error;
            }
        }
    }
    return valid;
}

struct BundleTarget {
    uint64_t addr = 0;
    uint64_t size = 0;
    uint64_t submit = 0;
};

uint64_t range_end(uint64_t addr, uint64_t size) {
    return size > UINT64_MAX - addr ? UINT64_MAX : addr + size;
}

bool ranges_overlap(uint64_t a, uint64_t as, uint64_t b, uint64_t bs) {
    return a < range_end(b, bs) && b < range_end(a, as);
}

int replay_bundle(const std::string& path, const char* output_path) {
    prosper::gpu::GpuCaptureBundle bundle;
    std::string error;
    if (!prosper::gpu::read_gpu_capture_bundle(path, bundle, error)) {
        std::fprintf(stderr, "gpu_replay: cannot read bundle: %s\n", error.c_str()); return 2;
    }
    const uint64_t unique_bytes = prosper::gpu::gpu_capture_bundle_unique_bytes(bundle);
    std::fprintf(stderr, "[gpureplay] bundle submits=%zu logical=%llu unique=%llu ratio=%.3f chunks=%zu\n",
                 bundle.submits.size(), static_cast<unsigned long long>(bundle.logical_bytes),
                 static_cast<unsigned long long>(unique_bytes),
                 bundle.logical_bytes ? static_cast<double>(unique_bytes) / bundle.logical_bytes : 0.0,
                 bundle.chunks.size());

    prosper::gpu::GpuCaptureFile final_capture;
    if (!prosper::gpu::materialize_gpu_capture_bundle_submit(
            bundle, bundle.submits.size() - 1, final_capture, error)) {
        std::fprintf(stderr, "gpu_replay: cannot reconstruct final submit: %s\n", error.c_str()); return 2;
    }
    const prosper::gpu::GpuCaptureMetadata final_metadata = final_capture.metadata;
    const bool expected_output_valid = final_capture.expected_output_valid;
    const uint64_t expected_output_bytes = final_capture.expected_output_bytes;
    const uint64_t expected_output_hash = final_capture.expected_output_hash;
    for (const auto& [name, value] : final_capture.metadata.renderer_env)
        if (!set_environment(name, value)) {
            std::fprintf(stderr, "gpu_replay: cannot set %s\n", name.c_str()); return 2;
        }
    set_environment("PROSPER_RENDER_FIRST", "0");
    set_environment("PROSPER_RENDER_LAST", "2147483647");
    set_environment("PROSPER_GPU_REPLAY_RTT_SEEDS", "1");
    prosper::frontend::register_live_renderer(".", false);
    final_capture = {};

    std::vector<BundleTarget> prior_targets;
    std::vector<uint8_t> final_pixels;
    uint64_t temporal_resolved = 0, temporal_seeded = 0, temporal_bounded = 0, temporal_unresolved = 0;
    for (size_t i = 0; i < bundle.submits.size(); ++i) {
        prosper::gpu::GpuCaptureFile capture;
        if (!prosper::gpu::materialize_gpu_capture_bundle_submit(bundle, i, capture, error)) {
            std::fprintf(stderr, "gpu_replay: cannot reconstruct bundle submit %zu: %s\n",
                         i, error.c_str()); return 2;
        }
        prosper::gpu::GpuReplayFrame replay;
        if (!prosper::gpu::materialize_gpu_replay(capture, replay, error)) {
            std::fprintf(stderr, "gpu_replay: cannot materialize bundle submit %llu: %s\n",
                         static_cast<unsigned long long>(capture.metadata.submit_index), error.c_str());
            return 2;
        }
        prosper::gpu::GpuDependencyGraph graph;
        if (!prosper::gpu::build_gpu_dependency_graph(replay, graph, error)) {
            std::fprintf(stderr, "gpu_replay: cannot graph bundle submit %llu: %s\n",
                         static_cast<unsigned long long>(capture.metadata.submit_index), error.c_str());
            return 2;
        }
        for (const auto& leaf : graph.external_leaves) {
            if (leaf.first_future_writer == UINT32_MAX ||
                (leaf.access.resource_class != prosper::gpu::ResourceClass::Texture &&
                 leaf.access.resource_class != prosper::gpu::ResourceClass::StorageImage)) continue;
            auto seed = std::find_if(replay.rtt_seeds.begin(), replay.rtt_seeds.end(),
                [&](const auto& candidate) { return candidate.guest_addr == leaf.access.addr; });
            auto producer = std::find_if(prior_targets.rbegin(), prior_targets.rend(),
                [&](const auto& target) {
                    return ranges_overlap(leaf.access.addr, leaf.access.size, target.addr, target.size);
                });
            const char* stop = i == 0 ? "configured-bound" : "unresolved-producer";
            uint64_t producer_submit = 0;
            if (producer != prior_targets.rend()) {
                stop = "included-producer"; producer_submit = producer->submit; ++temporal_resolved;
            } else if (seed != replay.rtt_seeds.end()) {
                stop = "initialized-seed"; ++temporal_seeded;
            } else {
                if (i == 0) ++temporal_bounded;
                else ++temporal_unresolved;
            }
            std::fprintf(stderr, "[gpureplay] frontier submit=%llu op=%u addr=%016llx dims=%ux%u "
                                 "stop=%s producer=%llu\n",
                         static_cast<unsigned long long>(capture.metadata.submit_index),
                         leaf.consumer_operations.front(),
                         static_cast<unsigned long long>(leaf.access.addr),
                         leaf.access.width, leaf.access.height, stop,
                         static_cast<unsigned long long>(producer_submit));
        }

        std::vector<prosper::gpu::GpuCaptureRttSeed> seeds;
        for (const auto& seed : replay.rtt_seeds)
            if (std::none_of(prior_targets.begin(), prior_targets.end(), [&](const auto& target) {
                    return target.addr == seed.guest_addr;
                })) seeds.push_back(seed);
        if (!prosper::gpu::restore_gpu_replay_rtt_seeds(seeds, error)) {
            std::fprintf(stderr, "gpu_replay: cannot restore bundle RTT seeds: %s\n", error.c_str());
            return 2;
        }
        final_pixels = execute_frame(replay);
        for (const auto& draw : replay.items)
            if (draw.color0_base && draw.color0_width && draw.color0_height)
                prior_targets.push_back({draw.color0_base,
                    static_cast<uint64_t>(draw.color0_width) * draw.color0_height * 4,
                    capture.metadata.submit_index});
        std::fprintf(stderr, "[gpureplay] bundle-submit=%llu operations=%zu output_bytes=%zu hash=%016llx\n",
                     static_cast<unsigned long long>(capture.metadata.submit_index),
                     replay.operations.size(), final_pixels.size(),
                     static_cast<unsigned long long>(prosper::gpu::gpu_capture_hash(final_pixels)));
    }
    std::fprintf(stderr, "[gpureplay] closure temporal-resolved=%llu seeded=%llu bounded=%llu "
                         "unresolved=%llu\n",
                 static_cast<unsigned long long>(temporal_resolved),
                 static_cast<unsigned long long>(temporal_seeded),
                 static_cast<unsigned long long>(temporal_bounded),
                 static_cast<unsigned long long>(temporal_unresolved));
    if (output_path && !final_pixels.empty() &&
        !prosper::test::dump_bmp(output_path, final_pixels,
                                 final_metadata.width, final_metadata.height)) {
        std::fprintf(stderr, "gpu_replay: cannot write %s\n", output_path); return 2;
    }
    if (expected_output_valid &&
        (final_pixels.size() != expected_output_bytes ||
         prosper::gpu::gpu_capture_hash(final_pixels) != expected_output_hash)) {
        std::fprintf(stderr, "gpu_replay: bundle output mismatch\n"); return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    bool inspect = false, inspect_only = false, validate_only = false, allow_mismatch = false;
    bool graph_only = false;
    int draw_first = -1, draw_last = -1;
    std::string dump_spec, dump_path, shader_spec, shader_path, graph_json_path, prepend_path, bundle_path;
    std::vector<const char*> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--inspect") inspect = true;
        else if (std::string(argv[i]) == "--inspect-only") inspect = inspect_only = true;
        else if (std::string(argv[i]) == "--validate") validate_only = true;
        else if (std::string(argv[i]) == "--graph") graph_only = true;
        else if (std::string(argv[i]) == "--graph-json" && i + 1 < argc) {
            graph_only = true; graph_json_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--allow-mismatch") allow_mismatch = true;
        else if (std::string(argv[i]) == "--bundle" && i + 1 < argc) bundle_path = argv[++i];
        else if (std::string(argv[i]) == "--prepend" && i + 1 < argc) prepend_path = argv[++i];
        else if (std::string(argv[i]) == "--draw" && i + 1 < argc) {
            std::string range = argv[++i]; size_t colon = range.find(':');
            draw_first = std::atoi(range.c_str());
            draw_last = colon == std::string::npos ? draw_first : std::atoi(range.c_str() + colon + 1);
        }
        else if (std::string(argv[i]) == "--dump-resource" && i + 2 < argc) {
            dump_spec = argv[++i]; dump_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--dump-shader" && i + 2 < argc) {
            shader_spec = argv[++i]; shader_path = argv[++i];
        }
        else positional.push_back(argv[i]);
    }
    if (!bundle_path.empty()) {
        if (positional.size() > 1 || inspect || inspect_only || validate_only || graph_only ||
            draw_first >= 0 || !dump_spec.empty() || !shader_spec.empty() || !prepend_path.empty()) {
            usage(argv[0]); return 2;
        }
        return replay_bundle(bundle_path, positional.empty() ? nullptr : positional[0]);
    }
    if (positional.empty() || positional.size() > 2) { usage(argv[0]); return 2; }
    prosper::gpu::GpuCaptureFile capture; std::string error;
    if (!prosper::gpu::read_gpu_capture(positional[0], capture, error)) {
        std::fprintf(stderr, "gpu_replay: %s: %s\n", positional[0], error.c_str()); return 2;
    }
    for (const auto& [name, value] : capture.metadata.renderer_env) {
        if (!set_environment(name, value)) {
            std::fprintf(stderr, "gpu_replay: cannot set %s\n", name.c_str()); return 2;
        }
    }
    // A replay is already at its target submit; live-run windowing must never skip it.
    set_environment("PROSPER_RENDER_FIRST", "0"); set_environment("PROSPER_RENDER_LAST", "2147483647");
    prosper::gpu::GpuReplayFrame replay;
    if (!prosper::gpu::materialize_gpu_replay(capture, replay, error)) {
        std::fprintf(stderr, "gpu_replay: cannot materialize: %s\n", error.c_str()); return 2;
    }
    prosper::gpu::GpuCaptureFile prepend_capture;
    prosper::gpu::GpuReplayFrame prepend;
    if (!prepend_path.empty() &&
        (!prosper::gpu::read_gpu_capture(prepend_path, prepend_capture, error) ||
         !prosper::gpu::materialize_gpu_replay(prepend_capture, prepend, error))) {
        std::fprintf(stderr, "gpu_replay: cannot load predecessor %s: %s\n",
                     prepend_path.c_str(), error.c_str()); return 2;
    }
    if (!prepend_path.empty() && prepend.metadata.submit_index >= replay.metadata.submit_index) {
        std::fprintf(stderr, "gpu_replay: predecessor submit must be earlier than consumer submit\n");
        return 2;
    }
    if (graph_only) {
        prosper::gpu::GpuDependencyGraph graph;
        if (!prosper::gpu::build_gpu_dependency_graph(replay, graph, error)) {
            std::fprintf(stderr, "gpu_replay: cannot build dependency graph: %s\n", error.c_str()); return 2;
        }
        if (graph_json_path.empty()) print_graph(graph);
        else if (!write_graph_json(graph_json_path, graph)) {
            std::fprintf(stderr, "gpu_replay: cannot write graph JSON %s\n", graph_json_path.c_str()); return 2;
        }
        return 0;
    }
    if (!dump_spec.empty()) {
        size_t c1 = dump_spec.find(':'), c2 = c1 == std::string::npos ? c1 : dump_spec.find(':', c1 + 1);
        if (c1 == std::string::npos || c2 == std::string::npos) { usage(argv[0]); return 2; }
        int di = std::atoi(dump_spec.c_str()); std::string stage = dump_spec.substr(c1 + 1, c2 - c1 - 1);
        int binding = std::atoi(dump_spec.c_str() + c2 + 1);
        if (di < 0 || static_cast<size_t>(di) >= replay.items.size() || (stage != "vs" && stage != "ps")) {
            std::fprintf(stderr, "gpu_replay: invalid resource selector %s\n", dump_spec.c_str()); return 2;
        }
        const auto* table = stage == "vs" ? replay.items[di].vrt.get() : replay.items[di].prt.get();
        const prosper::gpu::ShaderResource* found = nullptr;
        if (table) for (const auto& resource : table->resources)
            if (static_cast<int>(resource.binding) == binding) { found = &resource; break; }
        if (!found || !found->host_data) {
            std::fprintf(stderr, "gpu_replay: resource %s not found or has no captured bytes\n", dump_spec.c_str()); return 2;
        }
        FILE* f = std::fopen(dump_path.c_str(), "wb");
        if (!f || std::fwrite(found->host_data, 1, static_cast<size_t>(found->host_data_size), f) != found->host_data_size) {
            if (f) std::fclose(f); std::fprintf(stderr, "gpu_replay: cannot write %s\n", dump_path.c_str()); return 2;
        }
        std::fclose(f);
        std::fprintf(stderr, "[gpureplay] dumped %s (%llu bytes) to %s\n", dump_spec.c_str(),
                     static_cast<unsigned long long>(found->host_data_size), dump_path.c_str());
    }
    if (!shader_spec.empty()) {
        size_t colon = shader_spec.find(':');
        int di = std::atoi(shader_spec.c_str()); std::string stage =
            colon == std::string::npos ? std::string{} : shader_spec.substr(colon + 1);
        if (colon == std::string::npos || di < 0 || static_cast<size_t>(di) >= replay.items.size() ||
            (stage != "vs" && stage != "fs")) {
            std::fprintf(stderr, "gpu_replay: invalid shader selector %s\n", shader_spec.c_str()); return 2;
        }
        const auto& words = stage == "vs" ? replay.items[di].vs : replay.items[di].fs;
        FILE* f = std::fopen(shader_path.c_str(), "wb"); size_t bytes = words.size() * sizeof(uint32_t);
        if (!f || std::fwrite(words.data(), 1, bytes, f) != bytes) {
            if (f) std::fclose(f); std::fprintf(stderr, "gpu_replay: cannot write %s\n", shader_path.c_str()); return 2;
        }
        std::fclose(f);
        std::fprintf(stderr, "[gpureplay] dumped %s (%zu bytes) to %s\n", shader_spec.c_str(), bytes,
                     shader_path.c_str());
    }
    if (draw_first >= 0) {
        if (draw_last < draw_first || static_cast<size_t>(draw_last) >= replay.items.size()) {
            std::fprintf(stderr, "gpu_replay: draw range %d:%d is out of range\n", draw_first, draw_last); return 2;
        }
        std::vector<prosper::gpu::DrawItem> selected;
        for (int i = draw_first; i <= draw_last; ++i) selected.push_back(std::move(replay.items[i]));
        replay.items = std::move(selected); allow_mismatch = true;
        std::fprintf(stderr, "[gpureplay] selected original draws %d:%d\n", draw_first, draw_last);
    }
    const auto& m = replay.metadata;
    std::fprintf(stderr, "[gpureplay] rev=%s title=%s submit=%llu %ux%u draws=%zu computes=%zu "
                         "operations=%zu shaders=%zu blobs=%zu RTT-seeds=%zu oracle=%s\n",
                 m.revision.c_str(), m.title_id.c_str(), static_cast<unsigned long long>(m.submit_index),
                 m.width, m.height, replay.items.size(), replay.computes.size(), replay.operations.size(),
                 capture.shader_versions.size(), replay.blobs.size(), replay.rtt_seeds.size(),
                 replay.expected_output_valid ? "yes" : "no");
    if (inspect) inspect_frame(replay);
    if (validate_only) return validate_frame(replay) ? 0 : 1;
    if (inspect_only) return 0;
    if (!replay.rtt_seeds.empty() || !prepend.rtt_seeds.empty())
        set_environment("PROSPER_GPU_REPLAY_RTT_SEEDS", "1");
    prosper::frontend::register_live_renderer(".", false);
    std::unordered_set<uint64_t> predecessor_targets;
    if (!prepend_path.empty()) {
        if (!prosper::gpu::restore_gpu_replay_rtt_seeds(prepend.rtt_seeds, error)) {
            std::fprintf(stderr, "gpu_replay: cannot restore predecessor RTT seeds: %s\n",
                         error.c_str());
            return 2;
        }
        std::vector<uint8_t> predecessor_pixels = execute_frame(prepend);
        for (const auto& draw : prepend.items)
            if (draw.color0_base) predecessor_targets.insert(draw.color0_base);
        std::fprintf(stderr, "[gpureplay] prepended submit=%llu operations=%zu targets=%zu "
                             "output_bytes=%zu hash=%016llx\n",
                     static_cast<unsigned long long>(prepend.metadata.submit_index),
                     prepend.operations.size(), predecessor_targets.size(), predecessor_pixels.size(),
                     static_cast<unsigned long long>(prosper::gpu::gpu_capture_hash(predecessor_pixels)));
    }
    std::vector<prosper::gpu::GpuCaptureRttSeed> consumer_seeds;
    for (const auto& seed : replay.rtt_seeds)
        if (!predecessor_targets.count(seed.guest_addr)) consumer_seeds.push_back(seed);
    if (!prosper::gpu::restore_gpu_replay_rtt_seeds(consumer_seeds, error)) {
        std::fprintf(stderr, "gpu_replay: cannot restore RTT seeds: %s\n", error.c_str()); return 2;
    }
    std::vector<uint8_t> pixels = execute_frame(replay, draw_first >= 0);
    uint64_t hash = prosper::gpu::gpu_capture_hash(pixels);
    std::fprintf(stderr, "[gpureplay] output_bytes=%zu hash=%016llx expected_bytes=%llu expected_hash=%016llx\n",
                 pixels.size(), static_cast<unsigned long long>(hash),
                 static_cast<unsigned long long>(replay.expected_output_bytes),
                 static_cast<unsigned long long>(replay.expected_output_hash));
    if (positional.size() == 2 && !pixels.empty() && !prosper::test::dump_bmp(positional[1], pixels, m.width, m.height)) {
        std::fprintf(stderr, "gpu_replay: cannot write %s\n", positional[1]); return 2;
    }
    if (!allow_mismatch && replay.expected_output_valid &&
        (pixels.size() != replay.expected_output_bytes || hash != replay.expected_output_hash)) {
        std::fprintf(stderr, "gpu_replay: output mismatch\n"); return 1;
    }
    return 0;
}
