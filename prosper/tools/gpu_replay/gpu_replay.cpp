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
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
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
                         "[--graph-json PATH] [--draw N[:M]] [--draw-with-compute-prefix] "
                         "[--through-operation N] [--warmup-repeats N] "
                         "[--bundle capture.prgbundle] [--bundle-tail N] [--bundle-compact PATH] "
                         "[--bundle-intermediate-through-target WxH] "
                         "[--bundle-final-capsule PATH] "
                         "[--bundle-extract-submit N PATH] "
                         "[--bundle-ds-summary] "
                         "[--bundle-find-ds ADDR] "
                         "[--bundle-zero-boundary] "
                         "[--prepend producer.prgcap] "
                         "[--dump-resource DRAW:vs|ps:BINDING PATH] [--allow-mismatch] "
                         "[--dump-shader DRAW:vs|fs PATH] [--dump-compute N PATH] "
                         "[--dump-failed-shader FAILURE:STAGE PATH] "
                         "[--dump-compute-resource N:BINDING PATH] "
                         "[--legacy-htile-before-stencil] "
                         "<capture.prgcap> [output.bmp]\n", argv0);
}

std::vector<uint8_t> execute_frame(const prosper::gpu::GpuReplayFrame& replay,
                                   bool draws_only = false,
                                   size_t operation_limit = SIZE_MAX) {
    std::vector<prosper::gpu::SubmitOperation> operations;
    const size_t count = std::min(operation_limit, replay.operations.size());
    for (size_t i = 0; i < count; ++i) {
        const auto& operation = replay.operations[i];
        if (operation.realized)
            operations.push_back({operation.kind, static_cast<size_t>(operation.source_index),
                                  operation.command_order});
    }
    if (draws_only || replay.operations.empty())
        return prosper::gpu::render_submit_items(
            replay.items, replay.metadata.width, replay.metadata.height);
    if (operations.empty()) return {};
    auto result = prosper::gpu::execute_ordered_items(
        operations, replay.items, replay.computes,
        [](const auto& items, uint32_t width, uint32_t height) {
            return prosper::gpu::render_submit_items(items, width, height);
        },
        [](const auto& items) { return prosper::gpu::execute_compute_items(items); },
        replay.metadata.width, replay.metadata.height);
    return result.frame.bytes();
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
        std::printf("  %s %-7s b=%u addr=%016llx declared=%u footprint=%llu captured=%llu "
                    "nz=%zu hash=%016llx first=%08x "
                    "fmt=%u nc=%u stride=%u %ux%ux%u tile=%u addr=%u%u%u swz=%u%u%u%u filt=%u/%u/%u "
                    "srt=%08x sgpr=%08x pc=%08x%s\n",
                    stage, class_name(r.cls), r.binding, static_cast<unsigned long long>(r.gpu_addr),
                    r.size,
                    static_cast<unsigned long long>(prosper::gpu::gpu_capture_resource_footprint(r)),
                    static_cast<unsigned long long>(r.host_data_size), nz,
                    static_cast<unsigned long long>(hash), first,
                    static_cast<unsigned>(r.format), r.num_components, r.stride,
                    r.width, r.height, r.depth, r.tile_mode,
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
    for (const auto& seed : replay.ds_seeds) {
        const uint64_t depth_hash = prosper::gpu::gpu_capture_hash(seed.depth);
        const uint64_t stencil_hash = prosper::gpu::gpu_capture_hash(seed.stencil);
        std::printf("ds-seed base-z=%016llx/%016llx base-s=%016llx/%016llx htile=%016llx "
                    "extent=%ux%u format=%s valid=%d/%d depth=%zu/%016llx stencil=%zu/%016llx\n",
                    static_cast<unsigned long long>(seed.depth_read_base),
                    static_cast<unsigned long long>(seed.depth_write_base),
                    static_cast<unsigned long long>(seed.stencil_read_base),
                    static_cast<unsigned long long>(seed.stencil_write_base),
                    static_cast<unsigned long long>(seed.htile_data_base), seed.width, seed.height,
                    seed.format == prosper::gpu::GpuCaptureDsFormat::D32FloatS8 ? "D32S8" : "D32",
                    seed.depth_valid, seed.stencil_valid, seed.depth.size(),
                    static_cast<unsigned long long>(depth_hash), seed.stencil.size(),
                    static_cast<unsigned long long>(stencil_hash));
    }
    for (size_t i = 0; i < replay.items.size(); ++i) {
        const auto& d = replay.items[i];
        std::printf("draw[%zu] source=%llu target=%016llx extent=%ux%u vcount=%u indices=%zu topo=%u fmt=%u cwm=%x "
                    "depth=%d/%d/%u stencil=%d blend=%d raster=%u/%u/%u viewport=%d %.1f,%.1f %.1fx%.1f "
                    "vs=%zu/%016llx fs=%zu/%016llx\n",
                    i, static_cast<unsigned long long>(d.draw_index),
                    static_cast<unsigned long long>(d.color0_base), d.color0_width, d.color0_height,
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
        std::printf("  ds-surface view=%08x override=%08x/%08x htile=%016llx hsurface=%08x "
                    "size-xy=%08x depth-size=%08x slice=%08x info=%08x z=%08x s=%08x "
                    "dfsm=%08x rmi=%08x\n",
                    d.ps.db_depth_view, d.ps.db_render_override, d.ps.db_render_override2,
                    static_cast<unsigned long long>(d.ps.htile_data_base), d.ps.db_htile_surface,
                    d.ps.db_depth_size_xy, d.ps.db_depth_size, d.ps.db_depth_slice,
                    d.ps.db_depth_info, d.ps.db_z_info, d.ps.db_stencil_info,
                    d.ps.db_dfsm_control, d.ps.db_rmi_l2_cache_control);
        if (!d.indices.empty()) {
            const size_t preview_count = std::min<size_t>(d.indices.size(), 16);
            std::printf("  indices first[%zu/%zu]=", preview_count, d.indices.size());
            for (size_t index = 0; index < preview_count; ++index)
                std::printf("%s%u", index ? "," : "", d.indices[index]);
            std::printf("\n");
        }
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
    if (!replay.failure_diagnostics_available) {
        std::printf("failure-diagnostics: unavailable (capture predates v7)\n");
    } else if (replay.failure_diagnostics.empty()) {
        std::printf("failure-diagnostics: none\n");
    }
    for (size_t i = 0; i < replay.failure_diagnostics.size(); ++i) {
        const auto& failure = replay.failure_diagnostics[i];
        std::printf("failure[%zu] %s source=%llu order=%llu reason=%s stages=%zu\n", i,
                    failure.kind == prosper::gpu::SubmitOperationKind::Draw ? "draw" : "dispatch",
                    static_cast<unsigned long long>(failure.source_index),
                    static_cast<unsigned long long>(failure.command_order),
                    prosper::gpu::realization_failure_reason_name(failure.reason),
                    failure.stages.size());
        if (failure.kind == prosper::gpu::SubmitOperationKind::Draw) {
            std::printf("  target=%016llx extent=%ux%u vertices=%u pipeline=%s",
                        static_cast<unsigned long long>(failure.color0_base),
                        failure.color0_width, failure.color0_height, failure.vertex_count,
                        failure.pipeline_present ? "yes" : "no");
            if (failure.pipeline_present)
                std::printf(" fmt=%u cwm=%x depth=%d/%d/%u stencil=%d blend=%d",
                            failure.pipeline.color0_format, failure.pipeline.color_write_mask,
                            failure.pipeline.depth_test_enable, failure.pipeline.depth_write_enable,
                            failure.pipeline.depth_compare_op, failure.pipeline.stencil_enable,
                            failure.pipeline.blend_enable);
            std::printf("\n");
        } else {
            const auto& launch = failure.compute_launch;
            std::printf("  threads=%ux%ux%u local=%ux%ux%u groups=%ux%ux%u\n",
                        launch.threads_x, launch.threads_y, launch.threads_z,
                        launch.local_x, launch.local_y, launch.local_z,
                        launch.groups_x, launch.groups_y, launch.groups_z);
        }
        for (const auto& stage : failure.stages) {
            const prosper::gpu::GpuCaptureRawShaderVersion* raw = nullptr;
            if (stage.raw_shader_index < replay.raw_shader_versions.size())
                raw = &replay.raw_shader_versions[stage.raw_shader_index];
            std::printf("  %s program=%016llx raw=%s",
                        prosper::gpu::shader_program_stage_name(stage.stage),
                        static_cast<unsigned long long>(stage.program_addr), raw ? "yes" : "no");
            if (raw)
                std::printf(" words=%zu bytes=%zu hash=%016llx endpgm=%s",
                            raw->words.size(), raw->words.size() * sizeof(uint32_t),
                            static_cast<unsigned long long>(raw->content_hash),
                            raw->has_endpgm ? "yes" : "no");
            std::printf(" recompiled=%s resources=%s/%u descriptors=%u",
                        stage.recompiled ? "yes" : "no",
                        stage.resource_table_present ? "present" : "absent",
                        stage.resource_count, stage.descriptor_issue_count);
            if (stage.first_descriptor_issue != 0xFFFFFFFFu)
                std::printf(" first-descriptor=%s", prosper::gpu::descriptor_issue_name(
                    static_cast<prosper::gpu::DescriptorIssueCode>(stage.first_descriptor_issue)));
            std::printf("\n    coverage total=%u alu=%u exports=%u table-dependent=%u unsupported=%u",
                        stage.coverage.total, stage.coverage.alu, stage.coverage.exports,
                        stage.coverage.table_dependent, stage.coverage.unsupported);
            if (stage.coverage.first_bad_fmt >= 0)
                std::printf(" first-reject pc=%u fmt=%d op=0x%x", stage.coverage.first_bad_pc,
                            stage.coverage.first_bad_fmt, stage.coverage.first_bad_op);
            std::printf("\n");
        }
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

struct BundleDsIdentity {
    uint64_t depth_read = 0, depth_write = 0;
    uint64_t stencil_read = 0, stencil_write = 0;

    bool operator<(const BundleDsIdentity& other) const {
        return std::tie(depth_read, depth_write, stencil_read, stencil_write) <
               std::tie(other.depth_read, other.depth_write,
                        other.stencil_read, other.stencil_write);
    }
    bool operator==(const BundleDsIdentity& other) const {
        return depth_read == other.depth_read && depth_write == other.depth_write &&
               stencil_read == other.stencil_read && stencil_write == other.stencil_write;
    }
};

BundleDsIdentity ds_identity(const prosper::gpu::ResolvedPipelineState& ps) {
    return {ps.depth_read_base, ps.depth_write_base,
            ps.stencil_read_base, ps.stencil_write_base};
}

bool uses_depth_stencil(const prosper::gpu::ResolvedPipelineState& ps) {
    return ps.depth_test_enable || ps.depth_write_enable || ps.stencil_enable ||
           ps.depth_clear_enable || ps.stencil_clear_enable;
}

struct BundleDsProgramming {
    uint32_t view = 0, render_override = 0, render_override2 = 0;
    uint64_t htile = 0;
    uint32_t size_xy = 0, dfsm = 0, depth_info = 0, z_info = 0, stencil_info = 0;
    uint32_t depth_size = 0, depth_slice = 0, htile_surface = 0, rmi = 0;

    bool operator<(const BundleDsProgramming& other) const {
        return std::tie(view, render_override, render_override2, htile, size_xy, dfsm,
                        depth_info, z_info, stencil_info, depth_size, depth_slice,
                        htile_surface, rmi) <
               std::tie(other.view, other.render_override, other.render_override2, other.htile,
                        other.size_xy, other.dfsm, other.depth_info, other.z_info,
                        other.stencil_info, other.depth_size, other.depth_slice,
                        other.htile_surface, other.rmi);
    }
    bool operator==(const BundleDsProgramming& other) const {
        return !(*this < other) && !(other < *this);
    }
};

BundleDsProgramming ds_programming(const prosper::gpu::ResolvedPipelineState& ps) {
    return {ps.db_depth_view, ps.db_render_override, ps.db_render_override2,
            ps.htile_data_base, ps.db_depth_size_xy, ps.db_dfsm_control,
            ps.db_depth_info, ps.db_z_info, ps.db_stencil_info,
            ps.db_depth_size, ps.db_depth_slice, ps.db_htile_surface,
            ps.db_rmi_l2_cache_control};
}

int summarize_bundle_ds(const prosper::gpu::GpuCaptureBundle& bundle,
                        size_t first_submit_index, std::string& error) {
    struct Stats {
        uint64_t first_submit = UINT64_MAX, last_submit = 0;
        size_t submits = 0, passes = 0, draws = 0;
        size_t tests = 0, writes = 0, clears = 0;
        uint32_t compare_mask = 0;
        std::set<std::pair<uint32_t, uint32_t>> targets;
        std::set<BundleDsProgramming> programming;
    };
    std::map<BundleDsIdentity, Stats> totals;
    std::map<uint64_t, BundleDsIdentity> previous_by_target;
    std::map<uint64_t, BundleDsProgramming> previous_programming_by_target;
    size_t ds_submits = 0, ds_passes = 0, mixed_passes = 0;
    size_t anonymous_passes = 0, cache_key_transitions = 0, programming_transitions = 0;

    for (size_t i = first_submit_index; i < bundle.submits.size(); ++i) {
        prosper::gpu::GpuCaptureFile manifest;
        if (!prosper::gpu::materialize_gpu_capture_bundle_manifest(bundle, i, manifest, error)) {
            std::fprintf(stderr, "gpu_replay: cannot reconstruct bundle manifest %zu: %s\n",
                         i, error.c_str());
            return 2;
        }
        std::set<BundleDsIdentity> submit_identities;
        bool submit_uses_ds = false;
        size_t pass_begin = 0;
        while (pass_begin < manifest.draws.size()) {
            const uint64_t target = manifest.draws[pass_begin].color0_base;
            size_t pass_end = pass_begin + 1;
            while (pass_end < manifest.draws.size() &&
                   manifest.draws[pass_end].color0_base == target)
                ++pass_end;

            std::set<BundleDsIdentity> pass_identities;
            BundleDsIdentity selected_identity{};
            BundleDsProgramming selected_programming{};
            bool have_selected_identity = false;
            for (size_t draw_index = pass_begin; draw_index < pass_end; ++draw_index) {
                const auto& draw = manifest.draws[draw_index];
                const auto& ps = draw.ps;
                if (!uses_depth_stencil(ps)) continue;
                submit_uses_ds = true;
                const BundleDsIdentity identity = ds_identity(ps);
                if (!have_selected_identity) {
                    selected_identity = identity;
                    selected_programming = ds_programming(ps);
                    have_selected_identity = true;
                }
                pass_identities.insert(identity);
                submit_identities.insert(identity);
                Stats& stats = totals[identity];
                stats.first_submit = std::min(stats.first_submit, manifest.metadata.submit_index);
                stats.last_submit = std::max(stats.last_submit, manifest.metadata.submit_index);
                ++stats.draws;
                stats.tests += ps.depth_test_enable;
                stats.writes += ps.depth_write_enable;
                stats.clears += ps.depth_clear_enable || ps.stencil_clear_enable;
                if (ps.depth_test_enable && ps.depth_compare_op < 32)
                    stats.compare_mask |= 1u << ps.depth_compare_op;
                stats.targets.insert({draw.color0_width, draw.color0_height});
                stats.programming.insert(ds_programming(ps));
            }
            if (!pass_identities.empty()) {
                ++ds_passes;
                for (const auto& identity : pass_identities) ++totals[identity].passes;
                if (pass_identities.size() > 1) {
                    ++mixed_passes;
                    std::printf("ds-mixed-submit=%llu target=%016llx extent=%ux%u identities=%zu\n",
                                static_cast<unsigned long long>(manifest.metadata.submit_index),
                                static_cast<unsigned long long>(target),
                                manifest.draws[pass_begin].color0_width,
                                manifest.draws[pass_begin].color0_height,
                                pass_identities.size());
                }
                if (selected_identity == BundleDsIdentity{}) ++anonymous_passes;
                auto previous = previous_by_target.find(target);
                if (previous != previous_by_target.end() &&
                    !(previous->second == selected_identity))
                    ++cache_key_transitions;
                previous_by_target[target] = selected_identity;
                auto previous_programming = previous_programming_by_target.find(target);
                if (previous_programming != previous_programming_by_target.end() &&
                    !(previous_programming->second == selected_programming))
                    ++programming_transitions;
                previous_programming_by_target[target] = selected_programming;
            }
            pass_begin = pass_end;
        }
        if (submit_uses_ds) ++ds_submits;
        for (const auto& identity : submit_identities) ++totals[identity].submits;
    }

    for (const auto& [identity, stats] : totals) {
        const auto first_target = stats.targets.empty()
            ? std::pair<uint32_t, uint32_t>{0, 0} : *stats.targets.begin();
        std::printf("ds-identity z=%016llx/%016llx s=%016llx/%016llx "
                    "submits=%zu passes=%zu draws=%zu first=%llu last=%llu "
                    "tests=%zu writes=%zu clears=%zu compare-mask=%08x "
                    "targets=%zu first-target=%ux%u programming=%zu\n",
                    static_cast<unsigned long long>(identity.depth_read),
                    static_cast<unsigned long long>(identity.depth_write),
                    static_cast<unsigned long long>(identity.stencil_read),
                    static_cast<unsigned long long>(identity.stencil_write),
                    stats.submits, stats.passes, stats.draws,
                    static_cast<unsigned long long>(stats.first_submit),
                    static_cast<unsigned long long>(stats.last_submit),
                    stats.tests, stats.writes, stats.clears, stats.compare_mask,
                    stats.targets.size(), first_target.first, first_target.second,
                    stats.programming.size());
        if (stats.programming.size() > 1) {
            for (const auto& p : stats.programming)
                std::printf("  ds-programming view=%08x override=%08x/%08x htile=%016llx "
                            "hsurface=%08x size=%08x/%08x/%08x info=%08x/%08x/%08x "
                            "dfsm=%08x rmi=%08x\n",
                            p.view, p.render_override, p.render_override2,
                            static_cast<unsigned long long>(p.htile), p.htile_surface,
                            p.size_xy, p.depth_size, p.depth_slice,
                            p.depth_info, p.z_info, p.stencil_info, p.dfsm, p.rmi);
        }
    }
    std::fprintf(stderr,
                 "[gpureplay] DS summary submits=%zu/%zu passes=%zu identities=%zu "
                 "mixed-passes=%zu anonymous-passes=%zu target-key-transitions=%zu "
                 "programming-transitions=%zu\n",
                 ds_submits, bundle.submits.size() - first_submit_index, ds_passes,
                 totals.size(), mixed_passes, anonymous_passes, cache_key_transitions,
                 programming_transitions);
    return totals.empty() ? 1 : 0;
}

int replay_bundle(const std::string& path, const char* output_path, bool zero_boundary,
                  size_t tail_count, const std::string& compact_path,
                  uint32_t intermediate_target_width, uint32_t intermediate_target_height,
                  const std::string& final_capsule_path,
                  uint64_t extract_submit_no, const std::string& extract_submit_path,
                  uint64_t find_ds_addr, bool ds_summary) {
    prosper::gpu::GpuCaptureBundle bundle;
    std::string error;
    if (!prosper::gpu::read_gpu_capture_bundle(path, bundle, error)) {
        std::fprintf(stderr, "gpu_replay: cannot read bundle: %s\n", error.c_str()); return 2;
    }
    const uint64_t unique_bytes = prosper::gpu::gpu_capture_bundle_unique_bytes(bundle);
    const auto stats = prosper::gpu::gpu_capture_bundle_stats(bundle);
    size_t first_submit_index = tail_count && tail_count < bundle.submits.size()
        ? bundle.submits.size() - tail_count : 0;
    std::fprintf(stderr, "[gpureplay] bundle v%u submits=%zu logical=%llu unique=%llu ratio=%.3f "
                         "chunks=%zu resources=%zu refs=%llu exact-reuse=%llu "
                         "resource-logical=%llu resource-unique=%llu manifest-unique=%llu\n",
                 bundle.version,
                 bundle.submits.size(), static_cast<unsigned long long>(bundle.logical_bytes),
                 static_cast<unsigned long long>(unique_bytes),
                 bundle.logical_bytes ? static_cast<double>(unique_bytes) / bundle.logical_bytes : 0.0,
                 bundle.chunks.size(), bundle.resources.size(),
                 static_cast<unsigned long long>(stats.resource_reference_count),
                 static_cast<unsigned long long>(stats.exact_reuse_count),
                 static_cast<unsigned long long>(stats.resource_logical_bytes),
                 static_cast<unsigned long long>(stats.resource_unique_bytes),
                 static_cast<unsigned long long>(stats.manifest_unique_bytes));
    if (ds_summary) return summarize_bundle_ds(bundle, first_submit_index, error);
    if (find_ds_addr) {
        size_t matching_submits = 0;
        for (size_t i = first_submit_index; i < bundle.submits.size(); ++i) {
            prosper::gpu::GpuCaptureFile manifest;
            if (!prosper::gpu::materialize_gpu_capture_bundle_manifest(bundle, i, manifest, error)) {
                std::fprintf(stderr, "gpu_replay: cannot reconstruct bundle manifest %zu: %s\n",
                             i, error.c_str());
                return 2;
            }
            size_t matches = 0, tests = 0, writes = 0, clears = 0;
            uint32_t compare_mask = 0, width = 0, height = 0;
            uint64_t first_draw = UINT64_MAX, last_draw = 0;
            for (const auto& draw : manifest.draws) {
                const auto& ps = draw.ps;
                if (ps.depth_read_base != find_ds_addr && ps.depth_write_base != find_ds_addr &&
                    ps.stencil_read_base != find_ds_addr && ps.stencil_write_base != find_ds_addr)
                    continue;
                ++matches;
                tests += ps.depth_test_enable;
                writes += ps.depth_write_enable;
                clears += ps.depth_clear_enable || ps.stencil_clear_enable;
                if (ps.depth_test_enable && ps.depth_compare_op < 32)
                    compare_mask |= 1u << ps.depth_compare_op;
                if (first_draw == UINT64_MAX) {
                    first_draw = draw.draw_index; width = draw.color0_width; height = draw.color0_height;
                }
                last_draw = draw.draw_index;
            }
            if (!matches) continue;
            ++matching_submits;
            std::printf("ds-submit=%llu draws=%zu first=%llu last=%llu target=%ux%u "
                        "tests=%zu writes=%zu clears=%zu compare-mask=%08x\n",
                        static_cast<unsigned long long>(manifest.metadata.submit_index), matches,
                        static_cast<unsigned long long>(first_draw),
                        static_cast<unsigned long long>(last_draw), width, height,
                        tests, writes, clears, compare_mask);
        }
        std::fprintf(stderr, "[gpureplay] DS address %016llx matched %zu/%zu submits\n",
                     static_cast<unsigned long long>(find_ds_addr), matching_submits,
                     bundle.submits.size() - first_submit_index);
        return matching_submits ? 0 : 1;
    }
    if (first_submit_index)
        std::fprintf(stderr, "[gpureplay] replaying tail submits=%zu range=%llu..%llu\n",
                     bundle.submits.size() - first_submit_index,
                     static_cast<unsigned long long>(bundle.submits[first_submit_index].submit_index),
                     static_cast<unsigned long long>(bundle.submits.back().submit_index));
    if (!compact_path.empty()) {
        const uint64_t before = prosper::gpu::gpu_capture_bundle_unique_bytes(bundle);
        if (first_submit_index) {
            bundle.submits.erase(bundle.submits.begin(), bundle.submits.begin() + first_submit_index);
            bundle.logical_bytes = 0;
            for (const auto& submit : bundle.submits) bundle.logical_bytes += submit.logical_bytes;
            first_submit_index = 0;
        }
        if (!prosper::gpu::compact_gpu_capture_bundle(bundle, error) ||
            !prosper::gpu::write_gpu_capture_bundle(compact_path, bundle, error)) {
            std::fprintf(stderr, "gpu_replay: cannot compact bundle: %s\n", error.c_str());
            return 2;
        }
        const uint64_t after = prosper::gpu::gpu_capture_bundle_unique_bytes(bundle);
        std::fprintf(stderr, "[gpureplay] compacted unique=%llu -> %llu resources=%zu chunks=%zu -> %s\n",
                     static_cast<unsigned long long>(before),
                     static_cast<unsigned long long>(after), bundle.resources.size(),
                     bundle.chunks.size(), compact_path.c_str());
        if (!output_path && final_capsule_path.empty() && extract_submit_path.empty()) return 0;
    }
    if (!extract_submit_path.empty()) {
        const auto manifest = std::find_if(bundle.submits.begin(), bundle.submits.end(),
            [&](const auto& submit) { return submit.submit_index == extract_submit_no; });
        if (manifest == bundle.submits.end()) {
            std::fprintf(stderr, "gpu_replay: bundle has no submit %llu\n",
                         static_cast<unsigned long long>(extract_submit_no));
            return 2;
        }
        prosper::gpu::GpuCaptureFile extracted;
        const size_t index = static_cast<size_t>(manifest - bundle.submits.begin());
        if (!prosper::gpu::materialize_gpu_capture_bundle_submit(bundle, index, extracted, error) ||
            !prosper::gpu::write_gpu_capture(extract_submit_path, extracted, error)) {
            std::fprintf(stderr, "gpu_replay: cannot extract bundle submit: %s\n", error.c_str());
            return 2;
        }
        std::fprintf(stderr, "[gpureplay] extracted submit=%llu -> %s\n",
                     static_cast<unsigned long long>(extract_submit_no), extract_submit_path.c_str());
        if (!output_path && final_capsule_path.empty()) return 0;
    }

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
    set_environment("PROSPER_GPU_REPLAY_DS_SEEDS", "1");
    if (!final_capsule_path.empty()) {
        set_environment("PROSPER_GPU_REPLAY_EXPORT_RTT", "1");
        set_environment("PROSPER_GPU_REPLAY_EXPORT_DS", "1");
    }
    prosper::frontend::register_live_renderer(".", false);
    final_capture = {};

    std::vector<BundleTarget> prior_targets;
    std::vector<uint8_t> final_pixels;
    uint64_t temporal_resolved = 0, temporal_seeded = 0, temporal_bounded = 0, temporal_unresolved = 0;
    for (size_t i = first_submit_index; i < bundle.submits.size(); ++i) {
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
        // Operation sources are semantic draw IDs; unrealized draws leave holes in the compact item vector.
        std::unordered_map<uint64_t, size_t> draw_item_by_index;
        for (size_t item_index = 0; item_index < replay.items.size(); ++item_index)
            draw_item_by_index[replay.items[item_index].draw_index] = item_index;
        size_t operation_limit = replay.operations.size();
        if (i + 1 < bundle.submits.size() && intermediate_target_width) {
            operation_limit = 0;
            for (size_t operation_index = 0; operation_index < replay.operations.size(); ++operation_index) {
                const auto& operation = replay.operations[operation_index];
                if (!operation.realized ||
                    operation.kind != prosper::gpu::SubmitOperationKind::Draw) continue;
                const auto item = draw_item_by_index.find(operation.source_index);
                if (item == draw_item_by_index.end()) continue;
                const auto& draw = replay.items[item->second];
                if (draw.color0_width == intermediate_target_width &&
                    draw.color0_height == intermediate_target_height)
                    operation_limit = operation_index + 1;
            }
            if (!operation_limit) {
                std::fprintf(stderr,
                             "gpu_replay: bundle submit %llu has no realized %ux%u target\n",
                             static_cast<unsigned long long>(capture.metadata.submit_index),
                             intermediate_target_width, intermediate_target_height);
                return 2;
            }
            for (const auto& leaf : graph.external_leaves) {
                if (leaf.first_future_writer == UINT32_MAX ||
                    leaf.consumer_operations.front() >= operation_limit ||
                    (leaf.access.resource_class != prosper::gpu::ResourceClass::Texture &&
                     leaf.access.resource_class != prosper::gpu::ResourceClass::StorageImage) ||
                    (leaf.access.width == intermediate_target_width &&
                     leaf.access.height == intermediate_target_height)) continue;
                std::fprintf(stderr,
                             "gpu_replay: refusing intermediate truncation at submit %llu: "
                             "temporal leaf op=%u is %ux%u, not %ux%u\n",
                             static_cast<unsigned long long>(capture.metadata.submit_index),
                             leaf.consumer_operations.front(), leaf.access.width, leaf.access.height,
                             intermediate_target_width, intermediate_target_height);
                return 2;
            }
        }
        std::vector<uint64_t> diagnostic_zero_seeds;
        if (i == first_submit_index && zero_boundary) {
            for (const auto& leaf : graph.external_leaves) {
                if (leaf.first_future_writer == UINT32_MAX || !leaf.access.addr ||
                    !leaf.access.width || !leaf.access.height ||
                    (leaf.access.resource_class != prosper::gpu::ResourceClass::Texture &&
                     leaf.access.resource_class != prosper::gpu::ResourceClass::StorageImage) ||
                    std::any_of(replay.rtt_seeds.begin(), replay.rtt_seeds.end(), [&](const auto& seed) {
                        return seed.guest_addr == leaf.access.addr;
                    })) continue;
                prosper::gpu::GpuCaptureRttSeed seed;
                seed.guest_addr = leaf.access.addr;
                seed.width = leaf.access.width; seed.height = leaf.access.height;
                const uint64_t seed_bytes = static_cast<uint64_t>(seed.width) * seed.height * 4;
                constexpr uint64_t kMaxDiagnosticSeedBytes = 1ull << 30;
                if (seed_bytes > kMaxDiagnosticSeedBytes || seed_bytes > SIZE_MAX) {
                    std::fprintf(stderr,
                                 "[gpureplay] skip oversized diagnostic zero seed addr=%016llx "
                                 "dims=%ux%u bytes=%llu\n",
                                 static_cast<unsigned long long>(seed.guest_addr), seed.width, seed.height,
                                 static_cast<unsigned long long>(seed_bytes));
                    continue;
                }
                seed.rgba.resize(static_cast<size_t>(seed_bytes), 0);
                replay.rtt_seeds.push_back(std::move(seed));
                diagnostic_zero_seeds.push_back(leaf.access.addr);
            }
            std::fprintf(stderr, "[gpureplay] diagnostic zero-boundary seeds=%zu\n",
                         diagnostic_zero_seeds.size());
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
            const char* stop = i == first_submit_index ? "configured-bound" : "unresolved-producer";
            uint64_t producer_submit = 0;
            if (producer != prior_targets.rend()) {
                stop = "included-producer"; producer_submit = producer->submit; ++temporal_resolved;
            } else if (seed != replay.rtt_seeds.end()) {
                stop = std::find(diagnostic_zero_seeds.begin(), diagnostic_zero_seeds.end(),
                                 leaf.access.addr) != diagnostic_zero_seeds.end()
                    ? "diagnostic-zero-seed" : "initialized-seed";
                ++temporal_seeded;
            } else {
                if (i == first_submit_index) ++temporal_bounded;
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
        if (!prosper::gpu::restore_gpu_replay_ds_seeds(replay.ds_seeds, error)) {
            std::fprintf(stderr, "gpu_replay: cannot restore bundle DS seeds: %s\n", error.c_str());
            return 2;
        }
        bool final_capsule_prepared = false;
        size_t final_cache_surfaces = 0, final_required = 0;
        size_t final_exported = 0, final_refreshed = 0, final_migrated_failures = 0;
        uint64_t final_ds_snapshot_bytes = 0;
        if (i + 1 == bundle.submits.size() && !final_capsule_path.empty()) {
            std::unordered_set<uint64_t> required;
            for (const auto& leaf : graph.external_leaves) {
                if (leaf.first_future_writer == UINT32_MAX ||
                    (leaf.access.resource_class != prosper::gpu::ResourceClass::Texture &&
                     leaf.access.resource_class != prosper::gpu::ResourceClass::StorageImage)) continue;
                required.insert(leaf.access.addr);
            }
            std::vector<prosper::gpu::GpuCaptureRttSeed> snapshot;
            if (!prosper::gpu::read_all_gpu_capture_rtt_seeds(snapshot, error)) {
                std::fprintf(stderr, "gpu_replay: cannot snapshot final RTT cache: %s\n",
                             error.c_str());
                return 2;
            }
            for (auto& seed : snapshot) {
                const uint64_t addr = seed.guest_addr;
                auto existing = std::find_if(capture.rtt_seeds.begin(), capture.rtt_seeds.end(),
                    [&](const auto& candidate) { return candidate.guest_addr == addr; });
                if (existing == capture.rtt_seeds.end()) {
                    capture.rtt_seeds.push_back(std::move(seed));
                    ++final_exported;
                } else {
                    *existing = std::move(seed);
                    ++final_refreshed;
                }
            }
            for (uint64_t addr : required)
                if (std::none_of(capture.rtt_seeds.begin(), capture.rtt_seeds.end(),
                    [&](const auto& seed) { return seed.guest_addr == addr; })) {
                    std::fprintf(stderr,
                                 "gpu_replay: final RTT snapshot lacks required temporal seed %016llx\n",
                                 static_cast<unsigned long long>(addr));
                    return 2;
                }
            std::vector<prosper::gpu::GpuCaptureDsSeed> ds_snapshot;
            if (!prosper::gpu::read_all_gpu_capture_ds_seeds(ds_snapshot, error)) {
                std::fprintf(stderr, "gpu_replay: cannot snapshot final DS cache: %s\n",
                             error.c_str());
                return 2;
            }
            for (const auto& seed : ds_snapshot)
                final_ds_snapshot_bytes += seed.depth.size() + seed.stencil.size();
            capture.ds_seeds = std::move(ds_snapshot);
            if (!capture.failure_diagnostics_available) {
                for (const auto& operation : capture.operations) {
                    if (operation.realized) continue;
                    prosper::gpu::GpuCapturedOperationFailure failure;
                    failure.kind = operation.kind;
                    failure.source_index = operation.source_index;
                    failure.command_order = operation.command_order;
                    failure.reason = prosper::gpu::RealizationFailureReason::Unknown;
                    capture.failure_diagnostics.push_back(std::move(failure));
                    ++final_migrated_failures;
                }
                capture.failure_diagnostics_available = true;
            }
            for (const char* name : {"PROSPER_DS_GUEST_WRITE_INVALIDATE",
                                     "PROSPER_GPU_REPLAY_LEGACY_HTILE_BEFORE_STENCIL"}) {
                const char* value = std::getenv(name);
                if (!value) continue;
                auto existing = std::find_if(capture.metadata.renderer_env.begin(),
                                             capture.metadata.renderer_env.end(),
                    [&](const auto& entry) { return entry.first == name; });
                if (existing == capture.metadata.renderer_env.end())
                    capture.metadata.renderer_env.emplace_back(name, value);
                else
                    existing->second = value;
            }
            final_cache_surfaces = snapshot.size(); final_required = required.size();
            final_capsule_prepared = true;
        }
        if (!prosper::gpu::restore_gpu_replay_rtt_seeds(seeds, error)) {
            std::fprintf(stderr, "gpu_replay: cannot restore bundle RTT seeds: %s\n", error.c_str());
            return 2;
        }
        final_pixels = execute_frame(replay, false, operation_limit);
        if (final_capsule_prepared) {
            if (final_pixels.empty()) {
                std::fprintf(stderr, "gpu_replay: final submit produced no checkpoint oracle\n");
                return 2;
            }
            capture.expected_output_valid = true;
            capture.expected_output_bytes = final_pixels.size();
            capture.expected_output_hash = prosper::gpu::gpu_capture_hash(final_pixels);
            if (!prosper::gpu::write_gpu_capture(final_capsule_path, capture, error)) {
                std::fprintf(stderr, "gpu_replay: cannot write final seeded capsule: %s\n",
                             error.c_str());
                return 2;
            }
            std::fprintf(stderr,
                         "[gpureplay] exported final seeded capsule cache-surfaces=%zu required=%zu "
                         "new-seeds=%zu refreshed=%zu total-seeds=%zu DS-seeds=%zu DS-bytes=%llu "
                         "migrated-failures=%zu oracle=%016llx/%llu -> %s\n",
                         final_cache_surfaces, final_required, final_exported, final_refreshed,
                         capture.rtt_seeds.size(), capture.ds_seeds.size(),
                         static_cast<unsigned long long>(final_ds_snapshot_bytes),
                         final_migrated_failures,
                         static_cast<unsigned long long>(capture.expected_output_hash),
                         static_cast<unsigned long long>(capture.expected_output_bytes),
                         final_capsule_path.c_str());
        }
        for (size_t operation_index = 0; operation_index < operation_limit; ++operation_index) {
            const auto& operation = replay.operations[operation_index];
            if (!operation.realized ||
                operation.kind != prosper::gpu::SubmitOperationKind::Draw) continue;
            const auto item = draw_item_by_index.find(operation.source_index);
            if (item == draw_item_by_index.end()) continue;
            const auto& draw = replay.items[item->second];
            if (draw.color0_base && draw.color0_width && draw.color0_height)
                prior_targets.push_back({draw.color0_base,
                    static_cast<uint64_t>(draw.color0_width) * draw.color0_height * 4,
                    capture.metadata.submit_index});
        }
        std::fprintf(stderr, "[gpureplay] bundle-submit=%llu operations=%zu/%zu output_bytes=%zu hash=%016llx\n",
                     static_cast<unsigned long long>(capture.metadata.submit_index),
                     operation_limit, replay.operations.size(), final_pixels.size(),
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
    bool graph_only = false, bundle_zero_boundary = false, bundle_ds_summary = false;
    bool legacy_htile_before_stencil = false, draw_with_compute_prefix = false;
    size_t bundle_tail = 0;
    uint32_t bundle_intermediate_target_width = 0, bundle_intermediate_target_height = 0;
    int draw_first = -1, draw_last = -1;
    int through_operation = -1;
    uint32_t warmup_repeats = 0;
    std::string dump_spec, dump_path, shader_spec, shader_path;
    std::string compute_shader_spec, compute_shader_path;
    std::string compute_resource_spec, compute_resource_path;
    std::string failed_shader_spec, failed_shader_path;
    std::string graph_json_path, prepend_path;
    std::string bundle_path, bundle_compact_path;
    std::string bundle_final_capsule_path;
    std::string bundle_extract_submit_path;
    uint64_t bundle_extract_submit_no = 0;
    uint64_t bundle_find_ds_addr = 0;
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
        else if (std::string(argv[i]) == "--legacy-htile-before-stencil")
            legacy_htile_before_stencil = true;
        else if (std::string(argv[i]) == "--bundle" && i + 1 < argc) bundle_path = argv[++i];
        else if (std::string(argv[i]) == "--bundle-zero-boundary") bundle_zero_boundary = true;
        else if (std::string(argv[i]) == "--bundle-ds-summary") bundle_ds_summary = true;
        else if (std::string(argv[i]) == "--bundle-compact" && i + 1 < argc)
            bundle_compact_path = argv[++i];
        else if (std::string(argv[i]) == "--bundle-tail" && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 0);
            if (!end || *end || !value || value > SIZE_MAX) { usage(argv[0]); return 2; }
            bundle_tail = static_cast<size_t>(value);
        }
        else if (std::string(argv[i]) == "--bundle-intermediate-through-target" && i + 1 < argc) {
            unsigned width = 0, height = 0; char tail = 0;
            if (std::sscanf(argv[++i], "%ux%u%c", &width, &height, &tail) != 2 ||
                !width || !height) { usage(argv[0]); return 2; }
            bundle_intermediate_target_width = width;
            bundle_intermediate_target_height = height;
        }
        else if (std::string(argv[i]) == "--bundle-final-capsule" && i + 1 < argc)
            bundle_final_capsule_path = argv[++i];
        else if (std::string(argv[i]) == "--bundle-extract-submit" && i + 2 < argc) {
            char* end = nullptr;
            bundle_extract_submit_no = std::strtoull(argv[++i], &end, 0);
            if (!end || *end || !bundle_extract_submit_no) { usage(argv[0]); return 2; }
            bundle_extract_submit_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--bundle-find-ds" && i + 1 < argc) {
            char* end = nullptr;
            bundle_find_ds_addr = std::strtoull(argv[++i], &end, 0);
            if (!end || *end || !bundle_find_ds_addr) { usage(argv[0]); return 2; }
        }
        else if (std::string(argv[i]) == "--prepend" && i + 1 < argc) prepend_path = argv[++i];
        else if (std::string(argv[i]) == "--draw" && i + 1 < argc) {
            std::string range = argv[++i]; size_t colon = range.find(':');
            draw_first = std::atoi(range.c_str());
            draw_last = colon == std::string::npos ? draw_first : std::atoi(range.c_str() + colon + 1);
        }
        else if (std::string(argv[i]) == "--draw-with-compute-prefix")
            draw_with_compute_prefix = true;
        else if (std::string(argv[i]) == "--through-operation" && i + 1 < argc) {
            char* end = nullptr;
            const long value = std::strtol(argv[++i], &end, 0);
            if (!end || *end || value < 0 || value > INT_MAX) { usage(argv[0]); return 2; }
            through_operation = static_cast<int>(value);
        }
        else if (std::string(argv[i]) == "--warmup-repeats" && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long value = std::strtoul(argv[++i], &end, 0);
            if (!end || *end || !value || value > 1024) { usage(argv[0]); return 2; }
            warmup_repeats = static_cast<uint32_t>(value);
        }
        else if (std::string(argv[i]) == "--dump-resource" && i + 2 < argc) {
            dump_spec = argv[++i]; dump_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--dump-shader" && i + 2 < argc) {
            shader_spec = argv[++i]; shader_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--dump-compute" && i + 2 < argc) {
            compute_shader_spec = argv[++i]; compute_shader_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--dump-compute-resource" && i + 2 < argc) {
            compute_resource_spec = argv[++i]; compute_resource_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--dump-failed-shader" && i + 2 < argc) {
            failed_shader_spec = argv[++i]; failed_shader_path = argv[++i];
        }
        else positional.push_back(argv[i]);
    }
    if (legacy_htile_before_stencil)
        set_environment("PROSPER_GPU_REPLAY_LEGACY_HTILE_BEFORE_STENCIL", "1");
    if (!bundle_path.empty()) {
        if (bundle_ds_summary && bundle_find_ds_addr) { usage(argv[0]); return 2; }
        if (positional.size() > 1 || inspect || inspect_only || validate_only || graph_only ||
            draw_first >= 0 || draw_with_compute_prefix || through_operation >= 0 ||
            warmup_repeats || !dump_spec.empty() ||
            !shader_spec.empty() || !compute_shader_spec.empty() ||
            !compute_resource_spec.empty() || !failed_shader_spec.empty() ||
            !prepend_path.empty()) {
            usage(argv[0]); return 2;
        }
        return replay_bundle(bundle_path, positional.empty() ? nullptr : positional[0],
                             bundle_zero_boundary, bundle_tail, bundle_compact_path,
                             bundle_intermediate_target_width,
                             bundle_intermediate_target_height,
                             bundle_final_capsule_path,
                             bundle_extract_submit_no,
                             bundle_extract_submit_path,
                             bundle_find_ds_addr,
                             bundle_ds_summary);
    }
    if (bundle_zero_boundary || bundle_tail || !bundle_compact_path.empty() ||
        bundle_intermediate_target_width || !bundle_final_capsule_path.empty() ||
        !bundle_extract_submit_path.empty() || bundle_find_ds_addr || bundle_ds_summary) {
        usage(argv[0]); return 2;
    }
    if (positional.empty() || positional.size() > 2) { usage(argv[0]); return 2; }
    if (draw_first >= 0 && through_operation >= 0) { usage(argv[0]); return 2; }
    if (draw_with_compute_prefix && draw_first < 0) { usage(argv[0]); return 2; }
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
    if (!failed_shader_spec.empty()) {
        const size_t colon = failed_shader_spec.find(':');
        char* failure_end = nullptr;
        const long failure_index = std::strtol(failed_shader_spec.c_str(), &failure_end, 0);
        char* stage_end = nullptr;
        const long stage_index = colon == std::string::npos ? -1 :
            std::strtol(failed_shader_spec.c_str() + colon + 1, &stage_end, 0);
        if (colon == std::string::npos || failure_end != failed_shader_spec.c_str() + colon ||
            !stage_end || *stage_end || failure_index < 0 || stage_index < 0 ||
            static_cast<size_t>(failure_index) >= replay.failure_diagnostics.size() ||
            static_cast<size_t>(stage_index) >=
                replay.failure_diagnostics[static_cast<size_t>(failure_index)].stages.size()) {
            std::fprintf(stderr, "gpu_replay: invalid failed-shader selector %s\n",
                         failed_shader_spec.c_str());
            return 2;
        }
        const auto& stage = replay.failure_diagnostics[static_cast<size_t>(failure_index)]
                                .stages[static_cast<size_t>(stage_index)];
        if (stage.raw_shader_index >= replay.raw_shader_versions.size()) {
            std::fprintf(stderr, "gpu_replay: failed shader %s has no captured raw stream\n",
                         failed_shader_spec.c_str());
            return 2;
        }
        const auto& words = replay.raw_shader_versions[stage.raw_shader_index].words;
        FILE* f = std::fopen(failed_shader_path.c_str(), "wb");
        const size_t bytes = words.size() * sizeof(uint32_t);
        if (!f || std::fwrite(words.data(), 1, bytes, f) != bytes) {
            if (f) std::fclose(f);
            std::fprintf(stderr, "gpu_replay: cannot write %s\n", failed_shader_path.c_str());
            return 2;
        }
        std::fclose(f);
        std::fprintf(stderr, "[gpureplay] dumped failed shader %s (%zu bytes) to %s\n",
                     failed_shader_spec.c_str(), bytes, failed_shader_path.c_str());
        if (positional.size() == 1 && !inspect) return 0;
    }
    if (!compute_shader_spec.empty()) {
        char* end = nullptr;
        const long index = std::strtol(compute_shader_spec.c_str(), &end, 0);
        if (!end || *end || index < 0 || static_cast<size_t>(index) >= replay.computes.size()) {
            std::fprintf(stderr, "gpu_replay: invalid compute selector %s\n",
                         compute_shader_spec.c_str());
            return 2;
        }
        const auto& words = replay.computes[static_cast<size_t>(index)].spirv;
        FILE* f = std::fopen(compute_shader_path.c_str(), "wb");
        const size_t bytes = words.size() * sizeof(uint32_t);
        if (!f || std::fwrite(words.data(), 1, bytes, f) != bytes) {
            if (f) std::fclose(f);
            std::fprintf(stderr, "gpu_replay: cannot write %s\n", compute_shader_path.c_str());
            return 2;
        }
        std::fclose(f);
        std::fprintf(stderr, "[gpureplay] dumped compute %ld (%zu bytes) to %s\n",
                     index, bytes, compute_shader_path.c_str());
    }
    if (!compute_resource_spec.empty()) {
        const size_t colon = compute_resource_spec.find(':');
        char* index_end = nullptr;
        const long index = std::strtol(compute_resource_spec.c_str(), &index_end, 0);
        char* binding_end = nullptr;
        const long binding = colon == std::string::npos ? -1 :
            std::strtol(compute_resource_spec.c_str() + colon + 1, &binding_end, 0);
        if (colon == std::string::npos || index_end != compute_resource_spec.c_str() + colon ||
            !binding_end || *binding_end || index < 0 || binding < 0 ||
            static_cast<size_t>(index) >= replay.computes.size()) {
            std::fprintf(stderr, "gpu_replay: invalid compute resource selector %s\n",
                         compute_resource_spec.c_str());
            return 2;
        }
        const auto& compute = replay.computes[static_cast<size_t>(index)];
        const prosper::gpu::ShaderResource* found = nullptr;
        if (compute.resources)
            for (const auto& resource : compute.resources->resources)
                if (resource.binding == static_cast<uint32_t>(binding)) {
                    found = &resource;
                    break;
                }
        if (!found || !found->host_data) {
            std::fprintf(stderr, "gpu_replay: compute resource %s not found or has no captured bytes\n",
                         compute_resource_spec.c_str());
            return 2;
        }
        FILE* f = std::fopen(compute_resource_path.c_str(), "wb");
        if (!f || std::fwrite(found->host_data, 1, static_cast<size_t>(found->host_data_size), f) !=
                      found->host_data_size) {
            if (f) std::fclose(f);
            std::fprintf(stderr, "gpu_replay: cannot write %s\n", compute_resource_path.c_str());
            return 2;
        }
        std::fclose(f);
        std::fprintf(stderr, "[gpureplay] dumped compute resource %s (%llu bytes) to %s\n",
                     compute_resource_spec.c_str(),
                     static_cast<unsigned long long>(found->host_data_size),
                     compute_resource_path.c_str());
    }
    size_t selected_operation_limit = SIZE_MAX;
    if (draw_first >= 0) {
        if (draw_last < draw_first || static_cast<size_t>(draw_last) >= replay.items.size()) {
            std::fprintf(stderr, "gpu_replay: draw range %d:%d is out of range\n", draw_first, draw_last); return 2;
        }
        std::vector<prosper::gpu::DrawItem> selected;
        std::unordered_set<uint64_t> selected_draw_indexes;
        for (int i = draw_first; i <= draw_last; ++i) {
            selected_draw_indexes.insert(replay.items[i].draw_index);
            selected.push_back(std::move(replay.items[i]));
        }
        if (draw_with_compute_prefix) {
            for (size_t operation_index = 0; operation_index < replay.operations.size();
                 ++operation_index) {
                const auto& operation = replay.operations[operation_index];
                if (operation.kind == prosper::gpu::SubmitOperationKind::Draw &&
                    selected_draw_indexes.count(operation.source_index))
                    selected_operation_limit = operation_index + 1;
            }
            if (selected_operation_limit == SIZE_MAX) {
                std::fprintf(stderr, "gpu_replay: selected draw range has no submit operation\n");
                return 2;
            }
        }
        replay.items = std::move(selected); allow_mismatch = true;
        std::fprintf(stderr, "[gpureplay] selected original draws %d:%d%s\n", draw_first,
                     draw_last, draw_with_compute_prefix ? " with compute prefix" : "");
    }
    if (through_operation >= 0) {
        if (static_cast<size_t>(through_operation) >= replay.operations.size()) {
            std::fprintf(stderr, "gpu_replay: operation %d is out of range\n", through_operation);
            return 2;
        }
        allow_mismatch = true;
        std::fprintf(stderr, "[gpureplay] executing through mixed operation %d\n", through_operation);
    }
    const auto& m = replay.metadata;
    const bool metadata_only = std::any_of(
        m.renderer_env.begin(), m.renderer_env.end(), [](const auto& entry) {
            return entry.first == "PROSPER_GPU_CAPTURE_METADATA_ONLY" &&
                   !entry.second.empty() && entry.second != "0" && entry.second != "off";
        });
    std::fprintf(stderr, "[gpureplay] rev=%s title=%s submit=%llu %ux%u draws=%zu computes=%zu "
                         "operations=%zu shaders=%zu failed=%zu raw-failed-shaders=%zu blobs=%zu "
                         "RTT-seeds=%zu DS-seeds=%zu oracle=%s resource-data=%s\n",
                 m.revision.c_str(), m.title_id.c_str(), static_cast<unsigned long long>(m.submit_index),
                 m.width, m.height, replay.items.size(), replay.computes.size(), replay.operations.size(),
                 capture.shader_versions.size(), replay.failure_diagnostics.size(),
                 replay.raw_shader_versions.size(), replay.blobs.size(), replay.rtt_seeds.size(),
                 replay.ds_seeds.size(),
                 replay.expected_output_valid ? "yes" : "no",
                 metadata_only ? "omitted" : "present");
    if (inspect) inspect_frame(replay);
    if (validate_only) return validate_frame(replay) ? 0 : 1;
    if (inspect_only) return 0;
    if (metadata_only) {
        std::fprintf(stderr, "gpu_replay: metadata-only capture cannot render; use --inspect-only, "
                             "--validate, or --graph\n");
        return 2;
    }
    if (!replay.rtt_seeds.empty() || !prepend.rtt_seeds.empty())
        set_environment("PROSPER_GPU_REPLAY_RTT_SEEDS", "1");
    if (!replay.ds_seeds.empty() || !prepend.ds_seeds.empty())
        set_environment("PROSPER_GPU_REPLAY_DS_SEEDS", "1");
    prosper::frontend::register_live_renderer(".", false);
    std::unordered_set<uint64_t> predecessor_targets;
    if (!prepend_path.empty()) {
        if (!prosper::gpu::restore_gpu_replay_rtt_seeds(prepend.rtt_seeds, error)) {
            std::fprintf(stderr, "gpu_replay: cannot restore predecessor RTT seeds: %s\n",
                         error.c_str());
            return 2;
        }
        if (!prosper::gpu::restore_gpu_replay_ds_seeds(prepend.ds_seeds, error)) {
            std::fprintf(stderr, "gpu_replay: cannot restore predecessor DS seeds: %s\n",
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
    if (!prosper::gpu::restore_gpu_replay_ds_seeds(replay.ds_seeds, error)) {
        std::fprintf(stderr, "gpu_replay: cannot restore DS seeds: %s\n", error.c_str()); return 2;
    }
    for (uint32_t repeat = 0; repeat < warmup_repeats; ++repeat) {
        std::vector<uint8_t> warmup_pixels = execute_frame(replay);
        std::fprintf(stderr, "[gpureplay] warmup repeat=%u/%u output_bytes=%zu hash=%016llx\n",
                     repeat + 1, warmup_repeats, warmup_pixels.size(),
                     static_cast<unsigned long long>(
                         prosper::gpu::gpu_capture_hash(warmup_pixels)));
    }
    const size_t operation_limit = through_operation >= 0
        ? static_cast<size_t>(through_operation) + 1 : selected_operation_limit;
    std::vector<uint8_t> pixels = execute_frame(
        replay, draw_first >= 0 && !draw_with_compute_prefix, operation_limit);
    uint32_t output_width = m.width, output_height = m.height;
    if (through_operation >= 0) {
        for (size_t operation_index = 0; operation_index < operation_limit; ++operation_index) {
            const auto& operation = replay.operations[operation_index];
            if (!operation.realized ||
                operation.kind != prosper::gpu::SubmitOperationKind::Draw) continue;
            const auto item = std::find_if(replay.items.begin(), replay.items.end(),
                [&](const auto& draw) { return draw.draw_index == operation.source_index; });
            if (item != replay.items.end() && item->color0_width && item->color0_height) {
                output_width = item->color0_width;
                output_height = item->color0_height;
            }
        }
    }
    uint64_t hash = prosper::gpu::gpu_capture_hash(pixels);
    std::fprintf(stderr, "[gpureplay] output=%ux%u bytes=%zu hash=%016llx "
                         "expected_bytes=%llu expected_hash=%016llx\n",
                 output_width, output_height, pixels.size(), static_cast<unsigned long long>(hash),
                 static_cast<unsigned long long>(replay.expected_output_bytes),
                 static_cast<unsigned long long>(replay.expected_output_hash));
    if (positional.size() == 2 && !pixels.empty() &&
        (pixels.size() != static_cast<size_t>(output_width) * output_height * 4 ||
         !prosper::test::dump_bmp(positional[1], pixels, output_width, output_height))) {
        std::fprintf(stderr, "gpu_replay: cannot write %s\n", positional[1]); return 2;
    }
    if (!allow_mismatch && replay.expected_output_valid &&
        (pixels.size() != replay.expected_output_bytes || hash != replay.expected_output_hash)) {
        std::fprintf(stderr, "gpu_replay: output mismatch\n"); return 1;
    }
    return 0;
}
