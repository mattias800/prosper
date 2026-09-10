#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include "gpu/capture/gpu_capture.hpp"

#include <mutex>
#include "hle/fs/save_paths.hpp"   // the effective per-title /savedata0 dir (#2734)
#include "gpu/texture/bc_decode.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "build_revision.hpp"
#include "gpu/texture/guest_texture_layout.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/mip_chain_plan.hpp"
#include "gpu/texture/tile.hpp"
#include "gpu/present/videoout_present.hpp"

#include <algorithm>
#include <atomic>
#include <string>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <unordered_set>
#include <utility>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/uio.h>
#include <unistd.h>
#include "host/platform/posix_shim.hpp"
#endif
#include "gpu/capture/gpu_capture_internal.hpp"
#include "gpu/capture/capture_codecs.hpp"

namespace prosper::gpu {
namespace {



} // namespace

bool serialize_gpu_capture(const GpuCaptureFile& c, std::vector<uint8_t>& bytes, std::string& error) {
    error.clear(); Writer w; w.raw(kMagic, sizeof(kMagic)); w.u32(kVersion); w.u32(kEndian);
    w.u32(c.metadata.width); w.u32(c.metadata.height); w.u64(c.metadata.submit_index);
    w.string(c.metadata.revision); w.string(c.metadata.title_id); w.string(c.metadata.input_route); w.string(c.metadata.savedata_dir);
    w.u32(static_cast<uint32_t>(c.metadata.renderer_env.size()));
    for (const auto& [name, value] : c.metadata.renderer_env) { w.string(name); w.string(value); }
    w.u8(c.expected_output_valid); w.u64(c.expected_output_hash); w.u64(c.expected_output_bytes);
    w.u32(static_cast<uint32_t>(c.blobs.size()));
    for (const auto& b : c.blobs) {
        const uint64_t hash = gpu_capture_hash(b.bytes);
        if (b.content_hash && b.content_hash != hash) { error = "capture blob content hash mismatch"; return false; }
        w.u64(b.guest_addr); w.u64(b.bytes_read); w.u64(hash); w.bytes(b.bytes);
    }
    if (c.rtt_seeds.size() > kMaxResources) { error = "invalid RTT seed count"; return false; }
    uint64_t seed_total = 0; std::unordered_set<uint64_t> seed_addresses;
    w.u32(static_cast<uint32_t>(c.rtt_seeds.size()));
    for (const auto& seed : c.rtt_seeds) {
        if (!validate_rtt_seed(seed, error)) return false;
        if (!seed_addresses.insert(seed.guest_addr).second) { error = "duplicate RTT seed address"; return false; }
        if (seed_total > kMaxTotalRttSeedBytes - seed.rgba.size()) {
            error = "capture RTT seed data exceeds 1 GiB"; return false;
        }
        seed_total += seed.rgba.size();
        w.u64(seed.guest_addr); w.u32(seed.width); w.u32(seed.height);
        w.u32(static_cast<uint32_t>(seed.format)); w.bytes(seed.rgba);
    }
    std::vector<GpuCaptureShaderVersion> versions = c.shader_versions;
    auto add_shader = [&](const std::vector<uint32_t>& words) {
        const uint64_t hash = shader_hash(words);
        auto it = std::find_if(versions.begin(), versions.end(), [&](const auto& version) {
            return version.content_hash == hash && version.words == words;
        });
        if (it == versions.end()) versions.push_back({hash, words});
    };
    for (const auto& d : c.draws) {
        add_shader(d.vs);
        if (!d.gs.empty()) add_shader(d.gs);
        add_shader(d.fs);
    }
    for (const auto& compute : c.computes) add_shader(compute.spirv);
    if (versions.size() > kMaxResources) { error = "invalid shader-version count"; return false; }
    w.u32(static_cast<uint32_t>(versions.size()));
    for (const auto& version : versions) {
        const uint64_t hash = shader_hash(version.words);
        if (version.content_hash && version.content_hash != hash) {
            error = "capture shader content hash mismatch"; return false;
        }
        w.u64(hash); w.words(version.words);
    }
    if (c.draws.size() > kMaxDraws) { error = "invalid draw count"; return false; }
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& d : c.draws) {
        const uint32_t vs_index = shader_version_index(versions, d.vs);
        const uint32_t fs_index = shader_version_index(versions, d.fs);
        if (vs_index == 0xFFFFFFFFu || fs_index == 0xFFFFFFFFu) {
            error = "draw shader is missing from the version table"; return false;
        }
        w.u32(vs_index); w.u32(fs_index);
        write_pipeline(w, d.ps); write_table(w, d.vrt); write_table(w, d.prt);
        w.u32(d.vertex_count); w.words(d.indices); w.u64(d.color0_base);
        w.u32(d.color0_width); w.u32(d.color0_height);
        w.u64(d.draw_index); w.u64(d.command_order);
    }
    if (c.computes.size() > kMaxComputes) { error = "invalid compute count"; return false; }
    w.u32(static_cast<uint32_t>(c.computes.size()));
    for (const auto& compute : c.computes) {
        const uint32_t shader_index = shader_version_index(versions, compute.spirv);
        if (shader_index == 0xFFFFFFFFu) { error = "compute shader is missing from the version table"; return false; }
        w.u32(shader_index); write_table(w, compute.resources);
        w.u32(compute.launch.threads_x); w.u32(compute.launch.threads_y); w.u32(compute.launch.threads_z);
        w.u32(compute.launch.local_x); w.u32(compute.launch.local_y); w.u32(compute.launch.local_z);
        w.u32(compute.launch.groups_x); w.u32(compute.launch.groups_y); w.u32(compute.launch.groups_z);
        w.u64(compute.code_addr); w.u64(compute.dispatch_index); w.u64(compute.submit_no);
        w.u64(compute.command_order);
    }
    if (c.operations.size() > kMaxOperations || !validate_dma_copies(c, error)) {
        if (error.empty()) error = "invalid operation count";
        return false;
    }
    w.u32(static_cast<uint32_t>(c.operations.size()));
    for (const auto& operation : c.operations) {
        w.u8(static_cast<uint8_t>(operation.kind)); w.u64(operation.source_index);
        w.u64(operation.command_order); w.u8(operation.realized);
    }
    if (!validate_failure_diagnostics(c, error)) return false;
    w.u32(static_cast<uint32_t>(c.raw_shader_versions.size()));
    for (const auto& shader : c.raw_shader_versions) {
        w.u64(shader.content_hash); w.u8(shader.has_endpgm); w.words(shader.words);
    }
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) {
        w.u8(static_cast<uint8_t>(diagnostic.kind)); w.u64(diagnostic.source_index);
        w.u64(diagnostic.command_order); w.u8(static_cast<uint8_t>(diagnostic.reason));
        w.u8(diagnostic.pipeline_present);
        if (diagnostic.pipeline_present) write_pipeline(w, diagnostic.pipeline);
        w.u64(diagnostic.color0_base); w.u32(diagnostic.color0_width);
        w.u32(diagnostic.color0_height); w.u32(diagnostic.vertex_count);
        const auto& launch = diagnostic.compute_launch;
        w.u32(launch.threads_x); w.u32(launch.threads_y); w.u32(launch.threads_z);
        w.u32(launch.local_x); w.u32(launch.local_y); w.u32(launch.local_z);
        w.u32(launch.groups_x); w.u32(launch.groups_y); w.u32(launch.groups_z);
        w.u32(static_cast<uint32_t>(diagnostic.stages.size()));
        for (const auto& stage : diagnostic.stages) {
            w.u8(static_cast<uint8_t>(stage.stage)); w.u64(stage.program_addr);
            w.u32(stage.raw_shader_index); w.u8(stage.recompiled);
            w.u8(stage.resource_table_present); w.u32(stage.resource_count);
            w.u32(stage.coverage.total); w.u32(stage.coverage.alu);
            w.u32(stage.coverage.exports); w.u32(stage.coverage.unsupported);
            w.u32(stage.coverage.table_dependent);
            w.u32(static_cast<uint32_t>(stage.coverage.first_bad_fmt));
            w.u32(stage.coverage.first_bad_op); w.u32(stage.coverage.first_bad_pc);
            w.u32(stage.descriptor_issue_count); w.u32(stage.first_descriptor_issue);
        }
    }
    if (c.ds_seeds.size() > kMaxResources) { error = "invalid DS seed count"; return false; }
    uint64_t ds_seed_total = 0;
    std::set<decltype(ds_seed_key(GpuCaptureDsSeed{}))> ds_seed_keys;
    w.u32(static_cast<uint32_t>(c.ds_seeds.size()));
    for (const auto& seed : c.ds_seeds) {
        if (!validate_ds_seed(seed, error)) return false;
        if (!ds_seed_keys.insert(ds_seed_key(seed)).second) {
            error = "duplicate DS seed identity"; return false;
        }
        const uint64_t plane_bytes = seed.depth.size() + seed.stencil.size();
        if (plane_bytes > kMaxTotalDsSeedBytes ||
            ds_seed_total > kMaxTotalDsSeedBytes - plane_bytes) {
            error = "capture DS seed data exceeds 1 GiB"; return false;
        }
        ds_seed_total += plane_bytes;
        w.u64(seed.depth_read_base); w.u64(seed.depth_write_base);
        w.u64(seed.stencil_read_base); w.u64(seed.stencil_write_base);
        w.u64(seed.htile_data_base); w.u32(seed.width); w.u32(seed.height);
        w.u32(static_cast<uint32_t>(seed.format));
        w.u8(seed.depth_valid); w.u8(seed.stencil_valid);
        w.bytes(seed.depth); w.bytes(seed.stencil);
    }

    // v9 extends the resource contract with the base-level depth of 3D images. Keep the resource
    // record itself byte-compatible with v1-v8 and append depths in deterministic table order, so old
    // captures remain readable without duplicating every legacy table parser.
    uint64_t resource_depth_count = 0;
    for (const auto& draw : c.draws)
        resource_depth_count += draw.vrt.resources.size() + draw.prt.resources.size();
    for (const auto& compute : c.computes) resource_depth_count += compute.resources.resources.size();
    if (resource_depth_count > UINT32_MAX) { error = "invalid resource depth count"; return false; }
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_depths = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources) w.u32(captured.resource.depth);
    };
    for (const auto& draw : c.draws) { write_depths(draw.vrt); write_depths(draw.prt); }
    for (const auto& compute : c.computes) write_depths(compute.resources);
    // v10 appends MRT1 state in one deterministic extension, leaving every v1-v9 record prefix byte-
    // compatible. This also keeps old fixture construction simple: remove this tail and lower version.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        w.u64(draw.color1_base); w.u32(draw.color1_width); w.u32(draw.color1_height);
        write_mrt1_pipeline(w, draw.ps);
    }
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) {
        w.u64(diagnostic.color1_base); w.u32(diagnostic.color1_width);
        w.u32(diagnostic.color1_height);
        if (diagnostic.pipeline_present) write_mrt1_pipeline(w, diagnostic.pipeline);
    }
    // v11 preserves GFX10 DCC descriptor state in deterministic resource-table order. Like the v9
    // depth tail, this leaves the v1-v10 resource record byte-compatible while making compressed
    // base allocations explicit to offline inspection. Metadata bytes are not inferred or invented.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_dcc_state = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources) {
            const auto& resource = captured.resource;
            if (resource.max_uncompressed_block_size > 3u ||
                resource.max_compressed_block_size > 3u) {
                error = "invalid resource DCC state"; return false;
            }
            const uint8_t flags = (resource.meta_pipe_aligned ? 1u : 0u) |
                                  (resource.write_compress_enabled ? 2u : 0u) |
                                  (resource.compression_enabled ? 4u : 0u) |
                                  (resource.alpha_is_on_msb ? 8u : 0u) |
                                  (resource.color_transform ? 16u : 0u);
            w.u8(flags);
            w.u32(resource.max_uncompressed_block_size);
            w.u32(resource.max_compressed_block_size);
            w.u64(resource.metadata_addr);
        }
        return true;
    };
    for (const auto& draw : c.draws)
        if (!write_dcc_state(draw.vrt) || !write_dcc_state(draw.prt)) return false;
    for (const auto& compute : c.computes)
        if (!write_dcc_state(compute.resources)) return false;
    // v12 captures the exact DCC control surface as a normal content-addressed blob range. Keep the
    // references in a tail so v1-v11 resource records remain byte-compatible. An invalid blob index
    // with a non-zero size is intentional for metadata-only and legacy-upgraded captures.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_dcc_metadata = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources) {
            const uint64_t expected = dcc_metadata_footprint(captured.resource);
            if (captured.metadata_size != expected ||
                (!expected && (captured.metadata_blob_index != 0xFFFFFFFFu ||
                               captured.metadata_blob_offset != 0)) ||
                (captured.metadata_blob_index == 0xFFFFFFFFu &&
                 captured.metadata_blob_offset != 0)) {
                error = "invalid resource DCC metadata reference"; return false;
            }
            if (captured.metadata_blob_index != 0xFFFFFFFFu) {
                if (captured.metadata_blob_index >= c.blobs.size()) {
                    error = "resource DCC metadata references an invalid capture blob"; return false;
                }
                const auto& capture_blob = c.blobs[captured.metadata_blob_index];
                const auto& blob = capture_blob.bytes;
                if (!capture_blob_payload_omitted(capture_blob) &&
                    (captured.metadata_blob_offset > blob.size() ||
                     expected > blob.size() - captured.metadata_blob_offset)) {
                    error = "resource DCC metadata exceeds its capture blob"; return false;
                }
            }
            w.u64(captured.metadata_size);
            w.u32(captured.metadata_blob_index);
            w.u64(captured.metadata_blob_offset);
        }
        return true;
    };
    for (const auto& draw : c.draws)
        if (!write_dcc_metadata(draw.vrt) || !write_dcc_metadata(draw.prt)) return false;
    for (const auto& compute : c.computes)
        if (!write_dcc_metadata(compute.resources)) return false;
    // v14 appends ordered address-backed DMA_DATA records. Both endpoints reference the same
    // content-addressed pre-submit blobs used by draw/compute resources, so replay mutations are
    // visible to later consumers without changing any v1-v13 prefix.
    w.u32(static_cast<uint32_t>(c.dma_copies.size()));
    for (const auto& copy : c.dma_copies) {
        w.u64(copy.dst); w.u64(copy.src); w.u32(copy.bytes); w.u32(copy.sels);
        w.u64(copy.command_order); w.u64(copy.packet_addr);
        w.u32(copy.destination_blob_index); w.u64(copy.destination_blob_offset);
        w.u32(copy.source_blob_index); w.u64(copy.source_blob_offset);
    }
    // v15 records whether an image is declared for depth comparison. Keep this in a deterministic
    // extension tail, just like the v9 depth and v11/v12 DCC additions, so every older resource
    // record remains byte-compatible.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_depth_compare = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources)
            w.u8(captured.resource.depth_compare);
    };
    for (const auto& draw : c.draws) {
        write_depth_compare(draw.vrt);
        write_depth_compare(draw.prt);
    }
    for (const auto& compute : c.computes) write_depth_compare(compute.resources);
    // v16 records packed GFX10 mip-tail placement. Tail siblings share one captured allocation block,
    // so replay must retain both the common base and the selected in-block byte origin.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_mip_tail = [&](const GpuCapturedTable& table, const char* owner,
                              size_t owner_index) {
        for (size_t resource_index = 0; resource_index < table.resources.size();
             ++resource_index) {
            const auto& captured = table.resources[resource_index];
            const auto& resource = captured.resource;
            if ((resource.in_mip_tail &&
                 (resource.mip_tail_bytes != 4096u && resource.mip_tail_bytes != 65536u)) ||
                (resource.in_mip_tail && resource.mip_tail_offset >= resource.mip_tail_bytes) ||
                (!resource.in_mip_tail &&
                 (resource.mip_tail_offset != 0 || resource.mip_tail_bytes != 0 ||
                  resource.mip_tail_x != 0 || resource.mip_tail_y != 0))) {
                char detail[512]{};
                std::snprintf(detail, sizeof(detail),
                              "invalid resource mip-tail state: %s[%zu] resource[%zu] "
                              "class=%u binding=%u addr=0x%llx size=%u enabled=%u "
                              "offset=%u bytes=%u xy=(%u,%u)",
                              owner, owner_index, resource_index,
                              static_cast<unsigned>(resource.cls), resource.binding,
                              static_cast<unsigned long long>(resource.gpu_addr), resource.size,
                              resource.in_mip_tail ? 1u : 0u, resource.mip_tail_offset,
                              resource.mip_tail_bytes, resource.mip_tail_x,
                              resource.mip_tail_y);
                error = detail;
                return false;
            }
            w.u8(resource.in_mip_tail);
            w.u32(resource.mip_tail_offset);
            w.u32(resource.mip_tail_bytes);
            w.u32(resource.mip_tail_x);
            w.u32(resource.mip_tail_y);
        }
        return true;
    };
    for (size_t draw_index = 0; draw_index < c.draws.size(); ++draw_index) {
        const auto& draw = c.draws[draw_index];
        if (!write_mip_tail(draw.vrt, "draw-vs", draw_index) ||
            !write_mip_tail(draw.prt, "draw-ps", draw_index))
            return false;
    }
    for (size_t compute_index = 0; compute_index < c.computes.size(); ++compute_index)
        if (!write_mip_tail(c.computes[compute_index].resources, "compute", compute_index))
            return false;
    // v17 appends the effective guest scissor for each realized/failed draw pipeline. Keeping this
    // state in a deterministic tail preserves the byte-exact v1-v16 pipeline prefix and lets old
    // captures materialize with the historical full-target default.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) write_scissor_pipeline(w, draw.ps);
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics)
        if (diagnostic.pipeline_present) write_scissor_pipeline(w, diagnostic.pipeline);
    // v18 appends the optional generated geometry-stage identity without changing any legacy draw
    // prefix. UINT32_MAX means the normal VS->FS fast path; otherwise the index names the shared
    // content-addressed SPIR-V version, just like the base VS/FS indices.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        const uint32_t geometry_index = draw.gs.empty()
            ? 0xFFFFFFFFu : shader_version_index(versions, draw.gs);
        if (!draw.gs.empty() && geometry_index == 0xFFFFFFFFu) {
            error = "draw geometry shader is missing from the version table"; return false;
        }
        w.u32(geometry_index);
    }
    // v19 retains the raw RDNA2 identity for both realized graphics stages. The raw streams share
    // the bounded content-addressed table introduced for v7 failure diagnostics; old captures leave
    // both indices unavailable instead of inventing source bytes.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        w.u32(draw.vs_raw_shader_index);
        w.u32(draw.fs_raw_shader_index);
    }
    // v20 retains the hardware instance count per realized draw. Older captures default to one,
    // matching the command-processor and Vulkan defaults before IT_NUM_INSTANCES was consumed.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) w.u32(draw.instance_count);
    // v21 appends the resolved framebuffer logic operation for realized and failed draw pipelines.
    // Older captures retain the historical disabled/COPY default.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) write_logic_op_pipeline(w, draw.ps);
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics)
        if (diagnostic.pipeline_present) write_logic_op_pipeline(w, diagnostic.pipeline);
    // v22 explicitly captures host-backed compute GDS state. Unlike guest-addressed resources,
    // this state has no valid capture interval; one shared replay instance preserves dispatch order.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_internal_state = [&](const GpuCapturedTable& table,
                                    bool allow_packed_pointer) {
        for (const auto& captured : table.resources) {
            const bool gds = is_compute_internal_gds(captured.resource);
            const bool packed = is_gta5_packed_pointer_serialized_shadow(
                captured.resource, captured.internal_bytes.data(),
                captured.internal_bytes.size());
            const bool relocated = is_indirect_pointer_relocation_serialized(
                captured.resource, captured.internal_bytes.data(),
                captured.internal_bytes.size());
            const bool expected_packed =
                is_gta5_packed_pointer_marker_candidate(captured.resource);
            const bool expected_relocated =
                is_indirect_pointer_relocation_marker_candidate(captured.resource);
            if ((!allow_packed_pointer &&
                 (packed || expected_packed || relocated || expected_relocated)) ||
                (!gds && !packed && !relocated && !captured.internal_bytes.empty()) ||
                (gds && !captured.internal_bytes.empty() &&
                 captured.internal_bytes.size() != resource_footprint(captured.resource)) ||
                (expected_packed && !captured.internal_bytes.empty() && !packed) ||
                (expected_relocated && !captured.internal_bytes.empty() && !relocated)) {
                error = "invalid compute internal capture state";
                return false;
            }
            w.bytes(captured.internal_bytes);
        }
        return true;
    };
    for (const auto& draw : c.draws)
        if (!write_internal_state(draw.vrt, false) ||
            !write_internal_state(draw.prt, false)) return false;
    for (const auto& compute : c.computes)
        if (!write_internal_state(compute.resources, true)) return false;
    // v23 (#1256) appends the raw draw-packet state (DrawIndexAuto/DrawIndex index_count + indexed flag)
    // per realized draw, decoded from the guest BEFORE realization. Older captures default to 0/false
    // ("unknown"). Kept as a trailing block so every v1-v22 record prefix stays byte-exact.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) { w.u32(draw.raw_draw_count); w.u32(draw.raw_indexed ? 1u : 0u); }
    // v24 (#1280) appends each resource's T#-declared mip-chain length (declared_mip_levels), used by the
    // backend to bound generated-mip uploads (#1272). Kept as a trailing per-resource block (same resource
    // enumeration as the v16 mip-tail tail) so every v1-v23 prefix stays byte-exact and older captures
    // materialize with the historical default of 1.
    {
        size_t resource_count = 0;
        for (const auto& draw : c.draws)
            resource_count += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            resource_count += compute.resources.resources.size();
        w.u32(static_cast<uint32_t>(resource_count));
        auto write_declared_mips = [&](const GpuCapturedTable& table) {
            for (const auto& captured : table.resources) w.u32(captured.resource.declared_mip_levels);
        };
        for (const auto& draw : c.draws) { write_declared_mips(draw.vrt); write_declared_mips(draw.prt); }
        for (const auto& compute : c.computes) write_declared_mips(compute.resources);
    }
    // v25 (#1240) appends the resolved MODE=3 MSAA-copy intent. gpu_replay drives the same renderer
    // callback as live execution; without this bit it silently skips color0 -> color1 resolves.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) w.u8(draw.ps.cb_resolve ? 1u : 0u);
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics)
        if (diagnostic.pipeline_present) w.u8(diagnostic.pipeline.cb_resolve ? 1u : 0u);
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        w.u32(draw.ps.spi_shader_col_format);
        w.u32(draw.ps.sx_ps_downconvert);
    }
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) if (diagnostic.pipeline_present) {
        w.u32(diagnostic.pipeline.spi_shader_col_format);
        w.u32(diagnostic.pipeline.sx_ps_downconvert);
    }
    // v27: raw draw modifier plus GE_INDX_OFFSET-derived Vulkan draw parameter. Keep this in a
    // trailing block so v1-v26 prefixes remain byte-exact and older captures default both to zero.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        w.u64(draw.raw_draw_modifier);
        w.u32(static_cast<uint32_t>(draw.vertex_offset));
    }
    // v28: exact source row pitch and capture-planned byte span for each resource. A zero pitch remains
    // meaningful for resources that are not linear sampled images; enumeration matches v16/v24.
    {
        size_t resource_count = 0;
        for (const auto& draw : c.draws)
            resource_count += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            resource_count += compute.resources.resources.size();
        w.u32(static_cast<uint32_t>(resource_count));
        auto write_linear_layout = [&](const GpuCapturedTable& table) {
            for (const auto& captured : table.resources) {
                ShaderResource resource = captured.resource;
                if (c.format_version < 28 && !resource.linear_row_pitch_bytes &&
                    resource.cls == ResourceClass::Texture && resource.img_dim == 1u &&
                    resource.tile_mode == static_cast<uint32_t>(TileMode::Linear) &&
                    !resource.compression_enabled && !bc_block_bytes(resource.format)) {
                    uint32_t bpt = data_format_bytes(resource.format) *
                                   (resource.num_components ? resource.num_components : 1u);
                    const bool f16 = resource.format == DataFormat::Float16 &&
                                     (bpt == 2 || bpt == 4 || bpt == 8);
                    if (bpt == 0 || (bpt > 4 && !f16)) bpt = 4;
                    resource.linear_row_pitch_bytes = resolved_linear_row_pitch(
                        resource, resource.width ? resource.width : 4u, bpt);
                }
                w.u32(resource.linear_row_pitch_bytes);
                w.u64(captured.captured_size ? captured.captured_size
                                             : (c.format_version < 28
                                                    ? legacy_resource_footprint(resource)
                                                    : resource_footprint(resource)));
            }
        };
        for (const auto& draw : c.draws) {
            write_linear_layout(draw.vrt);
            write_linear_layout(draw.prt);
        }
        for (const auto& compute : c.computes) write_linear_layout(compute.resources);
    }
    // v29 (#1349) appends the resolved depth-bias state per realized/failed pipeline. Shadow-map
    // passes program PA_SU_POLY_OFFSET_*; replaying without it re-introduces the acne the live
    // renderer now avoids.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        w.u8(draw.ps.depth_bias_enable ? 1u : 0u);
        w.u32(std::bit_cast<uint32_t>(draw.ps.depth_bias_constant));
        w.u32(std::bit_cast<uint32_t>(draw.ps.depth_bias_slope));
        w.u32(std::bit_cast<uint32_t>(draw.ps.depth_bias_clamp));
    }
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) if (diagnostic.pipeline_present) {
        w.u8(diagnostic.pipeline.depth_bias_enable ? 1u : 0u);
        w.u32(std::bit_cast<uint32_t>(diagnostic.pipeline.depth_bias_constant));
        w.u32(std::bit_cast<uint32_t>(diagnostic.pipeline.depth_bias_slope));
        w.u32(std::bit_cast<uint32_t>(diagnostic.pipeline.depth_bias_clamp));
    }

    // v30 appends the dynamic-fold-proven fetch index source in deterministic resource order. NGG
    // merged shaders can select instance_id for one V# and vertex_id for another; replay must compile
    // the same address model as the live draw.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_fetch_index_modes = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources)
            w.u32(static_cast<uint32_t>(captured.resource.fetch_index_mode));
    };
    for (const auto& draw : c.draws) {
        write_fetch_index_modes(draw.vrt);
        write_fetch_index_modes(draw.prt);
    }
    for (const auto& compute : c.computes) write_fetch_index_modes(compute.resources);
    // v31 retains a separately-installed NGG vertex main stage and the exact graphics-LDS
    // allocation. This lets diagnostic replay reconstruct linked prolog+main programs instead of
    // incorrectly probing the fetch prolog alone. Older captures keep both fields unavailable.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        w.u32(draw.vs_chain_raw_shader_index);
        w.u32(draw.vertex_lds_dwords);
    }
    // v32 keeps the selected mip's placement inside every thin-2D array or cube slice. This is a
    // trailing block so v1-v31 records remain byte-exact and old captures retain their tight-layer
    // default.
    w.u32(static_cast<uint32_t>(resource_depth_count));
    auto write_layer_layout = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources) {
            const ShaderResource& resource = captured.resource;
            const bool enabled = resource.layer_stride_bytes != 0;
            if ((!enabled && resource.layer_mip_offset_bytes != 0) ||
                (enabled && ((resource.img_dim != 3 && resource.img_dim != 5) ||
                             resource.depth < 2 ||
                             resource.layer_mip_offset_bytes >= resource.layer_stride_bytes ||
                             (resource.in_mip_tail && resource.layer_mip_offset_bytes != 0)))) {
                error = "invalid resource layered mip layout";
                return false;
            }
            w.u32(resource.layer_stride_bytes);
            w.u32(resource.layer_mip_offset_bytes);
        }
        return true;
    };
    for (const auto& draw : c.draws)
        if (!write_layer_layout(draw.vrt) || !write_layer_layout(draw.prt)) return false;
    for (const auto& compute : c.computes)
        if (!write_layer_layout(compute.resources)) return false;
    // v34 retains all hardware color-buffer slots. MRT0/MRT1 keep their byte-exact historical
    // records; this tail adds slots 2..7 so live and standalone replay execute the same deferred
    // rendering graph instead of silently discarding G-buffer exports.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        for (uint32_t slot = 2; slot < kColorTargetCount; ++slot) {
            const auto& binding = draw.color_targets[slot];
            w.u64(binding.base); w.u32(binding.width); w.u32(binding.height);
            write_color_target_pipeline(w, draw.ps.color_targets[slot]);
        }
    }
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) {
        for (uint32_t slot = 2; slot < kColorTargetCount; ++slot) {
            const auto& binding = diagnostic.color_targets[slot];
            w.u64(binding.base); w.u32(binding.width); w.u32(binding.height);
        }
        if (diagnostic.pipeline_present)
            for (uint32_t slot = 2; slot < kColorTargetCount; ++slot)
                write_color_target_pipeline(w, diagnostic.pipeline.color_targets[slot]);
    }
    // v35 keeps failed-stage descriptor metadata in a self-contained tail. The ordinary table
    // prefix already includes every field the recompiler uses for binding/provenance decisions;
    // blob references are intentionally unset because a failed stage is retried, not rendered.
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) {
        w.u32(static_cast<uint32_t>(diagnostic.stages.size()));
        for (const auto& stage : diagnostic.stages) write_table(w, stage.resource_table);
    }
    // v36 keeps the exact pixel-stage ABI used for the stored modules. Fields are conditional so a
    // draw with no programmed SPI input state remains explicit instead of gaining an invented zero
    // mapping during replay.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) {
        if ((draw.has_pixel_inputs && !draw.pixel_inputs.valid_mask) ||
            (draw.pixel_inputs.passthrough_mask & ~draw.pixel_inputs.valid_mask) ||
            (draw.has_system_inputs && !draw.system_inputs.ena && !draw.system_inputs.addr)) {
            error = "invalid realized-draw pixel-stage ABI";
            return false;
        }
        const uint8_t flags = (draw.has_pixel_inputs ? 1u : 0u) |
                              (draw.has_system_inputs ? 2u : 0u);
        w.u8(flags);
        if (draw.has_pixel_inputs) {
            w.u32(draw.pixel_inputs.valid_mask);
            w.u32(draw.pixel_inputs.passthrough_mask);
            for (uint32_t control : draw.pixel_inputs.controls) w.u32(control);
        }
        if (draw.has_system_inputs) {
            w.u32(draw.system_inputs.ena);
            w.u32(draw.system_inputs.addr);
        }
    }
    // v37 keeps the pipeline-stage contract beside the already-stored native-subgroup module.
    // The realized translator only emits wave32/wave64 modules; reject invented values here so a
    // malformed capsule cannot ask Vulkan for an unrelated device-specific subgroup width.
    w.u32(static_cast<uint32_t>(c.computes.size()));
    for (const auto& compute : c.computes) {
        if (compute.required_subgroup_size != 0 && compute.required_subgroup_size != 32 &&
            compute.required_subgroup_size != 64) {
            error = "invalid compute required-subgroup size";
            return false;
        }
        w.u32(compute.required_subgroup_size);
    }
    // v39 makes realized compute modules rebuildable. Keep this after every historical tail so
    // lowering the version and removing this block reproduces a byte-valid v38 capsule.
    w.u32(static_cast<uint32_t>(c.computes.size()));
    for (const auto& compute : c.computes) {
        if (!validate_compute_recompile_state(compute, error)) return false;
        w.u8(compute.recompile_config_available ? 1u : 0u);
        w.u32(compute.raw_shader_index);
        if (!compute.recompile_config_available) continue;
        write_compute_config(w, compute.recompile_config);
    }
    // v40 keeps the BVH BOX_GROW descriptor field in deterministic resource-table order. Failed
    // stages are included because raw diagnostic replay may recompile those tables later.
    uint64_t bvh_resource_count = resource_depth_count;
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            bvh_resource_count += stage.resource_table.resources.size();
    if (bvh_resource_count > UINT32_MAX) {
        error = "invalid BVH resource-state count";
        return false;
    }
    w.u32(static_cast<uint32_t>(bvh_resource_count));
    auto write_bvh_box_grow = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources) {
            if (captured.resource.bvh_box_grow > 0xFFu) return false;
            w.u32(captured.resource.bvh_box_grow);
        }
        return true;
    };
    for (const auto& draw : c.draws)
        if (!write_bvh_box_grow(draw.vrt) || !write_bvh_box_grow(draw.prt)) {
            error = "invalid BVH box-grow value";
            return false;
        }
    for (const auto& compute : c.computes)
        if (!write_bvh_box_grow(compute.resources)) {
            error = "invalid BVH box-grow value";
            return false;
        }
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            if (!write_bvh_box_grow(stage.resource_table)) {
                error = "invalid BVH box-grow value";
                return false;
            }
    // v41 completes the resource records embedded by v35. Those tables deliberately carry no
    // resource blobs, but retry still needs every descriptor/codegen field that the older v9-v32
    // extension tails wrote only for realized draw/compute tables. Keep one bounded count followed
    // by fixed-size records in the already-stable failure/stage/resource order. BOX_GROW is already
    // retained for failed stages by v40.
    uint64_t failed_resource_count = 0;
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            failed_resource_count += stage.resource_table.resources.size();
    if (failed_resource_count > UINT32_MAX) {
        error = "invalid failed-stage resource-state count";
        return false;
    }
    w.u32(static_cast<uint32_t>(failed_resource_count));
    auto write_failed_resource_state = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources) {
            const ShaderResource& resource = captured.resource;
            if (!resource.depth || resource.depth > 8192u ||
                !resource.declared_mip_levels) {
                error = "invalid failed-stage resource image state";
                return false;
            }
            if (resource.max_uncompressed_block_size > 3u ||
                resource.max_compressed_block_size > 3u) {
                error = "invalid failed-stage resource DCC state";
                return false;
            }
            const uint8_t dcc_flags = (resource.meta_pipe_aligned ? 1u : 0u) |
                                      (resource.write_compress_enabled ? 2u : 0u) |
                                      (resource.compression_enabled ? 4u : 0u) |
                                      (resource.alpha_is_on_msb ? 8u : 0u) |
                                      (resource.color_transform ? 16u : 0u);
            const bool mip_tail_valid =
                (!resource.in_mip_tail && resource.mip_tail_offset == 0 &&
                 resource.mip_tail_bytes == 0 && resource.mip_tail_x == 0 &&
                 resource.mip_tail_y == 0) ||
                (resource.in_mip_tail &&
                 (resource.mip_tail_bytes == 4096u || resource.mip_tail_bytes == 65536u) &&
                 resource.mip_tail_offset < resource.mip_tail_bytes);
            if (!mip_tail_valid) {
                error = "invalid failed-stage resource mip-tail state";
                return false;
            }
            if (static_cast<uint32_t>(resource.fetch_index_mode) >
                static_cast<uint32_t>(VertexFetchIndexMode::Instance)) {
                error = "invalid failed-stage resource fetch-index mode";
                return false;
            }
            const bool layered = resource.layer_stride_bytes != 0;
            if ((!layered && resource.layer_mip_offset_bytes != 0) ||
                (layered && ((resource.img_dim != 3 && resource.img_dim != 5) ||
                             resource.depth < 2 ||
                             resource.layer_mip_offset_bytes >= resource.layer_stride_bytes ||
                             (resource.in_mip_tail &&
                              resource.layer_mip_offset_bytes != 0)))) {
                error = "invalid failed-stage resource layered mip layout";
                return false;
            }
            w.u32(resource.depth);
            w.u8(dcc_flags);
            w.u32(resource.max_uncompressed_block_size);
            w.u32(resource.max_compressed_block_size);
            w.u64(resource.metadata_addr);
            w.u8(resource.depth_compare ? 1u : 0u);
            w.u8(resource.in_mip_tail ? 1u : 0u);
            w.u32(resource.mip_tail_offset);
            w.u32(resource.mip_tail_bytes);
            w.u32(resource.mip_tail_x);
            w.u32(resource.mip_tail_y);
            w.u32(resource.declared_mip_levels);
            w.u32(resource.linear_row_pitch_bytes);
            w.u64(captured.captured_size);
            w.u32(static_cast<uint32_t>(resource.fetch_index_mode));
            w.u32(resource.layer_stride_bytes);
            w.u32(resource.layer_mip_offset_bytes);
            w.u32(resource.flat_base_sgpr);
        }
        return true;
    };
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            if (!write_failed_resource_state(stage.resource_table)) return false;
    // v42 retains the exact launch/user-SGPR specialization passed to a failed compute
    // translation. Failed dispatches never enter the realized-compute list, so v39 could not
    // preserve this state and offline retry had to refuse rather than invent an ABI. Keep the
    // append-only tail in diagnostic/stage order so v1-v41 readers remain reproducible by removing
    // only this block and lowering the version.
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics) {
        w.u32(static_cast<uint32_t>(diagnostic.stages.size()));
        for (const auto& stage : diagnostic.stages) {
            w.u8(stage.recompile_config_available ? 1u : 0u);
            if (stage.recompile_config_available)
                write_compute_config(w, stage.recompile_config);
        }
    }
    // v43 (#1459) appends the raw color-state triple that determines every resolved color write
    // mask. Presence travels with each value because an absent CB_TARGET_MASK/CB_SHADER_MASK means
    // write-all, not zero. Enumeration matches the v25/v26 blocks so the tail stays removable.
    w.u32(static_cast<uint32_t>(c.draws.size()));
    for (const auto& draw : c.draws) write_color_state(w, draw.ps);
    w.u32(static_cast<uint32_t>(c.failure_diagnostics.size()));
    for (const auto& diagnostic : c.failure_diagnostics)
        if (diagnostic.pipeline_present) write_color_state(w, diagnostic.pipeline);
    // v44 appends guest sample counts in deterministic resource-table order. Failed-stage tables are
    // included because offline raw retry consumes their descriptor identity just like realized tables.
    const uint64_t sample_resource_count = resource_depth_count + failed_resource_count;
    if (sample_resource_count > UINT32_MAX) {
        error = "invalid resource sample-count count";
        return false;
    }
    w.u32(static_cast<uint32_t>(sample_resource_count));
    auto write_sample_counts = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources) {
            const uint32_t samples = captured.resource.sample_count;
            if (!samples || samples > 32768u || (samples & (samples - 1u)) != 0) return false;
            w.u32(samples);
        }
        return true;
    };
    for (const auto& draw : c.draws)
        if (!write_sample_counts(draw.vrt) || !write_sample_counts(draw.prt)) {
            error = "invalid resource sample count"; return false;
        }
    for (const auto& compute : c.computes)
        if (!write_sample_counts(compute.resources)) {
            error = "invalid resource sample count"; return false;
        }
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            if (!write_sample_counts(stage.resource_table)) {
                error = "invalid resource sample count"; return false;
            }
    // v45 appends at most one explicitly selected temporal resource audit. Both samples are ordinary
    // capture blobs, so their independent bytes_read fields and payload hashes remain self-contained.
    if (!validate_resource_provenance(c, error)) return false;
    w.u32(static_cast<uint32_t>(c.resource_provenance.size()));
    for (const auto& provenance : c.resource_provenance) {
        w.u64(provenance.draw_index);
        w.u8(static_cast<uint8_t>(provenance.stage));
        w.u32(provenance.binding);
        w.u32(static_cast<uint32_t>(provenance.resource_class));
        w.u64(provenance.guest_addr);
        w.u64(provenance.requested_bytes);
        w.u32(provenance.srt_offset);
        w.u32(provenance.sgpr_base);
        w.u32(provenance.realization_blob_index);
        w.u32(provenance.post_blob_index);
        w.u64(provenance.post_blob_offset);
        w.u64(provenance.realization_content_hash);
        w.u64(provenance.post_content_hash);
    }
    // v46 appends the input-side witness separately so every v45 record remains byte-exact. Repeat
    // semantic identity before the raw dwords: a reordered vector must fail visibly instead of
    // silently assigning one draw's witness to another normalized descriptor.
    w.u32(static_cast<uint32_t>(c.resource_provenance.size()));
    for (const auto& provenance : c.resource_provenance) {
        w.u64(provenance.draw_index);
        w.u8(static_cast<uint8_t>(provenance.stage));
        w.u32(provenance.binding);
        w.u8(static_cast<uint8_t>(provenance.input_mode));
        w.u8(static_cast<uint8_t>(provenance.input_unavailable_reason));
        w.u8(provenance.input_write_provenance_mask);
        w.u32(provenance.input_sh_register_base);
        for (const uint32_t value : provenance.input_dwords) w.u32(value);
        for (const uint64_t write : provenance.input_last_writes) w.u64(write);
        for (const uint64_t source : provenance.input_write_sources) w.u64(source);
    }
    // v47 preserves the instruction-scoped zero-mip specialization marker for every realized and
    // failed-stage resource. Raw replay can recompile captured RDNA2, so omitting this bit would let
    // it silently reuse the level-zero lowering for an instruction whose mip value was never proven.
    // Keep the same complete resource enumeration as v44's sample-count tail.
    w.u32(static_cast<uint32_t>(sample_resource_count));
    auto write_zero_mip_state = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources)
            w.u8(captured.resource.proven_zero_mip ? 1u : 0u);
    };
    for (const auto& draw : c.draws) {
        write_zero_mip_state(draw.vrt);
        write_zero_mip_state(draw.prt);
    }
    for (const auto& compute : c.computes) write_zero_mip_state(compute.resources);
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            write_zero_mip_state(stage.resource_table);
    // v48 preserves the BVH sort semantic separately from the v40 BOX_GROW tail so every older
    // capture prefix remains byte-exact. Non-BVH resources carry false in the same deterministic
    // complete-resource enumeration used by v44/v47.
    w.u32(static_cast<uint32_t>(sample_resource_count));
    auto write_bvh_sort_state = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources)
            w.u8(captured.resource.bvh_sort_enabled ? 1u : 0u);
    };
    for (const auto& draw : c.draws) {
        write_bvh_sort_state(draw.vrt);
        write_bvh_sort_state(draw.prt);
    }
    for (const auto& compute : c.computes) write_bvh_sort_state(compute.resources);
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            write_bvh_sort_state(stage.resource_table);
    // v49 retains the exact qword record count in the same complete resource enumeration.
    w.u32(static_cast<uint32_t>(sample_resource_count));
    auto write_atomic_x2_state = [&](const GpuCapturedTable& table) {
        for (const auto& captured : table.resources)
            w.u32(captured.resource.atomic_x2_record_count);
    };
    for (const auto& draw : c.draws) {
        write_atomic_x2_state(draw.vrt);
        write_atomic_x2_state(draw.prt);
    }
    for (const auto& compute : c.computes) write_atomic_x2_state(compute.resources);
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            write_atomic_x2_state(stage.resource_table);
    // v50 retains the launch register only for configurations that exist. Keep this append-only:
    // COMPUTE_PGM_RSRC1 became semantically necessary after v42 embedded the config records, and
    // inserting it into those historical records would make every v42-v49 suffix non-removable.
    uint64_t compute_pgm_rsrc1_count = 0;
    for (const auto& compute : c.computes)
        compute_pgm_rsrc1_count += compute.recompile_config_available;
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            compute_pgm_rsrc1_count += stage.recompile_config_available;
    if (compute_pgm_rsrc1_count > UINT32_MAX) {
        error = "invalid compute PGM_RSRC1 state count";
        return false;
    }
    w.u32(static_cast<uint32_t>(compute_pgm_rsrc1_count));
    for (const auto& compute : c.computes)
        if (compute.recompile_config_available)
            w.u32(compute.recompile_config.compute_pgm_rsrc1);
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            if (stage.recompile_config_available)
                w.u32(stage.recompile_config.compute_pgm_rsrc1);
    // v52 appends the complete buffer descriptor-array payload. The base resource record remains
    // byte-identical for every prior version; old captures therefore reopen as ordinary scalar
    // resources, while a v52 file must carry one concrete V# and backing reference per declared slot.
    w.u32(static_cast<uint32_t>(sample_resource_count));
    auto write_buffer_tables = [&](const GpuCapturedTable& table) {
        for (const GpuCapturedResource& captured : table.resources) {
            const ShaderResource& resource = captured.resource;
            if (!valid_shader_buffer_table_contract(resource) ||
                (resource.table_index_count == 0u && !captured.table_entry_blobs.empty()) ||
                (resource.table_index_count != 0u &&
                 captured.table_entry_blobs.size() != resource.table_entries.size())) {
                error = "invalid captured buffer descriptor-table contract";
                return false;
            }
            w.u32(resource.table_index_count);
            w.u32(resource.table_entry_stride);
            w.u32(resource.table_index_sgpr);
            w.u32(static_cast<uint32_t>(resource.table_selector_mode));
            w.u32(resource.table_load_pc);
            w.u32(static_cast<uint32_t>(resource.table_entries.size()));
            for (size_t index = 0; index < resource.table_entries.size(); ++index) {
                const ShaderBufferTableEntry& entry = resource.table_entries[index];
                const auto& backing = captured.table_entry_blobs[index];
                if ((backing.blob_index == UINT32_MAX && backing.blob_offset != 0u) ||
                    (backing.blob_index != UINT32_MAX &&
                     (backing.blob_index >= c.blobs.size() ||
                      (!capture_blob_payload_omitted(c.blobs[backing.blob_index]) &&
                       (backing.blob_offset > c.blobs[backing.blob_index].bytes.size() ||
                        entry.size > c.blobs[backing.blob_index].bytes.size() -
                                         backing.blob_offset))))) {
                    error = "invalid captured buffer descriptor-table backing";
                    return false;
                }
                for (const uint32_t word : entry.vsharp) w.u32(word);
                w.u64(entry.gpu_addr);
                w.u32(entry.size);
                w.u32(entry.stride);
                w.u32(backing.blob_index);
                w.u64(backing.blob_offset);
            }
        }
        return true;
    };
    for (const auto& draw : c.draws)
        if (!write_buffer_tables(draw.vrt) || !write_buffer_tables(draw.prt)) return false;
    for (const auto& compute : c.computes)
        if (!write_buffer_tables(compute.resources)) return false;
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            if (!write_buffer_tables(stage.resource_table)) return false;
    // v54 appends one scalar-buffer dword count per resource. Zero is the ordinary/no-metadata
    // value; a nonzero value is serialized only after its normalized resource shape independently
    // authenticates the bound.
    w.u32(static_cast<uint32_t>(sample_resource_count));
    auto write_scalar_buffer_bounds = [&](const GpuCapturedTable& table) {
        for (const GpuCapturedResource& captured : table.resources) {
            const ShaderResource& resource = captured.resource;
            if (resource.scalar_buffer_dword_count) {
                const uint64_t binding_bytes =
                    shader_resource_buffer_binding_bytes(resource);
                if (!binding_bytes) {
                    error = "invalid scalar-buffer bound metadata";
                    return false;
                }
                if (captured.captured_size < binding_bytes) {
                    error = "scalar-buffer captured span is shorter than its bound";
                    return false;
                }
            }
            w.u32(resource.scalar_buffer_dword_count);
        }
        return true;
    };
    for (const auto& draw : c.draws)
        if (!write_scalar_buffer_bounds(draw.vrt) ||
            !write_scalar_buffer_bounds(draw.prt)) return false;
    for (const auto& compute : c.computes)
        if (!write_scalar_buffer_bounds(compute.resources)) return false;
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            if (!write_scalar_buffer_bounds(stage.resource_table)) return false;
    if (w.data.size() > kMaxFileBytes) { error = "capture file exceeds 4 GiB"; return false; }
    // v55: each DS seed's array slice, in the same order as the seed records. Written at the END of
    // the stream, not beside the records it belongs to.
    //
    // That placement is the whole point. Every version-gated addition here is a trailing tail so a
    // reader at version N can stop before it, and the test suite exercises exactly that by
    // serializing at the current version, LOWERING THE VERSION BYTE IN PLACE, and reparsing. A tail
    // written mid-stream survives its own round trip and desynchronises every later tail the moment
    // the version is downgraded -- eleven legacy-reopen assertions failed that way when this field
    // was first placed next to the seed records.
    for (const auto& seed : c.ds_seeds) w.u32(seed.slice);
    // v57 tail: allocation-wide mip placement per resource, in the same table order every other
    // per-resource tail uses (#3048).
    auto write_mip_chain_provenance = [&](const GpuCapturedTable& table) {
        for (const GpuCapturedResource& captured : table.resources) {
            const ShaderResource& resource = captured.resource;
            w.u32(resource.mip_chain_element_width);
            w.u32(resource.mip_chain_element_height);
            w.u32(resource.mip_chain_bytes_per_block);
            w.u32(resource.mip_chain_max_level);
            w.u32(resource.mip_chain_base_level);
        }
    };
    for (const auto& draw : c.draws) {
        write_mip_chain_provenance(draw.vrt);
        write_mip_chain_provenance(draw.prt);
    }
    for (const auto& compute : c.computes) write_mip_chain_provenance(compute.resources);
    for (const auto& diagnostic : c.failure_diagnostics)
        for (const auto& stage : diagnostic.stages)
            write_mip_chain_provenance(stage.resource_table);
    // Re-check the ceiling AFTER the final tail. The bound above was enforced before this tail
    // existed, so a capture sitting just under the maximum could serialize successfully into a file
    // that read_gpu_capture then rejects as oversized -- a write that reports success and produces
    // an unreadable artifact. Any tail added after this point must move this check again.
    if (w.data.size() > kMaxFileBytes) { error = "capture file exceeds 4 GiB"; return false; }
    bytes = std::move(w.data);
    return true;
}

bool deserialize_gpu_capture(const std::vector<uint8_t>& bytes, GpuCaptureFile& c, std::string& error) {
    error.clear(); c = {};
    Reader r{bytes.data(), bytes.size(), &error}; char magic[8]; uint32_t version, endian;
    if (!r.take(magic, 8)) { error = "truncated capture header"; return false; }
    if (std::memcmp(magic, kMagic, 8)) { error = "invalid capture magic"; return false; }
    if (!r.u32(version)) { error = "truncated capture version"; return false; }
    if (version < 1 || version > kVersion) {
        error = "unsupported capture version " + std::to_string(version);
        return false;
    }
    c.format_version = version;
    if (!r.u32(endian)) { error = "truncated capture byte-order marker"; return false; }
    if (endian != kEndian) { error = "unsupported capture byte order"; return false; }
    auto& m = c.metadata;
    if (!r.u32(m.width) || !r.u32(m.height) || !r.u64(m.submit_index) || !r.string(m.revision) ||
        !r.string(m.title_id) || !r.string(m.input_route) || !r.string(m.savedata_dir)) return false;
    uint32_t ne; if (!r.u32(ne) || ne > 256) { error = "invalid renderer environment count"; return false; }
    m.renderer_env.resize(ne);
    for (auto& [name, value] : m.renderer_env) if (!r.string(name) || !r.string(value)) return false;
    if (version >= 5) {
        uint8_t valid = 0;
        if (!r.u8(valid)) return false;
        c.expected_output_valid = valid != 0;
    } else {
        c.expected_output_valid = true;
    }
    if (!r.u64(c.expected_output_hash) || !r.u64(c.expected_output_bytes)) return false;
    uint32_t nb; if (!r.u32(nb) || nb > kMaxResources) { error = "invalid blob count"; return false; }
    uint64_t total = 0; c.blobs.resize(nb);
    for (auto& b : c.blobs) {
        if (!r.u64(b.guest_addr) || !r.u64(b.bytes_read) ||
            (version >= 5 && !r.u64(b.content_hash)) || !r.bytes(b.bytes)) return false;
        if (b.bytes_read > b.bytes.size() || total > kMaxTotalBlobBytes - b.bytes.size()) { error = "invalid blob metadata"; return false; }
        const uint64_t actual_hash = gpu_capture_hash(b.bytes);
        if (version >= 5 && b.content_hash != actual_hash) { error = "capture blob content hash mismatch"; return false; }
        b.content_hash = actual_hash;
        total += b.bytes.size();
    }
    if (version >= 4) {
        uint32_t ns; if (!r.u32(ns) || ns > kMaxResources) { error = "invalid RTT seed count"; return false; }
        uint64_t seed_total = 0; c.rtt_seeds.resize(ns);
        std::unordered_set<uint64_t> addresses;
        for (auto& seed : c.rtt_seeds) {
            uint32_t format = static_cast<uint32_t>(GpuCaptureColorFormat::Rgba8Unorm);
            if (!r.u64(seed.guest_addr) || !r.u32(seed.width) || !r.u32(seed.height) ||
                (version >= 13 && !r.u32(format)) || !r.bytes(seed.rgba)) return false;
            seed.format = static_cast<GpuCaptureColorFormat>(format);
            if (!validate_rtt_seed(seed, error)) return false;
            if (!addresses.insert(seed.guest_addr).second) { error = "duplicate RTT seed address"; return false; }
            if (seed_total > kMaxTotalRttSeedBytes - seed.rgba.size()) {
                error = "capture RTT seed data exceeds 1 GiB"; return false;
            }
            seed_total += seed.rgba.size();
        }
    }
    if (version >= 5) {
        uint32_t ns = 0;
        if (!r.u32(ns) || ns > kMaxResources) { error = "invalid shader-version count"; return false; }
        c.shader_versions.resize(ns);
        for (auto& shader : c.shader_versions) {
            if (!r.u64(shader.content_hash) || !r.words(shader.words)) return false;
            if (shader.content_hash != shader_hash(shader.words)) {
                error = "capture shader content hash mismatch"; return false;
            }
        }
    }
    uint32_t nd; if (!r.u32(nd) || nd > kMaxDraws || (version < 5 && !nd)) { error = "invalid draw count"; return false; }
    c.draws.resize(nd);
    for (auto& d : c.draws) {
        if (version >= 5) {
            uint32_t vs_index = 0, fs_index = 0;
            if (!r.u32(vs_index) || !r.u32(fs_index) || vs_index >= c.shader_versions.size() ||
                fs_index >= c.shader_versions.size()) { error = "invalid draw shader-version index"; return false; }
            d.vs = c.shader_versions[vs_index].words;
            d.fs = c.shader_versions[fs_index].words;
        } else if (!r.words(d.vs) || !r.words(d.fs)) return false;
        if (!read_pipeline(r, d.ps, version) || !read_table(r, d.vrt, version) ||
            !read_table(r, d.prt, version) || !r.u32(d.vertex_count) || !r.words(d.indices) || !r.u64(d.color0_base)) return false;
        if (version >= 3 && (!r.u32(d.color0_width) || !r.u32(d.color0_height))) return false;
        if (version >= 5 && (!r.u64(d.draw_index) || !r.u64(d.command_order))) return false;
    }
    if (version >= 5) {
        uint32_t nc = 0;
        if (!r.u32(nc) || nc > kMaxComputes) { error = "invalid compute count"; return false; }
        c.computes.resize(nc);
        for (auto& compute : c.computes) {
            uint32_t shader_index = 0;
            if (!r.u32(shader_index) || shader_index >= c.shader_versions.size()) {
                error = "invalid compute shader-version index"; return false;
            }
            compute.spirv = c.shader_versions[shader_index].words;
            if (!read_table(r, compute.resources, version) ||
                !r.u32(compute.launch.threads_x) || !r.u32(compute.launch.threads_y) || !r.u32(compute.launch.threads_z) ||
                !r.u32(compute.launch.local_x) || !r.u32(compute.launch.local_y) || !r.u32(compute.launch.local_z) ||
                !r.u32(compute.launch.groups_x) || !r.u32(compute.launch.groups_y) || !r.u32(compute.launch.groups_z) ||
                !r.u64(compute.code_addr) || !r.u64(compute.dispatch_index) || !r.u64(compute.submit_no) ||
                !r.u64(compute.command_order)) return false;
        }
        uint32_t no = 0;
        if (!r.u32(no) || no > kMaxOperations) { error = "invalid operation count"; return false; }
        c.operations.resize(no);
        for (auto& operation : c.operations) {
            uint8_t kind = 0, realized = 0;
            const uint8_t max_kind = static_cast<uint8_t>(version >= 14
                ? SubmitOperationKind::DmaCopy : SubmitOperationKind::Dispatch);
            if (!r.u8(kind) || kind > max_kind ||
                !r.u64(operation.source_index) || !r.u64(operation.command_order) || !r.u8(realized)) return false;
            operation.kind = static_cast<SubmitOperationKind>(kind);
            operation.realized = realized != 0;
        }
    } else {
        collect_shader_versions(c);
        for (size_t i = 0; i < c.draws.size(); ++i) {
            c.draws[i].draw_index = i;
            c.operations.push_back({SubmitOperationKind::Draw, i, c.draws[i].command_order, true});
        }
    }
    if (version >= 7) {
        c.failure_diagnostics_available = true;
        uint32_t raw_count = 0;
        if (!r.u32(raw_count) || raw_count > kMaxResources) {
            error = "invalid raw shader count";
            return false;
        }
        c.raw_shader_versions.resize(raw_count);
        uint64_t raw_words = 0;
        for (auto& shader : c.raw_shader_versions) {
            uint8_t has_endpgm = 0;
            if (!r.u64(shader.content_hash) || !r.u8(has_endpgm) ||
                !r.words_bounded(shader.words, kMaxRawShaderWords,
                                 "invalid raw shader length")) return false;
            shader.has_endpgm = has_endpgm != 0;
            if (shader.words.empty() || raw_words > kMaxShaderWords - shader.words.size()) {
                error = "raw shader data exceeds its bounded limit";
                return false;
            }
            raw_words += shader.words.size();
        }
        uint32_t diagnostic_count = 0;
        if (!r.u32(diagnostic_count) || diagnostic_count > kMaxOperations) {
            error = "invalid failed-operation diagnostic count";
            return false;
        }
        c.failure_diagnostics.resize(diagnostic_count);
        for (auto& diagnostic : c.failure_diagnostics) {
            uint8_t kind = 0, reason = 0, pipeline_present = 0;
            if (!r.u8(kind) || kind > static_cast<uint8_t>(SubmitOperationKind::Dispatch) ||
                !r.u64(diagnostic.source_index) || !r.u64(diagnostic.command_order) ||
                !r.u8(reason) || reason <= static_cast<uint8_t>(RealizationFailureReason::None) ||
                reason > static_cast<uint8_t>(kMaxRealizationFailureReason) ||
                !r.u8(pipeline_present)) {
                error = "invalid failed-operation diagnostic metadata";
                return false;
            }
            diagnostic.kind = static_cast<SubmitOperationKind>(kind);
            diagnostic.reason = static_cast<RealizationFailureReason>(reason);
            diagnostic.pipeline_present = pipeline_present != 0;
            if (diagnostic.pipeline_present && !read_pipeline(r, diagnostic.pipeline, version)) return false;
            if (!r.u64(diagnostic.color0_base) || !r.u32(diagnostic.color0_width) ||
                !r.u32(diagnostic.color0_height) || !r.u32(diagnostic.vertex_count)) return false;
            auto& launch = diagnostic.compute_launch;
            if (!r.u32(launch.threads_x) || !r.u32(launch.threads_y) || !r.u32(launch.threads_z) ||
                !r.u32(launch.local_x) || !r.u32(launch.local_y) || !r.u32(launch.local_z) ||
                !r.u32(launch.groups_x) || !r.u32(launch.groups_y) || !r.u32(launch.groups_z)) return false;
            uint32_t stage_count = 0;
            if (!r.u32(stage_count) || stage_count > kMaxFailureStages) {
                error = "invalid failed-stage diagnostic count";
                return false;
            }
            diagnostic.stages.resize(stage_count);
            for (auto& stage : diagnostic.stages) {
                uint8_t stage_kind = 0, recompiled = 0, table_present = 0;
                uint32_t first_bad_fmt = 0;
                if (!r.u8(stage_kind) || stage_kind > static_cast<uint8_t>(ShaderProgramStage::Compute) ||
                    !r.u64(stage.program_addr) || !r.u32(stage.raw_shader_index) ||
                    !r.u8(recompiled) || !r.u8(table_present) || !r.u32(stage.resource_count) ||
                    !r.u32(stage.coverage.total) || !r.u32(stage.coverage.alu) ||
                    !r.u32(stage.coverage.exports) || !r.u32(stage.coverage.unsupported) ||
                    !r.u32(stage.coverage.table_dependent) || !r.u32(first_bad_fmt) ||
                    !r.u32(stage.coverage.first_bad_op) || !r.u32(stage.coverage.first_bad_pc) ||
                    !r.u32(stage.descriptor_issue_count) || !r.u32(stage.first_descriptor_issue)) return false;
                stage.stage = static_cast<ShaderProgramStage>(stage_kind);
                stage.recompiled = recompiled != 0;
                stage.resource_table_present = table_present != 0;
                stage.coverage.first_bad_fmt = static_cast<int32_t>(first_bad_fmt);
            }
        }
    }
    if (version >= 8) {
        uint32_t seed_count = 0;
        if (!r.u32(seed_count) || seed_count > kMaxResources) {
            error = "invalid DS seed count"; return false;
        }
        c.ds_seeds.resize(seed_count);
        uint64_t seed_total = 0;
        for (auto& seed : c.ds_seeds) {
            uint32_t format = 0;
            uint8_t depth_valid = 0, stencil_valid = 0;
            if (!r.u64(seed.depth_read_base) || !r.u64(seed.depth_write_base) ||
                !r.u64(seed.stencil_read_base) || !r.u64(seed.stencil_write_base) ||
                !r.u64(seed.htile_data_base) || !r.u32(seed.width) || !r.u32(seed.height) ||
                !r.u32(format) || !r.u8(depth_valid) || !r.u8(stencil_valid) ||
                !r.bytes(seed.depth) || !r.bytes(seed.stencil)) return false;
            if (depth_valid > 1 || stencil_valid > 1) {
                error = "invalid DS seed plane validity"; return false;
            }
            seed.format = static_cast<GpuCaptureDsFormat>(format);
            seed.depth_valid = depth_valid != 0;
            seed.stencil_valid = stencil_valid != 0;
            if (!validate_ds_seed(seed, error)) return false;
            const uint64_t plane_bytes = seed.depth.size() + seed.stencil.size();
            if (plane_bytes > kMaxTotalDsSeedBytes ||
                seed_total > kMaxTotalDsSeedBytes - plane_bytes) {
                error = "capture DS seed data exceeds 1 GiB"; return false;
            }
            seed_total += plane_bytes;
        }
    }
    if (version >= 9) {
        uint64_t expected_depths = 0;
        for (const auto& draw : c.draws)
            expected_depths += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected_depths += compute.resources.resources.size();
        uint32_t depth_count = 0;
        if (!r.u32(depth_count) || depth_count != expected_depths) {
            error = "invalid resource depth count"; return false;
        }
        auto read_depths = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources)
                if (!r.u32(captured.resource.depth) || !captured.resource.depth ||
                    captured.resource.depth > 8192u) {
                    error = "invalid resource depth";
                    return false;
                }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_depths(draw.vrt) || !read_depths(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_depths(compute.resources)) return false;
    }
    if (version >= 10) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid MRT1 draw-state count"; return false;
        }
        for (auto& draw : c.draws)
            if (!r.u64(draw.color1_base) || !r.u32(draw.color1_width) ||
                !r.u32(draw.color1_height) || !read_mrt1_pipeline(r, draw.ps)) return false;
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid MRT1 failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) {
            if (!r.u64(diagnostic.color1_base) || !r.u32(diagnostic.color1_width) ||
                !r.u32(diagnostic.color1_height)) return false;
            if (diagnostic.pipeline_present && !read_mrt1_pipeline(r, diagnostic.pipeline)) return false;
        }
    }
    if (version >= 11) {
        uint64_t expected_states = 0;
        for (const auto& draw : c.draws)
            expected_states += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected_states += compute.resources.resources.size();
        uint32_t state_count = 0;
        if (!r.u32(state_count) || state_count != expected_states) {
            error = "invalid resource DCC-state count"; return false;
        }
        auto read_dcc_state = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                auto& resource = captured.resource;
                uint8_t flags = 0;
                if (!r.u8(flags) || (flags & ~0x1fu) ||
                    !r.u32(resource.max_uncompressed_block_size) ||
                    !r.u32(resource.max_compressed_block_size) ||
                    !r.u64(resource.metadata_addr) ||
                    resource.max_uncompressed_block_size > 3u ||
                    resource.max_compressed_block_size > 3u) {
                    error = "invalid resource DCC state"; return false;
                }
                resource.meta_pipe_aligned = (flags & 1u) != 0;
                resource.write_compress_enabled = (flags & 2u) != 0;
                resource.compression_enabled = (flags & 4u) != 0;
                resource.alpha_is_on_msb = (flags & 8u) != 0;
                resource.color_transform = (flags & 16u) != 0;
                captured.metadata_size = dcc_metadata_footprint(resource);
                resource.dcc_metadata_size = captured.metadata_size;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_dcc_state(draw.vrt) || !read_dcc_state(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_dcc_state(compute.resources)) return false;
    }
    if (version >= 12) {
        uint64_t expected_refs = 0;
        for (const auto& draw : c.draws)
            expected_refs += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected_refs += compute.resources.resources.size();
        uint32_t ref_count = 0;
        if (!r.u32(ref_count) || ref_count != expected_refs) {
            error = "invalid resource DCC-metadata reference count"; return false;
        }
        auto read_dcc_metadata = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                // v44 kept append-only compatibility, so a 2D_MSAA sample count is serialized
                // after this v12 metadata-reference tail. Defer only its footprint equality until
                // that authoritative count has been read; blob bounds and all other invariants are
                // still checked here. Older versions cannot represent the multi-sample span.
                const bool defer_msaa_footprint = version >= 44 &&
                    captured.resource.img_dim == 6u;
                if (!r.u64(captured.metadata_size) ||
                    !r.u32(captured.metadata_blob_index) ||
                    !r.u64(captured.metadata_blob_offset) ||
                    (!defer_msaa_footprint &&
                     captured.metadata_size != dcc_metadata_footprint(captured.resource)) ||
                    (!captured.metadata_size &&
                     (captured.metadata_blob_index != 0xFFFFFFFFu ||
                      captured.metadata_blob_offset != 0)) ||
                    (captured.metadata_blob_index == 0xFFFFFFFFu &&
                     captured.metadata_blob_offset != 0)) {
                    error = "invalid resource DCC metadata reference"; return false;
                }
                if (captured.metadata_blob_index != 0xFFFFFFFFu) {
                    if (captured.metadata_blob_index >= c.blobs.size()) {
                        error = "resource DCC metadata references an invalid capture blob"; return false;
                    }
                    const auto& capture_blob = c.blobs[captured.metadata_blob_index];
                    const auto& blob = capture_blob.bytes;
                    if (!capture_blob_payload_omitted(capture_blob) &&
                        (captured.metadata_blob_offset > blob.size() ||
                         captured.metadata_size > blob.size() - captured.metadata_blob_offset)) {
                        error = "resource DCC metadata exceeds its capture blob"; return false;
                    }
                }
                captured.resource.dcc_metadata_size = captured.metadata_size;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_dcc_metadata(draw.vrt) || !read_dcc_metadata(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_dcc_metadata(compute.resources)) return false;
    }
    if (version >= 14) {
        uint32_t dma_count = 0;
        if (!r.u32(dma_count) || dma_count > kMaxOperations) {
            error = "invalid ordered DMA count";
            return false;
        }
        c.dma_copies.resize(dma_count);
        for (auto& copy : c.dma_copies) {
            if (!r.u64(copy.dst) || !r.u64(copy.src) || !r.u32(copy.bytes) ||
                !r.u32(copy.sels) || !r.u64(copy.command_order) ||
                !r.u64(copy.packet_addr) || !r.u32(copy.destination_blob_index) ||
                !r.u64(copy.destination_blob_offset) || !r.u32(copy.source_blob_index) ||
                !r.u64(copy.source_blob_offset))
                return false;
        }
        if (!validate_dma_copies(c, error)) return false;
    }
    if (version >= 15) {
        uint64_t expected_states = 0;
        for (const auto& draw : c.draws)
            expected_states += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected_states += compute.resources.resources.size();
        uint32_t state_count = 0;
        if (!r.u32(state_count) || state_count != expected_states) {
            error = "invalid resource depth-compare count"; return false;
        }
        auto read_depth_compare = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                uint8_t enabled = 0;
                if (!r.u8(enabled) || enabled > 1) {
                    error = "invalid resource depth-compare state"; return false;
                }
                captured.resource.depth_compare = enabled != 0;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_depth_compare(draw.vrt) || !read_depth_compare(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_depth_compare(compute.resources)) return false;
    }
    if (version >= 16) {
        uint64_t expected_states = 0;
        for (const auto& draw : c.draws)
            expected_states += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected_states += compute.resources.resources.size();
        uint32_t state_count = 0;
        if (!r.u32(state_count) || state_count != expected_states) {
            error = "invalid resource mip-tail count"; return false;
        }
        auto read_mip_tail = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                auto& resource = captured.resource;
                uint8_t enabled = 0;
                if (!r.u8(enabled) || enabled > 1 ||
                    !r.u32(resource.mip_tail_offset) ||
                    !r.u32(resource.mip_tail_bytes) ||
                    !r.u32(resource.mip_tail_x) ||
                    !r.u32(resource.mip_tail_y) ||
                    (enabled && resource.mip_tail_bytes != 4096u &&
                                resource.mip_tail_bytes != 65536u) ||
                    (enabled && resource.mip_tail_offset >= resource.mip_tail_bytes) ||
                    (!enabled && (resource.mip_tail_offset != 0 ||
                                  resource.mip_tail_bytes != 0 ||
                                  resource.mip_tail_x != 0 ||
                                  resource.mip_tail_y != 0))) {
                    error = "invalid resource mip-tail state"; return false;
                }
                resource.in_mip_tail = enabled != 0;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_mip_tail(draw.vrt) || !read_mip_tail(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_mip_tail(compute.resources)) return false;
    }
    if (version >= 17) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid scissor draw-state count"; return false;
        }
        for (auto& draw : c.draws)
            if (!read_scissor_pipeline(r, draw.ps)) return false;
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid scissor failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics)
            if (diagnostic.pipeline_present && !read_scissor_pipeline(r, diagnostic.pipeline))
                return false;
    }
    if (version >= 18) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid geometry-stage draw-state count"; return false;
        }
        for (auto& draw : c.draws) {
            uint32_t geometry_index = 0;
            if (!r.u32(geometry_index)) return false;
            if (geometry_index == 0xFFFFFFFFu) continue;
            if (geometry_index >= c.shader_versions.size()) {
                error = "invalid draw geometry shader-version index"; return false;
            }
            draw.gs = c.shader_versions[geometry_index].words;
        }
    }
    if (version >= 19) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid realized raw-shader draw count"; return false;
        }
        for (auto& draw : c.draws)
            if (!r.u32(draw.vs_raw_shader_index) || !r.u32(draw.fs_raw_shader_index))
                return false;
    }
    if (version >= 20) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid draw instance-count state count"; return false;
        }
        for (auto& draw : c.draws)
            if (!r.u32(draw.instance_count)) return false;
    }
    if (version >= 21) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid logic-op draw-state count"; return false;
        }
        for (auto& draw : c.draws)
            if (!read_logic_op_pipeline(r, draw.ps)) return false;
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid logic-op failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics)
            if (diagnostic.pipeline_present && !read_logic_op_pipeline(r, diagnostic.pipeline))
                return false;
    }
    if (version >= 22) {
        uint64_t expected_states = 0;
        for (const auto& draw : c.draws)
            expected_states += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected_states += compute.resources.resources.size();
        uint32_t state_count = 0;
        if (!r.u32(state_count) || state_count != expected_states) {
            error = "invalid compute GDS state count";
            return false;
        }
        auto read_internal_state = [&](GpuCapturedTable& table,
                                       bool allow_packed_pointer) {
            for (auto& captured : table.resources) {
                if (!r.bytes(captured.internal_bytes)) return false;
                const bool gds = is_compute_internal_gds(captured.resource);
                const bool packed = version >= 51u &&
                    is_gta5_packed_pointer_serialized_shadow(
                        captured.resource, captured.internal_bytes.data(),
                        captured.internal_bytes.size());
                const bool relocated = version >= 53u &&
                    is_indirect_pointer_relocation_serialized(
                        captured.resource, captured.internal_bytes.data(),
                        captured.internal_bytes.size());
                if ((!allow_packed_pointer && (packed || relocated)) ||
                    (!gds && !packed && !relocated && !captured.internal_bytes.empty()) ||
                    (gds && !captured.internal_bytes.empty() &&
                     captured.internal_bytes.size() != resource_footprint(captured.resource))) {
                    error = "invalid compute internal capture state";
                    return false;
                }
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_internal_state(draw.vrt, false) ||
                !read_internal_state(draw.prt, false)) return false;
        for (auto& compute : c.computes)
            if (!read_internal_state(compute.resources, true)) return false;
    }
    if (version >= 23) {   // #1256: raw draw-packet state per realized draw
        uint32_t nd = 0;
        if (!r.u32(nd) || nd != c.draws.size()) { error = "invalid raw-draw-state count"; return false; }
        for (auto& draw : c.draws) {
            uint32_t ri = 0;
            if (!r.u32(draw.raw_draw_count) || !r.u32(ri)) return false;
            draw.raw_indexed = ri != 0;
        }
    }
    if (version >= 24) {   // #1280: per-resource T#-declared mip-chain length (trailing tail; default 1)
        size_t expected = 0;
        for (auto& draw : c.draws) expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (auto& compute : c.computes) expected += compute.resources.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) { error = "invalid declared-mip-levels count"; return false; }
        auto read_declared_mips = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources)
                if (!r.u32(captured.resource.declared_mip_levels)) return false;
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_declared_mips(draw.vrt) || !read_declared_mips(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_declared_mips(compute.resources)) return false;
    }
    if (version >= 25) {   // #1240: MODE=3 resolve intent for realized and failed pipelines
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid resolve draw-state count"; return false;
        }
        for (auto& draw : c.draws) {
            uint8_t enabled = 0;
            if (!r.u8(enabled) || enabled > 1) return false;
            draw.ps.cb_resolve = enabled != 0;
        }
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid resolve failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) if (diagnostic.pipeline_present) {
            uint8_t enabled = 0;
            if (!r.u8(enabled) || enabled > 1) return false;
            diagnostic.pipeline.cb_resolve = enabled != 0;
        }
    }
    if (version >= 26) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid color-export draw-state count"; return false;
        }
        for (auto& draw : c.draws)
            if (!r.u32(draw.ps.spi_shader_col_format) || !r.u32(draw.ps.sx_ps_downconvert)) return false;
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid color-export failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) if (diagnostic.pipeline_present)
            if (!r.u32(diagnostic.pipeline.spi_shader_col_format) ||
                !r.u32(diagnostic.pipeline.sx_ps_downconvert)) return false;
    }
    if (version >= 27) {
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid draw-parameter state count"; return false;
        }
        for (auto& draw : c.draws) {
            uint32_t vertex_offset = 0;
            if (!r.u64(draw.raw_draw_modifier) || !r.u32(vertex_offset)) return false;
            draw.vertex_offset = static_cast<int32_t>(vertex_offset);
        }
    }
    if (version >= 28) {
        size_t expected = 0;
        for (auto& draw : c.draws)
            expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (auto& compute : c.computes)
            expected += compute.resources.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid linear-row-pitch count"; return false;
        }
        auto read_linear_layout = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                if (!r.u32(captured.resource.linear_row_pitch_bytes) ||
                    !r.u64(captured.captured_size)) return false;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_linear_layout(draw.vrt) || !read_linear_layout(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_linear_layout(compute.resources)) return false;
    }
    if (version >= 29) {   // #1349: resolved depth-bias state for realized and failed pipelines
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid depth-bias draw-state count"; return false;
        }
        auto read_bias = [&](ResolvedPipelineState& pipeline) {
            uint8_t enabled = 0;
            uint32_t constant = 0, slope = 0, clamp = 0;
            if (!r.u8(enabled) || enabled > 1 || !r.u32(constant) || !r.u32(slope) ||
                !r.u32(clamp)) return false;
            pipeline.depth_bias_enable   = enabled;
            pipeline.depth_bias_constant = std::bit_cast<float>(constant);
            pipeline.depth_bias_slope    = std::bit_cast<float>(slope);
            pipeline.depth_bias_clamp    = std::bit_cast<float>(clamp);
            return true;
        };
        for (auto& draw : c.draws) if (!read_bias(draw.ps)) return false;
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid depth-bias failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) if (diagnostic.pipeline_present)
            if (!read_bias(diagnostic.pipeline)) return false;
    }
    if (version >= 30) {   // dynamic-fold-proven buffer fetch index source
        size_t expected = 0;
        for (auto& draw : c.draws) expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (auto& compute : c.computes) expected += compute.resources.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid fetch-index-mode count"; return false;
        }
        auto read_fetch_index_modes = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                uint32_t mode = 0;
                if (!r.u32(mode) || mode > static_cast<uint32_t>(VertexFetchIndexMode::Instance))
                    return false;
                captured.resource.fetch_index_mode = static_cast<VertexFetchIndexMode>(mode);
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_fetch_index_modes(draw.vrt) || !read_fetch_index_modes(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_fetch_index_modes(compute.resources)) return false;
    }
    if (version >= 31) {   // linked NGG vertex main stage + graphics LDS
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid linked-vertex draw-state count"; return false;
        }
        for (auto& draw : c.draws) {
            if (!r.u32(draw.vs_chain_raw_shader_index) || !r.u32(draw.vertex_lds_dwords))
                return false;
            if (draw.vertex_lds_dwords > 16384u) {
                error = "invalid linked-vertex LDS allocation"; return false;
            }
        }
    }
    if (version >= 32) {   // selected mip placement inside each thin-2D array or cube slice
        size_t expected = 0;
        for (auto& draw : c.draws) expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (auto& compute : c.computes) expected += compute.resources.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid resource layered layout count"; return false;
        }
        auto read_layer_layout = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                ShaderResource& resource = captured.resource;
                if (!r.u32(resource.layer_stride_bytes) ||
                    !r.u32(resource.layer_mip_offset_bytes))
                    return false;
                const bool enabled = resource.layer_stride_bytes != 0;
                if ((!enabled && resource.layer_mip_offset_bytes != 0) ||
                    (enabled && ((resource.img_dim != 3 && resource.img_dim != 5) ||
                                 resource.depth < 2 ||
                                 resource.layer_mip_offset_bytes >= resource.layer_stride_bytes ||
                                 (resource.in_mip_tail && resource.layer_mip_offset_bytes != 0)))) {
                    error = "invalid resource layered mip layout";
                    return false;
                }
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_layer_layout(draw.vrt) || !read_layer_layout(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_layer_layout(compute.resources)) return false;
    }
    if (version >= 34) {   // complete hardware color-target bindings and pipeline state
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid complete-MRT draw-state count"; return false;
        }
        for (auto& draw : c.draws) {
            for (uint32_t slot = 2; slot < kColorTargetCount; ++slot) {
                auto& binding = draw.color_targets[slot];
                if (!r.u64(binding.base) || !r.u32(binding.width) || !r.u32(binding.height) ||
                    !read_color_target_pipeline(r, draw.ps.color_targets[slot]))
                    return false;
            }
        }
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid complete-MRT failure-state count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) {
            for (uint32_t slot = 2; slot < kColorTargetCount; ++slot) {
                auto& binding = diagnostic.color_targets[slot];
                if (!r.u64(binding.base) || !r.u32(binding.width) || !r.u32(binding.height))
                    return false;
            }
            if (diagnostic.pipeline_present)
                for (uint32_t slot = 2; slot < kColorTargetCount; ++slot)
                    if (!read_color_target_pipeline(
                            r, diagnostic.pipeline.color_targets[slot])) return false;
        }
    }
    if (version >= 35) {   // exact failed-stage resource-descriptor metadata
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid failed-stage resource-table failure count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) {
            uint32_t stage_count = 0;
            if (!r.u32(stage_count) || stage_count != diagnostic.stages.size()) {
                error = "invalid failed-stage resource-table stage count"; return false;
            }
            for (auto& stage : diagnostic.stages)
                if (!read_table(r, stage.resource_table, version)) return false;
        }
    }
    if (version >= 36) {   // exact resolved pixel-stage ABI for raw graphics replay
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid pixel-stage ABI draw count"; return false;
        }
        for (auto& draw : c.draws) {
            uint8_t flags = 0;
            if (!r.u8(flags) || (flags & ~3u)) {
                error = "invalid realized-draw pixel-stage ABI flags"; return false;
            }
            draw.has_pixel_inputs = (flags & 1u) != 0;
            draw.has_system_inputs = (flags & 2u) != 0;
            if (draw.has_pixel_inputs) {
                if (!r.u32(draw.pixel_inputs.valid_mask) ||
                    !r.u32(draw.pixel_inputs.passthrough_mask)) return false;
                for (uint32_t& control : draw.pixel_inputs.controls)
                    if (!r.u32(control)) return false;
                if (!draw.pixel_inputs.valid_mask ||
                    (draw.pixel_inputs.passthrough_mask & ~draw.pixel_inputs.valid_mask)) {
                    error = "invalid realized-draw pixel input mapping"; return false;
                }
            }
            if (draw.has_system_inputs) {
                if (!r.u32(draw.system_inputs.ena) || !r.u32(draw.system_inputs.addr)) return false;
                if (!draw.system_inputs.ena && !draw.system_inputs.addr) {
                    error = "invalid realized-draw pixel system inputs"; return false;
                }
            }
        }
    }
    if (version >= 37) {   // exact required-size/full-subgroups compute pipeline contract
        uint32_t compute_count = 0;
        if (!r.u32(compute_count) || compute_count != c.computes.size()) {
            error = "invalid compute subgroup-contract count"; return false;
        }
        for (auto& compute : c.computes) {
            if (!r.u32(compute.required_subgroup_size)) return false;
            if (compute.required_subgroup_size != 0 && compute.required_subgroup_size != 32 &&
                compute.required_subgroup_size != 64) {
                error = "invalid compute required-subgroup size"; return false;
            }
        }
    }
    if (version >= 39) {   // raw realized compute program + semantic recompile state
        uint32_t compute_count = 0;
        if (!r.u32(compute_count) || compute_count != c.computes.size()) {
            error = "invalid compute recompile-state count"; return false;
        }
        for (auto& compute : c.computes) {
            uint8_t available = 0;
            if (!r.u8(available) || available > 1 || !r.u32(compute.raw_shader_index)) {
                error = "invalid compute recompile-state availability"; return false;
            }
            compute.recompile_config_available = available != 0;
            if (!compute.recompile_config_available) {
                if (compute.raw_shader_index != 0xFFFFFFFFu) {
                    error = "compute raw shader has no recompile state"; return false;
                }
                continue;
            }
            if (!read_compute_config(r, compute.recompile_config, version)) return false;
            if (!validate_compute_recompile_state(compute, error)) return false;
        }
    }
    if (version >= 40) {   // BVH descriptor BOX_GROW per resource
        size_t expected = 0;
        for (const auto& draw : c.draws)
            expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected += compute.resources.resources.size();
        for (const auto& diagnostic : c.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                expected += stage.resource_table.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid BVH resource-state count";
            return false;
        }
        auto read_bvh_box_grow = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources)
                if (!r.u32(captured.resource.bvh_box_grow) ||
                    captured.resource.bvh_box_grow > 0xFFu)
                    return false;
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_bvh_box_grow(draw.vrt) || !read_bvh_box_grow(draw.prt)) return false;
        for (auto& compute : c.computes)
            if (!read_bvh_box_grow(compute.resources)) return false;
        for (auto& diagnostic : c.failure_diagnostics)
            for (auto& stage : diagnostic.stages)
                if (!read_bvh_box_grow(stage.resource_table)) return false;
    }
    if (version >= 41) {   // complete versioned state for v35 failed-stage resource tables
        size_t expected = 0;
        for (const auto& diagnostic : c.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                expected += stage.resource_table.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid failed-stage resource-state count";
            return false;
        }
        auto read_failed_resource_state = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                ShaderResource& resource = captured.resource;
                uint8_t dcc_flags = 0, depth_compare = 0, in_mip_tail = 0;
                uint32_t fetch_index_mode = 0;
                if (!r.u32(resource.depth) || !resource.depth || resource.depth > 8192u ||
                    !r.u8(dcc_flags) || (dcc_flags & ~0x1fu) ||
                    !r.u32(resource.max_uncompressed_block_size) ||
                    !r.u32(resource.max_compressed_block_size) ||
                    !r.u64(resource.metadata_addr) ||
                    resource.max_uncompressed_block_size > 3u ||
                    resource.max_compressed_block_size > 3u ||
                    !r.u8(depth_compare) || depth_compare > 1u ||
                    !r.u8(in_mip_tail) || in_mip_tail > 1u ||
                    !r.u32(resource.mip_tail_offset) ||
                    !r.u32(resource.mip_tail_bytes) ||
                    !r.u32(resource.mip_tail_x) ||
                    !r.u32(resource.mip_tail_y) ||
                    !r.u32(resource.declared_mip_levels) || !resource.declared_mip_levels ||
                    !r.u32(resource.linear_row_pitch_bytes) ||
                    !r.u64(captured.captured_size) ||
                    !r.u32(fetch_index_mode) ||
                    fetch_index_mode > static_cast<uint32_t>(VertexFetchIndexMode::Instance) ||
                    !r.u32(resource.layer_stride_bytes) ||
                    !r.u32(resource.layer_mip_offset_bytes) ||
                    !r.u32(resource.flat_base_sgpr)) {
                    error = "invalid failed-stage resource state";
                    return false;
                }
                resource.meta_pipe_aligned = (dcc_flags & 1u) != 0;
                resource.write_compress_enabled = (dcc_flags & 2u) != 0;
                resource.compression_enabled = (dcc_flags & 4u) != 0;
                resource.alpha_is_on_msb = (dcc_flags & 8u) != 0;
                resource.color_transform = (dcc_flags & 16u) != 0;
                resource.depth_compare = depth_compare != 0;
                resource.in_mip_tail = in_mip_tail != 0;
                resource.fetch_index_mode =
                    static_cast<VertexFetchIndexMode>(fetch_index_mode);
                const bool mip_tail_valid =
                    (!resource.in_mip_tail && resource.mip_tail_offset == 0 &&
                     resource.mip_tail_bytes == 0 && resource.mip_tail_x == 0 &&
                     resource.mip_tail_y == 0) ||
                    (resource.in_mip_tail &&
                     (resource.mip_tail_bytes == 4096u ||
                      resource.mip_tail_bytes == 65536u) &&
                     resource.mip_tail_offset < resource.mip_tail_bytes);
                const bool layered = resource.layer_stride_bytes != 0;
                if (!mip_tail_valid ||
                    (!layered && resource.layer_mip_offset_bytes != 0) ||
                    (layered && ((resource.img_dim != 3 && resource.img_dim != 5) ||
                                 resource.depth < 2 ||
                                 resource.layer_mip_offset_bytes >=
                                     resource.layer_stride_bytes ||
                                 (resource.in_mip_tail &&
                                  resource.layer_mip_offset_bytes != 0)))) {
                    error = "invalid failed-stage resource layout state";
                    return false;
                }
                captured.metadata_size = dcc_metadata_footprint(resource);
                resource.dcc_metadata_size = captured.metadata_size;
            }
            return true;
        };
        for (auto& diagnostic : c.failure_diagnostics)
            for (auto& stage : diagnostic.stages)
                if (!read_failed_resource_state(stage.resource_table)) return false;
    }
    if (version >= 42) {   // exact failed-compute launch and user-SGPR specialization
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid failed-compute recompile-state failure count";
            return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) {
            uint32_t stage_count = 0;
            if (!r.u32(stage_count) || stage_count != diagnostic.stages.size()) {
                error = "invalid failed-compute recompile-state stage count";
                return false;
            }
            for (auto& stage : diagnostic.stages) {
                uint8_t available = 0;
                if (!r.u8(available) || available > 1u) {
                    error = "invalid failed-compute recompile-state availability";
                    return false;
                }
                stage.recompile_config_available = available != 0;
                if (!stage.recompile_config_available) continue;
                if (diagnostic.kind != SubmitOperationKind::Dispatch ||
                    stage.stage != ShaderProgramStage::Compute ||
                    !read_compute_config(r, stage.recompile_config, version) ||
                    !validate_compute_config(stage.recompile_config,
                                             diagnostic.compute_launch,
                                             stage.recompile_config.native_subgroup_size,
                                             "invalid failed-compute recompile state", error)) {
                    if (error.empty()) error = "invalid failed-compute recompile state";
                    return false;
                }
            }
        }
    }
    if (version >= 43) {   // #1459: raw color-state provenance for every resolved write mask
        uint32_t draw_count = 0;
        if (!r.u32(draw_count) || draw_count != c.draws.size()) {
            error = "invalid color-state draw count"; return false;
        }
        for (auto& draw : c.draws)
            if (!read_color_state(r, draw.ps)) { error = "invalid color-state draw record"; return false; }
        uint32_t failure_count = 0;
        if (!r.u32(failure_count) || failure_count != c.failure_diagnostics.size()) {
            error = "invalid color-state failure count"; return false;
        }
        for (auto& diagnostic : c.failure_diagnostics) if (diagnostic.pipeline_present)
            if (!read_color_state(r, diagnostic.pipeline)) {
                error = "invalid color-state failure record"; return false;
            }
    }
    if (version >= 44) {   // guest sample count; v1-v43 retain ShaderResource's default of one
        size_t expected = 0;
        for (const auto& draw : c.draws)
            expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected += compute.resources.resources.size();
        for (const auto& diagnostic : c.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                expected += stage.resource_table.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid resource sample-count count"; return false;
        }
        auto read_sample_counts = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                uint32_t samples = 0;
                if (!r.u32(samples) || !samples || samples > 32768u ||
                    (samples & (samples - 1u)) != 0) return false;
                captured.resource.sample_count = samples;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_sample_counts(draw.vrt) || !read_sample_counts(draw.prt)) {
                error = "invalid resource sample count"; return false;
            }
        for (auto& compute : c.computes)
            if (!read_sample_counts(compute.resources)) {
                error = "invalid resource sample count"; return false;
            }
        for (auto& diagnostic : c.failure_diagnostics)
            for (auto& stage : diagnostic.stages)
                if (!read_sample_counts(stage.resource_table)) {
                    error = "invalid resource sample count"; return false;
                }
        // Only realized draw/compute tables have v12 metadata blob references. Failed-stage tables
        // retain descriptor/DCC state through v41 and sample identity through v44, but deliberately
        // carry no resource or metadata blobs; there is therefore no deferred reference to validate
        // for those diagnostic tables.
        auto validate_deferred_msaa_metadata = [&](const GpuCapturedTable& table) {
            for (const auto& captured : table.resources)
                if (captured.resource.img_dim == 6u &&
                    captured.metadata_size != dcc_metadata_footprint(captured.resource))
                    return false;
            return true;
        };
        for (const auto& draw : c.draws)
            if (!validate_deferred_msaa_metadata(draw.vrt) ||
                !validate_deferred_msaa_metadata(draw.prt)) {
                error = "invalid resource DCC metadata reference"; return false;
            }
        for (const auto& compute : c.computes)
            if (!validate_deferred_msaa_metadata(compute.resources)) {
                error = "invalid resource DCC metadata reference"; return false;
            }
    }
    if (version >= 45) {
        uint32_t count = 0;
        if (!r.u32(count) || count > 1) {
            error = "invalid resource provenance record count";
            return false;
        }
        c.resource_provenance.resize(count);
        for (auto& provenance : c.resource_provenance) {
            uint8_t stage = 0;
            uint32_t resource_class = 0;
            if (!r.u64(provenance.draw_index) || !r.u8(stage) ||
                !r.u32(provenance.binding) || !r.u32(resource_class) ||
                !r.u64(provenance.guest_addr) || !r.u64(provenance.requested_bytes) ||
                !r.u32(provenance.srt_offset) || !r.u32(provenance.sgpr_base) ||
                !r.u32(provenance.realization_blob_index) ||
                !r.u32(provenance.post_blob_index) ||
                !r.u64(provenance.post_blob_offset) ||
                !r.u64(provenance.realization_content_hash) ||
                !r.u64(provenance.post_content_hash))
                return false;
            provenance.stage = static_cast<ShaderProgramStage>(stage);
            provenance.resource_class = static_cast<ResourceClass>(resource_class);
        }
        if (!validate_resource_provenance(c, error)) return false;
    }
    if (version >= 46) {
        uint32_t count = 0;
        if (!r.u32(count) || count != c.resource_provenance.size()) {
            error = "invalid resource input witness record count";
            return false;
        }
        for (auto& provenance : c.resource_provenance) {
            uint64_t draw_index = 0;
            uint8_t stage = 0, mode = 0, reason = 0, provenance_mask = 0;
            uint32_t binding = 0;
            if (!r.u64(draw_index) || !r.u8(stage) || !r.u32(binding) ||
                !r.u8(mode) || !r.u8(reason) || !r.u8(provenance_mask) ||
                !r.u32(provenance.input_sh_register_base))
                return false;
            if (draw_index != provenance.draw_index ||
                stage != static_cast<uint8_t>(provenance.stage) ||
                binding != provenance.binding) {
                error = "resource input witness identity mismatch";
                return false;
            }
            provenance.input_mode = static_cast<GpuCaptureResourceInputMode>(mode);
            provenance.input_unavailable_reason =
                static_cast<GpuCaptureResourceInputUnavailableReason>(reason);
            provenance.input_write_provenance_mask = provenance_mask;
            for (auto& value : provenance.input_dwords) if (!r.u32(value)) return false;
            for (auto& write : provenance.input_last_writes) if (!r.u64(write)) return false;
            for (auto& source : provenance.input_write_sources) if (!r.u64(source)) return false;
        }
        if (!validate_resource_provenance(c, error)) return false;
    }
    if (version >= 47) {
        size_t expected = 0;
        for (const auto& draw : c.draws)
            expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected += compute.resources.resources.size();
        for (const auto& diagnostic : c.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                expected += stage.resource_table.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid resource zero-mip state count";
            return false;
        }
        auto read_zero_mip_state = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                uint8_t proven = 0;
                if (!r.u8(proven) || proven > 1u) return false;
                captured.resource.proven_zero_mip = proven != 0;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_zero_mip_state(draw.vrt) || !read_zero_mip_state(draw.prt)) {
                error = "invalid resource zero-mip state";
                return false;
            }
        for (auto& compute : c.computes)
            if (!read_zero_mip_state(compute.resources)) {
                error = "invalid resource zero-mip state";
                return false;
            }
        for (auto& diagnostic : c.failure_diagnostics)
            for (auto& stage : diagnostic.stages)
                if (!read_zero_mip_state(stage.resource_table)) {
                    error = "invalid resource zero-mip state";
                    return false;
                }
    }
    if (version >= 48) {
        size_t expected = 0;
        for (const auto& draw : c.draws)
            expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected += compute.resources.resources.size();
        for (const auto& diagnostic : c.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                expected += stage.resource_table.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid BVH sort-state count";
            return false;
        }
        auto read_bvh_sort_state = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                uint8_t enabled = 0;
                if (!r.u8(enabled) || enabled > 1u) return false;
                captured.resource.bvh_sort_enabled = enabled != 0;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_bvh_sort_state(draw.vrt) || !read_bvh_sort_state(draw.prt)) {
                error = "invalid BVH sort state";
                return false;
            }
        for (auto& compute : c.computes)
            if (!read_bvh_sort_state(compute.resources)) {
                error = "invalid BVH sort state";
                return false;
            }
        for (auto& diagnostic : c.failure_diagnostics)
            for (auto& stage : diagnostic.stages)
                if (!read_bvh_sort_state(stage.resource_table)) {
                    error = "invalid BVH sort state";
                    return false;
                }
    }
    if (version >= 49) {
        size_t expected = 0;
        for (const auto& draw : c.draws)
            expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected += compute.resources.resources.size();
        for (const auto& diagnostic : c.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                expected += stage.resource_table.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid atomic-x2 state count";
            return false;
        }
        auto read_atomic_x2_state = [&](GpuCapturedTable& table) {
            for (auto& captured : table.resources) {
                uint32_t record_count = 0;
                if (!r.u32(record_count)) return false;
                captured.resource.atomic_x2_record_count = record_count;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_atomic_x2_state(draw.vrt) ||
                !read_atomic_x2_state(draw.prt)) {
                error = "invalid atomic-x2 state";
                return false;
            }
        for (auto& compute : c.computes)
            if (!read_atomic_x2_state(compute.resources)) {
                error = "invalid atomic-x2 state";
                return false;
            }
        for (auto& diagnostic : c.failure_diagnostics)
            for (auto& stage : diagnostic.stages)
                if (!read_atomic_x2_state(stage.resource_table)) {
                    error = "invalid atomic-x2 state";
                    return false;
                }
    }
    if (version >= 50) {
        size_t expected = 0;
        for (const auto& compute : c.computes)
            expected += compute.recompile_config_available;
        for (const auto& diagnostic : c.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                expected += stage.recompile_config_available;
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid compute PGM_RSRC1 state count";
            return false;
        }
        for (auto& compute : c.computes)
            if (compute.recompile_config_available &&
                !r.u32(compute.recompile_config.compute_pgm_rsrc1))
                return false;
        for (auto& diagnostic : c.failure_diagnostics)
            for (auto& stage : diagnostic.stages)
                if (stage.recompile_config_available &&
                    !r.u32(stage.recompile_config.compute_pgm_rsrc1))
                    return false;
    }
    if (version >= 52) {
        size_t expected = 0;
        for (const auto& draw : c.draws)
            expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected += compute.resources.resources.size();
        for (const auto& diagnostic : c.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                expected += stage.resource_table.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid buffer descriptor-table resource count";
            return false;
        }
        auto read_buffer_tables = [&](GpuCapturedTable& table) {
            for (GpuCapturedResource& captured : table.resources) {
                ShaderResource& resource = captured.resource;
                uint32_t selector_mode = 0;
                uint32_t entry_count = 0;
                if (!r.u32(resource.table_index_count) ||
                    !r.u32(resource.table_entry_stride) ||
                    !r.u32(resource.table_index_sgpr) || !r.u32(selector_mode) ||
                    selector_mode > static_cast<uint32_t>(
                        BufferTableSelectorMode::DynamicSbufferByteOffset) ||
                    !r.u32(resource.table_load_pc) || !r.u32(entry_count) ||
                    entry_count > 4096u || entry_count != resource.table_index_count)
                    return false;
                resource.table_selector_mode =
                    static_cast<BufferTableSelectorMode>(selector_mode);
                resource.table_entries.resize(entry_count);
                captured.table_entry_blobs.resize(entry_count);
                for (uint32_t index = 0; index < entry_count; ++index) {
                    ShaderBufferTableEntry& entry = resource.table_entries[index];
                    auto& backing = captured.table_entry_blobs[index];
                    for (uint32_t& word : entry.vsharp)
                        if (!r.u32(word)) return false;
                    if (!r.u64(entry.gpu_addr) || !r.u32(entry.size) ||
                        !r.u32(entry.stride) || !r.u32(backing.blob_index) ||
                        !r.u64(backing.blob_offset))
                        return false;
                    if ((backing.blob_index == UINT32_MAX && backing.blob_offset != 0u) ||
                        (backing.blob_index != UINT32_MAX &&
                         (backing.blob_index >= c.blobs.size() ||
                          (!capture_blob_payload_omitted(c.blobs[backing.blob_index]) &&
                           (backing.blob_offset > c.blobs[backing.blob_index].bytes.size() ||
                            entry.size > c.blobs[backing.blob_index].bytes.size() -
                                             backing.blob_offset)))))
                        return false;
                }
                if (!valid_shader_buffer_table_contract(resource)) return false;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_buffer_tables(draw.vrt) || !read_buffer_tables(draw.prt)) {
                error = "invalid captured buffer descriptor-table state";
                return false;
            }
        for (auto& compute : c.computes)
            if (!read_buffer_tables(compute.resources)) {
                error = "invalid captured buffer descriptor-table state";
                return false;
            }
        for (auto& diagnostic : c.failure_diagnostics)
            for (auto& stage : diagnostic.stages)
                if (!read_buffer_tables(stage.resource_table)) {
                    error = "invalid captured buffer descriptor-table state";
                    return false;
                }
    }
    if (version >= 54) {
        size_t expected = 0;
        for (const auto& draw : c.draws)
            expected += draw.vrt.resources.size() + draw.prt.resources.size();
        for (const auto& compute : c.computes)
            expected += compute.resources.resources.size();
        for (const auto& diagnostic : c.failure_diagnostics)
            for (const auto& stage : diagnostic.stages)
                expected += stage.resource_table.resources.size();
        uint32_t count = 0;
        if (!r.u32(count) || count != expected) {
            error = "invalid scalar-buffer bound resource count";
            return false;
        }
        auto read_scalar_buffer_bounds = [&](GpuCapturedTable& table) {
            for (GpuCapturedResource& captured : table.resources) {
                ShaderResource& resource = captured.resource;
                if (!r.u32(resource.scalar_buffer_dword_count))
                    return false;
                if (resource.scalar_buffer_dword_count) {
                    // #2528: the scalar bound IS the V#'s byte footprint, so this field is fully
                    // derivable from `size` and carries no independent authority — it can only ever
                    // be a marker plus a checksum. Captures written before that correction stored
                    // NUM_RECORDS (or one dword for STRIDE==0) here, which disagrees with `size` and
                    // would otherwise make every retained v54 capsule permanently unreadable. Derive
                    // the authoritative value rather than trusting the stored one; a disagreement
                    // cannot widen a binding, because the derived span is exactly `size` either way.
                    // The WRITER stays strict, so prosper never emits an inconsistent bound itself.
                    const uint32_t derived =
                        static_cast<uint32_t>(resource.size / sizeof(uint32_t));
                    if (resource.scalar_buffer_dword_count != derived) {
                        if (std::getenv("PROSPER_DBG"))
                            std::fprintf(stderr,
                                         "[capture] scalar-buffer bound %u disagrees with the V# "
                                         "footprint (size=%u); using the derived %u (#2528)\n",
                                         resource.scalar_buffer_dword_count, resource.size, derived);
                        resource.scalar_buffer_dword_count = derived;
                    }
                    if (!resource.scalar_buffer_dword_count) continue;
                    const uint64_t binding_bytes =
                        shader_resource_buffer_binding_bytes(resource);
                    if (!binding_bytes || captured.captured_size < binding_bytes)
                        return false;
                }
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_scalar_buffer_bounds(draw.vrt) ||
                !read_scalar_buffer_bounds(draw.prt)) {
                error = "invalid scalar-buffer bound state";
                return false;
            }
        for (auto& compute : c.computes)
            if (!read_scalar_buffer_bounds(compute.resources)) {
                error = "invalid scalar-buffer bound state";
                return false;
            }
        for (auto& diagnostic : c.failure_diagnostics)
            for (auto& stage : diagnostic.stages)
                if (!read_scalar_buffer_bounds(stage.resource_table)) {
                    error = "invalid scalar-buffer bound state";
                    return false;
                }
    }
    // v55 trailing tail: each DS seed's array slice, in seed order. Pre-v55 captures leave every
    // seed at slice 0, which is what every non-layered surface is.
    if (version >= 55)
        for (auto& seed : c.ds_seeds)
            if (!r.u32(seed.slice)) return false;
    // v57 trailing tail: allocation-wide mip placement per resource (#3048). See the writer.
    if (version >= 57) {
        auto read_mip_chain_provenance = [&](GpuCapturedTable& table) {
            for (GpuCapturedResource& captured : table.resources) {
                ShaderResource& resource = captured.resource;
                if (!r.u32(resource.mip_chain_element_width) ||
                    !r.u32(resource.mip_chain_element_height) ||
                    !r.u32(resource.mip_chain_bytes_per_block) ||
                    !r.u32(resource.mip_chain_max_level) ||
                    !r.u32(resource.mip_chain_base_level))
                    return false;
                // A modelled placement is completely present or completely absent. A partial record
                // cannot place a level, and reading one as if it could is exactly the silent
                // wrong-offset failure this provenance exists to prevent.
                const bool any = resource.mip_chain_element_width ||
                                 resource.mip_chain_element_height ||
                                 resource.mip_chain_bytes_per_block;
                if (any && (!resource.mip_chain_element_width ||
                            !resource.mip_chain_element_height ||
                            !resource.mip_chain_bytes_per_block ||
                            resource.mip_chain_max_level >= 16u))
                    return false;
            }
            return true;
        };
        for (auto& draw : c.draws)
            if (!read_mip_chain_provenance(draw.vrt) ||
                !read_mip_chain_provenance(draw.prt)) {
                error = "invalid mip-chain placement state";
                return false;
            }
        for (auto& compute : c.computes)
            if (!read_mip_chain_provenance(compute.resources)) {
                error = "invalid mip-chain placement state";
                return false;
            }
        for (auto& diagnostic : c.failure_diagnostics)
            for (auto& stage : diagnostic.stages)
                if (!read_mip_chain_provenance(stage.resource_table)) {
                    error = "invalid mip-chain placement state";
                    return false;
                }
    }
    // DS seed identity is checked HERE, not in the record loop, because the slice arrives in the
    // tail above: a per-record check would compare incomplete identities and reject two faces of one
    // cube that differ only in slice -- which is exactly the capture this version exists to allow.
    {
        std::set<decltype(ds_seed_key(GpuCaptureDsSeed{}))> identities;
        for (const auto& seed : c.ds_seeds)
            if (!identities.insert(ds_seed_key(seed)).second) {
                error = "duplicate DS seed identity"; return false;
            }
    }
    for (auto& draw : c.draws) restore_legacy_color_target_aliases(draw);
    for (auto& diagnostic : c.failure_diagnostics)
        restore_legacy_color_target_aliases(diagnostic);
    if (version >= 7 && !validate_failure_diagnostics(c, error)) return false;
    if (r.left) { error = "capture has trailing data"; return false; }
    return true;
}

namespace {
// Interactive one-shot capture request (armed by the app hotkey, consumed on the render thread). A
// non-empty path means a grab is armed; the guard makes arm/consume race-free across threads.
}  // namespace

namespace {


} // namespace

namespace {
}  // namespace

} // namespace prosper::gpu
