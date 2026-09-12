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
#include "gpu/capture/serialize/capture_codecs.hpp"

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

namespace {
// Interactive one-shot capture request (armed by the app hotkey, consumed on the render thread). A
// non-empty path means a grab is armed; the guard makes arm/consume race-free across threads.
}  // namespace

namespace {


} // namespace

namespace {
}  // namespace

} // namespace prosper::gpu
