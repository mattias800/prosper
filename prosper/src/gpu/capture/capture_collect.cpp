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
#include "gpu/capture/gpu_capture_internal.hpp"
#include "gpu/capture/gpu_capture_internal.hpp"
#include "gpu/capture/gpu_capture_internal.hpp"

namespace prosper::gpu {
namespace {


// `own_mip_chain_allocations` extends a materializable mip chain's range to the WHOLE guest
// allocation (#3202). A tiled chain stores level zero last, so the other levels sit BELOW the
// address the descriptor names and the ordinary per-resource range cannot reach them; without this
// a capsule replays with a single-level image and gpu_replay declines IMAGE_LOAD_MIP. It is a
// parameter rather than unconditional because it grows the capture, and a capture that fails is
// worse than one that cannot study this one operation -- see the caller's fallback.
bool collect_intervals(const std::vector<DrawItem>& draws,
                       const std::vector<ComputeItem>& computes,
                       const std::vector<GpuState::DmaCopy>& dma_copies,
                       uint64_t resource_limit_bytes, bool own_mip_chain_allocations,
                       std::vector<Interval>& intervals, std::string& error) {
    uint64_t total = 0;
    auto add_table = [&](const ShaderResourceTable* t,
                         const std::set<uint32_t>* used_bindings) -> bool {
        if (!t) return true;
        for (const auto& r : t->resources) {
            if (!valid_shader_buffer_table_contract(r)) {
                error = "resource has an invalid buffer descriptor-table contract";
                return false;
            }
            if (is_compute_internal_gds(r) || is_gta5_packed_pointer_resource(r) ||
                is_indirect_pointer_relocation_resource(r))
                continue;
            // Runtime tables intentionally retain candidates recovered while folding the guest
            // shader even when the final SPIR-V does not reference them. The Vulkan backend and
            // dependency graph both follow reflection at this boundary; capture must do the same.
            // Otherwise one unused V# with a multi-gigabyte declared range can make an exact
            // submit impossible to retain even though no emitted instruction can read it.
            if (used_bindings && !used_bindings->contains(r.binding) &&
                !capture_authority_requires_backing(t, r))
                continue;
            if (r.table_index_count) {
                for (const ShaderBufferTableEntry& entry : r.table_entries) {
                    const uint64_t n = entry.size;
                    if (!n) continue;
                    if (n > max_blob_bytes() || !entry.gpu_addr ||
                        entry.gpu_addr > std::numeric_limits<uint64_t>::max() - n) {
                        error = "buffer descriptor-table entry capture range is invalid or exceeds "
                                "the per-resource ceiling";
                        return false;
                    }
                    intervals.push_back({entry.gpu_addr, entry.gpu_addr + n});
                }
                continue;
            }
            uint64_t n = resource_footprint(r);
            if (n) {
                if (n > max_blob_bytes() || r.gpu_addr > std::numeric_limits<uint64_t>::max() - n) {
                    char detail[512];
                    std::snprintf(
                        detail, sizeof(detail),
                        "resource capture range is invalid or exceeds the %llu MiB per-resource "
                        "ceiling (raise with PROSPER_CAPTURE_BLOB_MAX_MB; note "
                        "PROSPER_CAPTURE_BUNDLE_MAX_MB is the TOTAL budget and does not govern this): "
                        "binding=%u class=%u "
                        "addr=0x%llx declared=%llu footprint=%llu format=%u components=%u "
                        "extent=%ux%ux%u img-dim=%u tile=%u layer-stride=%llu "
                        "layer-mip-offset=%llu mip-tail=%d/%llu",
                        static_cast<unsigned long long>(max_blob_bytes() >> 20),
                        r.binding, static_cast<unsigned>(r.cls),
                        static_cast<unsigned long long>(r.gpu_addr),
                        static_cast<unsigned long long>(r.size),
                        static_cast<unsigned long long>(n), static_cast<unsigned>(r.format),
                        r.num_components, r.width, r.height, r.depth, r.img_dim, r.tile_mode,
                        static_cast<unsigned long long>(r.layer_stride_bytes),
                        static_cast<unsigned long long>(r.layer_mip_offset_bytes),
                        r.in_mip_tail ? 1 : 0,
                        static_cast<unsigned long long>(r.mip_tail_bytes));
                    error = detail;
                    return false;
                }
                intervals.push_back({r.gpu_addr, r.gpu_addr + n});
                // The rest of a declared mip chain lives below gpu_addr. Own it too, so replay can
                // materialize the same levels the live backend does. Merged with the range above,
                // so an over-large or under-anchored span is skipped rather than made to fail: the
                // capsule then replays exactly as it did before this change.
                uint64_t chain_prefix = 0, chain_bytes = 0;
                if (own_mip_chain_allocations &&
                    prosper::gpu::shader_resource_compute_mip_chain_allocation(
                        r, chain_prefix, chain_bytes) &&
                    chain_prefix <= r.gpu_addr && chain_bytes <= max_blob_bytes() &&
                    chain_bytes >= chain_prefix &&
                    r.gpu_addr - chain_prefix <= std::numeric_limits<uint64_t>::max() - chain_bytes)
                    intervals.push_back({r.gpu_addr - chain_prefix,
                                         r.gpu_addr - chain_prefix + chain_bytes});
            }
            n = dcc_metadata_footprint(r);
            if (n) {
                if (n > max_blob_bytes() || r.metadata_addr > std::numeric_limits<uint64_t>::max() - n) {
                    error = "DCC metadata capture range is invalid or exceeds the per-resource ceiling "
                            "(raise with PROSPER_CAPTURE_BLOB_MAX_MB, not "
                            "PROSPER_CAPTURE_BUNDLE_MAX_MB)"; return false;
                }
                intervals.push_back({r.metadata_addr, r.metadata_addr + n});
            }
        }
        return true;
    };
    for (const auto& d : draws)
        if (!add_table(d.vrt.get(), nullptr) || !add_table(d.prt.get(), nullptr))
            return false;
    for (const auto& c : computes) {
        std::set<uint32_t> compute_bindings;
        const bool compute_reflected = capture_reflected_bindings(
            c.spirv, c.resources.get(), 0u, SpirvShaderStage::Compute,
            compute_bindings);
        if (!add_table(c.resources.get(),
                       compute_reflected ? &compute_bindings : nullptr)) {
            const std::string cause = error;
            if (cause.find("resource capture range is invalid or exceeds") ==
                std::string::npos)
                return false;
            char context[768];
            std::snprintf(
                context, sizeof(context),
                "compute program=0x%llx reflection=%s reflected-bindings=%zu: %s",
                static_cast<unsigned long long>(c.code_addr),
                compute_reflected ? "complete" : "incomplete",
                compute_bindings.size(), cause.c_str());
            error = context;
            return false;
        }
    }
    for (const auto& copy : dma_copies) {
        const bool destination_gds = captured_dma_destination_is_gds(copy.sels);
        const bool source_gds = captured_dma_source_is_gds(copy.sels);
        if ((!destination_gds && !copy.dst) || !copy.src || !copy.bytes ||
            (!destination_gds &&
             copy.dst > std::numeric_limits<uint64_t>::max() - copy.bytes) ||
            copy.src > std::numeric_limits<uint64_t>::max() - copy.bytes) {
            error = "ordered DMA capture range is invalid";
            return false;
        }
        if (source_gds) {
            error = "ordered DMA capture has an unsupported GDS source";
            return false;
        }
        if (!destination_gds)
            intervals.push_back({copy.dst, copy.dst + copy.bytes});
        intervals.push_back({copy.src, copy.src + copy.bytes});
    }
    std::sort(intervals.begin(), intervals.end(), [](auto a, auto b) { return a.begin < b.begin; });
    std::vector<Interval> merged;
    for (auto x : intervals) {
        if (!merged.empty() && x.begin <= merged.back().end) merged.back().end = std::max(merged.back().end, x.end);
        else merged.push_back(x);
    }
    uint64_t largest = 0;
    for (auto x : merged) {
        uint64_t n = x.end - x.begin;
        if (total > kMaxTotalBlobBytes - n) { error = "capture resource data exceeds 3 GiB"; return false; }
        total += n;
        largest = std::max(largest, n);
    }
    if (total > resource_limit_bytes) {
        error = "capture resource data requires " + std::to_string((total + (1u << 20) - 1) >> 20) +
                " MiB across " + std::to_string(merged.size()) + " range(s), largest " +
                std::to_string((largest + (1u << 20) - 1) >> 20) + " MiB; limit is " +
                std::to_string(resource_limit_bytes >> 20) +
                " MiB (raise PROSPER_GPU_CAPTURE_MAX_MB or set "
                "PROSPER_GPU_CAPTURE_METADATA_ONLY=1)";
        return false;
    }
    intervals = std::move(merged); return true;
}

bool capture_table(const ShaderResourceTable* src, const std::vector<Interval>& intervals,
                   bool include_resource_data, bool allow_packed_pointer,
                   GpuCapturedTable& dst, std::string& error,
                   const std::set<uint32_t>* used_bindings = nullptr) {
    dst.present = src != nullptr;
    if (!src) return true;
    for (const auto& r : src->resources) {
        if (!valid_shader_buffer_table_contract(r)) {
            error = "resource has an invalid buffer descriptor-table contract";
            return false;
        }
        if (!allow_packed_pointer &&
            (is_gta5_packed_pointer_marker_candidate(r) ||
             is_indirect_pointer_relocation_marker_candidate(r))) {
            error = "packed-pointer state is only valid for a compute dispatch";
            return false;
        }
        GpuCapturedResource c;
        c.resource = r;
        // Captures own their bytes through blob references, never through the caller's live backing
        // pointers. Arrays return early below, so clear the parent and metadata pointers before the
        // branch just as the scalar path has always done.
        c.resource.host_data = nullptr;
        c.resource.host_data_size = 0;
        c.resource.host_data_prefix_bytes = 0;
        c.resource.dcc_metadata_host_data = nullptr;
        c.resource.dcc_metadata_host_data_size = 0;
        if (!include_resource_data)
            c.resource.indirect_pointer_relocation = {};
        const bool capture_backing = !used_bindings ||
            used_bindings->contains(r.binding) ||
            capture_authority_requires_backing(src, r);
        if (r.table_index_count) {
            c.table_entry_blobs.resize(r.table_entries.size());
            for (size_t index = 0; index < r.table_entries.size(); ++index) {
                const ShaderBufferTableEntry& source = r.table_entries[index];
                ShaderBufferTableEntry& captured = c.resource.table_entries[index];
                captured.host_data = nullptr;
                captured.host_data_size = 0;
                if (source.size && include_resource_data && capture_backing &&
                    !assign_blob_range(intervals, source.gpu_addr, source.size,
                                       c.table_entry_blobs[index].blob_index,
                                       c.table_entry_blobs[index].blob_offset,
                                       "buffer descriptor-table entry was not assigned to a capture blob",
                                       error))
                    return false;
            }
            dst.resources.push_back(std::move(c));
            continue;
        }
        if (r.cls == ResourceClass::Texture &&
            (r.img_dim == 1u || (r.img_dim == 5u && r.layer_stride_bytes)) &&
            r.tile_mode == static_cast<uint32_t>(TileMode::Linear) &&
            !r.compression_enabled) {
            const uint32_t bc = bc_block_bytes(r.format);
            const uint32_t resource_width = r.width ? r.width : 4u;
            const uint32_t pitch_width = bc
                ? resource_width / 4u + static_cast<uint32_t>(resource_width % 4u != 0u)
                : resource_width;
            uint32_t bpt = bc ? bc : data_format_bytes(r.format) *
                (r.num_components ? r.num_components : 1u);
            if (bpt == 0) bpt = 4;
            c.resource.linear_row_pitch_bytes = resolved_linear_row_pitch(
                r, pitch_width, bpt);
        }
        c.metadata_size = dcc_metadata_footprint(r);
        c.resource.dcc_metadata_size = c.metadata_size;
        uint64_t n = resource_footprint(r);
        c.captured_size = n;
        if ((is_compute_internal_gds(r) || is_gta5_packed_pointer_resource(r) ||
             is_indirect_pointer_relocation_resource(r)) &&
            include_resource_data) {
            const uint64_t internal_size = is_compute_internal_gds(r) ? n : r.host_data_size;
            if (!r.host_data || r.host_data_size < internal_size ||
                (is_gta5_packed_pointer_resource(r) &&
                 !is_gta5_packed_pointer_serialized_shadow(
                     r, r.host_data, r.host_data_size)) ||
                (is_indirect_pointer_relocation_resource(r) &&
                 !is_indirect_pointer_relocation_serialized(
                     r, r.host_data, r.host_data_size))) {
                error = "compute internal resource has no complete host backing";
                return false;
            }
            c.internal_bytes.assign(r.host_data, r.host_data + internal_size);
        } else if (n && include_resource_data && capture_backing &&
            !assign_blob_range(intervals, r.gpu_addr, n, c.blob_index, c.blob_offset,
                               "resource was not assigned to a capture blob", error)) return false;
        if (c.metadata_size && include_resource_data && capture_backing &&
            !assign_blob_range(intervals, r.metadata_addr, c.metadata_size,
                               c.metadata_blob_index, c.metadata_blob_offset,
                               "DCC metadata was not assigned to a capture blob", error)) return false;
        dst.resources.push_back(std::move(c));
    }
    return true;
}

bool capture_raw_shader_version(uint64_t addr, const CaptureMemoryReader& reader,
                                GpuCaptureFile& capture, uint64_t& raw_words,
                                std::map<uint64_t, uint32_t>& index_by_address,
                                uint32_t& index, std::string& error) {
    index = 0xFFFFFFFFu;
    if (!addr) return true;
    const auto known = index_by_address.find(addr);
    if (known != index_by_address.end()) {
        index = known->second;
        return true;
    }
    std::vector<uint32_t> words(kMaxRawShaderWords);
    const size_t bytes_read = std::min<size_t>(
        reader(addr, reinterpret_cast<uint8_t*>(words.data()),
               words.size() * sizeof(uint32_t)),
        words.size() * sizeof(uint32_t));
    words.resize(bytes_read / sizeof(uint32_t));
    if (words.empty()) {
        index_by_address.emplace(addr, index);
        return true;
    }
    std::vector<Rdna2Inst> instructions;
    const size_t consumed = rdna2_walk(words.data(), words.size(), instructions);
    // A few compiler-generated shaders address constant tables stored after S_ENDPGM through an
    // s_getpc_b64-built descriptor. Keep the same proven code span as the live recompiler cache so
    // an offline raw replay sees every byte that affected the generated SPIR-V. The span helper is
    // deliberately fail-closed: unrelated post-program guest memory is still discarded.
    const size_t recompile_span = rdna2_recompile_code_span(words.data(), words.size());
    const size_t captured_span = std::max(consumed, recompile_span);
    if (captured_span && captured_span < words.size()) words.resize(captured_span);
    const bool has_endpgm = !instructions.empty() && instructions.back().is_end;
    const uint64_t hash = shader_hash(words);
    auto existing = std::find_if(capture.raw_shader_versions.begin(),
                                 capture.raw_shader_versions.end(), [&](const auto& candidate) {
        return candidate.content_hash == hash && candidate.words == words;
    });
    if (existing != capture.raw_shader_versions.end()) {
        index = static_cast<uint32_t>(existing - capture.raw_shader_versions.begin());
        index_by_address.emplace(addr, index);
        return true;
    }
    if (capture.raw_shader_versions.size() >= kMaxResources ||
        raw_words > kMaxShaderWords - words.size()) {
        error = "raw shader data exceeds its bounded limit";
        return false;
    }
    index = static_cast<uint32_t>(capture.raw_shader_versions.size());
    raw_words += words.size();
    capture.raw_shader_versions.push_back({hash, has_endpgm, std::move(words)});
    index_by_address.emplace(addr, index);
    return true;
}

bool capture_failure_diagnostics(
    const std::vector<OperationRealizationFailure>& failures,
    const CaptureMemoryReader& reader, GpuCaptureFile& capture, uint64_t& raw_words,
    std::map<uint64_t, uint32_t>& raw_shader_index_by_address,
    std::string& error) {
    if (failures.size() > kMaxOperations) {
        error = "invalid failed-operation diagnostic count";
        return false;
    }
    for (const auto& failure : failures) {
        GpuCapturedOperationFailure diagnostic;
        diagnostic.kind = failure.kind;
        diagnostic.source_index = failure.index;
        diagnostic.command_order = failure.command_order;
        diagnostic.reason = failure.reason;
        diagnostic.pipeline_present = failure.pipeline_present;
        diagnostic.pipeline = failure.pipeline;
        diagnostic.color0_base = failure.color0_base;
        diagnostic.color0_width = failure.color0_width;
        diagnostic.color0_height = failure.color0_height;
        diagnostic.color1_base = failure.color1_base;
        diagnostic.color1_width = failure.color1_width;
        diagnostic.color1_height = failure.color1_height;
        diagnostic.color_targets = failure.color_targets;
        diagnostic.color_targets[0] = {
            diagnostic.color0_base, diagnostic.color0_width, diagnostic.color0_height};
        diagnostic.color_targets[1] = {
            diagnostic.color1_base, diagnostic.color1_width, diagnostic.color1_height};
        diagnostic.vertex_count = failure.vertex_count;
        diagnostic.compute_launch = failure.compute_launch;
        for (const auto& runtime_stage : failure.stages) {
            GpuCapturedStageDiagnostic stage;
            stage.stage = runtime_stage.stage;
            stage.program_addr = runtime_stage.program_addr;
            stage.recompiled = runtime_stage.recompiled;
            stage.resource_table_present = runtime_stage.resources != nullptr;
            stage.resource_count = runtime_stage.resources
                ? static_cast<uint32_t>(runtime_stage.resources->resources.size()) : 0;
            stage.coverage = runtime_stage.coverage;
            stage.descriptor_issue_count = runtime_stage.descriptor_issue_count;
            stage.first_descriptor_issue = runtime_stage.first_descriptor_issue;
            stage.recompile_config = runtime_stage.recompile_config;
            stage.recompile_config_available = runtime_stage.recompile_config_available;
            // A retained COMPUTE failure carries compute state, including any packed-pointer or
            // indirect-relocation marker its table acquired before the dispatch failed. The guard
            // exists to keep compute-only state out of GRAPHICS tables — its own message says
            // "only valid for a compute dispatch" — and passing false unconditionally here rejected
            // exactly the case it is meant to permit. GTA V's later 234-dispatch submit could not be
            // captured at all because of it, which is the phase its remaining device loss lives in
            // (#2481). The realized-compute table at the other call site already allows this.
            const bool compute_stage =
                runtime_stage.stage == ShaderProgramStage::Compute;
            if (!capture_table(runtime_stage.resources.get(), {}, false, compute_stage,
                               stage.resource_table, error)) return false;
            if (!capture_raw_shader_version(stage.program_addr, reader, capture, raw_words,
                                            raw_shader_index_by_address,
                                            stage.raw_shader_index, error)) return false;
            if (stage.raw_shader_index < capture.raw_shader_versions.size()) {
                const auto& raw = capture.raw_shader_versions[stage.raw_shader_index].words;
                stage.coverage = recompile_coverage(raw.data(), raw.size());
            }
            diagnostic.stages.push_back(std::move(stage));
        }
        capture.failure_diagnostics.push_back(std::move(diagnostic));
    }
    for (const auto& operation : capture.operations) {
        if (operation.realized) continue;
        const auto existing = std::find_if(capture.failure_diagnostics.begin(),
                                           capture.failure_diagnostics.end(), [&](const auto& candidate) {
            return operation_identity(candidate.kind, candidate.source_index,
                                      candidate.command_order) ==
                   operation_identity(operation.kind, operation.source_index,
                                      operation.command_order);
        });
        if (existing == capture.failure_diagnostics.end()) {
            GpuCapturedOperationFailure unknown;
            unknown.kind = operation.kind;
            unknown.source_index = operation.source_index;
            unknown.command_order = operation.command_order;
            unknown.reason = RealizationFailureReason::Unknown;
            capture.failure_diagnostics.push_back(std::move(unknown));
        }
    }
    return validate_failure_diagnostics(capture, error);
}

} // namespace

bool capture_submit_items(const std::vector<DrawItem>& draws,
                          const std::vector<ComputeItem>& computes,
                          const std::vector<SubmitOperation>& operations,
                          const GpuCaptureMetadata& metadata,
                          const CaptureMemoryReader& reader, GpuCaptureFile& out,
                          std::string& error, const CaptureRttSeedReader& rtt_reader,
                          const std::vector<OperationRealizationFailure>& failures,
                          const std::vector<GpuState::DmaCopy>& dma_copies,
                          uint64_t resource_limit_bytes_override,
                          const uint8_t* pre_submit_compute_gds,
                          size_t pre_submit_compute_gds_bytes) {
    error.clear(); out = {}; out.format_version = kVersion; out.metadata = metadata;
    out.failure_diagnostics_available = true;
    if (draws.size() > kMaxDraws || computes.size() > kMaxComputes ||
        operations.size() > kMaxOperations) {
        error = "capture item or operation count is invalid";
        return false;
    }
    if (!reader) { error = "capture memory reader is missing"; return false; }
    const bool include_resource_data = !env_enabled("PROSPER_GPU_CAPTURE_METADATA_ONLY");
    uint64_t resource_limit_bytes = 0;
    if (!capture_resource_limit(resource_limit_bytes, error, resource_limit_bytes_override)) return false;
    if (!include_resource_data && std::none_of(
            out.metadata.renderer_env.begin(), out.metadata.renderer_env.end(),
            [](const auto& entry) { return entry.first == "PROSPER_GPU_CAPTURE_METADATA_ONLY"; }))
        out.metadata.renderer_env.emplace_back("PROSPER_GPU_CAPTURE_METADATA_ONLY", "1");
    std::vector<Interval> intervals;
    if (include_resource_data) {
        // Prefer owning each materializable mip chain's whole allocation (#3202), but never let
        // that extension be the reason a capture fails: it can only push the total over the
        // resource budget, and a capsule that omits the chain still replays everything it always
        // did. Retry without it and report THAT failure, so the error a user sees is the one they
        // would have seen before this change.
        std::string chain_error;
        if (!collect_intervals(draws, computes, dma_copies, resource_limit_bytes,
                               /*own_mip_chain_allocations=*/true, intervals, chain_error)) {
            intervals.clear();
            if (!collect_intervals(draws, computes, dma_copies, resource_limit_bytes,
                                   /*own_mip_chain_allocations=*/false, intervals, error))
                return false;
        }
    }
    for (auto& x : intervals) {
        GpuCaptureBlob b; b.guest_addr = x.begin; b.bytes.resize(static_cast<size_t>(x.end - x.begin), 0);
        b.bytes_read = std::min<uint64_t>(reader(x.begin, b.bytes.data(), b.bytes.size()), b.bytes.size());
        b.content_hash = gpu_capture_hash(b.bytes);
        auto existing = std::find_if(out.blobs.begin(), out.blobs.end(), [&](const auto& candidate) {
            return candidate.content_hash == b.content_hash && candidate.bytes == b.bytes;
        });
        if (existing == out.blobs.end()) {
            x.blob_index = static_cast<uint32_t>(out.blobs.size());
            out.blobs.push_back(std::move(b));
        } else {
            x.blob_index = static_cast<uint32_t>(existing - out.blobs.begin());
        }
    }
    uint64_t raw_shader_words = 0;
    std::map<uint64_t, uint32_t> raw_shader_index_by_address;
    for (const auto& d : draws) {
        GpuCapturedDraw c; c.vs = d.vs_words(); c.gs = d.gs_words(); c.fs = d.fs_words();
        c.ps = d.ps; c.vertex_count = d.vertex_count;
        c.instance_count = d.instance_count;
        c.raw_draw_count = d.raw_draw_count; c.raw_indexed = d.raw_indexed;   // #1256
        c.raw_draw_modifier = d.raw_draw_modifier;
        c.vertex_offset = d.vertex_offset;
        c.indices = d.indices; c.color0_base = d.color0_base;
        c.color0_width = d.color0_width; c.color0_height = d.color0_height;
        c.color1_base = d.color1_base;
        c.color1_width = d.color1_width; c.color1_height = d.color1_height;
        c.color_targets = d.color_targets;
        c.color_targets[0] = {c.color0_base, c.color0_width, c.color0_height};
        c.color_targets[1] = {c.color1_base, c.color1_width, c.color1_height};
        c.draw_index = d.draw_index; c.command_order = d.command_order;
        if (!capture_raw_shader_version(d.vs_guest_addr, reader, out, raw_shader_words,
                                        raw_shader_index_by_address,
                                        c.vs_raw_shader_index, error) ||
            !capture_raw_shader_version(d.fs_guest_addr, reader, out, raw_shader_words,
                                        raw_shader_index_by_address,
                                        c.fs_raw_shader_index, error) ||
            !capture_raw_shader_version(d.vs_chain_guest_addr, reader, out, raw_shader_words,
                                        raw_shader_index_by_address,
                                        c.vs_chain_raw_shader_index, error)) return false;
        c.vertex_lds_dwords = d.vertex_lds_dwords;
        c.pixel_inputs = d.pixel_inputs;
        c.system_inputs = d.system_inputs;
        c.has_pixel_inputs = d.has_pixel_inputs;
        c.has_system_inputs = d.has_system_inputs;
        if (!capture_table(d.vrt.get(), intervals, include_resource_data, false,
                           c.vrt, error) ||
            !capture_table(d.prt.get(), intervals, include_resource_data, false,
                           c.prt, error))
            return false;
        out.draws.push_back(std::move(c));
    }
    for (const auto& compute : computes) {
        GpuCapturedCompute c;
        c.spirv = compute.spirv;
        c.launch = compute.launch;
        c.code_addr = compute.code_addr;
        c.dispatch_index = compute.dispatch_index;
        c.submit_no = compute.submit_no;
        c.command_order = compute.command_order;
        c.required_subgroup_size = compute.required_subgroup_size;
        c.recompile_config = compute.recompile_config;
        c.recompile_config_available = compute.recompile_config_available;
        if (c.recompile_config_available &&
            !capture_raw_shader_version(compute.code_addr, reader, out, raw_shader_words,
                                        raw_shader_index_by_address,
                                        c.raw_shader_index, error)) return false;
        std::set<uint32_t> compute_bindings;
        const bool compute_reflected = capture_reflected_bindings(
            compute.spirv, compute.resources.get(), 0u,
            SpirvShaderStage::Compute, compute_bindings);
        if (!capture_table(compute.resources.get(), intervals, include_resource_data, true,
                           c.resources, error,
                           compute_reflected ? &compute_bindings : nullptr))
            return false;
        out.computes.push_back(std::move(c));
    }
    uint32_t gds_snapshot_blob_index = 0xFFFFFFFFu;
    if (include_resource_data && std::any_of(
            dma_copies.begin(), dma_copies.end(), [](const GpuState::DmaCopy& copy) {
                return captured_dma_destination_is_gds(copy.sels);
            })) {
        const uint8_t* snapshot = pre_submit_compute_gds_bytes
            ? pre_submit_compute_gds : nullptr;
        size_t snapshot_bytes = pre_submit_compute_gds_bytes;
        auto find_internal_snapshot = [&](const GpuCapturedTable& table) {
            if (snapshot) return;
            for (const auto& resource : table.resources) {
                if (!is_compute_internal_gds(resource.resource) ||
                    resource.internal_bytes.empty())
                    continue;
                snapshot = resource.internal_bytes.data();
                snapshot_bytes = resource.internal_bytes.size();
                return;
            }
        };
        for (const auto& draw : out.draws) {
            find_internal_snapshot(draw.vrt);
            find_internal_snapshot(draw.prt);
        }
        for (const auto& compute : out.computes)
            find_internal_snapshot(compute.resources);
        if (!snapshot || snapshot_bytes != 64u * 1024u) {
            error = "memory-to-GDS capture has no complete pre-submit GDS snapshot";
            return false;
        }
        for (const auto& compute : out.computes) {
            for (const auto& resource : compute.resources.resources) {
                if (is_compute_internal_gds(resource.resource) &&
                    !resource.internal_bytes.empty() &&
                    (resource.internal_bytes.size() != snapshot_bytes ||
                     std::memcmp(resource.internal_bytes.data(), snapshot,
                                 snapshot_bytes) != 0)) {
                    error = "compute GDS resources disagree with the pre-submit snapshot";
                    return false;
                }
            }
        }
        GpuCaptureBlob blob;
        blob.guest_addr = 0;
        blob.bytes_read = snapshot_bytes;
        blob.bytes.assign(snapshot, snapshot + snapshot_bytes);
        blob.content_hash = gpu_capture_hash(blob.bytes);
        gds_snapshot_blob_index = static_cast<uint32_t>(out.blobs.size());
        out.blobs.push_back(std::move(blob));
    }
    out.dma_copies.reserve(dma_copies.size());
    for (const auto& copy : dma_copies) {
        GpuCapturedDmaCopy captured;
        captured.dst = copy.dst; captured.src = copy.src; captured.bytes = copy.bytes;
        captured.sels = copy.sels; captured.command_order = copy.command_order;
        captured.packet_addr = copy.packet_addr;
        const bool destination_gds = captured_dma_destination_is_gds(copy.sels);
        if (include_resource_data && destination_gds) {
            captured.destination_blob_index = gds_snapshot_blob_index;
            captured.destination_blob_offset = copy.dst;
        }
        if (include_resource_data &&
            ((!destination_gds && !assign_blob_range(intervals, copy.dst, copy.bytes,
                                captured.destination_blob_index,
                                captured.destination_blob_offset,
                                "ordered DMA destination was not assigned to a capture blob", error)) ||
             !assign_blob_range(intervals, copy.src, copy.bytes,
                                captured.source_blob_index, captured.source_blob_offset,
                                "ordered DMA source was not assigned to a capture blob", error)))
            return false;
        out.dma_copies.push_back(captured);
    }
    std::unordered_set<uint64_t> realized_draws, realized_computes;
    for (const auto& draw : out.draws) realized_draws.insert(draw.draw_index);
    for (const auto& compute : out.computes) realized_computes.insert(compute.dispatch_index);
    for (const auto& operation : operations) {
        bool realized = false;
        switch (operation.kind) {
            case SubmitOperationKind::Draw:
                realized = realized_draws.count(operation.index) != 0;
                break;
            case SubmitOperationKind::Dispatch:
                realized = realized_computes.count(operation.index) != 0;
                break;
            case SubmitOperationKind::DmaCopy:
                if (operation.index >= out.dma_copies.size() ||
                    out.dma_copies[operation.index].command_order != operation.command_order) {
                    error = "ordered DMA operation references an invalid record";
                    return false;
                }
                realized = true;
                break;
        }
        out.operations.push_back({operation.kind, operation.index, operation.command_order, realized});
    }
    if (!validate_dma_copies(out, error)) return false;
    if (!capture_failure_diagnostics(failures, reader, out, raw_shader_words,
                                     raw_shader_index_by_address, error)) return false;
    collect_shader_versions(out);
    if (rtt_reader && include_resource_data) {
        std::vector<uint64_t> candidates;
        auto add_table = [&](const ShaderResourceTable* table) {
            if (!table) return;
            for (const auto& r : table->resources)
                if ((r.cls == ResourceClass::Texture || r.cls == ResourceClass::StorageImage) && r.gpu_addr)
                    candidates.push_back(r.gpu_addr);
        };
        for (const auto& item : draws) {
            add_table(item.vrt.get()); add_table(item.prt.get());
            for (const auto& target : item.color_targets)
                if (target.base) candidates.push_back(target.base);
            // Preserve direct callers that only populate the named compatibility fields.
            if (item.color0_base) candidates.push_back(item.color0_base);
            if (item.color1_base) candidates.push_back(item.color1_base);
        }
        for (const auto& item : computes) add_table(item.resources.get());
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        uint64_t total = 0;
        for (uint64_t addr : candidates) {
            GpuCaptureRttSeed seed;
            if (!rtt_reader(addr, seed)) continue;
            if (seed.guest_addr != addr) { error = "RTT seed reader returned the wrong guest address"; return false; }
            if (!validate_rtt_seed(seed, error)) return false;
            if (total > kMaxTotalRttSeedBytes - seed.rgba.size()) {
                error = "capture RTT seed data exceeds 1 GiB"; return false;
            }
            total += seed.rgba.size(); out.rtt_seeds.push_back(std::move(seed));
        }
    }
    return validate_dma_copies(out, error);
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
