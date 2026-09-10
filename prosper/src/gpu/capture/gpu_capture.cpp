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

CaptureRttSeedSnapshotReader g_rtt_seed_snapshot_reader;
ReplayRttSeedWriter g_rtt_seed_writer;
CaptureDsSeedSnapshotReader g_ds_seed_snapshot_reader;
ReplayDsSeedWriter g_ds_seed_writer;

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

const char* env_or_empty(const char* name) { const char* v = std::getenv(name); return v ? v : ""; }

using OperationIdentity = std::tuple<uint8_t, uint64_t, uint64_t>;

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

// Exported for test only (#2440): `max_blob_bytes()` caches in a `static`, so a test process can see
// exactly one value. The parse/clamp rules are the part worth asserting, so expose them purely.
uint64_t capture_blob_max_bytes_from_env_for_test(const char* env, std::string* note) {
    return blob_max_bytes_from_env(env, note);
}

void annotate_gpu_capture_scanout(GpuCaptureMetadata& metadata) {
    const uint64_t address = present_front_address();
    if (!address) return;
    char value[24];
    std::snprintf(value, sizeof(value), "0x%llx",
                  static_cast<unsigned long long>(address));
    const auto existing = std::find_if(metadata.renderer_env.begin(), metadata.renderer_env.end(),
        [](const auto& entry) { return entry.first == kGpuReplayScanoutAddressEnv; });
    if (existing == metadata.renderer_env.end())
        metadata.renderer_env.emplace_back(kGpuReplayScanoutAddressEnv, value);
    else
        existing->second = value;
}

void annotate_gpu_capture_save_roots(GpuCaptureMetadata& metadata) {
    const std::string effective = savedata0_root();
    const auto existing = std::find_if(metadata.renderer_env.begin(), metadata.renderer_env.end(),
        [](const auto& entry) { return entry.first == kGpuCaptureSave0Env; });
    if (existing == metadata.renderer_env.end())
        metadata.renderer_env.emplace_back(kGpuCaptureSave0Env, effective);
    else
        existing->second = effective;
}

uint64_t parse_gpu_replay_scanout_address(const char* value) {
    if (!value || !*value) return 0;
    char* end = nullptr;
    const uint64_t parsed = std::strtoull(value, &end, 0);
    return end && end != value && !*end ? parsed : 0;
}

uint64_t gpu_capture_resource_footprint(const ShaderResource& resource) {
    return resource_footprint(resource);
}

uint64_t gpu_capture_dcc_metadata_footprint(const ShaderResource& resource) {
    return dcc_metadata_footprint(resource);
}

// FNV-1a's prime with a basis of 1469598103934665603 (0x14650fb0739d0383) — the FNV-1a 64 offset
// basis with a digit dropped. **Do not "correct" it.** It is a sound 64-bit hash (the multiplier is
// odd, so the mixing is bijective) and nothing here claims FNV compliance, while the value is baked
// into every recorded capture, bundle manifest and `--inspect-only` seed hash in the project's
// evidence trail — changing it would invalidate all of them for a cosmetic gain. It is recorded
// here because the dropped digit reads as a typo, and because anyone verifying one of these hashes
// offline with a stock FNV-1a 64 gets a value that can never match and is liable to read the
// mismatch as "this buffer is not zero" (#1968, `docs/GRAPHICS.md` § Ruled out).
uint64_t gpu_capture_hash(const uint8_t* data, size_t size) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) { h ^= data[i]; h *= 1099511628211ull; }
    return h;
}

const char* shader_program_stage_name(ShaderProgramStage stage) {
    switch (stage) {
        case ShaderProgramStage::Vertex: return "vertex";
        case ShaderProgramStage::Fragment: return "fragment";
        case ShaderProgramStage::Compute: return "compute";
    }
    return "unknown";
}

const char* realization_failure_reason_name(RealizationFailureReason reason) {
    switch (reason) {
        case RealizationFailureReason::None: return "none";
        case RealizationFailureReason::Unknown: return "unknown";
        case RealizationFailureReason::MissingProgram: return "missing-program";
        case RealizationFailureReason::ShaderRecompile: return "shader-recompile";
        case RealizationFailureReason::DescriptorContract: return "descriptor-contract";
        case RealizationFailureReason::NoEffect: return "no-effect";
        case RealizationFailureReason::ZeroVertices: return "zero-vertices";
        case RealizationFailureReason::Filtered: return "filtered";
        case RealizationFailureReason::RetainedDrawNotSelected: return "retained-draw-not-selected";
        case RealizationFailureReason::IndirectArguments: return "indirect-arguments";
        case RealizationFailureReason::IndirectDependencies: return "indirect-dependencies";
        case RealizationFailureReason::ComputeBackendUnavailable:
            return "compute-backend-unavailable";
        case RealizationFailureReason::SuspiciousDispatchSkipped:
            return "suspicious-dispatch-skipped";
        case RealizationFailureReason::ComputeExecutionDeclined:
            return "compute-execution-declined";
    }
    return "unknown";
}

bool capture_draw_items(const std::vector<DrawItem>& items, const GpuCaptureMetadata& metadata,
                        const CaptureMemoryReader& reader, GpuCaptureFile& out, std::string& error,
                        const CaptureRttSeedReader& rtt_reader) {
    if (items.empty()) { error = "capture must contain at least one draw"; return false; }
    std::vector<SubmitOperation> operations;
    operations.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i)
        operations.push_back({SubmitOperationKind::Draw, static_cast<size_t>(items[i].draw_index),
                              items[i].command_order});
    return capture_submit_items(items, {}, operations, metadata, reader, out, error, rtt_reader);
}

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

bool capture_gpustate_submit(const GpuState& state, uint64_t submit_no,
                             uint32_t width, uint32_t height,
                             const GpuCaptureMetadata& metadata,
                             GpuCaptureFile& out, std::string& error,
                             uint64_t resource_limit_bytes) {
    const bool caplog = std::getenv("PROSPER_GPU_CAPTURE_LOG") != nullptr;
    std::vector<OperationRealizationFailure> failures, compute_failures;
    if (caplog) std::fprintf(stderr, "[cap] realize_gpustate_draws...\n");
    std::vector<DrawItem> draws = realize_gpustate_draws(state, 0x10000, 1.0f, 1.0f, &failures);
    if (caplog) std::fprintf(stderr, "[cap] realize draws=%zu; realize_compute_dispatches...\n", draws.size());
    std::vector<ComputeItem> computes = realize_compute_dispatches(state, submit_no, &compute_failures);
    if (caplog) std::fprintf(stderr, "[cap] realize computes=%zu; plan_submit_operations...\n", computes.size());
    failures.insert(failures.end(), std::make_move_iterator(compute_failures.begin()),
                    std::make_move_iterator(compute_failures.end()));
    GpuCaptureMetadata actual = metadata;
    actual.width = width;
    actual.height = height;
    actual.submit_index = submit_no;
    auto ops = plan_submit_operations(state);
    if (caplog) std::fprintf(stderr, "[cap] ops=%zu; capture_submit_items...\n", ops.size());
    // Raw guest memory is normally authoritative. A live renderer-owned target is the exception:
    // ordered DMA reads its host pixels, so overlay those exact bytes into the pre-submit closure.
    CaptureMemoryReader ordered_reader = ordered_gpustate_capture_reader(state);
    return capture_submit_items(draws, computes, ops, actual, ordered_reader, out, error,
                                g_rtt_seed_reader, failures, state.dma_copies,
                                resource_limit_bytes, compute_gds_backing(),
                                compute_gds_size());
}

bool capture_gpustate_target_submit(const GpuState& state, uint64_t submit_no,
                                    uint32_t width, uint32_t height,
                                    uint32_t target_width, uint32_t target_height,
                                    const GpuCaptureMetadata& metadata,
                                    GpuCaptureFile& out, std::string& error) {
    std::vector<DrawItem> draws = realize_gpustate_draws(state);
    draws.erase(std::remove_if(draws.begin(), draws.end(), [&](const DrawItem& draw) {
        return draw.color0_width != target_width || draw.color0_height != target_height;
    }), draws.end());
    std::vector<SubmitOperation> operations;
    operations.reserve(draws.size());
    for (const auto& draw : draws)
        operations.push_back({SubmitOperationKind::Draw, static_cast<size_t>(draw.draw_index),
                              draw.command_order});
    for (size_t i = 0; i < state.dma_copies.size(); ++i)
        operations.push_back({SubmitOperationKind::DmaCopy, i,
                              state.dma_copies[i].command_order});
    std::stable_sort(operations.begin(), operations.end(), [](const auto& a, const auto& b) {
        return a.command_order < b.command_order;
    });
    GpuCaptureMetadata actual = metadata;
    actual.width = width; actual.height = height; actual.submit_index = submit_no;
    return capture_submit_items(draws, {}, operations, actual,
                                ordered_gpustate_capture_reader(state),
                                out, error, g_rtt_seed_reader, {}, state.dma_copies, 0,
                                compute_gds_backing(), compute_gds_size());
}

bool write_gpu_capture(const std::string& path, const GpuCaptureFile& c, std::string& error) {
    std::vector<uint8_t> bytes;
    if (!serialize_gpu_capture(c, bytes, error)) return false;
    std::filesystem::path target(path), temp = target; temp += ".tmp";
    std::error_code ec; if (target.has_parent_path()) std::filesystem::create_directories(target.parent_path(), ec);
    std::ofstream f(temp, std::ios::binary | std::ios::trunc);
    if (!f || !f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        error = "cannot write capture temporary file"; return false;
    }
    f.close(); std::filesystem::rename(temp, target, ec);
    if (ec) { std::filesystem::remove(target, ec); ec.clear(); std::filesystem::rename(temp, target, ec); }
    if (ec) { error = "cannot install capture file: " + ec.message(); return false; }
    return true;
}

bool read_gpu_capture(const std::string& path, GpuCaptureFile& c, std::string& error) {
    error.clear(); std::error_code ec; uint64_t size = std::filesystem::file_size(path, ec);
    if (ec || size > kMaxFileBytes || size > std::numeric_limits<size_t>::max()) { error = "invalid capture file size"; return false; }
    std::vector<uint8_t> bytes(static_cast<size_t>(size)); std::ifstream f(path, std::ios::binary);
    if (!f || !f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        error = "cannot read capture file"; return false;
    }
    return deserialize_gpu_capture(bytes, c, error);
}

bool materialize_gpu_replay(const GpuCaptureFile& c, GpuReplayFrame& out, std::string& error) {
    error.clear();
    out = {};
    if (!validate_dma_copies(c, error) ||
        (c.format_version >= 7u && !validate_failure_diagnostics(c, error)))
        return false;
    out.metadata = c.metadata; out.blobs = c.blobs;
    out.rtt_seeds = c.rtt_seeds; out.ds_seeds = c.ds_seeds;
    out.raw_shader_versions = c.raw_shader_versions;
    out.failure_diagnostics = c.failure_diagnostics;
    out.resource_provenance = c.resource_provenance;
    out.failure_diagnostics_available = c.failure_diagnostics_available;
    out.expected_output_valid = c.expected_output_valid;
    out.expected_output_hash = c.expected_output_hash; out.expected_output_bytes = c.expected_output_bytes;
    size_t resource_reference_count = 0;
    for (const auto& draw : c.draws)
        for (const GpuCapturedTable* table : {&draw.vrt, &draw.prt})
            for (const auto& resource : table->resources)
                resource_reference_count += 1u + resource.resource.table_entries.size();
    for (const auto& compute : c.computes)
        for (const auto& resource : compute.resources.resources)
            resource_reference_count += 1u + resource.resource.table_entries.size();
    resource_reference_count += c.dma_copies.size() * 2;
    out.resource_instances.reserve(resource_reference_count * 2);
    std::map<std::pair<uint32_t, uint64_t>, size_t> instance_by_version_and_base;
    std::map<uint32_t, size_t> internal_instance_by_binding;
    // `prefix_bytes`, when supplied, receives how many bytes of the SAME allocation precede
    // `guest_addr` inside this blob. That is exactly `blob_offset`: a blob's byte i is the guest
    // byte at `blob.guest_addr + i` by construction (`collect_intervals` merges ranges and reads
    // them contiguously, and blob dedup only shares byte-identical content), so the bytes before
    // the resource's own address really are the guest's bytes at those addresses. A tiled mip
    // chain needs them -- it stores level zero last (#3202).
    auto bind_range = [&](uint32_t blob_index, uint64_t blob_offset, uint64_t guest_addr,
                          uint64_t need, uint8_t*& host_data, uint64_t& host_data_size,
                          const char* invalid_error, const char* exceeds_error,
                          const char* offset_error, uint64_t* prefix_bytes = nullptr) {
        if (blob_index == 0xFFFFFFFFu) return true;
        if (blob_index >= out.blobs.size() || blob_offset > out.blobs[blob_index].bytes.size()) {
            error = invalid_error; return false;
        }
        const auto& blob = out.blobs[blob_index].bytes;
        if (need > blob.size() - blob_offset) { error = exceeds_error; return false; }
        if (blob_offset > guest_addr) { error = offset_error; return false; }
        const uint64_t logical_base = guest_addr - blob_offset;
        const auto key = std::make_pair(blob_index, logical_base);
        auto [it, inserted] = instance_by_version_and_base.emplace(key, out.resource_instances.size());
        if (inserted)
            out.resource_instances.push_back({logical_base, blob_index, blob});
        auto& instance = out.resource_instances[it->second].bytes;
        host_data = instance.data() + blob_offset;
        host_data_size = need;
        if (prefix_bytes) *prefix_bytes = blob_offset;
        return true;
    };
    auto table = [&](const GpuCapturedTable& src, bool allow_packed_pointer,
                     std::shared_ptr<ShaderResourceTable>& dst) -> bool {
        if (!src.present) { dst.reset(); return src.resources.empty(); }
        dst = std::make_shared<ShaderResourceTable>();
        for (const auto& x : src.resources) {
            ShaderResource r = x.resource; r.host_data = nullptr; r.host_data_size = 0;
            r.host_data_prefix_bytes = 0;
            for (ShaderBufferTableEntry& entry : r.table_entries) {
                entry.host_data = nullptr;
                entry.host_data_size = 0;
            }
            if (r.table_index_count) {
                if (!valid_shader_buffer_table_contract(r) ||
                    x.table_entry_blobs.size() != r.table_entries.size()) {
                    error = "invalid buffer descriptor-table replay contract";
                    return false;
                }
                for (size_t index = 0; index < r.table_entries.size(); ++index) {
                    ShaderBufferTableEntry& entry = r.table_entries[index];
                    const auto& backing = x.table_entry_blobs[index];
                    if (!bind_range(
                            backing.blob_index, backing.blob_offset, entry.gpu_addr,
                            entry.size, entry.host_data, entry.host_data_size,
                            "buffer descriptor-table entry references an invalid capture blob",
                            "buffer descriptor-table entry exceeds its capture blob",
                            "buffer descriptor-table entry blob offset exceeds its logical address"))
                        return false;
                }
                dst->resources.push_back(std::move(r));
                continue;
            }
            const uint64_t captured_footprint = x.captured_size ? x.captured_size
                : (c.format_version >= 28 ? resource_footprint(r) : legacy_resource_footprint(r));
            if (r.scalar_buffer_dword_count) {
                const uint64_t binding_bytes = shader_resource_buffer_binding_bytes(r);
                if (!binding_bytes || x.captured_size < binding_bytes) {
                    error = "scalar-buffer replay span is shorter than its bound";
                    return false;
                }
            }
            // v1-v27 captured linear images tightly. Preserve that bounded backing so old files stay
            // readable, but derive the real guest pitch for best-effort replay of every retained row.
            if (c.format_version < 28 && r.cls == ResourceClass::Texture && r.img_dim == 1u &&
                r.tile_mode == static_cast<uint32_t>(TileMode::Linear) &&
                !r.compression_enabled && !bc_block_bytes(r.format)) {
                uint32_t bpt = data_format_bytes(r.format) *
                               (r.num_components ? r.num_components : 1u);
                const bool f16 = r.format == DataFormat::Float16 &&
                                 (bpt == 2 || bpt == 4 || bpt == 8);
                if (bpt == 0 || (bpt > 4 && !f16)) bpt = 4;
                r.linear_row_pitch_bytes = resolved_linear_row_pitch(
                    r, r.width ? r.width : 4u, bpt);
            }
            r.dcc_metadata_size = x.metadata_size;
            r.dcc_metadata_host_data = nullptr;
            r.dcc_metadata_host_data_size = 0;
            if (!x.internal_bytes.empty()) {
                const bool gds = is_compute_internal_gds(r);
                const bool packed = c.format_version >= 51u &&
                    is_gta5_packed_pointer_serialized_shadow(
                        r, x.internal_bytes.data(), x.internal_bytes.size());
                const bool relocated = c.format_version >= 53u &&
                    is_indirect_pointer_relocation_serialized(
                        r, x.internal_bytes.data(), x.internal_bytes.size());
                if ((!allow_packed_pointer && (packed || relocated)) ||
                    (!gds && !packed && !relocated)) {
                    error = "invalid compute internal replay state";
                    return false;
                }
                size_t instance_index = 0;
                if (packed || relocated) {
                    instance_index = out.resource_instances.size();
                    out.resource_instances.push_back({0, 0xFFFFFFFFu, x.internal_bytes});
                } else {
                    auto [it, inserted] = internal_instance_by_binding.emplace(
                        r.binding, out.resource_instances.size());
                    if (inserted)
                        out.resource_instances.push_back({0, 0xFFFFFFFFu, x.internal_bytes});
                    instance_index = it->second;
                    auto& instance = out.resource_instances[instance_index].bytes;
                    if (!inserted && instance != x.internal_bytes) {
                        error = "compute GDS resources disagree on initial state";
                        return false;
                    }
                }
                auto& instance = out.resource_instances[instance_index].bytes;
                r.host_data = instance.data();
                r.host_data_size = instance.size();
            } else if (!bind_range(x.blob_index, x.blob_offset, r.gpu_addr, captured_footprint,
                            r.host_data, r.host_data_size,
                            "resource references an invalid capture blob",
                            "resource footprint exceeds capture blob",
                            "resource blob offset exceeds its logical address",
                            &r.host_data_prefix_bytes)) return false;
            if (!bind_range(x.metadata_blob_index, x.metadata_blob_offset, r.metadata_addr,
                            x.metadata_size, r.dcc_metadata_host_data,
                            r.dcc_metadata_host_data_size,
                            "resource DCC metadata references an invalid capture blob",
                            "resource DCC metadata exceeds capture blob",
                            "resource DCC metadata blob offset exceeds its logical address"))
                return false;
            dst->resources.push_back(r);
        }
        return true;
    };
    out.items.reserve(c.draws.size());
    for (const auto& x : c.draws) {
        DrawItem d; d.vs = x.vs; d.gs = x.gs; d.fs = x.fs;
        d.ps = x.ps; d.vertex_count = x.vertex_count;
        d.instance_count = x.instance_count;
        d.raw_draw_count = x.raw_draw_count; d.raw_indexed = x.raw_indexed;   // #1256
        d.raw_draw_modifier = x.raw_draw_modifier;
        d.vertex_offset = x.vertex_offset;
        d.indices = x.indices; d.color0_base = x.color0_base;
        d.color0_width = x.color0_width; d.color0_height = x.color0_height;
        d.color1_base = x.color1_base;
        d.color1_width = x.color1_width; d.color1_height = x.color1_height;
        d.color_targets = x.color_targets;
        d.color_targets[0] = {d.color0_base, d.color0_width, d.color0_height};
        d.color_targets[1] = {d.color1_base, d.color1_width, d.color1_height};
        d.draw_index = x.draw_index; d.command_order = x.command_order;
        d.vs_raw_shader_index = x.vs_raw_shader_index;
        d.fs_raw_shader_index = x.fs_raw_shader_index;
        d.vs_chain_raw_shader_index = x.vs_chain_raw_shader_index;
        d.vertex_lds_dwords = x.vertex_lds_dwords;
        d.pixel_inputs = x.pixel_inputs;
        d.system_inputs = x.system_inputs;
        d.has_pixel_inputs = x.has_pixel_inputs;
        d.has_system_inputs = x.has_system_inputs;
        if (!table(x.vrt, false, d.vrt) || !table(x.prt, false, d.prt)) return false;
        if (d.vrt) d.vrt->vertices_per_instance = d.vertex_count;
        out.items.push_back(std::move(d));
    }
    out.computes.reserve(c.computes.size());
    for (const auto& x : c.computes) {
        if (!validate_captured_null_guarded_raw_store(c, x, error)) return false;
        if (!validate_captured_nullable_output_raw_buffer(c, x, error)) return false;
        if (!validate_captured_gta5_cf9200_no_backing(c, x, error)) return false;
        if (!validate_captured_indirect_pointer_relocations(c, x, error)) return false;
        ComputeItem compute;
        compute.spirv = x.spirv;
        compute.launch = x.launch;
        compute.code_addr = x.code_addr;
        compute.dispatch_index = x.dispatch_index;
        compute.submit_no = x.submit_no;
        compute.command_order = x.command_order;
        compute.required_subgroup_size = x.required_subgroup_size;
        compute.raw_shader_index = x.raw_shader_index;
        compute.recompile_config = x.recompile_config;
        compute.recompile_config_available = x.recompile_config_available;
        compute.null_guarded_raw_store_validated =
            captured_compute_has_null_guarded_raw_store(x);
        compute.nullable_output_raw_buffer_validated =
            captured_compute_has_nullable_output_raw_buffer(x);
        compute.gta5_cf9200_no_backing_validated = false;
        // Re-proved here from the retained raw stream rather than trusted from any serialized flag,
        // which is the same authority rule the three tokens above follow. A capture that carries no
        // raw words for this dispatch simply does not get the proof.
        compute.terminator_only_program_validated =
            x.raw_shader_index < c.raw_shader_versions.size() &&
            rdna2_program_is_terminator_only(
                c.raw_shader_versions[x.raw_shader_index].words.data(),
                c.raw_shader_versions[x.raw_shader_index].words.size());
        if (compute.recompile_config_available)
            compute.user_sgprs = compute.recompile_config.user_sgprs;
        const bool has_packed_pointer_state = c.format_version >= 51u &&
            std::any_of(x.resources.resources.begin(), x.resources.resources.end(),
                        [](const GpuCapturedResource& resource) {
                return is_gta5_packed_pointer_serialized_shadow(
                    resource.resource, resource.internal_bytes.data(),
                    resource.internal_bytes.size());
            });
        const bool has_indirect_pointer_state = c.format_version >= 53u &&
            std::any_of(x.resources.resources.begin(), x.resources.resources.end(),
                        captured_resource_has_indirect_pointer_state);
        if (!table(x.resources, true, compute.resources)) return false;
        const bool has_cf9200_no_backing =
            captured_compute_has_gta5_cf9200_no_backing(x);
        if (has_cf9200_no_backing &&
            (!compute.resources || !compute.recompile_config_available ||
             compute.raw_shader_index >= c.raw_shader_versions.size())) {
            error = "GTA root-record replay lacks exact raw shader or launch state";
            return false;
        }
        // The selected-SBUFFER marker is deliberately derived rather than serialized. A raw replay
        // with complete captured backing can reconstruct it from the exact shader/launch/source
        // domain; metadata-only captures retain their already-compiled SPIR-V and remain materializable.
        if (has_packed_pointer_state &&
            (!compute.resources || !compute.recompile_config_available ||
             compute.raw_shader_index >= c.raw_shader_versions.size())) {
            error = "packed-pointer replay lacks exact raw shader or launch state";
            return false;
        }
        if (has_indirect_pointer_state &&
            (!compute.resources || !compute.recompile_config_available ||
             compute.raw_shader_index >= c.raw_shader_versions.size())) {
            error = "indirect-pointer replay lacks exact raw shader or launch state";
            return false;
        }
        if (compute.resources && compute.recompile_config_available &&
            compute.raw_shader_index < c.raw_shader_versions.size()) {
            const auto& raw = c.raw_shader_versions[compute.raw_shader_index].words;
            // Rebuild this semantic proof from retained guest ISA, never from a serialized enum.
            // Replay still checks the captured descriptors, launch and owned backing at execution.
            if (classify_compute_cpu_fast_path(raw.data(), raw.size()) ==
                    ComputeCpuFastPath::BroadcastBufferU32)
                compute.cpu_fast_path = ComputeCpuFastPath::BroadcastBufferU32;
            if (has_cf9200_no_backing) {
                if (!rdna2_gta5_cf9200_no_backing_dispatch(
                        raw.data(), raw.size(), compute.recompile_config,
                        *compute.resources)) {
                    error = "GTA root-record replay failed exact dispatch validation";
                    return false;
                }
                compute.gta5_cf9200_no_backing_validated = true;
            }
            if (rdna2_gta5_selected_sbuffer_shader(raw.data(), raw.size()))
                (void)discover_rdna2_gta5_selected_sbuffer(
                    raw.data(), raw.size(), compute.recompile_config, *compute.resources);
            if (has_packed_pointer_state &&
                (!rdna2_gta5_packed_pointer_shader(raw.data(), raw.size()) ||
                 !discover_rdna2_gta5_packed_pointer(
                     raw.data(), raw.size(), compute.recompile_config, *compute.resources))) {
                error = "packed-pointer replay failed exact dispatch validation";
                return false;
            }
            if (has_indirect_pointer_state &&
                !discover_rdna2_indirect_pointer_relocations(
                    raw.data(), raw.size(), compute.recompile_config,
                    *compute.resources)) {
                error = "indirect-pointer replay failed exact dispatch validation";
                return false;
            }
        }
        out.computes.push_back(std::move(compute));
    }
    out.dma_copies.reserve(c.dma_copies.size());
    for (const auto& captured : c.dma_copies) {
        ReplayDmaCopy copy;
        copy.dst = captured.dst; copy.src = captured.src; copy.bytes = captured.bytes;
        copy.sels = captured.sels; copy.command_order = captured.command_order;
        copy.packet_addr = captured.packet_addr;
        uint8_t* source = nullptr;
        if (captured_dma_destination_is_gds(captured.sels)) {
            auto internal = internal_instance_by_binding.find(kComputeInternalGdsBinding);
            if (captured.destination_blob_index != 0xFFFFFFFFu) {
                if (captured.destination_blob_index >= out.blobs.size()) {
                    error = "ordered DMA GDS destination references an invalid capture blob";
                    return false;
                }
                const auto& blob = out.blobs[captured.destination_blob_index];
                if (blob.bytes.size() != 64u * 1024u) {
                    error = "ordered DMA GDS destination snapshot is unavailable";
                    return false;
                }
                if (internal == internal_instance_by_binding.end()) {
                    const size_t index = out.resource_instances.size();
                    out.resource_instances.push_back({0, captured.destination_blob_index,
                                                      blob.bytes});
                    internal = internal_instance_by_binding.emplace(
                        kComputeInternalGdsBinding, index).first;
                } else if (out.resource_instances[internal->second].bytes != blob.bytes) {
                    error = "compute GDS resources disagree with DMA pre-submit state";
                    return false;
                }
            }
            if (internal != internal_instance_by_binding.end()) {
                auto& instance = out.resource_instances[internal->second].bytes;
                if (captured.dst >= instance.size() ||
                    captured.bytes > instance.size() - captured.dst) {
                    error = "ordered DMA GDS destination exceeds captured internal GDS";
                    return false;
                }
                copy.destination_data = instance.data() + captured.dst;
                copy.destination_size = instance.size() - captured.dst;
            }
        } else if (!bind_range(
                       captured.destination_blob_index, captured.destination_blob_offset,
                       captured.dst, captured.bytes, copy.destination_data,
                       copy.destination_size,
                       "ordered DMA destination references an invalid capture blob",
                       "ordered DMA destination exceeds capture blob",
                       "ordered DMA destination blob offset exceeds its logical address")) {
            return false;
        }
        if (!bind_range(captured.source_blob_index, captured.source_blob_offset,
                        captured.src, captured.bytes, source, copy.source_size,
                        "ordered DMA source references an invalid capture blob",
                        "ordered DMA source exceeds capture blob",
                        "ordered DMA source blob offset exceeds its logical address"))
            return false;
        copy.source_data = source;
        out.dma_copies.push_back(copy);
    }
    out.operations = c.operations;
    return true;
}

void set_gpu_capture_rtt_seed_reader(CaptureRttSeedReader reader) { g_rtt_seed_reader = std::move(reader); }
bool read_gpu_capture_rtt_seed(uint64_t guest_addr, GpuCaptureRttSeed& seed, std::string& error) {
    error.clear();
    if (!g_rtt_seed_reader) { error = "live renderer has no RTT seed reader"; return false; }
    if (!g_rtt_seed_reader(guest_addr, seed)) { error = "render target is absent from live cache"; return false; }
    return validate_rtt_seed(seed, error);
}
bool capture_gpu_rtt_seed(GpuCaptureFile& capture, uint64_t guest_addr, std::string& error) {
    error.clear();
    if (!guest_addr) return true;
    if (std::any_of(capture.rtt_seeds.begin(), capture.rtt_seeds.end(),
                    [&](const GpuCaptureRttSeed& seed) {
                        return seed.guest_addr == guest_addr;
                    }))
        return true;
    GpuCaptureRttSeed seed;
    if (!read_gpu_capture_rtt_seed(guest_addr, seed, error)) return false;
    capture.rtt_seeds.push_back(std::move(seed));
    return true;
}
void set_gpu_capture_rtt_seed_snapshot_reader(CaptureRttSeedSnapshotReader reader) {
    g_rtt_seed_snapshot_reader = std::move(reader);
}
bool read_all_gpu_capture_rtt_seeds(std::vector<GpuCaptureRttSeed>& seeds, std::string& error) {
    error.clear(); seeds.clear();
    if (!g_rtt_seed_snapshot_reader) {
        error = "live renderer has no RTT seed snapshot reader"; return false;
    }
    if (!g_rtt_seed_snapshot_reader(seeds, error)) return false;
    if (seeds.size() > kMaxResources) { error = "invalid RTT seed count"; return false; }
    uint64_t total = 0;
    std::unordered_set<uint64_t> addresses;
    for (const auto& seed : seeds) {
        if (!validate_rtt_seed(seed, error)) return false;
        if (!addresses.insert(seed.guest_addr).second) {
            error = "duplicate RTT seed address"; return false;
        }
        if (seed.rgba.size() > kMaxTotalRttSeedBytes - total) {
            error = "RTT seed bytes exceed limit"; return false;
        }
        total += seed.rgba.size();
    }
    std::sort(seeds.begin(), seeds.end(), [](const auto& a, const auto& b) {
        return a.guest_addr < b.guest_addr;
    });
    return true;
}
void set_gpu_replay_rtt_seed_writer(ReplayRttSeedWriter writer) { g_rtt_seed_writer = std::move(writer); }

bool restore_gpu_replay_rtt_seeds(const std::vector<GpuCaptureRttSeed>& seeds, std::string& error) {
    error.clear();
    if (!seeds.empty() && !g_rtt_seed_writer) { error = "live renderer has no RTT seed writer"; return false; }
    for (const auto& seed : seeds) {
        if (!validate_rtt_seed(seed, error) || !g_rtt_seed_writer(seed, error)) return false;
    }
    return true;
}

void set_gpu_capture_ds_seed_snapshot_reader(CaptureDsSeedSnapshotReader reader) {
    g_ds_seed_snapshot_reader = std::move(reader);
}

bool read_all_gpu_capture_ds_seeds(std::vector<GpuCaptureDsSeed>& seeds, std::string& error) {
    error.clear(); seeds.clear();
    if (!g_ds_seed_snapshot_reader) {
        error = "live renderer has no DS seed snapshot reader"; return false;
    }
    if (!g_ds_seed_snapshot_reader(seeds, error)) return false;
    if (seeds.size() > kMaxResources) { error = "invalid DS seed count"; return false; }
    uint64_t total = 0;
    std::set<decltype(ds_seed_key(GpuCaptureDsSeed{}))> keys;
    for (const auto& seed : seeds) {
        if (!validate_ds_seed(seed, error)) return false;
        if (!keys.insert(ds_seed_key(seed)).second) {
            error = "duplicate DS seed identity"; return false;
        }
        const uint64_t plane_bytes = seed.depth.size() + seed.stencil.size();
        if (plane_bytes > kMaxTotalDsSeedBytes || total > kMaxTotalDsSeedBytes - plane_bytes) {
            error = "DS seed bytes exceed limit"; return false;
        }
        total += plane_bytes;
    }
    std::sort(seeds.begin(), seeds.end(), [](const auto& a, const auto& b) {
        return ds_seed_key(a) < ds_seed_key(b);
    });
    return true;
}

bool gpu_capture_ds_seed_snapshot_available() {
    return static_cast<bool>(g_ds_seed_snapshot_reader);
}

bool capture_referenced_gpu_ds_seeds(GpuCaptureFile& capture, std::string& error) {
    error.clear();
    if (!capture.ds_seeds.empty()) {
        error = "capture already contains DS checkpoints";
        return false;
    }
    const bool metadata_only = std::any_of(
        capture.metadata.renderer_env.begin(), capture.metadata.renderer_env.end(),
        [](const auto& entry) {
            return entry.first == "PROSPER_GPU_CAPTURE_METADATA_ONLY" &&
                   !entry.second.empty() && entry.second != "0" && entry.second != "off";
        });
    if (metadata_only) return true;

    // A depth/stencil checkpoint can only be consumed by a captured graphics draw that enables a
    // DS operation. Avoid draining the renderer's live DS cache for compute-only captures (and
    // graphics submits with no DS use); besides being unnecessary, that transfer must not prevent
    // an otherwise self-contained compute capsule from being written.
    const bool references_ds = std::any_of(
        capture.draws.begin(), capture.draws.end(), [](const GpuCapturedDraw& draw) {
            const auto& ps = draw.ps;
            return ps.depth_test_enable || ps.depth_write_enable || ps.depth_clear_enable ||
                   ps.stencil_enable || ps.stencil_clear_enable;
        });
    if (!references_ds) return true;

    std::vector<GpuCaptureDsSeed> live;
    if (!read_all_gpu_capture_ds_seeds(live, error)) return false;
    for (auto& seed : live) {
        const bool referenced = std::any_of(
            capture.draws.begin(), capture.draws.end(), [&](const GpuCapturedDraw& draw) {
                const auto& ps = draw.ps;
                const bool uses_ds = ps.depth_test_enable || ps.depth_write_enable ||
                                     ps.depth_clear_enable || ps.stencil_enable ||
                                     ps.stencil_clear_enable;
                return uses_ds && draw.color0_width == seed.width &&
                       draw.color0_height == seed.height &&
                       ps.depth_read_base == seed.depth_read_base &&
                       ps.depth_write_base == seed.depth_write_base &&
                       ps.stencil_read_base == seed.stencil_read_base &&
                       ps.stencil_write_base == seed.stencil_write_base &&
                       ps.htile_data_base == seed.htile_data_base;
            });
        if (referenced) capture.ds_seeds.push_back(std::move(seed));
    }
    return true;
}

void set_gpu_replay_ds_seed_writer(ReplayDsSeedWriter writer) {
    g_ds_seed_writer = std::move(writer);
}

bool restore_gpu_replay_ds_seeds(const std::vector<GpuCaptureDsSeed>& seeds, std::string& error) {
    error.clear();
    if (!seeds.empty() && !g_ds_seed_writer) {
        error = "live renderer has no DS seed writer"; return false;
    }
    std::set<decltype(ds_seed_key(GpuCaptureDsSeed{}))> keys;
    for (const auto& seed : seeds) {
        if (!validate_ds_seed(seed, error)) return false;
        if (!keys.insert(ds_seed_key(seed)).second) {
            error = "duplicate DS seed identity"; return false;
        }
        if (!g_ds_seed_writer(seed, error)) return false;
    }
    return true;
}

namespace {
// Interactive one-shot capture request (armed by the app hotkey, consumed on the render thread). A
// non-empty path means a grab is armed; the guard makes arm/consume race-free across threads.
std::mutex g_interactive_capture_mx;
std::string g_interactive_capture_path;
std::atomic<bool> g_output_capture_claimed{false};
std::string take_interactive_gpu_capture() {
    std::lock_guard<std::mutex> lk(g_interactive_capture_mx);
    return std::exchange(g_interactive_capture_path, std::string());
}
}  // namespace

void request_interactive_gpu_capture(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_interactive_capture_mx);
    g_interactive_capture_path = path;
}
bool interactive_gpu_capture_armed() {
    std::lock_guard<std::mutex> lk(g_interactive_capture_mx);
    return !g_interactive_capture_path.empty();
}

bool gpu_capture_output_nonzero_matches(const std::vector<uint8_t>& output, size_t min_nonzero,
                                        size_t max_nonzero, size_t* observed) {
    const size_t nonzero = static_cast<size_t>(std::count_if(
        output.begin(), output.end(), [](uint8_t value) { return value != 0; }));
    if (observed) *observed = nonzero;
    return !output.empty() && nonzero >= min_nonzero &&
           (!max_nonzero || nonzero <= max_nonzero);
}

bool parse_gpu_capture_resource_selector(std::string_view text,
                                         GpuCaptureResourceSelector& selector) {
    const size_t first_colon = text.find(':');
    const size_t second_colon = first_colon == std::string_view::npos
        ? std::string_view::npos : text.find(':', first_colon + 1);
    if (first_colon == std::string_view::npos || second_colon == std::string_view::npos ||
        text.find(':', second_colon + 1) != std::string_view::npos)
        return false;
    uint64_t draw_index = 0, binding = 0;
    if (!parse_diagnostic_draw_id(text.substr(0, first_colon), draw_index) ||
        !parse_diagnostic_uint64(text.substr(second_colon + 1), binding) ||
        binding > std::numeric_limits<uint32_t>::max())
        return false;
    const std::string_view stage = text.substr(first_colon + 1,
                                               second_colon - first_colon - 1);
    if (stage != "vs" && stage != "ps") return false;
    selector.draw_index = draw_index;
    selector.stage = stage == "vs" ? ShaderProgramStage::Vertex
                                    : ShaderProgramStage::Fragment;
    selector.binding = static_cast<uint32_t>(binding);
    return true;
}

const char* gpu_capture_resource_input_unavailable_reason_name(
        GpuCaptureResourceInputUnavailableReason reason) {
    switch (reason) {
        case GpuCaptureResourceInputUnavailableReason::None: return "none";
        case GpuCaptureResourceInputUnavailableReason::CapturePredatesWitness:
            return "capture-predates-v46";
        case GpuCaptureResourceInputUnavailableReason::NoSemanticState:
            return "no-semantic-draw-state";
        case GpuCaptureResourceInputUnavailableReason::NonDirectResource:
            return "not-a-direct-vsharp";
        case GpuCaptureResourceInputUnavailableReason::InvalidSgprRange:
            return "invalid-direct-sgpr-range";
        case GpuCaptureResourceInputUnavailableReason::NoRawOrigin:
            return "no-raw-sh-origin";
        case GpuCaptureResourceInputUnavailableReason::AmbiguousRawOrigin:
            return "ambiguous-raw-sh-origin";
        case GpuCaptureResourceInputUnavailableReason::MissingWriteProvenance:
            return "missing-sh-write-provenance";
        case GpuCaptureResourceInputUnavailableReason::RawFormatUnavailable:
            return "raw-format-not-architecturally-available";
        case GpuCaptureResourceInputUnavailableReason::RawSizeUnavailable:
            return "raw-size-not-architecturally-available";
        case GpuCaptureResourceInputUnavailableReason::RawFormatAndSizeUnavailable:
            return "raw-format-and-size-not-architecturally-available";
    }
    return "invalid-unavailable-reason";
}

GpuCaptureResourceInputVerdict gpu_capture_resource_input_verdict(
        const GpuCaptureResourceProvenance& provenance,
        const ShaderResource& normalized_resource) {
    if (provenance.input_mode == GpuCaptureResourceInputMode::Unavailable)
        return GpuCaptureResourceInputVerdict::Unavailable;
    if (provenance.input_mode != GpuCaptureResourceInputMode::DirectVSharp ||
        provenance.input_write_provenance_mask != 0x0fu ||
        provenance.input_sh_register_base == 0xFFFFFFFFu)
        return GpuCaptureResourceInputVerdict::Mismatch;

    // Decode only architecturally encoded V# fields. Shader-instruction formats and compatibility
    // allocations are front-half policy, not evidence in these four raw dwords.  Compare every
    // counterpart the raw descriptor does encode before reporting a missing field as unavailable:
    // NUM_RECORDS=0 still proves format/components, and FORMAT=INVALID still proves byte size.
    const DecodedBufferDescriptor decoded =
        decode_buffer_descriptor(provenance.input_dwords.data());
    if (normalized_resource.srt_offset != 0xFFFFFFFFu ||
        normalized_resource.sgpr_base != provenance.sgpr_base ||
        normalized_resource.gpu_addr != provenance.guest_addr ||
        normalized_resource.gpu_addr != decoded.base ||
        normalized_resource.stride != decoded.stride)
        return GpuCaptureResourceInputVerdict::Mismatch;

    const auto coverage_reason = direct_vsharp_coverage_reason(provenance.input_dwords);
    if (provenance.input_unavailable_reason != coverage_reason)
        return GpuCaptureResourceInputVerdict::Mismatch;

    const bool format_available =
        coverage_reason != GpuCaptureResourceInputUnavailableReason::RawFormatUnavailable &&
        coverage_reason !=
            GpuCaptureResourceInputUnavailableReason::RawFormatAndSizeUnavailable;
    const bool size_available =
        coverage_reason != GpuCaptureResourceInputUnavailableReason::RawSizeUnavailable &&
        coverage_reason !=
            GpuCaptureResourceInputUnavailableReason::RawFormatAndSizeUnavailable;
    if (format_available &&
        (normalized_resource.format != decoded.format ||
         normalized_resource.num_components != decoded.num_components))
        return GpuCaptureResourceInputVerdict::Mismatch;
    if (size_available && normalized_resource.size != decoded.size_bytes)
        return GpuCaptureResourceInputVerdict::Mismatch;
    if (coverage_reason != GpuCaptureResourceInputUnavailableReason::None)
        return GpuCaptureResourceInputVerdict::Unavailable;
    return GpuCaptureResourceInputVerdict::FullMatch;
}

namespace {

void capture_resource_input_witness(GpuCaptureResourceProvenance& provenance,
                                    const ShaderResource& selected,
                                    const DrawItem& draw,
                                    const GpuState* semantic_state) {
    provenance.input_mode = GpuCaptureResourceInputMode::Unavailable;
    provenance.input_unavailable_reason =
        GpuCaptureResourceInputUnavailableReason::NoSemanticState;
    provenance.input_write_provenance_mask = 0;
    provenance.input_sh_register_base = 0xFFFFFFFFu;
    provenance.input_dwords = {};
    provenance.input_last_writes = {};
    provenance.input_write_sources = {};

    if (selected.srt_offset != 0xFFFFFFFFu || selected.sgpr_base == 0xFFFFFFFFu) {
        provenance.input_unavailable_reason =
            GpuCaptureResourceInputUnavailableReason::NonDirectResource;
        return;
    }
    if (selected.direct_vsharp_sh_register_base ==
        ShaderResource::kDirectVSharpOriginAmbiguous) {
        provenance.input_unavailable_reason =
            GpuCaptureResourceInputUnavailableReason::AmbiguousRawOrigin;
        return;
    }
    if (selected.direct_vsharp_sh_register_base ==
        ShaderResource::kDirectVSharpOriginUnavailable) {
        provenance.input_unavailable_reason =
            GpuCaptureResourceInputUnavailableReason::NoRawOrigin;
        return;
    }
    const uint32_t shader_user_base =
        provenance.stage == ShaderProgramStage::Fragment ? 0u : 8u;
    if (selected.sgpr_base < shader_user_base ||
        selected.sgpr_base - shader_user_base > 28u) {
        provenance.input_unavailable_reason =
            GpuCaptureResourceInputUnavailableReason::InvalidSgprRange;
        return;
    }
    if (!semantic_state || draw.draw_index >= semantic_state->draws.size()) return;
    const GpuState& draw_state = semantic_state->state_at_draw(
        static_cast<size_t>(draw.draw_index));
    provenance.input_sh_register_base = selected.direct_vsharp_sh_register_base;
    if (provenance.input_sh_register_base > GpuState::kRegOffsetLimit - 4u) {
        provenance.input_sh_register_base = 0xFFFFFFFFu;
        provenance.input_unavailable_reason =
            GpuCaptureResourceInputUnavailableReason::NoRawOrigin;
        return;
    }
    for (uint32_t i = 0; i < 4; ++i) {
        const auto value = draw_state.sh.find(provenance.input_sh_register_base + i);
        if (value == draw_state.sh.end()) {
            provenance.input_sh_register_base = 0xFFFFFFFFu;
            provenance.input_dwords = {};
            provenance.input_unavailable_reason =
                GpuCaptureResourceInputUnavailableReason::NoRawOrigin;
            return;
        }
        provenance.input_dwords[i] = value->second;
    }
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t reg = provenance.input_sh_register_base + i;
        const auto write = draw_state.sh_prov.find(reg);
        const auto source = draw_state.sh_prov_src.find(reg);
        if (write == draw_state.sh_prov.end() || source == draw_state.sh_prov_src.end()) {
            provenance.input_write_provenance_mask = 0;
            provenance.input_sh_register_base = 0xFFFFFFFFu;
            provenance.input_dwords = {};
            provenance.input_last_writes = {};
            provenance.input_write_sources = {};
            provenance.input_unavailable_reason =
                GpuCaptureResourceInputUnavailableReason::MissingWriteProvenance;
            return;
        }
        provenance.input_write_provenance_mask |= static_cast<uint8_t>(1u << i);
        provenance.input_last_writes[i] = write->second;
        provenance.input_write_sources[i] = source->second;
    }
    provenance.input_mode = GpuCaptureResourceInputMode::DirectVSharp;
    provenance.input_unavailable_reason =
        direct_vsharp_coverage_reason(provenance.input_dwords);
}

} // namespace

void snapshot_pending_gpu_capture_draw_resource(PendingGpuCapture* pending,
                                                const DrawItem& draw,
                                                const CaptureMemoryReader& reader,
                                                const GpuState* semantic_state) {
    // This check is the normal-path cost: no table walk, allocation, or guest-memory read occurs
    // unless a capture explicitly requests one semantic resource.
    if (!pending || !pending->resource_provenance_armed ||
        !pending->resource_provenance_error.empty() ||
        draw.draw_index != pending->resource_provenance_selector.draw_index)
        return;
    const auto& selector = pending->resource_provenance_selector;
    const auto& table = selector.stage == ShaderProgramStage::Vertex ? draw.vrt : draw.prt;
    if (!table) return;

    const ShaderResource* selected = nullptr;
    for (const auto& resource : table->resources) {
        if (resource.binding != selector.binding) continue;
        if (selected) {
            pending->resource_provenance_error =
                "selected draw resource binding is ambiguous";
            return;
        }
        selected = &resource;
    }
    if (!selected) return;
    if (++pending->resource_provenance_matches != 1) {
        pending->resource_provenance_error =
            "selected draw resource matched more than once";
        return;
    }

    const uint64_t bytes = gpu_capture_resource_footprint(*selected);
    if (!selected->gpu_addr || !bytes || bytes > max_blob_bytes() ||
        bytes > std::numeric_limits<size_t>::max()) {
        pending->resource_provenance_error =
            "selected draw resource has no bounded guest byte span";
        return;
    }
    GpuCaptureBlob blob;
    blob.guest_addr = selected->gpu_addr;
    blob.bytes.resize(static_cast<size_t>(bytes), 0);
    const CaptureMemoryReader exact_reader = reader ? reader : CaptureMemoryReader(
        [](uint64_t addr, uint8_t* destination, size_t size) {
            return read_capture_guest_memory(addr, destination, size);
        });
    blob.bytes_read = std::min<uint64_t>(
        exact_reader(blob.guest_addr, blob.bytes.data(), blob.bytes.size()), blob.bytes.size());
    blob.content_hash = gpu_capture_hash(blob.bytes);

    auto& provenance = pending->resource_provenance;
    provenance.draw_index = draw.draw_index;
    provenance.stage = selector.stage;
    provenance.binding = selector.binding;
    provenance.resource_class = selected->cls;
    provenance.guest_addr = selected->gpu_addr;
    provenance.requested_bytes = bytes;
    provenance.srt_offset = selected->srt_offset;
    provenance.sgpr_base = selected->sgpr_base;
    provenance.realization_content_hash = blob.content_hash;
    capture_resource_input_witness(provenance, *selected, draw, semantic_state);
    pending->resource_provenance_realization_blob = std::move(blob);
}

namespace {
CaptureMemoryReader pending_capture_reader(const PendingGpuCapture& pending) {
    return [&pending](uint64_t addr, uint8_t* destination, size_t bytes) -> size_t {
        size_t copied = read_capture_guest_memory(addr, destination, bytes);
        if (!bytes || addr > std::numeric_limits<uint64_t>::max() - bytes) return copied;
        const uint64_t end = addr + bytes;
        for (const auto& snapshot : pending.pre_submit_memory) {
            if (snapshot.guest_addr > std::numeric_limits<uint64_t>::max() -
                    snapshot.bytes.size())
                continue;
            const uint64_t snapshot_end = snapshot.guest_addr + snapshot.bytes.size();
            const uint64_t overlap_begin = std::max(addr, snapshot.guest_addr);
            const uint64_t overlap_end = std::min(end, snapshot_end);
            if (overlap_begin >= overlap_end) continue;
            const size_t destination_offset = static_cast<size_t>(overlap_begin - addr);
            const size_t snapshot_offset = static_cast<size_t>(overlap_begin - snapshot.guest_addr);
            const size_t overlap_bytes = static_cast<size_t>(overlap_end - overlap_begin);
            std::memcpy(destination + destination_offset,
                        snapshot.bytes.data() + snapshot_offset, overlap_bytes);
            const size_t readable_overlap = snapshot.bytes_read > snapshot_offset
                ? std::min(overlap_bytes, snapshot.bytes_read - snapshot_offset) : 0;
            copied = std::max(copied, destination_offset + readable_overlap);
        }
        return copied;
    };
}

const char* resource_provenance_stage_name(ShaderProgramStage stage) {
    return stage == ShaderProgramStage::Vertex ? "vs" : "ps";
}

bool validate_pending_resource_provenance(const PendingGpuCapture& pending,
                                          std::string& error) {
    if (!pending.resource_provenance_armed) return true;
    if (!pending.resource_provenance_error.empty()) {
        error = "resource provenance selector failed: " +
                pending.resource_provenance_error;
        return false;
    }
    if (pending.resource_provenance_matches != 1) {
        const auto& selector = pending.resource_provenance_selector;
        error = "resource provenance selector " + std::to_string(selector.draw_index) + ":" +
                resource_provenance_stage_name(selector.stage) + ":" +
                std::to_string(selector.binding) + " matched " +
                std::to_string(pending.resource_provenance_matches) + " resources";
        return false;
    }
    return true;
}

bool finalize_pending_resource_provenance(PendingGpuCapture& pending,
                                          std::string& error) {
    if (!pending.resource_provenance_armed) return true;
    if (!validate_pending_resource_provenance(pending, error)) return false;
    auto& provenance = pending.resource_provenance;
    const GpuCapturedDraw* selected_draw = nullptr;
    for (const auto& draw : pending.capture.draws) {
        if (draw.draw_index != provenance.draw_index) continue;
        if (selected_draw) {
            error = "captured provenance draw identity is ambiguous";
            return false;
        }
        selected_draw = &draw;
    }
    if (!selected_draw) {
        error = "captured provenance draw is absent after materialization";
        return false;
    }
    const auto& table = provenance.stage == ShaderProgramStage::Vertex
        ? selected_draw->vrt : selected_draw->prt;
    const GpuCapturedResource* selected_resource = nullptr;
    for (const auto& resource : table.resources) {
        if (resource.resource.binding != provenance.binding) continue;
        if (selected_resource) {
            error = "captured provenance resource identity is ambiguous";
            return false;
        }
        selected_resource = &resource;
    }
    if (!selected_resource ||
        selected_resource->resource.cls != provenance.resource_class ||
        selected_resource->resource.gpu_addr != provenance.guest_addr ||
        selected_resource->captured_size != provenance.requested_bytes ||
        selected_resource->resource.srt_offset != provenance.srt_offset ||
        selected_resource->resource.sgpr_base != provenance.sgpr_base) {
        error = "captured provenance descriptor changed after draw realization";
        return false;
    }
    if (selected_resource->blob_index >= pending.capture.blobs.size()) {
        error = "captured provenance resource has no post-submit blob";
        return false;
    }
    const auto& post_blob = pending.capture.blobs[selected_resource->blob_index];
    if (selected_resource->blob_offset > post_blob.bytes.size() ||
        provenance.requested_bytes > post_blob.bytes.size() - selected_resource->blob_offset) {
        error = "captured provenance post-submit span exceeds its blob";
        return false;
    }
    const auto& realization_blob = pending.resource_provenance_realization_blob;
    if (realization_blob.guest_addr != provenance.guest_addr ||
        realization_blob.bytes.size() != provenance.requested_bytes ||
        realization_blob.content_hash != gpu_capture_hash(realization_blob.bytes)) {
        error = "captured provenance realization sample is inconsistent";
        return false;
    }

    provenance.post_blob_index = selected_resource->blob_index;
    provenance.post_blob_offset = selected_resource->blob_offset;
    provenance.post_content_hash = gpu_capture_hash(
        post_blob.bytes.data() + selected_resource->blob_offset,
        static_cast<size_t>(provenance.requested_bytes));
    provenance.realization_blob_index =
        static_cast<uint32_t>(pending.capture.blobs.size());
    pending.capture.blobs.push_back(std::move(pending.resource_provenance_realization_blob));
    pending.capture.resource_provenance.push_back(provenance);
    return true;
}

std::vector<ComputeItem> with_pending_gds_snapshot(
    const PendingGpuCapture& pending, const std::vector<ComputeItem>& computes) {
    if (pending.pre_submit_compute_gds.empty()) return computes;
    std::vector<ComputeItem> exact = computes;
    for (auto& compute : exact) {
        if (!compute.resources) continue;
        const bool has_gds = std::any_of(
            compute.resources->resources.begin(), compute.resources->resources.end(),
            [](const ShaderResource& resource) { return is_compute_internal_gds(resource); });
        if (!has_gds) continue;
        compute.resources = std::make_shared<ShaderResourceTable>(*compute.resources);
        for (auto& resource : compute.resources->resources) {
            if (!is_compute_internal_gds(resource)) continue;
            resource.host_data = const_cast<uint8_t*>(pending.pre_submit_compute_gds.data());
            resource.host_data_size = pending.pre_submit_compute_gds.size();
        }
    }
    return exact;
}

bool materialize_pending_gpu_capture(PendingGpuCapture& pending,
                                     const std::vector<DrawItem>& draws,
                                     const std::vector<ComputeItem>& computes,
                                     const std::vector<SubmitOperation>& operations,
                                     const GpuState* semantic_state,
                                     const std::vector<OperationRealizationFailure>* exact_failures,
                                     std::string& error) {
    const GpuCaptureMetadata metadata = pending.capture.metadata;
    // Preserve exact realized lists while retaining diagnostics for semantic operations rejected
    // before they reached those lists. DMA-bearing deferred captures must not fall back to eager
    // semantic realization: their indirect arguments do not exist until the ordered copy executes.
    std::vector<OperationRealizationFailure> failures = exact_failures
        ? *exact_failures : std::vector<OperationRealizationFailure>{};
    if (semantic_state && !exact_failures) {
        std::vector<OperationRealizationFailure> compute_failures;
        (void)realize_gpustate_draws(*semantic_state, 0x10000, 1.0f, 1.0f,
                                     &failures, false, false);
        (void)realize_compute_dispatches(*semantic_state, metadata.submit_index,
                                         &compute_failures);
        failures.insert(failures.end(),
                        std::make_move_iterator(compute_failures.begin()),
                        std::make_move_iterator(compute_failures.end()));
    } else if (semantic_state && exact_failures) {
        // Argument resolution can fail before execute_ordered_gpustate has enough information to
        // append an exact OperationRealizationFailure. Add those missing dispatch records here,
        // before capture_submit_items would otherwise synthesize a permanently empty Unknown.
        for (size_t index = 0; index < semantic_state->dispatches.size(); ++index) {
            const GpuState::Dispatch& dispatch = semantic_state->dispatches[index];
            // An exact ordered trace may legitimately contain no operations after zero-workgroup
            // dispatches are removed. Only the semantic fallback treats an empty plan as "all";
            // otherwise it would fabricate Unknown failures for hardware no-ops.
            const bool captured_operation = std::any_of(
                operations.begin(), operations.end(), [&](const SubmitOperation& operation) {
                    return operation.kind == SubmitOperationKind::Dispatch &&
                           operation.index == index &&
                           operation.command_order == dispatch.command_order;
                });
            if (!captured_operation) continue;
            const bool realized = std::any_of(
                computes.begin(), computes.end(), [&](const ComputeItem& item) {
                    return item.dispatch_index == index &&
                           item.command_order == dispatch.command_order;
                });
            const bool diagnosed = std::any_of(
                failures.begin(), failures.end(), [&](const OperationRealizationFailure& failure) {
                    return failure.kind == SubmitOperationKind::Dispatch &&
                           failure.index == index &&
                           failure.command_order == dispatch.command_order;
                });
            if (realized || diagnosed) continue;
            OperationRealizationFailure failure;
            failure.kind = SubmitOperationKind::Dispatch;
            failure.index = index;
            failure.command_order = dispatch.command_order;
            failure.reason = RealizationFailureReason::Unknown;
            failure.compute_launch = resolve_compute_launch(dispatch);
            failures.push_back(std::move(failure));
        }
        // Ordered execution can reject an indirect dispatch before shader realization when an
        // earlier producer leaves its argument buffer unreadable or empty. The exact trace owns the
        // operation outcome and launch facts, but its Unknown failure then has no stage at all. Use
        // the retained pre-submit register snapshot only to recover the shader diagnostic: this does
        // not resolve the arguments, mark the operation realized, or execute GPU work. Keeping the
        // raw program/resource metadata makes a COMPUTE_ADDR-selected capsule useful for offline
        // inspection even when the missing producer is the very bug under investigation.
        for (auto& failure : failures) {
            if (failure.kind != SubmitOperationKind::Dispatch || !failure.stages.empty() ||
                failure.index >= semantic_state->dispatches.size())
                continue;
            const GpuState::Dispatch& dispatch = semantic_state->dispatches[failure.index];
            if (failure.command_order != dispatch.command_order) continue;
            // The indirect producer may leave group/thread counts unresolved, but the shader
            // register snapshot still provides the exact local size. Preserve those known fields
            // (and explicit zero unknowns) so the captured config can be validated without
            // inventing dispatch arguments.
            failure.compute_launch = resolve_compute_launch(dispatch);
            GpuState diagnostic_state = dispatch.state ? *dispatch.state : *semantic_state;
            diagnostic_state.dispatches.clear();
            diagnostic_state.dispatches.push_back(dispatch);
            std::vector<OperationRealizationFailure> semantic_failures;
            std::vector<ComputeItem> semantic_items = realize_compute_dispatches(
                diagnostic_state, metadata.submit_index, &semantic_failures);
            if (!semantic_items.empty()) {
                ShaderRealizationDiagnostic stage;
                stage.stage = ShaderProgramStage::Compute;
                stage.program_addr = semantic_items.front().code_addr;
                stage.resources = semantic_items.front().resources;
                stage.recompile_config = semantic_items.front().recompile_config;
                stage.recompile_config_available =
                    semantic_items.front().recompile_config_available;
                failure.stages.push_back(std::move(stage));
            } else if (!semantic_failures.empty() &&
                       !semantic_failures.front().stages.empty()) {
                const ShaderRealizationDiagnostic& semantic_stage =
                    semantic_failures.front().stages.front();
                ShaderRealizationDiagnostic stage;
                stage.stage = semantic_stage.stage;
                stage.program_addr = semantic_stage.program_addr;
                stage.resources = semantic_stage.resources;
                stage.recompile_config = semantic_stage.recompile_config;
                stage.recompile_config_available =
                    semantic_stage.recompile_config_available;
                failure.stages.push_back(std::move(stage));
            }
        }
    }
    const std::vector<ComputeItem> snapshot_computes =
        with_pending_gds_snapshot(pending, computes);
    const std::vector<SubmitOperation> semantic_operations =
        operations.empty() && semantic_state && !exact_failures
            ? plan_submit_operations(*semantic_state)
            : std::vector<SubmitOperation>{};
    const std::vector<SubmitOperation>& exact_operations =
        semantic_operations.empty() ? operations : semantic_operations;
    const CaptureMemoryReader reader = pending_capture_reader(pending);
    const bool captured = capture_submit_items(
        draws, snapshot_computes, exact_operations, metadata, reader, pending.capture, error,
        g_rtt_seed_reader, failures,
        semantic_state ? semantic_state->dma_copies : std::vector<GpuState::DmaCopy>{}, 0,
        pending.pre_submit_compute_gds.data(), pending.pre_submit_compute_gds.size());
    if (!captured) return false;
    pending.materialized = true;
    return !gpu_capture_ds_seed_snapshot_available() ||
           capture_referenced_gpu_ds_seeds(pending.capture, error);
}
}  // namespace

void snapshot_pending_gpu_capture_compute_gds(PendingGpuCapture* pending,
                                              const uint8_t* data, size_t bytes) {
    if (!pending || pending->materialized || !data || !bytes) return;
    pending->pre_submit_compute_gds.assign(data, data + bytes);
}

std::unique_ptr<PendingGpuCapture> begin_requested_gpu_capture(
    const std::vector<DrawItem>& draws, const std::vector<ComputeItem>& computes,
    const std::vector<SubmitOperation>& operations, uint32_t width, uint32_t height,
    const GpuState* semantic_state, uint64_t submit_no, uint64_t semantic_draw_count,
    const std::vector<OperationRealizationFailure>* exact_failures,
    bool defer_materialization) {
    const uint64_t candidate_draw_count = semantic_state && semantic_draw_count != UINT64_MAX
        ? semantic_draw_count : static_cast<uint64_t>(draws.size());
    // Interactive one-shot (hotkey): consume the armed request on the first DRAWING invocation so the
    // grab lands on real frame content, and skip the env AT/AFTER/MIN selectors (a live keypress has no
    // submit index to predict). Falls through to the env PROSPER_GPU_CAPTURE path when nothing is armed.
    std::string interactive_path;
    if (candidate_draw_count > 0 && interactive_gpu_capture_armed())
        interactive_path = take_interactive_gpu_capture();
    const bool interactive = !interactive_path.empty();
    const char* env_path = std::getenv("PROSPER_GPU_CAPTURE");
    if (!interactive && (!env_path || !*env_path)) return {};
    const char* output_min_env = std::getenv("PROSPER_GPU_CAPTURE_OUTPUT_NZ_MIN");
    const bool output_triggered = !interactive && output_min_env && *output_min_env;
    if (output_triggered && g_output_capture_claimed.load(std::memory_order_acquire)) return {};
    uint64_t current = 0;
    if (!interactive) {
        static std::atomic<uint64_t> invocation_sequence{0};
        const uint64_t invocation = invocation_sequence.fetch_add(1);
        uint64_t after = 0;
        if (const char* v = std::getenv("PROSPER_GPU_CAPTURE_AFTER")) after = std::strtoull(v, nullptr, 0);
        if (invocation < after) return {};
        static const auto capture_started = std::chrono::steady_clock::now();
        if (const char* v = std::getenv("PROSPER_GPU_CAPTURE_AFTER_MS")) {
            const uint64_t after_ms = std::strtoull(v, nullptr, 0);
            const uint64_t elapsed_ms = static_cast<uint64_t>(std::chrono::duration_cast<
                std::chrono::milliseconds>(std::chrono::steady_clock::now() - capture_started).count());
            if (elapsed_ms < after_ms) return {};
        }
        if (const char* value = std::getenv("PROSPER_GPU_CAPTURE_COMPUTE_ADDR")) {
            char* end = nullptr;
            errno = 0;
            const uint64_t wanted_code_addr = std::strtoull(value, &end, 0);
            if (errno || end == value || !end || *end) return {};
            const bool realized_contains_program = std::any_of(
                computes.begin(), computes.end(), [&](const ComputeItem& compute) {
                    return compute.code_addr == wanted_code_addr;
                });
            const bool semantic_contains_program = semantic_state && std::any_of(
                semantic_state->dispatches.begin(), semantic_state->dispatches.end(),
                [&](const GpuState::Dispatch& dispatch) {
                    return compute_dispatch_code_addr(*semantic_state, dispatch) == wanted_code_addr;
                });
            if (!realized_contains_program && !semantic_contains_program) return {};
        }
        if (const char* value = std::getenv("PROSPER_GPU_CAPTURE_SHADER_ADDR")) {
            char* end = nullptr;
            errno = 0;
            const uint64_t wanted_code_addr = std::strtoull(value, &end, 0);
            if (errno || end == value || !end || *end) return {};
            const bool realized_contains_program = std::any_of(
                draws.begin(), draws.end(), [&](const DrawItem& draw) {
                    return draw.vs_guest_addr == wanted_code_addr ||
                           draw.vs_chain_guest_addr == wanted_code_addr ||
                           draw.fs_guest_addr == wanted_code_addr;
                });
            const bool semantic_contains_program = semantic_state && std::any_of(
                semantic_state->draws.begin(), semantic_state->draws.end(),
                [&](const GpuState::Draw& draw) {
                    const GpuState& draw_state = draw.state ? *draw.state : *semantic_state;
                    const RenderState state = extract_render_state(draw_state);
                    return state.es_addr == wanted_code_addr || state.gs_addr == wanted_code_addr ||
                           state.hs_addr == wanted_code_addr || state.ps_addr == wanted_code_addr;
                });
            if (!realized_contains_program && !semantic_contains_program) return {};
        }
        if (const char* value = std::getenv("PROSPER_GPU_CAPTURE_TARGET_DIM")) {
            uint32_t wanted_width = 0, wanted_height = 0;
            char trailing = 0;
            if (std::sscanf(value, "%ux%u%c", &wanted_width, &wanted_height, &trailing) != 2 ||
                !wanted_width || !wanted_height)
                return {};
            const bool realized_contains_target = std::any_of(
                draws.begin(), draws.end(), [&](const DrawItem& draw) {
                    return draw.color0_width == wanted_width && draw.color0_height == wanted_height;
                });
            const bool semantic_contains_target = semantic_state && std::any_of(
                semantic_state->draws.begin(), semantic_state->draws.end(),
                [&](const GpuState::Draw& draw) {
                    const GpuState& draw_state = draw.state ? *draw.state : *semantic_state;
                    const RenderState state = extract_render_state(draw_state);
                    return state.color0_width == wanted_width &&
                           state.color0_height == wanted_height;
                });
            if (!realized_contains_target && !semantic_contains_target) return {};
        }
        uint64_t min_draws = 0, max_draws = std::numeric_limits<uint64_t>::max();
        if (const char* v = std::getenv("PROSPER_GPU_CAPTURE_MIN_DRAWS")) min_draws = std::strtoull(v, nullptr, 0);
        if (const char* v = std::getenv("PROSPER_GPU_CAPTURE_MAX_DRAWS")) max_draws = std::strtoull(v, nullptr, 0);
        if (candidate_draw_count < min_draws || candidate_draw_count > max_draws) return {};
        static std::atomic<uint64_t> sequence{0}; static std::atomic<bool> claimed{false};
        current = sequence.fetch_add(1);
        // Enumerate every submit that PASSED the selectors, so `PROSPER_GPU_CAPTURE_AT` can be aimed
        // at a specific phase instead of guessed. A title reruns one compute program across wildly
        // different populations — GTA V's `0x413dc6700` appears in an early 2,735-dispatch submit and
        // again in a later ~234-dispatch one — and `MIN_DRAWS`/`MAX_DRAWS` cannot separate them,
        // because `candidate_draw_count` is the DRAW count and a compute-only submit has none. Before
        // this line the only way to reach the later phase was to re-run with successive AT values and
        // inspect each capsule, at one routed run apiece (#2481).
        if (std::getenv("PROSPER_GPU_CAPTURE_LOG")) {
            // The widest viewport any draw in this submit uses. A draw COUNT cannot tell a main-view
            // pass from a shadow pass — shadow atlases have high draw counts and share the main
            // target's dimensions, differing only in viewport — so a census without this cannot be
            // aimed at "the submit that draws the world" (#2481).
            float widest_w = 0.0f, widest_h = 0.0f;
            for (const DrawItem& draw : draws) {
                if (!draw.ps.has_viewport) continue;
                const float w = draw.ps.viewport_w < 0.0f ? -draw.ps.viewport_w : draw.ps.viewport_w;
                const float h = draw.ps.viewport_h < 0.0f ? -draw.ps.viewport_h : draw.ps.viewport_h;
                if (w * h > widest_w * widest_h) { widest_w = w; widest_h = h; }
            }
            std::fprintf(stderr,
                         "[gpucap] candidate at=%llu submit=%llu draws=%zu computes=%zu "
                         "semantic-dispatches=%zu viewport=%.0fx%.0f%s\n",
                         static_cast<unsigned long long>(current),
                         static_cast<unsigned long long>(submit_no), draws.size(), computes.size(),
                         semantic_state ? semantic_state->dispatches.size() : size_t{0},
                         widest_w, widest_h,
                         claimed.load() ? " (already claimed)" : "");
        }
        if (!output_triggered) {
            uint64_t wanted = 0;
            if (const char* at = std::getenv("PROSPER_GPU_CAPTURE_AT")) wanted = std::strtoull(at, nullptr, 0);
            if (current != wanted || claimed.exchange(true)) return {};
        }
    }
    auto pending = std::make_unique<PendingGpuCapture>();
    pending->materialized = false;
    pending->path = interactive ? interactive_path : std::string(env_path);
    if (const char* selector = std::getenv(kGpuCaptureResourceProvenanceEnv);
        selector && *selector) {
        if (!parse_gpu_capture_resource_selector(
                selector, pending->resource_provenance_selector)) {
            std::fprintf(stderr,
                         "[gpucap] resource provenance NOT ARMED: invalid selector %s\n",
                         selector);
            return {};
        }
        pending->resource_provenance_armed = true;
        // A submit may prepare graphics eagerly yet still execute through the ordered path because
        // it contains compute. In that shape the eager DrawItem is only a realization cache: sampling
        // it here both double-matches the later ordered hook and observes bytes before an earlier
        // producer. Deferred capture therefore has exactly one temporal owner, the ordered executor.
        if (!defer_materialization)
            for (const auto& draw : draws)
                snapshot_pending_gpu_capture_draw_resource(
                    pending.get(), draw, {}, semantic_state);
    }
    pending->output_triggered = output_triggered;
    if (output_triggered) {
        pending->output_min_nonzero = static_cast<size_t>(std::strtoull(output_min_env, nullptr, 0));
        if (const char* max = std::getenv("PROSPER_GPU_CAPTURE_OUTPUT_NZ_MAX"))
            pending->output_max_nonzero = static_cast<size_t>(std::strtoull(max, nullptr, 0));
    }
    GpuCaptureMetadata m; m.width = width; m.height = height;
    m.submit_index = submit_no ? submit_no : current;
    m.revision = embedded_build_revision();
    if (const char* revision = std::getenv("PROSPER_CAPTURE_REVISION")) m.revision = revision;
    m.title_id = env_or_empty("PROSPER_CAPTURE_TITLE"); m.input_route = env_or_empty("PROSPER_PAD_SCRIPT");
    m.savedata_dir = env_or_empty("PROSPER_SAVEDATA_DIR");
    static const char* render_env[] = {
        "PROSPER_ALPHA1", "PROSPER_CLEAR_DEBUG", "PROSPER_DEPTH_ALWAYS", "PROSPER_DETILE",
        "PROSPER_DRAW_ISO", "PROSPER_FLIP_FRONT_FACE", "PROSPER_FS_SPV", "PROSPER_FS_SPV_MATCH",
        "PROSPER_ISO_AT", "PROSPER_ISO_AT2",
        "PROSPER_ISO_RGB", "PROSPER_ISO_TOL", "PROSPER_NO_BLEND", "PROSPER_NO_CULL",
        "PROSPER_NO_DEPTH", "PROSPER_NO_STENCIL", "PROSPER_STENCIL_CLEAR", "PROSPER_STENCIL_REPLACE",
        "PROSPER_NO_SWIZZLE", "PROSPER_NODETILE",
        "PROSPER_DESCRIPTOR_VALIDATE",
        "PROSPER_PITCH", "PROSPER_RENDER_NOPS", "PROSPER_RENDER_REFVS", "PROSPER_RENDER_TESTPS",
        "PROSPER_RENDER_TESTPS_MATCH",
        "PROSPER_RTT", "PROSPER_RTT_NOSEED", "PROSPER_RTT_PERTARGET", "PROSPER_RTT_SINGLE_TARGET",
        "PROSPER_DUMP_RTGROUPS", "PROSPER_DUMP_RTGROUPS_RGBA", "PROSPER_DUMP_RTGROUPS_ADDR",
        "PROSPER_DUMP_DRAWSTEPS",
        "PROSPER_TESTTEX",
        "PROSPER_TESTLUT", "PROSPER_TESTLUT32"
    };
    for (const char* name : render_env) if (const char* value = std::getenv(name)) m.renderer_env.emplace_back(name, value);
    annotate_gpu_capture_save_roots(m);
    annotate_gpu_capture_scanout(m);
    pending->capture.metadata = m;
    if ((output_triggered || defer_materialization) && semantic_state &&
        !semantic_state->dma_copies.empty()) {
        const CaptureMemoryReader reader = ordered_gpustate_capture_reader(*semantic_state);
        auto snapshot = [&](uint64_t addr, uint32_t bytes) {
            const auto duplicate = std::find_if(
                pending->pre_submit_memory.begin(), pending->pre_submit_memory.end(),
                [&](const PendingGpuCapture::MemorySnapshot& existing) {
                    return existing.guest_addr == addr && existing.bytes.size() == bytes;
                });
            if (duplicate != pending->pre_submit_memory.end()) return;
            PendingGpuCapture::MemorySnapshot range;
            range.guest_addr = addr;
            range.bytes.resize(bytes);
            range.bytes_read = reader(addr, range.bytes.data(), range.bytes.size());
            pending->pre_submit_memory.push_back(std::move(range));
        };
        for (const auto& copy : semantic_state->dma_copies) {
            snapshot(copy.dst, copy.bytes);
            snapshot(copy.src, copy.bytes);
        }
    }
    // Full output-candidate materialization remains deferred until its pixels match. The small
    // pre-submit snapshots above are the deliberate exception: known DMA endpoints cannot be
    // recovered after execution, while every unrelated guest resource and renderer cache remains
    // untouched for rejected candidates.
    if (output_triggered || defer_materialization || pending->resource_provenance_armed)
        return pending;
    std::string error;
    if (!materialize_pending_gpu_capture(*pending, draws, computes, operations,
                                         semantic_state, exact_failures, error)) {
        std::fprintf(stderr, "[gpucap] capture failed: %s\n", error.c_str()); return {};
    }
    // A normal one-shot capture has the same temporal DS dependency as a timeline endpoint.  The
    // renderer callback runs before this submit is executed, so these are the exact pre-submit
    // planes consumed by depth/stencil reads in the capsule.  Timeline capture has always added
    // them explicitly; omitting them here made F9/environment capsules replay a different frame
    // whenever the selected submit reused persistent depth or stencil from an earlier submit.
    std::fprintf(stderr, "[gpucap] captured %s submit %llu: %zu draws, %zu computes, "
                         "%zu DMA copies, %zu operations, %zu blobs, %zu RTT seeds, "
                         "%zu DS seeds -> %s\n",
                 interactive ? "interactive" : "match",
                 static_cast<unsigned long long>(m.submit_index),
                 pending->capture.draws.size(), pending->capture.computes.size(),
                 pending->capture.dma_copies.size(), pending->capture.operations.size(),
                 pending->capture.blobs.size(),
                 pending->capture.rtt_seeds.size(), pending->capture.ds_seeds.size(),
                 pending->path.c_str());
    return pending;
}

bool finish_requested_gpu_capture(std::unique_ptr<PendingGpuCapture> pending,
                                  const std::vector<uint8_t>& output, std::string& error,
                                  const std::vector<DrawItem>* draws,
                                  const std::vector<ComputeItem>* computes,
                                  const std::vector<SubmitOperation>* operations,
                                  const GpuState* semantic_state,
                                  const std::vector<OperationRealizationFailure>* exact_failures) {
    if (!pending) return true;
    if (pending->output_triggered) {
        size_t nonzero = 0;
        if (!gpu_capture_output_nonzero_matches(output, pending->output_min_nonzero,
                                                pending->output_max_nonzero, &nonzero)) {
            std::fprintf(stderr,
                         "[gpucap] output candidate submit %llu rejected: nonzero=%zu range=%zu..%zu\n",
                         static_cast<unsigned long long>(pending->capture.metadata.submit_index), nonzero,
                         pending->output_min_nonzero, pending->output_max_nonzero);
            return true;
        }
        if (g_output_capture_claimed.load(std::memory_order_acquire)) return true;
        std::fprintf(stderr,
                     "[gpucap] output trigger matched submit %llu: nonzero=%zu range=%zu..%zu\n",
                     static_cast<unsigned long long>(pending->capture.metadata.submit_index), nonzero,
                     pending->output_min_nonzero, pending->output_max_nonzero);
        if (!draws || !operations) {
            error = "output-triggered capture is missing its synchronous submit state";
            return false;
        }
        if (!validate_pending_resource_provenance(*pending, error)) return false;
        static const std::vector<ComputeItem> no_computes;
        if (!materialize_pending_gpu_capture(*pending, *draws,
                                             computes ? *computes : no_computes,
                                             *operations, semantic_state, exact_failures, error))
            return false;
        if (!finalize_pending_resource_provenance(*pending, error)) return false;
        if (g_output_capture_claimed.exchange(true, std::memory_order_acq_rel)) return true;
        std::fprintf(stderr,
                     "[gpucap] captured output match submit %llu: %zu draws, %zu computes, "
                     "%zu operations, %zu blobs, %zu RTT seeds, %zu DS seeds -> %s\n",
                     static_cast<unsigned long long>(pending->capture.metadata.submit_index),
                     pending->capture.draws.size(), pending->capture.computes.size(),
                     pending->capture.operations.size(), pending->capture.blobs.size(),
                     pending->capture.rtt_seeds.size(), pending->capture.ds_seeds.size(),
                     pending->path.c_str());
    }
    if (!pending->materialized) {
        if (!draws || !operations) {
            error = "deferred capture is missing its exact submit realization";
            return false;
        }
        if (!validate_pending_resource_provenance(*pending, error)) return false;
        static const std::vector<ComputeItem> no_computes;
        if (!materialize_pending_gpu_capture(*pending, *draws,
                                             computes ? *computes : no_computes,
                                             *operations, semantic_state, exact_failures, error))
            return false;
        if (!finalize_pending_resource_provenance(*pending, error)) return false;
        std::fprintf(stderr,
                     "[gpucap] captured deferred submit %llu: %zu draws, %zu computes, "
                     "%zu operations, %zu failures -> %s\n",
                     static_cast<unsigned long long>(pending->capture.metadata.submit_index),
                     pending->capture.draws.size(), pending->capture.computes.size(),
                     pending->capture.operations.size(),
                     pending->capture.failure_diagnostics.size(), pending->path.c_str());
    }
    pending->capture.expected_output_valid = !output.empty();
    pending->capture.expected_output_bytes = output.size();
    pending->capture.expected_output_hash = output.empty() ? 0 : gpu_capture_hash(output);
    if (!write_gpu_capture(pending->path, pending->capture, error)) return false;
    if (output.empty()) {
        std::fprintf(stderr, "[gpucap] wrote %s without output oracle\n", pending->path.c_str());
    } else {
        std::fprintf(stderr, "[gpucap] wrote %s output_bytes=%zu hash=%016llx\n",
                     pending->path.c_str(), output.size(),
                     static_cast<unsigned long long>(pending->capture.expected_output_hash));
    }
    return true;
}

} // namespace prosper::gpu
