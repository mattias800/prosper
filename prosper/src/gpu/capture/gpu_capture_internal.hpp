#pragma once

// Lifted out of rdna2_to_spirv.cpp's anonymous namespaces so the emit functions that
// operate on them can live in their own translation units. These are INTERNAL to the
// recompiler: nothing outside src/gpu/recompiler/ should include this header.

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

namespace prosper::gpu {

// Declared here, DEFINED in the .cpp this header was lifted out of: other
// translation units link against those definitions, so they must not move.
using OperationIdentity = std::tuple<uint8_t, uint64_t, uint64_t>; OperationIdentity operation_identity(SubmitOperationKind kind, uint64_t source_index, uint64_t command_order);
uint64_t gpu_capture_hash(const uint8_t* data, size_t size);

constexpr char kMagic[8] = {'P','R','G','P','C','A','P','\0'};
// v23 (#1256): each realized draw also retains its RAW draw-packet state — the DrawIndexAuto/DrawIndex
// index_count and indexed flag as decoded from the guest, BEFORE realization. This lets gpu_replay
// --inspect-only surface raw-vs-realized offline and flag a decode/realization divergence (e.g. GTA
// #1163's non-indexed vertex-count inflation) without a live boot. Older captures read fine (the fields
// default to 0/false = "unknown").
// v24 (#1280): each resource's declared_mip_levels (the T#-declared mip-chain length) is serialized in a
// deterministic version-gated tail (like the v15 depth-compare / v16 mip-tail tails), so the byte-exact
// v1-v23 record prefix is preserved and older captures materialize with the historical default of 1. It
// was populated live but never written/read, so every reloaded .prgcap resource defaulted to 1 and a
// mip-declaring title replayed single-level.
// v25 (#1240): each realized/failed draw pipeline retains CB_COLOR_CONTROL.MODE=3 resolve intent in
// another version-gated tail, keeping every v1-v24 record prefix byte-exact.
// v26: retain SPI shader export and SX render-target downconversion formats per draw.
// v27: retain each draw's raw ShaderDrawModifier and signed GE_INDX_OFFSET. The latter is a draw
// parameter (Vulkan firstVertex/vertexOffset), not pipeline state; omitting it collapsed distinct
// ranges of a shared vertex pool onto vertex zero in both live rendering and replay.
// v28: retain the resolved byte row pitch and exact planned span of every resource. Guest-backed
// GFX10 linear sampled images use 256-byte-aligned rows, which differs from width*bytes-per-texel for
// video chroma planes.
// Older captures planned tight spans, so their materializer intentionally binds the legacy span while
// still deriving the guest pitch for best-effort rendering of the rows that were retained.
// v29: retain resolved depth-bias state per realized and failed pipeline.
// v30: each const-fold-resolved buffer fetch retains its proven vertex/instance/shader VADDR source.
// This is a per-resource trailing block so every v1-v29 record prefix stays byte-exact.
// v31: retain an optional linked vertex-main raw stream and the graphics-LDS allocation per draw.
// v32: retain the per-layer full-mip-chain stride and selected-level offset for thin 2D arrays and
// cube faces. Without them replay treated layer one's allocation-level bytes as layer zero's mip.
// v33: extend the append-only DataFormat enum with 8/16-bit USCALED and SSCALED buffer formats.
// Their numeric values follow every v1-v32 value, so old captures remain byte-identical/readable.
// v34: retain all eight hardware color-buffer slots and their fixed-function state.
// v35: retain the exact resource-descriptor metadata for each failed shader stage. This makes a
// table-dependent recompile retry deterministic offline instead of reducing the table to a count.
// v36: retain the resolved SPI_PS_INPUT_CNTL linkage (including metadata-only PARAM0 passthrough)
// and SPI_PS_INPUT_ENA/ADDR system-VGPR ABI for every realized draw. Raw graphics replay otherwise
// invents a different module even when the captured RDNA2 bytes are exact.
// v37: retain the required compute subgroup size. Native-subgroup SPIR-V reconstructs guest wave
// membership from SubgroupId/SubgroupLocalInvocationId and therefore must replay with the same
// required-size/full-subgroups pipeline contract used when the module was created.
// v38: extend the existing RTT-seed color-format enum with R8_UNORM and R32_UINT. No record layout
// changes: older versions never contain the new append-only enum values.
// v39: retain each realized compute program's raw RDNA2 stream plus its semantic launch/recompiler
// inputs. This lets gpu_replay --recompile-raw exercise current compute translation instead of the
// stored capture-safe SPIR-V, including native-width storage formats supported by the replay device.
// v40: retain the BVH descriptor BOX_GROW value for every resource table. The software RTIP 1.1
// lowering uses it to reproduce the guest's conservative box-interval expansion exactly.
// v41: complete the v35 failed-stage resource tables with descriptor fields that historical version
// tails only wrote for realized draw/compute tables, plus flat-load provenance omitted from the base
// record. Without this, raw failed-stage retry silently reset image/layout and codegen state.
// v42: retain the exact ComputeShaderConfig for failed compute stages. Raw code and v41 resources do
// not encode user SGPRs, dispatch dimensions, wave ABI, LDS size, or subgroup/storage policy, so an
// offline retry must never reconstruct those inputs from defaults.
// v43 (#1459): retain the raw color-state triple — CB_COLOR_CONTROL, CB_TARGET_MASK, CB_SHADER_MASK,
// each with an explicit present flag — for realized and failed draw pipelines. A capsule stores an
// already-resolved pipeline, so a zero color write mask was previously unattributable offline: an
// absent target/shader mask resolves to write-all, meaning a zero resolved mask implies one of those
// registers is PRESENT with value zero, or MODE is DISABLE. Those causes demand different fixes and
// only the raw registers separate them. Append-only tail; v1-v42 prefixes stay byte-exact and older
// captures report the state as unavailable rather than inventing write-all defaults.
// v44: retain each resource's guest sample count. 2D_MSAA image_load is materialized as host array
// layers, so replay must not collapse four guest sample planes to the historical single-layer default.
// The append-only tail covers realized draw/compute and failed-stage tables; v1-v43 default to one.
// v45 (#1849): retain one opt-in draw-time resource sample and its independently captured post-submit
// span. The append-only record references ordinary blobs so both read counts, bytes, and hashes remain
// auditable offline; captures without the selector append only a zero count and copy no extra bytes.
// v46 (#1853): retain the selected direct V#'s four raw stage USER_DATA dwords and each dword's
// PM4 last-write provenance. Replay decodes this input independently and compares it with the v45
// normalized descriptor; unsupported/ambiguous paths are recorded as explicitly unavailable.
// v47 (#2481): retain each realized/failed-stage image resource's instruction-scoped zero-mip proof.
// v48 (#2481): retain BVH BOX_SORT_EN for raw replay. Sorting changes generated SPIR-V, so replay
// must not silently compile a sorted guest descriptor with the historical physical-child order.
// v49 (#2481): retain the exact linear stride-8 qword-atomic record count. Size/stride alone do not
// preserve the descriptor's OOB_SELECT proof, which controls the all-or-nothing record range check.
// v50 (#2481): retain COMPUTE_PGM_RSRC1 in every exact compute recompile configuration. FP32 memory
// and LDS atomics consume its denormal mode; older artifacts leave the state explicitly unknown.
// v51 (#2481): internal resource bytes may also carry GTA V's exact pc26 packed-pointer shadow.
// Unlike shared GDS state, every dispatch owns a distinct snapshot even when bindings coincide.
// v52: retain the complete generic runtime-selected buffer descriptor-array contract. Each resource
// carries its declared arity/selector and every raw V# plus a separate captured backing reference;
// the legacy resource prefix stays byte-exact.
// v53 (#2481): internal compute bytes may carry the generic static-footprint indirect-pointer
// relocation snapshot. The proof marker is deliberately reconstructed from raw shader, launch,
// source records, and carrier witnesses instead of being trusted as serialized authority.
// v54: retain the explicit scalar S_BUFFER dword bound. The ordinary resource size remains the V#'s
// independent vector footprint; replay cannot reconstruct one from the other when STRIDE < 4.
// v55: append each depth/stencil seed's optional stencil bytes and stencil-presence marker.
// v56: extend the append-only RTT-seed format enum with native RGBA32F surfaces.
// v57 (#3048): each resource's allocation-wide mip placement -- the level-zero element extent, the
// bytes per block, the effective MAX_MIP, and the view's BASE_LEVEL. `declared_mip_levels` says how
// MANY levels a T# declares; these say WHERE they are, which nothing else preserves once
// image_base_level_view has shifted gpu_addr/width/height onto the SELECTED level.
//
// The BYTES half landed separately (#3202) and needs NO format version of its own, which is worth
// saying because the obvious reading is that it must. A tiled chain stores level zero last, so the
// rest of the allocation lies below the address the descriptor names; `collect_intervals` now
// extends that resource's captured range down to the allocation base, and the resource's existing
// serialized `blob_offset` is then already the count of owned bytes preceding `gpu_addr`. Replay
// publishes it as `ShaderResource::host_data_prefix_bytes` and the chain derivation accepts a
// host-backed resource whose span covers the whole allocation. Nothing new is written to the file.
//
// A capsule taken BEFORE that change reads back with a `blob_offset` that does not reach the
// allocation base, so the derivation declines and the capsule replays exactly as it always did --
// one level, IMAGE_LOAD_MIP refused. That is the intended outcome: a capture that cannot express
// the chain must keep declining visibly rather than fetching levels it does not own. Pre-v57 files
// leave all five placement fields zero, read everywhere as "not modelled", and decline earlier
// still.
constexpr uint32_t kVersion = 57;
constexpr uint32_t kEndian = 0x01020304u;
constexpr uint64_t kMaxFileBytes = 4ull << 30;
constexpr uint64_t kMaxBlobDefaultBytes = 1ull << 30;
constexpr uint64_t kMaxTotalBlobBytes = 3ull << 30;

// PROSPER_CAPTURE_BLOB_MAX_MB (#2440): the per-resource blob ceiling, 1 GiB by default.
//
// It is a runtime value rather than a constant because a real title exceeds it by a hair and loses the
// whole capture: GTA V gameplay binds a 1,105,723,396-byte buffer -- 1.0298x over -- and one resource
// over the line aborts the entire grab, which makes the F9/bundle workflow unavailable in exactly the
// phase under investigation.
//
// WHY RAISING IT IS SAFE, since this is not only a policy knob. Nine sites check it, and **three are
// on the read side** -- so a writer-only change would emit bundles a reader rejects. Each of the three
// has a DIFFERENT real bound, and this is spelled out per-site because a single blanket claim about
// them was wrong when first written (review on #2500):
//
//   * `Reader::bytes` -- bounded by `n > left`: a blob length cannot exceed the bytes actually
//     remaining in the buffer, so a corrupt or hostile length is caught without this term at all.
//   * `validate_dma_copies` -- operates on an ALREADY-LOADED file, so there is no `left` here. Each
//     copy must additionally pass `validate_blob(...)` for both endpoints, i.e. lie inside a real
//     blob at a real offset; it inherits the blob bounds transitively.
//   * `validate_resource_provenance` -- also post-load, also no `left`. `requested_bytes` must EQUAL
//     an already-read blob's size and fit inside a second blob at its recorded offset, so the
//     subsequent hash over `post.bytes.data() + post_blob_offset` is in range by that chain.
//
// In every case the aggregate is bounded independently by the neighbours above -- kMaxFileBytes
// (4 GiB) and kMaxTotalBlobBytes (3 GiB) -- and the per-blob ceiling is redundant defence layered over
// bounds that already hold. **Do not read `n > left` as the guard at the two validators**: it is not
// available there, and a future use that consumes `requested_bytes` BEFORE its equality check would
// not be covered by it.
//
// Writer and reader consult this same accessor, so a bundle produced under a raised ceiling is
// readable by the process that produced it.
//
// Clamped to kMaxTotalBlobBytes: a per-blob ceiling above the total-blob budget could never be
// satisfied, so accepting one would only move the failure later and make it harder to read. A
// malformed or out-of-range value leaves the default in place rather than guessing -- a typo must cost
// a capture, never silently widen a bound.
//
// A bundle written under a raised ceiling and read back LATER without the variable set will fail the
// reader's check. That is deliberate and loud (`invalid blob length`), not silent corruption; set the
// same value to read it.
// Pure, so the parse and clamp rules are testable without a process per case: the accessor below
// caches in a `static`, which makes both the default and a raised value unreachable from one test.
inline uint64_t blob_max_bytes_from_env(const char* s, std::string* note) {
    const auto say = [&](const std::string& m) { if (note) *note = m; };
    if (!s || !*s) return kMaxBlobDefaultBytes;
    // Reject a sign before strtoull sees it. strtoull ACCEPTS a leading '-' and WRAPS: "-4" yields
    // 18446744073709551612, which passes every check below and then clamps to the maximum -- so a
    // typo would silently widen the ceiling to the largest value the reader will accept, the exact
    // direction this parse must never fail in. Caught by the malformed-input arm in
    // test_gpu_capture, not by review.
    for (const char* c = s; *c; ++c) {
        if (*c == ' ' || *c == '\t') continue;
        if (*c == '-' || *c == '+') {
            say("must be an unsigned decimal count of MiB; keeping the default");
            return kMaxBlobDefaultBytes;
        }
        break;
    }
    char* end = nullptr;
    const unsigned long long mb = std::strtoull(s, &end, 10);
    if (end == s || (end && *end) || mb == 0) {
        say("not a positive integer; keeping the default");
        return kMaxBlobDefaultBytes;
    }
    if (mb > (kMaxTotalBlobBytes >> 20)) {
        say("exceeds the total-blob budget; clamped to it");
        return kMaxTotalBlobBytes;
    }
    return mb << 20;
}

inline uint64_t max_blob_bytes() {
    static const uint64_t v = [] {
        const char* s = getenv("PROSPER_CAPTURE_BLOB_MAX_MB");
        std::string note;
        const uint64_t v = blob_max_bytes_from_env(s, &note);
        if (!note.empty())
            std::fprintf(stderr, "[gpucapture] PROSPER_CAPTURE_BLOB_MAX_MB=%s %s (%llu MiB)\n",
                         s ? s : "", note.c_str(), (unsigned long long)(v >> 20));
        else if (v != kMaxBlobDefaultBytes)
            std::fprintf(stderr,
                         "[gpucapture] per-resource blob ceiling raised to %llu MiB "
                         "(PROSPER_CAPTURE_BLOB_MAX_MB); a bundle written now needs the same value "
                         "to be read back\n",
                         (unsigned long long)(v >> 20));
        return v;
    }();
    return v;
}
constexpr uint32_t kMaxDraws = 65536;
constexpr uint32_t kMaxComputes = 65536;
constexpr uint32_t kMaxOperations = 131072;
constexpr uint32_t kMaxResources = 65536;
constexpr uint32_t kMaxShaderWords = 16u << 20;
constexpr uint32_t kMaxRawShaderWords = 0x4000; // 64 KiB per raw stage
constexpr uint32_t kMaxFailureStages = 3;
constexpr uint32_t kMaxStringBytes = 1u << 20;
constexpr uint64_t kMaxTotalRttSeedBytes = 1ull << 30;
constexpr uint64_t kMaxTotalDsSeedBytes = 1ull << 30;

inline uint64_t checked_mul(uint64_t a, uint64_t b) {
    return a && b > std::numeric_limits<uint64_t>::max() / a ? std::numeric_limits<uint64_t>::max() : a * b;
}

inline uint32_t resolved_linear_row_pitch(const ShaderResource& r, uint32_t width, uint32_t bpt) {
    if (r.linear_row_pitch_bytes) return r.linear_row_pitch_bytes;
    const uint64_t tight = checked_mul(width, bpt);
    if (tight > UINT32_MAX) return UINT32_MAX;
    if (r.host_data) return static_cast<uint32_t>(tight);
    if (const uint32_t registered = guest_linear_texture_row_pitch(
            r.gpu_addr, static_cast<uint32_t>(tight)))
        return registered;
    const size_t aligned = linear_sampled_row_pitch(width, bpt);
    return aligned > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(aligned);
}

inline uint64_t resource_footprint_impl(const ShaderResource& r, bool legacy_linear_tight) {
    uint64_t result = r.size;
    if (r.cls != ResourceClass::Texture && r.cls != ResourceClass::StorageImage) {
        if (r.scalar_buffer_dword_count) {
            const uint64_t scalar_bytes = shader_resource_buffer_binding_bytes(r);
            if (scalar_bytes) result = scalar_bytes;
        }
        return result;
    }
    const uint64_t layers = r.img_dim == 3u ? 6u
        : ((r.img_dim == 2u || r.img_dim == 5u) ? std::max(r.depth, 1u) : 1u);
    uint32_t w = r.width ? r.width : 4, h = r.height ? r.height : 4;
    const uint32_t bc = bc_block_bytes(r.format);
    uint64_t decoded = 0;
    bool decoded_is_volume = false;
    if (bc) {
        uint32_t bw = (w + 3) / 4, bh = (h + 3) / 4;
        if (tile_mode_is_tiled(r.tile_mode)) {
            decoded = tiled_elements_bytes(bw, bh, bc, r.tile_mode);
        } else if (r.tile_mode == static_cast<uint32_t>(TileMode::Linear) &&
                   r.cls == ResourceClass::Texture &&
                   (r.img_dim == 1u ||
                    ((r.img_dim == 3u || r.img_dim == 5u) && r.layer_stride_bytes)) &&
                   !r.compression_enabled && !legacy_linear_tight) {
            const uint32_t row_pitch = resolved_linear_row_pitch(r, bw, bc);
            decoded = checked_mul(row_pitch, bh);
        } else {
            decoded = checked_mul(checked_mul(bw, bh), bc);
        }
    } else {
        uint32_t bpt = data_format_bytes(r.format) * (r.num_components ? r.num_components : 1);
        // This is the GUEST allocation footprint, not the backend's eventual upload width.  A
        // sampled image may be converted to RGBA8 later, but its captured source must still include
        // every native component.  Clamping sampled texels wider than four bytes truncated GTA V's
        // 640x360 Uint32x2 SW_64KB_R_X surface to its declared tight span (1,843,200 bytes) instead
        // of the padded tiled span (1,966,080 bytes).  The live compute consumer could read it while
        // replay declined the same ordered producer -> consumer edge as "sampled surface unreadable".
        // Packed formats deliberately report zero per-component width and retain the four-byte word
        // fallback below.
        if (bpt == 0) bpt = 4;
        if (tile_mode_is_tiled(r.tile_mode)) {
            const bool tiled_msaa = r.img_dim == 6u && r.sample_count > 1u;
            if (tiled_msaa) {
                decoded = tiled_msaa_surface_bytes(
                    w, h, r.tile_mode, bpt, r.sample_count);
            } else if (r.img_dim == 2u && r.depth > 1u) {
                decoded = tiled_volume_bytes(w, h, r.depth, r.tile_mode, bpt);
                decoded_is_volume = decoded != 0;
            }
            if (!tiled_msaa && !decoded_is_volume)
                decoded = tiled_surface_bytes(w, h, r.tile_mode, 0, bpt);
        } else if (r.tile_mode == static_cast<uint32_t>(TileMode::Linear) &&
                   ((r.cls == ResourceClass::Texture && r.img_dim == 1u) ||
                    ((r.img_dim == 3u || r.img_dim == 5u) && r.layer_stride_bytes)) &&
                   !r.compression_enabled && !legacy_linear_tight) {
            // Guest-backed linear sampled images default to GFX10's 256-byte row alignment. Exact
            // HLE-producer provenance may override it (AvPlayer's CPU-staged NV12 is tight). Capturing
            // only width*height*bpt made genuinely padded video planes drift and omitted their tail.
            const uint32_t row_pitch = resolved_linear_row_pitch(r, w, bpt);
            decoded = checked_mul(row_pitch, h);
        } else {
            decoded = checked_mul(checked_mul(w, h), bpt);
            if (r.img_dim == 6u && r.sample_count > 1u)
                decoded = checked_mul(decoded, r.sample_count);
        }
    }
    if (r.layer_stride_bytes && layers > 1) {
        const uint64_t selected_bytes = r.in_mip_tail ? r.mip_tail_bytes : decoded;
        const uint64_t level_offset = r.in_mip_tail ? 0u : r.layer_mip_offset_bytes;
        decoded = checked_mul(r.layer_stride_bytes, layers - 1u);
        decoded = decoded > UINT64_MAX - level_offset
            ? UINT64_MAX : decoded + level_offset;
        decoded = decoded > UINT64_MAX - selected_bytes
            ? UINT64_MAX : decoded + selected_bytes;
    } else if (!decoded_is_volume) {
        decoded = checked_mul(decoded, layers);
    }
    return std::max(result, decoded);
}

inline uint64_t resource_footprint(const ShaderResource& r) {
    return resource_footprint_impl(r, false);
}

inline uint64_t legacy_resource_footprint(const ShaderResource& r) {
    return resource_footprint_impl(r, true);
}

inline uint64_t dcc_metadata_footprint(const ShaderResource& r) {
    if (!r.compression_enabled || !r.metadata_addr ||
        (r.cls != ResourceClass::Texture && r.cls != ResourceClass::StorageImage) ||
        bc_block_bytes(r.format))
        return 0;
    // A compressed 2D_MSAA Z_X descriptor names HTILE, not color DCC. Retain the exact published
    // 16-pipe metadata span so offline replay can independently prove whether the base allocation
    // was decompressed. Other Z_X shapes remain unavailable instead of borrowing DCC sizing.
    if (r.img_dim == 6u && r.format == DataFormat::Float32 && r.num_components == 1u)
        return gfx10_htile_msaa_metadata_bytes(
            r.width, r.height, r.tile_mode, r.sample_count, r.meta_pipe_aligned);
    uint32_t bytes_per_texel = data_format_bytes(r.format) * (r.num_components ? r.num_components : 1u);
    if (!bytes_per_texel &&
        (r.format == DataFormat::Float10_11_11 || r.format == DataFormat::Unorm2_10_10_10))
        bytes_per_texel = 4;
    const uint32_t layers = r.img_dim == 3u ? 6u : (r.img_dim == 2u ? std::max(r.depth, 1u) : 1u);
    return gfx10_dcc_metadata_bytes(r.width, r.height, layers, r.tile_mode,
                                    bytes_per_texel, r.meta_pipe_aligned);
}

// Bundle manifests retain blob indices and offsets while their content lives in the bundle's
// deduplicated resource dictionary. make_capture_manifest marks those deliberately empty entries
// with the canonical empty-content hash; ordinary empty/default blobs do not bypass bounds checks.
inline bool capture_blob_payload_omitted(const GpuCaptureBlob& blob) {
    return blob.bytes.empty() && !blob.bytes_read &&
           blob.content_hash == gpu_capture_hash(nullptr, 0);
}

// True when EVERY blob in this capture had its payload deliberately omitted -- i.e. this is a bundle
// MANIFEST, not a capture.
//
// Capture-level and all_of, deliberately. make_capture_manifest() strips every blob, so that is the
// manifest's signature; sampling a single resource instead would let ONE degenerate blob in a
// hand-edited or corrupt .prgcap disable provenance validation for a whole compute -- including on
// the materialize_gpu_replay() path, which is the gate standing between a bundle and a relocation
// replayed without provenance. Note the limit of that claim: a file in which EVERY blob is degenerate
// is still accepted as a manifest. A .prgcap is a developer artifact rather than a trust boundary, so
// that is a stated bound, not a defence.
//
// The proof still happens, just not against the projection. Every capture reaching
// append_gpu_capture_bundle() is produced by capture_submit_items(), which runs
// validate_failure_diagnostics() on the payload-BEARING object at capture time. That is an invariant
// of its two callers (gpu_timeline.cpp's append_runtime_capture_bundle and
// append_capture_to_frame_bundle, both fed by capture_gpustate_submit /
// capture_gpustate_target_submit) and is enforced nowhere else: a third caller must preserve it, or
// restore validation by another route.
inline bool capture_is_manifest(const GpuCaptureFile& capture) {
    return !capture.blobs.empty() &&
           std::all_of(capture.blobs.begin(), capture.blobs.end(), capture_blob_payload_omitted);
}

inline bool is_compute_internal_gds(const ShaderResource& r) {
    return r.binding == kComputeInternalGdsBinding && r.gpu_addr == 0 &&
           r.cls == ResourceClass::ConstantBuffer && r.size == 64u * 1024u && r.stride == 4;
}

constexpr uint32_t kCaptureDmaSelGds = 1u;
constexpr uint32_t kCaptureDmaSelMemory = 3u;
inline bool captured_dma_destination_is_gds(uint32_t sels) {
    return (sels & 0xffu) == kCaptureDmaSelGds;
}
inline bool captured_dma_source_is_gds(uint32_t sels) {
    return ((sels >> 8u) & 0xffu) == kCaptureDmaSelGds;
}
inline bool captured_dma_memory_to_gds_is_supported(const GpuCapturedDmaCopy& copy) {
    const uint32_t source_selector = (copy.sels >> 8u) & 0xffu;
    const bool address_source = (copy.sels & kDmaDataAddressSource) != 0 ||
                                copy.src > UINT32_MAX;
    return captured_dma_destination_is_gds(copy.sels) &&
           source_selector == kCaptureDmaSelMemory && address_source;
}
inline bool capture_is_metadata_only(const GpuCaptureFile& capture) {
    return std::any_of(capture.metadata.renderer_env.begin(),
                       capture.metadata.renderer_env.end(), [](const auto& entry) {
        return entry.first == "PROSPER_GPU_CAPTURE_METADATA_ONLY" &&
               !entry.second.empty() && entry.second != "0" && entry.second != "off";
    });
}
inline bool capture_has_internal_gds_descriptor(const GpuCaptureFile& capture) {
    auto table_has_gds = [](const GpuCapturedTable& table) {
        return std::any_of(table.resources.begin(), table.resources.end(),
                           [](const GpuCapturedResource& resource) {
            return is_compute_internal_gds(resource.resource);
        });
    };
    for (const auto& draw : capture.draws)
        if (table_has_gds(draw.vrt) || table_has_gds(draw.prt)) return true;
    for (const auto& compute : capture.computes)
        if (table_has_gds(compute.resources)) return true;
    return false;
}

inline bool validate_rtt_seed(const GpuCaptureRttSeed& seed, std::string& error) {
    if (!seed.guest_addr || !seed.width || !seed.height) {
        error = "RTT seed has an invalid address or extent"; return false;
    }
    uint32_t bytes_per_pixel = 0;
    switch (seed.format) {
        case GpuCaptureColorFormat::Rgba8Unorm: bytes_per_pixel = 4; break;
        case GpuCaptureColorFormat::Rgba16Float: bytes_per_pixel = 8; break;
        case GpuCaptureColorFormat::R11G11B10Float: bytes_per_pixel = 4; break;
        case GpuCaptureColorFormat::R8Unorm: bytes_per_pixel = 1; break;
        case GpuCaptureColorFormat::R32Uint: bytes_per_pixel = 4; break;
        case GpuCaptureColorFormat::R32Float: bytes_per_pixel = 4; break;
        case GpuCaptureColorFormat::Rg8Unorm: bytes_per_pixel = 2; break;
        case GpuCaptureColorFormat::Rgba32Float: bytes_per_pixel = 16; break;
        case GpuCaptureColorFormat::Rg16Float: bytes_per_pixel = 4; break;
        case GpuCaptureColorFormat::R16Float: bytes_per_pixel = 2; break;
        default: error = "RTT seed has an unsupported color format"; return false;
    }
    const uint64_t pixels = checked_mul(seed.width, seed.height);
    const uint64_t bytes = checked_mul(pixels, bytes_per_pixel);
    if (bytes > max_blob_bytes() || bytes != seed.rgba.size()) {
        error = "RTT seed byte count does not match its color format and extent"; return false;
    }
    return true;
}

inline auto ds_seed_key(const GpuCaptureDsSeed& seed) {
    return std::tuple(seed.depth_read_base, seed.depth_write_base,
                      seed.stencil_read_base, seed.stencil_write_base,
                      seed.htile_data_base, seed.width, seed.height,
                      static_cast<uint32_t>(seed.format), seed.slice);
}

inline bool validate_ds_seed(const GpuCaptureDsSeed& seed, std::string& error) {
    if ((!seed.depth_read_base && !seed.depth_write_base && !seed.stencil_read_base &&
         !seed.stencil_write_base && !seed.htile_data_base) || !seed.width || !seed.height) {
        error = "DS seed has an invalid identity or extent"; return false;
    }
    if (seed.format != GpuCaptureDsFormat::D32Float &&
        seed.format != GpuCaptureDsFormat::D32FloatS8) {
        error = "DS seed has an unsupported format"; return false;
    }
    if (!seed.depth_valid && !seed.stencil_valid) {
        error = "DS seed has no valid plane"; return false;
    }
    const uint64_t pixels = checked_mul(seed.width, seed.height);
    const uint64_t depth_bytes = checked_mul(pixels, 4);
    if ((seed.depth_valid && (depth_bytes > max_blob_bytes() || depth_bytes != seed.depth.size())) ||
        (!seed.depth_valid && !seed.depth.empty())) {
        error = "DS seed depth byte count does not match its extent and validity"; return false;
    }
    if ((seed.stencil_valid &&
         (seed.format != GpuCaptureDsFormat::D32FloatS8 || pixels > max_blob_bytes() ||
          pixels != seed.stencil.size())) ||
        (!seed.stencil_valid && !seed.stencil.empty())) {
        error = "DS seed stencil byte count does not match its extent, format, and validity";
        return false;
    }
    return true;
}

inline uint64_t shader_hash(const std::vector<uint32_t>& words) {
    return gpu_capture_hash(reinterpret_cast<const uint8_t*>(words.data()),
                            words.size() * sizeof(uint32_t));
}

inline void collect_shader_versions(GpuCaptureFile& capture) {
    capture.shader_versions.clear();
    auto add = [&](const std::vector<uint32_t>& words) {
        const uint64_t hash = shader_hash(words);
        auto it = std::find_if(capture.shader_versions.begin(), capture.shader_versions.end(),
            [&](const auto& version) { return version.content_hash == hash && version.words == words; });
        if (it == capture.shader_versions.end())
            capture.shader_versions.push_back({hash, words});
    };
    for (const auto& draw : capture.draws) {
        add(draw.vs);
        if (!draw.gs.empty()) add(draw.gs);
        add(draw.fs);
    }
    for (const auto& compute : capture.computes) add(compute.spirv);
}

inline uint32_t shader_version_index(const std::vector<GpuCaptureShaderVersion>& versions,
                              const std::vector<uint32_t>& words) {
    const uint64_t hash = shader_hash(words);
    auto it = std::find_if(versions.begin(), versions.end(), [&](const auto& version) {
        return version.content_hash == hash && version.words == words;
    });
    return it == versions.end() ? 0xFFFFFFFFu : static_cast<uint32_t>(it - versions.begin());
}

inline OperationIdentity operation_identity(SubmitOperationKind kind, uint64_t source_index,
                                     uint64_t command_order) {
    return {static_cast<uint8_t>(kind), source_index, command_order};
}

inline bool validate_dma_copies(const GpuCaptureFile& capture, std::string& error) {
    if (capture.dma_copies.size() > kMaxOperations) {
        error = "invalid ordered DMA count";
        return false;
    }
    std::vector<uint8_t> referenced(capture.dma_copies.size(), 0);
    auto validate_blob = [&](uint32_t index, uint64_t offset, uint32_t bytes,
                             const char* invalid, const char* exceeds) {
        if (index == 0xFFFFFFFFu) {
            if (!offset) return true;
            error = invalid;
            return false;
        }
        if (index >= capture.blobs.size()) {
            error = invalid;
            return false;
        }
        const auto& blob = capture.blobs[index];
        if (!capture_blob_payload_omitted(blob) &&
            (offset > blob.bytes.size() || bytes > blob.bytes.size() - offset)) {
            error = exceeds;
            return false;
        }
        return true;
    };
    for (const auto& copy : capture.dma_copies) {
        const bool destination_gds = captured_dma_destination_is_gds(copy.sels);
        const bool source_gds = captured_dma_source_is_gds(copy.sels);
        if (source_gds) {
            error = "unsupported ordered DMA GDS source selector";
            return false;
        }
        if (destination_gds && !captured_dma_memory_to_gds_is_supported(copy)) {
            error = "unsupported ordered DMA memory-to-GDS selector form";
            return false;
        }
        const bool gds_destination_valid = !destination_gds ||
            (captured_dma_memory_to_gds_is_supported(copy) &&
             copy.dst < 64u * 1024u && copy.bytes <= 64u * 1024u - copy.dst &&
             !(copy.dst & 3u) && !(copy.bytes & 3u) &&
             ((copy.destination_blob_index == 0xFFFFFFFFu &&
               copy.destination_blob_offset == 0 &&
               (capture_is_metadata_only(capture) ||
                capture_has_internal_gds_descriptor(capture))) ||
              (copy.destination_blob_index != 0xFFFFFFFFu &&
               copy.destination_blob_offset == copy.dst)));
        if (destination_gds && copy.destination_blob_index != 0xFFFFFFFFu) {
            if (copy.destination_blob_index >= capture.blobs.size()) {
                error = "ordered DMA GDS destination references an invalid capture blob";
                return false;
            }
            const auto& blob = capture.blobs[copy.destination_blob_index];
            if (blob.guest_addr != 0 ||
                (!capture_blob_payload_omitted(blob) &&
                 (blob.bytes.size() != 64u * 1024u ||
                  blob.bytes_read != blob.bytes.size()))) {
                error = "ordered DMA GDS destination has an invalid pre-submit snapshot";
                return false;
            }
        }
        if ((!destination_gds && !copy.dst) || !copy.src || !copy.bytes ||
            copy.bytes > max_blob_bytes() || !gds_destination_valid ||
            (!destination_gds &&
             copy.dst > std::numeric_limits<uint64_t>::max() - copy.bytes) ||
            copy.src > std::numeric_limits<uint64_t>::max() - copy.bytes ||
            !validate_blob(copy.destination_blob_index, copy.destination_blob_offset, copy.bytes,
                           "ordered DMA destination references an invalid capture blob",
                           "ordered DMA destination exceeds its capture blob") ||
            !validate_blob(copy.source_blob_index, copy.source_blob_offset, copy.bytes,
                           "ordered DMA source references an invalid capture blob",
                           "ordered DMA source exceeds its capture blob")) {
            if (error.empty()) error = "invalid ordered DMA record";
            return false;
        }
    }
    for (const auto& operation : capture.operations) {
        if (operation.kind > SubmitOperationKind::DmaCopy) {
            error = "invalid operation kind";
            return false;
        }
        if (operation.kind != SubmitOperationKind::DmaCopy) continue;
        if (!operation.realized || operation.source_index >= capture.dma_copies.size() ||
            capture.dma_copies[operation.source_index].command_order != operation.command_order ||
            referenced[operation.source_index] == UINT8_MAX) {
            error = "ordered DMA operation does not match its record";
            return false;
        }
        ++referenced[operation.source_index];
    }
    if (std::find_if(referenced.begin(), referenced.end(), [](uint8_t count) { return count != 1; }) !=
        referenced.end()) {
        error = "ordered DMA record must have exactly one operation";
        return false;
    }
    return true;
}

inline GpuCaptureResourceInputUnavailableReason direct_vsharp_coverage_reason(
        const std::array<uint32_t, 4>& dwords) {
    const DecodedBufferDescriptor decoded = decode_buffer_descriptor(dwords.data());
    const bool format_unavailable =
        decoded.format == DataFormat::Unknown || decoded.num_components == 0;
    const bool size_unavailable = decoded.size_bytes == 0;
    if (format_unavailable && size_unavailable)
        return GpuCaptureResourceInputUnavailableReason::RawFormatAndSizeUnavailable;
    if (format_unavailable)
        return GpuCaptureResourceInputUnavailableReason::RawFormatUnavailable;
    if (size_unavailable)
        return GpuCaptureResourceInputUnavailableReason::RawSizeUnavailable;
    return GpuCaptureResourceInputUnavailableReason::None;
}

inline bool is_direct_vsharp_coverage_reason(GpuCaptureResourceInputUnavailableReason reason) {
    return reason == GpuCaptureResourceInputUnavailableReason::None ||
        reason == GpuCaptureResourceInputUnavailableReason::RawFormatUnavailable ||
        reason == GpuCaptureResourceInputUnavailableReason::RawSizeUnavailable ||
        reason == GpuCaptureResourceInputUnavailableReason::RawFormatAndSizeUnavailable;
}

inline bool validate_resource_provenance(const GpuCaptureFile& capture, std::string& error) {
    if (capture.resource_provenance.size() > 1) {
        error = "invalid resource provenance record count";
        return false;
    }
    for (const auto& provenance : capture.resource_provenance) {
        if ((provenance.stage != ShaderProgramStage::Vertex &&
             provenance.stage != ShaderProgramStage::Fragment) ||
            static_cast<uint32_t>(provenance.resource_class) >
                static_cast<uint32_t>(ResourceClass::StorageImage) ||
            !provenance.guest_addr || !provenance.requested_bytes ||
            provenance.requested_bytes > max_blob_bytes() ||
            provenance.realization_blob_index >= capture.blobs.size() ||
            provenance.post_blob_index >= capture.blobs.size()) {
            error = "invalid resource provenance identity";
            return false;
        }
        const auto& realization = capture.blobs[provenance.realization_blob_index];
        const auto& post = capture.blobs[provenance.post_blob_index];
        if (realization.guest_addr != provenance.guest_addr ||
            realization.bytes.size() != provenance.requested_bytes ||
            realization.content_hash != provenance.realization_content_hash ||
            provenance.post_blob_offset > post.bytes.size() ||
            provenance.requested_bytes > post.bytes.size() - provenance.post_blob_offset ||
            gpu_capture_hash(post.bytes.data() + provenance.post_blob_offset,
                             static_cast<size_t>(provenance.requested_bytes)) !=
                provenance.post_content_hash) {
            error = "invalid resource provenance blob reference";
            return false;
        }

        const GpuCapturedDraw* captured_draw = nullptr;
        const GpuCapturedResource* captured_resource = nullptr;
        for (const auto& draw : capture.draws) {
            if (draw.draw_index != provenance.draw_index) continue;
            if (captured_draw && captured_draw != &draw) {
                error = "ambiguous resource provenance draw identity";
                return false;
            }
            captured_draw = &draw;
            const auto& table = provenance.stage == ShaderProgramStage::Vertex
                ? draw.vrt : draw.prt;
            for (const auto& resource : table.resources) {
                if (resource.resource.binding != provenance.binding) continue;
                if (captured_resource) {
                    error = "ambiguous resource provenance identity";
                    return false;
                }
                captured_resource = &resource;
            }
        }
        if (!captured_resource ||
            captured_resource->resource.cls != provenance.resource_class ||
            captured_resource->resource.gpu_addr != provenance.guest_addr ||
            captured_resource->captured_size != provenance.requested_bytes ||
            captured_resource->resource.srt_offset != provenance.srt_offset ||
            captured_resource->resource.sgpr_base != provenance.sgpr_base ||
            captured_resource->blob_index != provenance.post_blob_index ||
            captured_resource->blob_offset != provenance.post_blob_offset) {
            error = "resource provenance does not match its captured descriptor";
            return false;
        }
        const auto reason_value = static_cast<uint32_t>(provenance.input_unavailable_reason);
        if (provenance.input_mode == GpuCaptureResourceInputMode::Unavailable) {
            if (provenance.input_unavailable_reason ==
                    GpuCaptureResourceInputUnavailableReason::None ||
                reason_value > static_cast<uint32_t>(
                    GpuCaptureResourceInputUnavailableReason::MissingWriteProvenance) ||
                provenance.input_write_provenance_mask != 0 ||
                provenance.input_sh_register_base != 0xFFFFFFFFu ||
                std::any_of(provenance.input_dwords.begin(), provenance.input_dwords.end(),
                            [](uint32_t value) { return value != 0; }) ||
                std::any_of(provenance.input_last_writes.begin(),
                            provenance.input_last_writes.end(),
                            [](uint64_t value) { return value != 0; }) ||
                std::any_of(provenance.input_write_sources.begin(),
                            provenance.input_write_sources.end(),
                            [](uint64_t value) { return value != 0; })) {
                error = "invalid unavailable resource input witness";
                return false;
            }
        } else if (provenance.input_mode == GpuCaptureResourceInputMode::DirectVSharp) {
            if (!captured_draw || provenance.input_write_provenance_mask != 0x0fu ||
                provenance.input_sh_register_base > GpuState::kRegOffsetLimit - 4u ||
                !is_direct_vsharp_coverage_reason(
                    provenance.input_unavailable_reason) ||
                provenance.input_unavailable_reason !=
                    direct_vsharp_coverage_reason(provenance.input_dwords)) {
                error = "invalid direct resource input witness";
                return false;
            }
            for (const uint64_t write : provenance.input_last_writes) {
                const uint64_t order = write & ~GpuState::kProvIndirect;
                if (!order || order >= captured_draw->command_order) {
                    error = "resource input write does not precede selected draw";
                    return false;
                }
            }
            for (const uint64_t source : provenance.input_write_sources) {
                const uint8_t queue = static_cast<uint8_t>(source & 0xffu);
                const uint8_t jump_depth = static_cast<uint8_t>((source >> 8u) & 0xffu);
                const uint32_t fold = static_cast<uint32_t>(source >> 16u);
                // pack_prov_src allocates bits 0..47 as q8/j8/fold32.  The command processor
                // admits queue origins 0..3 and records Jump depth 0..8 inclusive (the parent at
                // depth 7 may enter its final child at depth 8).  A top-level fold id is never zero.
                if (queue > 3u || jump_depth > 8u || fold == 0u || (source >> 48u) != 0u) {
                    error = "invalid resource input write source";
                    return false;
                }
            }
        } else {
            error = "invalid resource input witness mode";
            return false;
        }
    }
    return true;
}

inline bool validate_compute_config(const ComputeShaderConfig& config,
                             const ComputeLaunchDimensions& launch,
                             uint32_t required_subgroup_size,
                             const char* state_error,
                             std::string& error) {
    const bool subgroup_valid = config.native_subgroup_size == 0 ||
        config.native_subgroup_size == 32 || config.native_subgroup_size == 64;
    if ((required_subgroup_size != 0 && required_subgroup_size != 32 &&
         required_subgroup_size != 64) || !subgroup_valid) {
        error = "invalid compute required-subgroup size";
        return false;
    }
    if (config.user_sgprs.size() > 32 || !config.local_x || !config.local_y ||
        !config.local_z || config.local_x != launch.local_x ||
        config.local_y != launch.local_y || config.local_z != launch.local_z ||
        config.threads_x != launch.threads_x ||
        config.threads_y != launch.threads_y ||
        config.threads_z != launch.threads_z ||
        (config.wave_size != 32 && config.wave_size != 64) ||
        config.tidig_comp_cnt > 3 || config.lds_bytes > 65536 ||
        (config.lds_bytes & 511u) ||
        config.native_subgroup_size != required_subgroup_size ||
        (config.native_storage_format_support & ~kNativeStorageFormatSupportMask)) {
        error = state_error;
        return false;
    }
    return true;
}

inline bool validate_compute_recompile_state(const GpuCapturedCompute& compute,
                                      std::string& error) {
    if (!compute.recompile_config_available) {
        if (compute.raw_shader_index != 0xFFFFFFFFu) {
            error = "compute raw shader has no recompile state";
            return false;
        }
        return true;
    }
    return validate_compute_config(compute.recompile_config, compute.launch,
                                   compute.required_subgroup_size,
                                   "invalid compute recompile state", error);
}

inline bool captured_compute_has_null_guarded_raw_store(
        const GpuCapturedCompute& compute) {
    return compute.resources.present &&
        std::any_of(compute.resources.resources.begin(),
                    compute.resources.resources.end(),
                    [](const GpuCapturedResource& captured) {
                        return is_proven_null_guarded_raw_store(captured.resource);
                    });
}

inline bool validate_captured_null_guarded_raw_store(
        const GpuCaptureFile& capture, const GpuCapturedCompute& compute,
        std::string& error) {
    if (!captured_compute_has_null_guarded_raw_store(compute)) return true;
    if (!compute.recompile_config_available ||
        compute.raw_shader_index >= capture.raw_shader_versions.size()) {
        error = "guarded-null store marker lacks raw recompile provenance";
        return false;
    }
    const auto& raw = capture.raw_shader_versions[compute.raw_shader_index].words;
    const auto& user = compute.recompile_config.user_sgprs;
    if (!rdna2_gta5_null_guarded_raw_store_dispatch(
            raw.data(), raw.size(), user.data(), user.size())) {
        error = "guarded-null store marker has stale shader or dispatch provenance";
        return false;
    }
    for (const GpuCapturedResource& captured : compute.resources.resources) {
        const ShaderResource& marker = captured.resource;
        if (!is_proven_null_guarded_raw_store(marker)) continue;
        if (marker.fetch_pc >= raw.size()) {
            error = "guarded-null store marker has stale shader or dispatch provenance";
            return false;
        }
        Rdna2Inst site = rdna2_decode_one(raw.data() + marker.fetch_pc,
                                         raw.size() - marker.fetch_pc);
        site.pc = marker.fetch_pc;
        if (!rdna2_gta5_null_guarded_raw_store_site(site)) {
            error = "guarded-null store marker has stale shader or dispatch provenance";
            return false;
        }
    }
    return true;
}

inline bool captured_compute_has_nullable_output_raw_buffer(
        const GpuCapturedCompute& compute) {
    return compute.resources.present &&
        std::any_of(compute.resources.resources.begin(),
                    compute.resources.resources.end(),
                    [](const GpuCapturedResource& captured) {
                        return is_nullable_raw_buffer_marker_candidate(captured.resource);
                    });
}

// PROSPER_CAPTURE_ALLOW_UNPROVEN_INDIRECT=1 — accept a capture whose GPU-driven provenance cannot
// be proven exactly, and say so.
//
// These validators refuse the WHOLE BUNDLE when one compute dispatch's packed-pointer or
// indirect-pointer state is not exactly provable. That is right for a bundle meant to REPLAY: a
// relocation replayed without provenance would fabricate a pointer. It is wrong for a bundle meant
// to be READ -- `gpu_replay --inspect-only`, the draw list, the render targets -- and on GTA V it
// makes the F9 frame grab unusable on the one title whose world is GPU-driven, which is exactly
// where a draw-level view is needed. The charter calls that grab the fastest loop for a graphical
// bug; on this title it could not be run at all.
//
// Opt-in and reported, because a bundle that silently claimed replay fidelity it does not have
// would be worse than the refusal it replaces.
inline bool capture_allow_unproven_provenance(const char* what, const char* detail) {
    static const bool allowed = std::getenv("PROSPER_CAPTURE_ALLOW_UNPROVEN_INDIRECT") != nullptr;
    if (!allowed) return false;
    static std::atomic<int> reported{0};
    if (reported.fetch_add(1) < 12)
        std::fprintf(stderr,
                     "[capture] ALLOW_UNPROVEN_INDIRECT: accepting %s (%s). "
                     "THIS BUNDLE IS FOR INSPECTION, NOT FAITHFUL REPLAY.\n",
                     what, detail);
    return true;
}

inline bool validate_captured_nullable_output_raw_buffer(
        const GpuCaptureFile& capture, const GpuCapturedCompute& compute,
        std::string& error) {
    if (!captured_compute_has_nullable_output_raw_buffer(compute)) return true;
    if (!compute.recompile_config_available ||
        compute.raw_shader_index >= capture.raw_shader_versions.size()) {
        error = "nullable-output marker lacks raw recompile provenance";
        return false;
    }
    if (compute.launch.local_x != kGtaNullableOutputLocalSize ||
        compute.launch.local_y != 1u || compute.launch.local_z != 1u ||
        compute.recompile_config.user_sgprs.size() <= 7u ||
        compute.recompile_config.user_sgprs[7] == 0u ||
        compute.recompile_config.user_sgprs[7] > kGtaNullableOutputMaxRecordCount ||
        static_cast<uint64_t>(compute.launch.threads_x) !=
            static_cast<uint64_t>(compute.recompile_config.user_sgprs[7]) *
                kGtaNullableOutputLocalSize ||
        compute.launch.threads_y != 1u || compute.launch.threads_z != 1u ||
        compute.launch.groups_x != compute.recompile_config.user_sgprs[7] ||
        compute.launch.groups_y != 1u || compute.launch.groups_z != 1u) {
        if (capture_allow_unproven_provenance("a nullable-output marker",
                                              "stale captured launch provenance")) return true;
        error = "nullable-output marker has stale captured launch provenance";
        return false;
    }

    // Below the payload-INDEPENDENT checks above on purpose: a manifest carries markers, counts,
    // launch dimensions and recompile provenance in full, and on the deserialize side those are the
    // only validation it ever receives. What it cannot carry is the witness BYTES. Not everything
    // past this point needs them -- the sentinel and raw-shader checks below read descriptors and
    // `internal_bytes`, which the manifest carries in full -- but the blob dereference does:
    // `blob.bytes.data() + blob_offset` on an empty vector is `nullptr + offset`, declared as large
    // as the witness, which the proof then reads through.
    if (capture_is_manifest(capture)) return true;

    ShaderResourceTable validated_resources;
    validated_resources.resources.reserve(compute.resources.resources.size());
    for (const GpuCapturedResource& captured : compute.resources.resources) {
        ShaderResource resource = captured.resource;
        if (is_nullable_raw_buffer_marker_candidate(resource)) {
            if (!is_proven_null_nullable_raw_buffer(resource)) {
                error = "nullable-output marker has malformed sentinel metadata";
                return false;
            }
            if (captured.blob_index >= capture.blobs.size() ||
                captured.blob_offset > capture.blobs[captured.blob_index].bytes.size() ||
                kGtaNullableOutputWitnessBytes >
                    capture.blobs[captured.blob_index].bytes.size() - captured.blob_offset) {
                error = "nullable-output marker lacks its table witness";
                return false;
            }
            resource.host_data = const_cast<uint8_t*>(
                capture.blobs[captured.blob_index].bytes.data() + captured.blob_offset);
            resource.host_data_size = kGtaNullableOutputWitnessBytes;
        }
        validated_resources.resources.push_back(resource);
    }

    const auto& raw = capture.raw_shader_versions[compute.raw_shader_index].words;
    if (!rdna2_gta5_nullable_output_dispatch(
            raw.data(), raw.size(), compute.recompile_config, validated_resources)) {
        error = "nullable-output marker has stale shader, launch, or table provenance";
        return false;
    }
    return true;
}

inline bool captured_compute_has_gta5_cf9200_no_backing(
        const GpuCapturedCompute& compute) {
    return compute.resources.present &&
        std::any_of(compute.resources.resources.begin(),
                    compute.resources.resources.end(),
                    [](const GpuCapturedResource& captured) {
                        return is_gta5_cf9200_no_backing_marker_candidate(captured.resource);
                    });
}

inline bool validate_captured_gta5_cf9200_no_backing(
        const GpuCaptureFile& capture, const GpuCapturedCompute& compute,
        std::string& error) {
    if (!captured_compute_has_gta5_cf9200_no_backing(compute)) return true;
    if (!compute.recompile_config_available ||
        compute.raw_shader_index >= capture.raw_shader_versions.size() ||
        compute.launch.groups_x != 1u || compute.launch.groups_y != 1u ||
        compute.launch.groups_z != 1u) {
        error = "GTA root-record marker lacks exact shader or launch provenance";
        return false;
    }

    if (capture_is_manifest(capture)) return true;

    // Below the payload-INDEPENDENT checks above on purpose: a manifest carries markers, counts,
    // launch dimensions and recompile provenance in full, and on the deserialize side those are the
    // only validation it ever receives. What it cannot carry is the witness BYTES. Not everything
    // past this point needs them -- the sentinel and raw-shader checks below read descriptors and
    // `internal_bytes`, which the manifest carries in full -- but the blob dereference does:
    // `blob.bytes.data() + blob_offset` on an empty vector is `nullptr + offset`, declared as large
    // as the witness, which the proof then reads through.

    ShaderResourceTable validated_resources;
    validated_resources.resources.reserve(compute.resources.resources.size());
    for (const GpuCapturedResource& captured : compute.resources.resources) {
        ShaderResource resource = captured.resource;
        const bool relevant = resource.fetch_pc == kGtaCf9200RootPc ||
                              is_gta5_cf9200_no_backing_marker_candidate(resource);
        if (is_gta5_cf9200_no_backing_marker_candidate(resource) &&
            !is_proven_gta5_cf9200_no_backing(resource)) {
            error = "GTA root-record marker has malformed sentinel metadata";
            return false;
        }
        if (relevant) {
            if (captured.blob_index >= capture.blobs.size() ||
                captured.blob_offset > capture.blobs[captured.blob_index].bytes.size() ||
                kGtaCf9200RootBytes >
                    capture.blobs[captured.blob_index].bytes.size() - captured.blob_offset) {
                error = "GTA root-record marker lacks its 224-byte witness";
                return false;
            }
            resource.host_data = const_cast<uint8_t*>(
                capture.blobs[captured.blob_index].bytes.data() + captured.blob_offset);
            resource.host_data_size = kGtaCf9200RootBytes;
        }
        validated_resources.resources.push_back(resource);
    }
    const auto& raw = capture.raw_shader_versions[compute.raw_shader_index].words;
    if (!rdna2_gta5_cf9200_no_backing_dispatch(
            raw.data(), raw.size(), compute.recompile_config, validated_resources)) {
        error = "GTA root-record marker has stale shader, launch, root, or site provenance";
        return false;
    }
    return true;
}

inline bool captured_resource_has_packed_pointer_state(const GpuCapturedResource& captured) {
    return !captured.internal_bytes.empty() &&
           !is_compute_internal_gds(captured.resource);
}

inline bool captured_resource_has_indirect_pointer_state(
        const GpuCapturedResource& captured) {
    return !captured.internal_bytes.empty() &&
           is_indirect_pointer_relocation_serialized(
               captured.resource, captured.internal_bytes.data(),
               captured.internal_bytes.size());
}

inline bool captured_resource_has_gta5_packed_pointer_candidate(
        const GpuCapturedResource& captured) {
    return captured_resource_has_packed_pointer_state(captured) &&
           !captured_resource_has_indirect_pointer_state(captured);
}

inline bool validate_captured_gta5_packed_pointer(
        const GpuCaptureFile& capture, const GpuCapturedCompute& compute,
        std::string& error) {
    if (!compute.resources.present) return true;

    const size_t candidates = static_cast<size_t>(std::count_if(
        compute.resources.resources.begin(), compute.resources.resources.end(),
        captured_resource_has_gta5_packed_pointer_candidate));
    if (!candidates) return true;
    if (capture.format_version < 51u || candidates != 1u ||
        !compute.recompile_config_available ||
        compute.raw_shader_index >= capture.raw_shader_versions.size()) {
        if (capture_allow_unproven_provenance("packed-pointer state",
                                              "no exact compute provenance")) return true;
        error = "packed-pointer state lacks exact compute provenance";
        return false;
    }
    const auto& config = compute.recompile_config;
    const uint64_t groups_x =
        (static_cast<uint64_t>(config.threads_x) + config.local_x - 1u) / config.local_x;
    if (compute.launch.groups_x != groups_x || compute.launch.groups_y != 1u ||
        compute.launch.groups_z != 1u) {
        if (capture_allow_unproven_provenance("packed-pointer state",
                                              "stale captured launch provenance")) return true;
        error = "packed-pointer state has stale captured launch provenance";
        return false;
    }

    // Below the payload-INDEPENDENT checks above on purpose: a manifest carries markers, counts,
    // launch dimensions and recompile provenance in full, and on the deserialize side those are the
    // only validation it ever receives. What it cannot carry is the witness BYTES. Not everything
    // past this point needs them -- the sentinel and raw-shader checks below read descriptors and
    // `internal_bytes`, which the manifest carries in full -- but the blob dereference does:
    // `blob.bytes.data() + blob_offset` on an empty vector is `nullptr + offset`, declared as large
    // as the witness, which the proof then reads through.
    if (capture_is_manifest(capture)) return true;

    ShaderResourceTable validated_resources;
    validated_resources.resources.reserve(compute.resources.resources.size());
    for (const GpuCapturedResource& captured : compute.resources.resources) {
        ShaderResource resource = captured.resource;
        resource.indirect_buffer_contract_tag = 0u;
        resource.indirect_buffer_binding_bytes = 0u;
        resource.indirect_buffer_slot_count = 0u;
        resource.indirect_buffer_header_bytes = 0u;
        resource.indirect_buffer_slot_bytes = 0u;
        resource.host_data = nullptr;
        resource.host_data_size = 0u;
        if (captured_resource_has_gta5_packed_pointer_candidate(captured)) {
            if (!is_gta5_packed_pointer_serialized_shadow(
                    resource, captured.internal_bytes.data(), captured.internal_bytes.size())) {
                error = "packed-pointer state has malformed internal bytes";
                return false;
            }
            resource.host_data = const_cast<uint8_t*>(captured.internal_bytes.data());
            resource.host_data_size = captured.internal_bytes.size();
        } else if (captured.blob_index != UINT32_MAX) {
            if (captured.blob_index >= capture.blobs.size() ||
                captured.blob_offset > capture.blobs[captured.blob_index].bytes.size()) {
                error = "packed-pointer state references an invalid resource blob";
                return false;
            }
            const auto& blob = capture.blobs[captured.blob_index].bytes;
            resource.host_data = const_cast<uint8_t*>(blob.data() + captured.blob_offset);
            resource.host_data_size = blob.size() - captured.blob_offset;
        }
        validated_resources.resources.push_back(resource);
    }

    const auto& raw = capture.raw_shader_versions[compute.raw_shader_index].words;
    if (!rdna2_gta5_packed_pointer_shader(raw.data(), raw.size()) ||
        !discover_rdna2_gta5_packed_pointer(
            raw.data(), raw.size(), config, validated_resources)) {
        error = "packed-pointer state has stale shader, launch, or table provenance";
        return false;
    }
    return true;
}

inline bool validate_captured_indirect_pointer_relocations(
        const GpuCaptureFile& capture, const GpuCapturedCompute& compute,
        std::string& error) {
    if (!compute.resources.present) return true;

    const size_t marked = static_cast<size_t>(std::count_if(
        compute.resources.resources.begin(), compute.resources.resources.end(),
        [](const GpuCapturedResource& captured) {
            return is_indirect_pointer_relocation_marker_candidate(captured.resource);
        }));
    const size_t candidates = static_cast<size_t>(std::count_if(
        compute.resources.resources.begin(), compute.resources.resources.end(),
        captured_resource_has_indirect_pointer_state));
    if (!candidates) {
        if (!marked) return true;
        error = "indirect-pointer relocation marker has no complete carrier";
        return false;
    }
    if (marked && marked != candidates) {
        error = "indirect-pointer relocation marker/carrier count mismatch";
        return false;
    }
    // Name WHICH precondition failed. These four are independent, they fail for unrelated reasons,
    // and the one that fires decides whether the gap is a stale capture, a missing config, or a shape
    // this validator was simply never written for -- but the message used to be the same sentence for
    // all four, so the only way to tell them apart was to read this function. A whole title's F9
    // bundles aborted on it (GTA V PPSA04263) with nothing in the log to say which arm to pursue.
    const char* missing =
        capture.format_version < 53u                                    ? "capture format older than v53"
        : !compute.recompile_config_available                           ? "no captured recompile config"
        : compute.raw_shader_index >= capture.raw_shader_versions.size() ? "raw shader index out of range"
        : candidates != 1u                                              ? "carrier count is not exactly 1"
                                                                        : nullptr;
    if (missing) {
        if (capture_allow_unproven_provenance("an indirect-pointer relocation", missing)) return true;
        error = std::string("indirect-pointer relocation lacks exact compute provenance (") +
                missing + ", carriers=" + std::to_string(candidates) +
                " markers=" + std::to_string(marked) +
                " format_version=" + std::to_string(capture.format_version) + ")";
        return false;
    }
    const auto& config = compute.recompile_config;
    const uint64_t groups_x =
        (static_cast<uint64_t>(config.threads_x) + config.local_x - 1u) /
        config.local_x;
    if (compute.launch.groups_x != groups_x || compute.launch.groups_y != 1u ||
        compute.launch.groups_z != 1u) {
        if (capture_allow_unproven_provenance("an indirect-pointer relocation",
                                              "stale captured launch provenance")) return true;
        error = "indirect-pointer relocation has stale captured launch provenance";
        return false;
    }

    // Below the payload-INDEPENDENT checks above on purpose: a manifest carries markers, counts,
    // launch dimensions and recompile provenance in full, and on the deserialize side those are the
    // only validation it ever receives. What it cannot carry is the witness BYTES. Not everything
    // past this point needs them -- the sentinel and raw-shader checks below read descriptors and
    // `internal_bytes`, which the manifest carries in full -- but the blob dereference does:
    // `blob.bytes.data() + blob_offset` on an empty vector is `nullptr + offset`, declared as large
    // as the witness, which the proof then reads through.
    if (capture_is_manifest(capture)) return true;

    ShaderResourceTable validated_resources;
    validated_resources.resources.reserve(compute.resources.resources.size());
    for (const GpuCapturedResource& captured : compute.resources.resources) {
        ShaderResource resource = captured.resource;
        resource.indirect_pointer_relocation = {};
        resource.host_data = nullptr;
        resource.host_data_size = 0u;
        if (captured_resource_has_indirect_pointer_state(captured)) {
            resource.host_data = const_cast<uint8_t*>(captured.internal_bytes.data());
            resource.host_data_size = captured.internal_bytes.size();
        } else if (captured.blob_index != UINT32_MAX) {
            if (captured.blob_index >= capture.blobs.size() ||
                captured.blob_offset >
                    capture.blobs[captured.blob_index].bytes.size()) {
                error = "indirect-pointer relocation references an invalid resource blob";
                return false;
            }
            const auto& blob = capture.blobs[captured.blob_index].bytes;
            resource.host_data = const_cast<uint8_t*>(
                blob.data() + captured.blob_offset);
            resource.host_data_size = blob.size() - captured.blob_offset;
        }
        validated_resources.resources.push_back(resource);
    }

    const auto& raw = capture.raw_shader_versions[compute.raw_shader_index].words;
    if (!discover_rdna2_indirect_pointer_relocations(
            raw.data(), raw.size(), config, validated_resources) ||
        !validate_rdna2_indirect_pointer_relocations(
            raw.data(), raw.size(), config, validated_resources)) {
        error = "indirect-pointer relocation has stale shader, launch, source, or proof state";
        return false;
    }
    return true;
}

inline bool validate_failure_diagnostics(const GpuCaptureFile& capture, std::string& error) {
    if (capture.raw_shader_versions.size() > kMaxResources ||
        capture.failure_diagnostics.size() > kMaxOperations) {
        error = "invalid failed-operation diagnostic count";
        return false;
    }
    uint64_t raw_words = 0;
    for (const auto& shader : capture.raw_shader_versions) {
        if (shader.words.empty() || shader.words.size() > kMaxRawShaderWords ||
            raw_words > kMaxShaderWords - shader.words.size()) {
            error = "raw shader data exceeds its bounded limit";
            return false;
        }
        if (shader.content_hash != shader_hash(shader.words)) {
            error = "raw shader content hash mismatch";
            return false;
        }
        raw_words += shader.words.size();
    }

    std::set<OperationIdentity> diagnosed;
    std::vector<bool> raw_referenced(capture.raw_shader_versions.size(), false);
    for (const auto& draw : capture.draws) {
        for (uint32_t index : {draw.vs_raw_shader_index, draw.fs_raw_shader_index,
                               draw.vs_chain_raw_shader_index}) {
            if (index == 0xFFFFFFFFu) continue;
            if (index >= capture.raw_shader_versions.size()) {
                error = "realized draw references an invalid raw shader";
                return false;
            }
            raw_referenced[index] = true;
        }
    }
    for (const auto& compute : capture.computes) {
        if (!validate_compute_recompile_state(compute, error)) return false;
        if (compute.raw_shader_index != 0xFFFFFFFFu &&
            compute.raw_shader_index >= capture.raw_shader_versions.size()) {
            error = "realized compute references an invalid raw shader";
            return false;
        }
        if (!validate_captured_null_guarded_raw_store(capture, compute, error))
            return false;
        if (!validate_captured_nullable_output_raw_buffer(capture, compute, error))
            return false;
        if (!validate_captured_gta5_cf9200_no_backing(capture, compute, error))
            return false;
        if (!validate_captured_gta5_packed_pointer(capture, compute, error))
            return false;
        if (!validate_captured_indirect_pointer_relocations(capture, compute, error))
            return false;
        if (compute.raw_shader_index != 0xFFFFFFFFu)
            raw_referenced[compute.raw_shader_index] = true;
    }
    for (const auto& diagnostic : capture.failure_diagnostics) {
        if (diagnostic.kind > SubmitOperationKind::Dispatch ||
            diagnostic.reason <= RealizationFailureReason::None ||
            diagnostic.reason > kMaxRealizationFailureReason ||
            diagnostic.stages.size() > kMaxFailureStages) {
            error = "invalid failed-operation diagnostic metadata";
            return false;
        }
        const OperationIdentity identity = operation_identity(
            diagnostic.kind, diagnostic.source_index, diagnostic.command_order);
        const auto operation = std::find_if(capture.operations.begin(), capture.operations.end(),
            [&](const auto& candidate) {
                return operation_identity(candidate.kind, candidate.source_index,
                                          candidate.command_order) == identity;
            });
        if (operation == capture.operations.end() || operation->realized) {
            error = "failed-operation diagnostic does not match an unrealized operation";
            return false;
        }
        if (!diagnosed.insert(identity).second) {
            error = "duplicate failed-operation diagnostic";
            return false;
        }
        // A hardware graphics stage can be assembled from more than one separately allocated
        // program.  In particular, AGC's NGG vertex path binds a prolog that transfers control to
        // the separately registered main shader.  Keep both raw streams in a failed-operation
        // diagnostic: treating the logical stage as unique made exactly the capture needed to
        // diagnose a linked-stage failure impossible.  The (kind,address) identity must still be
        // unique so an accidentally duplicated record remains fail-visible.
        std::set<std::pair<uint8_t, uint64_t>> stage_programs;
        for (const auto& stage : diagnostic.stages) {
            if (stage.stage > ShaderProgramStage::Compute ||
                (stage.raw_shader_index != 0xFFFFFFFFu &&
                 stage.raw_shader_index >= capture.raw_shader_versions.size()) ||
                stage.resource_count > kMaxResources ||
                stage.coverage.total > kMaxRawShaderWords ||
                stage.coverage.alu > stage.coverage.total ||
                stage.coverage.exports > stage.coverage.total ||
                stage.coverage.table_dependent > stage.coverage.total ||
                stage.coverage.unsupported > stage.coverage.total ||
                (!stage.resource_table_present && stage.resource_count != 0) ||
                (!stage.program_addr && stage.raw_shader_index != 0xFFFFFFFFu) ||
                !stage_programs.emplace(static_cast<uint8_t>(stage.stage),
                                        stage.program_addr).second) {
                error = "invalid failed-stage diagnostic metadata";
                return false;
            }
            // Rewriting a v7-v34 capture cannot invent tables that the old file reduced to a
            // presence/count summary. Preserve that summary with an absent v35 table so retry
            // reports "capture predates v35" explicitly. Any table that is actually retained must
            // agree exactly with the summary.
            if ((stage.resource_table.present || !stage.resource_table.resources.empty()) &&
                (stage.resource_table.present != stage.resource_table_present ||
                 stage.resource_table.resources.size() != stage.resource_count)) {
                error = "failed-stage resource table disagrees with its diagnostic summary";
                return false;
            }
            if (stage.recompile_config_available &&
                (diagnostic.kind != SubmitOperationKind::Dispatch ||
                 stage.stage != ShaderProgramStage::Compute ||
                 !validate_compute_config(stage.recompile_config, diagnostic.compute_launch,
                                          stage.recompile_config.native_subgroup_size,
                                          "invalid failed-compute recompile state", error))) {
                if (error.empty()) error = "invalid failed-compute recompile state";
                return false;
            }
            if (stage.raw_shader_index != 0xFFFFFFFFu)
                raw_referenced[stage.raw_shader_index] = true;
            const uint32_t max_issue = static_cast<uint32_t>(DescriptorIssueCode::UnusedRuntimeBinding);
            if ((!stage.descriptor_issue_count && stage.first_descriptor_issue != 0xFFFFFFFFu) ||
                (stage.descriptor_issue_count && stage.first_descriptor_issue > max_issue)) {
                error = "invalid failed-stage descriptor diagnostic";
                return false;
            }
        }
    }
    for (const auto& operation : capture.operations) {
        if (!operation.realized && diagnosed.count(operation_identity(
                operation.kind, operation.source_index, operation.command_order)) == 0) {
            error = "unrealized operation is missing its failure diagnostic";
            return false;
        }
    }
    if (std::find(raw_referenced.begin(), raw_referenced.end(), false) != raw_referenced.end()) {
        error = "raw shader is not referenced";
        return false;
    }
    return true;
}

// ---- appended by a later promotion out of the same source ----
inline CaptureRttSeedReader g_rtt_seed_reader;

inline size_t read_capture_guest_memory(uint64_t addr, uint8_t* dst, size_t bytes) {
#if defined(__linux__) || defined(__APPLE__)
    size_t done = 0;
    while (done < bytes) {
        iovec local{dst + done, bytes - done};
        iovec remote{reinterpret_cast<void*>(static_cast<uintptr_t>(addr + done)), bytes - done};
        const ssize_t read = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
        if (read < 0 && errno == EINTR) continue;
        if (read <= 0) break;
        done += static_cast<size_t>(read);
    }
    return done;
#else
    size_t done = 0;
    constexpr size_t chunk_max = 0x10000;
    while (done < bytes) {
        const size_t n = std::min(bytes - done, chunk_max);
        if (!guest_readable(addr + done, static_cast<uint32_t>(n))) break;
        std::memcpy(dst + done, reinterpret_cast<const void*>(static_cast<uintptr_t>(addr + done)), n);
        done += n;
    }
    return done;
#endif
}

inline bool is_capture_authority_resource(const ShaderResource& resource) {
    return is_compute_internal_gds(resource) ||
           is_gta5_packed_pointer_marker_candidate(resource) ||
           is_indirect_pointer_relocation_marker_candidate(resource) ||
           is_nullable_raw_buffer_marker_candidate(resource) ||
           is_gta5_selected_sbuffer_marker_candidate(resource) ||
           is_gta5_cf9200_no_backing_marker_candidate(resource);
}

inline CaptureMemoryReader ordered_gpustate_capture_reader(const GpuState& state) {
    return [&state](uint64_t addr, uint8_t* destination, size_t bytes) -> size_t {
        size_t copied = read_capture_guest_memory(addr, destination, bytes);
        if (!bytes || addr > std::numeric_limits<uint64_t>::max() - bytes) return copied;
        const uint64_t end = addr + bytes;
        for (const auto& copy : state.dma_copies) {
            std::vector<uint8_t> source;
            if (read_live_render_target_bytes(copy.src, copy.bytes, source) !=
                    LiveTargetByteReadResult::Success || source.size() != copy.bytes)
                continue;
            const uint64_t copy_end = copy.src + copy.bytes;
            const uint64_t overlap_begin = std::max(addr, copy.src);
            const uint64_t overlap_end = std::min(end, copy_end);
            if (overlap_begin >= overlap_end) continue;
            const size_t destination_offset = static_cast<size_t>(overlap_begin - addr);
            const size_t source_offset = static_cast<size_t>(overlap_begin - copy.src);
            const size_t overlap_bytes = static_cast<size_t>(overlap_end - overlap_begin);
            std::memcpy(destination + destination_offset, source.data() + source_offset,
                        overlap_bytes);
            copied = std::max(copied, destination_offset + overlap_bytes);
        }
        return copied;
    };
}

}  // namespace prosper::gpu
