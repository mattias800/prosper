#pragma once
// Types shared by the submit-renderer callback (register_live_renderer in live_renderer.cpp) and the
// modules carved out of it (#3892). Each was a local type inside that callback; they moved here
// verbatim, unchanged, so that a function outside the callback can name them.
//
// Nothing here owns state. callback_state.hpp owns the process prelude; callback_thread_state.hpp
// owns the calling-thread prelude. The callback keeps the original lazy bindings. Only TYPES live
// here.

#include "gpu/resources/shader_resources.hpp"      // ResourceClass, DescriptorValidationReport
#include "shared/texture/validation_census.hpp"     // TextureValidationCensus
#include "shared/live/buffer_source_gate.hpp"       // BufferSourceGateCounters
#include "diagnostics/perf/perf_ledger.hpp"         // DropReason
#include "fixtures/render_runner.h"                 // FrameResource, backend timing stats
#include "fixtures/retained_depth_array_gpu.h"      // PersistentDsDepthArrayGpuImage
#include "fixtures/retained_depth_cube_gpu.h"       // PersistentDsDepthCubeGpuImage

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace prosper::frontend::submit_renderer {

using RC = prosper::gpu::ResourceClass;
struct PinnedScanout {
    uint64_t id = 0;
    uint32_t width = 0, height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
};
struct PinnedRendererMipTarget {
    uint64_t id = 0;
    uint32_t width = 0, height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    // A mipmapped T# has named this exact independently rendered level. Keep the pin
    // through the backend call that consumes it, then release it at the logical
    // submit's final span. Unconsumed producers deliberately cross submit boundaries.
    bool consumed = false;
};
using RenderClock = std::chrono::steady_clock;
struct RenderTiming {
    uint64_t callbacks = 0;
    double total_ms = 0, prelude_ms = 0, pass_ms = 0;
    double build_resources_ms = 0, backend_ms = 0, output_copy_ms = 0;
    double dcc_materialize_ms = 0;
    uint64_t dcc_materialize_surfaces = 0, dcc_materialize_bytes = 0;
    uint64_t backend_calls = 0, backend_draws = 0;
    uint64_t backend_command_buffers = 0, backend_queue_submits = 0;
    uint64_t backend_fence_waits = 0;
    // Why each of the ~16 flushes a submit happened (#2276). Exclusive and summing to
    // the flush count, so the arithmetic is checkable from the printed line alone.
    uint64_t flush_no_batch = 0, flush_readback = 0;
    uint64_t flush_storage_writeback = 0, flush_explicit = 0;
    uint64_t flush_cache_pressure = 0;
    double cache_pressure_ms = 0;
    uint64_t backend_gpu_timestamp_samples = 0;
    double backend_target_ms = 0, backend_draw_setup_ms = 0;
    double backend_record_upload_ms = 0, backend_gpu_wait_ms = 0;
    double backend_gpu_device_ms = 0;
    double backend_readback_ms = 0, backend_cleanup_ms = 0;
    double backend_setup_shader_ms = 0, backend_setup_fixed_ms = 0;
    double backend_res_fixed_index_upload_ms = 0;
    double backend_res_fixed_blend_ms = 0;
    double backend_res_fixed_depth_stencil_ms = 0;
    double backend_res_fixed_viewport_ms = 0;
    double backend_res_fixed_stages_ms = 0;
    double backend_res_fixed_prologue_ms = 0;
    double backend_res_prologue_subgroup_scan_ms = 0;
    double backend_setup_resources_ms = 0, backend_setup_pipeline_ms = 0;
    // #1284: sub-attribution of backend_setup_resources_ms. res_texture/res_buffer
    // cover the whole per-resource branch; the upload/bind pair is nested inside
    // res_texture and covers only cache-MISS work, so the remainder is what a cache
    // hit still pays per reference. res_descriptor is the per-draw set alloc/update.
    double backend_res_texture_ms = 0, backend_res_texture_upload_ms = 0;
    double backend_res_texture_bind_ms = 0, backend_res_buffer_ms = 0;
    uint64_t backend_texture_refs = 0, backend_texture_uploads = 0;
    uint64_t backend_texture_upload_bytes = 0;
    uint64_t backend_texture_persistent_hits = 0, backend_texture_persistent_misses = 0;
    uint64_t backend_texture_binding_refs = 0, backend_texture_binding_unique = 0;
    uint64_t backend_texture_binding_persistent_hits = 0;
    uint64_t backend_texture_binding_persistent_misses = 0;
    double backend_res_buffer_range_plan_ms = 0;
    double backend_res_buffer_acquire_ms = 0, backend_res_buffer_copy_ms = 0;
    double backend_res_buffer_resident_ms = 0;
    double backend_res_buffer_watch_ms = 0;
    uint64_t gpu_detile_preparations = 0, gpu_detile_2d_preparations = 0;
    uint64_t gpu_detile_source_bytes = 0;
    uint64_t buffer_range_uploads = 0;
    uint64_t buffer_range_bindings = 0;
    uint64_t buffer_range_upload_bytes = 0;
    uint64_t buffer_range_bound_bytes = 0;
    uint64_t buffer_upload_bytes = 0;
    uint64_t buffer_resident_hits = 0;
    uint64_t buffer_resident_compared_bytes = 0;
    uint64_t buffer_resident_reused_bytes = 0;
    uint64_t buffer_resident_admitted_bytes = 0;
    uint64_t buffer_resident_refreshed_bytes = 0;
    uint64_t buffer_resident_watched_bytes = 0;
    uint64_t buffer_resident_declined_bytes = 0;
    uint64_t buffer_resident_ineligible_bytes = 0;
    double backend_res_buffer_create_ms = 0;
    double backend_res_buffer_index_find_ms = 0;
    double backend_res_buffer_index_insert_ms = 0;
    double backend_res_buffer_hash_ms = 0;
    double backend_res_descriptor_ms = 0;
    uint64_t backend_pipeline_refs = 0, backend_pipeline_hits = 0;
    uint64_t backend_pipeline_misses = 0, backend_pipeline_bypasses = 0;
    uint64_t backend_pipeline_entries = 0, backend_pipeline_evictions = 0;
    uint64_t color_target_writes = 0, color_target_write_hits = 0;
    uint64_t color_target_sample_hits = 0, color_target_readbacks = 0;
    uint64_t color_target_cached_bytes = 0, color_target_cached_entries = 0;
    uint64_t textures = 0, texture_reuses = 0, buffers = 0, buffer_views = 0;
    uint64_t compact_buffer_resources = 0, full_buffer_resources = 0;
    BufferSourceGateCounters buffer_source_gate;
    uint64_t persistent_hits = 0, persistent_misses = 0, persistent_invalidations = 0;
    uint64_t persistent_submit_reuses = 0, persistent_validations = 0;
    uint64_t persistent_validation_bytes = 0;
    uint64_t persistent_watch_reuses = 0, persistent_watch_dirty = 0;
    uint64_t persistent_watch_unknown = 0, persistent_watch_disabled = 0;
    // WHY in-submit reuse never fires (#2289). persistent_submit_reuses is 0 across
    // 55,820 hits on PPSA25009, and the two ways that happens have OPPOSITE prospects:
    // Unknown means prosper is not tracking guest writes on this path at all (fixable,
    // and worth ~9x, since the same texture is referenced 9.5 times per submit), while
    // Overlap means the guest genuinely rewrote those bytes and the revalidation is
    // CORRECT. A single zero cannot tell those apart, so count them separately.
    uint64_t persistent_submit_unknown = 0, persistent_submit_overlap = 0;
    // ...and WHY the Unknowns were unknown (#2289). Two structurally different
    // causes that a single Unknown count cannot separate, and the difference
    // decides whether the cost is addressable by the submit journal AT ALL:
    //   unarmed  the journal is not armed on this thread -> missing instrumentation
    //   stale    it IS armed, but the snapshot came from an EARLIER submit. That is
    //            not a defect: the journal is intra-submit by construction, so a
    //            cross-submit question can only ever answer Unknown through it,
    //            and only a cross-submit WATCH can answer it at all.
    uint64_t persistent_submit_unarmed = 0, persistent_submit_stale = 0;
    // #2283's blocking arm. publish_selected_fmt0 must stay 0 over a full route for
    // the disabled-target readback skip to be safe by construction. `unknown` is
    // separate so a selection whose source pass could not be identified is never
    // silently counted as a clean one.
    uint64_t publish_selected = 0, publish_selected_fmt0 = 0;
    uint64_t publish_selected_unknown = 0, publish_candidate_fmt0 = 0;
    uint64_t texture_bytes = 0, buffer_bytes = 0, buffer_materialized_bytes = 0;
    double texture_ms = 0, buffer_ms = 0;
    // Attribution of texture_ms by OUTCOME class (#2262). After #2259 removed the
    // reflection cost this is the largest leaf of build_R -- 22.56 ms/submit on a Blue
    // Prince gameplay window, 55% of build_R -- and it is one span with nothing inside
    // it, so "4,449 references at an 89% reuse rate cost 22.56 ms" is as far as anyone
    // can currently get.
    //
    // An attribution rather than a phase partition, deliberately: these classes are
    // already computed as locals on the per-resource path and are decided in a fixed
    // order, so classifying in that same order makes them mutually exclusive by
    // construction -- whereas carving a 2,800-line branch into timed phases invites the
    // double-counting #2246 and #2250 had to be careful about.
    double tex_rtt_ms = 0, tex_compute_ms = 0, tex_local_ms = 0;
    double tex_persist_hit_ms = 0, tex_persist_reuse_ms = 0, tex_persist_miss_ms = 0;
    uint64_t tex_rtt_n = 0, tex_compute_n = 0, tex_local_n = 0;
    uint64_t tex_persist_hit_n = 0, tex_persist_reuse_n = 0, tex_persist_miss_n = 0;
    // The cache entry was FOUND and its guest bytes had changed, so the decode ran
    // again. It had no bucket of its own, which put it in `other` alongside "this
    // resource was never cacheable at all" -- two states with opposite fixes. On Stray's
    // title screen 930.78 of the texture leaf's 1110.10 ms landed in that undivided
    // residual, and naming this half is what identified the two 63.75 MiB RGBA16F HDR
    // intermediates that re-decode every frame (#3149 measured the splash, not this).
    // Snapshot handoff is nested inside frontend texture materialization.
    double tex_source_snapshot_handoff_ms = 0;
    uint64_t tex_source_snapshot_copied_bytes = 0;
    uint64_t tex_source_snapshot_transferred_bytes = 0;
    double tex_persist_invalid_ms = 0;
    uint64_t tex_persist_invalid_n = 0;
    uint64_t tex_other_n = 0;
    double tex_other_slowest_ms = 0;
    uint64_t tex_other_addr = 0, tex_other_source_bytes = 0;
    uint32_t tex_other_width = 0, tex_other_height = 0, tex_other_depth = 0;
    uint32_t tex_other_format = 0, tex_other_components = 0;
    uint32_t tex_other_tile_mode = 0, tex_other_img_dim = 0, tex_other_class = 0;
    bool tex_other_compute_candidate = false;
    bool tex_other_persistent_candidate = false;
    bool tex_other_compressed = false, tex_other_depth_compare = false;
    bool tex_other_host_backed = false;
    // Exhaustive partition of build_resources_ms, the frontend materialiser (#2215).
    // It had two named leaves -- texture_ms and buffer_ms, both from INSIDE build_R --
    // covering 10.2 of 48.95 ms on a Blue Prince gameplay submit, with no assertion
    // that they were exhaustive. 38.76 ms, 21.1% of the frame, was attributable to
    // nothing. That is not the same as being spent on nothing, and it is
    // indistinguishable from it in a report.
    //
    // These four plus a SIGNED residual cover the per-draw body of build_bds. The
    // residual is published rather than clamped for the reason #2246 records: an
    // unmeasured region does not read as unmeasured, it reads as someone else's number.
    // pass_control is a RESIDUAL, not a measurement: pass_ms - build_resources_ms -
    // backend_ms. On a Blue Prince gameplay window it is 85.57 ms/submit -- 27% of the
    // frame -- attributed to nothing, which is the state build_resources was in before
    // #2250: an unmeasured region does not read as unmeasured, it reads as somebody
    // else's cost.
    //
    // Two spans bound the per-group work either side of the measured pair. They are
    // accumulated where a `continue` cannot skip past them silently: `pre` at the same
    // place build_resources_ms is recorded, `post` at the loop's end. A group that
    // continues before reaching the build contributes to neither and lands in the
    // residual, which is correct -- and visible, rather than folded into a leaf.
    double pass_pre_ms = 0, pass_post_ms = 0;
    double post_stats_ms = 0, post_slot0_ms = 0, post_mrt_ms = 0,
           post_rest_ms = 0;
    // pre+post measured 0.23 ms of a 43.85 ms pass_control, so the cost is NOT the
    // per-group work either side of the measured pair. head/tail bound the two regions
    // outside the group loop. `tail` subtracts the build_resources+backend the tail
    // itself performs -- those are already excluded from pass_control, so counting them
    // here would make the leaves exceed the parent.
    double pass_head_ms = 0, pass_tail_ms = 0;
    // head+pre+post+tail measured 1.94 ms of 60.15, so the cost is in none of them.
    // The only region left is the group loop itself -- specifically iterations that
    // return before reaching the render, whose time no span above can see. `loop_ms`
    // is the whole loop; subtracting pre+post and the build/backend the loop performed
    // leaves exactly the early-exit time, without having to instrument every `continue`
    // (there are several, and one missed would under-report silently).
    double pass_loop_ms = 0;
    // The group loop has exactly ONE early exit -- the MSAA resolve at :5520 -- so the
    // 49.4 ms difference between `loop` and pre+post+build+backend is entirely resolve
    // work. Split into the three things a resolve does, because they have completely
    // different fixes: `stall` is submit_and_wait (a full GPU pipeline flush), `read` is
    // a whole-surface GPU->CPU readback, `copy` is the GPU-side destination copy that
    // already exists and is presumably cheap.
    double resolve_stall_ms = 0, resolve_read_ms = 0, resolve_copy_ms = 0;
    // The GPU copy's OWN submit_and_wait (#1382 requires it). Separate from `stall`
    // because removing the eager readback relocated cost into it rather than
    // eliminating it -- the copy's flush now drains a queue the eager flush used to.
    // Leaving it unmeasured would put an unmeasured region back inside `loop`, which is
    // the exact defect this partition exists to remove.
    double resolve_copy_stall_ms = 0;
    uint64_t resolve_n = 0, resolve_read_n = 0, resolve_bytes = 0;
    uint64_t pass_groups_seen = 0;
    uint64_t pass_groups = 0;
    double build_r_ms = 0, build_validate_ms = 0;
    double build_poison_ms = 0, build_indices_ms = 0;
    uint64_t build_draws = 0, build_rejected = 0;
    // One level deeper, into build_R's own residual (#2256). `identities` is a SET and
    // not a counter on purpose: its size is how many distinct shaders a submit-scoped
    // memo would have to hold, which is the ceiling on what such a memo could save.
    // It is cleared per submit with the rest of this struct.
    double build_reflect_ms = 0;
    uint64_t build_reflect_calls = 0, build_reflect_words = 0;
    uint64_t build_reflect_unidentified = 0;
    uint64_t build_reflect_memo_hits = 0, build_reflect_memo_clears = 0;
    uint64_t build_reflect_memo_mismatch = 0;
    std::unordered_set<uint64_t> build_reflect_identities;
};
struct RttTimingRecord {
    int submit = 0;
    uint64_t target = 0;
    uint32_t width = 0, height = 0;
    size_t draws = 0;
    bool first_span = false, final_span = false;
    bool authoritative_readback = false, deferred_readback = false;
    double measured_ms = 0;
    prosper::test::BackendRenderTimingStats timing;
    prosper::test::BackendColorTargetStats color_target;
};
struct ValidationCensusLog {
    prosper::frontend::TextureValidationCensus data;
    unsigned long thread = 0;
    bool active = false;
    ~ValidationCensusLog() {
        if (active) data.report(stderr, thread, "thread-exit");
    }
};
struct ReflectMemoEntry {
    prosper::gpu::DescriptorValidationReport report;
    uint32_t set = 0;
    prosper::gpu::SpirvShaderStage stage = prosper::gpu::SpirvShaderStage::Unknown;
};
constexpr size_t kReflectMemoMaxEntries = 4096;
using prosper::diagnostics::perf::DropReason;
struct BuiltFrameResources {
    bool complete = true;
    // #3891 phase 3: which site rejected the draw. The FIRST site to reject names it
    // (later bindings of the same draw can reject too, but the draw drops once).
    DropReason drop_reason = DropReason::Unattributed;
    void reject(DropReason why) {
        if (complete) drop_reason = why;
        complete = false;
    }
    std::vector<prosper::test::FrameResource> full;
    std::vector<prosper::test::FrameBufferResource> buffers;
    // High bit selects buffers; remaining bits index the selected vector. Present only
    // for the split representation, so the compatibility arm remains byte-for-byte in
    // the resource order it used before.
    std::vector<uint32_t> order;
};
constexpr uint32_t kCompactBufferResourceBit = 0x80000000u;
using DepthPlaneIdentity = std::tuple<
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint32_t, uint32_t, uint32_t, uint32_t, VkImage, uint64_t, bool, bool>;
struct DepthArraySnapshot {
    uint64_t base, stride;
    uint32_t width, height, layers;
    VkFormat format;
    std::vector<DepthPlaneIdentity> planes;
    std::shared_ptr<const std::vector<uint8_t>> pixels;
    // GPU-resident alternative to `pixels` (exactly one is set). Its image is filled by
    // a command buffer queued in this callback's ordered batch; see
    // tests/fixtures/retained_depth_array_gpu.h.
    std::shared_ptr<prosper::test::PersistentDsDepthArrayGpuImage> gpu;
};
struct DepthArraySnapshotCensus {
    bool enabled;
    uint64_t reads = 0, reuses = 0, payload_bytes = 0, producer_flushes = 0;
    ~DepthArraySnapshotCensus() {
        if (enabled && (reads || reuses))
            std::fprintf(stderr,
                "[depth-array-snapshots] scope=callback reads=%llu reuses=%llu "
                "payload_bytes=%llu producer_flushes=%llu\n",
                (unsigned long long)reads, (unsigned long long)reuses,
                (unsigned long long)payload_bytes, (unsigned long long)producer_flushes);
    }
};
constexpr size_t kDepthArraySnapshotEntries = 16;
constexpr size_t kDepthArraySnapshotBytes = 128u * 1024u * 1024u;
struct DepthCubeGpuSnapshot {
    uint64_t base;
    uint32_t width, height;
    std::shared_ptr<prosper::test::PersistentDsDepthCubeGpuImage> gpu;
};
constexpr size_t kDepthCubeGpuSnapshotEntries = 8;

constexpr uint32_t kNoPassFormat = UINT32_MAX;
// What later passes of the same submit do with a pass's target (render_per_target_passes).
struct LaterTargetConsumers {
    bool sampled_exact = false;
    bool feedback = false;
    bool cpu_needed = false;
    uint32_t storage_references = 0;
    uint32_t dimension_mismatches = 0;
    uint32_t extent_mismatches = 0;
    uint32_t feedback_references = 0;
};
} // namespace prosper::frontend::submit_renderer
