// record_backend_timing_stats, append_rtt_timing, print_rtt_timing -- see backend_timing.hpp. Moved verbatim out of live_renderer.cpp (#3892).
#include "shared/live/submit_renderer/backend_timing.hpp"
#include "shared/live/submit_renderer/guest_reads.hpp"

namespace prosper::frontend::submit_renderer {

void record_backend_timing_stats(BackendTimingContext& ctx,
                                 const prosper::test::BackendRenderTimingStats& backend,
                                 const prosper::test::BackendTextureUploadStats& textures,
                                 const prosper::test::BackendPipelineCacheStats& pipelines,
                                 const prosper::test::BackendResourceReuseStats& reuse) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& pending_timing = ctx.pending_timing;
    auto& timing_mode = ctx.timing_mode;
    pending_timing.backend_calls += backend.calls;
    pending_timing.backend_draws += backend.draws;
    pending_timing.backend_command_buffers += backend.command_buffers;
    pending_timing.backend_queue_submits += backend.queue_submits;
    pending_timing.backend_fence_waits += backend.fence_waits;
    pending_timing.flush_no_batch += backend.flush_no_batch;
    pending_timing.flush_readback += backend.flush_readback;
    pending_timing.flush_storage_writeback += backend.flush_storage_writeback;
    pending_timing.flush_explicit += backend.flush_explicit;
    pending_timing.flush_cache_pressure += backend.flush_cache_pressure;
    pending_timing.cache_pressure_ms += backend.cache_pressure_ms;
    pending_timing.backend_gpu_timestamp_samples += backend.gpu_timestamp_samples;
    pending_timing.backend_target_ms += backend.target_ms;
    pending_timing.backend_draw_setup_ms += backend.draw_setup_ms;
    pending_timing.backend_record_upload_ms += backend.record_upload_ms;
    pending_timing.backend_gpu_wait_ms += backend.gpu_wait_ms;
    pending_timing.backend_gpu_device_ms += backend.gpu_device_ms;
    pending_timing.backend_readback_ms += backend.readback_ms;
    pending_timing.backend_cleanup_ms += backend.cleanup_ms;
    pending_timing.backend_setup_shader_ms += backend.setup_shader_ms;
    pending_timing.backend_setup_fixed_ms += backend.setup_fixed_ms;
    pending_timing.backend_res_fixed_index_upload_ms += backend.res_fixed_index_upload_ms;
    pending_timing.backend_res_fixed_blend_ms += backend.res_fixed_blend_ms;
    pending_timing.backend_res_fixed_depth_stencil_ms += backend.res_fixed_depth_stencil_ms;
    pending_timing.backend_res_fixed_viewport_ms += backend.res_fixed_viewport_ms;
    pending_timing.backend_res_fixed_stages_ms += backend.res_fixed_stages_ms;
    pending_timing.backend_res_fixed_prologue_ms += backend.res_fixed_prologue_ms;
    pending_timing.backend_res_prologue_subgroup_scan_ms += backend.res_prologue_subgroup_scan_ms;
    pending_timing.backend_setup_resources_ms += backend.setup_resources_ms;
    pending_timing.backend_res_texture_ms += backend.res_texture_ms;
    pending_timing.backend_res_texture_upload_ms += backend.res_texture_upload_ms;
    pending_timing.backend_res_texture_bind_ms += backend.res_texture_bind_ms;
    pending_timing.backend_texture_refs += textures.references;
    pending_timing.backend_texture_uploads += textures.unique_uploads;
    pending_timing.backend_texture_upload_bytes += textures.upload_bytes;
    pending_timing.backend_texture_persistent_hits += textures.persistent_hits;
    pending_timing.backend_texture_persistent_misses += textures.persistent_misses;
    pending_timing.backend_texture_binding_refs += reuse.texture_binding_references;
    pending_timing.backend_texture_binding_unique += reuse.unique_texture_bindings;
    pending_timing.backend_texture_binding_persistent_hits +=
        reuse.persistent_texture_binding_hits;
    pending_timing.backend_texture_binding_persistent_misses +=
        reuse.persistent_texture_binding_misses;
    pending_timing.backend_res_buffer_ms += backend.res_buffer_ms;
    pending_timing.backend_res_buffer_acquire_ms += backend.res_buffer_acquire_ms;
    pending_timing.backend_res_buffer_range_plan_ms += backend.res_buffer_range_plan_ms;
    pending_timing.backend_res_buffer_copy_ms += backend.res_buffer_copy_ms;
    pending_timing.backend_res_buffer_resident_ms += backend.res_buffer_resident_ms;
    pending_timing.backend_res_buffer_watch_ms += backend.res_buffer_watch_ms;
    pending_timing.buffer_range_uploads += reuse.buffer_range_uploads;
    pending_timing.buffer_range_bindings += reuse.buffer_range_bindings;
    pending_timing.buffer_range_upload_bytes += reuse.buffer_range_upload_bytes;
    pending_timing.buffer_range_bound_bytes += reuse.buffer_range_bound_bytes;
    pending_timing.buffer_upload_bytes += reuse.buffer_upload_bytes;
    pending_timing.buffer_resident_hits += reuse.buffer_resident_hits;
    pending_timing.buffer_resident_compared_bytes += reuse.buffer_resident_compared_bytes;
    pending_timing.buffer_resident_reused_bytes += reuse.buffer_resident_reused_bytes;
    pending_timing.buffer_resident_admitted_bytes += reuse.buffer_resident_admitted_bytes;
    pending_timing.buffer_resident_refreshed_bytes += reuse.buffer_resident_refreshed_bytes;
    pending_timing.buffer_resident_watched_bytes += reuse.buffer_resident_watched_bytes;
    pending_timing.buffer_resident_declined_bytes += reuse.buffer_resident_declined_bytes;
    pending_timing.buffer_resident_ineligible_bytes += reuse.buffer_resident_ineligible_bytes;
    pending_timing.backend_res_buffer_create_ms += backend.res_buffer_create_ms;
    pending_timing.backend_res_buffer_index_find_ms += backend.res_buffer_index_find_ms;
    pending_timing.backend_res_buffer_index_insert_ms += backend.res_buffer_index_insert_ms;
    pending_timing.backend_res_buffer_hash_ms += backend.res_buffer_hash_ms;
    pending_timing.backend_res_descriptor_ms += backend.res_descriptor_ms;
    pending_timing.backend_setup_pipeline_ms += backend.setup_pipeline_ms;
    pending_timing.backend_pipeline_refs += pipelines.references;
    pending_timing.backend_pipeline_hits += pipelines.hits;
    pending_timing.backend_pipeline_misses += pipelines.misses;
    pending_timing.backend_pipeline_bypasses += pipelines.bypasses;
    pending_timing.backend_pipeline_entries = pipelines.entries;
    pending_timing.backend_pipeline_evictions += pipelines.evictions;
    // Buffer-copy economics. This leaf dominates the Blue Prince FMV collapse -- 539 ms
    // of 802 ms of backend resource setup, 67% (#2215) -- and nothing could say WHY: the
    // counters are collected per call and aggregated into the thread-local storage, and
    // the ONLY consumers in the tree were assertions in test_multidraw_render.cpp. The
    // sibling resource kind has reported its cache economics for ages
    // ("[render-timing] texture_cache hits=... misses=..."), while buffers -- an order of
    // magnitude more expensive in that regime -- reported nothing at all.
    //
    // Per SUBMIT rather than as a window mean, deliberately. The question is which of
    // three mutually exclusive stories produces the copy volume, and they need different
    // fixes, so an average across them answers none of it:
    //   skipped_unique dominates -> buffer_identity == 0, nothing can dedup, fix upstream
    //   skipped_large dominates  -> the 4 KiB hash-dedup cap is working as designed and
    //                               the payloads are genuinely distinct; the fix is not
    //                               dedup at all but not staging through a mapped copy
    //   refs >> memo_hits        -> repeat references within one call are being missed
    if (timing_mode.log) {
        // CUMULATIVE, not a per-call snapshot. The first draft printed one call's
        // numbers at diag_should_print's ordinals (first 64, then powers of two) --
        // which is a UNIFORM sample of a heavily skewed distribution, so it could not
        // see the expensive calls by construction. The sampled calls summed to ~290 KB
        // per submit against 30 ms of measured copy: 290 KB cannot take 30 ms, and that
        // contradiction is the only reason the sampling flaw was caught.
        //
        // Totals cannot miss a call. The rate limit now controls how often the running
        // total is PRINTED, never which calls it counts.
        static uint64_t refs = 0, uniq = 0, memo = 0, hashed = 0, dwords = 0;
        static uint64_t skipped_large = 0, skipped_unique = 0, fallbacks = 0, calls = 0;
        static uint64_t large_dwords = 0, upload_bytes = 0;
        static PassBufferLookupStats lookup;
        refs += reuse.buffer_references;      uniq += reuse.unique_buffers;
        memo += reuse.buffer_ref_memo_hits;   hashed += reuse.buffer_hash_calls;
        dwords += reuse.buffer_hash_dwords;   fallbacks += reuse.buffer_upload_fallbacks;
        skipped_large += reuse.buffer_hash_skipped_large;
        skipped_unique += reuse.buffer_hash_skipped_unique;
        large_dwords += reuse.buffer_skipped_large_dwords;
        upload_bytes += reuse.buffer_upload_bytes;
        lookup.add(reuse.buffer_lookup);
        if (prosper::diag_should_print(++calls)) {
            fprintf(stderr,
                    "[render-timing] buffer_reuse (cumulative over %llu backend calls) "
                    "refs=%llu unique=%llu memo_hits=%llu hashed=%llu (%.1f MiB) "
                    "skipped_large=%llu (%.1f MiB) skipped_unique=%llu "
                    "COPIED=%.1f MiB upload_fallbacks=%llu\n",
                    (unsigned long long)calls, (unsigned long long)refs,
                    (unsigned long long)uniq, (unsigned long long)memo,
                    (unsigned long long)hashed, (double)dwords * 4.0 / (1024.0 * 1024.0),
                    (unsigned long long)skipped_large,
                    (double)large_dwords * 4.0 / (1024.0 * 1024.0),
                    (unsigned long long)skipped_unique,
                    (double)upload_bytes / (1024.0 * 1024.0),
                    (unsigned long long)fallbacks);
            fprintf(stderr,
                    "[render-timing] buffer_lookup (cumulative over %llu backend calls) "
                    "observed_passes=%llu arena_passes=%llu allocations=%llu "
                    "allocation_bytes=%llu deallocations=%llu deallocation_bytes=%llu\n",
                    (unsigned long long)calls,
                    (unsigned long long)lookup.observed_passes,
                    (unsigned long long)lookup.arena_passes,
                    (unsigned long long)lookup.allocations,
                    (unsigned long long)lookup.allocation_bytes,
                    (unsigned long long)lookup.deallocations,
                    (unsigned long long)lookup.deallocation_bytes);
        }
    }
}

void append_rtt_timing(std::string& output, const RttTimingRecord& record) {
    const prosper::test::BackendRenderTimingStats& timing = record.timing;
    const double detail_ms = timing.total_ms();
    char line[512];
    const int length = snprintf(
        line, sizeof(line),
        "[rtt-timing] submit=%d target=0x%llx extent=%ux%u draws=%zu "
        "span=%d/%d authoritative=%d deferred=%d cmd=%llu submit=%llu wait=%llu "
        "measured=%.2f detail=%.2f other=%.2f target_setup=%.2f "
        "draw_setup=%.2f record_upload=%.2f batch_wait=%.2f "
        "batch_device=%.2f batch_overhead=%.2f batch_timestamps=%llu "
        "readback=%.2f cleanup=%.2f gpu_target=%llu load=%llu sample=%llu cpu=%llu\n",
        record.submit, (unsigned long long)record.target,
        record.width, record.height, record.draws,
        record.first_span, record.final_span, record.authoritative_readback,
        record.deferred_readback,
        (unsigned long long)timing.command_buffers,
        (unsigned long long)timing.queue_submits,
        (unsigned long long)timing.fence_waits,
        record.measured_ms, detail_ms,
        record.measured_ms - detail_ms,
        timing.target_ms, timing.draw_setup_ms, timing.record_upload_ms,
        timing.gpu_wait_ms, timing.gpu_device_ms,
        std::max(0.0, timing.gpu_wait_ms - timing.gpu_device_ms),
        (unsigned long long)timing.gpu_timestamp_samples,
        timing.readback_ms, timing.cleanup_ms,
        (unsigned long long)record.color_target.writes,
        (unsigned long long)record.color_target.write_hits,
        (unsigned long long)record.color_target.sampled_hits,
        (unsigned long long)record.color_target.readbacks);
    if (length > 0)
        output.append(line, std::min<size_t>(length, sizeof(line) - 1));
}

void print_rtt_timing(const RttTimingRecord& record) {
    std::string output;
    append_rtt_timing(output, record);
    if (!output.empty()) fwrite(output.data(), 1, output.size(), stderr);
}

} // namespace prosper::frontend::submit_renderer
