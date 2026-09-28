// report_render_timing_aggregates -- see timing_report.hpp. Moved verbatim out of live_renderer.cpp (#3892).
#include "shared/live/submit_renderer/timing_report.hpp"
#include "shared/live/submit_renderer/guest_reads.hpp"

namespace prosper::frontend::submit_renderer {

void report_render_timing_aggregates(RenderTimingReportContext& ctx) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& g_rtt = ctx.g_rtt;
    auto& pending_timing = ctx.pending_timing;
    auto& texstore = ctx.texstore;
    auto& submit_decode_scope_disabled = ctx.submit_decode_scope_disabled;
    auto& persistent_decoded_textures = ctx.persistent_decoded_textures;
    auto& persistent_decoded_texture_bytes = ctx.persistent_decoded_texture_bytes;
    auto& persistent_validation_scratch = ctx.persistent_validation_scratch;
    struct TimingTotals {
        uint64_t submits = 0, callbacks = 0;
        double total_ms = 0, prelude_ms = 0, pass_ms = 0;
        double build_resources_ms = 0, backend_ms = 0, output_copy_ms = 0;
        double dcc_materialize_ms = 0;
        uint64_t dcc_materialize_surfaces = 0, dcc_materialize_bytes = 0;
        uint64_t backend_calls = 0, backend_draws = 0;
        uint64_t backend_command_buffers = 0, backend_queue_submits = 0;
        uint64_t backend_fence_waits = 0;
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
        double backend_res_texture_ms = 0, backend_res_texture_upload_ms = 0;
        double backend_res_texture_bind_ms = 0, backend_res_buffer_ms = 0;
        double backend_res_buffer_range_plan_ms = 0;
        double backend_res_buffer_acquire_ms = 0, backend_res_buffer_copy_ms = 0;
        double backend_res_buffer_resident_ms = 0;
        double backend_res_buffer_watch_ms = 0;
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
        uint64_t persistent_submit_unknown = 0, persistent_submit_overlap = 0;
        uint64_t persistent_submit_unarmed = 0, persistent_submit_stale = 0;
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
        double tex_persist_invalid_ms = 0;
        uint64_t tex_persist_invalid_n = 0;
        uint64_t tex_rtt_n = 0, tex_compute_n = 0, tex_local_n = 0;
        uint64_t tex_persist_hit_n = 0, tex_persist_reuse_n = 0, tex_persist_miss_n = 0;
        uint64_t tex_other_n = 0;
        double pass_pre_ms = 0, pass_post_ms = 0;
        double pass_head_ms = 0, pass_tail_ms = 0;
        double pass_loop_ms = 0;
        double resolve_stall_ms = 0, resolve_read_ms = 0, resolve_copy_ms = 0;
        double resolve_copy_stall_ms = 0;
        uint64_t resolve_n = 0, resolve_read_n = 0, resolve_bytes = 0;
        uint64_t pass_groups_seen = 0;
        uint64_t pass_groups = 0;
        double build_r_ms = 0, build_validate_ms = 0;
        double build_poison_ms = 0, build_indices_ms = 0;
        uint64_t build_draws = 0, build_rejected = 0;
        // A COUNT here, not a set: the aggregate sums each submit's distinct count so
        // the report can divide by submits. A lifetime union would answer a different
        // question (how many shaders the title has) and would over-state the ceiling
        // for a memo whose scope is one submit.
        double build_reflect_ms = 0;
        uint64_t build_reflect_calls = 0, build_reflect_words = 0;
        uint64_t build_reflect_unidentified = 0, build_reflect_distinct = 0;
        uint64_t build_reflect_memo_hits = 0, build_reflect_memo_clears = 0;
        uint64_t build_reflect_memo_mismatch = 0;
    };
    static TimingTotals totals;
    static TimingTotals window;
    auto accumulate = [&](TimingTotals& timing) {
        timing.submits++;
        timing.callbacks += pending_timing.callbacks;
        timing.total_ms += pending_timing.total_ms;
        timing.prelude_ms += pending_timing.prelude_ms;
        timing.pass_ms += pending_timing.pass_ms;
        timing.build_resources_ms += pending_timing.build_resources_ms;
        timing.backend_ms += pending_timing.backend_ms;
        timing.output_copy_ms += pending_timing.output_copy_ms;
        timing.dcc_materialize_ms += pending_timing.dcc_materialize_ms;
        timing.dcc_materialize_surfaces += pending_timing.dcc_materialize_surfaces;
        timing.dcc_materialize_bytes += pending_timing.dcc_materialize_bytes;
        timing.backend_calls += pending_timing.backend_calls;
        timing.backend_draws += pending_timing.backend_draws;
        timing.backend_command_buffers += pending_timing.backend_command_buffers;
        timing.backend_queue_submits += pending_timing.backend_queue_submits;
        timing.backend_fence_waits += pending_timing.backend_fence_waits;
        timing.flush_no_batch += pending_timing.flush_no_batch;
        timing.flush_readback += pending_timing.flush_readback;
        timing.flush_storage_writeback += pending_timing.flush_storage_writeback;
        timing.flush_explicit += pending_timing.flush_explicit;
        timing.flush_cache_pressure += pending_timing.flush_cache_pressure;
        timing.cache_pressure_ms += pending_timing.cache_pressure_ms;
        timing.backend_gpu_timestamp_samples += pending_timing.backend_gpu_timestamp_samples;
        timing.backend_target_ms += pending_timing.backend_target_ms;
        timing.backend_draw_setup_ms += pending_timing.backend_draw_setup_ms;
        timing.backend_record_upload_ms += pending_timing.backend_record_upload_ms;
        timing.backend_gpu_wait_ms += pending_timing.backend_gpu_wait_ms;
        timing.backend_gpu_device_ms += pending_timing.backend_gpu_device_ms;
        timing.backend_readback_ms += pending_timing.backend_readback_ms;
        timing.backend_cleanup_ms += pending_timing.backend_cleanup_ms;
        timing.backend_setup_shader_ms += pending_timing.backend_setup_shader_ms;
        timing.backend_setup_fixed_ms += pending_timing.backend_setup_fixed_ms;
        timing.backend_res_fixed_index_upload_ms += pending_timing.backend_res_fixed_index_upload_ms;
        timing.backend_res_fixed_blend_ms += pending_timing.backend_res_fixed_blend_ms;
        timing.backend_res_fixed_depth_stencil_ms += pending_timing.backend_res_fixed_depth_stencil_ms;
        timing.backend_res_fixed_viewport_ms += pending_timing.backend_res_fixed_viewport_ms;
        timing.backend_res_fixed_stages_ms += pending_timing.backend_res_fixed_stages_ms;
        timing.backend_res_fixed_prologue_ms += pending_timing.backend_res_fixed_prologue_ms;
        timing.backend_res_prologue_subgroup_scan_ms += pending_timing.backend_res_prologue_subgroup_scan_ms;
        timing.backend_setup_resources_ms += pending_timing.backend_setup_resources_ms;
        timing.backend_res_texture_ms += pending_timing.backend_res_texture_ms;
        timing.backend_res_texture_upload_ms += pending_timing.backend_res_texture_upload_ms;
        timing.backend_res_texture_bind_ms += pending_timing.backend_res_texture_bind_ms;
        timing.backend_res_buffer_ms += pending_timing.backend_res_buffer_ms;
        timing.backend_res_buffer_acquire_ms += pending_timing.backend_res_buffer_acquire_ms;
        timing.backend_res_buffer_range_plan_ms += pending_timing.backend_res_buffer_range_plan_ms;
        timing.backend_res_buffer_copy_ms += pending_timing.backend_res_buffer_copy_ms;
        timing.backend_res_buffer_resident_ms += pending_timing.backend_res_buffer_resident_ms;
        timing.backend_res_buffer_watch_ms += pending_timing.backend_res_buffer_watch_ms;
        timing.buffer_range_uploads += pending_timing.buffer_range_uploads;
        timing.buffer_range_bindings += pending_timing.buffer_range_bindings;
        timing.buffer_range_upload_bytes += pending_timing.buffer_range_upload_bytes;
        timing.buffer_range_bound_bytes += pending_timing.buffer_range_bound_bytes;
        timing.buffer_upload_bytes += pending_timing.buffer_upload_bytes;
        timing.buffer_resident_hits += pending_timing.buffer_resident_hits;
        timing.buffer_resident_compared_bytes += pending_timing.buffer_resident_compared_bytes;
        timing.buffer_resident_reused_bytes += pending_timing.buffer_resident_reused_bytes;
        timing.buffer_resident_admitted_bytes += pending_timing.buffer_resident_admitted_bytes;
        timing.buffer_resident_refreshed_bytes += pending_timing.buffer_resident_refreshed_bytes;
        timing.buffer_resident_watched_bytes += pending_timing.buffer_resident_watched_bytes;
        timing.buffer_resident_declined_bytes += pending_timing.buffer_resident_declined_bytes;
        timing.buffer_resident_ineligible_bytes += pending_timing.buffer_resident_ineligible_bytes;
        timing.backend_res_buffer_create_ms += pending_timing.backend_res_buffer_create_ms;
        timing.backend_res_buffer_index_find_ms += pending_timing.backend_res_buffer_index_find_ms;
        timing.backend_res_buffer_index_insert_ms += pending_timing.backend_res_buffer_index_insert_ms;
        timing.backend_res_buffer_hash_ms += pending_timing.backend_res_buffer_hash_ms;
        timing.backend_res_descriptor_ms += pending_timing.backend_res_descriptor_ms;
        timing.backend_setup_pipeline_ms += pending_timing.backend_setup_pipeline_ms;
        timing.backend_pipeline_refs += pending_timing.backend_pipeline_refs;
        timing.backend_pipeline_hits += pending_timing.backend_pipeline_hits;
        timing.backend_pipeline_misses += pending_timing.backend_pipeline_misses;
        timing.backend_pipeline_bypasses += pending_timing.backend_pipeline_bypasses;
        timing.backend_pipeline_entries = pending_timing.backend_pipeline_entries;
        timing.backend_pipeline_evictions += pending_timing.backend_pipeline_evictions;
        timing.color_target_writes += pending_timing.color_target_writes;
        timing.color_target_write_hits += pending_timing.color_target_write_hits;
        timing.color_target_sample_hits += pending_timing.color_target_sample_hits;
        timing.color_target_readbacks += pending_timing.color_target_readbacks;
        timing.color_target_cached_bytes = pending_timing.color_target_cached_bytes;
        timing.color_target_cached_entries = pending_timing.color_target_cached_entries;
        timing.textures += pending_timing.textures;
        timing.texture_reuses += pending_timing.texture_reuses;
        timing.persistent_hits += pending_timing.persistent_hits;
        timing.persistent_misses += pending_timing.persistent_misses;
        timing.persistent_invalidations += pending_timing.persistent_invalidations;
        timing.persistent_submit_reuses += pending_timing.persistent_submit_reuses;
        timing.persistent_validations += pending_timing.persistent_validations;
        timing.persistent_validation_bytes += pending_timing.persistent_validation_bytes;
        timing.persistent_watch_reuses += pending_timing.persistent_watch_reuses;
        timing.persistent_watch_dirty += pending_timing.persistent_watch_dirty;
        timing.persistent_watch_unknown += pending_timing.persistent_watch_unknown;
        timing.persistent_submit_unknown += pending_timing.persistent_submit_unknown;
        timing.persistent_submit_overlap += pending_timing.persistent_submit_overlap;
        timing.persistent_submit_unarmed += pending_timing.persistent_submit_unarmed;
        timing.persistent_submit_stale += pending_timing.persistent_submit_stale;
        timing.publish_selected += pending_timing.publish_selected;
        timing.publish_selected_fmt0 += pending_timing.publish_selected_fmt0;
        timing.publish_selected_unknown += pending_timing.publish_selected_unknown;
        timing.publish_candidate_fmt0 += pending_timing.publish_candidate_fmt0;
        timing.persistent_watch_disabled += pending_timing.persistent_watch_disabled;
        timing.buffers += pending_timing.buffers;
        timing.buffer_views += pending_timing.buffer_views;
        timing.compact_buffer_resources += pending_timing.compact_buffer_resources;
        timing.full_buffer_resources += pending_timing.full_buffer_resources;
        timing.buffer_source_gate.add(pending_timing.buffer_source_gate);
        timing.texture_bytes += pending_timing.texture_bytes;
        timing.buffer_bytes += pending_timing.buffer_bytes;
        timing.buffer_materialized_bytes += pending_timing.buffer_materialized_bytes;
        timing.texture_ms += pending_timing.texture_ms;
        timing.tex_rtt_ms += pending_timing.tex_rtt_ms;
        timing.tex_compute_ms += pending_timing.tex_compute_ms;
        timing.tex_local_ms += pending_timing.tex_local_ms;
        timing.tex_persist_hit_ms += pending_timing.tex_persist_hit_ms;
        timing.tex_persist_reuse_ms += pending_timing.tex_persist_reuse_ms;
        timing.tex_persist_miss_ms += pending_timing.tex_persist_miss_ms;
        timing.tex_persist_invalid_ms += pending_timing.tex_persist_invalid_ms;
        timing.tex_persist_invalid_n += pending_timing.tex_persist_invalid_n;
        timing.tex_rtt_n += pending_timing.tex_rtt_n;
        timing.tex_compute_n += pending_timing.tex_compute_n;
        timing.tex_local_n += pending_timing.tex_local_n;
        timing.tex_persist_hit_n += pending_timing.tex_persist_hit_n;
        timing.tex_persist_reuse_n += pending_timing.tex_persist_reuse_n;
        timing.tex_persist_miss_n += pending_timing.tex_persist_miss_n;
        timing.tex_other_n += pending_timing.tex_other_n;
        timing.pass_pre_ms += pending_timing.pass_pre_ms;
        timing.pass_head_ms += pending_timing.pass_head_ms;
        timing.pass_loop_ms += pending_timing.pass_loop_ms;
        timing.resolve_stall_ms += pending_timing.resolve_stall_ms;
        timing.resolve_copy_stall_ms += pending_timing.resolve_copy_stall_ms;
        timing.resolve_read_ms += pending_timing.resolve_read_ms;
        timing.resolve_copy_ms += pending_timing.resolve_copy_ms;
        timing.resolve_n += pending_timing.resolve_n;
        timing.resolve_read_n += pending_timing.resolve_read_n;
        timing.resolve_bytes += pending_timing.resolve_bytes;
        timing.pass_groups_seen += pending_timing.pass_groups_seen;
        timing.pass_tail_ms += pending_timing.pass_tail_ms;
        timing.pass_post_ms += pending_timing.pass_post_ms;
        timing.pass_groups += pending_timing.pass_groups;
        timing.build_r_ms += pending_timing.build_r_ms;
        timing.build_validate_ms += pending_timing.build_validate_ms;
        timing.build_poison_ms += pending_timing.build_poison_ms;
        timing.build_indices_ms += pending_timing.build_indices_ms;
        timing.build_draws += pending_timing.build_draws;
        timing.build_rejected += pending_timing.build_rejected;
        timing.build_reflect_ms += pending_timing.build_reflect_ms;
        timing.build_reflect_calls += pending_timing.build_reflect_calls;
        timing.build_reflect_words += pending_timing.build_reflect_words;
        timing.build_reflect_unidentified += pending_timing.build_reflect_unidentified;
        timing.build_reflect_distinct += pending_timing.build_reflect_identities.size();
        timing.build_reflect_memo_hits += pending_timing.build_reflect_memo_hits;
        timing.build_reflect_memo_clears += pending_timing.build_reflect_memo_clears;
        timing.build_reflect_memo_mismatch += pending_timing.build_reflect_memo_mismatch;
        timing.buffer_ms += pending_timing.buffer_ms;
    };
    accumulate(totals);
    accumulate(window);
    if (totals.submits % 25 == 0) {
        const double nsub = static_cast<double>(totals.submits);
        const double pass_control = totals.pass_ms - totals.build_resources_ms -
                                    totals.backend_ms;
        const double other = totals.total_ms - totals.prelude_ms - totals.pass_ms -
                             totals.output_copy_ms;
        fprintf(stderr,
                "[render-timing] frontend submits=%llu callbacks=%llu avg_ms: total=%.2f "
                "prelude=%.2f build_resources=%.2f backend=%.2f pass_control=%.2f "
                "output_copy=%.2f other=%.2f dcc=%.2f/%0.2f/%.1fMiB\n",
                (unsigned long long)totals.submits, (unsigned long long)totals.callbacks,
                totals.total_ms / nsub, totals.prelude_ms / nsub,
                totals.build_resources_ms / nsub, totals.backend_ms / nsub,
                pass_control / nsub, totals.output_copy_ms / nsub, other / nsub,
                totals.dcc_materialize_ms / nsub,
                static_cast<double>(totals.dcc_materialize_surfaces) / nsub,
                totals.dcc_materialize_bytes / (nsub * 1024.0 * 1024.0));
        // #2395: the shader-analysis cache's counters existed and were read by NOTHING
        // outside two unit tests, so at runtime the cache was unobservable. A profile of
        // Blue Prince gameplay put ~17% of all CPU in rdna2_walk / make_shader_compile_key /
        // resolve_dynamic_fetch -- i.e. shaders being re-analysed during steady-state play --
        // and there was no way to tell WHY without rebuilding with instrumentation.
        //
        // `invalidations` is the discriminator and the reason this line exists. The cache is
        // keyed on the guest CODE ADDRESS and validated by memcmp, so:
        //   misses high, invalidations ~0  -> cold or capacity-bound (see evict/entries)
        //   invalidations high             -> the guest reuses addresses for DIFFERENT code,
        //                                     so every reuse costs a full re-analysis. That
        //                                     is a keying problem, not a sizing one, and no
        //                                     amount of cache MB fixes it.
        // Those two call for opposite fixes, which is exactly why the number has to be
        // visible rather than inferred from a flame graph.
        {
            const prosper::gpu::ShaderAnalysisCacheStats sa =
                prosper::gpu::shader_analysis_cache_stats();
            const uint64_t lookups = sa.hits + sa.misses;
            fprintf(stderr,
                    "[render-timing] shader_analysis hits=%llu misses=%llu (%.1f%% hit) "
                    "invalidations=%llu evictions=%llu bypasses=%llu entries=%llu %.1f MiB\n",
                    (unsigned long long)sa.hits, (unsigned long long)sa.misses,
                    lookups ? 100.0 * (double)sa.hits / (double)lookups : 0.0,
                    (unsigned long long)sa.invalidations, (unsigned long long)sa.evictions,
                    (unsigned long long)sa.bypasses, (unsigned long long)sa.entries,
                    sa.bytes / (1024.0 * 1024.0));
        }
        // pass_control, split (#2266). `pre` and `post` are the per-group work either
        // side of build_resources+backend; `other` is the SIGNED remainder -- groups that
        // returned before reaching the render, the tail after the group loop, and the
        // loop's own overhead. Published rather than clamped, because an unmeasured
        // region does not read as unmeasured: it reads as somebody else's cost, which is
        // how 85 ms/submit sat here attributed to nothing.
        {
            const double pc_other = pass_control - totals.pass_pre_ms -
                                    totals.pass_post_ms - totals.pass_head_ms -
                                    totals.pass_tail_ms -
                                    (totals.pass_loop_ms - totals.pass_pre_ms -
                                     totals.pass_post_ms);
            fprintf(stderr,
                    "[render-timing] pass_control %.2f ms/submit [head=%.2f loop=%.2f "
                    "(pre=%.2f post=%.2f) tail=%.2f other=%+.2f] "
                    "groups=%.1f rendered of %.1f seen\n",
                    pass_control / nsub, totals.pass_head_ms / nsub,
                    totals.pass_loop_ms / nsub,
                    totals.pass_pre_ms / nsub, totals.pass_post_ms / nsub,
                    totals.pass_tail_ms / nsub, pc_other / nsub,
                    (double)totals.pass_groups / nsub,
                    (double)totals.pass_groups_seen / nsub);
            if (pc_other < -0.05 * nsub)
                fprintf(stderr,
                        "[render-timing] *** pass_control leaves EXCEED their parent by %.2f "
                        "ms/submit -- instrument defect, not renderer\n",
                        -pc_other / nsub);
        }
        // The group loop's ONE early exit is the MSAA resolve, so the gap between
        // `loop` and pre+post+build+backend is resolve work. Split by what a resolve
        // does, because the three have different fixes: `stall` is submit_and_wait (a
        // full GPU pipeline flush), `read` is a whole-surface GPU->CPU readback done
        // only to materialise a CPU pixel mirror, `copy` is the GPU-side destination
        // copy that has to happen either way.
        fprintf(stderr,
                "[render-timing] resolve %.1f/submit [stall=%.2f read=%.2f "
                "copy_stall=%.2f copy=%.2f] "
                "reads=%.1f %.1f MiB/submit\n",
                (double)totals.resolve_n / nsub, totals.resolve_stall_ms / nsub,
                totals.resolve_read_ms / nsub, totals.resolve_copy_stall_ms / nsub,
                totals.resolve_copy_ms / nsub,
                (double)totals.resolve_read_n / nsub,
                totals.resolve_bytes / (nsub * 1024.0 * 1024.0));
        const double backend_detail_ms = totals.backend_target_ms +
            totals.backend_draw_setup_ms + totals.backend_record_upload_ms +
            totals.backend_gpu_wait_ms + totals.backend_readback_ms +
            totals.backend_cleanup_ms;
        fprintf(stderr,
                // detail is the SIX-term partition (target draw_setup record_upload
                // gpu_wait readback cleanup) and `other` is measured - detail, so the
                // row IS exhaustive. What it used to invite was summing NINE terms:
                // gpu_device and gpu_overhead are printed here too, and
                // gpu_overhead is COMPUTED as gpu_wait - gpu_device, so those two
                // add up to gpu_wait exactly and a nine-term sum double-counts it.
                // That is why the residual equalled gpu_wait to the hundredth
                // (#2248) -- not because gpu_wait was nested in a neighbour, but
                // because its own two CHILDREN were printed as its siblings.
                //
                // Parenthesised now, so the nesting is in the syntax rather than
                // recoverable only by arithmetic, and `other` is signed so the row
                // states whether it balances instead of leaving that to be derived.
                "[render-timing] backend-submit calls=%.2f draws=%.1f avg_ms: measured=%.2f "
                "detail=%.2f target=%.2f draw_setup=%.2f record_upload=%.2f "
                "gpu_wait=%.2f (device=%.2f overhead=%.2f) readback=%.2f "
                "cleanup=%.2f other=%+.2f\n",
                totals.backend_calls / nsub, totals.backend_draws / nsub,
                totals.backend_ms / nsub, backend_detail_ms / nsub,
                totals.backend_target_ms / nsub, totals.backend_draw_setup_ms / nsub,
                totals.backend_record_upload_ms / nsub, totals.backend_gpu_wait_ms / nsub,
                totals.backend_gpu_device_ms / nsub,
                std::max(0.0, totals.backend_gpu_wait_ms - totals.backend_gpu_device_ms) / nsub,
                totals.backend_readback_ms / nsub, totals.backend_cleanup_ms / nsub,
                (totals.backend_ms - backend_detail_ms) / nsub);
        fprintf(stderr,
                "[render-timing] backend-submit synchronization command_buffers=%.2f "
                "queue_submits=%.2f fence_waits=%.2f timestamps=%.2f  "
                "flush{readback=%.2f storage_wb=%.2f explicit=%.2f "
                "no_batch=%.2f cache_pressure=%.2f} cache_pressure_ms=%.2f\n",
                totals.backend_command_buffers / nsub,
                totals.backend_queue_submits / nsub,
                totals.backend_fence_waits / nsub,
                totals.backend_gpu_timestamp_samples / nsub,
                (double)totals.flush_readback / nsub,
                (double)totals.flush_storage_writeback / nsub,
                (double)totals.flush_explicit / nsub,
                (double)totals.flush_no_batch / nsub,
                (double)totals.flush_cache_pressure / nsub,
                totals.cache_pressure_ms / nsub);
        fprintf(stderr,
                // draw_setup is the PARENT one line above; these four are its leaves
                // and they did not add up to it -- 75.16 against 82.46 in the window
                // that found this, a 7.30 ms gap nothing named (#2249). An unmeasured
                // region does not read as unmeasured, it reads as somebody else's
                // cost, and 7.30 ms is small enough that nobody chases it and large
                // enough to hide an instrument -- which is exactly how trap 130's
                // 305 ms/submit got billed to the renderer.
                "[render-timing] backend-submit draw_setup avg_ms: shaders=%.2f fixed=%.2f "
                "resources=%.2f pipeline=%.2f other=%+.2f  fixed{index_upload=%.2f blend=%.2f "
                "depth_stencil=%.2f viewport=%.2f stages=%.2f pre_index_unsplit=%.2f "
                "other=%+.2f} pre_index{subgroup_scan=%.2f other=%+.2f}\n",
                totals.backend_setup_shader_ms / nsub,
                totals.backend_setup_fixed_ms / nsub,
                totals.backend_setup_resources_ms / nsub,
                totals.backend_setup_pipeline_ms / nsub,
                (totals.backend_draw_setup_ms - totals.backend_setup_shader_ms -
                 totals.backend_setup_fixed_ms - totals.backend_setup_resources_ms -
                 totals.backend_setup_pipeline_ms) / nsub,
                totals.backend_res_fixed_index_upload_ms / nsub,
                totals.backend_res_fixed_blend_ms / nsub,
                totals.backend_res_fixed_depth_stencil_ms / nsub,
                totals.backend_res_fixed_viewport_ms / nsub,
                totals.backend_res_fixed_stages_ms / nsub,
                totals.backend_res_fixed_prologue_ms / nsub,
                (totals.backend_setup_fixed_ms - totals.backend_res_fixed_index_upload_ms -
                 totals.backend_res_fixed_blend_ms - totals.backend_res_fixed_depth_stencil_ms -
                 totals.backend_res_fixed_viewport_ms - totals.backend_res_fixed_stages_ms -
                 totals.backend_res_fixed_prologue_ms) / nsub,
                totals.backend_res_prologue_subgroup_scan_ms / nsub,
                (totals.backend_res_fixed_prologue_ms - totals.backend_res_prologue_subgroup_scan_ms) / nsub);
        fprintf(stderr,
                "[render-timing] backend-submit resources avg_ms: texture=%.2f "
                "(upload=%.2f bind=%.2f lookup=%.2f) buffer=%.2f "
                "(acquire=%.2f copy=%.2f create=%.2f index_find=%.2f "
                "index_insert=%.2f hash=%.2f "
                "other=%+.2f) descriptor=%.2f other=%.2f\n",
                totals.backend_res_texture_ms / nsub,
                totals.backend_res_texture_upload_ms / nsub,
                totals.backend_res_texture_bind_ms / nsub,
                (totals.backend_res_texture_ms - totals.backend_res_texture_upload_ms -
                 totals.backend_res_texture_bind_ms) / nsub,
                totals.backend_res_buffer_ms / nsub,
                totals.backend_res_buffer_acquire_ms / nsub,
                totals.backend_res_buffer_copy_ms / nsub,
                totals.backend_res_buffer_create_ms / nsub,
                totals.backend_res_buffer_index_find_ms / nsub,
                totals.backend_res_buffer_index_insert_ms / nsub,
                totals.backend_res_buffer_hash_ms / nsub,
                (totals.backend_res_buffer_ms - totals.backend_res_buffer_acquire_ms -
                 totals.backend_res_buffer_copy_ms - totals.backend_res_buffer_create_ms -
                 totals.backend_res_buffer_index_find_ms -
                 totals.backend_res_buffer_index_insert_ms - totals.backend_res_buffer_hash_ms) / nsub,
                totals.backend_res_descriptor_ms / nsub,
                (totals.backend_setup_resources_ms - totals.backend_res_texture_ms -
                 totals.backend_res_buffer_ms - totals.backend_res_descriptor_ms) / nsub);
        fprintf(stderr,
                "[render-timing] backend-submit pipelines refs=%.1f hits=%.1f misses=%.1f "
                "bypass=%.1f entries=%llu evictions=%.1f\n",
                totals.backend_pipeline_refs / nsub,
                totals.backend_pipeline_hits / nsub,
                totals.backend_pipeline_misses / nsub,
                totals.backend_pipeline_bypasses / nsub,
                (unsigned long long)totals.backend_pipeline_entries,
                totals.backend_pipeline_evictions / nsub);
        const auto backend_buffers =
            prosper::test::render_host_buffer_pool_stats();
        fprintf(stderr,
                "[render-timing] backend_buffer_pool hits=%llu misses=%llu cached=%zu "
                "%.1f MiB evictions=%llu\n",
                (unsigned long long)backend_buffers.hits,
                (unsigned long long)backend_buffers.misses,
                backend_buffers.cached_buffers,
                backend_buffers.cached_bytes / (1024.0 * 1024.0),
                (unsigned long long)backend_buffers.evictions);
        // Sampled occupancy, not accumulated bytes or an upload working set.
        // Detached submission versions consume the owner/byte allowances too.
        const auto resident = prosper::test::resident_render_buffer_cache_snapshot();
        fprintf(stderr,
                "[render-timing] buffer_residency available=%u entries=%zu live_owners=%llu "
                "owner_limit=%llu charged_bytes=%llu byte_limit=%llu\n",
                unsigned(resident.available), resident.indexed_entries,
                (unsigned long long)resident.live_owners,
                (unsigned long long)resident.owner_limit,
                (unsigned long long)resident.charged_bytes,
                (unsigned long long)resident.byte_limit);
        fprintf(stderr,
                "[render-timing] publish_source selected=%llu fmt0=%llu unknown=%llu passes_fmt0=%llu\n",
                (unsigned long long)totals.publish_selected,
                (unsigned long long)totals.publish_selected_fmt0,
                (unsigned long long)totals.publish_selected_unknown,
                (unsigned long long)totals.publish_candidate_fmt0);
        fprintf(stderr,
                "[render-timing] color_targets writes=%llu load_hits=%llu sample_hits=%llu "
                "readbacks=%llu deferred=%llu cached=%llu %.1f MiB\n",
                (unsigned long long)totals.color_target_writes,
                (unsigned long long)totals.color_target_write_hits,
                (unsigned long long)totals.color_target_sample_hits,
                (unsigned long long)totals.color_target_readbacks,
                (unsigned long long)(totals.color_target_writes -
                                     totals.color_target_readbacks),
                (unsigned long long)totals.color_target_cached_entries,
                totals.color_target_cached_bytes / (1024.0 * 1024.0));
        fprintf(stderr,
                "[render-timing] resources textures=%llu reused=%llu %.1f MiB %.2f ms/submit; "
                "buffers=%llu views=%llu gate=%llu/%llu/%llu/%llu logical=%.1f MiB "
                "materialized=%.1f MiB %.2f ms/submit\n",
                (unsigned long long)totals.textures,
                (unsigned long long)totals.texture_reuses,
                totals.texture_bytes / (1024.0 * 1024.0), totals.texture_ms / nsub,
                (unsigned long long)totals.buffers,
                (unsigned long long)totals.buffer_views,
                (unsigned long long)totals.buffer_source_gate.tracked_cache_hits,
                (unsigned long long)totals.buffer_source_gate.tracked_cache_fills,
                (unsigned long long)totals.buffer_source_gate.tracked_untracked_misses,
                (unsigned long long)totals.buffer_source_gate.reserved_state_queries,
                totals.buffer_bytes / (1024.0 * 1024.0),
                totals.buffer_materialized_bytes / (1024.0 * 1024.0),
                totals.buffer_ms / nsub);
        // texture_ms attributed by OUTCOME class (#2262). The classes are decided in
        // a fixed order on the per-resource path and classified in that same order, so
        // they are mutually exclusive and every reference lands in exactly one -- or in
        // `unclassified`, which is a COUNT of references nothing above claimed. `other`
        // is the SIGNED millisecond remainder, published rather than clamped for the
        // reason #2246 records. The two residuals are independent: a non-zero `other`
        // with `unclassified=0` means the classification is losing time that a claimed
        // class should hold, which is a defect in this instrument and not in the
        // renderer.
        {
            const double tex_other = totals.texture_ms - totals.tex_rtt_ms -
                                     totals.tex_compute_ms - totals.tex_local_ms -
                                     totals.tex_persist_hit_ms - totals.tex_persist_reuse_ms -
                                     totals.tex_persist_miss_ms -
                                     totals.tex_persist_invalid_ms;
            fprintf(stderr,
                    "[render-timing] texture %.2f ms/submit [rtt=%.2f/%.1f compute=%.2f/%.1f "
                    "persist_reuse=%.2f/%.1f persist_hit=%.2f/%.1f "
                    "persist_miss=%.2f/%.1f persist_invalid=%.2f/%.1f "
                    "local=%.2f/%.1f] other=%+.2f "
                    "unclassified=%.1f  (ms/submit per class, refs/submit after the slash)\n",
                    totals.texture_ms / nsub,
                    totals.tex_rtt_ms / nsub, (double)totals.tex_rtt_n / nsub,
                    totals.tex_compute_ms / nsub, (double)totals.tex_compute_n / nsub,
                    totals.tex_persist_reuse_ms / nsub, (double)totals.tex_persist_reuse_n / nsub,
                    totals.tex_persist_hit_ms / nsub, (double)totals.tex_persist_hit_n / nsub,
                    totals.tex_persist_miss_ms / nsub, (double)totals.tex_persist_miss_n / nsub,
                    totals.tex_persist_invalid_ms / nsub, (double)totals.tex_persist_invalid_n / nsub,
                    totals.tex_local_ms / nsub, (double)totals.tex_local_n / nsub,
                    tex_other / nsub, (double)totals.tex_other_n / nsub);
            if (tex_other < -0.05 * nsub)
                fprintf(stderr,
                        "[render-timing] *** texture classes EXCEED their parent by %.2f "
                        "ms/submit -- the classification double-counts; instrument "
                        "defect, not renderer\n", -tex_other / nsub);
        }
        // build_resources, exhaustively (#2215). The leaves are per-draw spans inside
        // build_bds; `other` is the SIGNED residual -- everything in build_resources
        // that is not one of them, which on a healthy partition is the loop overhead
        // and the BackendDraw field copies. A large positive residual means work is
        // going somewhere unnamed; a NEGATIVE one means the leaves over-count their
        // parent, which is a defect in this instrument and not in the renderer. Both
        // are printed because a partition that cannot report its own incompleteness
        // attributes the gap to the subject -- the failure that cost #2246 two hours
        // and produced a 305 ms phantom.
        //
        // texture/buffer above are NOT leaves of this partition: they are measured
        // INSIDE build_R and are therefore a sub-split of `build_R` below. Adding them
        // here would double-count. That is the cross-layer subtraction #2243 fixed,
        // and this comment exists so nobody re-derives it from the field names.
        {
            // build_reflect_ms is INSIDE build_r_ms, so it is a sub-split of the
            // build_R leaf and must not be subtracted from build_resources here --
            // only from build_R's own residual above. Subtracting it at both levels
            // would drive the outer residual negative and trip the guard on a run
            // where nothing is wrong.
            const double br_other = totals.build_resources_ms - totals.build_r_ms -
                                    totals.build_validate_ms - totals.build_poison_ms -
                                    totals.build_indices_ms;
            fprintf(stderr,
                    "[render-timing] build_resources %.2f ms/submit [build_R=%.2f "
                    "(texture=%.2f buffer=%.2f reflect=%.2f other=%+.2f) "
                    "validate=%.2f indices=%.2f "
                    "poison=%.2f other=%+.2f] draws=%.1f (+%.1f rejected; build_R and validate cover both, poison and indices only the accepted)\n",
                    totals.build_resources_ms / nsub, totals.build_r_ms / nsub,
                    totals.texture_ms / nsub, totals.buffer_ms / nsub,
                    totals.build_reflect_ms / nsub,
                    // build_R's OWN residual, published for the same reason as its
                    // parent's. texture+buffer covered 3.71 of 5.20 ms on the first run
                    // of this instrument -- 29% of build_R with nothing naming it. A
                    // nested partition that reports only the outer remainder just moves
                    // the blind spot one level down, which is the mistake this whole
                    // line exists to stop repeating.
                    (totals.build_r_ms - totals.texture_ms - totals.buffer_ms -
                     totals.build_reflect_ms) / nsub,
                    totals.build_validate_ms / nsub, totals.build_indices_ms / nsub,
                    totals.build_poison_ms / nsub, br_other / nsub,
                    (double)totals.build_draws / nsub,
                    (double)totals.build_rejected / nsub);
            // A negative residual has one KNOWN trigger, and naming it here saves
            // the reader hunting a renderer defect: PROSPER_TARGET_STEP_HASH_DIM calls
            // build_bds() again at :5815, OUTSIDE the build_start/build_done span, so
            // every prefix rebuild lands in the leaves without landing in the parent --
            // and that loop is O(N^2) in draws. Zero occurrences on a default run.
            // The reflect leaf's own supporting numbers (#2256). `words` is what
            // the memo's key derivation and its equality check each traverse, so the
            // ms is checkable against it rather than merely assertable. `distinct` is
            // the per-submit ceiling on what an identity-keyed memo could remove:
            // calls - distinct - unidentified is the number of calls it could serve
            // from cache, and if that difference is small the memo is not worth
            // building however large the ms turns out to be.
            fprintf(stderr,
                    "[render-timing] build_R reflect %.3f ms/submit calls=%.1f "
                    "distinct=%.1f unidentified=%.1f words=%.0f/submit "
                    "memo_hits=%.1f clears=%.1f mismatch=%.1f\n",
                    totals.build_reflect_ms / nsub,
                    (double)totals.build_reflect_calls / nsub,
                    (double)totals.build_reflect_distinct / nsub,
                    (double)totals.build_reflect_unidentified / nsub,
                    (double)totals.build_reflect_words / nsub,
                    (double)totals.build_reflect_memo_hits / nsub,
                    (double)totals.build_reflect_memo_clears / nsub,
                    (double)totals.build_reflect_memo_mismatch / nsub);
            if (br_other < -0.05 * nsub)
                fprintf(stderr,
                        "[render-timing] *** build_resources leaves EXCEED their parent "
                        "by %.2f ms/submit -- the partition above is not trustworthy; "
                        "this is a defect in the instrument, not the renderer. Known "
                        "trigger: PROSPER_TARGET_STEP_HASH_DIM rebuilds draws outside "
                        "the measured span\n",
                        -br_other / nsub);
        }
        size_t persistent_source_bytes = 0;
        size_t persistent_pixel_bytes = 0;
        size_t persistent_watch_only_entries = 0;
        size_t persistent_watch_only_saved_bytes = 0;
        for (const auto& [key, texture] : persistent_decoded_textures) {
            (void)key;
            persistent_source_bytes += texture.source_prefix.size();
            persistent_pixel_bytes += texture.pixels ? texture.pixels->size() : 0;
            if (texture.source_watch_only) {
                ++persistent_watch_only_entries;
                persistent_watch_only_saved_bytes += texture.source_prefix_size;
            }
        }
        fprintf(stderr,
                "[render-timing] texture_cache hits=%llu submit_reuse=%llu "
                "(submit_unknown=%llu [unarmed=%llu stale=%llu] submit_overlap=%llu) misses=%llu "
                "watch_reuse=%llu watch_dirty=%llu watch_unknown=%llu watch_disabled=%llu "
                "invalid=%llu "
                "validations=%llu %.1f GiB entries=%zu %.1f MiB "
                "(source=%.1f pixels=%.1f MiB watch-only=%zu saved=%.1f MiB)\n",
                (unsigned long long)totals.persistent_hits,
                (unsigned long long)totals.persistent_submit_reuses,
                (unsigned long long)totals.persistent_submit_unknown,
                (unsigned long long)totals.persistent_submit_unarmed,
                (unsigned long long)totals.persistent_submit_stale,
                (unsigned long long)totals.persistent_submit_overlap,
                (unsigned long long)totals.persistent_misses,
                (unsigned long long)totals.persistent_watch_reuses,
                (unsigned long long)totals.persistent_watch_dirty,
                (unsigned long long)totals.persistent_watch_unknown,
                (unsigned long long)totals.persistent_watch_disabled,
                (unsigned long long)totals.persistent_invalidations,
                (unsigned long long)totals.persistent_validations,
                totals.persistent_validation_bytes / (1024.0 * 1024.0 * 1024.0),
                persistent_decoded_textures.size(),
                persistent_decoded_texture_bytes / (1024.0 * 1024.0),
                persistent_source_bytes / (1024.0 * 1024.0),
                persistent_pixel_bytes / (1024.0 * 1024.0),
                persistent_watch_only_entries,
                persistent_watch_only_saved_bytes / (1024.0 * 1024.0));
        // Identity-scope accounting (#1691). `cross_span` is the reuse submit scope adds
        // over the historical span-scoped map; `invalidated` is entries the in-submit
        // journal refused to carry across a span boundary.
        fprintf(stderr,
                "[render-timing] decode_scope decodes=%llu same_span=%llu "
                "cross_span=%llu invalidated=%llu pinned=%llu scope=%s\n",
                (unsigned long long)g_texture_decode_scope.decodes,
                (unsigned long long)g_texture_decode_scope.same_span_reuses,
                (unsigned long long)g_texture_decode_scope.cross_span_reuses,
                (unsigned long long)g_texture_decode_scope.invalidations,
                (unsigned long long)g_texture_decode_scope.scratch_pins,
                submit_decode_scope_disabled ? "span" : "submit");
        fprintf(stderr,
                "[render-timing] texture_preparation cumulative_this_thread "
                "dcc_reads=%llu dcc_bytes=%llu compute_span_queries=%llu "
                "generic_copy_bytes=%llu generic_copy_deferrals=%llu "
                "decoder_snapshot_candidate_bytes=%llu decoder_snapshot_reused_bytes=%llu "
                "late_snapshot_read_bytes=%llu cube_snapshot_guest_bytes=%llu "
                "cube_snapshot_renderer_bytes=%llu cube_snapshot_refusals=%llu\n",
                (unsigned long long)g_texture_decode_scope.dcc_metadata_read_attempts,
                (unsigned long long)g_texture_decode_scope.dcc_metadata_read_bytes,
                (unsigned long long)g_texture_decode_scope.compute_import_span_queries,
                (unsigned long long)g_texture_decode_scope.generic_source_copied_bytes,
                (unsigned long long)g_texture_decode_scope.generic_source_copy_deferrals,
                (unsigned long long)g_texture_decode_scope.decoder_snapshot_candidate_bytes,
                (unsigned long long)g_texture_decode_scope.decoder_snapshot_reused_bytes,
                (unsigned long long)g_texture_decode_scope.late_snapshot_read_bytes,
                (unsigned long long)g_texture_decode_scope.cube_snapshot_guest_bytes,
                (unsigned long long)g_texture_decode_scope.cube_snapshot_renderer_bytes,
                (unsigned long long)g_texture_decode_scope.cube_snapshot_refusals);
        size_t rtt_bytes = 0;
        for (const auto& [addr, surface] : g_rtt) {
            (void)addr;
            if (surface.rgba) rtt_bytes += surface.rgba->size();
        }
        size_t scratch_bytes = 0;
        for (const auto& scratch : texstore)
            scratch_bytes += scratch.capacity();
        // The decode INTERMEDIATE pool is retained memory too, and a census that omits
        // it under-reports the frontend's footprint by exactly the amount this change
        // stopped churning -- which is the number a reader would want to see.
        //
        // It is THIS THREAD's pool, and the label says so. The pool is thread_local, so
        // a process running N decoding threads retains up to N times this; printing a
        // per-thread figure under a heading that reads as a process total is the kind
        // of number that gets quoted as the whole footprint.
        const auto& intermediates = prosper::frontend::decode_scratch_pool();
        fprintf(stderr,
                "[render-timing] host_cache rtt=%zu %.1f MiB decode_scratch=%zu %.1f MiB "
                "intermediates(this thread)=%zu %.1f MiB validation=%.1f MiB\n",
                g_rtt.size(), rtt_bytes / (1024.0 * 1024.0),
                texstore.size(), scratch_bytes / (1024.0 * 1024.0),
                intermediates.retained_buffers(),
                intermediates.retained_bytes() / (1024.0 * 1024.0),
                persistent_validation_scratch.capacity() / (1024.0 * 1024.0));
        const auto write_watch = prosper::host::guest_write_watch_stats();
        fprintf(stderr,
                // `protect_fail` was spelled `protect` until #3317 added a field
                // counting mprotect CALLS to the same line. A reader then has
                // `protect=0` and `mprotect=N` side by side, and `protect=0` reads as
                // "nothing was protected" -- which is the opposite of what it means
                // (zero FAILURES is the healthy value). That misreading happened on the
                // first run of the new field; see GAME_COMPAT_ORCHESTRATION.md.
                "[render-timing] write_watch create=%llu ok=%llu pages=%llu no_map=%llu "
                "alias=%llu oversized=%llu sizes=%llu/%llu/%llu/%llu protect_fail=%llu "
                "query=%llu unchanged=%llu dirty=%llu "
                "unknown=%llu faults=%llu stale=%llu physical=%llu rearms=%llu "
                "host_write=%llu notifies/%llu no_alias pages_hit=%llu "
                "lock_contended=%llu page_protect=%llu calls/%.1f MiB\n",
                (unsigned long long)write_watch.create_attempts,
                (unsigned long long)write_watch.registrations,
                (unsigned long long)write_watch.registered_pages,
                (unsigned long long)write_watch.create_no_mapping,
                (unsigned long long)write_watch.create_incomplete_aliases,
                (unsigned long long)write_watch.create_oversized,
                (unsigned long long)write_watch.create_bytes_le_1m,
                (unsigned long long)write_watch.create_bytes_le_8m,
                (unsigned long long)write_watch.create_bytes_le_32m,
                (unsigned long long)write_watch.create_bytes_gt_32m,
                (unsigned long long)write_watch.create_protect_failures,
                (unsigned long long)write_watch.queries,
                (unsigned long long)write_watch.unchanged,
                (unsigned long long)write_watch.dirty,
                (unsigned long long)write_watch.unknown,
                (unsigned long long)write_watch.faults,
                (unsigned long long)write_watch.stale_faults,
                (unsigned long long)write_watch.physical_writes,
                (unsigned long long)write_watch.rearms,
                (unsigned long long)write_watch.host_write_notifies,
                (unsigned long long)write_watch.host_write_no_alias,
                (unsigned long long)write_watch.host_write_pages_hit,
                (unsigned long long)write_watch.host_write_lock_contended,
                (unsigned long long)write_watch.protect_calls,
                write_watch.protect_bytes / (1024.0 * 1024.0));
        // #3681: the scan cost of the three hot entry points, printed as visited/useful
        // pairs. Elapsed `watch_ms` alone cannot say whether a call walked a million
        // entries or waited on the mutex; these separate the two.
        // `query_pages` is counted only on the paths that actually walk the page list --
        // the audit and the legacy control. On the shipped path query() answers from the
        // flag and walks nothing, so printing a bare 0 here would state a measured "no
        // pages visited" where the truth is "not counted"; those need opposite reactions.
        char query_pages[64];
        if (write_watch.query_pages_visited)
            snprintf(query_pages, sizeof(query_pages), "%llu",
                     (unsigned long long)write_watch.query_pages_visited);
        else
            snprintf(query_pages, sizeof(query_pages), "0(not-walked)");
        fprintf(stderr,
                "[render-timing] write_watch_scan query_pages=%s/%llu queries "
                "host_write_pages=%llu scanned/%llu hit "
                "gpu_write=%llu notifies/%llu visited/%llu overlaps\n",
                query_pages,
                (unsigned long long)write_watch.queries,
                (unsigned long long)write_watch.host_write_pages_scanned,
                (unsigned long long)write_watch.host_write_pages_hit,
                (unsigned long long)write_watch.gpu_write_notifies,
                (unsigned long long)write_watch.gpu_write_registrations_visited,
                (unsigned long long)write_watch.gpu_write_overlaps);
        fprintf(stderr,
                "[render-timing] write_watch_index rearm_fast=%llu/%llu rearms "
                "audit_stale=%llu audit_conservative=%llu\n",
                (unsigned long long)write_watch.rearm_fast,
                (unsigned long long)write_watch.rearms,
                (unsigned long long)write_watch.query_audit_stale,
                (unsigned long long)write_watch.query_audit_conservative);
        const double wn = static_cast<double>(window.submits);
        const double window_pass_control = window.pass_ms -
            window.build_resources_ms - window.backend_ms;
        const double window_other = window.total_ms - window.prelude_ms -
            window.pass_ms - window.output_copy_ms;
        fprintf(stderr,
                "[render-window] frontend submits=%llu callbacks=%.1f avg_ms: total=%.2f "
                "prelude=%.2f build_resources=%.2f backend=%.2f pass_control=%.2f "
                "output_copy=%.2f other=%.2f dcc=%.2f/%0.2f/%.1fMiB; "
                "resources textures=%.1f "
                "reused=%.1f %.1f MiB %.2f ms buffers=%.1f views=%.1f "
                "gate=%.1f/%.1f/%.1f/%.1f "
                "logical=%.1f MiB materialized=%.1f MiB %.2f ms\n",
                (unsigned long long)window.submits, window.callbacks / wn,
                window.total_ms / wn, window.prelude_ms / wn,
                window.build_resources_ms / wn, window.backend_ms / wn,
                window_pass_control / wn, window.output_copy_ms / wn, window_other / wn,
                window.dcc_materialize_ms / wn,
                static_cast<double>(window.dcc_materialize_surfaces) / wn,
                window.dcc_materialize_bytes / (wn * 1024.0 * 1024.0),
                window.textures / wn,
                window.texture_reuses / wn,
                window.texture_bytes / (wn * 1024.0 * 1024.0), window.texture_ms / wn,
                window.buffers / wn, window.buffer_views / wn,
                window.buffer_source_gate.tracked_cache_hits / wn,
                window.buffer_source_gate.tracked_cache_fills / wn,
                window.buffer_source_gate.tracked_untracked_misses / wn,
                window.buffer_source_gate.reserved_state_queries / wn,
                window.buffer_bytes / (wn * 1024.0 * 1024.0),
                window.buffer_materialized_bytes / (wn * 1024.0 * 1024.0),
                window.buffer_ms / wn);
        {
            const double pc_other = window_pass_control - window.pass_pre_ms -
                                    window.pass_post_ms - window.pass_head_ms -
                                    window.pass_tail_ms -
                                    (window.pass_loop_ms - window.pass_pre_ms -
                                     window.pass_post_ms);
            fprintf(stderr,
                    "[render-window] pass_control %.2f ms/submit [head=%.2f loop=%.2f "
                    "(pre=%.2f post=%.2f) tail=%.2f other=%+.2f] "
                    "groups=%.1f rendered of %.1f seen\n",
                    window_pass_control / wn, window.pass_head_ms / wn,
                    window.pass_loop_ms / wn,
                    window.pass_pre_ms / wn, window.pass_post_ms / wn,
                    window.pass_tail_ms / wn, pc_other / wn,
                    (double)window.pass_groups / wn,
                    (double)window.pass_groups_seen / wn);
            if (pc_other < -0.05 * wn)
                fprintf(stderr,
                        "[render-window] *** pass_control leaves EXCEED their parent by "
                        "%.2f ms/submit -- instrument defect, not renderer\n",
                        -pc_other / wn);
        }
        fprintf(stderr,
                "[render-window] resolve %.1f/submit [stall=%.2f read=%.2f "
                "copy_stall=%.2f copy=%.2f] "
                "reads=%.1f %.1f MiB/submit\n",
                (double)window.resolve_n / wn, window.resolve_stall_ms / wn,
                window.resolve_read_ms / wn, window.resolve_copy_stall_ms / wn,
                window.resolve_copy_ms / wn,
                (double)window.resolve_read_n / wn,
                window.resolve_bytes / (wn * 1024.0 * 1024.0));
        // texture_ms attributed by OUTCOME class (#2262). The classes are decided in
        // a fixed order on the per-resource path and classified in that same order, so
        // they are mutually exclusive and every reference lands in exactly one -- or in
        // `unclassified`, which is a COUNT of references nothing above claimed. `other`
        // is the SIGNED millisecond remainder, published rather than clamped for the
        // reason #2246 records. The two residuals are independent: a non-zero `other`
        // with `unclassified=0` means the classification is losing time that a claimed
        // class should hold, which is a defect in this instrument and not in the
        // renderer.
        {
            const double tex_other = window.texture_ms - window.tex_rtt_ms -
                                     window.tex_compute_ms - window.tex_local_ms -
                                     window.tex_persist_hit_ms - window.tex_persist_reuse_ms -
                                     window.tex_persist_miss_ms -
                                     window.tex_persist_invalid_ms;
            fprintf(stderr,
                    "[render-window] texture %.2f ms/submit [rtt=%.2f/%.1f compute=%.2f/%.1f "
                    "persist_reuse=%.2f/%.1f persist_hit=%.2f/%.1f "
                    "persist_miss=%.2f/%.1f persist_invalid=%.2f/%.1f "
                    "local=%.2f/%.1f] other=%+.2f "
                    "unclassified=%.1f  (ms/submit per class, refs/submit after the slash)\n",
                    window.texture_ms / wn,
                    window.tex_rtt_ms / wn, (double)window.tex_rtt_n / wn,
                    window.tex_compute_ms / wn, (double)window.tex_compute_n / wn,
                    window.tex_persist_reuse_ms / wn, (double)window.tex_persist_reuse_n / wn,
                    window.tex_persist_hit_ms / wn, (double)window.tex_persist_hit_n / wn,
                    window.tex_persist_miss_ms / wn, (double)window.tex_persist_miss_n / wn,
                    window.tex_persist_invalid_ms / wn, (double)window.tex_persist_invalid_n / wn,
                    window.tex_local_ms / wn, (double)window.tex_local_n / wn,
                    tex_other / wn, (double)window.tex_other_n / wn);
            if (tex_other < -0.05 * wn)
                fprintf(stderr,
                        "[render-window] *** texture classes EXCEED their parent by %.2f "
                        "ms/submit -- the classification double-counts; instrument "
                        "defect, not renderer\n", -tex_other / wn);
        }
        const double window_backend_detail_ms = window.backend_target_ms +
            window.backend_draw_setup_ms + window.backend_record_upload_ms +
            window.backend_gpu_wait_ms + window.backend_readback_ms +
            window.backend_cleanup_ms;
        fprintf(stderr,
                // detail is the SIX-term partition (target draw_setup record_upload
                // gpu_wait readback cleanup) and `other` is measured - detail, so the
                // row IS exhaustive. What it used to invite was summing NINE terms:
                // gpu_device and gpu_overhead are printed here too, and
                // gpu_overhead is COMPUTED as gpu_wait - gpu_device, so those two
                // add up to gpu_wait exactly and a nine-term sum double-counts it.
                // That is why the residual equalled gpu_wait to the hundredth
                // (#2248) -- not because gpu_wait was nested in a neighbour, but
                // because its own two CHILDREN were printed as its siblings.
                //
                // Parenthesised now, so the nesting is in the syntax rather than
                // recoverable only by arithmetic, and `other` is signed so the row
                // states whether it balances instead of leaving that to be derived.
                "[render-window] backend-submit calls=%.2f draws=%.1f avg_ms: measured=%.2f "
                "detail=%.2f target=%.2f draw_setup=%.2f record_upload=%.2f "
                "gpu_wait=%.2f (device=%.2f overhead=%.2f) readback=%.2f "
                "cleanup=%.2f other=%+.2f\n",
                window.backend_calls / wn, window.backend_draws / wn,
                window.backend_ms / wn, window_backend_detail_ms / wn,
                window.backend_target_ms / wn, window.backend_draw_setup_ms / wn,
                window.backend_record_upload_ms / wn, window.backend_gpu_wait_ms / wn,
                window.backend_gpu_device_ms / wn,
                std::max(0.0, window.backend_gpu_wait_ms - window.backend_gpu_device_ms) / wn,
                window.backend_readback_ms / wn, window.backend_cleanup_ms / wn,
                (window.backend_ms - window_backend_detail_ms) / wn);
        fprintf(stderr,
                "[render-window] backend-submit synchronization command_buffers=%.2f "
                "queue_submits=%.2f fence_waits=%.2f timestamps=%.2f  "
                "flush{readback=%.2f storage_wb=%.2f explicit=%.2f "
                "no_batch=%.2f cache_pressure=%.2f} cache_pressure_ms=%.2f\n",
                window.backend_command_buffers / wn,
                window.backend_queue_submits / wn,
                window.backend_fence_waits / wn,
                window.backend_gpu_timestamp_samples / wn,
                (double)window.flush_readback / wn,
                (double)window.flush_storage_writeback / wn,
                (double)window.flush_explicit / wn,
                (double)window.flush_no_batch / wn,
                (double)window.flush_cache_pressure / wn,
                window.cache_pressure_ms / wn);
        fprintf(stderr,
                "[render-window] backend-submit draw_setup avg_ms: shaders=%.2f fixed=%.2f "
                "resources=%.2f pipeline=%.2f other=%+.2f  fixed{index_upload=%.2f blend=%.2f "
                "depth_stencil=%.2f viewport=%.2f stages=%.2f pre_index_unsplit=%.2f "
                "other=%+.2f} pre_index{subgroup_scan=%.2f other=%+.2f}\n",
                window.backend_setup_shader_ms / wn,
                window.backend_setup_fixed_ms / wn,
                window.backend_setup_resources_ms / wn,
                window.backend_setup_pipeline_ms / wn,
                (window.backend_draw_setup_ms - window.backend_setup_shader_ms -
                 window.backend_setup_fixed_ms - window.backend_setup_resources_ms -
                 window.backend_setup_pipeline_ms) / wn,
                window.backend_res_fixed_index_upload_ms / wn,
                window.backend_res_fixed_blend_ms / wn,
                window.backend_res_fixed_depth_stencil_ms / wn,
                window.backend_res_fixed_viewport_ms / wn,
                window.backend_res_fixed_stages_ms / wn,
                window.backend_res_fixed_prologue_ms / wn,
                (window.backend_setup_fixed_ms - window.backend_res_fixed_index_upload_ms -
                 window.backend_res_fixed_blend_ms - window.backend_res_fixed_depth_stencil_ms -
                 window.backend_res_fixed_viewport_ms - window.backend_res_fixed_stages_ms -
                 window.backend_res_fixed_prologue_ms) / wn,
                window.backend_res_prologue_subgroup_scan_ms / wn,
                (window.backend_res_fixed_prologue_ms - window.backend_res_prologue_subgroup_scan_ms) / wn);
        fprintf(stderr,
                "[render-window] backend-submit resources avg_ms: texture=%.2f "
                "(upload=%.2f bind=%.2f lookup=%.2f) buffer=%.2f "
                "(acquire=%.2f copy=%.2f create=%.2f index_find=%.2f "
                "index_insert=%.2f hash=%.2f "
                "other=%+.2f) descriptor=%.2f other=%.2f\n",
                window.backend_res_texture_ms / wn,
                window.backend_res_texture_upload_ms / wn,
                window.backend_res_texture_bind_ms / wn,
                (window.backend_res_texture_ms - window.backend_res_texture_upload_ms -
                 window.backend_res_texture_bind_ms) / wn,
                window.backend_res_buffer_ms / wn,
                window.backend_res_buffer_acquire_ms / wn,
                window.backend_res_buffer_copy_ms / wn,
                window.backend_res_buffer_create_ms / wn,
                window.backend_res_buffer_index_find_ms / wn,
                window.backend_res_buffer_index_insert_ms / wn,
                window.backend_res_buffer_hash_ms / wn,
                (window.backend_res_buffer_ms - window.backend_res_buffer_acquire_ms -
                 window.backend_res_buffer_copy_ms - window.backend_res_buffer_create_ms -
                 window.backend_res_buffer_index_find_ms -
                 window.backend_res_buffer_index_insert_ms - window.backend_res_buffer_hash_ms) / wn,
                window.backend_res_descriptor_ms / wn,
                (window.backend_setup_resources_ms - window.backend_res_texture_ms -
                 window.backend_res_buffer_ms - window.backend_res_descriptor_ms) / wn);
        // Same partition on the WINDOW path, so it is present wherever the buffer
        // and fixed ones are (#2246, #2252). The window path isolates a PHASE, which
        // for a title whose draw count swings 351 -> 2,106 between phases is the more
        // useful axis than a lifetime total -- and a partition visible on one path and
        // not the other invites exactly the cross-aggregation comparison that has cost
        // us two wrong numbers today.
        {
            const double br_other_w = window.build_resources_ms - window.build_r_ms -
                                      window.build_validate_ms - window.build_poison_ms -
                                      window.build_indices_ms;
            fprintf(stderr,
                    "[render-window] build_resources %.2f ms/submit [build_R=%.2f "
                    "(texture=%.2f buffer=%.2f reflect=%.2f other=%+.2f) "
                    "validate=%.2f indices=%.2f "
                    "poison=%.2f other=%+.2f] draws=%.1f (+%.1f rejected)\n",
                    window.build_resources_ms / wn, window.build_r_ms / wn,
                    window.texture_ms / wn, window.buffer_ms / wn,
                    window.build_reflect_ms / wn,
                    (window.build_r_ms - window.texture_ms - window.buffer_ms -
                     window.build_reflect_ms) / wn,
                    window.build_validate_ms / wn, window.build_indices_ms / wn,
                    window.build_poison_ms / wn, br_other_w / wn,
                    (double)window.build_draws / wn,
                    (double)window.build_rejected / wn);
            fprintf(stderr,
                    "[render-window] build_R reflect %.3f ms/submit calls=%.1f "
                    "distinct=%.1f unidentified=%.1f words=%.0f/submit "
                    "memo_hits=%.1f clears=%.1f mismatch=%.1f\n",
                    window.build_reflect_ms / wn,
                    (double)window.build_reflect_calls / wn,
                    (double)window.build_reflect_distinct / wn,
                    (double)window.build_reflect_unidentified / wn,
                    (double)window.build_reflect_words / wn,
                    (double)window.build_reflect_memo_hits / wn,
                    (double)window.build_reflect_memo_clears / wn,
                    (double)window.build_reflect_memo_mismatch / wn);
            if (br_other_w < -0.05 * wn)
                fprintf(stderr,
                        "[render-window] *** build_resources leaves EXCEED their parent "
                        "by %.2f ms/submit -- instrument defect, not renderer. Known "
                        "trigger: PROSPER_TARGET_STEP_HASH_DIM\n",
                        -br_other_w / wn);
        }
        fprintf(stderr,
                "[render-window] backend-submit pipelines refs=%.1f hits=%.1f misses=%.1f "
                "bypass=%.1f entries=%llu evictions=%.1f\n",
                window.backend_pipeline_refs / wn,
                window.backend_pipeline_hits / wn,
                window.backend_pipeline_misses / wn,
                window.backend_pipeline_bypasses / wn,
                (unsigned long long)window.backend_pipeline_entries,
                window.backend_pipeline_evictions / wn);
        fprintf(stderr,
                "[render-window] color_targets writes=%.1f load_hits=%.1f sample_hits=%.1f "
                "readbacks=%.1f deferred=%.1f cached=%llu %.1f MiB\n",
                window.color_target_writes / wn,
                window.color_target_write_hits / wn,
                window.color_target_sample_hits / wn,
                window.color_target_readbacks / wn,
                (window.color_target_writes - window.color_target_readbacks) / wn,
                (unsigned long long)window.color_target_cached_entries,
                window.color_target_cached_bytes / (1024.0 * 1024.0));
        fprintf(stderr,
                "[render-window] texture_cache hits=%.1f submit_reuse=%.1f misses=%.1f "
                "watch_reuse=%.1f watch_dirty=%.1f watch_unknown=%.1f watch_disabled=%.1f "
                "invalid=%.1f "
                "validations=%.1f %.1f MiB\n",
                window.persistent_hits / wn, window.persistent_submit_reuses / wn,
                window.persistent_misses / wn, window.persistent_watch_reuses / wn,
                window.persistent_watch_dirty / wn, window.persistent_watch_unknown / wn,
                window.persistent_watch_disabled / wn,
                window.persistent_invalidations / wn,
                window.persistent_validations / wn,
                window.persistent_validation_bytes / (wn * 1024.0 * 1024.0));
        window = {};
    }
}

} // namespace prosper::frontend::submit_renderer
