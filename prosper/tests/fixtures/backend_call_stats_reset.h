// Same-TU fixture: the per-call backend statistics a logical draw batch must clear when it is
// refused before any render pass runs (#4687). Included by render_runner.h just before
// render_draws_rgba, after every storage below is declared; kept out of that file because it is
// at its size ratchet.
#pragma once

// A caller reads these thread-locals right after a render_draws_rgba call and attributes them to
// that call; the live frontend does exactly that for colour-target writes, per-slot
// retained_slots and (under timing) pipeline-cache work. render_draw_pass_rgba clears them for
// every pass it runs. A batch refused BEFORE any pass (the resource-order preflight) never gets
// there, so it clears them here, or the previous pass's numbers read as this call's. #4687: a
// stale retained_slots bit let the frontend claim a volume version for a pass that never ran.
inline void reset_backend_per_call_stats() {
    backend_color_target_stats_storage() = {};
    backend_texture_upload_stats_storage() = {};
    backend_resource_reuse_stats_storage() = {};
    backend_pipeline_cache_stats_storage() = {};
    backend_render_timing_stats_storage() = {};
    fragment_draw_backend_stats() = {};
}
