// resolve_pass -- see resolve_pass.hpp. Moved verbatim out of per_target_passes.cpp (#3892).
#include "shared/live/submit_renderer/resolve_pass.hpp"

namespace prosper::frontend::submit_renderer {

void resolve_pass(ResolvePassContext& ctx) {
    prosper::test::BackendProducerAttempt producer_attempt;
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& batch_backend_submits = ctx.batch_backend_submits;
    auto& items = ctx.items;
    auto& pending_timing = ctx.pending_timing;
    auto& timing_enabled = ctx.timing_enabled;
    auto& pass_i = ctx.pass_i;
    auto& backend_submission = ctx.backend_submission;
    auto& pass = ctx.pass;
    // g_rtt is named, uncaptured, by the failure-cleanup lambda below, which the backend may run
    // after this function returns (a variable with static storage duration cannot be captured).
    // A function-scope static reference binds once to that same process-lifetime cache, so the
    // lambda keeps reaching it exactly as it did inside the pass loop.
    static RttCache& g_rtt = ctx.g_rtt;
    const uint64_t rsrc = pass.front()->color0_base;
    const uint64_t rdst = pass.front()->color1_base;
    auto src_it = rsrc ? g_rtt.find(rsrc) : g_rtt.end();
    if (src_it != g_rtt.end() && src_it->second.volume_depth) {
        // Fixed-function resolve currently copies one 2D plane. A retained
        // volume cannot be inherited as a 2D destination or read back through
        // that path. Revoke any old destination instead of publishing an alias.
        prosper::test::invalidate_persistent_color_target(rdst);
        auto dst_it = rdst ? g_rtt.find(rdst) : g_rtt.end();
        if (dst_it != g_rtt.end() && dst_it->second.volume_guest_bytes) {
            dst_it->second.rgba.reset();
            dst_it->second.has_uniform_color = false;
            dst_it->second.gpu_valid = false;
        } else if (dst_it != g_rtt.end()) {
            g_rtt.erase(dst_it);
        }
        src_it = g_rtt.end();
    }
    if (src_it != g_rtt.end() && src_it->second.has_uniform_color)
        materialize_uniform_rtt(src_it->second);
    // Deferred RTT readback (#1284): the resolve is a CPU copy of the source's
    // pixels, and the source pass usually rendered EARLIER IN THIS BATCH with its
    // readback deferred. The consumer scan cannot see a resolve (it consumes via
    // cb_resolve, not a sampled resource), so materialize here: flush the pending
    // batch first — a mid-batch readback would otherwise return stale pixels —
    // then read the persistent image back once.
    // PROSPER_EAGER_RESOLVE_READBACK: restore the pre-#2266 behaviour, where the
    // CPU mirror was materialised here unconditionally. Kept as a permanent
    // bisection lever so the A/B for this change is single-variable on ONE binary,
    // and because this path sits on the #1334/#1382/#1287 evidence chain -- if a
    // title regresses, the first question is whether the lazy mirror caused it and
    // that has to be answerable without a rebuild.
    //
    // Measured before the change, Blue Prince gameplay window (#2266):
    //   resolve 5.1/submit [stall=20.76 read=20.63 copy=1.54] 63.3 MiB/submit
    // 63.3 MiB at 3.1 GB/s -- a PCIe readback rate -- for a CPU mirror that this
    // file materialises ON DEMAND everywhere else (see the readback at the compute
    // consumer, whose comment says intermediates "normally stay in the persistent
    // Vulkan target" and are materialised "only when an ordered compute dispatch
    // actually consumes the surface"). The resolve was doing eagerly what the rest
    // of the renderer does lazily, and its own comment already calls the CPU pixels
    // the failure case: "on failure the shared CPU pixels are the only truth".
    static const bool eager_resolve_readback =
        getenv("PROSPER_EAGER_RESOLVE_READBACK") != nullptr;
    if (src_it != g_rtt.end() && src_it->second.gpu_valid &&
        src_it->second.w && src_it->second.h && rdst) {
        RttSurf& src_surface = src_it->second;
        const VkFormat src_format =
            prosper::test::backend_color_format(src_surface.format);
        const uint32_t src_bpp =
            prosper::test::backend_color_bytes_per_pixel(src_format);
        const size_t src_bytes = static_cast<size_t>(src_surface.w) *
            src_surface.h * src_bpp;
        if (src_bpp && eager_resolve_readback &&
            (!src_surface.rgba || src_surface.rgba->size() != src_bytes)) {
            const prosper::test::RenderVkCtx& ctx =
                prosper::test::render_vk_ctx();
            const auto rs0 = timing_enabled
                ? RenderClock::now() : RenderClock::time_point{};
            if (ctx.ok && backend_submission.pending())
                backend_submission.submit_and_wait(ctx.dev, ctx.queue, false);
            const auto rs1 = timing_enabled
                ? RenderClock::now() : RenderClock::time_point{};
            std::vector<uint8_t> materialized;
            std::string error;
            const bool read_ok = prosper::test::readback_persistent_color_target(
                    rsrc, src_surface.w, src_surface.h, src_format,
                    materialized, error);
            if (timing_enabled) {
                const auto rs2 = RenderClock::now();
                pending_timing.resolve_stall_ms +=
                    std::chrono::duration<double, std::milli>(rs1 - rs0).count();
                pending_timing.resolve_read_ms +=
                    std::chrono::duration<double, std::milli>(rs2 - rs1).count();
                ++pending_timing.resolve_read_n;
                pending_timing.resolve_bytes += src_bytes;
            }
            if (read_ok &&
                materialized.size() == src_bytes) {
                src_surface.rgba =
                    std::make_shared<const std::vector<uint8_t>>(
                        std::move(materialized));
            } else {
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    fprintf(stderr,
                            "[rtt] resolve source readback failed: "
                            "base=0x%llx extent=%ux%u error=%s\n",
                            (unsigned long long)rsrc, src_surface.w,
                            src_surface.h, error.c_str());
            }
        }
    }
    // A GPU-only source is now the ordinary case, so the destination no longer
    // requires CPU pixels to exist before the resolve. `gpu_valid` is what the
    // copy below needs; `rgba` is inherited when the source happens to have it
    // and is otherwise materialised on demand by the consumer path, exactly as
    // for every other graphics-to-graphics intermediate.
    if (rsrc && rdst && src_it != g_rtt.end() &&
        (src_it->second.rgba || src_it->second.gpu_valid)) {
        // Copy the source RttSurf out before the g_rtt[rdst] insert: operator[] may
        // rehash and invalidate src_it before the assignment reads it.
        RttSurf resolved = src_it->second;   // shares pixels (shared_ptr), no deep copy
        // A resolve copies one current 2D plane. It does not transfer the
        // source's older unpublished volume to another address, nor does it
        // publish slices of a volume formerly owned by the destination.
        const auto old_destination = g_rtt.find(rdst);
        resolved.guest_origins = old_destination != g_rtt.end()
            ? old_destination->second.guest_origins : prosper::test::BackendGuestProducerOrigins{};
        resolved.volume_depth = 0;
        resolved.volume_guest_bytes = old_destination != g_rtt.end()
            ? old_destination->second.volume_guest_bytes : 0u;
        resolved.volume_footprint_proven = old_destination != g_rtt.end() &&
            old_destination->second.volume_footprint_proven;
        // The resolve destination has its own descriptor/DCC allocation. Do not
        // transfer the source surface's metadata identity to an unrelated base.
        resolved.dcc_metadata_addr = 0;
        resolved.dcc_metadata_bytes = 0;
        resolved.dcc_num_components = 0;
        resolved.dcc_alpha_is_on_msb = false;
        resolved.dcc_width = resolved.dcc_height = 0;
        resolved.dcc_metadata_dirty = false;
        resolved.dcc_guest_origins = {};
        const uint32_t rw = resolved.w, rh = resolved.h;
        // The destination programming supplies its own complete physical layout. Source pixels
        // or a Vulkan extent cannot lend their allocation proof to another guest identity.
        uint64_t destination_bytes = 0;
        for (const auto* draw : pass) {
            const uint64_t bytes = raw_snapshot_color_footprint(*draw, 1u);
            if (!bytes) { destination_bytes = 0; break; }
            destination_bytes = std::max(destination_bytes, bytes);
        }
        bool batched_copy_recorded = false;
        // #1334: the destination-keyed persistent GPU image did NOT receive these
        // pixels — inheriting gpu_valid from the source let a later #780 CPU-copy
        // discard leave consumers importing a stale/zero image (Blue Prince's
        // compute tonemap read black; #1287/#1381 evidence chain). Copy the
        // device-local pixels into the destination identity so gpu_valid is
        // genuinely true; on failure the shared CPU pixels are the only truth.
        if (resolved.gpu_valid) {
            // The control submits the copy out of band, so every earlier pass —
            // including one targeting the destination — must complete first.
            // The default records the copy into the same ordered batch instead;
            // its barriers supply visibility and the final batch fence owns
            // publication, failure invalidation and resource lifetime.
            static const bool no_batched_resolve_copy =
                PROSPER_ENV_VALUE("PROSPER_NO_BATCHED_RESOLVE_COPY") != nullptr;
            const bool batch_resolve_copy =
                prosper::frontend::resolve_copy_may_batch(
                    batch_backend_submits, no_batched_resolve_copy);
            if (!batch_resolve_copy) {
                const prosper::test::RenderVkCtx& copy_ctx =
                    prosper::test::render_vk_ctx();
                const auto cs0 = timing_enabled
                    ? RenderClock::now() : RenderClock::time_point{};
                if (copy_ctx.ok && backend_submission.pending())
                    backend_submission.submit_and_wait(copy_ctx.dev,
                                                       copy_ctx.queue, false);
                if (timing_enabled)
                    pending_timing.resolve_copy_stall_ms +=
                        std::chrono::duration<double, std::milli>(
                            RenderClock::now() - cs0).count();
            }
            std::string copy_error;
            const auto rc0 = timing_enabled
                ? RenderClock::now() : RenderClock::time_point{};
            const bool copy_ok = prosper::test::copy_persistent_color_target(
                    rsrc, rdst, rw, rh,
                    prosper::test::backend_color_format(resolved.format),
                    copy_error,
                    batch_resolve_copy ? &backend_submission : nullptr,
                    destination_bytes);
            batched_copy_recorded = copy_ok && batch_resolve_copy;
            if (timing_enabled)
                pending_timing.resolve_copy_ms +=
                    std::chrono::duration<double, std::milli>(
                        RenderClock::now() - rc0).count();
            if (!copy_ok) {
                resolved.gpu_valid = false;
                // The CPU mirror is the documented fallback -- "on failure the
                // shared CPU pixels are the only truth" -- so with the eager
                // readback gone it has to be materialised HERE, on the failure
                // path it was always meant for. Without this the destination
                // would have neither a valid GPU image nor pixels, which is
                // strictly worse than before this change rather than merely
                // slower. The flush is required for the same reason the eager
                // path needed one: a mid-batch readback returns stale pixels.
                if (!resolved.rgba || resolved.rgba->size() !=
                        static_cast<size_t>(rw) * rh *
                        prosper::test::backend_color_bytes_per_pixel(
                            prosper::test::backend_color_format(resolved.format))) {
                    const prosper::test::RenderVkCtx& fb_ctx =
                        prosper::test::render_vk_ctx();
                    if (fb_ctx.ok && backend_submission.pending())
                        backend_submission.submit_and_wait(fb_ctx.dev,
                                                           fb_ctx.queue, false);
                    std::vector<uint8_t> fallback;
                    std::string fb_error;
                    if (prosper::test::readback_persistent_color_target(
                            rsrc, rw, rh,
                            prosper::test::backend_color_format(resolved.format),
                            fallback, fb_error))
                        resolved.rgba =
                            std::make_shared<const std::vector<uint8_t>>(
                                std::move(fallback));
                }
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 8)
                    fprintf(stderr,
                            "[msaa] resolve GPU copy failed (%s) — destination "
                            "keeps CPU pixels only (#1334)\n",
                            copy_error.c_str());
            }
        }
        const uint64_t expected = static_cast<uint64_t>(rw) * rh *
            prosper::test::backend_color_bytes_per_pixel(prosper::test::backend_color_format(resolved.format));
        producer_attempt.accepted = resolved.gpu_valid ||
            (expected && resolved.rgba && resolved.rgba->size() == expected);
        const auto* retained_destination = resolved.gpu_valid
            ? prosper::test::find_persistent_color_target(rdst, rw, rh,
                prosper::test::backend_color_format(resolved.format)) : nullptr;
        if (retained_destination) resolved.guest_origins.merge(retained_destination->guest_origins);
        else resolved.guest_origins.observe(rdst, destination_bytes);
        g_rtt[rdst] = std::move(resolved);    // dest inherits src content/extent/format
        // The one whole-entry copy into the cache: it may carry a footprint.
        if (g_rtt[rdst].volume_guest_bytes) g_volume_targets.note(rdst);
        if (batched_copy_recorded) {
            const VkFormat resolved_format = g_rtt[rdst].format;
            backend_submission.add_failure_cleanup(
                [rdst, rw, rh, resolved_format]() {
                    auto failed = g_rtt.find(rdst);
                    if (failed != g_rtt.end() && failed->second.w == rw &&
                        failed->second.h == rh &&
                        failed->second.format == resolved_format)
                        failed->second.gpu_valid = false;
                });
        }
        if (PROSPER_ENV_ON("PROSPER_MSAA_LOG"))
            fprintf(stderr, "[msaa] resolve copy 0x%llx -> 0x%llx (%ux%u)\n",
                    (unsigned long long)rsrc, (unsigned long long)rdst, rw, rh);
    } else {
        // A resolve that performs no copy is DROPPED RENDERED CONTENT: the
        // destination keeps whatever it held before, and every later census
        // reads it as "written by nothing". Report it unconditionally, bounded
        // like the copy-failure warning above, so the one remaining way a
        // resolve can produce nothing is fail-visible rather than silent.
        // PROSPER_MSAA_LOG still reports every occurrence.
        static std::atomic<int> skipped{0};
        if (PROSPER_ENV_VALUE("PROSPER_MSAA_LOG") ||
            skipped.fetch_add(1) < 8)
            fprintf(stderr, "[msaa] resolve SKIP src=0x%llx dst=0x%llx (%s)\n",
                    (unsigned long long)rsrc, (unsigned long long)rdst,
                    !rsrc || !rdst ? "missing base"
                                   : "source has no rendered surface");
    }
    if (timing_enabled) ++pending_timing.resolve_n;
    // A normal final render group flushes the ordered backend batch inside
    // render_draws_rgba. A resolve returns before that call, so a terminal
    // resolve must establish the same boundary here. Otherwise the tail below
    // can read back or publish g_rtt's speculative destination before the copy
    // has even reached the queue.
    if (prosper::frontend::terminal_resolve_must_flush(
            batch_backend_submits, pass_i == items.size(),
            backend_submission.pending())) {
        const prosper::test::RenderVkCtx& terminal_ctx =
            prosper::test::render_vk_ctx();
        if (terminal_ctx.ok)
            backend_submission.submit_and_wait(
                terminal_ctx.dev, terminal_ctx.queue, false);
        else
            backend_submission.discard();
    }
}

} // namespace prosper::frontend::submit_renderer
