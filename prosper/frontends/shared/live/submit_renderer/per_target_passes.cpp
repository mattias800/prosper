// render_per_target_passes -- see per_target_passes.hpp. Moved verbatim out of live_renderer.cpp (#3892).
#include "shared/live/submit_renderer/per_target_passes.hpp"
#include "shared/live/submit_renderer/guest_reads.hpp"
#include "shared/live/submit_renderer/resolve_pass.hpp"     // resolve_pass (#3892)
#include "shared/live/submit_renderer/pass_diagnostics.hpp" // the pass-loop diagnostics (#3892)

namespace prosper::frontend::submit_renderer {

void render_per_target_passes(PerTargetPassContext& ctx) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& g_pass_log_submit = ctx.g_pass_log_submit;
    auto& g_pass_log_window = ctx.g_pass_log_window;
    auto& g_persist_window = ctx.g_persist_window;
    auto& g_persist_filter = ctx.g_persist_filter;
    auto& g_persist_extent = ctx.g_persist_extent;
    auto& frame_no = ctx.frame_no;
    auto& live_gpu_targets = ctx.live_gpu_targets;
    auto& defer_intermediate_scanout = ctx.defer_intermediate_scanout;
    auto& batch_backend_submits = ctx.batch_backend_submits;
    auto& items = ctx.items;
    auto& w = ctx.w;
    auto& h = ctx.h;
    auto& phase = ctx.phase;
    auto& pinned_scanouts = ctx.pinned_scanouts;
    auto& pinned_renderer_mip_targets = ctx.pinned_renderer_mip_targets;
    auto& g_this_submit = ctx.g_this_submit;
    auto& rtt_log = ctx.rtt_log;
    auto& pending_timing = ctx.pending_timing;
    auto& pending_rtt_timing = ctx.pending_rtt_timing;
    auto& timing_enabled = ctx.timing_enabled;
    auto& lightweight_rtt_timing = ctx.lightweight_rtt_timing;
    auto& texstore = ctx.texstore;
    auto& texstore_pinned = ctx.texstore_pinned;
    auto& texstore_used = ctx.texstore_used;
    auto& decoded_textures = ctx.decoded_textures;
    auto& resource_hash_w = ctx.resource_hash_w;
    auto& resource_hash_h = ctx.resource_hash_h;
    auto& target_step_w = ctx.target_step_w;
    auto& target_step_h = ctx.target_step_h;
    auto& target_step_min_draws = ctx.target_step_min_draws;
    auto& pass_tail_start = ctx.pass_tail_start;
    auto& pass_tail_measured_before = ctx.pass_tail_measured_before;
    auto& pass_timing_start = ctx.pass_timing_start;
    auto& selected_pixels = ctx.selected_pixels;
    auto& selected_source_submit = ctx.selected_source_submit;
    auto& present_extent_bytes = ctx.present_extent_bytes;
    auto& present_choice = ctx.present_choice;
    auto& px_front_w = ctx.px_front_w;
    auto& px_front_h = ctx.px_front_h;
    auto& px_vo_w = ctx.px_vo_w;
    auto& px_vo_h = ctx.px_vo_h;
    auto& px_last_w = ctx.px_last_w;
    auto& px_last_h = ctx.px_last_h;
    auto& px_front_base = ctx.px_front_base;
    auto& px_vo_base = ctx.px_vo_base;
    auto& px_last_base = ctx.px_last_base;
    auto& px_front_source_submit = ctx.px_front_source_submit;
    auto& px_vo_source_submit = ctx.px_vo_source_submit;
    auto& px_last_source_submit = ctx.px_last_source_submit;
    auto& px_front_fmt = ctx.px_front_fmt;
    auto& px_vo_fmt = ctx.px_vo_fmt;
    auto& px_last_fmt = ctx.px_last_fmt;
    // The callback's process-lifetime g_rtt is named, uncaptured, by a failure-cleanup lambda
    // the backend may run after this function returns (a variable with static storage duration
    // cannot be captured). A function-scope static reference binds once to that same object,
    // so the lambda keeps reaching it exactly as it did inside the callback.
    static RttCache& g_rtt = ctx.g_rtt;
    // build_bds and record_backend_timing, over the same contexts the callback's forwarders use:
    // each call is still a direct call into build_backend_draws / record_backend_timing_stats.
    auto build_bds = [&](const std::vector<const prosper::gpu::DrawItem*>& group,
                         prosper::test::BackendSubmissionBatch* producer_batch = nullptr) {
        return build_backend_draws(ctx.backend_draw_ctx, group, producer_batch);
    };
    auto record_backend_timing = [&](const prosper::test::BackendRenderTimingStats& backend,
                                     const prosper::test::BackendTextureUploadStats& textures,
                                     const prosper::test::BackendPipelineCacheStats& pipelines,
                                     const prosper::test::BackendResourceReuseStats& reuse) {
        return record_backend_timing_stats(ctx.backend_timing_ctx, backend, textures, pipelines, reuse);
    };
    auto pin_renderer_mip_target = [&](uint64_t id, uint32_t width, uint32_t height,
                                       VkFormat format, bool gpu_valid) {
        if (!gpu_valid || !id || !width || !height ||
            !prosper::frontend::renderer_mip_target_requires_submit_retention(format))
            return;
        const auto already_pinned = std::find_if(
            pinned_renderer_mip_targets.begin(), pinned_renderer_mip_targets.end(),
            [&](const PinnedRendererMipTarget& target) {
                return target.id == id && target.width == width &&
                       target.height == height && target.format == format;
            });
        if (already_pinned != pinned_renderer_mip_targets.end()) {
            // A new production supersedes any earlier consumption of the same identity.
            // It must remain retained for the next mip consumer, which may be in a later
            // logical submit.
            already_pinned->consumed = false;
            return;
        }
        // This is a narrow bridge for independently rendered packed-HDR mip levels, not a
        // second unbounded residency cache. GTA V's observed scene uses 22 identities;
        // retaining substantially more without one being consumed indicates a different
        // workload and must fall back visibly to the ordinary LRU contract.
        constexpr size_t kMaxPendingRendererMipTargets = 64;
        if (pinned_renderer_mip_targets.size() >= kMaxPendingRendererMipTargets) {
            const PinnedRendererMipTarget oldest = pinned_renderer_mip_targets.front();
            prosper::test::unpin_persistent_color_target(
                oldest.id, oldest.width, oldest.height, oldest.format);
            pinned_renderer_mip_targets.erase(pinned_renderer_mip_targets.begin());
            static std::atomic<uint64_t> overflows{0};
            const uint64_t occurrence = overflows.fetch_add(1) + 1;
            if (prosper::diag_should_print(occurrence))
                fprintf(stderr,
                        "[renderer-mips] pending producer cap reached #%llu; "
                        "released unconsumed target=0x%llx extent=%ux%u\n",
                        (unsigned long long)occurrence,
                        (unsigned long long)oldest.id, oldest.width, oldest.height);
        }
        if (prosper::test::pin_persistent_color_target(id, width, height, format)) {
            pinned_renderer_mip_targets.push_back({id, width, height, format, false});
            return;
        }
        // The target was proven GPU-valid immediately before this call, so a failed pin
        // means frontend/backend cache identity drift. Report it even without diagnostics:
        // silently continuing would recreate the missing-middle-mip corruption this pin
        // exists to prevent.
        static std::atomic<uint64_t> failures{0};
        const uint64_t occurrence = failures.fetch_add(1) + 1;
        if (prosper::diag_should_print(occurrence))
            fprintf(stderr,
                    "[renderer-mips] submit-retention pin FAILED #%llu: "
                    "target=0x%llx extent=%ux%u format=%d\n",
                    (unsigned long long)occurrence, (unsigned long long)id,
                    width, height, (int)format);
    };
    // PER-TARGET RTT: a real frame is a sequence of passes, each rendering into a specific
    // color target (CB_COLOR0_BASE), and a final composite pass SAMPLES the earlier targets.
    // The single-framebuffer path below flattens ALL draws into one image and caches it under
    // only the FIRST item's base — so a draw that targets a different base never gets cached
    // under its OWN address, and a later composite that samples that address misses -> black.
    // Here we group items by color0_base (first-appearance order), render each group into its
    // own framebuffer, and cache each under its base. Because groups render in order and
    // build_bds re-reads g_rtt, a composite group that samples a scene group rendered earlier
    // THIS submit now hits — closing the intra-submit multi-pass gap (and cross-submit too,
    // since g_rtt persists). The final (last) group's pixels — the composite — are presented.
    // PASSES, not groups (#300): a real frame renders in SUBMIT ORDER as A->B->A->B... (a
    // ping-pong RT chain, each pass sampling the previous target). Collapsing all draws with
    // the same color0_base into one group loses that order — an A-pass that samples B then
    // renders before B exists, reading a stale/empty B -> black cascade (and a 4-deep UE4
    // post chain propagates nothing). So split items into CONTIGUOUS same-target runs below
    // and render each in order, caching its RT immediately so the next pass sampling it hits
    // fresh pixels. (Cross-submit persistence via g_rtt is unchanged.)
    // Flip-anchored present selection: the correct frame is whatever the guest FLIPS to
    // screen, i.e. the RTT group whose color-target VA is a registered VideoOut buffer
    // (preferring the CURRENT front buffer — the in-stream SetFlip fires during the Dcb
    // fold, before this render, so present_front_index() is this frame's scanout choice).
    const int      vo_n     = prosper_vo_buffer_count();
    const int      vo_front = prosper::gpu::present_front_index();
    const uint64_t front_va = vo_front >= 0 ? prosper_vo_buffer_addr(vo_front) : 0;
    if (rtt_log) {
        fprintf(stderr, "[rtt] flip state: front=%d va=0x%llx of %d registered:",
                vo_front, (unsigned long long)front_va, vo_n);
        for (int i = 0; i < vo_n && i < 8; i++)
            fprintf(stderr, " [%d]=0x%llx", i, (unsigned long long)prosper_vo_buffer_addr(i));
        fprintf(stderr, "\n");
    }
    std::shared_ptr<const std::vector<uint8_t>> px_front;
    std::shared_ptr<const std::vector<uint8_t>> px_vo;
    std::shared_ptr<const std::vector<uint8_t>> px_last;
    // px_front_*/px_vo_*/px_last_* (each candidate's own target extent, and px_last's
    // base) are declared at callback scope above so the final-span report can name which
    // sources this submit actually offered.
    // Render-target persistence (default on; PROSPER_RTT_NOSEED reverts to per-pass blue
    // clear): seed each group's framebuffer with the pixels last rendered into that SAME
    // target VA, so a pass that draws into an already-written target (UE4's UI pass onto
    // the backbuffer, incremental HUD updates) composites OVER the earlier content instead
    // of starting from the diagnostic clear. Real RT memory persists exactly this way.
    static const bool seed_rtt = getenv("PROSPER_RTT_NOSEED") == nullptr;
    const auto& no_seed_targets = rtt_no_seed_target_selector();
    const auto seed_target = [&](uint64_t target) {
        return seed_rtt && !(no_seed_targets.configured &&
            no_seed_targets.includes(target));
    };
    // Pass grouping and same-pass feedback detection must not disagree about what an
    // active binding is, so both go through frontends/shared/rtt/mrt_binding.hpp. These
    // were duplicated lambdas; a second, looser copy in the feedback path classified
    // stale named state as a live binding (#2550 review).
    auto color_binding = [](const prosper::gpu::DrawItem& draw, uint32_t slot) {
        return prosper::frontend::mrt_color_binding(draw, slot);
    };
    auto active_format = [](const prosper::gpu::DrawItem& draw, uint32_t slot) {
        return prosper::test::backend_color_format(static_cast<VkFormat>(
            prosper::frontend::mrt_raw_format(draw, slot)));
    };
    auto active_color = [](const prosper::gpu::DrawItem& draw, uint32_t slot) {
        return prosper::frontend::mrt_active_color(draw, slot, mrt_format_defined);
    };
    auto active_color_count = [](const prosper::gpu::DrawItem& draw) {
        return prosper::frontend::mrt_active_color_count(draw, mrt_format_defined);
    };
    // #3026 -- the slot-0/1 alias mirror, checked HERE, at the consumer.
    //
    // `DrawItem::color_targets[0]`/`[1]` mirror the named `color0_*`/`color1_*` triples
    // or are absent; every producer honours that and nothing enforced it. The moment
    // one does not, the readers below answer differently: this loop's pass target is
    // the RAW named field (`base`, and `native_w`/`native_h` for the framebuffer
    // extent), grouping is named-first (`mrt_pass_color_binding`), and the
    // active-binding rule behind the attachment count and feedback detection is
    // array-first (`mrt_color_binding`). #3023 is what one such disagreement cost: two
    // draws rendering to different addresses grouped into one pass, and the second
    // draw's surface was discarded with no pass, no published pixels and no diagnostic.
    //
    // It is checked here and not beside the assignments that establish it because an
    // assertion at `realize_draw_item`'s success exit would sit two lines below its own
    // mirror and could never fail. This is the place a divergence would have had to
    // survive to matter.
    //
    // Report-only, deliberately. Which representation should win is undecided, and
    // deciding it here would change the surface the renderer renders to.
    for (const auto& alias_draw : items) {
        prosper::frontend::mrt_check_color_alias_mirror(
            alias_draw,
            [](uint32_t slot,
               const prosper::gpu::DrawItem::ColorTargetBinding& carried,
               const prosper::gpu::DrawItem::ColorTargetBinding& named,
               uint64_t ordinal) {
                static const bool alias_log =
                    PROSPER_ENV_VALUE("PROSPER_MRT_ALIAS_LOG") != nullptr;
                if (ordinal > 8 && !alias_log) return;
                fprintf(stderr,
                        "[mrt-alias] c%u array=0x%llx %ux%u vs named=0x%llx %ux%u -- "
                        "the pass renders to the NAMED surface; grouping, feedback and "
                        "the attachment count may read the other one (#3026) x%llu%s\n",
                        slot, (unsigned long long)carried.base, carried.width,
                        carried.height, (unsigned long long)named.base, named.width,
                        named.height, (unsigned long long)ordinal,
                        ordinal == 8 && !alias_log
                            ? "  (further reports need PROSPER_MRT_ALIAS_LOG=1)" : "");
            });
    }
    size_t pass_i = 0;
    prosper::test::BackendSubmissionBatch backend_submission;
    if (timing_enabled)
        pending_timing.pass_head_ms += std::chrono::duration<double, std::milli>(
            RenderClock::now() - pass_timing_start).count();
    const auto pass_loop_start = timing_enabled
        ? RenderClock::now() : RenderClock::time_point{};
    const double pass_loop_measured_before = timing_enabled
        ? pending_timing.build_resources_ms + pending_timing.backend_ms : 0.0;
    while (pass_i < items.size()) {
        const auto group_start = timing_enabled
            ? RenderClock::now() : RenderClock::time_point{};
        // Counted at the TOP, before any `continue` can skip it -- so the difference
        // from pass_groups is exactly the number of iterations that returned without
        // rendering, which is the population the missing time must belong to.
        if (timing_enabled) ++pending_timing.pass_groups_seen;
        uint64_t base = items[pass_i].color0_base;
        const uint32_t requested_color_count = active_color_count(items[pass_i]);
        // PROSPER_MRT_CENSUS=1 — per slot, why an attachment did or did not become
        // active. A slot needs all three of base, a known format, and a non-zero write
        // mask; if any is absent the attachment is dropped silently and the draws that
        // wrote it produce nothing. On GTA V only c1 is ever published, while an exact
        // draw census shows 122,028 of 131,072 draws binding a colour target at SLOT 4 --
        // so one of these three is missing for that slot and nothing said which.
        if (PROSPER_ENV_ON("PROSPER_MRT_CENSUS")) {
            struct SlotCensus {
                std::atomic<uint64_t> seen{0}, has_base{0}, has_format{0},
                    has_mask{0}, active{0};
            };
            static std::array<SlotCensus, prosper::gpu::kColorTargetCount> census;
            static std::atomic<uint64_t> groups{0};
            const auto& d = items[pass_i];
            for (uint32_t slot = 0; slot < prosper::gpu::kColorTargetCount; ++slot) {
                auto& c = census[slot];
                c.seen.fetch_add(1, std::memory_order_relaxed);
                if (color_binding(d, slot).base)
                    c.has_base.fetch_add(1, std::memory_order_relaxed);
                if (active_format(d, slot) != VK_FORMAT_UNDEFINED)
                    c.has_format.fetch_add(1, std::memory_order_relaxed);
                if (d.ps.color_targets[slot].write_mask ||
                    (slot == 0 && d.ps.color_write_mask) ||
                    (slot == 1 && d.ps.color1_write_mask))
                    c.has_mask.fetch_add(1, std::memory_order_relaxed);
                // The per-slot mask the DrawItem CARRIES, against the one its own two
                // mask registers IMPLY. render_state.cpp computes the first as
                // (cb_target_mask & cb_shader_mask) >> (slot*4) & 0xf, so a disagreement
                // means the per-slot array did not survive the path that built this
                // DrawItem -- and the array is what decides whether the attachment
                // exists at all.
                {
                    const uint32_t implied =
                        ((d.ps.cb_target_mask & d.ps.cb_shader_mask) >> (slot * 4u))
                        & 0xfu;
                    const uint32_t carried = d.ps.color_targets[slot].write_mask;
                    if (implied != carried && color_binding(d, slot).base) {
                        static std::mutex disagree_mutex;
                        static std::map<std::tuple<uint32_t, uint32_t, uint32_t>,
                                        uint64_t> disagree;
                        std::lock_guard lock(disagree_mutex);
                        if (disagree.size() < 48)
                            ++disagree[{slot, implied, carried}];
                        static uint64_t n = 0;
                        if (++n % 8192 == 0) {
                            fprintf(stderr,
                                    "[mrt-mask-disagree] per-slot write_mask carried by "
                                    "the DrawItem vs implied by its own registers:\n");
                            for (const auto& e : disagree)
                                fprintf(stderr,
                                        "[mrt-mask-disagree]   c%u implied=0x%x "
                                        "carried=0x%x  x%llu\n",
                                        std::get<0>(e.first), std::get<1>(e.first),
                                        std::get<2>(e.first),
                                        (unsigned long long)e.second);
                        }
                    }
                }
                if (active_color(d, slot))
                    c.active.fetch_add(1, std::memory_order_relaxed);
            }
            // WHICH of the two masks is narrow. The write mask is
            // cb_target_mask & cb_shader_mask, and both default to 0xffffffff when the
            // register was never seen -- so a zero nibble means a register really was
            // programmed narrow, and the fix differs completely depending on which.
            // Histogrammed only over groups where slot 4 HAS a base, i.e. exactly the
            // population where the dropped attachment matters.
            if (color_binding(d, 4).base) {
                static std::mutex mask_mutex;
                static std::map<std::pair<uint32_t, uint32_t>, uint64_t> masks;
                std::lock_guard lock(mask_mutex);
                if (masks.size() < 64)
                    ++masks[{d.ps.cb_target_mask, d.ps.cb_shader_mask}];
                static uint64_t reported = 0;
                if (++reported % 4096 == 0) {
                    fprintf(stderr, "[mrt-census] masks where c4 has a base:\n");
                    for (const auto& e : masks)
                        fprintf(stderr,
                                "[mrt-census]   cb_target_mask=0x%08x "
                                "cb_shader_mask=0x%08x -> effective=0x%08x  x%llu\n",
                                e.first.first, e.first.second,
                                e.first.first & e.first.second,
                                (unsigned long long)e.second);
                }
            }
            // The SHAPE of each distinct pass: its eight slot bases next to the two
            // mask registers. The aggregate above says how often a slot activates; it
            // cannot say which surfaces a given pass meant to write, which is what
            // "buffer X is sampled 23 times and never written" actually needs.
            {
                static std::mutex shape_mutex;
                static std::map<std::array<uint64_t, 10>, uint64_t> shapes;
                std::array<uint64_t, 10> shape{};
                for (uint32_t slot = 0; slot < 8u; ++slot)
                    shape[slot] = color_binding(d, slot).base;
                shape[8] = d.ps.cb_target_mask;
                shape[9] = d.ps.cb_shader_mask;
                // PROSPER_MRT_SHAPE_FOR=0xADDR[,...] restricts recording to passes
                // whose slot-0 base is named. Without it the map fills with startup and
                // scanout passes long before the gameplay G-buffer pass appears, and an
                // unfiltered top-10 is then a list of the most FREQUENT shapes rather
                // than the ones being asked about.
                static const std::string shape_for =
                    PROSPER_ENV_VALUE("PROSPER_MRT_SHAPE_FOR")
                        ? PROSPER_ENV_VALUE("PROSPER_MRT_SHAPE_FOR") : "";
                bool shape_wanted = shape_for.empty();
                if (!shape_wanted && shape[0]) {
                    char needle[24];
                    std::snprintf(needle, sizeof needle, "0x%llx",
                                  (unsigned long long)shape[0]);
                    shape_wanted = shape_for.find(needle) != std::string::npos;
                }
                if (!shape_wanted) goto shape_done;
                {
                std::lock_guard lock(shape_mutex);
                if (shapes.size() < 96 || shapes.count(shape)) ++shapes[shape];
                static uint64_t shape_reports = 0;
                if (++shape_reports % (shape_for.empty() ? 8192 : 256) == 0) {
                    std::vector<std::pair<uint64_t, std::array<uint64_t, 10>>> ranked;
                    for (const auto& e : shapes) ranked.push_back({e.second, e.first});
                    std::sort(ranked.begin(), ranked.end(),
                              [](const auto& a, const auto& b) {
                                  return a.first > b.first; });
                    fprintf(stderr, "[mrt-shape] %zu distinct pass shapes\n",
                            shapes.size());
                    for (size_t i = 0; i < ranked.size() && i < 10; ++i) {
                        fprintf(stderr, "[mrt-shape]   x%-6llu tmask=0x%08llx "
                                "smask=0x%08llx bases:",
                                (unsigned long long)ranked[i].first,
                                (unsigned long long)ranked[i].second[8],
                                (unsigned long long)ranked[i].second[9]);
                        for (uint32_t slot = 0; slot < 8u; ++slot)
                            if (ranked[i].second[slot])
                                fprintf(stderr, " c%u=0x%llx", slot,
                                        (unsigned long long)ranked[i].second[slot]);
                        fprintf(stderr, "\n");
                    }
                }
                }
                shape_done: ;
            }
            const uint64_t g = groups.fetch_add(1) + 1;
            if ((g & (g - 1)) == 0 && g >= 4096) {
                fprintf(stderr, "[mrt-census] pass groups=%llu\n",
                        (unsigned long long)g);
                for (uint32_t slot = 0; slot < prosper::gpu::kColorTargetCount; ++slot)
                    fprintf(stderr,
                            "[mrt-census]   c%u base=%llu format=%llu mask=%llu "
                            "ACTIVE=%llu\n",
                            slot,
                            (unsigned long long)census[slot].has_base.load(),
                            (unsigned long long)census[slot].has_format.load(),
                            (unsigned long long)census[slot].has_mask.load(),
                            (unsigned long long)census[slot].active.load());
            }
        }
        std::array<uint64_t, prosper::gpu::kColorTargetCount> pass_bases{};
        std::array<VkFormat, prosper::gpu::kColorTargetCount> pass_formats{};
        for (uint32_t slot = 0; slot < requested_color_count; ++slot) {
            pass_bases[slot] = slot ? active_color(items[pass_i], slot) : base;
            pass_formats[slot] = active_format(items[pass_i], slot);
        }
        const uint64_t base1 = pass_bases[1];
        const VkFormat format0 = pass_formats[0];
        const VkFormat format1 = base1
            ? pass_formats[1] : VK_FORMAT_UNDEFINED;
        // The DEPTH/STENCIL attachment is part of a pass's target identity, not just its
        // colour attachments.
        //
        // A render pass has ONE depth attachment, and the backend picks one cached DS
        // image for the whole grouped call from its first meaningful draw. Grouping on
        // colour alone therefore let a single call span draws that name different DS
        // surfaces -- or, for a layered surface, different DB_DEPTH_VIEW SLICES of one
        // allocation. Every such draw then rendered into the first face the call
        // happened to select.
        //
        // Measured on GTA V (Codex, #2542): one grouped call crosses DB_DEPTH_VIEW from
        // slice 0 to slice 1 at draw 65, so several guest cube faces were being rendered
        // into one host face. That also makes any "the cube has six valid faces"
        // measurement void: the handles existed, their contents did not correspond to
        // six guest faces.
        auto ds_identity = [](const prosper::gpu::DrawItem& draw) {
            return std::tuple(draw.ps.depth_read_base, draw.ps.depth_write_base,
                              draw.ps.stencil_read_base, draw.ps.stencil_write_base,
                              draw.ps.htile_data_base, draw.ps.db_depth_size_xy,
                              prosper::test::ds_depth_view_slice_start(
                                  draw.ps.db_depth_view));
        };
        const prosper::gpu::DrawItem& pass_head = items[pass_i];
        const auto ds0 = ds_identity(pass_head);
        std::vector<const prosper::gpu::DrawItem*> pass;
        auto same_targets = [&](const prosper::gpu::DrawItem& draw) {
            if (!prosper::frontend::mrt_same_color_pass(
                    pass_head, draw, mrt_format_defined, active_format))
                return false;
            // One 3D allocation can be written through several bounded slice views.
            // Grouping them into one framebuffer would route later draws to the
            // first view even though their guest addresses and 2D extents match.
            const auto& first_view = pass_head.color_targets[0];
            const auto& next_view = draw.color_targets[0];
            if (std::tuple(first_view.selected_mip_depth,
                           first_view.first_slice, first_view.slice_count) !=
                std::tuple(next_view.selected_mip_depth,
                           next_view.first_slice, next_view.slice_count))
                return false;
            if (first_view.selected_mip_depth &&
                std::tuple(first_view.tile_mode, first_view.mip_level,
                           first_view.in_mip_tail,
                           first_view.native_layout_known) !=
                std::tuple(next_view.tile_mode, next_view.mip_level,
                           next_view.in_mip_tail,
                           next_view.native_layout_known))
                return false;
            if (ds_identity(draw) != ds0) return false;
            return true;
        };
        // A MODE=RESOLVE draw is answered by a COPY below, not by a render, so its
        // pass identity is the source/destination pair that copy acts with rather than
        // the attachment set `same_targets` compares. `mrt_same_resolve_pass` carries
        // that whole rule -- keeping a resolve out of a scene group, and keeping two
        // resolves with different destinations out of each other's (#3025) -- and its
        // header states why each half exists.
        // Array snapshots are materialized by build_bds BEFORE this group's
        // commands are recorded. Flush a preceding writer as its own group before
        // preparing an aliased consumer, even when all attachment identities match.
        // The bridge can then submit the pending producer batch before reading it.
        bool pass_writes_depth = false;
        const auto samples_pass_depth_array = [&](const prosper::gpu::DrawItem& draw) {
            const auto aliases = [&](const prosper::gpu::ShaderResourceTable* table) {
                if (!table) return false;
                for (const auto& resource : table->resources) {
                    if (resource.cls != RC::Texture || resource.img_dim != 5u ||
                        resource.depth <= 1u ||
                        resource.format != prosper::gpu::DataFormat::Float32 ||
                        !resource.gpu_addr)
                        continue;
                    if (resource.gpu_addr == pass_head.ps.depth_read_base ||
                        resource.gpu_addr == pass_head.ps.depth_write_base)
                        return true;
                }
                return false;
            };
            return aliases(draw.vrt.get()) || aliases(draw.prt.get());
        };
        // Same four conditions in the same short-circuit order as before; the only
        // change is that the loop now records WHICH one ended the pass. Pass length
        // sets what every fixed per-pass cost actually costs over a run, and the
        // aggregate cannot distinguish an irreducible target change from an
        // over-strict predicate. See pass_break_census.hpp.
        const size_t pass_begin = pass_i;
        prosper::gpu::PassBreak pass_break = prosper::gpu::PassBreak::EndOfItems;
        while (true) {
            if (pass_i >= items.size()) {
                pass_break = prosper::gpu::PassBreak::EndOfItems; break;
            }
            if (!same_targets(items[pass_i])) {
                pass_break = prosper::gpu::PassBreak::TargetsChanged; break;
            }
            if (!prosper::frontend::mrt_same_resolve_pass(pass_head, items[pass_i])) {
                pass_break = prosper::gpu::PassBreak::MrtResolveDiffers; break;
            }
            const auto& draw = items[pass_i];
            if (pass_writes_depth && samples_pass_depth_array(draw)) {
                pass_break = prosper::gpu::PassBreak::DepthFeedback; break;
            }
            pass.push_back(&draw); ++pass_i;
            pass_writes_depth |= prosper::test::persistent_ds_pass_may_write_depth(
                draw.ps.depth_clear_enable, draw.ps.depth_test_enable,
                draw.ps.depth_write_enable, draw.ps.depth_compare_op);
        }
        prosper::gpu::pass_break_census().note_break(pass_break, pass_i - pass_begin);

        // CB_COLOR_CONTROL.MODE=RESOLVE(3): the guest resolves an MSAA color0 surface into a
        // single-sample color1 destination (Blue Prince PPSA25009 resolves its 4x-MSAA scene
        // this way). prosper renders single-sample, so the resolve is a straight copy of the
        // already-rendered color0 surface into color1. Use the RAW color1_base, not
        // active_color1(): a fixed-function resolve exports nothing, so its color1_write_mask
        // is 0 and active_color1 would report no destination. Without this the resolved surface
        // the display later samples never receives the scene and the frame is a uniform fill.
        // Census every MODE=RESOLVE pass: source, destination and whether the
        // destination is one prosper can even express. prosper takes the destination
        // from `color1_base` alone, so a guest resolving into any other slot is
        // invisible to the resolve path -- a surface filled that way would read as
        // "written by nothing" in every other census.
        ResolveCensusContext note_resolve_census_ctx{
            .pass = pass};
        note_resolve_census(note_resolve_census_ctx);
        static const bool no_resolve = getenv("PROSPER_NO_RESOLVE") != nullptr;
        if (!pass.empty() && pass.front()->ps.cb_resolve && !no_resolve) {
            ResolvePassContext resolve_ctx{
                .batch_backend_submits = batch_backend_submits,
                .items = items,
                .pending_timing = pending_timing,
                .timing_enabled = timing_enabled,
                .g_rtt = g_rtt,
                .pass_i = pass_i,
                .backend_submission = backend_submission,
                .pass = pass};
            resolve_pass(resolve_ctx);
            continue;   // resolve is a copy, not an ordinary-draw render
        }

        // Gen5 render-target extent (#526). Large scene/scanout surfaces retain the
        // configured VideoOut render scale; small offscreen targets render at native
        // resolution so lookup textures (Messenger's 1024x32 grading LUT) preserve every
        // texel. The viewport was already scaled for the global w/h framebuffer by
        // execute_gpustate, so correct it from that scale to this pass-local scale.
        uint32_t native_w = pass.empty() ? 0u : pass.front()->color0_width;
        uint32_t native_h = pass.empty() ? 0u : pass.front()->color0_height;
        uint32_t gw = w, gh = h;
        const uint32_t present_w = prosper::gpu::present_width();
        const uint32_t present_h = prosper::gpu::present_height();
        bool is_vo = false;
        for (int i = 0; i < vo_n && !is_vo; i++)
            is_vo = base && base == prosper_vo_buffer_addr(i);
        // VideoOut registration is the visible scanout contract. CB_COLOR0_ATTRIB2 can
        // describe an overallocated backing surface (Terminator reports 4096x4096 for a
        // 1920x1080 scanout), which must not become the persistent target extent or the
        // final cache lookup will reject the rendered frame.
        if (is_vo && present_w && present_h) {
            native_w = present_w;
            native_h = present_h;
        }
        // A depth prepass may bind a tiny/dummy color target with CB_TARGET_MASK=0 while
        // rasterizing the full-size guest depth surface. Keying persistent DS from that
        // irrelevant color extent creates a small depth image that the later lighting pass
        // cannot reuse. Recover the attachment extent from the viewport, but only when the
        // complete inferred extent fits the known presentation surface. Viewport coordinates
        // position rasterization and are not allocation metadata; accepting translated or
        // otherwise oversized coordinates here previously created persistent images as large
        // as 16384x16384 and exhausted the host while Dead Cells loaded its first level.
        const bool color_disabled = !pass.empty() && std::all_of(
            pass.begin(), pass.end(), [](const auto* draw) {
                for (uint32_t slot = 0; slot < prosper::gpu::kColorTargetCount; ++slot)
                    if (prosper::frontend::mrt_write_mask(*draw, slot)) return false;
                return true;
            });
        const bool uses_ds = std::any_of(pass.begin(), pass.end(), [](const auto* draw) {
            return draw->ps.depth_test_enable || draw->ps.depth_write_enable ||
                   draw->ps.depth_clear_enable || draw->ps.stencil_enable ||
                   draw->ps.stencil_clear_enable;
        });
        // A masked color attachment does not determine the depth allocation's size.
        // DB_DEPTH_SIZE_XY is the guest extent; both clear and caster passes must name
        // the same persistent depth image even when their inactive color bindings differ.
        // Resolved/captured state does not retain this register's presence flag. Zero
        // therefore remains ambiguous (absent versus explicit 1x1) and keeps the legacy
        // fallback. Pass grouping above keeps distinct DB extents separate.
        const auto* depth_state = pass.empty() ? nullptr : &pass.front()->ps;
        const bool explicit_depth_extent = depth_state && color_disabled && uses_ds &&
            (depth_state->depth_read_base || depth_state->depth_write_base ||
             depth_state->stencil_read_base || depth_state->stencil_write_base) &&
            depth_state->db_depth_size_xy != 0;
        if (explicit_depth_extent) {
            native_w = PM4_FIELD(depth_state->db_depth_size_xy, DB_DEPTH_SIZE_XY, X_MAX) + 1u;
            native_h = PM4_FIELD(depth_state->db_depth_size_xy, DB_DEPTH_SIZE_XY, Y_MAX) + 1u;
            // All color writes are disabled. The backend still needs a dummy color
            // attachment, but resizing it to the depth extent must not create or
            // publish a new color authority under an inactive guest CB address.
            // Zero pass identities select transient attachments and bypass color
            // seeding, retained-target publication and scanout selection below.
            base = 0;
            pass_bases.fill(0);
            is_vo = false;
        }
        if (color_disabled && uses_ds && !explicit_depth_extent && w && h) {
            float viewport_x = 0.0f, viewport_y = 0.0f;
            for (const auto* draw : pass) {
                if (!draw->ps.has_viewport) continue;
                const float x1 = draw->ps.viewport_x;
                const float x2 = x1 + draw->ps.viewport_w;
                const float y1 = draw->ps.viewport_y;
                const float y2 = y1 + draw->ps.viewport_h;
                if (std::isfinite(x1) && std::isfinite(x2))
                    viewport_x = std::max(viewport_x, std::max(std::fabs(x1), std::fabs(x2)));
                if (std::isfinite(y1) && std::isfinite(y2))
                    viewport_y = std::max(viewport_y, std::max(std::fabs(y1), std::fabs(y2)));
            }
            const uint32_t max_native_w = present_w ? present_w : w;
            const uint32_t max_native_h = present_h ? present_h : h;
            const uint64_t viewport_native_w = static_cast<uint64_t>(
                std::ceil(viewport_x * static_cast<float>(max_native_w) / w));
            const uint64_t viewport_native_h = static_cast<uint64_t>(
                std::ceil(viewport_y * static_cast<float>(max_native_h) / h));
            // Depth-only surfaces legitimately exceed the presentation extent: Blue
            // Prince's directional-shadow cascades render translated 512x512/1024x1024
            // viewports into 2048x2048 and 4096x4096 atlases (#1275). Capping at the
            // presentation surface collapsed those atlases to 1x1, erasing every shadow.
            // The per-axis bound (each axis against its own presentation axis, or 4096)
            // keeps the Dead Cells pathology (translated viewports inferring
            // 16384x16384, the reason this cap exists) rejected and fail-visible.
            const bool viewport_extent_valid = viewport_native_w && viewport_native_h &&
                viewport_native_w <= std::max<uint64_t>(4096u, max_native_w) &&
                viewport_native_h <= std::max<uint64_t>(4096u, max_native_h);
            // Log the UNDECIDABLE case too. Gating this on a non-zero derived extent hid
            // the only outcome that silently changes the DS identity: a depth-only pass
            // whose draws carry no viewport register derives 0x0, falls through to the
            // global frame extent, and mints a second cache entry for a surface that
            // already has a correctly-sized one. A diagnostic that prints only when the
            // inference succeeded cannot report the inference not happening.
            if (PROSPER_ENV_ON("PROSPER_DSLOG")) {
                const size_t with_viewport = static_cast<size_t>(std::count_if(
                    pass.begin(), pass.end(),
                    [](const auto* draw) { return draw->ps.has_viewport; }));
                fprintf(stderr,
                        "[ds] viewport-derived extent %llux%llu (presentation %ux%u, "
                        "%zu/%zu draws with viewport) -> %s\n",
                        (unsigned long long)viewport_native_w,
                        (unsigned long long)viewport_native_h,
                        max_native_w, max_native_h, with_viewport, pass.size(),
                        viewport_extent_valid ? "accept"
                            : (viewport_native_w || viewport_native_h) ? "reject"
                                                                       : "undecidable");
            }
            if (viewport_extent_valid) {
                native_w = std::max(native_w, static_cast<uint32_t>(viewport_native_w));
                native_h = std::max(native_h, static_cast<uint32_t>(viewport_native_h));
            }
        }
        if (native_w && native_h) {
            uint64_t native_pixels = (uint64_t)native_w * native_h;
            uint64_t global_pixels = (uint64_t)w * h;
            if (native_pixels <= global_pixels) {
                gw = native_w; gh = native_h;
            } else if (present_w && present_h) {
                gw = std::max(1u, (uint32_t)(((uint64_t)native_w * w + present_w / 2) / present_w));
                gh = std::max(1u, (uint32_t)(((uint64_t)native_h * h + present_h / 2) / present_h));
            } else if (color_disabled && uses_ds) {
                // No presentation extent (offline replay): render scale is 1, so a
                // depth-only surface's viewport-derived extent is exact. Falling back
                // to the global w/h here silently truncated over-presentation atlases —
                // Blue Prince's 2048x2048 shadow atlas became a 1920x1080 DS image
                // whose sampled taps then missed the bridge (#1275). Color passes keep
                // the historical capture-extent truncation (their over-allocated
                // CB_COLOR0_ATTRIB2 backings are a different, hash-pinned contract).
                gw = native_w; gh = native_h;
            }
        }
        std::vector<prosper::gpu::DrawItem> adjusted;
        std::vector<const prosper::gpu::DrawItem*> render_pass = pass;
        if (native_w && native_h && present_w && present_h && w && h) {
            const float ax = ((float)gw / native_w) / ((float)w / present_w);
            const float ay = ((float)gh / native_h) / ((float)h / present_h);
            if (ax != 1.0f || ay != 1.0f) {
                adjusted.reserve(pass.size()); render_pass.clear(); render_pass.reserve(pass.size());
                for (const auto* src : pass) {
                    adjusted.push_back(*src);
                    auto& item = adjusted.back();
                    prosper::gpu::scale_resolved_render_area(item.ps, ax, ay);
                    render_pass.push_back(&item);
                }
            }
        }
        const uint8_t* seed = nullptr;
        const float* retained_uniform_clear = nullptr;
        bool gpu_seed_available = false;
        const bool seed_rtt0 = seed_target(base);
        const VkFormat pass_format = format0;
        const auto& primary_volume_view = pass.front()->color_targets[0];
        const uint32_t producer_volume_depth =
            primary_volume_view.selected_mip_depth;
        g_ever_volume_target |= producer_volume_depth != 0u;
        const uint32_t volume_bpp =
            prosper::test::backend_color_bytes_per_pixel(format0);
        const uint32_t volume_native_w = native_w ? native_w : gw;
        const uint32_t volume_native_h = native_h ? native_h : gh;
        const bool producer_volume_layout_supported = producer_volume_depth &&
            primary_volume_view.native_layout_known &&
            !primary_volume_view.mip_level &&
            !primary_volume_view.in_mip_tail &&
            prosper::gpu::tile_mode_supports_volume(primary_volume_view.tile_mode);
        const uint64_t producer_volume_physical_bytes =
            producer_volume_layout_supported
                ? prosper::gpu::tiled_volume_bytes(
                      volume_native_w, volume_native_h, producer_volume_depth,
                      primary_volume_view.tile_mode, volume_bpp)
                : 0u;
        const bool producer_volume_footprint_proven =
            producer_volume_physical_bytes != 0u;
        const uint64_t producer_volume_guard_bytes =
            producer_volume_footprint_proven
                ? producer_volume_physical_bytes
                : prosper::frontend::live_rtt_color_footprint_bytes(
                      volume_native_w, volume_native_h,
                      producer_volume_depth, volume_bpp);
        uint32_t mrt_count = requested_color_count;
        if (PROSPER_ENV_ON("PROSPER_NO_MRT1") || PROSPER_ENV_ON("PROSPER_NO_MRT")) mrt_count = 1;
        if (!render_pass.empty()) {
            for (uint32_t slot = 1; slot < mrt_count; ++slot) {
                const auto binding = color_binding(*render_pass.front(), slot);
                // Sparse exports retain their native Location through dummy attachments.
                // Only a real target whose extent is KNOWN to differ from the pass
                // extent truncates the prefix. A 0x0 on either side means the guest's
                // CB_COLORn_ATTRIB2 was never seen, not a 0-pixel surface, and an
                // unmeasured extent is no evidence of a conflict — comparing it for
                // equality dropped the whole attachment silently (#2114).
                if (pass_bases[slot] && prosper::frontend::mrt_extent_conflicts(
                        binding.width, binding.height, native_w, native_h)) {
                    if (rtt_log)
                        fprintf(stderr,
                                "[rtt] truncate MRT prefix at c%u target=0x%llx "
                                "extent=%ux%u; MRT0 is %ux%u\n",
                                slot, (unsigned long long)pass_bases[slot],
                                binding.width, binding.height, native_w, native_h);
                    mrt_count = slot;
                    break;
                }
                // Fail-visible: name the attachment that was kept on missing extent
                // data, so an unmeasured extent is a reported state rather than a
                // silent one in either direction.
                if (rtt_log && pass_bases[slot] &&
                    (!prosper::frontend::mrt_extent_known(binding.width, binding.height) ||
                     !prosper::frontend::mrt_extent_known(native_w, native_h)))
                    fprintf(stderr,
                            "[rtt] keep MRT c%u target=0x%llx on unmeasured extent "
                            "%ux%u; MRT0 is %ux%u\n",
                            slot, (unsigned long long)pass_bases[slot],
                            binding.width, binding.height, native_w, native_h);
            }
        }
        const bool use_color1 = mrt_count > 1;
        const VkFormat pass_format1 = use_color1 ? format1 : VK_FORMAT_UNDEFINED;
        const size_t pass_bytes = static_cast<size_t>(gw) * gh *
            prosper::test::backend_color_bytes_per_pixel(pass_format);
        if (seed_rtt0 && base) { auto sit = g_rtt.find(base);
            gpu_seed_available = live_gpu_targets && sit != g_rtt.end() &&
                sit->second.gpu_valid && sit->second.w == gw && sit->second.h == gh &&
                sit->second.format == pass_format &&
                sit->second.volume_depth == producer_volume_depth &&
                prosper::test::find_persistent_color_target(
                    base, gw, gh, pass_format, true,
                    producer_volume_depth) != nullptr;
            if (!gpu_seed_available && sit != g_rtt.end() &&
                !producer_volume_depth && !sit->second.volume_depth &&
                sit->second.w == gw && sit->second.h == gh &&
                sit->second.format == pass_format && sit->second.rgba &&
                sit->second.rgba->size() == pass_bytes)
                seed = sit->second.rgba->data();
            if (!gpu_seed_available && !seed && sit != g_rtt.end() &&
                !producer_volume_depth && !sit->second.volume_depth &&
                sit->second.w == gw && sit->second.h == gh &&
                sit->second.format == pass_format &&
                sit->second.has_uniform_color)
                retained_uniform_clear = sit->second.uniform_color.data();
            // Gated seed-decision diagnostic: a pass that should LOAD prior target
            // content but silently falls back to its clear color erases everything the
            // earlier pass produced (an opaque-black clear wipes a transparent UI RT —
            // #320's dialogue overlay). Make the decision and its reason visible.
            if (rtt_log && !seed && !gpu_seed_available) {
                if (sit == g_rtt.end())
                    fprintf(stderr, "[rtt] seed miss target=0x%llx reason=no-entry\n",
                            (unsigned long long)base);
                else
                    fprintf(stderr,
                            "[rtt] seed miss target=0x%llx reason=mismatch "
                            "entry=%ux%u fmt=%d rgba=%zu want=%ux%u fmt=%d bytes=%zu\n",
                            (unsigned long long)base, sit->second.w, sit->second.h,
                            (int)sit->second.format,
                            sit->second.rgba ? sit->second.rgba->size() : (size_t)0,
                            gw, gh, (int)pass_format, pass_bytes);
            } }
        // A later pass in THIS batch that reads the target's CPU bytes cannot be served
        // lazily: the producing commands are still unsubmitted when it binds, so a
        // mid-batch readback would return stale pixels. Only the direct GPU bind (2D
        // texture, exact extent, not feedback, not storage) is batch-ordered; every
        // other same-batch consumer forces the eager readback. A later color attachment
        // LOAD is now direct for both MRT attachments and needs no CPU copy. An exact
        // 3D texture may also borrow this producer's complete retained volume; other
        // volume consumers need a guest publication path that is not yet implemented.
        // pass_i has advanced past the current pass, so scanned items are genuine.
        // Cross-batch consumers materialize on demand at bind/seed/compute/DMA time
        // (#1284).
        const auto inspect_later_consumers = [&](uint64_t target_base) {
            LaterTargetConsumers result;
            if (!live_gpu_targets || !target_base) return result;
            static const uint32_t render_scale = [] {
                const char* e = PROSPER_ENV_VALUE("PROSPER_RENDER_SCALE");
                const long v = e ? std::strtol(e, nullptr, 10) : 1;
                return v > 0 ? static_cast<uint32_t>(v) : 1u;
            }();
            for (size_t later = pass_i; later < items.size(); ++later) {
                auto inspect = [&](const prosper::gpu::ShaderResourceTable* table) {
                    if (!table) return;
                    for (const auto& resource : table->resources) {
                        if ((resource.cls != RC::Texture &&
                             resource.cls != RC::StorageImage) ||
                            resource.gpu_addr != target_base ||
                            (resource.img_dim == 2u &&
                             (!producer_volume_depth || target_base != base)))
                            continue;
                        const uint32_t rw = resource.width ? resource.width : 4u;
                        const uint32_t rh = resource.height ? resource.height : 4u;
                        bool same_pass_target =
                            items[later].color0_base == target_base;
                        for (uint32_t slot = 1;
                             slot < prosper::gpu::kColorTargetCount; ++slot)
                            same_pass_target |=
                                active_color(items[later], slot) == target_base;
                        const bool sampled_extent_compatible =
                            prosper::frontend::rtt_sampled_extent_compatible(
                                rw, rh, gw, gh, render_scale, false);
                        const bool sampled_shape = producer_volume_depth &&
                                                   target_base == base
                            ? resource.img_dim == 2u &&
                              resource.depth == producer_volume_depth &&
                              resource.sample_count == 1u &&
                              resource.declared_mip_levels == 1u &&
                              !resource.in_mip_tail
                            : prosper::frontend::rtt_single_layer_sample_shape(
                                  resource.img_dim, resource.depth,
                                  resource.sample_count);
                        if (sampled_shape && sampled_extent_compatible) {
                            result.sampled_exact = true;
                            result.feedback |= same_pass_target;
                        }
                        // Exact 2D color feedback is GPU-bindable: the backend copies
                        // the prior attachment version to a distinct sampled image before
                        // the render pass. Storage and dimension mismatches still
                        // require the CPU representation. An extent mismatch is an alias
                        // rather than a consumer of this target.
                        const bool direct_bindable =
                            resource.cls == RC::Texture && sampled_shape &&
                            (resource.img_dim != 2u || !same_pass_target);
                        if (!sampled_extent_compatible) {
                            result.extent_mismatches++;
                        } else if (!direct_bindable) {
                            result.cpu_needed = true;
                            result.storage_references +=
                                resource.cls == RC::StorageImage;
                            result.dimension_mismatches += !sampled_shape;
                            result.feedback_references += same_pass_target;
                            static const uint64_t diagnose_min_submit = [] {
                                const char* text = PROSPER_ENV_VALUE(
                                    "PROSPER_READBACK_WHY_MIN_SUBMIT");
                                return text ? std::strtoull(text, nullptr, 0) : 0ull;
                            }();
                            static std::atomic<uint64_t> diagnose_lines{0};
                            if (PROSPER_ENV_ON("PROSPER_READBACK_WHY") &&
                                static_cast<uint64_t>(g_this_submit) >=
                                    diagnose_min_submit &&
                                diagnose_lines.fetch_add(
                                    1, std::memory_order_relaxed) < 64) {
                                fprintf(stderr,
                                        "[readback-consumer] submit=%d target=0x%llx "
                                        "producer-pass=%zu consumer-draw=%zu "
                                        "class=%s dim=%u extent=%ux%u target-extent=%ux%u "
                                        "storage=%u dimension=%u extent-mismatch=%u "
                                        "feedback=%u\n",
                                        g_this_submit,
                                        (unsigned long long)target_base, pass_i - 1,
                                        later,
                                        resource.cls == RC::StorageImage
                                            ? "storage" : "texture",
                                        resource.img_dim, rw, rh, gw, gh,
                                        resource.cls == RC::StorageImage ? 1u : 0u,
                                        !sampled_shape ? 1u : 0u,
                                        !sampled_extent_compatible ? 1u : 0u,
                                        same_pass_target ? 1u : 0u);
                            }
                        }
                    }
                };
                inspect(items[later].vrt.get());
                inspect(items[later].prt.get());
            }
            return result;
        };
        const LaterTargetConsumers consumers0 = inspect_later_consumers(base);
        const LaterTargetConsumers consumers1 = use_color1
            ? inspect_later_consumers(base1) : LaterTargetConsumers{};
        std::array<LaterTargetConsumers, prosper::gpu::kColorTargetCount>
            consumers_slots{};
        for (uint32_t slot = 2; slot < mrt_count; ++slot)
            consumers_slots[slot] = inspect_later_consumers(pass_bases[slot]);
        const bool sampled_exact_later = consumers0.sampled_exact;
        const bool feedback_later = consumers0.feedback;
        const bool cpu_needed_same_batch = consumers0.cpu_needed;
        if (producer_volume_depth &&
            (!producer_volume_footprint_proven || !live_gpu_targets ||
             phase.authoritative_readback ||
             cpu_needed_same_batch)) {
            // No 3D guest publication or same-pass feedback snapshot exists yet, so
            // the renderer cannot produce this volume in a form a consumer here can
            // read. Decline the producer. With no image, the volume claims nothing and
            // consumers read guest memory, which lacks this pass's writes. The
            // alternative was worse: claiming the footprint dropped every draw that
            // samples it, which kept GTA V's menus black whenever live GPU targets are
            // off (PROSPER_DUMP_*, replay seeding, and PROSPER_GPU_CAPTURE before
            // #3895; #3890).
            prosper::test::invalidate_persistent_color_target(base);
            if (base)
                note_volume_producer_denied(base, g_rtt[base], gw, gh,
                                            producer_volume_depth, pass_format);
            static std::atomic<uint32_t> volume_refusals{0};
            if (volume_refusals.fetch_add(1, std::memory_order_relaxed) < 16u)
                std::fprintf(stderr,
                    "[render-volume] guest-observed or unsupported volume pass "
                    "target=0x%llx authoritative=%d cpu-consumer=%d live=%d "
                    "physical=%d mode=%u\n",
                    static_cast<unsigned long long>(base),
                    phase.authoritative_readback, cpu_needed_same_batch,
                    live_gpu_targets, producer_volume_footprint_proven,
                    primary_volume_view.tile_mode);
            continue;
        }
        static const bool defer_rtt_readback =
            !PROSPER_ENV_VALUE("PROSPER_NO_RTT_READBACK_DEFER");
        const bool rtt_defer_ok = defer_rtt_readback
            ? !cpu_needed_same_batch
            : (sampled_exact_later && !feedback_later);
        const bool rtt_defer_ok1 = defer_rtt_readback
            ? !consumers1.cpu_needed
            : (consumers1.sampled_exact && !consumers1.feedback);
        // Keep intermediate scanout spans GPU-resident too: they cannot publish until the
        // final callback, where the cache is materialized on demand if no later scanout
        // pass already requested CPU pixels. A same-submit DMA asks its producer span for
        // authoritative readback, and compute consumers use the lazy target reader above.
        // A pending one-shot capture reads back every pass's colour0 for its one
        // submit, so the publish candidates (front / scanout / last) that become the
        // capsule's output oracle are exactly the readback path's (#3895).
        const bool capture_cpu_output = prosper::gpu::gpu_capture_requires_cpu_output();
        const bool final_gpu_present = phase.final_span &&
            !phase.authoritative_readback &&
            prosper::frontend::gpu_present_allowed_during_capture(
                prosper::gpu::gpu_present_active(), capture_cpu_output);
        const bool defer_readback = live_gpu_targets && vo_n > 0 && base &&
            !phase.authoritative_readback &&
            prosper::frontend::pass_readback_deferral_allowed_during_capture(
                capture_cpu_output) &&
            ((is_vo && can_defer_scanout_readback(
                           phase.allows_deferred_scanout_readback(),
                           final_gpu_present,
                           defer_intermediate_scanout, cpu_needed_same_batch)) ||
             (!is_vo && (base != front_va || final_gpu_present) && rtt_defer_ok));
        const bool defer_readback1 = live_gpu_targets && vo_n > 0 && use_color1 &&
            base1 && base1 != base && !phase.authoritative_readback &&
            (base1 != front_va || final_gpu_present) && rtt_defer_ok1;
        std::array<bool, prosper::gpu::kColorTargetCount> defer_readback_slots{};
        for (uint32_t slot = 2; slot < mrt_count; ++slot) {
            const LaterTargetConsumers& consumers = consumers_slots[slot];
            const bool rtt_defer_ok_slot = defer_rtt_readback
                ? !consumers.cpu_needed
                : (consumers.sampled_exact && !consumers.feedback);
            const uint64_t slot_base = pass_bases[slot];
            defer_readback_slots[slot] = live_gpu_targets && vo_n > 0 &&
                slot_base && !phase.authoritative_readback &&
                (slot_base != front_va || final_gpu_present) && rtt_defer_ok_slot;
        }
        // PROSPER_READBACK_WHY (#1284): classify WHY each non-deferred pass takes the
        // synchronous CPU readback (75-79 ms/window on Blue Prince's Day One frame).
        // The dominant reason selects the next optimization; behavior unchanged.
        if (!defer_readback) {
            ReadbackWhyContext note_readback_why_ctx{
                .live_gpu_targets = live_gpu_targets,
                .phase = phase,
                .vo_n = vo_n,
                .front_va = front_va,
                .base = base,
                .pass_bases = pass_bases,
                .gw = gw,
                .gh = gh,
                .is_vo = is_vo,
                .pass_format = pass_format,
                .mrt_count = mrt_count,
                .consumers0 = consumers0,
                .sampled_exact_later = sampled_exact_later,
                .feedback_later = feedback_later,
                .cpu_needed_same_batch = cpu_needed_same_batch};
            note_readback_why(note_readback_why_ctx);
        }
        const uint8_t* seed1 = nullptr;
        const float* retained_uniform_clear1 = nullptr;
        bool gpu_seed1_available = false;
        const bool seed_rtt1 = seed_target(base1);
        if (seed_rtt1 && use_color1) { auto sit = g_rtt.find(base1);
            gpu_seed1_available = live_gpu_targets && sit != g_rtt.end() &&
                sit->second.gpu_valid &&
                sit->second.w == gw && sit->second.h == gh &&
                sit->second.format == pass_format1 &&
                prosper::test::find_persistent_color_target(
                    base1, gw, gh, pass_format1) != nullptr;
            if (!gpu_seed1_available && sit != g_rtt.end() &&
                sit->second.w == gw && sit->second.h == gh &&
                sit->second.format == pass_format1 && sit->second.rgba &&
                sit->second.rgba->size() == static_cast<size_t>(gw) * gh *
                    prosper::test::backend_color_bytes_per_pixel(pass_format1))
                seed1 = sit->second.rgba->data();
            if (!gpu_seed1_available && !seed1 && sit != g_rtt.end() &&
                sit->second.w == gw && sit->second.h == gh &&
                sit->second.format == pass_format1 &&
                sit->second.has_uniform_color)
                retained_uniform_clear1 = sit->second.uniform_color.data(); }
        if (base1 && !use_color1 && rtt_log) {
            const auto* first = render_pass.empty() ? nullptr : render_pass.front();
            fprintf(stderr,
                    "[rtt] skip MRT1 target=0x%llx extent=%ux%u; MRT0 is %ux%u\n",
                    (unsigned long long)base1,
                    first ? first->color1_width : 0u,
                    first ? first->color1_height : 0u, native_w, native_h);
        }
        // #3891: one clock pair per pass group feeds the always-on ledger too.
        const bool perf_build_clock = prosper::diagnostics::perf::enabled();
        const auto build_start = timing_enabled || perf_build_clock
            ? RenderClock::now() : RenderClock::time_point{};
        prosper::test::BackendColorTarget backend_target{
            base, seed_rtt0, base != 0 && !defer_readback, pass_format};
        const auto& volume_view = primary_volume_view;
        if (base && volume_view.selected_mip_depth) {
            if (!volume_view.volume_view_consistent() || !volume_view.slice_count) {
                // This draw attempted to replace the retained version. Refusing
                // its view must also revoke the earlier version, including when
                // no backend call is made.
                prosper::test::invalidate_persistent_color_target(base);
                note_volume_producer_denied(base, g_rtt[base], gw, gh,
                                            volume_view.selected_mip_depth,
                                            pass_format);
                std::fprintf(stderr,
                    "[render-volume] invalid slot0 view target=0x%llx\n",
                    static_cast<unsigned long long>(base));
                continue;
            }
            backend_target.volume_depth = volume_view.selected_mip_depth;
            backend_target.volume_first_slice = volume_view.first_slice;
            backend_target.volume_slice_count = volume_view.slice_count;
            backend_target.volume_guest_bytes = producer_volume_physical_bytes;
            backend_target.readback = false;
        }
        backend_target.persistent_id1 = use_color1 ? base1 : 0;
        backend_target.load_existing1 = seed_rtt1;
        backend_target.readback1 = use_color1 && base1 != 0 && !defer_readback1;
        backend_target.format1 = pass_format1;
        // Slots 2..7 retain across render groups on the same terms as slots 0 and 1.
        // A G-buffer built by several groups against one set of allocations otherwise
        // loses every group's work but the last, because slots above 1 were transient
        // images cleared per backend call.
        for (uint32_t slot = 2; slot < mrt_count; ++slot) {
            backend_target.persistent_id_slots[slot] = pass_bases[slot];
            backend_target.load_existing_slots[slot] = seed_target(pass_bases[slot]);
            backend_target.readback_slots[slot] = pass_bases[slot] != 0 && !defer_readback_slots[slot];
        }
        auto backend_draws = build_bds(
            render_pass, batch_backend_submits ? &backend_submission : nullptr);
        const auto build_done = timing_enabled || perf_build_clock
            ? RenderClock::now() : RenderClock::time_point{};
        if (perf_build_clock) {
            prosper::diagnostics::perf::add_cost(
                prosper::diagnostics::perf::Cost::FrontendBuild,
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    build_done - build_start).count()));
            prosper::diagnostics::perf::flush_thread_texture_references();
        }
        prosper::test::BackendMrtOutputs mrt_outputs;
        mrt_outputs.color_count = mrt_count;
        // The colour target is passed whenever ANY slot is bound, not colour-0 alone.
        // The same union the readback flag below already uses, and for the same reason
        // the comment there gives: a pass with colour-0 unbound and a higher slot bound
        // is reachable by construction. On `base != 0` alone such a pass populated
        // persistent_id_slots[2..7] and then handed the backend a nullptr, so those
        // active slots stayed transient and were cleared by the next group -- exactly
        // the defect the persistence contract exists to remove, reintroduced at the one
        // call site that decides whether the contract is used at all.
        const bool any_slot_bound = prosper::frontend::mrt_any_slot_bound(
            pass_bases.data(),
            static_cast<uint32_t>(std::min<size_t>(mrt_count, pass_bases.size())));
        std::vector<uint8_t> gpx = prosper::test::render_draws_rgba(
            backend_draws, gw, gh, seed,
            retained_uniform_clear ? retained_uniform_clear : clear_for(render_pass), true,
            live_gpu_targets && any_slot_bound ? &backend_target : nullptr,
            seed1, retained_uniform_clear1 ? retained_uniform_clear1
                : (use_color1 ? render_pass.front()->ps.clear_color1 : nullptr),
            nullptr,
            batch_backend_submits ? &backend_submission : nullptr,
            pass_i == items.size(), &mrt_outputs,
            // #2283: only ask for colour pixels when something will read them.
            //
            // Keyed on the UNION of every bound slot, not on colour-0 alone. Review
            // caught that: `gpx`'s only consumer is indeed inside an `if (base ...)`
            // pair with no else, but `gpx` is not all the call returns. `mrt_outputs`
            // is filled by the same call and its consumer is keyed per slot on
            // `pass_bases[slot]`, where an EMPTY vector is not a no-op -- it calls
            // `surface.rgba.reset()` and drops that surface's cached pixels.
            //
            // `mrt_count` does not depend on `base`, so a pass with colour-0 unbound
            // and colour-1 bound is reachable by construction -- the same sparse export
            // hole the MRT loop already acknowledges, at slot 0 instead of slot 1. On
            // `base != 0` alone such a pass would silently lose a live colour-1 RTT.
            // Structural rather than observed, which is exactly why the 457/457
            // depth-only measurement could not clear it: that evidence is all slot 0.
            //
            // Depth-only passes have every slot zero, so the win is unchanged.
            // Keyed on the base rather than the draws' colour write masks for the
            // separate reason that 457/457 is evidence about THIS route, and a future
            // title could legally mix a colour-writing draw into a pass that has a base.
            /*want_color_readback=*/any_slot_bound);
        const auto backend_done = timing_enabled
            ? RenderClock::now() : RenderClock::time_point{};
        const prosper::test::BackendColorTargetStats color_target_call =
            prosper::test::backend_color_target_stats();
        const prosper::test::BackendRenderTimingStats backend_call_timing = timing_enabled
            ? prosper::test::backend_render_timing_stats()
            : prosper::test::BackendRenderTimingStats{};
        const prosper::test::BackendTextureUploadStats backend_texture_stats = timing_enabled
            ? prosper::test::backend_texture_upload_stats()
            : prosper::test::BackendTextureUploadStats{};
        const prosper::test::BackendPipelineCacheStats backend_pipeline_stats = timing_enabled
            ? prosper::test::backend_pipeline_cache_stats()
            : prosper::test::BackendPipelineCacheStats{};
        // Captured HERE, beside the pipeline stats, and PASSED IN -- not read inside
        // record_backend_timing. These are thread_local and published by the backend call;
        // reading them from wherever the recorder happens to run returns a different,
        // nearly empty instance. The first draft did exactly that and reported refs=11 for
        // a 2,102-draw submit -- the kind of small plausible number that gets believed.
        const prosper::test::BackendResourceReuseStats backend_reuse_stats = timing_enabled
            ? prosper::test::backend_resource_reuse_stats()
            : prosper::test::BackendResourceReuseStats{};
        if (timing_enabled) {
            pending_timing.build_resources_ms +=
                std::chrono::duration<double, std::milli>(build_done - build_start).count();
            pending_timing.backend_ms +=
                std::chrono::duration<double, std::milli>(backend_done - build_done).count();
            // Everything this group did BEFORE the measured pair -- target grouping,
            // format/extent resolution, resolve handling, seed selection. Recorded here
            // rather than at the loop head so a `continue` between the two points cannot
            // skip it and silently under-report (#2250's lesson, applied one level up).
            pending_timing.pass_pre_ms +=
                std::chrono::duration<double, std::milli>(build_start - group_start).count();
            ++pending_timing.pass_groups;
            record_backend_timing(backend_call_timing, backend_texture_stats,
                                  backend_pipeline_stats, backend_reuse_stats);
            pending_timing.color_target_writes += color_target_call.writes;
            pending_timing.color_target_write_hits += color_target_call.write_hits;
            pending_timing.color_target_sample_hits += color_target_call.sampled_hits;
            pending_timing.color_target_readbacks += color_target_call.readbacks;
            pending_timing.color_target_cached_bytes = color_target_call.cached_bytes;
            pending_timing.color_target_cached_entries = color_target_call.cached_entries;
        }
        const auto post_stats_done = timing_enabled
            ? RenderClock::now() : RenderClock::time_point{};
        static const std::string dump_spec =
            PROSPER_ENV_VALUE("PROSPER_DUMP_PASS")
                ? PROSPER_ENV_VALUE("PROSPER_DUMP_PASS") : "";
        static const int dump_pass_every =
            PROSPER_ENV_VALUE("PROSPER_DUMP_PASS_EVERY")
                ? std::max(1, atoi(getenv("PROSPER_DUMP_PASS_EVERY"))) : 60;
        auto pass_pixels = std::make_shared<const std::vector<uint8_t>>(std::move(gpx));
        if (base && backend_target.volume_depth && !color_target_call.writes) {
            // Backend admission failed before it could record a write (e.g. a device
            // without 2D-on-3D views), so the renderer has no image to serve.
            note_volume_producer_denied(base, g_rtt[base], gw, gh,
                                        backend_target.volume_depth, pass_format);
        }
        const auto* completed_target = base
            ? prosper::test::find_persistent_color_target(
                  base, gw, gh, pass_format, true,
                  backend_target.volume_depth)
            : nullptr;
        const uint64_t pass_source_submit = completed_target
            ? prosper::frontend::completed_source_submit(
                  prosper::test::persistent_color_producer_source(*completed_target))
            : 0;
        if (base && color_target_call.writes) {
            RttSurf& surface = g_rtt[base];
            surface.w = gw;
            surface.h = gh;
            surface.samples = pass.empty()
                ? 1u : 1u << pass.front()->ps.color_targets[0].log2_samples;
            surface.volume_depth = backend_target.volume_depth;
            surface.format = pass_format;
            surface.guest_format = pass.empty()
                ? pass_format
                : static_cast<VkFormat>(
                      prosper::frontend::mrt_raw_format(*pass.front(), 0));
            surface.has_uniform_color = false;
            surface.dcc_metadata_dirty = false;
            surface.gpu_valid = prosper::test::find_persistent_color_target(
                base, gw, gh, pass_format, true,
                backend_target.volume_depth) != nullptr;
            // Claim renderer authority over the volume's guest footprint ONLY when the
            // renderer now holds a valid image to serve it from. The backend marks a
            // retained volume valid only once every slice is proven written, so a
            // volume whose slices are never proven keeps guest memory authoritative, as
            // before #3842; a proven-complete volume keeps its protection while its
            // image survives. GTA V's 32x32x32 colour-grading LUT is the worked case:
            // its only raster pass writes slice 0 of 32, and a compute program writes
            // the LUT back to guest memory. Claiming it made that compute skip and every draw that
            // samples the LUT drop, blacking out the menus and HUD (#3842, #3889).
            settle_volume_guest_footprint(base, surface, surface.gpu_valid,
                                          producer_volume_guard_bytes,
                                          producer_volume_footprint_proven);
            if (!pass_pixels->empty()) surface.rgba = pass_pixels;
            else surface.rgba.reset();
            // GTA V builds its packed-HDR bloom pyramid as separate CB_COLOR targets,
            // then samples them as one mipmapped T#. A target may be hundreds of LRU
            // insertions old by that final draw. Pin every produced level now, while its
            // exact identity is known, and release the small set at the submit boundary.
            pin_renderer_mip_target(base, gw, gh, pass_format, surface.gpu_valid);
            if (defer_readback && is_vo && surface.gpu_valid) {
                const auto already_pinned = std::find_if(
                    pinned_scanouts.begin(), pinned_scanouts.end(),
                    [&](const PinnedScanout& target) {
                        return target.id == base && target.width == gw &&
                               target.height == gh && target.format == pass_format;
                    });
                if (already_pinned == pinned_scanouts.end()) {
                    if (prosper::test::pin_persistent_color_target(
                            base, gw, gh, pass_format)) {
                        pinned_scanouts.push_back({base, gw, gh, pass_format});
                    } else {
                        // Pinning is expected to succeed for the target just written. If
                        // cache state is inconsistent, preserve correctness by immediately
                        // restoring the authoritative CPU fallback.
                        std::vector<uint8_t> materialized;
                        std::string error;
                        if (prosper::test::readback_persistent_color_target(
                                base, gw, gh, pass_format, materialized, error))
                            surface.rgba =
                                std::make_shared<const std::vector<uint8_t>>(
                                    std::move(materialized));
                    }
                }
            }
        } else if (base && !pass_pixels->empty()) {
            RttSurf& surface = g_rtt[base];
            surface.rgba = pass_pixels;
            surface.w = gw;
            surface.h = gh;
            surface.volume_depth = 0;
            surface.format = pass_format;
            surface.guest_format = pass.empty()
                ? pass_format
                : static_cast<VkFormat>(
                      prosper::frontend::mrt_raw_format(*pass.front(), 0));
            surface.gpu_valid = false;
            surface.has_uniform_color = false;
            surface.dcc_metadata_dirty = false;
        }
        const auto post_slot0_done = timing_enabled
            ? RenderClock::now() : RenderClock::time_point{};
        for (uint32_t slot = 1; slot < mrt_count; ++slot) {
            auto& pixels = mrt_outputs.colors[slot];
            if (!pass_bases[slot]) continue;  // transient attachment for a sparse export hole
            if (rtt_log && !pixels.empty()) {
                size_t nz = 0, rgb_nz = 0;
                for (uint8_t byte : pixels) nz += byte != 0;
                if (pass_formats[slot] == VK_FORMAT_R8G8B8A8_UNORM)
                    for (size_t p = 0; p + 3 < pixels.size(); p += 4)
                        rgb_nz += pixels[p] != 0 || pixels[p + 1] != 0 ||
                                  pixels[p + 2] != 0;
                fprintf(stderr,
                        "[rtt] pass c%u=0x%llx extent=%ux%u (%zu draws) "
                        "px_nonzero=%zu rgb_nonblack=%zu\n",
                        slot, (unsigned long long)pass_bases[slot],
                        gw, gh, pass.size(), nz, rgb_nz);
            }
            // PROSPER_DUMP_PASS covers slots 1..7 too. The first version dumped only
            // `base`, so the busiest target in GTA V's frame -- 0x2085de0000, written by
            // 130,290 of 131,072 draws, 96% of them in SLOT 4 -- could not be dumped at
            // all, and its absence read as "that pass does not run".
            if (!dump_spec.empty() && pass_bases[slot]) {
                char needle[24];
                std::snprintf(needle, sizeof needle, "0x%llx",
                              (unsigned long long)pass_bases[slot]);
                if (dump_spec.find(needle) != std::string::npos) {
                    static std::map<uint64_t, size_t> slot_busiest;
                    static std::map<uint64_t, int> slot_counted;
                    const bool slot_peak = pass.size() > slot_busiest[pass_bases[slot]];
                    if (slot_peak) slot_busiest[pass_bases[slot]] = pass.size();
                    if (slot_peak &&
                        slot_counted[pass_bases[slot]]++ % dump_pass_every == 0) {
                        const std::vector<uint8_t> shot = inspection_rgba8(
                            pixels, gw, gh, pass_formats[slot]);
                        const char* dir = getenv("PROSPER_FRAME_DIR");
                        char path[512];
                        std::snprintf(path, sizeof path, "%s/pass_%llx_s%u_%ux%u.bmp",
                                      dir ? dir : ".",
                                      (unsigned long long)pass_bases[slot], slot,
                                      gw, gh);
                        const size_t want = size_t(gw) * gh * 4;
                        const bool ok = shot.size() == want &&
                            prosper::test::dump_bmp(path, shot, gw, gh);
                        fprintf(stderr,
                                "[dump-pass] target=0x%llx slot=%u %ux%u src=%zuB -> %s\n",
                                (unsigned long long)pass_bases[slot], slot, gw, gh,
                                pixels.size(),
                                ok ? path : "NO CPU PIXELS (readback deferred)");
                    }
                }
            }
            RttSurf& surface = g_rtt[pass_bases[slot]];
            surface.w = gw;
            surface.h = gh;
            surface.samples = pass.empty()
                ? 1u : 1u << pass.front()->ps.color_targets[slot].log2_samples;
            surface.volume_depth = 0;
            surface.format = pass_formats[slot];
            surface.guest_format = pass.empty()
                ? pass_formats[slot]
                : static_cast<VkFormat>(
                      prosper::frontend::mrt_raw_format(*pass.front(), slot));
            surface.has_uniform_color = false;
            surface.dcc_metadata_dirty = false;
            // Any slot with a retained target is GPU-valid, not only slot 1. The
            // `slot == 1` clause dated from when slots above 1 had no persistent image
            // to be valid about.
            surface.gpu_valid =
                prosper::test::find_persistent_color_target(
                    pass_bases[slot], gw, gh, pass_formats[slot]) != nullptr;
            pin_renderer_mip_target(pass_bases[slot], gw, gh, pass_formats[slot],
                                    surface.gpu_valid);
            if (!pixels.empty())
                surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                    std::move(pixels));
            else
                surface.rgba.reset();
        }
        // A CPU-only RTT import can come from a producer that never retained an image,
        // an invalidated image, or an evicted one. Trace only named addresses without
        // enabling PROSPER_RTTLOG, which itself turns off the GPU-resident path.
        const auto& residency_trace = rtt_residency_trace_selector();
        if (residency_trace.configured && residency_trace.valid) {
            PassResidencyTraceContext trace_pass_residency_ctx{
                .g_pass_log_submit = g_pass_log_submit,
                .live_gpu_targets = live_gpu_targets,
                .g_rtt = g_rtt,
                .pass_i = pass_i,
                .pass_bases = pass_bases,
                .pass_formats = pass_formats,
                .mrt_count = mrt_count,
                .color_target_call = color_target_call,
                .residency_trace = residency_trace};
            trace_pass_residency(trace_pass_residency_ctx);
        }
        const auto post_mrt_done = timing_enabled
            ? RenderClock::now() : RenderClock::time_point{};
        const std::vector<uint8_t>& rendered_pixels = *pass_pixels;
        // Both dims parse to 0 when PROSPER_RESOURCE_HASH_DIM is UNSET, so an extent
        // comparison alone made every 0x0 pass satisfy the filter and switch the
        // diagnostic on by itself. Require it to have actually been requested.
        if (resource_hash_w && resource_hash_h &&
            native_w == resource_hash_w && native_h == resource_hash_h &&
            !rendered_pixels.empty()) {
            PassResourceHashContext report_pass_resource_hash_ctx{
                .g_this_submit = g_this_submit,
                .base = base,
                .native_w = native_w,
                .native_h = native_h,
                .render_pass = render_pass,
                .seed = seed,
                .rendered_pixels = rendered_pixels};
            report_pass_resource_hash(report_pass_resource_hash_ctx);
        }
        // Same unset-is-0 trap as the resource hash above, and far more expensive here:
        // the loop below re-renders every growing draw prefix, so a 0x0 pass silently
        // turned one pass into O(draws^2) rendering plus a readback and pixel census per
        // step. Require PROSPER_TARGET_STEP_HASH_DIM to have actually been requested.
        if (target_step_w && target_step_h &&
            native_w == target_step_w && native_h == target_step_h &&
            render_pass.size() >= target_step_min_draws) {
            TargetStepHashContext report_target_step_hashes_ctx{
                .g_this_submit = g_this_submit,
                .texstore = texstore,
                .texstore_pinned = texstore_pinned,
                .texstore_used = texstore_used,
                .decoded_textures = decoded_textures,
                .base = base,
                .gw = gw,
                .gh = gh,
                .render_pass = render_pass,
                .seed = seed,
                .backend_draw_ctx = ctx.backend_draw_ctx};
            report_target_step_hashes(report_target_step_hashes_ctx);
        }
        const RttTimingRecord rtt_timing_record{
            g_this_submit, base, gw, gh, pass.size(),
            phase.first_span, phase.final_span, phase.authoritative_readback,
            defer_readback,
            std::chrono::duration<double, std::milli>(
                backend_done - build_done).count(),
            backend_call_timing, color_target_call};
        if (lightweight_rtt_timing) pending_rtt_timing.push_back(rtt_timing_record);
        else if (timing_enabled && rtt_log) print_rtt_timing(rtt_timing_record);
        // PROSPER_DUMP_PASS=0xADDR[,0xADDR…] — write what a pass actually produced, as an
        // image, for the named targets. `rgb_nonblack` alone cannot answer "is the world
        // there": it is computed from an RGBA8 conversion, so an HDR f16 target whose
        // values are all below 1/255 reports EXACTLY ZERO while carrying a complete
        // scene. That is not hypothetical here — GTA V's 0x20431c0000 reports
        // rgb_nonblack=0 while the very next pass extracts 47.5% non-black from it, so
        // reading the metric as "black" is a false negative on the whole lighting stage.
        //
        // Overwrites one file per target so the last write is the latest frame, and
        // dumps every PROSPER_DUMP_PASS_EVERY-th occurrence (default 60) because a 4K
        // BMP is 24 MB and a routed boot renders each of these a few hundred times.
        RenderedPassDumpContext dump_rendered_pass_ctx{
            .base = base,
            .pass = pass,
            .gw = gw,
            .gh = gh,
            .pass_format = pass_format,
            .dump_spec = dump_spec,
            .dump_pass_every = dump_pass_every,
            .rendered_pixels = rendered_pixels};
        dump_rendered_pass(dump_rendered_pass_ctx);
        if (rtt_log) {
            RenderedPassLogContext log_rendered_pass_ctx{
                .g_rtt = g_rtt,
                .front_va = front_va,
                .base = base,
                .pass = pass,
                .native_w = native_w,
                .native_h = native_h,
                .gw = gw,
                .gh = gh,
                .is_vo = is_vo,
                .pass_format = pass_format,
                .rendered_pixels = rendered_pixels};
            log_rendered_pass(log_rendered_pass_ctx);
        }
        // PROSPER_DUMP_DRAWSTEPS: for a pass targeting a SCANOUT buffer, re-render the pass
        // draw-by-draw (prefix 1, prefix 2, ...) and dump each cumulative result — a one-boot
        // bisect for "which draw of the final composite blacks the screen" (#319). Diagnostic.
        PassDrawStepsContext dump_pass_draw_steps_ctx{
            .frame_no = frame_no,
            .base = base,
            .pass = pass,
            .gw = gw,
            .gh = gh,
            .is_vo = is_vo,
            .render_pass = render_pass,
            .seed = seed,
            .backend_draw_ctx = ctx.backend_draw_ctx};
        dump_pass_draw_steps(dump_pass_draw_steps_ctx);
        // Per-target group dumps into PROSPER_FRAME_DIR. Two INDEPENDENT opt-ins (either may
        // be set alone); PROSPER_DUMP_RTGROUPS_ADDR optionally limits a long replay to one
        // target VA. Diagnostic; no default behavior change.
        //  - PROSPER_DUMP_RTGROUPS=<min-nonzero-bytes>: a 24-bit BMP (alpha dropped) of the
        //    format-inspected pixels — to eyeball an intermediate pass (e.g. a UI/banner RT).
        //  - PROSPER_DUMP_RTGROUPS_RGBA: the RAW RGBA8 backend bytes (alpha PRESERVED, no
        //    format inspection) — needed to reason about premultiplied-alpha UI compositing
        //    that samples an RT's alpha as a blend factor. Non-RGBA8 targets are skipped
        //    visibly rather than writing native bytes under a misleading .rgba contract.
        RtGroupDumpContext dump_rt_group_ctx{
            .frame_no = frame_no,
            .base = base,
            .pass_bases = pass_bases,
            .gw = gw,
            .gh = gh,
            .render_pass = render_pass,
            .pass_format = pass_format,
            .rendered_pixels = rendered_pixels};
        dump_rt_group(dump_rt_group_ctx);
        // The POPULATION that the publish zero is a zero OVER, counted BEFORE the
        // emptiness/format gates below -- so it sees every pass whose colour target
        // is disabled, not just those that already qualified as candidates.
        //
        // This placement is the whole point. A same-source positive control proves
        // the DISCRIMINATOR fires; it can never prove the DOMAIN contains the case.
        // Counting the population in the same run is what separates "disabled-target
        // passes exist and none is ever published" from "none exists, so the zero
        // says nothing" -- and those are opposite conclusions about whether #2283's
        // readback skip is safe.
        if (timing_enabled && !pass.empty() && pass.front()->ps.color0_format == 0)
            ++pending_timing.publish_candidate_fmt0;
        if (!explicit_depth_extent && !rendered_pixels.empty() &&
            pass_format == VK_FORMAT_R8G8B8A8_UNORM) {
            // Recorded alongside each candidate rather than re-derived at selection
            // time: by then the pass loop has moved on and `pass` no longer refers to
            // the pass that produced these pixels (#2283).
            const uint32_t candidate_fmt =
                pass.empty() ? kNoPassFormat : pass.front()->ps.color0_format;
            if (base && base == front_va) {                          // the flipped buffer
                px_front = pass_pixels; px_front_w = gw; px_front_h = gh;
                px_front_base = base; px_front_fmt = candidate_fmt;
                px_front_source_submit = pass_source_submit;
            }
            if (is_vo) {                                            // any registered scanout
                px_vo = pass_pixels; px_vo_w = gw; px_vo_h = gh;
                px_vo_base = base; px_vo_fmt = candidate_fmt;
                px_vo_source_submit = pass_source_submit;
            }
            px_last = pass_pixels;                                  // last non-empty (fallback)
            px_last_w = gw; px_last_h = gh; px_last_base = base; px_last_fmt = candidate_fmt;
            px_last_source_submit = pass_source_submit;
        } else if (!explicit_depth_extent && !rendered_pixels.empty() &&
                   prefix_inspect_publish()) {
            // #1330: under gpu_replay's ordered-prefix inspection (--draw/--draw-steps/
            // --through-operation set PROSPER_PREFIX_INSPECT), a prefix ending on a
            // non-RGBA8 pass (an FP16 HDR scene target) must return THAT surface, not
            // whatever stale RGBA8 pass ran earlier. Publish an inspection-converted copy
            // as the weakest fallback: any RGBA8 pass afterwards still overwrites it, and
            // with the env unset (every live/normal replay run) behavior is byte-identical.
            std::vector<uint8_t> converted =
                inspection_rgba8(rendered_pixels, gw, gh, pass_format);
            if (!converted.empty())
                px_last = std::make_shared<const std::vector<uint8_t>>(std::move(converted));
            px_last_source_submit = 0; // inspection conversion is not the native pixels
        }
        // PROSPER_PASS_LOG=<min-submit>|ms:<millis>: per-pass publish provenance for 3
        // submits — which pass produced pixels, its target identity, and the defer
        // decision. The `ms:` form aims the window by wall time (diagnostic_window.hpp).
        // PROSPER_PASS_LOG_NODEFER opens this block too. It was added INSIDE it and
        // armed by its own switch, so setting only PROSPER_PASS_LOG_NODEFER produced
        // ZERO lines -- a silent run indistinguishable from "there are no such passes".
        // That is #2149's class, committed by the author who had been citing #2149 all
        // session: a diagnostic whose producer and printer are armed by different
        // switches reports unarmed state as a confident zero.
        PassPublishLogContext log_pass_publish_ctx{
            .g_pass_log_submit = g_pass_log_submit,
            .g_pass_log_window = g_pass_log_window,
            .items = items,
            .pass_i = pass_i,
            .base = base,
            .pass = pass,
            .gw = gw,
            .gh = gh,
            .is_vo = is_vo,
            .seed_rtt0 = seed_rtt0,
            .pass_format = pass_format,
            .defer_readback = defer_readback,
            .color_target_call = color_target_call,
            .rendered_pixels = rendered_pixels};
        log_pass_publish(log_pass_publish_ctx);
        // Everything this group did AFTER the backend call returned -- RTT store,
        // scanout selection, per-pass diagnostics. `backend_done` is in scope only on
        // the paths that reached the render, which are exactly the paths `pre` counted,
        // so the two spans cover the same population and their sum is comparable to
        // pass_groups.
        if (timing_enabled) {
            const auto post_done = RenderClock::now();
            pending_timing.pass_post_ms += std::chrono::duration<double, std::milli>(
                post_done - backend_done).count();
            pending_timing.post_stats_ms += std::chrono::duration<double, std::milli>(
                post_stats_done - backend_done).count();
            pending_timing.post_slot0_ms += std::chrono::duration<double, std::milli>(
                post_slot0_done - post_stats_done).count();
            pending_timing.post_mrt_ms += std::chrono::duration<double, std::milli>(
                post_mrt_done - post_slot0_done).count();
            pending_timing.post_rest_ms += std::chrono::duration<double, std::milli>(
                post_done - post_mrt_done).count();
        }
    }
    // Everything after the group loop: present selection, scanout resolution, the final
    // render, RTT publication. The tail performs its own build_resources+backend, and
    // pass_control already excludes those -- so subtract what they grew by across this
    // span, or the leaves would exceed their parent and the guard would fire on a
    // correct run.
    if (timing_enabled) {
        pending_timing.pass_loop_ms += std::chrono::duration<double, std::milli>(
            RenderClock::now() - pass_loop_start).count() -
            (pending_timing.build_resources_ms + pending_timing.backend_ms -
             pass_loop_measured_before);
        pass_tail_start = RenderClock::now();
        pass_tail_measured_before =
            pending_timing.build_resources_ms + pending_timing.backend_ms;
    }
    // Present priority: the flipped front buffer > any registered scanout target > the
    // legacy "last group" fallback (unchanged behavior when no group targets a VO buffer).
    // Under the publish extent contract (#1986) a candidate whose byte count disagrees is
    // not a present source, so a correctly sized lower-priority candidate is preferred over
    // an incorrectly sized higher-priority one, and "nothing qualified" yields no source at
    // all rather than a frame the caller must discard. Outside a publish-gate submit
    // (offline replay, render_submit_items tests) this is byte-for-byte the historical
    // identity priority.
    auto candidate_bytes = [](const std::shared_ptr<const std::vector<uint8_t>>& p) {
        return p ? p->size() : size_t{0};
    };
    present_choice = prosper::frontend::select_present_source(
        candidate_bytes(px_front), candidate_bytes(px_vo), candidate_bytes(px_last),
        present_extent_bytes);
    using prosper::frontend::PresentSourceChoice;
    selected_pixels = present_choice == PresentSourceChoice::Front ? px_front
                    : present_choice == PresentSourceChoice::Vo    ? px_vo
                    : present_choice == PresentSourceChoice::Last  ? px_last
                    : nullptr;
    selected_source_submit = prosper::frontend::selected_source_submit(
        selected_pixels, {{{px_front, px_front_source_submit},
                           {px_vo, px_vo_source_submit},
                           {px_last, px_last_source_submit}}});
    // #2283's blocking arm, measured rather than argued: is a pass whose colour target is
    // DISABLED (CB_COLOR0_INFO.FORMAT == 0, CB_COLOR_INVALID) ever chosen as the frame
    // that gets published? The proposed fix stops forcing a readback on such a pass, and
    // it is safe by CONSTRUCTION only if the answer is never. Nothing checked it before,
    // and the guard that looks like it would -- pass_format == R8G8B8A8_UNORM at the
    // candidate site -- does NOT exclude a disabled target, because backend_color_format()
    // FALLS BACK to R8G8B8A8_UNORM for anything it does not recognise, including 0.
    if (timing_enabled) {
        const uint32_t chosen_fmt =
            present_choice == PresentSourceChoice::Front ? px_front_fmt
          : present_choice == PresentSourceChoice::Vo    ? px_vo_fmt
          : present_choice == PresentSourceChoice::Last  ? px_last_fmt
          : kNoPassFormat;
        if (selected_pixels) {
            ++pending_timing.publish_selected;
            if (chosen_fmt == 0) ++pending_timing.publish_selected_fmt0;
            else if (chosen_fmt == kNoPassFormat) ++pending_timing.publish_selected_unknown;
        }
    }
    // The one new semantic path: a non-empty higher-priority candidate was passed over on
    // extent. Unreachable today (a VO/front-buffer pass has its extent pinned to the present
    // extent at :5060-5063 before it renders, so those candidates always fit), which is
    // exactly why it gets a report rather than a comment — this PR's whole thesis is that a
    // silent substitution is what cost 31 consecutive submits, and that applies to its own
    // substitution too. If the pin is ever relaxed, this says so instead of quietly changing
    // which surface reaches the screen.
    if (prosper::frontend::present_source_demoted(
            present_choice, candidate_bytes(px_front), candidate_bytes(px_vo))) {
        static std::atomic<uint64_t> demotions{0};
        const uint64_t ord = demotions.fetch_add(1) + 1;
        if (prosper::diag_should_print(ord))
            fprintf(stderr,
                    "[rtt] PRESENT SOURCE DEMOTED #%llu: chose %s for a requested %ux%u "
                    "(%zu-byte) frame because a higher-priority candidate did not fit — "
                    "px_front=%ux%u(%zu bytes) px_vo=%ux%u(%zu bytes); this should be "
                    "unreachable while VO passes are pinned to the present extent "
                    "(live_renderer.cpp:5060) — check that pin first\n",
                    (unsigned long long)ord,
                    prosper::frontend::present_source_name(present_choice), w, h,
                    present_extent_bytes, px_front_w, px_front_h,
                    candidate_bytes(px_front), px_vo_w, px_vo_h, candidate_bytes(px_vo));
    }
    {
        const uint64_t at = g_pass_log_submit.fetch_add(1);
        if (PROSPER_ENV_ON("PROSPER_PASS_LOG")) {
            // The same window object and the same ordinal the per-pass report used, so
            // the two lines describe one set of callbacks — with one seam: if a `ms:`
            // deadline is crossed between the pass loop and this site, the latching
            // callback carries `selected=` with no per-pass lines beside it.
            if (g_pass_log_window.contains(at, diagnostic_elapsed_ms()))
                fprintf(stderr, "[pass] cb=%llu selected=%s\n", (unsigned long long)at,
                        prosper::frontend::present_source_name(present_choice));
        }
    }
    // PROSPER_DUMP_PERSISTENT=<min-submit>|ms:<millis>: read back persistent color
    // targets after this submit's passes render. The default is the original broad
    // census; PROSPER_DUMP_PERSISTENT_ADDRS limits it to explicitly named addresses.
    // Unlike
    // PROSPER_RTT*/DUMP_*, this flag is NOT in the live_gpu_targets disable list
    // (:1078-1085), so it observes the NORMAL persistent-render path (all other CPU-pixel
    // diagnostics change that path -> #1103). readback_persistent_color_target restores
    // the image layout, so rendering is unaffected. The `ms:` form aims the window by
    // wall time, because the ordinal below is a renderer-internal counter with no
    // published rate and the states worth censusing are named in seconds
    // (diagnostic_window.hpp).
    PersistentTargetDumpContext dump_persistent_targets_ctx{
        .g_persist_window = g_persist_window,
        .g_persist_filter = g_persist_filter,
        .g_persist_extent = g_persist_extent,
        .present_choice = present_choice,
        .g_rtt = g_rtt,
        .vo_n = vo_n,
        .vo_front = vo_front,
        .front_va = front_va};
    dump_persistent_targets(dump_persistent_targets_ctx);
}

} // namespace prosper::frontend::submit_renderer
