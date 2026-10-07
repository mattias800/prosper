// The pass-loop diagnostics -- see pass_diagnostics.hpp. Moved verbatim out of per_target_passes.cpp (#3892).
#include "shared/live/submit_renderer/pass_diagnostics.hpp"
#include "shared/live/submit_renderer/guest_reads.hpp"
#include "shared/live/submit_renderer/mrt_slots.hpp" // color_binding, active_format, active_color (the MRT census)

namespace prosper::frontend::submit_renderer {

void note_resolve_census(ResolveCensusContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& pass = ctx.pass;
    if (!pass.empty() && pass.front()->ps.cb_resolve &&
        PROSPER_ENV_ON("PROSPER_RESOLVE_CENSUS")) {
        static std::mutex mutex;
        static std::map<std::pair<uint64_t, uint64_t>, uint64_t> seen;
        const uint64_t rsrc = pass.front()->color0_base;
        const uint64_t rdst = pass.front()->color1_base;
        std::lock_guard lock(mutex);
        const uint64_t n = ++seen[{rsrc, rdst}];
        if (n <= 2 || (n & (n - 1)) == 0)
            fprintf(stderr,
                    "[resolve-census] src=0x%llx dst(color1)=0x%llx x%llu%s\n",
                    (unsigned long long)rsrc, (unsigned long long)rdst,
                    (unsigned long long)n,
                    rdst ? "" : "  <-- NO EXPRESSIBLE DESTINATION");
    }
}

void note_readback_why(ReadbackWhyContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& live_gpu_targets = ctx.live_gpu_targets;
    auto& phase = ctx.phase;
    auto& vo_n = ctx.vo_n;
    auto& front_va = ctx.front_va;
    auto& base = ctx.base;
    auto& pass_bases = ctx.pass_bases;
    auto& gw = ctx.gw;
    auto& gh = ctx.gh;
    auto& is_vo = ctx.is_vo;
    auto& pass_format = ctx.pass_format;
    auto& mrt_count = ctx.mrt_count;
    auto& consumers0 = ctx.consumers0;
    auto& sampled_exact_later = ctx.sampled_exact_later;
    auto& feedback_later = ctx.feedback_later;
    auto& cpu_needed_same_batch = ctx.cpu_needed_same_batch;
    static const bool why = getenv("PROSPER_READBACK_WHY") != nullptr;
    if (why) {
        // Counts identify the failing defer condition; bytes weight each bucket
        // by the actual copy size, since readback time scales with bytes.
        enum { kNoLive, kNoVo, kNoBase, kAuth, kVoFinal, kFront,
               kSameBatchCpu, kNotSampledExact, kFeedback, kBuckets };
        static std::atomic<uint64_t> counts[kBuckets], bytes[kBuckets];
        static std::atomic<uint64_t> same_batch_storage{0};
        static std::atomic<uint64_t> same_batch_dimension{0};
        static std::atomic<uint64_t> same_batch_extent{0};
        static std::atomic<uint64_t> same_batch_feedback{0};
        static std::atomic<uint64_t> c_total{0};
        int bucket = kNotSampledExact;
        if (!live_gpu_targets) bucket = kNoLive;
        else if (vo_n <= 0) bucket = kNoVo;
        else if (!base) bucket = kNoBase;
        else if (phase.authoritative_readback) bucket = kAuth;
        else if (is_vo) bucket = kVoFinal;
        else if (base == front_va) bucket = kFront;
        else if (cpu_needed_same_batch) bucket = kSameBatchCpu;
        else if (sampled_exact_later && feedback_later) bucket = kFeedback;
        ++counts[bucket];
        if (bucket == kSameBatchCpu) {
            same_batch_storage += consumers0.storage_references;
            same_batch_dimension += consumers0.dimension_mismatches;
            same_batch_extent += consumers0.extent_mismatches;
            same_batch_feedback += consumers0.feedback_references;
        }
        // #2398: bill bytes only when colour pixels are ACTUALLY requested.
        // #2283 narrowed the readback to `want_color_readback` (the same
        // any_of over pass_bases used at the call below) and this accounting
        // never followed it -- so a depth-only pass, whose slot bases are all
        // zero and which therefore lands in the `no_base` bucket BY
        // CONSTRUCTION, was billed a full frame of colour it never copied.
        //
        // That is not a rounding error: on Blue Prince it reported 15.6 GB
        // against `no_base` and made depth passes look like the frame-rate
        // wall. The count was right; the column that made it actionable was
        // measuring a world from before #2283. An instrument that drifts from
        // the code it measures reads as evidence, which is worse than silence.
        const bool billed_color = std::any_of(
            pass_bases.begin(),
            pass_bases.begin() + std::min<size_t>(mrt_count, pass_bases.size()),
            [](uint64_t slot_base) { return slot_base != 0; });
        if (billed_color)
            bytes[bucket] += static_cast<uint64_t>(gw) * gh *
                prosper::test::backend_color_bytes_per_pixel(pass_format);
        static std::atomic<uint64_t> last_report{0};
        const uint64_t t = ++c_total;
        if (t >= last_report.load(std::memory_order_relaxed) + 200) {
            last_report.store(t, std::memory_order_relaxed);
            static const char* names[kBuckets] = {
                "no_live", "no_vo", "no_base", "auth", "vo_final",
                "front", "same_batch_cpu", "not_sampled_exact", "feedback"};
            fprintf(stderr, "[readback-why] total=%llu",
                    (unsigned long long)t);
            for (int i = 0; i < kBuckets; ++i)
                fprintf(stderr, " %s=%llu/%lluMB", names[i],
                        (unsigned long long)counts[i].load(),
                        (unsigned long long)(bytes[i].load() >> 20));
            fprintf(stderr,
                    " same_batch_reasons=storage:%llu,dimension:%llu,"
                    "extent:%llu,feedback:%llu\n",
                    (unsigned long long)same_batch_storage.load(),
                    (unsigned long long)same_batch_dimension.load(),
                    (unsigned long long)same_batch_extent.load(),
                    (unsigned long long)same_batch_feedback.load());
        }
    }
}

void trace_pass_residency(PassResidencyTraceContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& g_pass_log_submit = ctx.g_pass_log_submit;
    auto& live_gpu_targets = ctx.live_gpu_targets;
    auto& g_rtt = ctx.g_rtt;
    auto& pass_i = ctx.pass_i;
    auto& pass_bases = ctx.pass_bases;
    auto& pass_formats = ctx.pass_formats;
    auto& mrt_count = ctx.mrt_count;
    auto& color_target_call = ctx.color_target_call;
    auto& residency_trace = ctx.residency_trace;
    for (uint32_t slot = 0; slot < mrt_count; ++slot) {
        const uint64_t address = pass_bases[slot];
        if (!address || !residency_trace.includes(address)) continue;
        const auto found = g_rtt.find(address);
        if (found == g_rtt.end()) continue;
        const RttSurf& surface = found->second;
        const auto* retained = prosper::test::find_persistent_color_target(
            address, surface.w, surface.h, pass_formats[slot], false,
            surface.volume_depth);   // every slot may be a volume (#4643)
        std::fprintf(stderr,
            "[rtt-residency] submit=%llu pass=%zu slot=%u addr=0x%llx "
            "extent=%ux%u format=%u live=%u writes=%llu readbacks=%llu "
            "gpu-valid=%u cpu-bytes=%zu cached=%u cache-valid=%u "
            "cache-image=%u\n",
            (unsigned long long)g_pass_log_submit.load(
                std::memory_order_relaxed), pass_i, slot,
            (unsigned long long)address, surface.w, surface.h,
            static_cast<unsigned>(pass_formats[slot]),
            live_gpu_targets ? 1u : 0u,
            (unsigned long long)color_target_call.writes,
            (unsigned long long)color_target_call.readbacks,
            surface.gpu_valid ? 1u : 0u,
            surface.rgba ? surface.rgba->size() : 0u,
            retained ? 1u : 0u,
            retained && retained->valid ? 1u : 0u,
            retained && retained->image ? 1u : 0u);
    }
}

void report_pass_resource_hash(PassResourceHashContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& g_this_submit = ctx.g_this_submit;
    auto& base = ctx.base;
    auto& native_w = ctx.native_w;
    auto& native_h = ctx.native_h;
    auto& render_pass = ctx.render_pass;
    auto& seed = ctx.seed;
    auto& rendered_pixels = ctx.rendered_pixels;
    uint64_t hash = 1469598103934665603ull;
    for (uint8_t byte : rendered_pixels) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    const auto* first = render_pass.empty() ? nullptr : render_pass.front();
    const auto* last = render_pass.empty() ? nullptr : render_pass.back();
    fprintf(stderr,
            "[target-version] render-submit=%llu target=0x%llx dims=%ux%u "
            "draws=%llu-%llu orders=%llu-%llu seed=%d clear=%d "
            "rgba=%.3f,%.3f,%.3f,%.3f hash=%016llx\n",
            (unsigned long long)g_this_submit,
            (unsigned long long)base, native_w, native_h,
            (unsigned long long)(first ? first->draw_index : 0),
            (unsigned long long)(last ? last->draw_index : 0),
            (unsigned long long)(first ? first->command_order : 0),
            (unsigned long long)(last ? last->command_order : 0),
            seed != nullptr, first ? (int)first->ps.has_clear_color : 0,
            first ? first->ps.clear_color[0] : 0.0f,
            first ? first->ps.clear_color[1] : 0.0f,
            first ? first->ps.clear_color[2] : 0.0f,
            first ? first->ps.clear_color[3] : 0.0f,
            (unsigned long long)hash);
}

void report_target_step_hashes(TargetStepHashContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& g_this_submit = ctx.g_this_submit;
    auto& texstore = ctx.texstore;
    auto& texstore_pinned = ctx.texstore_pinned;
    auto& texstore_used = ctx.texstore_used;
    auto& decoded_textures = ctx.decoded_textures;
    auto& base = ctx.base;
    auto& gw = ctx.gw;
    auto& gh = ctx.gh;
    auto& render_pass = ctx.render_pass;
    auto& seed = ctx.seed;
    auto& backend_draw_ctx = ctx.backend_draw_ctx;
    // build_bds over the same BackendDrawContext render_per_target_passes forwards to.
    auto build_bds = [&](const std::vector<const prosper::gpu::DrawItem*>& group,
                         prosper::test::BackendSubmissionBatch* producer_batch = nullptr) {
        return build_backend_draws(backend_draw_ctx, group, producer_batch);
    };
    for (size_t k = 1; k <= render_pass.size(); ++k) {
        std::vector<const prosper::gpu::DrawItem*> prefix(
            render_pass.begin(), render_pass.begin() + k);
        // PROSPER_TARGET_STEP_HASH_DIM re-renders growing draw prefixes through
        // build_bds(), so every prefix must resolve its own resources from
        // scratch. Release the scratch pins with the map they belonged to,
        // otherwise each prefix would strand another set of slots.
        texstore_used = 0;
        texstore_pinned.assign(texstore.size(), false);
        decoded_textures.clear();
        std::vector<uint8_t> step = prosper::test::render_draws_rgba(
            build_bds(prefix), gw, gh, seed, clear_for(prefix), true);
        uint64_t hash = 1469598103934665603ull;
        for (uint8_t byte : step) {
            hash ^= byte;
            hash *= 1099511628211ull;
        }
        size_t dark = 0, near_white = 0;
        uint64_t rgb_sum = 0;
        const VkFormat step_format = prosper::test::backend_color_format(
            static_cast<VkFormat>(prefix.front()->ps.color0_format));
        const std::vector<uint8_t> step_rgba = inspection_rgba8(
            step, gw, gh, step_format);
        for (size_t p = 0; p + 3 < step_rgba.size(); p += 4) {
            const uint8_t r = step_rgba[p], g = step_rgba[p + 1], b = step_rgba[p + 2];
            dark += std::max({r, g, b}) < 64;
            near_white += std::min({r, g, b}) > 240;
            rgb_sum += r + g + b;
        }
        const double mean_rgb = step_rgba.empty() ? 0.0 :
            (double)rgb_sum / ((step_rgba.size() / 4) * 3);
        const auto* draw = prefix.back();
        auto spirv_hash = [](const std::vector<uint32_t>& words) {
            uint64_t hash = 1469598103934665603ull;
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(words.data());
            for (size_t i = 0; i < words.size() * sizeof(uint32_t); ++i) {
                hash ^= bytes[i];
                hash *= 1099511628211ull;
            }
            return hash;
        };
        fprintf(stderr,
                "[target-step] render-submit=%llu target=0x%llx step=%zu/%zu "
                "draw=%llu order=%llu vs=%016llx ps=%016llx mask=0x%x "
                "blend=%d/%u/%u hash=%016llx dark=%zu white=%zu mean=%.2f\n",
                (unsigned long long)g_this_submit,
                (unsigned long long)base, k, render_pass.size(),
                (unsigned long long)draw->draw_index,
                (unsigned long long)draw->command_order,
                (unsigned long long)spirv_hash(draw->vs_words()),
                (unsigned long long)spirv_hash(draw->fs_words()),
                draw->ps.color_write_mask, (int)draw->ps.blend_enable,
                draw->ps.src_color_blend_factor,
                draw->ps.dst_color_blend_factor,
                (unsigned long long)hash, dark, near_white, mean_rgb);
    }
}

void log_rendered_pass(RenderedPassLogContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& g_rtt = ctx.g_rtt;
    auto& front_va = ctx.front_va;
    auto& base = ctx.base;
    auto& pass = ctx.pass;
    auto& native_w = ctx.native_w;
    auto& native_h = ctx.native_h;
    auto& gw = ctx.gw;
    auto& gh = ctx.gh;
    auto& is_vo = ctx.is_vo;
    auto& pass_format = ctx.pass_format;
    auto& rendered_pixels = ctx.rendered_pixels;
    const std::vector<uint8_t> inspected = inspection_rgba8(
        rendered_pixels, gw, gh, pass_format);
    size_t nz = 0, rgb_nz = 0;
    for (uint8_t b : rendered_pixels) nz += (b != 0);
    for (size_t p = 0; p + 3 < inspected.size(); p += 4)
        rgb_nz += (inspected[p] != 0 || inspected[p + 1] != 0 ||
                   inspected[p + 2] != 0);
    // `src=` is load-bearing, not decoration. Both counters are computed from
    // `rendered_pixels`, so a pass whose readback was deferred reports
    // `px_nonzero=0 rgb_nonblack=0` -- identical to a genuinely black target, and
    // indistinguishable from it without this field. The `[dump-pass]` line four
    // lines up already guards exactly this ("as 'the target is black' rather than
    // 'we never looked'"); this line did not, and readbacks are deferred routinely
    // -- `[readback-why]` buckets nine distinct reasons.
    //
    // It matters most for the conclusion it silently supports. Every "this pass
    // reads populated inputs and emits nothing" reading on GTA V rests on these
    // two counters, and "we never looked" produces that reading for free.
    fprintf(stderr, "[rtt] pass target=0x%llx extent=%ux%u native=%ux%u (%zu draws) "
            "src=%zuB px_nonzero=%zu rgb_nonblack=%zu cache_size=%zu%s%s\n",
            (unsigned long long)base, gw, gh, native_w, native_h, pass.size(),
            rendered_pixels.size(), nz, rgb_nz, g_rtt.size(),
            is_vo ? " SCANOUT" : "", base && base == front_va ? " FRONT" : "");
}

void dump_rendered_pass(RenderedPassDumpContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& base = ctx.base;
    auto& pass = ctx.pass;
    auto& gw = ctx.gw;
    auto& gh = ctx.gh;
    auto& pass_format = ctx.pass_format;
    auto& dump_spec = ctx.dump_spec;
    auto& dump_pass_every = ctx.dump_pass_every;
    auto& rendered_pixels = ctx.rendered_pixels;
    {
        if (!dump_spec.empty() && base) {
            char needle[24];
            std::snprintf(needle, sizeof needle, "0x%llx",
                          (unsigned long long)base);
            if (dump_spec.find(needle) != std::string::npos) {
                // Keep the occurrence with the MOST DRAWS, not the last. A
                // G-buffer target is written by one heavy pass and then touched by
                // several small ones, so "last write wins" reliably captures a
                // near-empty tail and reads as "this channel is empty" -- which it
                // did, for three channels that a draw census showed at 54%.
                static std::map<uint64_t, size_t> busiest;
                static std::map<uint64_t, int> counted;
                const bool new_peak = pass.size() > busiest[base];
                if (new_peak) busiest[base] = pass.size();
                if (new_peak && counted[base]++ % dump_pass_every == 0) {
                    const std::vector<uint8_t> shot = inspection_rgba8(
                        rendered_pixels, gw, gh, pass_format);
                    const char* dir = getenv("PROSPER_FRAME_DIR");
                    char path[512];
                    std::snprintf(path, sizeof path, "%s/pass_%llx_%ux%u.bmp",
                                  dir ? dir : ".", (unsigned long long)base,
                                  gw, gh);
                    // Report the source size, not just the intent. A pass whose
                    // readback was deferred has EMPTY cpu pixels, and both this dump
                    // and `rgb_nonblack` are then reporting on nothing — which reads
                    // as "the target is black" rather than "we never looked".
                    const size_t want = size_t(gw) * gh * 4;
                    const bool ok = shot.size() == want &&
                        prosper::test::dump_bmp(path, shot, gw, gh);
                    fprintf(stderr,
                            "[dump-pass] target=0x%llx %ux%u src=%zuB rgba=%zuB/%zuB"
                            " -> %s\n",
                            (unsigned long long)base, gw, gh,
                            rendered_pixels.size(), shot.size(), want,
                            ok ? path : "NO CPU PIXELS (readback deferred)");
                }
            }
        }
    }
}

void dump_pass_draw_steps(PassDrawStepsContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& frame_no = ctx.frame_no;
    auto& base = ctx.base;
    auto& pass = ctx.pass;
    auto& gw = ctx.gw;
    auto& gh = ctx.gh;
    auto& is_vo = ctx.is_vo;
    auto& render_pass = ctx.render_pass;
    auto& seed = ctx.seed;
    auto& backend_draw_ctx = ctx.backend_draw_ctx;
    // build_bds over the same BackendDrawContext render_per_target_passes forwards to.
    auto build_bds = [&](const std::vector<const prosper::gpu::DrawItem*>& group,
                         prosper::test::BackendSubmissionBatch* producer_batch = nullptr) {
        return build_backend_draws(backend_draw_ctx, group, producer_batch);
    };
    if (PROSPER_ENV_ON("PROSPER_DUMP_DRAWSTEPS") && is_vo && pass.size() > 1) {
        for (size_t k = 1; k <= pass.size(); k++) {
            std::vector<const prosper::gpu::DrawItem*> prefix(render_pass.begin(), render_pass.begin() + k);
            std::vector<uint8_t> spx = prosper::test::render_draws_rgba(
                build_bds(prefix), gw, gh, seed, clear_for(prefix));
            if (spx.empty()) continue;
            size_t snz = 0; for (uint8_t b : spx) snz += (b != 0);
            fprintf(stderr, "[rtt] drawstep %zu/%zu tgt=0x%llx px_nonzero=%zu\n",
                    k, pass.size(), (unsigned long long)base, snz);
            const char* dd = getenv("PROSPER_FRAME_DIR");
            char fn[512]; snprintf(fn, sizeof fn, "%s/drawstep_%04d_%zu.bmp",
                                   dd ? dd : ".", frame_no.load(), k);
            prosper::test::dump_bmp(fn, spx, gw, gh);
        }
    }
}

void dump_rt_group(RtGroupDumpContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& frame_no = ctx.frame_no;
    auto& base = ctx.base;
    auto& pass_bases = ctx.pass_bases;
    auto& gw = ctx.gw;
    auto& gh = ctx.gh;
    auto& render_pass = ctx.render_pass;
    auto& pass_format = ctx.pass_format;
    auto& rendered_pixels = ctx.rendered_pixels;
    if ((PROSPER_ENV_ON("PROSPER_DUMP_RTGROUPS") || (getenv("PROSPER_DUMP_RTGROUPS_RGBA") != nullptr)) &&
        !rendered_pixels.empty()) {
        size_t nz = 0; for (uint8_t b : rendered_pixels) nz += (b != 0);
        const char* address_filter = PROSPER_ENV_VALUE("PROSPER_DUMP_RTGROUPS_ADDR");
        const uint64_t wanted_base = address_filter && *address_filter
            ? strtoull(address_filter, nullptr, 0) : 0;
        // Match the address against EVERY MRT slot, not just slot 0.
        //
        // `base` is the pass's slot-0 attachment. A G-buffer names different
        // surfaces per slot, so filtering by a slot-2 address matched ONE pass in a
        // full routed run for a target the draw census credits with 6,672 draws --
        // and a near-empty result reads as "that target is empty" rather than as
        // "the filter asked the wrong question". The dumped pixels are still slot
        // 0's; what this fixes is WHICH PASSES are selected, so naming any slot of a
        // G-buffer selects that G-buffer's passes.
        bool slot_match = !wanted_base || base == wanted_base;
        uint32_t matched_slot = 0;
        if (wanted_base && !slot_match)
            for (uint32_t slot = 1; slot < pass_bases.size(); ++slot)
                if (pass_bases[slot] == wanted_base) {
                    slot_match = true; matched_slot = slot; break;
                }
        const char* dd = getenv("PROSPER_FRAME_DIR");
        // Identify the pass by its first..last draw index so multiple passes to the same
        // target VA in one frame do not silently overwrite each other.
        const uint64_t pass_d0 = render_pass.empty() ? 0u : render_pass.front()->draw_index;
        const uint64_t pass_d1 = render_pass.empty() ? 0u : render_pass.back()->draw_index;
        if (slot_match) {
            if (wanted_base && matched_slot) {
                static std::set<std::pair<uint64_t, uint32_t>> slot_seen;
                if (slot_seen.emplace(wanted_base, matched_slot).second)
                    fprintf(stderr,
                            "[rtt] rtgroup filter 0x%llx matched slot %u of a pass "
                            "whose slot-0 base is 0x%llx; dumped pixels are SLOT 0\n",
                            (unsigned long long)wanted_base, matched_slot,
                            (unsigned long long)base);
            }
            if (const char* rg = PROSPER_ENV_VALUE("PROSPER_DUMP_RTGROUPS"); rg && nz >= (size_t)atol(rg)) {
                const std::vector<uint8_t> inspected = inspection_rgba8(
                    rendered_pixels, gw, gh, pass_format);
                char fn[512]; snprintf(fn, sizeof fn, "%s/rtgrp_%llx_%04d.bmp",
                                       dd ? dd : ".", (unsigned long long)base, frame_no.load());
                prosper::test::dump_bmp(fn, inspected, gw, gh);
            }
            // Independent of the BMP variable AND its nonzero threshold, so a fully
            // transparent (all-zero) group is captured too. Self-describing filename
            // (extent + draw range) and failure-visible: a missing file must not be
            // mistaken for a transparent/empty result.
            if ((getenv("PROSPER_DUMP_RTGROUPS_RGBA") != nullptr)) {
                const uint64_t expected_bytes_u64 = static_cast<uint64_t>(gw) * gh * 4u;
                const bool rgba8_format = pass_format == VK_FORMAT_R8G8B8A8_UNORM;
                const bool size_valid = expected_bytes_u64 <= SIZE_MAX &&
                    rendered_pixels.size() == static_cast<size_t>(expected_bytes_u64);
                if (!rgba8_format || !size_valid) {
                    fprintf(stderr,
                            "[rtt] rgba-dump skipped target=0x%llx %ux%u draws=%llu..%llu "
                            "reason=%s format=%d expected=%llu actual=%zu\n",
                            (unsigned long long)base, gw, gh,
                            (unsigned long long)pass_d0, (unsigned long long)pass_d1,
                            rgba8_format ? "size-mismatch" : "unsupported-format",
                            static_cast<int>(pass_format),
                            (unsigned long long)expected_bytes_u64,
                            rendered_pixels.size());
                } else {
                    char rn[600]; snprintf(rn, sizeof rn, "%s/rtgrp_%llx_%ux%u_f%04d_d%04llu-%04llu.rgba",
                                           dd ? dd : ".", (unsigned long long)base, gw, gh,
                                           frame_no.load(), (unsigned long long)pass_d0,
                                           (unsigned long long)pass_d1);
                    FILE* rf = fopen(rn, "wb");
                    bool ok = rf != nullptr; size_t wrote = 0;
                    if (rf) {
                        wrote = fwrite(rendered_pixels.data(), 1, rendered_pixels.size(), rf);
                        ok = (wrote == rendered_pixels.size());
                        if (fclose(rf) != 0) ok = false;
                    }
                    fprintf(stderr, "[rtt] rgba-dump %s target=0x%llx %ux%u draws=%llu..%llu "
                            "bytes=%zu path=%s\n", ok ? "ok" : "FAILED",
                            (unsigned long long)base, gw, gh,
                            (unsigned long long)pass_d0, (unsigned long long)pass_d1,
                            wrote, rn);
                }
            }
        }
    }
}

void log_pass_publish(PassPublishLogContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& g_pass_log_submit = ctx.g_pass_log_submit;
    auto& g_pass_log_window = ctx.g_pass_log_window;
    auto& items = ctx.items;
    auto& pass_i = ctx.pass_i;
    auto& base = ctx.base;
    auto& pass = ctx.pass;
    auto& gw = ctx.gw;
    auto& gh = ctx.gh;
    auto& is_vo = ctx.is_vo;
    auto& seed_rtt0 = ctx.seed_rtt0;
    auto& pass_format = ctx.pass_format;
    auto& defer_readback = ctx.defer_readback;
    auto& color_target_call = ctx.color_target_call;
    auto& rendered_pixels = ctx.rendered_pixels;
    if (PROSPER_ENV_ON("PROSPER_PASS_LOG") ||
        PROSPER_ENV_ON("PROSPER_PASS_LOG_NODEFER")) {
        const uint64_t at = g_pass_log_submit.load(std::memory_order_relaxed);
        const bool in_window =
            g_pass_log_window.contains(at, diagnostic_elapsed_ms());
        // px_nonblack must be counted in the PASS's OWN format. The original loop
        // stepped 4 bytes and tested bytes `p`, `p+1`, `p+2` — never `p+3` — as if
        // every target were 8-bit RGBA. Over an 8-byte FP16 texel that reads
        // {R_lo, R_hi, G_lo} and then {B_lo, B_hi, A_lo}, so whether a texel counts
        // depends only on where a half-float happens to put its non-zero bytes.
        // Measured on Sonic Origins' 3840x2160 R16G16B16A16_SFLOAT scene target, whose
        // texels are RGB bit-zero with alpha 0x3c05 (1.00488): byte 6 is that alpha's
        // low mantissa byte 0x05, which lands on the second group's `p+2`, so exactly
        // one of the two groups per texel counted and the line reported 8,294,400 —
        // precisely w*h, reading as "every pixel has content" for a frame with no
        // colour in it at all. An alpha of exactly 1.0 would have counted ZERO:
        // 0x3c00's only non-zero byte is the `p+3` this loop skipped. The number was
        // never a property of the image, only of its bit layout — and it survived long
        // enough to become a hypothesis (#1905). Count through the same inspection
        // conversion the persistent dump uses; -1 says the format has no conversion,
        // which is honest where a wrong number is not.
        long long nz = 0;
        const bool direct_rgba8 = pass_format == VK_FORMAT_R8G8B8A8_UNORM &&
            rendered_pixels.size() == static_cast<size_t>(gw) * gh * 4u;
        const std::vector<uint8_t> inspected = direct_rgba8
            ? std::vector<uint8_t>{}
            : inspection_rgba8(rendered_pixels, gw, gh, pass_format);
        const std::vector<uint8_t>& counted =
            direct_rgba8 ? rendered_pixels : inspected;
        if (!rendered_pixels.empty() && counted.empty()) nz = -1;
        else
            for (size_t p = 0; p + 3 < counted.size(); p += 4)
                if (counted[p] || counted[p + 1] || counted[p + 2]) nz++;
        // In-window: every pass. Out of window: only content-bearing (or deferred)
        // passes, so the publish source can be found without knowing the callback —
        // plus `nz < 0`, a pass whose format the inspection path cannot convert. That
        // last case has to stay visible out of window: it is the one where this line
        // cannot say whether the pass carries content, which is exactly when the
        // reader needs to know the pass existed.
        // PROSPER_PASS_LOG_NODEFER: print the NON-DEFERRED passes specifically.
        // They are the population #2276 needs and the only one this line cannot
        // currently reach: a deferred pass always prints, a content-bearing pass
        // prints on `nz > 100`, and the window covers three callbacks -- but a
        // non-deferred pass with little content satisfies none of those, so 2,999
        // of them a route were invisible while a classifier counted 21 GB of
        // readback against them.
        //
        // Opt-in and separate from PROSPER_PASS_LOG rather than widening the window,
        // because the window's three-callback span is deliberate (its comment says
        // why) and this needs the whole run, not a wider sample. ~9.7 lines a submit.
        static const bool pass_log_nodefer =
            PROSPER_ENV_ON("PROSPER_PASS_LOG_NODEFER");
        if (in_window || nz > 100 || nz < 0 || defer_readback ||
            (pass_log_nodefer && !defer_readback))
            fprintf(stderr,
                    // `bytes` disambiguates px_nonblack=0, which today means EITHER
                    // "the readback returned a near-black surface" OR "the readback
                    // returned nothing and the counting loop never ran". Those are
                    // opposite facts wearing the same number -- #2255's absent-vs-zero
                    // collapse, and instrument trap 116's -- and on #2276 it is
                    // precisely the ambiguity blocking the diagnosis: 2,999 passes a
                    // route take a ~7.1 MB readback and report px_nonblack under 100,
                    // and nobody can say from this line whether those surfaces are
                    // black or absent.
                    "[pass] cb=%llu pass=%zu/%zu base=0x%llx %ux%u fmt=%d vo=%d "
                    // raw_fmt is CB_COLOR0_INFO.FORMAT BEFORE backend_color_format
                    // touches it, and it is the field that decides #2283. That
                    // converter falls back to R8G8B8A8_UNORM for anything it does
                    // not recognise -- INCLUDING 0 -- and R8G8B8A8_UNORM is itself
                    // enumerator 37, so the most common real format and "unknown"
                    // print identically. `fmt` above therefore cannot distinguish a
                    // live colour target from a disabled one, and I nearly published
                    // a correctness claim on it.
                    //
                    // raw_fmt=0 is CB_COLOR_INVALID: the target is DISABLED, the
                    // extent in CB_COLOR0_ATTRIB2 is stale from an earlier pass
                    // (context registers are sticky), and a base of 0 is correct --
                    // the readback is then simply waste. raw_fmt non-zero with
                    // base=0 is the opposite: the guest declared a live colour
                    // target and prosper does not have its address.
                    "seed=%d defer=%d writes=%llu px_nonblack=%lld bytes=%zu "
                    "raw_fmt=%u\n",
                    (unsigned long long)at, pass_i, items.size(),
                    (unsigned long long)base, gw, gh, (int)pass_format,
                    (int)is_vo, (int)seed_rtt0, (int)defer_readback,
                    (unsigned long long)color_target_call.writes, nz,
                    rendered_pixels.size(),
                    pass.empty() ? 0u : pass.front()->ps.color0_format);
    }
}

void dump_persistent_targets(PersistentTargetDumpContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& g_persist_window = ctx.g_persist_window;
    auto& g_persist_filter = ctx.g_persist_filter;
    auto& g_persist_extent = ctx.g_persist_extent;
    auto& present_choice = ctx.present_choice;
    auto& g_rtt = ctx.g_rtt;
    auto& vo_n = ctx.vo_n;
    auto& vo_front = ctx.vo_front;
    auto& front_va = ctx.front_va;
    if (PROSPER_ENV_ON("PROSPER_DUMP_PERSISTENT")) {
        static std::atomic<uint64_t> dp_submit{0};
        const uint64_t sub = dp_submit.fetch_add(1);
        if (g_persist_window.contains(sub, diagnostic_elapsed_ms())) {
            const char* dd = getenv("PROSPER_FRAME_DIR");
            prosper::frontend::PersistentReadbackBudget selected_readback_budget;
            fprintf(stderr, "[persist] submit=%llu present: front=%d/%d front_va=0x%llx "
                    "selected=%s vo:", (unsigned long long)sub, vo_front, vo_n,
                    (unsigned long long)front_va,
                    prosper::frontend::present_source_name(present_choice));
            for (int i = 0; i < vo_n && i < 8; i++)
                fprintf(stderr, " [%d]=0x%llx", i, (unsigned long long)prosper_vo_buffer_addr(i));
            fprintf(stderr, "\n");
            for (auto& kv : g_rtt) {
                if (!g_persist_filter.allows(kv.first)) continue;
                RttSurf& s = kv.second;
                if (g_persist_extent.first &&
                    (s.w != g_persist_extent.first ||
                     s.h != g_persist_extent.second)) continue;
                if (!s.gpu_valid || !s.w || !s.h ||
                    static_cast<uint64_t>(s.w) * s.h * std::max(1u, s.volume_depth) <
                        uint64_t{64u} * 64u) {
                    if (g_persist_filter.state ==
                            prosper::frontend::PersistentReadbackFilterState::Selected)
                        fprintf(stderr, "[persist] submit=%llu addr=0x%llx "
                                        "not GPU-valid or below 64x64\n",
                                (unsigned long long)sub,
                                (unsigned long long)kv.first);
                    continue;
                }
                const VkFormat fmt = prosper::test::backend_color_format(s.format);
                const uint32_t bpp = prosper::test::backend_color_bytes_per_pixel(fmt);
                uint64_t expected = 0;
                if (g_persist_extent.first || g_persist_filter.state ==
                    prosper::frontend::PersistentReadbackFilterState::Selected) {
                    const auto charge = selected_readback_budget.admit(s.w, s.h, bpp,
                                                                       expected);
                    if (charge != prosper::frontend::PersistentReadbackCharge::Admitted) {
                        const char* reason = "budget 256 MiB";
                        if (charge == prosper::frontend::PersistentReadbackCharge::InvalidSize)
                            reason = "invalid size";
                        else if (charge == prosper::frontend::PersistentReadbackCharge::SizeOverflow)
                            reason = "size overflow";
                        fprintf(stderr, "[persist] submit=%llu addr=0x%llx "
                                        "skipped: selected readback %s\n",
                                (unsigned long long)sub, (unsigned long long)kv.first,
                                reason);
                        continue;
                    }
                } else
                    expected = static_cast<uint64_t>(s.w) * s.h * bpp;
                expected *= std::max(1u, s.volume_depth);   // a volume reads back whole
                std::vector<uint8_t> px; std::string err;
                if (!prosper::test::readback_persistent_color_target(kv.first, s.w, s.h, fmt, px,
                                                                     err, s.volume_depth) ||
                    px.size() != expected) {
                    if (g_persist_filter.state ==
                            prosper::frontend::PersistentReadbackFilterState::Selected)
                        fprintf(stderr, "[persist] submit=%llu addr=0x%llx "
                                        "readback failed: %s (got=%zu expected=%llu)\n",
                                (unsigned long long)sub,
                                (unsigned long long)kv.first, err.c_str(), px.size(),
                                (unsigned long long)expected);
                    continue;
                }
                if (s.volume_depth) {
                    // A volume writes no BMP: its raw bytes and their non-zero count answer "did
                    // anything render into it" (#3135 P5: Kena's merged-NGG LUTs).
                    size_t nz = 0;
                    for (uint8_t byte : px) nz += byte != 0;
                    char fn[512];
                    std::snprintf(fn, sizeof fn, "%s/persist_s%04llu_%llx_%ux%ux%u.raw",
                                  dd ? dd : ".", (unsigned long long)sub,
                                  (unsigned long long)kv.first, s.w, s.h, s.volume_depth);
                    if (FILE* f = std::fopen(fn, "wb")) {
                        std::fwrite(px.data(), 1, px.size(), f);
                        std::fclose(f);
                    }
                    fprintf(stderr,
                            "[persist] submit=%llu addr=0x%llx volume %ux%ux%u fmt=%u "
                            "raw_nonzero_bytes=%zu/%zu\n",
                            (unsigned long long)sub, (unsigned long long)kv.first, s.w, s.h,
                            s.volume_depth, (unsigned)s.format, nz, px.size());
                    continue;
                }
                const std::vector<uint8_t> rgba = inspection_rgba8(px, s.w, s.h, fmt);
                size_t rgbnz = 0;
                for (size_t p = 0; p + 3 < rgba.size(); p += 4)
                    if (rgba[p] || rgba[p + 1] || rgba[p + 2]) rgbnz++;
                // "Black" and "empty" are different findings, and `rgb_nonblack` alone
                // cannot tell them apart: an HDR target whose texels are negative,
                // non-finite or below 1/510 converts to black from bits that are NOT
                // zero, which is a different defect signature from a zero-filled buffer
                // (#1905 recorded the wrong one from the converted count alone). Report
                // the PRE-conversion evidence beside it: how many raw bytes are non-zero,
                // and the first non-zero texel's own bytes, so the row a lane writes down
                // is a measurement.
                size_t rawnz = 0;
                size_t first_nz_texel = 0;
                bool have_first = false;
                const uint32_t texel_bytes = px.size() && s.w && s.h
                    ? (uint32_t)(px.size() / ((size_t)s.w * s.h)) : 0;
                for (size_t i = 0; i < px.size(); i++) {
                    if (!px[i]) continue;
                    rawnz++;
                    if (!have_first && texel_bytes) {
                        first_nz_texel = i / texel_bytes;
                        have_first = true;
                    }
                }
                // Two distinct sentinels, not one: "none" is the finding "no byte of this
                // target is non-zero", while an unprintable texel stride is the
                // instrument declining to answer. Sharing one string would let the
                // second read as the first — a silent zero, which is the class of lie
                // this whole diagnostic exists to stop reporting. Unreachable today
                // (every format the census dumps is <= 16 B/texel), so the point is that
                // it stays unambiguous if a wider one ever appears.
                char first_text[96] = "none";
                if (have_first && (!texel_bytes || texel_bytes > 16))
                    std::snprintf(first_text, sizeof first_text,
                                  "unprintable (%u bytes/texel)", texel_bytes);
                if (have_first && texel_bytes && texel_bytes <= 16) {
                    int at = std::snprintf(first_text, sizeof first_text,
                                           "texel %zu =", first_nz_texel);
                    for (uint32_t b = 0; b < texel_bytes && at > 0 &&
                                        at < (int)sizeof first_text - 4; b++)
                        at += std::snprintf(first_text + at, sizeof first_text - at,
                                            " %02x", px[first_nz_texel * texel_bytes + b]);
                }
                char fn[512];
                std::snprintf(fn, sizeof fn, "%s/persist_s%04llu_%llx_%ux%u.bmp",
                              dd ? dd : ".", (unsigned long long)sub,
                              (unsigned long long)kv.first, s.w, s.h);
                prosper::test::dump_bmp(fn, rgba, s.w, s.h);
                fprintf(stderr, "[persist] submit=%llu addr=0x%llx %ux%u fmt=%u "
                        "rgb_nonblack=%zu/%u raw_nonzero_bytes=%zu/%zu first_nonzero=%s\n",
                        (unsigned long long)sub,
                        (unsigned long long)kv.first, s.w, s.h, (unsigned)s.format,
                        rgbnz, s.w * s.h, rawnz, px.size(), first_text);
            }
            if (g_persist_filter.state ==
                    prosper::frontend::PersistentReadbackFilterState::Selected)
                for (uint64_t addr : g_persist_filter.addresses)
                    if (!g_rtt.contains(addr))
                        fprintf(stderr, "[persist] submit=%llu addr=0x%llx "
                                        "not retained by the renderer\n",
                                (unsigned long long)sub,
                                (unsigned long long)addr);
        }
    }
}

void note_mrt_census(MrtCensusContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& items = ctx.items;
    auto& pass_i = ctx.pass_i;
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
}

void note_mrt_alias_mirrors(MrtAliasMirrorContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& items = ctx.items;
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
}

void note_ds_viewport_extent(DsViewportExtentContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& pass = ctx.pass;
    auto& max_native_w = ctx.max_native_w;
    auto& max_native_h = ctx.max_native_h;
    auto& viewport_native_w = ctx.viewport_native_w;
    auto& viewport_native_h = ctx.viewport_native_h;
    auto& viewport_extent_valid = ctx.viewport_extent_valid;
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
}

void publish_prefix_inspection(PrefixInspectionContext& ctx) {
    // Every name the moved body used from the pass loop, bound once to the same object.
    auto& rendered_pixels = ctx.rendered_pixels;
    auto& gw = ctx.gw;
    auto& gh = ctx.gh;
    auto& pass_format = ctx.pass_format;
    auto& px_last = ctx.px_last;
    auto& px_last_source_submit = ctx.px_last_source_submit;
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

} // namespace prosper::frontend::submit_renderer
