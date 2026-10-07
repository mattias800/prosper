// select_final_span_present -- see final_span_present.hpp. Moved verbatim out of live_renderer.cpp (#3892).
#include "shared/live/submit_renderer/final_span_present.hpp"
#include "shared/present/compute_scanout.hpp"          // #3915: GPU present of a compute-written scanout

namespace prosper::frontend::submit_renderer {

void select_final_span_present(FinalSpanPresentContext& ctx) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& g_rtt = ctx.g_rtt;
    auto& w = ctx.w;
    auto& h = ctx.h;
    auto& selected_pixels = ctx.selected_pixels;
    auto& selected_source_submit = ctx.selected_source_submit;
    auto& frame_origin = ctx.frame_origin;
    auto& published_gpu = ctx.published_gpu;
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
    auto& px_front_fmt = ctx.px_front_fmt;
    auto& px_vo_fmt = ctx.px_vo_fmt;
    auto& px_last_fmt = ctx.px_last_fmt;
    // Fail-visible: a submit that renders yet offers no present-extent source is exactly the
    // Frontiers wall, and before #1968's diagnostics nothing anywhere said so — the publish
    // counter simply stopped while the guest kept submitting. Never expected, so it reports
    // unconditionally; capped per the rate-limit contract (ordinal on every line so the last
    // one bounds the population from below, plus a power-of-two tail so the tail exists at
    // all — src/diagnostics/diag_ratelimit.hpp, instrument trap 49).
    //
    // The two totals on every line settle a question the frame counter cannot: whether a title
    // whose publish rate looks healthy is publishing FRESH frames or re-serving one retained
    // frame. `frame_seq` climbing says nothing about which (instrument trap 90), and the answer
    // decides where the next investigation looks, so the counters are carried by the instrument
    // rather than left to be inferred.
    static std::atomic<uint64_t> present_frames_stored{0};   // publishable, this submit's own
    static std::atomic<uint64_t> present_frames_served{0};   // the retained frame, re-served
    static std::atomic<uint64_t> present_extent_reports{0};
    auto report_present_extent_shortfall = [&](const char* outcome, size_t offered_bytes) {
        const uint64_t ord = present_extent_reports.fetch_add(1) + 1;
        if (!prosper::diag_should_print(ord)) return;
        auto describe = [](char* out, size_t n, uint32_t cw, uint32_t ch, uint64_t cbase) {
            if (!cw || !ch) snprintf(out, n, "none");
            else snprintf(out, n, "%ux%u@0x%llx", cw, ch, (unsigned long long)cbase);
        };
        char front_text[64], vo_text[64], last_text[64];
        describe(front_text, sizeof front_text, px_front_w, px_front_h, px_front_base);
        describe(vo_text, sizeof vo_text, px_vo_w, px_vo_h, px_vo_base);
        describe(last_text, sizeof last_text, px_last_w, px_last_h, px_last_base);
        fprintf(stderr,
                "[rtt] PRESENT SOURCE EXTENT MISMATCH #%llu: no pass produced a %ux%u "
                "(%zu-byte) present source — px_front=%s px_vo=%s px_last=%s, offered %zu "
                "bytes; %s (published so far: fresh=%llu retained=%llu)\n",
                (unsigned long long)ord, w, h, present_extent_bytes,
                front_text, vo_text, last_text, offered_bytes, outcome,
                (unsigned long long)present_frames_stored.load(std::memory_order_relaxed),
                (unsigned long long)present_frames_served.load(std::memory_order_relaxed));
    };
    auto cached_scanout = [&](uint64_t addr) -> const RttSurf* {
        auto it = g_rtt.find(addr);
        if (it == g_rtt.end() || it->second.w != w || it->second.h != h ||
            it->second.format != VK_FORMAT_R8G8B8A8_UNORM ||
            it->second.volume_depth)
            return nullptr;
        RttSurf& surface = it->second;
        const size_t expected = static_cast<size_t>(w) * h * 4;
        if ((!surface.rgba || surface.rgba->size() != expected) &&
            surface.has_uniform_color)
            materialize_uniform_rtt(surface);
        if ((!surface.rgba || surface.rgba->size() != expected) && surface.gpu_valid) {
            std::vector<uint8_t> materialized;
            std::string error;
            if (prosper::test::readback_persistent_color_target(
                    addr, w, h, surface.format, materialized, error) &&
                materialized.size() == expected) {
                surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                    std::move(materialized));
            } else {
                static std::atomic<int> warned{0};
                if (warned.fetch_add(1) < 24)
                    std::fprintf(stderr,
                                 "[rtt] final scanout readback failed: base=0x%llx "
                                 "extent=%ux%u error=%s\n",
                                 static_cast<unsigned long long>(addr), w, h,
                                 error.c_str());
                return nullptr;
            }
        }
        return surface.rgba && surface.rgba->size() == expected ? &surface : nullptr;
    };
    prosper::VideoOutBufferSnapshot front_snapshot;
    const bool have_front = prosper::videoout_front_snapshot(front_snapshot);
    const int front = have_front ? front_snapshot.buffer_index : -1;
    // Selection, address, registration generation, and originating HLE flip are one
    // registry-locked snapshot. The global flip and present counters can cross-pair
    // when guest and GPU flip submitters interleave, so neither may label this image.
    const uint64_t front_flip = have_front ? front_snapshot.source_flip_seq : 0;
    static uint64_t last_gpu_publish_flip = UINT64_MAX;
    const uint64_t current_flip = prosper_vo_flip_count();
    const bool new_gpu_flip = front_flip && prosper::frontend::present_blit_has_new_flip(
        last_gpu_publish_flip, front_flip);
    PresentHandoffTrace handoff_trace(current_flip);
    if (handoff_trace.active)
        handoff_trace.emit(prosper::perf::PresentHandoffEvent::RendererGate, front, 0, prosper::gpu::present_count(),
                           front >= 0 ? prosper_vo_buffer_addr(front) : 0);
    // GPU present (#1270): when prosper-app has adopted this device and is consuming the
    // front-buffer image directly, blit it into a scanout slot on the GPU and SKIP the CPU
    // readback+reupload entirely. gpu_present_active() is false in every headless/test/
    // screenshot process, so this whole branch is inert there and the CPU path below is
    // byte-for-byte unchanged. On a MISS (image not resident/valid this frame) this falls
    // through to the CPU readback below, which still publishes a CPU frame; prosper-app
    // presents that CPU frame when no GPU frame was published (main.cpp), so a miss degrades to
    // the CPU present path rather than freezing the window.
    // A submit whose one-shot capture is pending presents through the CPU path below,
    // which materializes the scanout on demand: the capsule needs that frame as its
    // output oracle (#3895). Every other submit keeps GPU present.
    const bool gpu_present_now = prosper::frontend::gpu_present_allowed_during_capture(
        prosper::gpu::gpu_present_active(),
        prosper::gpu::gpu_capture_requires_cpu_output());
    using prosper::frontend::GpuPresentOutcome;
    GpuPresentOutcome gpu_outcome = GpuPresentOutcome::Published;
    // Diagnostic context for a decline line; filled as far as the checks get.
    uint64_t decline_va = front_snapshot.address;
    const RttSurf* decline_surf = nullptr;
    VkFormat decline_fmt = VK_FORMAT_UNDEFINED;
    bool decline_have_target = false;
    if (!prosper::gpu::gpu_present_active()) {
        gpu_outcome = GpuPresentOutcome::Inactive;
    } else if (!gpu_present_now) {
        gpu_outcome = GpuPresentOutcome::CaptureNeedsCpu;
    } else if (front < 0) {
        gpu_outcome = GpuPresentOutcome::NoFront;
    } else if (!front_flip) {
        gpu_outcome = GpuPresentOutcome::NoFlipIdentity;
    } else if (!new_gpu_flip) {
        // The previously published slot remains the correct scanout for this guest
        // flip. Treat it as a successful GPU publication so intermediate render
        // submissions do not fall through to the expensive CPU readback path.
        published_gpu = true;
        gpu_outcome = GpuPresentOutcome::SameFlip;
        handoff_trace.emit(prosper::perf::PresentHandoffEvent::SameFlipSuppressed, 0, 0, last_gpu_publish_flip);
    } else {
        const uint64_t front_va = front_snapshot.address;
        auto rit = g_rtt.find(front_va);
        if (rit != g_rtt.end()) decline_surf = &rit->second;
        if (rit == g_rtt.end()) {
            gpu_outcome = GpuPresentOutcome::NoRenderTarget;
        } else if (rit->second.volume_depth) {
            gpu_outcome = GpuPresentOutcome::VolumeTarget;
        } else if (!rit->second.gpu_valid || !rit->second.w || !rit->second.h) {
            gpu_outcome = GpuPresentOutcome::NotGpuResident;
        } else {
            const VkFormat fmt = prosper::test::backend_color_format(rit->second.format);
            decline_fmt = fmt;
            // The cache key is only a lookup hint. Hold its resource-domain lock from
            // the exact image/provenance snapshot through the synchronous scanout copy;
            // eviction or allocation reuse must not substitute another image mid-handoff.
            std::lock_guard resource_lock(
                prosper::test::backend_persistent_resource_mutex());
            prosper::test::PersistentColorTargetImage* tgt =
                prosper::test::find_persistent_color_target(
                    front_va, rit->second.w, rit->second.h, fmt);
            decline_have_target = tgt != nullptr;
            if (!tgt || !tgt->image) {
                gpu_outcome = GpuPresentOutcome::NoPersistentImage;
            } else if (tgt->layout == VK_IMAGE_LAYOUT_UNDEFINED) {
                gpu_outcome = GpuPresentOutcome::UndefinedLayout;
            } else {
                published_gpu = prosper::frontend::present_blit_publish(
                    tgt->image, tgt->layout, fmt, rit->second.w, rit->second.h,
                    front_flip,
                    prosper::test::persistent_color_producer_source(*tgt),
                    &front_snapshot);
                if (published_gpu) last_gpu_publish_flip = front_flip;
                else gpu_outcome = GpuPresentOutcome::PublishFailed;
            }
        }
    }
    // #3915: a display buffer written by a compute dispatch has no render target, but
    // the compute backend may have left a GPU mirror of exactly its bytes. The mirror
    // stands in only where the CPU fallback would itself present the guest buffer: no
    // renderer entry at the front address at all. Any entry, a pixel-less tombstone
    // included, makes the CPU path keep the previous frame, so the decision declines it
    // (RendererOwnsTarget) and the renderer's own reason stands.
    if (gpu_outcome == GpuPresentOutcome::NoRenderTarget ||
        gpu_outcome == GpuPresentOutcome::NotGpuResident) {
        // The CPU path's own priority (cached_scanout below): a render target at ANY
        // registered buffer that it could serve beats the guest buffer.
        const size_t scanout_bytes = static_cast<size_t>(w) * h * 4;
        bool renderer_scanout = false;
        for (int i = 0; i < prosper_vo_buffer_count() && !renderer_scanout; ++i) {
            auto it = g_rtt.find(prosper_vo_buffer_addr(i));
            renderer_scanout = it != g_rtt.end() && it->second.w == w &&
                it->second.h == h && it->second.format == VK_FORMAT_R8G8B8A8_UNORM &&
                !it->second.volume_depth &&
                ((it->second.rgba && it->second.rgba->size() == scanout_bytes) ||
                 it->second.has_uniform_color || it->second.gpu_valid);
        }
        prosper::frontend::ComputeScanoutPresentInputs renderer_inputs;
        renderer_inputs.have_selected_pixels = static_cast<bool>(selected_pixels);
        renderer_inputs.renderer_scanout = renderer_scanout;
        renderer_inputs.renderer_owns_front = decline_surf != nullptr;
        renderer_inputs.present_extent_bytes = present_extent_bytes;
        renderer_inputs.display_bytes =
            static_cast<uint64_t>(prosper::gpu::present_width()) *
            prosper::gpu::present_height() * 4u;
        const auto compute = prosper::frontend::compute_scanout_publish(
            front_snapshot, front_flip, renderer_inputs);
        using prosper::frontend::ComputeScanoutPresent;
        switch (compute.decision) {
        case ComputeScanoutPresent::Publish:
            gpu_outcome = compute.published ? GpuPresentOutcome::Published
                                            : GpuPresentOutcome::PublishFailed;
            break;
        case ComputeScanoutPresent::Absent:               // keep the renderer's reason
        case ComputeScanoutPresent::RendererOwnsTarget: break;
        case ComputeScanoutPresent::Stale:
            gpu_outcome = GpuPresentOutcome::ComputeScanoutStale; break;
        case ComputeScanoutPresent::Unwatched:
            gpu_outcome = GpuPresentOutcome::ComputeScanoutUnwatched; break;
        case ComputeScanoutPresent::ExtentMismatch:
            gpu_outcome = GpuPresentOutcome::ComputeScanoutExtentMismatch; break;
        case ComputeScanoutPresent::FormatMismatch:
            gpu_outcome = GpuPresentOutcome::ComputeScanoutFormatMismatch; break;
        case ComputeScanoutPresent::TileMismatch:
            gpu_outcome = GpuPresentOutcome::ComputeScanoutTileMismatch; break;
        case ComputeScanoutPresent::RendererSource:
            gpu_outcome = GpuPresentOutcome::ComputeScanoutRendererSource; break;
        case ComputeScanoutPresent::ScaledPresent:
            gpu_outcome = GpuPresentOutcome::ComputeScanoutScaled; break;
        case ComputeScanoutPresent::Count: break;          // never a decision
        }
        if (compute.published) {
            published_gpu = true;
            last_gpu_publish_flip = front_flip;
        }
    }
    if (prosper::frontend::gpu_present_outcome_is_decline(gpu_outcome)) {
        prosper::diagnostics::perf::note_present_decline(
            static_cast<size_t>(gpu_outcome),
            prosper::frontend::gpu_present_outcome_name(gpu_outcome));
        static_assert(static_cast<size_t>(GpuPresentOutcome::Count) <=
                          prosper::diagnostics::perf::kPresentDeclineSlots,
                      "every GPU-present outcome needs a perf-ledger decline slot");
        // Budgeted per reason, so one noisy reason cannot hide the first of another.
        // Counted per final render SPAN; a flip can end several, so this is not a
        // count of fallback presents (that is the fps line's cpu-fallback count).
        static std::atomic<uint64_t> decline_reports[(size_t)GpuPresentOutcome::Count]{};
        const uint64_t ord = decline_reports[(size_t)gpu_outcome].fetch_add(1) + 1;
        if (prosper::diag_should_print(ord))
            fprintf(stderr,
                    "[present] GPU PRESENT DECLINED #%llu: %s -- front=%d flip=%llu "
                    "va=0x%llx rtt=%s%ux%u fmt=%d gpu_valid=%d uniform=%d cpu_px=%d "
                    "persistent=%d display=%ux%u; this span's frame goes to the CPU fallback\n",
                    (unsigned long long)ord,
                    prosper::frontend::gpu_present_outcome_name(gpu_outcome), front,
                    (unsigned long long)front_flip, (unsigned long long)decline_va,
                    decline_surf ? "" : "none:", decline_surf ? decline_surf->w : 0,
                    decline_surf ? decline_surf->h : 0,
                    decline_surf ? (int)decline_surf->format : (int)decline_fmt,
                    decline_surf ? (int)decline_surf->gpu_valid : 0,
                    decline_surf ? (int)decline_surf->has_uniform_color : 0,
                    decline_surf && decline_surf->rgba ? 1 : 0,
                    (int)decline_have_target, w, h);
    }
    if (!published_gpu) handoff_trace.emit(prosper::perf::PresentHandoffEvent::CpuFallbackNeeded);
    const RttSurf* scanout = (!published_gpu && front >= 0)
        ? cached_scanout(prosper_vo_buffer_addr(front)) : nullptr;
    if (!published_gpu && !scanout) {
        for (int i = 0; i < prosper_vo_buffer_count(); ++i)
            if ((scanout = cached_scanout(prosper_vo_buffer_addr(i)))) break;
    }
    if (scanout) {
        const auto prior_selected_pixels = selected_pixels;
        const uint64_t prior_source_submit = selected_source_submit;
        selected_pixels = scanout->rgba;
        // Only the same immutable allocation from this callback's completed pass
        // inherits its submit. A cached/materialized surface with the same address but
        // different storage has no source proof here.
        selected_source_submit = selected_pixels && selected_pixels == prior_selected_pixels
            ? prior_source_submit : 0;
    }
    // Last resort before the retained frame: the flipped buffer's own guest memory.
    // The decision itself lives in guest_scanout_present.hpp so it can be unit-tested;
    // read that header for why each condition is there. Only the mechanism is here.
    //
    // Read once per guest flip, not once per submit: this callback runs for every span
    // of every submit (eleven per frame on Frontiers), the buffer only changes when the
    // guest flips a new one, and a 4K copy plus de-swizzle is not free. These statics
    // have the same single-present-thread contract as `last_scanout_present` below.
    static uint64_t guest_scanout_flip = UINT64_MAX;
    static std::shared_ptr<const std::vector<uint8_t>> guest_scanout_pixels;
    const size_t display_bytes =
        static_cast<size_t>(prosper::gpu::present_width()) *
        prosper::gpu::present_height() * 4u;
    if (prosper::frontend::guest_scanout_read_warranted(
            published_gpu, scanout != nullptr, (bool)selected_pixels,
            present_extent_bytes, display_bytes) ==
        prosper::frontend::GuestScanoutDecision::Publish) {
        if (guest_scanout_flip != current_flip) {
            guest_scanout_flip = current_flip;
            guest_scanout_pixels.reset();
            prosper::VideoOutLinearRead read;
            const bool got = prosper::videoout_read_front_linear(read);
            // Named rather than inlined so the #2932 probe below can REPORT it. Whether
            // the renderer owns a target at the flipped VA is the first thing
            // guest_scanout_publishable tests, and it short-circuits ahead of
            // guest_authored -- so a decline of SkipNotAuthored is itself evidence that
            // this was false at that flip. Reading that off the decision name is
            // inference; printing the input is measurement.
            const bool renderer_owns_flip =
                got && g_rtt.find(read.metadata.address) != g_rtt.end();
            const auto decision = prosper::frontend::guest_scanout_publishable(
                got ? read.pixels.size() : 0u, present_extent_bytes,
                got && read.metadata.address != 0,
                renderer_owns_flip,
                read.guest_authored);
            if (decision == prosper::frontend::GuestScanoutDecision::Publish)
                guest_scanout_pixels = std::make_shared<const std::vector<uint8_t>>(
                    std::move(read.pixels));
            // Report the OUTCOME, publish or decline, and name which. A negative
            // control that can only observe silence cannot tell "correctly declined"
            // from "never reached" — and a decline is exactly what the Bendy
            // (PPSA27616) reused-memory case must produce, so it has to be legible.
            //
            // Budgeted PER DECISION, which is the whole point on this line and not
            // ceremony (diag_ratelimit.hpp: "budget per key, or a noisy key exhausts the
            // log before the one under investigation gets a line"). A title that
            // declines on every flip would otherwise push the first PUBLISH past the
            // cap, and "no publish line" is exactly the reading the negative control
            // rests on. With its own counter the first publish is ordinal 1 and always
            // prints, so the absence of a `publish` line means zero publishes.
            static std::atomic<uint64_t>
                guest_scanout_reports[(size_t)
                    prosper::frontend::GuestScanoutDecision::SkipNotAuthored + 1]{};
            const uint64_t ord =
                guest_scanout_reports[(size_t)decision].fetch_add(1) + 1;
            if (prosper::diag_should_print(ord))
                fprintf(stderr,
                        "[rtt] GUEST SCANOUT #%llu: no present source and no renderer "
                        "target at the flipped buffer 0x%llx — %s (%ux%u tiling=%u "
                        "authored=%d footprint=%s)\n",
                        (unsigned long long)ord,
                        (unsigned long long)read.metadata.address,
                        prosper::frontend::guest_scanout_decision_name(decision), w, h,
                        read.metadata.tiling_mode, (int)read.guest_authored,
                        read.padded_footprint ? "padded" : "nominal");
            // #2932: what is actually true of the flipped buffer at the moment we
            // decline to publish it? Three facts, printed together because separately
            // each one invites a wrong inference:
            //
            //   phys=      the physical page behind the flipped VA. This began as a test
            //              of "scanout and the render targets are two mappings of one
            //              allocation"; that is FALSE (STRAY_STATUS.md, Ruled out).
            //   aliases=N  the SEARCHED DOMAIN. The table is populated only while
            //              page-protection watches are armed, so N=0 means "nothing was
            //              searched" and says nothing about this VA. A miss reported
            //              without its domain is how an unarmed instrument gets recorded
            //              as a negative result.
            //   rtt_owns=  whether g_rtt holds a target at this VA -- the FIRST input
            //              guest_scanout_publishable tests, short-circuiting ahead of
            //              guest_authored. Measured, not inferred from the decision name.
            //
            // Deliberately NOT asserted here: that no VA-keyed lookup can connect the
            // flipped buffer to a render target. Stray's scanout VAs appear in the
            // colour-target census as 4K attachments in passes carrying draws, so that
            // sentence -- which earlier revisions of this comment stated as fact -- is
            // withdrawn. rtt_owns is printed precisely so nobody has to assume it again.
            if (read.metadata.address && prosper::diag_should_print(ord)) {
                uint64_t flip_phys = 0;
                size_t alias_n = 0;
                const bool ok = prosper::host::guest_write_watch_va_to_phys(
                    read.metadata.address, flip_phys, &alias_n);
                if (ok)
                    fprintf(stderr,
                            "[rtt] GUEST SCANOUT #%llu phys: va=0x%llx -> phys=0x%llx "
                            "(aliases=%zu rtt_owns=%d decision=%s)\n",
                            (unsigned long long)ord,
                            (unsigned long long)read.metadata.address,
                            (unsigned long long)flip_phys, alias_n,
                            (int)renderer_owns_flip,
                            prosper::frontend::guest_scanout_decision_name(decision));
                else
                    fprintf(stderr,
                            "[rtt] GUEST SCANOUT #%llu phys: va=0x%llx -> UNRESOLVED "
                            "(aliases=%zu rtt_owns=%d decision=%s; %s)\n",
                            (unsigned long long)ord,
                            (unsigned long long)read.metadata.address, alias_n,
                            (int)renderer_owns_flip,
                            prosper::frontend::guest_scanout_decision_name(decision),
                            alias_n ? "no alias range covers this VA"
                                    : "alias table EMPTY -- nothing was searched");
            }
        }
        // Re-check the size on the cached path too: a two-set geometry switch can change
        // the present extent between spans of one flip, and the cache is keyed on the
        // flip alone.
        if (guest_scanout_pixels && guest_scanout_pixels->size() == present_extent_bytes) {
            selected_pixels = guest_scanout_pixels;
            frame_origin = prosper::gpu::PresentFrameOrigin::GuestScanout;
            selected_source_submit = 0;
        }
    }
    // Hold the last good scanout across VideoOut buffer rotation. Bendy (PPSA27616) rapidly
    // re-registers its scanout buffers; on the frames where the guest has registered new
    // buffers but not yet rendered+flipped them, none is gpu_valid and the present would be
    // a black flicker. Presenting the previous frame instead keeps a stable image (the new
    // buffers get drawn and flipped within a frame or two). CONFIDENCE: MED.
    // Thread-safety: this static is a plain (non-atomic) shared_ptr, correct only under the
    // renderer's single present thread (this callback and its sibling statics — dp_submit,
    // warned, frame_no — all assume the one serialized present path). It must not be read or
    // assigned from another thread; a concurrent present would race the object assignment.
    // The net's own test used to be `!empty()`, which is not the publish predicate (#1986):
    // a non-empty WRONG-EXTENT frame took the store branch, overwrote the retained good
    // frame, and made the recovery branch unreachable for the rest of the process. Sonic
    // Frontiers' 4K publish wall was permanent for exactly that reason. Retaining is now
    // the same predicate as publishing, so a frame prosper could not publish can never
    // become the frame it serves later.
    static std::shared_ptr<const std::vector<uint8_t>> last_scanout_present;
    // Provenance is a property of the RETAINED frame, not of the submit that re-serves
    // it. Post-wall, Frontiers re-serves one retained frame thousands of times (fresh +1
    // against retained +2,048 between two shortfall ordinals), so a label that reset on
    // re-serve would report the guest's own buffer as a composited frame for all but the
    // first of them — and `--require-composited-frame` would pass on the strength of the
    // copies.
    static bool last_scanout_present_from_guest = false;
    static uint64_t last_scanout_source_submit = 0;
    if (!published_gpu) {
        const size_t current_bytes = selected_pixels ? selected_pixels->size() : 0u;
        const size_t retained_bytes =
            last_scanout_present ? last_scanout_present->size() : 0u;
        switch (prosper::frontend::retained_frame_action(
                    current_bytes, retained_bytes, present_extent_bytes)) {
            case prosper::frontend::RetainedFrameAction::StoreCurrent:
                last_scanout_present = selected_pixels;
                last_scanout_present_from_guest =
                    frame_origin == prosper::gpu::PresentFrameOrigin::GuestScanout;
                last_scanout_source_submit = selected_source_submit;
                present_frames_stored.fetch_add(1, std::memory_order_relaxed);
                // `retained_frame_action` decides on BYTE SIZES ALONE -- by design, since
                // it cannot know what a frame should look like. The consequence is that a
                // correctly-sized but CONTENTLESS frame is retained just as readily as a
                // good one, and is then re-served on every later submit that produces no
                // present source. Little Nightmares III presents a uniform (255,255,0)
                // frame on ~2/3 of samples for exactly that reason (#2014): the retained
                // frame is itself uniform.
                //
                // This names the ORIGIN of a uniform retained frame, which is the one
                // fact the existing counters cannot supply -- `fresh=N retained=M` says a
                // substitution happened, not what was substituted or where it came from.
                // Sampled, not exhaustive: a full uniformity scan of a 33 MB frame on the
                // present path would cost more than the render.
                if (const char* uni = PROSPER_ENV_VALUE("PROSPER_UNIFORMLOG")) {
                    if (uni[0] == '1' && uni[1] == '\0' && selected_pixels &&
                        selected_pixels->size() >= 4) {
                        const uint8_t* px = selected_pixels->data();
                        const size_t n = selected_pixels->size();
                        bool uniform = true;
                        for (size_t off = 4; off + 4 <= n; off += ((n / 4096) | 4u) & ~3u)
                            if (px[off] != px[0] || px[off+1] != px[1] ||
                                px[off+2] != px[2] || px[off+3] != px[3]) {
                                uniform = false; break;
                            }
                        if (uniform) {
                            static std::atomic<uint64_t> uniform_stores{0};
                            const uint64_t ord =
                                uniform_stores.fetch_add(1, std::memory_order_relaxed) + 1;
                            // WHICH pass handed us the uniform frame. `origin=` only
                            // separates a composite from a republished guest scanout; it
                            // cannot say which colour target the composite read back, and
                            // that is the one fact needed to go from "a uniform frame was
                            // retained" to "this surface is uniform" (#2014).
                            using prosper::frontend::PresentSourceChoice;
                            const uint32_t src_w =
                                present_choice == PresentSourceChoice::Front ? px_front_w
                              : present_choice == PresentSourceChoice::Vo    ? px_vo_w
                              : present_choice == PresentSourceChoice::Last  ? px_last_w : 0u;
                            const uint32_t src_h =
                                present_choice == PresentSourceChoice::Front ? px_front_h
                              : present_choice == PresentSourceChoice::Vo    ? px_vo_h
                              : present_choice == PresentSourceChoice::Last  ? px_last_h : 0u;
                            const uint64_t src_base =
                                present_choice == PresentSourceChoice::Front ? px_front_base
                              : present_choice == PresentSourceChoice::Vo    ? px_vo_base
                              : present_choice == PresentSourceChoice::Last  ? px_last_base : 0u;
                            const uint32_t src_fmt =
                                present_choice == PresentSourceChoice::Front ? px_front_fmt
                              : present_choice == PresentSourceChoice::Vo    ? px_vo_fmt
                              : present_choice == PresentSourceChoice::Last  ? px_last_fmt
                                                                            : kNoPassFormat;
                            // A VkFormat, not the raw CB_COLOR0_INFO.FORMAT:
                            // `ResolvedPipelineState::color0_format` is already
                            // `vk_color_format(...)`'s output (render_state.cpp:789).
                            // Naming it `cbfmt` cost a wrong reading once -- 44 is
                            // VK_FORMAT_B8G8R8A8_UNORM, which is not a CB format value
                            // at all.
                            char fmt_text[16];
                            if (src_fmt == kNoPassFormat) snprintf(fmt_text, sizeof fmt_text, "none");
                            else snprintf(fmt_text, sizeof fmt_text, "%u", src_fmt);
                            if (prosper::diag_should_print(ord))
                                fprintf(stderr,
                                        "[uniformlog] #%llu retaining a UNIFORM frame "
                                        "rgba=(%u,%u,%u,%u) bytes=%zu origin=%s src=%s "
                                        "%ux%u@0x%llx vkfmt=%s\n",
                                        (unsigned long long)ord,
                                        px[0], px[1], px[2], px[3], n,
                                        frame_origin ==
                                            prosper::gpu::PresentFrameOrigin::GuestScanout
                                            ? "GuestScanout" : "Composited",
                                        prosper::frontend::present_source_name(present_choice),
                                        src_w, src_h, (unsigned long long)src_base, fmt_text);
                        }
                    }
                }
                break;
            case prosper::frontend::RetainedFrameAction::ServeRetained:
                selected_pixels = last_scanout_present;
                frame_origin = last_scanout_present_from_guest
                    ? prosper::gpu::PresentFrameOrigin::GuestScanout
                    : prosper::gpu::PresentFrameOrigin::Composited;
                selected_source_submit = last_scanout_source_submit;
                present_frames_served.fetch_add(1, std::memory_order_relaxed);
                // A stale-but-correct frame IS what the guest sees, so say so rather than
                // letting a silent substitution read as a healthy present. Under no extent
                // contract this branch is the historical empty-frame recovery and is not
                // itself a defect, so it stays quiet there.
                if (present_extent_bytes)
                    report_present_extent_shortfall(
                        "serving the retained frame instead", current_bytes);
                break;
            case prosper::frontend::RetainedFrameAction::NoUsableFrame:
                // Nothing publishable this submit and nothing retained to fall back on, so
                // the screen does not advance. selected_pixels is left as selection found
                // it — normally empty on this path, since selection already refused every
                // wrong-extent candidate — which keeps the caller's `[render] … Vulkan
                // render FAILED` and `[agc] PUBLISH DROPPED` backstops intact. This is the
                // one outcome that is a lost frame rather than a substituted one, so it is
                // reported whenever the contract applies, including when nothing rendered
                // at all: "no pass produced a present-extent source" is the finding either
                // way, and Frontiers' wall was invisible for exactly as long as it was not
                // stated anywhere.
                if (present_extent_bytes)
                    report_present_extent_shortfall(
                        "nothing is published for this submit", current_bytes);
                break;
        }
    }
    if (PROSPER_ENV_ON("PROSPER_DUMP_PERSISTENT")) {
        size_t nb = 0;
        if (selected_pixels)
            for (size_t p = 0; p + 3 < selected_pixels->size(); p += 4)
                if ((*selected_pixels)[p] || (*selected_pixels)[p+1] || (*selected_pixels)[p+2]) nb++;
        fprintf(stderr, "[persist] present: front=%d front_va=0x%llx scanout=%s rgb_nonblack=%zu\n",
                front, (unsigned long long)(front >= 0 ? prosper_vo_buffer_addr(front) : 0),
                scanout ? "HIT" : "MISS", nb);
    }
}

} // namespace prosper::frontend::submit_renderer
