#include "../tools/screenshot/capture_manifest.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace prosper::screenshot;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

int main() {
    std::printf("== test_capture_manifest ==\n");
    CaptureTracker tracker;
    std::vector<uint8_t> pixels(16 * 8 * 4, 0x11);
    CaptureObservation a{CaptureSource::Rendered, 10, 10, 100, 1, 16, 8, 0x12345678, 1.0};
    auto ca = tracker.observe(a, pixels);
    CHECK(ca.source_advanced && !ca.pixel_identical && ca.stale_seconds == 0,
          "first sample is a new source frame");

    auto b = a; b.elapsed_seconds = 3.5;
    auto cb = tracker.observe(b, pixels);
    CHECK(!cb.source_advanced && cb.pixel_identical && cb.stale_seconds == 2.5 &&
          cb.pixel_stale_seconds == 2.5,
          "same source identity is classified stale");

    auto c = b; c.source_seq = c.frame_seq = 11; c.pixel_crc32 = 0x87654321; c.elapsed_seconds = 4.0;
    c.distinct_rgb_colors = 37;
    c.nonblack_rgb_pixels = 101;
    c.average_hash = 0x0123456789abcdef;
    c.difference_hash = 0xfedcba9876543210;
    c.luma16x9.fill(0x7f);
    pixels[0] = 0x22;
    auto cc = tracker.observe(c, pixels);
    CHECK(cc.source_advanced && !cc.pixel_identical && cc.stale_seconds == 0,
          "new renderer publication resets stale duration");
    auto d = c; d.source_seq = d.frame_seq = 12; d.elapsed_seconds = 6.0;
    auto cd = tracker.observe(d, pixels);
    CHECK(cd.source_advanced && cd.pixel_identical && cd.stale_seconds == 0 &&
          cd.pixel_stale_seconds == 2.0,
          "new source publication can still be visually stale");
    CHECK(tracker.distinct_source_frames() == 3 && tracker.pixel_distinct_frames() == 2 &&
          tracker.rendered_samples() == 4, "tracker reports source, pixel, and composited counts");
    CHECK(tracker.max_stale_seconds() == 2.5 && tracker.max_frame_seq() == 12 &&
          tracker.max_present_count() == 100 && tracker.max_pixel_stale_seconds() == 2.5,
          "tracker preserves assertion maxima");

    const std::string line = manifest_sample_json(2, "a\\b\"c.png", c, cc, "@route\nname");
    CHECK(line.find("a\\\\b\\\"c.png") != std::string::npos, "manifest JSON escapes paths");
    CHECK(line.find("@route\\nname") != std::string::npos, "manifest JSON escapes route text");
    CHECK(line.find("\"source\":\"composited\"") != std::string::npos,
          "manifest names the capture source");
    CHECK(line.find("\"distinct_rgb_colors\":37") != std::string::npos,
          "manifest records a coarse content metric for presented pixels");
    CHECK(line.find("\"nonblack_rgb_pixels\":101") != std::string::npos,
          "manifest records black-frame coverage");
    CHECK(line.find("\"average_hash\":\"0123456789abcdef\"") != std::string::npos &&
          line.find("\"difference_hash\":\"fedcba9876543210\"") != std::string::npos,
          "manifest records tolerant perceptual fingerprints");
    CHECK(line.find("\"luma16x9\":\"7f7f7f7f") != std::string::npos,
          "manifest records a structural luminance signature");

    std::vector<uint8_t> gradient(9 * 8 * 4, 255);
    for (uint32_t y = 0; y < 8; ++y) for (uint32_t x = 0; x < 9; ++x) {
        const size_t offset = (y * 9 + x) * 4;
        gradient[offset] = gradient[offset + 1] = gradient[offset + 2] = static_cast<uint8_t>(x * 28);
    }
    const auto increasing = perceptual_hashes_rgba(gradient, 9, 8);
    CHECK(increasing.difference == 0, "difference hash follows the standard left-greater-than-right rule");
    for (uint32_t y = 0; y < 8; ++y) for (uint32_t x = 0; x < 9; ++x) {
        const size_t offset = (y * 9 + x) * 4;
        gradient[offset] = gradient[offset + 1] = gradient[offset + 2] = static_cast<uint8_t>((8 - x) * 28);
    }
    const auto decreasing = perceptual_hashes_rgba(gradient, 9, 8);
    CHECK(decreasing.difference == UINT64_MAX,
          "difference hash detects the opposite horizontal structure in all 64 cells");
    const auto luma = perceptual_luma16x9_rgba(gradient, 9, 8);
    CHECK(luma.size() == kPerceptualLumaCells && luma.front() > luma.back(),
          "16x9 luminance signature preserves broad spatial structure");
    for (size_t i = 3; i < gradient.size(); i += 4) gradient[i] = 0;
    const auto transparent_luma = perceptual_luma16x9_rgba(gradient, 9, 8);
    CHECK(transparent_luma.front() == 0 && transparent_luma.back() == 0,
          "structural signature ignores invisible RGB beneath zero alpha");

    // Renderer output is presented by prosper-app through an explicitly opaque swapchain. A render
    // target may therefore carry near-zero alpha while its RGB is fully visible on the desktop. The
    // screenshot persistence boundary must match the viewer, while raw guest scanout stays literal.
    std::vector<uint8_t> low_alpha_rendered = {
        220, 30, 10, 0,  20, 180, 40, 1,
        220, 30, 10, 0,  20, 180, 40, 1,
    };
    auto opaque_expected = low_alpha_rendered;
    for (size_t i = 3; i < opaque_expected.size(); i += 4) opaque_expected[i] = 255;
    auto raw_scanout = low_alpha_rendered;
    normalize_capture_rgba(CaptureSource::Rendered, low_alpha_rendered);
    normalize_capture_rgba(CaptureSource::RawScanout, raw_scanout);
    const auto visible_content = measure_pixel_content_rgba(low_alpha_rendered);
    CHECK(low_alpha_rendered == opaque_expected && visible_content.distinct_colors == 2 &&
          visible_content.nonblack_pixels == 4 &&
          perceptual_luma16x9_rgba(low_alpha_rendered, 2, 2) ==
              perceptual_luma16x9_rgba(opaque_expected, 2, 2),
          "opaque rendered capture preserves viewer-visible colors and structure despite low alpha");
    CHECK(raw_scanout[3] == 0 && raw_scanout[7] == 1,
          "raw scanout capture preserves guest alpha bytes");

    // A republished guest scanout (#1968) is a third provenance and needs both halves of its
    // contract asserted. It reaches the desktop through the same opaque swapchain as a composited
    // frame, so it is normalized the same way — otherwise a guest buffer holding RGB content behind
    // low alpha would score as black and read as a regression that the viewer cannot see.
    auto guest_scanout_pixels = low_alpha_rendered;
    for (size_t i = 3; i < guest_scanout_pixels.size(); i += 4) guest_scanout_pixels[i] = (i & 4) ? 1 : 0;
    normalize_capture_rgba(CaptureSource::GuestScanout, guest_scanout_pixels);
    CHECK(guest_scanout_pixels == opaque_expected,
          "a republished guest scanout is normalized like the composited frame it is presented as");
    // …and yet it must never be COUNTED as one: --require-composited-frame is enforced off
    // rendered_samples(), and #2026 was reverted (#2044) for letting exactly this frame satisfy it.
    CaptureTracker provenance_tracker;
    std::vector<uint8_t> tracked(16, 0x40);
    CaptureObservation composited_obs{CaptureSource::Rendered, 1, 1, 1, 0, 2, 2, 0xaa, 0.0};
    CaptureObservation republished_obs{CaptureSource::GuestScanout, 2, 2, 2, 0, 2, 2, 0xbb, 1.0};
    provenance_tracker.observe(republished_obs, tracked);
    CHECK(provenance_tracker.rendered_samples() == 0 &&
              provenance_tracker.guest_scanout_samples() == 1,
          "a republished guest scanout does not satisfy the composited-frame requirement");
    provenance_tracker.observe(composited_obs, tracked);
    CHECK(provenance_tracker.rendered_samples() == 1 &&
              provenance_tracker.guest_scanout_samples() == 1,
          "a genuinely composited frame still counts, so the gate is narrowed and not disabled");
    CHECK(std::string(capture_source_name(CaptureSource::GuestScanout)) == "guest_scanout" &&
              std::string(capture_source_name(CaptureSource::Rendered)) == "composited" &&
              std::string(capture_source_name(CaptureSource::RawScanout)) == "raw_scanout",
          "each capture provenance has its own manifest name");
    const std::string long_route(4096, 'x');
    const std::string long_line = manifest_sample_json(3, "long.png", c, cc, long_route);
    CHECK(long_line.size() > long_route.size() && long_line.find(long_route) != std::string::npos,
          "manifest serialization does not truncate long routes");

    CaptureRunConfig config;
    config.title = "PPSA15552";
    config.input_route = "@scripts/dead-cells/route.pad";
    config.time_mode = true;
    config.seconds = 1;
    config.requested = 120;
    config.render_every = "1000";
    config.render_every_for_ms = "120000";
    config.min_distinct_frames = 10;
    config.min_pixel_distinct_frames = 8;
    config.max_pixel_stale_seconds = 3.5;
    config.require_composited_frame = true;
    config.required_crc32_set = true;
    config.required_crc32 = 0x1234abcd;
    const std::string run = manifest_run_json(config);
    CHECK(run.find("\"capture_mode\":\"wall_seconds\"") != std::string::npos &&
          run.find("\"render_every\":\"1000\"") != std::string::npos &&
          run.find("\"render_every_for_ms\":\"120000\"") != std::string::npos,
          "run header records cadence and renderer policy");
    CHECK(run.find("\"required_crc32\":\"1234abcd\"") != std::string::npos,
          "run header records checkpoint assertions");
    CHECK(run.find("\"min_pixel_distinct_frames\":8") != std::string::npos &&
          run.find("\"max_pixel_stale_seconds\":3.500000") != std::string::npos,
          "run header records pixel-progress assertions");
    // Provenance: the revision the BINARY was compiled from. A lane that checks out new work and
    // measures without rebuilding produces a confident measurement of the wrong build, and nothing
    // else in the artifact can tell -- the source tree looks current and the run succeeds.
    // `tools/revision/check_build_revision.py` is what compares this against a ref.
    // Assert the SHAPE, not merely non-emptiness: a 40-char hex sha or the literal "unknown" that
    // the generator emits outside a git checkout. Non-emptiness alone was near-vacuous, since the
    // field can only be empty if the generator itself is broken.
    {
        const std::string key = "\"build_revision\":\"";
        const size_t at = run.find(key);
        std::string value;
        if (at != std::string::npos) {
            const size_t start = at + key.size();
            const size_t end = run.find('"', start);
            if (end != std::string::npos) value = run.substr(start, end - start);
        }
        const bool sha = value.size() == 40 &&
            value.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos;
        CHECK(sha || value == "unknown",
              "run header records a 40-char revision (or \"unknown\" outside a git checkout)");
    }

    CHECK(run.find("\"allow_guest_fault\":false") != std::string::npos,
          "run header records whether a guest fault was permitted");

    GuestOutcome running_guest;
    const RunVerdict failed_verdict = decide_run_verdict(true, running_guest, false);
    const std::string summary =
        manifest_summary_json(3, 5, SamplingStop::Timeout, tracker, failed_verdict, running_guest,
                              false, prosper::gpu::FrameRate{});
    CHECK(summary.find("\"timed_out\":true") != std::string::npos &&
          summary.find("\"stop_reason\":\"timeout\"") != std::string::npos &&
          summary.find("\"pixel_distinct_frames\":2") != std::string::npos &&
          summary.find("\"max_pixel_stale_seconds\":2.500000") != std::string::npos &&
          summary.find("\"exit_code\":1") != std::string::npos,
          "summary records incomplete failure");

    // #2007: the guest runs on a DETACHED thread, so its BootResult reached only the log. The run
    // then reported `status=ok` and exit 0 for a title that had died 0.4 s into the boot and been
    // sampled 25 times off one stale frame. Three things must hold: the outcome survives the trip
    // from the guest thread, a fault outranks a clean assertion sweep, and the verdict is the same
    // object the summary line, the manifest and the exit status are all built from.
    {
        CHECK(read_guest_outcome().state == GuestRunState::Running,
              "before the guest thread publishes anything the run is 'still running'");

        // The exact values the R-Type Delta reproduction produced on master.
        publish_guest_outcome(2, 0x410024055ull, 0,
                              "SIGSEGV at addr=(nil)  rip=0x410024055 (image+0x24055)");
        const GuestOutcome faulted = read_guest_outcome();
        CHECK(faulted.state == GuestRunState::Faulted && faulted.kind == 2 &&
              faulted.fault_rip == 0x410024055ull && faulted.fault_addr == 0 &&
              faulted.detail.find("image+0x24055") != std::string::npos,
              "a published guest fault survives the trip to the sampling thread intact");

        const RunVerdict fault_verdict = decide_run_verdict(/*assertions_failed=*/false, faulted,
                                                           /*allow_guest_fault=*/false);
        CHECK(std::string(fault_verdict.status) == "GUEST-FAULT" && fault_verdict.exit_code == 1,
              "a dead guest fails the run even though every capture assertion passed");

        // Positive control for the discriminator: the same call with no fault must still be clean,
        // so the check above cannot be satisfied by a verdict function that fails everything.
        const RunVerdict clean = decide_run_verdict(false, running_guest, false);
        CHECK(std::string(clean.status) == "ok" && clean.exit_code == 0,
              "an intact guest with passing assertions is still ok and still exits 0");

        // The opt-out permits the fault without laundering it: exit 0, but never `status=ok`.
        const RunVerdict allowed = decide_run_verdict(false, faulted, /*allow_guest_fault=*/true);
        CHECK(std::string(allowed.status) == "GUEST-FAULT-ALLOWED" && allowed.exit_code == 0,
              "--allow-guest-fault permits the exit code without hiding the fault");
        const RunVerdict allowed_but_failing = decide_run_verdict(true, faulted, true);
        CHECK(std::string(allowed_but_failing.status) == "FAILED" &&
              allowed_but_failing.exit_code == 1,
              "--allow-guest-fault does not disarm the capture assertions");

        const std::string fault_summary =
            manifest_summary_json(25, 25, SamplingStop::RequestSatisfied, tracker, fault_verdict,
                                  faulted, false, prosper::gpu::FrameRate{});
        CHECK(fault_summary.find("\"guest_state\":\"faulted\"") != std::string::npos &&
              fault_summary.find("\"guest_kind\":2") != std::string::npos &&
              fault_summary.find("\"guest_fault_rip\":\"0x410024055\"") != std::string::npos &&
              fault_summary.find("\"status\":\"GUEST-FAULT\"") != std::string::npos &&
              fault_summary.find("\"exit_code\":1") != std::string::npos,
              "the manifest summary carries the fault so batch consumers can filter on it");

        // A guest whose entry RETURNS is reported but is not a failure: titles legitimately exit,
        // and a truncated run is already caught by the saved/requested assertion.
        publish_guest_outcome(0, 0, 0, "entry returned");
        const GuestOutcome returned = read_guest_outcome();
        const RunVerdict returned_verdict = decide_run_verdict(false, returned, false);
        CHECK(returned.state == GuestRunState::Returned &&
              std::string(returned_verdict.status) == "ok" && returned_verdict.exit_code == 0,
              "a guest that exits normally is recorded but does not fail the run");
        const std::string returned_summary =
            manifest_summary_json(5, 5, SamplingStop::RequestSatisfied, tracker, returned_verdict,
                                  returned, false, prosper::gpu::FrameRate{});
        CHECK(returned_summary.find("\"guest_state\":\"returned\"") != std::string::npos &&
              returned_summary.find("\"guest_fault_rip\":null") != std::string::npos,
              "a normal guest exit is still visible in the manifest, with no fault address");
    }

    // #2584: the verdict was right and the LOOP was not — a run whose guest died at 0.4 s went on
    // sampling for 24 more seconds and wrote 24 byte-identical PNGs. The stop condition is the unit
    // under test here; the live arm (a routed PPSA26414 boot with and without
    // --no-stop-after-guest-fault) is what proves the sampler is actually wired to it.
    {
        GuestOutcome running;
        GuestOutcome returned;  returned.state = GuestRunState::Returned;
        GuestOutcome faulted;   faulted.state = GuestRunState::Faulted; faulted.kind = 2;
        const double settle = 1.0;

        CHECK(should_stop_after_guest_fault(faulted, true, 2.0, 2.0, settle),
              "a dead guest and a present layer that has gone quiet stops the sampler");

        // The two arms that separate this from "stop as soon as the guest thread dies". A primary
        // thread's death says nothing about other guest threads or a renderer backlog, so a stop
        // keyed on the fault alone would silently truncate a run that is still producing frames —
        // and the truncation would be indistinguishable from a satisfied request.
        CHECK(!should_stop_after_guest_fault(faulted, true, 2.0, 0.2, settle),
              "a present layer still publishing keeps the sampler running after the fault");
        CHECK(!should_stop_after_guest_fault(faulted, true, 0.2, 2.0, settle),
              "work already in flight gets the settle window before the sampler gives up");

        // Neither non-terminal nor voluntary exit is a stop. `Returned` is the load-bearing one: a
        // title exiting normally must keep tripping the saved/requested assertion (#2007), which
        // stopping here would silence.
        CHECK(!should_stop_after_guest_fault(running, true, 2.0, 2.0, settle),
              "a live guest never stops the sampler, however quiet the present layer is");
        CHECK(!should_stop_after_guest_fault(returned, true, 2.0, 2.0, settle),
              "a guest that exits on its own is not a fault and does not stop the sampler");

        CHECK(!should_stop_after_guest_fault(faulted, /*enabled=*/false, 30.0, 30.0, settle),
              "--no-stop-after-guest-fault samples the full request even from a dead guest");

        // Boundary: the windows are inclusive, so a zero settle stops on the first poll after the
        // fault. Nothing in the tool sets that, but the comparison must not be strict.
        CHECK(should_stop_after_guest_fault(faulted, true, 0.0, 0.0, 0.0),
              "a zero settle window stops as soon as the fault is observed");

        // The stop must be legible in the artifact, or a 3-PNG run reads as a crashed harness.
        const RunVerdict fault_verdict = decide_run_verdict(false, faulted, false);
        const std::string early =
            manifest_summary_json(3, 25, SamplingStop::GuestFault, tracker, fault_verdict, faulted,
                                  false, prosper::gpu::FrameRate{});
        CHECK(early.find("\"saved\":3") != std::string::npos &&
              early.find("\"requested\":25") != std::string::npos &&
              early.find("\"stop_reason\":\"guest-fault\"") != std::string::npos &&
              early.find("\"timed_out\":false") != std::string::npos,
              "a short run states saved, requested and why it is short, and is not a timeout");

        CHECK(std::string(sampling_stop_name(SamplingStop::RequestSatisfied)) ==
                  "request-satisfied" &&
              std::string(sampling_stop_name(SamplingStop::Timeout)) == "timeout" &&
              std::string(sampling_stop_name(SamplingStop::GuestFault)) == "guest-fault",
              "every stop reason has its own name, so the three are distinguishable");

        // The run header records the policy, so an archived manifest says whether the tail was
        // dropped by this stop or was never produced.
        CaptureRunConfig stopped_config;
        stopped_config.stop_after_guest_fault = false;
        stopped_config.guest_fault_settle_seconds = 2.5;
        const std::string stopped_run = manifest_run_json(stopped_config);
        CHECK(stopped_run.find("\"stop_after_guest_fault\":false") != std::string::npos &&
              stopped_run.find("\"guest_fault_settle_seconds\":2.500000") != std::string::npos,
              "the run header records the early-stop policy that produced the artifact set");

        // #2639 review B1. The finding was not the wording: ONE assertion family moves in the
        // PASSING direction under a shortened run, and the live A/B could not have seen it, because
        // neither arm set a `--max-*` flag and both therefore had the assertion off. So construct
        // the hazard by hand, outside whatever produced that null -- the same route, sampled to the
        // end of the request versus cut at the stop.
        {
            std::vector<uint8_t> frame(16 * 8 * 4, 0x44);
            const CaptureObservation shot{CaptureSource::Rendered, 7, 7, 70, 0, 16, 8,
                                          0xdeadbeef, 1.0};
            CaptureTracker full, cut;
            full.observe(shot, frame);
            cut.observe(shot, frame);   // the single sample an early-stopped run takes
            // The 24 byte-identical samples a full-length run goes on to take, one per second, from
            // a guest that died at 0.4 s: same source identity, same pixels, advancing clock.
            for (int second = 2; second <= 25; ++second) {
                CaptureObservation later = shot;
                later.elapsed_seconds = static_cast<double>(second);
                full.observe(later, frame);
            }
            CHECK(full.max_stale_seconds() == 24.0 && full.max_pixel_stale_seconds() == 24.0,
                  "a full-length run measures the whole quiet interval after the guest dies");
            CHECK(cut.max_stale_seconds() == 0.0 && cut.max_pixel_stale_seconds() == 0.0,
                  "an early-stopped run measures NO staleness: the tail it drops IS the evidence "
                  "the staleness bounds are computed from");
            // Which is the flip in full: one bound, two verdicts, same route.
            const double bound = 5.0;
            CHECK(full.max_pixel_stale_seconds() > bound &&
                  !(cut.max_pixel_stale_seconds() > bound),
                  "--max-pixel-stale-seconds 5 FAILS the full run and PASSES the shortened one");
        }

        // So the bound disarms the stop rather than being documented around it.
        CHECK(early_stop_armed(true, -1.0, -1.0),
              "with no staleness bound armed, the early stop is armed");
        CHECK(!early_stop_armed(true, 5.0, -1.0),
              "--max-stale-seconds disarms the early stop");
        CHECK(!early_stop_armed(true, -1.0, 5.0),
              "--max-pixel-stale-seconds disarms the early stop");
        CHECK(!early_stop_armed(true, 0.0, -1.0),
              "a ZERO staleness bound counts as armed: the assertions guard on >= 0, so reading "
              "zero as absent would silently drop the strictest bound of all");
        CHECK(!early_stop_armed(false, -1.0, -1.0),
              "--no-stop-after-guest-fault stays off whatever the assertions ask for");

        // The struct's own default is the policy an archived manifest reports for a producer that
        // does not set it, so it must be the tool's default and not the "stop on the first poll
        // after the fault" that zero means (#2639 review N1).
        CHECK(CaptureRunConfig{}.guest_fault_settle_seconds == 1.0,
              "the settle-window default matches the tool's, so a direct producer gets the same "
              "policy the command line gives");
    }

    // Framerate reaches the artifact, and reaches it as TWO numbers. A manifest that carried only
    // a presented rate would report a healthy figure for a title whose every publication is the
    // renderer's re-served retained frame -- the R-Type Delta (#2783) reading. The metric itself is
    // pinned in tests/gpu/present/test_present_frame_rate.cpp; what is under test here is that the
    // JSON says which number is which, and that a sample line carries the counters a consumer needs
    // to compute the rate over any window of its own choosing.
    {
        CaptureTracker tracker;

        prosper::gpu::PresentRateSnapshot live;
        live.published = 600; live.distinct = 600;
        live.first_publication_seconds = 0; live.now_seconds = 10.0;
        const prosper::gpu::FrameRate live_rate =
            prosper::gpu::frame_rate_since_first_publication(live);

        prosper::gpu::PresentRateSnapshot frozen;
        frozen.published = 600; frozen.distinct = 1;
        frozen.first_publication_seconds = 0; frozen.now_seconds = 10.0;
        const prosper::gpu::FrameRate frozen_rate =
            prosper::gpu::frame_rate_since_first_publication(frozen);

        const RunVerdict ok_verdict{"ok", 0};
        GuestOutcome running;
        const std::string live_summary = manifest_summary_json(
            5, 5, SamplingStop::RequestSatisfied, tracker, ok_verdict, running, false, live_rate);
        CHECK(live_summary.find("\"distinct_fps\":60.000") != std::string::npos &&
              live_summary.find("\"presented_fps\":60.000") != std::string::npos &&
              live_summary.find("\"frame_rate_measured\":true") != std::string::npos &&
              live_summary.find("\"mostly_retained\":false") != std::string::npos,
              "a live title's summary reports both rates at ~60 fps");

        const std::string frozen_summary = manifest_summary_json(
            5, 5, SamplingStop::RequestSatisfied, tracker, ok_verdict, running, false, frozen_rate);
        CHECK(frozen_summary.find("\"presented_fps\":60.000") != std::string::npos,
              "a FROZEN title still publishes at 60 fps -- the number that must not stand alone");
        CHECK(frozen_summary.find("\"distinct_fps\":0.100") != std::string::npos &&
              frozen_summary.find("\"distinct_frames\":1") != std::string::npos &&
              frozen_summary.find("\"mostly_retained\":true") != std::string::npos,
              "...and the same summary says so, in three independent fields");

        const std::string unmeasured = manifest_summary_json(
            0, 5, SamplingStop::Timeout, tracker, ok_verdict, running, false,
            prosper::gpu::FrameRate{});
        CHECK(unmeasured.find("\"frame_rate_measured\":false") != std::string::npos &&
              unmeasured.find("\"distinct_fps\":0.000") != std::string::npos,
              "a run that published nothing reports NOT MEASURED rather than a confident 0 fps");

        CaptureObservation timed;
        timed.width = 1920; timed.height = 1080;
        timed.elapsed_seconds = 12.5;
        timed.published_frames = 750;
        timed.distinct_frames = 12;
        CaptureClassification classification;
        const std::string sample = manifest_sample_json(4, "s.png", timed, classification, "");
        CHECK(sample.find("\"published_frames\":750") != std::string::npos &&
              sample.find("\"distinct_frames\":12") != std::string::npos,
              "a sample line carries both counters, so any window of the run can be measured later");

        CaptureRunConfig overlay_config;
        overlay_config.fps_overlay = true;
        CHECK(manifest_run_json(overlay_config).find("\"fps_overlay\":true") != std::string::npos,
              "the run header records that the PNGs carry a burned-in annotation");
        CHECK(manifest_run_json(CaptureRunConfig{}).find("\"fps_overlay\":false") != std::string::npos,
              "...and that the default run's PNGs do not");
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
