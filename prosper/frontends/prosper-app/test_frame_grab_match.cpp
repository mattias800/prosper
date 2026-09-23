#include "frame_grab_match.hpp"
#include "frame_grab_naming.hpp"
#include "present_policy.hpp"
#include "fixtures/test_scratch.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

using namespace prosper::frontend;

static int failures = 0;
#define CHECK(value, message) do { if (!(value)) { std::printf("FAIL: %s\n", message); ++failures; } } while (0)

int main() {
    const uint8_t pixel[] = {7, 13, 19, 255};
    FrameGrabScreenshotEvidence candidate_source;
    candidate_source.host_presented = true;
    candidate_source.source = FrameGrabSource::GpuScanout;
    candidate_source.source_seq = 42;
    candidate_source.publication_id = 900;
    candidate_source.producer = {99, {7, 11, 88}};
    FrameGrabCandidateRing delayed_closure;
    const bool readback_before_closure = frame_grab_needs_gpu_readback(
        true, true, false, FrameGrabTargetDecision::Wait);
    bool staged_before_closure = false;
    dispatch_presented_capture(PresentAttempt::presented, readback_before_closure, [&] {
        staged_before_closure = delayed_closure.stage(candidate_source, pixel, 1, 1);
    });
    CHECK(staged_before_closure &&
          frame_grab_resolve_candidate(delayed_closure, 0, 42).kind ==
              FrameGrabCandidateResolutionKind::Wait &&
          frame_grab_resolve_candidate(delayed_closure, 41, 42).kind ==
              FrameGrabCandidateResolutionKind::Missed,
          "the app readback policy retains a host-presented candidate before F9 closure");
    auto recovered = frame_grab_resolve_candidate(delayed_closure, 42, 42);
    CHECK(recovered.kind == FrameGrabCandidateResolutionKind::Candidate &&
          recovered.candidate && recovered.candidate->source.source_seq == 42 &&
          recovered.candidate->source.publication_id == 900 &&
          recovered.candidate->source.producer.completed.source_submit == 88 &&
          recovered.candidate->pixels == std::vector<uint8_t>(std::begin(pixel), std::end(pixel)),
          "closure after host present drains exact pixels and producer before stage reuse");
    candidate_source.host_presented = false;
    CHECK(!delayed_closure.stage(candidate_source, pixel, 1, 1) &&
          !delayed_closure.take_exact(42).has_value(),
          "a failed host present never becomes a retained screenshot candidate");
    candidate_source.host_presented = true;
    FrameGrabCandidateRing host_gate;
    auto commit_host_candidate = [&] { host_gate.stage(candidate_source, pixel, 1, 1); };
    CHECK(!dispatch_presented_capture(PresentAttempt::skipped, readback_before_closure,
                                      commit_host_candidate) &&
          frame_grab_resolve_candidate(host_gate, 0, 42).kind ==
              FrameGrabCandidateResolutionKind::Wait &&
          !host_gate.take_exact(42).has_value() &&
          !dispatch_presented_capture(PresentAttempt::out_of_date, readback_before_closure,
                                      commit_host_candidate) &&
          !host_gate.take_exact(42).has_value() &&
          dispatch_presented_capture(PresentAttempt::presented, readback_before_closure,
                                     commit_host_candidate) &&
          frame_grab_resolve_candidate(host_gate, 42, 42).kind ==
              FrameGrabCandidateResolutionKind::Candidate,
          "only successful host presentation admits a candidate for late F9 closure");
    FrameGrabCandidateRing readback_failure;
    readback_failure.note_host_presented(42);
    CHECK(frame_grab_resolve_candidate(readback_failure, 42).kind ==
              FrameGrabCandidateResolutionKind::Wait &&
          readback_failure.highest_presented_flip() == 42,
          "a shown target without pixels remains unresolved, not never presented");
    readback_failure.note_host_presented(43);
    CHECK(frame_grab_resolve_candidate(readback_failure, 42).kind ==
              FrameGrabCandidateResolutionKind::Missed,
          "a later shown source makes the missing target a definite miss");
    FrameGrabCandidateRing count_limited(2, 16), byte_limited(4, 8);
    for (uint64_t flip = 50; flip <= 52; ++flip) {
        candidate_source.source_seq = flip;
        CHECK(count_limited.stage(candidate_source, pixel, 1, 1) &&
              byte_limited.stage(candidate_source, pixel, 1, 1),
              "bounded candidate stores accept each valid host-presented source");
    }
    CHECK(count_limited.overflowed() && byte_limited.overflowed() &&
          !count_limited.take_exact(50).has_value() &&
          !byte_limited.take_exact(50).has_value() &&
          count_limited.take_exact(52).has_value() &&
          byte_limited.take_exact(52).has_value(),
          "frame-count and byte caps evict older evidence without substituting a newer source");
    CHECK(frame_grab_resolve_candidate(count_limited, 50, 52).kind ==
              FrameGrabCandidateResolutionKind::Missed &&
          frame_grab_resolve_candidate(byte_limited, 52, 52).kind ==
              FrameGrabCandidateResolutionKind::Missed,
          "a later flip or evicted exact target resolves incomplete rather than substituting pixels");
    FrameGrabCandidateRing oversized(4, 3);
    CHECK(!oversized.stage(candidate_source, pixel, 1, 1) && oversized.overflowed() &&
          !oversized.take_exact(52).has_value(),
          "a single image over the byte cap fails closed without retaining a partial pixel array");
    CHECK(frame_grab_target_decision(0, 42) == FrameGrabTargetDecision::Wait &&
          frame_grab_target_decision(42, 0) == FrameGrabTargetDecision::Wait,
          "an unclosed bundle or source without a selected-front token cannot choose pixels");
    CHECK(frame_grab_target_decision(42, 41) == FrameGrabTargetDecision::Wait &&
          frame_grab_target_decision(42, 42) == FrameGrabTargetDecision::Capture &&
          frame_grab_target_decision(42, 43) == FrameGrabTargetDecision::Missed,
          "opening frame waits, exact closing frame captures, later frame reports a skipped target");
    CHECK(frame_grab_needs_gpu_readback(true, true, false, FrameGrabTargetDecision::Wait) &&
          frame_grab_needs_gpu_readback(true, true, true, FrameGrabTargetDecision::Capture) &&
          !frame_grab_needs_gpu_readback(true, true, true, FrameGrabTargetDecision::Wait) &&
          !frame_grab_needs_gpu_readback(true, true, true, FrameGrabTargetDecision::Missed) &&
          !frame_grab_needs_gpu_readback(false, true, false, FrameGrabTargetDecision::Wait),
          "F9 stages before closure to survive selected-front publication racing ahead of the token");
    FrameGrabScreenshotEvidence shot;
    shot.bmp_written = true;
    shot.host_presented = true;
    shot.source = FrameGrabSource::GpuScanout;
    shot.source_seq = 42;
    shot.target_source_flip = 42;
    shot.publication_id = 700;
    // The source image is a representation-only copy; its registration differs from the work's.
    shot.producer = {99, {7, 11, 88}};
    shot.armed_present = 41;
    shot.written_present = 42;
    FrameGrabBundleEvidence bundle;
    bundle.serialized = true;
    // The two flip paths may interleave. Present count is a different clock from the exact
    // selected-front flip token, so these numbers intentionally disagree.
    bundle.opened_present = 701;
    bundle.closed_presents = {702};
    bundle.opened_source_flip = 41;
    bundle.closed_source_flips = {42};
    bundle.closed_submit_counts = {3};
    bundle.captured_submits = {87, 88, 89};
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::ProducerCaptured,
          "exact completed producer and frame boundary join");
    shot.target_source_flip = 0;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnmatchedFrame,
          "a coincidentally matching source without an owned closure token cannot certify a pair");
    shot.target_source_flip = 42;

    // Deliberately delay readback while guest presents race ahead. The WRITE clock must never be
    // substituted for the leased screenshot source, including in the serialized event.
    shot.written_present = 900;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::ProducerCaptured &&
          frame_grab_screenshot_event(shot).find("\"source_seq\":42") != std::string::npos &&
          frame_grab_screenshot_event(shot).find("\"target_source_flip\":42") != std::string::npos &&
          frame_grab_screenshot_event(shot).find("\"written_present\":900") != std::string::npos,
          "delayed readback does not relabel the source frame");
    shot.source_seq = 900;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnmatchedFrame,
          "the delayed write count is not accepted as a bundle frame boundary");
    shot.source_seq = 42;
    bundle.closed_presents = {702};
    bundle.closed_source_flips = {43};
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnmatchedFrame,
          "a failed selection retains the old front token, never joining the new flip by count");
    bundle.closed_source_flips = {42};
    bundle.opened_source_flip = 43;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnmatchedFrame,
          "reversed cross-thread flip order cannot certify an older source after a newer open");
    bundle.opened_source_flip = 42;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnmatchedFrame,
          "a source at the opening boundary was not produced inside the bundle window");
    bundle.opened_source_flip = 41;

    bundle.closed_presents = {702, 703};
    bundle.closed_source_flips = {42, 43};
    bundle.closed_submit_counts = {2, 3}; // submit 88 is present even at the earlier boundary
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnmatchedFrame,
          "an earlier closed frame with the same producer cannot replace the final target");
    shot.source_seq = 43;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnmatchedFrame,
          "a final-frame BMP needs its own matching owned closure target");
    shot.target_source_flip = 43;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::ProducerCaptured,
          "a BMP from the final target may join its captured producer");
    bundle.closed_source_flips = {43, 42};
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnmatchedFrame,
          "reversed selected-front boundaries cannot assert the last frame");
    shot.source_seq = 42;
    shot.target_source_flip = 42;
    bundle.closed_presents = {702};
    bundle.closed_source_flips = {42};
    bundle.closed_submit_counts.clear();
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnmatchedFrame,
          "a missing per-frame submit prefix cannot assert boundary membership");
    bundle.closed_submit_counts = {3};

    bundle.captured_submits = {87, 89}; // 88 is inside the range, but was not captured
    bundle.closed_submit_counts = {2};
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnmatchedProducer,
          "submit membership is exact, never inferred from first..last");
    bundle.captured_submits = {87, 88, 89};
    bundle.closed_submit_counts = {3};
    shot.producer.completed.source_submit = 0;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnknownProducer,
          "a known work token with no architectural origin cannot certify a join");
    shot.producer.completed.source_submit = 88;
    shot.producer.completed.work = 0;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnknownProducer,
          "an absent completed work token stays unknown");
    shot.producer.completed.work = 11;
    shot.source = FrameGrabSource::GpuCpuFallback;
    shot.publication_id = 901;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::UnknownSource,
          "CPU fallback has no exact GPU producer lineage");
    CHECK(frame_grab_screenshot_event(shot).find("\"source\":\"gpu_cpu_fallback\"") !=
              std::string::npos &&
          frame_grab_screenshot_event(shot).find("\"publication_id\":901") !=
              std::string::npos,
          "the usable CPU fallback BMP records its own publication without claiming a GPU join");
    shot.source = FrameGrabSource::GpuScanout;
    shot.host_presented = false;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::NotPresented,
          "a readback from a failed host-present attempt is not a shown frame");
    shot.host_presented = true;
    bundle.serialized = false;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::Incomplete,
          "an interrupted or failed bundle cannot claim a producer match");
    bundle.serialized = true;
    shot.bmp_written = false;
    CHECK(classify_frame_grab(shot, bundle) == FrameGrabMatch::Incomplete,
          "a failed BMP write cannot claim a pair");
    shot.bmp_written = true;
    FrameGrabScreenshotEvidence unpresented;
    unpresented.armed_present = 41;
    unpresented.target_source_flip = 42;
    unpresented.no_bmp_reason = FrameGrabNoBmpReason::NeverPresented;
    CHECK(classify_frame_grab(unpresented, bundle) == FrameGrabMatch::Incomplete &&
          frame_grab_screenshot_event(unpresented).find("\"source\":\"none\"") !=
              std::string::npos &&
          frame_grab_screenshot_event(unpresented).find("\"bmp_written\":false") !=
              std::string::npos &&
          frame_grab_screenshot_event(unpresented).find("\"no_bmp_reason\":\"never_presented\"") !=
              std::string::npos,
          "shutdown's empty BMP reservation reports no source and no completed pair");
    unpresented.no_bmp_reason = FrameGrabNoBmpReason::CandidateOverflow;
    CHECK(frame_grab_screenshot_event(unpresented).find(
              "\"no_bmp_reason\":\"candidate_overflow\"") != std::string::npos,
          "an evicted target reports the storage cap instead of claiming it was never shown");

    const auto dir = prosper_test::test_scratch_dir() / "frame_grab_match_test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    const FrameGrabPaths paths = reserve_frame_grab(
        dir.string(), "PPSA00000",
        std::chrono::system_clock::time_point(std::chrono::milliseconds(1000)));
    CHECK(paths.ok && std::filesystem::file_size(paths.manifest) == 0,
          "F9 owns a separate empty manifest reservation");
    const std::string shot_line = frame_grab_screenshot_event(shot);
    CHECK(append_frame_grab_event(paths.manifest, shot_line),
          "the screenshot event closes independently of the bundle writer");
    {
        std::ifstream file(paths.manifest);
        const std::string content((std::istreambuf_iterator<char>(file)), {});
        CHECK(content.find("\"event\":\"screenshot\"") != std::string::npos &&
              content.find("\"event\":\"join\"") == std::string::npos,
              "an interrupted pair preserves source identity without claiming a join");
    }
    CHECK(append_frame_grab_event(paths.manifest, frame_grab_bundle_event(bundle)) &&
          append_frame_grab_event(paths.manifest,
                                  frame_grab_join_event(classify_frame_grab(shot, bundle))),
          "bundle completion and joined verdict append only after the screenshot event");
    {
        std::ifstream file(paths.manifest);
        const std::string content((std::istreambuf_iterator<char>(file)), {});
        CHECK(content.find("\"verdict\":\"producer_submit_captured\"") != std::string::npos &&
              content.find("\"dependency_closure\":\"unknown\"") != std::string::npos &&
              content.find("\"replay_pixel_match\":\"unknown\"") != std::string::npos,
              "the exact submit membership is never promoted to replay-pixel proof");
    }
    CHECK(!append_frame_grab_event((dir / "not-reserved.f9.jsonl").string(), shot_line),
          "a missing sidecar cannot be recreated and mistaken for the owned reservation");
    std::filesystem::remove_all(dir, ec);
    return failures ? 1 : 0;
}
