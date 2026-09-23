#include "frame_grab_match.hpp"
#include "frame_grab_naming.hpp"
#include "present_policy.hpp"
#include "fixtures/test_scratch.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

using namespace prosper::frontend;

static int failures = 0;
#define CHECK(value, message) do { if (!(value)) { std::printf("FAIL: %s\n", message); ++failures; } } while (0)

// This narrow source gate complements the behavior tests below. The app owns the actual
// successful-host-present decision; removing that call site otherwise leaves every isolated
// timeline/ring test green while reinstating #3828. Comments cannot satisfy the gate.
static std::string active_app_source() {
    std::ifstream file(PROSPER_APP_MAIN_SOURCE);
    std::string line, active;
    while (std::getline(file, line)) {
        if (const size_t comment = line.find("//"); comment != std::string::npos)
            line.resize(comment);
        active += line + '\n';
    }
    return active;
}

static size_t matching_brace(std::string_view source, size_t open) {
    if (open == std::string::npos || source[open] != '{') return std::string::npos;
    size_t depth = 0;
    char quote = 0;
    bool escaped = false;
    for (size_t i = open; i < source.size(); ++i) {
        const char c = source[i];
        if (quote) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == quote) quote = 0;
        } else if (c == '"' || c == '\'') {
            quote = c;
        } else if (c == '{') {
            ++depth;
        } else if (c == '}' && --depth == 0) {
            return i;
        }
    }
    return std::string::npos;
}

static bool gpu_close_wired(const std::string& source) {
    const size_t begin = source.find("\n        if (vk.gpu_present) {");
    if (begin == std::string::npos) return false;
    const size_t end = source.find("} else if (gpu::present_frame_seq() != lastFrameSeq)", begin);
    if (end == std::string::npos) return false;
    const std::string_view block(source.data() + begin, end - begin);
    const size_t readback = block.find("frame_grab_needs_gpu_readback(");
    const size_t staged = block.find("stagedGrabCandidates.stage(");
    const size_t success = block.find("if (attempt == PresentAttempt::presented) {", staged);
    if (readback == std::string::npos || staged == std::string::npos ||
        readback >= staged || success == std::string::npos) return false;
    const size_t body_open = block.find('{', success);
    const size_t body_close = matching_brace(block, body_open);
    if (body_close == std::string::npos) return false;
    const std::string_view success_body = block.substr(body_open, body_close - body_open);
    const size_t first = success_body.find("drainStagedGrabCandidate(gf.frame_seq);");
    const size_t close = success_body.find("interactive_grab_on_host_presented_source(", first);
    const size_t second = success_body.find("drainStagedGrabCandidate(gf.frame_seq);", close);
    return first != std::string::npos && close != std::string::npos &&
           second != std::string::npos;
}

static bool fallback_and_expiry_wired(const std::string& source) {
    const size_t begin = source.find("} else if (gpu::present_frame_seq() != lastFrameSeq)");
    const size_t end = source.find("} else {\n        bool newFrame", begin);
    const size_t loop = source.find("while (running && !prosper_stop_requested()) {");
    const size_t expiry = source.find("auto expirePendingGrabProducer = [&] {");
    if (begin == std::string::npos || end == std::string::npos ||
        loop == std::string::npos || expiry == std::string::npos) return false;
    const std::string_view fallback(source.data() + begin, end - begin);
    const size_t convert = fallback.find("interactive_grab_on_cpu_fallback(");
    const size_t bmp = fallback.find("flushGrabScreenshot(cf.rgba->data()");
    const std::string_view loop_entry(source.data() + loop,
                                      std::min<size_t>(source.size() - loop, 250));
    const size_t expiry_open = source.find('{', expiry);
    const size_t expiry_close = matching_brace(source, expiry_open);
    if (expiry_close == std::string::npos) return false;
    const std::string_view expiry_body(source.data() + expiry_open,
                                        expiry_close - expiry_open);
    const size_t owned_poll = expiry_body.find("for (const auto& reservation : grabReservedBundles)");
    const size_t expire_call = expiry_body.find("interactive_grab_expire_producer_wait(");
    const size_t early_return = expiry_body.find("return;");
    return convert != std::string::npos && bmp != std::string::npos && convert < bmp &&
           loop_entry.find("expirePendingGrabProducer();") != std::string::npos &&
           owned_poll != std::string::npos && expire_call > owned_poll &&
           expire_call != std::string::npos &&
           (early_return == std::string::npos || early_return > expire_call);
}

int main() {
    const std::string app_source = active_app_source();
    CHECK(!app_source.empty() && gpu_close_wired(app_source) &&
          fallback_and_expiry_wired(app_source),
          "#3828: live app paths invoke exact GPU closure, CPU downgrade, and expiry polling");
    std::string omitted_gpu_close = app_source;
    if (const size_t call = omitted_gpu_close.find("interactive_grab_on_host_presented_source(",
            omitted_gpu_close.find("\n        if (vk.gpu_present) {")); call != std::string::npos)
        omitted_gpu_close.replace(call, sizeof("interactive_grab_on_host_presented_source") - 1,
                                  "omitted_host_present_closure");
    CHECK(!gpu_close_wired(omitted_gpu_close),
          "#3828: removing the live GPU closure call makes the wiring gate fail");
    std::string outside_success = app_source;
    const size_t gpu_block = outside_success.find("\n        if (vk.gpu_present) {");
    const size_t guarded_success = outside_success.find(
        "if (attempt == PresentAttempt::presented) {", gpu_block);
    const size_t guarded_call = outside_success.find(
        "(void)prosper::gpu::interactive_grab_on_host_presented_source(", guarded_success);
    if (guarded_call != std::string::npos) {
        const size_t statement_end = outside_success.find(';', guarded_call);
        if (statement_end != std::string::npos) {
            const std::string statement = outside_success.substr(
                guarded_call, statement_end - guarded_call + 1);
            outside_success.erase(guarded_call, statement.size());
            const size_t success_open = outside_success.find('{', guarded_success);
            const size_t success_close = matching_brace(outside_success, success_open);
            if (success_close != std::string::npos)
                outside_success.insert(success_close + 1, "\n" + statement + "\n");
        }
    }
    CHECK(!gpu_close_wired(outside_success),
          "#3828: moving producer closure after the successful-present guard fails the gate");
    std::string omitted_readback = app_source;
    if (const size_t call = omitted_readback.find("frame_grab_needs_gpu_readback(",
            omitted_readback.find("\n        if (vk.gpu_present) {")); call != std::string::npos)
        omitted_readback.replace(call, sizeof("frame_grab_needs_gpu_readback") - 1,
                                 "omitted_gpu_readback_policy");
    CHECK(!gpu_close_wired(omitted_readback),
          "#3828: removing pre-closure readback makes the wiring gate fail");
    std::string omitted_fallback = app_source;
    if (const size_t call = omitted_fallback.find("interactive_grab_on_cpu_fallback(",
            omitted_fallback.find("} else if (gpu::present_frame_seq() != lastFrameSeq)"));
            call != std::string::npos)
        omitted_fallback.replace(call, sizeof("interactive_grab_on_cpu_fallback") - 1,
                                 "omitted_cpu_fallback");
    CHECK(!fallback_and_expiry_wired(omitted_fallback),
          "#3828: removing the live CPU fallback transition makes the wiring gate fail");
    std::string bmp_gated_expiry = app_source;
    if (const size_t expiry = bmp_gated_expiry.find("auto expirePendingGrabProducer = [&] {");
            expiry != std::string::npos) {
        const size_t body = bmp_gated_expiry.find('{', expiry);
        if (body != std::string::npos)
            bmp_gated_expiry.insert(body + 1,
                                    "\nif (pendingGrabScreenshot.empty()) return;\n");
    }
    CHECK(!fallback_and_expiry_wired(bmp_gated_expiry),
          "#3828: gating the owned bundle expiry on a cleared BMP path fails the gate");
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
          delayed_closure.find_exact(42) &&
          delayed_closure.find_exact(42)->source.producer.completed.source_submit == 88 &&
          !delayed_closure.find_exact(41) &&
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
