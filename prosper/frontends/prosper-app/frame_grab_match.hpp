// F9's BMP and bundle are asynchronous artifacts. A shared stem is ownership, not frame identity.
#pragma once

#include "shared/present/producer_lineage.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace prosper::frontend {

enum class FrameGrabSource { None, GpuScanout, GpuCpuFallback, Cpu };
enum class FrameGrabNoBmpReason {
    None, TargetSkipped, CandidateOverflow, NeverPresented, UnresolvedAtShutdown
};

struct FrameGrabScreenshotEvidence {
    bool bmp_written = false;
    bool host_presented = false;
    FrameGrabSource source = FrameGrabSource::None;
    uint64_t source_seq = 0;       // leased selected-front flip token, never a sampled global count
    uint64_t target_source_flip = 0; // synchronous closing token of this owned bundle, if observed
    uint64_t publication_id = 0;  // GPU scanout handoff or CPU publication, not producer work
    ProducerSource producer;
    uint64_t armed_present = 0;
    uint64_t written_present = 0;  // timing only; must never select the bundle frame
    FrameGrabNoBmpReason no_bmp_reason = FrameGrabNoBmpReason::None;
};

struct FrameGrabGpuCandidate {
    FrameGrabScreenshotEvidence source;
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> pixels;
};

// A selected front may become visible before the guest flip thread announces the F9 window's
// closing token. Retain only successfully host-presented leases while that token is pending; once
// it arrives, the app can write the exact candidate even if the staging buffer has been reused.
// Overflow loses evidence, never guesses a replacement frame. Both limits are explicit because a
// 4K candidate is tens of MiB and a multi-frame bundle can hold the window open for 240 flips.
class FrameGrabCandidateRing {
public:
    explicit FrameGrabCandidateRing(size_t max_frames = 4,
                                    size_t max_bytes = 128u * 1024u * 1024u)
        : max_frames_(max_frames), max_bytes_(max_bytes) {}

    void note_host_presented(uint64_t source_flip) {
        highest_presented_flip_ = std::max(highest_presented_flip_, source_flip);
    }

    bool stage(FrameGrabScreenshotEvidence source, const uint8_t* rgba,
               uint32_t width, uint32_t height) {
        if (!source.host_presented || source.source != FrameGrabSource::GpuScanout ||
            !source.source_seq || !rgba || !width || !height) return false;
        note_host_presented(source.source_seq);
        const uint64_t pixels = uint64_t{width} * height;
        if (!max_frames_ || pixels > max_bytes_ / 4u) { overflowed_ = true; return false; }
        const size_t bytes = static_cast<size_t>(pixels * 4u);
        // Repeated representations of one flip cannot consume the bounded ring twice.
        for (const auto& candidate : candidates_)
            if (candidate.source.source_seq == source.source_seq) return true;
        while (!candidates_.empty() &&
               (candidates_.size() >= max_frames_ || held_bytes_ > max_bytes_ - bytes)) {
            held_bytes_ -= candidates_.front().pixels.size();
            candidates_.pop_front();
            overflowed_ = true;
        }
        FrameGrabGpuCandidate candidate;
        candidate.source = source;
        candidate.width = width;
        candidate.height = height;
        try { candidate.pixels.assign(rgba, rgba + bytes); }
        catch (const std::bad_alloc&) { overflowed_ = true; return false; }
        try { candidates_.push_back(std::move(candidate)); }
        catch (const std::bad_alloc&) { overflowed_ = true; return false; }
        held_bytes_ += bytes;
        return true;
    }

    std::optional<FrameGrabGpuCandidate> take_exact(uint64_t source_flip) {
        for (auto it = candidates_.begin(); it != candidates_.end(); ++it) {
            if (it->source.source_seq != source_flip) continue;
            FrameGrabGpuCandidate candidate = std::move(*it);
            held_bytes_ -= candidate.pixels.size();
            candidates_.erase(it);
            return candidate;
        }
        return std::nullopt;
    }
    uint64_t highest_presented_flip() const { return highest_presented_flip_; }
    bool overflowed() const { return overflowed_; }
    void clear() { candidates_.clear(); held_bytes_ = 0; highest_presented_flip_ = 0; overflowed_ = false; }

private:
    size_t max_frames_, max_bytes_, held_bytes_ = 0;
    uint64_t highest_presented_flip_ = 0;
    bool overflowed_ = false;
    std::deque<FrameGrabGpuCandidate> candidates_;
};

struct FrameGrabBundleEvidence {
    bool serialized = false;      // only after the owned writer closed and installed the bundle
    uint64_t opened_present = 0;
    std::vector<uint64_t> closed_presents;
    uint64_t opened_source_flip = 0;
    std::vector<uint64_t> closed_source_flips;
    std::vector<uint64_t> closed_submit_counts; // captured-submit prefix at each closed present
    std::vector<uint64_t> captured_submits; // exact membership, not first..last approximation
};

enum class FrameGrabMatch {
    Incomplete,
    NotPresented,
    UnknownSource,
    UnknownProducer,
    UnmatchedFrame,
    UnmatchedProducer,
    ProducerCaptured,
};

// The target is the capture window's exact closing guest flip, not a present count. Only a GPU
// scanout lease carrying that token proves the selected front matched. Earlier frames leave F9
// pending; later frames prove this host presenter skipped the target.
enum class FrameGrabTargetDecision { Wait, Capture, Missed };
inline FrameGrabTargetDecision frame_grab_target_decision(uint64_t target_source_flip,
                                                          uint64_t observed_source_flip) {
    if (!target_source_flip || !observed_source_flip ||
        observed_source_flip < target_source_flip) return FrameGrabTargetDecision::Wait;
    return observed_source_flip == target_source_flip ? FrameGrabTargetDecision::Capture
                                                      : FrameGrabTargetDecision::Missed;
}

inline bool frame_grab_needs_gpu_readback(bool screenshot_pending, bool owned_f9,
                                          bool closure_known, FrameGrabTargetDecision decision) {
    // The source image can be selected and host-presented before the guest flip thread closes the
    // F9 window. Staging while closure is unknown is what lets the bounded ring recover that race.
    return screenshot_pending &&
           (!owned_f9 || !closure_known || decision == FrameGrabTargetDecision::Capture);
}

enum class FrameGrabCandidateResolutionKind { Wait, Candidate, Missed };
struct FrameGrabCandidateResolution {
    FrameGrabCandidateResolutionKind kind = FrameGrabCandidateResolutionKind::Wait;
    std::optional<FrameGrabGpuCandidate> candidate;
    uint64_t newest_shown_flip = 0;
};

inline FrameGrabCandidateResolution frame_grab_resolve_candidate(
    FrameGrabCandidateRing& ring, uint64_t target_flip, uint64_t newest_shown_flip = 0) {
    if (!target_flip) return {};
    if (auto candidate = ring.take_exact(target_flip))
        return {FrameGrabCandidateResolutionKind::Candidate, std::move(candidate), 0};
    const uint64_t newest = std::max(newest_shown_flip, ring.highest_presented_flip());
    if (newest > target_flip || (ring.overflowed() && newest >= target_flip))
        return {FrameGrabCandidateResolutionKind::Missed, std::nullopt, newest};
    return {};
}

inline FrameGrabMatch classify_frame_grab(const FrameGrabScreenshotEvidence& shot,
                                          const FrameGrabBundleEvidence& bundle) {
    if (!shot.bmp_written || !bundle.serialized) return FrameGrabMatch::Incomplete;
    if (!shot.host_presented) return FrameGrabMatch::NotPresented;
    if (shot.source != FrameGrabSource::GpuScanout) return FrameGrabMatch::UnknownSource;
    if (!shot.source_seq || !shot.target_source_flip || !bundle.opened_source_flip ||
        bundle.closed_source_flips.empty() ||
        shot.source_seq != shot.target_source_flip ||
        shot.target_source_flip != bundle.closed_source_flips.back() ||
        bundle.closed_source_flips.size() != bundle.closed_presents.size() ||
        bundle.closed_submit_counts.size() != bundle.closed_source_flips.size())
        return FrameGrabMatch::UnmatchedFrame;
    // An interleaved later present must not relabel an older selected front as this bundle's
    // closing picture. Fail closed if either boundary clock or captured-submit prefix reverses.
    uint64_t previous_flip = bundle.opened_source_flip;
    uint64_t previous_present = bundle.opened_present;
    uint64_t previous_submit_count = 0;
    for (size_t i = 0; i < bundle.closed_source_flips.size(); ++i) {
        if (bundle.closed_source_flips[i] <= previous_flip ||
            bundle.closed_presents[i] <= previous_present ||
            bundle.closed_submit_counts[i] < previous_submit_count ||
            bundle.closed_submit_counts[i] > bundle.captured_submits.size())
            return FrameGrabMatch::UnmatchedFrame;
        previous_flip = bundle.closed_source_flips[i];
        previous_present = bundle.closed_presents[i];
        previous_submit_count = bundle.closed_submit_counts[i];
    }
    const uint64_t captured_through = bundle.closed_submit_counts.back();
    if (!shot.producer.known() || !shot.producer.completed.source_submit)
        return FrameGrabMatch::UnknownProducer;
    if (std::find(bundle.captured_submits.begin(),
                  bundle.captured_submits.begin() + static_cast<ptrdiff_t>(captured_through),
                  shot.producer.completed.source_submit) ==
        bundle.captured_submits.begin() + static_cast<ptrdiff_t>(captured_through))
        return FrameGrabMatch::UnmatchedProducer;
    // This proves the completed source image's originating guest submit was serialized and its
    // exact selected-front flip was a captured frame boundary. Replay pixels can still differ if
    // a dependency or seed is missing; it is not a pixel-equality assertion.
    return FrameGrabMatch::ProducerCaptured;
}

inline const char* frame_grab_match_name(FrameGrabMatch match) {
    switch (match) {
    case FrameGrabMatch::Incomplete: return "incomplete";
    case FrameGrabMatch::NotPresented: return "not_presented";
    case FrameGrabMatch::UnknownSource: return "unknown_source";
    case FrameGrabMatch::UnknownProducer: return "unknown_producer";
    case FrameGrabMatch::UnmatchedFrame: return "unmatched_frame";
    case FrameGrabMatch::UnmatchedProducer: return "unmatched_producer";
    case FrameGrabMatch::ProducerCaptured: return "producer_submit_captured";
    }
    return "unknown";
}

inline std::string frame_grab_screenshot_event(const FrameGrabScreenshotEvidence& shot) {
    const char* source = shot.source == FrameGrabSource::GpuScanout ? "gpu_scanout" :
                         shot.source == FrameGrabSource::GpuCpuFallback ? "gpu_cpu_fallback" :
                         shot.source == FrameGrabSource::Cpu ? "cpu" : "none";
    const char* reason = shot.no_bmp_reason == FrameGrabNoBmpReason::TargetSkipped ? "target_skipped" :
                         shot.no_bmp_reason == FrameGrabNoBmpReason::CandidateOverflow ? "candidate_overflow" :
                         shot.no_bmp_reason == FrameGrabNoBmpReason::NeverPresented ? "never_presented" :
                         shot.no_bmp_reason == FrameGrabNoBmpReason::UnresolvedAtShutdown
                             ? "unresolved_at_shutdown" : "none";
    return "{\"v\":1,\"event\":\"screenshot\",\"bmp_written\":" +
        std::string(shot.bmp_written ? "true" : "false") +
        ",\"host_presented\":" + (shot.host_presented ? "true" : "false") +
        ",\"no_bmp_reason\":\"" + reason + "\"" +
        ",\"source\":\"" + source + "\",\"source_seq\":" + std::to_string(shot.source_seq) +
        ",\"target_source_flip\":" + std::to_string(shot.target_source_flip) +
        ",\"publication_id\":" + std::to_string(shot.publication_id) +
        ",\"image_registration\":" + std::to_string(shot.producer.image_registration) +
        ",\"producer_registration\":" + std::to_string(shot.producer.completed.registration) +
        ",\"producer_work\":" + std::to_string(shot.producer.completed.work) +
        ",\"producer_submit\":" + std::to_string(shot.producer.completed.source_submit) +
        ",\"armed_present\":" + std::to_string(shot.armed_present) +
        ",\"written_present\":" + std::to_string(shot.written_present) + "}";
}

inline void frame_grab_append_u64_array(std::string& line, const std::vector<uint64_t>& values) {
    line += '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) line += ',';
        line += std::to_string(values[i]);
    }
    line += ']';
}

inline std::string frame_grab_bundle_event(const FrameGrabBundleEvidence& bundle) {
    std::string line = "{\"v\":1,\"event\":\"bundle\",\"serialized\":";
    line += bundle.serialized ? "true" : "false";
    line += ",\"opened_present\":" + std::to_string(bundle.opened_present) +
            ",\"closed_presents\":";
    frame_grab_append_u64_array(line, bundle.closed_presents);
    line += ",\"opened_source_flip\":" + std::to_string(bundle.opened_source_flip) +
            ",\"closed_source_flips\":";
    frame_grab_append_u64_array(line, bundle.closed_source_flips);
    line += ",\"closed_submit_counts\":";
    frame_grab_append_u64_array(line, bundle.closed_submit_counts);
    line += ",\"captured_submits\":";
    frame_grab_append_u64_array(line, bundle.captured_submits);
    line += '}';
    return line;
}

inline std::string frame_grab_join_event(FrameGrabMatch verdict) {
    return std::string("{\"v\":1,\"event\":\"join\",\"verdict\":\"") +
           frame_grab_match_name(verdict) +
           "\",\"dependency_closure\":\"unknown\",\"replay_pixel_match\":\"unknown\"}";
}

// The namer created this sidecar exclusively at arm time. Each successfully closed JSONL event
// reports one observed fact; an interrupted final line or absent join is not a completed-pair claim.
inline bool append_frame_grab_event(const std::string& path, const std::string& line) {
    if (path.empty()) return false;
    FILE* file = std::fopen(path.c_str(), "r+b");
    if (!file) return false;
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        return false;
    }
    const bool wrote = std::fwrite(line.data(), 1, line.size(), file) == line.size() &&
                       std::fputc('\n', file) != EOF;
    return std::fclose(file) == 0 && wrote;
}

} // namespace prosper::frontend
