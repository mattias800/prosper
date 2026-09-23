// F9's BMP and bundle are asynchronous artifacts. A shared stem is ownership, not frame identity.
#pragma once

#include "shared/present/producer_lineage.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace prosper::frontend {

enum class FrameGrabSource { GpuScanout, GpuCpuFallback, Cpu };

struct FrameGrabScreenshotEvidence {
    bool bmp_written = false;
    bool host_presented = false;
    FrameGrabSource source = FrameGrabSource::Cpu;
    uint64_t source_seq = 0;       // leased selected-front flip token, never a sampled global count
    uint64_t target_source_flip = 0; // synchronous closing token of this owned bundle, if observed
    uint64_t publication_id = 0;  // GPU scanout handoff or CPU publication, not producer work
    ProducerSource producer;
    uint64_t armed_present = 0;
    uint64_t written_present = 0;  // timing only; must never select the bundle frame
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
                         shot.source == FrameGrabSource::GpuCpuFallback ? "gpu_cpu_fallback" : "cpu";
    return "{\"v\":1,\"event\":\"screenshot\",\"bmp_written\":" +
        std::string(shot.bmp_written ? "true" : "false") +
        ",\"host_presented\":" + (shot.host_presented ? "true" : "false") +
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
