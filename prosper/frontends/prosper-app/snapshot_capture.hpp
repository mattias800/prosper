// Optional wall-clock evidence on the normal app's CPU presentation path. A retained frame
// remains sampleable when the guest stops publishing; fresh publication is never the sampler clock.
#pragma once

#include "gpu/present/videoout_present.hpp"
#include "shared/capture/capture_manifest.hpp"

#include <chrono>
#include <fstream>
#include <string>

namespace prosper::frontend {

struct SnapshotCaptureRequest {
    bool enabled = false, require_composited = false;
    double seconds = 0, timeout = 0;
    int count = 0;
    std::string output, manifest;
    bool argument(const std::string& option, int& index, int argc, char** argv, std::string& error);
    bool valid(std::string& error) const;
};

class SnapshotCaptureClock {
public:
    SnapshotCaptureClock(double interval, double timeout, int count)
        : interval_(interval), timeout_(timeout), count_(count) {}
    bool due(double elapsed) const { return saved_ < count_ && elapsed - last_ >= interval_; }
    bool timed_out(double elapsed) const { return elapsed >= timeout_ && !complete(); }
    bool complete() const { return saved_ == count_; }
    void sampled(double elapsed) {
        last_ = elapsed;
        ++saved_;
    }
    int saved() const { return saved_; }

private:
    double interval_, timeout_, last_ = 0;
    int count_, saved_ = 0;
};

// Preserve one immutable, completed frame with its source identity. Only a successful window
// presentation may replace it; an unpublished/failed attempt cannot become a capture of zeros.
class PresentedSnapshotFrame {
public:
    bool retain(const gpu::PresentFrameLease& frame, bool presented);
    const gpu::PresentFrameLease& get() const { return frame_; }
    bool available() const { return frame_.rgba != nullptr; }

private:
    gpu::PresentFrameLease frame_;
};

class SnapshotCapture {
public:
    SnapshotCaptureRequest request;
    bool enabled() const { return request.enabled; }
    bool start(const std::string& title, std::string& error);
    void retain(const gpu::PresentFrameLease& frame, bool presented);
    // True asks the existing app loop to shut down. The summary is flushed by finish before _Exit.
    bool poll(bool visible);
    bool poll_elapsed(double elapsed, bool visible);
    int finish();

private:
    bool line(const std::string& text);
    bool sample(double elapsed);
    bool started_ = false, failed_ = false, finished_ = false;
    double last_elapsed_ = 0;
    int verdict_ = 0;
    std::chrono::steady_clock::time_point start_;
    SnapshotCaptureClock clock_{1, 1, 1};
    PresentedSnapshotFrame frame_;
    std::ofstream manifest_;
    screenshot::CaptureTracker tracker_;
    screenshot::CaptureConditions conditions_;
    std::string route_;
};
}   // namespace prosper::frontend
