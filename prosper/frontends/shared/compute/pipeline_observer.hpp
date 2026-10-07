// pipeline_observer.hpp -- ADR 0009 Stage 1, step 2: what would pipelining have done? (observe only)
//
// Nothing here changes timing or results. The live compute backend reports each dispatch's guest
// read and write ranges, and the executor reports the other ordered operations (graphics spans,
// DMA copies) that sit between dispatches. This model feeds them to the RetirementQueue contract
// (PERF-P8) and counts, per dispatch, whether the queue would have let it be admitted without first
// retiring (= waiting for) anything older.
//
// Two bounds, because the executor can say WHEN a graphics span or DMA ran but not WHAT it read:
//   * ranges-only  -- optimistic: only dispatches' own ranges can force a retirement; a graphics
//                     span in between is assumed not to read any pending result.
//   * barriers     -- pessimistic: every graphics span and DMA copy is a full retirement point.
// The truth lies between them. Each is run at two depths (PERF-P6): the bounded one a real
// implementation would use, and a deep one that shows the ceiling when depth is not the limit.
//
// "free" means admitted with nothing forced to retire; "overlapped" means it was admitted while an
// older dispatch was still pending, i.e. the GPU would really have had two in flight. The overlapped
// wait fraction is the well-defined one across both models: the share of today's measured dispatch
// time that belongs to dispatches which would have run concurrently with an older one. The wait-weighted fraction uses each
// dispatch's measured submit-to-fence time, so it answers "how much of the time we currently spend
// waiting could have overlapped", not merely "how many dispatches".
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "shared/compute/retirement_queue.hpp"

namespace prosper::frontend {

// One guest range a dispatch binds, and how it uses it.
struct BindingRange {
    uint64_t addr = 0;
    uint64_t bytes = 0;
    bool read = false;
    bool write = false;
};

// A dispatch's ranges: every read binding contributes a read, every written one a write. A binding
// that is both (a writable buffer, which may read its previous contents) contributes both.
inline PendingOperation make_pending_operation(uint64_t id, const std::vector<BindingRange>& bindings) {
    PendingOperation op;
    op.id = id;
    for (const BindingRange& b : bindings) {
        if (!b.bytes) continue;
        if (b.read) op.reads.push_back({b.addr, b.bytes});
        if (b.write) op.writes.push_back({b.addr, b.bytes});
    }
    return op;
}

// What one dispatch cost on the CPU and the GPU today, measured by the backend: the CPU work before it
// is submitted, the span from submit to its completed fence wait, and the CPU work after it completes.
struct DispatchTimes {
    double setup_ms = 0.0;
    double gpu_ms = 0.0;         // submit-to-fence span: an upper bound on the time the GPU needs
    double writeback_ms = 0.0;
};

struct PipelineModelStats {
    uint64_t dispatches = 0;
    uint64_t free_admits = 0;            // admitted without retiring anything
    uint64_t forced_admits = 0;          // had to retire something first
    uint64_t overlapped_admits = 0;      // admitted while an older dispatch was still pending (real overlap)
    uint64_t conflict_retires = 0;       // operations retired because of a RAW/WAW/WAR conflict
    uint64_t depth_retires = 0;          // additional operations retired only to respect the depth bound
    uint64_t barrier_retires = 0;        // operations retired at a graphics span / DMA / submit end
    uint64_t max_pending = 0;
    uint64_t other_operations = 0;       // graphics spans / DMA copies seen between dispatches
    double wait_ms_total = 0.0;
    double wait_ms_free = 0.0;           // wait time of dispatches that were admitted free
    double wait_ms_overlapped = 0.0;     // wait time of dispatches admitted while an older one was pending
    double sequential_ms = 0.0;          // today: every dispatch's setup + gpu + writeback, one after another
    double pipelined_ms = 0.0;           // the same work on the CPU/GPU timeline this model allows

    double free_fraction() const { return dispatches ? double(free_admits) / double(dispatches) : 0.0; }
    double overlapped_fraction() const { return dispatches ? double(overlapped_admits) / double(dispatches) : 0.0; }
    double wait_overlapped_fraction() const { return wait_ms_total > 0.0 ? wait_ms_overlapped / wait_ms_total : 0.0; }
    // Predicted compute critical path with pipelining relative to today's (1.0 = no gain).
    double predicted_ratio() const { return sequential_ms > 0.0 ? pipelined_ms / sequential_ms : 1.0; }
    double wait_free_fraction() const { return wait_ms_total > 0.0 ? wait_ms_free / wait_ms_total : 0.0; }
};

class PipelineModel {
public:
    PipelineModel(std::string name, size_t depth, bool barrier_on_other_operations)
        : name_(std::move(name)), queue_(depth), barrier_(barrier_on_other_operations) {}

    void dispatch(uint64_t submit, const std::vector<BindingRange>& bindings, const DispatchTimes& times) {
        const double wait_ms = times.gpu_ms;
        if (submit != submit_) retire_all(/*barrier=*/true);   // a submit boundary is an effect
        submit_ = submit;
        PendingOperation op = make_pending_operation(++next_id_, bindings);
        const size_t conflict = queue_.conflict_prefix(op);
        const std::vector<uint64_t> required = queue_.required_before_admit(op);
        ++stats_.dispatches;
        stats_.wait_ms_total += wait_ms;
        if (required.empty()) {
            ++stats_.free_admits;
            stats_.wait_ms_free += wait_ms;
        } else {
            ++stats_.forced_admits;
            stats_.conflict_retires += std::min(conflict, required.size());
            stats_.depth_retires += required.size() - std::min(conflict, required.size());
        }
        retire_ops(required.size());
        queue_.retire(required);
        // Timeline: the CPU sets the dispatch up, submits it, and moves on; the GPU runs it after
        // whatever it is already running. Its writeback is paid at retirement.
        cpu_clock_ += times.setup_ms;
        const double gpu_start = std::max(cpu_clock_, gpu_clock_);
        gpu_clock_ = gpu_start + times.gpu_ms;
        timeline_.push_back({gpu_clock_, times.writeback_ms});
        stats_.sequential_ms += times.setup_ms + times.gpu_ms + times.writeback_ms;
        refresh_pipelined();
        if (!queue_.empty()) {
            ++stats_.overlapped_admits;
            stats_.wait_ms_overlapped += wait_ms;
        }
        queue_.admit(op);
        stats_.max_pending = std::max<uint64_t>(stats_.max_pending, queue_.size());
    }

    // A graphics span or DMA copy ran between dispatches. Only the pessimistic model retires.
    void other_operation() {
        ++stats_.other_operations;
        if (barrier_) retire_all(/*barrier=*/true);
    }

    const PipelineModelStats& stats() const { return stats_; }
    const std::string& name() const { return name_; }

    std::string report_line() const {
        char line[640];
        std::snprintf(line, sizeof line,
                      "[pipeline-observe] model=%s dispatches=%llu free=%llu (%.1f%%) overlapped=%.1f%% wait-free=%.1f%% wait-overlapped=%.1f%% predicted-time=%.0f%% of sequential (%.0f -> %.0f ms) "
                      "retires: conflict=%llu depth=%llu barrier=%llu max-pending=%llu other-ops-seen=%llu",
                      name_.c_str(), (unsigned long long)stats_.dispatches,
                      (unsigned long long)stats_.free_admits, 100.0 * stats_.free_fraction(),
                      100.0 * stats_.overlapped_fraction(), 100.0 * stats_.wait_free_fraction(),
                      100.0 * stats_.wait_overlapped_fraction(), 100.0 * stats_.predicted_ratio(),
                      stats_.sequential_ms, stats_.pipelined_ms, (unsigned long long)stats_.conflict_retires,
                      (unsigned long long)stats_.depth_retires, (unsigned long long)stats_.barrier_retires,
                      (unsigned long long)stats_.max_pending, (unsigned long long)stats_.other_operations);
        return line;
    }

private:
    void retire_all(bool barrier) {
        const std::vector<uint64_t> ids = queue_.required_for_effect();
        if (barrier) stats_.barrier_retires += ids.size();
        retire_ops(ids.size());
        queue_.retire(ids);
        refresh_pipelined();
    }

    // The oldest `n` operations retire: the CPU waits for each one's GPU completion, then writes back.
    void retire_ops(size_t n) {
        for (size_t i = 0; i < n && !timeline_.empty(); ++i) {
            cpu_clock_ = std::max(cpu_clock_, timeline_.front().gpu_done) + timeline_.front().writeback_ms;
            timeline_.pop_front();
        }
    }

    // The predicted total if everything still pending were retired right now.
    void refresh_pipelined() {
        double t = cpu_clock_;
        for (const PendingTiming& p : timeline_) t = std::max(t, p.gpu_done) + p.writeback_ms;
        stats_.pipelined_ms = t;
    }

    struct PendingTiming {
        double gpu_done = 0.0;
        double writeback_ms = 0.0;
    };
    std::deque<PendingTiming> timeline_;
    double cpu_clock_ = 0.0;
    double gpu_clock_ = 0.0;

    std::string name_;
    RetirementQueue queue_;
    bool barrier_;
    PipelineModelStats stats_;
    uint64_t next_id_ = 0;
    uint64_t submit_ = UINT64_MAX;
};

enum class ObservedOperation : uint8_t { GraphicsSpan, Dma };

// The four models behind one lock. The live backend runs on one guest thread at a time, but the
// executor and the report path may not, and this is a diagnostic: correctness over speed.
class PipelineObserver {
public:
    PipelineObserver()
        : models_{PipelineModel("ranges-only/depth4", 4, false), PipelineModel("ranges-only/depth64", 64, false),
                  PipelineModel("barriers/depth4", 4, true), PipelineModel("barriers/depth64", 64, true)} {}

    void dispatch(uint64_t submit, const std::vector<BindingRange>& bindings, const DispatchTimes& times) {
        std::lock_guard lock(mutex_);
        for (PipelineModel& m : models_) m.dispatch(submit, bindings, times);
    }
    void other_operation(ObservedOperation) {
        std::lock_guard lock(mutex_);
        for (PipelineModel& m : models_) m.other_operation();
    }
    std::vector<std::string> report() const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> lines;
        for (const PipelineModel& m : models_) lines.push_back(m.report_line());
        return lines;
    }
    PipelineModelStats stats(size_t model) const {
        std::lock_guard lock(mutex_);
        return models_[model].stats();
    }
    uint64_t dispatches() const {
        std::lock_guard lock(mutex_);
        return models_[0].stats().dispatches;
    }

private:
    mutable std::mutex mutex_;
    std::array<PipelineModel, 4> models_;
};

}  // namespace prosper::frontend
